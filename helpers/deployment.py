"""Publication lifecycle, published-configuration resolution and public lookup.

Three responsibilities, in order of importance:

1. **Publish / unpublish.** The lifecycle is one column, `chatbots.status`.
   Publishing snapshots the draft configuration into `published_config`;
   unpublishing flips the status and touches nothing else, so the public_id,
   the configuration and the knowledge base all survive and republishing
   restores the identical URL.

2. **Resolving what the public serves.** A published chatbot answers from
   `published_config`, never from `draft_config`. That is what makes the
   draft/published separation real: a customer can edit settings all day
   without changing what visitors see until they publish again. The
   corollary, which the brief calls out explicitly, is that an unpublished
   chatbot cannot keep serving from a stale snapshot — the status is checked
   on every single public request, not cached.

3. **Deciding whether a public request is allowed.** Four gates, each of
   which can refuse on its own: the handle must exist, the status must be
   `published`, the embedding origin must be authorised when the customer
   configured a list, and the request limits must not be exhausted.

On Origin and Referer: both are supplementary signals. A non-browser client
can send anything, so the allowed-domain check raises the cost of casual
misuse but is not authentication. The brief says exactly this, and the real
controls are the signed session token, the rate limits and the publication
status.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import customization, errors
from helpers import chatbots as chatbot_helpers
from helpers.customization import (  # re-exported: see the note below
    resolve_launcher,
    suggested_questions,
)
from helpers.prompt_builder import ChatbotAiConfig
from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.deployment import PublicBotConfigOut, PublishRequest
from models.user import AuthenticatedUser

logger = logging.getLogger(__name__)

# `resolve_launcher` and `suggested_questions` moved to helpers/customization
# in Phase 4, where the rest of the settings resolution lives. They are
# re-exported here because this module is the deployment-facing façade and
# every existing caller reads them as `deployment.resolve_launcher(...)`;
# re-exporting keeps one implementation while leaving those call sites
# meaningful — resolving what a *deployment* serves is this module's subject.
__all__ = [
    "PUBLISHED_CONFIG_VERSION",
    "PublishReadiness",
    "ai_config_for",
    "check_embedding_origin",
    "get_published_chatbot",
    "public_config",
    "publish",
    "publish_readiness",
    "resolve_launcher",
    "suggested_questions",
    "unpublish",
]

# Shape version stored alongside the snapshot, so a later change to the
# configuration format can migrate old published chatbots knowingly rather
# than by guessing. Bumped to 2 in Phase 4: the snapshot gained an
# `appearance` subtree, and the launcher gained shape, label and animations.
PUBLISHED_CONFIG_VERSION = 2


# ---------------------------------------------------------------------------
# AI configuration: draft vs published
# ---------------------------------------------------------------------------


def ai_config_for(chatbot: Chatbot, *, published: bool) -> ChatbotAiConfig:
    """Resolve the chatbot's AI settings from the right snapshot.

    `published=True` is the public path and reads `published_config`, so a
    visitor is never served an unpublished draft. `published=False` is the
    dashboard playground, which deliberately previews the draft.
    """
    source = chatbot.published_config if published else chatbot.draft_config
    return ChatbotAiConfig.from_chatbot(
        name=chatbot.name, website_url=chatbot.website_url, config=source
    )


# ---------------------------------------------------------------------------
# Publish / unpublish
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublishReadiness:
    """Whether publishing is possible, and if not, which gate refused.

    `blocker` is a code so `publish()` can raise the matching error, and
    `reason` is the sentence the dashboard shows. Returned rather than raised
    because the Deploy tab needs to disable the button and explain *before*
    the customer clicks, not after.
    """

    can_publish: bool
    blocker: str | None = None
    reason: str | None = None
    included_pages: int = 0


async def publish_readiness(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot
) -> PublishReadiness:
    """Can this chatbot be published, and if not, why not?"""
    settings = get_settings()
    stats = await chatbot_helpers.get_stats(db, user, chatbot.id)
    pages = stats.included_pages

    if settings.publish_requires_approval:
        if user.email.strip().lower() not in settings.publish_allowed_email_set:
            return PublishReadiness(
                False,
                "not_approved",
                "Public publishing is limited to approved accounts during the "
                "launch period. Contact the SKANDA AI team to enable it.",
                pages,
            )

    if pages == 0:
        return PublishReadiness(
            False,
            "empty_knowledge_base",
            "This chatbot has no website content yet. Scrape the website and "
            "include at least one page before publishing.",
            pages,
        )

    if not settings.gemma_configured:
        return PublishReadiness(
            False,
            "ai_not_configured",
            "The AI engine is not configured on the server, so a published "
            "chatbot could not answer questions yet.",
            pages,
        )

    return PublishReadiness(True, included_pages=pages)


async def publish(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    payload: PublishRequest | None = None,
) -> Chatbot:
    """Make a chatbot publicly available at its permanent embed URL.

    Steps, in the brief's order (section 11):
      1. verify the knowledge base is usable;
      2. validate the configuration;
      3. copy the draft settings into the published configuration;
      4. activate the public chatbot;
      5. deployment links are then generated from the public_id, which has
         existed since Phase 1 and never changes.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)

    readiness = await publish_readiness(db, user, chatbot)
    if not readiness.can_publish:
        if readiness.blocker == "not_approved":
            raise errors.publish_not_approved()
        raise errors.not_publishable(
            readiness.reason or "This chatbot cannot be published yet."
        )

    # Validate by resolving: `from_chatbot` type-checks and caps every value,
    # so anything that would break prompt construction is normalised here
    # rather than discovered on a visitor's first question.
    ai_config = ai_config_for(chatbot, published=False)
    launcher = resolve_launcher(chatbot, published=False)

    now = datetime.now(timezone.utc)
    draft = chatbot.draft_config if isinstance(chatbot.draft_config, dict) else {}
    draft_ai = draft.get("ai") if isinstance(draft.get("ai"), dict) else draft
    draft_appearance = draft.get("appearance")

    snapshot: dict[str, Any] = {
        "version": PUBLISHED_CONFIG_VERSION,
        "published_at": now.isoformat(),
        # The content and appearance subtrees are snapshotted as the customer
        # has them — the raw document, not resolved values — so a later
        # improvement to how a palette is derived reaches already-published
        # chatbots, and so a key this version does not know about is carried
        # through rather than silently dropped.
        "ai": dict(draft_ai) if isinstance(draft_ai, dict) else {},
        "appearance": dict(draft_appearance) if isinstance(draft_appearance, dict) else {},
        # The launcher is the exception: it is snapshotted *resolved*, because
        # these numbers are interpolated into a script running on the
        # customer's website and must not move under them.
        "deployment": launcher.model_dump(),
    }

    # One field is deliberately snapshotted *unresolved*: an uploaded icon is
    # stored as `asset:<uuid>` rather than as the absolute URL it resolves to.
    # Baking the URL in would tie every published chatbot to the API hostname
    # that was configured on the day it was published, and in development —
    # where that host is plain http — the resolver would refuse its own URL on
    # the way back out and the icon would silently vanish.
    draft_launcher = (
        chatbot.deployment_config if isinstance(chatbot.deployment_config, dict) else {}
    )
    if isinstance(draft_launcher.get("icon_image"), str):
        snapshot["deployment"]["icon_image"] = draft_launcher["icon_image"]
    # Make sure the resolved essentials are present even when the draft is
    # empty, so the public config endpoint never has to fall back to defaults.
    snapshot["ai"].setdefault("chatbot_name", ai_config.chatbot_name)
    snapshot["ai"].setdefault("greeting", ai_config.greeting)
    snapshot["ai"].setdefault("fallback_message", ai_config.fallback_message)

    if payload is not None and payload.allowed_domains is not None:
        chatbot.allowed_domains = payload.allowed_domains

    chatbot.published_config = snapshot
    chatbot.status = ChatbotStatus.PUBLISHED.value
    chatbot.published_at = now
    chatbot.unpublished_at = None

    await db.commit()
    await db.refresh(chatbot)

    logger.info(
        "Published chatbot %s (public_id=%s) with %d allowed domain(s).",
        chatbot.id,
        chatbot.public_id,
        len(chatbot.allowed_domains or []),
    )
    return chatbot


async def unpublish(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> Chatbot:
    """Stop serving public conversations, preserving everything else.

    What is deliberately *not* touched: `public_id`, `published_config`,
    `allowed_domains`, `deployment_config`, the pages, and the stored
    conversations. Republishing therefore restores the same URL and the same
    behaviour, and an installed snippet on the customer's website starts
    working again without being changed.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)

    if chatbot.status != ChatbotStatus.PUBLISHED.value:
        # Idempotent: unpublishing something already offline is not an error.
        return chatbot

    chatbot.status = ChatbotStatus.UNPUBLISHED.value
    chatbot.unpublished_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(chatbot)

    logger.info("Unpublished chatbot %s (public_id=%s).", chatbot.id, chatbot.public_id)
    return chatbot


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------


async def get_published_chatbot(db: AsyncSession, public_id: str) -> Chatbot:
    """Find a chatbot that is currently serving the public, or raise 404.

    The status predicate is in the query, so there is no window in which a
    caller holds a chatbot row and then decides whether it is published.
    Unpublishing takes effect on the very next request — no cache, no stale
    configuration.
    """
    if not public_id or len(public_id) > 64 or not public_id.isalnum():
        # Cheap shape check before touching the database: the handle alphabet
        # is lowercase alphanumeric (helpers/chatbots.generate_public_id).
        raise errors.chatbot_not_published()

    result = await db.execute(
        select(Chatbot).where(
            Chatbot.public_id == public_id,
            Chatbot.status == ChatbotStatus.PUBLISHED.value,
        )
    )
    chatbot = result.scalar_one_or_none()
    if chatbot is None:
        raise errors.chatbot_not_published()
    return chatbot


def check_embedding_origin(
    chatbot: Chatbot,
    *,
    embed_origin: str | None = None,
    origin: str | None = None,
    referer: str | None = None,
) -> None:
    """Enforce the customer's allowed-domain list, when they set one.

    No list configured -> no restriction. That is the default, and it is the
    right default: a customer who never asked for a restriction should not
    have their widget mysteriously refuse to load.

    When a list *is* configured, the hard part is knowing which website the
    request actually came from. The chatbot runs inside an iframe we serve,
    so its own `Origin` on an API call is **our** domain, not the customer's
    page — the browser's Origin header is useless for this question. What
    identifies the host page is:

      * `embed_origin` — the parent page URL, which the embed page reads from
        `document.referrer` and forwards in `X-Embed-Origin`;
      * `referer` / `origin` — used for a direct, non-iframe request to the
        standalone chatbot link.

    All three are page-supplied, so a determined caller can forge them. This
    check raises the cost of casually dropping someone else's snippet on an
    unrelated site; it is not authentication, and the brief says as much. The
    controls that actually hold are the publication status, the signed visitor
    session and the request limits.

    Browser-enforced `frame-ancestors` would be the unforgeable version of
    this check. It is a known gap — see the Phase 3 limitations in README.md.
    """
    allowed = chatbot.allowed_domains if isinstance(chatbot.allowed_domains, list) else []
    patterns = [d for d in allowed if isinstance(d, str) and d.strip()]
    if not patterns:
        return

    host = _host_of(embed_origin) or _host_of(origin) or _host_of(referer)
    if host is None:
        # A list is configured and we cannot tell where the request came from.
        # Refusing is the safe direction.
        raise errors.embedding_not_allowed()

    # The chatbot's own page is always permitted: that is the standalone link
    # and the dashboard preview, neither of which is a third-party embed.
    own_host = _host_of(get_settings().public_widget_base_url)
    if own_host is not None and host == own_host:
        return

    if not any(_host_matches(host, pattern) for pattern in patterns):
        logger.info(
            "Rejected a public request for chatbot %s from unauthorised host %r.",
            chatbot.id,
            host[:120],
        )
        raise errors.embedding_not_allowed()


def frame_ancestors(chatbot: Chatbot) -> str:
    """CSP source expressions, never raw header text from stored JSON."""
    from pydantic import ValidationError
    try:
        domains = PublishRequest(allowed_domains=chatbot.allowed_domains or []).allowed_domains or []
    except ValidationError:
        return "'none'"
    if not domains:
        return '*'
    sources = ["'self'"]
    for domain in domains:
        sources.append('https://' + domain)
        if get_settings().environment != 'production':
            sources.append('http://' + domain)
    return ' '.join(sources)


def _host_of(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    parsed = urlsplit(value.strip())
    host = (parsed.netloc or parsed.path).strip().lower()
    if not host or "/" in host:
        return None
    return host


def _host_matches(host: str, pattern: str) -> bool:
    """Exact host match, or a single leading-wildcard label.

    `*.example.com` matches `shop.example.com` and `example.com`, but not
    `notexample.com` and not `example.com.attacker.net`. The port, when the
    pattern specifies one, must match too.
    """
    pattern = pattern.strip().lower()
    if pattern.startswith("*."):
        suffix = pattern[2:]
        host_without_port = host.split(":", 1)[0]
        return host_without_port == suffix or host_without_port.endswith(f".{suffix}")
    if ":" in pattern:
        return host == pattern
    return host.split(":", 1)[0] == pattern


def public_config(chatbot: Chatbot, *, published: bool = True) -> PublicBotConfigOut:
    """The allowlisted visual configuration for an anonymous visitor.

    Everything here is already public or harmless: the business's own name,
    its greeting, its brand colour, its hostname. The persona, the custom
    instructions, the fallback wording, the owner id, the model identifier,
    the token budget and the knowledge base are all absent — see the module
    docstring of models/deployment.py.

    `published` defaults to True and every public endpoint leaves it that way,
    so a visitor can only ever be served the published snapshot. The one
    caller that passes False is the customisation studio's preview, which is
    authenticated, owner-scoped and showing the customer their own draft —
    and which needs this exact function so that what they preview is the
    contract a visitor would get rather than a second mapping of it.
    """
    settings = get_settings()
    content = customization.resolve_content(chatbot, published=published)
    appearance = customization.resolve_appearance(chatbot, published=published)
    launcher = resolve_launcher(chatbot, published=published)

    return PublicBotConfigOut(
        public_id=chatbot.public_id,
        chatbot_name=content.chatbot_name,
        business_name=content.business_name,
        greeting=content.greeting,
        welcome_message=content.welcome_message,
        website_host=_host_of(chatbot.website_url),
        avatar_url=appearance.avatar_url,
        logo_url=appearance.logo_url,
        cite_sources=content.cite_sources,
        suggested_questions=content.suggested_questions,
        appearance=appearance,
        launcher=launcher,
        show_branding=launcher.show_branding,
        max_message_length=settings.public_chat_max_message_length,
    )
