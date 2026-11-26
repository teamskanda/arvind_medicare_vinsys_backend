"""Deployment schemas: publication, install snippets and the public contract.

Two audiences, deliberately separated:

  * `DeploymentInfoOut` is for the authenticated customer in their dashboard.
    It may contain anything they own.
  * `PublicBotConfigOut` is served to anonymous visitors with no credentials.
    It is an **explicit allowlist** of visual settings — if a field is not
    listed in that class, a visitor cannot see it. The persona, the custom
    instructions, the fallback text, the knowledge base, the owner id and the
    model identifier are all absent on purpose.

That split is the Phase 3 brief's rule in section 15: "Only expose an
explicitly allowlisted subset of visual configuration to public visitors."
Enforcing it with a separate response model rather than a filter means a
future field added to the chatbot cannot leak by being forgotten.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import Field, field_validator

from models.base import ApiSchema, ChatbotStatus
from models.customization import (
    CustomizationState,
    LauncherSettings,
    ResolvedAppearance,
)

# A hostname, optionally with a leading wildcard label and an optional port.
# Deliberately strict: this value is compared against a request Origin, so a
# sloppy pattern here becomes an authorisation bug.
_DOMAIN_RE = re.compile(
    r"^(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?::\d{1,5})?$"
)

MAX_ALLOWED_DOMAINS = 20


# ---------------------------------------------------------------------------
# Launcher / widget settings
# ---------------------------------------------------------------------------


# `LauncherSettings` lives in models/customization.py, with the other
# settings schemas: Phase 4's studio is what writes it, and keeping the
# resolved shape next to the input shape that produces it is what stops the
# two drifting. It is imported here because the deployment contract carries
# it to the dashboard and to visitors.


# ---------------------------------------------------------------------------
# Publish / unpublish
# ---------------------------------------------------------------------------


class PublishRequest(ApiSchema):
    """Publish, optionally restricting which websites may embed the chatbot."""

    allowed_domains: list[str] | None = Field(default=None, max_length=MAX_ALLOWED_DOMAINS)

    @field_validator("allowed_domains")
    @classmethod
    def _normalise_domains(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                raise ValueError("Each allowed website must be a domain name.")
            domain = raw.strip().lower()
            if not domain:
                continue
            # Accept a pasted URL and keep only the host[:port].
            domain = re.sub(r"^[a-z][a-z0-9+.-]*://", "", domain)
            domain = domain.split("/", 1)[0].strip()
            if not domain:
                continue
            if len(domain) > 255 or not _DOMAIN_RE.match(domain):
                raise ValueError(f"'{raw.strip()[:60]}' is not a valid website address.")
            if domain not in cleaned:
                cleaned.append(domain)
        return cleaned


class DeploymentSnippet(ApiSchema):
    """One installation option, ready to copy."""

    mode: str  # "widget" | "iframe" | "link"
    label: str
    description: str
    url: str
    code: str | None = None
    language: str | None = None  # syntax hint for the dashboard, e.g. "html"
    recommended: bool = False


class PlatformInstruction(ApiSchema):
    """How to install on one website platform."""

    platform: str
    steps: list[str]
    supports_javascript: bool = True
    note: str | None = None


class DeploymentInfoOut(ApiSchema):
    """Everything the dashboard's Deploy tab renders."""

    chatbot_id: uuid.UUID
    public_id: str
    chatbot_name: str
    website_url: str
    status: ChatbotStatus
    is_published: bool
    published_at: datetime | None = None
    unpublished_at: datetime | None = None

    # Why publishing is or is not possible right now, so the dashboard can
    # explain it before the customer clicks and gets an error.
    can_publish: bool = False
    publish_blocked_reason: str | None = None
    included_pages: int = 0
    ai_ready: bool = False
    # The draft differs from the snapshot visitors are being served, so the
    # Deploy tab can offer "Publish changes" instead of implying the live
    # chatbot already reflects the studio.
    has_unpublished_changes: bool = False

    embed_url: str
    panel_url: str
    standalone_url: str
    script_url: str
    allowed_domains: list[str] = Field(default_factory=list)
    launcher: LauncherSettings = Field(default_factory=LauncherSettings)
    snippets: list[DeploymentSnippet] = Field(default_factory=list)
    platforms: list[PlatformInstruction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Public (unauthenticated) contract
# ---------------------------------------------------------------------------


class PublicBotConfigOut(ApiSchema):
    """What an anonymous visitor's browser is allowed to know.

    Allowlist, not a filter. Compare with `ChatConfigOut` in models/message.py,
    which is the authenticated playground's view and includes the persona, the
    fallback text, the model label and the token budget — none of which appear
    here.
    """

    public_id: str
    chatbot_name: str
    # The business the chatbot answers for, shown under its name. Public by
    # definition: it is the name on the website the visitor is already on.
    business_name: str | None = None
    greeting: str
    # A short line above the first message, e.g. "Ask about our opening hours".
    welcome_message: str | None = None
    # The business's own website, which the visitor is already looking at.
    website_host: str | None = None
    avatar_url: str | None = None
    logo_url: str | None = None
    cite_sources: bool = True
    suggested_questions: list[str] = Field(default_factory=list)
    # Theme, palettes, font and corner radius. Visual only: the persona, the
    # instructions, the tone preset, the grounding strictness and the fallback
    # wording are all content settings and all stay server-side.
    appearance: ResolvedAppearance
    launcher: LauncherSettings = Field(default_factory=LauncherSettings)
    show_branding: bool = True
    max_message_length: int = 1_000


class ChatbotPreviewOut(ApiSchema):
    """An unsaved configuration, resolved for the customisation studio.

    Two views of the same thing, both produced by the code that serves the
    real chatbot:

      * `state` is what the studio's controls display — including the content
        settings a visitor never sees, because the customer is editing them.
      * `public_config` is byte-for-byte the contract a visitor's browser
        would receive, built by `helpers/deployment.public_config`. The studio
        hands it straight to the same component the embed page uses, so the
        preview cannot drift from the deployed chatbot by way of a mapping
        written twice.
    """

    state: CustomizationState
    public_config: PublicBotConfigOut
    widget_script: str = ""
    widget_config: dict = Field(default_factory=dict)


class VisitorSessionOut(ApiSchema):
    """A freshly issued anonymous session.

    `token` is opaque to the browser: it is an HMAC the backend verifies, not
    something the page should parse. It is returned in the body rather than a
    cookie because the chatbot runs in a third-party iframe, where cookies are
    blocked by default in every modern browser.
    """

    token: str
    expires_at: datetime


class PublicChatRequest(ApiSchema):
    """A visitor question. Shorter cap than the playground's 2,000 characters.

    The visitor-facing surface is the open internet, so the ceiling is both an
    abuse control and a cost control. `PUBLIC_CHAT_MAX_MESSAGE_LENGTH` is the
    configured value; this is the hard schema limit.
    """

    message: str = Field(min_length=1, max_length=1_000)
    conversation_id: uuid.UUID | None = None

    @field_validator("message")
    @classmethod
    def _trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Please type a question.")
        return trimmed
