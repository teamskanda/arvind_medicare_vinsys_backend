"""Conversation and message persistence, history loading, and request limits.

Every function that reads a customer-owned row filters on `owner_id`, exactly
as `helpers/chatbots.py` does. There is deliberately no "get conversation by
id" without an owner argument, so a router cannot accidentally read another
customer's thread.

Cross-chatbot leakage is prevented by the same rule: a conversation lookup
requires BOTH `chatbot_id` and `owner_id`, so a valid conversation id from one
chatbot cannot be replayed against another.

Phase 3 note — two kinds of caller, one set of rules:

  * The **dashboard playground** is authenticated. `owner_id` comes from a
    verified Supabase token, and the `*_for_user` wrappers check chatbot
    ownership before anything else.
  * A **public visitor** has no account. The chatbot is resolved from its
    `public_id` first, which establishes `owner_id` from the chatbot row, and
    the visitor's identity is a signed session handle. So the owner filter
    still applies on every query — the visitor simply never supplies it.

Visitor-to-visitor isolation is a third predicate: `get_visitor_conversation`
requires the conversation to match the chatbot, the owner AND the session
handle, so one visitor's conversation id is useless to another.

The functions below take `owner_id` rather than an `AuthenticatedUser` for
exactly that reason: the public path has an owner but no user.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import errors
from helpers.chatbots import get_chatbot
from models.base import PageStatus
from models.conversation import Conversation, ConversationSource
from models.message import Message, MessageRole
from models.user import AuthenticatedUser
from models.website_page import WebsitePage

logger = logging.getLogger(__name__)

# How many stored turns we read before the token budget trims them further.
# Reading more than the budget can hold would be wasted I/O.
HISTORY_FETCH_MULTIPLIER = 2


def generate_session_id() -> str:
    return secrets.token_urlsafe(16)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


async def create_conversation(
    db: AsyncSession,
    owner_id: uuid.UUID,
    chatbot_id: uuid.UUID,
    *,
    session_id: str | None = None,
    source: ConversationSource = ConversationSource.PLAYGROUND,
) -> Conversation:
    """Start a thread, or return the existing one for this session handle.

    `conversations_chatbot_session_key` makes (chatbot_id, session_id) unique,
    so a double-click or a page reload resumes the same thread instead of
    forking it. The IntegrityError path handles the race between two
    simultaneous requests.

    This does **not** check ownership: the caller must already have
    established that `owner_id` owns `chatbot_id`, either by fetching the
    chatbot with `get_chatbot` (authenticated) or by resolving it from its
    `public_id` (public). `create_conversation_for_user` is the checked
    wrapper for the authenticated case.
    """
    handle = (session_id or generate_session_id())[:120]

    existing = await _find_by_session(db, owner_id, chatbot_id, handle)
    if existing is not None:
        return existing

    conversation = Conversation(
        chatbot_id=chatbot_id,
        owner_id=owner_id,
        session_id=handle,
        source=source.value,
    )
    db.add(conversation)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        conflicting = await _find_by_session(db, owner_id, chatbot_id, handle)
        if conflicting is None:
            raise errors.internal("Could not start the conversation. Please try again.")
        return conflicting

    await db.refresh(conversation)
    return conversation


async def create_conversation_for_user(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    *,
    session_id: str | None = None,
    source: ConversationSource = ConversationSource.PLAYGROUND,
) -> Conversation:
    """Ownership-checked `create_conversation` for authenticated callers."""
    await get_chatbot(db, user, chatbot_id)
    return await create_conversation(
        db, user.id, chatbot_id, session_id=session_id, source=source
    )


async def _find_by_session(
    db: AsyncSession, owner_id: uuid.UUID, chatbot_id: uuid.UUID, session_id: str
) -> Conversation | None:
    result = await db.execute(
        select(Conversation).where(
            Conversation.chatbot_id == chatbot_id,
            Conversation.owner_id == owner_id,
            Conversation.session_id == session_id,
        )
    )
    return result.scalar_one_or_none()


async def get_conversation(
    db: AsyncSession,
    owner_id: uuid.UUID,
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> Conversation:
    """Fetch one conversation belonging to this owner AND this chatbot.

    Both predicates matter: filtering on owner alone would let a customer
    replay a conversation id from chatbot A against chatbot B and mix two
    knowledge bases into one thread.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.chatbot_id == chatbot_id,
            Conversation.owner_id == owner_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise errors.not_found("conversation")
    return conversation


async def get_visitor_conversation(
    db: AsyncSession,
    owner_id: uuid.UUID,
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
    session_handle: str,
) -> Conversation:
    """Fetch a public visitor's own conversation. Four predicates, all required.

    The fourth — `session_id` — is what isolates one anonymous visitor from
    another. Without it, a visitor who learned another visitor's conversation
    id could read and continue their thread, because both are legitimately
    talking to the same public chatbot and both resolve to the same
    `owner_id`. The session handle comes from a token we signed, so it cannot
    be guessed or chosen.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.chatbot_id == chatbot_id,
            Conversation.owner_id == owner_id,
            Conversation.session_id == session_handle,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise errors.not_found("conversation")
    return conversation


async def list_conversations(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID, *, limit: int = 50
) -> list[tuple[Conversation, int]]:
    """Conversations for one chatbot, newest first, each with its message count."""
    await get_chatbot(db, user, chatbot_id)

    count_subquery = (
        select(Message.conversation_id, func.count().label("message_count"))
        .where(Message.owner_id == user.id)
        .group_by(Message.conversation_id)
        .subquery()
    )

    result = await db.execute(
        select(Conversation, func.coalesce(count_subquery.c.message_count, 0))
        .outerjoin(count_subquery, count_subquery.c.conversation_id == Conversation.id)
        .where(
            Conversation.chatbot_id == chatbot_id,
            Conversation.owner_id == user.id,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return [(row[0], int(row[1] or 0)) for row in result.all()]


async def set_conversation_title(
    db: AsyncSession, conversation: Conversation, question: str
) -> None:
    """Name a thread after its first question, for the conversation list."""
    if conversation.title:
        return
    title = " ".join(question.split())[:120]
    if title:
        conversation.title = title
        await db.commit()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


async def add_message(
    db: AsyncSession,
    conversation: Conversation,
    *,
    role: MessageRole,
    content: str,
    source_references: list[dict] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error_code: str | None = None,
    commit: bool = True,
) -> Message:
    message = Message(
        conversation_id=conversation.id,
        chatbot_id=conversation.chatbot_id,
        owner_id=conversation.owner_id,
        role=role.value,
        content=content,
        source_references=source_references or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_code=error_code,
    )
    db.add(message)
    if commit:
        await db.commit()
        await db.refresh(message)
    return message


async def list_messages(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> list[Message]:
    conversation = await get_conversation(db, user.id, chatbot_id, conversation_id)
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation.id,
            Message.owner_id == user.id,
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    return list(result.scalars().all())


async def load_history(
    db: AsyncSession, conversation: Conversation, *, turns: int
) -> list[tuple[str, str]]:
    """Recent (role, content) pairs for this conversation, oldest first.

    Scoped to one conversation and one owner, so history from another thread
    or another customer can never enter a prompt. Turns that recorded an
    error are skipped: replaying "the AI is unavailable" as context would
    teach the model to repeat it.
    """
    limit = max(2, turns * HISTORY_FETCH_MULTIPLIER)
    result = await db.execute(
        select(Message.role, Message.content)
        .where(
            Message.conversation_id == conversation.id,
            Message.owner_id == conversation.owner_id,
            Message.error_code.is_(None),
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    rows = list(result.all())
    rows.reverse()
    return [(str(role), str(content)) for role, content in rows]


# ---------------------------------------------------------------------------
# Knowledge base access
# ---------------------------------------------------------------------------


async def load_included_pages(
    db: AsyncSession, owner_id: uuid.UUID, chatbot_id: uuid.UUID
) -> list[WebsitePage]:
    """The pages this chatbot may answer from.

    Three filters, all required:
      * `chatbot_id` — only this chatbot's knowledge base;
      * `owner_id`   — only this customer's rows;
      * `is_included` and `status = 'scraped'` — the customer's own
        include/exclude choices, and never an empty or failed page.
    """
    result = await db.execute(
        select(WebsitePage)
        .where(
            WebsitePage.chatbot_id == chatbot_id,
            WebsitePage.owner_id == owner_id,
            WebsitePage.is_included.is_(True),
            WebsitePage.status == PageStatus.SCRAPED.value,
            WebsitePage.content != "",
        )
        .order_by(WebsitePage.priority.desc(), WebsitePage.url.asc())
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Request limits
# ---------------------------------------------------------------------------


async def enforce_rate_limits(
    db: AsyncSession, owner_id: uuid.UUID, chatbot_id: uuid.UUID
) -> None:
    """Per-user-per-minute and per-user-per-day caps on model requests.

    Counted from stored user messages rather than an in-memory counter, so the
    limit holds across worker processes and restarts. One extra indexed count
    query per chatbot turn is a fair price for a limit that cannot be bypassed
    by reconnecting.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(
            func.count()
            .filter(Message.created_at >= now - timedelta(minutes=1))
            .label("last_minute"),
            func.count()
            .filter(Message.created_at >= now - timedelta(days=1))
            .label("last_day"),
            func.count()
            .filter(
                Message.created_at >= now - timedelta(minutes=1),
                Message.chatbot_id == chatbot_id,
            )
            .label("chatbot_last_minute"),
        )
        .select_from(Message)
        .where(
            Message.owner_id == owner_id,
            Message.role == MessageRole.USER.value,
            Message.created_at >= now - timedelta(days=1),
        )
    )
    row = result.one()

    if int(row.last_minute or 0) >= settings.chat_rate_limit_per_minute:
        raise errors.chat_rate_limited(
            "You are sending messages too quickly. Please wait a moment and try again."
        )
    if int(row.chatbot_last_minute or 0) >= settings.chat_rate_limit_per_minute:
        raise errors.chat_rate_limited(
            "This chatbot is handling too many messages right now. "
            "Please wait a moment and try again."
        )
    if int(row.last_day or 0) >= settings.chat_rate_limit_per_day:
        raise errors.chat_rate_limited(
            "You have reached today's message limit for testing. "
            "Please try again tomorrow."
        )


# ---------------------------------------------------------------------------
# Public visitor limits (Phase 3)
# ---------------------------------------------------------------------------
#
# Three scopes, because they protect against different things:
#
#   per session  — one visitor hammering the widget;
#   per chatbot  — the customer's total public spend, including a distributed
#                  flood from many sessions;
#   per IP       — a single source cycling through fresh session tokens.
#
# The first two are counted in SQL from stored messages, so they survive a
# restart and hold across worker processes. The third is an in-process
# sliding window: an IP is not stored anywhere, which keeps a visitor's
# address out of the database entirely. The documented consequence is that
# the per-IP limit is per worker — with N workers the effective ceiling is N
# times the configured value. The per-chatbot SQL limit is the backstop that
# actually bounds cost, which is why the weaker control is the one that holds
# no personal data.

_ip_hits: dict[str, list[float]] = {}
_IP_WINDOW_SECONDS = 60.0
_IP_TABLE_MAX_KEYS = 20_000


def enforce_ip_rate_limit(client_ip: str | None) -> None:
    """In-process per-IP sliding window. See the note above for its limits."""
    settings = get_settings()
    if not client_ip:
        # Behind a proxy that sends no address we cannot attribute the
        # request. The session and chatbot limits still apply.
        return

    # Hashed and truncated: enough to count, not enough to identify, and it is
    # never written to disk or logged.
    key = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:24]
    now = time.monotonic()
    cutoff = now - _IP_WINDOW_SECONDS

    hits = [stamp for stamp in _ip_hits.get(key, ()) if stamp > cutoff]
    if len(hits) >= settings.public_chat_ip_limit_per_minute:
        _ip_hits[key] = hits
        raise errors.chat_rate_limited(
            "Too many messages from this connection. Please wait a minute and try again."
        )

    hits.append(now)
    _ip_hits[key] = hits

    # Bounded memory: drop everything stale once the table grows large.
    if len(_ip_hits) > _IP_TABLE_MAX_KEYS:
        for stale_key in [k for k, v in _ip_hits.items() if not v or max(v) <= cutoff]:
            _ip_hits.pop(stale_key, None)


async def enforce_public_rate_limits(
    db: AsyncSession,
    chatbot_id: uuid.UUID,
    session_handle: str,
    *,
    client_ip: str | None = None,
) -> None:
    """Per-session, per-chatbot and per-IP caps for public visitor traffic.

    Checked before any work that costs money. The per-IP check runs first
    because it needs no query.
    """
    settings = get_settings()
    enforce_ip_rate_limit(client_ip)

    now = datetime.now(timezone.utc)
    minute_ago = now - timedelta(minutes=1)
    day_ago = now - timedelta(days=1)

    # One query, four aggregates. Joined to conversations only to read the
    # session handle; the chatbot predicate is on `messages` itself, which is
    # what `messages_chatbot_created_idx` covers.
    result = await db.execute(
        select(
            func.count()
            .filter(
                Message.created_at >= minute_ago,
                Conversation.session_id == session_handle,
            )
            .label("session_minute"),
            func.count()
            .filter(Conversation.session_id == session_handle)
            .label("session_day"),
            func.count().filter(Message.created_at >= minute_ago).label("bot_minute"),
            func.count().label("bot_day"),
        )
        .select_from(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.chatbot_id == chatbot_id,
            Message.role == MessageRole.USER.value,
            Message.created_at >= day_ago,
            Conversation.source == ConversationSource.EMBED.value,
        )
    )
    row = result.one()

    if int(row.session_minute or 0) >= settings.public_chat_rate_limit_per_minute:
        raise errors.chat_rate_limited(
            "You are sending messages too quickly. Please wait a moment and try again."
        )
    if int(row.session_day or 0) >= settings.public_chat_rate_limit_per_day:
        raise errors.chat_rate_limited(
            "You have reached today's message limit for this chat. "
            "Please try again tomorrow."
        )
    if int(row.bot_minute or 0) >= settings.public_chat_chatbot_limit_per_minute:
        raise errors.chat_rate_limited(
            "This assistant is handling too many messages right now. "
            "Please wait a moment and try again."
        )
    if int(row.bot_day or 0) >= settings.public_chat_chatbot_limit_per_day:
        raise errors.chat_rate_limited(
            "This assistant has reached its daily message limit. "
            "Please try again tomorrow."
        )
