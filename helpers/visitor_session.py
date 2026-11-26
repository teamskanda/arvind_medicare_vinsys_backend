"""Anonymous visitor sessions for the public chatbot.

A website visitor has no SKANDA AI account, but a conversation still needs a
stable identity so follow-up questions work and so one visitor cannot read
another's thread. This module issues that identity.

Design: **stateless, HMAC-signed tokens.** There is no visitor_sessions table.

    token = v1.<base64url(payload json)>.<base64url(hmac_sha256)>
    payload = {"sid": <random 128-bit handle>, "bot": <public_id>, "exp": <unix>}

Why signed rather than stored:

  * A row per visitor on a public widget is unbounded growth with no value —
    the only fact we need is "this handle was issued by us, for this bot,
    recently", and a signature proves all three.
  * It makes the public chat path one database round trip shorter.
  * There is nothing to expire-sweep.

What the signature buys us, concretely: `conversations.session_id` is the
lookup key for a visitor's thread. If a visitor could choose it freely they
could type someone else's handle and read their conversation. Because the
handle only ever arrives inside a token we signed, they cannot.

The token is **bound to one chatbot** (`bot`), so a token issued by chatbot A
cannot be replayed against chatbot B even though both are public.

Security limits, stated plainly:
  * This authenticates a *session*, not a person. Anyone holding the token is
    that session. That is inherent to anonymous chat and is why the token is
    short-lived and the rate limits are per-session.
  * Tokens cannot be revoked individually before expiry. Unpublishing the
    chatbot stops every session immediately, which is the control that
    matters.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass

from config import get_settings
from helpers import errors

logger = logging.getLogger(__name__)

_TOKEN_VERSION = "v1"
_SESSION_ID_BYTES = 16  # 128 bits of entropy
_MAX_TOKEN_CHARACTERS = 512  # a well-formed token is ~120; reject anything odd

# Generated once per process when VISITOR_SESSION_SECRET is unset. Development
# then works with no configuration, at the cost of sessions not surviving a
# restart — which is logged loudly rather than left to be discovered.
_ephemeral_secret: bytes | None = None


def _signing_key() -> bytes:
    global _ephemeral_secret
    settings = get_settings()
    if settings.visitor_session_secret:
        return settings.visitor_session_secret.encode("utf-8")

    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_bytes(32)
        logger.warning(
            "VISITOR_SESSION_SECRET is not set. Using a random per-process key: "
            "public chat sessions will be lost whenever the API restarts, and "
            "will not work at all across multiple workers. Set it in backend/.env."
        )
    return _ephemeral_secret


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


@dataclass(frozen=True, slots=True)
class VisitorSession:
    """A verified anonymous visitor session."""

    session_id: str
    public_id: str
    expires_at: int

    @property
    def conversation_handle(self) -> str:
        """What goes into `conversations.session_id`.

        Prefixed so public visitor threads are visually distinguishable from
        playground threads in the database, and so the two namespaces can
        never collide on the (chatbot_id, session_id) unique index.
        """
        return f"visitor:{self.session_id}"


def issue(public_id: str) -> tuple[str, VisitorSession]:
    """Mint a new session token for one chatbot. Returns (token, session)."""
    settings = get_settings()
    session = VisitorSession(
        session_id=secrets.token_urlsafe(_SESSION_ID_BYTES),
        public_id=public_id,
        expires_at=int(time.time()) + settings.visitor_session_ttl_hours * 3600,
    )
    payload = json.dumps(
        {"sid": session.session_id, "bot": session.public_id, "exp": session.expires_at},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    body = f"{_TOKEN_VERSION}.{_b64encode(payload)}"
    signature = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64encode(signature)}", session


def verify(token: str | None, *, public_id: str) -> VisitorSession:
    """Validate a token and confirm it was issued for `public_id`.

    Raises `AppError` (401) for anything that is not a token we signed, for
    this chatbot, that has not expired. The message never explains which check
    failed — that detail only helps someone probing the format.
    """
    if not token or len(token) > _MAX_TOKEN_CHARACTERS:
        raise _rejected()

    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _TOKEN_VERSION:
        raise _rejected()

    body = f"{parts[0]}.{parts[1]}"
    expected = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64decode(parts[2])
    except (ValueError, TypeError):
        raise _rejected()

    # Constant-time: a timing side channel here would leak the signature.
    if not hmac.compare_digest(expected, provided):
        raise _rejected()

    try:
        payload = json.loads(_b64decode(parts[1]))
    except (ValueError, TypeError):
        raise _rejected()
    if not isinstance(payload, dict):
        raise _rejected()

    session_id = payload.get("sid")
    bound_bot = payload.get("bot")
    expires_at = payload.get("exp")
    if not isinstance(session_id, str) or not isinstance(bound_bot, str):
        raise _rejected()
    if not isinstance(expires_at, int):
        raise _rejected()

    # Binding check: a signature alone would let a token from one public
    # chatbot be replayed against another.
    if not hmac.compare_digest(bound_bot, public_id):
        raise _rejected()

    if expires_at <= int(time.time()):
        raise errors.AppError(
            "visitor_session_expired",
            "This chat session has expired. Please refresh the page to continue.",
            status_code=401,
        )

    return VisitorSession(
        session_id=session_id, public_id=bound_bot, expires_at=expires_at
    )


def _rejected() -> errors.AppError:
    return errors.AppError(
        "visitor_session_invalid",
        "This chat session is not valid. Please refresh the page to continue.",
        status_code=401,
    )
