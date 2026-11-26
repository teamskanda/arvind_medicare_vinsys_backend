"""Visitor session tokens: the thing that keeps one visitor out of another's chat.

These are the highest-value tests in Phase 3. The session handle inside the
token becomes `conversations.session_id`, which is the lookup key for a
visitor's thread — so if a token can be forged, edited or replayed against a
different chatbot, visitor isolation is gone.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from config import Settings, get_settings
from helpers import errors, visitor_session

BOT = "abc234defghijkmnpqrstu"
OTHER_BOT = "zyx987wvutsrqpnmkjihgf"


@pytest.fixture(autouse=True)
def fixed_secret(monkeypatch: pytest.MonkeyPatch):
    """A stable signing key, so tests do not depend on the ephemeral one."""
    original = get_settings()
    settings = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        visitor_session_secret="test-visitor-secret-do-not-use",
    )
    monkeypatch.setattr(visitor_session, "get_settings", lambda: settings)
    return settings


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_an_issued_token_verifies():
    token, session = visitor_session.issue(BOT)
    verified = visitor_session.verify(token, public_id=BOT)
    assert verified.session_id == session.session_id
    assert verified.public_id == BOT


def test_two_sessions_for_one_bot_are_different():
    """Two visitors on the same website must not share a session handle."""
    first, _ = visitor_session.issue(BOT)
    second, _ = visitor_session.issue(BOT)
    assert first != second
    assert (
        visitor_session.verify(first, public_id=BOT).session_id
        != visitor_session.verify(second, public_id=BOT).session_id
    )


def test_the_conversation_handle_is_namespaced():
    """Visitor threads must not collide with playground threads.

    `conversations` has a unique index on (chatbot_id, session_id). A
    playground session handle is a bare token; a visitor's is prefixed, so one
    can never be mistaken for or overwrite the other.
    """
    _token, session = visitor_session.issue(BOT)
    assert session.conversation_handle.startswith("visitor:")
    assert session.session_id in session.conversation_handle


def test_the_session_id_has_real_entropy():
    """A guessable handle would let a visitor address another's conversation."""
    _token, session = visitor_session.issue(BOT)
    assert len(session.session_id) >= 20


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        None,
        "",
        "garbage",
        "v1.abc",
        "v1.abc.def.ghi",
        "v2." + base64.urlsafe_b64encode(b"{}").decode() + ".sig",
        "x" * 600,
    ],
)
def test_malformed_tokens_are_rejected(token):
    with pytest.raises(errors.AppError) as caught:
        visitor_session.verify(token, public_id=BOT)
    assert caught.value.status_code == 401


def test_a_tampered_payload_is_rejected():
    """Changing the session id must invalidate the signature."""
    token, _session = visitor_session.issue(BOT)
    version, payload, signature = token.split(".")

    decoded = json.loads(_b64decode(payload))
    decoded["sid"] = "someone-elses-session"
    forged_payload = _b64encode(json.dumps(decoded, separators=(",", ":")).encode())

    with pytest.raises(errors.AppError):
        visitor_session.verify(
            f"{version}.{forged_payload}.{signature}", public_id=BOT
        )


def test_a_tampered_signature_is_rejected():
    token, _session = visitor_session.issue(BOT)
    version, payload, _signature = token.split(".")
    with pytest.raises(errors.AppError):
        visitor_session.verify(
            f"{version}.{payload}.{_b64encode(b'not the signature')}", public_id=BOT
        )


def test_a_token_from_another_chatbot_is_rejected():
    """The binding check. Both chatbots are public and both tokens are ours,
    so the signature alone would accept this."""
    token, _session = visitor_session.issue(OTHER_BOT)
    with pytest.raises(errors.AppError) as caught:
        visitor_session.verify(token, public_id=BOT)
    assert caught.value.code == "visitor_session_invalid"


def test_an_expired_token_is_rejected(monkeypatch: pytest.MonkeyPatch):
    original = get_settings()
    expired_settings = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        visitor_session_secret="test-visitor-secret-do-not-use",
        visitor_session_ttl_hours=0,
    )
    monkeypatch.setattr(visitor_session, "get_settings", lambda: expired_settings)

    token, _session = visitor_session.issue(BOT)
    # TTL of zero means `exp` equals now; verification requires exp > now.
    with pytest.raises(errors.AppError) as caught:
        visitor_session.verify(token, public_id=BOT)
    assert caught.value.code == "visitor_session_expired"


def test_a_token_signed_with_another_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """Rotating the secret must invalidate every outstanding token."""
    token, _session = visitor_session.issue(BOT)

    original = get_settings()
    rotated = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        visitor_session_secret="a-completely-different-secret",
    )
    monkeypatch.setattr(visitor_session, "get_settings", lambda: rotated)

    with pytest.raises(errors.AppError):
        visitor_session.verify(token, public_id=BOT)


# ---------------------------------------------------------------------------
# What the token does and does not contain
# ---------------------------------------------------------------------------


def test_the_token_carries_no_customer_information():
    """A visitor's browser holds this. It must reveal nothing about the owner."""
    token, _session = visitor_session.issue(BOT)
    payload = json.loads(_b64decode(token.split(".")[1]))
    assert set(payload) == {"sid", "bot", "exp"}
    # No owner id, no email, no chatbot uuid, no internal identifier at all.
    assert payload["bot"] == BOT


def test_the_expiry_is_in_the_future_and_bounded():
    _token, session = visitor_session.issue(BOT)
    now = int(time.time())
    settings = visitor_session.get_settings()
    assert now < session.expires_at <= now + settings.visitor_session_ttl_hours * 3600 + 5


# ---------------------------------------------------------------------------
# No configured secret
# ---------------------------------------------------------------------------


def test_an_unset_secret_still_works_within_one_process(
    monkeypatch: pytest.MonkeyPatch,
):
    """Development must work with no configuration.

    The documented cost is that the key is per-process, so sessions do not
    survive a restart. Within one process, issue/verify must still round-trip.
    """
    original = get_settings()
    no_secret = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        visitor_session_secret=None,
    )
    monkeypatch.setattr(visitor_session, "get_settings", lambda: no_secret)
    monkeypatch.setattr(visitor_session, "_ephemeral_secret", None)

    token, _session = visitor_session.issue(BOT)
    assert visitor_session.verify(token, public_id=BOT).public_id == BOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
