"""Demo access remains an explicit opt-in, signed, expiring shared identity."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest

import main
from config import get_settings
from helpers import auth, errors
from routers import auth as routes


@pytest.fixture
def enabled(monkeypatch):
    settings = get_settings().model_copy(update={
        "demo_auth_enabled": True,
        "demo_auth_secret": "test-demo-signing-secret-at-least-32-characters",
    })
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    return settings


def test_demo_is_disabled_without_explicit_opt_in(monkeypatch):
    settings = get_settings().model_copy(update={"demo_auth_enabled": False})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    with pytest.raises(errors.AppError) as caught:
        auth.issue_demo_token()
    assert caught.value.code == "demo_login_disabled"


def test_demo_token_roundtrip_and_expiration(enabled):
    token, lifetime = auth.issue_demo_token()
    claims = auth._decode_token(token)
    assert claims["sub"] == str(auth.demo_user().id)
    assert claims["mode"] == "demo"
    assert lifetime == 8 * 3600
    assert claims["exp"] - claims["iat"] == lifetime


def test_demo_requires_a_long_signing_secret(enabled, monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: enabled.model_copy(update={"demo_auth_secret": "short"}))
    assert not auth.demo_available()


def test_disabling_demo_rejects_already_issued_tokens(enabled, monkeypatch):
    token, _ = auth.issue_demo_token()
    monkeypatch.setattr(auth, "get_settings", lambda: enabled.model_copy(update={"demo_auth_enabled": False}))
    with pytest.raises(errors.AppError) as caught:
        auth._decode_token(token)
    assert caught.value.status_code == 401


@pytest.mark.parametrize("change", [
    {"sub": "00000000-0000-0000-0000-000000000000"},
    {"email": "other@example.com"}, {"mode": "admin"},
    {"iss": "other-app"}, {"aud": "other-api"},
    {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
])
def test_modified_identity_and_expired_tokens_are_rejected(enabled, change):
    token, _ = auth.issue_demo_token()
    claims = jwt.decode(token, options={"verify_signature": False})
    claims.update(change)
    tampered = jwt.encode(claims, enabled.demo_auth_secret, algorithm="HS256")
    with pytest.raises(errors.AppError) as caught:
        auth._decode_token(tampered)
    assert caught.value.status_code == 401


def test_wrong_signature_cannot_open_the_demo(enabled):
    token, _ = auth.issue_demo_token()
    claims = jwt.decode(token, options={"verify_signature": False})
    forged = jwt.encode(claims, "different-key-abcdefghijklmnopqrstuvwxyz", algorithm="HS256")
    with pytest.raises(errors.AppError):
        auth._decode_token(forged)


async def test_login_always_uses_the_configured_demo_identity(enabled, monkeypatch):
    ensure = AsyncMock()
    monkeypatch.setattr(routes, "ensure_profile", ensure)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/demo", json={"email": "attacker@example.com", "user_id": "chosen"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["user"]["email"] == enabled.demo_auth_email
    assert auth._decode_token(response.json()["access_token"])["sub"] == str(auth.demo_user().id)
    ensure.assert_awaited_once()
