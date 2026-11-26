"""Real first-party accounts: password hashing, account tokens, lockout,
per-IP rate limiting, and the signup/login endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
from pydantic import ValidationError

import main
from config import get_settings
from helpers import auth, errors
from models.user import LoginInput, Profile, SignupInput
from routers import auth as routes


def _profile(**overrides) -> Profile:
    defaults = dict(
        id=uuid.uuid4(), email="person@example.com", full_name="Person",
        password_hash=None, failed_login_attempts=0, locked_until=None,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Profile(**defaults)


# ---------------------------------------------------------------------------
# Password hashing — an established algorithm, used correctly.
# ---------------------------------------------------------------------------


def test_hashing_the_same_password_twice_salts_differently():
    first = auth.hash_password("correct horse battery staple")
    second = auth.hash_password("correct horse battery staple")
    assert first != second
    assert auth.verify_password("correct horse battery staple", first)
    assert auth.verify_password("correct horse battery staple", second)


def test_verify_password_rejects_the_wrong_password():
    hashed = auth.hash_password("right-password")
    assert not auth.verify_password("wrong-password", hashed)


def test_verify_password_handles_a_malformed_hash_without_raising():
    assert not auth.verify_password("anything", "not-a-bcrypt-hash-at-all")
    assert not auth.verify_password("anything", "")


# ---------------------------------------------------------------------------
# Account tokens
# ---------------------------------------------------------------------------


@pytest.fixture
def account_secret(monkeypatch):
    settings = get_settings().model_copy(update={
        "account_auth_secret": "account-signing-secret-at-least-32-chars",
        "account_auth_ttl_hours": 1,
    })
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def demo_secret(monkeypatch):
    settings = get_settings().model_copy(update={
        "demo_auth_enabled": True,
        "demo_auth_secret": "demo-signing-secret-at-least-32-characters",
    })
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    return settings


def test_account_token_round_trips(account_secret):
    profile = _profile()
    token, lifetime = auth.issue_account_token(profile)
    assert lifetime == 3600
    claims = auth._decode_token(token)
    assert claims["mode"] == "account"
    assert claims["sub"] == str(profile.id)
    assert claims["email"] == profile.email
    user = auth._claims_to_user(claims)
    assert user.id == profile.id
    assert user.email == profile.email
    assert user.full_name == profile.full_name


def test_an_unset_account_secret_still_works_but_warns_once(monkeypatch, caplog):
    settings = get_settings().model_copy(update={"account_auth_secret": None})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "_ephemeral_account_secret", None)
    profile = _profile()
    with caplog.at_level("WARNING"):
        token, _ = auth.issue_account_token(profile)
        auth.issue_account_token(profile)  # second call must not warn again
    assert caplog.text.count("ACCOUNT_AUTH_SECRET is not set") == 1
    assert auth._decode_token(token)["sub"] == str(profile.id)


@pytest.mark.parametrize("change", [
    {"mode": "demo"}, {"mode": "admin"},
    {"iss": "other-app"}, {"aud": "other-api"},
    {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
])
def test_tampered_account_tokens_are_rejected(account_secret, change):
    profile = _profile()
    token, _ = auth.issue_account_token(profile)
    claims = jwt.decode(token, options={"verify_signature": False})
    claims.update(change)
    tampered = jwt.encode(claims, account_secret.account_auth_secret, algorithm="HS256")
    with pytest.raises(errors.AppError) as caught:
        auth._decode_token(tampered)
    assert caught.value.status_code == 401


def test_wrong_signature_cannot_forge_an_account_token(account_secret):
    profile = _profile()
    token, _ = auth.issue_account_token(profile)
    claims = jwt.decode(token, options={"verify_signature": False})
    forged = jwt.encode(claims, "a-completely-different-signing-key-value", algorithm="HS256")
    with pytest.raises(errors.AppError):
        auth._decode_token(forged)


def test_a_demo_signed_token_cannot_pass_as_an_account_token(account_secret, demo_secret):
    # Claiming mode "account" while signed with the demo secret must fail:
    # the account path only ever verifies against _account_signing_key().
    now = datetime.now(timezone.utc)
    forged = jwt.encode({
        "sub": str(uuid.uuid4()), "email": "attacker@example.com", "name": None,
        "iss": "skanda-account", "aud": "skanda-api",
        "iat": now, "exp": now + timedelta(hours=1), "mode": "account",
    }, demo_secret.demo_auth_secret, algorithm="HS256")
    with pytest.raises(errors.AppError) as caught:
        auth._decode_token(forged)
    assert caught.value.status_code == 401


def test_an_unrecognised_mode_is_rejected_before_any_signature_check():
    now = datetime.now(timezone.utc)
    token = jwt.encode({
        "sub": str(uuid.uuid4()), "email": "x@example.com",
        "iss": "nobody", "aud": "nobody", "iat": now, "exp": now + timedelta(hours=1),
        "mode": "superadmin",
    }, "whatever-key-nobody-checks", algorithm="HS256")
    with pytest.raises(errors.AppError) as caught:
        auth._decode_token(token)
    assert caught.value.status_code == 401


# ---------------------------------------------------------------------------
# Signup / login input validation
# ---------------------------------------------------------------------------


def test_signup_input_normalises_email_and_trims_the_name():
    payload = SignupInput(email=" Person@Example.COM ", password="longenough1", full_name="  Jane Doe  ")
    assert payload.email == "person@example.com"
    assert payload.full_name == "Jane Doe"


def test_signup_input_blank_name_becomes_none():
    payload = SignupInput(email="a@b.com", password="longenough1", full_name="   ")
    assert payload.full_name is None


@pytest.mark.parametrize("email", ["not-an-email", "missing-domain@", "@missing-local.com", "   ", "a" * 250 + "@b.com"])
def test_signup_input_rejects_malformed_email(email):
    with pytest.raises(ValidationError):
        SignupInput(email=email, password="longenough1")


def test_signup_input_rejects_a_password_shorter_than_the_configured_minimum():
    with pytest.raises(ValidationError):
        SignupInput(email="a@b.com", password="short1")


def test_signup_input_rejects_an_absurdly_long_password():
    with pytest.raises(ValidationError):
        SignupInput(email="a@b.com", password="x" * 200)


def test_login_input_normalises_but_never_validates_the_email():
    # A malformed address must still reach authenticate_account, which
    # answers "incorrect email or password" rather than a validation error
    # that would tell an attacker the address is syntactically wrong.
    payload = LoginInput(email=" Weird@@Address ", password="whatever")
    assert payload.email == "weird@@address"


def test_login_input_rejects_an_empty_password():
    with pytest.raises(ValidationError):
        LoginInput(email="a@b.com", password="")


# ---------------------------------------------------------------------------
# create_account / authenticate_account — against a fake AsyncSession.
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeSession:
    """Just enough of AsyncSession for these two helpers."""

    def __init__(self, existing: Profile | None):
        self._existing = existing
        self.added: list[Profile] = []
        self.committed = 0
        self.refreshed: list[Profile] = []

    async def execute(self, _stmt):
        return _FakeResult(self._existing)

    def add(self, profile: Profile) -> None:
        self.added.append(profile)

    async def commit(self) -> None:
        self.committed += 1

    async def refresh(self, profile: Profile) -> None:
        self.refreshed.append(profile)


async def test_create_account_hashes_the_password_and_never_stores_it_plain():
    session = _FakeSession(existing=None)
    profile = await auth.create_account(session, email="new@example.com", password="longenough1", full_name="New")
    assert profile.email == "new@example.com"
    assert profile.password_hash != "longenough1"
    assert auth.verify_password("longenough1", profile.password_hash)
    assert session.committed == 1
    assert session.added == [profile]


async def test_create_account_refuses_a_duplicate_email():
    session = _FakeSession(existing=_profile(email="taken@example.com"))
    with pytest.raises(errors.AppError) as caught:
        await auth.create_account(session, email="taken@example.com", password="longenough1", full_name=None)
    assert caught.value.code == "email_taken"
    assert caught.value.status_code == 409
    assert session.added == []


async def test_authenticate_account_accepts_the_right_password():
    stored = _profile(password_hash=auth.hash_password("correct-password"))
    session = _FakeSession(existing=stored)
    profile = await auth.authenticate_account(session, email=stored.email, password="correct-password")
    assert profile is stored
    assert profile.failed_login_attempts == 0


async def test_authenticate_account_rejects_an_unknown_email_generically():
    session = _FakeSession(existing=None)
    with pytest.raises(errors.AppError) as caught:
        await auth.authenticate_account(session, email="ghost@example.com", password="whatever")
    assert caught.value.code == "invalid_credentials"
    assert caught.value.status_code == 401
    assert "incorrect" in caught.value.message.lower()


async def test_authenticate_account_rejects_the_demo_profile_which_has_no_password():
    session = _FakeSession(existing=_profile(password_hash=None))
    with pytest.raises(errors.AppError) as caught:
        await auth.authenticate_account(session, email="person@example.com", password="whatever")
    assert caught.value.code == "invalid_credentials"


async def test_authenticate_account_counts_failed_attempts_and_locks_after_the_threshold(monkeypatch):
    settings = get_settings().model_copy(update={"login_lockout_threshold": 3, "login_lockout_minutes": 15})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    stored = _profile(password_hash=auth.hash_password("correct-password"), failed_login_attempts=2)
    session = _FakeSession(existing=stored)
    with pytest.raises(errors.AppError) as caught:
        await auth.authenticate_account(session, email=stored.email, password="wrong")
    assert caught.value.code == "invalid_credentials"
    assert stored.failed_login_attempts == 3
    assert stored.locked_until is not None
    assert stored.locked_until > datetime.now(timezone.utc)


async def test_authenticate_account_refuses_even_the_right_password_while_locked():
    stored = _profile(
        password_hash=auth.hash_password("correct-password"),
        locked_until=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    session = _FakeSession(existing=stored)
    with pytest.raises(errors.AppError) as caught:
        await auth.authenticate_account(session, email=stored.email, password="correct-password")
    assert caught.value.code == "account_locked"
    assert caught.value.status_code == 423


async def test_authenticate_account_resets_the_counter_on_success():
    stored = _profile(password_hash=auth.hash_password("correct-password"), failed_login_attempts=2)
    session = _FakeSession(existing=stored)
    await auth.authenticate_account(session, email=stored.email, password="correct-password")
    assert stored.failed_login_attempts == 0
    assert stored.locked_until is None


# ---------------------------------------------------------------------------
# Per-IP rate limiting
# ---------------------------------------------------------------------------


def test_auth_ip_rate_limit_trips_after_the_configured_count(monkeypatch):
    monkeypatch.setattr(auth, "_auth_ip_hits", {})
    settings = get_settings().model_copy(update={"auth_ip_limit_per_minute": 3})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    for _ in range(3):
        auth.enforce_auth_ip_rate_limit("203.0.113.9")
    with pytest.raises(errors.AppError) as caught:
        auth.enforce_auth_ip_rate_limit("203.0.113.9")
    assert caught.value.status_code == 429


def test_auth_ip_rate_limit_is_scoped_per_address(monkeypatch):
    monkeypatch.setattr(auth, "_auth_ip_hits", {})
    settings = get_settings().model_copy(update={"auth_ip_limit_per_minute": 1})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    auth.enforce_auth_ip_rate_limit("203.0.113.1")
    auth.enforce_auth_ip_rate_limit("203.0.113.2")  # a different address, unaffected


def test_client_ip_from_prefers_the_forwarded_header():
    request = httpx.Request("GET", "http://test/", headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"})

    class _Scope:
        client = None

    fake = type("FakeRequest", (), {"headers": request.headers, "client": None})()
    assert auth.client_ip_from(fake) == "203.0.113.5"


# ---------------------------------------------------------------------------
# The endpoints themselves, against a mocked helper layer (no real database;
# mirrors the pattern in test_demo_auth.py).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ip_hits(monkeypatch):
    monkeypatch.setattr(auth, "_auth_ip_hits", {})


async def test_signup_endpoint_normalises_input_and_returns_a_token(account_secret, monkeypatch):
    created = _profile(email="new@example.com", full_name="New Person", password_hash="irrelevant-here")
    create = AsyncMock(return_value=created)
    monkeypatch.setattr(routes, "create_account", create)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/signup",
            json={"email": "New@Example.com", "password": "longenough1", "full_name": "New Person"},
        )

    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["user"]["email"] == "new@example.com"
    assert auth._decode_token(body["access_token"])["sub"] == str(created.id)
    create.assert_awaited_once()
    assert create.await_args.kwargs["email"] == "new@example.com"  # lowercased before reaching the helper


async def test_signup_endpoint_rejects_a_short_password_before_touching_the_database(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(routes, "create_account", create)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/signup", json={"email": "a@b.com", "password": "short"})
    assert response.status_code == 422
    create.assert_not_awaited()


async def test_signup_endpoint_surfaces_a_duplicate_email_as_409(account_secret, monkeypatch):
    create = AsyncMock(side_effect=errors.AppError(
        "email_taken", "An account with that email already exists. Try logging in instead.", status_code=409,
    ))
    monkeypatch.setattr(routes, "create_account", create)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/signup", json={"email": "dupe@example.com", "password": "longenough1"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


async def test_login_endpoint_returns_a_token_for_the_authenticated_profile(account_secret, monkeypatch):
    found = _profile(email="person@example.com", password_hash="irrelevant-here")
    authenticate = AsyncMock(return_value=found)
    monkeypatch.setattr(routes, "authenticate_account", authenticate)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/login", json={"email": "Person@Example.com", "password": "whatever-they-typed"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert auth._decode_token(response.json()["access_token"])["sub"] == str(found.id)
    assert authenticate.await_args.kwargs["email"] == "person@example.com"


async def test_login_endpoint_surfaces_invalid_credentials_as_401_without_detail(monkeypatch):
    authenticate = AsyncMock(side_effect=errors.AppError(
        "invalid_credentials", "The email or password is incorrect.", status_code=401,
    ))
    monkeypatch.setattr(routes, "authenticate_account", authenticate)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/login", json={"email": "a@b.com", "password": "wrong"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "invalid_credentials"
    # Generic wording either way — never "no such email" vs "wrong password",
    # which would let an attacker enumerate which addresses have accounts.
    assert body["error"]["message"] == "The email or password is incorrect."


async def test_login_endpoint_surfaces_a_locked_account_as_423(monkeypatch):
    authenticate = AsyncMock(side_effect=errors.AppError(
        "account_locked", "Too many failed attempts on this account. Try again in about 15 minute(s).", status_code=423,
    ))
    monkeypatch.setattr(routes, "authenticate_account", authenticate)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/api/auth/login", json={"email": "a@b.com", "password": "whatever"})
    assert response.status_code == 423


async def test_repeated_login_attempts_from_one_address_hit_the_ip_limit(monkeypatch):
    settings = get_settings().model_copy(update={"auth_ip_limit_per_minute": 2})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    authenticate = AsyncMock(side_effect=errors.AppError("invalid_credentials", "The email or password is incorrect.", status_code=401))
    monkeypatch.setattr(routes, "authenticate_account", authenticate)

    headers = {"x-forwarded-for": "198.51.100.7"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        for _ in range(2):
            response = await client.post("/api/auth/login", json={"email": "a@b.com", "password": "x"}, headers=headers)
            assert response.status_code == 401
        response = await client.post("/api/auth/login", json={"email": "a@b.com", "password": "x"}, headers=headers)
    assert response.status_code == 429
    assert authenticate.await_count == 2  # the third request never reached the helper
