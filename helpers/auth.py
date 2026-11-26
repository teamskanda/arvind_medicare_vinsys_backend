"""Authentication: the shared demo identity, and real first-party accounts.

Two signed-JWT modes share one verification entrypoint, `_decode_token`:

  "demo"    — one fixed, server-configured identity. Explicit opt-in via
              DEMO_AUTH_ENABLED; disabling it rejects all demo tokens.
  "account" — a real customer, created by `/api/auth/signup` and signed in
              by `/api/auth/login`, with a bcrypt password hash in `profiles`.

Every protected endpoint depends on `CurrentUser` (`require_user` below),
which returns an identity that came only from a verified token signature —
never from a request body or query parameter — so a caller cannot assert
someone else's `owner_id`. The browser carries the token in an HttpOnly
cookie via the Next.js backend proxy for both modes.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import anyio.to_thread
import bcrypt
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import errors
from helpers.database import get_db
from models.user import AuthenticatedUser, Profile

logger = logging.getLogger(__name__)
_bearer_scheme = HTTPBearer(auto_error=False)
_ensured_profiles: dict[uuid.UUID, tuple[str, str | None]] = {}
_DEMO_ISSUER = "skanda-demo"
_ACCOUNT_ISSUER = "skanda-account"
_JWT_AUDIENCE = "skanda-api"


# ---------------------------------------------------------------------------
# Demo mode — one shared, server-configured identity. Unchanged from before
# real accounts existed.
# ---------------------------------------------------------------------------


def demo_user() -> AuthenticatedUser:
    settings = get_settings()
    email = settings.demo_auth_email.strip().lower()
    return AuthenticatedUser(
        id=uuid.uuid5(uuid.NAMESPACE_URL, "skanda-demo:" + email),
        email=email, full_name=settings.demo_auth_name,
    )


def demo_available() -> bool:
    settings = get_settings()
    return settings.demo_auth_enabled and len(settings.demo_auth_secret) >= 32


def issue_demo_token() -> tuple[str, int]:
    if not demo_available():
        raise errors.AppError("demo_login_disabled", "Demo login is not enabled on this server.", status_code=503)
    settings = get_settings()
    user = demo_user()
    now = datetime.now(timezone.utc)
    lifetime = settings.demo_auth_ttl_hours * 3600
    token = jwt.encode({
        "sub": str(user.id), "email": user.email, "name": user.full_name,
        "iss": _DEMO_ISSUER, "aud": _JWT_AUDIENCE,
        "iat": now, "exp": now + timedelta(seconds=lifetime), "mode": "demo",
    }, settings.demo_auth_secret, algorithm="HS256")
    return token, lifetime


def _decode_demo_token(token: str) -> dict[str, Any]:
    try:
        if not demo_available():
            raise errors.unauthenticated()
        claims = jwt.decode(
            token, get_settings().demo_auth_secret, algorithms=["HS256"],
            issuer=_DEMO_ISSUER, audience=_JWT_AUDIENCE,
            options={"require": ["exp", "sub", "iss", "aud", "iat", "mode"]},
        )
        user = demo_user()
        if claims.get("mode") != "demo" or claims.get("sub") != str(user.id) or claims.get("email") != user.email:
            raise errors.unauthenticated()
        return claims
    except jwt.ExpiredSignatureError:
        raise errors.unauthenticated("Your demo session has expired. Please enter the demo again.") from None
    except jwt.PyJWTError:
        raise errors.unauthenticated("Your session token is not valid. Please sign in again.") from None


# ---------------------------------------------------------------------------
# Account mode — real customers, stored in `profiles` with a bcrypt hash.
# ---------------------------------------------------------------------------

_BCRYPT_ROUNDS = 12

# A precomputed hash of a fixed, never-used password. `authenticate_account`
# checks against it when the email does not exist, so that path costs the
# same bcrypt round as a real wrong-password check and a timing attack
# cannot tell the two apart.
_DUMMY_PASSWORD_HASH = bcrypt.hashpw(
    b"skanda-ai-timing-placeholder", bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
).decode("ascii")

_GENERIC_LOGIN_ERROR = "The email or password is incorrect."

_ephemeral_account_secret: bytes | None = None


def hash_password(password: str) -> str:
    """bcrypt, an established, audited password hash — never a custom scheme."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed or foreign hash is a mismatch, not a server error.
        return False


def _account_signing_key() -> bytes:
    """Mirrors `helpers.visitor_session._signing_key`'s ephemeral fallback."""
    global _ephemeral_account_secret
    settings = get_settings()
    if settings.account_auth_secret:
        return settings.account_auth_secret.encode("utf-8")

    if _ephemeral_account_secret is None:
        _ephemeral_account_secret = secrets.token_bytes(32)
        logger.warning(
            "ACCOUNT_AUTH_SECRET is not set. Using a random per-process key: "
            "every customer will be signed out whenever the API restarts, and "
            "sessions will not work at all across multiple workers. Set it in "
            "backend/.env before production."
        )
    return _ephemeral_account_secret


def issue_account_token(profile: Profile) -> tuple[str, int]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    lifetime = settings.account_auth_ttl_hours * 3600
    token = jwt.encode({
        "sub": str(profile.id), "email": profile.email, "name": profile.full_name,
        "iss": _ACCOUNT_ISSUER, "aud": _JWT_AUDIENCE,
        "iat": now, "exp": now + timedelta(seconds=lifetime), "mode": "account",
    }, _account_signing_key(), algorithm="HS256")
    return token, lifetime


def _decode_account_token(token: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            token, _account_signing_key(), algorithms=["HS256"],
            issuer=_ACCOUNT_ISSUER, audience=_JWT_AUDIENCE,
            options={"require": ["exp", "sub", "iss", "aud", "iat", "mode"]},
        )
        if claims.get("mode") != "account":
            raise errors.unauthenticated()
        return claims
    except jwt.ExpiredSignatureError:
        raise errors.unauthenticated("Your session has expired. Please sign in again.") from None
    except jwt.PyJWTError:
        raise errors.unauthenticated("Your session token is not valid. Please sign in again.") from None


async def create_account(db: AsyncSession, *, email: str, password: str, full_name: str | None) -> Profile:
    """Create a first-party account. `email` must already be normalised.

    No email-verification link is sent (no outbound mail provider is
    configured in this build) — see the limitation noted on `SignupInput`.
    """
    existing = await db.execute(select(Profile).where(func.lower(Profile.email) == email))
    if existing.scalar_one_or_none() is not None:
        raise errors.AppError(
            "email_taken", "An account with that email already exists. Try logging in instead.",
            status_code=409,
        )

    password_hash = await anyio.to_thread.run_sync(hash_password, password)
    profile = Profile(id=uuid.uuid4(), email=email, full_name=full_name, password_hash=password_hash)
    db.add(profile)
    try:
        await db.commit()
    except IntegrityError:
        # A concurrent signup won the unique-email race between the SELECT
        # above and this INSERT.
        await db.rollback()
        raise errors.AppError(
            "email_taken", "An account with that email already exists. Try logging in instead.",
            status_code=409,
        ) from None
    await db.refresh(profile)
    return profile


async def authenticate_account(db: AsyncSession, *, email: str, password: str) -> Profile:
    """`email` must already be normalised (`LoginInput` does this)."""
    result = await db.execute(select(Profile).where(func.lower(Profile.email) == email))
    profile = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if profile is None or profile.password_hash is None:
        await anyio.to_thread.run_sync(verify_password, password, _DUMMY_PASSWORD_HASH)
        raise errors.AppError("invalid_credentials", _GENERIC_LOGIN_ERROR, status_code=401)

    if profile.locked_until is not None and profile.locked_until > now:
        minutes_left = max(1, int((profile.locked_until - now).total_seconds() // 60) + 1)
        raise errors.AppError(
            "account_locked",
            f"Too many failed attempts on this account. Try again in about {minutes_left} minute(s).",
            status_code=423,
        )

    settings = get_settings()
    ok = await anyio.to_thread.run_sync(verify_password, password, profile.password_hash)
    if not ok:
        profile.failed_login_attempts += 1
        if profile.failed_login_attempts >= settings.login_lockout_threshold:
            profile.locked_until = now + timedelta(minutes=settings.login_lockout_minutes)
        await db.commit()
        raise errors.AppError("invalid_credentials", _GENERIC_LOGIN_ERROR, status_code=401)

    if profile.failed_login_attempts or profile.locked_until is not None:
        profile.failed_login_attempts = 0
        profile.locked_until = None
        await db.commit()
    return profile


# ---------------------------------------------------------------------------
# A coarse per-IP cap on login/signup attempts, independent of which email is
# targeted. Weak on its own — an attacker can rotate addresses — which is
# why the per-account lockout above is the control that actually holds.
# Mirrors `helpers.conversations.enforce_ip_rate_limit`.
# ---------------------------------------------------------------------------

_auth_ip_hits: dict[str, list[float]] = {}
_AUTH_IP_WINDOW_SECONDS = 60.0
_AUTH_IP_TABLE_MAX_KEYS = 20_000


def enforce_auth_ip_rate_limit(client_ip: str | None) -> None:
    settings = get_settings()
    if not client_ip:
        return
    key = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:24]
    now = time.monotonic()
    cutoff = now - _AUTH_IP_WINDOW_SECONDS

    hits = [stamp for stamp in _auth_ip_hits.get(key, ()) if stamp > cutoff]
    if len(hits) >= settings.auth_ip_limit_per_minute:
        _auth_ip_hits[key] = hits
        raise errors.AppError(
            "rate_limited", "Too many attempts from this connection. Please wait a minute and try again.",
            status_code=429, retryable=True,
        )

    hits.append(now)
    _auth_ip_hits[key] = hits
    if len(_auth_ip_hits) > _AUTH_IP_TABLE_MAX_KEYS:
        for stale_key in [k for k, v in _auth_ip_hits.items() if not v or max(v) <= cutoff]:
            _auth_ip_hits.pop(stale_key, None)


def client_ip_from(request: Request) -> str | None:
    """Best-effort client address. Duplicated from `routers.deployment`
    rather than imported, to keep auth free of a dependency on the
    deployment router; see that module for the forwarding caveat."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    return request.client.host if request.client else None


# ---------------------------------------------------------------------------
# Shared verification entrypoint, dispatching on the token's claimed mode.
# ---------------------------------------------------------------------------


def _decode_token(token: str) -> dict[str, Any]:
    """Verify a bearer token and return its claims, for either mode.

    The mode is read once *without* signature verification purely to choose
    which secret and issuer to check against — the actual trust comes from
    the fully verified decode that follows in `_decode_demo_token` /
    `_decode_account_token`. A token cannot forge its way into the account
    path by claiming `mode: "account"`: it still has to verify against
    `_account_signing_key()`, a secret a demo token was never signed with.
    """
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "HS256":
            raise errors.unauthenticated("Your session token uses an unsupported algorithm.")
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        raise errors.unauthenticated("Your session token is not valid. Please sign in again.") from None

    mode = unverified.get("mode")
    if mode == "demo":
        return _decode_demo_token(token)
    if mode == "account":
        return _decode_account_token(token)
    raise errors.unauthenticated()


def _claims_to_user(claims: dict[str, Any]) -> AuthenticatedUser:
    if claims.get("mode") == "demo":
        # Identity is configured server-side, never selected by the token.
        return demo_user()
    return AuthenticatedUser(id=uuid.UUID(claims["sub"]), email=claims["email"], full_name=claims.get("name"))


async def ensure_profile(db: AsyncSession, user: AuthenticatedUser) -> None:
    """Create or refresh the customer's `profiles` row.

    For a real account this row already exists (created at signup); the
    upsert below only ever touches `email`/`full_name`, so it never clears
    `password_hash`. For the demo identity this is how its row gets created
    lazily on first use.
    """
    identity = (user.email, user.full_name)
    if _ensured_profiles.get(user.id) == identity:
        return

    await db.execute(
        text(
            """
            INSERT INTO profiles (id, full_name, email)
            VALUES (:id, :full_name, :email)
            ON CONFLICT (id) DO UPDATE
                SET email     = EXCLUDED.email,
                    full_name = COALESCE(EXCLUDED.full_name, profiles.full_name)
            """
        ),
        {"id": user.id, "full_name": user.full_name, "email": user.email},
    )
    await db.commit()

    if len(_ensured_profiles) > 5000:
        _ensured_profiles.clear()
    _ensured_profiles[user.id] = identity


async def require_user(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthenticatedUser:
    """Dependency for every protected endpoint.

    The returned identity comes only from a verified token — never from a
    request body or query parameter — so a caller cannot assert someone
    else's `owner_id`.
    """
    if credentials is None or not credentials.credentials:
        raise errors.unauthenticated()

    claims = await anyio.to_thread.run_sync(_decode_token, credentials.credentials)
    user = _claims_to_user(claims)
    await ensure_profile(db, user)
    request.state.user_id = str(user.id)
    return user


CurrentUser = Annotated[AuthenticatedUser, Depends(require_user)]
