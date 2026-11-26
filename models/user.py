"""Application profiles stored in Neon.

Two identities share this one table: the single shared demo user (a stable
UUID, no password) and real first-party accounts created through
`/api/auth/signup` (a random UUID, a bcrypt password hash). Every other
table's `owner_id -> profiles.id` foreign key needs no change either way.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import field_validator
from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from config import get_settings
from models.base import ApiSchema, Base

# Deliberately permissive: this rejects obvious typos (no "@", no domain part)
# without the maintenance burden of a full RFC 5322 implementation. The only
# thing that actually proves an address works is receiving mail at it, which
# this phase does not implement (see the signup docstring below).
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
MAX_EMAIL_CHARACTERS = 254
MAX_FULL_NAME_CHARACTERS = 120
MAX_PASSWORD_CHARACTERS = 128  # bcrypt itself silently ignores bytes past 72


class Profile(Base):
    __tablename__ = "profiles"

    # Stable for the demo identity; a random uuid4 for a real account.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    # NULL for the demo profile and for rows created before accounts existed.
    # Never used to authenticate the demo flow, which keeps its own signed
    # JWT path in helpers/auth.py.
    password_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Profile id={self.id} email={self.email!r}>"


class ProfileOut(ApiSchema):
    id: uuid.UUID
    full_name: str | None
    email: str
    created_at: datetime


class AuthenticatedUser(ApiSchema):
    """The verified identity attached to a request by `helpers.auth`.

    Produced from a validated JWT — demo or account — never from
    client-supplied body fields.
    """

    id: uuid.UUID
    email: str
    full_name: str | None = None


def _normalised_email(value: str) -> str:
    value = value.strip().lower()
    if not value or len(value) > MAX_EMAIL_CHARACTERS or not _EMAIL_RE.match(value):
        raise ValueError("Enter a valid email address.")
    return value


def _checked_full_name(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > MAX_FULL_NAME_CHARACTERS:
        raise ValueError(f"Name must be {MAX_FULL_NAME_CHARACTERS} characters or fewer.")
    return value


class SignupInput(ApiSchema):
    """A new account.

    No email-verification link is sent — there is no outbound mail provider
    configured in this build — so an account is usable the moment it is
    created. That is a real limitation (anyone can claim any email address
    they merely type), not a hidden one: it is stated in the README rather
    than disguised behind a fake "check your inbox" screen.
    """

    email: str
    password: str
    full_name: str | None = None

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        return _normalised_email(value)

    @field_validator("full_name")
    @classmethod
    def _validate_full_name(cls, value: str | None) -> str | None:
        return _checked_full_name(value)

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        minimum = get_settings().password_min_length
        if len(value) < minimum:
            raise ValueError(f"Password must be at least {minimum} characters.")
        if len(value) > MAX_PASSWORD_CHARACTERS:
            raise ValueError(f"Password must be {MAX_PASSWORD_CHARACTERS} characters or fewer.")
        return value


class LoginInput(ApiSchema):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        # Normalised the same way as signup so "Foo@Example.com" matches the
        # row created as "foo@example.com" — but never rejected here: a typo
        # must fail as "incorrect email or password", not a 422 that tells an
        # attacker the address is malformed before the password is even seen.
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        if not value or len(value) > MAX_PASSWORD_CHARACTERS:
            raise ValueError("Enter your password.")
        return value
