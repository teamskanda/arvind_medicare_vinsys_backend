"""Shared demo entry, real account sign-up/sign-in, and the verified profile."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from helpers.auth import (
    CurrentUser,
    authenticate_account,
    client_ip_from,
    create_account,
    demo_available,
    demo_user,
    ensure_profile,
    enforce_auth_ip_rate_limit,
    issue_account_token,
    issue_demo_token,
)
from helpers.database import get_db
from models.user import LoginInput, Profile, ProfileOut, SignupInput

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/status")
async def auth_status() -> dict[str, bool]:
    return {"demo_enabled": demo_available()}


@router.post("/demo")
async def enter_demo(
    response: Response, db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Enter the explicitly enabled shared demo account. No customer signup."""
    token, lifetime = issue_demo_token()
    user = demo_user()
    await ensure_profile(db, user)
    response.headers["Cache-Control"] = "no-store"
    return {"access_token": token, "expires_in": lifetime, "token_type": "bearer", "user": user.model_dump(mode="json")}


@router.post("/signup", status_code=201)
async def signup(
    payload: SignupInput, request: Request, response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Create a real, first-party account.

    See `models.user.SignupInput` for the stated limitation: no
    email-verification link is sent, because no outbound mail provider is
    configured in this build.
    """
    enforce_auth_ip_rate_limit(client_ip_from(request))
    profile = await create_account(
        db, email=payload.email, password=payload.password, full_name=payload.full_name,
    )
    token, lifetime = issue_account_token(profile)
    response.headers["Cache-Control"] = "no-store"
    return {
        "access_token": token, "expires_in": lifetime, "token_type": "bearer",
        "user": ProfileOut.model_validate(profile).model_dump(mode="json"),
    }


@router.post("/login")
async def login(
    payload: LoginInput, request: Request, response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    enforce_auth_ip_rate_limit(client_ip_from(request))
    profile = await authenticate_account(db, email=payload.email, password=payload.password)
    token, lifetime = issue_account_token(profile)
    response.headers["Cache-Control"] = "no-store"
    return {
        "access_token": token, "expires_in": lifetime, "token_type": "bearer",
        "user": ProfileOut.model_validate(profile).model_dump(mode="json"),
    }


@router.get("/me", response_model=ProfileOut)
async def read_current_profile(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProfileOut:
    """Return the signed-in customer's profile.

    `require_user` has already created the profile row if this is the
    customer's first authenticated request, so this always finds a row.
    """
    result = await db.execute(select(Profile).where(Profile.id == user.id))
    profile = result.scalar_one_or_none()
    if profile is None:
        # Defensive: respond from verified token claims rather than erroring.
        return ProfileOut(
            id=user.id,
            full_name=user.full_name,
            email=user.email,
            created_at=datetime.now(timezone.utc),
        )
    return ProfileOut.model_validate(profile)
