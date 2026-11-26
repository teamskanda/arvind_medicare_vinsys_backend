"""Customisation endpoints: the studio's settings, and uploaded images.

Two routers, one trust boundary:

  * `router`        — authenticated, mounted at /api/chatbots. Reads and
                      writes one customer's own chatbot through helpers that
                      filter on `owner_id`.
  * `public_router` — anonymous, mounted at /api/public. Serves uploaded image
                      bytes, because the chat window that displays them runs
                      on a visitor's browser with no account.

Thin by design, as everywhere else: validation lives in
`models/customization.py`, resolution and persistence in
`helpers/customization.py`. What is here is HTTP.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from helpers import chat_engine, customization, deployment, widget
from helpers import chatbots as chatbot_helpers
from helpers.auth import CurrentUser
from helpers.database import get_db
from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.customization import (
    AssetOut,
    AssetUploadRequest,
    CustomizationOut,
    CustomizationUpdate,
    StudioChatRequest,
)
from models.message import ChatResponse
from models.deployment import ChatbotPreviewOut
from models.user import AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["customization"])
public_router = APIRouter(prefix="/api/public", tags=["public"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

# How long a browser may reuse an uploaded image. Assets are immutable —
# changing an avatar creates a new row with a new id — so this can be long
# without the staleness problem a mutable URL would have.
ASSET_CACHE_SECONDS = 60 * 60 * 24 * 30


async def _customization_out(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot
) -> CustomizationOut:
    published = (
        customization.resolve_state(chatbot, published=True)
        if isinstance(chatbot.published_config, dict) and chatbot.published_config
        else None
    )
    assets = await customization.list_assets(db, user, chatbot.id)
    readiness = await deployment.publish_readiness(db, user, chatbot)

    return CustomizationOut(
        chatbot_id=chatbot.id,
        public_id=chatbot.public_id,
        status=chatbot.status,
        is_published=chatbot.status == ChatbotStatus.PUBLISHED.value,
        can_publish=readiness.can_publish,
        publish_blocked_reason=readiness.reason,
        draft=customization.resolve_state(chatbot, published=False),
        published=published,
        has_unpublished_changes=customization.has_unpublished_changes(chatbot),
        overrides=customization.overridden_paths(chatbot),
        assets=[customization.asset_out(asset) for asset in assets],
        options=customization.options(),
    )


# ===========================================================================
# Settings
# ===========================================================================


@router.get("/{chatbot_id}/customization", response_model=CustomizationOut)
async def get_customization(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> CustomizationOut:
    """Everything the customisation studio renders.

    Returns the resolved draft, the published snapshot when there is one, the
    catalogue of valid choices, and which settings the customer has
    explicitly set. Resolved rather than raw, so the studio's controls show
    the values the chatbot is actually using — including the ones derived from
    the website's own brand colour.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    return await _customization_out(db, user, chatbot)


@router.put("/{chatbot_id}/customization", response_model=CustomizationOut)
async def save_customization(
    chatbot_id: uuid.UUID,
    payload: CustomizationUpdate,
    user: CurrentUser,
    db: DbSession,
) -> CustomizationOut:
    """Save the draft. Only the groups present in the body are written.

    PUT rather than PATCH because the studio saves a whole group at a time and
    the semantics within a group are still partial-by-key; an explicit `null`
    clears one setting back to its default. Nothing here changes what visitors
    see — that needs a publish.
    """
    chatbot = await customization.save_draft(db, user, chatbot_id, payload)
    return await _customization_out(db, user, chatbot)


@router.post("/{chatbot_id}/customization/preview", response_model=ChatbotPreviewOut)
async def preview_customization(
    chatbot_id: uuid.UUID,
    payload: CustomizationUpdate,
    user: CurrentUser,
    db: DbSession,
) -> ChatbotPreviewOut:
    """Resolve an unsaved configuration so the studio can show it instantly.

    Read-only: nothing is persisted, and the customer's saved draft and their
    live chatbot are both untouched. It exists so the studio's preview is
    produced by the code that serves the real chatbot — the same palette
    resolver and the same public contract — instead of a second
    implementation in the browser that would eventually disagree with it.
    """
    shadow = await customization.preview_chatbot(db, user, chatbot_id, payload)
    return ChatbotPreviewOut(
        state=customization.resolve_state(shadow, published=False),
        # `published=False` is what makes this a preview of the draft. Every
        # genuinely public caller leaves the default alone.
        public_config=deployment.public_config(shadow, published=False),
        widget_script=widget.build_loader_script(
            shadow,
            customization.resolve_launcher(shadow, published=False),
            window_radius=customization.resolve_appearance(shadow, published=False).border_radius,
            preview=True,
        ),
        widget_config=widget.build_loader_config(
            shadow, customization.resolve_launcher(shadow, published=False),
            window_radius=customization.resolve_appearance(shadow, published=False).border_radius,
            preview=True,
        ),
    )


@router.post("/{chatbot_id}/customization/chat", response_model=ChatResponse)
async def preview_chat(
    chatbot_id: uuid.UUID, payload: StudioChatRequest,
    user: CurrentUser, db: DbSession,
    stream: bool = False,
) -> ChatResponse | StreamingResponse:
    """Run the existing retrieval/Gemma pipeline with unsaved owner settings."""
    shadow = await customization.preview_chatbot(db, user, chatbot_id, payload.customization)
    request = await chat_engine.resolve_playground_turn(db, user, chatbot_id, payload)
    request = replace(request, chatbot=shadow, config=deployment.ai_config_for(shadow, published=False))
    turn = await chat_engine.prepare(db, request)
    if not stream:
        return await chat_engine.complete(db, turn)
    return StreamingResponse(
        chat_engine.sse_events(db, turn), media_type="text/event-stream",
        headers=chat_engine.SSE_HEADERS,
    )


@router.post("/{chatbot_id}/customization/reset", response_model=CustomizationOut)
async def reset_customization(
    chatbot_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    scope: Annotated[
        str, Query(description="all | content | appearance | launcher")
    ] = "all",
) -> CustomizationOut:
    """Clear customisations back to the chatbot's derived defaults.

    Uploaded images are kept, so a reset is recoverable — only the references
    to them are cleared.
    """
    chatbot = await customization.reset_draft(db, user, chatbot_id, scope=scope)
    return await _customization_out(db, user, chatbot)


# ===========================================================================
# Uploaded images
# ===========================================================================


@router.post(
    "/{chatbot_id}/assets",
    response_model=AssetOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_asset(
    chatbot_id: uuid.UUID,
    payload: AssetUploadRequest,
    user: CurrentUser,
    db: DbSession,
) -> AssetOut:
    """Store an avatar, logo or launcher icon.

    The declared type is ignored: the image is identified from its own bytes
    and only PNG, JPG, WebP and GIF are accepted. SVG is refused — it is a
    document that can carry script, and these bytes are served from our own
    origin.
    """
    asset = await customization.create_asset(
        db, user, chatbot_id, kind=payload.kind, data=payload.data
    )
    return customization.asset_out(asset)


@router.delete(
    "/{chatbot_id}/assets/{asset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_asset(
    chatbot_id: uuid.UUID,
    asset_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
) -> Response:
    """Delete an uploaded image, unless the published chatbot still uses it."""
    await customization.delete_asset(db, user, chatbot_id, asset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@public_router.get("/assets/{asset_id}")
async def get_asset(asset_id: uuid.UUID, db: DbSession) -> Response:
    """Serve one uploaded image to anyone holding its id.

    Anonymous on purpose: this is the avatar in a chat window on a stranger's
    screen. The asset id is an unguessable uuid acting as a capability, the
    same way `public_id` does — it is handed out only by a chatbot's own
    configuration, and what it grants is an image the customer chose to
    display publicly.

    The response headers are the interesting part. `nosniff` plus a
    `default-src 'none'` policy and `sandbox` mean that even if a file somehow
    got stored that a browser would rather treat as a document, it would have
    no script, no styles, no network and no origin to use them in.
    """
    asset = await customization.get_asset(db, asset_id)
    return Response(
        content=asset.data,
        media_type=asset.content_type,
        headers={
            "Cache-Control": f"public, max-age={ASSET_CACHE_SECONDS}, immutable",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Disposition": "inline",
            # Loaded by an <img> from the customer's own website.
            "Access-Control-Allow-Origin": "*",
        },
    )
