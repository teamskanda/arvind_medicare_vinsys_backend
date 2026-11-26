"""Chatbot and knowledge-base endpoints.

Routers stay thin: validate input, check ownership via helpers, shape the
response. All business logic lives in `helpers/`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from helpers import chatbots as chatbot_helpers
from helpers import crawling, errors, url_validation, screenshots
from helpers.auth import CurrentUser
from helpers.database import get_db
from models.base import ChatbotStatus
from models.chatbot import (
    Chatbot,
    ChatbotCreate,
    ChatbotListOut,
    ChatbotOut,
    ChatbotStats,
    ChatbotUpdate,
)
from models.website_page import (
    WebsitePageDetail,
    WebsitePageListOut,
    WebsitePageSummary,
    WebsitePageUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["chatbots"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


def _to_out(chatbot: Chatbot, stats: ChatbotStats, *, include_embed: bool = False) -> ChatbotOut:
    out = ChatbotOut.model_validate(chatbot)
    out.stats = stats
    out.is_published = chatbot.status == ChatbotStatus.PUBLISHED.value
    if include_embed:
        out.embed_url_preview = chatbot_helpers.embed_url_preview(chatbot.public_id)
    return out


@router.get("", response_model=ChatbotListOut)
async def list_chatbots(user: CurrentUser, db: DbSession) -> ChatbotListOut:
    """List the signed-in customer's chatbots, newest first."""
    chatbots = await chatbot_helpers.list_chatbots(db, user)
    stats = await chatbot_helpers.get_stats_map(db, user, [c.id for c in chatbots])
    return ChatbotListOut(
        chatbots=[_to_out(c, stats.get(c.id, ChatbotStats())) for c in chatbots],
        total=len(chatbots),
    )


@router.post("", response_model=ChatbotOut, status_code=status.HTTP_201_CREATED)
async def create_chatbot(
    payload: ChatbotCreate, user: CurrentUser, db: DbSession
) -> ChatbotOut:
    """Create a chatbot from a website URL and immediately start the crawl.

    The customer submits only a URL. The name, site title and brand colour
    are derived, so there is no form to fill before scraping begins.

    Order matters: validate and check robots.txt *before* creating anything,
    so a bad URL leaves no orphan record and spends no Firecrawl credits.
    """
    validated = await url_validation.validate_url(payload.website_url)

    allowed, note = await url_validation.check_robots_allowed(validated)
    if not allowed:
        raise errors.robots_disallowed(note or "This website disallows automated crawling.")

    # Best effort, never fatal: reachability plus title/brand extraction.
    metadata = await url_validation.probe_site(validated)

    chatbot = await chatbot_helpers.create_chatbot(
        db,
        user,
        validated=validated,
        metadata=metadata,
        requested_name=payload.name,
    )

    try:
        await crawling.start_crawl(db, user, chatbot)
    except errors.AppError as exc:
        # Keep the chatbot; surface the failure on the progress screen with a
        # Retry option instead of discarding the customer's work.
        logger.warning("Crawl could not start for chatbot %s: %s", chatbot.id, exc.message)
        await crawling.record_failed_start(db, user, chatbot, exc.message)
        await db.refresh(chatbot)

    screenshots.on_chatbot_created(chatbot.id, user)
    return _to_out(chatbot, ChatbotStats(), include_embed=True)


@router.get("/{chatbot_id}", response_model=ChatbotOut)
async def get_chatbot(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> ChatbotOut:
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    stats = await chatbot_helpers.get_stats(db, user, chatbot_id)
    return _to_out(chatbot, stats, include_embed=True)


@router.patch("/{chatbot_id}", response_model=ChatbotOut)
async def update_chatbot(
    chatbot_id: uuid.UUID, payload: ChatbotUpdate, user: CurrentUser, db: DbSession
) -> ChatbotOut:
    if payload.name is None:
        raise errors.AppError(
            "nothing_to_update", "No changes were provided.", status_code=422
        )
    chatbot = await chatbot_helpers.update_chatbot_name(db, user, chatbot_id, payload.name)
    stats = await chatbot_helpers.get_stats(db, user, chatbot_id)
    return _to_out(chatbot, stats, include_embed=True)


@router.delete(
    "/{chatbot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_chatbot(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> Response:
    """Delete a chatbot. Pages and crawl jobs cascade in the database."""
    await chatbot_helpers.delete_chatbot(db, user, chatbot_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


@router.get("/{chatbot_id}/pages", response_model=WebsitePageListOut)
async def list_pages(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> WebsitePageListOut:
    """List stored pages without their content, ordered by usefulness."""
    pages = await chatbot_helpers.list_pages(db, user, chatbot_id)
    return WebsitePageListOut(
        pages=[WebsitePageSummary.model_validate(p) for p in pages],
        total=len(pages),
    )


@router.get("/{chatbot_id}/pages/{page_id}", response_model=WebsitePageDetail)
async def get_page(
    chatbot_id: uuid.UUID, page_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> WebsitePageDetail:
    """Return one page including the extracted Markdown."""
    page = await chatbot_helpers.get_page(db, user, chatbot_id, page_id)
    return WebsitePageDetail.model_validate(page)


@router.patch("/{chatbot_id}/pages/{page_id}", response_model=WebsitePageSummary)
async def update_page(
    chatbot_id: uuid.UUID,
    page_id: uuid.UUID,
    payload: WebsitePageUpdate,
    user: CurrentUser,
    db: DbSession,
) -> WebsitePageSummary:
    """Include or exclude a page from the future AI knowledge base."""
    page = await chatbot_helpers.set_page_included(
        db, user, chatbot_id, page_id, payload.is_included
    )
    return WebsitePageSummary.model_validate(page)
