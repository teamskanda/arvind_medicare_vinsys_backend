"""Chatbot and knowledge-base business logic.

Every function that reads or writes a customer-owned row takes the
authenticated user and filters on `owner_id`. There is deliberately no
"fetch by id" helper without an owner argument, so a router cannot
accidentally expose another customer's data.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import Select, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import errors
from helpers.content_processing import derive_chatbot_name
from helpers.url_validation import SiteMetadata, ValidatedUrl
from models.base import ChatbotStatus, CrawlStatus, PageStatus
from models.chatbot import Chatbot, ChatbotStats
from models.crawl_job import CrawlJob
from models.user import AuthenticatedUser
from models.website_page import WebsitePage

logger = logging.getLogger(__name__)

_PUBLIC_ID_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"  # no look-alike characters
_PUBLIC_ID_LENGTH = 22


def generate_public_id() -> str:
    """Unguessable, URL-safe handle for the future iframe route.

    Random rather than sequential: the embed URL is effectively a public
    capability, so it must not be enumerable.
    """
    return "".join(secrets.choice(_PUBLIC_ID_ALPHABET) for _ in range(_PUBLIC_ID_LENGTH))


def embed_url_preview(public_id: str) -> str:
    """The chatbot's permanent public iframe URL.

    Generated from `PUBLIC_WIDGET_BASE_URL`, never a hardcoded domain, so
    moving the deployment host is one environment variable and no stored URL
    goes stale. Live once the chatbot is published (Phase 3).
    """
    return get_settings().embed_url(public_id)


# ---------------------------------------------------------------------------
# Create / read / update / delete
# ---------------------------------------------------------------------------


async def count_chatbots(db: AsyncSession, user: AuthenticatedUser) -> int:
    result = await db.execute(
        select(func.count()).select_from(Chatbot).where(Chatbot.owner_id == user.id)
    )
    return int(result.scalar_one())


async def create_chatbot(
    db: AsyncSession,
    user: AuthenticatedUser,
    *,
    validated: ValidatedUrl,
    metadata: SiteMetadata,
    requested_name: str | None = None,
) -> Chatbot:
    settings = get_settings()

    existing = await count_chatbots(db, user)
    if existing >= settings.max_chatbots_per_user:
        raise errors.AppError(
            "chatbot_limit_reached",
            f"You have reached the limit of {settings.max_chatbots_per_user} chatbots.",
            status_code=409,
        )

    name = requested_name or derive_chatbot_name(metadata.title, validated.host)

    for _ in range(5):  # retry on the astronomically unlikely id collision
        chatbot = Chatbot(
            public_id=generate_public_id(),
            owner_id=user.id,
            name=name[:120],
            website_url=metadata.final_url or validated.url,
            status=ChatbotStatus.DRAFT.value,
            crawl_status=CrawlStatus.PENDING.value,
            draft_config={},
            site_title=metadata.title,
            brand_color=metadata.brand_color,
            favicon_url=metadata.favicon_url,
        )
        db.add(chatbot)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            continue
        await db.refresh(chatbot)
        return chatbot

    raise errors.internal("Could not create the chatbot. Please try again.")


def _owned(statement: Select, user: AuthenticatedUser) -> Select:
    return statement.where(Chatbot.owner_id == user.id)


async def get_chatbot(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> Chatbot:
    """Fetch one chatbot the caller owns, or raise 404.

    Returns 404 (not 403) for a chatbot belonging to someone else, so the
    endpoint does not confirm that the id exists.
    """
    result = await db.execute(
        _owned(select(Chatbot).where(Chatbot.id == chatbot_id), user)
    )
    chatbot = result.scalar_one_or_none()
    if chatbot is None:
        raise errors.not_found("chatbot")
    return chatbot


async def list_chatbots(
    db: AsyncSession, user: AuthenticatedUser
) -> list[Chatbot]:
    result = await db.execute(
        _owned(select(Chatbot), user).order_by(Chatbot.created_at.desc())
    )
    return list(result.scalars().all())


async def update_chatbot_name(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID, name: str
) -> Chatbot:
    chatbot = await get_chatbot(db, user, chatbot_id)
    chatbot.name = name.strip()[:120]
    await db.commit()
    await db.refresh(chatbot)
    return chatbot


async def delete_chatbot(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> None:
    # Ownership check first, then a delete that is itself owner-scoped.
    await get_chatbot(db, user, chatbot_id)
    await db.execute(
        delete(Chatbot).where(Chatbot.id == chatbot_id, Chatbot.owner_id == user.id)
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Knowledge-base statistics
# ---------------------------------------------------------------------------


async def get_stats_map(
    db: AsyncSession, user: AuthenticatedUser, chatbot_ids: list[uuid.UUID]
) -> dict[uuid.UUID, ChatbotStats]:
    """Aggregate page counts for several chatbots in one query."""
    if not chatbot_ids:
        return {}

    scraped = PageStatus.SCRAPED.value
    result = await db.execute(
        select(
            WebsitePage.chatbot_id,
            func.count().label("total"),
            func.count().filter(WebsitePage.status == scraped).label("successful"),
            func.count().filter(WebsitePage.status == PageStatus.FAILED.value).label("failed"),
            func.count().filter(WebsitePage.status == PageStatus.DUPLICATE.value).label("duplicate"),
            func.count()
            .filter(WebsitePage.status == scraped, WebsitePage.is_included.is_(True))
            .label("included"),
            func.coalesce(
                func.sum(WebsitePage.content_length).filter(WebsitePage.status == scraped), 0
            ).label("characters"),
            func.max(WebsitePage.scraped_at).label("last_scraped_at"),
        )
        .where(
            WebsitePage.chatbot_id.in_(chatbot_ids),
            WebsitePage.owner_id == user.id,
        )
        .group_by(WebsitePage.chatbot_id)
    )

    stats: dict[uuid.UUID, ChatbotStats] = {}
    for row in result:
        stats[row.chatbot_id] = ChatbotStats(
            total_pages=int(row.total or 0),
            successful_pages=int(row.successful or 0),
            failed_pages=int(row.failed or 0),
            duplicate_pages=int(row.duplicate or 0),
            included_pages=int(row.included or 0),
            total_content_characters=int(row.characters or 0),
            last_scraped_at=row.last_scraped_at,
        )
    return stats


async def get_stats(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> ChatbotStats:
    stats = await get_stats_map(db, user, [chatbot_id])
    return stats.get(chatbot_id, ChatbotStats())


# ---------------------------------------------------------------------------
# Website pages
# ---------------------------------------------------------------------------


async def list_pages(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> list[WebsitePage]:
    await get_chatbot(db, user, chatbot_id)
    result = await db.execute(
        select(WebsitePage)
        .where(WebsitePage.chatbot_id == chatbot_id, WebsitePage.owner_id == user.id)
        .order_by(WebsitePage.priority.desc(), WebsitePage.url.asc())
    )
    return list(result.scalars().all())


async def get_page(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    page_id: uuid.UUID,
) -> WebsitePage:
    await get_chatbot(db, user, chatbot_id)
    result = await db.execute(
        select(WebsitePage).where(
            WebsitePage.id == page_id,
            WebsitePage.chatbot_id == chatbot_id,
            WebsitePage.owner_id == user.id,
        )
    )
    page = result.scalar_one_or_none()
    if page is None:
        raise errors.not_found("page")
    return page


async def set_page_included(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    page_id: uuid.UUID,
    is_included: bool,
) -> WebsitePage:
    page = await get_page(db, user, chatbot_id, page_id)
    if is_included and page.status != PageStatus.SCRAPED.value:
        raise errors.AppError(
            "page_not_includable",
            "This page has no usable content, so it cannot be added to the knowledge base.",
            status_code=409,
        )
    page.is_included = is_included
    await db.commit()
    await db.refresh(page)
    return page


# ---------------------------------------------------------------------------
# Crawl job lookups
# ---------------------------------------------------------------------------


async def get_latest_crawl_job(
    db: AsyncSession, chatbot_id: uuid.UUID
) -> CrawlJob | None:
    result = await db.execute(
        select(CrawlJob)
        .where(CrawlJob.chatbot_id == chatbot_id)
        .order_by(CrawlJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_active_crawl_job(
    db: AsyncSession, chatbot_id: uuid.UUID
) -> CrawlJob | None:
    active = [
        CrawlStatus.PENDING.value,
        CrawlStatus.CRAWLING.value,
        CrawlStatus.PROCESSING.value,
    ]
    result = await db.execute(
        select(CrawlJob)
        .where(CrawlJob.chatbot_id == chatbot_id, CrawlJob.status.in_(active))
        .order_by(CrawlJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def set_chatbot_crawl_status(
    db: AsyncSession, chatbot_id: uuid.UUID, status: CrawlStatus
) -> None:
    values: dict[str, object] = {
        "crawl_status": status.value,
        "updated_at": datetime.now(timezone.utc),
    }
    await db.execute(update(Chatbot).where(Chatbot.id == chatbot_id).values(**values))

    # A chatbot becomes "ready" as soon as it has usable content; a partial
    # crawl still produces a usable knowledge base. Only from "draft" though:
    # re-scraping a published chatbot must not quietly take it offline, and
    # re-scraping an unpublished one must not quietly mark it ready again.
    if status in (CrawlStatus.COMPLETED, CrawlStatus.PARTIALLY_COMPLETED):
        await db.execute(
            update(Chatbot)
            .where(
                Chatbot.id == chatbot_id,
                Chatbot.status == ChatbotStatus.DRAFT.value,
            )
            .values(status=ChatbotStatus.READY.value)
        )
    await db.commit()
