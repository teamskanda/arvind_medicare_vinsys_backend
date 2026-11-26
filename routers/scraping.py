"""Crawl control endpoints: start, progress, retry.

The progress endpoint reports only values read from the database. There is no
simulated percentage anywhere in this file.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from helpers import chatbots as chatbot_helpers
from helpers import crawling
from helpers.auth import CurrentUser
from helpers.database import get_db
from models.base import CrawlStatus
from models.crawl_job import CrawlJobOut, CrawlStatusOut

router = APIRouter(prefix="/api/chatbots", tags=["scraping"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.post("/{chatbot_id}/crawl", response_model=CrawlJobOut, status_code=status.HTTP_202_ACCEPTED)
async def start_crawl(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> CrawlJobOut:
    """Start a crawl for an existing chatbot.

    Creation already starts one automatically; this exists for a manual
    re-crawl. Returns 409 if a crawl is already running.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    job = await crawling.start_crawl(db, user, chatbot)
    return CrawlJobOut.model_validate(job)


@router.post(
    "/{chatbot_id}/crawl/retry",
    response_model=CrawlJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_crawl(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> CrawlJobOut:
    """Retry after a failed or partial crawl.

    Pages already stored are kept; re-read pages are refreshed in place.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    job = await crawling.retry_crawl(db, user, chatbot)
    return CrawlJobOut.model_validate(job)


@router.get("/{chatbot_id}/crawl/status", response_model=CrawlStatusOut)
async def get_crawl_status(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> CrawlStatusOut:
    """Report real crawl progress.

    Reads the persisted job row, so a page refresh or a backend restart does
    not lose progress. If the job looks stale (no poller touching it), this
    call also advances it before responding.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    job = await chatbot_helpers.get_latest_crawl_job(db, chatbot_id)

    if job is not None:
        job = await crawling.resync_if_stale(db, job)
        # The chatbot row may have been updated by the resync.
        await db.refresh(chatbot)

    stats = await chatbot_helpers.get_stats(db, user, chatbot_id)
    crawl_status = CrawlStatus(chatbot.crawl_status)

    error_message: str | None = None
    if job is not None and isinstance(job.error_details, dict):
        raw = job.error_details.get("message")
        if isinstance(raw, str):
            error_message = raw

    if job is None:
        progress_message = "No crawl has been started yet."
        discovered = 0
        failed = 0
    else:
        progress_message = job.progress_message or crawling.PROGRESS_MESSAGES[crawl_status]
        discovered = max(job.total_pages, stats.successful_pages)
        failed = job.failed_pages

    return CrawlStatusOut(
        chatbot_id=chatbot.id,
        website_url=chatbot.website_url,
        crawl_status=crawl_status,
        progress_message=progress_message,
        discovered_pages=discovered,
        successful_pages=stats.successful_pages,
        failed_pages=failed,
        duplicate_pages=stats.duplicate_pages,
        stored_pages=stats.total_pages,
        total_content_characters=stats.total_content_characters,
        is_terminal=crawl_status.is_terminal,
        error_message=error_message,
        job=CrawlJobOut.model_validate(job) if job is not None else None,
    )
