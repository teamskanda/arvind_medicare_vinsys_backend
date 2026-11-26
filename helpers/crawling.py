"""Crawl orchestration: start, poll, ingest, finalise.

Progress durability
-------------------
The `crawl_jobs` row is the single source of truth for progress. The frontend
polls an endpoint that reads that row, so a page refresh never loses state.

Two things can advance a job:
  * the background poller started when the crawl begins;
  * a *resync* triggered by the status endpoint when the job looks stale
    (e.g. the backend restarted and the poller died with it).

Both paths are safe to run concurrently because ingestion is idempotent
(upsert keyed on `(chatbot_id, url)`) and every counter is recomputed from
the database rather than incremented.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import chatbots as chatbot_helpers
from helpers import errors, firecrawl
from helpers.content_processing import ProcessingResult, process_documents
from helpers.database import session_scope
from models.base import CrawlStatus, PageStatus
from models.chatbot import Chatbot
from models.crawl_job import CrawlJob
from models.user import AuthenticatedUser
from models.website_page import WebsitePage

logger = logging.getLogger(__name__)

# Strong references to running pollers so the event loop cannot garbage
# collect a task mid-crawl.
_background_tasks: set[asyncio.Task[None]] = set()

PROGRESS_MESSAGES = {
    CrawlStatus.PENDING: "Validating website...",
    CrawlStatus.CRAWLING: "Discovering website pages...",
    CrawlStatus.PROCESSING: "Processing content...",
    CrawlStatus.COMPLETED: "Knowledge base ready.",
    CrawlStatus.PARTIALLY_COMPLETED: "Knowledge base ready with some pages skipped.",
    CrawlStatus.FAILED: "Scraping failed.",
}


# ---------------------------------------------------------------------------
# Starting a crawl
# ---------------------------------------------------------------------------


async def start_crawl(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot
) -> CrawlJob:
    """Begin a crawl for a chatbot the caller owns."""
    active = await chatbot_helpers.get_active_crawl_job(db, chatbot.id)
    if active is not None:
        raise errors.crawl_already_running()

    provider_job_id = await firecrawl.start_crawl(chatbot.website_url)

    job = CrawlJob(
        chatbot_id=chatbot.id,
        owner_id=user.id,
        provider="firecrawl",
        provider_job_id=provider_job_id,
        status=CrawlStatus.CRAWLING.value,
        progress_message=PROGRESS_MESSAGES[CrawlStatus.CRAWLING],
        last_synced_at=datetime.now(timezone.utc),
    )
    db.add(job)
    try:
        await db.commit()
    except IntegrityError:
        # The partial unique index caught a race: another request started a
        # crawl between our check and this insert.
        await db.rollback()
        raise errors.crawl_already_running()
    await db.refresh(job)

    await chatbot_helpers.set_chatbot_crawl_status(db, chatbot.id, CrawlStatus.CRAWLING)
    _spawn_poller(chatbot.id, job.id)
    return job


def _spawn_poller(chatbot_id: uuid.UUID, job_id: uuid.UUID) -> None:
    task = asyncio.create_task(_poll_until_done(chatbot_id, job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _poll_until_done(chatbot_id: uuid.UUID, job_id: uuid.UUID) -> None:
    """Background poller. Owns its own session: it outlives the request."""
    settings = get_settings()
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=settings.crawl_max_duration_seconds
    )

    while True:
        try:
            async with session_scope() as db:
                job = await _load_job(db, job_id)
                if job is None or CrawlStatus(job.status).is_terminal:
                    return

                timed_out = datetime.now(timezone.utc) >= deadline
                finished = await sync_job(db, job, force_finalise=timed_out)
                if finished:
                    return
        except errors.AppError as exc:
            logger.warning("Crawl %s provider error: %s", job_id, exc.message)
            if not exc.retryable or datetime.now(timezone.utc) >= deadline:
                await _fail_job(job_id, chatbot_id, exc.message)
                return
        except asyncio.CancelledError:
            logger.info("Crawl poller %s cancelled (server shutting down).", job_id)
            raise
        except Exception:
            logger.exception("Unexpected error while polling crawl %s", job_id)
            await _fail_job(
                job_id,
                chatbot_id,
                "Scraping stopped unexpectedly. You can retry the crawl.",
            )
            return

        await asyncio.sleep(settings.crawl_poll_interval_seconds)


# ---------------------------------------------------------------------------
# One synchronisation pass
# ---------------------------------------------------------------------------


async def _load_job(db: AsyncSession, job_id: uuid.UUID) -> CrawlJob | None:
    result = await db.execute(select(CrawlJob).where(CrawlJob.id == job_id))
    return result.scalar_one_or_none()


async def sync_job(
    db: AsyncSession, job: CrawlJob, *, force_finalise: bool = False
) -> bool:
    """Poll the provider once, ingest what is available, update the job row.

    Returns True when the job reached a terminal state.
    """
    if not job.provider_job_id:
        await _mark_failed(
            db, job, "The scraping job reference was lost. Please retry the crawl."
        )
        return True

    snapshot = await firecrawl.get_crawl_snapshot(job.provider_job_id)
    terminal = snapshot.is_terminal or force_finalise

    if terminal:
        # Be honest about the phase: ingestion of 20 pages is quick but not
        # instant, and the UI should not claim completion before storing.
        await _set_status(db, job, CrawlStatus.PROCESSING)

    processed = process_documents(snapshot.documents)
    await _store_pages(db, job, processed)

    if not terminal:
        await _update_progress(db, job, snapshot, processed)
        return False

    await _finalise(db, job, snapshot, timed_out=force_finalise and not snapshot.is_terminal)
    return True


def build_page_rows(
    chatbot_id: uuid.UUID,
    owner_id: uuid.UUID,
    processed: ProcessingResult,
    now: datetime,
) -> list[dict[str, Any]]:
    """Map processed pages onto `website_pages` column values."""
    return [
        {
            "chatbot_id": chatbot_id,
            "owner_id": owner_id,
            "url": page.url,
            "title": page.title,
            "content": page.content,
            "content_hash": page.content_hash,
            "content_length": page.content_length,
            "language": page.language,
            # Duplicate and empty pages are stored for transparency but kept
            # out of the AI knowledge base.
            "is_included": page.status == PageStatus.SCRAPED.value,
            "status": page.status,
            "priority": page.priority,
            "error": page.error,
            "scraped_at": now,
        }
        for page in processed.pages
    ]


def build_page_upsert(rows: list[dict[str, Any]]):
    """INSERT ... ON CONFLICT for idempotent crawl-result ingestion.

    Keyed on the `(chatbot_id, url)` unique constraint, so re-processing the
    same Firecrawl payload refreshes rows instead of duplicating them.
    """
    statement = pg_insert(WebsitePage).values(rows)
    return statement.on_conflict_do_update(
        constraint="website_pages_chatbot_url_key",
        set_={
            "title": statement.excluded.title,
            "content": statement.excluded.content,
            "content_hash": statement.excluded.content_hash,
            "content_length": statement.excluded.content_length,
            "language": statement.excluded.language,
            "status": statement.excluded.status,
            "priority": statement.excluded.priority,
            "error": statement.excluded.error,
            "scraped_at": statement.excluded.scraped_at,
            # `is_included` is intentionally NOT overwritten: a customer's
            # decision to exclude a page must survive a re-crawl.
        },
    )


def build_page_counts_query(chatbot_id: uuid.UUID):
    """Per-status page counters for one chatbot, in a single round trip."""
    return select(
        func.count().filter(WebsitePage.status == PageStatus.SCRAPED.value).label("scraped"),
        func.count().filter(WebsitePage.status == PageStatus.DUPLICATE.value).label("duplicate"),
        func.count().filter(WebsitePage.status == PageStatus.EMPTY.value).label("empty"),
        func.coalesce(
            func.sum(WebsitePage.content_length).filter(
                WebsitePage.status == PageStatus.SCRAPED.value
            ),
            0,
        ).label("characters"),
    ).where(WebsitePage.chatbot_id == chatbot_id)


async def _store_pages(
    db: AsyncSession, job: CrawlJob, processed: ProcessingResult
) -> None:
    """Upsert pages. Safe to call repeatedly with overlapping data."""
    if not processed.pages:
        return

    rows = build_page_rows(
        job.chatbot_id, job.owner_id, processed, datetime.now(timezone.utc)
    )
    await db.execute(build_page_upsert(rows))
    await db.commit()


async def _page_counts(db: AsyncSession, chatbot_id: uuid.UUID) -> dict[str, int]:
    """Counters derived from stored rows, never incremented in memory."""
    result = await db.execute(build_page_counts_query(chatbot_id))
    row = result.one()
    return {
        "scraped": int(row.scraped or 0),
        "duplicate": int(row.duplicate or 0),
        "empty": int(row.empty or 0),
        "characters": int(row.characters or 0),
    }


async def _update_progress(
    db: AsyncSession,
    job: CrawlJob,
    snapshot: firecrawl.CrawlSnapshot,
    processed: ProcessingResult,
) -> None:
    counts = await _page_counts(db, job.chatbot_id)
    discovered = max(snapshot.total, snapshot.completed, counts["scraped"])

    if counts["scraped"] == 0:
        message = "Discovering website pages..."
    else:
        message = f"Extracting website content... {counts['scraped']} of {discovered} pages"

    await db.execute(
        update(CrawlJob)
        .where(CrawlJob.id == job.id)
        .values(
            status=CrawlStatus.CRAWLING.value,
            progress_message=message,
            total_pages=discovered,
            successful_pages=counts["scraped"],
            duplicate_pages=counts["duplicate"],
            credits_used=snapshot.credits_used,
            last_synced_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()


async def _set_status(db: AsyncSession, job: CrawlJob, status: CrawlStatus) -> None:
    await db.execute(
        update(CrawlJob)
        .where(CrawlJob.id == job.id)
        .values(
            status=status.value,
            progress_message=PROGRESS_MESSAGES[status],
            last_synced_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    await chatbot_helpers.set_chatbot_crawl_status(db, job.chatbot_id, status)


async def _finalise(
    db: AsyncSession,
    job: CrawlJob,
    snapshot: firecrawl.CrawlSnapshot,
    *,
    timed_out: bool,
) -> None:
    """Decide the final status from what actually landed in the database."""
    report = await firecrawl.get_crawl_errors(job.provider_job_id or "")
    counts = await _page_counts(db, job.chatbot_id)

    provider_failures = len(report.failed_urls)
    # From the customer's point of view a page that yielded no text is as
    # unusable as one that errored, so they share the "failed" counter. The
    # knowledge-base table still shows the precise per-page status.
    failed = provider_failures + counts["empty"]

    details: dict[str, Any] = {}
    if report.robots_blocked:
        details["robots_blocked"] = report.robots_blocked[:20]
    if report.failed_urls:
        details["failed_urls"] = [
            {"url": item.get("url"), "error": item.get("error")}
            for item in report.failed_urls[:20]
        ]
    if snapshot.raw_error:
        details["provider_error"] = snapshot.raw_error
    if timed_out:
        details["timed_out"] = True

    error_message: str | None = None

    if counts["scraped"] > 0:
        if failed > 0 or timed_out or snapshot.provider_status in {"failed", "cancelled"}:
            status = CrawlStatus.PARTIALLY_COMPLETED
            if timed_out:
                error_message = (
                    "Scraping took longer than expected, so we saved the pages that "
                    "finished. You can retry to pick up the rest."
                )
            elif report.robots_blocked:
                error_message = (
                    f"{counts['scraped']} pages were saved. "
                    f"{len(report.robots_blocked)} page(s) were skipped because the "
                    "website's robots.txt disallows crawling them."
                )
            else:
                error_message = (
                    f"{counts['scraped']} pages were saved and {failed} could not be read."
                )
        else:
            status = CrawlStatus.COMPLETED
    else:
        status = CrawlStatus.FAILED
        if report.robots_blocked:
            error_message = (
                "This website's robots.txt blocks automated crawling, so no pages "
                "could be read."
            )
        elif timed_out:
            error_message = (
                "Scraping timed out before any page could be read. The website may be "
                "very slow. Please try again."
            )
        elif snapshot.raw_error:
            error_message = (
                "The website could not be scraped. It may block automated access."
            )
        else:
            error_message = (
                "No readable content was found on this website. It may rely on "
                "features our scraper cannot read, or block automated access."
            )

    message = PROGRESS_MESSAGES[status]
    if status == CrawlStatus.COMPLETED:
        message = f"Knowledge base ready. {counts['scraped']} pages saved."

    await db.execute(
        update(CrawlJob)
        .where(CrawlJob.id == job.id)
        .values(
            status=status.value,
            progress_message=message,
            total_pages=max(snapshot.total, counts["scraped"] + counts["duplicate"]),
            successful_pages=counts["scraped"],
            duplicate_pages=counts["duplicate"],
            failed_pages=failed,
            credits_used=snapshot.credits_used,
            error_details=({**details, "message": error_message} if error_message or details else None),
            last_synced_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    await chatbot_helpers.set_chatbot_crawl_status(db, job.chatbot_id, status)

    logger.info(
        "Crawl %s finished: status=%s scraped=%s failed=%s duplicates=%s",
        job.id,
        status.value,
        counts["scraped"],
        failed,
        counts["duplicate"],
    )


async def _mark_failed(db: AsyncSession, job: CrawlJob, message: str) -> None:
    counts = await _page_counts(db, job.chatbot_id)
    status = CrawlStatus.PARTIALLY_COMPLETED if counts['scraped'] > 0 else CrawlStatus.FAILED
    if counts['scraped']:
        message = f"{counts['scraped']} pages were saved before scraping stopped. {message} You can retry for the remaining pages."
    await db.execute(
        update(CrawlJob)
        .where(CrawlJob.id == job.id)
        .values(
            status=status.value,
            successful_pages=counts['scraped'],
            duplicate_pages=counts['duplicate'],
            progress_message=PROGRESS_MESSAGES[status],
            error_details={"message": message},
            last_synced_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    await chatbot_helpers.set_chatbot_crawl_status(db, job.chatbot_id, status)


async def _fail_job(job_id: uuid.UUID, chatbot_id: uuid.UUID, message: str) -> None:
    """Mark a job failed from the background poller, on its own session."""
    try:
        async with session_scope() as db:
            job = await _load_job(db, job_id)
            if job is not None and not CrawlStatus(job.status).is_terminal:
                await _mark_failed(db, job, message)
    except Exception:
        logger.exception("Could not mark crawl %s as failed", job_id)


# ---------------------------------------------------------------------------
# Resync driven by the status endpoint
# ---------------------------------------------------------------------------


async def resync_if_stale(db: AsyncSession, job: CrawlJob) -> CrawlJob:
    """Advance a job that no live poller appears to be touching.

    This is what makes progress survive a backend restart. The UPDATE acts as
    a claim: only the caller that successfully moves `last_synced_at` does the
    work, so two concurrent status requests cannot both poll the provider.
    """
    if CrawlStatus(job.status).is_terminal:
        return job

    settings = get_settings()
    claimed = await db.execute(
        text(
            """
            UPDATE crawl_jobs
               SET last_synced_at = now()
             WHERE id = :job_id
               AND status IN ('pending', 'crawling', 'processing')
               AND (last_synced_at IS NULL
                    OR last_synced_at < now() - make_interval(secs => :stale_seconds))
            RETURNING id
            """
        ),
        {"job_id": job.id, "stale_seconds": settings.crawl_resync_stale_after_seconds},
    )
    await db.commit()

    if claimed.first() is None:
        return job  # someone else is already syncing

    try:
        timed_out = datetime.now(timezone.utc) >= job.created_at + timedelta(seconds=settings.crawl_max_duration_seconds)
        await sync_job(db, job, force_finalise=timed_out)
    except errors.AppError as exc:
        logger.warning("Resync of crawl %s failed: %s", job.id, exc.message)
        if not exc.retryable or timed_out:
            await _mark_failed(db, job, exc.message)
    except Exception:
        logger.exception("Unexpected error resyncing crawl %s", job.id)

    refreshed = await _load_job(db, job.id)
    return refreshed or job


async def record_failed_start(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot, message: str
) -> CrawlJob:
    """Persist a crawl that could not even be started.

    Used when chatbot creation succeeds but Firecrawl rejects the job (no
    credits, provider outage). The customer still gets their chatbot, the
    progress screen explains what happened, and Retry is available — rather
    than the whole create request failing and losing the record.
    """
    job = CrawlJob(
        chatbot_id=chatbot.id,
        owner_id=user.id,
        provider="firecrawl",
        provider_job_id=None,
        status=CrawlStatus.FAILED.value,
        progress_message=PROGRESS_MESSAGES[CrawlStatus.FAILED],
        error_details={"message": message},
        last_synced_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    await chatbot_helpers.set_chatbot_crawl_status(db, chatbot.id, CrawlStatus.FAILED)
    return job


async def retry_crawl(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot
) -> CrawlJob:
    """Start a fresh crawl after a failed or partial one.

    Existing pages are left in place: a retry should never destroy a working
    knowledge base, and the upsert refreshes rows it re-reads.
    """
    active = await chatbot_helpers.get_active_crawl_job(db, chatbot.id)
    if active is not None:
        raise errors.crawl_already_running()
    return await start_crawl(db, user, chatbot)
