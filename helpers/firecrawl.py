"""Firecrawl v2 API client.

Verified against the v2 reference (POST /v2/crawl, GET /v2/crawl/{id},
GET /v2/crawl/{id}/errors) on 2026-11-26.

Design notes:
  * The API key never leaves this process; the frontend has no Firecrawl
    credentials at all.
  * We never set `ignoreRobotsTxt`. Firecrawl respects robots.txt by default
    and we keep it that way.
  * Provider status strings are translated into our own `CrawlStatus` by the
    caller, so provider vocabulary does not leak into the database or UI.
  * Every provider failure is mapped to a customer-readable `AppError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import get_settings
from helpers import errors
from helpers.content_processing import EXCLUDE_PATH_PATTERNS

logger = logging.getLogger(__name__)

# Provider statuses we may see. The two reference pages disagree on whether
# `cancelled` is included, so we handle it defensively.
TERMINAL_PROVIDER_STATUSES = {"completed", "failed", "cancelled"}

# `next` paginates in ~10MB chunks. Twenty cleaned marketing pages fit in one,
# so this cap is pure protection against a runaway loop.
MAX_RESULT_CHUNKS = 6


@dataclass(slots=True)
class CrawlSnapshot:
    """One read of a crawl job's state, with all available documents."""

    provider_status: str
    total: int = 0
    completed: int = 0
    credits_used: int = 0
    documents: list[dict[str, Any]] = field(default_factory=list)
    raw_error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.provider_status in TERMINAL_PROVIDER_STATUSES


@dataclass(slots=True)
class CrawlErrorReport:
    failed_urls: list[dict[str, Any]] = field(default_factory=list)
    robots_blocked: list[str] = field(default_factory=list)


def _headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.firecrawl_api_key:
        raise errors.AppError(
            "scraping_not_configured", "Website scraping is not configured on this server.",
            status_code=503,
        )
    return {
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
        "Content-Type": "application/json",
    }


def _client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.firecrawl_timeout_seconds, connect=15.0),
        headers=_headers(),
    )


def _raise_for_provider_status(response: httpx.Response) -> None:
    """Map Firecrawl HTTP failures onto our error vocabulary."""
    if response.status_code < 400:
        return

    detail: str | None = None
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("error") or body.get("message")
    except Exception:
        detail = None

    logger.warning(
        "Firecrawl responded %s: %s", response.status_code, detail or response.text[:300]
    )

    if response.status_code == 401 or response.status_code == 403:
        raise errors.provider_error(
            "The website scraping service rejected our credentials. "
            "Please check the server configuration.",
            code="provider_unauthorised",
            retryable=False,
        )
    if response.status_code == 402:
        raise errors.provider_error(
            "The website scraping service has run out of credits. "
            "Please top up the Firecrawl account and try again.",
            code="provider_out_of_credits",
            retryable=False,
            status_code=402,
        )
    if response.status_code == 429:
        raise errors.rate_limited()
    if response.status_code == 404:
        raise errors.provider_error(
            "That crawl job is no longer available. Firecrawl keeps results for 24 hours.",
            code="provider_job_expired",
            retryable=False,
            status_code=404,
        )
    if 400 <= response.status_code < 500:
        raise errors.provider_error(
            detail_to_message(detail),
            code="provider_rejected_request",
            retryable=False,
            status_code=422,
        )
    raise errors.provider_error(
        "The website scraping service is temporarily unavailable. Please try again."
    )


def detail_to_message(detail: str | None) -> str:
    if detail and len(detail) < 200:
        return f"The website scraping service could not start this crawl: {detail}"
    return "The website scraping service could not start this crawl."


def build_crawl_payload(url: str, *, limit: int, max_depth: int) -> dict[str, Any]:
    """Crawl options tuned for a 20-page business-website knowledge base."""
    return {
        "url": url,
        "limit": limit,
        # Use the sitemap when there is one, and fall back to link discovery.
        "sitemap": "include",
        # Business sites keep /about and /contact as siblings of the start
        # path, so restricting to child paths would miss the most useful pages.
        "crawlEntireDomain": True,
        "allowSubdomains": False,
        "allowExternalLinks": False,
        "ignoreQueryParameters": True,
        "maxDiscoveryDepth": max_depth,
        "excludePaths": EXCLUDE_PATH_PATTERNS,
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "excludeTags": ["nav", "footer", "header", "aside", "form", "script", "style"],
            "timeout": 30000,
            "blockAds": True,
        },
    }


async def start_crawl(url: str, *, limit: int | None = None) -> str:
    """Start a crawl. Returns the provider job id."""
    settings = get_settings()
    payload = build_crawl_payload(
        url,
        limit=limit or settings.crawl_page_limit,
        max_depth=settings.crawl_max_discovery_depth,
    )

    try:
        async with _client() as client:
            response = await client.post(f"{settings.firecrawl_base_url}/crawl", json=payload)
    except httpx.TimeoutException:
        raise errors.provider_error(
            "The website scraping service did not respond in time. Please try again."
        )
    except httpx.HTTPError as exc:
        logger.warning("Firecrawl connection error: %s", exc)
        raise errors.provider_error(
            "We could not reach the website scraping service. Please try again."
        )

    _raise_for_provider_status(response)

    try:
        body = response.json()
    except ValueError:
        raise errors.provider_error("The website scraping service returned an unreadable response.")

    job_id = body.get("id") if isinstance(body, dict) else None
    if not job_id or not isinstance(job_id, str):
        logger.error("Firecrawl start response had no job id: %s", str(body)[:300])
        raise errors.provider_error("The website scraping service did not return a job reference.")
    return job_id


async def get_crawl_snapshot(job_id: str) -> CrawlSnapshot:
    """Read job state and collect every document available so far.

    Firecrawl returns the first result chunk on the status call and a `next`
    URL for subsequent chunks. `next` is present even mid-crawl and persists
    on failed jobs, so the loop exits on an empty chunk rather than on `next`
    being absent.
    """
    settings = get_settings()
    base = settings.firecrawl_base_url
    url: str | None = f"{base}/crawl/{job_id}"

    snapshot: CrawlSnapshot | None = None
    documents: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    try:
        async with _client() as client:
            for _ in range(MAX_RESULT_CHUNKS):
                if url is None:
                    break
                response = await client.get(url)
                _raise_for_provider_status(response)

                try:
                    body = response.json()
                except ValueError:
                    raise errors.provider_error(
                        "The website scraping service returned an unreadable response."
                    )
                if not isinstance(body, dict):
                    raise errors.provider_error(
                        "The website scraping service returned an unexpected response."
                    )

                if snapshot is None:
                    snapshot = CrawlSnapshot(
                        provider_status=str(body.get("status") or "scraping"),
                        total=_as_int(body.get("total")),
                        completed=_as_int(body.get("completed")),
                        credits_used=_as_int(body.get("creditsUsed")),
                        raw_error=_as_optional_str(body.get("error")),
                    )

                chunk = body.get("data")
                chunk_documents = [d for d in chunk if isinstance(d, dict)] if isinstance(chunk, list) else []
                added = 0
                for document in chunk_documents:
                    metadata = document.get("metadata") or {}
                    key = (
                        (metadata.get("sourceURL") or metadata.get("url") or "")
                        if isinstance(metadata, dict)
                        else ""
                    )
                    if key and key in seen_urls:
                        continue
                    if key:
                        seen_urls.add(key)
                    documents.append(document)
                    added += 1

                if added == 0:
                    break

                next_url = body.get("next")
                # Only follow a `next` that stays on the configured API host.
                if isinstance(next_url, str) and next_url.startswith(base):
                    url = next_url
                else:
                    url = None
    except httpx.TimeoutException:
        raise errors.provider_error(
            "The website scraping service did not respond in time. Please try again."
        )
    except httpx.HTTPError as exc:
        logger.warning("Firecrawl status connection error: %s", exc)
        raise errors.provider_error(
            "We could not reach the website scraping service. Please try again."
        )

    if snapshot is None:
        raise errors.provider_error("The website scraping service returned no job status.")

    snapshot.documents = documents
    return snapshot


async def get_crawl_errors(job_id: str) -> CrawlErrorReport:
    """Fetch per-URL failures. Best effort: never raises."""
    settings = get_settings()
    report = CrawlErrorReport()
    try:
        async with _client() as client:
            response = await client.get(f"{settings.firecrawl_base_url}/crawl/{job_id}/errors")
            if response.status_code >= 400:
                return report
            body = response.json()
    except Exception as exc:
        logger.info("Could not read crawl errors for %s: %s", job_id, exc)
        return report

    if not isinstance(body, dict):
        return report

    raw_errors = body.get("errors")
    if isinstance(raw_errors, list):
        report.failed_urls = [e for e in raw_errors if isinstance(e, dict)][:100]

    blocked = body.get("robotsBlocked")
    if isinstance(blocked, list):
        report.robots_blocked = [b for b in blocked if isinstance(b, str)][:100]

    return report


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def capture_screenshot(url: str) -> str:
    """One viewport screenshot, never screenshot formats on a whole crawl.

    Firecrawl v2 returns an expiring image URL. The caller immediately validates,
    downloads and persists the image bytes; this URL is never saved or exposed.
    """
    async with _client() as client:
        response = await client.post(f"{get_settings().firecrawl_base_url}/scrape", json={
            'url': url, 'formats': [{'type': 'screenshot', 'fullPage': False,
                'viewport': {'width': 1440, 'height': 1000}}],
            'onlyMainContent': False, 'timeout': 30000, 'maxAge': 172800000,
        })
    _raise_for_provider_status(response)
    body = response.json()
    image = (body.get('data') or {}).get('screenshot') if isinstance(body, dict) else None
    if not isinstance(image, str) or not image.startswith('https://'):
        raise ValueError('No usable screenshot returned')
    return image


def _as_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()[:500]
    return None
