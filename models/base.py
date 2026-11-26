"""Declarative base, shared enums and the standard API envelopes."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models.

    The authoritative schema is `migrations/001_init.sql`; these models mirror
    it. We never call `create_all()` in application code.
    """


# ---------------------------------------------------------------------------
# Enums — values must match the CHECK constraints in 001_init.sql
# ---------------------------------------------------------------------------


class ChatbotStatus(StrEnum):
    """The chatbot lifecycle. Values must match `chatbots_status_check`.

    DRAFT       -> created, no usable knowledge base yet
    READY       -> has scraped content, testable in the playground
    PUBLISHED   -> serving public conversations at its embed URL
    UNPUBLISHED -> was published and has been switched off; the public_id,
                   configuration and knowledge base are all preserved, so
                   republishing restores the same URL.
    """

    DRAFT = "draft"
    READY = "ready"
    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"

    @property
    def is_public(self) -> bool:
        return self is ChatbotStatus.PUBLISHED


class CrawlStatus(StrEnum):
    PENDING = "pending"
    CRAWLING = "crawling"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            CrawlStatus.COMPLETED,
            CrawlStatus.PARTIALLY_COMPLETED,
            CrawlStatus.FAILED,
        }


class PageStatus(StrEnum):
    SCRAPED = "scraped"
    DUPLICATE = "duplicate"
    EMPTY = "empty"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# API envelopes
# ---------------------------------------------------------------------------


class ApiSchema(BaseModel):
    """Base for every response schema: reads straight off ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns.

    The frontend only ever renders `error.message`, so no raw exception or
    stack trace can reach the browser.
    """

    error: ErrorBody
