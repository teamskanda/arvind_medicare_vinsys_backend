"""Crawl job: ORM model + schemas.

The job row is the durable record of crawl progress. The frontend polls
`GET /api/chatbots/{id}/crawl/status`, which reads this row, so refreshing the
page (or restarting the backend) never loses progress.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import ApiSchema, Base, CrawlStatus


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String, nullable=False, default="firecrawl")
    provider_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=CrawlStatus.PENDING.value
    )
    progress_message: Mapped[str | None] = mapped_column(String, nullable=True)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    credits_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    chatbot = relationship("Chatbot", back_populates="crawl_jobs")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CrawlJob id={self.id} status={self.status}>"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CrawlJobOut(ApiSchema):
    id: uuid.UUID
    chatbot_id: uuid.UUID
    status: CrawlStatus
    progress_message: str | None
    total_pages: int
    successful_pages: int
    failed_pages: int
    duplicate_pages: int
    credits_used: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class CrawlStatusOut(ApiSchema):
    """Everything the progress screen renders. All values are real — the
    backend never fabricates a percentage."""

    chatbot_id: uuid.UUID
    website_url: str
    crawl_status: CrawlStatus
    progress_message: str
    discovered_pages: int
    successful_pages: int
    failed_pages: int
    duplicate_pages: int
    stored_pages: int
    total_content_characters: int
    is_terminal: bool
    error_message: str | None = None
    job: CrawlJobOut | None = None
