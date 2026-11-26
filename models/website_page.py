"""Website page: ORM model + schemas.

One row per scraped page. Content is deliberately kept per-page and never
concatenated, so Phase 2 keyword retrieval can select only what a visitor's
question needs (Gemma's ~10k-token context makes whole-site stuffing
impossible for anything but tiny sites).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import ApiSchema, Base, PageStatus


class WebsitePage(Base):
    __tablename__ = "website_pages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised so ownership can be checked (and RLS applied) without a join.
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    content_length: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    is_included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=PageStatus.SCRAPED.value
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    scraped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chatbot = relationship("Chatbot", back_populates="pages")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<WebsitePage id={self.id} url={self.url!r} status={self.status}>"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WebsitePageSummary(ApiSchema):
    """List view — deliberately excludes `content` to keep responses small."""

    id: uuid.UUID
    url: str
    title: str | None
    content_length: int
    language: str | None
    is_included: bool
    status: PageStatus
    priority: int
    error: str | None = None
    scraped_at: datetime | None


class WebsitePageDetail(WebsitePageSummary):
    """Detail view — includes the extracted Markdown."""

    content: str


class WebsitePageListOut(ApiSchema):
    pages: list[WebsitePageSummary]
    total: int


class WebsitePageUpdate(ApiSchema):
    is_included: bool
