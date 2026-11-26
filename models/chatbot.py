"""Chatbot: ORM model + request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator
from sqlalchemy import DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import ApiSchema, Base, ChatbotStatus, CrawlStatus


class Chatbot(Base):
    __tablename__ = "chatbots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # Unguessable public handle. Phase 4 serves the iframe at
    # https://chat.skandaai.in/embed/{public_id} — not implemented in Phase 1.
    public_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # Phase 3 serves the iframe at {PUBLIC_WIDGET_BASE_URL}/embed/{public_id}.
    # The handle is permanent: publishing, unpublishing and republishing never
    # change it, so an installed snippet keeps working.
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    website_url: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=ChatbotStatus.DRAFT.value
    )
    crawl_status: Mapped[str] = mapped_column(
        String, nullable=False, default=CrawlStatus.PENDING.value
    )
    draft_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    published_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # ---- Phase 3: public deployment (003_deployment.sql) ----
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    unpublished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Embedding origins the customer authorised. Empty = unrestricted.
    allowed_domains: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # Launcher appearance and chat-window geometry. Phase 4 writes it.
    deployment_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    site_title: Mapped[str | None] = mapped_column(String, nullable=True)
    brand_color: Mapped[str | None] = mapped_column(String, nullable=True)
    favicon_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    pages = relationship(
        "WebsitePage", back_populates="chatbot", cascade="all, delete-orphan",
        passive_deletes=True,
    )
    crawl_jobs = relationship(
        "CrawlJob", back_populates="chatbot", cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Chatbot id={self.id} public_id={self.public_id!r}>"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatbotCreate(ApiSchema):
    """The customer submits only a URL. Everything else is derived."""

    website_url: str = Field(min_length=4, max_length=2048)
    # Optional override; normally derived from the site's <title> or domain.
    name: str | None = Field(default=None, max_length=120)

    @field_validator("website_url")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()

    @field_validator("name")
    @classmethod
    def _trim_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class ChatbotUpdate(ApiSchema):
    name: str | None = Field(default=None, min_length=1, max_length=120)


class ChatbotStats(ApiSchema):
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    duplicate_pages: int = 0
    included_pages: int = 0
    total_content_characters: int = 0
    last_scraped_at: datetime | None = None


class ChatbotOut(ApiSchema):
    id: uuid.UUID
    public_id: str
    name: str
    website_url: str
    status: ChatbotStatus
    crawl_status: CrawlStatus
    site_title: str | None = None
    brand_color: str | None = None
    favicon_url: str | None = None
    created_at: datetime
    updated_at: datetime
    stats: ChatbotStats = Field(default_factory=ChatbotStats)
    # Phase 3: the chatbot's permanent public iframe URL. Present on the
    # detail endpoint. Live once `status` is "published".
    embed_url_preview: str | None = None
    published_at: datetime | None = None
    unpublished_at: datetime | None = None
    is_published: bool = False


class ChatbotListOut(ApiSchema):
    chatbots: list[ChatbotOut]
    total: int
