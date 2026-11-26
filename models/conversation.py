"""Conversation: ORM model + schemas.

A conversation is one chat thread between a visitor (in Phase 2, the customer
testing in the dashboard playground) and one chatbot. `owner_id` is stored on
the row so every read can be ownership-filtered without a join.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator
from sqlalchemy import DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import ApiSchema, Base


class ConversationSource(StrEnum):
    """Where the conversation came from. Must match the CHECK in 002_chat.sql."""

    PLAYGROUND = "playground"
    EMBED = "embed"


class Conversation(Base):
    __tablename__ = "conversations"

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
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(
        String, nullable=False, default=ConversationSource.PLAYGROUND.value
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Conversation id={self.id} chatbot_id={self.chatbot_id}>"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ConversationCreate(ApiSchema):
    """Start a thread. The client may supply its own session handle."""

    session_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("session_id")
    @classmethod
    def _trim(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class ConversationOut(ApiSchema):
    id: uuid.UUID
    chatbot_id: uuid.UUID
    session_id: str
    source: ConversationSource
    title: str | None = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ConversationListOut(ApiSchema):
    conversations: list[ConversationOut]
    total: int
