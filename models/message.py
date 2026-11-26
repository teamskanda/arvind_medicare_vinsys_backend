"""Message: ORM model + schemas, plus the chat request/response contract.

`source_references` holds citations the retrieval step actually produced from
this chatbot's own stored pages. It is never parsed out of Gemma's output, so
the model cannot invent a URL that then gets rendered as a real source.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import ApiSchema, Base


class MessageRole(StrEnum):
    """Must match the CHECK constraint in 002_chat.sql."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised: lets the rate limiter and ownership checks run without
    # joining through conversations.
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_references: Mapped[list[dict]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Message id={self.id} role={self.role}>"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SourceReference(ApiSchema):
    """One citation. `url` always comes from a stored website_pages row."""

    title: str
    url: str
    page_id: uuid.UUID | None = None


class MessageOut(ApiSchema):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: MessageRole
    content: str
    source_references: list[SourceReference] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_code: str | None = None
    created_at: datetime


class MessageListOut(ApiSchema):
    conversation_id: uuid.UUID
    messages: list[MessageOut]
    total: int


class ChatRequest(ApiSchema):
    """A visitor question.

    `max_length` is enforced here rather than deeper in the stack so an
    oversized body is rejected before it reaches retrieval or the provider.
    """

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    session_id: str | None = Field(default=None, max_length=120)

    @field_validator("message")
    @classmethod
    def _trim_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Please type a question.")
        return trimmed

    @field_validator("session_id")
    @classmethod
    def _trim_session(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class RetrievalDebug(ApiSchema):
    """What retrieval and the token budget actually did.

    Shown in the playground so the customer can see the chatbot is answering
    from their own pages, and so a wrong answer is diagnosable rather than
    mysterious. Every number is measured, none are estimated for display.
    """

    sections_considered: int
    sections_used: int
    pages_searched: int
    knowledge_tokens: int
    prompt_tokens_estimated: int
    history_messages_used: int
    whole_knowledge_base_used: bool
    keywords: list[str] = Field(default_factory=list)
    token_estimate_source: str = "estimator"


class ChatConfigOut(ApiSchema):
    """What the playground needs before the first message.

    `ai_ready` is false when the operator has not configured a Gemma key, so
    the UI can say that plainly instead of failing on the first question.
    """

    chatbot_id: uuid.UUID
    chatbot_name: str
    website_url: str
    greeting: str
    fallback_message: str
    persona: str
    response_length: str
    language: str
    cite_sources: bool
    included_pages: int
    knowledge_base_ready: bool
    ai_ready: bool
    model_label: str
    context_limit: int


class ChatResponse(ApiSchema):
    conversation_id: uuid.UUID
    user_message: MessageOut
    assistant_message: MessageOut
    retrieval: RetrievalDebug | None = None
    # True when we returned the configured fallback because nothing relevant
    # was found, rather than an answer grounded in retrieved content.
    used_fallback: bool = False
