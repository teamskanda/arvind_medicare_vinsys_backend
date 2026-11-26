"""Private studio backdrop metadata and its bounded image payload."""
from datetime import datetime
import uuid
from sqlalchemy import DateTime, ForeignKey, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import Field
from models.base import ApiSchema, Base

MAX_SCREENSHOT_BYTES = 5 * 1024 * 1024

class WebsiteScreenshot(Base):
    __tablename__ = 'chatbot_screenshots'
    chatbot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('chatbots.id', ondelete='CASCADE'), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('profiles.id', ondelete='CASCADE'))
    version: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    source_url: Mapped[str | None] = mapped_column(String)
    message: Mapped[str | None] = mapped_column(String)
    content_type: Mapped[str | None] = mapped_column(String)
    data: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class ScreenshotOut(ApiSchema):
    status: str
    source: str | None = None
    image_url: str | None = None
    message: str | None = None

class ScreenshotUpload(ApiSchema):
    data: str = Field(min_length=12, max_length=MAX_SCREENSHOT_BYTES * 4 // 3 + 2048)
