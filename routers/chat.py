"""Chat and conversation endpoints for the dashboard playground.

Thin by design: every endpoint validates authentication through `CurrentUser`,
delegates ownership checks and all business logic to `helpers/`, and shapes the
response. No retrieval, prompt or provider logic lives here.

Every endpoint here is authenticated and previews the chatbot's **draft**
configuration, because the playground is where a customer tries settings
before publishing them. The public, unauthenticated counterpart lives in
`routers/deployment.py` and serves the published configuration; both call the
same `helpers/chat_engine.py`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import chat_engine, deployment, errors
from helpers import chatbots as chatbot_helpers
from helpers import conversations as conversation_helpers
from helpers.auth import CurrentUser
from helpers.database import get_db
from helpers.token_manager import build_budget
from models.conversation import (
    ConversationCreate,
    ConversationListOut,
    ConversationOut,
)
from models.message import (
    ChatConfigOut,
    ChatRequest,
    ChatResponse,
    MessageListOut,
    MessageOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["chat"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Playground configuration
# ---------------------------------------------------------------------------


@router.get("/{chatbot_id}/chat/config", response_model=ChatConfigOut)
async def get_chat_config(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> ChatConfigOut:
    """Resolved AI settings and readiness for one chatbot.

    Lets the playground show the real greeting, the real included-page count
    and an honest "not configured" state, rather than discovering any of it by
    failing on the first question.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    stats = await chatbot_helpers.get_stats(db, user, chatbot_id)
    settings = get_settings()

    config = deployment.ai_config_for(chatbot, published=False)
    budget = build_budget()

    return ChatConfigOut(
        chatbot_id=chatbot.id,
        chatbot_name=config.chatbot_name,
        website_url=chatbot.website_url,
        greeting=config.greeting,
        fallback_message=config.fallback_message,
        persona=config.persona,
        response_length=config.response_length,
        language=config.language,
        cite_sources=config.cite_sources,
        included_pages=stats.included_pages,
        knowledge_base_ready=stats.included_pages > 0,
        ai_ready=settings.gemma_configured,
        # Model id only — never the endpoint or the key.
        model_label=settings.gemma_model,
        context_limit=budget.context_limit,
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@router.post("/{chatbot_id}/chat", response_model=ChatResponse)
async def send_message(
    chatbot_id: uuid.UUID,
    payload: ChatRequest,
    user: CurrentUser,
    db: DbSession,
    stream: Annotated[bool, Query(description="Stream the reply as SSE.")] = False,
) -> ChatResponse | StreamingResponse:
    """Ask this chatbot a question.

    `?stream=true` returns `text/event-stream`; otherwise one JSON response.
    Both go through the same prepare step, so retrieval, the token budget and
    persistence are identical either way.
    """
    request = await chat_engine.resolve_playground_turn(db, user, chatbot_id, payload)
    turn = await chat_engine.prepare(db, request)

    if not stream:
        return await chat_engine.complete(db, turn)

    return StreamingResponse(
        chat_engine.sse_events(db, turn),
        media_type="text/event-stream",
        headers=chat_engine.SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.post(
    "/{chatbot_id}/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    chatbot_id: uuid.UUID,
    payload: ConversationCreate,
    user: CurrentUser,
    db: DbSession,
) -> ConversationOut:
    """Start a conversation, or resume the one for this session handle."""
    conversation = await conversation_helpers.create_conversation_for_user(
        db, user, chatbot_id, session_id=payload.session_id
    )
    return ConversationOut.model_validate(conversation)


@router.get("/{chatbot_id}/conversations", response_model=ConversationListOut)
async def list_conversations(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> ConversationListOut:
    """List this chatbot's conversations, newest activity first."""
    rows = await conversation_helpers.list_conversations(db, user, chatbot_id)
    items: list[ConversationOut] = []
    for conversation, message_count in rows:
        out = ConversationOut.model_validate(conversation)
        out.message_count = message_count
        items.append(out)
    return ConversationListOut(conversations=items, total=len(items))


@router.get(
    "/{chatbot_id}/conversations/{conversation_id}/messages",
    response_model=MessageListOut,
)
async def list_messages(
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
) -> MessageListOut:
    """Full history for one conversation, oldest first."""
    messages = await conversation_helpers.list_messages(
        db, user, chatbot_id, conversation_id
    )
    return MessageListOut(
        conversation_id=conversation_id,
        messages=[MessageOut.model_validate(message) for message in messages],
        total=len(messages),
    )
