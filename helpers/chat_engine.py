"""The single reusable AI engine. One code path for every chatbot.

There is no per-customer model, prompt template or application here: a chatbot
is data (its pages, its configuration), and this module is the shared engine
that reads that data.

Three stages, with the identity question answered entirely in the first:

    1. resolve  -> who is asking, which chatbot, which configuration, are
                   they allowed, and are they within their request limits
    2. prepare  -> load included pages, keyword retrieval, token budget,
                   build the prompt. Calls no model.
    3. complete -> call Gemma, persist both turns
       stream_complete -> the same, as events

Stage 1 has two implementations and they are the *only* difference between
the dashboard playground and the public widget:

| | playground | public widget |
|---|---|---|
| identity | verified Supabase token | HMAC-signed visitor session |
| chatbot lookup | `id` + owner check | `public_id` + published check |
| configuration | `draft_config` | `published_config` |
| conversation source | `playground` | `embed` |
| request limits | per user, per day | per session, per chatbot, per IP |

Stages 2 and 3 are shared verbatim. That is what "the authenticated
playground and the public chatbot share the same underlying AI engine" means
here: not two call sites that happen to look similar, but one function each.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import conversations as conversation_helpers
from helpers import deployment, errors, gemma, prompt_builder, retrieval
from helpers.chatbots import get_chatbot
from helpers.prompt_builder import BuiltPrompt, ChatbotAiConfig
from helpers.token_manager import build_budget, count_tokens
from helpers.visitor_session import VisitorSession
from models.chatbot import Chatbot
from models.conversation import Conversation, ConversationSource
from models.message import (
    ChatRequest,
    ChatResponse,
    MessageOut,
    MessageRole,
    RetrievalDebug,
)
from models.user import AuthenticatedUser

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """Stage 1's output: who is asking and what they are allowed to see.

    By the time one of these exists, access has already been granted — the
    chatbot row was fetched with an ownership or publication predicate, and
    the request limits have been checked. Everything downstream reads this
    and never re-derives identity, so there is exactly one place per caller
    where an authorisation mistake could live.
    """

    chatbot: Chatbot
    config: ChatbotAiConfig
    question: str
    source: ConversationSource
    session_handle: str | None = None
    conversation_id: uuid.UUID | None = None
    # Public visitors only: a conversation may be resumed only when its
    # session handle also matches, so one visitor cannot continue another's
    # thread with a guessed conversation id.
    isolate_by_session: bool = False

    @property
    def owner_id(self) -> uuid.UUID:
        """The customer who owns the chatbot, and therefore the rows we write.

        A public visitor's conversation belongs to the customer, not to the
        visitor: the customer is who sees it in their dashboard and whose
        data-retention obligations it falls under.
        """
        return self.chatbot.owner_id


@dataclass(slots=True)
class PreparedTurn:
    """Everything resolved before the model is called."""

    chatbot: Chatbot
    conversation: Conversation
    config: ChatbotAiConfig
    prompt: BuiltPrompt
    retrieval_result: retrieval.RetrievalResult
    question: str
    token_estimate_source: str

    @property
    def has_context(self) -> bool:
        return bool(self.retrieval_result.sections) and self.prompt.sections_used > 0

    def debug(self) -> RetrievalDebug:
        return RetrievalDebug(
            sections_considered=self.retrieval_result.sections_considered,
            sections_used=self.prompt.sections_used,
            pages_searched=self.retrieval_result.pages_searched,
            knowledge_tokens=self.prompt.knowledge_tokens,
            prompt_tokens_estimated=self.prompt.prompt_tokens_estimated,
            history_messages_used=self.prompt.history_messages_used,
            whole_knowledge_base_used=self.retrieval_result.whole_knowledge_base_used,
            keywords=self.retrieval_result.keywords[:12],
            token_estimate_source=self.token_estimate_source,
        )


async def resolve_playground_turn(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    payload: ChatRequest,
) -> TurnRequest:
    """Stage 1 for the authenticated dashboard playground.

    Previews the **draft** configuration on purpose: the playground is where
    a customer tries settings before publishing them.
    """
    # 404 (not 403) for someone else's id, so the endpoint does not confirm
    # that the chatbot exists.
    chatbot = await get_chatbot(db, user, chatbot_id)

    # Request limits before any work that costs money.
    await conversation_helpers.enforce_rate_limits(db, user.id, chatbot_id)

    return TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=False),
        question=payload.message,
        source=ConversationSource.PLAYGROUND,
        session_handle=payload.session_id,
        conversation_id=payload.conversation_id,
    )


async def resolve_public_turn(
    db: AsyncSession,
    chatbot: Chatbot,
    session: VisitorSession,
    *,
    question: str,
    conversation_id: uuid.UUID | None,
    client_ip: str | None = None,
) -> TurnRequest:
    """Stage 1 for a public website visitor.

    `chatbot` must already have come from `deployment.get_published_chatbot`,
    which is where the publication status is enforced; passing the row in
    rather than the handle makes that ordering impossible to get wrong.

    Serves the **published** configuration, so an unpublished draft is never
    shown to a visitor.
    """
    await conversation_helpers.enforce_public_rate_limits(
        db, chatbot.id, session.conversation_handle, client_ip=client_ip
    )

    return TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=True),
        question=question,
        source=ConversationSource.EMBED,
        session_handle=session.conversation_handle,
        conversation_id=conversation_id,
        isolate_by_session=True,
    )


async def prepare(db: AsyncSession, request: TurnRequest) -> PreparedTurn:
    """Stages 2: retrieve, budget and build the prompt. Calls no model.

    Shared by both callers without a branch. Raises `AppError` for the
    conditions the person asking should see: no knowledge base, or a
    conversation that is not theirs.
    """
    settings = get_settings()
    chatbot = request.chatbot
    chatbot_id = chatbot.id

    # 1. Load this chatbot's included pages only.
    pages = await conversation_helpers.load_included_pages(
        db, request.owner_id, chatbot_id
    )
    if not pages:
        raise errors.knowledge_base_empty()

    # 2. Resume or start the conversation.
    if request.conversation_id is not None:
        if request.isolate_by_session:
            conversation = await conversation_helpers.get_visitor_conversation(
                db,
                request.owner_id,
                chatbot_id,
                request.conversation_id,
                request.session_handle or "",
            )
        else:
            conversation = await conversation_helpers.get_conversation(
                db, request.owner_id, chatbot_id, request.conversation_id
            )
    else:
        conversation = await conversation_helpers.create_conversation(
            db,
            request.owner_id,
            chatbot_id,
            session_id=request.session_handle,
            source=request.source,
        )

    config = request.config
    budget = build_budget()
    history = await conversation_helpers.load_history(db, conversation, turns=settings.chat_history_turns)

    # 3. Keyword retrieval over this chatbot's pages.
    rows = [
        retrieval.PageRow(
            id=page.id,
            url=page.url,
            title=page.title or page.url,
            content=page.content,
            priority=page.priority,
        )
        for page in pages
    ]
    result = retrieval.search(
        rows,
        retrieval.contextual_query(request.question, history),
        budget_tokens=budget.knowledge,
        max_sections=settings.retrieval_max_sections,
        min_score=settings.retrieval_min_score,
    )

    # 4. History, then the prompt within the budget.
    prompt = prompt_builder.build_prompt(
        config=config,
        sections=result.sections,
        question=request.question,
        history=history,
        budget=budget,
        system_role_mode=settings.gemma_system_role_mode,
    )

    return PreparedTurn(
        chatbot=chatbot,
        conversation=conversation,
        config=config,
        prompt=prompt,
        retrieval_result=result,
        question=request.question,
        token_estimate_source=budget.counter_source,
    )


async def complete(db: AsyncSession, turn: PreparedTurn) -> ChatResponse:
    """Non-streaming turn: call Gemma, persist both messages, return them."""
    settings = get_settings()

    user_message = await conversation_helpers.add_message(
        db,
        turn.conversation,
        role=MessageRole.USER,
        content=turn.question,
        commit=True,
    )
    await conversation_helpers.set_conversation_title(db, turn.conversation, turn.question)

    # No relevant content: return the configured fallback without spending a
    # model call on content that cannot answer the question.
    if not turn.has_context:
        logger.info(
            "No relevant content for chatbot %s; returning the configured fallback.",
            turn.chatbot.id,
        )
        assistant_message = await conversation_helpers.add_message(
            db,
            turn.conversation,
            role=MessageRole.ASSISTANT,
            content=turn.config.fallback_message,
            source_references=[],
        )
        return _response(turn, user_message, assistant_message, used_fallback=True)

    error_code: str | None = None
    try:
        result = await gemma.generate(
            turn.prompt.messages,
            system_text=turn.prompt.system_text,
            temperature=settings.gemma_temperature,
            max_output_tokens=settings.gemma_max_output_tokens,
        )
        answer = result.text
        input_tokens = result.usage.input_tokens or turn.prompt.prompt_tokens_estimated
        output_tokens = result.usage.output_tokens or count_tokens(answer)

        if not answer:
            # Empty generation (safety filter, or a provider returning no
            # candidate). Report the fallback rather than an empty bubble.
            logger.warning(
                "Gemma returned no text for chatbot %s (finish_reason=%s).",
                turn.chatbot.id,
                result.finish_reason,
            )
            answer = turn.config.fallback_message
            error_code = "empty_generation"

    except errors.AppError as exc:
        # Persist the failure so the thread shows what happened instead of
        # silently dropping the turn. The router re-raises for the HTTP status.
        await conversation_helpers.add_message(
            db,
            turn.conversation,
            role=MessageRole.ASSISTANT,
            content=exc.message,
            error_code=exc.code,
        )
        raise

    used_fallback = answer.strip() == turn.config.fallback_message.strip() or error_code is not None
    assistant_message = await conversation_helpers.add_message(
        db,
        turn.conversation,
        role=MessageRole.ASSISTANT,
        content=answer,
        source_references=[] if used_fallback else turn.prompt.citations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_code=error_code,
    )
    return _response(
        turn, user_message, assistant_message, used_fallback=used_fallback
    )


async def stream_complete(
    db: AsyncSession, turn: PreparedTurn
) -> AsyncIterator[dict]:
    """Streaming turn, yielding event dicts for the router to serialise as SSE.

    Event kinds:
      ``start``  — conversation and user-message ids, plus retrieval debug;
      ``delta``  — a chunk of answer text;
      ``sources``— the citations (sent once, before ``done``);
      ``done``   — the persisted assistant message;
      ``error``  — a handled failure; the stream then ends.

    An interrupted stream still persists whatever text arrived, so a dropped
    connection does not lose the answer.
    """
    settings = get_settings()

    user_message = await conversation_helpers.add_message(
        db, turn.conversation, role=MessageRole.USER, content=turn.question, commit=True
    )
    await conversation_helpers.set_conversation_title(db, turn.conversation, turn.question)

    yield {
        "type": "start",
        "conversation_id": str(turn.conversation.id),
        "user_message_id": str(user_message.id),
        "retrieval": turn.debug().model_dump(mode="json"),
    }

    if not turn.has_context:
        assistant_message = await conversation_helpers.add_message(
            db,
            turn.conversation,
            role=MessageRole.ASSISTANT,
            content=turn.config.fallback_message,
        )
        yield {"type": "delta", "text": turn.config.fallback_message}
        yield {
            "type": "done",
            "used_fallback": True,
            "message": MessageOut.model_validate(assistant_message).model_dump(mode="json"),
        }
        return

    collected: list[str] = []
    reported_input: int | None = None
    reported_output: int | None = None
    error: errors.AppError | None = None

    try:
        async for delta, usage in gemma.stream_generate(
            turn.prompt.messages,
            system_text=turn.prompt.system_text,
            temperature=settings.gemma_temperature,
            max_output_tokens=settings.gemma_max_output_tokens,
        ):
            if delta:
                collected.append(delta)
                yield {"type": "delta", "text": delta}
            if usage is not None and usage.reported:
                reported_input = usage.input_tokens
                reported_output = usage.output_tokens
    except errors.AppError as exc:
        error = exc
    except Exception:
        logger.exception("Unexpected failure while streaming for chatbot %s.", turn.chatbot.id)
        error = errors.ai_unavailable()

    answer = "".join(collected).strip()

    if error is not None and not answer:
        # Nothing usable arrived: record the failure and tell the client.
        await conversation_helpers.add_message(
            db,
            turn.conversation,
            role=MessageRole.ASSISTANT,
            content=error.message,
            error_code=error.code,
        )
        yield {
            "type": "error",
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        }
        return

    partial = error is not None and bool(answer)
    if not answer:
        answer = turn.config.fallback_message

    used_fallback = answer.strip() == turn.config.fallback_message.strip()
    citations = [] if used_fallback else turn.prompt.citations

    assistant_message = await conversation_helpers.add_message(
        db,
        turn.conversation,
        role=MessageRole.ASSISTANT,
        content=answer,
        source_references=citations,
        input_tokens=reported_input or turn.prompt.prompt_tokens_estimated,
        output_tokens=reported_output or count_tokens(answer),
        error_code="stream_interrupted" if partial else None,
    )

    if citations:
        yield {"type": "sources", "sources": citations}

    yield {
        "type": "done",
        "used_fallback": used_fallback,
        "partial": partial,
        "message": MessageOut.model_validate(assistant_message).model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# SSE framing — shared by the playground and the public widget
# ---------------------------------------------------------------------------

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",  # stops nginx buffering the stream
    "Connection": "keep-alive",
}


async def sse_events(db: AsyncSession, turn: PreparedTurn) -> AsyncIterator[str]:
    """`stream_complete` serialised as SSE frames, with terminal error handling.

    Lives here rather than in a router because both the authenticated and the
    public streaming endpoint need identical behaviour on a late failure: by
    the time the first byte is sent the HTTP status is already 200, so an
    error has to travel *inside* the stream. Two copies of that logic would
    eventually disagree, and the one that disagreed would be the public one.
    """
    try:
        async for event in stream_complete(db, turn):
            yield f"data: {json.dumps(event)}\n\n"
    except errors.AppError as exc:
        yield "data: " + json.dumps(
            {
                "type": "error",
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
            }
        ) + "\n\n"
    except Exception:
        logger.exception(
            "Unhandled error while streaming chat for chatbot %s", turn.chatbot.id
        )
        yield "data: " + json.dumps(
            {
                "type": "error",
                "code": "internal_error",
                "message": "Something went wrong on our side. Please try again.",
                "retryable": True,
            }
        ) + "\n\n"

    # Deliberately NOT in a `finally`. When the client disconnects, the server
    # throws GeneratorExit in here, and yielding while unwinding that raises
    # `RuntimeError: async generator ignored GeneratorExit` on every dropped
    # connection. Reaching this line means the stream ran to completion or
    # failed in a way we handled — the two cases where a terminator is
    # actually useful. If nobody is listening, nobody needs one.
    yield "data: [DONE]\n\n"


def _response(
    turn: PreparedTurn, user_message, assistant_message, *, used_fallback: bool
) -> ChatResponse:
    return ChatResponse(
        conversation_id=turn.conversation.id,
        user_message=MessageOut.model_validate(user_message),
        assistant_message=MessageOut.model_validate(assistant_message),
        retrieval=turn.debug(),
        used_fallback=used_fallback,
    )
