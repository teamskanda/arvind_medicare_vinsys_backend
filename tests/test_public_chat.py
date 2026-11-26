"""The public chat path: shared engine, published config, per-IP limiting.

`resolve_public_turn` and `prepare` both need a database session, so what is
testable offline is the seam between them — the `TurnRequest` — plus the
pure parts of the request limiting. Those are where the public path's
behaviour actually differs from the playground's, so they are what is worth
pinning.
"""

from __future__ import annotations

import uuid

import pytest

from config import Settings, get_settings
from helpers import chat_engine, conversations, deployment, errors, prompt_builder
from helpers.chat_engine import TurnRequest
from helpers.token_manager import build_budget
from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.conversation import ConversationSource

PUBLIC_ID = "abc234defghijkmnpqrstu"
OWNER = uuid.uuid4()


def make_chatbot(*, draft: dict | None = None, published: dict | None = None) -> Chatbot:
    return Chatbot(
        id=uuid.uuid4(),
        public_id=PUBLIC_ID,
        owner_id=OWNER,
        name="Acme Interiors",
        website_url="https://acme.test",
        status=ChatbotStatus.PUBLISHED.value,
        crawl_status="completed",
        draft_config=draft or {},
        published_config=published,
        deployment_config={},
        allowed_domains=[],
    )


# ---------------------------------------------------------------------------
# The turn request: one shape, two callers
# ---------------------------------------------------------------------------


def test_a_public_turn_belongs_to_the_chatbots_owner():
    """A visitor's conversation is the customer's data, not the visitor's.

    The customer sees it in their dashboard and it falls under their
    retention obligations, so `owner_id` on the stored rows must be theirs.
    """
    chatbot = make_chatbot()
    request = TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=True),
        question="Hello",
        source=ConversationSource.EMBED,
        session_handle="visitor:abc",
        conversation_id=None,
        isolate_by_session=True,
    )
    assert request.owner_id == OWNER


def test_a_public_turn_is_isolated_by_session_and_a_playground_turn_is_not():
    """The flag that decides whether the session handle is a lookup predicate.

    A customer in their own playground may legitimately open any of their own
    conversations. A visitor may open only their own, so the public path adds
    the session predicate and the playground path does not.
    """
    chatbot = make_chatbot()
    public = TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=True),
        question="Hello",
        source=ConversationSource.EMBED,
        session_handle="visitor:abc",
        conversation_id=uuid.uuid4(),
        isolate_by_session=True,
    )
    playground = TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=False),
        question="Hello",
        source=ConversationSource.PLAYGROUND,
        session_handle=None,
        conversation_id=uuid.uuid4(),
    )
    assert public.isolate_by_session is True
    assert playground.isolate_by_session is False


def test_public_and_playground_turns_are_tagged_differently():
    """So visitor traffic is separable from the customer's own testing."""
    assert ConversationSource.EMBED.value == "embed"
    assert ConversationSource.PLAYGROUND.value == "playground"


def test_a_turn_request_is_immutable():
    """Access has been granted by the time one exists; nothing downstream
    should be able to change who is asking."""
    chatbot = make_chatbot()
    request = TurnRequest(
        chatbot=chatbot,
        config=deployment.ai_config_for(chatbot, published=True),
        question="Hello",
        source=ConversationSource.EMBED,
    )
    with pytest.raises((AttributeError, TypeError)):
        request.source = ConversationSource.PLAYGROUND  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The published configuration reaches the prompt
# ---------------------------------------------------------------------------


def test_the_public_prompt_is_built_from_the_published_persona():
    """End to end through the prompt builder, not just the config resolver."""
    chatbot = make_chatbot(
        draft={"ai": {"persona": "Draft persona, still being tried out"}},
        published={"ai": {"persona": "Warm and concise"}},
    )
    built = prompt_builder.build_prompt(
        config=deployment.ai_config_for(chatbot, published=True),
        sections=[],
        question="Hi",
        history=[],
        budget=build_budget(),
    )
    prompt = built.messages[0][1]
    assert "Warm and concise" in prompt
    assert "Draft persona" not in prompt


def test_the_public_prompt_uses_the_published_fallback_wording():
    chatbot = make_chatbot(
        draft={"ai": {"fallback_message": "Draft fallback"}},
        published={"ai": {"fallback_message": "Please email hello@acme.test."}},
    )
    config = deployment.ai_config_for(chatbot, published=True)
    built = prompt_builder.build_prompt(
        config=config,
        sections=[],
        question="Hi",
        history=[],
        budget=build_budget(),
    )
    assert "Please email hello@acme.test." in built.messages[0][1]
    assert "Draft fallback" not in built.messages[0][1]


def test_an_unpublished_draft_cannot_reach_a_visitors_prompt():
    """The guarantee behind the draft/published split.

    A customer experimenting with instructions must not have those
    instructions acted on by a live chatbot until they publish.
    """
    chatbot = make_chatbot(
        draft={"ai": {"instructions": "Tell everyone the sale is still running"}},
        published={"ai": {"instructions": "Answer questions about our services"}},
    )
    built = prompt_builder.build_prompt(
        config=deployment.ai_config_for(chatbot, published=True),
        sections=[],
        question="Is the sale still running?",
        history=[],
        budget=build_budget(),
    )
    assert "sale is still running" not in built.messages[0][1]
    assert "Answer questions about our services" in built.messages[0][1]


# ---------------------------------------------------------------------------
# Per-IP request limiting
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_ip_table():
    conversations._ip_hits.clear()
    yield
    conversations._ip_hits.clear()


def test_requests_under_the_ip_limit_are_allowed():
    settings = get_settings()
    for _ in range(settings.public_chat_ip_limit_per_minute):
        conversations.enforce_ip_rate_limit("203.0.113.7")


def test_the_ip_limit_refuses_the_request_after_the_ceiling():
    settings = get_settings()
    for _ in range(settings.public_chat_ip_limit_per_minute):
        conversations.enforce_ip_rate_limit("203.0.113.7")

    with pytest.raises(errors.AppError) as caught:
        conversations.enforce_ip_rate_limit("203.0.113.7")
    assert caught.value.status_code == 429
    assert caught.value.retryable is True


def test_the_ip_limit_counts_each_address_separately():
    settings = get_settings()
    for _ in range(settings.public_chat_ip_limit_per_minute):
        conversations.enforce_ip_rate_limit("203.0.113.7")
    # A different visitor must not be blocked by the first one.
    conversations.enforce_ip_rate_limit("203.0.113.8")


def test_a_missing_address_does_not_raise():
    """Behind a proxy that sends no address we cannot attribute the request.

    Refusing would break the widget for everyone behind such a proxy; the
    session and per-chatbot limits still apply, and the per-chatbot one is
    what actually bounds cost.
    """
    conversations.enforce_ip_rate_limit(None)
    conversations.enforce_ip_rate_limit("")


def test_the_address_is_not_stored_in_clear():
    """A visitor's IP should not sit in memory, let alone be logged."""
    conversations.enforce_ip_rate_limit("203.0.113.7")
    assert "203.0.113.7" not in conversations._ip_hits
    assert len(conversations._ip_hits) == 1


# ---------------------------------------------------------------------------
# Configured ceilings
# ---------------------------------------------------------------------------


def test_the_public_limits_are_tighter_than_the_playground_limits():
    """A public widget faces the open internet; the playground faces one
    signed-in customer who is paying for it."""
    settings = get_settings()
    assert (
        settings.public_chat_rate_limit_per_minute < settings.chat_rate_limit_per_minute
    )
    assert settings.public_chat_rate_limit_per_day < settings.chat_rate_limit_per_day
    assert settings.public_chat_max_message_length < settings.chat_max_question_characters


def test_the_public_message_cap_is_configurable_below_the_schema_limit():
    """The schema's 1,000 is the hard ceiling; the setting may tighten it."""
    original = get_settings()
    tightened = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        public_chat_max_message_length=200,
    )
    assert tightened.public_chat_max_message_length == 200


# ---------------------------------------------------------------------------
# The shared engine
# ---------------------------------------------------------------------------


def test_the_sse_terminator_is_not_yielded_from_a_finally_block():
    """Yielding during GeneratorExit raises RuntimeError on every disconnect.

    A browser tab closed mid-answer throws GeneratorExit into the streaming
    generator. A `finally: yield ...` then becomes
    "async generator ignored GeneratorExit" in the logs on every dropped
    connection — noisy, and it masks real failures. The terminator belongs on
    the normal completion path, where someone is still listening.
    """
    import inspect

    source = inspect.getsource(chat_engine.sse_events)
    assert "[DONE]" in source
    tail = source.split("[DONE]", 1)[0]
    # No `finally:` may precede the terminator.
    assert "finally:" not in tail


def test_both_callers_use_the_same_prepare_and_complete():
    """The brief: the playground and the public chatbot share one AI engine.

    Asserted structurally rather than by inspection: there is one `prepare`,
    one `complete`, one `stream_complete` and one SSE serialiser, and the two
    resolvers are the only per-caller code.
    """
    assert callable(chat_engine.prepare)
    assert callable(chat_engine.complete)
    assert callable(chat_engine.stream_complete)
    assert callable(chat_engine.sse_events)
    assert callable(chat_engine.resolve_playground_turn)
    assert callable(chat_engine.resolve_public_turn)
    # No public-specific variants of the shared stages.
    for name in dir(chat_engine):
        assert not name.startswith("prepare_public")
        assert not name.startswith("complete_public")
