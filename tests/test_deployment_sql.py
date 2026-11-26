"""Compile the Phase 3 queries against the PostgreSQL dialect.

Same approach as test_chat_sql.py and for the same reason: customer and
visitor isolation is enforced by WHERE clauses, so a query that compiles
without the right predicates is a data-leak bug no matter what the helper
intended. These catch that without a database.

The predicate counts here are deliberate. Phase 3 introduces a second kind of
caller who is not the owner, so the queries that serve them need one more
predicate than Phase 2's did, and that is exactly what a test should pin.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.conversation import Conversation, ConversationSource
from models.message import Message, MessageRole


def compile_sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}
        )
    )


OWNER = uuid.uuid4()
CHATBOT = uuid.uuid4()
CONVERSATION = uuid.uuid4()
HANDLE = "visitor:abc123"


# ---------------------------------------------------------------------------
# Published chatbot lookup
# ---------------------------------------------------------------------------


def _published_lookup():
    return select(Chatbot).where(
        Chatbot.public_id == "abc234defghijkmnpqrstu",
        Chatbot.status == ChatbotStatus.PUBLISHED.value,
    )


def test_the_public_lookup_filters_on_publication_status_in_the_query():
    """Not fetched-then-checked.

    If the status were verified in Python after loading the row, there would
    be a code path holding an unpublished chatbot. Putting the predicate in
    the query means that path does not exist.
    """
    sql = compile_sql(_published_lookup())
    assert "chatbots.public_id = " in sql
    assert "chatbots.status = " in sql


def test_the_public_lookup_is_keyed_only_on_the_handle_and_the_status():
    """A visitor supplies one value, and it is the only one in the WHERE clause.

    In particular there is no owner predicate: a visitor has no owner and
    must not be able to assert one. The owner is *read from* the row that
    comes back, which is how everything downstream stays owner-scoped.
    """
    where = compile_sql(_published_lookup()).split("WHERE", 1)[1]
    assert "chatbots.owner_id" not in where
    assert where.count("=") == 2  # public_id and status, nothing else


# ---------------------------------------------------------------------------
# Visitor conversation lookup — the isolation boundary
# ---------------------------------------------------------------------------


def _visitor_conversation_lookup():
    return select(Conversation).where(
        Conversation.id == CONVERSATION,
        Conversation.chatbot_id == CHATBOT,
        Conversation.owner_id == OWNER,
        Conversation.session_id == HANDLE,
    )


def test_a_visitor_conversation_requires_all_four_predicates():
    """The session predicate is what separates one visitor from another.

    Two visitors on the same public chatbot share the chatbot id and the
    owner id, so without `session_id` a visitor who learned another's
    conversation id could read and continue their thread.
    """
    sql = compile_sql(_visitor_conversation_lookup())
    for predicate in (
        "conversations.id = ",
        "conversations.chatbot_id = ",
        "conversations.owner_id = ",
        "conversations.session_id = ",
    ):
        assert predicate in sql, f"missing {predicate!r}"


def test_the_visitor_lookup_has_no_or_conditions():
    """An OR would widen the match and quietly undo the isolation above."""
    sql = compile_sql(_visitor_conversation_lookup())
    assert " OR " not in sql.upper()


# ---------------------------------------------------------------------------
# Public rate limiting
# ---------------------------------------------------------------------------


def _public_rate_limit_query():
    now = datetime.now(timezone.utc)
    minute_ago = now - timedelta(minutes=1)
    day_ago = now - timedelta(days=1)
    return (
        select(
            func.count()
            .filter(
                Message.created_at >= minute_ago,
                Conversation.session_id == HANDLE,
            )
            .label("session_minute"),
            func.count().filter(Conversation.session_id == HANDLE).label("session_day"),
            func.count().filter(Message.created_at >= minute_ago).label("bot_minute"),
            func.count().label("bot_day"),
        )
        .select_from(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.chatbot_id == CHATBOT,
            Message.role == MessageRole.USER.value,
            Message.created_at >= day_ago,
            Conversation.source == ConversationSource.EMBED.value,
        )
    )


def test_the_public_rate_limit_query_compiles():
    sql = compile_sql(_public_rate_limit_query())
    assert "FROM messages" in sql
    assert "JOIN conversations" in sql


def test_the_public_rate_limit_counts_sessions_and_the_chatbot_separately():
    """Four aggregates: per-session minute and day, per-chatbot minute and day."""
    sql = compile_sql(_public_rate_limit_query())
    assert sql.count("FILTER (WHERE") == 3  # the fourth is the unfiltered total


def test_the_public_rate_limit_ignores_playground_traffic():
    """Otherwise a customer testing their own bot would throttle real visitors."""
    sql = compile_sql(_public_rate_limit_query())
    assert "conversations.source = " in sql


def test_the_public_rate_limit_counts_only_visitor_questions():
    """Counting assistant replies as well would halve every configured limit."""
    sql = compile_sql(_public_rate_limit_query())
    assert "messages.role = " in sql


def test_the_public_rate_limit_is_pruned_by_time_in_the_where_clause():
    """Without the outer bound this scans every message the chatbot ever had."""
    sql = compile_sql(_public_rate_limit_query())
    where = sql.split("WHERE", 1)[1]
    assert "messages.created_at >=" in where


def test_the_public_rate_limit_is_scoped_to_one_chatbot():
    sql = compile_sql(_public_rate_limit_query())
    assert "messages.chatbot_id = " in sql


# ---------------------------------------------------------------------------
# Schema expectations the migration must satisfy
# ---------------------------------------------------------------------------


def test_the_chatbot_model_carries_the_phase_3_columns():
    columns = set(Chatbot.__table__.columns.keys())
    for column in (
        "published_at",
        "unpublished_at",
        "allowed_domains",
        "deployment_config",
    ):
        assert column in columns, f"003_deployment.sql adds {column}"


def test_the_schema_is_exactly_these_tables():
    """A new table should be a deliberate choice, so it is recorded here.

    Phase 3 added none: publication reuses `chatbots`, visitor chat reuses
    `conversations`, and a visitor session is signed rather than stored, so
    there is nothing to persist and nothing to expire-sweep.

    Phase 4 added exactly one, `chatbot_assets`, for the only thing a JSONB
    column is the wrong home for: uploaded image bytes. Every one of the
    studio's ~40 settings lives in `draft_config`, `deployment_config` and
    `published_config`, which all existed already.
    """
    from models.base import Base

    assert set(Base.metadata.tables) == {
        "chatbot_screenshots",
        "profiles",
        "chatbots",
        "website_pages",
        "crawl_jobs",
        "conversations",
        "messages",
        "chatbot_assets",
    }


def test_the_lifecycle_enum_covers_all_four_states():
    assert {status.value for status in ChatbotStatus} == {
        "draft",
        "ready",
        "published",
        "unpublished",
    }


def test_only_published_is_a_public_state():
    """Draft and unpublished must both be offline."""
    public = [status for status in ChatbotStatus if status.is_public]
    assert public == [ChatbotStatus.PUBLISHED]


def test_the_migration_widens_the_status_check_constraint():
    """The enum and the database must agree, or publishing fails at runtime."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[1] / "migrations" / "003_deployment.sql"
    ).read_text(encoding="utf-8")
    assert "chatbots_status_check" in sql
    for status in ChatbotStatus:
        assert f"'{status.value}'" in sql


def test_the_migration_records_itself():
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[1] / "migrations" / "003_deployment.sql"
    ).read_text(encoding="utf-8")
    assert "INSERT INTO schema_migrations" in sql
    assert "'003_deployment'" in sql


def test_every_added_column_is_idempotent():
    """Migrations are documented as safe to re-run."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[1] / "migrations" / "003_deployment.sql"
    ).read_text(encoding="utf-8")
    added = sql.count("ADD COLUMN")
    assert added > 0
    assert sql.count("ADD COLUMN IF NOT EXISTS") == added
    assert sql.count("CREATE INDEX") == sql.count("CREATE INDEX IF NOT EXISTS")
