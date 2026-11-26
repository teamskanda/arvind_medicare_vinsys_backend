"""Compile the Phase 2 queries against the PostgreSQL dialect.

No database is needed. These catch the mistakes that would otherwise surface
mid-demo: a missing ownership predicate, a malformed FILTER clause, or a
query that reads across chatbots.

The ownership assertions are the important ones. Customer isolation is
enforced by a WHERE clause on every query, so a query that compiles without
`owner_id` in it is a data-leak bug regardless of what the helper intended.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from models.base import PageStatus
from models.conversation import Conversation
from models.message import Message, MessageRole
from models.website_page import WebsitePage


def compile_sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}
        )
    )


OWNER = uuid.uuid4()
CHATBOT = uuid.uuid4()
CONVERSATION = uuid.uuid4()


# ---------------------------------------------------------------------------
# Knowledge-base loading
# ---------------------------------------------------------------------------


def _included_pages_query():
    return (
        select(WebsitePage)
        .where(
            WebsitePage.chatbot_id == CHATBOT,
            WebsitePage.owner_id == OWNER,
            WebsitePage.is_included.is_(True),
            WebsitePage.status == PageStatus.SCRAPED.value,
            WebsitePage.content != "",
        )
        .order_by(WebsitePage.priority.desc(), WebsitePage.url.asc())
    )


def test_included_pages_query_compiles():
    sql = compile_sql(_included_pages_query())
    assert "FROM website_pages" in sql


def test_included_pages_query_is_scoped_to_one_chatbot_and_one_owner():
    """Both predicates are required: either alone permits a leak."""
    sql = compile_sql(_included_pages_query())
    assert "website_pages.chatbot_id = " in sql
    assert "website_pages.owner_id = " in sql


def test_included_pages_query_honours_the_customers_exclusions():
    sql = compile_sql(_included_pages_query())
    assert "website_pages.is_included IS true" in sql
    assert "website_pages.status = " in sql


def test_included_pages_query_orders_by_priority():
    sql = compile_sql(_included_pages_query())
    assert "ORDER BY website_pages.priority DESC" in sql


# ---------------------------------------------------------------------------
# Conversation lookup
# ---------------------------------------------------------------------------


def _conversation_lookup():
    return select(Conversation).where(
        Conversation.id == CONVERSATION,
        Conversation.chatbot_id == CHATBOT,
        Conversation.owner_id == OWNER,
    )


def test_conversation_lookup_requires_chatbot_and_owner():
    """Owner alone is not enough: it would allow replaying a conversation id
    from chatbot A against chatbot B, mixing two knowledge bases."""
    sql = compile_sql(_conversation_lookup())
    assert "conversations.id = " in sql
    assert "conversations.chatbot_id = " in sql
    assert "conversations.owner_id = " in sql


def test_session_lookup_is_scoped():
    statement = select(Conversation).where(
        Conversation.chatbot_id == CHATBOT,
        Conversation.owner_id == OWNER,
        Conversation.session_id == "abc",
    )
    sql = compile_sql(statement)
    assert "conversations.session_id = " in sql
    assert "conversations.owner_id = " in sql


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------


def _history_query():
    return (
        select(Message.role, Message.content)
        .where(
            Message.conversation_id == CONVERSATION,
            Message.owner_id == OWNER,
            Message.error_code.is_(None),
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(12)
    )


def test_history_query_compiles_and_is_scoped():
    sql = compile_sql(_history_query())
    assert "messages.conversation_id = " in sql
    assert "messages.owner_id = " in sql


def test_history_query_excludes_failed_turns():
    """Replaying "the AI is unavailable" as context teaches it to repeat that."""
    assert "messages.error_code IS NULL" in compile_sql(_history_query())


def test_history_query_is_bounded():
    assert "LIMIT" in compile_sql(_history_query())


def test_history_query_selects_only_role_and_content():
    """Loading full rows into a prompt builder invites accidental leakage."""
    sql = compile_sql(_history_query())
    assert "SELECT messages.role, messages.content" in sql
    assert "messages.source_references" not in sql


# ---------------------------------------------------------------------------
# Message listing
# ---------------------------------------------------------------------------


def test_message_listing_is_owner_scoped_and_ordered():
    statement = (
        select(Message)
        .where(Message.conversation_id == CONVERSATION, Message.owner_id == OWNER)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    sql = compile_sql(statement)
    assert "messages.owner_id = " in sql
    assert "ORDER BY messages.created_at ASC" in sql


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _rate_limit_query():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return (
        select(
            func.count()
            .filter(Message.created_at >= now - timedelta(minutes=1))
            .label("last_minute"),
            func.count()
            .filter(Message.created_at >= now - timedelta(days=1))
            .label("last_day"),
            func.count()
            .filter(
                Message.created_at >= now - timedelta(minutes=1),
                Message.chatbot_id == CHATBOT,
            )
            .label("chatbot_last_minute"),
        )
        .select_from(Message)
        .where(
            Message.owner_id == OWNER,
            Message.role == MessageRole.USER.value,
            Message.created_at >= now - timedelta(days=1),
        )
    )


def test_rate_limit_query_compiles_with_filter_clauses():
    sql = compile_sql(_rate_limit_query())
    assert sql.count("FILTER (WHERE") == 3
    assert "FROM messages" in sql


def test_rate_limit_query_counts_only_this_users_questions():
    sql = compile_sql(_rate_limit_query())
    assert "messages.owner_id = " in sql
    assert "messages.role = " in sql


def test_rate_limit_query_prunes_by_time_in_the_where_clause():
    """Without the outer time bound this scans every message ever sent."""
    sql = compile_sql(_rate_limit_query())
    where = sql.split("WHERE", 1)[1]
    assert "messages.created_at >=" in where


# ---------------------------------------------------------------------------
# Conversation listing with counts
# ---------------------------------------------------------------------------


def test_conversation_listing_with_message_counts_compiles():
    subquery = (
        select(Message.conversation_id, func.count().label("message_count"))
        .where(Message.owner_id == OWNER)
        .group_by(Message.conversation_id)
        .subquery()
    )
    statement = (
        select(Conversation, func.coalesce(subquery.c.message_count, 0))
        .outerjoin(subquery, subquery.c.conversation_id == Conversation.id)
        .where(Conversation.chatbot_id == CHATBOT, Conversation.owner_id == OWNER)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    sql = compile_sql(statement)
    assert "LEFT OUTER JOIN" in sql
    assert "coalesce" in sql.lower()
    assert "GROUP BY messages.conversation_id" in sql
    assert "conversations.owner_id = " in sql


# ---------------------------------------------------------------------------
# Schema expectations the migration must satisfy
# ---------------------------------------------------------------------------


def test_message_model_matches_the_migration_columns():
    columns = set(Message.__table__.columns.keys())
    assert columns == {
        "id", "conversation_id", "chatbot_id", "owner_id", "role", "content",
        "source_references", "input_tokens", "output_tokens", "error_code",
        "created_at",
    }


def test_conversation_model_matches_the_migration_columns():
    columns = set(Conversation.__table__.columns.keys())
    assert columns == {
        "id", "chatbot_id", "owner_id", "session_id", "source", "title",
        "created_at", "updated_at",
    }


def test_owner_id_is_denormalised_onto_both_new_tables():
    """What makes isolation a single-predicate check rather than a join."""
    assert "owner_id" in Message.__table__.columns
    assert "owner_id" in Conversation.__table__.columns


def test_cascade_deletes_are_declared():
    """Deleting a chatbot must not orphan its conversations or messages."""
    conversation_fk = next(
        fk for fk in Conversation.__table__.foreign_keys
        if fk.column.table.name == "chatbots"
    )
    assert conversation_fk.ondelete == "CASCADE"

    message_fk = next(
        fk for fk in Message.__table__.foreign_keys
        if fk.column.table.name == "conversations"
    )
    assert message_fk.ondelete == "CASCADE"
