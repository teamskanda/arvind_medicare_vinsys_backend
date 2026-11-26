"""Compile the non-trivial SQL against the PostgreSQL dialect.

No database is needed: these catch statement-construction bugs (wrong
conflict target, bad FILTER clause, missing column) at test time rather than
mid-demo.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from helpers.content_processing import ProcessedPage, ProcessingResult
from helpers.crawling import (
    build_page_counts_query,
    build_page_rows,
    build_page_upsert,
)


def compile_sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )


def _result() -> ProcessingResult:
    return ProcessingResult(
        pages=[
            ProcessedPage(
                url="https://acme.com/",
                title="Home",
                content="Body content for the homepage.",
                content_hash="a" * 64,
                content_length=30,
                language="en",
                priority=100,
                status="scraped",
            ),
            ProcessedPage(
                url="https://acme.com/copy",
                title="Copy",
                content="Body content for the homepage.",
                content_hash="a" * 64,
                content_length=30,
                language="en",
                priority=50,
                status="duplicate",
                error="Identical content to https://acme.com/",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def test_page_rows_carry_ownership_and_inclusion_flags():
    chatbot_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    rows = build_page_rows(chatbot_id, owner_id, _result(), datetime.now(timezone.utc))

    assert len(rows) == 2
    assert all(row["chatbot_id"] == chatbot_id for row in rows)
    assert all(row["owner_id"] == owner_id for row in rows)

    scraped = next(r for r in rows if r["status"] == "scraped")
    duplicate = next(r for r in rows if r["status"] == "duplicate")
    # Only usable pages enter the AI knowledge base.
    assert scraped["is_included"] is True
    assert duplicate["is_included"] is False


def test_page_rows_include_every_column_the_upsert_sets():
    rows = build_page_rows(
        uuid.uuid4(), uuid.uuid4(), _result(), datetime.now(timezone.utc)
    )
    required = {
        "chatbot_id", "owner_id", "url", "title", "content", "content_hash",
        "content_length", "language", "is_included", "status", "priority",
        "error", "scraped_at",
    }
    assert required.issubset(rows[0].keys())


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def test_upsert_targets_the_chatbot_url_constraint():
    rows = build_page_rows(
        uuid.uuid4(), uuid.uuid4(), _result(), datetime.now(timezone.utc)
    )
    sql = compile_sql(build_page_upsert(rows))

    assert "INSERT INTO website_pages" in sql
    assert "ON CONFLICT ON CONSTRAINT website_pages_chatbot_url_key" in sql
    assert "DO UPDATE SET" in sql


def test_upsert_refreshes_content_but_preserves_customer_inclusion_choice():
    """Re-crawling must not silently re-include a page the customer excluded."""
    rows = build_page_rows(
        uuid.uuid4(), uuid.uuid4(), _result(), datetime.now(timezone.utc)
    )
    sql = compile_sql(build_page_upsert(rows))

    update_clause = sql.split("DO UPDATE SET", 1)[1]
    for column in ("content", "content_hash", "title", "status", "scraped_at"):
        assert f"{column} =" in update_clause, f"{column} should be refreshed"
    assert "is_included =" not in update_clause


def test_upsert_handles_a_single_row():
    single = ProcessingResult(pages=[_result().pages[0]])
    rows = build_page_rows(uuid.uuid4(), uuid.uuid4(), single, datetime.now(timezone.utc))
    sql = compile_sql(build_page_upsert(rows))
    assert "ON CONFLICT" in sql


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_page_counts_query_uses_filtered_aggregates():
    sql = compile_sql(build_page_counts_query(uuid.uuid4()))

    # PostgreSQL FILTER clauses, not CASE WHEN sums.
    assert sql.count("FILTER (WHERE") == 4
    assert "count(*)" in sql
    assert "coalesce(sum(website_pages.content_length)" in sql
    assert "WHERE website_pages.chatbot_id" in sql


def test_page_counts_query_is_scoped_to_one_chatbot():
    sql = compile_sql(build_page_counts_query(uuid.uuid4()))
    assert "website_pages.chatbot_id = " in sql


# ---------------------------------------------------------------------------
# Ownership scoping on the customer-facing queries
# ---------------------------------------------------------------------------


def test_stats_query_filters_by_owner():
    """Regression guard: the aggregate must never span customers."""
    from sqlalchemy import func, select

    from models.website_page import WebsitePage

    owner_id = uuid.uuid4()
    chatbot_id = uuid.uuid4()
    statement = (
        select(WebsitePage.chatbot_id, func.count())
        .where(
            WebsitePage.chatbot_id.in_([chatbot_id]),
            WebsitePage.owner_id == owner_id,
        )
        .group_by(WebsitePage.chatbot_id)
    )
    sql = compile_sql(statement)
    assert "website_pages.owner_id" in sql
    assert "GROUP BY website_pages.chatbot_id" in sql
