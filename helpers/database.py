"""Async SQLAlchemy engine, session factory and the FastAPI session dependency."""

from __future__ import annotations

import logging
import ssl
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    settings = get_settings()
    database_url = make_url(settings.database_url)
    tls_options = {}
    if database_url.query.get("ssl") == "verify-full":
        # Use the system trust store on Windows too. asyncpg's string SSL
        # modes otherwise expect a separate PostgreSQL root.crt file.
        tls_options["ssl"] = ssl.create_default_context()
        database_url = database_url.difference_update_query(["ssl"])
    return create_async_engine(
        database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # survives Supabase idle-connection recycling
        pool_recycle=1800,
        connect_args={
            # Supabase's transaction-mode pooler (port 6543) cannot reuse
            # prepared statements. Disabling the cache and using unique
            # statement names makes the same code work against the session
            # pooler, the transaction pooler and a direct connection.
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__skanda_{uuid.uuid4().hex}__",
            "timeout": 15,
            "server_settings": {"application_name": "skanda-ai-backend"},
            **tls_options,
        },
    )


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, autoflush=False
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, rolled back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Session for code outside the request cycle (background crawl polling).

    Background tasks outlive the request, so they must not reuse the
    request-scoped session.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def verify_connection() -> None:
    """Fail fast at startup if the database is unreachable or unmigrated.

    The messages are written for whoever is setting the project up, because
    this is the first thing that breaks on a fresh machine.
    """
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            missing = await conn.execute(
                text(
                    """
                    SELECT t.name
                    FROM (VALUES ('profiles'), ('chatbots'),
                                 ('website_pages'), ('crawl_jobs'),
                                 ('conversations'), ('messages'), ('chatbot_assets')) AS t(name)
                    WHERE to_regclass('public.' || t.name) IS NULL
                    """
                )
            )
            absent = [row[0] for row in missing]
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to the database.\n"
            f"  Reason: {type(exc).__name__}: {exc}\n"
            "  Check DATABASE_URL in backend/.env. For Supabase, use the\n"
            "  Session pooler URI and URL-encode any special characters in\n"
            "  the password. Run `python scripts/check_setup.py` to diagnose."
        ) from exc

    if absent:
        raise RuntimeError(
            "The database is missing these tables: "
            + ", ".join(absent)
            + ".\n  Run `python scripts/migrate.py` before starting the API."
        )


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
