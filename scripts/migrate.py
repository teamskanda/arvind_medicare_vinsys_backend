"""Apply SQL migrations in `migrations/` in filename order.

Usage (from the backend directory, with the venv active):
    python scripts/migrate.py

Each migration is idempotent and records itself in `schema_migrations`, so
re-running is safe. Statements are executed as one script per file inside a
transaction, which keeps DDL and the bookkeeping insert atomic.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text  # noqa: E402

from helpers.database import dispose_engine, get_engine  # noqa: E402

MIGRATIONS_DIR = BACKEND_ROOT / "migrations"


async def run() -> int:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {MIGRATIONS_DIR}")
        return 1

    engine = get_engine()
    try:
        for path in files:
            sql = path.read_text(encoding="utf-8")
            print(f"-> applying {path.name} ...", flush=True)
            # asyncpg's prepared statements accept one command. Its native
            # execute(script) uses the simple-query protocol for SQL files.
            async with engine.connect() as conn:
                raw = await conn.get_raw_connection()
                async with raw.driver_connection.transaction():
                    await raw.driver_connection.execute(sql)
            print(f"   {path.name} applied.")

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT version, applied_at FROM schema_migrations ORDER BY version")
            )
            print("\nApplied migrations:")
            for row in result:
                print(f"  {row.version}  {row.applied_at}")
    finally:
        await dispose_engine()

    print("\nMigrations complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"\nMigration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
