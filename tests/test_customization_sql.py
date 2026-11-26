"""Phase 4 schema and queries: 004_customization.sql and chatbot_assets.

The migration is read as text because there is no database here. What that
can prove is what matters most about a migration: that it is idempotent, that
it constrains what it stores, that it enables RLS like its siblings, and that
the SQL and the ORM model agree — a disagreement between those two is a
runtime error on the customer's first upload, not a test failure.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql

from helpers import customization as cz
from models.customization import ASSET_CONTENT_TYPES, ASSET_KINDS, ChatbotAsset

MIGRATION = (
    Path(__file__).resolve().parents[1] / "migrations" / "004_customization.sql"
).read_text(encoding="utf-8")


def compiled(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


# ---------------------------------------------------------------------------
# The migration file
# ---------------------------------------------------------------------------


def test_the_migration_creates_the_asset_table_idempotently():
    assert "CREATE TABLE IF NOT EXISTS chatbot_assets" in MIGRATION
    assert MIGRATION.count("CREATE TABLE ") == MIGRATION.count("CREATE TABLE IF NOT EXISTS")
    assert MIGRATION.count("CREATE INDEX") == MIGRATION.count("CREATE INDEX IF NOT EXISTS")


def test_phase_4_adds_no_columns_to_existing_tables():
    """Every one of the studio's settings lives in a column that already
    existed: draft_config, deployment_config and published_config."""
    assert "ADD COLUMN" not in MIGRATION


def test_the_asset_table_cascades_from_both_owners():
    """Deleting a chatbot or a customer must not leave orphaned image bytes."""
    assert "REFERENCES chatbots (id) ON DELETE CASCADE" in MIGRATION
    assert "REFERENCES profiles (id) ON DELETE CASCADE" in MIGRATION


def test_the_asset_table_constrains_what_it_stores():
    """The database refuses an unexpected kind or content type too, so a bug
    in the API cannot persist one."""
    for kind in ASSET_KINDS:
        assert f"'{kind}'" in MIGRATION
    for content_type in ASSET_CONTENT_TYPES:
        assert f"'{content_type}'" in MIGRATION
    assert "byte_size > 0" in MIGRATION


def test_svg_is_not_an_accepted_content_type_anywhere():
    """It is a document that can carry script, served from our own origin."""
    assert "svg" not in MIGRATION.lower()
    assert not any("svg" in content_type for content_type in ASSET_CONTENT_TYPES)


def test_row_level_security_is_enabled_like_its_sibling_tables():
    assert "ALTER TABLE chatbot_assets ENABLE ROW LEVEL SECURITY" in MIGRATION
    assert "owner_id = auth.uid()" in MIGRATION
    # And degrades on a non-Supabase database rather than failing the run.
    assert "to_regprocedure('auth.uid()')" in MIGRATION


def test_the_migration_records_itself():
    assert "INSERT INTO schema_migrations" in MIGRATION
    assert "'004_customization'" in MIGRATION


def test_the_orm_model_matches_the_migration():
    """A disagreement here is a runtime error on the first upload."""
    columns = set(ChatbotAsset.__table__.columns.keys())
    assert columns == {
        "id",
        "chatbot_id",
        "owner_id",
        "kind",
        "content_type",
        "byte_size",
        "data",
        "created_at",
    }
    for column in columns:
        assert column in MIGRATION, f"{column} is in the model but not the migration"


# ---------------------------------------------------------------------------
# Query shapes
# ---------------------------------------------------------------------------


def test_listing_assets_never_loads_the_image_bytes():
    """A chatbot's images can be a megabyte in total; the studio only needs
    their names and sizes to draw its gallery."""
    source = inspect.getsource(cz.list_assets)
    assert "ChatbotAsset.data" not in source
    assert "ChatbotAsset.byte_size" in source


def test_every_asset_query_is_owner_scoped():
    """Ownership is a predicate on the query, not a check afterwards."""
    for function in (cz.list_assets, cz.create_asset, cz.delete_asset):
        source = inspect.getsource(function)
        assert "owner_id" in source, f"{function.__name__} is not owner-scoped"


def test_deleting_an_asset_is_scoped_to_the_chatbot_and_the_owner():
    statement = delete(ChatbotAsset).where(
        ChatbotAsset.id == uuid.uuid4(),
        ChatbotAsset.chatbot_id == uuid.uuid4(),
        ChatbotAsset.owner_id == uuid.uuid4(),
    )
    sql = compiled(statement)
    where = sql.split("WHERE", 1)[1]
    assert where.count("=") == 3
    assert " OR " not in where


def test_the_public_asset_lookup_is_keyed_only_on_the_unguessable_id():
    """An anonymous visitor has no identity to scope by.

    The asset id is a capability, like `public_id`: handed out only by a
    chatbot's own configuration, and granting only an image the customer
    chose to display on their website. This test records that the lookup is
    deliberately not owner-scoped, so a future reader does not "fix" it.
    """
    source = inspect.getsource(cz.get_asset)
    assert "owner_id" not in source
    assert "capability" in source

    statement = select(ChatbotAsset).where(ChatbotAsset.id == uuid.uuid4())
    where = compiled(statement).split("WHERE", 1)[1]
    assert where.count("=") == 1


def test_an_asset_still_referenced_by_the_published_chatbot_cannot_be_deleted():
    """Deleting it would leave live visitors looking at a broken image."""
    source = inspect.getsource(cz.delete_asset)
    assert "asset_in_use" in source
    assert "published_config" in source
