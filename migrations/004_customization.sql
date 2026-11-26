-- ============================================================================
-- SKANDA AI — Phase 4 schema: the customisation studio
-- ----------------------------------------------------------------------------
-- Idempotent: safe to run repeatedly. Run via `python scripts/migrate.py`
-- or paste into the Supabase SQL editor.
--
-- Design note — what Phase 4 does NOT add:
--
--   Every customisation setting (identity, appearance, personality, launcher)
--   is stored in columns that already exist:
--
--     chatbots.draft_config      -> the customer's working copy
--                                   { "ai": {...}, "appearance": {...} }
--     chatbots.deployment_config -> the launcher's working copy
--     chatbots.published_config  -> the snapshot visitors are served
--                                   { version, published_at, ai, appearance,
--                                     deployment }
--
--   JSONB rather than one column per setting: there are ~40 settings, they
--   are read as a whole document on every request, none of them is ever
--   queried or sorted on, and Phase 5 will add more. A 40-column ALTER would
--   buy nothing and cost a migration per new control. Every value is still
--   type-checked and clamped on the way in (models/customization.py) and
--   again on the way out (helpers/customization.py), so a malformed document
--   cannot reach the prompt or the browser.
--
-- So Phase 4 adds exactly one table, for the one thing a JSONB column is the
-- wrong home for: uploaded image bytes.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. chatbot_assets — customer-uploaded avatars, logos and launcher icons
-- ---------------------------------------------------------------------------
-- Why a table and not object storage: the studio needs working uploads with
-- no additional service to configure, and these images are tiny (hard-capped
-- at ASSET_MAX_BYTES, 256 KB by default, which is generous for a 96px
-- avatar). Moving them to Supabase Storage later is a change of one helper —
-- the configuration stores `asset:<uuid>`, never a bucket path or an absolute
-- URL, so nothing stored today has to be rewritten.
--
-- Rows are immutable: changing an avatar uploads a new row and repoints the
-- configuration. That is what makes publishing safe — the published snapshot
-- keeps referencing the asset it was published with, so editing a draft
-- avatar cannot change what live visitors see.
CREATE TABLE IF NOT EXISTS chatbot_assets (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    chatbot_id   uuid        NOT NULL REFERENCES chatbots (id) ON DELETE CASCADE,
    -- Denormalised, as on conversations and messages: ownership is then a
    -- single predicate rather than a join through chatbots.
    owner_id     uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    -- What the image is for. Not a constraint on where it may be used; it
    -- drives the dashboard's "replace this image" affordance.
    kind         text        NOT NULL
                             CHECK (kind IN ('avatar', 'logo', 'launcher_icon')),
    -- Allowlisted on upload and verified against the file's magic bytes, so
    -- this column cannot disagree with the bytes it describes.
    content_type text        NOT NULL
                             CHECK (content_type IN ('image/png', 'image/jpeg',
                                                     'image/webp', 'image/gif')),
    byte_size    integer     NOT NULL CHECK (byte_size > 0),
    data         bytea       NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- The dashboard lists a chatbot's uploads newest first; the public asset
-- endpoint reads by primary key.
CREATE INDEX IF NOT EXISTS chatbot_assets_chatbot_idx
    ON chatbot_assets (chatbot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chatbot_assets_owner_idx
    ON chatbot_assets (owner_id);

-- ---------------------------------------------------------------------------
-- 2. Row Level Security (defence-in-depth — same model as 001/002)
-- ---------------------------------------------------------------------------
-- The backend always filters on owner_id. These policies describe the same
-- rule for access arriving through Supabase's `authenticated` role.
--
-- Note the asymmetry with the public asset endpoint: an anonymous visitor
-- fetching an avatar does not reach the database as `authenticated` — the
-- backend reads the row as the database owner, having been given the asset's
-- unguessable uuid. The uuid is a capability, exactly like `public_id`: it is
-- in the page a visitor is already looking at, and it exposes nothing but an
-- image the customer chose to display.
ALTER TABLE chatbot_assets ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF to_regprocedure('auth.uid()') IS NULL THEN
        RAISE NOTICE 'auth.uid() not found - skipping RLS policies (non-Supabase database).';
        RETURN;
    END IF;

    DROP POLICY IF EXISTS chatbot_assets_owner ON chatbot_assets;
    CREATE POLICY chatbot_assets_owner ON chatbot_assets
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());
END $$;

-- ---------------------------------------------------------------------------
-- 3. Applied-migration bookkeeping
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version)
VALUES ('004_customization')
ON CONFLICT (version) DO NOTHING;
