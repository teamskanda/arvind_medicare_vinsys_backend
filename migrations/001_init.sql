-- ============================================================================
-- SKANDA AI — Phase 1 schema
-- ----------------------------------------------------------------------------
-- Idempotent: safe to run repeatedly. Run via `python scripts/migrate.py`
-- or paste into the Supabase SQL editor.
--
-- Authorisation model (read this before changing anything):
--   * The FastAPI backend connects as the database owner and enforces
--     ownership in `helpers/` on EVERY query (owner_id is always in the
--     WHERE clause). That is the primary control.
--   * The RLS policies below are defence-in-depth for any access that goes
--     through Supabase's `authenticated`/`anon` roles (e.g. someone calling
--     the REST API directly with a user's anon-key JWT). They do NOT protect
--     against bugs in the backend, because the owner role bypasses RLS.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- 1. profiles — one row per customer, keyed to Supabase auth.users.id
--    Populated lazily by the backend on first authenticated request, so no
--    trigger on auth.users is required (avoids needing elevated privileges).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS profiles (
    id          uuid        PRIMARY KEY,
    full_name   text,
    email       text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS profiles_email_idx ON profiles (lower(email));

-- ---------------------------------------------------------------------------
-- 2. chatbots
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chatbots (
    id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Short, URL-safe, unguessable. Future iframe route:
    --   https://chat.skandaai.in/embed/{public_id}
    public_id         text        NOT NULL UNIQUE,
    owner_id          uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    name              text        NOT NULL,
    website_url       text        NOT NULL,
    status            text        NOT NULL DEFAULT 'draft'
                                  CHECK (status IN ('draft', 'ready', 'published')),
    crawl_status      text        NOT NULL DEFAULT 'pending'
                                  CHECK (crawl_status IN ('pending', 'crawling', 'processing',
                                                          'completed', 'partially_completed', 'failed')),
    -- Phase 3 customisation studio writes here. Kept as jsonb now so no
    -- migration is needed later.
    draft_config      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    published_config  jsonb,
    -- Optional branding, extracted best-effort during URL validation.
    site_title        text,
    brand_color       text,
    favicon_url       text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chatbots_owner_created_idx
    ON chatbots (owner_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- 3. website_pages — one row per scraped page. Content is NEVER concatenated
--    into a single field; Phase 2 retrieval selects rows from here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS website_pages (
    id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    chatbot_id     uuid        NOT NULL REFERENCES chatbots (id) ON DELETE CASCADE,
    -- Denormalised for cheap ownership filtering and RLS without a join.
    owner_id       uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    url            text        NOT NULL,
    title          text,
    content        text        NOT NULL DEFAULT '',
    content_hash   text,
    content_length integer     NOT NULL DEFAULT 0,
    language       text,
    -- Customer-controlled: excluded pages stay stored but are invisible to
    -- the Phase 2 AI.
    is_included    boolean     NOT NULL DEFAULT true,
    status         text        NOT NULL DEFAULT 'scraped'
                               CHECK (status IN ('scraped', 'duplicate', 'empty', 'failed')),
    -- Heuristic importance (homepage/about/pricing rank higher). Phase 2 uses
    -- this to break retrieval ties.
    priority       integer     NOT NULL DEFAULT 50,
    error          text,
    scraped_at     timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now(),
    -- Makes crawl-result ingestion idempotent: re-processing the same
    -- Firecrawl payload updates rows instead of duplicating them.
    CONSTRAINT website_pages_chatbot_url_key UNIQUE (chatbot_id, url)
);

CREATE INDEX IF NOT EXISTS website_pages_chatbot_idx
    ON website_pages (chatbot_id);
CREATE INDEX IF NOT EXISTS website_pages_included_idx
    ON website_pages (chatbot_id, is_included)
    WHERE is_included = true;
CREATE INDEX IF NOT EXISTS website_pages_hash_idx
    ON website_pages (chatbot_id, content_hash);

-- ---------------------------------------------------------------------------
-- 4. crawl_jobs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    chatbot_id       uuid        NOT NULL REFERENCES chatbots (id) ON DELETE CASCADE,
    owner_id         uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    provider         text        NOT NULL DEFAULT 'firecrawl',
    provider_job_id  text,
    status           text        NOT NULL DEFAULT 'pending'
                                 CHECK (status IN ('pending', 'crawling', 'processing',
                                                   'completed', 'partially_completed', 'failed')),
    progress_message text,
    total_pages      integer     NOT NULL DEFAULT 0,
    successful_pages integer     NOT NULL DEFAULT 0,
    failed_pages     integer     NOT NULL DEFAULT 0,
    duplicate_pages  integer     NOT NULL DEFAULT 0,
    credits_used     integer     NOT NULL DEFAULT 0,
    error_details    jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    -- Drives the "stale job" resync so progress survives a backend restart.
    last_synced_at   timestamptz,
    completed_at     timestamptz
);

CREATE INDEX IF NOT EXISTS crawl_jobs_chatbot_created_idx
    ON crawl_jobs (chatbot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS crawl_jobs_provider_job_idx
    ON crawl_jobs (provider_job_id);
-- At most one live job per chatbot: prevents double-crawling (and double
-- Firecrawl spend) if the customer mashes the button.
CREATE UNIQUE INDEX IF NOT EXISTS crawl_jobs_one_active_per_chatbot_idx
    ON crawl_jobs (chatbot_id)
    WHERE status IN ('pending', 'crawling', 'processing');

-- ---------------------------------------------------------------------------
-- 5. updated_at maintenance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS profiles_set_updated_at ON profiles;
CREATE TRIGGER profiles_set_updated_at BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS chatbots_set_updated_at ON chatbots;
CREATE TRIGGER chatbots_set_updated_at BEFORE UPDATE ON chatbots
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS crawl_jobs_set_updated_at ON crawl_jobs;
CREATE TRIGGER crawl_jobs_set_updated_at BEFORE UPDATE ON crawl_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- 6. Row Level Security (defence-in-depth — see header note)
-- ---------------------------------------------------------------------------
ALTER TABLE profiles      ENABLE ROW LEVEL SECURITY;
ALTER TABLE chatbots      ENABLE ROW LEVEL SECURITY;
ALTER TABLE website_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE crawl_jobs    ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    -- auth.uid() only exists on Supabase. On a plain PostgreSQL instance we
    -- skip policy creation; the backend ownership checks still apply.
    IF to_regprocedure('auth.uid()') IS NULL THEN
        RAISE NOTICE 'auth.uid() not found - skipping RLS policies (non-Supabase database).';
        RETURN;
    END IF;

    DROP POLICY IF EXISTS profiles_self ON profiles;
    CREATE POLICY profiles_self ON profiles
        FOR ALL USING (id = auth.uid()) WITH CHECK (id = auth.uid());

    DROP POLICY IF EXISTS chatbots_owner ON chatbots;
    CREATE POLICY chatbots_owner ON chatbots
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());

    DROP POLICY IF EXISTS website_pages_owner ON website_pages;
    CREATE POLICY website_pages_owner ON website_pages
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());

    DROP POLICY IF EXISTS crawl_jobs_owner ON crawl_jobs;
    CREATE POLICY crawl_jobs_owner ON crawl_jobs
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());
END $$;

-- ---------------------------------------------------------------------------
-- 7. Applied-migration bookkeeping
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version)
VALUES ('001_init')
ON CONFLICT (version) DO NOTHING;
