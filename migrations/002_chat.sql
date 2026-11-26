-- ============================================================================
-- SKANDA AI — Phase 2 schema: conversations and messages
-- ----------------------------------------------------------------------------
-- Idempotent: safe to run repeatedly. Run via `python scripts/migrate.py`
-- or paste into the Supabase SQL editor.
--
-- Same authorisation model as 001_init.sql: the backend enforces ownership on
-- every query (owner_id is always in the WHERE clause) and the RLS policies
-- below are defence-in-depth for access arriving through Supabase's
-- `authenticated` role.
--
-- `owner_id` is denormalised onto both tables so a conversation or message can
-- be ownership-checked without joining back through chatbots. That is what
-- makes cross-customer and cross-chatbot leakage a single-predicate problem.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. conversations — one chat session with one chatbot
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    chatbot_id  uuid        NOT NULL REFERENCES chatbots (id) ON DELETE CASCADE,
    owner_id    uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    -- Opaque client-supplied session handle. In Phase 2 this is the dashboard
    -- playground; in Phase 3 the public iframe reuses the same column for an
    -- anonymous visitor session, which is why it is not a foreign key.
    session_id  text        NOT NULL,
    -- 'playground' now; 'embed' once the iframe exists. Lets us keep internal
    -- testing traffic separate from real visitor traffic later.
    source      text        NOT NULL DEFAULT 'playground'
                            CHECK (source IN ('playground', 'embed')),
    title       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_chatbot_created_idx
    ON conversations (chatbot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS conversations_owner_idx
    ON conversations (owner_id);
-- One live conversation per (chatbot, session) so a page reload resumes the
-- same thread instead of silently starting a new one.
CREATE UNIQUE INDEX IF NOT EXISTS conversations_chatbot_session_key
    ON conversations (chatbot_id, session_id);

-- ---------------------------------------------------------------------------
-- 2. messages — one row per turn
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id   uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    chatbot_id        uuid        NOT NULL REFERENCES chatbots (id) ON DELETE CASCADE,
    owner_id          uuid        NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
    role              text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content           text        NOT NULL,
    -- Citations actually retrieved from this chatbot's own pages. Never model
    -- output, so Gemma cannot invent a source URL.
    source_references jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- Provider-reported where available; NULL when the provider omits usage.
    input_tokens      integer,
    output_tokens     integer,
    -- Set when the turn ended in a handled failure, so an error is visible in
    -- history instead of looking like a normal answer.
    error_code        text,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON messages (conversation_id, created_at);
-- Backs the per-chatbot and per-user request limits in helpers/conversations.py.
CREATE INDEX IF NOT EXISTS messages_owner_created_idx
    ON messages (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS messages_chatbot_created_idx
    ON messages (chatbot_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- 3. updated_at maintenance (set_updated_at() is created in 001_init.sql)
-- ---------------------------------------------------------------------------
DROP TRIGGER IF EXISTS conversations_set_updated_at ON conversations;
CREATE TRIGGER conversations_set_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- 4. Full-text index on website_pages
-- ----------------------------------------------------------------------------
-- Phase 2 retrieval ranks in Python: a chatbot holds at most 20 pages, so the
-- included set is loaded and scored in memory, which keeps ranking identical
-- across PostgreSQL versions and unit-testable without a database.
-- This index is the scale-up path — once a knowledge base outgrows in-memory
-- scoring, helpers/retrieval.py can push a first-stage filter down to
-- PostgreSQL without a migration. Creating it now costs one index.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS website_pages_fts_idx
    ON website_pages
    USING gin (
        to_tsvector(
            'english',
            coalesce(title, '') || ' ' || coalesce(content, '')
        )
    );

-- ---------------------------------------------------------------------------
-- 5. Row Level Security (defence-in-depth — see header note)
-- ---------------------------------------------------------------------------
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages      ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF to_regprocedure('auth.uid()') IS NULL THEN
        RAISE NOTICE 'auth.uid() not found - skipping RLS policies (non-Supabase database).';
        RETURN;
    END IF;

    DROP POLICY IF EXISTS conversations_owner ON conversations;
    CREATE POLICY conversations_owner ON conversations
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());

    DROP POLICY IF EXISTS messages_owner ON messages;
    CREATE POLICY messages_owner ON messages
        FOR ALL USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());
END $$;

-- ---------------------------------------------------------------------------
-- 6. Applied-migration bookkeeping
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version)
VALUES ('002_chat')
ON CONFLICT (version) DO NOTHING;
