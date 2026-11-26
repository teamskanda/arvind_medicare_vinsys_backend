-- ============================================================================
-- SKANDA AI — Phase 3 schema: publication and public deployment
-- ----------------------------------------------------------------------------
-- Idempotent: safe to run repeatedly. Run via `python scripts/migrate.py`
-- or paste into the Supabase SQL editor.
--
-- Design note — why no new tables:
--
--   * Publication state extends `chatbots.status`, which already carried
--     'draft' | 'ready' | 'published'. Adding 'unpublished' to that CHECK is
--     the whole lifecycle; a second status column would be two sources of
--     truth for one fact.
--   * The published configuration goes in `chatbots.published_config`, which
--     001_init.sql already created for exactly this purpose.
--   * Public visitor conversations reuse `conversations` and `messages`.
--     `conversations.source` already distinguishes 'playground' from 'embed',
--     and `conversations.session_id` already holds an opaque, non-foreign-key
--     session handle — which is precisely what an anonymous visitor needs.
--   * Visitor sessions themselves are stateless: the token is HMAC-signed
--     (helpers/visitor_session.py), so there is nothing to store and nothing
--     to clean up. The signed session id becomes `conversations.session_id`.
--
-- So Phase 3 adds four columns and two indexes, and no tables at all.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Publication lifecycle on chatbots
-- ---------------------------------------------------------------------------
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS published_at   timestamptz;
ALTER TABLE chatbots ADD COLUMN IF NOT EXISTS unpublished_at timestamptz;

-- Embedding origins the customer has authorised. Empty array = no restriction
-- beyond our own checks; a non-empty array is enforced on every public
-- request. jsonb rather than text[] to match the other config columns.
ALTER TABLE chatbots
    ADD COLUMN IF NOT EXISTS allowed_domains jsonb NOT NULL DEFAULT '[]'::jsonb;

-- Launcher appearance and placement: icon, colour, size, position, offsets,
-- chat window dimensions. Read by helpers/widget.py with defaults from
-- config.py, so an empty object is always valid. Phase 4's customisation
-- studio writes it; Phase 3 only consumes it.
ALTER TABLE chatbots
    ADD COLUMN IF NOT EXISTS deployment_config jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Extend the existing lifecycle CHECK with 'unpublished'. Dropping and
-- recreating the constraint is the only way to widen it; the name below is
-- PostgreSQL's default for the inline CHECK in 001_init.sql.
DO $$
BEGIN
    ALTER TABLE chatbots DROP CONSTRAINT IF EXISTS chatbots_status_check;
    ALTER TABLE chatbots ADD CONSTRAINT chatbots_status_check
        CHECK (status IN ('draft', 'ready', 'published', 'unpublished'));
END $$;

-- The public embed route looks a chatbot up by public_id and must then reject
-- anything that is not published. public_id is already UNIQUE (001_init.sql);
-- this partial index keeps the published lookup cheap as the table grows.
CREATE INDEX IF NOT EXISTS chatbots_published_idx
    ON chatbots (public_id)
    WHERE status = 'published';

-- ---------------------------------------------------------------------------
-- 2. Public visitor traffic on conversations
-- ---------------------------------------------------------------------------
-- Public conversations are rows with source = 'embed'. This index backs the
-- per-chatbot public rate limit and the "recent visitor activity" lookups
-- without scanning playground traffic.
CREATE INDEX IF NOT EXISTS conversations_chatbot_source_idx
    ON conversations (chatbot_id, source, created_at DESC);

-- ---------------------------------------------------------------------------
-- 3. Row Level Security
-- ---------------------------------------------------------------------------
-- No change is needed. Public visitor rows are written by the backend as the
-- database owner (which bypasses RLS) and carry the chatbot owner's
-- `owner_id`, so the existing owner policies continue to describe exactly who
-- may read them through Supabase: the customer, and nobody else. A visitor
-- never holds a Supabase JWT and never reaches the database directly.
--
-- Visitor-to-visitor isolation is enforced in helpers/conversations.py: a
-- conversation is only ever found by (chatbot_id, session_id), and the
-- session_id comes from a signed token the visitor cannot forge.

-- ---------------------------------------------------------------------------
-- 4. Applied-migration bookkeeping
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version)
VALUES ('003_deployment')
ON CONFLICT (version) DO NOTHING;
