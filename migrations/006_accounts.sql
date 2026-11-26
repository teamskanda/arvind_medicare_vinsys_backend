-- ============================================================================
-- SKANDA AI — real customer accounts
-- ----------------------------------------------------------------------------
-- Until now the only identity in `profiles` was the single shared demo user
-- (see helpers/auth.py). This adds first-party email + password accounts on
-- the same table, so every other table's `owner_id -> profiles.id` foreign
-- key needs no change at all.
--
-- `password_hash` is NULL for the demo profile and for any row created before
-- this migration — it is never used to authenticate the demo flow, which
-- keeps its own signed-JWT path in helpers/auth.py untouched.
-- ============================================================================

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS password_hash text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS failed_login_attempts integer NOT NULL DEFAULT 0;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS locked_until timestamptz;

-- Phase 1 only indexed lower(email) for lookup speed; real signup requires
-- one account per email. Safe to tighten now — the only existing row is the
-- single demo profile.
DROP INDEX IF EXISTS profiles_email_idx;
CREATE UNIQUE INDEX IF NOT EXISTS profiles_email_unique ON profiles (lower(email));

-- ---------------------------------------------------------------------------
-- Applied-migration bookkeeping
-- ---------------------------------------------------------------------------
INSERT INTO schema_migrations (version)
VALUES ('006_accounts')
ON CONFLICT (version) DO NOTHING;
