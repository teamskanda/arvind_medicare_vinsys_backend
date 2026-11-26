-- One durable, private preview backdrop per chatbot. No expiring provider URLs.
CREATE TABLE IF NOT EXISTS chatbot_screenshots (
    chatbot_id uuid PRIMARY KEY REFERENCES chatbots(id) ON DELETE CASCADE,
    owner_id uuid NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    version uuid NOT NULL DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('capturing','ready','unavailable','removed')),
    source text NOT NULL CHECK (source IN ('automatic','manual')),
    source_url text,
    message text,
    content_type text,
    data bytea,
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (octet_length(data) <= 5242880)
);
ALTER TABLE chatbot_screenshots ENABLE ROW LEVEL SECURITY;
-- Application access is through the backend owner role and owner-scoped helpers.
-- Unprivileged database roles receive no policies or grants on screenshot bytes.
INSERT INTO schema_migrations(version) VALUES ('005_screenshots')
ON CONFLICT (version) DO NOTHING;
