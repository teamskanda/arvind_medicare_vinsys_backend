"""Application configuration, loaded from environment variables.

Every secret lives here and nowhere else. Nothing in this module is ever sent to
the frontend. The demo signing secret and database password remain server-side.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Core ----------------------------------------------------------
    app_name: str = "SKANDA AI API"
    environment: str = "development"
    debug: bool = False

    # ---- Database ------------------------------------------------------
    # Neon PostgreSQL URI. Accepts postgres:// or postgresql:// and is
    # normalised to the asyncpg driver below.
    database_url: str = Field(..., description="PostgreSQL connection URI")
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_echo: bool = False
    # Upper bound on any single statement. A dropped connection to a remote
    # database otherwise hangs the request until the OS gives up (~30s+).
    db_command_timeout_seconds: float = Field(default=15.0, gt=0, le=300)

    # ---- Demo authentication ------------------------------------------
    # Explicit opt-in: one shared demo identity, never customer registration.
    demo_auth_enabled: bool = False
    demo_auth_secret: str = Field(default="", repr=False)
    demo_auth_email: str = "demo@skanda.local"
    demo_auth_name: str = "Demo user"
    demo_auth_ttl_hours: int = Field(default=8, ge=1, le=24)

    # ---- Account authentication (real customer sign-up/login) ---------
    # Unset -> a random per-process key is generated, same trade-off as
    # VISITOR_SESSION_SECRET: sessions do not survive a restart. Set it in
    # production so a deploy does not log every customer out.
    account_auth_secret: str | None = Field(default=None, repr=False)
    account_auth_ttl_hours: int = Field(default=24 * 14, ge=1, le=24 * 90)
    password_min_length: int = Field(default=8, ge=8, le=128)
    # After this many wrong passwords in a row, the account is locked out
    # for login_lockout_minutes — independent of any per-IP limit, because
    # the attacker may rotate addresses but not the account they are after.
    login_lockout_threshold: int = Field(default=6, ge=3, le=20)
    login_lockout_minutes: int = Field(default=15, ge=1, le=120)
    # A coarse per-IP cap on login/signup attempts, regardless of which
    # email is targeted. Weak against a rotating address, same caveat as
    # the public chat IP limit below — the per-account lockout is the
    # control that actually holds.
    auth_ip_limit_per_minute: int = Field(default=10, ge=1, le=120)

    # ---- Firecrawl -----------------------------------------------------
    firecrawl_api_key: str = Field(default="", description="Firecrawl API key (fc-...)")
    firecrawl_base_url: str = "https://api.firecrawl.dev/v2"
    firecrawl_timeout_seconds: float = 60.0

    # ---- Crawl limits (Phase 1 demo guardrails) ------------------------
    crawl_page_limit: int = Field(default=20, ge=1, le=20)
    crawl_max_discovery_depth: int = 3
    crawl_poll_interval_seconds: float = 3.0
    crawl_max_duration_seconds: float = 420.0
    crawl_resync_stale_after_seconds: float = 15.0
    max_chatbots_per_user: int = 25

    # ---- Gemma inference (Phase 2) -------------------------------------
    # Provider-agnostic on purpose. Two request dialects are supported:
    #   "google" -> Google AI Studio / Gemini API native `:generateContent`
    #   "openai" -> any OpenAI-compatible `/chat/completions` endpoint
    #               (OpenRouter, Together, DeepInfra, Groq, vLLM, Ollama)
    # See backend/.env.example for a ready-to-paste block per provider.
    gemma_api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemma_api_key: str | None = None
    # Set false only for an OpenAI-compatible server that accepts requests
    # without credentials. Hosted providers continue to require a real key.
    gemma_api_key_required: bool = True
    gemma_model: str = "gemma-3-27b-it"
    gemma_api_flavor: str = "google"
    gemma_timeout_seconds: float = 90.0
    gemma_max_retries: int = 2
    gemma_temperature: float = 0.2
    gemma_max_output_tokens: int = 1000

    # Gemma's chat template has no system role. "merge" folds the system
    # instructions into the first user turn, which works on every provider and
    # every Gemma build; "system" sends a real system message/systemInstruction.
    gemma_system_role_mode: str = "merge"

    # Optional path to a local Gemma `tokenizer.json` for exact token counts.
    # Unset -> the calibrated estimator in helpers/token_manager.py is used.
    gemma_tokenizer_path: str | None = None

    # ---- Token budget --------------------------------------------------
    # The working context we allow ourselves. NOT the model's hard limit:
    # The configured server reports its own hard limit at /v1/models. 10,000
    # is our deliberate working budget, including the answer and safety margin.
    gemma_context_limit: int = 10_000
    token_budget_knowledge: int = 7_000
    token_budget_system: int = 1_000
    token_budget_history: int = 1_000
    # Held back from the arithmetic so estimator drift cannot overflow.
    token_safety_margin_ratio: float = 0.08

    # ---- Chat guardrails -----------------------------------------------
    chat_max_question_characters: int = 2_000
    chat_history_turns: int = 6
    chat_rate_limit_per_minute: int = 20
    chat_rate_limit_per_day: int = 300
    retrieval_max_sections: int = 12
    retrieval_min_score: float = 0.08
    # Source links shown under an answer, best match first. The model may
    # read more sections than this; only the most relevant pages are cited.
    chat_max_sources: int = Field(default=2, ge=0, le=10)
    # Clickable "you might also ask" questions generated with each answer,
    # in the same model call. 0 turns the feature off.
    chat_follow_up_questions: int = Field(default=2, ge=0, le=4)

    # ---- Public deployment (Phase 3) -----------------------------------
    # Where the universal embed page is served. In development this is the
    # Next.js dev server; in production it is the dedicated chat domain
    # (e.g. https://chat.skandaai.in). Never hardcoded anywhere else: every
    # iframe URL and install snippet is generated from this value.
    public_widget_base_url: str = "http://localhost:3000"
    # Where this API is reachable from a visitor's browser. The embed page
    # calls it directly, so it must be the public address, not an internal one.
    public_api_base_url: str = "http://localhost:8000"

    # HMAC key for anonymous visitor session tokens. Unset -> a random
    # per-process key is generated, which means visitor sessions do not
    # survive a restart. Set it in production.
    visitor_session_secret: str | None = None
    visitor_session_ttl_hours: int = 24

    # ---- Public chat guardrails ----------------------------------------
    # Lower than the authenticated playground limits: a public widget is
    # exposed to the open internet and each request costs model tokens.
    public_chat_rate_limit_per_minute: int = 8
    public_chat_rate_limit_per_day: int = 100
    public_chat_chatbot_limit_per_minute: int = 30
    public_chat_chatbot_limit_per_day: int = 500
    public_chat_ip_limit_per_minute: int = 20
    public_chat_max_message_length: int = 1_000

    # ---- Publication gate ----------------------------------------------
    # Off by default so every signed-up account can publish its chatbot. Set
    # PUBLISH_REQUIRES_APPROVAL=true to restrict publishing to the emails in
    # `publish_allowed_emails` (e.g. a controlled demo). Without approval,
    # nothing verifies that a customer owns the website they crawled.
    publish_requires_approval: bool = False
    publish_allowed_emails: str = ""

    # ---- Floating launcher defaults ------------------------------------
    # The starting point for every chatbot, overridable per chatbot through
    # `chatbots.deployment_config` — which is what Phase 4's customisation
    # studio writes. Nothing in the widget or the embed page hardcodes these.
    widget_launcher_size: int = 56
    widget_launcher_position: str = "bottom-right"
    widget_launcher_offset: int = 20
    widget_window_width: int = 400
    widget_window_height: int = 600
    widget_launcher_icon: str = "chat"
    widget_launcher_shape: str = "circle"
    widget_open_animation: str = "scale"
    widget_accent_color: str = "#7d2a47"  # SKANDA AI burgundy-600

    # ---- CORS / frontend ----------------------------------------------
    frontend_url: str = "http://localhost:3000"
    cors_allowed_origins: str = "http://localhost:3000"

    @model_validator(mode="after")
    def _production_configuration(self):
        if self.environment.lower() != "production":
            return self
        from urllib.parse import urlsplit
        import ipaddress
        for name in ("public_widget_base_url", "public_api_base_url", "frontend_url"):
            parsed = urlsplit(getattr(self, name))
            host = parsed.hostname or ""
            if parsed.scheme != "https" or not host or host == "localhost" or host.endswith((".localhost", ".local")) or parsed.username or parsed.password:
                raise ValueError(f"{name.upper()} must be a public HTTPS URL in production.")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                pass
            else:
                if not address.is_global:
                    raise ValueError(f"{name.upper()} cannot point to a private address in production.")
        if len(self.visitor_session_secret or "") < 32:
            raise ValueError("Set a persistent VISITOR_SESSION_SECRET of at least 32 characters in production.")
        if len(self.account_auth_secret or "") < 32:
            raise ValueError("Set a persistent ACCOUNT_AUTH_SECRET of at least 32 characters in production.")
        return self

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, value: str) -> str:
        """Force the asyncpg driver regardless of the scheme the user pasted."""
        for prefix in ("postgresql+asyncpg://",):
            if value.startswith(prefix):
                return value
        if value.startswith("postgres://"):
            return "postgresql+asyncpg://" + value[len("postgres://") :]
        if value.startswith("postgresql://"):
            return "postgresql+asyncpg://" + value[len("postgresql://") :]
        return value

    @field_validator(
        "gemma_api_url",
        "public_widget_base_url",
        "public_api_base_url",
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("widget_launcher_position")
    @classmethod
    def _check_position(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"bottom-right", "bottom-left"}:
            raise ValueError(
                "WIDGET_LAUNCHER_POSITION must be 'bottom-right' or 'bottom-left'."
            )
        return normalised

    @field_validator("widget_launcher_shape")
    @classmethod
    def _check_shape(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"circle", "rounded", "square"}:
            raise ValueError(
                "WIDGET_LAUNCHER_SHAPE must be 'circle', 'rounded' or 'square'."
            )
        return normalised

    @field_validator("widget_open_animation")
    @classmethod
    def _check_open_animation(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"scale", "slide", "fade", "none"}:
            raise ValueError(
                "WIDGET_OPEN_ANIMATION must be 'scale', 'slide', 'fade' or 'none'."
            )
        return normalised

    @field_validator("gemma_api_flavor")
    @classmethod
    def _check_flavor(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"google", "openai"}:
            raise ValueError(
                "GEMMA_API_FLAVOR must be 'google' (Gemini API :generateContent) "
                "or 'openai' (any OpenAI-compatible /chat/completions endpoint)."
            )
        return normalised

    @field_validator("gemma_system_role_mode")
    @classmethod
    def _check_system_mode(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"merge", "system"}:
            raise ValueError("GEMMA_SYSTEM_ROLE_MODE must be 'merge' or 'system'.")
        return normalised

    @property
    def cors_origins(self) -> list[str]:
        origins = {
            o.strip()
            for o in self.cors_allowed_origins.split(",")
            if o.strip()
        }
        origins.add(self.frontend_url.rstrip("/"))
        # The embed page is served from the widget domain and calls this API
        # from the visitor's browser, so that origin must be allowed too.
        origins.add(self.public_widget_base_url)
        return sorted(origins)

    @property
    def gemma_configured(self) -> bool:
        """Allow keyless local servers and explicitly configured remote servers."""
        return (
            bool(self.gemma_api_key)
            or self._is_local_gemma_endpoint()
            or (self.gemma_api_flavor == "openai" and not self.gemma_api_key_required)
        )

    def _is_local_gemma_endpoint(self) -> bool:
        lowered = self.gemma_api_url.lower()
        return any(
            host in lowered
            for host in ("//localhost", "//127.0.0.1", "//0.0.0.0", "//host.docker.internal")
        )

    @property
    def gemma_response_budget(self) -> int:
        return self.gemma_max_output_tokens

    # ---- Deployment URL generation -------------------------------------
    # One place builds every public URL, so changing the deployment domain is
    # a single environment variable and no stored URL goes stale.

    def embed_url(self, public_id: str, *, mode: str | None = None) -> str:
        base = f"{self.public_widget_base_url}/embed/{public_id}"
        return f"{base}?mode={mode}" if mode else base

    def widget_script_url(self, public_id: str) -> str:
        """The loader the customer pastes into their site.

        Served from the widget domain so a customer sees a single hostname.
        In development Next.js rewrites `/w/*` to this API (next.config.ts);
        in production the same path is proxied to the backend.
        """
        return f"{self.public_widget_base_url}/w/{public_id}.js"

    @property
    def publish_allowed_email_set(self) -> set[str]:
        return {
            email.strip().lower()
            for email in self.publish_allowed_emails.split(",")
            if email.strip()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
