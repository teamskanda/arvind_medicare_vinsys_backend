# SKANDA AI backend

FastAPI, Neon PostgreSQL, Firecrawl and Gemma backend for the SKANDA AI chatbot
builder. This repository contains the backend only. The Next.js frontend runs
separately and connects through the configured API URLs.

## Use uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The project
uses **Python 3.12**, pinned in `.python-version`, with dependencies declared in
`pyproject.toml` and reproducibly resolved in the committed `uv.lock`.

```sh
uv python install 3.12
uv sync --locked
```

The default sync includes the development group for tests. For a production-only
environment use `uv sync --locked --no-dev`. The application is not packaged as
a library; keep the current `routers/`, `models/` and `helpers/` layout.

Copy `.env.example` to `.env` and fill in your private configuration. If a real
`.env` already exists, preserve it instead of replacing it with the example.
Apply migrations and run the server from this directory:

```sh
uv run --locked python scripts/migrate.py
uv run --locked uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

On PowerShell, set `$env:DEBUG='false'` if your shell has an unrelated `DEBUG`
variable. The application expects a boolean value. uv manages `.venv`; manual
activation and `pip install` are not required.

Health: http://localhost:8000/api/health. Database readiness:
http://localhost:8000/api/ready. Interactive API docs:
http://localhost:8000/docs.

## Configuration

Configure these privately in `.env` or your hosting environment:

- `DATABASE_URL`: Neon PostgreSQL connection URI. The async driver supports
  verified TLS with `?ssl=verify-full`.
- `FIRECRAWL_API_KEY`: required for live website crawling and automatic homepage
  screenshot capture.
- Gemma endpoint/model settings: use the provider section in `.env.example`.
  OpenAI-compatible vLLM servers use `GEMMA_API_FLAVOR=openai`; the configured
  model can be `google/gemma-4-26B-A4B-it`. Explicitly keyless servers must set
  `GEMMA_API_KEY_REQUIRED=false`.
- Authentication/session settings: configure the private account signing and
  visitor-session secrets described in `.env.example`. Real account signup/login
  and explicitly enabled shared demo access use the existing auth router.
- `FRONTEND_URL`, `PUBLIC_WIDGET_BASE_URL` and `PUBLIC_API_BASE_URL`: the actual
  frontend/chat and API origins. Use HTTPS for public hosting.
- `PUBLISH_REQUIRES_APPROVAL` and `PUBLISH_ALLOWED_EMAILS`: control which accounts
  may publish during a controlled demonstration.

Keep database credentials, provider keys, passwords and session secrets out of
source control and frontend variables. The safe `.env.example` is tracked;
real environment files and local environments are ignored.

## Tests and integration checks

```sh
uv run --locked pytest -q
uv run --locked python scripts/check_setup.py
uv run --locked python scripts/check_gemma.py
```

The pytest suite uses mocked provider/database paths where appropriate and does
not require real service credentials. The setup and Gemma scripts contact the
configured real services; the Firecrawl setup check can spend a scraping credit.
Neither script is a replacement for an end-to-end browser test with the frontend.

## Dependency changes

```sh
uv add package-name
uv add --dev development-package
uv lock --check
```

Commit `pyproject.toml` and `uv.lock` together. `requirements.txt` and
`requirements-dev.txt` are compatibility exports, not separate dependency sources.
Regenerate them after dependency changes:

```sh
uv export --locked --no-dev --no-emit-project --output-file requirements.txt
uv export --locked --no-emit-project --output-file requirements-dev.txt
```

## Docker

The Dockerfile installs dependencies using pinned uv and
`uv sync --locked --no-dev`, then runs the application as an unprivileged user.

```sh
docker build -t skanda-backend .
docker run --env-file .env -p 8000:8000 skanda-backend
```

Apply migrations before serving production traffic. Configure a persistent
visitor signing secret and correct public HTTPS origins; production configuration
rejects localhost/private public URLs and missing visitor secrets. Use an HTTPS
reverse proxy with streaming enabled. Start with one worker for the controlled
demo because the additional per-IP limiter is process-local.

The shared demo flag is opt-in and must not be treated as individual customer
authentication. Current signup does not include email verification or password
recovery; configure those workflows before relying on account ownership for
sensitive production access.
