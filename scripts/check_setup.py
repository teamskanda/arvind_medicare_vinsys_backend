"""Pre-demo preflight check.

Run this once credentials are in place, BEFORE the demo:

    python scripts/check_setup.py

It verifies, in order: environment variables, database connectivity, that
migrations have been applied, that shared demo authentication is configured,
and that the Firecrawl key is accepted when supplied. Each check prints PASS or FAIL with
what to do about it. Nothing is written to the database.

The Firecrawl check spends 1 credit (it scrapes one small page) so that a
bad key is caught here rather than in front of the client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import httpx  # noqa: E402
from sqlalchemy import text  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

failures = 0
warnings = 0


def report(status: str, label: str, detail: str = "") -> None:
    global failures, warnings
    if status == FAIL:
        failures += 1
    elif status == WARN:
        warnings += 1
    line = f"[{status}] {label}"
    print(line)
    if detail:
        for raw in detail.strip().splitlines():
            print(f"       {raw}")


async def check_environment():
    try:
        from config import get_settings

        settings = get_settings()
    except Exception as exc:
        report(
            FAIL,
            "Environment variables",
            f"{exc}\nCopy .env.example to .env and fill in DATABASE_URL, "
            "and enable demo authentication if needed.",
        )
        return None

    report(PASS, "Environment variables loaded")

    if not settings.firecrawl_api_key.startswith("fc-"):
        report(
            WARN,
            "FIRECRAWL_API_KEY format",
            "Firecrawl keys normally start with 'fc-'. Double-check the value.",
        )
    if "YOUR-PASSWORD" in settings.database_url:
        report(
            FAIL,
            "Placeholder values still present",
            "DATABASE_URL still contains an example placeholder.",
        )
    return settings


async def check_database(settings) -> None:
    from helpers.database import dispose_engine, get_engine

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            version = (await conn.execute(text("SELECT version()"))).scalar_one()
            report(PASS, "Database connection", str(version)[:72])

            rows = await conn.execute(
                text(
                    """
                    SELECT t.name, to_regclass('public.' || t.name) IS NOT NULL AS present
                    FROM (VALUES ('profiles'), ('chatbots'),
                                 ('website_pages'), ('crawl_jobs'),
                                 ('conversations'), ('messages'),
                                 ('chatbot_assets'), ('chatbot_screenshots')) AS t(name)
                    """
                )
            )
            missing = [row.name for row in rows if not row.present]

        if missing:
            if set(missing) == {"chatbot_screenshots"}:
                phase = "005_screenshots"
            elif set(missing) == {"chatbot_assets"}:
                phase = "004_customization"
            elif set(missing) <= {"conversations", "messages"}:
                phase = "002_chat"
            else:
                phase = "001_init"
            report(
                FAIL,
                "Migrations applied",
                "Missing tables: "
                + ", ".join(missing)
                + f"\n       (from migration {phase})"
                + "\nRun: python scripts/migrate.py",
            )
        else:
            report(PASS, "Migrations applied (8 application tables present)")

        # Phase 3 adds columns rather than tables, so the table check above
        # cannot detect a missing 003_deployment.
        async with engine.connect() as conn:
            rows = await conn.execute(
                text(
                    """
                    SELECT c.name,
                           EXISTS (
                               SELECT 1 FROM information_schema.columns
                               WHERE table_schema = 'public'
                                 AND table_name = 'chatbots'
                                 AND column_name = c.name
                           ) AS present
                    FROM (VALUES ('published_at'), ('unpublished_at'),
                                 ('allowed_domains'), ('deployment_config'))
                         AS c(name)
                    """
                )
            )
            missing_columns = [row.name for row in rows if not row.present]

        if missing_columns:
            report(
                FAIL,
                "Phase 3 deployment columns",
                "Missing on chatbots: "
                + ", ".join(missing_columns)
                + "\n       (from migration 003_deployment)"
                + "\nRun: python scripts/migrate.py",
            )
        else:
            report(PASS, "Phase 3 deployment columns present")
    except Exception as exc:
        report(
            FAIL,
            "Database connection",
            f"{type(exc).__name__}: {exc}\n"
            "Check DATABASE_URL, verified TLS, and URL-encoding of the password.",
        )
    finally:
        await dispose_engine()


async def check_demo_auth(settings) -> None:
    from helpers.auth import demo_available
    if not settings.demo_auth_enabled:
        report(WARN, "Demo login disabled", "Set DEMO_AUTH_ENABLED=true only for your shared demo.")
    elif not demo_available():
        report(FAIL, "Demo signing secret", "Set DEMO_AUTH_SECRET to a random string of at least 32 characters.")
    else:
        report(PASS, "Shared demo login configured", "No external authentication service is required.")


async def check_firecrawl(settings) -> None:
    """Scrape one tiny page. Costs 1 credit and proves the key works."""
    if not settings.firecrawl_api_key:
        report(WARN, "Website scraping disabled", "FIRECRAWL_API_KEY is not configured; demo login still works.")
        return
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{settings.firecrawl_base_url}/scrape",
                headers={
                    "Authorization": f"Bearer {settings.firecrawl_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": "https://example.com",
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
            )
    except Exception as exc:
        report(
            FAIL,
            "Firecrawl API",
            f"{type(exc).__name__}: {exc}\nCheck network access to api.firecrawl.dev.",
        )
        return

    if response.status_code == 200:
        report(PASS, "Firecrawl API key accepted (1 credit used)")
        return
    if response.status_code in (401, 403):
        report(FAIL, "Firecrawl API key rejected", "Check FIRECRAWL_API_KEY.")
        return
    if response.status_code == 402:
        report(
            FAIL,
            "Firecrawl credits exhausted",
            "The account has no credits left. Top up before the demo.",
        )
        return
    if response.status_code == 429:
        report(WARN, "Firecrawl rate limited", "The key works but is throttled right now.")
        return

    detail = response.text[:200]
    report(FAIL, f"Firecrawl returned HTTP {response.status_code}", detail)


def check_deployment(settings) -> None:
    """Phase 3 configuration. Nothing here contacts a service.

    These are the settings that decide what URL a customer is told to install,
    so a wrong value here is discovered by the customer's website, not by us.
    """
    base = settings.public_widget_base_url
    report(PASS, "Public deployment base URL", base)
    print(f"       Embed URL pattern:  {base}/embed/<public_id>")
    print(f"       Widget script:      {base}/w/<public_id>.js")

    if settings.environment != "development" and base.startswith("http://"):
        report(
            WARN,
            "Deployment URL is not HTTPS",
            "Browsers refuse to load an insecure iframe inside an HTTPS page, "
            "so the widget will not appear on any real customer website.",
        )
    if "localhost" in base and settings.environment != "development":
        report(
            WARN,
            "Deployment URL points at localhost",
            "Installed snippets would only work on the machine running this "
            "server. Set PUBLIC_WIDGET_BASE_URL to the public chat domain.",
        )

    if settings.visitor_session_secret:
        report(PASS, "VISITOR_SESSION_SECRET is set")
    else:
        report(
            WARN,
            "VISITOR_SESSION_SECRET is not set",
            "A random per-process key will be used, so public chat sessions "
            "break on every restart and do not work across multiple workers. "
            "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\"",
        )

    if settings.publish_requires_approval:
        allowed = sorted(settings.publish_allowed_email_set)
        if allowed:
            report(
                PASS,
                "Publishing is limited to approved accounts",
                f"{len(allowed)} account(s): " + ", ".join(allowed),
            )
        else:
            report(
                FAIL,
                "Publishing is blocked for everyone",
                "PUBLISH_REQUIRES_APPROVAL is true but PUBLISH_ALLOWED_EMAILS is "
                "empty, so nobody can publish a chatbot. Add the demo account's "
                "email address, or set PUBLISH_REQUIRES_APPROVAL=false.",
            )
    else:
        report(
            WARN,
            "Publishing is open to every account",
            "Any signed-in customer can publish any chatbot publicly. That is "
            "intended only once domain ownership verification exists.",
        )


def check_customization(settings) -> None:
    """Phase 4 defaults. Nothing here contacts a service.

    These are only the starting points: every one of them is overridable per
    chatbot from the customisation studio, and the studio's own catalogue of
    valid choices comes from helpers/customization.py.
    """
    from helpers.customization import ASSET_MAX_BYTES, FONTS, options
    from models.customization import is_hex_color

    catalogue = options()
    report(
        PASS,
        "Customisation studio",
        f"{len(catalogue.fonts)} fonts, {len(catalogue.tones)} tone presets, "
        f"{len(catalogue.launcher_icons)} launcher icons, "
        f"{len(catalogue.color_presets)} colour presets",
    )
    print(
        f"       Launcher default:   {settings.widget_launcher_size}px "
        f"{settings.widget_launcher_shape} at {settings.widget_launcher_position}"
    )
    print(f"       Brand colour:       {settings.widget_accent_color}")
    print(f"       Image upload limit: {ASSET_MAX_BYTES // 1024} KB per image")

    if not is_hex_color(settings.widget_accent_color):
        report(
            FAIL,
            "WIDGET_ACCENT_COLOR is not a hex colour",
            "It is interpolated into generated CSS, so only #rgb or #rrggbb "
            "is accepted. Every chatbot without its own brand colour uses it.",
        )

    if settings.widget_open_animation not in {"scale", "slide", "fade", "none"}:
        report(FAIL, "WIDGET_OPEN_ANIMATION is not a known animation")

    # Served from PUBLIC_API_BASE_URL, inside a page served from the widget
    # domain, on top of the customer's HTTPS website.
    if (
        settings.environment != "development"
        and settings.public_api_base_url.startswith("http://")
    ):
        report(
            WARN,
            "Uploaded images would be served over plain HTTP",
            "Browsers block mixed content, so avatars and logos will not "
            "appear on an HTTPS customer website. Set PUBLIC_API_BASE_URL to "
            "the HTTPS address of this API.",
        )

    if FONTS:
        report(
            PASS,
            "Fonts need no third party",
            "Every font option is device-resident or self-hosted, so an "
            "embedded chatbot makes no request to another company's CDN.",
        )


async def main() -> int:
    # ASCII only: the default Windows console code page mangles em-dashes.
    print("SKANDA AI - setup check\n" + "-" * 48)

    settings = await check_environment()
    if settings is None:
        print("-" * 48)
        print("\n1 check failed. Fix the environment first, then re-run.")
        return 1

    await check_database(settings)
    await check_demo_auth(settings)
    check_deployment(settings)
    check_customization(settings)
    await check_firecrawl(settings)

    print("-" * 48)
    if failures:
        print(f"\n{failures} check(s) FAILED, {warnings} warning(s). "
              "Fix the failures above before the demo.")
        return 1
    if warnings:
        print(f"\nAll required checks passed, with {warnings} warning(s). "
              "Start the API with: uvicorn main:app --reload --port 8000")
        return 0
    print("\nAll checks passed. Start the API with: "
          "uvicorn main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
