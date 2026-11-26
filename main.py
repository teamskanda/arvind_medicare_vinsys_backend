"""SKANDA AI backend entrypoint.

Run with:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings
from helpers.database import dispose_engine, verify_connection
from helpers.errors import AppError
from helpers.gemma import close_client as close_gemma_client
from models.base import ErrorBody, ErrorResponse
from routers import auth, chat, chatbots, customization, deployment, scraping, screenshots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("skanda")
# Provider image URLs may contain temporary signatures. Keep request URLs out
# of normal logs; our own lifecycle messages describe operations without keys.
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)
    # Fail fast and loudly: a misconfigured database or an unmigrated schema
    # should surface at boot, not during a live demo.
    await verify_connection()
    logger.info("Database connection verified.")
    # Not fatal: Phase 1 (auth, scraping, knowledge base) works without a
    # Gemma key, and the chat endpoints return a clear 503 until one is set.
    if settings.gemma_configured:
        logger.info(
            "Gemma configured: flavor=%s model=%s context_budget=%s",
            settings.gemma_api_flavor,
            settings.gemma_model,
            settings.gemma_context_limit,
        )
    else:
        logger.warning(
            "GEMMA_API_KEY is not set - the chatbot playground will return "
            "'AI assistant is not configured'. Run scripts/check_gemma.py once set."
        )
    try:
        yield
    finally:
        await close_gemma_client()
        await dispose_engine()
        logger.info("Shutdown complete.")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
    description=(
        "Phase 1: authentication, chatbots and website scraping. "
        "Phase 2: Gemma AI engine with keyword retrieval. "
        "Phase 3: public iframe deployment and the floating widget. "
        "Phase 4: the chatbot customisation studio."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # X-Visitor-Token carries the anonymous session on public chat requests.
    allow_headers=[
        "Authorization",
        "Content-Type",
        # Public widget: the anonymous visitor session, and the host page the
        # chatbot was embedded on (see helpers/deployment.check_embedding_origin).
        "X-Visitor-Token",
        "X-Embed-Origin",
    ],
)


# ---------------------------------------------------------------------------
# Error handling — every failure leaves as {"error": {...}}
# ---------------------------------------------------------------------------


def _error_response(
    status_code: int, code: str, message: str, *, retryable: bool = False,
    details: dict | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, retryable=retryable, details=details)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return _error_response(
        exc.status_code, exc.code, exc.message, retryable=exc.retryable, details=exc.details
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Translate Pydantic errors into one readable sentence."""
    first = exc.errors()[0] if exc.errors() else None
    field = ""
    if first and first.get("loc"):
        field = str(first["loc"][-1]).replace("_", " ")
    message = (
        f"Please check the {field} you entered." if field else "Please check the details you entered."
    )
    return _error_response(422, "validation_error", message)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = {401: "unauthenticated", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}.get(
        exc.status_code, "http_error"
    )
    message = exc.detail if isinstance(exc.detail, str) else "That request could not be completed."
    return _error_response(exc.status_code, code, message)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the detail, tell the customer nothing specific.

    This is what keeps raw exceptions, SQL and provider payloads out of the
    browser.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error_response(
        500,
        "internal_error",
        "Something went wrong on our side. Please try again.",
        retryable=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(chatbots.router)
app.include_router(scraping.router)
app.include_router(chat.router)
# Phase 3. `deployment.router` is authenticated; `public_router` and
# `widget_router` are deliberately anonymous and enforce publication status,
# visitor sessions, allowed domains and request limits themselves.
app.include_router(deployment.router)
app.include_router(deployment.public_router)
app.include_router(deployment.widget_router)
# Phase 4. `customization.router` is authenticated; `public_router` serves
# uploaded avatars and logos to visitors, who have no account.
app.include_router(customization.router)
app.include_router(customization.public_router)
app.include_router(screenshots.router)


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get('/api/ready', tags=['system'])
async def ready() -> JSONResponse:
    import asyncio
    try:
        async with asyncio.timeout(8):
            await verify_connection()
    except Exception:
        return JSONResponse({'status': 'unavailable'}, status_code=503)
    return JSONResponse({'status': 'ready'})
