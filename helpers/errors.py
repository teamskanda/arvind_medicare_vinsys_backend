"""The one error type the application raises.

Routers never build error payloads by hand and never leak exceptions: helpers
raise `AppError`, and the handler registered in `main.py` renders it as the
standard `{"error": {...}}` envelope. Anything unexpected becomes a generic
500 with the detail logged server-side only.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """An error that is safe to show a customer.

    `message` is written for a non-technical business owner — it must never
    contain a stack trace, SQL, a provider payload or an internal hostname.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


# ---------------------------------------------------------------------------
# Constructors for the cases the spec requires us to handle well.
# Keeping them here stops the same wording drifting across routers.
# ---------------------------------------------------------------------------


def unauthenticated(message: str = "Please sign in to continue.") -> AppError:
    return AppError("unauthenticated", message, status_code=401)


def forbidden(message: str = "You do not have access to this resource.") -> AppError:
    return AppError("forbidden", message, status_code=403)


def not_found(resource: str = "resource") -> AppError:
    return AppError(
        "not_found",
        f"That {resource} could not be found.",
        status_code=404,
    )


def invalid_url(message: str) -> AppError:
    return AppError("invalid_url", message, status_code=422)


def unsafe_url(message: str) -> AppError:
    return AppError("unsafe_url", message, status_code=422)


def site_unreachable(message: str) -> AppError:
    return AppError("site_unreachable", message, status_code=422, retryable=True)


def robots_disallowed(message: str) -> AppError:
    return AppError("robots_disallowed", message, status_code=422)


def crawl_already_running() -> AppError:
    return AppError(
        "crawl_already_running",
        "A crawl is already running for this chatbot. Wait for it to finish before starting another.",
        status_code=409,
    )


def provider_error(
    message: str, *, code: str = "provider_error", retryable: bool = True,
    status_code: int = 502,
) -> AppError:
    return AppError(code, message, status_code=status_code, retryable=retryable)


def rate_limited(
    message: str = "The scraping service is rate limiting us. Please try again in a moment.",
) -> AppError:
    return AppError("rate_limited", message, status_code=429, retryable=True)


def internal(
    message: str = "Something went wrong on our side. Please try again.",
) -> AppError:
    return AppError("internal_error", message, status_code=500, retryable=True)


# ---------------------------------------------------------------------------
# Phase 2 — AI engine
# ---------------------------------------------------------------------------


def ai_not_configured() -> AppError:
    """The operator has not set GEMMA_API_KEY. Never leaks which key is missing."""
    return AppError(
        "ai_not_configured",
        "The AI assistant is not configured yet. Please contact your administrator.",
        status_code=503,
    )


def knowledge_base_empty() -> AppError:
    return AppError(
        "knowledge_base_empty",
        "This chatbot has no website content yet. Scrape the website first, "
        "then include at least one page.",
        status_code=409,
    )


def ai_unavailable(
    message: str = "The AI assistant is temporarily unavailable. Please try again in a moment.",
) -> AppError:
    return AppError("ai_unavailable", message, status_code=503, retryable=True)


def ai_timeout() -> AppError:
    return AppError(
        "ai_timeout",
        "The AI assistant took too long to respond. Please try again.",
        status_code=504,
        retryable=True,
    )


def ai_context_exceeded() -> AppError:
    """Raised only when reduction has already been tried and still overflows."""
    return AppError(
        "ai_context_exceeded",
        "That question needed more website content than the AI can read at once. "
        "Try asking something more specific.",
        status_code=422,
    )


def chat_rate_limited(message: str) -> AppError:
    return AppError("chat_rate_limited", message, status_code=429, retryable=True)


# ---------------------------------------------------------------------------
# Phase 3 — public deployment
# ---------------------------------------------------------------------------


def chatbot_not_published() -> AppError:
    """A public request for a chatbot that is not serving.

    Covers an unknown public_id, a draft chatbot and an unpublished one, all
    with the same 404 and the same wording. Distinguishing them would let
    someone enumerate which handles exist, and the visitor-facing answer is
    identical either way: this assistant is unavailable.
    """
    return AppError(
        "chatbot_unavailable",
        "This chat assistant is not available right now.",
        status_code=404,
    )


def not_publishable(message: str) -> AppError:
    """Publishing was refused because the chatbot is not ready for visitors."""
    return AppError("not_publishable", message, status_code=409)


def publish_not_approved() -> AppError:
    """The demo-period gate from the Phase 3 brief, section 11."""
    return AppError(
        "publish_not_approved",
        "Public publishing is limited to approved accounts during the launch "
        "period. Please contact the SKANDA AI team to enable it for your account.",
        status_code=403,
    )


def embedding_not_allowed() -> AppError:
    """The request came from a website the customer has not authorised."""
    return AppError(
        "embedding_not_allowed",
        "This chat assistant is not authorised for the website it was loaded from.",
        status_code=403,
    )

