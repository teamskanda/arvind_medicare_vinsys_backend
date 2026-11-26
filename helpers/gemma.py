"""Gemma inference client.

Provider-agnostic on purpose. Gemma is an open-weights family served by many
hosts, and which one the demo runs against depends on which key is available,
so this module speaks two request dialects over one transport:

  * ``google`` — Google AI Studio / Gemini API native REST:
      POST {GEMMA_API_URL}/models/{model}:generateContent
      POST {GEMMA_API_URL}/models/{model}:streamGenerateContent?alt=sse
      Auth: ``x-goog-api-key`` header.
  * ``openai`` — any OpenAI-compatible endpoint (OpenRouter, Together,
      DeepInfra, Groq, vLLM, Ollama, and Google's ``/v1beta/openai`` shim):
      POST {GEMMA_API_URL}/chat/completions
      Auth: ``Authorization: Bearer``.

Both dialects are exercised by unit tests against a mock transport, so the
request shaping and response parsing are verified offline. What cannot be
verified offline is which provider the operator actually configures — run
``python scripts/check_gemma.py`` to confirm the live endpoint, model id and
reported token usage before a demo.

Credentials never leave this module, and never appear in an error message
returned to a caller. Provider error bodies are logged server-side only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import get_settings
from helpers import errors
from helpers.token_manager import count_tokens
from models.message import MessageRole

logger = logging.getLogger(__name__)

# Shared client: connection reuse matters because every chat turn is a request.
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

# Provider statuses worth retrying. 429 and 5xx are transient; 4xx is not.
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 522, 524})

# Substrings providers use for "your prompt was too long". Checked
# case-insensitively against the error body so we can map to a specific,
# actionable error instead of a generic failure.
_CONTEXT_ERROR_MARKERS = (
    "context length",
    "context_length",
    "too many tokens",
    "maximum context",
    "token count",
    "input is too long",
    "prompt is too long",
    "reduce the length",
    "exceeds the maximum",
)


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                settings = get_settings()
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        settings.gemma_timeout_seconds,
                        connect=10.0,
                    ),
                    limits=httpx.Limits(
                        max_connections=20, max_keepalive_connections=10
                    ),
                    follow_redirects=False,
                )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GemmaUsage:
    """Token usage as reported by the provider. None means "not reported"."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def reported(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None


@dataclass(slots=True)
class GemmaResponse:
    text: str
    finish_reason: str | None = None
    usage: GemmaUsage = field(default_factory=GemmaUsage)
    model: str | None = None

    @property
    def truncated_by_length(self) -> bool:
        return (self.finish_reason or "").upper() in {
            "MAX_TOKENS", "LENGTH", "MAX_OUTPUT_TOKENS",
        }


# ---------------------------------------------------------------------------
# Request building — pure functions, unit-tested without a network
# ---------------------------------------------------------------------------


def _ensure_configured() -> Any:
    settings = get_settings()
    if not settings.gemma_configured:
        logger.error(
            "A chat request arrived but GEMMA_API_KEY is not set "
            "(flavor=%s, url=%s).",
            settings.gemma_api_flavor,
            settings.gemma_api_url,
        )
        raise errors.ai_not_configured()
    return settings


def build_google_payload(
    messages: list[tuple[str, str]],
    *,
    system_text: str = "",
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Gemini-API native body.

    Role names differ from the OpenAI dialect: the assistant is ``model``.
    `systemInstruction` is only included when there is system text, because
    Gemma builds may reject it — see GEMMA_SYSTEM_ROLE_MODE.
    """
    contents: list[dict[str, Any]] = []
    for role, content in messages:
        contents.append(
            {
                "role": "model" if role == MessageRole.ASSISTANT.value else "user",
                "parts": [{"text": content}],
            }
        )

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "candidateCount": 1,
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload


def build_openai_payload(
    messages: list[tuple[str, str]],
    *,
    model: str,
    system_text: str = "",
    temperature: float,
    max_output_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """OpenAI-compatible `/chat/completions` body."""
    chat: list[dict[str, str]] = []
    if system_text:
        chat.append({"role": "system", "content": system_text})
    chat.extend({"role": role, "content": content} for role, content in messages)

    payload: dict[str, Any] = {
        "model": model,
        "messages": chat,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if stream:
        payload["stream"] = True
        # Ask for usage on the final chunk; providers that do not support it
        # ignore the field.
        payload["stream_options"] = {"include_usage": True}
    return payload


def _endpoint(settings: Any, *, stream: bool) -> str:
    base = settings.gemma_api_url.rstrip("/")
    if settings.gemma_api_flavor == "google":
        method = "streamGenerateContent" if stream else "generateContent"
        suffix = "?alt=sse" if stream else ""
        return f"{base}/models/{settings.gemma_model}:{method}{suffix}"
    return f"{base}/chat/completions"


def _headers(settings: Any) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.gemma_api_flavor == "google":
        if settings.gemma_api_key:
            headers["x-goog-api-key"] = settings.gemma_api_key
    elif settings.gemma_api_key:
        headers["Authorization"] = f"Bearer {settings.gemma_api_key}"
    return headers


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_google_response(body: dict[str, Any]) -> GemmaResponse:
    """Pull text, finish reason and usage out of a `generateContent` body.

    A candidate may carry several parts; they are concatenated. A response
    blocked by the provider's safety filters has no parts at all, which
    surfaces as empty text plus a finish reason the caller can report.
    """
    candidates = body.get("candidates")
    text_parts: list[str] = []
    finish_reason: str | None = None

    if isinstance(candidates, list) and candidates:
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        finish_reason = first.get("finishReason")
        content = first.get("content")
        if isinstance(content, dict):
            for part in content.get("parts") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

    # A prompt rejected before generation reports promptFeedback instead.
    if not text_parts and not finish_reason:
        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            finish_reason = f"BLOCKED_{feedback['blockReason']}"

    usage_meta = body.get("usageMetadata")
    usage = GemmaUsage()
    if isinstance(usage_meta, dict):
        usage = GemmaUsage(
            input_tokens=_as_int(usage_meta.get("promptTokenCount")),
            output_tokens=_as_int(usage_meta.get("candidatesTokenCount")),
        )

    return GemmaResponse(
        text="".join(text_parts).strip(),
        finish_reason=finish_reason,
        usage=usage,
        model=body.get("modelVersion") if isinstance(body.get("modelVersion"), str) else None,
    )


def parse_openai_response(body: dict[str, Any]) -> GemmaResponse:
    choices = body.get("choices")
    text = ""
    finish_reason: str | None = None

    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = first.get("finish_reason")
        message = first.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            text = message["content"]
        elif isinstance(first.get("text"), str):  # legacy completions shape
            text = first["text"]

    usage_body = body.get("usage")
    usage = GemmaUsage()
    if isinstance(usage_body, dict):
        usage = GemmaUsage(
            input_tokens=_as_int(usage_body.get("prompt_tokens")),
            output_tokens=_as_int(usage_body.get("completion_tokens")),
        )

    return GemmaResponse(
        text=text.strip(),
        finish_reason=finish_reason,
        usage=usage,
        model=body.get("model") if isinstance(body.get("model"), str) else None,
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def extract_stream_delta(chunk: dict[str, Any], flavor: str) -> str:
    """Text carried by one SSE chunk, or "" if it carries none."""
    if flavor == "google":
        candidates = chunk.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            if isinstance(content, dict):
                return "".join(
                    part["text"]
                    for part in content.get("parts") or []
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
        return ""

    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
        # Some providers send the whole message on the final chunk.
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def extract_stream_usage(chunk: dict[str, Any], flavor: str) -> GemmaUsage:
    if flavor == "google":
        meta = chunk.get("usageMetadata")
        if isinstance(meta, dict):
            return GemmaUsage(
                input_tokens=_as_int(meta.get("promptTokenCount")),
                output_tokens=_as_int(meta.get("candidatesTokenCount")),
            )
        return GemmaUsage()

    usage = chunk.get("usage")
    if isinstance(usage, dict):
        return GemmaUsage(
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
        )
    return GemmaUsage()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _looks_like_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTEXT_ERROR_MARKERS)


def _raise_for_status(response: httpx.Response, body_text: str) -> None:
    """Map a provider error onto an AppError. Never leaks the provider body."""
    status = response.status_code
    if status < 400:
        return

    # Logged server-side only — this is the one place the real cause is visible.
    logger.error(
        "Gemma provider returned %s for %s: %s",
        status,
        response.request.url.path if response.request else "?",
        body_text[:600],
    )

    if _looks_like_context_overflow(body_text):
        raise errors.ai_context_exceeded()

    if status in (401, 403):
        raise errors.ai_unavailable(
            "The AI assistant could not authenticate with its provider. "
            "Please contact your administrator."
        )
    if status == 402:
        raise errors.ai_unavailable(
            "The AI assistant has run out of provider credits. "
            "Please contact your administrator."
        )
    if status == 404:
        raise errors.ai_unavailable(
            "The configured AI model is not available. Please contact your administrator."
        )
    if status == 429:
        raise errors.ai_unavailable(
            "The AI assistant is busy right now. Please try again in a moment."
        )
    if status in (400, 422):
        raise errors.ai_unavailable(
            "The AI assistant rejected that request. Please try rephrasing your question."
        )
    raise errors.ai_unavailable()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def generate(
    messages: list[tuple[str, str]],
    *,
    system_text: str = "",
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> GemmaResponse:
    """One non-streaming completion, with bounded retries on transient faults."""
    settings = _ensure_configured()
    client = await get_client()

    temperature = settings.gemma_temperature if temperature is None else temperature
    max_output = (
        settings.gemma_max_output_tokens if max_output_tokens is None else max_output_tokens
    )
    flavor = settings.gemma_api_flavor

    if flavor == "google":
        payload = build_google_payload(
            messages,
            system_text=system_text,
            temperature=temperature,
            max_output_tokens=max_output,
        )
    else:
        payload = build_openai_payload(
            messages,
            model=settings.gemma_model,
            system_text=system_text,
            temperature=temperature,
            max_output_tokens=max_output,
        )

    url = _endpoint(settings, stream=False)
    headers = _headers(settings)
    attempts = max(1, settings.gemma_max_retries + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning("Gemma request timed out (attempt %s/%s).", attempt + 1, attempts)
            if attempt + 1 >= attempts:
                raise errors.ai_timeout()
            await asyncio.sleep(_backoff(attempt))
            continue
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "Gemma request failed at the transport level (attempt %s/%s): %s",
                attempt + 1, attempts, exc,
            )
            if attempt + 1 >= attempts:
                raise errors.ai_unavailable()
            await asyncio.sleep(_backoff(attempt))
            continue

        if response.status_code in _RETRYABLE_STATUSES and attempt + 1 < attempts:
            logger.warning(
                "Gemma provider returned retryable %s (attempt %s/%s).",
                response.status_code, attempt + 1, attempts,
            )
            await asyncio.sleep(_backoff(attempt, response))
            continue

        _raise_for_status(response, response.text)

        try:
            body = response.json()
        except ValueError:
            logger.error("Gemma provider returned a non-JSON body: %s", response.text[:400])
            raise errors.ai_unavailable()

        if not isinstance(body, dict):
            logger.error("Gemma provider returned an unexpected JSON shape.")
            raise errors.ai_unavailable()

        result = (
            parse_google_response(body) if flavor == "google" else parse_openai_response(body)
        )
        _log_usage(result, messages, system_text)
        return result

    # Only reachable if every attempt was retryable.
    logger.error("Gemma request exhausted %s attempts: %s", attempts, last_error)
    raise errors.ai_unavailable()


async def stream_generate(
    messages: list[tuple[str, str]],
    *,
    system_text: str = "",
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> AsyncIterator[tuple[str, GemmaUsage | None]]:
    """Stream deltas as ``(text, usage)``; usage is non-None only at the end.

    Not retried: once bytes have been handed to the caller a retry would
    duplicate text. A failure before the first delta raises; a failure
    mid-stream ends the iterator, and the caller persists the partial answer
    (see `routers/chat.py`) rather than losing it.
    """
    settings = _ensure_configured()
    client = await get_client()

    temperature = settings.gemma_temperature if temperature is None else temperature
    max_output = (
        settings.gemma_max_output_tokens if max_output_tokens is None else max_output_tokens
    )
    flavor = settings.gemma_api_flavor

    if flavor == "google":
        payload = build_google_payload(
            messages,
            system_text=system_text,
            temperature=temperature,
            max_output_tokens=max_output,
        )
    else:
        payload = build_openai_payload(
            messages,
            model=settings.gemma_model,
            system_text=system_text,
            temperature=temperature,
            max_output_tokens=max_output,
            stream=True,
        )

    url = _endpoint(settings, stream=True)
    headers = {**_headers(settings), "Accept": "text/event-stream"}

    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                _raise_for_status(response, body)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue

                delta = extract_stream_delta(chunk, flavor)
                usage = extract_stream_usage(chunk, flavor)
                if delta:
                    yield delta, None
                if usage.reported:
                    yield "", usage
    except httpx.TimeoutException:
        logger.warning("Gemma stream timed out.")
        raise errors.ai_timeout()
    except httpx.HTTPError as exc:
        logger.warning("Gemma stream failed at the transport level: %s", exc)
        raise errors.ai_unavailable()


def _backoff(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff, honouring Retry-After when the provider sends it."""
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(10.0, max(0.5, float(header)))
            except ValueError:
                pass
    return min(8.0, 0.75 * (2**attempt))


def _log_usage(
    result: GemmaResponse, messages: list[tuple[str, str]], system_text: str
) -> None:
    """Record provider usage and how far our estimate was off.

    This is how estimator drift becomes visible: if the ratio strays from 1.0,
    the heuristic in `token_manager.estimate_tokens` needs recalibrating, or a
    local tokenizer should be configured.
    """
    estimated = count_tokens(system_text) + sum(
        count_tokens(content) for _role, content in messages
    )
    if result.usage.input_tokens:
        ratio = estimated / result.usage.input_tokens if result.usage.input_tokens else 0
        logger.info(
            "Gemma usage: input=%s output=%s (estimated input=%s, ratio=%.2f)",
            result.usage.input_tokens,
            result.usage.output_tokens,
            estimated,
            ratio,
        )
        if ratio < 0.85:
            logger.warning(
                "Token estimator is under-counting by %.0f%%. Consider setting "
                "GEMMA_TOKENIZER_PATH or lowering the token budgets.",
                (1 - ratio) * 100,
            )
    else:
        logger.info(
            "Gemma usage not reported by the provider; estimated input=%s tokens.",
            estimated,
        )
