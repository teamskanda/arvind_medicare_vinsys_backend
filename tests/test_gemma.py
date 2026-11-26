"""The Gemma provider client, against a mock transport.

Both request dialects are exercised end to end without a network: request
shaping, response parsing, streaming, retries and error mapping. What these
tests cannot prove is which provider the operator configures — that is what
`scripts/check_gemma.py` is for.

The error-mapping tests double as a secrets check: a provider body containing
a key must never reach the message a customer sees.
"""

from __future__ import annotations

import json

import httpx
import pytest

from config import Settings, get_settings
from helpers import errors, gemma
from helpers.gemma import (
    build_google_payload,
    build_openai_payload,
    extract_stream_delta,
    extract_stream_usage,
    parse_google_response,
    parse_openai_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def settings_with(**overrides) -> Settings:
    base = get_settings()
    fields = {
        "database_url": base.database_url,
        "firecrawl_api_key": base.firecrawl_api_key,
        "gemma_api_key": "test-key-abcdef123456",
        "gemma_api_flavor": "google",
        "gemma_api_url": "https://generativelanguage.googleapis.com/v1beta",
        "gemma_model": "gemma-3-27b-it",
        "gemma_api_key_required": True,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
async def reset_client():
    """Each test installs its own transport, so the shared client must go."""
    await gemma.close_client()
    yield
    await gemma.close_client()


def install_transport(monkeypatch, handler, **setting_overrides) -> list[httpx.Request]:
    """Point the module at a mock transport and record the requests it makes.

    `get_client` is async and `get_settings` is sync, so each needs its own
    stub shape.
    """
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    resolved = settings_with(**setting_overrides)

    async def fake_get_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr(gemma, "get_client", fake_get_client)
    monkeypatch.setattr(gemma, "get_settings", lambda: resolved)
    return seen


def google_ok(text: str = "READY", *, finish: str = "STOP") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 7,
                "totalTokenCount": 127,
            },
            "modelVersion": "gemma-3-27b-it",
        },
    )


def openai_ok(text: str = "READY", *, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "model": "google/gemma-3-27b-it",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text},
                 "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
        },
    )


# ---------------------------------------------------------------------------
# Payload shaping
# ---------------------------------------------------------------------------


def test_google_payload_uses_contents_and_parts():
    payload = build_google_payload(
        [("user", "hello")], temperature=0.2, max_output_tokens=500
    )
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert payload["generationConfig"]["temperature"] == 0.2
    assert payload["generationConfig"]["maxOutputTokens"] == 500


def test_google_payload_renames_the_assistant_role_to_model():
    payload = build_google_payload(
        [("user", "q"), ("assistant", "a"), ("user", "q2")],
        temperature=0.2, max_output_tokens=10,
    )
    assert [item["role"] for item in payload["contents"]] == ["user", "model", "user"]


def test_google_payload_omits_system_instruction_when_merged():
    """Gemma has no system role; merge mode must send no systemInstruction."""
    payload = build_google_payload(
        [("user", "hello")], system_text="", temperature=0.2, max_output_tokens=10
    )
    assert "systemInstruction" not in payload


def test_google_payload_includes_system_instruction_when_asked():
    payload = build_google_payload(
        [("user", "hello")], system_text="Be brief.", temperature=0.2, max_output_tokens=10
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "Be brief."}]}


def test_openai_payload_shape():
    payload = build_openai_payload(
        [("user", "hello")], model="google/gemma-3-27b-it",
        temperature=0.2, max_output_tokens=500,
    )
    assert payload["model"] == "google/gemma-3-27b-it"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 500
    assert "stream" not in payload


def test_openai_payload_prepends_the_system_message():
    payload = build_openai_payload(
        [("user", "hello")], model="m", system_text="Be brief.",
        temperature=0.2, max_output_tokens=10,
    )
    assert payload["messages"][0] == {"role": "system", "content": "Be brief."}


def test_openai_stream_payload_requests_usage():
    payload = build_openai_payload(
        [("user", "hi")], model="m", temperature=0.2, max_output_tokens=10, stream=True
    )
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_google_response():
    result = parse_google_response(google_ok("Hello there").json())
    assert result.text == "Hello there"
    assert result.finish_reason == "STOP"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 7
    assert result.model == "gemma-3-27b-it"


def test_parse_google_response_concatenates_several_parts():
    body = {
        "candidates": [
            {"content": {"parts": [{"text": "one "}, {"text": "two"}]}, "finishReason": "STOP"}
        ]
    }
    assert parse_google_response(body).text == "one two"


def test_parse_google_response_with_no_candidates_is_empty_not_an_error():
    result = parse_google_response({})
    assert result.text == ""
    assert result.usage.reported is False


def test_parse_google_response_surfaces_a_blocked_prompt():
    result = parse_google_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert result.text == ""
    assert result.finish_reason == "BLOCKED_SAFETY"


def test_parse_google_response_detects_a_length_cutoff():
    assert parse_google_response(google_ok(finish="MAX_TOKENS").json()).truncated_by_length


def test_parse_openai_response():
    result = parse_openai_response(openai_ok("Hello there").json())
    assert result.text == "Hello there"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 7


def test_parse_openai_response_without_usage():
    body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    result = parse_openai_response(body)
    assert result.text == "hi"
    assert result.usage.reported is False


def test_parse_openai_response_detects_a_length_cutoff():
    assert parse_openai_response(openai_ok(finish="length").json()).truncated_by_length


def test_usage_ignores_non_numeric_values():
    body = {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": "lots"}}
    assert parse_openai_response(body).usage.input_tokens is None


# ---------------------------------------------------------------------------
# Streaming chunk parsing
# ---------------------------------------------------------------------------


def test_google_stream_delta():
    chunk = {"candidates": [{"content": {"parts": [{"text": "abc"}]}}]}
    assert extract_stream_delta(chunk, "google") == "abc"


def test_openai_stream_delta():
    chunk = {"choices": [{"delta": {"content": "abc"}}]}
    assert extract_stream_delta(chunk, "openai") == "abc"


def test_stream_delta_of_an_empty_chunk_is_empty():
    assert extract_stream_delta({}, "google") == ""
    assert extract_stream_delta({"choices": [{"delta": {}}]}, "openai") == ""


def test_stream_usage_is_extracted_per_dialect():
    google = extract_stream_usage({"usageMetadata": {"promptTokenCount": 5}}, "google")
    assert google.input_tokens == 5
    openai = extract_stream_usage({"usage": {"completion_tokens": 9}}, "openai")
    assert openai.output_tokens == 9


# ---------------------------------------------------------------------------
# generate() over the transport
# ---------------------------------------------------------------------------


async def test_generate_google_hits_the_generate_content_endpoint(monkeypatch):
    seen = install_transport(
        monkeypatch, lambda request: google_ok(), gemma_api_flavor="google",
        gemma_api_url="https://generativelanguage.googleapis.com/v1beta",
        gemma_model="gemma-3-27b-it",
    )
    result = await gemma.generate([("user", "hi")])

    assert result.text == "READY"
    assert len(seen) == 1
    assert seen[0].url.path == "/v1beta/models/gemma-3-27b-it:generateContent"
    assert seen[0].headers["x-goog-api-key"] == "test-key-abcdef123456"
    assert "authorization" not in seen[0].headers


async def test_generate_openai_hits_chat_completions(monkeypatch):
    seen = install_transport(
        monkeypatch, lambda request: openai_ok(), gemma_api_flavor="openai",
        gemma_api_url="https://openrouter.ai/api/v1",
        gemma_model="google/gemma-3-27b-it",
    )
    result = await gemma.generate([("user", "hi")])

    assert result.text == "READY"
    assert seen[0].url.path == "/api/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key-abcdef123456"
    assert "x-goog-api-key" not in seen[0].headers


async def test_generate_raises_when_no_key_is_configured(monkeypatch):
    install_transport(monkeypatch, lambda request: google_ok(), gemma_api_key=None)
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_not_configured"
    assert caught.value.status_code == 503


async def test_a_local_endpoint_needs_no_key(monkeypatch):
    """Ollama and vLLM are legitimately unauthenticated."""
    install_transport(
        monkeypatch, lambda request: openai_ok(), gemma_api_key=None,
        gemma_api_flavor="openai", gemma_api_url="http://localhost:11434/v1",
    )
    assert (await gemma.generate([("user", "hi")])).text == "READY"


async def test_a_remote_endpoint_needs_a_key_by_default(monkeypatch):
    seen = install_transport(
        monkeypatch, lambda request: openai_ok(), gemma_api_key=None,
        gemma_api_flavor="openai", gemma_api_url="http://model.example.com:8049/v1",
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_not_configured"
    assert seen == []


async def test_an_explicitly_keyless_remote_endpoint_uses_no_auth_header(monkeypatch):
    model = "google/gemma-4-26B-A4B-it"
    seen = install_transport(
        monkeypatch, lambda request: openai_ok(), gemma_api_key=None,
        gemma_api_flavor="openai", gemma_api_url="http://model.example.com:8049/v1",
        gemma_model=model, gemma_api_key_required=False,
    )
    assert (await gemma.generate([("user", "hi")])).text == "READY"
    assert str(seen[0].url) == "http://model.example.com:8049/v1/chat/completions"
    assert json.loads(seen[0].content)["model"] == model
    assert "authorization" not in seen[0].headers
    assert "x-goog-api-key" not in seen[0].headers


async def test_keyless_opt_in_does_not_disable_google_auth(monkeypatch):
    seen = install_transport(
        monkeypatch, lambda request: google_ok(), gemma_api_key=None,
        gemma_api_key_required=False,
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_not_configured"
    assert seen == []


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


async def test_a_transient_503_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream briefly unavailable")
        return google_ok()

    install_transport(monkeypatch, handler, gemma_max_retries=2)
    monkeypatch.setattr(gemma.asyncio, "sleep", _no_sleep)

    assert (await gemma.generate([("user", "hi")])).text == "READY"
    assert calls["n"] == 2


async def test_a_400_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    install_transport(monkeypatch, handler, gemma_max_retries=2)
    with pytest.raises(errors.AppError):
        await gemma.generate([("user", "hi")])
    assert calls["n"] == 1, "a client error must not be retried"


async def test_retries_are_bounded(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="still down")

    install_transport(monkeypatch, handler, gemma_max_retries=2)
    monkeypatch.setattr(gemma.asyncio, "sleep", _no_sleep)

    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert calls["n"] == 3  # 1 attempt + 2 retries
    assert caught.value.code == "ai_unavailable"


async def test_a_timeout_maps_to_a_timeout_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    install_transport(monkeypatch, handler, gemma_max_retries=0)
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_timeout"
    assert caught.value.status_code == 504


async def _no_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# Error mapping — and the secrets check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, "ai_unavailable"),
        (403, "ai_unavailable"),
        (402, "ai_unavailable"),
        (404, "ai_unavailable"),
        (400, "ai_unavailable"),
    ],
)
async def test_provider_errors_map_to_app_errors(monkeypatch, status, expected_code):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(status, json={"error": {"message": "nope"}}),
        gemma_max_retries=0,
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == expected_code


async def test_a_context_overflow_is_identified_specifically(monkeypatch):
    """The customer gets an actionable message, not a generic failure."""
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            400,
            json={"error": {"message": "The input token count exceeds the maximum context length"}},
        ),
        gemma_max_retries=0,
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_context_exceeded"


async def test_the_provider_body_never_reaches_the_customer_message(monkeypatch):
    """A provider echoing our key back must not leak it into the response."""
    leaky = {
        "error": {
            "message": "API key test-key-abcdef123456 is invalid",
            "trace": "Traceback (most recent call last): ...",
        }
    }
    install_transport(
        monkeypatch, lambda request: httpx.Response(401, json=leaky), gemma_max_retries=0
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])

    message = caught.value.message
    assert "test-key-abcdef123456" not in message
    assert "Traceback" not in message
    assert "API key" not in message


async def test_a_non_json_body_is_handled(monkeypatch):
    install_transport(
        monkeypatch, lambda request: httpx.Response(200, text="<html>oops</html>"),
        gemma_max_retries=0,
    )
    with pytest.raises(errors.AppError) as caught:
        await gemma.generate([("user", "hi")])
    assert caught.value.code == "ai_unavailable"


# ---------------------------------------------------------------------------
# Streaming over the transport
# ---------------------------------------------------------------------------


def sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


async def test_streaming_google_yields_deltas_then_usage(monkeypatch):
    stream = sse(
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        {"usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2}},
    )
    seen = install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=stream,
                                       headers={"content-type": "text/event-stream"}),
        gemma_api_flavor="google",
    )

    deltas, usage = [], None
    async for text, reported in gemma.stream_generate([("user", "hi")]):
        if text:
            deltas.append(text)
        if reported is not None:
            usage = reported

    assert "".join(deltas) == "Hello"
    assert usage is not None and usage.input_tokens == 11
    assert seen[0].url.query == b"alt=sse"
    assert ":streamGenerateContent" in str(seen[0].url)


async def test_streaming_openai_yields_deltas(monkeypatch):
    stream = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}},
    )
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=stream,
                                       headers={"content-type": "text/event-stream"}),
        gemma_api_flavor="openai",
    )

    deltas = [text async for text, _ in gemma.stream_generate([("user", "hi")]) if text]
    assert "".join(deltas) == "Hello"


async def test_a_malformed_sse_line_is_skipped(monkeypatch):
    body = (
        b"data: not json at all\n\n"
        b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        b": a comment line\n\n"
        b"data: [DONE]\n\n"
    )
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=body,
                                       headers={"content-type": "text/event-stream"}),
        gemma_api_flavor="openai",
    )
    deltas = [text async for text, _ in gemma.stream_generate([("user", "hi")]) if text]
    assert deltas == ["ok"]


async def test_a_streaming_error_status_raises_before_any_delta(monkeypatch):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}),
        gemma_api_flavor="openai",
    )
    with pytest.raises(errors.AppError) as caught:
        async for _ in gemma.stream_generate([("user", "hi")]):
            pass
    assert caught.value.code == "ai_unavailable"
    assert caught.value.retryable is True
