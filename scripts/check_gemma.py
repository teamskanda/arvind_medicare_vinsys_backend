"""Verify the configured Gemma provider before a demo.

Run:  python scripts/check_gemma.py

Makes real requests, so it costs whatever your provider charges (on Google AI
Studio's free Gemma tier, nothing). It reports the facts the Phase 2 brief asks
us to establish rather than assume: which endpoint answers, the exact model
identifier it reports back, whether streaming works, whether a system role is
accepted, the token usage it reports, and how close our own estimator is.

Output is ASCII-only so it renders on the default Windows console code page.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config import get_settings  # noqa: E402
from helpers import gemma  # noqa: E402
from helpers.token_manager import count_tokens, estimate_tokens  # noqa: E402

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[ -- ]"

PROBE_QUESTION = (
    "Reply with exactly the word READY and nothing else."
)


def line(status: str, message: str) -> None:
    print(f"{status} {message}")


async def main() -> int:
    print("=" * 72)
    print("SKANDA AI - Gemma provider check")
    print("=" * 72)

    try:
        settings = get_settings()
    except Exception as exc:
        line(FAIL, f"Configuration could not be loaded: {exc}")
        return 1

    # ---- Configuration ------------------------------------------------
    print("\nConfiguration")
    line(INFO, f"GEMMA_API_FLAVOR   = {settings.gemma_api_flavor}")
    line(INFO, f"GEMMA_API_URL      = {settings.gemma_api_url}")
    line(INFO, f"GEMMA_MODEL        = {settings.gemma_model}")
    line(INFO, f"GEMMA_SYSTEM_ROLE_MODE = {settings.gemma_system_role_mode}")
    line(INFO, f"Context budget     = {settings.gemma_context_limit} tokens")
    line(INFO, f"Max output tokens  = {settings.gemma_max_output_tokens}")
    line(INFO, f"Temperature        = {settings.gemma_temperature}")

    if settings.gemma_api_key:
        key = settings.gemma_api_key
        masked = f"{key[:4]}...{key[-4:]}" if len(key) > 10 else "set"
        line(PASS, f"GEMMA_API_KEY is set ({masked}, {len(key)} characters)")
    elif settings.gemma_configured:
        line(PASS, "No API key set; a local or explicitly keyless server is configured")
    else:
        line(FAIL, "GEMMA_API_KEY is not set. The chat endpoints will return 503.")
        print("\nSet GEMMA_API_KEY in backend/.env, then run this script again.")
        print("See backend/.env.example for a ready-to-paste block per provider.")
        return 1

    counter_source = "gemma-tokenizer" if settings.gemma_tokenizer_path else "estimator"
    line(INFO, f"Token counting     = {counter_source}")

    # ---- Non-streaming request ----------------------------------------
    print("\nNon-streaming request")
    started = time.monotonic()
    try:
        result = await gemma.generate(
            [("user", PROBE_QUESTION)],
            temperature=0.0,
            max_output_tokens=16,
        )
    except Exception as exc:
        line(FAIL, f"Request failed: {type(exc).__name__}: {exc}")
        print(_diagnose(settings))
        await gemma.close_client()
        return 1
    elapsed = time.monotonic() - started

    if not result.text:
        line(
            WARN,
            f"The provider answered but returned no text (finish_reason="
            f"{result.finish_reason}).",
        )
    else:
        line(PASS, f"Answered in {elapsed:.2f}s: {result.text[:60]!r}")

    line(INFO, f"finish_reason      = {result.finish_reason}")
    line(
        INFO,
        f"model reported     = {result.model or 'not reported by this provider'}",
    )

    if result.usage.reported:
        estimated = estimate_tokens(PROBE_QUESTION)
        line(
            PASS,
            f"Token usage reported: input={result.usage.input_tokens} "
            f"output={result.usage.output_tokens}",
        )
        if result.usage.input_tokens:
            ratio = estimated / result.usage.input_tokens
            verdict = PASS if 0.85 <= ratio <= 2.5 else WARN
            line(
                verdict,
                f"Our estimator said {estimated} input tokens "
                f"(ratio {ratio:.2f}; >=1.0 means we over-estimate, which is safe)",
            )
    else:
        line(
            WARN,
            "This provider does not report token usage. Stored input_tokens/"
            "output_tokens will fall back to our estimate.",
        )

    # ---- Streaming -----------------------------------------------------
    print("\nStreaming request")
    chunks = 0
    streamed = ""
    try:
        async for delta, usage in gemma.stream_generate(
            [("user", "Count from one to five, words only.")],
            temperature=0.0,
            max_output_tokens=48,
        ):
            if delta:
                chunks += 1
                streamed += delta
            if usage is not None and usage.reported:
                line(INFO, f"stream usage: input={usage.input_tokens} output={usage.output_tokens}")
    except Exception as exc:
        line(
            WARN,
            f"Streaming failed ({type(exc).__name__}: {exc}). The playground "
            "falls back to non-streaming requests.",
        )
    else:
        if chunks > 1:
            line(PASS, f"Streamed {chunks} chunks: {streamed.strip()[:60]!r}")
        elif chunks == 1:
            line(
                WARN,
                "The provider returned the whole answer in one chunk - streaming "
                "works but will not look incremental.",
            )
        else:
            line(WARN, "Streaming produced no text chunks.")

    # ---- System role ---------------------------------------------------
    print("\nSystem role support")
    if settings.gemma_system_role_mode == "merge":
        line(
            INFO,
            "GEMMA_SYSTEM_ROLE_MODE=merge - instructions are folded into the "
            "first user turn. This works on every Gemma build; no probe needed.",
        )
    else:
        try:
            probe = await gemma.generate(
                [("user", "What is your name?")],
                system_text="You are called Probe. Answer with your name only.",
                temperature=0.0,
                max_output_tokens=16,
            )
        except Exception as exc:
            line(
                FAIL,
                f"A request with a system role was rejected ({type(exc).__name__}). "
                "Set GEMMA_SYSTEM_ROLE_MODE=merge in backend/.env.",
            )
            print(f"       detail: {exc}")
        else:
            if "probe" in probe.text.lower():
                line(PASS, "The system role is honoured by this provider.")
            else:
                line(
                    WARN,
                    f"The system role was accepted but seemingly ignored "
                    f"(answer: {probe.text[:40]!r}). Prefer GEMMA_SYSTEM_ROLE_MODE=merge.",
                )

    # ---- Context limit -------------------------------------------------
    print("\nContext window")
    line(
        INFO,
        f"We budget {settings.gemma_context_limit} tokens. Published windows: "
        "Gemma 3 4B/12B/27B = 128k, Gemma 4 = 256k.",
    )
    line(
        INFO,
        "The budget is a deliberate demo guardrail, not the model's limit. "
        "Raise GEMMA_CONTEXT_LIMIT if a site needs more context.",
    )
    probe_tokens = count_tokens("x " * 200)
    line(INFO, f"Sanity check: 200 repeated tokens counted as {probe_tokens}")

    await gemma.close_client()
    print("\n" + "=" * 72)
    print("Done. Record the model id and usage figures above in your report.")
    print("=" * 72)
    return 0


def _diagnose(settings) -> str:
    """Point at the most likely cause, without printing the key."""
    hints = [
        "",
        "Most common causes:",
        "  * Wrong GEMMA_API_FLAVOR for this endpoint.",
        f"      google -> {settings.gemma_api_url}/models/<model>:generateContent",
        f"      openai -> {settings.gemma_api_url}/chat/completions",
        "  * Model id not available on this provider (404). Check GEMMA_MODEL.",
        "  * Invalid or expired key (401/403).",
        "  * GEMMA_API_URL includes a trailing path it should not, or omits /v1.",
        "",
        "Known-good combinations are listed in backend/.env.example.",
    ]
    return "\n".join(hints)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except httpx.HTTPError as exc:  # pragma: no cover - top-level safety net
        print(f"{FAIL} Network error: {exc}")
        raise SystemExit(1)
