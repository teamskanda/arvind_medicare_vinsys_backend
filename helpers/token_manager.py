"""Token counting and budget allocation for the Gemma context window.

Two separate concerns live here, and keeping them apart matters:

  1. **Counting.** How many tokens a string costs. Exact if a local Gemma
     `tokenizer.json` is configured, otherwise a deliberately conservative
     estimator (see `estimate_tokens`).

  2. **Budgeting.** How the window is divided between system instructions,
     retrieved website knowledge, conversation history and the response.

On the numbers: the Phase 2 brief specifies a 10,000-token window split
7,000 / 1,000 / 1,000 / 1,000. Those are upper limits, not quotas — a short
question against a three-page site spends a few hundred tokens. The real
server's context limit must be verified on the deployed model endpoint;
10,000 is our self-imposed working budget. `GEMMA_CONTEXT_LIMIT` changes it
in one place and every allocation below rescales proportionally.

Nothing here truncates stored content. Reduction happens on the *selected*
sections for one request; `website_pages.content` is never rewritten.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

# Per-message overhead from Gemma's chat template. Each turn is wrapped in
# `<start_of_turn>role\n ... <end_of_turn>\n`, which is a handful of tokens
# before any content. Counted per message so many short turns are not
# under-budgeted.
CHAT_TEMPLATE_TOKENS_PER_MESSAGE = 8
# Tokens reserved for the template's leading BOS plus the generation prompt.
CHAT_TEMPLATE_FIXED_TOKENS = 8

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CJK_RE = re.compile(r"[　-鿿가-힯豈-﫿]")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Gemma's SentencePiece vocabulary.

    Blends two signals and takes the larger, because under-counting overflows
    the context window while over-counting only wastes budget:

      * characters / 3.6 — Gemma averages a little under 4 characters per
        token on English prose;
      * words * 1.35 + punctuation — catches token-dense text (code, URLs,
        tables) where the character ratio is too optimistic.

    CJK text is counted separately at roughly one token per character.

    This is an estimate and is labelled as such everywhere it surfaces. The
    actual counts the provider reports are logged in `helpers/gemma.py` so
    drift between the two is visible rather than silent.
    """
    if not text:
        return 0

    cjk = len(_CJK_RE.findall(text))
    latin_text = _CJK_RE.sub("", text)

    char_estimate = len(latin_text) / 3.6
    words = _WORD_RE.findall(latin_text)
    # Everything that is not a word character or whitespace tends to be its
    # own token in SentencePiece.
    punctuation = len(latin_text) - sum(len(w) for w in words)
    word_estimate = len(words) * 1.35 + punctuation * 0.5

    return int(math.ceil(max(char_estimate, word_estimate))) + cjk


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    @property
    def source(self) -> str: ...


class EstimatingCounter:
    """Default counter: no model download, no network, deterministic."""

    @property
    def source(self) -> str:
        return "estimator"

    def count(self, text: str) -> int:
        return estimate_tokens(text)


class TokenizerCounter:
    """Exact counts from a local Gemma `tokenizer.json`.

    Used only when GEMMA_TOKENIZER_PATH is set and `tokenizers` is installed.
    Guarded with a lock because `tokenizers` encoders are not thread-safe and
    FastAPI calls this from a worker thread.
    """

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    @property
    def source(self) -> str:
        return "gemma-tokenizer"

    def count(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            encoding = self._tokenizer.encode(text)  # type: ignore[attr-defined]
        return len(encoding.ids)


@lru_cache(maxsize=1)
def get_token_counter() -> TokenCounter:
    """Resolve the best available counter once per process.

    Falls back to the estimator on any failure: an unreadable tokenizer file
    must degrade the token accounting, not take the chat endpoint down.
    """
    settings = get_settings()
    path = settings.gemma_tokenizer_path
    if not path:
        return EstimatingCounter()

    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]

        return TokenizerCounter(Tokenizer.from_file(path))
    except Exception as exc:
        logger.warning(
            "Could not load the Gemma tokenizer from %s (%s). "
            "Falling back to the built-in estimator.",
            path,
            exc,
        )
        return EstimatingCounter()


def count_tokens(text: str) -> int:
    return get_token_counter().count(text)


def count_single_message_tokens(content: str) -> int:
    """Cost of adding one more turn to a conversation.

    Excludes `CHAT_TEMPLATE_FIXED_TOKENS`, which is charged once for the whole
    conversation. Summing `count_message_tokens` per message instead would
    charge that prefix once per turn and systematically over-estimate — which
    silently shrinks how much history fits in its budget.
    """
    return get_token_counter().count(content) + CHAT_TEMPLATE_TOKENS_PER_MESSAGE


def count_message_tokens(messages: list[tuple[str, str]]) -> int:
    """Token cost of a whole (role, content) conversation, template included."""
    total = CHAT_TEMPLATE_FIXED_TOKENS
    for _role, content in messages:
        total += count_single_message_tokens(content)
    return total


# ---------------------------------------------------------------------------
# Budgeting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """How one request may spend the context window.

    `knowledge + system + history + response + safety_margin <= context_limit`
    is guaranteed by construction; `assert_consistent` proves it.
    """

    context_limit: int
    knowledge: int
    system: int
    history: int
    response: int
    safety_margin: int
    counter_source: str

    @property
    def prompt_allowance(self) -> int:
        """Everything we may send, i.e. the window minus output and margin."""
        return self.context_limit - self.response - self.safety_margin

    def assert_consistent(self) -> None:
        total = (
            self.knowledge + self.system + self.history + self.response + self.safety_margin
        )
        if total > self.context_limit:  # pragma: no cover - guarded at build time
            raise ValueError(
                f"Token budget over-allocates: {total} > {self.context_limit}"
            )


def build_budget() -> TokenBudget:
    """Derive the budget from settings, rescaling if the parts overflow.

    The four configured components are treated as *proportions*. If someone
    lowers GEMMA_CONTEXT_LIMIT without lowering the components (or raises the
    components past the window), each is scaled down to fit rather than
    silently overflowing at request time.
    """
    settings = get_settings()

    limit = max(1_000, int(settings.gemma_context_limit))
    margin = max(64, int(limit * max(0.0, settings.token_safety_margin_ratio)))

    response = max(128, int(settings.gemma_max_output_tokens))
    system = max(128, int(settings.token_budget_system))
    history = max(0, int(settings.token_budget_history))
    knowledge = max(256, int(settings.token_budget_knowledge))

    requested = response + system + history + knowledge + margin
    if requested > limit:
        # Scale the three prompt components; the response reservation and the
        # safety margin are floors we refuse to erode.
        fixed = response + margin
        available = max(256, limit - fixed)
        prompt_requested = system + history + knowledge
        scale = available / prompt_requested
        system = max(96, int(system * scale))
        history = int(history * scale)
        knowledge = max(128, available - system - history)

        # The documented default (7000/1000/1000/1000) sums to exactly the
        # window, so the safety margin always has to come out of it. That is
        # expected, not a misconfiguration — only a genuine over-allocation
        # beyond the margin deserves a warning.
        overflow = requested - limit
        log = logger.info if overflow <= margin else logger.warning
        log(
            "Token budget rescaled to fit GEMMA_CONTEXT_LIMIT=%s (over by %s, "
            "safety margin %s): system=%s history=%s knowledge=%s response=%s",
            limit, overflow, margin, system, history, knowledge, response,
        )

    budget = TokenBudget(
        context_limit=limit,
        knowledge=knowledge,
        system=system,
        history=history,
        response=response,
        safety_margin=margin,
        counter_source=get_token_counter().source,
    )
    budget.assert_consistent()
    return budget


def reclaimable_knowledge_tokens(budget: TokenBudget, system_tokens: int, history_tokens: int) -> int:
    """Knowledge allowance, widened by whatever the other parts did not use.

    This is the "dynamic allocation" requirement: a short persona and an empty
    history hand their unused tokens to website content instead of wasting
    them, while the total still respects `prompt_allowance`.
    """
    spare = budget.prompt_allowance - system_tokens - history_tokens
    return max(0, min(spare, max(budget.knowledge, spare)))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Cut `text` down to `max_tokens`, on a sentence or line boundary.

    Used only on content already selected for one request — never on the
    stored page. Returns "" when nothing fits.
    """
    if max_tokens <= 0 or not text:
        return ""

    counter = get_token_counter()
    if counter.count(text) <= max_tokens:
        return text

    # Binary search on characters: far cheaper than tokenising repeatedly from
    # scratch, and the counter is monotonic in length.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if counter.count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1

    clipped = text[:low]
    if not clipped.strip():
        return ""

    # Prefer a clean break so we do not end mid-sentence.
    for boundary in ("\n\n", "\n", ". ", ", ", " "):
        index = clipped.rfind(boundary)
        if index > len(clipped) * 0.6:
            return clipped[: index + len(boundary)].rstrip()
    return clipped.rstrip()
