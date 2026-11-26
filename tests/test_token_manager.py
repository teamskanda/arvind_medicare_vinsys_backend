"""Token estimation and budget allocation.

The estimator must never under-count by much: under-counting overflows the
context window, which is the one failure mode that cannot be recovered from
at request time.
"""

from __future__ import annotations

import pytest

from helpers import token_manager
from helpers.token_manager import (
    build_budget,
    count_message_tokens,
    estimate_tokens,
    reclaimable_knowledge_tokens,
    trim_to_tokens,
)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0


def test_estimate_grows_with_length():
    short = estimate_tokens("The opening hours are nine to five.")
    long = estimate_tokens("The opening hours are nine to five. " * 10)
    assert long > short * 8


@pytest.mark.parametrize(
    "text",
    [
        "We offer web design, branding and SEO services to small businesses.",
        "Pricing starts at Rs. 4,999 per month for the starter plan.",
        "Contact us at hello@example.com or call +91 98765 43210.",
    ],
)
def test_prose_estimate_is_in_a_sane_range(text: str):
    """Roughly 3-5 characters per token for English prose."""
    tokens = estimate_tokens(text)
    assert len(text) / 6 <= tokens <= len(text) / 2


def test_token_dense_text_is_not_under_counted():
    """URLs and punctuation tokenise far worse than the character ratio implies.

    This is the case the word-based branch of the estimator exists for: a
    character-only estimate would under-count here, and under-counting is what
    overflows the window.
    """
    dense = "https://example.com/a/b?c=1&d=2 | {key: 'value'}, [1,2,3]; x=y+z"
    assert estimate_tokens(dense) >= len(dense) / 3.6


def test_cjk_counts_about_one_token_per_character():
    text = "営業時間は九時から五時までです"
    assert estimate_tokens(text) >= len(text)


def test_chat_template_overhead_is_charged_per_message():
    one = count_message_tokens([("user", "hello")])
    three = count_message_tokens(
        [("user", "hello"), ("assistant", "hi"), ("user", "bye")]
    )
    # Three short messages must cost meaningfully more than one, even though
    # the content is tiny — that is the per-turn template overhead.
    assert three >= one + 2 * token_manager.CHAT_TEMPLATE_TOKENS_PER_MESSAGE


def test_fixed_template_overhead_is_included():
    assert count_message_tokens([]) == token_manager.CHAT_TEMPLATE_FIXED_TOKENS


def test_per_message_cost_excludes_the_one_off_prefix():
    """Summing per-message costs must not charge the prefix once per turn.

    Charging it per turn over-estimates every conversation and silently
    shrinks how much history fits into its budget.
    """
    messages = [("user", "hello"), ("assistant", "hi"), ("user", "bye")]
    summed = sum(
        token_manager.count_single_message_tokens(content) for _role, content in messages
    )
    assert (
        count_message_tokens(messages)
        == summed + token_manager.CHAT_TEMPLATE_FIXED_TOKENS
    )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_default_budget_fits_the_window():
    """The brief's 7000/1000/1000/1000 split sums to exactly the window.

    The safety margin therefore has to be carved out of it. The four
    components are upper limits, so they scale down proportionally rather
    than overflowing — and the response reservation is never eroded.
    """
    budget = build_budget()
    budget.assert_consistent()

    assert budget.context_limit == 10_000
    assert budget.response == 1_000, "the output reservation is a floor"
    assert budget.safety_margin == 800

    # Scaled down by the margin's share, keeping the 7:1:1 proportions.
    assert 6_200 <= budget.knowledge <= 7_000
    assert 850 <= budget.system <= 1_000
    assert 850 <= budget.history <= 1_000
    assert budget.knowledge > budget.system + budget.history

    total = (
        budget.knowledge
        + budget.system
        + budget.history
        + budget.response
        + budget.safety_margin
    )
    assert total <= budget.context_limit


def test_raising_the_window_restores_the_full_components():
    """With room for the margin, the configured limits are used verbatim."""
    from config import Settings, get_settings

    original = get_settings()
    roomy = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        gemma_context_limit=16_000,
    )
    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(token_manager, "get_settings", lambda: roomy)
    try:
        budget = build_budget()
    finally:
        monkeypatch.undo()

    budget.assert_consistent()
    assert budget.knowledge == 7_000
    assert budget.system == 1_000
    assert budget.history == 1_000


def test_budget_reserves_a_safety_margin():
    budget = build_budget()
    assert budget.safety_margin > 0
    # The prompt allowance must exclude both the response and the margin.
    assert budget.prompt_allowance == (
        budget.context_limit - budget.response - budget.safety_margin
    )
    assert budget.prompt_allowance < budget.context_limit


def test_components_are_rescaled_when_they_exceed_the_window(monkeypatch):
    """Lowering the window must not silently over-allocate the components."""
    from config import Settings, get_settings

    original = get_settings()
    shrunk = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        gemma_context_limit=3_000,  # components still total 10,000
    )
    monkeypatch.setattr(token_manager, "get_settings", lambda: shrunk)

    budget = build_budget()
    budget.assert_consistent()
    assert budget.context_limit == 3_000
    assert budget.knowledge < 7_000
    assert budget.response >= 128
    assert budget.safety_margin > 0


def test_unused_system_and_history_tokens_become_knowledge():
    """Dynamic allocation: a short prompt widens the knowledge allowance."""
    budget = build_budget()
    generous = reclaimable_knowledge_tokens(budget, system_tokens=50, history_tokens=0)
    tight = reclaimable_knowledge_tokens(
        budget, system_tokens=budget.system, history_tokens=budget.history
    )
    assert generous > tight
    assert generous <= budget.prompt_allowance


def test_reclaimable_never_goes_negative():
    budget = build_budget()
    assert (
        reclaimable_knowledge_tokens(budget, system_tokens=99_999, history_tokens=99_999)
        == 0
    )


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def test_trim_leaves_short_text_untouched():
    text = "We are open from nine to five."
    assert trim_to_tokens(text, 500) == text


def test_trim_respects_the_limit():
    text = "Sentence number one. " * 200
    trimmed = trim_to_tokens(text, 50)
    assert trimmed
    assert token_manager.count_tokens(trimmed) <= 50


def test_trim_prefers_a_clean_boundary():
    text = "First sentence here. Second sentence here. Third sentence here. " * 30
    trimmed = trim_to_tokens(text, 40)
    assert trimmed.endswith((".", "!", "?")) or trimmed.endswith(("\n", " ".strip()))


def test_trim_to_zero_returns_empty():
    assert trim_to_tokens("anything at all", 0) == ""


def test_trim_of_empty_text_is_empty():
    assert trim_to_tokens("", 100) == ""
