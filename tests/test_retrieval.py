"""Keyword retrieval: sectioning, keyword extraction, ranking, selection.

These are the tests that decide whether the chatbot answers from the right
page. They use realistic small-business page content rather than lorem ipsum,
because the ranking heuristics are tuned for exactly that shape of text.
"""

from __future__ import annotations

import uuid

import pytest

from helpers import retrieval
from helpers.retrieval import (
    PageRow,
    Section,
    deduplicate,
    extract_keywords,
    score_sections,
    search,
    select_within_budget,
    split_into_sections,
)


def page(url: str, title: str, content: str, priority: int = 50) -> PageRow:
    return PageRow(
        id=uuid.uuid4(), url=url, title=title, content=content, priority=priority
    )


HOME = page(
    "https://acme.test/",
    "Acme Interiors - Home",
    """
# Acme Interiors

We design and build residential interiors in Bengaluru. Founded in 2011, we
have completed over four hundred homes across Karnataka.

## Why choose us

Ten year structural warranty, in-house carpentry, and a dedicated project
manager for every home.
""",
    priority=100,
)

PRICING = page(
    "https://acme.test/pricing",
    "Pricing - Acme Interiors",
    """
# Pricing

## Packages

Our Essentials package starts at Rs. 3,50,000 for a 2BHK and covers modular
kitchen, wardrobes and false ceiling.

The Premium package starts at Rs. 7,00,000 and adds imported hardware,
custom furniture and lighting design.

## Payment terms

Ten percent on booking, fifty percent on design sign-off, and the balance on
installation. We accept bank transfer and EMI through our finance partners.
""",
    priority=88,
)

CONTACT = page(
    "https://acme.test/contact",
    "Contact - Acme Interiors",
    """
# Contact us

Our experience centre is at 14 Lavelle Road, Bengaluru 560001.

Phone: +91 80 4123 5566
Email: hello@acmeinteriors.test

We are open Monday to Saturday, ten in the morning until seven in the evening.
Sunday visits are by appointment only.
""",
    priority=84,
)

SERVICES = page(
    "https://acme.test/services",
    "Services - Acme Interiors",
    """
# Services

## Modular kitchens

Soft-close hardware, quartz counters and a choice of eleven shutter finishes.

## Wardrobes and storage

Walk-in, sliding and hinged wardrobes built to the millimetre.

## Full home interiors

Turnkey delivery in forty five working days from design sign-off.
""",
    priority=90,
)

ALL_PAGES = [HOME, PRICING, CONTACT, SERVICES]


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


def test_sections_split_on_markdown_headings():
    sections = split_into_sections(PRICING)
    headings = [section.heading for section in sections]
    assert "Packages" in headings
    assert "Payment terms" in headings


def test_sections_carry_page_identity_for_citation():
    sections = split_into_sections(CONTACT)
    assert sections
    for section in sections:
        assert section.url == CONTACT.url
        assert section.page_id == CONTACT.id
        assert section.page_title == CONTACT.title


def test_section_label_combines_page_and_heading():
    sections = split_into_sections(PRICING)
    packages = next(s for s in sections if s.heading == "Packages")
    assert packages.label == "Pricing - Acme Interiors - Packages"


def test_section_label_does_not_repeat_the_page_title():
    row = page("https://x.test/about", "About Us", "# About Us\n\n" + "Body text. " * 20)
    section = split_into_sections(row)[0]
    # Heading duplicates the title, so the label stays short.
    assert section.label == "About Us"


def test_a_very_short_page_still_produces_one_section():
    row = page(
        "https://x.test/hours",
        "Hours",
        "We are open from nine until five, Monday to Friday, all year round.",
    )
    sections = split_into_sections(row)
    assert len(sections) == 1
    assert "nine until five" in sections[0].text


def test_a_page_below_the_minimum_produces_nothing():
    row = page("https://x.test/tiny", "Tiny", "Hi.")
    assert split_into_sections(row) == []


def test_oversized_text_is_split_into_several_sections():
    body = "\n\n".join(f"Paragraph {i} about interior design work." * 12 for i in range(40))
    row = page("https://x.test/long", "Long", body)
    sections = split_into_sections(row)
    assert len(sections) > 1
    for section in sections:
        assert len(section.text) <= retrieval.MAX_SECTION_CHARACTERS + 200


def test_a_single_huge_paragraph_is_hard_split():
    row = page("https://x.test/wall", "Wall", "word " * 4000)
    sections = split_into_sections(row)
    assert len(sections) > 1


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def test_stopwords_are_dropped():
    keywords = extract_keywords("What are the opening hours of your office?")
    assert "what" not in keywords
    assert "are" not in keywords
    assert "the" not in keywords
    assert "hour" in keywords or "hours" in keywords


def test_politeness_is_dropped():
    keywords = extract_keywords("Hi, could you please tell me about your wardrobes?")
    assert "hi" not in keywords
    assert "please" not in keywords
    assert "tell" not in keywords
    assert any("wardrobe" in keyword for keyword in keywords)


def test_how_much_implies_price():
    keywords = extract_keywords("How much for a 2BHK?")
    assert "price" in keywords or "cost" in keywords


def test_synonyms_are_expanded():
    keywords = extract_keywords("What is the cost of a kitchen?")
    assert "price" in keywords or "pricing" in keywords


def test_plural_and_gerund_forms_share_a_stem():
    assert extract_keywords("wardrobes") == extract_keywords("wardrobe")


def test_an_empty_question_yields_no_keywords():
    assert extract_keywords("") == []
    assert extract_keywords("the and of") == []


def test_every_synonym_key_is_its_own_stem():
    """A key that is not a stem can never be looked up, so it is dead weight.

    Guards the subtle failure where someone adds "hours" (stem: "hour") or
    "pricing" (stem: "pric") and the expansion silently never fires.
    """
    wrong = {
        key: retrieval._stem(key)
        for key in retrieval._SYNONYMS
        if retrieval._stem(key) != key
    }
    assert not wrong, f"synonym keys must equal their own stem: {wrong}"


def test_no_synonym_value_is_a_stopword():
    """Expanding into a stopword would add a term that matches everything."""
    for key, values in retrieval._SYNONYMS.items():
        for value in values:
            assert value not in retrieval.STOPWORDS, f"{key} -> {value}"


def test_located_question_reaches_a_page_that_says_centre():
    """The expansion exists for exactly this: no shared word with the answer.

    "Where is your showroom located?" shares no keyword with "Our experience
    centre is at 14 Lavelle Road" until the synonyms bridge it.
    """
    keywords = extract_keywords("Where is your showroom located?")
    assert "centre" in keywords
    assert "road" in keywords or "address" in keywords


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _top_url(question: str, pages=None) -> str:
    rows = pages or ALL_PAGES
    sections: list[Section] = []
    for row in rows:
        sections.extend(split_into_sections(row))
    ranked = score_sections(sections, extract_keywords(question))
    return ranked[0].url


@pytest.mark.parametrize(
    ("question", "expected_url"),
    [
        ("How much does a 2BHK cost?", "https://acme.test/pricing"),
        ("What are your payment terms?", "https://acme.test/pricing"),
        ("What is your phone number?", "https://acme.test/contact"),
        ("Where is your showroom located?", "https://acme.test/contact"),
        ("What time do you open on Saturday?", "https://acme.test/contact"),
        ("Do you build modular kitchens?", "https://acme.test/services"),
        ("Tell me about your wardrobes", "https://acme.test/services"),
        ("When was the company founded?", "https://acme.test/"),
    ],
)
def test_the_right_page_ranks_first(question: str, expected_url: str):
    assert _top_url(question) == expected_url


def test_heading_matches_outrank_body_mentions():
    """A page headed "Pricing" beats a passing mention of price elsewhere."""
    mentions = page(
        "https://acme.test/blog/post",
        "Blog - choosing interiors",
        "# Choosing interiors\n\n"
        + "Think about price and cost and your budget carefully. " * 8,
        priority=40,
    )
    assert _top_url("What is the price?", [PRICING, mentions]) == PRICING.url


def test_covering_more_of_the_question_scores_higher():
    sections = split_into_sections(PRICING) + split_into_sections(CONTACT)
    ranked = score_sections(sections, extract_keywords("premium package payment terms"))
    assert ranked[0].url == PRICING.url
    assert len(ranked[0].matched_terms) >= 2


def test_matched_terms_are_recorded():
    sections = split_into_sections(CONTACT)
    ranked = score_sections(sections, extract_keywords("email address"))
    assert ranked[0].matched_terms


def test_without_keywords_ranking_falls_back_to_page_priority():
    sections: list[Section] = []
    for row in ALL_PAGES:
        sections.extend(split_into_sections(row))
    ranked = score_sections(sections, [])
    assert ranked[0].priority == 100  # the homepage


def test_scoring_an_empty_list_is_safe():
    assert score_sections([], ["price"]) == []


def test_irrelevant_question_scores_everything_low():
    sections: list[Section] = []
    for row in ALL_PAGES:
        sections.extend(split_into_sections(row))
    ranked = score_sections(sections, extract_keywords("quantum chromodynamics lagrangian"))
    assert all(section.score == 0 for section in ranked)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_repeated_sections_are_dropped():
    shared = "Call us today on +91 80 4123 5566 to book a free design consultation."
    first = Section(
        page_id=uuid.uuid4(), page_title="A", url="https://x.test/a", heading=None,
        text=shared, priority=50, index=0, score=5.0,
    )
    second = Section(
        page_id=uuid.uuid4(), page_title="B", url="https://x.test/b", heading=None,
        text=shared, priority=50, index=0, score=4.0,
    )
    kept = deduplicate([first, second])
    assert len(kept) == 1
    assert kept[0].url == "https://x.test/a"  # the higher-scoring one survives


def test_distinct_sections_are_both_kept():
    first = Section(
        page_id=uuid.uuid4(), page_title="A", url="https://x.test/a", heading=None,
        text="Our modular kitchens use soft-close hardware and quartz counters.",
        priority=50, index=0,
    )
    second = Section(
        page_id=uuid.uuid4(), page_title="B", url="https://x.test/b", heading=None,
        text="Payment is ten percent on booking and the balance on installation.",
        priority=50, index=0,
    )
    assert len(deduplicate([first, second])) == 2


# ---------------------------------------------------------------------------
# Budget selection
# ---------------------------------------------------------------------------


def test_selection_stops_at_the_token_budget():
    sections: list[Section] = []
    for row in ALL_PAGES:
        sections.extend(split_into_sections(row))
    chosen, used = select_within_budget(sections, budget_tokens=60, max_sections=20)
    assert used <= 60 + 10  # the trimmed tail may round slightly over
    assert len(chosen) < len(sections)


def test_selection_respects_the_section_cap():
    sections: list[Section] = []
    for row in ALL_PAGES:
        sections.extend(split_into_sections(row))
    chosen, _ = select_within_budget(sections, budget_tokens=100_000, max_sections=2)
    assert len(chosen) == 2


def test_a_zero_budget_selects_nothing():
    sections = split_into_sections(PRICING)
    chosen, used = select_within_budget(sections, budget_tokens=0, max_sections=10)
    assert chosen == []
    assert used == 0


# ---------------------------------------------------------------------------
# search() — the entry point
# ---------------------------------------------------------------------------


def test_small_knowledge_base_is_sent_whole():
    """Nothing has to be left out, so nothing is ranked away."""
    result = search(
        ALL_PAGES,
        "What are your prices?",
        budget_tokens=100_000,
        max_sections=50,
        min_score=0.08,
    )
    assert result.whole_knowledge_base_used is True
    assert result.pages_searched == 4
    assert {section.url for section in result.sections} == {
        row.url for row in ALL_PAGES
    }


def test_small_knowledge_base_is_ordered_by_page_priority():
    result = search(
        ALL_PAGES, "anything", budget_tokens=100_000, max_sections=50, min_score=0.08
    )
    assert result.sections[0].priority == 100


def test_large_knowledge_base_uses_keyword_ranking():
    """Forcing a tight budget switches search() to the ranked path."""
    result = search(
        ALL_PAGES,
        "What is the phone number?",
        budget_tokens=120,
        max_sections=3,
        min_score=0.08,
    )
    assert result.whole_knowledge_base_used is False
    assert result.found_anything
    assert result.sections[0].url == CONTACT.url
    assert result.knowledge_tokens <= 140


def test_no_relevant_content_returns_nothing():
    """The caller's signal to use the fallback instead of calling the model."""
    result = search(
        ALL_PAGES,
        "What is the airspeed velocity of an unladen swallow?",
        budget_tokens=120,
        max_sections=3,
        min_score=0.5,
    )
    assert result.found_anything is False
    assert result.sections == []


def test_an_empty_knowledge_base_returns_nothing():
    result = search([], "anything", budget_tokens=5_000, max_sections=10, min_score=0.08)
    assert result.found_anything is False
    assert result.pages_searched == 0


def test_keywords_are_reported_for_display():
    result = search(
        ALL_PAGES, "modular kitchen cost", budget_tokens=200, max_sections=3, min_score=0.08
    )
    assert "kitchen" in result.keywords


def test_retrieval_never_crosses_into_another_chatbots_pages():
    """search() only ever sees the rows it is handed.

    Scoping is the caller's responsibility (`load_included_pages` filters on
    chatbot_id and owner_id); this asserts the invariant that retrieval adds
    nothing of its own.
    """
    result = search(
        [CONTACT], "What is your phone number?", budget_tokens=5_000,
        max_sections=10, min_score=0.08,
    )
    assert {section.url for section in result.sections} == {CONTACT.url}
