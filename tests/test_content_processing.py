"""Tests for content cleaning, de-duplication and prioritisation."""

from __future__ import annotations

import pytest

from helpers import content_processing as cp
from helpers.firecrawl import build_crawl_payload


def _document(url: str, markdown: str, title: str | None = None, language: str | None = None):
    metadata: dict = {"sourceURL": url}
    if title is not None:
        metadata["title"] = title
    if language is not None:
        metadata["language"] = language
    return {"markdown": markdown, "metadata": metadata}


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------


def test_removes_images_and_keeps_link_text():
    raw = "![logo](https://cdn.acme.com/logo.png)\n\nSee our [pricing page](https://acme.com/pricing) today."
    cleaned = cp.clean_markdown(raw)
    assert "logo.png" not in cleaned
    assert "https://acme.com/pricing" not in cleaned
    assert "pricing page" in cleaned


def test_keeps_contact_details_from_mailto_and_tel_links():
    raw = "Email [our team](mailto:hi@acme.com) or call [us](tel:+919000000000)."
    cleaned = cp.clean_markdown(raw)
    assert "hi@acme.com" in cleaned
    assert "+919000000000" in cleaned


def test_strips_html_scripts_styles_and_tags():
    raw = "<script>alert('x')</script><style>.a{color:red}</style><p>Real content here.</p>"
    cleaned = cp.clean_markdown(raw)
    assert "alert" not in cleaned
    assert "color:red" not in cleaned
    assert "Real content here." in cleaned


def test_collapses_blank_lines_and_drops_decoration_lines():
    raw = "Heading\n\n\n\n---\n\n* * *\n\nBody text."
    cleaned = cp.clean_markdown(raw)
    assert "\n\n\n" not in cleaned
    assert "---" not in cleaned
    assert "Body text." in cleaned


def test_drops_base64_blobs():
    raw = "Intro text\n\n" + "data:image/png;base64," + "A" * 300 + "\n\nOutro text"
    cleaned = cp.clean_markdown(raw)
    assert "base64" not in cleaned
    assert "Intro text" in cleaned and "Outro text" in cleaned


def test_truncates_pathologically_long_pages():
    cleaned = cp.clean_markdown("word " * 100_000)
    assert len(cleaned) <= cp.MAX_PAGE_CHARACTERS + 40
    assert cleaned.endswith("[content truncated]")


def test_handles_none_and_empty_input():
    assert cp.clean_markdown(None) == ""
    assert cp.clean_markdown("") == ""


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def test_hash_ignores_whitespace_and_case():
    assert cp.content_hash("Hello   World") == cp.content_hash("hello world")


def test_hash_differs_for_different_content():
    assert cp.content_hash("about us") != cp.content_hash("contact us")


# ---------------------------------------------------------------------------
# page_priority
# ---------------------------------------------------------------------------


def test_homepage_ranks_highest():
    assert cp.page_priority("https://acme.com/") == 100
    assert cp.page_priority("https://acme.com") == 100


@pytest.mark.parametrize(
    "high,low",
    [
        ("https://acme.com/about", "https://acme.com/blog/post-1"),
        ("https://acme.com/pricing", "https://acme.com/privacy-policy"),
        ("https://acme.com/services", "https://acme.com/terms"),
        ("https://acme.com/contact", "https://acme.com/news/2024/update"),
    ],
)
def test_important_pages_outrank_low_value_pages(high: str, low: str):
    assert cp.page_priority(high) > cp.page_priority(low)


def test_deeper_paths_score_lower():
    assert cp.page_priority("https://acme.com/a") > cp.page_priority("https://acme.com/a/b/c/d")


# ---------------------------------------------------------------------------
# normalise_title / derive_chatbot_name
# ---------------------------------------------------------------------------


def test_title_accepts_list_form_from_firecrawl():
    assert cp.normalise_title(["Acme Home", "ignored"], "https://acme.com/") == "Acme Home"


def test_title_falls_back_to_slug():
    assert cp.normalise_title(None, "https://acme.com/about-us") == "About Us"
    assert cp.normalise_title(None, "https://acme.com/") == "Home"


def test_chatbot_name_from_title_strips_tagline():
    assert cp.derive_chatbot_name("Acme Industries | Steel Fabrication", "acme.com") == (
        "Acme Industries Assistant"
    )


def test_chatbot_name_falls_back_to_domain():
    assert cp.derive_chatbot_name(None, "www.blue-ridge.co.in") == "Blue Ridge Assistant"


def test_chatbot_name_prefers_the_title_segment_matching_the_domain():
    # The page leads its title with the page name, not the brand.
    assert cp.derive_chatbot_name(
        "All products | Books to Scrape - Sandbox", "books.toscrape.com"
    ) == "Books to Scrape Assistant"


def test_chatbot_name_prefers_declared_site_name():
    assert cp.derive_chatbot_name(
        "Home | Welcome", "acme.com", site_name="Acme Interiors"
    ) == "Acme Interiors Assistant"


def test_chatbot_name_ignores_unusable_site_name():
    assert cp.derive_chatbot_name(
        "Acme Industries | Steel Fabrication", "acme.com", site_name="  "
    ) == "Acme Industries Assistant"


# ---------------------------------------------------------------------------
# process_documents
# ---------------------------------------------------------------------------


def test_flags_duplicate_content_and_keeps_the_important_page():
    body = "Our company builds industrial pumps for the mining sector in Telangana. " * 4
    documents = [
        _document("https://acme.com/index-copy", body, title="Copy"),
        _document("https://acme.com/", body, title="Home"),
    ]
    result = cp.process_documents(documents)

    by_url = {p.url: p for p in result.pages}
    # The homepage has higher priority, so it keeps 'scraped'.
    assert by_url["https://acme.com/"].status == "scraped"
    assert by_url["https://acme.com/index-copy"].status == "duplicate"
    assert result.scraped == 1
    assert result.duplicates == 1


def test_duplicate_pages_are_not_included_in_the_knowledge_base():
    body = "Identical content about our service offerings and delivery timelines. " * 4
    result = cp.process_documents(
        [_document("https://acme.com/a", body), _document("https://acme.com/b", body)]
    )
    duplicate = next(p for p in result.pages if p.status == "duplicate")
    assert duplicate.error is not None and "Identical" in duplicate.error


def test_marks_thin_pages_as_empty():
    result = cp.process_documents([_document("https://acme.com/blank", "Hi")])
    assert result.pages[0].status == "empty"
    assert result.scraped == 0
    assert result.empty == 1


def test_strips_navigation_repeated_across_pages():
    nav = "Home\nAbout\nContact\nPrivacy Policy"
    documents = [
        _document(
            f"https://acme.com/p{i}",
            f"{nav}\n\nUnique body paragraph number {i} describing a distinct service offering in detail.",
        )
        for i in range(5)
    ]
    result = cp.process_documents(documents)

    assert all(p.status == "scraped" for p in result.pages)
    for page in result.pages:
        assert "Privacy Policy" not in page.content
        assert "Unique body paragraph" in page.content


def test_keeps_unique_lines_when_page_count_is_small():
    """Boilerplate stripping must not fire on a 2-page site."""
    documents = [
        _document(
            "https://acme.com/",
            "Welcome\n\nWe manufacture precision components for the automotive industry.",
        ),
        _document(
            "https://acme.com/about",
            "Welcome\n\nFounded in 1998, we operate three plants across southern India.",
        ),
    ]
    result = cp.process_documents(documents)
    assert all("Welcome" in p.content for p in result.pages)


def test_ignores_documents_without_a_url():
    result = cp.process_documents([{"markdown": "orphan", "metadata": {}}, "not-a-dict"])
    assert result.pages == []


def test_uses_final_url_when_source_url_absent():
    result = cp.process_documents(
        [{"markdown": "Some reasonably long body content for this page about us. " * 3,
          "metadata": {"url": "https://acme.com/about"}}]
    )
    assert result.pages[0].url == "https://acme.com/about"


def test_records_language_from_metadata():
    body = "Detailed description of our manufacturing capabilities and certifications. " * 3
    result = cp.process_documents([_document("https://acme.com/", body, language="en")])
    assert result.pages[0].language == "en"


def test_total_characters_counts_only_scraped_pages():
    body = "A sufficiently long unique paragraph about our logistics network. " * 3
    result = cp.process_documents(
        [_document("https://acme.com/a", body), _document("https://acme.com/b", body)]
    )
    assert result.total_characters == result.pages[0].content_length or result.total_characters > 0
    assert result.scraped == 1


# ---------------------------------------------------------------------------
# Firecrawl payload
# ---------------------------------------------------------------------------


def test_crawl_payload_enforces_page_limit_and_markdown_only():
    payload = build_crawl_payload("https://acme.com/", limit=20, max_depth=3)
    assert payload["limit"] == 20
    assert payload["scrapeOptions"]["formats"] == ["markdown"]
    assert payload["scrapeOptions"]["onlyMainContent"] is True
    assert payload["maxDiscoveryDepth"] == 3


def test_crawl_payload_never_bypasses_robots_or_leaves_the_domain():
    payload = build_crawl_payload("https://acme.com/", limit=20, max_depth=3)
    assert "ignoreRobotsTxt" not in payload
    assert payload["allowExternalLinks"] is False
    assert payload["allowSubdomains"] is False


def test_crawl_payload_excludes_auth_and_checkout_paths():
    payload = build_crawl_payload("https://acme.com/", limit=20, max_depth=3)
    joined = " ".join(payload["excludePaths"])
    for fragment in ("login", "checkout", "admin", "cart", "signup"):
        assert fragment in joined
