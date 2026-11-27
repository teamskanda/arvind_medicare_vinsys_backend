"""Turning raw Firecrawl Markdown into storable knowledge-base pages.

Four jobs:
  * clean  — drop images, link noise, stray HTML and whitespace padding;
  * de-boilerplate — remove nav/footer lines that repeat across the site;
  * de-duplicate — flag pages whose cleaned content is byte-identical;
  * prioritise — score pages so Phase 2 retrieval can break ties toward the
    pages that actually answer customer questions.

Nothing here truncates the knowledge base to fit a model context. Pages are
stored whole and per-page; Phase 2 selects from them at question time.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Hard per-page cap so one pathological page cannot bloat a row. Far above
# any realistic marketing page (~200k characters is roughly 50k tokens).
MAX_PAGE_CHARACTERS = 200_000

# Minimum cleaned length for a page to count as useful content.
MIN_USEFUL_CHARACTERS = 80

# A line repeating on at least this share of pages is treated as chrome.
BOILERPLATE_PAGE_RATIO = 0.6
BOILERPLATE_MIN_PAGES = 4
BOILERPLATE_MAX_LINE_LENGTH = 200


# ---------------------------------------------------------------------------
# Firecrawl path filters (RE2-compatible: no look-around, no backreferences)
# ---------------------------------------------------------------------------

EXCLUDE_PATH_PATTERNS: list[str] = [
    # Authentication and account areas
    r"^/(login|signin|sign-in|log-in|signup|sign-up|register|registration)",
    r"^/(logout|signout|sign-out)",
    r"^/(account|my-account|profile|dashboard|portal)",
    r"^/(password|forgot-password|reset-password)",
    # Commerce funnels
    r"^/(cart|basket|checkout|order|orders|payment|billing|invoice)",
    r"^/(wishlist|compare)",
    # Admin surfaces
    r"^/(admin|administrator|wp-admin|wp-login|cpanel|phpmyadmin)",
    r"^/(api|graphql|rest|xmlrpc)",
    # Low-value archives and feeds
    r"^/(feed|rss|atom|sitemap)",
    r"/(tag|tags|author|archive|archives)/",
    r"/page/[0-9]+",
    r"^/(search|q)",
    # Non-document assets
    r"\.(pdf|zip|rar|gz|tar|exe|dmg|pkg|csv|xlsx?|docx?|pptx?)$",
    r"\.(jpg|jpeg|png|gif|svg|webp|avif|ico|bmp|mp4|webm|mp3|wav|woff2?|ttf|eot)$",
    # Localisation duplicates of the same content
    r"^/(cdn-cgi|cgi-bin)",
]


# ---------------------------------------------------------------------------
# Page priority
# ---------------------------------------------------------------------------

_PRIORITY_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^/?$"), 100),
    (re.compile(r"(^|/)(about|about-us|who-we-are|our-story|company)(/|$)"), 92),
    (re.compile(r"(^|/)(service|services|solutions|what-we-do|offerings)(/|$)"), 90),
    (re.compile(r"(^|/)(product|products|shop|catalogue|catalog)(/|$)"), 88),
    (re.compile(r"(^|/)(pricing|plans|price|packages|rates)(/|$)"), 88),
    (re.compile(r"(^|/)(faq|faqs|help|support|knowledge-base)(/|$)"), 86),
    (re.compile(r"(^|/)(contact|contact-us|get-in-touch|locations?)(/|$)"), 84),
    (re.compile(r"(^|/)(team|leadership|careers?|jobs)(/|$)"), 60),
    (re.compile(r"(^|/)(case-stud|portfolio|work|projects?)(/|$)"), 58),
    (re.compile(r"(^|/)(blog|news|press|articles?|insights?)(/|$)"), 40),
    (re.compile(r"(^|/)(privacy|terms|cookie|legal|disclaimer|refund)(/|$)"), 20),
]


def page_priority(url: str) -> int:
    """Heuristic importance score, 0-100. Higher means more useful."""
    try:
        path = urlparse(url).path.lower().rstrip("/")
    except ValueError:
        return 50
    if not path:
        return 100
    for pattern, score in _PRIORITY_RULES:
        if pattern.search(path):
            return score
    # Shallow pages tend to matter more than deep ones.
    depth = len([segment for segment in path.split("/") if segment])
    return max(30, 55 - (depth - 1) * 5)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|iframe|svg)\b.*?</\1>", re.DOTALL | re.IGNORECASE
)
_HTML_TAG_RE = re.compile(r"<[^>]{1,400}>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_CONTACT_LINK_RE = re.compile(r"\[([^\]]*)\]\((mailto:|tel:)([^)]+)\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*<?([^)\s]*)>?[^)]*\)")
_DATA_URI_RE = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_BASE64_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_SYMBOL_ONLY_RE = re.compile(r"^[\s\-_*=~`|•·–—>#.+\[\]()]{0,}$")


def clean_markdown(raw: str | None) -> str:
    """Normalise one page's Markdown into compact, readable text."""
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub("", text)
    text = _DATA_URI_RE.sub("", text)
    text = _BASE64_BLOB_RE.sub("", text)

    # Keep contact details, which customers ask about constantly.
    text = _MD_CONTACT_LINK_RE.sub(
        lambda m: f"{m.group(1).strip()} ({m.group(2)}{m.group(3).strip()})".strip(),
        text,
    )
    # Other links collapse to their anchor text; the raw href is token noise.
    text = _MD_LINK_RE.sub(lambda m: m.group(1).strip(), text)
    text = _HTML_TAG_RE.sub(" ", text)

    lines: list[str] = []
    previous: str | None = None
    for line in text.split("\n"):
        stripped = line.rstrip()
        collapsed = stripped.strip()
        # Drop decoration-only lines and consecutive duplicates (nav remnants).
        if collapsed and _SYMBOL_ONLY_RE.match(collapsed):
            continue
        if collapsed and collapsed == previous:
            continue
        if collapsed:
            previous = collapsed
        lines.append(stripped)

    text = "\n".join(lines)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text).strip()

    if len(text) > MAX_PAGE_CHARACTERS:
        text = text[:MAX_PAGE_CHARACTERS].rstrip() + "\n\n[content truncated]"
    return text


def content_hash(text: str) -> str:
    """Hash of whitespace-normalised content, for exact duplicate detection."""
    normalised = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def normalise_title(title: object, fallback_url: str) -> str:
    """Firecrawl's `metadata.title` may be a string, a list, or absent."""
    candidate: str | None = None
    if isinstance(title, str):
        candidate = title
    elif isinstance(title, (list, tuple)) and title:
        first = title[0]
        if isinstance(first, str):
            candidate = first

    if candidate:
        cleaned = re.sub(r"\s+", " ", candidate).strip()
        if cleaned:
            return cleaned[:200]

    path = urlparse(fallback_url).path.strip("/")
    if not path:
        return "Home"
    last = path.split("/")[-1]
    last = re.sub(r"\.(html?|php|aspx?)$", "", last, flags=re.IGNORECASE)
    pretty = re.sub(r"[-_]+", " ", last).strip()
    return (pretty.title() or "Untitled page")[:200]


def _host_labels(host: str) -> list[str]:
    """Meaningful domain labels: no www/m prefix, no TLD, nothing tiny.

    `www.books.toscrape.com` -> ["books", "toscrape"]; `blue-ridge.co.in`
    -> ["blue-ridge"] (the "co" second-level suffix is too short to count).
    """
    label = host.lower().rstrip(".")
    for prefix in ("www.", "m."):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    parts = label.split(".")[:-1] or label.split(".")
    return [part for part in parts if len(part) >= 3]


def _usable_name(value: str | None) -> str | None:
    if not value:
        return None
    name = re.sub(r"\s+", " ", value).strip()
    return name if 2 <= len(name) <= 60 else None


def derive_chatbot_name(
    site_title: str | None, host: str, *, site_name: str | None = None
) -> str:
    """Name the chatbot after the business, falling back to the domain.

    Preference order:
      1. `og:site_name`, the brand name the site declares for itself;
      2. the title segment that matches the domain ("All products | Books to
         Scrape" on books.toscrape.com -> "Books to Scrape"), because a page
         title often leads with the page, not the brand;
      3. the first title segment (the usual "Brand | Tagline" layout);
      4. the domain itself.
    """
    declared = _usable_name(site_name)
    if declared:
        return f"{declared} Assistant"

    labels = _host_labels(host)
    if site_title:
        segments = [
            segment
            for segment in (
                _usable_name(part)
                for part in re.split(r"\s+[\|\-–—:·•]\s+", site_title.strip())
            )
            if segment
        ]
        for segment in segments:
            compact = re.sub(r"[^a-z0-9]", "", segment.lower())
            if any(re.sub(r"[^a-z0-9]", "", label) in compact for label in labels):
                return f"{segment} Assistant"
        if segments:
            return f"{segments[0]} Assistant"

    root = labels[0] if labels else (host.lower().split(".")[0] or "website")
    pretty = re.sub(r"[-_]+", " ", root).strip().title() or "Website"
    return f"{pretty} Assistant"


# ---------------------------------------------------------------------------
# Multi-page processing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProcessedPage:
    url: str
    title: str
    content: str
    content_hash: str
    content_length: int
    language: str | None
    priority: int
    status: str  # 'scraped' | 'duplicate' | 'empty'
    error: str | None = None


@dataclass(slots=True)
class ProcessingResult:
    pages: list[ProcessedPage] = field(default_factory=list)

    @property
    def scraped(self) -> int:
        return sum(1 for p in self.pages if p.status == "scraped")

    @property
    def duplicates(self) -> int:
        return sum(1 for p in self.pages if p.status == "duplicate")

    @property
    def empty(self) -> int:
        return sum(1 for p in self.pages if p.status == "empty")

    @property
    def total_characters(self) -> int:
        return sum(p.content_length for p in self.pages if p.status == "scraped")


def _find_boilerplate_lines(cleaned_pages: list[str]) -> set[str]:
    """Lines that repeat across most pages are site chrome, not content."""
    if len(cleaned_pages) < BOILERPLATE_MIN_PAGES:
        return set()

    counter: Counter[str] = Counter()
    for content in cleaned_pages:
        seen_on_this_page = {
            line.strip()
            for line in content.split("\n")
            if line.strip() and len(line.strip()) <= BOILERPLATE_MAX_LINE_LENGTH
        }
        counter.update(seen_on_this_page)

    threshold = max(BOILERPLATE_MIN_PAGES - 1, int(len(cleaned_pages) * BOILERPLATE_PAGE_RATIO))
    return {line for line, count in counter.items() if count >= threshold}


def _strip_lines(content: str, blocked: set[str]) -> str:
    if not blocked:
        return content
    kept = [line for line in content.split("\n") if line.strip() not in blocked]
    return _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(kept)).strip()


def process_documents(documents: list[dict]) -> ProcessingResult:
    """Clean, de-boilerplate and de-duplicate a batch of Firecrawl documents.

    `documents` are raw items from the Firecrawl crawl `data` array. Pages
    whose content duplicates an earlier page are kept but marked `duplicate`
    and excluded from the AI knowledge base, so the customer can see what
    happened rather than silently losing pages.
    """
    extracted: list[tuple[str, str, str, str | None]] = []  # url, title, cleaned, language

    for document in documents:
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        url = (
            metadata.get("sourceURL")
            or metadata.get("url")
            or document.get("url")
            or ""
        )
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()[:2048]

        cleaned = clean_markdown(document.get("markdown"))
        title = normalise_title(metadata.get("title"), url)
        language = metadata.get("language")
        language = language[:16] if isinstance(language, str) and language else None
        extracted.append((url, title, cleaned, language))

    # Second pass: strip chrome that appears across the whole site.
    boilerplate = _find_boilerplate_lines([item[2] for item in extracted])

    result = ProcessingResult()
    seen_hashes: dict[str, str] = {}  # hash -> first URL that had it
    seen_urls: set[str] = set()

    # Process high-priority pages first so the canonical copy of duplicated
    # content is the important page, not an arbitrary one.
    ordered = sorted(extracted, key=lambda item: -page_priority(item[0]))

    for url, title, cleaned, language in ordered:
        if url in seen_urls:
            continue
        seen_urls.add(url)

        content = _strip_lines(cleaned, boilerplate)
        priority = page_priority(url)

        if len(content) < MIN_USEFUL_CHARACTERS:
            result.pages.append(
                ProcessedPage(
                    url=url,
                    title=title,
                    content=content,
                    content_hash=content_hash(content),
                    content_length=len(content),
                    language=language,
                    priority=priority,
                    status="empty",
                    error="No meaningful text content was found on this page.",
                )
            )
            continue

        digest = content_hash(content)
        if digest in seen_hashes:
            result.pages.append(
                ProcessedPage(
                    url=url,
                    title=title,
                    content=content,
                    content_hash=digest,
                    content_length=len(content),
                    language=language,
                    priority=priority,
                    status="duplicate",
                    error=f"Identical content to {seen_hashes[digest]}",
                )
            )
            continue

        seen_hashes[digest] = url
        result.pages.append(
            ProcessedPage(
                url=url,
                title=title,
                content=content,
                content_hash=digest,
                content_length=len(content),
                language=language,
                priority=priority,
                status="scraped",
            )
        )

    return result
