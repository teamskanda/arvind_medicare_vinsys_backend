"""Lightweight keyword retrieval over a chatbot's own stored pages.

No embeddings, no vector database, no pgvector — by design for Phase 2.

Why ranking happens in Python rather than in PostgreSQL: a chatbot holds at
most `CRAWL_PAGE_LIMIT` (20) pages, so the whole included set is well under a
megabyte. Loading it and scoring in memory keeps ranking identical across
PostgreSQL versions, makes every scoring rule unit-testable without a
database, and lets the "small knowledge base fits entirely in the budget"
shortcut be an exact decision rather than a guess. `002_chat.sql` creates the
GIN full-text index so a first-stage SQL filter can be pushed down later
without a migration or an API change.

The module boundary is deliberate: `search()` takes rows and a token budget
and returns ranked `Section`s. Swapping in vector retrieval later means
reimplementing `score_sections()` only — `prompt_builder` and `routers/chat`
never learn how relevance was computed.

Everything returned is untrusted website text. Sanitising it for the prompt is
`prompt_builder`'s job, not this module's.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field

from helpers.token_manager import count_tokens, trim_to_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------

# Target section size in characters. Small enough that several fit in the
# knowledge budget, large enough to carry a complete answer (a pricing table,
# an FAQ entry, an address block).
TARGET_SECTION_CHARACTERS = 1_400
MAX_SECTION_CHARACTERS = 2_600
MIN_SECTION_CHARACTERS = 40

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Common English function words. Dropped from queries so "what are your
# opening hours" ranks on "opening" and "hours", not on "what" and "are".
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can cannot can't
    could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he her
    here hers herself him himself his how i if in into is isn't it its
    itself just me more most my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan't she should
    shouldn't so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was
    wasn't we were weren't what when where which while who whom why will
    with won't would wouldn't you your yours yourself yourselves
    tell know want need please thanks thank hi hello hey okay ok im i'm
    give show get got also would like help
    """.split()
)

# Query expansion. Visitors rarely use the website's own wording: they ask
# "how much" against a page headed "Pricing", or "where are you" against an
# "experience centre" on a named road.
#
# IMPORTANT: keys are matched against *stemmed* terms, and the values are
# stemmed before use. So the key must be what `_stem()` produces — "hour" not
# "hours", "pric" not "pricing" — or the entry can never fire. The tests in
# `test_retrieval.py::test_every_synonym_key_is_its_own_stem` enforce this.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "price": ("pricing", "cost", "fee", "rate", "charge", "plan", "package"),
    "pric": ("price", "cost", "fee", "rate", "plan"),
    "cost": ("price", "pricing", "fee", "rate", "charge"),
    "phone": ("telephone", "mobile", "call", "contact", "number"),
    "email": ("mail", "contact", "write"),
    "address": ("location", "office", "centre", "center", "road", "street", "visit"),
    "locate": ("location", "address", "office", "centre", "center", "visit"),
    "located": ("location", "address", "office", "centre", "center", "road", "street", "visit"),
    "location": ("address", "office", "centre", "center", "road", "street", "visit"),
    "showroom": ("office", "centre", "center", "store", "studio", "address", "visit"),
    "office": ("address", "location", "centre", "center", "studio"),
    "hour": ("timing", "open", "opening", "schedule", "time"),
    "time": ("hour", "open", "opening", "schedule", "timing"),
    "service": ("offering", "solution", "work", "provide"),
    "product": ("item", "range", "catalogue"),
    "purchase": ("buy", "sale", "product", "item", "catalogue"),
    "buy": ("purchase", "sale", "product", "item", "catalogue"),
    "sell": ("sale", "product", "item", "catalogue", "shop"),
    "read": ("book", "article", "content"),
    "refund": ("return", "cancellation", "cancel", "money-back"),
    # Both stems are reachable: "ship" from "ship/ships", "shipp" from
    # "shipping" (which `_stem` reduces to "shipp", not "ship").
    "ship": ("shipping", "delivery", "dispatch"),
    "shipp": ("shipping", "delivery", "dispatch"),
    # "help" is deliberately absent: it is a stopword, so expanding into it
    # would add a term that matches nothing.
    "support": ("assistance", "service", "care"),
    "team": ("staff", "people", "founder", "leadership"),
    "warranty": ("guarantee", "warrantee"),
    "book": ("booking", "appointment", "reserve", "schedule", "consultation"),
}


@dataclass(slots=True)
class PageRow:
    """The subset of a `website_pages` row retrieval needs."""

    id: uuid.UUID
    url: str
    title: str
    content: str
    priority: int


@dataclass(slots=True)
class Section:
    """One retrievable chunk of one page."""

    page_id: uuid.UUID
    page_title: str
    url: str
    heading: str | None
    text: str
    priority: int
    index: int
    score: float = 0.0
    matched_terms: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Human-readable citation label, e.g. "Services - Pricing"."""
        if self.heading and self.heading.lower() not in self.page_title.lower():
            return f"{self.page_title} - {self.heading}"
        return self.page_title


@dataclass(slots=True)
class RetrievalResult:
    sections: list[Section] = field(default_factory=list)
    sections_considered: int = 0
    pages_searched: int = 0
    keywords: list[str] = field(default_factory=list)
    knowledge_tokens: int = 0
    whole_knowledge_base_used: bool = False

    @property
    def found_anything(self) -> bool:
        return bool(self.sections)


# ---------------------------------------------------------------------------
# Splitting pages into sections
# ---------------------------------------------------------------------------


def _flush(buffer: list[str]) -> str:
    return "\n".join(buffer).strip()


def split_into_sections(page: PageRow) -> list[Section]:
    """Split one page's Markdown into heading-aware sections.

    Markdown headings are the primary boundary because they are what the
    website author used to group meaning. Oversized runs of text between
    headings are split again on blank lines so one long page cannot produce a
    single section that eats the whole budget.
    """
    sections: list[Section] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def emit() -> None:
        body = _flush(buffer)
        buffer.clear()
        if len(body) < MIN_SECTION_CHARACTERS:
            return
        for piece in _split_oversized(body):
            sections.append(
                Section(
                    page_id=page.id,
                    page_title=page.title,
                    url=page.url,
                    heading=current_heading,
                    text=piece,
                    priority=page.priority,
                    index=len(sections),
                )
            )

    for line in (page.content or "").split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            emit()
            current_heading = re.sub(r"\s+", " ", match.group(2)).strip()[:160]
            continue
        buffer.append(line)
        if sum(len(item) + 1 for item in buffer) >= MAX_SECTION_CHARACTERS:
            emit()

    emit()

    # A page with no usable sections (very short page) still contributes its
    # whole body, so a one-line contact page remains answerable.
    if not sections:
        body = (page.content or "").strip()
        if len(body) >= MIN_SECTION_CHARACTERS:
            sections.append(
                Section(
                    page_id=page.id,
                    page_title=page.title,
                    url=page.url,
                    heading=None,
                    text=body,
                    priority=page.priority,
                    index=0,
                )
            )

    return sections


def _split_oversized(body: str) -> list[str]:
    """Break a long run of text on paragraph boundaries near the target size."""
    if len(body) <= MAX_SECTION_CHARACTERS:
        return [body]

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in body.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if size and size + len(paragraph) > TARGET_SECTION_CHARACTERS:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        # A single paragraph larger than the cap is hard-split on whitespace.
        while len(paragraph) > MAX_SECTION_CHARACTERS:
            cut = paragraph.rfind(" ", 0, MAX_SECTION_CHARACTERS)
            cut = cut if cut > MAX_SECTION_CHARACTERS // 2 else MAX_SECTION_CHARACTERS
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        current.append(paragraph)
        size += len(paragraph) + 2

    if current:
        pieces.append("\n\n".join(current))
    return [piece for piece in pieces if len(piece) >= MIN_SECTION_CHARACTERS]


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------


def extract_keywords(question: str) -> list[str]:
    """Normalise a question into ranked search terms.

    Keeps order of first appearance (so the caller can show them back to the
    customer) and expands a small synonym map, because visitors rarely use the
    website's own wording — "how much" against a page headed "Pricing".
    """
    lowered = (question or "").lower()
    tokens = _WORD_RE.findall(lowered)

    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 2 or token in STOPWORDS:
            continue
        stem = _stem(token)
        if stem in seen:
            continue
        seen.add(stem)
        terms.append(stem)

    # "how much" / "how many" are strong price/quantity signals that survive
    # stopword removal only as a phrase.
    if "how much" in lowered or "how many" in lowered:
        for extra in ("price", "cost"):
            if extra not in seen:
                seen.add(extra)
                terms.append(extra)

    expanded = list(terms)
    for term in terms:
        for synonym in _SYNONYMS.get(term, ()):
            stem = _stem(synonym)
            if stem not in seen:
                seen.add(stem)
                expanded.append(stem)

    return expanded[:24]


def contextual_query(question: str, history: list[tuple[str, str]]) -> str:
    """Resolve lightweight follow-up references using this thread's last question.

    Only retrieval sees the extra terms; Gemma still gets the actual question
    and the independently budgeted history. No cross-chatbot content is loaded.
    """
    if not re.search(r'\b(that|those|these|it|its|their|they|them|more briefly)\b|उस|इसके|उनके', question, re.IGNORECASE):
        return question
    previous = next((text for role, text in reversed(history) if role == 'user'), '')
    return f'{question}\n{previous[:2000]}' if previous else question


# Broad "what is this place?" questions. They rarely share vocabulary with the
# site's own pages, so keyword ranking finds nothing; these get the site's
# highest-priority (homepage/about) content instead of the fallback reply.
_PLACE = r"(?:website|web\s?site|site|business|company|shop|store|brand|organi[sz]ation|place|page)"
_OVERVIEW_RE = re.compile(
    r"\boverview\b"
    r"|\bwhat (?:does|do)\b.{0,80}\b(?:sell|offer|do|provide)\b"
    r"|\bwhat (?:kind|type|sort) of " + _PLACE + r"\b"
    r"|\bwhat(?:'s| is| are)? (?:this|that|your|the) " + _PLACE + r"\b"
    r"|\bwhat (?:is|are) (?:this|these|you) (?:all )?about\b"
    r"|\btell me (?:more )?about (?:this|your|the|you|yourselves)\b"
    r"|\bwho (?:are you|are you guys|is this|runs this)\b"
    r"|\babout (?:this|your|the) " + _PLACE + r"\b",
    re.IGNORECASE,
)


def needs_overview(question: str) -> bool:
    return bool(_OVERVIEW_RE.search(question or ""))


def _stem(word: str) -> str:
    """Crude suffix stripping, enough to match plural and gerund variants.

    Order and the guards matter. Stripping a bare "es" over-stems: it turns
    "wardrobes" into "wardrob" while "wardrobe" stays whole, so a question
    about wardrobes stops matching a page that says wardrobe. Plural "s" is
    therefore removed before anything else, and "es" only after the sibilants
    that actually require it.
    """
    if word.endswith("'s"):
        word = word[:-2]

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "ches", "shes", "xes", "zes")):
        return word[:-2]
    # "ss"/"us"/"is" endings are not plurals ("address", "campus", "analysis").
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    # "opening" -> "open". Guarded so the stem stays a real word fragment.
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    return word


def _section_terms(section: Section) -> Counter[str]:
    body = f"{section.text}".lower()
    return Counter(_stem(token) for token in _WORD_RE.findall(body))


def _heading_terms(section: Section) -> set[str]:
    source = f"{section.page_title} {section.heading or ''}".lower()
    return {_stem(token) for token in _WORD_RE.findall(source)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Weights. Title/heading matches dominate because a website's headings are the
# author's own index of the page.
WEIGHT_BODY = 1.0
WEIGHT_HEADING = 3.0
WEIGHT_COVERAGE = 2.5
WEIGHT_PRIORITY = 0.4
# Saturating term frequency: the tenth mention of "pricing" is not ten times
# the evidence of the first.
TF_SATURATION = 1.6


def score_sections(sections: list[Section], keywords: list[str]) -> list[Section]:
    """Rank sections against the query terms. Mutates and returns `sections`.

    The scoring is BM25-flavoured: saturating term frequency, an inverse
    document frequency factor so a term appearing in every section carries
    little signal, plus heading and page-priority boosts.
    """
    if not sections:
        return []
    if not keywords:
        # No usable query terms: fall back to the website's own structure.
        for section in sections:
            section.score = section.priority / 100.0
            section.matched_terms = ()
        return sorted(sections, key=_sort_key)

    total = len(sections)
    section_terms = [_section_terms(section) for section in sections]

    # Document frequency per keyword across this chatbot's sections.
    document_frequency: Counter[str] = Counter()
    for terms in section_terms:
        for keyword in keywords:
            if terms.get(keyword):
                document_frequency[keyword] += 1

    for section, terms in zip(sections, section_terms, strict=True):
        headings = _heading_terms(section)
        length = max(1, sum(terms.values()))
        score = 0.0
        matched: list[str] = []

        for keyword in keywords:
            frequency = terms.get(keyword, 0)
            in_heading = keyword in headings
            if not frequency and not in_heading:
                continue
            matched.append(keyword)

            # IDF: log-scaled, floored at a small positive value so a term
            # present everywhere still counts for something.
            appearances = document_frequency.get(keyword, 0) or 1
            idf = max(0.25, _log((total + 1) / appearances))

            if frequency:
                # Normalise by section length so a long section does not win
                # on volume alone.
                normalised = frequency / (frequency + TF_SATURATION * (length / 160 + 1))
                score += WEIGHT_BODY * idf * normalised
            if in_heading:
                score += WEIGHT_HEADING * idf

        if matched:
            # Covering more of the question matters more than hammering one term.
            coverage = len(matched) / len(keywords)
            score += WEIGHT_COVERAGE * coverage
            score += WEIGHT_PRIORITY * (section.priority / 100.0)
            # Earliest section of a page is usually its summary.
            if section.index == 0:
                score += 0.15

        section.score = round(score, 6)
        section.matched_terms = tuple(matched)

    return sorted(sections, key=_sort_key)


def _log(value: float) -> float:
    return math.log(value) if value > 0 else 0.0


def _sort_key(section: Section) -> tuple[float, int, int]:
    # Descending score, then descending page priority, then document order.
    return (-section.score, -section.priority, section.index)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _shingles(text: str, size: int = 6) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def deduplicate(sections: list[Section], threshold: float = 0.75) -> list[Section]:
    """Drop sections that substantially repeat one already kept.

    Page-level exact duplicates were already filtered during Phase 1 ingestion;
    this catches *section*-level repetition, which is common — a shared call-to-
    action block, or the same address on three pages. Jaccard overlap of word
    shingles, compared only against already-kept sections.
    """
    kept: list[Section] = []
    kept_shingles: list[set[str]] = []

    for section in sections:
        current = _shingles(section.text)
        if not current:
            continue
        duplicate = False
        for existing in kept_shingles:
            union = len(current | existing)
            if not union:
                continue
            if len(current & existing) / union >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(section)
        kept_shingles.append(current)

    return kept


# ---------------------------------------------------------------------------
# Selection against the token budget
# ---------------------------------------------------------------------------


def select_within_budget(
    sections: list[Section], budget_tokens: int, max_sections: int
) -> tuple[list[Section], int]:
    """Take the best sections that fit `budget_tokens`.

    Walks the ranked list and keeps whatever fits. The final section is
    trimmed to the remaining space rather than dropped, so a budget boundary
    does not throw away a partly usable answer. Returns the chosen sections
    and the tokens they occupy.
    """
    chosen: list[Section] = []
    used = 0

    for section in sections:
        if len(chosen) >= max_sections or used >= budget_tokens:
            break

        cost = count_tokens(section.text)
        remaining = budget_tokens - used

        if cost <= remaining:
            chosen.append(section)
            used += cost
            continue

        # Trim the tail section into whatever room is left, if that leaves
        # enough text to be worth sending.
        trimmed = trim_to_tokens(section.text, remaining)
        if len(trimmed) >= MIN_SECTION_CHARACTERS * 3:
            section.text = trimmed + "\n[...]"
            chosen.append(section)
            used += count_tokens(section.text)
        break

    return chosen, used


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def search(
    pages: list[PageRow],
    question: str,
    *,
    budget_tokens: int,
    max_sections: int,
    min_score: float,
) -> RetrievalResult:
    """Retrieve the website content most likely to answer `question`.

    Two paths:

      * **Small knowledge base.** If every section fits in the token budget,
        all of it is sent in the website's own priority order. Keyword ranking
        cannot help when nothing has to be left out, and it could hurt by
        hiding the one page that happened to phrase things differently.
      * **Larger knowledge base.** Rank, drop sections below `min_score`,
        de-duplicate, then fill the budget best-first.

    A result with no sections is the caller's signal to return the configured
    fallback rather than send an unrelated prompt to Gemma.
    """
    all_sections: list[Section] = []
    for page in pages:
        all_sections.extend(split_into_sections(page))

    result = RetrievalResult(
        sections_considered=len(all_sections),
        pages_searched=len(pages),
        keywords=extract_keywords(question),
    )
    if not all_sections:
        return result

    total_tokens = sum(count_tokens(section.text) for section in all_sections)

    if total_tokens <= budget_tokens and len(all_sections) <= max_sections:
        ordered = sorted(all_sections, key=lambda s: (-s.priority, s.index))
        result.sections = ordered
        result.knowledge_tokens = total_tokens
        result.whole_knowledge_base_used = True
        logger.debug(
            "Knowledge base fits the budget (%s/%s tokens); sending all %s sections.",
            total_tokens, budget_tokens, len(ordered),
        )
        return result

    ranked = score_sections(all_sections, result.keywords)
    relevant = [section for section in ranked if section.score >= min_score]
    if not relevant and needs_overview(question):
        # A broad introduction need not contain a site's product vocabulary.
        # Supply real homepage/about content; the model still follows grounding.
        relevant = sorted(all_sections, key=lambda section: (-section.priority, section.index))
    if not relevant:
        logger.debug("No section scored at or above %s for %r.", min_score, question[:80])
        return result

    chosen, used = select_within_budget(
        deduplicate(relevant), budget_tokens, max_sections
    )
    result.sections = chosen
    result.knowledge_tokens = used
    return result
