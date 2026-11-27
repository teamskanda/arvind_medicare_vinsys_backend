"""Suggested follow-up questions, generated in the same model call as the answer.

The model is asked (see `prompt_builder.build_system_text`) to end its reply
with a marker line followed by questions the visitor is likely to ask next:

    Jane Eyre is £38.43 and in stock.
    [[FOLLOW_UPS]] Do you have other classics? | How long does delivery take?

Asking in the same call costs a few dozen output tokens and no extra round
trip, where a second call would add a full model latency to every answer.

This module separates the two parts. The visitor never sees the marker: the
streaming splitter holds back any tail that could still turn into it, so the
marker cannot flash on screen before it is recognised. Models do not always
copy a marker exactly, so a few common variants are accepted too
("Follow-ups:", "**Follow-up questions:**"), but only at the start of a line,
so an answer that merely mentions following up is never cut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MARKER = "[[FOLLOW_UPS]]"

# The marker, or a "Follow-up(s)/Follow-up questions" label at a line start.
_MARKER_RE = re.compile(
    r"\[\[\s*follow[\s_-]?ups?\s*\]\]"
    r"|(?:^|\n)[ \t]*[*_#>-]*[ \t]*follow[\s_-]?up(?:s|\s+questions?)?[ \t]*[*_]*[ \t]*:",
    re.IGNORECASE,
)

# What a tail must look like (lower-cased, punctuation removed) to possibly
# grow into a marker. Used only to decide what to hold back while streaming.
_MARKER_WORDS = ("followups", "followupquestions")

MAX_QUESTION_CHARACTERS = 120

# Questions the assistant would ask the visitor. As a button the visitor
# clicks, "Would you like to see more?" makes no sense, so these are dropped.
_ASSISTANT_VOICE_RE = re.compile(
    r"^(?:would you like|do you want|do you need|is there anything|anything else|"
    r"can i help|may i help|how (?:else )?can i help|shall i|should i|"
    r"what (?:else )?can i (?:help|do)|are you (?:looking|interested))\b",
    re.IGNORECASE,
)


def _could_become_marker(tail: str) -> bool:
    compact = re.sub(r"[\s\[\]*_#>:-]", "", tail.lower())
    if not compact:
        # Only brackets or a bare line start so far: "[[" or "\n" could still
        # be the start of the marker.
        return "[" in tail or tail.startswith("\n") or tail == ""
    return any(word.startswith(compact) or compact.startswith(word) for word in _MARKER_WORDS)


def clean_questions(raw: str, *, limit: int, question: str = "") -> list[str]:
    """Turn the text after the marker into at most `limit` clean questions."""
    if limit <= 0 or not raw:
        return []
    asked = question.strip().lower().rstrip("?")
    pieces = re.split(r"\s*\|\s*|\n+", raw)
    questions: list[str] = []
    for piece in pieces:
        text = re.sub(r"^\s*(?:[-*•>]|\d+[.)])\s*", "", piece)  # bullets, numbering
        text = text.strip().strip("\"'“”*_` ").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 4 or len(text) > MAX_QUESTION_CHARACTERS:
            continue
        if _MARKER_RE.search(text) or _ASSISTANT_VOICE_RE.search(text):
            continue
        if text.lower().rstrip("?") == asked:
            continue
        if any(text.lower() == existing.lower() for existing in questions):
            continue
        questions.append(text)
        if len(questions) >= limit:
            break
    return questions


def split_answer(text: str, *, limit: int, question: str = "") -> tuple[str, list[str]]:
    """Split a complete reply into (answer, follow-up questions)."""
    match = _MARKER_RE.search(text or "")
    if match is None:
        return (text or "").strip(), []
    answer = text[: match.start()].rstrip()
    return answer, clean_questions(text[match.end() :], limit=limit, question=question)


@dataclass(slots=True)
class StreamSplitter:
    """Feed streamed deltas; get back only the text the visitor should see.

    Everything before the marker is released as soon as it cannot be the
    start of the marker. Everything after it is collected for `finish()`.
    """

    limit: int
    question: str = ""
    _buffer: str = ""
    _tail: str = ""
    _found: bool = False
    _released: list[str] = field(default_factory=list)

    def feed(self, delta: str) -> str:
        if self._found:
            self._tail += delta
            return ""
        self._buffer += delta
        match = _MARKER_RE.search(self._buffer)
        if match is not None:
            self._found = True
            visible = self._buffer[: match.start()].rstrip()
            self._tail = self._buffer[match.end() :]
            self._buffer = ""
            return self._release(visible)

        # Hold back from the earliest point in the last line ("\n" or "[")
        # whose remainder could still grow into the marker; release the rest.
        cut = self._hold_from()
        visible, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._release(visible)

    def _hold_from(self) -> int:
        start = max(self._buffer.rfind("\n"), 0)
        for index in range(start, len(self._buffer)):
            if self._buffer[index] in "\n[" and _could_become_marker(self._buffer[index:]):
                return index
        return len(self._buffer)

    def finish(self) -> tuple[str, list[str]]:
        """Return (text still held back, follow-up questions)."""
        if self._found:
            return "", clean_questions(self._tail, limit=self.limit, question=self.question)
        remaining, self._buffer = self._buffer, ""
        return self._release(remaining), []

    @property
    def answer(self) -> str:
        return "".join(self._released).strip()

    def _release(self, text: str) -> str:
        if text:
            self._released.append(text)
        return text
