"""Follow-up question extraction, for complete and streamed replies."""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from helpers import database
from helpers.follow_ups import StreamSplitter, clean_questions, split_answer

REPLY = (
    "Jane Eyre is £38.43 and in stock.\n"
    "[[FOLLOW_UPS]] Do you have other classics? | How long does delivery take?"
)


def test_split_answer_separates_answer_and_questions():
    answer, questions = split_answer(REPLY, limit=2)
    assert answer == "Jane Eyre is £38.43 and in stock."
    assert questions == ["Do you have other classics?", "How long does delivery take?"]


def test_a_reply_without_the_marker_is_returned_whole():
    assert split_answer("Just an answer.", limit=2) == ("Just an answer.", [])


@pytest.mark.parametrize("label", [
    "Follow-ups:", "Follow-up questions:", "**Follow-ups:**", "FOLLOW_UPS:", "[[ follow ups ]]",
])
def test_common_marker_variants_are_recognised(label):
    answer, questions = split_answer(f"We open at 9.\n{label} When do you close? | Where are you?", limit=2)
    assert answer == "We open at 9."
    assert questions == ["When do you close?", "Where are you?"]


def test_mentioning_follow_up_inside_a_sentence_is_not_cut():
    text = "Our team will follow up within a day: usually by email."
    assert split_answer(text, limit=2) == (text, [])


def test_questions_are_cleaned_capped_and_deduplicated():
    raw = "1. Do you deliver?\n- \"Do you deliver?\"\n* What are your hours?\nThird one?"
    assert clean_questions(raw, limit=2) == ["Do you deliver?", "What are your hours?"]


def test_the_question_just_asked_is_not_suggested_again():
    assert clean_questions("How much is it? | Is it in stock?", limit=2, question="How much is it") == [
        "Is it in stock?"
    ]


def test_questions_in_the_assistants_voice_are_dropped():
    raw = "Is there anything else you need? | Would you like to see more? | Do you deliver?"
    assert clean_questions(raw, limit=2) == ["Do you deliver?"]


def test_zero_limit_disables_suggestions():
    assert split_answer(REPLY, limit=0) == ("Jane Eyre is £38.43 and in stock.", [])


def _stream(text: str, size: int, limit: int = 2) -> tuple[str, str, list[str]]:
    splitter = StreamSplitter(limit=limit)
    shown = "".join(splitter.feed(text[i : i + size]) for i in range(0, len(text), size))
    rest, questions = splitter.finish()
    return shown + rest, splitter.answer, questions


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 1000])
def test_streaming_never_shows_the_marker_whatever_the_chunking(size):
    shown, answer, questions = _stream(REPLY, size)
    assert "FOLLOW" not in shown and "[[" not in shown
    assert answer == "Jane Eyre is £38.43 and in stock."
    assert questions == ["Do you have other classics?", "How long does delivery take?"]


@pytest.mark.parametrize("size", [1, 4, 1000])
def test_streaming_without_a_marker_shows_everything(size):
    text = "Line one.\nLine two with [brackets] and\nfollowed by more."
    shown, answer, questions = _stream(text, size)
    assert shown == text
    assert answer == text
    assert questions == []


@pytest.mark.parametrize("host, port, pooled", [
    ("ep-billowing-paper-bd6lkdy4.c-8.us-east-2.aws.neon.tech", None, False),
    ("ep-billowing-paper-bd6lkdy4-pooler.c-8.us-east-2.aws.neon.tech", None, True),
    ("aws-0-ap-south-1.pooler.supabase.com", 6543, True),
    ("localhost", 5432, False),
])
def test_statement_cache_is_disabled_only_behind_a_pooler(host, port, pooled):
    url = make_url(f"postgresql+asyncpg://u:p@{host}{':' + str(port) if port else ''}/db")
    assert database._uses_transaction_pooler(url) is pooled
