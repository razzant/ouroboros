"""The one tokenization rule and the one word predicate (``ouroboros/dialogue_words.py``, consensus K4.1)."""
from __future__ import annotations

import unicodedata

import pytest

from ouroboros.dialogue_words import Field, fields_of, match_fields, match_text, normalize, parse, tokens


def test_normalize_is_nfc_casefold_and_yo_as_ye():
    decomposed = unicodedata.normalize("NFD", "ЁЛКА Straße")
    assert normalize(decomposed) == "елка strasse"
    assert tokens("Чат_History, follow-up: «ёж»!") == ["чат_history", "follow", "up", "еж"]


def test_words_are_prefixes_in_any_order_and_all_required():
    query = parse(["сокет тайм"])
    assert match_text(query, "Таймаут на сокете вырос")
    assert match_text(query, "сокет … тайминг")
    assert not match_text(query, "сокет без второго слова")
    # a prefix starts a token, never sits inside one
    assert not match_text(parse(["кет"]), "сокет")


def test_case_nfc_and_yo_match_both_ways():
    assert match_text(parse(["ЁЖИК"]), "ежик в тумане")
    assert match_text(parse(["ежик"]), unicodedata.normalize("NFD", "Ёжик"))
    assert match_text(parse(["Café"]), unicodedata.normalize("NFD", "café au lait"))


def test_quoted_part_is_an_exact_token_or_a_phrase():
    assert match_text(parse(['"socket"']), "a socket here")
    assert not match_text(parse(['"sock"']), "a socket here")  # quoted: exact, never a prefix
    phrase = parse(['"chose (2"'])
    assert match_text(phrase, "[answer q1] chose (2) Merge now")
    assert not match_text(parse(['"merge now"']), "now merge")  # a phrase keeps its order
    assert match_text(parse(['«решили так» сок']), "Мы решили так: сокет")
    assert not match_text(parse(['«решили так» сок']), "Мы решили так")  # the loose word is still required


def test_alternatives_are_ored():
    query = parse(["alpha beta", "gamma"])
    assert match_text(query, "gamma only")
    assert match_text(query, "beta and alpha")
    assert not match_text(query, "alpha only")


@pytest.mark.parametrize("words, said", [
    ([], "words is empty"),
    ("socket", "is a list"),
    ([" , !"], "holds no word"),
    (['"open'], "not closed"),
    (['""'], "quoted part holds no word"),
    ([3], "not text"),
])
def test_unusable_query_is_an_error_fit_for_the_caller(words, said):
    with pytest.raises(ValueError, match=said):
        parse(words)


def test_one_alternative_is_satisfied_within_one_field():
    query = parse(["alpha beta"])
    fields = [("question", "alpha here"), ("comment", "beta there"), ("text", "beta then alpha")]
    assert match_fields(query, fields) == ["text"]
    assert match_fields(query, [("question", "alpha here"), ("comment", "beta there")]) == []
    # Fields pass as well as pairs, and every matching field is named in order
    assert match_fields(parse(["alpha"]), [Field("a", "alpha", "author"), Field("b", "x", "host"), Field("c", "alpha b", "owner")]) == ["a", "c"]


def _answer(**quiz):
    return {"ts": "2026-11-15T10:00:00Z", "type": "quiz_answer", "direction": "system", "task_id": "asker-1",
            "text": "[answer q1] chose (2) Merge now — \"after the exam\"",
            "quiz": {"quiz_id": "q1", "question": "When do we merge?", "options": ["Wait", "Merge now"],
                     "answered_index": 1, "comment": "after the exam", **quiz}}


def test_answered_card_fields_keep_their_own_speakers():
    fields = fields_of(_answer())
    assert fields == [
        Field("question", "When do we merge?", "asker"),
        Field("option 1", "Wait", "asker"),
        Field("option 2", "Merge now", "asker"),
        Field("press", "(2) Merge now", "owner", words=False),
        Field("comment", "after the exam", "owner"),
    ]
    # the host's frame of the same facts is not a field: the comment is found in the comment, once
    assert match_fields(parse(["after exam"]), fields) == ["comment"]
    assert match_fields(parse(["merge now"]), fields) == ["option 2", "press"]


def test_asked_card_and_closed_card_fields():
    asked = {"type": "quiz", "direction": "out", "text": "x",
             "quiz": {"quiz_id": "q2", "question": "Pick", "options": [{"label": "A"}, {"label": "B"}],
                      "option_details": ["first", ""]}}
    assert fields_of(asked) == [Field("question", "Pick", "author"), Field("option 1", "A — first", "author"),
                                Field("option 2", "B", "author")]
    closed = {"type": "quiz_closed", "direction": "system", "task_id": "t",
              "quiz": {"quiz_id": "q3", "question": "Q", "options": [], "closed_actor": "owner",
                       "closed_reason": "Closed by the owner without an answer."}}
    assert fields_of(closed)[-1] == Field("closure", "Closed by the owner without an answer.", "host")
    by_task = {**closed, "quiz": {**closed["quiz"], "closed_actor": "task", "closed_reason": "superseded"}}
    assert fields_of(by_task)[-1] == Field("closure", "superseded", "closer")


def test_secret_request_card_yields_only_its_non_value_receipt():
    secret = _answer(secret_request={"plugin_name": "tg", "fields": [{"id": "botToken", "setting_key": "K"}]},
                     secret_outcome={"event": "secret_filled", "pluginName": "tg", "fields": ["botToken"]},
                     comment="", answered_index=None)
    assert fields_of(secret) == [Field("receipt", "secret request q1: secret_filled: tg, botToken", "host")]
    asked = {"type": "quiz", "direction": "out", "text": "Enter the token",
             "quiz": {"quiz_id": "q9", "question": "Enter the token", "options": ["x"],
                      "secret_request": {"plugin_name": "tg", "fields": [{"id": "botToken"}]}}}
    assert fields_of(asked) == [Field("receipt", "secret request q9: tg, botToken", "host")]


def test_plain_rows_press_rows_and_empty_rows():
    assert fields_of({"direction": "in", "text": "hello"}) == [Field("text", "hello", "author")]
    press = {"direction": "in", "text": "2. Merge now", "corpus_answer": {"source": "owner_press"}}
    assert fields_of(press) == [Field("press", "2. Merge now", "owner", words=False)]
    assert fields_of({"direction": "out", "text": "  "}) == []
    facts = fields_of({"type": "task_summary", "direction": "system", "task_id": "t1", "status": "completed"})
    assert facts and facts[0].name == "text" and "host facts for t1" in facts[0].text
