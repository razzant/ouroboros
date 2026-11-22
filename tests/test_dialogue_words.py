"""The one tokenization rule and the one word predicate (``ouroboros/dialogue_words.py``, consensus K4.1)."""
from __future__ import annotations

import json
import unicodedata

import pytest

from ouroboros.dialogue_words import Field, excerpt, fields_of, locate, match_fields, match_text, normalize, parse, tokens


def test_normalize_is_nfc_casefold_and_yo_as_ye():
    decomposed = unicodedata.normalize("NFD", "ЁЛКА Straße")
    assert normalize(decomposed) == "елка strasse"
    assert tokens("Чат_History, follow-up: «ёж»!") == ["чат", "history", "follow", "up", "еж"]


def test_underscore_splits_identifiers_into_findable_words():
    assert tokens("see memory_view.py") == ["see", "memory", "view", "py"]
    assert match_text(parse(["history"]), "the chat_history tool")
    assert match_text(parse(["chat_history"]), "history of a chat")  # unquoted: two prefixes, any order
    assert match_text(parse(['"chat_history"']), "run chat_history now")  # quoted: the consecutive pair
    assert not match_text(parse(['"chat_history"']), "history of a chat")


def test_combining_marks_stay_inside_their_word():
    assert tokens("हिन्दी नमस्ते") == ["हिन्दी", "नमस्ते"]
    assert tokens("שָׁלוֹם עולם") == ["שָׁלוֹם", "עולם"]
    assert match_text(parse(["नमस"]), "हिन्दी नमस्ते")
    # a mark with no word character before it starts nothing
    assert tokens("\u0301x") == ["x"]


def test_text_without_spaces_is_one_token_per_run_a_named_limit():
    assert tokens("日本語のテキスト") == ["日本語のテキスト"]
    assert match_text(parse(["日本"]), "日本語のテキスト")
    assert not match_text(parse(["テキスト"]), "日本語のテキスト")  # a piece inside a run: use search


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
    # a host facts row without text is the host's typed facts, not anyone's words: no field
    assert fields_of({"type": "task_summary", "direction": "system", "task_id": "t1", "status": "completed"}) == []
    assert fields_of({"type": "task_summary", "direction": "system", "text": "done: merged"}) == [
        Field("text", "done: merged", "author")]


def test_the_receipt_says_what_owner_quiz_says():
    from ouroboros.owner_quiz import secret_outcome_summary

    block = {"quiz_id": "q1", "secret_request": {"plugin_name": "tg"},
             "secret_outcome": {"event": "secret_replaced", "pluginName": "tg", "fields": ["a", "b"]}}
    assert fields_of({"type": "quiz_answer", "quiz": block})[0].text == f"secret request q1: {secret_outcome_summary(block)}"


_EVERY_KIND = [
    {"direction": "in", "text": "plain"},
    {"direction": "in", "text": "2. x", "corpus_answer": {"source": "owner_press"}},
    {"type": "task_summary", "direction": "system", "task_id": "t", "status": "completed"},
    {"type": "task_summary", "direction": "system", "text": "facts"},
    {"type": "quiz", "direction": "out", "quiz": {"quiz_id": "q", "question": "Q", "options": [{"label": "A"}],
                                                  "option_details": ["d"]}},
    {"type": "quiz_answer", "task_id": "t", "quiz": {"quiz_id": "q", "question": "Q", "options": ["A"],
                                                     "answered_index": 0, "comment": "c"}},
    {"type": "quiz_closed", "quiz": {"quiz_id": "q", "question": "Q", "options": [], "closed_reason": "r"}},
    {"type": "quiz_answer", "quiz": {"quiz_id": "q", "secret_request": {"plugin_name": "p", "fields": [{"id": "f"}]},
                                     "secret_outcome": {"event": "secret_filled", "pluginName": "p", "fields": ["f"]}}},
    {"type": "quiz", "quiz": {"quiz_id": "q", "secret_request": {"plugin_name": "p", "fields": [{"id": "f"}]}}},
]


@pytest.mark.parametrize("blocked", [True, False])
def test_fields_of_runs_on_every_row_kind_with_no_store_importable(blocked, tmp_path):
    """The offline harness calls fields_of with the store modules made unimportable; so does this test (and,
    the other way, with them importable, so the guard is the blocker and not a broken call)."""
    import subprocess
    import sys
    from pathlib import Path

    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.abc, json, sys\n"
        "BLOCKED = ('sqlite3', '_sqlite3', 'ouroboros.store', 'ouroboros.dialogue_store', 'ouroboros.task_store',\n"
        "           'ouroboros.chronicle_store', 'ouroboros.owner_quiz', 'ouroboros.dialogue_provenance',\n"
        "           'ouroboros.config', 'ouroboros.contracts')\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if any(name == b or name.startswith(b + '.') for b in BLOCKED):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "if sys.argv[2] == 'blocked':\n"
        "    sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from ouroboros.dialogue_words import fields_of\n"
        "print(json.dumps([[list(f) for f in fields_of(row)] for row in json.loads(sys.stdin.read())]))\n",
        encoding="utf-8")
    repo = str(Path(__file__).resolve().parents[1])
    done = subprocess.run([sys.executable, str(script), repo, "blocked" if blocked else "open"],
                          input=json.dumps(_EVERY_KIND), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert [[tuple(f) for f in kind] for kind in json.loads(done.stdout)] == [
        [tuple(f) for f in fields_of(row)] for row in _EVERY_KIND]


_FILLER = "Filler sentence goes here. " * 20


@pytest.mark.parametrize("words, text, found", [
    (["art"], "A smart start. " + _FILLER + "Then the art of it.", "art"),  # inside another word earlier
    (["needle"], "Straße. " + _FILLER + "Here is the needle.", "needle"),  # normalizing changes the length
    (["plan zebra", "needle"], "The plan is first. " + _FILLER + "Here is the needle.", "needle"),  # not satisfied
    (['"socket timeout"'], "Words. " + _FILLER + "the socket,  timeout came.", "socket,  timeout"),  # parted
    (["needle"], "Plain. " + _FILLER + "a needle here", "needle"),  # the control
])
def test_the_one_locator_finds_the_matched_run_in_the_original_text(words, text, found):
    query = parse(words)
    start, end = locate(query, text)
    assert text[start:end] == found
    shown = excerpt(query, text, 240)
    assert " ".join(found.split()) in shown and f"[{len(text)} chars in full]" in shown


def test_the_locator_says_none_where_nothing_matches_and_a_short_field_is_whole():
    assert locate(parse(["zebra"]), "no such animal") is None
    assert excerpt(parse(["zebra"]), "no such   animal") == "no such animal"
    assert excerpt(parse(["x"]), "x" * 300).startswith("x" * 240) and "[300 chars in full]" in excerpt(parse(["x"]), "x" * 300)


def test_a_quoted_part_against_its_neighbours_never_joins_them():
    assert match_text(parse(['one"two"three']), "one two three")
    assert not match_text(parse(['one"two"three']), "onethree two")
