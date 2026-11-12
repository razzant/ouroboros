"""The two-way change between two note versions is exact in both directions, whatever the diff finds."""
from __future__ import annotations

import random

import pytest

from ouroboros import text_change as tc

CASES = {
    "empty to empty": ("", ""),
    "empty to text": ("", "# Note\n\nFirst.\n"),
    "text to empty": ("# Note\n\nFirst.\n", ""),
    "crlf": ("# Река\r\nСтарый берег.\r\nДо встречи.\r\n", "# Река\r\nНовый берег.\r\nДо встречи.\r\n"),
    "no final newline": ("one\ntwo\nthree", "one\n2\nthree"),
    "final newline added": ("one\ntwo", "one\ntwo\n"),
    "multibyte inside a line": ("Привет, мир 🌍!\n", "Привет, дивный мир 🌏!\n"),
    "repeated lines": ("- a\n- a\n- a\n\n- b\n- a\n", "- a\n- b\n- a\n- a\n\n- a\n"),
    "pure insertion": ("# A\n\nkeep\n", "# A\n\nnew line\nkeep\n"),
    "pure deletion": ("# A\n\ngone\nkeep\n", "# A\n\nkeep\n"),
    "full rewrite": ("alpha\nbeta\ngamma\n", "один\nдва\n"),
    "lone carriage return": ("a\rb\nc\n", "a\rB\nc\n"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_change_rebuilds_both_versions_byte_exactly(name):
    old, new = CASES[name]
    change = tc.exact_change(old, new)
    assert tc.apply_change(old, change) == new
    assert tc.unapply_change(new, change) == old
    starts = [start for start, _removed, _added in change]
    assert starts == sorted(starts)
    raw = old.encode("utf-8")
    for start, removed, _added in change:  # every span sits where it says, on character boundaries
        assert raw[start:start + len(removed.encode("utf-8"))].decode("utf-8") == removed
    assert (change == []) == (old == new)


def test_random_versions_round_trip():
    rng = random.Random(20261112)
    pieces = ["# H\n", "\n", "- item\n", "Текст строки.\n", "line\r\n", "🌍", "x", "## Sub\n"]
    for _ in range(300):
        old = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 40)))
        new = list(old)
        for _ in range(rng.randrange(0, 6)):
            at = rng.randrange(0, len(new) + 1)
            if rng.random() < 0.5 and new:
                del new[at:at + rng.randrange(1, 8)]
            else:
                new[at:at] = rng.choice(pieces)
        new = "".join(new)
        change = tc.compute_change(old, new)
        assert tc.apply_change(old, change) == new and tc.unapply_change(new, change) == old


def test_counts_separate_what_left_from_what_came():
    change = tc.exact_change("The host is old.\n", "The host is new.\n")
    assert change == [(12, "old", "new")]
    assert tc.change_chars(change) == (3, 3)  # a net-zero replacement still removed and added


def test_a_change_that_does_not_fit_its_text_refuses():
    change = tc.exact_change("one two\n", "one 2\n")
    with pytest.raises(ValueError):
        tc.apply_change("one three\n", change)
    with pytest.raises(ValueError):
        tc.unapply_change("one two\n", change)
    with pytest.raises(ValueError):
        tc.apply_change("abc", [(2, "c", ""), (0, "a", "")])  # out of order
    with pytest.raises(ValueError):
        tc.apply_change("abc", [(9, "", "x")])  # past the end


def test_an_unproven_diff_falls_back_to_one_whole_span(monkeypatch):
    monkeypatch.setattr(tc, "compute_change", lambda old, new: [(0, "wrong", "")])
    assert tc.exact_change("old text", "new text") == [(0, "old text", "new text")]
    assert tc.exact_change("same", "same") == []


def test_repeated_lines_past_the_work_bound_still_change_exactly(monkeypatch):
    rng = random.Random(7)
    vocab = ["- item\n", "\n", "## Notes\n", "text line here\n", "---\n", "* point\n"]
    old = "".join(rng.choice(vocab) for _ in range(4000))
    new = "".join(rng.choice(vocab) for _ in range(4000))
    matched = []
    real = tc.SequenceMatcher

    def counting(*args, **kwargs):
        matched.append(kwargs.get("autojunk", True))
        return real(*args, **kwargs)

    monkeypatch.setattr(tc, "SequenceMatcher", counting)
    monkeypatch.setattr(tc, "_MATCH_PAIRS", 10_000)
    change = tc.exact_change(old, new)
    assert tc.apply_change(old, change) == new and tc.unapply_change(new, change) == old
    assert False not in matched  # past the bound the unjunked matcher never ran
    monkeypatch.setattr(tc, "_MATCH_PAIRS", 10**12)
    matched.clear()
    fine = tc.exact_change(old, new)
    assert matched == [False] and tc.apply_change(old, fine) == new
