"""TZ-1 cluster E: a missed exact edit says WHERE it missed, bounded.

``edit_text`` used to answer a miss with the first 2000 chars of the file — no
help for a miss at line 800 — and ``edit_batch``/``apply_patch`` with a bare
count or "context not found". ``locate_edit_miss`` is the one diagnosis all
three attach: the closest region, the first line whose bytes differ, the
read_file window to copy from, and (for a patch hunk) whether the region sits
before the hunk's search cursor. Every tier is bounded, so a miss on a huge
file costs one bounded pass and a bounded block.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from ouroboros.tools import edit_ops
from ouroboros.tools.core import _str_match_replace
from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch, locate_edit_miss


def _big_file(lines: int = 1200) -> str:
    text = "\n".join(f"line {i}: value = {i}" for i in range(1, lines + 1)) + "\n"
    return text.replace("line 800: value = 800", "        def compute(x):\n            return x * 800")


# --- tiers ----------------------------------------------------------------------

def test_indentation_miss_names_the_region_and_the_first_differing_line():
    out = locate_edit_miss(_big_file(), "    def compute(x):\n        return x * 800")
    assert out.startswith("old_str matches lines 800–801 ignoring whitespace (indentation differs"), out
    assert " 800|         def compute(x):" in out and " 801|             return x * 800" in out
    # The needle's first line is a SUFFIX of file line 800 (a needle may start
    # mid-line), so the first byte difference is the second line.
    assert "first difference at line 801:" in out
    assert "file   : '            return x * 800'" in out and "old_str: '        return x * 800'" in out
    assert "read_file start_line=800 max_lines=2" in out


def test_trailing_whitespace_and_inner_whitespace_are_named():
    assert "(trailing whitespace differs)" in locate_edit_miss("x = 1   \ny = 2\n", "x = 1\ny = 2")
    assert "(whitespace differs inside the line)" in locate_edit_miss("a\tb\n", "a b")


def test_a_needle_may_start_mid_line():
    text = "    total = compute(x) + 1\n        return total\n"
    out = locate_edit_miss(text, "compute(x) + 1\n    return total")
    assert out.startswith("old_str matches lines 1–2 ignoring whitespace (indentation differs"), out
    assert "first difference at line 2:" in out


def test_cr_line_endings_are_diagnosed_first():
    out = locate_edit_miss("a\nb\nc\n", "b\r\nc")
    assert out.startswith("old_str matches at line 2 once line endings are normalized"), out
    assert "carries CR" in out and "read_file start_line=2 max_lines=2" in out


def test_nearest_region_shows_the_typo():
    out = locate_edit_miss(_big_file(), "        def compute(y):\n            return y * 801")
    assert out.startswith("Nearest region: lines 800–801 (0 of 2 old_str line(s) match"), out
    assert "file   : '        def compute(x):'" in out and "old_str: '        def compute(y):'" in out


def test_ambiguous_relaxed_matches_are_listed():
    out = locate_edit_miss("  x = 1\ny\n  x = 1\n", "x = 1")
    assert "also at lines 3" in out, out


def test_nothing_similar_is_said_plainly():
    out = locate_edit_miss("a\nb\nc\n", "zzzzzzzz completely different")
    assert out.startswith("No line similar to old_str was found (4 lines scanned)"), out
    assert "search_code" in out
    assert locate_edit_miss("", "x").startswith("The file is empty")
    assert locate_edit_miss("a\n", "") == ""
    assert "whitespace only" in locate_edit_miss("a\n", "  \n  ")


def test_a_region_before_the_search_cursor_is_reported_as_an_ordering_problem():
    out = locate_edit_miss("one\ntwo\nthree\nfour\n", "two", cursor_line=3, needle_name="the hunk context")
    assert "the hunk context matches line 2 ignoring whitespace (the bytes match)" in out, out
    assert "BEFORE line 3" in out and "@@ anchor" in out
    assert "copy the exact bytes into the hunk context" in out


# --- bounded ----------------------------------------------------------------------

def test_a_huge_file_costs_one_bounded_pass_and_a_bounded_block():
    text = "\n".join(f"row {i} = {i % 97}" for i in range(60_000)) + "\n"
    needle = "\n".join(f"row {i} = {i % 89}" for i in range(30_000, 30_040))  # 40 nearly-right lines
    started = time.monotonic()
    out = locate_edit_miss(text, needle)
    assert time.monotonic() - started < 3.0
    assert len(out) <= edit_ops._LOCATE_MAX_CHARS
    assert "only the first 20000 of 60001 lines were scanned" in out, out[-200:]
    assert out.count("\n") <= edit_ops._LOCATE_EXCERPT_LINES + 12


def test_a_needle_longer_than_the_compared_window_is_not_explained_by_blank_lines():
    """TZ-1 PR1 review F2: only the first 200 needle lines are compared. When
    those all match and the real miss sits after them, the locator used to say
    "only leading/trailing blank lines differ" — a false diagnosis that sent the
    editor chasing whitespace. It now says which lines were never compared and
    widens the re-read window to the whole needle."""
    text = "\n".join(f"row {i}" for i in range(1, 301)) + "\n"
    needle_lines = [f"row {i}" for i in range(1, 251)]
    needle_lines[230] = "row 231 CHANGED"  # the miss is beyond the compared window
    out = locate_edit_miss(text, "\n".join(needle_lines))
    assert "only leading/trailing blank lines differ" not in out, out
    assert "the bytes match" not in out, out
    assert out.startswith("old_str matches lines 1–200 ignoring whitespace (its first 200 lines match exactly; "
                          "the difference is in the 50 lines after them, which were not compared)"), out
    assert "read_file start_line=1 max_lines=250" in out, out
    assert "(only the first 200 of 250 old_str lines were compared; the miss may be after them)" in out, out
    # A before-the-cursor region keeps the ordering note but never claims the bytes match.
    out = locate_edit_miss(text, "\n".join(needle_lines), cursor_line=290, needle_name="the hunk context")
    assert "the bytes match" not in out and "BEFORE line 290" in out, out
    # Tier 2 (a typo inside the compared window) discloses the window the same way.
    needle_lines[100] = "row 101 TYPO"
    out = locate_edit_miss(text, "\n".join(needle_lines))
    assert out.startswith("Nearest region: lines 1–200 (199 of 200 old_str line(s) match"), out
    assert "read_file start_line=1 max_lines=250" in out, out
    assert "(only the first 200 of 250 old_str lines were compared; the miss may be after them)" in out, out
    # A needle inside the window is untouched: no window note, the ordinary window.
    assert "were compared" not in locate_edit_miss(_big_file(), "    def compute(x):\n        return x * 800")


# --- the three editors --------------------------------------------------------------

def test_edit_text_miss_carries_the_locator_and_previews_only_a_small_file_whole():
    _new, err = _str_match_replace(_big_file(), "    def compute(x):\n        return x * 800", "X", "big.py", "EDIT_TEXT_ERROR")
    assert err.startswith("⚠️ EDIT_TEXT_ERROR: old_str not found in big.py.\nold_str matches lines 800–801"), err
    assert "File preview" not in err  # a head cut of a large file previews nothing about a miss
    _new, small = _str_match_replace("alpha\nbeta\n", "zeta", "q", "runtime_data:notes.txt", "EDIT_TEXT_ERROR")
    assert small.startswith("⚠️ EDIT_TEXT_ERROR: old_str not found in runtime_data:notes.txt.\nNearest region: line 2"), small
    assert small.endswith("File preview (whole file, 11 chars):\nalpha\nbeta\n"), small


class _FakeCtx:
    def __init__(self, repo: pathlib.Path):
        self.repo_dir = repo
        self.drive_root = repo / ".drive"
        self.task_metadata = {}
        self.task_id = "test-task"

    def is_workspace_mode(self):
        return True


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    repo = tmp_path / "ws"
    repo.mkdir()
    (repo / "m.py").write_text(_big_file(), encoding="utf-8")
    ctx = _FakeCtx(repo)
    from ouroboros.utils import safe_relpath

    def resolver(_ctx, path, _root, *, error_tag, _resolved_binding=None):
        return (repo / path).resolve(), safe_relpath(path), _resolved_binding, ""

    monkeypatch.setattr(edit_ops, "_resolve_edit_target", resolver)
    monkeypatch.setattr(edit_ops, "_finish_mutation", lambda ctx_, paths, tool, binding=None: "NOT committed.")
    return ctx


def test_edit_batch_miss_carries_the_locator_bounded_per_call(ws):
    misses = [{"path": "m.py", "old_str": f"    def compute(x):\n        return x * {800 + i}", "new_str": "X"}
              for i in range(5)]
    out = edit_ops._edit_batch(ws, misses)
    assert out.startswith("⚠️ EDIT_BATCH_ERROR: batch aborted, NOTHING was written (atomic)."), out
    assert "edit 1 (m.py): old_str occurs 0 time(s), expected 1." in out
    assert "      old_str matches lines 800–801 ignoring whitespace (indentation differs" in out, out
    assert out.count("Re-read that region") == edit_ops._LOCATE_BATCH_MAX  # five misses, three located
    assert (ws.repo_dir / "m.py").read_text(encoding="utf-8") == _big_file()


def test_apply_patch_context_miss_carries_the_locator():
    ops, err = _parse_patch("*** Update File: m.py\n-    def compute(x):\n-        return x * 800\n+pass\n")
    assert err == ""
    new, _notes, herr = _apply_hunks_to_text(_big_file(), ops[0].hunks, "m.py")
    assert new is None
    assert herr.startswith("hunk 1: context not found in m.py (searched from line 1)."), herr
    assert "the hunk context matches lines 800–801 ignoring whitespace (indentation differs" in herr, herr
    assert "read_file start_line=800 max_lines=2" in herr


def test_apply_patch_out_of_order_hunk_is_told_to_reorder():
    ops, err = _parse_patch("*** Update File: m.py\n c\n-d\n+D\n@@\n a\n-b\n+B\n")
    assert err == ""
    new, _notes, herr = _apply_hunks_to_text("a\nb\nc\nd\n", ops[0].hunks, "m.py")
    assert new is None
    assert herr.startswith("hunk 2: context not found in m.py (searched from line 5)."), herr
    assert "the hunk context matches lines 1–2 ignoring whitespace (the bytes match)" in herr, herr
    assert "BEFORE line 5" in herr and "move this hunk earlier or add an @@ anchor" in herr
