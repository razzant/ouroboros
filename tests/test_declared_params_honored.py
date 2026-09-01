"""#447: a parameter declared in a tool's public schema is honored on EVERY
dispatch branch that accepts the call (or that branch refuses by name).

Three regressions of the same class are pinned here:
  D1 - read_file(start_char=...) was silently dropped by the active_workspace /
       system_repo / runtime_data branches (only task_drive & co honored it), so a
       long one-line file re-read the identical head forever; the reread-nudge
       cache also collided two different sub-line windows on those branches.
  D2 - write_file(mode="append") silently became overwrite on the repo roots and
       in the generic batch loop, destroying every prior chunk of a chunked
       large-file write while reporting success.
  D6 - query_code(op=structural) collected only `limit` rows before slicing
       rows[offset:], so page 2 was always empty and blamed the query.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from ouroboros.tools.core import _data_read, _read_file, _write_file
from ouroboros.tools.registry import ToolContext

_NUDGE = "This exact view is unchanged"


def _ctx(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "data"
    drive.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return ToolContext(repo_dir=repo, drive_root=drive)


# ---------------------------------------------------------------------------
# D1: read_file start_char on the previously-broken roots
# ---------------------------------------------------------------------------

def test_read_file_start_char_honored_on_repo_and_data_roots(tmp_path):
    """The three branches that dropped start_char now advance within the line
    and disclose the sub-line cursor in the header, like task_drive always did."""
    ctx = _ctx(tmp_path)
    line = "0123456789ABCDEFGHIJ\n"
    for root in ("active_workspace", "system_repo", "runtime_data"):
        # One file per root: active_workspace and system_repo resolve to the same
        # repo here, and the reread nudge keys on the resolved path.
        base = ctx.drive_root if root == "runtime_data" else ctx.repo_dir
        (base / f"one_line_{root}.txt").write_text(line, encoding="utf-8")
        result = _read_file(ctx, f"one_line_{root}.txt", root=root, start_char=10)
        assert "(from char 10 of this window)" in result, (root, result)
        assert result.endswith("ABCDEFGHIJ\n"), (root, result)
        assert "0123456789" not in result, (root, result)


def test_read_file_distinct_start_char_windows_do_not_collide_in_reread_cache(tmp_path):
    """Two different sub-line windows are different views: the second must NOT be
    nudged as a re-read (the cache key used to omit start_char on these branches),
    while a true repeat of the same window still is."""
    ctx = _ctx(tmp_path)
    (ctx.repo_dir / "one_line.txt").write_text("0123456789ABCDEFGHIJ\n", encoding="utf-8")

    first = _read_file(ctx, "one_line.txt", root="active_workspace", start_char=0)
    assert _NUDGE not in first
    advanced = _read_file(ctx, "one_line.txt", root="active_workspace", start_char=10)
    assert _NUDGE not in advanced, "a different sub-line window is not a re-read"
    repeat = _read_file(ctx, "one_line.txt", root="active_workspace", start_char=10)
    assert _NUDGE in repeat, "a true repeat of the same window is still nudged"


def test_data_read_cognitive_full_read_shortcut_yields_to_start_char(tmp_path):
    """memory/* default reads return raw content; an explicit start_char is a
    cursor request and must be honored instead of silently swallowed."""
    ctx = MagicMock()
    ctx.drive_root = tmp_path
    ctx.drive_path.side_effect = lambda p: tmp_path / p
    target = tmp_path / "memory" / "scratchpad.md"
    target.parent.mkdir(parents=True)
    target.write_text("0123456789ABCDEFGHIJ\n", encoding="utf-8")

    assert _data_read(ctx, "memory/scratchpad.md") == "0123456789ABCDEFGHIJ\n"
    sliced = _data_read(ctx, "memory/scratchpad.md", start_char=10)
    assert "(from char 10 of this window)" in sliced
    assert sliced.endswith("ABCDEFGHIJ\n") and "0123456789" not in sliced


# ---------------------------------------------------------------------------
# D2: write_file mode="append" on the previously-broken branches
# ---------------------------------------------------------------------------

def test_write_file_append_on_repo_root_appends_instead_of_overwriting(tmp_path):
    ctx = _ctx(tmp_path)
    assert _write_file(ctx, path="chunks.py", content="def f():\n    pass\n",
                       root="active_workspace").startswith("✅")
    # The second chunk alone is not parseable Python: append must not run the
    # full-file syntax guard against a partial chunk.
    res = _write_file(ctx, path="chunks.py", content="    return 2\n",
                      root="active_workspace", mode="append")
    assert res.startswith("✅") and "appended" in res
    assert (ctx.repo_dir / "chunks.py").read_text(encoding="utf-8") == \
        "def f():\n    pass\n    return 2\n"


def test_write_file_append_on_repo_root_batch_form(tmp_path):
    ctx = _ctx(tmp_path)
    assert _write_file(ctx, files=[
        {"path": "a.txt", "content": "one"}, {"path": "b.txt", "content": "ONE"},
    ], root="active_workspace").startswith("✅")
    res = _write_file(ctx, files=[
        {"path": "a.txt", "content": "two"}, {"path": "b.txt", "content": "TWO"},
    ], root="active_workspace", mode="append")
    assert res.startswith("✅")
    assert (ctx.repo_dir / "a.txt").read_text(encoding="utf-8") == "onetwo"
    assert (ctx.repo_dir / "b.txt").read_text(encoding="utf-8") == "ONETWO"


def test_write_file_append_in_generic_batch_loop(tmp_path):
    """The generic batch loop (task_drive & co) had mode in scope and never
    consulted it; both chunks must survive a chunked batch append."""
    ctx = _ctx(tmp_path)
    assert "OK: wrote" in _write_file(ctx, files=[
        {"path": "log_a.txt", "content": "head-a "}, {"path": "log_b.txt", "content": "head-b "},
    ], root="task_drive")
    res = _write_file(ctx, files=[
        {"path": "log_a.txt", "content": "tail-a"}, {"path": "log_b.txt", "content": "tail-b"},
    ], root="task_drive", mode="append")
    assert "OK: wrote" in res and "PARTIAL_FAILURE" not in res
    read_a = _read_file(ctx, "log_a.txt", root="task_drive")
    read_b = _read_file(ctx, "log_b.txt", root="task_drive")
    assert "head-a tail-a" in read_a, read_a
    assert "head-b tail-b" in read_b, read_b


def test_write_file_repo_root_overwrite_still_default(tmp_path):
    ctx = _ctx(tmp_path)
    _write_file(ctx, path="o.txt", content="first version here", root="active_workspace")
    assert _write_file(ctx, path="o.txt", content="second version here",
                       root="active_workspace").startswith("✅")
    assert (ctx.repo_dir / "o.txt").read_text(encoding="utf-8") == "second version here"


# ---------------------------------------------------------------------------
# D6: query_code op=structural pagination
# ---------------------------------------------------------------------------

def _structural_page(ctx, offset):
    from ouroboros.tools.query_code import _query_code

    return _query_code(ctx, "structural", query="FunctionDef",
                       root="active_workspace", limit=40, offset=offset)


def test_structural_pagination_page_two_returns_the_next_rows(tmp_path):
    ctx = _ctx(tmp_path)
    for f in range(3):
        body = "\n".join(f"def fn_{f}_{i}():\n    return {i}" for i in range(30))
        (ctx.repo_dir / f"mod_{f}.py").write_text(body + "\n", encoding="utf-8")

    page1 = _structural_page(ctx, offset=0)
    page2 = _structural_page(ctx, offset=40)
    page3 = _structural_page(ctx, offset=80)

    assert "No results" not in page2, page2
    rows1 = set(page1.split("\n\n", 1)[1].splitlines())
    rows2 = set(page2.split("\n\n", 1)[1].splitlines())
    rows3 = set(page3.split("\n\n", 1)[1].splitlines())
    assert len(rows1) == 40 and len(rows2) == 40 and len(rows3) == 10
    assert not rows1 & rows2, "page 2 must be the rows page 1 did not show"
    assert not (rows1 | rows2) & rows3
    assert rows1 | rows2 | rows3 == {
        f"mod_{f}.py:{2 * i + 1} FunctionDef" for f in range(3) for i in range(30)
    }


def test_structural_pagination_beyond_cap_is_typed_truncation_not_no_results(tmp_path):
    """#447 S3: collection stops at the 200-row cap, so an offset beyond it used
    to render honest matches as "No results" (success-shaped completeness lie).
    It must be a typed truncation instead, and a capped full page must say the
    collection was capped rather than imply "N of N" completeness."""
    ctx = _ctx(tmp_path)
    for f in range(8):
        body = "\n".join(f"def fn_{f}_{i}():\n    return {i}" for i in range(30))
        (ctx.repo_dir / f"mod_{f}.py").write_text(body + "\n", encoding="utf-8")

    # The collector may overshoot the 200 cap by up to one file's rows; the tail
    # page past the cap must disclose the cap instead of implying completeness.
    tail = _structural_page(ctx, offset=200)
    assert "collection capped at 200" in tail.splitlines()[0], tail[:200]
    assert "No results" not in tail

    beyond = _structural_page(ctx, offset=400)
    assert beyond.startswith("⚠️ QUERY_CODE_TRUNCATED"), beyond[:200]
    assert "No results" not in beyond
