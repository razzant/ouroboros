"""#447: a parameter declared in a tool's public schema is honored on EVERY
dispatch branch that accepts the call (or that branch refuses by name).

Regressions of the same class are pinned here:
  D1 - read_file(start_char=...) was silently dropped by the active_workspace /
       system_repo / runtime_data branches (only task_drive & co honored it), so a
       long one-line file re-read the identical head forever; the reread-nudge
       cache also collided two different sub-line windows on those branches.
  D2 - write_file(mode="append") silently became overwrite on the repo roots and
       in the generic batch loop, destroying every prior chunk of a chunked
       large-file write while reporting success.
  D6 - query_code(op=structural) collected only `limit` rows before slicing
       rows[offset:], so page 2 was always empty and blamed the query.
  D7 - the mirror image: an UNDECLARED per-item key inside a `files`/`edits`
       payload was silently dropped, so write_file(files=[{..., "root":
       "task_drive"}]) bound every item to the ONE top-level root and wrote
       into the Ouroboros repo instead. A target the tool cannot honor is
       refused by name, atomically, never dropped.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from ouroboros.tools.core import _data_read, _read_file, _write_file
from ouroboros.tools.edit_ops import _edit_batch
from ouroboros.tools.registry import ToolContext, ToolRegistry

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


# ---------------------------------------------------------------------------
# D7: a payload item key the tool does NOT read is refused, never dropped
# ---------------------------------------------------------------------------

@pytest.mark.serial
def test_write_file_batch_refuses_a_per_item_root_it_cannot_honor(tmp_path):
    """Live defect: write_file(files=[{path, content, root}]) reads only path and
    content, binding every item to the ONE top-level root, so a per-item
    `root="task_drive"` was dropped and two scratch files landed in the
    Ouroboros repo while the call reported success."""
    ctx = _ctx(tmp_path)
    out = _write_file(ctx, files=[
        {"path": "scratch.py", "content": "print(1)\n", "root": "task_drive"},
    ])
    assert "TOOL_ARG_ERROR" in out, out
    assert "root" in out, out
    assert not (ctx.repo_dir / "scratch.py").exists(), "a refused write must not land in the repo"
    assert not (ctx.drive_root / "scratch.py").exists(), "nor on the drive it named"


@pytest.mark.serial
def test_write_file_batch_item_key_refusal_aborts_the_whole_batch(tmp_path):
    """All-or-nothing, like edit_batch's count mismatch: a clean sibling item is
    not written when another item declares a target this tool cannot honor."""
    ctx = _ctx(tmp_path)
    out = _write_file(ctx, files=[
        {"path": "clean.txt", "content": "A\n"},
        {"path": "stray.txt", "content": "B\n", "root": "task_drive"},
    ])
    assert "TOOL_ARG_ERROR" in out, out
    assert not (ctx.repo_dir / "clean.txt").exists(), "nothing is written before the refusal"
    assert not (ctx.repo_dir / "stray.txt").exists(), out


@pytest.mark.serial
def test_edit_batch_refuses_a_per_item_root_it_cannot_honor(tmp_path):
    """The sibling surface shares the class: edit items declare
    {path, old_str, new_str, count} and the tool's root enum is repo-only, so a
    per-item root would silently edit the repo instead of the named target."""
    ctx = _ctx(tmp_path)
    target = ctx.repo_dir / "mod.txt"
    target.write_text("alpha\n", encoding="utf-8")
    out = _edit_batch(ctx, edits=[
        {"path": "mod.txt", "old_str": "alpha", "new_str": "beta", "root": "task_drive"},
    ])
    assert "TOOL_ARG_ERROR" in out, out
    assert "root" in out, out
    assert target.read_text(encoding="utf-8") == "alpha\n", "a refused edit leaves the file untouched"


def test_published_item_schemas_are_derived_from_the_one_declaration():
    """The published schema is DERIVED from the same declaration the guard reads,
    so the two cannot drift: agreeing key sets are not enough, because a literal
    schema beside the constant is a second definition that can silently diverge."""
    from ouroboros.tools.core import _WRITE_FILE_ITEM_KEYS, _WRITE_FILE_ITEM_PROPERTIES
    from ouroboros.tools.core import get_tools as core_tools
    from ouroboros.tools.edit_ops import _EDIT_BATCH_ITEM_KEYS, _EDIT_BATCH_ITEM_PROPERTIES
    from ouroboros.tools.edit_ops import _EDIT_BATCH_ITEM_REQUIRED
    from ouroboros.tools.edit_ops import get_tools as edit_tools

    for tools, tool_name, payload_key, declared, allowed, required in (
        (core_tools(), "write_file", "files",
         _WRITE_FILE_ITEM_PROPERTIES, _WRITE_FILE_ITEM_KEYS, _WRITE_FILE_ITEM_KEYS),
        (edit_tools(), "edit_batch", "edits",
         _EDIT_BATCH_ITEM_PROPERTIES, _EDIT_BATCH_ITEM_KEYS, _EDIT_BATCH_ITEM_REQUIRED),
    ):
        entry = next(e for e in tools if e.name == tool_name)
        item_shape = entry.schema["parameters"]["properties"][payload_key]["items"]
        # Whole-property equality, not just the key set: a type or description that
        # drifts from the declaration is the same silent-divergence class.
        assert item_shape["properties"] == declared, tool_name
        assert item_shape["additionalProperties"] is False, tool_name
        assert tuple(allowed) == tuple(declared), tool_name
        assert item_shape["required"] == list(required), tool_name
        assert set(required) <= set(allowed), tool_name


def test_published_item_schema_is_a_copy_the_caller_cannot_corrupt():
    """get_tools() hands out schemas that other layers narrow in place (the acting
    subagent rewrites `root`), so the item properties must be a copy: sharing the
    module declaration would let one caller's edit rewrite the guard's vocabulary."""
    from ouroboros.tools.core import _WRITE_FILE_ITEM_PROPERTIES
    from ouroboros.tools.core import get_tools as core_tools

    def item_properties():
        entry = next(e for e in core_tools() if e.name == "write_file")
        return entry.schema["parameters"]["properties"]["files"]["items"]["properties"]

    mutated = item_properties()
    mutated["root"] = {"type": "string"}
    mutated["path"]["type"] = "corrupted"

    assert "root" not in _WRITE_FILE_ITEM_PROPERTIES
    assert _WRITE_FILE_ITEM_PROPERTIES["path"] == {"type": "string"}
    assert item_properties() == {"path": {"type": "string"}, "content": {"type": "string"}}


@pytest.mark.serial
def test_non_dict_batch_item_refuses_the_whole_call_on_every_root(tmp_path):
    """Same class as the per-item root, on the item's own shape rather than its
    keys: a `files` entry that is not an object was silently DROPPED by the
    runtime_data and generic loops while its siblings reported success (the repo
    lane refused with its own separate code). One pre-write guard now refuses the
    whole call, by name, identically on every root — nothing is written anywhere."""
    ctx = _ctx(tmp_path)
    ctx.task_id = "batch-argument-test"
    for root in ("runtime_data", "task_drive", "active_workspace", "artifact_store"):
        out = _write_file(ctx, files=[
            {"path": "kept.txt", "content": "A\n"},
            "not-an-object",
        ], root=root)
        assert "TOOL_ARG_ERROR" in out, (root, out)
        assert "file 2: not an object" in out, (root, out)
        assert "OK: wrote" not in out and "✅" not in out, (root, out)
    # The clean sibling never landed on the two roots whose base path is known here.
    assert not (ctx.repo_dir / "kept.txt").exists()
    assert not (ctx.drive_root / "kept.txt").exists()


@pytest.mark.serial
@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                 "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_declared_batch_items_keep_write_and_append_on_every_root(tmp_path, monkeypatch, root):
    ctx = _ctx(tmp_path)
    ctx.task_id = "batch-argument-test"
    owner_home = tmp_path / "owner"
    owner_home.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(owner_home))
    selectors = {}
    if root == "skill_payload":
        from tests.test_skill_exec import _build_skill

        _build_skill(ctx.drive_root / "skills" / "external", "alpha")
        selectors = {"bucket": "external", "skill_name": "alpha"}
    for mode, contents in [("overwrite", ("hello ", "first ")), ("append", ("мир", "second"))]:
        result = _write_file(ctx, files=[
            {"path": "a.txt", "content": contents[0]},
            {"path": "b.txt", "content": contents[1]},
        ], root=root, mode=mode, **selectors)
        assert "TOOL_ARG_ERROR" not in result and "PARTIAL_FAILURE" not in result, result
        assert result.startswith(("✅", "OK: wrote")), result
    assert "hello мир" in _read_file(ctx, "a.txt", root=root, **selectors)
    assert "first second" in _read_file(ctx, "b.txt", root=root, **selectors)


@pytest.mark.serial
def test_declared_edit_batch_items_keep_count_and_sequential_edits(tmp_path):
    ctx = _ctx(tmp_path)
    target = ctx.repo_dir / "edit.txt"
    target.write_text("alpha alpha\n", encoding="utf-8")
    result = _edit_batch(ctx, edits=[
        {"path": "edit.txt", "old_str": "alpha", "new_str": "beta", "count": 2},
        {"path": "edit.txt", "old_str": "beta beta", "new_str": "done"},
    ])
    assert result.startswith("✅ edit_batch applied 2 edit(s)"), result
    assert target.read_text(encoding="utf-8") == "done\n"


@pytest.mark.serial
def test_write_file_registry_refuses_json_string_batch_once_and_keeps_list_valid(tmp_path, monkeypatch):
    from ouroboros import safety

    ctx = _ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    files = [{"path": "batch.txt", "content": "x" * 5000}]
    result = registry.execute_result("write_file", {"root": "runtime_data", "files": json.dumps(files)})
    assert result.text.count("TOOL_ARG_ERROR") == 1
    assert result.status == "error"
    assert not (ctx.drive_root / "batch.txt").exists()
    assert not (ctx.repo_dir / "batch.txt").exists()

    result = registry.execute_result("write_file", {"root": "runtime_data", "files": files})
    assert result.status == "ok", result.text
    assert (ctx.drive_root / "batch.txt").read_text(encoding="utf-8") == files[0]["content"]
