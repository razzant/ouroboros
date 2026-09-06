"""Tests for ouroboros.tools.edit_ops (apply_patch / edit_batch)."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ouroboros.tools import edit_ops
from ouroboros.tools.edit_ops import (
    _apply_hunks_to_text,
    _find_sequence,
    _parse_patch,
    _syntax_check,
)


SAMPLE = "\n".join([
    "def ddd(x):",
    "    return x * 3",
    "",
    "",
    "def other(x):",
    "    return ddd(x)",
    "",
    "def caller():",
    "    return ddd(1) + ddd(2)",
    "",
])


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def test_parse_patch_update_add_delete():
    ops, err = _parse_patch(
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "@@ def ddd\n"
        " def ddd(x):\n"
        "-    return x * 3\n"
        "+    return x * 4\n"
        "*** Add File: b.py\n"
        "+print('hi')\n"
        "*** Delete File: c.py\n"
        "*** End Patch\n"
    )
    assert err == ""
    assert [op.kind for op in ops] == ["update", "add", "delete"]
    assert ops[0].path == "a.py"
    assert ops[0].hunks[0].anchor == "def ddd"
    assert ops[1].add_lines == ["print('hi')"]


def test_parse_patch_tolerates_decorative_asterisks():
    ops, err = _parse_patch(
        "*** Begin Patch ***\n"
        "*** Update File: a.py ***\n"
        "-old\n"
        "+new\n"
        "*** End Patch ***\n"
    )
    assert err == ""
    assert ops[0].path == "a.py"


def test_parse_patch_envelope_optional():
    ops, err = _parse_patch(
        "*** Update File: a.py\n"
        " context\n"
        "-old\n"
        "+new\n"
    )
    assert err == ""
    assert len(ops) == 1


def test_parse_patch_rejects_stray_content():
    _, err = _parse_patch("hello\n*** Update File: a.py\n-x\n+y\n")
    assert "before the first file header" in err


def test_parse_patch_rejects_bad_add_body():
    _, err = _parse_patch("*** Add File: a.py\nno-plus-prefix\n")
    assert "must start with '+'" in err


def test_parse_patch_rejects_empty():
    _, err = _parse_patch("*** Begin Patch\n*** End Patch\n")
    assert "no file operations" in err


# ---------------------------------------------------------------------------
# hunk matching / application
# ---------------------------------------------------------------------------

def test_apply_single_hunk():
    ops, err = _parse_patch(
        "*** Update File: s.py\n"
        " def ddd(x):\n"
        "-    return x * 3\n"
        "+    return x * 30\n"
    )
    assert err == ""
    new, notes, herr = _apply_hunks_to_text(SAMPLE, ops[0].hunks, "s.py")
    assert herr == ""
    assert "x * 30" in new
    assert notes == []


def test_ambiguous_context_errors():
    content = "a\nb\na\nb\n"
    ops, _ = _parse_patch("*** Update File: s.py\n a\n-b\n+B\n")
    new, _, herr = _apply_hunks_to_text(content, ops[0].hunks, "s.py")
    assert new is None
    assert "ambiguous" in herr


def test_anchor_disambiguates():
    content = "def one():\n    x = 1\n\ndef two():\n    x = 1\n"
    ops, _ = _parse_patch(
        "*** Update File: s.py\n"
        "@@ def two\n"
        "-    x = 1\n"
        "+    x = 2\n"
    )
    new, _, herr = _apply_hunks_to_text(content, ops[0].hunks, "s.py")
    assert herr == ""
    assert new == "def one():\n    x = 1\n\ndef two():\n    x = 2\n"


def test_context_not_found_reports_lines():
    ops, _ = _parse_patch("*** Update File: s.py\n-does not exist\n+x\n")
    new, _, herr = _apply_hunks_to_text(SAMPLE, ops[0].hunks, "s.py")
    assert new is None
    assert "context not found" in herr


def test_fuzzy_trailing_whitespace_match():
    content = "line one   \nline two\n"
    ops, _ = _parse_patch("*** Update File: s.py\n-line one\n+line ONE\n")
    new, notes, herr = _apply_hunks_to_text(content, ops[0].hunks, "s.py")
    assert herr == ""
    assert new.startswith("line ONE")
    assert any("whitespace" in n for n in notes)


def test_pure_insertion_requires_anchor():
    ops, _ = _parse_patch("*** Update File: s.py\n+new line\n")
    new, _, herr = _apply_hunks_to_text(SAMPLE, ops[0].hunks, "s.py")
    assert new is None
    assert "anchor" in herr


def test_pure_insertion_with_anchor():
    ops, _ = _parse_patch("*** Update File: s.py\n@@ def other\n+    # inserted\n")
    new, _, herr = _apply_hunks_to_text(SAMPLE, ops[0].hunks, "s.py")
    assert herr == ""
    assert "def other(x):\n    # inserted\n    return ddd(x)" in new


def test_sequential_hunks_advance_cursor():
    ops, _ = _parse_patch(
        "*** Update File: s.py\n"
        "-    return ddd(x)\n"
        "+    return aaa(x)\n"
        "@@ def caller\n"
        "-    return ddd(1) + ddd(2)\n"
        "+    return aaa(1) + aaa(2)\n"
    )
    new, _, herr = _apply_hunks_to_text(SAMPLE, ops[0].hunks, "s.py")
    assert herr == ""
    assert "aaa(x)" in new and "aaa(1) + aaa(2)" in new
    assert "return ddd" not in new
    assert "def ddd(x):" in new  # the def line was not part of either hunk


def test_find_sequence_caps_matches():
    lines = ["x"] * 20
    assert len(_find_sequence(lines, ["x"], 0, fuzzy=False)) == 5


# ---------------------------------------------------------------------------
# shared verification helpers
# ---------------------------------------------------------------------------

def test_syntax_check():
    assert _syntax_check("x.py", "def f(:\n") != ""
    assert _syntax_check("x.py", "def f():\n    return 1\n") == ""
    assert _syntax_check("x.json", "{bad") != ""
    assert _syntax_check("x.json", '{"ok": 1}') == ""
    assert _syntax_check("x.txt", "anything") == ""


# ---------------------------------------------------------------------------
# end-to-end handler tests on a fake workspace ctx
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, repo: pathlib.Path):
        self._repo = repo
        self.repo_dir = repo
        self.drive_root = repo / ".drive"
        self.task_metadata = {}
        self.event_queue = None
        self.pending_events = []
        self.task_id = "test-task"

    def is_workspace_mode(self):
        return True

    def repo_path(self, rel):
        p = (self._repo / rel).resolve()
        if not str(p).startswith(str(self._repo.resolve())):
            raise ValueError(f"path escapes workspace: {rel}")
        return p


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    repo = tmp_path / "ws"
    repo.mkdir()
    (repo / "s.py").write_text(SAMPLE, encoding="utf-8")
    ctx = _FakeCtx(repo)
    # Route guard helpers around ToolContext specifics: keep the real access
    # logic out of scope — these tests exercise edit mechanics.
    monkeypatch.setattr(edit_ops, "_resolve_edit_target", _fake_resolver(ctx))
    monkeypatch.setattr(
        edit_ops,
        "_finish_mutation",
        lambda ctx_, paths, tool, binding=None: "NOT committed.",
    )
    return ctx


def _fake_resolver(ctx):
    from ouroboros.utils import safe_relpath

    def resolver(_ctx, path, _root, *, error_tag, _resolved_binding=None):
        if not path:
            return None, "", None, f"⚠️ {error_tag}: path is required."
        try:
            return ctx.repo_path(path), safe_relpath(path), _resolved_binding, ""
        except ValueError as e:
            return None, "", None, f"⚠️ PATH_ERROR: {e}"
    return resolver


def test_apply_patch_end_to_end(ws):
    result = edit_ops._apply_patch(
        ws,
        "*** Begin Patch\n"
        "*** Update File: s.py\n"
        "-def ddd(x):\n"
        "+def aaa(x):\n"
        "@@ def other\n"
        "-    return ddd(x)\n"
        "+    return aaa(x)\n"
        "*** Add File: extra.py\n"
        "+VALUE = 1\n"
        "*** End Patch\n",
    )
    assert result.startswith("✅")
    text = (ws.repo_dir / "s.py").read_text()
    assert "def aaa(x):" in text and "return aaa(x)" in text
    assert (ws.repo_dir / "extra.py").read_text() == "VALUE = 1\n"


def test_apply_patch_atomic_on_bad_hunk(ws):
    before = (ws.repo_dir / "s.py").read_text()
    result = edit_ops._apply_patch(
        ws,
        "*** Update File: s.py\n"
        "-def ddd(x):\n"
        "+def aaa(x):\n"
        "@@ def nowhere\n"
        "-missing\n"
        "+present\n",
    )
    assert "APPLY_PATCH_ERROR" in result
    assert (ws.repo_dir / "s.py").read_text() == before


def test_apply_patch_add_existing_fails(ws):
    result = edit_ops._apply_patch(ws, "*** Add File: s.py\n+x\n")
    assert "already exists" in result


def test_apply_patch_delete(ws):
    (ws.repo_dir / "gone.py").write_text("x = 1\n")
    result = edit_ops._apply_patch(ws, "*** Delete File: gone.py\n")
    assert result.startswith("✅")
    assert not (ws.repo_dir / "gone.py").exists()


def test_edit_batch_counted_replace(ws):
    result = edit_ops._edit_batch(
        ws,
        [
            {"path": "s.py", "old_str": "ddd(", "new_str": "aaa(", "count": 4},
        ],
    )
    assert result.startswith("✅")
    text = (ws.repo_dir / "s.py").read_text()
    assert "ddd(" not in text
    assert text.count("aaa(") == 4


def test_edit_batch_count_mismatch_is_atomic(ws):
    before = (ws.repo_dir / "s.py").read_text()
    result = edit_ops._edit_batch(
        ws,
        [
            {"path": "s.py", "old_str": "def other", "new_str": "def another", "count": 1},
            {"path": "s.py", "old_str": "ddd(", "new_str": "aaa(", "count": 2},  # actually 4
        ],
    )
    assert "EDIT_BATCH_ERROR" in result
    assert "occurs 4 time(s), expected 2" in result
    assert (ws.repo_dir / "s.py").read_text() == before


def test_edit_batch_sequential_edits_see_prior_results(ws):
    result = edit_ops._edit_batch(
        ws,
        [
            {"path": "s.py", "old_str": "def ddd(x):", "new_str": "def aaa(x):", "count": 1},
            {"path": "s.py", "old_str": "def aaa(x):", "new_str": "def aaa(value):", "count": 1},
        ],
    )
    assert result.startswith("✅")
    assert "def aaa(value):" in (ws.repo_dir / "s.py").read_text()


def test_registry_registration():
    names = {e.name for e in edit_ops.get_tools()}
    assert names == {"apply_patch", "edit_batch"}
    for entry in edit_ops.get_tools():
        assert entry.is_code_tool
        assert entry.mutates_worktree


# ---------------------------------------------------------------------------
# write_file rails (git._repo_write): syntax guard + overwrite diff
# ---------------------------------------------------------------------------

def _ws_ctx(tmp_path):
    import subprocess

    from ouroboros.tools.registry import ToolContext

    ws = tmp_path / "extws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    (ws / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "seed"], cwd=ws, check=True)
    drive = tmp_path / "drive"
    drive.mkdir()
    return ToolContext(repo_dir=tmp_path / "repo", drive_root=drive,
                       workspace_root=str(ws), workspace_mode="external"), ws


def test_repo_write_blocks_broken_python(tmp_path):
    from ouroboros.tools.git import _repo_write

    ctx, ws = _ws_ctx(tmp_path)
    before = (ws / "mod.py").read_text()
    out = _repo_write(ctx, path="mod.py", content="def f(:\n    broken\n")
    assert "WRITE_BLOCKED_SYNTAX" in out
    assert (ws / "mod.py").read_text() == before


def test_repo_write_force_bypasses_syntax_guard(tmp_path):
    from ouroboros.tools.git import _repo_write

    ctx, ws = _ws_ctx(tmp_path)
    out = _repo_write(ctx, path="broken_fixture.py", content="def f(:\n", force=True)
    assert out.startswith("✅")
    assert (ws / "broken_fixture.py").exists()


def test_repo_write_overwrite_appends_diff(tmp_path):
    from ouroboros.tools.git import _repo_write

    ctx, ws = _ws_ctx(tmp_path)
    out = _repo_write(ctx, path="mod.py", content="def f():\n    return 2\n")
    assert out.startswith("✅")
    assert "Diff vs the previous version" in out
    assert "-    return 1" in out and "+    return 2" in out


def test_repo_write_new_file_has_no_diff_section(tmp_path):
    from ouroboros.tools.git import _repo_write

    ctx, ws = _ws_ctx(tmp_path)
    out = _repo_write(ctx, path="fresh.py", content="X = 1\n")
    assert out.startswith("✅")
    assert "Diff vs the previous version" not in out


# ---------------------------------------------------------------------------
# governance rails: envelopes, advisory staleness (P3), force disclosure
# ---------------------------------------------------------------------------

def test_capability_profiles_pin_new_tools():
    # Write-capable lanes see the tools; the read-only subagent lane and the
    # heal-mode allowlist must NOT (P3: the read-only lane stays write-free,
    # and heal mode edits skill payloads, which these tools refuse).
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )
    from ouroboros.tools.registry import _HEAL_MODE_ALLOWED_TOOLS

    for name in ("apply_patch", "edit_batch"):
        assert name in CORE_TOOL_NAMES
        assert name in ACTING_SUBAGENT_TOOL_NAMES
        assert name not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
        assert name not in _HEAL_MODE_ALLOWED_TOOLS


def test_tool_policy_and_smoke_registration():
    from ouroboros.safety import TOOL_POLICY

    assert TOOL_POLICY["apply_patch"] == TOOL_POLICY["edit_text"]
    assert TOOL_POLICY["edit_batch"] == TOOL_POLICY["edit_text"]


def test_mutations_invalidate_advisory(tmp_path, monkeypatch):
    # P3: every worktree-mutating tool must mark the advisory snapshot stale.
    calls = []
    from ouroboros.tools import commit_gate

    monkeypatch.setattr(
        commit_gate, "_invalidate_advisory",
        lambda ctx, **kw: calls.append(kw.get("source_tool")),
    )
    repo = tmp_path / "ws"
    repo.mkdir()
    (repo / "s.py").write_text(SAMPLE, encoding="utf-8")
    ctx = _FakeCtx(repo)
    monkeypatch.setattr(edit_ops, "_resolve_edit_target", _fake_resolver(ctx))
    out = edit_ops._apply_patch(ctx, "*** Update File: s.py\n-def ddd(x):\n+def aaa(x):\n")
    assert out.startswith("✅")
    out = edit_ops._edit_batch(ctx, [{"path": "s.py", "old_str": "aaa", "new_str": "bbb", "count": 1}])
    assert out.startswith("✅")
    assert calls == ["apply_patch", "edit_batch"]


def test_repo_write_force_bypass_is_disclosed(tmp_path):
    # P3: silent bypass is forbidden — a forced write of invalid content names it.
    from ouroboros.tools.git import _repo_write

    ctx, ws = _ws_ctx(tmp_path)
    out = _repo_write(ctx, path="fixture_broken.py", content="def f(:\n", force=True)
    assert out.startswith("✅")
    assert "SYNTAX_GUARD_BYPASSED" in out


# ---------------------------------------------------------------------------
# guard parity with edit_text (the fences these tools must NOT be weaker than)
# ---------------------------------------------------------------------------

def _guard_registry(tmp_path):
    """A registry over a throwaway repo, so guards run for real (no fake resolver)."""
    import subprocess

    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    drive = tmp_path / "drive"
    drive.mkdir()
    return ToolRegistry(repo_dir=repo, drive_root=drive), repo


def _protected_call(tool, path):
    if tool == "edit_text":
        return {"path": path, "old_str": "P1 honest", "new_str": "P1 loose"}
    if tool == "apply_patch":
        return {"patch": f"*** Update File: {path}\n-P1 honest\n+P1 loose\n"}
    return {"edits": [{"path": path, "old_str": "P1 honest", "new_str": "P1 loose"}]}


@pytest.mark.parametrize("tool", ["edit_text", "apply_patch", "edit_batch"])
@pytest.mark.parametrize("spelling", ["canonical", "absolute", "root_basename"])
def test_protected_path_blocked_in_every_spelling(tmp_path, tool, spelling):
    """A protected path is refused however it is SPELLED.

    ``ctx.repo_path`` collapses an absolute-inside-root path and a redundant
    root-basename prefix onto the same file, so a guard reading the raw spelling
    would pass `repo/BIBLE.md` straight through to a write on `BIBLE.md`.
    edit_text is canonicalized at dispatch; apply_patch/edit_batch carry their
    paths inside the payload and must canonicalize before their own guards.
    """
    reg, repo = _guard_registry(tmp_path)
    spellings = {
        "canonical": "BIBLE.md",
        "absolute": str(repo / "BIBLE.md"),
        "root_basename": "repo/BIBLE.md",
    }
    (repo / "BIBLE.md").write_text("P1 honest\n", encoding="utf-8")
    result = str(reg.execute(tool, _protected_call(tool, spellings[spelling])))
    assert "BLOCKED" in result, result[:200]
    assert (repo / "BIBLE.md").read_text() == "P1 honest\n"


def _workspace_guard_registry(tmp_path, monkeypatch):
    import subprocess

    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = tmp_path / "system"
    project = tmp_path / "project"
    drive = tmp_path / "drive"
    for path in (system, project, drive):
        path.mkdir()
    for repo in (system, project):
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=drive,
        workspace_root=project,
        workspace_mode="external",
    )
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    registry.set_context(ctx)
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    return registry, ctx, system, project


@pytest.mark.parametrize("tool", ["apply_patch", "edit_batch"])
def test_repo_batch_tool_explicit_system_binding_mutates_only_system_and_invalidates_it(
    tmp_path, monkeypatch, tool,
):
    from ouroboros.tools import commit_gate

    registry, _ctx, system, project = _workspace_guard_registry(tmp_path, monkeypatch)
    (system / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "mod.py").write_text("VALUE = 9\n", encoding="utf-8")
    invalidations = []
    monkeypatch.setattr(
        commit_gate,
        "_invalidate_advisory",
        lambda _ctx, **kwargs: invalidations.append(kwargs),
    )
    args = {
        "apply_patch": {
            "root": "system_repo",
            "patch": "*** Update File: mod.py\n-VALUE = 1\n+VALUE = 2\n",
        },
        "edit_batch": {
            "root": "system_repo",
            "edits": [{"path": "mod.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}],
        },
    }[tool]

    result = registry.execute(tool, args)

    assert result.startswith("✅"), result
    assert "Run commit_reviewed" in result
    assert "headless runner" not in result
    assert (system / "mod.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (project / "mod.py").read_text(encoding="utf-8") == "VALUE = 9\n"
    assert invalidations
    assert pathlib.Path(invalidations[-1]["mutation_root"]).resolve() == system.resolve()


@pytest.mark.parametrize("tool", ["apply_patch", "edit_batch"])
def test_repo_batch_tool_protected_name_depends_on_physical_target(
    tmp_path, monkeypatch, tool,
):
    from ouroboros import config

    registry, _ctx, system, project = _workspace_guard_registry(tmp_path, monkeypatch)
    (system / "BIBLE.md").write_text("SYSTEM = 1\n", encoding="utf-8")
    (project / "BIBLE.md").write_text("PROJECT = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    monkeypatch.setattr(edit_ops, "get_runtime_mode", lambda: "light")
    project_args = {
        "apply_patch": {
            "patch": "*** Update File: BIBLE.md\n-PROJECT = 1\n+PROJECT = 2\n",
        },
        "edit_batch": {
            "edits": [{"path": "BIBLE.md", "old_str": "PROJECT = 1", "new_str": "PROJECT = 2"}],
        },
    }[tool]
    system_args = {
        "apply_patch": {
            "root": "system_repo",
            "patch": "*** Update File: BIBLE.md\n-SYSTEM = 1\n+SYSTEM = 2\n",
        },
        "edit_batch": {
            "root": "system_repo",
            "edits": [{"path": "BIBLE.md", "old_str": "SYSTEM = 1", "new_str": "SYSTEM = 2"}],
        },
    }[tool]

    project_result = registry.execute(tool, project_args)
    system_result = registry.execute(tool, system_args)

    assert project_result.startswith("✅"), project_result
    assert "LIGHT_MODE_BLOCKED" in system_result or "CORE_PROTECTION_BLOCKED" in system_result
    assert (project / "BIBLE.md").read_text(encoding="utf-8") == "PROJECT = 2\n"
    assert (system / "BIBLE.md").read_text(encoding="utf-8") == "SYSTEM = 1\n"


@pytest.mark.parametrize("tool", ["write_file", "apply_patch", "edit_batch"])
def test_acting_subagent_without_workspace_cannot_touch_the_live_repo(tmp_path, tool):
    """An acting child with no isolated workspace must not reach the live repo.

    active_workspace falls back to the LIVE repo for such a child, which is why
    the fence exists; a new repo-lane write tool that misses it is a weaker lane,
    not a new capability.
    """
    from ouroboros.contracts.task_constraint import TaskConstraint

    reg, repo = _guard_registry(tmp_path)
    (repo / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    reg._is_acting_subagent = lambda: True
    reg._ctx.task_constraint = TaskConstraint(mode="acting")
    args = {
        "write_file": {"path": "mod.py", "content": "VALUE = 2\n"},
        "apply_patch": {"patch": "*** Update File: mod.py\n-VALUE = 1\n+VALUE = 2\n"},
        "edit_batch": {"edits": [{"path": "mod.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}]},
    }[tool]
    result = str(reg.execute(tool, args))
    assert "ACTING_NO_WORKSPACE_BLOCKED" in result, result[:200]
    assert (repo / "mod.py").read_text() == "VALUE = 1\n"


def test_acting_subagent_schema_narrows_root_for_every_repo_write_tool():
    from ouroboros.tools.registry import _ROOT_ARG_REPO_WRITE_TOOLS

    assert {"write_file", "edit_text", "apply_patch", "edit_batch"} == set(_ROOT_ARG_REPO_WRITE_TOOLS)


def test_patch_target_paths_come_from_the_real_parser():
    from ouroboros.tools.edit_ops import patch_target_paths

    assert patch_target_paths(
        "*** Begin Patch\n*** Update File: a.py\n-x\n+y\n*** Add File: b.py\n+z\n"
        "*** Delete File: c.py\n*** End Patch\n"
    ) == ["a.py", "b.py", "c.py"]
    # An unparseable patch yields no targets; the handler refuses it before any write.
    assert patch_target_paths("garbage without a header") == []


# ---------------------------------------------------------------------------
# content fidelity of the overwrite-verification rail
# ---------------------------------------------------------------------------

def test_unified_diff_reports_a_final_newline_change():
    # The rail exists so the agent can VERIFY an overwrite: claiming "no textual
    # changes" for a file whose bytes changed is the one unacceptable answer.
    out = edit_ops._unified_diff("f.txt", "value\n", "value")
    assert "No newline at end of file" in out
    assert out != "(no textual changes)"
    assert edit_ops._unified_diff("f.txt", "value", "value\n").startswith("\\ Newline added")
    assert edit_ops._unified_diff("f.txt", "same\n", "same\n") == "(no textual changes)"


def test_syntax_check_names_the_format_it_checked():
    # compile() raises a bare ValueError on a NUL byte; calling that "not valid
    # JSON" for a .py file sends the fix in the wrong direction.
    message = edit_ops._syntax_check("mod.py", "x = 1\x00\n")
    assert message and "JSON" not in message
    assert "Python" in message


# ---------------------------------------------------------------------------
# partial-write disclosure (validation is atomic; the write phase is not)
# ---------------------------------------------------------------------------

def test_partial_write_marks_advisory_stale_and_says_so(tmp_path, monkeypatch):
    """Files written before an I/O failure are real mutations.

    If the advisory snapshot stayed fresh, commit_reviewed would accept them
    against a pre-review taken before they existed.
    """
    from ouroboros.tools import commit_gate

    invalidated = []
    monkeypatch.setattr(
        commit_gate, "_invalidate_advisory",
        lambda ctx, **kw: invalidated.append(tuple(kw.get("changed_paths") or ())),
    )
    repo = tmp_path / "ws"
    repo.mkdir()
    (repo / "one.py").write_text("A = 1\n", encoding="utf-8")
    (repo / "two.py").write_text("B = 1\n", encoding="utf-8")
    ctx = _FakeCtx(repo)
    monkeypatch.setattr(edit_ops, "_resolve_edit_target", _fake_resolver(ctx))
    real_write = edit_ops.write_text

    def flaky(target, content):
        if target.name == "two.py":
            raise OSError("disk full")
        return real_write(target, content)

    monkeypatch.setattr(edit_ops, "write_text", flaky)
    result = edit_ops._apply_patch(
        ctx,
        "*** Update File: one.py\n-A = 1\n+A = 2\n*** Update File: two.py\n-B = 1\n+B = 2\n",
    )
    assert "EDIT_OPS_PARTIAL_WRITE_FAILED" in result and "PARTIALLY APPLIED" in result and "one.py" in result
    assert (repo / "one.py").read_text() == "A = 2\n"
    assert invalidated == [("one.py",)]


def test_edit_ops_refusals_are_policy_denials_not_execution_failures():
    """A counted/context refusal is the DESIGNED path, not a broken executor.

    Untyped, these fall through to the generic `error` status and degrade
    execution health with a false tool_failure headline — the exact regression
    v6.57.0 removed for the other write tools.
    """
    from ouroboros.loop_tool_execution import _extract_result_metadata
    from ouroboros.outcomes import _POLICY_DENIAL_STATUSES

    for text in (
        "⚠️ APPLY_PATCH_ERROR: hunk 1: context not found in m.py (searched from line 1).",
        "⚠️ EDIT_BATCH_ERROR: batch aborted, NOTHING was written (atomic).",
    ):
        status = _extract_result_metadata("t", text, False)["status"]
        assert status == "edit_ops_blocked"
        assert status in _POLICY_DENIAL_STATUSES


def test_one_file_under_two_spellings_is_one_target(tmp_path):
    """Two spellings of one file in a single call must not race each other.

    ``ctx.repo_path`` maps them to the same file, so keying the plan by the RAW
    spelling produced two buffers, two writes, and a last-write-wins that
    silently dropped the first edit while reporting both as applied.
    """
    reg, repo = _guard_registry(tmp_path)
    (repo / "f.txt").write_text("A = 1\nB = 1\n", encoding="utf-8")
    result = str(reg.execute("edit_batch", {"edits": [
        {"path": "f.txt", "old_str": "A = 1", "new_str": "A = 2"},
        {"path": "repo/f.txt", "old_str": "B = 1", "new_str": "B = 2"},
    ]}))
    assert result.startswith("✅"), result[:200]
    assert (repo / "f.txt").read_text() == "A = 2\nB = 2\n"
    assert "across 1 file" in result, result[:200]

    (repo / "g.txt").write_text("X = 1\nY = 1\n", encoding="utf-8")
    result = str(reg.execute("apply_patch", {"patch":
        "*** Update File: g.txt\n-X = 1\n+X = 2\n"
        f"*** Update File: {repo / 'g.txt'}\n-Y = 1\n+Y = 2\n"
    }))
    assert result.startswith("✅"), result[:200]
    assert (repo / "g.txt").read_text() == "X = 2\nY = 2\n"


def test_partial_write_is_an_execution_failure_not_a_policy_denial():
    """A validation refusal is harmless telemetry; a half-applied write is not."""
    from ouroboros.loop_tool_execution import _extract_result_metadata
    from ouroboros.outcomes import _POLICY_DENIAL_STATUSES

    partial = (
        "⚠️ EDIT_OPS_PARTIAL_WRITE_FAILED (APPLY_PATCH_ERROR): write failed for b.py: disk full\n"
        "PARTIALLY APPLIED — these files WERE written: a.py."
    )
    status = _extract_result_metadata("t", partial, False)["status"]
    assert status not in _POLICY_DENIAL_STATUSES
    assert status == "error"


@pytest.mark.parametrize("tool", ["write_file", "apply_patch", "edit_batch"])
def test_pro_mode_protected_edit_announces_itself(tmp_path, monkeypatch, tool):
    """A pro-mode protected write is ALLOWED; the notice is what keeps it visible.

    git._repo_write and _str_replace_editor both append it, so a repo-write tool
    that stays silent makes a protected edit look like an ordinary one.
    """
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    from ouroboros import config

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "pro")
    reg, repo = _guard_registry(tmp_path)
    (repo / "BIBLE.md").write_text("P1 honest\n", encoding="utf-8")
    args = {
        "write_file": {"path": "BIBLE.md", "content": "P1 honest v2\n"},
        "apply_patch": {"patch": "*** Update File: BIBLE.md\n-P1 honest\n+P1 honest v2\n"},
        "edit_batch": {"edits": [{"path": "BIBLE.md", "old_str": "P1 honest", "new_str": "P1 honest v2"}]},
    }[tool]
    result = str(reg.execute(tool, args))
    assert result.startswith("✅"), result[:200]
    assert "CORE_PATCH_NOTICE" in result, result[:300]


def test_managed_update_resolver_keeps_its_exemption(tmp_path, monkeypatch):
    """The assisted resolver edits whatever official file the merge conflicts on.

    git._repo_write carries this exemption; withholding it here would make these
    tools the one lane that cannot finish a conflict resolution.
    """
    from ouroboros.tools import registry as registry_mod
    from ouroboros.tools import registry_guards

    reg, repo = _guard_registry(tmp_path)
    (repo / "BIBLE.md").write_text("P1 honest\n", encoding="utf-8")
    monkeypatch.setattr(registry_guards, "_authorized_managed_update_resolver", lambda ctx: True)
    monkeypatch.setattr(registry_mod, "_authorized_managed_update_resolver", lambda ctx: True)
    result = str(reg.execute("apply_patch", {
        "patch": "*** Update File: BIBLE.md\n-P1 honest\n+P1 resolved\n",
    }))
    assert result.startswith("✅"), result[:200]
    assert (repo / "BIBLE.md").read_text() == "P1 resolved\n"
