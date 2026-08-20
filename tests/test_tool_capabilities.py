"""Tests for tool capability SSOT and no-drift invariants.

Verifies:
- tool_capabilities.py is the single source of truth
- tool_policy.py imports from capabilities (no local copy)
- loop_tool_execution.py imports from capabilities (no local copy)
- search_code is classified correctly
- run_shell list-cmd happy path (string-cmd cascade lives in test_shell_run_shell.py)
- search_code tool works
"""
import inspect
import json
import os
import base64
import pathlib
import re

import pytest
import sys
import tempfile


# ---------------------------------------------------------------------------
# SSOT drift tests
# ---------------------------------------------------------------------------


def test_tool_policy_defines_no_local_tool_sets():
    """tool_policy.py must not define its own tool-name sets (SSOT lives in tool_capabilities)."""
    import ouroboros.tool_policy as tp
    source = inspect.getsource(tp)
    assert not re.search(r"^(CORE_TOOL_NAMES|META_TOOL_NAMES)\s*[:=]", source, re.MULTILINE)
    assert "frozenset({" not in source


def test_loop_execution_imports_from_capabilities():
    """loop_tool_execution.py must import sets from tool_capabilities."""
    import ouroboros.loop_tool_execution as lte
    source = inspect.getsource(lte)
    assert "from ouroboros.tool_capabilities import" in source
    # Must NOT have local frozenset definitions for these sets
    for name in ("READ_ONLY_PARALLEL_TOOLS", "STATEFUL_BROWSER_TOOLS",
                 "_UNTRUNCATED_TOOL_RESULTS", "_UNTRUNCATED_REPO_READ_PATHS"):
        # Check there's no local `X = frozenset({` pattern
        pattern = rf'^{re.escape(name)}\s*[:=]\s*frozenset'
        assert not re.search(pattern, source, re.MULTILINE), (
            f"{name} is locally defined in loop_tool_execution.py — should import from tool_capabilities"
        )


def test_capabilities_sets_are_frozensets():
    """All exported sets must be frozensets (immutable)."""
    from ouroboros.tool_capabilities import (
        CORE_TOOL_NAMES, META_TOOL_NAMES, READ_ONLY_PARALLEL_TOOLS,
        PARALLEL_SAFE_ENQUEUE_TOOLS,
        STATEFUL_BROWSER_TOOLS, UNTRUNCATED_TOOL_RESULTS,
        UNTRUNCATED_REPO_READ_PATHS,
    )
    for name, obj in [
        ("CORE_TOOL_NAMES", CORE_TOOL_NAMES),
        ("META_TOOL_NAMES", META_TOOL_NAMES),
        ("READ_ONLY_PARALLEL_TOOLS", READ_ONLY_PARALLEL_TOOLS),
        ("PARALLEL_SAFE_ENQUEUE_TOOLS", PARALLEL_SAFE_ENQUEUE_TOOLS),
        ("STATEFUL_BROWSER_TOOLS", STATEFUL_BROWSER_TOOLS),
        ("UNTRUNCATED_TOOL_RESULTS", UNTRUNCATED_TOOL_RESULTS),
        ("UNTRUNCATED_REPO_READ_PATHS", UNTRUNCATED_REPO_READ_PATHS),
    ]:
        assert isinstance(obj, frozenset), f"{name} must be a frozenset"


def test_child_profiles_remain_explicit_narrowing_sets():
    """Top-level surface parity must not widen delegated-child profiles."""
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert ACTING_SUBAGENT_TOOL_NAMES
    assert "commit_reviewed" not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "commit_reviewed" not in ACTING_SUBAGENT_TOOL_NAMES
    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES != CORE_TOOL_NAMES
    assert ACTING_SUBAGENT_TOOL_NAMES != CORE_TOOL_NAMES


def test_top_level_workspace_focus_has_tool_and_schema_parity(tmp_path, monkeypatch):
    """Ordinary top-level presets differ in default target, not built-in names."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.tools.search as search

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(search, "_available_web_search_backends", lambda: ["ddgs"])
    system_repo = tmp_path / "system"
    project = tmp_path / "project"
    external = tmp_path / "external"
    data = tmp_path / "data"
    for path in (system_repo, project, external, data):
        path.mkdir()

    contexts = {
        "plain": ToolContext(repo_dir=system_repo, drive_root=data, task_id="plain"),
        "workspace": ToolContext(
            repo_dir=system_repo, drive_root=data, task_id="workspace",
            workspace_root=project, workspace_mode="project",
        ),
        "external_workspace": ToolContext(
            repo_dir=system_repo, drive_root=data, task_id="external",
            workspace_root=external, workspace_mode="external",
        ),
    }
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    snapshots = {}
    for label, ctx in contexts.items():
        registry.set_context(ctx)
        names = frozenset(registry.available_tools())
        schemas = frozenset(
            name for name in registry._entries
            if registry.get_schema_by_name(name) is not None
        )
        snapshots[label] = (names, schemas)

    assert snapshots["plain"] == snapshots["workspace"] == snapshots["external_workspace"]
    names, schemas = snapshots["workspace"]
    assert names == schemas
    assert {
        "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
        "switch_model", "send_photo", "send_video", "send_file",
        "commit_reviewed", "promote_chat_to_task", "vcs_restore",
    } <= names

    # Successor of the retired _WORKSPACE_ALLOWED_TOOLS subset invariant: the
    # workspace surface is now the full registry, so every child-profile tool
    # must be a registered, workspace-visible name. Guards the 2026-08-10 saga
    # shape (a profile tool invisible exactly where children are spawned) —
    # delegate_answer riding only the child profiles would silently degrade
    # workspace-scoped nannies to the engine's benign decline.
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES <= names, (
        f"read-only child tools missing from the workspace tool surface: "
        f"{sorted(LOCAL_READONLY_SUBAGENT_TOOL_NAMES - names)}"
    )
    assert ACTING_SUBAGENT_TOOL_NAMES <= names, (
        f"acting child tools missing from the workspace tool surface: "
        f"{sorted(ACTING_SUBAGENT_TOOL_NAMES - names)}"
    )


@pytest.mark.parametrize("workspace_mode", ["", "project", "external"])
def test_top_level_contract_and_resource_filters_narrow_independently(
    tmp_path, monkeypatch, workspace_mode,
):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = tmp_path / f"system-{workspace_mode or 'plain'}"
    workspace = tmp_path / f"workspace-{workspace_mode or 'plain'}"
    data = tmp_path / f"data-{workspace_mode or 'plain'}"
    for path in (system, workspace, data):
        path.mkdir()
    contract = {
        "disabled_tools": ["commit_reviewed"],
        "allowed_resources": {"web": False, "network": False},
    }
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        task_id=f"filter-{workspace_mode or 'plain'}",
        workspace_root=workspace if workspace_mode else None,
        workspace_mode=workspace_mode,
        task_contract=contract,
        task_metadata={"task_contract": contract},
    )
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))

    assert registry.get_schema_by_name("commit_reviewed") is None
    assert "disabled by this task's contract" in registry.policy_hidden_reason("commit_reviewed")
    assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("youtube_transcript", {"url": "https://example.test"})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("vcs_pull_ff", {})


def test_frozen_registry_includes_pr_integration_tools(tmp_path, monkeypatch):
    import sys
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    names = set(registry.available_tools())
    assert {
        "fetch_pr_ref",
        "create_integration_branch",
        "cherry_pick_pr_commits",
        "stage_adaptations",
        "stage_pr_merge",
    } <= names


def test_loop_execution_parallel_tools_from_capabilities():
    """READ_ONLY_PARALLEL_TOOLS in loop_tool_execution is from capabilities."""
    from ouroboros.loop_tool_execution import READ_ONLY_PARALLEL_TOOLS as loop_set
    from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS as cap_set
    assert loop_set is cap_set


# ---------------------------------------------------------------------------
# search_code classification tests
# ---------------------------------------------------------------------------


def test_search_code_in_core_tools():
    """search_code must be in CORE_TOOL_NAMES."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "search_code" in CORE_TOOL_NAMES


def test_search_code_is_parallel_safe():
    """search_code must be in READ_ONLY_PARALLEL_TOOLS."""
    from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS
    assert "search_code" in READ_ONLY_PARALLEL_TOOLS


def test_search_code_has_result_limit():
    """search_code must have an explicit result size limit."""
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS
    assert "search_code" in TOOL_RESULT_LIMITS
    from ouroboros.tool_capabilities import UNTRUNCATED_TOOL_RESULTS
    assert "plan_task" in UNTRUNCATED_TOOL_RESULTS
    # Child-handoff tools stay transport-uncapped: wait_task/get_task_result are
    # FULL by contract, and wait_tasks' compact projection must not additionally
    # be char-capped (child_result_sha256 pins the exact result text seen).
    for _handoff_tool in ("wait_task", "wait_tasks", "get_task_result"):
        assert _handoff_tool in UNTRUNCATED_TOOL_RESULTS
    from ouroboros.tool_capabilities import FOREGROUND_MUTATIVE_TOOLS
    # Publication can create a remote branch/commit/PR. Its outer timeout must
    # not return while that foreground mutator is still running.
    assert FOREGROUND_MUTATIVE_TOOLS == frozenset({"submit_skill_to_hub"})


def test_extract_video_frames_visible_where_media_siblings_are_visible():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    for tool_set in (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES, ACTING_SUBAGENT_TOOL_NAMES):
        assert {"ocr_pdf", "youtube_transcript", "extract_video_frames"} <= tool_set


def test_extract_video_frames_visible_to_workspace_tasks(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    repo.mkdir()
    workspace.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, workspace_root=workspace, workspace_mode="external"))

    assert registry.get_schema_by_name("extract_video_frames") is not None


# ---------------------------------------------------------------------------
# search_code tool behavior tests
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext
    from unittest.mock import MagicMock
    ctx = MagicMock(spec=ToolContext)
    ctx.repo_dir = tmp_path
    ctx.repo_path = lambda p: tmp_path / p
    return ctx


def _populate_repo(tmp_path):
    """Create a mini repo structure for search tests."""
    (tmp_path / "foo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (tmp_path / "bar.py").write_text("import os\ndef hello_bar():\n    pass\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "baz.py").write_text("class MyClass:\n    hello = True\n", encoding="utf-8")
    # Binary-like file (should be skipped)
    (tmp_path / "data.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    # Cache dir (should be skipped)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "foo.cpython-310.pyc").write_bytes(b'\x00' * 50)


def test_code_search_literal(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "hello")
    assert "foo.py:1:" in result
    assert "bar.py:2:" in result
    assert "sub/baz.py:2:" in result


def test_code_search_regex(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, r"def \w+\(\)", regex=True)
    assert "foo.py:1:" in result
    assert "bar.py:2:" in result


def test_code_search_scoped_path(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "hello", path="sub")
    assert "sub/baz.py" in result
    assert "foo.py" not in result


def test_code_search_include_filter(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    (tmp_path / "readme.md").write_text("hello from markdown\n", encoding="utf-8")
    result = _code_search(ctx, "hello", include="*.md")
    assert "readme.md" in result
    assert "foo.py" not in result


def test_code_search_no_matches(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "zzz_nonexistent_zzz")
    assert "No matches found" in result


def test_code_search_skips_binaries(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "PNG")
    # .png file should be skipped even though it contains "PNG" bytes
    assert "data.png" not in result


def test_code_search_skips_cache_dirs(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "foo")
    assert "__pycache__" not in result


def test_code_search_max_results(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    # Create many matching lines
    lines = "\n".join(f"match_line_{i}" for i in range(50))
    (tmp_path / "many.py").write_text(lines, encoding="utf-8")
    result = _code_search(ctx, "match_line", max_results=10)
    assert "truncated at 10" in result


def test_code_search_empty_query(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    result = _code_search(ctx, "")
    assert "SEARCH_ERROR" in result


def test_code_search_invalid_regex(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    result = _code_search(ctx, "[invalid", regex=True)
    assert "SEARCH_ERROR" in result


# ---------------------------------------------------------------------------
# run_shell string contract
# ---------------------------------------------------------------------------
#
# String-cmd recovery (shlex.split for plain strings, json.loads for JSON
# arrays, ast.literal_eval for Python literals) is covered by
# tests/test_shell_run_shell.py::TestShellArgContract.  This file keeps only
# the list-cmd happy-path sibling so the capability sets module owns the
# round-1 tool surface assertions, not the string-cascade contract itself.


def test_run_shell_list_cmd_works(tmp_path):
    """run_shell with a list cmd should work normally."""
    from ouroboros.tools.shell import _run_shell
    from unittest.mock import MagicMock
    from ouroboros.tools.registry import ToolContext
    ctx = MagicMock(spec=ToolContext)
    ctx.repo_dir = tmp_path
    ctx.drive_logs.return_value = tmp_path
    result = _run_shell(ctx, ["echo", "hello"])
    assert "hello" in result


# ---------------------------------------------------------------------------
# Initial tool visibility
# ---------------------------------------------------------------------------


def test_search_code_in_initial_schemas():
    """search_code must appear in initial tool schemas."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import initial_tool_schemas
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert "search_code" in names


def test_search_code_registered():
    """search_code must be registered in the tool registry."""
    from ouroboros.tools.registry import ToolRegistry
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    available = {t["function"]["name"] for t in registry.schemas()}
    assert "search_code" in available


def test_search_code_ripgrep_path_filters_protected_files(tmp_path, monkeypatch):
    """The rg fast path must receive only files that passed Ouroboros gates."""
    import json
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (repo / "safe.py").write_text("needle public\n", encoding="utf-8")
    (repo / "auth").mkdir()
    (repo / "auth" / "secret.py").write_text("needle secret\n", encoding="utf-8")
    seen = tmp_path / "seen.json"
    fake_rg_py = tmp_path / "fake_rg.py"
    fake_rg_py.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "needle=args[args.index('--')+1]\n"
        "paths=args[args.index('--')+2:]\n"
        f"pathlib.Path({str(seen)!r}).write_text(json.dumps(paths))\n"
        "for p in paths:\n"
        "    text=pathlib.Path(p).read_text(errors='replace')\n"
        "    if needle in text:\n"
        "        print(json.dumps({'type':'match','data':{'path':{'text':p},'line_number':1,'lines':{'text':text.splitlines()[0]+'\\\\n'}}}))\n",
        encoding="utf-8",
    )
    fake_rg_py.chmod(0o755)
    if os.name == "nt":
        fake_rg = tmp_path / "fake_rg.cmd"
        fake_rg.write_text(f"@echo off\r\n\"{sys.executable}\" \"{fake_rg_py}\" %*\r\n", encoding="utf-8")
    else:
        fake_rg = fake_rg_py
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: str(fake_rg))

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_constraint = TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE)
    result = registry.execute("search_code", {"query": "needle"})
    assert "safe.py" in result
    assert "auth/secret.py" not in result
    assert all("auth/secret.py" not in path for path in json.loads(seen.read_text(encoding="utf-8")))


def test_search_code_ripgrep_fallback_when_unavailable(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (repo / "safe.py").write_text("needle public\n", encoding="utf-8")
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    result = registry.execute("search_code", {"query": "needle"})
    assert "safe.py" in result
    assert "files searched" in result


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_search_code_does_not_follow_symlink_outside_root(tmp_path, monkeypatch):
    """A symlink inside the workspace that points OUTSIDE the resource root must not
    be read by search_code (rg path resolved-containment + is_search_skippable)."""
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("needle CONFIDENTIAL_OUTSIDE\n", encoding="utf-8")
    (repo / "in_root.txt").write_text("needle in_root_ok\n", encoding="utf-8")
    (repo / "escape.txt").symlink_to(outside)  # symlink whose target escapes the root

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    # rg path
    result = registry.execute("search_code", {"query": "needle"})
    assert "in_root_ok" in result
    assert "CONFIDENTIAL_OUTSIDE" not in result
    # python fallback path (rg unavailable) must also refuse the symlink
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")
    fallback = registry.execute("search_code", {"query": "needle"})
    assert "CONFIDENTIAL_OUTSIDE" not in fallback


# ---------------------------------------------------------------------------
# schedule_subagent core classification tests
# ---------------------------------------------------------------------------


def test_schedule_subagent_in_core():
    """schedule_subagent is core for first-class parallel delegation."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "schedule_subagent" in CORE_TOOL_NAMES


def test_wait_task_in_core():
    """wait_task/wait_tasks are core so delegated work can be joined."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "wait_task" in CORE_TOOL_NAMES
    assert "wait_tasks" in CORE_TOOL_NAMES


def test_get_task_result_in_core():
    """get_task_result is core so child handoffs can be read."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "get_task_result" in CORE_TOOL_NAMES


def test_schedule_subagent_available_in_registry():
    """schedule_subagent must still be registered."""
    from ouroboros.tools.registry import ToolRegistry
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    all_names = {t["function"]["name"] for t in registry.schemas()}
    assert "schedule_subagent" in all_names, (
        "schedule_subagent must be discoverable via list_available_tools / enable_tools"
    )


def test_schedule_subagent_in_initial_schemas():
    """schedule_subagent appears in parent initial schemas as a core tool."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import initial_tool_schemas
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert "schedule_subagent" in names
    schedule_schema = next(s for s in initial_tool_schemas(registry) if s["function"]["name"] == "schedule_subagent")
    props = schedule_schema["function"]["parameters"]["properties"]
    assert "required_capabilities" in props
    assert "shell" in props["required_capabilities"]["items"]["enum"]


def test_schedule_subagent_required_capabilities_fail_fast_for_readonly(tmp_path):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    result = _schedule_task(
        ctx,
        objective="Need git diff",
        expected_output="diff summary",
        required_capabilities=["shell", "vcs"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" in result


def test_schedule_subagent_required_delegate_capability_is_satisfied_for_readonly(tmp_path):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    events = []
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    ctx.pending_events = events
    ctx.task_id = "parent1"
    result = _schedule_task(
        ctx,
        objective="Delegate deeper readonly work",
        expected_output="child id",
        required_capabilities=["delegate"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" not in result
    assert events and events[0]["type"] == "schedule_subagent"
    assert events[0]["required_capabilities"] == ["delegate"]


def test_schedule_subagent_required_vcs_capability_is_satisfied_for_readonly(tmp_path):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    events = []
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    ctx.pending_events = events
    ctx.task_id = "parent1"
    result = _schedule_task(
        ctx,
        objective="Inspect git status in readonly child",
        expected_output="status summary",
        required_capabilities=["vcs"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" not in result
    assert events and events[0]["required_capabilities"] == ["vcs"]


def test_local_readonly_subagent_initial_schemas_are_allowlisted(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    from ouroboros.tool_policy import initial_tool_schemas, list_non_core_tools
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES <= names
    assert "enable_tools" not in names
    assert "schedule_subagent" in names
    assert "write_file" not in names
    assert "run_command" not in names
    assert "browse_page" in names
    assert "browser_action" in names
    schemas = {s["function"]["name"]: s["function"] for s in initial_tool_schemas(registry)}
    for tool_name in ("read_file", "list_files", "search_code"):
        root_enum = schemas[tool_name]["parameters"]["properties"]["root"]["enum"]
        assert "user_files" not in root_enum
    assert set(schemas["search_code"]["parameters"]["properties"]["root"]["enum"]) == {"active_workspace", "system_repo", "skill_payload"}
    action_schema = schemas["browser_action"]["parameters"]["properties"]["action"]
    assert "evaluate" not in action_schema["enum"]
    assert "send_photo" not in schemas["browse_page"]["description"]
    assert "analyze_screenshot" in schemas["browse_page"]["description"]
    assert schemas["browse_page"]["parameters"]["properties"]["engine"]["enum"] == ["chromium", "webkit"]
    assert "device" in schemas["browse_page"]["parameters"]["properties"]
    assert list_non_core_tools(registry) == []


def test_local_readonly_subagent_execute_blocks_forbidden_tools(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.mcp_client as mcp_client

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert registry.get_schema_by_name("write_file") is None
    assert registry.get_schema_by_name("enable_tools") is None
    assert registry.get_schema_by_name("schedule_subagent") is not None
    # switch_model changes COGNITIVE POWER, not authority: a child that started cheap and
    # found the work harder raises its own strength, and nothing about its sandbox moves.
    # It was on the blocked list until v6.87.7 purely because power and authority were
    # conflated; a read-only child stays read-only at any model.
    assert registry.get_schema_by_name("switch_model") is not None
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" not in registry.execute("switch_model", {})
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MCP touched")))
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" not in registry.execute("list_files", {"path": "."})
    assert registry.get_schema_by_name("vcs_status") is not None
    assert "TOOL_ACCESS_BLOCKED" not in registry.execute("vcs_status", {"root": "system_repo"})
    blocked_tools = [
        "write_file",
        "edit_text",
        "knowledge_write",
        "update_scratchpad",
        "update_identity",
        "commit_reviewed",
        "advisory_review",
        "task_acceptance_review",
        "skill_review",
        "request_restart",
        "enable_tools",
        "run_command",
        "skill_exec",
        "list_skills",
    ]
    for name in blocked_tools:
        assert registry.get_schema_by_name(name) is None
        assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute(name, {})


def test_workspace_parent_keeps_the_ordinary_top_level_control_surface(tmp_path, monkeypatch):
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.mcp_client as mcp_client

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))

    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda *args, **kwargs: None)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: type("_M", (), {"list_tools_for_registry": lambda self: []})())
    names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}

    assert "plan_task" in names
    assert "task_acceptance_review" in names
    assert "commit_reviewed" in names
    assert "request_restart" in names

    registry.override_handler("task_acceptance_review", lambda ctx=None, **_kwargs: "review-ok")
    registry.override_handler("commit_reviewed", lambda ctx=None, **_kwargs: "commit-ok")
    assert registry.execute("task_acceptance_review", {}) == "review-ok"
    assert registry.execute("commit_reviewed", {"commit_message": "system target"}) == "commit-ok"


def test_workspace_focus_does_not_turn_top_level_cancel_into_child_only(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import join_ledger
    from ouroboros.tools.registry import ToolContext

    system, workspace, data = tmp_path / "system", tmp_path / "workspace", tmp_path / "data"
    for path in (system, workspace, data):
        path.mkdir()
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="parent",
    )
    monkeypatch.setattr(join_ledger, "_is_own_child", lambda *_a, **_k: False)
    # NB: no ``write_task_result`` patch — the cancel tool no longer writes a
    # status latch at all (phase A: it records a durable cancel INTENT), and the
    # symbol is not imported here any more, so patching it raised AttributeError.
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event", lambda *_a, **_k: "live")

    assert join_ledger._cancel_task(ctx, "foreign-task").startswith("Cancel requested")

    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    assert "may only cancel its own children" in join_ledger._cancel_task(ctx, "foreign-task")


def test_local_readonly_subagent_allows_enabled_extension_tool(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._shared import clean_extension_runtime_state
    from tests.test_extension_loader import _mark_isolated_deps_installed, _prepare_extension

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    plugin = (
        "def _lookup(ctx, query=''):\n"
        "    return 'external-ok:' + query\n"
        "def register(api):\n"
        "    api.register_tool('lookup', _lookup, description='External lookup', "
        "schema={'type': 'object', 'properties': {'query': {'type': 'string'}}}, timeout_sec=5)\n"
    )
    loaded, skills_repo, parent_drive = _prepare_extension(
        tmp_path,
        "research",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(parent_drive, loaded)
    child_drive = tmp_path / "child-drive"
    child_drive.mkdir()
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=parent_drive)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("research", "lookup")
    assert extension_loader.is_extension_live("research", parent_drive, repo_path=str(skills_repo))
    assert not extension_loader.is_extension_live("research", child_drive, repo_path=str(skills_repo))
    assert extension_loader.get_tool(tool_name)["out_of_process"] is True
    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    registry = ToolRegistry(repo_dir=repo_dir, drive_root=child_drive)
    try:
        registry.set_context(
            ToolContext(
                repo_dir=repo_dir,
                drive_root=child_drive,
                task_metadata={"budget_drive_root": str(parent_drive)},
                task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
            )
        )
        assert registry.get_schema_by_name(tool_name) is not None
        assert "external-ok:budget-root" in registry.execute(tool_name, {"query": "budget-root"})
    finally:
        clean_extension_runtime_state()


def test_allowed_resources_block_web_and_external_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from ouroboros import extension_loader
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    task_contract = build_task_contract({
        "id": "task-resources",
        "allowed_resources": {"web": "false", "network": "false"},
    })
    tool_name = extension_loader.extension_surface_name("research", "lookup")
    with extension_loader._lock:
        extension_loader._tools[tool_name] = {
            "name": tool_name,
            "handler": lambda ctx, **kwargs: "external-ok",
            "description": "External lookup",
            "schema": {"type": "object", "properties": {}},
            "timeout_sec": 5,
            "skill": "research",
        }
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    try:
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path / "repo",
                drive_root=tmp_path / "data",
                task_contract=task_contract,
                task_metadata={"task_contract": task_contract},
            )
        )
        assert task_contract["allowed_resources"] == {"web": False, "network": False}
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("web_search", {"query": "x"})
        # VLM tools are first-class vision tools, not web egress. Benchmark isolation
        # withholds them by name via disabled_tools instead of relying on web=false.
        assert "RESOURCE_CONSTRAINT_BLOCKED" not in registry.execute("vlm_query", {"prompt": "x"})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute(
            "vlm_query", {"prompt": "x", "image_url": "https://example.com/a.png"}
        )
        assert registry.get_schema_by_name(tool_name) is None
        assert tool_name not in {schema["function"]["name"] for schema in registry.schemas()}
        assert any(item.get("surface") == "extensions" and item.get("reason") == "resource_blocked" for item in registry.capability_omissions())
        blocked = registry.execute(tool_name, {})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked
        assert "network=false" in blocked

        alias_contract = build_task_contract({
            "id": "task-resource-aliases",
            "allowed_resources": {"allow_network": "false"},
        })
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path / "repo",
                drive_root=tmp_path / "data",
                task_contract=alias_contract,
                task_metadata={"task_contract": alias_contract},
            )
        )
        assert alias_contract["allowed_resources"] == {"allow_network": False}
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("web_search", {"query": "x"})
    finally:
        with extension_loader._lock:
            extension_loader._tools.pop(tool_name, None)


def test_protected_black_box_artifact_policy_blocks_introspection(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    if os.name == "nt":
        protected = repo / "reference.cmd"
        generated = repo / "generated.cmd"
        direct_cmd = ["cmd.exe", "/c", str(protected)]
        protected.write_text("@echo reference\r\n", encoding="utf-8")
        generated.write_text("@echo generated\r\n", encoding="utf-8")
    else:
        protected = repo / "reference.sh"
        generated = repo / "generated.sh"
        direct_cmd = [str(protected)]
        protected.write_text("#!/bin/sh\nprintf 'reference\\n'\n", encoding="utf-8")
        generated.write_text("#!/bin/sh\nprintf 'generated\\n'\n", encoding="utf-8")
    protected_dir = repo / "protected_dir"
    protected_dir.mkdir()
    (protected_dir / "secret.txt").write_text("secret\n", encoding="utf-8")
    protected.chmod(0o755)
    generated.chmod(0o755)
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": [str(protected)],
                    "allow": ["execute"],
                    "deny": ["read_bytes", "copy", "hash", "static_introspection", "dynamic_trace", "debug"],
                },
                {
                    "id": "reference-dir",
                    "role": "black_box_reference",
                    "paths": [str(protected_dir)],
                    "allow": ["execute"],
                }
            ]
        }
    })
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_contract=task_contract,
        task_metadata={"task_contract": task_contract},
    ))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    direct = registry.execute("run_command", {"cmd": direct_cmd})
    assert "RESOURCE_POLICY_BLOCKED" not in direct
    assert "reference" in direct
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("read_file", {"path": protected.name})
    protected_content = protected.read_text(encoding="utf-8")
    write_attempt = registry.execute("write_file", {"path": protected.name, "content": "tamper\n"})
    assert "RESOURCE_POLICY_BLOCKED" in write_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    edit_attempt = registry.execute("edit_text", {"path": protected.name, "old_str": "reference", "new_str": "tamper"})
    assert "RESOURCE_POLICY_BLOCKED" in edit_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    shell_write_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", f"printf tamper > {protected.name}"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in shell_write_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    shell_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["rm", protected.name], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in shell_delete_attempt
    assert protected.exists()
    recursive_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["rm", "-rf", "."], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in recursive_delete_attempt
    assert protected.exists()
    glob_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", "rm -rf *"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in glob_delete_attempt
    assert protected.exists()
    glob_read_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", "cat *"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in glob_read_attempt
    find_exec_read = registry.execute(
        "run_command",
        {"cmd": ["find", ".", "-type", "f", "-exec", "cat", "{}", "+"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in find_exec_read
    find_delete = registry.execute(
        "run_command",
        {"cmd": ["find", ".", "-delete"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in find_delete
    assert protected.exists()
    pathless_find_exec_read = registry.execute(
        "run_command",
        {"cmd": ["find", "-type", "f", "-exec", "cat", "{}", "+"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in pathless_find_exec_read
    pathless_find_delete = registry.execute(
        "run_command",
        {"cmd": ["find", "-delete"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in pathless_find_delete
    assert protected.exists()
    safe_interpreter = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                "print(1)",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" not in safe_interpreter
    assert "1" in safe_interpreter
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("list_files", {"path": protected_dir.name})
    interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                f"from pathlib import Path; print(Path(r'{protected}').read_bytes())",
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in interpreter_read
    relative_interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                f"from pathlib import Path; print(Path({protected.name!r}).read_bytes())",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in relative_interpreter_read
    versioned_interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3.12",
                "-c",
                f"from pathlib import Path; print(Path({protected.name!r}).read_bytes())",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in versioned_interpreter_read
    constructed_path_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"print((Path(r'{protected.parent}') / ({protected.stem!r} + {protected.suffix!r})).read_bytes())"
                ),
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in constructed_path_read
    backslash_parent = str(protected.parent).replace("/", "\\")
    backslash_constructed_path_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"print((Path({backslash_parent!r}) / ({protected.stem!r} + {protected.suffix!r})).read_bytes())"
                ),
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in backslash_constructed_path_read
    env_assignment_read = registry.execute(
        "run_command",
        {
            "cmd": [
                f"REF={protected}",
                sys.executable,
                "-c",
                "import os; print(open(os.environ['REF'], 'rb').read())",
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in env_assignment_read
    shell_script_read = registry.execute("run_command", {"cmd": ["sh", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in shell_script_read
    for cmd in (
        ["cmd.exe", "/c", "type", protected.name],
        ["cmd.exe", "/c", "copy", protected.name, str(repo / "copy.cmd")],
        ["cmd.exe", "/c", "xcopy", protected.name, str(repo / "copy-dir")],
        ["powershell.exe", "-Command", "Get-Content", protected.name],
        ["powershell.exe", "-Command", "Select-String", "reference", protected.name],
        ["powershell.exe", "-Command", "Copy-Item", protected.name, str(repo / "copy.ps1")],
        ["pwsh", "-Command", "Get-FileHash", protected.name],
        ["cmd.exe", "/c", "certutil", "-hashfile", protected.name],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    encoded_read = base64.b64encode(f"Get-Content {protected.name}".encode("utf-16le")).decode("ascii")
    for cmd in (
        ["powershell.exe", "-EncodedCommand", encoded_read],
        ["pwsh", "-enc", encoded_read],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    search_direct = registry.execute("search_code", {"query": "reference", "path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" in search_direct
    search_protected_dir = registry.execute("search_code", {"query": "secret", "path": protected_dir.name})
    assert "RESOURCE_POLICY_BLOCKED" in search_protected_dir
    query_protected = registry.execute("query_code", {"op": "structural", "query": "reference", "path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" not in query_protected
    assert protected.name not in query_protected
    for cache in (data / "state" / "code_intel").glob("*/inventory.json"):
        assert protected.name not in cache.read_text(encoding="utf-8")
    grep_read = registry.execute("run_command", {"cmd": ["grep", "reference", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in grep_read
    grep_recursive = registry.execute("run_command", {"cmd": ["grep", "-R", "reference", "."], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in grep_recursive
    rg_read = registry.execute("run_command", {"cmd": ["rg", "reference", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in rg_read
    rg_recursive = registry.execute("run_command", {"cmd": ["rg", "reference", "."], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in rg_recursive
    copy_recursive = registry.execute("run_command", {"cmd": ["cp", "-R", ".", str(repo / "copy")], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in copy_recursive
    for cmd in (
        ["git", "diff", "--", protected.name],
        ["git", "diff"],
        ["git", "show", f"HEAD:{protected.name}"],
        ["git", "show", "HEAD"],
        ["git", "grep", "reference", "--", protected.name],
        ["git", "grep", "reference"],
        ["git", "cat-file", "-p", f"HEAD:{protected.name}"],
        ["git", "log", "-p", "--", protected.name],
        ["git", "log", "-p"],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("vcs_diff", {"path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("vcs_diff", {})
    import ouroboros.code_intelligence as code_intelligence

    original_file_fact = code_intelligence._file_fact

    def guarded_file_fact(repo_root, path):
        assert pathlib.Path(path).resolve(strict=False) != protected.resolve(strict=False)
        return original_file_fact(repo_root, path)

    monkeypatch.setattr(code_intelligence, "_file_fact", guarded_file_fact)
    digest = registry.execute("query_code", {"op": "digest"})
    assert protected.name not in digest
    assert generated.name in digest
    run_output_export = registry.execute("run_command", {"cmd": direct_cmd, "outputs": [protected.name], "cwd": str(repo)})
    assert "ARTIFACT_OUTPUT_ERROR" in run_output_export
    assert "RESOURCE_POLICY_BLOCKED" in run_output_export
    script_output_export = registry.execute(
        "run_script",
        {"interpreter": "python3", "script": "print('ok')", "outputs": [protected.name], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in script_output_export
    service_cmd = ["cmd.exe", "/c", "ping", "127.0.0.1", "-n", "30"] if os.name == "nt" else ["sleep", "30"]
    service_start = registry.execute(
        "start_service",
        {
            "name": "protected-output",
            "cmd": service_cmd,
            "cwd": str(repo),
            "outputs": [protected.name],
        },
    )
    assert "protected-output" in service_start
    service_stop = registry.execute("stop_service", {"name": "protected-output"})
    assert "ARTIFACT_OUTPUT_ERROR" in service_stop
    assert "RESOURCE_POLICY_BLOCKED" in service_stop
    for cmd in (
        ["strings", str(protected)],
        ["objdump", "-d", str(protected)],
        ["cat", str(protected)],
        ["sha256sum", str(protected)],
        ["strace", str(protected)],
        ["gdb", str(protected)],
        ["lldb", str(protected)],
        ["cp", str(protected), str(repo / "copy.sh")],
        ["dd", f"if={protected}", f"of={repo / 'copy2.sh'}"],
        ["tar", "-czf", str(repo / "out.tgz"), protected.name],
        ["tar", "-czf", str(repo / "tree.tgz"), "."],
        ["zip", str(repo / "out.zip"), protected.name],
        ["rsync", protected.name, str(repo / "copy.sh")],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd

    generated_result = registry.execute("run_command", {"cmd": ["strings", str(generated)]})
    assert "RESOURCE_POLICY_BLOCKED" not in generated_result


def test_protected_black_box_recursive_policy_maps_executor_backend_paths(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True, exist_ok=True)
    protected = workspace / "executable"
    protected.write_text("reference bytes\n", encoding="utf-8")
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": ["/workspace/executable"],
                    "allow": ["execute"],
                    "deny": ["read_bytes", "copy", "hash", "static_introspection", "dynamic_trace", "debug"],
                }
            ]
        }
    })
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=system_repo,
            drive_root=data,
            workspace_root=workspace,
            workspace_mode="external",
            task_contract=task_contract,
            task_metadata={"task_contract": task_contract},
            executor_ref={
                "type": "docker_exec",
                "id": "pb-container",
                "container_name": "pb-container",
                "network": "none",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
            },
        )
    )
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    grep_recursive = registry.execute("run_command", {"cmd": ["grep", "-R", "reference", "."], "cwd": str(workspace)})
    copy_recursive = registry.execute("run_command", {"cmd": ["cp", "-R", ".", str(workspace / "copy")], "cwd": str(workspace)})
    import ouroboros.code_intelligence as code_intelligence

    original_file_fact = code_intelligence._file_fact

    def guarded_file_fact(repo_root, path):
        assert pathlib.Path(path).resolve(strict=False) != protected.resolve(strict=False)
        return original_file_fact(repo_root, path)

    monkeypatch.setattr(code_intelligence, "_file_fact", guarded_file_fact)
    digest = registry.execute("query_code", {"op": "digest"})

    assert "RESOURCE_POLICY_BLOCKED" in grep_recursive
    assert "RESOURCE_POLICY_BLOCKED" in copy_recursive
    assert "executable" not in digest


def test_schedule_subagent_inherits_workspace_executor_ref(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": ["/workspace/executable"],
                    "allow": ["execute"],
                }
            ]
        }
    })
    executor_ref = {
        "type": "docker_exec",
        "id": "pb-container",
        "container_name": "pb-container",
        "network": "none",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="parent-task",
        task_contract=task_contract,
        task_metadata={"task_contract": task_contract},
        executor_ref=executor_ref,
    )
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")

    result = registry.execute(
        "schedule_subagent",
        {
            "objective": "Inspect the workspace contract.",
            "expected_output": "A concise report.",
            "role": "auditor",
            "model_lane": "light",
        },
    )

    assert "Subagent request queued" in result
    assert ctx.pending_events
    event = ctx.pending_events[0]
    assert event["executor_ref"] == executor_ref
    assert event["metadata"]["executor_ref"] == executor_ref
    child_id = event["task_id"]
    persisted = json.loads((data / "task_results" / f"{child_id}.json").read_text(encoding="utf-8"))
    assert persisted["executor_ref"] == executor_ref
    assert persisted["task_contract"]["resource_policy"]["protected_artifacts"][0]["paths"] == ["/workspace/executable"]


def test_capability_omission_manifest_surfaces_extension_discovery_failure(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.tools import tool_discovery
    from ouroboros.tools.registry import ToolRegistry

    class BoomLock:
        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    monkeypatch.setattr(extension_loader, "_lock", BoomLock())

    registry.schemas()
    tool_discovery.set_registry(registry)
    text = tool_discovery._list_available_tools(registry._ctx)

    assert "CAPABILITY_OMISSION_MANIFEST" in text
    assert "extensions" in text
    assert "boom" in text


def test_local_readonly_subagent_data_read_denies_secret_files(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    (tmp_path / "settings.json").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / "settings.tmp").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / ".settings.json.tmp.123").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "prod.env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather").mkdir(parents=True)
    (tmp_path / "state" / "skills" / "weather" / "grants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather" / ".grants.json.tmp.123").write_text("{}", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather" / "review.json.lock").write_text("{}", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "events.jsonl").write_text("{}", encoding="utf-8")
    try:
        os.symlink("settings.json", tmp_path / "alias.txt")
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(tmp_path / "settings.json", tmp_path / "hardlink.txt")
    except (OSError, NotImplementedError):
        pass

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    blocked = registry.execute("read_file", {"root": "runtime_data", "path": "settings.json"})
    assert "DATA_READ_BLOCKED" in blocked
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "settings.tmp"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": ".settings.json.tmp.123"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": ".env.local"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "prod.env"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "state/skills/weather/.grants.json.tmp.123"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "state/skills/weather/review.json.lock"})
    alias_result = registry.execute("read_file", {"root": "runtime_data", "path": "alias.txt"})
    if (tmp_path / "alias.txt").exists():
        assert "DATA_READ_BLOCKED" in alias_result
    hardlink_result = registry.execute("read_file", {"root": "runtime_data", "path": "hardlink.txt"})
    if (tmp_path / "hardlink.txt").exists():
        assert "DATA_READ_BLOCKED" in hardlink_result
    listing = registry.execute("list_files", {"root": "runtime_data", "path": "."})
    assert "settings.json" not in listing
    assert "settings.tmp" not in listing
    assert ".settings.json.tmp.123" not in listing
    assert ".env.local" not in listing
    assert "prod.env" not in listing
    assert "alias.txt" not in listing
    assert "hardlink.txt" not in listing
    assert "secret/control" in listing
    skill_state_listing = registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather"})
    assert "grants.json" not in skill_state_listing
    assert ".grants.json.tmp.123" not in skill_state_listing
    assert "review.json.lock" not in skill_state_listing
    assert "secret/control" in skill_state_listing
    assert "DATA_LIST_BLOCKED" in registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather/grants.json"})
    assert "DATA_LIST_BLOCKED" in registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather/.grants.json.tmp.123"})
    readable = registry.execute("read_file", {"root": "runtime_data", "path": "logs/events.jsonl"})
    assert "{}" in readable


def test_runtime_data_write_blocks_workspace_executor_control_state(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    state_dir = data / "state" / "workspace_executor_processes"
    state_dir.mkdir(parents=True)
    existing = state_dir / "foreground-forged.json"
    existing.write_text("original", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    direct_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "state/workspace_executor_processes/foreground-forged.json",
            "content": "{}",
        },
    )
    assert "DATA_WRITE_BLOCKED" in direct_write
    assert existing.read_text(encoding="utf-8") == "original"

    nested_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "state/headless_tasks/child/data/state/workspace_executor_processes/foreground-forged.json",
            "content": "{}",
        },
    )
    assert "DATA_WRITE_BLOCKED" in nested_write

    edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": "state/workspace_executor_processes/foreground-forged.json",
            "old_str": "original",
            "new_str": "tampered",
        },
    )
    assert "EDIT_TEXT_BLOCKED" in edit
    assert existing.read_text(encoding="utf-8") == "original"

    shell_write = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path(r'{existing}').write_text('{{\"owner\":\"ouroboros_workspace_executor\"}}')"
                ),
            ],
        },
    )
    assert "WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED" in shell_write
    assert existing.read_text(encoding="utf-8") == "original"

    node_eval_write = registry.execute(
        "run_command",
        {
            "cmd": [
                "node",
                "-e",
                f"require('fs').writeFileSync({str(existing)!r}, '{{}}')",
            ],
        },
    )
    assert "WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED" in node_eval_write
    assert existing.read_text(encoding="utf-8") == "original"


def test_local_readonly_subagent_repo_read_denies_secret_files(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / ".git").mkdir(parents=True)
    data.mkdir()
    (repo / ".git" / "credentials").write_text("https://token@example.invalid\n", encoding="utf-8")
    (repo / ".git" / "config").write_text("[credential]\n", encoding="utf-8")
    (repo / ".env.local").write_text("TOKEN=secret\nLEAK_MARKER=env\n", encoding="utf-8")
    (repo / "auth_token.json").write_text('{"token":"TOKEN_LEAK"}\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "public.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "src" / "skill_token.py").write_text("TOKEN_NAME = 'safe source symbol'\n", encoding="utf-8")
    try:
        os.symlink(".git/credentials", repo / "alias.txt")
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(repo / ".git" / "credentials", repo / "hardlink.txt")
    except (OSError, NotImplementedError):
        pass

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".git/credentials"})
    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "system_repo", "path": ".git/credentials"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".git/config"})
    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "system_repo", "path": ".git/config"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".env.local"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": "auth_token.json"})
    alias_result = registry.execute("read_file", {"path": "alias.txt"})
    if (repo / "alias.txt").exists():
        assert "REPO_READ_BLOCKED" in alias_result
    hardlink_result = registry.execute("read_file", {"path": "hardlink.txt"})
    if (repo / "hardlink.txt").exists():
        assert "REPO_READ_BLOCKED" in hardlink_result
    listing = registry.execute("list_files", {"path": "."})
    assert ".git/" not in listing
    assert ".env.local" not in listing
    assert "auth_token.json" not in listing
    assert "alias.txt" not in listing
    assert "hardlink.txt" not in listing
    assert "src/" in listing
    assert "secret/control" in listing
    system_listing = registry.execute("list_files", {"root": "system_repo", "path": "."})
    assert ".git/" not in system_listing
    assert "auth_token.json" not in system_listing
    assert "secret/control" in system_listing
    assert "REPO_LIST_BLOCKED" in registry.execute("list_files", {"path": ".git"})
    readable = registry.execute("read_file", {"path": "src/public.py"})
    assert "print('ok')" in readable
    source_with_token_name = registry.execute("read_file", {"path": "src/skill_token.py"})
    assert "safe source symbol" in source_with_token_name
    secret_search = registry.execute("search_code", {"query": "TOKEN_LEAK"})
    assert "No matches found" in secret_search
    assert "auth_token.json:" not in secret_search
    assert "SEARCH_BLOCKED" in registry.execute("search_code", {"query": "TOKEN_LEAK", "path": "auth_token.json"})
    public_search = registry.execute("search_code", {"query": "safe source symbol"})
    assert "src/skill_token.py" in public_search
    digest = registry.execute("query_code", {"op": "digest"})
    assert "auth_token.json" not in digest
    assert ".env.local" not in digest
    assert "src/skill_token.py" in digest
    cached = list((data / "state" / "code_intel").glob("*/inventory.json"))
    assert not cached


def test_local_readonly_subagent_task_drive_and_skill_payload_filters(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (data / "settings.json").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (data / "skills" / "external" / "alpha").mkdir(parents=True)
    (data / "skills" / "external" / "alpha" / "SKILL.md").write_text("hello", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "task_drive", "path": "settings.json"})
    traversal = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "../../settings.json", "path": "."},
    )
    assert "TOOL_ACCESS_BLOCKED" in traversal or "READ_FILE_ERROR" in traversal or "TOOL_ARG_ERROR" in traversal
    skill_payload_read = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": "SKILL.md"},
    )
    # v6.70.0 (owner-approved): read-only scouts may READ skill payloads — a scout
    # sent to review a skill used to be structurally blind to it. Mutation stays
    # blocked (pinned in test_owner_facing_honesty.py).
    assert "TOOL_ACCESS_BLOCKED" not in skill_payload_read
    assert "hello" in skill_payload_read


# ---------------------------------------------------------------------------
# Discovery path drift test
# ---------------------------------------------------------------------------


def test_discovery_uses_ssot_not_registry_core_names():
    """tool_discovery.py must use SSOT (via tool_policy), not registry.CORE_TOOL_NAMES."""
    import ouroboros.tools.tool_discovery as td
    source = inspect.getsource(td)
    # Must import from tool_policy (SSOT-aware)
    assert "tool_policy" in source, (
        "tool_discovery.py must import from tool_policy for SSOT-aware non-core listing"
    )
    # Must NOT call _registry.list_non_core_tools() — that uses the registry's own set
    assert "_registry.list_non_core_tools()" not in source, (
        "tool_discovery.py must not call _registry.list_non_core_tools() — "
        "that uses registry.py's local CORE_TOOL_NAMES, not the SSOT"
    )


def test_enable_tools_distinguishes_policy_hidden_from_missing(tmp_path):
    """F3 (2026-08-10 saga): a registered tool filtered by policy must answer
    'hidden by policy: <reason>', not the same 'Not found' as a typo'd name."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import tool_discovery as td
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )
    td.set_registry(registry)
    out = td._enable_tools(registry._ctx, tools="write_file, definitely_not_a_tool")
    assert "Hidden by policy" in out
    assert "write_file — hidden by the read-only subagent profile" in out
    assert "❌ Not found: definitely_not_a_tool" in out
    assert "write_file" not in out.split("Not found")[-1]

    # Workspace focus is not a hidden-policy reason: system lifecycle tools are
    # visible and retain their own target/commit governance.
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True, exist_ok=True)
    ws_registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    ws_registry.set_context(
        ToolContext(
            repo_dir=system_repo, drive_root=data,
            workspace_root=workspace, workspace_mode="external",
        )
    )
    td.set_registry(ws_registry)
    out = td._enable_tools(ws_registry._ctx, tools="commit_reviewed")
    assert "hidden by" not in out.lower()
    assert ws_registry.get_schema_by_name("commit_reviewed") is not None


def test_policy_hidden_reason_pins_get_schema_by_name(tmp_path):
    """Drift pin (adversarial review of 9e59b05d, finding 1): policy_hidden_reason
    promises "same predicates, same order" as get_schema_by_name. Enforce the
    XOR invariant — for every registered entry, in every context variant, a tool
    is either visible (schema, no reason) or policy-hidden (no schema, reason).
    A predicate added to one method but not the other breaks this immediately."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True, exist_ok=True)

    def ctx_variants():
        yield "plain", ToolContext(repo_dir=system_repo, drive_root=data)
        yield "workspace", ToolContext(
            repo_dir=system_repo, drive_root=data,
            workspace_root=workspace, workspace_mode="external",
        )
        yield "readonly_child", ToolContext(
            repo_dir=system_repo, drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
        yield "acting_child", ToolContext(
            repo_dir=system_repo, drive_root=data,
            task_constraint=TaskConstraint(
                mode="acting_subagent", allow_enable=False, surface="external_workspace",
            ),
        )
        ephemeral = ToolContext(repo_dir=system_repo, drive_root=data)
        ephemeral.is_ephemeral_turn = True
        yield "ephemeral", ephemeral
        disabled = ToolContext(repo_dir=system_repo, drive_root=data)
        disabled.task_contract = {"disabled_tools": ["write_file", "delegate_start"]}
        yield "contract_disabled", disabled

    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    for label, ctx in ctx_variants():
        registry.set_context(ctx)
        drift = []
        for name in list(registry._entries):
            schema = registry.get_schema_by_name(name)
            reason = registry.policy_hidden_reason(name)
            if (schema is None) != (reason is not None):
                drift.append((name, schema is not None, reason))
        assert not drift, f"policy_hidden_reason drifted from get_schema_by_name in ctx={label}: {drift}"


def test_policy_hidden_reason_covers_contract_disabled_unregistered_names(tmp_path):
    """ADDENDUM 4 (2026-08-10 amendments): the declarative contract policy applies
    across ALL discovery sources, so a contract-disabled extension/MCP name (not
    in ``_entries``) must answer with the disabled reason instead of "not found"
    — the contract check precedes the registration check, mirroring
    get_schema_by_name's order. Unknown un-disabled names still answer None."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system_repo, data = tmp_path / "system", tmp_path / "data"
    for path in (system_repo, data):
        path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    ctx = ToolContext(repo_dir=system_repo, drive_root=data)
    ctx.task_contract = {"disabled_tools": ["someext_generate", "write_file"]}
    registry.set_context(ctx)

    assert "someext_generate" not in registry._entries  # an extension-shaped name
    assert registry.policy_hidden_reason("someext_generate") == (
        "disabled by this task's contract (disabled_tools)"
    )
    assert registry.policy_hidden_reason("write_file") == (
        "disabled by this task's contract (disabled_tools)"
    )
    assert registry.policy_hidden_reason("no_such_tool_anywhere") is None
    assert registry.policy_hidden_reason("") is None


def test_enable_tools_hidden_label_is_shared_between_surfaces():
    """Drift pin (adversarial review of 9e59b05d, finding 5): the hidden-vs-missing
    classification exists on TWO enable_tools surfaces (tool_discovery and the
    loop's override). Pin both to policy_hidden_reason and the identical label so
    the surfaces cannot silently diverge in honesty wording."""
    import ouroboros.loop as loop_mod
    import ouroboros.tools.tool_discovery as td

    label = "🚫 Hidden by policy (the tool exists but this task cannot use it)"
    loop_src = inspect.getsource(loop_mod)
    td_src = inspect.getsource(td)
    assert label in loop_src, "loop enable_tools override lost the shared hidden-by-policy label"
    assert label in td_src, "tool_discovery lost the shared hidden-by-policy label"
    assert loop_src.count("policy_hidden_reason(") >= 1
    assert td_src.count("policy_hidden_reason(") >= 1


def test_discovery_path_consistent_with_policy():
    """list_available_tools must return the same non-core set as tool_policy.list_non_core_tools."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import list_non_core_tools as policy_non_core
    import ouroboros.tools.tool_discovery as td

    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    td.set_registry(registry)

    # Get what tool_policy says (SSOT)
    policy_names = {t["name"] for t in policy_non_core(registry)}
    # Remove meta-tools (discovery excludes them from its listing)
    policy_names -= {"list_available_tools", "enable_tools"}

    # Get what discovery tool shows
    from ouroboros.tools.registry import ToolContext
    ctx = ToolContext(repo_dir=tmp, drive_root=tmp)
    output = td._list_available_tools(ctx)

    if not policy_names:
        assert "All tools are already" in output
    else:
        for name in policy_names:
            assert name in output, (
                f"tool_policy says '{name}' is non-core but discovery doesn't show it"
            )
