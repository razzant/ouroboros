"""Tests for tool capability SSOT and no-drift invariants.

Verifies:
- tool_capabilities.py is the single source of truth
- tool_policy.py imports from capabilities (no local copy)
- loop_tool_execution.py imports from capabilities (no local copy)
- profile visibility parity for the round-1 tool surface
- run_shell list-cmd happy path (string-cmd cascade lives in test_shell_run_shell.py)
- tool discovery follows the SSOT rather than registry core names

The search_code, subagent-scheduling, readonly-subagent and black-box
policy halves were split verbatim into
``tests/test_tool_capabilities_search_code.py``,
``tests/test_tool_capabilities_subagent_scheduling.py``,
``tests/test_tool_capabilities_readonly_subagent.py`` and
``tests/test_tool_capabilities_black_box_policy.py``.
"""
import inspect
import pathlib
import re

import pytest
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
    from tests._shared import configure_frozen_tool_registry

    registry_cls = configure_frozen_tool_registry(monkeypatch, tmp_path)
    registry = registry_cls(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
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
