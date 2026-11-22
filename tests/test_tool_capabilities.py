"""Tests for tool capability SSOT and no-drift invariants.

Verifies:
- tool_capabilities.py is the single source of truth
- tool_policy.py imports from capabilities (no local copy)
- loop_tool_execution.py imports from capabilities (no local copy)
- run_shell list-cmd happy path (string-cmd cascade lives in test_shell_run_shell.py)

search_code classification and behavior were split out verbatim into
tests/test_tool_capabilities_search_code.py.
"""
import inspect
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
    # Nano's schema-residency projection is intentionally a derived set in
    # this module; capability sets themselves remain owned by tool_capabilities.
    assert "NANO_SCHEMA_META_NAMES = META_TOOL_NAMES | frozenset" in source


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
        "delegate_message",
        "switch_model", "send_photo", "send_video", "send_file", "send_links",
        "commit_reviewed", "start_task", "vcs_restore",
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

    # Exact callable aliases intentionally have no second advertised schema.
    aliases = {entry.name for entry in registry._entries.values() if entry.alias_for in names}
    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES <= names | aliases, (
        f"read-only child tools missing from the workspace tool surface: "
        f"{sorted(LOCAL_READONLY_SUBAGENT_TOOL_NAMES - names)}"
    )
    assert ACTING_SUBAGENT_TOOL_NAMES <= names | aliases, (
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
    """The executor's name rules come from capabilities: the frozen legacy pool and the segment plan."""
    from ouroboros import loop_tool_execution as loop, tool_capabilities as cap
    assert loop.legacy_pool_response is cap.legacy_pool_response and loop.call_segments is cap.call_segments


INERT_READS = {  # a name joins only after its handler and its observable effects are checked
    "read_file", "list_files", "read_archive", "search_code", "query_code", "recent_tasks", "chat_history",
    "vcs_read", "get_task_result", "list_projects", "memory_read", "search_ui_translations", "inspect_subagent",
    "knowledge_read", "knowledge_list", "workpad_read", "journal_read", "tree_read", "peek_task", "live_roots",
    "memory_map", "list_skills", "youtube_transcript", "compare_subagent_patches",
}
# Left out, each for a reproduced reason (review of 21.11.2026).
KEPT_ALONE = {
    "web_search": "paid: a second search in one group passes the budget admission that stopped it before",
    "query_data": "save_as publishes a file a neighbour reads, and it reads logs a neighbour writes",
    "inspect_harness": "it publishes state/harness_programs.json",
    "review_status": "it writes git objects a neighbouring vcs_read resolves",
    "list_image_models": "the engine gateway it reads can publish state/harness_programs.json",
    "schedule_subagent": "a task-state or events-log read in the same group does not see the child yet",
    "github_read": "a download or a large answer publishes a file a following artifact_store listing sees",
    "list_available_tools": "registry.schemas() rebuilds the registry's shared capability-omission list",
    "browse_page": "it changes the current page of the task's one sticky browser",
    "view_image": "it appends to ctx.messages from the call thread",
    "extract_video_frames": "a producer", "vlm_query": "a model call that produces", "analyze_screenshot":
    "it reads the page a neighbouring browser call leaves", "ocr_pdf": "heavy", "extract_pdf_text": "heavy",
    "codebase_health": "heavy", "skill_search": "it refreshes a source cache", "skill_sources": "it edits sources",
    "delegate_start": "a later step, with the folder registration it holds",
    "service_status": "it advances one shared readiness cursor per service: two at once can skip the READY mark",
    "service_logs": "it publishes the log blob whose reference it returns",
}


def test_consecutive_inert_reads_form_groups_and_every_other_call_runs_alone():
    from ouroboros.tool_capabilities import call_segments

    assert call_segments(["read_file", "knowledge_read", "run_command", "search_code", "tree_read"]) == [
        [0, 1], [2], [3, 4]]
    assert call_segments(["edit_text", "read_file"]) == [[0], [1]], "a read after a write waits for it"
    for writer in ("write_file", "edit_text", "edit_batch", "apply_patch"):
        assert call_segments(["read_file", "knowledge_read", writer, "search_code", "tree_read"]) == [
            [0, 1], [2], [3, 4]], writer
    assert call_segments(["plan_task", "plan_task"]) == [[0], [1]]
    assert call_segments([]) == [] and call_segments(["read_file"]) == [[0]]
    for wait in ("wait_task", "wait_tasks", "await_messages", "delegate_wait", "finish_task", "escalate"):
        assert call_segments(["read_file", wait, "read_file"]) == [[0], [1], [2]], wait
    for alone in [*KEPT_ALONE, "ext_7_probe_read", "mcp_probe__lookup"]:
        assert call_segments(["knowledge_read", alone, "memory_read"]) == [[0], [1], [2]], alone
    for reproduced in (["query_data", "read_file", "write_file"], ["web_search", "web_search", "write_file"],
                       ["schedule_subagent", "tree_read", "write_file"], ["review_status", "vcs_read"],
                       ["service_status", "service_status", "write_file"], ["service_logs", "list_files", "stop_service"]):
        assert call_segments(reproduced) == [[i] for i in range(len(reproduced))], reproduced


def test_the_inert_class_is_its_closed_names_and_the_older_sets_keep_their_members():
    """The older sets have other readers (cards, the check journal, Presence speech), so they stay as they were; the
    inert class is its own closed list, pinned name by name, and shares no name with a writing, mutating, delivering
    or completion set."""
    from ouroboros import tool_capabilities as cap

    assert cap.INERT_READ_TOOLS == INERT_READS, "a name joins only after its handler and its observable effects are checked"
    for name, reason in KEPT_ALONE.items():
        assert name not in cap.INERT_READ_TOOLS, f"{name} runs alone: {reason}"
    assert not any(name.startswith(("ext_", "mcp_")) for name in cap.INERT_READ_TOOLS), "extension and MCP calls run alone"
    effects = (cap.OBSERVE_WORLD_MUTATION_TOOLS | cap.FOREGROUND_MUTATIVE_TOOLS | cap.REVIEWED_MUTATIVE_TOOLS
               | cap.COMPLETION_RECEIPT_TOOLS | cap.OWNER_DELIVERY_TOOL_NAMES | cap.STATEFUL_BROWSER_TOOLS
               | cap.OWN_FORM_TOOL_NAMES | cap.PARALLEL_SAFE_ENQUEUE_TOOLS)
    assert not cap.INERT_READ_TOOLS & effects, cap.INERT_READ_TOOLS & effects
    assert len(cap.READ_ONLY_PARALLEL_TOOLS) == 18 and cap.PARALLEL_SAFE_ENQUEUE_TOOLS == {"schedule_subagent"}
    assert cap.LEGACY_POOL_TOOLS == {  # history as frozen 21.11.2026, never the live sets above
        "inspect_harness", "inspect_subagent", "read_file", "list_files", "read_archive", "search_code", "query_code",
        "query_data", "recent_tasks", "web_search", "chat_history", "vcs_read", "service_status", "service_logs",
        "get_task_result", "list_projects", "memory_read", "search_ui_translations", "schedule_subagent"}
    assert cap.PURE_READ_TOOLS - cap.READ_ONLY_PARALLEL_TOOLS == {
        "browse_page", "view_image", "github_read", "knowledge_read", "youtube_transcript", "workpad_read"}


def test_every_inert_read_is_an_unchecked_silent_call_by_its_real_declarations(tmp_path):
    """A group never asks the Safety model several times at once or holds declared words: every name's effective
    policy is ``skip`` (TOOL_POLICY, else its own ToolEntry.safety_policy), no action of it is checked, and it
    declares no speech. ``safety`` is only read here."""
    from ouroboros import safety
    from ouroboros.tool_capabilities import INERT_READ_TOOLS
    from ouroboros.tool_effects import effect_of
    from ouroboros.tools.registry import ToolRegistry

    entries = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")._entries

    def unchecked_silent(name):
        entry = entries[name]
        policy = safety.TOOL_POLICY.get(name) or entry.safety_policy
        return (policy == safety.POLICY_SKIP and name not in safety.CHECKED_ACTIONS
                and not safety._declared_speech(name, effect_of(name)))

    for name in sorted(INERT_READ_TOOLS):
        assert name in entries, f"{name} is not a registered tool"
        assert unchecked_silent(name), name
    # The check itself tells such calls apart: a conditionally checked command and a checked action are not.
    assert not unchecked_silent("run_command") and not unchecked_silent("manage_schedules")


def test_extract_video_frames_visible_where_media_siblings_are_visible():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        ROUTE_TAIL_TOOL_NAMES,
    )

    for tool_set in (LOCAL_READONLY_SUBAGENT_TOOL_NAMES, ACTING_SUBAGENT_TOOL_NAMES):
        assert {"ocr_pdf", "youtube_transcript", "extract_video_frames"} <= tool_set
    assert {"ocr_pdf", "extract_pdf_text"} <= CORE_TOOL_NAMES  # the document readers stay always sent
    # On a route that cannot carry every schema the two one-off media helpers wait together in its tail.
    assert {"youtube_transcript", "extract_video_frames"} <= ROUTE_TAIL_TOOL_NAMES - CORE_TOOL_NAMES


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
    result = _run_shell(ctx, [sys.executable, "-c", "print('hello')"])
    assert "hello" in result


# ---------------------------------------------------------------------------
# Initial tool visibility
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# schedule_subagent core classification tests
# ---------------------------------------------------------------------------











@pytest.mark.parametrize(
    "constraint",
    [
        pytest.param(
            "readonly", id="local-readonly",
        ),
        pytest.param(
            "acting", id="acting",
        ),
    ],
)
def test_child_profiles_expose_existing_descendant_controls(tmp_path, constraint):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_constraint = (
        TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
        if constraint == "readonly"
        else TaskConstraint(
            mode="acting_subagent",
            surface="external_workspace",
            write_root=str(workspace),
            allow_enable=False,
        )
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data")
    registry.set_context(ToolContext(
        repo_dir=tmp_path,
        drive_root=tmp_path / "data",
        workspace_root=workspace if constraint == "acting" else None,
        workspace_mode="external" if constraint == "acting" else "",
        task_constraint=task_constraint,
    ))

    names = {item["function"]["name"] for item in initial_tool_schemas(registry)}
    assert {"peek_task", "cancel_task", "discard_child_result"} <= names


def test_local_readonly_verify_is_typed_zero_run_only(tmp_path, monkeypatch):
    """Readonly actor-first sessions may disclose no-leaf state, never run generic checks."""
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.outcomes import (
        read_verification_receipts,
        verification_receipts_path,
    )
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    data = tmp_path / "data"
    data.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path,
        drive_root=data,
        task_id="readonly-actor",
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
    )
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
        "exact_start_pending": False,
        "zero_run_evidence_status": "unknown",
        "zero_run_evidence_gaps": ["malformed_jsonl"],
    }
    verification_receipts_path(data, "readonly-actor", create=True).write_text(
        '{"contract_kind":"delegation_zero_run","zero_run":true',
        encoding="utf-8",
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=data)
    registry.set_context(ctx)

    from ouroboros.tool_policy import initial_tool_schemas
    initial_names = {item["function"]["name"] for item in initial_tool_schemas(registry)}
    assert "verify_and_record" in initial_names
    verify_schema = registry.get_schema_by_name("verify_and_record")
    assert verify_schema is not None
    assert verify_schema["function"]["parameters"]["properties"]["contract_kind"]["enum"] == ["delegation_zero_run"]
    assert "check" not in verify_schema["function"]["parameters"]["properties"]

    blocked = registry.execute(
        "verify_and_record",
        {"contract_kind": "explicit_command", "check": [sys.executable, "-c", "print('must not run')"]},
    )
    assert "local_readonly_subagent" in blocked
    assert read_verification_receipts(data, "readonly-actor") == []

    allowed = registry.execute(
        "verify_and_record",
        {
            "contract_kind": "delegation_zero_run",
            "zero_run_decision": "unknown",
            "zero_run_basis": "The configured route has not started a physical leaf.",
        },
    )
    assert "UNKNOWN" in allowed
    receipts = read_verification_receipts(data, "readonly-actor")
    assert receipts[-1]["contract_kind"] == "delegation_zero_run"
    assert receipts[-1]["regrounds_zero_run_authority"] is True
    assert receipts[-1]["prior_zero_run_evidence_gaps"] == ["malformed_jsonl"]
    assert ctx._configured_actor_bootstrap["exact_start_pending"] is False
    assert "zero_run_evidence_status" not in ctx._configured_actor_bootstrap
    assert "zero_run_evidence_gaps" not in ctx._configured_actor_bootstrap
    assert registry.get_schema_by_name("verify_and_record") is None

    # A forced direct handler call must also fail closed after the physical leaf
    # starts, even if a caller bypasses registry schema/dispatch discovery.
    ctx._configured_actor_bootstrap["physical_started"] = True
    assert registry.get_schema_by_name("verify_and_record") is None
    from ouroboros.tools.verify import _verify_and_record
    late = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="complete",
        zero_run_basis="too late",
    )
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in late


def test_ordinary_fresh_start_ignores_actor_first_zero_run_receipt_surface(tmp_path):
    from ouroboros.outcomes import verification_receipts_path
    from ouroboros.tools.delegate_integration import claimed_start_request
    from ouroboros.tools.registry import ToolContext

    data = tmp_path / "data"
    data.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path,
        drive_root=data,
        task_id="ordinary-root",
    )
    verification_receipts_path(data, ctx.task_id, create=True).write_text(
        '{"contract_kind":"delegation_zero_run"', encoding="utf-8",
    )

    claimed, refusal = claimed_start_request(
        data,
        actor_ctx=ctx,
        run_id="",
        task_id=ctx.task_id,
        idempotency_key="ordinary-root-invocation",
        invocation_id="ordinary-root-invocation",
        max_seconds=30,
        request={"prompt": "ordinary root delegation"},
        project_id="",
        project_owned=False,
        route="codex",
    )

    assert claimed is True
    assert refusal == {}


def test_verify_never_claims_recorded_when_receipt_custody_fails(
    tmp_path, monkeypatch,
):
    import ouroboros.tools.verify as verify
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="receipt-fail")
    monkeypatch.setattr(verify, "append_verification_receipt", lambda *_a, **_k: False)

    output = verify._verify_and_record(
        ctx,
        contract_kind="no_visible_machine_contract",
        check="manual visual check with disclosed residual risk",
    )

    assert "receipt_custody_failed" in output
    assert "no durable receipt was recorded" in output
    assert "recorded as a receipt" not in output















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
        disabled = ToolContext(repo_dir=system_repo, drive_root=data)
        disabled.task_contract = {"disabled_tools": ["write_file", "delegate_start"]}
        yield "contract_disabled", disabled

    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    for label, ctx in ctx_variants():
        registry.set_context(ctx)
        drift = []
        for name in list(registry._entries):
            if registry._entries[name].alias_for:
                # A compat alias is a non-public name by design: both methods
                # answer as for an unknown name (no schema, no reason).
                assert registry.get_schema_by_name(name) is None
                assert registry.policy_hidden_reason(name) is None
                continue
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


def test_discovery_has_one_implementation_bound_by_the_loop(tmp_path):
    """#1262 (supersedes the two-surface drift pin): the loop binds tool_discovery's
    handlers to its resident list instead of carrying its own copies, so the
    hidden-vs-missing answer has exactly one author."""
    import ouroboros.loop as loop_mod
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    loop_src = inspect.getsource(loop_mod)
    assert "policy_hidden_reason(" not in loop_src

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_contract = {"disabled_tools": ["write_file"]}
    registry.set_context(ctx)
    loop_mod._setup_dynamic_tools(registry, [], [])
    out = registry.execute("enable_tools", {"tools": "write_file, definitely_not_a_tool"})
    assert "write_file — disabled by this task's contract (disabled_tools)" in out
    assert "❌ Not found: definitely_not_a_tool" in out
    assert "write_file" not in out.split("Not found")[-1]


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_discovery_lists_the_callable_catalog_in_every_mode(mode):
    """list_available_tools reads ToolRegistry.schemas() in every context mode and
    names every callable tool once, whether or not its schema is resident."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.tools.tool_discovery as td

    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    td.set_registry(registry)
    ctx = ToolContext(repo_dir=tmp, drive_root=tmp, active_context_mode=mode)
    registry.set_context(ctx)
    output = td._list_available_tools(ctx, namespace="builtin")

    names = [s["function"]["name"] for s in registry.schemas()]
    assert names
    for name in names:
        assert f"- {name} [" in output, f"schemas() offers {name!r} but discovery does not list it"
    assert ("[not loaded]" in output) is (mode == "nano")


def test_burst_absorb_clause_states_the_prefix_write_cost():
    """The burst affordance names its cash side, not only its latency win, as facts; the choice
    between them is the caller's (#1175), and absorbing is wait_tasks' own contract."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import initial_tool_schemas
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    descriptions = {s["function"]["name"]: s["function"]["description"] for s in initial_tool_schemas(registry)}
    description = descriptions["schedule_subagent"]
    assert "Children scheduled in one round run concurrently" in description
    assert ("on cache-write-priced routes each sibling started before the first responds pays a full "
            "prefix write") in description
    assert "With mode=any_terminal it returns as soon as the FIRST child finishes" in descriptions["wait_tasks"]


def test_escalate_is_available_wherever_planning_runs():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES, CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    assert all("escalate" in names for names in
               (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES, ACTING_SUBAGENT_TOOL_NAMES))


def test_enable_tools_loads_a_whole_skill_by_its_name(tmp_path, monkeypatch):
    """The 27.10 call enabled 7 of 24 desktop tools one name at a time and told the owner the rest did not
    exist. A skill's name (or its extension namespace) now loads each of its tools exactly as naming it alone
    would: the request's tools change once, a withheld tool is counted and never named, resident schemas
    answer "already loaded", and a skill that is not live or a stray name loads nothing. An MCP server with
    tools, then a tool, then a policy-hidden tool, then the exact namespace keep their answer before a
    same-named skill; a skill called mcp_<slug> loads when no such server has tools."""
    from ouroboros import extension_loader
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools import tool_discovery as td
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    def register(skill, *shorts):
        made = [extension_surface_name(skill, short) for short in shorts]
        for name in made:
            monkeypatch.setitem(extension_loader._tools, name, {
                "name": name, "handler": lambda ctx, **kwargs: "ok", "description": f"{name} tool",
                "schema": {"type": "object", "properties": {}}, "timeout_sec": 5, "skill": skill})
        return made

    names = register("unix_computer_use", "screenshot", "click", "key", "type_text", "scroll")
    namespace = td.tool_namespace(names[0])
    dead = register("dead_skill", "probe")
    mcp_named = register("mcp_probe", "ping")
    shadow = register(namespace, "other")  # a skill named after another skill's exact namespace
    hidden_twin = register("write_file", "ghost")  # a skill named after a policy-hidden built-in
    register("broken_skill", "probe")
    observed = []

    def is_live(skill, *_a, **_k):
        observed.append(skill)
        if skill == "broken_skill":  # an unreadable review record (skill_loader raises on a non-object panel)
            raise ValueError("panel is not an object")
        return skill != "dead_skill"

    monkeypatch.setattr(extension_loader, "is_extension_live", is_live)
    adoptions = []
    monkeypatch.setattr(extension_dispatch, "_adopt_extension_generation", lambda ctx: adoptions.append(ctx) or {})
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_contract = {"disabled_tools": [names[4], "write_file"]}
    registry.set_context(ctx)
    fresh = lambda: [registry.get_schema_by_name("read_file")]  # noqa: E731
    resident, enabled = fresh(), set()

    out = td.enable_tools(registry, ctx, "unix_computer_use", resident, enabled)
    loaded = sorted(names[:4])
    assert [s["function"]["name"] for s in resident[1:]] == loaded and enabled == set(loaded)
    assert "Call schemas loaded" in out and "unix_computer_use (all 4 tools)" in out and len(adoptions) == 1
    # One family observation for the skill, then each member's own single-name check (the withheld one stops
    # at its contract before liveness): 1 + 4, where one observation per family row would make 5 + 4.
    assert observed.count("unix_computer_use") == 1 + len(loaded), observed
    assert "unix_computer_use — 1 of its 5 tools are not callable in this task (names withheld)" in out
    assert names[4] not in out
    # Every member resolves exactly as when named alone: the withheld one is refused, the others load.
    assert td._load_set(registry, [names[4]])[0] == {}
    assert set(td._load_set(registry, names[:4])[0]) == set(loaded)

    again = td.enable_tools(registry, ctx, namespace, resident, enabled)
    assert "Already loaded: " in again and "Call schemas loaded" not in again and len(resident) == 5
    assert shadow[0] not in again and len(td._load_set(registry, [namespace])[0]) == 4

    one = fresh()
    single = td.enable_tools(registry, ctx, names[1], one, set())
    assert [s["function"]["name"] for s in one] == ["read_file", names[1]] and "(all" not in single

    for missing in ("dead_skill", "windows_computer_use", "definitely_not_a_tool"):
        untouched = fresh()
        refusal = td.enable_tools(registry, ctx, missing, untouched, set())
        assert f"❌ Not found: {missing}." in refusal and "only while that skill is enabled" in refusal
        assert "Hidden by policy" not in refusal and dead[0] not in refusal and len(untouched) == 1

    by_mcp_name = fresh()
    assert "mcp_probe (all 1 tools)" in td.enable_tools(registry, ctx, "mcp_probe", by_mcp_name, set())
    assert [s["function"]["name"] for s in by_mcp_name[1:]] == mcp_named

    twin = fresh()
    refusal = td.enable_tools(registry, ctx, "write_file", twin, set())
    assert "write_file — disabled by this task's contract" in refusal and hidden_twin[0] not in refusal
    assert len(twin) == 1

    # A liveness read that raises makes that skill not live; the other names of the call still load, and the
    # whole call adopts the published generation once.
    adoptions.clear()
    mixed = fresh()
    answer = td.enable_tools(registry, ctx, f"broken_skill, {names[1]}, no_such_a, no_such_b", mixed, set())
    assert [s["function"]["name"] for s in mixed[1:]] == [names[1]] and "Not found: broken_skill" in answer
    assert len(adoptions) == 1

    # A retained Presence ceiling refuses any name outside its grants, tool or not. The same grants load the
    # granted tools by the skill's name and by a tool's own name; a refused built-in keeps its own refusal.
    import ouroboros.presence_authority as authority

    grants = {"enable_tools", names[0], names[1]}
    monkeypatch.setattr(authority, "presence_ceiling_from_context", lambda _ctx: object())
    monkeypatch.setattr(authority, "presence_ceiling_allows_tool", lambda _ceiling, tool: tool in grants)
    by_skill, by_tool = [], []
    family = td.enable_tools(registry, ctx, "unix_computer_use", by_skill, set())
    assert [s["function"]["name"] for s in by_skill] == sorted(names[:2]) and "unix_computer_use (all 2 tools)" in family
    td.enable_tools(registry, ctx, names[0], by_tool, set())
    assert [s["function"]["name"] for s in by_tool] == [names[0]]
    ceiling = td.enable_tools(registry, ctx, "read_file", [], set())
    assert "read_file — outside this presence task's positive capability ceiling" in ceiling


def test_every_catalog_that_names_extract_pdf_text_names_read_archive():
    """The two local file readers share every class; the one deliberate difference is the round-one core set,
    which read_archive leaves for on-demand loading (read_file's binary refusal names it)."""
    import ouroboros.tool_capabilities as caps
    import ouroboros.consciousness_authority as authority
    import ouroboros.safety as safety
    import ouroboros.tools.registry_guards as guards

    catalogs = {f"{module.__name__}.{name}": value
                for module in (caps, authority, safety, guards) for name, value in vars(module).items()
                if isinstance(value, (frozenset, set, tuple, list, dict)) and "extract_pdf_text" in value}
    assert {"ouroboros.tool_capabilities.CORE_TOOL_NAMES", "ouroboros.tool_capabilities.ACTING_SUBAGENT_TOOL_NAMES",
            "ouroboros.tool_capabilities.LOCAL_READONLY_SUBAGENT_TOOL_NAMES"} <= set(catalogs)
    missing = [name for name, value in catalogs.items()
               if name != "ouroboros.tool_capabilities.CORE_TOOL_NAMES" and "read_archive" not in value]
    assert not missing, missing
    assert "read_archive" not in caps.CORE_TOOL_NAMES
