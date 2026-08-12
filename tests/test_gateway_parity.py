from __future__ import annotations

import pathlib
import re
from typing import get_args, get_type_hints

from ouroboros.gateway.contracts import (
    HTTP_ENDPOINTS,
    WS_MESSAGE_TYPES,
    ChatInbound,
    ChatOutbound,
    OwnerScopeReviewFloorResponse,
    PhotoOutbound,
    ProjectDeleteResponse,
    ProjectEntry,
    ProjectFromTaskResponse,
    ProjectInitGitResponse,
    SkillDeleteResponse,
    SkillLifecycleQueueResponse,
    StateResponse,
    TaskCostBreakdown,
    TaskDetailResponse,
    TaskDiffResponse,
    ThreadBranchBase,
    ThreadBranchBasesResponse,
    ThreadBranchOffRequest,
    ThreadCreateRequest,
    ThreadDeleteRequest,
    ThreadDiffResponse,
    ThreadEntry,
    ThreadLifecycleResponse,
    ThreadQueueNotice,
    ThreadLocation,
    ThreadMergeBackRequest,
    ThreadResponse,
    ThreadUpdateRequest,
    ThreadWorktreeRemoveRequest,
    ThreadWorktreeResponse,
    UiPreferencesResponse,
    UpdateApplyErrorResponse,
    UpdateApplyRequest,
    UpdateApplySuccessResponse,
    UpdateMergePlan,
    UpdatePreflightRequest,
    UpdatePreflightResponse,
    UpdateStatusReadyOutbound,
    VideoOutbound,
    WorkspaceGitInitDecision,
    ClaudexorStatusReads,
    ClaudexorStatusResponse,
)
from ouroboros.gateway.router import collect_routes


def _js_typedef_fields(text: str, name: str) -> set[str]:
    match = re.search(rf"@typedef \{{Object\}} {name}\b(?P<body>.*?)\n \*/", text, re.S)
    assert match, f"api_types.js missing {name}"
    # Types nest braces (``{Object<string, {project_id: string}>}``), so scan for the BALANCED
    # closing brace instead of the first one — a non-greedy ``[^}]+`` silently mis-parses those
    # properties and makes the field set look like it drifted when it has not.
    fields: set[str] = set()
    for line in match.group("body").split("\n"):
        head, sep, rest = line.partition("@property {")
        if not sep:
            continue
        depth = 1
        for idx, char in enumerate(rest):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                identifier = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", rest[idx + 1:])
                if identifier:
                    fields.add(identifier.group(1))
                break
    return fields


def _contains_none(annotation) -> bool:
    return annotation is type(None) or any(_contains_none(arg) for arg in get_args(annotation))


def test_gateway_contract_endpoint_index_matches_router_and_types(tmp_path):
    tokens: set[str] = set()
    for route in collect_routes(data_dir=tmp_path):
        path = getattr(route, "path", "")
        if not path:
            continue
        methods = getattr(route, "methods", None)
        if methods is None:
            tokens.add(f"WS {path}")
            continue
        normalized = sorted(m for m in methods if m not in {"HEAD", "OPTIONS"})
        if set(normalized) == {"DELETE", "GET", "PATCH", "POST", "PUT"}:
            tokens.add(f"ANY {path}")
        else:
            for method in normalized:
                tokens.add(f"{method} {path}")
    contract_tokens = set(HTTP_ENDPOINTS)
    missing = contract_tokens - tokens
    extra = tokens - contract_tokens
    assert not missing, f"HTTP_ENDPOINTS includes routes not mounted by gateway.router: {sorted(missing)}"
    assert not extra, f"gateway.router mounts routes missing from HTTP_ENDPOINTS: {sorted(extra)}"
    text = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    version = (pathlib.Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip()
    assert f"GATEWAY_CONTRACT_VERSION = '{version}'" in text
    for name in (
        "StateResponse",
        "HealthResponse",
        "SettingsMeta",
        "OpenAICompatibleModelsResponse",
        "UiPreferencesResponse",
        "ChatInbound",
        "ChatOutbound",
        "PhotoOutbound",
        "VideoOutbound",
        "MessageAnnotationOutbound",
        "UploadResponse",
        "TaskCreateResponse",
        "TaskEvent",
        "TaskListResponse",
        "TaskCostBreakdown",
        "TaskDetailResponse",
        "TaskCancelResponse",
        "TaskDiffResponse",
        "LogTailResponse",
        "SkillDeleteResponse",
        "UpdateMergePlan",
        "UpdatePreflightRequest",
        "UpdatePreflightResponse",
        "UpdateApplyRequest",
        "UpdateApplySuccessResponse",
        "UpdateApplyErrorResponse",
        "UpdateStatusReadyOutbound",
        "ThreadEntry",
        "ThreadCreateRequest",
        "ThreadUpdateRequest",
        "ThreadResponse",
        "ThreadLocation",
        "ThreadBranchBase",
        "ThreadBranchBasesResponse",
        "ThreadBranchOffRequest",
        "ThreadMergeBackRequest",
        "ThreadWorktreeRemoveRequest",
        "ThreadDeleteRequest",
        "ThreadWorktreeResponse",
        "ThreadDiffResponse",
        "ThreadLifecycleResponse",
        "ThreadQueueNotice",
        "WorkspaceGitInitDecision",
        "ProjectInitGitResponse",
        "ProjectFromTaskResponse",
    ):
        assert re.search(rf"@typedef \{{Object\}} {name}\b", text), f"api_types.js missing {name}"
    api_client = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_client.js").read_text(
        encoding="utf-8"
    )
    assert "openAICompatibleModels" in api_client
    assert "taskDiff" in api_client
    # Thread lifecycle: the browser must reach every route the gateway mounts,
    # or the UI phase would hand-roll fetches outside the api_client seam.
    for call in ("projectThreadCreate", "projectThreadUpdate", "projectThreadFork"):
        assert call in api_client, f"api_client.js missing {call}"
    # T3: branching, the inspected removal and the thread-checkout diff ride the
    # SAME seam. A UI phase that hand-rolls one of these fetches loses the shared
    # path builder and, with it, the encoding of an owner-typed project id.
    for call in (
        "threadBranchBases", "threadBranchOff", "threadMergeBack",
        "threadWorktree", "threadWorktreeRemove", "threadDiff",
        "threadArchive", "threadRestore", "threadDelete",
    ):
        assert call in api_client, f"api_client.js missing {call}"
    # T2: the owner's YES to the git_init_required offer must be reachable through
    # the one client seam, or the UI phase hand-rolls a fetch for the single route
    # that writes into the owner's own folder.
    assert "projectInitGit" in api_client, "api_client.js missing projectInitGit"
    # v6.80.0: the two contracts extended this release join the FIELD-level parity list. The name-level
    # loop above cannot see a new @property, so an ABI field added on the Python side would otherwise
    # never have to appear in the browser's typedef (ARCHITECTURE.md §11.3).
    # Project threads (T0): UiPreferencesResponse and ProjectEntry were BOTH missing from this loop —
    # ProjectEntry's JSDoc had already drifted (no `origin`, no `created_at`) with nothing to catch it,
    # and UiPreferencesResponse is the surface the nested per-thread read cursor will migrate.
    for cls in (ChatInbound, ChatOutbound, PhotoOutbound, VideoOutbound,
                StateResponse, OwnerScopeReviewFloorResponse, UpdateMergePlan,
                UpdatePreflightRequest, UpdatePreflightResponse, UpdateApplyRequest,
                UpdateApplySuccessResponse, UpdateApplyErrorResponse,
                UpdateStatusReadyOutbound, TaskCostBreakdown, TaskDetailResponse,
                ClaudexorStatusReads, ClaudexorStatusResponse, TaskDiffResponse,
                UiPreferencesResponse, ProjectEntry,
                ThreadEntry, ThreadCreateRequest, ThreadUpdateRequest, ThreadResponse,
                ThreadLocation, ThreadBranchBase, ThreadBranchBasesResponse,
                ThreadBranchOffRequest, ThreadMergeBackRequest, ThreadWorktreeRemoveRequest,
                ThreadDeleteRequest,
                ThreadWorktreeResponse, ThreadDiffResponse, ThreadLifecycleResponse, ThreadQueueNotice,
                WorkspaceGitInitDecision, ProjectInitGitResponse,
                # I1: the project delete answers what happened to its threads' checkouts,
                # and this class was only in the NAME-level loop above — so the new
                # fields could have landed on one side alone.
                ProjectDeleteResponse,
                ProjectFromTaskResponse):
        expected = set(get_type_hints(cls, include_extras=True))
        actual = _js_typedef_fields(text, cls.__name__)
        assert actual == expected, f"{cls.__name__} JSDoc fields drifted: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    assert UpdatePreflightResponse.__required_keys__ == frozenset({"merge_plan"})
    assert re.search(r"@property \{'auto_merge'\|'assisted'\|'manual'\|'replace'\} strategy\b", text)
    assert re.search(r"@property \{string=\} expected_base_sha\b", text)
    assert re.search(r"@property \{string=\} expected_target_sha\b", text)
    assert re.search(r"@property \{boolean=\} confirm_recovery\b", text)
    assert re.search(r"@property \{'ok'\|'restart_required'\|'assisted_started'\|'manual'\} status\b", text)
    assert re.search(r"@typedef \{Object\} UpdateApplyErrorResponse.*?@property \{string\} error\b", text, re.S)
    assert re.search(r"@property \{boolean\} context_mode_auto_low\b", text), (
        "StateResponse.context_mode_auto_low must be a JSDoc boolean — the owner control branches on it"
    )
    assert re.search(r"@property \{string\} deprecation_notice\b", text), (
        "OwnerScopeReviewFloorResponse.deprecation_notice must be declared for the browser"
    )
    # The bounded-listing count: a `[:N]` slice on an owner-facing envelope must
    # travel with the TRUE size of the set (DEVELOPMENT.md, "No silent
    # truncation"). It is a NUMBER on both sides — a client left to reach for
    # `dirty_files.length` is a client that states the CAP as the magnitude,
    # which is exactly what the removal refusal did with 800 modified files.
    assert "dirty_files_total" in get_type_hints(ThreadWorktreeResponse, include_extras=True), (
        "ThreadWorktreeResponse must declare dirty_files_total beside dirty_files"
    )
    assert re.search(r"@property \{number=\} dirty_files_total\b", text), (
        "ThreadWorktreeResponse.dirty_files_total must be a JSDoc number in the browser mirror"
    )
    assert re.search(r"@property \{boolean=\} force_plan\b", text), "ChatInbound missing force_plan"
    for field in ("model_lane", "requested_model_lane", "effective_model_lane", "model", "task_group_id"):
        assert re.search(rf"@property \{{string=\}} {field}\b", text), f"ChatOutbound missing {field}"
    for field in ("source", "line", "root"):
        assert re.search(rf"@property \{{[^}}]+=\}} {field}\b", text), f"TaskEvent missing {field}"
    for field in (
        "subagent_event",
        "subagent_task_id",
        "root_task_id",
        "parent_task_id",
        "delegation_role",
        "subagent_role",
        "task_event",
        "status",
        "result",
        "trace_summary",
        "error",
        "artifact_status",
    ):
        assert re.search(rf"@property \{{string=\}} {field}\b", text), f"ChatOutbound missing {field}"
    assert re.search(r"@property \{\?number=\} cost_usd\b", text), "ChatOutbound cost_usd must be nullable"
    assert re.search(r"@property \{number=\} chat_id\b", text), "ChatOutbound missing chat_id"
    assert re.search(r"@property \{boolean=\} worker_saturation_warning\b", text), "ChatOutbound missing worker_saturation_warning"
    assert "review_projection" in get_type_hints(ChatOutbound, include_extras=True)
    assert re.search(r"@property \{Object=\} review_projection\b", text)
    assert "setup_contract" in text
    assert re.search(r"@property \{string=\} error\b", text), "SkillDeleteResponse missing optional error"
    assert {"chat", "command", "photo", "video", "typing", "log", "heartbeat", "extension_lifecycle"} <= set(WS_MESSAGE_TYPES)
    assert "message_annotation" in WS_MESSAGE_TYPES
    assert "update_status_ready" in WS_MESSAGE_TYPES
    assert _js_typedef_fields(text, "MessageAnnotationOutbound") == {
        "type",
        "annotation_type",
        "chat_id",
        "client_message_id",
        "action",
        "target",
        "status",
        "options",
        "suppress_bubble",
        "ts",
    }


def test_task_diff_contract_declares_typed_lifecycle_in_both_mirrors():
    """The diff ABI must stay typed and stat-free in BOTH mirrors.

    The client derives the file list and +/- counts from the patch bytes, so the
    envelope must never grow a server-side stats field (a second, silently
    diverging truth), and the four lifecycle statuses plus the boolean HEAD-drift
    disclosure must be visible to a browser contributor.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    python_contract = (repo / "ouroboros" / "gateway" / "contracts.py").read_text(encoding="utf-8")
    js_contract = (repo / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")

    assert set(get_type_hints(TaskDiffResponse, include_extras=True)) == {
        "status", "source", "base_commit", "head_advanced", "blockers", "patch", "patch_sha256", "error",
    }
    assert "GET /api/tasks/{task_id}/diff" in HTTP_ENDPOINTS
    assert '"pending"|"ready"|"empty"|"blocked"' in js_contract
    assert '"workspace_patch"|"mutation_baseline"' in js_contract
    # REQUIREDNESS mirrors too, not just the field names and their vocabularies: the
    # Python side is `total=False` (an additive frozen surface promises nothing as
    # required), so the JSDoc marks the same fields optional with `=`. Pinned in this
    # spelling because the two sides drifted here once — the contract said "assume
    # nothing" while the browser mirror said "always present".
    assert "@property {boolean=} head_advanced" in js_contract
    assert "@property {string[]=} blockers" in js_contract
    assert not TaskDiffResponse.__required_keys__
    declaration = python_contract.split("class TaskDiffResponse")[1].split("\nclass ")[0]
    for banned in ("files:", "diffstat:", "add:", "del:", "truncated:"):
        assert banned not in declaration, f"TaskDiffResponse must not carry server-side stats ({banned})"


def test_gateway_money_contracts_keep_unavailable_distinct_from_zero():
    from ouroboros.gateway.contracts import StateResponse

    state_hints = get_type_hints(StateResponse, include_extras=True)
    for field in ("spent_usd", "budget_pct", "spent_calls"):
        assert _contains_none(state_hints[field]), f"StateResponse.{field} must admit ledger-unavailable null"

    chat_hints = get_type_hints(ChatOutbound, include_extras=True)
    for field in (
        "cost_usd",
        "cost_usd_with_children",
        "reserved_usd",
        "unresolved_upper_bound_usd",
        "unknown_unmetered",
    ):
        assert _contains_none(chat_hints[field]), f"ChatOutbound.{field} must admit ledger-unavailable null"
    assert {"cost_accounting_status", "cost_final", "cost_with_children_partial"} <= set(chat_hints)


def test_skill_lifecycle_queue_contract_matches_runtime_shape():
    fields = set(SkillLifecycleQueueResponse.__annotations__)

    assert {"active", "events"} <= fields
    assert {"queue", "recent_events", "running"}.isdisjoint(fields)


def test_skill_delete_contract_matches_runtime_shape():
    fields = set(SkillDeleteResponse.__annotations__)

    assert {
        "ok",
        "skill",
        "source",
        "deleted_payload_root",
        "deleted_state",
        "extension_action",
        "extension_reason",
        "error",
    } <= fields


def test_v682_cancellation_contract_fields_are_mirrored_in_both_languages():
    """The additive cancellation ABI (v6.82) must exist in BOTH mirrors: the
    host-attested cancelable marker and the cancel endpoint's cascade echo."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    python_contract = (repo / "ouroboros" / "gateway" / "contracts.py").read_text(encoding="utf-8")
    js_contract = (repo / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")

    assert "cancelable: NotRequired[bool]" in python_contract
    assert "cascade: bool" in python_contract
    assert "@property {boolean=} cancelable" in js_contract
    assert "@property {boolean=} cascade" in js_contract


def test_task_detail_cost_breakdown_emission_matches_contract(monkeypatch, tmp_path):
    """api_task_get's additive cost_breakdown object must carry EXACTLY the key
    set TaskCostBreakdown declares (the JS mirror is held to the same set by the
    field-level parity loop above), stay optional on the detail response, and be
    omitted on non-root task details."""
    import ouroboros.usage_accounting as usage_accounting
    from ouroboros.gateway.tasks import _task_cost_breakdown_view

    fake_breakdown = {
        "accounted_usd": 1.5,
        "attempt_counts": {"settled": 2},
        "subscription_sessions": 1,
        "by_task": {"root1": {"accounted_usd": 0.5}},
        "unattributed": {"task": {"accounted_usd": 0.25}},
        "delegated": {"settled_usd": 0.75},
        "unknown_unmetered": 0,
        "non_final_rows": 0,
        "cost_final": True,
    }
    monkeypatch.setattr(usage_accounting, "usage_breakdown", lambda *args, **kwargs: fake_breakdown)
    view = _task_cost_breakdown_view(tmp_path, {"task_id": "root1", "root_task_id": "root1"})
    assert view is not None
    assert set(view) == set(get_type_hints(TaskCostBreakdown, include_extras=True))
    assert view["authority"] == "physical_attempt_ledger"
    # Non-root details omit the view: subtree math is ledger-attributable only at the root.
    assert _task_cost_breakdown_view(tmp_path, {"task_id": "child1", "root_task_id": "root1"}) is None
    # The detail response declares the projection as genuinely optional — absence is a
    # legal shape (unavailable accounting is never rendered as a confident $0).
    detail_hints = get_type_hints(TaskDetailResponse, include_extras=True)
    assert detail_hints["cost_breakdown"] is TaskCostBreakdown
    assert TaskDetailResponse.__required_keys__ == frozenset()


def test_thread_entry_declares_every_field_the_projection_actually_ships(tmp_path):
    """T3R2-B3: the field-level mirror cannot catch two sides that are equally wrong.

    Live rows carry `lifecycle`, `archived_at` and `delete_error` on `/api/state`,
    on `GET /api/projects` and inside every `ThreadResponse` — the canonical
    projection normalises all three onto every row, and the client already reads
    `thread.lifecycle` to decide what a thread menu may offer. Neither the Python
    contract nor the JS typedef declared them, and ARCHITECTURE §11.3 asserted
    that T3 had added them. Parity passed because both mirrors were wrong
    identically, so this pins the projection ITSELF against the contract.
    """
    from ouroboros.project_threads_registry import project_threads
    from ouroboros.projects_registry import create_project, create_thread

    declared = set(get_type_hints(ThreadEntry, include_extras=True))
    create_project(tmp_path, "racer", name="Racer")
    forked_source = create_thread(tmp_path, "racer", name="Tuning")
    from ouroboros.projects_registry import archive_thread, fork_thread, get_project

    fork_thread(tmp_path, "racer", forked_source["id"])
    archive_thread(tmp_path, "racer", forked_source["id"])
    rows = project_threads(get_project(tmp_path, "racer"))

    assert len(rows) >= 3, rows
    shipped: set = set()
    for row in rows:
        shipped |= set(row)
    assert shipped <= declared, f"ThreadEntry does not declare: {sorted(shipped - declared)}"
    # And the three the lifecycle actually adds are really on every row.
    for row in rows:
        assert {"lifecycle", "archived_at", "delete_error"} <= set(row)
