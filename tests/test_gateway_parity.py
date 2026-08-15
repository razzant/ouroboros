from __future__ import annotations

import pathlib
import re
from typing import get_args, get_type_hints

from ouroboros.gateway.contracts import (
    HTTP_ENDPOINTS,
    WS_MESSAGE_TYPES,
    ChatInbound,
    ChatOutbound,
    OnboardingCompleteRequest,
    OnboardingCompleteResponse,
    OnboardingPresetFailureResponse,
    OnboardingPresetProjection,
    OwnerScopeReviewFloorResponse,
    PhotoOutbound,
    SettingsPostCommitFailureResponse,
    SkillDeleteResponse,
    SkillLifecycleQueueResponse,
    StateResponse,
    TaskCostBreakdown,
    TaskDetailResponse,
    UpdateApplyErrorResponse,
    UpdateApplyRequest,
    UpdateApplySuccessResponse,
    UpdateMergePlan,
    UpdatePreflightRequest,
    UpdatePreflightResponse,
    UpdateStatusReadyOutbound,
    VideoOutbound,
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
        "LogTailResponse",
        "SkillDeleteResponse",
        "UpdateMergePlan",
        "UpdatePreflightRequest",
        "UpdatePreflightResponse",
        "UpdateApplyRequest",
        "UpdateApplySuccessResponse",
        "UpdateApplyErrorResponse",
        "UpdateStatusReadyOutbound",
        "OnboardingCompleteRequest",
        "OnboardingPresetProjection",
        "OnboardingCompleteResponse",
        "OnboardingPresetFailureResponse",
        "SettingsPostCommitFailureResponse",
    ):
        assert re.search(rf"@typedef \{{Object\}} {name}\b", text), f"api_types.js missing {name}"
    api_client = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_client.js").read_text(
        encoding="utf-8"
    )
    assert "openAICompatibleModels" in api_client
    # D-8: the wizard's ONE atomic completion call must exist on the browser client.
    assert "completeOnboarding" in api_client
    assert "'/api/onboarding/complete'" in api_client
    # v6.80.0: the two contracts extended this release join the FIELD-level parity list. The name-level
    # loop above cannot see a new @property, so an ABI field added on the Python side would otherwise
    # never have to appear in the browser's typedef (ARCHITECTURE.md §11.3).
    for cls in (ChatInbound, ChatOutbound, PhotoOutbound, VideoOutbound,
                StateResponse, OwnerScopeReviewFloorResponse, UpdateMergePlan,
                UpdatePreflightRequest, UpdatePreflightResponse, UpdateApplyRequest,
                UpdateApplySuccessResponse, UpdateApplyErrorResponse,
                UpdateStatusReadyOutbound, TaskCostBreakdown, TaskDetailResponse,
                OnboardingCompleteRequest, OnboardingPresetProjection,
                OnboardingCompleteResponse, OnboardingPresetFailureResponse,
                SettingsPostCommitFailureResponse,
                ClaudexorStatusReads, ClaudexorStatusResponse):
        expected = set(get_type_hints(cls, include_extras=True))
        actual = _js_typedef_fields(text, cls.__name__)
        assert actual == expected, f"{cls.__name__} JSDoc fields drifted: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    # The client's own list of facets. The shared status store is the ONE reader
    # of the `reads` block (`facetReadState`), and STATUS_FACETS is the list its
    # per-facet map and every "did the daemon answer anything at all?" predicate
    # iterate — so a facet added to the contract but not there would be
    # INVISIBLE to those consumers: a payload in which only the new facet landed
    # would read as total silence, and a surface would print a dead-daemon
    # verdict over data the daemon had just handed over. The name-level and
    # field-level loops above cannot see this — they compare Python against the
    # browser's typedef, never against the module that consumes it.
    store_js = (pathlib.Path(__file__).resolve().parent.parent
                / "web" / "modules" / "claudexor_status_store.js").read_text(encoding="utf-8")
    facets = re.search(r"export const STATUS_FACETS = \[([^\]]*)\]", store_js)
    assert facets, "claudexor_status_store.js no longer declares STATUS_FACETS"
    # Strip comments inside the literal first: a facet name mentioned in a
    # comment there would otherwise satisfy this check while the exported array
    # — the thing every consumer iterates — never grew. The array is composed of
    # named constants (FACET_CATALOG, …), so each identifier is resolved to the
    # string its `const` declaration binds; a bare string literal counts as-is.
    literal = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", facets.group(1)), flags=re.S)
    declared: set[str] = set()
    for token in re.findall(r"'[^']*'|[A-Za-z_$][A-Za-z0-9_$]*", literal):
        if token.startswith("'"):
            declared.add(token.strip("'"))
            continue
        binding = re.search(rf"const {re.escape(token)} = '([^']+)'", store_js)
        assert binding, f"STATUS_FACETS references {token}, whose string binding was not found"
        declared.add(binding.group(1))
    assert declared == set(
        get_type_hints(ClaudexorStatusReads, include_extras=True)
    ), "STATUS_FACETS drifted from ClaudexorStatusReads; the store's per-facet reads would go blind to a facet"

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
    """The additive cancellation ABI (v6.82 + phase A) must exist in BOTH
    mirrors: the host-attested cancelable marker, the cancel endpoint's cascade
    echo, and the phase-A ``cancel_state`` pending projection on the task
    detail envelope (AR2-8; the field-level parity loop above pins the exact
    TaskDetailResponse key set in both languages)."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    python_contract = (repo / "ouroboros" / "gateway" / "contracts.py").read_text(encoding="utf-8")
    js_contract = (repo / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")

    assert "cancelable: NotRequired[bool]" in python_contract
    assert "cascade: bool" in python_contract
    assert "@property {boolean=} cancelable" in js_contract
    assert "@property {boolean=} cascade" in js_contract
    assert "cancel_state: str" in python_contract
    assert "@property {string=} cancel_state" in js_contract
    # GR2-11: the intent's reason is public beside the state, in both mirrors.
    assert "cancel_reason: str" in python_contract
    assert "@property {string=} cancel_reason" in js_contract


def test_task_detail_serves_the_cancel_state_projection(tmp_path):
    """AR2-8 runtime half: an ACTIVE durable cancel intent rides the effective
    task read as ``cancel_state: "pending"`` (with ``cancel_reason`` beside it
    when the intent carries one — GR2-11) and passes through the public
    projection ``api_task_get`` serves; a settled task never carries it."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(tmp_path, "cs1", "running", result="working")
    request_cancel(tmp_path, "cs1", reason="stop")
    payload = public_task_result(load_effective_task_result(tmp_path, "cs1"))
    assert payload["cancel_state"] == "pending"
    assert payload["cancel_reason"] == "stop"

    # A reason-less intent serves the state alone — no empty-string field.
    write_task_result(tmp_path, "cs3", "running", result="working")
    request_cancel(tmp_path, "cs3")
    payload = public_task_result(load_effective_task_result(tmp_path, "cs3"))
    assert payload["cancel_state"] == "pending"
    assert "cancel_reason" not in payload

    write_task_result(tmp_path, "cs2", "completed", result="done")
    request_cancel(tmp_path, "cs2")  # completion wins: nothing minted
    payload = public_task_result(load_effective_task_result(tmp_path, "cs2"))
    assert "cancel_state" not in payload
    assert "cancel_reason" not in payload


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


def test_connection_contract_family_is_additive_and_indexed():
    """Pin the RWS v2 connections contract family (Appendix C-3 #2/#3).

    The names are NEW, so nothing can break by renaming — what needs pinning is
    that the family stays ADDITIVE: the eight owner routes are in the endpoint
    index, the live WS envelope is registered, ``ExecutorRef`` keeps every legacy
    key while gaining the DERIVED ssh arm, and no durable-secret field ever
    appears in a wire row (the store holds no secrets; ``connection_store.py``
    owns the durable half).

    ``__required_keys__`` is NOT the authority here: ``contracts.py`` uses
    ``from __future__ import annotations``, so ``Required``/``NotRequired`` are
    invisible to TypedDict at class-creation time and only resolve through
    ``get_type_hints(..., include_extras=True)``.
    """

    from typing import Literal, Required, get_args, get_origin, get_type_hints

    from ouroboros.gateway import contracts

    def required_marked(cls) -> set[str]:
        hints = get_type_hints(cls, include_extras=True)
        return {name for name, hint in hints.items() if get_origin(hint) is Required}

    for endpoint in (
        "GET /api/owner/connections",
        "POST /api/owner/connections",
        "POST /api/owner/connections/{connection_id}/test",
        "POST /api/owner/connections/{connection_id}/bootstrap",
        "POST /api/owner/connections/{connection_id}/reconnect",
        "POST /api/owner/connections/{connection_id}/retrust",
        "GET /api/owner/connections/{connection_id}/dirs",
        "DELETE /api/owner/connections/{connection_id}",
    ):
        assert endpoint in contracts.HTTP_ENDPOINTS, endpoint
    assert "connection_state" in contracts.WS_MESSAGE_TYPES

    for name in (
        "ConnectionEntry",
        "ConnectionAddRequest",
        "ConnectionListResponse",
        "ConnectionActionResponse",
        "ConnectionDirsResponse",
        "ConnectionStateOutbound",
        "ProjectWorkspaceRef",
    ):
        assert name in contracts.__all__, name

    entry_keys = set(contracts.ConnectionEntry.__annotations__)
    # Durable identity/trust/lifecycle from the store + the bounded live
    # projection; the durable half never carries a secret.
    assert {
        "id", "name", "ssh_alias", "expected_host_id", "host_id_history",
        "lifecycle", "retired_at", "created_at", "updated_at",
        "status", "phase", "platform", "architecture", "build",
        "bootstrap_compatible", "health_fresh", "error_code", "action",
        "diagnostic", "log_refs",
    } <= entry_keys
    assert required_marked(contracts.ConnectionEntry) == {"id", "name", "ssh_alias"}
    assert not {
        "password", "passphrase", "private_key", "secret", "token",
        "ssh_options", "session",
    } & entry_keys
    assert contracts.ConnectionAddRequest.__required_keys__ == frozenset(
        {"name", "ssh_alias"}
    )
    assert required_marked(contracts.ConnectionStateOutbound) == {
        "type", "connection_id",
    }

    # ExecutorRef: every legacy key intact, ssh arm added as a DERIVED projection.
    executor_keys = set(contracts.ExecutorRef.__annotations__)
    assert {
        "type", "id", "network", "workspace_host_path", "workspace_backend_path",
        "container_name", "path_mappings",
    } <= executor_keys
    assert {"connection_id", "remote_root", "workspace_id"} <= executor_keys
    discriminator = get_type_hints(contracts.ExecutorRef, include_extras=True)["type"]
    assert get_origin(discriminator) is Required
    literal = get_args(discriminator)[0]
    assert get_origin(literal) is Literal
    assert set(get_args(literal)) == {"local", "docker_exec", "ssh"}

    # The wire placement mirror matches the persisted descriptor's payload keys.
    ref_keys = set(contracts.ProjectWorkspaceRef.__annotations__)
    assert {
        "kind", "local_root", "connection_id", "remote_root", "workspace_id",
    } <= ref_keys
    assert required_marked(contracts.ProjectWorkspaceRef) == {"kind"}

    # The live projection emits these on the wire (gateway/connections.py
    # ``_public_live_fields``), so the contract has to declare them or the
    # browser mirror below would be pinned to a shape the server does not send.
    assert "warnings" in entry_keys and "completion" in entry_keys
    assert "warnings" in set(contracts.ConnectionActionResponse.__annotations__)
    assert "warnings" in set(contracts.ConnectionStateOutbound.__annotations__)


def test_every_field_the_live_projection_emits_is_declared_and_mirrored():
    """Drive the PRODUCER, not a hand-written list of its fields.

    ``_public_live_fields`` is the single projection behind ``ConnectionEntry``'s live
    half, ``ConnectionActionResponse`` and the ``connection_state`` frame. A name-level
    parity test cannot see a new key it starts emitting, and the field-level mirror
    test below only compares Python to JavaScript — so a field could be produced,
    read by the browser, and declared by NEITHER side. That is how
    ``platform``/``architecture``/``build``/``bootstrap_compatible``/``health_fresh``
    came to be rendered by `connections_ui.js` off a WS frame that no contract said
    could carry them.
    """

    from ouroboros.gateway import contracts
    from ouroboros.gateway.connections import _public_live_fields

    emitted = set(_public_live_fields({
        "status": "degraded",
        "phase": "connect",
        "task_id": "task-1",
        "project_id": "project-1",
        "system": "Linux",
        "machine": "x86_64",
        "build": "execd-1",
        "completion": "not_started",
        "error_code": "SSH_EOF",
        "action": "retry",
        "bootstrap_compatible": True,
        "health_fresh": False,
        "diagnostic": {"code": "ssh_eof"},
        "log_refs": [{"stream": "stderr"}],
        "warnings": [{"code": "ssh_forwarding_neutralized"}],
    }))
    # `platform`/`architecture` are the normalized spellings of `system`/`machine`.
    assert {"platform", "architecture"} <= emitted
    # Every bounded list is emitted WITH its pre-cap total: a silent cap makes a
    # shortened list indistinguishable from a complete one.
    assert {"log_refs_count", "warnings_count"} <= emitted

    state_keys = set(get_type_hints(contracts.ConnectionStateOutbound, include_extras=True))
    action_keys = set(get_type_hints(contracts.ConnectionActionResponse, include_extras=True))
    entry_keys = set(get_type_hints(contracts.ConnectionEntry, include_extras=True))
    # The WS frame is this projection verbatim, so it must declare all of it.
    assert emitted <= state_keys, sorted(emitted - state_keys)
    # A connection ROW is a store row plus the projection over a broker status row:
    # those carry `project_id` (the project whose session it is) but never a task.
    assert emitted - {"task_id"} <= entry_keys, sorted(emitted - {"task_id"} - entry_keys)
    # An ACTION response is the service envelope plus Home's fields; the projection
    # reaches it through `_service_call`'s typed-exception arm, which is per-connection
    # and so names neither a task nor a project.
    assert emitted - {"task_id", "project_id"} <= action_keys, sorted(
        emitted - {"task_id", "project_id"} - action_keys
    )

    # Retrust rests on these, and they were undeclared: `host_id` and `handshake` are
    # the only way any surface learns the CURRENTLY observed host identity.
    assert {"host_id", "handshake"} <= action_keys


def test_project_placement_contracts_are_additive_and_mirrored():
    """Appendix C-3 #1/#7 for the OWNER's remote-project surface.

    The remote source is two flat fields (``connection_id`` + ``remote_root``) and
    NOT a serialized ``workspace_ref``: the third field of a placement is allocated
    by the target at admission, so a client able to name it could claim a workspace
    it never opened. This test pins that shape on both sides of the wire, and pins
    that the pre-RWS keys are all still there — a remote-placement field must never
    arrive by REPLACING part of the local create contract.

    ``__required_keys__`` is not the authority (``from __future__ import
    annotations`` hides Required/NotRequired from TypedDict); the pins go through
    ``get_type_hints(..., include_extras=True)``.

    BOUNDARY: the Python half is real type introspection, but the browser half is read as
    TEXT, so it holds only for the JSDoc spelling currently written. A reordered union, a
    reflowed ``@typedef``, an ``import()``-typed alias, or a field inherited from a base
    typedef the mirror extends are all invisible to it. What is closed is a field added on
    one side and forgotten on the other, in the spellings this codebase actually uses.
    """

    from typing import get_type_hints

    from ouroboros.gateway import contracts

    for name in ("ProjectCreateRequest", "ProjectUpdateRequest", "ProjectEntry"):
        assert name in contracts.__all__, name

    create_keys = set(get_type_hints(contracts.ProjectCreateRequest, include_extras=True))
    # Every pre-RWS source key survives, and the remote source is exactly two fields.
    assert {"id", "name", "path", "init_git", "git_url", "with_workspace"} <= create_keys
    assert {"connection_id", "remote_root"} <= create_keys
    assert "workspace_ref" not in create_keys and "placement" not in create_keys
    # Nothing is required: a name-only body is still a valid create.
    assert contracts.ProjectCreateRequest.__total__ is False

    update_keys = set(get_type_hints(contracts.ProjectUpdateRequest, include_extras=True))
    assert update_keys == {"name", "connection_id", "remote_root"}
    assert contracts.ProjectUpdateRequest.__total__ is False

    entry_keys = set(get_type_hints(contracts.ProjectEntry, include_extras=True))
    assert {
        "id", "name", "chat_id", "working_dir", "provenance", "clone_url", "trusted_at",
        "origin", "created_at", "last_active_at", "lifecycle", "routing_generation",
        "visible_revision", "delete_error",
    } <= entry_keys
    assert "placement" in entry_keys

    web = pathlib.Path(__file__).resolve().parent.parent / "web" / "modules"
    types_text = (web / "api_types.js").read_text(encoding="utf-8")
    for cls in (
        contracts.ProjectCreateRequest,
        contracts.ProjectUpdateRequest,
        contracts.ProjectEntry,
    ):
        expected = set(get_type_hints(cls, include_extras=True))
        actual = _js_typedef_fields(types_text, cls.__name__)
        assert actual == expected, (
            f"{cls.__name__} JSDoc mirror drifted: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    # The browser's placement field must be the ref mirror, not a loose object: the
    # sidebar renders `connection_id`/`remote_root` off it.
    assert re.search(r"@property \{\?ProjectWorkspaceRef=\} placement\b", types_text), (
        "ProjectEntry.placement must be mirrored as an optional ProjectWorkspaceRef"
    )
    # The rebind rides the existing update method (one endpoint, one client spelling).
    client_text = (web / "api_client.js").read_text(encoding="utf-8")
    assert "projectUpdate:" in client_text and "ProjectUpdateRequest" in client_text
    assert "projectCreate:" in client_text and "ProjectCreateRequest" in client_text


def test_connection_contracts_are_mirrored_for_the_browser():
    """Appendix C-3 #3/#4: the JSDoc mirrors are MANDATORY, not optional.

    The endpoint-index test only catches router↔contracts drift. Nothing there
    forces a Python-side field to reach ``web/modules/api_types.js``, so the
    connection family gets the same FIELD-level treatment the v6.80.0 contracts
    got: exact set equality between ``get_type_hints`` and the browser typedef.
    A new ConnectionEntry key now fails this test until the mirror is updated.
    """

    from typing import get_type_hints

    from ouroboros.gateway import contracts

    web = pathlib.Path(__file__).resolve().parent.parent / "web" / "modules"
    types_text = (web / "api_types.js").read_text(encoding="utf-8")
    for cls in (
        contracts.ConnectionEntry,
        contracts.ConnectionAddRequest,
        contracts.ConnectionListResponse,
        contracts.ConnectionActionResponse,
        contracts.ConnectionDirsResponse,
        contracts.ConnectionStateOutbound,
        contracts.ProjectWorkspaceRef,
        contracts.ExecutorRef,
    ):
        expected = set(get_type_hints(cls, include_extras=True))
        actual = _js_typedef_fields(types_text, cls.__name__)
        assert actual == expected, (
            f"{cls.__name__} JSDoc mirror drifted: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    # The ssh arm must be visible to the browser as a DERIVED third variant, and
    # the placement mirror must not offer a docker arm it can never carry.
    assert re.search(
        r'@property \{"local"\|"docker_exec"\|"ssh"\} type\b', types_text
    ), "ExecutorRef JSDoc must declare the ssh arm"
    assert re.search(
        r'@property \{"local"\|"ssh"\} kind\b', types_text
    ), "ProjectWorkspaceRef JSDoc must declare exactly the local|ssh discriminator"

    # Every owner connection route needs a named client method: a UI module that
    # hand-rolls fetch() bypasses the one browser-side gateway boundary.
    client_text = (web / "api_client.js").read_text(encoding="utf-8")
    for method in (
        "connections:",
        "connectionAdd:",
        "connectionTest:",
        "connectionBootstrap:",
        "connectionReconnect:",
        "connectionRetrust:",
        "connectionRetire:",
        "connectionDirs:",
        "ownerLogin:",
    ):
        assert method in client_text, f"api_client.js missing {method}"
    # The task-cancel endpoint has exactly ONE client spelling. It had two — the
    # exported `cancelTask(id, {cascade})` helper and an `apiClient.taskCancel(id)`
    # entry that could not express `cascade` at all — and the remote card reached for
    # the second, so cancelling a remote orchestrator orphaned its subagents while the
    # two other buttons for that same action cascaded. This list used to pin the
    # duplicate; it now forbids it.
    assert "export function cancelTask(" in client_text
    assert "taskCancel:" not in client_text, (
        "a second spelling of /api/tasks/{id}/cancel is how a caller loses `cascade`"
    )
    for route in (
        "/api/owner/connections",
        "/api/owner/connections/${encodeURIComponent(connectionId)}/test",
        "/api/owner/connections/${encodeURIComponent(connectionId)}/bootstrap",
        "/api/owner/connections/${encodeURIComponent(connectionId)}/reconnect",
        "/api/owner/connections/${encodeURIComponent(connectionId)}/retrust",
        "/api/owner/connections/${encodeURIComponent(connectionId)}/dirs",
    ):
        assert route in client_text, f"api_client.js missing route {route}"
    assert "method: 'DELETE'" in client_text, "connectionRetire must use DELETE"
