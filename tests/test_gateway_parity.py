from __future__ import annotations

import pathlib
import re
from typing import get_args, get_origin, get_type_hints

from ouroboros.gateway.contracts import (
    HTTP_ENDPOINTS,
    WS_MESSAGE_TYPES,
    ActiveChatActivity,
    ActiveDirectTurn,
    AvailableSubagentsSettingsMeta,
    ChatInbound,
    ChatOutbound,
    ClaudexorCredentialProfileDeleteResponse,
    ClaudexorLoginJobProblem,
    ClaudexorLoginJobResponse,
    ClaudexorStatusReads,
    ClaudexorStatusResponse,
    ClaudexorVendorCredentialDisposition,
    OnboardingCompleteRequest,
    OnboardingCompleteResponse,
    OnboardingPresetFailureResponse,
    OnboardingPresetProjection,
    OnboardingSubagentsPreviewResponse,
    OwnerHurryProjection,
    OwnerSkillPresenceRuntimeRequest,
    OwnerSkillPresenceRuntimeResponse,
    OwnerScopeReviewFloorResponse,
    DocumentOutbound,
    LinkAction,
    LinksOutbound,
    QuizOption,
    DecisionRequest,
    DecisionResponse,
    QuizOutbound,
    QuizStateOutbound,
    LogOutbound,
    PhotoOutbound,
    ProviderTestRequest,
    ProviderTestResponse,
    SettingsMeta,
    SettingsPostCommitFailureResponse,
    SkillDeleteResponse,
    SkillLifecycleQueueResponse,
    SkillPublishPreflightResponse,
    StateResponse,
    TaskCostBreakdown,
    TaskDetailResponse,
    TaskHurryRequest,
    TaskHurryResponse,
    TypingOutbound,
    UiPreferencesResponse,
    UpdateApplyErrorResponse,
    UpdateApplyRequest,
    UpdateApplySuccessResponse,
    UpdateMergePlan,
    UpdatePreflightRequest,
    UpdatePreflightResponse,
    UpdateStatusReadyOutbound,
    VideoOutbound,
)
from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.widgets import WidgetTab, WidgetsResponse


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
                    field = identifier.group(1)
                    assert field not in fields, f"{name} JSDoc declares duplicate property {field}"
                    fields.add(field)
                break
    return fields


def _contains_none(annotation) -> bool:
    return annotation is type(None) or any(_contains_none(arg) for arg in get_args(annotation))


def _notrequired_fields(cls) -> set[str]:
    """Read NotRequired markers robustly on supported Python 3.10 setups."""
    marked = set()
    for name, ann in cls.__annotations__.items():
        rendered = getattr(ann, "__forward_arg__", None) or str(ann)
        if rendered.startswith("NotRequired["):
            marked.add(name)
    return marked


def _js_typedef_properties(text: str, name: str) -> set[tuple[str, str]]:
    match = re.search(rf"@typedef \{{Object\}} {name}\b(?P<body>.*?)\n \*/", text, re.S)
    assert match, f"api_types.js missing {name}"
    return set(re.findall(
        r"^\s*\*\s+@property\s+\{([^}]*)\}\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
        match.group("body"),
        re.M,
    ))


def test_provider_test_gateway_contract_shape_is_exact():
    request_hints = get_type_hints(ProviderTestRequest, include_extras=True)
    response_hints = get_type_hints(ProviderTestResponse, include_extras=True)

    assert set(request_hints) == {"provider_id", "overrides"}
    assert _notrequired_fields(ProviderTestRequest) == {"overrides"}
    assert request_hints["provider_id"] is str
    overrides_type = get_args(request_hints["overrides"])[0]
    assert get_origin(overrides_type) is dict
    assert get_args(overrides_type) == (str, str)

    assert set(response_hints) == {"ok", "error"}
    assert _notrequired_fields(ProviderTestResponse) == {"error"}
    assert response_hints["ok"] is bool
    assert get_args(response_hints["error"]) == (str,)

    text = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    assert _js_typedef_properties(text, "ProviderTestRequest") == {
        ("string", "provider_id"),
        ("Object<string, string>=", "overrides"),
    }
    assert _js_typedef_properties(text, "ProviderTestResponse") == {
        ("boolean", "ok"),
        ("string=", "error"),
    }


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
    settings_meta_fields = {
        "custom_secret_keys", "setup_contract", "available_subagents",
    }
    assert settings_meta_fields <= set(SettingsMeta.__annotations__)
    assert _js_typedef_fields(text, "SettingsMeta") == settings_meta_fields
    for name in (
        "StateResponse",
        "ActiveDirectTurn",
        "ActiveChatActivity",
        "TypingOutbound",
        "HealthResponse",
        "SettingsMeta",
        "OpenAICompatibleModelsResponse",
        "ProviderTestRequest",
        "ProviderTestResponse",
        "UiPreferencesResponse",
        "ChatInbound",
        "ChatOutbound",
        "PhotoOutbound",
        "VideoOutbound",
        "DocumentOutbound",
        "LinkAction",
        "LinksOutbound",
        "QuizOption",
        "QuizOutbound",
        "QuizStateOutbound",
        "DecisionRequest",
        "DecisionResponse",
        "MessageAnnotationOutbound",
        "UploadResponse",
        "TaskCreateResponse",
        "TaskEvent",
        "TaskListResponse",
        "TaskCostBreakdown",
        "TaskDetailResponse",
        "TaskCancelResponse",
        "TaskHurryRequest",
        "TaskHurryResponse",
        "OwnerHurryProjection",
        "OwnerSkillPresenceRuntimeRequest",
        "OwnerSkillPresenceRuntimeResponse",
        "LogTailResponse",
        "SkillDeleteResponse",
        "SkillPublishPreflightResponse",
        "UpdateMergePlan",
        "UpdatePreflightRequest",
        "UpdatePreflightResponse",
        "UpdateApplyRequest",
        "UpdateApplySuccessResponse",
        "UpdateApplyErrorResponse",
        "UpdateStatusReadyOutbound",
        "AvailableSubagentsSettingsMeta",
        "OnboardingCompleteRequest",
        "OnboardingSubagentsPreviewResponse",
        "OnboardingPresetProjection",
        "OnboardingCompleteResponse",
        "OnboardingPresetFailureResponse",
        "SettingsPostCommitFailureResponse",
        "ClaudexorLoginJobResponse",
        "ClaudexorLoginJobProblem",
        "ClaudexorCredentialProfileDeleteResponse",
        "ClaudexorVendorCredentialDisposition",
        "WidgetTab",
        "WidgetsResponse",
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
    for cls in (ChatInbound, ChatOutbound, PhotoOutbound, VideoOutbound, DocumentOutbound,
                DecisionRequest, DecisionResponse, LinkAction, LinksOutbound, QuizOption, QuizOutbound, QuizStateOutbound,
                ActiveDirectTurn, ActiveChatActivity, TypingOutbound,
                StateResponse, OwnerScopeReviewFloorResponse, UpdateMergePlan,
                UpdatePreflightRequest, UpdatePreflightResponse, UpdateApplyRequest,
                UpdateApplySuccessResponse, UpdateApplyErrorResponse,
                UpdateStatusReadyOutbound, TaskCostBreakdown, TaskDetailResponse,
                TaskHurryRequest, TaskHurryResponse, OwnerHurryProjection,
                OwnerSkillPresenceRuntimeRequest, OwnerSkillPresenceRuntimeResponse,
                OnboardingCompleteRequest, OnboardingPresetProjection,
                OnboardingSubagentsPreviewResponse, OnboardingCompleteResponse,
                OnboardingPresetFailureResponse, AvailableSubagentsSettingsMeta,
                SettingsPostCommitFailureResponse,
                ProviderTestRequest, ProviderTestResponse,
                SkillPublishPreflightResponse,
                ClaudexorLoginJobResponse, ClaudexorLoginJobProblem,
                ClaudexorCredentialProfileDeleteResponse,
                ClaudexorVendorCredentialDisposition,
                ClaudexorStatusReads, ClaudexorStatusResponse,
                WidgetTab, WidgetsResponse,
                # widgets-lifecycle W1b: the owner's per-card start-mode override is checked field by field.
                UiPreferencesResponse):
        expected = set(get_type_hints(cls, include_extras=True))
        actual = _js_typedef_fields(text, cls.__name__)
        assert actual == expected, f"{cls.__name__} JSDoc fields drifted: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    # Field-set parity alone would accept an optional marker on the two
    # discriminators. Pin the browser mirror's requiredness as well as names.
    success_decl = re.search(
        r"@typedef \{Object\} ClaudexorLoginJobResponse\b([\s\S]*?)\*/", text)
    problem_decl = re.search(
        r"@typedef \{Object\} ClaudexorLoginJobProblem\b([\s\S]*?)\*/", text)
    assert success_decl and "@property {Object} job" in success_decl.group(1)
    assert problem_decl and "@property {string} error" in problem_decl.group(1)
    delete_decl = re.search(
        r"@typedef \{Object\} ClaudexorCredentialProfileDeleteResponse\b([\s\S]*?)\*/",
        text,
    )
    assert delete_decl
    for declaration in (
        "@property {Object} profile",
        "@property {boolean} removed",
        "@property {('config_dir_removed'|'secret_deleted'|'none')} credentialCleanup",
    ):
        assert declaration in delete_decl.group(1)
    delete_optional = {
        name for name, annotation
        in ClaudexorCredentialProfileDeleteResponse.__annotations__.items()
        if (getattr(annotation, "__forward_arg__", None) or str(annotation)).startswith(
            "NotRequired["
        )
    }
    assert delete_optional == {"cleanupWarning", "vendorCredentialDisposition"}
    assert set(ClaudexorCredentialProfileDeleteResponse.__annotations__) - delete_optional == {
        "profile", "removed", "credentialCleanup",
    }
    delete_hints = get_type_hints(
        ClaudexorCredentialProfileDeleteResponse, include_extras=True)
    assert set(get_args(delete_hints["credentialCleanup"])) == {
        "config_dir_removed", "secret_deleted", "none",
    }
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
    # In-flight turn ABI: field-set parity alone would accept requiredness
    # drift. Snapshot rows always emit every field (required); typing frames
    # stamp the typed fields only for registry-tracked turns (optional).
    # (__required_keys__ ignores NotRequired on some 3.10 setups, so inspect
    # the declared annotations instead.)
    assert _notrequired_fields(ActiveDirectTurn) == set(), (
        "ActiveDirectTurn snapshot rows always emit every field: keep them all required"
    )
    assert _notrequired_fields(ActiveChatActivity) == set(), (
        "ActiveChatActivity snapshot rows always emit every field: keep them all required"
    )
    assert ActiveChatActivity.__annotations__.keys() == ActiveDirectTurn.__annotations__.keys(), (
        "ActiveChatActivity must mirror ActiveDirectTurn's field shape so one client reducer hydrates both"
    )
    assert _notrequired_fields(TypingOutbound) == {
        "chat_id", "activity_id", "client_message_id", "phase", "kind",
        "project_thread",
    }, "TypingOutbound typed fields are stamped only for registry-tracked turns: keep them optional"
    turn_decl = re.search(r"@typedef \{Object\} ActiveDirectTurn\b([\s\S]*?)\*/", text)
    assert turn_decl and not re.search(r"@property \{[^}]*=\}", turn_decl.group(1)), (
        "ActiveDirectTurn browser mirror must declare every field required"
    )
    typing_decl = re.search(r"@typedef \{Object\} TypingOutbound\b([\s\S]*?)\*/", text)
    assert typing_decl
    for optional_field in ("chat_id", "activity_id", "client_message_id", "phase", "kind"):
        assert re.search(
            rf"@property \{{[^}}]*=\}} {optional_field}\b", typing_decl.group(1)
        ), f"TypingOutbound.{optional_field} must stay optional in the browser mirror"
    assert re.search(r"@property \{'auto_merge'\|'assisted'\|'manual'\|'replace'\} strategy\b", text)
    assert re.search(r"@property \{string=\} expected_base_sha\b", text)
    assert re.search(r"@property \{string=\} expected_target_sha\b", text)
    assert re.search(r"@property \{boolean=\} confirm_recovery\b", text)
    assert re.search(r"@property \{'ok'\|'restart_required'\|'assisted_started'\|'manual'\} status\b", text)
    assert re.search(r"@typedef \{Object\} UpdateApplyErrorResponse.*?@property \{string\} error\b", text, re.S)
    assert re.search(r"@property \{boolean\} context_mode_auto_low\b", text), (
        "StateResponse.context_mode_auto_low must remain a JSDoc boolean compatibility field"
    )
    assert re.search(r"@property \{string\} deprecation_notice\b", text), (
        "OwnerScopeReviewFloorResponse.deprecation_notice must be declared for the browser"
    )
    assert re.search(r"@property \{boolean=\} force_plan\b", text), "ChatInbound missing force_plan"
    assert re.search(r"@property \{Object=\} client_surface\b", text), "ChatInbound missing client_surface"
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
    # Main-thread fan-out stamp: every card/bubble-MINTING outbound frame family
    # declares the same additive-optional boolean in both mirrors (message_annotation
    # is thread-routed but no-ops on unknown ids and stays unstamped by design).
    for cls in (ChatOutbound, PhotoOutbound, VideoOutbound, DocumentOutbound, LinksOutbound, QuizOutbound, LogOutbound, TypingOutbound):
        assert "project_thread" in get_type_hints(cls, include_extras=True), f"{cls.__name__} missing project_thread"
    assert len(re.findall(r"@property \{boolean=\} project_thread\b", text)) >= 6, "api_types.js missing project_thread mirrors"
    assert re.search(r"@property \{boolean=\} worker_saturation_warning\b", text), "ChatOutbound missing worker_saturation_warning"
    assert "review_projection" in get_type_hints(ChatOutbound, include_extras=True)
    assert re.search(r"@property \{Object=\} review_projection\b", text)
    assert "setup_contract" in text
    assert re.search(r"@property \{string=\} error\b", text), "SkillDeleteResponse missing optional error"
    assert {"chat", "command", "photo", "video", "links", "quiz", "quiz_state", "typing", "log", "heartbeat", "extension_lifecycle"} <= set(WS_MESSAGE_TYPES)
    assert "message_annotation" in WS_MESSAGE_TYPES
    assert "update_status_ready" in WS_MESSAGE_TYPES
    assert _js_typedef_fields(text, "MessageAnnotationOutbound") == {
        "type",
        "annotation_type",
        "chat_id",
        "client_message_id",
        "action",
        "target",
        "target_label",
        "routing_token",
        "status",
        "options",
        "attachment_manifest",
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


def test_max_quiz_options_pinned_across_python_and_js():
    """The quiz-option cap is one number in both languages: the shared
    validator gate (ouroboros.tools.core._MAX_QUIZ_OPTIONS) and the UI
    contract mirror (web/modules/api_types.js MAX_QUIZ_OPTIONS)."""
    from ouroboros.tools.core import _MAX_QUIZ_OPTIONS

    text = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^export const MAX_QUIZ_OPTIONS = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_QUIZ_OPTIONS"
    assert int(match.group(1)) == _MAX_QUIZ_OPTIONS


def test_decision_comment_limit_and_optional_index_pinned_across_python_and_js():
    """The verbatim-comment cap is ONE number in both languages (the ingress
    refuses a longer comment instead of truncating it, so the card must not
    offer to send one), and the browser mirror must declare option_index as
    OPTIONAL — the field-set parity loop cannot see requiredness, and an
    owner's own answer (no option taken) carries no index at all."""
    from ouroboros.gateway.contracts import DecisionRequest
    from ouroboros.gateway.task_decision import _COMMENT_MAX

    text = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^export const MAX_DECISION_COMMENT = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_DECISION_COMMENT"
    assert int(match.group(1)) == _COMMENT_MAX

    optional = {
        name for name, annotation in DecisionRequest.__annotations__.items()
        if (getattr(annotation, "__forward_arg__", None) or str(annotation)).startswith("NotRequired[")
    }
    assert optional == {"option_index", "comment"}
    request_decl = re.search(r"@typedef \{Object\} DecisionRequest\b([\s\S]*?)\*/", text)
    assert request_decl
    assert "@property {number=} option_index" in request_decl.group(1)
    assert "@property {string=} comment" in request_decl.group(1)

    # The settled quiz card replayed from history carries the recorded answer:
    # the chosen index when there was one, and the owner's verbatim words —
    # which, with no index, ARE the answer.
    quiz_decl = re.search(r"@typedef \{Object\} QuizOutbound\b([\s\S]*?)\*/", text)
    assert quiz_decl
    assert "@property {number=} answered_index" in quiz_decl.group(1)
    assert "@property {string=} comment" in quiz_decl.group(1)


def test_max_link_actions_pinned_across_python_and_js():
    """The links-action cap is one number in both languages: the tool gate
    (ouroboros.tools.core._MAX_LINK_ACTIONS) and the UI contract mirror
    (web/modules/api_types.js MAX_LINK_ACTIONS) must never drift apart."""
    from ouroboros.tools.core import _MAX_LINK_ACTIONS

    text = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^export const MAX_LINK_ACTIONS = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_LINK_ACTIONS"
    assert int(match.group(1)) == _MAX_LINK_ACTIONS
