"""Checks for the Python gateway contract and its browser consumers.

The Python TypedDicts are the one statement of the wire contract; the browser
carries no JSDoc copy of them. What this file pins:

- every wire guarantee the old mirror used to carry, read from the Python
  contract itself (fields, requiredness, literal sets, nested shapes);
- the release-version carrier in ``web/modules/api_types.js`` (RES-15a: it is
  the RELEASE version, not the ABI version) and the three UI limits that live
  modules import — that file's surviving byte-exact constants;
- no same-name, field-identical wire copies in the shared browser type carriers.
  A local JSDoc name is not an API claim.
"""

from __future__ import annotations

import pathlib
import re
from typing import get_args, get_origin, get_type_hints

from ouroboros.gateway.contracts import (
    HTTP_ENDPOINTS,
    WS_MESSAGE_TYPES,
    ActiveChatActivity,
    ActiveDirectTurn,
    ChatInbound,
    ChatOutbound,
    ClaudexorCredentialProfileDeleteResponse,
    ClaudexorLoginJobProblem,
    ClaudexorLoginJobResponse,
    ClaudexorPassiveReadError,
    ClaudexorQuotaResponse,
    ClaudexorStatusReads,
    ClaudexorStatusResponse,
    ContinuationOffer,
    DecisionRequest,
    DecisionResponse,
    DocumentOutbound,
    LinksOutbound,
    LogOutbound,
    MessageAnnotationOutbound,
    PhotoOutbound,
    ProviderTestRequest,
    QuizOption,
    QuizOutbound,
    QuizStateOutbound,
    RequestedExecutorRoute,
    SettingsMeta,
    SkillDeleteResponse,
    SkillLifecycleQueueResponse,
    StateResponse,
    TaskCostBreakdown,
    TaskContinueRequest,
    TaskContinueResponse,
    TaskDetailResponse,
    TaskEvent,
    TaskPauseRequest,
    TaskPauseResponse,
    TypingOutbound,
    UiPreferencesResponse,
    UpdatePreflightResponse,
    VideoOutbound,
)
from ouroboros.gateway.router import collect_routes

REPO = pathlib.Path(__file__).resolve().parents[1]
API_TYPES = REPO / "web" / "modules" / "api_types.js"


def _js_typedef_fields(body: str) -> set[str]:
    # Types nest braces (``{Object<string, {project_id: string}>}``), so scan for the BALANCED
    # closing brace instead of the first one — a non-greedy ``[^}]+`` silently mis-parses those
    # properties and makes the field set look like it drifted when it has not.
    fields: set[str] = set()
    for line in body.splitlines():
        _, sep, rest = line.partition("@property {")
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


def _notrequired_fields(cls) -> set[str]:
    """Read NotRequired markers robustly on supported Python 3.10 setups."""
    marked = set()
    for name, ann in cls.__annotations__.items():
        rendered = getattr(ann, "__forward_arg__", None) or str(ann)
        if rendered.startswith("NotRequired["):
            marked.add(name)
    return marked


# ---------------------------------------------------------------------------
# Shared browser JSDoc must not repeat a Python wire type's name and fields.
# ---------------------------------------------------------------------------

def _copied_wire_shapes(text: str) -> set[str]:
    from ouroboros.gateway.schema import contract_typed_dict
    from ouroboros.physical_attempt_context import ContextAttemptFact

    copies = set()
    for name, body in re.findall(r"@typedef \{Object\} (\w+)\b(.*?)\*/", text, re.S):
        try:
            cls = ContextAttemptFact if name == "ContextAttemptFact" else contract_typed_dict(name)
        except KeyError:
            continue
        fields = _js_typedef_fields(body)
        if fields and fields == set(get_type_hints(cls, include_extras=True)):
            copies.add(name)
    return copies


def test_shared_browser_types_do_not_copy_gateway_wire_shapes():
    """The four shared carriers keep UI shapes, not a second declaration of the API."""
    for carrier in ("api_types.js", "client_contract_types.js", "place_contract_types.js", "task_activity_types.js"):
        copies = _copied_wire_shapes(API_TYPES.with_name(carrier).read_text(encoding="utf-8"))
        assert not copies, f"{carrier} repeats gateway wire shapes: {sorted(copies)}"


def test_a_shared_name_only_does_not_make_a_ui_type_a_wire_copy():
    start = "/** @typedef {Object} AuthLoginResponse\n * @property {boolean} ok\n"
    assert not _copied_wire_shapes(start + " * @property {string} viewLabel\n */")
    assert _copied_wire_shapes(start + " * @property {string} next\n */") == {"AuthLoginResponse"}


def test_release_version_carrier_stays_pinned_to_the_version_file():
    """``api_types.js`` keeps the RELEASE version carrier (RES-15a: it is not
    the ABI version). ``ouroboros.tools.release_sync`` rewrites it beside
    ``pyproject.toml`` and ``web/package.json``; this pin is the carrier
    check the parity role has always carried."""
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    text = API_TYPES.read_text(encoding="utf-8")
    assert f"GATEWAY_CONTRACT_VERSION = '{version}'" in text


# ---------------------------------------------------------------------------
# The wire guarantees the old mirror carried, read from the Python contract.
# ---------------------------------------------------------------------------

def test_requested_executor_route_schema_is_derived_from_the_single_source():
    from ouroboros.gateway.schema import json_schema_for

    schema = json_schema_for(ChatOutbound)
    field = "requested_executor_route"
    route = schema["properties"][field]
    assert field not in schema["required"]
    assert route == json_schema_for(RequestedExecutorRoute)
    assert route["type"] == "object"
    assert set(route["required"]) == {"kind", "target_id"}
    kinds = ["api_model", "agent_session"]
    assert route["properties"] == {"kind": {"enum": kinds}, "target_id": {"type": "string"}}

    # GET /api/tasks/{id} serves the same frozen selection for a card that missed its terminal frame (settled_card.js).
    detail = json_schema_for(TaskDetailResponse)
    assert field not in detail.get("required", []) and detail["properties"][field] == route


def test_provider_test_gateway_contract_shape_is_exact():
    """The published provider-test schema keeps the frozen ABI (ARCHITECTURE §11.1): the
    request the JS client sends and the response the runtime serves — the runtime half is
    tests/test_provider_key_test.py, the browser half web/tests/provider_test.test.js — and
    the route publishes exactly these shapes."""
    from ouroboros.gateway.schema import contract_document

    document = contract_document()
    route = next(row for row in document["http"]
                 if row["method"] == "POST" and row["path"] == "/api/providers/test")
    assert route["request"]["schemas"] == ["ProviderTestRequest"]
    assert route["responses"]["200"]["schemas"] == ["ProviderTestResponse"]

    request = document["schemas"]["ProviderTestRequest"]
    response = document["schemas"]["ProviderTestResponse"]
    assert request["type"] == response["type"] == "object"

    assert request["required"] == ["provider_id"]
    assert request["properties"] == {
        "provider_id": {"type": "string"},
        "overrides": {"type": "object"},
        "connection": {"type": "string"},
        "api_key_setting": {"enum": ["OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY"]},
    }
    # The published schema represents mappings as objects; Python also pins the flat string values.
    request_hints = get_type_hints(ProviderTestRequest, include_extras=True)
    overrides_type = get_args(request_hints["overrides"])[0]
    assert get_origin(overrides_type) is dict
    assert get_args(overrides_type) == (str, str)

    assert response["required"] == ["ok"]
    assert response["properties"] == {
        "ok": {"type": "boolean"},
        "error": {"type": "string"},
        "caution": {"type": "string"},
        "code": {"enum": ["ollama_guard_consent", "ollama_guard_held", "ollama_guard_unavailable"]},
    }


def test_claudexor_quota_view_has_separate_required_contract():
    hints = get_type_hints(ClaudexorQuotaResponse, include_extras=True)
    assert set(hints) == {
        "view", "profiles", "quota", "quota_absences", "unified_accounts",
        "reads", "read_errors", "timings_ms",
        "resources", "availability_scope",
    }
    assert ClaudexorQuotaResponse.__total__
    assert _notrequired_fields(ClaudexorQuotaResponse) == {"resources", "availability_scope"}
    assert get_args(hints["view"]) == ("quota",)
    assert hints["reads"] is ClaudexorStatusReads
    assert hints["unified_accounts"] is bool
    phases, error_type = get_args(hints["read_errors"])
    assert set(get_args(phases)) == {"discovery", "accounts", "quota"}
    assert error_type is ClaudexorPassiveReadError
    assert get_args(hints["timings_ms"]) == (str, int)
    errors = get_type_hints(ClaudexorPassiveReadError, include_extras=True)
    assert errors["code"] is str
    assert get_args(errors["status_code"]) == (int,)
    assert _notrequired_fields(ClaudexorPassiveReadError) == {"status_code"}
    assert not {"view", "read_errors", "timings_ms"} & set(ClaudexorStatusResponse.__annotations__)


def test_full_claudexor_status_scope_has_optional_typed_fields():
    hints = get_type_hints(ClaudexorStatusResponse, include_extras=True)
    assert get_origin(hints["availability_scope"]) is dict
    assert not ClaudexorStatusResponse.__total__


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

    settings_meta_fields = {
        "custom_secret_keys", "setup_contract", "available_subagents", "policy_state", "restart_state",
        "connections", "main_connection_choice", "known_spend_gaps", "partly_counted_spend", "effects", "tokens",
    }
    assert settings_meta_fields <= set(SettingsMeta.__annotations__)
    api_client = (REPO / "web" / "modules" / "api_client.js").read_text(encoding="utf-8")
    assert "openAICompatibleModels" in api_client
    # D-8: the wizard's ONE atomic completion call must exist on the browser client.
    assert "completeOnboarding" in api_client
    assert "'/api/onboarding/complete'" in api_client

    assert UpdatePreflightResponse.__required_keys__ == frozenset({"merge_plan"})

    # ABI 7.0 (ABI-3) removal pins: the retired aliases never come back to the
    # contract. The checks above cover field-identical browser copies.
    for cls, removed in (
        (ChatOutbound, ("cost_usd", "cost_usd_with_children", "telegram_chat_id")),
        (PhotoOutbound, ("telegram_chat_id",)),
        (VideoOutbound, ("telegram_chat_id",)),
        (DocumentOutbound, ("telegram_chat_id",)),
        (UiPreferencesResponse, ("project_last_viewed", "project_hidden")),
    ):
        hints = set(get_type_hints(cls, include_extras=True))
        for field in removed:
            assert field not in hints, f"{cls.__name__}.{field} resurrected in Python"

    # Requiredness the old mirror pinned on the JSDoc side, read from the contract.
    assert "job" in get_type_hints(ClaudexorLoginJobResponse, include_extras=True)
    assert "error" in get_type_hints(ClaudexorLoginJobProblem, include_extras=True)
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
    # verdict over data the daemon had just handed over. There is no browser
    # typedef to compare against: this check compares the store's exported list
    # directly with the Python TypedDict.
    store_js = (REPO / "web" / "modules" / "claudexor_status_store.js").read_text(encoding="utf-8")
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

    # In-flight turn ABI: snapshot rows always emit the base fields (required);
    # typing frames stamp the typed fields only for registry-tracked turns
    # (optional). (__required_keys__ ignores NotRequired on some 3.10 setups,
    # so inspect the declared annotations instead.)
    assert _notrequired_fields(ActiveDirectTurn) == {"model_waits", "task_attempt", "model_activity", "turn_phase",
                                                   "executor", "carrier", "context_attempt", "lead", "processing"}, (
        "ActiveDirectTurn keeps its required base; waits, attempt, start phase, the lane's carrier facts, the "
        "published lead and its tier are optional"
    )
    assert _notrequired_fields(ActiveChatActivity) == {
        "model_waits", "model_activity", "context_attempt", "task_attempt", "turn_phase", "executor", "carrier", "required_question", "required_question_unavailable", "project_admission_hold", "pause_cause", "owner_wait",
        "status", "outcome_axes", "reason_code", "cancel_origin", "delegated_runs_unreconciled", "root_phase_checkpoint",
        "timeout_retry_from", "original_task_id", "finalization_recovery", "terminal_origin",
        "finishing_reviews", "late_work", "can_resume", "close_expired", "cancelable", "presence_pause",
        "last_line", "last_line_at", "tool_calls", "detached", "place_origin", "project_name", "lead", "processing",
    }, "ActiveChatActivity adds optional outcome/retry/review evidence without changing required activity facts"
    from ouroboros.physical_attempt_context import ContextAttemptFact

    activity_fields = get_type_hints(ActiveChatActivity, include_extras=True)
    question_keys = {"required_question", "required_question_unavailable", "project_admission_hold", "pause_cause", "owner_wait"}
    question_keys |= {"status", "outcome_axes", "reason_code", "cancel_origin", "delegated_runs_unreconciled", "root_phase_checkpoint",
                      "timeout_retry_from", "original_task_id", "finalization_recovery", "terminal_origin"}
    question_keys |= {"finishing_reviews", "late_work", "can_resume", "close_expired", "cancelable", "presence_pause"}
    question_keys |= {"last_line", "last_line_at", "tool_calls", "detached", "place_origin", "project_name"}
    assert {key: value for key, value in activity_fields.items() if key not in question_keys} == get_type_hints(ActiveDirectTurn, include_extras=True), (
        "ActiveChatActivity must mirror ActiveDirectTurn's field shape so one client reducer hydrates both"
    )
    from ouroboros.gateway.schema import json_schema_for

    activity_schema = json_schema_for(ActiveChatActivity)
    context_schema = activity_schema["properties"]["context_attempt"]
    assert set(context_schema["properties"]["previous"]["properties"]) == set(ContextAttemptFact.__annotations__) - {"previous"}
    assert activity_schema["properties"].pop("required_question")["type"] == "object"
    assert activity_schema["properties"].pop("required_question_unavailable")["type"] == "boolean"
    assert activity_schema["properties"].pop("project_admission_hold")["type"] == "object"
    assert activity_schema["properties"].pop("pause_cause")["type"] == "string"
    assert activity_schema["properties"].pop("owner_wait")["type"] == "object"
    for field in ("can_resume", "cancelable", "presence_pause"):
        assert activity_schema["properties"].pop(field)["type"] == "boolean"
    for field in ("status", "reason_code", "timeout_retry_from", "original_task_id", "terminal_origin"):
        assert activity_schema["properties"].pop(field)["type"] == "string"
    for field in ("outcome_axes", "cancel_origin", "finalization_recovery", "root_phase_checkpoint", "late_work", "close_expired"):
        assert activity_schema["properties"].pop(field)["type"] == "object"
    assert activity_schema["properties"].pop("finishing_reviews")["type"] == "boolean"
    assert activity_schema["properties"].pop("delegated_runs_unreconciled") == {"type": "array", "items": {"type": "string"}}
    for field, kind in {"last_line": "string", "last_line_at": "string", "tool_calls": "integer",
                        "detached": "boolean", "place_origin": "object", "project_name": "string"}.items():
        assert activity_schema["properties"].pop(field)["type"] == kind
    assert not question_keys & set(activity_schema["required"])
    assert activity_schema == json_schema_for(ActiveDirectTurn), (
        "the shared activity shape must preserve flat keys, types and requiredness"
    )
    assert _notrequired_fields(TypingOutbound) == {
        "chat_id", "activity_id", "client_message_id", "phase", "kind",
        "project_thread",
    }, "TypingOutbound typed fields are stamped only for registry-tracked turns: keep them optional"

    # Field pins the old mirror carried on the JSDoc side, read from the contract.
    assert {"force_plan", "client_surface"} <= set(get_type_hints(ChatInbound, include_extras=True))
    chat_hints = get_type_hints(ChatOutbound, include_extras=True)
    assert {"model_lane", "requested_model_lane", "effective_model_lane", "model", "task_group_id"} <= set(chat_hints)
    assert {"subagent_event", "subagent_task_id", "root_task_id", "parent_task_id",
            "delegation_role", "subagent_role", "task_event", "status", "result",
            "trace_summary", "error", "artifact_status"} <= set(chat_hints)
    assert {"source", "line", "root"} <= set(get_type_hints(TaskEvent, include_extras=True))
    assert "context_mode_auto_low" in get_type_hints(StateResponse, include_extras=True), (
        "StateResponse.context_mode_auto_low stays the frozen compatibility field"
    )
    # The HOST names where a task-keyed System row belongs inside the task's card,
    # so the browser reads one typed fact instead of keeping its own list of
    # system types. Pin the literal set on the contract.
    card_row_hint = get_type_hints(ChatOutbound, include_extras=True)["card_row"]
    assert get_args(get_args(card_row_hint)[0]) == ("timeline", "reviews")
    # Main-thread fan-out stamp: every card/bubble-MINTING outbound frame family
    # declares the same additive-optional boolean (message_annotation is
    # thread-routed but no-ops on unknown ids and stays unstamped by design).
    for cls in (ChatOutbound, PhotoOutbound, VideoOutbound, DocumentOutbound, LinksOutbound, QuizOutbound, LogOutbound, TypingOutbound):
        assert "project_thread" in get_type_hints(cls, include_extras=True), f"{cls.__name__} missing project_thread"
    assert "worker_saturation_warning" in get_type_hints(ChatOutbound, include_extras=True)
    assert "review_projection" in get_type_hints(ChatOutbound, include_extras=True)
    assert "review_overrides" in get_type_hints(ChatOutbound, include_extras=True)  # the work's card field, on page load
    # The host stamps the exact physical row of the annotated message so the
    # browser can put the receipt on THAT message. The field is additive-optional
    # (frames without it stay valid) and a string on the contract.
    assert "source_row_address" in _notrequired_fields(MessageAnnotationOutbound), (
        "MessageAnnotationOutbound.source_row_address must stay optional"
    )
    source_row_annotation = MessageAnnotationOutbound.__annotations__["source_row_address"]
    source_row_rendered = getattr(source_row_annotation, "__forward_arg__", None) or str(source_row_annotation)
    assert "NotRequired[str]" in source_row_rendered, source_row_rendered
    # The notice carrier is one literal pair on the contract: an ACK and the
    # System row beside it read the same words.
    for cls in (MessageAnnotationOutbound, ChatOutbound):
        carrier = get_type_hints(cls, include_extras=True)["routing_notice_carrier"]
        assert get_args(get_args(carrier)[0]) == ("routing_ack", "system"), cls.__name__
    assert {"chat", "command", "photo", "video", "links", "quiz", "quiz_state", "typing", "log", "heartbeat", "extension_lifecycle"} <= set(WS_MESSAGE_TYPES)
    assert "message_annotation" in WS_MESSAGE_TYPES
    assert "update_status_ready" in WS_MESSAGE_TYPES


def test_gateway_money_contracts_keep_unavailable_distinct_from_zero():
    from ouroboros.gateway.contracts import StateResponse

    state_hints = get_type_hints(StateResponse, include_extras=True)
    for field in ("spent_usd", "budget_pct", "spent_calls"):
        assert _contains_none(state_hints[field]), f"StateResponse.{field} must admit ledger-unavailable null"

    chat_hints = get_type_hints(ChatOutbound, include_extras=True)
    for field in (
        "accounted_upper_bound_usd",
        "accounted_upper_bound_usd_with_children",
        "reserved_usd",
        "unresolved_upper_bound_usd",
        "unknown_unmetered",
    ):
        assert _contains_none(chat_hints[field]), f"ChatOutbound.{field} must admit ledger-unavailable null"
    assert {"cost_accounting_status", "cost_final", "cost_with_children_partial"} <= set(chat_hints)
    # ABI 7.0 (ABI-3): the retired aliases never come back to the contract.
    assert {"cost_usd", "cost_usd_with_children", "telegram_chat_id"}.isdisjoint(chat_hints)


def test_skill_lifecycle_queue_contract_matches_runtime_shape():
    """The published lifecycle-queue schema stays the runtime's answer (ABI 7.0 retired
    ``queue``/``recent_events``/``running``); the served route is bound to this shape in
    tests/test_client_contract_bindings.py."""
    from ouroboros.gateway.schema import json_schema_for

    schema = json_schema_for(SkillLifecycleQueueResponse)

    assert schema["required"] == ["active", "events"]
    assert {"queue", "recent_events", "running"}.isdisjoint(schema["properties"]), (
        "the ABI 7.0 retirement must not come back to the published schema")


def test_skill_delete_contract_matches_runtime_shape():
    """The published skill-delete answer keeps its runtime fields (``state_delete_pending``
    retired with the lane split); the delete routes publish this shape
    (tests/test_client_contract_coverage.py)."""
    from ouroboros.gateway.schema import json_schema_for

    schema = json_schema_for(SkillDeleteResponse)

    assert "state_delete_pending" not in schema["properties"]
    assert {
        "ok",
        "skill",
        "source",
        "deleted_payload_root",
        "deleted_state",
        "extension_action",
        "extension_reason",
        "error",
    } <= set(schema["properties"])


def test_v682_cancellation_contract_fields_are_pinned_in_python():
    """The additive cancellation ABI (v6.82 + phase A) is pinned on the published schemas:
    the host-attested ``cancelable`` marker, the cancel endpoint's ``cascade`` echo, and the
    phase-A ``cancel_state`` pending projection on the task detail envelope (AR2-8; the
    runtime projection is pinned below)."""
    from ouroboros.gateway.contracts import TaskCancelResponse
    from ouroboros.gateway.schema import json_schema_for

    chat = json_schema_for(ChatOutbound)
    detail = json_schema_for(TaskDetailResponse)
    cancel = json_schema_for(TaskCancelResponse)

    assert chat["properties"]["cancelable"] == {"type": "boolean"}
    assert "cancelable" not in chat.get("required", []), "the marker stays additive-optional"
    assert cancel["properties"]["cascade"] == {"type": "boolean"}
    assert "cascade" not in cancel.get("required", []), "the echo stays additive-optional"
    assert cancel["properties"]["cancel_state"] == {"type": "string"}
    # GR2-11: the intent's reason is public beside the state.
    assert detail["properties"]["cancel_state"] == {"type": "string"}
    assert detail["properties"]["cancel_reason"] == {"type": "string"}


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
    set TaskCostBreakdown declares, stay optional on the detail response, and be
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
    validator gate (ouroboros.tools.core_artifacts._MAX_QUIZ_OPTIONS) and the UI
    constant carrier (web/modules/api_types.js MAX_QUIZ_OPTIONS)."""
    from ouroboros.tools.core_artifacts import _MAX_QUIZ_OPTIONS

    text = API_TYPES.read_text(encoding="utf-8")
    match = re.search(r"^export const MAX_QUIZ_OPTIONS = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_QUIZ_OPTIONS"
    assert int(match.group(1)) == _MAX_QUIZ_OPTIONS


def test_decision_comment_limit_and_optional_index_pinned_across_python_and_js():
    """The verbatim-comment cap is ONE number in both languages (the ingress
    refuses a longer comment instead of truncating it, so the card must not
    offer to send one — api_types.js keeps the constant), and the request
    contract must declare option_index as OPTIONAL: an owner's own answer
    (no option taken) carries no index at all."""
    from ouroboros.gateway.task_decision import _COMMENT_MAX

    text = API_TYPES.read_text(encoding="utf-8")
    match = re.search(r"^export const MAX_DECISION_COMMENT = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_DECISION_COMMENT"
    assert int(match.group(1)) == _COMMENT_MAX

    optional = {
        name for name, annotation in DecisionRequest.__annotations__.items()
        if (getattr(annotation, "__forward_arg__", None) or str(annotation)).startswith("NotRequired[")
    }
    assert optional == {"option_index", "comment", "reason", "revision", "action", "auto_continue",
                        "model", "credential_profile_id", "use_local", "persist_role", "values"}

    # The settled quiz card replayed from history carries the recorded answer:
    # the chosen index when there was one, and the owner's verbatim words —
    # which, with no index, ARE the answer. Additive: replay names the answer
    # row its card shows; the frozen live quiz_state frame does not grow.
    assert "answered_index" in QuizOutbound.__annotations__
    assert "comment" in QuizOutbound.__annotations__
    assert "answer_row_address" in QuizOutbound.__annotations__
    assert "answer_row_address" not in QuizStateOutbound.__annotations__


def test_max_link_actions_pinned_across_python_and_js():
    """The links-action cap is one number in both languages: the tool gate
    (ouroboros.tools.core._MAX_LINK_ACTIONS) and the UI constant carrier
    (web/modules/api_types.js MAX_LINK_ACTIONS) must never drift apart."""
    from ouroboros.tools.core_artifacts import _MAX_LINK_ACTIONS

    text = API_TYPES.read_text(encoding="utf-8")
    match = re.search(r"^export const MAX_LINK_ACTIONS = (\d+);", text, flags=re.MULTILINE)
    assert match, "api_types.js missing MAX_LINK_ACTIONS"
    assert int(match.group(1)) == _MAX_LINK_ACTIONS


def test_quiz_option_recommendation_is_an_additive_optional_field():
    """The asker marks its recommendation on the option itself (owner batch 1,
    Q7=B): QuizOption grows by ONE optional field in the frozen gateway
    contract; nothing is renamed or removed and no version moves."""

    optional = {
        name for name, annotation in QuizOption.__annotations__.items()
        if (getattr(annotation, "__forward_arg__", None) or str(annotation)).startswith("NotRequired[")
    }
    assert set(QuizOption.__annotations__) == {"label", "detail", "recommended"} and optional == {"detail", "recommended"}


def test_cost_presentation_has_a_closed_nullable_wire_shape():
    from typing import get_args, get_type_hints
    from ouroboros.cost_projection import CostPresentation

    hints = get_type_hints(CostPresentation)
    assert set(hints) == {'scope', 'tracked_amount', 'has_unpriced', 'tracked_final', 'accounting_open', 'has_rows'}
    assert set(get_args(hints['scope'])) == {'own', 'root_tree'}
    assert _contains_none(hints['tracked_amount'])
    assert _contains_none(get_type_hints(ChatOutbound, include_extras=True)['cost_presentation'])


def test_owner_continuation_envelopes_are_exported():
    from ouroboros.gateway import contracts
    for cls in (TaskPauseRequest, TaskPauseResponse, TaskContinueRequest, TaskContinueResponse, ContinuationOffer):
        assert cls.__name__ in contracts.__all__


def test_vendor_program_contracts_are_pinned_in_python():
    from ouroboros.gateway import management_contracts as contracts

    prefixes = ("ClaudexorSelectedProgram", "ClaudexorProgram", "ClaudexorNewest", "ClaudexorMaintenance")
    types = [value for name, value in vars(contracts).items()
             if name.startswith(prefixes) and hasattr(value, "__annotations__")]
    assert types, "the vendor program contract family lost its TypedDicts"
    operation = get_type_hints(contracts.ClaudexorMaintenanceOperation, include_extras=True)
    assert contracts.ClaudexorMaintenanceOperation.__total__
    assert _notrequired_fields(contracts.ClaudexorMaintenanceOperation) == {"profileCleanup"}
    assert get_args(operation["outcome"]) == ("changed", "not_changed", "changed_elsewhere", "unknown")
    entry = get_type_hints(contracts.ClaudexorMaintenanceEntry)
    assert get_args(entry["updateAvailable"]) == ("yes", "no", "cannot_tell")
    assert not {"mechanism", "targetVersion"} & operation.keys()
    assert not {"maintainable", "targets", "installed", "releaseTested", "previous", "available"} & entry.keys()


def test_the_reset_receipt_names_the_accepted_reset_in_python():
    from ouroboros.gateway import management_contracts as contracts

    hints = get_type_hints(contracts.ResetResponse, include_extras=True)
    assert "reset_id" in hints and get_args(hints["reset_id"]) == (str,)
    assert "reset_id" in _notrequired_fields(contracts.ResetResponse)


def test_quiz_answer_file_contract_shape():
    """Dedicated display-only QuizAnswerFile contract: closed shape on the
    Python contract, distinct from ChatAttachmentView, and free of internal
    storage/download locators."""
    from ouroboros.gateway import contracts
    from ouroboros.gateway.attachment_contracts import QuizAnswerFile
    from ouroboros.gateway.history_contracts import ChatHistoryQuiz, ChatSearchQuestion

    expected = {"name", "kind", "mime", "size", "sha256", "unavailable", "limit_bytes"}
    forbidden = {"upload", "mtime_ns", "url", "available", "path", "abs_path"}
    assert "QuizAnswerFile" in contracts.__all__
    assert contracts.QuizAnswerFile is QuizAnswerFile
    py_fields = set(get_type_hints(QuizAnswerFile, include_extras=True))
    assert py_fields == expected
    assert not (py_fields & forbidden)

    # Every quiz-answer carrier types its attachments as the display-only shape.
    assert "attachments" in get_type_hints(ChatSearchQuestion, include_extras=True)
    for cls in (QuizOutbound, QuizStateOutbound, DecisionResponse, ChatHistoryQuiz, ChatSearchQuestion):
        ann = get_type_hints(cls, include_extras=True)["attachments"]
        assert "QuizAnswerFile" in str(ann), cls.__name__
    assert "QuizAnswerFile" in str(get_type_hints(ChatOutbound, include_extras=True)["attachments"])
