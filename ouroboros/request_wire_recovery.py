"""Provider-neutral, same-route request-wire recovery driver.

This leaf owns request-shape adaptation, never provider/model/API choice. Learnable
actions stay pending until exact success; task-local degraded-rung repairs never persist.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import inspect
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from ouroboros.openai_chat_custom import CustomToolProjectionError
from ouroboros.request_wire_contract import (
    NESTED_REASONING_FIELD,
    OPTIONAL_REQUEST_FIELDS,
    PendingWireAction,
    RequestWireProfile,
    apply_effort_action,
    build_request_wire_profile,
    commit_wire_compatibility,
    infer_tool_dialect,
    payload_effort,
    physical_candidate_sha256,
    read_wire_action_records,
    wire_action_identity,
)
from ouroboros.request_wire_receipts import (
    WireAppliedAction,
    WireCandidateManifest,
    WireCandidateSpec,
    bind_wire_candidate,
    bind_wire_compatibility_receipt,
    observe_wire_semantics,
)
from ouroboros.request_wire_resolution import resolve_wire_actions
from ouroboros.usage_accounting import PhysicalAttemptCapture

_MAX_COMPOSED_ACTIONS = 8
_REQUEST_WIRE_HISTORY_MAX = 128
_VALUE_REJECTION_MARKERS = (
    "invalid value",
    "out of range",
    "must be between",
    "must be one of",
    "allowed values",
    "input should be",
)
_CAPABILITY_REJECTION_MARKERS = (
    "unsupported",
    "not supported",
    "unknown parameter",
    "unrecognized",
    "invalid parameter",
    "not permitted",
    "extra inputs",
    "requested parameter",
    "no endpoints found",
)
_MANDATORY_MARKERS = ("mandatory", "cannot be disabled", "must be enabled")
_NON_COMPATIBILITY_4XX = frozenset({401, 402, 403, 408, 409, 425, 429})
_NON_REASONING_OPTIONAL_FIELDS = tuple(
    field for field in OPTIONAL_REQUEST_FIELDS
    if field not in {"reasoning_effort", "output_config", "thinking"}
)


@dataclass(frozen=True)
class _RegisteredCandidate:
    candidate: WireCandidateManifest
    source_payload: Mapping[str, Any]
    target: Mapping[str, Any]


@dataclass(frozen=True)
class _WireCallState:
    active: bool = False
    registered: Tuple[_RegisteredCandidate, ...] = ()
    current: Optional[_RegisteredCandidate] = None
    settled: Optional[Tuple[_RegisteredCandidate, PhysicalAttemptCapture]] = None
    metadata_drop_fields: Tuple[str, ...] = ()
    disclosures: Tuple[Mapping[str, Any], ...] = ()


_WIRE_CALL_STATE: contextvars.ContextVar[_WireCallState] = contextvars.ContextVar(
    "ouroboros_request_wire_call_state", default=_WireCallState(),
)


@contextlib.contextmanager
def request_wire_call_scope() -> Iterator[None]:
    """Isolate candidate/disclosure custody for one sync or async LLM call."""
    token = _WIRE_CALL_STATE.set(_WireCallState(active=True))
    try:
        yield
    finally:
        _WIRE_CALL_STATE.reset(token)


def request_wire_scoped(function: Any) -> Any:
    """Decorator form keeps production transport call sites thin."""
    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            if _WIRE_CALL_STATE.get().active:
                return await function(*args, **kwargs)
            with request_wire_call_scope():
                return await function(*args, **kwargs)
        return _async

    @functools.wraps(function)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        if _WIRE_CALL_STATE.get().active:
            return function(*args, **kwargs)
        with request_wire_call_scope():
            return function(*args, **kwargs)
    return _sync


def _safe_target(target: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(target.get(key))
        for key in (
            "provider", "resolved_model", "usage_model", "base_url", "contract_headers",
        )
        if target.get(key) is not None
    }


def register_wire_candidate(
    candidate: WireCandidateManifest,
    *,
    source_payload: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    """Register a factory-bound candidate (the Phase-2A custom seam)."""
    if not isinstance(candidate, WireCandidateManifest):
        raise TypeError("request-wire candidate must be factory-bound")
    if physical_candidate_sha256(candidate.physical_payload()) != candidate.candidate_sha256:
        raise ValueError("request-wire candidate payload changed before registration")
    registered = _RegisteredCandidate(
        candidate=candidate,
        source_payload=copy.deepcopy(dict(source_payload)),
        target=_safe_target(target),
    )
    state = _WIRE_CALL_STATE.get()
    kept = tuple(
        item for item in state.registered
        if item.candidate.candidate_sha256 != candidate.candidate_sha256
    )
    _WIRE_CALL_STATE.set(replace(
        state,
        registered=(*kept, registered),
        current=registered,
        settled=None,
    ))


def note_provider_metadata_drop_fields(fields: Sequence[str]) -> None:
    """Stage structured exact-route metadata as success-confirmed actions."""
    normalized = tuple(sorted({
        str(field).strip()
        for field in fields
        if str(field).strip() in {*OPTIONAL_REQUEST_FIELDS, NESTED_REASONING_FIELD}
    }))
    if not normalized:
        return
    state = _WIRE_CALL_STATE.get()
    if not state.active:
        return
    _WIRE_CALL_STATE.set(replace(
        state,
        metadata_drop_fields=tuple(sorted(set(state.metadata_drop_fields) | set(normalized))),
    ))


def _profile(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    api_surface: str,
    *,
    value_field: str = "",
) -> RequestWireProfile:
    return build_request_wire_profile(
        target,
        payload,
        api_surface=api_surface,
        evidence_scope="value" if value_field else "capability",
        value_fields=(value_field,) if value_field else (),
    )


def _candidate_spec_for_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
) -> WireCandidateSpec:
    dialect = infer_tool_dialect(payload)
    effort = payload_effort(payload)
    if action.get("kind") == "replace_dialect":
        dialect = str(action.get("to") or dialect)
    elif action.get("kind") == "set_value":
        effort = apply_effort_action(effort, action)
    elif action.get("kind") == "drop_field":
        fields = set(action.get("fields") or ())
        if fields & {
            "reasoning_effort", "thinking", "output_config", NESTED_REASONING_FIELD,
        }:
            effort = "provider_default"
    return WireCandidateSpec(
        dialect,
        effort,
        str(action.get("reason_code") or "provider_recovery_succeeded"),
    )


def _bind_with_applications(
    *,
    target: Mapping[str, Any],
    api_surface: str,
    source_payload: Mapping[str, Any],
    requested_effort: str,
    applications: Sequence[WireAppliedAction],
    fixed_spec: Optional[WireCandidateSpec] = None,
    fixed_ordinal: Optional[int] = None,
) -> WireCandidateManifest:
    if fixed_spec is not None:
        effort = requested_effort
        dialect = fixed_spec.tool_dialect
        for item in applications:
            action = item.action
            if action.get("kind") == "set_value":
                effort = apply_effort_action(effort, action)
            elif action.get("kind") == "drop_field" and set(
                action.get("fields") or ()
            ) & {
                "reasoning_effort", "thinking", "output_config",
                NESTED_REASONING_FIELD,
            }:
                effort = "provider_default"
            elif action.get("kind") == "replace_dialect":
                dialect = str(action.get("to") or dialect)
        spec = WireCandidateSpec(
            dialect,
            effort,
            fixed_spec.reason_code,
            fixed_spec.task_local,
        )
        return bind_wire_candidate(
            target=target,
            api_surface=api_surface,
            source_payload=source_payload,
            candidate_spec=spec,
            requested_effort=requested_effort,
            ladder_ordinal=fixed_ordinal or 1,
            applied_actions=applications,
        )
    if applications:
        current = bind_wire_candidate(
            target=target,
            api_surface=api_surface,
            source_payload=source_payload,
            candidate_spec=WireCandidateSpec(
                infer_tool_dialect(source_payload),
                requested_effort,
                "requested_wire_form",
            ),
            requested_effort=requested_effort,
            ladder_ordinal=1,
        ).physical_payload()
        for index, item in enumerate(applications, start=1):
            spec = _candidate_spec_for_action(current, item.action)
            candidate = bind_wire_candidate(
                target=target,
                api_surface=api_surface,
                source_payload=source_payload,
                candidate_spec=spec,
                requested_effort=requested_effort,
                ladder_ordinal=index + 1,
                applied_actions=applications[:index],
            )
            current = candidate.physical_payload()
        return candidate
    return bind_wire_candidate(
        target=target,
        api_surface=api_surface,
        source_payload=source_payload,
        candidate_spec=WireCandidateSpec(
            infer_tool_dialect(source_payload), requested_effort, "requested_wire_form",
        ),
        requested_effort=requested_effort,
        ladder_ordinal=1,
    )


def _first_durable_action(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    api_surface: str,
    requested_effort: str,
    applications: Sequence[WireAppliedAction],
    *,
    allow_dialect: bool,
) -> Optional[WireAppliedAction]:
    seen = {
        (item.profile.fingerprint, wire_action_identity(item.action))
        for item in applications
    }
    for profile, action in _records_for_step(
        target, payload, api_surface, requested_effort,
    ):
        if not allow_dialect and action.get("kind") == "replace_dialect":
            continue
        if action.get("kind") == "drop_field" and not all(
            _field_is_present(payload, str(field))
            for field in action.get("fields") or ()
        ):
            continue
        if (
            action.get("kind") == "replace_dialect"
            and action.get("from") != infer_tool_dialect(payload)
        ):
            continue
        identity = (profile.fingerprint, wire_action_identity(action))
        if identity not in seen:
            return WireAppliedAction(profile, action, "durable")
    return None


def _direct_openai_tool_source(
    target: Mapping[str, Any], payload: Mapping[str, Any],
) -> bool:
    provider = str(target.get("provider") or "").strip().lower()
    return provider == "openai" and infer_tool_dialect(payload) == "function" and (
        payload_effort(payload) not in {"", "none"}
    )


def _prepare_direct_rung_candidate(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    api_surface: str,
    *,
    dialect: str,
    reason_code: str,
    ordinal: int,
) -> WireCandidateManifest:
    requested = payload_effort(payload)
    source = copy.deepcopy(dict(payload))
    applications: list[WireAppliedAction] = []
    spec = WireCandidateSpec(dialect, requested, reason_code)
    for _ in range(_MAX_COMPOSED_ACTIONS):
        candidate = _bind_with_applications(
            target=target,
            api_surface=api_surface,
            source_payload=source,
            requested_effort=requested,
            applications=applications,
            fixed_spec=spec,
            fixed_ordinal=ordinal,
        )
        addition = _first_durable_action(
            target,
            candidate.physical_payload(),
            api_surface,
            requested,
            applications,
            allow_dialect=False,
        )
        if addition is None:
            return candidate
        applications.append(addition)
    return _bind_with_applications(
        target=target,
        api_surface=api_surface,
        source_payload=source,
        requested_effort=requested,
        applications=applications,
        fixed_spec=spec,
        fixed_ordinal=ordinal,
    )

def _prepare_direct_openai_candidate(
    target: Mapping[str, Any], payload: Mapping[str, Any], api_surface: str,
) -> WireCandidateManifest:
    return _prepare_direct_rung_candidate(
        target, payload, api_surface,
        dialect="openai_chat_custom",
        reason_code="requested_wire_form", ordinal=1,
    )


def _records_for_step(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    api_surface: str,
    requested_effort: str,
) -> Tuple[Tuple[RequestWireProfile, Mapping[str, Any]], ...]:
    profiles = [_profile(target, payload, api_surface)]
    carrier = profiles[0].reasoning_carrier
    value_path = {
        "reasoning_effort": "reasoning_effort",
        NESTED_REASONING_FIELD: "extra_body.reasoning.effort",
        "anthropic.adaptive": "output_config.effort",
    }.get(carrier, "")
    if value_path:
        profiles.append(_profile(target, payload, api_surface, value_field=value_path))
    selected = []
    for profile in profiles:
        resolution = resolve_wire_actions(
            profile,
            requested_effort=payload_effort(payload) or requested_effort,
            records=read_wire_action_records(profile),
        )
        if resolution.effort_conflict:
            continue
        selected.extend((profile, action) for action in resolution.actions)
    return tuple(selected)


def _prepare_durable_candidate(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    api_surface: str,
) -> Optional[WireCandidateManifest]:
    requested = payload_effort(payload)
    if not requested or infer_tool_dialect(payload) == "openai_chat_custom":
        return None
    source = copy.deepcopy(dict(payload))
    applications: list[WireAppliedAction] = []
    seen = set()

    metadata_fields = tuple(
        field for field in _WIRE_CALL_STATE.get().metadata_drop_fields
        if field in source or field == NESTED_REASONING_FIELD
    )
    if metadata_fields:
        metadata = PendingWireAction(_profile(target, source, api_surface), {
            "kind": "drop_field",
            "fields": list(metadata_fields),
            "reason_code": "provider_metadata_constraint",
        })
        applications.append(WireAppliedAction.pending(metadata))
        seen.add((metadata.profile.fingerprint, wire_action_identity(metadata.action)))

    for _ in range(_MAX_COMPOSED_ACTIONS - len(applications)):
        candidate = _bind_with_applications(
            target=target,
            api_surface=api_surface,
            source_payload=source,
            requested_effort=requested,
            applications=applications,
        )
        current = candidate.physical_payload()
        addition = None
        for profile, action in _records_for_step(
            target, current, api_surface, requested,
        ):
            identity = (profile.fingerprint, wire_action_identity(action))
            if identity in seen:
                continue
            addition = WireAppliedAction(profile, action, "durable")
            seen.add(identity)
            break
        if addition is None:
            return candidate
        applications.append(addition)
    return _bind_with_applications(
        target=target,
        api_surface=api_surface,
        source_payload=source,
        requested_effort=requested,
        applications=applications,
    )


def prepare_wire_payload_for_send(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    api_surface: str,
) -> Dict[str, Any]:
    """Apply frozen evidence and bind the exact payload immediately before send."""
    detached = copy.deepcopy(dict(payload))
    digest = physical_candidate_sha256(detached)
    state = _WIRE_CALL_STATE.get()
    existing = next(
        (item for item in reversed(state.registered)
         if item.candidate.candidate_sha256 == digest),
        None,
    )
    if existing is not None:
        _WIRE_CALL_STATE.set(replace(state, current=existing, settled=None))
        return existing.candidate.physical_payload()
    try:
        candidate = (
            _prepare_direct_openai_candidate(target, detached, api_surface)
            if _direct_openai_tool_source(target, detached)
            else _prepare_durable_candidate(target, detached, api_surface)
        )
    except CustomToolProjectionError:
        # An unrepresentable catalog is a representation failure of the CUSTOM
        # rung, not a malformed payload: fall to the function dialect (the
        # canonical source form) so a registered candidate keeps state.current
        # set and every retry rung stays reachable (E4). The generic clause
        # below used to swallow this and send the raw payload with the ladder
        # severed. The fallback bind gets its own catch: a catalog broken enough
        # to also fail the FUNCTION bind (duplicate/unnamed tools) must degrade
        # to the raw-send path below, not escape as a local ValueError that
        # kills the call before anything reaches the provider.
        try:
            candidate = _prepare_direct_rung_candidate(
                target, detached, api_surface,
                dialect="function",
                reason_code="requested_wire_form", ordinal=1,
            )
        except (TypeError, ValueError):
            candidate = None
    except (TypeError, ValueError):
        candidate = None
    if candidate is None:
        for field in state.metadata_drop_fields:
            if field in _NON_REASONING_OPTIONAL_FIELDS:
                detached.pop(field, None)
        _WIRE_CALL_STATE.set(replace(state, current=None, settled=None))
        return detached
    register_wire_candidate(candidate, source_payload=detached, target=target)
    return candidate.physical_payload()


def current_wire_candidate() -> Optional[WireCandidateManifest]:
    current = _WIRE_CALL_STATE.get().current
    return current.candidate if current is not None else None


def note_wire_send_succeeded(capture: Any) -> None:
    state = _WIRE_CALL_STATE.get()
    if state.current is None or not isinstance(capture, PhysicalAttemptCapture):
        return
    if capture.candidate_raw_sha256 != state.current.candidate.candidate_sha256:
        return
    _WIRE_CALL_STATE.set(replace(state, settled=(state.current, capture)))


def note_wire_send_failed() -> None:
    state = _WIRE_CALL_STATE.get()
    _WIRE_CALL_STATE.set(replace(state, settled=None))


def _status_and_message_from_exception(exc: BaseException) -> Tuple[Optional[int], str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    body = getattr(exc, "body", None)
    if body is None and response is not None and callable(getattr(response, "json", None)):
        try:
            body = response.json()
        except Exception:
            body = None
    error = body.get("error") if isinstance(body, Mapping) else None
    message = (
        str(error.get("message") or "")
        if isinstance(error, Mapping)
        else str((body or {}).get("message") or "") if isinstance(body, Mapping)
        else ""
    )
    return status, message or str(exc or "")


def _context_overflow_evidence(error: Any) -> bool:
    try:
        from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES
    except Exception:
        return False
    values = []
    if isinstance(error, BaseException):
        values.extend((getattr(error, "code", None), getattr(error, "type", None)))
        payload = getattr(error, "body", None)
        response = getattr(error, "response", None)
        if payload is None and response is not None and callable(getattr(response, "json", None)):
            try:
                payload = response.json()
            except Exception:
                payload = None
    else:
        payload = error
    if isinstance(payload, Mapping):
        values.extend(payload.get(key) for key in ("code", "type", "kind"))
        nested = payload.get("error")
        if isinstance(nested, Mapping):
            values.extend(nested.get(key) for key in ("code", "type", "kind"))
    return any(str(value or "").strip().lower() in CONTEXT_OVERFLOW_CODES for value in values)


def _field_is_present(payload: Mapping[str, Any], field: str) -> bool:
    if field != NESTED_REASONING_FIELD:
        return field in payload
    extra = payload.get("extra_body")
    return isinstance(extra, Mapping) and isinstance(extra.get("reasoning"), Mapping)


def _error_tokens(message: str) -> frozenset[str]:
    normalized = "".join(
        character if character.isalnum() or character in {"_", "-"} else " "
        for character in str(message or "").lower()
    )
    return frozenset(normalized.split())


def _names_exact_scalar_value(message: str, value: Any) -> bool:
    if not isinstance(value, (str, int, float, bool)) or value == "":
        return False
    low = str(message or "").lower()
    rendered = str(value).lower()
    for label in ("value", "tier"):
        for marker in (
            f"{label} '{rendered}'",
            f'{label} "{rendered}"',
            f"{label} {rendered}",
        ):
            start = low.find(marker)
            if start < 0:
                continue
            end = start + len(marker)
            if end == len(low) or not (low[end].isalnum() or low[end] in "._-"):
                return True
    return False


def _names_exact_effort_value(message: str, effort: str) -> bool:
    return bool(effort and _names_exact_scalar_value(message, effort))


def _value_evidence(message: str) -> bool:
    tokens = _error_tokens(message)
    return bool(tokens.intersection({"value", "tier"})) or any(
        marker in message for marker in _VALUE_REJECTION_MARKERS
    )


def _classify_action(
    registered: _RegisteredCandidate,
    *,
    status_code: Optional[int],
    message: str,
) -> Optional[PendingWireAction]:
    if (
        status_code is None
        or not 400 <= status_code < 500
        or status_code in _NON_COMPATIBILITY_4XX
    ):
        return None
    low = str(message or "").lower()
    if not low or not any(marker in low for marker in (
        *_VALUE_REJECTION_MARKERS,
        *_CAPABILITY_REJECTION_MARKERS,
        *_MANDATORY_MARKERS,
    )):
        return None
    candidate = registered.candidate
    payload = candidate.physical_payload()
    profile = candidate.accepted_profile
    current_effort = payload_effort(payload)
    error_tokens = _error_tokens(low)
    effort_implicated = any(
        marker in low
        for marker in ("reasoning_effort", "reasoning.effort", "reasoning", "effort", "thinking", "output_config")
    )
    named_effort_value = _names_exact_effort_value(low, current_effort)
    value_implicated = _value_evidence(low)
    if (
        profile.provider == "anthropic"
        and profile.reasoning_carrier == "anthropic.disabled"
        and effort_implicated
    ):
        return PendingWireAction(profile, {
            "kind": "drop_field",
            "fields": ["thinking"],
            "reason_code": "provider_unsupported_field",
        })
    if (
        effort_implicated
        and current_effort in {"none", "minimal"}
        and any(marker in low for marker in _MANDATORY_MARKERS)
    ):
        return PendingWireAction(profile, {
            "kind": "set_value",
            "field": "effort",
            "mode": "floor",
            "from": current_effort,
            "to": "low",
            "reason_code": "provider_required_reasoning",
        })
    if effort_implicated and named_effort_value:
        from ouroboros.config import EFFORT_SCALE, effort_one_step_down, effort_rank
        # QUOTED tiers inside [low, current) prescribe — even a negatively-quoted one (accepted FP); prose walks one rung.
        prescribed = [t for t in EFFORT_SCALE[effort_rank("low"):max(effort_rank(current_effort), 0)]
                      if f"'{t}'" in low or f'"{t}"' in low]
        next_effort = prescribed[-1] if prescribed else effort_one_step_down(current_effort)
        if effort_rank(next_effort) >= effort_rank("low") and next_effort != current_effort:
            value_path = {
                "reasoning_effort": "reasoning_effort",
                NESTED_REASONING_FIELD: "extra_body.reasoning.effort",
                "anthropic.adaptive": "output_config.effort",
            }.get(profile.reasoning_carrier, "")
            exact_profile = _profile(
                registered.target,
                payload,
                profile.api_surface,
                value_field=value_path,
            ) if value_path else profile
            return PendingWireAction(exact_profile, {
                "kind": "set_value",
                "field": "effort",
                "mode": "exact",
                "from": current_effort,
                "to": next_effort,
                "reason_code": "provider_prescribed_value",
            })
        return None
    if effort_implicated and value_implicated:
        return None
    if effort_implicated and current_effort in error_tokens:
        return None
    named = []
    compact = low.replace(".", "_")
    for field in (*OPTIONAL_REQUEST_FIELDS, NESTED_REASONING_FIELD):
        aliases = {field, field.replace(".", "_"), field.split(".")[-1]}
        if _field_is_present(payload, field) and any(alias in low or alias in compact for alias in aliases):
            if value_implicated or (
                field != NESTED_REASONING_FIELD
                and _names_exact_scalar_value(low, payload.get(field))
            ):
                return None
            named.append(field)
    if not named:
        return None
    return PendingWireAction(profile, {
        "kind": "drop_field",
        "fields": named,
        "reason_code": "provider_unsupported_field",
    })


def _plan_retry(status_code: Optional[int], message: str) -> Optional[Dict[str, Any]]:
    state = _WIRE_CALL_STATE.get()
    registered = state.current
    if registered is None:
        return None
    try:
        pending = _classify_action(
            registered,
            status_code=status_code,
            message=message,
        )
        if pending is None:
            return None
        existing = registered.candidate.applied_actions
        identity = (pending.profile.fingerprint, wire_action_identity(pending.action))
        if any(
            (item.profile.fingerprint, wire_action_identity(item.action)) == identity
            for item in existing
        ):
            return None
        applied = WireAppliedAction.reactive(pending, task_local=registered.candidate.task_local)
        applications = (*existing, applied)
        if len(applications) > _MAX_COMPOSED_ACTIONS:
            return None
        from ouroboros.openai_chat_dispatch import is_direct_openai_ladder_candidate

        direct_rung = is_direct_openai_ladder_candidate(registered.candidate)
        candidate = _bind_with_applications(
            target=registered.target,
            api_surface=registered.candidate.source_profile.api_surface,
            source_payload=registered.source_payload,
            requested_effort=registered.candidate.requested_effort,
            applications=applications,
            fixed_spec=(registered.candidate.candidate_spec if direct_rung else None),
            fixed_ordinal=(registered.candidate.ladder_ordinal if direct_rung else None),
        )
        register_wire_candidate(
            candidate,
            source_payload=registered.source_payload,
            target=registered.target,
        )
        return candidate.physical_payload()
    except (TypeError, ValueError):
        # CustomToolProjectionError (a ValueError) is included by DESIGN here,
        # unlike prepare_wire_payload_for_send: the closed action vocabulary
        # cannot mutate tools/messages/tool_choice, so a catalog that projected
        # at bind time projects deterministically on retry, and an
        # unrepresentable one can never succeed by retrying -- "no plan" is the
        # correct plan, and the original provider error stays visible upstream.
        return None


def _plan_direct_dialect_retry(
    error: Any,
    *,
    body_error: bool,
) -> Optional[Dict[str, Any]]:
    state = _WIRE_CALL_STATE.get()
    registered = state.current
    if registered is None:
        return None
    try:
        from ouroboros.openai_chat_dispatch import plan_direct_openai_dialect_candidate

        candidate = plan_direct_openai_dialect_candidate(
            target=registered.target,
            source_payload=registered.source_payload,
            current=registered.candidate,
            error=error,
            body_error=body_error,
        )
        if candidate is None:
            return None
        if registered.candidate.accepted_profile.tool_dialect == "openai_chat_custom":
            candidate = _prepare_direct_rung_candidate(
                registered.target,
                registered.source_payload,
                registered.candidate.source_profile.api_surface,
                dialect="function",
                reason_code="provider_rejected_tool_dialect",
                ordinal=2,
            )
        register_wire_candidate(
            candidate,
            source_payload=registered.source_payload,
            target=registered.target,
        )
        return candidate.physical_payload()
    except (TypeError, ValueError):
        # Includes CustomToolProjectionError on purpose -- see _plan_retry.
        return None


def plan_wire_retry_from_exception(exc: BaseException) -> Optional[Dict[str, Any]]:
    if _context_overflow_evidence(exc):
        return None
    status, message = _status_and_message_from_exception(exc)
    return _plan_retry(status, message)


def plan_nonlearning_optional_retry(
    payload: Mapping[str, Any],
    *,
    error: Any,
    body_error: bool = False,
) -> Optional[Dict[str, Any]]:
    """Keep carrier-less optional-field recovery closed and non-durable."""
    if payload_effort(payload):
        return None
    if _context_overflow_evidence(error):
        return None
    if body_error:
        if not isinstance(error, Mapping):
            return None
        raw_status = error.get("status_code", error.get("status", error.get("code")))
        message = str(error.get("message") or "")
        try:
            status = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError, OverflowError):
            status = None
    else:
        if not isinstance(error, BaseException):
            return None
        status, message = _status_and_message_from_exception(error)
    if (
        status is None
        or not 400 <= status < 500
        or status in _NON_COMPATIBILITY_4XX
    ):
        return None
    low = message.lower()
    if not any(marker in low for marker in (
        *_VALUE_REJECTION_MARKERS, *_CAPABILITY_REJECTION_MARKERS,
    )):
        return None
    if _value_evidence(low):
        return None
    named = [
        field for field in _NON_REASONING_OPTIONAL_FIELDS
        if field in payload and (
            field in low or field.replace("_", ".") in low
        )
    ]
    if not named:
        return None
    repaired = copy.deepcopy(dict(payload))
    for field in named:
        repaired.pop(field, None)
    return repaired


def plan_wire_retry_from_body_error(error: Any) -> Optional[Dict[str, Any]]:
    """Body-error parity; status-less and 5xx bodies never authorize recovery."""
    if not isinstance(error, Mapping):
        return None
    if _context_overflow_evidence(error):
        return None
    code = error.get("status_code", error.get("status", error.get("code")))
    try:
        status = int(code) if code is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    return _plan_retry(status, str(error.get("message") or ""))


def plan_next_wire_retry(
    payload: Mapping[str, Any],
    *,
    error: Any,
    body_error: bool = False,
) -> Optional[Dict[str, Any]]:
    """One exception/body-parity entrypoint for the bounded transport drivers."""
    planned = (
        plan_wire_retry_from_body_error(error)
        if body_error else
        plan_wire_retry_from_exception(error)
        if isinstance(error, BaseException) else None
    )
    if planned is not None:
        return planned
    dialect_retry = _plan_direct_dialect_retry(
        error,
        body_error=body_error,
    )
    if dialect_retry is not None:
        return dialect_retry
    return plan_nonlearning_optional_retry(
        payload,
        error=error,
        body_error=body_error,
    )


def finalize_wire_response(
    normalized_response: Mapping[str, Any],
    normalized_usage: Dict[str, Any],
    *,
    custom_receipts: Sequence[Any] = (),
) -> None:
    """Disclose the physical candidate and commit only terminal semantic success."""
    state = _WIRE_CALL_STATE.get()
    settled = state.settled
    if settled is None:
        return
    registered, capture = settled
    candidate = registered.candidate
    try:
        from ouroboros.request_wire_attempt import WireUsageDisclosure

        disclosure = WireUsageDisclosure.from_candidate(candidate, capture).as_dict()
        existing = normalized_usage.get("request_wire")
        if isinstance(existing, Mapping) and dict(existing) != disclosure:
            raise ValueError("request-wire disclosure differs from settled candidate")
        normalized_usage["request_wire"] = disclosure
        disclosures = (*state.disclosures, copy.deepcopy(disclosure))
        state = replace(state, disclosures=disclosures, settled=None)
        _WIRE_CALL_STATE.set(state)
    except (TypeError, ValueError):
        _WIRE_CALL_STATE.set(replace(state, settled=None))
        return
    try:
        observation = observe_wire_semantics(
            candidate=candidate,
            normalized_response=normalized_response,
            normalized_usage=normalized_usage,
            custom_receipts=custom_receipts,
        )
        receipt = bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=capture,
            semantic_observation=observation,
        )
        commit_wire_compatibility(receipt)
    except (TypeError, ValueError):
        return


def merge_request_wire_usage(total: Dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Preserve exact per-attempt disclosures across nested usage aggregation."""
    incoming = []
    history = usage.get("request_wire_history")
    if isinstance(history, list):
        incoming.extend(item for item in history if isinstance(item, Mapping))
    current = usage.get("request_wire")
    if isinstance(current, Mapping):
        incoming.append(current)
    if not incoming:
        return
    existing = total.get("request_wire_history")
    merged = [dict(item) for item in existing] if isinstance(existing, list) else []
    identities = {
        (str(item.get("attempt_id") or ""), str(item.get("candidate_sha256") or ""))
        for item in merged
    }
    omitted = int(total.get("request_wire_history_omitted") or 0)
    try:
        omitted += max(0, int(usage.get("request_wire_history_omitted") or 0))
    except (TypeError, ValueError, OverflowError):
        pass
    for item in incoming:
        identity = (str(item.get("attempt_id") or ""), str(item.get("candidate_sha256") or ""))
        if identity in identities:
            continue
        if len(merged) < _REQUEST_WIRE_HISTORY_MAX:
            merged.append(copy.deepcopy(dict(item)))
            identities.add(identity)
        else:
            omitted += 1
    total["request_wire"] = copy.deepcopy(dict(incoming[-1]))
    total["request_wire_history"] = merged
    if omitted:
        total["request_wire_history_omitted"] = omitted


def request_wire_disclosures() -> Tuple[Dict[str, Any], ...]:
    return tuple(copy.deepcopy(dict(item)) for item in _WIRE_CALL_STATE.get().disclosures)
