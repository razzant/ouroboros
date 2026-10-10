"""Provider-neutral, same-route request-wire recovery driver.

This leaf owns request-shape adaptation, never provider/model/API choice. Learnable
actions stay pending until exact success; task-local degraded-rung repairs never persist.

Actions are ``set_value``, ``drop_field`` and ``replace_dialect``, at most
``_MAX_COMPOSED_ACTIONS`` per call. A positive enum bound to the effort field needs no
scalar echo: the retry value is the highest comparable lower tier, or the known minimum
when every advertised tier is higher. Negative quotes prescribe nothing, a value
rejection never drops its carrier, and the mandatory-reasoning floor and the
unsupported-carrier drop remain. Only a normalized success bound to the exact settled
physical capture teaches ``state/request_wire_compatibility.json`` (entries expire after
14 days); unsettled attempts never teach. A returned response discloses
``usage.request_wire`` after ``validate_wire_attempt_identity`` even when monetary
settlement failed; ``request_wire_history`` is a bounded, ordered disclosure, while
``usage_store.py`` / ``state/usage.sqlite`` owns money.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import inspect
import re
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
    physical_payload: Optional[Mapping[str, Any]] = None
    logical_payload_sha256: str = ""


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
            "processing_preference",
            "processing_native_origin",
            "requested_reasoning_effort",
        )
        if target.get(key) is not None
    }


def register_wire_candidate(
    candidate: WireCandidateManifest,
    *,
    source_payload: Mapping[str, Any],
    target: Mapping[str, Any],
    logical_payload: Optional[Mapping[str, Any]] = None,
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
        physical_payload=candidate.physical_payload(),
        logical_payload_sha256=physical_candidate_sha256(
            logical_payload if logical_payload is not None else source_payload),
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


def refresh_wire_clock(payload: Dict[str, Any], *, api_surface: str) -> Dict[str, Any]:
    """Refresh a NEW physical candidate without losing its canonical recovery source.

    Recognition precedes stamping: a projected custom-tool payload is not the
    source from which its effort/dialect actions were derived. Rebuild through
    the original factory and prove that only this call's clock changed. Same
    invocation rejoins never enter this preparation seam.
    """
    from ouroboros.send_clock import split_clock_note, stamp_clock_note

    digest = physical_candidate_sha256(payload)
    registered = next((item for item in reversed(_WIRE_CALL_STATE.get().registered)
                       if item.candidate.candidate_sha256 == digest), None)
    if registered is None:
        return stamp_clock_note(payload, blocks=api_surface == "messages")
    source = stamp_clock_note(dict(registered.source_payload), blocks=api_surface == "messages")
    if source == registered.source_payload:
        return payload  # no Main policy, or the clock sample is byte-identical
    prior = registered.candidate
    refreshed = bind_wire_candidate(
        target=registered.target, api_surface=prior.source_profile.api_surface,
        source_payload=source, candidate_spec=prior.candidate_spec,
        requested_effort=prior.requested_effort, ladder_ordinal=prior.ladder_ordinal,
        applied_actions=prior.applied_actions,
    )
    physical = refreshed.physical_payload()
    note, clock_free = split_clock_note(physical)
    if note != split_clock_note(source)[0] or clock_free != split_clock_note(payload)[1]:
        raise ValueError("clock refresh changed unrelated physical input")
    register_wire_candidate(refreshed, source_payload=source, target=registered.target)
    return physical


def prepare_wire_payload_for_send(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    api_surface: str,
    logical_payload: Optional[Mapping[str, Any]] = None,
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
        register_wire_candidate(existing.candidate, source_payload=existing.source_payload,
                                target=existing.target, logical_payload=logical_payload)
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
        # Carrierless sends still own exact physical/logical identity. They can
        # repair optional fields without learning a durable wire contract.
        _WIRE_CALL_STATE.set(replace(state, current=None, settled=None,
            physical_payload=copy.deepcopy(detached),
            logical_payload_sha256=physical_candidate_sha256(
                logical_payload if logical_payload is not None else payload)))
        return detached
    register_wire_candidate(candidate, source_payload=detached, target=target,
                            logical_payload=logical_payload)
    return candidate.physical_payload()


def current_wire_candidate() -> Optional[WireCandidateManifest]:
    current = _WIRE_CALL_STATE.get().current
    return current.candidate if current is not None else None


def registered_source_payload(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The canonical source a registered wire form was bound from; None for any other payload.

    A re-finalized wire form (a clock refresh, a same-invocation rejoin) is measured on
    that source so the reply allowance stays a function of the source bytes and the
    unchanged form keeps its registration (a projected dialect carries extra bytes).
    """
    digest = physical_candidate_sha256(payload)
    return next((item.source_payload for item in reversed(_WIRE_CALL_STATE.get().registered)
                 if item.candidate.candidate_sha256 == digest), None)


def note_wire_send_succeeded(capture: Any) -> None:
    state = _WIRE_CALL_STATE.get()
    if state.current is None or not isinstance(capture, PhysicalAttemptCapture):
        return
    if getattr(capture, "candidate_raw_sha256", None) != state.current.candidate.candidate_sha256:
        return
    _WIRE_CALL_STATE.set(replace(state, settled=(state.current, capture)))


def note_wire_send_failed() -> None:
    state = _WIRE_CALL_STATE.get()
    _WIRE_CALL_STATE.set(replace(state, settled=None))


@dataclass(frozen=True)
class _WireRejection:
    status: Optional[int]
    message: str
    param: str = ""
    code: str = ""
    allowed: Tuple[str, ...] = ()
    value_constraint: bool = False
    rejected_value: Optional[str] = None


def _wire_rejection(error: Any) -> _WireRejection:
    """Keep structured constraints through exception/body projection alike.

    Only the positive enum clause supplies alternatives: quotes elsewhere can
    name rejected values. A structured parameter, when present, owns binding.
    """
    response = getattr(error, "response", None)
    body = error if isinstance(error, Mapping) else getattr(error, "body", None)
    if body is None and response is not None and callable(getattr(response, "json", None)):
        try:
            body = response.json()
        except Exception:
            body = None
    body = body if isinstance(body, Mapping) else {}
    node = body.get("error") if isinstance(body.get("error"), Mapping) else body
    status = (getattr(error, "status_code", None) or getattr(response, "status_code", None)
              or body.get("status_code", body.get("status", body.get("code"))))
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    message = str(node.get("message") or str(error or "")).lower()
    param = str(node.get("param") or getattr(error, "param", "") or "").lower()
    code = str(node.get("code") or getattr(error, "code", "") or "").lower()
    values = node.get("allowed_values", node.get("enum"))
    allowed = tuple(v.lower() for v in values if isinstance(v, str)) if isinstance(values, list) else ()
    value_constraint = isinstance(values, list)
    if not allowed:
        clause = re.search(r"\b(?:expected|must be) one of\b|\b(?:allowed|supported) values\b|\binput should be\b", message)
        if clause:
            value_constraint = True
            prefix = message[:clause.start()]
            if re.search(r"\b(?:not|no|unsupported|disallowed)\s*$", prefix):
                return _WireRejection(status, message, param, code, value_constraint=True)
            if not param:
                fields = {*OPTIONAL_REQUEST_FIELDS, "reasoning", "reasoning.effort",
                          "extra_body.reasoning", "extra_body.reasoning.effort", "output_config.effort", "thinking.type"}
                named = re.findall(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*", prefix)
                param = next((name for name in reversed(named) if name in fields), "")
            # Consume the list itself, stopping at prose (including a negative
            # clause). This accepts quoted JSON/Zod enums and plain comma lists.
            # Each item is quoted like its predecessor: a quoted 'or' stays a
            # value and bare prose ends the list. A conjunction ("a, b, or c")
            # joins only the last item of a comma list; what follows is prose.
            tail = message[clause.end():].lstrip(" :[=({")
            tail = re.sub(r"^(?:are|is)\b", "", tail).lstrip(" :[=({")
            tokens, lead, final, piped, comma_list = [], "[\"']?", False, False, False
            while match := re.match(rf"({lead})([a-z][a-z0-9_-]*)[\"']?", tail):
                rest = tail[match.end():]
                # A token named in an explicit refusal after the positive list
                # is not itself an advertised option, even after a comma or
                # the first conjunction ("low, high and xhigh is unsupported").
                if re.match(
                    r"\s*\(?\s*(?:(?:is|are)\s+not\s+(?:supported|allowed|available|permitted)\b"
                    r"|(?:isn't|aren't)\s+(?:supported|allowed|available|permitted)\b"
                    r"|not\s+(?:supported|allowed|available|permitted)\b"
                    r"|(?:(?:is|are)\s+)?(?:unsupported|disallowed|unavailable|requires?)\b)", rest,
                ):
                    break
                tokens.append(match.group(2))
                tail = rest
                lead = match.group(1) or lead
                conjunction = not piped and re.match(rf"\s*,?\s*\b(?:or|and)\b\s*(?={lead}[a-z])", tail)
                separator = conjunction or re.match(r"\s*[,|]\s*", tail)
                if final or not separator:
                    break
                comma_list = comma_list or "," in separator.group()
                final, piped = bool(conjunction and comma_list), piped or "|" in separator.group()
                tail = tail[separator.end():]
            if param and tokens:
                allowed = tuple(tokens)
    rejected = node.get("value")
    if rejected is None:
        from ouroboros.config import EFFORT_SCALE

        scalar = re.search(r"\b(?:value|tier)\s+['\"]?([a-z][a-z0-9_-]*)", message)
        rejected = scalar.group(1) if scalar and scalar.group(1) in EFFORT_SCALE else None
    return _WireRejection(status, message, param, code, allowed, value_constraint,
                          str(rejected).lower() if rejected is not None else None)


def _status_and_message_from_exception(exc: BaseException) -> Tuple[Optional[int], str]:
    evidence = _wire_rejection(exc)
    return evidence.status, evidence.message


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
    evidence: _WireRejection,
) -> Optional[PendingWireAction]:
    status_code, message = evidence.status, evidence.message
    if (
        status_code is None
        or not 400 <= status_code < 500
        or status_code in _NON_COMPATIBILITY_4XX
    ):
        return None
    low = str(message or "").lower()
    if any(token in evidence.code for token in ("quota", "rate_limit", "billing", "auth", "context", "policy")):
        return None
    if not low or not (evidence.value_constraint or any(marker in low for marker in (
        *_VALUE_REJECTION_MARKERS,
        *_CAPABILITY_REJECTION_MARKERS,
        *_MANDATORY_MARKERS,
    ))):
        return None
    candidate = registered.candidate
    payload = candidate.physical_payload()
    profile = candidate.accepted_profile
    current_effort = payload_effort(payload)
    # Rejecting the disabled carrier's native TYPE is not rejecting an effort
    # tier. Preserve the bounded omission repair without dropping a valid
    # carrier when only output_config.effort (or a scalar tier) was refused.
    if (profile.provider == "anthropic" and profile.reasoning_carrier == "anthropic.disabled"
            and evidence.param == "thinking.type" and evidence.allowed
            and "disabled" not in evidence.allowed
            and evidence.rejected_value in {None, "disabled"}):
        return PendingWireAction(profile, {
            "kind": "drop_field", "fields": ["thinking"],
            "reason_code": "provider_unsupported_field",
        })
    error_tokens = _error_tokens(low)
    value_path = {
        "reasoning_effort": "reasoning_effort",
        NESTED_REASONING_FIELD: "extra_body.reasoning.effort",
        "anthropic.adaptive": "output_config.effort",
        "anthropic.disabled": "thinking",
    }.get(profile.reasoning_carrier, "")
    aliases = {value_path, value_path.removeprefix("extra_body."),
               value_path.removeprefix("extra_body.").split(".")[0]}
    if value_path:
        aliases.update({"reasoning", "effort"})
    aliases.discard("")
    named_paths = set(re.findall(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*", low))
    effort_implicated = evidence.param in aliases if evidence.param else bool(aliases & named_paths)
    if effort_implicated and evidence.rejected_value is not None and evidence.rejected_value != current_effort:
        return None
    named_effort_value = _names_exact_effort_value(low, current_effort)
    value_implicated = evidence.value_constraint or _value_evidence(low) or evidence.code in {"invalid_value", "invalid_enum_value", "invalid_option"}
    if (
        profile.provider == "anthropic"
        and profile.reasoning_carrier == "anthropic.disabled"
        and effort_implicated
        and not value_implicated
        and not any(marker in low for marker in _MANDATORY_MARKERS)
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
        and not evidence.value_constraint
    ):
        return PendingWireAction(profile, {
            "kind": "set_value",
            "field": "effort",
            "mode": "floor",
            "from": current_effort,
            "to": "low",
            "reason_code": "provider_required_reasoning",
        })
    if effort_implicated and (named_effort_value or evidence.allowed):
        from ouroboros.config import effort_one_step_down, effort_rank
        if current_effort in evidence.allowed:
            return None  # This constraint does not reject the candidate's value.
        # Any positively advertised reasoning tier is comparable, minimal
        # included; "none" disables reasoning and unknown tokens have no rank.
        supported = [t for t in evidence.allowed if effort_rank(t) >= effort_rank("minimal")]
        prescribed = [t for t in supported if effort_rank(t) < effort_rank(current_effort)]
        # A positive enum owns the minimum when every supported tier is higher;
        # mandatory wording alone retains the legacy low-floor behavior above.
        next_effort = (max(prescribed, key=effort_rank) if prescribed else
                       min(supported, key=effort_rank) if supported else
                       "" if evidence.allowed else effort_one_step_down(current_effort))
        # An unadvertised prose walk still stops at low.
        floor = "minimal" if supported else "low"
        if effort_rank(next_effort) >= effort_rank(floor) and next_effort != current_effort:
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
        implicated = (evidence.param in aliases if evidence.param else
                      field in error_tokens if field in {"stream", "stream_options"}
                      else any(alias in low or alias in compact for alias in aliases))
        if _field_is_present(payload, field) and implicated:
            if value_implicated or (
                field != NESTED_REASONING_FIELD
                and _names_exact_scalar_value(low, payload.get(field))
            ):
                return None
            named.append(field)
    if "stream" in named and "stream_options" in payload and "stream_options" not in named:
        named.append("stream_options")
    if not named:
        return None
    return PendingWireAction(profile, {
        "kind": "drop_field",
        "fields": named,
        "reason_code": "provider_unsupported_field",
    })


def _plan_retry(evidence: _WireRejection) -> Optional[Dict[str, Any]]:
    state = _WIRE_CALL_STATE.get()
    registered = state.current
    if registered is None:
        return None
    try:
        pending = _classify_action(
            registered,
            evidence=evidence,
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
    capture = getattr(exc, "physical_attempt_capture", None)
    candidate = current_wire_candidate()
    if capture is not None and (candidate is None or getattr(capture, "candidate_raw_sha256", None) != candidate.candidate_sha256):
        return None
    return _plan_retry(_wire_rejection(exc))


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
            (field in _error_tokens(low)) if field in {"stream", "stream_options"}
            else (field in low or field.replace("_", ".") in low)
        )
    ]
    if "stream" in named and "stream_options" in payload and "stream_options" not in named:
        named.append("stream_options")
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
    return _plan_retry(_wire_rejection(error))


def _processing_retry(payload: Mapping[str, Any], error: Any,
                      target: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """One ordinary-speed retry after a released, typed no-start receipt."""
    from ouroboros.llm_attempt import ProcessingNotStarted
    from ouroboros.request_wire_receipts import rebind_processing_candidate

    capture = getattr(error, "physical_attempt_capture", None)
    if not isinstance(error, ProcessingNotStarted) or getattr(capture, "state", None) != "released":
        return None
    registered = _WIRE_CALL_STATE.get().current
    route = registered.target if registered is not None else target or {}
    provider, preference = route.get("provider"), route.get("processing_preference")
    if route.get("processing_native_origin") != "preference":
        return None
    current = registered.candidate.physical_payload() if registered is not None else copy.deepcopy(dict(payload))
    if getattr(capture, "candidate_raw_sha256", None) != physical_candidate_sha256(current):
        return None
    if provider in {"openai", "openrouter"}:
        field, standard = "service_tier", "default"
        allowed = {"fast": {"fast", "priority"}, "economy": {"flex"}}.get(preference, set())
        # An explicit extra_body override has precedence and must not be
        # accidentally replaced by the top-level advisory projection.
        if isinstance(current.get("extra_body"), dict) and "service_tier" in current["extra_body"]:
            return None
    elif provider == "anthropic":
        field, standard = "speed", "standard"
        allowed = {"fast"} if preference == "fast" else set()
    else:
        return None
    if current.get(field) not in allowed:
        return None
    if registered is None:
        current[field] = standard
        return current
    try:
        candidate, source = rebind_processing_candidate(
            registered.candidate, target=route, source_payload=registered.source_payload,
            field_name=field, standard_value=standard,
        )
    except (TypeError, ValueError):
        return None  # The original provider refusal remains the caller's fact.
    register_wire_candidate(candidate, source_payload=source, target=route)
    return candidate.physical_payload()


def plan_next_wire_retry(
    payload: Mapping[str, Any],
    *,
    error: Any,
    body_error: bool = False,
    target: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """One exception/body-parity entrypoint for the bounded transport drivers."""
    if getattr(error, "stream_incomplete", False):
        return None
    state = _WIRE_CALL_STATE.get()
    registered, physical = state.current, state.physical_payload
    digest = physical_candidate_sha256(physical) if physical is not None else ""
    if physical is not None and physical_candidate_sha256(payload) not in {
        digest, state.logical_payload_sha256,
        physical_candidate_sha256(registered.source_payload) if registered is not None else "",
    }:
        return None  # Only inputs bound by this attempt's preparation seam may recover it.
    capture = getattr(error, "physical_attempt_capture", None)
    if capture is not None and (not digest or getattr(capture, "candidate_raw_sha256", None) != digest):
        return None
    current = physical if physical is not None else payload
    processing = _processing_retry(current, error, target)
    if processing is not None:
        return processing
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
        current,
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
