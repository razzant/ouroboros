"""Physical-attempt candidates and send-time prompt-cache policy.

One provider send is a *candidate*: the exact payload object that goes on the
wire. This module owns the facts every lane must state about that object —
the send copy, its canonical digest, the accounting request built from it, the
durable candidate manifest written before dispatch — plus the finalizer that
decides what cache markers the assembled payload actually ships with, since the
finalizer is the only point that sees tools, system and messages together.
Structured provider-overflow predicates live here too: they read the same
candidate-attached facts rather than provider prose.
"""


from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
from typing import Any, Dict, List, Optional, Set

from ouroboros.anthropic_native_custody import is_replayed_native_content
from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES
from ouroboros.request_wire_recovery import prepare_wire_payload_for_send
from ouroboros.transport_custody import ProviderNotDispatched, is_loopback_base_url
from ouroboros.usage_accounting import (
    AttemptRequest,
    PhysicalAttemptPreconditionFailed,
    PhysicalAttemptPreparationFailed,
    adopt_physical_attempt_capture,
    current_physical_attempt_context,
    current_physical_attempt_predicate,
    current_usage_scope,
    execute_physical_attempt,
    execute_physical_attempt_async,
    last_physical_attempt_capture,
)


# Provider-valid Anthropic ephemeral-cache tiers.
_VALID_CACHE_TTLS = frozenset({"5m", "1h"})


# Only explicit wire tiers have a knowable horizon; bare "default" does not.
_CACHE_TTL_SECONDS = {"5m": 300, "1h": 3600}


PROVIDER_POLICY_REFUSAL = "provider_policy_refusal"


class PhysicalDispatchInterrupted(PhysicalAttemptPreconditionFailed):
    """The existing caller control/deadline refused a send before request bytes."""

    code = "model_operation_interrupted"

    def __init__(self, reason: str):
        super().__init__(f"Physical dispatch interrupted: {reason}")
        self.control_reason = reason


class _PhysicalSendNotStarted(PhysicalDispatchInterrupted, ProviderNotDispatched):
    """Positive no-dispatch evidence for this send alone, not the recovery ladder."""


def physical_attempt_headroom() -> Optional[int]:
    """Sends still claimable in this actor context; None when unbounded.

    An advisory read beside the window above: the claim itself is what enforces
    the bound, this only lets a caller decline to spend a send it knows belongs
    to its own later repair.
    """
    from ouroboros.usage_accounting import _PHYSICAL_LIMIT

    state = _PHYSICAL_LIMIT.get()
    return None if state is None else max(0, state.maximum - state.used)


def require_physical_dispatch_window() -> Optional[float]:
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec

    owner = current_model_wait()
    reason = owner.control_reason() if owner is not None else None
    if reason:
        raise _PhysicalSendNotStarted(reason)
    remaining = dispatch_deadline_remaining_sec()
    if remaining is not None and remaining <= 0:
        raise _PhysicalSendNotStarted("deadline")
    return remaining


def preserve_prior_dispatch(error: BaseException, prior: Any) -> None:
    """Raise with paid custody instead of a ladder-wide no-dispatch assertion."""
    if (isinstance(error, PhysicalDispatchInterrupted)
            and getattr(prior, "state", None) in {"dispatched", "unresolved"}):
        failure = PhysicalDispatchInterrupted(error.control_reason)
        failure.deadline_attempt_capture = getattr(error, "physical_attempt_capture", None)
        failure.physical_attempt_capture = prior
        adopt_physical_attempt_capture(prior)
        raise failure from error


def strongest_dispatch_capture(prior: Any, current: Any) -> Any:
    """A later closed attempt cannot erase an earlier unresolved recovery send."""
    return prior if getattr(prior, "state", None) in {"dispatched", "unresolved"} else current or prior


class ProviderPolicyRefusal(RuntimeError):
    """Typed refusal: a policy layer would not let this call reach a provider.

    Not a provider failure — nothing upstream answered — so no rung of the
    recovery ladder can repair it: dropping a parameter, rerouting the endpoint
    or stripping replayed reasoning all re-attempt a call that was refused, and
    the caller ends up seeing whatever the re-attempt produced instead of the
    refusal. It carries the machine-readable ``code`` so the ladder can classify
    it structurally, exactly as the subscription-window refusal is classified in
    ``loop_llm_call.classify_llm_exception`` — never by matching prose.

    A transport that cannot import this class states the same fact by setting
    ``code`` to :data:`PROVIDER_POLICY_REFUSAL` on its own exception type; a
    family of refusals (connection not permitted, egress denied, tenant blocked)
    either subclasses this or carries the same code.
    """

    code = PROVIDER_POLICY_REFUSAL


def _is_provider_policy_refusal(exc: BaseException) -> bool:
    """Structural test: a typed refusal, by class or by the declared ``code``."""
    return isinstance(exc, ProviderPolicyRefusal) or (
        str(getattr(exc, "code", "") or "") == PROVIDER_POLICY_REFUSAL
    )


def _structured_error_values(payload: Any) -> Set[str]:
    if not isinstance(payload, dict):
        return set()
    nodes = [payload]
    if isinstance(payload.get("error"), dict):
        nodes.append(payload["error"])
    return {
        str(node.get(key) or "").strip().lower()
        for node in nodes
        for key in ("code", "type")
        if str(node.get(key) or "").strip()
    }


def _is_structured_context_overflow_exception(exc: BaseException) -> bool:
    """Read only facts attached to this exception; never a stale ContextVar."""
    values = {
        str(getattr(exc, key, "") or "").strip().lower()
        for key in ("code", "type")
        if str(getattr(exc, key, "") or "").strip()
    }
    values.update(_structured_error_values(getattr(exc, "body", None)))
    capture = getattr(exc, "physical_attempt_capture", None)
    if capture is not None:
        values.update({
            str(getattr(capture, key, "") or "").strip().lower()
            for key in ("provider_code", "provider_error_type")
            if str(getattr(capture, key, "") or "").strip()
        })
    return bool(values & CONTEXT_OVERFLOW_CODES)


def _is_structured_context_overflow_body(error: Any) -> bool:
    return bool(_structured_error_values(error) & CONTEXT_OVERFLOW_CODES)


def cache_ttl_seconds(applied_ttl: Any) -> Optional[int]:
    """Return seconds only for an explicit TTL carried by the candidate."""
    return _CACHE_TTL_SECONDS.get(str(applied_ttl or "").strip().lower())


def supports_message_cache_control(model: str) -> bool:
    """Whether the OpenRouter family honors message cache breakpoints."""
    m = str(model or "").strip().lstrip("~")
    return m.startswith("anthropic/") or m.startswith("google/gemini-")


def openai_family_model(model: str) -> bool:
    """Whether a model id names OpenAI's public-API family (``openai/…`` on OpenRouter,
    ``openai::…`` direct; the ``~`` processing prefix and a ``:online`` suffix keep it).

    Dated external fact (probes 2026-09-25; inventory row in DEVELOPMENT §2): this family
    reuses a prompt cache only for the WHOLE leading system section plus tool schemas as
    one unit, or for an exact earlier prompt as a prefix, and the routing key partitions
    it. That is why its send copy keeps mutable context out of the leading system message
    (``llm_messages.split_leading_system_prefix``) and shares one sticky session per model
    and governance prefix (``_openrouter_session_identity``). OpenRouter ``openai/gpt-oss-*``
    ids are served by third parties and merely inherit the projection: disclosed, not gated.
    """
    from ouroboros.provider_models import normalize_model_identity

    raw = str(model or "").strip().lstrip("~")
    identity = normalize_model_identity(raw) or raw
    return identity.strip().lower().startswith("openai/")


def openai_family_route(target: Dict[str, Any]) -> bool:
    """The send-copy predicate: direct ``openai``, or an OpenRouter ``openai/…`` id — never
    a generic OpenAI-compatible server that happens to serve an ``openai/…`` name."""
    provider = str(target.get("provider") or "").strip().lower()
    if provider == "openai":
        return True
    if provider != "openrouter":
        return False
    return openai_family_model(str(target.get("usage_model") or target.get("resolved_model") or ""))


def _route_normalizes_cache_breakpoints(target: Dict[str, Any]) -> bool:
    """Whether the send-time finalizer may normalize cache breakpoints."""
    if str(target.get("provider") or "") == "anthropic":
        return True
    model = str(target.get("resolved_model") or "").strip().lstrip("~")
    return bool(
        target.get("supports_openrouter_extensions")
        and supports_message_cache_control(model)
        and model.startswith("anthropic/")
    )


def _applied_payload_cache_ttl(payload: Dict[str, Any]) -> Optional[str]:
    """Strongest cache TTL carried by THIS exact candidate payload.

    Same reporting rule as the send-time finalizer's return value
    (``_normalize_payload_cache_ttl``: 1h > 5m > bare markers = "default";
    None when the payload carries no markers). Read per candidate rather than
    plumbed from the finalizer because the retry ladder can strip markers
    (``_retry_without_prompt_cache_parameter``) after the finalizer ran — the
    reservation must price the payload actually being sent, not the original.
    """
    breakpoints = _PayloadCachePolicyMixin._payload_cache_breakpoints(payload)
    ttls = {
        str((holder.get("cache_control") or {}).get("ttl") or "").strip().lower()
        for holder in breakpoints
    }
    if "1h" in ttls:
        return "1h"
    if "5m" in ttls:
        return "5m"
    return "default" if breakpoints else None


def submitted_processing_mode(target: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """Read the actual native carrier, never infer execution from requested intent."""
    provider = target.get("provider")
    if provider in {"openai", "openrouter"}:
        extra = payload.get("extra_body")
        value = (extra["service_tier"] if isinstance(extra, dict) and "service_tier" in extra
                 else payload.get("service_tier"))
    elif provider == "anthropic":
        value = payload.get("speed")
    elif provider == "claudexor":
        options = payload.get("options") or {}
        value = options.get("processingPreference") or options.get("serviceTier")
    else:
        value = None
    return value if isinstance(value, str) else ""


def apply_processing_preference(target: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Project already captured advisory intent before sealing this send copy.

    These are provider protocol fields, not a model eligibility table. Explicit
    native options win; unknown transports retain their ordinary request shape.
    A retry already carries its replacement native mode and is never re-resolved.
    """
    preference = target.get("processing_preference")
    if not preference:
        return
    if submitted_processing_mode(target, payload):
        target["processing_native_origin"] = "native_override"
        return
    provider = target.get("provider")
    if provider in {"openai", "openrouter"}:
        payload["service_tier"] = {
            "standard": "default", "fast": "priority", "economy": "flex",
        }[preference]
    elif provider == "anthropic":
        # Messages has no synchronous Economy speed. Its explicit ordinary
        # projection preserves the advisory request without inventing a tier.
        payload["speed"] = "fast" if preference == "fast" else "standard"
    else:
        return
    target["processing_native_origin"] = "preference"


def attach_processing_receipt(target: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Project the matching terminal attempt; never reconstruct a pre-fallback mode."""
    from ouroboros._usage_response import processing_receipt

    provider = str(target.get("provider") or "")
    model = str(target.get("usage_model") or target.get("resolved_model") or "")
    capture = last_physical_attempt_capture()
    matched = capture is not None and capture.provider == provider and capture.model == model
    requested = (capture.processing_preference if matched
                 else str(target.get("processing_preference") or ""))
    submitted = capture.submitted_processing_mode if matched else ""
    receipt = processing_receipt(provider, usage, requested=requested, submitted_native=submitted)
    if receipt is not None:
        usage["processing"] = receipt


def processing_contract_headers(target: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, str]:
    """The native Messages speed beta belongs to the same exact request profile."""
    headers = dict(target.get("contract_headers") or {})
    if target.get("provider") == "anthropic" and payload.get("speed") == "standard":
        betas = [value.strip() for value in headers.get("anthropic-beta", "").split(",")
                 if value.strip() != "fast-mode-2026-02-01"]
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        else:
            headers.pop("anthropic-beta", None)
    if target.get("provider") == "anthropic" and (
        payload.get("speed") == "fast" or (
            "speed" not in payload and target.get("processing_preference") == "fast"
        )
    ):
        betas = [value.strip() for value in headers.get("anthropic-beta", "").split(",") if value.strip()]
        if "fast-mode-2026-02-01" not in betas:
            betas.append("fast-mode-2026-02-01")
        headers["anthropic-beta"] = ",".join(betas)
    return headers


class ProcessingNotStarted(ProviderNotDispatched):
    """A provider-owned processing refusal proving this generation never began."""

    def __init__(self, error: BaseException, *, reason: str):
        super().__init__(str(error))
        self.processing_reason = reason
        for name in ("body", "code", "type", "status_code", "response"):
            if hasattr(error, name):
                setattr(self, name, getattr(error, name))


def processing_refusal(target: Dict[str, Any], payload: Dict[str, Any],
                       error: BaseException) -> BaseException:
    """Normalize only a documented native refusal; socket/stream errors stay unknown.

    Dedicated Flex resource refusal and unsupported request fields precede
    generation. Generic quota, overload, timeout and stream failures do not.
    """
    if getattr(error, "stream_incomplete", False) or isinstance(error, ProviderNotDispatched):
        return error
    if (target.get("processing_preference") not in {"fast", "economy"}
            or target.get("processing_native_origin") != "preference"):
        return error
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    body = getattr(error, "body", None)
    if body is None and response is not None and callable(getattr(response, "json", None)):
        try:
            body = response.json()
        except (ValueError, TypeError):
            return error
    native_error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(native_error, dict):
        return error
    provider, mode = target.get("provider"), submitted_processing_mode(target, payload)
    reason = ""
    if (provider == "anthropic" and mode == "fast" and status == 429
            and native_error.get("type") == "rate_limit_error"):
        reason = "capacity"
    elif provider in {"openai", "openrouter"} and mode in {"priority", "fast", "flex"}:
        if mode == "flex" and status in {400, 429, 503} and native_error.get("code") in {
            "resource_unavailable", "unsupported_service_tier",
        }:
            reason = "capacity"
        elif (status == 400 and native_error.get("code") == "unsupported_parameter"
              and native_error.get("param") == "service_tier"):
            reason = "unsupported"
    if reason:
        normalized = ProcessingNotStarted(error, reason=reason)
        normalized.body = copy.deepcopy(body)
        return normalized
    return error


def _attempt_request(
    target: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    source: Optional[str] = None,
) -> AttemptRequest:
    """Build secret-free facts for one final inspectable candidate."""
    prompt_payload = {
        key: value
        for key, value in payload.items()
        if key not in {
            "model", "max_tokens", "max_completion_tokens", "temperature",
            "top_p", "top_k", "timeout", "stream",
        }
    }
    try:
        prompt_chars = len(json.dumps(prompt_payload, ensure_ascii=False, default=str))
    except Exception:
        prompt_chars = len(str(prompt_payload or ""))
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload

    bounded_tokens = bounded_prompt_tokens_for_payload(prompt_payload, prompt_chars)
    request_source = source
    if request_source is None:
        bound_scope = current_usage_scope()
        request_source = (
            str(bound_scope.source)
            if bound_scope is not None and bound_scope.source
            else "llm.chat"
        )
    raw = _canonical_candidate_bytes(payload)
    context = _canonical_candidate_bytes({
        key: payload[key] for key in ("system", "messages", "tools", "functions") if key in payload
    })
    return AttemptRequest(
        model=str(target.get("usage_model") or target.get("resolved_model") or payload.get("model") or ""),
        provider=str(target.get("provider") or "unknown"),
        prompt_tokens_estimate=max(0, prompt_chars // 4),
        max_completion_tokens=int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 0),
        source=str(request_source or ""),
        prompt_cache_ttl=_applied_payload_cache_ttl(payload) or "",
        candidate_raw_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_raw_size_bytes=len(raw),
        candidate_context_sha256=hashlib.sha256(context).hexdigest(),
        candidate_context_size_bytes=len(context),
        candidate_measurement_kind="canonical_json_v1",
        physical_context=current_physical_attempt_context(),
        route_is_loopback=is_loopback_base_url(target.get("base_url")),
        prompt_tokens_bounded_estimate=bounded_tokens,
        processing_preference=str(target.get("processing_preference") or ""),
        submitted_processing_mode=submitted_processing_mode(target, payload),
        processing_basis=copy.deepcopy(target.get("processing_basis")),
    )


def _canonical_candidate_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def _physical_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the send copy with capsule metadata removed only from context turns."""
    candidate = copy.deepcopy(payload)

    def _strip(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("_context_capsule", None)
            for child in value.values():
                _strip(child)
        elif isinstance(value, list):
            for child in value:
                _strip(child)

    for key in ("system", "messages"):
        _strip(candidate.get(key))
    return candidate


def _finalized_physical_candidate(
    target: Dict[str, Any], payload: Dict[str, Any], api_surface: str,
) -> Dict[str, Any]:
    physical = _physical_candidate(payload)
    if target.get("context_mode") == "nano":
        physical = _fit_output_payload(target, physical, api_surface)
    return prepare_wire_payload_for_send(
        {**target, "contract_headers": processing_contract_headers(target, physical)},
        physical, api_surface=api_surface,
    )



def _prepared_input_measurement(target: Dict[str, Any], payload: Dict[str, Any]) -> dict:
    """Current routes offer an actual-shape estimate, never an exact template count."""
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload

    measured = target.get("local_input_measurement") or {}
    if target.get("provider") == "local" and measured.get("supported") and measured.get("input_is_exact") is True:
        from ouroboros.local_model_server import input_fingerprint

        if measured.get("native_input_sha256") == input_fingerprint(payload):
            return {"input_tokens": measured["input_tokens"], "input_is_exact": True,
                    "tokenizer_template_provenance": measured.get("tokenizer_template_provenance"),
                    "route_capacity_tokens": measured.get("context_window"), "route_capacity_confirmed": True}
    context = {key: payload[key] for key in ("system", "messages", "input", "instructions", "tools", "functions") if key in payload}
    chars = len(_canonical_candidate_bytes(context).decode("utf-8"))
    return {"input_tokens": bounded_prompt_tokens_for_payload(context, chars),
            "input_is_exact": False, "tokenizer_template_provenance": None,
            "route_capacity_tokens": target.get("context_window_tokens", getattr(current_physical_attempt_context(), "capacity_total_tokens", None)),
            "route_capacity_confirmed": bool(target.get("context_window_confirmed", False))}


def _fit_output_payload(target: Dict[str, Any], payload: Dict[str, Any], api_surface: str) -> Dict[str, Any]:
    """Use the shared Nano arithmetic after native tool projection and before sealing."""
    from dataclasses import asdict
    from ouroboros.context_budget import OWNER_NANO_TARGET_TOKENS, NANO_MIN_HEADROOM_TOKENS
    from ouroboros.context_fit import resolve_call_context_fit

    field = next((key for key in ("max_completion_tokens", "max_tokens") if isinstance(payload.get(key), int)), None)
    if field is None:
        return payload  # An opaque route has no enforceable native output field here.
    measured = _prepared_input_measurement(target, payload)
    provider = target.get("provider")
    # Local formatters can make additional internal generations outside this cap.
    limit_enforced = provider == "openai" and field == "max_completion_tokens" or provider == "anthropic" and field == "max_tokens"

    if provider == "local":
        limit_enforced = measured["input_is_exact"] and (target.get("local_input_measurement") or {}).get("output_limit_enforced") is True
    nano = target.get("context_mode") == "nano"
    fit = resolve_call_context_fit(**measured, caller_max_tokens=payload[field],
        total_target_tokens=OWNER_NANO_TARGET_TOKENS if nano else None,
        minimum_free_tokens=NANO_MIN_HEADROOM_TOKENS if nano else 0, output_limit_enforced=limit_enforced,
        reasoning_included_in_limit=True if limit_enforced else None)
    facts = asdict(fit)
    if provider == "local" and measured["input_is_exact"]:
        facts["serving_process_id"] = target["local_input_measurement"].get("process_id")
    target["call_context_fit"] = facts
    if fit.effective_max_tokens <= 0 or measured["input_is_exact"] and fit.fit_status in {"unfit", "insufficient_headroom"}:
        error = PhysicalAttemptPreparationFailed("Exact prepared input does not fit the selected context allowance")
        error.call_context_fit = facts
        raise error
    result = {**payload, field: fit.effective_max_tokens}
    facts["candidate_raw_sha256"] = hashlib.sha256(_canonical_candidate_bytes(result)).hexdigest()
    return result


def _candidate_before_dispatch(candidate: Dict[str, Any], request: AttemptRequest):
    """Close over one final candidate without putting it in accounting rows."""
    predicate = current_physical_attempt_predicate()

    def _persist(reservation):
        fresh = _attempt_request(
            {"provider": request.provider, "usage_model": request.model}, candidate,
            source=request.source,
        )
        identity = (
            fresh.candidate_raw_sha256, fresh.candidate_raw_size_bytes,
            fresh.candidate_context_sha256, fresh.candidate_context_size_bytes,
        )
        expected = (
            request.candidate_raw_sha256, request.candidate_raw_size_bytes,
            request.candidate_context_sha256, request.candidate_context_size_bytes,
        )
        if identity != expected:
            raise PhysicalAttemptPreparationFailed(
                "physical candidate changed before dispatch", attempt_id=reservation.attempt_id,
            )
        from ouroboros.observability import persist_physical_candidate

        scope = current_usage_scope()
        task_id = str(scope.task_id if scope is not None else request.task_id)
        persisted = persist_physical_candidate(
            reservation.drive_root,
            task_id=task_id,
            attempt_id=reservation.attempt_id,
            candidate=candidate,
            candidate_facts={
                "candidate_raw_sha256": request.candidate_raw_sha256,
                "candidate_raw_size_bytes": request.candidate_raw_size_bytes,
                "candidate_context_sha256": request.candidate_context_sha256,
                "candidate_context_size_bytes": request.candidate_context_size_bytes,
                "candidate_measurement_kind": request.candidate_measurement_kind,
                "physical_context": (
                    dict(vars(request.physical_context)) if request.physical_context is not None else None
                ),
            },
        )
        # CPL-5 forward invariant (model-visible ⟺ logged): reconstruct the
        # durable record just written and byte-compare it with the wire-bound
        # candidate. A mismatch is a typed durable fact, never a second dispatch
        # gate — the in-memory identity refusal above stays the only blocking
        # authority. The fresh seam digests are reused so the raw candidate is
        # not serialized again.
        from ouroboros.model_send_seal import verify_sealed_candidate

        verify_sealed_candidate(
            reservation.drive_root,
            task_id=task_id,
            attempt_id=reservation.attempt_id,
            candidate=candidate,
            manifest_ref=persisted["manifest_ref"],
            raw_sha256=fresh.candidate_raw_sha256,
            raw_size_bytes=fresh.candidate_raw_size_bytes,
        )
        if predicate is not None:
            try:
                accepted = predicate(request)
            except BaseException as exc:
                # Persistence already succeeded. Preserve the only durable link
                # even when the host predicate itself raises before returning.
                try:
                    exc.candidate_manifest_ref = persisted["manifest_ref"]
                except Exception:
                    pass
                raise
            if accepted is False:
                failure = PhysicalAttemptPreconditionFailed(
                    "physical candidate precondition rejected dispatch",
                    attempt_id=reservation.attempt_id,
                )
                failure.candidate_manifest_ref = persisted["manifest_ref"]
                raise failure
        return persisted["manifest_ref"]

    return _persist


def _execute_candidate(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    """Keep existing two-argument injected executors usable."""
    adopt_physical_attempt_capture(None)
    require_physical_dispatch_window()
    send, before_dispatch = _deadline_checked_send(send, before_dispatch)
    if "before_dispatch" not in inspect.signature(execute_physical_attempt).parameters:
        return execute_physical_attempt(request, send)
    return execute_physical_attempt(request, send, before_dispatch=before_dispatch)


async def _execute_candidate_async(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    adopt_physical_attempt_capture(None)
    require_physical_dispatch_window()
    send, before_dispatch = _deadline_checked_send(send, before_dispatch)
    if "before_dispatch" not in inspect.signature(execute_physical_attempt_async).parameters:
        return await execute_physical_attempt_async(request, send)
    return await execute_physical_attempt_async(request, send, before_dispatch=before_dispatch)


def _deadline_checked_send(send: Any, before_dispatch: Any):
    def prepare(reservation):
        manifest = before_dispatch(reservation) if before_dispatch is not None else None
        try:
            require_physical_dispatch_window()
        except PhysicalDispatchInterrupted as exc:
            exc.candidate_manifest_ref = manifest
            raise
        return manifest

    def dispatch():
        require_physical_dispatch_window()
        return send()

    return dispatch, prepare


class _PayloadCachePolicyMixin:
    """Send-time cache policy on the fully assembled payload."""

    # Anthropic accepts at most four declared cache breakpoints per request.
    _MAX_CACHE_BREAKPOINTS = 4

    @staticmethod
    def _payload_cache_breakpoints(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Blocks carrying a ``cache_control`` marker, in the real wire prefix order
        ``tools -> system -> messages`` — NOT the order arguments happen to arrive in.

        Descends one level INTO a block's own ``content`` list: a direct-Anthropic
        ``tool_result`` block nests its blocks (``_anthropic_messages`` builds it from a
        ``role="tool"`` message whose content is a list), so the sealed transcript anchor
        (``context_fit.seal_task_transcript``) sits at ``messages[i].content[j].content[k]``.
        Missing it undercounts the cap and leaves that anchor out of TTL ordering exactly
        on the lane whose provider enforces both. ``tool_result`` is the only nested-content
        shape, and the descent is route-independent because no other payload nests."""
        holders: List[Dict[str, Any]] = []
        for key in ("tools", "system", "messages"):
            part = payload.get(key)
            for item in (part if isinstance(part, list) else [part]):
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("cache_control"), dict):
                    holders.append(item)
                content = item.get("content")
                if isinstance(content, list):
                    if is_replayed_native_content(content):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if isinstance(block.get("cache_control"), dict):
                            holders.append(block)
                        nested = block.get("content")
                        if isinstance(nested, list):
                            holders.extend(
                                inner for inner in nested
                                if isinstance(inner, dict)
                                and isinstance(inner.get("cache_control"), dict)
                            )
        return holders

    def _normalize_payload_cache_ttl(
        self,
        target: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Finalize cache policy on the FULLY ASSEMBLED payload; report its strongest TTL.

        The one point where tools, system and messages coexist, hence the single home for
        send-time cache policy (v6.77.0 — replaces two per-builder "mark the last tool"
        copies and restores the TTL ordering guard lost in 176567b BY CONSTRUCTION): a
        ``1h`` breakpoint promotes the earlier EXISTING breakpoints to ``1h`` (a longer TTL
        must precede a shorter one — 5m tools before 1h system is a hard 400) and never
        creates a marker on an earlier segment; a bare marker is the provider default and
        ranks as 5m; the ONLY marker it ever adds is on the last tool schema, and only when
        the tools segment carries none (unconditional on this family in both deleted sites —
        a tool-free payload therefore stays uncached HERE, and system/messages never gain a
        marker they did not declare; a tool-free lane is cached only by DECLARING its stable
        prefix at the caller, as the review surfaces and the safety supervisor do via
        ``review_helpers.cached_prompt_blocks``); above the four-breakpoint cap the four EARLIEST
        (governance-prefix) markers are kept, the tail MARKERS — never content — are dropped
        and the reduction is disclosed in usage (rationale and the builder-side loud layer:
        ``docs/ARCHITECTURE.md``). Only this freshly assembled payload is normalized —
        never caller-owned messages/tools, the canonical transcript, or a route that cannot
        carry these markers (``_route_normalizes_cache_breakpoints``). "1h" wins over
        "default" so pricing bills the extended-tier write multiplier.

        The owner's global TTL (``config.resolve_prompt_cache_ttl``, owner decision
        2026-08-08 Q2=A) has its single WIRE authority here — the one place that decides
        what every marker on this family actually ships as. It is not the only READER:
        ``review_helpers.cached_prompt_blocks(ttl=None)`` projects the same setting into
        the block it owns so a non-normalizing route still carries the owner's tier; on
        this family the finalizer would stamp that block to the same value anyway, so the
        two readers cannot diverge on the wire (``config.resolve_prompt_cache_ttl`` names
        both). When the setting names an explicit tier
        ('5m'/'1h') it is stamped onto EVERY existing breakpoint of this family —
        including caller-declared review/safety prefixes, which is what makes it an
        HONEST override rather than a floor — before the promotion rule runs, so
        ordering stays legal by construction (the 176567b every-call-400 class).
        'default' keeps the pre-setting behavior byte-for-byte: bare markers stay bare
        and a caller-declared ttl stands. It never CREATES a marker (the d32f703d
        empty-block 400 class), and non-Anthropic wire formats are untouched (the
        v5.30.0 Gemini ttl-field class).
        """
        breakpoints = self._payload_cache_breakpoints(payload)
        note: Optional[Dict[str, Any]] = None
        if _route_normalizes_cache_breakpoints(target):
            tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
            if not any(isinstance(t, dict) and isinstance(t.get("cache_control"), dict) for t in tools):
                for tool in reversed(tools):
                    # Schema entries only — skips an appended openrouter:web_search tool.
                    if isinstance(tool, dict) and (
                        isinstance(tool.get("function"), dict)
                        or tool.get("input_schema") is not None
                    ):
                        tool["cache_control"] = {"type": "ephemeral"}
                        breakpoints = self._payload_cache_breakpoints(payload)
                        break
            declared = len(breakpoints)
            if declared > self._MAX_CACHE_BREAKPOINTS:
                for holder in breakpoints[self._MAX_CACHE_BREAKPOINTS:]:
                    holder.pop("cache_control", None)
                breakpoints = breakpoints[:self._MAX_CACHE_BREAKPOINTS]
                note = {"declared": declared, "kept": len(breakpoints),
                        "dropped": declared - len(breakpoints)}
            from ouroboros.config import resolve_prompt_cache_ttl

            global_ttl = resolve_prompt_cache_ttl()
            if global_ttl in _VALID_CACHE_TTLS:
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = global_ttl
            if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = "1h"
        if not hasattr(self, "_cache_breakpoint_tls"):
            self._cache_breakpoint_tls = threading.local()
        self._cache_breakpoint_tls.pending = note
        # Report the strongest APPLIED TTL — the value that flows into usage metadata
        # (llm_usage/llm_round events) and prices the write tier. Readers consume this
        # recorded fact; nothing re-derives an "effective TTL" from the route.
        if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
            return "1h"
        if any(str(b["cache_control"].get("ttl") or "") == "5m" for b in breakpoints):
            return "5m"
        return "default" if breakpoints else None

    def _stage_reasoning_pin_disclosure(self, candidate: Dict[str, Any]) -> None:
        """Stage the pin fact on send SUCCESS so it describes the TERMINAL sent
        candidate (the recovery ladder can strip and unpin). Only a wire
        ``allow_fallbacks=false`` over a genuinely sealed transcript reports; an
        owner ``repro`` pin on a portable transcript is never laundered in."""
        from ouroboros.reasoning_artifacts import (
            _REASONING_PIN_CVAR,
            sealed_reasoning_pin_fact,
        )

        extra_body = candidate.get("extra_body")
        provider = extra_body.get("provider") if isinstance(extra_body, dict) else None
        pinned = isinstance(provider, dict) and provider.get("allow_fallbacks") is False
        _REASONING_PIN_CVAR.set(sealed_reasoning_pin_fact(
            candidate.get("messages") or [], candidate.get("model"),
        ) if pinned else None)
