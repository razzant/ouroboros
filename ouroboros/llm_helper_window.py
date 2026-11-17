"""Reply allowance at the final JSON candidate, with route-bound helper calibration.

Main carries its measured context; standalone chat helpers read cached route facts.
The pure Nano/Low/Max formula remains in context_budget.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict

#: ``(candidate, context)``: the helper context ``bound_reply_allowance`` priced for that exact candidate object.
_PRICED: ContextVar = ContextVar("helper_priced_candidate", default=(None, None))


def priced_for(candidate, context, *, source=None):
    """Bind ``context`` to ``candidate``, or carry the binding of ``source`` to it (none: unbound); returns ``candidate``."""
    bound, priced = _PRICED.get()
    if source is None:
        _PRICED.set((candidate, context))
    else:
        _PRICED.set((candidate, priced) if bound is source else (None, None))
    return candidate


def helper_context_for(target: Dict[str, Any], payload: Dict[str, Any]):
    """The helper context of this final candidate: the one already priced for it (read once), else one build."""
    bound, context = _PRICED.get()
    if bound is not payload:
        return _helper_attempt_context(target, payload)
    _PRICED.set((None, None))
    return context


def _helper_attempt_context(target: Dict[str, Any], payload: Dict[str, Any]):
    """Known API-route capacity for a call without a Main view; reads cached evidence only.

    The helper's own usage calibrates the helper view, never Main's memory view.
    A missing/stale window keeps the existing unknown-window behavior. Subscription
    dispatch and its account binding remain owned by the native lane. A body is read
    as every estimator reads it (``context_fit.payload_prompt_rows``: ``messages`` or a
    Responses ``input``). A connection's API is no part of the capability identity: the
    window belongs to the model behind the endpoint, and each density witness compares
    the provider's count with the estimator's count of that same physical body.
    """
    from ouroboros import capability_evidence as ce
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload, payload_prompt_rows
    from ouroboros.context_window import resolve_window, route_observation_scope
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    provider = target.get("provider", "")
    if provider in {"", "claudexor"} or payload_prompt_rows(payload) is None:
        return None  # no inspectable input rows: an engine's opaque body
    # Production targets carry the original route from resolution. Literal legacy
    # targets keep only their own supplied identity; never reconstruct it or read Settings.
    identity = target.get("capability_route") or target
    model = str(identity.get("model") or target.get("requested_model") or target.get("resolved_model") or "")
    root = ce.canonical_evidence_root()
    route = dict(provider=provider, model=model, base_url=identity.get("base_url", ""), options={},
                 use_local=provider == "local")
    # Local dispatch already supplies its serving instance's count/window. Do not
    # query health from the finalizer: this path must never add a network request.
    if provider == "local":
        window = int(target.get("context_window_tokens") or 0) if target.get("context_window_confirmed") else 0
        evidence = ce.CapabilityEvidence(window, ce.STATUS_CONFIRMED if window else ce.STATUS_UNPROBEABLE,
            ce.SOURCE_LOCAL_HEALTH, ce.route_fingerprint(provider=provider, model=model, base_url=route["base_url"]),
            model, provider, ts=ce.utc_now_iso(), window_instance_id=str(target.get("window_instance_id") or ""))
    else:
        evidence = ce.probe(root, provider=provider, model=model, base_url=route["base_url"], allow_fetch=False,
                            target=target, api_key=target.get("api_key", ""))
    scope = route_observation_scope(route, None, target=target)
    record = resolve_window(evidence, 0, root=root, observation_fp=scope)
    # An accepted input is a lower bound, never proof of room for the reply.
    limits = [limit.tokens for limit in (record.input, record.total) if limit and limit.source != "accepted"]
    if not limits:
        return None
    raw = bounded_prompt_tokens_for_payload(payload, 0)
    density, basis = ce.resolve_main_token_density(root, evidence.route_fp, model, estimate_tokens=raw, view_kind="helper")
    return PhysicalAttemptContext(profile="task_local_nano", rendered_mode="nano", measurement_basis=basis,
        route_fp=evidence.route_fp, round_id="", target_total_tokens=None, capacity_total_tokens=min(limits),
        context_target_miss=False, automatic_pass_used=False, measurement_density=density, view_kind="helper",
        window_witness_bound=True, window_observation_fp=scope)


def window_reply_candidate(payload, field, context, window, measured, *, helper):
    """Price reply room in the complete candidate; only an exact count can refuse.

    Helpers keep intact input on an estimated shortfall. A required handbook
    retains the reply allowance of the candidate without it; admission charges both.
    """
    import math

    from ouroboros.context_budget import exact_reply_shortfall, reply_allowance_tokens
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload
    from ouroboros.request_wire_recovery import registered_source_payload

    exact = bool(measured)
    density = 1.0 if exact else float(getattr(context, "measurement_density", None) or 1.0)
    if exact:
        raw = input_tokens = int(measured["input_tokens"])
        window = measured.get("context_window") or window
    else:  # the estimator's own reading: Chat/Messages or Responses rows (payload_prompt_rows) and tool schemas
        source = registered_source_payload(payload) or payload
        raw = bounded_prompt_tokens_for_payload(source, 0)
        input_tokens = math.ceil(raw * density)
    frame = dict(caller_max_tokens=int(payload[field]), nano=True, input_tokens=input_tokens, window_tokens=window)
    if helper and not exact and exact_reply_shortfall(**{**frame, "input_tokens": max(input_tokens, raw)}):
        return payload, False, input_tokens, window
    book = int(getattr(context, "required_book_tokens", 0) or 0)
    candidate = {**payload, field: reply_allowance_tokens(owner_nano=context.profile == "owner_nano", exact=exact,
        **{**frame, "input_tokens": input_tokens - book}, raw_input_tokens=raw - book / density)}
    return candidate, exact and exact_reply_shortfall(**frame), input_tokens, window
