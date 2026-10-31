"""Task-bound facts about the primary route while another route acts.

A host-driven switch renders its facts once, from what the host recorded: both bindings,
the triggering attempt's own failure, the answering call's own money and the primary's last
accepted call with the binding it answered on. Observations are changed, dated,
non-generating metadata at natural round boundaries; cadence bounds that work independently
of retry cooldown. The mind chooses whether to return; only an accepted real response
records a route transition.
"""
from __future__ import annotations

import json
import time

OBSERVATION_INTERVAL_SEC = 120.0
OBSERVATION_TIMEOUT_SEC = 3.0
#: The primary's own last accepted call (``_last_round_cache_usage`` shape) and the ``binding`` it answered on.
PRIMARY_CALL_KEY = "_primary_accepted_call"


def failure_facts(usage: dict) -> dict:
    """The last attempt's own record, independent of the recovery wait's mutable error kind.

    Producers retain their row by reference; each new attempt clears it before admission.
    A refusal without a row keeps only its own code and words, never an older timestamp.
    """
    if failure := usage.get("_last_llm_failure"):
        return dict(failure)
    kind, error = str(usage.get("_last_llm_error_kind") or ""), usage.get("_last_llm_error")
    return {key: value for key, value in (("failure_code", kind), ("error", error)) if value}


def accepted_call(usage: dict, round_idx) -> dict:
    """This round's accepted call: its recorded usage facts, plus its call id when its meta is that same call."""
    call, meta = usage.get("_last_round_cache_usage") or {}, usage.get("_last_llm_call_meta") or {}
    if round_idx is None or not call or call.get("round") != round_idx:
        return {}
    same = meta.get("round") == round_idx and meta.get("llm_call_id") and not meta.get("failure_code")
    return {**call, **({"llm_call_id": meta["llm_call_id"]} if same else {})}


def _call_money(call: dict) -> str:
    """One accepted call's own cost and billing as its producer stated them; unknown never reads as zero."""
    cost, billing = call.get("cost"), call.get("billing")
    if call.get("provider") == "local":
        return "local route, no provider charge"
    known = isinstance(cost, (int, float)) and not isinstance(cost, bool)
    text = (f"cost ${float(cost):.4f} {'reported' if call.get('cost_final') is True else 'estimated'}"
            if known else "cost unknown")
    return text + (f", engine billing {billing}" if isinstance(billing, str) and billing else ", billing basis not reported")


def route_facts_text(tool_ctx, *, model: str, use_local: bool, role: str, failed_model: str,
                     failure: dict, waiter, usage: dict | None = None, round_idx: int | None = None) -> str:
    """Facts after a host-driven switch away from the primary; the acting model decides what follows.

    Each route is named by the binding its send applies, a wait-card choice included. The
    answering call's own usage counts only when this round recorded it, its tokens as reported;
    the primary's evidence is its last accepted call on its current binding, an earlier binding
    named as one, or its absence. Tariffs come from the cache only, for the named binding at the
    answering input's reported size. Rendered once: a later price or ledger correction never
    rewrites what was sent.
    """
    from ouroboros.event_usage import REPORTED_TOKENS_KEY, event_token_counts
    from ouroboros.loop_messages import cached_tariff_text
    from ouroboros.model_slots import route_binding
    from ouroboros.provider_models import provider_for_model

    primary = getattr(tool_ctx, "primary_route", None)
    if not isinstance(primary, dict) or not primary.get("model"):
        return ""
    overrides = waiter.overrides if waiter else None
    target = route_binding(primary["model"], bool(primary["use_local"]), primary["role"], overrides=overrides)

    def label(binding, route_role: str) -> str:
        name, local, account = binding
        if local:
            return f"{name} (local) (role {route_role})"
        provider = provider_for_model(name)  # the configured transport, never a payment basis
        account_text = f"; account {account or 'Auto'}" if provider == "claudexor" else ""
        return f"{name} (role {route_role}{account_text}) via {provider}"

    usage = usage or {}
    call = accepted_call(usage, round_idx)
    reported = event_token_counts(call) if isinstance(call.get(REPORTED_TOKENS_KEY), dict) else {}

    def tariff(name: str, local: bool, provider: str = "") -> str:
        provider = provider or ("local" if local else provider_for_model(name))
        return "" if provider == "local" else "; " + cached_tariff_text(name, provider, reported.get("prompt_tokens"))

    if not call:
        answer = "That answering call's own usage was not recorded."
    else:
        counts = ", ".join(f"{name} {'unknown' if reported.get(key) is None else format(reported[key], ',')}" for name, key
                           in (("input", "prompt_tokens"), ("cache read", "cached_tokens"), ("cache write", "cache_write_tokens")))
        answer = (f"That answering call{' ' + call['llm_call_id'] if call.get('llm_call_id') else ''}: "
                  f"{_call_money(call)}; tokens {counts}"
                  f"{tariff(str(call.get('model') or model), use_local, str(call.get('provider') or ''))}.")
    prior, current = usage.get(PRIMARY_CALL_KEY) or {}, tariff(target[0], target[1]) + "."
    where = f"(round {prior.get('round')}{', ' + prior['llm_call_id'] if prior.get('llm_call_id') else ''})"
    answered_on = prior.get("binding")
    if not prior:
        before = "No accepted primary call is retained in this task's route-facts record" + current
    elif isinstance(answered_on, list) and answered_on == list(target):
        before = f"The primary's own last accepted call in this task {where}: {_call_money(prior)}" + current
    else:  # another binding's call, or one recorded without its binding, is no evidence about this one
        before = ("The one retained primary call does not establish an accepted call on its current binding" + current
                  + (f" Its last accepted call was on the earlier binding {label(answered_on, primary['role'])} {where}"
                     if isinstance(answered_on, list) else f" Its last accepted call {where} has no recorded binding")
                  + f": {_call_money(prior)}.")
    code, reset = str(failure.get("failure_code") or "an error"), str(failure.get("reset_at") or "")
    detail = ", ".join(template.format(failure[key]) for key, template in (  # the recorded code may be a type name
        ("status_code", "HTTP {}"), ("provider_code", "code {}"), ("resource_refusal", "resource refusal {}"))
        if failure.get(key))
    meaning = {"provider_outcome_unknown": " That attempt's outcome and cost stay unknown.",
               "transport_unavailable": " The complete request did not reach the provider; generation did not start."}.get(code, "")
    error = (f" Recorded error text: {json.dumps(str(failure['error']), ensure_ascii=False)}."
             if failure.get("error") else "")
    return (f"[ROUTE FACTS] This task's primary route is {label(target, primary['role'])}; "
            f"{label(route_binding(model, use_local, role, overrides=overrides), role)} answered this round after "
            f"{failure.get('model') or failed_model} failed with {code}{f' ({detail})' if detail else ''} at {failure.get('ts') or 'an unrecorded time'}"
            f"{f' (its engine dated a reset at {reset})' if reset else ''}.{meaning}{error} {answer} {before} "
            "Facts, not an instruction: stay on this route, return with switch_model(primary=\"return\") (the next real "
            "request tests the primary; a refusal there moves on to configured routes again), or return and wait for it "
            "with switch_model(primary=\"wait\"). No timer returns this task; a primary-route observation, when shown, "
            "is non-generating metadata and never proves that the primary generates.")


def _settled(facts: dict) -> dict:
    """What a change means: a quota row's newer observation date alone is no new fact."""
    quota = facts.get("quota") if isinstance(facts, dict) else None
    if not isinstance(quota, dict):
        return facts
    return {**facts, "quota": {**quota, **{key: [{k: v for k, v in row.items() if k != "observed_at"}
                                                for row in quota.get(key) or []] for key in ("snapshots", "absences")}}}


def observe_primary_route(ctx) -> str:
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.llm_probe import upstream_transport_reachable
    from ouroboros.loop_transport import task_deadline_epoch, transport_repeat_stop_requested
    from ouroboros.model_slots import route_binding, task_model_binding
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec
    from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso

    owner = ctx.tools._ctx
    primary = getattr(owner, "primary_route", None)
    if (ctx.defer_resource_wait is not None or getattr(owner, "exact_model_route", False)
            or not isinstance(primary, dict) or not primary.get("model")):
        return ""
    waiter = current_model_wait()
    overrides = waiter.overrides if waiter else None
    role, _ = task_model_binding({"task_metadata": getattr(owner, "task_metadata", {})},
                                context_fit_plan=ctx.context_fit_plan, overrides=overrides)
    target = route_binding(primary["model"], bool(primary["use_local"]), primary["role"], overrides=overrides)
    acting = route_binding(ctx.active_model, ctx.active_use_local, role, overrides=overrides)
    if target == acting:
        return ""
    previous = getattr(owner, "_primary_route_observation", None) or {}
    now = time.monotonic()
    if previous.get("binding") == target and now - previous["checked_monotonic"] < OBSERVATION_INTERVAL_SEC:
        return ""

    def stopped():
        return transport_repeat_stop_requested(owner) or bool(waiter and waiter.control_reason())

    bounds = [dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(ctx.tools), reserve_sec=0),
              dispatch_deadline_remaining_sec(), waiter.execution_window_remaining() if waiter else None]
    timeout = min([OBSERVATION_TIMEOUT_SEC] + [value for value in bounds if value is not None])
    if timeout <= 0 or stopped():
        return ""
    quota = None
    if target[1]:
        from ouroboros.local_model import get_manager
        try:
            local = get_manager().serving_context_evidence() or {}
        except Exception:
            local = {}
        observed = {"kind": "local_serving_metadata", "source": local.get("source")} if local.get("confirmed") else {}
    else:
        from ouroboros.provider_models import parse_claudexor_model, provider_for_model
        if provider_for_model(target[0]) == "claudexor":
            from ouroboros.claudexor_observations import quota_observation
            # Quota first: a read inside the engine, so the catalog's upstream discovery
            # cannot starve it; both share this one budget, and either may fail alone.
            try:
                source, native = parse_claudexor_model(target[0])
            except ValueError:  # no engine binding to read; the catalog check reports it unconfirmed
                source = native = ""
            quota = quota_observation(source, native, target[2], timeout_sec=timeout) if source else None
            if stopped():
                return ""
        remaining = timeout - (time.monotonic() - now)
        observed = upstream_transport_reachable(
            ctx.llm, target[0], timeout=remaining, model_role=primary["role"], account_override=target[2],
            observed_after=time.time(), expected_route=None) if remaining > 0 else {}
    if stopped():
        return ""
    facts = {key: observed[key] for key in ("kind", "status_code", "source", "provenance",
             "credential_profile_id", "account_fingerprint") if key in observed}
    facts.setdefault("kind", "local_not_observed" if target[1] else "unconfirmed")
    if quota is not None:
        facts["quota"] = quota
    stamp = observed.get("observed_at") or utc_now_iso()
    owner._primary_route_observation = {"binding": target, "checked_monotonic": now,
                                        "observed_at": stamp, "facts": facts}
    if previous.get("binding") == target and _settled(previous.get("facts")) == _settled(facts):
        return ""
    row = {"primary_binding": target, "observed_at": stamp, "facts": facts, "generation_tested": False}
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("primary_route_observations", []).append(row)
    return sanitize_tool_result_for_log(
        f"[PRIMARY ROUTE OBSERVATION] {json.dumps(row)}. Earlier observations are historical. "
        "Metadata does not establish generation recovery."
        + (" Quota facts are the engine's quota-only reading: unknown is not exhaustion and available is not admission."
           if quota is not None else "")
        + ' Stay on the acting route or choose switch_model(primary="return") / primary="wait". '
        "Return reprojects context and may change cache reuse.")


def record_primary_return_request(owner, model: str, use_local: bool, role: str) -> None:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.utils import utc_now_iso

    waiter = current_model_wait()
    row = {"requested_binding": route_binding(model, use_local, role, overrides=waiter.overrides if waiter else None),
           "status": "requested", "at": utc_now_iso()}
    owner._primary_return_requested = row
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("primary_return_requests", []).append(row)


def record_round_route_result(ctx, role: str, *, accepted: bool) -> None:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.utils import utc_now_iso

    owner = ctx.tools._ctx
    waiter = current_model_wait()
    binding = route_binding(ctx.active_model, ctx.active_use_local, role, overrides=waiter.overrides if waiter else None)
    observed = dict(ctx.accumulated_usage.get("_model_route") or {})
    request = getattr(owner, "_primary_return_requested", None)
    owner._primary_return_requested = None
    if request:
        # Only the requested route's own accepted reply is a return; a reply that a
        # fallback or wait reroute produced in that round answered on another route.
        status = ("no_accepted_response" if not accepted else "accepted_response"
                  if tuple(binding) == tuple(request["requested_binding"]) else "answered_by_other_route")
        outcome = {**request, "actual_binding": binding, "model_route": observed, "at": utc_now_iso(), "status": status}
        ctx.accumulated_usage.setdefault("primary_return_results", []).append(outcome)
        if isinstance(getattr(owner, "_execution_trace", None), dict):
            owner._execution_trace.setdefault("primary_return_results", []).append(outcome)
    if not accepted:
        return
    primary, call = getattr(owner, "primary_route", None), accepted_call(ctx.accumulated_usage, getattr(ctx, "round_idx", None))
    if (isinstance(primary, dict) and primary.get("model") and call
            and tuple(binding) == route_binding(primary["model"], bool(primary["use_local"]), primary["role"],
                                                overrides=waiter.overrides if waiter else None)):
        ctx.accumulated_usage[PRIMARY_CALL_KEY] = {**call, "binding": list(binding)}  # its own facts and binding
        owner._primary_view = None  # the primary answered on its own view: nothing is held for a return any more
    previous = getattr(owner, "_accepted_route_binding", None)
    previous_observed = getattr(owner, "_accepted_model_route", None)
    owner._accepted_route_binding = binding
    owner._accepted_model_route = observed
    if previous == binding and previous_observed == observed:
        return
    row = {"from_binding": previous, "to_binding": binding, "status": "accepted_response", "at": utc_now_iso(),
           "model_route": observed}
    ctx.accumulated_usage.setdefault("accepted_routes", []).append(row)
    if isinstance(getattr(owner, "_execution_trace", None), dict):
        owner._execution_trace.setdefault("accepted_routes", []).append(row)


def held_primary_view(owner, plan, start_mode, *, model: str, use_local: bool, role: str):
    """The plan and start mode a route switch rebinds from.

    The first departure from the task's primary holds the view it showed there, keyed to the primary's model and
    locality: rebinds before adoption (a walk candidate, its observed account, a wait card's choice) still read the
    primary as acting and never replace it. A return to the primary resumes the held view, so the narrowing another
    route's window wrote into every book leaves with that route, while the primary's own lowering (its start fit, a
    real overflow before it left) and my forms selected meanwhile (``hold_selection``) stay, and the window of the
    account now serving it may narrow the view again. Accounts are not part of this identity: one model shows one
    view, and an account is not reliably known while a walk or a wait card is still switching. So a fallback on the
    primary's model with another account keeps the view it is shown, as before. The primary and the target are the
    models their roles resolve to under the owner's wait-card choices, while the acting pair is the route that showed
    the view (a card writes its choice before this rebind). A switch on the same route (account rotation, a wait's
    reprepare) or to a primary whose model changed meanwhile never raises the mode. The hold is in memory only;
    after a restart a return keeps the acting view.
    """
    primary = getattr(owner, "primary_route", None)
    if not isinstance(primary, dict) or not primary.get("model"):
        return plan, start_mode
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait

    overrides = getattr(current_model_wait(), "overrides", None)
    home = route_binding(str(primary["model"]), bool(primary["use_local"]), primary["role"], overrides=overrides)[:2]
    acting = (str(getattr(owner, "active_model", "") or ""), bool(getattr(owner, "active_use_local", False)))
    target = route_binding(str(model or ""), bool(use_local), role, overrides=overrides)[:2]
    held = getattr(owner, "_primary_view", None)
    if held is not None and (held[0] != home or held[1].core_sha256 != plan.core_sha256):
        held = owner._primary_view = None  # another primary or core: that view is no longer the primary's own
    if held is None and acting == home != target:
        owner._primary_view = (home, plan)
    elif held is not None and target == home != acting:
        return held[1], held[1].initial_mode
    return plan, start_mode


def hold_selection(owner, addresses, *, owner_note: bool) -> None:
    """Forms I select while another route acts stay mine on the held primary view too; a form the held view
    cannot take releases the hold, so a return keeps the acting view with my choice instead of losing it."""
    held = getattr(owner, "_primary_view", None)
    if held is None or not addresses:
        return
    from ouroboros.memory_floor import select_view

    try:
        owner._primary_view = (held[0], select_view(held[1], held[1].initial_mode, addresses, owner_note=owner_note))
    except ValueError:
        owner._primary_view = None
