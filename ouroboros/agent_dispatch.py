"""The delegated-child dispatch seam of the agent (v7 L-C2 split).

Everything the agent runs between taking a task and entering the loop when the
task is (or may be) a delegated child, plus the pre-loop owner surfaces beside
it: the one dispatch-axes resolution and its durable/supervisor mirrors, the
delegate-visibility preflight, the executor disclosures composed into the
child's prompt, the nanny-economics mark reset, the budget-rail owner messages,
and the early origin persistence. Extracted from agent.py; agent.py re-exports
every name, so historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import json
import logging
import pathlib

from typing import Any, Dict, Optional, Tuple
from ouroboros.agent_startup_checks import persist_early_origin_stub as _persist_early_origin_stub_impl
from ouroboros.config import EFFORT_SCALE, resolve_effort, runtime_setting
from ouroboros.route_choice import model_role_slot_override  # the role slot a wake follows; resolved per call there
from ouroboros.subagents import CapabilityDelta, SUBAGENT_RESOLUTION_FIELDS, SubagentDispatch, SubagentExecutorResolution, SubagentLaneResolution, capability_delta_disclosures, resolve_subagent_dispatch
from ouroboros.task_results import STATUS_RUNNING
from ouroboros.utils import append_jsonl, utc_now_iso


log = logging.getLogger("ouroboros.agent")


def _agent():
    """The parent agent module, read at call time.

    The agent's members stay monkeypatch-addressable at their historical
    ``ouroboros.agent`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import agent

    return agent


def dispatch_executor_note(decision: Optional[SubagentExecutorResolution],
                           lane: Optional["SubagentLaneResolution"] = None,
                           *, cognitive_route: Optional[Dict[str, Any]] = None) -> str:
    """The configured nanny's VISIBLE marker for its substrate ('' = silent).

    Only a configured agent-session row dispatches onto the delegated substrate (its
    lane says ``configured_subagent``), and its nanny must know the host starts the
    exact leaf and that its own rounds are for judgment. Every other dispatch, an
    API row or a row-less child, runs natively and has nothing to announce. The note
    rides only the FINAL post-preflight dispatch (the call site runs after the
    delegate-visibility preflight), so a blocked child never receives it.
    """
    if (decision is None or decision.executor != "harness"
            or lane is None or lane.provenance != "configured_subagent"):
        return ""
    route = decision.route.route_id if decision.route else ""
    from ouroboros.supervisor_route import cognitive_route_provenance

    provenance = cognitive_route_provenance(cognitive_route)
    cognition = (f" Your cognitive route is {lane.model} ({provenance}); "
                 "its calls bill as that route bills. You may change it with switch_model. "
                 if provenance else " ")
    return (
        f"EXECUTOR: your parent selected the configured agent-session route ({route}). "
        "You are its Ouroboros NANNY." + cognition + "For fresh work the host starts the exact leaf "
        "before your first round. Restored supervision follows its continuation receipt; "
        "the receipt states custody, and waiting is your own call "
        "(delegate_wait when you want its facts). Your "
        "supervisory rounds are for judgment — verify and integrate what the run "
        "produces, answer its authorized questions, decide recovery — never for "
        "rebuilding its work: co-building beside a delegated run is a "
        "duplicate of already-paid work. After a terminal failure, replace through "
        "delegate_start(prompt='') only once cancellation and terminal settlement are "
        "verified; an unknown start outcome is retried with "
        "delegate_start(retry_of=..., prompt=''). Auxiliary or API-backed work must be an "
        "explicit separate child, so its authorship and spend remain visible. When "
        "no physical run exists and none can be started, record the typed zero-run "
        "terminal through verify_and_record(contract_kind=delegation_zero_run, "
        "zero_run_decision=incomplete|unknown, zero_run_basis=...); that receipt is "
        "terminal for this actor — after it is durably recorded, do not start a "
        "physical leaf in the same task."
    )


def executor_blocked_outcome(
    decision: SubagentExecutorResolution,
    *,
    availability: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The terminal ``(text, usage)`` of a child the host could not start: one notice for API and session rows.

    Deliberately NOT a fallback: the task ends unrun and typed, having spent nothing. The text and the usage carry
    the same facts (the cause, the producer's detail, the reset instant, the current alternatives, not ranked), so
    the parent reads why and what it may choose instead (#1241); ``reason_code`` keeps its codes.
    """
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE

    availability = availability if isinstance(availability, dict) else {}
    api_row = availability.get("route_kind") == "api_model" or (
        decision.requested == "native" and decision.reason == "credentials_unavailable")
    reason = "credentials_unavailable" if api_row else str(decision.reason or "configured_session_unavailable")
    verbs = "delegate tools (delegate_start/delegate_wait/delegate_cancel)"
    # Q1A preflight (2026-08-10): a healthy route whose child toolset hides the verbs, or whose introspection
    # failed (visibility UNKNOWN, not disproven); the fix is tool policy, not waiting for the route.
    cause = {
        "credentials_unavailable": "its API-model route has no usable credentials",
        "delegate_tools_invisible": f"the {verbs} are not visible in its materialized toolset; fix the tool "
                                    "policy or task contract that hides them",
        "delegate_visibility_unverified": f"the toolset introspection failed, so the visibility of the {verbs} "
                                          "could not be verified",
    }.get(reason, f"its route cannot run: {reason}")
    alternatives = availability.get("alternatives") if isinstance(availability.get("alternatives"), list) else []
    text = (
        f"⚠️ SUBAGENT_UNAVAILABLE: this subagent could not start: {cause}."
        + (f" It resets at {decision.reset_at}." if decision.reset_at else "")
        # The producer's own words (a refused snapshot provision names the lock holder), not only a code (#1241).
        + (f" {availability['detail']}" if availability.get("detail") else "")
        + " The task was NOT run and spent nothing; the host did not substitute another model, route or substrate."
        + (" Configured alternatives now (not ranked): " + json.dumps(alternatives, ensure_ascii=False, sort_keys=True)
           if alternatives else "")
    )
    # The codes stay literal (the provenance drift guard reads them from this keyword).
    return text, dict(
        execution_status="infra_failed", unavailable_reason=reason, alternatives=alternatives, host_fallback=False,
        terminal_origin=TERMINAL_ORIGIN_HOST_NOTICE,
        reason_code=("delegate_tools_invisible" if reason == "delegate_tools_invisible"
                     else "delegate_visibility_unverified" if reason == "delegate_visibility_unverified"
                     else "subagent_executor_unavailable"),
    )


def _record_executor_resolution(
    drive_logs: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> None:
    """Durably record the typed substrate decision (re-homed from the retired
    `_announce_dispatch_executor`): who was asked for, who runs it, why, and —
    when every plan window is spent — the instant it heals."""
    if dispatch is None or dispatch.executor_resolution is None:
        return
    res = dispatch.executor_resolution
    row = {
        "ts": utc_now_iso(), "type": "subagent_executor_resolved",
        "task_id": str(task.get("id") or ""),
        "requested": res.requested,
        "executor": res.executor,
        "reason": res.reason,
        "reset_at": res.reset_at,
        "route": res.route.route_id if res.route else "",
    }
    append_jsonl(drive_logs / "events.jsonl", row)
    # ALSO the canonical events log: a delegated child's forked drive is pruned
    # with the task, so this used to be the ONLY copy of the substrate decision
    # (submarine forensics: zero subagent_executor_resolved rows in the canonical
    # events.jsonl). The accounting axis the task already carries names the
    # canonical root; the root agent's own drive IS canonical, so skip the dup.
    try:
        budget_root = str(task.get("budget_drive_root") or "").strip()
        if budget_root:
            canonical_logs = pathlib.Path(budget_root) / "logs"
            if canonical_logs.resolve(strict=False) != pathlib.Path(drive_logs).resolve(strict=False):
                append_jsonl(canonical_logs / "events.jsonl", row)
    except Exception:
        log.debug("Failed to mirror executor resolution to canonical events", exc_info=True)
    # D28 exhaustion beacon: surface the spent-window fact to the WAITING parent
    # NOW (via the task-tree attention channel the wait tools already poll),
    # not at absorption after the wait window burned.
    if res.reason == "subscription_window_exhausted" and str(task.get("parent_task_id") or "").strip():
        root_id = str(task.get("root_task_id") or "").strip()
        if root_id:
            try:
                from ouroboros.task_tree_ledger import record_subscription_window_exhausted

                record_subscription_window_exhausted(
                    root_id,
                    child_task_id=str(task.get("id") or ""),
                    reset_at=res.reset_at,
                    route=res.route.route_id if res.route else "",
                    executor=res.executor,
                )
            except Exception:
                log.debug("Failed to append subscription-window beacon", exc_info=True)


def _blocked_executor_terminal(cap_info: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """p34's typed terminal for a blocked executor pin, rebuilt from the facts
    cap_info carried across the (ctx, messages, cap_info) seam. The placeholder
    method p2 kept for exactly this synthesis is deleted; this is the one body."""
    text, usage = executor_blocked_outcome(SubagentExecutorResolution(
        requested=str(cap_info.get("executor_blocked_requested") or "harness"),
        executor="blocked",
        reason=str(cap_info.get("executor_blocked_reason") or ""),
        reset_at=str(cap_info.get("executor_blocked_reset_at") or ""),
    ), availability=(task or {}).get("subagent_availability")
        if isinstance((task or {}).get("subagent_availability"), dict) else {})
    return text, usage, {"reasoning_notes": ["subagent_executor_unavailable"], "tool_calls": []}


def _context_capacity_terminal(unfit: Any) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """The named outcome of a known window that cannot hold a pooled task's start (``MinimumViewUnfit``).

    In the reader's words: what the start needs against the route's limit, the step that changes it (for a
    directly required handbook, the choices that remain), then the measured parts; the host switches nothing. Money
    stays the ledger's answer, and an earlier attempt's or started run's effects stay theirs: this attempt's own
    loop evidence is absent, never a zero, so nothing here claims the task had no effect.
    """
    from ouroboros.context_budget import REQUIRED_BOOK_CHOICES
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE

    # The comparison the fitter refused on: input plus the reserve kept for the reply and the work, against the
    # route's limit. The further room a start wants is no threshold (a start with less runs): named only as measured.
    # The paths named are the ones every surface keeps: the room's model button, Settings › Models, Context Length.
    # A window is a role's setting: a chat with its own model has none to correct, so only Main's is named.
    limit, book = unfit.window_tokens, unfit.required_book_tokens
    needs = -(-(unfit.estimated_input_tokens + unfit.response_reserve_tokens) // 1000) * 1000  # "about": rounded up
    text = (f"⚠️ This model's context window is too small to start this work: the start needs about {needs:,} tokens "
            f"(its input plus the room kept for the reply and the work), and the model's limit here is {limit:,}.")
    if book:
        text += (f" This task directly requires the whole development handbook (~{book:,} tokens); without it the "
                 "start fits. " + REQUIRED_BOOK_CHOICES.format("; a Continue keeps it"))  # a Continue cause
    else:
        text += (" Choose a model with a larger window and send the request again: in the app, the model button under "
                 "the message box sets it for this chat, and Settings › Models sets the default for chats that follow "
                 "Main. For a local model, raise its Context Length in Settings and restart it. If this chat follows "
                 "Main and the model's real limit is higher, set Main's window in Settings › Models.")
    inside = [("core, request and tools", unfit.fixed_input_tokens - book), ("handbook", book),
              ("further memory", unfit.memory_view_tokens)]
    text += (f"\n\nMeasured, in tokens: input {unfit.estimated_input_tokens:,} ("
             + "; ".join(f"{name} {value:,}" for name, value in inside if value)
             + f") · reserve for the reply and the work {unfit.response_reserve_tokens:,} · limit {limit:,}"
             + (f" · further room the start wanted {unfit.work_room_tokens:,}" if unfit.work_room_tokens else "") + ".")
    usage = {"execution_status": "infra_failed", "reason_code": "context_capacity",
             "terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE, "loop_evidence_unavailable": True}
    return text, usage, {"reasoning_notes": ["context_capacity"], "tool_calls": [], "loop_evidence_unavailable": True}


def _persist_early_origin_stub(drive_root: Any, task: Dict[str, Any]) -> None:
    _persist_early_origin_stub_impl(drive_root, task, write_result=_agent().write_task_result)


def _budget_exhausted_message() -> str:
    return (
        "🚫 Model budget exhausted before another dispatch. Increase or reset the "
        "global/root budget, then retry or resume this task. Starting a new run before "
        "changing the exhausted budget will hit the same limit."
    )


def _budget_resume_policy(*, replay_safe: bool, direct_chat: bool) -> str:
    if direct_chat:
        return "increase_or_reset_budget_then_retry"
    if replay_safe:
        return "manual_same_generation"
    return "cancel_or_new_run"


def _queued_budget_exhausted_message() -> str:
    return (
        "🚫 Resource limit reached before another model dispatch. The task was not "
        "auto-resumed; cancel it or start a new run unless the recorded checkpoint "
        "is explicitly replay-safe."
    )


def _physical_calls_after_budget_rail(budget_root: Any, task_id: str) -> Optional[int]:
    """How many provider sends this task really made, for an honest budget-rail message.

    ``None`` means UNKNOWN, and an integrity-degraded ledger yields exactly that rather
    than a count that might be missing a paid tail — "0 calls" and "we cannot tell" must
    not read the same to the owner.
    """
    try:
        from ouroboros.usage_accounting import usage_breakdown

        evidence = usage_breakdown(pathlib.Path(budget_root), task_id=task_id)
        if evidence.get("integrity_degraded"):
            return None
        return int(evidence.get("physical_calls") or 0)
    except Exception:
        log.exception("Could not inspect task attempts after agent budget rail")
        return None


def _initial_effort_for(task: Dict[str, Any], task_type: str) -> str:
    """The effort a task starts on.

    For a delegated child this is what ``resolve_subagent_dispatch`` derived and
    wrote onto the record moments ago — read back rather than recomputed so the
    loop runs the effort the record states. For
    everything else, and for an unrecognized STORED value (durable data outlives the
    schema that wrote it), it is the task-type default directly — or, when the task
    names a model ROLE in its metadata (a consciousness wake-up), that role's effort
    slot: the role owns its effort the way it owns its account binding.
    """
    stored = str(task.get("reasoning_effort") or "").strip().lower()
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if str(task.get("delegation_role") or "").lower() != "subagent":
        # A level encoded in the model name wins over a pin and a default (``choose_effort``).
        named = main_model_named_effort(metadata)
        if named:
            return named
    if stored in EFFORT_SCALE:
        return stored
    role = str(metadata.get("model_role") or "").strip().lower()
    return resolve_effort(role or task_type)


def main_model_named_effort(task_metadata: Dict[str, Any]) -> str:
    """The level the task's own model name carries — its admitted lead, the role slot's
    model or Main's (``route_spec.api_model_named_effort``) — '' when the model names none."""
    from ouroboros.route_choice import task_lead
    from ouroboros.route_spec import api_model_named_effort

    metadata = task_metadata if isinstance(task_metadata, dict) else {}
    carried = task_lead(metadata)
    override = None if carried else model_role_slot_override(metadata)
    model = carried["model"] if carried else (override[0] if override else str(runtime_setting("OUROBOROS_MODEL", "") or ""))
    return api_model_named_effort(model)


def resolve_dispatch_axes(task: Dict[str, Any]) -> Optional[SubagentDispatch]:
    """Resolve WHAT THIS CHILD GETS, once, and stamp it onto the record it came from.

    ``None`` when the task is not a delegated child. This is the ONE place a child's
    model, effort, route, tool profile and effective executor are decided, and the
    one author of its ``capability_delta``. It writes back onto the live task dict so
    every downstream surface — the RUNNING task result, the task-metadata projection
    the loop reads, the completion write, the envelope — describes the SAME
    resolution instead of each re-deriving its own from whatever it happens to hold.
    """
    if str(task.get("delegation_role") or "").lower() != "subagent":
        return None
    dispatch = resolve_subagent_dispatch(task, task_type=str(task.get("type") or "task"))
    task.update(dispatch.record_fields())
    # The envelope is the child's public description, so it is rebuilt from the
    # record the resolution just wrote rather than left holding the requested-status
    # copy the scheduler made — through the ONE record->envelope mapping, so it
    # cannot describe a different child than the record does.
    task["subagent_envelope"] = _agent().envelope_from_task(task, status=STATUS_RUNNING)
    return dispatch


_DELEGATE_VERBS = ("delegate_start", "delegate_wait", "delegate_cancel")


def preflight_delegate_visibility(
    tools: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> Tuple[Optional[SubagentDispatch], bool]:
    """Verify a harness dispatch can actually SEE its delegate verbs — after the
    real toolset is materialized, BEFORE the first paid LLM round.

    The dispatch resolution proves the ROUTE is healthy; it does not prove the
    child's toolset carries the delegate verbs (its delegated-child profile,
    contract disabled_tools, credential/resource availability, or future policy
    drift can hide them). The e9108a09c6574184
    audit: nine children dispatched as nannies with the verbs invisible made zero
    delegated runs and burned ~$29-54 of metered API while telemetry said harness.

    One check at toolset materialization (owner decision Q1A). Only a configured
    agent-session row dispatches onto the harness, and that row is a pin: hidden
    verbs make it the typed blocked outcome that terminalizes with zero spend
    (``executor_blocked_outcome``), with the amended ``capability_delta`` and
    dispatch fields re-stamped onto the task record so telemetry does not lie. A
    broken introspection fails CLOSED the same way: a probe that cannot prove
    visibility cannot prove the pinned contract is executable. Returns the
    (possibly amended) dispatch and whether it amended.
    """
    if (
        dispatch is None
        or dispatch.executor_resolution is None
        or dispatch.executor_resolution.executor != "harness"
    ):
        return dispatch, False
    import dataclasses

    def _stamp(amended: SubagentDispatch) -> Tuple[SubagentDispatch, bool]:
        # The same two writes resolve_dispatch_axes made: the record fields and
        # the envelope rebuilt from them, so every downstream surface describes
        # the amended resolution instead of the one the preflight just falsified.
        task.update(amended.record_fields())
        task["subagent_envelope"] = _agent().envelope_from_task(task, status=STATUS_RUNNING)
        return amended, True

    def _append_reason(delta: CapabilityDelta, note: str, **changes: Any) -> CapabilityDelta:
        from ouroboros.subagents import derive_capability_reason

        # Seed from the legacy string when the typed list is empty but a reason
        # exists (a stored pre-lists delta): rebuilding purely from the list
        # would silently DISCARD that disclosure text (P1).
        base = delta.reduction_reasons or ((delta.reason,) if delta.reason else ())
        reasons = (*base, note)
        return dataclasses.replace(
            delta, reduction_reasons=reasons,
            reason=derive_capability_reason(reasons, delta.substrate_disclosures),
            **changes)

    reason = "delegate_tools_invisible"
    try:
        available = set(tools.available_tools())
        if all(verb in available for verb in _DELEGATE_VERBS):
            return dispatch, False
    except Exception:
        log.warning("delegate visibility preflight: introspection failed", exc_info=True)
        # A broken probe blocks with the honest reason: visibility is UNKNOWN,
        # not disproven.
        reason = "delegate_visibility_unverified"
    return _stamp(dataclasses.replace(
        dispatch,
        executor="blocked",
        route="",
        delta=_append_reason(dispatch.delta, reason,
                             effective_executor="blocked", reduced=True),
        executor_resolution=dataclasses.replace(
            dispatch.executor_resolution,
            executor="blocked", reason=reason, reset_at="",
        ),
    ))


def reset_nanny_economics_marks(ctx: Any, *, route_dispatched: bool, delegate_activity_seed: bool = False) -> None:
    """Reset EVERY nanny-economics mark for a fresh dispatch (F4).

    DEFENSIVE, not load-bearing: ``_prepare_task_context`` builds a FRESH
    ToolContext per task, so nothing stale can leak today — this states the
    marks' lifecycle in one place and keeps it true even if a refactor ever
    reuses a context (leaked marks would misstate the facts line's nanny burn)."""
    ctx._nanny_route_dispatched = bool(route_dispatched)
    ctx._nanny_metered_progress = None
    ctx._nanny_delegate_baseline = ({"round": 0, "cost": 0.0} if delegate_activity_seed else None)
    ctx._nanny_physical_activity_seed = False


def emit_dispatch_resolution(
    event_queue: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> None:
    """Report the dispatch-time resolution back to the supervisor (XG-2R.1).

    ``resolve_dispatch_axes`` stamps the WORKER process's clone of the task; the
    supervisor's RUNNING copy — the one ``persist_queue_snapshot`` serializes — is a
    separate dict made at assignment, so without this report a restart restored the
    unresolved intent and lost `effective_model_lane`, `reasoning_effort`, the
    executor fields and `capability_delta`. The report rides the SAME worker event
    channel every other worker fact uses (no second channel);
    ``supervisor/events.py::_handle_task_dispatch_resolved`` merges exactly
    ``SUBAGENT_RESOLUTION_FIELDS`` into RUNNING under the queue lock. Best-effort by
    design: the durable task_result written moments before this remains the record
    of authority — the merge keeps the supervisor's live mirror and its snapshot
    telling the same story.
    """
    if dispatch is None or event_queue is None:
        return
    try:
        event_queue.put({
            "type": "task_dispatch_resolved",
            "task_id": str(task.get("id") or ""),
            "resolution": {
                key: task.get(key) for key in SUBAGENT_RESOLUTION_FIELDS if key in task
            },
            "ts": utc_now_iso(),
        })
    except Exception:
        log.debug("Failed to report dispatch resolution to the supervisor", exc_info=True)


def capability_delta_prompt_block(dispatch: Optional[SubagentDispatch]) -> str:
    """What the CHILD is told about the gap between what was asked and what it got.

    The child is the only actor that can say "I could not do this well at this
    strength", and it cannot say so about a fact it was never given. Composed here,
    at dispatch, because that is when the fact exists: the supervisor builds the
    child's prompt text before the child is admitted, so a reduction discovered when
    the child actually starts could never reach that copy.
    """
    if dispatch is None:
        return ""
    delta = dispatch.delta.as_dict()
    parts: list[str] = []
    disclosures = capability_delta_disclosures(delta) if delta.get("reduced") else []
    if disclosures:
        # `reduced` with NO disclosable axis (an executor-only reduction) renders
        # nothing here: "BELOW what your parent asked for:" over an empty list
        # told the child nothing and read as a broken sentence.
        # The parenthetical carries the typed DISPATCH axes only (B4): substrate
        # facts are completion-seam and never fuse into this dispatch sentence
        # (a fresh resolution carries none anyway).
        reduction = delta.get("reduction_reasons")
        reason_text = (
            "; ".join(reduction) if isinstance(reduction, list) and reduction
            else (delta.get("reason") or "unspecified")
        )
        action = (
            "Do the work anyway — routed through your delegated run "
            "(delegate_start / delegate_wait), not your own rounds — but say "
            if delta.get("effective_executor") == "harness"
            else "Do the work anyway, but say "
        )
        parts.append(
            "You are running BELOW what your parent asked for: "
            + "; ".join(disclosures)
            + f" ({reason_text}). " + action
            + "so in your answer if the gap actually limited it — do not quietly "
            "return a weaker result as if it were full strength."
        )
    if delta.get("legacy_note"):
        parts.append(f"Ignored on your record: {delta['legacy_note']}.")
    # The effort decision, when it moved or set aside the parent's request or a pin/name decided.
    from ouroboros.settings_scales import effort_fact_phrase

    fact = getattr(dispatch, "effort_fact", None)
    phrase = effort_fact_phrase(fact) if isinstance(fact, dict) and fact else ""
    if phrase:
        whose = "Your delegated run's" if delta.get("effective_executor") == "harness" else "Your"
        parts.append(f"{whose} {phrase}.")
    return "[CAPABILITY DELTA]\n" + "\n".join(parts) if parts else ""
