"""The delegated-child dispatch seam of the agent (v7 L-C2 split).

Everything the agent runs between taking a task and entering the loop when the
task is (or may be) a delegated child, plus the pre-loop owner surfaces beside
it: the one dispatch-axes resolution and its durable/supervisor mirrors, the
delegate-visibility preflight, the executor disclosures composed into the
child's prompt, the nanny-economics mark reset, the budget-rail owner messages,
and the early origin persistence. Extracted from agent.py; agent.py re-exports
every name, so historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

from ouroboros.config import EFFORT_SCALE, resolve_effort
from ouroboros.subagents import (
    CapabilityDelta,
    SUBAGENT_RESOLUTION_FIELDS,
    SubagentDispatch,
    SubagentExecutorResolution,
    SubagentLaneResolution,
    capability_delta_disclosures,
    envelope_from_task,
    resolve_subagent_dispatch,
)
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
                           lane: Optional["SubagentLaneResolution"] = None) -> str:
    """The child's VISIBLE marker for a substrate decision it did not make ('' = silent).

    The rule table's `auto` rows are only honest if the child can see which way they
    went: a nanny must know to delegate, and a child that fell back to metered tokens
    must know its route was unavailable rather than discovering it by spending.

    ``lane`` is the same dispatch's lane resolution: a nanny that landed on the
    LIGHT lane by policy is told so, with the sanctioned escalation
    (``switch_model`` for real acceptance judgment) named beside it — a policy the
    child cannot see is a policy it will fight by accident.

    The harness branch SUPERSEDES any native-self-execution framing in the frozen
    task text (owner decision 2A): the composed text is written at schedule time,
    when the executor is unknown, so its execution framing describes the metered
    fallback — and this note rides ONLY the FINAL post-preflight harness dispatch
    (the call site runs after the delegate-visibility preflight), so a native or
    preflight-demoted child never receives the override.
    """
    if decision is None or decision.blocked:
        return ""
    if decision.executor == "harness":
        route = decision.route.route_id if decision.route else ""
        note = (
            f"EXECUTOR: your parent scheduled you on the delegated substrate ({route}). "
            "You are a NANNY. Decide your delegation plan FIRST — right after reading "
            "your objective and constraints, before any substantive work. Cost classes: "
            "a subscription-lane run has known-zero marginal cost when the route reports "
            "its settled spend as $0 (an estimated or undisclosed spend is estimated/unknown, "
            "not zero); every token YOU think on is metered API money. "
            "While the lane is healthy, delegate everything you can — even small tasks — "
            "with delegate_start / delegate_wait, and verify what comes back rather than "
            "believing it. After a delegated run SUCCEEDS, your job is to VERIFY and "
            "INTEGRATE its output — never to rebuild the same work yourself on metered "
            "tokens. Follow-up work (fixes, the next increment, a retry with a corrected "
            "prompt) is delegated too, with a new delegate_start; your own metered rounds "
            "are for judgment — acceptance, integration, honest settlement — not for "
            "co-building around a $0 run. If your run asks a question (delegate_wait "
            "returns waiting_on_user), answer it from the task context with "
            "delegate_answer; a question above your authority — money, scope, external "
            "actions — goes to your human via progress while you keep waiting (a timeout_at "
            "question benign-declines at the engine timeout; timeout_at=null waits until answered). "
            "If your task text instructs you to execute the work natively yourself, that "
            "instruction described the metered fallback and is superseded by this dispatch. "
            "Route thinking-work (code, research, generation) through "
            "delegate_start/delegate_wait; your own run_command/read_file rounds are for "
            "verification, integration, and acceptance. The parent's step-by-step context "
            "is the WORK ORDER for your delegated run's prompt, not a script for you to "
            "execute natively."
        )
        if lane is not None and lane.provenance == "policy" and lane.effective_lane == "light":
            note += (
                " You run on the LIGHT model lane by dispatch policy: custody chores "
                "(starting runs, waiting, reading results, relaying) belong on this "
                "cheap lane. For a genuine acceptance or integration judgment you may "
                "raise your own power with switch_model and drop back after — that is "
                "the sanctioned escalation, not a workaround."
            )
        if decision.reset_at:
            note += (
                f" The route's plan window is currently spent and resets at "
                f"{decision.reset_at}. Decide explicitly: wait for the reset, deliver "
                "partial work, or say you fell back — do not drift into spending."
            )
        return note
    if decision.reason in {"requested_native", "harness_not_configured"}:
        return ""  # the ordinary case has nothing to announce
    if decision.reset_at:
        # D28's fallback, stated as the CAPABILITY DELTA it is: the parent asked for the
        # already-paid substrate to be used when available, every profile of it is spent,
        # and the work is proceeding on metered money instead. Destination 2 of 3 (the
        # child's own prompt); the durable event and the parent's envelope carry the same
        # two facts. The reset instant is named so the child can weigh waiting against
        # spending instead of guessing.
        return (
            "EXECUTOR CAPABILITY DELTA: every plan window of the configured delegated "
            f"substrate is spent (resets at {decision.reset_at}), so you FELL BACK to "
            "METERED API tokens. Your parent asked for 'auto', which permits this "
            "fallback rather than a wait — but it is real money that the subscription "
            "would have covered: keep the work proportionate, and say in your result "
            "that you ran below the substrate you were scheduled for and why."
        )
    return (
        f"EXECUTOR: the configured delegated substrate is unavailable "
        f"({decision.reason}), so you are running on METERED API tokens. Your parent "
        "asked for 'auto', which permits this — but say so in your result."
    )


def executor_blocked_outcome(decision: SubagentExecutorResolution) -> Tuple[str, Dict[str, Any]]:
    """The terminal ``(text, usage)`` of a child that was pinned and could not run.

    Deliberately NOT a fallback: the task ends unrun and typed, having spent nothing.
    """
    if decision.reason in ("delegate_tools_invisible", "delegate_visibility_unverified"):
        # Q1A preflight (2026-08-10 amendments): the route is healthy but the
        # child's MATERIALIZED toolset does not carry the delegate verbs — or
        # the toolset introspection itself failed, so visibility is UNKNOWN,
        # not disproven (distinct reason: the terminal states exactly what is
        # known). Either way the pin cannot be honored, and the fix is tool
        # policy/contract, not waiting for the route to recover.
        detail = (
            "the delegate tools (delegate_start/delegate_wait/delegate_cancel) "
            "are not visible in its materialized toolset"
            if decision.reason == "delegate_tools_invisible"
            else "the toolset introspection failed, so the delegate tools' "
            "(delegate_start/delegate_wait/delegate_cancel) visibility could "
            "not be verified"
        )
        text = (
            "⚠️ EXECUTOR_UNAVAILABLE: this subagent was pinned to the delegated "
            f"substrate (executor='harness'), but {detail}, so the pin cannot be "
            "honored. The task was NOT run on metered API tokens. Fix the tool "
            "policy / task contract that hides the delegate verbs, or schedule "
            "again with executor='auto' to accept metered spend."
        )
        # Literal codes (not `decision.reason`) so the provenance drift guard
        # keeps seeing every code the runtime can emit.
        if decision.reason == "delegate_visibility_unverified":
            return text, {"execution_status": "infra_failed", "reason_code": "delegate_visibility_unverified"}
        return text, {"execution_status": "infra_failed", "reason_code": "delegate_tools_invisible"}
    # ":delegation_" is route_health's structural refinement (Phase D3): the
    # catalog row's manifest cannot run delegated work AT ALL, so "reschedule
    # once the route recovers" would honestly mean "wait forever" (e.g. agy).
    text = (
        "⚠️ EXECUTOR_UNAVAILABLE: this subagent was pinned to the delegated substrate "
        f"(executor='harness') and the route cannot run: {decision.reason}."
        + (f" It resets at {decision.reset_at}." if decision.reset_at else "")
        + " The task was NOT run on metered API tokens, because that spend is exactly "
        "what the pin exists to prevent. "
        + ("This harness structurally cannot run delegated work (its manifest does not "
           "support it), so waiting will not heal it: change the delegated route, or "
           "schedule it again with executor='auto' to accept metered spend."
           if ":delegation_" in decision.reason else
           "Reschedule once the route recovers, or "
           "schedule it again with executor='auto' to accept metered spend.")
    )
    return text, {
        "execution_status": "infra_failed",
        "reason_code": "subagent_executor_unavailable",
    }


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


def _blocked_executor_terminal(cap_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """p34's typed terminal for a blocked executor pin, rebuilt from the facts
    cap_info carried across the (ctx, messages, cap_info) seam. The placeholder
    method p2 kept for exactly this synthesis is deleted; this is the one body."""
    text, usage = executor_blocked_outcome(SubagentExecutorResolution(
        requested=str(cap_info.get("executor_blocked_requested") or "harness"),
        executor="blocked",
        reason=str(cap_info.get("executor_blocked_reason") or ""),
        reset_at=str(cap_info.get("executor_blocked_reset_at") or ""),
    ))
    return text, usage, {"reasoning_notes": ["subagent_executor_unavailable"], "tool_calls": []}


def _persist_early_origin_stub(drive_root: Any, task: Dict[str, Any]) -> None:
    """Durably persist the ingress-captured origin BEFORE the convertible card
    exists (v6.73.0). Merge-write only; the full RUNNING write follows and
    overlays it. Ephemeral decision turns write no durable record by design
    (they are never convertible), and tasks without an origin write nothing.

    A persistence failure is LOUD (warning + typed events.jsonl anomaly) but
    deliberately non-fatal: the owner's task is worth more than its start
    message, and the same storage fault would fail the full RUNNING write
    moments later anyway — the residual convert-in-window exposure requires a
    disk fault racing an instant owner click."""
    if bool(task.get("_ephemeral_turn")):
        return
    ref = task.get("origin_message_ref")
    if not (isinstance(ref, dict) and ref):
        return
    for _attempt in range(2):
        try:
            _agent().write_task_result(
                drive_root,
                str(task.get("id") or ""),
                STATUS_RUNNING,
                chat_id=task.get("chat_id"),
                origin_message_ref=dict(ref),
                origin_message_text=task.get("origin_message_text"),
                result="Task is starting.",
            )
            return
        except Exception:
            if _attempt:
                log.warning("Early origin stub persistence failed", exc_info=True)
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "origin_stub_persist_failed",
            "task_id": str(task.get("id") or ""),
        })
    except Exception:
        log.debug("origin_stub_persist_failed event write failed", exc_info=True)


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
    wrote onto the record moments ago, which is ``resolve_effort(task_type)`` — read
    back rather than recomputed so the loop runs the effort the record states. For
    everything else, and for an unrecognized STORED value (durable data outlives the
    schema that wrote it), it is the task-type default directly.
    """
    stored = str(task.get("reasoning_effort") or "").strip().lower()
    return stored if stored in EFFORT_SCALE else resolve_effort(task_type)


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
    task["subagent_envelope"] = envelope_from_task(task, status=STATUS_RUNNING)
    return dispatch


# The dispatched harness contract needs the whole CUSTODY verb set: a child that
# can start a run but not wait on or cancel it is still broken. `delegate_answer`
# is deliberately NOT part of this preflight — a nanny without it is degraded
# (questions benign-decline at the engine timeout), never custody-broken, and
# failing a dispatch over a missing convenience verb would cost real work.
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

    One check at toolset materialization (owner decision Q1A): an AUTO-resolved
    executor falls back LOUDLY to native — the amended ``capability_delta``
    (reason ``delegate_tools_invisible``, ``reduced=True``) and the corrected
    dispatch fields are re-stamped onto the task record so telemetry does not
    lie; an EXPLICIT ``harness`` pin becomes the typed blocked outcome that
    terminalizes with zero spend (``executor_blocked_outcome``). A broken
    introspection follows the same split: a pinned harness fails CLOSED (a probe
    that cannot prove visibility cannot prove the pinned contract is executable),
    an auto one proceeds fail-open with the probe failure disclosed as a
    ``capability_delta`` note. Returns the (possibly amended) dispatch and
    whether it amended.
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
        task["subagent_envelope"] = envelope_from_task(task, status=STATUS_RUNNING)
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

    pinned = str(task.get("requested_executor") or "auto").strip().lower() == "harness"
    reason = "delegate_tools_invisible"
    try:
        available = set(tools.available_tools())
        if all(verb in available for verb in _DELEGATE_VERBS):
            return dispatch, False
    except Exception:
        log.warning("delegate visibility preflight: introspection failed", exc_info=True)
        if not pinned:
            # Fail-open for auto, but never silently: the note rides the delta.
            return _stamp(dataclasses.replace(
                dispatch,
                delta=_append_reason(dispatch.delta, "delegate_visibility_unverified")))
        # Pinned + broken probe blocks with the honest reason: visibility is
        # UNKNOWN, not disproven.
        reason = "delegate_visibility_unverified"

    if not pinned:
        # F10 (sol #2): the auto fallback runs NATIVE, so lane/model/effort are
        # re-resolved WITHOUT the harness light-lane policy — a native child of
        # a heavy parent must not stay on policy-light with a cheap model. The
        # re-resolution lives with the other dispatch policy in `subagents`.
        from ouroboros.subagents import preflight_native_fallback_dispatch

        return _stamp(preflight_native_fallback_dispatch(task, dispatch, reason))
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


def reset_nanny_economics_marks(ctx: Any, *, route_dispatched: bool) -> None:
    """Reset EVERY nanny-economics mark for a fresh dispatch (F4).

    DEFENSIVE, not load-bearing: ``_prepare_task_context`` builds a FRESH
    ToolContext per task, so nothing stale can leak today — this states the
    marks' lifecycle in one place and keeps it true even if a refactor ever
    reuses a context (leaked cursors would mute or misfire the reminder)."""
    ctx._nanny_route_dispatched = bool(route_dispatched)
    ctx._nanny_finalization_injected = False
    ctx._nanny_metered_progress = None
    ctx._nanny_delegate_baseline = None
    ctx._nanny_reminder_mark = None


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
        # `reduced` with NO disclosable axis is the executor-only case (an `auto`
        # fallback the axis renderer deliberately keeps out of this list) — that
        # fact reaches the child through `dispatch_executor_note` beside this
        # block, so rendering "BELOW what your parent asked for:" over an empty
        # list here told the child nothing and read as a broken sentence.
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
            "(delegate_start / delegate_wait), not your own metered rounds — but say "
            if delta.get("effective_executor") == "harness"
            else "Do the work anyway, but say "
        )
        parts.append(
            "You are running BELOW what your parent asked for: "
            + "; ".join(disclosures)
            + f" ({reason_text}). " + action
            + "so in blockers if the gap actually limited your answer — do not quietly "
            "return a weaker result as if it were full strength."
        )
    if delta.get("legacy_note"):
        parts.append(f"Ignored on your record: {delta['legacy_note']}.")
    return "[CAPABILITY DELTA]\n" + "\n".join(parts) if parts else ""
