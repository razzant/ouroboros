"""Collection side of ``plan_task``'s event route.

A fresh plan-review dispatch returns control at the dispatch barrier
(``ReviewRequest.drain_deadline``): the wave is recorded open with typed
``pending_dispatch`` rows and the reviewer workers keep running under
process-local custody (``ouroboros/review_custody.py``). This leaf owns what
happens next: the per-slot progress line and the ONE system frame the last
settlement writes into the task's mailbox (the mind wakes exactly as it does
for a child result — ``wait_task``/``wait_tasks`` return on
``owner_mailbox_pending``), and the $0 collection that closes or advances the
recorded wave through the existing ``review_disposition`` mode. A collection uses drain window 0 (settled slots
are reconciled, nothing is re-sent or awaited), and a new envelope collects
every other custody-pending wave before it supersedes anything. Closing and
aggregating a wave stays with the collecting call (the sole wave writer);
Historical settlement attaches exact source references through the same wave
owner; it never reaggregates or changes the current plan.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

log = logging.getLogger(__name__)


def announce_released_settlement(
    usage_ctx: Any, *, request: Any, task_id: str, actor: Any,
    settled_wave: Dict[str, str], roster_size: int = 0,
) -> None:
    """Attach late evidence and announce whole-wave settlement through its mailbox.

    The cognitive-operation owner emits per-slot progress. This frame carries
    counts only; the collector remains the sole aggregate writer.
    """
    if str(getattr(request, "surface", "") or "") != "plan_review":
        return
    fingerprint = str((getattr(request, "reconciliation_identity", {}) or {}).get("subject_hash") or "")
    if usage_ctx is not None and getattr(usage_ctx, "drive_root", None):
        try:
            from ouroboros.tools.plan_review import _planning_state_location
            state_root, _ = _planning_state_location(usage_ctx)
            attach_historical_results(
                state_root, task_id, fingerprint=fingerprint,
                operation_id=str(getattr(actor, "operation_id", "") or ""),
            )
        except Exception:
            log.warning("plan review historical settlement could not be attached", exc_info=True)
    if not settled_wave or usage_ctx is None or not getattr(usage_ctx, "drive_root", None):
        return
    ok = sum(1 for status in settled_wave.values() if status in {"ok", "empty"})
    from ouroboros.owner_mailbox import write_task_message

    try:
        write_task_message(
            pathlib.Path(str(usage_ctx.drive_root)),
            f"Plan review wave {fingerprint[:8] or '?'}: {len(settled_wave)} of "
            f"{max(int(roster_size or 0), len(settled_wave))} reviewer slot(s) settled "
            f"({ok} ok, {len(settled_wave) - ok} failed); not yet collected "
            f"(review_fingerprint {fingerprint}).",
            task_id, source_task_id=task_id, provenance="system",
        )
    except Exception:
        log.warning("plan review settled-wave frame failed for %s", task_id, exc_info=True)


def run_plan_coroutine(coro: Any) -> Any:
    """Run one plan-review coroutine to completion from a synchronous tool handler.

    The ToolEntry envelope is the outer settlement bound: the substrate owns each
    review slot's logical window and late-result custody, so no second
    ``asyncio.wait_for`` is nested here (it would cancel the coroutine while its
    executor worker keeps running, then ``asyncio.run`` waits for that worker at
    shutdown and defeats the apparent timeout). ``copy_context``: the registry's
    tool-result sidecar is a ContextVar, and the published native plan result must
    reach the dispatching thread's slot (D02) — a bare pool thread would publish
    into the void."""
    import asyncio
    import concurrent.futures
    import contextvars

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(contextvars.copy_context().run, asyncio.run, coro).result()


def prepared_from_wave(ctx: Any, exact: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    """The exact inputs of a RECORDED wave, for the engine's resume path: the
    restored spec, prose and evidence manifest of the wave itself — never a
    re-read of the evidence, so the collection cannot change the wave's identity."""
    from ouroboros.review_substrate import review_repo_dirs_for
    from ouroboros.tools.plan_review import _PlanRequest

    system_root, active_root = review_repo_dirs_for(ctx)
    spec = dict(exact.get("spec") or {})
    request = _PlanRequest(
        goal=str(spec.get("goal") or ""), plan=str(exact.get("plan_prose") or ""), spec=spec,
        reviewer_effort=str(exact.get("reviewer_effort") or ""),  # the same roster the wave dispatched with
    )
    prepared = {
        "spec": spec, "system_root": system_root, "active_root": active_root,
        "constitutional": bool(exact.get("constitutional")),
        "constitutional_note": str(exact.get("constitutional_note") or ""),
        "manifest": dict(exact.get("evidence_manifest_full") or exact.get("evidence_manifest") or {}),
        "manifest_hash": str(exact.get("evidence_manifest_hash") or ""),
        "reminder": "", "fingerprint": str(exact.get("request_fingerprint") or ""),
    }
    return request, prepared


async def collect_open_wave(ctx: Any, *, state_root: Any, task_id: str, wave: Dict[str, Any]) -> str:
    """Collect one open wave: the engine's own resume path over the wave's recorded
    inputs with drain window 0 — settled slots are reconciled, nothing is re-sent,
    nothing is waited for, and the engine remains the sole wave writer/reducer."""
    from ouroboros.tools import plan_review as engine

    exact = engine._authority_wave(state_root, task_id, wave) or wave
    request, prepared = prepared_from_wave(ctx, exact)
    return await engine._run_plan_review_async(ctx, request, collect=prepared)


def collect_wave_sync(ctx: Any, *, state_root: Any, task_id: str, wave: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Disposition-mode collection: ``(rendered text, reloaded state, authority wave)``."""
    from ouroboros.task_results import load_plan_review_state, plan_review_wave
    from ouroboros.tools import plan_review as engine

    fingerprint = str(wave.get("request_fingerprint") or "")
    text = run_plan_coroutine(collect_open_wave(ctx, state_root=state_root, task_id=task_id, wave=wave))
    state = load_plan_review_state(state_root, task_id)
    stored = plan_review_wave(state, fingerprint) or wave
    return text, state, engine._authority_wave(state_root, task_id, stored) or stored


async def collect_before_supersede(
    ctx: Any, *, state_root: Any, task_id: str, state: Dict[str, Any], fingerprint: str,
) -> Dict[str, Any]:
    """Reconcile-before-supersede (I3): a NEW envelope first collects what has
    settled of EVERY custody-pending wave that is not its own at $0 (window 0), not
    only the current one: a wave superseded earlier under cap room is still money in
    flight, and only its collection can prove or clear its cycle. The caller then
    writes its own superseding reference. Returns the (re)loaded state; an unreadable
    wave is logged and left as it was.

    Only the currently open wave uses the live collector. Older/closed waves
    receive historical supplements without becoming current, even transiently.
    """
    from ouroboros.task_results import load_plan_review_state

    current_fp = str((state.get("current_attempt") or {}).get("fingerprint") or "")
    for wave in state.get("waves") or []:
        if not isinstance(wave, dict) or not wave.get("custody_pending"):
            continue
        fp = str(wave.get("request_fingerprint") or "")
        if fp == str(fingerprint or ""):
            continue
        try:
            if fp == current_fp and not wave.get("closed"):
                await collect_open_wave(ctx, state_root=state_root, task_id=task_id, wave=wave)
            else:
                attach_historical_results(state_root, task_id, fingerprint=fp)
        except (OSError, ValueError, TimeoutError) as exc:
            log.warning("in-flight plan wave %s could not be collected: %s", fp[:8], exc)
    return load_plan_review_state(state_root, task_id)


def in_flight_hold(state: Dict[str, Any], *, fingerprint: str, cap: Any) -> str:
    """The refusal body for a REVISED envelope while ANOTHER wave is still custody-pending
    and the cap has no room for another committed panel, or ``''`` when the envelope
    may proceed. Every in-flight wave occupies one cap slot: a wave that already proved
    a dispatch counts through ``cycles_paid`` (never again as pending), an unproven one
    counts as committed money whose spend only its collection proves (a wave of typed $0
    refusals leaves the cap untouched). Nothing is written here: no superseding reference,
    no cycles_exhausted. The pending wave stays the current, collectible wave and the
    text names its $0 collection. The identical envelope is never held (it resumes).
    Already-paid lost-only custody at a spent cap reaches the existing exhausted
    exit; this does not assert worker death or clear its unknown late outcome."""
    if cap is None:
        return ""
    def paid_lost_at_cap(wave):
        actors = wave.get("actors")
        return (wave.get("paid") is True and int(state.get("cycles_paid") or 0) >= int(cap)
                and isinstance(actors, list) and bool(actors)
                and all(isinstance(actor, dict) and actor.get("operation_state") == "custody_lost"
                        for actor in actors))

    pending = [
        w for w in state.get("waves") or []
        if isinstance(w, dict) and w.get("custody_pending") and not paid_lost_at_cap(w)
        and str(w.get("request_fingerprint") or "") != str(fingerprint or "")
    ]
    unproven = sum(1 for w in pending if not w.get("paid"))
    if not pending or int(state.get("cycles_paid") or 0) + unproven < int(cap):
        return ""
    current_fp = str((state.get("current_attempt") or {}).get("fingerprint") or "")
    wave = next((w for w in pending if str(w.get("request_fingerprint") or "") == current_fp), pending[-1])
    fp = str(wave.get("request_fingerprint") or "")
    from ouroboros.tools.plan_review_runtime import plan_pending_actors
    running = len(plan_pending_actors(wave))
    outcome = "(a wave that proves no physical dispatch leaves the cap untouched; a paid one spends it)"
    route = (
        f"Collect it at $0 with plan_task(review_disposition={{review_fingerprint: '{fp}', items: []}}) "
        f"{outcome}, or resubmit the identical envelope to wait for it."
        if fp == current_fp else
        "It is no longer the current wave, so a review_disposition cannot address it: resubmit its "
        f"identical envelope (review_fingerprint {fp}) to collect it {outcome}."
    )
    return (
        f"plan-review wave {fp[:8]} still has {running} reviewer slot(s) "
        f"in flight and the cycle cap ({cap}) has no room for another panel until that wave is collected. "
        f"{route} No plan attempt was recorded; the current wave is unchanged."
    )


def collect_before_gate(ctx: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """ONE free collection before finalization in every enforcement mode, hurry
    included: when the current wave has custody pending, collect what has
    settled at $0 (window 0, never a wait) and return the reloaded state; any
    other state is returned untouched. ``owner_hurry.force_plan_decision`` then
    projects the verdict. Older waves keep their historical-supplement path."""
    from ouroboros.task_results import current_plan_review_wave, load_plan_review_state
    from ouroboros.tools.plan_review import _planning_state_location

    current = current_plan_review_wave(state)
    if not current or not current.get("custody_pending"):
        return state
    try:
        state_root, task_id = _planning_state_location(ctx)
        run_plan_coroutine(collect_open_wave(ctx, state_root=state_root, task_id=task_id, wave=current))
        return load_plan_review_state(state_root, task_id)
    except (OSError, ValueError, TimeoutError) as exc:
        log.warning("plan wave %s could not be collected before the gate: %s",
                    str(current.get("request_fingerprint") or "")[:8], exc)
        return state


def attach_historical_results(
    state_root: Any, task_id: str, *, fingerprint: str, operation_id: str = "",
) -> int:
    """Attach complete producers to one exact old wave, never invoke a panel.

    Current open review remains the ordinary collector's responsibility. This
    also handles the dispatch-barrier race: record_exact_wave calls it for the
    known historical set after publishing a new current wave. No read projection
    writes state, and neither callback waits for a worker or re-enters its queue.
    """
    from types import SimpleNamespace
    from dataclasses import asdict
    from ouroboros.observability import read_call_payload
    from ouroboros.review_custody import recover_review_producer
    from ouroboros.task_results import (
        load_plan_review_state, plan_review_wave,
    )
    from ouroboros.tools.plan_review_artifacts import (
        authority_wave, persist_historical_result, record_plan_review_supplement,
    )
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    state = load_plan_review_state(state_root, task_id)
    hot = plan_review_wave(state, fingerprint)
    if hot is None or (not hot.get("closed") and
            (state.get("current_attempt") or {}).get("fingerprint") == fingerprint):
        return 0
    wave = authority_wave(state_root, task_id, hot)
    attached = 0
    for row in wave.get("actors") or []:
        op = str(row.get("operation_id") or "")
        if not op or (operation_id and op != operation_id):
            continue
        if not (row.get("late_result_pending") or row.get("operation_state") in
                {"pending_dispatch", "in_flight", "custody_lost"}):
            continue
        if any(item.get("operation_id") == op for item in wave.get("historical_supplements") or []):
            continue
        try:
            _, prompt, _ = read_call_payload(state_root, task_id=task_id, call_id=f"{op}_prompt")
            request, slot = SimpleNamespace(**prompt["request"]), SimpleNamespace(**prompt["slot"])
            from ouroboros.review_execution import ReviewRouteKind
            slot.route = ReviewRouteKind(slot.route)
            identity = dict(getattr(request, "reconciliation_identity", {}) or {})
            retry_key = str(wave.get("retry_key") or f"plan_review:{fingerprint}:{wave['cycle_index']}")
            if (request.surface != "plan_review" or request.task_id != task_id
                    or str(request.retry_key) != retry_key
                    or identity.get("subject_hash") != fingerprint
                    or identity.get("epoch") != retry_key
                    or identity.get("roster_hash") != wave.get("reviewer_config_fingerprint")
                    or identity.get("health_epoch", []) != wave.get("health_epoch", [])
                    or slot.slot_id != row.get("slot_id")):
                continue
            actor = recover_review_producer(state_root, request, slot, row)
            if actor is None or actor.late_result_pending or actor.operation_state not in {
                "settled", "late_settled", "not_dispatched",
            }:
                continue
            result = _plan_row_from_actor(asdict(actor), slot)
            ref = persist_historical_result(state_root, task_id, wave, result)
            if record_plan_review_supplement(state_root, task_id, wave=wave, result=result, source_ref=ref):
                attached += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("historical plan reviewer %s remains unresolved: %s", op, exc)
    return attached


def attach_late_plan_event(state_root: Any, event: Dict[str, Any]) -> int:
    """Recover the addressed plan producer after a supervisor late-result event."""
    if event.get("surface") != "plan_review":
        return 0
    from ouroboros.observability import read_call_payload

    task_id, op = str(event.get("task_id") or ""), str(event.get("operation_id") or "")
    if not task_id or not op:
        return 0
    _, prompt, _ = read_call_payload(state_root, task_id=task_id, call_id=f"{op}_prompt")
    identity = (prompt.get("request") or {}).get("reconciliation_identity") or {}
    return attach_historical_results(
        state_root, task_id, fingerprint=str(identity.get("subject_hash") or ""), operation_id=op,
    )
