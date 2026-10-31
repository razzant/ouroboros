"""The read-side execution-evidence projection over the delegate custody rows.

Extracted from ``ouroboros/delegate_custody.py`` at its module-size ceiling: this
is one coherent READ concern — what a task's delegated runs provably did — with a
single completion-seam consumer (``subagents.envelope_from_task``).
``delegate_custody`` re-exports it (same object), so every existing reference and
monkeypatch target keeps the one historical name. Custody primitives are imported
lazily inside the function so this leaf never participates in an import cycle
with the module that owns the rows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)

# The nanny finalization-nudge stamp (B3, owner decision 3A): written by the
# WORKER at the loop's injection seam through ``delegate_custody.emit`` — only
# when the composed reminder was NON-EMPTY (the ctx `_nanny_finalization_injected`
# flag is set even on suppression, so it can never be the signal). Task-scoped
# (no run_id — the run-custody replay skips it by design); the type keeps the
# ``delegate_run`` prefix so the custody scan prefilter still yields it. Defined
# HERE, beside its one reader, because ``delegate_custody`` sits exactly at its
# module-size ceiling.
NANNY_NUDGE_STAMP = "delegate_run_nanny_nudge_injected"

# A delegate_start attempt the substrate REFUSED before any invocation was
# minted (a typed route_health blocker or a pre-POST argument refusal).
# Task-scoped like the stamp above (no
# run_id/invocation_id — pending-invocation recovery must never sweep it);
# the ``delegate_run`` prefix keeps the custody scan prefilter yielding it.
# Post-preflight attempts already leave durable START_REQUESTED rows, so the
# two together answer "did the nanny TRY?" without a new scan.
START_BLOCKED = "delegate_run_start_blocked"

# The host's pre-start found the dispatch BLOCKED (charter D2): a typed,
# task-scoped attempt row. The ``delegate_run`` prefix is LOAD-BEARING —
# ``delegate_custody._iter_rows`` prefilters rows to that prefix, so an
# unprefixed spelling is invisible to every evidence reader (delta finding D1:
# the first spelling of this row was exactly that dead letter).
STARTUP_FAULT = "delegate_run_configured_startup_fault"


def record_start_blocked(ctx: Any, task_id: str, reason: str) -> None:
    """Durably record "delegate_start was attempted and refused typed".

    Fail-soft, same shape as the nudge stamp: a row that cannot land loses only
    the completion-seam nuance (an attempted nanny might read as nudge-ignoring).
    """
    try:
        from ouroboros import delegate_custody as custody

        custody.emit(custody.custody_root(ctx), START_BLOCKED, {
            "task_id": str(task_id or ""), "reason": str(reason or "")})
    except Exception:
        log.debug("start-blocked row failed", exc_info=True)


def record_nanny_nudge_stamp(ctx: Any, task_id: str, nanny_code: str) -> None:
    """Durably stamp "the finalization nudge was really injected" for this task.

    Synchronous same-process append on the CANONICAL (budget) root — the same
    root ``task_execution_evidence`` reads back at the completion seam, so the
    fact needs no supervisor round-trip and no O(history) side scan. Fail-soft:
    a stamp that cannot land loses only the completion-seam disclosure.
    """
    try:
        from ouroboros import delegate_custody as custody

        custody.emit(custody.custody_root(ctx), NANNY_NUDGE_STAMP, {
            "task_id": str(task_id or ""), "nanny_code": str(nanny_code or "")})
    except Exception:
        log.debug("nanny nudge stamp failed", exc_info=True)


def task_execution_evidence(drive_root: Any, task_id: str) -> Dict[str, Any]:
    """Aggregate ONE task's delegated-run facts from its custody rows (indexed reads).

    The completion-seam reconciliation reads this: `executor_route` is a DISPATCH
    decision, these rows are the EVIDENCE of what actually ran, and the two are
    compared exactly once (``subagents.envelope_from_task``) instead of in every
    reader. ``subscription_cost_usd`` is the sum of DISCLOSED settled spend and
    ``None`` while nothing settled or any settled run left its spend undisclosed —
    unknown never renders as zero.

    A run a REVIEW surface registered is that panel's substrate, never this task's:
    its rows carry their own ``source``, and neither its attempt, its counters, its
    model, its applied access nor its failure state is evidence here (issue #1006).
    The task-scoped facts without a run (the nudge stamp, a refused attempt) are the
    task's ``custody_marks`` row, kept by the same reducer transaction.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import marks, runs as current_runs
    from ouroboros.store import StoreError

    tid = str(task_id or "")
    started: set = set()
    settled: set = set()
    succeeded: set = set()
    failure_states: List[str] = []
    models: List[str] = []
    applied_access_profiles: List[str] = []
    cost_total, cost_known, cost_estimated, cost_partial = 0.0, False, False, False
    # Scope finding (a5e59bdf gate): UNREADABLE custody must not collapse into the
    # same zero-count result as a proven empty one — a reader would then accuse a
    # nanny of "zero attempts" on evidence it never saw. The store answers or raises;
    # a raise is the typed unknown.
    evidence_read_failed = False
    try:
        runs = current_runs(drive_root, "task_id=?", (tid,))
        mark = marks(drive_root, tid)
    except StoreError:
        runs, mark, evidence_read_failed = {}, {}, True
    nudge_recorded = bool(mark.get("nanny_nudge"))
    start_attempted = bool(mark.get("start_attempted"))
    partial_work_order_seen = False
    for run_id, entry in runs.items():
        if entry.review_owned:
            continue
        start_attempted = True  # a bound run had a durable request
        started.add(run_id)
        source_request = entry.work_order_source_request
        partial_work_order_seen = partial_work_order_seen or (
            entry.work_order_coverage == "partial"
            # a continuation that adopted a snapshot carries its predecessor's sources
            or bool(isinstance(source_request, dict) and source_request.get("inherited_sources"))
        )
        if not entry.settled:
            continue
        settled.add(run_id)
        if not entry.ledger_recorded and not entry.terminal_state:
            # Closed-without-settlement is still TERMINAL: leaving it in the
            # started-minus-settled gap would read as "still executing" to the
            # pending/settled readers (nanny reminder) forever. No ledger row was
            # written, so its spend is undisclosed, never zero.
            failure_states.append("closed_absent")
            cost_partial = True
            continue
        if entry.terminal_state in custody.SUCCEEDED_STATES:
            succeeded.add(run_id)
        elif entry.terminal_state:
            failure_states.append(entry.terminal_state)
        if entry.spend_disclosed and entry.cost_usd is not None:
            cost_known = True
            cost_total += float(entry.cost_usd)
            cost_estimated = cost_estimated or entry.spend_estimated
            cost_partial = cost_partial or entry.cost_partial
        else:
            cost_partial = True
        # ENGINE-reported models only (SETTLED facts): the STARTED row's requested pin
        # with an owner default model is routinely non-empty and names a model that
        # never executed.
        if entry.engine_model and entry.engine_model not in models:
            models.append(entry.engine_model)
        # D29 applied-access disclosure: the settlement recorded the ACCESS the
        # engine actually served (effectiveAccess), distinct from the granted shape.
        # Bounded to the known vocabulary and a short cap: the value rides a durable
        # evidence row and the chip tooltip.
        if (entry.access_profile and entry.access_profile not in applied_access_profiles
                and len(applied_access_profiles) < 8):
            applied_access_profiles.append(entry.access_profile)
    # A partial work-order run is not a successful delegated substrate until the
    # durable verified-range union covers the complete canonical brief. Reuse the
    # custody rows (the same SSOT used by wait/apply) so a restart cannot turn a
    # prompt-only claim into ``harness_used``.
    source_unresolved = 0
    if partial_work_order_seen:
        for entry in runs.values():
            if not entry.settled or entry.review_owned:
                continue
            if custody.work_order_source_verification(entry).get("status") == "cannot_verify":
                source_unresolved += 1
                succeeded.discard(entry.run_id)
                if "source_incomplete" not in failure_states:
                    failure_states.append("source_incomplete")
    from ouroboros.delegate_bootstrap_identity import read_identity
    from ouroboros.delegate_recovery import unsettled_start_ids

    try:
        initial = read_identity(drive_root, task_id)
        initial_run = next((run for run in runs.values() if initial and run.invocation_id == initial["invocation_id"]), None)
        if initial:
            fate = custody.invocation_record(drive_root, initial["invocation_id"]) or {}
            initial = {**initial, "run_id": initial_run.run_id if initial_run else str(fate.get("run_id") or ""),
                       "state": initial_run.terminal_state if initial_run and initial_run.settled else str(fate.get("state") or "unknown")}
        obligations = unsettled_start_ids(drive_root, task_id)
    except Exception:
        initial, obligations = {"state": "unknown"}, {"status": "unknown"}
        evidence_read_failed = True
    return {
        # A settled row whose started row fell out of the log is still a run that ran.
        "delegated_runs_started": len(started | settled),
        "initial_execution": initial,
        "delegated_obligations": obligations,
        "delegated_runs_settled": len(settled),
        # The terminal-state axis (F4, 2026-08-10 saga): a run that STARTED and
        # FAILED is an ATTEMPTED route, not a refusal to delegate. Readers (the
        # nanny nudge, the forced-path note, the completion seam) must be able to
        # tell "never tried" from "tried and the run died" without re-parsing the
        # event log; the forced-path nanny note keys on the succeeded count to
        # stop nagging over finished work.
        "delegated_runs_succeeded": len(succeeded),
        # C3 (additive raw counter): settled runs that did NOT succeed, including
        # closed-absent. Counts are delegated-run facts only — the NATIVE (metered)
        # contribution beside them is unknown, and no share/ratio is derivable here.
        "delegated_runs_failed": len(settled) - len(succeeded),
        "delegated_runs_source_unresolved": source_unresolved,
        "delegated_run_failure_states": sorted(set(failure_states)),
        # True only when the canonical log EXISTS but could not be opened —
        # zero counts are then "unknown", not "established" (additive key).
        "evidence_read_failed": evidence_read_failed,
        # B3 (additive key): a non-empty finalization nudge was durably stamped
        # for this task. The completion seam combines it with completed status
        # + zero started runs into the typed substrate disclosure.
        "nanny_nudge_recorded": nudge_recorded,
        # Additive: ANY durable delegate_start attempt (blocked-typed or
        # requested), started or not. The seam reads it so a nanny the
        # substrate refused is never disclosed as having ignored the nudge.
        "delegate_start_attempted": start_attempted or bool(started | settled),
        "subscription_cost_usd": round(cost_total, 6) if (settled and cost_known) else None,
        # The settlement row's own estimated/final distinction, carried instead of
        # dropped: an estimated sum must never render as an exact receipt.
        "subscription_cost_estimated": bool(settled and cost_known and cost_estimated),
        "subscription_cost_partial": bool(settled and cost_partial),
        "harness_models": models,
        # Additive (D29 projection): the access the engine actually served,
        # from SETTLED rows only. Empty list = no receipt disclosed it.
        "applied_access_profiles": applied_access_profiles,
    }


__all__ = ["NANNY_NUDGE_STAMP", "START_BLOCKED", "record_nanny_nudge_stamp",
           "record_start_blocked", "task_execution_evidence"]
