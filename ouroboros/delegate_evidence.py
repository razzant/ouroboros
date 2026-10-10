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
    """Aggregate ONE task's delegated-run facts from the durable custody rows.

    The completion-seam reconciliation reads this: `executor_route` is a DISPATCH
    decision, these rows are the EVIDENCE of what actually ran, and the two are
    compared exactly once (``subagents.envelope_from_task``) instead of in every
    reader. ``subscription_cost_usd`` is the sum of DISCLOSED settled spend and
    ``None`` while nothing settled or any settled run left its spend undisclosed —
    unknown never renders as zero.

    A run a REVIEW surface registered is that panel's substrate, never this task's:
    its rows carry their own ``source``, and neither its attempt, its counters, its
    model, its applied access nor its failure state is evidence here (issue #1006).
    """
    from ouroboros import delegate_custody as custody

    tid = str(task_id or "")
    started: set = set()
    settled: set = set()
    succeeded: set = set()
    failure_states: List[str] = []
    models: List[str] = []
    applied_access_profiles: List[str] = []
    cost_total, cost_known, cost_estimated = 0.0, True, False
    # Scope finding (a5e59bdf gate): an UNREADABLE log must not collapse into
    # the same zero-count result as a proven empty one — a reader would then
    # accuse a nanny of "zero attempts" on evidence it never saw. A missing
    # file IS a positively-established empty state (no row could exist);
    # existing-but-unreadable is not, and _iter_rows swallows its own OSError.
    evidence_read_failed = False
    nudge_recorded = False
    start_attempted = False
    partial_work_order_seen = False
    # Run ids a review-owned row has already named. The log is append-only, so a
    # run's own STARTED (or START_REQUESTED, or its SETTLED receipt) is seen before
    # the rest of its rows; a SETTLED row that outlived its start names itself.
    review_ids: set = set()
    _log_path = custody.event_log_path(drive_root)
    try:
        if _log_path.exists():
            with _log_path.open("rb"):
                pass
    except OSError:
        evidence_read_failed = True
    for row in custody.custody_rows(drive_root):
        if str(row.get("task_id") or "") != tid:
            continue
        if str(row.get("type") or "") == NANNY_NUDGE_STAMP:
            # Task-scoped stamp, no run_id — read here so the completion seam
            # gets the fact from the scan it already pays for (B3).
            nudge_recorded = True
            continue
        run_id = str(row.get("run_id") or "")
        if custody.review_owned_source(row.get("source")):
            if run_id:
                review_ids.add(run_id)
            continue
        if run_id and run_id in review_ids:
            continue
        if str(row.get("type") or "") in (
            START_BLOCKED, STARTUP_FAULT, custody.START_REQUESTED,
        ):
            # An ATTEMPT is evidence of obedience even when nothing started:
            # a typed pre-mint refusal (START_BLOCKED) or a durable request
            # whose POST then failed (START_REQUESTED with no STARTED row).
            start_attempted = True
        if not run_id:
            continue
        kind = str(row.get("type") or "")
        if kind == custody.STARTED:
            started.add(run_id)
            source_request = row.get("work_order_source_request")
            partial_work_order_seen = partial_work_order_seen or (
                str(row.get("work_order_coverage") or "") == "partial"
                # a continuation that adopted a snapshot carries its predecessor's sources
                or bool(isinstance(source_request, dict) and source_request.get("inherited_sources"))
            )
        elif kind == custody.CLOSED_ABSENT and run_id not in settled:
            # Closed-without-settlement is still TERMINAL: leaving it in the
            # started-minus-settled gap would read as "still executing" to the
            # pending/settled readers (nanny reminder) forever. No ledger row was
            # written, so its spend is undisclosed, never zero.
            settled.add(run_id)
            failure_states.append("closed_absent")
            cost_known = False
        elif kind == custody.SETTLED and run_id not in settled:
            settled.add(run_id)
            state = str(row.get("state") or "")
            if state in custody.SUCCEEDED_STATES:
                succeeded.add(run_id)
            elif state:
                failure_states.append(state)
            if row.get("spend_disclosed") and row.get("cost_usd") is not None:
                try:
                    cost_total += float(row.get("cost_usd") or 0.0)
                except (TypeError, ValueError):
                    cost_known = False
                if row.get("spend_estimated"):
                    cost_estimated = True
            else:
                cost_known = False
        # ENGINE-reported models only (SETTLED rows): a STARTED row carries the
        # requested pin, and with an owner default model that pin is routinely
        # non-empty — listing it would name a model that never executed.
        if kind == custody.SETTLED:
            model = str(row.get("model") or "")
            if model and model not in models:
                models.append(model)
            # D29 applied-access disclosure: the settlement row records the
            # ACCESS the engine actually served (effectiveAccess), distinct
            # from the STARTED row's granted shape. Projected so the chip and
            # acceptance readers can answer asked-vs-applied without joining
            # the ledger. Empty when telemetry predates the receipt.
            # Bounded to the known access-profile vocabulary and a short cap:
            # the value rides a durable evidence row and the chip tooltip, so a
            # garbage/oversized engine string must not reach either verbatim.
            applied_access = str(row.get("access_profile") or "")[:32]
            if (
                applied_access
                and applied_access not in applied_access_profiles
                and len(applied_access_profiles) < 8
            ):
                applied_access_profiles.append(applied_access)
    # A partial work-order run is not a successful delegated substrate until the
    # durable verified-range union covers the complete canonical brief. Reuse the
    # custody replay (the same SSOT used by wait/apply) so a restart cannot turn a
    # prompt-only claim into ``harness_used``.
    source_unresolved = 0
    if partial_work_order_seen:
        try:
            for entry in custody.replay(drive_root).values():
                if entry.task_id != tid or not entry.settled or entry.review_owned:
                    continue
                if custody.work_order_source_verification(entry).get("status") == "cannot_verify":
                    source_unresolved += 1
                    succeeded.discard(entry.run_id)
                    if "source_incomplete" not in failure_states:
                        failure_states.append("source_incomplete")
        except Exception:
            evidence_read_failed = True
    return {
        # A settled row whose started row fell out of the log is still a run that ran.
        "delegated_runs_started": len(started | settled),
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
        "harness_models": models,
        # Additive (D29 projection): the access the engine actually served,
        # from SETTLED rows only. Empty list = no receipt disclosed it.
        "applied_access_profiles": applied_access_profiles,
    }


__all__ = ["NANNY_NUDGE_STAMP", "START_BLOCKED", "record_nanny_nudge_stamp",
           "record_start_blocked", "task_execution_evidence"]


def acceptance_substrate_facts(ctx: Any, task_id: str) -> Dict[str, Any]:
    """Substrate FACTS for the task under review, from durable custody rows.

    Acceptance used to see substrate truth only through the terminal result
    envelope — written AFTER acceptance runs — so a harness-dispatched task was
    always judged as if it ran as scheduled. Custody rows are durable and
    complete before finalization, so the packet reads them directly. FACTS
    ONLY, never a verdict rule (owner 2026-08-28: acceptance judges quality,
    not the execution route — these rows are context, they gate nothing).
    Empty for tasks never dispatched toward the delegated substrate.
    """
    bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
    if not bool(getattr(ctx, "_nanny_route_dispatched", False)) and not isinstance(
        bootstrap, dict,
    ):
        return {}
    out: Dict[str, Any] = {}
    try:
        from ouroboros.delegate_custody import custody_root
        from ouroboros.subagents import actual_substrate

        evidence = task_execution_evidence(custody_root(ctx), str(task_id or ""))
        evidence = evidence if isinstance(evidence, dict) else {}
        if evidence.get("evidence_read_failed"):
            # Unreadable custody must not read as a proven-empty substrate —
            # neither the enum NOR the zero-valued counters (which would be
            # exactly the proven-empty reading beside the flag).
            out["evidence_read_failed"] = True
        else:
            out["actual_substrate"] = actual_substrate(evidence)
            for key in (
                "delegated_runs_started", "delegated_runs_settled",
                "delegated_runs_succeeded", "delegated_runs_failed",
                "delegated_runs_source_unresolved",
            ):
                out[key] = int(evidence.get(key) or 0)
            if evidence.get("delegate_start_attempted"):
                # Durable start-blocked/requested rows: an ATTEMPT is evidence
                # of obedience even when nothing started (plan D8: start-blocked
                # facts ride the packet, not only the run counters).
                out["delegate_start_attempted"] = True
        failure_states = [
            str(s) for s in (evidence.get("delegated_run_failure_states") or [])
        ]
        if failure_states:
            out["delegated_run_failure_states"] = failure_states[:12]
            if len(failure_states) > 12:
                # Bounded but DISCLOSED (P1): never a silent truncation.
                out["failure_states_omitted"] = len(failure_states) - 12
        if evidence.get("subscription_cost_usd") is not None:
            out["subscription_cost_usd"] = evidence.get("subscription_cost_usd")
    except Exception:
        log.debug("Failed to read substrate custody evidence for acceptance", exc_info=True)
        return {"evidence_read_failed": True}
    if isinstance(bootstrap, dict):
        out["configured_session"] = True
        if bootstrap.get("zero_run_receipt_recorded"):
            out["zero_run_decision"] = str(bootstrap.get("zero_run_decision") or "")
            out["zero_run_basis"] = str(bootstrap.get("zero_run_basis") or "")
        if bootstrap.get("route_available") is False:
            out["route_available"] = False
    refusal = getattr(ctx, "_configured_startup_refusal", None)
    if isinstance(refusal, dict):
        out["startup_refused"] = str(refusal.get("reason") or "")
        if refusal.get("detail"):
            out["startup_refused_detail"] = str(refusal.get("detail"))
    return out


_ACCEPT_DELTA_CHILD_CAP = 20  # reduced-children rows in the finalizer aggregate
_ACCEPT_PATCH_DISPOSITION_CAP = 20  # disposition rows in the acceptance section


def acceptance_patch_dispositions(drive_root: Any, task_id: str) -> Dict[str, Any]:
    """Typed aggregate of this parent's patch apply/reject decisions (D-trace).

    ``integrate_delegated_patch`` applied a child's diff on FIVE mechanical
    manifest fields with zero review facts anywhere on the path; the owner
    decision (4=A) is to ATTEST the apply rather than invent a review: every
    verdict now also lands as a ``subagent_patch_verdict`` custody row, and
    this section projects those rows host-attested into the acceptance packet
    (the ``capability_deltas`` shape — bounded, first-class, never squeezed
    through the 4KB artifact-preview cliff). ABSENCE of the section means "no
    disposition recorded", never "reviewed clean"; an unreadable log is the
    typed ``evidence_read_failed`` marker, never an empty-therefore-clean
    section (the ``task_execution_evidence`` rule, GR6-4).

    Rows keep the newest ``_ACCEPT_PATCH_DISPOSITION_CAP`` (20) entries and
    ``omitted`` counts the rest. ``unreviewed_delegated_apply`` is set when any
    applied row of the ``delegated`` pipeline exists, judged before the cap.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.utils import truncate_review_artifact

    tid = str(task_id or "")
    out: Dict[str, Any] = {}
    log_path = custody.event_log_path(drive_root)
    try:
        if log_path.exists():
            with log_path.open("rb"):
                pass
        else:
            return out
    except OSError:
        return {"evidence_read_failed": True}
    rows: List[Dict[str, Any]] = []
    for row in custody.custody_rows(drive_root):
        if str(row.get("type") or "") != "delegate_run_patch_verdict":
            continue
        if str(row.get("task_id") or "") != tid:
            continue
        rows.append({
            "child": str(row.get("child_task_id") or ""),
            "pipeline": str(row.get("pipeline") or ""),
            "disposition": str(row.get("disposition") or ""),
            "applied": bool(row.get("applied")),
            "reason": truncate_review_artifact(str(row.get("reason") or ""), limit=600),
            "patch_sha256": str(row.get("patch_sha256") or ""),
            **({"target_root": str(row["target_root"])} if row.get("target_root") else {}),
            **({"verdict_artifact_write_failed": True}
               if row.get("verdict_artifact_write_failed") else {}),
        })
    if not rows:
        return out
    out["total"] = len(rows)
    # The honest headline the panel weighs — computed over the COMPLETE row
    # set BEFORE bounding: a delegated apply among the omitted-oldest rows is
    # exactly the fact the owner's attest decision (4=A) exists to surface,
    # and deriving it from the truncated view false-negatives past the cap.
    if any(r["applied"] and r.get("pipeline") == "delegated" for r in rows):
        out["unreviewed_delegated_apply"] = True
    if len(rows) > _ACCEPT_PATCH_DISPOSITION_CAP:
        out["omitted"] = len(rows) - _ACCEPT_PATCH_DISPOSITION_CAP
        rows = rows[-_ACCEPT_PATCH_DISPOSITION_CAP:]
    out["rows"] = rows
    return out


def acceptance_capability_deltas(drive_root: Any, task_id: str, root_task_id: str) -> Dict[str, Any]:
    """Typed aggregate of capability reductions for the FINALIZER (one section).

    The task's own dispatch delta plus every DIRECT child that ran below what
    was asked for (lane served on Main, executor fallback to metered tokens,
    profile reduction). Each delta is disclosed at absorption — but absorption
    happens mid-flight, dozens of rounds before the final claim is written, and
    nothing carried the accumulated picture to finalization: a result built on
    degraded runs was judged as if everything ran as scheduled. One bounded,
    host-attested section; ``disclosable_capability_delta`` is the SAME predicate
    the absorption surfaces use, so this cannot disagree with what the parent
    was told. Empty dict when nothing was reduced (noise-free by construction).
    """
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools.control import disclosable_capability_delta

    out: Dict[str, Any] = {}
    try:
        own = disclosable_capability_delta(load_task_result(drive_root, task_id) or {})
        if own:
            out["own"] = own
        children: List[Dict[str, Any]] = []
        for row in find_child_tasks(
            drive_root,
            parent_task_id=task_id,
            root_task_id=root_task_id or task_id,
            scope="direct",
        ):
            delta = disclosable_capability_delta(row)
            if delta:
                children.append({
                    "task_id": str(row.get("task_id") or ""),
                    "status": str(row.get("status") or ""),
                    "capability_delta": delta,
                })
        if children:
            out["children_reduced_count"] = len(children)
            if len(children) > _ACCEPT_DELTA_CHILD_CAP:
                out["children_omitted"] = len(children) - _ACCEPT_DELTA_CHILD_CAP
                children = children[:_ACCEPT_DELTA_CHILD_CAP]
            out["children"] = children
    except Exception:
        log.debug("Failed to aggregate capability deltas for acceptance evidence", exc_info=True)
    return out
