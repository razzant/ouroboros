"""Reconciliation of delegated runs: the settle-or-cancel sweeps and their recovery.

The open-run and pending-invocation projections, the loop-exit release, the
kill-path and orphan sweeps and the per-run settle-or-cancel core. Extracted
from delegate_custody.py (v7 DEL1 split); delegate_custody.py re-exports every
name, so every existing reference — the tools, the supervisor sweeps, the
tests and monkeypatch targets — still finds them there.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros.delegate_registration_policy import record_persistent as _record_persistent

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only name; lazy under future annotations, never imported at runtime
    from ouroboros.delegate_custody import RunCustody


# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.delegate_custody")


def _custody():
    """The parent custody module, read at call time.

    The custody members stay monkeypatch-addressable at their historical
    ``ouroboros.delegate_custody`` bindings (tests rebind them there), so this
    leaf resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import delegate_custody

    return delegate_custody


def open_runs(drive_root: Any, state: Optional[Dict[str, "RunCustody"]] = None) -> List[RunCustody]:
    """Runs with a durable start and no durable settlement (``state``: a shared
    pre-replayed snapshot, so a batch of audits pays one log traversal)."""
    return [custody for custody in (state if state is not None else _custody().replay(drive_root)).values()
            if not custody.settled]


def pending_invocations(drive_root: Any,
                        rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Invocations with a durable request row, no bound run, no definite refusal.

    The launched-never-collected class one step EARLIER than ``open_runs``: a
    worker death between the accepted POST and ``record_started`` leaves only the
    ``START_REQUESTED`` row. Facts come from the FIRST request row (the minting,
    same rule as ``invocation_record``); a record whose canonical body never
    landed is excluded (nothing byte-identical can be replayed). ``rows`` shares
    one pre-read snapshot with ``replay`` (atomic payload busy claim)."""
    from ouroboros.delegate_pending import pending_invocations as replay_pending

    return replay_pending(drive_root, rows)


def release_task_runs(drive_root: Any, task_id: str, *,
                      gateway_factory: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Run the one non-panic terminal custody boundary for a normal loop exit."""

    from ouroboros.delegate_terminal import (
        record_terminal_reconciliation, terminal_reconcile_task,
    )

    result = terminal_reconcile_task(
        drive_root, task_id, gateway_factory=gateway_factory, trigger="loop_exit",
    )
    record_terminal_reconciliation(drive_root, task_id, result)
    return list(result.get("outcomes") or [])


def reconcile_task_runs(drive_root: Any, task_id: str, *,
                        gateway_factory: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Settle or cancel ONE task's open runs from the DURABLE rows (kill path).

    The supervisor-side twin of ``release_task_runs`` for a task whose worker was
    just KILLED (cancellation custody / reap): the graceful release never ran and
    its memo died with the process, so the durable rows are the only complete
    view. Covers pending invocations like the orphan sweep; cheap when the task
    delegated nothing.
    """
    mine = str(task_id or "")
    if not mine:
        return []
    held = [c for c in _custody().open_runs(drive_root) if c.task_id == mine]
    stray = [record for record in _custody().pending_invocations(drive_root)
             if record["task_id"] == mine]
    # Also the registration retry lane for the task's OWN settled-but-owned
    # runs in retire-eligible projects (a one-shot process may never see a
    # sweep tick); deferred registrations stay with the periodic sweep.
    snapshot = _custody().replay(drive_root).values()
    live = {r.project_id for r in snapshot
            if r.project_id and r.run_id and not r.settled}
    owed = [r for r in snapshot
            if r.task_id == mine and r.project_owned and r.project_id
            and r.settled and r.project_id not in live]
    if not held and not stray and not owed:
        return []
    return _reconcile_each(drive_root, held, gateway_factory, pending=stray)


def reconcile_orphaned_runs(
    drive_root: Any,
    running_task_ids: Optional[set] = None,
    *,
    gateway_factory: Optional[Callable[[], Any]] = None,
    recoverable_task_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Settle or cancel every open run whose owning task is no longer running.

    The owner-is-gone predicate is the SAME one ``process_custody.reap_orphaned_processes``
    already uses (the supervisor's live task set), so a delegated run and a spawned
    process cannot disagree about whether their owner still exists. ``running_task_ids``
    of None means UNKNOWN and reconciles nothing — never mass-cancel on missing info.
    """
    if running_task_ids is None:
        return []
    spared = set(recoverable_task_ids or ())
    live_or_reserved = set(running_task_ids) | spared
    orphans = [c for c in _custody().open_runs(drive_root) if c.task_id and c.task_id not in live_or_reserved]
    # The class ONE STEP EARLIER (P34R.2): an invocation whose POST the daemon may have
    # accepted but whose worker died before record_started has no run row for the sweep
    # above to find — a live mutating run nobody could ever collect. Recovered here on
    # the SAME owner-is-gone predicate; a pending invocation whose owner is ALIVE stays
    # untouched, because that owner holds the retry token and decides.
    stray = [record for record in _custody().pending_invocations(drive_root)
             if record["task_id"] and record["task_id"] not in live_or_reserved]
    return _reconcile_each(drive_root, orphans, gateway_factory, pending=stray)


def _reconcile_each(drive_root: Any, runs: List[RunCustody],
                    gateway_factory: Optional[Callable[[], Any]],
                    pending: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One transport, one settle-or-cancel pass. Shared by both release surfaces.

    ``pending`` is the durable sweep's extra duty: START_REQUESTED-only invocations
    (P34R.2). The in-process twin ``release_task_runs`` never passes it — its memo is
    run-keyed and cannot name an unbound invocation — so that class is covered by the
    startup/periodic sweep, within its cadence. The registration pass at the end is
    the third duty: settlement no longer discharges project ownership, and the
    startup/periodic sweep surface is where settled-but-owned registrations get
    their retry lane (the task-scoped surface can exit early with none open).
    """
    # Registration duty is real work only for a retire-ELIGIBLE project:
    # a deferred one must not spin the daemon up.
    snapshot = _custody().replay(drive_root).values()
    unsettled_projects = {row.project_id for row in snapshot
                          if row.project_id and row.run_id and not row.settled}
    registrations = [row for row in snapshot
                     if row.project_owned and row.project_id
                     and row.project_id not in unsettled_projects]
    if not runs and not pending and not registrations:
        return []
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if gateway_factory is None:
        # Attach to the installation's surviving daemon, or restart a dead one
        # when reconciliation has real work (never on the empty early return).
        # A staged runtime pin applies at that natural start, not by killing
        # a live daemon merely because its spawning server generation ended.
        from ouroboros.claudexor_daemon import ensure_owned_gateway

        gateway_factory = ensure_owned_gateway
    try:
        gateway = gateway_factory()
        gateway.handshake()
    except ClaudexorUnavailable:
        log.debug("delegated-run reconciliation skipped: transport unavailable", exc_info=True)
        return []
    outcomes: List[Dict[str, Any]] = []
    try:
        for custody in runs:
            outcomes.append(_custody()._reconcile_one(drive_root, gateway, custody))
        for record in pending or []:
            outcomes.append(_recover_pending_invocation(drive_root, gateway, record))
        # Recomputed inside: a run settled this very pass may have made its
        # project eligible - the pre-pass gate is not the last word.
        if registrations or runs:
            _custody().retire_settled_registrations(drive_root, gateway)
    finally:
        try:
            gateway.close()
        except Exception:
            log.debug("delegated-run reconciliation close failed", exc_info=True)
    return outcomes


def _recover_pending_invocation(drive_root: Any, gateway: Any,
                                record: Dict[str, Any]) -> Dict[str, Any]:
    """Recover the run (if any) behind an orphaned pending invocation, idempotently.

    The stored canonical body is re-POSTed under the invocation's own wire key:
    the engine returns the ORIGINAL handle when the first POST was accepted, and
    starts fresh only when the daemon truly never saw it. A definite 4xx retires
    the invocation and its registration; an unknown outcome stays pending.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    invocation_id = str(record["invocation_id"])
    task_id = str(record["task_id"])
    try:
        handle = gateway.start_run(dict(record["request"]), idempotency_key=invocation_id)
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        if 400 <= status < 500:
            retired = _retire_recovered_registration(gateway, record)
            _custody().emit(drive_root, _custody().START_FAILED, {
                "run_id": "", "task_id": task_id, "project_id": record["project_id"],
                "project_retired": retired, "reason": f"recovery_refused_{exc.code}",
                "invocation_id": invocation_id, "definite": True,
            })
            result = {"invocation_id": invocation_id, "task_id": task_id,
                      "action": "invocation_retired"}
        else:
            result = {"invocation_id": invocation_id, "task_id": task_id,
                      "action": "recovery_unreachable"}
        _custody().emit(drive_root, _custody().RECONCILED, result)
        return result
    run_id = str(handle.get("runId") or handle.get("jobId") or "")
    if not run_id:
        # Queued without an id: durably enqueued, still unnameable. Leave the
        # invocation pending; the next sweep replays the same key and tries again.
        result = {"invocation_id": invocation_id, "task_id": task_id,
                  "action": "recovery_pending"}
        _custody().emit(drive_root, _custody().RECONCILED, result)
        return result
    body = record["request"]
    execution = body.get("execution") if isinstance(body.get("execution"), dict) else {}
    scope = body.get("scope") if isinstance(body.get("scope"), dict) else {}
    custody = _custody().RunCustody(
        run_id=run_id, task_id=task_id,
        route_id=str(record["route"] or body.get("primaryHarness") or ""),
        model=str(body.get("model") or ""),
        profile_id=str(body.get("credentialProfileId") or ""),
        project_id=record["project_id"], project_owned=bool(record["project_owned"]),
        project_persistent=_record_persistent(record),
        root_task_id=str(record.get("root_task_id") or ""),
        parent_task_id=str(record.get("parent_task_id") or ""),
        category=str(record.get("category") or "subagent"),
        source=str(record.get("source") or "delegated_subagent"),
        **{key: str(record.get(key) or "") for key in REVIEW_ATTRIBUTION_KEYS},
        # The sweep runs against the canonical root; a recovered run's ledger row
        # belongs there like every other (P34R.1).
        ledger_root=str(drive_root),
        idempotency_key=str(record["idempotency_key"]), invocation_id=invocation_id,
        selected_subagent_id=str(record.get("selected_subagent_id") or ""),
        config_fingerprint=str(record.get("config_fingerprint") or ""),
        work_order_fingerprint=str(record.get("work_order_fingerprint") or ""),
        work_order_coverage=str(record.get("work_order_coverage") or ""),
        work_order_source_request=(
            dict(record.get("work_order_source_request"))
            if isinstance(record.get("work_order_source_request"), dict) else {}
        ),
        authority_fingerprint=str(record.get("authority_fingerprint") or ""),
        # The C1 isolation binding survives recovery VERBATIM: the recovered
        # run executes in the originally provisioned snapshot (the replayed
        # body's scope.root), so its STARTED row must name that binding or the
        # snapshot and the child's work become GC food once no longer pending.
        snapshot_id=str(record.get("snapshot_id") or ""),
        execution_root=str(record.get("execution_root") or ""),
        baseline_sha=str(record.get("baseline_sha") or ""),
        target_root=str(record.get("target_root") or ""),
        authority_source=str(record.get("authority_source") or ""),
        # Carried opaquely VERBATIM — recovery never re-authorizes a target (R1-2).
        resource_ref=record.get("resource_ref") if isinstance(record.get("resource_ref"), dict) else {},
        # The GRANTED shape on the recovered OBJECT too, not only the row (gate
        # fix 8c): the memo must answer the same lookups the replay does.
        access=str(body.get("access") or ""),
        mode=str(body.get("mode") or ""),
        isolation=str(execution.get("isolation") or ""),
        delegated=bool(execution.get("delegated")))
    _custody().record_started(drive_root, custody, shape={
        # The stored invocation is the single source of a replay's facts — the same
        # doctrine the explicit retry path follows.
        "effort": str(body.get("effort") or ""), "access": str(body.get("access") or ""),
        "mode": str(body.get("mode") or ""), "isolation": str(execution.get("isolation") or ""),
        "delegated": bool(execution.get("delegated")), "root": str(scope.get("root") or ""),
        "recovered_from_pending_invocation": True,
    })
    return _custody()._reconcile_one(drive_root, gateway, custody)


def _retire_recovered_registration(gateway: Any, record: Dict[str, Any]) -> bool:
    """Discharge the registration an ORIGINAL attempt owned, when its invocation dies."""
    if _record_persistent(record) or not (record.get("project_owned") and record.get("project_id")):
        return False
    try:
        gateway.remove_project(record["project_id"])
        return True
    except Exception as exc:
        if _custody().daemon_says_absent(exc):
            return True
        log.warning("Failed to retire project %s of a dead invocation",
                    record["project_id"], exc_info=True)
        return False


def _reconcile_one(drive_root: Any, gateway: Any, custody: RunCustody) -> Dict[str, Any]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.delegate_integration import capture_stranded_patch

    try:
        detail = gateway.get_run(custody.run_id)
    except ClaudexorUnavailable as exc:
        if _custody().daemon_says_absent(exc):
            _custody().close_absent_run(drive_root, gateway, custody, "reconcile_absent")
            result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "absent"}
        else:
            _custody().record_containment_fault(drive_root, custody, "reconcile_unreadable", f"{exc.code}: {exc}")
            result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "unreadable"}
        # NO capture here (C1-R2): across the D30 boundary an absent run may
        # still be WRITING its snapshot; an eager capture would freeze an
        # incomplete patch the idempotent core then serves forever. Custody
        # closes, the snapshot survives (undisposed - GC keeps it), the duty
        # surfaces via undisposed_patches(); capture happens at disposition.
        _custody().emit(drive_root, _custody().RECONCILED, result)
        return result
    if _custody().is_terminal(detail):
        settled = _custody().settle_run(drive_root, gateway, custody, detail)
        # Sweep custody REPLAYS with staged fields, so unlike the wait path
        # the omission is already knowable here - and an ownerless run is the
        # one nobody will come back to read (the D7 launched-never-collected
        # half becomes durable in this row instead of inferred).
        _custody().record_settled_unread(drive_root, custody)
        # The action names the ATTEMPT; the facts ride separately (the old
        # shape wrote action="settled" even when the returned flag was false).
        result = {"run_id": custody.run_id, "task_id": custody.task_id,
                  "action": "settle_attempted",
                  "settled": settled["settled"],
                  "project_retired": settled.get("project_retired"),
                  **_custody().output_disposition(custody)}
        # The C1 half: a TERMINAL DETAIL proves the run is over, so the sweep — its
        # last terminal observer — captures the diff eagerly here.
        result.update(capture_stranded_patch(drive_root, custody))
    else:
        cancelled = _custody().cancel_and_verify(drive_root, gateway, custody, "owner_task_gone")
        result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "cancelled",
                  "outcome": cancelled["outcome"], **_custody().output_disposition(custody)}
        # Capture ONLY on a verified terminal receipt (the read-back proved the run
        # over). A cancel merely requested leaves the run live and its snapshot
        # still being written; a cancel confirmed by ABSENCE proves nothing about
        # the run (same unknowable-state doctrine as above) — both leave the
        # capture to disposition.
        if cancelled["state"] in _custody().TERMINAL_STATES:
            result.update(capture_stranded_patch(drive_root, custody))
    _custody().emit(drive_root, _custody().RECONCILED, result)
    return result
