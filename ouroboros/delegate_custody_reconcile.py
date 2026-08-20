"""Reconciliation of delegated runs: the settle-or-cancel sweeps and their recovery.

The open-run and pending-invocation projections, the loop-exit release, the
kill-path and orphan sweeps, stranded-patch capture and the per-run
settle-or-cancel core. Extracted from delegate_custody.py (v7 DEL1 split);
delegate_custody.py re-exports every name, so every existing reference — the
tools, the supervisor sweeps, the tests and monkeypatch targets — still finds
them there.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

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


def open_runs(drive_root: Any) -> List[RunCustody]:
    """Runs with a durable start and no durable settlement."""
    return [custody for custody in _custody().replay(drive_root).values() if not custody.settled]


def pending_invocations(drive_root: Any,
                        rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Invocations with a durable request row, no bound run, no definite refusal.

    The launched-never-collected class one step EARLIER than ``open_runs``: a
    worker death between the accepted POST and ``record_started`` leaves only the
    ``START_REQUESTED`` row. Facts come from the FIRST request row (the minting,
    same rule as ``invocation_record``); a record whose canonical body never
    landed is excluded (nothing byte-identical can be replayed). ``rows`` shares
    one pre-read snapshot with ``replay`` (atomic payload busy claim)."""
    found: Dict[str, Dict[str, Any]] = {}
    state: Dict[str, str] = {}
    for row in rows if rows is not None else _custody()._iter_rows(_custody().event_log_path(drive_root)):
        invocation_id = str(row.get("invocation_id") or "")
        if not invocation_id:
            continue
        kind = str(row.get("type") or "")
        if kind == _custody().START_REQUESTED and invocation_id not in found:
            found[invocation_id] = {
                "invocation_id": invocation_id,
                "task_id": str(row.get("task_id") or ""),
                "request": row.get("request") if isinstance(row.get("request"), dict) else None,
                "route": str(row.get("route") or ""),
                "project_id": str(row.get("project_id") or ""),
                "project_owned": bool(row.get("project_owned")),
                "idempotency_key": str(row.get("idempotency_key") or ""),
                "root_task_id": str(row.get("root_task_id") or ""),
                "parent_task_id": str(row.get("parent_task_id") or ""),
                # The FULL C1 isolation binding, not just the GC key: recovery
                # re-records it on the bound run's STARTED row (snapshot_id alone
                # left recovered runs bindingless and their snapshots GC-deleted).
                "snapshot_id": str(row.get("snapshot_id") or ""),
                "execution_root": str(row.get("execution_root") or ""),
                "baseline_sha": str(row.get("baseline_sha") or ""),
                "target_root": str(row.get("target_root") or ""),
                "authority_source": str(row.get("authority_source") or ""),
                "resource_ref": row.get("resource_ref") if isinstance(row.get("resource_ref"), dict) else {},
            }
        elif kind == _custody().STARTED:
            state[invocation_id] = "started"
        elif kind == _custody().START_FAILED and row.get("definite") is True \
                and state.get(invocation_id) != "started":
            state[invocation_id] = "failed_definite"
    return [record for invocation_id, record in found.items()
            if state.get(invocation_id, "pending") == "pending"
            and isinstance(record["request"], dict) and record["request"]]


def release_task_runs(drive_root: Any, task_id: str, *,
                      gateway_factory: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Settle or cancel the runs a task still holds, as its loop exits.

    The in-process twin of ``reconcile_orphaned_runs``: this runs in the very process
    that started them, so the memo IS complete here and the durable scan is not needed —
    a task that delegated nothing pays nothing. The durable path still covers the case
    this one cannot: a worker that died before reaching its own teardown. Without this,
    a terminalized parent left its run mutating until the next 10-minute sweep.
    """
    mine = str(task_id or "")
    held = [c for c in list(_custody()._CUSTODY.values()) if c.task_id == mine and mine and not c.settled]
    return _reconcile_each(drive_root, held, gateway_factory) if held else []


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
    if not held and not stray:
        return []
    return _reconcile_each(drive_root, held, gateway_factory, pending=stray)


def reconcile_orphaned_runs(
    drive_root: Any,
    running_task_ids: Optional[set] = None,
    *,
    gateway_factory: Optional[Callable[[], Any]] = None,
) -> List[Dict[str, Any]]:
    """Settle or cancel every open run whose owning task is no longer running.

    The owner-is-gone predicate is the SAME one ``process_custody.reap_orphaned_processes``
    already uses (the supervisor's live task set), so a delegated run and a spawned
    process cannot disagree about whether their owner still exists. ``running_task_ids``
    of None means UNKNOWN and reconciles nothing — never mass-cancel on missing info.
    """
    if running_task_ids is None:
        return []
    orphans = [c for c in _custody().open_runs(drive_root) if c.task_id and c.task_id not in running_task_ids]
    # The class ONE STEP EARLIER (P34R.2): an invocation whose POST the daemon may have
    # accepted but whose worker died before record_started has no run row for the sweep
    # above to find — a live mutating run nobody could ever collect. Recovered here on
    # the SAME owner-is-gone predicate; a pending invocation whose owner is ALIVE stays
    # untouched, because that owner holds the retry token and decides.
    stray = [record for record in _custody().pending_invocations(drive_root)
             if record["task_id"] and record["task_id"] not in running_task_ids]
    return _reconcile_each(drive_root, orphans, gateway_factory, pending=stray)


def _reconcile_each(drive_root: Any, runs: List[RunCustody],
                    gateway_factory: Optional[Callable[[], Any]],
                    pending: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One transport, one settle-or-cancel pass. Shared by both release surfaces.

    ``pending`` is the durable sweep's extra duty: START_REQUESTED-only invocations
    (P34R.2). The in-process twin ``release_task_runs`` never passes it — its memo is
    run-keyed and cannot name an unbound invocation — so that class is covered by the
    startup/periodic sweep, within its cadence.
    """
    if not runs and not pending:
        return []
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if gateway_factory is None:
        # The startup sweep REAPS the previous generation's owned daemon right before
        # calling here, so a bare discovery-only gateway always found a corpse and the
        # whole reconciliation silently no-opped on every restart — open runs stayed
        # unsettled until the next delegate_start happened to revive the daemon. The
        # ensure path starts our own daemon when there is real work to reconcile
        # (never on the empty early-return above), and as a side effect activates a
        # staged runtime update the old always-running daemon could never adopt.
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
        root_task_id=str(record.get("root_task_id") or ""),
        parent_task_id=str(record.get("parent_task_id") or ""),
        # The sweep runs against the canonical root; a recovered run's ledger row
        # belongs there like every other (P34R.1).
        ledger_root=str(drive_root),
        idempotency_key=str(record["idempotency_key"]), invocation_id=invocation_id,
        # The C1 isolation binding survives recovery VERBATIM: the recovered run
        # executes in the snapshot the original attempt provisioned (the replayed
        # body's scope.root), so its STARTED row must name that binding or the
        # snapshot — and the child's work in it — becomes GC food the moment the
        # invocation stops being pending.
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
    if not (record.get("project_owned") and record.get("project_id")):
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


def _capture_stranded_patch(drive_root: Any, run: RunCustody) -> Dict[str, Any]:
    """Capture a reconciled mutating run's diff into the ordinary patch artifact.

    The reconcile path is the ONLY terminal observer a dead-owner run gets, so
    without this the child's work stayed in the snapshot with no captured patch
    and no apply/reject material — stranded, invisible, and one binding loss away
    from GC. Called ONLY where a terminal receipt PROVES the run is over (C1-R2):
    a run closed absent/unreadable has unknowable state, and freezing a patch
    there would put a "captured" receipt over work the child might still be
    writing — those runs are captured lazily at disposition instead. Reuses the
    one existing capture primitive (idempotent, durable ``PATCH_CAPTURED`` row);
    capture ONLY — the apply/reject decision belongs to a live owner and is NEVER
    taken by a sweep. Fail-soft: a capture error is disclosed in the reconcile
    row, and the snapshot persists either way because the run has no recorded
    disposition.
    """
    if not (run.execution_root and run.settled and not run.patch_disposed):
        return {}
    try:
        from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

        block = capture_terminal_patch_for_drive(drive_root, run) or {}
    except Exception:
        log.warning("Reconcile patch capture failed for %s", run.run_id, exc_info=True)
        return {"patch_capture": "failed", "patch_disposition": "pending"}
    return {"patch_capture": str(block.get("status") or ""),
            "patch_artifact": block.get("patch_artifact"),
            # The typed disposition-pending disclosure: this rides the durable
            # RECONCILED row, and the health surface (``undisposed_patches``)
            # keeps the fact visible until an explicit apply/reject lands.
            "patch_disposition": "pending"}


def _reconcile_one(drive_root: Any, gateway: Any, custody: RunCustody) -> Dict[str, Any]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        detail = gateway.get_run(custody.run_id)
    except ClaudexorUnavailable as exc:
        if _custody().daemon_says_absent(exc):
            _custody().close_absent_run(drive_root, gateway, custody, "reconcile_absent")
            result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "absent"}
        else:
            _custody().record_containment_fault(drive_root, custody, "reconcile_unreadable", f"{exc.code}: {exc}")
            result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "unreadable"}
        # NO capture here (C1-R2): an absent run's state is unknowable from this
        # daemon — across the D30 owned-daemon provisioning boundary the child may
        # still be alive and WRITING to the snapshot, and an eager capture would
        # freeze a potentially incomplete patch which the idempotent capture core
        # would then serve forever. Custody closes, the snapshot stays preserved
        # (undisposed, so the GC keeps it), the obligation surfaces through
        # ``undisposed_patches()``, and the capture happens at disposition
        # (``integrate_delegated_patch``) — the honest latest-possible point.
        _custody().emit(drive_root, _custody().RECONCILED, result)
        return result
    if _custody().is_terminal(detail):
        settled = _custody().settle_run(drive_root, gateway, custody, detail)
        # The sweep's custody REPLAYS with the staged fields on it, so unlike the wait
        # path this is a place the omission is already knowable — and a run whose owner
        # is gone is exactly the one nobody will ever come back to read.
        _custody().record_settled_unread(drive_root, custody)
        # The D7 half of the disposition: a reconciled run whose staged artifact was
        # never acknowledged is the "launched and never collected" shape, and this row
        # is where that fact becomes durable instead of inferred.
        result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "settled",
                  "settled": settled["settled"], **_custody().output_disposition(custody)}
        # The C1 half: a TERMINAL DETAIL proves the run is over, so the sweep — its
        # last terminal observer — captures the diff eagerly here.
        result.update(_capture_stranded_patch(drive_root, custody))
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
            result.update(_capture_stranded_patch(drive_root, custody))
    _custody().emit(drive_root, _custody().RECONCILED, result)
    return result
