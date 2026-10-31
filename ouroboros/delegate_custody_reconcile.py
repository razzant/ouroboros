"""Reconciliation of delegated runs: the settle-or-cancel sweeps and their recovery.

The open-run and pending-invocation projections, the loop-exit release, the
kill-path, tree-Stop and orphan sweeps and the per-run settle-or-cancel core. Extracted
from delegate_custody.py (v7 DEL1 split); delegate_custody.py re-exports every
name, so every existing reference — the tools, the supervisor sweeps, the
tests and monkeypatch targets — still finds them there.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros.delegate_registration_policy import record_persistent as _record_persistent
from ouroboros.observability import timed_phase
from ouroboros.subagent_history import session_request_facts

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
    pre-read snapshot, so a batch of audits pays one read; else the held rows)."""
    if state is None:
        from ouroboros.delegate_custody_current import held_runs

        state = held_runs(drive_root)
    return [custody for custody in state.values() if not custody.settled]


def run_tree_id(custody: RunCustody) -> str:
    """The task tree a run belongs to: its recorded root, else its own task (the custody store's tree key)."""
    return str(custody.root_task_id or custody.task_id or "")


def tree_open_runs(drive_root: Any, task_id: str) -> List[RunCustody]:
    """The open runs a cascade Stop of ``task_id`` reaches (``stop_tree_runs``): the task's own and, when it is a
    tree's root, every run of that tree, which the work projection counts as that root's live members."""
    target = str(task_id or "")
    return [custody for custody in open_runs(drive_root)
            if target and target in {run_tree_id(custody), str(custody.task_id or "")}]


def stop_tree_runs(drive_root: Any, task_id: str, *,
                   gateway_factory: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Cancel the open runs a cascade Stop of ``task_id`` reaches (``tree_open_runs``), behind the owner's verdict.

    A run outlives every owner terminal that was not a verdict (``_reconcile_one``'s inverted floor), so
    the open runs of a settled tree waited for exactly this one: the owner's cancellation, which also
    speaks for a review panel's run. A Stop is a release surface: attach-only, no daemon is started and no
    pending invocation is replayed (exit does not admit). An unconfirmed cancel leaves the run open.
    """
    from ouroboros.task_results import STATUS_CANCELLED

    runs = tree_open_runs(drive_root, task_id)
    if not runs:
        return []
    return _reconcile_each(drive_root, runs, gateway_factory or attach_only_gateway,
                           deliberate_terminal=STATUS_CANCELLED)


def pending_invocations(drive_root: Any,
                        rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Invocations with a durable request row, no bound run, no definite refusal.

    The launched-never-collected class one step EARLIER than ``open_runs``: a
    worker death between the accepted POST and ``record_started`` leaves only the
    ``START_REQUESTED`` row. Facts come from the FIRST request row (the minting,
    same rule as ``invocation_record``). Legacy rows without a body stay excluded;
    an unreadable request reference retains identity with ``request=None``.
    Reconciliation cannot replay it without that body. ``rows`` shares
    one pre-read snapshot with ``replay`` for consistent inventory reads."""
    from ouroboros.delegate_pending import pending_invocations as replay_pending

    return replay_pending(drive_root, rows)


def attach_only_gateway() -> Any:
    """The release surfaces' transport: attach to a serving daemon, never start one.

    Exit does not admit (DEVELOPMENT "Process Custody Rule"): a loop exit, a kill
    boundary or a stop settles what is already running and provisions, downloads,
    spawns and admits nothing. ``read_owned_gateway`` raises the typed
    ``ClaudexorUnavailable`` when no daemon answers, which every reconcile pass
    already treats as "skipped, the sweep re-judges".
    """
    from ouroboros.claudexor_daemon import read_owned_gateway

    return read_owned_gateway()


_loop_exit_audit: ContextVar[Optional[Dict[str, Any]]] = ContextVar("loop_exit_audit", default=None)


@contextmanager
def defer_loop_exit_audit(usage: Dict[str, Any]):
    """Retain only audit facts for delivery; resource release still finishes in this scope."""
    token = _loop_exit_audit.set(usage)
    try:
        yield
    finally:
        _loop_exit_audit.reset(token)


@timed_phase("release_task_runs")
def release_task_runs(drive_root: Any, task_id: str, *,
                      gateway_factory: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Run the one non-panic terminal custody boundary for a normal loop exit.

    A release path: ``gateway_factory`` defaults to ``attach_only_gateway``, and a
    pending invocation of this task is never replayed here (``reconcile_task_runs``).
    """

    from ouroboros.delegate_terminal import (
        record_terminal_reconciliation,
        terminal_reconcile_task,
    )

    deferred = _loop_exit_audit.get()
    result = terminal_reconcile_task(
        drive_root, task_id, gateway_factory=gateway_factory or attach_only_gateway,
        trigger="loop_exit", audit=deferred is None,
    )
    if deferred is None:
        record_terminal_reconciliation(drive_root, task_id, result)
    else:
        deferred["delegate_terminal_reconciliation"] = result
    return list(result.get("outcomes") or [])


@timed_phase("custody_reconcile", within="release_task_runs")
def reconcile_task_runs(drive_root: Any, task_id: str, *,
                        gateway_factory: Optional[Callable[[], Any]] = None,
                        deliberate_terminal: str = "",
                        run_ids: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    """Settle or cancel ONE task's open runs from the DURABLE rows (kill path).

    The supervisor-side twin of ``release_task_runs`` for a task whose worker was
    just KILLED (cancellation custody / reap): the graceful release never ran and
    its memo died with the process, so the durable rows are the only complete
    view. Cheap when the task delegated nothing. A release surface: the task's
    pending (START_REQUESTED-only) invocations are NOT replayed here — a re-POST
    starts work, and exit does not admit; they stay durable for the admitted
    startup/periodic sweep (``reconcile_orphaned_runs``), which recovers them
    under the same key once the owner is gone. A review panel's own pending
    invocation, which that sweep leaves to its panel (#1006), is named here once
    as retained, through no transport. ``gateway_factory`` defaults to
    ``attach_only_gateway``; a kill boundary names it explicitly.

    ``deliberate_terminal`` is the terminal status the KILLING caller is about to
    write, when that terminal is the task's own deliberate end (an owner
    cancellation). The kill boundary audits custody before its terminal write, so
    the durable result the inverted floor otherwise reads does not exist yet; the
    caller already knows the verdict and states it here rather than having the
    host guess or re-read a file that is not there. Empty means "no verdict from
    me", which is what every host bound (deadline, reap) and every read-only
    sweep passes.
    """
    mine = str(task_id or "")
    if not mine:
        return []
    from ouroboros.delegate_custody_current import task_snapshot
    from ouroboros.delegate_custody_current import held_runs

    view = task_snapshot(drive_root, mine)  # one read; pending starts remain the admitted sweep's duty
    if run_ids is not None:
        view = {"runs": {rid: run for rid, run in view["runs"].items() if rid in run_ids}, "pending": []}
    held = [c for c in view["runs"].values() if not c.settled]
    named = [_retain_panel_invocation(drive_root, record) for record in view["pending"]
             if _custody().review_owned_source(record.get("source"))]
    # Also the registration retry lane for the task's OWN settled-but-owned
    # runs in retire-eligible projects (a one-shot process may never see a
    # sweep tick); deferred registrations stay with the periodic sweep. A
    # project with ANY unsettled run is live; unsettled runs are held rows.
    owned = [r for r in view["runs"].values() if r.project_owned and r.project_id and r.settled]
    live = ({r.project_id for r in held_runs(drive_root).values() if r.project_id and r.run_id and not r.settled}
            if owned else set())
    owed = [r for r in owned if r.project_id not in live]
    if not held and not owed:
        return named  # pending invocations alone open no transport: the sweep or their panel owns them
    return [*named, *_reconcile_each(drive_root, held, gateway_factory or attach_only_gateway,
                                     deliberate_terminal=deliberate_terminal)]


def reconcile_orphaned_runs(
    drive_root: Any,
    running_task_ids: Optional[Any] = None,
    *,
    gateway_factory: Optional[Callable[[], Any]] = None,
    recoverable_task_ids: Optional[set] = None,
    recoverable_handles: Optional[set] = None,
    replay_pending: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    """Settle or cancel every open run whose owning task is no longer running.

    The owner-is-gone predicate is the SAME one ``process_custody.reap_orphaned_processes``
    already uses (the supervisor's live task set), so a delegated run and a spawned
    process cannot disagree about whether their owner still exists. ``running_task_ids``
    is that set, or a zero-arg callable producing it; None means UNKNOWN and reconciles
    nothing — never mass-cancel on missing info.

    ``gateway_factory`` is named by every production caller: the admitted startup/periodic
    sweep ensures, a stop or the owner's Restart attaches; a caller that names none gets
    the attach-only transport (a bare sweep can only attach). ``replay_pending`` is the admission
    predicate for re-POSTing an orphaned pending invocation (``_recover_pending_invocation``);
    it is re-asked immediately before each replay, so a stop that lands mid-pass defers
    the rest. ``None`` (the release and Restart surfaces) never replays: the rows stay
    durable and are reported ``invocation_retained``.
    """
    if running_task_ids is None:
        return []
    # CANDIDATES FIRST, LIVENESS SECOND, exactly as the process reaper reads its ledger:
    # replaying the custody log takes seconds, and a candidate exists ⇒ its owner was
    # registered earlier (admission takes ``_queue_lock`` before any spawn), so an owner
    # absent from the LATER snapshot is really gone rather than merely not yet admitted.
    from ouroboros.delegate_custody_current import held_runs
    candidates = [c for c in held_runs(drive_root).values()
                  if not c.settled]  # settled collection debt belongs to the body's independent held-row reader
    # The class ONE STEP EARLIER (P34R.2): an invocation whose POST the daemon may have
    # accepted but whose worker died before record_started has no run row for the sweep
    # above to find — a live mutating run nobody could ever collect. Recovered here on
    # the SAME owner-is-gone predicate; a pending invocation whose owner is ALIVE stays
    # untouched, because that owner holds the retry token and decides.
    unbound = _custody().pending_invocations(drive_root)
    live = running_task_ids() if callable(running_task_ids) else running_task_ids
    if live is None:
        return []  # the snapshot could not be taken: still UNKNOWN, still touch nothing
    live_or_reserved = set(live)
    protected = set(recoverable_handles or ())
    if recoverable_task_ids:
        from ouroboros.delegate_recovery import recoverable_handles as saved_handles

        protected.update(pair for pair in saved_handles(drive_root) if pair[0] in recoverable_task_ids)
    orphans = [c for c in candidates if c.task_id and c.task_id not in live_or_reserved
               and (c.task_id, c.run_id, c.invocation_id) not in protected
               and (c.task_id, c.run_id, "") not in protected]
    # A review surface's pending invocation belongs to its panel, which rejoins it by
    # its exact key (#1006): this sweep would only re-report "retained" every pass.
    stray = [record for record in unbound
             if record["task_id"] and record["task_id"] not in live_or_reserved
             and (record["task_id"], "", record["invocation_id"]) not in protected
             and not _custody().review_owned_source(record.get("source"))]
    return _reconcile_each(drive_root, orphans, gateway_factory or attach_only_gateway, pending=stray,
                           live_task_ids=live_or_reserved, recoverable_handles=protected, replay_pending=replay_pending)


def _reconcile_each(drive_root: Any, runs: List[RunCustody],
                    gateway_factory: Callable[[], Any],
                    pending: Optional[List[Dict[str, Any]]] = None,
                    deliberate_terminal: str = "", live_task_ids=None, recoverable_handles=(),
                    replay_pending: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
    """One transport, one settle-or-cancel pass. Shared by every reconcile surface.

    ``gateway_factory`` has NO default: the caller names whether this pass may ensure
    (the admitted sweep) or only attach (every release, kill, stop and Restart path).
    ``pending`` is the durable sweep's extra duty: START_REQUESTED-only invocations
    (P34R.2), passed by ``reconcile_orphaned_runs`` alone; the task-scoped
    ``reconcile_task_runs`` leaves them to the sweep. A pending row is replayed only
    while ``replay_pending()`` answers True, asked again right before each re-POST;
    without the predicate the row is retained. The registration pass at the end is
    the third duty: settlement no longer discharges project ownership, and the
    startup/periodic sweep surface is where settled-but-owned registrations get
    their retry lane (the task-scoped surface can exit early with none open).
    """
    if gateway_factory is None:
        raise ValueError("delegated-run reconciliation needs an explicit gateway factory: "
                         "ensure_owned_gateway for the admitted sweep, attach_only_gateway otherwise")
    # Registration duty is real work only for a retire-ELIGIBLE project:
    # a deferred one must not spin the daemon up.
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.delegate_custody_current import held_runs

    # Owned registrations and unsettled runs are HELD rows: one indexed read.
    snapshot = list(held_runs(drive_root).values())
    # Only a still-owned registration can be retired, so only its rows are asked
    # (each ask reads task results); one continuable row keeps the whole project.
    owned = {row.project_id for row in snapshot if row.project_owned and row.project_id}
    unsettled_projects: set = set()
    for row in snapshot:
        if (row.project_id in owned and row.project_id not in unsettled_projects and row.run_id
                and (not row.settled or still_continuable(
                    drive_root, row, live_task_ids, recoverable_handles=recoverable_handles))):
            unsettled_projects.add(row.project_id)
    registrations = [row for row in snapshot
                     if row.project_owned and row.project_id
                     and row.project_id not in unsettled_projects]
    def _replay_admitted() -> bool:
        return replay_pending is not None and bool(replay_pending())

    retained = [] if _replay_admitted() else [
        {"invocation_id": str(record["invocation_id"]), "task_id": str(record["task_id"]),
         "action": "invocation_retained", "reason": "replay_not_admitted"}
        for record in pending or []]
    if not runs and not registrations and (not pending or retained):
        return retained  # no transport is opened for rows this pass may not replay
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    gateway = None
    outcomes: List[Dict[str, Any]] = list(retained)
    settlement_pending = any(not row.settled for row in runs)
    try:
        try:
            gateway = gateway_factory()
            gateway.handshake()
        except ClaudexorUnavailable:
            # The body collects retained sources independently. A transport gap
            # proves nothing about execution; control custody stays open.
            log.debug("delegated-run reconciliation: transport unavailable", exc_info=True)
            return retained
        for custody in runs:
            outcomes.append(_custody()._reconcile_one(
                drive_root, gateway, custody,
                deliberate_terminal=deliberate_terminal,
            ))
        for record in [] if retained else (pending or []):
            if not _replay_admitted():  # rechecked at dispatch: a stop mid-pass defers the rest
                outcomes.append({"invocation_id": str(record["invocation_id"]),
                                 "task_id": str(record["task_id"]),
                                 "action": "invocation_retained", "reason": "replay_not_admitted"})
                continue
            outcomes.append(_recover_pending_invocation(
                drive_root, gateway, record, deliberate_terminal=deliberate_terminal,
            ))
        # A newly settled run may make its project eligible. Collection of an
        # already-settled run cannot: keep the pre-pass continuation deferral
        # without reading all of that project's historical sharers again.
        if registrations or settlement_pending:
            _custody().retire_settled_registrations(drive_root, gateway, live_task_ids=live_task_ids,
                                                  recoverable_handles=recoverable_handles)
    finally:
        try:
            if gateway is not None:
                gateway.close()
        except Exception:
            log.debug("delegated-run reconciliation close failed", exc_info=True)
    return outcomes


def _retain_panel_invocation(drive_root: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    """Name a REVIEW surface's pending invocation as retained: its panel rejoins it (#1006)."""
    result = {"invocation_id": str(record["invocation_id"]), "task_id": str(record["task_id"]),
              "action": "invocation_retained", "reason": "review_panel_owns_invocation"}
    _custody().emit(drive_root, _custody().RECONCILED, result)
    return result


@timed_phase("custody_pending", within="release_task_runs")
def _recover_pending_invocation(drive_root: Any, gateway: Any,
                                record: Dict[str, Any], *,
                                deliberate_terminal: str = "") -> Dict[str, Any]:
    """Recover the run (if any) behind an orphaned pending invocation, idempotently.

    The stored canonical body is re-POSTed under the invocation's own wire key:
    the engine returns the ORIGINAL handle when the first POST was accepted, and
    starts fresh only when the daemon truly never saw it. A definite 4xx retires
    the invocation and its registration; an unknown outcome stays pending.

    A REVIEW surface's own pending invocation is retained untouched: its panel
    rejoins it through ``review_session_custody``, and a delegation-domain
    re-POST here would bind a reviewer nobody on this task asked for (#1006).
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    invocation_id = str(record["invocation_id"])
    task_id = str(record["task_id"])
    if _custody().review_owned_source(record.get("source")):
        return _retain_panel_invocation(drive_root, record)
    from ouroboros.delegate_pending import request_body
    body = request_body(drive_root, record)
    if not isinstance(body, dict) or not body:
        result = {"invocation_id": invocation_id, "task_id": task_id,
                  "action": "invocation_retained", "reason": "invocation_request_unrecorded"}
        _custody().emit(drive_root, _custody().RECONCILED, result)
        return result
    record = {**record, "request": body}
    from types import SimpleNamespace

    from ouroboros.owner_pause import OwnerPauseRefused, run_operation

    request = dict(body)
    try:
        # START_REQUESTED records custody, not an earlier executor handoff.
        # The same-key POST may create a run: submit under Pause exclusion,
        # then join outside the launch lock just like ordinary delegation.
        handle = run_operation(SimpleNamespace(drive_root=drive_root, task_id=task_id,
            root_task_id=record.get("root_task_id") or task_id),
            gateway.start_run, request, idempotency_key=invocation_id)
    except OwnerPauseRefused as exc:
        return {"invocation_id": invocation_id, "task_id": task_id,
                "action": "invocation_retained", "reason": str(exc)}
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        if 400 <= status < 500:
            retired = _retire_recovered_registration(gateway, record)
            _custody().emit(drive_root, _custody().START_FAILED, {
                "run_id": "", "task_id": task_id, "project_id": record["project_id"],
                "project_retired": retired, "reason": f"recovery_refused_{exc.code}",
                "invocation_id": invocation_id, "definite": True,
                **session_request_facts(record["request"], selected_subagent_id=record.get("selected_subagent_id", ""),
                    task_id=task_id, route=record.get("route", ""), processing=record.get("processing") or {}),
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
    execution = body.get("execution") if isinstance(body.get("execution"), dict) else {}
    scope = body.get("scope") if isinstance(body.get("scope"), dict) else {}
    from ouroboros.delegate_plan import plan_facts

    custody = _custody().RunCustody(
        run_id=run_id, task_id=task_id,
        **plan_facts(record),
        route_id=str(record["route"] or body.get("primaryHarness") or ""),
        model=str(body.get("model") or ""),
        profile_id=str(body.get("credentialProfileId") or ""),
        effort=body.get("effort"),
        processing_preference=(record.get("processing") or {}).get("requested"),
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
        continuation_of=str(record.get("continuation_of") or ""),
        capture_id=str(record.get("capture_id") or ""),
        snapshot_task_id=str(record.get("snapshot_task_id") or ""),
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
    return _custody()._reconcile_one(
        drive_root, gateway, custody, deliberate_terminal=deliberate_terminal,
    )


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


def _owner_terminal_is_deliberate(drive_root: Any, task_id: str, *,
                                  review_owned: bool = False) -> bool:
    """Whether the run's owning task ended ON PURPOSE (owner answer B1-A, the floor
    INVERTED: cancel only behind a verdict).

    True only when the owner's durable result is readable, truly terminal, and its
    execution axis says the task finished by its own decision: a completed, degraded
    or best-effort answer, a failure the agent itself declared, or a cancellation.
    A provider or transport death, a worker crash (``infra_failed``), an interrupted
    row, a missing, unreadable or still-running result all answer False and SPARE
    the run: an undignified nanny death must not kill a healthy paid run, and
    "unknown" is exactly the case the answer is about. No reason-code table and no
    special crash branch: the axis the finalizers already write is the class. The
    spared run's own ``maxSeconds`` stays its damage limit (ARCHITECTURE, custody).

    ``review_owned`` raises that floor to the ONE verdict that also speaks for a
    review panel's run: an explicit cancellation. Task association does not confer
    lifecycle authority, so a reviewed task that merely finished — completed,
    degraded, best-effort or failed — is not a verdict about its own reviewer, and
    killing the live panel on it is exactly the incident behind issue #1006. The panel's own bounds (its slot window and the run's ``maxSeconds``)
    stay the reviewer's damage limit.
    """
    from ouroboros.outcomes import (
        EXECUTION_BEST_EFFORT,
        EXECUTION_CANCELLED,
        EXECUTION_DEGRADED,
        EXECUTION_FAILED,
        EXECUTION_OK,
        normalize_outcome_axes,
    )
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    try:
        result = load_task_result(drive_root, str(task_id or ""))
    except Exception:
        return False
    if not isinstance(result, dict) or str(result.get("status") or "") not in _TRULY_TERMINAL_STATUSES:
        return False
    execution = str((normalize_outcome_axes(result).get("execution") or {}).get("status") or "")
    if review_owned:
        return execution == EXECUTION_CANCELLED
    return execution in {EXECUTION_OK, EXECUTION_DEGRADED, EXECUTION_BEST_EFFORT,
                         EXECUTION_FAILED, EXECUTION_CANCELLED}


def _reconcile_collection(drive_root: Any, gateway: Any, custody: RunCustody, *,
                          detail=None) -> Dict[str, Any]:
    """Retain observation and money here; the body's collector owns bulk output and files."""
    from ouroboros.delegate_custody_current import collection_pending, run

    before = run(drive_root, custody.run_id) or custody
    if detail is not None:
        _custody().settle_run(drive_root, gateway, custody, detail)
    after = run(drive_root, custody.run_id) or before
    _custody().record_settled_unread(drive_root, after)
    result = {"run_id": after.run_id, "task_id": after.task_id,
              "action": "settle_attempted", "settled": after.settled,
              "collection": {"status": "pending" if collection_pending(after) else "collected"},
              **_custody().output_disposition(after)}
    if after != before:
        _custody().emit(drive_root, _custody().RECONCILED, result)
    return result


@timed_phase("custody_run", within="release_task_runs")
def _reconcile_one(drive_root: Any, gateway: Any, custody: RunCustody, *,
                   deliberate_terminal: str = "") -> Dict[str, Any]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.task_results import STATUS_CANCELLED

    review_owned = custody.review_owned
    # The caller's claimed verdict speaks for a review panel's run only when it is
    # the cancellation itself. `STATUS_CANCELLED` is the exact literal the two kill
    # boundaries pass here (supervisor/cancel_publication.py and
    # supervisor/task_lifecycle.py); every other host bound — deadline, reap, an
    # ordinary finished task — is not a verdict about the reviewer (issue #1006).
    claimed_verdict_cancels = (
        str(deliberate_terminal or "") == STATUS_CANCELLED if review_owned
        else bool(deliberate_terminal)
    )
    try:
        detail = gateway.get_run(custody.run_id)
    except ClaudexorUnavailable as exc:
        if _custody().daemon_says_absent(exc, gateway=gateway, run=custody):
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
        if review_owned:
            settled = _custody().settle_run(drive_root, gateway, custody, detail)
            _custody().record_settled_unread(drive_root, custody)
            result = {"run_id": custody.run_id, "task_id": custody.task_id,
                      "action": "settle_attempted", **settled, **_custody().output_disposition(custody)}
        else:
            return _reconcile_collection(drive_root, gateway, custody, detail=detail)
    elif not (claimed_verdict_cancels
              or _owner_terminal_is_deliberate(drive_root, custody.task_id,
                                               review_owned=review_owned)):
        # The inverted floor (B1-A): a live run outlives every owner terminal that
        # was not a verdict. Two ways to know the verdict, and the caller's is
        # first because a kill boundary audits custody BEFORE it writes its own
        # terminal (the A4 ordering): `deliberate_terminal` is the status that
        # caller is about to write. Everyone else, including the periodic sweep,
        # reads the durable result. Without either, the run stays live: it settles
        # itself through the terminal arm above when it ends, and its snapshot
        # work becomes an undisposed_patches obligation like any other.
        result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "left_live",
                  "state": str(_custody().summary_of(detail).get("state") or ""),
                  # A spared reviewer names its owner, so the receipt is not read
                  # as an unexplained survival of the task's own delegation.
                  **({"reason": "review_panel_owns_run"} if review_owned else {}),
                  **_custody().output_disposition(custody)}
    else:
        cancelled = _custody().cancel_and_verify(drive_root, gateway, custody, "owner_task_gone")
        result = {"run_id": custody.run_id, "task_id": custody.task_id, "action": "cancelled",
                  "outcome": cancelled["outcome"], **_custody().output_disposition(custody)}
        # settle_run retained a verified terminal read-back. Bulk capture stays
        # with the body or native review owner, including on Stop and Restart.
        # An accepted cancel or absence supplies no terminal execution evidence.
    _custody().emit(drive_root, _custody().RECONCILED, result)
    return result
