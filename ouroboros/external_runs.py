"""A task's delegated runs under one pause policy: observe, stop its own, or stop all.

The ONE observer body behind the loop-side pause, the supervisor-side Resume
grant, the owner Pause's settle tick and the writer census (owner Q8; owner
2026-10-07 fork 1 = A). A FRESH custody read, never a pause row's saved
summary; stops go through the verified cancel seam and their outcomes are
recorded per run. ``requested`` and ``unknown`` are not death: the run stays
under the task's custody and no second writer may start over it.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

from ouroboros.delegate_registration_policy import review_owned_source

log = logging.getLogger(__name__)

# External-run stop outcomes as recorded on the pause row (independent facts,
# never collapsed into one boolean).
EXTERNAL_RUNNING = "running"
EXTERNAL_STOP_REQUESTED = "stop_requested"
EXTERNAL_STOP_CONFIRMED = "stop_confirmed"
EXTERNAL_STOP_UNKNOWN = "stop_unknown"

# The three stop policies one observer translates (owner 2026-10-07 fork 1 = A):
# observe every run; stop the task's OWN runs and let its already-started
# critics finish (the owner's Pause, its settle tick and its Resume); stop all
# (the budget rail). No harness name, surface table or timer decides it: a run
# is a critic only by its recorded custody ``source``
# (``delegate_registration_policy.review_owned_source``).
STOP_POLICY_OBSERVE = "observe_only"
STOP_POLICY_TASK_OWNED = "stop_task_owned"
STOP_POLICY_ALL = "stop_all"
STOP_POLICIES = frozenset({STOP_POLICY_OBSERVE, STOP_POLICY_TASK_OWNED, STOP_POLICY_ALL})


def observe_task_runs(root: Any, task_id: str, *, reason: str = "budget_resume_uncovered_cost",
                      read_error: str = "", request_stop: bool = True,
                      stop_policy: Optional[str] = None,
                      prior: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ONE observer body behind the loop-side pause and the supervisor-side grant.

    A FRESH custody read for ``task_id`` on ``root`` (never a pause row's saved
    summary), requesting a stop for every run the policy names — pre-terminal
    subscription cost coverage cannot be proved from the ledger (owner Q8), so
    its remaining cost is uncovered/unknown while it runs — and recording the
    typed outcome through the verified cancel seam. ``requested`` and
    ``unknown`` are NOT death: the run stays under this task's custody and no
    second writer may be started over it. An unreadable custody store
    (``read_error`` from the caller, or the replay failing here) is a typed
    ``custody_read=failed`` observation, never an empty (clean-looking) list.

    ``stop_policy`` is the one translator: ``observe_only`` reports every open
    run ``running``; ``stop_task_owned`` (the owner's Pause, owner 2026-10-07
    fork 1 = A) cancels the task's own runs with their work retained by the
    engine and OBSERVES its already-started critics (``review_owned=True``
    rows, never cancelled); ``stop_all`` (the budget rail) cancels every run.
    ``request_stop`` is the historical boolean spelling of the two extremes.
    ``prior`` is this same pause's earlier observation: a run whose stop it
    already issued is only re-READ (terminal or absent confirms it), never
    cancelled again — a repeated control is neither proof nor progress.
    """
    if stop_policy is None:
        stop_policy = STOP_POLICY_ALL if request_stop else STOP_POLICY_OBSERVE
    if stop_policy not in STOP_POLICIES:
        raise ValueError(f"unknown stop policy: {stop_policy!r}")
    runs: List[Any] = []
    pending_rows: List[Dict[str, Any]] = []
    if not read_error:
        try:
            from ouroboros import delegate_custody as custody

            mine = str(task_id or "")
            # The memo silently falls back to a lenient read that skips an
            # unreadable segment; probe the chain first so hidden custody is
            # UNKNOWN, never "no open runs" (Astra run-a882315dbcd7 #2).
            if custody.custody_log_unreadable(pathlib.Path(root)):
                raise OSError("custody_log_unreadable")
            from ouroboros.delegate_custody_memo import custody_rows_with_integrity

            # ONE snapshot feeds both projections: a START_REQUESTED that becomes
            # STARTED between two reads must land in one of them (Astra 6fe5 #1),
            # and its integrity is judged on that same read. An unparseable custody
            # line naming this task may hide its request.
            rows_read, malformed = custody_rows_with_integrity(pathlib.Path(root), mine)
            snapshot = list(rows_read)
            if malformed is None or malformed:
                raise OSError(f"custody_rows_incomplete:{'unknown' if malformed is None else malformed}")
            runs = [run for run in custody.replay(pathlib.Path(root), rows=snapshot).values()
                    if str(getattr(run, "task_id", "") or "") == mine and not getattr(run, "settled", True)]
            # A START_REQUESTED whose response was lost has no run id yet but may
            # be a live remote writer: unknown custody, never absence (#3).
            from ouroboros.delegate_pending import pending_invocations

            pending = [row for row in pending_invocations(pathlib.Path(root), rows=snapshot)
                       if str(row.get("task_id") or "") == mine]
            pending_rows = [{"run_id": "", "invocation_id": str(row.get("invocation_id") or ""),
                             "route": str(row.get("route") or ""),
                             "cost_coverage": "unproven_preterminal", "stop_policy": "reconcile_first",
                             "review_owned": review_owned_source(row.get("source")),
                             "state": EXTERNAL_STOP_UNKNOWN, "stop_outcome": "pending_invocation_unbound",
                             "detail": ""} for row in pending]
        except Exception as exc:
            log.warning("External custody rows unreadable for %s", task_id, exc_info=True)
            read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    if read_error:
        # Held as UNKNOWN on the pause row: the grant re-reads custody and
        # refuses while it stays unreadable (never "no runs").
        return {"runs": [], "observed_at": time.time(), "custody_read": "failed",
                "error": read_error, "coverage_basis": "custody_unreadable"}
    if not runs:
        if pending_rows:
            return {"runs": pending_rows, "observed_at": time.time(), "custody_read": "ok",
                    "coverage_basis": "pending_invocations_unbound"}
        return {"runs": [], "observed_at": time.time(), "custody_read": "ok", "coverage_basis": "no_open_runs"}

    def _row(run: Any, **fields: Any) -> Dict[str, Any]:
        return {"run_id": str(getattr(run, "run_id", "") or ""),
                "route": str(getattr(run, "route", "") or getattr(run, "route_id", "") or ""),
                "cost_coverage": "unproven_preterminal", "stop_policy": "observe_only",
                "review_owned": review_owned_source(getattr(run, "source", "")),
                "state": EXTERNAL_RUNNING, "stop_outcome": "", "detail": "", **fields}

    if stop_policy == STOP_POLICY_OBSERVE:
        return {"runs": [_row(run) for run in runs] + pending_rows, "observed_at": time.time(),
                "custody_read": "ok", "coverage_basis": "observed_without_stop_request"}
    stopping = [run for run in runs
                if stop_policy == STOP_POLICY_ALL or not review_owned_source(getattr(run, "source", ""))]
    # The owner's exception: an already-started critic finishes the same check
    # separately; it is observed beside the task's own stopped runs, never cancelled.
    rows = [_row(run) for run in runs if run not in stopping]
    issued = {str(row.get("run_id") or "") for row in ((prior or {}).get("runs") or [])
              if isinstance(row, dict) and row.get("stop_policy") == "request_stop"
              and not str(row.get("stop_outcome") or "").startswith("not_issued")}
    gateway = None
    if stopping:
        try:
            from ouroboros.gateways.claudexor import ClaudexorGateway

            gateway = ClaudexorGateway()
            gateway.handshake()
        except Exception as exc:
            log.warning("Budget pause cannot reach the harness gateway to request stops: %s", exc)
            gateway = None
    try:
        from ouroboros import delegate_custody as custody

        for run in stopping:
            row = _row(run, stop_policy="request_stop", review_owned=False)
            run_id = row["run_id"]
            if gateway is None:
                row.update(state=EXTERNAL_STOP_UNKNOWN, stop_outcome=(
                    "requested_unverified_gateway_unavailable" if run_id in issued
                    else "not_issued_gateway_unavailable"))
            else:
                try:
                    result = (_reread_requested_stop(root, gateway, run) if run_id in issued
                              else custody.cancel_and_verify(pathlib.Path(root), gateway, run, reason))
                    outcome = str(result.get("outcome") or "")
                    row.update(stop_outcome=outcome, detail=str(result.get("detail") or ""))
                    if outcome == custody.CANCEL_CONFIRMED:
                        row["state"] = EXTERNAL_STOP_CONFIRMED
                    elif outcome == getattr(custody, "CANCEL_REQUESTED", "requested"):
                        row["state"] = EXTERNAL_STOP_REQUESTED
                    else:
                        row["state"] = EXTERNAL_STOP_UNKNOWN
                except Exception as exc:
                    row.update(state=EXTERNAL_STOP_UNKNOWN, stop_outcome=f"error:{type(exc).__name__}",
                               detail=str(exc)[:300])
            rows.append(row)
    finally:
        if gateway is not None:
            try:
                gateway.close()
            except Exception:
                log.debug("Gateway close after pause stop requests failed", exc_info=True)
    # Open runs still get their stop requests; unbound invocations ride beside them.
    return {"runs": rows + pending_rows, "observed_at": time.time(), "custody_read": "ok",
            "coverage_basis": "preterminal_subscription_coverage_unprovable"}


def _reread_requested_stop(root: Any, gateway: Any, run: Any) -> Dict[str, Any]:
    """Re-READ a run whose stop this pause already issued: no second control.

    Only a terminal read-back (settled through the existing custody writer) or
    the daemon answering it has no such run confirms the stop; anything else
    keeps the earlier ``requested`` fact, and an unreachable read is unknown.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        detail = gateway.get_run(run.run_id)
    except ClaudexorUnavailable as exc:
        if custody.daemon_says_absent(exc):
            custody.close_absent_run(pathlib.Path(root), gateway, run, "get_run_absent")
            return {"outcome": custody.CANCEL_CONFIRMED, "detail": "absent"}
        return {"outcome": "unverified", "detail": f"{exc.code}: {exc}"}
    if custody.is_terminal(detail):
        custody.settle_run(pathlib.Path(root), gateway, run, detail)
        return {"outcome": custody.CANCEL_CONFIRMED, "detail": "terminal"}
    return {"outcome": custody.CANCEL_REQUESTED, "detail": "stop requested earlier; run not terminal yet"}


def task_owned_runs_open(external: Dict[str, Any]) -> bool:
    """Whether an observation still holds work the owner's Pause must see stopped.

    Unreadable custody is open. A run the Pause spares (``review_owned``) is
    never counted; every other run counts until its stop is CONFIRMED —
    ``requested`` and ``unknown`` keep the member unsettled and its Resume refused.
    """
    if (external or {}).get("custody_read") != "ok":
        return True
    return any(isinstance(run, dict) and not run.get("review_owned")
               and str(run.get("state") or "") != EXTERNAL_STOP_CONFIRMED
               for run in (external or {}).get("runs") or [])
