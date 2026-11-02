"""A task's delegated runs under one pause policy: observe, stop its own, or stop all.

The ONE observer body behind the loop-side pause, the supervisor-side Resume
grant, the owner Pause's settle tick and the writer census (owner Q8; owner
2026-10-07 fork 1 = A). A FRESH custody read, never a pause row's saved
summary; stops go through the verified cancel seam and their outcomes are
recorded per run. ``requested`` and ``unknown`` are not death: the run stays
under the task's custody and no second writer may start over it.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

from ouroboros.delegate_registration_policy import independent_review_without_apply, review_owned_source

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
            # ONE store read feeds both projections: a START_REQUESTED that becomes
            # STARTED between two reads must land in one of them (Astra 6fe5 #1).
            # The store answers or raises; a raise is UNKNOWN custody, never "no
            # open runs" (Astra run-a882315dbcd7 #2).
            from ouroboros.delegate_custody_current import task_snapshot

            view = task_snapshot(pathlib.Path(root), mine)
            runs = [run for run in view["runs"].values() if not getattr(run, "settled", True)]
            pending = view["pending"]
            # A START_REQUESTED whose response was lost has no run id yet but may
            # be a live remote writer: unknown custody, never absence (#3).
            # Every projected row also carries the custody facts a purpose-bound
            # reader may filter on: the registering ``source`` and whether the
            # row is an independent review without candidate-apply possibility
            # (``delegate_registration_policy``). The census itself never filters.
            pending_rows = [{"run_id": "", "invocation_id": str(row.get("invocation_id") or ""),
                             "route": str(row.get("route") or ""),
                             "cost_coverage": "unproven_preterminal", "stop_policy": "reconcile_first",
                             "review_owned": review_owned_source(row.get("source")),
                             "source": str(row.get("source") or ""),
                             "independent_review": independent_review_without_apply(row),
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
                "invocation_id": str(getattr(run, "invocation_id", "") or ""),
                "terminal_state": str(getattr(run, "terminal_state", "") or ""),
                "terminal_source": getattr(run, "terminal_source", {}) or {},
                "interaction_source": getattr(run, "interaction_source", {}) or {},
                "pending_interaction_ids": getattr(run, "pending_interaction_ids", []) or [],
                "output_complete": bool(getattr(run, "output_complete", False)),
                "patch_captured": bool(getattr(run, "patch_captured", False)),
                "ledger_recorded": bool(getattr(run, "ledger_recorded", False)),
                "route": str(getattr(run, "route", "") or getattr(run, "route_id", "") or ""),
                "cost_coverage": "unproven_preterminal", "stop_policy": "observe_only",
                "review_owned": review_owned_source(getattr(run, "source", "")),
                "source": str(getattr(run, "source", "") or ""),
                "independent_review": independent_review_without_apply(run),
                "state": EXTERNAL_RUNNING, "stop_outcome": "", "detail": "", **fields}

    if stop_policy == STOP_POLICY_OBSERVE:
        observed = {"runs": [_row(run) for run in runs] + pending_rows, "observed_at": time.time(),
                    "custody_read": "ok", "coverage_basis": "observed_without_stop_request"}
        try:
            from ouroboros.sleep_custody import observe_return
            body = observe_return(root, str(task_id), observed)
            if body:
                observed["body_custody"] = body
        except Exception:
            log.debug("Cold-return body observation unavailable", exc_info=True)
        return observed
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
    the owning daemon store answering it has no such run confirms the stop; anything else
    keeps the earlier ``requested`` fact, and an unreachable read is unknown.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        detail = gateway.get_run(run.run_id)
    except ClaudexorUnavailable as exc:
        if custody.daemon_says_absent(exc, gateway=gateway, run=run):
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


# --- one owner Pause's participants (#1562) ----------------------------------------------
# The census above is liveness: it decides Resume, and a settled run leaves it for good. The
# owner's Pause also keeps, on its own pause row or warm wait and under that record's existing
# compare-and-set, every run or unbound start its observations named, with the stop it chose,
# so its Resume can say what became of each. A new Pause starts its own list. A list begun on a
# record its Pause had already observed without one (an older body, a dead worker's stops) or
# grown from an unreadable custody read is ``episode_runs_partial``: its gap stays stated.
_PARTICIPANT_KEYS = ("run_id", "invocation_id", "route", "review_owned")
_DETAILED = 8  # participants described in full; the rest are named with their reader
PARTIAL_KEY = "episode_runs_partial"
PARTIAL_LINE = ("- this Pause's participant list is partial: a run it stopped that settled before the list "
                "began is not named here")


def episode_runs(prior: Any, observation: Any) -> List[Dict[str, Any]]:
    """The participants after one more observation (or participant list) of the same Pause.

    Never fewer: a start seen unbound and later bound is one row (its invocation id), and
    a stop this Pause chose keeps its latest outcome, which an observe-only read never erases.
    """
    rows = [dict(row) for row in (prior if isinstance(prior, list) else []) if isinstance(row, dict)]
    for run in ((observation or {}).get("runs") or []) if isinstance(observation, dict) else []:
        run_id = str(run.get("run_id") or "") if isinstance(run, dict) else ""
        invocation = str(run.get("invocation_id") or "") if isinstance(run, dict) else ""
        if not (run_id or invocation):
            continue
        row = next((row for row in rows if (run_id and row.get("run_id") == run_id)
                    or (invocation and row.get("invocation_id") == invocation)), None)
        if row is None:
            rows.append(row := {})
        row.update({key: run[key] for key in _PARTICIPANT_KEYS if run.get(key) not in (None, "")})
        if run.get("stop_policy") == "request_stop" or run.get("stop_requested"):
            row.update(stop_requested=True, stop_outcome=str(run.get("stop_outcome") or row.get("stop_outcome") or ""))
    return rows


def episode_fields(record: Any, observation: Any, *, starts: bool = False) -> Dict[str, Any]:
    """The participant fields a writer stores on ``record`` (a pause row or warm ``owner_pause``)
    after one more observation. ``starts``: this observation begins the Pause's list (its park, or
    its first look at a record it never observed); a record without a list is otherwise one the
    Pause observed before it was kept, so the result is partial, never whole."""
    record = record if isinstance(record, dict) else {}
    prior = record.get("episode_runs")
    partial = (record.get(PARTIAL_KEY) is True or (not isinstance(prior, list) and not starts)
               or (isinstance(observation, dict) and observation.get("custody_read") not in (None, "ok")))
    return {"episode_runs": episode_runs(prior, observation), **({PARTIAL_KEY: True} if partial else {})}


def owner_pause_runs(ctx: Any, record: Any, observation: Any = None) -> str:
    """The participant lines of the owner Pause ``record`` belongs to: its own row (reason
    ``owner``) or a record the Pause kept its list on (a sleep or technical pause whose runs it
    stopped). ``""`` for any other pause, whose own disclosure stands unchanged."""
    from ouroboros.owner_pause import REASON_OWNER

    record = record if isinstance(record, dict) else {}
    if record.get("reason") != REASON_OWNER and not isinstance(record.get("episode_runs"), list):
        return ""
    return paused_runs_notice(ctx, record, observation)


def paused_runs_notice(ctx: Any, record: Any, observation: Any = None) -> str:
    """A Resume's lines for one owner Pause record (a cold row or a warm ``owner_pause``).

    A record without a whole list (written before lists were kept, or ``episode_runs_partial``)
    says so: runs that settled before its list began are named nowhere here, a gap, not their absence."""
    recorded = isinstance(record, dict) and isinstance(record.get("episode_runs"), list)
    participants = episode_runs(record.get("episode_runs") if recorded else None, observation)
    gap = "" if recorded and record.get(PARTIAL_KEY) is not True else PARTIAL_LINE + "\n"
    try:
        from ouroboros.delegate_custody import custody_root

        root = custody_root(ctx)
    except Exception as exc:
        return (f"{gap}- custody root unresolvable ({type(exc).__name__}): what became of "
                + (", ".join(_name(row) for row in participants) or "this Pause's runs") + " is UNKNOWN")
    return gap + episode_disclosure(root, str(getattr(ctx, "task_id", "") or ""), participants)


def episode_disclosure(root: Any, task_id: str, participants: Any) -> str:
    """What became of one Pause's participants, from current custody (settled included) and
    each run's locally verified retained terminal source. No engine call or paid round; nothing
    here continues, replaces or grants anything: engine facts are evidence, not authorization."""
    rows = [row for row in participants if isinstance(row, dict)] if isinstance(participants, list) else []
    if not rows:
        return "- none named"
    try:
        from ouroboros.delegate_custody_current import task_snapshot

        view = task_snapshot(pathlib.Path(root), str(task_id))
    except Exception as exc:
        return (f"- custody unreadable now ({type(exc).__name__}): what became of "
                + ", ".join(_name(row) for row in rows) + ' is UNKNOWN; delegate_wait(run_id="...") reads each run')
    pending = {str(item.get("invocation_id") or "") for item in view["pending"] if isinstance(item, dict)}
    lines = [_participant_line(root, task_id, row, view["runs"], pending) for row in rows[:_DETAILED]]
    if len(rows) > _DETAILED:
        lines.append(f"- {len(rows) - _DETAILED} more, not detailed here: "
                     + ", ".join(_name(row) for row in rows[_DETAILED:]) + '; delegate_wait(run_id="...") reads each run')
    return "\n".join(lines)


def _name(row: Dict[str, Any]) -> str:
    return f"run {row['run_id']}" if row.get("run_id") else f"start {row.get('invocation_id') or '?'}"


def _participant_line(root: Any, task_id: str, row: Dict[str, Any], runs: Dict[str, Any], pending: set) -> str:
    run_id, invocation = str(row.get("run_id") or ""), str(row.get("invocation_id") or "")
    run = runs.get(run_id) if run_id else next(
        (entry for entry in runs.values() if invocation and entry.invocation_id == invocation), None)
    review = bool(row.get("review_owned") or getattr(run, "review_owned", False))
    route = str(row.get("route") or getattr(run, "route_id", "") or "")
    facts = [f"- {_name({'run_id': getattr(run, 'run_id', '') or run_id, 'invocation_id': invocation})}"
             + (f" ({route})" if route else "") + ": " + (
                 "a review panel's run: its panel owns it and it finishes separately; this Pause did not stop it"
                 if review else f"this Pause requested its stop (stop outcome: {row.get('stop_outcome') or 'unrecorded'})"
                 if row.get("stop_requested") else "this Pause did not request its stop")]
    if run is None:
        facts.append(f"no run is bound to start {invocation} yet: its start outcome is UNKNOWN; "
                     f'delegate_start(retry_of="{invocation}") retrieves it without starting another'
                     if invocation and invocation in pending else "custody holds no row for it now: its fate is UNKNOWN")
        return "; ".join(facts)
    if not run.terminal_state:
        facts.append("custody closed it without a recorded engine terminal: what became of its work is UNKNOWN"
                     if run.settled else
                     "no engine terminal observed yet: it may still be running; never start a second writer over it")
    else:
        facts.append(f"engine terminal: {run.terminal_state}" + (f" ({run.terminal_reason})" if run.terminal_reason else ""))
        if row.get("stop_requested") and run.terminal_state == "succeeded":
            facts.append("it finished on its own; the stop did not end it")
        facts += _retained_terminal(root, task_id, run)
        facts.append(f"output retained ({'complete' if run.output_complete else 'partial'}, "
                     f"{'read' if run.output_consumed else 'not read yet'})" if run.output_artifact
                     else "no retained output recorded")
        if run.patch_captured or run.patch_disposed in ("applied", "rejected"):  # a lost result was never captured
            facts.append(("patch captured, " if run.patch_captured else "no patch captured, ")
                         + (str(run.patch_disposed) or "awaiting apply or reject") + _decided_by(run, task_id))
        if run.superseded_by:
            facts.append(f"already continued by run {run.superseded_by}")
        elif run.patch_apply_pending:
            facts.append("an apply of its patch is ambiguous: inspect it before anything else")
        elif not review:
            facts.append(f'it may be continued: delegate_start(subagent_id=..., continue_from="{run.run_id}") '
                         "(a configured session omits subagent_id)")
    if not review:
        facts.append(f'inspect: delegate_wait(run_id="{run.run_id}")')
    return "; ".join(facts)


def _decided_by(run: Any, task_id: str) -> str:
    """Who recorded an apply or reject (#1651), from current custody; a row reduced before the
    disposer was projected says so instead of naming anyone."""
    if run.patch_disposed not in ("applied", "rejected"):
        return ""
    by = run.patch_disposed_by
    return ((" as lost (its private copy was gone with no usable capture)" if run.patch_result_lost else "")
            + (" by this task" if by and by == task_id else f" by task {by}" if by else " (disposer not recorded)"))


def _retained_terminal(root: Any, task_id: str, run: Any) -> List[str]:
    """The engine's own continuation facts from the run's retained terminal source, verified locally."""
    ref = run.terminal_source if isinstance(run.terminal_source, dict) else {}
    if not ref:
        return ["its terminal source was not retained: whether its session can be reused is unknown"]
    try:
        from ouroboros.artifacts import read_actor_source_bytes
        from ouroboros.delegate_continuation import terminal_continuity
        from ouroboros.delegate_custody_usage import summary_of

        detail = json.loads(read_actor_source_bytes(root, str(ref.get("task_id") or task_id), ref))
        facts = terminal_continuity(run.run_id, summary_of(detail if isinstance(detail, dict) else {}))
    except Exception as exc:
        return [f"its retained terminal source failed local verification ({type(exc).__name__}): "
                f'read_file(root="artifact_store", path="{ref.get("path") or "?"}") shows what remains']
    lines = [("engine resumable facts: " + json.dumps(facts["resumable"], ensure_ascii=False, sort_keys=True))
             if facts.get("resumable") else "the engine reported no resumable block: whether its session can be reused is unknown"]
    continuity = list(facts.get("continuity") or [])
    if len(continuity) > 3:
        lines.append(f"{len(continuity) - 3} earlier continued tries: delegate_wait shows every try")
    return lines + continuity[-3:]
