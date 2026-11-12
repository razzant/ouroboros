"""The owner's Restart Claudexor action: what a restart ends, then the restart itself (D30).

The engine belongs to the installation, not to this server: every program that uses this
installation's engine has its runs on it, so the header Restart leaves it alone. This is
the one deliberate way to replace a sick engine process, and it says first what it ends.

``restart_preview`` is passive: it attaches to a serving engine (never starts one), reads
the engine's own list of live runs and attributes each by this installation's delegated-run
custody records. A run whose id custody holds, or which the engine started on behalf of
such a run, is Ouroboros's; every other live run belongs to another program. Nothing is
guessed from a path or a prompt, and a fact that could not be read stays unknown: an
engine that does not answer has an unknown number of runs, never zero.

``restart_engine`` acts only on the preview the owner saw (its ``token``); a changed set of
runs, engine or pin is refused with the fresh preview. It makes the attested owned-daemon
stop (``OwnedClaudexorDaemon.stop_outcome``, as Panic and the planned pin change make it)
and then the ordinary start (``ensure_owned_gateway``). The stop and the next start stay
separate facts in the receipt: an unconfirmed stop, or a stopped engine that did not start
again, is said, not folded into success. The runs this installation delegated end with
the engine as ``interrupted`` and settle through the existing custody readers; this module
reconciles nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import threading
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PIN_BELOW_SERVING = "engine_pin_below_serving"
PREVIEW_STALE = "restart_preview_stale"
RESTART_BUSY = "restart_in_progress"
OWNER_OUROBOROS, OWNER_OTHER = "ouroboros", "other"
RUNS_KNOWN, RUNS_UNATTRIBUTED, RUNS_UNKNOWN = "known", "unattributed", "unknown"
ENGINE_SERVING, ENGINE_NOT_ANSWERING, ENGINE_NOT_STARTED = "serving", "not_answering", "not_started"
ITEM_LIMIT = 50  # the rows a dialog can show; the counts always cover every run

_RESTART_LOCK = threading.Lock()


class RestartRefused(Exception):
    """A typed refusal that changed nothing; ``preview`` is the fresh one after a stale token."""

    def __init__(self, code: str, message: str, *, preview: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code, self.preview = code, preview


def _custody_of(run_ids: List[str]) -> Dict[str, Any]:
    """This installation's custody rows for exactly these engine run ids, in one read."""
    from ouroboros.config import DATA_DIR
    from ouroboros.delegate_custody_current import runs

    return runs(DATA_DIR, "run_id IN (SELECT value FROM json_each(?))", (json.dumps(sorted(run_ids)),))


def _own_open_run_ids() -> List[str]:
    """The delegated runs this installation's own records hold open (no engine asked)."""
    from ouroboros.config import DATA_DIR
    from ouroboros.delegate_custody import open_runs

    return sorted(str(custody.run_id) for custody in open_runs(DATA_DIR) if custody.run_id)


def _attributed(live: List[Dict[str, Any]], custody: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per live run, oldest first, each owned by Ouroboros or by another program.

    Ownership is custody's: the run's own id, or the id of the run the engine started it for
    (``delegatedFromRunId``), followed inside the live set until a custody row or none.
    """
    parent = {row["runId"]: str(row.get("delegatedFromRunId") or "") for row in live}

    def owner_row(run_id: str) -> Any:
        seen: set = set()
        while run_id and run_id not in seen:
            if run_id in custody:
                return custody[run_id]
            seen.add(run_id)
            run_id = parent.get(run_id, "")
        return None

    items = []
    for row in live:
        owned = owner_row(row["runId"])
        project = row.get("project") if isinstance(row.get("project"), dict) else {}
        items.append({
            "run_id": row["runId"], "state": row["state"],
            "owner": OWNER_OUROBOROS if owned is not None else OWNER_OTHER,
            "task_id": str(getattr(owned, "task_id", "") or ""),
            "kind": str(getattr(owned, "category", "") or ""),
            "harness": str(row.get("primaryHarness") or ""),
            "project": str(project.get("projectName") or project.get("root") or ""),
            "started_at": str(row.get("startedAt") or row.get("createdAt") or ""),
        })
    return sorted(items, key=lambda item: (item["started_at"], item["run_id"]))


def _next_start_pin() -> Any:
    """The pin this process's next engine start selects, or None when none is known to.

    An explicit engine binary or an attach-only selection starts no pinned release, and an
    unpublished or unreadable pin orders nothing.
    """
    from ouroboros.claudexor_runtime import get_runtime_manager
    from ouroboros.review_run_isolation import attach_home

    if attach_home() is not None or os.environ.get("OUROBOROS_CLAUDEXOR_BIN"):
        return None
    return get_runtime_manager().pin


def restart_preview() -> Dict[str, Any]:
    """What Restart Claudexor would end and start, read now; changes nothing, starts nothing."""
    from ouroboros.claudexor_daemon import owned_daemon_provisioned, read_owned_gateway
    from ouroboros.claudexor_exit_facts import runtime_facts
    from ouroboros.engine_pin_policy import engine_pin_transition
    from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC
    from ouroboros.gateways.claudexor_live_runs import read_live_runs

    state, serving, live, calls, error = ENGINE_NOT_STARTED, None, None, None, ""
    if owned_daemon_provisioned():
        state = ENGINE_NOT_ANSWERING
        try:
            with read_owned_gateway(timeout_sec=SHORT_POLL_TIMEOUT_SEC) as gateway:
                state, serving = ENGINE_SERVING, (gateway.engine_version, gateway.engine_build_sha)
                live = read_live_runs(gateway, timeout_sec=SHORT_POLL_TIMEOUT_SEC)
                try:  # optional: older engines have no status route, and a count is never invented
                    calls = (runtime_facts(gateway.daemon_status(timeout_sec=SHORT_POLL_TIMEOUT_SEC))["admission"]
                             or {})
                except Exception:
                    calls = None
        except Exception as exc:
            error = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
    runs: Dict[str, Any] = {"status": RUNS_KNOWN, "total": 0, OWNER_OUROBOROS: 0, OWNER_OTHER: 0,
                            "error": None, "items": [], "items_omitted": 0}
    fingerprint: Any = []
    if state == ENGINE_NOT_STARTED:
        pass  # no engine process exists under this home: nothing can end
    elif live is None:
        # No list: the engine did not answer, or answered without one. Its own records still say
        # what this installation has open; other programs' runs stay uncounted, never zero.
        try:
            fingerprint = _own_open_run_ids()
            own: Optional[int] = len(fingerprint)
        except Exception:
            log.warning("Restart Claudexor preview: own custody is unreadable", exc_info=True)
            fingerprint, own = None, None
        runs.update(status=RUNS_UNKNOWN, total=None, error=error, **{OWNER_OUROBOROS: own, OWNER_OTHER: None})
    else:
        try:
            items = _attributed(live, _custody_of([row["runId"] for row in live]) if live else {})
        except Exception as exc:
            log.warning("Restart Claudexor preview: own custody is unreadable", exc_info=True)
            fingerprint = sorted(row["runId"] for row in live)
            runs.update(status=RUNS_UNATTRIBUTED, total=len(live), error=f"{type(exc).__name__}: {exc}",
                        **{OWNER_OUROBOROS: None, OWNER_OTHER: None})
        else:
            fingerprint = sorted([item["run_id"], item["owner"]] for item in items)
            mine = sum(item["owner"] == OWNER_OUROBOROS for item in items)
            runs.update(total=len(items), items=items[:ITEM_LIMIT], items_omitted=max(0, len(items) - ITEM_LIMIT),
                        **{OWNER_OUROBOROS: mine, OWNER_OTHER: len(items) - mine})
    try:
        pin = _next_start_pin()
    except Exception:
        log.warning("Restart Claudexor preview: the engine pin is unreadable", exc_info=True)
        pin = None
    transition = engine_pin_transition(pin, serving)
    warnings = []
    if transition["direction"] == "backward":
        warnings.append({
            "code": PIN_BELOW_SERVING, "pin_version": pin.version, "serving_version": serving[0],
            "message": (f"This Ouroboros version pins Claudexor {pin.version}, older than the serving {serving[0]}. "
                        f"Engine data only moves forward: after the restart {pin.version} may refuse the data "
                        f"{serving[0]} wrote, and then Claudexor does not start."),
        })
    engine = {"state": state, "serving": transition["serving"], "pin": transition["target"],
              "direction": transition["direction"]}
    active, queued = ((calls.get(key) or {}).get("model") for key in ("active", "queued")) if calls else (None, None)
    seen = {"engine": engine, "runs": [runs["status"], fingerprint], "warnings": [w["code"] for w in warnings]}
    return {
        "engine": engine, "runs": runs, "warnings": warnings,
        "model_calls": {"status": RUNS_KNOWN if type(active) is int else RUNS_UNKNOWN,
                        "active": active if type(active) is int else None,
                        "queued": queued if type(queued) is int else None},
        # What the owner is shown and confirms: the engine, the pin, and exactly which runs end and
        # whose they are. Model calls come and go by the second and are not part of it.
        "token": hashlib.sha256(json.dumps(seen, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def _stop_engine() -> str:
    """The attested owned-daemon stop; a raising stop is disclosed as Panic and Restart disclose it."""
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE, get_owned_daemon
    from ouroboros.config import DATA_DIR
    from ouroboros.utils import append_jsonl, utc_now_iso

    try:
        return get_owned_daemon().stop_outcome()
    except Exception as exc:
        log.critical("Restart Claudexor: owned Claudexor stop raised %s; custody is unconfirmed", type(exc).__name__)
        try:
            append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "process_stop_unconfirmed",
                "purpose": CUSTODY_PURPOSE, "reason": f"stop raised {type(exc).__name__}",
            })
        except Exception:
            log.critical("Restart Claudexor: failed to record unconfirmed daemon stop")
        return "unconfirmed"


def restart_engine(token: str) -> Dict[str, Any]:
    """Stop the owned engine and start it again, for the preview ``token`` names.

    Returns the receipt ``{stop, serving, error, runs}``: the stop's own outcome, whether an
    engine answers afterwards (after an unconfirmed stop that may be the same process), the
    start's typed refusal and the counts the owner confirmed. Raises ``RestartRefused``
    (nothing changed) for a restart already in flight or a preview that no longer holds, and
    the typed ``ClaudexorUnavailable`` when this process may not restart an engine at all.
    """
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE, ensure_owned_gateway, get_owned_daemon
    from ouroboros.config import DATA_DIR
    from ouroboros.delegate_recovery import owner_restart_in_progress
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_run_isolation import attach_home
    from ouroboros.utils import append_jsonl, utc_now_iso

    if not _RESTART_LOCK.acquire(blocking=False):
        raise RestartRefused(RESTART_BUSY, "Claudexor is already restarting.")
    try:
        if attach_home() is not None:
            raise ClaudexorUnavailable("attach_only_engine", "an attached engine belongs to another installation")
        if owner_restart_in_progress(DATA_DIR):  # the start below would refuse: never stop without it
            raise ClaudexorUnavailable("owner_restart", "The owning generation is restarting.")
        preview = restart_preview()
        if not isinstance(token, str) or token != preview["token"]:
            raise RestartRefused(PREVIEW_STALE, "What a restart would end has changed since it was shown.",
                                 preview=preview)
        counts = {key: preview["runs"][key] for key in ("status", "total", OWNER_OUROBOROS, OWNER_OTHER)}
        receipt: Dict[str, Any] = {"stop": _stop_engine(), "serving": False, "error": None, "runs": counts}
        try:
            # The owner's explicit action releases a start-failure latch, as Refresh does.
            get_owned_daemon().clear_start_failure_latch(cleared_by="owner_restart_claudexor")
            ensure_owned_gateway().close()
            receipt["serving"] = True
        except ClaudexorUnavailable as exc:
            receipt["error"] = f"{exc.code}: {exc}"
        try:
            append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "claudexor_owner_restart", "purpose": CUSTODY_PURPOSE,
                "requested_by": "owner", **receipt, "engine": preview["engine"]})
        except Exception:
            log.warning("Restart Claudexor: the supervisor row was not written", exc_info=True)
        return receipt
    finally:
        _RESTART_LOCK.release()
