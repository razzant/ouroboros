"""The retained observation of the existing managed-update executor.

Presentation state, never update or recovery authority: nothing here decides,
retries, schedules or recovers an update. Only the synchronous executor that
acquired the update lock advances its own operation; cancelling the HTTP await
does not finish work still running in ``asyncio.to_thread``. The LAST admitted
attempt outlives the process as one ``state.json`` value (``STATE_KEY``), bound
to its initiating generation and, before any transaction is written, to that
transaction's existing ``attempt_id``. A later generation concludes it only
from a boot branch's explicit meaning (``observed``) for the transaction that
branch processed, while serving the commit that meaning implies; missing,
unbound or unreadable evidence is a named ``unknown``, never success. A request
refused before admission stays its own HTTP answer.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import threading
import uuid

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)
_lock = threading.Lock()
_current: dict = {}
_boot_settled = threading.Event()
STATE_KEY = "managed_update_attempt"
# Each is advanced immediately before worker/service teardown, and ``applying``
# immediately before each lane's first repository step (pinned by source tests).
_TEARDOWN_STAGES = frozenset({"stopping_workers", "stopping_services"})
_MUTATION_STAGE = "applying"
_BOOT_OUTCOMES = frozenset({"applied", "not_applied", "rolled_back"})
_FIELDS = ("operation_id", "generation", "stage", "started_at", "stage_started_at", "active", "result",
           "error", "restart_required", "outcome", "outcome_at", "concluded_by", "reason", "blockers",
           "work_stop", "base_sha", "target_sha", "tx_attempt_id", "keep_branch", "stash_note", "generation_facts")


def _generation() -> str:
    from ouroboros.process_custody import current_custody_session_id

    return current_custody_session_id() or f"pid:{os.getpid()}"


def _shape(record: dict) -> dict:
    """The stored and public fields, whole: one record already bounds the history."""
    return {key: record[key] for key in _FIELDS if key in record}


def _store(mutate) -> bool:
    """Serialized cross-process read/check/write under the state lock; ``mutate(stored)`` returns
    the replacement or ``None`` to keep it. True means an accepted replacement was saved."""
    from supervisor.state import update_state

    accepted = False

    def apply(state: dict) -> None:
        nonlocal accepted
        value = state.get(STATE_KEY)
        replacement = mutate(dict(value) if isinstance(value, dict) else {})
        if replacement is not None:
            state[STATE_KEY] = replacement
            accepted = True

    update_state(apply)
    return accepted


def _persist(record: dict, *, replace: bool = False) -> None:
    """Recover a failed first save only over its unchanged admission predecessor."""
    baseline = record.get("_retention_base")  # None is unknown; {} is known absence.

    def mutate(stored: dict):
        nonlocal baseline
        if replace:
            # Only begin replaces the record, under admission's update lock. Capture the
            # predecessor even if persistence then fails, including after the primary write.
            baseline = copy.deepcopy(stored)
        same = stored.get("operation_id") == record.get("operation_id") and stored.get("concluded_by") != "boot"
        unchanged = baseline is not None and stored == baseline
        return _shape(record) if replace or same or unchanged else None

    try:
        failed = not _store(mutate)
    except Exception:
        failed = True
        log.warning("Could not retain the update attempt", exc_info=True)
    with _lock:
        if _current.get("operation_id") == record.get("operation_id"):
            _current["retention_unconfirmed"] = failed
            if not failed:
                _current.pop("_retention_base", None)
            elif baseline is not None:
                _current["_retention_base"] = baseline


def _read_saved() -> tuple[str, dict]:
    """``(status, record)`` through the typed state read: absent, valid, corrupt or unreadable."""
    from supervisor.state import read_state

    observed = read_state()
    if observed.quality == "uninitialized":
        return "absent", {}
    if observed.quality not in ("current", "recovered"):
        return "unreadable", {}
    value = observed.values.get(STATE_KEY)
    if value is None:
        return "absent", {}
    if isinstance(value, dict) and value.get("operation_id"):
        return "valid", dict(value)
    return "corrupt", {}


def _unknown(reason: str) -> dict:
    """A record that exists or may exist but cannot be read: named, never absent."""
    return {"operation_id": reason, "active": False, "outcome": "unknown", "reason": reason,
            "current_generation": False}


def _bound(record: dict, tx: dict | None) -> bool:
    """Whether the transaction this attempt was bound to is still active."""
    attempt = str(record.get("tx_attempt_id") or "")
    if not attempt:
        return False
    if tx is None:
        try:
            from supervisor.update_merge import active_update_tx

            tx = active_update_tx()
        except Exception:
            return True  # unreadable custody keeps the attempt waiting, never concluded
    return str((tx or {}).get("attempt_id") or "") == attempt


def _public(record: dict, tx: dict | None) -> dict:
    view = _shape(record)
    if record.get("retention_unconfirmed"):
        view["retention_unconfirmed"] = True  # live fact; an unsuccessful write cannot preserve it
    if not view.get("operation_id"):
        return {}
    here = view.get("generation") == _generation()
    view["current_generation"] = here
    if not here:
        view["active"] = False  # an older server's executor is gone, whatever it last wrote
    if view.get("concluded_by") != "executor":
        view.pop("work_stop", None)  # only the executor's own conclusion carries its teardown fact
    if view.get("outcome") or view.get("active"):
        return view
    # Something can still conclude it: this server's restart, the boot finalizer, or the bound transaction.
    waiting = (view.get("result") in ("restart_requested", "restart_required") if here
               else not _boot_settled.is_set()) or _bound(view, tx)
    if not waiting:
        view.update(outcome="unknown", reason=view.get("reason") or "no_completion_record")
    return view


def snapshot(tx: dict | None = None) -> dict:
    """This server's live attempt, else the retained one; ``tx`` is the caller's active-transaction read."""
    with _lock:
        live = {key: value for key, value in _current.items() if key != "owner_thread"}
    try:
        status, saved = ("valid", live) if live else _read_saved()
    except Exception:
        log.debug("Could not read the retained update attempt", exc_info=True)
        status, saved = "unreadable", {}
    if status in ("unreadable", "corrupt"):
        return _unknown("state_unreadable" if status == "unreadable" else "record_unreadable")
    try:
        return _public(saved, tx)
    except Exception:
        log.debug("Could not project the update attempt", exc_info=True)
        return _unknown("record_unreadable")


def _notify() -> None:
    # A progress notice is NOT update_status_ready: that event proves the boot
    # finalizer returned, and the UI uses it to settle a reconnect episode.
    from ouroboros.gateway.ws import broadcast_ws_sync

    broadcast_ws_sync({"type": "update_progress_changed"})


def notify_changed() -> None:
    """Ask every client to read the status again; observes nothing itself."""
    try:
        _notify()
    except Exception:
        log.debug("Could not publish update status invalidation", exc_info=True)


def begin(*, base_sha: str = "", target_sha: str = "") -> None:
    """Called only after the executor acquired the existing update lock: the next admitted attempt."""
    global _current
    try:
        now = utc_now_iso()
        record = {
            "operation_id": uuid.uuid4().hex, "generation": _generation(),
            "stage": "preparing", "started_at": now, "stage_started_at": now,
            "active": True, "result": "", "error": "", "restart_required": False,
            "outcome": "", "work_stop": "none", "base_sha": str(base_sha or ""), "target_sha": str(target_sha or ""),
        }
        from ouroboros.update_generation_facts import sample
        record["generation_facts"] = {"before": sample(), "after": None}
        # A failure before the state mutator runs otherwise leaves no comparison
        # baseline. This read is only a candidate: later recovery compares it under
        # the state lock, including all conclusion fields, before replacing anything.
        try:
            status, saved = _read_saved()
            if status in ("absent", "valid"):
                record["_retention_base"] = copy.deepcopy(saved)
        except Exception:
            log.debug("Could not read the update retention baseline", exc_info=True)
        with _lock:
            _current = {**record, "owner_thread": threading.get_ident()}
        _persist(record, replace=True)
        _notify()
    except Exception:
        log.debug("Could not publish update progress", exc_info=True)


def advance(stage: str) -> None:
    try:
        with _lock:
            if (not _current.get("active")
                    or _current.get("owner_thread") != threading.get_ident()
                    or _current.get("stage") == stage):
                return
            _current.update(stage=stage, stage_started_at=utc_now_iso())
            if stage in _TEARDOWN_STAGES:
                _current["work_stop"] = "invoked"
            if stage == _MUTATION_STAGE:
                _current["mutating"] = True  # in memory only: read by this executor's own _finish
        _notify()
    except Exception:
        log.debug("Could not publish update stage", exc_info=True)


def link(tx: dict, *, keep_branch: str = "") -> None:
    """Bind this executor's attempt to its transaction BEFORE that transaction is written."""
    try:
        with _lock:
            if not _current.get("active") or _current.get("owner_thread") != threading.get_ident():
                return
            _current["tx_attempt_id"] = str(tx.get("attempt_id") or "")
            if keep_branch:
                _current["keep_branch"] = str(keep_branch)
            record = {key: value for key, value in _current.items() if key != "owner_thread"}
        _persist(record)
    except Exception:
        log.debug("Could not bind the update attempt", exc_info=True)


def _executor_outcome(body: dict, code: int, error: Exception | None, mutating: bool) -> str:
    """What this executor proved; "" while a restart or the transaction owner must conclude.

    An attempt that never announced ``applying`` reached no repository step of the update,
    whatever its status code."""
    if error is None and code < 400:
        return ""
    if not mutating:
        return "not_applied"
    if code == 409 and body.get("outcome") == "not_applied":
        return "not_applied"  # the physical first step refused; recovery verified no tree effect
    if code == 409 and body.get("rolled_back") is True:
        return "rolled_back"  # rollback_managed_update verified the restored checkout
    if code == 409 and body.get("abandoned") is True:
        return "not_applied"  # Git refused the first tree step; HEAD verified at its base
    return "failed"


def _finish(response, error: Exception | None) -> None:
    try:
        body = json.loads(response.body) if response is not None else {}
        code = getattr(response, "status_code", 500)
        failure = str(body.get("error") or error or "")
        result = ("failed" if error or code >= 400
                  else "restart_requested" if body.get("restarting")
                  else str(body.get("status") or "completed"))
        with _lock:
            if not _current.get("active") or _current.get("owner_thread") != threading.get_ident():
                return
            outcome = _executor_outcome(body, code, error, bool(_current.get("mutating")))
            _current.update(
                active=False, result=result, error=failure, outcome=outcome,
                restart_required=bool(body.get("restart_required") or result == "restart_required"),
                reason=str(body.get("reason") or ""), blockers=[str(item) for item in body.get("blockers") or []],
                stash_note=str(body.get("stash_note") or ""),
                **({"keep_branch": str(body["keep_branch"])} if body.get("keep_branch") else {}),
                **({"outcome_at": utc_now_iso(), "concluded_by": "executor"} if outcome else {}),
            )
            record = {key: value for key, value in _current.items() if key != "owner_thread"}
        _persist(record)
        _notify()
    except Exception:
        log.debug("Could not finish update progress", exc_info=True)


def _conclude(seen: dict, result: dict, bootstrap_ok: bool) -> None:
    from ouroboros.server_process import server_source_baseline

    outcome, reason = str(seen["outcome"]), ""
    if outcome == "applied" and not bootstrap_ok:
        outcome, reason = "unknown", "startup_unconfirmed"
    elif not (seen.get("commit") and server_source_baseline() == seen["commit"]):
        outcome, reason = "unknown", "running_source_mismatch"
    change = {"active": False, "outcome": outcome, "outcome_at": utc_now_iso(), "concluded_by": "boot",
              "restart_required": False, **({"reason": reason} if reason else {}),
              **({"stash_note": str(result["stash_note"])} if result.get("stash_note") else {})}
    from ouroboros.update_generation_facts import sample
    after = sample() if outcome == "applied" else None

    def mutate(stored: dict):
        if stored.get("tx_attempt_id") != seen["attempt_id"] or stored.get("generation") == _generation():
            return None  # another attempt, or this very server: no restart happened in between
        return _shape({**stored, **change, "generation_facts": {
            **(stored.get("generation_facts") or {"before": None}), "after": after}})

    _store(mutate)


def observe_boot(finalize, *, ready: bool, bootstrap_ok: bool) -> dict:
    """Run the existing boot finalizer unchanged, then conclude the attempt bound to the transaction
    the returning branch processed, from the meaning it named. Grants the finalizer nothing."""
    result = finalize(supervisor_ready=ready)
    try:
        seen = (result or {}).get("observed")
        if isinstance(seen, dict) and seen.get("attempt_id") and seen.get("outcome") in _BOOT_OUTCOMES:
            _conclude(seen, result, bootstrap_ok)
    except Exception:
        log.debug("Could not observe the boot update outcome", exc_info=True)
    finally:
        _boot_settled.set()
        notify_changed()
    return result


def observe_result(execute):
    """Observe the actual executor's return, including a disconnected client."""
    @functools.wraps(execute)
    def observed(*args, **kwargs):
        response, error = None, None
        try:
            response = execute(*args, **kwargs)
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            _finish(response, error)

    return observed
