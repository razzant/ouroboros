"""Durable per-extension health vector for live->broken regression detection.

A small immune-system instrument (BIBLE P1 material state discrepancies;
P3 health invariants): if an extension that was live at a
prior code version stops loading after a self-modification + restart, record the
regression and surface it to the owner/agent. This is warning-only — it never
disables a skill and never mutates git. The record lives next to the other per-skill
owner-state files at ``data/state/skills/<name>/health.json``.

The same record keeps each companion's open terminal failure by name
(``companion_failures``): an unexpected exit without a successful replacement, or a failed settings
replacement after stopping a running companion. A later start closes it as of its own time, whether that
start, a reconcile or the start's own later failure writes the closing; every server
observation re-applies them under the record's lock, so a reconcile's earlier read
neither restores a closed failure nor hides a newer one, and keeps only those of
companions the loaded publication registers. A listing and a reconcile's answer show the record
through one projection (``_show_health``). ``logs/events.jsonl`` keeps unexpected exits,
failed settings replacements and recoveries.
The record also keeps each companion's last closed failure
(``companion_recovered``), so a closing whose event could not be appended stays
readable until that companion's next one.
It also keeps the permissions the skill itself reported unavailable
(``permission_refusals``: permission id -> its open refusal's start, code and repair hint):
a refusal opens when a reported failure says so, keeps its start on repeats, and closes
when a later success of the skill reports that permission available. It never changes
the health projection.
"""

from __future__ import annotations

import functools
import logging
import pathlib
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.skill_loader import skill_state_dir, skill_state_dir_path
from ouroboros.utils import append_jsonl, json_snapshot, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

HEALTH_FILENAME = "health.json"
_SCHEMA = 1

# Health statuses derived from the extension runtime state.
LIVE = "live"        # desired_live and loaded successfully
BROKEN = "broken"    # desired_live but failed to load
INACTIVE = "inactive"  # disabled, deps pending, review-gated, or not an extension
UNKNOWN = "unknown"    # no authoritative server observation has been recorded yet
HEALTH_UNAVAILABLE = "health_unavailable"
COMPANION_REFRESH_PENDING = "companion_refresh_pending"
COMPANION_RESTART_EXHAUSTED = "companion_restart_exhausted"
COMPANION_FAILED = "companion_failed"  # one or more open failures are not exhaustion
# companion name -> ts, load_error, optional returncode, failure-class flags, started_at (the start ended or replaced)
_FAILURES = "companion_failures"
_RECOVERED = "companion_recovered"  # name -> last closed failure: returncode?, load_error?, failed_at, closed_at, by
_REFUSALS = "permission_refusals"  # permission id -> open refusal the skill reported: since, code, repair?, where?
# A failure's write is retried after these waits (only its own monitor thread waits, never a lock holder).
_EXHAUSTED_WRITE_RETRY_SEC = (0.5, 2.0)
_RetainedCompanions = Dict[str, str] | Callable[[], Dict[str, str]]


def health_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    return skill_state_dir(pathlib.Path(drive_root), skill_name) / HEALTH_FILENAME


def read_extension_health(drive_root: pathlib.Path, skill_name: str) -> Optional[Dict[str, Any]]:
    """Passive health snapshot. A missing file is None; unreadable evidence has unknown status."""
    try:
        return json_snapshot(skill_state_dir_path(pathlib.Path(drive_root), skill_name) / HEALTH_FILENAME, strict=True)
    except (OSError, ValueError):
        return {"status": UNKNOWN, "reason": HEALTH_UNAVAILABLE}


def _open_companion_failures(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read open failures by name. An old joint diagnostic belongs to its original observation.
    Earlier conversions copied it onto every name. Replace only those exact copies, preserving later failures."""
    stored = record.get(_FAILURES)
    converted = isinstance(stored, dict)
    failures = {str(name): dict(fact) for name, fact in stored.items() if isinstance(fact, dict)} if converted else {}
    observed = record.get("legacy_companion_failure" if converted else "last_observed")
    if not isinstance(observed, dict) or observed.get("status") != BROKEN:
        return failures
    reason, prefix = str(observed.get("reason") or ""), f"{COMPANION_RESTART_EXHAUSTED}:"
    names = [name for name in reason[len(prefix):].split(",") if name] if reason.startswith(prefix) else []
    original = {"ts": str(observed.get("ts") or ""), "load_error": str(observed.get("load_error") or "")}
    for name in names:
        if converted and failures.get(name) != original:
            continue
        if len(names) == 1:
            failures[name] = dict(original)
            if type(observed.get("returncode")) is int:
                failures[name]["returncode"] = observed["returncode"]
        else:
            failures[name] = {**original, "load_error": f"companion {name!r} exhausted its restart budget"}
    return failures


def recorded_failed_companions(drive_root: pathlib.Path, skill_name: str) -> set[str]:
    """Unclosed records, even when a later start already makes the passive card look healthy."""
    return set(_open_companion_failures(read_extension_health(drive_root, skill_name) or {}))


def _companion_overlay(failures: Dict[str, Dict[str, Any]]) -> tuple[str, str]:
    """The reason and words open failures show: exhaustion is named only when every one of them is one."""
    names = sorted(failures)
    failed = any(fact.get("restart_failed") or fact.get("exited") or fact.get("refresh_failed")
                 for fact in failures.values())
    return (f"{COMPANION_FAILED if failed else COMPANION_RESTART_EXHAUSTED}:{','.join(names)}", "; ".join(dict.fromkeys(
        str(failures[name].get("load_error") or f"companion {name!r} exhausted its restart budget")
        for name in names)))


def _status_with_failures(status: str, failed: Any) -> str:
    """An open companion failure shows any status not BROKEN of its own as BROKEN."""
    return BROKEN if failed and status != BROKEN else status


def _card_observation(observed: Dict[str, Any]) -> tuple:
    """A recovered companion and the next ready reconcile show the same state on the card."""
    reason = observed.get("reason") or ""
    return (
        observed.get("status"), "ready" if reason == "companion_restarted" else reason,
        observed.get("load_error") or "", observed.get("retained_companions") or {},
    )


def record_extension_health(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    status: str,
    version: str = "",
    sha: str = "",
    reason: str = "",
    load_error: str = "",
    process: str = "server",
    server_reconcile: str = "",
    companions: Optional[List[str]] = None,
    retained_companions: Optional[_RetainedCompanions] = None,
    reconcile_succeeded: bool = True,
) -> Dict[str, Any]:
    """Persist one process-qualified observation and the server health projection.

    ``regressed`` (persisted) stays true while a once-live extension is broken, so
    the UI and health invariants keep surfacing it until it loads again.
    ``newly_regressed`` (returned, not persisted) marks the live->broken transition
    so callers can log/alert once per transition rather than every restart.
    Worker observations are qualifiers only: they never replace server authority or
    advance the server's ``last_known_good``; one equal to the recorded worker
    observation but for its time is not written again (every worker birth repeats it),
    so a worker observation's ``ts`` names when workers first saw it so.
    A server observation not BROKEN of its own carries the open companion failures
    as BROKEN, never as a code regression; ``companions``, the names the loaded
    publication registers (none once unloaded), keeps only theirs.
    """
    return _record(drive_root, skill_name, status=status, version=version, sha=sha, reason=reason,
                   load_error=load_error, process=process, server_reconcile=server_reconcile,
                   companions=companions, retained_companions=retained_companions,
                   reconcile_succeeded=reconcile_succeeded)[0]


def _record(drive_root, skill_name, *, status, version=None, sha=None, reason="", load_error="",
            process="server", server_reconcile="", companions=None, retained_companions=None,
            reconcile_succeeded=False, change=None) -> tuple[Dict[str, Any], str]:
    """``record_extension_health``, also answering ``written``, ``unchanged`` (nothing to write, or ``change``
    declined) or ``failed`` (the write did not land). ``change(failures, recovered, ts)`` edits the open and the
    last closed companion failures under the record's lock (False writes nothing); ``version``/``sha`` None keep
    the server observation's. A server's retained-process reader is sampled at each serialized update."""
    process = "worker" if process == "worker" else "server"
    result: Dict[str, Any] = {}
    written = projection_changed = False

    def _update(prior: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal written, projection_changed
        now = utc_now_iso()
        failures = _open_companion_failures(prior)
        prior_failures = failures.copy()
        legacy_failure = prior.get("legacy_companion_failure")
        if failures and not isinstance(prior.get(_FAILURES), dict):
            # Conversion may coincide with a disable, removal or another companion's start.
            # Keep the one original observation, including its own code stamp and diagnostic.
            legacy_failure = prior["last_observed"]
        recovered = prior.get(_RECOVERED)
        recovered = dict(recovered) if isinstance(recovered, dict) else {}
        if change is not None and not change(failures, recovered, now):
            result.update(prior, newly_regressed=False)
            return None
        if companions is not None and process == "server":
            failures = {name: fact for name, fact in failures.items() if name in companions}
        observations = prior.get("observations")
        observations = dict(observations) if isinstance(observations, dict) else {}
        legacy = prior.get("last_observed")
        if "server" not in observations and isinstance(legacy, dict) and legacy:
            observations["server"] = dict(legacy, process="server")
        prior_server = observations.get("server")
        prior_server = prior_server if isinstance(prior_server, dict) else {}
        shown = _status_with_failures(status, process == "server" and failures)
        overlay = shown != status
        observed_reason, observed_error = _companion_overlay(failures) if overlay else (reason, load_error)
        retained = None
        if process == "server":
            retained = retained_companions() if callable(retained_companions) else retained_companions
        if retained and observed_reason in ("ready", "companion_restarted"):
            observed_reason = "retained_unresolved"
        observed = {
            "version": str(prior_server.get("version") or "") if version is None else version,
            "sha": str(prior_server.get("sha") or "") if sha is None else sha,
            "status": shown,
            "reason": observed_reason,
            "load_error": (observed_error or "")[:2000],
            "ts": now,
            "process": process,
            **({"retained_companions": retained} if retained else {}),
        }
        if process == "server" and shown == BROKEN:  # an open failure keeps its start; an older record's is unknown
            observed["broken_since"] = prior_server.get("broken_since") if prior_server.get("status") == BROKEN else now
        if process == "worker":
            observed["server_reconcile"] = str(server_reconcile or "")
        prior_worker = observations.get("worker")
        if process == "worker" and "observations" in prior and isinstance(prior_worker, dict) \
                and {**prior_worker, "ts": now} == observed:
            result.update(prior, newly_regressed=False)
            return None  # the same worker observation again: nothing new to record
        observations[process] = observed
        last_known_good = prior.get("last_known_good")
        if not isinstance(last_known_good, dict):
            last_known_good = None
        regressed = bool(prior.get("regressed"))
        newly_regressed = False
        if process == "server":
            if observed["status"] == LIVE:
                if reconcile_succeeded and not retained and _companions_allow_known_good(skill_name, companions):
                    last_known_good = {"version": observed["version"], "sha": observed["sha"], "ts": now}
                regressed = False
            elif observed["status"] == BROKEN and not overlay and last_known_good is not None:
                # A same-sha break is environmental, not a code regression.
                regressed = str(last_known_good.get("sha") or "") != str(observed["sha"] or "")
                newly_regressed = regressed and str(prior_server.get("status") or "") == LIVE
            else:
                regressed = False
            authoritative = observed
        else:
            authoritative = prior_server
        record = with_schema_version({
            "skill": skill_name,
            "status": str(authoritative.get("status") or UNKNOWN),
            "regressed": regressed,
            "last_known_good": last_known_good,
            "last_observed": authoritative,
            "observations": observations,
            _FAILURES: failures,
            **({"legacy_companion_failure": legacy_failure} if legacy_failure else {}),
            **({_RECOVERED: recovered} if recovered else {}),
            **({_REFUSALS: prior[_REFUSALS]} if isinstance(prior.get(_REFUSALS), dict) and prior[_REFUSALS] else {}),
        }, _SCHEMA)
        result.update(record, newly_regressed=newly_regressed)
        written = True
        projection_changed = process == "server" and (
            _card_observation(prior_server) != _card_observation(observed)
            or bool(prior.get("regressed")) != regressed
            or (prior_failures != failures and _companion_overlay(prior_failures) != _companion_overlay(failures))
        )
        return record

    try:
        update_json_locked(health_path(drive_root, skill_name), _update, strict_existing_dict=True)
    except Exception:
        log.debug("Failed to persist extension health for %s", skill_name, exc_info=True)
        # The mutator may have run before a failed atomic write. Only persisted observations can be returned.
        return read_extension_health(drive_root, skill_name) or {}, "failed"
    if projection_changed:
        _notify_health_changed()
    return result, "written" if written else "unchanged"


def open_permission_refusals(record: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The permissions a skill reported unavailable and no later success reported available, by id."""
    stored = (record or {}).get(_REFUSALS)
    return {str(name): dict(fact) for name, fact in stored.items() if isinstance(fact, dict)} \
        if isinstance(stored, dict) else {}


def record_permission_observation(drive_root: pathlib.Path, skill_name: str, *,
                                  refused: Optional[Dict[str, Any]] = None, available: Sequence[str] = ()) -> bool:
    """Open the refusal of one permission the skill reported unavailable (``refused``: ``permission``, ``code``,
    ``repair``?, ``where``?), keeping its start while it stays open; close those a success reports ``available``.
    A change is written under the record's lock; an unchanged observation reads only the stat-cached snapshot.
    Without a health record there is nothing to attach it to. True when written."""
    permission = str((refused or {}).get("permission") or "")
    facts = {key: str(refused[key]) for key in ("code", "repair", "where") if refused and refused.get(key)}
    closing = set(available) - {permission}

    def changed(current: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        after = {name: fact for name, fact in current.items() if name not in closing}
        if permission:
            after[permission] = {"since": str((current.get(permission) or {}).get("since") or utc_now_iso()), **facts}
        return after

    snapshot = read_extension_health(drive_root, skill_name)
    held = open_permission_refusals(snapshot)
    if not snapshot or changed(held) == held:
        return False
    written = False

    def _update(prior: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal written
        current = open_permission_refusals(prior)
        after = changed(current)
        if not prior or after == current:
            return None
        written = True
        record = {key: value for key, value in prior.items() if key != _REFUSALS}
        return {**record, _REFUSALS: after} if after else record

    try:
        update_json_locked(health_path(drive_root, skill_name), _update, strict_existing_dict=True)
    except Exception:
        log.debug("Failed to persist a permission observation for %s", skill_name, exc_info=True)
        return False
    return written


def _notify_health_changed() -> None:
    try:
        from ouroboros.gateway.ws import notify_extension_health_changed

        notify_extension_health_changed()
    except Exception:
        log.debug("Extension health notification failed", exc_info=True)


def _companions_allow_known_good(skill_name: str, companions: Optional[List[str]]) -> bool:
    if not companions:
        return True
    from ouroboros.extension_companion import get_global_supervisor

    supervisor = get_global_supervisor()
    return supervisor is not None and supervisor.companions_allow_known_good(skill_name, companions)


def _journal(drive_root: pathlib.Path, kind: str, skill_name: str, companion: str, **facts: Any) -> None:
    """One companion transition in ``logs/events.jsonl``, which keeps it after the record changes again."""
    version, sha = code_stamp()
    row = {"ts": facts.pop("ts"), "type": kind, "skill": skill_name, "companion": companion, **facts,
           "version": version, "git_sha": sha}
    try:
        appended = append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", row)
    except Exception:
        log.debug("Failed to append %s event", kind, exc_info=True)
        appended = False
    if not appended:
        log.warning("extension companion event not journaled: %s", row)


def record_companion_restart_exhausted(
    drive_root: pathlib.Path,
    skill_name: str,
    companion_name: str,
    *,
    returncode: int,
    observed_at: Optional[str] = None,
    still_current: Optional[Callable[[], bool]] = None,
    retained: Optional[_RetainedCompanions] = None,
    restart_failed: bool = False,
    restart_attempted_at: str = "",
    started_at: str = "",
) -> Dict[str, Any]:
    """Journal an exhausted budget or failed restart; open it only while this start is still current."""
    return _record_companion_failure(
        drive_root, skill_name, companion_name, returncode=returncode, observed_at=observed_at or utc_now_iso(),
        still_current=still_current, retained=retained, kind="restart_failed" if restart_failed else "exhausted",
        details={**({"restart_failed": True} if restart_failed else {}),
                 **({"restart_attempted_at": restart_attempted_at} if restart_attempted_at else {})},
        started_at=started_at,
    )


def _record_companion_failure(drive_root, skill_name, companion_name, *, returncode, observed_at,
                              still_current, retained, kind, details, started_at=""):
    """One history row per observed exit; only its still-current owner may open health under the record lock.

    A failed health write retries on the monitor thread, never on a lock holder. Do not journal twice.
    ``started_at`` is the failed start's own time. A failure still open from before it was closed by that start,
    whose own close did not land: it closes here as of that start (``by: "exit"``) instead of being overwritten.
    """
    name, code = str(companion_name or "").strip(), int(returncode)
    _journal(drive_root, f"extension_companion_{kind}", skill_name, name, returncode=code, ts=observed_at,
             **({"restart_attempted_at": details["restart_attempted_at"]} if "restart_attempted_at" in details else {}))
    if still_current is not None and not still_current():
        return read_extension_health(drive_root, skill_name) or {}  # a withdrawn monitor has history only
    words = f"companion {name!r} exited with code {code}" + {
        "restart_failed": " and its automatic restart failed",
        "exhausted": " after exhausting its restart budget",
        "exited": "",
    }[kind]
    closed: Dict[str, Any] = {}

    def _open(failures: Dict[str, Dict[str, Any]], recovered: Dict[str, Any], _ts: str) -> bool:
        closed.clear()
        if still_current is not None and not still_current():
            return False
        if name in failures and _precedes_start(failures[name], started_at):
            closed.update(_close_failure_fact(failures, recovered, name, started_at, "exit"))
        failures[name] = {"returncode": code, "ts": observed_at, "load_error": words, **details,
                          **({"started_at": started_at} if started_at else {})}
        return True

    for wait in (*_EXHAUSTED_WRITE_RETRY_SEC, None):
        health, outcome = _record(drive_root, skill_name, status=LIVE, retained_companions=retained, change=_open)
        if outcome != "failed" or wait is None or (still_current is not None and not still_current()):
            break
        time.sleep(wait)
    if outcome == "written" and closed:
        _journal(drive_root, "extension_companion_recovered", skill_name, name, ts=closed.pop("closed_at"), **closed)
    if outcome == "failed":
        log.warning("extension %s: %s; its health record could not be written", skill_name, words)
        if still_current is not None and still_current():
            _notify_health_changed()  # the supervisor knows the exit even though health did not persist it
    return health


def record_companion_refresh_failure(drive_root: pathlib.Path, skill_name: str, name: str, *,
                                     still_current: Callable[[], bool], retained: _RetainedCompanions,
                                     replaced_start: str) -> None:
    """Record the failed replacement, not an unexpected exit or a spent restart budget."""
    observed_at = utc_now_iso()
    words = f"companion {name!r} could not be replaced after a settings refresh"
    closed: Dict[str, Any] = {}

    def open_failure(failures: Dict[str, Dict[str, Any]], recovered: Dict[str, Any], _ts: str) -> bool:
        if not still_current():
            return False
        if name in failures:
            if not _precedes_start(failures[name], replaced_start):
                return False  # a retry of this replacement does not reopen its diagnosis
            closed.update(_close_failure_fact(failures, recovered, name, replaced_start, "refresh"))
        failures[name] = {"ts": observed_at, "load_error": words, "refresh_failed": True, "started_at": replaced_start}
        return True

    _journal(drive_root, "extension_companion_refresh_failed", skill_name, name, ts=observed_at)
    _, outcome = _record(drive_root, skill_name, status=LIVE, retained_companions=retained, change=open_failure)
    if outcome == "written" and closed:
        _journal(drive_root, "extension_companion_recovered", skill_name, name, ts=closed.pop("closed_at"), **closed)
    if outcome == "failed":
        log.warning("extension %s: %s; its health record could not be written", skill_name, words)
        if still_current():
            _notify_health_changed()


def _precedes_start(failure: Dict[str, Any], started: str) -> bool:
    """Whether ``started`` came after this failure and so closed it. The failure of a start, or of the
    replacement of one, names that start (``started_at``): it never precedes that same start, whatever the clock."""
    failed = failure.get("ts")
    if not isinstance(failed, str) or not started or failure.get("started_at") == started:
        return False
    try:
        # A missing offset on a legacy record does not establish its order against this start.
        left, right = (datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in (failed, started))
        return left.tzinfo is not None and right.tzinfo is not None and left < right
    except ValueError:
        return False


def _close_failure_fact(failures, recovered, name, closed_at, by) -> Dict[str, Any]:
    failure = failures.pop(name)
    closed = dict(returncode=failure.get("returncode"), failed_at=failure.get("ts") or None, closed_at=closed_at, by=by)
    for key in ("load_error", "restart_failed", "restart_attempted_at", "exited", "refresh_failed"):
        if failure.get(key):
            closed[key] = failure[key]
    recovered[name] = dict(closed)
    return closed


def clear_companion_restart_exhausted(
    drive_root: pathlib.Path,
    skill_name: str,
    companion_name: str,
    *,
    still_current: Optional[Callable[[], bool]] = None,
    by: str = "start",
    retained: Optional[_RetainedCompanions] = None,
    closed_at: str = "",
) -> bool:
    """Close this companion's open terminal failure as of a later start, its current one, and journal it.

    The record's lock orders this after any failure still being written, never before it. ``closed_at`` is
    that start's own time (the observation the closing rests on); ``by`` names who wrote it: the ``start``
    itself, or a ``reconcile`` after that start's write was lost. ``still_current()`` is asked inside the
    record's update, so a newer death's failure stays; ``retained`` as for ``record_companion_restart_exhausted``.
    """
    name = str(companion_name or "").strip()
    closed: Dict[str, Any] = {}

    def _close(failures: Dict[str, Dict[str, Any]], recovered: Dict[str, Any], ts: str) -> bool:
        if name not in failures or (still_current is not None and not still_current()):
            return False
        closed.clear()
        closed.update(_close_failure_fact(failures, recovered, name, closed_at or ts, by))
        return True

    if _record(drive_root, skill_name, status=LIVE, reason="companion_restarted", retained_companions=retained,
               change=_close)[1] == "written":
        _journal(drive_root, "extension_companion_recovered", skill_name, name, ts=closed.pop("closed_at"), **closed)
        return True
    return False


def record_companion_exit(drive_root: pathlib.Path, skill_name: str, companion_name: str, *, returncode: int,
                          observed_at: str, still_current: Optional[Callable[[], bool]] = None,
                          retained: Optional[_RetainedCompanions] = None, started_at: str = "") -> None:
    """Journal an unexpected exit; if still current without a replacement, keep its failure open too."""
    if still_current is None:
        _journal(drive_root, "extension_companion_exited", skill_name, str(companion_name or "").strip(),
                 returncode=int(returncode), ts=observed_at)
    else:
        _record_companion_failure(drive_root, skill_name, companion_name, returncode=returncode,
                                  observed_at=observed_at, still_current=still_current, retained=retained,
                                  kind="exited", details={"exited": True}, started_at=started_at)


def _show_companion_failures(state: Dict[str, Any], failures: Dict[str, Dict[str, Any]], *,
                             fallback_reason: str = "", record_unavailable: bool = False) -> Dict[str, Any]:
    """A ready desired state shows the open companion failures; one showing failures that have all closed is ready
    again. A deferred reload (``extension_reload_pending``: the old publication stays live, at this revision or an
    older one) keeps its own reason and shows them in ``load_error`` unless it carries a load error of its own; the
    caller filters them by that held publication. Any other reason (a load error, a
    refusal, Off) keeps its own words. ``companion_failed`` marks words shown here over a state whose own
    ``load_error`` was none: a reason set later leaves without them, and a caller writing its own words drops the
    flag."""
    reason = str(state.get("reason") or "")
    shown = reason.startswith((f"{COMPANION_RESTART_EXHAUSTED}:", f"{COMPANION_FAILED}:"))
    deferred = state.get("action") == "extension_reload_pending" and bool(
        state.get("live_loaded") or state.get("loaded_present"))
    if not state.get("desired_live") or not (shown or deferred or reason in (
        "ready", "companion_restarted", "retained_unresolved", HEALTH_UNAVAILABLE, COMPANION_REFRESH_PENDING,
    )):
        if state.pop("companion_failed", None):
            state["load_error"] = None
        return state
    if record_unavailable and (shown or (deferred and state.get("companion_failed"))) and not failures:
        # A reconcile may have its earlier diagnosis, but no authoritative record to supersede it.
        return state
    if deferred:
        own_words = bool(state.get("load_error")) and not state.get("companion_failed")
        if failures and not own_words:
            state.update(companion_failed=True, load_error=_companion_overlay(failures)[1])
        elif state.pop("companion_failed", None):
            state["load_error"] = None
        return state
    if failures:
        reason, load_error = _companion_overlay(failures)
        state.update(companion_failed=True, reason=reason, load_error=load_error)
    else:
        state.pop("companion_failed", None)
        state["reason"] = "retained_unresolved" if state.get("retained_companions") else fallback_reason or "ready"
        if shown:
            state["load_error"] = None
    return state


def apply_companion_failure_to_runtime_state(
    state: Dict[str, Any],
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    skill_dir: str = "",
    companions: Optional[Sequence[str]] = None,
    foreign: bool = False,
) -> Dict[str, Any]:
    """Project server companion health, never a worker's local registration (``companions``, ``foreign``:
    ``_show_health``)."""
    if not state.get("desired_live") or state.get("process") == "worker":
        return state
    server = state.get("process") == "server"
    return _show_health(state, read_extension_health(drive_root, skill_name) or {}, skill_name, skill_dir,
                        companions if server else None, foreign=foreign and server)


def show_recorded_companion_health(state: Dict[str, Any], drive_root: pathlib.Path, skill_name: str) -> Dict[str, Any]:
    """A worker's read-only skill listing shows the companion failures the server last recorded, which no worker
    observes. The worker's own runtime state, reconcile receipt and health observation never carry them."""
    if state.get("desired_live") and state.get("process") == "worker":
        _show_health(state, read_extension_health(drive_root, skill_name) or {})
    return state


def _show_health(state: Dict[str, Any], health: Dict[str, Any], skill_name: str = "", skill_dir: str = "",
                 companions: Optional[Sequence[str]] = None, *, foreign: bool = False) -> Dict[str, Any]:
    """The one passive projection of a health record just read, by a listing or by the reconcile that wrote it.

    ``companions`` are the names the loaded publication of ``skill_dir`` registers (its bytes current or older,
    a reload waiting), once this server verified it (possibly none): a failure of any other name left with the
    publication that registered it, and one preceding its companion's last observed start was closed by that start,
    whether or not the close reached the record. None verifies nothing (a worker, nothing loaded): every recorded
    failure shows. ``foreign``: the publication this server holds came from another folder than the state's (or from
    one a settled reconcile cannot name), so the record's failures and the starts that closed them speak for that
    folder and none shows here; nothing vouches for this folder's companions, so a ready state reads
    ``health_unavailable`` (its own words and pending exits stay)."""
    if foreign:
        return _show_companion_failures(state, {}, fallback_reason=HEALTH_UNAVAILABLE)
    failures, unavailable = _open_companion_failures(health), health.get("reason") == HEALTH_UNAVAILABLE
    pending = unconfirmed = False
    if companions is not None:
        failures = {name: fact for name, fact in failures.items() if name in companions}
        from ouroboros.extension_companion import get_global_supervisor

        supervisor = get_global_supervisor() if companions else None
        if supervisor is not None:
            starts, pending, unconfirmed = supervisor.observed_companion_starts(skill_name, skill_dir, companions)
            failures = {name: fact for name, fact in failures.items()
                        if not _precedes_start(fact, starts.get(name, ""))}
    return _show_companion_failures(state, failures, record_unavailable=unavailable, fallback_reason=(
        COMPANION_REFRESH_PENDING if pending else HEALTH_UNAVAILABLE if unconfirmed or unavailable else ""))


def regressed_extensions(drive_root: pathlib.Path) -> List[Dict[str, Any]]:
    """Return health records for extensions currently flagged as regressed."""
    root = pathlib.Path(drive_root) / "state" / "skills"
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        record = read_extension_health(drive_root, skill_dir.name)
        if not (record and record.get("regressed")):
            continue
        # A regression alarm only matters while the skill still exists and is still
        # enabled. An uninstalled or owner-disabled skill must not raise a permanent
        # false CRITICAL (its health.json can outlive the payload).
        name = str(record.get("skill") or skill_dir.name)
        try:
            from ouroboros.skill_loader import find_skill

            skill = find_skill(pathlib.Path(drive_root), name)
            if skill is None or not skill.enabled:  # the effective switch: a required package stays alarmed
                continue
        except Exception:
            pass
        out.append(record)
    return out


@functools.lru_cache(maxsize=1)
def code_stamp() -> tuple[str, str]:
    """Return ``(version, git_sha)`` for live->broken attribution; ``("", "")`` on failure."""
    try:
        from ouroboros.config import read_version
        from ouroboros.utils import get_git_info

        return str(read_version()), get_git_info(pathlib.Path(__file__).resolve().parents[1])[1]
    except Exception:
        return "", ""


def fresh_code_stamp() -> tuple[str, str]:
    """Re-read the code stamp, dropping a value cached before a self-modification.

    ``code_stamp`` is cached so one reload does not re-run git per skill, but a
    live commit between reloads would otherwise attribute the new state to the
    pre-commit sha.
    """
    code_stamp.cache_clear()
    return code_stamp()


def status_for_runtime_state(state: Dict[str, Any]) -> str:
    """Map an extension runtime-state dict to a health status: a desired state showing the record's open companion
    failures (``companion_failed``) is BROKEN, as the record is."""
    if state.get("live_loaded"):
        status = LIVE
    elif state.get("desired_live") and (
        state.get("action") == "extension_load_error" or state.get("reason") == "load_error"
    ):
        status = BROKEN
    else:
        status = INACTIVE
    return _status_with_failures(status, state.get("desired_live") and state.get("companion_failed"))


def record_health_for_runtime_state(
    drive_root: pathlib.Path,
    skill_name: str,
    state: Dict[str, Any],
    *,
    stamp: tuple[str, str] | None = None,
    companions: Optional[List[str]] = None,
    observe_retained: Optional[Callable[[], None]] = None,
    skill_dir: str = "",
    foreign: bool = False,
) -> Dict[str, Any]:
    """Persist the health projection of one completed runtime reconcile.

    A companion failure the state shows is the record's own, re-applied under its lock at
    this write: the state's earlier copy is never written back, and ``state`` then shows
    the record this write left, as a listing would (``_show_health``; ``companions`` of the
    publication loaded from ``skill_dir``: see ``record_extension_health``), also when the
    write did not land. A server reconcile refreshes this state's retained companions through
    ``observe_retained`` at the same write. ``foreign``: the server holds a publication of the
    name this operation cannot place in its own folder (another folder's, or any under a shared
    name or none), so it observed nothing of the name's record (which workers read as the
    server's) and writes none; ``state`` shows none of it and only its own folder's pending exits
    are observed, and the record is returned as it stands.
    """
    process = str(state.get("process") or "server")
    if foreign and process == "server":
        if observe_retained is not None:
            observe_retained()
        _show_health(state, {}, foreign=True)
        return read_extension_health(drive_root, skill_name) or {}
    version, sha = fresh_code_stamp() if stamp is None else stamp
    base = _show_companion_failures({**state, "retained_companions": {}}, {})
    if process == "worker":
        state.pop("retained_companions", None)

    def retained() -> Dict[str, str]:
        if observe_retained is not None:
            observe_retained()
        return state.get("retained_companions") or {}

    # A later observation may release a retained process. Only the operation's own result says it succeeded.
    action = state.get("action")
    companion_action = (state.get("companions") or {}).get("action")
    succeeded = action == "extension_loaded" or (
        action == "extension_already_live"
        and companion_action in ("already_running", "started_missing", "no_registered_companions")
    )
    health, outcome = _record(
        drive_root,
        skill_name,
        status=status_for_runtime_state(base),
        version=version,
        sha=sha,
        reason=str(base.get("reason") or ""),
        load_error=str(base.get("load_error") or ""),
        process=process,
        server_reconcile=str(state.get("server_reconcile") or ""),
        companions=companions,
        retained_companions=retained,
        reconcile_succeeded=succeeded and base.get("reason") == "ready",
    )
    if process == "server":
        if outcome == "failed":
            retained()  # the writer may have failed before calling its mutator
        _show_health(state, health or {"reason": HEALTH_UNAVAILABLE}, skill_name, skill_dir, companions)
    if health.get("newly_regressed"):
        observed = health.get("last_observed") or {}
        regression = {
            "skill": skill_name,
            "last_known_good_sha": (health.get("last_known_good") or {}).get("sha", ""),
            "sha": sha,
            "load_error": str(state.get("load_error") or ""),
        }
        log.error(
            "Extension regression: %s was live at %s, broken now at %s: %s",
            skill_name, (regression["last_known_good_sha"] or "?")[:12],
            (sha or "?")[:12], regression["load_error"],
        )
        try:
            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
                "ts": str(observed.get("ts") or utc_now_iso()), "type": "extension_regression",
                "git_sha": sha, "version": version, "regressions": [regression],
            })
        except Exception:
            log.debug("Failed to append extension_regression event", exc_info=True)
    return health


__all__ = [
    "HEALTH_FILENAME",
    "LIVE",
    "BROKEN",
    "COMPANION_FAILED",
    "COMPANION_RESTART_EXHAUSTED",
    "INACTIVE",
    "UNKNOWN",
    "apply_companion_failure_to_runtime_state",
    "clear_companion_restart_exhausted",
    "code_stamp",
    "fresh_code_stamp",
    "health_path",
    "read_extension_health",
    "recorded_failed_companions",
    "record_companion_exit",
    "record_companion_refresh_failure",
    "record_companion_restart_exhausted",
    "record_extension_health",
    "record_health_for_runtime_state",
    "regressed_extensions",
    "show_recorded_companion_health",
    "status_for_runtime_state",
]
