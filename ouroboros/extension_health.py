"""Durable per-extension health vector for live->broken regression detection.

A small immune-system instrument (BIBLE P1 "discrepancy between expected and actual
state — immediate alert"; P3 health invariants): if an extension that was live at a
prior code version stops loading after a self-modification + restart, record the
regression and surface it to the owner/agent. This is warning-only — it never
disables a skill and never mutates git. The record lives next to the other per-skill
owner-state files at ``data/state/skills/<name>/health.json``.
"""

from __future__ import annotations

import functools
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.skill_loader import skill_state_dir
from ouroboros.utils import append_jsonl, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

HEALTH_FILENAME = "health.json"
_SCHEMA = 1

# Health statuses derived from the extension runtime state.
LIVE = "live"        # desired_live and loaded successfully
BROKEN = "broken"    # desired_live but failed to load
INACTIVE = "inactive"  # disabled, deps pending, review-gated, or not an extension
UNKNOWN = "unknown"    # no authoritative server observation has been recorded yet
COMPANION_RESTART_EXHAUSTED = "companion_restart_exhausted"


def health_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    return skill_state_dir(pathlib.Path(drive_root), skill_name) / HEALTH_FILENAME


def read_extension_health(drive_root: pathlib.Path, skill_name: str) -> Optional[Dict[str, Any]]:
    return read_json_dict(health_path(drive_root, skill_name))


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
) -> Dict[str, Any]:
    """Persist one process-qualified observation and the server health projection.

    ``regressed`` (persisted) stays true while a once-live extension is broken, so
    the UI and health invariants keep surfacing it until it loads again.
    ``newly_regressed`` (returned, not persisted) marks the live->broken transition
    so callers can log/alert once per transition rather than every restart.
    Worker observations are qualifiers only: they never replace server authority or
    advance the server's ``last_known_good``.
    """
    process = "worker" if process == "worker" else "server"
    now = utc_now_iso()
    observed = {
        "version": version,
        "sha": sha,
        "status": status,
        "reason": reason,
        "load_error": (load_error or "")[:2000],
        "ts": now,
        "process": process,
    }
    if process == "worker":
        observed["server_reconcile"] = str(server_reconcile or "")
    result: Dict[str, Any] = {}

    def _update(prior: Dict[str, Any]) -> Dict[str, Any]:
        observations = prior.get("observations")
        observations = dict(observations) if isinstance(observations, dict) else {}
        legacy = prior.get("last_observed")
        if "server" not in observations and isinstance(legacy, dict) and legacy:
            observations["server"] = dict(legacy, process="server")
        prior_server = observations.get("server")
        prior_status = str(prior_server.get("status") or "") if isinstance(prior_server, dict) else ""
        observations[process] = observed
        last_known_good = prior.get("last_known_good")
        if not isinstance(last_known_good, dict):
            last_known_good = None
        regressed = bool(prior.get("regressed"))
        newly_regressed = False
        if process == "server":
            if status == LIVE:
                last_known_good = {"version": version, "sha": sha, "ts": now}
                regressed = False
            elif status == BROKEN and last_known_good is not None:
                # A same-sha break is environmental, not a code regression.
                regressed = str(last_known_good.get("sha") or "") != str(sha or "")
                newly_regressed = regressed and prior_status == LIVE
            else:
                regressed = False
            authoritative = observed
            authoritative_status = status
        else:
            authoritative = prior_server if isinstance(prior_server, dict) else {}
            authoritative_status = str(authoritative.get("status") or UNKNOWN)
        record = with_schema_version({
            "skill": skill_name,
            "status": authoritative_status,
            "regressed": regressed,
            "last_known_good": last_known_good,
            "last_observed": authoritative,
            "observations": observations,
        }, _SCHEMA)
        result.update(record, newly_regressed=newly_regressed)
        return record

    try:
        update_json_locked(health_path(drive_root, skill_name), _update)
    except Exception:
        log.debug("Failed to persist extension health for %s", skill_name, exc_info=True)
        if not result:
            _update(read_extension_health(drive_root, skill_name) or {})
    return result


def _companion_failure_reason(companion_name: str) -> str:
    return f"{COMPANION_RESTART_EXHAUSTED}:{str(companion_name or '').strip()}"


def record_companion_restart_exhausted(
    drive_root: pathlib.Path,
    skill_name: str,
    companion_name: str,
    *,
    returncode: int,
) -> Dict[str, Any]:
    """Persist terminal companion failure in the extension's existing health row."""
    prior = read_extension_health(drive_root, skill_name) or {}
    observed = prior.get("last_observed") or {}
    name = str(companion_name or "").strip()
    return record_extension_health(
        drive_root,
        skill_name,
        status=BROKEN,
        version=str(observed.get("version") or ""),
        sha=str(observed.get("sha") or ""),
        reason=_companion_failure_reason(name),
        load_error=(
            f"companion {name!r} exited with code {int(returncode)} "
            "after exhausting its restart budget"
        ),
    )


def clear_companion_restart_exhausted(
    drive_root: pathlib.Path,
    skill_name: str,
    companion_name: str,
) -> None:
    """Clear this companion's terminal failure after a successful fresh start."""
    prior = read_extension_health(drive_root, skill_name) or {}
    observed = prior.get("last_observed") or {}
    if (
        observed.get("status") != BROKEN
        or observed.get("reason") != _companion_failure_reason(companion_name)
    ):
        return
    record_extension_health(
        drive_root,
        skill_name,
        status=LIVE,
        version=str(observed.get("version") or ""),
        sha=str(observed.get("sha") or ""),
        reason="companion_restarted",
    )


def apply_companion_failure_to_runtime_state(
    state: Dict[str, Any],
    drive_root: pathlib.Path,
    skill_name: str,
) -> Dict[str, Any]:
    """Project a durable exhausted companion into the normal runtime state."""
    if not state.get("desired_live"):
        return state
    health = read_extension_health(drive_root, skill_name) or {}
    observed = health.get("last_observed") or {}
    reason = str(observed.get("reason") or "")
    if observed.get("status") != BROKEN or not reason.startswith(
        f"{COMPANION_RESTART_EXHAUSTED}:"
    ):
        return state
    state.update(
        companion_failed=True,
        reason=reason,
        load_error=str(observed.get("load_error") or "companion restart budget exhausted"),
    )
    return state


def regressed_extensions(drive_root: pathlib.Path) -> List[Dict[str, Any]]:
    """Return health records for extensions currently flagged as regressed."""
    root = pathlib.Path(drive_root) / "state" / "skills"
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        record = read_json_dict(skill_dir / HEALTH_FILENAME)
        if not (record and record.get("regressed")):
            continue
        # A regression alarm only matters while the skill still exists and is still
        # enabled. An uninstalled or owner-disabled skill must not raise a permanent
        # false CRITICAL (its health.json can outlive the payload).
        name = str(record.get("skill") or skill_dir.name)
        try:
            from ouroboros.skill_loader import find_skill, load_enabled

            if find_skill(pathlib.Path(drive_root), name) is None or not load_enabled(pathlib.Path(drive_root), name):
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
    """Map an extension runtime-state dict to a health status."""
    if state.get("desired_live") and state.get("companion_failed"):
        return BROKEN
    if state.get("live_loaded"):
        return LIVE
    if state.get("desired_live") and (
        state.get("action") == "extension_load_error" or state.get("reason") == "load_error"
    ):
        return BROKEN
    return INACTIVE


def record_health_for_runtime_state(
    drive_root: pathlib.Path,
    skill_name: str,
    state: Dict[str, Any],
    *,
    stamp: tuple[str, str] | None = None,
) -> Dict[str, Any]:
    """Persist the health projection of one completed runtime reconcile."""
    version, sha = fresh_code_stamp() if stamp is None else stamp
    health = record_extension_health(
        drive_root,
        skill_name,
        status=status_for_runtime_state(state),
        version=version,
        sha=sha,
        reason=str(state.get("reason") or ""),
        load_error=str(state.get("load_error") or ""),
        process=str(state.get("process") or "server"),
        server_reconcile=str(state.get("server_reconcile") or ""),
    )
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
    "COMPANION_RESTART_EXHAUSTED",
    "INACTIVE",
    "UNKNOWN",
    "apply_companion_failure_to_runtime_state",
    "clear_companion_restart_exhausted",
    "code_stamp",
    "fresh_code_stamp",
    "health_path",
    "read_extension_health",
    "record_companion_restart_exhausted",
    "record_extension_health",
    "record_health_for_runtime_state",
    "regressed_extensions",
    "status_for_runtime_state",
]
