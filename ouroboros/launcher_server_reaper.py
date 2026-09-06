"""Launcher-owned reaping of same-install stray `server.py` processes.

A launcher that holds the single-instance pid lock is, by that lock, the only actor entitled to run
a managed server against this data directory — so any OTHER process running THIS install's
`server.py` under THIS launcher's stamped environment is a leftover generation, not a peer. This
module finds those processes from live kernel state and kills them; `launcher.py` owns only the
thin wrapper and the call sites.

A pid is PROVEN only on three live facts, all read fresh: the exact `<REPO_DIR>/server.py` path in
its command line, `OUROBOROS_DATA_DIR=<our data dir>` in its environment, and
`OUROBOROS_MANAGED_BY_LAUNCHER=1` in its environment. `start_agent` stamps both assignments, so
every launcher-started generation carries them, while a direct or dev run of the same checkout does
not and is spared with a warning. An environment that cannot be READ is never a licence to kill.

The custody ledger is deliberately never consulted: missing ledger entries are the defect this
sweep repairs, so a ledger lookup would spare exactly the strays that matter. Enforcement requires a
byte-exact environment source, so kills happen only on /proc hosts (the field-incident platform);
elsewhere the sweep says so and does nothing — Windows orphans already die with the launcher's
kill-on-close Job Object, and the ps -E fallback mixes argv into the environment column, which
must never authorize a kill.

CONTRACT (disclosed residual): the lock-implies-orphan inference assumes PID_FILE and DATA_DIR
derive from one APP_ROOT, which every packaged install satisfies. An owner who deliberately
decouples them via environment overrides — two launchers holding DIFFERENT lock files over the
SAME data directory — is outside this contract: such a topology already corrupts the shared
state files that the single-instance lock exists to protect, with or without this sweep.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import time
from typing import Iterable, List, Optional, Set, Tuple

# Module-object access (not from-imports): tests monkeypatch these names on platform_layer.
from ouroboros import platform_layer as _pl
from ouroboros.process_containment import (
    ENV_ASSIGNMENT_PRESENT,
    pid_environment_assignment_state,
)

log = logging.getLogger(__name__)

# Both stamped by `launcher.start_agent`; both required before anything is signalled.
MANAGED_MARKER_ENV = "OUROBOROS_MANAGED_BY_LAUNCHER"
MANAGED_MARKER_VALUE = "1"
DATA_DIR_ENV = "OUROBOROS_DATA_DIR"

# A stray can fork between the scan and the signal and the child inherits both the cmdline and the
# environment, so one pass proves nothing. Bounded so a pid that refuses to die cannot turn this
# into an unbounded kill loop — the caller is told about survivors instead.
REAP_PASSES = 3
_SETTLE_SEC = 0.05
# How long a signalled pid gets to actually die before it is counted a survivor.
_CONFIRM_DEADLINE_SEC = 1.0


def _path_forms(base, leaf: str = "") -> Set[str]:
    """The literal and the symlink-resolved spelling of a path. A command line or an environment
    carries whatever spelling the launcher passed, which `resolve()` may not reproduce (``/var`` vs
    ``/private/var``), so both are accepted as the same install."""
    path = pathlib.Path(base)
    forms = {str(path / leaf) if leaf else str(path)}
    try:
        resolved = path.resolve()
        forms.add(str(resolved / leaf) if leaf else str(resolved))
    except OSError:
        pass
    return forms


def _candidate_commands() -> "Optional[dict]":
    """``pid -> full command line`` for THIS user's processes from ONE ``ps`` read, or ``None``
    when enumeration itself failed — which must never read as a clean sweep: nothing was checked.

    ``ps`` (not a pattern-scoped ``pgrep``): candidate selection must not depend on the install
    path containing any particular word — REPO_DIR is configurable, and the exact-token matcher is
    the real filter. ``-u <uid>``: other accounts run their own legitimate installs and are
    unsignallable anyway. ``-ww``: BSD ps truncates otherwise and the matcher needs exact argv."""
    try:
        out = subprocess.run(
            ["ps", "-ww", "-u", str(os.getuid()), "-o", "pid=,command="],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    commands: dict = {}
    for line in (out.stdout or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid > 0:
            commands[pid] = parts[1]
    return commands


def install_server_path_forms(repo_dir) -> Set[str]:
    """The spellings of THIS install's server.py a live command line may carry."""
    return _path_forms(repo_dir, "server.py")


def command_names_our_server(command: str, server_paths: Set[str]) -> bool:
    """Whether a live command line IS this install's server: an exact argv token equal to
    ``<REPO_DIR>/server.py`` directly after a python-named interpreter token — the launcher's only
    spawn shape. A bare substring test would also match editors, pagers or log tools whose
    arguments merely mention the path; those are not server generations and must not even appear
    as spared candidates. A sibling install, a bench clone and a dev checkout carry different
    paths and never match. Shared with the startup stray check so its ``scope`` labels agree with
    what this sweep would actually treat as a server."""
    tokens = (command or "").split()
    for i in range(1, len(tokens)):
        if tokens[i] in server_paths and pathlib.PurePath(tokens[i - 1]).name.lower().startswith("python"):
            return True
    return False


def _runs_our_server(pid: int, server_paths: Set[str]) -> bool:
    return command_names_our_server(_pl.process_command(pid) or "", server_paths)


def _is_launcher_managed(pid: int, data_dir_values: Set[str]) -> bool:
    """Whether ``pid``'s live environment carries BOTH launcher assignments.

    ABSENT and UNREADABLE are equally not-a-proof: a direct run of this checkout lacks the marker,
    and an environment that could not be read has answered nothing."""
    if (pid_environment_assignment_state(pid, MANAGED_MARKER_ENV, MANAGED_MARKER_VALUE)
            != ENV_ASSIGNMENT_PRESENT):
        return False
    return any(
        pid_environment_assignment_state(pid, DATA_DIR_ENV, value) == ENV_ASSIGNMENT_PRESENT
        for value in data_dir_values
    )


def find_same_install_server_pids(
    repo_dir, data_dir, exclude_pids: "Optional[Iterable[int]]" = None
) -> "Tuple[List[int], List[int]]":
    """``(proven, unproven)`` pids running this install's ``server.py``, from live kernel state.

    ``proven`` carries all three facts and may be killed; ``unproven`` matches the path only and is
    reported so a spared process is never invisible."""
    if _pl.IS_WINDOWS:
        return [], []
    server_paths = _path_forms(repo_dir, "server.py")
    data_dir_values = _path_forms(data_dir)
    if any(" " in path for path in server_paths):
        # The exact-token proof cannot represent a whitespace path, so nothing
        # could ever be proven; the caller (reap) names this once per sweep.
        return [], []
    candidates = _candidate_commands()
    if candidates is None:
        # Enumeration itself failed: nothing was CHECKED, which is different from
        # nothing found. Raising (instead of returning empty) routes a MID-sweep
        # failure into reap's aborted-mid-work path, so pids already proven in
        # this sweep are still reported as survivors; the systemic no-ps case
        # is answered by reap's pre-check with its own named warning.
        raise RuntimeError("process enumeration unavailable")
    known = {os.getpid(), os.getppid()}
    known.update(int(pid) for pid in (exclude_pids or ()) if int(pid) > 0)
    known_groups: Set[int] = set()
    for pid in known:
        if pid > 0:
            try:
                known_groups.add(_pl.process_group_id(pid))
            except Exception:
                continue
    known_groups.discard(0)
    proven: List[int] = []
    unproven: List[int] = []
    for pid, command in sorted(candidates.items()):
        if pid in known or not command_names_our_server(command, server_paths):
            continue
        try:
            if _pl.process_group_id(pid) in known_groups:
                continue  # part of a tree we already know about
        except Exception:
            pass
        (proven if _is_launcher_managed(pid, data_dir_values) else unproven).append(pid)
    return proven, unproven


def _env_proof_available() -> bool:
    """Whether this host offers a byte-exact environment source (/proc). The ps -E fallback
    appends the environment to the SAME column as the argv, so an argv token spelled
    ``KEY=value`` is indistinguishable from a real assignment — that is not a kill-grade
    proof. Capability probe, not platform sniffing: a /proc-less POSIX host answers no."""
    return os.path.isdir("/proc/self")


def _signal_pid(pid: int) -> None:
    """Force-kill one pid via the platform layer; every failure mode is answered by the
    liveness read that follows (the primitive swallows per-pid errors)."""
    _pl.force_kill_pid(pid)


def _pid_gone(pid: int) -> bool:
    """Positive nonexistence via the platform layer: EPERM means the process EXISTS and a
    survivor must block the boot — the liveness read treats it as present, so a colliding
    generation never starts next to a live one it merely may not signal."""
    return _pl.pid_provably_gone(pid)


def _revalidate_and_kill(pid: int, server_paths: Set[str], data_dir_values: Set[str]) -> bool:
    """Re-prove ``pid`` from live state and signal the PROVEN ROOT with NOTHING in between: any
    lookup in that gap is a window for the pid to exit and be recycled onto a stranger —
    descendants are therefore captured before revalidation, then the root is signalled directly
    before those captured children. Trees, never process groups: server workers hold their own
    sessions and a reused pgid reaches bystanders.

    True means CONFIRMED DEAD, not merely signalled: the kill primitives swallow per-pid errors,
    so only a liveness read after the signal can say what it achieved — a pid logged as reaped
    while it survived would contradict the survivor report from the same generation."""
    # Descendants are captured BEFORE the root signal: SIGKILLing the root
    # reparents its children to init, after which no parent-walk finds them. A
    # fork landing after this capture is the next pass's job — that is what the
    # bounded rescans exist for.
    descendants = _pl.collect_descendant_pids(pid)
    if not _runs_our_server(pid, server_paths) or not _is_launcher_managed(pid, data_dir_values):
        return False
    _signal_pid(pid)
    for child in descendants:
        _signal_pid(child)
    deadline = time.time() + _CONFIRM_DEADLINE_SEC
    while True:
        if _pid_gone(pid):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(_SETTLE_SEC)


def reap_same_install_strays(
    repo_dir, data_dir, reason: str = "startup",
    exclude_pids: "Optional[Iterable[int]]" = None,
) -> List[int]:
    """Kill proven same-install strays; return the proven pids still alive after the last pass.

    A non-empty return is the caller's signal that starting another generation would collide."""
    if _pl.IS_WINDOWS:
        return []
    if not _env_proof_available():
        # ps -E mixes argv and environment into one column, so a KEY=value argv
        # token reads as an assignment — argv must never authorize a kill. The
        # field incident is a /proc platform; elsewhere the sweep is honest
        # about doing nothing rather than killing on a spoofable proof.
        log.warning(
            "Same-install stray sweep is report-only on this host (%s): no byte-exact "
            "environment source (/proc), and the ps fallback cannot distinguish an argv "
            "token from a real assignment — no process was killed.", reason,
        )
        return []
    server_paths = _path_forms(repo_dir, "server.py")
    data_dir_values = _path_forms(data_dir)
    if any(" " in path for path in server_paths):
        log.warning(
            "Same-install stray sweep disabled (%s): the repo path contains whitespace, which the "
            "exact-token identity proof cannot represent — no process was checked.", reason,
        )
        return []
    if _candidate_commands() is None:
        log.warning(
            "Same-install stray sweep could not enumerate processes (%s) — nothing was checked, "
            "which is not a clean sweep.", reason,
        )
        return []
    killed: Set[int] = set()
    proven_seen: Set[int] = set()
    unproven_seen: Set[int] = set()
    survivors: List[int] = []
    try:
        for attempt in range(REAP_PASSES):
            proven, unproven = find_same_install_server_pids(repo_dir, data_dir, exclude_pids)
            proven_seen.update(proven)
            # Accumulated across passes: a spared process that first appears on
            # pass 2 or 3 (a fork of an unproven direct run mid-sweep) must be
            # just as visible as one seen up front.
            unproven_seen.update(unproven)
            if not proven:
                break
            for pid in sorted(proven):
                if _revalidate_and_kill(pid, server_paths, data_dir_values):
                    killed.add(pid)
            time.sleep(_SETTLE_SEC)
        else:
            # The pass budget ran out with pids still proven: read survivors FRESH rather than
            # inferring them from the last kill attempt, which cannot see what the signal achieved.
            survivors, _ = find_same_install_server_pids(repo_dir, data_dir, exclude_pids)
    except Exception:
        # A sweep interrupted mid-work must not read as swept-clean: everything proven in this
        # sweep and not confirmed dead is reported as surviving, so the caller cannot start a
        # colliding generation on the strength of an exception.
        log.warning("Same-install stray sweep aborted mid-work (%s)", reason, exc_info=True)
        survivors = sorted(proven_seen - killed)
    if unproven_seen:
        log.warning(
            "Sparing same-install server process(es) with no launcher proof (%s): %s — a "
            "direct run of this checkout, or an environment that could not be read.",
            reason, sorted(unproven_seen),
        )
    if killed:
        log.info("Reaped same-install stray server process(es) (%s): %s", reason, sorted(killed))
    return sorted(survivors)
