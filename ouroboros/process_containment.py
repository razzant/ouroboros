"""Process containment for the hermetic commit gate (v6.88 two-pass work).

`ProcessContainer` is DETECTION for a spawned tree that outlives its root — an
honest membership answer at teardown, not a kill guarantee. Env-token membership
(`OURO_PROC_CONTAINER_<uuid>`) is read from live kernel state (`/proc` environ on
Linux, `ps -E` on macOS); Windows membership is a kill-on-close Job Object. The
OS-primitive helpers (group kill, Job Object ABI, force kill, resume) stay in
`ouroboros.platform_layer`; this module owns the containment POLICY built on them.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
import time
import uuid
from typing import List, Optional

# Module-object access (not from-imports): tests and callers monkeypatch these
# names on platform_layer, and rebinding them here would freeze the originals.
from ouroboros import platform_layer as _pl

log = logging.getLogger(__name__)


# `ProcessContainer` token prefix. Deliberately NOT in the scrubbed `OUROBOROS_*` namespace: a
# nested container must keep the OUTER token so an outer reap sees the whole tree; uuids compose.
CONTAINMENT_ENV_PREFIX = "OURO_PROC_CONTAINER_"


# Tri-state membership: UNREADABLE is deliberately NOT a "no" — reading a nondumpable member as a
# non-member is how a live descendant would leave containment without exiting.
MARKER_MEMBER = "member"
MARKER_ABSENT = "absent"
MARKER_UNREADABLE = "unreadable"


def pid_marker_state(pid: int, marker: str) -> str:
    """Tri-state membership for ONE pid from live kernel state: ABSENT means ANSWERED-not-a-member;
    UNREADABLE means unanswerable — a leak only for an attributed member. Windows: ABSENT (job = membership)."""
    if _pl.IS_WINDOWS or not marker or int(pid) <= 0:
        return MARKER_ABSENT
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{int(pid)}/environ", "rb") as handle:
                data = handle.read()
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH, errno.ENOTDIR):
                return MARKER_ABSENT  # the pid is gone
            return MARKER_UNREADABLE  # nondumpable, or another user's process
        return MARKER_MEMBER if marker.encode("utf-8", "replace") in data else MARKER_ABSENT
    try:
        out = subprocess.run(["ps", "-E", "-ww", "-p", str(int(pid)), "-o", "command="],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        # No usable `ps` on a system with no /proc: unanswered, not answered "no".
        return MARKER_UNREADABLE
    if out.returncode != 0:
        # `ps -p` fails only when there is no such process, so this is the liveness probe too.
        return MARKER_ABSENT
    if marker in (out.stdout or ""):
        return MARKER_MEMBER
    # ALIVE, and `ps` showed no token. Unlike /proc's EACCES, `ps -E` reports a process whose
    # environment it may not read by OMITTING it — identical to one that never carried the token.
    # Unanswered, then, and unanswered is a leak: it stops a nondumpable member leaving quietly.
    return MARKER_UNREADABLE


# Tri-state env ASSIGNMENT, a different question from `pid_marker_state`'s membership: the caller
# KILLS on the answer, so PRESENT is the only affirmative and both ABSENT and UNREADABLE deny it.
ENV_ASSIGNMENT_PRESENT = "present"
ENV_ASSIGNMENT_ABSENT = "absent"
ENV_ASSIGNMENT_UNREADABLE = "unreadable"


def pid_environment_assignment_state(pid: int, key: str, value: str) -> str:
    """Whether ``pid``'s LIVE environment carries exactly ``key=value``.

    ABSENT means answered-not-carried; UNREADABLE means unanswerable. Windows is always UNREADABLE:
    no caller may claim a Windows proof from this."""
    if _pl.IS_WINDOWS or not key or int(pid) <= 0:
        return ENV_ASSIGNMENT_UNREADABLE
    assignment = f"{key}={value}"
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{int(pid)}/environ", "rb") as handle:
                data = handle.read()
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH, errno.ENOTDIR):
                return ENV_ASSIGNMENT_ABSENT  # the pid is gone
            return ENV_ASSIGNMENT_UNREADABLE  # nondumpable, or another user's process
        # Split on the NUL delimiter and compare WHOLE entries: a substring test would let
        # `OTHER_KEY=<ours>` or `KEY=<ours>/sub` answer for `KEY=<ours>`.
        entries = data.split(b"\0")
        return (ENV_ASSIGNMENT_PRESENT
                if assignment.encode("utf-8", "replace") in entries
                else ENV_ASSIGNMENT_ABSENT)
    try:
        out = subprocess.run(["ps", "-E", "-ww", "-p", str(int(pid)), "-o", "command="],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return ENV_ASSIGNMENT_UNREADABLE
    if out.returncode != 0:
        # `ps -p` fails only when there is no such process.
        return ENV_ASSIGNMENT_ABSENT
    if assignment in (out.stdout or "").split():
        return ENV_ASSIGNMENT_PRESENT
    # ALIVE, and `ps -E` showed no such assignment. Unlike /proc's EACCES, `ps -E` reports a process
    # whose environment it may not read by OMITTING it — indistinguishable from one that never
    # carried the assignment (as is a value containing whitespace). Unanswered, not answered "no".
    return ENV_ASSIGNMENT_UNREADABLE


def pid_is_zombie(pid: int) -> bool:
    """Whether ``pid`` is an already-exited process still holding a table slot. A SIGKILLed child of
    THIS process keeps its pid, pgid and ``ps`` row until someone ``wait()``s it, and the preflight
    reaps before waiting pytest; a corpse can execute nothing, so counting it only burns time."""
    if _pl.IS_WINDOWS or int(pid) <= 0:
        return False
    try:
        if os.path.isdir("/proc"):
            # comm is parenthesised and may contain ')', so state is the field after the LAST.
            with open(f"/proc/{int(pid)}/stat", "rb") as handle:
                fields = handle.read().rpartition(b")")[2].split()
            return bool(fields) and fields[0] == b"Z"
        out = subprocess.run(["ps", "-o", "state=", "-p", str(int(pid))],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and (out.stdout or "").strip().startswith("Z")
    except Exception:
        return False


def process_group_has_live_members(pgid: int) -> bool:
    """Whether a recorded service group can still execute; unknown stays alive.

    Signal-zero sees a zombie-only group as present. Inspect ALL group members,
    because an exited leader can still have a live child. This is a fresh read,
    not a saved process-table snapshot or new authority to signal a group.
    """
    if not _pl.process_group_is_alive(pgid):
        return False
    try:
        rows = subprocess.run(["ps", "-A", "-o", "pgid=,state="],
                              capture_output=True, text=True, timeout=5)
        if rows.returncode != 0:
            return True
        states = [fields[1] if len(fields) == 2 else "unknown"
                  for line in rows.stdout.splitlines()
                  if (fields := line.split()) and fields[0] == str(pgid)]
        if states:
            return any(not state.startswith("Z") for state in states)
    except (OSError, subprocess.TimeoutExpired):
        return True
    # No visible member is not proof that a still-existing group is empty.
    return _pl.process_group_is_alive(pgid)


def _unattributed_candidate(pid: int, since_ticks: int) -> bool:
    """Select plausible unreadable strangers for disclosure, never membership.

    Same euid and a recent start cannot attribute an SSH session or other
    concurrent process to this pass. Known roots, groups and token members
    remain fail-closed separately in ``ProcessContainer._scan``.
    """
    started = _pl._proc_start_ticks(pid)
    if since_ticks > 0 and 0 < started < since_ticks:
        return False
    try:
        with open(f"/proc/{int(pid)}/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"Uid:"):  # real, EFFECTIVE, saved, fs
                    return int(line.split()[2]) == os.geteuid()
    except (OSError, ValueError, IndexError):
        return True
    return True


def pids_with_env_marker(marker: str, pgid: int = 0, since_ticks: int = 0) -> "Optional[List[int]]":
    """Pids that belong to a container, read from live kernel state; ``None`` when the process
    table could NOT be read at all (conflating that with "empty" reports a clean reap).

    TWO positive signals: the kernel-copied ENVIRONMENT token (survives
    setsid/fd-closing/reparenting) and the kernel-held PROCESS GROUP (names
    nondumpable and env-replaced members). Plausible unreadable strangers are
    disclosed, not attributed by their uid or start time.
    The group is an ENUMERATION input only — ``reap`` never signals by pgid."""
    if _pl.IS_WINDOWS or not marker:
        return []
    found: List[int] = []
    if os.path.isdir("/proc"):
        try:
            entries = os.listdir("/proc")
        except OSError:
            return None
        unattributed: List[int] = []
        for name in entries:
            if not name.isdigit():
                continue
            state = pid_marker_state(int(name), marker)
            if (state == MARKER_MEMBER
                    or (pgid and _pl.process_group_id(int(name)) == pgid)):
                found.append(int(name))
            elif state == MARKER_UNREADABLE and _unattributed_candidate(int(name), since_ticks):
                unattributed.append(int(name))
        if unattributed:
            log.warning("Unattributed processes have unreadable environments; not counted as "
                        "container members or signalled: %s. Detached descendants that hid their "
                        "token before observation cannot be ruled out.", unattributed)
        return found
    try:
        out = subprocess.run(["ps", "-E", "-ww", "-Ao", "pid=,pgid=,command="],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    for line in (out.stdout or "").splitlines():
        head = line.split(maxsplit=2)
        if len(head) < 3 or not head[0].isdigit():
            continue
        if marker in line or (pgid and head[1] == str(int(pgid))):
            found.append(int(head[0]))
    return found


# A member can fork between scans and its child inherits the token, so one quiet scan proves little:
# `reap` needs `_REAP_QUIET_SCANS` scans in a row with nothing live and nothing undeterminable, and
# FAILS if that has not happened by `_REAP_DEADLINE_SEC`.
_REAP_QUIET_SCANS = 2
_REAP_DEADLINE_SEC = 10.0
_REAP_SETTLE_SEC = 0.05


def _containment_leak_reason(alive: List[int], undetermined: List[int]) -> str:
    """The failure text ``reap`` returns. Both lists are leaks; they differ only in remediation."""
    parts = []
    if alive:
        parts.append("still alive after a best-effort kill: "
                     + ", ".join(str(pid) for pid in alive))
    if undetermined:
        parts.append("liveness could not be determined (the attributed process environment could not be "
                     "read): " + ", ".join(str(pid) for pid in undetermined))
    detail = "; ".join(parts) or ("no member was visible in the last scan, but two consecutive "
                                  "quiet scans were never reached, so nothing is proven gone")
    return (f"the contained tree could not be proven gone within {_REAP_DEADLINE_SEC:.0f}s — "
            f"{detail}")


class ProcessContainer:
    """DETECTION for a spawned tree that outlives its root — not a teardown guarantee.

    POSIX offers no guaranteed teardown of a detached descendant (pids are reusable names,
    membership is not kernel-held, a signal can be refused or land on a recycled stranger), so this
    container claims an honest ANSWER instead: ``reap`` resolves membership
    (``pids_with_env_marker``) from the LIVE table at teardown — never from mid-run samples — then
    attempts one bounded best-effort kill sweep and reports every member still alive or
    undeterminable to the caller, which hard-blocks. Residual limit on Linux: a descendant never
    observed as a member that leaves the group and hides/replaces the token; without ``/proc`` (macOS/BSD) one that
    sheds the token and leaves the group. Windows membership is the Job Object (kernel-enforced
    teardown — but only when the API confirms it). Prefer ``spawn`` over ``Popen`` + ``adopt``:
    POSIX ``adopt`` can neither plant the token nor vouch for a pid or group."""

    def __init__(self) -> None:
        self._job = None
        # The pid `spawn` started and the group it leads: known WITHOUT having to be read.
        self._root = 0
        self._pgid = 0
        # The root's start time: nothing the container spawned can predate it.
        self._start_ticks = 0
        self._suspended = False
        # Non-empty when containment was never ESTABLISHED: else it reads as "everything reaped".
        self._setup_error = ""
        # Unique per instance: a nested preflight and its outer run never claim each other's.
        self._token = f"{CONTAINMENT_ENV_PREFIX}{uuid.uuid4().hex}"

    def containment_env(self) -> "dict[str, str]":
        """Environment entries that make a process and its descendants members. ``spawn`` applies
        these; a caller building its own env for ``Popen`` + ``adopt`` must merge them in."""
        return {self._token: "1"}

    def spawn(self, argv: List[str], **popen_kwargs) -> subprocess.Popen:
        """``Popen`` the process already inside the container, with no gap. The token is merged into
        the caller's ``env`` (defaulting to the inherited one) BEFORE the process exists, so no
        descendant can start outside the membership. On Windows the root is created SUSPENDED too:
        a descendant preceding the job assignment would survive terminate/close. ``adopt`` resumes."""
        kwargs = dict(popen_kwargs)
        group_kwargs = dict(_pl.subprocess_new_group_kwargs())
        flags = int(kwargs.pop("creationflags", 0)) | int(group_kwargs.pop("creationflags", 0))
        kwargs.update(group_kwargs)
        env = kwargs.get("env")
        kwargs["env"] = {**(os.environ if env is None else env), **self.containment_env()}
        if _pl.IS_WINDOWS:
            flags |= getattr(subprocess, "CREATE_SUSPENDED", 0x4)
            self._suspended = True
        if flags:
            kwargs["creationflags"] = flags
        proc = subprocess.Popen(argv, **kwargs)
        # Knowledge no scan can re-derive: enumeration claims members by READING them, so a root
        # turning nondumpable before the first scan is in no list. `start_new_session` makes it its
        # own group LEADER (pgid == pid); enumerating any OTHER pgid would report bystanders.
        self._root = int(getattr(proc, "pid", 0) or 0)
        pgid = _pl.process_group_id(self._root)
        self._pgid = pgid if pgid and pgid == self._root else 0
        self._start_ticks = _pl._proc_start_ticks(self._root)
        self.adopt(proc)
        return proc

    def adopt(self, proc: subprocess.Popen) -> None:
        """Take custody of a just-spawned process; call right after ``Popen``. A no-op on POSIX: the
        token can only be planted before the process exists, and only ``spawn`` knows the pid and
        the group it planted it into. A ``Popen`` + ``adopt`` caller must merge
        ``containment_env()`` into the env it passes, or nothing is contained."""
        if _pl.IS_WINDOWS:
            self._job = self._adopt_windows(proc)
            if self._job is None:
                self._setup_error = (
                    "the Windows Job Object could not be created, or the pytest root could not "
                    "be assigned to it, so nothing the run spawned would be kernel-held; the "
                    "still-suspended root was terminated rather than resumed uncontained"
                )

    def _adopt_windows(self, proc: subprocess.Popen):
        """Put ``proc`` in a kill-on-close Job Object; return the handle or None. It is created
        suspended so nothing escapes before assignment, never LEFT suspended (a caller waiting on it
        would deadlock), and never resumed uncontained — an unheld root starts descendants that
        survive terminate/close. A failed create/assign kills the root."""
        pid = int(getattr(proc, "pid", 0) or 0)
        job = _pl.create_kill_on_close_job()
        if job is not None and not _pl.assign_pid_to_job(job, pid):
            _pl.close_job(job)
            job = None
        if job is None:
            self._suspended = False
            _pl.force_kill_pid(pid)
            return None
        if self._suspended:
            self._suspended = False
            # If even the resume fails the process is unusable, so tear it down.
            if not _pl.resume_process(pid):
                _pl.terminate_job(job)
                _pl.close_job(job)
                _pl.force_kill_pid(pid)
                return None
        return job

    def _scan(self, token: str, pgid: int, known: "set[int]", kill: bool, since: int = 0
              ) -> "tuple[List[int], List[int], str]":
        """ONE membership scan: ``(still alive, undeterminable, enumeration error)``. ``kill`` is
        true for the single sweep only. ``known`` is carried across scans and only ever GROWS: once
        a pid has been seen as a member, dropping out of a later enumeration proves nothing — that
        is what a member turning unreadable looks like too."""
        members = pids_with_env_marker(token, pgid, since)
        if members is None:
            return [], [], ("the live process table could not be enumerated, so the container "
                            "cannot say whether the tree it spawned is still running")
        me = os.getpid()
        known.update(pid for pid in members if pid != me)
        alive: List[int] = []
        undetermined: List[int] = []
        for pid in sorted(known):
            if pid == me or pid_is_zombie(pid):
                continue  # exited; only its parent's `wait()` frees the table slot
            state = pid_marker_state(pid, token)
            if state == MARKER_MEMBER:
                alive.append(pid)
                if kill:
                    # NOTHING stands between the revalidation above and this signal: any lookup
                    # in that gap is a window for the pid to be recycled onto a stranger.
                    _pl.force_kill_pid(pid)
            elif pgid and _pl.process_group_id(pid) == pgid:
                # The kernel still places it in the container's own group: alive, and a member
                # however its environment reads. Reported, never signalled — a pgid is a borrowed
                # name, so this pid is not proven ours the way a token-bearer is.
                alive.append(pid)
            elif state == MARKER_UNREADABLE:
                undetermined.append(pid)
        return alive, undetermined, ""

    def reap(self) -> str:
        """SCAN the container; return "" only when nothing of it is left, else why not. Detection,
        not guaranteed teardown. ONE bounded best-effort kill sweep runs first; after it nothing is
        ever signalled again, so a member is targeted at most once and the unavoidable
        exit-then-reuse race is entered once rather than every 50ms for ten seconds. The rest is
        rescanning. A member still alive, or one whose liveness could not be DETERMINED (unreadable
        environment, unenumerable table, a Windows job that will not confirm its own termination),
        fails naming the pids — "cannot tell" is not "gone". Handles are consumed once."""
        if _pl.IS_WINDOWS:
            if self._setup_error:
                return self._setup_error
            job, self._job = self._job, None
            if job is None:
                return ""
            # Teardown happens HERE, where a failure can still reach the verdict. Terminate AND
            # close — kill-on-close backstops a termination that did not take.
            return "; ".join(text for text in (_pl.terminate_job(job), _pl.close_job(job)) if text)
        token, self._token = self._token, ""
        root, self._root = self._root, 0
        pgid, self._pgid = self._pgid, 0
        since, self._start_ticks = self._start_ticks, 0
        if self._setup_error or not token:
            return self._setup_error
        # Seeded with the spawned root: a member by construction, not by having been read.
        known: "set[int]" = {root} if root > 0 else set()
        deadline = time.monotonic() + _REAP_DEADLINE_SEC
        alive, undetermined, error = self._scan(token, pgid, known, True, since)  # the ONE sweep
        last, quiet = (alive, undetermined), 0
        while not error:
            if alive or undetermined:
                quiet, last = 0, (alive, undetermined)
            else:
                quiet += 1
            if quiet >= _REAP_QUIET_SCANS:
                return ""
            if time.monotonic() >= deadline:
                # The last NON-EMPTY scan, not the current one: a scan coming back empty just as the
                # deadline passes would name no pid, under a remediation that says to kill them.
                return _containment_leak_reason(*last)
            time.sleep(_REAP_SETTLE_SEC)
            alive, undetermined, error = self._scan(token, pgid, known, False, since)
        return error

    def close(self) -> None:
        """Release the container handle. Inert after ``reap``, which closes it itself."""
        if _pl.IS_WINDOWS and self._job is not None:
            _pl.close_job(self._job)
            self._job = None

