"""Physical admission budget shared by the test entrypoints and nested runners.

This bounds initial fan-out, not later allocation: the process owner separately
watches its tree for overload. The tree is admitted within the room above the host
reserve (``host_memory.physical_memory``); a failed probe admits one worker.

Honest workers (owner 10.10, 2026-11-05): a run admits the CPU default (or the
documented request) within measured memory, minus what the OTHER live test runs
on this host hold: the slots their owners leased in the host ledger (this user's
own 0700 directory ``<default_temp_parent()>/ouroboros-test-slots-<uid>.noindex``,
one ``<pid>.json`` row each: pid, birth token, admitted workers; ``SlotLease``)
and one conservative slot per unleased ``pytest``/``node --test`` root the
process census sees. ``reserve`` reads the ledger, admits and publishes the row
under the ledger's one lock, so two runs started at once never both take the
whole machine. Floor one; a run never waits for a free slot. It waits only while
the macOS kernel confirms critical memory pressure (owner choice, 2026-11-14):
``reserve`` then holds neither the lock nor a row, says why with the reading's
date, re-reads the machine every ``config.TEST_PROCESS_SAMPLE_SEC`` and reserves
at the first reading that is not critical; the caller's interrupt (Ctrl-C, a pooled
task's Stop: its worker's kill) or its ``stop`` reader (an in-process turn's controls)
ends the wait, and an unread verdict admits as before, said as unknown.
Each ``scripts/run_tests.py`` pass reserves on its own, so a battery may wait at a
pass boundary, never inside a running pass; the hermetic gate reserves once, before
its first lane. A child with a parent allocation reads neither and never waits:
its parent accounted for its slots, and its run is work already running.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Mapping, Optional

from ouroboros.host_memory import physical_memory
from ouroboros.utils import utc_now_iso

# Transport state must survive the OUROBOROS_ runtime-settings scrub, just like
# ProcessContainer's membership token. A test explicitly allocates extra slots
# to a child by supplying this key in that child's environment.
NESTED_WORKERS_ENV = "OURO_TEST_WORKER_BUDGET"
WORKERS_ENV = "OUROBOROS_PREFLIGHT_TEST_WORKERS"


def _positive(value: object, default: int) -> int:
    try:
        return max(1, int(str(value).strip()))
    except ValueError:
        return default


def nested_workers(env: Mapping[str, str]) -> int:
    """A child has one slot unless its test explicitly allocates more."""
    return _positive(env.get(NESTED_WORKERS_ENV, "1"), 1)


def requested_workers(cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None) -> int:
    """The deterministic request: the override or CPU default within parent slots."""
    env = os.environ if env is None else env
    requested = _positive(env.get(WORKERS_ENV, ""), cpu_workers or os.cpu_count() or 1)
    if NESTED_WORKERS_ENV in env:
        requested = min(requested, nested_workers(env))
    return max(1, requested)


@dataclass(frozen=True)
class HostRuns:
    """What the other live test runs on this host hold: leased slots and the roots nobody leased."""

    slots: int = 0
    leases: tuple = ()      # (owner pid, command) per live lease outside this process's lineage
    strangers: tuple = ()   # (pid, command) per unleased pytest/node root, one slot each
    note: str = ""          # why the census is partial, when it is

    @property
    def held(self) -> int:
        return self.slots + len(self.strangers)

    def disclosure(self) -> str:
        """One line naming what was subtracted, empty when nothing was."""
        parts = []
        if self.leases:
            parts.append(f"other test runs hold {self.slots} slot(s): "
                         + ", ".join(f"{pid} {command}".rstrip() for pid, command in self.leases))
        if self.strangers:
            parts.append(f"{len(self.strangers)} unleased pytest/node root(s) counted one slot each: "
                         + ", ".join(f"{pid} {command}".rstrip() for pid, command in self.strangers))
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)


@dataclass(frozen=True)
class Admission:
    workers: int
    host: HostRuns
    critical_at: str = ""  # the reading's date when the kernel confirmed critical pressure: no run starts on it
    waited: tuple = ()     # (first critical reading, the reading that ended it, how: ``WAIT_ENDED``) when one preceded


def admission(cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None) -> Admission:
    """The workers this run may start: the request within physical room, minus what other runs hold.

    An ``env`` carrying a parent allocation (``NESTED_WORKERS_ENV``) is a child whose slots its parent
    already accounted for on the host, so it reads no ledger, runs no census and is never held for pressure.
    ``workers`` keeps its floor of one under critical pressure too: ``critical_at`` says no run starts now."""
    from ouroboros.config import TEST_WORKER_PEAK_RSS_BYTES

    env = os.environ if env is None else env
    memory = physical_memory()
    capacity = max(1, memory.room_bytes // TEST_WORKER_PEAK_RSS_BYTES) if memory else 1
    own = min(requested_workers(cpu_workers, env=env), capacity)
    host = HostRuns() if NESTED_WORKERS_ENV in env else other_runs()
    critical = memory is not None and memory.pressure is True and NESTED_WORKERS_ENV not in env
    return Admission(max(1, own - host.held), host, utc_now_iso() if critical else "")


# ── The host ledger and the census ──────────────────────────────────────────────────────
SLOT_LEDGER_DIR = "ouroboros-test-slots-{user}.noindex"  # one directory per user, 0700 (`_owned_ledger`)
_TEST_EXECUTABLES = frozenset({"pytest", "py.test", "pytest.exe"})
_NODE_EXECUTABLES = frozenset({"node", "node.exe"})


def _user_tag() -> str:
    """This user in the ledger directory's name: the uid, or the login where there is none (Windows)."""
    if hasattr(os, "getuid"):
        return str(os.getuid())
    import getpass

    return re.sub(r"[^A-Za-z0-9_.-]", "_", getpass.getuser() or "user")


def _ledger_dir() -> Path:
    from ouroboros.test_environment import default_temp_parent

    return default_temp_parent() / SLOT_LEDGER_DIR.format(user=_user_tag())


def _owned_ledger() -> Optional[Path]:
    """The ledger directory as this user's own, 0700: None when the path is a link, a file or another user's
    (a planted entry in a shared temp parent), so no row is read or written through it; the census still
    counts that user's runs. A directory of ours left wider than 0700 is narrowed."""
    path = _ledger_dir()
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        facts = os.lstat(path)
        if not stat.S_ISDIR(facts.st_mode):
            return None
        if hasattr(os, "getuid"):
            if facts.st_uid != os.getuid():
                return None
            if facts.st_mode & 0o077:
                os.chmod(path, 0o700)
    except OSError:
        return None
    return path


@contextlib.contextmanager
def _ledger_lock(directory: Optional[Path]):
    """The ledger's one lock while a run reads it, admits and publishes its row: two runs started at once
    would otherwise both see an empty ledger and both take the whole machine. Held for the milliseconds of
    one census and one write, never while a run waits (owner 10.10: never for a slot; critical pressure: outside)."""
    if directory is None:
        yield
        return
    from ouroboros import platform_layer as pl

    try:
        descriptor = os.open(directory / "ledger.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        yield  # no lock to take: the row write below answers for itself
        return
    try:
        pl.file_lock_exclusive(descriptor)
        yield
    finally:
        try:
            pl.file_unlock(descriptor)
        except OSError:
            pass
        os.close(descriptor)


def reserve(command: str, cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None,
            slots=None, say: Optional[Callable[[str], None]] = None,
            tell: Optional[Callable[[str], None]] = None,
            stop: Optional[Callable[[], Optional[bool]]] = None) -> tuple[Admission, "SlotLease"]:
    """Admit this run's workers and publish its row as ONE reservation under the ledger lock: no other run's
    census, admission or write comes between this run's. ``slots`` maps the admission to the count the row
    holds, within the admitted workers (default: all of them; a one-process pass holds one, a Node lane its
    concurrency). A child with a parent allocation reads no ledger and holds no row.

    An admission that confirms critical pressure starts nothing: the lock is released with no row written,
    ``say`` (default: stderr) tells the wait and its end with each reading's date, ``tell`` the same two moments
    in the owner's words (``OWNER_WAIT``), and the machine alone is re-read outside the lock until a reading is
    not critical; the reservation then admits as usual and its admission carries ``waited``. ``stop``, read only
    while it waits, ends the wait with ``WaitStopped`` when it answers True (an unknown ``None`` stops nothing):
    nothing was admitted, leased or started."""
    env = os.environ if env is None else env
    say = say or (lambda text: print(text, file=sys.stderr, flush=True))
    directory = None if NESTED_WORKERS_ENV in env else _owned_ledger()
    since, ended = "", ()
    while True:
        with _ledger_lock(directory):
            admitted = admission(cpu_workers, env=env)
            if not admitted.critical_at:
                admitted = replace(admitted, waited=(since, *ended)) if since else admitted
                held = admitted.workers if slots is None else max(1, min(int(slots(admitted)), admitted.workers))
                return admitted, SlotLease(held, command, locked=True)
        since = since or admitted.critical_at
        say(f"waiting to start: the kernel reports critical memory pressure (read {admitted.critical_at}); work "
            "already running continues; this run starts when a reading is no longer critical; Ctrl-C or Stop ends "
            "the wait")
        if tell:
            tell(OWNER_WAIT["waiting"])
        try:
            ended = _until_not_critical(stop)
        except WaitStopped as stopped:
            say(str(stopped))
            raise
        say(WAIT_ENDED[ended[1]].format(at=ended[0], since=since))
        if tell:
            tell(OWNER_WAIT[ended[1]])


# How a wait ended, said as such: only a read verdict is "no longer critical"; an unknown admits as it always did.
WAIT_ENDED = {
    "not_critical": "memory pressure no longer critical (read {at}); starting after waiting since {since}",
    "pressure_unknown": "critical memory pressure no longer confirmed: the kernel's level is unreadable (read {at}), "
                        "an unknown that admits as before; starting after waiting since {since}",
    "memory_unknown": "critical memory pressure no longer confirmed: the machine's memory is unreadable (read {at}), "
                      "an unknown that admits as before; starting after waiting since {since}",
}


# The same moments on an owner's progress line: what happens and what they can do; the readings stay in ``say``.
OWNER_WAIT = {
    "waiting": "waiting for memory: the computer is critically short of it. Running work continues; the tests "
               "start by themselves once memory recovers. Stop ends the wait",
    "not_critical": "starting: memory recovered",
    "pressure_unknown": "starting as usual: memory pressure can no longer be read",
    "memory_unknown": "starting as usual: memory can no longer be read",
}


class WaitStopped(Exception):
    """A ``stop`` reader ended a wait for critical memory pressure: nothing was admitted, leased or started."""


def _until_not_critical(stop: Optional[Callable[[], Optional[bool]]] = None) -> tuple:
    """Sleep a sample interval and re-read the machine until a reading is not critical: ``(its date, how)``, how a
    ``WAIT_ENDED`` key (a read verdict, or an unknown one). No lock, row or child is held meanwhile. ``stop`` is
    read once a sample; True raises ``WaitStopped``."""
    from ouroboros import config

    while True:
        time.sleep(config.TEST_PROCESS_SAMPLE_SEC)
        if stop is not None and stop() is True:
            raise WaitStopped(f"a stop of this work ended the wait for memory (read {utc_now_iso()}); nothing started")
        memory = physical_memory()
        if memory is None or memory.pressure is not True:
            return utc_now_iso(), ("memory_unknown" if memory is None else
                                   "pressure_unknown" if memory.pressure is None else "not_critical")


def _read_lease(path: Path) -> Optional[dict]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return {"pid": int(row["pid"]), "birth": str(row["birth"]), "workers": max(0, int(row["workers"])),
                "command": str(row.get("command", ""))}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _lease_live(row: dict) -> bool:
    """The row's pid is alive and is the process that wrote it, by its measured birth."""
    from ouroboros import platform_layer as pl

    return pl.pid_is_alive(row["pid"]) and pl.process_start_time(row["pid"]) == row["birth"]


class SlotLease:
    """This run's ledger row while it runs: the owner's pid, birth token and admitted workers.

    The owner that admits workers writes it (scripts/run_tests.py per pass, the hermetic gate per run)
    and removes it when the run ends; a lease covers the owner's whole process tree, and holds what the
    pass really uses: the admitted workers for an xdist pass, one for a one-process pass. A row whose pid is
    dead or reborn is ignored by every reader and unlinked by the next writer, so nothing piles up: one
    row of about a hundred bytes per live owner. A child with a parent allocation holds no row (its
    parent's lease covers it), and a ledger the host refuses to write, or one that is not this user's
    own directory (``_owned_ledger``), costs nobody a slot: the census still sees the run as one
    unleased root.
    """

    def __init__(self, workers: int, command: str = "", *, locked: bool = False):
        """``locked``: the caller holds the ledger lock already (``reserve``); otherwise the write takes it."""
        self.path: Optional[Path] = None
        self.workers, self.command = int(workers), command
        if NESTED_WORKERS_ENV in os.environ:
            return
        directory = _owned_ledger()
        if directory is None:
            return
        if locked:
            self._publish(directory)
        else:
            with _ledger_lock(directory):
                self._publish(directory)

    def _publish(self, directory: Path) -> None:
        """Sweep the dead rows and write ours: a fresh staging file opened exclusively without following a
        link (nothing planted under our name is truncated), then renamed over the row (a planted link there
        is replaced, not followed)."""
        import uuid

        from ouroboros import platform_layer as pl

        path = directory / f"{os.getpid()}.json"
        row = {"pid": os.getpid(), "birth": pl.process_start_time(os.getpid()), "workers": self.workers,
               "command": self.command}
        try:
            for stale in directory.glob("*.json"):
                if stale != path and ((old := _read_lease(stale)) is None or not _lease_live(old)):
                    stale.unlink(missing_ok=True)
            staging = directory / f"{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(row))
            os.replace(staging, path)
        except OSError:
            return
        self.path = path

    def hold(self, workers: int) -> None:
        """Narrow this row to ``workers`` slots at a pass boundary (the one-process pass after the xdist pass
        holds one): rewritten in place under the lock, never widened, which would take a new admission."""
        workers = max(1, int(workers))
        if workers >= self.workers:
            return
        self.workers = workers
        if self.path is not None:
            with _ledger_lock(self.path.parent):
                self._publish(self.path.parent)

    def close(self) -> None:
        if self.path is not None:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.path = None

    def __enter__(self) -> "SlotLease":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _process_table():
    """``([(pid, ppid, argv)], "")`` for every process, or ``(None, why)`` when the table cannot be read."""
    from ouroboros import platform_layer as pl

    rows = []
    try:
        if pl.IS_WINDOWS:
            import psutil  # The packaged Windows runtime includes psutil.
            for process in psutil.process_iter(["pid", "ppid", "cmdline"]):
                rows.append((int(process.info["pid"]), int(process.info.get("ppid") or 0),
                             list(process.info.get("cmdline") or [])))
        else:
            out = subprocess.run(["ps", "-ww", "-Ao", "pid=,ppid=,command="], capture_output=True,
                                 text=True, timeout=20, check=True)
            for line in out.stdout.splitlines():
                head = line.split(maxsplit=2)
                if len(head) >= 2 and head[0].isdigit() and head[1].isdigit():
                    rows.append((int(head[0]), int(head[1]), head[2].split() if len(head) == 3 else []))
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as error:
        return None, f"process census unavailable ({type(error).__name__}): unleased runs were not counted"
    return rows, ""


def _is_test_command(argv) -> bool:
    names = [os.path.basename(str(arg)).lower() for arg in argv]
    return any(name in _TEST_EXECUTABLES for name in names) or (
        bool(names) and names[0] in _NODE_EXECUTABLES and "--test" in argv)


def _ancestors(pid: int, parents: Mapping[int, int]) -> list:
    """Parents of ``pid`` from the nearest up, stopping at an unknown pid, pid 0 or a loop."""
    chain = []
    while (pid := parents.get(pid, 0)) > 0 and pid not in chain:
        chain.append(pid)
    return chain


def unleased_test_roots(table, me: int, leased) -> tuple:
    """Pytest/node roots (no test command above them) outside this process's own lineage and outside
    every live lease's tree: each is a run that admitted itself, counted as one slot."""
    parents = {pid: ppid for pid, ppid, _ in table}
    tests = {pid: argv for pid, _, argv in table if _is_test_command(argv)}
    mine = {me, *_ancestors(me, parents)}
    roots = []
    for pid in sorted(tests):
        above = _ancestors(pid, parents)
        if pid in mine or pid in leased or any(parent in tests or parent in leased for parent in above):
            continue
        roots.append((pid, " ".join(os.path.basename(str(arg)) for arg in tests[pid])[:80]))
    return tuple(roots)


def other_runs() -> HostRuns:
    """What the other live test runs hold on this host: the ledger's live rows and one process census."""
    me = os.getpid()
    table, note = _process_table()
    parents = {pid: ppid for pid, ppid, _ in table} if table else {}
    mine = {me, *_ancestors(me, parents)}
    slots, leases = 0, []
    directory = _owned_ledger()
    try:
        rows = sorted(directory.glob("*.json")) if directory is not None else []
    except OSError:
        rows = []
    for path in rows:
        row = _read_lease(path)
        if row is None or row["pid"] in mine or not _lease_live(row):
            continue
        slots += row["workers"]
        leases.append((row["pid"], row["command"]))
    strangers = unleased_test_roots(table, me, {pid for pid, _ in leases}) if table is not None else ()
    return HostRuns(slots, tuple(leases), strangers, note)


def _cap_requested(value, slots: int, *, pytest_count: bool = False) -> str:
    text = str(value)
    if pytest_count and text in {"auto", "logical"}:
        return str(slots)
    try:
        count = int(text)
    except ValueError:
        return text  # Preserve the native CLI's refusal of invalid input.
    return str(min(slots, count)) if count > 0 else text


def bounded_test_command(command, env: Mapping[str, str]):
    """Apply the child allocation to recognizable Node and pytest argv lists.

    Popen's shared test seam covers legacy helpers too. Shell strings remain
    untouched; their owners must use argv or a runner reading the budget key.
    """
    if not isinstance(command, (list, tuple)) or not command:
        return command
    args = list(command)
    executable = Path(os.fsdecode(args[0])).name.lower()
    slots = nested_workers(env)
    if executable in {"node", "node.exe"} and "--test" in args:
        option = "--test-concurrency"
        supplied = False
        for index, arg in enumerate(args):
            if arg == option:
                supplied = True
                if index + 1 < len(args):
                    args[index + 1] = _cap_requested(args[index + 1], slots)
            if isinstance(arg, str) and arg.startswith(option + "="):
                supplied = True
                args[index] = option + "=" + _cap_requested(arg.split("=", 1)[1], slots)
        if not supplied:
            args.insert(args.index("--test") + 1, f"{option}={slots}")
    elif (executable.startswith("pytest") or
          any(args[i:i + 2] == ["-m", "pytest"] for i in range(len(args) - 1))):
        for index, arg in enumerate(args):
            if arg in {"-n", "--numprocesses"} and index + 1 < len(args):
                args[index + 1] = _cap_requested(args[index + 1], slots, pytest_count=True)
            elif isinstance(arg, str) and (arg.startswith("--numprocesses=") or
                                          arg.startswith("-n") and len(arg) > 2):
                value = arg.split("=", 1)[1] if "=" in arg else arg[2:]
                prefix = "--numprocesses=" if arg.startswith("--") else "-n"
                args[index] = prefix + _cap_requested(value, slots, pytest_count=True)
    return args
