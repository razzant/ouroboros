"""Physical admission budget shared by the test entrypoints and nested runners.

This bounds initial fan-out, not later allocation: the process owner separately
watches its tree for overload. A failed memory probe admits one worker.

Honest workers (owner 10.10, 2026-11-05): a run admits the CPU default (or the
documented request) within measured memory, minus what the OTHER live test runs
on this host hold: the slots their owners leased in the host ledger
(``<default_temp_parent()>/ouroboros-test-slots.noindex/<pid>.json``: pid, birth
token, admitted workers; ``SlotLease``) and one conservative slot per unleased
``pytest``/``node --test`` root the process census sees. Floor one; a run never
waits for a free slot. A child with a parent allocation reads neither: its
parent accounted for its slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Optional, Sequence

# Transport state must survive the OUROBOROS_ runtime-settings scrub, just like
# ProcessContainer's membership token. A test explicitly allocates extra slots
# to a child by supplying this key in that child's environment.
NESTED_WORKERS_ENV = "OURO_TEST_WORKER_BUDGET"
WORKERS_ENV = "OUROBOROS_PREFLIGHT_TEST_WORKERS"


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    pressure: bool = False  # the kernel itself reports memory pressure (macOS)


def _native_output(argv: Sequence[str]) -> str:
    # Bounded, same-call OS probes; no test process or daemon is started here.
    return subprocess.run(argv, check=True, capture_output=True, text=True,
                          encoding="utf-8", timeout=2).stdout


def _darwin_pressure() -> bool:
    """The kernel's own critical verdict (levels: 1 normal, 2 warn, 4 critical). Free + inactive
    + speculative barely moves under pressure: XNU keeps refilling the inactive queue with
    anonymous pages that it then compresses. Warn is routine on a busy 8 GB Mac running a
    healthy battery, so only critical counts. An unreadable level reads as normal, silently;
    the free-memory reading still applies."""
    try:
        return int(_native_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()) >= 4
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def physical_memory() -> Optional[MemorySnapshot]:
    """Read total and reclaimable physical memory using native OS facilities."""
    pressure = False
    try:
        if sys.platform == "win32":
            import psutil  # The packaged Windows runtime includes psutil.
            memory = psutil.virtual_memory()
            total, available = memory.total, memory.available
        elif sys.platform == "darwin":
            total = int(_native_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip())
            vm = _native_output(["/usr/bin/vm_stat"])
            page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
            pages = {name.strip(): int(value.strip().rstrip("."))
                     for name, value in (line.split(":", 1) for line in vm.splitlines()[1:] if ":" in line)}
            # Admission counts inactive pages as room, anonymous ones included, which
            # overstates it under pressure; the overload stop also reads the kernel's
            # verdict. Purgeable is already inside active/inactive, so it is not added.
            available = page_size * sum(pages[name] for name in
                                        ("Pages free", "Pages inactive", "Pages speculative"))
            pressure = _darwin_pressure()
        elif sys.platform.startswith("linux"):
            fields = {name: int(value.split()[0]) * 1024 for name, value in
                      (line.split(":", 1) for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines())}
            total, available = fields["MemTotal"], fields["MemAvailable"]
            # Inside a container, its cgroup v2 limit is the physical room, not the host's RAM;
            # reclaimable file cache counts as room, as MemAvailable counts it on the host.
            limit = Path("/sys/fs/cgroup/memory.max")
            if limit.is_file() and (cap := limit.read_text(encoding="ascii").strip()) != "max":
                used = int(Path("/sys/fs/cgroup/memory.current").read_text(encoding="ascii"))
                try:
                    stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text(
                        encoding="ascii").splitlines() if len(line.split()) == 2)
                except OSError:
                    stat = {}
                used = max(0, used - int(stat.get("inactive_file", 0)))
                total, available = min(total, int(cap)), min(available, max(0, int(cap) - used))
        else:
            return None
        if total <= 0 or available < 0:
            return None
        return MemorySnapshot(total, min(total, available), pressure)
    except (OSError, ValueError, KeyError, AttributeError, ImportError, subprocess.SubprocessError):
        return None


def memory_budget(memory: MemorySnapshot) -> int:
    """Leave the configured physical reserve outside the admitted test tree."""
    from ouroboros.config import TEST_MEMORY_RESERVE_FRACTION

    return max(0, memory.available_bytes - int(memory.total_bytes * TEST_MEMORY_RESERVE_FRACTION))


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


def admission(cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None) -> Admission:
    """The workers this run may start: the request within physical room, minus what other runs hold.

    An ``env`` carrying a parent allocation (``NESTED_WORKERS_ENV``) is a child whose slots its parent
    already accounted for on the host, so it reads no ledger and runs no census."""
    from ouroboros.config import TEST_WORKER_PEAK_RSS_BYTES

    env = os.environ if env is None else env
    memory = physical_memory()
    capacity = max(1, memory_budget(memory) // TEST_WORKER_PEAK_RSS_BYTES) if memory else 1
    own = min(requested_workers(cpu_workers, env=env), capacity)
    host = HostRuns() if NESTED_WORKERS_ENV in env else other_runs()
    return Admission(max(1, own - host.held), host)


def worker_budget(cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None) -> int:
    """Keep the caller's CPU default, capped by physical room, parent slots and the other runs' slots."""
    return admission(cpu_workers, env=env).workers


# ── Serial groups of the one xdist pass ─────────────────────────────────────────────────
_MAX_SERIAL_SHARDS = 4


def serial_share(table, system: str):
    """The serial part of ``system``'s recorded test time, or None when the table does not say."""
    total = sum(((table or {}).get("platforms") or {}).get(system, {}).values())
    serial = ((table or {}).get("serial") or {}).get(system)
    return sum(serial.values()) / total if total and serial is not None else None


def serial_shards(workers: int, table=None, system: str = "") -> int:
    """How many groups of bare-`serial` files may run at once in the one xdist pass (tests/conftest.py).

    With recorded durations that split this system's test time (``serial``, tests/ci_verdicts.py), the serial
    files get the share of the workers their share of the time needs: at least one group, at most half the
    workers. Without them, one group per four workers, at most four. Either way three workers or fewer keep
    ONE group, i.e. serial files never overlap each other there; they only stop waiting for the other tests."""
    share = serial_share(table, system or _system_name())
    if share is None:
        return max(1, min(_MAX_SERIAL_SHARDS, workers // 4))
    return max(1, min(workers // 2, round(workers * share)))


def _system_name() -> str:
    value = sys.platform.lower()
    return "linux" if value.startswith("linux") else "windows" if value.startswith(("win", "cygwin")) else value


# ── The host ledger and the census ──────────────────────────────────────────────────────
SLOT_LEDGER_DIR = "ouroboros-test-slots.noindex"
_TEST_EXECUTABLES = frozenset({"pytest", "py.test", "pytest.exe"})
_NODE_EXECUTABLES = frozenset({"node", "node.exe"})


def _ledger_dir() -> Path:
    from ouroboros.test_environment import default_temp_parent

    return default_temp_parent() / SLOT_LEDGER_DIR


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
    and removes it when the run ends; a lease covers the owner's whole process tree. A row whose pid is
    dead or reborn is ignored by every reader and unlinked by the next writer, so nothing piles up: one
    row of about a hundred bytes per live owner. A child with a parent allocation holds no row (its
    parent's lease covers it), and a ledger the host refuses to write costs nobody a slot: the census
    still sees the run as one unleased root.
    """

    def __init__(self, workers: int, command: str = ""):
        from ouroboros import platform_layer as pl

        self.path: Optional[Path] = None
        if NESTED_WORKERS_ENV in os.environ:
            return
        path = _ledger_dir() / f"{os.getpid()}.json"
        row = {"pid": os.getpid(), "birth": pl.process_start_time(os.getpid()), "workers": int(workers),
               "command": command}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            for stale in path.parent.glob("*.json"):
                if stale != path and ((old := _read_lease(stale)) is None or not _lease_live(old)):
                    stale.unlink(missing_ok=True)
            staging = path.with_name(path.name + ".tmp")
            staging.write_text(json.dumps(row), encoding="utf-8")
            os.replace(staging, path)
        except OSError:
            return
        self.path = path

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
    try:
        rows = sorted(_ledger_dir().glob("*.json"))
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
