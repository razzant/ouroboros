"""Physical admission budget shared by the test entrypoints and nested runners.

This bounds initial fan-out, not later allocation: the process owner separately
watches its tree for overload. A failed memory probe admits one worker.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def worker_budget(cpu_workers: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None) -> int:
    """Keep the caller's CPU default, capped by physical room and parent slots."""
    from ouroboros.config import TEST_WORKER_PEAK_RSS_BYTES

    memory = physical_memory()
    capacity = max(1, memory_budget(memory) // TEST_WORKER_PEAK_RSS_BYTES) if memory else 1
    return max(1, min(requested_workers(cpu_workers, env=env), capacity))


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
