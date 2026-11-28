"""This host's physical memory: the one reader test admission and the overload stop share."""
from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import host_memory
from ouroboros.host_memory import MemorySnapshot

GIB = 1024 ** 3
ROOT_MOUNT = "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"
V2_MOUNT = "30 23 0:26 {root} {point} rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate"
V1_MEMORY = "35 30 0:31 {root} /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime master:16 - cgroup cgroup rw,memory"
V1_CPU = "34 30 0:30 {root} /sys/fs/cgroup/cpu,cpuacct ro,nosuid,nodev,noexec,relatime master:15 - cgroup cgroup rw,cpu,cpuacct"
HOST = MemorySnapshot(32 * GIB, 16 * GIB)  # the fake host's /proc/meminfo
# One `vm_stat` on an Apple Silicon Mac (16 KiB pages, 128 GiB, kernel level 1), 2026-11-27T21:48:17Z, verbatim.
VM_STAT_16K = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             1620420.
Pages active:                           2869396.
Pages inactive:                         2454324.
Pages speculative:                       412516.
Pages throttled:                              0.
Pages wired down:                        402243.
Pages purgeable:                          34142.
"Translation faults":               26494222798.
Pages copy-on-write:                 2480366299.
Pages zero filled:                   8867459886.
Pages reactivated:                   1317822813.
Pages purged:                         205069859.
File-backed pages:                      1968917.
Anonymous pages:                        3767319.
Pages stored in compressor:             1135366.
Pages occupied by compressor:            523644.
Decompressions:                       853873871.
Compressions:                        1244157417.
Pageins:                              547312593.
Pageouts:                               5090585.
Swapins:                                 729800.
Swapouts:                               2020129.
"""
# A 4 KiB-page host (Intel Macs): vm_stat.c's labels, in its order and its "%-30s %16llu." lines. The counts are an
# illustrative 8 GiB host, not a capture (no Intel Mac was at hand): little file cache beside many inactive pages.
VM_STAT_4K = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n" + "".join(
    f"{label + ':':<30} {count:>16}.\n" for label, count in (
        ("Pages free", 13000), ("Pages active", 640000), ("Pages inactive", 610000), ("Pages speculative", 9000),
        ("Pages throttled", 0), ("Pages wired down", 380000), ("Pages purgeable", 12000),
        ('"Translation faults"', 912345678), ("Pages copy-on-write", 40123456), ("Pages zero filled", 400123456),
        ("Pages reactivated", 30123456), ("Pages purged", 2345678), ("File-backed pages", 150000),
        ("Anonymous pages", 1109000), ("Pages stored in compressor", 1100000),
        ("Pages occupied by compressor", 420000), ("Decompressions", 60123456), ("Compressions", 90123456),
        ("Pageins", 20123456), ("Pageouts", 345678), ("Swapins", 1234567), ("Swapouts", 2345678)))
RATES = ("pageins", "swapins", "swapouts", "compressions", "decompressions")


def _write(root, files):
    """Each relative path's text, replacing a file or directory there; ``None`` removes it."""
    for relative, text in files.items():
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path)
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="ascii")


def _v2(directory, limit, current, inactive=0):
    """A v2 cgroup's limit, usage and stat files."""
    return {f"{directory}/memory.max": f"{limit}\n", f"{directory}/memory.current": f"{current}\n",
            f"{directory}/memory.stat": f"anon {current - inactive}\ninactive_file {inactive}\n"}


@pytest.fixture
def linux(tmp_path, monkeypatch):
    """A fake Linux root the probe reads: a 32 GiB host with 16 GiB available, seen from ``cgroup`` and ``mounts``."""
    root = tmp_path / "root"
    monkeypatch.setattr(host_memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(host_memory, "_LINUX_ROOT", root)

    def make(cgroup, *mounts, **files):
        _write(root, {"proc/meminfo": f"MemTotal: {32 * GIB // 1024} kB\nMemAvailable: {16 * GIB // 1024} kB\n",
                      "proc/self/cgroup": cgroup,
                      "proc/self/mountinfo": "".join(line + "\n" for line in (ROOT_MOUNT, *mounts))})
        _write(root, files)
        return lambda **more: _write(root, more)

    return make


def test_macos_reads_reclaimable_pages_and_only_the_kernels_critical_level(monkeypatch, request):
    """XNU keeps refilling the inactive queue with anonymous pages it then compresses, so free + inactive +
    speculative barely moves under pressure: the kernel's critical level is read too. Warn is routine on a busy
    small Mac running a healthy battery. An unreadable level is unknown (None), never normal, and leaves the
    free-memory reading alone; purgeable is already inside active/inactive, so it is not added; an unreadable
    page table is unknown."""
    vm = ("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
          "Pages free: 2.\nPages inactive: 3.\nPages speculative: 1.\nPages purgeable: 2.\n")
    level = None
    calls = []
    host_memory._darwin_total.cache_clear()
    request.addfinalizer(host_memory._darwin_total.cache_clear)

    def native(argv):
        calls.append(argv[-1])
        if argv[-1] == "kern.memorystatus_vm_pressure_level":
            if level is None:
                raise subprocess.CalledProcessError(1, argv)
            return level
        return "65536\n" if argv[-1] == "hw.memsize" else vm

    monkeypatch.setattr(host_memory, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host_memory, "_native_output", native)
    for level, pressure in (("1\n", False), ("2\n", False), ("4\n", True), ("garbled\n", None), (None, None)):
        assert host_memory.physical_memory() == MemorySnapshot(65536, 24576, pressure), level
    assert calls.count("hw.memsize") == 1
    assert calls.count("/usr/bin/vm_stat") == calls.count("kern.memorystatus_vm_pressure_level") == 5
    vm = vm.replace("Pages free: 2.", "Pages free: 4.")
    assert host_memory.physical_memory() == MemorySnapshot(65536, 32768, None)  # the level is still unreadable
    vm = "unreadable"
    assert host_memory.physical_memory() is None


@pytest.mark.parametrize("text, total, available, figures", [
    (VM_STAT_16K, 128 * GIB, 16384 * (1620420 + 2454324 + 412516),
     (16384 * (1620420 + 412516 + 1968917), 16384 * 1968917, 16384 * 523644, 16384 * 1135366)),
    (VM_STAT_4K, 8 * GIB, 4096 * (13000 + 610000 + 9000),
     (4096 * (13000 + 9000 + 150000), 4096 * 150000, 4096 * 420000, 4096 * 1100000)),
], ids=["apple-silicon-16k-capture", "4k-pages"])
def test_macos_keeps_measurement_figures_from_the_same_two_reads(monkeypatch, request, text, total, available,
                                                                 figures):
    """Issue #3397 is decided on numbers: the vm_stat and level reads the reading already makes also give, for
    measurement only, the file-bounded upper estimate beside today's available (free + speculative + the smaller
    of inactive and file-backed pages), file-backed pages, the compressor's occupancy and the level in words.
    Nothing more is run. A line this host does not print, or an unreadable level, is unknown, never 0, and
    today's reading stays what it was."""
    answers = {"hw.memsize": f"{total}\n", "kern.memorystatus_vm_pressure_level": "2\n", "/usr/bin/vm_stat": text}
    calls = []
    host_memory._darwin_total.cache_clear()
    request.addfinalizer(host_memory._darwin_total.cache_clear)

    def native(argv):
        calls.append(argv[-1])
        if answers[argv[-1]] is None:
            raise subprocess.CalledProcessError(1, argv)
        return answers[argv[-1]]

    monkeypatch.setattr(host_memory, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host_memory, "_native_output", native)
    memory = host_memory.physical_memory()
    assert memory == MemorySnapshot(total, available, False), "today's reading: free + inactive + speculative"
    assert calls == ["hw.memsize", "/usr/bin/vm_stat", "kern.memorystatus_vm_pressure_level"], "no extra probe"
    assert memory.vm.measurement() == {
        "pressure_level": "warn", "file_bounded_upper_bytes": figures[0], "file_backed_bytes": figures[1],
        "compressor_occupied_bytes": figures[2], "compressor_stored_bytes": figures[3],
        "rate_interval_sec": None, "rates_bytes_per_sec": dict.fromkeys(RATES)}
    answers["/usr/bin/vm_stat"] = "".join(line + "\n" for line in text.splitlines() if not line.startswith((
        "File-backed", "Pages stored", "Pages occupied", "Swapins")))
    answers["kern.memorystatus_vm_pressure_level"] = None
    older = host_memory.physical_memory()
    assert older == MemorySnapshot(total, available, None)
    unknown = older.vm.measurement(memory.vm)
    assert (unknown["pressure_level"], unknown["rates_bytes_per_sec"]["swapins"]) == ("unknown", None)
    assert [unknown[key] for key in ("file_bounded_upper_bytes", "file_backed_bytes", "compressor_occupied_bytes",
                                     "compressor_stored_bytes")] == [None] * 4


def test_rates_are_per_second_between_two_readings_and_a_counter_that_went_back_has_none():
    """The paging counters are cumulative since boot: a rate is the bytes one moved per second of monotonic time
    between two readings of one process. A counter that went down (a reset or wrap) or that one side does not
    print has no rate this interval, never a negative or a huge one, while an unmoved one is a measured 0; no
    earlier reading, or one stamped later, gives no interval. With fewer inactive than file-backed pages the
    upper estimate takes the inactive ones."""
    def vm(at, **counters):
        pages = {"Pages free": 1, "Pages speculative": 2, "Pages inactive": 3, "File-backed pages": 5, **counters}
        return host_memory.DarwinVM(4096, pages, 4, at)

    before = vm(100.0, Pageins=100, Swapins=50, Swapouts=7, Compressions=1000, Decompressions=900)
    after = vm(110.0, Pageins=120, Swapins=40, Compressions=1000, Decompressions=950)
    figures = after.measurement(before)
    assert (figures["pressure_level"], figures["file_bounded_upper_bytes"], figures["rate_interval_sec"]) == (
        "critical", 4096 * (1 + 2 + 3), 10.0)
    assert figures["rates_bytes_per_sec"] == {
        "pageins": 2 * 4096, "swapins": None, "swapouts": None, "compressions": 0, "decompressions": 5 * 4096}
    for earlier in (None, after):
        figures = before.measurement(earlier)
        assert (figures["rate_interval_sec"], figures["rates_bytes_per_sec"]) == (None, dict.fromkeys(RATES))


def test_windows_reads_psutil_and_without_it_memory_is_unknown(monkeypatch):
    monkeypatch.setattr(host_memory, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=65536, available=24576)))
    assert host_memory.physical_memory() == MemorySnapshot(65536, 24576)
    assert host_memory.physical_memory().vm is None, "macOS's measurement means nothing here: absent"
    monkeypatch.setitem(sys.modules, "psutil", None)  # the import fails
    assert host_memory.physical_memory() is None


def test_linux_reads_the_host_until_a_cgroup_on_its_path_sets_a_limit(linux):
    """A systemd host: every cgroup on this process's path says ``max`` and the root has no limit file,
    so the host's own numbers hold; so do they on a kernel without cgroups, and with a hierarchy that
    has no memory controller. A limit set anywhere on the path bounds them."""
    edit = linux("1:name=systemd:/user.slice/user-1000.slice/session-2.scope\n"
                 "0::/user.slice/user-1000.slice/session-2.scope\n",
                 V2_MOUNT.format(root="/", point="/sys/fs/cgroup"),
                 **{f"sys/fs/cgroup/{path}/memory.max": "max\n" for path in (
                     "user.slice", "user.slice/user-1000.slice", "user.slice/user-1000.slice/session-2.scope")})
    assert host_memory.physical_memory() == HOST
    assert host_memory.physical_memory().vm is None, "macOS's measurement means nothing here: absent"
    edit(**{"proc/self/cgroup": None})
    assert host_memory.physical_memory() == HOST
    edit(**{"proc/self/cgroup": "0::/user.slice/user-1000.slice/session-2.scope\n"},
         **_v2("sys/fs/cgroup/user.slice", 8 * GIB, 7 * GIB, inactive=GIB))
    assert host_memory.physical_memory() == MemorySnapshot(8 * GIB, 2 * GIB)


def test_a_containers_v2_limit_bounds_memory_and_credits_its_reclaimable_cache(linux):
    """A 4 GiB container on a 32 GiB host, 3 GiB charged, 1 GiB of it inactive file cache."""
    edit = linux("0::/\n", V2_MOUNT.format(root="/", point="/sys/fs/cgroup"),
                 **_v2("sys/fs/cgroup", 4 * GIB, 3 * GIB, inactive=GIB))
    assert host_memory.physical_memory() == MemorySnapshot(4 * GIB, 2 * GIB)  # the cap, minus anon only
    edit(**{"sys/fs/cgroup/memory.stat": None})  # without the stat file the cache is not credited
    assert host_memory.physical_memory() == MemorySnapshot(4 * GIB, GIB)


def test_the_tightest_limit_on_a_nested_v2_path_bounds_memory(linux):
    """The mount's top sets nothing; a pod and its container below it each set a limit. Total is the smallest
    finite limit, available the least room among them, each bound independently, whichever level holds it."""
    edit = linux("0::/kubepods/pod1/ctr\n", V2_MOUNT.format(root="/", point="/sys/fs/cgroup"),
                 **{"sys/fs/cgroup/kubepods/memory.max": "max\n"},
                 **_v2("sys/fs/cgroup/kubepods/pod1", 8 * GIB, 3 * GIB),
                 **_v2("sys/fs/cgroup/kubepods/pod1/ctr", 2 * GIB, GIB))
    assert host_memory.physical_memory() == MemorySnapshot(2 * GIB, GIB)  # the deeper, tighter limit
    edit(**_v2("sys/fs/cgroup/kubepods/pod1/ctr", 16 * GIB, GIB))
    assert host_memory.physical_memory() == MemorySnapshot(8 * GIB, 5 * GIB)  # a looser one leaves the pod's
    edit(**_v2("sys/fs/cgroup/kubepods/pod1", 8 * GIB, 7 * GIB + GIB // 2),
         **_v2("sys/fs/cgroup/kubepods/pod1/ctr", 2 * GIB, GIB // 2))
    assert host_memory.physical_memory() == MemorySnapshot(2 * GIB, GIB // 2)  # the pod's room, the container's cap


def test_a_v1_memory_limit_bounds_memory_and_its_unlimited_spelling_does_not(linux):
    """Docker on cgroup v1: the memory hierarchy is mounted at the container's own cgroup. Usage less the
    hierarchical ``total_inactive_file`` is held; v1 spells "no limit" as LONG_MAX rounded down to the
    kernel page (LONG_MAX itself on old kernels), which bounds nothing and reads no usage."""
    edit = linux("12:cpu,cpuacct:/docker/abc\n9:memory:/docker/abc\n1:name=systemd:/docker/abc\n",
                 V1_CPU.format(root="/docker/abc"), V1_MEMORY.format(root="/docker/abc"),
                 **{"sys/fs/cgroup/memory/memory.limit_in_bytes": f"{2 * GIB}\n",
                    "sys/fs/cgroup/memory/memory.usage_in_bytes": f"{GIB + GIB // 2}\n",
                    "sys/fs/cgroup/memory/memory.stat": f"inactive_file 0\ntotal_inactive_file {GIB // 2}\n"})
    assert host_memory.physical_memory() == MemorySnapshot(2 * GIB, GIB)
    edit(**{"sys/fs/cgroup/memory/memory.usage_in_bytes": None})
    for unlimited in (2 ** 63 - 4096, 2 ** 63 - 65536, 2 ** 63 - 1):
        edit(**{"sys/fs/cgroup/memory/memory.limit_in_bytes": f"{unlimited}\n"})
        assert host_memory.physical_memory() == HOST, unlimited


def test_a_limit_that_exists_but_cannot_be_read_is_unknown(linux):
    """An absent limit file sets no limit; one present but unreadable, unparseable, or without a readable
    usage leaves the room unknown, so the whole reading is unknown, never the host's numbers."""
    edit = linux("0::/\n", V2_MOUNT.format(root="/", point="/sys/fs/cgroup"))
    assert host_memory.physical_memory() == HOST
    edit(**{"sys/fs/cgroup/memory.max": "lots\n"})
    assert host_memory.physical_memory() is None
    edit(**{"sys/fs/cgroup/memory.max": None})
    (host_memory._LINUX_ROOT / "sys/fs/cgroup/memory.max").mkdir()  # exists, cannot be read as a file
    assert host_memory.physical_memory() is None
    edit(**{"sys/fs/cgroup/memory.max": f"{4 * GIB}\n"})
    assert host_memory.physical_memory() is None  # a finite limit without its usage
    edit(**{"sys/fs/cgroup/memory.current": f"{GIB}\n"})
    assert host_memory.physical_memory() == MemorySnapshot(4 * GIB, 3 * GIB)


def test_a_cgroup_its_mount_does_not_show_is_unknown(linux):
    """This process's cgroup must lie below a mount of its hierarchy (escaped mount points read as the
    kernel wrote them); one outside every such mount cannot have its limits read. A hierarchy that is not
    mounted at all shows no limit."""
    escaped = V2_MOUNT.format(root="/kubepods/pod1", point="/run/cgroup\\040view")
    edit = linux("0::/kubepods/pod1/ctr\n", escaped, **_v2("run/cgroup view/ctr", 2 * GIB, GIB))
    assert host_memory.physical_memory() == MemorySnapshot(2 * GIB, GIB)
    for outside in ("0::/kubepods/pod2/ctr\n", "0::/kubepods/pod1/../pod2\n"):
        edit(**{"proc/self/cgroup": outside})
        assert host_memory.physical_memory() is None, outside
    edit(**{"proc/self/mountinfo": ROOT_MOUNT + "\n"})
    assert host_memory.physical_memory() == HOST


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin", "win32")), reason="no native probe here")
def test_this_host_reports_its_physical_memory():
    """The real probe on each CI platform: a reading, never unknown."""
    memory = host_memory.physical_memory()
    assert memory is not None
    assert memory.total_bytes > 0 and 0 <= memory.available_bytes <= memory.total_bytes
