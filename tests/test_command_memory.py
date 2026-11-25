"""Per-command memory leaves (#3792): probe, prepare, outcome and the stdin gate, on a fake cgroup tree.

macOS-compatible: no cgroup is touched. ``command_memory._ROOT`` and the unit directory point into a
temporary tree, the host reading is patched, and the gate shim runs under the real ``/bin/sh``.
"""
from __future__ import annotations

import errno
import os
import pathlib
import subprocess
import sys
import time

import pytest

from ouroboros import command_memory as cm
from ouroboros import host_memory
from ouroboros.host_memory import MemorySnapshot

GIB = 1 << 30
PAGE = 4096  # the page size the fake kernel rounds memory.max down to
TOTAL, RESERVE = 100 * GIB, 10 * GIB  # HOST_MEMORY_RESERVE_FRACTION is 0.10


@pytest.fixture
def tree(tmp_path, monkeypatch):
    unit = tmp_path / "unit"
    (unit / "commands").mkdir(parents=True)
    (unit / "cgroup.subtree_control").write_text("")
    commands = unit / "commands"
    (commands / "cgroup.subtree_control").write_text("")
    for name, value in (("memory.current", "0"), ("memory.stat", ""), ("memory.swap.max", "max"),
                        ("memory.events.local", "oom 0\noom_kill 0\n"), ("memory.max", "max")):
        (commands / name).write_text(value)
    proc = tmp_path / "proc"
    (proc / "meminfo").parent.mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\nSwapTotal: 0 kB\n")
    state = cm.Enabled("ouroboros.service", "/u/ouroboros.service", unit, commands)
    monkeypatch.setattr(cm, "_ROOT", tmp_path)
    monkeypatch.setattr(cm, "probe", lambda *a, **k: state)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(cm, "_PROBED", None)
    monkeypatch.setattr(cm, "_TRANSIENT", None)
    monkeypatch.setattr(cm, "_page_size", lambda: PAGE)
    real_write = cm._write

    def kernel_write(path, value):  # cgroup v2 keeps a byte limit in whole pages: a write is rounded down
        if path.name == "memory.max" and value.isdigit():
            value = str(int(value) // PAGE * PAGE)
        real_write(path, value)

    monkeypatch.setattr(cm, "_write", kernel_write)
    return type("Tree", (), {"root": tmp_path, "state": state, "commands": commands, "proc": proc})


def host(monkeypatch, available, total=TOTAL):
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(total, available))


def put(commands, current=0, stat=""):
    (commands / "memory.current").write_text(str(current))
    (commands / "memory.stat").write_text(stat)


def num(path):
    return int(path.read_text(encoding="utf-8"))


def test_frozen_gate_uses_bundle_python_without_system_path(tmp_path, monkeypatch):
    from ouroboros.platform_layer import embedded_python_candidates

    bundle = tmp_path / "_internal"
    python = embedded_python_candidates(bundle)[0]
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    launcher = tmp_path / "Ouroboros"
    launcher.touch(mode=0o700)
    monkeypatch.setattr(sys, "executable", str(launcher))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.delenv("OUROBOROS_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("PATH", "")
    assert cm.gate_argv(["/bin/true"], 7)[:4] == [str(python), "-I", "-S", "-c"]


def test_leaf_takes_all_free_room_and_parent_never_below_current(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    put(tree.commands, current=5 * GIB, stat=f"inactive_file {GIB}\n")
    lease = cm.begin()
    assert num(lease.leaf / "memory.max") == 30 * GIB  # available - reserve
    assert (lease.leaf / "memory.swap.max").read_text(encoding="utf-8") == "0" and (lease.leaf / "memory.oom.group").read_text(encoding="utf-8") == "1"
    assert (tree.commands / "memory.swap.max").read_text(encoding="utf-8") == "0"
    # room = 40 - 10 = 30 GiB; the parent is what the commands hold (5 - 1 inactive) plus that room
    assert num(tree.commands / "memory.max") == 34 * GIB
    assert "memory" in (tree.state.unit_root / "cgroup.subtree_control").read_text(encoding="utf-8")


def test_runtime_larger_than_reserve_leaves_less_room(tree, monkeypatch):
    host(monkeypatch, 25 * GIB)  # the runtime and OS hold 75 GiB: MemAvailable already excludes them
    lease = cm.begin()
    assert num(lease.leaf / "memory.max") == 15 * GIB
    assert num(tree.commands / "memory.max") == 15 * GIB  # min(90, 0 held + 25 - 10)


def test_under_reserve_refuses_before_any_write(tree, monkeypatch):
    host(monkeypatch, 8 * GIB)
    (tree.proc / "meminfo").write_text(f"MemAvailable: {8 * GIB // 1024} kB\nSwapTotal: 0 kB\n")
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == "under_reserve" and refused.value.process_not_started
    assert (tree.commands / "memory.max").read_text(encoding="utf-8") == "max"  # the parent was not rewritten
    assert not list(tree.commands.glob("cmd-*"))


def test_budget_exhausted_is_distinct_from_under_reserve(tree, monkeypatch):
    host(monkeypatch, 8 * GIB)  # the unit's own limit leaves 8 GiB while the host has 40 free
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == "budget_exhausted"


def test_a_remainder_of_one_page_is_the_whole_leaf_limit(tree, monkeypatch):
    host(monkeypatch, RESERVE + PAGE)
    assert num(cm.begin().leaf / "memory.max") == PAGE


def test_every_limit_is_written_in_whole_pages_and_read_back_as_such(tree, monkeypatch):
    host(monkeypatch, RESERVE + 3 * PAGE + 100)  # 100 bytes past a page boundary
    put(tree.commands, current=0, stat="")
    lease = cm.begin()
    assert num(lease.leaf / "memory.max") == 3 * PAGE  # not 3 pages + 100, which the kernel would shorten
    assert num(tree.commands / "memory.max") % PAGE == 0
    live = tree.commands / "cmd-1-aa"
    live.mkdir()
    (live / "memory.max").write_text(str(PAGE))
    (live / "memory.current").write_text("0")
    host(monkeypatch, RESERVE + 7 * PAGE + 5)
    cm.begin()
    assert num(live / "memory.max") == 7 * PAGE


@pytest.mark.parametrize("held", [10, 40, 80])
def test_steady_growth_keeps_the_parent_at_or_above_what_is_held_and_below_the_headroom(tree, monkeypatch, held):
    host(monkeypatch, 50 * GIB)
    put(tree.commands, current=held * GIB)
    quota, parent, _, room = cm._compute(tree.commands)
    assert held * GIB <= parent <= (TOTAL - RESERVE)
    assert parent == min(TOTAL - RESERVE, held * GIB + room)  # what is held plus the room, never clamped to it
    assert quota + held * GIB <= parent  # the leaf limit trips before the parent's


def test_a_stationary_group_holding_more_than_the_room_keeps_its_headroom(tree, monkeypatch):
    host(monkeypatch, 15 * GIB)  # room 5 GiB while the commands hold 20 GiB
    put(tree.commands, current=20 * GIB)
    quota, parent, _, _ = cm._compute(tree.commands)
    assert parent == 25 * GIB and quota == 5 * GIB  # held + room: a next allocation finds headroom, not the parent's OOM


def test_file_cache_and_reclaimable_slab_are_not_held_memory(tree, monkeypatch):
    """100 GiB host, 10 GiB reserve, 30 GiB outside the commands. They charge 50 GiB, 40 of it active file cache: 10 GiB
    anonymous, so what they may hold is 100 - 30 - 10 = 60 GiB (counting the cache as held would grant 90)."""
    host(monkeypatch, 60 * GIB)  # MemAvailable: 100 - 30 outside - 10 anonymous
    put(tree.commands, current=50 * GIB, stat=f"active_file {40 * GIB}\ninactive_file 0\nslab_reclaimable 0\nanon {10 * GIB}\n")
    assert cm._held(tree.commands) == (50 * GIB, 10 * GIB)
    _, parent, _, room = cm._compute(tree.commands)
    assert room == 50 * GIB and parent == 60 * GIB
    put(tree.commands, current=50 * GIB, stat=f"inactive_file {20 * GIB}\nactive_file {10 * GIB}\nslab_reclaimable {5 * GIB}\n")
    assert cm._held(tree.commands) == (50 * GIB, 15 * GIB)  # all three are the kernel's to give back


def sampled(monkeypatch, commands_reads, available, runtime=None, total=TOTAL):
    """The commands' nonreclaimable charge as ``_compute`` reads it just before and after the ONE host snapshot
    (``available``, the moment between them); ``runtime`` is the runtime subgroup's own charge (None: unreadable)."""
    reads = iter(commands_reads)

    def held(group):
        if group.name == "runtime":
            if runtime is None:
                raise OSError("no runtime subgroup")
            return runtime, runtime
        value = next(reads)
        return value, value
    monkeypatch.setattr(cm, "_held", held)
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(total, available))


@pytest.mark.parametrize("reads,available", [
    ((0, 40 * GIB), 70 * GIB),  # growing across the snapshot: it saw 20 of them
    ((40 * GIB, 0), 70 * GIB),  # releasing across the snapshot
    ((40 * GIB, 40 * GIB), 50 * GIB),  # steady
])
def test_commands_moving_monotonically_across_the_snapshot_never_over_grant(tree, monkeypatch, reads, available):
    """10 GiB outside the commands: what they may safely hold is 100 - 10 - 10 = 80 GiB, however they moved."""
    sampled(monkeypatch, reads, available)
    _, parent, _, free = cm._compute(tree.commands)
    assert parent <= 80 * GIB and max(reads) + free <= parent


def test_a_build_cycling_between_the_reads_cannot_hide_the_runtime(tree, monkeypatch):
    """Review witness: a 256 MiB unit, 100 MiB in the runtime, a build that holds 150 MiB at both commands reads and
    nothing at the snapshot. Differencing alone sees no one outside (a 230 MiB parent: the unit's OOM would reach the
    workers at 156 MiB); the runtime's own charge, read directly, keeps the parent at 256 - 25.6 - 100 MiB."""
    mib = 1 << 20
    unit = 256 * mib
    sampled(monkeypatch, (150 * mib, 150 * mib), unit - 100 * mib, runtime=100 * mib, total=unit)
    _, parent, snap, free = cm._compute(tree.commands)
    assert parent == (unit - snap.reserve_bytes - 100 * mib) // PAGE * PAGE and parent < 156 * mib
    assert free == 0  # the build already holds more than the parent leaves: a new command is refused, not over-granted


def test_the_parent_is_the_room_commands_may_hold_never_their_raw_charge(tree, monkeypatch):
    """Review witness: 100 GiB host, 40 GiB outside, the commands charge 59 GiB of which 55 is reclaimable cache. The
    parent is 100 - 10 - 40 = 50 GiB, not the 59 GiB raw charge: the kernel reclaims the cache under it, and an
    anonymous replacement of that cache can never eat the reserve."""
    host(monkeypatch, 56 * GIB)  # 100 - 40 outside - 4 anonymous
    put(tree.commands, current=59 * GIB, stat=f"inactive_file {55 * GIB}\n")
    lease = cm.begin()
    assert num(tree.commands / "memory.max") == 50 * GIB and num(lease.leaf / "memory.max") == 46 * GIB


@pytest.mark.parametrize("usage,cache,runtime,sibling,limit,busy,available", [
    (70, 0, 20, 100, 256, 0, 136),  # build had 70 MiB at both bookends, but 0 at the snapshot
    (424, 360, 50, 0, 512, 424, 38),  # 360 MiB active cache inflated the ancestor's raw charge
])
def test_finite_ancestor_bounds_commands_with_direct_sibling_usage_and_consistent_reclaim(
    tree, monkeypatch, usage, cache, runtime, sibling, limit, busy, available,
):
    mib = 1 << 20
    sliced = tree.root / "finite-slice"
    unit = sliced / "ouroboros.service"
    commands = unit / "commands"
    for group, charge, file_cache in (
        (sliced, busy + runtime + sibling, cache), (unit, busy + runtime, cache),
        (unit / "runtime", runtime, 0), (sliced / "peer.service", sibling, 0), (commands, usage, cache),
    ):
        group.mkdir(parents=True, exist_ok=True)
        (group / "cgroup.controllers").write_text("memory")
        put(group, charge * mib, f"active_file {file_cache * mib}\n")
    (sliced / "memory.max").write_text(str(limit * mib))
    (tree.proc / "meminfo").write_text("MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n")
    host(monkeypatch, available * mib, limit * mib)
    quota, parent, snap, _ = cm._compute(commands)
    assert parent == (limit * mib - snap.reserve_bytes - (runtime + sibling) * mib) // PAGE * PAGE
    assert quota == parent - (usage - cache) * mib > 0


def test_live_leaves_only_gain_room(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    small, large = tree.commands / "cmd-1-aa", tree.commands / "cmd-2-bb"
    for leaf, value in ((small, 5 * GIB), (large, 80 * GIB)):
        leaf.mkdir()
        (leaf / "memory.max").write_text(str(value))
        (leaf / "memory.current").write_text(str(value))
    cm.begin()
    assert num(small / "memory.max") == 30 * GIB and num(large / "memory.max") == 80 * GIB


def test_two_healthy_commands_both_start(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    first = cm.begin()
    host(monkeypatch, 30 * GIB)  # the first one holds 10 GiB
    put(tree.commands, current=10 * GIB)
    (first.leaf / "memory.current").write_text(str(10 * GIB))
    second = cm.begin()
    assert first.leaf != second.leaf and num(second.leaf / "memory.max") == 20 * GIB
    assert num(first.leaf / "memory.max") == 30 * GIB  # a running command is never lowered


def test_a_command_that_started_low_gets_room_the_others_freed_once_its_limit_is_hit(tree, monkeypatch):
    host(monkeypatch, 15 * GIB)  # another command holds most of the room at the start: the leaf is small
    lease = cm.begin()
    assert num(lease.leaf / "memory.max") == 5 * GIB
    (lease.leaf / "memory.current").write_text("0")
    host(monkeypatch, 60 * GIB)  # that command ended
    (lease.leaf / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\n")
    reader = lease.watch(lambda: False)
    assert reader() is False and num(lease.leaf / "memory.max") == 5 * GIB  # no limit hit: nothing is recomputed
    (lease.leaf / "memory.events").write_text("max 3\n")
    reader()
    assert num(lease.leaf / "memory.max") == 50 * GIB and lease.seen_max == 3  # raised, never lowered
    host(monkeypatch, 20 * GIB)
    reader()
    assert num(lease.leaf / "memory.max") == 50 * GIB  # the same count again: no second recompute
    assert cm.Lease(tree.state, lease.leaf, 0).watch()() is None  # no Stop reader: nothing to stop


def test_ebusy_retries_are_bounded(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    real, calls = cm._write, []

    def flaky(path, value):
        if path.name == "memory.max" and path.parent == tree.commands:
            calls.append(value)
            if len(calls) < 3:
                raise OSError(errno.EBUSY, "busy")
        real(path, value)

    monkeypatch.setattr(cm, "_write", flaky)
    assert cm.begin() is not None and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(cm, "_write", lambda p, v: (_ for _ in ()).throw(OSError(errno.EBUSY, "busy"))
                        if p.name == "memory.max" and p.parent == tree.commands else real(p, v))
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == "prepare_failed"


def test_swap_file_absent_with_swap_is_unavailable_without_swap_allowed(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    (tree.commands / "memory.swap.max").unlink()
    assert cm.begin() is not None  # SwapTotal is 0
    (tree.commands / "memory.swap.max").unlink(missing_ok=True)
    (tree.proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\nSwapTotal: 1024 kB\n")
    assert cm.begin() is None
    assert isinstance(cm._PROBED, cm.Unavailable) and cm._PROBED.code == "swap_unmanaged"


def test_unsupported_swap_layout_changes_nothing_before_the_parent_is_written(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    (tree.commands / "memory.swap.max").unlink()
    (tree.proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\nSwapTotal: 1024 kB\n")
    assert cm.begin() is None
    assert (tree.commands / "memory.max").read_text(encoding="utf-8") == "max"  # the parent was never lowered


def test_move_is_read_back_and_a_wrong_group_refuses(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    lease = cm.begin()
    (tree.proc / "4242").mkdir()
    (tree.proc / "4242" / "cgroup").write_text("0::/u/ouroboros.service/runtime\n")
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        lease.place(4242)
    assert refused.value.code == "move_failed"
    (tree.proc / "4242" / "cgroup").write_text(f"0::/u/ouroboros.service/commands/{lease.leaf.name}\n")
    lease.place(4242)
    assert (lease.leaf / "cgroup.procs").read_text(encoding="utf-8") == "4242"


def test_outcome_reports_separate_facts(tree):
    leaf = tree.commands / "cmd-1-aa"
    leaf.mkdir()
    (leaf / "memory.events.local").write_text("oom 1\noom_kill 1\noom_group_kill 1\n")
    (tree.commands / "memory.events.local").write_text("oom 3\noom_kill 1\n")
    facts = cm.outcome(leaf, 9, baseline=2)
    assert facts == {"victim_in_leaf": True, "leaf_local_oom": 1, "commands_local_oom": 1, "unknown": False, "pid": 9}
    assert cm.facts_worth_publishing(facts, -9) == facts


def test_external_sigkill_and_unrelated_oom_publish_no_service_verdict(tree):
    leaf = tree.commands / "cmd-1-aa"
    leaf.mkdir()
    (leaf / "memory.events.local").write_text("oom 0\noom_kill 0\n")
    facts = cm.outcome(leaf, 9)
    assert facts["victim_in_leaf"] is False and cm.facts_worth_publishing(facts, -9) is None  # someone else's SIGKILL
    # a neighbour's collision in commands/: observed separately, the victim was not this leaf
    (tree.commands / "memory.events.local").write_text("oom 2\n")
    only_commands = cm.outcome(leaf, 9)
    assert only_commands["victim_in_leaf"] is False and only_commands["commands_local_oom"] == 2
    from ouroboros.tools.process_facts import describe_returncode

    text = describe_returncode(-9, memory_oom=only_commands)
    assert "no kill in its memory group" in text and "service" not in text


def test_unreadable_events_are_unknown_never_a_claim(tree):
    leaf = tree.commands / "cmd-1-aa"
    leaf.mkdir()  # no memory.events.local at all
    facts = cm.outcome(leaf, 9)
    assert facts["unknown"] is True and facts["victim_in_leaf"] is None
    assert cm.facts_worth_publishing(facts, 1) is None and cm.facts_worth_publishing(facts, -9) == facts


def test_finish_removes_empty_leaf_and_recomputes_parent(tree, monkeypatch):
    host(monkeypatch, 40 * GIB)
    lease = cm.begin()
    host(monkeypatch, 50 * GIB)
    for control in lease.leaf.iterdir():  # cgroupfs lets rmdir pass over its own control files
        control.unlink()
    lease.finish(1)
    assert not lease.leaf.exists() and num(tree.commands / "memory.max") == 40 * GIB  # min(90, 0 + 50 - 10)


def test_sweep_removes_dead_creators_empty_leaves_and_keeps_live_ones(tree, monkeypatch):
    (tree.proc / str(os.getpid())).mkdir()
    dead, live = tree.commands / "cmd-999999-aa", tree.commands / f"cmd-{os.getpid()}-bb"
    dead.mkdir(), live.mkdir()
    cm._sweep(tree.state)
    assert not dead.exists() and live.exists()


def test_a_finished_commands_busy_leaf_is_removed_by_a_later_sweep_once_empty(tree, monkeypatch):
    leaf = tree.commands / f"cmd-{os.getpid()}-cc"
    leaf.mkdir()
    (tree.proc / str(os.getpid())).mkdir()  # its creator is alive: only the finished mark releases it
    real_rmdir = pathlib.Path.rmdir
    busy = [True]

    def rmdir(self):
        if self == leaf and busy[0]:
            raise OSError(errno.EBUSY, "background child still in the leaf")
        real_rmdir(self)

    monkeypatch.setattr(pathlib.Path, "rmdir", rmdir)
    host(monkeypatch, 40 * GIB)
    cm.Lease(tree.state, leaf, 0).finish(1)
    cm._sweep(tree.state)
    assert leaf.exists() and leaf in cm._FINISHED_BUSY
    busy[0] = False
    cm._sweep(tree.state)
    assert not leaf.exists() and leaf not in cm._FINISHED_BUSY


# -------------------------------------------------------------------------------------------- probe

def props(**changed):
    base = {"LoadState": "loaded", "ControlGroup": "/u/ouroboros.service", "OOMPolicy": "continue",
            "DelegateControllers": "cpu memory pids", "DelegateSubgroup": "runtime"}
    return {**base, **changed}


def test_probe_reports_unmanaged_swap_up_front_like_begin(tmp_path, monkeypatch):
    unit = tmp_path / "sys" / "fs" / "cgroup" / "u" / "ouroboros.service"
    unit.mkdir(parents=True)
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/u/ouroboros.service/runtime\n", encoding="utf-8")
    (tmp_path / "proc" / "meminfo").write_text("MemAvailable: 1 kB\nSwapTotal: 1024 kB\n", encoding="utf-8")
    monkeypatch.setattr(cm, "_ROOT", tmp_path)
    monkeypatch.setattr(cm.sys, "platform", "linux")
    monkeypatch.setattr(cm, "_uid", lambda: 1000)  # no getuid on Windows: the probe must not need it there
    monkeypatch.setattr(cm, "_unit_dir", lambda _cgroup: unit)
    monkeypatch.setattr(cm, "_show", lambda *a: props())
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(TOTAL, 40 * GIB))
    assert cm._probe().code == "swap_unmanaged"  # swap exists, the unit has no memory.swap.max
    (unit / "memory.swap.max").write_text("max")
    assert isinstance(cm._probe(), cm.Enabled)


def test_unit_checks_name_the_missing_piece():
    ok = cm._check_unit(props(), "ouroboros.service", subgroup=True, cgroup="/u/ouroboros.service")
    assert ok is None
    assert cm._check_unit(props(LoadState="not-found"), "x", subgroup=True, cgroup="/u/ouroboros.service").code == "unit_not_loaded"
    assert cm._check_unit(props(DelegateControllers="cpu"), "x", subgroup=True, cgroup="/u/ouroboros.service").code == "unit_policy"
    assert cm._check_unit(props(DelegateSubgroup=""), "x", subgroup=True, cgroup="/u/ouroboros.service").code == "unit_policy"
    manager = props(ControlGroup="/u", DelegateSubgroup="")
    assert cm._check_unit(manager, "user@1.service", subgroup=False, cgroup="/u/ouroboros.service") is None


def test_user_shadow_is_named_not_overridden_and_reload_is_separate():
    shadowed = cm._check_unit(props(OOMPolicy="stop", DropInPaths="/home/a/.config/systemd/user/ouroboros.service.d/x.conf"),
                              "ouroboros.service", subgroup=True, cgroup="/u/ouroboros.service")
    assert shadowed.code == "unit_policy" and "/home/a/.config/systemd/user/ouroboros.service.d/x.conf" in shadowed.reason
    fresh = cm._check_unit(props(OOMPolicy="stop", NeedDaemonReload="yes"), "ouroboros.service", subgroup=True,
                           cgroup="/u/ouroboros.service")
    assert fresh.code == "needs_reload" and fresh.transient


def test_transient_failure_is_reused_briefly_then_asked_again_and_a_definite_one_is_cached(monkeypatch):
    clock, calls = [100.0], []
    results = iter([cm.Unavailable("systemctl_unreadable", "x", True), cm.Unavailable("unit_policy", "y")])
    monkeypatch.setattr(cm, "_PROBED", None)
    monkeypatch.setattr(cm, "_TRANSIENT", None)
    monkeypatch.setattr(cm, "_now", lambda: clock[0])
    monkeypatch.setattr(cm, "_probe", lambda: calls.append(1) or next(results))
    assert cm.probe().code == "systemctl_unreadable" and cm.probe().code == "systemctl_unreadable"
    assert len(calls) == 1  # /status does not spawn systemctl again within the window
    clock[0] += cm._TRANSIENT_TTL_SEC + 1
    assert cm.probe().code == "unit_policy" and cm.probe().code == "unit_policy"
    assert len(calls) == 2  # a definite verdict is cached for good


@pytest.mark.skipif(os.name == "nt", reason="cgroup v2 is Linux; the lock helper is exercised with POSIX flock")
def test_lock_goes_through_the_platform_layer_is_per_uid_off_xdg_and_bounded(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(cm.tempfile, "gettempdir", lambda: str(tmp_path))
    seen = []
    real = platform_layer.file_lock_exclusive_nb
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", lambda fd: seen.append(fd) or real(fd))
    with cm._lock():
        assert (tmp_path / f"ouroboros-command-memory-{os.getuid()}.lock").exists() and seen
        monkeypatch.setattr(cm, "_LOCK_WAIT_SEC", 0.2)
        with pytest.raises(cm.CommandMemoryRefused) as refused, cm._lock():  # a second holder is refused, not awaited forever
            pass
        assert refused.value.code == "lock_timeout" and refused.value.process_not_started
    with cm._lock():  # released: free again
        pass


def test_probe_only_shows_never_changes_the_unit(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.append(argv) or subprocess.CompletedProcess(argv, 0, "LoadState=loaded\n"))
    cm._show("--user", "show", "ouroboros.service")
    assert seen == [["systemctl", "--user", "show", "ouroboros.service"]]  # the one place a command is run


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="non-Linux status is empty")
def test_non_linux_is_unavailable_and_silent(monkeypatch):
    monkeypatch.setattr(cm, "_PROBED", None)
    assert cm.probe().code == "platform" and cm.begin() is None and cm.status_note() == ""


def observed(monkeypatch, state):
    monkeypatch.setattr(cm, "_PROBED", state)
    monkeypatch.setattr(cm, "_TRANSIENT", None)


def test_status_is_silent_only_off_linux(monkeypatch):
    monkeypatch.setattr(cm.sys, "platform", "darwin")
    observed(monkeypatch, cm.Unavailable("platform", "not Linux"))
    assert cm.status_note() == ""


@pytest.mark.parametrize("code", ["no_cgroup_v2", "no_runtime_subgroup", "unit_policy"])
def test_linux_status_names_an_unprotected_run_so_it_differs_from_a_protected_one(monkeypatch, code):
    """Review: a desktop/AppImage session and a protected unit used to show the same empty status."""
    monkeypatch.setattr(cm.sys, "platform", "linux")
    observed(monkeypatch, cm.Unavailable(code, "why"))
    assert cm.status_note() == f"command_memory: off ({code}: why)"
    observed(monkeypatch, cm.Enabled("a", "/a", pathlib.Path("/a"), pathlib.Path("/a/commands")))
    assert cm.status_note().startswith("command_memory: on") and cm.isolating()


def test_status_never_probes_on_the_callers_thread_and_asks_a_daemon_for_a_verdict(monkeypatch):
    import threading

    asked, caller = threading.Event(), threading.get_ident()
    where = []
    monkeypatch.setattr(cm.sys, "platform", "linux")
    observed(monkeypatch, None)
    monkeypatch.setattr(cm, "probe", lambda *a, **k: where.append(threading.get_ident()) or asked.set())
    monkeypatch.setattr(cm, "_in_packaged_unit", lambda: False)
    note = cm.status_note()  # no packaged unit around this process: the cheap read names it off, no thread
    assert note.startswith("command_memory: off (not started by the packaged ouroboros.service") and not where
    monkeypatch.setattr(cm, "_in_packaged_unit", lambda: True)
    assert cm.status_note() == "command_memory: checking"  # nothing observed yet: the verdict comes off this thread
    assert asked.wait(10) and where and where[0] != caller
    cm._REFRESHING.wait(0)
    for _ in range(100):
        if not cm._REFRESHING.is_set():
            break
        time.sleep(0.05)
    assert not cm._REFRESHING.is_set()


def cgroup_tree(tmp_path, monkeypatch, own):
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text(own, encoding="utf-8")
    monkeypatch.setattr(cm, "_ROOT", tmp_path)
    monkeypatch.setattr(cm.sys, "platform", "linux")
    monkeypatch.setattr(cm, "_uid", lambda: 1000)


def test_our_unit_without_a_runtime_subgroup_is_named_not_silent(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, "0::/user.slice/user-1000.slice/user@1000.service/app.slice/ouroboros.service\n")
    shown = {"NeedDaemonReload": "no", "Version": "255.4-1"}
    monkeypatch.setattr(cm, "_show", lambda *a: dict(shown))
    state = cm._probe()
    assert state.code == "unit_without_runtime" and "older package" in state.reason
    shown["NeedDaemonReload"] = "yes"
    reload = cm._probe()
    assert reload.code == "needs_reload" and reload.transient and "daemon-reload and restart" in reload.reason
    shown.update(NeedDaemonReload="no", Version="253.1")
    old = cm._probe()
    assert old.code == "unit_without_runtime" and "systemd 253 ignores DelegateSubgroup" in old.reason
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/user.slice/user-1000.slice/session-3.scope\n", encoding="utf-8")
    assert cm._probe().code == "no_runtime_subgroup"


def test_only_the_packaged_unit_is_ever_a_candidate_for_rewriting(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, "0::/user.slice/user-1000.slice/user@1000.service/app.slice/other.service/runtime\n")
    monkeypatch.setattr(cm, "_show", lambda *a: pytest.fail("another service's unit is never even read"))
    assert cm._own_cgroup((tmp_path / "proc/self/cgroup").read_text(encoding="utf-8")) is None
    assert cm._probe().code == "no_runtime_subgroup"


def test_the_packaged_unit_on_cgroup_v1_and_a_desktop_session_are_both_named(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, "12:memory:/user.slice/user-1000.slice/user@1000.service/app.slice/ouroboros.service\n"
                                      "1:name=systemd:/user.slice/user-1000.slice/user@1000.service/app.slice/ouroboros.service\n")
    state = cm._probe()
    assert state.code == "cgroup_v1" and "ouroboros.service runs on cgroup v1" in state.reason
    observed(monkeypatch, state)
    assert cm.status_note().startswith("command_memory: off (cgroup_v1:")
    (tmp_path / "proc" / "self" / "cgroup").write_text("12:memory:/user.slice/user-1000.slice/session-3.scope\n", encoding="utf-8")
    assert cm._probe().code == "no_cgroup_v2"
    observed(monkeypatch, cm._probe())
    assert cm.status_note().startswith("command_memory: off (no_cgroup_v2:")  # the desktop session is named, not silent


def test_membership_is_read_once_so_a_failing_second_read_cannot_become_a_cached_absence(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, "0::/u/ouroboros.service/runtime\n")
    reads = []
    real = cm._cgroup_text

    def flaky():
        reads.append(1)
        if len(reads) > 1:
            raise OSError(errno.EIO, "second read fails")
        return real()

    monkeypatch.setattr(cm, "_cgroup_text", flaky)
    monkeypatch.setattr(cm, "_unit_dir", lambda _cgroup: tmp_path)
    monkeypatch.setattr(cm, "_show", lambda *a: props())
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(TOTAL, 40 * GIB))
    (tmp_path / "proc" / "meminfo").write_text("SwapTotal: 0 kB\n", encoding="utf-8")
    assert isinstance(cm._probe(), cm.Enabled) and len(reads) == 1  # the unit was identified from the one read


def test_an_unreadable_mount_table_after_the_unit_is_identified_is_a_retryable_refusal_a_confirmed_absence_falls_back(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, "0::/u/ouroboros.service/runtime\n")
    unreadable = cm._probe()  # no /proc/self/mountinfo in the fake tree
    assert unreadable.code == "mounts_unreadable" and unreadable.transient and unreadable.code in cm._IDENTIFIED_TRANSIENT
    monkeypatch.setattr(cm, "probe", lambda refresh=False: unreadable)
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == "mounts_unreadable" and refused.value.process_not_started
    (tmp_path / "proc" / "self" / "mountinfo").write_text("36 25 0:30 /elsewhere /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n", encoding="utf-8")
    absent = cm._probe()  # the table is read and shows no mount for the unit: a definite layout
    assert absent.code == "cgroup_not_visible" and not absent.transient
    monkeypatch.setattr(cm, "probe", lambda refresh=False: absent)
    assert cm.begin() is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_a_forked_child_starts_with_a_free_probe_lock_and_no_inherited_verdict(monkeypatch):
    """OUROBOROS_WORKER_START_METHOD=fork: a lock held by another thread at fork time is never inherited."""
    monkeypatch.setattr(cm, "_PROBED", cm.Unavailable("unit_policy", "x"))
    cm._REFRESHING.set()
    try:
        assert cm._PROBE_LOCK.acquire(timeout=1)  # stands for the status-refresh thread mid-probe
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:  # a child: the at-fork hook has run; report and leave without running pytest's teardown
            ok = cm._PROBE_LOCK.acquire(timeout=1) and not cm._REFRESHING.is_set() and cm._PROBED is None and cm._TRANSIENT is None
            os.write(write, b"1" if ok else b"0")
            os._exit(0)
        os.close(write)
        verdict = os.read(read, 1)
        os.close(read)
        os.waitpid(pid, 0)
    finally:
        cm._REFRESHING.clear()
        cm._PROBE_LOCK.release()
    assert verdict == b"1"


def test_a_unit_cgroup_without_the_memory_controller_is_a_definite_layout(tmp_path, monkeypatch):
    unit = tmp_path / "sys" / "fs" / "cgroup" / "u" / "ouroboros.service"
    unit.mkdir(parents=True)
    cgroup_tree(tmp_path, monkeypatch, "0::/u/ouroboros.service/runtime\n")
    monkeypatch.setattr(cm, "_unit_dir", lambda _cgroup: unit)
    monkeypatch.setattr(cm, "_show", lambda *a: props())
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(TOTAL, 40 * GIB))
    (tmp_path / "proc" / "meminfo").write_text("SwapTotal: 0 kB\n", encoding="utf-8")
    (unit / "cgroup.controllers").write_text("cpu pids\n")
    state = cm._probe()
    assert state.code == "memory_controller_missing" and not state.transient
    (unit / "cgroup.controllers").write_text("cpu memory pids\n")
    assert isinstance(cm._probe(), cm.Enabled)


@pytest.mark.parametrize("err", [errno.ENOENT, errno.EINVAL])
def test_a_controller_that_cannot_be_enabled_is_unsupported_not_a_permanent_prepare_failure(tree, monkeypatch, err):
    host(monkeypatch, 40 * GIB)
    real = cm._write

    def deny(path, value):
        if path.name == "cgroup.subtree_control":
            raise OSError(err, "no such controller")
        real(path, value)

    monkeypatch.setattr(cm, "_write", deny)
    assert cm.begin() is None  # the old path runs
    assert cm._PROBED.code == "memory_controller_missing"
    assert (tree.commands / "memory.max").read_text(encoding="utf-8") == "max"


def test_a_known_layout_whose_first_host_sample_fails_refuses_instead_of_running_unisolated(tree, monkeypatch):
    results = iter([cm.Unavailable("memory_unreadable", "host memory could not be read", True)] * 2 + [tree.state])
    monkeypatch.setattr(cm, "probe", lambda refresh=False: next(results))
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == "memory_unreadable" and refused.value.process_not_started
    host(monkeypatch, 40 * GIB)
    monkeypatch.setattr(cm, "probe", lambda refresh=False: tree.state)  # the next readable sample admits
    assert cm.begin() is not None
    monkeypatch.setattr(cm, "probe", lambda refresh=False: cm.Unavailable("unit_policy", "OOMPolicy=stop"))
    assert cm.begin() is None  # a confirmed-unsupported layout keeps the fallback


# ---------------------------------------------------------------------------- exec semantics (process_facts)

def program(directory, name="prog", body="#!/bin/sh\nexit 0\n", mode=0o755):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    path.chmod(mode)
    return path


def run_gate(argv, release=None, env=None, cwd=None, wait=10):
    """``(returncode, stdout, exec errno)``: the helper started as ``gated_start`` starts it, then released (or fed
    ``release`` bytes and EOF)."""
    status_r, status_w = os.pipe()
    try:
        proc = subprocess.Popen(cm.gate_argv(argv, status_w), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env={"PATH": os.defpath}, cwd=cwd, pass_fds=(status_w,))
    finally:
        os.close(status_w)
    try:
        if release is None:
            cm.release_gate(proc, {"PATH": "/usr/bin:/bin", **(env or {})})
        else:
            os.write(proc.stdin.fileno(), release)
        proc.stdin.close()
        proc.stdin = None  # closed: communicate() must not flush it
        failed = cm.exec_verdict(status_r, wait)
        out, _ = proc.communicate(timeout=wait)
        return proc.returncode, out, failed
    finally:
        os.close(status_r)


posix = pytest.mark.skipif(os.name == "nt", reason="POSIX exec semantics")


@posix
def test_gate_runs_the_argv_only_after_release_with_null_stdin():
    assert run_gate(["/bin/sh", "-c", "read x; echo got:$x"]) == (0, b"got:\n", None)


@posix
def test_gate_refusal_or_a_short_or_unsentinelled_stream_never_runs_the_argv(tmp_path):
    flag = tmp_path / "ran"
    argv = ["/bin/sh", "-c", f"touch {flag}"]
    blob = b"PATH=/usr/bin:/bin"
    for release in (b"", b"no\n", b"%d\n" % (len(blob) + 5) + blob,  # EOF; garbage; a count the blob never reached
                    b"%d\n" % len(blob) + blob,  # the whole blob, the sentinel never came
                    b"%d\n" % len(blob) + blob + b"\0wrong-sentinel-here\n"):
        assert run_gate(argv, release)[0] == cm.GATE_NOT_STARTED
    assert not flag.exists()


@posix
def test_every_admitted_name_reaches_the_target_exactly_and_never_shows_in_argv(tmp_path):
    """A direct Python target (no shell, which would drop a name like ``build.profile``): a bash-readonly UID and a name
    no shell can export arrive as given, and the helper's argv carries no secret before the release."""
    secret = "s3cr3t-value"
    probe = ("import json,os,sys;print(json.dumps({k:os.environ.get(k) for k in ('UID','build.profile','TOKEN')}))")
    status_r, status_w = os.pipe()
    try:
        proc = subprocess.Popen(cm.gate_argv([sys.executable, "-c", probe], status_w), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"PATH": os.defpath}, pass_fds=(status_w,))
    finally:
        os.close(status_w)
    try:
        assert not any(secret in part for part in proc.args)  # before the release: nothing but the argv itself
        cm.release_gate(proc, {"PATH": "/usr/bin:/bin", "UID": "999999", "build.profile": "a\nb'c\\d", "TOKEN": secret})
        proc.stdin.close()
        proc.stdin = None
        assert cm.exec_verdict(status_r, 10) is None
        out, err = proc.communicate(timeout=10)
    finally:
        os.close(status_r)
    import json
    assert proc.returncode == 0 and json.loads(out) == {"UID": "999999", "build.profile": "a\nb'c\\d", "TOKEN": secret}, err


@posix
def test_a_pipeline_under_the_gate_keeps_the_default_sigpipe():
    """The helper's ``-I -S`` start ignores SIGPIPE and SIGXFSZ; they are default again at the exec, so ``yes | head`` ends quietly."""
    code, out, failed = run_gate(["/bin/sh", "-c", "{ yes | head -n 1; } 2>&1"])  # stderr folded in: empty of any EPIPE complaint
    assert (code, out, failed) == (0, b"y\n", None)


@posix  # os.pipe() under DefaultSelector: Windows selectors take sockets only
def test_the_exec_verdict_tells_success_an_early_end_and_a_deadline_that_ran_out():
    """``x`` then EOF is the exec; EOF without ``x`` is a helper that ended before the argv (``GATE_ENDED``, never an
    exec); ``x`` then an errno is a refused exec; an open silent pipe is unknown (``EXEC_TIMEOUT``)."""
    def pipe(data=None):
        r, w = os.pipe()
        if data is not None:
            os.write(w, data) if data else None
            os.close(w)
            return r, None
        return r, w

    fds = [pipe(b"x"), pipe(b""), pipe(b"x%d" % errno.ENOENT), pipe()]
    try:
        assert cm.exec_verdict(fds[0][0], 1) is None
        assert cm.exec_verdict(fds[1][0], 1) == cm.GATE_ENDED
        assert cm.exec_verdict(fds[2][0], 1) == errno.ENOENT
        assert cm.exec_verdict(fds[3][0], 0.05) == cm.EXEC_TIMEOUT != None  # noqa: E711
    finally:
        for r, w in fds:
            os.close(r)
            w is not None and os.close(w)


@posix
def test_a_helper_that_fails_while_preparing_never_claims_an_exec(tmp_path, monkeypatch):
    """Review witness: the memory-limited helper dying (a MemoryError) while it built the environment closed the status
    pipe like an exec, so its exit was persisted as the check's. The failure is injected into the real helper source at
    that step: it exits 125 with no ``x`` (``GATE_ENDED``) and the argv never runs; a helper killed while it still
    waits for its release reads the same."""
    assert "e=dict(" in cm._ENV_HELPER
    monkeypatch.setattr(cm, "_ENV_HELPER", cm._ENV_HELPER.replace("e=dict(", "e=(_ for _ in ()).throw(MemoryError()) or dict("))
    flag = tmp_path / "ran"
    code, _, failed = run_gate(["/bin/sh", "-c", f"touch {flag}"])
    assert code == cm.GATE_NOT_STARTED and failed == cm.GATE_ENDED and not flag.exists()
    status_r, status_w = os.pipe()
    try:
        proc = subprocess.Popen(cm.gate_argv(["/bin/sh", "-c", f"touch {flag}"], status_w), stdin=subprocess.PIPE,
                                env={"PATH": os.defpath}, pass_fds=(status_w,))
        os.close(status_w)
        proc.kill()
        proc.wait(10)
        assert cm.exec_verdict(status_r, 5) == cm.GATE_ENDED and not flag.exists()
    finally:
        os.close(status_r)
        proc.stdin.close()


@posix
def test_dollar_zero_is_the_argvs_own_and_a_relative_path_entry_is_the_childs_cwd(tmp_path):
    assert run_gate(["sh", "-c", 'echo "$0"'])[1] == b"sh\n"  # not the resolved /bin/sh
    program(tmp_path / "bin", body="#!/bin/sh\necho child\n")
    assert run_gate(["prog"], env={"PATH": "bin"}, cwd=tmp_path)[1] == b"child\n"


@posix
def test_an_exec_refusal_is_a_typed_errno_never_a_started_commands_exit(tmp_path):
    program(tmp_path, "text", body="echo no shebang\n")
    program(tmp_path, "noexec", mode=0o644)
    assert run_gate(["missing-program"])[2] == errno.ENOENT
    assert run_gate(["./text"], cwd=tmp_path)[2] == errno.ENOEXEC  # Popen's exec: no shell fallback
    assert run_gate(["./noexec"], cwd=tmp_path)[2] == errno.EACCES
    program(tmp_path, "xonly", mode=0o111)  # execute-only: exec'd by the kernel though unreadable
    assert run_gate(["./xonly"], cwd=tmp_path)[2] is None


@posix
def test_a_path_search_selects_the_second_entry_past_a_script_whose_interpreter_is_missing(tmp_path):
    program(tmp_path / "first", "tool", body="#!/nonexistent/interpreter\n")
    program(tmp_path / "second", "tool", body="#!/bin/sh\necho second\n")
    code, out, failed = run_gate(["tool"], env={"PATH": f"{tmp_path / 'first'}:{tmp_path / 'second'}"})
    assert (code, out, failed) == (0, b"second\n", None)


@posix
def test_gate_exit_codes_of_a_completed_run_are_the_commands_own():
    for code in (0, 3, 125, 126, 127):
        assert run_gate(["/bin/sh", "-c", f"exit {code}"])[:3:2] == (code, None)


@pytest.mark.parametrize("code", ["systemctl_unreadable", "memory_unreadable"])
def test_a_transient_failure_on_the_identified_unit_is_a_retryable_refusal_never_an_unisolated_run(tree, monkeypatch, code):
    monkeypatch.setattr(cm, "probe", lambda refresh=False: cm.Unavailable(code, "reading failed", True))
    with pytest.raises(cm.CommandMemoryRefused) as refused:
        cm.begin()
    assert refused.value.code == code and refused.value.process_not_started
    monkeypatch.setattr(cm, "probe", lambda refresh=False: cm.Unavailable("needs_reload", "daemon-reload applies it", True))
    assert cm.begin() is None  # a confirmed layout that cannot isolate keeps the old path


def test_stop_is_read_before_a_pressure_refresh_and_a_busy_lock_never_delays_it(tree, monkeypatch):
    host(monkeypatch, 15 * GIB)
    lease = cm.begin()
    (lease.leaf / "memory.events").write_text("max 3\n")
    order = []
    monkeypatch.setattr(lease, "pressure", lambda: order.append("pressure"))
    assert lease.watch(lambda: order.append("stop") or True)() is True
    assert order == ["stop"]  # Stop-now answered: no pressure refresh stands before it
    assert lease.watch(lambda: order.append("stop") or False)() is False and order[-2:] == ["stop", "pressure"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_contended_lock_leaves_the_pressure_refresh_for_a_later_slice(tree, monkeypatch):
    host(monkeypatch, 15 * GIB)
    lease = cm.begin()
    (lease.leaf / "memory.current").write_text("0")
    host(monkeypatch, 60 * GIB)  # the room the others freed
    (lease.leaf / "memory.events").write_text("max 3\n")
    monkeypatch.setattr(cm, "_LOCK_WAIT_SEC", 30.0)
    began = time.monotonic()
    with cm._lock():  # another worker holds the cross-worker lock
        lease.pressure()
    assert time.monotonic() - began < 5 and lease.seen_max == 0 and num(lease.leaf / "memory.max") == 5 * GIB  # no wait, nothing seen
    lease.pressure()  # the next slice, lock free
    assert lease.seen_max == 3 and num(lease.leaf / "memory.max") == 50 * GIB


# What _bias ranks: a 60 GiB build of seven processes, a 34 GiB single process and an empty command.
BIAS_LEAVES = (("cmd-1-aa", 60 * GIB, (101, 102, 103, 104, 105, 106, 107)), ("cmd-2-bb", 34 * GIB, (201,)),
               ("cmd-3-cc", 0, (301,)))


def bias_tree(tree):
    """Everything ``_bias`` reads: the mount table, each leaf's charge and pids, the parent's 100 GiB limit."""
    (tree.proc / "self").mkdir()
    (tree.proc / "self" / "mountinfo").write_text("1 0 0:1 / / rw - cgroup2 cgroup rw\n")
    for name, held, pids in BIAS_LEAVES:
        leaf = tree.commands / name
        leaf.mkdir()
        (leaf / "memory.max").write_text("max")
        (leaf / "memory.current").write_text(str(held))
        (leaf / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids))
    (tree.commands / "memory.max").write_text(str(100 * GIB))


def test_bias_proposes_1000_for_every_process_of_the_largest_command_and_a_share_for_the_others(tree, monkeypatch):
    """The choice, on every platform: ``_bias`` over real files, its /proc reader replaced by a collector of the
    (pid, group, score) it proposes. All seven pids of the build go to the build's group at 1000, 106 and 107 too
    (whether a pid still belongs is the reader's call); the single process goes to its group at 340: its 34 GiB of
    the parent's 100 GiB, in thousandths; the empty command gets none. Reaching a process needs POSIX dir_fd and runs
    on Linux only: this is the algorithm, not a Windows capability (that contract is
    ``test_non_linux_is_unavailable_and_silent``)."""
    bias_tree(tree)
    proposed = []
    monkeypatch.setattr(cm, "_raise_member_score", lambda pid, cgroup, score: proposed.append((pid, cgroup, score)))
    cm._bias(tree.commands)
    build, single = pathlib.Path("/unit/commands/cmd-1-aa"), pathlib.Path("/unit/commands/cmd-2-bb")
    assert sorted(proposed) == [(str(pid), build, 1000) for pid in range(101, 108)] + [("201", single, 340)]


# The reader opens each pid's /proc directory and reads through that fd: POSIX dir_fd. Windows has none and never gets
# here (begin() is None off Linux: test_non_linux_is_unavailable_and_silent); what _bias chooses is the test above.
@pytest.mark.skipif(os.open not in os.supports_dir_fd,
                    reason="POSIX dir_fd: the /proc reader opens each pid's directory and reads through it")
def test_every_process_of_the_largest_whole_command_outranks_any_other_and_scores_are_only_raised(tree, monkeypatch):
    """Review witness: six 10 GiB processes of a 60 GiB build against a 34 GiB single process. Scored per process the
    smaller command would die; every process of the largest command gets 1000 (a whole limit above anyone else), the
    others their command's share of the parent limit. Never lowered, never below 0; an exited pid is skipped."""
    host(monkeypatch, 40 * GIB)
    bias_tree(tree)
    for name, _, pids in BIAS_LEAVES:
        for pid in pids:
            if pid != 106:  # 106 has exited: no /proc entry
                (tree.proc / str(pid)).mkdir()
                (tree.proc / str(pid) / "oom_score_adj").write_text("0")
                membership = "/unit/runtime" if pid == 107 else f"/unit/commands/{name}"
                (tree.proc / str(pid) / "cgroup").write_text(f"0::{membership}\n")
    score = lambda pid: int((tree.proc / str(pid) / "oom_score_adj").read_text())
    cm._bias(tree.commands)
    assert [score(pid) for pid in (101, 102, 103, 104, 105)] == [1000] * 5
    assert score(201) == 34 * 1000 // 100 == 340 and score(301) == 0
    assert score(107) == 0  # a PID from the leaf's stale list now belongs to a runtime worker
    (tree.proc / "201" / "oom_score_adj").write_text("900")  # already higher: never lowered
    cm._bias(tree.commands)
    assert score(201) == 900
    # the 10 GiB processes beat the 34 GiB singleton in the kernel's badness (rss + adj * limit / 1000)
    assert 10 * GIB + 1000 * 100 * GIB // 1000 > 34 * GIB + 340 * 100 * GIB // 1000


def test_the_slow_probe_does_not_hold_the_lock_that_guards_the_cached_verdict(monkeypatch):
    import threading

    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        release.wait(10)
        return cm.Unavailable("platform", "not Linux")

    monkeypatch.setattr(cm, "_probe", slow)
    monkeypatch.setattr(cm, "_PROBED", None)
    monkeypatch.setattr(cm, "_TRANSIENT", None)
    worker = threading.Thread(target=cm.probe)
    worker.start()
    try:
        assert entered.wait(5)
        assert cm._PROBE_LOCK.acquire(timeout=1)  # the cached verdict's lock is free while systemctl runs
        cm._PROBE_LOCK.release()
        assert cm.isolating() is False and cm._observed() is None
    finally:
        release.set()
        worker.join(10)
    assert isinstance(cm._PROBED, cm.Unavailable)
def test_show_gives_user_bus_without_a_login_session(monkeypatch):
    """The packaged service inherits no session env; --user must still reach its own bus."""
    seen = {}

    def fake_run(cmd, **kw):
        seen.clear()
        seen.update(kw)
        seen["cmd"] = cmd

        class _R:
            stdout = "LoadState=loaded\n"

        return _R()

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    assert cm._show("--user", "show", "ouroboros.service")["LoadState"] == "loaded"
    assert seen["env"]["XDG_RUNTIME_DIR"] == f"/run/user/{cm._uid()}"
    assert seen["env"]["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path=/run/user/{cm._uid()}/bus"
    cm._show("show", "user@1.service")  # system scope keeps the inherited env untouched
    assert "XDG_RUNTIME_DIR" not in seen["env"]
    assert "DBUS_SESSION_BUS_ADDRESS" not in seen["env"]


def _bus_denied(monkeypatch):
    def deny(*a):
        raise cm.subprocess.CalledProcessError(1, ["systemctl"], stderr="Failed to connect to bus: Permission denied")

    monkeypatch.setattr(cm, "_show", deny)


def _kernel_tree(tmp_path, monkeypatch, unit_controllers="cpu memory pids", unit_oom="0",
                 mgr_controllers="cpu memory pids", mgr_oom="0"):
    unit = tmp_path / "cg" / "u" / "ouroboros.service"
    unit.mkdir(parents=True)
    (unit / "cgroup.controllers").write_text(unit_controllers + "\n", encoding="ascii")
    (unit / "memory.oom.group").write_text(unit_oom + "\n", encoding="ascii")
    (unit.parent / "cgroup.controllers").write_text(mgr_controllers + "\n", encoding="ascii")
    (unit.parent / "memory.oom.group").write_text(mgr_oom + "\n", encoding="ascii")
    cgroup_tree(tmp_path, monkeypatch, "0::/u/ouroboros.service/runtime\n")
    monkeypatch.setattr(cm, "_unit_dir", lambda _cgroup: unit)
    monkeypatch.setattr(host_memory, "physical_memory", lambda: MemorySnapshot(TOTAL, 40 * GIB))
    (tmp_path / "proc" / "meminfo").write_text("SwapTotal: 0 kB\n", encoding="utf-8")
    return unit


def test_a_denied_user_bus_reads_the_same_layout_from_the_kernel(tmp_path, monkeypatch):
    """Run 44959707977: inside the packaged user service the user bus answered EACCES while the
    layout was in force; the probe must read that layout from the kernel, not refuse the host."""
    _kernel_tree(tmp_path, monkeypatch)
    _bus_denied(monkeypatch)
    assert isinstance(cm._probe(), cm.Enabled)


def test_the_kernel_fallback_refuses_group_kill_and_undelegated_memory(tmp_path, monkeypatch):
    unit = _kernel_tree(tmp_path, monkeypatch, unit_oom="1")
    _bus_denied(monkeypatch)
    assert cm._probe().code == "unit_policy"
    (unit / "memory.oom.group").write_text("0\n", encoding="ascii")
    (unit / "cgroup.controllers").write_text("cpu pids\n", encoding="ascii")
    assert cm._probe().code == "unit_policy"
    (unit / "cgroup.controllers").write_text("cpu memory pids\n", encoding="ascii")
    (unit.parent / "cgroup.controllers").write_text("cpu pids\n", encoding="ascii")
    assert cm._probe().code == "unit_policy"


def test_the_kernel_fallback_without_readable_cgroup_files_stays_retryable(tmp_path, monkeypatch):
    unit = _kernel_tree(tmp_path, monkeypatch)
    (unit / "cgroup.controllers").unlink()
    _bus_denied(monkeypatch)
    state = cm._probe()
    assert state.code == "systemctl_unreadable" and state.transient