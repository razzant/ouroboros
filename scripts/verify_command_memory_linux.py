#!/usr/bin/env python3
"""Real-kernel check of per-command memory leaves (#3792). Linux, cgroup v2 only; otherwise it SKIPs.

Run it only on a DISPOSABLE, exclusive machine (a hosted CI runner, a throwaway VM). It drives real kernel OOM kills
and the kernel may pick any process of the cgroup it is limiting: the isolation guarantee below holds when nothing
else of value shares the host or the delegated subtree. Two layouts:

* default: a test subtree it alone owns, ``<own cgroup>/verify-cm-<pid>``: its ``runtime`` child (this script) and
  ``commands/<leaf>``, the subtree's limit the LOWER of ``UNIT_LIMIT`` and the caller's own. The caller's cgroup is never
  given a limit; every setup step has its own undo, run in reverse on success and on any failure, and the script moves
  back out only once the cgroup it returns to no longer distributes memory. Outside a delegated scope it re-runs itself
  under ``systemd-run --scope -p Delegate=yes -p MemoryMax=512M`` (``sudo`` for root); in a container pass
  ``--own-cgroup``. The product's probe sees no packaged unit here, so scenarios that need ``begin()`` SKIP.
* ``--in-unit``: this script is the main process of the PACKAGED unit's layout (the workflow installs the unit under a
  test user with only ``ExecStart`` and a 512M limit changed). Nothing forces the product's verdict: ``cm.probe()`` itself
  must be Enabled, and every scenario, ``begin()`` included, runs against the unit's real ``commands`` group.

Everything is synthetic: a few processes touch memory, never more than ``UNIT_LIMIT`` (512 MiB) in total. Before each
scenario the host's ``MemAvailable`` and every ancestor cgroup's limit less its usage are read; a scenario whose fixed
load does not fit with a margin SKIPs instead of pressing the machine. ``--require NAME,...`` makes a scenario that did
not pass (skipped included) fail the run: CI must not read a skipped isolation scenario as green.

Documented residual (owner choice 1): when two commands together hit the shared ``commands`` limit the kernel
selects a victim task among the commands (by its OOM badness) and kills that task's whole leaf group
(``memory.oom.group``), so a healthy sibling can die; the service (runtime) never does.

Prints one JSON summary (passed / skipped / failed scenarios); ``--json-out FILE`` also writes it. Exit 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
UNIT_LIMIT = 512 * MIB
SCOPE_ENV = "VERIFY_CM_SCOPE"
# Touch ``argv[1]`` MiB in 1 MiB steps, print ``ready``, then hold: a bounded synthetic allocation.
ALLOC = ("import sys,time\nkeep=[]\nfor _ in range(int(sys.argv[1])):\n keep.append(b'x'*(1<<20));time.sleep(.003)\n"
         "print('ready',flush=True);time.sleep(300)")
HOLD = "import time;time.sleep(300)"
MARGIN = 64 * MIB
NEED_MIB = {"page_rounded_readback": 0, "leaf_oom_kills_only_its_group": 64, "parent_pressure_selects_among_leaves": 130,
            "external_sigkill_is_not_oom": 2, "two_healthy_commands_start": 20, "tiny_remainder_still_starts": 24,
            "parent_limit_covers_held": 44, "multi_process_build_is_the_victim": 120}  # each scenario's fixed synthetic load (the filler of the tiny remainder is sized to the room, below)


def headroom(start: Path) -> int:
    """Bytes the synthetic loads may take: the host's MemAvailable and, up from ``start``, every ancestor cgroup's limit
    less its usage; the smallest."""
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    room = int(meminfo["MemAvailable"].split()[0]) * 1024
    path = start
    while (path / "cgroup.controllers").exists():
        try:
            limit, used = (path / "memory.max").read_text().strip(), int((path / "memory.current").read_text())
            if limit.isdigit():
                room = min(room, int(limit) - used)
        except (OSError, ValueError):
            pass
        path = path.parent
    return room


class Skip(Exception):
    pass


def emit(results, status, reason=""):
    missing = [n for n in ARGS.require.split(",") if n and not any(r["name"] == n and r["status"] == "passed" for r in results)]
    if missing and status != "failed":
        status, reason = "failed", f"required scenarios did not pass: {', '.join(missing)}" + (f" ({reason})" if reason else "")
    summary = {"status": status, "reason": reason, "kernel": os.uname().release if hasattr(os, "uname") else "",
               "counts": {k: sum(r["status"] == k for r in results) for k in ("passed", "skipped", "failed")},
               "scenarios": results}
    text = json.dumps(summary, indent=2)
    if ARGS.json_out:
        Path(ARGS.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if status == "failed" else 0


def rewrap() -> int | None:
    """Re-run inside a delegated transient scope; None when that is not possible (the caller SKIPs)."""
    run = shutil.which("systemd-run")
    if run is None:
        return None
    user = [] if os.geteuid() == 0 else ["--user"]
    base = [run, *user, "--scope", "--quiet", "-p", "Delegate=yes"]
    if subprocess.run([*base, "true"], capture_output=True, timeout=30).returncode != 0:
        return None
    argv = [*base, "-p", f"MemoryMax={UNIT_LIMIT // MIB}M", "-p", "MemorySwapMax=0", "env", f"{SCOPE_ENV}=1",
            sys.executable, *sys.argv]
    return subprocess.run(argv).returncode


def ev(path: Path) -> dict:
    out = {}
    for line in (path / "memory.events.local").read_text().splitlines():
        key, _, value = line.partition(" ")
        out[key] = int(value)
    return out


def join(leaf: Path):
    target = str(leaf / "cgroup.procs")

    def move():  # runs in the forked child: writing "0" moves the writer itself
        fd = os.open(target, os.O_WRONLY)
        os.write(fd, b"0")
        os.close(fd)
    return move


class Rig:
    def __init__(self, cm, base: Path, commands: Path):
        self.cm, self.base, self.commands, self.procs = cm, base, commands, []

    def leaf(self, name: str, limit: int | str, oom_group=True) -> Path:
        leaf = self.commands / f"cmd-{name}"
        leaf.mkdir()
        (leaf / "memory.max").write_text(str(limit))
        if (leaf / "memory.swap.max").exists():
            (leaf / "memory.swap.max").write_text("0")
        (leaf / "memory.oom.group").write_text("1" if oom_group else "0")
        return leaf

    def spawn(self, leaf: Path, code: str, *args, **kw):
        proc = subprocess.Popen([sys.executable, "-c", code, *map(str, args)], stdout=subprocess.PIPE, bufsize=0,
                                preexec_fn=join(leaf), **kw)
        self.procs.append(proc)
        return proc

    @staticmethod
    def ready(proc, timeout=30.0) -> bool:
        return bool(select.select([proc.stdout], [], [], timeout)[0]) and proc.stdout.readline().strip() == b"ready"

    @staticmethod
    def dead(proc, timeout=20.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc.poll() is not None:
                return True
            time.sleep(.05)
        return False

    def reset(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for pipe in (proc.stdout, proc.stdin):
                if pipe:
                    pipe.close()
        self.procs.clear()
        for leaf in self.commands.glob("cmd-*"):
            self.rmdir(leaf)
        self.cm._FINISHED_BUSY.clear()
        (self.commands / "memory.max").write_text("max")

    @staticmethod
    def rmdir(path: Path):
        for _ in range(40):
            try:
                path.rmdir()
                return
            except OSError:
                time.sleep(.05)

    def gated(self, lease, code: str, *args, executable=None):
        """A command through the product's stdin gate, placed in ``lease``'s leaf, then released."""
        cm = self.cm
        status_r, status_w = os.pipe()
        try:
            proc = subprocess.Popen(cm.gate_argv([executable or sys.executable, "-c", code, *map(str, args)], status_w), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, bufsize=0, pass_fds=(status_w,))
        finally:
            os.close(status_w)
        try:
            self.procs.append(proc)
            lease.place(proc.pid)
            cm.release_gate(proc, {"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
            failed = cm.exec_verdict(status_r, 30)
        finally:
            os.close(status_r)
        assert failed is None, f"the gate's exec did not confirm: {'timed out' if failed == cm.EXEC_TIMEOUT else f'errno {failed}'}"
        return proc


def scenarios(rig: Rig):
    cm, commands, base = rig.cm, rig.commands, rig.base
    from ouroboros.host_memory import physical_memory

    swap_ok = (commands / "memory.swap.max").exists() or cm._meminfo("SwapTotal") == 0

    def packaged_layout():
        """The product's own verdict, never forced: scenarios that call ``begin()`` need the packaged unit's layout."""
        state = cm.probe()
        if not isinstance(state, cm.Enabled):
            raise Skip(f"the product's probe says {getattr(state, 'code', state)} here: run with --in-unit under the packaged unit")

    def assert_largest_bias(*procs):
        cm._bias(commands)
        scores = {p.pid: int(Path(f"/proc/{p.pid}/oom_score_adj").read_text()) for p in procs}
        assert all(score == 1000 for score in scores.values()), f"missing bias before pressure: {scores}"

    def page_rounded_readback():
        leaf = rig.leaf("v-page", "max")
        for raw in (24 * MIB + 1234, 24 * MIB + 4095, 8 * MIB):
            kept = cm._write_max(leaf / "memory.max", raw)
            got = int((leaf / "memory.max").read_text())
            assert got == kept == cm._floor(raw) and got % cm._page_size() == 0, (raw, kept, got)
        return "memory.max reads back the page-rounded value the product writes"

    def leaf_oom_kills_only_its_group():
        if not swap_ok:
            raise Skip("swap exists and memory.swap.max is absent")
        cm._write_max(commands / "memory.max", 300 * MIB)
        x, y = rig.leaf("v-x", 40 * MIB), rig.leaf("v-y", 40 * MIB)
        c0, b0 = ev(commands), ev(base)
        healthy = rig.spawn(y, ALLOC, 12)
        assert rig.ready(healthy), "sibling did not start"
        pid_x = rig.spawn(x, HOLD)
        grower = rig.spawn(x, ALLOC, 200)
        assert rig.dead(grower) and grower.returncode == -9, f"grower rc={grower.returncode}"
        assert rig.dead(pid_x), "oom.group did not take the leaf's other process"
        assert healthy.poll() is None, "sibling leaf was killed"
        facts = cm.outcome(x, grower.pid, c0["oom"])
        assert facts["victim_in_leaf"] and facts["leaf_local_oom"] >= 1 and facts["commands_local_oom"] == 0, facts
        assert ev(base)["oom"] == b0["oom"] and ev(y)["oom_kill"] == 0, "parent or sibling counters moved"
        return facts

    def parent_pressure_selects_among_leaves():
        """Shared-limit pressure kills only command leaves; the runtime/service subgroup never OOMs (the required
        guarantee). Which leaf dies is best-effort per the owner's accepted residual: the ``oom_score_adj`` bias prefers
        the largest, but in a two-command collision the kernel may take the other leaf (or both) as well."""
        if not swap_ok:
            raise Skip("swap exists and memory.swap.max is absent")
        cm._write_max(commands / "memory.max", 96 * MIB)
        small, large = rig.leaf("v-a", 96 * MIB), rig.leaf("v-b", 96 * MIB)
        c0, b0 = ev(commands), ev(base)
        big = rig.spawn(large, ALLOC, 60)  # the larger, healthy one
        assert rig.ready(big), "the larger command did not start"
        assert_largest_bias(big)
        grower = rig.spawn(small, ALLOC, 70)  # grows past what the shared limit leaves: 60 + 70 > 96
        end = time.monotonic() + 30
        while time.monotonic() < end and big.poll() is None and grower.poll() is None:
            cm._bias(commands)
            time.sleep(.02)
        time.sleep(1.0)
        dead = [n for n, p in (("larger_healthy", big), ("smaller_grower", grower)) if p.poll() is not None]
        assert dead, f"no command leaf died under shared-limit pressure, got {dead}"
        c1 = ev(commands)
        assert c1["oom"] - c0["oom"] >= 1 and c1["oom_kill"] == c0["oom_kill"], "parent event not at commands"
        assert ev(small)["oom"] == 0 and ev(large)["oom"] == 0, "a leaf limit was hit, not the parent's"
        assert ev(base)["oom"] == b0["oom"], "the runtime/service subgroup OOMed under command pressure"
        assert ev(large)["oom_kill"] + ev(small)["oom_kill"] >= 1, "no leaf recorded the OOM kill"
        return {"victims": dead, "largest_preferred": "larger_healthy" in dead, "service_untouched": True}

    def external_sigkill_is_not_oom():
        packaged_layout()
        lease = cm.begin()
        assert lease is not None, "begin() returned None: layout not recognised"
        proc = rig.gated(lease, HOLD)
        time.sleep(.5)
        os.kill(proc.pid, signal.SIGKILL)
        assert rig.dead(proc) and proc.returncode == -9
        facts = lease.finish(proc.pid)
        assert facts["victim_in_leaf"] is False and facts["leaf_local_oom"] == 0 and not facts["unknown"], facts
        assert cm.facts_worth_publishing(facts, -9) is None, "an external SIGKILL was reported as an OOM"
        return facts

    def two_healthy_commands_start():
        packaged_layout()
        first, second = cm.begin(), cm.begin()
        assert first and second and first.leaf != second.leaf
        procs = [rig.gated(lease, ALLOC, 8) for lease in (first, second)]
        assert all(rig.ready(p) for p in procs), "a healthy command did not start"
        for lease in (first, second):
            assert int((lease.leaf / "memory.max").read_text()) % cm._page_size() == 0
        return "both started in separate leaves"

    def tiny_remainder_still_starts():
        packaged_layout()
        snap = physical_memory()
        extra = (snap.available_bytes - snap.reserve_bytes) - 24 * MIB  # leave ~24 MiB of room above the reserve
        if extra <= 8 * MIB:
            raise Skip("unit too small to shape a remainder")
        if extra > UNIT_LIMIT:
            raise Skip("the room is larger than this verifier's bound: it never fills more than UNIT_LIMIT")
        (commands / "memory.max").write_text("max")  # the literal ``max``: ``_write_max`` takes a byte count
        fill = rig.spawn(rig.leaf("v-fill", "max"), ALLOC, extra // MIB)
        assert rig.ready(fill, 60), "filler did not start"
        snap = physical_memory()
        room = snap.available_bytes - snap.reserve_bytes
        if not 0 < room < 48 * MIB:
            raise Skip(f"could not shape a tiny remainder (room={room})")
        ooms = lambda: int(dict(line.split() for line in (commands / "memory.events.local").read_text().splitlines()).get("oom", 0))
        before = ooms()
        lease = cm.begin()
        assert lease is not None, "begin() refused a command with a positive remainder"
        quota = int((lease.leaf / "memory.max").read_text())
        proc = rig.gated(lease, ALLOC, 2)
        assert quota > 0 and rig.ready(proc), f"tiny-remainder command did not start (quota={quota})"
        time.sleep(1)  # the stationary filler keeps its headroom: the parent is held + room, never clamped to held
        assert fill.poll() is None, "the stationary filler died: the parent left it no headroom"
        assert ooms() == before, "an OOM at the commands parent while the group was stationary"
        return {"room_mib": room // MIB, "leaf_limit_mib": quota // MIB}

    def multi_process_build_is_the_victim():
        """The LARGEST WHOLE command dies, even when it is many small processes: a 4 x 16 MiB build (64 MiB) against a
        single process that reaches ~46 MiB when the shared 110 MiB limit runs out. Scored per process, the singleton
        would lose (46 + 46 > 16 + 64); the product's whole-command bias (``cm._bias``) makes the build the victim."""
        if not swap_ok:
            raise Skip("swap exists and memory.swap.max is absent")
        cm._write_max(commands / "memory.max", 110 * MIB)
        build_leaf, single_leaf = rig.leaf("v-build", 110 * MIB), rig.leaf("v-single", 110 * MIB)
        c0, b0 = ev(commands), ev(base)
        build = [rig.spawn(build_leaf, ALLOC, 16) for _ in range(4)]
        assert all(rig.ready(proc) for proc in build), "the build's processes did not start"
        assert_largest_bias(*build)
        single = rig.spawn(single_leaf, ALLOC, 60)  # 64 + 60 > 110: the collision comes at ~46 MiB
        end = time.monotonic() + 30
        while time.monotonic() < end and single.poll() is None and all(proc.poll() is None for proc in build):
            cm._bias(commands)
            time.sleep(.02)
        time.sleep(1.0)
        assert all(rig.dead(proc, 5) for proc in build), "the larger multi-process build was not the victim"
        assert single.poll() is None, "the smaller single-process command died instead of the larger build"
        assert ev(commands)["oom"] - c0["oom"] >= 1 and ev(single_leaf)["oom_kill"] == 0, "the collision was not at the commands limit"
        assert ev(base)["oom"] == b0["oom"], "the runtime's ancestor, not just commands, reached OOM"
        return {"victim": "multi_process_build"}

    def parent_limit_covers_held():
        """The parent is the room commands may hold, never below their nonreclaimable charge on a quiet host."""
        packaged_layout()
        (commands / "memory.max").write_text("max")  # the literal ``max``: ``_write_max`` takes a byte count
        holder = rig.spawn(rig.leaf("v-hold", "max"), ALLOC, 40)
        assert rig.ready(holder), "holder did not start"
        lease = cm.begin()
        assert lease is not None
        held = cm._held(commands)[1]
        written = int((commands / "memory.max").read_text())
        assert written >= held - cm._page_size(), (written, held)
        assert holder.poll() is None, "the holder died when the parent was written"
        return {"parent_mib": written // MIB, "held_mib": held // MIB}

    def packaged_probe_is_enabled():
        if not ARGS.in_unit:
            raise Skip("only meaningful in the packaged unit's layout (--in-unit)")
        state = cm.probe(refresh=True)
        assert isinstance(state, cm.Enabled), f"the real probe says {state}"
        return {"unit": state.unit, "commands": str(state.commands)}

    def packaged_gate_without_system_python():
        if not ARGS.in_unit:
            raise Skip("only meaningful in the packaged unit's layout (--in-unit)")
        from unittest.mock import patch

        packaged_layout()
        lease = cm.begin()
        assert lease is not None
        embedded = Path(sys.executable).absolute()
        bundle = embedded.parents[2]
        assert bundle.name == "_internal", f"expected packaged interpreter layout, got {embedded}"
        with patch.multiple(sys, frozen=True, _MEIPASS=str(bundle), executable=str(bundle.parent / "Ouroboros"),
                            create=True), patch.dict(os.environ, PATH="", OUROBOROS_BUNDLE_DIR=""):
            proc = rig.gated(lease, ALLOC, 2, executable=str(embedded))
        assert rig.ready(proc), "the command did not start through the bundle's Python without PATH"
        return {"interpreter": str(embedded), "path_empty": True, "launcher_metadata": "synthetic"}

    return [page_rounded_readback, packaged_probe_is_enabled, leaf_oom_kills_only_its_group, parent_pressure_selects_among_leaves,
            packaged_gate_without_system_python, external_sigkill_is_not_oom, two_healthy_commands_start, tiny_remainder_still_starts,
            parent_limit_covers_held, multi_process_build_is_the_victim]


def run_scenarios(rig: Rig, results: list, start: Path) -> None:
    """Every scenario in order, each from a reset rig; one that does not fit the machine's headroom SKIPs."""
    for scenario in scenarios(rig):
        rig.reset()
        entry = {"name": scenario.__name__, "status": "passed", "detail": ""}
        need, room = NEED_MIB.get(scenario.__name__, 0) * MIB + MARGIN, headroom(start)
        try:
            if room < need:
                raise Skip(f"needs {need // MIB} MiB of headroom, the host and its cgroups allow {room // MIB} MiB")
            entry["detail"] = scenario()
        except Skip as exc:
            entry.update(status="skipped", detail=str(exc))
        except AssertionError as exc:
            entry.update(status="failed", detail=f"assertion: {exc}")
        except Exception as exc:  # noqa: BLE001 - a scenario's own error is a failed scenario, not a crash
            entry.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
        results.append(entry)
        print(f"{entry['status'].upper():8} {entry['name']}", file=sys.stderr)


def verdict(results: list) -> int:
    failed = any(r["status"] == "failed" for r in results)
    return emit(results, "failed" if failed else "passed" if any(r["status"] == "passed" for r in results) else "skipped")


def in_unit(cm) -> int:
    """This process is the packaged unit's main process (``<unit>/runtime``): the product's real probe decides, the unit's
    own ``commands`` group is prepared by the product's ``begin()``, and no layout is built or forced here."""
    state = cm.probe(refresh=True)
    if not isinstance(state, cm.Enabled):
        return emit([{"name": "packaged_probe_is_enabled", "status": "failed", "detail": f"the real probe says {state}"}], "failed")
    lease = cm.begin()  # creates ``commands``, enables its controller and sets its swap limit, exactly as a real start does
    if lease is None:
        return emit([{"name": "packaged_probe_is_enabled", "status": "failed", "detail": "begin() returned None under an Enabled probe"}], "failed")
    lease.finish(0)
    rig, results = Rig(cm, state.unit_root, state.commands), []
    try:
        run_scenarios(rig, results, state.unit_root)
    finally:
        rig.reset()
    return verdict(results)


def main() -> int:
    if not sys.platform.startswith("linux"):
        return emit([], "skipped", "not Linux")
    if not any(" cgroup2 " in line for line in Path("/proc/self/mountinfo").read_text().splitlines()):
        return emit([], "skipped", "cgroup v2 is not mounted")
    sys.path.insert(0, str(ROOT))
    from ouroboros import command_memory as cm

    if ARGS.in_unit:
        return in_unit(cm)
    if not (os.environ.get(SCOPE_ENV) or ARGS.own_cgroup):
        code = rewrap()
        if code is None:
            return emit([], "skipped", "no delegated cgroup: systemd-run --scope -p Delegate=yes is unavailable "
                        "(run as root on systemd, or pass --own-cgroup inside a container)")
        return code
    own_path = next((line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")), None)
    try:
        own = cm._unit_dir(own_path or "")
    except (OSError, ValueError):
        own = None
    if own is None or not (own / "cgroup.controllers").exists():
        return emit([], "skipped", "this process's cgroup directory is not visible")
    if "memory" not in (own / "cgroup.controllers").read_text().split():
        return emit([], "skipped", "the memory controller is not available in this cgroup")
    owned = Subtree(own, own_path.rstrip("/"))
    results = []
    try:
        try:
            owned.build()
        except Skip as exc:
            return emit([], "skipped", str(exc))
        except OSError as exc:
            return emit([], "skipped", f"cgroup tree is not writable here: {exc}")
        rig = Rig(cm, owned.unit, owned.commands)  # the product's verdict is left alone: begin() scenarios SKIP in this layout
        try:
            run_scenarios(rig, results, owned.unit)
        finally:
            rig.reset()
    finally:
        left = owned.unwind()
        if left:
            print(f"left behind (not restorable here): {'; '.join(left)}", file=sys.stderr)
    return verdict(results)


class Subtree:
    """The test subtree and the steps that built it. Each step records its own undo, so a failure partway (or the end of
    the run) unwinds exactly what was done, in reverse, each undo on its own: one that fails does not stop the others and
    is named. The caller's cgroup is never limited; the script leaves ``runtime`` only after ``own`` no longer distributes
    memory (a cgroup that does cannot hold processes)."""

    def __init__(self, own: Path, own_path: str):
        self.own, self.unit = own, own / f"verify-cm-{os.getpid()}"
        self.path, self.runtime, self.commands = f"{own_path}/{self.unit.name}", self.unit / "runtime", self.unit / "commands"
        self.undo = []

    def step(self, do, undo=None):
        do()
        if undo is not None:
            self.undo.append(undo)

    @staticmethod
    def _control(path: Path, text: str):
        return lambda: path.write_text(text)

    def _return(self):
        """Move back into the caller's cgroup, which holds processes again only if it distributes no memory."""
        if "memory" in (self.own / "cgroup.subtree_control").read_text().split():
            raise OSError("the caller's cgroup still distributes memory, so it cannot hold this process")
        (self.own / "cgroup.procs").write_text(str(os.getpid()))

    def build(self):
        own_control, unit_control = self.own / "cgroup.subtree_control", self.unit / "cgroup.subtree_control"
        self.step(self.unit.mkdir, lambda: Rig.rmdir(self.unit))  # exclusive: an existing one is not ours
        self.step(self.runtime.mkdir, lambda: Rig.rmdir(self.runtime))
        self.step(self._control(self.runtime / "cgroup.procs", "0"), self._return)
        if "memory" not in own_control.read_text().split():  # only what this run turned on is turned off again
            self.step(self._control(own_control, "+memory"), self._control(own_control, "-memory"))
        if "memory" not in (self.unit / "cgroup.controllers").read_text().split():
            raise Skip("the memory controller cannot be delegated to a child of this cgroup")
        self.step(self._control(unit_control, "+memory"), self._control(unit_control, "-memory"))
        limit = Path(self.own / "memory.max")
        caller = int(limit.read_text()) if limit.exists() and limit.read_text().strip().isdigit() else UNIT_LIMIT
        (self.unit / "memory.max").write_text(str(min(UNIT_LIMIT, caller)))  # the lower limit; the caller's own is never written
        self.step(self.commands.mkdir, lambda: Rig.rmdir(self.commands))
        self.step(self._control(self.commands / "cgroup.subtree_control", "+memory"))

    def unwind(self) -> list:
        left = []
        for undo in reversed(self.undo):
            try:
                undo()
            except OSError as exc:
                left.append(f"{type(exc).__name__}: {exc}")
        self.undo.clear()
        return left


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", help="also write the JSON summary here")
    parser.add_argument("--own-cgroup", action="store_true", help="use this process's cgroup as the parent of the test subtree (containers)")
    parser.add_argument("--in-unit", action="store_true", help="run as the main process of the packaged unit; the product's real probe decides")
    parser.add_argument("--require", default="", help="comma-separated scenario names that must pass: a skip or failure fails the run")
    ARGS = parser.parse_args()
    sys.exit(main())
