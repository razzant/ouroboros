"""The local battery entry point keeps the gate's recipe and a bounded serial overlap."""

import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time
import types

import pytest

from ouroboros.preflight_inventory import SELECT
from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS
from ouroboros.safe_test_supervisor import _GRACE_SEC as SAFE_TEST_INT_WINDOW_SEC
from tests import conftest
from tests.test_selected_test_cli import plugin_scope

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_tests():
    spec = importlib.util.spec_from_file_location("run_tests_script", REPO / "scripts" / "run_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def recorded(run_tests, monkeypatch):
    """A battery here names no durations table unless a test records one itself (the real reader, returned,
    reads the evidence beside the running test session)."""
    real = run_tests.recorded_durations
    monkeypatch.setattr(run_tests, "recorded_durations", lambda: (None, "no recorded durations"))
    return real


@pytest.fixture
def standalone_budget(monkeypatch, tmp_path):
    """A standalone owner on a quiet host; returns its private ledger directory."""
    from ouroboros import host_memory, test_parallelism

    # These command-shape cases model the standalone operator entrypoint.
    # Real nested runners retain their inherited one-slot allocation.
    monkeypatch.delenv(test_parallelism.NESTED_WORKERS_ENV, raising=False)
    monkeypatch.setattr(test_parallelism, "physical_memory", lambda:
                        host_memory.MemorySnapshot(100 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(test_parallelism, "other_runs", lambda: test_parallelism.HostRuns())
    monkeypatch.setattr(test_parallelism, "_ledger_dir", lambda: tmp_path / "slots")
    return tmp_path / "slots"


def _started(argv, workers=3):
    """A pass's command as ``_run`` starts it on a reservation admitting ``workers`` (a builder gets that Admission)."""
    from ouroboros import test_parallelism

    return argv(test_parallelism.Admission(workers, test_parallelism.HostRuns())) if callable(argv) else argv


def _item(nodeid, *, serial, parent=None):
    """A collected item; ``serial`` is None (no mark), "" (a bare mark) or the resource the mark names."""
    markers = [] if serial is None else [types.SimpleNamespace(name="serial", args=(serial,) if serial else ())]
    scope = parent or nodeid.split("[", 1)[0].rsplit("::", 1)[0]
    item = types.SimpleNamespace(nodeid=nodeid, parent=types.SimpleNamespace(nodeid=scope), stash={})
    item.get_closest_marker = lambda name: next((m for m in markers if m.name == name), None)
    return item


def _config(loadgroup=True, seconds=None):
    """A config as an xdist WORKER holds it: `--dist` reads "no" there, `loadgroup` carries the mode and the
    controller's reading of the durations table arrives in `workerinput`."""
    return types.SimpleNamespace(option=types.SimpleNamespace(dist="no", loadgroup=loadgroup),
                                 workerinput={"lane_group_seconds": seconds or {}})


def _group(item):
    return item.stash[conftest._LANE_GROUP]


def test_the_battery_names_the_durations_table_its_groups_are_ordered_by(run_tests, recorded, monkeypatch, tmp_path,
                                                                        capsys, standalone_budget):
    """The table is the owner's persisted one beside the evidence parent, named for the battery by
    `point_at_durations` and brought up to date by every session at its end; the battery says which table orders
    its xdist groups. A caller's own name for the table stands; without an evidence root nobody owns one."""
    from tests.ci_verdicts import TEST_DURATIONS_ENV, platform_name

    system = platform_name(run_tests.sys.platform)
    evidence = tmp_path / ".test-evidence"
    evidence.mkdir()
    (evidence / "test-durations.json").write_text(json.dumps({
        "format": 1, "platforms": {system: {"tests/test_a.py": 5000, "tests/test_s.py": 5000}},
        "serial": {system: {"tests/test_s.py": 5000}}, "source": {"run_id": "session 2026-11-05T12:00:00Z", "sha": ""},
        "ui_nodes": {}}), encoding="utf-8")
    monkeypatch.delenv("OUROBOROS_CI_DURATIONS", raising=False)
    monkeypatch.delenv(TEST_DURATIONS_ENV, raising=False)
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(evidence / "run-now"))
    run_tests.point_at_durations()
    assert os.environ[TEST_DURATIONS_ENV] == str(evidence / "test-durations.json")
    monkeypatch.setattr(run_tests, "recorded_durations", recorded)
    run_tests.battery_commands()
    assert "=== xdist groups ordered by session 2026-11-05T12:00:00Z\n" in capsys.readouterr().out
    monkeypatch.setenv(TEST_DURATIONS_ENV, "mine.json")
    run_tests.point_at_durations()
    assert os.environ[TEST_DURATIONS_ENV] == "mine.json"
    monkeypatch.delenv(TEST_DURATIONS_ENV)
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT")
    run_tests.point_at_durations()
    assert TEST_DURATIONS_ENV not in os.environ and recorded() == (None, "no recorded durations")
    run_tests.battery_commands()
    assert "=== xdist groups ordered by no recorded durations\n" in capsys.readouterr().out


def test_the_battery_is_the_gates_two_passes(run_tests, monkeypatch, standalone_budget):
    """The same marker split and flag vector as the gate and CI, and nothing of the battery's own."""
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    (parallel_label, parallel), (serial_label, serial) = run_tests.battery_commands()
    assert (parallel_label, serial_label) == ("parallel", "serial")
    assert parallel[3:] == ["tests/", "-m", f"not exclusive and {LANE_EXCLUSION_EXPR}", *PARALLEL_PASS_FLAGS]
    assert serial[3:] == ["tests/", "-m", f"exclusive and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS]


@pytest.mark.parametrize("host, workers", [
    (None, 3),
    # An unleased pytest root elsewhere on the host takes one slot off this run, and the runner says so.
    ("busy", 2),
])
def test_the_worker_override_steers_xdist_like_the_gate(run_tests, monkeypatch, standalone_budget, capsys, host, workers):
    from ouroboros import process_containment, test_parallelism

    seen = {}
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3")
    if host:
        monkeypatch.setattr(test_parallelism, "other_runs",
                            lambda: test_parallelism.HostRuns(strangers=((700, "python3 -m pytest -n 8"),)))

    class Owner:
        def __init__(self, **options):
            seen["options"] = options

        def spawn(self, argv, *, cwd, env):
            seen.update(env=env, argv=argv, cwd=cwd)
            # While the tree runs, this owner's row in the host ledger holds exactly what it admitted.
            seen["leases"] = [json.loads(path.read_text(encoding="utf-8")) for path in standalone_budget.glob("*.json")]
            return types.SimpleNamespace(wait=lambda **kwargs: 0)

        def reap(self):
            seen["reaped"] = True
            return ""

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    assert run_tests._run("x", [run_tests.sys.executable, "-m", "pytest", "-n", "8"], run_tests.REPO) == 0
    assert seen["argv"][-2:] == ["-n", str(workers)]
    out = capsys.readouterr().out
    assert f"=== x: pytest -n {workers}\n" in out
    assert ("=== x: 2 worker(s); 1 unleased pytest/node root(s) counted one slot each: 700 python3 -m pytest -n 8\n"
            in out) is bool(host)
    assert seen["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == str(workers)
    assert seen["env"]["OURO_TEST_WORKER_BUDGET"] == str(workers)
    assert seen["options"] == {"lifeline": True, "overload": True}
    assert seen["reaped"] and seen["closed"]
    (lease,) = seen["leases"]
    assert lease["pid"] == os.getpid() and lease["workers"] == workers and lease["command"] == "run_tests.py x"
    assert list(standalone_budget.glob("*.json")) == [], "the lease outlived its run"


def test_a_pass_waits_out_critical_memory_pressure_says_so_and_then_starts_by_itself(
        run_tests, monkeypatch, capsys, standalone_budget):
    """A new pass starts nothing while the kernel confirms critical pressure: the runner names the wait and its end
    with each reading's date, holds no ledger row meanwhile, and starts the tree at the first reading that is not
    critical, with the admission that reading allows (owner choice, 2026-11-14)."""
    from ouroboros import config, host_memory, process_containment, test_parallelism

    seen, verdicts = {}, [True, True, False, False]  # first and last: the admissions; between: the wait's readings
    monkeypatch.setattr(config, "TEST_PROCESS_SAMPLE_SEC", 0.01)

    def physical_memory():
        seen.setdefault("rows", []).append(len(list(standalone_budget.glob("*.json"))))
        return host_memory.MemorySnapshot(100 * 1024 ** 3, 100 * 1024 ** 3, verdicts.pop(0))

    monkeypatch.setattr(test_parallelism, "physical_memory", physical_memory)

    class Owner:
        def __init__(self, **options):
            pass

        def spawn(self, argv, *, cwd, env):
            seen["leases"] = [json.loads(path.read_text(encoding="utf-8")) for path in standalone_budget.glob("*.json")]
            return types.SimpleNamespace(wait=lambda **kwargs: 0)

        def reap(self):
            return ""

        def close(self):
            pass

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3")
    assert run_tests._run("focused", [run_tests.sys.executable, "-m", "pytest", "-n", "8"], REPO) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("=== focused: ")]
    assert lines[0].startswith("=== focused: waiting to start: the kernel reports critical memory pressure (read ")
    assert lines[1].startswith("=== focused: memory pressure no longer critical (read ") and "pytest -n 3" in lines[2]
    assert seen["rows"] == [0, 0, 0, 0] and not verdicts, "no row while it waited"
    assert [lease["workers"] for lease in seen["leases"]] == [3], "the tree started on the cleared admission"
    assert list(standalone_budget.glob("*.json")) == []


def test_a_pass_sized_by_its_workers_is_built_from_the_one_admission_it_reserved(
        run_tests, monkeypatch, standalone_budget):
    """Carbon r1: the warm's shards and the Node lane's concurrency were read before the pass reserved, so after a
    wait for memory during which another run took seven of eight slots the ledger held one while the warm (which no
    bound reaches) started eight. Each is now built once, from the admission its reservation returned, wait included."""
    from ouroboros import host_memory, preflight_node, process_containment, test_parallelism

    state = {"cleared": False}
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    monkeypatch.setattr(test_parallelism, "physical_memory", lambda: host_memory.MemorySnapshot(
        100 * 1024 ** 3, 100 * 1024 ** 3, not state["cleared"]))
    monkeypatch.setattr(test_parallelism, "other_runs", lambda: test_parallelism.HostRuns(7 * state["cleared"]))

    def until_not_critical(stop=None):  # meanwhile pressure clears and another run reserves seven slots
        state["cleared"] = True
        return "2026-11-15T00:00:00Z", "not_critical"

    monkeypatch.setattr(test_parallelism, "_until_not_critical", until_not_critical)
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/one.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "node")
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    started = []

    class Owner:
        def __init__(self, **options):
            pass

        def spawn(self, argv, *, cwd, env):
            leases = [json.loads(path.read_text(encoding="utf-8")) for path in standalone_budget.glob("*.json")]
            started.append((argv, [lease["workers"] for lease in leases]))
            return types.SimpleNamespace(wait=lambda **kwargs: 0)

        def reap(self):
            return ""

        def close(self):
            pass

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    for lane in (run_tests._warm_bytecode, run_tests._node_lane):
        state["cleared"] = False
        assert lane() in (None, (0, "node"))  # the Node lane also returns the Node it ran
    (warm, warm_held), (node, node_held) = started
    assert (warm[-2:], warm_held) == (["--workers", "1"], [1])
    assert ([arg for arg in node if arg.startswith("--test-concurrency")], node_held) == (["--test-concurrency=1"], [1])
    admissions, state["cleared"] = [], False
    assert run_tests._run("parallel", lambda admitted: admissions.append(admitted) or [
        run_tests.sys.executable, "-m", "pytest", "-n", "8"], run_tests.REPO) == 0
    assert started[-1][0][-2:] == ["-n", "1"] and started[-1][1] == [1]
    (admitted,) = admissions
    assert admitted.workers == 1 and admitted.waited[1:] == ("2026-11-15T00:00:00Z", "not_critical")


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT/SIGTERM delivery to one process; Windows Ctrl-C reaches a "
                    "console group and its Stop is the Job")
@pytest.mark.parametrize("ending", ["clears", "interrupt", "stop"])
def test_a_real_runner_waits_out_pressure_and_then_runs_or_is_ended_with_nothing_started(tmp_path, ending):
    """A real runner process waiting out critical pressure holds neither the ledger's lock (another run reserves
    meanwhile) nor a row. When a reading clears it starts its own small child through the real container; an
    interrupt (Ctrl-C, safe_test's forwarded INT) ends the wait at once with exit 130, said; a stop (a task's Stop
    kills its worker) ends it with no row left. Ended either way, its command never ran."""
    from ouroboros import platform_layer as pl, test_parallelism

    ledger, ran = tmp_path / "slots", tmp_path / "ran"
    readings = [True] * 40 + [False] if ending == "clears" else [True]  # clears after about two seconds
    child = textwrap.dedent(f"""
        import os, runpy, sys
        sys.path.insert(0, {str(REPO)!r})
        from pathlib import Path
        from ouroboros import config, host_memory, test_parallelism as budget
        os.environ.pop(budget.NESTED_WORKERS_ENV, None)  # a run of its own, not a test's child
        budget._ledger_dir = lambda: Path({str(ledger)!r})
        budget._process_table = lambda: ([], "")
        verdicts = {readings!r}
        budget.physical_memory = lambda: host_memory.MemorySnapshot(
            2 ** 40, 2 ** 40, verdicts.pop(0) if len(verdicts) > 1 else verdicts[0])
        config.TEST_PROCESS_SAMPLE_SEC = 0.05
        runner = runpy.run_path({str(REPO / "scripts" / "run_tests.py")!r})
        raise SystemExit(runner["_run"]("focused", [sys.executable, "-c", "open({str(ran)!r}, 'w')"], Path.cwd()))
    """)
    env = {key: value for key, value in os.environ.items() if key != test_parallelism.NESTED_WORKERS_ENV}
    proc = subprocess.Popen([sys.executable, "-c", child], cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        lines = []
        while not any("waiting to start" in line for line in lines):
            line = proc.stdout.readline()
            assert line, "".join(lines)
            lines.append(line)
        descriptor = os.open(ledger / "ledger.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            pl.file_lock_exclusive_nb(descriptor)  # raises if the waiting run still held the lock
            pl.file_unlock(descriptor)
        finally:
            os.close(descriptor)
        assert not list(ledger.glob("*.json")), "a waiting run holds no row"
        started = time.monotonic()
        if ending != "clears":
            proc.send_signal(signal.SIGINT if ending == "interrupt" else signal.SIGTERM)
        out, _ = proc.communicate(timeout=60 if ending != "clears" else 300)  # a loaded host starts the child slowly
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    out = "".join(lines) + out
    assert ending == "clears" or time.monotonic() - started < 30, "an ended wait must end promptly:\n" + out
    assert not list(ledger.glob("*.json")), out
    if ending == "clears":
        assert proc.returncode == 0 and ran.exists(), out
        assert "=== focused: memory pressure no longer critical (read " in out and "=== focused: exit 0" in out
    else:
        assert proc.returncode == (130 if ending == "interrupt" else -signal.SIGTERM) and not ran.exists(), out
        assert ("=== focused: interrupted before it started; the run ends with exit 130" in out) is (
            ending == "interrupt")


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT delivery to one process; Windows Ctrl-C reaches a console group")
def test_an_interrupt_while_the_warm_waits_for_memory_ends_the_whole_battery(tmp_path):
    """The warm decides nothing when it fails (the passes run cold), but an interrupt while it waits out critical
    pressure is the owner cancelling the run: the real ``main`` ends with 130 before any pass, instead of carrying
    on into the next pass's wait. Only the machine's readings are substituted (always critical); the Node lane is
    taken as green so the battery reaches its warm."""
    import queue
    import threading

    from ouroboros import test_parallelism

    ledger = tmp_path / "slots"
    child = textwrap.dedent(f"""
        import os, runpy, sys
        sys.path.insert(0, {str(REPO)!r})
        from pathlib import Path
        from ouroboros import config, host_memory, test_parallelism as budget
        os.environ.pop(budget.NESTED_WORKERS_ENV, None)  # a battery of its own, not a test's child
        budget._ledger_dir = lambda: Path({str(ledger)!r})
        budget._process_table = lambda: ([], "")
        budget.physical_memory = lambda: host_memory.MemorySnapshot(2 ** 40, 2 ** 40, True)
        config.TEST_PROCESS_SAMPLE_SEC = 0.05
        main = runpy.run_path({str(REPO / "scripts" / "run_tests.py")!r})["main"]
        main.__globals__["_node_lane"] = lambda: (0, "")  # run_path returns a copy: the functions read their own globals
        raise SystemExit(main([]))
    """)
    env = {key: value for key, value in os.environ.items() if key != test_parallelism.NESTED_WORKERS_ENV}
    proc = subprocess.Popen([sys.executable, "-c", child], cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    arrived: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [*map(arrived.put, proc.stdout), arrived.put("")], daemon=True).start()
    lines = []
    try:
        while not any(line.startswith("=== warm: waiting to start") for line in lines):
            line = arrived.get(timeout=120)
            assert line and ("waiting to start" not in line or line.startswith("=== warm:")), "".join(lines) + line
            lines.append(line)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            while (line := arrived.get(timeout=30)):
                lines.append(line)
            pytest.fail("the battery went on after its warm's wait was interrupted:\n" + "".join(lines))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    while (line := arrived.get(timeout=30)):
        lines.append(line)
    out = "".join(lines)
    assert proc.returncode == 130, out
    assert "=== warm: interrupted before it started; the run ends with exit 130" in out, out
    assert "decides nothing" not in out and "=== parallel" not in out and not list(ledger.glob("*.json")), out


@pytest.mark.parametrize("interrupted, diagnostic, expected", [
    (False, "overload_stop: largest pid=123 rss=456", 1),
    (True, "", 130),
])
def test_runner_never_greens_an_overload_or_interrupt(
    run_tests, monkeypatch, capsys, interrupted, diagnostic, expected,
):
    from ouroboros import process_containment

    events = []

    class Owner:
        def __init__(self, **options):
            pass

        def spawn(self, *args, **kwargs):
            return self

        def wait(self, timeout=None):
            if interrupted and timeout is None:
                raise KeyboardInterrupt
            events.extend([f"wait {timeout}"] if timeout is not None else [])
            return 0

        def interrupt_root(self):
            events.append("interrupted")
            return True

        def terminate_tree(self, grace):
            events.append(f"terminate {grace}")

        def reap(self):
            events.append("reaped")
            return diagnostic

        def close(self):
            events.append("closed")

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    assert run_tests._run("x", [run_tests.sys.executable, "-c", "pass"], REPO) == expected
    # The export wait and TERM grace are allotted inside safe_test's window; its deadline bounds the reap.
    assert run_tests._INTERRUPT_EXPORT_SEC + run_tests._INTERRUPT_TERM_SEC < SAFE_TEST_INT_WINDOW_SEC
    stop = ["interrupted", "wait 7", "terminate 2"] if interrupted else []
    assert events == [*stop, "reaped", "wait 5", "closed"]
    if diagnostic:
        assert diagnostic in capsys.readouterr().err


@pytest.mark.parametrize("argv, held", [
    (["python", "-m", "pytest", "tests/", "-n", "8", "--dist", "loadgroup"], 3),  # the xdist pass: the admitted count
    (["python", "-m", "pytest", "tests/", "-m", "exclusive", "--timeout=300"], 1),  # the one-process pass
    (["node", "--test", "--test-timeout=300000", "--test-concurrency=2", "a.test.js"], 2),  # a lane's concurrency
    (["node", "--test", "--test-concurrency=9", "a.test.js"], 3),  # within the admitted count
    (["python", "-c", "bootstrap", "test_bytecode.py", "--repo", ".", "--workers", "2"], 2),  # the warm's shards
])
def test_a_pass_leases_what_it_uses(run_tests, argv, held):
    """grok 7 / astra 14: every pass used to lease the admitted xdist count, so a neighbour started during the
    one-process pass lost N slots to a run using one."""
    assert run_tests._pass_slots(argv, 3) == held


def test_the_one_process_pass_holds_one_slot_in_the_ledger(run_tests, monkeypatch, standalone_budget):
    from ouroboros import process_containment

    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3")
    seen = {}

    class Owner:
        def __init__(self, **options):
            pass

        def spawn(self, argv, *, cwd, env):
            seen["leases"] = [json.loads(path.read_text(encoding="utf-8")) for path in standalone_budget.glob("*.json")]
            seen["env"] = env
            return types.SimpleNamespace(wait=lambda **kwargs: 0)

        def reap(self):
            return ""

        def close(self):
            pass

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    serial = [run_tests.sys.executable, "-m", "pytest", "tests/", "-m", "exclusive", "--timeout=300"]
    assert run_tests._run("serial", serial, run_tests.REPO) == 0
    (lease,) = seen["leases"]
    assert (lease["workers"], lease["command"]) == (1, "run_tests.py serial")
    assert seen["env"]["OURO_TEST_WORKER_BUDGET"] == "3"  # the pass's own children keep the admitted allocation
    assert list(standalone_budget.glob("*.json")) == []


def test_node_lane_uses_the_shared_worker_budget(run_tests, monkeypatch):
    from ouroboros import preflight_node

    seen = []
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/one.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "node")
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd, node: seen.append((_started(argv), node)) or 0)
    assert run_tests._node_lane() == (0, "node")  # the Node it ran, for the passes after it
    assert seen == [(["node", "--test", "--test-timeout=300000", "--test-concurrency=3", "tests/one.test.js"], "node")]


def test_the_passes_after_the_node_lane_run_the_node_it_ran(run_tests, monkeypatch, tmp_path):
    """The battery and --selected carry the lane's Node to every pass after it; a lane that ran none (inactive, a
    stubbed one) leaves the passes to name one without running anything. The warm runs no Node."""
    from ouroboros import preflight_inventory, test_selection

    calls = []
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd, node="": calls.append((label, node)) or 0)
    for lane in ((0, "/lane/node"), (0, "")):
        calls.clear()
        monkeypatch.setattr(run_tests, "_node_lane", lambda lane=lane: lane)
        assert run_tests.main([]) == 0
        assert calls == [("warm", ""), ("parallel", lane[1]), ("serial", lane[1])]
    chosen = test_selection.TestSelection("selected", "base", ("tests/test_floor.py",), (), (), "policy", 3)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(preflight_inventory, "read_inventory", lambda root, label: ({}, ""))
    monkeypatch.setattr(preflight_inventory, "inventory_problem", lambda selection, inventory: "")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    calls.clear()
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (0, "/lane/node"))
    assert run_tests.main(["--selected", "--evidence-dir", str(tmp_path / "selected")]) == 0
    assert [node for label, node in calls if label != "warm"] == ["/lane/node", "/lane/node"]


def test_empty_lane_with_cleanup_error_makes_the_battery_red(run_tests, monkeypatch, capsys):
    from ouroboros import process_containment

    def owner(code, diagnostic):
        proc = types.SimpleNamespace(wait=lambda **kwargs: code)
        return types.SimpleNamespace(spawn=lambda *args, **kwargs: proc,
                                     reap=lambda: diagnostic, close=lambda: None)

    diagnostic = "overload_stop: largest pid=123 rss=456"
    owners = iter([owner(0, ""), owner(5, diagnostic)])
    monkeypatch.setattr(process_containment, "ProcessContainer", lambda **options: next(owners))
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (0, ""))
    assert run_tests.main([]) == 1
    captured = capsys.readouterr()
    assert diagnostic in captured.err
    assert "battery: RED" in captured.out


def test_the_node_option_runs_only_the_owned_node_lane(run_tests, monkeypatch):
    lanes = []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: lanes.append("node") or (7, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: pytest.fail("pytest must not start"))
    assert run_tests.main(["--node"]) == 7 and lanes == ["node"]  # its exit status is the step's
    for combined in (["--node", "--keep-going"], ["--node", "--selected"], ["--node", "tests/test_x.py"]):
        with pytest.raises(SystemExit):
            run_tests.main(combined)


@pytest.mark.parametrize("node_exit, cleanup, expected", [(0, "", 0), (1, "", 1), (0, "reap incomplete", 2),
                                                         (1, "reap incomplete", 2), (None, "", 2)])
def test_node_evidence_requires_the_test_process_and_its_cleanup(
        run_tests, monkeypatch, tmp_path, standalone_budget, node_exit, cleanup, expected):
    from ouroboros import preflight_node, process_containment

    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda root: ["tests/one.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "inert-node" if node_exit is not None else None)
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    proc = types.SimpleNamespace(wait=lambda **kwargs: node_exit)
    monkeypatch.setattr(process_containment, "ProcessContainer", lambda **options: types.SimpleNamespace(
        spawn=lambda *args, **kw: proc, reap=lambda: cleanup, close=lambda: None))
    assert run_tests.main(["--node", "--evidence-dir", str(tmp_path / "node")]) == expected
    receipt = tmp_path / "node" / "node.json"
    if expected in (0, 1):
        assert json.loads(receipt.read_text(encoding="utf-8")) == {"exit": expected}
    else:
        assert not receipt.exists()


@pytest.mark.parametrize("tier", ["full", "selected"])
def test_a_deferred_full_escalation_runs_the_floor_and_says_so(run_tests, monkeypatch, tmp_path, tier):
    from ouroboros import test_selection

    chosen = test_selection.TestSelection(
        tier, "base", ("tests/test_floor.py",), (("tests/conftest.py", "verification apparatus changed"),),
        ("tests/conftest.py",), "policy", 3)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    summary, ran = tmp_path / "summary.md", []
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setattr(run_tests, "_node_lane", lambda: ran.append("node") or (0, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: ran.append((label, _started(argv))) or 0)
    exit_after_warm = run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "evidence")])
    record = json.loads((tmp_path / "evidence" / "selection.json").read_text(encoding="utf-8"))
    # The leg still runs the Node lane, warms the prefix for the selection, and runs a pytest pass.
    assert ran[0] == "node" and ran[1][0] == "warm" and "--files-from" in ran[1][1] and ran[2][0] == "parallel"
    if tier == "full":  # the PR minimum runs here; the rest goes to the landed shards
        assert record["deferred_full"] is True and record["files"] == sorted(test_selection.FLOOR_TESTS)
        # The pass is started with exactly the floor in its inventory plugin, not on its command line.
        assert plugin_scope(ran[2][1], tmp_path / "evidence" / "probe") == sorted(test_selection.FLOOR_TESTS)
        assert not [arg for arg in ran[2][1] if arg.startswith(SELECT)]
        assert "tests/conftest.py: verification apparatus changed" in summary.read_text(encoding="utf-8")
    else:
        assert "deferred_full" not in record and record["files"] == ["tests/test_floor.py"]
        assert not summary.exists()
    # The warm decides nothing: a warm stopped for memory (-9, run 42126810926) leaves the leg its own exit.
    before = len(ran)
    monkeypatch.setattr(run_tests, "_run",
                        lambda label, argv, cwd: ran.append((label, _started(argv))) or -9 * (label == "warm"))
    assert run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "cold")]) == exit_after_warm
    assert ran[before] == "node" and ran[before + 1][0] == "warm" and ran[before + 2][0] == "parallel"


def test_a_deferred_full_escalation_keeps_what_the_other_changes_select(run_tests, monkeypatch, tmp_path):
    """A shared test helper escalates to full; the changed test and the importers of a changed module
    that arrived in the same window still run on the leg that defers the rest to the landed matrix."""
    import dataclasses

    from ouroboros import test_selection

    changed = ("ouroboros/utils.py", "tests/_store_journal_shared.py", "tests/test_jsonl_lock_budget.py")
    chosen = test_selection.TestSelection(
        "full", "b" * 40, ("tests/test_everything.py",), (("tests/_store_journal_shared.py", "verification apparatus changed"),
                                                          ("tests/test_jsonl_lock_budget.py", "changed test module")),
        changed, "policy", 3)
    asked = []

    def select_from_paths(repo, paths, *, base, extra_tests=(), test_reason=""):
        asked.append((list(paths), base))
        return dataclasses.replace(chosen, tier="selected", files=("tests/test_importer.py", "tests/test_jsonl_lock_budget.py"))

    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(test_selection, "select_from_paths", select_from_paths)
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (0, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: 0)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "kept")])
    record = json.loads((tmp_path / "kept" / "selection.json").read_text(encoding="utf-8"))
    # The escalating helper is left to the matrix; the other two changes are selected as if alone.
    assert asked == [(["ouroboros/utils.py", "tests/test_jsonl_lock_budget.py"], "b" * 40)]
    assert record["deferred_full"] is True and record["files"] == sorted(
        {*test_selection.FLOOR_TESTS, "tests/test_importer.py", "tests/test_jsonl_lock_budget.py"})
    assert "2 module(s) the other changes select" in summary.read_text(encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise ValueError("import scan unavailable for tests/test_broken.py")

    monkeypatch.setattr(test_selection, "select_from_paths", unavailable)
    run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "floor")])
    record = json.loads((tmp_path / "floor" / "selection.json").read_text(encoding="utf-8"))
    assert record["files"] == sorted(test_selection.FLOOR_TESTS)  # the floor still runs, and the note says why
    assert "the other changes could not be selected: import scan unavailable" in summary.read_text(encoding="utf-8")

    # The other changes escalate too (a reason this leg does not name): the floor alone, never the whole suite.
    monkeypatch.setattr(test_selection, "select_from_paths", lambda *a, **k: dataclasses.replace(
        chosen, reasons=(("ouroboros/utils.py", "a renamed escalation"),)))
    run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "again")])
    record = json.loads((tmp_path / "again" / "selection.json").read_text(encoding="utf-8"))
    assert record["files"] == sorted(test_selection.FLOOR_TESTS)
    assert "the other changes escalate too (a renamed escalation)" in summary.read_text(encoding="utf-8")


def test_a_bare_call_never_means_a_subset(run_tests, monkeypatch):
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT", raising=False)
    calls = []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (0, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, argv)) or 0)
    assert run_tests.main([]) == 0
    assert [label for label, _ in calls] == ["warm", "parallel", "serial"]
    calls.clear()
    assert run_tests.main(["tests/test_x.py", "-k", "one"]) == 0
    assert calls == [("focused", [run_tests.sys.executable, "-m", "pytest", "tests/test_x.py", "-k", "one"])]
    calls.clear()
    explicit = ["tests/test_x.py", "--ci-evidence-dir", "my-exact-pytest-directory"]
    assert run_tests.main(explicit) == 0
    assert calls == [("focused", [run_tests.sys.executable, "-m", "pytest", *explicit])]


@pytest.mark.parametrize("arguments, phases", [([], ["warm", "parallel", "serial"]),
                                              (["tests/test_x.py", "-k", "one"], ["focused"])])
def test_evidence_option_keeps_selection_and_gives_each_pytest_phase_its_own_argv(
    run_tests, monkeypatch, tmp_path, arguments, phases,
):
    """A battery warms the run's bytecode prefix once, after the Node lane and before its pytest phases (one owned
    root with the run's workers as shards); a focused run does not."""
    calls, node = [], []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: node.append(True) or (0, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, _started(argv))) or 0)
    monkeypatch.chdir(tmp_path)
    assert run_tests.main(["--evidence-dir", "diagnostics", *arguments]) == 0
    assert [label for label, _ in calls] == phases
    for label, argv in calls:
        if label == "warm":
            assert argv[:2] == [run_tests.sys.executable, "-c"] and argv[-2:] == ["--workers", "3"]
            assert "--ci-evidence-dir" not in argv and "--files-from" not in argv
    assert node == ([] if phases == ["focused"] else [True])
    for label, command in calls:
        if label == "warm":
            continue
        offset = command.index("--ci-evidence-dir")
        assert command[offset + 1] == str(tmp_path / "diagnostics" / label)
        selection = command[:offset] + command[offset + 2:]
        if label == "focused":
            assert selection == [run_tests.sys.executable, "-m", "pytest", *arguments]
        else:
            assert selection == dict(run_tests.battery_commands())[label]


def test_a_missing_node_is_a_red_not_run_never_a_silent_skip(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "resolve_node", lambda: None, raising=False)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: pytest.fail("pytest must not start"))
    from ouroboros import preflight_node
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: None)
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/x.test.js"])
    assert run_tests.main([]) == run_tests._NODE_INFRA_ERROR


def test_a_red_lane_stops_the_battery_and_an_empty_battery_is_red(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (0, ""))
    codes = iter([0, 1])  # the warm, then a red first lane
    seen = []
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: seen.append(label) or next(codes))
    assert run_tests.main([]) == 1
    assert seen == ["warm", "parallel"]
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: 0 if label == "warm" else 5)
    assert run_tests.main([]) == 1
    # A red warm decides nothing: the battery still runs, cold, and its own exit is the run's.
    seen.clear()
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: seen.append(label) or -9 * (label == "warm"))
    assert run_tests.main([]) == 0 and seen == ["warm", "parallel", "serial"]


def test_lane_groups_pin_resources_and_keep_other_files_whole(run_tests, monkeypatch):
    """A named resource's tests share one group across files; every other test keeps its file (or class) scope,
    a mark that names no resource included; with recorded durations the longest groups go first, resource groups
    first among equals."""
    from tests import ci_verdicts

    items = [
        _item("tests/test_a.py::test_one", serial=None),
        _item("tests/test_s.py::test_one", serial=""),
        _item("tests/test_a.py::test_two", serial=None),
        _item("tests/test_s.py::test_two[x]", serial=""),
        _item("tests/test_t.py::test_one", serial=None),
        _item("tests/test_c.py::TestBox::test_one", serial=None),
        _item("tests/test_s.py::TestSer::test_three", serial=""),
        _item("tests/test_d.py::test_db", serial="db"),
        _item("tests/test_e.py::test_db_too", serial="db"),
        _item("tests/test_x.py::test_alone", serial="exclusive"),
    ]
    conftest._pin_lane_groups(items, _config(seconds={"tests/test_a.py": 900, "tests/test_d.py": 50,
                                                      "tests/test_e.py": 60, "tests/test_s.py": 100}))
    groups = {item.nodeid: _group(item) for item in items}
    assert groups["tests/test_a.py::test_one"] == groups["tests/test_a.py::test_two"] == "tests/test_a.py"
    assert groups["tests/test_c.py::TestBox::test_one"] == "tests/test_c.py::TestBox"
    assert groups["tests/test_s.py::test_one"] == groups["tests/test_s.py::test_two[x]"] == "tests/test_s.py"
    assert groups["tests/test_s.py::TestSer::test_three"] == "tests/test_s.py::TestSer"
    assert groups["tests/test_d.py::test_db"] == groups["tests/test_e.py::test_db_too"] == "serial:db"
    assert groups["tests/test_x.py::test_alone"] == "serial:exclusive"
    # Longest group first (a: 900 ms), then the db resource (110), then s (100); of the unrecorded, the resource.
    assert [groups[item.nodeid] for item in items[:5]] == ["tests/test_a.py", "tests/test_a.py", "serial:db", "serial:db",
                                                            "tests/test_s.py"]
    assert [item.nodeid for item in items[-3:]] == ["tests/test_x.py::test_alone", "tests/test_t.py::test_one",
                                                    "tests/test_c.py::TestBox::test_one"]
    # Parameter text is not a collector scope, even when it contains the node separator.
    for scope in ("tests/test_a.py", "tests/test_a.py::TestBox"):
        parameter_items = [_item(f"{scope}::test_case[{value}]", serial=None, parent=scope)
                           for value in ("x@y", "x::y", "x@y::z")]
        original = [item.nodeid for item in parameter_items]
        conftest._pin_lane_groups(parameter_items, _config())
        assert [_group(item) for item in parameter_items] == [scope] * 3
        assert [item.nodeid for item in parameter_items] == original
    # The controller reads the table once and hands every worker that reading: a table a neighbour's session
    # rewrote between two workers' starts cannot sort them into two orders. A plain `-n` run hands nothing.
    system = ci_verdicts.platform_name(run_tests.sys.platform)
    tables = [{"platforms": {system: {"tests/test_a.py": 900}}}, {"platforms": {system: {"tests/test_a.py": 1}}}]
    monkeypatch.setattr(ci_verdicts, "recorded_durations_table", lambda: (tables.pop(0), ""))
    controller = types.SimpleNamespace(option=types.SimpleNamespace(dist="loadgroup"))
    nodes = [types.SimpleNamespace(config=controller, workerinput={}) for _ in range(2)]
    for node in nodes:
        conftest.pytest_configure_node(node)
    assert [node.workerinput["lane_group_seconds"] for node in nodes] == [{"tests/test_a.py": 900}] * 2
    assert len(tables) == 1
    plain = types.SimpleNamespace(config=types.SimpleNamespace(option=types.SimpleNamespace(dist="load")), workerinput={})
    conftest.pytest_configure_node(plain)
    assert plain.workerinput == {}


def test_lane_groups_are_inert_outside_the_one_xdist_pass():
    """A worker of a plain `-n` run (no `--dist loadgroup`) and a session without workers leave scheduling and
    order untouched."""
    items = [_item("tests/test_s.py::test_one", serial="db"), _item("tests/test_a.py::test_one", serial=None)]
    conftest._pin_lane_groups(items, _config(loadgroup=False))
    assert [item.stash for item in items] == [{}, {}]
    conftest._pin_lane_groups(items, types.SimpleNamespace(option=types.SimpleNamespace(dist="loadgroup")))
    assert [item.stash for item in items] == [{}, {}]
    assert [item.nodeid for item in items] == ["tests/test_s.py::test_one", "tests/test_a.py::test_one"]


def test_a_real_xdist_pass_runs_a_resources_tests_in_its_group_on_one_worker(tmp_path):
    """The groups are pinned where items are collected, inside xdist's workers, so they are witnessed there: a
    real two-worker `--dist loadgroup` session under this conftest. The controller schedules by the group each
    worker's collection report carries (`id@group`): one resource's tests in two files ran on one worker, an
    unmarked test rides its file's group. A test, and so every result, lane proof, inventory and retry, keeps
    its plain id, a parameter's own `@` included; every worker holds the controller's reading of the durations
    table. (Keyed on the controller's `--dist`, which a worker reads as "no", the pass pinned nothing.)"""
    import subprocess
    import sys

    from ouroboros.test_environment import isolated_environment
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV
    from tests.ci_verdicts import TEST_DURATIONS_ENV, platform_name

    assert [conftest._plain_id(nodeid) for nodeid in ("t.py::test_a@serial:port", "t.py::test_a[x@y]@t.py",
                                                      "t.py::test_a[x@y]", "t.py::test_a")] == [
        "t.py::test_a", "t.py::test_a[x@y]", "t.py::test_a[x@y]", "t.py::test_a"]
    suite, seen, owner = tmp_path / "suite", tmp_path / "seen", tmp_path / "owner"
    (suite / "tests").mkdir(parents=True)
    seen.mkdir()
    seconds = {"tests/test_a.py": 10, "tests/test_b.py": 5000}
    (tmp_path / "durations.json").write_text(json.dumps({
        "format": 1, "platforms": {platform_name(sys.platform): seconds}, "serial": {platform_name(sys.platform): {}},
        "source": {"run_id": "session 1", "sha": ""}, "ui_nodes": {}}), encoding="utf-8")
    (suite / "tests" / "conftest.py").write_text(
        "import json, pathlib, pytest\n\n\n@pytest.hookimpl(optionalhook=True)\n"
        "def pytest_xdist_node_collection_finished(node, ids):\n"
        f"    pathlib.Path({str(seen)!r}, 'reported-' + node.gateway.id).write_text(json.dumps(list(ids)))\n",
        encoding="utf-8")
    for name in ("a", "b"):
        (suite / "tests" / f"test_{name}.py").write_text(
            "import json, os, pathlib, pytest\n\n\n"
            f"def note(request, name):\n    pathlib.Path({str(seen)!r}, name).write_text(json.dumps([\n"
            "        request.node.nodeid, os.environ['PYTEST_XDIST_WORKER'],\n"
            "        request.config.workerinput['lane_group_seconds']]), encoding='utf-8')\n\n\n"
            f"@pytest.mark.serial('port')\ndef test_shared(request):\n    note(request, {name + '-shared'!r})\n\n\n"
            "@pytest.mark.parametrize('who', ['x@y', 'x::y', 'x@y::z'])\ndef test_free(request, who):\n"
            f"    note(request, {name + '-free-'!r} + who.replace('::', '-').replace('@', '-at-'))\n\n"
            "class TestBox:\n"
            "    @pytest.mark.parametrize('who', ['x@y', 'x::y', 'x@y::z'])\n"
            "    def test_free(self, request, who):\n"
            f"        note(request, {name + '-class-'!r} + who.replace('::', '-').replace('@', '-at-'))\n", encoding="utf-8")
    env = {**isolated_environment(owner, REPO), "OUROBOROS_TEST_TEMP_ROOT": str(owner), NESTED_WORKERS_ENV: "2",
           TEST_DURATIONS_ENV: str(tmp_path / "durations.json")}
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "tests.conftest", "--rootdir", str(suite), "--confcutdir",
                          str(suite), "-o", "addopts=", "-q", "-p", "no:cacheprovider", "-n", "2", "--dist", "loadgroup",
                          str(suite / "tests")], cwd=REPO, env=env, text=True, capture_output=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    ran = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in seen.iterdir()}
    reported = [ran.pop(name) for name in sorted(ran) if name.startswith("reported-")]
    assert len(reported) == 2 and reported[0] == reported[1], "two workers reported two collections"
    # Longest group first: the resource's (its two files' 5010 ms), then b's (5000), then a's (10).
    expected = {f"{name}-shared": f"tests/test_{name}.py::test_shared" for name in ("a", "b")}
    collection = [node + "@serial:port" for node in expected.values()]
    for name in ("b", "a"):
        for kind, suffix in (("free", ""), ("class", "::TestBox")):
            scope = f"tests/test_{name}.py{suffix}"
            for value in ("x@y", "x::y", "x@y::z"):
                plain = f"{scope}::test_free[{value}]"
                collection.append(f"{plain}@{scope}")
                expected[f"{name}-{kind}-" + value.replace("::", "-").replace("@", "-at-")] = plain
    assert reported[0] == collection
    assert {name: row[0] for name, row in ran.items()} == expected
    assert ran["a-shared"][1] == ran["b-shared"][1], "one resource's tests ran on two workers"
    assert [row[2] for row in ran.values()] == [seconds] * 14


def test_the_documented_command_is_the_script():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "test:\n\tuv run --locked python scripts/run_tests.py\n" in makefile
    for doc in ("CONTRIBUTING.md", "README.md", "docs/development/14-build-and-ci.md", "prompts/SYSTEM.md"):
        assert "scripts/run_tests.py" in (REPO / doc).read_text(encoding="utf-8"), doc


def test_runner_evidence_stays_with_its_controller_across_nested_and_sequential_pytest(tmp_path):
    """Exercise the runner's real argv; inheriting PYTEST_ADDOPTS used to truncate both files."""
    import json
    import subprocess
    import sys
    import textwrap
    from tests.test_ci_evidence import _results, _started

    outer, inner = tmp_path / "outer", tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    (inner / "test_child.py").write_text("def test_child():\n    assert True\n", encoding="utf-8")
    evidence, child_evidence = tmp_path / "diagnostics", tmp_path / "child-diagnostics"
    public = evidence / "focused"
    public.mkdir(parents=True)
    child_args = ["-p", "tests.conftest", "-o", "addopts=", "--rootdir", str(inner),
                  "--confcutdir", str(inner), "-q", str(inner)]
    (outer / "test_parent.py").write_text(textwrap.dedent(f"""
        import os, subprocess, sys
        from pathlib import Path
        def test_parent():
            assert "--ci-evidence-dir" not in os.environ.get("PYTEST_ADDOPTS", "")
            public = Path({str(public)!r})
            before = {{name: (public / name).read_bytes() for name in ("events.jsonl", "results.json")}}
            for explicit in ([], ["--ci-evidence-dir", {str(child_evidence)!r}]):
                result = subprocess.run([sys.executable, "-m", "pytest", *{child_args!r}, *explicit],
                                        cwd={str(REPO)!r}, capture_output=True, text=True,
                                        encoding="utf-8", timeout=60)
                assert result.returncode == 0, result.stdout + result.stderr
                assert {{name: (public / name).read_bytes() for name in before}} == before
        """), encoding="utf-8")
    arguments = ["--evidence-dir", str(evidence), "-p", "tests.conftest", "-o", "addopts=",
                 "--rootdir", str(outer), "--confcutdir", str(outer), "-q", str(outer)]
    entry = tmp_path / "run_parent.py"
    entry.write_text(
        "import runpy\n"
        f"runner = runpy.run_path({str(REPO / 'scripts/run_tests.py')!r})\n"
        f"raise SystemExit(runner['main']({arguments!r}))\n", encoding="utf-8")
    command = [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"),
               "--temp-parent", str(tmp_path), "--", sys.executable, str(entry)]
    # Reuse after the preceding controller exited is valid, without stale ownership locks.
    for previous in range(2):
        (public / "results.json").write_text(json.dumps({"previous": previous}), encoding="utf-8")
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                encoding="utf-8", timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _results(public)["tests_collected"] == 1
        assert _results(public)["session_exit_code"] == 0
        assert {row["nodeid"] for row in _results(public)["reports"]} == {"test_parent.py::test_parent"}
        raw = (public / "events.jsonl").read_text(encoding="utf-8")
        assert "\x00" not in raw
        rows = [json.loads(line) for line in raw.splitlines()]
        assert _started(rows) == ["test_parent.py::test_parent"]
        assert rows[-1]["event"] == "finish"
        assert _results(child_evidence)["session_exit_code"] == 0
        assert {row["nodeid"] for row in _results(child_evidence)["reports"]} == {"test_child.py::test_child"}


_DRIVER_KEYS = ("PLAYWRIGHT_NODEJS_PATH", "OUROBOROS_PLAYWRIGHT_NODE_SELECTED", "OUROBOROS_PINNED_NODE_CARRIED")


def _prepared_caller(data):
    """A caller installation's data root ``data`` holding only the pinned Node, prepared (record and file)."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.platform_layer import embedded_node_candidates, node_distribution_platform

    pin, key = runtime.load_runtime_pin(), node_distribution_platform()
    artifact = pin.node_artifacts.get(key)
    if artifact is None:
        pytest.skip("this host has no pinned Node archive")
    root = runtime.managed_runtime_root(data) / "node" / f"{pin.node_version}-{key}.g-{artifact.sha256[:12]}-00000000"
    node = embedded_node_candidates(root)[0]
    node.parent.mkdir(parents=True)
    node.write_bytes(b"prepared pinned node")
    (root / "managed-node.json").write_text(json.dumps({
        "schema_version": 1, "version": pin.node_version, "platform": key, "archive_url": artifact.archive_url,
        "archive_sha256": artifact.sha256, "archive_size": artifact.size_bytes,
        "archive_executable": artifact.executable}), encoding="utf-8")
    return data, str(node)


@pytest.mark.parametrize("operator,layout", [("", "explicit"), ("/operator/chosen-node", "explicit"),
                                             ("=carried", "explicit"), ("", "default"), ("", "app_root")])
def test_safe_test_keeps_an_operators_driver_node_and_carries_the_callers_prepared_pin(tmp_path, operator, layout):
    """Carbon R2: safe_test replaced the data root and scrubbed PLAYWRIGHT_NODEJS_PATH, so the prepared pinned Node of
    the installation running it was invisible inside (Node lane NOT_RUN or PATH) and an operator's literal override
    was lost. The launcher keeps a literal value and carries the caller's prepared Node as one selected executable;
    the hermetic gate keeps scrubbing both. ``default``: a shell with neither OUROBOROS_DATA_DIR nor APP_ROOT, whose
    installation is ``~/Ouroboros/data`` (``launch_roots``), carried the same (plan rev5). Plan rev12 (Carbon R5): the
    carried executable is its own fact, so an operator's literal driver equal to it stays unmarked, the operator's.
    ``app_root``: only OUROBOROS_APP_ROOT, whose installation is ``<APP_ROOT>/data`` (R7: now derived without an
    import from the tree, ``test_safe_test_stage_one_derives_the_default_root_without_importing_the_tree``)."""
    import subprocess
    import sys

    data, node = _prepared_caller(tmp_path / {"default": "home/Ouroboros/data", "app_root": "app/data"}.get(
        layout, "caller-data"))
    operator = node if operator == "=carried" else operator
    probe = ("import json, os; print('FACTS ' + json.dumps({key: os.environ.get(key, '') for key in "
             f"{_DRIVER_KEYS + ('OUROBOROS_DATA_DIR',)!r}}}))")
    env = {key: value for key, value in os.environ.items() if key not in _DRIVER_KEYS
           and key not in ("OUROBOROS_DATA_DIR", "OUROBOROS_APP_ROOT")}
    if layout == "default":
        env["HOME"] = env["USERPROFILE"] = str(tmp_path / "home")
    elif layout == "app_root":
        env["OUROBOROS_APP_ROOT"] = str(tmp_path / "app")
    else:
        env["OUROBOROS_DATA_DIR"] = str(data)
    if operator:
        env["PLAYWRIGHT_NODEJS_PATH"] = operator
    (tmp_path / "t").mkdir()
    launcher = [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py"), "--temp-parent",
                str(tmp_path / "t"), "--evidence-dir", str(tmp_path / "evidence"), "--", sys.executable, "-c", probe]
    if layout != "explicit":  # the conftest Popen gives every child the test's data root; this caller has none
        launcher = [sys.executable, "-I", "-S", "-c", "import os, subprocess, sys; os.environ.pop("
                    "'OUROBOROS_DATA_DIR'); raise SystemExit(subprocess.call(sys.argv[1:]))", *launcher]
    run = subprocess.run(launcher, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-2000:]
    facts = json.loads(next(line for line in run.stdout.splitlines() if line.startswith("FACTS "))[6:])
    assert facts["OUROBOROS_DATA_DIR"] != str(data)  # still a disposable data root: only the executable crossed
    assert facts["OUROBOROS_PINNED_NODE_CARRIED"] == node
    assert facts["PLAYWRIGHT_NODEJS_PATH"] == (operator or node)  # a literal override stays the driver's ...
    assert facts["OUROBOROS_PLAYWRIGHT_NODE_SELECTED"] == ("" if operator else node)  # ... never marked automatic


def test_safe_test_stage_one_derives_the_default_root_without_importing_the_tree(tmp_path, monkeypatch):
    """R7 (Fable): stage one imported ``ouroboros.launch_roots`` (and the package, ``version``, ``installation_stop``)
    from the tree under verification, before its boundary and outside the audit it gives ``test_environment.py``. The
    derivation equals ``launch_roots.data_dir`` for each layout, and loading and calling it imports no ``ouroboros``.
    R8: equal for the spellings ``Path`` normalizes too (an empty value, ``./``, doubled and trailing separators),
    which the first derivation returned raw."""
    import runpy
    import subprocess
    import sys

    from ouroboros.launch_roots import data_dir

    derive = runpy.run_path(str(REPO / "scripts" / "safe_test.py"),
                            run_name="safe_test_under_test")["_default_data_root"]
    home = str(tmp_path / "home")
    for layout in ({}, {"OUROBOROS_APP_ROOT": str(tmp_path / "app")}, {"OUROBOROS_DATA_DIR": str(tmp_path / "data")},
                   {"OUROBOROS_APP_ROOT": str(tmp_path / "app"), "OUROBOROS_DATA_DIR": str(tmp_path / "data")},
                   {"OUROBOROS_DATA_DIR": ""}, {"OUROBOROS_DATA_DIR": "./data"},
                   {"OUROBOROS_DATA_DIR": f"{tmp_path}//data/"}, {"OUROBOROS_APP_ROOT": ""},
                   {"OUROBOROS_APP_ROOT": "./app/"}, {"OUROBOROS_APP_ROOT": f"{tmp_path}//app//"},
                   {"HOME": f"{home}//"}, {"HOME": f"{home}/./"}):
        for key in ("OUROBOROS_APP_ROOT", "OUROBOROS_DATA_DIR"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HOME", home)
        monkeypatch.setenv("USERPROFILE", home)
        for key, value in layout.items():
            monkeypatch.setenv(key, value)
        assert derive() == str(data_dir()), layout
    # The conftest Popen gives every child the test's data root; this caller has none.
    probe = ("import os, runpy, sys; os.environ.pop('OUROBOROS_DATA_DIR', None); "
             "derive = runpy.run_path(sys.argv[1], run_name='probe')['_default_data_root']; "
             "print(derive()); print(sorted(m for m in sys.modules if m.split('.')[0] == 'ouroboros'))")
    ran = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(REPO / "scripts" / "safe_test.py")],
                         cwd=REPO, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "HOME": str(tmp_path / "home"), "OUROBOROS_APP_ROOT": str(tmp_path / "a")})
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [str(tmp_path / "a" / "data"), "[]"]


def test_the_hermetic_gate_scrubs_the_driver_variables_and_the_carried_node(tmp_path, monkeypatch):
    from ouroboros import preflight_runner

    for key in _DRIVER_KEYS:
        monkeypatch.setenv(key, "/carried/node")
    env = preflight_runner._preflight_env(tmp_path / "gate", tmp_path / "gate" / "repo", create=False, workers=1)
    assert not set(_DRIVER_KEYS) & set(env)
