"""The local battery entry point keeps the gate's recipe and a bounded serial overlap."""

import importlib.util
import json
import os
import pathlib
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
    monkeypatch.setattr(run_tests, "_workers", lambda: 3)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: seen.append(argv) or 0)
    assert run_tests._node_lane() == 0
    assert seen == [["node", "--test", "--test-timeout=300000", "--test-concurrency=3", "tests/one.test.js"]]


def test_empty_lane_with_cleanup_error_makes_the_battery_red(run_tests, monkeypatch, capsys):
    from ouroboros import process_containment

    def owner(code, diagnostic):
        proc = types.SimpleNamespace(wait=lambda **kwargs: code)
        return types.SimpleNamespace(spawn=lambda *args, **kwargs: proc,
                                     reap=lambda: diagnostic, close=lambda: None)

    diagnostic = "overload_stop: largest pid=123 rss=456"
    owners = iter([owner(0, ""), owner(5, diagnostic)])
    monkeypatch.setattr(process_containment, "ProcessContainer", lambda **options: next(owners))
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
    assert run_tests.main([]) == 1
    captured = capsys.readouterr()
    assert diagnostic in captured.err
    assert "battery: RED" in captured.out


def test_the_node_option_runs_only_the_owned_node_lane(run_tests, monkeypatch):
    lanes = []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: lanes.append("node") or 7)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: pytest.fail("pytest must not start"))
    assert run_tests.main(["--node"]) == 7 and lanes == ["node"]  # its exit status is the step's
    for combined in (["--node", "--keep-going"], ["--node", "--selected"], ["--node", "tests/test_x.py"]):
        with pytest.raises(SystemExit):
            run_tests.main(combined)


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
    monkeypatch.setattr(run_tests, "_node_lane", lambda: ran.append("node") or 0)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: ran.append((label, argv)) or 0)
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
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: ran.append((label, argv)) or -9 * (label == "warm"))
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
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
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
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
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
    monkeypatch.setattr(run_tests, "_node_lane", lambda: node.append(True) or 0)
    monkeypatch.setattr(run_tests, "_workers", lambda: 3)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, argv)) or 0)
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
    assert run_tests.main([]) == 1


def test_a_red_lane_stops_the_battery_and_an_empty_battery_is_red(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
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
