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
    """A battery here sizes its serial groups by today's rule unless a test records durations itself (the real
    reader, returned, reads the evidence beside the running test session)."""
    real = run_tests.recorded_durations
    monkeypatch.setattr(run_tests, "recorded_durations", lambda: (None, "no recorded durations"))
    return real


@pytest.fixture
def standalone_budget(monkeypatch, tmp_path):
    """A standalone owner on a quiet host; returns its private ledger directory."""
    from ouroboros import test_parallelism

    # These command-shape cases model the standalone operator entrypoint.
    # Real nested runners retain their inherited one-slot allocation.
    monkeypatch.delenv(test_parallelism.NESTED_WORKERS_ENV, raising=False)
    monkeypatch.setattr(test_parallelism, "physical_memory", lambda:
                        test_parallelism.MemorySnapshot(100 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(test_parallelism, "other_runs", lambda: test_parallelism.HostRuns())
    monkeypatch.setattr(test_parallelism, "_ledger_dir", lambda: tmp_path / "slots")
    return tmp_path / "slots"


def _item(nodeid, *, serial):
    """A collected item; ``serial`` is None (no mark), "" (a bare mark) or the resource the mark names."""
    markers = [] if serial is None else [types.SimpleNamespace(name="serial", args=(serial,) if serial else ())]
    item = types.SimpleNamespace(nodeid=nodeid, added=[])
    item.get_closest_marker = lambda name: next((m for m in markers if m.name == name), None)
    item.add_marker = item.added.append
    return item


def _config(shards="0", dist="loadgroup", workers=4):
    return types.SimpleNamespace(getoption=lambda name: shards, option=types.SimpleNamespace(numprocesses=workers, dist=dist))


def _group(item):
    (mark,) = item.added
    assert mark.name == "xdist_group"
    return mark.args[0]


@pytest.mark.parametrize("workers, shards", [(2, 1), (4, 1), (7, 1), (8, 2), (12, 3), (16, 4), (64, 4)])
def test_serial_overlap_is_bounded_by_the_worker_count(run_tests, workers, shards):
    assert run_tests.serial_shards(workers) == shards


def test_serial_groups_take_the_recorded_serial_share_of_the_workers(run_tests, recorded, monkeypatch, tmp_path,
                                                                     capsys, standalone_budget):
    """With recorded durations the serial tests get the workers their share of the time needs: at least one
    group, at most half the workers. The table is the owner's persisted one beside the evidence parent, named for
    the battery by `point_at_durations` and brought up to date by every session at its end; the plan names each
    group's predicted seconds. A caller's own name for the table stands; without an evidence root nobody owns one."""
    from tests.ci_verdicts import TEST_DURATIONS_ENV, platform_name, serial_group

    system = platform_name(run_tests.sys.platform)
    rows = {system: {"tests/test_a.py": 7000, "tests/test_s.py": 3000}}
    assert [run_tests.serial_shards(18, {"platforms": rows, "serial": {system: {"tests/test_s.py": ms}}})
            for ms in (3000, 0, 10_000)] == [5, 1, 9]
    assert run_tests.serial_shards(18, {"platforms": rows}) == 4  # a table that does not split the time
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
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    (_label, command), _serial = run_tests.battery_commands()
    assert command[-1] == "--serial-shards=4"  # half the time is serial; today's rule would say 2
    plan = capsys.readouterr().out
    assert "4 serial groups of 8 workers, sized by session 2026-11-05T12:00:00Z" in plan
    assert f"predicted seconds per group: {serial_group('tests/test_s.py', 4)} 5;" in plan
    monkeypatch.setenv(TEST_DURATIONS_ENV, "mine.json")
    run_tests.point_at_durations()
    assert os.environ[TEST_DURATIONS_ENV] == "mine.json"
    monkeypatch.delenv(TEST_DURATIONS_ENV)
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT")
    run_tests.point_at_durations()
    assert TEST_DURATIONS_ENV not in os.environ and recorded() == (None, "no recorded durations")


def test_the_battery_is_the_gates_two_passes(run_tests, monkeypatch, standalone_budget):
    """The same marker split and flag vector as the gate and CI, with the bare-serial file groups of the xdist
    pass sized here (so the plan line is true) instead of left to the conftest's `auto`."""
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    (parallel_label, parallel), (serial_label, serial) = run_tests.battery_commands()
    assert (parallel_label, serial_label) == ("parallel", "serial")
    assert parallel[3:] == ["tests/", "-m", f"not exclusive and {LANE_EXCLUSION_EXPR}", *PARALLEL_PASS_FLAGS,
                            "--serial-shards=2"]
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


def test_node_lane_uses_the_shared_worker_budget(run_tests, monkeypatch):
    from ouroboros import preflight_node

    seen = []
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/one.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "node")
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    monkeypatch.setattr(run_tests, "_workers", lambda: 3)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: seen.append(argv) or 0)
    assert run_tests._node_lane() == 0
    assert seen == [["node", "--test", "--test-concurrency=3", "tests/one.test.js"]]


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
    for combined in (["--node", "--sequential"], ["--node", "--selected"], ["--node", "tests/test_x.py"]):
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
    run_tests.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / "evidence")])
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


def test_lane_groups_pin_resources_and_serial_files_and_keep_other_files_whole(run_tests, monkeypatch, tmp_path):
    """A named resource's tests share one group across files; a bare mark's file joins one of the shards; every
    other test keeps its file (or class) scope; with recorded durations the longest groups go first, serial
    groups first among equals; `auto` sizes the shards from the workers and the table."""
    from tests import ci_verdicts

    items = [
        _item("tests/test_a.py::test_one", serial=None),
        _item("tests/test_s.py::test_one", serial=""),
        _item("tests/test_a.py::test_two", serial=None),
        _item("tests/test_s.py::test_two[x]", serial=""),
        _item("tests/test_t.py::test_one", serial=""),
        _item("tests/test_c.py::TestBox::test_one", serial=None),
        _item("tests/test_s.py::TestSer::test_three", serial=""),
        _item("tests/test_d.py::test_db", serial="db"),
        _item("tests/test_e.py::test_db_too", serial="db"),
        _item("tests/test_x.py::test_alone", serial="exclusive"),
    ]
    system = ci_verdicts.platform_name(run_tests.sys.platform)
    table = {"format": 1, "platforms": {system: {"tests/test_a.py": 900, "tests/test_d.py": 50, "tests/test_e.py": 60,
                                                 "tests/test_s.py": 100}}, "serial": {}, "source": {"run_id": "1"}}
    monkeypatch.setattr(ci_verdicts, "recorded_durations_table", lambda: (table, ""))
    conftest._pin_lane_groups(items, _config(shards="2"))
    groups = {item.nodeid: _group(item) for item in items}
    assert groups["tests/test_a.py::test_one"] == groups["tests/test_a.py::test_two"] == "tests/test_a.py"
    assert groups["tests/test_c.py::TestBox::test_one"] == "tests/test_c.py::TestBox"
    assert groups["tests/test_s.py::TestSer::test_three"] == groups["tests/test_s.py::test_one"]
    assert groups["tests/test_s.py::test_one"] == groups["tests/test_s.py::test_two[x]"]
    assert {groups["tests/test_s.py::test_one"], groups["tests/test_t.py::test_one"]} <= {"serial0", "serial1"}
    assert groups["tests/test_d.py::test_db"] == groups["tests/test_e.py::test_db_too"] == "serial:db"
    assert groups["tests/test_x.py::test_alone"] == "serial:exclusive"
    # Longest group first (a: 900 ms), then the db resource (110), then s (100), then the unrecorded ones.
    assert [groups[item.nodeid] for item in items[:5]] == ["tests/test_a.py", "tests/test_a.py", "serial:db", "serial:db",
                                                            groups["tests/test_s.py::test_one"]]
    # `auto` under the one xdist pass: the recorded serial share of this system sizes the shards.
    monkeypatch.setattr(ci_verdicts, "recorded_durations_table", lambda: (None, ""))
    sized = [_item("tests/test_s.py::test_one", serial=""), _item("tests/test_t.py::test_one", serial="")]
    conftest._pin_lane_groups(sized, _config(shards="auto", workers=16))
    assert {_group(item) for item in sized} <= {f"serial{n}" for n in range(4)}  # 16 workers, no table: 4 shards
    conftest._pin_lane_groups([items[0]], _config(shards="", workers=8))  # unset under loadgroup is `auto`
    assert items[0].added[-1].args == ("tests/test_a.py",)


def test_lane_groups_are_inert_outside_the_one_xdist_pass():
    """A plain `-n` run (no `--dist loadgroup`) and an explicit 0 leave scheduling and order untouched."""
    items = [_item("tests/test_s.py::test_one", serial=""), _item("tests/test_a.py::test_one", serial=None)]
    conftest._pin_lane_groups(items, _config(shards="0", dist="load"))
    assert [item.added for item in items] == [[], []]
    conftest._pin_lane_groups(items, _config(shards="0", dist="loadscope"))
    assert [item.added for item in items] == [[], []]
    assert [item.nodeid for item in items] == ["tests/test_s.py::test_one", "tests/test_a.py::test_one"]


def test_the_documented_command_is_the_script():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "test:\n\tuv run --locked python scripts/run_tests.py\n" in makefile
    for doc in ("CONTRIBUTING.md", "README.md", "docs/development/14-build-and-ci.md", "prompts/SYSTEM.md"):
        assert "scripts/run_tests.py" in (REPO / doc).read_text(encoding="utf-8"), doc


@pytest.mark.serial
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
