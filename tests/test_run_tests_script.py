"""The local battery entry point keeps the gate's recipe and a bounded serial overlap."""

import importlib.util
import json
import pathlib
import types

import pytest

from ouroboros.preflight_inventory import SELECT
from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS
from ouroboros.safe_test_supervisor import _GRACE_SEC as SAFE_TEST_INT_WINDOW_SEC
from tests import conftest

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
def standalone_budget(monkeypatch):
    from ouroboros import test_parallelism

    # These command-shape cases model the standalone operator entrypoint.
    # Real nested runners retain their inherited one-slot allocation.
    monkeypatch.delenv(test_parallelism.NESTED_WORKERS_ENV, raising=False)
    monkeypatch.setattr(test_parallelism, "physical_memory", lambda:
                        test_parallelism.MemorySnapshot(100 * 1024**3, 100 * 1024**3))


def _item(nodeid, *, serial):
    markers = [types.SimpleNamespace(name="serial")] if serial else []
    item = types.SimpleNamespace(nodeid=nodeid, added=[])
    item.get_closest_marker = lambda name: next((m for m in markers if m.name == name), None)
    item.add_marker = item.added.append
    return item


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
    group, at most half the workers. A run without a table of its own reads the newest earlier one-run battery
    beside its evidence that ended (an interrupted one favours the serial groups handed out first); the plan
    names each group's predicted seconds."""
    from tests.ci_verdicts import platform_name, serial_group
    from tests.test_ci_durations import _projection, _row

    system = platform_name(run_tests.sys.platform)
    rows = {system: {"tests/test_a.py": 7000, "tests/test_s.py": 3000}}
    assert [run_tests.serial_shards(18, {"platforms": rows, "serial": {system: {"tests/test_s.py": ms}}})
            for ms in (3000, 0, 10_000)] == [5, 1, 9]
    assert run_tests.serial_shards(18, {"platforms": rows}) == 4  # a table that does not split the time
    for run, code, serial in (("run-ended", 1, 5.0), ("run-interrupted", 130, 1.0)):
        _projection(tmp_path / run / "pytest" / "overlapped", [_row("tests/test_s.py::test_s@serial0", serial),
                                                               _row("tests/test_a.py::test_a@tests/test_a.py", 5.0)],
                    system=run_tests.sys.platform)
        (tmp_path / run / "result.json").write_text(json.dumps({"returncode": code}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(tmp_path / "run-now"))
    monkeypatch.setattr(run_tests, "recorded_durations", recorded)
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    ((_label, command),) = run_tests.battery_commands(sequential=False)
    assert command[-1] == "--serial-shards=4"  # half the time is serial; today's rule would say 2
    plan = capsys.readouterr().out
    assert "4 serial groups of 8 workers, sized by the battery of run-ended" in plan
    assert f"predicted seconds per group: {serial_group('tests/test_s.py', 4)} 5;" in plan


def test_sequential_mode_keeps_the_gate_marker_split_and_xdist_flags(run_tests, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "6")
    (parallel_label, parallel), (serial_label, serial) = run_tests.battery_commands(sequential=True)
    assert (parallel_label, serial_label) == ("parallel", "serial")
    assert parallel[3:] == ["tests/", "-m", f"not serial and {LANE_EXCLUSION_EXPR}", *PARALLEL_PASS_FLAGS]
    assert serial[3:] == ["tests/", "-m", f"serial and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS]


def test_default_mode_is_one_run_over_both_lanes(run_tests, monkeypatch, standalone_budget):
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "8")
    ((label, command),) = run_tests.battery_commands(sequential=False)
    assert label == "overlapped"
    assert command[:3] == [run_tests.sys.executable, "-m", "pytest"]
    command = command[3:]
    assert command[command.index("-m") + 1] == LANE_EXCLUSION_EXPR
    assert command[command.index("--dist") + 1] == "loadgroup"
    assert command[command.index("-n") + 1] == "auto"
    assert command[-1] == "--serial-shards=2"
    untouched = [flag for flag in PARALLEL_PASS_FLAGS if flag.startswith("--") and flag != "--dist"]
    assert all(flag in command for flag in untouched)


def test_the_worker_override_steers_xdist_like_the_gate(run_tests, monkeypatch, standalone_budget, capsys):
    from ouroboros import process_containment

    seen = {}
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3")

    class Owner:
        def __init__(self, **options):
            seen["options"] = options

        def spawn(self, argv, *, cwd, env):
            seen.update(env=env, argv=argv, cwd=cwd)
            return types.SimpleNamespace(wait=lambda **kwargs: 0)

        def reap(self):
            seen["reaped"] = True
            return ""

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(process_containment, "ProcessContainer", Owner)
    assert run_tests._run("x", [run_tests.sys.executable, "-m", "pytest", "-n", "8"], run_tests.REPO) == 0
    assert seen["argv"][-2:] == ["-n", "3"]
    assert "=== x: pytest -n 3\n" in capsys.readouterr().out
    assert seen["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "3"
    assert seen["env"]["OURO_TEST_WORKER_BUDGET"] == "3"
    assert seen["options"] == {"lifeline": True, "overload": True}
    assert seen["reaped"] and seen["closed"]


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
    assert run_tests.main(["--sequential"]) == 1
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
    assert ran[0] == "node" and ran[1][0] == "parallel"  # the leg still runs the Node lane and a pytest pass
    if tier == "full":  # the PR minimum runs here; the rest goes to the landed shards
        assert record["deferred_full"] is True and record["files"] == sorted(test_selection.FLOOR_TESTS)
        assert all(f"{SELECT}={path}" in ran[1][1] for path in test_selection.FLOOR_TESTS)
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


def test_sequential_with_pytest_arguments_is_refused(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: pytest.fail("nothing must run"))
    with pytest.raises(SystemExit) as exit_info:
        run_tests.main(["--sequential", "tests/test_x.py"])
    assert exit_info.value.code == 2


def test_a_bare_call_never_means_a_subset(run_tests, monkeypatch):
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT", raising=False)
    calls = []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, argv)) or 0)
    assert run_tests.main([]) == 0
    assert [label for label, _ in calls] == ["overlapped"]
    calls.clear()
    assert run_tests.main(["tests/test_x.py", "-k", "one"]) == 0
    assert calls == [("focused", [run_tests.sys.executable, "-m", "pytest", "tests/test_x.py", "-k", "one"])]
    calls.clear()
    explicit = ["tests/test_x.py", "--ci-evidence-dir", "my-exact-pytest-directory"]
    assert run_tests.main(explicit) == 0
    assert calls == [("focused", [run_tests.sys.executable, "-m", "pytest", *explicit])]


@pytest.mark.parametrize("arguments, phases", [([], ["overlapped"]), (["--sequential"], ["parallel", "serial"]),
                                              (["tests/test_x.py", "-k", "one"], ["focused"])])
def test_evidence_option_keeps_selection_and_gives_each_pytest_phase_its_own_argv(
    run_tests, monkeypatch, tmp_path, arguments, phases,
):
    calls, node = [], []
    monkeypatch.setattr(run_tests, "_node_lane", lambda: node.append(True) or 0)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, argv)) or 0)
    monkeypatch.chdir(tmp_path)
    assert run_tests.main(["--evidence-dir", "diagnostics", *arguments]) == 0
    assert [label for label, _ in calls] == phases
    assert node == ([] if phases == ["focused"] else [True])
    for label, command in calls:
        offset = command.index("--ci-evidence-dir")
        assert command[offset + 1] == str(tmp_path / "diagnostics" / label)
        selection = command[:offset] + command[offset + 2:]
        if label == "focused":
            assert selection == [run_tests.sys.executable, "-m", "pytest", *arguments]
        else:
            assert selection == dict(run_tests.battery_commands("--sequential" in arguments))[label]


def test_a_missing_node_is_a_red_not_run_never_a_silent_skip(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "resolve_node", lambda: None, raising=False)
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: pytest.fail("pytest must not start"))
    from ouroboros import preflight_node
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: None)
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/x.test.js"])
    assert run_tests.main([]) == 1


def test_a_red_lane_stops_the_battery_and_an_empty_battery_is_red(run_tests, monkeypatch):
    monkeypatch.setattr(run_tests, "_node_lane", lambda: 0)
    codes = iter([1])
    seen = []
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: seen.append(label) or next(codes))
    assert run_tests.main(["--sequential"]) == 1
    assert seen == ["parallel"]
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: 5)
    assert run_tests.main(["--sequential"]) == 1


def test_lane_groups_pin_serial_files_and_keep_other_files_whole():
    items = [
        _item("tests/test_a.py::test_one", serial=False),
        _item("tests/test_s.py::test_one", serial=True),
        _item("tests/test_a.py::test_two", serial=False),
        _item("tests/test_s.py::test_two[x]", serial=True),
        _item("tests/test_t.py::test_one", serial=True),
        _item("tests/test_c.py::TestBox::test_one", serial=False),
        _item("tests/test_s.py::TestSer::test_three", serial=True),
    ]
    conftest._pin_lane_groups(items, 2)
    assert [item.nodeid.split("::")[0] for item in items[:4]] == ["tests/test_s.py", "tests/test_s.py", "tests/test_t.py", "tests/test_s.py"]
    groups = {item.nodeid: _group(item) for item in items}
    assert groups["tests/test_a.py::test_one"] == groups["tests/test_a.py::test_two"] == "tests/test_a.py"
    assert groups["tests/test_c.py::TestBox::test_one"] == "tests/test_c.py::TestBox"
    assert groups["tests/test_s.py::TestSer::test_three"] == groups["tests/test_s.py::test_one"]
    assert groups["tests/test_s.py::test_one"] == groups["tests/test_s.py::test_two[x]"]
    assert {groups["tests/test_s.py::test_one"], groups["tests/test_t.py::test_one"]} <= {"serial0", "serial1"}


def test_lane_groups_are_inert_without_the_option():
    items = [_item("tests/test_s.py::test_one", serial=True), _item("tests/test_a.py::test_one", serial=False)]
    conftest._pin_lane_groups(items, 0)
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
