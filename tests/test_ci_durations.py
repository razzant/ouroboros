"""Shards weighed by recorded durations, retry verdicts in the lane proof, and the run's verdict list."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR
from ouroboros.test_parallelism import NESTED_WORKERS_ENV
from tests import browser_lane, default_lane
from tests.ci_evidence import file_shards, node_shards, reconcile_shards
from tests.ci_verdicts import DURATIONS_ENV, collect_durations, platform_name

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-I", "-S", str(REPO / "tests/ci_evidence.py")]
SHA, RUN = "d" * 40, "900"
SYSTEM = platform_name(sys.platform)
EXPRESSIONS = {name: f"{prefix}serial and {LANE_EXCLUSION_EXPR}"
               for name, prefix in (("parallel", "not "), ("serial", ""))}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return path


def _table(tmp_path, platforms=None, ui_nodes=None, name="durations.json"):
    return _write(tmp_path / name, {"format": 1, "platforms": platforms or {},
                                    "source": {"run_id": RUN, "sha": SHA}, "ui_nodes": ui_nodes or {}})


# ---------------------------------------------------------------- the weight table


def _row(node, seconds, phase="call", outcome="passed", subtest=False):
    return {"nodeid": node, "phase": phase, "outcome": outcome, "duration_seconds": seconds, "is_subtest": subtest}


def _projection(directory, reports, *, system="Windows", attempt="1", **blocks):
    return _write(Path(directory) / "results.json", {
        "format": 1, "session_exit_code": 0, "reports": reports, "collection_failures": [],
        "environment": {"platform": system}, "github": {"sha": SHA, "run_id": RUN, "run_attempt": attempt},
        **blocks})


def _block(pass_, shard=None):
    return {"full": [], "file_weights": [["tests/x.py", 1]], "pass": pass_,
            **({"shard": shard, "assigned": []} if shard else {})}


def test_durations_sum_each_file_per_system_and_each_ui_node(tmp_path):
    root = tmp_path / "evidence"
    a = "tests/test_a.py::test_one"
    # Windows shard 1: attempt 2 supersedes attempt 1 of the same pass.
    _projection(root / "default-ci-shard-Windows-1-a1" / "parallel", [_row(a, 9.0)],
                default_parallel=_block("parallel", [1, 2]))
    _projection(root / "default-ci-shard-Windows-1-a2" / "parallel", [
        _row(a, 0.25, "setup"), _row(a, 1.5), _row(a, 0.25, "teardown"),
        _row(a, 0.7, subtest=True),  # Within its test's call: never counted twice.
        _row("tests/test_a.py::test_two", None), _row("tests/test_b.py::TestC::test_three", 0.0004)],
        attempt="2", default_parallel=_block("parallel", [1, 2]))
    # The serial pass adds to the same file; a copy of it in a second artifact counts once.
    for artifact in ("default-ci-shard-Windows-1-a2", "ci-evidence-full-test-windows-shard-1-a2"):
        _projection(root / artifact / "serial", [_row("tests/test_a.py::test_s", 2.0)], attempt="2",
                    default_serial=_block("serial", [1, 2]))
    # Another system; a manifest, a retry round, another producer and an unknown system add nothing.
    _projection(root / "default-ci-shard-Linux-1" / "parallel", [_row(a, 0.5)], system="Linux",
                default_parallel=_block("parallel", [1, 4]))
    _projection(root / "default-ci-manifest-Linux" / "parallel", [], system="Linux",
                default_parallel=_block("parallel"))
    _projection(root / "default-ci-shard-Linux-1" / "parallel" / "retry-1", [_row(a, 60.0)], system="Linux")
    _projection(root / "ci-evidence-full-test-linux" / "size", [_row("tests/test_size.py::t", 3.0)], system="Linux")
    _projection(root / "elsewhere" / "parallel", [_row(a, 3.0)], system="Plan9")
    # An unsharded full-suite pass: a `parallel` or `serial` directory without a lane block.
    _projection(root / "ci-evidence-full-test-macos" / "parallel", [_row("tests/test_m.py::t", 0.125)], system="Darwin")
    _projection(root / "ui-ci-full-shard-1" / "host", [_row("tests/test_ui.py::page[chromium]", 4.0, "setup"),
                                                       _row("tests/test_ui.py::page[chromium]", 1.0)],
                system="Linux", ui_browser={"full": [], "shard": [1, 8], "assigned": []})
    table = {"format": 1, "platforms": {"darwin": {"tests/test_m.py": 125}, "linux": {"tests/test_a.py": 500},
                                        "windows": {"tests/test_a.py": 4000, "tests/test_b.py": 0}},
             "serial": {"windows": {"tests/test_a.py": 2000}},
             "source": {"run_id": RUN, "sha": SHA}, "ui_nodes": {"tests/test_ui.py::page[chromium]": 5000}}
    assert collect_durations(root, run_id=RUN, sha=SHA) == table
    out, payloads = tmp_path / "out" / "ci-durations.json", []
    for _ in range(2):
        result = subprocess.run(CLI + ["durations", "--evidence-root", str(root), "--run-id", RUN, "--sha", SHA,
                                       "--out", str(out)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        payloads.append(out.read_bytes())
    assert payloads[0] == payloads[1] and json.loads(payloads[0]) == table
    assert list(json.loads(payloads[0])) == sorted(table) and "windows 2 files" in result.stdout
    empty = subprocess.run(CLI + ["durations", "--evidence-root", str(tmp_path / "none"), "--run-id", RUN,
                                  "--sha", SHA, "--out", str(out)], capture_output=True, text=True, timeout=30)
    assert empty.returncode == 0 and "::warning::" in empty.stdout and _json(out)["platforms"] == {}


def test_the_one_run_battery_records_the_time_of_its_serial_groups(tmp_path):
    """scripts/run_tests.py's one-run battery (an ``overlapped`` pass) pins its serial tests to ``serial<N>``
    xdist groups: their time is their file's serial part; a test in a file or class group is not serial."""
    root = tmp_path / "evidence"
    _projection(root / "run-1" / "pytest" / "overlapped", [
        _row("tests/test_s.py::test_one@serial1", 2.0), _row("tests/test_s.py::test_one@serial1", 0.5, "teardown"),
        _row("tests/test_p.py::test_two@tests/test_p.py", 1.0),
        _row("tests/test_q.py::TestK::test_three@tests/test_q.py::TestK", 0.25)], system="Darwin")
    table = collect_durations(root, run_id=RUN, sha=SHA)
    assert table["platforms"] == {"darwin": {"tests/test_p.py": 1000, "tests/test_q.py": 250, "tests/test_s.py": 2500}}
    assert table["serial"] == {"darwin": {"tests/test_s.py": 2500}}


def test_file_weights_are_recorded_time_else_nodes_times_the_median_time_per_node(tmp_path, monkeypatch):
    counts = {"tests/test_a.py": 4, "tests/test_b.py": 2, "tests/test_c.py": 3, "tests/test_new.py": 5}
    monkeypatch.delenv(DURATIONS_ENV, raising=False)
    assert default_lane.file_weights(counts) == (counts, "nodes")
    monkeypatch.setenv(DURATIONS_ENV, str(_table(tmp_path, {
        SYSTEM: {"tests/test_a.py": 400, "tests/test_b.py": 0, "tests/test_c.py": 900, "tests/test_gone.py": 50_000},
        "plan9": {"tests/test_new.py": 1}})))
    # Per node a 100, b 0, c 300; the deleted file is not collected: a median of 100 ms per node.
    assert default_lane.file_weights(counts) == ({"tests/test_a.py": 400, "tests/test_b.py": 1,
                                                  "tests/test_c.py": 900, "tests/test_new.py": 500}, f"durations:{RUN}")


@pytest.mark.parametrize("content, reason", [
    (None, "the durations table is unreadable (FileNotFoundError)"),
    ("not json", "the durations table is unreadable (JSONDecodeError)"),
    ("[]", "the durations table has another shape"),
    ('{"format": 2, "source": {"run_id": "1"}, "platforms": {}}', "the durations table has another shape"),
    ('{"format": 1, "source": {"run_id": ""}, "platforms": {}}', "the durations table has another shape"),
    ('{"format": 1, "source": {"run_id": "1"}, "platforms": {"linux": {"tests/a.py": -1}}}',
     "the durations table has another shape"),
    ('{"format": 1, "source": {"run_id": "1"}, "platforms": {}, "ui_nodes": {"x": 1.5}}',
     "the durations table has another shape"),
    ('{"format": 1, "source": {"run_id": "1"}, "platforms": {}, "serial": {"linux": {"tests/a.py": 1.5}}}',
     "the durations table has another shape"),
    ('{"format": 1, "source": {"run_id": "1"}, "platforms": {}}',
     f"the durations of run 1 record no collected file on {SYSTEM}"),
], ids=["missing", "garbage", "array", "format", "no-run", "negative", "float", "serial-float", "other-system"])
def test_an_unusable_table_falls_back_to_node_counts_and_says_why(tmp_path, monkeypatch, content, reason):
    path = tmp_path / "durations.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv(DURATIONS_ENV, str(path))
    counts = {"tests/test_a.py": 2}
    assert default_lane.file_weights(counts) == (counts, f"nodes: {reason}")


# ---------------------------------------------------------------- real default-lane sessions


def _suite(tmp_path, files):
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    for name, source in files.items():
        (suite / "tests" / name).write_text(source, encoding="utf-8")
    entry = _write(tmp_path / "run.py",
                   "import os, sys\n"
                   f"sys.path.insert(0, {str(REPO)!r})\n"
                   f"os.chdir({str(suite)!r})\n"
                   f"os.environ.update(GITHUB_SHA={SHA!r}, GITHUB_RUN_ID={RUN!r}, GITHUB_RUN_ATTEMPT='1')\n"
                   "import pytest\n"
                   "raise SystemExit(pytest.main(sys.argv[1:]))\n")
    return suite, entry


def _session(tmp_path, suite, entry, lane, public, *options, env=None):
    args = ["tests/", "-p", "tests.conftest", "--rootdir", str(suite), "--confcutdir", str(suite), "-o", "addopts=",
            "-m", EXPRESSIONS[lane], "-q", "--ci-evidence-dir", str(public), *options]
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"), "--temp-parent", str(tmp_path), "--",
         sys.executable, str(entry), *args],
        env={**{key: value for key, value in os.environ.items() if key != DURATIONS_ENV},
             NESTED_WORKERS_ENV: "2", **(env or {})},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert (public / "results.json").exists(), result.stdout + result.stderr
    return result, _json(public / "results.json")


PARAMETRIZED = "import pytest\n@pytest.mark.parametrize('n', range({}))\ndef test_n(n):\n    pass\n"


@pytest.mark.serial
def test_real_shards_pack_whole_files_by_recorded_time_and_reconcile(tmp_path):
    suite, entry = _suite(tmp_path, {
        # The table shapes the collection only: the session's own tests never inherit it.
        "test_heavy.py": f"import os\ndef test_one():\n    assert {DURATIONS_ENV!r} not in os.environ\n",
        "test_light_a.py": PARAMETRIZED.format(3), "test_light_b.py": PARAMETRIZED.format(3),
        "test_new.py": PARAMETRIZED.format(2)})
    durations = {DURATIONS_ENV: str(_table(tmp_path, {SYSTEM: {
        "tests/test_heavy.py": 9000, "tests/test_light_a.py": 100, "tests/test_light_b.py": 100}}))}
    _, manifest = _session(tmp_path, suite, entry, "parallel", tmp_path / "manifest" / "m1" / "parallel",
                           "--collect-only", "--default-suite-manifest", env=durations)
    lane = manifest["default_parallel"]
    assert lane["weights_source"] == f"durations:{RUN}"
    # test_new.py is not recorded: 2 nodes at the median of 1/3 s per node (light files), 9 s (heavy).
    assert lane["file_weights"] == [["tests/test_heavy.py", 9000], ["tests/test_light_a.py", 100],
                                    ["tests/test_light_b.py", 100], ["tests/test_new.py", 67]]
    files = {}
    for shard in (1, 2):
        result, data = _session(tmp_path, suite, entry, "parallel", tmp_path / "shards" / f"s{shard}" / "parallel",
                                f"--default-suite-shard={shard}/2", env=durations)
        assert result.returncode == 0, result.stdout + result.stderr
        files[shard] = {node.split("::", 1)[0] for node in data["default_parallel"]["assigned"]}
    assert files == {1: {"tests/test_heavy.py"},
                     2: {"tests/test_light_a.py", "tests/test_light_b.py", "tests/test_new.py"}}
    counted = file_shards({"tests/test_heavy.py": 1, "tests/test_light_a.py": 3, "tests/test_light_b.py": 3,
                           "tests/test_new.py": 2}, 2)
    assert {path for path, owner in counted.items() if owner == 1} != files[1], "node counts would pack otherwise"
    gaps, notes = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2, sha=SHA, run_id=RUN,
                                   lane_key="default_parallel")
    assert gaps == [] and notes[0] == f"manifest: 9 nodes, from attempt 1; weights durations:{RUN}"
    # A shard that did not read the run's table proves another packing.
    os.replace(tmp_path / "shards" / "s2", tmp_path / "s2-weighted")
    _session(tmp_path, suite, entry, "parallel", tmp_path / "shards" / "s2" / "parallel", "--default-suite-shard=2/2")
    gaps, _ = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2, sha=SHA, run_id=RUN,
                               lane_key="default_parallel")
    assert (f"shard 2/2: file weights differ from the manifest (weights nodes here, durations:{RUN} in the manifest)"
            in gaps), gaps


@pytest.mark.serial
def test_a_shard_without_a_file_of_a_pass_proves_its_empty_slice(tmp_path):
    """Few files carry `serial`: with every OS sharded, a shard can get none of a pass. It runs nothing,
    exits 0 (not pytest's 5) and proves the empty slice the manifest's packing gives it."""
    suite, entry = _suite(tmp_path, {
        "test_parallel_only.py": PARAMETRIZED.format(3),
        "test_serial_only.py": "import pytest\npytestmark = pytest.mark.serial\n" + PARAMETRIZED.format(2)})
    # Node counts over both passes: the parallel file (3) on shard 1, the serial file (2) on shard 2.
    for lane, empty, options in (("parallel", 2, ["-n", "2", "--dist", "loadscope"]), ("serial", 1, [])):
        _session(tmp_path, suite, entry, lane, tmp_path / "manifest" / "m1" / lane, "--collect-only",
                 "--default-suite-manifest")
        for shard in (1, 2):
            result, data = _session(tmp_path, suite, entry, lane, tmp_path / "shards" / f"s{shard}" / lane,
                                    f"--default-suite-shard={shard}/2", *options)
            assert result.returncode == 0 and data["session_exit_code"] == 0, result.stdout + result.stderr
            assigned = data[f"default_{lane}"]["assigned"]
            assert (assigned == [] and data["reports"] == []) if shard == empty else assigned, (lane, shard)
        gaps, notes = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2, sha=SHA, run_id=RUN,
                                       lane_key=f"default_{lane}")
        assert gaps == [] and f"shard {empty}/2: proof from attempt 1; assigned 0, executed 0" in notes, (lane, gaps)


# ---------------------------------------------------------------- the UI lane


class _UiConfig(SimpleNamespace):
    def getoption(self, name):
        return self.shard if name == "--ui-browser-shard" else True

    def getini(self, name):
        return {"python_files": ["test_*.py"], "norecursedirs": [".*"]}[name]


def _ui_lane(tmp_path, nodes, shard=None):
    """browser_lane's own guard and slicing over the given ids; the lane facts it records and the items kept."""
    plugins, deselected = {}, []
    config = _UiConfig(shard=shard, args=["tests/"], option=SimpleNamespace(collectonly=False),
                       rootpath=tmp_path / "no-such-root",
                       hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)),
                       pluginmanager=SimpleNamespace(get_plugin=plugins.get,
                                                     register=lambda plugin, name: plugins.__setitem__(name, plugin)))
    browser_lane.pytest_configure(config)
    items = [SimpleNamespace(nodeid=node, get_closest_marker=lambda marker: marker == "ui_browser")
             for node in reversed(nodes)]
    hook = browser_lane.pytest_collection_modifyitems(config, items)
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)
    return config._ui_browser_lane, sorted(item.nodeid for item in items)


UI = [f"tests/test_ui.py::test_{name}" for name in "abcdef"]
UI_WEIGHTS = [[UI[0], 6000], [UI[1], 100], [UI[2], 100], [UI[3], 3000], [UI[4], 900], [UI[5], 900]]


def test_ui_shards_pack_the_sorted_lane_by_recorded_node_time(tmp_path, monkeypatch):
    monkeypatch.delenv(DURATIONS_ENV, raising=False)
    assert _ui_lane(tmp_path, UI, "1/2") == ({"full": UI, "shard": [1, 2], "assigned": UI[0::2]}, UI[0::2])
    # Unknown nodes weigh the table's median (900 ms, of five recorded ones, one of them gone).
    monkeypatch.setenv(DURATIONS_ENV, str(_table(tmp_path, ui_nodes={
        UI[0]: 6000, UI[1]: 100, UI[2]: 100, UI[3]: 3000, "tests/test_gone.py::test_x": 900})))
    assert node_shards([weight for _, weight in UI_WEIGHTS], 2) == [1, 2, 2, 2, 2, 2]
    for shard, assigned in ((1, UI[:1]), (2, UI[1:])):
        lane, kept = _ui_lane(tmp_path, UI, f"{shard}/2")
        assert lane == {"full": UI, "weights_source": f"durations:{RUN}", "node_weights": UI_WEIGHTS,
                        "shard": [shard, 2], "assigned": assigned} and kept == assigned
    assert _ui_lane(tmp_path, UI)[0] == {"full": UI, "weights_source": f"durations:{RUN}", "node_weights": UI_WEIGHTS}
    # A table without UI nodes, or none that can be read, keeps the positions and says why.
    monkeypatch.setenv(DURATIONS_ENV, str(_table(tmp_path, {"linux": {}}, name="files-only.json")))
    assert _ui_lane(tmp_path, UI, "1/2")[0] == {
        "full": UI, "weights_source": f"positions: the durations of run {RUN} record no UI node",
        "shard": [1, 2], "assigned": UI[0::2]}
    monkeypatch.setenv(DURATIONS_ENV, str(tmp_path / "absent.json"))
    assert _ui_lane(tmp_path, UI)[0]["weights_source"] == (
        "positions: the durations table is unreadable (FileNotFoundError)")


def _ui_run(tmp_path, *, change=None):
    owners = node_shards([weight for _, weight in UI_WEIGHTS], 2)
    for shard in (None, 1, 2):
        lane = {"full": UI, "node_weights": [list(row) for row in UI_WEIGHTS]}
        if shard:
            lane.update(shard=[shard, 2], assigned=[node for node, owner in zip(UI, owners) if owner == shard])
        if change:
            change(shard, lane)
        _projection(tmp_path / ("manifest/m1" if shard is None else f"shards/s{shard}") / "host",
                    [_row(node, 1.0) for node in lane.get("assigned", [])], system="Linux", ui_browser=lane)
    return reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2, sha=SHA, run_id=RUN)


@pytest.mark.parametrize("change, gap", [
    (None, None),
    (lambda shard, lane: shard and lane.update(assigned=UI[shard - 1::2]),
     "shard 1/2: its assignment is not slice 1 of 2 of the manifest lane"),
    (lambda shard, lane: shard == 2 and lane["node_weights"][0].__setitem__(1, 1),
     "shard 2/2: node weights differ from the manifest"),
    (lambda shard, lane: shard == 2 and lane.pop("node_weights"), "shard 2/2: node weights differ from the manifest"),
    (lambda shard, lane: shard is None and lane["node_weights"].reverse(), "invalid ui_browser block"),
], ids=["packed", "positional-slice", "other-weights", "no-weights", "weights-not-the-lane"])
def test_a_packed_ui_lane_reconciles_only_against_the_manifests_packing(tmp_path, change, gap):
    gaps, _ = _ui_run(tmp_path, change=change)
    assert gaps == [] if gap is None else any(gap in text for text in gaps), gaps


# ---------------------------------------------------------------- retry verdicts in the lane proof


FULL = [f"tests/test_{name}.py::test_case[{index}]" for name in "abcd" for index in range(2)]
WEIGHTS = dict(Counter(node.split("::", 1)[0] for node in FULL))
OWNERS = file_shards(WEIGHTS, 2)
ASSIGNED = {shard: [node for node in FULL if OWNERS[node.split("::", 1)[0]] == shard] for shard in (1, 2)}


def _call(node, outcome="passed"):
    return {"nodeid": node, "phase": "call", "outcome": outcome, "error_type": "AssertionError" * (outcome == "failed")}


def _default_run(tmp_path, *, lane="default_parallel", system="Windows"):
    """A complete run of one default lane in two shards; returns the pass directory of each shard."""
    passes = {}
    for shard in (None, 1, 2):
        block = {"full": FULL, "file_weights": [[path, weight] for path, weight in WEIGHTS.items()],
                 "pass": lane.split("_", 1)[1]}
        if shard:
            block.update(shard=[shard, 2], assigned=ASSIGNED[shard])
        directory = tmp_path / (f"manifest/m-{system}" if shard is None else f"shards/s{shard}-{system}") / block["pass"]
        _projection(directory, [_call(node) for node in block.get("assigned", [])], system=system, **{lane: block})
        passes[shard] = directory
    return passes


def _verdict(directory, verdict="flaky", *, rounds=(".", "retry-1"), flaky=(), failed=(), accepted=(), not_run=(),
             run_id=RUN, attempt="1", payload=None):
    return _write(Path(directory) / "verdict.json", payload if payload is not None else {
        "format": 1, "pass": Path(directory).name, "github": {"sha": SHA, "run_id": run_id, "run_attempt": attempt},
        "platform": "windows", "rounds": [{"dir": name, "exit": 0, "ran": 1} for name in rounds],
        "flaky": [{"nodeid": node, "first": {"phase": "call", "error_type": "AssertionError"}} for node in flaky],
        "failed": [{"nodeid": node, "attempts": [{"phase": "call", "error_type": "AssertionError"}] * 2}
                   for node in failed],
        "accepted": [{"nodeid": node, "issue": 7} for node in accepted], "not_run": list(not_run),
        "continued": 0, "counts": {}, "verdict": verdict, "reason": ""})


def _red_first_session(passes):
    """Shard 1's session failed one test and stopped before another: exit 1."""
    first, failed, left = passes[1], ASSIGNED[1][0], ASSIGNED[1][1]
    data = _json(first / "results.json")
    data.update(session_exit_code=1, reports=[_call(failed, "failed")] + [
        _call(node) for node in ASSIGNED[1] if node not in (failed, left)])
    _write(first / "results.json", data)
    _projection(first / "retry-1", [_call(failed), _call(left)])  # The round: no lane block.
    return first, failed, left


def _reconciled(tmp_path, lane="default_parallel", system=""):
    return reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2, sha=SHA, run_id=RUN,
                            lane_key=lane, system=system)


def test_a_red_shard_session_with_a_passing_verdict_is_proven_with_its_rounds(tmp_path):
    first, failed, left = _red_first_session(_default_run(tmp_path))
    gaps, _ = _reconciled(tmp_path)
    assert gaps == ["shard 1/2: its session ended with exit status 1",
                    f"shard 1/2: assigned but not executed (1): {left}",
                    f"lane: 1 of {len(FULL)} manifest nodes are executed by no shard: {left}"]
    _verdict(first, flaky=[failed])
    gaps, notes = _reconciled(tmp_path)
    assert gaps == [] and (f"shard 1/2: proof from attempt 1; assigned {len(ASSIGNED[1])}, executed "
                           f"{len(ASSIGNED[1])}; retry verdict flaky over 2 round(s): FLAKY 1, ACCEPTED 0") in notes
    _verdict(first, "accepted", failed=[failed], accepted=[failed])
    assert _reconciled(tmp_path)[0] == []


@pytest.mark.parametrize("verdict, gap", [
    ({"verdict": "red", "failed": [ASSIGNED[1][0]]}, "its retry verdict is red (1 failed, 0 not run)"),
    ({"verdict": "red", "not_run": [ASSIGNED[1][1]]}, "its retry verdict is red (0 failed, 1 not run)"),
    # A verdict that says flaky but lists an unaccepted failure is judged by its lists.
    ({"verdict": "flaky", "failed": [ASSIGNED[1][0]]}, "its retry verdict is red (1 failed, 0 not run)"),
    ({"verdict": "flaky", "run_id": "901"}, f"its retry verdict belongs to commit {SHA} run 901"),
    ({"payload": "{not json"}, "its retry verdict is unreadable"),
    ({"payload": {"format": 1, "verdict": "green"}}, "its retry verdict is invalid"),
], ids=["red-failed", "red-not-run", "inconsistent", "foreign", "unreadable", "invalid"])
def test_a_red_shard_session_without_a_passing_verdict_of_this_run_stays_a_gap(tmp_path, verdict, gap):
    first, _, _ = _red_first_session(_default_run(tmp_path))
    _verdict(first, **verdict)
    gaps, _ = _reconciled(tmp_path)
    assert f"shard 1/2: its session ended with exit status 1; {gap}" in gaps, gaps


def test_retry_rounds_are_never_lane_proofs_nor_red_producers(tmp_path):
    passes = _default_run(tmp_path)
    # A round that ended red (rounds carry no lane block), and one that carries a lane block all the same.
    _write(passes[1] / "retry-1" / "results.json",
           {key: value for key, value in _json(passes[1] / "results.json").items() if key != "default_parallel"}
           | {"session_exit_code": 1})
    _projection(passes[2] / "retry-2", [], **{"default_parallel": _json(passes[2] / "results.json")["default_parallel"]})
    assert _reconciled(tmp_path)[0] == []
    # The same projections under any other name are a red producer beside the proof and a second proof.
    os.replace(passes[1] / "retry-1", passes[1] / "rerun-1")
    os.replace(passes[2] / "retry-2", passes[2] / "rerun-2")
    gaps, _ = _reconciled(tmp_path)
    assert "s1-Windows/parallel/rerun-1/results.json: its session ended with exit status 1" in gaps
    assert "shard 2/2: 2 proofs from attempt 1" in gaps


def _journal_only(directory, *, run_id=RUN, lane_row=True, executed=()):
    """A killed session: no export, a journal with the lane block it recorded at collection."""
    data = _json(directory / "results.json")
    (directory / "results.json").unlink()
    rows = [{"event": "lane", "nodeid": "", "block": "default_parallel", "lane": data["default_parallel"],
             "github": {"sha": SHA, "run_id": run_id, "run_attempt": "1"},
             "environment": {"platform": "Windows"}}] if lane_row else []
    for node in executed:
        rows += [{"event": "start", "nodeid": node}, dict(_call(node), event="report"), {"event": "finish", "nodeid": node}]
    rows.append({"event": "start", "nodeid": data["default_parallel"]["assigned"][len(executed)]})
    _write(directory / "events.jsonl", "".join(json.dumps(row) + "\n" for row in rows))


def test_a_killed_shard_session_is_proven_by_its_journal_and_a_passing_verdict(tmp_path):
    passes = _default_run(tmp_path)
    killed = passes[2]
    _journal_only(killed, executed=ASSIGNED[2][:1])
    gaps, _ = _reconciled(tmp_path)
    assert gaps[0] == "shard 2/2: its session ended before its final export" and len(gaps) == 3, gaps
    _projection(killed / "retry-1", [_call(node) for node in ASSIGNED[2][1:]])
    _verdict(killed, flaky=ASSIGNED[2][1:2])
    gaps, notes = _reconciled(tmp_path)
    assert gaps == [] and any(note.startswith(f"shard 2/2: proof from attempt 1; assigned {len(ASSIGNED[2])}, "
                                              f"executed {len(ASSIGNED[2])}; retry verdict flaky") for note in notes)
    _verdict(killed, "red", not_run=ASSIGNED[2][-1:])
    assert _reconciled(tmp_path)[0][0] == ("shard 2/2: its session ended before its final export; "
                                           "its retry verdict is red (0 failed, 1 not run)")


@pytest.mark.parametrize("journal, gap", [
    ({"run_id": "901"}, f"s2-Windows/parallel/events.jsonl: belongs to commit {SHA} run 901"),
    ({"lane_row": False}, "shard 2/2: no proof"),
], ids=["foreign-journal", "no-lane-row"])
def test_a_journal_proves_only_the_lane_block_of_this_run(tmp_path, journal, gap):
    passes = _default_run(tmp_path)
    _journal_only(passes[2], executed=ASSIGNED[2][:1], **journal)
    _projection(passes[2] / "retry-1", [_call(node) for node in ASSIGNED[2][1:]])
    _verdict(passes[2])
    gaps, _ = _reconciled(tmp_path)
    assert any(gap in text for text in gaps), gaps


def test_an_empty_slice_is_a_proof_only_where_the_manifest_packs_nothing(tmp_path):
    """A serial lane whose only file landed on shard 2: shard 1 is assigned nothing and proves it."""
    serial = [node for node in FULL if node.startswith("tests/test_b.py")]
    owners = file_shards(WEIGHTS, 2)
    for shard in (None, 1, 2):
        block = {"full": serial, "file_weights": [[path, weight] for path, weight in WEIGHTS.items()], "pass": "serial"}
        if shard:
            block.update(shard=[shard, 2], assigned=[node for node in serial if owners[node.split("::", 1)[0]] == shard])
        _projection(tmp_path / ("manifest/m" if shard is None else f"shards/s{shard}") / "serial",
                    [_call(node) for node in block.get("assigned", [])], default_serial=block)
    empty = 3 - owners["tests/test_b.py"]
    gaps, notes = _reconciled(tmp_path, "default_serial")
    assert gaps == [] and f"shard {empty}/2: proof from attempt 1; assigned 0, executed 0" in notes
    # The shard the packing gives that file cannot claim an empty slice.
    path = tmp_path / "shards" / f"s{3 - empty}" / "serial" / "results.json"
    data = _json(path)
    data["default_serial"]["assigned"], data["reports"] = [], []
    _write(path, data)
    gaps, _ = _reconciled(tmp_path, "default_serial")
    assert f"shard {3 - empty}/2: its assignment is not slice {3 - empty} of 2 of the manifest lane" in gaps


def test_a_system_filter_reconciles_one_operating_system_of_a_shared_tree(tmp_path):
    _default_run(tmp_path, system="Windows")
    _default_run(tmp_path, system="Linux")
    assert _reconciled(tmp_path)[0], "two systems' proofs are no single lane"
    for system in ("windows", "linux"):
        assert _reconciled(tmp_path, system=system)[0] == []
        result = subprocess.run(CLI + ["reconcile-shards", "--lane", "default_parallel", "--os", system,
                                       "--manifest", str(tmp_path / "manifest"), "--shards", str(tmp_path / "shards"),
                                       "--count", "2", "--sha", SHA, "--run-id", RUN,
                                       "--summary", str(tmp_path / "summary.md")],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    assert f"## default parallel lane ({system}) reconciliation — complete" in (tmp_path / "summary.md").read_text(
        encoding="utf-8")


# ---------------------------------------------------------------- the run's verdict list and diagnostics


def test_the_run_verdict_lists_every_test_completely_and_links_accepted_issues(tmp_path):
    root = tmp_path / "evidence"
    many = [f"tests/test_many.py::test_{index:02}" for index in range(12)]
    first = _verdict(root / "ci-evidence-full-test-windows-shard-1-77-1" / "parallel", flaky=many)
    # The same pass downloaded in a second artifact is listed once; a round's directory is never a pass.
    _write(root / "default-ci-shard-1-77-1" / "parallel" / "verdict.json", first.read_text(encoding="utf-8"))
    _verdict(root / "ci-evidence-full-test-windows-shard-1-77-1" / "parallel" / "retry-1", flaky=["tests/x.py::t"])
    _verdict(root / "ci-evidence-full-test-windows-shard-1-77-1" / "serial", "red",
             failed=["tests/test_f.py::test_fails", "tests/test_k.py::test_known"],
             accepted=["tests/test_k.py::test_known"], not_run=["tests/test_n.py::test_never"])
    _write(root / "ci-evidence-full-test-linux-77-1" / "serial" / "verdict.json", "{")
    _verdict(root / "ui-ci-full-shard-2-77-1" / "host", "green", rounds=(".",))
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        CLI + ["run-verdict", "--evidence-root", str(root), "--summary", str(summary), "--out", str(tmp_path / "v.json"),
               "--required-lanes", "ci-evidence-full-test-*/parallel", "ui-ci-full-shard-9-*/host"],
        env={**os.environ, "GITHUB_REPOSITORY": "owner/repo"}, capture_output=True, text=True, encoding="utf-8",
        timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    text, facts = summary.read_text(encoding="utf-8"), _json(tmp_path / "v.json")
    assert text.startswith("## CI test verdicts — FLAKY 12, ACCEPTED 1, FAILED 1, NOT RUN 1")
    assert all(node in text for node in many) and text.count("tests/test_many.py::test_00") == 1
    assert "- tests/test_k.py::test_known — [#7](https://github.com/owner/repo/issues/7)" in text
    assert "- tests/test_f.py::test_fails (call, AssertionError; call, AssertionError)" in text
    assert text.count("tests/test_k.py::test_known") == 1, "an accepted failure is not listed as failed"
    assert "| ci-evidence-full-test-linux-77-1/serial | unreadable | 0 |" in text
    assert "**NO VERDICT** for the required lane `ui-ci-full-shard-9-*/host`." in text
    assert "tests/x.py::t" not in text and "default-ci-shard-1-77-1" not in text
    assert [row["nodeid"] for row in facts["flaky"]] == many
    assert {row["label"] for row in facts["flaky"]} == {"ci-evidence-full-test-windows-shard-1-77-1/parallel"}
    assert facts["accepted"] == [{"label": "ci-evidence-full-test-windows-shard-1-77-1/serial",
                                  "nodeid": "tests/test_k.py::test_known", "issue": 7}]
    assert [row["nodeid"] for row in facts["failed"]] == ["tests/test_f.py::test_fails"]
    assert facts["not_run"] == [{"label": "ci-evidence-full-test-windows-shard-1-77-1/serial",
                                 "nodeid": "tests/test_n.py::test_never"}]
    assert facts["missing"] == ["ui-ci-full-shard-9-*/host"] and len(facts["lanes"]) == 4
    # Diagnostic only: neither an empty tree nor a usage error fails its step.
    for arguments in (["--evidence-root", str(tmp_path / "none"), "--summary", str(summary), "--out",
                       str(tmp_path / "w.json")], ["--evidence-root", str(root)]):
        quiet = subprocess.run(CLI + ["run-verdict", *arguments], capture_output=True, text=True, timeout=30)
        assert quiet.returncode == 0, quiet.stdout + quiet.stderr
    assert "No retry verdict was found" in summary.read_text(encoding="utf-8")


def test_a_pass_rerun_in_a_later_attempt_counts_only_there(tmp_path):
    """A re-run of a red shard that comes back green must not leave its first attempt's failure
    in the table the release reads; a pass that was not re-run keeps its own verdict."""
    root, rerun, kept = tmp_path / "evidence", "tests/test_b.py::test_rerun", "tests/test_c.py::test_kept"
    shard = "ci-evidence-full-test-ubuntu-latest-shard-{}-" + RUN + "-{}"
    _verdict(root / shard.format(1, 1) / "parallel", "red", failed=[rerun])
    _verdict(root / shard.format(1, 2) / "parallel", "green", rounds=(".",), attempt="2")
    _verdict(root / shard.format(2, 1) / "parallel", "red", failed=[kept])
    result = subprocess.run(CLI + ["run-verdict", "--evidence-root", str(root), "--summary", str(tmp_path / "s.md"),
                                   "--out", str(tmp_path / "v.json")],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    facts = _json(tmp_path / "v.json")
    assert [row["nodeid"] for row in facts["failed"]] == [kept]
    assert sorted(lane["label"] for lane in facts["lanes"]) == [
        shard.format(1, 2) + "/parallel", shard.format(2, 1) + "/parallel"]


def test_summary_and_annotations_follow_the_verdict_of_a_retried_pass(tmp_path):
    flaky, failed, accepted, never = (f"tests/test_{name}.py::test_case" for name in ("flaky", "f", "k", "n"))
    green = tmp_path / "green"
    _projection(green, [_call(flaky, "failed")])
    _verdict(green, flaky=[flaky])
    red = tmp_path / "red"
    _projection(red, [_call(failed, "failed"), _call(accepted, "failed")])
    _verdict(red, "red", failed=[failed, accepted], accepted=[accepted], not_run=[never])
    summary = tmp_path / "summary.md"
    result = subprocess.run(CLI + ["summarize", "--evidence-dir", str(green), "--summary", str(summary),
                                   "--producer-outcome", "success", "--artifact-outcome", "success"],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("Retry verdict after 2 round(s): **flaky** — FLAKY 1, ACCEPTED 0, FAILED 0, NOT RUN 0."
            in summary.read_text(encoding="utf-8"))
    result = subprocess.run(CLI + ["annotate", "--evidence-dir", str(green), "--evidence-dir", str(red)],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.stdout.splitlines() == [
        f"::error title=Failed test::{failed} (call, AssertionError; call, AssertionError)",
        f"::error title=Test never ran::{never}",
        f"::warning title=Flaky test (passed on its retry)::{flaky} (first: call, AssertionError)",
        f"::warning title=Accepted failure (#7)::{accepted}"]
