"""Shards weighed by recorded durations, the run's frozen plan, retry verdicts in the proof, the run's verdict list."""
from __future__ import annotations

from collections import Counter
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR
from ouroboros.test_evidence import TEST_DURATIONS_ENV as OWNERS_TABLE_KEY
from ouroboros.test_parallelism import NESTED_WORKERS_ENV
from tests import browser_lane, default_lane
from tests.ci_evidence import file_shards, node_shards, reconcile_shards
from tests.ci_verdicts import (
    DURATIONS_ENV, TEST_DURATIONS_ENV, collect_durations, durations_table, platform_name, record_session_durations,
    recorded_durations_table,
)

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-I", "-S", str(REPO / "tests/ci_evidence.py")]
SHA, RUN = "d" * 40, "900"
SYSTEM = platform_name(sys.platform)
EXPRESSIONS = {name: f"{prefix}exclusive and {LANE_EXCLUSION_EXPR}"
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


def _test(node, *seconds, outcome="passed"):
    """One test's reports: setup, call and teardown seconds, as many phases as given (fewer: it never ended)."""
    return [_row(node, value, phase, outcome if phase == "call" else "passed")
            for phase, value in zip(("setup", "call", "teardown"), seconds)]


def _no_body(node, setup, teardown, outcome):
    """pytest's reports of a test whose setup failed or skipped: that setup and a teardown, no call."""
    return [_row(node, setup, "setup", outcome), _row(node, teardown, "teardown")]


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
    # The one-process pass's time joins its files' totals (ids stay plain: no part of the table is "serial");
    # a copy of the projection in a second artifact counts once.
    for artifact in ("default-ci-shard-Windows-1-a2", "ci-evidence-full-test-windows-shard-1-a2"):
        _projection(root / artifact / "serial", [_row("tests/test_a.py::test_s", 2.0),
                                                 _row("tests/test_x.py::test_alone", 1.0)], attempt="2",
                    default_serial=_block("serial", [1, 2]))
    # Another system; a manifest, a default lane's retry round (even one its verdict lists), another producer and
    # an unknown system add nothing.
    _projection(root / "default-ci-shard-Linux-1" / "parallel", [_row(a, 0.5)], system="Linux",
                default_parallel=_block("parallel", [1, 4]))
    _projection(root / "default-ci-manifest-Linux" / "parallel", [], system="Linux",
                default_parallel=_block("parallel"))
    _projection(root / "default-ci-shard-Linux-1" / "parallel" / "retry-1", [_row(a, 60.0)], system="Linux")
    _verdict(root / "default-ci-shard-Linux-1" / "parallel", flaky=[a])
    _projection(root / "ci-evidence-full-test-linux" / "size", [_row("tests/test_size.py::t", 3.0)], system="Linux")
    _projection(root / "elsewhere" / "parallel", [_row(a, 3.0)], system="Plan9")
    # An unsharded full-suite pass: a `parallel` or `serial` directory without a lane block.
    _projection(root / "ci-evidence-full-test-macos" / "parallel", [_row("tests/test_m.py::t", 0.125)], system="Darwin")
    page = "tests/test_ui.py::page[chromium]"
    _projection(root / "ui-ci-full-shard-1" / "host", _test(page, 4.0, 1.0, 0.0),
                system="Linux", ui_browser={"full": [page], "shard": [1, 8], "assigned": [page]})
    table = {"format": 1, "platforms": {"darwin": {"tests/test_m.py": 125}, "linux": {"tests/test_a.py": 500},
                                        "windows": {"tests/test_a.py": 4000, "tests/test_b.py": 0, "tests/test_x.py": 1000}},
             "source": {"run_id": RUN, "sha": SHA}, "ui_nodes": {"tests/test_ui.py::page[chromium]": 5000}}
    assert collect_durations(root, run_id=RUN, sha=SHA) == table
    out, payloads = tmp_path / "out" / "ci-durations.json", []
    for _ in range(2):
        result = subprocess.run(CLI + ["durations", "--evidence-root", str(root), "--run-id", RUN, "--sha", SHA,
                                       "--out", str(out)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        payloads.append(out.read_bytes())
    assert payloads[0] == payloads[1] and json.loads(payloads[0]) == table
    assert list(json.loads(payloads[0])) == sorted(table) and "windows 3 files" in result.stdout
    empty = subprocess.run(CLI + ["durations", "--evidence-root", str(tmp_path / "none"), "--run-id", RUN,
                                  "--sha", SHA, "--out", str(out)], capture_output=True, text=True, timeout=30)
    assert empty.returncode == 0 and "::warning::" in empty.stdout and _json(out)["platforms"] == {}


def _killed(directory, rows, *, block=None, lane=None, system="Linux"):
    """A session killed before its export: its journal holds the lane block it recorded at collection (a round
    records none), then the reports it got to; the kill tore the last line."""
    lines = [{"event": "lane", "nodeid": "", "block": block, "lane": lane, "environment": {"platform": system},
              "github": {"sha": SHA, "run_id": RUN, "run_attempt": "1"}}] if block else []
    lines += [dict(row, event="report") for row in rows]
    _write(Path(directory) / "events.jsonl", "".join(json.dumps(line) + "\n" for line in lines) + '{"event": "rep')


def test_a_ui_test_weighs_its_first_complete_execution_in_its_session_or_a_round(tmp_path):
    """A UI shard session stops at its deadline and scripts/ci_retry.py runs the rest in a round (run 42412862321:
    45 of 2504 tests first ran there). Each test of a pass's assignment weighs its first complete execution, the
    session's else a round's in its verdict's order, a red one too; its retry adds nothing; a crashed or
    interrupted execution has no whole time, even one whose teardown reached xdist's controller before its
    worker died (the crash report names the test still pending there); a round's test of another shard is
    that shard's alone. A complete execution ran its body: its setup passed and its call and teardown are on
    record, whatever those two's outcomes. A failed or skipped setup or a missing phase holds no place ahead of
    a later real execution, a test that never ran whole stays out, and a default lane's times do not change."""
    root = tmp_path / "evidence"
    a, b, c, d, e, f, g, x, y = (f"tests/test_ui.py::test_{name}" for name in "abcdefgxy")
    full = [a, b, c, d, e, f, g, x, y]
    one = root / "ui-ci-full-shard-1-1" / "host"
    _projection(one, [*_test(a, 1.0, 2.0, 0.5), _row(a, 9.0, subtest=True),
                      *_test(b, 0.5, 10.0, 0.5, outcome="failed"),
                      _row(d, 0.4, "setup"), _row(d, 0.0, "crash", "failed"), _row(e, 0.4, "setup"),
                      *_test(g, 0.5, 0.5, 0.5), _row(g, 0.0, "crash", "failed")],
                system="Linux", ui_browser={"full": full, "shard": [1, 2], "assigned": [a, b, c, d, e, f, g]})
    _projection(one / "retry-1", [*_test(b, 0.1, 0.1, 0.1), *_test(c, 0.2, 1.0, 0.3), *_test(d, 0.3, 2.0, 0.2),
                                  _row(e, 0.1, "setup"), _row(e, 0.0, "crash", "failed"),
                                  *_test(g, 0.25, 0.25, 0.25), *_test(x, 5.0, 5.0, 5.0)], system="Linux")
    _verdict(one, "red", flaky=[b, d, g], failed=[e], not_run=[f])
    # Shard 2 was killed before its export, with y in flight; the round that completed y was killed after it.
    two = root / "ui-ci-full-shard-2-1" / "host"
    _killed(two, [*_test(x, 1.0, 1.0, 1.0), _row(y, 0.5, "setup")], block="ui_browser",
            lane={"full": full, "shard": [2, 2], "assigned": [x, y]})
    _killed(two / "retry-1", _test(y, 0.2, 0.2, 0.2))
    _verdict(two, flaky=[y])
    # Shard 3: the 500 ms of a failed setup must not stand for the 31 s its retry really took.
    failed, skipped, no_setup, no_call, late, earlier, unrun, only_skipped, red_teardown, body_skip, called = (
        f"tests/test_ui.py::test_{name}" for name in ("failed", "skipped", "no_setup", "no_call", "late", "earlier",
                                                      "unrun", "only_skipped", "red_teardown", "body_skip", "called"))
    three = root / "ui-ci-full-shard-3-1" / "host"
    assigned = [failed, skipped, no_setup, no_call, late, earlier, unrun, only_skipped, red_teardown, body_skip, called]
    _projection(three, [*_no_body(failed, 0.4, 0.1, "failed"), *_no_body(skipped, 0.2, 0.0, "skipped"),
                        _row(no_setup, 2.0), _row(no_setup, 0.5, "teardown"),
                        _row(no_call, 0.5, "setup"), _row(no_call, 9.0, subtest=True), _row(no_call, 0.5, "teardown"),
                        _row(called, 0.4, "setup", "failed"), _row(called, 1.0), _row(called, 0.1, "teardown"),
                        *_no_body(earlier, 0.1, 0.1, "failed"), *_no_body(unrun, 0.1, 0.1, "failed"),
                        *_no_body(only_skipped, 0.1, 0.0, "skipped"), _row(red_teardown, 0.5, "setup"),
                        _row(red_teardown, 1.0), _row(red_teardown, 0.5, "teardown", "failed"),
                        *_test(body_skip, 0.5, 1.0, 0.5, outcome="skipped")],
                system="Linux", ui_browser={"full": assigned, "shard": [3, 3], "assigned": assigned})
    _projection(three / "retry-1", [*_test(failed, 1.0, 29.0, 1.0), *_test(skipped, 1.0, 4.0, 1.0),
                                    *_test(no_setup, 1.0, 1.0, 1.0), *_test(no_call, 2.0, 2.0, 2.0),
                                    *_no_body(late, 0.1, 0.1, "failed"), *_test(earlier, 1.0, 1.0, 1.0),
                                    *_no_body(unrun, 0.1, 0.1, "failed"), *_test(red_teardown, 9.0, 9.0, 9.0),
                                    *_test(called, 1.0, 2.0, 1.0)],
                system="Linux")
    _projection(three / "retry-2", [*_test(late, 3.0, 3.0, 1.0), *_test(earlier, 9.0, 9.0, 9.0)], system="Linux")
    _verdict(three, "red", rounds=(".", "retry-1", "retry-2"), flaky=[failed, late], failed=[unrun])
    _projection(root / "default-ci-shard-Linux-1-1" / "parallel",
                _no_body("tests/test_d.py::test_d", 0.4, 0.1, "failed"), system="Linux",
                default_parallel=_block("parallel", [1, 1]))
    table = collect_durations(root, run_id=RUN, sha=SHA)
    assert table["ui_nodes"] == {a: 3500, b: 11000, c: 1500, d: 2500, g: 750, x: 3000, y: 600,
                                 failed: 31000, skipped: 6000, no_setup: 3000, no_call: 6000, late: 7000, earlier: 3000,
                                 red_teardown: 2000, body_skip: 2000, called: 4000}
    assert table["platforms"] == {"linux": {"tests/test_d.py": 500}}


def test_of_one_ui_shard_only_the_highest_attempt_that_timed_a_test_counts(tmp_path):
    """A shard re-run in a later attempt replaces the earlier one whole, rounds included; one that timed nothing
    replaces nothing; a pass downloaded twice counts once. A round no verdict lists is no part of its pass, and a
    killed default-lane session adds nothing."""
    root = tmp_path / "evidence"
    a, b, c = (f"tests/test_ui.py::test_{name}" for name in "abc")
    lane = {"full": [a, b, c], "shard": [1, 1], "assigned": [a, b, c]}
    first = root / "ui-ci-full-shard-1-1" / "host"
    _projection(first, _test(a, 1.0, 1.0, 1.0), system="Linux", ui_browser=lane)
    _projection(first / "retry-1", [*_test(b, 2.0, 2.0, 2.0), *_test(c, 3.0, 3.0, 3.0)], system="Linux")
    verdict = _verdict(first)
    _killed(root / "default-ci-shard-Linux-1" / "parallel", _test("tests/test_d.py::test_d", 1.0, 1.0, 1.0),
            block="default_parallel", lane=_block("parallel", [1, 1]))

    def table():
        return collect_durations(root, run_id=RUN, sha=SHA)
    assert table()["ui_nodes"] == {a: 3000, b: 6000, c: 9000} and table()["platforms"] == {}
    verdict.unlink()
    assert table()["ui_nodes"] == {a: 3000}
    _verdict(first)
    _projection(root / "ui-ci-full-shard-1-2" / "host", _test(a, 0.5, 0.5, 0.5), system="Linux", attempt="2",
                ui_browser=lane)
    _projection(root / "ui-ci-full-shard-1-3" / "host", [_row(a, 0.1, "setup"), _row(a, 0.0, "crash", "failed")],
                system="Linux", attempt="3", ui_browser=lane)
    _projection(root / "ui-ci-full-shard-1-3" / "host" / "retry-1", _test(b, 1.0, 1.0), system="Linux")
    _verdict(root / "ui-ci-full-shard-1-3" / "host", "red", failed=[a], attempt="3")
    assert table()["ui_nodes"] == {a: 1500}
    shutil.copytree(root / "ui-ci-full-shard-1-2", root / "ui-ci-full-shard-1-2-copy")
    assert table()["ui_nodes"] == {a: 1500}
    _projection(root / "ui-ci-full-shard-1-4" / "host", _no_body(a, 0.1, 0.1, "failed"), system="Linux", attempt="4",
                ui_browser=lane)
    assert table()["ui_nodes"] == {a: 1500}  # a whole attempt of failed setups timed nothing


def test_an_owned_session_folds_its_whole_file_times_into_the_owners_table(tmp_path, monkeypatch):
    """Each session replaces its files' entries for its system, drops files the tree no longer has and files a
    crashed worker left partial, keeps the other systems' entries (a legacy `serial` part is dropped), and
    starts afresh over an unreadable table. `durations_table` reads the owner's table when CI names none; a table
    no session wrote yet is no table and no reason; CI's table wins."""
    assert TEST_DURATIONS_ENV == OWNERS_TABLE_KEY  # the stdlib reader and the product spell one key
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    for name in ("test_a.py", "test_s.py", "test_n.py"):
        (tree / "tests" / name).write_text("", encoding="utf-8")
    table = tmp_path / "state" / "test-durations.json"
    _write(table, "{garbage")
    first = [_row("tests/test_a.py::test_one", 0.25, "setup"), _row("tests/test_a.py::test_one", 1.5),
             _row("tests/test_a.py::test_one", 0.7, subtest=True),  # within its test's call: never counted twice
             _row("tests/test_s.py::test_s", 2.0), _row("tests/test_gone.py::test_g", 9.0),
             _row("tests/test_n.py::test_crash", 1.0), _row("tests/test_n.py::test_crash", 0.0, "crash", "failed")]
    assert record_session_durations(table, first, system="Darwin", rootpath=tree, run_id="session 1") == ""
    assert _json(table) == {"format": 1, "platforms": {"darwin": {"tests/test_a.py": 1750, "tests/test_s.py": 2000}},
                            "source": {"run_id": "session 1", "sha": ""}, "ui_nodes": {}}
    # A later session replaces a file's entry; another system stays; a legacy `serial` part is dropped.
    _write(table, {**_json(table), "serial": {"darwin": {"tests/test_s.py": 2000}}})
    assert record_session_durations(table, [_row("tests/test_a.py::test_one", 3.0)], system="Darwin",
                                    rootpath=tree, run_id="session 2") == ""
    assert record_session_durations(table, [_row("tests/test_s.py::test_s", 1.0)], system="Linux",
                                    rootpath=tree, run_id="session 3", sha="abc") == ""
    assert _json(table)["platforms"] == {"darwin": {"tests/test_a.py": 3000, "tests/test_s.py": 2000},
                                         "linux": {"tests/test_s.py": 1000}}
    assert "serial" not in _json(table)
    assert record_session_durations(table, [_row("tests/test_a.py::test_one", 1.0)], system="Darwin",
                                    rootpath=tree, run_id="session 4") == ""
    assert _json(table)["source"] == {"run_id": "session 4", "sha": ""}
    assert record_session_durations(table, [_row("tests/test_n.py::t", 1.0, "crash")], system="Darwin",
                                    rootpath=tree, run_id="session 5") == "no whole test file ran"
    assert _json(table)["source"]["run_id"] == "session 4"
    # The owner's table sizes and orders the serial groups; it never weighs a lane's shards, which every session
    # of a run must pack alike (only CI's run table does that). A table nobody wrote yet is no table and no reason.
    monkeypatch.delenv(DURATIONS_ENV, raising=False)
    monkeypatch.setenv(TEST_DURATIONS_ENV, str(table))
    assert recorded_durations_table()[0]["source"]["run_id"] == "session 4" and durations_table() == (None, "")
    monkeypatch.setenv(TEST_DURATIONS_ENV, str(tmp_path / "nobody" / "test-durations.json"))
    assert recorded_durations_table() == (None, "")
    monkeypatch.setenv(DURATIONS_ENV, str(_table(tmp_path)))
    monkeypatch.setenv(TEST_DURATIONS_ENV, str(table))
    assert recorded_durations_table()[0]["source"]["run_id"] == durations_table()[0]["source"]["run_id"] == RUN


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
    ('{"format": 1, "source": {"run_id": "1"}, "platforms": {"linux": {"tests/a.py": -1}}}',
     "the durations table has another shape"),
], ids=["missing", "garbage", "array", "negative"])
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


def _session(tmp_path, suite, entry, lane, public, *options, env=None, paths=("tests/",)):
    args = [*paths, "-p", "tests.conftest", "--rootdir", str(suite), "--confcutdir", str(suite), "-o", "addopts=",
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


def test_an_owned_session_records_its_durations_at_its_end_and_a_narrowed_one_does_not(tmp_path):
    """The owner names the table through the environment; the session drops the name at collection (its tests'
    children inherit no table) and folds its whole files' seconds in at its end; a `-k` session writes nothing."""
    suite, entry = _suite(tmp_path, {
        "test_a.py": f"import os\ndef test_one():\n    assert {TEST_DURATIONS_ENV!r} not in os.environ\n",
        "test_b.py": "def test_two():\n    pass\n"})
    table = tmp_path / "owner" / "test-durations.json"
    _session(tmp_path, suite, entry, "parallel", tmp_path / "public", env={TEST_DURATIONS_ENV: str(table)})
    recorded = _json(table)
    assert set(recorded["platforms"][SYSTEM]) == {"tests/test_a.py", "tests/test_b.py"}
    assert "serial" not in recorded and recorded["source"]["run_id"].startswith("session ")
    narrowed = tmp_path / "owner" / "narrowed.json"
    _session(tmp_path, suite, entry, "parallel", tmp_path / "public-k", "-k", "two", env={TEST_DURATIONS_ENV: str(narrowed)})
    assert not narrowed.exists()


def test_a_shard_without_a_file_of_a_pass_proves_its_empty_slice(tmp_path):
    """Few files carry `serial`: with every OS sharded, a shard can get none of a pass. It runs nothing,
    exits 0 (not pytest's 5) and proves the empty slice the manifest's packing gives it."""
    suite, entry = _suite(tmp_path, {
        "test_parallel_only.py": PARAMETRIZED.format(3),
        "test_serial_only.py": "import pytest\npytestmark = pytest.mark.serial('exclusive')\n" + PARAMETRIZED.format(2)})
    # Node counts over both passes: the parallel file (3) on shard 1, the serial file (2) on shard 2.
    for lane, empty, options in (("parallel", 2, ["-n", "2", "--dist", "loadgroup"]), ("serial", 1, [])):
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


def test_a_pass_over_modules_holding_none_of_their_tests_is_green_and_an_empty_module_is_not(tmp_path):
    """journal-persist, scripts/run_tests.py's battery and the gate run each pass over explicit modules, unsharded:
    a pass that holds none of their tests ran all it was asked (exit 0, said so), in one process or under xdist;
    modules that collect nothing, or a narrowed run, keep pytest's 5."""
    suite, entry = _suite(tmp_path, {
        "test_parallel_only.py": PARAMETRIZED.format(2), "test_nothing.py": "X = 1\n",
        "test_exclusive_only.py": "import pytest\npytestmark = pytest.mark.serial('exclusive')\n" + PARAMETRIZED.format(2)})
    result, data = _session(tmp_path, suite, entry, "serial", tmp_path / "serial", paths=["tests/test_parallel_only.py"])
    assert (result.returncode, data["session_exit_code"], data["tests_collected"]) == (0, 0, 0), result.stdout + result.stderr
    assert "DEFAULT_PASS_EMPTY: the serial pass holds none of the 2 collected test(s)" in result.stdout
    result, data = _session(tmp_path, suite, entry, "parallel", tmp_path / "parallel", "-n", "2", "--dist", "loadgroup",
                            paths=["tests/test_exclusive_only.py"])
    assert (result.returncode, data["session_exit_code"]) == (0, 0), result.stdout + result.stderr
    assert "DEFAULT_PASS_EMPTY: the parallel pass holds none of the 2 collected test(s)" in result.stdout
    result, data = _session(tmp_path, suite, entry, "serial", tmp_path / "nothing", paths=["tests/test_nothing.py"])
    assert (result.returncode, data["session_exit_code"]) == (5, 5), result.stdout + result.stderr
    result, data = _session(tmp_path, suite, entry, "serial", tmp_path / "narrowed", "-k", "none_match",
                            paths=["tests/test_parallel_only.py"])
    assert (result.returncode, data["session_exit_code"]) == (5, 5), result.stdout + result.stderr


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
    (lambda shard, lane: shard is None and lane["node_weights"].reverse(), "invalid ui_browser block"),
], ids=["packed", "positional-slice", "weights-not-the-lane"])
def test_a_packed_ui_lane_reconciles_only_against_the_manifests_packing(tmp_path, change, gap):
    gaps, _ = _ui_run(tmp_path, change=change)
    assert gaps == [] if gap is None else any(gap in text for text in gaps), gaps


# ---------------------------------------------------------------- the run's frozen plan


PLANNED, BEGAN = "500", "2026-11-08T12:00:00Z"
ACTION = REPO / ".github" / "actions" / "shard-plan" / "action.yml"
WORKFLOWS = REPO / ".github" / "workflows"
_POSIX_STEP = pytest.mark.skipif(os.name == "nt", reason="these workflow steps run on the Linux runner only")


@pytest.fixture(scope="module")
def shard_plan():
    spec = importlib.util.spec_from_file_location("ci_shard_plan_script", REPO / "scripts" / "ci_shard_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hour(hour):
    return f"2026-11-08T{hour:02}:00:00Z"


def _artifact(run, created, *, ui=None, files=None, number=None, source=None, expired=False, commit="b" * 40,
              head_sha="b" * 40):
    """One earlier run's `ci-durations` artifact as GitHub serves it: (its listing row, its real zip).
    Without `ui` its table has no `ui_nodes` key at all, a shape `durations_table` accepts; `commit=None` leaves
    `sha` out of its `source`."""
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("ci-durations.json", json.dumps({
            "format": 1, "platforms": {} if files is None else {SYSTEM: files},
            "source": {"run_id": str(run) if source is None else source, **({} if commit is None else {"sha": commit})},
            **({} if ui is None else {"ui_nodes": ui})}))
    return {"id": number or int(run) * 10, "expired": expired, "created_at": created,
            "workflow_run": {"id": int(run), "head_sha": head_sha}}, body.getvalue()


def _listing(page=1):
    return f"repos/o/r/actions/artifacts?name=ci-durations&per_page=100&page={page}"


def _paged(*pages, began=BEGAN):
    """Every `gh api` answer the plan needs, by path, its artifacts listed page by page."""
    return {f"repos/o/r/actions/runs/{PLANNED}": json.dumps({"created_at": began}).encode(),
            **{_listing(number): json.dumps({"artifacts": [row for row, _ in page]}).encode()
               for number, page in enumerate(pages, 1)},
            **{f"repos/o/r/actions/artifacts/{row['id']}/zip": body for page in pages for row, body in page}}


def _answers(*artifacts, began=BEGAN):
    return _paged(artifacts, began=began)


def _prepared(module, answers, asked=None):
    def read(path, _seconds):
        if asked is not None:
            asked.append(path)
        return answers[path]  # A path the fixture does not hold is an unreadable answer, as `gh api` failing is.
    return module.prepare("o/r", PLANNED, SHA, attempt="1", read=read)


def _reasons(plan):
    return {str(row.get("run_id")): row["reason"] for row in plan["skipped"]}


def test_a_ui_node_weighs_the_median_low_of_the_last_three_earlier_tables_and_files_the_newest_table(shard_plan):
    """Median_low stays an integer some table observed; a node three tables do not all time takes the median of those
    that do, and the plan says how many did; the fourth table and the older file weights do not count."""
    a, b, c, d = (f"tests/test_ui.py::test_{name}" for name in "abcd")
    plan, table = _prepared(shard_plan, _answers(
        _artifact(38, _hour(9), ui={a: 200, c: 8, d: 5}),
        _artifact(40, _hour(11), ui={a: 100, b: 10}, files={"tests/x.py": 1000}),
        _artifact(37, _hour(8), ui={a: 99_999, d: 99_999}, files={"tests/x.py": 7}),
        _artifact(39, _hour(10), ui={a: 300, b: 20, c: 7}, files={"tests/x.py": 5000})))
    assert table == {"format": 1, "platforms": {SYSTEM: {"tests/x.py": 1000}},
                     "source": {"run_id": f"shard-plan-{PLANNED}", "sha": SHA},
                     "ui_nodes": {a: 200, b: 10, c: 7, d: 5}}
    assert plan["durations"] == "ci-durations.json"
    assert plan["target"] == {"run_id": PLANNED, "run_attempt": "1", "sha": SHA, "created_at": BEGAN}
    assert [row["run_id"] for row in plan["ui_sources"]] == ["40", "39", "38"]
    assert plan["ui_sources"][0] == {"artifact": 400, "run_id": "40", "sha": "b" * 40, "created_at": _hour(11),
                                     "ui_nodes": 2}
    assert plan["platform_source"]["run_id"] == "40"
    assert plan["ui_node_observations"] == {a: 3, b: 2, c: 2, d: 1}
    assert plan["skipped"] == []  # The planner stopped reading at the third table that had both.


def test_only_another_runs_earlier_table_weighs_a_run_and_a_run_counts_once(shard_plan):
    """Neither this run, a later table, an expired one, an older copy of a run's table nor a table naming another
    run can reach the weights, whatever the listing shows."""
    node = "tests/test_ui.py::test_a"
    plan, table = _prepared(shard_plan, _answers(
        _artifact(int(PLANNED), _hour(11), ui={node: 9000}),  # Its own earlier attempt, before the cutoff in time.
        _artifact(70, BEGAN, ui={node: 9001}),  # The cutoff itself is not earlier.
        _artifact(71, _hour(13), ui={node: 9002}),
        _artifact(72, _hour(5), ui={node: 9003}, expired=True),
        _artifact(40, _hour(10), ui={node: 10}, files={"tests/x.py": 1}),
        _artifact(40, _hour(9), ui={node: 9004}, number=401),  # An older artifact of the same run.
        _artifact(41, _hour(8), ui={node: 9005}, source=PLANNED),  # A copy: it names another run.
        _artifact(38, _hour(7), ui={node: 30})))
    assert table["ui_nodes"] == {node: 10}  # median_low of 10 and 30
    assert [row["run_id"] for row in plan["ui_sources"]] == ["40", "38"] and plan["ui_node_observations"] == {node: 2}
    assert _reasons(plan) == {
        PLANNED: "this run's own", "70": "created after this run", "71": "created after this run", "72": "expired",
        "40": "a newer artifact of its run was read", "41": "not a table this run recorded"}


@pytest.mark.parametrize("no_ui", [{}, {"ui": {}}], ids=["no-ui-nodes-key", "empty-ui-nodes"])
def test_a_ui_only_table_cannot_displace_the_file_weights_nor_a_file_only_table_a_ui_place(
        shard_plan, tmp_path, monkeypatch, no_ui):
    """A file-only table in either shape the table's reader accepts keeps its file weights and takes no UI place."""
    node = "tests/test_ui.py::test_a"
    file_only = _artifact(39, _hour(10), files={"tests/x.py": 700}, **no_ui)
    plan, table = _prepared(shard_plan, _answers(
        _artifact(40, _hour(11), ui={node: 100}),
        file_only,
        _artifact(38, _hour(9), ui={node: 300}),
        _artifact(37, _hour(8), ui={node: 500}, files={"tests/x.py": 9})))
    assert table["platforms"] == {SYSTEM: {"tests/x.py": 700}} and plan["platform_source"]["run_id"] == "39"
    assert table["ui_nodes"] == {node: 300} and [row["run_id"] for row in plan["ui_sources"]] == ["40", "38", "37"]
    # Alone it still weighs the files: the real readers pack by them, and the UI lane says why it keeps its positions.
    plan, table = _prepared(shard_plan, _answers(file_only))
    assert table == {"format": 1, "platforms": {SYSTEM: {"tests/x.py": 700}},
                     "source": {"run_id": f"shard-plan-{PLANNED}", "sha": SHA}, "ui_nodes": {}}
    assert plan["durations"] == "ci-durations.json" and plan["platform_source"]["run_id"] == "39"
    assert plan["ui_sources"] == [] and plan["ui_node_observations"] == {}
    monkeypatch.setenv(DURATIONS_ENV, str(_write(tmp_path / "ci-durations.json", table)))
    assert default_lane.file_weights({"tests/x.py": 3}) == ({"tests/x.py": 700}, f"durations:shard-plan-{PLANNED}")
    assert _ui_lane(tmp_path, UI, "1/2")[0]["weights_source"] == (
        f"positions: the durations of run shard-plan-{PLANNED} record no UI node")


@pytest.mark.parametrize("damage, reason", [
    (lambda answers: answers.pop(f"repos/o/r/actions/runs/{PLANNED}"),
     "the artifact history is unavailable (KeyError)"),
    (lambda answers: answers.__setitem__(f"repos/o/r/actions/runs/{PLANNED}", b"{}"),
     "the artifact history is unavailable (KeyError)"),
    (lambda answers: answers.pop(_listing()), "the artifact history is unavailable (KeyError)"),
    (lambda answers: answers.__setitem__(f"repos/o/r/actions/artifacts/{400}/zip", b"not a zip"),
     "no earlier ci-durations table is usable"),
], ids=["no-run", "no-created-at", "no-listing", "unreadable-table"])
def test_a_plan_that_could_not_read_the_history_freezes_the_positional_split_and_says_why(shard_plan, damage, reason):
    answers = _answers(_artifact(40, _hour(11), ui={"tests/test_ui.py::test_a": 100}, files={"tests/x.py": 1}))
    damage(answers)
    plan, table = _prepared(shard_plan, answers)
    assert table is None and plan["durations"] is None and plan["reason"] == reason
    assert plan["format"] == 1 and plan["target"]["run_id"] == PLANNED


@pytest.mark.parametrize("damage, error", [("container", "BadZipFile"), ("deflate", "error")])
def test_an_unreadable_newest_table_is_passed_over_by_name_and_the_rest_still_weigh(shard_plan, damage, error):
    node = "tests/test_ui.py::test_a"
    answers = _answers(_artifact(40, _hour(11), ui={node: 100}, files={"tests/x.py": 1}),
                       _artifact(39, _hour(10), ui={node: 300}, files={"tests/x.py": 3}))
    address = "repos/o/r/actions/artifacts/400/zip"
    if damage == "container":
        answers[address] = b"truncated"
    else:
        with zipfile.ZipFile(io.BytesIO(answers[address])) as archive:
            content = archive.read("ci-durations.json")
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("ci-durations.json", content)
            info = archive.getinfo("ci-durations.json")
        # The ZIP directory is intact; reserved DEFLATE block type 3 makes its payload unreadable.
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        broken = bytearray(body.getvalue())
        broken[offset] = 0x07
        answers[address] = bytes(broken)
    plan, table = _prepared(shard_plan, answers)
    assert table["ui_nodes"] == {node: 300} and table["platforms"] == {SYSTEM: {"tests/x.py": 3}}
    assert _reasons(plan) == {"40": f"unreadable ({error})"}


def test_nothing_weighs_nothing_and_the_collector_table_is_what_the_plan_reads(shard_plan, tmp_path):
    """A table of no weights is no plan; a table the real collector wrote passes the plan's own reader unchanged."""
    plan, table = _prepared(shard_plan, _answers(_artifact(40, _hour(11))))
    assert table is None and _reasons(plan) == {"40": "no weight this plan still needs"}
    page = "tests/test_ui.py::page[chromium]"
    root = tmp_path / "evidence"
    _projection(root / "ui-ci-full-shard-1-1" / "host", _test(page, 1.0, 2.0, 0.5), system="Linux",
                ui_browser={"full": [page], "shard": [1, 8], "assigned": [page]})
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("ci-durations.json", json.dumps(collect_durations(root, run_id="40", sha="b" * 40)))
    row = {"id": 400, "expired": False, "created_at": _hour(11), "workflow_run": {"id": 40, "head_sha": "b" * 40}}
    plan, table = _prepared(shard_plan, _answers((row, body.getvalue())))
    assert table["ui_nodes"] == {page: 3500} and plan["ui_sources"][0]["run_id"] == "40"


@pytest.mark.parametrize("commit, head_sha", [
    ("a" * 40, "b" * 40), ("", "b" * 40), (None, "b" * 40), ("b" * 40, ""), ("b" * 40, None), ("", ""),
], ids=["another-commit", "empty-sha", "no-sha", "listing-empty", "listing-null", "both-empty"])
def test_a_table_speaks_only_for_the_commit_its_artifacts_run_was_on(shard_plan, commit, head_sha):
    """Naming its run is not enough: a stale or copied file can do that. The commit is compared with the artifact's own
    `head_sha` (the planned commit, `SHA` here, is another one by construction). Unprovable is passed over, by name."""
    node = "tests/test_ui.py::test_a"
    foreign = _artifact(41, _hour(11), ui={node: 9000}, files={"tests/x.py": 9}, commit=commit, head_sha=head_sha)
    plan, table = _prepared(shard_plan, _answers(foreign, _artifact(40, _hour(10), ui={node: 10}, files={"tests/x.py": 1})))
    assert table["ui_nodes"] == {node: 10} and table["platforms"] == {SYSTEM: {"tests/x.py": 1}}
    assert [row["run_id"] for row in plan["ui_sources"]] == ["40"] and plan["platform_source"]["run_id"] == "40"
    passed_over = {"artifact": 410, "run_id": "41", "sha": head_sha or "", "created_at": _hour(11),
                   "reason": "not a table of this run's commit"}
    assert plan["skipped"] == [passed_over]
    plan, table = _prepared(shard_plan, _answers(foreign))
    assert table is None and plan["reason"] == "no earlier ci-durations table is usable"
    assert plan["skipped"] == [passed_over] and plan["stopped_early"] is None


def _ruled_out(count, ui, first=1000):
    """Artifacts the listing alone rules out, so that none costs a table read: newer than the planned run."""
    return [_artifact(first + index, _hour(13), ui=ui) for index in range(count)]


def _zips(asked):
    return [int(path.split("/")[-2]) for path in asked if path.endswith("/zip")]


def test_the_search_reads_past_a_full_page_and_the_newest_three_still_decide(shard_plan):
    """A listing page of passed-over artifacts is no reason for positions: the tables behind it weigh the run, the newest
    three of them and no fourth, one table per run; what the listing rules out costs no table read."""
    node, wrong = "tests/test_ui.py::test_a", {"tests/test_ui.py::test_a": 9}
    ruled_out = [*_ruled_out(60, wrong), *(_artifact(2000 + index, _hour(5), ui=wrong, expired=True) for index in range(37)),
                 _artifact(int(PLANNED), _hour(11), ui=wrong)]
    stale, copied = _artifact(90, _hour(11), ui=wrong, commit="a" * 40), _artifact(91, _hour(11), ui=wrong, source=PLANNED)
    assert len(ruled_out) + 2 == shard_plan.PAGE
    older = [_artifact(40, _hour(10), ui={node: 100}, files={"tests/x.py": 7}), _artifact(39, _hour(9), ui={node: 300}),
             _artifact(38, _hour(8), ui={node: 200}), _artifact(37, _hour(7), ui=wrong, files={"tests/x.py": 9})]
    asked = []
    plan, table = _prepared(shard_plan, _paged([*ruled_out, stale, copied], older), asked)
    assert table["ui_nodes"] == {node: 200} and table["platforms"] == {SYSTEM: {"tests/x.py": 7}}
    assert [row["run_id"] for row in plan["ui_sources"]] == ["40", "39", "38"] and plan["platform_source"]["run_id"] == "40"
    assert _zips(asked) == [910, 900, 400, 390, 380] and _listing(3) not in asked
    assert Counter(row["reason"] for row in plan["skipped"]) == {
        "created after this run": 60, "expired": 37, "this run's own": 1,
        "not a table this run recorded": 1, "not a table of this run's commit": 1}
    assert plan["stopped_early"] is None


def test_a_full_page_that_holds_what_the_plan_wants_ends_the_listing(shard_plan):
    node, asked = "tests/test_ui.py::test_a", []
    usable = [_artifact(40 - age, _hour(10 - age), ui={node: 100}, files={"tests/x.py": 7}) for age in range(3)]
    _prepared(shard_plan, _paged([*_ruled_out(shard_plan.PAGE - 3, {node: 9}), *usable]), asked)
    assert _listing(1) in asked and _listing(2) not in asked and _zips(asked) == [400, 390, 380]


def test_a_later_listing_page_that_fails_keeps_what_the_earlier_ones_proved_and_says_so(shard_plan):
    node = "tests/test_ui.py::test_a"
    usable = _artifact(40, _hour(10), ui={node: 100})
    plan, table = _prepared(shard_plan, _paged([*_ruled_out(shard_plan.PAGE - 1, {node: 9}), usable]))
    assert table["ui_nodes"] == {node: 100} and [row["run_id"] for row in plan["ui_sources"]] == ["40"]
    assert plan["stopped_early"] == "listing page 2 is unreadable (KeyError)"
    plan, table = _prepared(shard_plan, _paged(_ruled_out(shard_plan.PAGE, {node: 9})))
    assert table is None and plan["reason"] == "no earlier ci-durations table is usable"
    assert plan["stopped_early"] == "listing page 2 is unreadable (KeyError)"
    assert "Its search stopped early: listing page 2 is unreadable (KeyError)." in shard_plan._summary(plan)


def test_the_search_stops_at_its_page_and_table_bounds_and_says_so(shard_plan):
    node, asked = "tests/test_ui.py::test_a", []
    pages = [_ruled_out(shard_plan.PAGE, {node: 9}, first=1000 * (number + 1)) for number in range(shard_plan.PAGES + 1)]
    plan, table = _prepared(shard_plan, _paged(*pages), asked)
    assert table is None and plan["stopped_early"] == f"the {shard_plan.PAGES} listing pages it may read are used up"
    assert asked[1:] == [_listing(page) for page in range(1, shard_plan.PAGES + 1)]  # Asked for no table of them.

    asked = []
    unusable = [_artifact(1000 + index, _hour(10), ui={node: 9}, commit="a" * 40) for index in range(shard_plan.READS + 5)]
    plan, table = _prepared(shard_plan, _answers(*unusable, _artifact(40, _hour(9), ui={node: 100})), asked)
    assert table is None and len(_zips(asked)) == shard_plan.READS
    assert plan["stopped_early"] == f"its {shard_plan.READS} table reads are used up"


def _on_a_slow_clock(shard_plan, answers, taken):
    """The plan over `gh api` reads that take `taken(path)` seconds on a clock only they move, so no test waits. A read
    that outlasts what it was allowed is stopped there, as `subprocess.run` stops `gh`."""
    now, allowed = [0.0], []

    def read(path, seconds):
        allowed.append(seconds)
        now[0] += min(taken(path), seconds)
        if taken(path) > seconds:
            raise subprocess.TimeoutExpired("gh", seconds)
        return answers[path]
    plan, table = shard_plan.prepare("o/r", PLANNED, SHA, attempt="1", read=read, clock=lambda: now[0])
    return plan, table, allowed, now[0]


HANG = 10_000
HISTORY = [_artifact(40 - age, _hour(10), ui={"tests/test_ui.py::test_a": 100}) for age in range(20)]
FULL_PAGE = [_artifact(1000 + index, _hour(13), ui={"tests/test_ui.py::test_a": 9}) for index in range(100)]


@pytest.mark.parametrize("pages, taken, allowed, reason, stopped", [
    # The run's record and the listing are slow, then every table hangs: the last read gets only what is left.
    ([HISTORY], lambda path: HANG if path.endswith("/zip") else 45, [120, 120, 120, 90],
     "no earlier ci-durations table is usable", "its 300 s of reads ran out"),
    # A stalled run record or first listing leaves nothing to search: the plan says so after that one read.
    ([HISTORY], lambda path: HANG, [120], "the artifact history is unavailable (TimeoutExpired)", None),
    ([HISTORY], lambda path: HANG if "artifacts?" in path else 1, [120, 120],
     "the artifact history is unavailable (TimeoutExpired)", None),
    # Listing page after listing page is slow: the budget covers them as it covers the tables.
    ([FULL_PAGE, FULL_PAGE, FULL_PAGE, FULL_PAGE, FULL_PAGE], lambda path: 100, [120, 120, 100],
     "no earlier ci-durations table is usable", "its 300 s of reads ran out"),
], ids=["tables-hang", "run-record-hangs", "listing-hangs", "listing-pages-slow"])
def test_slow_reads_share_one_budget_and_still_leave_a_positional_plan(shard_plan, pages, taken, allowed, reason, stopped):
    plan, table, reads, elapsed = _on_a_slow_clock(shard_plan, _paged(*pages), taken)
    assert table is None and plan["durations"] is None and plan["reason"] == reason and plan["stopped_early"] == stopped
    assert reads == allowed and elapsed <= shard_plan.BUDGET_SEC == 300
    assert plan["format"] == 1 and plan["target"]["run_id"] == PLANNED


def test_a_search_out_of_time_uses_the_tables_it_verified_and_names_the_rest(shard_plan):
    plan, table, reads, elapsed = _on_a_slow_clock(
        shard_plan, _answers(*HISTORY), lambda path: 1 if not path.endswith("/zip") or path.endswith("/400/zip") else HANG)
    assert table["ui_nodes"] == {"tests/test_ui.py::test_a": 100} and [row["run_id"] for row in plan["ui_sources"]] == ["40"]
    assert plan["stopped_early"] == "its 300 s of reads ran out" and elapsed <= shard_plan.BUDGET_SEC
    assert _reasons(plan) == {run: "unreadable (TimeoutExpired)" for run in ("39", "38", "37")}
    assert reads == [120, 120, 120, 120, 120, 57]


@_POSIX_STEP
def test_a_real_gh_that_never_answers_is_stopped_and_the_plan_is_still_written(shard_plan, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/usr/bin/env bash\nexec sleep 60\n", encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(shard_plan, "BUDGET_SEC", 0.2)
    began = time.monotonic()
    code = shard_plan.main(["--out", str(tmp_path / "plan"), "--repo", "o/r", "--run-id", PLANNED, "--run-attempt", "1",
                            "--sha", SHA])
    assert code == 0 and time.monotonic() - began < 30
    plan = _json(tmp_path / "plan" / "plan.json")
    assert plan["durations"] is None and plan["reason"] == "the artifact history is unavailable (TimeoutExpired)"
    assert not (tmp_path / "plan" / "ci-durations.json").exists()


def test_the_whole_search_fits_well_inside_both_callers_job_ceiling(shard_plan):
    """All the reads share half of the job's 10 minutes; checkout, the re-run check and the upload have the other half."""
    assert shard_plan._GH_SECONDS == 120 and shard_plan.BUDGET_SEC == 300
    for name in ("ci.yml", "ui-browser-push.yml"):
        job = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))["jobs"]["shard-plan"]
        assert job["timeout-minutes"] == 10 and shard_plan.BUDGET_SEC * 2 <= job["timeout-minutes"] * 60, name


UI_SHARDED = [f"tests/test_ui.py::test_{index:02}" for index in range(24)]


def _frozen_lane(tmp_path, monkeypatch, shard_plan):
    """Every leg of one frozen plan: the real table on disk, each shard's and the manifest's real lane block."""
    history = [_artifact(run, _hour(11 - index), files={"tests/x.py": 900},
                         ui={node: (position + 1) * (100 + 50 * index) for position, node in enumerate(UI_SHARDED)})
               for index, run in enumerate((40, 39, 38))]
    plan, table = _prepared(shard_plan, _answers(*history))
    frozen = _write(tmp_path / "plan" / "ci-durations.json", table)
    monkeypatch.setenv(DURATIONS_ENV, str(frozen))
    blocks = {shard: _ui_lane(tmp_path, UI_SHARDED, f"{shard}/9")[0] for shard in range(1, 10)}
    return plan, table, frozen, _ui_lane(tmp_path, UI_SHARDED)[0], blocks


def _prove(tmp_path, manifest, blocks):
    root = tmp_path / "proof"
    _projection(root / "manifest" / "m1" / "host", [], system="Linux", ui_browser=manifest)
    for shard, block in blocks.items():
        _projection(root / "shards" / f"s{shard}" / "host", [_row(node, 1.0) for node in block["assigned"]],
                    system="Linux", ui_browser=block)
    return reconcile_shards(root / "manifest", root / "shards", count=9, sha=SHA, run_id=RUN)


def test_the_manifest_and_all_nine_shards_pack_by_the_one_frozen_table(tmp_path, monkeypatch, shard_plan):
    """The real readers over the real table: one weight list, one honest source, a complete proof; a leg that read
    anything else is a gap in the same reconciliation."""
    plan, table, frozen, manifest, blocks = _frozen_lane(tmp_path, monkeypatch, shard_plan)
    assert manifest["weights_source"] == f"durations:shard-plan-{PLANNED}"
    assert all(block["node_weights"] == manifest["node_weights"] and block["weights_source"] == manifest["weights_source"]
               for block in blocks.values())
    assert sorted(node for block in blocks.values() for node in block["assigned"]) == UI_SHARDED
    assert [row[1] for row in manifest["node_weights"]] == [table["ui_nodes"][node] for node in UI_SHARDED]
    gaps, notes = _prove(tmp_path, manifest, blocks)
    assert gaps == [] and notes[0].endswith(f"weights durations:shard-plan-{PLANNED}")
    assert default_lane.file_weights({"tests/x.py": 3, "tests/y.py": 2}) == (
        {"tests/x.py": 900, "tests/y.py": 600}, f"durations:shard-plan-{PLANNED}")
    # A leg that read another table: the weights differ from the manifest, by name.
    data = _json(frozen)
    data["ui_nodes"][UI_SHARDED[0]] += 100_000
    other = _write(tmp_path / "plan" / "other.json", data)
    monkeypatch.setenv(DURATIONS_ENV, str(other))
    blocks[3] = _ui_lane(tmp_path, UI_SHARDED, "3/9")[0]
    gaps, _ = _prove(tmp_path / "second", manifest, blocks)
    assert "shard 3/9: node weights differ from the manifest" in gaps, gaps


def _gh_shim(tmp_path, answers):
    """`gh` and `python3` for a workflow step: `gh api PATH [...]` prints the saved answer of PATH or fails."""
    bin_dir, saved = tmp_path / "bin", tmp_path / "gh-answers"
    bin_dir.mkdir(exist_ok=True)
    saved.mkdir(exist_ok=True)
    for path, body in answers.items():
        name = "".join(character if character.isalnum() else "_" for character in path)
        (saved / name).write_bytes(body)
    shim = bin_dir / "gh"
    shim.write_text('#!/usr/bin/env bash\nname="$(printf %s "$2" | tr -c "[:alnum:]" _)"\n'
                    'test "$1" = api && test -f "$GH_ANSWERS/$name" && cat "$GH_ANSWERS/$name"\n', encoding="utf-8")
    interpreter = bin_dir / "python3"
    interpreter.write_text(f'#!/usr/bin/env bash\nexec {sys.executable!r} "$@"\n', encoding="utf-8")
    for tool in (shim, interpreter):
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GH_ANSWERS": str(saved)}


def _step(step, tmp_path, tools):
    """Run a workflow step's own shell, with the expressions a runner would have rendered."""
    summary, values = tmp_path / "step-summary", tmp_path / "github-env"
    render = {"${{ github.token }}": "t", "${{ github.repository }}": "o/r", "${{ github.run_id }}": PLANNED,
              "${{ github.run_attempt }}": "2", "${{ github.sha }}": SHA, "${{ runner.temp }}": str(tmp_path)}
    env = {key: value for key, value in os.environ.items() if key != "GITHUB_ENV"}
    for key, value in step.get("env", {}).items():
        for expression, rendered in render.items():
            value = value.replace(expression, rendered)
        env[key] = value
    env.update(tools, RUNNER_TEMP=str(tmp_path), GITHUB_ENV=str(values), GITHUB_STEP_SUMMARY=str(summary))
    completed = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step["run"]], cwd=REPO, env=env,
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
    return completed, *(path.read_text(encoding="utf-8") if path.exists() else "" for path in (values, summary))


def _action_steps():
    return yaml.safe_load(ACTION.read_text(encoding="utf-8"))["runs"]["steps"]


@_POSIX_STEP
def test_the_actions_own_step_freezes_the_plan_and_says_what_it_read(tmp_path):
    step = next(step for step in _action_steps() if "scripts/ci_shard_plan.py" in step.get("run", ""))
    node = "tests/test_ui.py::test_a"
    tools = _gh_shim(tmp_path, _answers(_artifact(40, _hour(11), ui={node: 100}, files={"tests/x.py": 1000}),
                                        _artifact(39, _hour(10), ui={node: 300})))
    completed, _, summary = _step(step, tmp_path, tools)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan, table = _json(tmp_path / "shard-plan" / "plan.json"), _json(tmp_path / "shard-plan" / "ci-durations.json")
    assert table["source"] == {"run_id": f"shard-plan-{PLANNED}", "sha": SHA} and table["ui_nodes"] == {node: 100}
    assert plan["target"] == {"run_id": PLANNED, "run_attempt": "2", "sha": SHA, "created_at": BEGAN}
    assert summary == f"Shard plan shard-plan-{PLANNED}: UI weights: median_low of runs 40, 39 (1 node(s) timed by 2); file weights: run 40.\n"
    # A history holding only a table with no `ui_nodes` key: its file weights stand and the summary names the gap.
    (tmp_path / "second").mkdir()
    completed, _, summary = _step(step, tmp_path / "second", _gh_shim(tmp_path / "second", _answers(
        _artifact(39, _hour(10), files={"tests/x.py": 700}))))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _json(tmp_path / "second" / "shard-plan" / "ci-durations.json")["platforms"] == {SYSTEM: {"tests/x.py": 700}}
    assert summary == f"Shard plan shard-plan-{PLANNED}: no earlier table records UI nodes; file weights: run 39.\n"


@_POSIX_STEP
def test_the_actions_step_freezes_an_explicit_fallback_when_gh_fails(tmp_path):
    step = next(step for step in _action_steps() if "scripts/ci_shard_plan.py" in step.get("run", ""))
    completed, _, summary = _step(step, tmp_path, _gh_shim(tmp_path, {}))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "shard-plan" / "ci-durations.json").exists()
    assert _json(tmp_path / "shard-plan" / "plan.json")["durations"] is None
    assert summary == ("Shard plan: the artifact history is unavailable (CalledProcessError); "
                       "shards keep their positions and node counts.\n")


def _lane_answers(*, table=True):
    """This run's plan as the lane's fetch step reads it: the artifact id its --jq leaves, then the real zip."""
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("plan.json", json.dumps({"format": 1, "durations": "ci-durations.json" if table else None}))
        if table:
            archive.writestr("ci-durations.json", json.dumps({"frozen": True}))
    return {f"repos/o/r/actions/runs/{PLANNED}/artifacts?name=shard-plan-{PLANNED}": b"77\n",
            "repos/o/r/actions/artifacts/77/zip": body.getvalue()}


def _lane_step(job):
    steps = yaml.safe_load((WORKFLOWS / "ui-browser.yml").read_text(encoding="utf-8"))["jobs"][job]["steps"]
    return next(step for step in steps if step.get("name") == "Fetch this run's shard plan")


@_POSIX_STEP
@pytest.mark.parametrize("job", ["ui-shard", "ui-manifest"])
def test_every_lane_job_reads_the_frozen_plan_and_cannot_choose_its_own_weights(tmp_path, job):
    step = _lane_step(job)
    tools = _gh_shim(tmp_path, _lane_answers())
    completed, values, _ = _step(step, tmp_path, tools)
    table = tmp_path / "shard-plan" / "ci-durations.json"
    assert completed.returncode == 0 and values == f"OUROBOROS_CI_DURATIONS={table}\n", completed.stdout + completed.stderr
    assert _json(table) == {"frozen": True}

    # The plan's own positional fallback: the same for every leg, and said so.
    (tmp_path / "second").mkdir()
    completed, values, _ = _step(step, tmp_path / "second", _gh_shim(tmp_path / "second", _lane_answers(table=False)))
    assert completed.returncode == 0 and values == "" and "keeps the positional split" in completed.stdout

    # No plan published, or an unreadable API, stops the leg: neither is a positional split of its own.
    for name, answers, said in (
            ("third", {f"repos/o/r/actions/runs/{PLANNED}/artifacts?name=shard-plan-{PLANNED}": b"\n"},
             "published no shard plan"), ("fourth", {}, "")):
        (tmp_path / name).mkdir()
        completed, values, _ = _step(step, tmp_path / name, _gh_shim(tmp_path / name, answers))
        assert completed.returncode != 0 and values == "" and said in completed.stdout, completed.stdout + completed.stderr


def test_one_action_freezes_the_plan_for_ci_and_the_manual_full_lane_and_the_lane_only_reads_it():
    def workflow(name):
        return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    ci, manual, lane = workflow("ci.yml"), workflow("ui-browser-push.yml"), workflow("ui-browser.yml")
    for caller in (ci, manual):
        job = caller["jobs"]["shard-plan"]
        assert [step.get("uses") for step in job["steps"]][-1] == "./.github/actions/shard-plan"
        assert any(str(step.get("uses", "")).startswith("actions/checkout") for step in job["steps"])
        assert {"contents": "read", "actions": "read"}.items() <= job["permissions"].items()
        assert caller["jobs"]["ui-smoke"]["needs"] == "shard-plan"
    assert manual["jobs"]["shard-plan"]["if"] == manual["jobs"]["ui-smoke"]["if"]
    assert "needs" not in manual["jobs"]["ui-diagnostic"]  # A partial scenario packs nothing.
    # The lane is no second producer: no step of it calls the plan action or uploads a plan.
    assert not [step for job in lane["jobs"].values() for step in job.get("steps", [])
                if "actions/shard-plan" in str(step.get("uses"))
                or ("upload-artifact" in str(step.get("uses")) and "shard-plan" in str(step.get("with")))]
    # A re-run keeps the published plan; only a run that has none freezes one and uploads it, unconditionally named.
    steps = {step["name"]: step for step in _action_steps()}
    reuse, freeze, upload = (steps[name] for name in ("Reuse this run's plan on a re-run", "Freeze the weights from the prior tables",
                                                      "Upload the plan of this run"))
    assert reuse["continue-on-error"] is True and reuse["with"]["name"] == "shard-plan-${{ github.run_id }}"
    assert freeze["if"] == upload["if"] == "${{ steps.existing.outcome != 'success' }}"
    assert upload["with"]["name"] == reuse["with"]["name"] and "overwrite" not in upload["with"]


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
], ids=["red-failed", "red-not-run", "inconsistent"])
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


# ---------------------------------------------------------------- the run's verdict list and diagnostics


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
