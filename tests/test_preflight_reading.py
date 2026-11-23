"""The local gate's reader: inert case facts, positive setup and the caller's real blocker."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ouroboros import preflight_reading as reading, preflight_receipts as receipts, preflight_runner as gate
from ouroboros.preflight_case_facts import case_key
from ouroboros.preflight_inventory import read_inventory
from ouroboros.preflight_probe import (
    _collection_path, _results_pointer, fresh_results_pointer, recorded_collection, recorded_session,
    write_worker_probe,
)
from ouroboros.test_evidence import CapturedPass


SALT = b"\x19" * 16
SESSION = "fixture-session"
A, B, C = (f"tests/test_case.py::test_{name}" for name in "abc")
MASKED = f"{A}[<redacted>]"


def row(node, *, label=None, phase="call", outcome="failed", error="AssertionError", **extra):
    return {"nodeid": label or node, "case_key": case_key(node, SALT), "phase": phase,
            "outcome": outcome, "error_type": error, **extra}


def event(kind, node):
    return {"event": kind, "nodeid": node, "case_key": case_key(node, SALT)}


def session(root, *, code=1, reports=(), events=(), final=True, complete=False, name="pytest-pass", **other):
    directory = root / name / "session"
    directory.mkdir(parents=True)
    if final:
        (directory / "results.json").write_text(json.dumps({"session": SESSION, "session_exit_code": code,
            "reports": list(reports), **other}), encoding="utf-8")
    if events or complete:
        records = [{"event": "session", "session": SESSION}, *events]
        if complete:
            records.append({"event": "starts_complete", "session": SESSION})
        (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return directory


def captured(directory=None, code=1, *, leak=""):
    result = CapturedPass(code, "last line of pytest output", leak)
    result.session_directory, result.case_salt = directory, SALT
    return result


def inventory(*names, known=True):
    return {"expected": list(names), "assignment": "known" if known else "unobserved"}


def test_distinct_cases_with_one_display_label_can_fail_or_never_run(tmp_path):
    first, second, third = (f"{A}[example-{n}]" for n in range(3))
    directory = session(tmp_path, reports=[row(first, label=MASKED), row(second, label=MASKED)], complete=True)
    (tmp_path / "inventory-parallel.jsonl").write_text(json.dumps({"event": "collection", "ids": [first, second, third]}),
                                                       encoding="utf-8")
    projection = reading.observe(tmp_path, captured(directory), "parallel", "pytest_exit",
                                 inventory(first, second, third))
    assert projection["counts"]["failed"] == 2
    assert [entry["nodeid"] for entry in projection["failed"]] == [MASKED, MASKED]
    assert projection["not_run"] == [third]
    assert projection["evidence"] == {"gate": "result.json", "results": "pytest-pass/session/results.json",
                                       "events": "pytest-pass/session/events.jsonl",
                                       "inventory": "inventory-parallel.jsonl"}


@pytest.mark.parametrize("collections,state", [([], "unobserved"), ([[]], "known"),
                                               ([[A], [B]], "disagreed"), ([[A], [A]], "known")])
def test_inventory_keeps_observed_empty_and_worker_agreement_apart(tmp_path, collections, state):
    events = [{"event": "start"}, *({"event": "collection", "ids": nodes} for nodes in collections),
              {"event": "finish", "exitcode": 5}]
    (tmp_path / "inventory-serial.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    local, problem = read_inventory(tmp_path, "serial", diagnostics=True)
    legacy, legacy_problem = read_inventory(tmp_path, "serial")
    assert local == {**legacy, "assignment": state} and problem == legacy_problem
    facts = reading.observe(tmp_path, captured(code=5), "serial", "inventory", local)
    if state == "known" and collections[0] == []:
        assert facts["not_run"] == []
    else:
        assert facts["counts"]["not_run"] is None


def test_worker_loss_is_distinct_from_a_sibling_still_in_flight(tmp_path):
    start = {"event": "start", "nodeid": B, "case_key": case_key(B, SALT)}
    facts = reading.observe(tmp_path, captured(session(tmp_path, reports=[row(A, phase="crash", error="WorkerCrash",
                       worker="gw0", timeout_callback_seconds=300)], events=[start], worker_crash={"worker": "gw2"})),
                           "parallel", "pytest_exit", None)
    assert facts["counts"]["failed"] == 0
    assert facts["in_flight"] == [B] and facts["not_run"] is None
    assert [worker["worker"] for worker in facts["worker_lost"]] == ["gw0", "gw2"]
    assert "timeout callback invoked (300s), worker subsequently lost" in reading.render(facts, 8000)


def test_torn_final_journal_preserves_observed_failure_collection_and_started_case(tmp_path):
    directory = session(tmp_path, events=[{"event": "report", **row(A)},
        {"event": "collection_error", "nodeid": "tests/test_collect.py"},
        {"event": "start", "nodeid": B, "case_key": case_key(B, SALT)}], final=False)
    (directory / "results.json").write_text("{incomplete", encoding="utf-8")
    facts = reading.observe(tmp_path, captured(directory, None), "single", "timeout", inventory(A, B, C))
    assert facts["counts"]["failed"] == 1 and facts["counts"]["worker_lost"] is None
    assert facts["collection_error"] == ["tests/test_collect.py"]
    assert facts["coverage"] == "partial" and facts["not_run"] is None and facts["in_flight"] == [B]
    assert facts["evidence"] == {"gate": "result.json", "events": "pytest-pass/session/events.jsonl"}
    assert "IN FLIGHT " + B in reading.render(facts, 8000)


@pytest.mark.parametrize("outcome,failed", [("passed", 0), ("failed", 1)])
@pytest.mark.parametrize("finished", [False, True])
def test_call_does_not_close_an_observed_start(tmp_path, outcome, failed, finished):
    rows = [event("start", A), {"event": "report", **row(A, outcome=outcome)}]
    if finished:
        rows.append(event("finish", A))
    directory = session(tmp_path, reports=[row(A, outcome=outcome)], events=rows, complete=finished)
    facts = reading.observe(tmp_path, captured(directory), "single", "pytest_exit", inventory(A, B))
    assert facts["counts"]["failed"] == failed and facts["coverage"] == "final"
    assert facts["in_flight"] == ([] if finished else [A])
    assert facts["not_run"] == ([B] if finished else None)
    if not finished:
        assert facts["gap"] == "pytest start coverage incomplete"


@pytest.mark.parametrize("damage", ["cut", "different session"])
def test_final_export_does_not_fill_missing_start_coverage(tmp_path, damage):
    directory = session(tmp_path, reports=[row(A)], events=[event("start", B)])
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"unfinished' if damage == "cut" else json.dumps(
            {"event": "starts_complete", "session": "another-session"}) + "\n")
    facts = reading.observe(tmp_path, captured(directory), "parallel", "pytest_exit", inventory(A, B, C))
    assert facts["coverage"] == "final" and facts["counts"]["failed"] == 1
    assert facts["in_flight"] == [B] and facts["not_run"] is None


def test_unreadable_pointer_cannot_supply_a_previous_pass_or_replace_legacy_text(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    root.mkdir()
    first = session(tmp_path, code=0)
    pointer = _results_pointer(root)
    pointer.write_text(str(first), encoding="utf-8")
    seen = set()
    assert recorded_session(root, root=tmp_path, seen=seen)["session_exit_code"] == 0
    assert fresh_results_pointer(root) is True and not pointer.exists()
    pointer.write_text(str(first), encoding="utf-8")  # second pass names the same exporter
    assert recorded_session(root, root=tmp_path, seen=seen) is None
    pointer.write_bytes(b"\xff")
    assert recorded_session(root) is None
    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    evidence, message = {"path": tmp_path}, "⚠️ PRE_PUSH_TEST_ERROR: serial pass\noriginal pytest explanation"
    shown = gate._with_reading(evidence, message, captured(), "serial", "pytest_exit", None, 8000)
    assert shown.splitlines() == [message.splitlines()[0],
                                  "Pytest diagnostics unavailable: no pass-owned pytest evidence",
                                  message.splitlines()[-1]]
    assert evidence["reading"]["coverage"] == "unavailable"


@pytest.mark.parametrize("mixed", [False, True])
def test_reader_gap_has_same_shape_when_missing_or_incompatible(tmp_path, monkeypatch, mixed):
    if mixed:
        def old_observe(_root, _result, _label, _cause, _inventory):
            raise AssertionError("wrong reader generation")

        monkeypatch.setattr(receipts, "precommit_reader", lambda: SimpleNamespace(
            observe=old_observe, block=lambda *a, **k: {"counts": {"leak": 0}}))
    else:
        monkeypatch.delattr(receipts, "precommit_reader")
    evidence, message = {"path": tmp_path}, "⚠️ PRE_PUSH_TEST_ERROR: serial pass\noriginal pytest explanation"
    shown = gate._with_reading(evidence, message, captured(), "serial", "pytest_exit", None, 8000)
    assert shown.splitlines()[0] == message.splitlines()[0] and shown.endswith("original pytest explanation")
    facts = evidence["reading"]
    assert "Pytest diagnostics unavailable: " + facts["gap"] in shown
    assert facts == reading.block("pytest_exit", label="serial", exit_code=1, gap=facts["gap"])


def test_lane_cleanup_fact_survives_without_a_pytest_export(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    evidence, message = {"path": tmp_path}, "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block)"
    diagnostic = gate._with_reading(evidence, message, captured(code=0, leak="inspection unavailable"),
                                    "parallel", "containment", None, 8000)
    assert diagnostic.splitlines()[0] == message
    assert "LEAK parallel: child cleanup was not confirmed" in diagnostic
    assert evidence["reading"]["counts"]["leak"] == 1


def test_display_cut_keeps_actual_worker_loss_and_private_root_out_of_exhibit(tmp_path, monkeypatch):
    dies = row(B, phase="crash", error="WorkerCrash", worker="gw0")
    many = [f"tests/test_case.py::test_{n}" for n in range(160)]
    directory = session(tmp_path, reports=[dies, *(row(node) for node in many)], worker_crash={"worker": "gw0"},
                        events=[{"event": "report", **dies}])
    (tmp_path / "inventory-parallel.jsonl").write_text(json.dumps({"event": "collection", "ids": many}),
                                                      encoding="utf-8")
    facts = reading.observe(tmp_path, captured(directory), "parallel", "pytest_exit", inventory(*many))
    text = reading.render(facts, 1200)
    assert facts["omitted"] == {"failed": 110} and facts["counts"]["not_run"] is None
    assert "WORKER LOST gw0" in text and "not shown" in text and "NOT RUN unknown" in text
    assert str(tmp_path) not in text
    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    evidence = {"path": tmp_path}
    header = "⚠️ PRE_PUSH_TEST_ERROR: parallel pass, exit 1"
    legacy = f"{header}\n" + "row\n" * 1000 + "end of pytest output"
    shown = gate._with_reading(evidence, legacy, captured(directory), "parallel", "pytest_exit", inventory(*many), 1200)
    exhibit = receipts.execution_section({"status": "not_run", "observed": [{"source": "hermetic_tests",
                                         "outcome": "failed", "diagnosis": shown}]})
    assert str(tmp_path) not in exhibit and "WORKER LOST gw0" in exhibit
    assert shown.startswith(header) and shown.endswith("end of pytest output") and len(shown) <= 1200


def test_reader_is_pinned_with_running_generation_and_never_loaded_by_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_baseline", None)
    monkeypatch.setattr(receipts, "_precommit_reader", None)
    monkeypatch.setattr(receipts, "executing_source_digest", lambda: "serving generation")
    receipts.capture_executing_source_baseline()
    assert receipts.precommit_reader() is reading
    # An inert replacement models a later lookup yielding a different reader.
    # The gate consumes the pinned object and must never make this lookup again.
    import ouroboros
    spy = SimpleNamespace(observe=lambda *a: pytest.fail("read a different generation"))
    monkeypatch.setattr(ouroboros, "preflight_reading", spy)
    receipts.capture_executing_source_baseline()
    facts = {"path": tmp_path}
    gate._with_reading(facts, "legacy", captured(session(tmp_path, reports=[row(A)])),
                       "parallel", "pytest_exit", None, 8000)
    assert facts["reading"]["failed"][0]["nodeid"] == A


def test_adjacent_process_launches_each_get_their_own_diagnostic_directory(tmp_path, monkeypatch):
    """A normal two-launch observation: no dirty module, custom payload, kill or external process."""
    from ouroboros import process_containment, test_evidence

    run_root, scratch = tmp_path / "retained", tmp_path / "scratch"
    run_root.mkdir()
    scratch.mkdir()
    (scratch / gate._PRECOMMIT_READING_MARKER).touch()
    (scratch / test_evidence.EVIDENCE_POINTER).write_text(str(run_root), encoding="utf-8")
    calls = []

    def spawn(_argv, *, env, **_):
        calls.append(env)
        # The controller of the second invocation left its own export; the first
        # never reached that point. Neither fallback scans a neighbor's folders.
        if len(calls) == 2:
            directory = Path(env["OUROBOROS_PRECOMMIT_SESSION_DIR"])
            directory.mkdir()
            node = {**row(A), "case_key": case_key(A, bytes.fromhex(env["OUROBOROS_PRECOMMIT_CASE_SALT"]))}
            (directory / "results.json").write_text(json.dumps({"session_exit_code": 1, "reports": [node]}),
                                                        encoding="utf-8")
        return SimpleNamespace(wait=lambda **_: 1, poll=lambda: 1)

    monkeypatch.setattr(process_containment, "ProcessContainer", lambda **_: SimpleNamespace(
        spawn=spawn, reap=lambda: "", close=lambda: None, token=""))
    monkeypatch.setattr(gate, "_preflight_env", lambda *a, **k: {})
    first, second = (gate._execute_lane(["pytest"], tmp_path, scratch, 10, "pytest") for _ in range(2))
    assert first.session_directory.parent != second.session_directory.parent
    assert first.case_salt != second.case_salt
    assert first.session_directory.is_relative_to(run_root) and second.session_directory.is_relative_to(run_root)
    assert reading.observe(run_root, first, "parallel", "pytest_exit", None)["coverage"] == "unavailable"
    assert reading.observe(run_root, second, "serial", "pytest_exit", None)["failed"][0]["nodeid"] == A


@pytest.mark.parametrize("codes,counts,cause", [([0, 1], [1, 1], "pytest_exit"),
    ([0], [None], "parallelism"), ([5, 5], [None, None], "all_passes_empty"),
    ([0, 0], [0, 0], "all_passes_empty"), ([0, 0], [1, None], None),
    ([0, 0], [1, 0], None), ([], [], "plugin_missing")], ids=[
        "second-pass-red", "parallelism-without-count", "both-exits-empty", "both-counts-empty",
        "second-count-unknown", "one-empty-pass", "before-pytest"])
def test_gate_return_persists_the_same_block_without_retry(tmp_path, monkeypatch, codes, counts, cause):
    from ouroboros import test_evidence, test_parallelism
    from ouroboros.test_evidence import EVIDENCE_POINTER

    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    monkeypatch.setattr(test_parallelism, "reserve", lambda *a, **k: (None, None))
    monkeypatch.setattr(gate, "install_worker_probe", lambda *a: "probe")
    monkeypatch.setattr(gate, "recorded_admission", lambda *a: None)
    monkeypatch.setattr(gate, "_verify_preflight_plugins",
                        lambda *a: ["fixture plugin missing"] if cause == "plugin_missing" else [])
    monkeypatch.setattr(gate, "_capture_source_index_tree", lambda *a: (None, None))
    monkeypatch.setattr(gate, "_assemble_live_source", lambda *a: None)
    monkeypatch.setattr(gate, "_run_git", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(gate, "_release_hermetic_tree", lambda *a: None)
    monkeypatch.setattr(gate, "run_node_tests", lambda *a: None)
    monkeypatch.setattr(gate, "_warm_bytecode", lambda *a: "")
    monkeypatch.setattr(gate, "observed_worker_ids", lambda *a: set() if cause == "parallelism" else {"gw0", "gw1"})
    monkeypatch.setattr(gate, "allocated_worker_count", lambda *a: 2)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    repo, scratch, out = (tmp_path / name for name in ("repo", "scratch", "out"))
    for path in (repo / "tests", repo / ".git", scratch, out):
        path.mkdir(parents=True)
    monkeypatch.setattr(gate, "gate_root", lambda **kw: scratch)
    pending = iter(zip(codes, counts))
    ran = []

    def run(_cmd, _candidate, temp, *_rest):
        code, count = next(pending)
        ran.append(code)
        if count is not None:
            _collection_path(temp).write_text(str(count), encoding="ascii")
        dest = Path((temp / EVIDENCE_POINTER).read_text(encoding="utf-8"))
        name = f"pytest-fixture-{len(ran)}"
        directory = session(dest, name=name, code=code, reports=[row(A)] if code == 1 else [], final=code == 1)
        return captured(directory, code)

    monkeypatch.setattr(gate, "_execute_pytest_pass", run)
    monkeypatch.setattr(test_evidence, "evidence_directory", lambda *a, **k: out)
    message = gate.run_hermetic_pytest(repo, timeout=60, phase=gate.PRE_COMMIT_PHASE)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == message and ran == codes
    assert [lane["tests_collected"] for lane in result["lanes"] if lane["lane"] == "pytest"] == counts
    if cause is None:
        assert message is None and "reading" not in result
        return
    assert result["reading"]["cause"] == cause
    if cause == "pytest_exit":
        assert "FAILED " + A in message
        assert result["reading"]["evidence"]["results"] == "pytest-fixture-2/session/results.json"
        assert result["reading"]["attempts"] == 1
    elif cause == "parallelism":
        assert "PREFLIGHT_PARALLELISM_LOST" in message
        assert result["reading"]["attempts"] == 1
    elif cause == "plugin_missing":
        assert result["reading"]["pass"] is None and result["reading"]["attempts"] == 0
    else:
        assert result["reading"]["pass"] is None and result["reading"]["attempts"] is None


def test_bound_parsed_export_is_not_read_again(tmp_path, monkeypatch):
    directory = session(tmp_path, reports=[row(A)], complete=True)
    final = directory / "results.json"
    parsed = json.loads(final.read_text(encoding="utf-8"))
    read_text = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path != final, "read the already parsed pass a second time"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    evidence = {"path": tmp_path}
    shown = gate._with_reading(evidence, "original", captured(directory), "single", "pytest_exit", None, 8000,
                               recorded=parsed)
    assert shown.startswith("original\n") and "FAILED " + A in shown
    assert evidence["reading"]["coverage"] == "final"


@pytest.mark.parametrize("second_finished", [False, True])
def test_controller_attests_starts_only_after_each_worker_finishes(second_finished):
    from tests.ci_evidence import _Results

    result = object.__new__(_Results)
    result.case_salt = SALT
    result.config = SimpleNamespace(_ci_evidence_session=SESSION)
    result.journal = BytesIO()
    result.started_workers, result.finished_workers = set(), set()
    first, second = (SimpleNamespace(gateway=SimpleNamespace(id=f"gw{n}"), workeroutput={"exitstatus": n})
                     for n in (0, 1))
    for node in (first, second):
        result.pytest_configure_node(node)
    result.pytest_testnodedown(first, None)
    if second_finished:
        result.pytest_testnodedown(second, None)
    result._close_start_coverage(SimpleNamespace(exitstatus=1))
    if second_finished:
        assert json.loads(result.journal.getvalue()) == {"event": "starts_complete", "session": SESSION}
    else:
        assert result.journal.getvalue() == b""


def test_real_pytest_produces_one_named_arithmetic_failure_and_no_nested_keys(tmp_path, monkeypatch):
    from ouroboros.test_evidence import EVIDENCE_POINTER

    suite, nested, scratch, retained = (tmp_path / name for name in ("suite", "nested", "scratch", "retained"))
    for directory in (suite, nested, scratch, retained):
        directory.mkdir()
    repo = Path(__file__).resolve().parents[1]
    conftest = f"import sys\nsys.path.insert(0, {str(repo)!r})\npytest_plugins = ['tests.conftest']\n"
    for directory in (suite, nested):
        (directory / "conftest.py").write_text(conftest, encoding="utf-8")
    (nested / "test_nested.py").write_text("def test_addition():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    nested_evidence = retained / "nested"
    (suite / "test_arithmetic.py").write_text(
        "import os, subprocess, sys\n"
        "def test_arithmetic():\n"
        "    assert not any(name.startswith('OUROBOROS_PRECOMMIT_') for name in os.environ)\n"
        "    usual = subprocess.run([sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q',\n"
        f"        '--ci-evidence-dir', {str(nested_evidence)!r}, 'test_nested.py'],\n"
        f"        cwd={str(nested)!r}, check=False, capture_output=True, text=True, timeout=30)\n"
        "    assert usual.returncode == 0, usual.stdout + usual.stderr\n"
        "    assert 2 + 2 == 5, 'arithmetic specimen'\n", encoding="utf-8")
    (scratch / gate._PRECOMMIT_READING_MARKER).touch()
    (scratch / EVIDENCE_POINTER).write_text(str(retained), encoding="utf-8")
    probe = write_worker_probe(scratch)
    (gate.probe_dir(scratch) / "admitted-workers").write_text("1", encoding="ascii")
    spec = gate._make_pass("single", ["-o", "addopts=", "-q", "--tb=line", "-p", probe, "test_arithmetic.py"])

    result = gate._execute_pytest_pass(sys.executable, suite, scratch, spec.args, 60)
    assert result[0] == 1 and result[2] == "", result[:3]
    assert "AssertionError: arithmetic specimen" in result[1], result[1]
    assert recorded_collection(scratch) == 1
    records = recorded_session(scratch, root=retained, seen=set(), expected_directory=result.session_directory)
    nodeid = "test_arithmetic.py::test_arithmetic"
    key = case_key(nodeid, result.case_salt)
    assert records["precommit_assignment"] == [[key, nodeid]]
    reports = [(r["nodeid"], r["case_key"], r["phase"], r["outcome"]) for r in records["reports"]]
    assert reports == [(nodeid, key, "setup", "passed"), (nodeid, key, "call", "failed"),
                       (nodeid, key, "teardown", "passed")]
    rows = [json.loads(line) for line in (result.session_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["event"], r["case_key"]) for r in rows if r["event"] in ("start", "finish")] == [
        ("start", key), ("finish", key)]
    assert rows[-1] == {"event": "starts_complete", "session": records["session"]}
    assert len(list(retained.glob("pytest-*/result.json"))) == 1

    cause, legacy = gate._pass_block(spec, result, 1, 60, 60, "", records, 1, probe, scratch, sys.executable, 8000)
    monkeypatch.setattr(receipts, "_precommit_reader", reading)
    evidence = {"path": retained}
    shown = gate._with_reading(evidence, legacy, result, "single", cause, None, 8000, recorded=records)
    facts = evidence["reading"]
    assert cause == facts["cause"] == "pytest_exit" and facts["coverage"] == "final" and not facts["gap"]
    assert facts["attempts"] == facts["exit"] == 1 and facts["not_run"] == facts["in_flight"] == []
    assert facts["failed"] == [{"nodeid": nodeid, "attempt": "call, AssertionError"}]
    assert shown.startswith(legacy.splitlines()[0] + "\n") and "FAILED " + nodeid in shown
    assert "one attempt, no retry" in shown
    usual = json.loads((nested_evidence / "results.json").read_text(encoding="utf-8"))
    assert usual["session_exit_code"] == 0 and usual["tests_collected"] == 1
    assert "precommit_assignment" not in usual and all("case_key" not in row for row in usual["reports"])
    nested_journal = (nested_evidence / "events.jsonl").read_text(encoding="utf-8")
    assert "case_key" not in nested_journal and "starts_complete" not in nested_journal
    print("PRECOMMIT_ARITHMETIC: one failed call with start/finish; collection 1; nested ordinary pass unkeyed")
