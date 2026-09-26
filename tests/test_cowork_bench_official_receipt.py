"""The official evaluator receipt survives every Cowork outcome without changing its bytes or the score."""

from __future__ import annotations

from functools import partial
import hashlib
import json
import pathlib

import pytest

from devtools.benchmarks.common.result_index import task_result_row
from devtools.benchmarks.cowork_bench import run_cowork_bench as launcher
from devtools.benchmarks.cowork_bench.audit_cowork_bench import audit_run
from devtools.benchmarks.cowork_bench.official_receipt import read_official_receipt

legacy_ledger_row = partial(launcher.ledger_row, protocol="legacy")
GATE = "Task status: {}, only SUCCESS counts as pass; pass is null"
SECRET = "PRIVATE_EVALUATOR_OUTPUT_ЖЁЛТЫЙ"

# Status each execution branch had BEFORE receipts existed; a receipt never changes it.
BRANCHES = {
    "pg_fail": ({}, {"status": "pg_fail"}, "infra_failed", "pg_fail"),
    "no_summary": ({}, {"status": "failed"}, "infra_failed", "missing_adapter_summary"),
    "not_attempted": ({}, {}, "not_attempted", "missing_result"),
    "adapter_infra": ({"bench_status": "failed", "infra_failed": True, "reason_code": "llm_api_error"},
                      {"status": "failed"}, "infra_failed", "llm_api_error"),
    "deadline_local": ({"bench_status": "max_turns_reached", "reason_code": "deadline_local"},
                       {"status": "failed"}, "agent_failed", "deadline_local"),
    "wall_clock_timeout": ({"bench_status": "failed", "reason_code": "wall_clock_timeout"},
                           {"status": "failed"}, "agent_failed", "wall_clock_timeout"),
}


def write_raw(path: pathlib.Path, data: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def dump_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def gate_fixture(dump: pathlib.Path, log_status: str, details_status: str | None = None, **extra) -> bytes:
    write_raw(dump / "traj_log.json", dump_json({"config": {}, "status": log_status}))
    return write_raw(dump / "eval_res.json",
                     dump_json({"pass": None, "details": GATE.format(details_status or log_status), **extra}))


@pytest.mark.parametrize("branch", sorted(BRANCHES))
@pytest.mark.parametrize("payload,expected_official,expected_pass", [
    ({"pass": True, "details": "All evaluation checks passed"}, "completed", True),
    ({"pass": False, "failure": SECRET}, "completed", False),
    ({"pass": None, "details": "no linked gate"}, "unknown", None),
    ("gate", "declined", None),
    (None, "unreported", None),
])
def test_every_branch_keeps_the_exact_receipt_and_its_status(tmp_path, branch, payload, expected_official,
                                                             expected_pass):
    summary, runner, status, reason = BRANCHES[branch]
    if summary:
        write_raw(tmp_path / "ouroboros_summary.json", dump_json(summary))
    if payload == "gate":
        raw = gate_fixture(tmp_path, summary.get("bench_status") or "failed")
    elif payload is not None:
        raw = write_raw(tmp_path / "eval_res.json", dump_json(payload))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    row = legacy_ledger_row("task", tmp_path, runner)

    assert (row["status"], row["reason_code"]) == (status, reason)
    assert row["official_eval_status"] == expected_official != "not_run"
    receipt = row["details"]["official_receipt"]
    assert receipt["official_eval_status"] == expected_official
    assert receipt["pass"] is expected_pass
    assert receipt["path"] == str(tmp_path / "eval_res.json")
    if payload is None:
        assert (receipt["bytes"], receipt["sha256"], receipt["cause"]) == (None, None, "file_absent")
    else:
        assert receipt["bytes"] == len(raw)
        assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "eval" not in row["details"]
    assert SECRET not in json.dumps(row, ensure_ascii=False)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("payload,status,reason,official", [
    ({"pass": True}, "passed", "passed", "completed"),
    ({"pass": False, "failure": "verifier output"}, "failed", "verifier_failed", "completed"),
    ({"pass": None}, "infra_failed", "missing_eval_result", "unknown"),
    ({}, "infra_failed", "missing_eval_result", "invalid"),
    ({"pass": "false"}, "infra_failed", "missing_eval_result", "invalid"),
    ({"pass": 1}, "infra_failed", "missing_eval_result", "invalid"),
    ({"pass": 0.0}, "infra_failed", "missing_eval_result", "invalid"),
    (None, "infra_failed", "missing_eval_result", "unreported"),
])
def test_successful_agent_phase_scores_only_a_literal_boolean(tmp_path, payload, status, reason, official):
    write_raw(tmp_path / "ouroboros_summary.json", dump_json({"bench_status": "success"}))
    if payload is not None:
        write_raw(tmp_path / "eval_res.json", dump_json(payload))
    row = legacy_ledger_row("task", tmp_path, {"status": "success"})
    assert (row["status"], row["reason_code"], row["official_eval_status"]) == (status, reason, official)
    # The scored branch keeps its previous payload detail; every other branch carries only the receipt.
    assert ("eval" in row["details"]) is (status in {"passed", "failed"})
    if status in {"passed", "failed"}:
        assert row["details"]["eval"] == payload


def test_receipt_classifies_every_malformed_file_without_coercion(tmp_path):
    cases = {
        "missing": None,
        "directory": "dir",
        "utf8": b'{"pass": true, "x": "\xff"}',
        "truncated": b'{"pass": tr',
        "not_object": b"[true]",
        "deep": b"[" * 200_000,
        "empty_object": b"{}",
        "string": b'{"pass": "true"}',
        "number": b'{"pass": 1}',
        "array": b'{"pass": [true]}',
        "nan_detail": b'{"pass":true,"details":NaN}',
        "infinite_detail": b'{"pass":true,"details":Infinity}',
        "duplicate_pass": b'{"pass":false,"pass":true}',
    }
    expected = {
        "missing": ("unreported", "file_absent"),
        "directory": ("unreadable", "os_error:"),
        "utf8": ("unreadable", "utf8_error:byte_21"),
        "truncated": ("unreadable", "json_error:line_1_col_10"),
        "not_object": ("unreadable", "not_object:list"),
        "deep": ("unreadable", "json_error:RecursionError"),
        "empty_object": ("invalid", "pass_missing"),
        "string": ("invalid", "pass_not_boolean:str"),
        "number": ("invalid", "pass_not_boolean:int"),
        "array": ("invalid", "pass_not_boolean:list"),
        "nan_detail": ("unreadable", "json_error:ValueError"),
        "infinite_detail": ("unreadable", "json_error:ValueError"),
        "duplicate_pass": ("unreadable", "json_error:ValueError"),
    }
    for name, data in cases.items():
        dump = tmp_path / name
        dump.mkdir()
        if data == "dir":
            (dump / "eval_res.json").mkdir()
        elif data is not None:
            write_raw(dump / "eval_res.json", data)
        receipt, payload = read_official_receipt(dump)
        expected_status, expected_cause = expected[name]
        assert receipt["official_eval_status"] == expected_status, name
        if name == "directory":
            # Windows can report PermissionError where POSIX reports IsADirectoryError.
            assert receipt["cause"].startswith(expected_cause), name
        else:
            assert receipt["cause"] == expected_cause, name
        assert receipt["pass"] is None and payload is None, name
        assert len(receipt["cause"]) <= 200
        if isinstance(data, bytes):
            assert receipt["bytes"] == len(data)
            assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
            assert (dump / "eval_res.json").read_bytes() == data


@pytest.mark.parametrize("log_status,details_status,extra,log_missing,expected", [
    ("max_turns_reached", None, {}, False, "declined"),
    ("failed", None, {}, False, "declined"),
    ("success", None, {}, False, "unknown"),           # the gate never declines a successful log
    ("failed", "max_turns_reached", {}, False, "unknown"),  # receipt names another status
    ("failed", None, {"failure": "x"}, False, "unknown"),   # not the gate's exact payload
    ("failed", None, {}, True, "unknown"),              # no linked log to prove the gate
])
def test_null_verdict_is_declined_only_with_a_linked_status_gate(tmp_path, log_status, details_status, extra,
                                                                  log_missing, expected):
    gate_fixture(tmp_path, log_status, details_status, **extra)
    log_bytes = (tmp_path / "traj_log.json").read_bytes()
    if log_missing:
        (tmp_path / "traj_log.json").unlink()
    receipt, _payload = read_official_receipt(tmp_path)
    assert receipt["official_eval_status"] == expected
    assert receipt["pass"] is None
    if expected == "declined":
        assert receipt["gate"] == {
            "log_path": str(tmp_path / "traj_log.json"), "log_bytes": len(log_bytes),
            "log_sha256": hashlib.sha256(log_bytes).hexdigest(), "log_status": log_status,
            "rule": "upstream_status_gate",
        }
    else:
        assert "gate" not in receipt


def test_valid_fixture_score_and_settlement_parity(tmp_path):
    """Statuses, counts and resume settlement on valid receipts equal the pre-receipt ledger."""
    bench = tmp_path / "bench"
    dumps = bench / "dumps" / launcher.dump_dir_name("m")
    fixtures = {
        "pass": ({"bench_status": "success"}, {"pass": True}, "passed"),
        "fail": ({"bench_status": "success"}, {"pass": False, "failure": "x"}, "failed"),
        "timeout": ({"bench_status": "max_turns_reached", "reason_code": "deadline_local"}, "gate", "agent_failed"),
        "infra": ({"bench_status": "failed", "infra_failed": True, "reason_code": "llm_api_error"}, None,
                  "infra_failed"),
        "unstarted": (None, None, "not_attempted"),
    }
    runner_rows = ["task,status,eval_pass,duration_s"]
    for task, (summary, payload, _status) in fixtures.items():
        dump = dumps / f"SingleUserTurn-{task}"
        dump.mkdir(parents=True)
        if summary is None:
            continue
        runner_rows.append(f"{task},{'success' if summary['bench_status'] == 'success' else 'failed'},null,1")
        write_raw(dump / "ouroboros_summary.json", dump_json(summary))
        if payload == "gate":
            gate_fixture(dump, summary["bench_status"])
        elif payload is not None:
            write_raw(dump / "eval_res.json", dump_json(payload))
    write_raw(bench / "benchmark_logs" / "fully_parallel_1" / "summary.csv",
              ("\n".join(runner_rows) + "\n").encode("utf-8"))
    # A pre-protocol manifest positively identifies these old unclaimed receipts.
    write_raw(tmp_path / "run_manifest.json", dump_json({"harness": {"applied_config": {"model": "m"}}}))
    run = tmp_path / "run"
    counts = launcher.write_ledger(run / "result_index.jsonl", bench, "m", list(fixtures))
    assert counts == {"passed": 1, "failed": 1, "agent_failed": 1, "infra_failed": 1, "not_attempted": 1}
    assert launcher.settled_tasks([run]) == {"pass", "fail", "timeout"}
    rows = {row["instance_id"]: row for row in map(json.loads, (run / "result_index.jsonl").read_text().splitlines())}
    assert {task: row["status"] for task, row in rows.items()} == {task: f[2] for task, f in fixtures.items()}
    assert {task: row["official_eval_status"] for task, row in rows.items()} == {
        "pass": "completed", "fail": "completed", "timeout": "declined",
        "infra": "unreported", "unstarted": "unreported",
    }


@pytest.mark.parametrize("provenance", ["current_on", "current_off", "missing", "unreadable", "legacy"])
def test_refusal_before_claim_never_scores_old_pass_in_current_run(tmp_path, provenance):
    """Exercise write_ledger, not the entrypoint: admission may refuse before it starts."""
    bench = tmp_path / "bench"
    dump = bench / "dumps" / launcher.dump_dir_name("m") / "SingleUserTurn-task"
    old = write_raw(dump / "eval_res.json", dump_json({"pass": True}))
    write_raw(dump / "ouroboros_summary.json", dump_json({"bench_status": "success"}))
    write_raw(bench / "benchmark_logs" / "fully_parallel_1" / "summary.csv",
              b"task,status,eval_pass,duration_s\ntask,success,null,1\n")
    if provenance == "unreadable":
        write_raw(tmp_path / "run_manifest.json", b"{broken")
    elif provenance != "missing":
        config = {"model": "m"}
        if provenance != "legacy":
            config["diagnostic_eval_on_truncation"] = provenance == "current_on"
        write_raw(tmp_path / "run_manifest.json", dump_json({"harness": {"applied_config": config}}))
    ledger = tmp_path / "result_index.jsonl"
    counts = launcher.write_ledger(ledger, bench, "m", ["task"], cause="runner_exited")
    row = json.loads(ledger.read_text(encoding="utf-8"))
    if provenance == "legacy":
        assert counts == {"passed": 1}
        assert (row["status"], row["official_eval_status"]) == ("passed", "completed")
    else:
        assert counts == {"infra_failed": 1}
        assert (row["reason_code"], row["official_eval_status"]) == ("missing_eval_result", "unknown")
        assert launcher.settled_tasks([tmp_path]) == set()
    assert row["details"]["official_receipt"]["pass"] is True  # evidence, not modern score
    assert (dump / "eval_res.json").read_bytes() == old
    assert not (dump / "ouroboros_eval_claim.json").exists()


def test_audit_shows_the_literal_verdict_without_promoting_an_unscored_row(tmp_path):
    dump = tmp_path / "dump"
    write_raw(dump / "ouroboros_summary.json", dump_json({"bench_status": "failed", "reason_code": "wall_clock_timeout"}))
    raw = write_raw(dump / "eval_res.json", dump_json({"pass": True, "details": SECRET}))
    row = legacy_ledger_row("late", dump, {"status": "failed"})
    historical = {"instance_id": "old", "status": "agent_failed", "official_eval_status": "not_run",
                  "output_paths": {"task_dump": str(tmp_path / "old")}}
    legacy = {"instance_id": "bare", "status": "infra_failed", "output_paths": {"task_dump": str(tmp_path / "bare")}}
    (tmp_path / "result_index.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in (row, historical, legacy)), encoding="utf-8")

    report = audit_run(tmp_path)
    late, old, bare = report["tasks"]
    assert row["status"] == late["ledger_status"] == "agent_failed"
    assert late["classification"] == "genuine_failure"
    assert late["official_eval_status"] == "completed"
    assert late["official_pass"] is True          # evaluator fact, not promotion of execution/score
    assert late["official_receipt"] == {"available": True, "pass": True, "bytes": len(raw),
                                        "sha256": hashlib.sha256(raw).hexdigest()}
    assert report["classifications"] == {"genuine_failure": 2, "infrastructure": 1}
    # Historical rows are read as written: an explicit `not_run` stays; a missing field is unreported.
    assert (old["official_eval_status"], old["official_receipt"]["available"]) == ("not_run", False)
    assert (bare["official_eval_status"], bare["official_receipt"]["pass"]) == ("unreported", None)
    assert SECRET not in json.dumps(report, ensure_ascii=False)
    assert (dump / "eval_res.json").read_bytes() == raw


@pytest.mark.parametrize("given,expected", [
    ({}, "unreported"),
    ({"official_eval_status": ""}, "unreported"),
    ({"official_eval_status": None}, "unreported"),
    ({"official_eval_status": "not_run"}, "not_run"),
    ({"metadata": {"official_eval_status": "not_run"}}, "not_run"),
    ({"official_eval_status": "pending"}, "pending"),
])
def test_shared_default_is_unreported_and_explicit_not_run_survives(given, expected):
    row = task_result_row(benchmark="unit", instance_id="x", status="failed", **given)
    assert row["official_eval_status"] == expected


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_runtime_disclosure_reads_only_exact_linked_task(tmp_path, branch):
    summary, runner, _status, _reason = BRANCHES[branch]
    # An unrelated result must never fill an unavailable link, including missing summary.
    write_raw(tmp_path / "ouroboros" / "unrelated.json", dump_json({
        "task_id": "unrelated", "status": "failed", "reason_code": "provider_unavailable"}))
    if not summary:
        row = legacy_ledger_row("task", tmp_path, runner)
        assert row["runtime_outcome"] == {"available": False}
        assert row["details"]["runtime_result_source"]["cause"] == "task_id_unavailable"
        return
    summary = {**summary, "ouroboros_task_id": "exact-task"}
    write_raw(tmp_path / "ouroboros_summary.json", dump_json(summary))
    target = tmp_path / "ouroboros" / "exact-task.json"
    write_raw(target, dump_json({"task_id": "different", "reason_code": "provider_unavailable"}))
    row = legacy_ledger_row("task", tmp_path, runner)
    assert row["runtime_outcome"] == {"available": False}
    assert row["details"]["runtime_result_source"]["cause"] == "task_id_mismatch"
    raw = write_raw(target, dump_json({"task_id": "exact-task", "status": "completed",
                                      "reason_code": "deadline_local"}))
    row = legacy_ledger_row("task", tmp_path, runner)
    assert row["runtime_outcome"]["reason_code"] == "deadline_local"
    assert row["runtime_outcome"]["truncated"] is True
    assert row["details"]["runtime_result_source"]["sha256"] == hashlib.sha256(raw).hexdigest()


def test_linked_runtime_failure_stays_gap_and_success_branch_has_disclosure(tmp_path):
    write_raw(tmp_path / "ouroboros_summary.json", dump_json({
        "bench_status": "success", "ouroboros_task_id": "exact"}))
    write_raw(tmp_path / "eval_res.json", b'{"pass": false}')
    target = tmp_path / "ouroboros" / "exact.json"
    for raw in (None, b"\xff", b"{}", b'{"task_id":"foreign"}'):
        if raw is not None:
            write_raw(target, raw)
        row = legacy_ledger_row("task", tmp_path, {})
        assert row["status"] == "failed"
        assert row["runtime_outcome"] == {"available": False}
    write_raw(target, b'{"task_id":"exact","status":"completed","reason_code":"final_message"}')
    row = legacy_ledger_row("task", tmp_path, {})
    assert row["runtime_outcome"]["reason_code"] == "final_message"
    assert row["status"] == "failed"


def test_dotted_runtime_id_links_exact_result_and_cannot_traverse(tmp_path):
    write_raw(tmp_path / "ouroboros_summary.json", dump_json({
        "bench_status": "success", "ouroboros_task_id": "exact.task"}))
    write_raw(tmp_path / "eval_res.json", b'{"pass":false}')
    write_raw(tmp_path / "ouroboros" / "exact.task.json", dump_json({
        "task_id": "exact.task", "status": "completed", "reason_code": "deadline_local"}))
    row = legacy_ledger_row("task", tmp_path, {})
    assert row["runtime_outcome"]["reason_code"] == "deadline_local"
    assert row["details"]["runtime_result_source"]["state"] == "linked"
    write_raw(tmp_path / "ouroboros_summary.json", dump_json({
        "bench_status": "success", "ouroboros_task_id": "../foreign"}))
    row = legacy_ledger_row("task", tmp_path, {})
    assert row["runtime_outcome"] == {"available": False}
    assert row["details"]["runtime_result_source"]["cause"] == "task_id_unavailable"
