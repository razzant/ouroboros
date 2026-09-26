"""Offline evidence tests: the audit cannot turn missing traces into a clean claim."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.benchmarks.cowork_bench.audit_cowork_bench import audit_run, call_findings, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_fixture(tmp_path: Path, *, status="passed", events=None, tools=None) -> Path:
    task_dump = tmp_path / "bench" / "dumps" / "model" / "SingleUserTurn-задача"
    (task_dump / "ouroboros").mkdir(parents=True)
    (task_dump / "ouroboros_summary.json").write_text(
        json.dumps({"task": "задача", "bench_status": "success"}), encoding="utf-8"
    )
    write_jsonl(tmp_path / "result_index.jsonl", [{
        "instance_id": "задача", "status": status,
        "official_eval_status": "completed" if status in {"passed", "failed"} else "not_run",
        "output_paths": {"task_dump": str(task_dump.relative_to(tmp_path))},
    }])
    if events is not None:
        write_jsonl(task_dump / "ouroboros" / "events.jsonl", events)
    if tools is not None:
        write_jsonl(task_dump / "ouroboros" / "tools.jsonl", tools)
    return task_dump


def usage(**overrides) -> dict:
    return {"type": "llm_usage", "prompt_tokens": 100, "completion_tokens": 10,
            "cached_tokens": 80, "cost": 0.002, "cost_known": True,
            "model": "moonshotai/kimi-k3", "provider": "openrouter", **overrides}


def test_official_verdict_survives_flags_without_copying_gold(tmp_path):
    gold = "PRIVATE_GOLD_ЖЁЛТЫЙ"
    dump = run_fixture(tmp_path, events=[usage()], tools=[{
        "type": "tool_call", "tool": "run_command", "args": {
            "command": json.dumps({"argv": ["bash", "-lc", "cat /workspace/tasks/x/evaluation/answer.txt"],
                                   "note": gold}),
        }, "result_preview": gold,
    }, {"type": "tool_call", "tool": "mcp_pptx__create", "args": {}, "is_error": False}])
    # The auditor must not inspect the evaluator payload, even though it exists.
    (dump / "eval_res.json").write_text(json.dumps({"pass": True, "gold": gold}), encoding="utf-8")
    before = (dump / "eval_res.json").read_bytes()
    report = audit_run(tmp_path)
    row = report["tasks"][0]
    assert row["official_pass"] is True
    assert row["classification"] == "passed"
    assert report["manual_review_tasks"] == ["задача"]
    assert row["manual_review"] == [{"source": "tools.jsonl", "line": 1,
                                     "reason": "answer_source_or_evaluator_reference"}]
    assert gold not in json.dumps(report, ensure_ascii=False)
    assert (dump / "eval_res.json").read_bytes() == before
    assert row["activity"]["mcp_calls"] == 1
    assert row["cost"]["total_usd"] == 0.002
    assert row["billing_providers"] == ["openrouter"]
    assert row["observed_response_providers"] == []
    assert row["response_provider_coverage"] == "unavailable"


def test_unknown_price_and_observed_endpoint_remain_separate(tmp_path):
    run_fixture(tmp_path, events=[usage(), usage(cost=None, cost_known=False,
                usage={"response_provider": "Endpoint A"}), usage(cost=0, cost_estimated=True)], tools=[])
    report = audit_run(tmp_path)
    row = report["tasks"][0]
    assert report["cost"] == {"known_usd": 0.002, "total_usd": None, "complete": False}
    assert row["cost"]["unknown_usage_records"] == 1
    assert row["cost"]["estimated_usage_records"] == 1
    assert row["observed_response_providers"] == ["Endpoint A"]
    assert row["response_provider_coverage"] == "observed_subset"
    assert row["model_activity_observed"] is True
    assert row["mcp_activity_observed"] is False


def test_corrupt_or_missing_logs_cannot_be_a_clean_zero(tmp_path):
    dump = run_fixture(tmp_path, status="infra_failed", events=[usage()], tools=None)
    with (dump / "ouroboros" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type":\n')
    row = audit_run(tmp_path)["tasks"][0]
    assert row["classification"] == "infrastructure"
    assert row["official_pass"] is None
    assert row["cost"]["known_usd"] == 0.002
    assert row["cost"]["total_usd"] is None
    assert {g["source"] for g in row["gaps"]} == {"tools.jsonl", "events.jsonl"}
    assert row["capability_omissions"]["absence_proves_none"] is False


@pytest.mark.parametrize("tool,args,expected", [
    ("run_command", {"argv": ["/usr/bin/psql", "-c", "select 1"]}, ["possible_direct_postgres_access"]),
    ("run_script", {"code": "import psycopg2; psycopg2.connect()"}, ["possible_direct_postgres_access"]),
    ("mcp_terminal__python_execute", {"code": "import asyncpg"}, ["possible_direct_postgres_access"]),
    ("mcp_clickhouse__query", {"sql": "SELECT 'psql'"}, []),
    ("run_command", {"command": "curl https://raw.githubusercontent.com/0717376/cowork_bench/main/a"},
     ["answer_source_or_evaluator_reference"]),
    ("run_command", json.dumps({"command": "cat '/workspace/groundtruth_workspace/test.txt'"}),
     ["answer_source_or_evaluator_reference"]),
    ("run_command", {"command": "echo ordinary"}, []),
])
def test_requested_argv_json_and_sql_are_diagnostic_only(tool, args, expected):
    assert call_findings({"tool": tool, "args": args,
                          "result_preview": "groundtruth_workspace/answer.txt psql"}) == expected


def test_timeouts_and_not_attempted_denominator_are_retained(tmp_path):
    run_fixture(tmp_path, status="agent_failed", events=[usage()], tools=[])
    with (tmp_path / "result_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"instance_id": "not-started", "status": "not_attempted",
                                 "reason_code": "missing_result"}) + "\n")
    report = audit_run(tmp_path)
    assert report["task_count"] == 2
    assert report["classifications"] == {"genuine_failure": 1, "not_attempted": 1}
    assert report["tasks"][0]["model_activity_observed"] is True
    assert report["tasks"][1]["cost"]["total_usd"] is None


def test_omissions_are_referenced_without_republishing_their_contents(tmp_path):
    secret = "do-not-copy-gold-or-arbitrary-text"
    run_fixture(tmp_path, events=[usage(), {"type": "capability_report", "capability_omissions": [secret]}], tools=[])
    row = audit_run(tmp_path)["tasks"][0]
    assert row["capability_omissions"]["reported_count"] == 1
    assert row["capability_omissions"]["references"] == [{"source": "events.jsonl", "line": 2, "count": 1}]
    assert secret not in json.dumps(row)


def test_cli_is_offline_and_never_overwrites_existing_report(tmp_path):
    run_fixture(tmp_path, events=[], tools=[])
    target = tmp_path / "audit.json"
    assert main(["--run-dir", str(tmp_path), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["tasks"][0]["cost"]["total_usd"] is None
    with pytest.raises(FileExistsError):
        main(["--run-dir", str(tmp_path), "--output", str(target)])


def _official(dump: Path, *, returned: dict | None = {"pass": None}, terminal: bool = True) -> str:
    """A settled (or, with ``terminal=False``, a claimed-only) official attempt in ``dump``."""
    from devtools.benchmarks.cowork_bench import eval_attempt as attempts

    dump.mkdir(parents=True, exist_ok=True)
    if terminal:
        def evaluate(_log):
            (dump / "eval_res.json").write_text(json.dumps(returned), encoding="utf-8")
            return returned
        record, _ = attempts.official_attempt(dump, {"b": 1}, {}, lambda: "log", evaluate)
        return record["attempt_id"]
    attempt_id = "a" * 32
    attempts.publish_exclusive(dump / attempts.CLAIM_NAME, {"kind": "official", "attempt_id": attempt_id})
    return attempt_id


def _diagnostic(dump: Path, attempt_id: str, receipt: dict | None) -> None:
    from devtools.benchmarks.cowork_bench import eval_attempt as attempts

    attempts.publish_exclusive(dump / attempts.DIAGNOSTIC_CLAIM_NAME, {"kind": "diagnostic", "attempt_id": attempt_id})
    if receipt is not None:
        attempts.publish_exclusive(dump / f"{attempts.DIAGNOSTIC_RECEIPT_PREFIX}{attempt_id}.json",
                                   {"kind": "diagnostic", "attempt_id": attempt_id, **receipt})


def eval_run(tmp_path: Path, *, enabled=True, cleanup=True) -> dict[str, Path]:
    tasks = {name: tmp_path / "bench" / "dumps" / "m" / f"SingleUserTurn-{name}" for name in (
        "checked", "claimed", "refused", "unrecorded", "cut", "ineligible", "passed")}
    statuses = {"checked": "declined", "cut": "declined", "ineligible": "declined", "passed": "completed"}
    write_jsonl(tmp_path / "result_index.jsonl", [{
        "instance_id": name, "status": "passed" if name == "passed" else "agent_failed",
        "official_eval_status": statuses.get(name, "unreported"),
        "output_paths": {"task_dump": str(dump)}} for name, dump in tasks.items()])
    (tmp_path / "run_manifest.json").write_text(json.dumps({"harness": {"applied_config": (
        {"diagnostic_eval_on_truncation": True} if enabled else {"model": "m"})}}), encoding="utf-8")
    if cleanup:
        (tmp_path / "monitor.json").write_text(
            json.dumps({"finished": True, "stop_reason": "campaign_budget_reserve"}), encoding="utf-8")
        (tmp_path / "resource_stop").write_text("campaign_budget_reserve\n", encoding="utf-8")
    admission = tmp_path / "eval_admission"
    admission.mkdir()
    (admission / "7-aaaa.task").write_text("refused\n", encoding="utf-8")
    (admission / "7-aaaa.eval_refused").write_text("admission_stopped\n", encoding="utf-8")
    (admission / "7-bbbb.task").write_text("unrecorded\n", encoding="utf-8")
    _diagnostic(tasks["checked"], _official(tasks["checked"]), {
        "outcome": "checks_failed", "reason": "", "eligible": True, "checker_exit_code": 1})
    _official(tasks["claimed"], terminal=False)
    _diagnostic(tasks["cut"], _official(tasks["cut"]), None)
    _diagnostic(tasks["ineligible"], _official(tasks["ineligible"]), {
        "outcome": "ineligible", "reason": "no_agent_activity", "eligible": False})
    _official(tasks["passed"], returned={"pass": True})
    for dump in tasks.values():
        dump.mkdir(parents=True, exist_ok=True)
    return tasks


def test_eval_attempt_block_projects_stop_evidence_without_touching_verdicts(tmp_path):
    eval_run(tmp_path)
    report = audit_run(tmp_path)
    rows = {row["instance_id"]: row for row in report["tasks"]}
    official = {name: row["eval_attempt"]["official"]["projection"] for name, row in rows.items()}
    assert official == {"checked": "evaluated", "claimed": "unknown", "refused": "unavailable",
                        "unrecorded": "unknown", "cut": "evaluated", "ineligible": "evaluated",
                        "passed": "evaluated"}
    assert rows["refused"]["eval_attempt"]["official"]["reason"] == "campaign_stopped_before_eval"
    diagnostic = {name: row["eval_attempt"]["residual_diagnostic"]["outcome"] for name, row in rows.items()}
    assert diagnostic == {"checked": "checks_failed", "claimed": "unknown", "refused": "unavailable",
                          "unrecorded": "not_applicable", "cut": "unknown", "ineligible": "ineligible",
                          "passed": "not_applicable"}
    block = report["eval_attempts"]["residual_diagnostic"]
    assert (block["enabled"], block["task_count"], block["official_declined"]) == (True, 7, 3)
    assert (block["eligible"], block["checked"]) == (1, 1)
    assert block["ineligible_reasons"] == {"no_agent_activity": 1}
    assert block["semantics"] == "residual_state_after_agent_stop_not_exact_cutoff"
    # The diagnostic never changes execution classification or the literal official verdict.
    assert report["classifications"] == {"genuine_failure": 6, "passed": 1}
    assert rows["checked"]["official_pass"] is None and rows["passed"]["official_pass"] is True


def test_stop_projection_needs_proven_cleanup_and_flag_off_is_disabled(tmp_path):
    eval_run(tmp_path, enabled=False, cleanup=False)
    report = audit_run(tmp_path)
    rows = {row["instance_id"]: row["eval_attempt"] for row in report["tasks"]}
    assert rows["claimed"]["official"]["projection"] == "unknown"
    assert rows["cut"]["residual_diagnostic"]["outcome"] == "unknown"
    assert rows["refused"]["official"]["projection"] == "unavailable"
    assert rows["refused"]["official"]["reason"] == "eval_creation_refused:admission_stopped"
    assert rows["unrecorded"]["residual_diagnostic"]["outcome"] == "disabled"
    assert report["eval_attempts"]["residual_diagnostic"]["enabled"] is False
    assert report["eval_attempts"]["cleanup_proven"] is False


@pytest.mark.parametrize(("stop_reason", "refusal", "expected"), [
    ("disk_reserve", "admission_stopped", "eval_creation_refused:admission_stopped"),
    ("campaign_budget_reserve", "disk_probe_failed", "eval_creation_refused:disk_probe_failed"),
    ("runner_finished", "admission_stopped", "eval_creation_refused:admission_stopped"),
    ("budget_meter_unavailable", "admission_stopped", "campaign_stopped_before_eval"),
])
def test_only_linked_campaign_refusal_has_campaign_cause(tmp_path, stop_reason, refusal, expected):
    eval_run(tmp_path)
    (tmp_path / "monitor.json").write_text(
        json.dumps({"finished": True, "stop_reason": stop_reason}), encoding="utf-8")
    (tmp_path / "resource_stop").write_text(stop_reason + "\n", encoding="utf-8")
    (tmp_path / "eval_admission/7-aaaa.eval_refused").write_text(refusal + "\n", encoding="utf-8")
    row = next(row for row in audit_run(tmp_path)["tasks"] if row["instance_id"] == "refused")
    assert row["eval_attempt"]["official"]["reason"] == expected


def test_first_owner_stop_wins_over_later_budget_monitor_projection(tmp_path):
    eval_run(tmp_path)
    (tmp_path / "resource_stop").write_text("signal_15\n", encoding="utf-8")
    row = next(row for row in audit_run(tmp_path)["tasks"] if row["instance_id"] == "refused")
    assert row["eval_attempt"]["official"]["reason"] == "eval_creation_refused:admission_stopped"


def test_agent_references_to_attempt_records_need_manual_review():
    assert call_findings({"tool": "run_command", "args": {"command": "cat ../ouroboros_eval_claim.json"}}) == [
        "answer_source_or_evaluator_reference"]
