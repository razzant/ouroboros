"""Focused truth/transport/runtime tests for managed skill publication."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.skill_publish_result import (
    SKILL_PUBLISH_TARGET_METADATA_KEY,
    apply_skill_publish_receipt_veto,
    extract_skill_publish_result_metadata,
    serialize_skill_publish_result,
    validate_skill_publish_receipt,
)
from ouroboros.skill_publish_scanner import SecretFinding
from ouroboros.tool_capabilities import FOREGROUND_MUTATIVE_TOOLS, tool_result_limit

SNAPSHOT_HASH = "a" * 64
RULESET_SHA256 = "b" * 64
REPOSITORY = "OuroborosHub/OuroborosHub"


def _receipt(*, skill: str = "demo", repository: str = REPOSITORY, number: int = 17):
    return {
        "kind": "github_pull_request",
        "repository": repository,
        "url": f"https://github.com/{repository}/pull/{number}",
        "number": number,
        "skill": skill,
        "snapshot_hash": SNAPSHOT_HASH,
        "ruleset_sha256": RULESET_SHA256,
    }


def _success_result(*, skill: str = "demo", repository: str = REPOSITORY) -> str:
    receipt = _receipt(skill=skill, repository=repository)
    return serialize_skill_publish_result(
        ok=True,
        status="pr_opened",
        reason_code="",
        skill=skill,
        snapshot_hash=SNAPSHOT_HASH,
        scanner={
            "engine": "betterleaks",
            "version": "1.8.1",
            "ruleset_sha256": RULESET_SHA256,
        },
        completed_stage="pr_opened",
        completed_effects=[
            {"stage": "branch_created", "kind": "branch", "branch": "submit/demo-v1"},
            {"stage": "commit_created", "kind": "commit", "commit_sha": "c" * 40},
            {"stage": "pr_opened", "kind": "pull_request"},
        ],
        receipt=receipt,
        expected_repository=repository,
    )


def _failed_result(*, skill: str = "demo", status: str = "scanner_blocked") -> str:
    return serialize_skill_publish_result(
        ok=False,
        status=status,
        reason_code="scanner_high_confidence",
        skill=skill,
        snapshot_hash=SNAPSHOT_HASH,
        scanner={
            "engine": "betterleaks",
            "version": "1.8.1",
            "ruleset_sha256": RULESET_SHA256,
        },
        completed_stage="local_preflight",
        completed_effects=[],
        blocker_count=1,
        repair_hint="Remove the finding and run a fresh skill review.",
    )


def _task(*, task_type: str = "skill_publish", skill: str = "demo", repository: str = REPOSITORY):
    return {
        "type": task_type,
        "metadata": {
            SKILL_PUBLISH_TARGET_METADATA_KEY: {
                "skill": skill,
                "repository": repository,
            },
        },
    }


def _loop_outcome(objective_status: str):
    return {
        "outcome_axes": {
            "execution": {"status": "ok", "reason_code": "final_message"},
            "objective": {
                "status": objective_status,
                "source": "task_acceptance_review" if objective_status != "not_evaluated" else "none",
            },
        },
    }


def _trace_call(result: str, *, is_error: bool = False):
    metadata = extract_skill_publish_result_metadata(result)
    return {
        "tool": "submit_skill_to_hub",
        "is_error": is_error,
        "status": "tool_reported_failure" if is_error else "ok",
        **metadata,
    }


def test_bounded_result_is_deterministic_parseable_and_exact_about_omissions():
    candidate = "candidate-value-must-never-be-visible"
    findings = [
        {
            "path": f"payload/{index:04d}-" + "x" * 500,
            "line": index + 1,
            "detector": "generic-password-" + "d" * 200,
            "confidence": "medium",
            "reason": "safe reason " + "r" * 500,
            "verification": "not_attempted",
            "disposition": "warning",
            "column": 2,
            "rule_id": "obsolete-rule-id",
            "classification": "obsolete-warning",
            "match": candidate,
            "raw": {"secret": candidate},
        }
        for index in range(600)
    ]
    kwargs = {
        "ok": False,
        "status": "scanner_blocked",
        "reason_code": "scanner_high_confidence",
        "skill": "demo",
        "snapshot_hash": SNAPSHOT_HASH,
        "scanner": {
            "engine": "betterleaks",
            "version": "1.8.1",
            "ruleset_sha256": RULESET_SHA256,
            "raw_config": candidate,
        },
        "completed_stage": "commit_created",
        "completed_effects": [
            {
                "stage": "branch_created",
                "kind": "branch",
                "branch": "b" * 600,
                "unredacted": candidate,
            },
            {"stage": "commit_created", "kind": "commit", "commit_sha": "c" * 64},
        ],
        "findings": findings,
        "blocker_count": 3,
        "warning_count": 597,
        "audited_false_positive_count": 9,
        "repair_hint": "repair " + "h" * 2000,
    }

    encoded = serialize_skill_publish_result(**kwargs)
    reversed_encoded = serialize_skill_publish_result(**{**kwargs, "findings": list(reversed(findings))})
    assert encoded == reversed_encoded
    assert len(encoded) < tool_result_limit("submit_skill_to_hub")
    parsed = json.loads(encoded)
    assert parsed["omitted_count"] == len(findings) - len(parsed["findings"])
    assert parsed["omitted_count"] > 0
    assert candidate not in encoded
    assert all(
        set(row)
        == {
            "path",
            "line",
            "detector",
            "confidence",
            "reason",
            "verification",
            "disposition",
        }
        for row in parsed["findings"]
    )
    assert [row["stage"] for row in parsed["completed_effects"]] == [
        "branch_created",
        "commit_created",
    ]
    assert len(parsed["repair_hint"]) == 600


def test_bounded_result_keeps_blockers_ahead_of_nonblocking_findings():
    base = {
        "line": 1,
        "detector": "provider-key",
        "confidence": "high",
        "reason": "Scanner finding requires review before publication.",
        "verification": "not_attempted",
    }
    findings = [
        {
            **base,
            "path": f"fixtures/{index:03d}/" + "x" * 250,
            "disposition": "audited_false_positive",
        }
        for index in range(160)
    ]
    findings.append(
        {
            **base,
            "path": "payload/real-blocker.txt",
            "disposition": "blocker",
        }
    )
    parsed = json.loads(
        serialize_skill_publish_result(
            ok=False,
            status="scanner_blocked",
            reason_code="scanner_high_confidence",
            skill="demo",
            findings=findings,
            blocker_count=1,
            audited_false_positive_count=160,
        )
    )
    assert parsed["omitted_count"] > 0
    assert parsed["findings"][0]["disposition"] == "blocker"
    assert any(row["path"] == "payload/real-blocker.txt" for row in parsed["findings"])


def test_result_finding_vocabulary_matches_scanner_and_preserves_audited_high():
    finding = SecretFinding(
        path="fixtures/provider.txt",
        line=7,
        detector="provider-key",
        confidence="high",
        reason="Inline allowance was surfaced by the audit pass.",
        verification="not_attempted",
        disposition="audited_false_positive",
    )

    encoded = serialize_skill_publish_result(
        ok=False,
        status="scanner_findings",
        reason_code="scanner_findings",
        skill="demo",
        findings=[dataclasses.asdict(finding)],
        blocker_count=0,
        audited_false_positive_count=1,
    )

    row = json.loads(encoded)["findings"][0]
    assert set(row) == set(dataclasses.asdict(finding))
    assert row["confidence"] == "high"
    assert row["disposition"] == "audited_false_positive"
    assert "column" not in row
    assert "rule_id" not in row
    assert "classification" not in row


def test_success_has_one_nested_authority_and_no_top_level_pr_aliases():
    parsed = json.loads(_success_result())
    assert parsed["ok"] is True
    assert parsed["status"] == "pr_opened"
    assert parsed["receipt"] == _receipt()
    assert not ({"url", "number", "pr_url", "pr_number", "pr_opened"} & set(parsed))


@pytest.mark.parametrize(
    ("changes", "expected_repository"),
    [
        ({"url": "http://github.com/OuroborosHub/OuroborosHub/pull/17"}, REPOSITORY),
        ({"url": "https://example.com/OuroborosHub/OuroborosHub/pull/17"}, REPOSITORY),
        ({"url": "https://github.com/other/repo/pull/17"}, REPOSITORY),
        ({"url": "https://github.com/OuroborosHub/OuroborosHub/issues/17"}, REPOSITORY),
        ({"url": "https://github.com/OuroborosHub/OuroborosHub/pull/nope"}, REPOSITORY),
        ({"url": "https://github.com/OuroborosHub/OuroborosHub/pull/17/"}, REPOSITORY),
        ({"url": "https://github.com/OuroborosHub/OuroborosHub/pull/17?x=1"}, REPOSITORY),
        ({"url": "arbitrary nonempty text"}, REPOSITORY),
        ({"number": 18}, REPOSITORY),
        ({"number": 0, "url": "https://github.com/OuroborosHub/OuroborosHub/pull/0"}, REPOSITORY),
        ({"snapshot_hash": SNAPSHOT_HASH + "c"}, REPOSITORY),
        ({}, "different/repository"),
    ],
)
def test_receipt_rejects_wrong_host_repo_path_number_or_empty_text(changes, expected_repository):
    receipt = {**_receipt(), **changes}
    assert (
        validate_skill_publish_receipt(
            receipt,
            expected_repository=expected_repository,
            expected_skill="demo",
        )
        is None
    )


def test_receipt_validation_is_case_insensitive_only_for_repository():
    receipt = _receipt(repository="OuroborosHub/OuroborosHub")
    assert (
        validate_skill_publish_receipt(
            receipt,
            expected_repository="ouroboroshub/ouroboroshub",
            expected_skill="demo",
        )
        is not None
    )
    assert (
        validate_skill_publish_receipt(
            receipt,
            expected_repository=REPOSITORY,
            expected_skill="DEMO",
        )
        is None
    )


def test_canonical_unicode_skill_identifier_is_preserved_and_matches_exactly():
    skill = "навык"
    receipt = _receipt(skill=skill)
    encoded = serialize_skill_publish_result(
        ok=True,
        status="pr_opened",
        reason_code="",
        skill=skill,
        snapshot_hash=SNAPSHOT_HASH,
        scanner={
            "engine": "betterleaks",
            "version": "1.8.1",
            "ruleset_sha256": RULESET_SHA256,
        },
        completed_stage="pr_opened",
        receipt=receipt,
        expected_repository=REPOSITORY,
    )
    assert json.loads(encoded)["skill"] == skill
    assert extract_skill_publish_result_metadata(encoded)["skill_publish_receipt"]["skill"] == skill


def test_overlong_skill_identity_is_rejected_instead_of_prefix_truncated():
    with pytest.raises(ValueError, match="invalid skill"):
        serialize_skill_publish_result(
            ok=False,
            status="blocked",
            reason_code="invalid_skill",
            skill="x" * 65,
        )


def test_metadata_is_extracted_from_full_result_before_visible_head_truncation():
    from ouroboros.loop_tool_execution import _extract_result_metadata, _truncate_tool_result

    payload = json.loads(_success_result())
    receipt = payload.pop("receipt")
    payload["ignored_padding"] = "x" * (tool_result_limit("submit_skill_to_hub") + 5000)
    payload["receipt"] = receipt
    full = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    visible = _truncate_tool_result(full, tool_name="submit_skill_to_hub")
    assert receipt["url"] not in visible
    metadata = _extract_result_metadata("submit_skill_to_hub", full, False)
    assert metadata["skill_publish_receipt"]["url"] == receipt["url"]
    assert metadata["skill_publish_attempt"]["completed_stage"] == "pr_opened"


def test_typed_failed_publish_is_delivered_to_the_next_llm_turn():
    from ouroboros.loop_tool_execution import (
        _extract_result_metadata,
        _is_tool_execution_failure,
        process_tool_results,
    )

    result = _failed_result()
    is_error = _is_tool_execution_failure(True, result)
    assert is_error is True
    result_meta = _extract_result_metadata("submit_skill_to_hub", result, is_error)
    messages: list = []
    trace = {"tool_calls": []}
    errors = process_tool_results(
        [
            {
                "tool_call_id": "publish-1",
                "fn_name": "submit_skill_to_hub",
                "result": result,
                "is_error": is_error,
                "tool_args": {"skill": "demo"},
                "args_for_log": {"skill": "demo"},
                "is_code_tool": False,
                "result_meta": result_meta,
            }
        ],
        messages,
        trace,
        emit_progress=lambda _message: None,
    )

    assert errors == 1
    assert messages == [{"role": "tool", "tool_call_id": "publish-1", "content": result}]
    assert trace["tool_calls"][0]["status"] == "tool_reported_failure"
    assert trace["tool_calls"][0]["skill_publish_attempt"]["status"] == "scanner_blocked"


@pytest.mark.parametrize("tool", ["skill_preflight", "ext_1_demo_diagnostic"])
def test_unrelated_json_ok_false_keeps_current_nonblocking_execution_semantics(tool):
    from ouroboros.loop_tool_execution import _extract_result_metadata
    from ouroboros.outcomes import derive_loop_outcome

    result = '{"ok":false,"error":"diagnostic finding"}'
    metadata = _extract_result_metadata(tool, result, True)
    assert metadata == {"status": "tool_reported_failure"}
    outcome = derive_loop_outcome(
        "Diagnostic reported.",
        {},
        {
            "tool_calls": [
                {
                    "tool": tool,
                    "is_error": True,
                    "result": result,
                    **metadata,
                }
            ]
        },
    )
    assert outcome["outcome_axes"]["execution"]["status"] == "ok"
    assert outcome["outcome_axes"]["objective"]["status"] == "not_evaluated"


@pytest.mark.parametrize("objective_status", ["pass", "best_effort"])
def test_absent_partial_or_wrong_target_receipt_demotes_only_success(objective_status):
    from ouroboros.outcomes import normalize_outcome_axes

    partial = _trace_call(_failed_result(), is_error=True)
    wrong_skill = _trace_call(_success_result(skill="other"))
    wrong_repository = _trace_call(_success_result(repository="other/repository"))

    for trace in (
        {"tool_calls": []},
        {"tool_calls": [partial]},
        {"tool_calls": [wrong_skill]},
        {"tool_calls": [wrong_repository]},
    ):
        outcome = _loop_outcome(objective_status)
        returned = apply_skill_publish_receipt_veto(outcome, _task(), trace)
        assert returned is outcome
        objective = outcome["outcome_axes"]["objective"]
        assert objective["status"] == "fail"
        assert objective["source"] == "task_acceptance_review"
        assert objective["outcome_tier"] == "blocked_with_evidence"
        assert objective["receipt_veto"]["reason"] == objective["reason"]
        normalized = normalize_outcome_axes({"outcome_axes": outcome["outcome_axes"]})
        assert normalized["objective"]["status"] == "fail"
        assert normalized["objective"]["receipt_veto"] == objective["receipt_veto"]


def test_later_valid_same_skill_receipt_wins_without_erasing_earlier_attempt():
    earlier = _trace_call(_failed_result(), is_error=True)
    later = _trace_call(_success_result())
    trace = {"tool_calls": [earlier, later]}
    outcome = _loop_outcome("pass")

    apply_skill_publish_receipt_veto(outcome, _task(), trace)

    assert outcome["outcome_axes"]["objective"]["status"] == "pass"
    assert trace["tool_calls"][0]["is_error"] is True
    assert trace["tool_calls"][0]["skill_publish_attempt"]["status"] == "scanner_blocked"


def test_receipt_without_its_validated_attempt_metadata_cannot_satisfy_veto():
    outcome = _loop_outcome("pass")
    apply_skill_publish_receipt_veto(
        outcome,
        _task(),
        {
            "tool_calls": [
                {
                    "tool": "submit_skill_to_hub",
                    "is_error": False,
                    "skill_publish_receipt": _receipt(),
                }
            ]
        },
    )
    assert outcome["outcome_axes"]["objective"]["status"] == "fail"
    assert outcome["outcome_axes"]["objective"]["reason"] == "skill_publish_receipt_mismatch"


@pytest.mark.parametrize("objective_status", ["fail", "not_evaluated", "degraded"])
def test_valid_receipt_never_promotes_existing_objective(objective_status):
    outcome = _loop_outcome(objective_status)
    apply_skill_publish_receipt_veto(
        outcome,
        _task(),
        {"tool_calls": [_trace_call(_success_result())]},
    )
    assert outcome["outcome_axes"]["objective"]["status"] == objective_status


@pytest.mark.parametrize("task_type", ["task", "Skill_publish", "skill_publish ", ""])
def test_veto_activates_only_for_literal_skill_publish_type(task_type):
    outcome = _loop_outcome("pass")
    apply_skill_publish_receipt_veto(outcome, _task(task_type=task_type), {"tool_calls": []})
    assert outcome["outcome_axes"]["objective"]["status"] == "pass"


def test_foreground_publish_timeout_waits_until_fake_mutator_terminalizes(tmp_path):
    from ouroboros.loop_tool_execution import _execute_with_timeout

    assert FOREGROUND_MUTATIVE_TOOLS == frozenset({"submit_skill_to_hub"})
    lifecycle: list[str] = []
    live_events: list = []

    def execute(_name, _args):
        lifecycle.append("running")
        time.sleep(0.05)
        lifecycle.append("terminal")
        return _failed_result(status="pr_open_indeterminate")

    tools = SimpleNamespace(
        CODE_TOOLS=set(),
        _ctx=SimpleNamespace(
            event_queue=SimpleNamespace(put_nowait=lambda event: live_events.append(event)),
            task_metadata={},
        ),
        execute=execute,
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    started = time.perf_counter()
    result = _execute_with_timeout(
        tools,
        {"id": "publish-call", "function": {"name": "submit_skill_to_hub", "arguments": "{}"}},
        logs,
        timeout_sec=0.001,
        task_id="task-publish",
    )
    elapsed = time.perf_counter() - started

    assert lifecycle == ["running", "terminal"]
    assert elapsed >= 0.04
    assert result["result"] == _failed_result(status="pr_open_indeterminate")
    payloads = [event.get("data") or {} for event in live_events]
    assert any(payload.get("type") == "tool_call_late" for payload in payloads)
    assert any(payload.get("terminal_wait") is True for payload in payloads)
    frozen = list(lifecycle)
    time.sleep(0.02)
    assert lifecycle == frozen


def test_api_preserves_literal_type_without_workspace_and_queue_gives_priority_zero(
    tmp_path,
    monkeypatch,
):
    import ouroboros.gateway.tasks as gateway_tasks
    import supervisor.queue as queue

    captured = {}

    def complete(task, **_kwargs):
        captured.update(task)
        return JSONResponse({"ok": True, "task_id": task["id"], "status": "scheduled"})

    monkeypatch.setattr(
        "supervisor.queue.reserve_task_admission",
        lambda *_args, **_kwargs: {"status": "reserved", "reason": ""},
    )
    monkeypatch.setattr(gateway_tasks, "prepare_task_drive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("ouroboros.artifacts.stage_task_attachments", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(gateway_tasks, "_complete_api_task_admission", complete)

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    app = Starlette(routes=[Route("/api/tasks", endpoint=gateway_tasks.api_tasks_create, methods=["POST"])])
    app.state.repo_dir = repo
    app.state.drive_root = data
    payload = {
        "description": "Publish the selected skill.",
        "type": "skill_publish",
        "metadata": {
            SKILL_PUBLISH_TARGET_METADATA_KEY: {
                "skill": "demo",
                "repository": REPOSITORY,
            },
        },
    }

    response = TestClient(app).post("/api/tasks", json=payload)

    assert response.status_code == 200, response.text
    assert "workspace_root" not in payload
    assert captured["type"] == "skill_publish"
    assert captured["workspace_root"] == ""
    assert captured["task_contract"]["task_type"] == "skill_publish"
    assert (
        captured["metadata"][SKILL_PUBLISH_TARGET_METADATA_KEY]
        == payload["metadata"][SKILL_PUBLISH_TARGET_METADATA_KEY]
    )

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    admitted = queue.enqueue_task(
        {
            "id": "queued-publish",
            "type": captured["type"],
            "metadata": captured["metadata"],
        }
    )
    assert admitted["type"] == "skill_publish"
    assert admitted["priority"] == 0
    assert admitted["task_contract"]["task_type"] == "skill_publish"


def test_pipeline_uses_leaf_veto_and_stays_below_hard_module_ceiling(monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline

    monkeypatch.setattr(pipeline, "_attach_host_mutation_projection", lambda *_args: None)
    monkeypatch.setattr(
        pipeline,
        "derive_loop_outcome",
        lambda *_args: _loop_outcome("pass"),
    )
    outcome = pipeline._derive_host_bound_loop_outcome(
        SimpleNamespace(),
        _task(),
        "done",
        {},
        {"tool_calls": []},
    )
    assert outcome["outcome_axes"]["objective"]["status"] == "fail"
    assert len(pathlib.Path(pipeline.__file__).read_text(encoding="utf-8").splitlines()) < 1600
