"""Task-local decisions and check evidence survive post-task fanout/recovery."""

from tests import _dialogue as dialogue_test

from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline
from ouroboros.outcome_receipt_store import append_verification_receipt
from ouroboros.post_task_synthesis import capture_task_inputs, build_trace_summary


def _receipt(ts, criterion, status, code):
    return {
        "ts": ts, "tool": "verify_and_record", "contract_kind": "run",
        "criterion_id": criterion, "check": "python -m pytest tests/target.py",
        "status": status, "returncode": code, "matched": status == "pass",
        "summary": "78 passed" if status == "pass" else "check failed",
    }


def test_capture_keeps_owner_answers_peer_provenance_and_successes(tmp_path):
    question_answer = "Question: publish as which account?\nOwner rejected all options: use razzant."
    ctx = SimpleNamespace(_owner_directives=[
        {"source": "owner_quiz_answer", "msg_id": "owner-1", "content": question_answer},
        {"source": "task_message", "msg_id": "peer-1", "content": "Peer proposes stopping",
         "source_task_id": "peer-task", "relayed_from_task_id": "peer-task"},
    ])
    receipts = [
        _receipt("1", "target", "fail", 1),
        _receipt("2", "other", "fail", 2),
        _receipt("3", "target", "pass", 0),
    ]
    frozen = capture_task_inputs(ctx, {"id": "task"}, tmp_path, receipts)
    ctx._owner_directives[0]["content"] = "changed after completion"
    receipts[-1]["returncode"] = 4

    assert frozen["owner_requirements_and_decisions"][0]["content"] == question_answer
    assert frozen["owner_requirements_and_decisions"][1]["relayed_from_task_id"] == "peer-task"
    assert frozen["verification_receipts"][-1]["returncode"] == 0
    assert frozen["verification_summary"]["latest_status"] == "pass"
    assert frozen["verification_summary"]["unreconciled_red_count"] == 1
    assert frozen["verification_summary"]["unreconciled_red_identity"]["criterion_id"] == "other"


def test_real_completion_persists_receipt_union_before_cleanup_and_recovery(tmp_path, monkeypatch):
    local, canonical = tmp_path / "local", tmp_path / "canonical"
    local.mkdir()
    canonical.mkdir()
    task = {"id": "task", "root_task_id": "task", "type": "task", "chat_id": 1,
            "text": "Check and finish", "budget_drive_root": str(canonical)}
    ctx = SimpleNamespace(drive_root=local, budget_drive_root=str(canonical),
                          _owner_directives=[{"source": "owner_quiz_answer", "content": "Proceed as requested."}])
    assert append_verification_receipt(local, "task", _receipt("1", "target", "fail", 1))
    assert append_verification_receipt(canonical, "task", _receipt("2", "target", "pass", 0))
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=local, repo_dir=tmp_path), None, None, [], task,
        "Done", {"rounds": 2}, {"tool_calls": [], "reasoning_notes": []}, 0,
        local / "logs", ctx=ctx,
    )
    saved = pipeline.load_task_result(local, "task")["review_evidence"]
    assert [r["returncode"] for r in saved["task_inputs"]["verification_receipts"]] == [1, 0]
    assert saved["task_inputs"]["verification_summary"]["unreconciled_red"] is False
    ctx._owner_directives.clear()
    # Recovery reads the durable record, not the vanished ctx, and leaves its package intact.
    pipeline._set_root_post_task_checkpoint(SimpleNamespace(drive_root=local), task, "pending_once")
    seen = []
    def dispatch(env, task, usage, trace, logs, **kw):
        seen.append((task["id"], kw["sealed_final"]))
        pipeline._set_root_post_task_checkpoint(env, task, "completed")
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                        dispatch)
    assert pipeline.recover_pending_root_post_task_synthesis(local, tmp_path) == 1
    assert [task_id for task_id, _sealed in seen] == ["task"] and seen[0][1]
    assert pipeline.load_task_result(local, "task")["review_evidence"] == saved
    assert pipeline.recover_pending_root_post_task_synthesis(local, tmp_path) == 0
    assert len(seen) == 1, "a stale child obligation never repeats its canonical completed synthesis"


def test_frozen_task_inputs_stay_whole_and_the_facts_row_buys_no_model_call(tmp_path, monkeypatch):
    import json

    def no_model(*_args, **_kwargs):
        raise AssertionError("the free facts row buys no model call")

    monkeypatch.setattr("ouroboros.llm.LLMClient", no_model)
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", no_model)
    owner_text = "Question and options\n" + "owner context " * 800 + "\nExact answer: use the approved account."
    frozen = capture_task_inputs(
        SimpleNamespace(_owner_directives=[{"source": "owner_quiz_answer", "content": owner_text}]),
        {"id": "task"}, tmp_path, [_receipt("1", "target", "pass", 0)],
    )
    packed = json.dumps(frozen, ensure_ascii=False)
    assert frozen["owner_requirements_and_decisions"][0]["content"] == owner_text
    assert packed.count("Question and options") == 1
    assert frozen["verification_receipts"][0]["returncode"] == 0
    assert frozen["verification_receipts"][0]["summary"] == "78 passed"
    task = {"id": "task", "root_task_id": "task", "type": "task", "text": "Verify work",
            "drive_root": str(tmp_path)}
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok", "exit_code": 0}],
             "reasoning_notes": []}
    (tmp_path / "logs").mkdir()
    pipeline._record_task_facts(None, task, {"rounds": 2}, trace, tmp_path / "logs")
    [row] = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert row["summary_kind"] == "host_task_facts"


def test_explicit_zero_is_not_inferred(tmp_path):
    trace = build_trace_summary({"tool_calls": [
        {"tool": "run_command", "status": "ok", "exit_code": 0},
        {"tool": "run_command", "status": "ok"},
    ]})
    assert trace.count("exit_code=0") == 1
    assert trace.count("status=ok") == 2


def test_failed_owner_capture_is_disclosed_without_losing_check_receipt(tmp_path, monkeypatch):
    from ouroboros import review_evidence_sections

    def fail(*args):
        raise OSError("source missing")

    monkeypatch.setattr(review_evidence_sections, "_accept_owner_directives", fail)
    frozen = capture_task_inputs(None, {"id": "task"}, tmp_path, [_receipt("1", "target", "pass", 0)])
    assert frozen["unavailable_sections"] == ["owner_requirements_and_decisions"]
    assert frozen["verification_receipts"][0]["returncode"] == 0
