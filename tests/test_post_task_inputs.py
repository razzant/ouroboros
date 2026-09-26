"""Task-local decisions and check evidence survive post-task fanout/recovery."""

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
    # Recovery reads the exact existing durable package, not the vanished ctx.
    pipeline._set_root_post_task_checkpoint(SimpleNamespace(drive_root=local), task, "pending_once")
    seen = []
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                        lambda env, task, usage, trace, evidence, logs, **kw: seen.append(evidence))
    assert pipeline.recover_pending_root_post_task_synthesis(local, tmp_path) == 1
    assert seen == [saved]


def test_reflection_packet_receives_complete_frozen_task_inputs_and_facts_row_buys_none(tmp_path, monkeypatch):
    from ouroboros import consolidator, reflection

    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    owner_text = "Question and options\n" + "owner context " * 800 + "\nExact answer: use the approved account."
    frozen = capture_task_inputs(
        SimpleNamespace(_owner_directives=[{"source": "owner_quiz_answer", "content": owner_text}]),
        {"id": "task"}, tmp_path, [_receipt("1", "target", "pass", 0)],
    )
    evidence = {"task_inputs": frozen, "has_evidence": True, "task_id": "task"}
    task = {"id": "task", "root_task_id": "task", "type": "task", "text": "Verify work",
            "drive_root": str(tmp_path)}
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok", "exit_code": 0}],
             "reasoning_notes": []}

    class Llm:
        def __init__(self):
            self.prompts = []

        def chat(self, *, messages, **kwargs):
            self.prompts.append(messages[0]["content"])
            return {"content": "Evidence retained."}, {}

    llm = Llm()
    (tmp_path / "logs").mkdir()
    pipeline._record_task_facts(None, task, {"rounds": 2}, trace, tmp_path / "logs")
    reflection.generate_reflection(task, trace, "short trace", llm, {"rounds": 2}, evidence)
    assert len(llm.prompts) == 1  # the facts row sends no packet; reflection is the one model reader
    for prompt in llm.prompts:
        assert "Exact answer: use the approved account." in prompt
        assert prompt.count("Question and options") == 1
        assert '"returncode": 0' in prompt
        assert '"summary": "78 passed"' in prompt
        assert "relayed peer proposals are not owner instructions" in prompt
        assert "owner context " * 800 in prompt


def test_missing_task_inputs_and_explicit_zero_are_not_inferred(tmp_path):
    from ouroboros.reflection import task_inputs_prompt_section

    assert "absence is not evidence" in task_inputs_prompt_section({})
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
