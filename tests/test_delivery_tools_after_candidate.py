"""A retained answer constrains final text without disabling further tool work."""

import json
import queue


def _finish(answer):
    return {"content": None, "tool_calls": [{"id": "complete", "type": "function", "function": {
        "name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": answer})}}]}


def _run_loop(tmp_path, monkeypatch, responses, *, nudges=(True,)):
    """Main through the real loop. A finalization nudge meets the first plain answer: the host
    retains that answer and offers the completion door (``_arm_delivery_control``)."""
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    answers, pending_nudges, calls = iter(responses), iter(nudges), []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(row) for row in request_messages])
        answer = next(answers)
        return ({"role": "assistant", **answer} if isinstance(answer, dict)
                else {"role": "assistant", "content": answer}), 0.0

    def nudge(_tools, _drive_root, _task_id, _trace, _content, messages, _emit_progress):
        if not next(pending_nudges, False):
            return False
        messages.append({"role": "user", "content": "[SYSTEM REMINDER] Check the requested effect before finishing."})
        return True

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", nudge)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "parent1"}
    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}], tools=registry, llm=FakeLLM(),
        drive_logs=tmp_path, emit_progress=lambda *_args, **_kwargs: None,
        incoming_messages=queue.Queue(), task_id="parent1", drive_root=tmp_path,
    )
    return result, usage, trace, calls


def test_further_write_then_complete_replacement_remain_available(tmp_path, monkeypatch):
    replacement = "The requested effect.txt now contains the verified result."
    result, _usage, trace, calls = _run_loop(
        tmp_path, monkeypatch,
        ["The first answer is retained.",
         {"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
             "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "verified result"}),
         }}]},
         _finish(replacement)],
    )
    prompt = json.dumps(calls[1])
    assert "Continue useful work" in prompt and "available completion tool" in prompt
    assert "action=finish" in prompt and "action=stop" in prompt
    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "verified result"
    assert result == replacement
    assert trace["delivery_candidate"]["revision"] == 2
    assert trace["delivery_candidate"]["finalization_control"] == "candidate"
    assert trace["delivery_candidate"]["degraded"] is False
