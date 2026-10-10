"""Pause abandons a sent fallback just as it abandons the initial author call."""
from __future__ import annotations

import threading

import pytest

from ouroboros import budget_pause, loop, model_wait, owner_pause
from ouroboros import usage_accounting as ua
from ouroboros.task_results import write_task_result
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_pause_model_receiver import _extract, _request
from tests.test_working_checkpoint import _registry

pytestmark = pytest.mark.serial


def test_pause_returns_from_actual_loop_fallback_before_sender_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", 0.02)
    monkeypatch.setattr("ouroboros.loop_model_call._route_candidates", lambda *_: [
        ("fallback-model", "fallback", False, False)])
    monkeypatch.setattr("ouroboros.fallback_cooldown.is_cooling_down", lambda *_a, **_k: False)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_k: pytest.fail("no provider"))
    write_task_result(tmp_path, "t-wait", "running", root_task_id="t-wait")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    sent, release, returned = threading.Event(), threading.Event(), threading.Event()
    calls, outcome = [], {}

    def sender():
        sent.set()
        assert release.wait(10)
        return {"role": "assistant", "content": "late fallback", "tool_calls": []}

    def dispatch(call, _disposition, **_kwargs):
        calls.append(call.active_model)
        if len(calls) == 1:
            call.accumulated_usage["_last_llm_error_kind"] = "auth_error"
            return None, 0.0
        return ua.execute_physical_attempt(_request(), sender, extractor=_extract), 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    owner = model_wait.TaskModelWait(task={"id": "t-wait", "_attempt": 1}, drive_root=tmp_path,
                                     event_queue=None, worker_slot_held=True)

    def run():
        try:
            with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait")), \
                    model_wait.operation_wait_scope(owner):
                outcome["result"] = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
        except BaseException as error:  # inspect the actual cold-pause exit
            outcome["error"] = error
        finally:
            returned.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert sent.wait(5), repr(outcome)
        owner_pause.install_fence(tmp_path, "t-wait", request_id="pause-fallback")
        assert returned.wait(3), "the author must pause before the fallback sender answers"
        assert isinstance(outcome.get("error"), budget_pause.BudgetPauseRequested), repr(outcome)
        assert calls == ["test-model", "fallback-model"]
        assert "result" not in outcome
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
