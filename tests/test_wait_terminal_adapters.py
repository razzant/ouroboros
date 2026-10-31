"""Existing physical waits use one terminal predicate and keep their own custody."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import model_sleep
from ouroboros.task_results import write_task_result
from ouroboros.tools import control_task_results as tasks
from tests.test_delegate_wake_truthful_facts import _ctx


@pytest.mark.parametrize("ready", [False, True])
def test_supervised_wait_reads_shared_terminal_custody_without_fabricating_a_success(tmp_path, monkeypatch, ready):
    from ouroboros import delegate_supervision as supervision

    ctx, calls = _ctx(tmp_path), []
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda *_a, **_kw: {
        "run": SimpleNamespace(settled=ready, terminal_state="failed" if ready else "running", review_owned=False)})

    def observe(*_a, **_kw):
        calls.append(True)
        return json.dumps({"status": "no_progress" if len(calls) == 1 else "needs_input", "run_id": "run"})
    result = json.loads(supervision.supervised_wait(ctx, "run", wait_once=observe).text)
    assert len(calls) == (1 if ready else 2)
    assert result["status"] == ("terminal_custody" if ready else "needs_input")
    if ready:
        assert result["terminal_facts"][0]["id"] == "run:run"
        assert result["terminal_facts"][0]["state"] == "failed"
    assert not getattr(ctx, "_sleep_interaction_seen", None)


@pytest.mark.parametrize("status,ready", [("completed", True), ("failed", True), ("running", False), ("cancel_pending", False)])
def test_task_wait_adapters_use_same_status_snapshot_as_pure_readiness(tmp_path, status, ready):
    ctx = _ctx(tmp_path)
    probe = tasks._terminal_probe(ctx, ["selected", "live"])
    rows = {"selected": {"status": status}, "live": {"status": "running"}}
    assert probe(rows) == {"selected": ready, "live": False}
    assert rows["selected"]["status"] == status


@pytest.mark.parametrize("many", [False, True])
def test_public_task_wait_calls_the_shared_predicate_and_keeps_timeout(tmp_path, monkeypatch, many):
    ctx, calls = _ctx(tmp_path), []
    ctx.model = "model"
    write_task_result(tmp_path, "child", "completed", parent_task_id=ctx.task_id, result="done")
    original = model_sleep.terminal_readiness

    def observed(*args, **kwargs):
        calls.append(kwargs.get("task_results"))
        return original(*args, **kwargs)
    monkeypatch.setattr(model_sleep, "terminal_readiness", observed)
    if many:
        result = json.loads(tasks._wait_for_tasks(ctx, ["child"], timeout_sec=0, mode="any_terminal"))
        assert result["all_terminal"] and result["timeout_sec"] == 0
    else:
        assert "Task wait completed" in tasks._wait_for_task(ctx, "child", timeout_sec=0)
    assert calls and calls[0]["child"]["status"] == "completed"


@pytest.mark.parametrize("cancel", [False, True])
def test_supervised_control_preempts_terminal_peeks(tmp_path, monkeypatch, cancel):
    from ouroboros import delegate_supervision as supervision
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_mailbox import write_owner_message

    ctx = _ctx(tmp_path)
    if cancel:
        write_task_result(tmp_path, ctx.task_id, "running")
        request_cancel(tmp_path, ctx.task_id, reason="owner stop", source="test")
    else:
        write_owner_message(tmp_path, "finish", ctx.task_id, kind="finalize_now")
    monkeypatch.setattr(model_sleep, "terminal_readiness", lambda *_a, **_kw: pytest.fail("control waited behind a peek"))
    def observe(*_a, **_kw):
        assert not cancel, "cancellation must preempt transport"
        return json.dumps({"status": "no_progress", "run_id": "run"})
    result = json.loads(supervision.supervised_wait(ctx, "run", wait_once=observe).text)
    assert any(event.get("type") == "cancellation_intent" if cancel else event.get("kind") == "finalize_now"
               for event in result["wake_events"])
