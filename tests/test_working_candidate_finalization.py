"""Saved complete answers rejoin real acceptance, including paid critic custody."""
from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros import working_checkpoint as wc
from ouroboros.tools.registry import ToolRegistry
from tests.test_acceptance_async_loop import ANSWER, call, keep
from tests.test_acceptance_async_loop import full_loop as _full_loop

full_loop = _full_loop


def _restart(f, monkeypatch, *, new_id=False):
    old = f.ctx
    handoff = wc.prepare_recovery(old.drive_root, old.task_id, from_attempt=1, cause="worker_crash")
    assert handoff["boundary"] == "candidate"
    new_tools = ToolRegistry(repo_dir=old.repo_dir, drive_root=old.drive_root)
    new = new_tools._ctx
    for key in ("task_id", "task_metadata", "task_contract", "budget_drive_root", "current_chat_id"):
        setattr(new, key, copy.deepcopy(getattr(old, key)))
    if new_id:
        from tests.test_loop_acceptance_gate import _seed_acceptance_root
        _seed_acceptance_root(new.drive_root, old.task_id + "-retry", new)
        new.task_contract["expected_output"] = old.task_contract["expected_output"]
    new.task_attempt = 2
    new.context_fit_plan = None
    new.working_recovery = handoff
    new.owner_message_admission_agent = SimpleNamespace(
        _owner_message_generation=0, _accepting_owner_messages=True, _busy=True, _current_task_id=new.task_id)
    new.owner_message_admission_lock = threading.RLock()
    new.owner_wait_callback = f.park
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *_a, **_kw: (None, "max"))
    f.ctx, f.tools = new, new_tools
    f.run_args.update(tools=new_tools, messages=[], task_id=new.task_id)


@pytest.mark.parametrize("settled", [False, True])
def test_saved_candidate_collects_the_original_pending_panel(full_loop, monkeypatch, settled):
    f = full_loop
    real_gate = loop._enforce_swarm_actions
    class InterruptedFinalization(BaseException):
        pass
    def interrupted(*args, **kwargs):
        if f.model_step >= 2:
            raise InterruptedFinalization()
        return real_gate(*args, **kwargs)
    monkeypatch.setattr(loop, "_enforce_swarm_actions", interrupted)
    def first_main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.model_step == 2 and f.entered.wait(5)
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", first_main)
    with pytest.raises(InterruptedFinalization):
        f.run()
    saved = json.loads(wc.checkpoint_path(f.ctx.drive_root, f.ctx.task_id, 1).read_bytes())
    original_run = saved["trace"]["review_runs"][-1]
    assert original_run["actors"][0]["operation_state"] in {"pending_dispatch", "in_flight"}
    assert saved["delivery_candidate"]["full_text"] == ANSWER
    _restart(f, monkeypatch)
    monkeypatch.setattr(loop, "_enforce_swarm_actions", real_gate)
    if settled:
        f.release.set()
        assert f.settled.wait(10)
    calls = []
    def continued(_llm, messages, *_a, **_kw):
        assert not settled, "settled unchanged answer needs no additional Main request"
        assert f.waits, "real saved finalization must collect/wait before author control"
        f.model_inputs.append(copy.deepcopy(messages))
        calls.append(messages)
        assert f.ctx._delivery_candidate.full_text == ANSWER
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", continued)
    answer, _, trace = f.run()
    assert answer == ANSWER
    assert len(calls) <= 1
    assert len(f.review_sends) == 1, "recovery collects the paid operation without a second panel"
    run = trace["review_runs"][-1]
    assert run["request"] == original_run["request"]
    assert run["actors"][0]["operation_id"] == original_run["actors"][0]["operation_id"]
    assert trace["acceptance_decision"]["status"] == "accepted"


def test_saved_candidate_does_not_turn_unavailable_evidence_into_acceptance(full_loop, monkeypatch):
    f = full_loop
    class InterruptedFinalization(BaseException):
        pass
    real_gate = loop._enforce_swarm_actions
    def interrupted(*args, **kwargs):
        raise InterruptedFinalization()
    monkeypatch.setattr(loop, "_enforce_swarm_actions", interrupted)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: ({"content": ANSWER}, 0.0))
    with pytest.raises(InterruptedFinalization):
        f.run()
    _restart(f, monkeypatch)
    monkeypatch.setattr(loop, "_enforce_swarm_actions", real_gate)
    def unavailable(*_a, **_kw):
        raise OSError("saved candidate evidence unavailable")
    monkeypatch.setattr("ouroboros.loop_acceptance_review._build_host_acceptance_evidence", unavailable)
    class AuthorDecisionRequired(BaseException):
        pass
    def continued(*_a, **_kw):
        assert f.ctx._delivery_candidate.full_text == ANSWER
        decision = f.ctx._execution_trace["acceptance_decision"]
        assert decision["status"] != "accepted"
        assert not f.ctx._delivery_candidate.acceptance_binding.get("authoritative")
        assert "saved candidate evidence unavailable" in str(f.ctx._execution_trace)
        raise AuthorDecisionRequired()
    monkeypatch.setattr(loop, "call_llm_with_retry", continued)
    with pytest.raises(AuthorDecisionRequired):
        f.run()
    assert not f.review_sends


def test_new_execution_retains_answer_but_cannot_inherit_old_pass(full_loop, monkeypatch):
    f = full_loop
    f.release.set()
    f.ctx.owner_wait_callback = None  # finish the real paid panel before explicit completion
    real_gate = loop._enforce_swarm_actions
    class InterruptedFinalization(BaseException):
        pass
    def interrupted(*args, **kwargs):
        if f.model_step == 2:
            raise InterruptedFinalization()
        return real_gate(*args, **kwargs)
    monkeypatch.setattr(loop, "_enforce_swarm_actions", interrupted)
    def first_main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.model_step == 2 and f.settled.is_set()
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", first_main)
    with pytest.raises(InterruptedFinalization):
        f.run()
    saved = json.loads(wc.checkpoint_path(f.ctx.drive_root, f.ctx.task_id, 1).read_bytes())
    assert saved["trace"]["review_runs"][-1]["aggregate_signal"] == "PASS"
    _restart(f, monkeypatch, new_id=True)
    monkeypatch.setattr(loop, "_enforce_swarm_actions", real_gate)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: pytest.fail("no replacement author call"))
    answer, _, trace = f.run()
    assert answer == ANSWER
    run = trace["review_runs"][0]
    assert run["superseded_by_revision"] and run["aggregate_signal"] == "PASS"
    assert trace["acceptance_decision"]["status"] != "accepted", trace["acceptance_decision"]
    assert not trace["delivery_candidate"]["acceptance_binding"].get("authoritative")
    assert len(f.review_sends) == 1, "the original result remains evidence, not a new paid panel"
