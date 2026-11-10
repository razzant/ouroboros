"""A completed batch's delegation requests are anchored before it is saved or parked.

The real loop executes the batch, writes ``post_batch`` and may park in an owner
wait before the next round's budget tail runs. A cold continuation of either
source rejoins that tail with no batch to replay, so the anchors must already be
in the saved state: nothing is re-executed and no history is scanned. A legacy
checkpoint's unknown anchors stay stated in the forced-finalization note."""
from __future__ import annotations

import json

import pytest

from ouroboros import loop
from ouroboros import working_checkpoint as wc
from ouroboros.agent_dispatch import reset_nanny_economics_marks
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.owner_wait import set_owner_wait
from ouroboros.task_results import write_task_result
from tests.test_loop_compaction import _fit, _measure_cycle
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_working_checkpoint import _registry


class ProcessLost(BaseException):
    """The worker ends here; only what was saved crosses to the next process."""


def _call(call_id, name, arguments=None):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {})}}


def _kwargs(tmp_path, registry):
    return {**_loop_kwargs(tmp_path, registry, []), "emit_progress": lambda *_a, **_k: None}


def _nanny(tmp_path, monkeypatch, attempt=1):
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))  # fit is not under test
    registry = _registry(tmp_path, monkeypatch, "t-wait", attempt)
    # agent dispatch seeds every start, before a cold loop restores its source
    reset_nanny_economics_marks(registry._ctx, route_dispatched=True)
    return registry


def _start_batch(tmp_path, monkeypatch, registry, verb, *, park, batch_cost=0.0):
    """Round 1 requests ``verb`` (and an owner question when parking); the process ends
    at the ``post_batch`` save or at the park, before the next round's budget tail."""
    write_task_result(tmp_path, "t-wait", "running")
    executed = []
    def execute(ctx, **_kw):
        executed.append(verb)
        # Synthetic in-batch cost tests the consumer boundary, not a production caption path.
        ctx._accumulated_usage["cost"] += batch_cost
        return "run requested"
    registry.override_handler(verb, execute)
    calls = [_call("verb", verb)]
    if park:
        def ask(ctx, **_kw):
            executed.append("escalate")
            ctx._owner_wait_requested = "quiz-1"  # the quiz tool's observable request (core_artifacts)
            return "Question posted; waiting for the owner"
        registry.override_handler("escalate", ask)
        calls.append(_call("ask", "escalate", {"question": "Continue?", "wait_for_answer": True}))
    parked, tails = [], []

    def respond(_llm, _messages, *args, **_kw):
        round_idx, usage = args[6], args[8]
        assert round_idx == 1, "the process ends before another author call"
        usage["cost"] = 1.5
        return {"role": "assistant", "content": "", "tool_calls": calls}, 0.0

    def park_and_restart(ctx, checkpoint):
        parked.append(checkpoint)
        set_owner_wait(tmp_path, "t-wait", {**checkpoint, "state": "waiting"})
        raise ProcessLost("planned restart while parked")

    real_save = wc.save_or_log

    def save_then_lose(limit_ctx, boundary):
        saved = real_save(limit_ctx, boundary)
        if boundary == "post_batch" and not park:
            raise ProcessLost("worker lost after the post_batch save")
        return saved

    registry._ctx.owner_wait_callback = park_and_restart
    with monkeypatch.context() as process:
        process.setattr(loop, "call_llm_with_retry", respond)
        process.setattr(wc, "save_or_log", save_then_lose)
        process.setattr(loop, "_finish_tool_round_budget", lambda *a, **k: tails.append(a) or pytest.fail(
            "the process ended before its budget tail"))
        with pytest.raises(ProcessLost):
            loop.run_llm_loop(**_kwargs(tmp_path, registry))
    assert executed == ([verb, "escalate"] if park else [verb]) and tails == []
    return parked[0] if park else None


def _continue_cold(tmp_path, monkeypatch, registry, verb, *, total_cost=1.5):
    """The next process: the real loop restores the saved source and rejoins its tool tail."""
    for name in (verb, "escalate"):
        registry.override_handler(name, lambda *_a, _name=name, **_k: pytest.fail(f"{_name} replayed"))
    seen, tails = {}, []
    real_tail = loop._finish_tool_round_budget

    def tail(limit_ctx, *args, **kwargs):
        tails.append(limit_ctx.round_idx)
        return real_tail(limit_ctx, *args, **kwargs)

    def respond(_llm, messages, *args, **_kw):
        # The continuation's first author call is the observation; how it then
        # finishes is not under test here.
        seen["round"] = args[6]
        seen["facts"] = [row["content"] for row in messages if row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME]
        raise ProcessLost("observed the first author call")

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(loop, "_finish_tool_round_budget", tail)
    with pytest.raises(ProcessLost):
        loop.run_llm_loop(**_kwargs(tmp_path, registry))
    assert registry._ctx._accumulated_usage["cost"] == total_cost
    # The cold tail is the same single budget decision at the batch's round.
    assert tails == [1] and seen["round"] == 2
    return seen["facts"][-1]


@pytest.mark.parametrize(("batch_cost", "park", "verb"), [
    pytest.param(0.0, False, "delegate_start", id='no_tool_cost-post_batch_recovery-delegate_start'),
    pytest.param(0.75, True, "delegate_wait", id='metered_tool-owner_wait_park-delegate_wait'),
])
def test_a_cold_continuation_keeps_the_completed_batch_delegation_anchors(tmp_path, monkeypatch, verb, park, batch_cost):
    wait = _start_batch(tmp_path, monkeypatch, _nanny(tmp_path, monkeypatch), verb,
                        park=park, batch_cost=batch_cost)
    if park:
        registry = _nanny(tmp_path, monkeypatch)
        registry._ctx.owner_wait_resume = {**wait, "restart_transaction_id": "observed-restart"}
        registry._ctx.owner_wait_callback = lambda *_: None
    else:
        task: dict = {"id": "t-wait"}
        assert wc.attach_recovery(tmp_path, task, source_task_id="t-wait", from_attempt=1, cause="worker_crash")
        registry = _nanny(tmp_path, monkeypatch, attempt=2)
        registry._ctx.working_recovery = task["_working_recovery"]
    facts = _continue_cold(tmp_path, monkeypatch, registry, verb, total_cost=1.5 + batch_cost)
    assert "nanny: 0 own rounds since the last delegate verb call (round 1)" in facts
    if verb == "delegate_start":
        assert registry._ctx._nanny_delegate_baseline == {"round": 1, "cost": 1.5, "cost_round": 1}
        assert f"own recorded cost ${batch_cost:.2f} since the last delegate_start/schedule_subagent call (round 1)" in facts
    else:  # supervision moves the round anchor only; dollars still count from task start
        assert registry._ctx._nanny_delegate_baseline == {"round": 1, "cost": None, "cost_round": None}
        assert f"own recorded cost ${1.5 + batch_cost:.2f} since task start (no delegate_start/schedule_subagent call yet)" in facts
    assert "no delegate_start/schedule_subagent call yet; since task start" not in facts


# -- forced finalization over a succeeded run ---------------------------------------


def _restored_legacy(tmp_path, *marks):
    """The real serializer's checkpoint from an older writer (no ``nanny_activity``)."""
    from types import SimpleNamespace

    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    old = SimpleNamespace(task_id="child-1", task_attempt=1)
    saved = json.loads(json.dumps(continuation_state(old, [], {}, {"cost": 5.0}, 8, [], set())))
    saved.pop("nanny_activity", None)
    ctx = SimpleNamespace(task_id="child-1", task_attempt=1, drive_root=tmp_path)
    reset_nanny_economics_marks(ctx, route_dispatched=True)
    usage: dict = {}
    restore_continuation_state(SimpleNamespace(_ctx=ctx), saved, [], {}, usage, set())
    ctx._accumulated_usage = usage
    for round_idx, cost, verbs in marks:
        usage["cost"] = cost
        loop._note_nanny_delegate_activity(ctx, round_idx, usage, [_call("c", v) for v in verbs])
    return ctx


def _succeeded(tmp_path):
    from tests.test_nanny_economics import _custody_drive, _succeeded_run

    _succeeded_run(_custody_drive(tmp_path))


def test_forced_note_states_unknown_legacy_anchors(tmp_path):
    _succeeded(tmp_path)
    note = loop._forced_delegation_note(_restored_legacy(tmp_path), {"tool_calls": []})
    assert "delegated run(s) succeeded" in note and "unknown own rounds" in note
    assert "historical cost anchor unavailable" in note
    # A new supervision request supplies the round anchor only; the dollar anchor stays unknown.
    partial = _restored_legacy(tmp_path, (9, 6.0, ["delegate_wait"]))
    note = loop._forced_delegation_note(partial, {"tool_calls": []})
    assert "0 own rounds since the last delegate verb call (round 9)" in note
    assert "historical cost anchor unavailable" in note


def test_forced_note_is_silent_again_once_new_requests_supply_known_anchors(tmp_path):
    _succeeded(tmp_path)
    renewed = _restored_legacy(tmp_path, (9, 6.0, ["delegate_start"]))
    assert loop._forced_delegation_note(renewed, {"tool_calls": []}) == ""
    burned = _restored_legacy(tmp_path, (9, 6.0, ["delegate_start"]), (11, 7.0, []))
    assert "2 own rounds since the last delegate verb call (round 9)" in loop._forced_delegation_note(
        burned, {"tool_calls": []})
