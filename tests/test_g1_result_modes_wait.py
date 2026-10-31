"""What outlives the root acceptance wait: a review wait's owner controls and the
pending-review choice.

A review park (``owner_wait.wait_after_tools(review_binding=...)``) that the owner's Stop
or Panic ends keeps its paid operation's binding. The author's ``finish_task(
pending_review="wait"|"finish")`` choice, which ``review_result.result_review_at_delivery``
reads for a result review still running at delivery, survives a cold resume and never
leaks into a new turn. Nomination, the acceptance panel and its paid wait are retired
with task acceptance.
"""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros import loop

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('control', ['stop', 'panic'])
def test_native_paid_wait_retains_operation_when_owner_controls_end_wait(tmp_path, monkeypatch, control):
    from tests.test_native_owner_wait import native_context
    from ouroboros.owner_wait import wait_after_tools
    from ouroboros.task_results import load_task_result
    from ouroboros.budget_pause import _hold_control_reason
    from ouroboros.cancel_intents import request_cancel
    ctx = native_context(tmp_path)
    trace = {'review_runs': [{'authority': 'host_root', 'binding_hash': 'already-paid',
                             'actors': [{'operation_id': 'same-operation', 'operation_state': 'in_flight'}]}]}
    before = copy.deepcopy(trace)
    # This direct actor binds the existing owner control reader, with real
    # disposable-root cancellation and the existing direct control callback.
    ctx.model_wait_context.owner_control = lambda: _hold_control_reason(ctx)
    if control == 'stop':
        request_cancel(tmp_path, ctx.task_id, reason='owner stopped', source='owner', requested_by='owner')
    else:
        ctx.model_wait_context.owner_control = lambda: 'panic'
    monkeypatch.setattr('ouroboros.owner_wait.time.sleep', lambda *_: pytest.fail('control must precede sleep'))
    wait_after_tools(ctx, [], trace, {}, 1, [], set(), review_binding='already-paid')
    assert trace == before
    wait = load_task_result(tmp_path, ctx.task_id)['owner_wait']
    assert wait['review_binding'] == 'already-paid'
    assert wait['resume_reason'] == ('control:cancelled' if control == 'stop' else 'control:panic')


@pytest.mark.parametrize("choice", ["wait", "finish", None])
def test_cold_resume_keeps_the_current_choice_and_never_a_stale_finish(tmp_path, monkeypatch, choice):
    """The validated wait/finish choice rides the one continuation serializer through a JSON
    (cold) checkpoint and a fresh restore after the hold cleared its selection; a new turn and
    a new execution's working restore start from waiting."""
    from types import SimpleNamespace

    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from ouroboros.tools.control_runtime import stage_completion_request
    from tests.test_delivery_control_lineage import _start_control_episode

    def finishes(tools_ctx):
        """Whether the author chose to deliver over a review still running (the field the
        delivery step reads); waiting is the default."""
        return str(getattr(tools_ctx, "_acceptance_pending_review_choice", "") or "") == "finish"

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    loop_module, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop_module._arm_delivery_control(registry, ctx, trace)
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    request = {"action": "finish", "answer_sha256": candidate.content_sha256, **({"pending_review": choice} if choice else {})}
    assert json.loads(stage_completion_request(registry._ctx, request))["status"] == "completion_requested"
    assert consume_completion_request(registry, ctx, trace)
    expected = choice == "finish"
    assert finishes(registry._ctx) is expected
    registry._ctx._completion_selected = None  # the real hold clears the consumed selection
    state = json.loads(json.dumps(continuation_state(registry._ctx, ctx.messages, trace, {}, 2, [], set())))

    def restored(saved):
        # The resumed task's own live context: the same task and attempt that saved the state.
        tools = SimpleNamespace(_ctx=SimpleNamespace(task_id=saved["task_id"], task_attempt=saved["task_attempt"]))
        restore_continuation_state(tools, saved, [], {}, {}, set())
        return tools

    cold = restored(state)
    assert finishes(cold._ctx) is expected, "a cold resume never turns an explicit finish into wait"
    turn = restored(state)
    loop._reset_turn_state(turn._ctx)
    assert finishes(turn._ctx) is False, "a new turn inherits no stale finish"
    retry = restored({**state, "acceptance": {}})  # resume_from_working's new-execution projection
    assert finishes(retry._ctx) is False
