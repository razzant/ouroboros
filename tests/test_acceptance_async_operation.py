"""Delivery control around a review: the review park, an owner follow-up's release of
a selected completion, the re-armed selection contract and the pending-review choice.

The frozen task-acceptance operation this module once pinned (its free collection,
its settlement wakes and its late-settlement rows) is gone with the acceptance
machine; a task's result is reviewed as ``review_change(subject="result")``.
"""

import json

import pytest


def test_review_park_is_not_a_question_and_preserves_operation(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import wait_after_tools
    from tests.test_owner_wait import context

    ctx, captured = context(tmp_path), []
    ctx._owner_wait_requested = ""
    ctx.owner_wait_callback = lambda owner, checkpoint: captured.append(checkpoint)
    trace = {"review_runs": [{"binding_hash": "binding-original", "request": {"subject": "full result"}}]}
    wait_after_tools(ctx, [], trace, {}, 3, [], set(), review_binding="binding-original")
    assert len(captured) == 1
    assert captured[0]["quiz_id"] == ""
    assert captured[0]["reason"] == "review"
    source = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, captured[0]["source_ref"]))
    assert source["review_binding"] == "binding-original"
    assert source["trace"] == trace


def _host_run(**overrides):
    run = {"authority": "host_root", "request": {"surface": "task_acceptance", "retry_key": "subject"},
           "slot_roster": [{"slot_id": "one", "model": "m", "route": "api_chat"}],
           "actors": [{"operation_state": "pending_dispatch"}]}
    run.update(overrides)
    return run


@pytest.mark.parametrize("choice", [None, "wait", "finish"])
def test_an_owner_followup_reopens_delivery_and_keeps_the_review_and_the_wait_choice(tmp_path, choice):
    """New input reopens delivery: the selected completion is void, while the paid
    review and the author's wait choice stay exactly as they were.

    Arrival itself cannot cancel the review or choose early finish on the author's
    behalf, and it writes no decision of its own over the review's record.
    """
    from ouroboros import loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, _ctx, trace = _forced_test_context(tmp_path)
    if choice is not None:
        registry._ctx._acceptance_pending_review_choice = choice
    trace["review_decision"] = {}
    paid = _host_run(binding_hash="binding-one", paid_identity="paid-one")
    trace["review_runs"] = [paid]
    before = json.loads(json.dumps(paid))
    trace["task_completion"] = {"action": "stop", "rationale": "Unfinished.", "answer_sha256": "a" * 64}
    registry._ctx._task_acceptance_fence_generation_mismatch = True
    assert loop._release_for_owner_followup(registry._ctx, trace)
    assert getattr(registry._ctx, "_acceptance_pending_review_choice", None) == choice
    assert "task_completion" not in trace
    assert registry._ctx._task_acceptance_fence_generation_mismatch is False
    assert trace["review_runs"] == [paid] and paid == before  # raw actors/request/paid identity stay exact
    assert trace["review_decision"] == {} and "acceptance_decision" not in trace


def test_the_rearmed_contract_never_rewrites_an_already_sent_row(tmp_path):
    """Issue #906: merging into a sent row discards the conversation cache. Every
    arm appends the control block in the execution slot."""
    import copy as _copy

    from ouroboros.transcript_prefix import observe_send
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(registry, ctx, trace, "Complete answer.", control="candidate")
    ctx.messages.append({"role": "user", "content": "An owner follow-up that already went out."})
    observe_send(registry._ctx, ctx.messages, round_idx=1)
    sent = _copy.deepcopy(ctx.messages[-1])

    loop._arm_delivery_control(registry, ctx, trace)

    assert sent in ctx.messages, "an already-sent message was rewritten"
    control = [row for row in ctx.messages
               if "[DELIVERY_FINALIZATION_CONTROL]" in str(row.get("content") or "")]
    assert len(control) == 1 and control[0] is not ctx.messages[ctx.messages.index(sent)]
    # Only a repeated re-offer (``skip_if_unchanged``) is deduplicated. Every other
    # caller arms because something changed, so it always appends.
    loop._arm_delivery_control(registry, ctx, trace)
    assert str(ctx.messages).count("[DELIVERY_FINALIZATION_CONTROL]") == 2


@pytest.mark.parametrize("choice", ["wait", "finish"])
def test_pending_review_rides_beside_the_verb_and_is_recorded_on_every_answer(tmp_path, choice):
    """WP-7: the optional wait/finish choice is a sibling of ``acceptance_subject``,
    never an extra key that invalidates the body, and every control answer records
    it (an answer without the key means wait)."""
    from tests.test_delivery_control_lineage import _start_control_episode
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import stage_completion_request

    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    assert json.loads(stage_completion_request(registry._ctx, {
        "action": "finish", "answer_sha256": candidate.content_sha256, "pending_review": choice,
    }))["status"] == "completion_requested"
    assert consume_completion_request(registry, ctx, trace)
    assert registry._ctx._delivery_candidate.full_text == candidate.full_text
    assert registry._ctx._acceptance_pending_review_choice == choice
    loop._arm_delivery_control(registry, ctx, trace)
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    assert json.loads(stage_completion_request(registry._ctx, {
        "action": "finish", "answer_sha256": candidate.content_sha256,
    }))["status"] == "completion_requested"
    assert consume_completion_request(registry, ctx, trace)
    assert registry._ctx._acceptance_pending_review_choice == "wait"


@pytest.mark.parametrize("content", ["", [{"type": "thinking", "thinking": "reasoning only"}], "invalid control"])
def test_held_prose_keeps_the_sent_control_prefix(tmp_path, content):
    import copy
    from ouroboros.transcript_prefix import observe_send
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    ctx.messages.insert(0, {"role": "system", "content": "Complete the task."})
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Complete retained answer.", control="candidate")
    loop._arm_delivery_control(registry, ctx, trace)
    observe_send(registry._ctx, ctx.messages, round_idx=1)
    sent = copy.deepcopy(ctx.messages)

    status, text = loop._resolve_delivery_control(content, registry, ctx, trace)

    assert (status, text) == ("retry", "Complete retained answer.")
    assert ctx.messages[:len(sent)] == sent
    assert "No completion selection was made" in ctx.messages[-1]["content"]
    assert ctx.messages[-1]["role"] == "user"
    assert observe_send(registry._ctx, ctx.messages, round_idx=2) is None
    assert candidate.full_text == "Complete retained answer."
