"""N2: complete acceptance prose and optional controls share the existing gates."""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros import loop
from ouroboros.acceptance_settlement import acceptance_wait_chosen
from ouroboros.loop_acceptance_review import advance_explicit_acceptance, wait_for_acceptance_feedback
from tests.test_acceptance_async_loop import ANSWER, call
from tests.test_acceptance_async_loop import full_loop as _full_loop
from tests.test_delivery_forced_finalization import _bind_host_pass, _forced_test_context

full_loop = _full_loop  # noqa: F811 - shared real-loop fixture


def _feedback(tmp_path, monkeypatch, entry="implicit"):
    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, ANSWER, control="candidate")
    registry._ctx._task_acceptance_pending = "paid-binding"
    monkeypatch.setattr("ouroboros.owner_wait.wait_after_tools", lambda *_a, **_k: None)
    if entry == "explicit":
        registry._ctx._acceptance_request_pending = {"subject": ANSWER}
        monkeypatch.setattr(loop, "_no_tool_final_answer", lambda *_a, **_k: None)
        advance_explicit_acceptance(registry, ctx, trace, None, set(), lambda *_a: None)
    else:
        wait_for_acceptance_feedback(registry, ctx, trace, [], set())
    assert candidate.finalization_control == "acceptance_feedback"
    return registry, ctx, trace, candidate


@pytest.mark.parametrize("entry", ["implicit", "explicit"])
@pytest.mark.parametrize("enforcement,mode", [
    ("advisory", "advanced"), ("blocking", "advanced"), ("blocking", "cyber_pro"),
])
@pytest.mark.parametrize("answer", ["prose", "keep", "replace", "finish"])
def test_acceptance_answer_choice_preserves_wait_policy(tmp_path, monkeypatch, entry, enforcement, mode, answer):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    registry, ctx, trace, candidate = _feedback(tmp_path, monkeypatch, entry)
    revised = ANSWER + " Budget: $12.\nThe complete timeline is two weeks."
    registry._ctx._acceptance_pending_review_choice = "finish"
    raw = revised if answer == "prose" else json.dumps({
        "delivery_control": "replace" if answer == "replace" else "keep",
        **({"full_answer": revised} if answer == "replace" else {}),
        **({"pending_review": "finish"} if answer == "finish" else {}),
    })
    status, text = loop._resolve_delivery_control(raw, registry, ctx, trace)
    assert status == ("fresh" if answer == "prose" else "resolved")
    assert text == (revised if answer in {"prose", "replace"} else ANSWER)
    assert registry._ctx._acceptance_pending_review_choice == ("finish" if answer == "finish" else "wait")
    assert acceptance_wait_chosen(registry._ctx) is (
        mode != "cyber_pro" and (enforcement == "blocking" or answer != "finish")
    )
    assert registry._ctx._task_acceptance_pending == "paid-binding"
    assert candidate.control_episode_seen is True
    assert "complete revised user-facing answer as ordinary prose" in str(ctx.messages)
    assert ('"pending_review":"finish"' in str(ctx.messages)) is (
        enforcement == "advisory" and mode != "cyber_pro"
    )


@pytest.mark.parametrize("raw", [
    "", "   ", [{"type": "thinking", "thinking": "reasoning only"}],
    '{"delivery_control":"unknown"}', '{"delivery_control":"replace","full_answer":""}',
    '{"delivery_control":"replace","full_answer":', '{"full_answer":"missing verb"}',
    '{"delivery_control":"keep","pending_review":"unknown"}',
    '{"delivery_control":"keep","delivery_control":"replace","full_answer":"bad"}',
    'A notice.\n{"delivery_control":"keep"}', '```json\n{"delivery_control":\n```',
])
def test_malformed_acceptance_control_retains_complete_answer_after_one_repair(tmp_path, monkeypatch, raw):
    registry, ctx, trace, candidate = _feedback(tmp_path, monkeypatch)
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", "")
    assert candidate.full_text == ANSWER and candidate.repair_attempted
    wait_for_acceptance_feedback(registry, ctx, trace, [], set())
    assert candidate.repair_attempted, "a repeated wake must not refund the repair"
    status, text = loop._resolve_delivery_control(raw, registry, ctx, trace)
    assert (status, text) == ("degraded", ANSWER)
    assert candidate.degraded_reason == "invalid_delivery_control_after_repair"
    assert registry._ctx._task_acceptance_pending == "paid-binding"


def test_complete_prose_can_repair_a_malformed_optional_control(tmp_path, monkeypatch):
    registry, ctx, trace, candidate = _feedback(tmp_path, monkeypatch)
    assert loop._resolve_delivery_control("", registry, ctx, trace) == ("retry", "")
    revised = ANSWER + " Budget: $12."
    assert loop._resolve_delivery_control(revised, registry, ctx, trace) == ("fresh", revised)
    assert candidate.repair_attempted and not candidate.degraded
    assert registry._ctx._acceptance_pending_review_choice == "wait"


def test_changed_owner_source_still_requires_an_exact_current_acknowledgement(tmp_path, monkeypatch):
    registry, ctx, trace, old = _feedback(tmp_path, monkeypatch)
    registry._ctx._owner_directives = [{"content": "Also give the delivery date."}]
    wait_for_acceptance_feedback(registry, ctx, trace, [], set())
    revised = ANSWER + " Delivery date: Friday."
    assert loop._resolve_delivery_control(revised, registry, ctx, trace) == ("retry", "")
    assert old.full_text == ANSWER
    observed = registry._ctx._acceptance_observation
    assert loop._resolve_delivery_control(json.dumps({
        "delivery_control": "replace", "full_answer": revised,
        "acceptance_subject": {"owner_source_sha256": observed["owner_source_sha256"]},
    }), registry, ctx, trace) == ("resolved", revised)
    assert registry._ctx._delivery_candidate.owner_source_sha256 == observed["owner_source_sha256"]


def test_prose_replacement_after_material_change_cannot_inherit_an_old_pass(tmp_path, monkeypatch):
    registry, ctx, trace, old = _feedback(tmp_path, monkeypatch)
    run = _bind_host_pass(loop, registry, trace, old)
    trace["tool_calls"].append({"tool": "write_file", "status": "ok", "is_error": False,
                                "result": "new material evidence"})
    revised = ANSWER + " Budget: $12."
    status, text = loop._resolve_delivery_control(revised, registry, ctx, trace)
    assert (status, text) == ("fresh", revised)
    fresh = loop._replace_delivery_candidate(registry, ctx, trace, text, control="candidate")
    assert fresh.evidence_revision > old.evidence_revision
    assert fresh.evidence_fingerprint != old.evidence_fingerprint
    assert fresh.acceptance_binding["authoritative"] is False
    assert fresh.control_episode_seen is True and run["superseded_by_revision"]


@pytest.mark.parametrize("control", [
    "awaiting_control", "repair_requested", "owner_revision_required",
    "effect_revision_required", "effect_revision_required_repair_requested",
    "skill_revision_required", "skill_action_or_revision_required", "child_absorption_or_revision_required",
])
@pytest.mark.parametrize("entry", ["implicit", "explicit"])
def test_acceptance_arm_does_not_replace_another_gates_control(tmp_path, monkeypatch, control, entry):
    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, ANSWER, control="candidate")
    registry._ctx._task_acceptance_pending = "paid-binding"

    def hold(*_a, **_k):
        if control in loop._DELIVERY_HOLD_CONTROLS:
            loop._hold_delivery_for_skill_action(registry, trace, control=control)
        else:
            loop._arm_delivery_control(registry, ctx, trace, control=control)

    monkeypatch.setattr("ouroboros.owner_wait.wait_after_tools", lambda *_a, **_k: None)
    if entry == "explicit":
        registry._ctx._acceptance_request_pending = {"subject": ANSWER}
        monkeypatch.setattr(loop, "_no_tool_final_answer", hold)
        advance_explicit_acceptance(registry, ctx, trace, None, set(), lambda *_a: None)
    else:
        hold()
        wait_for_acceptance_feedback(registry, ctx, trace, [], set())
    assert candidate.finalization_control == control
    assert registry._ctx._delivery_control_required is (control not in loop._DELIVERY_HOLD_CONTROLS)
    if control not in loop._DELIVERY_HOLD_CONTROLS:
        assert loop._resolve_delivery_control("Status notice.", registry, ctx, trace) == ("retry", "")
        assert candidate.full_text == ANSWER


@pytest.mark.parametrize("raw,expected", [
    ('{"delivery_control":"replace","full_answer":', (ANSWER, True, True, True, False)),
    ('{"delivery_control":"keep"}', (ANSWER, True, False, True, False)),
    ('{"delivery_control":"replace","full_answer":"Complete replacement."}',
     ("Complete replacement.", False, False, True, True)),
    ("Forced complete answer.", ("Forced complete answer.", False, False, True, False)),
])
def test_acceptance_optional_control_keeps_forced_resolution(tmp_path, monkeypatch, raw, expected):
    from ouroboros.loop_delivery import _resolve_forced_delivery_control_body

    registry, _ctx, _trace, candidate = _feedback(tmp_path, monkeypatch)
    assert _resolve_forced_delivery_control_body(
        raw, candidate, armed=registry._ctx._delivery_control_required,
    ) == expected


@pytest.mark.parametrize("entry", ["implicit", "explicit"])
@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_whole_loop_delivers_complete_prose_after_settlement(full_loop, monkeypatch, entry, enforcement):
    f = full_loop
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    revised = ANSWER + " Budget: $12.\nTimeline: two weeks."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            if entry == "explicit":
                return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "review")]}, 0.0
            return {"content": ANSWER}, 0.0
        assert f.model_step < 5, f.progress
        return {"content": revised}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == revised and len(f.review_sends) == 1
    assert [wait["reason"] for wait in f.waits] == ["review"]
    assert not any("[DELIVERY_CONTROL_REPAIR]" in str(messages) for messages in f.model_inputs)
    assert trace["acceptance_decision"]["reason"] == "previous_revision_accepted"
    assert f.review_requests[0].subject == ANSWER
    assert trace["delivery_candidate"]["content_sha256"] != trace["acceptance_decision"]["reviewed_candidate_hash"]


def test_cyber_explicit_nomination_keeps_its_existing_no_wait_power(full_loop, monkeypatch):
    f = full_loop
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    revised = ANSWER + " Budget: $12."
    f.ctx.owner_wait_callback = lambda *_a, **_kw: pytest.fail("Cyber must not acquire a wait")

    def main(_llm, messages, *_a, **_kw):
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "review")]}, 0.0
        assert f.model_step == 2 and f.entered.wait(5) and not f.release.is_set()
        assert "Prose never requests pending_review:finish" in str(messages)
        return {"content": revised}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, usage, trace = f.run()
    assert result == revised and len(f.review_sends) == 1 and not f.waits
    assert trace["acceptance_decision"]["reason"] == "author_finish"
    author = trace["acceptance_decision"]["author_disposition"]
    assert author["source"] == "author_final_response"
    # The submitted final is Main's act: the host records "finish" and invents
    # neither an "accepted" stance nor a "solved" tier (TZ-2 C4).
    assert author["action"] == "finish" and author["disposition"] == ""
    assert trace["review_decision"]["review_pending"] is True
    from ouroboros.outcomes import derive_loop_outcome

    objective = derive_loop_outcome(result, usage, trace)["outcome_axes"]["objective"]
    assert (objective["status"], objective["source"], objective["reason"]) == ("pass", "author_acceptance", "author_finish")
    assert "outcome_tier" not in objective


@pytest.mark.parametrize("early", ["settled", "queued_wake"])
@pytest.mark.parametrize("reply", ["keep", "replace", "prose", "malformed"])
def test_feedback_ready_before_parking_preserves_answer_protocol(tmp_path, monkeypatch, early, reply):
    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(tools, ctx, trace, ANSWER, control="candidate")
    tools._ctx._task_acceptance_pending = "paid-binding"
    monkeypatch.setattr("ouroboros.acceptance_settlement.awaited_panel_has_settled", lambda *_: early == "settled")
    # The stub takes the seam's keyword (owner_authority_only): the observation capture
    # no longer swallows a stub's TypeError behind a blanket except.
    monkeypatch.setattr("ouroboros.loop_transport._owner_signal_pending", lambda *_a, **_k: early == "queued_wake")
    monkeypatch.setattr("ouroboros.owner_wait.wait_after_tools", lambda *_a, **_k: pytest.fail("settled feedback must not park"))
    wait_for_acceptance_feedback(tools, ctx, trace, [], set())
    assert candidate.control_episode_seen
    assert "complete revised user-facing answer as ordinary prose" in str(ctx.messages)
    revised = ANSWER + " The total budget is $12."
    raw = {"keep": json.dumps({"delivery_control": "keep"}),
           "replace": json.dumps({"delivery_control": "replace", "full_answer": revised}),
           "prose": revised, "malformed": '{"delivery_control":"replace","full_answer":'}[reply]
    status, text = loop._resolve_delivery_control(raw, tools, ctx, trace)
    if reply == "malformed":
        assert (status, text) == ("retry", "") and candidate.full_text == ANSWER
    else:
        assert status == ("fresh" if reply == "prose" else "resolved")
        assert text == (ANSWER if reply == "keep" else revised)
        assert tools._ctx._acceptance_pending_review_choice == "wait"
    assert tools._ctx._task_acceptance_pending == "paid-binding"


@pytest.mark.parametrize("order,next_action", [
    (order, action) for order in ("ready", "pending")
    for action in ("rewrite", "effect", "criterion", "nominate", "held_effect")
])
def test_return_order_preserves_feedback_identity_and_new_subjects(full_loop, monkeypatch, order, next_action):
    from tests.test_loop_acceptance_gate import _order_acceptance_feedback
    from tests.test_acceptance_async_loop import keep

    f = full_loop
    revised = ANSWER + " Budget: $12."
    _order_acceptance_feedback(f, monkeypatch, ANSWER, order)
    if next_action == "held_effect" and order == "ready":
        # The new subject's own admission fence is REFUSED (no token). That buys no model
        # round any more: its panel runs on the disclosed rail and the final seal asks again.
        begins = []
        def begin(**_kwargs):
            begins.append(True)
            return None if len(begins) == 2 else {"token": f"fence-{len(begins)}", "owner_message_generation": 0}
        f.ctx.begin_acceptance_fence = begin
        f.ctx.end_acceptance_fence = lambda **kw: {"ok": True, "status": "sealed" if kw["outcome"] == "terminal" else "released"}
        f.ctx.inspect_acceptance_fence = lambda **_kw: {"owner_message_generation": 0}

    def main(_llm, messages, *_args, **_kwargs):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first")]}, 0.0
        if f.model_step == 2:
            if order == "ready":
                assert "Review verdict: PASS for the nominated answer" in str(messages)
                assert "A paid acceptance panel on an earlier revision is still running" not in str(messages)
                assert not f.ctx._task_acceptance_pending
                runs = f.ctx._execution_trace["review_runs"]
                index, offered = next((i, row) for i, row in enumerate(runs) if row.get("authority") == "host_root")
                assert offered["feedback_offered"] and not offered.get("feedback_delivered")
                assert any(source == {"task_id": f.run_args["task_id"], "run_index": index,
                                      "binding_hash": offered["binding_hash"]}
                           for message in messages for source in message.get("review_feedback", []))
            if next_action == "effect" or (next_action == "held_effect" and order == "ready"):
                return {"content": "", "tool_calls": [call("write_file", {
                    "root": "task_drive", "path": "new-effect.txt", "content": "Additional evidence.",
                }, "effect")]}, 0.0
            if next_action == "nominate":
                return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": revised}, "second")]}, 0.0
        if f.model_step == 3 and next_action == "held_effect" and order == "pending":
            assert f.waits  # The real pending-panel wait held the rewritten answer.
            return {"content": "", "tool_calls": [call("write_file", {
                "root": "task_drive", "path": "new-effect.txt", "content": "Additional evidence.",
            }, "held-effect")]}, 0.0
        if (f.model_step == 2 or (f.model_step == 3 and next_action == "effect")
                or (f.model_step == 3 and next_action == "held_effect" and order == "ready")
                or (f.model_step == 4 and next_action == "held_effect")):
            subject = {"owner_source_sha256": f.ctx._acceptance_observation["owner_source_sha256"]}
            if next_action == "criterion":
                subject["effective_criteria"] = "Complete report including a verified budget of $12."
            if next_action in {"effect", "held_effect"} and f.model_step > 2:
                subject["material_tool_indices"] = [1]
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": revised,
                                           "acceptance_subject": subject})}, 0.0
        assert f.model_step < 7, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == revised
    assert len(f.review_sends) == (1 if next_action == "rewrite" else 2)
    assert f.review_requests[0].subject == ANSWER
    if order == "ready":
        first_run = next(row for row in trace["review_runs"] if row.get("authority") == "host_root")
        assert first_run["feedback_delivered"]  # The real loop's returned-request observer exposed it.
    if next_action in {"effect", "held_effect"}:
        effect = f.ctx.drive_root / "task_drives" / f.run_args["task_id"] / "new-effect.txt"
        assert effect.read_text(encoding="utf-8") == "Additional evidence."
    if next_action == "held_effect" and order == "ready":
        # The refused fence bought no round: the new subject was reviewed at once on the
        # disclosed rail (the fourth round is the ordinary post-PASS control round, as for
        # `effect`), and the answered re-seal closed admission as usual.
        assert f.model_step == 4 and "TASK ACCEPTANCE WAIT" not in str(f.model_inputs)
        assert len(begins) == 3  # the nomination's, the refused one, the answered re-seal
        assert trace["acceptance_decision"]["reason"] == "clean_pass"
    if next_action == "rewrite":
        decision = trace["acceptance_decision"]
        assert decision["reason"] == "previous_revision_accepted"
        assert decision["reviewed_candidate_hash"] != trace["delivery_candidate"]["content_sha256"]
        assert not trace["delivery_candidate"]["acceptance_binding"]["authoritative"]
    else:
        assert f.review_requests[-1].subject == revised
        assert trace["acceptance_decision"]["reason"] != "previous_revision_accepted"


@pytest.mark.parametrize("delivered,reason,signal,reused", [
    (True, "delivery_candidate_replaced", "PASS", True),
    (False, "delivery_candidate_replaced", "PASS", False),  # Main never received the verdict
    (True, "delivery_evidence_changed_after_host_acceptance", "PASS", False),
    (True, "delivery_candidate_replaced", "FAIL", False),
    (True, "delivery_candidate_replaced", "DEGRADED", False),
])
def test_ready_fallback_reuses_only_a_delivered_pass_on_a_replaced_text(
        tmp_path, monkeypatch, delivered, reason, signal, reused):
    """The ready-order lookup falls toward a fresh panel, never toward acceptance."""
    from types import SimpleNamespace

    from ouroboros.acceptance_settlement import _deliver_under_running_panel
    from ouroboros.loop_delivery import delivery_subject_hash

    loop_mod, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop_mod._replace_delivery_candidate(registry, ctx, trace, ANSWER, control="candidate")
    trace["review_runs"] = [{
        "authority": "host_root", "request": {"subject": ANSWER}, "aggregate_signal": signal,
        "actors": [{"operation_state": "settled"}], "binding_hash": "binding-one", "panel_id": "panel-one",
        "candidate_hash": candidate.content_sha256,
        "subject_hash": delivery_subject_hash(registry._ctx, trace, ANSWER),
        "feedback_delivered": delivered, "superseded_by_revision": True, "superseded_reason": reason,
    }]
    revised = ANSWER + " Reworded only."
    loop_mod._replace_delivery_candidate(registry, ctx, trace, revised, control="replace")
    registry._ctx._task_acceptance_pending = ""  # a ready return never arms the pending binding
    monkeypatch.setattr("ouroboros.loop_acceptance_review._end_acceptance_terminal", lambda *_a, **_k: None)
    review_ctx = SimpleNamespace(tools=registry, llm_trace=trace, content=revised, task_id=registry._ctx.task_id,
                                 emit_progress=lambda _text: None, review_binding={}, messages=ctx.messages)

    outcome = _deliver_under_running_panel(review_ctx, None)

    assert outcome is (False if reused else None)
    assert ((trace.get("acceptance_decision") or {}).get("reason") == "previous_revision_accepted") is reused
