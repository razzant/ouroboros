"""Forced delivery applies the same source-addressed Main subject decision.

A forced rail's control answer may carry ``acceptance_subject``: it acknowledges
the exact owner source Main was shown and may change the criteria, through the
same ``apply_delivery_subject_decision`` an ordinary completion uses, and the
trace records whether it applied (``forced_acceptance_subject``). The forced rail
buys no review. The task-acceptance binding these tests once checked for
authority is gone; a retained answer's binding is never authoritative now.
"""
from __future__ import annotations

import copy
import json
import queue
import time

import pytest

from ouroboros import loop_forced_finalization as forced
from ouroboros.loop_delivery import delivery_subject_hash
from ouroboros.loop_messages import _record_owner_directive, capture_acceptance_observation, owner_source_sha256
from tests.test_delivery_forced_finalization import _forced_test_context


def _bound(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path, incoming=queue.Queue())
    _record_owner_directive(registry._ctx, source="initial_user", content="Prepare the full answer.", msg_id="initial")
    capture_acceptance_observation(registry._ctx, trace, ctx.incoming_messages)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "The complete verified answer.", control="awaiting_control",
    )
    registry._ctx._delivery_control_required = True
    return loop, registry, ctx, trace, candidate


@pytest.mark.parametrize("followup", ["Как дела?", "Нужен тот же полный ответ, как договаривались."])
def test_forced_status_ack_keeps_the_old_subject(tmp_path, monkeypatch, followup):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    old = delivery_subject_hash(registry._ctx, trace)
    old_source = owner_source_sha256(registry._ctx)
    ctx.incoming_messages.put(followup)
    calls = []

    def answer(_ctx, **_kwargs):
        observed = dict(registry._ctx._acceptance_observation)
        calls.append(observed)
        assert observed["owner_source_sha256"] != old_source
        assert observed["owner_source_sha256"] in json.dumps(ctx.messages)
        assert followup in json.dumps(ctx.messages, ensure_ascii=False)
        return json.dumps({"delivery_control": "keep", "acceptance_subject": {
            "owner_source_sha256": observed["owner_source_sha256"],
        }})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text, reason_code="round_limit",
    )
    assert len(calls) == 1
    assert text == candidate.full_text
    assert delivery_subject_hash(registry._ctx, trace) == old
    assert result["forced_acceptance_subject"] == {"applied": True, "reason": ""}
    assert registry._ctx._owner_directives[-1]["content"] == followup


def test_forced_new_criterion_with_same_answer_is_a_new_subject(tmp_path, monkeypatch):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    old = delivery_subject_hash(registry._ctx, trace)
    old_answer_hash = candidate.content_sha256
    ctx.incoming_messages.put("Also check the budget.")
    calls = []

    def answer(_ctx, **_kwargs):
        calls.append(1)
        return json.dumps({"delivery_control": "keep", "acceptance_subject": {
            "owner_source_sha256": registry._ctx._acceptance_observation["owner_source_sha256"],
            "effective_criteria": "Full verified answer including the checked budget.",
        }})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    monkeypatch.setattr("ouroboros.tools.review_change.run_review_change",
                        lambda *_a, **_kwargs: pytest.fail("no forced paid review"))
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text, reason_code="round_limit",
    )
    assert calls == [1] and text == candidate.full_text
    assert registry._ctx._delivery_candidate.content_sha256 == old_answer_hash
    assert delivery_subject_hash(registry._ctx, trace) != old
    assert registry._ctx._delivery_candidate.effective_criteria.endswith("checked budget.")
    assert result["forced_acceptance_subject"] == {"applied": True, "reason": ""}


@pytest.mark.parametrize("invalid", [
    {"owner_source_sha256": "not-the-observed-source"},
    {"effective_criteria": "Changed without an acknowledgement"},
    {"material_tool_indices": [999]},
])
def test_an_invalid_forced_subject_is_not_applied(tmp_path, monkeypatch, invalid):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)

    def answer(_ctx, **_kwargs):
        subject = {"owner_source_sha256": registry._ctx._acceptance_observation["owner_source_sha256"]}
        subject.update(invalid)
        if "owner_source_sha256" not in invalid and "effective_criteria" in invalid:
            subject.pop("owner_source_sha256")
        return json.dumps({"delivery_control": "keep", "acceptance_subject": subject})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text, reason_code="round_limit",
    )
    assert text == candidate.full_text
    assert result["forced_acceptance_subject"]["applied"] is False


def test_arrival_during_single_forced_send_does_not_ack_unseen_source_or_resend(tmp_path, monkeypatch):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    observed = []

    def answer(_ctx, **_kwargs):
        snapshot = dict(registry._ctx._acceptance_observation)
        observed.append(snapshot)
        ctx.incoming_messages.put("Late new requirement.")
        return json.dumps({"delivery_control": "keep", "acceptance_subject": {
            "owner_source_sha256": snapshot["owner_source_sha256"],
        }})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    _text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text,
        reason_code="owner_requested_finalization", single_semantic_turn=True,
    )
    assert len(observed) == 1
    assert registry._ctx._acceptance_observation == observed[0]
    assert registry._ctx._acceptance_ack_source_sha256 != owner_source_sha256(registry._ctx)
    assert registry._ctx._owner_directives[-1]["content"] == "Late new requirement."


def test_existing_refresh_reobserves_before_second_send_not_after_first_reply(tmp_path, monkeypatch):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    snapshots = []

    def answer(_ctx, **_kwargs):
        snapshot = dict(registry._ctx._acceptance_observation)
        snapshots.append(snapshot)
        if len(snapshots) == 1:
            ctx.incoming_messages.put("Как дела?")
        return json.dumps({"delivery_control": "keep", "acceptance_subject": {
            "owner_source_sha256": snapshot["owner_source_sha256"],
        }})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text, reason_code="round_limit",
    )
    assert len(snapshots) == 2
    assert snapshots[0]["owner_source_sha256"] != snapshots[1]["owner_source_sha256"]
    assert registry._ctx._acceptance_ack_source_sha256 == snapshots[1]["owner_source_sha256"]
    assert text == candidate.full_text
    assert result["forced_acceptance_subject"] == {"applied": True, "reason": ""}


def test_prepared_budget_request_keeps_its_first_observation_and_exact_messages(tmp_path, monkeypatch):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    ctx.incoming_messages.put("Status?")
    prompt = forced._prepare_forced_prompt(ctx, "Budget final", trace)
    first_observation = copy.deepcopy(registry._ctx._acceptance_observation)
    send_messages = copy.deepcopy(ctx.messages)
    loop._append_or_merge_user_message(send_messages, prompt)
    prepared = object()
    calls = []

    def answer(_ctx, **kwargs):
        calls.append(kwargs)
        assert registry._ctx._acceptance_observation == first_observation
        assert ctx.messages == send_messages
        return json.dumps({"delivery_control": "keep", "acceptance_subject": {
            "owner_source_sha256": first_observation["owner_source_sha256"],
        }})

    monkeypatch.setattr(loop, "_call_forced_model_once", answer)
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt=prompt, fallback_text=candidate.full_text, reason_code="budget_exhausted",
        _prompt_prepared=True, _initial_messages=send_messages, _admitted_request=prepared,
    )
    assert len(calls) == 1
    assert calls[0]["admitted_request"] is prepared
    assert calls[0]["initial_messages"] is send_messages
    assert text == candidate.full_text and result["forced_acceptance_subject"]["applied"]


def test_elapsed_deadline_preserves_input_without_claiming_it_was_processed(tmp_path, monkeypatch):
    loop, registry, ctx, trace, candidate = _bound(tmp_path)
    old_ack = registry._ctx._acceptance_ack_source_sha256
    ctx.incoming_messages.put("New budget criterion.")
    ctx.deadline_ts = time.time() - 1
    monkeypatch.setattr(loop, "_call_forced_model_once", lambda *_a, **_k: pytest.fail("deadline already elapsed"))
    text, _usage, result = forced._forced_final_answer(
        ctx, prompt="Finish now", fallback_text=candidate.full_text, reason_code="deadline",
    )
    assert candidate.full_text in text
    assert registry._ctx._acceptance_ack_source_sha256 == old_ack
    assert old_ack != owner_source_sha256(registry._ctx)
    assert "forced_acceptance_subject" not in result
