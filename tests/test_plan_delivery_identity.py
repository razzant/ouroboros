"""Delivery participates in open-wave identity; saved authority stays frozen."""
import dataclasses
import pathlib

import pytest

from ouroboros.review_records import ReviewSlot
from ouroboros.tools.plan_review_artifacts import frozen_delivery_inputs, frozen_plan_slots, slot_row
from ouroboros.tools.plan_review_runtime import plan_reviewer_config_fingerprint


def test_delivery_fingerprint_preserves_legacy_and_separates_direct_native():
    slot = ReviewSlot(slot_id="one", model="m/a", effort="high")
    assert plan_reviewer_config_fingerprint([slot]) == plan_reviewer_config_fingerprint([
        dataclasses.replace(slot, native_retrieval_override=False)])
    assert plan_reviewer_config_fingerprint([slot]) != plan_reviewer_config_fingerprint([
        dataclasses.replace(slot, native_retrieval_override=True)])


def test_legacy_paid_packet_uses_saved_input_even_when_current_slot_is_native():
    slot = ReviewSlot(slot_id="one", model="m/a", effort="high")
    legacy = {k: v for k, v in slot_row(slot).items()
              if k not in ("delivery", "subagent_id", "use_local", "processing_preference")}
    messages = [{"role": "user", "content": "exact saved packet"}]
    wave = {"slots": [legacy], "actors": [{"slot_id": "one", "operation_state": "in_flight"}],
            "reviewer_outputs": [{"slot_id": "one", "request_messages": messages}],
            "request_policy": {"old_policy": True}, "slot_prompt_chars": {"one": 123}}
    current = dataclasses.replace(slot, native_retrieval_override=True, model="changed")
    assert frozen_delivery_inputs(wave, [current])["slot_messages"] == {"one": messages}
    frozen, = frozen_plan_slots(wave)
    assert not frozen.native_retrieval and frozen.model == "m/a"


@pytest.mark.parametrize("damage", ["", "missing", "mismatched"])
def test_legacy_actor_binding_comes_from_verified_paid_request(tmp_path, damage):
    from ouroboros.observability import persist_call
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

    slot = ReviewSlot(slot_id="one", model="m/a", effort="high", subagent_id="saved-actor",
                      use_local=True, processing_preference="saved-preference")
    legacy = {k: v for k, v in slot_row(slot).items()
              if k not in ("delivery", "subagent_id", "use_local", "processing_preference")}
    ref = persist_call(tmp_path, task_id="task", call_id="paid_prompt", call_type="plan_review_prompt",
                       payload={"slot": dataclasses.asdict(slot)})
    wave = {"slots": [legacy], "reviewer_outputs": [
        {"slot_id": "one", "delivery_class": "native_retrieving", "prompt_ref": ref}]}
    if damage == "missing":
        pathlib.Path(ref["manifest_ref"]["path"]).unlink()
    elif damage == "mismatched":
        legacy["model"] = "another-model"
    if damage:
        with pytest.raises(PlanReviewSourceUnavailable, match="legacy paid slot"):
            frozen_plan_slots(wave, state_root=tmp_path, task_id="task")
    else:
        frozen, = frozen_plan_slots(wave, state_root=tmp_path, task_id="task")
        assert frozen.native_retrieval and frozen.subagent_id == slot.subagent_id
        assert frozen.use_local and frozen.processing_preference == slot.processing_preference
        # The lane-era actor-bound row retrieved natively; F8 states that as the
        # slot's own delivery fact, so the restored identity carries it explicitly.
        assert plan_reviewer_config_fingerprint([frozen]) == plan_reviewer_config_fingerprint([
            dataclasses.replace(slot, native_retrieval_override=True)])
