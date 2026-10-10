"""Processing inheritance belongs to the admitted role/actor, not its model ID."""

import copy
import json

import pytest

from ouroboros.configured_subagents import normalize_configured_subagents
from ouroboros.model_slots import (
    MODEL_PROCESSING_PREFERENCES_KEY as ROLES,
    PROCESSING_PREFERENCE_KEY as GLOBAL,
    normalize_model_role_options,
    normalize_processing_preference,
    resolve_processing_preference,
    task_processing_preference,
)
from ouroboros.reviewer_slot_config import review_pool_rows
from ouroboros.subagent_runtime import select_subagent_snapshot, validate_subagent_snapshot


def actor_settings(preference=None, **row_fields):
    row = {"subagent_id": "critic", "recommended_use": "Review the assigned work", "review_eligible": True,
           "route": {"kind": "api_model", "target_id": "openai::same-model"}, "effort": "high", **row_fields}
    if preference is not None:
        row["processing_preference"] = preference
    return {GLOBAL: "fast", "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]})}


def test_equal_model_roles_and_ordered_fallback_keep_independent_preferences():
    settings = {GLOBAL: "fast", ROLES: {"main": "standard", "light": "economy",
                "fallback": ["", "standard", "economy"]},
                "OUROBOROS_MODEL": "same", "OUROBOROS_MODEL_LIGHT": "same"}
    assert [resolve_processing_preference(role, settings=settings) for role in
            ("main", "light", "vision", "fallback:0", "fallback:1", "fallback:2")] == [
                "standard", "economy", "fast", "fast", "standard", "economy"]
    assert resolve_processing_preference("main", override="", settings=settings) == ""
    assert resolve_processing_preference("main", override="economy", settings=settings) == "economy"
    assert resolve_processing_preference(settings={}) == ""


def test_unconfigured_cross_model_fallback_keeps_the_acting_role_intent(monkeypatch):
    monkeypatch.setenv(GLOBAL, "fast")
    monkeypatch.setenv(ROLES, json.dumps({"main": "standard", "fallback": ["", "economy"]}))
    assert task_processing_preference({}, model_role="fallback:0") == "standard"
    assert task_processing_preference({}, model_role="fallback:1") == "economy"
    snapshot, _ = select_subagent_snapshot(actor_settings("economy"), subagent_id="critic")
    task = {"task_metadata": {"configured_subagent": snapshot}}
    assert task_processing_preference(task, model_role="subagent:critic") == "economy"
    assert task_processing_preference(task, model_role="fallback:0") == "economy"
    assert task_processing_preference(task, model_role="light") == "fast"


@pytest.mark.parametrize("value", [True, 1, {}, "urgent", "auto", "priority"])
def test_only_product_vocabulary_is_authored(value):
    with pytest.raises(ValueError):
        normalize_processing_preference(value)
    with pytest.raises(ValueError):
        normalize_model_role_options(ROLES, {"main": value})


@pytest.mark.parametrize("authored,effective", [(None, "fast"), ("", "fast"),
                                               ("standard", "standard"), ("economy", "economy")])
def test_actor_snapshot_and_reviewer_capture_the_same_preference(authored, effective, monkeypatch):
    settings = actor_settings(authored)
    before = copy.deepcopy(settings)
    snapshot, _ = select_subagent_snapshot(settings, subagent_id="critic")
    # The catalog row IS the reviewer row: the pool reads the one authored choice.
    (row,) = review_pool_rows(settings)
    assert snapshot["processing_preference"] == row.processing_preference == effective
    monkeypatch.setenv(GLOBAL, "economy")
    assert validate_subagent_snapshot(snapshot)["processing_preference"] == effective
    assert settings == before
    old_snapshot = dict(snapshot)
    old_snapshot.pop("processing_preference")
    assert validate_subagent_snapshot(old_snapshot)["processing_preference"] == ""
    assert "processing_preference" not in old_snapshot
    encoded = json.loads(normalize_configured_subagents(settings["OUROBOROS_SUBAGENTS"])[1])
    assert encoded["items"][0].get("processing_preference", "") == (authored or "")


def test_inline_standard_overrides_global_fast_without_changing_model_or_effort():
    (row,) = review_pool_rows(actor_settings("standard"))
    assert (row.target_id, row.effort, row.processing_preference) == (
        "openai::same-model", "high", "standard")
    (inherited,) = review_pool_rows(actor_settings())
    assert inherited.processing_preference == "fast"
    (under_economy,) = review_pool_rows({**actor_settings(), GLOBAL: "economy"})
    assert under_economy.processing_preference == "economy"

def test_reviewer_last_execution_preserves_mixed_observation_and_never_echoes_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import reviewer_slot_config as slots
    monkeypatch.setattr(slots, "_last_execution_path", lambda: tmp_path / "last.json")
    receipt = {"requested": "fast", "submitted": "fast", "submittedNative": "fast", "observed": "mixed", "observedNative": ["fast", "standard"], "reason": None, "source": "native_session"}
    row = SimpleNamespace(slot_id="one", model="m", processing_preference="fast", route=SimpleNamespace(value="agent_session"))
    actor = SimpleNamespace(slot_id="one", usage={"processing": receipt}, status="ok")
    slots.record_reviewer_slot_executions("review", [actor], {"one": row})
    assert slots.reviewer_slot_last_executions()["one"]["effective"]["processing"] == receipt
    actor.usage = {}
    slots.record_reviewer_slot_executions("review", [actor], {"one": row})
    current = slots.reviewer_slot_last_executions()["one"]
    assert current["requested"]["processing_preference"] == "fast"
    assert "processing" not in current["effective"]
