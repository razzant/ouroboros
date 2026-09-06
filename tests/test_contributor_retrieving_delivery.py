"""Configured reviewer references keep their native retrieval delivery."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_external_review as runner
from scripts.contributor_review_evidence import _compare_dispatch
from ouroboros.reviewer_slot_config import load_reviewer_slot_config


@pytest.fixture
def configured(monkeypatch):
    roster = {"enabled": True, "items": [{
        "subagent_id": "critic", "name": "Critic", "recommended_use": "review",
        "route": {"kind": "api_model", "target_id": "openrouter::openai/test"},
        "effort": "high",
    }]}
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(roster))
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t", "subagent_id": "critic"}],
        "scope": [{"slot_id": "s", "subagent_id": "critic"}],
    }))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    return roster


def test_freeze_keeps_reference_and_resolved_evidence(configured, monkeypatch):
    config = runner._resolved_review_config(profile="external_pr_readiness")
    frozen = runner._freeze_contributor_slots(config)
    import os

    wire = json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"])
    assert wire["triad"][0] == {"slot_id": "t", "subagent_id": "critic", "effort": "high"}
    assert "route" not in wire["scope"][0]
    assert frozen["triad_slots"][0]["route"]["kind"] == "api_chat"
    assert frozen["triad_slots"][0]["subagent_id"] == "critic"
    assert load_reviewer_slot_config().triad[0].native_retrieval
    assert not runner._diff_size_refusal(SimpleNamespace(contributor=True), frozen, 100, 1)
    assert runner._configured_openrouter_models(frozen) == ["openai/test"]
    # The evidence fingerprint also binds what the actor reference resolved to.
    configured["items"][0]["route"]["target_id"] = "openrouter::openai/changed"
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(configured))
    assert runner._slot_plan_sha256(runner._resolved_review_config()) != frozen["slot_plan_sha256"]


def test_native_retrieval_keeps_api_budget_and_probe(configured, monkeypatch):
    monkeypatch.setattr(runner, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(runner, "_contributor_snapshot", lambda *a: {"base_sha": "base"})
    probes = []
    monkeypatch.setattr(runner, "_select_healthy_openrouter_key", lambda **kw: probes.append(kw))
    args = SimpleNamespace(contributor=True, base_ref="base", head_ref="head")
    monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    with pytest.raises(RuntimeError, match="TOTAL_BUDGET is required"):
        runner._prepare_review_configuration(args)
    assert not probes
    monkeypatch.setenv("TOTAL_BUDGET", "1")
    runner._prepare_review_configuration(args)
    assert probes == [{"required": True, "probe_all_models": True, "probe_models": ["openai/test"]}]


@pytest.mark.parametrize("actor", ["critic", ""])
def test_execution_receipt_checks_actor_delivery(configured, actor):
    row = runner._resolved_review_config()["triad_slots"][0]
    mismatches = []
    receipt = _compare_dispatch(surface="triad", slot_id="t", row=row, mismatches=mismatches, dispatched_slot={
        "route": "api_chat", "model": "openrouter::openai/test", "effort": "high",
        "subagent_id": actor,
    })
    assert bool(mismatches) == (actor == "")
    if not actor:
        assert mismatches == ["dispatch_subagent_id_mismatch:triad:t:critic->absent"]
    assert receipt["subagent_id"] == (actor or None)
