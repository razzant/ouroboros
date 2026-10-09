"""Review pool rows keep their stated delivery through the contributor wrapper."""

import json
import os
from types import SimpleNamespace

import pytest

from scripts import run_external_review as runner
from scripts.contributor_review_evidence import _compare_dispatch
from ouroboros.reviewer_slot_config import review_pool_slots
from tests.review_pool_rosters import pool_roster, pool_seat


@pytest.fixture
def configured(monkeypatch):
    """One review-eligible api row that reads the subject natively."""
    roster = json.loads(pool_roster(pool_seat("critic", "openrouter::openai/test", delivery="native", effort="high")))
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(roster))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    return roster


def test_freeze_keeps_reference_and_resolved_evidence(configured, monkeypatch):
    config = runner._resolved_review_config(profile="external_pr_readiness")
    frozen = runner._freeze_contributor_slots(config)

    pinned = json.loads(os.environ["OUROBOROS_SUBAGENTS"])["items"]
    assert [(row["subagent_id"], row["route"], row["effort"], row["delivery"]) for row in pinned] == [
        ("critic", {"kind": "api_model", "target_id": "openrouter::openai/test"}, "high", "native")]
    assert frozen["pool_slots"] == [{"slot_id": "critic", "effort": "high", "delivery": "native",
                                     "route": {"kind": "api_chat", "target_id": "openrouter::openai/test"}}]
    assert review_pool_slots()[0].native_retrieval
    assert not runner._diff_size_refusal(SimpleNamespace(contributor=True), frozen, 100, 1)
    assert runner._configured_openrouter_models(frozen) == ["openai/test"]
    # The evidence fingerprint binds what the pool row resolved to.
    configured["items"][0]["route"]["target_id"] = "openrouter::openai/changed"
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(configured))
    assert runner._slot_plan_sha256(runner._resolved_review_config()) != frozen["slot_plan_sha256"]


def test_native_retrieval_keeps_run_cap_and_probe(configured, monkeypatch):
    isolated = []
    monkeypatch.setattr(runner, "isolate_review_data",
                        lambda **kwargs: isolated.append(kwargs) or {"run_cap_usd": 1.0, "review_data_root": "/d"})
    monkeypatch.setattr(runner, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(runner, "_contributor_proposal", lambda *a: {"base_sha": "base"})
    probes = []
    monkeypatch.setattr(runner, "_select_healthy_openrouter_key", lambda **kw: probes.append(kw))
    args = SimpleNamespace(contributor=True, base_ref="base", head_ref="head",
                           drive_root="", run_cap_usd="1", attach_host_engine=False)
    monkeypatch.delenv("TOTAL_BUDGET", raising=False)
    _proposal, resolved = runner._prepare_review_configuration(args)
    assert [call["run_cap"] for call in isolated] == ["1"]  # isolated before settings load
    assert resolved["data_isolation"]["run_cap_usd"] == 1.0
    assert probes == [{"required": True, "probe_all_models": True, "probe_models": ["openai/test"]}]


@pytest.mark.parametrize("model", ["openrouter::openai/test", "openrouter::openai/other"])
def test_execution_receipt_checks_the_dispatched_route(configured, model):
    row = runner._resolved_review_config()["pool_slots"][0]
    mismatches = []
    receipt = _compare_dispatch(surface="pool", slot_id="critic", row=row, mismatches=mismatches, dispatched_slot={
        "route": "api_chat", "model": model, "effort": "high",
    })
    drifted = model != "openrouter::openai/test"
    assert bool(mismatches) == drifted
    if drifted:
        assert mismatches == ["dispatch_model_mismatch:pool:critic:openrouter::openai/test->openrouter::openai/other"]
    assert receipt["model"] == model


@pytest.mark.parametrize("kind, delivery, refuses", [
    ("agent_session", "", False), ("api_model", "native", False), ("api_model", "", True),
])
def test_a_retrieving_pool_does_not_create_a_packet_size_refusal(monkeypatch, kind, delivery, refuses):
    """The diff cap binds packet recipients only: a pool whose every seat reads the
    subject itself (sessions, natively retrieving api rows) has no packet to cap."""
    target = "codex=test" if kind == "agent_session" else "openrouter::openai/test"
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", pool_roster(
        pool_seat("r1", target, kind=kind, delivery=delivery or "packet"),
        pool_seat("r2", "openrouter::openai/test", delivery="native")))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    frozen = runner._freeze_contributor_slots(runner._resolved_review_config())
    assert [slot.retrieves for slot in review_pool_slots()] == [not refuses, True]
    # I3-D1: the same rule on both lanes — the operator lane is not refused on its own.
    assert runner._diff_size_refusal(SimpleNamespace(contributor=True), frozen, 500001, 500000) is refuses
    assert runner._diff_size_refusal(SimpleNamespace(contributor=False), frozen, 500001, 500000) is refuses
