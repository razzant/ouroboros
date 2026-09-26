"""Exact structured reviewer-route and effort-authority regressions."""

import json

import pytest

from ouroboros.reviewer_slot_config import (
    REVIEWER_SLOTS_ENV,
    commit_triad_delivery,
    load_reviewer_slot_config,
    parse_reviewer_slots,
    structured_scope_review_slots,
)


def _payload() -> dict:
    return {
        "triad": [
            {
                "slot_id": "triad-route",
                "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
            },
        ],
        "scope": [
            {
                "slot_id": "scope-route",
                "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol"},
            },
        ],
        "advisory": {"enabled": True, "route": {"kind": "api", "target_id": ""}},
    }


@pytest.mark.parametrize("target", ["off", "OFF", "=malformed", ":high"])
@pytest.mark.parametrize("surface", ["triad", "scope", "advisory"])
def test_structured_session_target_must_name_a_concrete_harness(target, surface):
    payload = _payload()
    if surface == "advisory":
        payload["advisory"] = {
            "enabled": True,
            "route": {"kind": "agent_session", "target_id": target},
        }
    else:
        payload[surface][0]["route"] = {
            "kind": "agent_session",
            "target_id": target,
        }

    with pytest.raises(ValueError, match="does not name a concrete harness route"):
        parse_reviewer_slots(json.dumps(payload))


def test_disabled_advisory_allows_empty_session_but_not_persisted_junk():
    payload = _payload()
    payload["advisory"] = {
        "enabled": False,
        "route": {"kind": "agent_session", "target_id": ""},
    }
    advisory = parse_reviewer_slots(json.dumps(payload)).advisory
    assert advisory.enabled is False and advisory.target_id == ""

    payload["advisory"]["route"]["target_id"] = "off"
    with pytest.raises(ValueError, match="does not name a concrete harness route"):
        parse_reviewer_slots(json.dumps(payload))


def test_settings_save_refuses_unparseable_session_target_before_persistence():
    from starlette.requests import Request

    from ouroboros.gateway.settings import _api_settings_post_locked

    payload = _payload()
    payload["triad"][0]["route"]["target_id"] = "=malformed"
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/settings",
        "headers": [],
        "query_string": b"",
    })
    response = _api_settings_post_locked(
        request,
        {REVIEWER_SLOTS_ENV: json.dumps(payload)},
    )
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body["saved"] is False
    assert "does not name a concrete harness route" in body["error"]


def test_malformed_advisory_target_never_consults_the_shared_route(monkeypatch):
    from ouroboros.tools import claude_advisory_review as advisory

    payload = _payload()
    payload["advisory"] = {
        "enabled": True,
        "route": {"kind": "agent_session", "target_id": "off"},
    }
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_REVIEW_SESSION_ROUTE", "codex=gpt-5.6-sol:high")

    with pytest.raises(ValueError, match="does not name a concrete harness route"):
        advisory.advisory_gate_unavailability_reason()


def test_compound_session_effort_precedes_surface_defaults(monkeypatch):
    from ouroboros.tools import plan_review_runtime

    payload = _payload()
    payload["triad"] = [
        {
            "slot_id": "cursor-row",
            "route": {
                "kind": "agent_session",
                "target_id": "cursor=cursor-grok-4.6-xhigh-fast",
            },
        },
        {
            "slot_id": "plain-row",
            "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
        },
    ]
    payload["scope"] = [
        {
            "slot_id": "agy-row",
            "route": {
                "kind": "agent_session",
                "target_id": "agy=gemini-3.1-pro-max-fast",
            },
        },
    ]
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "medium")

    config = load_reviewer_slot_config()
    assert [row.effort for row in config.triad] == ["", ""]
    assert commit_triad_delivery()["efforts"] == ["xhigh", "low"]
    assert [slot.effort for slot in structured_scope_review_slots()] == ["max"]
    # The owner's review-effort setting reaches the plan panel exactly like the
    # commit triad (no plan-local constant overrides it any more).
    assert [slot.effort for slot in plan_review_runtime.plan_review_slots()] == ["xhigh", "low"]
    assert [slot.declared_effort for slot in plan_review_runtime.plan_review_slots()] == ["", ""]


def test_declared_plan_effort_outranks_row_pins_but_not_compound_slugs(monkeypatch):
    """The envelope's reviewer_effort is an ORDER for this plan: it outranks the
    owner's per-row pin; only a compound Cursor/Agy slug keeps its encoded effort. It
    travels as an argument of the plan builder alone, so the commit gate, scope,
    acceptance and skill-review identities are byte-identical before and after."""
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint
    from ouroboros.tools import plan_review_runtime
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    payload = _payload()
    payload["triad"] = [
        {"slot_id": "cursor-row", "route": {"kind": "agent_session", "target_id": "cursor=cursor-grok-4.6-xhigh-fast"}},
        {"slot_id": "plain-row", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}},
        {"slot_id": "pinned-row", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol"}, "effort": "low"},
    ]
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "medium")
    before = (commit_triad_delivery(), [s.effort for s in structured_scope_review_slots()],
              commit_review_contract_fingerprint(),
              skill_review_contract_fingerprint(["m"], delivery=commit_triad_delivery()))
    declared = plan_review_runtime.plan_review_slots(default_effort="max")
    assert [s.effort for s in declared] == ["xhigh", "max", "max"]  # the pinned `low` row runs the order
    assert [s.declared_effort for s in declared] == ["", "max", "max"]
    assert [s.effort for s in plan_review_runtime.plan_review_slots()] == ["xhigh", "medium", "low"]
    assert [s.declared_effort for s in plan_review_runtime.plan_review_slots()] == ["", "", ""]
    after = (commit_triad_delivery(), [s.effort for s in structured_scope_review_slots()],
             commit_review_contract_fingerprint(),
             skill_review_contract_fingerprint(["m"], delivery=commit_triad_delivery()))
    assert before == after and before[0]["efforts"] == ["xhigh", "medium", "low"]


def test_last_execution_projection_keeps_a_declared_effort_apart_from_the_row(tmp_path, monkeypatch):
    """«Выполняется как» must not show the agent's one-off panel strength as the
    row's saved configuration: requested.effort is the ROW's effort ('' when the
    declaration filled it) and the declaration rides its own field."""
    from types import SimpleNamespace

    from ouroboros import reviewer_slot_config
    from ouroboros.review_substrate import ReviewSlot

    monkeypatch.setattr(reviewer_slot_config, "_last_execution_path", lambda: tmp_path / "last.json")
    slots = {
        "declared": ReviewSlot(slot_id="declared", model="m/a", effort="max", declared_effort="max"),
        "own": ReviewSlot(slot_id="own", model="m/b", effort="low"),
    }
    actors = [SimpleNamespace(slot_id=sid, status="ok", usage={}) for sid in slots]
    reviewer_slot_config.record_reviewer_slot_executions("plan_review", actors, slots)
    last = reviewer_slot_config.reviewer_slot_last_executions()
    assert last["declared"]["requested"]["effort"] == "" and last["declared"]["requested"]["declared_effort"] == "max"
    assert last["own"]["requested"]["effort"] == "low" and "declared_effort" not in last["own"]["requested"]


def test_compound_effort_stabilizes_replay_identity_against_global_drift(monkeypatch):
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    payload = _payload()
    payload["triad"][0]["route"]["target_id"] = "cursor=cursor-grok-4.6-xhigh"
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    first = commit_triad_delivery()
    first_fp = skill_review_contract_fingerprint(
        first["models"], required_items=("manifest_schema",), delivery=first,
    )

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    same = commit_triad_delivery()
    same_fp = skill_review_contract_fingerprint(
        same["models"], required_items=("manifest_schema",), delivery=same,
    )
    assert first["efforts"] == same["efforts"] == ["xhigh"]
    assert first_fp == same_fp

    payload["triad"][0]["route"]["target_id"] = "cursor=cursor-grok-4.6-max"
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    changed = commit_triad_delivery()
    changed_fp = skill_review_contract_fingerprint(
        changed["models"], required_items=("manifest_schema",), delivery=changed,
    )
    assert changed["efforts"] == ["max"]
    assert changed_fp != first_fp


def test_compound_effort_stabilizes_commit_fingerprint_against_global_drift(
    monkeypatch,
):
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    payload = _payload()
    payload["triad"][0]["route"]["target_id"] = "cursor=cursor-grok-4.6-xhigh"
    payload["scope"][0]["route"] = {
        "kind": "agent_session",
        "target_id": "agy=gemini-3.1-pro-max-fast",
    }
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "medium")
    first = commit_review_contract_fingerprint()

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "low")
    assert commit_review_contract_fingerprint() == first

    payload["scope"][0]["route"]["target_id"] = "agy=gemini-3.1-pro-xhigh-fast"
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    assert commit_review_contract_fingerprint() != first
