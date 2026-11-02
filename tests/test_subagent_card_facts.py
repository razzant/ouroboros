"""Actual live and history frames feed the card's shared Python/JS fixture."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS, subagent_message_meta
from ouroboros.subagent_runtime import cognitive_route_billing_basis

CASES = json.loads((Path(__file__).parent / "fixtures/subagent_card_facts.json").read_text())
COORDINATOR_FIELDS = {"parent_cognitive_route", "coordinator_route", "coordinator_billing",
                      "coordinator_provenance", "use_local_model"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_producer_metadata_carries_the_card_facts(case):
    from ouroboros.supervisor_route import cognitive_route_provenance

    frame = subagent_message_meta(case["task"], task_id="child", event="running")
    expected = {key: "" for key in ("subagent_task_id", "root_task_id", "parent_task_id", "delegation_role",
        "subagent_role", "write_surface", "task_group_id", "model_lane", "effective_model_lane", "model",
        "executor_route", "effort_level", "effort_requested", "effort_source")}
    expected.update(subagent_task_id="child", delegation_role="subagent", subagent_event="running", **case["facts"])
    assert frame == expected
    assert set(frame) <= set(SUBAGENT_MESSAGE_FIELDS), "speech persistence must retain the same facts"
    assert subagent_message_meta({"model": "root-model"}) == {}
    assert (frame.get("coordinator_provenance") or cognitive_route_provenance(
        frame.get("parent_cognitive_route"))) == case["provenance"]


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "absent"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_history_keeps_only_recorded_card_facts_without_task_result(tmp_path, case, recorded):
    from ouroboros.gateway.contracts import ChatOutbound
    from ouroboros.gateway.history import _load_terminal_result, make_chat_history_endpoint

    frame = subagent_message_meta({**case["task"], "root_task_id": "parent", "parent_task_id": "parent"},
                                  task_id="child", event="running")
    facts = case["facts"] if recorded else {k: v for k, v in case["facts"].items() if k not in COORDINATOR_FIELDS}
    if not recorded:
        frame = {k: v for k, v in frame.items() if k not in COORDINATOR_FIELDS}
    assert set(frame) <= get_type_hints(ChatOutbound).keys()
    unrelated = {"raw_prompt": "private", "coordinator_account": "not a card fact", "chosen_by": "delegator",
                 "source_role": "fallback", "selected_supervisor_snapshot": {"handle": "unrelated"}}
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-10-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
            {"ts": "2026-10-22T00:00:01Z", "task_id": "child", "content": "child working", **frame, **unrelated}]
    (logs / "progress.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert _load_terminal_result(tmp_path, "child", {}) == {"_history_result_absent": True}
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    assert response.status_code == 200
    replay = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "child")
    assert replay["is_progress"] and {key: replay[key] for key in frame} == frame
    assert {key: replay[key] for key in facts} == facts, "the web fixture is the actual replayed card input"
    assert not unrelated.keys() & replay.keys(), "replay must not broaden its metadata whitelist"
    if not recorded:
        assert not COORDINATOR_FIELDS & replay.keys(), "replay cannot reconstruct absent facts"


@pytest.mark.parametrize("price", [None, .001])
@pytest.mark.parametrize("model,local,cost,billing", [
    ("openai/gpt-5", False, None, "unknown"),
    ("openai/gpt-5", False, .125, "measured"),
    ("future-provider/reviewer", False, 0.0, "measured"),
    ("local-model", True, None, "unknown"),
    ("claudexor::codex=gpt-5", False, None, "subscription"),
])
def test_billing_basis_ignores_price_cache_and_requires_observed_cost(monkeypatch, price, model, local, cost, billing):
    consulted = []
    def estimate(*_args, **_kwargs):
        consulted.append(True)
        return price
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", estimate)
    route, actual = cognitive_route_billing_basis(model, use_local=local, cost=cost)
    assert route == model + (" (local)" if local else "") and actual == billing
    ctx = SimpleNamespace(_accumulated_usage={"cost": cost, "_last_llm_call_meta": {"model": model}})
    assert cognitive_route_billing_basis(model, use_local=local, ctx=ctx) == (route, billing)
    assert not consulted, "a billing fact cannot depend on the cache or pricing service"


@pytest.mark.parametrize("last_meta,expected", [
    (None, "unknown"), ({}, "unknown"), ({"model": "other"}, "unknown"),
    ({"model": "openai/gpt-5"}, "measured"),
])
def test_accumulated_cost_needs_the_same_recorded_route(last_meta, expected):
    ctx = SimpleNamespace(_accumulated_usage={"cost": 0.125, "_last_llm_call_meta": last_meta})
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", expected)
