"""Previous usable provider usage survives a restart and stays separate from current fit."""
from copy import deepcopy
import json

import pytest

from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.loop_messages import append_context_facts
from tests.test_loop_compaction import _ctx


@pytest.mark.parametrize("metrics, expected", [
    ({"prompt_tokens": 1000, "cached_tokens": 750, "cache_write_tokens": 250}, "750/1,000 tokens (75.0%)"),
    ({"prompt_tokens": 1000, "cached_tokens": 0, "cache_write_tokens": 1000}, "0/1,000 tokens (0.0%)"),
    ({"prompt_tokens": 1000}, "unknown"),
    ({"cached_tokens": 750}, "unknown"),
    ({"prompt_tokens": 0, "cached_tokens": 0}, "unknown"),
])
def test_usable_call_then_cold_next_route_facts(tmp_path, metrics, expected):
    class Provider:
        def chat(self, **_kwargs):
            return {"content": "usable"}, {**metrics, "completion_tokens": 2, "cost": 0,
                "provider": "openrouter", "resolved_model": "served-old-model"}

    usage = {}
    reply, _ = call_llm_with_retry(Provider(), [{"role": "user", "content": "task"}],
        "requested-model", None, "medium", 1, tmp_path / "logs", "cache-facts", 1, None,
        usage, "task", False)
    assert reply == {"content": "usable"}
    recorded = usage["_last_round_cache_usage"]
    assert recorded["model"] == "served-old-model"
    assert recorded["provider"] == "openrouter"
    assert recorded["cached_tokens"] == metrics.get("cached_tokens")
    # The normal continuation carrier serializes accumulated_usage. Unknown stays
    # null, and a different next route must not relabel the old response's facts.
    context = _ctx(tmp_path)
    context.context_fit_plan = None
    context.active_model = "different-next-model"
    context.round_idx = 2
    context.accumulated_usage.update(json.loads(json.dumps(usage)))
    before = deepcopy(context.messages)
    assert append_context_facts(context)
    assert context.messages[:-1] == before
    line = context.messages[-1]["content"]
    assert f"previous usable input cache {expected}" in line
    assert "[round 1, openrouter, served-old-model]" in line
    assert "model different-next-model" in line
    write = metrics.get("cache_write_tokens")
    assert (f"write {write:,} tokens" if write is not None else "write unknown") in line
    assert not append_context_facts(context)


def test_provider_refusal_keeps_previous_usable_cache_facts(tmp_path):
    class Refusal:
        def chat(self, **_kwargs):
            return {"content": ""}, {"provider_error": True, "provider_error_kind": "context_overflow",
                "provider_error_permanent": True, "prompt_tokens": 9, "cached_tokens": 1,
                "cost": 0, "provider": "openrouter"}

    previous = {"round": 1, "model": "earlier", "provider": "openai", "prompt_tokens": 100,
                "cached_tokens": 80, "cache_write_tokens": None}
    usage = {"_last_round_cache_usage": deepcopy(previous)}
    reply, _ = call_llm_with_retry(Refusal(), [{"role": "user", "content": "task"}],
        "model", None, "medium", 1, tmp_path / "logs", "cache-refusal", 2, None,
        usage, "task", False)
    assert reply is None
    assert usage["_last_round_cache_usage"] == previous


def _facts_line(tmp_path, usage):
    context = _ctx(tmp_path)
    context.context_fit_plan = None
    context.active_model = "next-model"
    context.round_idx = 3
    context.accumulated_usage.update(usage)
    assert append_context_facts(context)
    return context.messages[-1]["content"]


@pytest.mark.parametrize("reported_cache, expected", [(None, "unknown"), (0, "0/1,000 tokens (0.0%)")])
def test_the_providers_own_report_outranks_a_billing_zero(tmp_path, reported_cache, expected):
    """An adapter that synthesizes a billing 0 beside an unreported cache (GigaChat) is unknown, not 0%."""
    previous = {"round": 2, "provider": "gigachat", "model": "m", "prompt_tokens": 1000, "cached_tokens": 0,
                "reported_tokens": {"prompt_tokens": 1000, "cached_tokens": reported_cache,
                                    "completion_tokens": 5, "cache_write_tokens": None}}
    assert f"previous usable input cache {expected} [round 2, gigachat, m]" in _facts_line(
        tmp_path, {"_last_round_cache_usage": previous})


@pytest.mark.parametrize("change, clause", [
    ({"kind": "rewritten", "index": 174, "sanctioned_by": "superseded_skill_snapshot"},
     "; before that request sent message 174 changed (superseded_skill_snapshot)"),
    ({"kind": "rewritten", "index": 9, "sanctioned_by": None},
     "; before that request sent message 9 changed (origin not recorded)"),
    ({"kind": "rewritten", "index": 31, "sanctioned_by": "image_eviction"},
     "; before that request sent message 31 changed (image_eviction)"),
    ({"kind": "shrunk", "index": 10, "sanctioned_by": "compaction", "previous_messages": 10, "current_messages": 4},
     "; before that request sent history shrank from 10 to 4 messages (compaction)"),
    ({"kind": "tools_changed", "index": 0, "sanctioned_by": None}, "; before that request: tool set changed"),
    ({"kind": "system_rewritten", "index": 0, "sanctioned_by": None}, "; before that request: system view rebuilt"),
    ({"kind": "no_baseline", "index": 0, "sanctioned_by": None},
     "; before that request: resume without a send record"),
])
def test_a_change_of_sent_history_is_named_beside_the_cache_it_preceded(tmp_path, change, clause):
    previous = {"round": 2, "provider": "claudexor", "model": "codex", "prompt_tokens": 682_741,
                "cached_tokens": 438_912, "cache_write_tokens": 0}
    line = _facts_line(tmp_path, {"_last_round_cache_usage": previous, "_recent_prefix_breaks": [{**change, "round": 2}]})
    assert "previous usable input cache 438,912/682,741 tokens (64.3%) [round 2, claudexor, codex]; write 0 tokens" + clause in line


def test_the_cause_survives_the_next_rounds_own_change(tmp_path):
    """Signed-reasoning routes count this round's change before the facts line names the previous one.

    #2076 changed sent history before every request; one remembered change would always
    be the current round's, and the clause about the previous request would never print."""
    from ouroboros.transcript_prefix import count_history_change

    usage = {"_last_round_cache_usage": {"round": 2, "provider": "anthropic", "model": "m", "prompt_tokens": 1000,
                                         "cached_tokens": 40, "cache_write_tokens": 960}}
    for round_idx, index in ((1, 5), (2, 7), (2, 3), (3, 8)):
        count_history_change(usage, {"round": round_idx, "index": index, "kind": "rewritten",
                                     "sanctioned_by": "superseded_skill_snapshot"})
    assert usage["prompt_prefix_breaks"] == 4
    assert [(item["round"], item["index"]) for item in usage["_recent_prefix_breaks"]] == [(2, 7), (3, 8)]
    assert "; before that request sent message 7 changed (superseded_skill_snapshot)" in _facts_line(tmp_path, usage)


def test_no_clause_without_a_change_before_that_request(tmp_path):
    previous = {"round": 2, "provider": "claudexor", "model": "codex", "prompt_tokens": 1000,
                "cached_tokens": 997, "cache_write_tokens": 0}
    earlier = {"round": 1, "kind": "rewritten", "index": 3, "sanctioned_by": "compaction"}
    later = {"round": 3, "kind": "rewritten", "index": 4, "sanctioned_by": "compaction"}
    for usage in ({"_last_round_cache_usage": previous},
                  {"_last_round_cache_usage": previous, "_recent_prefix_breaks": [earlier, later]}):
        line = _facts_line(tmp_path, usage)
        assert "write 0 tokens |" in line and "before that request" not in line


@pytest.mark.parametrize("kind,target", [("api_model", "openai/gpt-fixture"), ("agent_session", "codex=gpt-fixture")])
@pytest.mark.parametrize("input_sources", ["shared", "declared"])
def test_context_cache_facts(tmp_path, monkeypatch, kind, target, input_sources):
    from ouroboros import maintenance
    from ouroboros.context import _capture_context_core
    from ouroboros.context_fit import _render_context_system_content
    from ouroboros.memory import Memory
    from ouroboros.subagent_history import execution_identity
    from ouroboros.subagent_metrics import publish
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_available_subagents_catalog import _context_env, _row, _settings

    env = _context_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    settings = _settings(_row("worker", kind=kind, target=target, recommendation="Use for implementation."))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: settings)
    task = {"id": "task-one", "type": "task", "text": "work", "delegation_role": "subagent",
        "input_sources": input_sources, "configured_subagent": {
        "selected_subagent_id": "worker", "route": {"kind": kind, "target_id": target},
        **({"access": "workspace_write"} if kind == "agent_session" else {})}}
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)

    def capture():
        core = _capture_context_core(env, Memory(drive_root=tmp_path), task, None, None)
        return core, _render_context_system_content(env, core, mode="max"), json.dumps(registry.schemas(), sort_keys=True)

    before, before_blocks, before_tools = capture()
    publish(tmp_path, {"row_id": "worker", "row_identity": execution_identity(task["configured_subagent"]),
        "purpose": "execution", "producer_id": "fixture", "task_id": "task-one", "dispatched": True,
        "terminal": False, "answer_received": None, "usable_verdict": None})
    pending, _, _ = capture()
    assert '"stream":"pending"' in pending.dynamic_text
    assert maintenance.run_job(tmp_path, "subagent_metrics")["state"] == "done"
    after, after_blocks, after_tools = capture()
    assert after.semi_stable_text.encode() == before.semi_stable_text.encode()
    assert json.dumps(after_blocks[:-1], sort_keys=True).encode() == json.dumps(before_blocks[:-1], sort_keys=True).encode()
    assert after_tools.encode() == before_tools.encode()
    assert "Subagent measurements" not in before.dynamic_text
    assert '"pending":1' in after.dynamic_text
    assert "Subagent measurements" not in after.semi_stable_text
    assert "Subagent measurements" in after_blocks[-1]["text"]

    other_task = {**task, "input_sources": "shared" if input_sources == "declared" else "declared"}
    other = _capture_context_core(env, Memory(drive_root=tmp_path), other_task, None, None)
    markers = ("An Agent-session row", "Subagent measurements")
    facts = lambda core: [line for line in core.dynamic_text.splitlines() if line.startswith(markers)]
    assert facts(after) == facts(other)
    assert len(facts(after)) == (2 if kind == "agent_session" else 1)
