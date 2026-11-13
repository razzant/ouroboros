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
    ({"kind": "shrunk", "index": 10, "sanctioned_by": "compaction", "previous_messages": 10, "current_messages": 4},
     "; before that request sent history shrank from 10 to 4 messages (compaction)"),
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


@pytest.mark.parametrize("call, money", [
    ({"provider": "openrouter", "cost": 0.2154924, "cost_final": True}, "; write 4,625 tokens; cost $0.2155 reported |"),
    ({"provider": "openrouter", "cost": 0.2154924, "cost_final": False}, "; write 4,625 tokens; cost $0.2155 estimated |"),
    # An estimated zero (nothing to price it by) is no proven zero: unknown.
    ({"provider": "openrouter", "cost": 0.0, "cost_final": False}, "; write 4,625 tokens; cost unknown (not in known spend) |"),
    # A proven zero is a number; a charge below the printed precision is never that zero.
    ({"provider": "openrouter", "cost": 0.0, "cost_final": True}, "; write 4,625 tokens; cost $0.0000 reported |"),
    ({"provider": "openrouter", "cost": 0.00003, "cost_final": True}, "; write 4,625 tokens; cost <$0.0001 reported |"),
    # The subscription engine's own cash, when it states one, is a number like any other.
    ({"provider": "claudexor", "cost": 0.5, "cost_final": True}, "; write 4,625 tokens; cost $0.5000 reported |"),
    # An API call without a price: unknown is said, and why the money facts beside it do not move.
    ({"provider": "openai"}, "; write 4,625 tokens; cost unknown (not in known spend) |"),
    # The subscription engine and the local lane state no cash: nothing is said, never "unknown" or a zero.
    ({"provider": "claudexor", "billing": "unknown"}, "; write 4,625 tokens |"),
    ({"provider": "local", "cost": 0.0, "cost_final": True}, "; write 4,625 tokens |"),
])
def test_the_previous_calls_own_money_stands_beside_its_cache(tmp_path, call, money):
    previous = {"round": 2, "model": "m", "prompt_tokens": 567_946, "cached_tokens": 562_502,
                "cache_write_tokens": 4_625, **call}
    line = _facts_line(tmp_path, {"_last_round_cache_usage": previous})
    assert f"[round 2, {call['provider']}, m]{money}" in line
    assert "tariff" not in line and "$0.00 " not in line


@pytest.mark.parametrize("applied, bracket", [
    ("1h", "[round 2, anthropic, m, cache TTL 1h]"), ("5m", "[round 2, anthropic, m, cache TTL 5m]"),
    # A bare marker names no tier and an unmarked send none at all: the bracket stays as it was.
    ("default", "[round 2, anthropic, m]"), ("", "[round 2, anthropic, m]"),
])
def test_only_a_stamped_cache_ttl_is_named(tmp_path, applied, bracket):
    previous = {"round": 2, "provider": "anthropic", "model": "m", "prompt_tokens": 1000, "cached_tokens": 900,
                "cache_write_tokens": 100, "cost": 0.01, "cost_final": True}
    line = _facts_line(tmp_path, {"_last_round_cache_usage": previous, "_last_prompt_cache_ttl": applied})
    assert bracket in line and ("cache TTL" in line) == (applied in ("1h", "5m"))
    assert "expire" not in line and "cold" not in line  # the fact, no forecast


@pytest.mark.parametrize(("input_sources", "kind", "target"), [
    ("shared", "api_model", "openai/gpt-fixture"),
    ("declared", "agent_session", "codex=gpt-fixture"),
])
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


def test_history_changes_carry_what_the_request_after_them_read_uncached(tmp_path):
    """C2 S3 (06.11.2026): the facts name this task's changes by origin with the provider's uncached
    input of the request that first carried each; an unanswered one stays unmeasured, never estimated."""
    from ouroboros.transcript_prefix import count_history_change, settle_rewrite_reread

    usage = {}
    count_history_change(usage, {"round": 4, "index": 9, "kind": "rewritten", "sanctioned_by": "compaction"})
    count_history_change(usage, {"round": 5, "index": 0, "kind": "system_rewritten", "sanctioned_by": None})
    usage["_last_round_cache_usage"] = {"round": 5, "provider": "claudexor", "model": "codex",
                                        "prompt_tokens": 700_000, "cached_tokens": 422_000}
    settle_rewrite_reread(usage)
    settle_rewrite_reread(usage)  # one answer settles once
    line = _facts_line(tmp_path, usage)
    assert ("history changes this task: compaction 1 (requests carrying 1 of them read 278,000 uncached); "
            "system view rebuilt, origin unknown 1 |") in line
    count_history_change(usage, {"round": 6, "index": 3, "kind": "rewritten", "sanctioned_by": "compaction"})
    usage["_last_round_cache_usage"] = {"round": 7, "provider": "claudexor", "model": "codex", "prompt_tokens": 9}
    settle_rewrite_reread(usage)  # no cache count reported: the change is counted, its re-read unknown
    assert "compaction 2 (requests carrying 1 of them read 278,000 uncached)" in _facts_line(tmp_path, usage)


def test_the_serving_account_its_move_and_the_last_response_age_are_named(tmp_path, monkeypatch):
    from ouroboros.transcript_prefix import served_account_fields

    first = {"round": 2, "provider": "claudexor", "model": "codex", "prompt_tokens": 1000, "cached_tokens": 990,
             **served_account_fields({"claudexor": {"route": {"credentialProfileId": "acct-a"}}}, {}, "claudexor", "codex")}
    moved = {"round": 3, "provider": "claudexor", "model": "codex", "prompt_tokens": 1000, "cached_tokens": 300,
             "finished_at": 1_000.0,
             **served_account_fields({"claudexor": {"route": {"credentialProfileId": "acct-b"}}}, first, "claudexor", "codex")}
    assert "[round 2, claudexor, codex, served by account acct-a]" in _facts_line(tmp_path, {"_last_round_cache_usage": first})
    monkeypatch.setattr("time.time", lambda: 1_000.0 + 7 * 60)
    line = _facts_line(tmp_path, {"_last_round_cache_usage": moved})
    assert "[round 3, claudexor, codex, served by account acct-b (moved from acct-a)]" in line
    assert "| last response 7 min ago |" in line
    other_route = served_account_fields({"claudexor": {"route": {"credentialProfileId": "acct-b"}}}, first, "claudexor", "sol")
    assert other_route == {"served_account": "acct-b", "previous_served_account": None}  # no move across models
    assert served_account_fields({}, first, "claudexor", "codex") == {}  # no engine report, no account named


def test_live_images_and_a_size_pass_rows_are_facts_of_their_round(tmp_path):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.transcript_prefix import SIZE_PASS_ROWS_ATTR, observe_send, size_pass_host_rows

    context = _ctx(tmp_path)
    context.context_fit_plan = None
    context.round_idx = 3
    observe_send(context.tools._ctx, list(context.messages), round_idx=2)  # the last request
    image = {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
    context.messages += [{"role": "tool", "tool_call_id": "t", "content": "result"},
                         {"role": "user", "content": [image, image]},
                         {"role": "user", HOST_CONTEXT_KIND_KEY: "children_reminder", "content": "x" * 400}]
    # the size pass keeps this record once it applies: the host row after the results the last request did not carry
    setattr(context.tools._ctx, SIZE_PASS_ROWS_ATTR, size_pass_host_rows(context))
    assert append_context_facts(context)
    line = context.messages[-1]["content"]
    assert "| images in view 2 (past 10 the oldest are replaced down to 5) |" in line
    assert "before this round's size pass, host rows added after the results' first show: children_reminder ~" in line
    context.round_idx = 4
    assert append_context_facts(context)
    assert "size pass" not in context.messages[-1]["content"]  # a fact of its own round only
