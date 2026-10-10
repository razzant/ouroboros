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
