"""Tests for supervisor/events.py _handle_llm_usage event persistence."""

import json


def test_llm_usage_writes_cached_tokens_and_cache_write_tokens(tmp_path):
    from supervisor import events as ev_module

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        def update_budget_from_usage(self, usage):
            self.last_usage = usage

    evt = {
        "type": "llm_usage",
        "model": "anthropic/claude-sonnet-4.6",
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 300,
            "cost": 0.01,
            "cached_tokens": 1200,
            "cache_write_tokens": 400,
            "prompt_cache_ttl": "default",
        },
        "category": "compaction",
        "provider": "openrouter",
        "source": "loop",
        "model_category": "light",
        "api_key_type": "openrouter",
        "cost_estimated": False,
        "task_id": "task-1",
        "root_task_id": "root-1",
        "parent_task_id": "parent-1",
        "delegation_role": "subagent",
    }
    ctx = FakeCtx()
    ev_module._handle_llm_usage(evt, ctx)

    events_file = tmp_path / "logs" / "events.jsonl"
    written = json.loads(events_file.read_text(encoding="utf-8").strip())
    assert written.get("cached_tokens") == 1200
    assert written.get("cache_write_tokens") == 400
    assert written.get("prompt_cache_ttl") == "default"
    assert written.get("category") == "compaction"
    assert written.get("provider") == "openrouter"
    assert written.get("source") == "loop"
    assert written.get("model_category") == "light"
    assert written.get("api_key_type") == "openrouter"
    assert written.get("cost_estimated") is False
    assert written.get("task_id") == "task-1"
    assert written.get("root_task_id") == "root-1"
    assert written.get("parent_task_id") == "parent-1"
    assert written.get("delegation_role") == "subagent"
    assert ctx.last_usage["cached_tokens"] == 1200
    assert ctx.last_usage["prompt_cache_ttl"] == "default"


def test_llm_usage_persists_reasoning_effort_projection(tmp_path):
    from supervisor import events as ev_module

    (tmp_path / "logs").mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            self.last_usage = usage

    note = {"requested": "medium", "applied": "high",
            "reason": "provider_wire_mapping", "model": "deepseek-v4-flash"}
    ev_module._handle_llm_usage(
        {"type": "llm_usage", "task_id": "t", "usage": {"prompt_tokens": 3, "reasoning_effort_clamped": note}},
        FakeCtx(),
    )
    written = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8"))
    assert written["reasoning_effort_clamped"] == note


def test_llm_usage_preserves_unknown_cost_as_null(tmp_path):
    from supervisor import events as ev_module

    (tmp_path / "logs").mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            self.last_usage = usage

    ctx = FakeCtx()
    ev_module._handle_llm_usage(
        {"type": "llm_usage", "task_id": "unknown", "usage": {"prompt_tokens": 3}},
        ctx,
    )
    written = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8"))
    assert written["cost"] is None
    assert written["cost_known"] is False
    assert ctx.last_usage["cost"] is None


def test_cost_breakdown_aggregates_cache_tokens_and_ttl(tmp_path):
    import asyncio
    import json
    from ouroboros.gateway.history import make_cost_breakdown_endpoint

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "events.jsonl").write_text(
        "\n".join([
            json.dumps({
                "type": "llm_usage",
                "model": "google/gemini-3.5-flash",
                "api_key_type": "openrouter",
                "model_category": "light",
                "category": "task",
                "cost": 0.25,
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cached_tokens": 600,
                "cache_write_tokens": 200,
                "prompt_cache_ttl": "default",
            }),
            json.dumps({
                "type": "llm_usage",
                "model": "malformed/model",
                "cost": 0.10,
                "cached_tokens": "n/a",
            }),
            json.dumps({
                "type": "llm_usage",
                "model": "nan/model",
                "cost": "NaN",
                "prompt_tokens": 50,
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    response = asyncio.run(make_cost_breakdown_endpoint(tmp_path)(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["total_cost"] == 0.35
    assert payload["total_prompt_tokens"] == 1050
    assert payload["total_cached_tokens"] == 600
    assert payload["total_cache_write_tokens"] == 200
    assert payload["prompt_cache_ttls"] == {"default": 1}
    by_model = payload["by_model"]["google/gemini-3.5-flash"]
    assert by_model["cached_tokens"] == 600
    assert by_model["cache_write_tokens"] == 200
    assert by_model["prompt_cache_ttls"] == {"default": 1}
    assert "malformed/model" in payload["by_model"]
    assert payload["by_model"]["nan/model"]["cost"] == 0.0


def test_task_metrics_are_persisted_and_forwarded_to_live_logs(tmp_path):
    from supervisor import events as ev_module

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    pushed = []

    class FakeBridge:
        def push_log(self, payload):
            pushed.append(payload)

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        bridge = FakeBridge()

        @staticmethod
        def append_jsonl(path, payload):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    evt = {
        "ts": "2026-03-31T10:11:12Z",
        "task_id": "task-99",
        "task_type": "task",
        "duration_sec": 3.14159,
        "tool_calls": 4,
        "tool_errors": 1,
    }
    ev_module._handle_task_metrics(evt, FakeCtx())

    written = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").strip())
    assert written["type"] == "task_metrics_event"
    assert written["task_id"] == "task-99"
    assert written["tool_calls"] == 4
    assert written["duration_sec"] == 3.142
    assert pushed[0]["task_id"] == "task-99"
    assert pushed[0]["tool_errors"] == 1


def test_llm_usage_serializer_carries_web_search_sources():
    """The llm_usage serializer must persist web_search_sources (GAIA
    campaign contract). Moved here from tests/test_devtools_benchmarks.py:
    after the D08 split the serializer lives with its budget family, and
    this suite owns the llm_usage event surface."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "supervisor" / "events_budget.py").read_text(encoding="utf-8")
    assert "web_search_sources" in src
