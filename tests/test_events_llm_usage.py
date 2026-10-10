"""Tests for supervisor/events.py _handle_llm_usage event persistence."""

import json

import pytest


def test_llm_usage_writes_cached_tokens_and_cache_write_tokens(tmp_path):
    from supervisor import events as ev_module

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            pytest.fail("the writer runs once per loop turn, never per event")

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
    assert written["projection_update_status"] == "deferred"
    assert ctx.budget_projection_dirty is True


def test_llm_usage_persists_reasoning_effort_projection(tmp_path):
    from supervisor import events as ev_module

    (tmp_path / "logs").mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            pytest.fail("the writer runs once per loop turn, never per event")

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
            pytest.fail("the writer runs once per loop turn, never per event")

    ctx = FakeCtx()
    ev_module._handle_llm_usage(
        {"type": "llm_usage", "task_id": "unknown", "usage": {"prompt_tokens": 3}},
        ctx,
    )
    written = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8"))
    assert written["cost"] is None
    assert written["cost_known"] is False
    assert ctx.budget_projection_dirty is True


def test_llm_usage_defers_the_projection_write_and_keeps_paid_usage(tmp_path):
    """The event never pays the projection write: it marks the loop context dirty and
    records ``deferred``; the paid usage row itself is retained either way."""
    from supervisor import events as ev_module
    (tmp_path / "logs").mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            pytest.fail("the writer runs once per loop turn, never per event")

    ctx = FakeCtx()
    for _ in range(3):
        ev_module._handle_llm_usage(
            {"type": "llm_usage", "task_id": "paid", "usage": {"prompt_tokens": 4, "cost": 0.75}},
            ctx,
        )
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["projection_update_status"] for row in rows] == ["deferred"] * 3
    assert all(row["cost"] == 0.75 for row in rows)
    assert ctx.budget_projection_dirty is True


def test_llm_usage_real_corrupt_ledger_keeps_the_projection_dirty_and_paid_event_survives(tmp_path):
    from ouroboros.server_liveness import flush_budget_projection
    from supervisor import events as ev_module
    from supervisor import state
    from ouroboros.usage_ledger import LEDGER_REL

    (tmp_path / "logs").mkdir()
    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({"spent_usd": 1.25})
    ledger = tmp_path / LEDGER_REL
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{broken}\n{}\n", encoding="utf-8")

    class Ctx:
        DRIVE_ROOT = tmp_path

        @staticmethod
        def update_budget_from_usage(usage):
            return state.update_budget_from_usage(usage)

    ctx = Ctx()
    ev_module._handle_llm_usage(
        {"type": "llm_usage", "task_id": "paid-real", "usage": {"prompt_tokens": 2, "cost": 0.5}},
        ctx,
    )
    written = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8"))
    assert written["projection_update_status"] == "deferred"
    assert written["cost"] == 0.5
    flush_budget_projection(ctx)  # the one write of the turn: refused on a corrupt ledger
    assert state.load_state()["spent_usd"] == 1.25
    assert ctx.budget_projection_dirty is True and ctx.budget_projection_retry_at > 0


_PRICED_CACHE_USAGE = {
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
}


def test_cost_breakdown_aggregates_cache_tokens_and_ttl(tmp_path):
    import asyncio
    import json
    from ouroboros.gateway.history import make_cost_breakdown_endpoint

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "events.jsonl").write_text(
        "\n".join([
            json.dumps(_PRICED_CACHE_USAGE),
            json.dumps({
                "type": "llm_usage",
                "model": "malformed/model",
                "cost": 0.10,
                "cached_tokens": "n/a",
            }),
            # An honestly unknown price: its tokens count, its dollars stay undisclosed.
            json.dumps({
                "type": "llm_usage",
                "model": "unpriced/model",
                "cost": None,
                "prompt_tokens": 50,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    from ouroboros import usage_store

    usage_store.migrate_from_journal(tmp_path)  # the server's lifecycle import of legacy telemetry
    response = asyncio.run(make_cost_breakdown_endpoint(tmp_path)(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["total_cost"] == 0.35
    assert payload["total_prompt_tokens"] == 1050
    assert payload["total_cached_tokens"] == 600
    assert payload["total_cache_write_tokens"] == 200
    assert payload["prompt_cache_ttls"] == {"default": 1}
    by_model = payload["by_model"]["google/gemini-3.5-flash"]
    assert by_model["cached_tokens"] == 600
    assert by_model["cache_write_tokens"] == 200
    assert by_model["prompt_cache_ttls"] == {"default": 1}
    assert by_model["cost_final"] is True
    assert "malformed/model" in payload["by_model"]
    unpriced = payload["by_model"]["unpriced/model"]
    assert unpriced["cost"] == 0.0
    assert unpriced["unknown_unmetered"] == 1
    assert unpriced["cost_final"] is False
    accounting = payload["accounting"]
    assert accounting["available"] is True
    assert accounting["settled_usd"] == 0.35
    assert accounting["unknown_unmetered"] == 1
    assert accounting["non_final_rows"] == 1
    assert accounting["cost_final"] is False


@pytest.mark.parametrize("encoding", ["string", "json_token"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_cost_breakdown_refuses_nonfinite_legacy_cost_without_import(tmp_path, caplog, literal, encoding):
    """Nonfinite legacy money is an integrity failure: the store's one-time import
    refuses it (no completed import that would hide it, no store), and the real
    gateway then answers 503, never a fabricated $0 total."""
    import asyncio
    from ouroboros import usage_store
    from ouroboros.gateway.history import make_cost_breakdown_endpoint
    from ouroboros.usage_journal import IMPORT_REL
    from ouroboros.usage_ledger import LEDGER_REL, QUARANTINE_REL, UsageLockUnavailable, UsageNonFiniteMoney

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    events_path = logs_dir / "events.jsonl"
    # ``json.dumps`` of a float writes the bare NaN/Infinity token a Python writer would emit.
    cost = literal if encoding == "string" else float(literal)
    source = (
        json.dumps(_PRICED_CACHE_USAGE) + "\n"
        + json.dumps({"type": "llm_usage", "model": "nonfinite/model", "cost": cost, "prompt_tokens": 50}) + "\n"
    ).encode("utf-8")
    events_path.write_bytes(source)
    endpoint = make_cost_breakdown_endpoint(tmp_path)

    for _ in range(2):  # no watermark was written, so a retry refuses again
        with pytest.raises(UsageNonFiniteMoney):
            usage_store.migrate_from_journal(tmp_path)
        caplog.clear()
        response = asyncio.run(endpoint(None))
        payload = json.loads(response.body.decode("utf-8"))

        assert response.status_code == 503
        assert "total_cost" not in payload
        assert payload["accounting"] == {
            "available": False,
            "authority": "physical_attempt_ledger",
            "cost_final": False,
            "error_code": "ledger_unavailable",
        }
        refused = [record for record in caplog.records if record.exc_info]
        # A display never re-runs the refused history import: it reports the store unavailable.
        assert [type(record.exc_info[1]) for record in refused] == [UsageLockUnavailable]

    assert events_path.read_bytes() == source
    archived = list((tmp_path / "archive" / "usage_import").glob("*/events.jsonl"))
    assert [path.read_bytes() for path in archived] == [source]
    assert not (tmp_path / IMPORT_REL).exists()
    assert not (tmp_path / QUARANTINE_REL).exists()
    assert not (tmp_path / usage_store.STORE_REL).exists()
    assert not (tmp_path / LEDGER_REL).exists()


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


def test_effort_facts_with_legacy_option_status_survive_logs_without_notices(tmp_path):
    """An old option_status stays in both logs without creating a notice."""
    from types import SimpleNamespace
    from supervisor import events

    (tmp_path / "logs").mkdir()
    frames = []
    facts = {
        "effort": {"requested": "ultra", "sent": {"reasoning_effort": "max"},
                   "sent_state": "explicit", "reported": None, "report_source": None},
        "request_wire": {"requested_effort": "ultra", "applied_effort": "max",
                         "applied_effort_source": "sent_candidate"},
        "claudexor": {"requested_options": {"reasoningEffort": "ultra"},
                      "applied_options": {}, "option_status": {"reasoningEffort": "unknown"}},
    }
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=frames.append))
    events._handle_llm_usage({"type": "llm_usage", "task_id": "t", "usage": facts}, ctx)
    written = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert {key: written[key] for key in facts} == facts
    assert frames == [written]
    assert "toast_once" not in written and "task_incident" not in written


def test_llm_usage_keeps_review_wave_and_slot_ids_only_when_the_review_scope_set_them(tmp_path):
    """#807: the durable row keeps the ids the review emitter stamped and invents none."""
    from supervisor import events as ev_module

    (tmp_path / "logs").mkdir()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        def update_budget_from_usage(self, usage):
            pytest.fail("the writer runs once per loop turn, never per event")

    review = {"review_skill": "skill-a", "review_wave_id": "wave-1", "review_slot_id": "slot-2"}
    base = {"type": "llm_usage", "task_id": "t", "category": "review", "usage": {"prompt_tokens": 3}}
    ev_module._handle_llm_usage({**base, **review}, FakeCtx())
    ev_module._handle_llm_usage({**base, "review_slot_id": ""}, FakeCtx())
    first, second = [json.loads(line) for line in
                     (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {key: first[key] for key in review} == review
    assert not any(key in second for key in review)
