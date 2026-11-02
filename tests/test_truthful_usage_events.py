"""#1578: diagnostic counters and session receipts retain their own evidence."""

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.event_usage import TOKEN_COUNTER_ALIASES, event_token_counts
from tests._review_session_route_shared import (
    FakeLLM, _agent_request, _agent_slot, _run_session_directly, _terminal_detail,
    fake_route as _fake_route_fixture,
    _owned_gateway_uses_each_test_transport as _transport_fixture,
)

fake_route = _fake_route_fixture
_owned_gateway_uses_each_test_transport = _transport_fixture
pytestmark = pytest.mark.serial
COUNTERS = tuple(TOKEN_COUNTER_ALIASES)


@pytest.mark.parametrize("usage,expected", [
    ({}, [None, None, None, None]),
    (dict.fromkeys(COUNTERS), [None, None, None, None]),
    (dict.fromkeys(COUNTERS, 0), [0, 0, 0, 0]),
    ({"prompt_tokens": 37, "completion_tokens": 0}, [37, 0, None, None]),
    ({"input_tokens": 19, "output_tokens": 8, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 2}, [19, 8, 0, 2]),
    ({"prompt_tokens": "bad", "completion_tokens": float("inf"),
      "cached_tokens": False, "cache_write_tokens": -1}, [None, None, None, None]),
])
def test_review_helper_and_supervisor_preserve_each_counter(tmp_path, usage, expected):
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage

    ctx = SimpleNamespace(task_id="review", pending_events=[])
    emit_review_usage(ctx, model="requested", usage=usage, source="review", session_id="sdk-session")
    event = ctx.pending_events[0]
    assert [event["usage"][key] for key in COUNTERS] == expected
    frames = []
    supervisor = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=frames.append))
    _handle_llm_usage(event, supervisor)
    persisted = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert [persisted[key] for key in COUNTERS] == expected
    assert persisted["session_id"] == "sdk-session"
    assert persisted["record_contract_version"] == 2
    assert frames == [persisted]


def test_review_receipt_and_provider_survive_helper_and_supervisor(tmp_path):
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage

    receipt = {"run_id": "run", "requested_route": "asked", "delegated_route": "actual",
               "observed_attempt": {"attempt_id": "a2", "harness_id": "actual", "model": "observed"}}
    ctx = SimpleNamespace(task_id="review", pending_events=[])
    usage = {"provider": "claudexor", "claudexor": receipt, "prompt_tokens": 0}
    emit_review_usage(ctx, model="vendor/requested", usage=usage, source="review")
    event = ctx.pending_events[0]
    assert event["provider"] == "claudexor"
    assert event["usage"]["claudexor"] == receipt
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path))
    persisted = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert persisted["claudexor"] == receipt and persisted["provider"] == "claudexor"
    receipt["observed_attempt"]["model"] = "mutated"
    assert event["usage"]["claudexor"]["observed_attempt"]["model"] == "observed"


def test_loop_usage_emitter_does_not_reintroduce_zero_fallback(tmp_path):
    from ouroboros.pricing import emit_llm_usage_event
    from supervisor.events_budget import _handle_llm_usage

    events = queue.Queue()
    emit_llm_usage_event(events, "t", "model", {"prompt_tokens": 5}, None)
    event = events.get_nowait()
    assert event_token_counts(event) == dict(zip(COUNTERS, [5, None, None, None]))
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path))
    row = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert row["completion_tokens"] is None


def test_explicit_null_counter_outranks_legacy_fallback():
    assert event_token_counts({"prompt_tokens": None}, {"prompt_tokens": 0})["prompt_tokens"] is None


@pytest.mark.parametrize("emitter", ["vision", "web_search"])
@pytest.mark.parametrize("tokens,expected", [
    ({}, [None] * 4),
    (dict.fromkeys(COUNTERS, 0), [0] * 4),
    ({"prompt_tokens": 17, "cache_write_tokens": 0}, [17, None, None, 0]),
])
def test_tool_producers_keep_unknown_and_partial_usage(emitter, tokens, expected):
    usage = {**tokens, "cost": 0.0, "provider": "openai"}
    ctx = SimpleNamespace(task_id="tools", current_task_type="task", pending_events=[], event_queue=queue.Queue())
    if emitter == "vision":
        from ouroboros.tools.vision import _emit_usage
        _emit_usage(ctx, usage, "model")
        event = ctx.event_queue.get_nowait()
    else:
        from ouroboros.tools.search import _emit_simple_usage
        _emit_simple_usage(ctx, provider="openai", model="model", usage=usage)
        event = ctx.pending_events[0]
    assert [event[key] for key in COUNTERS] == expected
    assert event["usage"] == usage and event["cost"] == 0.0


def test_loop_round_preserves_partial_counters_and_known_scope(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.usage_accounting import UsageScope, usage_scope

    class LLM:
        def chat(self, **kwargs):
            return {"content": "done"}, {"prompt_tokens": 3, "completion_tokens": 0, "cost": 0}

    with usage_scope(UsageScope(task_id="child", root_task_id="root", parent_task_id="parent")):
        call_llm_with_retry(LLM(), [{"role": "user", "content": "hi"}], "test/model", None,
                            "low", 1, tmp_path, "child", 1, None, {}, "task", False)
    row = next(json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()
               if json.loads(line).get("type") == "llm_round")
    assert [row[key] for key in COUNTERS] == [3, 0, None, None]
    assert row["record_contract_version"] == 2
    assert row["root_task_id"] == "root" and row["parent_task_id"] == "parent"


@pytest.mark.parametrize("summary,expected", [
    ({}, [None, None, None, None]),
    ({"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0}, [0, 0, 0, None]),
    ({"inputTokens": 41, "cachedInputTokens": 0}, [41, None, 0, None]),
])
def test_agent_session_emits_one_row_with_actual_receipt(tmp_path, monkeypatch, fake_route, summary, expected):
    from ouroboros import delegate_custody_usage
    from ouroboros.review_substrate import run_review_request

    monkeypatch.setattr(delegate_custody_usage, "_EMITTED_SESSION_USAGE", {})
    fake_route.detail["summary"].update(summary)
    ctx = SimpleNamespace(task_id="t-agent", pending_events=[])
    result = run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                                llm=FakeLLM(), usage_ctx=ctx)
    rows = [row for row in ctx.pending_events if row["type"] == "llm_usage"]
    assert len(rows) == 1
    assert [rows[0]["usage"][key] for key in COUNTERS] == expected
    receipt = rows[0]["usage"]["claudexor"]
    assert receipt["run_id"] == "run-1"
    assert receipt["requested_route"] == receipt["delegated_route"] == "fake-review"
    assert receipt["observed_attempt"]["attempt_id"] == "a01"
    assert rows[0]["provider"] == "claudexor"
    assert rows[0]["model"] == "fake-small"
    assert result.actors[0]["usage"]["claudexor"] == receipt


def test_missing_final_attempt_never_uses_requested_model(tmp_path, monkeypatch, fake_route):
    from ouroboros import delegate_custody_usage
    from ouroboros.review_substrate import run_review_request

    monkeypatch.setattr(delegate_custody_usage, "_EMITTED_SESSION_USAGE", {})
    fake_route.telemetry = {"run_id": "run-1", "final_attempt_id": "missing", "attempts": []}
    ctx = SimpleNamespace(task_id="t-agent", pending_events=[])
    run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                       llm=FakeLLM(), usage_ctx=ctx)
    event = next(row for row in ctx.pending_events if row["type"] == "llm_usage")
    assert event["model"] == ""
    assert event["usage"]["claudexor"]["observed_attempt"] == {}
    assert event["usage"]["claudexor"]["delegated_route"] == ""


def test_failed_after_start_preserves_run_without_inventing_observation(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_usage
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment, ReviewPollUnavailable

    monkeypatch.setattr(delegate_custody_usage, "_EMITTED_SESSION_USAGE", {})
    executor = AgentSessionReviewExecutor(ReviewAssignment(_agent_request(), _agent_slot(), custody_root=tmp_path))
    def fail():
        raise ReviewPollUnavailable("unreachable", run_id="started-run")
    monkeypatch.setattr(executor, "_run_session", fail)
    rows = []
    executor.usage_observer = rows.append
    with pytest.raises(ReviewPollUnavailable):
        executor.execute()
    assert len(rows) == 1
    assert [rows[0][key] for key in COUNTERS] == [None] * 4
    assert rows[0]["claudexor"]["run_id"] == "started-run"
    assert rows[0]["claudexor"]["observed_attempt"] == {}
    assert rows[0]["resolved_model"] == "" and rows[0]["provider"] == "claudexor"
    assert executor.failure_custody()["claudexor"] == rows[0]["claudexor"]


def test_extraction_counters_are_not_native_session_usage(tmp_path, monkeypatch, fake_route):
    from ouroboros import delegate_custody_usage
    from ouroboros.review_substrate import run_review_request

    monkeypatch.setattr(delegate_custody_usage, "_EMITTED_SESSION_USAGE", {})
    fake_route.detail = _terminal_detail("I reviewed the change and found no problems.")
    ctx = SimpleNamespace(task_id="t-agent", pending_events=[])
    result = run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                                llm=FakeLLM(), usage_ctx=ctx)
    session = result.actors[0]["usage"]
    assert session["extraction"]["prompt_tokens"] == 5
    assert [session[key] for key in COUNTERS] == [None] * 4
    rows = [row for row in ctx.pending_events if row["type"] == "llm_usage"]
    assert len(rows) == 1  # session aggregate is never emitted again after extraction
    assert [rows[0]["usage"][key] for key in COUNTERS] == [None] * 4


def test_poll_failure_keeps_dispatched_route_receipt(tmp_path, monkeypatch, fake_route):
    from ouroboros import review_execution

    def fail(*args, **kwargs):
        raise RuntimeError("poll unavailable")
    monkeypatch.setattr(review_execution, "_poll_session_terminal", fail)
    with pytest.raises(RuntimeError, match="poll unavailable") as caught:
        _run_session_directly(tmp_path)
    usage = caught.value.delegated_usage
    assert usage["claudexor"] == {"run_id": "run-1", "requested_route": "fake-review",
                                  "delegated_route": "", "observed_attempt": {}}
    assert [usage[key] for key in COUNTERS] == [None] * 4


@pytest.mark.parametrize("model", ["", "anthropic/claude-sonnet-4", "openai/gpt-5"])
def test_a_claudexor_receipt_keeps_its_own_key_type_through_the_supervisor(tmp_path, model):
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage

    ctx = SimpleNamespace(task_id="review", pending_events=[])
    emit_review_usage(ctx, model=model, usage={"provider": "claudexor", "prompt_tokens": 4}, source="review")
    event = ctx.pending_events[0]
    assert event["api_key_type"] == "claudexor"  # a seat or observed model name never relabels the route
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path))
    persisted = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert persisted["api_key_type"] == "claudexor" and persisted["provider"] == "claudexor"


def test_round_events_keep_an_unreported_cache_unknown(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    replies = [{"prompt_tokens": 100, "cached_tokens": 40}] * 5 + [{"prompt_tokens": 900}] * 3
    replies.append({"prompt_tokens": 100, "cached_tokens": 0})  # an explicit zero is a measured miss

    class LLM:
        def chat(self, **kwargs):
            return {"content": "done"}, {**replies.pop(0), "completion_tokens": 1, "cost": 0}

    logs = tmp_path / "logs"
    logs.mkdir()
    while replies:
        call_llm_with_retry(LLM(), [{"role": "user", "content": "hi"}], "test/model", None,
                            "low", 1, logs, "task", 1, None, {}, "task", False)
    rows = [json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()]
    assert sum(row.get("type") == "llm_round" and row["cached_tokens"] is None for row in rows) == 3
    assert [row["cached_tokens"] for row in rows if row.get("type") == "llm_round"][-1] == 0  # a reported zero stays 0
