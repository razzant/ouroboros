"""A wake's complete observation is delivered whole when the actual request fits it,
and by its exact stored source when it does not (``loop_model_call._project_wake_input``).

The decision reads the real Main measurement — tool schemas, the answer reserve and a
clock line against a KNOWN capacity — at every round boundary until a response has
consumed the wake input (a grown tool set or a switched route is measured again);
unknown capacity sends the whole input, and a real provider overflow before consumption
may use the smaller delivery as its one strict-shrink retry. Consumed history is never
rewritten, and only a strictly smaller delivery is ever swapped in. The task text and
stored source never change.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import loop, usage_accounting as ua
from ouroboros.llm import _canonical_candidate_bytes
from ouroboros.loop_model_call import _project_wake_input, _RoundModelCallContext
from ouroboros.send_clock import CLOCK_NOTE_PREFIX
from tests.test_context_fit_integration import _plan
from tests.test_llm_claudexor import ledger, result, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_fixture

setup = gateway_fixture
live_wait = wait_fixture
main_call = main_fixture

FULL = "Wake facts:\n" + "\n".join(f"- task t{index:04d} completed: " + "x" * 120 for index in range(2500))
PROJECTION = "Wake facts: 2500 task terminals; the complete observation is stored at <source>."
WAKE = {"projection_text": PROJECTION, "composition": {"task_terminal": 2500},
        "source": {"sha256": "f" * 64, "read": {"tool": "read_file"}}}
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "d" * 400,
                                           "parameters": {"type": "object"}}}]


@pytest.fixture(autouse=True)
def _density(monkeypatch):
    from ouroboros import capability_evidence

    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "fresh_route_usage"))


def _ctx(tmp_path, *, window, text=FULL, wake=WAKE, tools=TOOLS, status="confirmed"):
    plan = replace(_plan(preferred="max", window=window), status=status)
    messages = [plan.messages_for("max")[0], {"role": "user", "content": text}]
    events = []
    inner = SimpleNamespace(task_id="wake0001", task_metadata={"wake_observation": wake} if wake else {},
                            messages=messages, context_fit_plan=plan)
    ctx = _RoundModelCallContext(
        llm=None, messages=messages, tools=SimpleNamespace(_ctx=inner), context_fit_plan=plan,
        active_model=plan.model, tool_schemas=tools, active_effort="medium", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="wake0001", round_idx=1, event_queue=None,
        accumulated_usage={}, task_type="task", active_use_local=False, active_context_mode="max",
        drive_root=tmp_path)
    return ctx, events


def _record_events(monkeypatch):
    events = []
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: events.append(_a[-1]))
    return events


def test_a_fitting_observation_is_sent_whole(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=500_000)
    assert _project_wake_input(ctx) is False
    assert ctx.messages[1]["content"] == FULL and events == []


def test_a_route_switch_before_any_response_is_measured_again(tmp_path, monkeypatch):
    """No response consumed the input yet (the first send failed): the next round's route is
    smaller, so its own measurement decides again."""
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=500_000)
    assert _project_wake_input(ctx) is False
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=120_000)
    assert _project_wake_input(ctx) is True and ctx.messages[1]["content"] == PROJECTION
    assert [event["after_overflow"] for event in events] == [False]


@pytest.mark.parametrize('overflow', [False, True])
def test_failed_cross_family_fallback_projects_each_actual_candidate(main_call, monkeypatch, overflow):
    from ouroboros import context, fallback_cooldown
    from ouroboros.context_fit import extract_plain_text_from_content

    ctx, gateway, _controller, _events, _decide, _observations = main_call
    targets = ['claudexor::claude=claude-test', 'claudexor::gemini=gemini-test']
    monkeypatch.setenv('OUROBOROS_MODEL_FALLBACKS', ','.join(targets))
    monkeypatch.setattr(fallback_cooldown, 'is_cooling_down', lambda *_args: False)
    monkeypatch.setattr(fallback_cooldown, 'attempts_per_model', lambda: 1)
    route_evidence = context._context_fit_route
    def capacity(task, **kwargs):
        route, evidence = route_evidence(task, **kwargs)
        evidence.window_tokens = 500_000 if overflow and task['model'] == targets[0] else 120_000
        return route, evidence
    monkeypatch.setattr(context, '_context_fit_route', capacity)
    ctx.messages[:] = [ctx.context_fit_plan.messages_for('max')[0], {'role': 'user', 'content': FULL}]
    ctx.max_retries = ctx.attempt_cap = 1
    ctx.tools._ctx.task_metadata = {'wake_observation': deepcopy(WAKE)}
    failure = result(outcome='failed', problem={'code': 'invalid_request', 'message': 'fixture failure',
                                               'context': {'httpStatus': 400}})
    too_long = result(outcome='failed', problem={'code': 'provider_failed', 'message': 'too long', 'context': {
        'httpStatus': 400, 'vendorCode': 'context_length_exceeded', 'parameter': 'input'}})
    gateway.results = [failure, *([too_long] if overflow else []), failure, result()]
    gateway.dispatch = ['response_received'] * len(gateway.results)
    assert loop._call_round_model(ctx)[0] is None  # large primary sends full input and fails
    answer, *_rest = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages,
        active_model=ctx.active_model, active_use_local=False, tool_schemas=[], active_effort='medium',
        max_retries=1, drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=1,
        event_queue=ctx.event_queue, accumulated_usage=ctx.accumulated_usage, task_type='task',
        emit_progress=lambda _text, **_kwargs: None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode='max')
    assert answer is not None
    sent = [extract_plain_text_from_content(payload['messages'][1]['content']) for payload, _key in gateway.uploads]
    assert sent == [FULL, *([FULL] if overflow else []), PROJECTION, PROJECTION]
    assert extract_plain_text_from_content(ctx.messages[1]['content']) == PROJECTION


def test_a_grown_tool_set_before_any_response_is_measured_again(tmp_path, monkeypatch):
    _record_events(monkeypatch)
    small = FULL[:120_000]
    ctx, _ = _ctx(tmp_path, window=100_000, text=small, tools=[])
    assert _project_wake_input(ctx) is False
    ctx.tool_schemas = [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 4_000,
                                                          "parameters": {"type": "object"}}} for i in range(12)]
    assert _project_wake_input(ctx) is True


def test_consumed_history_is_never_rewritten_even_by_an_overflow(tmp_path, monkeypatch):
    """A response followed the whole input: later rounds, a smaller route and a real overflow
    all leave it as sent (the ordinary overflow path shrinks other material)."""
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=500_000)
    assert _project_wake_input(ctx) is False
    ctx.messages.extend([{"role": "user", "content": "[Host clock at this request: …]"},
                         {"role": "assistant", "content": "I read the whole wake."},
                         {"role": "user", "content": "a later owner line"}])
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=70_000)
    assert _project_wake_input(ctx) is False and _project_wake_input(ctx, overflowed=True) is False
    assert ctx.messages[1]["content"] == FULL and events == []


def test_only_a_strictly_smaller_delivery_is_swapped_in(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    tiny = "Wake facts: one owner line."
    ctx, _ = _ctx(tmp_path, window=0, status="unknown", text=tiny,
                  wake={**WAKE, "projection_text": tiny + " " + "The complete observation is stored at <source>."})
    assert _project_wake_input(ctx, overflowed=True) is False
    assert ctx.messages[1]["content"] == tiny and events == []


def test_a_known_window_that_cannot_hold_it_gets_the_source_overview(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=120_000)
    assert _project_wake_input(ctx) is True
    assert ctx.messages[1]["content"] == PROJECTION
    assert ctx.messages[1]["cache_control"] == {"type": "ephemeral"}  # re-sealed as the prefix anchor
    assert ctx.tools._ctx.task_metadata["wake_observation"] is WAKE  # the source facts are untouched
    assert events == [{"checkpoint_kind": "wake_input_by_source", "round": 1, "after_overflow": False,
                       "composition": {"task_terminal": 2500}, "source_sha256": "f" * 64}]


def test_the_decision_is_tool_inclusive(tmp_path, monkeypatch):
    _record_events(monkeypatch)
    small = FULL[:120_000]  # ~30K tokens: fits a 100K window beside the 65K reserve without tools
    assert _project_wake_input(_ctx(tmp_path, window=100_000, text=small, tools=[])[0]) is False
    heavy = [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 4_000,
                                               "parameters": {"type": "object"}}} for i in range(12)]
    assert _project_wake_input(_ctx(tmp_path, window=100_000, text=small, tools=heavy)[0]) is True


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_unknown_capacity_sends_the_whole_input_and_overflow_may_shrink_it(tmp_path, monkeypatch, status):
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=0, status=status)
    assert _project_wake_input(ctx) is False and ctx.messages[1]["content"] == FULL
    # A real provider overflow is proof: the stored-source delivery is the smaller retry.
    assert _project_wake_input(ctx, overflowed=True) is True
    assert ctx.messages[1]["content"] == PROJECTION
    assert events[-1]["after_overflow"] is True
    assert _project_wake_input(ctx, overflowed=True) is False  # once


@pytest.mark.parametrize("wake", [None, {"source_error": "OSError: store unavailable", "composition": {}}])
def test_without_a_stored_source_the_complete_input_is_the_only_delivery(tmp_path, monkeypatch, wake):
    _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=120_000, wake=wake)
    assert _project_wake_input(ctx) is False and _project_wake_input(ctx, overflowed=True) is False
    assert ctx.messages[1]["content"] == FULL


def test_the_projected_wake_is_what_is_measured_priced_sealed_and_sent(main_call, monkeypatch):
    """Through the real Main round and the real Claudexor lane: one candidate, one set of bytes."""
    ctx, gateway, _controller, events, _decide, _observations = main_call
    plan = replace(ctx.context_fit_plan, window_tokens=120_000)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.messages[:] = [plan.messages_for("max")[0], {"role": "user", "content": FULL}]
    ctx.tools._ctx.task_metadata = {"wake_observation": deepcopy(WAKE)}
    gateway.results, gateway.dispatch = [result()], ["response_received"]
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer and len(gateway.uploads) == 1
    sent = gateway.uploads[0][0]
    assert sent["messages"][1]["content"] == PROJECTION  # the seal's mark stays host-only
    assert sent["messages"][-1]["content"].startswith(CLOCK_NOTE_PREFIX)
    assert FULL not in json.dumps(sent)
    rows = ledger(ctx.drive_root)
    settled = [row for row in rows if row["state"] == "settled"][-1]
    digest = hashlib.sha256(_canonical_candidate_bytes(sent)).hexdigest()
    assert {row["candidate_raw_sha256"] for row in rows if row["attempt_id"] == settled["attempt_id"]} == {digest}
    # The canonical transcript now holds the projected input and the consumed clock line.
    assert ctx.messages[1]["content"] == PROJECTION
    assert ctx.messages[-1]["content"] == sent["messages"][-1]["content"]
    checkpoints = [event for event in list(events.queue)
                   if "wake_input_by_source" in json.dumps(event, default=str)]
    assert len(checkpoints) == 1 and ua.last_physical_attempt_capture().state == "settled"


def test_a_real_overflow_before_any_response_retries_with_the_strictly_smaller_delivery(main_call, monkeypatch):
    """Through the real Main round and Claudexor lane: the whole input fits the prediction, the
    provider proves otherwise, and the one strict-shrink retry carries the stored-source delivery —
    measured, priced, sealed and sent as one candidate."""
    ctx, gateway, _controller, events, _decide, _observations = main_call
    ctx.messages[:] = [ctx.context_fit_plan.messages_for("max")[0], {"role": "user", "content": FULL}]
    ctx.tools._ctx.task_metadata = {"wake_observation": deepcopy(WAKE)}
    overflow = result(outcome="failed", problem={"code": "provider_failed", "message": "too long", "context": {
        "httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
    gateway.results, gateway.dispatch = [overflow, result()], ["response_received", "response_received"]
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer and len(gateway.uploads) == 2
    first, retry = gateway.uploads[0][0], gateway.uploads[1][0]
    assert first["messages"][1]["content"] == FULL  # unsealed: the whole input went out first
    assert retry["messages"][1]["content"] == PROJECTION
    assert json.dumps(FULL)[1:-1] not in json.dumps(retry)
    rows = ledger(ctx.drive_root)
    settled = [row for row in rows if row["state"] == "settled"][-1]
    digest = hashlib.sha256(_canonical_candidate_bytes(retry)).hexdigest()
    assert {row["candidate_raw_sha256"] for row in rows if row["attempt_id"] == settled["attempt_id"]} == {digest}
    checkpoints = [event for event in list(events.queue) if "wake_input_by_source" in json.dumps(event, default=str)]
    assert len(checkpoints) == 1 and '"after_overflow": true' in json.dumps(checkpoints[0], default=str)


def test_unconsumed_pointer_returns_to_full_input_on_larger_route_then_stays_history(tmp_path, monkeypatch):
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(tmp_path, window=120_000)
    assert _project_wake_input(ctx)
    assert PROJECTION in json.dumps(ctx.messages[1])
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=500_000)
    assert _project_wake_input(ctx)
    assert FULL in json.dumps(ctx.messages[1], ensure_ascii=False).replace("\\n", "\n")
    assert [e["checkpoint_kind"] for e in events] == ["wake_input_by_source", "wake_input_inline"]
    ctx.messages.append({"role": "assistant", "content": "consumed"})
    before = deepcopy(ctx.messages)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=120_000)
    assert not _project_wake_input(ctx)
    assert ctx.messages == before


@pytest.mark.parametrize("initial_window,next_window", [(900_000, 120_000), (120_000, 500_000)])
def test_waiting_account_repair_reprojects_wake_before_dispatch(main_call, monkeypatch, initial_window, next_window):
    from ouroboros import context
    from tests.test_subscription_main_wait import ROUTE_B, _failed

    ctx, gateway, _controller, events, decide, _observations = main_call
    route_evidence = context._context_fit_route
    def changed_capacity(*args, **kwargs):
        route, evidence = route_evidence(*args, **kwargs)
        evidence.window_tokens = next_window
        return route, evidence
    monkeypatch.setattr(context, "_context_fit_route", changed_capacity)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=initial_window)
    ctx.messages[:] = [ctx.context_fit_plan.messages_for("max")[0], {"role": "user", "content": FULL}]
    ctx.tools._ctx.task_metadata = {"wake_observation": deepcopy(WAKE)}
    gateway.results = [_failed("subscription_window_exhausted"), result(route=ROUTE_B)]
    def select_account(*_args, **_kwargs):
        wait = next(e for e in reversed(list(events.queue)) if e.get("type") == "task_model_wait")
        response = decide({"request_id": "wake-account-switch", "decision_id": f"model_wait:task-one:{wait['wait_id']}",
                           "revision": wait["revision"], "action": "switch", "model": ctx.active_model,
                           "credential_profile_id": "account-b", "use_local": False, "persist_role": False})
        assert response.status_code == 202
        return {}
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", select_account)
    gateway.dispatch = ["not_started", "response_received"]
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer, json.dumps({k: v for k, v in ctx.accumulated_usage.items() if "error" in k}, default=str)
    assert len(gateway.uploads) == 2
    first, second = (entry[0]["messages"][1] for entry in gateway.uploads)
    def text_of(message):
        value = message["content"]
        return value if isinstance(value, str) else "".join(b.get("text", "") for b in value)
    assert text_of(first) == (FULL if initial_window > 120_000 else PROJECTION)
    assert text_of(second) == (FULL if next_window > 120_000 else PROJECTION)
    settled = [row for row in ledger(ctx.drive_root) if row["state"] == "settled"][-1]
    assert settled["physical_context"]["capacity_total_tokens"] == next_window
    assert settled["candidate_raw_sha256"] == hashlib.sha256(_canonical_candidate_bytes(gateway.uploads[1][0])).hexdigest()
