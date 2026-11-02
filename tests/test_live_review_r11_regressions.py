"""R11 integration checks: synthetic roots and fake provider construction only."""
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest

from tests.test_live_conversation import host_world as host_world
from tests.test_call_task_lifecycle import direct_call as direct_call


@pytest.mark.parametrize("window,can_raise", [(92_000, False), (80_000, False), (133_554, True)])
@pytest.mark.parametrize("accepted", [0, 200_000])
def test_session_seal_obeys_window_record_and_reserve(host_world, monkeypatch, window, can_raise, accepted):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_window import record_observation
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate

    host = host_world.host
    root = host.ctx.drive_root
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=window, acceptance_can_raise=can_raise,
        basis="estimator" if window == 80_000 else "provider"))
    scope = host.turn.observation_fingerprint()
    if accepted:
        record_observation(root, scope, accepted=accepted, reserve=10_000)
    session, request, evidence = session_candidate(host, host_world.provider["session"], LiveOptions())
    record = host.plan.window_record
    if can_raise and accepted > window:
        assert record.input.tokens == accepted and record.total is None
        assert record.usable == accepted
        expected_capacity = accepted + 10_000
    else:
        assert record.total.tokens == window and record.input is None
        assert record.usable == window - 10_000
        expected_capacity = window
    assert record.declared[0].tokens == window
    assert record.declared[0].acceptance_can_raise is can_raise
    assert request.working_reserve_tokens == 10_000
    assert request.physical_context.capacity_total_tokens == expected_capacity
    assert evidence["fit"]["measurement"]["response_reserve_tokens"] == 10_000
    assert not session.opened

    # A changed account has no right to borrow this account's larger success.
    host_world.settings["TEST_CREDENTIAL"] = "second-synthetic-account"
    _, rotated, _ = session_candidate(host, host_world.provider["session"], LiveOptions())
    assert host.plan.window_record.route_fp != scope
    assert host.plan.window_record.accepted == 0
    assert rotated.physical_context.capacity_total_tokens == window


def test_legacy_nonphysical_preparation_refusal_is_still_terminalized_once(direct_call, tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.context_fit import MinimumViewUnfit
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.utils import append_jsonl
    from supervisor import direct_pause, worker_chat_lane, workers
    from tests.test_context_minimum_view_unfit import FACTS

    world = direct_call
    bus, finished, failures = queue.Queue(), [], []
    world.turn.catalog_model = SimpleNamespace(facts=FACTS)
    world.turn.publish = finished.append
    monkeypatch.setattr(workers, "get_event_q", lambda: bus)
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", append_jsonl)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: None)
    monkeypatch.setattr(direct_pause, "recover_direct_pause",
        lambda *a, **kw: pytest.fail("capacity refusal entered budget-pause recovery"))

    def prepare(task):
        write_task_result(tmp_path, task["id"], "running", chat_id=1, _is_direct_chat=True)
        # The new ordinary fitter admits this soft shortfall. Keep compatibility
        # with an already prepared legacy refusal and its once-only terminalization.
        error = MinimumViewUnfit('Legacy work-room refusal', fixed_tokens=75_000, memory_tokens=1_000,
            reserve_tokens=10_000, window_tokens=92_000, work_room_tokens=11_500, ratio=1.0)
        error.physical = False
        failures.append(error)
        raise error

    agent = SimpleNamespace(handle_task=prepare, env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
                            memory=None, llm=None, tools=SimpleNamespace(_ctx=None))
    admitted = worker_chat_lane._admit_chat_task(agent, 1, "Open the call.", realtime_turn=world.turn)
    task_id = admitted["task"]["id"]
    agent.tools._ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id,
                                  current_chat_id=1, is_direct_chat=True)
    assert worker_chat_lane._execute_chat_task(admitted) is False
    error, = failures
    assert error.physical is False and error.deficit_tokens == 0
    assert error.work_room_deficit_tokens > 0
    assert len(finished) == 1 and world.hub._state["reason"] == "context_capacity"
    assert load_task_result(tmp_path, task_id)["status"] == "failed"
    done, = [row for row in bus.queue if row["type"] == "task_done"]
    assert done["status"] == "failed" and done["reason_code"] == "context_capacity"
    rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    refusal, = [row for row in rows if row["type"] == "call.fit_refused"]
    assert refusal["physical"] is False and refusal["deficit_tokens"] == 0
    assert refusal["fit"]["measurement"]["work_room_deficit_tokens"] == error.work_room_deficit_tokens
    assert [row["ok"] for row in rows if row["type"] == "task_eval"] == [False]
    worker_chat_lane._close_failed_call(admitted, error)
    assert len(finished) == 1
    assert not world.other.closed and not world.notices


def _complete_larger_live_input(host_world, monkeypatch):
    import asyncio
    from ouroboros import capability_evidence as ce
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.contracts.live_conversation import LiveOptions

    host = host_world.host
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: host.ctx.drive_root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=133_554, acceptance_can_raise=True))
    # An ordinary cold estimate undercounts a large text input by about 25%.
    # The successful answer supplies the exact modality witness, not a fabricated ceiling.
    before = estimate_context_prompt_tokens(host.messages, host.schemas)
    host.messages.append({"role": "user", "content": "x" * max(0, (120_000 - before) * 4)})
    host._context_changed()

    async def complete(session):
        assert session.opened
        host.response_started("larger-success")
        host.response_done("larger-success", usage={"text_input_tokens": 150_000,
            "audio_input_tokens": 0, "image_input_tokens": 0, "video_input_tokens": 0,
            "text_output_tokens": 20, "audio_output_tokens": 0, "cached_input_tokens": 0,
            "thinking_tokens": 0}, raw_usage={"input_tokens": 150_000,
            "input_token_details": {"text_tokens": 150_000, "audio_tokens": 0}}, status="completed")

    asyncio.run(host.session(complete, LiveOptions()))
    density, basis = ce.resolve_main_token_density(host.ctx.drive_root, host.plan.route_fp,
        host.turn.model, estimate_tokens=120_000, view_kind=host.plan.view_kind)
    assert density > 1.2 and basis == "fresh_route_usage", "the successful response did reach the density learner"
    return host


def test_completed_live_response_teaches_the_ordinary_window_record(host_world, monkeypatch):
    from ouroboros.context_window import resolve_window
    host = _complete_larger_live_input(host_world, monkeypatch)
    current = resolve_window(host.plan, 10_000, root=host.ctx.drive_root)
    assert current.accepted >= 150_000, "live completion lost the accepted-input fact"
    assert current.input.tokens == current.usable == 150_000


@pytest.mark.parametrize("supply_missing_observation", [False, True])
def test_larger_accepted_live_input_survives_renewal(host_world, monkeypatch, supply_missing_observation):
    from ouroboros.context_window import record_observation
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate
    host = _complete_larger_live_input(host_world, monkeypatch)
    if supply_missing_observation:
        record_observation(host.ctx.drive_root, host.plan.route_fp, accepted=150_000)
    session, request, _ = session_candidate(host, host_world.provider["session"], LiveOptions())
    assert request.physical_context.capacity_total_tokens == 160_000
    assert not session.opened


@pytest.mark.parametrize("can_raise,expected_due", [(True, False), (False, True)])
def test_renewal_uses_the_resolved_window_not_the_catalog_lower_bound(host_world, monkeypatch, can_raise, expected_due):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_window import record_observation
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate

    host = host_world.host
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: host.ctx.drive_root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=133_554, acceptance_can_raise=can_raise))
    record_observation(host.ctx.drive_root, host.turn.observation_fingerprint(), accepted=200_000)
    _, request, _ = session_candidate(host, host_world.provider["session"], LiveOptions())
    host._physical_context = request.physical_context
    host._held_input, host._growth = 140_000, 0
    host._history_at_receipt = host._history_estimate()
    assert host._physical_context.capacity_total_tokens == (210_000 if can_raise else 133_554)
    assert host.renewal_due() is expected_due
