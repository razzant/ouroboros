"""Authoring handovers exercised through the real loop, dispatcher and finalizer.

Only model responses are scripted. Tool calls read a real isolated file; fallback
selection, wait re-preparation, transcript custody and final admission stay real.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop, model_wait
from ouroboros.llm import LLMClient
from ouroboros.outcomes import derive_loop_outcome
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
from ouroboros.tools.registry import ToolRegistry
from tests.test_completion_selection import finish

PRIMARY = "openai/model-a"
SUCCESSOR = "anthropic/model-b"
TASK = "handover-loop"


def _text(content):
    return {"role": "assistant", "content": content}


def _tool(name="read_file", **args):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "read-" + str(args.get("path", name)), "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", SUCCESSOR)
    monkeypatch.setenv("USE_LOCAL_MAIN", "false")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setenv("OUROBOROS_REASONING_SUMMARY", "off")
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    (tmp_path / "evidence.txt").write_text("Verified predecessor evidence\n", encoding="utf-8")
    from ouroboros import context
    from tests.test_context_fit_integration import _plan
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="test-route-" + task["model"], status="confirmed",
                        stale=False, window_tokens=900_000, source="advertised"),
    ))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.context_fit_plan = replace(_plan(preferred="max", window=900_000), model=PRIMARY)
    state = SimpleNamespace(root=tmp_path, tools=registry, calls=[], notes=[], steps=[], task_id=TASK)

    def scripted(_llm, messages, model, _schemas, _effort, _retries, _logs,
                 _task_id, _round, _events, usage, *_args, **kwargs):
        state.calls.append({"model": model, "messages": deepcopy(messages)})
        assert state.steps, "Unexpected model call after the scripted terminal response"
        step = state.steps.pop(0)
        if callable(step):
            step = step(state, usage, kwargs)
        if step is None:
            usage["_last_llm_error_kind"] = "invalid_continuation"
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return deepcopy(step), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", scripted)

    def run(steps, *, task_id=TASK):
        from ouroboros.task_results import write_task_result
        write_task_result(tmp_path, task_id, "running", root_task_id=task_id, _attempt=1)
        state.steps = list(steps)
        state.task_id = task_id
        state.result = loop.run_llm_loop(
            messages=registry._ctx.context_fit_plan.messages_for("max"),
            tools=registry, llm=SimpleNamespace(default_model=lambda: PRIMARY),
            drive_logs=tmp_path / "logs", drive_root=tmp_path,
            emit_progress=lambda text, **_meta: state.notes.append(text),
            incoming_messages=queue.Queue(), task_id=task_id,
        )
        assert not state.steps, "The loop finalized before consuming the expected continuation"
        return state.result

    state.run = run
    return state


def _read():
    return _tool(path="evidence.txt")


def _handover_notes(trace):
    return [note for note in trace["reasoning_notes"] if "handover recovery" in note.lower()]


@pytest.mark.parametrize("first_status", ["The worktree is clean.", "Finished successfully.", "Работа ещё продолжается."])
def test_real_fallback_after_tool_result_requires_one_recovery(episode, first_status):
    text, usage, trace = episode.run([_read(), None, _text(first_status), finish("No additional work done.")])
    assert [call["model"] for call in episode.calls] == [PRIMARY, PRIMARY, SUCCESSOR, SUCCESSOR]
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "finish_task"]
    assert trace["tool_calls"][0]["is_error"] is False
    assert any(row.get("role") == "tool" and "Verified predecessor evidence" in row.get("content", "")
               for row in episode.calls[2]["messages"])
    assert len(_handover_notes(trace)) == 1
    assert text == "No additional work done."
    assert usage["reason_code"] == "authoring_handover_incomplete"
    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    assert axes["execution"]["status"] == "degraded"
    assert axes["execution"]["reason_code"] == "authoring_handover_incomplete"
    assert trace["route_handovers"][0]["tool_calls_at_handover"] == 1


def test_recovery_tools_publish_fresh_final_without_json_only_instruction(episode):
    text, usage, trace = episode.run([
        _read(), None, _text("WIP before recovery"), _read(), finish("Fresh verified report"),
    ])
    assert text == "Fresh verified report"
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "read_file", "finish_task"]
    assert len(_handover_notes(trace)) == 1
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert "authoring_handover_incomplete" not in trace
    assert not any("DELIVERY_FINALIZATION_CONTROL" in str(row.get("content", ""))
                   for call in episode.calls for row in call["messages"])
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_zero_prior_tools_fallback_remains_normal_final(episode):
    text, usage, trace = episode.run([None, _text("Ordinary answer")])
    assert text == "Ordinary answer"
    assert len(episode.calls) == 2
    assert not trace.get("route_handovers")
    assert not _handover_notes(trace)
    assert usage.get("reason_code") != "authoring_handover_incomplete"


def test_explicit_switch_model_is_not_host_handover(episode, monkeypatch):
    monkeypatch.setattr(LLMClient, "available_models", lambda _self: [PRIMARY, SUCCESSOR])
    text, usage, trace = episode.run([
        _read(), _tool("switch_model", model=SUCCESSOR), _text("Deliberately switched final"),
    ])
    assert text == "Deliberately switched final"
    assert [call["model"] for call in episode.calls] == [PRIMARY, PRIMARY, SUCCESSOR]
    assert all(row["is_error"] is False for row in trace["tool_calls"])
    assert not trace.get("route_handovers")
    assert usage.get("reason_code") != "authoring_handover_incomplete"


@pytest.mark.parametrize("targets", [(PRIMARY,), (SUCCESSOR,), (SUCCESSOR, SUCCESSOR)])
@pytest.mark.parametrize("prior_tools", [False, True])
def test_model_wait_reprepare_preserves_original_author_and_tool_baseline(episode, targets, prior_tools):
    target = targets[-1]
    def switch(_state, _usage, kwargs):
        owner = model_wait.current_model_wait()
        for destination in targets:
            prepared = owner.reprepare(kwargs["model_role"], {
                "model": destination, "model_role": kwargs["model_role"], "use_local": False,
                "messages": episode.calls[-1]["messages"],
            })
            assert isinstance(prepared, model_wait.PreparedModelCall)
        assert any(row.get("role") == "tool" for row in prepared.kwargs["messages"]) is prior_tools
        return _text("Wait switched response")

    steps = [_read(), switch] if prior_tools else [switch]
    if target != PRIMARY and prior_tools:
        steps.append(finish("Second successor response"))
    with model_wait.task_model_wait_scope(task={"id": TASK}, drive_root=episode.root,
                                         event_queue=None, worker_slot_held=False) as owner:
        owner.tool_context = episode.tools._ctx
        text, usage, trace = episode.run(steps)
    if target == PRIMARY or not prior_tools:
        assert text == "Wait switched response"
        assert not trace.get("route_handovers")
        assert not _handover_notes(trace)
    else:
        assert text == "Second successor response"
        assert episode.calls[-1]["model"] == SUCCESSOR
        assert trace["route_handovers"][0]["reason"] == "model_wait"
        assert trace["route_handovers"][0]["from_model"] == PRIMARY
        assert trace["route_handovers"][0]["tool_calls_at_handover"] == 1
        assert len(_handover_notes(trace)) == 1
        assert usage["reason_code"] == "authoring_handover_incomplete"


@pytest.mark.parametrize("control", ["owner_stop", "deadline"])
def test_forced_finalization_takes_precedence_over_pending_handover(episode, control):
    def trigger(state, _usage, _kwargs):
        if control == "owner_stop":
            from ouroboros.cancel_intents import request_cancel, STOP_POLICY_FINALIZE
            from supervisor.owner_stop import owner_stop_control_id
            intent = request_cancel(state.root, TASK, requested_stop_policy=STOP_POLICY_FINALIZE)
            write_owner_message(state.root, "owner_requested_finalization", TASK,
                                msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)
        else:
            state.tools._ctx.task_metadata = {
                "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat(),
            }
        return _text("Interim response")

    steps = [_read(), None, trigger]
    if control == "deadline":
        steps.append(_text("Best effort on interruption"))
    text, usage, trace = episode.run(steps)
    assert text, (usage, trace)
    if control == "deadline":
        assert "Best effort on interruption" in text
    assert usage["reason_code"] == ("owner_requested_finalization" if control == "owner_stop" else "deadline_local")
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert len(_handover_notes(trace)) == 1


@pytest.mark.parametrize("prior_terminal", ["incomplete", "pending_at_finalize"])
def test_reused_tool_context_does_not_inherit_previous_handover(episode, prior_terminal):
    def interim(state, _usage, _kwargs):
        if prior_terminal == "pending_at_finalize":
            write_owner_message(state.root, "deadline", TASK, kind=KIND_FINALIZE_NOW)
        return _text("First status")

    episode.run([_read(), None, interim,
                 _text("Still status") if prior_terminal == "pending_at_finalize" else finish("Still status")])
    text, usage, trace = episode.run([_text("Unrelated new task")], task_id="later-task")
    assert text == "Unrelated new task"
    assert not trace.get("route_handovers")
    assert not trace.get("authoring_handover_incomplete")
    assert usage.get("reason_code") != "authoring_handover_incomplete"


def test_owner_followup_after_incomplete_can_resume_tools_and_heal_warning(episode, monkeypatch):
    from ouroboros.tools import review_result

    agent = SimpleNamespace(_owner_message_admission_lock=threading.Lock(),
                            _accepting_owner_messages=True, _busy=True, _current_task_id=TASK)
    episode.tools._ctx.owner_message_admission_lock = agent._owner_message_admission_lock
    episode.tools._ctx.owner_message_admission_agent = agent

    # The owner writes while the incomplete answer is at its delivery step, right before
    # the admission close drains the mailbox (the result review's step in the real path).
    at_delivery = review_result.result_review_at_delivery
    sent = []

    def followup_at_delivery(ctx, content, llm_trace, hold):
        if not sent and llm_trace.get("authoring_handover_incomplete"):
            write_owner_message(episode.root, "Read the evidence once more and finish.", TASK, msg_id="followup")
            sent.append(1)
        return at_delivery(ctx, content, llm_trace, hold)

    monkeypatch.setattr(review_result, "result_review_at_delivery", followup_at_delivery)

    def informed_final(state, _usage, _kwargs):
        source = state.tools._ctx._acceptance_observation["owner_source_sha256"]
        assert source in str(state.calls[-1]["messages"])
        return finish("Complete after follow-up", acceptance_subject={"owner_source_sha256": source})

    text, usage, trace = episode.run([
        _read(), None, _text("Initial WIP"), finish("Still incomplete before follow-up"), _read(), informed_final,
    ])
    assert text == "Complete after follow-up"
    assert sent == [1]
    assert [row["tool"] for row in trace["tool_calls"]] == ["read_file", "finish_task", "read_file", "finish_task"]
    assert len(_handover_notes(trace)) == 1
    assert any("Read the evidence once more" in str(row.get("content", ""))
               for row in episode.calls[-1]["messages"])
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert "authoring_handover_incomplete" not in trace
    incident = trace["route_handovers"][0]
    assert incident["incomplete_observed"] is True
    assert incident["status"] == "recovered"
    outcome = derive_loop_outcome(text, usage, trace)
    assert outcome["outcome_axes"]["execution"]["reason_code"] != "authoring_handover_incomplete"


def test_second_toolless_handover_preserves_real_verification_nudge(episode):
    from ouroboros.outcomes import append_verification_receipt
    assert append_verification_receipt(episode.root, TASK, {
        "status": "fail", "kind": "run", "check": "existing project check", "returncode": 1,
    })
    replacement = finish("Final with disclosed failed check")
    text, usage, trace = episode.run([
        _read(), None, _text("Initial WIP"), finish("Still incomplete"), replacement,
    ])
    assert text == "Final with disclosed failed check"
    assert len(_handover_notes(trace)) == 1
    assert sum("Red-verification nudge" in note for note in trace["reasoning_notes"]) == 1
    assert any("host-attested verification is RED" in str(row.get("content", ""))
               for row in episode.calls[-1]["messages"])
    assert usage["reason_code"] == "authoring_handover_incomplete"


def test_second_toolless_handover_preserves_real_skill_readiness_nudge(episode):
    from ouroboros.outcomes import append_verification_receipt
    from tests.test_loop_skill_finalization import _write_self_authored_skill

    _write_self_authored_skill(episode.root)
    # Isolate the skill reminder from the separately covered verification rail.
    assert append_verification_receipt(episode.root, TASK, {"status": "observed", "kind": "artifact"})
    write = _tool("write_file", path="notes.txt", content="Authored skill notes",
                  root="skill_payload", bucket="external", skill_name="alpha")
    text, usage, trace = episode.run([
        write, None, _text("Initial WIP"), finish("Still incomplete"),
        finish("Skill is authored but still needs review and enablement."),
    ])
    assert text == "Skill is authored but still needs review and enablement."
    assert trace["tool_calls"][0]["is_error"] is False
    assert (episode.root / "skills" / "external" / "alpha" / "notes.txt").read_text(encoding="utf-8") == "Authored skill notes"
    assert len(_handover_notes(trace)) == 1
    assert any("SKILL_NOT_FINALIZED" in str(row.get("content", ""))
               for row in episode.calls[-1]["messages"])
    assert usage["reason_code"] == "authoring_handover_incomplete"
