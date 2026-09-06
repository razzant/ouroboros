"""Outage facts reach real delivery projections without rewriting model sources."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, cancel_intents, loop, loop_llm_call, loop_transport
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.task_finalization import send_provider_death_notice
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolRegistry
from supervisor.owner_stop import owner_stop_control_id
from supervisor.terminal_delivery import build_completed_result_event, pending_deliveries
from tests.test_delivery_forced_finalization import _forced_test_context
from tests.test_loop_transport_wait import _loop_kwargs, _read_network_wait_events
from tests.test_presence_runner import _admission, _event


RAW = "Exact model source: λ\n\nA useful intermediate result."


def _terminal(tmp_path, *, current, task_id="parent1"):
    usage = {"_last_llm_error_kind": "provider_outcome_unknown",
             loop_llm_call.TRANSPORT_DEATHS_KEY: {"round_id": "round", "count": 1,
                                                "error_kind": "provider_outcome_unknown"}}
    _loop, registry, ctx, trace = _forced_test_context(tmp_path, usage=usage)
    ctx.task_id = registry._ctx.task_id = task_id
    registry._ctx.task_metadata["root_task_id"] = task_id
    if current:
        loop._replace_delivery_candidate(registry, ctx, trace, RAW, control="replace")
    else:
        ctx.messages.append({"role": "assistant", "content": RAW})
    text, usage, trace = loop._handle_provider_unavailable(ctx, error_kind="provider_outcome_unknown",
        wait_cause="transport_unavailable", waited_sec=125.0)
    assert text == RAW
    assert "waited and redialed for 2.1 min" in usage["terminal_provider_notice"]
    assert "no terminal provider outcome" in usage["terminal_provider_notice"]
    assert "no further retry or paid fallback was sent" in usage["terminal_provider_notice"]
    return text, usage, trace


@pytest.mark.parametrize("mode", ["managed", "direct", "ephemeral"])
@pytest.mark.parametrize("current", [False, True])
def test_pipeline_delivery_and_rebuild_keep_raw_bytes_and_known_wait_custody(tmp_path, monkeypatch, mode, current):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    text, usage, trace = _terminal(tmp_path, current=current)
    task = {"id": "parent1", "type": "task", "chat_id": 7, "text": "finish the task"}
    if mode != "managed":
        task["_is_direct_chat"] = True
    if mode == "ephemeral":
        task["_ephemeral_turn"] = True
    pending = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, text, usage, trace, start_time=0.0, drive_logs=tmp_path / "logs")
    sent = next(row for row in pending if row["type"] == "send_message")
    notice = usage["terminal_provider_notice"]
    if mode == "ephemeral":
        assert load_task_result(tmp_path, "parent1") is None
        assert Path(usage["terminal_salvage_path"]).read_text() == RAW
        assert RAW in sent["text"] and sent["text"].count("[Host status]") == 1
        assert notice in sent["text"] and "task details" not in sent["text"]
        assert sent["log_text"] == sent["text"]
        return
    stored = load_task_result(tmp_path, "parent1")
    assert stored["result"] == RAW and stored["terminal_provider_notice"] == notice
    assert stored["status"] == "failed"  # same provider-outage category
    if current:
        assert sent["text"] == RAW
    else:
        assert RAW not in sent["text"] and notice in sent["text"]
        assert Path(stored["terminal_salvage_path"]).read_text() == RAW
    replay = build_completed_result_event(tmp_path, task, "parent1", stored)
    assert replay["text"] == sent["text"] and replay["delivery_id"] == sent["delivery_id"]
    assert pending_deliveries(tmp_path)[0]["text"] == sent["text"]
    incidents = []
    notified = send_provider_death_notice(SimpleNamespace(send_with_budget=lambda *a, **k: incidents.append((a, k))),
                                          7, "parent1", stored)
    assert notified is current
    if current:
        assert notice in incidents[0][0][1]
        assert "re-run the task" not in incidents[0][0][1]


@pytest.mark.parametrize("outcome", ["message", "deferred", "silent", "tool_delivered"])
@pytest.mark.parametrize("current", [False, True])
def test_actual_presence_render_and_cached_read_have_one_notice_and_keep_outcome(tmp_path, monkeypatch, outcome, current):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    created = []
    class Agent:
        def handle_task(self, task):
            text, usage, trace = _terminal(data, current=current, task_id=task["id"])
            task["_skip_post_task_synthesis"] = True
            ctx = SimpleNamespace(_presence_completion={"outcome": outcome, "message": "Typed Presence reply"},
                                  _swarm_handoff_attempt={"status": "scheduled", "task_id": "next-task"})
            pending = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=data, repo_dir=repo), None, None,
                pending, task, text, usage, trace, start_time=0.0, drive_logs=data / "logs", ctx=ctx)
            return pending
    def factory(**kwargs):
        created.append(kwargs)
        return Agent()
    args = dict(admission=_admission(), event=_event(), repo_dir=repo, drive_root=data,
                agent_factory=factory, gate=PresenceTurnGate(2))
    first = run_presence_turn(**args)
    cached = run_presence_turn(**args)
    assert cached == first and len(created) == 1
    assert first.outcome == outcome
    stored = load_task_result(data, first.task_id)
    assert stored["result"] == RAW
    assert "no terminal provider outcome" in stored["terminal_provider_notice"]
    if outcome in {"message", "deferred"}:
        assert first.text.startswith("Typed Presence reply\n\n[Host status]")
        assert first.text.count("[Host status]") == 1
        assert stored["metadata"]["presence_result_text"] == first.text
    else:
        assert first.text == ""  # silence/tool delivery authority is not overwritten
    if outcome == "deferred":
        assert first.work_ref == "next-task"


@pytest.mark.parametrize("reason", [REASON_OWNER_REQUESTED_FINALIZATION, "deadline", "budget ceiling reached"])
def test_real_mailbox_wait_exit_carries_the_control_cause_without_another_call(tmp_path, monkeypatch, reason):
    calls = []
    def fail(_llm, _messages, _model, _tools, _effort, _retries, _logs, _tid, _round, _queue, usage, *_a, **_k):
        calls.append(_tid)
        mid = "control"
        if reason == REASON_OWNER_REQUESTED_FINALIZATION:
            intent = cancel_intents.request_cancel(tmp_path, "t-wait", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            mid = owner_stop_control_id(intent)
        write_owner_message(tmp_path, reason, "t-wait", msg_id=mid, kind=KIND_FINALIZE_NOW)
        usage["_last_llm_error_kind"] = "transport_unavailable"
        return None, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", fail)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    text, usage, trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), []))
    assert len(calls) == 1
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == "finalize_now"
    if reason == REASON_OWNER_REQUESTED_FINALIZATION:
        assert "owner requested Wrap up" in text and "own limits ran out" not in text
        assert "no new summary request was sent" in usage["terminal_provider_notice"]
    else:
        assert "owner requested Wrap up" not in text and "provider outage" in text


def test_deadline_text_does_not_hide_an_existing_unknown_attempt():
    usage = {"_last_llm_error_kind": "provider_outcome_unknown"}
    text = loop_transport.provider_terminal_fallback_text(usage, is_context_overflow=False,
        is_transport_wait=False, waited_sec=0.0, interactive=False, is_deadline_exhausted=True)
    assert "owner deadline" in text and "no terminal provider outcome" in text
    assert "no retry or paid fallback was sent" in text
