"""A real tool-admission Stop after a model response uses the existing control exit."""
import queue
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, owner_mailbox, owner_pause, usage_accounting as ua
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import ModelWaitInterrupted, task_model_wait_scope
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.task_results import write_task_result
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
from tests._usage_store_testing import ledger_rows
from tests.test_billing_authority_unavailable import _loop_driver



@pytest.mark.parametrize("stop,pooled", [(False, False), (True, False), (True, True)])
def test_tool_gate_after_response_keeps_original_stop_without_another_send(tmp_path, monkeypatch, stop, pooled):
    tid = "tool-stop"
    write_task_result(tmp_path, tid, "running", root_task_id=tid)
    loop, registry, _ = _loop_driver(tmp_path, monkeypatch, tid)
    registry._ctx.is_direct_chat = not pooled
    registry._ctx.task_metadata = {"root_task_id": tid}
    sends, effects, gates, physical, outputs = [], [], [], [], []
    registry.override_handler("knowledge_read", lambda *_a, **_k: effects.append(True) or "read once")
    actual = owner_pause.run_tool_handler

    def gate(*args, **kwargs):
        gates.append(True)
        if stop:
            cancel_intents.request_cancel(tmp_path, tid, source="isolated-tool-gate")
            owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, tid,
                                             msg_id="original-stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
        try:
            return actual(*args, **kwargs)
        except PhysicalDispatchInterrupted as error:
            physical.append(error)
            raise

    monkeypatch.setattr(owner_pause, "run_tool_handler", gate)

    def model(call):
        first = not sends
        response = {"role": "assistant", "content": None if first else "Completed."}
        if first:
            response["tool_calls"] = [{"id": "read-1", "type": "function", "function": {
                "name": "knowledge_read", "arguments": '{"topic":"fixture"}'}}]
        ua.execute_physical_attempt(
            ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=0.1),
            lambda: sends.append(response) or response,
            extractor=lambda _response: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True))
        return response, 0.01, call.active_context_mode

    monkeypatch.setattr(loop, "_call_round_model", model)
    task = {"id": tid, "root_task_id": tid, "_attempt": 1, "_is_direct_chat": not pooled}
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=tid, root_task_id=tid,
                                    root_limit_usd=20, global_limit_usd=1000)):
        with task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None,
                                   worker_slot_held=pooled):
            def run():
                return loop.run_llm_loop(
                    messages=[{"role": "user", "content": "read the fixture once"}], tools=registry,
                    llm=SimpleNamespace(default_model=lambda: "fixture"), drive_logs=tmp_path / "logs",
                    emit_progress=lambda *a, **_k: outputs.append(a), incoming_messages=queue.Queue(),
                    task_id=tid, drive_root=tmp_path, task_type="task" if pooled else "presence")
            if pooled:
                with pytest.raises(ModelWaitInterrupted) as caught:
                    run()
                error = caught.value
                assert error.control_reason == "cancelled" and error.control_rails_seen
                while isinstance(error, ModelWaitInterrupted):
                    error = error.previous_error
                assert error is physical[0]
            else:
                _text, usage, trace = run()
                if stop:
                    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
                    assert trace["forced_finalization"]["source"] == "owner_stopped_direct_turn"
                    assert registry._ctx._skip_post_task_synthesis is True
                else:
                    assert _text == "Completed."
    assert len(sends) == (1 if stop else 2) and effects == ([] if stop else [True])
    assert gates == [True]  # no admission retry or tool re-execution
    assert len(physical) == (1 if stop else 0)
    if stop:
        assert physical[0].control_reason == "cancelled"
        assert not any("read once" in str(row) or "Completed." in str(row) for row in outputs)
    attempts = {row["attempt_id"] for row in ledger_rows(tmp_path) if row.get("kind") == "attempt"}
    assert len(attempts) == len(sends)  # original settled sends remain; no replacement generation


@pytest.mark.parametrize("kind", ["generic", "lookalike", "already_routed"])
def test_outer_exit_keeps_unrelated_errors_and_already_routed_control_identity(kind):
    from ouroboros.loop_round_limits import _loop_exit_after_exception

    error = ModelWaitInterrupted("cancelled") if kind == "already_routed" else RuntimeError("tool failed")
    error.control_rails_seen = kind == "already_routed"
    if kind == "lookalike":
        error.code, error.control_reason = "model_operation_interrupted", "cancelled"
    attached = []
    with pytest.raises(type(error)) as caught:
        _loop_exit_after_exception(error, SimpleNamespace(),
                                   SimpleNamespace(attach_exception_evidence=attached.append), {}, None)
    assert caught.value is error and attached == [error]
