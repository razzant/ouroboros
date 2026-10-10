"""Working-state fault boundaries exercised through the loop and durable owners."""
from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_working_checkpoint import _registry


@pytest.mark.parametrize("fault", ["compaction", "checkpoint_write"])
def test_loop_failure_never_acknowledges_mail_missing_from_saved_state(tmp_path, monkeypatch, fault):
    from ouroboros import loop, utils
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    write_task_result(tmp_path, "t-wait", "running")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    assert write_owner_message(tmp_path, "retain this instruction", "t-wait", msg_id="unsaved")
    real_write = utils.write_bytes_atomic

    def failed_write(path, data, **kwargs):
        if path.name.startswith(wc.FILE_PREFIX):
            raise OSError(28, "checkpoint disk full")
        return real_write(path, data, **kwargs)

    def fail_after_drain(*args, **kwargs):
        raise RuntimeError("injected loop failure")

    if fault == "compaction":
        monkeypatch.setattr(loop, "_run_round_compaction", fail_after_drain)
    else:
        monkeypatch.setattr(utils, "write_bytes_atomic", failed_write)
    monkeypatch.setattr(loop, "_dispatch_round_model", fail_after_drain)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))

    with pytest.raises(RuntimeError, match="injected loop failure"):
        loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert not wc.checkpoint_path(tmp_path, "t-wait", 1).exists()
    assert [row["msg_id"] for row in drain_owner_entries(tmp_path, "t-wait", set(), 1)] == ["unsaved"], \
        "cleanup cannot ACK content no saved continuation holds"


def test_a_reused_loop_context_cannot_acknowledge_an_unsaved_previous_turn(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    write_task_result(tmp_path, "previous", "running")
    registry = _registry(tmp_path, monkeypatch, "previous", 1)
    assert write_owner_message(tmp_path, "previous unsaved instruction", "previous", msg_id="previous-mail")
    loop._drain_incoming_messages([], _loop_kwargs(tmp_path, registry, [])["incoming_messages"],
        tmp_path, "previous", None, set(), owner_ctx=registry._ctx, defer_content_ack=True)
    registry._ctx.task_id = "t-wait"
    write_task_result(tmp_path, "t-wait", "running")
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *a, **k: (
        {"role": "assistant", "content": "new turn answer", "tool_calls": []}, 0.0))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))

    answer, _, _ = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    assert answer == "new turn answer"
    assert [row["msg_id"] for row in drain_owner_entries(tmp_path, "previous", set(), 1)] == ["previous-mail"]


@pytest.mark.parametrize("receipt", ["owed", "delivered", "delivered_without_source", "corrupt"])
def test_final_delivery_ownership_prevents_recovery_without_losing_checkpoint(tmp_path, receipt):
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "pause-task", "running")
    _, limit = _loop_ctx(tmp_path)
    wc.save_round(limit, "candidate")
    path = wc.checkpoint_path(tmp_path, "pause-task", 1)
    exact = path.read_bytes()
    text = "Already owned final answer"
    did = td.delivery_id_for("pause-task", text)
    event = {"delivery_id": did, "task_id": "pause-task", "chat_id": 1,
             "text": text, "type": "send_message"}
    if receipt == "corrupt":
        target = tmp_path / "state" / "terminal_deliveries.json"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"{interrupted registry")
    elif receipt == "owed":
        assert td.register_pending_delivery(tmp_path, event)
    else:
        assert td.register_delivery(tmp_path, did, emitted=event if receipt == "delivered" else None)

    assert wc.prepare_recovery(tmp_path, "pause-task", from_attempt=1, cause="app_stop") == {}
    assert path.read_bytes() == exact, "refused recovery keeps its source intact"


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("owner_changed", [False, True])
@pytest.mark.parametrize("interrupted_again", [False, True])
def test_candidate_is_saved_before_a_later_finalization_failure(
    tmp_path, monkeypatch, new_id, owner_changed, interrupted_again,
):
    from ouroboros import loop

    write_task_result(tmp_path, "t-wait", "running")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    answer = "Exact candidate before finalization failure"
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *a, **k: (
        {"role": "assistant", "content": answer, "tool_calls": []}, 0.0))

    def finalization_failure(*args, **kwargs):
        raise RuntimeError("finalization interrupted")

    real_gate = loop._enforce_swarm_actions
    monkeypatch.setattr(loop, "_enforce_swarm_actions", finalization_failure)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    with pytest.raises(RuntimeError, match="finalization interrupted"):
        loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))

    saved = json.loads(wc.checkpoint_path(tmp_path, "t-wait", 1).read_bytes())
    assert saved["working"]["boundary"] == "candidate"
    assert saved["delivery_candidate"]["full_text"] == answer
    handoff = wc.prepare_recovery(tmp_path, "t-wait", from_attempt=1, cause="worker_crash")
    restored = wc.load_recovery(registry._ctx, handoff)
    assert restored["delivery_candidate"] == saved["delivery_candidate"]

    task_id = "t-retry" if new_id else "t-wait"
    write_task_result(tmp_path, task_id, "running", task_attempt=2)
    registry = _registry(tmp_path, monkeypatch, task_id, 2)
    registry._ctx.working_recovery = handoff
    if interrupted_again:
        monkeypatch.setattr(loop, "_dispatch_round_model", lambda *a, **k: pytest.fail("no replacement generation"))
        retry_kwargs = {**_loop_kwargs(tmp_path, registry, []), "task_id": task_id}
        with pytest.raises(RuntimeError, match="finalization interrupted"):
            loop.run_llm_loop(**retry_kwargs)
        again = json.loads(wc.checkpoint_path(tmp_path, task_id, 2).read_bytes())
        assert again["delivery_candidate"]["full_text"] == answer
        handoff = wc.prepare_recovery(tmp_path, task_id, from_attempt=2, cause="worker_crash")
        registry = _registry(tmp_path, monkeypatch, task_id, 3)
        registry._ctx.working_recovery = handoff
    monkeypatch.setattr(loop, "_enforce_swarm_actions", real_gate)
    calls = []
    def resumed_generation(call, *args, **kwargs):
        assert owner_changed, "saved complete candidate must finalize before any replacement author generation"
        assert "Use the corrected budget" in str(call.messages)
        assert registry._ctx._delivery_candidate.full_text == answer
        assert registry._ctx._delivery_candidate.finalization_control == "owner_revision_required"
        calls.append(call)
        return {"role": "assistant", "content": "Corrected budget answer", "tool_calls": []}, 0.0
    monkeypatch.setattr(loop, "_dispatch_round_model", resumed_generation)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = task_id
    if owner_changed:
        kwargs["incoming_messages"].put("Use the corrected budget")
    recovered, _, trace = loop.run_llm_loop(**kwargs)
    assert recovered == ("Corrected budget answer" if owner_changed else answer)
    assert len(calls) == int(owner_changed)
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False


def test_failed_result_storage_keeps_work_until_real_terminal_write(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.agent import _retire_working_checkpoint
    from ouroboros.task_results import load_task_result

    write_task_result(tmp_path, "pause-task", "running", task_attempt=1)
    ctx, limit = _loop_ctx(tmp_path)
    wc.save_round(limit, "candidate")
    path = wc.checkpoint_path(tmp_path, "pause-task", 1)
    exact = path.read_bytes()
    task = {"id": "pause-task", "type": "task", "chat_id": 0, "_attempt": 1}
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    real_write = pipeline.write_task_result
    attempted = []

    def failed_result_write(*args, **kwargs):
        attempted.append(args[1])
        raise OSError(28, "terminal result disk full")

    monkeypatch.setattr(pipeline, "write_task_result", failed_result_write)
    pipeline._store_task_result(env, task, "answer", {}, {"tool_calls": [], "reasoning_notes": []})
    assert attempted == ["pause-task"], "exercise the real fail-soft result writer"
    assert load_task_result(tmp_path, "pause-task")["status"] == "running"
    _retire_working_checkpoint(ctx, task)
    assert path.read_bytes() == exact, "a returning fail-soft writer cannot retire the sole checkpoint"

    monkeypatch.setattr(pipeline, "write_task_result", real_write)
    pipeline._store_task_result(env, task, "answer", {}, {"tool_calls": [], "reasoning_notes": []})
    assert load_task_result(tmp_path, "pause-task")["status"] == "completed"
    _retire_working_checkpoint(ctx, task)
    assert not path.exists(), "a durably stored current-attempt result retires the working file"


@pytest.mark.parametrize("final_state", ["owed", "corrupt", "legacy_delivered"])
def test_retirement_requires_positive_durable_final_source(tmp_path, final_state):
    from ouroboros.agent import _retire_working_checkpoint
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "pause-task", "running", task_attempt=1)
    ctx, limit = _loop_ctx(tmp_path)
    wc.save_round(limit, "candidate")
    path = wc.checkpoint_path(tmp_path, "pause-task", 1)
    exact = path.read_bytes()
    event = {"type": "send_message", "task_id": "pause-task", "chat_id": 1,
             "text": "answer", "delivery_id": td.delivery_id_for("pause-task", "answer")}
    if final_state == "owed":
        assert td.register_pending_delivery(tmp_path, event)
    elif final_state == "legacy_delivered":
        assert td.register_delivery(tmp_path, event["delivery_id"])
    else:
        target = tmp_path / "state" / "terminal_deliveries.json"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"{broken")
    _retire_working_checkpoint(ctx, {"id": "pause-task", "_attempt": 1})
    if final_state == "owed":
        assert not path.exists()
    else:
        assert path.read_bytes() == exact, "unknown final bytes suppress replay but cannot retire its source"


@pytest.mark.parametrize("kind", ["owner_wait", "budget_pause"])
@pytest.mark.parametrize("damage", ["none", "digest", "identity"])
def test_retirement_reads_the_exact_park_before_discarding_working_fallback(tmp_path, kind, damage):
    from ouroboros import budget_pause, owner_wait
    from ouroboros.agent import _retire_working_checkpoint

    write_task_result(tmp_path, "pause-task", "running", task_attempt=1)
    ctx, limit = _loop_ctx(tmp_path)
    wc.save_round(limit, "post_batch")
    path = wc.checkpoint_path(tmp_path, "pause-task", 1)
    exact = path.read_bytes()
    if kind == "owner_wait":
        park = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 4, [], set())
        park["state"] = "waiting"
        writer, identity = owner_wait.set_owner_wait, "wait_id"
    else:
        seed = {"pause_id": "p1", "state": budget_pause.STATE_PAUSING, "reason": "budget",
                "rail": "global_exhausted", "pause_generation": 1, "scope": "global", "task_attempt": 1}
        budget_pause.set_budget_pause(tmp_path, "pause-task", seed)
        park = wc.complete_pause_from_working(tmp_path, "pause-task", seed, 1)
        writer, identity = budget_pause.set_budget_pause, "pause_id"
    if damage == "digest":
        park["source_ref"]["sha256"] = "0" * 64
    elif damage == "identity":
        park[identity] = "wrong-episode"
    writer(tmp_path, "pause-task", park)
    _retire_working_checkpoint(ctx, {"id": "pause-task", "_attempt": 1})
    if damage == "none":
        assert not path.exists(), "a positively readable park owns the saved cognition"
    else:
        assert path.read_bytes() == exact, "a locator alone cannot retire the fallback"


def test_ready_checkpoint_holds_actual_compacted_transcript_and_inline_attachment(tmp_path, monkeypatch):
    import base64
    from ouroboros import loop
    from tests.test_context_fit_v664 import _plan
    from tests.test_live_image_delivery import pixels

    image_base64 = base64.b64encode(pixels()).decode()

    write_task_result(tmp_path, "t-wait", "running")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    registry._ctx._pending_compaction = 1
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["messages"] = _plan().messages_for("max") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "old", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "prior detailed observation"}]
    kwargs["incoming_messages"].put({"text": "keep this diagram", "image_base64": image_base64,
                                      "image_mime": "image/png", "msg_id": "diagram"})

    def compact(messages, **_kwargs):
        # The compactor's provider is a test double; its real loop consumer must
        # save the returned list, including the image and source-bearing rows.
        transformed = [dict(row, content="retained observation summary")
                       if row.get("tool_call_id") == "old" else dict(row) for row in messages]
        return transformed, SimpleNamespace(status="applied", reclaimed_tokens=1,
            goal_reached=True, checkpoint_ref={"source": "test-compactor"}), None

    def crash_after_ready(*args, **kwargs):
        raise RuntimeError("interrupted after ready")

    monkeypatch.setattr(loop, "compact_tool_history_llm", compact)
    monkeypatch.setattr(loop, "_dispatch_round_model", crash_after_ready)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    with pytest.raises(RuntimeError, match="interrupted after ready"):
        loop.run_llm_loop(**kwargs)

    handoff = wc.prepare_recovery(tmp_path, "t-wait", from_attempt=1, cause="worker_crash")
    saved = wc.load_recovery(registry._ctx, handoff)
    assert saved["working"]["boundary"] == "ready"
    assert next(row for row in saved["messages"] if row.get("tool_call_id") == "old")["content"] == \
        "retained observation summary"
    images = [block for row in saved["messages"] if isinstance(row.get("content"), list)
              for block in row["content"] if block.get("type") == "image_url"]
    assert images and images[-1]["image_url"]["url"] == f"data:image/png;base64,{image_base64}"
    assert "keep this diagram" in str(saved["owner_directives"])


def test_late_owner_mail_at_delivery_close_survives_failure_of_the_next_round(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    write_task_result(tmp_path, "t-wait", "running")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    registry._ctx.owner_message_admission_lock = threading.RLock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(
        _accepting_owner_messages=True, _busy=True, _current_task_id="t-wait")
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *a, **k: (
        {"role": "assistant", "content": "candidate before new instruction", "tool_calls": []}, 0.0))

    def late_mail(**kwargs):
        assert write_owner_message(tmp_path, "late instruction", "t-wait", msg_id="late-mail")
        return False

    real_compaction = loop._run_round_compaction

    def next_round_failure(messages, ctx):
        if any("late instruction" in str(row.get("content")) for row in messages):
            raise RuntimeError("next round interrupted")
        return real_compaction(messages, ctx)

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", late_mail)
    monkeypatch.setattr(loop, "_run_round_compaction", next_round_failure)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    with pytest.raises(RuntimeError, match="next round interrupted"):
        loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    saved = json.loads(wc.checkpoint_path(tmp_path, "t-wait", 1).read_bytes())
    assert "late-mail" not in saved["seen"]
    assert [row["msg_id"] for row in drain_owner_entries(tmp_path, "t-wait", set(), 1)] == ["late-mail"]


@pytest.mark.parametrize("consumer", ["forced", "model_control", "authored_stop"])
def test_finalization_drains_defer_content_ack_until_saved(tmp_path, monkeypatch, consumer):
    from ouroboros import loop, loop_delivery, loop_forced_finalization, loop_round_limits
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    write_task_result(tmp_path, "pause-task", "running")
    ctx, limit = _loop_ctx(tmp_path)
    limit.task_id, limit.drive_root, limit.status_drive_root = "pause-task", tmp_path, tmp_path
    limit.root_task_id = "pause-task"
    limit.incoming_messages, limit.event_queue = queue.Queue(), None
    limit.llm_trace = {"tool_calls": [], "reasoning_notes": []}
    limit.owner_msg_seen = set()
    assert write_owner_message(tmp_path, "instruction during finalization", "pause-task", msg_id="final-mail")

    def fail_after_drain(*args, **kwargs):
        raise RuntimeError("finalization source consumer interrupted")

    if consumer == "forced":
        monkeypatch.setattr(loop, "_delivery_evidence_state", fail_after_drain)
        call = lambda: loop_forced_finalization._drain_forced_owner_directives(limit, limit.llm_trace)
    elif consumer == "model_control":
        monkeypatch.setattr(loop, "_finalize_forced_services", fail_after_drain)
        call = lambda: loop_round_limits._handle_model_wait_control(limit, ModelWaitInterrupted("deadline"))
    else:
        ctx._completion_request = {"action": "stop", "answer": "retained answer", "rationale": "done",
            "observation": {"owner_source_sha256": owner_source_sha256(ctx), "tool_count": 0}}
        monkeypatch.setattr(loop, "_arm_delivery_control", fail_after_drain)
        call = lambda: loop_delivery.finish_completed_stop(limit.tools, limit, lambda *a, **k: None, None, None)

    with pytest.raises(RuntimeError, match="finalization source consumer interrupted"):
        call()
    assert [row["msg_id"] for row in drain_owner_entries(tmp_path, "pause-task", set(), 1)] == ["final-mail"]
    wc.save_ready(limit)
    saved = json.loads(wc.checkpoint_path(tmp_path, "pause-task", 1).read_bytes())
    assert "final-mail" in saved["seen"]
    assert any("instruction during finalization" in str(row.get("content")) for row in saved["messages"])
    assert drain_owner_entries(tmp_path, "pause-task", set(), 1) == []


def _unread_ids(root, task_id):
    from ouroboros.task_results import load_task_result

    held = load_task_result(root, task_id).get("unread_mailbox") or {}
    return [json.loads(row)["msg_id"] for row in held.get("rows", [])]


@pytest.mark.parametrize("exit_path", ["model_call", "outer_control", "budget_hold", "admission_close",
                                       "write_fails"])
def test_a_returned_terminal_acknowledges_its_finalization_mail_only_once_saved(tmp_path, monkeypatch, exit_path):
    """R1: a terminal rail drained the owner's message into the final but never ACKed
    it, so the terminal write kept it as unread custody. Every returned exit saves it
    first: a control met in the model call, one met outside it (the outer handler's
    returned rail), a budget hold ended by control (its nested handler) and the
    final admission close. A failed save keeps it unread."""
    from ouroboros import loop, utils
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.owner_mailbox import PROVENANCE_PEER_TASK, mail_read_state, write_owner_message, write_task_message
    from ouroboros.task_results import STATUS_FAILED
    from ouroboros.usage_accounting import BudgetExceeded

    write_task_result(tmp_path, "t-wait", "running")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    text = "answer this before you stop"

    def mail():
        assert write_owner_message(tmp_path, text, "t-wait", msg_id="final-mail")

    def deadline_during_the_call(_model_call):
        mail()
        raise ModelWaitInterrupted("deadline")

    real_write = utils.write_bytes_atomic

    def failed_write(path, data, **kwargs):
        if path.name.startswith(wc.FILE_PREFIX):
            raise OSError(28, "checkpoint disk full")
        return real_write(path, data, **kwargs)

    if exit_path == "write_fails":
        monkeypatch.setattr(utils, "write_bytes_atomic", failed_write)
    if exit_path in {"model_call", "write_fails"}:
        monkeypatch.setattr(loop, "_call_round_model", deadline_during_the_call)
    elif exit_path == "outer_control":
        def deadline_outside_the_call(*_a, **_k):
            mail()
            raise ModelWaitInterrupted("deadline")

        monkeypatch.setattr(loop, "_maybe_early_finalize", deadline_outside_the_call)
    elif exit_path == "budget_hold":
        def refused_dispatch(_model_call):
            mail()
            raise BudgetExceeded("money gone")

        monkeypatch.setattr(loop, "_call_round_model", refused_dispatch)
        monkeypatch.setattr(loop, "_handle_budget_exceeded",
                            lambda *_a, **_k: (_ for _ in ()).throw(ModelWaitInterrupted("deadline")))
    else:
        registry._ctx.owner_message_admission_lock = threading.RLock()
        registry._ctx.owner_message_admission_agent = SimpleNamespace(
            _accepting_owner_messages=True, _busy=True, _current_task_id="t-wait")
        text = "a peer's context arriving at the final admission close"

        def answer_then_peer_mail(*_a, **_k):
            assert write_task_message(tmp_path, text, "t-wait", source_task_id="peer-root",
                                      provenance=PROVENANCE_PEER_TASK, msg_id="final-mail")
            return {"role": "assistant", "content": "final answer", "tool_calls": []}, 0.0

        monkeypatch.setattr(loop, "_dispatch_round_model", answer_then_peer_mail)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    outer_exits: list = []
    real_exit = loop._loop_exit_after_exception
    monkeypatch.setattr(loop, "_loop_exit_after_exception",
                        lambda exc, *a, **k: outer_exits.append(type(exc).__name__) or real_exit(exc, *a, **k))

    answer, usage, _trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    # The result returned from inside the loop's own exception handlers is still a returned result.
    assert outer_exits == (["ModelWaitInterrupted"] if exit_path in {"outer_control", "budget_hold"} else [])
    if exit_path == "admission_close":
        assert answer == "final answer"
    else:
        assert usage["reason_code"] == "deadline_local"
    write_task_result(tmp_path, "t-wait", STATUS_FAILED)  # the terminal write captures unread mail
    if exit_path == "write_fails":
        assert mail_read_state(tmp_path, "t-wait", "final-mail") is False
        assert _unread_ids(tmp_path, "t-wait") == ["final-mail"], "an unsaved read is never acknowledged"
        return
    saved = json.loads(wc.checkpoint_path(tmp_path, "t-wait", 1).read_bytes())
    assert "final-mail" in saved["seen"]
    assert any(text in str(row.get("content")) for row in saved["messages"])
    assert mail_read_state(tmp_path, "t-wait", "final-mail") is True
    assert _unread_ids(tmp_path, "t-wait") == []


@pytest.mark.parametrize("ending", ["durable_pause", "source_unwritten_then_stop", "row_unpublished_then_stop"])
def test_an_exact_pause_acknowledges_the_mail_its_durable_source_holds(tmp_path, monkeypatch, ending):
    """The same round's drain took the owner's message, then the boundary met the
    Pause: the resumed loop restores that drain's ``seen``, so only the durable
    exact source can acknowledge it, once its row is published. A source never
    written, or written but never published, acknowledges nothing."""
    from ouroboros import budget_pause, loop, owner_pause
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.owner_mailbox import mail_read_state, write_owner_message
    from ouroboros.task_results import STATUS_FAILED
    from tests._budget_pause_exact_helpers import _controls, _fast_hold, _quiet_external

    write_task_result(tmp_path, "pause-task", "running")
    ctx, limit = _loop_ctx(tmp_path)
    assert write_owner_message(tmp_path, "do the other part first", "pause-task", msg_id="pause-mail")
    owner_pause.install_fence(tmp_path, "pause-task", request_id="press")
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    loop._drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, "pause-task", None,
                                  limit.owner_msg_seen, owner_ctx=ctx, defer_content_ack=True)
    assert mail_read_state(tmp_path, "pause-task", "pause-mail") is False
    if ending != "durable_pause":
        stored: list = []
        if ending == "source_unwritten_then_stop":
            def unwritable(*_a, **_k):
                raise OSError(28, "source disk full")

            monkeypatch.setattr(budget_pause, "_exact_continuation_row", unwritable)
        else:
            real_row, real_set = budget_pause._exact_continuation_row, budget_pause.set_budget_pause
            monkeypatch.setattr(budget_pause, "_exact_continuation_row",
                                lambda *a, **k: stored.append(1) or real_row(*a, **k))

            def unpublished(root, task_id, row, **kw):
                if row.get("source_ref"):
                    raise OSError(28, "row disk full")
                return real_set(root, task_id, row, **kw)

            monkeypatch.setattr(budget_pause, "set_budget_pause", unpublished)
        monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "cancelled"))
        with pytest.raises(ModelWaitInterrupted):
            budget_pause.enter_owner_pause(limit)
        assert stored == ([1] if ending == "row_unpublished_then_stop" else []), "the source alone is not a pause"
        assert mail_read_state(tmp_path, "pause-task", "pause-mail") is False
        write_task_result(tmp_path, "pause-task", STATUS_FAILED)
        assert _unread_ids(tmp_path, "pause-task") == ["pause-mail"]
        return
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    source = json.loads(read_actor_source_bytes(tmp_path, "pause-task", raised.value.pause["source_ref"]))
    assert "pause-mail" in source["seen"]
    assert any("do the other part first" in str(row.get("content")) for row in source["messages"])
    assert mail_read_state(tmp_path, "pause-task", "pause-mail") is True


def test_a_paused_partial_batch_resumes_with_its_acked_mail_once_and_reruns_nothing(tmp_path, monkeypatch):
    """The real consumers of that ACK: the supervisor's Resume grant, the loop's restore
    and its next drain. The message is in the restored transcript exactly once and is
    never redelivered; the batch's unanswered call is closed UNKNOWN, never re-run; the
    terminal write keeps no unread copy of what the resumed work read."""
    import queue as stdqueue

    from ouroboros import budget_pause, loop, owner_wait
    from ouroboros.owner_mailbox import mail_read_state, write_owner_message
    from ouroboros.task_results import STATUS_FAILED
    from tests import _budget_pause_exact_helpers as helpers

    queue_mod, state, _workers = helpers._install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    original_ctx = helpers._loop_ctx

    def drained_before_the_pause(*args, **kwargs):
        context, limits = original_ctx(*args, **kwargs)
        assert write_owner_message(tmp_path, "keep the second call for later", "loop-1", msg_id="pause-mail")
        loop._drain_incoming_messages(limits.messages, stdqueue.Queue(), tmp_path, "loop-1", None,
                                      limits.owner_msg_seen, owner_ctx=context, defer_content_ack=True)
        assert mail_read_state(tmp_path, "loop-1", "pause-mail") is False
        return context, limits

    monkeypatch.setattr(helpers, "_loop_ctx", drained_before_the_pause)
    task, row = helpers._parked(tmp_path, monkeypatch, task_id="loop-1")
    monkeypatch.setattr(helpers, "_loop_ctx", original_ctx)
    assert row["resume_point"]["phase"] == "partial_tool_batch_unknown"
    assert row["resume_point"]["unanswered_tool_call_ids"] == ["call_b"]
    assert mail_read_state(tmp_path, "loop-1", "pause-mail") is True

    assert queue_mod.resume_budget_paused_task("loop-1")["ok"] is True
    ctx, _limit = original_ctx(tmp_path, "loop-1")
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, trace, usage, seen = [], {}, {}, set()
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), budget_pause.load_budget_pause(ctx),
                                    messages, trace, usage, seen, budget_remaining_usd=5.0)
    assert sum("keep the second call for later" in str(row.get("content")) for row in messages) == 1
    unknown = [row for row in messages if row.get("role") == "tool" and row.get("tool_call_id") == "call_b"]
    assert len(unknown) == 1 and "NOT re-executed" in unknown[0]["content"]
    again: list = []
    loop._drain_incoming_messages(again, stdqueue.Queue(), tmp_path, "loop-1", None, seen,
                                  owner_ctx=ctx, defer_content_ack=True)
    assert again == [] and not getattr(ctx, "_pending_content_acks", None)
    write_task_result(tmp_path, "loop-1", STATUS_FAILED)
    assert _unread_ids(tmp_path, "loop-1") == []
