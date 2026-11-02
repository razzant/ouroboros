"""Wake text reaches the real loop once, with durable ACK after incorporation."""
import json
import queue
from dataclasses import replace

import pytest

from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import InputReached

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("origin", ["owner", "parent", "relay"])
@pytest.mark.parametrize("delivery", ["live_wait", "late_bind"])
def test_wake_mail_reaches_next_real_model_input_once(tmp_path, monkeypatch, origin, delivery):
    from ouroboros import delegate_supervision as ds, loop, loop_round_limits, working_checkpoint as wc
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message, write_task_message
    from ouroboros.tools import delegate
    from tests.test_delegate_pending_recovery import _pending_handoff
    from tests.test_delegate_recovery import _bound
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "nanny", 2)
    ctx = registry._ctx
    text = f"Unique {origin} correction carried by the wake"
    msg_id, run_id = "correction-mail", "bound-run"
    events, directive_calls, stamps = queue.Queue(), [], []
    record = loop._record_owner_directive
    stamp = loop_round_limits._stamp_owner_delivery

    def record_directive(*args, **kwargs):
        if kwargs.get("msg_id") == msg_id:
            directive_calls.append(kwargs)
        return record(*args, **kwargs)

    def stamp_delivery(*args, **kwargs):
        if kwargs.get("msg_id") == msg_id:
            stamps.append(kwargs)
        return stamp(*args, **kwargs)

    monkeypatch.setattr(loop, "_record_owner_directive", record_directive)
    monkeypatch.setattr(loop_round_limits, "_stamp_owner_delivery", stamp_delivery)

    def post_mail():
        if origin == "owner":
            assert write_owner_message(tmp_path, text, ctx.task_id, msg_id=msg_id)
        else:
            assert write_task_message(tmp_path, text, ctx.task_id, msg_id=msg_id,
                                      source_task_id="parent", provenance=(
                                          "ancestor_task" if origin == "parent" else "peer_via_ancestor"),
                                      relayed_from_task_id="sibling" if origin == "relay" else "")

    if delivery == "late_bind":
        _, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, ctx.task_id)
        task["_working_recovery"] = {}
        post_mail()
        wake = {"status": "waiting_on_user", "run_id": run_id, "supervision_wake_id": "old-wake",
                "wake_events": ds._addressed_wakes(ctx, {})}
        row = recovery._read(tmp_path, ctx.task_id)
        row["pending_wake"] = {"wake_id": "old-wake", "attempt_key": "1", "payload": wake,
                               "mailbox_ids": [msg_id]}
        recovery._write(tmp_path, row)
        assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"

    def wait_once(_ctx, rid, *_args, **_kwargs):
        assert delivery == "live_wait", "late binding must replay the retained wake before observing"
        post_mail()
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})

    def wait_entry(ctx, run_id):
        return ds.supervised_wait(ctx, run_id, wait_once=wait_once)

    # Only the physical observer is replaced. Registry publication, supervision,
    # result delivery, next-round drain and checkpoint/ACK are production paths.
    entry = registry._entries["delegate_wait"]
    monkeypatch.setitem(registry._entries, "delegate_wait", replace(
        entry, handler=delegate._published_entry(wait_entry)))
    ack = ds.acknowledge_pending_wake
    acknowledged = []

    def acknowledge(actual_ctx, delivered=None):
        saved = wc.read_working(tmp_path, ctx.task_id, 2)[1]
        assert any(text in str(row.get("content")) for row in saved["messages"])
        acknowledged.append(True)
        return ack(actual_ctx, delivered)

    monkeypatch.setattr(ds, "acknowledge_pending_wake", acknowledge)
    drain = loop._drain_incoming_messages
    pending_drains = []

    def drain_messages(*args, **kwargs):
        if ds._load_state(ctx, "").get("pending_wake"):
            assert not acknowledged, "no durable wake ACK before the incorporated boundary"
            assert msg_id in ctx._loop_mailbox_seen_ids
            if delivery == "live_wait":
                assert drain_owner_entries(tmp_path, ctx.task_id, set(), 2), "not durably ACKed yet"
            pending_drains.append(True)
        return drain(*args, **kwargs)

    monkeypatch.setattr(loop, "_drain_incoming_messages", drain_messages)
    rounds = []

    def model_input(call, *_args, **_kwargs):
        rounds.append(call.messages)
        if len(rounds) == 1:
            if delivery == "late_bind":
                assert sum(str(row.get("content") or "").count(text) for row in call.messages) == 1
                _bound(tmp_path, task, invocation)
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "wait", "type": "function", "function": {
                    "name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]}, 0.0
        assert sum(str(row.get("content") or "").count(text) for row in call.messages) == 1
        assert len(directive_calls) == int(delivery == "late_bind")
        assert len(stamps) == int(delivery == "late_bind" and origin == "owner")
        if directive_calls and origin != "owner":
            assert directive_calls[0]["origin"]["source_task_id"] == "parent"
            assert directive_calls[0]["origin"]["relayed_from_task_id"] == ("sibling" if origin == "relay" else "")
        injected = [event for event in list(events.queue)
                    if event.get("type") == "owner_message_injected" and event.get("text") == text]
        assert len(injected) <= int(delivery == "late_bind")
        assert pending_drains and acknowledged
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", model_input)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs.update(task_id=ctx.task_id, event_queue=events)
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert len(rounds) == 2


@pytest.mark.parametrize("delivery", ["tool", "startup"])
def test_only_shown_non_control_mail_is_seen_before_save(tmp_path, monkeypatch, delivery):
    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import KIND_HURRY
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests.test_sleep_questions import _deliver
    from tests.test_working_checkpoint import _registry

    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    seen = ctx._loop_mailbox_seen_ids = {"already-shown"}
    payload = {"status": "waiting_on_user", "supervision_wake_id": "wake", "wake_events": [
        {"msg_id": "shown", "kind": "owner_text", "text": "shown words"},
        {"msg_id": "omitted", "kind": "owner_text"},
        {"msg_id": "control", "kind": KIND_HURRY, "text": "structural control"}]}
    ds._save_state(ctx, {"run_id": "run", "pending_wake": {
        "wake_id": "wake", "payload": payload, "mailbox_ids": ["shown", "omitted", "control", "absent"]}})
    if delivery == "tool":
        _deliver(ctx, json.dumps(payload), monkeypatch, tool="delegate_wait", meta={"supervision_wake_id": "wake"})
    else:
        append_startup_receipt(ctx, [], json.dumps({"status": "configured_session_continuation", "wake": payload}), seen)
    assert ctx._loop_mailbox_seen_ids is seen
    assert seen == {"already-shown", "shown"}
    assert ds._load_state(ctx, "")["pending_wake"], "only an incorporated save may flush the ACK"


@pytest.mark.parametrize("case", ["incorporated", "unsaved", "unusable_checkpoint"])
def test_crash_handoff_restores_wake_mail_once_and_keeps_later_correction_latest(
        tmp_path, monkeypatch, stub_delegate_engines, case):
    """Real wait, save, freeze, crash handoff, bootstrap and loop: the startup receipt
    renders one wake copy and omits mail the restored transcript already holds."""
    from types import SimpleNamespace

    from ouroboros import delegate_recovery as recovery, delegate_supervision as ds, loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.owner_mailbox import write_owner_message
    from tests.test_configured_session_restore import _known_custody, _restored_registry, _task
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_sleep_questions import _deliver
    from tests.test_working_checkpoint import _registry

    task_id, run_id = "nanny", "run-one"
    old, correction = "Old instruction: paint it blue", "Later correction: paint it green"
    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    task = _task(ctx, "working")
    _known_custody(tmp_path, task, "live")
    messages, trace, usage, seen = [], {"tool_calls": []}, {}, set()
    route = (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan)
    loop._resume_continuation(registry, ({}, {}, loop._load_working_recovery(ctx, messages)), messages, trace,
                              usage, seen, route, ctx._cost_ceiling, 100.0)
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    if case == "unsaved":
        wc.save_ready(limit)  # The newest saved state predates the wake below.

    def observe(_ctx, rid, *_args, **_kwargs):
        assert write_owner_message(tmp_path, old, task_id, msg_id="old-mail")
        return json.dumps({"status": "no_progress", "run_id": rid})

    wake = ds.supervised_wait(ctx, run_id, wait_once=observe)
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "wait-call", "type": "function",
                     "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]})
    messages.extend(_deliver(ctx, wake.text, monkeypatch, tool="delegate_wait", meta=wake.meta))
    if case != "unsaved":
        wc.save_ready(limit)  # Incorporated: the wake and its mail are ACKed only after this save.
        assert ds._load_state(ctx, run_id)["last_acknowledged_wake"]["wake_id"]
        assert write_owner_message(tmp_path, correction, task_id, msg_id="correction")
        loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, task_id, None, seen,
                                      owner_ctx=ctx, defer_content_ack=True)
        wc.save_ready(limit)
    engine_calls = list(stub_delegate_engines)
    # The worker dies: the existing retry freezes saved work, then reserves the exact run.
    successor = {**task, "_attempt": 3}
    assert wc.attach_recovery(tmp_path, successor, source_task_id=task_id, from_attempt=2,
                              cause="worker_crash", prior_task=task)
    if case == "unusable_checkpoint":
        successor["_working_recovery"] = {"source_ref": {"path": "missing"}}
    handoff = recovery.prepare_worker_crash_handoff(tmp_path, successor, old_attempt=2, new_attempt=3,
                                                    worker_id=1, exitcode=1)
    assert handoff["run_id"] == run_id
    replayed = handoff["pending_wake"].get("replay_reason") == "acknowledged_in_dead_transcript"
    assert replayed is (case != "unsaved")
    cold = _registry(tmp_path, monkeypatch, task_id, 3)
    cctx = cold._ctx
    cctx.working_recovery = successor["_working_recovery"]
    cctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(
        cctx, successor, resolve_dispatch_axes(successor))
    canonical = json.loads(cctx._configured_startup_wake)
    assert canonical["status"] == "configured_session_continuation"
    assert canonical["wake"]["wake_events"][0]["text"] == old, "the deferred ACK keeps its canonical source"

    def first_input(call, *_args, **_kwargs):
        texts = [str(row.get("content") or "") for row in call.messages]
        joined = "\n".join(texts)
        receipt = next(text for text in texts if "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in text)
        assert joined.count(old) == 1, "restored or replayed, the old mail reaches the model once"
        assert receipt.count('"supervision_wake_id"') == 1, "one wake copy"
        assert (old in receipt) is (case != "incorporated")
        if case != "unsaved":
            assert joined.rindex(correction) > joined.rindex(old), "the later correction stays latest"
        assert not ds._load_state(cctx, run_id)["pending_wake"], "the ready save ACKed the canonical wake"
        assert stub_delegate_engines == engine_calls
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = task_id
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
