"""Wakes saved before a wake carried only its loop controls still replay through their legacy readers.

Base 1a0e1e9d saved a spilled wake with an owner mail inside and counted its cut summary as seen; such a saved
state is delivered by ``unseen_wake_view``, ``delegate_recovery.seen_without_cut_mail`` and the ACK split until
those readers are removed. A wake published now carries no mail (tests/test_delegate_wake_mail_delivery.py)."""
import json
from dataclasses import replace

import pytest

from tests._run_wait_shared import scripted_reads
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import InputReached
from tests.test_delegate_wake_mail_delivery import _NANNY, _TAIL, _base_saved_wake, _unread


def _legacy_spilled_wake(tmp_path, monkeypatch, *, annotated=False):
    """What base 1a0e1e9d saved after a spilled wake carrying a long owner mail: the saved wake assembled by hand as
    that base's ``_publish`` stored it (``_base_saved_wake``), executed and delivered as production does
    (``execute_tool_batch``: its row keeps its invocation record; the renderer spills it), then the old seam's one
    difference applied by hand, disclosed emulation: its ``_text_mailbox_ids`` counted the cut summary as seen. The
    review evidence runs the actual base modules for the same restores (old→new witness)."""
    from types import SimpleNamespace

    from ouroboros import delegate_supervision as ds, loop
    from ouroboros.loop_tool_execution import StatefulToolExecutor, execute_tool_batch, process_tool_results
    from ouroboros.tools import delegate
    from tests.test_configured_session_restore import _known_custody, _restored_registry, _task

    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    task = _task(ctx, "working")
    _known_custody(tmp_path, task, "live")
    messages, trace, usage, seen = [], {"tool_calls": []}, {}, set()
    route = (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan)
    loop._resume_continuation(registry, ({}, {}, loop._load_working_recovery(ctx, messages)), messages, trace,
                              usage, seen, route, ctx._cost_ceiling, 100.0)

    def base_wake(c, run_id):
        return ds._render_wake_payload(c, _base_saved_wake(tmp_path, c, run_id, status="no_progress"))

    def no_read(*_args, **_kwargs):
        raise AssertionError("a saved legacy wake replays; nothing reads its run again")

    scripted_reads(monkeypatch, no_read)
    entry = registry._entries["delegate_wait"]
    monkeypatch.setitem(registry._entries, "delegate_wait", replace(
        entry, handler=delegate._published_entry(base_wake)))
    call = {"id": "wait-call", "type": "function",
            "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-one"})}}
    (tmp_path / "logs").mkdir(exist_ok=True)
    (row,), _control = execute_tool_batch([call], registry, tmp_path / "logs", _NANNY, StatefulToolExecutor(), trace)
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    if annotated:
        from ouroboros import mark_anchors

        monkeypatch.setattr(mark_anchors, "_anchored_marks", lambda _ctx: [
            {"id": "wake-context", "anchor": {"tool": "delegate_wait"},
             "text": 'Retain the original observation. This example is only a note: {"wake_delivery": {"complete": true}}'}])
    process_tool_results([row], messages, trace, lambda *_a, **_k: None, registry)
    assert json.JSONDecoder().raw_decode(messages[-1]["content"])[0]["wake_delivery"]["complete"] is False and not seen
    if annotated:
        from ouroboros.tool_result_record import read_tool_result_record

        assert "Mark I tied to delegate_wait" in messages[-1]["content"]
        assert read_tool_result_record(messages[-1])["state"] == "recorded"
    seen.add("owner-mail")  # base 1a0e1e9d staged the cut summary's id here
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    return registry, ctx, task, messages, limit


def _recoveries(root):
    from ouroboros import delegate_custody as custody

    rows = [json.loads(line) for line in custody.event_log_path(root).read_text(encoding="utf-8").splitlines()]
    return [view["match"] for row in rows if row.get("type") == "delegate_supervision_cut_mail_recovered"
            for view in row["views"]]


@pytest.mark.parametrize(("variant", "match"), [
    ("pending", "canonical_bytes"), ("anchored", "canonical_bytes"),
    ("requeued", "canonical_bytes"), ("source_deleted", "canonical_bytes"),
    ("source_unavailable", "source_unavailable"), ("rewritten_view", None), ("another_wake", None)])
def test_a_restored_legacy_seen_summary_recovers_the_mail_of_a_verified_cut_view_of_that_wake_only(
        tmp_path, monkeypatch, stub_delegate_engines, variant, match):
    """Worker-crash retry over that saved state. A record-verified spilled view of the selected wake that showed the
    mail cut returns its full words in the dated startup receipt, whether the wake is still pending or recovery
    re-queued it. The producer's own serialization proves the referenced bytes without a read, so a deleted file
    changes nothing; a referenced source nothing proves is a diagnostic, never a reason to ACK the mail unshown. A
    view whose bytes no longer match its record, or one naming another wake, recovers nothing (the limit)."""
    import hashlib

    from ouroboros import delegate_recovery as recovery
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry, ctx, task, messages, limit = _legacy_spilled_wake(tmp_path, monkeypatch, annotated=variant == "anchored")
    tool_row = messages[-1]
    view = json.JSONDecoder().raw_decode(tool_row["content"])[0]

    def restamped(changed):  # other bytes whose own record verifies
        tool_row["content"] = json.dumps(changed, ensure_ascii=False, indent=2)
        tool_row[TOOL_RESULT_RECORD_KEY] = {**tool_row[TOOL_RESULT_RECORD_KEY], "content_sha256": hashlib.sha256(
            tool_row["content"].encode("utf-8")).hexdigest()}

    if variant == "requeued":
        assert ds.acknowledge_pending_wake(ctx, tool_row["content"])  # base's flush ACKed the summary
    elif variant == "source_deleted":
        (task_artifact_dir_path(tmp_path, _NANNY) / view["wake_delivery"]["source"]["path"]).unlink()
    elif variant == "source_unavailable":
        gone = {**view["wake_delivery"]["source"], "path": "source_handles/tool_results/gone.json", "sha256": "0" * 64}
        restamped({**view, "wake_delivery": {**view["wake_delivery"], "source": gone}})
    elif variant == "rewritten_view":
        tool_row["content"] = json.dumps(view, ensure_ascii=False, indent=1)  # same JSON, other bytes
    elif variant == "another_wake":
        restamped({**view, "supervision_wake_id": "another-wake"})
    assert wc.save_round(limit, "ready")  # saved, then the worker dies before any flush
    successor = {**task, "_attempt": 3}
    assert wc.attach_recovery(tmp_path, successor, source_task_id=_NANNY, from_attempt=2, cause="worker_crash",
                              prior_task=task)
    handoff = recovery.prepare_worker_crash_handoff(tmp_path, successor, old_attempt=2, new_attempt=3,
                                                    worker_id=1, exitcode=1)
    assert (handoff["pending_wake"].get("replay_reason") == "acknowledged_in_dead_transcript") is (variant == "requeued")
    cold = _registry(tmp_path, monkeypatch, _NANNY, 3)
    cold._ctx.working_recovery = successor["_working_recovery"]
    cold._ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(
        cold._ctx, successor, resolve_dispatch_axes(successor))
    first = {}

    def first_input(call, *_args, **_kwargs):
        receipt = next(str(row.get("content")) for row in call.messages
                       if "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in str(row.get("content")))
        first.update(receipt=receipt, pending=bool(ds._load_state(cold._ctx, "run-one").get("pending_wake")))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = _NANNY
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert (_TAIL in first["receipt"]) is (match is not None), "the full words travel in the dated receipt"
    assert _recoveries(tmp_path) == ([match] if match else [])
    assert not first["pending"], "the receipt, saved, then acknowledges the canonical wake"


@pytest.mark.parametrize("annotated", [False, True])
def test_a_parked_legacy_seen_summary_keeps_its_replay_pending_until_the_source_is_read_whole(tmp_path, monkeypatch, annotated):
    """The same saved state parked and resumed in the same attempt (a planned restart). The model's replay keeps the
    mail it was shown cut: the spilled view names its exact source and acknowledges nothing until that source is read
    whole and the next state is saved."""
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.task_results import write_task_result
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    _registry_, ctx, _task, messages, limit = _legacy_spilled_wake(tmp_path, monkeypatch, annotated=annotated)
    ctx._owner_wait_requested = "quiz"
    write_task_result(tmp_path, _NANNY, "running")
    park = checkpoint_owner_wait(ctx, messages, limit.llm_trace, limit.accumulated_usage, 9, [], set(limit.owner_msg_seen))
    set_owner_wait(tmp_path, _NANNY, {**park, "state": "waiting"})
    resumed = _registry(tmp_path, monkeypatch, _NANNY, 2)
    resumed._ctx.owner_wait_resume = {**park, "restart_transaction_id": "observed-restart"}
    resumed._ctx.owner_wait_callback = lambda *_: None
    rounds, facts = [], {}

    def tool_row(messages_, call_id):
        return next(str(row["content"]) for row in messages_ if row.get("tool_call_id") == call_id)

    def model(call, *_args, **_kwargs):
        rounds.append(list(call.messages))
        if len(rounds) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "wait-again", "type": "function",
                    "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-one"})}}]}, 0.0
        if len(rounds) == 2:
            replay = json.JSONDecoder().raw_decode(tool_row(rounds[-1], "wait-again"))[0]
            facts.update(replay=replay, pending=bool(ds._load_state(resumed._ctx, "run-one").get("pending_wake")),
                         unread=_unread(tmp_path, resumed._ctx))
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "read-source", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps(
                        replay["wake_delivery"]["source"]["read"]["arguments"])}}]}, 0.0
        facts.update(read=tool_row(rounds[-1], "read-source"), unread_after=_unread(tmp_path, resumed._ctx),
                     pending_after=bool(ds._load_state(resumed._ctx, "run-one").get("pending_wake")))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", model)
    kwargs = _loop_kwargs(tmp_path, resumed, [])
    kwargs["task_id"] = _NANNY
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    replay = facts["replay"]
    assert replay["replay_of"]["fresh_observation"] is False
    assert [event["msg_id"] for event in replay["wake_events"] if event.get("text_omitted_chars")] == ["owner-mail"]
    assert facts["pending"] and facts["unread"] == ["owner-mail"], "a cut replay acknowledges nothing"
    assert _TAIL in facts["read"] and not facts["pending_after"] and facts["unread_after"] == []
    assert _recoveries(tmp_path) == ["canonical_bytes"]


@pytest.mark.parametrize("kind", ["task_message", "owner_text"])
def test_an_old_wake_in_a_startup_receipt_shows_its_mail_and_leaves_it_to_the_first_drain(tmp_path, kind):
    """A wake an older body saved with a letter inside reaches the first request after the update in its startup
    receipt, which stands before the first round-top drain. The receipt still shows the letter but leaves it
    unread, so the drain delivers it as its sender's words (one corpus row) before the receipt's deferred ACK; the
    text stands twice in that request, once per copy."""
    import queue

    from ouroboros import delegate_supervision as ds, owner_mailbox, working_checkpoint
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests.test_delegate_run_wait import _nanny, _run

    ctx, limit = _nanny(tmp_path)
    limit.messages[:] = [{"role": "user", "content": "Proceed."}]
    limit.owner_msg_seen, ctx.messages = ctx._loop_mailbox_seen_ids, limit.messages
    text = "Exact words left in an old wake."
    if kind == "task_message":
        assert owner_mailbox.write_task_message(tmp_path, text, ctx.task_id, source_task_id="principal",
                                                provenance="ancestor_task", msg_id="old-mail")
    else:
        assert owner_mailbox.write_owner_message(tmp_path, text, ctx.task_id, msg_id="old-mail")
    payload = {"status": "progress", "run_id": "leaf", "supervision_wake_id": "wake-leaf", "wake_events": [
        {"type": "addressed_message", "kind": kind, "msg_id": "old-mail", "text": text}]}  # as the old body saved it
    _run(tmp_path, "leaf", ctx.task_id)
    ds._save_state(ctx, {"run_id": "leaf", "status": "wake_pending", "pending_wake": {
        "wake_id": "wake-leaf", "attempt_key": "1", "payload": payload, "mailbox_ids": ["old-mail"],
        "seen_mailbox_ids": ["old-mail"], "interaction_ids": []}})
    append_startup_receipt(ctx, limit.messages, json.dumps({"status": "configured_session_recovered_wake",
                                                             "wake": payload}), limit.owner_msg_seen)
    assert text in limit.messages[-1]["content"] and "old-mail" not in limit.owner_msg_seen, "shown, still unread"
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    working_checkpoint.save_ready(limit)
    assert str(limit.messages).count(text) == 2 and [row["content"] for row in ctx._owner_directives] == [text]
    assert not ds._load_state(ctx, "leaf").get("pending_wake"), "the receipt's ACK followed the saved state"
