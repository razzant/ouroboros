"""Mail that ends a wait reaches the real loop once, as its sender's words from the round-top drain, ACKed after a
saved state holds it; a wake carries only its loop controls, and wakes saved before keep their readers."""
import json
import queue

import pytest

from tests._run_wait_shared import held_run, publish, scripted_reads
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import InputReached
from tests.test_delegate_run_wait import reads as reads



@pytest.mark.parametrize("origin", ["owner", "parent", "relay"])
@pytest.mark.parametrize("delivery", ["live_wait", "late_bind"])
def test_wake_mail_reaches_next_real_model_input_once(tmp_path, monkeypatch, origin, delivery):
    """The home of both sides of the wake contract, on the real loop, wait and drain. Mail that ends a wait reaches
    the next real model input once as its sender's words: the round-top drain's row, directive and (the owner's)
    delivery stamp, durably ACKed only by the saved state that holds it. A loop control posted with that mail rides
    in the published wake and the drain applies it: the wake carries the control and never the mail (``live_wait``).
    A wake saved before this contract (``late_bind``, built by hand: the mail inside ``wake_events``) replays
    without the mail the first round's drain already delivered."""
    from ouroboros import delegate_supervision as ds, loop, loop_round_limits, owner_hurry, working_checkpoint as wc
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.owner_mailbox import KIND_HURRY, drain_owner_entries, write_owner_message, write_task_message
    from ouroboros.utils import utc_now_iso
    from tests.test_delegate_pending_recovery import _pending_handoff
    from tests.test_delegate_recovery import _bound
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "nanny", 2)
    ctx = registry._ctx
    text = f"Unique {origin} correction that ends the wait"
    msg_id, hurry_id, run_id = "correction-mail", "hurry:posted-with-the-mail", "bound-run"
    provenance = {"owner": "owner", "parent": "ancestor_task", "relay": "peer_via_ancestor"}[origin]
    events, directive_calls, stamps, drained = queue.Queue(), [], [], []
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
            assert write_task_message(tmp_path, text, ctx.task_id, msg_id=msg_id, source_task_id="parent",
                                      provenance=provenance,
                                      relayed_from_task_id="sibling" if origin == "relay" else "")
        assert write_owner_message(tmp_path, "owner_hurry", ctx.task_id, msg_id=hurry_id, kind=KIND_HURRY)

    def unread():
        return [entry["msg_id"] for entry in drain_owner_entries(tmp_path, ctx.task_id, set(), 2)]

    if delivery == "late_bind":
        _, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, ctx.task_id)
        task["_working_recovery"] = {}
        post_mail()
        # The wake an earlier producer saved: every addressed kind rode in it, the mail beside the control, and its
        # record owed the mail's ACK.
        legacy_events = [
            {"type": "addressed_message", "msg_id": msg_id,
             "kind": "owner_text" if origin == "owner" else "task_message",
             "provenance": provenance, "source_task_id": "" if origin == "owner" else "parent",
             "relayed_from_task_id": "sibling" if origin == "relay" else "", "text": text, "ts": utc_now_iso()},
            {"type": "addressed_message", "msg_id": hurry_id, "kind": KIND_HURRY, "provenance": "owner",
             "source_task_id": "", "relayed_from_task_id": "", "text": "owner_hurry", "ts": utc_now_iso()}]
        wake = {"status": "waiting_on_user", "run_id": run_id, "supervision_wake_id": "old-wake",
                "wake_events": legacy_events}
        row = recovery._read(tmp_path, ctx.task_id)
        row["pending_wake"] = {"wake_id": "old-wake", "attempt_key": "1", "payload": wake,
                               "mailbox_ids": [msg_id]}
        recovery._write(tmp_path, row)
        assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"

    def wait_once(_ctx, rid, *_args, **_kwargs):
        assert delivery == "live_wait", "late binding must replay the retained wake before observing"
        post_mail()
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})

    # Only the one addressed read is replaced. Registry publication, the wait's readiness (the run's terminal the body
    # recorded), result delivery, next-round drain and checkpoint/ACK are production paths.
    scripted_reads(monkeypatch, wait_once)
    if delivery == "live_wait":
        held_run(tmp_path, run_id, ctx.task_id, ended="succeeded")
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
        pending = bool(ds._load_state(ctx, "").get("pending_wake"))
        if pending:
            assert not acknowledged, "no durable wake ACK before the incorporated boundary"
            if not pending_drains:  # the first round top after the wake (the batch's tool tail)
                # The published wake took no mail: unread and unseen, it is this drain's. The wake saved before
                # held it, and the first round's drain delivered it before that wake's replay.
                seen = getattr(ctx, "_loop_mailbox_seen_ids", None) or set()
                assert (msg_id in seen) is (delivery == "late_bind"), "the wake neither showed nor staged the mail"
                assert (msg_id in unread()) is (delivery == "live_wait"), "the wake did not acknowledge the mail"
        controls = drain(*args, **kwargs)
        drained.append(controls)
        if pending:
            assert msg_id in ctx._loop_mailbox_seen_ids
            assert (msg_id in unread()) is (delivery == "live_wait"), "seen now, durably ACKed only by the save"
            pending_drains.append(True)
        return controls

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
        # The words: one row of them, the drain's typed row of its sender's words, recorded once, and the saved state
        # before this request acknowledged them.
        holding = [row for row in call.messages if text in str(row.get("content") or "")]
        assert [str(row["content"]).count(text) for row in holding] == [1], "the words reach the model once"
        assert holding[0]["role"] == "user" and holding[0].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND
        assert len(directive_calls) == 1
        assert len(stamps) == int(origin == "owner")
        if origin != "owner":
            assert directive_calls[0]["origin"]["source_task_id"] == "parent"
            assert directive_calls[0]["origin"]["relayed_from_task_id"] == ("sibling" if origin == "relay" else "")
        injected = [event for event in list(events.queue)
                    if event.get("type") == "owner_message_injected" and event.get("text") == text]
        assert len(injected) == int(origin == "owner")
        assert pending_drains and acknowledged and msg_id not in unread()
        # The control: it rides in the wake the model was shown, and the drain applied it once.
        wake_row = next(row for row in call.messages if row.get("tool_call_id") == "wait")
        carried = json.JSONDecoder().raw_decode(str(wake_row["content"]))[0].get("wake_events") or []
        assert [event.get("msg_id") for event in carried] == ([hurry_id] if delivery == "live_wait" else []), \
            "a wake carries its loop controls, never the mail"
        assert [controls["hurry"] for controls in drained if controls.get("hurry")] == [hurry_id]
        assert owner_hurry.latched(ctx)["msg_id"] == hurry_id
        assert sum(str(row.get("content") or "").count(owner_hurry.HURRY_WORDS) for row in call.messages) == 1
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
    assert seen == ({"already-shown", "shown"} if delivery == "tool" else {"already-shown"})  # startup: the drain's
    assert ds._load_state(ctx, "")["pending_wake"], "only an incorporated save may flush the ACK"


@pytest.mark.parametrize("case", ["incorporated", "unsaved", "unusable_checkpoint"])
def test_crash_handoff_restores_wake_mail_once_and_keeps_later_correction_latest(
        tmp_path, monkeypatch, stub_delegate_engines, case):
    """Real wait, drain, save, freeze, crash handoff, bootstrap and loop: the startup receipt renders one wake copy,
    which carries no mail. The mail that ended the wait reaches the successor's first model input once, as the
    owner's words a round-top drain delivered (restored with the transcript, or drained again from the mailbox an
    unsaved or unusable state never acknowledged for this attempt), and the later correction stays latest."""
    from types import SimpleNamespace

    from ouroboros import delegate_recovery as recovery, delegate_supervision as ds, loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
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

    def round_top_drain():  # the loop's own mailbox drain; the next saved state acknowledges what it delivered
        loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, task_id, None, seen,
                                      owner_ctx=ctx, defer_content_ack=True)

    scripted_reads(monkeypatch, observe)
    wake = publish(ctx, run_id)
    assert old not in wake.text, "the wake carries no mail: the old mail is the drain's"
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "wait-call", "type": "function",
                     "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]})
    messages.extend(_deliver(ctx, wake.text, monkeypatch, tool="delegate_wait", meta=wake.meta))
    round_top_drain()  # the next round top: the old mail arrives as the owner's words
    if case != "unsaved":
        wc.save_ready(limit)  # Incorporated: the wake and the drained mail are ACKed only after this save.
        assert ds._load_state(ctx, run_id)["last_acknowledged_wake"]["wake_id"]
        assert write_owner_message(tmp_path, correction, task_id, msg_id="correction")
        round_top_drain()
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
    assert canonical["wake"]["supervision_wake_id"] and old not in cctx._configured_startup_wake, \
        "the canonical wake the deferred ACK keeps carries no mail"
    assert stub_delegate_engines == engine_calls, "the handoff and the bootstrap observe no engine"

    def first_input(call, *_args, **_kwargs):
        texts = [str(row.get("content") or "") for row in call.messages]
        joined = "\n".join(texts)
        receipt = next(text for text in texts if "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in text)
        holding = [row for row in call.messages if old in str(row.get("content") or "")]
        assert joined.count(old) == 1 and len(holding) == 1, "restored or drained again, the old mail arrives once"
        assert holding[0].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND and old not in receipt, \
            "as the owner's words a drain delivered, never inside the startup receipt"
        assert receipt.count('"supervision_wake_id"') == 1, "one wake copy"
        if case != "unsaved":
            assert joined.rindex(correction) > joined.rindex(old), "the later correction stays latest"
        assert not ds._load_state(cctx, run_id)["pending_wake"], "the ready save ACKed the canonical wake"
        # The one contact since the handoff is the starter's round-top placement of the owner's rows (restored with
        # the transcript, or drained again from the mailbox) into the run it still holds open. The mail that ended
        # the wait is one of those rows now, a corpus row the drain recorded, never a wake's tool result, so the
        # unsaved case owes it too. The undiscovered engine leaves the rows for the next round and this first round
        # goes on to its model call.
        assert stub_delegate_engines == engine_calls + ["gateway"]
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = task_id
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)


# --- the owner's answer to a card while the asking work waits on its delegate ---------------------------------

@pytest.mark.parametrize("wait", ["ready", "in_slot"])
def test_an_answer_to_a_card_that_arrives_while_the_work_waits_becomes_the_owners_words_for_it(
        tmp_path, monkeypatch, reads, wait):
    """The owner answers the work's card after its round top, while the model thinks or while the work waits in its
    slot with no continuation owner. The wait still ends at once, its wake carries none of the answer, and the next
    round top makes it the owner's words for the work: the corpus row reviewers read, a quote handle held on its
    record and the carriage to its live member (13 Nov: a wake read the answer itself and none of these held)."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_relay import quote_rows
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.working_checkpoint import flush_content_acks
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_delegate_run_wait import _ended, _run
    from tests.test_question_answer_routing import answer, runtime

    app, _bridge, _q, _task = runtime(tmp_path, monkeypatch)
    write_task_result(tmp_path, "member", "running", delegation_role="subagent", parent_task_id="asker",
                      root_task_id="asker")
    ctx, _limit = _loop_ctx(tmp_path, "asker")
    ctx.budget_drive_root, ctx.task_metadata, ctx._loop_mailbox_seen_ids = tmp_path, {"root_task_id": "asker"}, set()
    ctx.current_chat_id, ctx.pending_events, ctx.drive_logs = 1, [], lambda: tmp_path / "logs"
    (tmp_path / "logs").mkdir(exist_ok=True)
    if wait == "in_slot":
        ctx.owner_wait_callback = None  # no continuation owner: the wait holds its slot (_wait_in_slot)
    _run(tmp_path, "run-1", "asker")
    _ended(tmp_path, "run-1")  # the wake publishes at once: only what it carries is under test

    def round_top():
        _drain_incoming_messages([], queue.Queue(), tmp_path, "asker", None, ctx._loop_mailbox_seen_ids,
                                 owner_ctx=ctx, defer_content_ack=True)
        flush_content_acks(ctx)

    round_top()
    assert answer(app).status_code == 200  # after this round's drain
    result = ds.run_wait(ctx, "run-1")  # the model's delegate_wait in this round's tool batch
    events = delegate_payload(result).get("wake_events") or []
    assert result.meta.get("supervision_wake_id") and not [e for e in events if e.get("type") == "addressed_message"]
    ds.deliver_wake(ctx, {"result_meta": {"tool_result_meta": result.meta}}, {"result": result.text}, result.text)
    round_top()  # the next round top

    corpus = [row["content"] for row in ctx._owner_directives if row.get("source") == "owner_quiz_answer"]
    quotes, error = quote_rows(ctx, list(ctx._held_owner_messages))
    carried = {quote.get("text") for entry in drain_owner_entries(tmp_path, "member")
               for quote in (entry.get("relay") or {}).get("quotes") or []}
    assert corpus == ["exact answer"] and not error and [quote["text"] for quote in quotes] == ["exact answer"]
    assert set((load_task_result(tmp_path, "asker") or {}).get("held_owner_messages") or {}) == set(
        ctx._held_owner_messages) and carried == {"exact answer"}


# --- #2992: a wake the transcript holds only in part ---------------------------------------------------------
# Production chain, only the wake's one read scripted: the registry's ``delegate_wait`` entry (``run_wait``), the
# loop's measured delivery (``process_tool_results``), ``save_ready`` and its deferred ACK, the next wait. Mail no
# longer enters a wake (only one saved before holds it: ``_base_saved_wake``), so a wake is long by its own facts:
# here the parent's intent note, which every wake carries whole in its coordination context.

_NANNY, _RUN, _TAIL, _WAKE_TAIL = "nanny", "run-1", "OWNER-MAIL-TAIL-7c1f", "WAKE-BODY-TAIL-3d9a"


def _nanny(tmp_path, monkeypatch, observe, runs=(_RUN,)):
    """The nanny's real ``delegate_wait``: its runs ended (the body recorded it), so each wait is ready at once;
    only the one addressed read is scripted (``observe``)."""
    from ouroboros import delegate_output
    from ouroboros.task_results import write_task_result
    from tests.test_working_checkpoint import _registry

    write_task_result(tmp_path, _NANNY, "running")
    registry = _registry(tmp_path, monkeypatch, _NANNY, 1)
    ctx = registry._ctx
    ctx.budget_drive_root, ctx.task_metadata = tmp_path, {"root_task_id": _NANNY}
    ctx._loop_mailbox_seen_ids = set()  # the loop's round-top drain and wake exposure share it
    (tmp_path / "logs").mkdir(exist_ok=True)
    for run_id in runs:
        held_run(tmp_path, run_id, _NANNY, ended="succeeded")
    scripted_reads(monkeypatch, observe)
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a fresh worker's reading ledger
    return registry, ctx


def _observer(tmp_path, polls, mail="", **first):
    """Each read answers a terminal leaf; the first one also posts ``mail`` and carries ``first``."""
    from ouroboros.owner_mailbox import write_owner_message

    def observe(_ctx, rid, *_args, **_kwargs):
        polls.append(rid)
        if len(polls) == 1 and mail:
            assert write_owner_message(tmp_path, mail, _NANNY, msg_id="owner-mail")
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded", "last_seq": len(polls),
                           **(first if len(polls) == 1 else {})})

    return observe


def _intent(ctx, words):
    """The wake's own bulk: the parent's intent note, carried whole in every wake's coordination context."""
    ctx.task_contract = {"delegation_budget": {"intent_note": words}}


def _spilled():
    """An intent long enough that every ``delegate_wait`` view spills the wake to its exact source."""
    return "preserve-complete-intent:" + "y" * 30_000 + _WAKE_TAIL


def _cut():
    """An intent a crowded frame cuts out of the wake's middle, short of a spill."""
    return "Parent intent: " + "keep the reviewed shape; " * 80 + _WAKE_TAIL + " " + "then hand it back; " * 40


def _base_saved_wake(root, ctx, run_id, mail="", **read):
    """A wake as base 1a0e1e9d's ``_publish`` saved it, by hand: its one read (``read``) found the owner mail (the
    long ``_mail()`` unless given), which rode inside ``wake_events`` (that base's ``_addressed_wakes`` projection),
    and its pending record owed the mail's ACK. Today's producer leaves mail to the round-top drain, so only such a
    saved wake still carries it."""
    import uuid

    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.utils import utc_now_iso

    mail = mail or _mail()
    assert write_owner_message(root, mail, _NANNY, msg_id="owner-mail")
    payload = {**read, "run_id": run_id, "wake_events": [{
        "type": "addressed_message", "msg_id": "owner-mail", "kind": "owner_text", "provenance": "owner",
        "source_task_id": "", "relayed_from_task_id": "", "text": mail, "ts": utc_now_iso()}],
        "coordination_context": ds.coordination_live_context(ctx), "leaf_live_input": "unknown",
        "supervision_wake_id": uuid.uuid4().hex}
    state = ds._load_state(ctx, run_id)
    state.update(status="wake_pending", pending_wake={
        "wake_id": payload["supervision_wake_id"], "attempt_key": ds._attempt_key(ctx), "payload": payload,
        "mailbox_ids": ["owner-mail"], "seen_mailbox_ids": ["owner-mail"], "interaction_ids": [],
        "coordination_cursor": {"attention_after_ts": "", "attention_seen": [], "children": {}},
        "created_at": utc_now_iso()})
    ds._save_state(ctx, state, "wake_publication")
    return payload


def _call(registry, tmp_path, call_id, name, args):
    from ouroboros.loop_tool_execution import _execute_single_tool

    return _execute_single_tool(registry, {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}, tmp_path / "logs", _NANNY)


def _shown(registry, row, messages, trace, fit):
    from ouroboros.loop_tool_execution import process_tool_results

    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": row["tool_call_id"], "type": "function",
                     "function": {"name": row["fn_name"], "arguments": json.dumps(row["tool_args"])}}]})
    process_tool_results([row], messages, trace, lambda *_a, **_k: None, registry, fit_candidate=fit, tool_schemas=[])
    return messages[-1]["content"], trace["tool_calls"][-1]


def _ready(registry, messages, trace):
    from types import SimpleNamespace

    from ouroboros.working_checkpoint import save_ready

    save_ready(SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage={},
                               owner_msg_seen=set(), round_idx=len(trace["tool_calls"]), tool_schemas=[],
                               budget_tail="tool"))


def _wake_state(ctx):
    """(pending wake id, last acknowledged wake id)."""
    from ouroboros import delegate_supervision as ds

    state = ds._load_state(ctx, _RUN)
    return ((state.get("pending_wake") or {}).get("wake_id", ""),
            (state.get("last_acknowledged_wake") or {}).get("wake_id", ""))


def _unread(tmp_path, ctx):
    from ouroboros.owner_mailbox import drain_owner_entries

    return [entry["msg_id"] for entry in drain_owner_entries(tmp_path, _NANNY, set(), str(ctx.task_attempt))]


def _wake_id(row):
    return row["result_meta"]["tool_result_meta"]["supervision_wake_id"]


def _page(ref, start, end):
    return {"root": ref["root"], "path": ref["path"], "start_line": 1, "max_lines": 100_000,
            "start_char": start, "max_chars": end - start}


def _refused(root):
    """(wake id, reason, delivered wake ids) of each refused acknowledgement row."""
    from ouroboros import delegate_custody as custody

    rows = [json.loads(line) for line in custody.event_log_path(root).read_text(encoding="utf-8").splitlines()]
    return [(row["wake_id"], row["reason"], row.get("delivered_wake_ids")) for row in rows
            if row.get("type") == "delegate_supervision_wake_ack_refused"]


@pytest.mark.parametrize("shape", ["outer_cut", "inner_spill"])
def test_a_cold_continuation_keeps_pending_wake_sources_and_their_read_coverage(tmp_path, monkeypatch, shape):
    """A resumed worker completes the original source without waiting again, and the ranges the saved transcript
    already holds ride with it (E3): the rest of the source completes it. No mailbox entry can hide a missing source
    map by delivering the obligation through the round-top drain."""
    from ouroboros import owner_wait

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    ctx.task_contract = {"delegation_budget": {"intent_note": "x" * (30_000 if shape == "inner_spill" else 4_000)}}
    fit, messages, trace = _wide() if shape == "inner_spill" else _crowded()
    wake = _call(registry, tmp_path, "wait-before-cold", "delegate_wait", {"run_id": _RUN})
    _shown(registry, wake, messages, trace, fit)
    sources = dict(ctx._wake_sources)
    assert len(sources) == 1
    source = next(iter(sources.values()))
    half, end = source["size"] // 2, source["size"] + 1
    fit = _wide()[0]
    _shown(registry, _call(registry, tmp_path, "read-before-cold", "read_file", _page(source, 0, half)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "")
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, trace, {}, 1, [], set())))

    fresh, resumed = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    messages, trace = [], {}
    owner_wait.restore_continuation_state(fresh, state, messages, trace, {}, set())
    _shown(fresh, _call(fresh, tmp_path, "read-second-half", "read_file", _page(source, half, end)),
           messages, trace, fit)
    assert _wake_state(resumed) == (_wake_id(wake), ""), "a complete read still waits for checkpoint"
    _ready(fresh, messages, trace)
    assert _wake_state(resumed) == ("", _wake_id(wake)), "the restored coverage and the rest of the source complete it"
    assert resumed._wake_sources == {}
    _call(fresh, tmp_path, "wait-after-cold", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2, "one fresh poll, no duplicate run"

    resumed._wake_sources = {"stale": {}}
    owner_wait.restore_continuation_state(fresh, {key: value for key, value in state.items() if key != "wake_sources"},
                                         [], {}, {}, set())
    assert resumed._wake_sources == {}, "a legacy state does not inherit another continuation's source map"


def test_a_crowded_frame_cuts_the_wake_and_only_its_source_read_whole_acknowledges_it(tmp_path, monkeypatch):
    """#2992 A/B/C shape: the measured frame cuts the wake's JSON, so its ACK is refused (fail closed); before the
    repair the next wait replayed it forever. A whole read of the exact retained source now acknowledges it at the
    next saved state and the next wait observes the run."""
    from tests.test_tool_result_delivery import _view_json

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _cut())
    fit, messages, trace = _crowded()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and _WAKE_TAIL not in shown, "premise: the frame cut the wake"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "")
    omitted = _view_json(shown)["read_omitted"]
    read = _call(registry, tmp_path, "read-1", "read_file", {key: omitted[key] for key in (
        "root", "path", "start_line", "max_lines", "start_char", "max_chars")})
    read_shown, read_row = _shown(registry, read, messages, trace, fit)
    assert not read_row.get("result_partial") and _WAKE_TAIL in read_shown, "the omitted middle arrived whole"
    assert _wake_state(ctx) == (_wake_id(wake), ""), "the ACK waits for a saved state"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake))
    again = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2 and _wake_id(again) != _wake_id(wake), "the next wait observes the run"


def test_a_spilled_wake_waits_for_its_source_and_the_next_round_drain_delivers_the_omitted_mail_first(
        tmp_path, monkeypatch):
    """Mail that arrives with a spilled wake is omitted from it: the complete view of the wake's summary holds none
    and nothing is seen or acknowledged. A replay before any read is the same canonical source, marked as a replay and
    without the mail either. The next round's mailbox drain delivers the mail first, whole and once as the owner's
    words, and its saved state acknowledges the mail alone: the wake still waits for its own source, read whole."""
    from ouroboros import loop
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, _mail()))
    _intent(ctx, _spilled())
    fit, messages, trace = _wide()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    envelope = json.loads(wake["result"])
    source = envelope["wake_delivery"]["source"]
    assert envelope["wake_delivery"]["complete"] is False
    assert envelope["wake_delivery"]["wake_events_total"] == 0 and _TAIL not in wake["result"], \
        "the mail never rides in the wake, not even in its spilled source"
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert not row.get("result_partial") and _WAKE_TAIL not in shown, "premise: a complete view of a summary"
    assert "owner-mail" not in ctx._loop_mailbox_seen_ids, "nothing of the mail is seen"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]

    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    replayed = json.loads(replay["result"])
    assert len(polls) == 1 and _wake_id(replay) == _wake_id(wake), "replayed before any new observation"
    assert replayed["replay_of"]["fresh_observation"] is False and replayed["replay_of"]["published_at"]
    assert replayed["wake_delivery"]["source"] == source, "the same canonical bytes: earlier reading still counts"
    assert replayed["wake_delivery"]["wake_events_total"] == 0 and _TAIL not in replay["result"], \
        "nor does the mail ride in a replay"
    _shown(registry, replay, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]

    seen = ctx._loop_mailbox_seen_ids
    loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, _NANNY, None, seen,
                                  owner_ctx=ctx, defer_content_ack=True)
    holding = [row for row in messages if _TAIL in str(row.get("content") or "")]
    assert len(holding) == 1 and holding[0].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND and "owner-mail" in seen, \
        "the round-top drain delivered it whole, once, as the owner's words"
    _ready(registry, messages, trace)
    assert _unread(tmp_path, ctx) == [] and _wake_state(ctx) == (_wake_id(wake), ""), "its save ACKed the mail only"

    again = _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 1 and _wake_id(again) == _wake_id(wake), "the wake still waits for its own source"
    assert json.loads(again["result"])["wake_delivery"]["source"] == source
    _shown(registry, again, messages, trace, fit)
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", _page(source, 0, source["size"] + 1)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake))
    assert len(polls) == 1
    _call(registry, tmp_path, "wait-4", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2, "an acknowledged wake lets the next wait observe"
    assert wake["result_meta"]["tool_result_meta"]["wake_source"] == source, "the spilled source rides typed meta"


def test_a_spilled_wake_with_mail_and_a_question_counts_only_what_was_read_of_that_exact_source(
        tmp_path, monkeypatch):
    """A question waits for the spilled source read whole: half of it shows and acknowledges nothing, the same bytes
    under another name credit nothing, and the remaining half completes it. The mail that arrived with the wake is in
    no source of it: the whole read stages none of it as seen, and the wake's ACK leaves it unread for the drain."""
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts

    polls = []
    question = {"interaction_id": "q1", "questions": [{"id": "choice", "question": "Which path?"}]}
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(
        tmp_path, polls, _mail(), status="waiting_on_user", state="running", waiting_on_user=True,
        pending_interactions=[question]))
    _intent(ctx, _spilled())
    fit, messages, trace = _wide()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    source = json.loads(wake["result"])["wake_delivery"]["source"]
    _shown(registry, wake, messages, trace, fit)
    total = source["size"]
    store = task_artifact_dir_path(tmp_path, task_id_for_artifacts(ctx))
    shutil.copyfile(store / source["path"], store / "copy.json")
    copy_view = _shown(registry, _call(registry, tmp_path, "read-copy", "read_file", {
        "root": "artifact_store", "path": "copy.json", "start_line": 1, "max_lines": 100_000}), messages, trace, fit)
    assert not copy_view[1].get("result_partial"), "the copy was delivered whole"
    half = _shown(registry, _call(registry, tmp_path, "read-half", "read_file", _page(source, 0, total // 2)),
                  messages, trace, fit)
    assert not half[1].get("result_partial")
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]
    assert not (getattr(ctx, "_sleep_interaction_seen", None) or {}), "the question was not shown yet"
    rest = _call(registry, tmp_path, "read-rest", "read_file", _page(source, total // 2, total + 1))
    rest_shown, _rest_row = _shown(registry, rest, messages, trace, fit)
    assert ctx._sleep_interaction_seen == {_RUN: ["q1"]} and _WAKE_TAIL in rest_shown
    assert "owner-mail" not in ctx._loop_mailbox_seen_ids and _TAIL not in rest_shown, "no source of it holds the mail"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == ["owner-mail"], \
        "the wake's ACK leaves the mail to the drain"


@pytest.mark.parametrize("saved", ["current", "legacy"])
def test_a_refused_ack_after_the_source_read_is_recorded_and_the_replay_acknowledges_without_another_read(
        tmp_path, monkeypatch, saved):
    """The read whole, its saved state, then an acknowledgement that does not land: the wake stays pending with a
    typed diagnostic and no callback is kept. The replay names the same canonical source, already read in this
    process, and acknowledges at the next saved state. A current wake owes only its acknowledgement row; a wake
    saved before this contract (``legacy``, by hand, pending at the upgrade so the first wait replays it) also owes
    the ACK of the short mail inside it, which the replay's acknowledgement then writes."""
    from ouroboros import delegate_custody as custody
    from ouroboros import owner_mailbox

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _spilled())
    if saved == "legacy":
        _base_saved_wake(tmp_path, ctx, _RUN, "Short owner note " + _TAIL, status="terminal", state="succeeded",
                         last_seq=1)
    fit, messages, trace = _wide()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    source = json.loads(wake["result"])["wake_delivery"]["source"]
    _shown(registry, wake, messages, trace, fit)
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", _page(source, 0, source["size"] + 1)),
           messages, trace, fit)
    with monkeypatch.context() as refusing:
        if saved == "legacy":
            refusing.setattr(owner_mailbox, "acknowledge_task_messages", lambda *_a, **_k: False)
        else:
            emit = custody.emit
            refusing.setattr(custody, "emit", lambda root, kind, payload, **kwargs: (
                False if kind == "delegate_supervision_wake_acknowledged" else emit(root, kind, payload, **kwargs)))
        _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and not getattr(ctx, "_pending_content_acks", None)
    reason = "mailbox_acknowledgement_unwritten" if saved == "legacy" else "acknowledgement_row_unwritten"
    assert _refused(tmp_path) == [(_wake_id(wake), reason, None)]
    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == int(saved == "current") and json.loads(replay["result"])["wake_delivery"]["source"] == source
    _shown(registry, replay, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == []
    assert replay["result_meta"]["tool_result_meta"]["wake_source"] == source, "found without parsing the view"


def test_a_startup_receipt_carries_an_oversized_recovered_wake_whole_before_its_saved_acknowledgement(
        tmp_path, monkeypatch):
    """The bootstrap's overflow path: a recovered wake longer than any ``delegate_wait`` view is never summarized by
    the startup receipt. The restored transcript gets the canonical wake whole, exposure (its question) stages from
    that text and the canonical ACK lands only after a saved state holds it."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from ouroboros.tool_capabilities import tool_result_limit
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, _NANNY, 2)
    ctx = registry._ctx
    ctx._loop_mailbox_seen_ids = set()
    _intent(ctx, _spilled())
    payload = {"status": "waiting_on_user", "run_id": _RUN, "waiting_on_user": True, "pending_interactions": [
                   {"interaction_id": "q1", "questions": [{"id": "choice", "question": "Which path?"}]}],
               "coordination_context": ds.coordination_live_context(ctx), "supervision_wake_id": "wake"}
    assert len(json.dumps(payload, indent=2)) > tool_result_limit("delegate_wait"), "premise: a wait view spills it"
    ds._save_state(ctx, {"run_id": _RUN, "pending_wake": {"wake_id": "wake", "attempt_key": "1", "payload": payload,
                                                          "mailbox_ids": [], "interaction_ids": ["q1"]}})
    messages, trace = [], {"tool_calls": []}
    append_startup_receipt(ctx, messages, json.dumps({"status": "configured_session_recovered_wake",
                                                      "recovery": {}, "wake": payload}, indent=2), set())
    assert _WAKE_TAIL in messages[-1]["content"] and "wake_delivery" not in messages[-1]["content"]
    assert ctx._sleep_interaction_seen == {_RUN: ["q1"]} and _wake_state(ctx) == ("wake", "")
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", "wake")


def _crowded(window=40_000, reserve=2_000, free=900):
    """A measured frame and one user turn leaving ``free`` tokens beside the reply reserve: unasked bodies are cut."""
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from tests._tool_result_delivery_shared import measured_fit

    filler = "filler words for an already long transcript. " * 64
    messages = [{"role": "user", "content": filler}]
    while estimate_context_prompt_tokens(messages, []) < window - reserve - free:
        messages[0]["content"] += filler
    return measured_fit(window=window, reserve=reserve), messages, {"tool_calls": []}


def _wide():
    from tests._tool_result_delivery_shared import measured_fit

    return measured_fit(window=500_000, reserve=16_000), [{"role": "user", "content": "go"}], {"tool_calls": []}


def _mail():
    return "Owner instruction: " + "the exact long specification line; " * 600 + _TAIL


def _key(ref):
    return f"{ref['path']}|{ref['sha256']}"


def test_a_spilled_wake_the_frame_also_cuts_waits_for_its_typed_inner_source_not_the_cut_summary(
        tmp_path, monkeypatch):
    """Both shortened: the frame cuts the spilled summary too. The cut text is never parsed for a source; the typed
    inner source is the one obligation, and the outer retained text (only the summary) read whole credits nothing."""
    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _spilled())
    fit, messages, trace = _crowded(reserve=2_300, free=600)  # the same transcript, less room than the summary needs
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    inner = wake["result_meta"]["tool_result_meta"]["wake_source"]
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and json.loads(wake["result"])["wake_delivery"]["complete"] is False
    assert list(ctx._wake_sources) == [_key(inner)], "the typed inner source, not one parsed from the cut text"
    outer = row["result_source_ref"]
    summary = _shown(registry, _call(registry, tmp_path, "read-outer", "read_file", _page(outer, 0, outer["size"] + 1)),
                     messages, trace, fit)
    assert not summary[1].get("result_partial") and _WAKE_TAIL not in summary[0], \
        "the summary arrived whole, the wake's body only in its source"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "")
    whole = _shown(registry, _call(registry, tmp_path, "read-inner", "read_file", _page(inner, 0, inner["size"] + 1)),
                   messages, trace, fit)
    assert not whole[1].get("result_partial") and _WAKE_TAIL in whole[0]
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and ctx._wake_sources == {}


def test_a_cut_source_read_and_an_unsaved_checkpoint_each_leave_the_wake_pending_until_the_next_saved_state(
        tmp_path, monkeypatch):
    """A read the frame cuts shows a head and tail, so it completes nothing, nor does its repeat. The whole read then
    defers the ACK; a checkpoint whose write fails flushes nothing and keeps it, and the next saved state ACKs."""
    from ouroboros import working_checkpoint as wc
    from tests._tool_result_delivery_shared import measured_fit

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _spilled())
    fit, messages, trace = _wide()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    inner = wake["result_meta"]["tool_result_meta"]["wake_source"]
    _shown(registry, wake, messages, trace, fit)
    for attempt in ("cut-1", "cut-2"):
        cut = _shown(registry, _call(registry, tmp_path, attempt, "read_file", _page(inner, 0, inner["size"] + 1)),
                     messages, trace, measured_fit(window=3_000))
        assert cut[1].get("result_partial"), "the read alone exceeds this frame's usable input"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "")
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", _page(inner, 0, inner["size"] + 1)),
           messages, trace, fit)
    real = wc._write
    monkeypatch.setattr(wc, "_write", lambda *_a, **_k: (_ for _ in ()).throw(OSError("controlled unsaved checkpoint")))
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and len(ctx._pending_content_acks) == 1, "kept, not flushed"
    monkeypatch.setattr(wc, "_write", real)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake))


def test_a_cut_wake_without_a_retained_source_is_never_acknowledged_and_its_replay_recovers_once_one_is_kept(
        tmp_path, monkeypatch):
    """Source persistence fails: the cut view promises no source, so nothing can complete it and nothing is ACKed.
    The replay is marked and cut again the same way; once a source is kept, reading it whole acknowledges."""
    from ouroboros import tool_result_delivery
    from tests.test_tool_result_delivery import _view_json

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _cut())
    fit, messages, trace = _crowded()
    real = tool_result_delivery.persist_result_source
    monkeypatch.setattr(tool_result_delivery, "persist_result_source", lambda *_a, **_k: {})
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and row.get("result_source_status") == "source_unavailable"
    assert not getattr(ctx, "_wake_sources", None)
    _ready(registry, messages, trace)
    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 1 and json.loads(replay["result"])["replay_of"]["fresh_observation"] is False
    again = _shown(registry, replay, messages, trace, fit)[1]
    _ready(registry, messages, trace)
    assert again.get("result_source_status") == "source_unavailable" and _wake_state(ctx) == (_wake_id(wake), "")
    monkeypatch.setattr(tool_result_delivery, "persist_result_source", real)
    kept, kept_row = _shown(registry, _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN}),
                            messages, trace, fit)
    omitted = _view_json(kept)["read_omitted"]
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", {key: omitted[key] for key in (
        "root", "path", "start_line", "max_lines", "start_char", "max_chars")}), messages, trace, fit)
    _ready(registry, messages, trace)
    assert kept_row.get("result_source_status") == "ready" and _wake_state(ctx) == ("", _wake_id(wake))


def test_a_stale_source_read_whole_after_its_wake_was_acknowledged_cannot_acknowledge_the_next_wake(
        tmp_path, monkeypatch):
    """No pending source is dropped, so a later read of an older wake's exact source still completes; it names that
    acknowledged wake, never the one pending now, which keeps waiting while the refusal records what was delivered.
    The older wake's first view was cut, its replay shown whole acknowledged it, and the cut view's source stayed."""
    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    _intent(ctx, _cut())
    crowded, messages, trace = _crowded()
    wide = _wide()[0]
    first = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    cut = _shown(registry, first, messages, trace, crowded)[1]
    source = cut["result_source_ref"]
    assert cut.get("result_partial") and list(ctx._wake_sources) == [_key(source)], "premise: the frame cut it"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(first), "")
    _shown(registry, _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN}), messages, trace, wide)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(first)) and _key(source) in ctx._wake_sources, "acknowledged, kept"
    second = _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2 and _wake_id(second) != _wake_id(first)
    _shown(registry, _call(registry, tmp_path, "read-old", "read_file", _page(source, 0, source["size"] + 1)),
           messages, trace, wide)
    _ready(registry, messages, trace)
    assert _refused(tmp_path) == [(_wake_id(second), "pending_wake_not_named", [_wake_id(first)])]
    assert _wake_state(ctx) == (_wake_id(second), _wake_id(first)) and ctx._wake_sources == {}
    _shown(registry, second, messages, trace, wide)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(second))


def test_every_pending_wake_source_stays_completable_beyond_any_count(tmp_path, monkeypatch):
    """Forty runs each hold a spilled wake (their bulk is the shared coordination context, so no mail is cut); the
    oldest exact source, read whole after the other thirty-nine arrived, still acknowledges its own wake only."""
    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls), runs=[f"run-{n}" for n in range(40)])
    ctx.task_contract = {"delegation_budget": {"intent_note": "preserve-complete-intent:" + "y" * 30_000}}
    fit, messages, trace = _wide()
    wakes = [_call(registry, tmp_path, f"wait-{n}", "delegate_wait", {"run_id": f"run-{n}"}) for n in range(40)]
    for wake in wakes:
        _shown(registry, wake, messages, trace, fit)
    sources = [wake["result_meta"]["tool_result_meta"]["wake_source"] for wake in wakes]
    assert len({_key(ref) for ref in sources}) == 40 and len(ctx._wake_sources) == 40
    _shown(registry, _call(registry, tmp_path, "read-oldest", "read_file", _page(sources[0], 0, sources[0]["size"] + 1)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    from ouroboros import delegate_supervision as ds

    pending = {n: (ds._load_state(ctx, f"run-{n}").get("pending_wake") or {}).get("wake_id") for n in range(40)}
    assert pending[0] is None and all(pending[n] == _wake_id(wakes[n]) for n in range(1, 40))
    assert len(ctx._wake_sources) == 39
