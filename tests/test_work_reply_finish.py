"""A regular finish waits for my human's open direct Reply; words keyed to a late Reply outlive a stale retained final.
A forced return without its action stays private; an answer already kept answers only its own Reply's row.

Scripted turns over the real loop and its real send tool, isolated in ``tmp_path``. The host never answers for the
work: the local result is no reply to a question it was not keyed to, only a ``send_user_message(reply_to_relay=...)``
the author wrote is (ARCHITECTURE §6)."""
import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
from ouroboros.owner_relay import owner_quote, write_relay
from ouroboros.tools.registry import ToolRegistry
from tests._delivery_candidate_shared import holds
from tests.test_acceptance_preparation_loop import _delivery_ctx
from tests.test_owner_relay import _forced_reply_json, _replies

TASK, LOCAL, ANSWER = "reply-root", "Migration finished.", "Yes: schema first, then backfill."


def _call(name, **arguments):
    return {"content": None, "tool_calls": [{"id": name, "type": "function",
                                              "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _finish(answer=None, *, by_hash=False, action="finish", **more):
    return _call("finish_task", action=action, **(
        {"answer_sha256": hashlib.sha256(LOCAL.encode()).hexdigest()} if by_hash else {"answer": answer or LOCAL}), **more)


def _turn(tmp_path, monkeypatch, responses, *, disable_send=False, metadata=None, before="", after=""):
    """A scripted root turn in Main with my human's Reply to it open from its first round (``before``/``after``: my
    human's own words mailed to the work just before or after that Reply): ``(result, requests, tool context,
    [(row address, words)] of the relay answers sent, relay id, row address)``."""
    from ouroboros.owner_mailbox import write_owner_message

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    [(row, addr)] = _replies(tmp_path, "Can you confirm the migration order too?")
    rid = f"work-reply:{addr}"
    if before:
        write_owner_message(tmp_path, before, TASK)
    write_relay(tmp_path, TASK, relay_id=rid, note="", chat_id=1, teller_task_id="",
                quotes=[owner_quote(tmp_path, row, address=addr)], canonical_root=tmp_path, work_reply=True)
    if after:
        write_owner_message(tmp_path, after, TASK)
    registry, events, calls, scripted = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), queue.Queue(), [], iter(responses(rid))
    registry._ctx.current_chat_id, registry._ctx.task_metadata = 1, {"root_task_id": TASK, **(metadata or {})}
    if disable_send:
        registry._ctx.task_contract = {"disabled_tools": ["send_user_message"]}

    def respond(_llm, messages, *_a, **_kw):
        calls.append(json.loads(json.dumps(messages)))
        return {"role": "assistant", **next(scripted)}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    result, _usage, _trace = loop.run_llm_loop(
        [{"role": "user", "content": "Run the migration"}], registry, SimpleNamespace(default_model=lambda: "test-model"),
        tmp_path / "logs", lambda *_a, **_kw: None, queue.Queue(), event_queue=events, task_id=TASK, drive_root=tmp_path)
    sent = [e for e in iter(lambda: events.get_nowait() if not events.empty() else None, None) if e.get("type") == "send_message"]
    return result, calls, registry._ctx, [(e["progress_meta"]["reply_to"]["address"], e["text"]) for e in sent
                                          if e.get("system_type") == "relay_reply"], rid, addr


def _held(call, rid):
    return sum("work_reply_open" in str(row.get("content")) and rid in str(row.get("content")) for row in call)


@pytest.mark.parametrize("form", ["finish_task", "prose"])
def test_a_regular_finish_waits_for_the_open_reply_and_stands_beside_its_answer(tmp_path, monkeypatch, form):
    """Selected or plain, the final is held once while an answerable Reply is open; the keyed answer lands under my
    human's row, the local result stays the work's own final and is never echoed as that answer."""
    result, calls, ctx, sent, rid, addr = _turn(tmp_path, monkeypatch, lambda rid: [
        _finish() if form == "finish_task" else {"content": LOCAL},
        _call("send_user_message", text=ANSWER, reply_to_relay=rid), _finish(by_hash=True)])
    assert result == LOCAL and len(calls) == 3
    assert [_held(call, rid) for call in calls] == [0, 1, 1]  # the notice stays in the transcript the next rounds read
    assert sent == [(addr, ANSWER)] and not ctx._open_relays and not ctx.pending_events
    assert [(row["origin"], row["reason"]) for row in holds(tmp_path)] == [
        ("selection" if form == "finish_task" else "prose", "work_reply_open")]  # the one typed hold


@pytest.mark.parametrize("case", ["stop", "cannot_answer", "unknown_send"])
def test_a_stop_and_a_work_that_cannot_answer_are_never_held_and_an_unknown_send_keeps_the_hold(tmp_path, monkeypatch, case):
    """``action=stop`` and a work without its return tool end as ever, the Reply left open; a send whose first answer
    cannot be read sends nothing, retires nothing, and the finish stays held."""
    if case == "unknown_send":
        monkeypatch.setattr("ouroboros.tools.control_runtime._first_answer_witness", lambda *_a: {"recorded": None})
    stop = _finish(action="stop", rationale="unfinished: the migration order is unknown")
    result, calls, ctx, sent, rid, _addr = _turn(tmp_path, monkeypatch, lambda rid: {
        "stop": [stop], "cannot_answer": [_finish()],
        "unknown_send": [_finish(), _call("send_user_message", text=ANSWER, reply_to_relay=rid), _finish(by_hash=True), stop],
    }[case], disable_send=case == "cannot_answer")
    assert result == LOCAL and not sent and set(ctx._open_relays) == {rid}
    assert [_held(call, rid) for call in calls] == ([0, 1, 1, 2] if case == "unknown_send" else [0])  # one notice per held finish
    if case == "unknown_send":
        assert "RELAY_RETURN_UNKNOWN" in str(calls[-1]) and not ctx.pending_events


def _stale_forced(tmp_path, monkeypatch, body, *, stop=False):
    """One forced call whose body selects a retained answer that a late Reply, drained after it, made stale."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result

    [(row, addr), (row_tell, addr_tell)] = _replies(tmp_path, "Late Reply", "Forwarded by a teller")
    r1, rtell, tid = f"work-reply:{addr}", f"tell:{addr_tell}", "stale-forced"
    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    ctx.task_id = registry._ctx.task_id = ctx.root_task_id = tid
    ctx.tool_schemas = registry.schemas()
    registry._ctx.current_chat_id = 1
    registry._ctx.event_queue = ctx.event_queue = queue.Queue()
    stale = loop._replace_delivery_candidate(registry, ctx, trace, "Pre-Reply draft.", control="candidate")
    for rid, source, source_addr, mine in ((r1, row, addr, True), (rtell, row_tell, addr_tell, False)):
        write_relay(tmp_path, tid, relay_id=rid, note="" if mine else "teller note", chat_id=1,
                    teller_task_id="" if mine else "teller", quotes=[owner_quote(tmp_path, source, address=source_addr)],
                    canonical_root=tmp_path, work_reply=mine)
    if stop:
        write_task_result(tmp_path, tid, "running", chat_id=1, root_task_id=tid)
        registry._ctx.model_wait_context = SimpleNamespace(
            task={"_presence_turn": True}, worker_slot_held=False, task_id=tid, canonical_root=tmp_path)

    def respond(*_a, **_kw):
        if stop:
            request_cancel(tmp_path, tid, reason="owner_stop", source="owner_ui")
        return _forced_reply_json(answer_sha256=stale.content_sha256, relay_replies=body(r1, rtell)), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, usage, out = loop._forced_final_answer(
        ctx, prompt="[ROUND_LIMIT]", fallback_text="fallback", reason_code="round_limit_exceeded")
    sent = [(e["progress_meta"]["reply_to"]["address"], e["text"]) for e in
            iter(lambda: ctx.event_queue.get_nowait() if not ctx.event_queue.empty() else None, None)
            if e.get("type") == "send_message" and e.get("system_type") == "relay_reply"]
    return text, usage, out, sent, registry._ctx, (r1, rtell, addr)


def test_a_stale_retained_selection_never_drops_words_keyed_to_a_late_reply(tmp_path, monkeypatch):
    """The late Reply stales the old final, not the answer written to it: the keyed words go out through the ordinary
    send, the stale final stays the fallback and is not the answer to anything."""
    text, usage, out, sent, ctx, (r1, _rtell, addr) = _stale_forced(
        tmp_path, monkeypatch, lambda r1, _t: {r1: "Answer to the late Reply"})
    assert sent == [(addr, "Answer to the late Reply")] and not ctx._open_relays.keys() & {r1}
    assert out["forced_work_replies"] == [{"relay_id": r1, "status": "ok", "code": "OK",
                                           "delivery_id": f"relay-return:{r1}", "delivery_state": "owed"}]
    assert "Pre-Reply draft." in text and usage["terminal_origin"] != "model_final" and usage["owner_message_sent"] is True


@pytest.mark.parametrize("case", ["selection_only", "unknown_and_foreign", "stop"])
def test_a_stale_selection_alone_foreign_ids_and_a_direct_stop_send_nothing(tmp_path, monkeypatch, case):
    """No keyed words: the retained result is not sent on a guessed association. An unknown id or another work's relay
    is blocked by id, and a direct Stop sends nothing even with valid keyed words; the Reply stays open for its owner."""
    text, usage, out, sent, ctx, (r1, rtell, _addr) = _stale_forced(
        tmp_path, monkeypatch,
        lambda r1, rtell: {"selection_only": {}, "unknown_and_foreign": {"work-reply:unknown": "x", rtell: "y"},
                           "stop": {r1: "Must not send on Stop"}}[case], stop=case == "stop")
    assert not sent and r1 in ctx._open_relays
    assert ("owner stopped this chat turn" in text) if case == "stop" else "Pre-Reply draft." in text
    if case == "unknown_and_foreign":
        assert [(row["relay_id"], row["status"], row["reason_code"]) for row in out["forced_work_replies"]] == [
            ("work-reply:unknown", "blocked", "RELAY_RETURN_UNRESOLVED"), (rtell, "blocked", "RELAY_RETURN_UNRESOLVED")]
    else:
        assert "forced_work_replies" not in out


@pytest.mark.parametrize("case", ["fallback", "kept", "misaddressed", "report"])
def test_an_offered_relay_return_without_its_action_stays_private_and_the_final_falls_back(tmp_path, monkeypatch, case):
    """Words returned to an offered relay (by its id or its source row) are a completion request, so one missing its
    action is refused as invalid: nothing reaches the source room, the Reply stays open, and the final is the kept
    answer or the host's fallback, never that JSON with its relay id. A report whose ``relay_replies`` names no offered
    relay (its counts in and out) is the author's data, kept as written while the Reply is open."""
    [(row, addr)] = _replies(tmp_path, "Can you confirm the migration order too?")
    rid = f"work-reply:{addr}"
    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    ctx.tool_schemas = registry.schemas()
    registry._ctx.current_chat_id = 1
    registry._ctx.event_queue = ctx.event_queue = queue.Queue()
    write_relay(tmp_path, ctx.task_id, relay_id=rid, note="", chat_id=1, teller_task_id="",
                quotes=[owner_quote(tmp_path, row, address=addr)], canonical_root=tmp_path, work_reply=True)
    loop._drain_forced_owner_directives(ctx, trace)
    if case == "kept":
        loop._replace_delivery_candidate(registry, ctx, trace, "Retained report.", control="candidate")
    body = json.dumps({"relay_replies": {"in": 3, "out": 2}, "period": "today"} if case == "report" else {
        "answer": LOCAL, "relay_replies": {addr if case == "misaddressed" else rid: ANSWER}})
    prompts = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda _llm, messages, *_a, **_kw: (
        prompts.append(messages[-1]["content"]) or {"role": "assistant", "content": body}, 0.0))
    text, _usage, out = loop._forced_final_answer(
        ctx, prompt="[ROUND_LIMIT]", fallback_text="fallback", reason_code="round_limit_exceeded")
    sent = [e for e in iter(lambda: ctx.event_queue.get_nowait() if not ctx.event_queue.empty() else None, None)
            if e.get("type") == "send_message" and not e.get("is_progress")]
    assert '"relay_replies"' in prompts[0] and rid in prompts[0]  # the forced prompt offered this Reply's return
    assert not sent and set(registry._ctx._open_relays) == {rid} and "forced_work_replies" not in out
    if case == "report":
        assert text == body and "forced_completion_request" not in out
    else:
        assert text == ("Retained report." if case == "kept" else "fallback") and "relay_replies" not in text
        assert out["forced_completion_request"]["status"] == "refused"


@pytest.mark.parametrize("case", ["alone", "earlier", "later"])
def test_an_answer_already_kept_answers_only_its_own_reply_and_a_stop_stays_a_stop(tmp_path, monkeypatch, case):
    """A scheduled run holding my human's Reply open again (restored from before its send) meets that Reply's answer
    already kept in history: its new words do not go out, and the kept answer answers the Reply's own row, so the run
    may finish silent. My human's earlier or later unanswered row still holds that silent finish; Stop ends it as ever."""
    from ouroboros.loop_messages import owner_reply_owed

    monkeypatch.setattr("ouroboros.tools.control_runtime._first_answer_witness", lambda *_a: {"recorded": True})
    other = {"earlier": {"before": "Pause the backfill first?"}, "later": {"after": "Also pause the backfill?"}}.get(case, {})
    result, calls, ctx, sent, _rid, addr = _turn(tmp_path, monkeypatch, lambda rid: [
        _call("send_user_message", text=ANSWER, reply_to_relay=rid), _call("finish_task", action="finish", silent=True),
        _finish(action="stop", rationale="unfinished: whether to pause the backfill is not decided")],
        metadata={"schedule_occurrence": {"schedule_id": "nightly", "token": "occurrence-1"}}, **other)
    assert "already has its first answer" in str(calls[1]) and not sent and not ctx._open_relays
    assert ctx._answered_reply_rows == [addr] and not ctx._owner_reply_revision  # the Reply's own row, nothing else
    if case == "alone":
        assert result == "" and len(calls) == 2 and not owner_reply_owed(ctx)
    else:
        assert result == LOCAL and len(calls) == 3 and "owner_reply_owed" in str(calls[2]) and owner_reply_owed(ctx)


def test_answers_kept_for_two_replies_met_in_reverse_order_each_discharge_their_own_row(tmp_path, monkeypatch):
    """A run restored from before it answered my human's two Replies meets both answers already kept, the second one
    first and a checkpoint between them: each discharges its own row, so with no other question the run may finish
    silent; my human's own new question mailed after them still holds that silent finish. Those marks are this
    turn's: a new turn starts without them, as without its live-reply mark."""
    from ouroboros.loop import _reset_turn_state
    from ouroboros.loop_messages import owner_reply_owed
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from ouroboros.tools.control_runtime import _send_user_message

    monkeypatch.setattr("ouroboros.tools.control_runtime._first_answer_witness", lambda *_a: {"recorded": True})
    (row1, addr1), (row2, addr2) = _replies(tmp_path, "Change the priority?", "And drop the chart.")
    r1, r2 = f"work-reply:{addr1}", f"work-reply:{addr2}"

    def attempt():
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        registry._ctx.task_id, registry._ctx.current_chat_id, registry._ctx.event_queue = TASK, 1, queue.Queue()
        registry._ctx.task_metadata = {"root_task_id": TASK}
        return registry

    first = attempt()
    for rid, row, addr in ((r1, row1, addr1), (r2, row2, addr2)):
        write_relay(tmp_path, TASK, relay_id=rid, note="", chat_id=1, teller_task_id="",
                    quotes=[owner_quote(tmp_path, row, address=addr)], canonical_root=tmp_path, work_reply=True)
    _drain_incoming_messages([], queue.Queue(), tmp_path, TASK, None, set(), owner_ctx=first._ctx)
    assert "already has its first answer" in _send_user_message(first._ctx, "New words.", reply_to_relay=r2)
    assert set(first._ctx._open_relays) == {r1} and owner_reply_owed(first._ctx)
    state = json.loads(json.dumps(continuation_state(first._ctx, [], {}, {}, 1, [], set())))
    resumed = attempt()
    restore_continuation_state(resumed, state, [], {}, {}, set())
    assert "already has its first answer" in _send_user_message(resumed._ctx, "New words.", reply_to_relay=r1)
    assert resumed._ctx._answered_reply_rows == sorted([addr1, addr2])
    assert not getattr(resumed._ctx, "_owner_reply_revision", None)  # no live send moved the mark
    assert not owner_reply_owed(resumed._ctx) and first._ctx.event_queue.empty() and resumed._ctx.event_queue.empty()
    write_owner_message(tmp_path, "Also: is the backfill paused?", TASK)
    _drain_incoming_messages([], queue.Queue(), tmp_path, TASK, None, set(), owner_ctx=resumed._ctx)
    assert not resumed._ctx._open_relays and owner_reply_owed(resumed._ctx)  # owed for that new question alone
    _reset_turn_state(resumed._ctx)
    assert resumed._ctx._answered_reply_rows is None and resumed._ctx._owner_reply_revision is None
