"""Historical delegate publications and wait results at the restore boundary."""
import json
from dataclasses import replace

import pytest

from ouroboros import mailbox_restore

from tests._run_wait_shared import scripted_reads
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import InputReached
from tests.test_delegate_wake_mail_delivery import _NANNY, _TAIL, _base_saved_wake, _unread


def _legacy_spilled_wake(tmp_path, monkeypatch, *, annotated=False):
    """What base 1a0e1e9d saved after a spilled wake carrying a long owner mail: the saved wake assembled by hand as
    that base's ``_publish`` stored it (``_base_saved_wake``), executed and delivered as production does
    (``execute_tool_batch``: its row keeps its invocation record; the renderer spills it), then the old seam's one
    difference applied by hand: the old ``_text_mailbox_ids`` counted the cut summary as seen."""
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
    del ctx._restored_wake_mail  # the historical worker predates both registration and import
    if hasattr(ctx, "_mailbox_content_ids"):
        del ctx._mailbox_content_ids
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    return registry, ctx, task, messages, limit


@pytest.mark.parametrize("variant", ["pending", "anchored", "requeued", "source_deleted", "source_unavailable",
                                     "rewritten_view", "another_wake"])
def test_cold_restore_registers_the_letter_without_crediting_a_cut_or_unrelated_view(
        tmp_path, monkeypatch, stub_delegate_engines, variant):
    """The addressed mailbox proves the letter independently of a cut, absent or rewritten old tool view."""
    import hashlib

    from ouroboros import delegate_recovery as recovery
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
    from tests._loop_resume_shared import _loop_kwargs
    from tests._loop_resume_shared import _registry

    registry, ctx, task, messages, limit = _legacy_spilled_wake(tmp_path, monkeypatch, annotated=variant == "anchored")
    tool_row = messages[-1]
    view = json.JSONDecoder().raw_decode(tool_row["content"])[0]

    def restamped(changed):  # other bytes whose own record verifies
        tool_row["content"] = json.dumps(changed, ensure_ascii=False, indent=2)
        tool_row[TOOL_RESULT_RECORD_KEY] = {**tool_row[TOOL_RESULT_RECORD_KEY], "content_sha256": hashlib.sha256(
            tool_row["content"].encode("utf-8")).hexdigest()}

    if variant == "requeued":
        # Persist the old producer's state. The current consumer must refuse this cut.
        from ouroboros.owner_mailbox import acknowledge_task_messages

        state = ds._load_state(ctx, "run-one")
        pending = state["pending_wake"]
        state.update(status="awake", pending_wake={}, last_acknowledged_wake={
            **pending, "acknowledged_at": "2025-01-05T12:00:00Z"})
        ds._save_state(ctx, state)
        assert acknowledge_task_messages(tmp_path, ctx.task_id, ["owner-mail"], wake_id=pending["wake_id"],
                                         attempt_key=2)
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
        first.update(receipt=receipt, words=sum(str(row.get("content") or "").count(_TAIL) for row in call.messages),
                     sources=dict(cold._ctx._wake_sources),
                     pending=bool(ds._load_state(cold._ctx, "run-one").get("pending_wake")))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = _NANNY
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert _TAIL not in first["receipt"] and first["words"] == 1
    [letter] = [row for row in cold._ctx._owner_directives if row.get("msg_id") == "owner-mail"]
    assert letter["source"] == "owner_mailbox" and _TAIL in letter["content"]
    assert view["wake_delivery"]["source"] in first["sources"].values()
    assert first["pending"], "the requeued observation must also respect the original source's outstanding read"


@pytest.mark.parametrize("annotated", [False, True])
def test_a_parked_legacy_seen_summary_keeps_its_replay_pending_until_the_source_is_read_whole(tmp_path, monkeypatch, annotated):
    """The same saved state parked and resumed in the same attempt (a planned restart). The model's replay keeps the
    mail it was shown cut: the spilled view names its exact source and acknowledges nothing until that source is read
    whole and the next state is saved."""
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.task_results import write_task_result
    from tests._loop_resume_shared import _loop_kwargs
    from tests._loop_resume_shared import _registry

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
    assert replay["restored_mailbox"]["letter_ids"] == ["owner-mail"] and not replay.get("wake_events")
    assert replay["wake_delivery"]["source"] == replay["restored_mailbox"]["original_wake"]
    assert sum(str(row.get("content") or "").count(_TAIL) for row in rounds[0]) == 1
    assert facts["pending"] and facts["unread"] == [], "the letter has its own receipt; the partial wake still waits"
    assert _TAIL in facts["read"] and not facts["pending_after"] and facts["unread_after"] == []


@pytest.mark.parametrize("kind", ["task_message", "owner_text"])
def test_an_old_wake_leaves_its_letter_to_the_first_drain_and_retains_the_original(tmp_path, kind):
    from ouroboros.artifacts import read_actor_source_bytes
    import queue

    from ouroboros import delegate_supervision as ds, owner_mailbox, working_checkpoint
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests._run_wait_shared import _nanny, _run

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
    assert text not in limit.messages[-1]["content"] and "old-mail" not in limit.owner_msg_seen
    ref = ctx._restored_wake_mail[("leaf", "wake-leaf")]["source"]
    assert read_actor_source_bytes(tmp_path, ctx.task_id, ref) == json.dumps(payload, ensure_ascii=False, indent=2).encode()
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    working_checkpoint.save_ready(limit)
    assert str(limit.messages).count(text) == 1 and [row["content"] for row in ctx._owner_directives] == [text]
    assert not ds._load_state(ctx, "leaf").get("pending_wake"), "the receipt's ACK followed the saved state"


@pytest.mark.parametrize("stage", ["before_delivery", "materialized", "different_bytes", "different_wake",
                                   "outer_partial", "unrelated_outer", "unavailable_outer", "only_saved_text"])
def test_saved_delegate_wait_keeps_original_bytes_and_binds_only_its_own_wake(tmp_path, stage):
    import hashlib
    import queue
    from ouroboros import delegate_supervision as ds, native_task_wait, owner_mailbox, owner_wait, working_checkpoint
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from tests._run_wait_shared import _body, _nanny, _notices, _run

    old, _ = _nanny(tmp_path)
    words = "Keep the draft — including its final newline.\n"
    if stage != "only_saved_text":
        assert owner_mailbox.write_task_message(tmp_path, words, old.task_id, msg_id="saved-mail",
                                                source_task_id="writer", provenance="peer_task", relation="sibling")
    _run(tmp_path, "leaf", old.task_id)
    wake = {"run_id": "leaf", "status": "progress", "supervision_wake_id": "w", "last_seq": 17,
            "wake_events": [{"type": "addressed_message", "kind": "task_message", "msg_id": "saved-mail",
                             "provenance": "peer_task", "source_task_id": "writer", "text": words},
                            {"type": "child_result", "summary": "No more changes."}]}
    raw = json.dumps(wake, ensure_ascii=False, indent=2)
    if stage == "only_saved_text":
        # The materialized result is the only copy of the words. Its slot holds just an ID.
        wake["wake_events"][0].pop("text")
    ds._save_state(old, {"run_id": "leaf", "pending_wake": {"wake_id": "w", "attempt_key": "1",
                   "payload": wake, "mailbox_ids": ["saved-mail"]}})
    descriptor = {"tool": "delegate_wait", "runs": ["leaf"], "sleep_id": "park",
                  "arguments": {"run_id": "leaf"}, "wake": {"run_id": "leaf", "wake_id": "w"}}
    if stage in ("outer_partial", "unrelated_outer", "unavailable_outer"):
        from ouroboros.tool_result_delivery import persist_result_source

        outer_body = {"replay_of": ds._replay_fact({"created_at": "past"}), **wake}
        if stage == "unrelated_outer":
            outer_body["supervision_wake_id"] = "other"
        outer_text = json.dumps(outer_body, ensure_ascii=False, indent=2)
        outer = persist_result_source(tmp_path, old.task_id, "previous-call", outer_text)
        if stage == "unavailable_outer":
            outer = {**outer, "path": "source_handles/tool_results/unavailable.json"}
        old._wake_sources = {outer["path"] + "|" + outer["sha256"]: outer}
    if stage != "before_delivery":
        if stage == "different_bytes":
            raw += "\nAdditional exact facts from the old wait."
        descriptor["delivery"] = {"text": raw, "observed_at": "2025-01-05T12:00:00Z",
                                  "meta": {"supervision_wake_id": "other" if stage == "different_wake" else "w"}}
    old._pending_task_wait = descriptor
    checkpoint = owner_wait.continuation_state(old, [], {}, {}, 2, [], set())
    ctx, limit = _nanny(tmp_path)
    owner_wait.restore_continuation_state(limit.tools, checkpoint, limit.messages, limit.llm_trace,
                                          limit.accumulated_usage, limit.owner_msg_seen)
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    assert native_task_wait.deliver_owed(limit) and not native_task_wait.deliver_owed(limit)
    [notice] = _notices(limit)
    ref = limit.llm_trace["task_wait_deliveries"][-1]["source_ref"]
    original = read_actor_source_bytes(tmp_path, ctx.task_id, ref)
    if stage != "before_delivery":
        assert original == raw.encode() and ref["sha256"] == hashlib.sha256(original).hexdigest()
    working_checkpoint.save_ready(limit)
    if stage in ("different_wake", "different_bytes", "only_saved_text"):
        assert "mailbox_recovery" in notice and raw in notice
        assert ds._load_state(ctx, "leaf")["pending_wake"]["payload"] == wake
        if stage == "only_saved_text":
            assert json.loads(raw)["wake_events"][0]["text"] == words
            assert "saved-mail" not in ctx._mailbox_content_ids
            assert not owner_mailbox._ack_path(tmp_path, ctx.task_id).exists()
    else:
        view = _body(notice)
        assert view["restored_mailbox"]["letter_ids"] == ["saved-mail"]
        assert view["last_seq"] == 17 and view["wake_events"] == [wake["wake_events"][1]]
        assert read_actor_source_bytes(tmp_path, ctx.task_id, view["restored_mailbox"]["original_wake"]) == raw.encode()
        assert "Keep the draft" not in notice
        assert sum(str(row.get("content") or "").count(words) for row in limit.messages) == 1
        if stage == "outer_partial":
            assert view["wake_delivery"] == {"complete": False, "source": outer,
                "note": "The previous view was partial. Its outstanding exact-source read keeps its original coverage."}
            assert list(ctx._wake_sources.values()) == [outer]
            assert read_actor_source_bytes(tmp_path, ctx.task_id, outer) == outer_text.encode()
            assert ds._load_state(ctx, "leaf")["pending_wake"]["payload"] == wake
        elif stage == "unavailable_outer":
            assert view["restored_mailbox"]["unbound_partial_source"] == outer
            assert list(ctx._wake_sources.values()) == [outer]
            assert ds._load_state(ctx, "leaf")["pending_wake"]["payload"] == wake
        else:
            assert ds._load_state(ctx, "leaf")["last_acknowledged_wake"]["payload"] == wake
            if stage == "unrelated_outer":
                assert list(ctx._wake_sources.values()) == [outer]
    assert not [r for r in ctx._owner_directives if "Keep the draft" in str(r)]


@pytest.mark.parametrize("plural", [False, True])
def test_recovery_holders_bind_run_local_mail_before_flushing_their_wakes(tmp_path, plural):
    import queue
    from ouroboros import delegate_supervision as ds, owner_mailbox as mail, working_checkpoint
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests._run_wait_shared import _nanny, _run

    ctx, limit = _nanny(tmp_path)
    limit.messages[:] = [{"role": "user", "content": "Proceed."}]
    limit.owner_msg_seen = ctx._loop_mailbox_seen_ids
    holders = []
    for rid in (("one", "two") if plural else ("one",)):
        _run(tmp_path, rid, ctx.task_id)
        mid, words = f"letter-{rid}", f"Observation from {rid}."
        assert mail.write_task_message(tmp_path, words, ctx.task_id, msg_id=mid, source_task_id=rid,
                                       provenance="peer_task", relation="sibling")
        # The containing holder is the sole run identity. The two publications even share a wake id.
        wake = {"status": "progress", "supervision_wake_id": "saved", "wake_events": [
            {"type": "addressed_message", "kind": "task_message", "msg_id": mid, "text": words},
            {"type": "child_result", "summary": f"Result of {rid}."}]}
        holders.append({"run_id": rid, "wake": wake})
        ds._save_state(ctx, {"run_id": rid, "pending_wake": {"wake_id": "saved", "attempt_key": "1",
                                                              "payload": wake, "mailbox_ids": [mid]}})
    recovery = {"holders": holders} if plural else holders[0]
    raw = json.dumps({"status": "configured_session_continuation", "recovery": recovery})
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    append_startup_receipt(ctx, limit.messages, raw, limit.owner_msg_seen)
    receipt = limit.messages[-1]["content"]
    for holder in holders:
        rid = holder["run_id"]
        assert f"Observation from {rid}." not in receipt and f"Result of {rid}." in receipt
        assert ds._load_state(ctx, rid)["pending_wake"]["payload"] == holder["wake"]
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    altered = {**holders[0]["wake"], "wake_events": [{"type": "child_result", "summary": "Rewritten."}]}
    assert not ds.acknowledge_pending_wake(ctx, {**holders[0], "wake": altered})
    if plural:
        assert not ds.acknowledge_pending_wake(ctx, holders[0]["wake"])
        assert all(ds._load_state(ctx, h["run_id"]).get("pending_wake") for h in holders)
    working_checkpoint.save_ready(limit)
    for holder in holders:
        rid = holder["run_id"]
        assert sum(str(row["content"]).count(f"Observation from {rid}.") for row in limit.messages) == 1
        assert ds._load_state(ctx, rid)["last_acknowledged_wake"]["payload"] == holder["wake"]
    assert not ctx._owner_directives
    assert [json.loads(line)["msg_id"] for line in mail._ack_path(tmp_path, ctx.task_id).read_text().splitlines()] == [
        f"letter-{h['run_id']}" for h in holders]


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("source", ["original", "addressed_held", "unbound_held", "unbound_row", "other_task", "torn_mailbox"])
def test_legacy_teller_note_uses_only_its_addressed_relay_and_preserves_sibling_wakes(tmp_path, schema, source):
    import queue
    from ouroboros import delegate_supervision as ds, owner_mailbox as mail, owner_relay, working_checkpoint
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from tests._run_wait_shared import _nanny
    from tests._owner_relay_shared import context, source as retained_row

    ctx, limit = _nanny(tmp_path, "reader")
    limit.messages[:] = [{"role": "user", "content": "Proceed."}]
    row, handle = retained_row(tmp_path)
    quotes, error = owner_relay.quote_rows(context(tmp_path, row), [handle])
    assert not error
    target = "neighbour" if source == "other_task" else ctx.task_id
    note = "The teller describes the plan."
    entry, _ = owner_relay.write_relay(tmp_path, target, relay_id="letter", note=note, quotes=quotes,
                                        chat_id=1, teller_task_id="teller")
    mailbox = mail._mailbox_path(tmp_path, target)
    unbound = source in ("unbound_held", "unbound_row")
    if source == "addressed_held" or unbound:
        held = row if source == "addressed_held" else {**row, "text": "Another source entirely."}
        ctx._held_owner_messages = {owner_relay.owner_row_address(held): held}
        envelope = dict(entry)
        del envelope["owner_rows"]  # only the envelope and a separately held row survive
        if unbound:
            envelope["relay"]["quotes"][0]["attachments"].append(
                {"name": "unverified-file.txt", "content": "Unverified file contents."})
        if source == "unbound_row":
            envelope.update(owner_rows=[held], source_task_id={"detail": "Nested sender metadata."},
                            ts={"detail": "Nested timestamp metadata."})
        mailbox.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    mid = entry["msg_id"]
    event = {"type": "addressed_message", "msg_id": mid, "ts": entry["ts"], "kind": "relay_entry",
             "source_task_id": "teller", "provenance": "owner", "text": note}  # old wake omitted its quotes
    wake = {"run_id": "one", "supervision_wake_id": "first", "status": "progress", "wake_events": [event]}
    old = {"wake_id": "first", "attempt_key": "1", "mailbox_ids": [mid], "payload": wake}
    slot = {"pending_wake": old, "journal_cursor": 5}
    state = {"run_id": "one", **slot}
    if schema == 2:
        other = {"run_id": "two", "supervision_wake_id": "second", "status": "terminal",
                 "wake_events": [{"type": "addressed_message", "kind": "task_message", "msg_id": "missing"}]}
        state = {"schema": 2, "runs": {"one": slot, "two": {"journal_cursor": 9,
                 "last_acknowledged_wake": {"wake_id": "second", "payload": other, "mailbox_ids": ["missing"]}}}}
    path = ds._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(state, indent=2).encode()
    path.write_bytes(original)
    assert mail.acknowledge_task_messages(tmp_path, ctx.task_id, [mid], wake_id="historical")
    if source == "torn_mailbox":
        with mailbox.open("ab") as handle_file:
            handle_file.write(b'{"text":"\xc3')
    original_mail = mailbox.read_bytes()
    receipts = mail._ack_path(tmp_path, ctx.task_id).read_bytes()
    ctx._loop_mailbox_seen_ids = limit.owner_msg_seen = {mid}
    ctx._legacy_mailbox_seen = {mid}
    ctx._restore_mailbox = True
    ctx._mailbox_content_ids = set()
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    assert path.read_bytes() == original
    view = mailbox_restore.wake_view(ctx, wake)
    assert read_actor_source_bytes(tmp_path, ctx.task_id, view["restored_mailbox"]["original_wake"]) == (
        json.dumps(wake, ensure_ascii=False, indent=2).encode())
    events = queue.Queue()
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, events, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    working_checkpoint.save_ready(limit)
    assert mailbox.read_bytes() == original_mail
    assert not list(events.queue) and mail._ack_path(tmp_path, ctx.task_id).read_bytes() == receipts
    contents = [message["content"] for message in limit.messages]
    model_input = "\n".join(contents)
    if source != "other_task" and not unbound:
        assert [r["content"] for r in ctx._owner_directives] == [row["text"]]
        assert handle in owner_relay.held_owner_messages(ctx)
        assert row["text"] in model_input and note in model_input and not getattr(ctx, "_open_relays", {})
    else:
        assert not ctx._owner_directives and handle not in owner_relay.held_owner_messages(ctx)
    if unbound:
        [notice] = [body for body in contents if body.startswith(f"[Historical mailbox entry {mid};")]
        _, facts, words = notice.split("\n", 2)
        assert words == note
        expected = {"ts": entry["ts"], "source_task_id": "teller"} if source == "unbound_held" else {}
        assert json.loads(facts) == {"kind": "relay_entry", **expected}
        for withheld in (row["text"], json.dumps(row["text"]), held["text"], "unverified-file.txt",
                         "Unverified file contents.", "Nested sender metadata.", "Nested timestamp metadata."):
            assert withheld not in model_input
    if source in ("original", "addressed_held", "torn_mailbox"):
        assert view["wake_events"] == []
    else:
        assert view["wake_events"] == [event] and view["restored_mailbox"]["unresolved"] == [mid]
        assert not ds.acknowledge_pending_wake(ctx, wake)
    assert ds.acknowledge_pending_wake(ctx, view)
    assert ds._load_state(ctx, "one")["last_acknowledged_wake"]["payload"] == wake
    assert mail._ack_path(tmp_path, ctx.task_id).read_bytes() == receipts
    if schema == 2:
        assert ds._load_state(ctx, "two")["last_acknowledged_wake"]["payload"] == other
        assert "missing" in str(limit.messages)


@pytest.mark.parametrize(("recorded", "retain_source"), [(True, True), (False, True), (True, False)])
def test_old_tool_occurrence_keeps_its_producer_record_and_unverified_words_stay_visible(
        tmp_path, monkeypatch, recorded, retain_source):
    import queue
    from ouroboros import delegate_supervision as ds, owner_mailbox as mail
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record, read_tool_result_record
    from tests._run_wait_shared import _nanny

    ctx, limit = _nanny(tmp_path)
    assert mail.write_task_message(tmp_path, "Exact parent words.", ctx.task_id, source_task_id="principal", msg_id="m")
    payload = {"supervision_wake_id": "wake", "run_id": "leaf", "status": "progress",
               "wake_events": [{"type": "addressed_message", "kind": "task_message", "msg_id": "m",
                                "source_task_id": "principal", "text": "Exact parent words."}]}
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\nAn original host annotation."
    record = make_tool_result_record({"invocation_id": "occurrence", "tool_call_id": "wait"}, raw)
    retained = {"role": "tool", "tool_call_id": "wait", "content": raw}
    if recorded:
        retained[TOOL_RESULT_RECORD_KEY] = record
    limit.messages[:] = [{"role": "assistant", "tool_calls": [
        {"id": "wait", "type": "function", "function": {"name": "delegate_wait", "arguments": "{}"}}]}, retained]
    ds._save_state(ctx, {"run_id": "leaf", "pending_wake": {"wake_id": "wake", "payload": payload, "mailbox_ids": ["m"]}})
    if not retain_source:
        monkeypatch.setattr("ouroboros.tool_result_delivery.persist_result_source", lambda *_args: {})
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    tool_row = limit.messages[1]
    assert "An original host annotation." in tool_row["content"]
    projected = recorded and retain_source
    assert sum(str(r.get("content") or "").count("Exact parent words.") for r in limit.messages) == (1 if projected else 2)
    assert [r["content"] for r in ctx._owner_directives] == ["Exact parent words."]
    if projected:
        provenance = read_tool_result_record(tool_row)
        assert provenance["state"] == "recorded" and provenance["restored_from"]["record"] == record
        assert read_actor_source_bytes(tmp_path, ctx.task_id, provenance["restored_from"]["source"]) == raw.encode()
    else:
        assert tool_row == retained and "historical wait rows" in str(limit.messages)


def test_reused_wake_id_cannot_hide_the_only_surviving_teller_text(tmp_path):
    from ouroboros import delegate_supervision as ds
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests._run_wait_shared import _nanny

    ctx, limit = _nanny(tmp_path)
    words = "This is the teller's only surviving note.\n"
    event = {"type": "addressed_message", "kind": "relay_entry", "msg_id": "lost", "text": words}
    wake = {"run_id": "one", "supervision_wake_id": "reused", "wake_events": [event]}
    slot = {"wake_id": "reused", "payload": wake, "mailbox_ids": ["lost"]}
    ds._save_state(ctx, {"run_id": "one", "pending_wake": slot})
    other = {**wake, "wake_events": []}
    ctx._configured_startup_wake = json.dumps({"status": "configured_session_continuation", "wake": other})
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    append_startup_receipt(ctx, limit.messages, ctx._configured_startup_wake)
    [history] = [r["content"] for r in limit.messages if "unresolved historical delegate wake" in r["content"]]
    assert json.loads(history.split("\n", 1)[1])["wake_events"] == [event]
    assert not ds.acknowledge_pending_wake(ctx, other) and not ctx._owner_directives
    assert ds._load_state(ctx, "one")["pending_wake"] == slot


@pytest.mark.parametrize("fault", ["missing_letter", "mailbox_read", "source_write"])
def test_next_ordinary_wait_observes_the_run_after_a_historical_mail_gap(
        tmp_path, monkeypatch, stub_delegate_engines, fault):
    import queue
    from pathlib import Path
    from ouroboros import artifacts, delegate_supervision as ds, owner_mailbox as mail, working_checkpoint
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from tests._run_wait_shared import _asks, _nanny, _run

    ctx, limit = _nanny(tmp_path)
    limit.messages[:] = [{"role": "user", "content": "Keep working."}]
    limit.owner_msg_seen = ctx._loop_mailbox_seen_ids
    _run(tmp_path, "leaf", ctx.task_id)
    words = "These are the old sender's words.\n"
    event = {"type": "addressed_message", "kind": "task_message", "msg_id": "letter",
             "source_task_id": "sibling", "provenance": "peer_task", "relation": "sibling", "text": words}
    wake = {"run_id": "leaf", "supervision_wake_id": "older", "status": "progress", "wake_events": [event]}
    pending = {"wake_id": "older", "payload": wake, "mailbox_ids": ["letter"], "attempt_key": "1"}
    ds._save_state(ctx, {"run_id": "leaf", "pending_wake": pending})
    mailbox = mail._mailbox_path(tmp_path, ctx.task_id)
    if fault != "missing_letter":
        assert mail.write_task_message(tmp_path, words, ctx.task_id, source_task_id="sibling", msg_id="letter",
                                       provenance="peer_task", relation="sibling")
    with monkeypatch.context() as patch:
        if fault == "mailbox_read":
            original = Path.read_text
            def read(path, *args, **kwargs):
                if path == mailbox:
                    raise OSError("temporarily unavailable")
                return original(path, *args, **kwargs)
            patch.setattr(Path, "read_text", read)
        elif fault == "source_write":
            def unavailable(*_args, **_kwargs):
                raise OSError("temporarily full")
            patch.setattr(artifacts, "store_actor_source_bytes", unavailable)
        mailbox_restore.restore_wake_mail(ctx, limit.messages)
        first = json.loads(ds.run_wait(ctx, "leaf").text)
        assert first["wake_events"] == [event]
        if fault != "missing_letter":
            assert not ds.acknowledge_pending_wake(ctx, first)
    replay = ds.run_wait(ctx, "leaf")
    body = json.loads(replay.text)
    binding = body["restored_mailbox"]
    assert artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, binding["original_wake"]) == (
        json.dumps(wake, ensure_ascii=False, indent=2).encode())
    if fault == "missing_letter":
        assert binding["unresolved"] == ["letter"] and body["wake_events"] == [event]
    else:
        assert binding["letter_ids"] == ["letter"] and body["wake_events"] == []
        assert not ds.acknowledge_pending_wake(ctx, body)
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    limit.messages.append({"role": "tool", "content": replay.text, "tool_call_id": "old-wait"})
    ds.deliver_wake(ctx, {"result_meta": {"tool_result_meta": replay.meta}}, {}, replay.text)
    assert ds._load_state(ctx, "leaf")["pending_wake"] == pending
    assert not mail._ack_path(tmp_path, ctx.task_id).exists()
    working_checkpoint.save_ready(limit)
    state = ds._load_state(ctx, "leaf")
    assert not state["pending_wake"] and state["last_acknowledged_wake"]["payload"] == wake
    assert "letter" not in state["mailbox_acknowledged_ids"] and not ctx._owner_directives
    assert mail.mail_read_state(tmp_path, ctx.task_id, "letter", transcript_only=True) is (fault != "missing_letter")
    assert state["last_acknowledged_wake"].get("unresolved_mailbox_ids", []) == (
        ["letter"] if fault == "missing_letter" else [])
    reads = []
    def observe(_ctx, run_id, _wait, since, **_kwargs):
        reads.append(run_id)
        return json.dumps({"status": "waiting_on_user", "run_id": run_id, "last_seq": since + 1})
    scripted_reads(monkeypatch, observe)
    _asks(tmp_path, "leaf", "later-question")
    next_wake = json.loads(ds.run_wait(ctx, "leaf").text)
    assert reads == ["leaf"] and next_wake["supervision_wake_id"] != "older"
    assert "later-question" in str(next_wake["questions"]) and "replay_of" not in next_wake


def test_identical_runless_wakes_settle_only_the_addressed_run(tmp_path, stub_delegate_engines):
    from ouroboros import delegate_supervision as ds, owner_mailbox as mail, working_checkpoint
    from tests._run_wait_shared import _nanny, _run

    ctx, limit = _nanny(tmp_path)
    event = {"type": "addressed_message", "kind": "task_message", "msg_id": "lost",
             "text": "Keep the old exact wording."}
    wake = {"supervision_wake_id": "reused", "status": "progress", "wake_events": [event]}
    pending = {"wake_id": "reused", "payload": wake, "mailbox_ids": ["lost"], "attempt_key": "1"}
    for rid in ("one", "two"):
        _run(tmp_path, rid, ctx.task_id)
        ds._save_state(ctx, {"run_id": rid, "pending_wake": pending})
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    assert not ds.acknowledge_pending_wake(ctx, wake)
    delivery = ds.run_wait(ctx, "two")
    body = json.loads(delivery.text)
    assert body["wake_events"] == [event] and body["restored_mailbox"]["run_id"] == "two"
    assert body["restored_mailbox"]["unresolved"] == ["lost"]
    limit.messages.append({"role": "tool", "tool_call_id": "two-wait", "content": delivery.text})
    ds.deliver_wake(ctx, {"result_meta": {"tool_result_meta": delivery.meta}}, {}, delivery.text)
    assert ds._load_state(ctx, "two")["pending_wake"] == pending
    working_checkpoint.save_ready(limit)
    assert ds._load_state(ctx, "two")["last_acknowledged_wake"]["payload"] == wake
    assert ds._load_state(ctx, "one")["pending_wake"] == pending
    assert not mail._ack_path(tmp_path, ctx.task_id).exists() and not ctx._owner_directives


def test_same_wake_bytes_do_not_discharge_a_changed_list_of_letter_ids(tmp_path):
    import queue

    from ouroboros import delegate_supervision as ds, owner_mailbox as mail
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from tests._run_wait_shared import _nanny

    ctx, limit = _nanny(tmp_path)
    assert mail.write_task_message(tmp_path, "The original letter.", ctx.task_id, msg_id="first", source_task_id="writer")
    wake = {"run_id": "run", "supervision_wake_id": "w", "status": "progress"}
    ds._save_state(ctx, {"run_id": "run", "pending_wake": {"wake_id": "w", "payload": wake, "mailbox_ids": ["first"]}})
    mailbox_restore.restore_wake_mail(ctx, limit.messages)
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    assert ctx._mailbox_content_ids == {"first"}
    state = ds._load_state(ctx, "run")
    state["pending_wake"] = {**state["pending_wake"], "mailbox_ids": ["first", "missing"]}
    ds._save_state(ctx, state)
    assert not ds.acknowledge_pending_wake(ctx, wake)
    pending = ds._load_state(ctx, "run")["pending_wake"]
    view = mailbox_restore.wake_view(ctx, wake, pending=pending)
    assert pending == state["pending_wake"] and view["restored_mailbox"]["unresolved"] == ["missing"]
