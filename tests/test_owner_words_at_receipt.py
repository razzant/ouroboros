"""My human's words for running work are carried by the host at receipt (invariant 26, D6).

The moment a work receives my human's row, the host puts it in the mail of every live member of the work's tree
(``owner_relay.carry_at_receipt``: one ``host_carried`` entry by address, sleeping members woken as by any relay)
and frames it into every open session of the tree on a host thread (``delegate_interactions.place_at_receipt``),
with no turn of the root, of a nanny or of any focus, and no seat. The root's drain (``carry_to_members``) and the
starter's round top (``place_rows_into_runs``) stay as catch-up: one transaction chooses a row's frame for a run
whoever places it (``delegate_message_store.place_frame``), so nothing reaches a member or a session twice, and the
starter is told once what the host placed in its sessions (``untold``).
"""
from __future__ import annotations

import json
import queue

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.task_results import write_task_result
from tests.test_delegate_message import _Stub, _ctx, _install, _row


@pytest.fixture(autouse=True)
def _fresh_custody_memo():
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


def _open_run(tmp_path, run_id, *, task_id="t-nanny", root_task_id="t-root", route_id="some-route", source=""):
    """One run with a durable start, open (the census reads the held rows, never the memo)."""
    from ouroboros import delegate_custody as custody

    entry = custody.RunCustody(run_id=run_id, task_id=task_id, route_id=route_id, model="m", root_task_id=root_task_id,
                               project_id="prj", project_owned=False, source=source)
    assert custody.record_started(tmp_path, entry, shape={
        "access": "readonly", "mode": "ask", "isolation": "", "delegated": False, "root": "/r",
        "max_seconds": 90, "max_seconds_basis": "requested"})
    custody._CUSTODY.clear()
    return entry


def _tree(tmp_path):
    """A running root with a nanny, a scheduled grandchild not yet started and a settled helper; the nanny's live
    session, its session on a route with no live input, a review's session, the root's own session and a foreign
    tree's session; my human's row accepted in Main. Returns ``(row, address)``."""
    row = {"chat_id": 1, "direction": "in", "ts": "2026-11-09T10:00:00+00:00", "text": "Add the charts too",
           "client_message_id": "m-1"}
    dialogue_store.append(tmp_path, "chat", row)
    write_task_result(tmp_path, "t-root", "running", delegation_role="root", root_task_id="t-root")
    for task_id, status, parent in (("t-nanny", "running", "t-root"), ("t-child", "scheduled", "t-nanny"),
                                    ("t-done", "completed", "t-root")):
        write_task_result(tmp_path, task_id, status, delegation_role="subagent", parent_task_id=parent,
                          root_task_id="t-root")
    _open_run(tmp_path, "run-live")
    _open_run(tmp_path, "run-deaf", route_id="deaf-route")
    _open_run(tmp_path, "run-review", source="review_substrate")
    _open_run(tmp_path, "run-root", task_id="t-root")
    _open_run(tmp_path, "run-foreign", task_id="t-other", root_task_id="t-other")
    return row, format_address(row_address(row))


def _mail(tmp_path, task_id):
    return {entry["msg_id"]: entry for entry in drain_owner_entries(tmp_path, task_id, include_acknowledged=True)}


def _posts(stub):
    """``(run_id, idempotency key, frame address, the words line)`` of every POST the stub received."""
    out = []
    for post in stub.posts:
        wire = json.loads(post["text"])
        assert wire["kind"] == "task_message" and wire["frame"]["relation"] == "starter"
        out.append((post["run_id"], post["key"], wire["frame"]["owner_words"]["address"], wire["text"].split("\n")[1]))
    return out


def _receive(tmp_path):
    """The host's receipt of the row by the root (the room delivered it): its placement thread, joined."""
    from ouroboros.owner_relay import carry_row_at_receipt

    thread = carry_row_at_receipt(tmp_path, "t-root", 1, "m-1")
    if thread is not None:
        thread.join(10)
        assert not thread.is_alive()
    return thread


def _nanny(tmp_path):
    ctx = _ctx(tmp_path, "t-nanny")
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root", "delegation_role": "subagent"}
    ctx._owner_directives, ctx.task_attempt = [], 1
    return ctx


def _round_top(ctx, messages):
    from ouroboros.loop_round_limits import _drain_incoming_messages

    return _drain_incoming_messages(messages, queue.Queue(), ctx.drive_root, ctx.task_id, None, set(), owner_ctx=ctx,
                                    place_live_runs=True)


def _said(messages):
    """Everything the turn's messages say (the drain merges consecutive user content)."""
    return "\n".join(str(message["content"]) for message in messages)


def test_the_host_carries_at_receipt_with_no_turn_and_the_catch_up_sends_nothing_twice(tmp_path, monkeypatch):
    """No worker, no seat and no turn of the root or the nanny, both asleep (their drain and round top are tripwires
    here): the row reaches the mail of every live member, the grandchild scheduled but not started included, which
    wakes the nanny whatever it chose to sleep on, and each open session of the tree on a route that takes live
    input (the nanny's and the root's own) once, framed as its starter's; the route with no live input records
    ``unsupported`` with no POST; a review's session, a foreign tree's session and a settled member get nothing. A
    receipt retried writes and sends nothing more. Then the catch-up: the nanny's round top sends nothing twice and
    tells it once what the host placed in each of its sessions, while it still places a row of its own set the host
    never carried, and a helper it schedules now is born holding the row; the root's drain writes no member again."""
    from ouroboros import delegate_interactions, delegate_message_store as frames, model_sleep, owner_relay
    from ouroboros import owner_words as ow
    from ouroboros.owner_relay import carry_to_members, retain_owner_message

    row, address = _tree(tmp_path)
    assert model_sleep.wake_reason(_nanny(tmp_path), {"senders": ["t-other"]}) == ""  # asleep: nothing for it yet
    stub = _install(monkeypatch, _Stub(result={"outcome": "delivered", "reason": ""},
                                       harnesses=[_row(), _row("deaf-route", live_input="none")]))

    def tripwire(*_a, **_k):
        raise AssertionError("the receipt must not wait for a drain or a round top")

    with monkeypatch.context() as patch:
        for module, name in ((owner_relay, "carry_to_members"), (delegate_interactions, "place_rows_into_runs")):
            patch.setattr(module, name, tripwire)
        assert _receive(tmp_path) is not None
    carried = f"relay:carry:{address}:"
    for member in ("t-nanny", "t-child"):
        [entry] = _mail(tmp_path, member).values()
        assert entry["msg_id"] == carried + member and entry["host_carried"] == {"went_to": ""}
        assert entry["source_task_id"] == "t-root" and entry["owner_rows"][0]["text"] == row["text"]
    assert _mail(tmp_path, "t-done") == {} and _mail(tmp_path, "t-root") == {}
    posted = _posts(stub)
    assert sorted(run for run, *_rest in posted) == ["run-live", "run-root"]
    assert {(addr, line) for _run, _key, addr, line in posted} == {(address, row["text"])}
    assert [(f["outcome"], f["untold"]) for f in frames.placed_words(_nanny(tmp_path), "run-deaf")] == [("unsupported", True)]
    assert all(frames.placed_words(_nanny(tmp_path), run) == [] for run in ("run-review", "run-foreign"))
    sources = {json.loads(post["text"])["frame"]["source_task_id"] for post in stub.posts}
    assert sources == {"t-nanny", "t-root"}  # each frame names its session's starter
    _receive(tmp_path)  # a retried receipt: the members hold the id, each session's frame ended the row
    assert len(stub.posts) == 2 and all(len(_mail(tmp_path, member)) == 1 for member in ("t-nanny", "t-child"))
    assert model_sleep.wake_reason(_nanny(tmp_path), {"senders": ["t-other"]}) == "control:relay_entry"

    # The nanny wakes: its round top shows the carried row, writes its child nothing, sends nothing twice, tells it
    # once what the host placed, and still places a row of its own set the host never carried.
    nanny = _nanny(tmp_path)
    own = ow._row("Keep the API", source="owner_mailbox", carrier="held", task_id="t-root",
                  ts="2026-11-09T09:00:00+00:00", ref="chat 1 / m-0")
    nanny.task_metadata[ow.FIELD] = [own]
    own_address = owner_relay._pristine_form(nanny, own)[0]
    messages = [{"role": "user", "content": "your assignment"}]
    controls = _round_top(nanny, messages)
    assert controls["owner_carried_addressed"] == [address]
    assert set(_mail(tmp_path, "t-child")) == {carried + "t-child", f"relay:carry:{own_address}:t-child"}
    said = _said(messages)
    for fact in (f"[Host fact: my human's row {address} for session run run-live: delivered]",
                 f"[Host fact: my human's row {address} for session run run-deaf: unsupported (route_live_input_none)]"):
        assert said.count(fact) == 1
    assert [line for _run, _key, _addr, line in _posts(stub)[2:]] == ["Keep the API"]  # its own row, placed once
    assert {run for run, *_rest in _posts(stub)[2:]} == {"run-live"} and len(stub.posts) == 3
    _round_top(nanny, messages)
    assert _said(messages).count(f"{address} for session run run-live: delivered]") == 1  # told once
    assert len(stub.posts) == 3 and len(_mail(tmp_path, "t-child")) == 2  # nothing sent or written twice
    assert "chat 1 / m-1" in {r["ref"] for r in ow.governing_words_for_schedule(nanny)[ow.FIELD]}  # a new helper's birth
    # The root's drain, the catch-up of the same receipt: no member is written again, its session hears nothing new.
    root = _ctx(tmp_path, "t-root")
    root.task_metadata, root._owner_directives = {"root_task_id": "t-root"}, []
    retain_owner_message(root, row)
    carry_to_members(root, delivered=True)
    assert len(_mail(tmp_path, "t-nanny")) == 1 and len(_mail(tmp_path, "t-child")) == 2
    assert delegate_interactions.place_rows_into_runs(root) == [
        f"[Host fact: my human's row {address} for session run run-root: delivered]"]
    assert len(stub.posts) == 3 and delegate_interactions.place_rows_into_runs(root) == []


def test_delivery_unknown_replays_the_same_frame_bytes_whoever_places_next(tmp_path, monkeypatch):
    """The host's POST whose fate is unknown records ``delivery_unknown``; the next pass sends the SAME frame, its
    id and bytes, never a re-derivation: the starter's round top first (still unknown: no new fact line, the host's
    frame told), then a receipt retried by the host, which delivers it and makes it untold again, so the starter's
    next round top tells it the delivery once."""
    from ouroboros import delegate_message_store as frames
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _row_, address = _tree(tmp_path)
    lost = _install(monkeypatch, _Stub(error=ClaudexorUnavailable("daemon_unreachable", "reset", status_code=503)))
    _receive(tmp_path)
    first = {run: (key, text) for run, key, _addr, text in _posts(lost)}
    assert set(first) == {"run-live", "run-root"}
    assert [f["outcome"] for f in frames.placed_words(_nanny(tmp_path), "run-live")] == ["delivery_unknown"]
    nanny, messages = _nanny(tmp_path), [{"role": "user", "content": "your assignment"}]
    again = _install(monkeypatch, _Stub(error=ClaudexorUnavailable("daemon_unreachable", "reset", status_code=503)))
    _round_top(nanny, messages)
    assert [(run, key, text) for run, key, _addr, text in _posts(again)] == [("run-live", *first["run-live"])]
    assert "for session run run-live" not in _said(messages)  # the same outcome: no new fact line
    delivered = _install(monkeypatch, _Stub(result={"outcome": "delivered", "reason": ""}))
    _receive(tmp_path)  # the host's next receipt of the row: the same bytes under the same key, both sessions
    assert sorted((run, key, text) for run, key, _addr, text in _posts(delivered)) == sorted(
        (run, *sent) for run, sent in first.items())
    assert [(f["outcome"], f.get("untold")) for f in frames.placed_words(nanny, "run-live")] == [("delivered", True)]
    _round_top(nanny, messages)
    assert _said(messages).count(f"{address} for session run run-live: delivered]") == 1
    assert len(delivered.posts) == 2 and len(frames.placed_words(nanny, "run-live")) == 1  # one frame, ever


def test_a_route_without_live_input_hears_the_row_with_its_next_continuation(tmp_path, monkeypatch):
    """The session whose route takes no live input gets no POST at receipt (``unsupported``, ended); once the nanny
    holds the carried row, the continuation it starts carries the row's words whole, while a START body that already
    holds the row's stamp carries nothing of it."""
    from ouroboros import delegate_continuation
    from ouroboros.delegate_continuation import continuation_words

    row, address = _tree(tmp_path)
    stub = _install(monkeypatch, _Stub(result={"outcome": "delivered", "reason": ""},
                                       harnesses=[_row(), _row("deaf-route", live_input="none")]))
    _receive(tmp_path)
    assert "run-deaf" not in {run for run, *_rest in _posts(stub)}
    nanny = _nanny(tmp_path)
    _round_top(nanny, [{"role": "user", "content": "your assignment"}])
    monkeypatch.setattr(delegate_continuation, "line_start_bodies", lambda _root, _run: ["OBJECTIVE\nthe report"])
    tail = continuation_words(nanny, tmp_path, "run-deaf")
    assert f"\n{row['text']}\n" in tail and "chat 1 / m-1]" in tail
    monkeypatch.setattr(delegate_continuation, "line_start_bodies", lambda _root, _run: [
        f"OBJECTIVE\nthe report\n[{row['ts']} · owner · owner_mailbox of task t-root · chat 1 / m-1]\n{row['text']}"])
    assert row["text"] not in continuation_words(nanny, tmp_path, "run-deaf")


def test_place_frame_chooses_one_frame_per_row_and_run_for_every_placer(tmp_path):
    """Two placers of one row into one run get one frame: the second gets the first's frame back while it may be
    delivered (its id and bytes), nothing once an outcome ended the row, and a new frame only after the run had no
    live turn; a host frame is untold until marked, a starter's frame never is."""
    from ouroboros import delegate_message_store as frames

    entry = _open_run(tmp_path, "run-1")
    starter, host = _nanny(tmp_path), _nanny(tmp_path)
    words = {"address": "row:1@2026-11-09T10:00:00+00:00#0123456789ab", "chat_id": 1, "ts": "2026-11-09T10:00:00+00:00"}
    created, prior = frames.place_frame(host, entry, "the words", words, host=True)
    assert prior == "" and created["untold"] is True and created["outcome"] == "prepared"
    again, prior = frames.place_frame(starter, entry, "the words", words)
    assert prior == "prepared" and again["frame"]["message_id"] == created["frame"]["message_id"]
    assert again["wire_text"] == created["wire_text"]
    frames.record_outcome(starter, "run-1", created["frame"]["message_id"], "not_active", "between_turns")
    fresh, prior = frames.place_frame(starter, entry, "the words", words)
    assert prior == "not_active" and fresh["frame"]["message_id"] != created["frame"]["message_id"]
    assert "untold" not in fresh
    frames.record_outcome(starter, "run-1", fresh["frame"]["message_id"], "delivered", "")
    assert frames.place_frame(host, entry, "the words", words, host=True) == (None, "delivered")
    frames.mark_told(starter, "run-1", [created["frame"]["message_id"]])
    assert [f.get("untold") for f in frames.placed_words(starter, "run-1")] == [None, None]


def test_each_ingress_carries_at_receipt(tmp_path, monkeypatch):
    """The room's sole reader takes the row as addressed mail and its member hears it at once; the room's other live
    root hears it as context and so does its member, naming where it went; a head's ``tell_task`` quoting a row to an
    independent root carries the row on to that root's member at the steer, before any drain of the root."""
    from supervisor import queue as q
    from supervisor.steering import _handle_steer_task
    from ouroboros import server_owner_routing, server_routing_context as routing
    from ouroboros.owner_relay import quote_rows
    from ouroboros.projects_registry import create_project
    from tests.test_owner_relay import context
    from tests.test_project_routing_v664 import _ctx as routing_ctx
    from tests.test_routing_root_snapshot import _Now, _root, _route
    from tests.test_steer_relay import _supervisor_ctx

    room = create_project(tmp_path, "receipt-room")
    chat = int(room["chat_id"])
    running = {tid: {"task": _root(tid, chat_id=chat, project_id=room["id"], drive_root=str(tmp_path)), "started_at": 1.0}
               for tid in ("root-a", "root-b")}
    for tid in running:
        write_task_result(tmp_path, tid, "running", delegation_role="root", root_task_id=tid)
        write_task_result(tmp_path, f"{tid}-helper", "running", delegation_role="subagent", parent_task_id=tid,
                          root_task_id=tid)
    from ouroboros import owner_pause
    owner_pause.install_fence(tmp_path, "root-b", request_id="press-1")  # root-a is the room's sole reader
    ctx = routing_ctx(tmp_path, running=running)
    monkeypatch.setattr(server_owner_routing.threading, "Thread", _Now)
    monkeypatch.setattr(routing, "_recent_root_results", lambda *_a: ([], {}))
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    row = {"chat_id": chat, "direction": "in", "ts": "2026-11-09T10:01:00Z", "text": "and the charts?",
           "client_message_id": "m-2"}
    dialogue_store.append(tmp_path, "chat", row)
    _route(ctx, "m-2", chat_id=chat)
    address = format_address(row_address(row))
    assert "m-2:root-a" in _mail(tmp_path, "root-a")
    assert _mail(tmp_path, "root-a-helper")[f"relay:carry:{address}:root-a-helper"]["host_carried"] == {"went_to": ""}
    assert _mail(tmp_path, "root-b-helper")[f"relay:carry-context:{address}:root-b-helper"]["host_carried"] == {
        "went_to": "root-a"}
    # A tell_task quoting a Main row to an independent root: its helper holds the row at the steer.
    main_row = {"chat_id": 1, "direction": "in", "ts": "2026-11-09T10:02:00Z", "text": "Tell them: ship Friday",
                "client_message_id": "m-3"}
    dialogue_store.append(tmp_path, "chat", main_row)
    main_address = format_address(row_address(main_row))
    quotes, _error = quote_rows(context(tmp_path, main_row), [main_address])
    supervisor_ctx = _supervisor_ctx(tmp_path, [])
    supervisor_ctx.RUNNING, supervisor_ctx.persist_queue_snapshot = {"root-b": running["root-b"]}, lambda **_kw: True
    _handle_steer_task({"target_task_id": "root-b", "message": "yours now", "quotes": quotes, "relay_id": "r-3",
                        "routing_token": "r-3", "chat_id": 1, "client_message_id": "m-3",
                        "issuer": {"kind": "task", "task_id": "head"}}, supervisor_ctx)
    entry = _mail(tmp_path, "root-b-helper")[f"relay:carry:{main_address}:root-b-helper"]
    assert entry["host_carried"] == {"went_to": ""} and entry["owner_rows"][0]["text"] == main_row["text"]
    assert entry["source_task_id"] == "root-b"


@pytest.mark.parametrize("receipt_before_body", [False, True])
def test_frames_the_receipt_could_not_send_wait_for_the_body_which_sends_them_once_the_engine_answers(
        tmp_path, monkeypatch, receipt_before_body):
    """An engine that cannot be discovered at receipt sends nothing: the host writes the frames ahead (``prepared``,
    untold). A relay published without any body leaves the same durable obligation as a receipt beside a live
    body. Its next observation pass sends one frame per row and run. The starter's round top stays the
    idempotent catch-up: it sends nothing the body ended and tells the outcome once."""
    import time

    from ouroboros import delegate_collection, delegate_custody_current as current
    from ouroboros import delegate_message_store as frames, delegate_feed
    from ouroboros.gateways import claudexor as gw

    row, address = _tree(tmp_path)

    class Engine:  # the body's observing transport: every held run answers that it runs
        def get_run(self, run_id, timeout_sec=None):
            return {"summary": {"runId": run_id, "state": "running", "effectiveAccess": "readonly"}, "lastSeq": 1,
                    "pendingInteractions": []}

        def close(self):
            pass

    def undiscovered(*_a, **_k):
        raise gw.ClaudexorUnavailable("daemon_not_discovered", "no descriptor")

    def written(run_id):
        return [(f["outcome"], f.get("untold")) for f in frames.placed_words(
            frames.starter(tmp_path, current.run(tmp_path, run_id)), run_id)]

    def until(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.02)
        return predicate()

    monkeypatch.setattr(delegate_feed, "_TICK_SEC", 0.05)  # the body's pass, quickened
    monkeypatch.setattr(gw, "ClaudexorGateway", undiscovered)
    if receipt_before_body:
        from ouroboros import owner_relay
        from tests.test_owner_relay import context

        assert delegate_collection.body_runtime(tmp_path) is None
        quotes, error = owner_relay.quote_rows(context(tmp_path, row), [address])
        assert not error
        relay, _ = owner_relay.write_relay(tmp_path, "t-root", relay_id="outside-body", note="Carry these words",
                                          quotes=quotes, chat_id=1, teller_task_id="teller")
        threads, carry = [], owner_relay.carry_relay_at_receipt
        monkeypatch.setattr(owner_relay, "carry_relay_at_receipt",
                            lambda *args: threads.append(carry(*args)) or threads[-1])
        owner_relay.publish_relay(tmp_path, "t-root", relay, None)
        assert len(threads) == 1 and threads[0] is not None
        threads[0].join(10)
        assert not threads[0].is_alive()
        assert written("run-live") == written("run-root") == [("prepared", True)]
    runtime = delegate_collection.start_body_runtime(tmp_path, gateway_factory=Engine)
    try:
        if not receipt_before_body:
            _receive(tmp_path)
        assert written("run-live") == written("run-root") == [("prepared", True)], "written ahead, nothing sent"
        assert written("run-review") == written("run-foreign") == []
        assert until(lambda: runtime.observation("run-live").get("answered"))
        assert frames.pending_runs(tmp_path, ("run-live", "run-root")) == {"run-live", "run-root"}
        stub = _install(monkeypatch, _Stub(result={"outcome": "delivered", "reason": ""}))
        assert until(lambda: not frames.pending_runs(tmp_path, ("run-live", "run-root"))), "the body sends what receipt could not"
    finally:
        delegate_collection.stop_body_runtime(tmp_path, wait=True)
    posted = {(run, addr, line) for run, _key, addr, line in _posts(stub)}
    assert {("run-live", address, row["text"]), ("run-root", address, row["text"])} <= posted
    assert written("run-live") == written("run-root") == [("delivered", True)], "one frame each, ended, untold"
    sent = len(stub.posts)
    nanny, messages = _nanny(tmp_path), [{"role": "user", "content": "your assignment"}]
    _round_top(nanny, messages)
    assert len(stub.posts) == sent, "the round top sends nothing the body ended"
    assert _said(messages).count(f"[Host fact: my human's row {address} for session run run-live: delivered]") == 1
