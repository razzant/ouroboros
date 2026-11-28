"""Server coordination of link delivery, custody, and physical submission controls."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import (
    delegate_custody, delegate_resume, owner_pause, peer_channel, peer_channel_queue, peer_links, peer_replies,
    place_projection, places,
)
from ouroboros.gateways.peer import PeerGateway, _unavailable
from ouroboros.install_identity import acting_as
from ouroboros.tools.peer_delegate import _message_ouroboros
from tests.test_peer_channel import live_channel_pair
from tests.test_peer_messages import _poll_until, introduce


def test_worker_queues_message_without_opening_the_wire(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    ctx = SimpleNamespace(task_id="tool-writer", drive_root=pair.root_a, budget_drive_root=pair.root_a,
                          task_metadata={})
    wakes, posts = [], []
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    monkeypatch.setattr(peer_channel_queue, "_OUTCOME_WAIT_SEC", 0)
    monkeypatch.setattr(peer_channel_queue, "_WORKER_WAKE_QUEUE", SimpleNamespace(put=wakes.append))
    monkeypatch.setattr(PeerGateway, "send_message", lambda *a, **kw: posts.append(kw))
    with acting_as(pair.root_a):
        result = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="hello").text)
        assert posts == []
        assert result["ok"] is True and result["status"] == result["code"] == "delivery_pending", result
        assert "retry_of" in result["note"]
        assert peer_channel_queue.queued_message(pair.root_a, pair.link_id, result["letter_id"])["text"] == "hello"
    assert [row["link_id"] for row in wakes] == [pair.link_id]


def test_tool_observation_budget_does_not_wait_for_a_stalled_server_pass(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    root, lid = pair.root_a, pair.link_id
    ctx = SimpleNamespace(task_id="bounded-tool", drive_root=root, budget_drive_root=root, task_metadata={})
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    key = (str(root.resolve()), lid)

    def stall(*_args, **_kw):
        started.set()
        release.wait(2.0)  # fail without deadlocking even if the caller still runs this pass inline
        finished.set()
        raise OSError("injected server pass failure")

    monkeypatch.setattr(peer_channel_queue, "_OUTCOME_WAIT_SEC", 0.1)
    monkeypatch.setattr(peer_channel_queue, "_flush_one_queued", stall)
    try:
        with acting_as(root):
            result = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Bounded note").text)
        assert started.wait(1.0), "the server did not begin its pass"
        assert not finished.is_set(), "the tool waited for the stalled server pass before observing the queue"
        assert peer_channel_queue.queued_message(root, lid, result["letter_id"])["pending"] is True
    finally:
        release.set()
        _poll_until(lambda: key not in peer_channel_queue._FLUSHING_LINKS, timeout=3.0, label="server pass retired")


def test_task_stop_is_read_inside_physical_admission(tmp_path, monkeypatch):
    sends = []
    stopped = False

    def finish_admission(_source, submit):
        nonlocal stopped
        stopped = True
        return submit()

    monkeypatch.setattr(owner_pause, "run_operation", finish_admission)
    monkeypatch.setattr(peer_channel_queue, "_check_task_stop", lambda *a: (
        ("task_stopped", "nothing was sent") if stopped else None))
    gate = peer_channel_queue._queued_task_gate(tmp_path, "origin", "ancestor")
    with pytest.raises(peer_replies.AttemptHeld, match="nothing was sent"):
        gate(lambda: sends.append("sent"))
    assert sends == []


def test_last_channel_session_refuses_wrap():
    with pytest.raises(peer_links.PeerLinkError) as exc:
        peer_channel.next_channel_session("ffffffffffff")
    assert exc.value.code == "channel_session_exhausted"


def test_loopback_explicit_retry_preserves_receiver_deadline_and_identity(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    ctx = SimpleNamespace(task_id="url-retry", drive_root=pair.root_a, budget_drive_root=pair.root_a,
                          task_metadata={"work_kind": "owner"})
    posts = []

    def send(_self, text, **kw):
        posts.append((text, kw["letter_id"], kw["reply_to"]))
        if len(posts) == 1:
            exc = _unavailable("presence_rate_limited", "wait", 429, refusal="presence_rate_limited")
            exc.retry_after = 60.0
            raise exc

    monkeypatch.setattr(PeerGateway, "send_message", send)
    with acting_as(pair.root_a):
        def call(**kw):
            return json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="same", **kw).text)

        first = call()
        lid = first["letter_id"]
        saved = peer_links.sending_record(pair.root_a, pair.link_id, lid)
        assert saved["not_before"] > time.time() + 50
        sleepers = []
        monkeypatch.setattr(peer_channel_queue, "trigger_link_flush", peer_channel_queue.flush_link_sync)
        monkeypatch.setattr(peer_channel_queue, "_start_flush_thread", lambda _key, _state, fn: sleepers.append(fn))
        early = call(retry_of=lid)
        assert early["status"] == "queued" and early["not_before"] == saved["not_before"]
        assert len(posts) == len(sleepers) == 1

        # Read persisted time in a new coordinator generation (no process-local elapsed-time cache).
        key = (str(pair.root_a.resolve()), pair.link_id, lid)
        monkeypatch.delitem(peer_channel_queue._UNPERSISTED_RETRY, key, raising=False)
        monkeypatch.setattr(peer_channel_queue, "time", SimpleNamespace(
            time=lambda: saved["not_before"] + 1, monotonic=time.monotonic, sleep=time.sleep))
        monkeypatch.setattr(peer_channel_queue, "_wait_for_deadline", lambda *_: None)
        sleepers.pop()()
        assert call(retry_of=lid)["status"] == "received"
        assert posts == [("same", lid, "")] * 2


@pytest.mark.parametrize("hold", ["silent", "paused", "panic"])
@pytest.mark.parametrize("explicit", [False, True], ids=["repeat", "retry_of"])
def test_held_loopback_retry_stays_parked_after_return(tmp_path, monkeypatch, hold, explicit):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    root, link = pair.root_a, pair.link_id
    ctx = SimpleNamespace(task_id="parked-retry", drive_root=root, budget_drive_root=root, task_metadata={})
    posts, switch = [], ["off"]

    def send(_self, text, **kw):
        posts.append((kw["letter_id"], text, kw["reply_to"]))
        if len(posts) == 1:
            exc = _unavailable("presence_rate_limited", "wait", 429, refusal="presence_rate_limited")
            exc.retry_after = 0.0
            raise exc

    def set_hold(active):
        if hold == "panic":
            switch[0] = "panic" if active else "off"
            if not active:
                peer_replies.resume_returned(root, link_id=link)
        else:
            places.edit_place(root, pair.place_a, "set_state", author=places.AUTHOR_OWNER,
                              state=hold if active else places.STATE_ACTIVE)

    monkeypatch.setattr(PeerGateway, "send_message", send)
    monkeypatch.setattr(place_projection, "switch_state", lambda _root: switch[0])
    # A skipped pass under Panic has no outcome to report; keep that observation wait short.
    monkeypatch.setattr(peer_channel_queue, "_OUTCOME_WAIT_SEC", 0.4)
    key = (str(root.resolve()), link)
    with acting_as(root):
        first = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Original note").text)
        lid = first["letter_id"]
        active = peer_channel_queue._queued_path(root, link, lid)
        parked = active.with_suffix(".parked")
        original = parked.read_bytes()
        _poll_until(lambda: key not in peer_channel_queue._FLUSHING_LINKS, label="first pass done")
        set_hold(True)
        held = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Original note",
                                             **({"retry_of": lid} if explicit else {})).text)
        intact = parked.is_file() and parked.read_bytes() == original and not active.exists()
        assert posts == [(lid, "Original note", "")]
        set_hold(False)
        peer_channel_queue.flush_link_sync(root, link)  # the same pass made by the next wake
        _poll_until(lambda: key not in peer_channel_queue._FLUSHING_LINKS, label="return pass done")
        assert posts == [(lid, "Original note", "")], "return activated a retry that had been refused"
        assert intact, held
        assert (held["status"], held["reason"]) == ("refused", "panic_hold" if hold == "panic" else f"place_{hold}"), held


def test_held_admitted_tool_note_reports_same_pending_letter_and_delivers_on_wake(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    root, link = pair.root_a, pair.link_id
    ctx = SimpleNamespace(task_id="held-before-wire", drive_root=root, budget_drive_root=root, task_metadata={})
    text = "A note admitted but held before wire"
    lid = peer_replies.message_letter_id(link, ctx.task_id, "", text)
    posts = []

    def unavailable(*args, **kwargs):
        raise peer_links.PeerLinkError("peer_links_unreadable", "sent record cannot be saved now", 503)

    monkeypatch.setattr(PeerGateway, "send_message", lambda _self, note, **kw: posts.append((kw["letter_id"], note)))
    with acting_as(root):
        with monkeypatch.context() as failure:
            failure.setattr(peer_links, "record_sending", unavailable)
            held = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text=text).text)
            assert posts == [] and peer_links.sending_record(root, link, lid) is None
            assert peer_channel_queue.queued_message(root, link, lid)["pending"] is True
            _poll_until(lambda: (str(root.resolve()), link) not in peer_channel_queue._FLUSHING_LINKS,
                        label="held pass done")
        peer_channel_queue.flush_link_sync(root, link)
        _poll_until(lambda: posts == [(lid, text)], label="the admitted note is posted on the next wake")
        readback = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text=text, retry_of=lid).text)
        assert (readback["status"], readback["letter_id"]) == ("received", lid)
        assert posts == [(lid, text)]
        assert (held["status"], held["reason"], held.get("letter_id")) == ("queued", "sending_unrecorded", lid), held
        assert "sent record cannot be saved" in held["note"] and lid in held["note"] and "retry_of" in held["note"]


def test_two_outputs_of_one_producer_have_independent_retry_deadlines(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    root, link = pair.root_a, pair.link_id
    account = peer_links.read_registry(root)["links"][link]["peer_install_id"]
    speaker = SimpleNamespace(place_id=pair.place_a, conversation_id=link, account_id=account)
    durations = {"interim": 60.0, "final": 15.0}
    posts = []
    wall = [time.time()]
    monkeypatch.setattr(peer_channel_queue, "time", SimpleNamespace(time=lambda: wall[0],
                         monotonic=lambda: wall[0], sleep=time.sleep))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (1.0,))
    monkeypatch.setattr(peer_replies, "_said_held", lambda *a: False)
    monkeypatch.setattr(peer_replies, "resolve_outgoing", lambda *a, **kw: (True, None))
    monkeypatch.setattr(peer_replies, "check_speech_basis", lambda *a, **kw: None)
    monkeypatch.setattr(peer_replies, "earlier_state", lambda r, l, ac, id: (
        "none", set(), peer_links.sending_record(r, l, id) or {}, ""))
    monkeypatch.setattr(peer_replies, "_tool_sent", lambda *a: False)
    monkeypatch.setattr(peer_links, "owed_record", lambda *a: {"approved": {
        hashlib.sha256(s.encode()).hexdigest(): "basis" for s in durations}})

    def wire(r, p, *, letter_id, text, reply_to, origin, **kw):
        posts.append(text)
        if posts.count(text) == 1:
            peer_links.record_sending(r, link, letter_id, account_id=account, text=text,
                                      reply_to=reply_to, origin=origin, place_id=p)
            return {"state": "retry", "retry_after": durations[text]}
        return {"state": "delivered", "recorded": True}

    monkeypatch.setattr(peer_replies, "send_peer_message", wire)
    with acting_as(root):
        def output(text):
            return peer_replies._deliver_output(root, speaker, "room", producer="one", output_ref=text,
                           text=text, reply_to="", source_event_id="event", kind="presence")

        assert output("interim") == (False, 60.0)
        assert output("final") == (False, 15.0)
        monkeypatch.setattr(peer_channel_queue, "_UNPERSISTED_RETRY", {})  # process restarted
        wall[0] += 16.0
        assert output("interim") == (False, 44.0)
        assert output("final") == (True, None)
        assert posts == ["interim", "final", "final"]
        wall[0] += 45.0
        assert output("interim") == (True, None)


def test_revocation_wins_over_retry_deadline_repair(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    root, lid, letter = pair.root_a, pair.link_id, "deadline-repair"
    key = (str(root.resolve()), lid, letter)
    account = peer_links.read_registry(root)["links"][lid]["peer_install_id"]
    store = peer_links.set_sending_deadline

    def store_then_revoke(*args):
        assert store(*args)
        peer_links.revoke(root, lid)
        assert key not in peer_channel_queue._UNPERSISTED_RETRY
        return True

    monkeypatch.setattr(peer_links, "set_sending_deadline", store_then_revoke)
    with acting_as(root):
        peer_links.record_sending(root, lid, letter, account_id=account, text="held note", reply_to="",
                                  origin={"kind": "tool", "task_id": "producer"}, place_id=pair.place_a)
        monkeypatch.setitem(peer_channel_queue._UNPERSISTED_RETRY, key, (time.time() + 60, time.monotonic() + 60, 1))
        outcome = peer_channel_queue.deadline(root, lid, letter, {})
        assert key not in peer_channel_queue._UNPERSISTED_RETRY
        assert outcome[:2] == (False, None)


def test_link_pass_attempts_owed_before_sleeping_on_queue_and_preserves_old_deadline(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "off")
    monkeypatch.setattr(peer_channel_queue, "queued_messages", lambda *a: [("link", {"letter_id": "queued"})])
    monkeypatch.setattr(peer_channel_queue, "_flush_one_queued", lambda *a, **kw: (
        events.append("queued"), 3600.0))
    monkeypatch.setattr(peer_links, "owed_refs", lambda *a: [("link", "old"), ("link", "new")])
    monkeypatch.setattr(peer_replies, "_one_pass", lambda _root, _link, ref: (
        events.append(ref), 60.0 if ref == "old" else None))
    before = time.monotonic()
    deadline = peer_channel_queue._delivery_pass(tmp_path, "link")
    assert before + 60 <= deadline <= time.monotonic() + 60
    assert events == ["queued", "old", "new"]


def test_recovered_peer_stop_stays_owed_only_when_connect_proves_nothing_left(tmp_path, monkeypatch):
    run = SimpleNamespace(run_id="remote-assignment", task_id="origin", settled=False,
                          resume_start_state="returned", engine_ref={"kind": "peer", "link_id": "link"})
    monkeypatch.setattr(delegate_resume, "prepare_recovered_cancel", lambda *a: None)
    monkeypatch.setattr(delegate_custody, "_cancel_result", lambda *a, **kw: kw)

    def unavailable(*a, **kw):
        raise _unavailable("peer_unreachable", "could not read back")

    monkeypatch.setattr(delegate_resume, "stop_read_run", unavailable)

    def case(nothing_left):
        def cancel(*a, **kw):
            exc = _unavailable("peer_unreachable", "no acknowledgement")
            exc.channel_connect_failed = nothing_left
            raise exc
        gw = SimpleNamespace(cancel_run=cancel)
        return delegate_custody.cancel_and_verify(tmp_path, gw, run, reason="stop", owner_stop=True)

    assert case(True)["control_owed"] is True
    assert case(False)["control_owed"] is False


def test_channel_explicit_retry_respects_saved_deadline(tmp_path, monkeypatch):
    """An ordinary retry of the same approved letter must honor its existing delivery schedule."""
    from ouroboros import peer_letters

    if not peer_channel.is_door_supported():
        pytest.skip("Unix-domain letters door")
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as pair:
        ctx = SimpleNamespace(task_id="scheduled-letter", drive_root=pair.root_a,
                              budget_drive_root=pair.root_a, task_metadata={"work_kind": "owner"})
        posts = []
        monkeypatch.setattr(peer_channel_queue, "trigger_link_flush", peer_channel_queue.flush_link_sync)
        sleepers = []
        monkeypatch.setattr(peer_channel_queue, "_start_flush_thread", lambda _key, _state, fn: sleepers.append(fn))
        monkeypatch.setattr(PeerGateway, "send_message", lambda _self, _text, **kw: posts.append(kw["letter_id"]))
        with acting_as(pair.root_a):
            lid = "msg-" + "7" * 32
            account = peer_links.read_registry(pair.root_a)["links"][pair.link_id]["peer_install_id"]
            basis = peer_letters.speech_basis(pair.root_a, pair.place_a, kind="owner")
            # A previous delivery pass already recorded the receiver's deadline. Seed that
            # same durable precondition on both trees rather than requiring the initial send.
            peer_channel_queue.record_queued_message(
                pair.root_a, pair.link_id, lid, place_id=pair.place_a, account_id=account,
                text="note", reply_to="", origin={"kind": "tool", "task_id": ctx.task_id},
                approved_basis=basis, coverage_kind="owner", wake=False)
            peer_links.record_sending(pair.root_a, pair.link_id, lid, account_id=account,
                place_id=pair.place_a, text="note", reply_to="", origin={"kind": "tool", "task_id": ctx.task_id})
            assert peer_links.set_sending_deadline(pair.root_a, pair.link_id, lid, time.time() + 60, 1)
            record = peer_links.sending_record(pair.root_a, pair.link_id, lid)
            assert record is not None and record["not_before"] > time.time() + 45
            again = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="note", retry_of=lid).text)
        assert posts == [], "retry before the saved deadline must not dispatch"
        assert again["status"] == "queued" and again["not_before"] == record["not_before"]
        assert len(sleepers) == 1
        peer_channel_queue.retire_link_waits(pair.root_a, pair.link_id)
        peer_channel_queue._FLUSHING_LINKS.pop((str(pair.root_a.resolve()), pair.link_id), None)


def _scheduler_fixture(root, monkeypatch):
    key = (str(root.resolve()), "link")
    monkeypatch.setattr(place_projection, "switch_state", lambda _root: "off")
    monkeypatch.setattr(peer_channel_queue, "_has_pending_files_for_link", lambda root, lid, *, if_unreadable=False:
                        not if_unreadable or bool(peer_channel_queue.queued_messages(root, lid)
                                                  or peer_links.owed_refs(root, lid)))
    monkeypatch.setattr(peer_channel_queue, "queued_messages", lambda *a: [])
    monkeypatch.setattr(peer_links, "owed_refs", lambda *a: [])
    return lambda: key not in peer_channel_queue._FLUSHING_LINKS


def test_new_producer_wake_keeps_older_due_reply(tmp_path, monkeypatch):
    idle = _scheduler_fixture(tmp_path, monkeypatch)
    pending, posts, deadline = ["older"], [], [None]
    monkeypatch.setattr(peer_links, "owed_refs", lambda *a: [("link", ref) for ref in pending])
    monkeypatch.setattr(peer_replies, "_peer_speaker", lambda *a: SimpleNamespace(conversation_id="link"))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.01, 0.01))

    def deliver(_root, _link, ref, *_gate):
        if ref == "older":
            if deadline[0] is None:
                deadline[0] = time.monotonic() + 0.2
                pending.append("newer")
                peer_replies.deliver_owed(tmp_path, "newer", record={})
            remaining = deadline[0] - time.monotonic()
            if remaining > 0:
                return False, remaining
        pending.remove(ref)
        posts.append(ref)
        return True, None

    monkeypatch.setattr(peer_replies, "_one_pass", deliver)
    peer_channel_queue.trigger_link_flush(tmp_path, "link")
    _poll_until(idle, timeout=5.0, label="link retired after follow-up")
    assert posts == ["newer", "older"] and not pending


def test_ready_owed_reply_finishes_before_waiting_on_a_queued_letter(tmp_path, monkeypatch):
    from threading import Event

    idle = _scheduler_fixture(tmp_path, monkeypatch)
    events, queued = [], [("link", {"letter_id": "queued"})]

    class ObserveWait(Event):
        def wait(self, timeout=None):
            events.append("wait")
            queued.clear()
            return True

    threading = peer_channel_queue.threading
    monkeypatch.setattr(peer_channel_queue, "threading", SimpleNamespace(
        Event=ObserveWait, Thread=threading.Thread, TIMEOUT_MAX=threading.TIMEOUT_MAX))
    monkeypatch.setattr(peer_channel_queue, "queued_messages", lambda *a: list(queued))
    monkeypatch.setattr(peer_channel_queue, "_flush_one_queued", lambda *a, **kw: ("retry", 3600.0))
    monkeypatch.setattr(peer_links, "owed_refs", lambda *a: [("link", "ready")] if "reply" not in events else [])

    def deliver(*_args):
        events.append("reply")
        return True, None

    monkeypatch.setattr(peer_replies, "_one_pass", deliver)
    peer_channel_queue.trigger_link_flush(tmp_path, "link")
    _poll_until(idle, timeout=5.0, label="link retired after scheduled wait")
    assert events == ["reply", "wait"]


def test_verified_exchange_in_same_pass_wakes_other_pending_letter(tmp_path, monkeypatch):
    idle = _scheduler_fixture(tmp_path, monkeypatch)
    pending, arrived = ["first", "second"], []
    monkeypatch.setattr(peer_channel_queue, "queued_messages", lambda *a: [
        ("link", {"letter_id": letter}) for letter in pending])

    def deliver(root, link, row, **_kw):
        letter = row["letter_id"]
        if letter == "first" and "second" not in arrived:
            return "pending", None
        pending.remove(letter)
        arrived.append(letter)
        peer_channel_queue.note_verified_exchange(root, link)
        return "done", None

    monkeypatch.setattr(peer_channel_queue, "_flush_one_queued", deliver)
    peer_channel_queue.trigger_link_flush(tmp_path, "link")
    _poll_until(idle, timeout=5.0, label="link retired after confirmed progress")
    assert arrived == ["second", "first"] and not pending


def test_recovered_stop_crosses_reconnected_door_once(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current, peer_letters
    from ouroboros.task_results import load_task_result, write_task_result
    from tests._peer_letters_shared import ready_queue
    from tests.test_peer_delegate import _seed_recovered_peer_run

    if not peer_channel.is_door_supported():
        pytest.skip("Unix-domain letters door")
    with live_channel_pair(tmp_path, monkeypatch) as pair:
        ready_queue(monkeypatch, pair.root_b)
        lid = "assign-stop-reconnect"
        receiver_task = peer_letters.letter_task_id(pair.place_b, lid)
        with acting_as(pair.root_a):
            gateway = PeerGateway(peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a))
        attempts, cancel = [], gateway.cancel_run

        def count_cancel(run_id, **kw):
            attempts.append(run_id)
            return cancel(run_id, **kw)

        monkeypatch.setattr(gateway, "cancel_run", count_cancel)
        try:
            with acting_as(pair.root_a):
                assert gateway.start_run({"letter_id": lid, "text": "Count test items", "protocol": 1})["runId"] == lid
            with acting_as(pair.root_b):
                write_task_result(pair.root_b, receiver_task, "running")
            entry = _seed_recovered_peer_run(pair, lid, resume_start_state="running")
            pair.fwd_a_to_b.set_mode("absent")
            with acting_as(pair.root_a):
                first = delegate_custody.cancel_and_verify(pair.root_a, gateway, entry, reason="stop")
            with acting_as(pair.root_b):
                assert load_task_result(pair.root_b, receiver_task)["status"] == "running"
            pair.fwd_a_to_b.set_mode("forward")
            with acting_as(pair.root_a):
                fresh = delegate_custody_current.run(pair.root_a, lid)
                second = delegate_custody.cancel_and_verify(pair.root_a, gateway, fresh, reason="stop")
            with acting_as(pair.root_b):
                assert load_task_result(pair.root_b, receiver_task)["status"] == "cancelled", (first, second)
            assert first.get("control_owed") is True
            assert second["outcome"] == delegate_custody.CANCEL_CONFIRMED
            with acting_as(pair.root_a):
                settled = delegate_custody_current.run(pair.root_a, lid)
                assert delegate_custody.cancel_and_verify(pair.root_a, gateway, settled,
                                                          reason="stop")["outcome"] == delegate_custody.CANCEL_CONFIRMED
            assert attempts == [lid, lid]  # one offline, one after reconnect, none after confirmation
        finally:
            gateway.close()


@pytest.mark.parametrize("inline", [False, True])
def test_spurious_pass_without_wait_keeps_link_wake_at_previous_deadline(tmp_path, monkeypatch, inline):
    delivery = peer_channel_queue
    wake = threading.Event()
    state = {"wake_event": wake}
    key = (str(tmp_path), "link")
    times, waits, continuations = [], [], []
    now = 100.0

    def work(_root, _lid, *, results=None):
        times.append(now)
        return 160.0 if len(times) == 1 else None

    def wait(event, at):
        nonlocal now
        waits.append(at)
        if len(waits) == 1:
            event.set()  # a different letter wakes the link before this letter's deadline
        else:
            now = at

    monkeypatch.setattr(delivery, "_delivery_pass", work)
    monkeypatch.setattr(delivery, "_has_pending_files_for_link", lambda *_args, **_kw: now < 160.0)
    monkeypatch.setattr(delivery, "_wait_for_deadline", wait)
    monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, fn: continuations.append(fn))
    monkeypatch.setattr(delivery, "time", SimpleNamespace(monotonic=lambda: now))
    delivery._run_claimed_flush(tmp_path, "link", key, state, first_pass_results={} if inline else None)
    if inline:
        continuations.pop()()
    assert times == [100.0, 100.0, 160.0]
    assert waits == [160.0, 160.0]
