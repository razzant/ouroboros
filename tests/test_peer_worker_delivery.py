"""A worker publishes a tool-message intent; the server owns its wire pass and outcome."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import peer_channel_queue as delivery, peer_letters, peer_links, peer_replies, place_projection
from ouroboros.gateways.peer import PeerGateway, _unavailable
from ouroboros.install_identity import acting_as
from ouroboros.tools.peer_delegate import _message_ouroboros
from tests._peer_letters_shared import introduce
from tests.test_peer_messages import _both_sides, _enable_consciousness


@pytest.fixture
def url_pair(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    with _both_sides(pair, monkeypatch):
        yield pair


def _tool(pair, *, task_id="message-origin"):
    ctx = SimpleNamespace(task_id=task_id, drive_root=pair.root_a, budget_drive_root=pair.root_a,
                          task_metadata={"work_kind": "owner"})

    def call(text="A local test note", **kw):
        with acting_as(pair.root_a):
            return json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text=text, **kw).text)

    return call


@pytest.fixture
def worker_and_server(monkeypatch):
    """Run the two roles in separate threads, with only an event between the caller and the real server pass."""
    sender = threading.current_thread()
    from ouroboros import utils

    in_worker = lambda: threading.current_thread() is sender
    monkeypatch.setattr(delivery, "in_worker_process", in_worker)
    monkeypatch.setattr(utils, "in_worker_process", in_worker)
    threads = []

    def bind(pair, *, before_pass=lambda: None):
        def publish(event):
            assert event["type"] == "peer_link_wake" and event["link_id"] == pair.link_id

            def run():
                before_pass()
                delivery.flush_link_sync(pair.root_a, pair.link_id)

            t = threading.Thread(target=run, daemon=True)
            threads.append(t)
            t.start()

        monkeypatch.setattr(delivery, "_WORKER_WAKE_QUEUE", SimpleNamespace(put=publish))

    yield bind
    for t in threads:
        t.join(timeout=12)
        assert not t.is_alive()


def test_worker_reports_healthy_url_receipt_after_supervisor_wake_delay(url_pair, monkeypatch, worker_and_server):
    pair = url_pair
    worker_and_server(pair, before_pass=lambda: time.sleep(1.0))
    result = _tool(pair)()
    assert result["status"] == "received", result


def test_worker_url_unconfirmed_has_retry_of_and_retries_original_once(url_pair, monkeypatch, worker_and_server):
    pair = url_pair
    posts, send = [], PeerGateway.send_message
    ready = False
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0.4)

    def intermittent(self, text, **kw):
        posts.append((text, kw["letter_id"], kw["reply_to"]))
        if not ready:
            raise _unavailable("peer_unreachable", "local receiver unavailable")
        return send(self, text, **kw)

    monkeypatch.setattr(PeerGateway, "send_message", intermittent)
    worker_and_server(pair)
    call = _tool(pair)
    first = call()
    assert first["status"] == "refused" and first["reason"] == "unconfirmed", first
    lid = first["letter_id"]
    assert f"retry_of={lid!r}" in first["detail"], first
    assert len(posts) == 3
    ready = True
    assert call(retry_of=lid)["status"] == "received"
    assert call(retry_of=lid)["status"] == "received"
    assert posts == [("A local test note", lid, "")] * 4


@pytest.mark.parametrize(("switch", "reason"), [("panic", "panic_hold"), ("unknown", "switch_unreadable")])
def test_switch_hold_after_admission_is_reported_by_the_pass(url_pair, monkeypatch, switch, reason):
    root, link = url_pair.root_a, url_pair.link_id
    task_id, words = "switch-after-admission", "A note that can wait"
    lid = peer_replies.message_letter_id(link, task_id, "", words)
    state, elapsed, posts = "off", 0.0, []
    send, read_switch = PeerGateway.send_message, place_projection.switch_state

    def wake_after_switch(r, l):
        nonlocal state
        assert (r, l) == (root, link)
        assert delivery.queued_message(root, link, lid)["pending"] is True
        state = switch  # precisely between persisting the letter and the server pass
        delivery.flush_link_sync(r, l)

    def advance(delay):
        nonlocal elapsed
        elapsed += delay

    def wire(self, text, **kw):
        posts.append((kw["letter_id"], text, kw["reply_to"]))
        return send(self, text, **kw)

    call = _tool(url_pair, task_id=task_id)
    monkeypatch.setattr(PeerGateway, "send_message", wire)
    monkeypatch.setattr(place_projection, "switch_state", lambda r: state if r == root else read_switch(r))
    with monkeypatch.context() as held:
        held.setattr(delivery, "trigger_link_flush", wake_after_switch)
        held.setattr(delivery, "time", SimpleNamespace(time=time.time, monotonic=lambda: elapsed, sleep=advance))
        result = call(words)
    assert posts == []
    assert (result["status"], result["reason"], result["letter_id"]) == ("queued", reason, lid), result
    assert elapsed == 0 and "retry_of" in result["note"]
    assert delivery.queued_message(root, link, lid)["pending"] is True
    state = "off"
    delivery.flush_link_sync(root, link)
    assert call(words, retry_of=lid)["status"] == "received"
    assert posts == [(lid, words, "")]
    assert delivery.queued_message(root, link, lid) is None


@pytest.mark.parametrize("interleave", ["idle", "in_flight"])
def test_pending_url_retry_observes_the_woken_pass_not_the_prior_hold(url_pair, monkeypatch, interleave):
    root, link = url_pair.root_a, url_pair.link_id
    words = "One pending note"
    call = _tool(url_pair, task_id="url-fresh-pass")
    send, real_record = PeerGateway.send_message, peer_links.record_sending
    posts, wakes = [], []
    entered, release = threading.Event(), threading.Event()
    recovering = False
    old_pass = None

    def wire(self, text, **kw):
        posts.append((kw["letter_id"], text, kw["reply_to"]))
        return send(self, text, **kw)

    def fail_before_wire(*a, **kw):
        if recovering:
            return real_record(*a, **kw)
        if old_pass is not None:
            entered.set()
            assert release.wait(5), "the retry did not observe the sleeping pass"
        raise peer_links.PeerLinkError("peer_links_unreadable", "cannot write the send record now", 503)

    monkeypatch.setattr(PeerGateway, "send_message", wire)
    monkeypatch.setattr(peer_links, "record_sending", fail_before_wire)
    with monkeypatch.context() as held:
        held.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
        first = call(words)
    lid = first["letter_id"]
    prior = delivery.queued_message(root, link, lid)
    assert (first["status"], first["reason"]) == ("queued", "sending_unrecorded"), first
    assert prior["pending"] is True and prior["last_outcome"]["code"] == "sending_unrecorded"
    assert posts == []

    def run_pending_pass(_delay):
        # The async wake has been published but the new pass hasn't run at poll 1.
        assert wakes == [(root, link)] and posts == []
        delivery.flush_link_sync(root, link)

    def publish(r, l):
        nonlocal recovering
        if not recovering and old_pass is not None:
            # A pass started before retry can write a *different* at during the call.
            # Its hold is still stale; poll must await the pass requested by this wake.
            release.set()
            old_pass.join(5)
            assert not old_pass.is_alive()
            assert delivery.queued_message(root, link, lid)["at"] != prior["at"]
        recovering = True
        wakes.append((r, l))

    try:
        if interleave == "in_flight":
            old_pass = threading.Thread(target=delivery.flush_link_sync, args=(root, link), daemon=True)
            old_pass.start()
            assert entered.wait(5)
        with monkeypatch.context() as pending:
            pending.setattr(delivery, "trigger_link_flush", publish)
            pending.setattr(delivery, "time", SimpleNamespace(time=time.time, monotonic=time.monotonic,
                                                             sleep=run_pending_pass))
            again = call(words, retry_of=lid)
    finally:
        release.set()
        if old_pass is not None:
            old_pass.join(5)
    assert (again["status"], again["letter_id"]) == ("received", lid), again
    assert call(words, retry_of=lid)["status"] == "received"
    assert posts == [(lid, words, "")]


def test_url_retry_rearms_429_deadline_after_unreadable_switch(url_pair, monkeypatch):
    root, link = url_pair.root_a, url_pair.link_id
    wall, mono = time.time(), 100.0
    state, posts, sleepers = "off", [], []
    read_switch = place_projection.switch_state

    def wire(_self, _text, **kw):
        posts.append(kw["letter_id"])
        if len(posts) == 1:
            err = _unavailable("presence_rate_limited", "wait", 429, refusal="presence_rate_limited")
            err.retry_after = 60.0
            raise err

    monkeypatch.setattr(PeerGateway, "send_message", wire)
    monkeypatch.setattr(place_projection, "switch_state", lambda r: state if r == root else read_switch(r))
    monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=lambda: mono, sleep=time.sleep))
    monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
    monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, resume: sleepers.append(resume))
    call = _tool(url_pair, task_id="url-rearm-429")
    first = call()
    lid = first["letter_id"]
    assert first["retry_after"] == 60 and not sleepers
    state_before_wake = delivery.trigger_link_flush

    def interrupt(r, l):
        nonlocal state
        state = "unknown"  # the tool already reactivated the parked letter
        state_before_wake(r, l)

    with monkeypatch.context() as unreadable:
        unreadable.setattr(delivery, "trigger_link_flush", interrupt)
        assert call(retry_of=lid)["reason"] == "retry_not_before"
    assert delivery.queued_message(root, link, lid)["pending"] and len(sleepers) == 1
    state = "off"
    with monkeypatch.context() as observer:
        observer.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=lambda: mono,
                                                          sleep=lambda _sec: delivery._delivery_pass(root, link)))
        retry = call(retry_of=lid)
    assert retry["reason"] == "retry_not_before" and retry["not_before"] == wall + 60
    assert posts == [lid]
    assert len(sleepers) == 1, "the second retry did not rearm the server's deadline wake"
    wall, mono = wall + 60, mono + 60
    sleepers.pop()()  # run the server's own deadline continuation; no external wake
    assert posts == [lid, lid] and not sleepers
    assert delivery.queued_message(root, link, lid) is None


def test_switch_hold_keeps_a_channel_attempt_uncertain(tmp_path, monkeypatch):
    from tests.test_peer_channel import live_channel_pair

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as pair:
        root, link = pair.root_a, pair.link_id
        read_switch = place_projection.switch_state
        send, posts = PeerGateway.send_message, []

        def wire(self, text, **kw):
            posts.append(kw["letter_id"])
            return send(self, text, **kw)

        monkeypatch.setattr(PeerGateway, "send_message", wire)
        monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
        pair.fwd_a_to_b.set_mode("accept_close")
        call = _tool(pair, task_id="channel-hold-uncertain")
        first = call()
        lid = first["letter_id"]
        assert first["reason"] == "channel_unconfirmed" and posts == [lid], first
        monkeypatch.setattr(place_projection, "switch_state",
                            lambda r: "panic" if r.resolve() == root.resolve() else read_switch(r))
        again = call(retry_of=lid)
        assert (again["status"], again["reason"], again["letter_id"]) == ("queued", "panic_hold", lid), again
        assert "An earlier attempt may have arrived" in again["note"], again
        assert posts == [lid] and delivery.queued_message(root, link, lid)["last_outcome"]["state"] == "uncertain"


def test_switch_hold_reconciles_delivery_after_failed_local_receipt(url_pair, monkeypatch):
    root, link = url_pair.root_a, url_pair.link_id
    sender_switch, posts, receipts = "off", [], []
    read_switch, write = place_projection.switch_state, peer_replies._record_receipt

    def wire(_self, _text, **kw):
        posts.append(kw["letter_id"])

    def receipt(*args, **kw):
        nonlocal sender_switch
        receipts.append(args[4])
        if len(receipts) == 1:
            sender_switch = "panic"  # after wire confirmation, before the pass woken to reconcile
            return False
        return write(*args, **kw)

    monkeypatch.setattr(place_projection, "switch_state", lambda r: sender_switch if r == root else read_switch(r))
    monkeypatch.setattr(PeerGateway, "send_message", wire)
    monkeypatch.setattr(peer_replies, "_record_receipt", receipt)
    monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
    monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, resume: resume())
    response = _tool(url_pair, task_id="switch-confirmed-recovery")()
    lid = response["letter_id"]
    assert response["status"] == "received", response
    assert posts == [lid] and receipts == ["delivered", "delivered"]
    assert delivery.queued_message(root, link, lid) is None


@pytest.mark.parametrize("source", ["cancelled_task", "recorded_notice"])
def test_switch_hold_does_not_mask_terminal_tool_letter(url_pair, monkeypatch, source):
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    root, link = url_pair.root_a, url_pair.link_id
    task_id, words = "terminal-during-switch-hold", "No further attempt"
    lid = peer_replies.message_letter_id(link, task_id, "", words)
    read_switch = place_projection.switch_state
    posts = []
    monkeypatch.setattr(PeerGateway, "send_message", lambda *a, **kw: posts.append(kw))
    monkeypatch.setattr(place_projection, "switch_state", lambda r: "unknown" if r == root else read_switch(r))
    monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
    with acting_as(root):
        account = peer_links.read_registry(root)["links"][link]["peer_install_id"]
        origin = {"kind": "tool", "task_id": task_id}
        delivery.record_queued_message(root, link, lid, place_id=url_pair.place_a, account_id=account,
                                       text=words, reply_to="", origin=origin, approved_basis="basis",
                                       coverage_kind="owner", wake=False)
        code = "task_stopped" if source == "cancelled_task" else "attempts_spent"
        if source == "cancelled_task":
            write_task_result(root, task_id, STATUS_CANCELLED, chat_id=1)
        else:
            assert delivery._hold_queued_message(root, place_id=url_pair.place_a, link_id=link,
                account_id=account, letter_id=lid, text=words, reply_to="", origin=origin,
                outcome={"state": "held", "code": code, "message": "Existing terminal hold"},
                identity=f"{lid}:held", earlier_uncertain=False, states=set())
        response = _tool(url_pair, task_id=task_id)(words, retry_of=lid)
        assert response["status"] == "refused" and response["reason"] == code, response
        assert delivery.queued_message(root, link, lid) is None and posts == []


def test_retry_observes_a_fresh_pass_in_the_same_clock_tick(url_pair, monkeypatch):
    elapsed, pass_count = 0.0, 0
    posted = []

    def advance(delay):
        nonlocal elapsed
        elapsed += 90.0  # advance only if the observer actually waits after the targeted pass

    def fail_before_wire(*_a, **_kw):
        nonlocal pass_count
        pass_count += 1
        raise peer_links.PeerLinkError("peer_links_unreadable", f"hold from pass {pass_count}", 503)

    monkeypatch.setattr(delivery, "time", SimpleNamespace(time=time.time, monotonic=lambda: elapsed, sleep=advance))
    monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
    monkeypatch.setattr(peer_links, "record_sending", fail_before_wire)
    monkeypatch.setattr(PeerGateway, "send_message", lambda *a, **kw: posted.append(kw))
    call = _tool(url_pair, task_id="retry-same-clock-tick")
    first = call()
    assert first["reason"] == "sending_unrecorded" and "hold from pass 1" in first["note"], first
    second = call(retry_of=first["letter_id"])
    assert second["reason"] == "sending_unrecorded" and "hold from pass 2" in second["note"], second
    assert (elapsed, pass_count, posted) == (0.0, 2, [])


def test_worker_url_refusal_preserves_receiver_code_and_detail(url_pair, monkeypatch, worker_and_server):
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place

    pair = url_pair
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
    worker_and_server(pair)
    result = _tool(pair)()
    assert result["status"] == "refused" and result["refusal_code"] == "link_paused", result
    assert "paused" in result["detail"].lower(), result


def test_worker_reads_terminal_rules_hold_after_intent_is_removed(url_pair, monkeypatch, worker_and_server):
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = url_pair

    def change_rules():
        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "add_note", author=AUTHOR_OWNER, text="New delivery rules.")

    worker_and_server(pair, before_pass=change_rules)
    result = _tool(pair)()
    assert result["status"] == "refused" and result["reason"] == "rules_changed", result


def test_url_repeat_without_retry_of_after_429_uses_same_id(url_pair, monkeypatch):
    pair = url_pair
    clock = [time.time()]
    monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: clock[0], monotonic=time.monotonic,
                                                         sleep=time.sleep))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.0,))
    tried, send = [], PeerGateway.send_message

    def rate_limit(self, text, **kw):
        tried.append(kw["letter_id"])
        if len(tried) == 1:
            e = _unavailable("presence_rate_limited", "receiver is busy", 429, refusal="presence_rate_limited")
            e.retry_after = 60
            raise e
        return send(self, text, **kw)

    monkeypatch.setattr(PeerGateway, "send_message", rate_limit)
    call = _tool(pair)
    result = call()
    assert result.get("retry_after") == 60 and result["reason"] == "presence_rate_limited", result
    assert len(tried) == 1
    lid = result["letter_id"]
    clock[0] += 61
    delivery._UNPERSISTED_RETRY.pop((str(pair.root_a.resolve()), pair.link_id, lid), None)
    again = call()
    assert again["status"] == "received" and tried == [lid, lid], (again, tried)


def test_url_parked_record_leaves_active_enumeration_without_body_reads(url_pair, monkeypatch):
    pair = url_pair
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0)

    def unavailable(*a, **kw):
        raise _unavailable("peer_unreachable", "receiver is absent")

    monkeypatch.setattr(PeerGateway, "send_message", unavailable)
    result = _tool(pair)()
    root, link, lid = pair.root_a, pair.link_id, result["letter_id"]
    with acting_as(root):
        assert delivery.queued_message(root, link, lid)["pending"] is False
        assert not delivery._has_pending_files_for_link(root, link)
        assert delivery.queued_messages(root, link) == []
        assert peer_replies.resolve_message_retry(root, link, peer_links.read_registry(root)["links"][link][
            "peer_install_id"], lid)[0]["origin"]["task_id"] == "message-origin"


def test_new_approval_replaces_old_basis_and_can_clear_proven_unsent_rules_hold(url_pair, monkeypatch):
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = url_pair
    root, link = pair.root_a, pair.link_id
    text, tid = "An unchanged note", "fresh-approval"
    call = _tool(pair, task_id=tid)
    with acting_as(root):
        def seed(words):
            lid = peer_replies.message_letter_id(link, tid, "", words)
            account = peer_links.read_registry(root)["links"][link]["peer_install_id"]
            return lid, delivery.record_queued_message(root, link, lid, place_id=pair.place_a, account_id=account,
                text=words, reply_to="", origin={"kind": "tool", "task_id": tid},
                approved_basis=peer_letters.speech_basis(root, pair.place_a, kind="owner"),
                coverage_kind="owner", wake=False)

        lid, seeded = seed(text)
        edit_place(root, pair.place_a, "add_note", author=AUTHOR_OWNER, text="New guidance for messages.")
        # Unchanged bytes receive a new approval even though the old queued time/deadline still apply.
        basis_now = peer_letters.speech_basis(root, pair.place_a, kind="owner")
        updated = delivery.record_queued_message(root, link, lid, place_id=pair.place_a,
            account_id=seeded["account_id"], text=text, reply_to="", origin=seeded["origin"],
            approved_basis=basis_now, coverage_kind="owner", wake=False)
        held_lid, held = seed("Different note")
        edit_place(root, pair.place_a, "add_note", author=AUTHOR_OWNER, text="Additional guidance.")
        outcome, _ = delivery._flush_one_queued(root, link, held)
        assert outcome["code"] == "rules_changed"
        assert delivery.queued_message(root, link, held_lid) is None
        fresh = call("Different note")
        assert fresh["status"] == "received" and fresh["letter_id"] == held_lid, fresh
        assert updated["approved_basis"] == basis_now != seeded["approved_basis"]
        assert updated["queued_at"] == seeded["queued_at"]


def test_tool_origin_survives_automatic_adoption_and_creator_retry(url_pair, monkeypatch):
    pair = url_pair
    root, producer, words = pair.root_a, "producer-adoption", "One response only"
    send, posts = PeerGateway.send_message, []
    ready = False
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0)

    def wire(self, text, **kw):
        posts.append(kw["letter_id"])
        if not ready:
            raise _unavailable("peer_unreachable", "temporary receiver outage")
        return send(self, text, **kw)

    monkeypatch.setattr(PeerGateway, "send_message", wire)
    call = _tool(pair, task_id=producer)
    first = call(words)
    assert first["reason"] == "unconfirmed", first
    tool_lid = first["letter_id"]
    with acting_as(root):
        from ouroboros.places import get_place
        from ouroboros.presence_bindings import conversation_key
        place = get_place(root, pair.place_a)
        key = conversation_key(place.provider, place.account_id, place.conversation_id, "")
        monkeypatch.setattr(peer_links, "owed_record", lambda *a: {"approved": {
            hashlib.sha256(words.encode()).hexdigest(): peer_letters.speech_basis(root, pair.place_a, kind="owner")}})
        ready = True
        assert peer_replies._deliver_output(root, place, key, producer=producer, output_ref="terminal",
                text=words, reply_to="", source_event_id="event", kind="owner") == (True, None)
    again = call(words, retry_of=tool_lid)
    assert again["status"] == "received" and posts == [tool_lid] * 4, (again, posts)


@pytest.mark.parametrize("retire_at", ["before", "between", "after"])
def test_loopback_automatic_adoption_keeps_429_deadline_after_parked_cleanup(url_pair, monkeypatch, retire_at):
    from ouroboros.places import get_place
    from ouroboros.presence_bindings import conversation_key

    pair = url_pair
    root, link, producer, words = pair.root_a, pair.link_id, "producer-rate-limit", "One answer"
    wall = [time.time()]
    monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: wall[0], monotonic=time.monotonic,
                                                         sleep=time.sleep))
    send, posts = PeerGateway.send_message, []

    def wire(self, text, **kw):
        posts.append(kw["letter_id"])
        if len(posts) == 1:
            err = _unavailable("presence_rate_limited", "wait", 429, refusal="presence_rate_limited")
            err.retry_after = 60.0
            raise err
        return send(self, text, **kw)

    monkeypatch.setattr(PeerGateway, "send_message", wire)
    call = _tool(pair, task_id=producer)
    first = call(words)
    assert first["reason"] == "presence_rate_limited" and first["retry_after"] == 60.0, first
    lid = first["letter_id"]
    with acting_as(root):
        assert peer_links.sending_record(root, link, lid)["not_before"] == wall[0] + 60.0
        place = get_place(root, pair.place_a)
        key = conversation_key(place.provider, place.account_id, place.conversation_id, "")
        basis = peer_letters.speech_basis(root, pair.place_a, kind="owner")
        peer_letters.note_reply_approved(root, place, producer, words, basis, basis)
        def retire():
            peer_replies.deliver_owed(root, producer, {"status": "completed"})
            assert delivery.queued_message(root, link, lid) is None

        retired = False
        read_sending = peer_links.sending_record

        def read_then_retire(r, l, letter_id):
            nonlocal retired
            snapshot = read_sending(r, l, letter_id)
            if not retired and (r, l, letter_id) == (root, link, lid):
                assert snapshot["not_before"] == wall[0] + 60.0
                retired = True
                retire()  # producer completes after this read, before the reply reads the queue
            return snapshot

        def send_output():
            return peer_replies._deliver_output(root, place, key, producer=producer, output_ref="terminal",
                                                text=words, reply_to="", source_event_id="event", kind="owner")

        if retire_at == "before":
            retire()
        with monkeypatch.context() as interleave:
            if retire_at == "between":
                interleave.setattr(peer_links, "sending_record", read_then_retire)
            outcome = send_output()
        if retire_at == "after":
            retire()
        if retire_at == "between":
            assert retired
        assert (outcome, posts) == ((False, 60.0), [lid])
        assert peer_links.sending_record(root, link, lid)["not_before"] == wall[0] + 60.0
        wall[0] += 60.0
        assert send_output() == (True, None)
    assert call(words, retry_of=lid)["status"] == "received" and posts == [lid, lid]


def _rate_limited(wait=60.0):
    exc = _unavailable("presence_rate_limited", "wait", 429, refusal="presence_rate_limited")
    exc.retry_after = wait
    return exc


def test_retry_wait_keeps_server_deadline_and_uncertainty(tmp_path, monkeypatch):
    from tests.test_peer_channel import live_channel_pair

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as pair:
        root, link = pair.root_a, pair.link_id
        wall, posts, sleepers, switch = time.time(), [], [], "off"
        real_send, read_switch = PeerGateway.send_message, place_projection.switch_state

        def wire(self, text, **kw):
            posts.append(kw["letter_id"])
            if len(posts) == 2:
                raise _rate_limited()
            return real_send(self, text, **kw)

        monkeypatch.setattr(PeerGateway, "send_message", wire)
        monkeypatch.setattr(place_projection, "switch_state", lambda r: switch if r == root else read_switch(r))
        monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=time.monotonic,
                                                             sleep=time.sleep))
        monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
        monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, resume: sleepers.append(resume))
        pair.fwd_a_to_b.set_mode("accept_close")
        call = _tool(pair, task_id="uncertain-with-deadline")
        first = call()
        lid = first["letter_id"]
        assert first["reason"] == "channel_unconfirmed" and posts == [lid], first
        pair.fwd_a_to_b.set_mode("forward")
        try:
            delivery.flush_link_sync(root, link)
            assert len(sleepers) == 1 and posts == [lid, lid]
            switch = "panic"
            with monkeypatch.context() as observer:
                observer.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=time.monotonic,
                    sleep=lambda _sec: delivery._delivery_pass(root, link)))
                result = call(retry_of=lid)
            assert (result["status"], result["letter_id"], result["not_before"]) == ("queued", lid, wall + 60)
            assert "An earlier attempt may have arrived" in result["note"], result
            assert posts == [lid, lid]
        finally:
            delivery.retire_link_waits(root, link)
            delivery._FLUSHING_LINKS.pop((str(root.resolve()), link), None)


@pytest.mark.parametrize("terminal", ["receipt", "repeat", "task_stopped", "rules_changed", "held"])
def test_retry_reads_indexed_result_when_settlement_wins_between_reads(url_pair, monkeypatch, terminal):
    from ouroboros import presence_delivery

    root, link = url_pair.root_a, url_pair.link_id
    task_id, words = "observe-concurrent-settlement", "Note to settle"
    lid = peer_replies.message_letter_id(link, task_id, "", words)
    account = peer_links.read_registry(root)["links"][link]["peer_install_id"]
    origin = {"kind": "tool", "task_id": task_id}
    read_states, read_queue, read_switch = presence_delivery.recorded_states, delivery.queued_message, place_projection.switch_state
    fired, posts = False, []
    monkeypatch.setattr(PeerGateway, "send_message", lambda *a, **kw: posts.append(kw))
    monkeypatch.setattr(place_projection, "switch_state", lambda r: "panic" if r == root else read_switch(r))
    with acting_as(root):
        delivery.record_queued_message(root, link, lid, place_id=url_pair.place_a, account_id=account,
                                       text=words, reply_to="", origin=origin, approved_basis="basis",
                                       coverage_kind="owner", wake=False)
        if terminal in {"receipt", "repeat"}:
            peer_links.confirm_sending(root, link, lid, state="delivered", account_id=account,
                                       place_id=url_pair.place_a, text=words, origin=origin)
        else:
            approval = read_queue(root, link, lid)
            identity = f"{lid}:{terminal}"
            if terminal == "rules_changed":
                identity += f":{approval.get('approval_at') or approval['queued_at']}"
            code = "attempts_spent" if terminal == "held" else terminal
            assert delivery._hold_queued_message(root, place_id=url_pair.place_a, link_id=link, account_id=account,
                letter_id=lid, text=words, reply_to="", origin=origin,
                outcome={"state": "held", "code": code, "message": "Recorded terminal hold"},
                identity=identity, earlier_uncertain=False, states=set())

        def finish_pass():
            nonlocal fired
            if not fired:
                fired = True
                delivery.flush_link_sync(root, link)
                assert read_queue(root, link, lid) is None

        def interleaved_states(r, provider, to, letter, part):
            snapshot = read_states(r, provider, to, letter, part)
            if (r, to, letter) == (root, account, lid):
                finish_pass()
            return snapshot

        def interleaved_queue(r, to, letter):
            snapshot = read_queue(r, to, letter)
            if (r, to, letter) == (root, link, lid):
                finish_pass()
            return snapshot

        if terminal in {"receipt", "repeat"}:
            monkeypatch.setattr(presence_delivery, "recorded_states", interleaved_states)
        else:
            monkeypatch.setattr(delivery, "queued_message", interleaved_queue)
        result = _tool(url_pair, task_id=task_id)(words, **({} if terminal == "repeat" else {"retry_of": lid}))
    assert fired and posts == []
    if terminal in {"receipt", "repeat"}:
        assert (result["status"], result["letter_id"]) == ("received", lid), result
    else:
        assert (result["status"], result["reason"]) == ("refused", code), result


def test_early_wake_does_not_retire_url_deadline_when_switch_is_unreadable(url_pair, monkeypatch):
    root, link = url_pair.root_a, url_pair.link_id
    wall, mono, switch = time.time(), 100.0, "off"
    posts, waits, sleepers = [], [], []
    read_switch = place_projection.switch_state

    def wire(_self, _text, **kw):
        posts.append((kw["letter_id"], mono))
        if len(posts) == 1:
            raise _rate_limited()

    def simulate_wait(event, at):
        nonlocal wall, mono, switch
        waits.append(at)
        if len(waits) == 1:
            switch = "unknown"
            # Publish the early wake as a worker would; don't observe the result on the same thread
            # that must run the server's next pass.
            delivery.flush_link_sync(root, link)
            assert event.is_set()
        else:
            assert len(waits) <= 4, waits
            wall, mono = wall + (at - mono), at
            switch = "off"  # recovers without an external wake

    monkeypatch.setattr(PeerGateway, "send_message", wire)
    monkeypatch.setattr(place_projection, "switch_state", lambda r: switch if r == root else read_switch(r))
    monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=lambda: mono, sleep=time.sleep))
    monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
    monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, resume: sleepers.append(resume))
    monkeypatch.setattr(delivery, "_wait_for_deadline", simulate_wait)
    call = _tool(url_pair, task_id="switch-recovery-before-url-deadline")
    lid = call()["letter_id"]
    assert call(retry_of=lid)["not_before"] == wall + 60
    sleepers.pop()()
    assert posts == [(lid, 100.0), (lid, 160.0)], (posts, waits)
    assert delivery.queued_message(root, link, lid) is None and not sleepers


def test_tree_stop_does_not_erase_deadline_for_matching_owed_reply(tmp_path, monkeypatch):
    from ouroboros.places import get_place
    from ouroboros.presence_bindings import conversation_key
    from tests.test_peer_channel import live_channel_pair

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as pair:
        root, link, producer, words = pair.root_a, pair.link_id, "stop-after-approved-reply", "The approved reply"
        wall, posts, sleepers = time.time(), [], []
        stopped = False

        def wire(_self, _text, **kw):
            posts.append(kw["letter_id"])
            if len(posts) == 1:
                raise _rate_limited()

        monkeypatch.setattr(PeerGateway, "send_message", wire)
        monkeypatch.setattr(delivery, "time", SimpleNamespace(time=lambda: wall, monotonic=time.monotonic,
                                                             sleep=time.sleep))
        monkeypatch.setattr(delivery, "trigger_link_flush", delivery.flush_link_sync)
        monkeypatch.setattr(delivery, "_start_flush_thread", lambda _key, _state, resume: sleepers.append(resume))
        monkeypatch.setattr(delivery, "_check_task_stop", lambda *a: ("task_stopped", "tree stopped") if stopped else None)
        call = _tool(pair, task_id=producer)
        lid = call(words)["letter_id"]
        assert posts == [lid] and len(sleepers) == 1
        try:
            with acting_as(root):
                place = get_place(root, pair.place_a)
                key = conversation_key(place.provider, place.account_id, place.conversation_id, "")
                basis = peer_letters.speech_basis(root, pair.place_a, kind="owner")
                peer_letters.note_reply_approved(root, place, producer, words, basis, basis)
                stopped = True
                row = delivery.queued_message(root, link, lid)
                delivery._flush_one_queued(root, link, row, switch_hold=("switch_unreadable", "unreadable"))
                assert delivery.queued_message(root, link, lid) is None

                def output():
                    return peer_replies._deliver_output(root, place, key, producer=producer, output_ref="terminal",
                                                        text=words, reply_to="", source_event_id="event", kind="owner")

                assert (output(), posts) == ((False, 60.0), [lid])
                wall += 60.0
                assert (output(), posts) == ((True, None), [lid, lid])
        finally:
            delivery.retire_link_waits(root, link)
            delivery._FLUSHING_LINKS.pop((str(root.resolve()), link), None)
