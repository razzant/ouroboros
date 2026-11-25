"""A reply a restart interrupted: the built-in connector's owed records (``peer_links.record_owed``) and their resume.

Each test stands in for a restart by a new call of the start's own handler (``peer_replies.resume_owed``) over the
state an earlier "process" left: in-process, not a process restart. Two real letter doors (``_both_sides``) and the
carrier stand-in of ``tests/test_peer_messages.py`` record each turn as its runner would; no model is called.
"""
from __future__ import annotations

from tests._peer_letters_shared import introduce
from tests.test_peer_messages import (_approve, _arrival, _both_sides, _enable_consciousness, _msg, _poll_until,
                                      _receipts, _replies_idle, _room, _settle_turns)


def _owed(root) -> list:
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    with acting_as(root):
        return peer_links.owed_refs(root)


def _unreported_turn(pair, monkeypatch, client_b, text):
    """A turn whose end landed in its record but was never reported (a crash between the two): owed, unsent."""
    import ouroboros.peer_replies as replies

    real = replies.deliver_owed
    monkeypatch.setattr(replies, "deliver_owed", lambda *a, **k: None)
    assert client_b.post("/api/letters", json=_msg(letter_id=f"m-{text}", text=text),
                         headers=pair.headers()).status_code == 200
    monkeypatch.setattr(replies, "deliver_owed", real)
    return _arrival(pair.root_b, text)["task_id"]


def test_a_reply_a_restart_interrupted_goes_once_from_its_record_and_never_again(tmp_path, monkeypatch):
    """Owed from its turn's start, unsent at its end: the next start's resume sends the recorded words under their
    stable letter id, once; its record then leaves, and another resume sends nothing (no second turn, no history scan:
    a turn that answered normally leaves no record behind)."""
    import ouroboros.peer_replies as replies

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched: list = []
    _settle_turns(pair, monkeypatch, {"Answered.": ("message", "Reply answered.", ""),
                                      "Interrupted.": ("message", "Reply interrupted.", "")}, dispatched)

    def texts_a(client_a):
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        assert client_b.post("/api/letters", json=_msg(letter_id="m-1", text="Answered."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: "Reply answered." in texts_a(client_a), label="A received the answered reply")
        _poll_until(_replies_idle, label="reply threads ended")
        assert _owed(pair.root_b) == [], "an answered turn leaves no owed record"
        turn = _unreported_turn(pair, monkeypatch, client_b, "Interrupted.")
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and "Reply interrupted." not in texts_a(client_a)
        replies.resume_owed(pair.root_b)  # the next start's handler over the saved state (not a process restart)
        _poll_until(lambda: "Reply interrupted." in texts_a(client_a), label="A received the interrupted reply")
        _poll_until(_replies_idle, label="reply threads ended")
        assert _owed(pair.root_b) == [], "delivered: its record left"
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert [texts_a(client_a).count(t) for t in ("Reply answered.", "Reply interrupted.")] == [1, 1]
    assert [text for _root, text in dispatched] == ["Answered.", "Interrupted."], "one model turn per message"
    assert sorted(r["transport"]["delivery"]["state"] for r in _receipts(pair.root_b)) == ["delivered"] * 2


def test_a_handed_off_works_reply_a_restart_interrupted_goes_once_from_its_record(tmp_path, monkeypatch):
    """A deferred exit: the turn hands its answer to a work, which owes it from then on (the turn's own record
    leaves); the work ended but its end was never reported, so the next start's resume sends its terminal words
    once, under the same link, and nothing after: no second model turn, no rerun of the work."""
    import ouroboros.peer_replies as replies
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched: list = []
    _settle_turns(pair, monkeypatch, {"Please rerun eval.": ("deferred", "", "handoff-eval-1")}, dispatched)
    result = "Eval rerun: 0.91."
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        assert client_b.post("/api/letters", json=_msg(letter_id="ask-1", text="Please rerun eval."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: _owed(pair.root_b) == [(pair.link_id, "handoff-eval-1")], label="the work owes the reply")
        _poll_until(_replies_idle, label="reply threads ended")
        with acting_as(pair.root_b):  # the work's end, written but never reported (a crash between the two)
            producer = load_task_result(pair.root_b, _arrival(pair.root_b, "Please rerun eval.")["task_id"])
            _approve(pair, "handoff-eval-1", result, {"presence": producer["metadata"]["presence"]})
            write_task_result(pair.root_b, "handoff-eval-1", STATUS_COMPLETED, result=result,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": producer["metadata"]["presence"]})
        assert result not in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        replies.resume_owed(pair.root_b)  # the next start's handler over the saved state (not a process restart)
        _poll_until(lambda: result in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)],
                    label="A received the handed-off result")
        _poll_until(_replies_idle, label="reply threads ended")
        assert _owed(pair.root_b) == []
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)].count(result) == 1
    assert [text for _root, text in dispatched] == ["Please rerun eval."], "one model turn, the work never rerun"
    assert [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)] == [(result, "delivered")]


def test_a_panic_awaiting_my_humans_return_keeps_the_owed_reply_unsent(tmp_path, monkeypatch):
    """A start under a Panic sends nothing and keeps the record (mock of the switch's one reading); the first start
    after my human's return sends it."""
    import ouroboros.peer_replies as replies
    from ouroboros import place_projection

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Held.": ("message", "Reply held by Panic.", "")}, [])
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Held.")
        real_switch = place_projection.switch_state
        monkeypatch.setattr(place_projection, "switch_state", lambda root: "panic")
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="no reply thread")
        assert _owed(pair.root_b) == [(pair.link_id, turn)]
        assert "Reply held by Panic." not in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        monkeypatch.setattr(place_projection, "switch_state", real_switch)
        replies.resume_owed(pair.root_b)
        _poll_until(lambda: "Reply held by Panic." in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)],
                    label="A received it after the return")
        _poll_until(_replies_idle, label="reply threads ended")
    assert _owed(pair.root_b) == []


def test_nothing_is_promised_to_a_revoked_or_unknown_link_and_a_heard_message_owes_nothing(tmp_path, monkeypatch):
    """A revoked link takes its owed records with it, so no start sends there; a registry that cannot be read keeps the
    records unread for a later start; a message heard with consciousness Off started no turn and owes nothing."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    with _both_sides(pair, monkeypatch) as (_client_a, client_b):
        assert client_b.post("/api/letters", json=_msg(letter_id="m-off", text="Heard only."),
                             headers=pair.headers()).status_code == 200
    assert _owed(pair.root_b) == [], "heard, no turn: nothing owed"
    with acting_as(pair.root_b):
        peer_links.record_owed(pair.root_b, pair.link_id, "turn-unsent", place_id=pair.place_b)
    real_registry = peer_links.read_registry
    monkeypatch.setattr(peer_links, "read_registry", lambda root: (_ for _ in ()).throw(
        peer_links.PeerLinkError("peer_links_unreadable", "needs repair", 503)))
    sent: list = []
    monkeypatch.setattr(replies, "_resume_quietly", lambda root, owed: sent.append(owed))
    replies.resume_owed(pair.root_b)
    monkeypatch.setattr(peer_links, "read_registry", real_registry)
    assert sent == [] and _owed(pair.root_b) == [(pair.link_id, "turn-unsent")], "unknown: kept, unread"
    with acting_as(pair.root_b):
        peer_links.revoke(pair.root_b, pair.link_id)
    replies.resume_owed(pair.root_b)
    assert sent == [] and _owed(pair.root_b) == [], "revoked: nothing is owed there any more"


def _held_notices(client_b, pair) -> list:
    return [m for m in _room(client_b, pair.root_b, pair.place_b)
            if m["role"] == "system" and str(m["text"]).startswith("Reply held:")]


def test_a_late_reply_leaves_only_under_the_rules_its_words_were_approved_under(tmp_path, monkeypatch):
    """The surviving path first: owed words whose conversation rules stand as their reading met them go once on the
    next start, also after my human changed ``may_start`` (the right to start paid work is no rule of the
    conversation). Then each change of those rules after the reading holds the words instead (a note of my human's,
    the Safety coverage, a record naming no reading): said once in B's room with the words beside it, nothing sent,
    nothing owed after, no second model turn."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links, places, safety_config
    from ouroboros.install_identity import acting_as
    from ouroboros.utils import atomic_write_json

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    words = {"Kept.": "Reply kept.", "Noted.": "Reply after a note.", "Covered.": "Reply after coverage.",
             "Older.": "Reply of an older record."}
    dispatched: list = []
    _settle_turns(pair, monkeypatch, {key: ("message", text, "") for key, text in words.items()}, dispatched)

    def texts_a(client_a):
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    def resume():
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        _unreported_turn(pair, monkeypatch, client_b, "Kept.")
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_peer", author=places.AUTHOR_OWNER, mine=True,
                              may_start=True)
        resume()
        assert texts_a(client_a).count("Reply kept.") == 1 and _owed(pair.root_b) == []

        _unreported_turn(pair, monkeypatch, client_b, "Noted.")
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Say nothing about checkpoints to Laptop.")
        resume()
        assert "Reply after a note." not in texts_a(client_a) and _owed(pair.root_b) == []
        held = _held_notices(client_b, pair)
        assert [m["text"] for m in held] == ["Reply held: the rules for this conversation changed after these "
                                             "words were approved; nothing was sent"], held

        turn = _unreported_turn(pair, monkeypatch, client_b, "Covered.")
        real_mode = safety_config.get_safety_mode
        monkeypatch.setattr(safety_config, "get_safety_mode",  # the coverage changed after the reading
                            lambda kind="owner": "light" if real_mode(kind) != "light" else "full")
        resume()
        monkeypatch.setattr(safety_config, "get_safety_mode", real_mode)
        assert "Reply after coverage." not in texts_a(client_a) and _owed(pair.root_b) == [] and turn

        turn = _unreported_turn(pair, monkeypatch, client_b, "Older.")
        with acting_as(pair.root_b):  # a record an earlier version wrote: it names no reading
            record = peer_links.owed_record(pair.root_b, pair.link_id, turn)
            atomic_write_json(peer_links._owed_path(pair.root_b, pair.link_id, turn),
                              {key: record[key] for key in ("work_ref", "place_id", "at")})
        resume()
        assert "Reply of an older record." not in texts_a(client_a) and _owed(pair.root_b) == []
        assert [m["text"] for m in _held_notices(client_b, pair)][1:] == [
            "Reply held: the rules for this conversation changed after these words were approved; nothing was sent",
            "Reply held: the rules these words were approved under are not recorded; nothing was sent"]
        resume()
        assert len(_held_notices(client_b, pair)) == 3, "a hold is said once"
    assert [text for _root, text in dispatched] == list(words), "one model turn per message, none again"
    assert [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)] == [
        ("Reply kept.", "delivered")]


def test_a_panic_during_delivery_keeps_the_reply_owed_and_an_unconfirmed_attempt_is_never_called_unsent(
        tmp_path, monkeypatch):
    """A Panic that begins after a start's check still stops the send: read before every attempt, it keeps the reply
    owed and unsent. A Panic between two attempts ends the retries with the first one's outcome, unconfirmed (A saved
    it; B never heard). When the rules then change, the held retry says the earlier attempt may have arrived."""
    import ouroboros.peer_replies as replies
    from ouroboros import place_projection, places
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", ())  # one pass per start here: its follow-ups are their own test
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Wait.": ("message", "Reply after the Panic.", ""),
                                      "Lost.": ("message", "Reply whose answer was lost.", "")}, [])
    real_switch, panic = place_projection.switch_state, {"reads": 0, "after": None}

    def switch(root):  # the switch's one reading, turned to a Panic from a given reading on
        panic["reads"] += 1
        after = panic["after"]
        return "panic" if after is not None and panic["reads"] > after else real_switch(root)

    monkeypatch.setattr(place_projection, "switch_state", switch)
    real_request = peer_gw.PeerGateway._request

    def lose_answer_then_panic(self, method, path, **kw):
        answer = real_request(self, method, path, **kw)  # A saved it ...
        if self._link.place_id == pair.place_b and panic.get("lose"):
            panic.update(lose=False, after=panic["reads"])  # ... and a Panic begins before the next attempt
            raise peer_gw._unavailable("peer_unreachable", "the answer was lost on the way back")  # B never heard
        return answer

    def texts_a(client_a):
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Wait.")
        panic.update(reads=0, after=1)  # the start's own check passes; the send's first reading meets the Panic
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and "Reply after the Panic." not in texts_a(client_a)
        assert _held_notices(client_b, pair) == [], "a Panic's hold is said nowhere: the reply stays owed"
        panic["after"] = None
        replies.resume_owed(pair.root_b)
        _poll_until(lambda: "Reply after the Panic." in texts_a(client_a), label="A received it after the return")
        _poll_until(_replies_idle, label="reply threads ended")

        monkeypatch.setattr(peer_gw.PeerGateway, "_request", lose_answer_then_panic)
        panic["lose"] = True
        lost = _unreported_turn(pair, monkeypatch, client_b, "Lost.")
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert texts_a(client_a).count("Reply whose answer was lost.") == 1 and _owed(pair.root_b) == [
            (pair.link_id, lost)]
        monkeypatch.setattr(peer_gw.PeerGateway, "_request", real_request)
        panic["after"] = None
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Ask me before replying to Laptop.")
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert texts_a(client_a).count("Reply whose answer was lost.") == 1 and _owed(pair.root_b) == []
        assert [m["text"] for m in _held_notices(client_b, pair)] == [
            "Reply held: the rules for this conversation changed after these words were approved; nothing was sent "
            "now, and an earlier attempt of these words is unconfirmed: it may have arrived"]
    states = sorted((r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b))
    assert states == [("Reply after the Panic.", "delivered"), ("Reply whose answer was lost.", "uncertain")]



def _unconfirmed_then(pair, monkeypatch, second, *, after_first=lambda: None):
    """``PeerGateway._request`` for B's sends: the first attempt reaches A whose answer is lost on the way back (A
    keeps the message; B never hears; ``after_first`` runs then), and ``second(self)`` answers any next attempt."""
    from ouroboros.gateways import peer as peer_gw

    real_request, calls = peer_gw.PeerGateway._request, {"n": 0}

    def request(self, method, path, **kw):
        if self._link.place_id != pair.place_b:
            return real_request(self, method, path, **kw)
        calls["n"] += 1
        if calls["n"] == 1:
            real_request(self, method, path, **kw)
            after_first()
            raise peer_gw._unavailable("peer_unreachable", "the answer was lost on the way back")
        return second(self)

    monkeypatch.setattr(peer_gw.PeerGateway, "_request", request)
    return calls


def test_a_later_attempts_hold_or_refusal_keeps_the_earlier_unconfirmed_one_and_rechecks_each_attempt(
        tmp_path, monkeypatch):
    """Every physical attempt of an automatic reply meets the link, its place and the approved rules as they stand
    then: a note my human adds while the first attempt waits stops the next one. Whatever a later attempt meets (that
    change, the receiver's typed nothing-sent code, its 429 or its definite refusal), it never turns the earlier
    unconfirmed attempt into "nothing was sent" or "failed": the receipt stays ``uncertain`` and the reply owed."""
    import ouroboros.peer_replies as replies
    from ouroboros import places
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as

    def raising(*args):
        def second(self):
            raise peer_gw._unavailable(*args)
        return second

    def never(self):
        raise AssertionError("an attempt left after the rules changed")

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", ())  # one pass per start here: its follow-ups are their own test
    cases = {"rules": never, "nothing_sent": raising("place_paused", "paused there"),
             "busy": raising("presence_capacity_full", "busy", 429, "presence_capacity_full"),
             "refused": raising("link_unknown", "not known there", 401, "link_unknown")}
    for case, second in cases.items():
        pair = introduce(tmp_path / case, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
        _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
        _settle_turns(pair, monkeypatch, {"Ping.": ("message", f"Reply {case}.", "")}, [])

        def add_note(pair=pair):
            with acting_as(pair.root_b):
                places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                                  text="Ask me first.")

        with _both_sides(pair, monkeypatch) as (client_a, client_b):
            turn = _unreported_turn(pair, monkeypatch, client_b, "Ping.")
            calls = _unconfirmed_then(pair, monkeypatch, second,
                                      after_first=add_note if case == "rules" else lambda: None)
            replies.resume_owed(pair.root_b)
            _poll_until(_replies_idle, label="reply threads ended")
            texts_a = [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
            assert texts_a.count(f"Reply {case}.") == 1, (case, texts_a)  # the first attempt arrived
            assert calls["n"] == (1 if case == "rules" else 2), (case, calls)
            assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == [], case
        states = [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)]
        assert states == [(f"Reply {case}.", "uncertain")], (case, states)


def test_the_start_resumes_owed_replies_only_once_its_recovery_reconciled_their_producers(tmp_path, monkeypatch):
    """The start's own handler boundary on both boot paths (``server_maintenance._run_startup_task_recovery``): a
    completed recovery pass hands the replies a restart left owed to their resume (``server_boot.resume_peer_replies``
    -> ``peer_replies.resume_owed``, the handler the tests above call); a deferred pass (a test's live data root,
    prior workers not confirmed ended) hands nothing on and leaves them owed. In-process, not a process restart."""
    from ouroboros import server_boot, server_maintenance, startup_migrations as migrations

    migrations.prepare_startup_state(tmp_path)
    resumed: list = []
    monkeypatch.setattr(server_boot, "resume_peer_replies", lambda root: resumed.append(root))
    server_maintenance._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=True)
    server_maintenance._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False, prior_worker_pids=None)
    assert resumed == [], "a deferred recovery hands nothing on"
    report = server_maintenance._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False,
                                                           prior_worker_pids=set())
    assert report.get("prior_workers_ended") is True and resumed == [tmp_path], (report, resumed)


class _Wire:
    """The receiver's answers to physical attempts, in order (None delivers, an exception is its typed answer), and the
    order of events around them; no network."""

    def __init__(self, journal: list, answers: list) -> None:
        self.journal, self.answers = journal, answers

    def __call__(self, link):
        return self

    def send_message(self, *args, **kwargs):
        self.journal.append("POST")
        if (answer := self.answers.pop(0)) is not None:
            raise answer

    def close(self):
        pass


def _abstract_sender(tmp_path, monkeypatch, *, states=(), record=None, link_usable=True):
    """``send_peer_message`` over abstract stand-ins: the registry's address of the link, its current link, the receipt
    store, the sending record and the wire. Returns ``(send, journal, receipts, held)``; no network, no store."""
    from types import SimpleNamespace

    import ouroboros.peer_replies as replies
    from ouroboros import peer_links, place_projection, presence_delivery
    from ouroboros.gateways import peer as peer_gw

    journal, receipts, answers, held = [], [], [], {"record": dict(record) if record else None}
    link = peer_links.OutgoingLink(link_id="link-0001", place_id="place-b", peer_url="http://127.0.0.1:9",
                                   peer_label="Laptop", peer_install_id="install-a", own_install_id="install-b",
                                   drive_root=str(tmp_path))
    monkeypatch.setattr(peer_gw, "PeerGateway", _Wire(journal, answers))
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda root, place_id: None)
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "on")
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda *args: set(states))
    monkeypatch.setattr(peer_links, "link_of_place", lambda root, place_id: ("link-0001", {"peer_install_id": "install-a"}))
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: link if link_usable else None)
    monkeypatch.setattr(peer_links, "sending_record", lambda root, link_id, letter_id: held["record"])

    def record_sending(root, link_id, letter_id, **fields):
        journal.append("record")
        held["record"] = {"letter_id": letter_id, **fields, "passes": int((held["record"] or {}).get("passes") or 0) + 1}
        return held["record"]["passes"]

    def confirm_sending(root, link_id, letter_id, *, state, **fields):
        journal.append(f"confirm:{state}")
        base = dict(held["record"] or {"letter_id": letter_id, "passes": 1})
        kept = "delivered" if base.get("confirmed_state") == "delivered" else state
        base.update({k: v for k, v in fields.items() if v}, confirmed_state=kept)
        held["record"] = base
        return base

    def rollback_sending(root, link_id, letter_id, *, expected_passes):
        journal.append("rollback")
        cur = held["record"]
        if not cur or cur.get("confirmed_state") or int(cur.get("passes") or 0) != int(expected_passes):
            return False
        if int(expected_passes) <= 1:
            held["record"] = None
        else:
            held["record"] = {**cur, "passes": int(expected_passes) - 1}
        return True

    def settle_sending(root, link_id, letter_id):
        journal.append("settle")
        held["record"] = None

    monkeypatch.setattr(peer_links, "record_sending", record_sending)
    monkeypatch.setattr(peer_links, "confirm_sending", confirm_sending)
    monkeypatch.setattr(peer_links, "rollback_sending", rollback_sending)
    monkeypatch.setattr(peer_links, "settle_sending", settle_sending)
    monkeypatch.setattr(replies, "_receipts", lambda root: SimpleNamespace(
        record=lambda connector, receipt: receipts.append(receipt["state"])))
    monkeypatch.setattr(replies, "_RETRY_PAUSE_SEC", 0.0)

    def send(*replies_in):
        answers[:] = list(replies_in)
        return replies.send_peer_message(tmp_path, "place-b", letter_id="msg-0001", text="Reply.",
                                         origin={"kind": "automatic", "task_id": "turn-1"})

    return send, journal, receipts, held


def test_a_later_call_never_turns_an_earlier_unconfirmed_delivery_into_failed(tmp_path, monkeypatch):
    """A call starts from what earlier, finished calls recorded for its delivery id (``send_peer_message``): after a
    recorded ``uncertain``, the receiver's definite refusal of a new attempt keeps the delivery ``uncertain`` and writes
    no receipt over it; only a confirmed delivery clears it. Control: with nothing unconfirmed before, the same refusal
    is that delivery's own ``failed``. Abstract stand-ins for the wire, the switch and the receipt store: no network."""
    from ouroboros.gateways import peer as peer_gw

    refusal = peer_gw._unavailable("letter_id_reused", "refused there", 409, "letter_id_reused")
    send, journal, receipts, held = _abstract_sender(tmp_path / "first", monkeypatch)
    assert send(refusal)["state"] == "failed" and receipts == ["failed"], "nothing unconfirmed before: its own end"
    assert journal == ["record", "POST", "settle"], "recorded as sending before its POST, settled by its receipt"

    send, journal, receipts, held = _abstract_sender(tmp_path / "later", monkeypatch, states={"uncertain"},
                                                     record={"letter_id": "msg-0001", "passes": 1})
    kept = send(refusal)
    assert (kept["state"], kept["code"], kept["recorded"], receipts) == (
        "uncertain", "earlier_attempt_unconfirmed", True, []), kept
    assert send(None)["state"] == "delivered" and receipts == ["delivered"], "a confirmed delivery clears it"
    assert held["record"] is None, "its delivered receipt settles the sending record"


def test_an_earlier_outcome_is_read_at_its_address_before_the_current_admission(tmp_path, monkeypatch):
    """The same letter again answers its earlier outcome from its receipt, whatever the link's state now: a delivered
    or refused letter sends nothing, even with the link unusable; an unconfirmed one meets the unusable link as
    unconfirmed, never "nothing was sent"; with nothing earlier the same link holds it."""
    send, journal, receipts, _held = _abstract_sender(tmp_path / "d", monkeypatch, states={"delivered"},
                                                      link_usable=False)
    done = send()
    assert (done["state"], done["earlier"], journal, receipts) == ("delivered", True, [], []), done
    send, journal, receipts, _held = _abstract_sender(tmp_path / "f", monkeypatch, states={"failed"})
    refused = send()
    assert (refused["state"], refused["code"], journal) == ("failed", "refused_earlier", []), refused
    send, journal, receipts, _held = _abstract_sender(tmp_path / "u", monkeypatch, states={"uncertain"},
                                                      link_usable=False)
    kept = send()
    assert (kept["state"], kept["recorded"], journal, receipts) == ("uncertain", True, [], []), kept
    assert "may have arrived" in kept["message"] and "nothing new was sent now" in kept["message"]
    send, journal, receipts, _held = _abstract_sender(tmp_path / "n", monkeypatch, link_usable=False)
    held = send()
    assert (held["state"], held["code"], journal) == ("held", "peer_link_unavailable", []), held


def test_an_attempt_a_stop_interrupted_reads_as_unconfirmed_on_the_next_call_and_at_a_start(tmp_path, monkeypatch):
    """A sending record with no receipt after it (a stop between a POST and its receipt) is an attempt that may have
    arrived: the next call's refusal keeps the letter ``uncertain`` and gives its room that receipt, never ``failed``;
    at a start the reconciliation writes the owed ``uncertain`` receipt from the record alone, sends nothing, and a
    record whose settling receipt landed leaves."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links, presence_delivery
    from ouroboros.gateways import peer as peer_gw

    refusal = peer_gw._unavailable("link_paused", "paused there", 403, "link_paused")
    send, journal, receipts, held = _abstract_sender(tmp_path, monkeypatch,
                                                     record={"letter_id": "msg-0001", "passes": 1})
    kept = send(refusal)
    assert (kept["state"], kept["code"], journal, receipts) == ("uncertain", "earlier_attempt_unconfirmed",
                                                                ["record", "POST"], ["uncertain"]), kept
    assert held["record"]["passes"] == 2, "the record keeps counting the calls that sent it"

    written, settled = [], []
    monkeypatch.setattr(replies, "_record_receipt", lambda root, link_id, account_id, letter_id, state, **kw: written.append(
        (link_id, account_id, letter_id, state, kw["text"], kw["reply_to"], kw["origin"])) or True)
    monkeypatch.setattr(peer_links, "settle_sending", lambda root, link_id, letter_id: settled.append(letter_id))
    states = {"msg-a": set(), "msg-b": {"delivered"}, "msg-c": {"uncertain"}, "msg-d": set(), "msg-e": {"uncertain"}}
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda root, skill, account, letter_id, part:
                        states[letter_id])
    replies._reconcile_sending(tmp_path, [
        ("link-0001", {"letter_id": "msg-a", "account_id": "install-a", "text": "Lost reply.", "reply_to": "m-1",
                       "origin": {"kind": "tool", "task_id": "t-1"}}, False),
        ("link-0001", {"letter_id": "msg-b", "account_id": "install-a", "text": "Delivered."}, False),
        ("link-0001", {"letter_id": "msg-c", "account_id": "install-a", "text": "Said already."}, False),
        ("link-0002", {"letter_id": "msg-d", "account_id": "install-c", "text": "To a revoked link."}, True),
        ("link-0002", {"letter_id": "msg-e", "account_id": "install-c", "text": "Said, revoked."}, True)])
    assert [row[2] for row in written] == ["msg-a", "msg-d"]
    assert written[0] == ("link-0001", "install-a", "msg-a", "uncertain", "Lost reply.", "m-1",
                          {"kind": "tool", "task_id": "t-1"})
    assert settled == ["msg-b", "msg-d", "msg-e"], "a record leaves only once its room's receipt says it"


def test_each_physical_attempt_meets_its_gate_and_a_closed_gate_keeps_an_earlier_unconfirmed_attempt(
        tmp_path, monkeypatch):
    """A gate (a tool's task launch gate) runs each physical attempt: closed before the first, nothing was sent and
    nothing is recorded as sending; closed after an unconfirmed attempt, the letter stays ``uncertain``."""
    import ouroboros.peer_replies as replies
    from ouroboros.gateways import peer as peer_gw

    send, journal, receipts, held = _abstract_sender(tmp_path, monkeypatch)
    calls = {"n": 0}

    def gate(attempt):
        calls["n"] += 1
        if calls["n"] > closes_after:
            raise replies.AttemptHeld("owner_pause", "owner_pause; nothing was sent")
        return attempt()

    closes_after = 0
    shut = replies.send_peer_message(tmp_path, "place-b", letter_id="msg-0001", text="Reply.", gate=gate,
                                     origin={"kind": "tool", "task_id": "t-1"})
    assert (shut["state"], shut["code"], journal, receipts) == ("held", "owner_pause", [], []), shut
    lost = peer_gw._unavailable("peer_unreachable", "no answer")
    closes_after, calls["n"] = 1, 0
    peer_gw.PeerGateway.answers[:] = [lost]  # the helper's stand-in wire: one lost answer
    kept = replies.send_peer_message(tmp_path, "place-b", letter_id="msg-0001", text="Reply.", gate=gate,
                                     origin={"kind": "tool", "task_id": "t-1"})
    assert (kept["state"], journal, receipts) == ("uncertain", ["record", "POST"], ["uncertain"]), kept
    assert calls["n"] == 2 and held["record"]["passes"] == 1, "the second attempt met the closed gate"


def test_one_call_at_a_time_sends_a_letter_and_a_readable_delivery_outlives_an_unreadable_record(
        tmp_path, monkeypatch):
    """A second observer of a letter being sent here answers ``busy`` without waiting and sends nothing; a delivered
    receipt in readable history answers ``delivered`` though the sending record cannot be read, while with no receipt
    that unreadable record is ``unknown`` (no POST, no fabricated ``uncertain`` receipt)."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links

    send, journal, receipts, held = _abstract_sender(tmp_path, monkeypatch)
    claim = f"{tmp_path.resolve()}\0link-0001\0msg-0001"
    replies._SENDING.add(claim)
    try:
        busy = send(None)
    finally:
        replies._SENDING.discard(claim)
    assert (busy["state"], busy["code"], journal, receipts) == ("busy", "letter_in_flight", [], []), busy

    def unreadable(root, link_id, letter_id):
        raise peer_links.PeerLinkError("peer_links_unreadable", "needs repair", 503)

    send, journal, receipts, held = _abstract_sender(tmp_path / "d", monkeypatch, states={"delivered"})
    monkeypatch.setattr(peer_links, "sending_record", unreadable)
    assert send()["state"] == "delivered" and journal == []
    send, journal, receipts, held = _abstract_sender(tmp_path / "u", monkeypatch, link_usable=False)
    monkeypatch.setattr(peer_links, "sending_record", unreadable)
    kept = send()
    assert (kept["state"], kept["code"], kept["recorded"], journal, receipts) == (
        "unknown", "peer_links_unreadable", False, [], []), kept


def test_a_typed_retry_after_an_unconfirmed_attempt_is_that_posts_own_fact(tmp_path, monkeypatch):
    """An earlier unconfirmed attempt and this POST's typed retry (429) are two facts: the letter stays ``uncertain``
    and carries the receiver's own delay for the same letter."""
    from ouroboros.gateways import peer as peer_gw

    busy = peer_gw._unavailable("presence_capacity_full", "busy", 429, "presence_capacity_full")
    busy.retry_after = 7
    send, journal, receipts, held = _abstract_sender(tmp_path, monkeypatch, states={"uncertain"},
                                                     record={"letter_id": "msg-0001", "passes": 1})
    kept = send(busy)
    assert (kept["state"], kept["retry_code"], kept["retry_after"], receipts) == (
        "uncertain", "presence_capacity_full", 7, []), kept


def _handed_off_work(pair, turn_text, child, result, *, approve=True):
    """The work a turn handed off (``child``), ended: its terminal as the runner writes it, with the turn that drafted
    it as its objective author and its selection's approval; the turn's own record is read for both."""
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

    with acting_as(pair.root_b):
        turn = load_task_result(pair.root_b, _arrival(pair.root_b, turn_text)["task_id"])
        metadata = {"presence": turn["metadata"]["presence"],
                    "objective_author": {"kind": "task", "task_id": turn["task_id"]}}
        if approve:
            _approve(pair, child, result, metadata)
        write_task_result(pair.root_b, child, STATUS_COMPLETED, result=result, terminal_origin="model_final",
                          chat_id=1, source="presence", metadata=metadata)
        return load_task_result(pair.root_b, child)


def test_a_handed_off_works_answer_goes_once_whether_it_ends_before_or_after_its_turn(tmp_path, monkeypatch):
    """A turn hands its answer to a work; its deferred exit answers another input than its waking message (a second
    message its model covered). Ended after the turn, the work answers at its own end; ended before it (its end
    reported while the turn still runs), it sends nothing then and the turn's end answers it. Both answers name the
    input the turn's exit answered, never the waking message; each goes once, no record is left owed and no second
    model turn starts."""
    import ouroboros.peer_replies as replies

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched, seen = [], {}

    def child_first(task, _words):
        if task["metadata"]["presence"]["event"]["message"]["message_id"] != "fast-1":
            return
        work = _handed_off_work(pair, "Quick count.", "handoff-fast-1", "Count: 12.")
        replies.deliver_owed(pair.root_b, "handoff-fast-1", record=work)  # its end, while the turn still runs
        _poll_until(_replies_idle, label="the work's reply thread ended")
        seen["early"] = "Count: 12." in [m["text"] for m in _room(seen["a"], pair.root_a, pair.place_a)]
        seen["owed"] = (pair.link_id, "handoff-fast-1") in _owed(pair.root_b)

    _settle_turns(pair, monkeypatch, {"Please rerun eval.": ("deferred", "", "handoff-slow-1", "slow-anchor"),
                                      "Quick count.": ("deferred", "", "handoff-fast-1", "fast-anchor")}, dispatched,
                  before_end=child_first)
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        seen["a"] = client_a
        assert client_b.post("/api/letters", json=_msg(letter_id="slow-1", text="Please rerun eval."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: _owed(pair.root_b) == [(pair.link_id, "handoff-slow-1")], label="the work owes it")
        work = _handed_off_work(pair, "Please rerun eval.", "handoff-slow-1", "Eval rerun: 0.91.")
        replies.deliver_owed(pair.root_b, "handoff-slow-1", record=work)  # its end, after the turn's
        _poll_until(lambda: "Eval rerun: 0.91." in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)],
                    label="A received the later work's answer")
        assert client_b.post("/api/letters", json=_msg(letter_id="fast-1", text="Quick count."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: "Count: 12." in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)],
                    label="A received the earlier work's answer at its turn's end")
        _poll_until(_replies_idle, label="reply threads ended")
        assert seen == {"a": client_a, "early": False, "owed": True}, "before its turn's end it sent nothing, owed"
        texts = [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        assert texts.count("Eval rerun: 0.91.") == 1 and texts.count("Count: 12.") == 1, texts
        assert _owed(pair.root_b) == [], "no record of an answered work is left owed"
    assert _arrival(pair.root_a, "Eval rerun: 0.91.")["transport"]["message"]["reply_to"] == "slow-anchor"
    assert _arrival(pair.root_a, "Count: 12.")["transport"]["message"]["reply_to"] == "fast-anchor"
    assert [text for _root, text in dispatched] == ["Please rerun eval.", "Quick count."]
    assert sorted(r["transport"]["delivery"]["state"] for r in _receipts(pair.root_b)) == ["delivered"] * 2


def test_a_handed_off_works_held_answer_is_said_once_and_never_sent_later(tmp_path, monkeypatch):
    """The work ends first and its answer is held (a note of my human's after its reading): said once in B's room. The
    turn's end, and a later start, find that end: no second notice, nothing sent, nothing left owed."""
    import ouroboros.peer_replies as replies
    from ouroboros import places
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)

    def child_first(task, _words):
        work = _handed_off_work(pair, "Count again.", "handoff-held-1", "Count: 13.")
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Say nothing about counts to Laptop.")
        replies.deliver_owed(pair.root_b, "handoff-held-1", record=work)
        _poll_until(_replies_idle, label="the work's reply thread ended")

    _settle_turns(pair, monkeypatch, {"Count again.": ("deferred", "", "handoff-held-1")}, [],
                  before_end=child_first)
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        assert client_b.post("/api/letters", json=_msg(letter_id="held-1", text="Count again."),
                             headers=pair.headers()).status_code == 200
        _poll_until(_replies_idle, label="reply threads ended")
        replies.resume_owed(pair.root_b)  # a later start
        _poll_until(_replies_idle, label="reply threads ended")
        assert [m["text"] for m in _held_notices(client_b, pair)] == [
            "Reply held: the rules for this conversation changed after these words were approved; nothing was sent"]
        assert "Count: 13." not in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        assert _owed(pair.root_b) == []
    assert _receipts(pair.root_b) == []


def test_a_typed_retry_is_followed_up_after_its_delay_and_a_letters_last_pass_ends_it_honestly(tmp_path, monkeypatch):
    """The receiver's typed retry (429) of an automatic reply is followed up in the same thread after its delay, as the
    same letter under the rules as they stand then, and delivered once. A letter no pass of which was confirmed ends at
    its last pass (``_MAX_PASSES``): said once in B's room, nothing more sent, nothing left owed."""
    import ouroboros.peer_replies as replies
    from ouroboros.gateways import peer as peer_gw

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", (0.0, 0.0, 0.0))
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Later.": ("message", "Reply after a retry.", ""),
                                      "Never.": ("message", "Reply never taken.", "")}, [])
    real_request, refusals = peer_gw.PeerGateway._request, {"left": 1, "always": False}

    def busy(self, method, path, **kw):
        if self._link.place_id == pair.place_b and (refusals["always"] or refusals["left"]):
            refusals["left"] = max(0, refusals["left"] - 1)
            exc = peer_gw._unavailable("presence_capacity_full", "busy there", 429, "presence_capacity_full")
            exc.retry_after = 0
            raise exc
        return real_request(self, method, path, **kw)

    def texts_a(client_a):
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        monkeypatch.setattr(peer_gw.PeerGateway, "_request", busy)
        assert client_b.post("/api/letters", json=_msg(letter_id="later-1", text="Later."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: "Reply after a retry." in texts_a(client_a), label="the followed-up reply arrived")
        _poll_until(_replies_idle, label="reply threads ended")
        refusals["always"] = True
        assert client_b.post("/api/letters", json=_msg(letter_id="never-1", text="Never."),
                             headers=pair.headers()).status_code == 200
        _poll_until(_replies_idle, label="reply threads ended")
        assert texts_a(client_a).count("Reply after a retry.") == 1 and "Reply never taken." not in texts_a(client_a)
        notices = [m["text"] for m in _held_notices(client_b, pair)] + [
            m["text"] for m in _room(client_b, pair.root_b, pair.place_b)
            if m["role"] == "system" and str(m["text"]).startswith("Reply not confirmed:")]
        assert len(notices) == 1 and "nothing more is sent automatically" in notices[0], notices
        assert _owed(pair.root_b) == []
    states = sorted((r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b))
    assert states == [("Reply after a retry.", "delivered"), ("Reply never taken.", "uncertain")], states


def test_a_pause_of_my_place_keeps_the_owed_reply_and_it_goes_once_the_place_is_active_again(tmp_path, monkeypatch):
    """While my place for the link is Paused, a start reads the reply's own recorded words (never the empty outward
    view the Pause gives them): nothing is sent or said and the reply stays owed; the first start after my human makes
    the place Active again sends it once, under the rules its words were approved under."""
    import ouroboros.peer_replies as replies
    from ouroboros import places
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Paused.": ("message", "Reply after the pause.", "")}, [])

    def texts_a(client_a):
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Paused.")
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_state", author=places.AUTHOR_OWNER, state="paused")
        replies.resume_owed(pair.root_b)
        _poll_until(_replies_idle, label="reply threads ended")
        assert "Reply after the pause." not in texts_a(client_a) and _owed(pair.root_b) == [(pair.link_id, turn)]
        assert [m for m in _room(client_b, pair.root_b, pair.place_b) if m["role"] == "system"] == []
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_state", author=places.AUTHOR_OWNER, state="active")
        replies.resume_owed(pair.root_b)
        _poll_until(lambda: "Reply after the pause." in texts_a(client_a), label="A received it after the pause")
        _poll_until(_replies_idle, label="reply threads ended")
        assert texts_a(client_a).count("Reply after the pause.") == 1 and _owed(pair.root_b) == []


def test_a_rules_change_before_a_send_tells_an_unfinished_attempt_from_unwritten_history(tmp_path, monkeypatch):
    """An owed reply whose last attempt left with no receipt after it (its sending record, a stop between its POST and
    its receipt), and whose rules then changed: its end reported again holds it without a POST, gives its room the
    ``uncertain`` receipt that attempt owed and says it may have arrived, never "nothing was sent"."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links, places
    from ouroboros.install_identity import acting_as
    from ouroboros.presence_continuation import output_ref
    from ouroboros.task_results import load_task_result

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Cut.": ("message", "Reply cut short.", "")}, [])
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Cut.")
        with acting_as(pair.root_b):
            event = load_task_result(pair.root_b, turn)["metadata"]["presence"]["event"]
            letter_id = replies.message_letter_id(pair.link_id, output_ref(turn, "terminal", event["conversation_key"]))
            peer_links.record_sending(pair.root_b, pair.link_id, letter_id, account_id=event["account_id"],
                                      text="Reply cut short.", reply_to="m-Cut.",
                                      origin={"kind": "automatic", "task_id": turn, "source_event_id": "peer-msg:m-Cut."})
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Ask me before replying to Laptop.")
        replies.deliver_owed(pair.root_b, turn)  # its end reported again: no start's reconciliation runs first
        _poll_until(_replies_idle, label="reply threads ended")
        assert [m["text"] for m in _held_notices(client_b, pair)] == [
            "Reply held: the rules for this conversation changed after these words were approved; nothing was sent "
            "now, and an earlier attempt of these words is unconfirmed: it may have arrived"]
        assert _owed(pair.root_b) == []
    assert [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)] == [
        ("Reply cut short.", "uncertain")]


def test_an_address_written_beside_an_approval_reads_and_replaces_it_as_one_step(tmp_path, monkeypatch):
    """A work's owed record is read and replaced under the registry's writer lock (``peer_links.record_owed``): a
    writer of only its address keeps an approval another writer recorded, and the read happens inside that lock, so
    no concurrent approval falls between them. Pure stand-ins: no store beyond the record files."""
    import contextlib

    from ouroboros import peer_links, utils

    journal: list = []
    real_read = peer_links.owed_record

    @contextlib.contextmanager
    def lock(path, **_kw):
        journal.append("lock")
        yield
        journal.append("unlock")

    monkeypatch.setattr(utils, "json_update_lock", lock)
    monkeypatch.setattr(peer_links, "owed_record", lambda *a: journal.append("read") or real_read(*a))
    peer_links.record_owed(tmp_path, "link-0001", "work-1", place_id="place-b", approved={"sha": "basis-1"})
    journal.clear()
    peer_links.record_owed(tmp_path, "link-0001", "work-1", place_id="place-b")  # an address only
    assert journal == ["lock", "read", "unlock"], journal
    assert real_read(tmp_path, "link-0001", "work-1")["approved"] == {"sha": "basis-1"}


def test_automatic_reply_lost_receipt_reconciles_confirmed_even_under_pause_and_rules_change(tmp_path, monkeypatch):
    """When an automatic reply is delivered at A (or definitely refused on first send) but writing its canonical
    receipt fails, ``sent/`` preserves ``confirmed_state``. Even if B is then Paused and its rules change (or the link
    is revoked on startup reconciliation), reconciling writes the confirmed receipt with zero extra POSTs and never
    emits ``Reply held`` or downgrades to ``uncertain``."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_links, places, presence_delivery
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", ())
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {"Confirm.": ("message", "Confirmed automatic reply.", "")}, [])

    real_record = replies._record_receipt
    fail_receipt = {"on": True}
    posts = {"n": 0}
    real_request = peer_gw.PeerGateway._request

    def counting_request(self, method, path, **kw):
        if self._link.place_id == pair.place_b and method == "POST" and path == "/api/letters":
            posts["n"] += 1
        return real_request(self, method, path, **kw)

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Confirm.")
        monkeypatch.setattr(peer_gw.PeerGateway, "_request", counting_request)
        monkeypatch.setattr(replies, "_record_receipt", lambda *a, **kw: False if fail_receipt["on"] else real_record(*a, **kw))
        replies.resume_owed(pair.root_b, at_start=False)
        _poll_until(_replies_idle, label="first pass ended")
        assert posts["n"] == 1 and _owed(pair.root_b) == [(pair.link_id, turn)]
        assert [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)].count("Confirmed automatic reply.") == 1

        # Now Pause AND change conversation rules on B before receipt storage recovers.
        fail_receipt["on"] = False
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_state", author=places.AUTHOR_OWNER, state="paused")
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Rule changed after confirmed delivery.")
        replies.resume_owed(pair.root_b, at_start=False)
        _poll_until(_replies_idle, label="reconcile pass ended")
        assert posts["n"] == 1, "already-confirmed automatic reply sends no second POST"
        assert _owed(pair.root_b) == [] and _held_notices(client_b, pair) == []
    assert [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)] == [
        ("Confirmed automatic reply.", "delivered")]

    # Startup _reconcile_sending also preserves confirmed_state (delivered over prior uncertain, failed, revoked).
    written, settled = [], []
    monkeypatch.setattr(replies, "_record_receipt", lambda root, link_id, account_id, letter_id, state, **kw:
                        written.append((letter_id, state)) or True)
    monkeypatch.setattr(peer_links, "settle_sending", lambda root, link_id, letter_id: settled.append(letter_id))
    existing = {"m-del": {"uncertain"}, "m-fail": set()}
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda root, skill, acc, lid, part: existing[lid])
    replies._reconcile_sending(tmp_path, [
        ("link-1", {"letter_id": "m-del", "account_id": "inst-a", "text": "Yes", "confirmed_state": "delivered"}, True),
        ("link-1", {"letter_id": "m-fail", "account_id": "inst-a", "text": "No", "confirmed_state": "failed"}, True),
    ])
    assert written == [("m-del", "delivered"), ("m-fail", "failed")]
    assert settled == ["m-del", "m-fail"]


def test_transient_storage_pause_and_unreadable_switch_preserve_owed_while_revocation_stays_terminal(
        tmp_path, monkeypatch):
    """Transient storage errors (``owed_record``, ``speech_basis``, ``record_sending``, ``places_unreadable``,
    ``peer_links_unreadable``), unreadable switch (``switch_state == 'unknown'``), and lawful Pause/Silence between
    check and send all keep an unsent reply owed with no ``Reply held``; real revocation (even though it pauses the
    place) keeps its terminal hold."""
    import ouroboros.peer_replies as replies
    from ouroboros import peer_letters, peer_links, place_projection
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", ())
    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _settle_turns(pair, monkeypatch, {
        "Keep.": ("message", "Preserved reply.", ""),
        "Revoke.": ("message", "Revoked reply.", ""),
    }, [])

    def resume():
        replies.resume_owed(pair.root_b, at_start=False)
        _poll_until(_replies_idle, label="reply threads ended")

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        turn = _unreported_turn(pair, monkeypatch, client_b, "Keep.")

        # 1. Unreadable switch ("unknown") in resume_owed and in _deliver_output.
        real_switch = place_projection.switch_state
        monkeypatch.setattr(place_projection, "switch_state", lambda root: "unknown")
        resume()
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        assert replies._deliver_until_settled(pair.root_b, pair.link_id, turn) is False
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        monkeypatch.setattr(place_projection, "switch_state", real_switch)

        # 2. Transient owed_record PeerLinkError.
        real_owed_read = peer_links.owed_record
        monkeypatch.setattr(peer_links, "owed_record", lambda *a: (_ for _ in ()).throw(
            peer_links.PeerLinkError("peer_links_unreadable", "busy", 503)))
        resume()
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        monkeypatch.setattr(peer_links, "owed_record", real_owed_read)

        # 3. Transient speech_basis unreadable ("") while link & place are active -> rules_unreadable (still owed).
        real_basis = peer_letters.speech_basis
        monkeypatch.setattr(peer_letters, "speech_basis", lambda *a, **kw: "")
        resume()
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        monkeypatch.setattr(peer_letters, "speech_basis", real_basis)

        # 4. Transient record_sending lock/IO failure -> sending_unrecorded (still owed).
        real_rec_send = peer_links.record_sending
        monkeypatch.setattr(peer_links, "record_sending", lambda *a, **kw: (_ for _ in ()).throw(
            peer_links.PeerLinkError("peer_links_unreadable", "lock timeout", 503)))
        resume()
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        monkeypatch.setattr(peer_links, "record_sending", real_rec_send)

        # 5. Lawful Pause occurring BETWEEN _deliver_output's check and gateway.send_message (mid-send race).
        real_send_msg = peer_gw.PeerGateway.send_message
        monkeypatch.setattr(peer_gw.PeerGateway, "send_message", lambda *a, **kw: (_ for _ in ()).throw(
            peer_gw._unavailable("place_paused", "paused right before wire")))
        resume()
        assert _owed(pair.root_b) == [(pair.link_id, turn)] and _held_notices(client_b, pair) == []
        monkeypatch.setattr(peer_gw.PeerGateway, "send_message", real_send_msg)

        # 6. Once transient faults clear, the exact preserved reply delivers once!
        resume()
        assert [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)].count("Preserved reply.") == 1
        assert _owed(pair.root_b) == [] and _held_notices(client_b, pair) == []

        # 7. Control: a revoked link (even though revoke() sets place state=paused!) keeps its terminal hold.
        rev_turn = _unreported_turn(pair, monkeypatch, client_b, "Revoke.")
        with acting_as(pair.root_b):
            saved_owed = peer_links.owed_record(pair.root_b, pair.link_id, rev_turn)
            peer_links.revoke(pair.root_b, pair.link_id)
            # Re-plant owed record to simulate a race where revoke ran while _deliver_output was about to deliver.
            peer_links.record_owed(pair.root_b, pair.link_id, rev_turn, place_id=pair.place_b,
                                   approved=saved_owed.get("approved"))
        assert replies._deliver_until_settled(pair.root_b, pair.link_id, rev_turn) is True
        assert _owed(pair.root_b) == []
        assert len(_held_notices(client_b, pair)) == 1 and "revoked" in _held_notices(client_b, pair)[0]["text"]

    # _prestart_unowed also keeps any owed record when owed_record raises PeerLinkError.
    from ouroboros.gateway import peer_letters_api
    monkeypatch.setattr(peer_links, "owed_record", lambda *a: (_ for _ in ()).throw(
        peer_links.PeerLinkError("peer_links_unreadable", "busy", 503)))
    assert peer_letters_api._prestart_unowed(pair.root_b, pair.link_id, "turn-x", "conv-x", "id-x") is False


def test_resume_owed_does_not_head_of_line_block_healthy_peer_behind_unreachable_peer(tmp_path, monkeypatch):
    """When returning/resuming owed replies across multiple peer links, one unreachable/slow peer sleeping or blocking
    in its retry queue must not delay another healthy peer's first delivery attempt."""
    import threading

    import ouroboros.peer_replies as replies
    from ouroboros import peer_links

    release_a = threading.Event()
    a_entered = threading.Event()
    b_delivered = threading.Event()
    order: list[str] = []

    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", (60.0, 120.0))
    monkeypatch.setattr(peer_links, "owed_refs", lambda root, lid="": [("link-slow", "turn-a"), ("link-fast", "turn-b")])
    monkeypatch.setattr(peer_links, "sending_records", lambda root: [])
    monkeypatch.setattr(peer_links, "owed_version", lambda root, lid, ref: (lid, ref))
    monkeypatch.setattr("ouroboros.place_projection.switch_state", lambda root: "on")

    def fake_deliver(root, ref, *, link_id, seen):
        if link_id == "link-slow":
            order.append("a-start")
            a_entered.set()
            release_a.wait(timeout=5.0)
            order.append("a-end")
            return False, None  # exit cleanly once unblocked
        order.append("b-delivered")
        b_delivered.set()
        return True, None

    monkeypatch.setattr(replies, "_deliver_producer", fake_deliver)
    replies.resume_owed(tmp_path, at_start=False)
    try:
        assert a_entered.wait(timeout=2.0), "slow peer worker started"
        assert b_delivered.wait(timeout=2.0), "healthy peer delivered without waiting for slow peer"
        assert not release_a.is_set() and "a-end" not in order
    finally:
        release_a.set()
        _poll_until(_replies_idle, label="all resume workers exited cleanly")
    assert "b-delivered" in order and "a-end" in order

    # Same-link batch also runs pass 1 for every queued ref before sleeping on any follow-up retry.
    seq: list[str] = []
    monkeypatch.setattr(replies, "_FOLLOW_UP_SEC", (0.05,))
    monkeypatch.setattr(replies, "_deliver_producer", lambda root, ref, *, link_id, seen: (
        seq.append(ref),
        (False, 0.05) if ref == "ref-1" and seq.count("ref-1") == 1 else (True, None),
    )[1])
    assert replies._deliver_batch(tmp_path, [("link-one", "ref-1"), ("link-one", "ref-2")]) is True
    assert seq == ["ref-1", "ref-2", "ref-1"], seq


def test_prewire_unavailable_reclassification_and_silent_speech_basis_mismatch_stay_owed(tmp_path, monkeypatch):
    """(1) Prewire ``peer_link_unavailable`` from ``PeerGateway._exchange`` is re-classified against fresh proven
    state: Pause, Silence, unreadable store, or no proven terminal (``peer_link_transient``) keeps the reply owed;
    confirmed revocation stays terminal. Non-empty ``speech_basis`` mismatch caused by lawful ``Silent`` returns
    ``place_silent`` (owed) instead of terminal ``rules_changed``."""
    from types import SimpleNamespace

    import ouroboros.peer_replies as replies
    from ouroboros import peer_letters, peer_links, place_projection, places, presence_delivery
    from ouroboros.gateways import peer as peer_gw

    link = peer_links.OutgoingLink(
        link_id="link-0001", place_id="place-b", peer_url="http://127.0.0.1:9",
        peer_label="Laptop", peer_install_id="install-a", own_install_id="install-b", drive_root=str(tmp_path),
    )
    place = SimpleNamespace(place_id="place-b", conversation_id="link-0001", account_id="install-a",
                            provider="ouroboros", connector="ouroboros")
    notices: list[str] = []
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "on")
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda *a: set())
    monkeypatch.setattr(peer_links, "sending_record", lambda *a: None)
    monkeypatch.setattr(peer_links, "record_sending", lambda *a, **kw: 1)
    monkeypatch.setattr(peer_links, "rollback_sending", lambda *a, **kw: True)
    monkeypatch.setattr(peer_links, "owed_record", lambda *a: {
        "work_ref": "turn-1",
        "approved": {__import__("hashlib").sha256(b"Reply.").hexdigest(): "basis-active"},
    })
    monkeypatch.setattr(replies, "_said_held", lambda *a: False)
    monkeypatch.setattr(replies, "_tool_sent", lambda *a: False)
    monkeypatch.setattr("ouroboros.place_reply.record_notice", lambda *a, **kw: notices.append(kw["text"]))

    # 1a. Non-empty speech_basis mismatch while place is Silent -> place_silent (owed), NOT terminal rules_changed.
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: ("link-0001", {"peer_install_id": "install-a"}))
    monkeypatch.setattr(peer_links, "outgoing_link", lambda *a, **kw: link)
    monkeypatch.setattr(peer_letters, "speech_basis", lambda *a, **kw: "basis-silent-nonempty")
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda *a: ("place_silent", "Silent here; nothing was sent"))
    hold = replies._check_speech_basis(tmp_path, "place-b", "link-0001", kind="presence",
                                       approved_basis="basis-active", expected_account_id="install-a")
    assert hold is not None and hold[0] == "place_silent" and hold[0] in replies._STILL_OWED

    # Control: non-empty mismatch while Active & usable -> terminal rules_changed.
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda *a: None)
    hold_rules = replies._check_speech_basis(tmp_path, "place-b", "link-0001", kind="presence",
                                             approved_basis="basis-active", expected_account_id="install-a")
    assert hold_rules is not None and hold_rules[0] == "rules_changed"

    # 1b. Prewire peer_link_unavailable from PeerGateway.send_message re-classified per fresh state.
    monkeypatch.setattr(peer_letters, "speech_basis", lambda *a, **kw: "basis-active")
    state = {"speech_refusal": None, "row": {"peer_install_id": "install-a"}, "outgoing": link}
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: ("link-0001", state["row"]))
    monkeypatch.setattr(peer_links, "outgoing_link", lambda *a, **kw: state["outgoing"])
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda *a: state["speech_refusal"])

    for case, after_wire, expect_settled in (
        ("paused", {"speech_refusal": ("place_paused", "Paused; nothing was sent")}, False),
        ("silent", {"speech_refusal": ("place_silent", "Silent; nothing was sent")}, False),
        ("places_unreadable", {"speech_refusal": ("places_unreadable", "Unreadable; nothing was sent")}, False),
        ("unproven_transient", {}, False),  # back to Active/usable before recheck -> peer_link_transient (owed)
        ("revoked", {"row": {"peer_install_id": "install-a", "revoked_at": "2026-11-25T00:00:00Z"},
                     "speech_refusal": ("place_paused", "Paused by revoke")}, True),
    ):
        notices.clear()
        state.update(speech_refusal=None, row={"peer_install_id": "install-a"}, outgoing=link)

        def wire_fail(*a, _after=after_wire, **kw):
            state.update(_after)
            raise peer_gw._unavailable("peer_link_unavailable", "no usable link in _exchange")

        monkeypatch.setattr(peer_gw, "PeerGateway", lambda _l: SimpleNamespace(
            send_message=wire_fail, close=lambda: None))
        settled, wait = replies._deliver_output(
            tmp_path, place, "conv-key", producer="turn-1", output_ref="out-1",
            text="Reply.", reply_to="m-1", source_event_id="ev-1", kind="presence",
        )
        assert (settled, wait) == (expect_settled, None), (case, settled, wait, notices)
        assert bool(notices) == expect_settled, (case, notices)

    # Also verify _explain_missing_outgoing treats unproven None outgoing_link as transient while Active+mine+secret.
    monkeypatch.setattr(places, "place_records", lambda root: SimpleNamespace(
        places={"place-b": SimpleNamespace(state=places.STATE_ACTIVE, label="Laptop")},
        peer=lambda pid: {"mine": True},
    ))
    monkeypatch.setattr(peer_links, "_read_secret", lambda root, lid: "secret-key")
    state.update(speech_refusal=None, row={"peer_install_id": "install-a"}, outgoing=None)
    code, _msg = replies._explain_missing_outgoing(tmp_path, "place-b", "link-0001", expected_account_id="install-a")
    assert code == "peer_link_transient" and code in replies._STILL_OWED

    # Control: fail-soft empty _read_secret with outgoing_link=None on proven Active/mine/confirmed link stays
    # transient (owed, no held notice), and the normal send path is admissible again once readable.
    notices.clear()
    receipts: list[str] = []
    posts: list[str] = []
    monkeypatch.setattr(replies, "_receipts", lambda root: SimpleNamespace(
        record=lambda conn, rcpt: receipts.append(rcpt["state"])))
    monkeypatch.setattr(peer_links, "settle_sending", lambda *a: None)
    monkeypatch.setattr(peer_links, "_read_secret", lambda root, lid: "")
    state.update(
        speech_refusal=None,
        row={"peer_install_id": "install-a", "created_at": "2026-11-25T00:00:00Z",
             "code_confirmed_at": "2026-11-25T00:00:01Z"},
        outgoing=None,
    )
    code_empty, _msg_empty = replies._explain_missing_outgoing(
        tmp_path, "place-b", "link-0001", expected_account_id="install-a",
    )
    assert code_empty == "peer_link_transient" and code_empty in replies._STILL_OWED
    monkeypatch.setattr(peer_gw, "PeerGateway", lambda _l: SimpleNamespace(
        send_message=lambda text, **kw: posts.append(text) or {"state": "received"},
        close=lambda: None,
    ))
    settled_tmp, wait_tmp = replies._deliver_output(
        tmp_path, place, "conv-key", producer="turn-1", output_ref="out-1",
        text="Reply.", reply_to="m-1", source_event_id="ev-1", kind="presence",
    )
    assert (settled_tmp, wait_tmp, posts, notices) == (False, None, [], [])
    monkeypatch.setattr(peer_links, "_read_secret", lambda root, lid: "secret-key")
    state["outgoing"] = link
    settled_ok, wait_ok = replies._deliver_output(
        tmp_path, place, "conv-key", producer="turn-1", output_ref="out-1",
        text="Reply.", reply_to="m-1", source_event_id="ev-1", kind="presence",
    )
    assert (settled_ok, wait_ok, posts, receipts, notices) == (True, None, ["Reply."], ["delivered"], [])


def test_automatic_output_repeat_freezes_original_empty_or_addressed_reply_to_and_rejects_mismatch(
        tmp_path, monkeypatch):
    """(2) Repeating an automatic output with the same ``output_ref`` keeps its original recorded ``reply_to``
    (including empty ``""``) even after a newer ``reply_anchor`` arrives, never lets ``record_sending`` or
    ``_reconcile_confirmed`` overwrite ``""``, and rejects mismatched text or recipient without POSTing."""
    from types import SimpleNamespace

    import ouroboros.peer_replies as replies
    from ouroboros import peer_letters, peer_links, place_projection, presence_delivery
    from ouroboros.gateways import peer as peer_gw

    link = peer_links.OutgoingLink(
        link_id="link-0001", place_id="place-b", peer_url="http://127.0.0.1:9",
        peer_label="Laptop", peer_install_id="install-a", own_install_id="install-b", drive_root=str(tmp_path),
    )
    place = SimpleNamespace(place_id="place-b", conversation_id="link-0001", account_id="install-a",
                            provider="ouroboros", connector="ouroboros")
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "on")
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: ("link-0001", {"peer_install_id": "install-a"}))
    monkeypatch.setattr(peer_links, "outgoing_link", lambda *a, **kw: link)
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda *a: None)
    monkeypatch.setattr(peer_letters, "speech_basis", lambda *a, **kw: "basis-1")
    monkeypatch.setattr(replies, "_said_held", lambda *a: False)
    monkeypatch.setattr(replies, "_tool_sent", lambda *a: False)
    monkeypatch.setattr(replies, "_RETRY_PAUSE_SEC", 0.0)
    monkeypatch.setattr(peer_links, "owed_record", lambda *a: {
        "work_ref": "turn-1",
        "approved": {__import__("hashlib").sha256(b"Early output.").hexdigest(): "basis-1"},
    })

    wire_calls: list[dict] = []
    receipts: list[dict] = []
    monkeypatch.setattr(replies, "_receipts", lambda root: SimpleNamespace(
        record=lambda conn, rcpt: receipts.append(rcpt)))
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda *a: set())

    # Pass 1: original reply_to is "" (empty standalone); wire fails unconfirmed, leaving sent/ with reply_to="".
    monkeypatch.setattr(peer_gw, "PeerGateway", lambda _l: SimpleNamespace(
        send_message=lambda text, **kw: (wire_calls.append({"text": text, **kw}),
                                         (_ for _ in ()).throw(peer_gw._unavailable("peer_unreachable", "lost"))),
        close=lambda: None,
    ))
    settled1, _wait1 = replies._deliver_output(
        tmp_path, place, "conv-key", producer="turn-1", output_ref="turn-1:out:0",
        text="Early output.", reply_to="", source_event_id="ev-1", kind="presence",
    )
    assert settled1 is False
    lid = replies.message_letter_id("link-0001", "turn-1:out:0")
    assert peer_links.sending_record(tmp_path, "link-0001", lid)["reply_to"] == ""

    # Mismatched text or recipient on same letter_id is rejected with zero POSTs.
    wire_calls.clear()
    bad_text = replies.send_peer_message(
        tmp_path, "place-b", letter_id=lid, text="Different words.", reply_to="m-2",
        origin={"kind": "automatic", "task_id": "turn-1"}, link_id="link-0001", account_id="install-a",
    )
    bad_rcpt = replies.send_peer_message(
        tmp_path, "place-b", letter_id=lid, text="Early output.", reply_to="m-2",
        origin={"kind": "automatic", "task_id": "turn-1"}, link_id="link-0001", account_id="install-other",
    )
    assert bad_text["state"] == "held" and bad_text["code"] == "retry_text_mismatch"
    assert bad_rcpt["state"] == "held" and bad_rcpt["code"] == "retry_recipient_mismatch"
    assert wire_calls == []

    # Pass 2: reply_anchor moved to "m-2", yet repeat keeps exact original empty reply_to="" on wire, sent/, and receipt!
    monkeypatch.setattr(peer_gw, "PeerGateway", lambda _l: SimpleNamespace(
        send_message=lambda text, **kw: wire_calls.append({"text": text, **kw}) or {"state": "received"},
        close=lambda: None,
    ))
    receipts.clear()
    settled2, _wait2 = replies._deliver_output(
        tmp_path, place, "conv-key", producer="turn-1", output_ref="turn-1:out:0",
        text="Early output.", reply_to="m-2", source_event_id="ev-1", kind="presence",
    )
    assert settled2 is True and len(wire_calls) == 1 and wire_calls[0]["reply_to"] == ""
    assert len(receipts) == 1 and "reply_to" not in receipts[0]["message"]

    # _reconcile_confirmed + confirm_sending also preserve original empty reply_to="" when called with reply_to="m-2".
    lid2 = "msg-empty-confirmed"
    peer_links.record_sending(tmp_path, "link-0001", lid2, account_id="install-a", place_id="place-b",
                              text="Empty confirmed.", reply_to="", origin={"kind": "automatic", "task_id": "t-2"})
    peer_links.confirm_sending(tmp_path, "link-0001", lid2, state="delivered", reply_to="m-2")
    assert peer_links.sending_record(tmp_path, "link-0001", lid2)["reply_to"] == ""
    receipts.clear()
    done = replies.send_peer_message(
        tmp_path, "place-b", letter_id=lid2, text="Empty confirmed.", reply_to="m-2",
        origin={"kind": "automatic", "task_id": "t-2"}, link_id="link-0001", account_id="install-a",
    )
    assert done["state"] == "delivered" and done["recorded"] is True
    assert len(receipts) == 1 and "reply_to" not in receipts[0]["message"]
