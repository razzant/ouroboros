"""A letter's card in the room of the work that wrote it (``place_projection.publish_letter_card``): one typed fact,
``peer_letter``, on that work's own progress line from the send's own outcome (``card_of_send``) and at each
``delegate_wait`` read, the sleeping wake's retained end included; a read that changes nothing adds no row, a state not
stated stays unknown, a reply shows only on the receiver's ``result_available``, long words are absent, never cut, and
the card never decides the letter. The stand-in receiver and the sending context are ``test_peer_delegate``'s."""
from __future__ import annotations

import json

import httpx
import pytest

from ouroboros import place_projection as letters
from tests.test_peer_delegate import PLACE_ID, _call, ctx, link, receiver  # noqa: F401 - fixtures


@pytest.fixture
def cards(ctx, monkeypatch):  # noqa: F811 - the imported fixture
    """The facts the work's progress line received, in order; the place's name and face seed as its link gives them."""
    said: list = []
    ctx.emit_progress_fn = lambda text, **kw: said.append((text, kw["_progress_meta_override"]["peer_letter_card"]))
    monkeypatch.setattr(letters, "_peer_names", lambda root, place_id: ("Velm", "install-b"))
    letters._PUBLISHED.clear()
    return said


def test_the_card_follows_the_start_a_question_its_answer_and_the_end(ctx, receiver, cards):  # noqa: F811
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="Count the files, please.", title="Count", cost_ceiling_usd=2)["run_id"]
    text, sent = cards[-1]
    assert text == "Letter to Velm: sent, at work"
    assert sent == {"letter_id": run_id, "place_id": PLACE_ID, "to": "Velm", "to_seed": "install-b",
                    "text": "Count the files, please.", "title": "Count", "ceiling_usd": 2.0, "spent_usd": None,
                    "spent_final": False, "state": "accepted_running", "phase": "queued"}
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "Which folder?"})
    delegate_tools._read_run(ctx, run_id, 0)
    asked = cards[-1][1]
    assert (asked["state"], asked["text"], asked["question"]) == (
        "needs_input", "Count the files, please.", {"question_id": "q1", "text": "Which folder?"})
    delegate_tools._read_run(ctx, run_id, 0)
    assert len(cards) == 2, "a read that changes nothing adds no row"
    _delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "/tmp"}])
    answered = cards[-1][1]
    assert len(cards) == 3 and answered["answered"] == {"question_id": "q1", "text": "/tmp"}, cards[-1]
    assert answered["state"] == "needs_input" and "question" not in answered, \
        "an answer taken in proves no phase there: the next read says it"
    receiver.letters[run_id].update(state="accepted_running", phase="running")
    receiver.letters[run_id].pop("question", None)
    delegate_tools._read_run(ctx, run_id, 0)
    assert (cards[-1][1]["state"], cards[-1][1]["phase"]) == ("accepted_running", "running")
    receiver.letters[run_id].update(state="done", result="There are 3 files.", result_available=True,
                                    cost={"known_usd": 0.4, "final": True})
    delegate_tools._read_run(ctx, run_id, 0)
    text, done = cards[-1]
    assert text == "Letter to Velm: finished"
    assert (done["state"], done["reply"], done["answer"], done["spent_usd"], done["spent_final"]) == (
        "done", "given", "There are 3 files.", 0.4, True)


@pytest.mark.parametrize("stated, reply", [(False, "none"), (None, "unknown")])
def test_a_final_the_receiver_did_not_give_this_sender_is_no_reply(ctx, receiver, cards, stated, reply):  # noqa: F811
    """The receiver's own final is never this sender's reply: ``result_available`` false says none, a version that
    states nothing is unknown, and neither carries words onto the card."""
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="Count them.")["run_id"]
    end = {"state": "done", "result": "B's own final words.", "cost": {"known_usd": 0.1, "final": False}}
    end.update({"result_available": stated} if stated is not None else {"omit_result_available": True})
    receiver.letters[run_id].update(end)
    delegate_tools._read_run(ctx, run_id, 0)
    done = cards[-1][1]
    assert (done["state"], done["reply"], done["spent_usd"], done["spent_final"]) == ("done", reply, 0.1, False)
    assert "answer" not in done and "B's own final" not in json.dumps(done)


def test_a_refusal_and_an_unknown_acceptance_are_cards_of_their_own(ctx, receiver, cards):  # noqa: F811
    receiver.hooks.append(lambda request: httpx.Response(403, json={"error": "link_paused", "refusal_code": "link_paused"})
                          if request.url.path == "/api/letters" and not receiver.posts[1:] else None)
    refused = _call(ctx, place_id=PLACE_ID, text="while paused")
    assert refused["refusal_code"] == "link_paused"
    assert (cards[-1][0], cards[-1][1]["state"], cards[-1][1]["refusal_code"]) == (
        "Letter to Velm: refused (link_paused)", "refused", "link_paused")
    receiver.hooks.append(lambda request: httpx.Response(503, json={"error": "unconfirmed"})
                          if request.url.path == "/api/letters" else None)
    unknown = _call(ctx, place_id=PLACE_ID, text="flaky")
    assert unknown["status"] == "unconfirmed" and cards[-1][1]["state"] == "unconfirmed"
    assert cards[-1][1]["letter_id"] == unknown["pending_invocation_id"] != refused["pending_invocation_id"]


def test_the_send_outcome_maps_to_its_card_and_a_refused_retry_stays_unknown(ctx, cards):  # noqa: F811
    """``card_of_send`` from the typed fact its producer decided (``peer_delegate._post._out``), never from the words the
    model reads: started with the receiver's state and phase, unknown acceptance, the receiver's typed refusal, nothing
    sent from here, and a retry's refusal that never proves the first send unrun. A Safety note composed onto the result
    afterwards changes none of it."""
    letter = {"letter_id": "L9", "text": "Count.", "cost_ceiling_usd": 1.0}
    outcomes = [
        {"sent": "started", "state": "accepted_running", "phase": "queued"},
        {"sent": "started", "state": "running"},
        {"sent": "unknown"},
        {"sent": "refused", "code": "peer_may_not_start"},
        {"sent": "no", "code": "place_paused"},
        {"sent": "unknown", "code": "link_paused"},
    ]
    for sent in outcomes:
        letters._PUBLISHED.clear()
        letters.card_of_send(ctx, sent, letter=letter, place_id=PLACE_ID)
    assert [(fact["state"], fact.get("phase", ""), fact.get("refusal_code", "")) for _text, fact in cards] == [
        ("accepted_running", "queued", ""), ("accepted_running", "", ""), ("unconfirmed", "", ""),
        ("refused", "", "peer_may_not_start"), ("not_sent", "", "place_paused"), ("unconfirmed", "", "link_paused")]
    letters.card_of_send(ctx, "not a typed send fact", letter=letter, place_id=PLACE_ID)
    assert len(cards) == 6, "an unreadable outcome publishes no card"


def test_the_card_never_decides_the_letter(ctx, receiver, monkeypatch):  # noqa: F811
    def broken(text, **kw):
        raise RuntimeError("the progress line is down")

    ctx.emit_progress_fn = broken
    assert _call(ctx, place_id=PLACE_ID, text="send anyway")["status"] == "started"
    monkeypatch.setattr(letters, "_peer_names", lambda root, place_id: (_ for _ in ()).throw(OSError("unread")))
    ctx.emit_progress_fn = lambda text, **kw: pytest.fail("an unreadable place publishes no card")
    assert _call(ctx, place_id=PLACE_ID, text="and again")["status"] == "started"


def test_the_sleeping_wake_publishes_the_card_from_a_retained_end(ctx, receiver, cards, monkeypatch):  # noqa: F811
    from ouroboros.tools import delegate as delegate_tools
    from tests._run_wait_shared import publish

    run_id = _call(ctx, place_id=PLACE_ID, text="Count them.")["run_id"]
    receiver.letters[run_id].update(state="done", result="Three.", result_available=True,
                                    cost={"known_usd": 0.2, "final": True})
    delegate_tools._read_run(ctx, run_id, 0)  # the end is collected and retained
    cards.clear()
    letters._PUBLISHED.clear()
    monkeypatch.setattr(delegate_tools, "_read_run", lambda *a, **k: pytest.fail("a retained end is read, not fetched"))
    wake = json.loads(publish(ctx, run_id).text)
    assert wake.get("status") == "terminal", wake
    assert [(fact["state"], fact["reply"], fact.get("answer"), fact["spent_usd"], fact["spent_final"])
            for _text, fact in cards] == [("done", "given", "Three.", 0.2, True)]


def test_a_state_not_stated_stays_unknown(ctx, receiver, cards):  # noqa: F811
    from ouroboros.tools import delegate as delegate_tools

    receiver.hooks.append(lambda request: httpx.Response(200, json={"letter_id": json.loads(request.content)["letter_id"]})
                          if request.url.path == "/api/letters" and request.method == "POST" else None)
    run_id = _call(ctx, place_id=PLACE_ID, text="no state in the answer")["run_id"]
    assert cards[-1][1]["state"] == "unknown" and "phase" not in cards[-1][1]
    receiver.letters.setdefault(run_id, {"state": "accepted_running"})
    receiver.letters[run_id].pop("state", None)
    delegate_tools._read_run(ctx, run_id, 0)
    assert cards[-1][1]["state"] == "unknown", "a read that names no state is not a working letter"


class _Entry:  # a ``peer`` custody record as observe_letter_run reads it
    run_id = "L1"
    engine_ref = {"kind": "peer", "place_id": PLACE_ID, "link_id": "link-0001"}


@pytest.mark.parametrize("carrier", ["live", "retained"])
def test_the_end_shows_a_reply_only_on_the_receivers_explicit_word(ctx, cards, carrier):  # noqa: F811
    """The carriers as the core writes them (live ``finalSummary.result_available`` with ``peerCost``, retained
    ``final_summary.result_available`` with ``cost.receiver_cost``): given, none and absent, and a spend not final."""
    def end(available=..., cost=None):
        cost = cost or {"known_usd": 0.3, "final": False}
        final = {"cost": cost, **({} if available is ... else {"result_available": available})}
        if carrier == "live":
            return {"detail": {"summary": {"runId": "L1", "state": "succeeded", "peerState": "done"},
                               "primaryOutput": {"text": "Here is your count: 3.", "truncated": False},
                               "finalSummary": final, "peerCost": cost}}
        return {"payload": {"status": "terminal", "state": "succeeded", "final_summary": final,
                            "primary_output": {"text": "Here is your count: 3.", "truncated": False},
                            "cost": {"billed_by": "peer", "receiver_cost": cost}}}

    for available in (True, False, ..., "yes"):
        letters._PUBLISHED.clear()
        letters.observe_letter_run(ctx, _Entry, **end(available))
    given, none, absent, loose = (fact for _text, fact in cards[-4:])
    assert (given["reply"], given["answer"], given["spent_usd"], given["spent_final"]) == ("given", "Here is your count: 3.", 0.3, False)
    assert (none["reply"], "answer" in none) == ("none", False)
    assert (absent["reply"], "answer" in absent) == ("unknown", False), "an older version states nothing: unknown"
    assert (loose["reply"], "answer" in loose) == ("unknown", False), "only a boolean is the receiver's word"
    letters._PUBLISHED.clear()
    letters.observe_letter_run(ctx, _Entry, **end(True, {"known_usd": 0.5, "final": True}))
    assert (cards[-1][1]["spent_usd"], cards[-1][1]["spent_final"]) == (0.5, True)
    letters.observe_letter_run(ctx, type("Owned", (), {"run_id": "R", "engine_ref": {}}), **end(True))
    assert len(cards) == 5, "a run of the owned engine is no letter: no card"


def test_long_words_are_absent_never_cut_and_an_unknown_spend_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(letters, "_peer_names", lambda root, place_id: ("Velm", "install-b"))
    long = "x" * (letters.CARD_TEXT_MAX + 1)
    fact = letters.letter_card_fact(tmp_path, "L1", state="done", letter={"text": long, "cost_ceiling_usd": True},
                                    place_id=PLACE_ID, reply="given", answer=long, spent_usd=None, spent_final=True)
    assert (fact["text"], fact["answer"], fact["spent_usd"], fact["ceiling_usd"]) == (None, None, None, None)
    assert fact["spent_final"] is False, "an unknown spend is never a final total"
    assert letters.letter_card_fact(tmp_path, "L1", state="no-such", letter={}, place_id=PLACE_ID)["state"] == "unknown"


def test_history_replays_the_card_with_its_row():
    from ouroboros.gateway.history import _PROGRESS_META_FIELDS

    assert "peer_letter_card" in _PROGRESS_META_FIELDS
    # The receiving side's task metadata keeps its own name (``peer_letter``); a card never rides under it.
    assert "peer_letter" not in _PROGRESS_META_FIELDS
    assert json.loads(json.dumps(letters._STATE_WORDS)), "every state has plain words for text-only readers"
    assert set(letters._STATE_WORDS) == set(letters._CARD_STATES)


def test_what_the_receiver_holds_now_is_never_a_working_letter_or_an_empty_answer(ctx, cards):  # noqa: F811
    """The read's ``summary.peerHold`` as ``gateways/peer.get_run`` writes it: a question whose words are held carries
    no words and its code; a done letter's answer held now is ``held`` with its code, a passing hold too (its run then
    still reads running), and ``rules_changed`` ends with ``finalSummary.result_held``."""
    def read(summary, **rest):
        letters._PUBLISHED.clear()
        letters.observe_letter_run(ctx, _Entry, detail={"summary": {"runId": "L1", **summary}, "pendingInteractions": [], **rest})
        return cards[-1][1]

    asked = read({"state": "running", "peerState": "needs_input",
                  "peerHold": {"what": "question", "reason": "peer_silent", "question_id": "q7", "note": "…"}})
    assert asked["state"] == "needs_input" and asked["question"] == {"question_id": "q7", "text": None, "held": "peer_silent"}
    passing = read({"state": "running", "peerState": "done", "peerHold": {"what": "result", "reason": "peer_silent"}})
    assert (passing["state"], passing["reply"], passing["held"], "answer" in passing) == ("done", "held", "peer_silent", False)
    changed = read({"state": "succeeded", "peerState": "done", "peerHold": {"what": "result", "reason": "rules_changed"}},
                   finalSummary={"result_available": False, "result_held": "rules_changed"})
    assert (changed["reply"], changed["held"]) == ("held", "rules_changed")
    plain = read({"state": "succeeded", "peerState": "done"}, finalSummary={"result_available": False})
    assert (plain["reply"], "held" in plain) == ("none", False), "a stated none stays none"


def test_long_words_carry_their_length_and_a_cut_reason_is_marked(tmp_path, monkeypatch):
    """Past the card's bound the words are null beside their length (the reader serves them whole); a failure reason
    past it keeps its first part beside its length, and a short reason carries no mark."""
    monkeypatch.setattr(letters, "_peer_names", lambda root, place_id: ("Velm", "install-b"))
    long = "x" * (letters.CARD_TEXT_MAX + 1)
    done = letters.letter_card_fact(tmp_path, "L1", state="done", letter={"text": long}, place_id=PLACE_ID,
                                    reply="given", answer=long)
    assert (done["text"], done["text_chars"], done["answer"], done["answer_chars"]) == (None, len(long), None, len(long))
    failed = letters.letter_card_fact(tmp_path, "L2", state="failed", letter={}, place_id=PLACE_ID, error=long)
    assert failed["error"] == long[:letters.CARD_TEXT_MAX] and failed["error_chars"] == len(long)
    short = letters.letter_card_fact(tmp_path, "L3", state="failed", letter={}, place_id=PLACE_ID, error="boom")
    assert short["error"] == "boom" and "error_chars" not in short


def test_a_reserved_answer_and_an_unconfirmed_stop_are_the_cards_own_words(ctx, cards):  # noqa: F811
    """A question whose answer the receiver reserved says so until its receipt confirms delivery; a Stop its read-back
    did not confirm keeps the state that read-back saw beside what is known of the Stop, never cancelled."""
    letters._PUBLISHED.clear()
    letters.observe_letter_run(ctx, _Entry, detail={
        "summary": {"runId": "L1", "state": "running", "peerState": "needs_input"},
        "pendingInteractions": [{"deliveryState": "reserved", "questions": [{"id": "q1", "question": "Which?"}]}]})
    assert cards[-1][1]["question"] == {"question_id": "q1", "text": "Which?", "delivery_state": "reserved"}
    for outcome, state, stop in (({"outcome": "requested", "state": "running"}, "accepted_running", "requested"),
                                 ({"outcome": "failed", "state": "running"}, "accepted_running", "refused"),
                                 ({"outcome": "containment_fault_run_may_still_be_live", "state": "running"},
                                  "accepted_running", "unverified"),
                                 ({"outcome": "confirmed", "state": "cancelled"}, "cancelled", None)):
        letters._PUBLISHED.clear()
        letters.observe_letter_stop(ctx, _Entry, outcome)
        fact = cards[-1][1]
        assert (fact["state"], fact.get("stop")) == (state, stop)
    letters._PUBLISHED.clear()
    letters.observe_letter_stop(ctx, _Entry, {"outcome": "requested", "state": "succeeded"})
    assert (cards[-1][1]["state"], cards[-1][1].get("stop")) == ("done", "requested"), \
        "a Stop over an end already reached never claims a cancellation"


def test_a_retained_end_restores_its_failure_reason(ctx, cards):  # noqa: F811
    """On replay the retained end is the failure reason's own record: its summary, else its final summary; a reason it
    does not carry stays unstated (no second journal)."""
    letters._PUBLISHED.clear()
    letters.observe_letter_run(ctx, _Entry, payload={"state": "failed", "summary": {"error": "the receiver ran out of room"},
                                                     "final_summary": {}, "primary_output": {}})
    assert cards[-1][1]["error"] == "the receiver ran out of room"
    letters._PUBLISHED.clear()
    letters.observe_letter_run(ctx, _Entry, payload={"state": "failed", "final_summary": {"error": "from the final"}})
    assert cards[-1][1]["error"] == "from the final"
    letters._PUBLISHED.clear()
    letters.observe_letter_run(ctx, _Entry, payload={"state": "failed", "final_summary": {}})
    assert "error" not in cards[-1][1]


def test_a_letter_only_page_carries_the_parents_facts_on_a_synthetic_carrier(tmp_path):
    """A history page whose only progress rows of a task are letter cards still carries that task's own replay facts:
    the synthetic evidence carrier stands in for the absent ordinary progress, and the letter rows carry none of it."""
    from ouroboros.gateway.history import _annotate_terminal_task_truth

    fact = {"letter_id": "L1", "place_id": PLACE_ID, "to": "Velm", "to_seed": "b", "text": "hi", "title": "",
            "ceiling_usd": None, "spent_usd": None, "spent_final": False, "state": "accepted_running"}
    letter_row = {"task_id": "p", "ts": "2026-11-25T00:00:00+00:00", "role": "system", "is_progress": True,
                  "text": "Letter to Velm: sent, at work", "peer_letter_card": fact, "history_id": "h1"}
    combined = [letter_row]
    _annotate_terminal_task_truth(combined, tmp_path, {"p": {"status": "running", "review_overrides": {"required": True}}})
    carriers = [m for m in combined if m.get("system_type") == "task_evidence"]
    assert len(carriers) == 1 and carriers[0]["task_id"] == "p"
    assert carriers[0].get("review_overrides") == {"required": True}
    assert "peer_letter_card" not in carriers[0] and "review_overrides" not in letter_row
