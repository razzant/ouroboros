"""Ordinary peer messages (``POST /api/letters`` with ``kind="message"`` and ``message_ouroboros``).

Covers:
  - claim_1: Consciousness Off -> a verified peer message (even with ``may_start=False``) is saved and visible in the
    receiver's room with ``attention: heard/off`` and ``owner_confirmed: false``; NO model turn and NO task.
  - claim_2: Consciousness On -> the Host's one Presence intake (``host_service_places.presence_intake``, version 2):
    place attention, wakes, allowance and the Host's reservation; the answer waits for the turn's typed start, and a
    typed retry (synchronous or asynchronous) is resent as the same letter; never attributed to the owner; an approved
    reply leaves at its turn's settlement (released speech while it is parked), a handed-off work's result at that
    work's terminal, each once over the saved link with a canonical v1 receipt (``presence_delivery``) and without a
    ping-pong loop; a lost acknowledgement sends the same letter again, a missed end stays owed until reported (no
    history sweep), and words a tool already sent never leave twice.
  - Wire & idempotency invariants: one ``letter_id`` = one outcome across work letters, question answers, and
    standalone messages; strict field rejection; pause/revoke/unconfirmed/mine refusals before capture.
"""
from __future__ import annotations

import contextlib
import json
import time
from types import SimpleNamespace

import pytest

from tests._peer_letters_shared import introduce, letter, receiver_client, safety_reads


def _msg(**fields) -> dict:
    return letter(kind="message", text=fields.pop("text", "Epoch 4 finished on train-1; loss=0.142."), **fields)


def _enable_consciousness(root, monkeypatch, *, enabled: bool = True, autonomy: str = "act",
                          allowance_usd: float = 5.0) -> None:
    from supervisor import state as st

    prev_drive, prev_path = st.DRIVE_ROOT, st.STATE_PATH
    try:
        st.init(root, root)
        st.save_state(st.ensure_state_defaults({"bg_consciousness_enabled": bool(enabled)}))
    finally:
        st.DRIVE_ROOT, st.STATE_PATH = prev_drive, prev_path
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", autonomy)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", str(allowance_usd))


def _task_ids(root) -> list[str]:
    from ouroboros import store

    if not (root / store.STORE_REL).exists():
        return []
    with store.open_store(root, write=False) as txn:
        return [str(r[0]) for r in txn.conn.execute("SELECT task_id FROM tasks").fetchall()]


def _room_chat_id(root, place_id: str) -> int:
    from ouroboros.gateway.peer_letters_api import _room
    from ouroboros.places import get_place

    return _room(get_place(root, place_id))


def _early_answer(execution) -> None:
    """A carrier's version-2 early answer: the author's RUNNING row is durable and its loop takes mail; the turn runs on
    with its reservation (``presence_continuation.answer_early``)."""
    from ouroboros.presence_runner import PresenceTurnResult

    execution.publish_initial(PresenceTurnResult(outcome="deferred", text="", task_id=execution.turn_id,
                                                 status="continuing", continuation_ref=execution.turn_id,
                                                 continuation_version=2), return_capacity=False)


def _poll_until(predicate, *, timeout: float = 10.0, label: str = "condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {label}: last={last!r}")


# ---------------------------------------------------------------------------
# claim_1: Consciousness Off — saved and visible, zero tasks, zero model turns
# ---------------------------------------------------------------------------


def test_claim_1_consciousness_off_saves_message_without_may_start_task_or_model_turn(tmp_path, monkeypatch):
    """Consciousness Off + may_start=False: peer message is captured in the room, visible in /rooms and /history
    with attention=heard/off and owner_confirmed=False; creates zero tasks and dispatches zero model turns."""
    from ouroboros import presence_carrier

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    dispatched = []
    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", lambda *a, **kw: dispatched.append((a, kw)))
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)

    body = _msg(letter_id="note-off-1", text="Training checkpoint 4 saved at /models/ckpt4.")
    with receiver_client(pair, monkeypatch) as client:
        # Work letter is still refused when may_start=False:
        work_resp = client.post("/api/letters", json=letter(letter_id="work-refused-1"), headers=pair.headers())
        assert work_resp.status_code == 403 and work_resp.json()["refusal_code"] == "peer_may_not_start"

        # Standalone message succeeds even when may_start=False:
        resp = client.post("/api/letters", json=body, headers=pair.headers())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"letter_id": "note-off-1", "kind": "message", "state": "received"}

        # Not a work assignment: GET status/result/cancel report letter_not_found:
        for method, path in (("GET", "/api/letters/note-off-1"),
                             ("GET", "/api/letters/note-off-1/result"),
                             ("POST", "/api/letters/note-off-1/cancel")):
            r = client.request(method, path, headers=pair.headers())
            assert r.status_code == 404 and r.json()["refusal_code"] == "letter_not_found", (path, r.text)

        # Visible in GET /api/places/{place_id}/rooms:
        rooms_resp = client.get(f"/api/places/{pair.place_b}/rooms")
        assert rooms_resp.status_code == 200, rooms_resp.text
        rooms = rooms_resp.json()["rooms"]
        assert len(rooms) == 1
        chat_id = rooms[0]["room_chat_id"]
        assert chat_id == _room_chat_id(pair.root_b, pair.place_b)
        assert rooms[0]["label"] == "Laptop"
        assert rooms[0]["unanswered_count"] in (0, None)

        # Visible in GET /api/chat/history?chat_id=<room>:
        hist_resp = client.get("/api/chat/history", params={"chat_id": chat_id})
        assert hist_resp.status_code == 200, hist_resp.text
        rows = hist_resp.json()["messages"]
        assert len(rows) == 1
        row = rows[0]
        assert row["text"] == "Training checkpoint 4 saved at /models/ckpt4."
        assert row["sender_label"] == "Laptop"
        assert row["place_history"]["kind"] == "message"
        assert row["place_history"]["owner_confirmed"] is False
        assert row["place_history"]["attention"] == {"decision": "heard", "reason": "off"}

        # Peer is never suggested as an owner account candidate:
        places_resp = client.get("/api/places")
        place_row = next(p for p in places_resp.json()["places"] if p["place_id"] == pair.place_b)
        assert place_row["account_candidates"] == []

    assert dispatched == [], "Consciousness Off must dispatch zero focus turns"
    assert _task_ids(pair.root_b) == [], "Consciousness Off must create zero task rows"


def test_peer_message_idempotency_and_cross_kind_letter_id_uniqueness(tmp_path, monkeypatch):
    """One letter_id = one outcome across standalone messages, work letters, and question answers."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=True)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)

    body = _msg(letter_id="shared-id-1", text="Same note twice.")
    with receiver_client(pair, monkeypatch) as client:
        assert client.post("/api/letters", json=body, headers=pair.headers()).status_code == 200
        # Same id + different text -> 409 letter_id_reused:
        changed = client.post("/api/letters", json=_msg(letter_id="shared-id-1", text="Mutated text."),
                              headers=pair.headers())
        assert changed.status_code == 409 and changed.json()["refusal_code"] == "letter_id_reused"
        # Same id used as a work letter -> 409 letter_id_reused:
        as_work = client.post("/api/letters", json=letter(letter_id="shared-id-1", text="Start work"),
                              headers=pair.headers())
        assert as_work.status_code == 409 and as_work.json()["refusal_code"] == "letter_id_reused"
        # Same id + identical bytes repeats idempotently even after the link is Paused:
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        repeat = client.post("/api/letters", json=body, headers=pair.headers())
        assert repeat.status_code == 200 and repeat.json()["state"] == "received"
        # History still contains exactly one arrival row (no duplicate):
        chat_id = _room_chat_id(pair.root_b, pair.place_b)
        assert len(client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]) == 1

        # Reverse collision: a work letter id cannot later be reused for a message:
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="active")
        assert client.post("/api/letters", json=letter(letter_id="work-id-1"), headers=pair.headers()).status_code == 200
        as_msg = client.post("/api/letters", json=_msg(letter_id="work-id-1", text="Collision"),
                             headers=pair.headers())
        assert as_msg.status_code == 409 and as_msg.json()["refusal_code"] == "letter_id_reused"


def test_peer_message_wire_and_link_refusals(tmp_path, monkeypatch):
    """Strict wire fields, unconfirmed code, paused link, and non-mine refusal happen before any capture."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, confirm=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)

    from ouroboros.install_identity import recorded_id

    raw_headers = {
        "X-Ouroboros-Link": f"{pair.link_id}.{pair.secret_a()}",
        "X-Ouroboros-Install": recorded_id(pair.root_a),
        "X-Ouroboros-Protocol": "1",
    }

    with receiver_client(pair, monkeypatch) as client:
        # Unconfirmed code -> 403 link_paused (code_check_pending):
        r = client.post("/api/letters", json=_msg(letter_id="unconf-1"), headers=raw_headers)
        assert r.status_code == 403 and r.json()["refusal_code"] == "link_paused"
        assert r.json().get("detail") == "code_check_pending"

        # Invalid field combinations:
        for bad in (
            _msg(letter_id="bad-1", title="No titles on messages"),
            _msg(letter_id="bad-2", cost_ceiling_usd=1.0),
            _msg(letter_id="bad-3", chain=[]),
            _msg(letter_id="bad-4", in_reply_to={"letter_id": "x", "question_id": "q1"}),
            letter(letter_id="bad-5", reply_to="m1"),
            letter(letter_id="bad-6", kind="unknown"),
        ):
            res = client.post("/api/letters", json=bad, headers=raw_headers)
            assert res.status_code == 400, (bad, res.text)

        # Confirm both sides, then pause B's place -> 403 link_paused:
        from ouroboros import peer_links
        for root in (pair.root_a, pair.root_b):
            with acting_as(root):
                peer_links.confirm(root, pair.link_id)
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        paused = client.post("/api/letters", json=_msg(letter_id="paused-1"), headers=pair.headers())
        assert paused.status_code == 403 and paused.json()["refusal_code"] == "link_paused"

        # Active but mine=False -> 403 peer_may_not_start:
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="active")
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)
        not_mine = client.post("/api/letters", json=_msg(letter_id="notmine-1"), headers=pair.headers())
        assert not_mine.status_code == 403 and not_mine.json()["refusal_code"] == "peer_may_not_start"

        # Nothing was captured into rooms:
        assert client.get(f"/api/places/{pair.place_b}/rooms").json()["rooms"] == []


# ---------------------------------------------------------------------------
# claim_2: Consciousness On — place attention, wakes, non-owner, holds, Safety, reply back
# ---------------------------------------------------------------------------


def test_claim_2_peer_cannot_be_confirmed_as_owner_and_wakes_follow_place_rules(tmp_path, monkeypatch):
    """Consciousness On: wakes match place.wakes (direct_message + from_bot); clearing wakes or exhausting
    allowance leaves the message heard without starting a model turn; peer can never be confirmed as owner."""
    from ouroboros import places, presence_carrier
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched = []
    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                        lambda execution, **kw: (dispatched.append(((execution,), kw)), _early_answer(execution)))

    # Linked Ouroboros can never be confirmed as an owner account:
    with acting_as(pair.root_b):
        with pytest.raises(places.PlaceError, match="never an owner account"):
            places.edit_place(pair.root_b, pair.place_b, "confirm_owner_account", author=places.AUTHOR_OWNER,
                              platform_actor_id=f"peer:{pair.link_id}", label="Laptop")
        # Even if a stored row somehow held a matching actor id, owner_author still returns False for kind=ouroboros:
        fake_accounts = {pair.place_b: (places.OwnerAccount(f"peer:{pair.link_id}", "Laptop", "", ""),)}
        assert places.owner_author(fake_accounts, pair.place_b,
                                   {"platform_actor_id": f"peer:{pair.link_id}", "authenticated": True,
                                    "kind": "ouroboros", "is_bot": True},
                                   pair.root_a.name) is False

    with receiver_client(pair, monkeypatch) as client:
        # 1. Default wakes (direct_message, from_bot) + Consciousness On -> dispatches focus!
        r1 = client.post("/api/letters", json=_msg(letter_id="wake-1", text="Hello while On"), headers=pair.headers())
        assert r1.status_code == 200
        assert len(dispatched) == 1
        admitted_event = dispatched[0][1]["event"]
        assert admitted_event.actor["kind"] == "ouroboros" and admitted_event.actor["is_bot"] is True

        # 2. Owner clears wakes -> heard/no_wake_match, no dispatch!
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_wakes", author=places.AUTHOR_OWNER, wakes=[])
        r2 = client.post("/api/letters", json=_msg(letter_id="wake-muted-1", text="Muted note"), headers=pair.headers())
        assert r2.status_code == 200
        assert len(dispatched) == 1  # unchanged!
        chat_id = _room_chat_id(pair.root_b, pair.place_b)
        rows = client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]
        muted = next(m for m in rows if m["text"] == "Muted note")
        assert muted["place_history"]["attention"] == {"decision": "heard", "reason": "no_wake_match"}
        assert muted["place_history"]["owner_confirmed"] is False

        # 3. Restore wakes, exhaust allowance -> the Host's typed retry (as /presence/turn answers a connector): the
        # room saved it, no decision and no dispatch; the same letter is resent once the allowance is back.
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_wakes", author=places.AUTHOR_OWNER,
                              wakes=["direct_message", "from_bot"])
        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=0.0)
        r3 = client.post("/api/letters", json=_msg(letter_id="wake-no-budget-1", text="No allowance"),
                         headers=pair.headers())
        assert r3.status_code == 429, r3.text
        assert r3.json()["refusal_code"] == "consciousness_allowance_exhausted" and r3.json()["retry_after"] >= 1
        assert len(dispatched) == 1
        rows = client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]
        exhausted = next(m for m in rows if m["text"] == "No allowance")
        assert "attention" not in exhausted["place_history"]
        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
        again = client.post("/api/letters", json=_msg(letter_id="wake-no-budget-1", text="No allowance"),
                            headers=pair.headers())
        assert again.status_code == 200 and again.json()["state"] == "received"
        assert len(dispatched) == 2  # the resent letter met the same intake once; its room row stays one
        rows = client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]
        assert [m["text"] for m in rows].count("No allowance") == 1


def test_claim_2_silent_advises_and_safety_govern_peer_focus_replies(tmp_path, monkeypatch):
    """A peer focus turn crosses the exact same place_reply Silent/Advises/Safety boundary as every Place."""
    from ouroboros import place_reply, places
    from ouroboros.gateway.peer_letters_api import _peer_message_event_payload
    from ouroboros.install_identity import acting_as
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_events import presence_turn_event
    from ouroboros.presence_runner import _build_task

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, note="Never disclose local secrets to Laptop.")
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)

    def _ctx(state: str = "active", autonomy: str = "act"):
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "set_state", author=places.AUTHOR_OWNER, state=state)
            _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy=autonomy, allowance_usd=5.0)
            place = places.get_place(pair.root_b, pair.place_b)
            caller = {"link_id": pair.link_id, "place_id": pair.place_b, "peer_label": "Laptop",
                      "peer_install_id": place.account_id}
            payload = _peer_message_event_payload(place, caller, _msg(letter_id=f"m-{state}-{autonomy}"))
            admission = admit_presence_turn(
                drive_root=pair.root_b, authenticated_transport_skill="ouroboros", binding_id=pair.place_b,
                provider="ouroboros", account_id=place.account_id, conversation_id=place.conversation_id,
                thread_id="", live_event=True)
            event = presence_turn_event(payload, admission)
            task = _build_task(admission, event, drive_root=pair.root_b, staged_files=())
            assert task["metadata"]["focus"]["actor"]["owner_confirmed"] is False
            return SimpleNamespace(task_id=task["id"], task_metadata=task["metadata"],
                                   drive_root=pair.root_b, budget_drive_root=pair.root_b, task_contract={})

    # 1. Silent place -> reply held, outcome becomes silent, notice recorded with held_text:
    ctx_silent = _ctx(state="silent", autonomy="act")
    ok, msg = place_reply.prepare_focus_reply(ctx_silent, "Trying to reply while silent")
    assert ok and msg == ""
    assert ctx_silent._presence_completion["outcome"] == "silent"
    assert "Silent" in ctx_silent._focus_reply_hold

    # 2. Advises autonomy -> reply held, outcome becomes silent:
    ctx_advises = _ctx(state="active", autonomy="observe")
    ok, msg = place_reply.prepare_focus_reply(ctx_advises, "Trying to reply under Advises")
    assert ok and msg == ""
    assert ctx_advises._presence_completion["outcome"] == "silent"
    assert "Advises" in ctx_advises._focus_reply_hold

    # 3. Active + Acts -> Safety Supervisor checks place_reply (refusal blocks sending):
    reads = safety_reads(monkeypatch, (False, "Refused by Safety: place note forbids secret disclosure."))
    ctx_active = _ctx(state="active", autonomy="act")
    ok, msg = place_reply.prepare_focus_reply(ctx_active, "Here is the local secret.")
    assert ok is False and "Refused by Safety" in msg
    assert reads and reads[0][0] == "place_reply" and reads[0][1]["place_id"] == pair.place_b

    # 4. Active + Acts + Safety SAFE -> outcome == "message" admitted for delivery:
    safety_reads(monkeypatch, (True, ""))
    ok, msg = place_reply.prepare_focus_reply(ctx_active, "Checkpoint acknowledged.")
    assert ok and msg == ""
    assert ctx_active._presence_completion["outcome"] == "message"

    # 5. Only words a reading let go keep an approval, with the rules that reading met (``speech_basis``):
    import hashlib

    from ouroboros.peer_letters import speech_basis
    from ouroboros.peer_links import owed_record
    from ouroboros.safety_config import coverage_kind

    with acting_as(pair.root_b):
        approved = (owed_record(pair.root_b, pair.link_id, ctx_active.task_id) or {}).get("approved")
        basis = speech_basis(pair.root_b, pair.place_b, kind=coverage_kind(ctx_active.task_metadata))
        assert basis and approved == {hashlib.sha256(b"Checkpoint acknowledged.").hexdigest(): basis}, approved
        assert owed_record(pair.root_b, pair.link_id, ctx_silent.task_id) is None, "held words keep no approval"
        assert owed_record(pair.root_b, pair.link_id, ctx_advises.task_id) is None

    # 6. A rule my human changes while the reading runs: the reading met the older rules, so its words keep no
    # approval (held at their send), never the newer rules' stamp.
    def edit_during_reading(name, args, **kw):
        with acting_as(pair.root_b):
            places.edit_place(pair.root_b, pair.place_b, "add_note", author=places.AUTHOR_OWNER,
                              text="Nothing about checkpoints.")
        return True, ""

    monkeypatch.setattr("ouroboros.safety.check_safety", edit_during_reading)
    ctx_edited = _ctx(state="active", autonomy="act")
    assert place_reply.prepare_focus_reply(ctx_edited, "Checkpoint four kept.") == (True, "")
    with acting_as(pair.root_b):
        approved = (owed_record(pair.root_b, pair.link_id, ctx_edited.task_id) or {}).get("approved") or {}
        assert approved.get(hashlib.sha256(b"Checkpoint four kept.").hexdigest()) == "", approved


# ---------------------------------------------------------------------------
# claim_2: an arrival's typed start, and replies from each producer's own end
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _both_sides(pair, monkeypatch):
    """B's and A's real gateways behind their auth gates; a PeerGateway aimed at a side's address reaches its client."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.router import collect_routes
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as
    from ouroboros.server_auth import NetworkAuthGate

    app_a = Starlette(routes=collect_routes(data_dir=pair.root_a))
    app_a.state.drive_root, app_a.state.repo_dir = pair.root_a, pair.repo
    gate_a = NetworkAuthGate(app_a)

    async def as_a(scope, receive, send):
        with acting_as(pair.root_a):
            await gate_a(scope, receive, send)

    with receiver_client(pair, monkeypatch) as client_b, TestClient(
            as_a, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client_a:
        clients = {"http://127.0.0.1:8": client_a, "http://127.0.0.1:9": client_b}
        real_init = peer_gw.PeerGateway.__init__

        def routed(self, link, *, timeout_sec=None):
            real_init(self, link, timeout_sec=timeout_sec)
            self._client.close()
            self._client = clients[str(link.peer_url).rstrip("/")]

        monkeypatch.setattr(peer_gw.PeerGateway, "__init__", routed)
        monkeypatch.setattr(peer_gw.PeerGateway, "close", lambda self: None)  # the shared clients outlive a send
        try:
            yield client_a, client_b
        finally:
            _poll_until(_replies_idle, label="reply threads ended")


def _replies_idle() -> bool:
    """No short-lived reply thread (``peer_replies.deliver_owed``) is still sending."""
    import threading

    return not [thread for thread in threading.enumerate() if thread.name.startswith("peer-repl")]


def _settled_frame(pair, turn_id: str, result: dict) -> None:
    """The Host's own end of a carried turn, as its carrier's frame reaches the supervisor."""
    from ouroboros.presence_carrier import handle_presence_turn_settled

    handle_presence_turn_settled({"type": "presence_turn_settled", "turn_id": turn_id, "task_id": turn_id,
                                  "result": result}, SimpleNamespace(RUNNING={}, WORKERS={}, DRIVE_ROOT=pair.root_b))


def _approve(pair, task_id: str, words: str, metadata: dict) -> None:
    """What B's selection records when its reading lets ``words`` go to A (``place_reply.prepare_focus_reply`` ->
    ``peer_letters.note_reply_approved``): the rules they were read under, on the work's owed record."""
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import note_reply_approved, reply_basis
    from ouroboros.places import place_records

    with acting_as(pair.root_b):
        place = place_records(pair.root_b).places[pair.place_b]
        basis = reply_basis(pair.root_b, place, metadata)
        note_reply_approved(pair.root_b, place, task_id, words, basis, basis)


def _settle_turns(pair, monkeypatch, replies: dict, dispatched: list, *, before_end=None) -> None:
    """B's carrier stand-in: each admitted turn records its model final as the runner does (``presence_result`` facts
    on its own row, its selection's approval: ``_approve``) and ends through the Host's settlement frame;
    ``replies[text]`` is ``(outcome, words, work_ref)`` or ``(outcome, words, work_ref, anchor_message_id)``.
    ``before_end(task, words)`` runs inside the turn."""
    from ouroboros import presence_carrier
    from ouroboros.presence_continuation import output_ref
    from ouroboros.presence_runner import _build_task
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    def carrier(execution, **kwargs):
        dispatched.append((str(kwargs["drive_root"]), kwargs["event"].text))
        if str(kwargs["drive_root"]) != str(pair.root_b):
            return
        task = _build_task(kwargs["admission"], kwargs["event"], drive_root=pair.root_b, staged_files=())
        outcome, words, work_ref, *anchor = replies[kwargs["event"].text]
        write_task_result(pair.root_b, task["id"], STATUS_RUNNING, metadata=task["metadata"], chat_id=task["chat_id"],
                          source=task["source"], result="Task is running.")  # its start record, as the runner's
        if before_end is not None:
            before_end(task, words)
        if words:
            _approve(pair, task["id"], words, task["metadata"])
        ref = output_ref(task["id"], "terminal", kwargs["event"].conversation_key) if words else ""
        write_task_result(pair.root_b, task["id"], STATUS_COMPLETED, metadata={
            **task["metadata"], "presence_outcome": outcome, "presence_result_text": words,
            **({"presence_output_ref": ref} if ref else {}), **({"presence_work_ref": work_ref} if work_ref else {}),
            **({"presence_reply_anchor": {"source_event_id": f"peer-msg:{anchor[0]}",
                                          "message": {"message_id": anchor[0]}}} if anchor else {})},
            terminal_origin="model_final", result=words or "quiet", chat_id=task["chat_id"], source=task["source"])
        _settled_frame(pair, task["id"], {"outcome": outcome, "text": words, "task_id": task["id"],
                                          "work_ref": work_ref, "output_ref": ref})

    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", carrier)


def _room(client, root, place_id) -> list:
    return client.get("/api/chat/history", params={"chat_id": _room_chat_id(root, place_id)}).json()["messages"]


def _receipts(root) -> list:
    from ouroboros import dialogue_store

    return [r.row for r in dialogue_store.newest(root, "chat", limit=200) if r.row.get("type") == "presence_delivery"]


def _arrival(root, text: str) -> dict:
    from ouroboros import dialogue_store

    return next(r.row for r in dialogue_store.newest(root, "chat", limit=200)
                if r.row.get("direction") == "in" and r.row.get("text") == text)


def test_claim_2_reply_and_handoff_result_leave_once_with_receipts_and_no_ping_pong(tmp_path, monkeypatch):
    """B wakes on A's message through the Host's intake; at the turn's settlement its approved reply leaves over the
    saved link once, addressed to the input its reply anchors, and B's room shows it Delivered (a v1 receipt, never an
    authored row); A hears it as context (no ping-pong, both On). A turn that hands off work answers at that work's own
    terminal, once; the same end reported again sends nothing."""
    from ouroboros.gateway.host_service_places import host_context
    from ouroboros.peer_replies import deliver_owed
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched: list = []
    reply = "Velm: received, keeping checkpoint 4."
    _settle_turns(pair, monkeypatch, {"Status check from Laptop.": ("message", reply, "", "ping-1b"),
                                      "Please rerun eval.": ("deferred", "", "handoff-eval-1")}, dispatched)
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        resp = client_b.post("/api/letters", json=_msg(letter_id="ping-1", text="Status check from Laptop."),
                             headers=pair.headers())
        assert resp.status_code == 200 and resp.json()["state"] == "received"
        assert host_context(pair.root_b).presence_turns.live() == []  # the Host's own registry carried and settled it
        rows_a = _poll_until(lambda: [m for m in _room(client_a, pair.root_a, pair.place_a)
                                      if m["text"] == reply and m["place_history"].get("attention")],
                             label="A received and heard B's reply")
        assert rows_a[0]["sender_label"] == "Velm" and rows_a[0]["place_history"]["owner_confirmed"] is False
        assert rows_a[0]["place_history"]["attention"] == {"decision": "heard", "reason": "peer_reply"}
        assert _arrival(pair.root_a, reply)["transport"]["message"]["reply_to"] == "ping-1b"  # its anchored input
        replies_b = _poll_until(lambda: [m for m in _room(client_b, pair.root_b, pair.place_b) if m["text"] == reply],
                                label="B recorded its receipt")  # written once A's answer named the letter
        assert [m["place_history"]["delivery_state"] for m in replies_b] == ["delivered"], replies_b
        assert replies_b[0]["role"] == "assistant"

        # A handed-off work: the turn defers; the work (it answers the same conversation) answers at its terminal.
        resp = client_b.post("/api/letters", json=_msg(letter_id="ask-2", text="Please rerun eval."),
                             headers=pair.headers())
        assert resp.status_code == 200
        with acting_as(pair.root_b):
            producer = load_task_result(pair.root_b, _arrival(pair.root_b, "Please rerun eval.")["task_id"])
            _approve(pair, "handoff-eval-1", "Eval rerun: 0.91.", {"presence": producer["metadata"]["presence"]})
            write_task_result(pair.root_b, "handoff-eval-1", STATUS_COMPLETED, result="Eval rerun: 0.91.",
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": producer["metadata"]["presence"]})
            work = load_task_result(pair.root_b, "handoff-eval-1")
        deliver_owed(pair.root_b, "handoff-eval-1", record=work)  # the work's terminal (events_task_done)
        _poll_until(lambda: [m for m in _room(client_a, pair.root_a, pair.place_a) if m["text"] == "Eval rerun: 0.91."],
                    label="A received the handed-off result")
        _poll_until(_replies_idle, label="reply threads ended")
        deliver_owed(pair.root_b, "handoff-eval-1", record=work)  # the same ends reported again
        first = _arrival(pair.root_b, "Status check from Laptop.")["task_id"]
        _settled_frame(pair, first, {"outcome": "message", "text": reply, "task_id": first})
        _poll_until(_replies_idle, label="reply threads ended")
        texts_a = [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        assert texts_a.count(reply) == 1 and texts_a.count("Eval rerun: 0.91.") == 1
    states = sorted((r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b))
    assert states == [("Eval rerun: 0.91.", "delivered"), (reply, "delivered")]
    assert [root for root, _text in dispatched] == [str(pair.root_b)] * 2, "A never started a turn: no ping-pong"


def test_claim_2_lost_acknowledgement_and_a_reported_end_send_the_same_letter_never_a_new_turn(tmp_path, monkeypatch):
    """B's reply whose confirmation is lost goes again as the same letter (A keeps one row). A turn whose end was never
    reported (a crash between its terminal and its delivery) owes its reply: another letter's turn does not sweep it
    (no history scan); the end reported again sends it once from its recorded turn, never a second model turn."""
    import ouroboros.peer_replies as replies
    from ouroboros.gateways import peer as peer_gw

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched: list = []
    _settle_turns(pair, monkeypatch, {"First.": ("message", "Reply one.", ""),
                                      "Second.": ("message", "Reply two.", ""),
                                      "Third.": ("message", "Reply three.", "")}, dispatched)
    real_request = peer_gw.PeerGateway._request
    lost = {"left": 1}

    def lose_first_answer(self, method, path, **kw):
        answer = real_request(self, method, path, **kw)  # A saved it ...
        if self._link.place_id == pair.place_b and lost["left"]:
            lost["left"] -= 1
            raise peer_gw._unavailable("peer_unreachable", "the answer was lost on the way back")  # ... B never heard
        return answer

    def texts_a():
        return [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]

    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        monkeypatch.setattr(peer_gw.PeerGateway, "_request", lose_first_answer)
        assert client_b.post("/api/letters", json=_msg(letter_id="m-1", text="First."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: "Reply one." in texts_a(), label="A received reply one")
        _poll_until(_replies_idle, label="reply threads ended")
        assert lost["left"] == 0
        real_deliver = replies.deliver_owed
        monkeypatch.setattr(replies, "deliver_owed", lambda *a, **k: None)  # its end is recorded, never reported
        assert client_b.post("/api/letters", json=_msg(letter_id="m-2", text="Second."),
                             headers=pair.headers()).status_code == 200
        monkeypatch.setattr(replies, "deliver_owed", real_deliver)
        assert client_b.post("/api/letters", json=_msg(letter_id="m-3", text="Third."),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: "Reply three." in texts_a(), label="A received reply three")
        _poll_until(_replies_idle, label="reply threads ended")
        assert "Reply two." not in texts_a()  # nothing swept it: it stays owed, with no receipt
        second = _arrival(pair.root_b, "Second.")["task_id"]
        replies.deliver_owed(pair.root_b, second)  # its end, reported again
        _poll_until(lambda: "Reply two." in texts_a(), label="A received reply two")
        _poll_until(_replies_idle, label="reply threads ended")
        replies.deliver_owed(pair.root_b, second)  # and again: its receipt settles it
        _poll_until(_replies_idle, label="reply threads ended")
        assert [texts_a().count(t) for t in ("Reply one.", "Reply two.", "Reply three.")] == [1, 1, 1]
    assert [text for _root, text in dispatched] == ["First.", "Second.", "Third."], "one model turn per message"
    assert sorted(r["transport"]["delivery"]["state"] for r in _receipts(pair.root_b)) == ["delivered"] * 3


def test_speech_released_while_the_turn_is_pending_leaves_then_and_never_again(tmp_path, monkeypatch):
    """A parked author's released output (the pending ``outputs`` of its retained work, as ``/presence/work`` shows a
    connector) leaves when that state is reported, before the turn ends; the terminal then sends only its own output."""
    from ouroboros.peer_replies import deliver_owed
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    interim, final = "Interim: 3 of 5 runs done.", "Final: all 5 runs agree."
    seen: dict = {}

    def park_with_interim(task, _words):
        _approve(pair, task["id"], interim, task["metadata"])  # its selection read before it was released
        write_task_result(pair.root_b, task["id"], STATUS_RUNNING, presence_continuation={
            "initial": {"outcome": "deferred", "text": ""}, "continuation_version": 2,
            "outputs": [{"output_ref": "presence-output-interim0000", "outcome": "message", "text": interim}]})
        deliver_owed(pair.root_b, task["id"])  # the park reported while the author waits
        seen["interim"] = _poll_until(lambda: interim in [m["text"] for m in _room(seen["a"], pair.root_a,
                                                                                    pair.place_a)],
                                      label="A received the released output while the turn ran")

    _settle_turns(pair, monkeypatch, {"Run all five?": ("message", final, "")}, [], before_end=park_with_interim)
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        seen["a"] = client_a
        assert client_b.post("/api/letters", json=_msg(letter_id="r-1", text="Run all five?"),
                             headers=pair.headers()).status_code == 200
        _poll_until(lambda: final in [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)],
                    label="A received the final")
        _poll_until(_replies_idle, label="reply threads ended")
        texts = [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)]
        assert seen["interim"] and texts.count(interim) == 1 and texts.count(final) == 1
    assert sorted(r["text"] for r in _receipts(pair.root_b)) == sorted([final, interim])


def test_arrival_answers_with_the_turns_typed_start_and_an_asynchronous_refusal_keeps_the_resend(tmp_path,
                                                                                                monkeypatch):
    """The answer waits for the turn's own typed start (version 2), as ``/presence/turn`` does: a carrier refusal that
    arrives after the dispatch is the sender's typed retry (503, never 200), the same letter later starts its one turn
    with no second room row, and an early answer published later still answers received while the turn runs on."""
    import threading
    import time

    from ouroboros import presence_carrier
    from ouroboros.presence_executions import PresenceTurnNotStarted

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    starts: list = []

    def refuse_later(execution, **kw):  # the carrier's refusal frame arrives after the dispatch returned
        starts.append(kw["event"].text)
        threading.Timer(0.2, lambda: _registry(execution).settle_from_frame(
            execution, error=PresenceTurnNotStarted("the worker pool of this server has not started yet"))).start()

    def answer_later(execution, **kw):
        starts.append(kw["event"].text)
        threading.Timer(0.3, _early_answer, args=(execution,)).start()

    with receiver_client(pair, monkeypatch) as client:
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", refuse_later)
        refused = client.post("/api/letters", json=_msg(letter_id="late-1", text="Async refusal"),
                              headers=pair.headers())
        assert refused.status_code == 503, refused.text
        assert refused.json()["refusal_code"] == "presence_turn_not_started"
        assert refused.json()["disposition"] == "retry"
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", answer_later)
        began = time.monotonic()
        taken = client.post("/api/letters", json=_msg(letter_id="late-1", text="Async refusal"), headers=pair.headers())
        assert taken.status_code == 200 and taken.json()["state"] == "received", taken.text
        assert time.monotonic() - began >= 0.3  # it answered the turn's start, not the dispatch
        from ouroboros.gateway.host_service_places import host_context
        assert len(host_context(pair.root_b).presence_turns.live()) == 1  # the turn runs on: nothing cancelled it
        assert starts == ["Async refusal", "Async refusal"]
        assert [m["text"] for m in _room(client, pair.root_b, pair.place_b)] == ["Async refusal"]
        def gate_busy(execution, **kw):  # version 2 never waits in a busy gate
            raise presence_carrier._capacity_full("conversation")

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", gate_busy)
        held = client.post("/api/letters", json=_msg(letter_id="busy-2", text="Busy conversation"),
                           headers=pair.headers())
        assert held.status_code == 429 and held.json()["refusal_code"] == "presence_capacity_full", held.text
        assert held.json()["retry_after"] == 1


def _registry(execution):
    from ouroboros.presence_executions import _find_execution

    return _find_execution(execution.turn_id)[0]


def test_an_output_the_producer_already_sent_by_tool_never_leaves_twice(tmp_path, monkeypatch):
    """A focus that sent its words with its own tool (a canonical tool receipt of that turn) and then finished with
    the same words leaves them once: the automatic output reads the tool's receipt and sends nothing more."""
    from ouroboros import peer_links
    from ouroboros.peer_replies import message_letter_id, send_peer_message

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False)
    _enable_consciousness(pair.root_a, monkeypatch, enabled=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    words = "Done: checkpoint 4 kept."

    def tool_speech(task, text):  # the tool's answer to the message that woke the turn (its reply anchor)
        link = peer_links.outgoing_link(pair.place_b, drive_root=pair.root_b)
        sent = send_peer_message(pair.root_b, pair.place_b, text=text, reply_to="q-1",
                                 letter_id=message_letter_id(link.link_id, task["id"], "q-1", text),
                                 origin={"kind": "tool", "task_id": task["id"]})
        assert sent["state"] == "delivered" and sent["recorded"], sent

    _settle_turns(pair, monkeypatch, {"Keep ckpt4?": ("message", words, "")}, [], before_end=tool_speech)
    with _both_sides(pair, monkeypatch) as (client_a, client_b):
        assert client_b.post("/api/letters", json=_msg(letter_id="q-1", text="Keep ckpt4?"),
                             headers=pair.headers()).status_code == 200
        _poll_until(_replies_idle, label="reply threads ended")
        assert [m["text"] for m in _room(client_a, pair.root_a, pair.place_a)].count(words) == 1
    assert [(r["transport"]["origin"]["kind"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_b)] \
        == [("tool", "delivered")]


def test_receiver_capacity_is_the_hosts_typed_retry_and_the_same_letter_is_taken_later(tmp_path, monkeypatch):
    """The Host's own reservation bounds peer turns: a full one answers 429 presence_capacity_full (saved, nothing
    admitted); the same letter resent after a slot returns is admitted once."""
    from ouroboros import presence_carrier
    from ouroboros.gateway.host_service_places import host_context

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched = []
    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                        lambda execution, **kw: (dispatched.append(kw["event"]), _early_answer(execution)))
    ctx = host_context(pair.root_b)
    held = [ctx._enter_inflight("ouroboros:presence") for _ in range(5)]
    try:
        with receiver_client(pair, monkeypatch) as client:
            full = client.post("/api/letters", json=_msg(letter_id="busy-1", text="While busy"),
                               headers=pair.headers())
            assert full.status_code == 429 and full.json()["refusal_code"] == "presence_capacity_full", full.text
            assert dispatched == []
            ctx._leave_inflight("ouroboros:presence")
            held.pop()
            later = client.post("/api/letters", json=_msg(letter_id="busy-1", text="While busy"),
                                headers=pair.headers())
            assert later.status_code == 200 and later.json()["state"] == "received"
            assert [event.text for event in dispatched] == ["While busy"]
            rows = _room(client, pair.root_b, pair.place_b)
            assert [m["text"] for m in rows] == ["While busy"]
    finally:
        for _ in range(len(held) + 2):
            ctx._leave_inflight("ouroboros:presence")


def test_a_message_that_comes_before_the_host_service_shares_its_one_context(tmp_path):
    """While a boot still runs, a linked Ouroboros's message can reach the letter door before the Host Service starts.
    The context it makes for the data root is the one that service then serves, with the service's own dependencies:
    the root never holds two turn registries or two reservations. A Host Service started again on the same root
    (a restart inside one process, as tests do) begins its own."""
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.gateway.host_service_places import host_context

    early = host_context(tmp_path)
    turns, runs = early.presence_turns, []
    budget = early._enter_inflight("ouroboros:presence")  # a reservation the early message's turn holds

    def runner(**kwargs):
        runs.append(kwargs)

    served = create_host_service_app(tmp_path, presence_runner=runner).state.host_service_context
    assert served is early and host_context(tmp_path) is early
    assert served.presence_turns is turns and served.presence_runner is runner
    assert budget and served._inflight["ouroboros:presence"] == 1, "the reservation is still the Host's"
    again = create_host_service_app(tmp_path).state.host_service_context
    assert again is not early and host_context(tmp_path) is again and again.presence_runner is None


def test_an_owed_record_that_cannot_be_written_now_is_the_typed_retry_before_any_turn_starts(tmp_path, monkeypatch):
    """A new turn's owed record is written before its dispatch (``_start_intake``). When the link's files cannot take
    it now (a held registry lock, a failed write), the door answers 503 ``peer_links_unreadable`` with the Host's
    retry disposition, never a definite 4xx its sender would record as failed: no turn was dispatched and its
    reservation returned, so the same letter again starts it once."""
    from ouroboros import peer_links, presence_carrier
    from ouroboros.gateway.host_service_places import host_context
    from tests._peer_letters_shared import events

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched = []
    monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                        lambda execution, **kw: (dispatched.append(kw["event"]), _early_answer(execution)))
    real, unwritable = peer_links.record_owed, [True]

    def record_owed(*args, **kwargs):
        if unwritable[0]:
            raise peer_links.PeerLinkError("peer_links_unreadable", "the owed reply could not be recorded now", 503)
        return real(*args, **kwargs)

    monkeypatch.setattr(peer_links, "record_owed", record_owed)
    ctx = host_context(pair.root_b)
    body = _msg(letter_id="owed-1", text="Before its record")
    with receiver_client(pair, monkeypatch) as client:
        refused = client.post("/api/letters", json=body, headers=pair.headers())
        assert refused.status_code == 503, refused.text
        assert (refused.json()["refusal_code"], refused.json()["disposition"]) == ("peer_links_unreadable", "retry")
        assert dispatched == [] and ctx.presence_turns.live() == [] and not ctx._inflight.get("ouroboros:presence")
        assert [row["refusal_code"] for row in events(pair.root_b, "peer_message_retry")] == ["peer_links_unreadable"]
        assert events(pair.root_b, "peer_letter_refused") == [], "a retry is never recorded as a refusal"
        unwritable[0] = False
        later = client.post("/api/letters", json=body, headers=pair.headers())
        assert later.status_code == 200 and later.json()["state"] == "received", later.text
        assert [event.text for event in dispatched] == ["Before its record"]
        assert [m["text"] for m in _room(client, pair.root_b, pair.place_b)] == ["Before its record"]


def test_message_ouroboros_tool_sends_one_stable_letter_and_reads_received_only_beside_its_receipt(tmp_path,
                                                                                                    monkeypatch):
    """``message_ouroboros`` is gated by has_outgoing_link and the place's state, sends to a receiver with
    may_start=False, and is received only beside its delivered receipt; an unconfirmed send records uncertain and the
    same call later sends the same letter, which the receiver keeps once."""
    from ouroboros import places
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as
    from ouroboros.tool_capabilities import OBSERVE_WORLD_MUTATION_TOOLS
    from ouroboros.tools.peer_delegate import _message_ouroboros
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    assert "message_ouroboros" in OBSERVE_WORLD_MUTATION_TOOLS
    bare_ctx = SimpleNamespace(task_id="t0", drive_root=tmp_path / "bare", task_metadata={}, task_contract={})
    (tmp_path / "bare").mkdir()
    avail, reason, _ = _builtin_tool_availability("message_ouroboros", bare_ctx)
    assert avail is False and reason == "no_linked_ouroboros"

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    ctx_a = SimpleNamespace(task_id="t-send-1", drive_root=pair.root_a, budget_drive_root=pair.root_a,
                            task_metadata={"root_task_id": "t-send-1"}, task_contract={})
    with acting_as(pair.root_a):
        assert _builtin_tool_availability("message_ouroboros", ctx_a)[0] is True

    with _both_sides(pair, monkeypatch) as (_client_a, client_b):
        with acting_as(pair.root_a):
            places.edit_place(pair.root_a, pair.place_a, "set_state", author=places.AUTHOR_OWNER, state="silent")
            silent = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Blocked while silent").text)
            assert silent["status"] == "refused" and silent["reason"] == "place_silent"
            places.edit_place(pair.root_a, pair.place_a, "set_state", author=places.AUTHOR_OWNER, state="active")
            sent = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Quick status note.").text)
            again = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Quick status note.").text)
        assert sent["status"] == "received" and again["status"] == "received"
        assert sent["letter_id"] == again["letter_id"] and sent["letter_id"].startswith("msg-")
        rows_b = _room(client_b, pair.root_b, pair.place_b)
        assert [m["text"] for m in rows_b] == ["Quick status note."]

        # The receiver unreachable for every attempt: uncertain, recorded; the same call later is the same letter.
        real_request = peer_gw.PeerGateway._request
        down = {"on": True}

        def unreachable(self, method, path, **kw):
            if down["on"]:
                raise peer_gw._unavailable("peer_unreachable", "no answer")
            return real_request(self, method, path, **kw)

        monkeypatch.setattr(peer_gw.PeerGateway, "_request", unreachable)
        monkeypatch.setattr("ouroboros.peer_replies._RETRY_PAUSE_SEC", 0.0)
        with acting_as(pair.root_a):
            lost = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Second note.").text)
            down["on"] = False
            later = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Second note.").text)
        assert lost["status"] == "refused" and lost["reason"] == "unconfirmed"
        assert later["status"] == "received" and later["letter_id"] == lost["letter_id"]
        assert [m["text"] for m in _room(client_b, pair.root_b, pair.place_b)] == ["Quick status note.", "Second note."]

    states = {(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_a)}
    assert states == {("Quick status note.", "delivered"), ("Second note.", "uncertain"), ("Second note.", "delivered")}
    assert all(r["transport"]["origin"] == {"kind": "tool", "task_id": "t-send-1"} for r in _receipts(pair.root_a))


def test_a_focus_tool_send_answers_the_input_its_model_covered_and_dedup_matches_that_target(monkeypatch):
    """The tool's reply target is the focus's shared reply anchor (``presence_continuation.reply_anchor``): with a
    second message of the link covered by its model's last answered send, that message, not the waking one; before
    version 2, the waking message; outside a focus of that place, none. An automatic output counts as the tool's words
    only when the tool sent the same words to the same message. Pure stand-ins: no store, no network."""
    import ouroboros.peer_replies as replies
    from ouroboros import presence_runner
    from ouroboros.presence_continuation import EARLY_CONTINUATION_VERSION
    from ouroboros.tools.peer_delegate import _reply_target

    event = {"source_event_id": "peer-msg:m-1", "message": {"message_id": "m-1", "peer_message": True}}
    metadata = {"wake_cause": "focus", "presence": {"place_id": "place-b", "event": event},
                "presence_continuation_version": EARLY_CONTINUATION_VERSION}
    second = {"source_event_id": "peer-msg:m-2", "message": {"message_id": "m-2", "peer_message": True}}
    ctx = SimpleNamespace(task_metadata=metadata, _presence_delivered=[{**second, "wake": False}], _presence_covered=1)
    assert _reply_target(ctx, "place-b") == "m-2"
    ctx._presence_covered = 0  # drained after the last answered send: not covered yet
    assert _reply_target(ctx, "place-b") == "m-1"
    older = SimpleNamespace(task_metadata={**metadata, "presence_continuation_version": 1})
    assert _reply_target(older, "place-b") == "m-1"
    assert _reply_target(ctx, "place-c") == "" and _reply_target(SimpleNamespace(task_metadata={}), "place-b") == ""

    sent = {"type": "presence_delivery", "text": "Done.", "transport": {
        "origin": {"kind": "tool"}, "delivery": {"state": "delivered"}, "message": {"reply_to": "m-1"}}}
    monkeypatch.setattr(presence_runner, "_live_task_rows", lambda root, producer, key: [sent])
    assert replies._tool_sent(None, "turn-1", "key", "Done.", "m-1") is True
    assert replies._tool_sent(None, "turn-1", "key", "Done.", "m-2") is False, "the same words to another message"
    assert replies._tool_sent(None, "turn-1", "key", "Other.", "m-1") is False


def test_unstarted_receipt_rechecks_permissions_and_drops_owed_while_started_turn_stays_observable(
    tmp_path, monkeypatch,
):
    """A capture receipt whose intake hit a pre-start retry leaves no owed record and cannot bypass a later pause or
    mine=False, whereas an already-settled hearing or an already-started turn stays observable without re-running or
    cancelling its shared initial Future on one reader's timeout."""
    import concurrent.futures
    import threading

    from ouroboros import peer_links, presence_carrier
    from ouroboros.gateway import peer_letters_api
    from ouroboros.gateway.host_service_places import host_context
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from ouroboros.presence_executions import PresenceTurnNotStarted
    from ouroboros.presence_runner import presence_turn_task_id

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    dispatched: list = []

    with receiver_client(pair, monkeypatch) as client:
        # 1. Off: one saved row, zero dispatches, zero owed records; stays observable after mine=False.
        _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
        off_body = _msg(letter_id="settled-off-1", text="Saved while off.")
        assert client.post("/api/letters", json=off_body, headers=pair.headers()).status_code == 200
        with acting_as(pair.root_b):
            assert peer_links.owed_refs(pair.root_b) == []
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)
        repeat_off = client.post("/api/letters", json=off_body, headers=pair.headers())
        assert repeat_off.status_code == 200 and repeat_off.json()["state"] == "received"
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=False)

        # 2. Pre-start allowance 429: row + receipt captured, zero owed left; lost permission refuses retry.
        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=0.0)
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", lambda *a, **kw: dispatched.append(kw))
        unstarted = _msg(letter_id="unstarted-429", text="Held before start.")
        first_429 = client.post("/api/letters", json=unstarted, headers=pair.headers())
        assert first_429.status_code == 429
        assert first_429.json()["refusal_code"] == "consciousness_allowance_exhausted"
        with acting_as(pair.root_b):
            assert peer_links.message_receipt(pair.root_b, pair.link_id, "unstarted-429") is not None
            assert peer_links.owed_refs(pair.root_b) == []

        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        paused_retry = client.post("/api/letters", json=unstarted, headers=pair.headers())
        assert paused_retry.status_code == 403 and paused_retry.json()["refusal_code"] == "link_paused"
        assert dispatched == []

        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)
        not_mine_retry = client.post("/api/letters", json=unstarted, headers=pair.headers())
        assert not_mine_retry.status_code == 403 and not_mine_retry.json()["refusal_code"] == "peer_may_not_start"
        assert dispatched == []
        with acting_as(pair.root_b):
            assert peer_links.owed_refs(pair.root_b) == []
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=False)

        # 3. Async pre-start 503 (PresenceTurnNotStarted) cleans owed and still checks permissions on retry.
        def fail_async(execution, **kw):
            dispatched.append(kw["event"].text)
            threading.Timer(0.05, lambda: _registry(execution).settle_from_frame(
                execution, error=PresenceTurnNotStarted("pool not ready"))).start()

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", fail_async)
        unstarted_503 = _msg(letter_id="unstarted-503", text="Async no-start.")
        res_503 = client.post("/api/letters", json=unstarted_503, headers=pair.headers())
        assert res_503.status_code == 503 and res_503.json()["refusal_code"] == "presence_turn_not_started"
        with acting_as(pair.root_b):
            assert peer_links.owed_refs(pair.root_b) == []
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        refused_after_503 = client.post("/api/letters", json=unstarted_503, headers=pair.headers())
        assert refused_after_503.status_code == 403 and refused_after_503.json()["refusal_code"] == "link_paused"
        assert dispatched == ["Async no-start."]
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)

        # 4. Real live turn whose reader times out: shared Future is uncancelled, owed stays, repeat joins after Pause.
        plain_future: concurrent.futures.Future = concurrent.futures.Future()  # not marked running: .cancel() would succeed
        live_holders: list = []

        def hold_live(execution, **kw):
            execution.initial = plain_future  # prove timeout does not call .cancel() on a cancellable Future
            dispatched.append(kw["event"].text)
            live_holders.append(execution)

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", hold_live)
        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 0.05)
        started_msg = _msg(letter_id="started-live-1", text="Slow turn start.")
        timed_out = client.post("/api/letters", json=started_msg, headers=pair.headers())
        assert timed_out.status_code == 503 and timed_out.json()["refusal_code"] == "presence_start_unconfirmed"
        assert not plain_future.cancelled(), "one reader's timeout must never cancel the shared initial Future"
        turn_id = presence_turn_task_id(pair.place_b, "peer-msg:started-live-1")
        with acting_as(pair.root_b):
            assert peer_links.owed_refs(pair.root_b) == [(pair.link_id, turn_id)]
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
            edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)

        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 2.0)
        threading.Timer(0.05, lambda: plain_future.set_result({"outcome": "accepted"})).start()
        joined = client.post("/api/letters", json=started_msg, headers=pair.headers())
        assert joined.status_code == 200 and joined.json()["state"] == "received"
        assert dispatched == ["Async no-start.", "Slow turn start."], "joining a live turn must not dispatch twice"
        host_context(pair.root_b).presence_turns.settle_from_frame(
            live_holders[0], result={"outcome": "silent", "text": ""})
        with acting_as(pair.root_b):
            peer_links.settle_owed(pair.root_b, pair.link_id, turn_id, seen=peer_links.owed_version(pair.root_b, pair.link_id, turn_id))


def test_single_attention_owner_late_prestart_settlement_retry_proof_and_live_future_binding(tmp_path, monkeypatch):
    """Covers v2 invariants:
    1. ``presence_intake`` decides attention once; Off, Silent/heard, and handed-to-author arrivals never call
       ``record_owed``.
    2. A late typed pre-start settlement arriving after the HTTP waiter already timed out cleans up ``owed`` via the
       execution's settlement callback without any background loop.
    3. ``owed`` is never cleared on a bare ``resource_refusal_no_resend`` string without canonical
       ``presence_retry_proof`` or on ``PresenceTurnInterrupted``, while a certified proof clears ``owed`` and rechecks
       current link/place permissions on retry.
    4. Observing a live execution on repeat binds its exact ``initial`` Future: if that execution fails pre-start while
       the link is now Paused, the repeat gets that exact 503 without falling through into a new dispatch, and the next
       retry meets the 403 permission refusal."""
    import datetime as dt
    import threading

    from ouroboros import conversation_lease, peer_links, place_intake, presence_carrier
    from ouroboros.gateway import peer_letters_api
    from ouroboros.install_identity import acting_as
    from ouroboros.places import (
        AUTHOR_OWNER, FACT_DIRECT_MESSAGE, FACT_FROM_BOT, STATE_ACTIVE, STATE_PAUSED, edit_place,
    )
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_events import presence_turn_event
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted
    from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    owed_writes: list[str] = []
    attention_calls: list[str] = []
    real_record_owed = peer_links.record_owed
    real_decide = place_intake.decide_attention
    real_offer = place_intake.offer_arrival
    monkeypatch.setattr(peer_links, "record_owed",
                        lambda root, link_id, work_ref, **kw: (owed_writes.append(work_ref),
                                                               real_record_owed(root, link_id, work_ref, **kw))[1])
    monkeypatch.setattr(place_intake, "decide_attention",
                        lambda root, adm, ev, **kw: (attention_calls.append(ev.source_event_id),
                                                     real_decide(root, adm, ev, **kw))[1])

    with receiver_client(pair, monkeypatch) as client:
        # 1a. Off & heard (no_wake_match while On): single attention decision, zero record_owed calls.
        _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
        assert client.post("/api/letters", json=_msg(letter_id="no-owed-off", text="Off note."),
                           headers=pair.headers()).status_code == 200
        _enable_consciousness(pair.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_wakes", author=AUTHOR_OWNER, wakes=[])
        assert client.post("/api/letters", json=_msg(letter_id="no-owed-heard", text="No wake match note."),
                           headers=pair.headers()).status_code == 200
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_wakes", author=AUTHOR_OWNER,
                       wakes=[FACT_DIRECT_MESSAGE, FACT_FROM_BOT])
        assert attention_calls == ["peer-msg:no-owed-off", "peer-msg:no-owed-heard"]
        assert owed_writes == []

        # 1b. Handed to live conversation author: single attention decision, zero record_owed calls.
        monkeypatch.setattr(place_intake, "offer_arrival",
                            lambda root, cap, ev, wake=None: (place_intake.HANDED_TO_AUTHOR, "live-author-1"))
        assert client.post("/api/letters", json=_msg(letter_id="no-owed-handed", text="Handed note."),
                           headers=pair.headers()).status_code == 200
        assert attention_calls[-1] == "peer-msg:no-owed-handed"
        assert owed_writes == []
        monkeypatch.setattr(place_intake, "offer_arrival", real_offer)
        monkeypatch.setattr(peer_links, "record_owed", real_record_owed)
        monkeypatch.setattr(place_intake, "decide_attention", real_decide)

    # Re-apply targeted mocks cleanly in a fresh scope for steps 2-4.
    pair2 = introduce(tmp_path / "p2", monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair2.root_b, monkeypatch, enabled=True, autonomy="act", allowance_usd=5.0)
    dispatched: list = []
    held_execs: list = []

    with receiver_client(pair2, monkeypatch) as client:
        # 2. HTTP reader times out first (503 presence_start_unconfirmed), then execution settles later with
        # PresenceTurnNotStarted -> owed is cleaned up by the execution's settlement callback without a sweeper.
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                            lambda execution, **kw: (dispatched.append(kw["event"].text), held_execs.append(execution)))
        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 0.04)
        late_msg = _msg(letter_id="late-prestart-1", text="Late pool refusal.")
        t_out = client.post("/api/letters", json=late_msg, headers=pair2.headers())
        assert t_out.status_code == 503 and t_out.json()["refusal_code"] == "presence_start_unconfirmed"
        late_turn = presence_turn_task_id(pair2.place_b, "peer-msg:late-prestart-1")
        with acting_as(pair2.root_b):
            assert peer_links.owed_refs(pair2.root_b) == [(pair2.link_id, late_turn)]
        late_exec = held_execs.pop()
        _registry(late_exec).settle_from_frame(late_exec, error=PresenceTurnNotStarted("worker pool never started"))
        with acting_as(pair2.root_b):
            assert peer_links.owed_refs(pair2.root_b) == [], "late typed pre-start settlement clears owed"

        # 3a. PresenceTurnInterrupted (unknown carrier state) or bare resource_refusal_no_resend without proof
        # MUST NOT clear owed; canonical presence_retry_proof DOES clear owed and rechecks permissions on retry.
        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 2.0)

        def interrupt_async(execution, **kw):
            dispatched.append(kw["event"].text)
            threading.Timer(0.02, lambda: _registry(execution).settle_from_frame(
                execution, error=PresenceTurnInterrupted("carrier vanished"))).start()

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", interrupt_async)
        intr_res = client.post("/api/letters", json=_msg(letter_id="intr-1", text="Interrupted."),
                               headers=pair2.headers())
        assert intr_res.status_code == 503 and intr_res.json()["refusal_code"] == "presence_turn_interrupted"
        intr_turn = presence_turn_task_id(pair2.place_b, "peer-msg:intr-1")
        with acting_as(pair2.root_b):
            assert (pair2.link_id, intr_turn) in peer_links.owed_refs(pair2.root_b)
            peer_links.settle_owed(pair2.root_b, pair2.link_id, intr_turn, seen=peer_links.owed_version(pair2.root_b, pair2.link_id, intr_turn))

        def uncertified_refusal(execution, **kw):
            task_id = presence_turn_task_id(kw["admission"].binding_id, kw["event"].source_event_id)
            ident = presence_event_identity(kw["admission"].binding_id, kw["event"])
            write_task_result(pair2.root_b, task_id, STATUS_FAILED, reason_code="resource_refusal_no_resend",
                              metadata={"presence_event_identity": ident,
                                        "presence": {"binding_id": pair2.place_b, "place_id": pair2.place_b,
                                                     "delivery_reporting_version": 1}})
            from ouroboros.presence_runner import _refused_turn, _stored_turn
            err = _refused_turn(pair2.root_b, task_id, _stored_turn(pair2.root_b, task_id, ident),
                                "presence_resources_unavailable", identity=ident)
            _registry(execution).settle_from_frame(execution, error=err)

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", uncertified_refusal)
        uncert = client.post("/api/letters", json=_msg(letter_id="no-proof-1", text="Round two refusal."),
                             headers=pair2.headers())
        assert uncert.status_code == 200  # retained disposition -> 200 received, owed preserved for producer owner
        no_proof_turn = presence_turn_task_id(pair2.place_b, "peer-msg:no-proof-1")
        with acting_as(pair2.root_b):
            assert (pair2.link_id, no_proof_turn) in peer_links.owed_refs(pair2.root_b)
            peer_links.settle_owed(pair2.root_b, pair2.link_id, no_proof_turn, seen=peer_links.owed_version(pair2.root_b, pair2.link_id, no_proof_turn))

        def certified_refusal(execution, **kw):
            task_id = presence_turn_task_id(kw["admission"].binding_id, kw["event"].source_event_id)
            ident = presence_event_identity(kw["admission"].binding_id, kw["event"])
            past_reset = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=5)).isoformat()
            write_task_result(pair2.root_b, task_id, STATUS_FAILED, reason_code="resource_refusal_no_resend",
                              metadata={"presence_event_identity": ident,
                                        "presence": {"binding_id": pair2.place_b, "place_id": pair2.place_b,
                                                     "delivery_reporting_version": 1},
                                        "presence_retry_proof": {
                                            "kind": "first_round_engine_not_started", "task_id": task_id,
                                            "event_identity": ident, "reset_at": past_reset}})
            from ouroboros.presence_runner import _refused_turn, _stored_turn
            err = _refused_turn(pair2.root_b, task_id, _stored_turn(pair2.root_b, task_id, ident),
                                "presence_resources_unavailable", identity=ident)
            _registry(execution).settle_from_frame(execution, error=err)

        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn", certified_refusal)
        cert_msg = _msg(letter_id="with-proof-1", text="Pre-dispatch refusal.")
        cert_res = client.post("/api/letters", json=cert_msg, headers=pair2.headers())
        assert cert_res.status_code == 503 and cert_res.json()["refusal_code"] == "presence_resources_unavailable"
        with acting_as(pair2.root_b):
            assert peer_links.owed_refs(pair2.root_b) == [], "canonical presence_retry_proof clears unstarted owed"
            edit_place(pair2.root_b, pair2.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        cert_retry = client.post("/api/letters", json=cert_msg, headers=pair2.headers())
        assert cert_retry.status_code == 403 and cert_retry.json()["refusal_code"] == "link_paused"
        with acting_as(pair2.root_b):
            edit_place(pair2.root_b, pair2.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)

        # 4. Live execution observed while link is Paused: binds exact live Future, never falls through to new dispatch.
        dispatched.clear()
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                            lambda execution, **kw: (dispatched.append(kw["event"].text), held_execs.append(execution)))
        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 0.04)
        race_msg = _msg(letter_id="live-race-1", text="Race check.")
        first_race = client.post("/api/letters", json=race_msg, headers=pair2.headers())
        assert first_race.status_code == 503 and first_race.json()["refusal_code"] == "presence_start_unconfirmed"
        assert dispatched == ["Race check."]
        live_exec = held_execs.pop()
        with acting_as(pair2.root_b):
            edit_place(pair2.root_b, pair2.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        monkeypatch.setattr(peer_letters_api, "_START_WAIT_SEC", 2.0)
        # The turn must still be live when the repeat reaches the door: its end comes well after the request starts.
        threading.Timer(0.5, lambda: _registry(live_exec).settle_from_frame(
            live_exec, error=PresenceTurnNotStarted("pool failed while paused"))).start()
        joined_fail = client.post("/api/letters", json=race_msg, headers=pair2.headers())
        assert joined_fail.status_code == 503 and joined_fail.json()["refusal_code"] == "presence_turn_not_started"
        assert dispatched == ["Race check."], "must never re-dispatch when joined live Future fails"
        with acting_as(pair2.root_b):
            assert peer_links.owed_refs(pair2.root_b) == []
        after_fail = client.post("/api/letters", json=race_msg, headers=pair2.headers())
        assert after_fail.status_code == 403 and after_fail.json()["refusal_code"] == "link_paused"
        assert dispatched == ["Race check."]
        _ = conversation_lease, admit_presence_turn, presence_turn_event


def test_peer_message_stamps_machine_origin_in_history_memory_and_focus_without_altering_owner_speech(
    tmp_path, monkeypatch,
):
    """Verified peer messages record canonical MACHINE_ORIGIN (#4219) on the real path to dialogue history, memory
    projection and focus framing, while real owner speech keeps human/owner provenance."""
    from ouroboros import chat_chain, chat_references, dialogue_store
    from ouroboros.dialogue_provenance import (
        MACHINE_ORIGIN, owner_speech, place_machine_actor, program_source, render_memory_row, row_author,
    )
    from ouroboros.install_identity import acting_as
    from ouroboros.memory_inventory import row_meta
    from ouroboros.owner_words import owner_row
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_context import focus_metadata, frame_focus_content
    from ouroboros.presence_events import presence_turn_event

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)

    with acting_as(pair.root_b):
        dialogue_store.append(pair.root_b, "chat", {
            "ts": "2026-11-25T07:00:00+00:00", "direction": "in", "chat_id": 0,
            "user_id": 1, "text": "Owner note in Main.", "client_message_id": "owner-main-1",
        })

    with receiver_client(pair, monkeypatch) as client:
        assert client.post("/api/letters", json=_msg(letter_id="m-orig-1", text="Agent report from Laptop."),
                           headers=pair.headers()).status_code == 200
        assert client.post("/api/letters", json=_msg(letter_id="m-orig-2", text="Context reply.", reply_to="out-1"),
                           headers=pair.headers()).status_code == 200
        chat_id = _room_chat_id(pair.root_b, pair.place_b)
        hist = client.get("/api/chat/history", params={"chat_id": chat_id}).json()["messages"]
        assert len(hist) == 2
        for item in hist:
            assert item["sender_label"] == "Laptop"
            assert item["place_history"]["actor_origin"] == MACHINE_ORIGIN
            assert item["place_history"]["owner_confirmed"] is False

    with acting_as(pair.root_b):
        entries = list(chat_chain.iter_stream(pair.root_b, lower=0,
                                              upper=dialogue_store.stream_end(pair.root_b, "chat")))
        owner_entry = next(e.row for e, _leg, _pos in entries if e.row.get("client_message_id") == "owner-main-1")
        peer_item, peer_leg, _ = next((e, leg, pos) for e, leg, pos in entries
                                      if e.row.get("text") == "Agent report from Laptop.")
        peer_row = peer_item.row
        peer_addr = chat_chain.address_at(peer_row, peer_item.seq, peer_leg, sha=chat_chain.source_row_id(peer_row))

        assert peer_row["transport"]["actor"]["origin"] == MACHINE_ORIGIN
        assert place_machine_actor(peer_row) is True
        assert program_source(peer_row) == "place_machine"

        projected = row_meta(peer_row)
        assert projected["transport"]["actor"]["origin"] == MACHINE_ORIGIN
        for candidate in (peer_row, projected):
            author = row_author(candidate)
            assert author["kind"] == "agent" and author["via"] == "place_machine"
            assert author["label"].startswith('Program "Laptop" [provider=ouroboros;')
            assert owner_speech(candidate, author) is False
            assert owner_row(peer_row, pair.root_b) is False

        rendered = render_memory_row(peer_addr, peer_row, author=row_author(peer_row))
        assert 'Program "Laptop"' in rendered and "Agent report from Laptop." in rendered
        target = chat_references.reply_target(pair.root_b, {"address": chat_chain.format_address(peer_addr)})
        assert target["program_source"] == "place_machine"
        assert target["author"].startswith('Program "Laptop" [provider=ouroboros;')

        owner_auth = row_author(owner_entry)
        assert owner_auth["kind"] == "human"
        assert owner_speech(owner_entry, owner_auth) is True
        assert owner_row(owner_entry, pair.root_b) is True

        from ouroboros.gateway.peer_letters_api import _peer_message_event_payload
        from ouroboros.places import get_place

        place = get_place(pair.root_b, pair.place_b)
        caller = {"link_id": pair.link_id, "peer_label": "Laptop", "peer_install_id": place.account_id}
        payload = _peer_message_event_payload(place, caller, _msg(letter_id="m-focus-1", text="Ping"))
        adm = admit_presence_turn(drive_root=pair.root_b, authenticated_transport_skill=place.connector,
                                  binding_id=place.place_id, provider=place.provider, account_id=place.account_id,
                                  conversation_id=place.conversation_id, thread_id="", live_event=True)
        ev = presence_turn_event(payload, adm, reporting_version=1, continuing=2)
        meta = focus_metadata(adm, ev, "t-focus", chat_id, pair.root_b)
        assert meta["focus"]["actor"]["origin"] == MACHINE_ORIGIN
        assert meta["focus"]["actor"]["kind"] == "ouroboros"
        framed = frame_focus_content({"id": "t-focus", "metadata": meta}, "Ping")
        assert 'Program "Laptop"' in str(framed)


def test_a_repeat_whose_turn_left_a_durable_row_is_received_without_any_intake(tmp_path, monkeypatch):
    """``_repeat_message_gate`` says ``reconcile`` (the turn left a durable row no retry may follow, and no replay):
    the repeat is answered ``received`` from that row, and nothing is admitted, captured or offered to
    ``presence_intake``, whose dispatch could start work. Abstract states only: the gate's verdict is given."""
    from ouroboros.gateway import host_service_places, peer_letters_api

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    with receiver_client(pair, monkeypatch) as client:
        _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
        body = _msg(letter_id="reconcile-1", text="Saved once.")
        assert client.post("/api/letters", json=body, headers=pair.headers()).status_code == 200

        def forbidden(*_a, **_k):
            raise AssertionError("a repeat the gate reconciles reaches no admission or intake")

        monkeypatch.setattr(peer_letters_api, "_repeat_message_gate", lambda *a, **k: ("reconcile", None))
        monkeypatch.setattr(host_service_places, "presence_intake", forbidden)
        monkeypatch.setattr("ouroboros.presence_admission.admit_presence_turn", forbidden)
        monkeypatch.setattr("ouroboros.presence_runner.capture_presence_event", forbidden)
        repeat = client.post("/api/letters", json=body, headers=pair.headers())
        assert repeat.status_code == 200, repeat.text
        assert repeat.json() == {"letter_id": "reconcile-1", "kind": "message", "state": "received"}


def test_message_ouroboros_explicit_retry_preserves_original_reply_address_and_bindings(tmp_path, monkeypatch):
    """Explicit ``retry_of`` retries the original message under its original ``reply_to`` even after a newer incoming
    message arrives during a focus turn, while omitting ``retry_of`` creates a new message targeting the new address.
    Mismatched task, place, recipient, text or unknown/unreadable retry records are refused, and Safety still runs."""
    from ouroboros import peer_links, safety
    from ouroboros import peer_replies
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as
    from ouroboros.presence_continuation import EARLY_CONTINUATION_VERSION
    from ouroboros.tools.peer_delegate import _message_ouroboros, get_tools

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0.0)

    event1 = {"source_event_id": "peer-msg:m-1", "message": {"message_id": "m-1", "peer_message": True}}
    event2 = {"source_event_id": "peer-msg:m-2", "message": {"message_id": "m-2", "peer_message": True}}
    meta = {"root_task_id": "focus-1", "wake_cause": "focus",
            "presence": {"place_id": pair.place_a, "event": event1},
            "presence_continuation_version": EARLY_CONTINUATION_VERSION}
    ctx = SimpleNamespace(
        task_id="focus-1", drive_root=pair.root_a, budget_drive_root=pair.root_a,
        task_metadata=meta, task_contract={},
        _presence_delivered=[], _presence_covered=0,
    )

    seen_bodies: list[dict] = []
    real_request = peer_gw.PeerGateway._request

    with _both_sides(pair, monkeypatch) as (_client_a, client_b):
        # Force all 3 attempts of the first call to fail unconfirmed before succeeding on retry_of.
        drops = {"left": 3}

        def drop_all_first_call(self, method, path, **kw):
            if method == "POST" and path == "/api/letters" and kw.get("content"):
                seen_bodies.append(json.loads(kw["content"].decode("utf-8")))
                if drops["left"] > 0:
                    drops["left"] -= 1
                    raise peer_gw._unavailable("peer_unreachable", "connection dropped")
            return real_request(self, method, path, **kw)

        monkeypatch.setattr(peer_gw.PeerGateway, "_request", drop_all_first_call)
        with acting_as(pair.root_a):
            first = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.").text)
        assert first["status"] == "refused" and first["reason"] == "unconfirmed", first
        orig_id = first["letter_id"]
        assert seen_bodies and all(b["letter_id"] == orig_id and b.get("reply_to") == "m-1" for b in seen_bodies)

        # A newer incoming message m-2 arrives and is covered by the model during the same focus turn.
        ctx._presence_delivered = [{**event2, "wake": False}]
        ctx._presence_covered = 1

        # Wrong task / wrong text / malformed / unknown retry_of are all refused before any POST.
        seen_bodies.clear()
        other_task = SimpleNamespace(**{**ctx.__dict__, "task_id": "focus-other"})
        with acting_as(pair.root_a):
            wrong_task = json.loads(_message_ouroboros(other_task, place_id=pair.place_a, text="Answer to m-1.",
                                                       retry_of=orig_id).text)
            wrong_text = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Changed text.",
                                                       retry_of=orig_id).text)
            bad_id = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                   retry_of="bad id!").text)
            missing_id = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                       retry_of="msg-00000000000000000000000000000000").text)
        assert wrong_task["reason"] == "retry_not_owned"
        assert wrong_text["reason"] == "retry_text_mismatch"
        assert bad_id["reason"] == "invalid_retry_of"
        assert missing_id["reason"] == "unknown_message_retry"
        assert seen_bodies == []

        # Recipient mismatch and place mismatch on recorded sent/ are refused before any POST.
        real_resolve = peer_replies.resolve_message_retry
        with acting_as(pair.root_a):
            rec = peer_links.sending_record(pair.root_a, pair.link_id, orig_id)
            monkeypatch.setattr(peer_replies, "resolve_message_retry",
                                lambda *a: ({**rec, "account_id": "other-install"}, ""))
            wrong_rcpt = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                       retry_of=orig_id).text)
            monkeypatch.setattr(peer_replies, "resolve_message_retry",
                                lambda *a: ({**rec, "place_id": "other-place"}, ""))
            wrong_place = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                        retry_of=orig_id).text)
            monkeypatch.setattr(peer_replies, "resolve_message_retry",
                                lambda *a: ({**rec, "origin": {"kind": "automatic", "task_id": "focus-1"}}, ""))
            auto_origin = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                        retry_of=orig_id).text)
            monkeypatch.setattr(peer_replies, "resolve_message_retry", lambda *a: (None, "peer_links_unreadable"))
            unreadable = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                       retry_of=orig_id).text)
            monkeypatch.setattr(peer_replies, "resolve_message_retry", real_resolve)
        assert wrong_rcpt["reason"] == "retry_recipient_mismatch"
        assert wrong_place["reason"] == "retry_place_mismatch"
        assert auto_origin["reason"] == "retry_not_owned"
        assert unreadable["reason"] == "peer_links_unreadable"
        assert seen_bodies == []

        # Explicit retry_of keeps orig_id and reply_to="m-1" despite m-2 being the current _reply_target!
        with acting_as(pair.root_a):
            retried = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                    retry_of=orig_id).text)
        assert retried["status"] == "received" and retried["letter_id"] == orig_id, retried
        assert len(seen_bodies) == 1 and seen_bodies[0]["letter_id"] == orig_id and seen_bodies[0]["reply_to"] == "m-1"

        # Repeating retry_of AFTER delivery & sent/ cleanup resolves from canonical receipt with ZERO POSTs.
        seen_bodies.clear()
        with acting_as(pair.root_a):
            assert peer_links.sending_record(pair.root_a, pair.link_id, orig_id) is None
            replayed = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.",
                                                     retry_of=orig_id).text)
        assert replayed["status"] == "received" and replayed["letter_id"] == orig_id
        assert seen_bodies == []

        # A NEW message (without retry_of) takes the new reply_to="m-2" and a distinct letter_id.
        with acting_as(pair.root_a):
            newer = json.loads(_message_ouroboros(ctx, place_id=pair.place_a, text="Answer to m-1.").text)
        assert newer["status"] == "received" and newer["letter_id"] != orig_id
        assert len(seen_bodies) == 1 and seen_bodies[0]["reply_to"] == "m-2"

        rows_b = _room(client_b, pair.root_b, pair.place_b)
        assert len(rows_b) == 2

    # Safety schema contract: text remains required and safety_policy remains "check" (never bypassed on retry_of).
    entry = next(t for t in get_tools() if t.name == "message_ouroboros")
    assert entry.safety_policy == "check"
    assert set(entry.schema["parameters"]["required"]) == {"place_id", "text"}
    assert "retry_of" in entry.schema["parameters"]["properties"]
    # Verify check_safety does NOT defer message_ouroboros with retry_of (exact words reach _assess_call).
    seen_safety: list[tuple[str, dict]] = []
    monkeypatch.setattr(safety, "_assess_call", lambda tool_name, arguments, *a, **kw: (
        seen_safety.append((tool_name, dict(arguments))),
        (False, "blocked by safety on retry"),
    )[1])
    allowed, msg = safety.check_safety("message_ouroboros", {"place_id": pair.place_a, "text": "Answer to m-1.",
                                                             "retry_of": orig_id}, ctx=ctx)
    assert allowed is False and "blocked by safety on retry" in msg
    assert seen_safety and seen_safety[0][0] == "message_ouroboros" and seen_safety[0][1]["text"] == "Answer to m-1."


def test_lost_receipt_preserves_confirmed_outcome_across_pause_revoke_and_panic(tmp_path, monkeypatch):
    """A confirmed delivery (or definite first-send refusal) whose local receipt write fails saves ``confirmed_state``
    in ``sent/``. Even if the link is later Paused, Revoked, or in Panic, retrying reconciles the exact confirmed
    receipt with zero extra POSTs and never turns it into ``uncertain`` or a refusal."""
    from ouroboros import peer_links, place_projection, places
    from ouroboros import peer_replies
    from ouroboros.install_identity import acting_as
    from ouroboros.tools.peer_delegate import _message_ouroboros

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    ctx_a = SimpleNamespace(task_id="t-lost-rcpt", drive_root=pair.root_a, budget_drive_root=pair.root_a,
                            task_metadata={"root_task_id": "t-lost-rcpt"}, task_contract={})

    real_record = peer_replies._record_receipt
    fail_receipt = {"on": True}
    post_count = {"n": 0}
    from ouroboros.gateways import peer as peer_gw
    real_request = peer_gw.PeerGateway._request

    def counting_request(self, method, path, **kw):
        if method == "POST" and path == "/api/letters":
            post_count["n"] += 1
        return real_request(self, method, path, **kw)

    def flaky_receipt(*a, **kw):
        if fail_receipt["on"]:
            return False
        return real_record(*a, **kw)

    with _both_sides(pair, monkeypatch) as (_client_a, client_b):
        monkeypatch.setattr(peer_gw.PeerGateway, "_request", counting_request)
        monkeypatch.setattr(peer_replies, "_record_receipt", flaky_receipt)
        with acting_as(pair.root_a):
            first = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Confirmed once.").text)
        assert first["status"] == "refused" and first["reason"] == "receipt_unrecorded", first
        assert post_count["n"] == 1
        lid = first["letter_id"]
        with acting_as(pair.root_a):
            rec = peer_links.sending_record(pair.root_a, pair.link_id, lid)
            assert rec and rec["confirmed_state"] == "delivered"

        # Now Panic + Pause + Revoke all happen on A, and receipt storage recovers: retry still reconciles delivered
        # with ZERO extra POSTs!
        fail_receipt["on"] = False
        monkeypatch.setattr(place_projection, "switch_state", lambda root: "panic")
        with acting_as(pair.root_a):
            places.edit_place(pair.root_a, pair.place_a, "set_state", author=places.AUTHOR_OWNER, state="paused")
            peer_links.revoke(pair.root_a, pair.link_id)
            reconciled = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Confirmed once.",
                                                       retry_of=lid).text)
        assert reconciled["status"] == "received" and reconciled["letter_id"] == lid, reconciled
        assert post_count["n"] == 1, "already-confirmed delivery never POSTs again"
        with acting_as(pair.root_a):
            assert peer_links.sending_record(pair.root_a, pair.link_id, lid) is None
        assert [m["text"] for m in _room(client_b, pair.root_b, pair.place_b)] == ["Confirmed once."]

    states = [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair.root_a)]
    assert states == [("Confirmed once.", "delivered")]

    # Definite first-send refusal whose receipt write fails also preserves confirmed_state="failed" across pause.
    pair_f = introduce(tmp_path / "fail_case", monkeypatch, b_lets_a_start=False)
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "on")
    fail_receipt["on"] = True
    f_posts = {"n": 0}
    monkeypatch.setattr(peer_gw.PeerGateway, "send_message", lambda *a, **kw: (
        f_posts.__setitem__("n", f_posts["n"] + 1),
        (_ for _ in ()).throw(peer_gw._unavailable("letter_id_reused", "refused first send", 409, "letter_id_reused")),
    ))
    with acting_as(pair_f.root_a):
        r1 = peer_replies.send_peer_message(pair_f.root_a, pair_f.place_a, letter_id="msg-fail-1", text="Refused.",
                                            origin={"kind": "tool", "task_id": "t-f"})
        assert r1["state"] == "failed" and r1["recorded"] is False and f_posts["n"] == 1
        assert peer_links.sending_record(pair_f.root_a, pair_f.link_id, "msg-fail-1")["confirmed_state"] == "failed"
        fail_receipt["on"] = False
        places.edit_place(pair_f.root_a, pair_f.place_a, "set_state", author=places.AUTHOR_OWNER, state="paused")
        r2 = peer_replies.send_peer_message(pair_f.root_a, pair_f.place_a, letter_id="msg-fail-1", text="Refused.",
                                            origin={"kind": "tool", "task_id": "t-f"})
        assert r2["state"] == "failed" and r2["recorded"] is True and r2.get("earlier") is True
        assert f_posts["n"] == 1 and peer_links.sending_record(pair_f.root_a, pair_f.link_id, "msg-fail-1") is None
    assert [(r["text"], r["transport"]["delivery"]["state"]) for r in _receipts(pair_f.root_a)] == [
        ("Refused.", "failed")]


def test_unreadable_history_returns_unknown_without_post_or_fake_receipt_and_rolls_back_unstarted(tmp_path, monkeypatch):
    """Unreadable receipt history or unreadable ``sent/`` returns ``unknown`` with zero POSTs and zero fabricated
    ``uncertain`` receipts; a definitely-unstarted ``NOTHING_SENT`` rolls back ``passes`` so it never falsely exhausts
    ``_MAX_PASSES``."""
    from ouroboros import peer_links, presence_delivery
    from ouroboros import peer_replies
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.install_identity import acting_as
    from ouroboros.tools.peer_delegate import _message_ouroboros

    pair = introduce(tmp_path, monkeypatch, b_lets_a_start=False)
    _enable_consciousness(pair.root_b, monkeypatch, enabled=False)
    ctx_a = SimpleNamespace(task_id="t-unk", drive_root=pair.root_a, budget_drive_root=pair.root_a,
                            task_metadata={"root_task_id": "t-unk"}, task_contract={})
    posts: list = []
    real_request = peer_gw.PeerGateway._request
    monkeypatch.setattr(peer_gw.PeerGateway, "_request",
                        lambda self, m, p, **kw: (posts.append((m, p)), real_request(self, m, p, **kw))[1])

    with _both_sides(pair, monkeypatch):
        real_states = presence_delivery.recorded_states
        monkeypatch.setattr(presence_delivery, "recorded_states", lambda *a: (_ for _ in ()).throw(OSError("db busy")))
        with acting_as(pair.root_a):
            unk = json.loads(_message_ouroboros(ctx_a, place_id=pair.place_a, text="Never sent.").text)
        assert unk["status"] == "refused" and unk["reason"] == "receipts_unreadable", unk
        assert posts == [] and _receipts(pair.root_a) == []
        monkeypatch.setattr(presence_delivery, "recorded_states", real_states)

        # Definitely-unstarted send (place_paused raised by gateway before wire) rolls back passes from 4 -> 3.
        with acting_as(pair.root_a):
            account_b = str(peer_links.link_of_place(pair.root_a, pair.place_a)[1]["peer_install_id"])
            lid = peer_replies.message_letter_id(pair.link_id, "t-unk", "", "Prior unconfirmed.")
            for _ in range(3):
                peer_links.record_sending(pair.root_a, pair.link_id, lid, account_id=account_b,
                                          place_id=pair.place_a, text="Prior unconfirmed.", reply_to="",
                                          origin={"kind": "tool", "task_id": "t-unk"})
            monkeypatch.setattr(peer_gw.PeerGateway, "send_message",
                                lambda *a, **kw: (_ for _ in ()).throw(
                                    peer_gw._unavailable("place_paused", "paused mid-send")))
            res = peer_replies.send_peer_message(pair.root_a, pair.place_a, letter_id=lid, text="Prior unconfirmed.",
                                                 origin={"kind": "tool", "task_id": "t-unk"})
            rec = peer_links.sending_record(pair.root_a, pair.link_id, lid)
        assert res["state"] == "uncertain" and res["held_code"] == "place_paused" and not res.get("spent")
        assert res["code"] == "earlier_attempt_unconfirmed"
        assert rec["passes"] == 3, "unstarted attempt must roll back pass counter instead of burning attempts_spent"


def test_tool_retry_keeps_proven_address_on_rebind_and_unreadable_never_advertises_missing_retry_id(
        tmp_path, monkeypatch):
    """(2) Explicit tool retry passes proven ``(link_id, account_id)`` into ``send_peer_message`` so a concurrent
    ``link_of_place`` rebind never redirects the lookup or send, and an unreadable lookup fails honest ``unknown``.
    (3) Unreadable storage on a fresh call omits ``letter_id`` and does not advertise ``retry_of``, while unreadable
    storage during an explicit ``retry_of`` preserves ``retry_of`` and ``letter_id``."""
    from ouroboros import peer_links, place_projection, presence_delivery
    from ouroboros import peer_replies
    from ouroboros.gateways import peer as peer_gw
    from ouroboros.tools.peer_delegate import _message_ouroboros, get_tools

    entry = next(t for t in get_tools() if t.name == "message_ouroboros")
    assert "or unreadable outcome" not in entry.schema["description"]
    assert "keep the same retry_of" in entry.schema["description"]

    ctx = SimpleNamespace(task_id="t-bound", drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_metadata={}, task_contract={})
    monkeypatch.setattr(place_projection, "switch_state", lambda root: "on")
    monkeypatch.setattr(peer_gw, "peer_outgoing_speech_refusal", lambda root, place_id: None)
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0.0)

    # 3a. Fresh call hitting unreadable link registry or unreadable receipts: no letter_id, no retry_of prompt.
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: (_ for _ in ()).throw(
        peer_links.PeerLinkError("peer_links_unreadable", "busy", 503)))
    fresh_reg_err = json.loads(_message_ouroboros(ctx, place_id="place-1", text="Hello.").text)
    assert fresh_reg_err["reason"] == "peer_links_unreadable"
    assert "letter_id" not in fresh_reg_err and "retry_of" not in fresh_reg_err["detail"]

    # 3b. Explicit retry_of hitting unreadable link registry: preserves retry_of and letter_id!
    retry_reg_err = json.loads(_message_ouroboros(ctx, place_id="place-1", text="Hello.",
                                                  retry_of="msg-orig1234").text)
    assert retry_reg_err["reason"] == "peer_links_unreadable"
    assert retry_reg_err["letter_id"] == "msg-orig1234" and retry_reg_err["retry_of"] == "msg-orig1234"
    assert "retry_of='msg-orig1234'" in retry_reg_err["detail"]

    # 3c. Fresh call hitting unreadable receipts inside send_peer_message: omits letter_id and retry_of hint.
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: ("link-orig", {"peer_install_id": "inst-orig"}))
    monkeypatch.setattr(peer_links, "sending_record", lambda *a: None)
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda *a: (_ for _ in ()).throw(OSError("db busy")))
    fresh_rcpt_err = json.loads(_message_ouroboros(ctx, place_id="place-1", text="Hello.").text)
    assert fresh_rcpt_err["reason"] == "receipts_unreadable"
    assert "letter_id" not in fresh_rcpt_err and "retry_of=" not in fresh_rcpt_err["detail"]

    # 3d. Explicit retry_of hitting unreadable receipts inside send_peer_message: preserves retry_of and letter_id!
    orig_rec = {"letter_id": "msg-orig1234", "account_id": "inst-orig", "place_id": "place-1",
                "text": "Hello.", "reply_to": "", "origin": {"kind": "tool", "task_id": "t-bound"}, "passes": 1}
    monkeypatch.setattr(peer_replies, "resolve_message_retry", lambda *a: (orig_rec, ""))
    monkeypatch.setattr(peer_links, "sending_record", lambda *a: orig_rec)
    retry_rcpt_err = json.loads(_message_ouroboros(ctx, place_id="place-1", text="Hello.",
                                                   retry_of="msg-orig1234").text)
    assert retry_rcpt_err["reason"] == "receipts_unreadable"
    assert retry_rcpt_err["letter_id"] == "msg-orig1234" and retry_rcpt_err["retry_of"] == "msg-orig1234"
    assert "retry_of='msg-orig1234'" in retry_rcpt_err["detail"]

    # 2. Proven (link_id, account_id) passed into send_peer_message: rebind after resolve never redirects lookup/send.
    posts: list = []
    monkeypatch.setattr(peer_gw, "PeerGateway", lambda link: SimpleNamespace(
        send_message=lambda *a, **kw: posts.append((link, kw)), close=lambda: None))
    looked_up: list = []
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda root, skill, acc, lid, part: (
        looked_up.append((acc, lid)), set(),
    )[1])
    calls = {"n": 0}

    def rebind_after_first(root, place_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return "link-orig", {"peer_install_id": "inst-orig"}
        return "link-orig", {"peer_install_id": "inst-other"}  # recipient changed before admission!

    monkeypatch.setattr(peer_links, "link_of_place", rebind_after_first)
    monkeypatch.setattr(peer_links, "outgoing_link", lambda *a, **kw: peer_links.OutgoingLink(
        link_id="link-orig", place_id="place-1", peer_url="http://127.0.0.1:9",
        peer_label="Other", peer_install_id="inst-other", own_install_id="inst-self", drive_root=str(tmp_path)))
    monkeypatch.setattr(peer_replies, "_receipts", lambda root: SimpleNamespace(record=lambda *a: None))
    rebound = json.loads(_message_ouroboros(ctx, place_id="place-1", text="Hello.", retry_of="msg-orig1234").text)
    assert rebound["status"] == "refused" and posts == []
    assert looked_up == [("inst-orig", "msg-orig1234")], "lookup of past outcome stays at proven original address"

    # Already-delivered letter reconciles at proven original address even if link_of_place is now rebound!
    looked_up.clear()
    monkeypatch.setattr(presence_delivery, "recorded_states", lambda root, skill, acc, lid, part: (
        looked_up.append((acc, lid)), {"delivered"},
    )[1])
    monkeypatch.setattr(peer_links, "link_of_place", lambda *a: ("link-new", {"peer_install_id": "inst-new"}))
    done = peer_replies.send_peer_message(
        tmp_path, "place-1", letter_id="msg-orig1234", text="Hello.", reply_to="",
        origin={"kind": "tool", "task_id": "t-bound"}, link_id="link-orig", account_id="inst-orig",
    )
    assert done["state"] == "delivered" and done.get("earlier") is True and posts == []
    assert looked_up == [("inst-orig", "msg-orig1234")]
