"""S45 — the first «Multibrain» step on two real servers of one checkout (``harness.two_isolated_servers``).

My human introduces two of their installations at their owner doors: B invites and lets A start work there, A
accepts and lets B only answer. Until the six digits are confirmed on a side, a letter there is refused
``link_paused`` (``code_check_pending``); my human confirms on both sides. A's work then writes B a letter with
``delegate_to_ouroboros``; B's letter root asks its sender one question (``ask_sender``) and waits; A, asleep in
``delegate_wait``, wakes on that question, answers with ``delegate_answer``, and B's result returns into A's work.
B's link never starts work on A (``peer_may_not_start``). The owner then pauses B's place, and A's next letter is
refused ``link_paused`` before any work starts there; then revokes the link on B, and A's key is
``link_unknown`` there.

Each install answers its model calls from its own loopback stub with a reactive step: it reads the transcript
the server sent and chooses the next call, so the scenario follows what the two servers actually did rather
than a timed script. The wire checks use the key A's data directory holds, as A's own requests do.

Opt in with ``OUROBOROS_E2E_DEEP=mock`` (the keyless mock lane); no provider key, no network, no spend.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import pytest

from ouroboros.install_identity import IDENTITY_PATH, acting_as
from ouroboros.peer_letters import letter_task_id
from ouroboros.peer_links import token_path
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    body_text,
    require_lane,
    two_isolated_servers,
    wait_durable_result,
)

LETTER = "Count the files in the folder I name when you ask me which one, then reply with the count. [S45-LETTER]"
QUESTION = "Which folder exactly? [S45-QUESTION]"
ANSWER = "S45-ANSWER: the folder is /tmp"
B_RESULT = "S45_B_RESULT: there are 3 files."
A_FINAL = "S45_A_FINAL"
A_PAUSED_FINAL = "S45_A_PAUSED_FINAL"
STEPS = 40  # reactive steps per install: each decides from the transcript, a spare one only repeats a wait

_RUN_RE = re.compile(r'"run_id": "([0-9a-f]{32})"')  # a letter's run id is its 32-hex letter id
_IID_RE = re.compile(r'"interaction_id": "([^"]+)"')


def _poll_until(predicate, *, timeout: float = 45.0, label: str = "condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {label}: last={last!r}")


def _task_result(install, task_id: str) -> dict:
    with acting_as(install.data_dir):
        return ArtifactOracle(install.data_dir).task_result(task_id) or {}


def _wait_result(install, task_id: str, *, timeout: float = 180) -> dict:
    with acting_as(install.data_dir):
        return wait_durable_result(ArtifactOracle(install.data_dir), task_id, timeout=timeout)


def _sleep_carrier(row: dict) -> dict:
    for key in ("budget_pause", "owner_wait"):
        block = row.get(key)
        if isinstance(block, dict) and block.get("reason") == "sleep":
            return block
    return {}


def _a_sleep_wait(a, task_a: str, letter_id: str, *, excluding_sleep_id: str = "") -> dict | None:
    """Return A's durable sleep record + /api/state work row once A is positively parked in ``delegate_wait``."""
    if not task_a or not letter_id:
        return None
    carrier = _sleep_carrier(_task_result(a, task_a))
    sleep = carrier.get("sleep") if isinstance(carrier.get("sleep"), dict) else {}
    sleep_id = str(sleep.get("sleep_id") or carrier.get("pause_id") or carrier.get("wait_id") or "")
    if (carrier.get("state") not in {"paused", "waiting"} or not sleep_id
            or sleep_id == excluding_sleep_id or letter_id not in (sleep.get("runs") or ())):
        return None
    state = a.api("GET", "/api/state")
    work = next((w for w in (state.get("works") or ()) if w.get("work_id") == task_a or w.get("task_id") == task_a), None)
    if not isinstance(work, dict) or work.get("activity") != "waiting":
        return None
    return {"sleep_id": sleep_id, "carrier": dict(carrier), "work": work}


def _b_peer_wait(b, task_b: str) -> dict | None:
    """Return B's durable peer-question wait + /api/state snapshot once B is parked in ``ask_sender``."""
    row = _task_result(b, task_b)
    wait = row.get("owner_wait") if isinstance(row.get("owner_wait"), dict) else {}
    if wait.get("state") != "waiting":
        return None
    state = b.api("GET", "/api/state")
    work = next((w for w in (state.get("works") or ()) if w.get("work_id") == task_b or w.get("task_id") == task_b), None)
    if not isinstance(work, dict) or work.get("activity") != "waiting":
        return None
    return {"wait": dict(wait), "work": work, "state": state}


def _a_step(place_id: str, observed: dict):
    """A's reactive step: the first work writes, sleeps, wakes on B's question, answers, sleeps again and finishes."""
    def step(body: dict) -> dict:
        text = body_text(body)
        if "[S45-PAUSED]" in text:
            if '"tool": "delegate_to_ouroboros"' in text:  # its refusal is in the transcript
                return {"final": A_PAUSED_FINAL}
            return {"tool": "delegate_to_ouroboros", "arguments": {"place_id": place_id, "text": "One more count?"}}
        runs = _RUN_RE.findall(text)
        if not runs:
            return {"tool": "delegate_to_ouroboros", "arguments": {
                "place_id": place_id, "title": "S45 count", "text": LETTER, "cost_ceiling_usd": 1.0}}
        letter_id = runs[-1]
        observed["letter_id"] = letter_id
        if B_RESULT in text:
            return {"final": f"{A_FINAL}: {B_RESULT}"}
        interactions = _IID_RE.findall(text)
        if interactions and text.rfind('"status": "waiting_on_user"') > text.rfind('"status": "delivered"'):
            if "b_parked" not in observed:
                task_b = letter_task_id(observed["place_b_id"], letter_id)
                observed["b_parked"] = _poll_until(lambda: _b_peer_wait(observed["b"], task_b),
                                                   label="B parked in ask_sender peer wait")
                observed["b_wire_waiting"] = _wire(observed["b"], "GET", f"/api/letters/{letter_id}", observed["a_key"])
                observed["a_after_wake_1"] = dict(_task_result(observed["a"], observed["task_a"]))
            return {"tool": "delegate_answer", "arguments": {
                "run_id": letter_id, "interaction_id": interactions[-1],
                "answers": [{"question_id": interactions[-1], "free_text": ANSWER}]}}
        return {"tool": "delegate_wait", "arguments": {"run_id": letter_id}}
    return step


def _b_step(observed: dict):
    """B's letter root: waits for A's first sleep before asking, and for A's second sleep before finishing."""
    def step(body: dict) -> dict:
        letter_id = _poll_until(lambda: observed.get("letter_id"), label="A recorded letter_id")
        task_a = _poll_until(lambda: observed.get("task_a"), label="task_a submitted")
        if ANSWER not in body_text(body):
            observed["a_sleep_1"] = _poll_until(
                lambda: _a_sleep_wait(observed["a"], task_a, letter_id),
                label="A parked in first delegate_wait before B asks")
            return {"tool": "ask_sender", "arguments": {"question": QUESTION}}
        first_sleep_id = observed["a_sleep_1"]["sleep_id"]
        observed["b_wire_answered"] = _wire(observed["b"], "GET", f"/api/letters/{letter_id}", observed["a_key"])
        observed["a_sleep_2"] = _poll_until(
            lambda: _a_sleep_wait(observed["a"], task_a, letter_id, excluding_sleep_id=first_sleep_id),
            label="A parked in second delegate_wait before B finishes")
        return {"final": B_RESULT}
    return step


def _key_headers(install, link_id: str) -> dict:
    """The three headers of a letter from ``install``: its link secret and install id, as its own requests carry."""
    secret = json.loads(token_path(install.data_dir, link_id).read_text(encoding="utf-8"))["token"]
    install_id = json.loads((install.data_dir / IDENTITY_PATH).read_text(encoding="utf-8"))["id"]
    return {"X-Ouroboros-Link": f"{link_id}.{secret}", "X-Ouroboros-Install": str(install_id),
            "X-Ouroboros-Protocol": "1"}


def _wire(install, method: str, path: str, headers: dict, body: dict | None = None) -> tuple[int, dict]:
    """One request to ``install`` as a linked Ouroboros sends it (no proxy), its status and JSON body."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(install.base_url + path, data=data, method=method, headers={
        **headers, **({"Content-Type": "application/json"} if data is not None else {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=60) as response:
            return int(response.status), json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read() or b"{}")


def _peer_place(install) -> dict:
    return next(row for row in install.api("GET", "/api/places")["places"] if row.get("connector") == "ouroboros")


def _events(install, kind: str) -> list:
    return ArtifactOracle(install.data_dir).events(kind)


@pytest.mark.integration
def test_s45_two_installations_meet_confirm_and_exchange_one_letter(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s45")
    with two_isolated_servers(e2e_clone, root) as (a, b):
        # -- the introduction at both owner doors: B invites, A accepts; only hashes cross.
        invite = b.api("POST", "/api/peer-links/invite", {
            "own_url": b.base_url, "label": "Laptop", "own_label": "Velm", "may_start": True,
            "note": "Laptop may ask me to count files."})
        accepted = a.api("POST", "/api/peer-links/accept", {
            "invite": invite["invite"], "own_url": a.base_url, "label": "Velm", "own_label": "Laptop",
            "may_start": False})
        link_id, place_a = accepted["link_id"], accepted["place_id"]
        place_b = _peer_place(b)
        assert link_id == invite["link_id"] and place_b["link"]["link_id"] == link_id, (accepted, place_b)
        assert accepted["code"] == place_b["link"]["code"] and len(accepted["code"]) == 6, (accepted, place_b)
        assert accepted["code_check"] == place_b["link"]["code_check"] == "pending"
        a_key = _key_headers(a, link_id)
        early = _wire(b, "POST", "/api/letters", a_key, {"letter_id": "early-1", "text": "too soon", "protocol": 1})
        assert early[0] == 403 and (early[1]["refusal_code"], early[1].get("detail")) == (
            "link_paused", "code_check_pending"), early

        # -- my human compares the six digits and confirms on each side; both places start.
        for install in (b, a):
            confirmed = install.api("POST", f"/api/peer-links/{link_id}/confirm", {})
            assert confirmed["code_check"] == "confirmed" and confirmed["state"] == "active", confirmed
        for install in (a, b):
            facts = _peer_place(install)
            assert facts["peer"]["mine"] is True and "mine: yes" in facts["peer_facts"], facts
        assert "may_start: yes" in _peer_place(b)["peer_facts"] and "may_start: no" in _peer_place(a)["peer_facts"]

        # -- B's link only answers on A: a letter from B starts nothing there.
        b_key = _key_headers(b, link_id)
        reverse = _wire(a, "POST", "/api/letters", b_key, {"letter_id": "reverse-1", "text": "start", "protocol": 1})
        assert reverse[0] == 403 and reverse[1]["refusal_code"] == "peer_may_not_start", reverse

        # -- the letter, its question and answer, and the result in A's work (with B's consciousness Off).
        assert b.api("GET", "/api/state")["bg_consciousness_enabled"] is False
        observed: dict = {"a": a, "b": b, "a_key": a_key, "place_b_id": place_b["place_id"]}
        a.stub.script = [_a_step(place_a, observed)] * STEPS
        b.stub.script = [_b_step(observed)] * STEPS
        task_a = a.submit("Ask Velm, my other Ouroboros, to count the files it is told about, and report. [S45-RUN]")
        observed["task_a"] = task_a
        result = a.server.wait_task(task_a, timeout=600)
        assert result.get("status") == "completed", result
        stored = _wait_result(a, task_a)
        assert A_FINAL in str(stored.get("result") or "") and B_RESULT in str(stored.get("result") or ""), stored
        assert (stored["task_id"], stored.get("task_attempt", 1), stored.get("original_task_id")) == (task_a, 1, None)
        assert observed["a_sleep_1"]["sleep_id"] != observed["a_sleep_2"]["sleep_id"]
        a_final_sleep = _sleep_carrier(stored)
        assert a_final_sleep.get("state") == "resumed", a_final_sleep
        assert (a_final_sleep.get("sleep") or {}).get("sleep_id") == observed["a_sleep_2"]["sleep_id"], a_final_sleep
        assert (observed["a_after_wake_1"]["task_id"], _sleep_carrier(observed["a_after_wake_1"]).get("state")) == (
            task_a, "resumed")
        a_transcript = "\n".join(body_text(call) for _kind, call in a.stub.calls)
        letter_id = _RUN_RE.findall(a_transcript)[0]
        assert '"status": "waiting_on_user"' in a_transcript and QUESTION in a_transcript, "A saw B's question"
        assert '"status": "delivered"' in a_transcript, "A's answer letter was accepted"

        requested = [row for row in _events(a, "delegate_run_start_requested") if row.get("invocation_id") == letter_id]
        started = [row for row in _events(a, "delegate_run_started") if row.get("run_id") == letter_id]
        assert len(requested) == 1 and requested[0]["engine_ref"] == {
            "kind": "peer", "place_id": place_a, "link_id": link_id}, requested
        assert len(started) == 1, started

        # While B waited on ask_sender, it parked on peer_question (reason='owner', quiz_id='', never an owner turn).
        b_parked = observed["b_parked"]
        qid = b_parked["wait"]["peer_question"]["question_id"]
        assert (b_parked["wait"]["reason"], b_parked["wait"]["quiz_id"]) == ("owner", ""), b_parked["wait"]
        assert b_parked["wait"]["peer_question"] == {"question_id": qid, "sender": "Laptop"}, b_parked["wait"]
        assert (b_parked["work"]["activity"], b_parked["work"]["release"], b_parked["work"]["turn"]) == (
            "waiting", "self", []), b_parked["work"]
        assert b_parked["work"]["because"] == [{"code": "peer_answer", "text": "Waiting for Laptop's answer"}]
        assert b_parked["state"]["work_summary"]["turn"] == 0 and b_parked["state"]["current_questions"] == []
        assert observed["b_wire_waiting"] == (200, {
            "letter_id": letter_id, "state": "needs_input",
            "question": {"question_id": qid, "text": QUESTION}}), observed["b_wire_waiting"]
        assert observed["b_wire_answered"][0] == 200 and observed["b_wire_answered"][1]["state"] == "accepted_running"

        # B ran one root for the letter, in the link's room, under the letter's ceiling, authored by the link key.
        task_b = letter_task_id(place_b["place_id"], letter_id)
        row_b = _wait_result(b, task_b)
        assert row_b["status"] == "completed" and B_RESULT in str(row_b.get("result") or ""), row_b
        assert (row_b["task_id"], row_b.get("task_attempt", 1), row_b.get("original_task_id")) == (task_b, 1, None)
        assert (row_b["owner_wait"]["state"], row_b["owner_wait"]["resume_reason"]) == ("resumed", "answer"), row_b
        metadata = row_b.get("metadata") or {}
        assert metadata["peer_letter"]["link_id"] == link_id and metadata["root_cost_ceiling_usd"] == 1.0, metadata
        assert metadata["letter_author"] == {"kind": "agent", "label": "Laptop", "basis": "link_key",
                                                "key_id": link_id}, metadata
        assert [question["state"] for question in (row_b.get("peer_questions") or {}).values()] == ["answered"]
        assert len(_events(b, "peer_letter_accepted")) == 1 and len(_events(b, "peer_letter_reply_accepted")) == 1
        b_transcript = "\n".join(body_text(call) for _kind, call in b.stub.calls)
        assert "a letter from Laptop" in b_transcript and ANSWER in b_transcript, "B read the letter and the answer"
        assert "not a confirmed owner answer" not in b_transcript and "has not answered yet" not in b_transcript

        # The status and the result, as A's key reads them.
        status = _wire(b, "GET", f"/api/letters/{letter_id}", a_key)
        assert status == (200, {"letter_id": letter_id, "state": "done"}), status
        final = _wire(b, "GET", f"/api/letters/{letter_id}/result", a_key)
        assert final[0] == 200 and final[1]["state"] == "done" and B_RESULT in final[1]["result"], final
        assert set(final[1]["cost"]) == {"known_usd", "final"}, final

        # -- my human pauses B's place: A's next letter is refused there before any work starts.
        paused = b.api_status("POST", f"/api/places/{place_b['place_id']}", {"state": "paused"})
        assert paused["status"] == 200, paused
        task_paused = a.submit("Write Velm one more letter. [S45-PAUSED]")
        assert a.server.wait_task(task_paused, timeout=300).get("status") == "completed"
        assert A_PAUSED_FINAL in str(_wait_result(a, task_paused).get("result"))
        refused = {row.get("letter_id") for row in _events(b, "peer_letter_refused")
                   if row.get("refusal_code") == "link_paused"} - {"early-1"}
        assert len(refused) == 1, _events(b, "peer_letter_refused")
        failed = [row for row in _events(a, "delegate_run_start_failed") if row.get("reason") == "link_paused"]
        assert len(failed) == 1 and failed[0].get("definite") is True, _events(a, "delegate_run_start_failed")
        assert len(_events(b, "peer_letter_accepted")) == 1, "the paused link started no work"

        # -- my human revokes the link on B: A's key is unknown there, on every letter route.
        revoked = b.api("POST", f"/api/peer-links/{link_id}/revoke", {})
        assert revoked["state"] == "removed" and revoked["revoked_at"], revoked
        for method, path, body in (("POST", "/api/letters", {"letter_id": "late-1", "text": "x", "protocol": 1}),
                                   ("GET", f"/api/letters/{letter_id}", None)):
            gone = _wire(b, method, path, a_key, body)
            assert gone[0] == 401 and gone[1]["refusal_code"] == "link_unknown", (path, gone)
        assert not token_path(b.data_dir, link_id).exists(), "B's own secret left with the revocation"


@pytest.mark.integration
def test_s46_two_installations_exchange_standalone_messages_off_and_on(e2e_clone, tmp_path_factory):
    """Two real servers: Velm (B) has may_start=False on Laptop (A), yet initiates standalone messages.

    1) Laptop Consciousness Off -> message captured in room, visible via /api/places/{id}/rooms and
       /api/chat/history with attention=heard/off and owner_confirmed=False; 0 model calls and 0 tasks on Laptop.
    2) Laptop Consciousness On (Acts) -> second message wakes a real Presence focus turn on Laptop, which replies
       back over the saved link; Velm receives the reply in its room with owner_confirmed=False and no ping-pong loop.
    3) Each sender's own room reads its words as delivered only from the receiver's typed confirmation (a v1 receipt:
       saved in that room, never read): Laptop's reply and Velm's two notes.
    """
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s46")
    with two_isolated_servers(e2e_clone, root) as (a, b):
        invite = b.api("POST", "/api/peer-links/invite", {
            "own_url": b.base_url, "label": "Laptop", "own_label": "Velm", "may_start": True})
        accepted = a.api("POST", "/api/peer-links/accept", {
            "invite": invite["invite"], "own_url": a.base_url, "label": "Velm", "own_label": "Laptop",
            "may_start": False})
        link_id, place_a = accepted["link_id"], accepted["place_id"]
        place_b = _peer_place(b)["place_id"]
        for install in (b, a):
            confirmed = install.api("POST", f"/api/peer-links/{link_id}/confirm", {})
            assert confirmed["code_check"] == "confirmed" and confirmed["state"] == "active", confirmed

        # -- Phase 1: Laptop Consciousness is Off; Velm sends a standalone message despite may_start=False on Laptop.
        assert a.api("GET", "/api/state")["bg_consciousness_enabled"] is False
        a_calls_before = len(a.stub.calls)
        note_off = "Velm note while Laptop is Off: epoch 3 loss=0.21 [S46-OFF]"
        b.stub.script = [
            lambda body: (
                {"final": "S46_VELM_SENT_OFF"} if '"status": "received"' in body_text(body)
                else {"tool": "message_ouroboros", "arguments": {"place_id": place_b, "text": note_off}}
            )
        ] * 8
        t_off = b.submit("Send Laptop a standalone note about epoch 3. [S46-SEND-OFF]")
        assert b.server.wait_task(t_off, timeout=180).get("status") == "completed"
        assert "S46_VELM_SENT_OFF" in str(_wait_result(b, t_off).get("result") or "")

        rooms_a = a.api("GET", f"/api/places/{place_a}/rooms")["rooms"]
        assert len(rooms_a) == 1, rooms_a
        chat_a = rooms_a[0]["room_chat_id"]
        hist_a = a.api("GET", f"/api/chat/history?chat_id={chat_a}")["messages"]
        assert len(hist_a) == 1, hist_a
        assert hist_a[0]["text"] == note_off
        assert hist_a[0]["sender_label"] == "Velm"
        assert hist_a[0]["place_history"]["owner_confirmed"] is False
        assert hist_a[0]["place_history"]["attention"] == {"decision": "heard", "reason": "off"}
        assert len(a.stub.calls) == a_calls_before, "Consciousness Off must make zero model calls"
        assert a.api("GET", "/api/state").get("works") == [], "Consciousness Off must create zero tasks"

        # -- Phase 2: Laptop switches Consciousness On (Acts); Velm sends another note; Laptop wakes & replies back!
        a.api("POST", "/api/settings", {"OUROBOROS_CONSCIOUSNESS_AUTONOMY": "act",
                                        "OUROBOROS_CONSCIOUSNESS_DAILY_USD": "5.0"})
        a.api("POST", "/api/command", {"cmd": "/bg start"})
        _poll_until(lambda: a.api("GET", "/api/state").get("bg_consciousness_enabled") is True,
                    label="Laptop consciousness enabled")

        note_on = "Velm note while Laptop is On: epoch 4 finished, keep ckpt4? [S46-ON]"
        laptop_reply = "Laptop reply to Velm: yes, keep checkpoint 4. [S46-REPLY]"
        a.stub.script = [
            lambda body: (
                {"final": laptop_reply} if note_on in body_text(body)
                else {"tool": "finish_task", "arguments": {"action": "finish", "silent": True, "answer": "quiet"}}
            )
        ] * 8
        b.stub.script = [
            lambda body: (
                {"final": "S46_VELM_SENT_ON"} if '"status": "received"' in body_text(body)
                else {"tool": "message_ouroboros", "arguments": {"place_id": place_b, "text": note_on}}
            )
        ] * 8
        t_on = b.submit("Send Laptop a standalone note about epoch 4. [S46-SEND-ON]")
        assert b.server.wait_task(t_on, timeout=180).get("status") == "completed"
        assert "S46_VELM_SENT_ON" in str(_wait_result(b, t_on).get("result") or "")

        # Wait for Laptop's focus reply to arrive back in Velm's conversation room with Laptop:
        def _velm_got_reply():
            rooms = b.api("GET", f"/api/places/{place_b}/rooms")["rooms"]
            if not rooms:
                return None
            msgs = b.api("GET", f"/api/chat/history?chat_id={rooms[0]['room_chat_id']}")["messages"]
            found = [m for m in msgs if laptop_reply in str(m.get("text") or "") and m.get("role") == "user"
                     and (m.get("place_history") or {}).get("attention")]  # saved first, then its intake settles
            return found[0] if found else None

        reply_row = _poll_until(_velm_got_reply, timeout=90.0, label="Velm receives Laptop focus reply")
        assert reply_row["sender_label"] == "Laptop"
        assert reply_row["place_history"]["owner_confirmed"] is False
        assert reply_row["place_history"]["attention"]["decision"] == "heard"

        # Verify Laptop's room shows Velm's second note as focus_started and never owner_confirmed:
        hist_a_after = a.api("GET", f"/api/chat/history?chat_id={chat_a}")["messages"]
        on_row = next(m for m in hist_a_after if m.get("text") == note_on)
        assert on_row["place_history"]["owner_confirmed"] is False
        assert on_row["place_history"]["attention"] == {"decision": "focus_started"}

        # The author's room confirms its reply from the receiver's typed answer (its v1 receipt), never "not confirmed".
        def _laptop_reply_delivered():
            rows = a.api("GET", f"/api/chat/history?chat_id={chat_a}")["messages"]
            return [m for m in rows if laptop_reply in str(m.get("text") or "") and m.get("role") == "assistant"]

        laptop_rows = _poll_until(_laptop_reply_delivered, timeout=60.0, label="Laptop's reply receipt")
        assert [m["place_history"].get("delivery_state") for m in laptop_rows] == ["delivered"], laptop_rows
        # The receipt lands before its audit line (``send_peer_message``): wait for that line, never race it.
        delivered_a = _poll_until(lambda: _events(a, "peer_message_delivered"), timeout=30.0,
                                  label="Laptop's delivered audit line")
        assert len(delivered_a) == 1 and delivered_a[0].get("reply_to"), delivered_a  # one reply, to Velm's note
        chat_b = b.api("GET", f"/api/places/{place_b}/rooms")["rooms"][0]["room_chat_id"]
        hist_b = b.api("GET", f"/api/chat/history?chat_id={chat_b}")["messages"]
        sent_b = {m["text"]: m["place_history"].get("delivery_state") for m in hist_b if m.get("role") == "assistant"}
        assert sent_b == {note_off: "delivered", note_on: "delivered"}, sent_b
        assert not _events(a, "peer_message_retry") and not _events(b, "peer_message_retry")
