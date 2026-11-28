"""T4 — End-to-end two isolated servers connected ONLY via their POSIX letters doors (no SSH).

Modelled on S45 & S46 (tests/system_e2e/test_system_scenarios_w11.py):
1. Server B issues a v2 remote invitation (``remote: True``); a small Unix-socket forwarder bridges
   A's ``l-<link16>.sock`` -> B's ``door.sock`` and B's ``r-<link16>-<session12>.sock`` -> A's ``door.sock``.
2. Server A accepts via ``POST /api/peer-links/accept`` with ``session``; both sides confirm the six digits.
   No main-port URL is exchanged between A and B.
3. A delegates an assignment to B (``delegate_to_ouroboros``); B asks a clarification (``ask_sender``);
   A wakes from ``delegate_wait``, answers (``delegate_answer``), and receives B's final result.
4. B (with ``may_start=False`` on A) sends a standalone message first while A's Consciousness is Off
   (captured, ``attention=heard/off``, ``owner_confirmed=False``, 0 model calls, 0 tasks), then switches
   A's Consciousness On and sends a second message that wakes a focus turn on A and replies back to B's room.

Opt in with ``OUROBOROS_E2E_DEEP=mock`` (keyless mock lane); no provider key, no network, no SSH.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile
import time

import pytest

from ouroboros import peer_channel
from ouroboros.install_identity import acting_as
from ouroboros.peer_letters import letter_task_id
from tests.candidate_checkout import CandidateError, candidate_checkout, require_candidate_interpreter
from tests.system_e2e.harness import (
    LANE_MOCK,
    REPO_ROOT,
    ArtifactOracle,
    body_text,
    clone_repo,
    require_lane,
    two_isolated_servers,
    wait_durable_result,
)
from tests.test_peer_channel import UnixSocketForwarder

_POSIX_DOOR_ONLY = pytest.mark.skipif(
    not peer_channel.is_door_supported(),
    reason="POSIX Unix-domain letters door only",
)

LETTER = "Count the items after asking me which bucket, then reply with the count. [T4-LETTER]"
QUESTION = "Which bucket should I count? [T4-QUESTION]"
ANSWER = "T4-ANSWER: bucket alpha"
B_RESULT = "T4_B_RESULT: bucket alpha has 7 items."
A_FINAL = "T4_A_FINAL"
STEPS = 40

_RUN_RE = re.compile(r'"run_id": "([0-9a-f]{32})"')
_IID_RE = re.compile(r'"interaction_id": "([^"]+)"')
_MESSAGE_RECEIPT_RE = re.compile(r'"status": "(?:received|queued)"[^}]*"letter_id": "(msg-[0-9a-f]{32})"')


def _poll_until(predicate, *, timeout: float = 60.0, label: str = "condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {label}: last={last!r}")


def _wait_result(install, task_id: str, *, timeout: float = 180) -> dict:
    with acting_as(install.data_dir):
        return wait_durable_result(ArtifactOracle(install.data_dir), task_id, timeout=timeout)


def _peer_place(install) -> dict:
    return next(row for row in install.api("GET", "/api/places")["places"] if row.get("connector") == "ouroboros")


def _a_assignment_step(place_id: str):
    def step(body: dict) -> dict:
        text = body_text(body)
        runs = _RUN_RE.findall(text)
        if not runs:
            return {"tool": "delegate_to_ouroboros", "arguments": {
                "place_id": place_id, "title": "T4 door count", "text": LETTER, "cost_ceiling_usd": 1.0}}
        letter_id = runs[-1]
        if B_RESULT in text:
            return {"final": f"{A_FINAL}: {B_RESULT}"}
        interactions = _IID_RE.findall(text)
        if interactions and text.rfind('"status": "waiting_on_user"') > text.rfind('"status": "delivered"'):
            return {"tool": "delegate_answer", "arguments": {
                "run_id": letter_id, "interaction_id": interactions[-1],
                "answers": [{"question_id": interactions[-1], "free_text": ANSWER}]}}
        return {"tool": "delegate_wait", "arguments": {"run_id": letter_id}}

    return step


def _b_assignment_step(body: dict) -> dict:
    if ANSWER not in body_text(body):
        return {"tool": "ask_sender", "arguments": {"question": QUESTION}}
    return {"final": B_RESULT}


def _send_one_note(place_id: str, text: str, final: str):
    def step(body: dict) -> dict:
        receipt = _MESSAGE_RECEIPT_RE.search(body_text(body))
        if receipt:
            return {"final": f"{final}: {receipt[1]}"}
        return {"tool": "message_ouroboros", "arguments": {"place_id": place_id, "text": text}}

    return step


def _received_note(install, place_id: str, text: str):
    rooms = install.api("GET", f"/api/places/{place_id}/rooms")["rooms"]
    rows = [row for room in rooms for row in install.api(
        "GET", f"/api/chat/history?chat_id={room['room_chat_id']}")["messages"] if row.get("text") == text]
    assert len(rows) <= 1, rows
    return rows[0] if rows else None


@pytest.fixture(scope="module")
def t4_candidate(tmp_path_factory):
    """Byte-faithful candidate checkout via tests/candidate_checkout.py (never skips symlinks)."""
    require_lane(LANE_MOCK)
    try:
        require_candidate_interpreter()
    except CandidateError as exc:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        if dirty.strip():
            pytest.fail(f"{exc} — and the working tree is dirty, so a HEAD clone would not be the candidate")
        yield clone_repo(tmp_path_factory.mktemp("t4"))
        return
    with candidate_checkout(REPO_ROOT, tmp_path_factory.mktemp("t4") / "candidate", origin_proof=True) as candidate:
        yield candidate


@_POSIX_DOOR_ONLY
@pytest.mark.integration
def test_t4_two_isolated_servers_via_doors_without_ssh(t4_candidate, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = pathlib.Path(tempfile.mkdtemp(prefix="t4d-", dir="/tmp"))
    door_dirs: list[pathlib.Path] = []
    try:
        with two_isolated_servers(t4_candidate, root) as (a, b):
            door_dirs.extend([peer_channel.resolve_door_dir(a.data_dir), peer_channel.resolve_door_dir(b.data_dir)])
            # Wait for both isolated servers' POSIX letters doors to bind door.sock.
            door_a = peer_channel.door_socket_path(a.data_dir)
            door_b = peer_channel.door_socket_path(b.data_dir)
            _poll_until(lambda: door_a.exists() and door_b.exists(), timeout=30.0, label="both letters doors bound")

            # 1. B creates a v2 remote invitation; forwarders link A's opener socket and B's receiver socket to doors.
            invite = b.api("POST", "/api/peer-links/invite", {
                "remote": True, "label": "Laptop", "own_label": "Velm", "may_start": True,
                "note": "Laptop may ask Velm to count items over the channel.",
            })
            assert invite["invite"].startswith("ouroboros-invite:2:"), invite
            link_id = invite["link_id"]
            session = peer_channel.next_channel_session()

            fwd_a_to_b = UnixSocketForwarder(peer_channel.opener_socket_path(a.data_dir, link_id), door_b, mode="forward")
            fwd_b_to_a = UnixSocketForwarder(
                peer_channel.receiver_socket_path(b.data_dir, link_id, session), door_a, mode="forward"
            )
            try:
                accepted = a.api("POST", "/api/peer-links/accept", {
                    "invite": invite["invite"], "session": session,
                    "label": "Velm", "own_label": "Laptop", "may_start": False,
                })
                place_a = accepted["place_id"]
                place_b_row = _peer_place(b)
                place_b = place_b_row["place_id"]
                assert accepted["link_id"] == link_id == place_b_row["link"]["link_id"]
                assert accepted["code"] == place_b_row["link"]["code"] and len(accepted["code"]) == 6
                assert accepted.get("transport") == "channel" and place_b_row["link"].get("transport") == "channel"

                for inst in (b, a):
                    confirmed = inst.api("POST", f"/api/peer-links/{link_id}/confirm", {})
                    assert confirmed["code_check"] == "confirmed" and confirmed["state"] == "active", confirmed

                # 2. Assignment with clarification (ask_sender + delegate_answer) and final result over doors.
                a.stub.script = [_a_assignment_step(place_a)] * STEPS
                b.stub.script = [_b_assignment_step] * STEPS
                task_a = a.submit("Ask Velm over the door channel to count the bucket items. [T4-ASSIGN]")
                res_a = a.server.wait_task(task_a, timeout=300)
                assert res_a.get("status") == "completed", res_a
                stored_a = _wait_result(a, task_a)
                assert A_FINAL in str(stored_a.get("result") or "") and B_RESULT in str(stored_a.get("result") or "")
                a_transcript = "\n".join(body_text(call) for _kind, call in a.stub.calls)
                letter_id = _RUN_RE.findall(a_transcript)[0]
                assert QUESTION in a_transcript and '"status": "delivered"' in a_transcript
                row_b = _wait_result(b, letter_task_id(place_b, letter_id))
                assert row_b["status"] == "completed" and B_RESULT in str(row_b.get("result") or "")

                # 3. Standalone message from B (may_start=False on A) with A's Consciousness Off.
                assert a.api("GET", "/api/state")["bg_consciousness_enabled"] is False
                a_calls_before = len(a.stub.calls)
                note_off = "Velm standalone message while Laptop is Off [T4-OFF]"
                b.stub.script = [_send_one_note(place_b, note_off, "T4_VELM_SENT_OFF")] * 8
                t_off = b.submit("Send Laptop an offline-consciousness note. [T4-SEND-OFF]")
                assert b.server.wait_task(t_off, timeout=180).get("status") == "completed"
                assert "T4_VELM_SENT_OFF" in str(_wait_result(b, t_off).get("result") or "")

                off_row = _poll_until(lambda: _received_note(a, place_a, note_off), label="offline note delivered")
                rooms_a = a.api("GET", f"/api/places/{place_a}/rooms")["rooms"]
                assert len(rooms_a) == 1, rooms_a
                assert off_row["place_history"]["owner_confirmed"] is False
                assert off_row["place_history"]["attention"] == {"decision": "heard", "reason": "off"}
                assert len(a.stub.calls) == a_calls_before, "Consciousness Off must make zero model calls"

                # 4. Standalone message from B with A's Consciousness On -> focus wake + reply back through door.
                a.api("POST", "/api/settings", {
                    "OUROBOROS_CONSCIOUSNESS_AUTONOMY": "act",
                    "OUROBOROS_CONSCIOUSNESS_DAILY_USD": "5.0",
                })
                a.api("POST", "/api/command", {"cmd": "/bg start"})
                _poll_until(lambda: a.api("GET", "/api/state").get("bg_consciousness_enabled") is True,
                            label="Laptop consciousness enabled")

                note_on = "Velm standalone message while Laptop is On [T4-ON]"
                laptop_reply = "Laptop focus reply over door channel [T4-REPLY]"
                a.stub.script = [
                    lambda body: (
                        {"final": laptop_reply} if note_on in body_text(body)
                        else {"tool": "finish_task", "arguments": {"action": "finish", "silent": True, "answer": "quiet"}}
                    )
                ] * 8
                b.stub.script = [_send_one_note(place_b, note_on, "T4_VELM_SENT_ON")] * 8
                t_on = b.submit("Send Laptop a message while Consciousness is On. [T4-SEND-ON]")
                assert b.server.wait_task(t_on, timeout=180).get("status") == "completed"
                assert "T4_VELM_SENT_ON" in str(_wait_result(b, t_on).get("result") or "")

                def _velm_got_reply():
                    rooms = b.api("GET", f"/api/places/{place_b}/rooms")["rooms"]
                    if not rooms:
                        return None
                    msgs = b.api("GET", f"/api/chat/history?chat_id={rooms[0]['room_chat_id']}")["messages"]
                    found = [
                        m for m in msgs
                        if laptop_reply in str(m.get("text") or "")
                        and m.get("role") == "user"
                        and (m.get("place_history") or {}).get("attention")
                    ]
                    return found[0] if found else None

                reply_row = _poll_until(_velm_got_reply, timeout=90.0, label="Velm receives Laptop focus reply via door")
                assert reply_row["sender_label"] == "Laptop"
                assert reply_row["place_history"]["owner_confirmed"] is False
                assert reply_row["place_history"]["attention"]["decision"] == "heard"
            finally:
                fwd_a_to_b.close()
                fwd_b_to_a.close()
    finally:
        for d in door_dirs:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)
