"""S26 — v7 follow-up Ф1 (sprint plan §5.1 Ф1-B, D-05): the owner stops a
DIRECT-CHAT turn mid-round.

THE CLASS (rc.7 QA regress, item 010). A message typed into the main chat runs as
a DIRECT-CHAT TURN. It once ran on the chat agent INSIDE the supervisor with no
queue row, so ``GET /api/tasks?status=running`` listed the turn while every
queue-keyed owner control answered 404 "task not found or not active", and the
spend kept growing until ``/restart``. Since #2654 the turn runs on the
CONVERSATION LANE: routing publishes its RUNNING row first (never PENDING) and a
custodied worker process carries it (``supervisor/conversation_lane.py``), so
"a lane turn stops like any worker" (ARCHITECTURE §5 Cancellation): one durable
intent, the one custody claim, the carrier's death confirmed, a typed
``cancelled`` terminal and its owed receipt. The cooperative ``finalize_now``
control (``REASON_OWNER_STOPPED_DIRECT_TURN``) remains only for the disclosed
inline executors (``inline_carrier.stop_direct_chat_turn``), pinned by
``tests/test_direct_chat_turn_owner_control.py``; THIS scenario pins the whole
worker-carried path on a real server, keyless, over the same WS chat and HTTP
cancel surfaces the SPA drives.

DETERMINISM. "Mid-turn" is not a sleep. The loopback model HOLDS the turn's first
tool-bearing round on a ``ModelGate`` (harness): the request thread blocks until the
scenario releases it, so the owner's stop provably lands while the turn is inside
a model round. Everything else is durable-event polling over ``ArtifactOracle``
readers.

WHAT S26 ASSERTS, in order:
1. IN FLIGHT and ADDRESSABLE — inside a model round (``gate.arrived``), the turn is
   in ``/api/state.active_chat_activities`` as kind ``direct_chat`` linked to the
   client_message_id, listed by ``GET /api/tasks?status=running`` (its durable
   running row), its published RUNNING row is the conversation lane's held by a
   live carrier process (``conversation_turn_row``), and ``events.jsonl`` carries
   its durable ``task_received`` row with ``_is_direct_chat=true`` and the
   client_message_id link;
2. STOP-NOW over the UI's ``POST /api/tasks/{id}/cancel`` (empty body = immediate)
   mid-round is NOT 404: custody tore the carrier down before answering, so the
   answer is the typed ``200 {"ok": true, "task_id": ...}``; ONE durable request
   with the forensic trail requested(http_single, single) → claimed →
   settled(cancelled); no ``finalize_now`` control (that is the inline
   executors' stop); the carrier process is gone and the projection drained;
3. the turn ENDED with ZERO further model rounds — the durable terminal is
   ``cancelled`` with its ``http_single`` cancel origin, ``task_done`` names the
   same status, the held round never completed (the gate matched exactly ONE
   round, the keepalive script was never run down, no other call reached the
   model even after the release) and no paid post-task work exists (no open
   ``root_phase_checkpoint``, no paid ``authored_root_summary`` row, no
   reflection);
4. the chat CONCLUDED (no "Working…" forever) — over the SAME /ws the SPA opens
   the turn announced itself (typing frame: activity_id = task id, kind
   ``direct_chat``, the client_message_id link) and the keyed system
   ``cancel_receipt`` frame for this request arrived; the activity left
   ``/api/state``, the running list no longer names the turn, the receipt
   landed in chat.jsonl under the task id, and the root's terminal projection
   row is final with the same status;
5. a LATER stop is the typed 404 with nothing re-armed: no new intent, no new
   forensic request, the terminal fields unchanged.

The default-lane test pins the ``ModelGate`` contract itself, through the stub's
real HTTP surface: match-only, once-only, held OUTSIDE the call lock (unrelated
calls and ``kinds()`` keep flowing), released on the event.
"""

from __future__ import annotations

import json
import threading
import urllib.request
import uuid

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    conversation_turn_row,
    keyless_settings,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
    ws_url,
)
# The wave-2 cancellation glue, reused rather than re-derived: one author for the
# keepalive step and the cancel-intent forensic readers across the cancel scenarios.
from tests.system_e2e.test_system_scenarios_w2 import (
    KEEPALIVE_STEP,
    _cancel_forensics,
    _forensic_events,
)

from devtools.benchmarks.common.server_runner import _api

# ===========================================================================
# Default lane: the ModelGate contract (no server, one loopback stub).
# ===========================================================================


def _post_completion(stub: ScriptedStubModel, payload: dict, out: list) -> None:
    req = urllib.request.Request(
        stub.base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out.append(json.loads(resp.read().decode("utf-8")))


def _tool_body(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "list_files"}}]}


def test_model_gate_holds_once_outside_the_call_lock_and_releases():
    gate = ModelGate(lambda body: bool(body.get("tools")) and "HOLD-ME" in body_text(body),
                     timeout=30)
    with ScriptedStubModel([dict(KEEPALIVE_STEP) for _ in range(3)], gate=gate) as stub:
        held: list = []
        worker = threading.Thread(target=_post_completion,
                                  args=(stub, _tool_body("HOLD-ME"), held), daemon=True)
        worker.start()
        assert gate.arrived.wait(10), "the matching call never reached the gate"
        assert held == [] and gate.held == 1 and gate.matched == 1
        # Unrelated calls and the scenario's reader are NOT behind the hold: a
        # tool-less periphery call quoting the marker is answered at once.
        other: list = []
        _post_completion(stub, {"messages": [{"role": "user", "content": "periphery HOLD-ME"}]}, other)
        assert other and stub.kinds() == ["final"]
        assert held == [] and gate.matched == 1
        gate.release.set()
        worker.join(10)
        assert not worker.is_alive() and held, "the held call did not return after release"
        assert held[0]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "list_files"
        # A later match passes through untouched: counted, not held.
        again: list = []
        _post_completion(stub, _tool_body("HOLD-ME again"), again)
        assert again and gate.matched == 2 and gate.held == 1
        assert stub.kinds() == ["final", "agent", "agent"]
        assert gate.timed_out is False


def test_model_gate_expiry_is_loud_not_a_silent_release():
    """An unreleased hold must fail BY NAME: flag set, TimeoutError raised."""
    gate = ModelGate(lambda body: True, timeout=0.2)
    raised: list = []

    def _call():
        try:
            gate({"messages": []})
        except TimeoutError as exc:
            raised.append(str(exc))

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive()
    assert gate.timed_out is True and gate.held == 1 and gate.matched == 1
    assert raised and "never released" in raised[0], raised


# ===========================================================================
# Mock lane: S26 on a real isolated server.
# ===========================================================================

S26_CHAT_ID = 1  # WEB_UI_CHAT_ID: the ordinary owner chat thread
S26_MARKER = "S26_DIRECT_TURN_e2e_w6"
S26_PROBE = (f"[{S26_MARKER}] List the repository root and keep listing it "
             "until the owner stops you.")
# Literal of the tree under test (the cancel endpoint's answer for a settled task).
S26_NOT_ACTIVE = "task not found or not active"


def _is_turn_round(body: dict) -> bool:
    """A TOOL-BEARING call carrying the probe = one model ROUND of the direct turn.

    The turn's tool-less periphery (the proactive card namer, compaction) quotes
    the owner's text too and must be neither held nor counted as a round."""
    return bool(body.get("tools")) and S26_MARKER in body_text(body)


class _WsFrames:
    """Every frame the SAME /ws the SPA opens delivers, collected on its own thread.

    ``websockets.sync`` allows ``send`` from one thread while another blocks in
    ``recv``; the scenario sends from the test thread and reads the collected
    list. Never a sleep: readers poll the list through ``wait_until``.
    """

    def __init__(self, ws) -> None:
        self.ws = ws
        self.frames: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def _pump(self) -> None:
        from websockets.exceptions import ConnectionClosed

        while not self._stop.is_set():
            try:
                raw = self.ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except ConnectionClosed:
                return
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            if isinstance(frame, dict):
                self.frames.append(frame)

    def __enter__(self) -> "_WsFrames":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(5)

    def find(self, **want) -> list:
        return [f for f in list(self.frames) if all(f.get(k) == v for k, v in want.items())]


def _running_task_ids(server) -> set:
    """The PUBLIC running list (the durable-row projection the SPA's task list reads)."""
    listing = _api(server.base_url, "GET", "/api/tasks?status=running", timeout=30)
    return {str(row.get("task_id") or "") for row in (listing.get("tasks") or [])
            if isinstance(row, dict)}


def _direct_activities(server, client_message_id: str) -> list:
    """The direct turn(s) the /api/state activity snapshot links to *client_message_id*."""
    state = _api(server.base_url, "GET", "/api/state", timeout=30)
    return [a for a in (state.get("active_chat_activities") or [])
            if isinstance(a, dict) and a.get("kind") == "direct_chat"
            and str(a.get("client_message_id") or "") == client_message_id]


def _mailbox_controls(data_root, task_id: str) -> list:
    """The ``finalize_now`` rows of the turn's canonical owner mailbox, via the
    tree's own path derivation (the mailbox the loop drains)."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, _mailbox_path

    path = _mailbox_path(data_root, task_id)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if isinstance(row, dict) and row.get("kind") == KIND_FINALIZE_NOW]


def _post_task_synthesis_is_open(stored: dict) -> bool:
    """The durable checkpoint the boot reconciler re-dispatches (and re-pays) from."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open

    checkpoint = stored.get("root_phase_checkpoint")
    status = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    return post_task_synthesis_is_open(status)


def _post_stop_synthesis_rows(oracle: ArtifactOracle, task_id: str) -> list:
    """Durable traces of PAID post-task work for the turn: any reflection row and
    any paid narrative row (the paid summary no longer exists, so this stays a
    regression guard). The free host facts row is not paid work and may exist."""
    summaries = [r for r in oracle._jsonl("logs/chat.jsonl", type_filter="task_summary")
                 if str(r.get("task_id") or "") == task_id
                 and r.get("summary_kind") == "authored_root_summary"]
    reflections = [r for r in oracle._jsonl("logs/task_reflections.jsonl")
                   if str(r.get("task_id") or "") == task_id]
    return summaries + reflections


def _terminal_fields(oracle: ArtifactOracle, task_id: str) -> dict:
    row = oracle.task_result(task_id)
    return {key: row.get(key) for key in ("status", "reason_code", "result")}


@pytest.mark.integration
def test_s26_direct_chat_turn_owner_stop_mid_round_is_typed_and_ends_the_turn(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from ouroboros.platform_layer import pid_is_alive

    root = tmp_path_factory.mktemp("s26")
    gate = ModelGate(_is_turn_round, timeout=240)
    # A SHORT keepalive script on purpose: were the stop ever not honoured, the turn
    # would run these rounds and end on the stub's final answer — the round-count pin
    # below then names that failure instead of a long burn.
    with ScriptedStubModel([dict(KEEPALIVE_STEP) for _ in range(6)], gate=gate) as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            oracle = ArtifactOracle(server.data_root)
            from websockets.sync.client import connect as ws_connect

            client_message_id = f"e2e-s26-{uuid.uuid4().hex[:12]}"
            # proxy=None: the loopback /ws is never proxied — websockets otherwise
            # honours an operator shell's HTTP(S)_PROXY even for ws:// and a
            # hermetic blackhole proxy (the isolation env) refuses the connect.
            with ws_connect(ws_url(server), open_timeout=30, proxy=None) as ws, \
                    _WsFrames(ws) as frames:
                ws.send(json.dumps({
                    "type": "chat", "content": S26_PROBE, "chat_id": S26_CHAT_ID,
                    "client_message_id": client_message_id,
                }))

                # 1. IN FLIGHT inside a model round — and ADDRESSABLE.
                assert gate.arrived.wait(180), "the direct turn never reached its first model round"
                # The provider gate and supervisor event consumer are separate:
                # the registered Preparing row may precede consumption of the
                # main_llm_call_state event that projects Working (#1740).
                activity = wait_until(
                    lambda: next((row for row in _direct_activities(server, client_message_id)
                                  if row.get("phase") == "working"), None), 30)
                assert activity, "the direct turn never became Working in /api/state.active_chat_activities"
                task_id = str(activity.get("activity_id") or "")
                # #1740 uses neutral Working once a physical model call starts.
                assert task_id and activity.get("phase") == "working", activity
                assert wait_until(lambda: task_id in _running_task_ids(server), 30), (
                    "the live direct turn is not listed by GET /api/tasks?status=running")
                # #2654: the conversation lane's RUNNING row, held by a live carrier process.
                carrier = conversation_turn_row(oracle, task_id)["carrier_pid"]
                # The durable running row agent.py writes for the turn (the row the
                # public running list projects) — bound to the owner's chat thread.
                running_row = oracle.task_result(task_id)
                assert running_row.get("status") == "running", running_row
                assert int(running_row.get("chat_id") or 0) == S26_CHAT_ID, running_row
                received = wait_until(
                    lambda: [r for r in oracle.events("task_received")
                             if str((r.get("task") or {}).get("id") or "") == task_id] or None, 30)
                assert received, [str((r.get("task") or {}).get("id")) for r in oracle.events("task_received")]
                received_task = received[-1].get("task") or {}
                assert received_task.get("_is_direct_chat") is True, received_task
                assert ((received_task.get("metadata") or {}).get("origin_message_ref") or {}).get(
                    "client_message_id") == client_message_id, received_task

                # 2. STOP-NOW over the UI's endpoint, mid-round: typed, never 404. Custody
                #    confirmed the carrier's death before answering, as for any worker.
                first = server.cancel_task(task_id)
                assert first.get("status") == 200, first
                assert first.get("body") == {"ok": True, "task_id": task_id}, first
                rows = _cancel_forensics(oracle, task_id)
                assert _forensic_events(rows)[:3] == ["requested", "claimed", "settled"], rows
                requested = next(r for r in rows if r.get("event") == "requested")
                assert (requested.get("source"), requested.get("scope")) == ("http_single", "single"), requested
                settled = [r for r in rows if r.get("event") == "settled"]
                assert len(settled) == 1 and settled[0].get("outcome") == "cancelled", settled
                assert settled[0].get("request_id") == requested.get("request_id"), (requested, settled)
                assert _mailbox_controls(server.data_root, task_id) == [], "a worker carrier is stopped, not asked"
                assert wait_until(lambda: not pid_is_alive(carrier), 30), f"the carrier {carrier} outlived the stop"
                assert wait_until(lambda: task_id not in oracle.cancel_intents(), 60), oracle.cancel_intents()

                # 3. ENDED with ZERO further model rounds: the held round never completes.
                stored = wait_durable_result(oracle, task_id, timeout=180)
                assert stored.get("status") == "cancelled", stored
                origin = stored.get("cancel_origin") or {}
                assert (origin.get("source"), origin.get("request_id")) == (
                    "http_single", requested.get("request_id")), origin
                done = wait_until(
                    lambda: [r for r in oracle.events("task_done")
                             if str(r.get("task_id") or "") == task_id] or None, 60)
                assert done and done[-1].get("status") == stored.get("status"), done
                gate.release.set()  # the killed carrier's held request has nowhere to answer

                # 4. The chat CONCLUDED on the same surfaces the SPA reads.
                typing = wait_until(lambda: frames.find(type="typing", activity_id=task_id) or None, 30)
                assert typing, [f.get("type") for f in frames.frames]
                assert typing[0].get("kind") == "direct_chat", typing[0]
                assert typing[0].get("client_message_id") == client_message_id, typing[0]
                # The keyed FINAL frame: the host-authored stop receipt of THIS request, a typed
                # SYSTEM row keyed by the turn id so the client concludes exactly this activity.
                delivery_id = f"cancel:{task_id}:{requested.get('request_id')}"
                final = wait_until(
                    lambda: [f for f in frames.find(type="chat", task_id=task_id)
                             if not f.get("is_progress") and f.get("delivery_id") == delivery_id] or None, 60)
                assert final, "no keyed stop receipt for the turn arrived over /ws"
                assert final[-1].get("role") == "system" and final[-1].get("system_type") == "cancel_receipt", final[-1]
                assert wait_until(lambda: not _direct_activities(server, client_message_id), 60), (
                    "the direct turn is still an active chat activity after it ended")
                assert wait_until(lambda: task_id not in _running_task_ids(server), 60)
                # The published queue drops the row on the supervisor's NEXT tick: custody's removal is
                # a mark (supervisor/queue_publication.py, `persist_queue_snapshot`), the tick writes the
                # document at the end of its iteration and drains `task_done` at the start of the next,
                # so a Stop settled right after one publication shows `task_done` and every drain above
                # about a second before the document follows. Waited like S21's read of the same document.
                assert wait_until(lambda: task_id not in oracle.running_ids(), 60), (
                    oracle.queue_snapshot().get("running"))
                chat_rows = wait_until(
                    lambda: [r for r in oracle._jsonl("logs/chat.jsonl")
                             if str(r.get("task_id") or "") == task_id
                             and r.get("delivery_id") == delivery_id] or None, 60)
                assert chat_rows and chat_rows[-1].get("direction") == "system", chat_rows
                # ...and the durable terminal projection the chat renders for the
                # card: the root's terminal summary row, final, with the same status.
                terminal_rows = wait_until(
                    lambda: [r for r in oracle._jsonl("logs/chat.jsonl", type_filter="task_summary")
                             if str(r.get("task_id") or "") == task_id
                             and r.get("summary_kind") == "terminal_root_projection"] or None, 60)
                assert terminal_rows, "no terminal_root_projection summary row for the turn"
                assert terminal_rows[-1].get("outcome_final") is True, terminal_rows[-1]
                assert terminal_rows[-1].get("status") == stored.get("status"), terminal_rows[-1]

            # 5. A LATER stop is the typed 404 with nothing re-armed.
            before = _terminal_fields(oracle, task_id)
            again = server.cancel_task(task_id)
            assert again.get("status") == 404, again
            assert (again.get("body") or {}).get("error") == S26_NOT_ACTIVE, again
            assert (again.get("body") or {}).get("task_id") == task_id, again
            assert task_id not in oracle.cancel_intents()
            assert _mailbox_controls(server.data_root, task_id) == []
            requested_rows = [r for r in _cancel_forensics(oracle, task_id)
                              if r.get("event") == "requested"]
            assert len(requested_rows) == 1, requested_rows
            assert _terminal_fields(oracle, task_id) == before

            # ZERO further model rounds, end to end: one held round, nothing after it,
            # even once the gate let the held request go.
            assert gate.matched == 1 and gate.held == 1, (gate.matched, gate.held)
            assert gate.timed_out is False, "the gate expired: the scenario lost its mid-round premise"
            # The released request of the killed carrier may still be recorded as its one round;
            # nothing else (no further round, no finalization, no summary) ever reached the model.
            assert set(stub.kinds()) <= {"agent"} and stub.kinds().count("agent") <= 1, stub.kinds()
            assert not stub.script_consumed(), "the keepalive script was run down: the stop was not honored"
            # ...and ZERO paid POST-TASK work: the matcher above counts tool-bearing
            # rounds only, so a tool-less summary/reflection call after the stop is
            # invisible to it. Read the durable artifacts instead. The sync point
            # is structural — the stored terminal carries no open checkpoint, so no
            # synthesis worker was ever dispatched — and the row probes get a short
            # bounded grace so a leaked worker's write cannot slip past a fast reader.
            assert not _post_task_synthesis_is_open(stored), stored.get("root_phase_checkpoint")
            leaked = wait_until(lambda: _post_stop_synthesis_rows(oracle, task_id) or None, 3)
            assert not leaked, leaked
        finally:
            gate.release.set()  # never leave the stub's request thread parked on a failed scenario
            server.stop()
