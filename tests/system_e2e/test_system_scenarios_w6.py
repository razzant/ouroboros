"""S26 — v7 follow-up Ф1 (sprint plan §5.1 Ф1-B, D-05): the owner stops an
in-process DIRECT-CHAT turn.

THE CLASS (rc.7 QA regress, item 010). A message typed into the main chat runs as
a DIRECT-CHAT TURN: minted by ``worker_chat_lane.py`` with an 8-hex id and executed
on the long-lived chat agent INSIDE the supervisor — no queue row, no worker
process. ``agent.py`` writes its ordinary durable ``running`` row, so
``GET /api/tasks?status=running`` listed the turn while every queue-keyed owner
control answered 404 "task not found or not active", and the spend kept growing
until ``/restart``. Ф1-B made ``supervisor.workers.direct_chat_turn`` the ONE
ownership reader every ingress resolves a live direct turn through; custody stops
the turn COOPERATIVELY — the typed ``finalize_now`` control carrying
``REASON_OWNER_STOPPED_DIRECT_TURN`` on the canonical owner mailbox, drained at the
loop's next round boundary (``_handle_direct_turn_hard_stop``: ZERO further model
calls, the post-task synthesis included — the hard stop records the existing
``_skip_post_task_synthesis`` marker the pipeline honours, so no summary or
reflection bills after the stop). ``tests/test_direct_chat_turn_owner_control.py`` pins the seams against a
stubbed agent; THIS scenario pins the whole path on a real server, keyless, over
the same WS chat and HTTP cancel surfaces the SPA drives.

DETERMINISM. "Mid-turn" is not a sleep. The loopback model HOLDS the turn's first
tool-bearing round on a ``ModelGate`` (harness): the request thread blocks until the
scenario releases it, so the owner's stop provably lands while the turn is inside
a model round, the endpoint's typed "still live" is the ONLY answer it can give
(a turn cannot reach its round boundary while its round is held), and the armed
control, the toast and the marker are all read back DURABLY before the release.
Everything else is durable-event polling over ``ArtifactOracle`` readers.

WHAT S26 ASSERTS, in order:
1. IN FLIGHT and ADDRESSABLE — inside a model round (``gate.arrived``), the turn is
   in ``/api/state.active_chat_activities`` as kind ``direct_chat`` linked to the
   client_message_id, listed by ``GET /api/tasks?status=running`` (its durable
   running row), absent from the queue snapshot (the class: no queue row), and
   ``events.jsonl`` carries its durable ``task_received`` row with
   ``_is_direct_chat=true`` and the client_message_id link (``task_started`` is a
   session-only live log, never persisted);
2. STOP-NOW over the UI's ``POST /api/tasks/{id}/cancel`` (empty body = immediate)
   mid-round is NOT 404: the typed 503 "still live" naming the task; the
   cooperative control is ARMED — exactly one ``finalize_now`` row with the
   direct-turn reason in the canonical mailbox; the durable intent is open with
   the forensic trail requested(http_single, single) → claimed → claim_released
   ("has not reached its next step"); the owner toast went out as a progress frame
   carrying the host-attested ``cancelable`` marker (the direct-turn branch of the
   delivery seam) — durable in progress.jsonl and live on /ws; the durable row is
   still honestly ``running`` (no fabricated cancelled over a live turn);
3. a SECOND stop-now while armed is typed and idempotent — 503 again, still ONE
   control, still ONE toast, no model round in between (stamp latch, no re-arm);
4. RELEASE — the turn ends at its next step under the owner-stop reason with ZERO
   further model rounds: the durable terminal carries
   ``reason_code=owner_requested_finalization`` and the typed no-further-work
   answer, ``task_done`` names the same status, the gate matched exactly ONE round
   and the keepalive script was never run down; the DURABLE artifacts show no
   post-stop paid work either — the tool-bearing gate cannot see a tool-less
   summary/reflection call, so the pin reads the task result (no open
   ``root_phase_checkpoint``), chat.jsonl (no ``authored_root_summary`` row) and
   ``task_reflections.jsonl`` (no row) for the turn;
5. the chat CONCLUDED (no "Working…" forever) — over the SAME /ws the SPA opens
   the turn announced itself (typing frame: activity_id = task id, kind
   ``direct_chat``, the client_message_id link), the toast frame carried
   ``cancelable``, the keyed FINAL system frame (the host-authored stop notice;
   task_id = the turn, not
   progress) arrived, the activity left ``/api/state``, the running list no
   longer names the turn, and the final landed in chat.jsonl under the task id;
6. custody SETTLED the intent against the turn's OWN terminal (the sweep's
   ``settled`` row: outcome ``already_settled``, detail = the stored status) and
   the projection drained;
7. a LATER stop is the typed 404 with nothing re-armed: no new intent, no new
   control, no new forensic request, the terminal fields unchanged.

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


@pytest.mark.serial  # a real loopback port (ScriptedStubModel) — the serial pass
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
# Literals of the tree under test (worker_chat_lane.stop_direct_chat_turn's toast,
# loop_round_limits._handle_direct_turn_hard_stop's typed no-candidate answer).
S26_TOAST_TEXT = "The owner stopped chat turn"
S26_STOP_TEXT = "The owner stopped this chat turn"
S26_STILL_LIVE = "still live"
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


def _toast_rows(oracle: ArtifactOracle, task_id: str) -> list:
    """The owner-stop toast as persisted (progress frames land in progress.jsonl)."""
    return [row for row in oracle._jsonl("logs/progress.jsonl")
            if str(row.get("task_id") or "") == task_id
            and S26_TOAST_TEXT in str(row.get("text") or "")]


def _post_task_synthesis_is_open(stored: dict) -> bool:
    """The durable checkpoint the boot reconciler re-dispatches (and re-pays) from."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open

    checkpoint = stored.get("root_phase_checkpoint")
    status = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    return post_task_synthesis_is_open(status)


def _post_stop_synthesis_rows(oracle: ArtifactOracle, task_id: str) -> list:
    """Durable traces of the post-task worker for the turn: the authored summary
    row (written even for a trivial turn, so its absence means the phase never
    ran) and any reflection row."""
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
@pytest.mark.serial
def test_s26_direct_chat_turn_owner_stop_mid_round_is_typed_and_ends_the_turn(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from ouroboros.cancel_intents import INTENT_REQUESTED, STOP_POLICY_IMMEDIATE, stop_policy
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION

    root = tmp_path_factory.mktemp("s26")
    gate = ModelGate(_is_turn_round, timeout=240)
    # A SHORT keepalive script on purpose: were the stop control ever not drained,
    # the turn would run these rounds and end on the stub's final answer — the
    # round-count pin below then names that failure instead of a long burn.
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
                activity = wait_until(
                    lambda: (_direct_activities(server, client_message_id) or [None])[0], 30)
                assert activity, "the direct turn is not in /api/state.active_chat_activities"
                task_id = str(activity.get("activity_id") or "")
                assert task_id and activity.get("phase") == "thinking", activity
                assert wait_until(lambda: task_id in _running_task_ids(server), 30), (
                    "the live direct turn is not listed by GET /api/tasks?status=running")
                assert task_id not in oracle.running_ids(), "a direct turn must have no queue row"
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

                # 2. STOP-NOW over the UI's endpoint, mid-round: typed, never 404,
                #    the cooperative control ARMED.
                first = server.cancel_task(task_id)
                assert first.get("status") == 503, first
                first_body = first.get("body") or {}
                assert S26_STILL_LIVE in str(first_body.get("error") or ""), first
                assert first_body.get("task_id") == task_id, first
                controls = _mailbox_controls(server.data_root, task_id)
                assert len(controls) == 1, controls
                control_reason = str(controls[0].get("text") or "").splitlines()[0].strip()
                assert control_reason == REASON_OWNER_STOPPED_DIRECT_TURN, controls
                # The durable intent: minted by the legacy immediate shape (the
                # tree's own reader says IMMEDIATE), claim released back to
                # ``requested`` for the sweep with the typed "still live" error.
                intent = oracle.cancel_intents().get(task_id) or {}
                assert intent, oracle.cancel_intents()
                assert stop_policy(intent) == STOP_POLICY_IMMEDIATE, intent
                assert intent.get("state") == INTENT_REQUESTED, intent
                assert "not reached its next step" in str(intent.get("last_error") or ""), intent
                rows = _cancel_forensics(oracle, task_id)
                events = _forensic_events(rows)
                assert events[:3] == ["requested", "claimed", "claim_released"], events
                requested = next(r for r in rows if r.get("event") == "requested")
                assert requested.get("source") == "http_single", requested
                assert requested.get("scope") == "single", requested
                released = next(r for r in rows if r.get("event") == "claim_released")
                assert "not reached its next step" in str(released.get("error") or ""), released
                # The owner toast: a progress frame addressed to the turn's card
                # that the delivery seam stamps with the host-attested marker —
                # durable, then over the wire.
                toasts = wait_until(lambda: _toast_rows(oracle, task_id) or None, 30)
                assert toasts and len(toasts) == 1, toasts
                assert toasts[0].get("cancelable") is True, toasts[0]
                # The typed incident vocabulary rides the toast.
                assert toasts[0].get("task_incident") == REASON_OWNER_STOPPED_DIRECT_TURN, toasts[0]
                toast_frames = wait_until(
                    lambda: [f for f in frames.find(type="chat", task_id=task_id, is_progress=True)
                             if S26_TOAST_TEXT in str(f.get("content") or "")] or None, 30)
                assert toast_frames and toast_frames[0].get("cancelable") is True, toast_frames
                # No fabricated terminal over a live turn.
                assert oracle.task_result(task_id).get("status") == "running", oracle.task_result(task_id)

                # 3. A SECOND stop while armed: typed and idempotent.
                second = server.cancel_task(task_id)
                assert second.get("status") == 503, second
                assert S26_STILL_LIVE in str((second.get("body") or {}).get("error") or ""), second
                assert len(_mailbox_controls(server.data_root, task_id)) == 1, "the stop was re-armed"
                assert len(_toast_rows(oracle, task_id)) == 1, "the owner was toasted twice"
                assert gate.matched == 1 and gate.held == 1, (gate.matched, gate.held)

                # 4. RELEASE: the turn ends at its next step, ZERO further rounds.
                gate.release.set()
                stored = wait_durable_result(oracle, task_id, timeout=180)
                assert stored.get("reason_code") == REASON_OWNER_REQUESTED_FINALIZATION, stored
                assert stored.get("status") == "failed", stored
                assert S26_STOP_TEXT in str(stored.get("result") or ""), stored
                done = wait_until(
                    lambda: [r for r in oracle.events("task_done")
                             if str(r.get("task_id") or "") == task_id] or None, 60)
                assert done and done[-1].get("status") == stored.get("status"), done

                # 5. The chat CONCLUDED on the same surfaces the SPA reads.
                typing = wait_until(lambda: frames.find(type="typing", activity_id=task_id) or None, 30)
                assert typing, [f.get("type") for f in frames.frames]
                assert typing[0].get("kind") == "direct_chat", typing[0]
                assert typing[0].get("client_message_id") == client_message_id, typing[0]
                # The keyed FINAL frame: a HOST-authored terminal (terminal_origin
                # host_notice) is published as a typed SYSTEM row — the owner's
                # stop notice, never Ouroboros's own speech — keyed by the turn id
                # so the client concludes exactly this activity.
                final = wait_until(
                    lambda: [f for f in frames.find(type="chat", task_id=task_id)
                             if not f.get("is_progress")
                             and S26_STOP_TEXT in str(f.get("content") or "")] or None, 60)
                assert final, "no keyed final frame for the turn arrived over /ws"
                assert final[-1].get("role") == "system", final[-1]
                assert wait_until(lambda: not _direct_activities(server, client_message_id), 60), (
                    "the direct turn is still an active chat activity after it ended")
                assert wait_until(lambda: task_id not in _running_task_ids(server), 60)
                chat_rows = wait_until(
                    lambda: [r for r in oracle._jsonl("logs/chat.jsonl")
                             if str(r.get("task_id") or "") == task_id
                             and S26_STOP_TEXT in str(r.get("text") or "")] or None, 60)
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

            # 6. Custody settles against the turn's OWN terminal; the projection drains.
            assert wait_until(lambda: task_id not in oracle.cancel_intents(), 120), (
                oracle.cancel_intents())
            settled = [r for r in _cancel_forensics(oracle, task_id) if r.get("event") == "settled"]
            assert len(settled) == 1, settled
            assert settled[0].get("outcome") == "already_settled", settled[0]
            assert settled[0].get("detail") == stored.get("status"), settled[0]

            # 7. A LATER stop is the typed 404 with nothing re-armed.
            before = _terminal_fields(oracle, task_id)
            third = server.cancel_task(task_id)
            assert third.get("status") == 404, third
            assert (third.get("body") or {}).get("error") == S26_NOT_ACTIVE, third
            assert (third.get("body") or {}).get("task_id") == task_id, third
            assert task_id not in oracle.cancel_intents()
            assert len(_mailbox_controls(server.data_root, task_id)) <= 1
            requested_rows = [r for r in _cancel_forensics(oracle, task_id)
                              if r.get("event") == "requested"]
            assert len(requested_rows) == 1, requested_rows
            assert _terminal_fields(oracle, task_id) == before

            # ZERO further model rounds, end to end: one held round, nothing after it.
            assert gate.matched == 1 and gate.held == 1, (gate.matched, gate.held)
            assert gate.timed_out is False, "the gate expired: the scenario lost its mid-round premise"
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
