"""#2012: one live-custody owner for every Stop door, with the Presence witness typed and three-state.

``task_has_live_ownership`` counts an unsettled inline Presence turn whose author census witnesses
it, or cannot rule it out (an unreadable census), as live custody, so the single and cascade Stop
and Wrap up doors read one owner. Only such a row reads the census. An unknown census keeps a Stop
pending and is recorded as unknown; it is never reported as a witnessed run, so it never hides the
owner-visible ``task_cancel_cascade_noop`` incident. Ordinary pooled tasks never read the census.

The rows are synthetic (no turn runs); the live turn itself is ``tests/test_presence_owner_stop.py``.
Only the census source (``conversation_lease.live_authors``) is replaced, so every caller above it
is the production code; the overlap witness also delays, never replaces, the episode's control write.
"""
from __future__ import annotations

import json
import queue as queues
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import cancel_intents as ci
from ouroboros.task_results import load_task_result, write_task_result

UNKNOWN = "Presence liveness unknown: its author census is unreadable"
WITNESSED = "Presence turn has not reached its next stop check"


@pytest.fixture
def doors(tmp_path, monkeypatch):
    from ouroboros import conversation_lease
    from ouroboros.gateway.tasks import api_task_cancel
    from supervisor import queue, workers

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue, "_emit_cancel_task_done", lambda *_a, **_k: None)
    bus = queues.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda *_a, **_k: bus)
    census = SimpleNamespace(reads=0, state="absent", authors=[])

    def live_authors(*, drive_root=None):
        census.reads += 1
        if census.state == "unknown":
            raise OSError("author census unreadable")
        return list(census.authors)

    monkeypatch.setattr(conversation_lease, "live_authors", live_authors)
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = tmp_path

    def press(task_id, *, cascade=False, policy="", action="owner-stop-1"):
        body = {**({"cascade": True} if cascade else {}), **({"stop_policy": policy} if policy else {}),
                "stop_action_id": action}
        with TestClient(app) as client:
            response = client.post(f"/api/tasks/{task_id}/cancel", json=body)
        for thread in threading.enumerate():  # a Wrap up arms its episode off the request thread
            if thread.name.startswith("owner-stop-"):
                thread.join(30)
        return response

    return SimpleNamespace(root=tmp_path, queue=queue, census=census, press=press, bus=bus)


def _row(root, task_id, kind):
    from ouroboros.task_status import execution_owner_record

    write_task_result(root, task_id, "running", result="Task is running.", chat_id=7,
                      execution_owner=execution_owner_record(root, {"id": task_id}, kind),
                      **({"_is_direct_chat": True} if kind == "presence" else {}))


def _census(doors, state, task_id):
    doors.census.state = state
    doors.census.authors = [task_id] if state == "witnessed" else []


def _incidents(root, task_id):
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row["type"] for row in rows if row.get("task_id") == task_id
            and str(row.get("type", "")).startswith("task_cancel_cascade")]


@pytest.mark.parametrize("state", ["witnessed", "absent", "unknown"])
def test_the_shared_owner_reads_a_presence_witness_in_three_states(doors, state):
    from ouroboros.gateway.tasks import _cascade_waits_only_on_runs
    from supervisor.queue import task_has_live_ownership

    _row(doors.root, "presence-turn", "presence")
    _census(doors, state, "presence-turn")
    # Custody: a witnessed or unknown stack stays owned; only a readable census without it is gone.
    assert task_has_live_ownership("presence-turn") is (state != "absent")
    assert doors.census.reads == 1
    # The incident: only a witnessed run waits; an unknown census proves nothing.
    assert _cascade_waits_only_on_runs("presence-turn") is (state == "witnessed")
    assert doors.census.reads == 2


@pytest.mark.parametrize("state", ["witnessed", "unknown"])
def test_a_stop_on_a_turn_the_census_cannot_rule_out_stays_pending_and_says_why(doors, state):
    _row(doors.root, "presence-turn", "presence")
    _census(doors, state, "presence-turn")
    single = doors.press("presence-turn")
    assert single.status_code == 202, single.text
    assert single.json()["cancel_state"] == "pending"
    assert ci.active_intent(doors.root, "presence-turn")["last_error"] == (WITNESSED if state == "witnessed" else UNKNOWN)
    assert load_task_result(doors.root, "presence-turn")["status"] == "running"  # its runner's row
    assert _incidents(doors.root, "presence-turn") == []
    tree = doors.press("presence-turn", cascade=True, action="owner-stop-2")
    assert tree.status_code == 202, tree.text
    # A witnessed run waits for its runner; an unknown one is never reported as waiting.
    assert _incidents(doors.root, "presence-turn") == ([] if state == "witnessed" else ["task_cancel_cascade_noop"])


def test_a_presence_row_the_census_readably_lacks_keeps_every_door_closed(doors):
    _row(doors.root, "presence-turn", "presence")
    _census(doors, "absent", "presence-turn")
    assert doors.press("presence-turn").status_code == 404
    assert doors.press("presence-turn", cascade=True, action="owner-stop-2").status_code == 404
    wrap = doors.press("presence-turn", cascade=True, policy="finalize_then_cancel", action="owner-wrap-up")
    assert wrap.status_code == 404, wrap.text
    assert ci.active_intents(doors.root) == {}
    assert load_task_result(doors.root, "presence-turn")["status"] == "running"


def test_wrap_up_reaches_a_presence_turn_through_the_shared_owner(doors):
    from ouroboros.owner_mailbox import drain_owner_entries

    _row(doors.root, "presence-turn", "presence")
    _census(doors, "witnessed", "presence-turn")
    wrap = doors.press("presence-turn", cascade=True, policy="finalize_then_cancel", action="owner-wrap-up")
    assert wrap.status_code == 202 and wrap.json()["stop_policy"] == "finalize_then_cancel", wrap.text
    intent = ci.active_intent(doors.root, "presence-turn")
    [control] = [entry for entry in drain_owner_entries(doors.root, "presence-turn", set())
                 if entry["kind"] == "finalize_now"]
    assert control["msg_id"] == f"ownerstop:{intent['request_id']}"
    # The sweep replays the same episode without a second control or toast.
    from supervisor.owner_stop import begin_graceful_stop
    begin_graceful_stop("presence-turn")
    assert len([entry for entry in drain_owner_entries(doors.root, "presence-turn", set())
                if entry["kind"] == "finalize_now"]) == 1
    toasts = []
    while not doors.bus.empty():
        toasts.append(doors.bus.get_nowait())
    assert [(event["system_type"], event["chat_id"]) for event in toasts] == [("finalization_notice", 7)]


@pytest.mark.parametrize("kind", ["presence", "pooled", "direct"])
def test_overlapping_wrap_up_passes_write_one_control_and_one_notice(doors, monkeypatch, kind):
    """The ingress kick-off and a sweep tick overlap on one live episode of each shape. The witness
    is causal: the first pass to write holds its write until the other pass has ended, so a second
    pass that armed too would write and announce the episode again; the overlapping pass only holds."""
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor import owner_stop, task_reaper
    from tests.test_direct_chat_turn_owner_control import TURN_ID, _live_chat_agent

    task_id = TURN_ID if kind == "direct" else f"{kind}-turn"
    _row(doors.root, task_id, kind)
    if kind == "presence":
        _census(doors, "witnessed", task_id)
    elif kind == "pooled":
        doors.queue.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 7, "root_task_id": task_id,
                                                 "delegation_role": "root"}, "started_at": time.time()}
    else:
        _live_chat_agent(monkeypatch, task_id)
    ci.request_cancel(doors.root, task_id, source="http_graceful", requested_stop_policy="finalize_then_cancel")
    write, writers, overlapped = task_reaper.request_finalization_grace, [], []
    other_ended, entry = threading.Event(), threading.Lock()

    def first_write_waits_for_the_other_pass(*args, **kwargs):
        with entry:
            writers.append(threading.current_thread().name)
            first = len(writers) == 1
        if first:
            overlapped.append(other_ended.wait(timeout=20))
        return write(*args, **kwargs)

    def arm_pass():
        owner_stop.begin_graceful_stop(task_id)
        other_ended.set()  # the first writer's own pass cannot end while it waits

    monkeypatch.setattr(task_reaper, "request_finalization_grace", first_write_waits_for_the_other_pass)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for call in [pool.submit(arm_pass) for _ in range(2)]:
            call.result(timeout=60)
    notices = []
    while not doors.bus.empty():
        notices.append(doors.bus.get_nowait())
    assert [(event["system_type"], event["task_id"]) for event in notices] == [("finalization_notice", task_id)]
    assert len(writers) == 1, writers
    assert overlapped == [True], "the other pass must end inside the first pass's write, or nothing overlapped"
    [control] = drain_owner_entries(doors.root, task_id, set())
    assert control["msg_id"] == f"ownerstop:{ci.active_intent(doors.root, task_id)['request_id']}"


def test_an_ordinary_pooled_task_reads_no_census_at_any_door(doors):
    """The positive pooled control: a RUNNING pooled task is owned through its queue row, its
    Wrap up arms through that row, and an unreadable census is never consulted."""
    from ouroboros.gateway.tasks import _cascade_waits_only_on_runs
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor.queue import task_has_live_ownership

    _census(doors, "unknown", "")
    _row(doors.root, "pooled-live", "pooled")
    task = {"id": "pooled-live", "type": "task", "chat_id": 7, "root_task_id": "pooled-live", "delegation_role": "root"}
    doors.queue.RUNNING["pooled-live"] = {"task": task, "started_at": time.time()}
    assert task_has_live_ownership("pooled-live") is True
    wrap = doors.press("pooled-live", policy="finalize_then_cancel", action="owner-wrap-up")
    assert wrap.status_code == 202 and wrap.json()["stop_policy"] == "finalize_then_cancel", wrap.text
    assert doors.queue.RUNNING["pooled-live"]["finalization_control_msg_id"] == (
        f"ownerstop:{ci.active_intent(doors.root, 'pooled-live')['request_id']}")
    assert [entry["kind"] for entry in drain_owner_entries(doors.root, "pooled-live", set())] == ["finalize_now"]
    # A pooled tree whose cascade cancelled nothing keeps its incident: the census is not its witness.
    _row(doors.root, "pooled-orphan", "pooled")
    assert _cascade_waits_only_on_runs("pooled-orphan") is False
    assert doors.census.reads == 0
