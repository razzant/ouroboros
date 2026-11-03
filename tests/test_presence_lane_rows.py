"""A Presence turn's conversation-lane row is its place's conversation, exactly as the in-server author is (#2604).

Every Host Presence turn is a RUNNING conversation-lane row from its admission (``presence_carrier.dispatch_presence_turn``
-> ``conversation_lane.admit_turn``), published in the queue snapshot while its author runs. The roster, the mind's
live works, the Main router and the peer receipt read that snapshot; #2604's own tests run the author inline with no
lane row. Here a REAL author (``tests.test_presence_project_activity._live_turn``) runs while the lane admits and
dispatches its row (review N2): an unmoved turn is never a roster row, is the live work "busy with its own
conversation", is not steerable by the Main router and answers a peer with the Presence receipt; a moved one is
listed once, as the place's turn (``inline_presence``), never as an ordinary pooled root.
"""

from __future__ import annotations

import pytest

from tests.test_presence_project_activity import PROJECT, _live_turn, _move, _routing_ctx, root as root


def _admit_lane_row(task):
    """The lane's own admission and dispatch of this turn's row, as ``dispatch_presence_turn`` leaves it."""
    from ouroboros.queue_observation import load_queue_document
    from supervisor import conversation_lane, queue, workers
    from tests.conversation_carrier_support import install_lane_spare

    spare = install_lane_spare(workers, 90)
    seed = {"id": task["id"], "type": "presence", "source": "presence", "chat_id": task["chat_id"],
            "text": str(task.get("text") or ""), "_is_direct_chat": True, "_presence_turn": True,
            "metadata": dict(task.get("metadata") or {})}
    with queue._queue_lock, workers._repo_writer_gate_lock:
        meta = conversation_lane.admit_turn(seed, chat_id=int(seed["chat_id"]), client_message_id="presence-event",
                                            quiet=True, reply_expected=False)
    assert meta["lane"] == "conversation" and meta["worker_id"] == spare.wid  # a real, dispatched lane row
    published = {row["id"]: row for row in load_queue_document(workers.DRIVE_ROOT)["running"]}
    assert published[task["id"]]["task"]["_execution_owner"]["kind"] == "presence"  # what the readers see


def test_a_presence_turns_lane_row_is_its_places_conversation_in_every_mind_view(root):
    from ouroboros.live_works import live_works_present
    from ouroboros.peer_roster import independent_roots, render_roster_note
    from ouroboros.server_routing_context import addressable_roots
    from ouroboros.tools.core import _forward_to_worker
    from ouroboros.tools.registry import ToolContext

    seen = {}

    def scenario(author, task):
        _admit_lane_row(task)
        seen["listed"] = [row["task_id"] for row in independent_roots(root)["roots"]]
        seen["addressable"] = [row for row in addressable_roots(_routing_ctx(root)).rows() if row["task_id"] == task["id"]]
        present = live_works_present(root)
        seen["works"] = [work["task_id"] for work in present["works"]]
        seen["other"] = [row for row in present.get("other_live_work") or [] if row["work_id"] == task["id"]]
        peer = ToolContext(repo_dir=root, drive_root=root, budget_drive_root=root, task_id="peer-root", task_metadata={})
        seen["receipt"] = _forward_to_worker(peer, task["id"], "peer context")
        _move(root, task["id"], PROJECT)
        roster = independent_roots(root)
        seen["moved"] = [row for row in roster["roots"] if row["task_id"] == task["id"]]
        seen["note"] = render_roster_note(roster)
        seen["moved_addressable"] = [row for row in addressable_roots(_routing_ctx(root)).rows()
                                     if row["task_id"] == task["id"]]
        present = live_works_present(root)
        seen["moved_works"] = ([work["task_id"] for work in present["works"]],
                               [row["work_id"] for row in present.get("other_live_work") or []])

    task_id = _live_turn(root, scenario)
    # Unmoved: no roster row, not the Main router's to steer, the place's live turn, the Presence receipt.
    assert task_id not in seen["listed"] and seen["addressable"] == [] and task_id not in seen["works"]
    [row] = seen["other"]
    assert row["kind"] == "inline_presence" and row["root_status"] == "running"
    assert "not that its model read it" in seen["receipt"] and '"kind": "presence"' in seen["receipt"]
    assert "it reads it at its next checkpoint" not in seen["receipt"]
    # Moved: one listed Project work, the place's turn, never the lane row as an ordinary root.
    [moved] = seen["moved"]
    assert moved["inline_presence"] is True and moved["project_id"] == PROJECT
    assert f"{task_id} · " in seen["note"] and "live place conversation turn" in seen["note"]
    assert all(row.get("inline_presence") for row in seen["moved_addressable"])  # a local author's own door only
    assert task_id in seen["moved_works"][0] and task_id not in seen["moved_works"][1]


def test_pause_of_a_presence_turns_lane_row_refuses_at_every_door_and_changes_nothing(root):
    """Review N6: the lane row makes the turn a found task, so the Pause door accepted it and the author held its
    conversation lock, its Presence slot, a carrier and the Host's reservation until Resume. Nothing saves a
    Presence turn's stack (#2566, chapter 12): every door refuses it typed and writes nothing, as for the
    in-server author."""
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from supervisor import queue as q
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_presence_project_activity import _pause_doors

    seen = {}

    def scenario(author, task):
        tid = task["id"]
        _admit_lane_row(task)
        before = load_task_result(root, tid, strict=True)
        seen["owner"] = request_owner_pause(tid, request_id="pause-lane")
        seen["http"], seen["mind"] = _pause_doors(root, tid, "lane")
        seen["unchanged"] = load_task_result(root, tid, strict=True) == before
        seen["fence"], seen["latch"] = read_fence(root, tid), tid in q.BUDGET_ROOT_FENCES

    _live_turn(root, scenario)
    assert seen["owner"] == seen["mind"] == {"ok": False, "error": "presence_pause_unsupported"}
    assert seen["http"] == (409, "presence_pause_unsupported")
    assert not seen["fence"] and seen["latch"] is False and seen["unchanged"] is True


# The server's census while a carried turn's author starts in its carrier (another process): the server holds the
# turn's lane row, the author's witness names no room yet (a non-local preparing author reads as an empty row).
_CENSUS_BESIDE_A_STARTING_CARRIED_AUTHOR = r'''
import json, sys
from pathlib import Path
import pytest
root, tid = Path(sys.argv[1]), sys.argv[2]
from tests._budget_pause_exact_helpers import _install_queue
_install_queue(root, pytest.MonkeyPatch())
import ouroboros.config
ouroboros.config.DATA_DIR = root
from supervisor import message_bus, workers
from tests.conversation_carrier_support import lane_row
from tests.test_presence_project_activity import _Bridge, _state
message_bus._BRIDGE = _Bridge()
lane_row(workers, {"id": tid, "type": "presence", "source": "presence", "chat_id": 5, "text": "hi",
                   "_is_direct_chat": True, "_presence_turn": True, "_execution_owner": {"kind": "presence"}}, wid=97)
census, works, complete = _state(root)
print(json.dumps({"complete": complete, "kinds": [row.get("kind") for row in census if row.get("activity_id") == tid]}))
'''


def test_a_starting_carried_presence_turn_leaves_the_census_whole(root):
    """Review N12: the server's census lists a carried turn from its lane row while its author starts in its carrier;
    the author's room-less witness there is that same known turn, never an unknown row (the Restart dialog's)."""
    import json
    import os
    import subprocess
    import sys

    from tests.test_presence_project_activity import REPO, _starting_turn

    seen = {}

    def starting(tid):
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        child = subprocess.run([sys.executable, "-c", _CENSUS_BESIDE_A_STARTING_CARRIED_AUTHOR, str(root), tid],
                               cwd=str(REPO), env=env, text=True, capture_output=True, timeout=90)
        assert child.returncode == 0, child.stdout + child.stderr
        seen.update(json.loads(child.stdout.strip().splitlines()[-1]))

    _starting_turn(root, starting)
    assert seen == {"complete": True, "kinds": ["presence"]}


def test_an_inline_executors_turn_is_listed_once_as_its_live_actor(root):
    """Census minor (a): an inline executor's turn is both a registered actor and a lane row; the census lists it
    once, as the actor (its own phase and model wait), never a second lane-shaped row beside it."""
    from ouroboros.gateway.state import _direct_turns_snapshot_safe
    from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED
    from supervisor import workers
    from supervisor.active_activity import get_direct_activity_registry
    from tests.conversation_carrier_support import lane_row

    registry = get_direct_activity_registry()
    registry.clear()
    try:
        registry.register("inline-turn", 1, kind="direct_chat", phase="working")
        lane_row(workers, {"id": "inline-turn", "type": "task", "chat_id": 1, "text": "hi", "_is_direct_chat": True},
                 executor=EXECUTOR_INLINE_DEGRADED, pid=None)
        rows = [row for row in _direct_turns_snapshot_safe() if row.get("activity_id") == "inline-turn"]
        assert rows == [row for row in registry.snapshot() if row.get("activity_id") == "inline-turn"]
    finally:
        registry.clear()


def _checkpoint(root, tid, attempt=1):
    """What a turn's loop writes at its first ``ready`` boundary (identity fields only): saved work to a stop door."""
    import json

    from ouroboros.working_checkpoint import checkpoint_path

    checkpoint_path(root, tid, attempt, create=True).write_text(json.dumps(
        {"task_id": tid, "task_attempt": attempt, "working": {"boundary": "ready", "seq": 1,
                                                               "saved_at": "2026-11-02T00:00:00+00:00"}}))


def _lane_turn(root, kind, task_id, wid):
    """A running lane row and its durable RUNNING result: an owner turn, or a correspondent's Presence turn."""
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import execution_owner_record
    from supervisor import workers
    from tests.conversation_carrier_support import lane_row

    presence = {"_presence_turn": True, "source": "presence",
                "metadata": {"source": "presence", "presence": {"binding_id": "b1"}}} if kind == "presence" else {}
    task = {"id": task_id, "type": "task", "chat_id": 5, "text": "hi", "_is_direct_chat": True,
            "_execution_owner": {"kind": kind}, **presence}
    lane_row(workers, task, wid=wid)
    write_task_result(root, task_id, "running", chat_id=5, _is_direct_chat=True, result="Task is running.",
                      execution_owner=execution_owner_record(root, task, kind),
                      **({"source": "presence", "metadata": presence["metadata"]} if presence else {}))
    return workers.RUNNING[task_id]["task"]


def test_no_stop_door_returns_retains_holds_or_waits_for_a_presence_turn(root):
    """Review R1: a Restart, an update, a quit or a crash during a Presence turn handed the correspondent's turn back
    as the owner's saved work (its loop writes working checkpoints like any loop): returned (the owner's notice
    counted it), retained for boot, recovered as a dispatchable or held successor whose re-entry only refuses, a
    cancel intent minted for it ("Stopping active task.") and the agent-restart drain waiting for it. Nothing saves a
    Presence turn's stack (#2566), and base's queue never held one: no door takes it. An owner turn with the same
    saved work beside it is the control."""
    import copy
    from types import SimpleNamespace

    from ouroboros import server_restart
    from ouroboros.queue_observation import load_queue_document
    from supervisor import workers
    from supervisor.restart_retention import prepare_restart_returns, recover_saved_running, retain_saved_running

    seen = {}

    def scenario(author, task):
        tid = seen["tid"] = task["id"]
        _admit_lane_row(task)
        owner = _lane_turn(root, "direct", "owner-turn", 96)
        for task_id in (tid, "owner-turn"):
            _checkpoint(root, task_id)
        ctx = SimpleNamespace(RUNNING=workers.RUNNING, PENDING=workers.PENDING)
        seen["returns"] = prepare_restart_returns(root, dict(workers.RUNNING), [], transaction_id="tx-r1")
        seen["retained"] = retain_saved_running(copy.deepcopy(dict(workers.RUNNING)), (), root)
        rows = [copy.deepcopy(next(row for row in load_queue_document(root)["running"] if row["id"] == tid)),
                {"id": "owner-turn", "task": copy.deepcopy(owner), "attempt": 1}]
        returned = {"return_ids": [tid, "owner-turn"], "task_ids": [tid, "owner-turn"]}
        seen["auto"] = [t["id"] for t in recover_saved_running(copy.deepcopy(rows), root, transaction=returned,
                                                               cause="owner_restart")]
        seen["held"] = [t["id"] for t in recover_saved_running(copy.deepcopy(rows), root, transaction={},
                                                               cause="app_stop")]
        seen["owned"] = server_restart._owned_live_task_ids(ctx)
        seen["drain"] = server_restart._live_running_task_ids(ctx)

    _live_turn(root, scenario)
    tid = seen["tid"]
    assert seen["returns"] == {"owner-turn"} and seen["retained"] == ["owner-turn"]
    assert seen["auto"] == ["owner-turn"] and seen["held"] == ["owner-turn"]
    assert tid not in seen["owned"] and "owner-turn" in seen["owned"]
    assert tid not in seen["drain"] and "owner-turn" in seen["drain"]


@pytest.mark.parametrize("kind, fenced", [("presence", False), ("direct", True)])
def test_boot_restore_fences_an_unrecovered_owner_row_never_a_presence_turn(root, kind, fenced):
    """Review R1 at boot: a RUNNING row the stop left unrecovered is fenced with a cancel intent and counted among the
    boot's "Settling N tasks"; a correspondent's Presence turn is its place's (the Presence sweep settles a lost
    attempt), never fenced or counted, as base's queue never held one."""
    from ouroboros.cancel_intents import active_intents
    from supervisor.queue_snapshot import _fence_snapshot_running_rows

    task = _lane_turn(root, kind, f"{kind}-turn", 97)
    assert (_fence_snapshot_running_rows([{"id": task["id"], "task": task}], restored_ids=set())
            == ([task["id"]] if fenced else []))
    assert (task["id"] in active_intents(root, strict=True)) is fenced


def _hurry_door(root, task_id):
    """The owner's HTTP Hurry door for one id: ``(status, reason_code)``."""
    import asyncio
    import json
    from types import SimpleNamespace

    from starlette.requests import Request

    from ouroboros.gateway.task_hurry import api_task_hurry

    body = json.dumps({"request_id": "hurry-r3"}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": f"/api/tasks/{task_id}/hurry", "headers": [],
                       "path_params": {"task_id": task_id},
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=root))}, receive)
    response = asyncio.run(api_task_hurry(request))
    return response.status_code, json.loads(response.body).get("reason_code")


def test_hurry_of_a_presence_turns_lane_row_refuses_typed_and_seeds_nothing(root, monkeypatch):
    """Review R3: the lane row made the Hurry door treat a correspondent's turn as a pooled RUNNING root: it seeded
    the control task result and the hurry mail reached that turn. A live Presence turn takes no Hurry (as Pause):
    typed 409, nothing seeded. Without the row (the in-server author alone) the door kept the target's answer."""
    from ouroboros.gateway.task_hurry import _admit_hurry_locked
    from supervisor import queue as q

    seeded, seen = [], {}
    monkeypatch.setattr(q, "ensure_control_task_result", lambda task_id: seeded.append(task_id))

    def scenario(author, task):
        tid = task["id"]
        seen["without_row"] = _admit_hurry_locked(tid)[1]
        _admit_lane_row(task)
        seen["with_row"] = _admit_hurry_locked(tid)[1]
        seen["http"] = _hurry_door(root, tid)

    _live_turn(root, scenario)
    assert seen["without_row"] == "task_not_live"
    assert seen["with_row"] == "presence_hurry_unsupported" and seen["http"] == (409, "presence_hurry_unsupported")
    assert seeded == []


def test_the_owners_running_list_and_count_leave_a_presence_turn_to_its_place(root):
    """Review R7: the mind's recent running tasks and its queue fact's ``running_count`` listed and counted a
    correspondent's Presence turn as an unlabelled running task. As the roster (N2) and the owner-turn count (N8)
    read it, it is its place's (live works names it ``inline_presence``); an owner turn beside it stays listed."""
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.tools.recent_tasks import _running_tasks
    from supervisor import queue

    seen = {}

    def scenario(author, task):
        _admit_lane_row(task)
        _lane_turn(root, "direct", "owner-turn", 96)
        queue.publish_queue_snapshot(reason="r7_probe")
        seen["tid"] = task["id"]
        seen["listed"] = [row["task_id"] for row in _running_tasks(root)]
        seen["count"] = _queue_context_fact({"budget_drive_root": str(root)})["running_count"]

    _live_turn(root, scenario)
    assert seen["listed"] == ["owner-turn"] and seen["count"] == 1
