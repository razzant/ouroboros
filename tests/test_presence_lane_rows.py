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
    install_lane_spare(workers, 91)  # Presence leaves one ready carrier for the owner.
    seed = {"id": task["id"], "type": "presence", "source": "presence", "chat_id": task["chat_id"],
            "text": str(task.get("text") or ""), "_is_direct_chat": True, "_presence_turn": True,
            "metadata": dict(task.get("metadata") or {})}
    with queue._queue_lock, workers._repo_writer_gate_lock:
        meta = conversation_lane.admit_turn(seed, chat_id=int(seed["chat_id"]), client_message_id="presence-event",
                                            quiet=True, reply_expected=False)
    conversation_lane.dispatch_admitted_turn(meta)  # publish outside the queue/gate locks
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


def test_a_moved_turns_saved_work_takes_every_stop_door_as_its_projects_work(root, monkeypatch):
    """#2566 S2 beside the two tests above (their negative controls): once the owner moved the turn to a Project as
    text, its saved stack is that Project's work, so the same doors return, retain, hold, address and wait for it
    as they do the owner's turn, and the boot fence settles it when nothing returned it. The predicate is read at
    each door, never at the turn's start: the move happens while it runs."""
    import copy
    from types import SimpleNamespace

    from ouroboros import server_restart
    from ouroboros.cancel_intents import active_intents
    from ouroboros.queue_observation import load_queue_document
    from supervisor import workers
    from supervisor.queue_snapshot import _fence_snapshot_running_rows
    from supervisor.restart_retention import prepare_restart_returns, recover_saved_running, retain_saved_running
    from tests.test_presence_project_activity import _move

    monkeypatch.setattr(server_restart, "DATA_DIR", root)
    seen = {}

    def scenario(author, task):
        tid = seen["tid"] = task["id"]
        _admit_lane_row(task)
        ctx = SimpleNamespace(RUNNING=workers.RUNNING, PENDING=workers.PENDING)
        _checkpoint(root, tid)
        seen["before"] = (retain_saved_running(copy.deepcopy(dict(workers.RUNNING)), (), root),
                          server_restart._owned_live_task_ids(ctx))
        _move(root, tid, PROJECT)
        seen["returns"] = prepare_restart_returns(root, dict(workers.RUNNING), [], transaction_id="tx-moved")
        seen["retained"] = retain_saved_running(copy.deepcopy(dict(workers.RUNNING)), (), root)
        rows = [copy.deepcopy(next(row for row in load_queue_document(root)["running"] if row["id"] == tid))]
        seen["auto"] = [t["id"] for t in recover_saved_running(copy.deepcopy(rows), root, cause="owner_restart",
                                                               transaction={"return_ids": [tid], "task_ids": [tid]})]
        seen["held"] = [t["id"] for t in recover_saved_running(copy.deepcopy(rows), root, transaction={},
                                                               cause="app_stop")]
        seen["owned"] = server_restart._owned_live_task_ids(ctx)
        seen["drain"] = server_restart._live_running_task_ids(ctx)
        seen["fenced"] = _fence_snapshot_running_rows(copy.deepcopy(rows), restored_ids=set())

    _live_turn(root, scenario)
    tid = seen["tid"]
    assert seen["before"] == ([], [])  # the same turn, unmoved: its place's, at the same doors
    assert seen["returns"] == {tid} and seen["retained"] == [tid]
    assert seen["auto"] == [tid] and seen["held"] == [tid]
    assert tid in seen["owned"] and tid in seen["drain"]
    assert seen["fenced"] == [tid] and tid in active_intents(root, strict=True)


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


def test_a_warm_paused_moved_turn_keeps_its_lifecycle_so_activity_offers_resume_and_never_pause(root):
    """The moved-text lifecycle is the work's, not its working phase's: the warm park's own durable shape leaves the row
    paused, and each action is gated on its own (Pause only while working, Resume while paused, Stop as before)."""
    from ouroboros.owner_pause import install_fence, set_fence_state
    from ouroboros.task_results import write_task_result
    from tests.test_presence_project_activity import _actions, _row, _state, _work

    found = {}

    def scenario(_author, task):
        tid = task["id"]
        _admit_lane_row(task)
        _move(root, tid, PROJECT)
        census, works, _complete = _state(root)
        found["working"] = (_row(census, tid), _work(works, tid))
        fence, _created = install_fence(root, tid, request_id="warm-ui")
        set_fence_state(root, tid, fence_id=fence["fence_id"], state="paused")
        # The durable shape ``checkpoint_owner_wait`` writes for the owner Pause's warm park.
        write_task_result(root, tid, "running", owner_wait={
            "state": "waiting", "reason": "owner_pause", "task_attempt": 1,
            "owner_pause": {"fence_id": fence["fence_id"], "root_task_id": tid}})
        census, works, complete = _state(root)
        found["paused"] = (_row(census, tid), _work(works, tid), complete)

    _live_turn(root, scenario)
    row, work = found["working"]
    assert row["presence_pause"] is True and _actions(work) == ["pause", "stop"]
    row, work, complete = found["paused"]
    assert complete is True and row["phase"] == "budget_paused" and row["presence_pause"] is True
    assert work["activity"] == "paused" and _actions(work) == ["resume", "stop"]
    assert all(control["target"] == row["activity_id"] for control in work["controls"])


def test_the_moved_lifecycle_flag_needs_a_moved_started_text_turn_whatever_its_phase():
    from ouroboros.gateway.state import _mark_carried_presence
    from ouroboros.task_carrier import EXECUTOR_INLINE_CALL

    moved, origin_bound = {"t": {"project_id": "room"}}, {"t": {"project_id": "room", "origin_bound": True}}

    def flagged(phase, bindings=moved, **row):
        activity = {"activity_id": "t", "kind": "presence", "phase": phase, **row}
        _mark_carried_presence(None, [activity], bindings, None, True)
        return activity.get("presence_pause") is True

    assert flagged("working") and flagged("budget_paused")
    assert not flagged("preparing")  # nothing of it is saved yet: no Pause door and no return through a Restart
    assert not flagged("budget_paused", origin_bound) and not flagged("working", {})  # never moved
    assert not flagged("budget_paused", executor=EXECUTOR_INLINE_CALL)  # a live call never takes this lifecycle


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


def test_a_witnessed_carried_authors_stop_is_its_runners_never_a_carrier_kill(root, monkeypatch):
    """#2012 on the lane (#2654): the Stop of a Presence turn whose author the census witnesses is its runner's,
    wherever it runs. Custody takes no carrier it holds (``runner_takes_stop``, read before Q), and the author reads
    that Stop itself though it holds a worker slot (``owner_stop.in_process_stop_now``), so its next stop check ends
    it; an author the census cannot witness keeps the capture and kill. The custody path end to end (the Stop pending
    while the held round runs, then the runner's cancelled terminal) is the browser walk's
    (``test_presence_project_activity_browser``: the unmoved turn and the moved one's step 5)."""
    from types import SimpleNamespace

    from ouroboros import presence_runner
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import cancel_publication, queue as q
    from supervisor.owner_stop import in_process_stop_now

    seen = {}

    def scenario(author, task):
        tid = task["id"]
        _admit_lane_row(task)
        request_cancel(root, tid, source="http_cascade", requested_stop_policy="immediate")
        seen["runner_owned"] = cancel_publication.runner_takes_stop(q, tid, load_task_result(root, tid))
        held = SimpleNamespace(worker_slot_held=True, task_id=tid, canonical_root=root)
        seen["reads"] = (in_process_stop_now(SimpleNamespace(model_wait_context=SimpleNamespace(
                             **vars(held), task={"_presence_turn": True}))),
                         in_process_stop_now(SimpleNamespace(model_wait_context=SimpleNamespace(**vars(held), task={}))))
        monkeypatch.setattr(presence_runner, "presence_execution_witness", lambda *_a, **_k: None)
        seen["unwitnessed"] = cancel_publication.runner_takes_stop(q, tid, load_task_result(root, tid))

    _live_turn(root, scenario)
    assert seen["runner_owned"] is True and seen["unwitnessed"] is False
    assert seen["reads"] == (True, False)  # a carried author reads its Stop; a pooled worker's Stop stays the kill


def test_a_moved_turns_pooled_author_the_census_witnesses_takes_its_stop_on_its_runners_rail(root, monkeypatch):
    """Review R2 F3 (Grok): a Resume or a restart's return runs a moved text turn's saved work as its author in a
    pool worker, its row stamped ``pooled``. Once the census witnesses that author, the owner's Stop is its runner's
    as on the lane (#2012): custody captures and kills no worker and writes nothing, the Stop stays pending
    (``finish_in_process_turn``) while the author reads it itself, and custody settles it from the author's own
    terminal. An author not yet witnessed (its carrier has not taken the conversation) and ordinary pooled work keep
    the capture and kill; a parked turn's saved work keeps the queue's cancel (``test_a_stop_reaches_saved_work_...``
    in ``test_presence_moved_lifecycle``)."""
    import os
    from types import SimpleNamespace

    from ouroboros import presence_runner
    from ouroboros.cancel_intents import active_intent, request_cancel
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import cancel_publication, queue as q, task_lifecycle, workers
    from supervisor.owner_stop import in_process_stop_now, owner_stop_origin
    from supervisor.worker_assignment import _mirror_assigned_running_status
    from tests.test_presence_moved_lifecycle import _boot_window_row, _seed_and_built, _slot
    from tests.test_presence_runner import _event

    captured = []
    monkeypatch.setattr(task_lifecycle, "_finish_captured_running",
                        lambda task_id, worker, *_a, **_k: captured.append(task_id) or "cancelled")
    _seed, built = _seed_and_built(root)

    def running(task_id, wid, task):
        _mirror_assigned_running_status(task)
        workers.RUNNING[task_id] = {"task": task, "worker_id": wid, "attempt": 2}
        request_cancel(root, task_id, source="http_cascade", requested_stop_policy="immediate")
        return _slot(workers, wid, os.getpid(), task_id, lane="project")

    def continuation(task_id, wid):
        _boot_window_row(root, task_id, {**built, "id": task_id})
        return running(task_id, wid, {**built, "id": task_id, "drive_root": str(root), "_attempt": 2,
                                      "_working_recovery": {"cause": "owner_restart"},
                                      "metadata": {**built["metadata"], "presence_request_waits": False}})

    # Not witnessed yet: the ordinary capture and kill of its worker.
    early = built["id"] + "-early"
    continuation(early, 92)
    assert cancel_publication.runner_takes_stop(q, early, load_task_result(root, early)) is False
    task_lifecycle.cancel_task_custody(early)
    assert captured == [early]
    # Ordinary pooled work keeps the capture and kill whatever the census says.
    write_task_result(root, "ordinary-work", "scheduled", result="Task is queued.")  # assignment stamps it pooled
    running("ordinary-work", 93, {"id": "ordinary-work", "type": "task", "chat_id": 0, "drive_root": str(root),
                                  "_attempt": 2})
    with monkeypatch.context() as census:
        census.setattr(presence_runner, "presence_execution_witness", lambda *_a, **_k: True)
        ordinary = load_task_result(root, "ordinary-work")
        assert ordinary["execution_owner"]["kind"] == "pooled"
        assert cancel_publication.runner_takes_stop(q, "ordinary-work", ordinary) is False
        task_lifecycle.cancel_task_custody("ordinary-work")
    assert captured == [early, "ordinary-work"]
    # The author its carrier bound (``presence_carrier._SavedCarrier.bind``): the census witnesses it.
    tid = built["id"]
    slot = continuation(tid, 91)
    lease, _busy = presence_runner._configured_gate(root).try_admit(_event().conversation_key)
    lease.bind_author(tid)
    try:
        assert cancel_publication.runner_takes_stop(q, tid, load_task_result(root, tid)) is True
        assert task_lifecycle.cancel_task_custody(tid) == "failed"  # pending, for its runner's next stop check
        assert captured == [early, "ordinary-work"] and slot.busy_task_id == tid and not slot.reaping
        assert workers.RUNNING[tid]["worker_id"] == 91 and load_task_result(root, tid)["status"] == "running"
        intent = active_intent(root, tid)
        assert intent["state"] == "requested"
        assert intent["last_error"] == "Presence turn has not reached its next stop check"
        waiter = SimpleNamespace(worker_slot_held=True, task_id=tid, canonical_root=root, task={"_presence_turn": True})
        assert in_process_stop_now(SimpleNamespace(model_wait_context=waiter)) is True
        # Its stop check lands its own cancelled terminal while custody's next pass is under way: that pass settles the
        # Stop from it (``cancelled`` for this very request), and nothing of custody's is written over it.
        origin = owner_stop_origin(root, tid)
        real_read = cancel_publication._load_result_row

        def read_after_terminal(queue_module, task_id):
            write_task_result(root, tid, "cancelled", result="", cancel_origin=origin)
            return real_read(queue_module, task_id)

        monkeypatch.setattr(cancel_publication, "_load_result_row", read_after_terminal)
        assert task_lifecycle.cancel_task_custody(tid) == "cancelled"
    finally:
        lease.release()
    assert captured == [early, "ordinary-work"] and active_intent(root, tid) is None
    assert load_task_result(root, tid)["cancel_origin"]["request_id"] == origin["request_id"] == intent["request_id"]


def test_a_pooled_author_that_leaves_its_census_before_its_terminal_keeps_the_stop_for_the_next_pass(root, monkeypatch):
    """The race beside the test above: custody read the pooled moved author as witnessed (``runner_takes_stop``), so it
    captured no worker, and the author left its census before ``finish_in_process_turn`` read it, its row still
    unsettled (its carrier ended without a terminal). That pass writes nothing and keeps the Stop pending, recorded as
    what it is, with its claim released; the next pass, the author still absent, takes the ordinary capture and kill."""
    import os

    from ouroboros import presence_runner
    from ouroboros.cancel_intents import active_intent, request_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import cancel_publication, queue as q, task_lifecycle, workers
    from supervisor.worker_assignment import _mirror_assigned_running_status
    from tests.test_presence_moved_lifecycle import _boot_window_row, _seed_and_built, _slot
    from tests.test_presence_runner import _event

    captured = []
    monkeypatch.setattr(task_lifecycle, "_finish_captured_running",
                        lambda task_id, worker, *_a, **_k: captured.append(task_id) or "cancelled")
    _seed, built = _seed_and_built(root)
    tid = built["id"]
    _boot_window_row(root, tid, built)
    task = {**built, "drive_root": str(root), "_attempt": 2, "_working_recovery": {"cause": "owner_restart"},
            "metadata": {**built["metadata"], "presence_request_waits": False}}
    _mirror_assigned_running_status(task)
    workers.RUNNING[tid] = {"task": task, "worker_id": 91, "attempt": 2}
    slot = _slot(workers, 91, os.getpid(), tid, lane="project")
    request_cancel(root, tid, source="http_cascade", requested_stop_policy="immediate")
    lease, _busy = presence_runner._configured_gate(root).try_admit(_event().conversation_key)
    lease.bind_author(tid)  # what ``presence_carrier._SavedCarrier.bind`` does once its carrier holds the conversation
    real_takes_stop = cancel_publication.runner_takes_stop
    read_before_q = []

    def author_leaves_after_the_read(queue_module, task_id, row):
        read_before_q.append(real_takes_stop(queue_module, task_id, row))
        lease.release()  # its carrier ends, no terminal written, before custody's settlement read
        return read_before_q[-1]

    try:
        with monkeypatch.context() as race:
            race.setattr(cancel_publication, "runner_takes_stop", author_leaves_after_the_read)
            assert task_lifecycle.cancel_task_custody(tid) == "failed"
    finally:
        lease.release()
    assert read_before_q == [True] and captured == [] and not slot.reaping and slot.busy_task_id == tid
    stored = load_task_result(root, tid)
    assert stored["status"] == "running" and "cancel_origin" not in stored
    intent = active_intent(root, tid)
    assert intent["state"] == "requested" and not intent.get("claim_owner")
    assert intent["last_error"] == "Presence author no longer witnessed before its terminal"
    # The next pass, the author still absent: its worker takes the ordinary capture and kill, no Stop stays stuck.
    assert presence_runner.presence_execution_witness(tid, drive_root=root) is False
    assert cancel_publication.runner_takes_stop(q, tid, load_task_result(root, tid)) is False
    task_lifecycle.cancel_task_custody(tid)
    assert captured == [tid] and slot.reaping
