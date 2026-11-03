"""A conversation-lane turn saved by an application stop comes back once, on the folder it was admitted with.

Since #2654 an owner turn owns a RUNNING row in the conversation lane, so the stop keeps it there with
its saved work (``workers.kill_workers`` -> ``restart_retention.retain_saved_running``). Two regressions
of that move, both seen on the real keyless server (system_e2e S42):

1. Duplicate restore. The next boot's queue init also adopts the same id as an orphaned direct root
   (``direct_roots.adopt_orphaned_direct_results``) whenever the dead server's snapshot is older than
   the ownership observation window — any owner who quits and reopens later. Restore filtered those
   handed-over ids by the snapshot's PENDING rows only, so ``recover_saved_running`` built a successor
   from the RUNNING row AND one from the handover: two queue owners of one id, which the Project hold
   then refuses for both ("Another queue owner holds this task id"), so Resume answered ok over a turn
   that never ran.
2. Recaptured basis. The saved row keeps the carrier's first-step obligation (``_conversation_prepare``),
   so the resumed attempt re-ran ``prepare_conversation_task``, which captured a fresh room basis: a
   folderless turn moved into the folder a sibling gave the room meanwhile (#2547's guard regressed).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.utils import utc_now_iso
from supervisor import worker_chat_lane, workers
from tests._shared import write_queue_document

pytestmark = pytest.mark.serial

TID = "lane-saved-turn"


def _next_generation(monkeypatch, drive):
    """The process booting after the stop: a new custody generation holding the root's admission."""
    from ouroboros import process_custody
    from ouroboros.server_entrypoint import admit_data_root

    monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
    admit_data_root(drive, wait_sec=0)  # released by conftest's autouse admission fixture


def _stopped_lane_turn(tmp_path, monkeypatch, *, snapshot_age_sec: float):
    """A lane turn whose attempt saved work, caught by an application stop ``snapshot_age_sec`` ago."""
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_status import execution_owner_record
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
    from tests._cancel_intents_shared import _CaptureQueue

    queue, state, pool = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(pool, "get_event_q", lambda create=True: _CaptureQueue(), raising=False)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    owner = {**execution_owner_record(tmp_path, {"id": TID, "_attempt": 1}, "direct"),
             "lane": "conversation", "executor": "worker"}
    write_task_result(tmp_path, TID, STATUS_RUNNING, chat_id=1, task_attempt=1, _is_direct_chat=True,
                      execution_owner=owner)
    _ctx, limit = _loop_ctx(tmp_path, TID, direct=True, attempt=1)
    wc.save_round(limit, "pre_effect")
    task = {"id": TID, "type": "task", "chat_id": 1, "text": "write the note", "_is_direct_chat": True,
            "_attempt": 1, "root_task_id": TID, "depth": 0, "_execution_owner": owner,
            "_conversation_prepare": {"quiet": False}}
    stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - snapshot_age_sec))
    path = tmp_path / "state" / "queue_snapshot.json"
    write_queue_document(path, {"ts": stopped_at, "pending": [], "acceptance_fences": [], "budget_root_fences": [],
                                "running": [{"id": TID, "attempt": 1, "worker_id": 0, "task": task}]})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", path, raising=False)
    (tmp_path / "state" / "direct_roots.json").unlink(missing_ok=True)
    _next_generation(monkeypatch, tmp_path)
    queue.PENDING.clear()
    queue.init(tmp_path)
    fenced: list = []
    queue.restore_pending_from_snapshot(terminalized=fenced)
    return queue, pool, fenced


@pytest.mark.parametrize("age_sec", [60, 86_400])  # past the 10 s ownership window: a later reopen, a next day
def test_a_saved_lane_turn_restores_under_exactly_one_queue_row(tmp_path, monkeypatch, age_sec):
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    queue, pool, fenced = _stopped_lane_turn(tmp_path, monkeypatch, snapshot_age_sec=age_sec)

    # The premise: queue init's orphan sweep named the same turn as the snapshot's RUNNING row.
    assert TID in (queue.PRIOR_DIRECT_ROOTS.get("task_ids") or [])
    rows = [row for row in pool.PENDING if row.get("id") == TID]
    assert len(rows) == 1, [{k: row.get(k) for k in ("_attempt", "_working_recovery")} for row in rows]
    task = rows[0]
    assert task["_working_recovery"]["from_attempt"] == 1 and task["_attempt"] == 2
    hold = budget_hold_fact(task)
    assert hold and hold["reason"] == HOLD_SAVED_WORK and hold["door"] == "app_stop"
    assert fenced == [] and not ci.has_active_intent(tmp_path, TID, strict=True)
    # One owner of the id, so the owner's Resume releases it for dispatch.
    assert queue.resume_budget_paused_task(TID)["ok"]
    assert budget_hold_fact(task) is None and task["_working_recovery"]["source_ref"]
    assert [row.get("id") for row in pool.PENDING].count(TID) == 1


def _room(tmp_path, folder: str = ""):
    from ouroboros.project_admission import direct_turn_basis
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "s42-room", working_dir=folder)
    return direct_turn_basis(tmp_path, "s42-room")


def _resumed_seed(tmp_path, basis):
    task = worker_chat_lane._seed_task(1, "write the proof note", None, None, {"client_message_id": "cm-s42"})
    task.update(project_id="s42-room", _attempt=2, _conversation_prepare={"quiet": False},
                **({"_project_admission": basis} if basis is not None else {}))
    return task


def test_a_resumed_first_step_keeps_the_basis_the_turn_was_admitted_with(tmp_path, monkeypatch):
    from ouroboros.project_admission import admitted_folder
    from ouroboros.projects_registry import update_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    admitted = _room(tmp_path)  # folderless when the turn was admitted
    assert admitted_folder(admitted, "s42-room") == ""
    sibling = tmp_path / "sibling-folder"
    sibling.mkdir()
    update_project(tmp_path, "s42-room", working_dir=str(sibling))  # a sibling's start gave the room a folder

    task = _resumed_seed(tmp_path, admitted)
    frames: list = []
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path, broadcast=frames.append)

    assert admitted_folder(task["_project_admission"], "s42-room") == "", task["_project_admission"]
    assert task["_project_admission"] == admitted
    assert not [frame for frame in frames if frame.get("type") == "direct_turn_bound"], frames
    assert "_conversation_prepare" not in task


def test_a_first_step_without_a_carried_basis_still_binds_the_rooms_current_folder(tmp_path, monkeypatch):
    """Positive control: an admission (no basis yet) captures and broadcasts the room as it is now."""
    from ouroboros.project_admission import admitted_folder

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    _room(tmp_path, str(folder))

    task = _resumed_seed(tmp_path, None)
    frames: list = []
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path, broadcast=frames.append)

    assert admitted_folder(task["_project_admission"], "s42-room") == str(folder)
    bound = [frame for frame in frames if frame.get("type") == "direct_turn_bound"]
    assert bound and bound[0]["facts"]["_project_admission"] == task["_project_admission"], frames


def test_a_basis_for_another_room_is_replaced_by_the_rooms_own(tmp_path, monkeypatch):
    """Placement moves a turn by replacing its room: a carried basis of another room never binds it."""
    from ouroboros.project_admission import admitted_folder

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    _room(tmp_path, str(folder))
    other = {"project_id": "another-room", "project": {"id": "another-room", "working_dir": ""}}

    task = _resumed_seed(tmp_path, other)
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path)

    assert task["_project_admission"]["project_id"] == "s42-room"
    assert admitted_folder(task["_project_admission"], "s42-room") == str(folder)


def test_restored_saved_turn_rows_are_dated_by_the_stop(tmp_path):
    """Fixture honesty: the snapshot the restore test writes is older than the ownership window."""
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_status import queue_snapshot_observation

    path = tmp_path / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True)
    stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 60))
    write_queue_document(path, {"ts": stopped_at, "pending": [], "running": []})
    assert queue_snapshot_observation(load_queue_document(tmp_path, fresh=True))["fresh"] is False
    write_queue_document(path, {"ts": utc_now_iso(), "pending": [], "running": []})
    assert queue_snapshot_observation(load_queue_document(tmp_path, fresh=True))["fresh"] is True
