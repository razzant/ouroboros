"""Queued roots are explicit mail targets, never implicit room readers."""
from types import SimpleNamespace

import pytest

from ouroboros.room_reader import is_reading
from ouroboros.server_routing_context import _addressable_root_tasks


@pytest.mark.parametrize("fields,running,expected", [
    ({}, True, True), ({}, False, False), ({"_owner_hold": True}, True, False),
    ({"_project_admission_restore_hold": {"reason": "verify"}}, True, False),
    ({"_cancel_intent_authority_hold": {"reason": "unreadable"}}, True, False),
    ({"_budget_pause": {"status": "paused"}}, True, False),
    ({"owner_wait": {"state": "waiting", "kind": "quiz"}}, True, True),
])
def test_structural_reader_state(tmp_path, fields, running, expected):
    assert is_reading(tmp_path, "work", {"id": "work", **fields}, running=running, fences={}) is expected


BINDING = "binding-1"
RELATION = {"presence_binding_relation": {"binding_id": BINDING}}  # the owner's work related to a place
AUTHORITY = {"presence_binding_authority": {"binding_id": BINDING}}  # work acting for a place, or an old owner root
DOOR = {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-late-answer", "ts": "2026-10-29T13:00:00Z"}}


@pytest.mark.parametrize("record,visible", [
    ({}, True),
    ({"metadata": RELATION}, True),  # a late answer's root, as admission writes it now
    ({"metadata": {"work_kind": "owner", **AUTHORITY}, **DOOR}, True),  # an old one: the owner door's stamp at top level
    ({"metadata": {**AUTHORITY, **DOOR}}, True), ({"metadata": {**AUTHORITY, "origin_suppressed": True}}, True),
    ({"metadata": AUTHORITY}, False),  # a place's delegated work: authority alone
    ({"metadata": AUTHORITY, "parent_task_id": "parent", **DOOR}, False),
    ({"metadata": AUTHORITY, "delegation_role": "subagent", **DOOR}, False),
    ({"metadata": {**AUTHORITY, **RELATION}}, False),  # several carriers never widen
    ({"metadata": {"presence_binding_relation": "malformed"}}, False),  # it narrows to nothing, never widens
    ({"metadata": {"presence": {"binding_id": BINDING}}, **DOOR}, False),  # a speaker stays a speaker
    ({"_presence_turn": True}, False),
    ({"task_contract": {"capability_ceiling": {}}}, False),  # a lost carrier under a Presence ceiling
    ({"_execution_owner": {"kind": "direct", "executor": "inline_call"}}, False),  # a live call keeps its own intake
])
def test_an_owner_visible_reader_is_decided_by_the_whole_record(record, visible):
    from ouroboros.room_reader import owner_visible_reader

    assert owner_visible_reader({"id": "work", "chat_id": 1, "delegation_role": "root", **record}) is visible


def test_only_a_work_with_a_tool_loop_drains_mail():
    from ouroboros.room_reader import drains_mail

    assert drains_mail({"type": "task"}) and drains_mail({})
    assert not drains_mail({"type": "deep_self_review"}) and not drains_mail({"type": "Deep_Self_Review"})


def test_the_hold_keys_readers_judge_are_the_ones_the_dispatcher_writes_and_refuses():
    """One predicate, backed by its producers: the real writer's key, the dispatcher's own refusal and the restore's
    cancel-authority field. A key no producer writes is no hold."""
    from ouroboros.project_admission import hold_unreadable_result
    from ouroboros.room_reader import DISPATCH_HOLDS, dispatch_held
    from supervisor import queue, worker_assignment, workers

    written = {"id": "project-work", "project_id": "p"}
    assert hold_unreadable_result(written) is True  # the Project producer writes its own key
    assert [key for key in written if key.endswith("_hold")] == ["_project_admission_restore_hold"]
    assert dispatch_held(written) and "_project_admission_restore_hold" in DISPATCH_HOLDS
    assert workers._CANCEL_INTENT_AUTHORITY_HOLD_FIELD in DISPATCH_HOLDS
    for key in ("_owner_hold", "_project_admission_restore_hold"):  # the dispatcher refuses exactly these rows
        with queue._queue_lock:
            assert worker_assignment._candidate_ineligible(queue, {"id": "queued", key: {"reason": "x"}}, 5.0, {}) is True
        assert key in DISPATCH_HOLDS
    assert not dispatch_held({"_project_admission_hold": {"reason": "no producer writes this key"}})
    assert not dispatch_held({"_owner_hold": None, "_project_admission_restore_hold": {}})


def test_restored_direct_root_stays_addressable_but_queued_root_does_not_read(tmp_path):
    task = {"id": "restored", "chat_id": 1, "drive_root": str(tmp_path), "_is_direct_chat": True}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[task], BUDGET_ROOT_FENCES={})
    rows = _addressable_root_tasks(ctx, 1)
    assert rows[0]["task_id"] == "restored" and rows[0]["room_reader"] is False
    ctx.RUNNING, ctx.PENDING = {"restored": {"task": task}}, []
    assert _addressable_root_tasks(ctx, 1)[0]["room_reader"] is True


@pytest.mark.parametrize("hold", ["_owner_hold", "_project_admission_restore_hold", "_cancel_intent_authority_hold"])
def test_the_routing_roster_names_a_held_root_no_room_reader(tmp_path, hold):
    task = {"id": "held", "chat_id": 1, "drive_root": str(tmp_path), hold: {"reason": "x"}}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={"held": {"task": task}}, PENDING=[], BUDGET_ROOT_FENCES={})
    (row,) = _addressable_root_tasks(ctx, 1)
    assert row["task_id"] == "held" and row["room_reader"] is False, "listed and addressable, but it reads nothing now"


@pytest.mark.parametrize("hold_at_admission", [False, True])
def test_room_intake_rechecks_the_current_task_under_the_queue_lock(tmp_path, monkeypatch, hold_at_admission):
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor import queue as q
    from tests.test_project_routing_v664 import _ctx
    room = create_project(tmp_path, "reader-race")
    task = {"id": "reader", "chat_id": room["chat_id"], "drive_root": str(tmp_path)}
    ctx = _ctx(tmp_path, running={"reader": {"task": task}})
    class AdmissionLock:
        def acquire(self):
            ctx.RUNNING["reader"] = {"task": {**task, "_owner_hold": hold_at_admission}}
        def release(self):
            pass
    monkeypatch.setattr(q, "_queue_lock", AdmissionLock())
    result = _route_project_chat_to_running_task(ctx, room["chat_id"], "followup", "reader-race")
    assert result == ("" if hold_at_admission else "reader")
    assert [e["text"] for e in drain_owner_entries(tmp_path, "reader")] == ([] if hold_at_admission else ["followup"])


def test_an_unheld_cold_sleep_reads_question_mail_but_never_room_intake(tmp_path):
    from ouroboros.room_reader import unheld_cold_sleep

    sleep = {"reason": "sleep", "sleep": {"sleep_id": "s"}}
    task = {"id": "work", "_budget_pause": sleep}
    assert is_reading(tmp_path, "work", task, running=False, fences={}) is False  # room intake: a new head
    assert unheld_cold_sleep(tmp_path, "work", task, fences={}) is True
    granted = {"id": "work", "_budget_pause_resume": {"selected_by": "sleep_wake", "pause": sleep}}
    assert unheld_cold_sleep(tmp_path, "work", granted, fences={}) is True  # its capacity-delayed start
    for blocked in ({"_owner_hold": True}, {"_project_admission_restore_hold": {"reason": "verify"}},
                    {"_cancel_intent_authority_hold": {"reason": "unreadable"}},
                    {"_budget_pause_hold": {"reason": "owner_restart_hold"}}):
        assert unheld_cold_sleep(tmp_path, "work", {**task, **blocked}, fences={}) is False
    assert unheld_cold_sleep(tmp_path, "work", task, fences={"work": {"status": "active"}}) is False  # money
    for paused in ({"reason": "budget"}, {"reason": "sleep"}):  # money; an interrupted pause awaiting Resume
        assert unheld_cold_sleep(tmp_path, "work", {"id": "work", "_budget_pause": paused}, fences={}) is False
    assert unheld_cold_sleep(tmp_path, "work", {"id": "work"}, fences={}) is False  # plain queued work
