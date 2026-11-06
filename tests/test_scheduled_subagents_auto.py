"""Work the clock started runs on Auto (#2399; owner 03.11): never the room's Swarm, never past its Off.

Driven through the real producers: the schedule table and occurrence admission, the queue's stamp, the actor a
worker binds, the round's fact, ``schedule_subagent`` and its supervisor admission, Continue, a restored queue row,
a promotion and the public API. The provenance is the scheduler's own stamp, which no template or caller supplies.
It holds whoever wrote the schedule, including a wake or a Presence conversation; ordinary wakes, Presence and Evolve
without clock provenance keep their own controls (negative controls below).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import config
from ouroboros.composer_cognition import update_project_cognition
from ouroboros.gateway.tasks import _create_task_from_body
from ouroboros.loop_messages import subagents_fact_line
from ouroboros.settings_integrity import (
    actor_settings_snapshot, effective_subagents_mode, live_subagents_mode, task_settings_scope, task_settings_snapshot,
)
from ouroboros.subagent_runtime import SubagentSelectionError, _require_room_subagents
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.control_delegation import check_delegation_admission
from supervisor.events_schedule_task import _subagent_room_mode
from tests.test_schedule_occurrence import _project, _row, q as queue_fixture
from tests.test_subagents_mode_messages import mode_settings
from tests.test_swarm_host_admission import host as host_fixture

q = queue_fixture
host = host_fixture
FACT_AUTO = "Subagents: Auto — scheduled work"
FACT_OFF = "Subagents: Off — set by my human for this chat"
FACT_SETTINGS_OFF = "Subagents: Off — off in Settings"


@pytest.fixture(autouse=True)
def _one_api_actor(monkeypatch):
    from ouroboros.tools import control
    from tests.test_available_subagents_runtime import _api_row, _settings as actor_settings

    monkeypatch.setattr(control, "load_settings", lambda: actor_settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")


def _settings(monkeypatch, mode, availability="available"):
    values = mode_settings(mode, availability)
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    return values


def _fire(q, *, project_id="", writer="owner"):
    _row(q, intent={"kind": "explicit_none", **({"project_id": project_id} if project_id else {})}
         if project_id else {"kind": "system_repo"}, project_id=project_id, metadata={"work_kind": writer})
    q.queue.check_scheduled_tasks()
    [task] = [row for row in q.pending if (row.get("metadata") or {}).get("schedule_occurrence")]
    return task


def _actor(task, root, settings=None):
    """The private view a worker binds for this task, entry settings included."""
    return actor_settings_snapshot(task_settings_snapshot(dict(settings or {}), dict(settings or {})), task, root)


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("mode,mode_seen,fact", [("swarm", "auto", FACT_AUTO), ("auto", "auto", FACT_AUTO), ("off", "off", FACT_OFF)])
def test_a_due_schedule_admits_on_auto_and_keeps_the_rooms_off(q, monkeypatch, room, mode, mode_seen, fact):
    _settings(monkeypatch, "auto" if room == "project" else mode)
    project_id = ""
    if room == "project":
        project_id = _project(q)["id"]
        update_project_cognition(q.root, project_id, subagents_mode=mode)
    task = _fire(q, project_id=project_id)
    assert task["metadata"]["schedule_writer_kind"] == "owner"
    assert task["metadata"]["subagents_mode"] == mode_seen
    assert load_task_result(q.root, task["id"])["metadata"]["subagents_mode"] == mode_seen
    with task_settings_scope(_actor(task, q.root)):
        assert live_subagents_mode() == effective_subagents_mode() == mode_seen
        assert subagents_fact_line() == fact
        if mode == "off":
            with pytest.raises(SubagentSelectionError, match="off in this chat"):
                _require_room_subagents()
        else:
            _require_room_subagents()


@pytest.mark.parametrize("availability", ["disabled", "unsaved"])
def test_global_off_holds_for_a_schedule_in_a_swarm_room(q, monkeypatch, availability):
    _settings(monkeypatch, "swarm", availability)
    task = _fire(q)
    assert task["metadata"]["subagents_mode"] == "off"
    with task_settings_scope(_actor(task, q.root)):
        assert effective_subagents_mode() == "off"
        assert subagents_fact_line() == FACT_SETTINGS_OFF
        with pytest.raises(SubagentSelectionError):
            _require_room_subagents()


def test_room_changes_stay_live_for_a_running_scheduled_actor(q, monkeypatch):
    values = _settings(monkeypatch, "swarm")
    task = _fire(q)
    with task_settings_scope(_actor(task, q.root)):
        assert subagents_fact_line() == FACT_AUTO
        values["OUROBOROS_SUBAGENTS_MODE"] = "off"
        assert live_subagents_mode() == "off" and subagents_fact_line() == FACT_OFF
        values["OUROBOROS_SUBAGENTS_MODE"] = "swarm"
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO


def _spawn(tmp_root, metadata, *, parent="parent", project_id=""):
    """One real ``schedule_subagent``: the tool's event, then the supervisor's payload the child's worker receives."""
    import queue as stdqueue
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from supervisor.task_dispatch import build_scheduled_task_payload

    ctx = ToolContext(repo_dir=tmp_root, drive_root=tmp_root, current_task_type="task")
    ctx.task_id, ctx.event_queue, ctx.project_id = parent, stdqueue.Queue(), project_id
    ctx.task_metadata = {"root_task_id": parent, **metadata}
    ctx.task_contract = build_task_contract({"id": parent, "type": "task", "text": "Scheduled check"})
    result = _schedule_task(ctx, subagent_id="api-builder", objective="Look at one thing", expected_output="fact",
                            memory_mode="empty")
    assert not result.startswith("⚠️"), result
    event = ctx.event_queue.get_nowait()
    return event, build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": parent,
                                                "text": event["objective"], "desc": event["objective"]})


def test_children_and_grandchildren_keep_the_clocks_provenance(q, monkeypatch):
    from tests.test_available_subagents_runtime import _api_row, _settings as actor_settings

    values = {**actor_settings(_api_row()), "OUROBOROS_SUBAGENTS_MODE": "swarm"}
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    task = _fire(q)
    with task_settings_scope(_actor(task, q.root, values)):
        event, child = _spawn(q.root, task["metadata"], parent=task["id"])
    room = {"id": task["id"], "root_task_id": task["id"], "metadata": event["origin_metadata"]}
    host = SimpleNamespace(DRIVE_ROOT=q.root)
    # The supervisor reads the same event before the child exists.
    assert _subagent_room_mode(host, room) == ("auto", None)
    assert check_delegation_admission(event["task_contract"]["delegation_budget"], room_mode="auto").ok
    assert child["metadata"]["schedule_writer_kind"] == "owner"
    with task_settings_scope(_actor(child, q.root, values)):
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO
        _event, grandchild = _spawn(q.root, child["metadata"], parent=child["id"])
    assert grandchild["metadata"]["schedule_writer_kind"] == "owner"
    with task_settings_scope(_actor(grandchild, q.root, values)):
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO
        # The room's Off reaches every generation: at the tool, and again at the supervisor's admission.
        values["OUROBOROS_SUBAGENTS_MODE"] = "off"
        assert subagents_fact_line() == FACT_OFF
        with pytest.raises(AssertionError, match="subagents_off_in_room"):
            _spawn(q.root, grandchild["metadata"], parent=grandchild["id"])
    assert _subagent_room_mode(host, room) == ("off", None)
    refused = check_delegation_admission(event["task_contract"]["delegation_budget"], room_mode="off")
    assert not refused.ok and refused.reason_code == "subagents_off_in_room"


def test_a_template_cannot_claim_another_writer(q, monkeypatch):
    _settings(monkeypatch, "swarm")
    _row(q, intent={"kind": "system_repo"}, metadata={"work_kind": "owner", "schedule_writer_kind": "forged"})
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["metadata"]["schedule_writer_kind"] == "owner", "the host's stamp, not the template's"


def test_a_restored_row_keeps_its_stamp_and_its_live_mode(q, monkeypatch):
    values = _settings(monkeypatch, "swarm")
    task = _fire(q)
    q.pending.clear()
    values["OUROBOROS_SUBAGENTS_MODE"] = "off"
    restored = q.queue.enqueue_task(json.loads(json.dumps(task)), restoring_snapshot=True)
    assert not restored.get("_admission_blocked")
    assert restored["metadata"]["subagents_mode"] == "auto", "the admission fact is kept"
    with task_settings_scope(_actor(restored, q.root)):
        assert live_subagents_mode() == "off", "the room's Off is live"
    values["OUROBOROS_SUBAGENTS_MODE"] = "swarm"
    with task_settings_scope(_actor(restored, q.root)):
        assert live_subagents_mode() == "auto"


def test_continue_of_a_real_scheduled_row_stays_on_auto_and_keeps_the_rooms_off(tmp_path, monkeypatch):
    """The predecessor is the row the real occurrence admitted, cancelled, then continued by the real admission."""
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    values = _settings(monkeypatch, "swarm")
    fired = _fire(SimpleNamespace(queue=queue, pending=workers.PENDING, root=tmp_path))
    assert load_task_result(tmp_path, fired["id"])["metadata"]["schedule_writer_kind"] == "owner"
    workers.PENDING.clear()
    write_task_result(tmp_path, fired["id"], "cancelled", result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"},
                      billing_group={"billing_group_id": fired["id"], "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    accepted = admit_continuation(fired["id"], action_nonce=NONCE)
    assert accepted["ok"], accepted
    successor = load_task_result(tmp_path, accepted["successor_task_id"])
    assert successor["metadata"]["schedule_writer_kind"] == "owner"
    assert successor["metadata"]["subagents_mode"] == "auto"
    with task_settings_scope(_actor(successor, tmp_path)):
        assert subagents_fact_line() == FACT_AUTO
        values["OUROBOROS_SUBAGENTS_MODE"] = "off"
        assert live_subagents_mode() == "off" and subagents_fact_line() == FACT_OFF


def test_continue_of_owner_work_keeps_the_rooms_swarm(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted

    _install_queue(tmp_path, monkeypatch)
    _settings(monkeypatch, "swarm")
    _interrupted(tmp_path)
    accepted = admit_continuation("pred-1", action_nonce=NONCE)
    assert accepted["ok"], accepted
    assert load_task_result(tmp_path, accepted["successor_task_id"])["metadata"]["subagents_mode"] == "swarm"


def test_ordinary_owner_work_stays_swarm(q, monkeypatch):
    _settings(monkeypatch, "swarm")
    owner = {"id": "owner-root", "type": "task", "text": "Think it through", "chat_id": 42, "root_task_id": "owner-root",
             "metadata": {"origin_message_ref": {"chat_id": 42, "client_message_id": "m1"}, "resource_intent": {"kind": "system_repo"}}}
    admitted = q.queue.enqueue_task(owner)
    assert admitted["metadata"]["subagents_mode"] == "swarm"
    with task_settings_scope(_actor(admitted, q.root)):
        assert live_subagents_mode() == "swarm"
        assert subagents_fact_line() == "Subagents: Swarm — set by my human for this chat"


def _wake(task_id="wake-1", **extra):
    """A consciousness wake as its producer stamps it (``consciousness_wake.wake_task_metadata``), at level Act."""
    from ouroboros.consciousness_wake import wake_task_metadata

    return {**wake_task_metadata("act", "heartbeat"), "task_id": task_id, "root_task_id": task_id,
            "resource_intent": {"kind": "system_repo"}, **extra}


def _fire_conscious(q, *, door="followup", project_id="", **extra):
    """A schedule a wake wrote, through the follow-up tool's writer or the host writer, fired by occurrence admission."""
    wake = _wake(**({"project_id": project_id, "resource_intent": {"kind": "explicit_none", "project_id": project_id}}
                    if project_id else {}), **extra)
    if door == "followup":
        from ouroboros.tools.followup import _register_followup

        ctx = SimpleNamespace(task_id="wake-1", task_metadata=wake, task_contract={})
        result = _register_followup(ctx, "wake-1", q.root, "Look at the backlog again", "",
                                    {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"}, "", "", "",
                                    {"task_id": "wake-1", "root_task_id": "wake-1"}, {"kind": "independent"})
        assert result.startswith("FOLLOWUP_SCHEDULED"), result
    else:
        _row(q, intent=wake["resource_intent"], project_id=project_id, metadata=wake)
    q.queue.check_scheduled_tasks()
    [task] = [row for row in q.pending if (row.get("metadata") or {}).get("schedule_occurrence")]
    assert (task["metadata"]["initiator"], task["metadata"]["schedule_writer_kind"]) == ("consciousness", "consciousness")
    return task


def _catalog_settings(monkeypatch, mode):
    from tests.test_available_subagents_runtime import _api_row, _settings as actor_settings

    values = {**actor_settings(_api_row()), "OUROBOROS_SUBAGENTS_MODE": mode}
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    return values


@pytest.mark.parametrize(("mode", "mode_seen", "fact", "door"), [
    pytest.param("swarm", "auto", FACT_AUTO, "followup", id='swarm-auto-Subagents: Auto — scheduled work-followup'),
    pytest.param("auto", "auto", FACT_AUTO, "writer", id='auto-auto-Subagents: Auto — scheduled work-writer'),
    pytest.param("off", "off", FACT_OFF, "followup", id='off-off-Subagents: Off — set by my human for this chat-followup'),
])
def test_a_schedule_consciousness_wrote_follows_its_room_like_any_scheduled_work(q, monkeypatch, door, mode, mode_seen, fact):
    """At every consumption point: the queue's admission fact, the live mode, the round's fact, the catalog the model
    sees and the tool's refusal."""
    from ouroboros.subagent_runtime import current_model_visible_subagent_catalog

    values = _catalog_settings(monkeypatch, mode)
    task = _fire_conscious(q, door=door)
    assert task["metadata"]["subagents_mode"] == load_task_result(q.root, task["id"])["metadata"]["subagents_mode"] == mode_seen
    with task_settings_scope(_actor(task, q.root, values)):
        assert live_subagents_mode() == effective_subagents_mode() == mode_seen
        assert subagents_fact_line() == fact
        assert bool(current_model_visible_subagent_catalog()) is (mode != "off")
        if mode == "off":
            with pytest.raises(SubagentSelectionError, match="off in this chat"):
                _require_room_subagents()
        else:
            _require_room_subagents()


@pytest.mark.parametrize("main,room,seen", [("swarm", "off", "off"), ("off", "swarm", "auto")])
def test_a_consciousness_schedule_reads_its_own_projects_room_not_mains(q, monkeypatch, main, room, seen):
    values = _settings(monkeypatch, main)
    project_id = _project(q)["id"]
    update_project_cognition(q.root, project_id, subagents_mode=room)
    task = _fire_conscious(q, door="writer", project_id=project_id)
    assert task["metadata"]["subagents_mode"] == seen
    with task_settings_scope(_actor(task, q.root, values)):
        assert live_subagents_mode() == seen
        assert subagents_fact_line() == (FACT_OFF if seen == "off" else FACT_AUTO)


@pytest.mark.parametrize("availability", ["disabled", "unsaved"])
def test_settings_off_holds_for_a_consciousness_schedule_in_a_swarm_room(q, monkeypatch, availability):
    _settings(monkeypatch, "swarm", availability)
    task = _fire_conscious(q)
    assert task["metadata"]["subagents_mode"] == "off"
    with task_settings_scope(_actor(task, q.root)):
        assert effective_subagents_mode() == "off" and subagents_fact_line() == FACT_SETTINGS_OFF
        with pytest.raises(SubagentSelectionError):
            _require_room_subagents()


def test_a_consciousness_schedules_children_follow_its_room_at_the_tool_and_the_supervisor(q, monkeypatch):
    values = _catalog_settings(monkeypatch, "swarm")
    task = _fire_conscious(q)
    with task_settings_scope(_actor(task, q.root, values)):
        event, child = _spawn(q.root, task["metadata"], parent=task["id"])
    origin = event["origin_metadata"]
    assert (origin["initiator"], origin["schedule_writer_kind"]) == ("consciousness", "consciousness")
    room = {"id": task["id"], "root_task_id": task["id"], "metadata": origin}
    host = SimpleNamespace(DRIVE_ROOT=q.root)
    assert _subagent_room_mode(host, room) == ("auto", None)
    assert (child["metadata"]["initiator"], child["metadata"]["schedule_writer_kind"]) == ("consciousness", "consciousness")
    with task_settings_scope(_actor(child, q.root, values)):
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO
        values["OUROBOROS_SUBAGENTS_MODE"] = "off"
        assert subagents_fact_line() == FACT_OFF
        with pytest.raises(AssertionError, match="subagents_off_in_room"):
            _spawn(q.root, child["metadata"], parent=child["id"])
    assert _subagent_room_mode(host, room) == ("off", None)
    refused = check_delegation_admission(event["task_contract"]["delegation_budget"], room_mode="off")
    assert not refused.ok and refused.reason_code == "subagents_off_in_room"


def test_continue_of_a_consciousness_schedule_is_admitted_under_the_rooms_off(tmp_path, monkeypatch):
    """Continue computes its successor's admission afresh: the room's Off at that moment, not the predecessor's Auto."""
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    values = _settings(monkeypatch, "swarm")
    fired = _fire_conscious(SimpleNamespace(queue=queue, pending=workers.PENDING, root=tmp_path), door="writer")
    assert fired["metadata"]["subagents_mode"] == "auto"
    workers.PENDING.clear()
    write_task_result(tmp_path, fired["id"], "cancelled", result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"},
                      billing_group={"billing_group_id": fired["id"], "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    values["OUROBOROS_SUBAGENTS_MODE"] = "off"
    accepted = admit_continuation(fired["id"], action_nonce=NONCE)
    assert accepted["ok"], accepted
    successor = load_task_result(tmp_path, accepted["successor_task_id"])
    assert (successor["metadata"]["initiator"], successor["metadata"]["schedule_writer_kind"]) == ("consciousness", "consciousness")
    assert successor["metadata"]["subagents_mode"] == "off"
    with task_settings_scope(_actor(successor, tmp_path)):
        assert live_subagents_mode() == "off" and subagents_fact_line() == FACT_OFF
        values["OUROBOROS_SUBAGENTS_MODE"] = "swarm"
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO


def test_a_restored_consciousness_schedule_keeps_its_fact_and_follows_the_live_room(q, monkeypatch):
    values = _settings(monkeypatch, "swarm")
    task = _fire_conscious(q)
    q.pending.clear()
    values["OUROBOROS_SUBAGENTS_MODE"] = "off"
    restored = q.queue.enqueue_task(json.loads(json.dumps(task)), restoring_snapshot=True)
    assert not restored.get("_admission_blocked") and restored["metadata"]["subagents_mode"] == "auto"
    with task_settings_scope(_actor(restored, q.root)):
        assert live_subagents_mode() == "off" and subagents_fact_line() == FACT_OFF


@pytest.mark.parametrize("room", ["off", "swarm"])
def test_negative_control_a_wake_and_what_it_starts_off_the_clock_keep_consciousness_controls(q, monkeypatch, room):
    """No scheduler stamp, no room: the wake its producer stamps, a root it starts and a child it schedules, at the
    queue's admission, a direct turn's admission, the live mode, the round's fact, the tool and the supervisor."""
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.settings_integrity import record_subagents_admission

    values = _settings(monkeypatch, room)
    wake = _wake()
    started = consciousness_origin_metadata(wake, started_root=True)
    child_origin = consciousness_origin_metadata(wake)
    assert not {"schedule_writer_kind"} & (set(wake) | set(started) | set(child_origin))
    for metadata in (wake, started):
        queued = {"id": "w-q", "type": "task", "chat_id": 1, "metadata": dict(metadata)}
        record_subagents_admission(queued, q.root, settings=values)
        direct = {"id": "w-d", "type": "task", "chat_id": 1, "metadata": dict(metadata)}
        record_subagents_admission(direct, q.root)
        assert queued["metadata"]["subagents_mode"] == direct["metadata"]["subagents_mode"] == "auto"
        with task_settings_scope(_actor(queued, q.root)):
            assert live_subagents_mode() == "auto" and subagents_fact_line() == ""
            _require_room_subagents()
    assert _subagent_room_mode(SimpleNamespace(DRIVE_ROOT=q.root), {
        "id": "wake-1", "root_task_id": "wake-1", "metadata": child_origin}) == ("auto", None)


@pytest.mark.parametrize("main,project_mode,seen", [("off", None, "off"), ("swarm", "off", "off"), ("off", "swarm", "auto")])
def test_presence_scheduled_work_uses_its_bound_room_or_main_and_keeps_off(q, monkeypatch, main, project_mode, seen):
    """A real due Presence promise obeys its room, regardless of the schedule author's autonomy."""
    from ouroboros.dialogue_provenance import PRESENCE_BINDING_AUTHORITY_KEY

    values = _catalog_settings(monkeypatch, main)
    project_id = _project(q)["id"] if project_mode is not None else ""
    if project_id:
        update_project_cognition(q.root, project_id, subagents_mode=project_mode)
    task = _fire_conscious(q, door="writer", project_id=project_id,
                          **{PRESENCE_BINDING_AUTHORITY_KEY: "place-binding"})
    assert task["metadata"]["subagents_mode"] == seen
    assert task["metadata"]["schedule_writer_kind"] == "consciousness"
    assert task["metadata"][PRESENCE_BINDING_AUTHORITY_KEY] == "place-binding"
    with task_settings_scope(_actor(task, q.root, values)):
        assert effective_subagents_mode() == live_subagents_mode() == seen
        assert subagents_fact_line() == (FACT_OFF if seen == "off" else FACT_AUTO)
        if seen == "off":
            with pytest.raises(SubagentSelectionError):
                _require_room_subagents()
        else:
            event, child = _spawn(q.root, task["metadata"], parent=task["id"], project_id=project_id)
            assert event["origin_metadata"]["schedule_writer_kind"] == "consciousness"
            assert child["project_id"] == project_id
            with task_settings_scope(_actor(child, q.root, values)):
                assert live_subagents_mode() == "auto"
                if project_id:
                    update_project_cognition(q.root, project_id, subagents_mode="off")
                else:
                    values["OUROBOROS_SUBAGENTS_MODE"] = "off"
                assert effective_subagents_mode() == "off"


def test_negative_control_unscheduled_evolve_and_presence_keep_their_own_controls(q, monkeypatch):
    _settings(monkeypatch, "off")
    for task in ({"id": "evo", "type": "evolution", "metadata": {"evolution_transaction": {"task_id": "evo"}}},
                 {"id": "turn", "type": "presence", "_presence_turn": True, "metadata": {}}):
        with task_settings_scope(_actor(task, q.root)):
            assert live_subagents_mode() == "auto" and subagents_fact_line() == "", task["id"]


def test_an_unstamped_row_still_reads_as_a_room_choice(q, monkeypatch):
    _settings(monkeypatch, "swarm")
    with task_settings_scope(_actor({"id": "plain", "metadata": {}}, q.root)):
        assert live_subagents_mode() == "swarm"


def test_an_api_caller_cannot_claim_the_clock(host, monkeypatch):
    from tests.test_project_surface_admission import request_for

    _settings(monkeypatch, "swarm")
    response = _create_task_from_body(request_for(host), {"task_id": "forged", "description": "Owner work",
        "metadata": {"schedule_writer_kind": "owner", "subagents_mode": "auto"}})
    assert response.status_code == 200, response.body
    [admitted] = host.pending
    assert "schedule_writer_kind" not in admitted["metadata"]
    assert admitted["metadata"]["subagents_mode"] == "swarm"


@pytest.mark.parametrize("carried,mode", [({"schedule_writer_kind": "owner"}, "auto"), ({}, "swarm")])
def test_a_root_a_scheduled_run_promotes_carries_on_from_the_clock(host, monkeypatch, carried, mode):
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    _settings(monkeypatch, "swarm")
    result = _handle_promote_chat_to_task({"task_id": "promoted", "routing_token": "promoted-token",
        "objective": "Work the run started", "project_id": "", "chat_id": 1, "source_text": "Work the run started",
        "source_ref": build_owner_message_ref(chat_id=1, client_message_id="promoted-origin",
                                              ts="2026-11-03T00:00:00Z", text="Work the run started"), **carried}, host.ctx)
    assert result["status"] == "scheduled", result
    [admitted] = host.pending
    assert admitted["metadata"]["subagents_mode"] == mode


def test_a_new_owner_message_keeps_its_own_swarm_label_and_scheduled_work_stays_auto(q, monkeypatch):
    """Two facts of two origins: the message says what its room held when the host accepted it, the running work keeps
    the clock's Auto. The label reads the room alone, and nothing of it reaches the task."""
    from supervisor import message_ingress
    from tests.test_chat_attachments import _web_bridge

    monkeypatch.setattr(config, "DATA_DIR", q.root)
    _settings(monkeypatch, "swarm")
    task = _fire(q)
    before = json.dumps(task, sort_keys=True, default=str)
    bridge, echoes = _web_bridge(q.root, monkeypatch)
    bridge.ui_send("And check the second region too", broadcast=True, sender_session_id="surface",
                   client_message_id="while-scheduled", chat_id=42, project_id="")
    row = message_ingress.accepted_chat_message(q.root, 42, "while-scheduled")
    assert row["subagents_mode"] == "swarm" and echoes[0]["subagents_mode"] == "swarm"
    assert json.dumps(task, sort_keys=True, default=str) == before
    assert task["metadata"]["subagents_mode"] == "auto"
    with task_settings_scope(_actor(task, q.root)):
        assert live_subagents_mode() == "auto" and subagents_fact_line() == FACT_AUTO


def test_auto_scheduled_work_keeps_its_children_runs_and_refusals_on_record(q, monkeypatch):
    """Dropping the Swarm stamp drops the Swarm line and its telemetry, not one fact of what the work did."""
    from ouroboros import delegate_custody as custody, swarm_receipt
    from ouroboros.task_finalization import build_swarm_efficiency
    from ouroboros.tools.control_scheduling import maybe_emit_delegated_run_fanout
    from ouroboros.tools.registry import ToolContext

    _settings(monkeypatch, "swarm")
    task = _fire(q)
    assert build_swarm_efficiency(SimpleNamespace(drive_root=q.root), task) is None, "no Swarm intent, no zero block"
    ctx = ToolContext(repo_dir=q.root, drive_root=q.root, task_id=task["id"], task_metadata=dict(task["metadata"]))
    maybe_emit_delegated_run_fanout(ctx, run_id="run-direct", route_id="route", objective="o", durable=True)
    assert not (q.root / "logs" / "events.jsonl").exists() or "swarm_fanout" not in (q.root / "logs" / "events.jsonl").read_text()
    write_task_result(q.root, "child", "completed", root_task_id=task["id"], parent_task_id=task["id"],
                      started_at="2026-11-03T20:00:00+00:00")
    for invocation in ("started", "refused"):
        assert custody.record_start_requested(q.root, task_id=task["id"], root_task_id=task["id"], invocation_id=invocation,
                                              request={"prompt": "work"})
    assert custody.record_started(q.root, custody.RunCustody(run_id="run-direct", task_id=task["id"], root_task_id=task["id"],
                                                             invocation_id="started"))
    assert custody.emit(q.root, custody.START_FAILED, {"task_id": task["id"], "invocation_id": "refused", "definite": True})
    published = swarm_receipt.publish(q.root, task["id"])
    assert {"task:child", "run:run-direct"} <= {actor["id"] for actor in published["actor_intervals"]}
    assert [row["invocation_id"] for row in published["refused_starts"]] == ["refused"]
