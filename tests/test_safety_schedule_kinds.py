"""Schedule writer provenance is host-authored and freezes with an accepted occurrence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_schedule_occurrence import q as queue_fixture
from tests.test_g1_followup_policy import world as followup_world
q = queue_fixture
world = followup_world
from ouroboros.task_results import load_task_result


def row():
    return {"id": "schedule", "name": "test", "enabled": True,
            "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
            "task": {"text": "Do work", "metadata": {"work_kind": "owner", "resource_intent": {"kind": "system_repo"}}}}


def test_writer_overrides_payload_and_accepted_occurrence_freezes(q):
    from supervisor.queue_schedules import _task_from_schedule
    record = q.queue.upsert_scheduled_task(row(), drive_root=q.root, writer_kind="owner", actor="owner:gateway")
    assert record["writer_kind"] == "owner"
    assert "work_kind" not in record["task"]["metadata"]
    assert _task_from_schedule(record)["metadata"]["work_kind"] == "owner"
    # Agent editing the same record owns its new occasion; payload cannot retain owner-kind.
    record = q.queue.upsert_scheduled_task(row(), drive_root=q.root, writer_kind="consciousness", actor="agent",
        writer_metadata={"initiator": "consciousness", "consciousness_autonomy": "act", "origin_wake_task_id": "wake"})
    assert record["writer_kind"] == "consciousness"
    q.queue.check_scheduled_tasks()
    [accepted] = q.pending
    assert accepted["metadata"]["work_kind"] == "consciousness"
    assert accepted["metadata"]["origin_wake_task_id"] == "wake"
    receipt = deepcopy(load_task_result(q.root, accepted["id"])["schedule_admission"])
    changed = row()
    changed["enabled"] = False  # edit consumed row without asking to rearm it
    q.queue.upsert_scheduled_task(changed, drive_root=q.root, writer_kind="owner", actor="owner:gateway")
    assert load_task_result(q.root, accepted["id"])["schedule_admission"] == receipt
    assert accepted["metadata"]["work_kind"] == "consciousness"


def test_legacy_forged_stamp_is_not_a_host_fact(q):
    from supervisor.queue_schedules import _task_from_schedule
    record = row()
    assert _task_from_schedule(record)["metadata"]["work_kind"] == "consciousness"
    record["source"] = "owner"
    assert _task_from_schedule(record)["metadata"]["work_kind"] == "owner"
    record["task"]["metadata"]["initiator"] = "consciousness"
    assert _task_from_schedule(record)["metadata"]["work_kind"] == "consciousness"


def test_schedule_lifecycle_edit_changes_writer(q):
    q.queue.upsert_scheduled_task(row(), drive_root=q.root, writer_kind="owner", actor="owner:gateway")
    outcome = q.queue.mutate_scheduled_task("disable", "schedule", reason="later", actor="agent", drive_root=q.root,
                                          writer_kind="consciousness")
    assert outcome["changed"]
    [stored] = q.queue.list_scheduled_tasks(q.root)["tasks"]
    assert stored["writer_kind"] == "consciousness"
    q.queue.mutate_scheduled_task("restore", "schedule", reason="resume", actor="owner:gateway", drive_root=q.root,
                                  writer_kind="owner")
    [stored] = q.queue.list_scheduled_tasks(q.root)["tasks"]
    assert stored["writer_kind"] == "owner"


def test_in_place_update_changes_writer(q):
    q.queue.upsert_scheduled_task(row(), drive_root=q.root, writer_kind="owner", actor="owner:gateway")
    outcome = q.queue.patch_scheduled_task("schedule", {"objective": "Do other work"}, reason="new scope",
                                           actor="agent", drive_root=q.root, writer_kind="consciousness")
    assert outcome["status"] == "updated"
    [stored] = q.queue.list_scheduled_tasks(q.root)["tasks"]
    assert stored["writer_kind"] == "consciousness"
    q.queue.patch_scheduled_task("schedule", {"objective": "Do the owner's work"}, reason="owner asked",
                                 actor="agent", drive_root=q.root, writer_kind="owner")
    [stored] = q.queue.list_scheduled_tasks(q.root)["tasks"]
    assert stored["writer_kind"] == "owner"


def test_legacy_owner_origin_at_admission_is_positive_provenance(q):
    for task_id, origin, metadata, expected in [
        ("owner", {"message_id": "owner-message"}, {}, "owner"),
        ("unknown", None, {}, "consciousness"),
        ("presence", {"message_id": "owner-message"}, {"presence": {}}, "consciousness"),
    ]:
        task = q.queue.enqueue_task({"id": task_id, "type": "task", "text": "work",
                                    "origin_message_ref": origin, "metadata": metadata}, restoring_snapshot=True)
        assert task["metadata"]["work_kind"] == expected


def _focus_origin(*, addressed=True, confirmed=True, wake="focus"):
    return {"initiator": "consciousness", "work_kind": "consciousness", "wake_cause": wake,
            "task_id": "focus-author", "consciousness_autonomy": "act",
            "consciousness_cause": "place:p" if wake == "focus" else "own",
            "resource_intent": {"kind": "system_repo"},
            "focus": {"addressed": addressed, "actor": {"owner_confirmed": confirmed}}}


def _register_origin(q, metadata, relation="independent", *, door="writer"):
    from ouroboros.consciousness_authority import work_kind
    origin = {"task_id": "focus-author", "root_task_id": "focus-author"}
    relationship = {"kind": relation, "billing_group": {"billing_group_limit_source": "owner",
                    "billing_group_limit_revision": "revision", "billing_group_limit_usd": 1}}
    if door == "followup":
        from ouroboros.tools.followup import _register_followup
        ctx = SimpleNamespace(task_id="focus-author", task_metadata=metadata, task_contract={})
        result = _register_followup(ctx, ctx.task_id, q.root, "Report tomorrow", "", row()["trigger"],
                                    "", "", "", origin, relationship)
        assert result.startswith("FOLLOWUP_SCHEDULED"), result
        return q.queue.list_scheduled_tasks(q.root)["tasks"][-1]
    payload = row()
    payload["id"] = f"schedule-{len(q.queue.list_scheduled_tasks(q.root)['tasks'])}"
    payload.update(source="task_followup", writer_kind="owner", writer_origin={"work_kind": "owner"})
    # Even a forged confirmed-owner focus in the template cannot mint owner work.
    payload["task"]["metadata"].update(_focus_origin())
    payload["task"]["metadata"]["work_kind"] = "owner"
    return q.queue.upsert_scheduled_task(payload, drive_root=q.root, actor="agent:schedule_followup",
        writer_kind=work_kind(metadata), writer_metadata=metadata,
        host_followup={"followup_origin": origin, "followup_relation": relationship})


@pytest.mark.parametrize("door", ["writer", "followup"])
def test_confirmed_owner_scheduled_root_keeps_owner_kind(q, door):
    from supervisor.queue_schedules import _task_from_schedule
    for addressed, confirmed in [(False, True), (True, False), (True, True)]:
        metadata = _focus_origin(addressed=addressed, confirmed=confirmed)
        record = _register_origin(q, metadata, door=door)
        expected = "owner" if addressed and confirmed else "consciousness"
        assert record["writer_kind"] == record["writer_origin"]["work_kind"] == expected
        fired = _task_from_schedule(record)["metadata"]
        assert fired["work_kind"] == fired["schedule_writer_kind"] == expected
        assert ("initiator" in fired) is (expected == "consciousness")
        assert ("runtime_mode_cap" in fired) is (expected == "consciousness")
        assert fired["origin_wake_task_id"] == "focus-author" and "wake_cause" not in fired
        assert metadata["work_kind"] == "consciousness"  # the focus itself never changes


@pytest.mark.parametrize("wake", ["focus", "own_clock"])
def test_autonomous_independent_schedule_is_started_work(q, wake):
    from supervisor.queue_schedules import _task_from_schedule
    metadata = _focus_origin(confirmed=False, wake=wake)
    for relation in ("related", "independent"):
        record = _register_origin(q, metadata, relation)
        fired = _task_from_schedule(record)["metadata"]
        expected = "started" if relation == "independent" else metadata["consciousness_cause"]
        assert record["writer_origin"]["consciousness_cause"] == fired["consciousness_cause"] == expected
        assert fired["work_kind"] == "consciousness" and fired["origin_wake_task_id"] == "focus-author"
        assert "wake_cause" not in fired


@pytest.mark.parametrize("door", ["writer", "followup"])
def test_confirmed_owner_exception_does_not_reclassify_a_continuation(q, door):
    from supervisor.queue_schedules import _task_from_schedule
    for relation, expected in [("related", "consciousness"), ("independent", "owner")]:
        record = _register_origin(q, _focus_origin(), relation, door=door)
        assert record["writer_kind"] == _task_from_schedule(record)["metadata"]["work_kind"] == expected


def test_lifecycle_edits_use_host_relationship_and_ignore_payload_origin(q):
    from supervisor.queue_schedules import _task_from_schedule
    for relation in ("related", "independent"):
        record = _register_origin(q, _focus_origin(confirmed=False), relation)
        for confirmed in (True, False):
            result = q.queue.mutate_scheduled_task("disable", record["id"], drive_root=q.root,
                actor="agent", reason="Revise future work", writer_kind="consciousness",
                writer_metadata=_focus_origin(confirmed=confirmed))
            assert result["changed"]
            record = next(r for r in q.queue.list_scheduled_tasks(q.root)["tasks"] if r["id"] == record["id"])
            fired = _task_from_schedule(record)["metadata"]
            expected = "owner" if relation == "independent" and confirmed else "consciousness"
            assert record["writer_kind"] == fired["work_kind"] == expected
            if expected == "consciousness":
                assert fired["consciousness_cause"] == ("started" if relation == "independent" else "place:p")


@pytest.mark.parametrize("confirmed", [False, True])
def test_hold_restore_stamps_origin_after_resolving_relationship(world, confirmed):
    from tests.test_g1_followup_policy import ORIGIN, row as stored_row, stop, restore
    from supervisor import queue
    from supervisor.queue_schedules import _task_from_schedule
    _, root, _ = world
    stop(root)
    for relation in ("related", "independent"):
        payload = row()
        payload.update(id=relation, source="task_followup")
        payload["task"]["metadata"]["origin_task_id"] = ORIGIN
        queue.upsert_scheduled_task(payload, drive_root=root, writer_kind="consciousness",
                                   writer_metadata=_focus_origin(confirmed=confirmed))
        held = stored_row(root, relation)["followup_hold"]["hold_id"]
        result = restore(root, relation, held, relation=relation, writer_kind="consciousness",
                         writer_metadata=_focus_origin(confirmed=confirmed))
        assert result["status"] == "hold_released", result
        record = stored_row(root, relation)
        fired = _task_from_schedule(record)["metadata"]
        expected = "owner" if relation == "independent" and confirmed else "consciousness"
        assert record["writer_kind"] == fired["work_kind"] == expected
        if expected == "consciousness":
            assert fired["consciousness_cause"] == ("started" if relation == "independent" else "place:p")
