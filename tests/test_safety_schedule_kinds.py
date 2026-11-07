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


CITE = {"source": "place_owner", "content": "Report tomorrow", "basis": "cited", "carried": True}


def _focus_origin(*, cited=False, confirmed=True, wake="focus"):
    """A place focus's metadata as the schedule doors see it; ``cited``: the writer passed a host-verified cite."""
    return {"initiator": "consciousness", "work_kind": "consciousness", "wake_cause": wake,
            "task_id": "focus-author", "consciousness_autonomy": "act",
            "consciousness_cause": "place:p" if wake == "focus" else "own",
            "resource_intent": {"kind": "system_repo"},
            "focus": {"actor": {"owner_confirmed": confirmed}}, **({"owner_basis": CITE} if cited else {})}


def _register_origin(q, metadata, relation="independent", *, door="writer"):
    from ouroboros.consciousness_authority import work_kind
    origin = {"task_id": "focus-author", "root_task_id": "focus-author"}
    relationship = {"kind": relation, "billing_group": {"billing_group_limit_source": "owner",
                    "billing_group_limit_revision": "revision", "billing_group_limit_usd": 1}}
    if door == "followup":
        from ouroboros.tools.followup import _register_followup
        ctx = SimpleNamespace(task_id="focus-author", task_metadata=metadata, task_contract={})
        result = _register_followup(ctx, ctx.task_id, q.root, "Report tomorrow", "", row()["trigger"],
                                    "", "", "", origin, relationship, owner_basis=metadata.get("owner_basis"))
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
def test_only_a_cited_scheduled_root_keeps_owner_kind(q, door):
    """A schedule or follow-up a place focus writes is its own work; one written on my human's cited words is theirs
    (owner 20.10: their requests from a confirmed account are their orders), whatever the focus's actor flags say."""
    from supervisor.queue_schedules import _task_from_schedule
    for cited, confirmed in [(False, True), (True, False), (True, True)]:
        metadata = _focus_origin(cited=cited, confirmed=confirmed)
        record = _register_origin(q, metadata, door=door)
        expected = "owner" if cited else "consciousness"
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
def test_a_cite_does_not_reclassify_a_continuation(q, door):
    from supervisor.queue_schedules import _task_from_schedule
    for relation, expected in [("related", "consciousness"), ("independent", "owner")]:
        record = _register_origin(q, _focus_origin(cited=True), relation, door=door)
        assert record["writer_kind"] == _task_from_schedule(record)["metadata"]["work_kind"] == expected


def test_lifecycle_edits_use_host_relationship_and_ignore_payload_origin(q):
    from supervisor.queue_schedules import _task_from_schedule
    for relation in ("related", "independent"):
        record = _register_origin(q, _focus_origin(confirmed=False), relation)
        for cited in (True, False):
            result = q.queue.mutate_scheduled_task("disable", record["id"], drive_root=q.root,
                actor="agent", reason="Revise future work", writer_kind="consciousness",
                writer_metadata=_focus_origin(cited=cited))
            assert result["changed"]
            record = next(r for r in q.queue.list_scheduled_tasks(q.root)["tasks"] if r["id"] == record["id"])
            fired = _task_from_schedule(record)["metadata"]
            expected = "owner" if relation == "independent" and cited else "consciousness"
            assert record["writer_kind"] == fired["work_kind"] == expected
            if expected == "consciousness":
                assert fired["consciousness_cause"] == ("started" if relation == "independent" else "place:p")


def _stored(q, schedule_id):
    return next(r for r in q.queue.list_scheduled_tasks(q.root)["tasks"] if r["id"] == schedule_id)


def test_a_pause_or_restore_keeps_the_cite_a_schedule_runs_on(q):
    """Review R1 MINOR-1 and Astra finding 1: «Напомни мне каждый день…» created on my human's cited words stays theirs
    when a later place turn pauses, restores or re-times it without a cite (the cite lives on the schedule's next
    occurrence); an owner-kind record that stores no cite is re-stamped from the writer, as before. The host carries
    no old declaration onto new work: an uncited rewrite of the objective is the writer's own, and a rewrite that
    passes a verified cite is my human's on that cite."""
    from ouroboros.tools.followup import _update_schedule
    from supervisor.queue_schedules import _task_from_schedule
    cited = _register_origin(q, _focus_origin(cited=True), "independent", door="followup")
    assert cited["writer_kind"] == "owner" and CITE in cited["task"]["metadata"]["owner_corpus"]
    main = _register_origin(q, {"work_kind": "owner", "task_id": "main-author"}, "independent")
    assert main["writer_kind"] == "owner" and not any(row.get("basis") == "cited" for row in main["task"]["metadata"].get("owner_corpus") or [])
    writer = SimpleNamespace(task_id="focus-author", task_metadata=_focus_origin(cited=False), task_contract={})
    for action in ("disable", "restore", "retime"):
        for record, expected in ((cited, "owner"), (main, "consciousness")):
            if action == "retime":  # a timing-only edit: the same work, moved
                _update_schedule(writer, q.root, record["id"], "Move it", {"run_at": "2099-01-01T12:00:00+00:00"})
                assert _stored(q, record["id"])["trigger"] == {"type": "once", "run_at": "2099-01-01T12:00:00+00:00"}
            else:
                result = q.queue.mutate_scheduled_task(action, record["id"], drive_root=q.root, actor="agent",
                    reason="Pause it for now", writer_kind="consciousness", writer_metadata=_focus_origin(cited=False))
                assert result["changed"], (action, record["id"], result)
            stored = _stored(q, record["id"])
            fired = _task_from_schedule(stored)["metadata"]
            assert stored["writer_kind"] == fired["work_kind"] == expected, (action, record["id"])
            assert ("consciousness_cause" in fired) is (expected == "consciousness")
    # An uncited replacement of the assignment is the writer's own work; the old cite minted nothing new.
    _update_schedule(writer, q.root, cited["id"], "Replace it with my own investigation",
                     {"objective": "Investigate my independent optimization idea"})
    stored = _stored(q, cited["id"])
    assert stored["task"]["text"] == "Investigate my independent optimization idea"
    assert stored["writer_kind"] == _task_from_schedule(stored)["metadata"]["work_kind"] == "consciousness"
    # A replacement that passes a verified cite is my human's work on that cite, stored where Off and the lifecycle read.
    again = {**CITE, "content": "Report tomorrow, and the margins too", "cited": "the margins too"}
    _update_schedule(writer, q.root, cited["id"], "They asked for the margins", {"objective": "Report the margins"}, owner_basis=again)
    stored = _stored(q, cited["id"])
    assert stored["task"]["text"] == "Report the margins" and again in stored["task"]["metadata"]["owner_corpus"]
    assert stored["writer_kind"] == _task_from_schedule(stored)["metadata"]["work_kind"] == "owner"


@pytest.mark.parametrize("door", ["notification", "time_only_update"])
def test_every_cited_producer_stores_the_cite_where_off_and_the_lifecycle_read(q, door):
    """Astra finding 2: a notify=true reminder and a timing-only update that pass a verified cite store it on the
    record's next occurrence (the one home), so Off names the schedule and a pause or restore keeps owner kind; the
    same doors without a cite leave the writer's own work, unlisted."""
    from ouroboros.tools.followup import _register_note, _update_schedule
    from supervisor.consciousness_carriers import cited_owner_schedules
    from supervisor.queue_schedules import _task_from_schedule
    ids = {}
    for label, basis in (("cited", CITE), ("own", None)):
        ctx = SimpleNamespace(task_id="focus-author", task_metadata=_focus_origin(), task_contract={})
        if door == "notification":
            result = _register_note(ctx, ctx.task_id, q.root, "Report tomorrow",
                                    {"type": "once", "run_at": "2099-01-01T12:00:00+00:00"}, "", "", "",
                                    {"task_id": ctx.task_id, "root_task_id": ctx.task_id}, {"kind": "independent"}, owner_basis=basis)
            assert result.startswith("FOLLOWUP_SCHEDULED"), result
            ids[label] = q.queue.list_scheduled_tasks(q.root)["tasks"][-1]["id"]
        else:
            ids[label] = _register_origin(q, _focus_origin(), door="followup")["id"]
            _update_schedule(ctx, q.root, ids[label], "Move it as requested", {"run_at": "2099-01-01T12:00:00+00:00"}, owner_basis=basis)
    for label, expected in (("cited", "owner"), ("own", "consciousness")):
        stored = _stored(q, ids[label])
        assert stored["writer_kind"] == expected and (CITE in (stored["task"]["metadata"].get("owner_corpus") or [])) is (expected == "owner"), (door, label)
        assert (ids[label] in dict(cited_owner_schedules(q.root))) is (expected == "owner"), (door, label)
        for action in ("disable", "restore"):
            result = q.queue.mutate_scheduled_task(action, ids[label], drive_root=q.root, actor="agent",
                reason="Pause or resume the same work", writer_kind="consciousness", writer_metadata=_focus_origin())
            assert result["changed"], (door, label, action, result)
            stored = _stored(q, ids[label])
            assert stored["writer_kind"] == expected, (door, label, action)
            if door != "notification":  # a note fires as a System reminder, never as a task
                assert _task_from_schedule(stored)["metadata"]["work_kind"] == expected, (door, label, action)
    assert dict(cited_owner_schedules(q.root)).get(ids["cited"]) == "Report tomorrow"


@pytest.mark.parametrize("cited", [False, True])
def test_hold_restore_stamps_origin_after_resolving_relationship(world, cited):
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
                                   writer_metadata=_focus_origin(cited=cited))
        held = stored_row(root, relation)["followup_hold"]["hold_id"]
        result = restore(root, relation, held, relation=relation, writer_kind="consciousness",
                         writer_metadata=_focus_origin(cited=cited))
        assert result["status"] == "hold_released", result
        record = stored_row(root, relation)
        fired = _task_from_schedule(record)["metadata"]
        expected = "owner" if relation == "independent" and cited else "consciousness"
        assert record["writer_kind"] == fired["work_kind"] == expected
        if expected == "consciousness":
            assert fired["consciousness_cause"] == ("started" if relation == "independent" else "place:p")
