"""#1315: a due schedule becomes at most one admitted root, on the existing carriers.

Driven through the real ``check_scheduled_tasks`` over an isolated data root: claims,
off-lock preparation, the recheck/admit transaction, each durable boundary failing,
the dispatch barrier, snapshot restore, resource intent and the continuation exemption.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result


@pytest.fixture
def q(tmp_path, monkeypatch):
    from supervisor import queue, queue_schedules, state, workers

    root = tmp_path / "data"  # project folders live beside, never inside, the data root
    root.mkdir()
    state.init(root)
    state.save_state({"owner_chat_id": 1})
    queue.init(root)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(queue_schedules, "resync_skill_schedules", lambda *_a: {})
    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)  # waits end at once here
    return SimpleNamespace(queue=queue, pending=pending, root=root)


def _row(q, schedule_id="s1", *, intent=None, cron=False, project_id="", chat_id=42, metadata=None, **extra):
    meta = dict(metadata or {})
    if intent is not None:
        meta["resource_intent"] = intent
    trigger = ({"type": "cron", "expr": "* * * * *"} if cron
               else {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"})
    record = {"id": schedule_id, "name": "Follow-up", "enabled": True, "source": "task_followup",
              "trigger": trigger, "task": {"type": "task", "text": f"continue {schedule_id}", "chat_id": chat_id,
                                           **({"project_id": project_id} if project_id else {}), "metadata": meta},
              **({"next_run_at": "2000-01-01T00:00:00+00:00"} if cron else {})}
    q.queue.upsert_scheduled_task({**record, **{k: v for k, v in extra.items() if k != "continuation_of"}},
                                  continuation_of=extra.get("continuation_of"),
                                  host_followup={"followup_relation": {"kind": "independent"},
                                      "followup_origin": {"task_id": meta.get("origin_task_id", ""),
                                                          "root_task_id": meta.get("origin_root_task_id")
                                                          or meta.get("origin_task_id", "")}})


def _rows(q):
    return {row["id"]: row for row in q.queue.list_scheduled_tasks(q.root)["tasks"]}


def _project(q, pid="proj", folder=None):
    from ouroboros.projects_registry import create_project

    return create_project(q.root, pid, name="Room", working_dir=str(folder or ""))


def test_one_occurrence_one_receipt_with_the_room_address(q):
    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    receipt = load_task_result(q.root, task["id"])
    admission = receipt["schedule_admission"]
    assert receipt["status"] == "scheduled" and receipt["chat_id"] == 42  # never the hidden chat 0
    assert (admission["status"], admission["dispatch"]) == ("accepted", "none")
    assert task["metadata"]["schedule_occurrence"]["token"] == admission["token"]
    row = _rows(q)["s1"]
    assert row["occurrence"]["phase"] == "admitted" and row["enabled"] is False and row["completed_at"]


def test_capacity_waits_on_the_row_without_phantom_roots(q, monkeypatch):
    from ouroboros import consciousness_allowance
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "1")
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _root: {
        "status": "available", "limit_usd": 10.0, "settled_usd": 0.0, "accounted_usd": 0.0, "unknown_unmetered": 0, "resets_at": ""})
    q.pending.append({"id": "live-wake", "delegation_role": "root", "metadata": {"initiator": "consciousness"}})
    _row(q, intent={"kind": "system_repo"}, metadata={"initiator": "consciousness"})
    for _ in range(3):
        q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert row["hold"]["reason"] == "consciousness_task_limit" and not row.get("failure_count")
    assert not list((q.root / "task_results").glob("*.json"))  # no failed root, ever
    held_task = row["occurrence"]["task_id"]
    token = row["occurrence"]["token"]
    assert row["occurrence"]["phase"] == "claimed"
    # Reconstruct the live queue; the persisted claim carries recovery across boot.
    q.pending = []
    q.queue.init(q.root)
    q.queue.init_queue_refs(q.pending, {}, {"value": 0})
    for _ in range(3):
        q.queue.check_scheduled_tasks()
    assert [task["id"] for task in q.pending] == [held_task]  # the SAME occurrence runs
    receipt = load_task_result(q.root, held_task)["schedule_admission"]
    assert receipt["token"] == token and receipt["dispatch"] == "none"
    assert "admission" not in _rows(q)["s1"]["occurrence"]


def test_a_named_continuation_is_outside_the_cap_but_not_outside_money(q, monkeypatch):
    from ouroboros import consciousness_allowance

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "1")
    window = {"status": "available", "limit_usd": 10.0, "settled_usd": 0.0, "accounted_usd": 0.0, "unknown_unmetered": 0, "resets_at": ""}
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _root: dict(window))
    q.pending.append({"id": "live-wake", "delegation_role": "root", "metadata": {"initiator": "consciousness"}})
    wake = {"initiator": "consciousness"}
    _row(q, "cont", intent={"kind": "system_repo"}, metadata=wake, continuation_of={"task_id": "t0"})
    _row(q, "spont", intent={"kind": "system_repo"}, metadata=wake)
    # A payload cannot author the host fact.
    q.queue.upsert_scheduled_task({**_rows(q)["spont"], "continuation_of": {"task_id": "forged"}})
    assert "continuation_of" not in _rows(q)["spont"]
    q.queue.check_scheduled_tasks()
    rows = _rows(q)
    assert [t["id"] for t in q.pending if t["id"] != "live-wake"] == [rows["cont"]["occurrence"]["task_id"]]
    assert rows["spont"]["hold"]["reason"] == "consciousness_task_limit"
    assert q.queue.live_consciousness_root_count() == 1  # the continuation is not counted
    window.update(status="exhausted", settled_usd=10.0, accounted_usd=10.0)
    _row(q, "cont2", intent={"kind": "system_repo"}, metadata=wake, continuation_of={"task_id": "t0"})
    q.queue.check_scheduled_tasks()
    assert _rows(q)["cont2"]["hold"]["reason"] == "consciousness_allowance_exhausted"


@pytest.mark.parametrize("case", ["room_current", "room_folderless", "explicit_kept", "explicit_gone",
                                  "explicit_none", "registry_unreadable"])
def test_resource_intent_is_resolved_at_admission(q, tmp_path, case):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    if case == "room_current":
        _project(q, folder=old)
        _row(q, intent={"kind": "room_default", "project_id": "proj"}, project_id="proj")
        from ouroboros.projects_registry import update_project

        update_project(q.root, "proj", working_dir=str(new))  # rebound before the occurrence
    elif case == "room_folderless":
        _project(q)
        _row(q, intent={"kind": "room_default", "project_id": "proj"}, project_id="proj")
    elif case in {"explicit_kept", "explicit_gone"}:
        _project(q, folder=new)
        _row(q, intent={"kind": "explicit_resource", "root": str(old)}, project_id="proj")
        if case == "explicit_gone":
            old.rmdir()
    elif case == "explicit_none":
        _project(q, folder=new)
        _row(q, intent={"kind": "explicit_none", "project_id": "proj"}, project_id="proj")
    else:
        _project(q, folder=new)
        _row(q, intent={"kind": "room_default", "project_id": "proj"}, project_id="proj")
        (q.root / "state" / "projects.json").write_text("{not json", encoding="utf-8")
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    if case in {"explicit_gone", "registry_unreadable"}:
        assert q.pending == [] and row["hold"]["reason"] == (
            "workspace_unusable" if case == "explicit_gone" else "registry_unreadable")
        assert not list((q.root / "task_results").glob("*.json"))
        return
    [task] = q.pending
    expected = {"room_current": new, "explicit_kept": old}.get(case)
    assert task.get("workspace_root", "") == (str(expected.resolve()) if expected else "")
    if expected:
        assert task["workspace_mode"] == "external" and task["memory_mode"] == "forked"
        assert pathlib.Path(task["drive_root"]).is_dir()
    assert task["metadata"]["resource_intent"]["kind"] == {
        "room_current": "room_default", "room_folderless": "room_default",
        "explicit_kept": "explicit_resource", "explicit_none": "explicit_none"}[case]


@pytest.mark.parametrize("origin", ["external_workspace", "main", "missing", "project_without_folder",
                                    "unaddressed", "bound_to_project"])
def test_a_legacy_followup_recovers_only_from_its_origin_record(q, tmp_path, origin):
    from ouroboros.task_results import write_task_result

    folder = tmp_path / "ext"
    folder.mkdir()
    if origin == "external_workspace":
        write_task_result(q.root, "origin1", "completed", workspace_root=str(folder), workspace_mode="external")
    elif origin == "main":
        write_task_result(q.root, "origin1", "completed", chat_id=1)  # destination alone is no workspace authority
    elif origin == "unaddressed":
        write_task_result(q.root, "origin1", "completed")  # bare absence proves nothing
    elif origin == "bound_to_project":
        from ouroboros.projects_registry import bind_task_to_project

        _project(q)
        write_task_result(q.root, "origin1", "completed", chat_id=1)
        bind_task_to_project(q.root, "origin1", "proj", origin={"absent": "system"})
    elif origin == "project_without_folder":
        write_task_result(q.root, "origin1", "completed", project_id="proj")
    _row(q, metadata={"origin_task_id": "origin1"})
    q.queue.check_scheduled_tasks()
    if origin in {"missing", "main", "project_without_folder", "unaddressed", "bound_to_project"}:
        assert q.pending == [] and _rows(q)["s1"]["hold"]["reason"] == "resource_intent_unknown"
        return
    [task] = q.pending
    if origin == "external_workspace":
        assert task["workspace_root"] == str(folder.resolve())
        assert task["metadata"]["resource_intent"]["recovered_from"] == "origin1"
    else:
        assert not task.get("workspace_root") and task["metadata"]["resource_intent"]["kind"] == "system_repo"


@pytest.mark.parametrize("change", ["disable", "edit", "rebind", "delete"])
def test_a_change_during_prepare_never_launches_the_old_choice(q, tmp_path, monkeypatch, change):
    from supervisor import schedule_occurrence as occurrences

    folder, moved = tmp_path / "f1", tmp_path / "f2"
    folder.mkdir()
    moved.mkdir()
    _project(q, folder=folder)
    _row(q, intent={"kind": "room_default", "project_id": "proj"}, project_id="proj")
    real_prepare = occurrences.prepare

    def prepare_then_change(claimed):
        prepared = real_prepare(claimed)
        if change == "disable":
            q.queue.mutate_scheduled_task("disable", "s1", reason="owner", actor="owner")
        elif change == "edit":
            q.queue.upsert_scheduled_task({**_rows(q)["s1"], "task": {**_rows(q)["s1"]["task"], "text": "edited"}})
        elif change == "rebind":
            from ouroboros.projects_registry import update_project

            update_project(q.root, "proj", working_dir=str(moved))
        else:
            q.queue.mutate_scheduled_task("delete", "s1", reason="owner", actor="owner")
        return prepared

    monkeypatch.setattr(occurrences, "prepare", prepare_then_change)
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and not list((q.root / "task_results").glob("*.json"))
    if change == "delete":
        assert "s1" not in _rows(q)
        return
    if change == "rebind":
        assert _rows(q)["s1"]["hold"]["reason"] == "project_routing_fence_changed"
    else:
        assert "occurrence" not in _rows(q)["s1"]  # disabled/edited claims are dropped
    monkeypatch.setattr(occurrences, "prepare", real_prepare)
    q.queue.check_scheduled_tasks()
    if change == "disable":
        assert q.pending == []
    else:
        [task] = q.pending
        assert ("edited" in task["text"]) if change == "edit" else task["workspace_root"] == str(moved.resolve())


def test_each_durable_boundary_failure_reconciles_to_the_same_occurrence(q, monkeypatch):
    import ouroboros.task_results as task_results
    from supervisor import queue_schedules

    _row(q, intent={"kind": "system_repo"})
    # The receipt write fails: nothing stays queued, the row waits on the SAME claim.

    real_result_write = task_results.write_task_result
    monkeypatch.setattr(task_results, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert q.pending == [] and row["hold"]["reason"] == "receipt_failed" and row["occurrence"]["phase"] == "claimed"
    task_id = row["occurrence"]["task_id"]
    monkeypatch.setattr(task_results, "write_task_result", real_result_write)
    # The row write after admission fails: the task is withdrawn; its receipt stands.
    real_table_write = queue_schedules._write_scheduled_tasks
    calls = {"n": 0}

    def flaky_table(data, drive_root=None):
        calls["n"] += 1
        if calls["n"] == 2:  # the admission commit (the claim pass write succeeded)
            raise OSError("table")
        return real_table_write(data, drive_root)

    monkeypatch.setattr(queue_schedules, "_write_scheduled_tasks", flaky_table)
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and load_task_result(q.root, task_id)["schedule_admission"]["dispatch"] == "none"
    monkeypatch.setattr(queue_schedules, "_write_scheduled_tasks", real_table_write)
    # The snapshot reports False: withdrawn again, the row stays admitted.
    monkeypatch.setattr(q.queue, "persist_queue_snapshot", lambda reason="": False)
    q.queue.check_scheduled_tasks()
    assert q.pending == []
    row = _rows(q)["s1"]
    assert row["occurrence"]["task_id"] == task_id and row["occurrence"]["phase"] == "admitted"
    # A later edit is future-only: the admitted occurrence republishes its FROZEN task.
    q.queue.upsert_scheduled_task({**row, "task": {**row["task"], "text": "edited later"}})
    monkeypatch.setattr(q.queue, "persist_queue_snapshot", lambda reason="": True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["id"] == task_id and "edited later" not in task["text"]


def test_dispatch_barrier_restore_and_settlement(q, monkeypatch):
    import datetime

    from supervisor import queue_schedules, schedule_time
    from supervisor import schedule_occurrence as occurrences

    class TickTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 27, 19, 2, 30, tzinfo=datetime.timezone.utc).astimezone(tz)

    # These two ticks settle ONE occurrence, not a newly due cron minute. Freeze the
    # scheduler's own ``datetime`` names, never the stdlib class: dateutil binds
    # ``datetime`` at import, so a first import under the fake class keeps it for the
    # rest of the worker and every later cron check fails its fromutc() type check.
    clock = SimpleNamespace(datetime=TickTime, timezone=datetime.timezone, timedelta=datetime.timedelta)
    for module in (queue_schedules, occurrences, schedule_time):
        monkeypatch.setattr(module, "datetime", clock)

    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert occurrences.restore_allowed(task) is True  # accepted, never dispatched: may be revived
    stored = load_task_result(q.root, task["id"])
    receipt = stored["schedule_admission"]
    for frozen in ({"id": "another-task"}, None):
        write_task_result(q.root, task["id"], "scheduled", schedule_admission={**receipt, "task": frozen})
        assert occurrences.restore_allowed(task) is False
        assert occurrences.record_dispatch_possible(task) is False
    write_task_result(q.root, task["id"], "scheduled", schedule_admission=receipt)
    assert occurrences.record_dispatch_possible(task) is True
    receipt = load_task_result(q.root, task["id"])
    assert receipt["status"] == "running" and receipt["schedule_admission"]["dispatch"] == "possible"
    assert occurrences.restore_allowed(task) is False  # a stale pending row never replays it
    forged = {**task, "metadata": {**task["metadata"], "schedule_occurrence": {"schedule_id": "s1", "token": "x"}}}
    assert occurrences.record_dispatch_possible(forged) is False  # another token: not dispatched
    # The row reconciles once the task is no longer in flight: the occurrence settles,
    # nothing is re-run, and the one overdue cron point moved to the NEXT FUTURE instant.
    q.pending.clear()
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert "occurrence" not in row or row["occurrence"]["task_id"] != task["id"]
    assert row["last_task_id"] == task["id"] and row["next_run_at"] > "2026"
    (q.root / "task_results" / f"{task['id']}.json").write_text("{", encoding="utf-8")
    assert occurrences.restore_allowed(task) is False  # unreadable is unknown, never revived


def test_running_cron_task_blocks_a_second_due_occurrence(q):
    """Retire the dispatched token while live, but never claim the next point
    until that scheduled root leaves the queue."""
    from supervisor import schedule_occurrence as occurrences

    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert occurrences.record_dispatch_possible(task) is True
    q.pending.clear()
    q.queue.RUNNING[task["id"]] = {"task": task}
    try:
        record = _rows(q)["s1"]
        record["next_run_at"] = "2000-01-02T00:00:00+00:00"
        from supervisor import queue_schedules

        store = q.queue.load_schedule_store(q.root)
        store["tasks"][0] = record
        queue_schedules._write_scheduled_tasks(store, q.root)
        q.queue.check_scheduled_tasks()
        assert q.pending == []
        settled = _rows(q)["s1"]
        assert "occurrence" not in settled
        assert settled["last_task_id"] == task["id"]
        assert settled["next_run_at"] == "2000-01-02T00:00:00+00:00"
    finally:
        q.queue.RUNNING.pop(task["id"], None)


def test_recreated_cron_row_without_last_task_id_still_waits_for_live_root(q):
    """A skill can recreate a due row while its earlier task is still live.

    The new row has no history fields, but the running root retains schedule_id.
    It must not admit a second root merely because last_task_id disappeared.
    """
    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    from supervisor import schedule_occurrence as occurrences

    assert occurrences.record_dispatch_possible(task) is True
    q.pending.clear()
    q.queue.RUNNING[task["id"]] = {"task": task}
    try:
        from supervisor import queue_schedules

        removed = q.queue.mutate_scheduled_task(
            "delete", "s1", reason="owner removed reminder", actor="owner:gateway", drive_root=q.root,
        )
        assert removed["changed"] is True
        assert "s1" not in _rows(q)
        _row(q, intent={"kind": "system_repo"}, cron=True)
        assert not _rows(q)["s1"].get("last_task_id")
        assert queue_schedules._schedule_running_or_queued("s1", q.root) is True
        q.queue.check_scheduled_tasks()
        assert q.pending == []
        assert "occurrence" not in _rows(q)["s1"]
        assert _rows(q)["s1"]["next_run_at"] == "2000-01-01T00:00:00+00:00"
    finally:
        q.queue.RUNNING.pop(task["id"], None)


def test_an_unreadable_or_missing_receipt_holds_instead_of_replaying(q):
    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    q.pending.clear()
    receipt = q.root / "task_results" / f"{task['id']}.json"
    receipt.write_text("{", encoding="utf-8")
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and _rows(q)["s1"]["hold"]["reason"] == "occurrence_result_unreadable"
    receipt.unlink()
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and _rows(q)["s1"]["hold"]["reason"] == "occurrence_evidence_missing"


def test_a_folderless_project_task_defaults_to_its_own_scratch_without_workspace_authority(tmp_path):
    from ouroboros.tool_access import folderless_scratch_dir, resource_root_path
    from ouroboros.tools.control_scheduling import _inherited_workspace_from_active_repo
    from ouroboros.tools.registry import ToolContext

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()

    def ctx(task_id, **meta):
        c = ToolContext(repo_dir=repo, drive_root=data, task_id=task_id, project_id=meta.pop("project_id", ""))
        c.task_metadata = meta
        return c

    parent = ctx("parent", project_id="proj", resource_intent={"kind": "room_default", "project_id": "proj"})
    scratch = parent.active_repo_dir()
    assert scratch == resource_root_path(parent, "task_drive") and scratch.is_dir()
    assert _inherited_workspace_from_active_repo(parent, "", "") == ("", "")  # never an inherited workspace
    child = ctx("child", project_id="proj")
    assert child.active_repo_dir() == resource_root_path(child, "task_drive") != scratch  # its OWN scratch
    for main in (ctx("main"), ctx("self", project_id="proj", resource_intent={"kind": "system_repo"}),
                 ctx("room", project_id="proj", _project_room_dir=str(repo))):
        assert folderless_scratch_dir(main) is None
    # The parent's inputs stay readable to the child through the existing lineage rule.
    (scratch / "input.txt").write_text("from the parent", encoding="utf-8")
    from ouroboros.tool_access import build_resolved_resource_binding

    child.task_metadata = {"parent_task_id": "parent", "root_task_id": "parent"}
    binding = build_resolved_resource_binding(child, root="task_drive", operation="read",
                                              path=str(scratch / "input.txt"))
    assert binding.target_path.read_text(encoding="utf-8") == "from the parent"


def test_a_folderless_delegated_run_reads_but_never_writes_through_scratch(tmp_path):
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate_integration import _mutation_authority
    from ouroboros.tools.registry import ToolContext

    (tmp_path / "repo").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="t", project_id="proj")
    ctx.task_metadata = {"resource_intent": {"kind": "explicit_none", "project_id": "proj"}}
    readonly, refusal = _mutation_authority(ctx, delegated_run_shape(False, "readonly"))
    assert refusal is None and readonly["capture_mode"] == "none"
    assert pathlib.Path(readonly["target_root"]).parts[-2:] == ("task_drives", "t")  # native separators
    _, refused = _mutation_authority(ctx, delegated_run_shape(True, "workspace_write"))
    assert refused is not None and "workspace_not_active" in refused.text


def test_receipt_and_dispatch_mark_are_monotonic_and_read_back_from_disk(q, monkeypatch):
    import ouroboros.task_results as task_results
    from supervisor import schedule_occurrence as occurrences

    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    occ = _rows(q)["s1"]["occurrence"]
    item = {"schedule_id": "s1", "task": task, "occurrence": occ}
    assert occurrences.record_dispatch_possible(task) is True
    # A late/duplicate receipt never downgrades a possibly-dispatched occurrence.
    assert occurrences._write_receipt(item, _rows(q)["s1"]) is False
    stored = load_task_result(q.root, task["id"])
    assert stored["status"] == "running" and stored["schedule_admission"]["dispatch"] == "possible"
    # A terminal occurrence is never dispatched again through the barrier.
    task_results.write_task_result(q.root, task["id"], "completed", result="done")
    assert occurrences.record_dispatch_possible(task) is False
    # The verdict is what the FILE says, not what a writer returned.
    _row(q, "s2", intent={"kind": "system_repo"}, cron=True)
    monkeypatch.setattr(task_results, "write_task_result", lambda *_a, **k: {
        "status": "scheduled", "schedule_admission": dict(k.get("schedule_admission") or {})})
    q.pending.clear()
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and _rows(q)["s2"]["hold"]["reason"] == "receipt_failed"


def test_the_allowance_is_read_just_before_admission_not_during_prepare(q, monkeypatch):
    from ouroboros import consciousness_allowance
    from supervisor import schedule_occurrence as occurrences

    window = {"status": "available", "limit_usd": 10.0, "settled_usd": 0.0, "accounted_usd": 0.0, "unknown_unmetered": 0, "resets_at": ""}
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _root: dict(window))
    _row(q, intent={"kind": "system_repo"}, metadata={"initiator": "consciousness"})
    real_prepare = occurrences.prepare

    def prepare_then_spend(claimed):
        prepared = real_prepare(claimed)
        window.update(status="exhausted", settled_usd=10.0, accounted_usd=10.0)  # spent while the prepare ran
        return prepared

    monkeypatch.setattr(occurrences, "prepare", prepare_then_spend)
    q.queue.check_scheduled_tasks()
    assert q.pending == [] and _rows(q)["s1"]["hold"]["reason"] == "consciousness_allowance_exhausted"


def test_deleting_a_row_never_takes_back_an_accepted_occurrence(q, monkeypatch):
    from supervisor import schedule_occurrence as occurrences

    _row(q, intent={"kind": "system_repo"}, cron=True)
    monkeypatch.setattr(q.queue, "persist_queue_snapshot", lambda reason="": False)
    q.queue.check_scheduled_tasks()  # accepted, then withdrawn: the snapshot did not persist
    assert q.pending == []
    task_id = _rows(q)["s1"]["occurrence"]["task_id"]
    outcome = q.queue.mutate_scheduled_task("delete", "s1", reason="owner", actor="owner")
    row = _rows(q)["s1"]
    assert outcome["status"] == "delete_deferred" and "owed" in outcome["detail"]
    assert row["enabled"] is False and row["delete_requested_at"]
    monkeypatch.setattr(q.queue, "persist_queue_snapshot", lambda reason="": True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["id"] == task_id  # the SAME accepted occurrence still runs
    assert occurrences.record_dispatch_possible(task) is True
    q.pending.clear()
    q.queue.check_scheduled_tasks()
    assert "s1" not in _rows(q)  # removed once its run was dispatched
    # A row whose occurrence was never accepted is deleted at once (future-only).
    _row(q, "s3", intent={"kind": "system_repo"}, cron=True)
    assert q.queue.mutate_scheduled_task("delete", "s3", reason="owner", actor="owner")["status"] == "deleted"
    assert "s3" not in _rows(q)


def test_a_folderless_direct_parent_dispatches_a_readonly_child_with_its_own_scratch(tmp_path, monkeypatch):
    """The ACTUAL dispatch shape: the real tool emits the event, the supervisor's own
    admission builds the task, and the child's context resolves from that task."""
    from types import SimpleNamespace

    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tool_access import build_resolved_resource_binding, resource_root_path
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext
    from supervisor.events_subagent_admission import _resolve_subagent_constraint
    from supervisor.task_dispatch import build_scheduled_task_payload

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    import json

    from devtools.benchmarks.common.model_slots import single_model_subagents_setting

    subagents = json.loads(single_model_subagents_setting("openai/test-actor"))
    monkeypatch.setattr(control, "load_settings", lambda: {"OUROBOROS_SUBAGENTS": subagents})
    parent = ToolContext(repo_dir=repo, drive_root=data, task_id="parent")
    parent.task_metadata = {"resource_intent": {"kind": "explicit_none"}}
    parent.pending_events = []
    scratch = parent.active_repo_dir()
    (scratch / "notes.txt").write_text("parent input", encoding="utf-8")
    control._schedule_task(parent, subagent_id=subagents["items"][0]["subagent_id"],
                           objective="Research the question in notes.txt.", expected_output="An answer.")
    [event] = [e for e in parent.pending_events if e.get("type") == "schedule_subagent"]
    assert not event.get("workspace_root") and not event.get("workspace_mode")  # scratch is never a workspace
    constraint, workspace, mode, refusal = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=repo, DRIVE_ROOT=data), tid=event["task_id"],
        requested_constraint=event["task_constraint"], workspace_root="", workspace_mode="",
        base_sha=event.get("base_sha", ""), parent_task_id="parent")
    assert not refusal and not workspace and not mode
    task = build_scheduled_task_payload({**event, "tid": event["task_id"], "desc": event["objective"],
                                        "text": event["objective"], "task_constraint": constraint,
                                        "workspace_root": workspace, "workspace_mode": mode, "parent_id": "parent"})
    assert not task.get("workspace_root") and str(data) not in str(task.get("workspace_root") or "")
    child = ToolContext(repo_dir=repo, drive_root=data, task_id=task["id"], project_id=task.get("project_id", ""))
    child.task_metadata = {**task["metadata"], "parent_task_id": "parent", "root_task_id": "parent"}
    child.task_constraint = normalize_task_constraint(task.get("task_constraint"))
    own = child.active_repo_dir()
    assert own == resource_root_path(child, "task_drive") != scratch  # its OWN scratch as default cwd
    binding = build_resolved_resource_binding(child, root="task_drive", operation="read",
                                              path=str(scratch / "notes.txt"))
    assert binding.target_path.read_text(encoding="utf-8") == "parent input"  # lineage input, read-only
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(child)
    assert "parent input" in registry.execute("read_file", {"root": "task_drive", "path": str(scratch / "notes.txt")})
    sibling = data / "task_drives" / "sibling"
    sibling.mkdir(parents=True)
    (sibling / "notes.txt").write_text("sibling-private-input", encoding="utf-8")
    assert "sibling-private-input" not in registry.execute("read_file", {"root": "task_drive", "path": str(sibling / "notes.txt")})
