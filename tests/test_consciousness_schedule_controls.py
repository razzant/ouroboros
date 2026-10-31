"""Owner schedule controls: one audited seam, one lifecycle truth.

``supervisor/queue_schedules.py`` owns ``state/scheduled_tasks.json``. Every
read-modify-write — the supervisor tick, the gateway, the
follow-up tool and ``manage_schedules`` — takes the queue lock and then the
table's own sidecar lock, so the file has exactly one writer at a time. A
mutation announces its INTENT to the existing events log before it changes
anything and records its OUTCOME after; an unreadable table refuses the write
instead of replacing real rows with an empty document. Disable/delete govern
FUTURE dispatch only: admitted work retains its existing occurrence receipts.
A fired one-shot stays history until a fresh instant is authored.
"""

from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

from ouroboros import utils as ouro_utils
from supervisor import queue_schedules, schedule_lifecycle


class _Queue:
    DRIVE_ROOT = pathlib.Path(".")
    SCHEDULED_TASKS_FILE = pathlib.Path("state") / "scheduled_tasks.json"
    _queue_lock = threading.RLock()
    PENDING: list = []
    RUNNING: dict = {}


def _bind(monkeypatch, root):
    _Queue.DRIVE_ROOT = root
    _Queue.PENDING, _Queue.RUNNING = [], {}
    monkeypatch.setattr(queue_schedules, "_queue", lambda: _Queue)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


def _write(root, tasks):
    (root / "state" / "scheduled_tasks.json").write_text(
        json.dumps({"schema_version": 1, "tasks": tasks}), encoding="utf-8"
    )


def _rows(root):
    return queue_schedules.list_scheduled_tasks(root)["tasks"]


def _events(root):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _skill(name="demo", tasks=(("daily", "0 * * * *"),), permissions=("supervised_task",), content_hash="hash"):
    return SimpleNamespace(
        name=name, identity_collision=False, content_hash=content_hash,
        manifest=SimpleNamespace(
            permissions=list(permissions),
            scheduled_tasks=[{"name": n, "cron": c, "timezone": "UTC"} for n, c in tasks],
        ),
    )


def _ready(monkeypatch, ready=True):
    monkeypatch.setattr("ouroboros.skill_readiness.skill_readiness_for_execution",
                        lambda *_a, **_k: SimpleNamespace(ready=ready))


def _skill_row(schedule_id="skill-demo-daily", **extra):
    return {"id": schedule_id, "name": "demo/daily", "enabled": True, "source": "skill_manifest",
            "skill": "demo", "trigger": {"type": "cron", "expr": "0 * * * *"}, "timezone": "UTC",
            "task": {"type": "task", "text": "run"}, "next_run_at": "2026-09-22T00:00:00+00:00",
            **extra}


# --- lifecycle controls -----------------------------------------------------------


# --- the consumed one-shot ---------------------------------------------------------


def _once_row(**extra):
    return {"id": "once-1", "name": "once", "enabled": False,
            "completed_at": "2026-09-21T00:00:00+00:00",
            "trigger": {"type": "once", "run_at": "2026-09-20T00:00:00+00:00"},
            "task": {"type": "task", "text": "run"}, **extra}


def test_consumed_once_is_history_and_restore_does_not_rearm(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_once_row()])
    result = schedule_lifecycle.mutate_scheduled_task("restore", "once-1", reason="try restore",
                                                   actor="agent", drive_root=tmp_path)
    assert result["ok"] is False and result["status"] == "consumed_not_rearmed"
    assert result["changed"] is False and _rows(tmp_path)[0]["enabled"] is False
    projected = queue_schedules.schedule_activity_projection(
        queue_schedules.list_scheduled_tasks(tmp_path))
    row = projected["tasks"][0]
    assert row["status"] == "consumed" and row["active"] is False
    assert row["retained"] is True and row["restorable"] is False
    # Even a refused action is auditable: the intent fact names what was attempted.
    assert _events(tmp_path)[-1]["result"] == "consumed_not_rearmed"


def test_a_timezone_only_edit_does_not_resurrect_a_fired_one_shot(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_once_row(timezone="UTC")])
    kept = schedule_lifecycle.upsert_scheduled_task({
        "id": "once-1", "name": "once", "enabled": False, "timezone": "Europe/Moscow",
        "trigger": {"type": "once", "run_at": "2026-09-20T00:00:00+00:00"},
        "task": {"type": "task", "text": "run"},
    }, drive_root=tmp_path)
    assert kept["completed_at"] == "2026-09-21T00:00:00+00:00" and kept["enabled"] is False
    assert kept["timezone"] == "Europe/Moscow"
    # The same edit with enabled=true is the re-arm this rule exists to refuse.
    with pytest.raises(queue_schedules.ScheduleRefused) as refusal:
        schedule_lifecycle.upsert_scheduled_task({
            "id": "once-1", "name": "once", "enabled": True, "timezone": "Europe/Moscow",
            "trigger": {"type": "once", "run_at": "2026-09-20T00:00:00+00:00"},
            "task": {"type": "task", "text": "run"},
        }, drive_root=tmp_path)
    assert refusal.value.status == "consumed_not_rearmed"
    assert _rows(tmp_path)[0]["completed_at"] == "2026-09-21T00:00:00+00:00"
    # The announced intent gets its outcome even when the answer is "no".
    intent, outcome = _events(tmp_path)[-2:]
    assert intent["phase"] == "intent" and outcome["phase"] == "outcome"
    assert intent["operation_id"] == outcome["operation_id"]
    assert outcome["result"] == "consumed_not_rearmed" and "after" not in outcome


def test_a_new_run_at_is_a_new_one_shot_and_clears_the_receipt(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_once_row()])
    rearmed = schedule_lifecycle.upsert_scheduled_task({
        "id": "once-1", "name": "once", "enabled": True,
        "trigger": {"type": "once", "run_at": "2026-10-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "run"},
    }, drive_root=tmp_path)
    assert "completed_at" not in rearmed and rearmed["enabled"] is True


# --- one writer, one transaction ---------------------------------------------------


def test_the_transaction_is_reentrant_per_table_not_per_depth(tmp_path, monkeypatch):
    """A nested transaction on ANOTHER root still takes that root's own lock."""
    other = tmp_path / "other"
    _bind(monkeypatch, tmp_path)
    (other / "state").mkdir(parents=True)
    taken: list[str] = []
    real = queue_schedules.acquire_exclusive_file_lock

    def _record(path, **kwargs):
        taken.append(pathlib.Path(path).parent.parent.name)
        return real(path, **kwargs)

    monkeypatch.setattr(queue_schedules, "acquire_exclusive_file_lock", _record)
    with queue_schedules.schedule_transaction(tmp_path):
        with queue_schedules.schedule_transaction(tmp_path):  # same table: rides the hold
            assert taken == [tmp_path.name]
        with queue_schedules.schedule_transaction(other):     # another table: its own lock
            assert (other / "state" / "scheduled_tasks.json.lock").exists()
    assert taken == [tmp_path.name, "other"]


def test_schedule_lock_key_uses_platform_case_normalization(monkeypatch):
    monkeypatch.setattr(queue_schedules.os.path, "normcase", lambda value: value.lower())
    assert queue_schedules._schedule_lock_key(
        pathlib.Path("/tmp/State/Scheduled.lock")) == queue_schedules._schedule_lock_key(
            pathlib.Path("/tmp/state/scheduled.lock"))


def test_the_transaction_itself_takes_the_queue_lock_before_the_table_lock(tmp_path, monkeypatch):
    """One owner of the ORDER, so no caller can compose the pair the other way.

    ``schedule_followup`` wraps its pending read and its write in one transaction; if
    the queue lock were taken inside that hold (by ``upsert_scheduled_task``, as
    it used to be) it would run table-then-queue against the tick's
    queue-then-table. The order lives in the transaction so there is nothing per
    caller left to audit.
    """
    _bind(monkeypatch, tmp_path)
    order: list[str] = []
    real_file_lock = queue_schedules.acquire_exclusive_file_lock
    monkeypatch.setattr(
        queue_schedules, "acquire_exclusive_file_lock",
        lambda path, **kw: (order.append("table"), real_file_lock(path, **kw))[1])
    real_queue_lock = _Queue._queue_lock

    class _RecordingLock:
        def __enter__(self):
            order.append("queue")
            return real_queue_lock.__enter__()

        def __exit__(self, *exc_info):
            return real_queue_lock.__exit__(*exc_info)

    _Queue._queue_lock = _RecordingLock()
    try:
        schedule_lifecycle.upsert_scheduled_task(
            {"id": "cron-1", "name": "cron", "trigger": {"type": "cron", "expr": "0 * * * *"},
             "task": {"type": "task", "text": "run"}}, drive_root=tmp_path)
        assert order == ["queue", "table"]
        order.clear()
        schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="pause",
                                              actor="agent", drive_root=tmp_path)
        assert order == ["queue", "table"]
        order.clear()
        # The follow-up shape: an OUTER transaction around a read and a write.
        with queue_schedules.schedule_transaction(tmp_path):
            queue_schedules.list_scheduled_tasks(tmp_path)
            schedule_lifecycle.upsert_scheduled_task(
                {"id": "cron-2", "name": "cron", "trigger": {"type": "cron", "expr": "0 * * * *"},
                 "task": {"type": "task", "text": "run"}}, drive_root=tmp_path)
        assert order == ["queue", "table"]
    finally:
        _Queue._queue_lock = real_queue_lock


def test_a_transaction_wrapped_write_and_a_queue_lock_holder_do_not_deadlock(tmp_path, monkeypatch):
    """The two real orders, run against each other, with every row surviving.

    Thread A is the follow-up tool: one transaction around its pending read and its
    write. Thread B is a supervisor pass: the queue lock, then the table. While
    the transaction left the queue lock to its callers these two could wedge, and
    this test would HANG rather than fail — so the join has a hard timeout.
    """
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [])
    rounds = 15
    start = threading.Barrier(2, timeout=10)
    errors: list[BaseException] = []

    def _row(schedule_id):
        return {"id": schedule_id, "name": schedule_id, "timezone": "UTC",
                "trigger": {"type": "cron", "expr": "0 * * * *"},
                "task": {"type": "task", "text": "run"}}

    def _tool():
        try:
            start.wait()
            for index in range(rounds):
                with queue_schedules.schedule_transaction(tmp_path):
                    queue_schedules.list_scheduled_tasks(tmp_path)
                    schedule_lifecycle.upsert_scheduled_task(_row(f"tool-{index}"), drive_root=tmp_path)
        except BaseException as exc:  # noqa: BLE001 -- reported to the main thread
            errors.append(exc)

    def _tick():
        try:
            start.wait()
            for index in range(rounds):
                with _Queue._queue_lock:
                    schedule_lifecycle.upsert_scheduled_task(_row(f"tick-{index}"), drive_root=tmp_path)
        except BaseException as exc:  # noqa: BLE001 -- reported to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=_tool, daemon=True),
               threading.Thread(target=_tick, daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not [t for t in threads if t.is_alive()], "the schedule transaction deadlocked"
    assert not errors, errors
    # Serialized, not merely unblocked: no interleaved read-modify-write lost a row.
    assert {row["id"] for row in _rows(tmp_path)} == (
        {f"tool-{i}" for i in range(rounds)} | {f"tick-{i}" for i in range(rounds)})


def test_every_writer_reaches_the_same_transaction_through_the_queue_facade(tmp_path, monkeypatch):
    """One lock, one owner: the tool, the gateway and the follow-up share it."""
    from supervisor import queue as queue_module

    assert queue_module.schedule_transaction is queue_schedules.schedule_transaction
    assert queue_module.SCHEDULE_ACTIONS == queue_schedules.SCHEDULE_ACTIONS
    assert queue_module.ScheduleStoreUnreadable is queue_schedules.ScheduleStoreUnreadable


# --- an unreadable table refuses, it does not erase --------------------------------


@pytest.mark.parametrize("contents", ["{ not json", '["a list, not a table"]', '{"tasks": 7}'])
def test_a_corrupt_table_refuses_every_mutation_instead_of_erasing_it(tmp_path, monkeypatch, contents):
    _bind(monkeypatch, tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    path.write_text(contents, encoding="utf-8")
    refused = schedule_lifecycle.mutate_scheduled_task("disable", "any", reason="x", actor="agent",
                                                    drive_root=tmp_path)
    assert refused["ok"] is False and refused["status"] == "store_unreadable"
    assert refused["changed"] is False and refused["audit"] == "not_written"
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
        schedule_lifecycle.upsert_scheduled_task({"id": "new", "trigger": {"type": "cron", "expr": "0 * * * *"}},
                                              drive_root=tmp_path)
    queue_schedules.check_scheduled_tasks()  # the tick skips the pass rather than rewriting
    assert path.read_text(encoding="utf-8") == contents
    # The agent's own read says unavailable rather than reporting an empty table.
    # The handler answers TEXT (the registered ToolEntry ABI); its typed status
    # rides the published sidecar, which is pinned in the observe-dispatch suite.
    from ouroboros.tools.followup import _manage_schedules

    ctx = SimpleNamespace(task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path, task_id="t")
    unavailable = _manage_schedules(ctx, action="list")
    assert isinstance(unavailable, str)
    assert "CAPABILITY_UNAVAILABLE" in unavailable and "readable" in unavailable


def test_invalid_utf8_schedule_bytes_are_unreadable_not_a_decode_crash(tmp_path, monkeypatch):
    """A present table with invalid UTF-8 is an unavailable store, not a leaked decode error."""
    _bind(monkeypatch, tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    path.write_bytes(b'{"tasks": [\xff]}')
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as refusal:
        queue_schedules.load_schedule_store(tmp_path)
    assert "readable schedule table" in str(refusal.value)
    outcome = schedule_lifecycle.mutate_scheduled_task(
        "disable", "any", reason="pause", actor="agent", drive_root=tmp_path)
    assert outcome["status"] == "store_unreadable" and outcome["changed"] is False
    assert path.read_bytes() == b'{"tasks": [\xff]}'


def test_manage_schedule_list_is_bounded_and_explicitly_paginated(tmp_path, monkeypatch):
    """The model-facing list omits templates and exposes a bounded page contract."""
    _bind(monkeypatch, tmp_path)
    objective = "objective " + ("x" * 1_000)
    tasks = [{
        "id": f"schedule-{index}", "name": f"Schedule {index}", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 * * * *"},
        "task": {"text": objective, "context": "secret context " * 10_000,
                 "attachments": [{"path": "private"}]},
    } for index in range(25)]
    _write(tmp_path, tasks)
    from ouroboros.tools.followup import _manage_schedules

    ctx = SimpleNamespace(task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path, task_id="t")
    first_text = _manage_schedules(ctx, action="list", offset=0, limit=20)
    first = json.loads(first_text)
    assert first["total"] == 25 and first["offset"] == 0
    assert first["limit"] == 20
    assert 0 < len(first["tasks"]) <= first["limit"]
    assert first["next_offset"] == len(first["tasks"])
    assert len(first_text) < 15_000
    row = first["tasks"][0]
    assert "task" not in row and "context" not in row and "attachments" not in row
    assert len(row["objective_preview"]) <= 240
    assert row["objective_truncated"] is True and row["objective_preview_truncated"] is True

    second = json.loads(_manage_schedules(ctx, action="list", offset=first["next_offset"], limit=20))
    assert second["total"] == 25 and second["offset"] == first["next_offset"]
    assert second["next_offset"] is None
    assert len(second["tasks"]) == 25 - first["next_offset"]


def test_manage_schedule_projection_preserves_identity_selectors(tmp_path, monkeypatch):
    """Bounded display fields may shorten; ids used by a later action may not."""
    _bind(monkeypatch, tmp_path)
    schedule_id = "schedule-" + ("x" * 72)
    task_id = "task-" + ("y" * 72)
    projected = queue_schedules.schedule_tool_projection({"tasks": [{
        "id": schedule_id, "name": "long", "enabled": True,
        "last_task_id": task_id, "skill": "skill-" + ("z" * 72),
        "task": {"text": "objective " + ("x" * 500)},
    }]})
    row = projected["tasks"][0]
    assert row["id"] == schedule_id
    assert row["last_task_id"] == task_id
    assert row["skill"] == "skill-" + ("z" * 72)


def _root_ctx(tmp_path):
    """A real root turn: schedule mutation authority is resolved, not asserted."""
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="turn-1", task_metadata={})


def test_a_row_too_large_to_render_still_reports_the_change_it_already_made(tmp_path, monkeypatch):
    """A receipt this process cannot fit is a missing detail, not an action that did not run.

    Identity fields are deliberately unbounded so a later action addresses the
    same record, which means one manually edited row CAN exceed the tool result
    limit. By then the table is already rewritten and both audit facts are
    written: answering ``CAPABILITY_UNAVAILABLE`` would tell the model nothing
    happened while the owner's table says otherwise.
    """
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{
        "id": "nightly", "name": "nightly", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 * * * *"},
        # A real field, read back from a prior run, that nothing bounds.
        "last_task_id": "task-" + ("y" * 20_000),
    }])
    from ouroboros.tools.followup import _manage_schedules

    text = _manage_schedules(_root_ctx(tmp_path), action="disable",
                             schedule_id="nightly", reason="owner pause")
    assert "CAPABILITY_UNAVAILABLE" not in text
    receipt = json.loads(text)
    assert receipt["changed"] is True and receipt["ok"] is True
    assert receipt["status"] == "updated" and receipt["audit"] == "recorded"
    assert receipt["schedule_id"] == "nightly"
    assert receipt["schedule_omitted"] is True
    assert "projection_too_large" in receipt["schedule_omitted_reason"]
    assert "schedule" not in receipt
    # The durable change and its audit pair are real, not implied by the receipt.
    assert _rows(tmp_path)[0]["enabled"] is False
    phases = [event["phase"] for event in _events(tmp_path) if event["type"] == "schedule_mutation"]
    assert phases == ["intent", "outcome"]


def test_an_identity_too_large_for_a_truthful_receipt_changes_nothing(tmp_path, monkeypatch):
    """Refuse BEFORE the write when the selector itself cannot come back intact."""
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row()])
    from ouroboros.tools.followup import _manage_schedules

    receipt = json.loads(_manage_schedules(
        _root_ctx(tmp_path), action="disable",
        schedule_id="s" * 14_000, reason="owner pause"))
    assert receipt["status"] == "identity_too_large"
    assert receipt["changed"] is False and receipt["audit"] == "not_written"
    assert _rows(tmp_path)[0]["enabled"] is True
    assert _events(tmp_path) == []


def test_a_list_page_whose_first_row_cannot_fit_refuses_instead_of_reporting_no_schedules(
        tmp_path, monkeypatch):
    """An unrenderable row must not read as a table that has nothing in it."""
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [
        {"id": "huge", "name": "huge", "enabled": True,
         "last_task_id": "task-" + ("y" * 20_000)},
        {"id": "ordinary", "name": "ordinary", "enabled": True},
    ])
    from ouroboros.tools.followup import _manage_schedules

    refused = _manage_schedules(_root_ctx(tmp_path), action="list", offset=0, limit=20)
    assert "CAPABILITY_UNAVAILABLE" in refused and "projection_too_large" in refused
    assert '"total":0' not in refused and "no schedules" not in refused.lower()
    # The refusal names where to resume, so the rest of the table stays reachable.
    assert "offset 1" in refused
    following = json.loads(_manage_schedules(_root_ctx(tmp_path), action="list", offset=1, limit=20))
    assert [row["id"] for row in following["tasks"]] == ["ordinary"]


def test_an_oversized_action_receipt_stays_valid_json_and_keeps_its_selectors(
        tmp_path, monkeypatch):
    """The compact receipt is a whole document; the outer cap must never cut it."""
    _bind(monkeypatch, tmp_path)
    schedule_id = "s" * 12_000
    _write(tmp_path, [{"id": schedule_id, "name": "long id", "enabled": True,
                       "trigger": {"type": "cron", "expr": "0 * * * *"}}])
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.followup import _manage_schedules

    text = _manage_schedules(_root_ctx(tmp_path), action="disable",
                             schedule_id=schedule_id, reason="owner pause")
    assert len(text) <= tool_result_limit("manage_schedules")
    receipt = json.loads(text)  # a truncated document would not parse
    assert receipt["schedule_id"] == schedule_id
    assert receipt["changed"] is True and receipt["schedule_omitted"] is True
    assert _rows(tmp_path)[0]["enabled"] is False


def test_a_present_but_unusable_path_is_unreadable_rather_than_an_empty_table(tmp_path, monkeypatch):
    """Only an ABSENT path is an empty store; anything else present is unknown.

    ``is_file()`` alone answered False for a directory and for a dangling symlink
    just as it does for nothing at all, so a write would have treated real state
    (or a link out of the drive) as a table it may replace.
    """
    _bind(monkeypatch, tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    assert queue_schedules.load_schedule_store(tmp_path) == {"schema_version": 1, "tasks": []}

    path.mkdir()
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as directory:
        queue_schedules.load_schedule_store(tmp_path)
    assert "regular file" in str(directory.value)
    path.rmdir()

    path.symlink_to(tmp_path / "state" / "nothing-here.json")
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as dangling:
        queue_schedules.load_schedule_store(tmp_path)
    assert "regular file" in str(dangling.value)
    refused = schedule_lifecycle.mutate_scheduled_task("disable", "any", reason="x", actor="agent",
                                                    drive_root=tmp_path)
    assert refused["status"] == "store_unreadable" and refused["audit"] == "not_written"
    assert path.is_symlink() and not path.exists()  # nothing followed the link


def test_a_table_it_cannot_open_is_named_apart_from_one_it_cannot_parse(tmp_path, monkeypatch):
    """Two different owner problems, so two different messages.

    The lenient reader answers None for "could not open" and "could not parse"
    alike; a permission/IO fault is a host problem and corrupt bytes are a file
    problem, and the refusal has to say which one it met.
    """
    _bind(monkeypatch, tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    path.write_text('{ not json', encoding="utf-8")
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as unparseable:
        queue_schedules.load_schedule_store(tmp_path)
    assert "is not a readable schedule table (" in str(unparseable.value)

    path.write_text(json.dumps({"schema_version": 1, "tasks": []}), encoding="utf-8")
    path.chmod(0o000)
    try:
        readable = True
        try:
            path.read_text(encoding="utf-8")
        except OSError:
            readable = False
        if readable:
            pytest.skip("this process can read a mode-000 file (root); the branch needs a denied read")
        with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as unopenable:
            queue_schedules.load_schedule_store(tmp_path)
        assert "could not be read (" in str(unopenable.value)
        refused = schedule_lifecycle.mutate_scheduled_task("disable", "any", reason="x", actor="agent",
                                                        drive_root=tmp_path)
        assert refused["status"] == "store_unreadable" and refused["audit"] == "not_written"
    finally:
        path.chmod(0o600)


def test_a_malformed_row_is_refused_not_dropped_by_a_write_about_another_row(tmp_path, monkeypatch):
    """A mutation may not quietly delete what it could not parse.

    The writers used to filter non-object rows out of the list and then write the
    filtered list back, so one disable could erase a neighbouring row and the
    only record of it was gone.
    """
    _bind(monkeypatch, tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    contents = json.dumps({"schema_version": 1, "tasks": [
        {"id": "cron-1", "enabled": True, "trigger": {"type": "cron", "expr": "0 * * * *"},
         "task": {"type": "task", "text": "run"}},
        "a row that is not an object",
    ]})
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as refusal:
        queue_schedules.load_schedule_store(tmp_path)
    assert "not objects" in str(refusal.value) and "index 1" in str(refusal.value)
    refused = schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="pause",
                                                    actor="agent", drive_root=tmp_path)
    assert refused["status"] == "store_unreadable" and refused["changed"] is False
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
        schedule_lifecycle.upsert_scheduled_task({"id": "new", "trigger": {"type": "cron", "expr": "0 * * * *"}},
                                              drive_root=tmp_path)
    queue_schedules.check_scheduled_tasks()
    assert path.read_text(encoding="utf-8") == contents
    # The lenient READ projection still renders what it can understand.
    assert [r["id"] for r in _rows(tmp_path) if isinstance(r, dict)] == ["cron-1"]


def test_an_absent_table_is_an_empty_one_and_may_be_created(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    stored = schedule_lifecycle.upsert_scheduled_task(
        {"id": "first", "name": "first", "trigger": {"type": "cron", "expr": "0 * * * *"},
         "task": {"type": "task", "text": "run"}}, drive_root=tmp_path)
    assert stored["id"] == "first" and stored["audit"] == "recorded"
    assert [r["id"] for r in _rows(tmp_path)] == ["first"]


# --- audit before the change, audit after it ---------------------------------------


def test_an_unwritable_audit_stops_the_mutation_before_it_happens(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row()])
    monkeypatch.setattr(ouro_utils, "append_jsonl", lambda *_a, **_k: False)
    refused = schedule_lifecycle.mutate_scheduled_task("disable", "skill-demo-daily", reason="pause",
                                                    actor="agent", drive_root=tmp_path)
    assert refused["ok"] is False and refused["status"] == "audit_unavailable"
    assert refused["changed"] is False and refused["audit"] == "not_written"
    assert _rows(tmp_path)[0]["enabled"] is True  # nothing changed
    with pytest.raises(queue_schedules.ScheduleRefused) as refusal:
        schedule_lifecycle.upsert_scheduled_task(
            {"id": "x", "trigger": {"type": "cron", "expr": "0 * * * *"}}, drive_root=tmp_path)
    assert refusal.value.status == "audit_unavailable"
    assert [r["id"] for r in _rows(tmp_path)] == ["skill-demo-daily"]


def test_a_lost_outcome_record_is_disclosed_and_never_rolled_back(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row()])
    calls = {"n": 0}
    real = ouro_utils.append_jsonl

    def _first_only(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs) if calls["n"] == 1 else False

    monkeypatch.setattr(ouro_utils, "append_jsonl", _first_only)
    outcome = schedule_lifecycle.mutate_scheduled_task("disable", "skill-demo-daily", reason="pause",
                                                    actor="agent", drive_root=tmp_path)
    assert outcome["ok"] is False and outcome["status"] == "changed_audit_incomplete"
    assert outcome["changed"] is True and outcome["audit"] == "incomplete"
    # No rollback: an automatic undo would be a second, equally unaudited mutation.
    assert _rows(tmp_path)[0]["enabled"] is False


# --- the projection and the merge --------------------------------------------------


def test_the_projection_names_one_lifecycle_word_per_row(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [
        {"id": "a", "enabled": True, "trigger": {"type": "cron", "expr": "* * * * *"}},
        {"id": "b", "enabled": False, "trigger": {"type": "cron", "expr": "* * * * *"}},
        _skill_row("c", enabled=False),
        _once_row(),
    ])
    rows = queue_schedules.schedule_activity_projection(
        queue_schedules.list_scheduled_tasks(tmp_path))["tasks"]
    assert [r["status"] for r in rows] == ["active", "disabled", "disabled", "consumed"]
    assert [r["retained"] for r in rows] == [False, False, False, True]
    assert [r["restorable"] for r in rows] == [False, False, False, False]
    for row in rows:
        assert queue_schedules.schedule_lifecycle_status(row) == row["status"]


def test_upsert_merges_stale_gateway_record_without_clobbering_runtime_history(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{
        "id": "cron-1", "name": "cron", "enabled": True, "source": "skill_manifest", "skill": "demo",
        "created_at": "2026-09-01T00:00:00+00:00", "last_run_at": "2026-09-21T01:00:00+00:00",
        "last_task_id": "task-new", "next_run_at": "2026-09-22T01:00:00+00:00",
        "trigger": {"type": "cron", "expr": "0 * * * *"}, "task": {"type": "task", "text": "run"},
    }])
    stored = schedule_lifecycle.upsert_scheduled_task({
        "id": "cron-1", "name": "edited", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 * * * *"}, "task": {"type": "task", "text": "edited"},
        "source": "task_followup", "skill": "impostor", "created_at": "stale",
        "last_run_at": "stale", "last_task_id": "stale", "next_run_at": "stale",
    }, drive_root=tmp_path)
    assert stored["created_at"] == "2026-09-01T00:00:00+00:00"
    assert stored["last_task_id"] == "task-new"
    assert stored["next_run_at"] == "2026-09-22T01:00:00+00:00"
    # Provenance is the runtime's, not the payload's: a stale (or wrong) source
    # and skill in the body cannot relabel which lifecycle owns this row.
    assert stored["source"] == "skill_manifest" and stored["skill"] == "demo"
    assert stored["name"] == "edited" and stored["enabled"] is True


def test_a_timing_edit_recomputes_the_next_run(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{
        "id": "cron-1", "name": "cron", "enabled": True, "timezone": "UTC",
        "next_run_at": "2026-09-22T01:00:00+00:00",
        "trigger": {"type": "cron", "expr": "0 * * * *"}, "task": {"type": "task", "text": "run"},
    }])
    same = schedule_lifecycle.upsert_scheduled_task({
        "id": "cron-1", "name": "cron", "enabled": True, "timezone": "UTC",
        "trigger": {"type": "cron", "expr": "0 * * * *"}, "task": {"type": "task", "text": "run"},
    }, drive_root=tmp_path)
    assert same["next_run_at"] == "2026-09-22T01:00:00+00:00"
    moved = schedule_lifecycle.upsert_scheduled_task({
        "id": "cron-1", "name": "cron", "enabled": True, "timezone": "UTC",
        "trigger": {"type": "cron", "expr": "30 5 * * *"}, "task": {"type": "task", "text": "run"},
    }, drive_root=tmp_path)
    assert moved["next_run_at"] != "2026-09-22T01:00:00+00:00"
    assert moved["next_run_at"].endswith(("05:30:00+00:00", "05:30:00"))
