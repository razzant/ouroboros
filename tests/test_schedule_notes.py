"""Notes (``kind: "notify"`` rows): the mind's own words, shown to the owner at their time.

The scheduler tick consumes a due note in its one table write and, after the lock,
``supervisor/schedule_notes.py`` puts the words in the owner's chat as one signed
``reminder`` System row through the existing ``send_with_budget`` seam: no model
call, no task. These tests pin the accepted contract (DESIGN "Chat authorship and
System rows"): consumed before shown, shown once, visible in history and in the next
turn's context, both times after downtime, and never a model task.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

ONCE_PAST = {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"}


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The real scheduler tick, message bus and durable chat log on a temp root."""
    from supervisor import message_bus, queue

    frames, published = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: published.append((topic, data)))
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    return types.SimpleNamespace(root=tmp_path, queue=queue, pending=pending, frames=frames, published=published)


def _note(queue, schedule_id="note-1", *, text="Call mother", trigger=None, source="task_followup", **extra):
    record = {"id": schedule_id, "name": "Reminder of task t1", "description": text, "kind": "notify",
              "source": source, "enabled": True, "timezone": "", "trigger": trigger or dict(ONCE_PAST),
              "notification": {"text": text, "set_at": "1999-12-31T22:00:00+00:00"}, **extra}
    relation = {"kind": "independent", "declared_by": "t1", "revision": "r1"}
    return queue.upsert_scheduled_task(
        record, host_followup={"followup_origin": {"task_id": "t1", "root_task_id": "t1"}, "followup_relation": relation})


def _chat_rows(root):
    path = root / "logs" / "chat.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row(queue, root, schedule_id="note-1"):
    return next(r for r in queue.list_scheduled_tasks(root)["tasks"] if r["id"] == schedule_id)


def test_a_due_note_is_one_signed_system_row_and_never_a_task(host):
    _note(host.queue)
    host.queue.check_scheduled_tasks()
    host.queue.check_scheduled_tasks()  # a consumed note is history: a repeated tick shows nothing new
    assert host.pending == [], "no task is admitted and so no model call is made"
    [row] = _chat_rows(host.root)
    assert (row["direction"], row["type"], row["chat_id"], row["task_id"]) == ("system", "reminder", 1, "")
    signature, body = row["text"].split("\n", 1)
    assert signature.startswith("Reminder · Ouroboros · written ") and " · for " in signature
    assert body == "Call mother", "the mind's words ride verbatim"
    [frame] = [f for f in host.frames if f.get("type") == "chat"]
    assert (frame["role"], frame["system_type"], frame["source"]) == ("system", "reminder", "Ouroboros")
    assert frame["set_at"] == "1999-12-31T22:00:00+00:00"
    assert frame["scheduled_for"] == ONCE_PAST["run_at"] and frame["delivered_at"]
    assert [topic for topic, _ in host.published] == ["chat.outbound"], "transports see it like any System row"
    done = _row(host.queue, host.root)
    assert done["enabled"] is False and done["completed_at"] and done["next_run_at"] == ""
    assert done["last_error"] == "" and not done.get("last_task_id") and "occurrence" not in done


def test_an_overdue_note_still_arrives_and_shows_both_times(host):
    _note(host.queue)  # due in 2000: the process was "off" for decades
    host.queue.check_scheduled_tasks()
    signature = _chat_rows(host.root)[0]["text"].split("\n", 1)[0]
    assert " · for Jan 1 " in signature and " · delivered " in signature, signature


def test_an_overdue_note_reaches_extension_subscribers_on_a_provider_boot(host, monkeypatch):
    """The first scheduler tick may consume an overdue note and the event bus keeps no
    history, so a provider boot attaches the extensions' subscriptions (Telegram's
    ``chat.outbound``) before it starts the supervisor. Real lifespan, scheduler tick,
    message bus and global event bus; the supervisor start runs that first tick to its
    end, the race's worst case (extension loading slower than supervisor init)."""
    import threading

    from starlette.testclient import TestClient

    import server as srv
    from ouroboros import event_bus, extension_loader
    from supervisor import message_bus
    from tests._shared import clean_extension_runtime_state
    from tests.test_extensions_api import _patch_lifespan_for_drive_root_test

    monkeypatch.setattr(message_bus, "publish_event", event_bus.publish_event)  # the real bus, not the capture
    _note(host.queue)
    monkeypatch.setattr(srv.app.app.state, "drive_root", host.root, raising=False)
    monkeypatch.setattr(srv.app.app.state, "repo_dir", host.root / "repo", raising=False)
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, {})
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda _settings: True)
    monkeypatch.setattr(srv, "_boot_managed_update_tasks", lambda: None)
    delivered = []

    def reload_extensions(_root, _reader, *, repo_path=None):
        event_bus.get_global_event_bus().subscribe("telegram", event_bus.CHAT_OUTBOUND, delivered.append)
        return {}

    def start_supervisor(_settings):
        first_tick = threading.Thread(target=host.queue.check_scheduled_tasks)
        first_tick.start()
        first_tick.join(timeout=30)
        return True

    monkeypatch.setattr(extension_loader, "reload_all", reload_extensions)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", start_supervisor)
    try:
        with TestClient(srv.app):
            pass
    finally:
        event_bus.init_global_event_bus()
        clean_extension_runtime_state()
    assert len(_chat_rows(host.root)) == 1 and _row(host.queue, host.root)["completed_at"]
    assert [event.get("system_type") for event in delivered] == ["reminder"], "the late subscriber missed the note"


def test_the_note_is_consumed_on_disk_before_it_is_shown_and_never_retried(host, monkeypatch):
    """A crash or an unknown write after consumption may lose one note; it can never show twice."""
    from supervisor import message_bus

    _note(host.queue)
    seen = []

    def crashing_send(*_args, **_kwargs):
        seen.append(_row(host.queue, host.root))  # what a restart would read at this instant
        raise RuntimeError("canonical message acceptance could not be persisted")

    monkeypatch.setattr(message_bus, "send_with_budget", crashing_send)
    host.queue.check_scheduled_tasks()
    host.queue.check_scheduled_tasks()
    assert len(seen) == 1 and seen[0]["completed_at"] and seen[0]["enabled"] is False
    assert _row(host.queue, host.root)["last_error"] == "delivery not confirmed"
    assert _chat_rows(host.root) == []


def test_an_unconfirmed_note_says_so_to_the_model_and_in_activity(host, monkeypatch):
    """The stored delivery error is on both schedule surfaces: the model's bounded
    ``manage_schedules`` list and the owner's Activity rows. Still never retried."""
    from ouroboros.tools.followup import _manage_schedules
    from supervisor import message_bus, queue_schedules

    def unconfirmed(*_args, **_kwargs):
        raise RuntimeError("canonical message acceptance could not be persisted")

    _note(host.queue)
    _note(host.queue, "note-2", trigger={"type": "once", "run_at": "2999-01-01T00:00:00+00:00"})
    monkeypatch.setattr(message_bus, "send_with_budget", unconfirmed)
    host.queue.check_scheduled_tasks()
    ctx = types.SimpleNamespace(task_metadata={}, drive_root=host.root, budget_drive_root=host.root, task_id="t")
    listed = {row["id"]: row for row in json.loads(_manage_schedules(ctx, action="list"))["tasks"]}
    assert (listed["note-1"]["status"], listed["note-1"]["last_error"]) == ("consumed", "delivery not confirmed")
    assert not listed["note-2"]["last_error"], "a row with no stored error says none"
    activity = queue_schedules.schedule_activity_projection(host.queue.list_scheduled_tasks(host.root))["tasks"]
    assert {row["id"]: row.get("last_error") for row in activity}["note-1"] == "delivery not confirmed"
    long_error = {"id": "x", "trigger": {"type": "cron", "expr": "0 9 * * *"}, "last_error": "E" * 500}
    assert len(queue_schedules.schedule_tool_projection({"tasks": [long_error]})["tasks"][0]["last_error"]) <= 96


def test_a_failed_table_write_shows_nothing_and_the_note_stays_due(host, monkeypatch):
    from supervisor import queue_schedules

    _note(host.queue)
    real_write = queue_schedules._write_scheduled_tasks

    def refuse(*_args, **_kwargs):
        raise OSError("scheduled task store write returned False")

    monkeypatch.setattr(queue_schedules, "_write_scheduled_tasks", refuse)
    with pytest.raises(OSError):
        host.queue.check_scheduled_tasks()
    assert _chat_rows(host.root) == [] and _row(host.queue, host.root)["enabled"] is True
    monkeypatch.setattr(queue_schedules, "_write_scheduled_tasks", real_write)
    host.queue.check_scheduled_tasks()
    assert len(_chat_rows(host.root)) == 1


def test_a_cron_note_collapses_missed_points_into_one_and_moves_on(host):
    from supervisor import queue_schedules

    _note(host.queue, trigger={"type": "cron", "expr": "0 9 * * *"})
    store = host.queue.load_schedule_store(host.root)
    store["tasks"][0]["next_run_at"] = "2000-01-01T09:00:00+00:00"  # every morning since 2000 was missed
    queue_schedules._write_scheduled_tasks(store, host.root)
    host.queue.check_scheduled_tasks()
    host.queue.check_scheduled_tasks()
    assert len(_chat_rows(host.root)) == 1
    row = _row(host.queue, host.root)
    assert row["enabled"] is True and not row.get("completed_at") and row["next_run_at"] > "2026"


def test_a_cron_note_that_cannot_advance_is_never_shown(host):
    from supervisor import queue_schedules

    _note(host.queue, trigger={"type": "cron", "expr": "0 9 * * *"})
    store = host.queue.load_schedule_store(host.root)
    store["tasks"][0].update(next_run_at="2000-01-01T09:00:00+00:00", trigger={"type": "cron", "expr": "not a cron"})
    queue_schedules._write_scheduled_tasks(store, host.root)
    host.queue.check_scheduled_tasks()
    host.queue.check_scheduled_tasks()
    assert _chat_rows(host.root) == []
    assert _row(host.queue, host.root)["last_error"]


def test_the_owner_governs_a_note_like_any_schedule(host):
    _note(host.queue)
    common = {"reason": "owner chose it in Activity", "actor": "owner:gateway", "drive_root": host.root}
    assert host.queue.mutate_scheduled_task("disable", "note-1", **common)["status"] == "updated"
    host.queue.check_scheduled_tasks()
    assert _chat_rows(host.root) == []
    assert host.queue.mutate_scheduled_task("restore", "note-1", **common)["status"] == "updated"
    host.queue.check_scheduled_tasks()
    assert len(_chat_rows(host.root)) == 1
    _note(host.queue, "note-2", trigger={"type": "once", "run_at": "2999-01-01T00:00:00+00:00"})
    outcome = host.queue.mutate_scheduled_task("delete", "note-2", **common)
    assert outcome["status"] == "deleted" and outcome["changed"] is True, "a note owes no task: removed outright"


def test_history_shows_the_row_and_only_live_frames_ring(host):
    """Reopening the chat replays the row through history, which never reaches the notifier
    (DEVELOPMENT "notifications ring for live events only"); the row itself stays visible."""
    from ouroboros.gateway.history import make_chat_history_endpoint

    _note(host.queue)
    host.queue.check_scheduled_tasks()
    endpoint = make_chat_history_endpoint(host.root)
    response = asyncio.run(endpoint(types.SimpleNamespace(query_params={"chat_id": "1"})))
    messages = json.loads(response.body)["messages"]
    [replayed] = [m for m in messages if m.get("system_type") == "reminder"]
    assert replayed["role"] == "system" and replayed["text"].endswith("\nCall mother")


def test_the_next_turn_reads_the_note_as_a_host_fact(host):
    from tests._memory_view_context import room_text

    _note(host.queue)
    host.queue.check_scheduled_tasks()
    rendered = room_text(host.root)  # the room of Main's next turn: a notice shows by its words
    [line] = [line for line in rendered.splitlines() if "Reminder · Ouroboros · written " in line]
    assert line.startswith("[") and "; host" in line.split("]", 1)[0], line
    assert "Call mother" in rendered and "; Ouroboros;" not in line  # a host fact, never my speech


def test_a_note_never_becomes_a_model_task(host):
    stored = _note(host.queue)
    with pytest.raises(ValueError, match="note"):
        host.queue._task_from_schedule(stored)


def test_a_note_from_another_source_keeps_its_own_author(host):
    """The voice follows authorship: only the mind's own follow-up is signed Ouroboros."""
    _note(host.queue, source="owner")
    host.queue.check_scheduled_tasks()
    assert _chat_rows(host.root)[0]["text"].startswith("Reminder · owner · written ")


def test_note_label_reads_both_times_in_the_schedule_zone():
    from supervisor.schedule_notes import note_text

    note = {"text": "Call mother", "set_at": "2026-10-03T11:05:00+00:00",
            "scheduled_for": "2026-10-03T12:00:00+00:00", "timezone": "Europe/Moscow", "author": "Ouroboros"}
    assert note_text(note, "2026-10-03T12:00:20+00:00") == (
        "Reminder · Ouroboros · written Oct 3 14:05 · for Oct 3 15:00 (UTC+3)\nCall mother")
    assert note_text(note, "2026-10-04T06:12:00+00:00") == (
        "Reminder · Ouroboros · written Oct 3 14:05 · for Oct 3 15:00 · delivered Oct 4 09:12 (UTC+3)\nCall mother")
    assert note_text({**note, "set_at": "", "timezone": "UTC"}, "2026-10-03T12:00:00+00:00") == (
        "Reminder · Ouroboros · for Oct 3 12:00 (UTC)\nCall mother")


@pytest.mark.parametrize(("zone", "set_at", "due", "delivered", "label"), [
    # Clocks go back at 01:00Z: due 02:30 summer time, shown an hour later at 02:30 winter time.
    ("Europe/Berlin", "2026-10-24T10:00:00Z", "2026-10-25T00:30:00Z", "2026-10-25T01:30:00Z",
     "written Oct 24 12:00 (UTC+2) · for Oct 25 02:30 (UTC+2) · delivered Oct 25 02:30 (UTC+1)"),
    ("America/New_York", "2026-10-31T12:00:00-04:00", "2026-11-01T01:30:00-04:00", "2026-11-01T01:30:00-05:00",
     "written Oct 31 12:00 (UTC-4) · for Nov 1 01:30 (UTC-4) · delivered Nov 1 01:30 (UTC-5)"),
    # On time, but written before the change: each time keeps its own offset.
    ("Europe/Berlin", "2026-10-24T10:00:00Z", "2026-10-25T02:00:00Z", "2026-10-25T02:00:10Z",
     "written Oct 24 12:00 (UTC+2) · for Oct 25 03:00 (UTC+1)"),
])
def test_note_label_across_a_dst_change_compares_instants_and_names_each_offset(zone, set_at, due, delivered, label):
    from supervisor.schedule_notes import note_text

    note = {"text": "Call mother", "set_at": set_at, "scheduled_for": due, "timezone": zone, "author": "Ouroboros"}
    assert note_text(note, delivered) == f"Reminder · Ouroboros · {label}\nCall mother"


@pytest.mark.parametrize(("stored", "expected"), [
    ((True, 5_000_000_001), 5_000_000_001),  # a headless install's first owner transport
    ((True, None), 1), ((False, 77), 1), ((True, 0), 1), ((True, -1001), 1), ((True, "x"), 1),
])
def test_notes_go_to_the_owner_chat_else_main(tmp_path, monkeypatch, stored, expected):
    from supervisor import schedule_notes, state

    monkeypatch.setattr(state, "control_in_copy", lambda _path, _key: stored)
    assert schedule_notes.owner_chat_id(tmp_path) == expected
