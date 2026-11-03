"""Checkpoint rows are the producer's own (owner decision 1A(3), 14 Oct 2026).

«промежуточные логи воркер пишет на диск сам»: ``loop_messages._emit_checkpoint_event``
appends the events.jsonl row (and the progress.jsonl row of a System checkpoint)
itself, then emits the one live frame; the supervisor's nested log branch only
addresses and pushes it. One checkpoint is exactly one events row, at most one
progress row and one live frame, queue attached or detached, pooled or
direct-turn producer. Both directions of each guard are pinned.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest
from tests._dialogue import progress_rows

from ouroboros.loop_messages import _emit_checkpoint_event
from supervisor.log_addressing import task_event_queue

SYSTEM_ROW = {"checkpoint_kind": "usage_lock_wait", "owner_visible": True, "phase": "entered",
              "role": "system", "system_type": "task_checkpoint", "content": "Waiting for accounting access"}
TASK = {"id": "t1", "type": "task", "chat_id": 7, "root_task_id": "root-1", "parent_task_id": "parent-1",
        "delegation_role": "subagent"}


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def logs(tmp_path):
    (tmp_path / "logs").mkdir()
    return tmp_path / "logs"


def test_a_system_checkpoint_is_one_events_row_one_progress_row_and_one_frame(logs):
    bus: "queue.Queue" = queue.Queue()
    _emit_checkpoint_event(task_event_queue(bus, TASK), "t1", logs, dict(SYSTEM_ROW))
    [event_row] = _rows(logs / "events.jsonl")
    [progress_row] = progress_rows(logs.parent)
    frame = bus.get_nowait()
    assert bus.empty() and frame["type"] == "log_event"
    assert event_row == progress_row
    assert event_row["type"] == "task_checkpoint" and event_row["task_id"] == "t1"
    assert event_row["ts"] == frame["data"]["ts"], "the row is the frame's own event, one ts"
    # The row is attributed the way the frame is: lineage and chat by value, before the write.
    assert event_row["chat_id"] == 7 and event_row["root_task_id"] == "root-1"
    assert event_row["parent_task_id"] == "parent-1" and frame["data"]["root_task_id"] == "root-1"


def test_a_plain_checkpoint_writes_events_only_and_a_detached_queue_still_writes(logs):
    bus: "queue.Queue" = queue.Queue()
    _emit_checkpoint_event(bus, "t1", logs, {"checkpoint_kind": "context_reclaim", "round": 3})
    assert len(_rows(logs / "events.jsonl")) == 1 and not progress_rows(logs.parent)
    assert bus.get_nowait()["data"]["checkpoint_kind"] == "context_reclaim" and bus.empty()

    _emit_checkpoint_event(None, "t1", logs, dict(SYSTEM_ROW))
    assert len(_rows(logs / "events.jsonl")) == 2 and len(progress_rows(logs.parent)) == 1

    _emit_checkpoint_event(bus, "t1", None, dict(SYSTEM_ROW))  # no log dir: the frame alone
    assert len(_rows(logs / "events.jsonl")) == 2 and bus.get_nowait()["data"]["phase"] == "entered"


def test_a_failed_row_append_still_sends_the_live_frame(logs, monkeypatch):
    from ouroboros import utils

    def refuse(path, obj, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "append_jsonl", refuse)
    bus: "queue.Queue" = queue.Queue()
    _emit_checkpoint_event(bus, "t1", logs, dict(SYSTEM_ROW))
    assert bus.get_nowait()["data"]["checkpoint_kind"] == "usage_lock_wait"
    assert not (logs / "events.jsonl").exists()


def test_the_supervisor_handler_writes_nothing_for_a_checkpoint_and_keeps_the_settings_disclosure(tmp_path):
    from supervisor.events_worker_reports import _handle_log_event

    written, frames = [], []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=lambda path, row: written.append((path, row)),
                          bridge=SimpleNamespace(push_log=frames.append))
    _handle_log_event({"type": "log_event", "data": {"type": "task_checkpoint", "task_id": "t1", "chat_id": 7, **SYSTEM_ROW}}, ctx)
    assert written == [] and len(frames) == 1 and frames[0]["type"] == "task_checkpoint" and frames[0]["chat_id"] == 7
    _handle_log_event({"type": "log_event", "data": {"type": "task_start_settings_reload_failed", "task_id": "t1",
                                                     "error": "RuntimeError: boom"}}, ctx)
    assert len(written) == 1 and written[0][0] == tmp_path / "logs" / "events.jsonl"
    assert written[0][1]["type"] == "task_start_settings_reload_failed" and len(frames) == 2


def test_a_direct_turn_producer_in_the_server_process_writes_once_and_shows_once(tmp_path):
    """The server process runs the direct lane's producers with the production
    server sink installed: the producer's append is suppressed there, the
    handler pushes the one frame — one row, one frame."""
    from ouroboros.utils import set_log_sink
    from supervisor.events import make_server_log_sink
    from supervisor.events_worker_reports import _handle_log_event
    from supervisor.log_addressing import TurnEventQueue

    (tmp_path / "logs").mkdir()
    frames, running = [], {}
    bridge = SimpleNamespace(push_log=frames.append)
    set_log_sink(make_server_log_sink(bridge, tmp_path, running=running))
    try:
        bus: "queue.Queue" = queue.Queue()
        _emit_checkpoint_event(TurnEventQueue(bus, "turn-1", 42, initiator="consciousness"), "turn-1",
                               tmp_path / "logs", dict(SYSTEM_ROW))
        assert frames == [], "the producer's own append is not a second live copy"
        [row] = _rows(tmp_path / "logs" / "events.jsonl")
        assert row["chat_id"] == 42 and row["_is_direct_chat"] is True and row["initiator"] == "consciousness"
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=running, bridge=bridge,
                              append_jsonl=lambda path, obj: pytest.fail("the handler persists no checkpoint"))
        _handle_log_event(bus.get_nowait(), ctx)
        assert len(frames) == 1 and frames[0]["chat_id"] == 42 and frames[0]["ts"] == row["ts"]
        assert len(_rows(tmp_path / "logs" / "events.jsonl")) == 1 and len(progress_rows(tmp_path)) == 1
    finally:
        set_log_sink(None)


def test_the_budget_pause_checkpoint_rides_the_same_producer_path(logs):
    from ouroboros.agent import _emit_budget_pause_checkpoint

    limit = {"scope": "root", "status": "resource_limited", "resume_policy": "cancel_or_new_run"}
    bus: "queue.Queue" = queue.Queue()
    _emit_budget_pause_checkpoint(task_event_queue(bus, TASK), logs, "t1", dict(limit))
    [row] = _rows(logs / "events.jsonl")
    frame = bus.get_nowait()["data"]
    assert bus.empty() and not progress_rows(logs.parent)
    assert row["checkpoint_kind"] == frame["checkpoint_kind"] == "budget_scope_paused"
    assert row["toast_once"] == "t1:budget-paused:root" and row["scope"] == "root" and row["chat_id"] == 7
    assert row["ts"] == frame["ts"]
    _emit_budget_pause_checkpoint(None, logs, "t1", dict(limit))
    assert len(_rows(logs / "events.jsonl")) == 2
