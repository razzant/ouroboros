"""One global census projects the same recorded narration for Main and its room."""
from types import SimpleNamespace

import pytest

from ouroboros.gateway.state import _chat_activities_snapshot_safe
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_pacing import record_tool_activity
from supervisor import message_bus, queue


def _queue(monkeypatch, root, pending=(), running=None):
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "RUNNING", running or {})
    monkeypatch.setattr(queue, "PENDING", list(pending))
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})


def _delivery(monkeypatch, root):
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a: None)
    return frames


def test_moved_work_uses_narration_and_incremental_counts_without_history_reads(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_activity_facts import record_narration
    from pathlib import Path

    frames = _delivery(monkeypatch, tmp_path)
    project = create_project(tmp_path, "room", name="Room")
    bind_task_to_project(tmp_path, "work", project["id"], project["chat_id"], origin={"absent": "system"})
    write_task_result(tmp_path, "work", "running", chat_id=1, metadata={"detached_at": 100})
    write_task_result(tmp_path, "sibling", "running", chat_id=1)
    _queue(monkeypatch, tmp_path, running={tid: {"task": {"id": tid, "chat_id": 1}, "attempt": 1,
        "started_at": 100.0} for tid in ("work", "sibling")})
    bindings = {"work": {"project_id": project["id"], "chat_id": project["chat_id"], "project_name": "Room"}}
    usage = {"_pacing_tool_activity": {"read_file": {"calls": 2, "error_calls": 0}}}
    context = SimpleNamespace(task_id="work", drive_root=tmp_path, _accumulated_usage=usage)
    record_tool_activity(context, {"fn_name": "read_file", "is_error": False})
    for text, voice, stamp in (("💬 **Checking** the change", True, "2026-10-16T12:00:00Z"),
                               ("💬 Host checkpoint", False, "2026-10-16T12:00:01Z"),
                               (" ", True, "2026-10-16T12:00:02Z")):
        message_bus.send_with_budget(1, text, is_progress=True, task_id="work", narration=voice, ts=stamp)
    # A delayed older delivery cannot overwrite the newer visible model narration.
    record_narration(tmp_path, {"task_id": "work", "narration": True, "text": "Older",
                                "ts": "2026-10-16T11:59:00Z"})
    original_open = Path.open
    reads = []
    def checked_open(path, *args, **kwargs):
        if path.suffix == ".jsonl":
            reads.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    for _ in range(3):
        rows = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(tmp_path, bindings, direct_turns=[])}
        work = rows["work"]
        assert work["last_line"] == "**Checking** the change"
        assert work["last_line_at"] == "2026-10-16T12:00:00Z"
        assert work["tool_calls"] == 3 and work["detached"] is True
        assert (work["project_id"], work["project_name"], work["chat_id"]) == ("room", "Room", project["chat_id"])
        assert work["phase"] == "working" and work["started_at"] == 100.0
        assert rows["sibling"]["chat_id"] == 1 and rows["sibling"]["project_name"] == ""
        assert rows["sibling"]["last_line"] == "" and rows["sibling"]["detached"] is False
    assert reads == []
    assert next(frame for frame in frames if frame.get("narration"))['content'] == "💬 **Checking** the change"
    record_narration(tmp_path, {"task_id": "work", "narration": True, "text": "Verification finished",
                                "ts": "2026-10-16T12:01:00Z"})
    record_tool_activity(context, {"fn_name": "read_file", "is_error": True})
    refreshed = next(row for row in _chat_activities_snapshot_safe(tmp_path, bindings, direct_turns=[])
                     if row["activity_id"] == "work")
    assert refreshed["last_line"] == "Verification finished" and refreshed["tool_calls"] == 4
    assert reads == []  # new task signature refreshes the memo, without a history replay


@pytest.mark.parametrize("carrier", ["direct", "pending", "running"])
@pytest.mark.parametrize("detached", [False, True])
def test_detached_is_a_durable_fact_independent_of_carrier(tmp_path, monkeypatch, carrier, detached):
    task = {"id": "work", "chat_id": 1, "_is_direct_chat": True, "detached": True,
            "detached_at": 100 if detached else None}  # only K7's timestamp is evidence
    write_task_result(tmp_path, "work", "running", **{k: v for k, v in task.items() if k != "id"})
    _queue(monkeypatch, tmp_path, pending=[task] if carrier == "pending" else [],
           running={"work": {"task": task, "attempt": 1}} if carrier == "running" else {})
    direct = [{"activity_id": "work", "chat_id": 1, "project_id": "", "kind": "direct_chat", "phase": "thinking"}]
    [row] = _chat_activities_snapshot_safe(tmp_path, direct_turns=direct if carrier == "direct" else [])
    assert row["detached"] is detached
    assert row["phase"] == {"direct": "thinking", "pending": "queued", "running": "working"}[carrier]
    assert row["tool_calls"] == 0 and row["last_line"] == ""


def test_unstored_narration_and_missing_task_never_create_display_truth(tmp_path, monkeypatch):
    from ouroboros.task_activity_facts import record_narration
    from ouroboros.task_store import signature
    _delivery(monkeypatch, tmp_path)
    write_task_result(tmp_path, "work", "running")
    before = signature(tmp_path, "work")
    monkeypatch.setattr(message_bus.dialogue_store, "append_quietly", lambda *a, **k: None)
    message_bus.send_with_budget(1, "Unsaved", is_progress=True, task_id="work", narration=True)
    assert signature(tmp_path, "work") == before
    assert "last_line" not in load_task_result(tmp_path, "work")
    record_narration(tmp_path, {"task_id": "missing", "narration": True, "text": "Hello"})
    assert load_task_result(tmp_path, "missing") is None
