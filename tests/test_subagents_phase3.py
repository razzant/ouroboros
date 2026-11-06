from __future__ import annotations

import queue


def test_schedule_subagent_drive_failure_is_fail_closed(monkeypatch, tmp_path):
    """A drive that cannot be prepared leaves NOTHING behind: no event, no durable record,
    no half-provisioned state directory."""
    import ouroboros.tools.control_scheduling as control
    from ouroboros.headless import HEADLESS_TASKS_DIR
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    def fake_prepare(_root, _tid, _mode):
        raise RuntimeError("boom")

    monkeypatch.setattr(control, "prepare_task_drive", fake_prepare)
    event_queue: queue.Queue = queue.Queue()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = event_queue
    ctx.task_metadata = {"root_task_id": "root1", "session_id": "sess1"}

    result = control._schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Review the design",
        expected_output="One findings list",
        role="reviewer",
    )

    assert "SUBTASK_DRIVE_ERROR" in result
    assert event_queue.empty()
    assert not any((tmp_path / "task_results").glob("*.json"))
    assert not any((tmp_path / HEADLESS_TASKS_DIR).glob("*"))
