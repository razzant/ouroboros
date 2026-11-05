"""Regression coverage for quiet review references on Chat history rails."""

from __future__ import annotations

from ouroboros import dialogue_store as ds
from ouroboros.gateway.history import (
    _collect_progress_rows,
)


def test_duplicate_lifecycle_pointer_replays_without_a_fake_task(tmp_path):
    progress = tmp_path / "logs" / "progress.jsonl"
    pointer = {
        "job_id": "skill-job-1", "kind": "review", "target": "alpha",
        "status": "running", "chat_id": 17,
        "group_id": "task:root:alpha", "task_id": "child",
        "root_task_id": "root", "origin_task_id": "child",
        "origin_root_task_id": "root", "presentation_owner_task_id": "root",
        "source": "tool",
    }
    ds.append(tmp_path, "progress", {
        "ts": "2026-08-25T00:00:00Z", "type": "send_message",
        "task_id": "", "is_progress": True, "chat_id": 25,
        "content": "already running", "lifecycle_pointer": pointer,
    })

    rows, _quota = _collect_progress_rows(
        progress, tmp_path / "archive", 10, lambda _chat, _entry: True,
    )
    assert len(rows) == 1
    assert rows[0]["task_id"] == ""
    assert rows[0]["lifecycle_pointer"] == pointer
