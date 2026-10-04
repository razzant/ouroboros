"""Tests for chat/system provenance handling."""

from __future__ import annotations

import json

from ouroboros.dialogue_provenance import render_row_text, row_author
from ouroboros.memory import Memory


def test_chat_history_marks_system_entries(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-03-19T16:53:30.629879+00:00",
            "direction": "system",
            "type": "task_summary",
            "text": "Reflection from the previous task.",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    memory = Memory(drive_root=tmp_path)
    history = memory.chat_history()

    assert "[task_summary] Reflection from the previous task." in history
    assert "[User]" not in history


def test_memory_read_keeps_a_system_row_a_host_fact():
    """The retired block formatter signed system rows "Ouroboros"; the row's own fields now
    sign it as the host's fact, never my speech, with its text intact."""
    row = {
        "ts": "2026-03-19T16:53:30.629879+00:00",
        "direction": "system",
        "type": "task_summary",
        "text": "Detailed task summary.",
    }
    author = row_author(row)
    assert author["kind"] == "host" and author["type"] == "task_summary"
    assert "Ouroboros" not in author["label"]
    assert render_row_text(row) == "Detailed task summary."
    assert row_author({**row, "direction": "out"})["kind"] == "ouroboros"
