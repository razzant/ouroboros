"""Tests for chat/system provenance handling."""

from __future__ import annotations

import json

from ouroboros.chronicle_sources import format_source_row
from ouroboros.memory import Memory


def test_summarize_chat_marks_system_entries(tmp_path):
    memory = Memory(drive_root=tmp_path)
    summary = memory.summarize_chat([
        {
            "ts": "2026-03-19T16:53:30.629879+00:00",
            "direction": "system",
            "type": "task_summary",
            "text": "The user requested a restart of the scenario.",
        }
    ])

    assert "📋 16:53 [task_summary] The user requested a restart of the scenario." in summary
    assert "[User]" not in summary


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


def test_chronicle_source_preserves_system_direction():
    formatted = format_source_row(
        {
            "ts": "2026-03-19T16:53:30.629879+00:00",
            "direction": "system",
            "type": "task_summary",
            "text": "Detailed task summary.",
        }
    )

    assert formatted.startswith("[2026-03-19T16:53:30.629879+00:00; Ouroboros;")
    assert json.loads(formatted.splitlines()[1]) == {"direction": "system", "type": "task_summary"}
    assert formatted.endswith("\nDetailed task summary.")
