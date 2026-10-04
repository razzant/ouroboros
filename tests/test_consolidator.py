"""The chat stream the legacy memory was cut from, after the old dialogue writer is retired.

The writer's tests (block size, cursor, eras, formatter) left with it. What stays is
the stream it read and the chronicle import still positions legacy blocks on: every
room of the one identity, A2A transport excluded.
"""
import json

from ouroboros.chat_chain import _read_chat_entries


def test_read_chat_entries_includes_project_rows_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the one identity's dialogue memory is its WHOLE
    conversation — main AND project threads — because Ouroboros is one awareness/biography
    (BIBLE P1). Only A2A virtual transport is excluded from the stream."""
    from ouroboros.projects_registry import create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    chat_path = logs / "chat.jsonl"
    rows = [
        {"chat_id": 1, "direction": "in", "text": "main-1"},
        {"chat_id": project_chat, "direction": "in", "text": "project-visible"},
        {"chat_id": 1, "direction": "out", "text": "main-2"},
        {"chat_id": -1001, "direction": "out", "text": "a2a-noise"},
    ]
    chat_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    texts = [e.get("text") for e in _read_chat_entries(chat_path)]
    assert texts == ["main-1", "project-visible", "main-2"]  # full awareness, A2A excluded
