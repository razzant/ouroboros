"""
Tests for BackgroundConsciousness helpers.

Verifies progress events have the correct shape, reach the queue,
and respect pause / chat_id=None semantics. Also covers backlog digest
inclusion in background context.

Run: pytest tests/test_consciousness.py -v
"""

import json
import os
import pathlib
import queue
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEmitProgress(unittest.TestCase):
    """Tests for BackgroundConsciousness._emit_progress."""

    def _make_consciousness(self, chat_id=42, event_queue=None):
        """Create a BackgroundConsciousness with mocked dependencies."""
        from ouroboros.consciousness import BackgroundConsciousness

        tmpdir = tempfile.mkdtemp()
        drive_root = pathlib.Path(tmpdir)
        (drive_root / "logs").mkdir(parents=True, exist_ok=True)
        repo_dir = pathlib.Path(tmpdir) / "repo"
        repo_dir.mkdir()

        eq = event_queue if event_queue is not None else queue.Queue()

        with patch.object(BackgroundConsciousness, '_build_registry', return_value=MagicMock()):
            bc = BackgroundConsciousness(
                drive_root=drive_root,
                repo_dir=repo_dir,
                event_queue=eq,
                owner_chat_id_fn=lambda: chat_id,
            )
        return bc, eq, drive_root

    def test_event_shape(self):
        """Event has type, chat_id, text, is_progress, ts."""
        bc, eq, _ = self._make_consciousness(chat_id=99)
        bc._emit_progress("thinking about things")
        evt = eq.get_nowait()

        self.assertEqual(evt["type"], "send_message")
        self.assertEqual(evt["chat_id"], 99)
        self.assertEqual(evt["text"], "💬 thinking about things")
        self.assertEqual(evt["format"], "markdown")
        self.assertTrue(evt["is_progress"])
        self.assertIn("ts", evt)

    def test_empty_content_skipped(self):
        """Empty or whitespace-only content produces no event."""
        bc, eq, drive_root = self._make_consciousness()
        progress_path = drive_root / "logs" / "progress.jsonl"

        bc._emit_progress("")
        bc._emit_progress("   ")
        bc._emit_progress(None)

        self.assertTrue(eq.empty())
        # Also should not persist to file
        self.assertFalse(progress_path.exists())

    def test_chat_id_none_skips_queue_but_persists(self):
        """When chat_id is None, event is NOT queued but IS persisted."""
        bc, eq, drive_root = self._make_consciousness(chat_id=None)
        bc._emit_progress("background thought")

        # Queue should be empty
        self.assertTrue(eq.empty())

        # File should have the entry
        progress_path = drive_root / "logs" / "progress.jsonl"
        self.assertTrue(progress_path.exists())
        entry = json.loads(progress_path.read_text().strip())
        self.assertEqual(entry["type"], "send_message")
        self.assertEqual(entry["content"], "background thought")
        self.assertTrue(entry["is_progress"])

    def test_paused_events_go_to_deferred(self):
        """When paused, events go to _deferred_events, not the queue."""
        bc, eq, _ = self._make_consciousness()
        bc._paused = True
        bc._emit_progress("deferred thought")

        self.assertTrue(eq.empty())
        self.assertEqual(len(bc._deferred_events), 1)
        self.assertEqual(bc._deferred_events[0]["type"], "send_message")
        self.assertEqual(bc._deferred_events[0]["text"], "💬 deferred thought")


class TestBackgroundContext(unittest.TestCase):
    def test_build_context_includes_improvement_backlog_digest(self):
        from ouroboros.consciousness import BackgroundConsciousness

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        drive_root = tmpdir / "drive"
        repo_dir = tmpdir / "repo"
        (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
        (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
        (drive_root / "logs").mkdir(parents=True, exist_ok=True)
        (drive_root / "state").mkdir(parents=True, exist_ok=True)

        (repo_dir / "prompts" / "CONSCIOUSNESS.md").write_text("Consciousness prompt", encoding="utf-8")
        (repo_dir / "BIBLE.md").write_text("Bible", encoding="utf-8")
        (repo_dir / "VERSION").write_text("1.2.3", encoding="utf-8")
        (repo_dir / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
        (repo_dir / "README.md").write_text("README", encoding="utf-8")
        (repo_dir / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
        (repo_dir / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
        (drive_root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
        (drive_root / "memory" / "identity.md").write_text("I am Ouroboros", encoding="utf-8")
        (drive_root / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
        (drive_root / "memory" / "knowledge" / "improvement-backlog.md").write_text(
            "# Improvement Backlog\n\n### ibl-1\n- status: open\n- created_at: 2026-04-14T09:00:00+00:00\n- source: execution_reflection\n- category: process\n- task_id: task-1\n- requires_plan_review: yes\n- fingerprint: fp-1\n- summary: Reduce recurring task friction around REVIEW_BLOCKED\n",
            encoding="utf-8",
        )
        for name in ("chat.jsonl", "progress.jsonl", "tools.jsonl", "events.jsonl", "supervisor.jsonl", "task_reflections.jsonl"):
            (drive_root / "logs" / name).write_text("", encoding="utf-8")

        with patch.object(BackgroundConsciousness, '_build_registry', return_value=MagicMock()):
            bc = BackgroundConsciousness(
                drive_root=drive_root,
                repo_dir=repo_dir,
                event_queue=None,
                owner_chat_id_fn=lambda: None,
            )
        text = bc._build_context()
        self.assertIn("## Improvement Backlog", text)
        self.assertIn("Reduce recurring task friction around REVIEW_BLOCKED", text)


class TestBackgroundConsciousnessToolScope(unittest.TestCase):
    def test_background_consciousness_cannot_execute_or_delegate(self):
        from ouroboros.consciousness import BackgroundConsciousness

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        drive_root = tmpdir / "drive"
        repo_dir = tmpdir / "repo"
        (drive_root / "logs").mkdir(parents=True, exist_ok=True)
        repo_dir.mkdir(parents=True, exist_ok=True)
        eq = queue.Queue()

        bc = BackgroundConsciousness(
            drive_root=drive_root,
            repo_dir=repo_dir,
            event_queue=eq,
            owner_chat_id_fn=lambda: 42,
        )

        schema_names = {s.get("function", {}).get("name") for s in bc._tool_schemas()}
        self.assertIn("send_user_message", schema_names)
        self.assertIn("update_identity", schema_names)
        self.assertIn("recent_tasks", schema_names)
        self.assertNotIn("schedule_subagent", schema_names)
        self.assertNotIn("get_task_result", schema_names)
        self.assertNotIn("wait_task", schema_names)
        self.assertNotIn("wait_tasks", schema_names)
        self.assertNotIn("run_command", schema_names)
        self.assertNotIn("commit_reviewed", schema_names)


if __name__ == "__main__":
    unittest.main()
