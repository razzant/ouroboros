"""
Tests for v6 message routing: single-consumer delivery,
per-task mailbox, and forward_to_worker tool.

Run: pytest tests/test_message_routing.py -v
"""

import json
import pathlib
import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestOwnerInjectPerTask(unittest.TestCase):
    """Test per-task mailbox in owner_mailbox.py."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.drive_root = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_write_creates_per_task_file(self):
        from ouroboros.owner_mailbox import write_owner_message, _mailbox_path
        write_owner_message(self.drive_root, "hello", task_id="abc123", msg_id="m1")
        path = _mailbox_path(self.drive_root, "abc123")
        self.assertTrue(path.exists())
        content = path.read_text()
        entry = json.loads(content.strip())
        self.assertEqual(entry["text"], "hello")
        self.assertEqual(entry["msg_id"], "m1")

    def test_drain_reads_only_own_task(self):
        from ouroboros.owner_mailbox import write_owner_message, drain_owner_messages
        write_owner_message(self.drive_root, "for task A", task_id="taskA", msg_id="m1")
        write_owner_message(self.drive_root, "for task B", task_id="taskB", msg_id="m2")

        msgs_a = drain_owner_messages(self.drive_root, task_id="taskA")
        msgs_b = drain_owner_messages(self.drive_root, task_id="taskB")

        self.assertEqual(msgs_a, ["for task A"])
        self.assertEqual(msgs_b, ["for task B"])

    def test_drain_dedup_with_seen_ids(self):
        from ouroboros.owner_mailbox import write_owner_message, drain_owner_messages
        write_owner_message(self.drive_root, "msg1", task_id="t1", msg_id="id1")
        write_owner_message(self.drive_root, "msg2", task_id="t1", msg_id="id2")

        seen = set()
        first_read = drain_owner_messages(self.drive_root, task_id="t1", seen_ids=seen)
        self.assertEqual(len(first_read), 2)
        self.assertEqual(seen, {"id1", "id2"})

        write_owner_message(self.drive_root, "msg3", task_id="t1", msg_id="id3")
        second_read = drain_owner_messages(self.drive_root, task_id="t1", seen_ids=seen)
        self.assertEqual(second_read, ["msg3"])
        self.assertIn("id3", seen)

    def test_control_entries_are_typed_and_hidden_from_text_view(self):
        """finalize_now controls surface via drain_owner_entries with their
        kind; the legacy text-only view returns owner dialogue only."""
        from ouroboros.owner_mailbox import (
            KIND_FINALIZE_NOW,
            drain_owner_entries,
            drain_owner_messages,
            write_owner_message,
        )
        write_owner_message(self.drive_root, "real owner text", task_id="t9", msg_id="m1")
        write_owner_message(self.drive_root, "deadline", task_id="t9", msg_id="c1", kind=KIND_FINALIZE_NOW)

        entries = drain_owner_entries(self.drive_root, task_id="t9")
        kinds = {e["msg_id"]: e["kind"] for e in entries}
        self.assertEqual(kinds["m1"], "owner_text")
        self.assertEqual(kinds["c1"], "finalize_now")

        texts = drain_owner_messages(self.drive_root, task_id="t9")
        self.assertEqual(texts, ["real owner text"])

    def test_loop_drain_routes_finalize_now_control(self):
        """The loop drain returns the typed control instead of injecting it
        as owner prose."""
        import queue as _q
        from ouroboros.loop import _drain_incoming_messages
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

        write_owner_message(self.drive_root, "hard_timeout", task_id="t10", kind=KIND_FINALIZE_NOW)
        write_owner_message(self.drive_root, "keep going please", task_id="t10")
        messages = []
        controls = _drain_incoming_messages(
            messages, _q.Queue(), self.drive_root, "t10", None, set()
        )
        self.assertEqual(controls, {"finalize_now": "hard_timeout"})
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertIn("keep going please", joined)
        self.assertNotIn("hard_timeout", joined)

    def test_cleanup_removes_file(self):
        from ouroboros.owner_mailbox import write_owner_message, cleanup_task_mailbox, _mailbox_path
        write_owner_message(self.drive_root, "hello", task_id="t1", msg_id="m1")
        path = _mailbox_path(self.drive_root, "t1")
        self.assertTrue(path.exists())

        cleanup_task_mailbox(self.drive_root, "t1")
        self.assertFalse(path.exists())

    def test_drain_nonexistent_task_returns_empty(self):
        from ouroboros.owner_mailbox import drain_owner_messages
        msgs = drain_owner_messages(self.drive_root, task_id="nonexistent")
        self.assertEqual(msgs, [])

    def test_messages_not_cleared_on_read(self):
        """Messages persist after read (append-only). Only cleanup removes them."""
        from ouroboros.owner_mailbox import write_owner_message, drain_owner_messages, _mailbox_path
        write_owner_message(self.drive_root, "persistent", task_id="t1", msg_id="m1")

        drain_owner_messages(self.drive_root, task_id="t1")

        path = _mailbox_path(self.drive_root, "t1")
        self.assertTrue(path.exists())
        self.assertIn("persistent", path.read_text())

    def test_mailbox_rejects_unsafe_task_id(self):
        from ouroboros.owner_mailbox import _mailbox_path

        with self.assertRaises(ValueError):
            _mailbox_path(self.drive_root, "../settings")


class TestForwardToWorkerTool(unittest.TestCase):
    """Test that forward_to_worker tool is registered."""

    def test_forward_to_worker_routes_to_child_drive_and_rejects_non_running(self):
        from types import SimpleNamespace
        from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
        from ouroboros.tools.core import _forward_to_worker

        with tempfile.TemporaryDirectory() as tmp:
            parent_drive = pathlib.Path(tmp) / "parent"
            child_drive = pathlib.Path(tmp) / "child"
            child_drive.mkdir(parents=True)
            write_task_result(parent_drive, "child1", STATUS_RUNNING, child_drive_root=str(child_drive), parent_task_id="parent1", root_task_id="parent1", result="running")
            write_task_result(parent_drive, "queued1", STATUS_SCHEDULED, result="queued")
            write_task_result(parent_drive, "otherchild", STATUS_RUNNING, parent_task_id="otherparent", root_task_id="otherroot", result="running")
            ctx = SimpleNamespace(drive_root=parent_drive, task_id="parent1")

            output = _forward_to_worker(ctx, "child1", "continue")
            blocked = _forward_to_worker(ctx, "queued1", "too soon")
            forbidden = _forward_to_worker(ctx, "otherchild", "wrong root")

            self.assertIn("Message forwarded", output)
            self.assertIn("TASK_NOT_ACTIVE", blocked)
            self.assertIn("TASK_FORBIDDEN", forbidden)
            self.assertFalse((parent_drive / "memory" / "owner_mailbox" / "child1.jsonl").exists())
            mailbox = child_drive / "memory" / "owner_mailbox" / "child1.jsonl"
            self.assertTrue(mailbox.exists())
            self.assertIn("continue", mailbox.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
