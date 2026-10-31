"""
Tests for v6 message routing: single-consumer delivery,
per-task mailbox, and forward_to_worker tool.

Run: pytest tests/test_message_routing.py -v
"""

import contextlib
import json
import pathlib
import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests._store_shared import release_store_handles  # noqa: E402


@contextlib.contextmanager
def _temporary_drives():
    """A temporary folder for the test's drives, removed after the store lets go of it.

    The store keeps this thread's connection to each drive's ``state/current.sqlite`` open
    (#1982) and Windows refuses to delete an open file (WinError 32), so the handles under the
    folder are released first, as the product does before it moves or removes a drive.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            yield tmp
        finally:
            release_store_handles(tmp)


class TestOwnerInjectPerTask(unittest.TestCase):
    """Test per-task mailbox in owner_mailbox.py."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.drive_root = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        release_store_handles(self.drive_root)  # see _temporary_drives
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
        from ouroboros import task_pacing
        from ouroboros.deadline_utils import parse_deadline_ts
        from ouroboros.loop import _drain_incoming_messages
        from ouroboros.owner_mailbox import (
            KIND_FINALIZE_NOW,
            drain_owner_entries,
            write_owner_message,
        )

        write_owner_message(
            self.drive_root, "hard_timeout", task_id="t10",
            msg_id="finalize-1", kind=KIND_FINALIZE_NOW,
        )
        write_owner_message(self.drive_root, "keep going please", task_id="t10")
        control_entry = next(
            row for row in drain_owner_entries(self.drive_root, "t10")
            if row["msg_id"] == "finalize-1"
        )
        expected_deadline = (
            parse_deadline_ts(control_entry["ts"]).timestamp()
            + task_pacing.effective_finalization_reserve_sec(None)
        )
        messages = []
        controls = _drain_incoming_messages(
            messages, _q.Queue(), self.drive_root, "t10", None, set()
        )
        self.assertEqual(set(controls), {"finalize_now", "finalize_deadline_ts"})
        self.assertEqual(controls["finalize_now"], "hard_timeout")
        self.assertEqual(controls["finalize_deadline_ts"], expected_deadline)
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertIn("keep going please", joined)
        self.assertNotIn("hard_timeout", joined)

    def test_cleanup_removes_file(self):
        from ouroboros.owner_mailbox import write_owner_message, cleanup_task_mailbox, _mailbox_path
        from ouroboros.task_results import load_task_result, write_task_result
        write_owner_message(self.drive_root, "hello", task_id="t1", msg_id="m1")
        path = _mailbox_path(self.drive_root, "t1")
        self.assertTrue(path.exists())

        # TZ-1 V10: an unread row leaves only into a canonical row that holds it.
        self.assertFalse(cleanup_task_mailbox(self.drive_root, "t1"))
        self.assertTrue(path.exists())
        write_task_result(self.drive_root, "t1", "cancelled", result="Cancelled before start.")
        self.assertIn('"msg_id": "m1"', load_task_result(self.drive_root, "t1")["unread_mailbox"]["rows"][0])
        self.assertTrue(cleanup_task_mailbox(self.drive_root, "t1"))
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

    def test_retry_keeps_task_message_global_ack_semantics(self):
        """Phase 1C changes owner-text retry replay, not ancestor task messages."""
        from ouroboros.owner_hurry import retry_reset
        from ouroboros.owner_mailbox import (
            acknowledge_transcript_entry,
            drain_owner_entries,
            write_task_message,
        )

        self.assertTrue(write_task_message(
            self.drive_root,
            "child instruction",
            "child-task",
            source_task_id="parent-task",
            msg_id="task-msg-1",
        ))
        entry = drain_owner_entries(
            self.drive_root, "child-task", attempt_key=1,
        )[0]
        acknowledge_transcript_entry(self.drive_root, "child-task", entry)
        retry_reset(
            self.drive_root, self.drive_root, "child-task", reason="worker_crash_requeue",
        )

        self.assertEqual(
            drain_owner_entries(self.drive_root, "child-task", attempt_key=2),
            [],
        )

    def test_legacy_ack_is_attempt_local_for_owner_but_global_for_task_message(self):
        from ouroboros.owner_mailbox import (
            acknowledge_task_messages,
            acknowledged_task_message_ids,
            drain_owner_entries,
            write_owner_message,
            write_task_message,
        )

        self.assertTrue(write_owner_message(
            self.drive_root, "exact legacy owner bytes", "legacy-mixed",
            msg_id="owner-legacy",
        ))
        self.assertTrue(write_task_message(
            self.drive_root, "ancestor bytes", "legacy-mixed",
            source_task_id="parent", msg_id="task-legacy",
        ))
        self.assertTrue(acknowledge_task_messages(
            self.drive_root, "legacy-mixed", ["owner-legacy", "task-legacy"],
            wake_id="legacy-consumer",
        ))

        self.assertEqual(
            acknowledged_task_message_ids(
                self.drive_root, "legacy-mixed", attempt_key=2,
            ),
            {"task-legacy"},
        )
        replay = drain_owner_entries(
            self.drive_root, "legacy-mixed", attempt_key=2,
        )
        self.assertEqual(
            [(row["msg_id"], row["kind"], row["text"]) for row in replay],
            [("owner-legacy", "owner_text", "exact legacy owner bytes")],
        )

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

        with _temporary_drives() as tmp:
            parent_drive = pathlib.Path(tmp) / "parent"
            child_drive = pathlib.Path(tmp) / "child"
            queued_drive = pathlib.Path(tmp) / "queued-child"
            child_drive.mkdir(parents=True)
            write_task_result(parent_drive, "child1", STATUS_RUNNING, child_drive_root=str(child_drive), parent_task_id="parent1", root_task_id="parent1", result="running")
            write_task_result(parent_drive, "queued1", STATUS_SCHEDULED, child_drive_root=str(queued_drive), parent_task_id="parent1", root_task_id="parent1", result="queued")
            asked_drive = pathlib.Path(tmp) / "asked-child"
            write_task_result(parent_drive, "asked1", "requested", child_drive_root=str(asked_drive), parent_task_id="parent1", root_task_id="parent1", result="requested")
            write_task_result(parent_drive, "ended1", "completed", child_drive_root=str(asked_drive), parent_task_id="parent1", root_task_id="parent1", result="done")
            write_task_result(parent_drive, "otherchild", STATUS_RUNNING, parent_task_id="otherparent", root_task_id="otherroot", result="running")
            ctx = SimpleNamespace(drive_root=parent_drive, task_id="parent1")

            output = _forward_to_worker(ctx, "child1", "continue")
            queued = _forward_to_worker(ctx, "queued1", "read this when you start")
            requested = _forward_to_worker(ctx, "asked1", "read this once admitted")
            blocked = _forward_to_worker(ctx, "ended1", "too late")
            forbidden = _forward_to_worker(ctx, "otherchild", "wrong root")

            self.assertIn("Message forwarded", output)
            # TZ-1 V10: a queued task's mailbox takes the message; the receipt says nothing read it.
            self.assertIn("(queued)", queued)
            self.assertIn("has not started, so nothing has read it", queued)
            queued_mailbox = queued_drive / "memory" / "owner_mailbox" / "queued1.jsonl"
            self.assertIn("read this when you start", queued_mailbox.read_text(encoding="utf-8"))
            # A requested child's mailbox already exists in its recorded drive: the message waits
            # there for its start (never dropped); a settled task still refuses.
            self.assertIn("(queued)", requested)
            asked_mailbox = asked_drive / "memory" / "owner_mailbox" / "asked1.jsonl"
            self.assertIn("read this once admitted", asked_mailbox.read_text(encoding="utf-8"))
            self.assertIn("TASK_NOT_ACTIVE", blocked)
            self.assertFalse((asked_drive / "memory" / "owner_mailbox" / "ended1.jsonl").exists())
            self.assertIn("TASK_FORBIDDEN", forbidden)
            self.assertFalse((parent_drive / "memory" / "owner_mailbox" / "child1.jsonl").exists())
            mailbox = child_drive / "memory" / "owner_mailbox" / "child1.jsonl"
            self.assertTrue(mailbox.exists())
            self.assertIn("continue", mailbox.read_text(encoding="utf-8"))

    def test_forward_to_worker_refuses_while_cancellation_is_pending(self):
        """AR2-6: the effective status honestly stays ``running`` while a durable
        cancel intent is open, so the status checks pass — the steering write is
        refused typed instead of feeding a task mid-teardown."""
        from types import SimpleNamespace
        from ouroboros.cancel_intents import request_cancel
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            parent_drive = pathlib.Path(tmp) / "parent"
            write_task_result(
                parent_drive, "child2", STATUS_RUNNING,
                parent_task_id="parent1", root_task_id="parent1", result="running",
            )
            request_cancel(parent_drive, "child2", reason="tearing down")
            ctx = SimpleNamespace(drive_root=parent_drive, task_id="parent1")

            refused = _forward_to_worker(ctx, "child2", "keep going")

            self.assertIn("TASK_CANCEL_PENDING", refused)
            self.assertFalse(
                (parent_drive / "memory" / "owner_mailbox" / "child2.jsonl").exists(),
            )

    # --- serial addressed turns: peer contributions under the same writer -------

    @staticmethod
    def _peer_tree(tmp):
        """root1 ─┬─ me (the caller) ─ kid          canonical = the status drive
                  ├─ sib (drive recorded)           sib_drive = sib's own drive
                  └─ sib2 ─ nephew                  nephew: a sibling's child, no relation
           other-root: stranger (parent_task_id=root1 but root_task_id=other)"""
        from ouroboros.task_results import STATUS_RUNNING, write_task_result

        canonical = pathlib.Path(tmp) / "canonical"
        sib_drive = pathlib.Path(tmp) / "sib-drive"
        sib_drive.mkdir(parents=True)
        write_task_result(canonical, "root1", STATUS_RUNNING, root_task_id="root1", result="running")
        write_task_result(canonical, "me", STATUS_RUNNING, parent_task_id="root1", root_task_id="root1", result="running")
        write_task_result(canonical, "kid", STATUS_RUNNING, parent_task_id="me", root_task_id="root1", result="running")
        write_task_result(canonical, "sib", STATUS_RUNNING, parent_task_id="root1", root_task_id="root1",
                          child_drive_root=str(sib_drive), result="running")
        write_task_result(canonical, "sib2", STATUS_RUNNING, parent_task_id="root1", root_task_id="root1", result="running")
        write_task_result(canonical, "nephew", STATUS_RUNNING, parent_task_id="sib2", root_task_id="root1", result="running")
        write_task_result(canonical, "stranger", STATUS_RUNNING, parent_task_id="root1", root_task_id="other", result="running")
        return canonical, sib_drive

    @staticmethod
    def _caller(canonical, *, drive_root=None, metadata=True):
        from types import SimpleNamespace

        ctx = SimpleNamespace(drive_root=drive_root or canonical, task_id="me")
        if metadata:
            ctx.task_metadata = {"parent_task_id": "root1", "root_task_id": "root1",
                                 "budget_drive_root": str(canonical)}
        else:
            ctx.budget_drive_root = str(canonical)
        return ctx

    def test_sibling_contribution_is_peer_task_with_relation_sibling(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            out = _forward_to_worker(self._caller(canonical), "sib", "my objection to your draft")

            self.assertTrue(out.startswith("Message forwarded to task sib: written to its mailbox as a message from a peer task"))
            self.assertIn("your sibling", out)
            self.assertIn("never owner text", out)
            [row] = drain_owner_entries(sib_drive, "sib")
            self.assertEqual(
                (row["kind"], row["provenance"], row["relation"], row["source_task_id"], row["text"]),
                ("task_message", "peer_task", "sibling", "me", "my objection to your draft"),
            )
            self.assertEqual(row["relayed_from_task_id"], "")
            self.assertFalse((canonical / "memory" / "owner_mailbox" / "sib.jsonl").exists())

    def test_child_to_parent_contribution_is_peer_task_with_relation_parent(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            out = _forward_to_worker(self._caller(canonical), "root1", "interim position, not my final")

            self.assertIn("message from a peer task", out)
            self.assertIn("your parent", out)
            [row] = drain_owner_entries(canonical, "root1")
            self.assertEqual((row["provenance"], row["relation"], row["source_task_id"]),
                             ("peer_task", "parent", "me"))
            # The descendant rule is untouched: the caller's own child still gets ancestor steering.
            self.assertEqual(_forward_to_worker(self._caller(canonical), "kid", "steer"), "Message forwarded to task kid")
            [steer] = drain_owner_entries(canonical, "kid")
            self.assertEqual(steer["provenance"], "ancestor_task")
            self.assertNotIn("relation", steer)

    def test_self_is_not_a_sibling(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            out = _forward_to_worker(self._caller(canonical), "me", "not a peer")
            self.assertIn("TASK_FORBIDDEN", out)
            self.assertEqual(drain_owner_entries(canonical, "me"), [])

    def test_direct_chat_parent_without_root_carrier_is_still_parent(self):
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            write_task_result(canonical, "root1", STATUS_RUNNING,
                              root_task_id=None, _is_direct_chat=True)
            out = _forward_to_worker(self._caller(canonical), "root1", "direct parent original")
            self.assertIn("message from a peer task", out)
            [row] = drain_owner_entries(canonical, "root1")
            self.assertEqual((row["provenance"], row["relation"]), ("peer_task", "parent"))

    def test_caller_lineage_fallback_with_missing_metadata(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            ctx = self._caller(canonical, metadata=False)  # no task_metadata at all

            out = _forward_to_worker(ctx, "sib", "lineage from my own task result")

            self.assertIn("message from a peer task", out)
            [row] = drain_owner_entries(sib_drive, "sib")
            self.assertEqual(row["relation"], "sibling")

    def test_a_siblings_child_is_a_tree_peer_and_another_root_stays_forbidden(self):
        """A nephew shares the caller's root: admitted as context-only ``peer_task`` with
        the stamped relation ``tree`` (the blackboard's own scope). The same parent id
        under another root is still no peer at all."""
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            ctx = self._caller(canonical)

            nephew = _forward_to_worker(ctx, "nephew", "a sibling's child shares my tree")
            stranger = _forward_to_worker(ctx, "stranger", "same parent id, other root")
            sibling = _forward_to_worker(ctx, "sib2", "same parent, same root")

            self.assertIn("message from a peer task", nephew)
            self.assertIn("a task in your tree", nephew)
            self.assertNotIn("independent", nephew)
            [row] = drain_owner_entries(canonical, "nephew")
            self.assertEqual((row["provenance"], row["relation"], row["source_task_id"]),
                             ("peer_task", "tree", "me"))
            self.assertIn("TASK_FORBIDDEN", stranger)
            self.assertIn("the parent nor a sibling", stranger)
            self.assertIn("sharing its root", stranger)
            self.assertIn("nor an active independent root", stranger)
            self.assertEqual(drain_owner_entries(canonical, "stranger"), [])
            self.assertIn("your sibling", sibling)

    @staticmethod
    def _grandchild(canonical):
        from types import SimpleNamespace

        return SimpleNamespace(drive_root=canonical, task_id="kid", task_metadata={
            "parent_task_id": "me", "root_task_id": "root1", "budget_drive_root": str(canonical)})

    def test_grandchild_reaches_its_root_as_a_tree_peer_under_the_strict_cancel_read(self):
        """On a live install a grandchild's message to its own root rode the host-listed-roots
        path as ``independent_task`` (mislabelled, fail-soft cancel read). It is a tree peer:
        context-only, labelled, and refused typed when the recipient's cancel state is unreadable."""
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            out = _forward_to_worker(self._grandchild(canonical), "root1", "a grandchild's interim position")

            self.assertIn("message from a peer task", out)
            self.assertIn("a task in your tree", out)
            self.assertNotIn("independent", out)
            [row] = drain_owner_entries(canonical, "root1")
            self.assertEqual((row["provenance"], row["relation"], row["source_task_id"]),
                             ("peer_task", "tree", "kid"))

            projection = canonical / "state" / "cancel_intents.json"
            projection.parent.mkdir(parents=True, exist_ok=True)
            projection.write_bytes(b'{"intents": [broken')
            refused = _forward_to_worker(self._grandchild(canonical), "root1", "second position")
            self.assertIn("TASK_CANCEL_STATE_UNAVAILABLE", refused)
            self.assertIn("NOT written", refused)
            self.assertNotIn("second position", [r["text"] for r in drain_owner_entries(canonical, "root1")])

    def test_continuation_root_without_a_parent_reaches_its_predecessors_child(self):
        """A continuation root carries its predecessor's root label and has no parent (the
        one refusal observed live, 2026-09-28): same tree, admitted as ``tree``; another
        root stays refused."""
        from types import SimpleNamespace
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            write_task_result(canonical, "cont", STATUS_RUNNING, root_task_id="root1", result="running")
            ctx = SimpleNamespace(drive_root=canonical, task_id="cont",
                                  task_metadata={"root_task_id": "root1", "budget_drive_root": str(canonical)})

            out = _forward_to_worker(ctx, "sib", "the predecessor's child hears its successor")
            stranger = _forward_to_worker(ctx, "stranger", "other root")

            self.assertIn("a task in your tree", out)
            [row] = drain_owner_entries(sib_drive, "sib")
            self.assertEqual((row["provenance"], row["relation"], row["source_task_id"]),
                             ("peer_task", "tree", "cont"))
            self.assertIn("TASK_FORBIDDEN", stranger)
            self.assertEqual(drain_owner_entries(canonical, "stranger"), [])

    def test_grandchild_reaches_a_root_without_a_root_carrier_but_no_sibling_root_is_inferred(self):
        from types import SimpleNamespace
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical = pathlib.Path(tmp) / "canonical"
            write_task_result(canonical, "root0", STATUS_RUNNING, root_task_id=None, _is_direct_chat=True, result="running")
            write_task_result(canonical, "c0", STATUS_RUNNING, parent_task_id="root0", root_task_id="root0", result="running")
            write_task_result(canonical, "g0", STATUS_RUNNING, parent_task_id="c0", root_task_id="root0", result="running")
            # A sibling of c0 whose root carrier is missing: its root is never inferred.
            write_task_result(canonical, "s0", STATUS_RUNNING, parent_task_id="root0", result="running")
            ctx = SimpleNamespace(drive_root=canonical, task_id="g0", task_metadata={
                "parent_task_id": "c0", "root_task_id": "root0", "budget_drive_root": str(canonical)})

            to_root = _forward_to_worker(ctx, "root0", "up two levels")
            to_uncarried = _forward_to_worker(ctx, "s0", "a sibling root is never inferred")

            self.assertIn("a task in your tree", to_root)
            [row] = drain_owner_entries(canonical, "root0")
            self.assertEqual((row["provenance"], row["relation"]), ("peer_task", "tree"))
            self.assertIn("TASK_FORBIDDEN", to_uncarried)
            self.assertEqual(drain_owner_entries(canonical, "s0"), [])

    def test_relay_to_a_tree_peer_is_an_ancestor_only_act(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            refused = _forward_to_worker(self._caller(canonical), "nephew", "relayed", relayed_from_task_id="kid")

            self.assertIn("TASK_FORBIDDEN", refused)
            self.assertIn("ancestor-only act", refused)
            self.assertIn("a task in your tree", refused)
            self.assertEqual(drain_owner_entries(canonical, "nephew"), [])

    def test_a_presence_root_reached_from_its_tree_keeps_the_execution_observation(self):
        """The roster path put the Presence root's shared observation on the receipt; a
        sender inside its tree keeps it (persistence proof, not a read). An ordinary peer
        receipt carries none."""
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            # The observation is derived by the effective reader from the Presence record
            # (a direct-chat row whose metadata carries the presence binding), never stored.
            write_task_result(canonical, "root1", STATUS_RUNNING, root_task_id="root1", result="running",
                              _is_direct_chat=True, metadata={"source": "presence", "presence": {"room": "r"}})

            observed = _forward_to_worker(self._grandchild(canonical), "root1", "to the presence root")
            plain = _forward_to_worker(self._caller(canonical), "sib", "an ordinary peer")

            self.assertIn("a task in your tree", observed)
            self.assertIn("execution_observation=", observed)
            self.assertIn('"kind": "presence"', observed)
            # The roster path's conditional wording survives: a Presence turn may be over.
            self.assertIn("if the turn continues, its checkpoint can read it", observed)
            self.assertNotIn("reads it at its next checkpoint", observed)
            self.assertNotIn("execution_observation", plain)
            self.assertIn("reads it at its next checkpoint", plain)

    def test_relay_on_the_peer_branch_is_forbidden(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            ctx = self._caller(canonical)

            to_sibling = _forward_to_worker(ctx, "sib", "relayed", relayed_from_task_id="kid")
            to_parent = _forward_to_worker(ctx, "root1", "relayed", relayed_from_task_id="kid")

            for refused in (to_sibling, to_parent):
                self.assertIn("TASK_FORBIDDEN", refused)
                self.assertIn("ancestor-only act", refused)
            self.assertEqual(drain_owner_entries(sib_drive, "sib"), [])
            self.assertEqual(drain_owner_entries(canonical, "root1"), [])

    def test_recipient_without_recorded_drive_is_written_under_the_canonical_root_never_the_sender_drive(self):
        """A forked sender's ``ctx.drive_root`` is its private execution drive;
        the recipient (here the parent, which records no child_drive_root) drains
        the canonical status root, so that is where the message must land."""
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, _ = self._peer_tree(tmp)
            sender_drive = pathlib.Path(tmp) / "me-execution"
            sender_drive.mkdir()
            ctx = self._caller(canonical, drive_root=sender_drive)

            out = _forward_to_worker(ctx, "root1", "written where you read")
            self.assertIn("message from a peer task", out)
            # And the descendant branch obeys the same rule for a child without a recorded drive.
            self.assertEqual(_forward_to_worker(ctx, "kid", "ancestor steering"), "Message forwarded to task kid")

            self.assertEqual([row["text"] for row in drain_owner_entries(canonical, "root1")], ["written where you read"])
            self.assertEqual([row["text"] for row in drain_owner_entries(canonical, "kid")], ["ancestor steering"])
            self.assertFalse((sender_drive / "memory").exists())

    def test_message_over_8000_chars_is_refused_whole(self):
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            ctx = self._caller(canonical)
            self.assertEqual(TASK_MESSAGE_MAX_CHARS, 8000)

            refused = _forward_to_worker(ctx, "sib", "x" * 8001)
            accepted = _forward_to_worker(ctx, "sib", "y" * 8000)

            self.assertTrue(refused.startswith("⚠️ TOOL_ARG_ERROR (forward_to_worker)"))
            self.assertIn("8000", refused)
            self.assertIn("never truncated", refused)
            self.assertIn("message from a peer task", accepted)
            [row] = drain_owner_entries(sib_drive, "sib")
            self.assertEqual(len(row["text"]), 8000)


    def test_peer_contribution_refuses_when_cancellation_state_is_unreadable(self):
        """A peer holds no authority over the recipient: with the recipient's cancel
        projection torn, the peer write is refused typed and nothing is written,
        while an ancestor's steering to its own child keeps the existing fail-soft
        path (written and reported as written, not read)."""
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            projection = canonical / "state" / "cancel_intents.json"
            projection.parent.mkdir(parents=True, exist_ok=True)
            projection.write_bytes(b'{"intents": [broken')
            ctx = self._caller(canonical)

            to_sibling = _forward_to_worker(ctx, "sib", "interim position")
            to_parent = _forward_to_worker(ctx, "root1", "interim position")
            to_child = _forward_to_worker(ctx, "kid", "ancestor steering")

            for refused in (to_sibling, to_parent):
                self.assertIn("TASK_CANCEL_STATE_UNAVAILABLE", refused)
                self.assertIn("NOT written", refused)
            self.assertEqual(drain_owner_entries(sib_drive, "sib"), [])
            self.assertEqual(drain_owner_entries(canonical, "root1"), [])
            self.assertEqual(to_child, "Message forwarded to task kid")
            self.assertEqual([row["text"] for row in drain_owner_entries(canonical, "kid")], ["ancestor steering"])

    def test_peer_contribution_is_refused_while_the_recipient_cancels(self):
        from ouroboros.cancel_intents import request_cancel
        from ouroboros.owner_mailbox import drain_owner_entries
        from ouroboros.tools.core import _forward_to_worker

        with _temporary_drives() as tmp:
            canonical, sib_drive = self._peer_tree(tmp)
            request_cancel(canonical, "sib", reason="tearing down")

            refused = _forward_to_worker(self._caller(canonical), "sib", "interim position")

            self.assertIn("TASK_CANCEL_PENDING", refused)
            self.assertEqual(drain_owner_entries(sib_drive, "sib"), [])


if __name__ == "__main__":
    unittest.main()
