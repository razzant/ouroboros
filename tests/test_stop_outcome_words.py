"""The word an owner reads after a Stop outside the task card: the receipt row and the /evolve off reply.

The status code stays ``cancelled`` wherever it is typed (``settled_status``, intent outcomes, the
descendant digest); only the presentation prose says "stopped". Each pin has both directions: the cancel
callers speak the new word, while the statuses and callers that never said it stay as they were.
"""
from __future__ import annotations

import types

from ouroboros.task_results import load_task_result, write_task_result

from tests._cancel_intents_shared import _CaptureQueue


def test_owed_stop_receipt_reads_stopped_and_keeps_the_typed_status(tmp_path):
    from supervisor import cancel_publication as cancel
    from supervisor import terminal_delivery as td

    stored = write_task_result(tmp_path, "own-1", "cancelled", result="retained draft")
    q = types.SimpleNamespace(DRIVE_ROOT=tmp_path)
    assert cancel._register_owed_terminal_delivery(
        q, {"chat_id": 4}, "own-1", stored, deliver=True, salvage_text="retained draft")
    (owed,) = td.pending_deliveries(tmp_path)
    assert owed["text"].startswith("⚠️ Task own-1 was stopped. Below is the last persisted")
    assert owed["system_type"] == "cancel_receipt"
    block = load_task_result(tmp_path, "own-1")["cancel_receipt"]
    assert (block["settled_status"], block["outcome"]) == ("cancelled", "stopped")


def test_miss_lane_stop_receipt_says_stopped_and_other_statuses_stay_quiet(tmp_path):
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "miss-1", "cancelled", result="retained draft")
    queue = _CaptureQueue()
    assert td.deliver_miss_lane_outcome(
        tmp_path, tmp_path, {"task_id": "miss-1", "chat_id": 4}, "miss-1", "cancelled", event_queue=queue) is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert event["text"].startswith("⚠️ Task miss-1 was stopped.")
    assert load_task_result(tmp_path, "miss-1")["cancel_receipt"]["settled_status"] == "cancelled"

    # A failed task keeps its own path's message: this lane sends nothing for it.
    write_task_result(tmp_path, "miss-2", "failed", result="died")
    quiet = _CaptureQueue()
    assert td.deliver_miss_lane_outcome(
        tmp_path, tmp_path, {"task_id": "miss-2", "chat_id": 4}, "miss-2", "failed", event_queue=quiet) is True
    assert quiet.events == []


def test_cascade_root_outcome_says_stop_and_keeps_each_status_distinct():
    from supervisor.terminal_cascade import _cascade_root_outcome

    assert _cascade_root_outcome("cancelled") == _cascade_root_outcome("") == "stopped"
    assert _cascade_root_outcome("completed") == "completed before the stop (result preserved)"
    assert _cascade_root_outcome("failed") == "already settled as failed before the stop (result preserved)"
    assert not any("cancel" in _cascade_root_outcome(status)
                   for status in ("cancelled", "", "completed", "failed", "interrupted"))


def test_evolve_off_reply_names_the_stopped_tasks_and_only_those():
    import supervisor.queue as q

    base = {"cancelled": [], "already_settled": [], "not_found": [], "failed": [], "intent_write_failed": []}
    lines, incomplete = q.evolution_stop_report({**base, "cancelled": ["e1", "e2"], "already_settled": ["e3"]})
    assert lines[0] == "🛑 Stopped evolution task(s): e1, e2" and incomplete is False
    assert not any("Cancelled evolution" in line for line in lines)

    # Nothing stopped by this call: no stop line, and a task still live makes the stop incomplete.
    lines, incomplete = q.evolution_stop_report({**base, "already_settled": ["e3"], "failed": ["e4"]})
    assert not any(line.startswith("🛑") for line in lines) and incomplete is True
