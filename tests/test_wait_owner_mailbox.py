"""Child waits yield addressed input to the ordinary delivery/ack owner."""

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import acknowledged_task_message_ids, write_owner_message
from ouroboros.task_results import write_task_result
from ouroboros.task_status import wait_for_effective_tasks
from ouroboros.tools.control_task_results import _wait_attention_poll, _wait_for_task, _wait_for_tasks


@pytest.mark.parametrize("forked", [False, True])
@pytest.mark.parametrize("kind", ["owner_text", "quiz_answer", "task_message"])
@pytest.mark.parametrize("batch", [False, True])
def test_wait_yields_mailbox_without_acknowledging_or_stopping_child(tmp_path, forked, kind, batch):
    mailbox_root = tmp_path / "execution" if forked else tmp_path
    ctx = SimpleNamespace(drive_root=mailbox_root, task_id="parent", task_attempt=1,
                          task_metadata={"budget_drive_root": str(tmp_path), "root_task_id": "parent"},
                          _loop_mailbox_seen_ids=set())
    write_task_result(tmp_path, "child", "running", root_task_id="parent", parent_task_id="parent")
    assert write_owner_message(mailbox_root, "Use the blue version.", "parent", msg_id="answer", kind=kind)
    result = _wait_for_tasks(ctx, ["child"], timeout_sec=0) if batch else _wait_for_task(ctx, "child", timeout_sec=0)
    if batch:
        decoded = json.loads(result)
        assert decoded["early_return"]["reason"] == "owner_mailbox_pending"
        assert decoded["all_terminal"] is False
        assert decoded["tasks"]["child"]["status"] == "running"
    else:
        assert "unread message for this task" in result
        assert "child [running]" in result
    assert acknowledged_task_message_ids(mailbox_root, "parent", attempt_key=1) == set()
    assert ctx._loop_mailbox_seen_ids == set()
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), mailbox_root, "parent", queue.Queue(),
                             ctx._loop_mailbox_seen_ids, owner_ctx=ctx)
    assert "Use the blue version." in json.dumps(messages)
    assert "answer" in acknowledged_task_message_ids(mailbox_root, "parent", attempt_key=1)
    assert _wait_attention_poll(ctx, "", ["child"])({}, {}) is None
    assert write_owner_message(mailbox_root, "Also preserve the original.", "parent", msg_id="next")
    assert _wait_attention_poll(ctx, "", ["child"])({}, {})["reason"] == "owner_mailbox_pending"


def test_message_arriving_inside_the_existing_wait_interrupts_its_long_window(tmp_path):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="parent", task_attempt=1,
                          task_metadata={"root_task_id": "parent"}, _loop_mailbox_seen_ids=set())
    write_task_result(tmp_path, "child", "running", parent_task_id="parent", root_task_id="parent")
    hook = _wait_attention_poll(ctx, "", ["child"])
    first_poll = threading.Event()
    def observe(*args):
        result = hook(*args)
        first_poll.set()
        return result
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait_for_effective_tasks, tmp_path, ["child"], timeout_sec=120,
                                 poll_interval_sec=0.01, on_poll=observe)
        assert first_poll.wait(5)
        assert write_owner_message(tmp_path, "Continue with option B.", "parent", msg_id="during-wait")
        result = future.result(timeout=5)
    assert result["early_return"]["reason"] == "owner_mailbox_pending"
    assert result["all_terminal"] is False
    assert result["elapsed_sec"] < 5
    assert acknowledged_task_message_ids(tmp_path, "parent", attempt_key=1) == set()
