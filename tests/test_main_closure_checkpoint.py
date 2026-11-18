"""An owner closure must survive a crash between mailbox drain and ready save."""
import asyncio
import queue

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.gateway.task_decision import answer_decision
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.owner_wait import restore_continuation_state
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_quiz_history_evidence import _ask, runtime as runtime


@pytest.mark.parametrize("authored", [False, True])
def test_legacy_closure_checkpoint_reprojects_only_the_typed_owner_reason(runtime, authored):
    from copy import deepcopy
    from ouroboros.review_evidence_sections import _accept_owner_directives

    root, tid = runtime.root, runtime.task["id"]
    write_task_result(root, tid, "running")
    _ask(runtime, "retired")
    reason = "The owner withdrew the planned output" if authored else ""
    status, _receipt = asyncio.run(answer_decision(root, {
        "request_id": "close-legacy", "decision_id": f"quiz:{tid}:retired", "action": "close", "reason": reason,
    }))
    assert status == 200
    [letter] = drain_owner_entries(root, tid)
    ctx, limit = _loop_ctx(root, tid)
    limit.messages[:] = [{"role": "user", "content": letter["text"]}]
    ctx._owner_directives = [{"source": "owner_mailbox", "content": letter["text"], "msg_id": letter["msg_id"]}]
    wc.save_ready(limit)
    path = wc.checkpoint_path(root, tid, 1)
    original_bytes = path.read_bytes()
    _, state = wc.read_working(root, tid, 1)
    old_directives = deepcopy(state["owner_directives"])
    resumed, restored = _loop_ctx(root, tid)
    restore_continuation_state(restored.tools, state, restored.messages, restored.llm_trace,
                               restored.accumulated_usage, restored.owner_msg_seen)
    assert state["owner_directives"] == old_directives and path.read_bytes() == original_bytes
    assert restored.messages[0]["content"] == letter["text"]
    rows = _accept_owner_directives(resumed, root, tid)
    assert [row["content"] for row in rows if row.get("msg_id") == letter["msg_id"]] == ([reason] if authored else [])


@pytest.mark.parametrize("failed_save", [False, True], ids=["crash-before-save", "failed-save"])
def test_owner_close_replays_until_its_reason_is_checkpointed(runtime, monkeypatch, failed_save):
    root, tid = runtime.root, runtime.task["id"]
    write_task_result(root, tid, "running")
    ctx, limit = _loop_ctx(root, tid)
    limit.messages = [{"role": "user", "content": "Work awaiting a decision"}]
    wc.save_ready(limit)
    before = wc.checkpoint_path(root, tid, 1).read_bytes()

    _ask(runtime, "retired")
    reason = "The owner withdrew this planned output"
    status, receipt = asyncio.run(answer_decision(root, {
        "request_id": "close-once", "decision_id": f"quiz:{tid}:retired",
        "action": "close", "reason": reason,
    }))
    assert status == 200 and receipt["state"] == "closed", receipt
    assert "answered_index" not in receipt

    def drain(context, working):
        _drain_incoming_messages(working.messages, queue.Queue(), root, tid, None,
                                 working.owner_msg_seen, owner_ctx=context, defer_content_ack=True)

    def restore():
        context, working = _loop_ctx(root, tid)
        _, saved = wc.read_working(root, tid, 1)
        working.owner_msg_seen = set()
        restore_continuation_state(working.tools, saved, working.messages, working.llm_trace,
                                   working.accumulated_usage, working.owner_msg_seen)
        return context, working

    drain(ctx, limit)
    assert reason in str(limit.messages) and "no approval was given" in str(limit.messages)
    if failed_save:
        from ouroboros import utils
        original = utils.write_bytes_atomic

        def fail_checkpoint(path, data, **kwargs):
            if path.name.startswith(wc.FILE_PREFIX):
                raise OSError("injected checkpoint failure")
            return original(path, data, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(utils, "write_bytes_atomic", fail_checkpoint)
            wc.save_ready(limit)
    assert wc.checkpoint_path(root, tid, 1).read_bytes() == before
    # Discard all in-memory messages, seen ids and ACK callbacks as a dead worker does.
    resumed, resumed_limit = restore()
    assert reason not in str(resumed_limit.messages)
    assert [row["msg_id"] for row in drain_owner_entries(root, tid, set(), 1)] == ["quiz_closed:retired"]
    drain(resumed, resumed_limit)
    assert str(resumed_limit.messages).count(reason) == 1
    assert reason in str(resumed._owner_directives)
    wc.save_ready(resumed_limit)
    _, saved = wc.read_working(root, tid, 1)
    assert "quiz_closed:retired" in saved["seen"] and reason in str(saved["messages"])
    assert reason in str(saved["owner_directives"])
    assert drain_owner_entries(root, tid, set(), 1) == []
    resumed, resumed_limit = restore()
    drain(resumed, resumed_limit)
    assert str(resumed_limit.messages).count(reason) == 1, "saved closure is not injected twice"
