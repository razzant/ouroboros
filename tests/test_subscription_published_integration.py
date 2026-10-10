"""Published stop/peek contracts coexist with task-local subscription controls."""

from types import SimpleNamespace

from ouroboros import loop, loop_model_call, owner_mailbox as mailbox


def test_proven_empty_conversation_peek_leaves_model_wait_for_its_own_reader(tmp_path):
    mailbox.write_owner_message(tmp_path, "wait choice", "task", msg_id="wait", kind=mailbox.KIND_MODEL_WAIT)
    peek = mailbox.OwnerMailboxPeek()
    seen = set()
    assert not peek.pending(tmp_path, "task", seen, 1)
    assert not peek.pending(tmp_path, "task", seen, 1)
    status = {}
    entries = mailbox.drain_owner_entries(tmp_path, "task", seen, 1,
                                         _read_status=status, kinds={mailbox.KIND_MODEL_WAIT})
    assert status == {"complete": True} and [row["msg_id"] for row in entries] == ["wait"]
    assert mailbox.write_owner_message(tmp_path, "deadline", "task", msg_id="stop", kind=mailbox.KIND_FINALIZE_NOW)
    assert peek.pending(tmp_path, "task", seen, 1)
    assert "stop" not in seen
    assert mailbox.acknowledged_task_message_ids(tmp_path, "task", attempt_key=1) == set()


def test_primary_dispatch_retains_explicit_role_and_published_stop_callback(tmp_path, monkeypatch):
    tool_ctx = SimpleNamespace(task_id="task", drive_root=tmp_path, task_attempt=1,
                               task_metadata={}, _loop_mailbox_seen_ids=set())
    ctx = SimpleNamespace(llm=object(), messages=[], active_model="same-model", tool_schemas=[],
                          active_effort="high", max_retries=3, drive_logs=tmp_path / "logs",
                          task_id="task", round_idx=1, event_queue=None, accumulated_usage={},
                          task_type="task", active_use_local=False, tools=SimpleNamespace(_ctx=tool_ctx),
                          model_role="light", context_fit_plan=None, active_context_mode="max", drive_root=tmp_path)
    calls = []
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _tools: None)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **kwargs: calls.append(kwargs) or (None, None))
    loop_model_call._dispatch_round_model(ctx, None, attempt_cap=None)
    assert calls[0]["model_role"] == "light"
    check = calls[0]["stop_retry_check"]
    assert not check()
    mailbox.write_owner_message(tmp_path, "choice", "task", msg_id="wait", kind=mailbox.KIND_MODEL_WAIT)
    assert not check()
    mailbox.write_owner_message(tmp_path, "deadline", "task", msg_id="stop", kind=mailbox.KIND_FINALIZE_NOW)
    assert check() and tool_ctx._transport_repeat_control_reason == "deadline"
    assert tool_ctx._loop_mailbox_seen_ids == set()
