"""Published stop/peek contracts coexist with task-local subscription controls."""

from types import SimpleNamespace

from ouroboros import loop, loop_model_call, loop_transport, owner_mailbox as mailbox


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
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    tool_ctx = SimpleNamespace(task_id="task", drive_root=tmp_path, task_attempt=1,
                               model_role="light", task_metadata={}, _loop_mailbox_seen_ids=set())
    ctx = SimpleNamespace(llm=object(), messages=[], active_model="same-model", tool_schemas=[],
                          active_effort="high", max_retries=3, drive_logs=tmp_path / "logs",
                          task_id="task", round_idx=1, event_queue=None, accumulated_usage={},
                          task_type="task", active_use_local=False, tools=SimpleNamespace(_ctx=tool_ctx),
                          model_role="light", context_fit_plan=None, active_context_mode="max", drive_root=tmp_path,
                          physical_send_claim=None)
    calls = []
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _tools: None)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **kwargs: calls.append(kwargs) or (None, None))
    loop_model_call._dispatch_round_model(ctx, None, attempt_cap=None)
    assert calls[0]["model_role"] == "light"
    mailbox.write_owner_message(tmp_path, "Keep this for my next round.", "task", msg_id="words")
    probes = []
    def observed(_llm, model, **options):
        probes.append((model, options["model_role"]))
        mailbox.write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, "task",
                                    msg_id="stop", kind=mailbox.KIND_FINALIZE_NOW)
        return {"kind": "upstream_http"}

    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", observed)
    episode = loop_transport.TransportWaitEpisode(wait_cause="provider_outcome_unknown")
    assert not loop_transport.continue_unknown_transport(
        episode, llm=ctx.llm, tools=ctx.tools, messages=ctx.messages, accumulated_usage=ctx.accumulated_usage,
        drive_logs=ctx.drive_logs, task_id="task", model=ctx.active_model, emit_progress=lambda *_a, **_kw: None)
    assert probes == [("same-model", "light")] and not episode.continuation_granted
    assert tool_ctx._skip_post_task_synthesis and tool_ctx._transport_repeat_control_reason == REASON_OWNER_STOPPED_DIRECT_TURN
    assert tool_ctx._loop_mailbox_seen_ids == set()
    assert mailbox.acknowledged_task_message_ids(tmp_path, "task", attempt_key=1) == set()
    words, = mailbox.drain_owner_entries(tmp_path, "task", set(), 1, kinds={mailbox.KIND_OWNER_TEXT})
    assert (words["msg_id"], words["text"]) == ("words", "Keep this for my next round.")
