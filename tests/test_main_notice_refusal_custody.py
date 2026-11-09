"""Refused owner messages end their own invocation before any delivery effect."""
import queue as stdqueue

import pytest

from tests.test_local_refusal_custody import _world, _pause_and_resume



@pytest.mark.parametrize(("next_action", "case", "chat", "args", "status", "code", "text"), [
    pytest.param("pause", "empty", 1, {"text": ""}, "error", "TOOL_ARG_ERROR", "⚠️ Empty message.", id='pause-empty'),
    pytest.param("pause", "main", 1, {"text": "Hello", "destination": "main"}, "blocked", "ACCESS_BLOCKED", "⚠️ MAIN_NOTICE_BLOCKED: destination='main' refused: this root already runs in Main, its current room, "
     "so destination='current' reaches your human there; 'main' is for a Project or owner-started root, and this one is neither. "
     "Nothing was sent.", id='pause-main'),
    pytest.param("continue", "no_chat", None, {"text": "Hello"}, "unavailable", "CAPABILITY_UNAVAILABLE", "⚠️ No active chat — cannot send proactive message.", id='continue-no-chat'),
    pytest.param("continue", "destination", 1, {"text": "Hello", "destination": "elsewhere"}, "error", "TOOL_ARG_ERROR", "⚠️ SEND_USER_MESSAGE_DESTINATION: destination='elsewhere' is not a destination; use 'current' "
     "(this room) or 'main' (the owner's main chat). Nothing was sent.", id='continue-destination'),
])
def test_message_refusal_preserves_result_and_releases_only_its_call(
        tmp_path, monkeypatch, case, chat, args, status, code, text, next_action):
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE, _interrupted, _owner_mail

    registry, task_queue, workers, _repo = _world(tmp_path, monkeypatch)
    ctx = registry._ctx
    ctx.current_chat_id = chat
    ctx.event_queue = stdqueue.Queue()
    result = registry.execute_result("send_user_message", args)
    assert (result.status, result.code, result.text) == (status, code, text)
    assert ctx.event_queue.empty() and not ctx.pending_events
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    assert result.meta["operation_outcome"] == "completed_no_effect"
    assert cold_blockers(ctx) == []
    if next_action == "pause":
        _pause_and_resume(tmp_path, monkeypatch, task_queue, workers)
    else:
        workers.RUNNING.clear()
        _interrupted(tmp_path, task_id="root")
        _owner_mail(tmp_path, task_id="root")
        continued = admit_continuation("root", action_nonce=NONCE)
        assert continued["ok"] and not continued["held"], continued


@pytest.mark.parametrize(("live", "destination", "system_type"), [
    (False, "current", "proactive_message"),
    (True, "main", "main_notice"),
])
def test_successful_message_still_delivers_once_and_is_not_a_no_effect_refusal(
        tmp_path, monkeypatch, destination, system_type, live):
    from ouroboros.task_results import load_task_result

    registry, task_queue, workers, _repo = _world(tmp_path, monkeypatch)
    ctx = registry._ctx
    ctx.current_chat_id = 1
    ctx.task_metadata = {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-source"}}
    ctx.event_queue = stdqueue.Queue() if live else None
    result = registry.execute_result("send_user_message", {"text": "Hello", "destination": destination})
    assert result.status == "ok"
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    if live:
        delivered = ctx.event_queue.get_nowait()
        assert ctx.event_queue.empty() and not ctx.pending_events
    else:
        assert len(ctx.pending_events) == 1
        delivered = ctx.pending_events[0]
    assert delivered["text"] == "Hello" and delivered["chat_id"] == 1
    assert delivered["system_type"] == system_type
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _pause_and_resume(tmp_path, monkeypatch, task_queue, workers)


def test_refused_message_does_not_release_an_unrelated_unknown_operation(tmp_path, monkeypatch):
    from ouroboros.owner_pause import operation_start, tool_handoff
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import conflicting_writers

    registry, task_queue, _workers, _repo = _world(tmp_path, monkeypatch)
    ctx = registry._ctx
    ctx.current_chat_id = 1
    with pytest.raises(TimeoutError):
        with tool_handoff(ctx, "unknown-external-effect"):
            with operation_start(ctx):
                raise TimeoutError("a submitted effect has no terminal receipt")
    before = load_task_result(tmp_path, "root").get("launch_handoffs", {})
    result = registry.execute_result("send_user_message", {"text": "Hello", "destination": "main"})
    assert result.meta["operation_outcome"] == "completed_no_effect"
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == before
    assert any(row["kind"] == "tool_handoff" for row in conflicting_writers(task_queue, "root"))
