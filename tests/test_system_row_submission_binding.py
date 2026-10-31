"""A System row that answers an owner submission names it; nothing else settles Sending (#1403).

Voice confers no finality, so an unkeyed System reply names the submission it
answers with a live-only ``client_message_id``. Registered direct turns link that
submission through a typing receipt and conclude with a task-keyed terminal.
Only that ``Sending...`` settles; history never folds the System row into the
owner's bubble, and nothing here is a read or delivery receipt.
"""

from tests import _dialogue as dialogue_test

import asyncio
import json
import types

import pytest

from supervisor import message_bus


def test_live_frame_names_the_submission_and_the_durable_row_does_not(tmp_path, monkeypatch):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.task_finalization import host_operation_reply_kwargs

    (tmp_path / "logs").mkdir()
    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    message_bus.log_chat("in", 1, 7, "/bg", client_message_id="cm-owner", drive_root=tmp_path)

    message_bus.send_with_budget(1, "Background consciousness: on", role="system", system_type="command_reply",
                                 **host_operation_reply_kwargs(None, "", "cm-owner"))

    (frame,) = [row for row in frames if row.get("type") == "chat"]
    assert frame["role"] == "system" and frame["client_message_id"] == "cm-owner"
    stored = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert [row["client_message_id"] for row in stored] == ["cm-owner", ""]
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        types.SimpleNamespace(query_params={"chat_id": "1"})))
    replayed = {row["text"]: row for row in json.loads(response.body)["messages"]}
    assert replayed["Background consciousness: on"]["client_message_id"] == ""
    assert replayed["/bg"]["client_message_id"] == "cm-owner"
    # Without a submission the helper adds nothing, exactly as before.
    assert host_operation_reply_kwargs(None, "") == {} and host_operation_reply_kwargs(None, "", "") == {}


def test_owner_command_reply_names_its_submission(monkeypatch):
    import server

    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *_a, **_k: None)
    state = {"owner_id": 5, "owner_chat_id": 1}
    sent = []
    ctx = types.SimpleNamespace(load_state=lambda: dict(state), update_state=lambda fn: fn(dict(state)),
                                send_with_budget=lambda *a, **k: sent.append((a, k)), consciousness=None,
                                kill_workers=None)
    update = {"update_id": 1, "message": {"chat": {"id": 1}, "from": {"id": 5}, "text": "/bg",
                                          "source": "web", "client_message_id": "cm-bg"}}
    server._handle_bridge_update_batch(types.SimpleNamespace(), [update], 0, ctx, [0])
    (_args, kwargs), = sent
    assert kwargs["system_type"] == "command_reply"
    assert kwargs["progress_meta"] == {"client_message_id": "cm-bg"}


def test_direct_lane_refusals_name_their_submission(tmp_path, monkeypatch):
    from queue import SimpleQueue
    from supervisor import state as state_mod
    from supervisor import worker_chat_lane, workers
    from tests._agent_entry_helpers import task_entry_agent

    sent, frames = [], []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    agent = task_entry_agent(tmp_path, lambda _task: pytest.fail("exhausted budget entered the task body"))
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: agent)
    monkeypatch.setattr(workers, "get_event_q", SimpleQueue)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(state_mod, "budget_remaining", lambda _state, *, strict: 0)
    monkeypatch.setattr(state_mod, "load_state", lambda: {})
    worker_chat_lane._handle_chat_direct_locked(1, "hi", task_metadata={"client_message_id": "cm-1"})
    ref = {"chat_id": 1, "client_message_id": "cm-2", "ts": "2026-10-13T00:00:00+00:00", "text_sha256": "0" * 64}
    worker_chat_lane._handle_chat_direct_locked(1, "hi", task_metadata={
        "client_message_id": "cm-2", "origin_message_ref": ref, "_host_operation": True})
    first, second = sent
    assert first[0][1].startswith("🚫 Budget exhausted.")
    assert second[0][1] == first[0][1]
    assert first[1]["task_id"] != second[1]["task_id"]
    # The receipt links Sending to this registered turn; its keyed refusal closes only that turn.
    for (args, kwargs), client_id in zip(sent, ("cm-1", "cm-2")):
        assert args[0] == 1
        assert kwargs["role"] == "system" and kwargs["system_type"] == "task_not_started"
        assert kwargs["task_id"]
        receipts = [row for row in frames if row.get("type") == "typing"
                    and row.get("activity_id") == kwargs["task_id"]]
        assert receipts
        assert all(row["client_message_id"] == client_id and row["chat_id"] == 1 for row in receipts)
    assert first[1]["progress_meta"] == {"task_terminal_status": "failed"}
    # Only a host operation keeps durable terminal correlation; client ids stay on the live receipt.
    assert second[1]["progress_meta"] == {"origin_message_ref": ref, "task_terminal_status": "failed"}

    # The update gate's refusal names the submission it answers, too.
    sent.clear()
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update")
    assert worker_chat_lane.owner_conversation_admitted(1, "cm-3") is False
    (args, kwargs), = sent
    assert args[1].startswith("🔒 An update is using the repository.")
    assert kwargs["progress_meta"] == {"client_message_id": "cm-3"} and kwargs["system_type"] == "managed_update_notice"


def test_ordinary_host_failure_names_its_submission():
    from supervisor.worker_chat_lane import _host_operation_failure

    assert _host_operation_failure({"client_message_id": "cm-swarm", "force_plan": True})["progress_meta"] == {"client_message_id": "cm-swarm"}


def test_initialization_notice_names_the_refused_frame():
    from ouroboros.gateway.ws import _initialization_notice

    named = json.loads(_initialization_notice("cm-early"))
    assert named["role"] == "system" and named["system_type"] == "initialization_notice"
    assert named["client_message_id"] == "cm-early"
    assert "client_message_id" not in json.loads(_initialization_notice())


@pytest.mark.parametrize("reviewer", ["no-such-row", ""])
def test_review_rows_name_the_owner_submission(monkeypatch, reviewer):
    from supervisor import queue

    sent = []
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(queue, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: None)
    monkeypatch.setattr(queue, "enqueue_with_admission_receipt", lambda task: {**task})
    queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1, reviewer=reviewer,
                                      client_message_id="cm-review")
    (args, kwargs), = sent
    assert kwargs["system_type"] == ("deep_self_review_unavailable" if reviewer else "deep_self_review_queued")
    assert kwargs["progress_meta"] == {"client_message_id": "cm-review"}


@pytest.mark.parametrize("chat_id", [1, 42])
@pytest.mark.parametrize("bridge_ready", [False, True])
def test_ws_refusal_keeps_the_submitting_room(monkeypatch, chat_id, bridge_ready):
    """The real socket ingress must answer the Project that sent the failed frame."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from starlette.testclient import TestClient

    from ouroboros.gateway.ws import ws_endpoint

    def fail_accept(*_args, **_kwargs):
        raise OSError("synthetic ingress persistence failure")

    bridge = types.SimpleNamespace(panic=types.SimpleNamespace(request=lambda _: False), ui_send=fail_accept)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: bridge if bridge_ready else None)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        socket.send_json({"type": "chat", "content": "hello", "chat_id": chat_id,
                          "client_message_id": "cm-room"})
        reply = socket.receive_json()
    assert reply["role"] == "system" and reply["system_type"] == "initialization_notice"
    assert reply["client_message_id"] == "cm-room"
    assert reply.get("chat_id") == chat_id


def test_accepted_restart_reply_names_its_submission(monkeypatch):
    """No real restart or thread: execute the real admitted-command reply callback."""
    import threading

    from ouroboros.server_control import dispatch_accepted_restart

    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(threading, "Thread", lambda **kw: types.SimpleNamespace(start=kw["target"]))
    bridge = types.SimpleNamespace(activate_update_transport=lambda _: None,
                                  enqueue_local_message=lambda *_a, **_k: pytest.fail("unexpected fallback"))

    def callback(_text, **kwargs):
        kwargs["reply"]("Restart cancelled: the update is still in progress.", "failed")
        return lambda: True

    dispatch_accepted_restart(bridge, "/restart", callback=callback, source="web", user_id=7,
                              chat_id=42, client_message_id="cm-restart")
    (args, kwargs), = sent
    assert args[0] == 42
    assert kwargs["progress_meta"]["client_message_id"] == "cm-restart"
