"""Named ingress keeps terminal reply identity and the normal cancellation owner."""

from tests import _dialogue as dialogue_test
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import message_bus, state
from supervisor.events import _handle_send_message
from tests.test_cancel_cascade_v664 import _fake_worker, _install_worker, _isolate_queue
from tests.test_host_service_operations import CHAT, MSG, _client, _headers, _inbound, _origin_ref, _receipt


@pytest.mark.parametrize("host_operation", [True, False])
def test_actual_final_producer_preserves_named_reply_and_rejoin(tmp_path, monkeypatch, host_operation):
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    monkeypatch.setattr(state, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "answer once"}
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    ref = bridge.get_updates(0, timeout=0)[0]["message"]["accepted_source_ref"]
    task_id = f"inline-{host_operation}"
    task = {"id": task_id, "type": "task", "chat_id": CHAT, "text": "answer once",
            "_is_direct_chat": True,
            "origin_message_ref": ref, "metadata": {"_host_operation": host_operation}}
    from ouroboros.agent import OuroborosAgent

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    agent = OuroborosAgent.__new__(OuroborosAgent)
    agent.env = env
    agent._persist_running_record(task)
    events = []
    pipeline.emit_task_results(
        env, None, None, events, task,
        "The exact requested answer.", {"execution_status": "ok", "reason_code": "final_message"},
        {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs",
    )
    final = next(row for row in events if row["type"] == "send_message")
    assert final["progress_meta"].get("origin_message_ref") == (ref if host_operation else None)
    if host_operation:
        assert final["progress_meta"]["origin_message_ref"] is not ref
    assert final["progress_meta"]["task_phase"] == "finalizing"
    _handle_send_message(final, SimpleNamespace(
        DRIVE_ROOT=tmp_path, send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *a, **k: None,
    ))
    result = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert result["status"] == "completed" and result["text"] == "The exact requested answer."
    rejoined = client.post("/chat/inject", headers=_headers(), json={**body, "wait_for_response": True})
    assert rejoined.status_code == 200 and rejoined.json()["rejoined"]
    assert rejoined.json()["response"] == result["text"]
    assert bridge._inbox.empty()


@pytest.mark.parametrize("case,chat,user,owner,expected", [
    ("missing_identity", -42, 0, {}, "failed"),
    ("registration", 42, 42, {}, "completed"),
    ("status", 42, 42, {"owner_external_id": 42, "owner_external_chat_id": 42}, "completed"),
    ("wrong_owner", 43, 43, {"owner_external_id": 42, "owner_external_chat_id": 42}, "failed"),
])
def test_server_inline_replies_settle_their_exact_ingress(tmp_path, monkeypatch, case, chat, user, owner, expected):
    import server

    bridge = message_bus.LocalChatBridge()
    # This is an initialized owner with a known-empty external binding, not
    # a missing/lost state that could legitimately refuse first registration.
    live_state = {"owner_id": 1, "owner_external_id": None,
                  "owner_external_chat_id": None, **owner}
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: live_state)
    monkeypatch.setattr(state, "status_text", lambda *a: "actual status reply")
    ctx = SimpleNamespace(load_state=lambda: dict(live_state), update_state=lambda fn: fn(live_state),
                          send_with_budget=message_bus.send_with_budget, WORKERS={}, PENDING=[], RUNNING={})
    client = _client(tmp_path, bridge)
    body = {"chat_id": chat, "user_id": user, "client_message_id": case, "text": "/status"}
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    server._process_bridge_updates(bridge, 0, ctx)
    rows = list(dialogue_test.log_rows(tmp_path / "logs/chat.jsonl"))
    assert len(rows) == 2 and rows[-1]["task_terminal_status"] == expected
    assert rows[-1]["origin_message_ref"]["client_message_id"] == case
    result = client.get(f"/chat/operations/{chat}:{case}", headers=_headers()).json()
    assert result["status"] == expected and result["text"] == rows[-1]["text"]
    rejoined = client.post("/chat/inject", headers=_headers(), json={**body, "wait_for_response": chat < 0})
    assert rejoined.status_code == 200 and rejoined.json()["rejoined"]
    assert bridge._inbox.empty() and not list((tmp_path / "task_results").glob("*.json"))


@pytest.mark.parametrize("kind", ["root_worker", "child_worker", "settled"])
def test_completed_result_uses_existing_live_cancellation_owner(tmp_path, monkeypatch, kind):
    from supervisor import workers

    task_queue, _ = _isolate_queue(monkeypatch, tmp_path, [])
    monkeypatch.setattr(task_queue, "_emit_cancel_task_done", lambda *a, **k: None)
    client = _client(tmp_path)
    _inbound(tmp_path, "work")
    ref = _origin_ref(tmp_path)
    task = {"id": "root", "root_task_id": "root", "chat_id": CHAT, "origin_message_ref": ref}
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "root")
    write_task_result(tmp_path, "root", "completed", result="Final answer", origin_message_ref=ref)
    if kind != "settled":
        running = dict(task) if kind == "root_worker" else {
            "id": "child", "root_task_id": "root", "parent_task_id": "root", "chat_id": CHAT,
        }
        task_queue.RUNNING[running["id"]] = {"task": running}
        worker, physical = _fake_worker(running["id"])
        _install_worker(monkeypatch, worker)
        monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", lambda *a, **k: physical.__setitem__("alive", False))
        monkeypatch.setattr(workers, "get_event_q", __import__("queue").Queue)
    view = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert view["status"] == "completed" and view["cancel_supported"] is (kind != "settled")
    response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": f"{CHAT}:{MSG}"})
    assert response.status_code == 200 and response.json()["outcome"] == "already_terminal"
    assert load_task_result(tmp_path, "root")["status"] == "completed"
    assert load_task_result(tmp_path, "root")["result"] == "Final answer"
    if kind != "settled":
        assert not physical["alive"] and (tmp_path / "state/cancel_intents.json").exists()
        assert not task_queue.RUNNING
    else:
        assert not (tmp_path / "state/cancel_intents.json").exists()


def test_restart_acknowledgement_does_not_claim_the_operation_finished(tmp_path, monkeypatch):
    import server
    from ouroboros import server_restart

    bridge = message_bus.LocalChatBridge()
    live_state = {"owner_id": 1, "owner_external_id": 42, "owner_external_chat_id": 42}
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: live_state)
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", lambda *a, **k: (False, "controlled refusal"))
    ctx = SimpleNamespace(load_state=lambda: dict(live_state), update_state=lambda fn: fn(live_state),
                          send_with_budget=message_bus.send_with_budget, safe_restart=object())
    client = _client(tmp_path, bridge)
    body = {"chat_id": 42, "user_id": 42, "client_message_id": "restart-refused", "text": "/restart"}
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    server._process_bridge_updates(bridge, 0, ctx)
    inbound, acknowledgement, refused = list(dialogue_test.log_rows(tmp_path / "logs/chat.jsonl"))
    assert inbound["direction"] == "in"
    assert acknowledgement["text"] == "♻️ Restarting."
    assert acknowledgement["origin_message_ref"] == refused["origin_message_ref"]
    assert "task_terminal_status" not in acknowledgement
    assert refused["task_terminal_status"] == "failed" and "controlled refusal" in refused["text"]
    assert not acknowledgement.get("is_progress") and not refused.get("is_progress")
    view = client.get("/chat/operations/42:restart-refused", headers=_headers()).json()
    assert view["status"] == "failed" and view["text"] == refused["text"]
