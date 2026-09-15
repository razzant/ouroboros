"""Real ingress and task-authority controls for repeated skill deliveries."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.task_results import write_task_result
from ouroboros.utils import iter_jsonl_objects
from supervisor import message_bus
from tests.test_host_service_operations import (
    CHAT, MSG, _client, _headers, _chat_row, _inbound, _isolate_queue, _origin_ref, _receipt,
)


def test_atomic_acceptance_before_dequeue_and_exact_source_consumption(tmp_path, monkeypatch):
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
            "accepted_source_ref": {"chat_id": 999}, "task_metadata": {"origin_message_ref": {"chat_id": 999}}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: client.post("/chat/inject", headers=_headers(), json=body), range(4)))
    assert all(reply.status_code == 202 for reply in replies)
    assert bridge._inbox.qsize() == 1
    assert sum(reply.json().get("rejoined", False) for reply in replies) == 3
    rows = list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))
    assert len(rows) == 1
    update = bridge.get_updates(0, timeout=0)[0]["message"]
    ref = message_bus.record_inbound_message(
        bridge, update, chat_id=CHAT, user_id=0, client_message_id=MSG, text="one message", ts="later",
    )
    assert ref == build_owner_message_ref(chat_id=CHAT, client_message_id=MSG, ts=rows[0]["ts"], text="one message")
    assert len(list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))) == 1
    assert not update.get("task_metadata", {}).get("origin_message_ref")
    replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert replay.json()["rejoined"] is True and bridge._inbox.empty()
    distinct = client.post("/chat/inject", headers=_headers(), json={**body, "client_message_id": "message-two"})
    assert distinct.status_code == 202 and bridge._inbox.qsize() == 1


def test_failed_acceptance_never_queues_and_restart_never_requeues(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    atomic_write_json(tmp_path / "state/state.json", {"session_id": "before"})
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message"}
    with monkeypatch.context() as scoped:
        scoped.setattr(message_bus, "append_jsonl", lambda *a, **kw: False)
        assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 500
    assert bridge._inbox.empty()
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    atomic_write_json(tmp_path / "state/state.json", {"session_id": "after"})
    restarted = message_bus.LocalChatBridge()
    restarted_client = _client(tmp_path, restarted)
    assert restarted_client.post("/chat/inject", headers=_headers(), json=body).json()["rejoined"]
    state = restarted_client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert state["status"] == "lost" and restarted._inbox.empty()


@pytest.mark.parametrize("mismatch", ["chat_id", "ts", "text_sha256", "source"])
def test_annotation_collision_never_authorizes_foreign_task(tmp_path, monkeypatch, mismatch):
    client = _client(tmp_path)
    _inbound(tmp_path, "own source")
    ref = _origin_ref(tmp_path)
    if mismatch == "source":
        _chat_row(tmp_path, "in", "foreign source", chat_id=1, client_message_id=MSG, source="web")
        ref = build_owner_message_ref(chat_id=1, client_message_id=MSG, ts="foreign", text="foreign source")
    else:
        ref[mismatch] = 1 if mismatch == "chat_id" else "0" * 64 if mismatch == "text_sha256" else "foreign"
    _, pending = _isolate_queue(monkeypatch, tmp_path, [{"id": "foreign", "chat_id": 1, "origin_message_ref": ref}])
    write_task_result(tmp_path, "foreign", "scheduled", origin_message_ref=ref, result="foreign answer")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "foreign")
    state = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert state["status"] == "pending" and "task_id" not in state and "text" not in state
    result = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": f"{CHAT}:{MSG}"})
    assert result.status_code == 409 and len(pending) == 1
    assert not (tmp_path / "state/cancel_intents.json").exists()


def test_interleaved_same_chat_answers_are_matched_by_task_origin(tmp_path):
    client = _client(tmp_path)
    _inbound(tmp_path, "request A")
    ref_a = _origin_ref(tmp_path)
    _chat_row(tmp_path, "in", "request B", client_message_id="B")
    write_task_result(tmp_path, "task-a", "completed", origin_message_ref=ref_a, result="answer A")
    _chat_row(tmp_path, "out", "answer A", task_id="task-a")
    a = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    b = client.get(f"/chat/operations/{CHAT}:B", headers=_headers()).json()
    assert a["status"] == "completed" and a["text"] == "answer A"
    assert b["status"] == "pending" and "text" not in b


def test_pre_task_budget_refusal_remains_an_exact_failed_operation(tmp_path, monkeypatch):
    from supervisor import state, worker_chat_lane, workers

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 0)
    monkeypatch.setattr(workers, "send_with_budget", message_bus.send_with_budget)
    client = _client(tmp_path, bridge)
    assert client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "do work",
    }).status_code == 202
    ref = bridge.get_updates(0, timeout=0)[0]["message"]["accepted_source_ref"]
    worker_chat_lane._handle_chat_direct_locked(CHAT, "do work", task_metadata={"origin_message_ref": ref, "_host_operation": True})
    result = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert result["status"] == "failed" and "Budget exhausted" in result["text"]


@pytest.mark.parametrize("chat_id", [1, 123456])
def test_ordinary_main_and_transport_refusals_keep_their_original_envelope(monkeypatch, chat_id):
    from supervisor import state, worker_chat_lane, workers

    sent = []
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 0)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: sent.append((a, kw)))
    worker_chat_lane._handle_chat_direct_locked(chat_id, "hello", task_metadata={
        "origin_message_ref": build_owner_message_ref(chat_id=chat_id, client_message_id="ordinary", ts="now", text="hello"),
    })
    assert sent == [((chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings."), {})]


@pytest.mark.parametrize("host_operation", [False, True])
def test_chat_crash_preserves_only_the_host_accepted_operation(tmp_path, monkeypatch, host_operation):
    from queue import SimpleQueue
    from supervisor import worker_chat_lane, workers

    class CrashingAgent:
        def handle_task(self, task):
            raise RuntimeError("test chat execution failed")

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "get_event_q", SimpleQueue)
    monkeypatch.setattr(workers, "send_with_budget", message_bus.send_with_budget)
    client = _client(tmp_path, bridge)
    assert client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "do work",
    }).status_code == 202
    ref = bridge.get_updates(0, timeout=0)[0]["message"]["accepted_source_ref"]
    worker_chat_lane._run_chat_task(
        CrashingAgent(), CHAT, "do work",
        task_metadata={"origin_message_ref": ref, "_host_operation": host_operation},
    )
    rows = list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))
    final = rows[-1]
    assert "test chat execution failed" in final["text"]
    assert final["task_terminal_status"] == "failed"
    assert final.get("origin_message_ref") == (ref if host_operation else None)
    result = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    if host_operation:
        assert result["status"] == "failed" and "test chat execution failed" in result["text"]
    else:
        assert result["status"] == "pending" and "text" not in result


def test_real_server_consumes_one_source_and_authors_only_its_operation_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import server

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    routed = []
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, row: routed.append(row))
    ctx = SimpleNamespace(load_state=lambda: {"owner_id": 1}, update_state=lambda mutator: mutator({"owner_id": 1}))
    client = _client(tmp_path, bridge)
    client.post("/chat/inject", headers=_headers(), json={"chat_id": CHAT, "client_message_id": MSG, "text": "hello"})
    ref = _origin_ref(tmp_path)
    server._process_bridge_updates(bridge, 0, ctx)
    assert routed[-1]["origin_message_ref"] == ref
    assert routed[-1]["task_metadata"]["_host_operation"] is True
    assert len(list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))) == 1
    bridge.enqueue_local_message("ordinary", task_metadata={"_host_operation": True})
    server._process_bridge_updates(bridge, 1, ctx)
    assert "_host_operation" not in routed[-1]["task_metadata"]


def test_rotation_during_source_read_cannot_turn_replay_into_new_work(tmp_path, monkeypatch):
    import sys
    from ouroboros import utils
    from supervisor.state import rotate_chat_log_if_needed

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "hello"}
    accepted = client.post("/chat/inject", headers=_headers(), json=body)
    assert accepted.status_code == 202
    source = _origin_ref(tmp_path)
    original = utils.jsonl_archive_segments
    outcomes = []

    def rotate_after_live_open(path, **kwargs):
        if not outcomes:
            try:
                rotate_chat_log_if_needed(tmp_path, max_bytes=1)
            except PermissionError:
                # Windows can defer the independent rotator while the reader
                # owns the live handle. Its failure is not a reader failure.
                assert sys.platform == "win32"
                assert (tmp_path / "logs/chat.jsonl").stat().st_size > 0
                outcomes.append("open_handle")
            else:
                assert original(path)
                outcomes.append("rotated")
        return original(path, **kwargs)

    monkeypatch.setattr(utils, "jsonl_archive_segments", rotate_after_live_open)
    replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert replay.status_code == 202 and replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source
    assert outcomes in (["rotated"], ["open_handle"])
    if sys.platform != "win32":
        assert outcomes == ["rotated"]
    # With all read handles closed, both platforms must really rotate and
    # recover the same source from its archive without another enqueue.
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    assert original(tmp_path / "logs/chat.jsonl")
    archived_replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert archived_replay.status_code == 202 and archived_replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source


def test_racing_named_upload_rejoin_keeps_only_the_accepted_copy(tmp_path, monkeypatch):
    import pathlib
    import threading
    from ouroboros.gateway import host_service

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    barrier = threading.Barrier(2)
    original = host_service.store_chat_upload
    def copy(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(host_service, "store_chat_upload", copy)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
            "attachments": [{"path": str(source)}]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post("/chat/inject", headers=_headers(), json=body), range(2)))
    assert all(response.status_code == 202 for response in responses)
    assert sum(response.json().get("rejoined", False) for response in responses) == 1
    updates = bridge.get_updates(0, timeout=0)
    assert len(updates) == 1 and bridge._inbox.empty()
    stored = pathlib.Path(updates[0]["message"]["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert list((tmp_path / "uploads").iterdir()) == [stored]
    assert stored.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("failure", ["write_unknown_empty", "write_unknown_landed", "queue_unknown"])
def test_named_upload_custody_survives_unknown_write_or_queue_outcome(tmp_path, monkeypatch, failure):
    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    original = message_bus.log_chat
    def fail(*args, **kwargs):
        if failure == "write_unknown_landed":
            original(*args, **kwargs)
        raise OSError("controlled admission failure")
    if failure.startswith("write_unknown"):
        monkeypatch.setattr(message_bus, "log_chat", fail)
    else:
        monkeypatch.setattr(bridge, "enqueue_local_message", fail)
    response = client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "one message",
        "attachments": [{"path": str(source)}],
    })
    assert response.status_code == 500 and bridge._inbox.empty()
    copies = list((tmp_path / "uploads").iterdir())
    assert len(copies) == 1
    assert copies[0].read_bytes() == source.read_bytes()
    row = message_bus.accepted_chat_message(tmp_path, CHAT, MSG)
    assert bool(row) is (failure != "write_unknown_empty")
    assert source.read_bytes() == b"complete attachment"


@pytest.mark.parametrize("cancel_mode", ["asyncio", "anyio"])
def test_cancelled_named_acceptance_settles_before_upload_cleanup(tmp_path, monkeypatch, cancel_mode):
    import asyncio
    import threading
    import anyio
    from types import SimpleNamespace
    from ouroboros.gateway import host_service

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    entered, release = threading.Event(), threading.Event()
    original = message_bus.log_chat
    def held(*args, **kwargs):
        row = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return row
    monkeypatch.setattr(message_bus, "log_chat", held)
    async def body():
        return {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
                "attachments": [{"path": str(source)}]}
    request = SimpleNamespace(app=client.app, headers={k.lower(): v for k, v in _headers().items()}, json=body)
    ctx = client.app.state.host_service_context
    async def run():
        scope = anyio.CancelScope()
        async def call():
            with scope:
                return await host_service._api_chat_inject(request)
        task = asyncio.create_task(call())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            scope.cancel() if cancel_mode == "anyio" else task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and ctx._inflight["a2a"] == 1
        finally:
            release.set()
            if cancel_mode == "anyio":
                await task
                assert scope.cancelled_caught
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert ctx._inflight["a2a"] == 0
    asyncio.run(run())
    message = bridge.get_updates(0, timeout=0)[0]["message"]
    from pathlib import Path
    stored = Path(message["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert stored.read_bytes() == source.read_bytes()
