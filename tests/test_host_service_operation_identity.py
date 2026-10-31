"""Real ingress and task-authority controls for repeated skill deliveries."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.task_results import write_task_result
from supervisor import message_bus
from tests._dialogue import chat_rows
from tests.test_host_service_operations import (
    CHAT, MSG, _client, _headers, _chat_row, _inbound, _isolate_queue, _origin_ref, _receipt,
)


@pytest.fixture
def direct_budget_refusal(tmp_path, monkeypatch):
    from queue import SimpleQueue
    from supervisor import state, workers
    from tests._agent_entry_helpers import task_entry_agent

    agent = task_entry_agent(tmp_path, lambda _task: pytest.fail("exhausted budget entered the task body"))
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: agent)
    monkeypatch.setattr(workers, "get_event_q", SimpleQueue)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda _state, *, strict: 0)


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
    rows = chat_rows(tmp_path)
    assert len(rows) == 1
    update = bridge.get_updates(0, timeout=0)[0]["message"]
    ref = message_bus.record_inbound_message(
        bridge, update, chat_id=CHAT, user_id=0, client_message_id=MSG, text="one message", ts="later",
    )
    assert ref == build_owner_message_ref(chat_id=CHAT, client_message_id=MSG, ts=rows[0]["ts"], text="one message")
    assert len(chat_rows(tmp_path)) == 1
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
        scoped.setattr(message_bus, "accept_chat", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("canonical message acceptance could not be persisted")))
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


def test_pre_task_budget_refusal_remains_an_exact_failed_operation(tmp_path, monkeypatch, direct_budget_refusal):
    from supervisor import worker_chat_lane, workers

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
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
def test_ordinary_main_and_transport_refusals_keep_their_original_envelope(monkeypatch, chat_id, direct_budget_refusal):
    """A registered refusal is keyed, but only host operations retain a terminal origin."""
    from supervisor import worker_chat_lane, workers

    sent, frames = [], []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: sent.append((a, kw)))
    worker_chat_lane._handle_chat_direct_locked(chat_id, "hello", task_metadata={
        "origin_message_ref": build_owner_message_ref(chat_id=chat_id, client_message_id="ordinary", ts="now", text="hello"),
    })
    receipt = next(row for row in frames if row.get("type") == "typing")
    assert receipt["client_message_id"] == "ordinary" and receipt["chat_id"] == chat_id
    assert receipt["activity_id"]
    assert sent == [((chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings."), {
        "task_id": receipt["activity_id"], "progress_meta": {"task_terminal_status": "failed"},
        "role": "system", "system_type": "task_not_started",
    })]


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
    rows = chat_rows(tmp_path)
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
    assert len(chat_rows(tmp_path)) == 1
    bridge.enqueue_local_message("ordinary", task_metadata={"_host_operation": True})
    server._process_bridge_updates(bridge, 1, ctx)
    assert "_host_operation" not in routed[-1]["task_metadata"]


def test_the_archive_import_between_an_acceptance_and_its_replay_cannot_turn_the_replay_into_new_work(tmp_path, monkeypatch):
    """The accepted row keeps its identity through the one-time archive import (a legacy chain
    imported underneath it) and through a cutover of a legacy-written live file: a replay rejoins
    the same source, nothing is enqueued twice."""
    from ouroboros import dialogue_import

    (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive" / "chat_20261001T000000.jsonl").write_text(
        '{"direction": "out", "chat_id": 1, "text": "older history", "ts": "2026-10-01T00:00:00+00:00"}\n', encoding="utf-8")
    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "hello"}
    accepted = client.post("/chat/inject", headers=_headers(), json=body)
    assert accepted.status_code == 202
    source = _origin_ref(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["pending"] == 1, "the legacy segment is not imported yet"
    replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert replay.status_code == 202 and replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source
    dialogue_import.run_dialogue_import(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["imported"]
    with (tmp_path / "logs" / "chat.jsonl").open("a", encoding="utf-8") as handle:  # an older body appended
        handle.write('{"direction": "out", "chat_id": 1, "text": "from an older body", "ts": "2026-10-17T00:00:00+00:00"}\n')
    dialogue_import.cutover(tmp_path)
    archived_replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert archived_replay.status_code == 202 and archived_replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source
    assert [row["text"] for row in chat_rows(tmp_path)] == ["older history", "hello", "from an older body"]


def test_racing_named_upload_rejoin_keeps_only_the_accepted_copy(tmp_path, monkeypatch):
    import pathlib
    import threading
    from ouroboros.gateway import host_service

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    barrier = threading.Barrier(2)
    original = host_service.store_upload
    def copy(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(host_service, "store_upload", copy)
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
    original = message_bus.accept_chat
    def fail(*args, **kwargs):
        if failure == "write_unknown_landed":
            original(*args, **kwargs)
        raise OSError("controlled admission failure")
    if failure.startswith("write_unknown"):
        monkeypatch.setattr(message_bus, "accept_chat", fail)
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
    original = message_bus.accept_chat
    def held(*args, **kwargs):
        receipt = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return receipt
    monkeypatch.setattr(message_bus, "accept_chat", held)
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


def test_solitary_inline_photo_retains_a_staged_original_after_acceptance(tmp_path, monkeypatch):
    import base64
    from pathlib import Path
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.context import build_user_content
    from tests.test_live_image_delivery import pixels, image_blocks, decode_block

    raw = pixels(size=(9000, 24))
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "image_base64": base64.b64encode(raw).decode(),
            "image_mime": "image/png", "text": ""}
    response = client.post("/chat/inject", headers=_headers(), json=body)
    assert response.status_code == 202, response.text
    message = bridge.get_updates(0, timeout=0)[0]["message"]
    uploads = message["task_metadata"]["chat_attachment_uploads"]
    assert len(uploads) == 1 and Path(uploads[0]["path"]).read_bytes() == raw
    manifest = stage_task_attachments(tmp_path, "photo-task", uploads)
    content = build_user_content({"id": "photo-task", "drive_root": tmp_path, "attachment_images": manifest})
    block = image_blocks([{"content": content}])[0]
    assert Path(block["_source_path"]).read_bytes() == raw
    assert decode_block(block) == raw
    assert client.post("/chat/inject", headers=_headers(), json=body).json()["rejoined"] is True
    assert bridge._inbox.empty()
