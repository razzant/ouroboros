"""Transport skills relay files and decision answers through the host (#668, #472).

``/chat/inject`` ``attachments`` are copied into the browser paperclip's
``data/uploads`` store and forwarded as ``chat_attachment_uploads``; a file-only
message passes the two empty-message gates; ``/chat/decision`` is the same
``answer_decision`` ingress as ``POST /api/decisions``.
"""
from __future__ import annotations

import os
import pathlib
import re

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.host_service import create_host_service_app
from tests.test_host_service_api import FakeBridge, _seed_token

_STORED_NAME = re.compile(r"^[0-9a-f]{32}_report\.pdf$")


def _skill_file(tmp_path: pathlib.Path, name: str = "a.pdf", payload: bytes = b"%PDF-1.4 hello") -> pathlib.Path:
    inbox = tmp_path / "state" / "skills" / "telegram" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    source = inbox / name
    source.write_bytes(payload)
    return source


def _client(tmp_path: pathlib.Path, bridge: FakeBridge, permissions=("inject_chat",)) -> TestClient:
    _seed_token(tmp_path, skill="telegram", token="token", permissions=list(permissions))
    return TestClient(create_host_service_app(tmp_path, bridge_getter=lambda: bridge))


def test_inject_copies_attachments_into_the_shared_upload_store(tmp_path):
    source = _skill_file(tmp_path)
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)

    response = client.post(
        "/chat/inject", headers={"X-Skill-Token": "token"},
        json={"text": "", "chat_id": 42, "user_id": 42, "client_message_id": "tg:17",
              "attachments": [{"path": str(source), "name": "report.pdf", "mime": "application/pdf"}]},
    )

    assert response.status_code == 202, response.json()
    message = bridge.messages[0]
    assert message["text"] == ""
    assert message["client_message_id"] == "tg:17"
    uploads = message["task_metadata"]["chat_attachment_uploads"]
    assert len(uploads) == 1
    stored = pathlib.Path(uploads[0]["path"])
    assert stored.parent == tmp_path / "uploads"
    assert _STORED_NAME.match(stored.name), stored.name  # the paperclip's naming → secret-name rule
    assert stored.read_bytes() == b"%PDF-1.4 hello"
    assert uploads[0]["label"] == "report.pdf" and uploads[0]["mime"] == "application/pdf"
    assert source.exists(), "the host copies; the skill owns its parked file"
    assert not list((tmp_path / "uploads").glob(".*.uploading"))


def test_inject_without_attachments_keeps_the_historical_kwargs(tmp_path):
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    response = client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                           json={"text": "hello", "chat_id": 42})
    assert response.status_code == 202
    assert "task_metadata" not in bridge.messages[0]
    assert "client_message_id" not in bridge.messages[0]


@pytest.mark.parametrize("copy_fails", [False, True])
def test_cancelled_inject_retains_copy_and_inflight_until_worker_settles(tmp_path, monkeypatch, copy_fails):
    import asyncio
    import threading
    from types import SimpleNamespace
    from ouroboros.gateway import host_service

    source = _skill_file(tmp_path)
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    ctx = client.app.state.host_service_context
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original_copy, original_leave = host_service.store_chat_upload, ctx._leave_inflight
    left = []

    def copy(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            if copy_fails:
                raise OSError("controlled disk copy failure")
            return original_copy(*args, **kwargs)
        finally:
            finished.set()

    def leave(skill):
        left.append(skill)
        original_leave(skill)

    async def payload():
        return {"chat_id": 42, "text": "file", "attachments": [{"path": str(source)}]}

    monkeypatch.setattr(host_service, "store_chat_upload", copy)
    monkeypatch.setattr(ctx, "_leave_inflight", leave)
    request = SimpleNamespace(app=client.app, headers={"x-skill-token": "token"}, json=payload)

    async def run():
        task = asyncio.create_task(host_service._api_chat_inject(request))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and ctx._inflight["telegram"] == 1
            assert left == [] and not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert finished.is_set() and ctx._inflight["telegram"] == 0
        assert left == ["telegram"] and bridge.messages == []
        assert source.is_file()

    asyncio.run(run())


def test_inject_refuses_files_outside_the_skill_state_and_bad_shapes(tmp_path):
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x", encoding="utf-8")
    inbox = tmp_path / "state" / "skills" / "telegram" / "inbox"
    inbox.mkdir(parents=True)
    link = inbox / "escape.txt"
    os.symlink(outside, link)
    cases = [
        ([{"path": str(outside)}], "outside this skill's state"),
        ([{"path": str(link)}], "outside this skill's state"),
        ([{"path": str(inbox / "missing.bin")}], "not a regular file"),
        ([{"path": str(inbox)}], "not a regular file"),
        (["not-an-object"], "must be an object"),
        ({"path": "x"}, "must be a list"),
    ]
    for attachments, needle in cases:
        response = client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                               json={"text": "", "chat_id": 42, "attachments": attachments})
        assert response.status_code == 400, (attachments, response.json())
        assert needle in response.json()["error"], (attachments, response.json())
    assert bridge.messages == []
    assert not (tmp_path / "uploads").exists()


def test_inject_refuses_oversize_attachments_with_413(tmp_path, monkeypatch):
    import ouroboros.gateway.files as files

    monkeypatch.setattr(files, "_CHAT_UPLOAD_MAX_BYTES", 4)
    source = _skill_file(tmp_path, payload=b"12345")
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    response = client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                           json={"text": "", "chat_id": 42, "attachments": [{"path": str(source)}]})
    assert response.status_code == 413
    assert bridge.messages == []


def test_decision_route_requires_the_inject_grant(tmp_path):
    client = _client(tmp_path, FakeBridge(), permissions=())
    response = client.post("/chat/decision", headers={"X-Skill-Token": "token"},
                           json={"request_id": "tg:1", "decision_id": "quiz:t:q", "option_index": 0})
    assert response.status_code == 403


def test_decision_route_relays_to_the_one_decision_ingress(tmp_path, monkeypatch):
    import ouroboros.gateway.task_decision as td

    captured = {}

    async def fake_answer(drive_root, body):
        captured["drive_root"] = drive_root
        captured["body"] = body
        return 200, {"ok": True, "decision_id": body["decision_id"], "state": "answered", "answered_index": 1}

    monkeypatch.setattr(td, "answer_decision", fake_answer)
    client = _client(tmp_path, FakeBridge())
    body = {"request_id": "tg:99", "decision_id": "quiz:task-1:q1", "option_index": 1}
    response = client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json=body)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "decision_id": "quiz:task-1:q1", "state": "answered", "answered_index": 1}
    assert captured == {"drive_root": tmp_path, "body": body}


def test_decision_route_returns_the_ingress_refusals_verbatim(tmp_path):
    client = _client(tmp_path, FakeBridge())
    # No projection exists for this quiz → the ingress's own 404.
    response = client.post("/chat/decision", headers={"X-Skill-Token": "token"},
                           json={"request_id": "tg:1", "decision_id": "quiz:task-1:q1", "option_index": 0})
    assert response.status_code == 404
    assert response.json()["reason_code"] == "quiz_not_found"
    assert response.json()["ok"] is False
    bad = client.post("/chat/decision", headers={"X-Skill-Token": "token", "content-type": "application/json"},
                      content=b"{not json")
    assert bad.status_code == 400


def test_decision_route_is_rate_limited_per_skill(tmp_path, monkeypatch):
    import ouroboros.gateway.task_decision as td

    async def fake_answer(drive_root, body):
        return 200, {"ok": True}

    monkeypatch.setattr(td, "answer_decision", fake_answer)
    client = _client(tmp_path, FakeBridge())
    app_ctx = client.app.state.host_service_context
    app_ctx.rate_limiter.limit = 2
    statuses = [
        client.post("/chat/decision", headers={"X-Skill-Token": "token"},
                    json={"request_id": f"tg:{i}", "decision_id": "quiz:t:q", "option_index": 0}).status_code
        for i in range(3)
    ]
    assert statuses == [200, 200, 429]


# --- the two empty-message gates ---------------------------------------------


def test_bus_admits_a_file_only_message_and_still_drops_an_empty_one():
    import supervisor.message_bus as message_bus

    bridge = message_bus.LocalChatBridge({})
    bridge.enqueue_local_message("", chat_id=42, user_id=42, source="skill:telegram")
    assert bridge.get_updates(offset=0, timeout=0) == []
    uploads = [{"path": "/tmp/x.pdf", "label": "x.pdf", "mime": "application/pdf"}]
    bridge.enqueue_local_message("", chat_id=42, user_id=42, source="skill:telegram",
                                 task_metadata={"chat_attachment_uploads": uploads})
    updates = bridge.get_updates(offset=0, timeout=0)
    assert len(updates) == 1
    assert updates[0]["message"]["text"] == ""
    assert updates[0]["message"]["task_metadata"] == {"chat_attachment_uploads": uploads}


def test_server_routes_a_file_only_owner_message(monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    from tests.test_transport_commands import Bridge, Ctx

    routed = []
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, incoming: routed.append(incoming))
    uploads = [{"path": "/tmp/x.pdf", "label": "x.pdf", "mime": "application/pdf"}]
    bridge = Bridge([
        {"chat": {"id": 42}, "from": {"id": 7}, "text": "", "source": "skill:telegram",
         "task_metadata": {"chat_attachment_uploads": uploads}},
        {"chat": {"id": 42}, "from": {"id": 7}, "text": "", "source": "skill:telegram"},
    ])
    server._process_bridge_updates(bridge, 0, Ctx({"owner_id": 7, "owner_chat_id": 42}))

    assert len(routed) == 1, "the text-less file message must reach routing; the truly empty one must not"
    assert routed[0]["text"] == ""
    assert routed[0]["log_text"] == "(file attached)"
    assert routed[0]["task_metadata"] == {"chat_attachment_uploads": uploads}
