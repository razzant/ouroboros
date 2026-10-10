"""The owner-notice route of the Host Service (``POST /notify``, ``gateway/host_notify.py``).

A skill holding the owner's ``notify_owner`` grant tells the owner one thing, now:
one System row in the owner's chat (``system_type="skill_notice"``) signed
``Notice · <skill>``, written through the seam every host row uses, so it reaches
history, transports and the next turn like any System row and wakes no model. A
missing grant, a disabled skill, a bad body or the rate lane refuse before anything
is written; a write that cannot be confirmed answers one 503.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import types

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.host_service import create_host_service_app
from tests.test_host_service_api import _seed_token

HEADERS = {"X-Skill-Token": "tok"}


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """The real message bus bound to this data root, as the running server holds it."""
    from supervisor import message_bus

    frames, published = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: published.append((topic, data)))
    return types.SimpleNamespace(frames=frames, published=published)


def _client(tmp_path: pathlib.Path, *, granted: bool = True):
    _seed_token(tmp_path, skill="cal", token="tok", permissions=["notify_owner"] if granted else [],
                manifest_permissions=["notify_owner"])
    app = create_host_service_app(tmp_path)
    return TestClient(app), app


def _rows(tmp_path: pathlib.Path) -> list[dict]:
    path = tmp_path / "logs" / "chat.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_identity_advertises_the_notify_route(tmp_path: pathlib.Path) -> None:
    client, _app = _client(tmp_path)
    response = client.get("/identity", headers=HEADERS)
    assert response.status_code == 200 and response.json()["notify_version"] == 1


def test_a_notice_is_one_signed_system_row_in_the_owners_chat(tmp_path: pathlib.Path, chat) -> None:
    client, _app = _client(tmp_path)
    response = client.post("/notify", headers=HEADERS,
                           json={"text": "  ⏰ Meeting with Ivan in 15 min ", "source": "Ouroboros"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True and body["chat_id"] == 1 and body["ts"]
    [row] = _rows(tmp_path)
    assert (row["direction"], row["type"], row["chat_id"], row["task_id"]) == ("system", "skill_notice", 1, "")
    assert row["text"] == "Notice · cal\n⏰ Meeting with Ivan in 15 min", "the host signs with the skill's own name"
    assert row["ts"] == body["ts"]
    [frame] = [f for f in chat.frames if f.get("type") == "chat"]
    assert (frame["role"], frame["system_type"], frame["source"]) == ("system", "skill_notice", "cal")
    assert [topic for topic, _ in chat.published] == ["chat.outbound"]


def test_a_notice_goes_to_the_bound_owner_chat(tmp_path: pathlib.Path, chat, monkeypatch) -> None:
    from supervisor import state

    monkeypatch.setattr(state, "control_in_copy", lambda _path, _key: (True, 5_000_000_001))
    client, _app = _client(tmp_path)
    assert client.post("/notify", headers=HEADERS, json={"text": "Backup finished"}).json()["chat_id"] == 5_000_000_001
    assert _rows(tmp_path)[0]["chat_id"] == 5_000_000_001


def test_notices_are_refused_before_anything_is_written(tmp_path: pathlib.Path, chat) -> None:
    from ouroboros.skill_loader import GRANTS_FILENAME, load_skill_grants, save_enabled, skill_state_dir

    client, _app = _client(tmp_path, granted=False)
    refused = client.post("/notify", headers=HEADERS, json={"text": "hi"})
    assert refused.status_code == 403 and "notify_owner" in refused.json()["error"]
    assert client.post("/notify", json={"text": "hi"}).status_code == 403, "no token"
    client, _app = _client(tmp_path)
    assert client.post("/notify", headers=HEADERS, json={"text": "granted"}).status_code == 200
    grant_file = skill_state_dir(tmp_path, "cal") / GRANTS_FILENAME
    revoked = {**json.loads(grant_file.read_text(encoding="utf-8")), "granted_permissions": []}
    grant_file.write_text(json.dumps(revoked), encoding="utf-8")  # the owner's grant no longer holds
    assert load_skill_grants(tmp_path, "cal")["granted_permissions"] == []
    assert client.post("/notify", headers=HEADERS, json={"text": "revoked"}).status_code == 403
    _client(tmp_path)  # granted again
    save_enabled(tmp_path, "cal", False)
    disabled = client.post("/notify", headers=HEADERS, json={"text": "disabled"})
    assert disabled.status_code == 403 and "disabled" in disabled.json()["error"]
    assert [row["text"] for row in _rows(tmp_path)] == ["Notice · cal\ngranted"]


def test_bad_bodies_and_the_rate_lane_write_nothing(tmp_path: pathlib.Path, chat) -> None:
    client, app = _client(tmp_path)
    for request in ({"content": b"not json"}, {"json": ["x"]}, {"json": {"text": "   "}}, {"json": {"text": 5}},
                    {"json": {}}, {"json": {"text": "x" * 401}}):
        assert client.post("/notify", headers=HEADERS, **request).status_code == 400, request
    assert _rows(tmp_path) == []
    assert client.post("/notify", headers=HEADERS, json={"text": "x" * 400}).status_code == 200
    app.state.host_service_context.rate_limiter.allow = lambda key: key != "cal:notify"
    assert client.post("/notify", headers=HEADERS, json={"text": "again"}).status_code == 429
    assert len(_rows(tmp_path)) == 1


def test_without_a_chat_writer_the_notice_is_one_503_and_nothing_is_written(tmp_path, monkeypatch) -> None:
    """A provider-less install never starts the supervisor, so no chat writer exists (disclosed)."""
    from supervisor import message_bus

    client, _app = _client(tmp_path)
    monkeypatch.setattr(message_bus, "DATA_DIR", None)
    response = client.post("/notify", headers=HEADERS, json={"text": "hi"})
    assert response.status_code == 503
    assert response.json()["status"] == "not_confirmed" and "nothing was written" in response.json()["error"]
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path / "another-install")
    assert client.post("/notify", headers=HEADERS, json={"text": "hi"}).status_code == 503
    assert _rows(tmp_path) == []


def test_an_unconfirmed_write_is_the_same_503(tmp_path: pathlib.Path, chat, monkeypatch) -> None:
    from supervisor import message_bus

    def failing_send(*_args, **_kwargs):
        raise RuntimeError("canonical message acceptance could not be persisted")

    monkeypatch.setattr(message_bus, "send_with_budget", failing_send)
    client, _app = _client(tmp_path)
    response = client.post("/notify", headers=HEADERS, json={"text": "hi"})
    assert response.status_code == 503
    assert response.json() == {"ok": False, "status": "not_confirmed",
                               "error": "write not confirmed; check the owner's chat before sending it again"}


def test_the_next_turn_reads_a_notice_as_a_host_fact(tmp_path: pathlib.Path, chat) -> None:
    from tests._memory_view_context import room_text

    client, _app = _client(tmp_path)
    assert client.post("/notify", headers=HEADERS, json={"text": "Meeting in 15 min"}).status_code == 200
    rendered = room_text(tmp_path)  # the room of Main's next turn: a notice shows by its words
    [line] = [line for line in rendered.splitlines() if "Notice · cal" in line]
    assert "; host" in line.split("]", 1)[0] and "; Ouroboros;" not in line, line


def test_notify_token_discovery_does_not_hold_the_asgi_loop(tmp_path: pathlib.Path, chat, monkeypatch) -> None:
    import httpx

    _client_unused, app = _client(tmp_path)
    ctx = app.state.host_service_context
    authenticate = ctx.authenticate_token_payload
    started, release = threading.Event(), threading.Event()

    def slow_auth(token):
        started.set()
        assert release.wait(5), "test must release the simulated disk read"
        return authenticate(token)

    monkeypatch.setattr(ctx, "authenticate_token_payload", slow_auth)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            notice = asyncio.create_task(client.post("/notify", headers=HEADERS, json={"text": "test"}))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                # If auth runs inline, this coroutine cannot resume until release.
                assert not release.is_set()
            finally:
                release.set()
            assert (await notice).status_code == 200

    timer = threading.Timer(2, release.set)
    timer.start()  # fail boundedly instead of deadlocking if auth regresses to inline I/O
    try:
        asyncio.run(exercise())
    finally:
        release.set()
        timer.cancel()
        timer.join()
