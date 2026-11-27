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


@pytest.mark.parametrize(("cancel_mode", "copy_fails"), [("asyncio", False), ("anyio", True)])
def test_cancelled_inject_retains_copy_and_inflight_until_worker_settles(tmp_path, monkeypatch, copy_fails, cancel_mode):
    import anyio
    import asyncio
    import threading
    from types import SimpleNamespace
    from ouroboros.gateway import host_service

    source = _skill_file(tmp_path)
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    ctx = client.app.state.host_service_context
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original_copy, original_leave = host_service.store_upload, ctx._leave_inflight
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

    monkeypatch.setattr(host_service, "store_upload", copy)
    monkeypatch.setattr(ctx, "_leave_inflight", leave)
    request = SimpleNamespace(app=client.app, headers={"x-skill-token": "token"}, json=payload)
    scopes = []
    cancelled = []

    async def call():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await host_service._api_chat_inject(request)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    async def run():
        task = asyncio.create_task(call())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            scopes[0].cancel() if cancel_mode == "anyio" else task.cancel()
            await asyncio.sleep(0)
            if cancel_mode == "asyncio":
                task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and ctx._inflight["telegram"] == 1
            assert left == [] and not finished.is_set()
        finally:
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                assert cancel_mode == "asyncio"
        assert cancelled == [True]
        if cancel_mode == "anyio":
            assert scopes[0].cancelled_caught
        assert finished.is_set() and ctx._inflight["telegram"] == 0
        assert left == ["telegram"] and bridge.messages == []
        assert source.is_file()
        assert list((tmp_path / "uploads").glob("*")) == []

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


def test_inject_captures_file_above_former_upload_cap(tmp_path):
    from ouroboros.artifacts import stream_artifact_file

    source = _skill_file(tmp_path, payload=b"first bytes")
    with source.open("ab") as handle:
        handle.truncate(51 * 1024 * 1024)
    expected = stream_artifact_file(source)
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    response = client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                           json={"text": "", "chat_id": 42, "attachments": [{"path": str(source)}]})
    assert response.status_code == 202, response.json()
    stored = pathlib.Path(bridge.messages[0]["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert stream_artifact_file(stored) == expected
    assert source.exists()


def test_decision_route_requires_the_inject_grant(tmp_path):
    client = _client(tmp_path, FakeBridge(), permissions=())
    response = client.post("/chat/decision", headers={"X-Skill-Token": "token"},
                           json={"request_id": "tg:1", "decision_id": "quiz:t:q", "option_index": 0})
    assert response.status_code == 403


def test_decision_route_relays_to_the_one_decision_ingress(tmp_path, monkeypatch):
    import ouroboros.gateway.task_decision as td

    captured = {}

    async def fake_answer(drive_root, body, *, source="web", request=None):
        captured["drive_root"] = drive_root
        captured["body"] = body
        captured["source"] = source  # the relaying skill is the answer's transport, never "web"
        # the relay hands the ingress its own request: a model-wait card's saved choice records its origin
        captured["path"] = request.url.path if request is not None else None
        return 200, {"ok": True, "decision_id": body["decision_id"], "state": "answered", "answered_index": 1}

    monkeypatch.setattr(td, "answer_decision", fake_answer)
    client = _client(tmp_path, FakeBridge())
    body = {"request_id": "tg:99", "decision_id": "quiz:task-1:q1", "option_index": 1}
    response = client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json=body)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "decision_id": "quiz:task-1:q1", "state": "answered", "answered_index": 1}
    assert captured == {"drive_root": tmp_path, "body": body, "source": "skill:telegram", "path": "/chat/decision"}


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

    async def fake_answer(drive_root, body, *, source="web", request=None):
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


def test_decision_relay_keeps_only_the_recorded_winners_copy_of_a_reply_file(tmp_path, monkeypatch):
    """A Reply's file rides the bundled bridge's decision relay: stored like any owner upload, kept only while the
    recorded winner names it. A redelivered update or another request with the same bytes keeps no copy."""
    import ouroboros.chat_references as references
    import ouroboros.gateway.task_decision as td
    import supervisor.queue as q
    from ouroboros.owner_quiz import quiz_states, record_asked, record_publication

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A", "B"], assumption="A")
    record_publication(tmp_path, "task-1", "q1", 1)
    monkeypatch.setattr(references, "bundled_bridge", lambda root: True)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, **_: ({"id": task_id, "chat_id": 1}, ""))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: tmp_path, raising=False)
    client = _client(tmp_path, FakeBridge())

    def post(request_id, source):
        return client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json={
            "request_id": request_id, "decision_id": "quiz:task-1:q1", "comment": "this one",
            "attachments": [{"path": str(source), "name": "shot.png", "mime": "image/png"}]})

    first = post("tg:9:m:42:7", _skill_file(tmp_path, "a", b"\x89PNG\r\n\x1a\nfirst"))
    assert first.status_code == 200, first.json()
    assert [(item["name"], item["kind"]) for item in first.json()["attachments"]] == [("shot.png", "image")]
    kept = sorted((tmp_path / "uploads").iterdir())
    assert len(kept) == 1 and kept[0].read_bytes() == b"\x89PNG\r\n\x1a\nfirst"
    assert quiz_states(tmp_path, "task-1")["q1"]["attachments"][0]["upload"] == kept[0].name
    rejoin = client.post("/chat/decision", headers={"X-Skill-Token": "token"},
                         json={"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True})
    assert rejoin.status_code == 200 and rejoin.json()["duplicate"] is True  # the bridge's read before a new download
    assert post("tg:9:m:42:7", _skill_file(tmp_path, "b", b"\x89PNG\r\n\x1a\nfirst")).json()["duplicate"] is True
    assert post("tg:9:m:42:8", _skill_file(tmp_path, "c", b"\x89PNG\r\n\x1a\nfirst")).status_code == 409
    assert sorted((tmp_path / "uploads").iterdir()) == kept, "a replay or a losing request keeps no copy"

    monkeypatch.setattr(references, "bundled_bridge", lambda root: False)  # any other skill: the field is refused
    refused = post("tg:9:m:42:9", _skill_file(tmp_path, "d"))
    assert refused.status_code == 400 and refused.json()["reason_code"] == "unexpected_fields"
    assert client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json={
        "request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True}).status_code == 400
    assert sorted((tmp_path / "uploads").iterdir()) == kept


def _answer_relay(tmp_path, monkeypatch):
    import ouroboros.chat_references as references
    import ouroboros.gateway.task_decision as td
    import supervisor.queue as q
    from ouroboros.owner_quiz import record_asked, record_publication

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A", "B"], assumption="A")
    record_publication(tmp_path, "task-1", "q1", 1)
    monkeypatch.setattr(references, "bundled_bridge", lambda root: True)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, **_: ({"id": task_id, "chat_id": 1}, ""))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: tmp_path, raising=False)
    client = _client(tmp_path, FakeBridge())
    return lambda body: client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json=body)


def test_decision_relay_records_a_caption_with_a_file_the_bridge_never_received(tmp_path, monkeypatch):
    """The bundled bridge's Reply whose video was over its download limit: the caption and the typed fact are the one
    answer; nothing is uploaded."""
    from ouroboros.owner_quiz import quiz_states

    post = _answer_relay(tmp_path, monkeypatch)
    fact = {"name": "clip.mp4", "mime": "video/mp4", "size": 30_000_000, "unavailable": "too_large",
            "limit_bytes": 10_485_760}

    answered = post({"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "comment": "see the clip",
                     "attachments": [fact]})

    assert answered.status_code == 200, answered.json()
    assert answered.json()["comment"] == "see the clip"
    assert [(ref["name"], ref["unavailable"]) for ref in answered.json()["attachments"]] == [("clip.mp4", "too_large")]
    recorded = quiz_states(tmp_path, "task-1")["q1"]
    assert recorded["comment"] == "see the clip" and not recorded["attachments"][0].get("upload")
    assert not (tmp_path / "uploads").exists() or not list((tmp_path / "uploads").iterdir())


def test_rejoin_reads_the_recorded_winner_from_history_after_the_projection_let_it_go(tmp_path, monkeypatch):
    """A settled answer can leave the bounded question projection; its canonical history row still names the winner,
    so the bridge's rejoin and the copy settlement read it, and another request is a definite loser."""
    from ouroboros.gateway.task_decision import answer_holds_uploads
    from ouroboros.owner_quiz import _mutate_projection, quiz_states

    post = _answer_relay(tmp_path, monkeypatch)
    source = _skill_file(tmp_path, "a", b"\x89PNG\r\n\x1a\nfirst")
    assert post({"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1",
                 "attachments": [{"path": str(source), "name": "shot.png", "mime": "image/png"}]}).status_code == 200
    recorded = quiz_states(tmp_path, "task-1")["q1"]
    assert (recorded.get("delivery") or {}).get("history_at"), "the winner's history row was written"
    _mutate_projection(tmp_path, "task-1", lambda quizzes: quizzes.pop("q1"))  # as the cap evicts a settled answer
    assert "q1" not in quiz_states(tmp_path, "task-1")

    rejoin = post({"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True})
    other = post({"request_id": "tg:9:m:42:8", "decision_id": "quiz:task-1:q1", "rejoin": True})

    assert rejoin.status_code == 200 and rejoin.json()["duplicate"] is True
    assert [ref["name"] for ref in rejoin.json()["attachments"]] == ["shot.png"]
    assert other.status_code == 404 and other.json()["state"] == "answered"
    assert answer_holds_uploads(tmp_path, {"decision_id": "quiz:task-1:q1"}, recorded["attachments"])


def test_a_redelivered_reply_overlapping_the_first_before_its_answer_is_written_makes_one_answer(tmp_path, monkeypatch):
    """Telegram can deliver the same update again while the first attempt's answer is not yet written: both reach the
    host under the one request id. The record names ONE winner with its own file; the other replays that answer and
    keeps no copy. Nothing stands between them: no queue, journal or reservation."""
    import ouroboros.chat_references as references
    import ouroboros.gateway.task_decision as td
    import ouroboros.owner_quiz as owner_quiz
    import supervisor.queue as q
    from ouroboros.owner_quiz import quiz_states, record_asked, record_publication

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A", "B"], assumption="A")
    record_publication(tmp_path, "task-1", "q1", 1)
    monkeypatch.setattr(references, "bundled_bridge", lambda root: True)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, **_: ({"id": task_id, "chat_id": 1}, ""))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: tmp_path, raising=False)
    client = _client(tmp_path, FakeBridge())

    def post(source):
        return client.post("/chat/decision", headers={"X-Skill-Token": "token"}, json={
            "request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "comment": "this one",
            "attachments": [{"path": str(source), "name": "shot.png", "mime": "image/png"}]})

    real, overlapping = owner_quiz.record_answered, []

    def racing(*args, **kwargs):
        if not overlapping:  # the first attempt is about to write: the redelivery runs to its end first
            overlapping.append(None)  # marked before it runs, so its own write is the ordinary one
            overlapping[0] = post(_skill_file(tmp_path, "b", b"\x89PNG\r\n\x1a\nsecond"))
        return real(*args, **kwargs)

    monkeypatch.setattr(owner_quiz, "record_answered", racing)

    first = post(_skill_file(tmp_path, "a", b"\x89PNG\r\n\x1a\nfirst"))

    (second,) = overlapping
    assert second.status_code == 200 and second.json()["duplicate"] is False, "the redelivery wrote the answer"
    assert first.status_code == 200 and first.json()["duplicate"] is True, "the first attempt replays that one answer"
    recorded = quiz_states(tmp_path, "task-1")["q1"]
    assert recorded["state"] == "answered" and recorded["request_id"] == "tg:9:m:42:7"
    (winner,) = sorted((tmp_path / "uploads").iterdir())
    assert winner.read_bytes() == b"\x89PNG\r\n\x1a\nsecond" and [ref["upload"] for ref in recorded["attachments"]] == [winner.name]
    assert first.json()["attachments"] == second.json()["attachments"], "both receipts state the one winner's file"


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


@pytest.mark.parametrize("failure", ["missing_second", "bad_shape_second", "invalid_user"])
def test_unaccepted_inject_removes_only_its_new_copies(tmp_path, monkeypatch, failure):
    from ouroboros.gateway import host_service

    source = _skill_file(tmp_path)
    second = _skill_file(tmp_path, "second.pdf")
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    existing = uploads / "earlier-accepted.pdf"
    existing.write_bytes(b"already owned")
    body = {"text": "file", "chat_id": 42, "attachments": [{"path": str(source)}]}
    if failure == "missing_second":
        body["attachments"].append({"path": str(source.parent / "missing.pdf")})
    elif failure == "bad_shape_second":
        body["attachments"].append("not an object")
    elif failure == "copy_second":
        original = host_service.store_upload
        def fail_second(path, *args, **kwargs):
            if path == second:
                raise OSError("controlled second-copy failure")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(host_service, "store_upload", fail_second)
        body["attachments"].append({"path": str(second)})
    elif failure == "invalid_wait":
        body["wait_for_response"] = True
    else:
        body["user_id"] = "invalid"
    response = client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=body)
    assert response.status_code == (500 if failure in {"copy_second", "invalid_user"} else 400)
    assert bridge.messages == []
    assert list(uploads.iterdir()) == [existing]
    assert existing.read_bytes() == b"already owned"
    assert source.is_file() and second.is_file()
    assert client.app.state.host_service_context._inflight["telegram"] == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_accepted_inject_keeps_attachment_after_disconnect(tmp_path, cancel):
    import asyncio
    from types import SimpleNamespace
    from ouroboros.gateway import host_service
    from ouroboros.contracts.chat_id_policy import A2A_CHAT_ID_MAX

    source = _skill_file(tmp_path)
    bridge = FakeBridge()
    # The response stays pending; the message itself is already accepted.
    bridge.enqueue_local_message = lambda text, **kwargs: bridge.messages.append({"text": text, **kwargs})
    client = _client(tmp_path, bridge)
    async def body():
        return {"text": "file", "chat_id": A2A_CHAT_ID_MAX, "wait_for_response": True,
                "attachments": [{"path": str(source)}]}
    async def disconnected():
        assert len(bridge.messages) == 1
        if cancel:
            raise asyncio.CancelledError()
        return True
    request = SimpleNamespace(app=client.app, headers={"x-skill-token": "token"},
                              json=body, is_disconnected=disconnected)
    async def run():
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await host_service._api_chat_inject(request)
        else:
            response = await host_service._api_chat_inject(request)
            assert response.status_code == 499
    asyncio.run(run())
    stored = pathlib.Path(bridge.messages[0]["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert stored.read_bytes() == source.read_bytes()
    assert bridge._subs == {}
    assert client.app.state.host_service_context._inflight["telegram"] == 0
