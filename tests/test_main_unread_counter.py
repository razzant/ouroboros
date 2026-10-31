"""Main's number counts what a Project's counts: the dialogue store's room revision against a read cursor.

A standalone message moves the room's revision; a row without one (a task card's line) does not. The cursor
is a monotonic, server-clamped paint acknowledgement, null until a client first shows Main (owner Q1 = B,
19.10.2026: both sidebar numbers count unread standalone messages; DESIGN "Unread").
"""
from __future__ import annotations

from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from supervisor.message_bus import log_chat
from tests.test_main_state_projection import _state, gateway  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401


def _say(root, text, *, feed=True, chat_id=1):
    log_chat("out", chat_id, 1, text, drive_root=root, require_write=True, feed=feed)


def test_state_carries_mains_room_revision_moved_only_by_standalone_messages(gateway):  # noqa: F811
    assert _state(gateway)["main_visible_revision"] == 0
    _say(gateway.root, "An answer")
    _say(gateway.root, "A question")
    assert _state(gateway)["main_visible_revision"] == 2
    _say(gateway.root, "A line inside a task card", feed=False)
    assert _state(gateway)["main_visible_revision"] == 2
    _say(gateway.root, "A message in another room", chat_id=777)
    assert _state(gateway)["main_visible_revision"] == 2


def test_state_says_unknown_when_the_store_cannot_be_read(gateway, monkeypatch):  # noqa: F811
    from ouroboros import dialogue_store

    def unreadable(*_args, **_kwargs):
        raise dialogue_store.StoreError("busy")

    monkeypatch.setattr(dialogue_store, "room", unreadable)
    assert _state(gateway)["main_visible_revision"] is None


def test_mains_read_cursor_is_null_until_set_then_clamped_and_monotonic(tmp_path):
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        assert client.get("/api/ui/preferences").json()["main_seen_revision"] is None
        for text in ("one", "two", "three"):
            _say(tmp_path, text)
        assert client.post("/api/ui/preferences", json={"main_seen_revision": 1}).json()["main_seen_revision"] == 1
        # A future value is clamped to what Main holds; a stale tab cannot move it back; null moves nothing.
        assert client.post("/api/ui/preferences", json={"main_seen_revision": 999}).json()["main_seen_revision"] == 3
        assert client.post("/api/ui/preferences", json={"main_seen_revision": 0}).json()["main_seen_revision"] == 3
        assert client.post("/api/ui/preferences", json={"main_seen_revision": None}).json()["main_seen_revision"] == 3
        assert client.get("/api/ui/preferences").json()["main_seen_revision"] == 3
        for bad in ("3", 2.5, True):
            assert client.post("/api/ui/preferences", json={"main_seen_revision": bad}).status_code == 400


def test_an_unreadable_store_moves_no_cursor(tmp_path, monkeypatch):
    from ouroboros import dialogue_store

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        _say(tmp_path, "one")
        assert client.post("/api/ui/preferences", json={"main_seen_revision": 1}).json()["main_seen_revision"] == 1
        _say(tmp_path, "two")

        def unreadable(*_args, **_kwargs):
            raise dialogue_store.StoreError("busy")

        monkeypatch.setattr(dialogue_store, "room", unreadable)
        answer = client.post("/api/ui/preferences", json={"main_seen_revision": 2})
        assert answer.status_code == 200 and answer.json()["main_seen_revision"] == 1
