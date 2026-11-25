"""A newest read that fails names its room while the core still counts it among this profile's ordinary rooms.

Beside its reason, the history door's 503 says what the core knows of the room the read asked for
(``history_paging.failed_read_facts`` over ``room_archive.ordinary_room``, the same readiness answer Main's history
decides by): Main while the Project registry tells every room apart, a Project room while its one registry row is
active, each with the serving generation's frozen ``profile_scope``. The page shows the sends it kept for that exact
room on this word alone; a room the core no longer counts is named no more. A Send again standing on that word names
the room (``ChatInbound.ordinary_room``), and the ordered acceptance asks the word again before anything is saved or
handed over, refusing by the typed ``reason_code`` ``ordinary_room_unconfirmed``. Built on the synthetic world of
``tests._room_archive_world``; every word below is fictional.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.gateway import history
from ouroboros.gateway import ws as gateway_ws
from ouroboros.gateway.history_contracts import ChatHistoryResponse
from ouroboros.gateway.schema import validate_ingress
from ouroboros.store import StoreError
from tests._room_archive_world import TS, lost_registry, project_row, write_registry
from tests._task_settings_helpers import isolated_runtime  # noqa: F401 -- the sealed install every case runs in

SCOPE = "0123456789abcdef"


@pytest.fixture
def root(isolated_runtime):  # noqa: F811 -- the sealed install's data root
    """Main and the Project room ``alpha``, an owner's words in each, beside the readable registry an older body kept."""
    lost_registry(isolated_runtime)
    for chat_id, text in ((1, "main words"), (project_chat_id("alpha"), "alpha words")):
        ds.append(isolated_runtime, "chat", {"direction": "in", "task_id": "", "ts": TS, "chat_id": chat_id, "text": text})
    write_registry(isolated_runtime, [project_row("alpha", "Alpha room")])
    return isolated_runtime


def _busy(*_args, **_kwargs):
    raise StoreError("store busy")


def _newest_read(root, **params):
    """One newest read through the history door of a generation that serves ``SCOPE``."""
    served = SimpleNamespace(serving_profile_scope=lambda: SCOPE)
    request = SimpleNamespace(query_params=params, app=SimpleNamespace(state=SimpleNamespace(served_ui=served)))
    response = asyncio.run(history.make_chat_history_endpoint(root)(request))
    body = json.loads(response.body)
    assert not validate_ingress(body, ChatHistoryResponse), validate_ingress(body, ChatHistoryResponse)
    return response.status_code, body


def test_mains_unreadable_newest_read_names_main_under_the_profile(root, monkeypatch):
    monkeypatch.setattr(history, "_assemble_history_response", _busy)
    status, body = _newest_read(root)
    assert status == 503 and body["reason_code"] == "history_source_unavailable" and body["detail"] == "store busy"
    assert body["ordinary_room"] == 1 and body["profile_scope"] == SCOPE and "project_registry" not in body


def test_an_active_project_rooms_unreadable_newest_read_names_that_room(root, monkeypatch):
    monkeypatch.setattr(history, "_assemble_history_response", _busy)
    alpha = project_chat_id("alpha")
    status, body = _newest_read(root, chat_id=alpha)
    assert status == 503 and body["reason_code"] == "history_source_unavailable"
    assert body["ordinary_room"] == alpha and body["profile_scope"] == SCOPE


def test_a_room_the_core_stops_counting_is_named_no_more(root, monkeypatch):
    """Its Project starts deleting; later the registry is lost, and Main's refusal offers the archive instead."""
    alpha = project_chat_id("alpha")
    write_registry(root, [project_row("alpha", "Alpha room", lifecycle="deleting")])
    with monkeypatch.context() as patch:
        patch.setattr(history, "_assemble_history_response", _busy)
        status, body = _newest_read(root, chat_id=alpha)
        assert status == 503 and "ordinary_room" not in body and "profile_scope" not in body
        assert _newest_read(root)[1]["ordinary_room"] == 1, "the registry still tells Main apart"
    (root / "state" / "projects.json").unlink()
    status, body = _newest_read(root)
    assert status == 503 and body["project_registry"] == "unavailable"
    assert "ordinary_room" not in body and "profile_scope" not in body


@pytest.mark.parametrize("named, claimed, serving", [
    (1, SCOPE, ""),  # the connection proves no profile
    (1, "", SCOPE),  # the page claims none
    (2, SCOPE, SCOPE),  # the word named another room
])
def test_a_send_again_stands_only_on_this_rooms_word_under_this_connections_own_profile(
        root, monkeypatch, named, claimed, serving):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    alpha, handed = project_chat_id("alpha"), []
    with pytest.raises(gateway_ws._RoomWordLapsed):
        gateway_ws._accept_on_room_word(lambda: handed.append("refused"), 1, named, claimed, serving)
    gateway_ws._accept_on_room_word(lambda: handed.append("main"), 1, 1, SCOPE, SCOPE)
    gateway_ws._accept_on_room_word(lambda: handed.append("alpha"), alpha, alpha, SCOPE, SCOPE)
    assert handed == ["main", "alpha"], "only a word the core still holds hands the frame over"


def _socket(root, monkeypatch):
    """The real ``/ws`` of a generation serving ``SCOPE``, before a bridge that records each frame handed to it."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from starlette.testclient import TestClient
    from supervisor import message_bus

    handed = []
    bridge = SimpleNamespace(ui_send=lambda payload, **kwargs: handed.append((payload, kwargs["client_message_id"])))
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    app = Starlette(routes=[WebSocketRoute("/ws", endpoint=gateway_ws.ws_endpoint)])
    app.state.served_ui = SimpleNamespace(serving_profile_scope=lambda: SCOPE)
    return TestClient(app), handed


def _handed(handed, count, seconds=5.0):
    deadline = time.monotonic() + seconds
    while len(handed) < count and time.monotonic() < deadline:
        time.sleep(0.05)
    return list(handed)


def test_the_acceptance_asks_the_rooms_word_again_before_a_send_again_is_saved(root, monkeypatch):
    """The Project starts deleting after its word was read: the Send again standing on it is refused in the ordered
    acceptance by its typed cause, its id named, and nothing reaches the writer. In Main, an ordinary room, an
    ordinary frame and a Send again on Main's word that still holds are handed over in their order."""
    alpha = project_chat_id("alpha")
    write_registry(root, [project_row("alpha", "Alpha room", lifecycle="deleting")])
    client, handed = _socket(root, monkeypatch)
    main = {"type": "chat", "profile_scope": SCOPE}
    with client, client.websocket_connect("/ws") as conn:
        assert json.loads(conn.receive_text()) == {"type": "connection_profile", "profile_scope": SCOPE}
        conn.send_text(json.dumps({**main, "content": "kept words", "chat_id": alpha, "client_message_id": "cm-again",
                                   "ordinary_room": alpha}))
        refused = json.loads(conn.receive_text())
        assert (refused["system_type"], refused["reason_code"]) == ("initialization_notice", "ordinary_room_unconfirmed")
        assert (refused["client_message_id"], refused["chat_id"]) == ("cm-again", alpha)
        conn.send_text(json.dumps({**main, "content": "main words", "client_message_id": "cm-main"}))
        conn.send_text(json.dumps({**main, "content": "main again", "client_message_id": "cm-main-again",
                                   "ordinary_room": 1}))
    assert _handed(handed, 2) == [("main words", "cm-main"), ("main again", "cm-main-again")]
