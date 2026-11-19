"""Room and document custody at the macOS launcher boundary. No GUI or server needed."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros.desktop_room_session import (
    FD_ENV, KEY_ENV, META, PRESENTATION_HEADER, REQUEST_HEADER, RESPONSE_HEADER, RoomSession, SessionRefused, document_ticket,
)
from ouroboros.server_web import ServedUI, TELEGRAM_MINIAPP_HEADER, UiSelection, make_index_page


def fixture_slots():
    frame = {"type": "chat", "content": "Sent once", "sender_session_id": "sender-first",
             "client_message_id": "message-first", "chat_id": 37, "project_id": "room",
             "attachments": [{"filename": "server-ref", "display_name": "sent.txt", "mime": "text/plain"}]}
    return {"ouro_chat_draft": "Main only", "ouro_chat_draft:37": "  Текст\n📝\t",
            "ouro_chat_reply:37": '{"chat_id":37,"message_id":"source-first"}',
            "ouro_chat_files:37": "2", "ouro_chat_session_id": "sender-first",
            "ouro_chat_active_project": "room", "ouro_chat_unconfirmed:37": json.dumps([{"frame": frame}])}


def test_only_room_slots_change_together(tmp_path):
    session = RoomSession(tmp_path)
    session.update("ouro_chat_session_id", "sender-first")
    assert not session.has_room_input and not session.needs_transfer
    session.apply(list(fixture_slots().items()))
    assert session.has_room_input and session.needs_transfer
    before = session.revision
    with pytest.raises(ValueError, match="room slot"):
        session.apply([["ouro_chat_draft", None], ["arbitrary storage", "value"]])
    assert session.items == fixture_slots() and session.revision == before


def test_source_descriptor_carries_exact_room_state_and_is_consumed(tmp_path, monkeypatch):
    pytest.importorskip("_posixshmem")
    session = RoomSession(tmp_path)
    session.apply(list(fixture_slots().items()))
    env = {}
    with session.prepare_exec() as (validate, pass_to_exec):
        validate()
        assert env == {}
        with pass_to_exec(env):
            fd, size = map(int, env[FD_ENV].split(":"))
            assert os.get_inheritable(fd) and set(env) == {FD_ENV}
            monkeypatch.setenv(FD_ENV, f"{os.dup(fd)}:{size}")  # the successor takes custody of this descriptor
            assert RoomSession.receive(tmp_path).items == fixture_slots()
            assert FD_ENV not in os.environ
            other = os.dup(fd)
            try:
                monkeypatch.setenv(FD_ENV, f"{other}:{size}")
                assert RoomSession.receive(tmp_path / "other").restore_failed
                os.fstat(other)  # a mismatched handoff does not close somebody else's descriptor
            finally:
                os.close(other)
        assert not os.get_inheritable(fd) and not env
    assert RoomSession.receive(tmp_path).items == {}  # a subsequent ordinary launch
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("number", ["", "1:30", "not-a-number:4", "3:0"])
def test_unreadable_handoff_starts_an_empty_session(tmp_path, monkeypatch, caplog, number):
    monkeypatch.setenv(FD_ENV, number)
    session = RoomSession.receive(tmp_path)
    assert session.restore_failed and session.items == {} and FD_ENV not in os.environ
    assert "Could not restore room input" in caplog.text


def test_unrecognized_payload_does_not_change_the_descriptor(tmp_path, monkeypatch, caplog):
    with (tmp_path / "ordinary-file").open("w+b") as file:
        file.write(b"private example")
        file.flush()
        at = file.tell()
        monkeypatch.setenv(FD_ENV, f"{file.fileno()}:{at}")
        assert RoomSession.receive(tmp_path).restore_failed
        assert file.tell() == at and "private example" not in caplog.text


def test_source_transfer_refuses_a_late_keystroke_or_an_unsaved_editor(tmp_path):
    pytest.importorskip("_posixshmem")
    session = RoomSession(tmp_path)
    with session.prepare_exec() as (_, transfer):
        session.update("ouro_chat_draft", "last key")
        with pytest.raises(SessionRefused, match="input changed"), transfer({}):
            pytest.fail("a changed draft reached teardown")
    with session.prepare_exec() as (_, transfer):
        session.edits.add("Files")
        with pytest.raises(SessionRefused, match="Files"), transfer({}):
            pytest.fail("an editor with unsaved text reached teardown")
    assert session.items["ouro_chat_draft"] == "last key"


def test_only_the_builtin_main_response_receives_a_navigation_ticket(tmp_path, monkeypatch):
    key, challenge = "a" * 64, "b" * 64
    builtin = tmp_path / "web"
    builtin.mkdir()
    (builtin / "index.html").write_text("<html><head></head><body>room</body></html>")
    index = make_index_page(builtin, desktop_key=key)
    request = lambda headers: SimpleNamespace(headers=headers)
    stamped = asyncio.run(index(request({REQUEST_HEADER: challenge})))
    ticket = document_ticket(key, challenge)
    assert ticket == hmac.digest(key.encode(), challenge.encode(), "sha256").hex()
    assert stamped.headers[RESPONSE_HEADER] == ticket and ticket.encode() in stamped.body
    assert PRESENTATION_HEADER not in stamped.headers
    assert document_ticket("c" * 64, challenge) != ticket
    for headers in ({}, {REQUEST_HEADER: "short"}, {REQUEST_HEADER: challenge, TELEGRAM_MINIAPP_HEADER: "1"}):
        response = asyncio.run(index(request(headers)))
        assert RESPONSE_HEADER not in response.headers and META.encode() not in response.body
    monkeypatch.setenv(KEY_ENV, key)
    serving = ServedUI(tmp_path)
    assert KEY_ENV not in os.environ
    directory = tmp_path / "custom"
    directory.mkdir()
    (directory / "index.html").write_text("<html>custom interface</html>")
    seen = {ticket}
    for selection, status in [(UiSelection("directory", directory), 200), (UiSelection("none"), 404)]:
        serving._current = SimpleNamespace(selection=selection)
        response = asyncio.run(serving.index_page(request({REQUEST_HEADER: challenge})))
        assert response.status_code == status and RESPONSE_HEADER not in response.headers
        proof = response.headers[PRESENTATION_HEADER]
        assert proof == document_ticket(key, challenge, source=selection.source)
        assert proof != document_ticket(key, "c" * 64, source=selection.source)
        assert proof != document_ticket("d" * 64, challenge, source=selection.source)
        assert proof not in seen
        seen.add(proof)
        for headers in ({}, {REQUEST_HEADER: "short"}, {REQUEST_HEADER: challenge, TELEGRAM_MINIAPP_HEADER: "1"}):
            ordinary = asyncio.run(serving.index_page(request(headers)))
            assert ordinary.status_code == status
            assert RESPONSE_HEADER not in ordinary.headers and PRESENTATION_HEADER not in ordinary.headers

    # This directory was legitimate when selected but can no longer serve its index.
    # Its 404 and the intentional no-interface page have different native outcomes.
    serving._current = SimpleNamespace(selection=UiSelection("directory", tmp_path / "missing"))
    missing = asyncio.run(serving.index_page(request({REQUEST_HEADER: challenge})))
    assert missing.status_code == 404
    assert RESPONSE_HEADER not in missing.headers and PRESENTATION_HEADER not in missing.headers
