"""``POST /api/ui/holds``: the holds a page found when it opened become one System row per owning chat.

A held app card is named in the chat it was published in, a skill's widget and page code in the owner's chat; a
``key@rev`` named once adds no row when another tab or device reports it again, and one being named right now waits
for that report; a row whose write is not confirmed is reported again; a body of the wrong shape answers 400 and
writes nothing; with no chat writer serving this data root nothing is written and the answer is 503.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.card_store import store_app_bytes, write_card
from ouroboros.gateway import ui_holds


@pytest.fixture
def rows(tmp_path, monkeypatch):
    from supervisor import message_bus, schedule_notes

    sent = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat_id, text, **kw: sent.append((chat_id, text, kw)))
    monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda _root: 1)
    monkeypatch.setattr(ui_holds, "_NAMED", type(ui_holds._NAMED)())
    return sent


@pytest.fixture
def client(tmp_path):
    app = Starlette(routes=[Route("/api/ui/holds", endpoint=ui_holds.api_ui_holds_post, methods=["POST"])])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _app_card(root, chat_id=7):
    return write_card(root, chat_id=chat_id, task_id="task-1", title="Котокулятор", fallback_text="A cat calculator",
                      kind="app", body={"app": store_app_bytes(root, b"<p>cats</p>")})


def test_the_found_holds_are_named_once_in_the_chats_that_own_them(tmp_path, client, rows):
    card = _app_card(tmp_path)
    body = {"holds": [{"key": f"app:{card['card_id']}", "rev": card["app"]["sha256"]},
                      {"key": "widget:kotokulator:calculator", "rev": "7"},
                      {"key": "module:device_voice:tts", "rev": "r1"}],
            "surface": {"pywebview": True, "telegram": False}}
    response = client.post("/api/ui/holds", json=body)
    assert response.status_code == 200 and response.json() == {"ok": True, "named": 3}
    by_chat = {chat_id: (text, kw) for chat_id, text, kw in rows}
    assert set(by_chat) == {7, 1}, "the card in its own chat, the skill's items in Main"
    card_text, card_kw = by_chat[7]
    assert card_text == "In the desktop app, the page stopped while the card «Котокулятор» was starting."
    assert card_kw == {"role": "system", "system_type": "ui_hold", "require_write": True, "ensure_record_boundary": True}
    assert "the calculator widget of kotokulator, the page code of device_voice were starting" in by_chat[1][0]
    again = client.post("/api/ui/holds", json=body)
    assert again.json() == {"ok": True, "named": 0} and len(rows) == 2, "another tab finding the same holds adds no row"
    newer = client.post("/api/ui/holds", json={"holds": [{"key": "widget:kotokulator:calculator", "rev": "8"}]})
    assert newer.json()["named"] == 1, "a new revision that stops the page again is a new fact"
    assert rows[-1][1].startswith("In a browser, the page stopped")


def test_a_card_published_in_another_room_is_named_there(tmp_path, client, rows):
    from ouroboros.card_store import record_publication

    card = _app_card(tmp_path, chat_id=7)
    record_publication(tmp_path, card["card_id"], 42)
    client.post("/api/ui/holds", json={"holds": [{"key": f"app:{card['card_id']}", "rev": card["app"]["sha256"]}]})
    assert [chat_id for chat_id, _text, _kw in rows] == [42], "the room the card was published in, not its origin"


def test_a_hold_whose_card_is_gone_names_nothing(tmp_path, client, rows):
    response = client.post("/api/ui/holds", json={"holds": [{"key": "app:card_missing", "rev": "x"}]})
    assert response.json() == {"ok": True, "named": 0} and rows == []


@pytest.mark.parametrize("body", [
    {"holds": "app:card_1"},
    {"holds": [{"key": "file:/etc/passwd", "rev": ""}]},
    {"holds": [{"rev": "x"}]},
    {"holds": [{"key": f"widget:s:t{i}", "rev": "1"} for i in range(ui_holds.MAX_REPORTED_HOLDS + 1)]},
    [],
])
def test_a_body_of_the_wrong_shape_is_refused_and_writes_nothing(client, rows, body):
    response = client.post("/api/ui/holds", json=body)
    assert response.status_code == 400 and rows == []


def test_a_row_whose_write_is_not_confirmed_is_reported_again(tmp_path, client, rows, monkeypatch):
    from supervisor import message_bus

    def refuse(chat_id, text, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(message_bus, "send_with_budget", refuse)
    response = client.post("/api/ui/holds", json={"holds": [{"key": "module:device_voice:tts", "rev": "r1"}]})
    assert response.status_code == 503 and response.json()["reason_code"] == "not_confirmed"
    assert ui_holds._NAMED == {}, "the page keeps it unreported, and the next report names it"


def test_rows_written_before_an_unconfirmed_one_are_not_written_again(tmp_path, client, rows, monkeypatch):
    from supervisor import message_bus

    card, chats = _app_card(tmp_path, chat_id=7), []

    def main_is_full(chat_id, text, **kw):
        chats.append(chat_id)
        if chat_id == 1:
            raise RuntimeError("disk full")

    monkeypatch.setattr(message_bus, "send_with_budget", main_is_full)
    body = {"holds": [{"key": f"app:{card['card_id']}", "rev": card["app"]["sha256"]},
                      {"key": "module:device_voice:tts", "rev": "r1"}]}
    assert client.post("/api/ui/holds", json=body).status_code == 503
    assert client.post("/api/ui/holds", json=body).status_code == 503
    assert chats == [7, 1, 1], "the card's row was written once; only the unconfirmed one is tried again"


def test_a_hold_another_report_is_naming_now_is_left_for_the_next_load(client, rows, monkeypatch):
    monkeypatch.setattr(ui_holds, "_NAMING", {"module:device_voice:tts@r1"})
    response = client.post("/api/ui/holds", json={"holds": [{"key": "module:device_voice:tts", "rev": "r1"}]})
    assert response.status_code == 503 and response.json()["reason_code"] == "in_flight" and rows == []
    monkeypatch.setattr(ui_holds, "_NAMING", set())
    assert client.post("/api/ui/holds", json={"holds": [{"key": "module:device_voice:tts", "rev": "r1"}]}).json() == {
        "ok": True, "named": 1}, "once that report settled, this one names it"


def test_without_a_chat_writer_nothing_is_written(tmp_path, client, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: None)
    monkeypatch.setattr(ui_holds, "_NAMED", type(ui_holds._NAMED)())
    response = client.post("/api/ui/holds", json={"holds": [{"key": "module:device_voice:tts", "rev": "r1"}]})
    assert response.status_code == 503 and response.json()["reason_code"] == "chat_unavailable"
    assert ui_holds._NAMED == {}, "a hold that was not named is named by the next report"
