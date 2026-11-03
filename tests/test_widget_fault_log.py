"""``POST /api/widgets/fault``: a fault a running module widget's card shows becomes one ``widget_fault`` row in the
existing ``logs/events.jsonl``, read back through ``GET /api/logs/events``.

The row is the page's report under the mount's identity, redacted before its bytes are kept; it carries no task, so
no task's context reads it and the live sink pushes it unaddressed; nothing else is written (no chat row, no start
hold). A body of another shape answers 400 and a body past 8 KiB 413, writing nothing. Rows are bounded per widget
and for all widgets together per window; a window replenishes when it ends, and the next row counts the faults its
widget lost meanwhile; the widgets remembered are bounded too.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway import widgets
from ouroboros.gateway.router import collect_routes

REVISION = "3f1c" * 16  # a live loader content hash
FAULT = {"skill": "kotokulator", "tab_id": "calculator", "revision": REVISION, "where": "board", "kind": "error",
         "message": "TypeError: x is undefined", "source": "about:srcdoc", "line": 17}


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(widgets, "_FAULT_WINDOWS", type(widgets._FAULT_WINDOWS)())
    monkeypatch.setattr(widgets, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


@pytest.fixture
def client(tmp_path, clock):
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as test_client:
        yield test_client


def _rows(client):
    response = client.get("/api/logs/events", params={"limit": 2000})
    assert response.status_code == 200
    return [row for row in response.json()["entries"] if row.get("type") == "widget_fault"]


def test_a_fault_is_one_redacted_task_less_row_read_back_through_the_events_log(tmp_path, client):
    from ouroboros.gateway import ui_holds
    from ouroboros.memory import Memory

    holds_before = list(ui_holds._NAMED)
    files_before = {path for path in tmp_path.rglob("*") if path.is_file()}
    secret = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0" * 2
    body = {**FAULT, "where": "message", "kind": "csp", "message": f"blocked with {secret}",
            "source": "https://user:hunter2@example.com/x.js", "line": 0}
    response = client.post("/api/widgets/fault", json=body)
    assert response.status_code == 200 and response.json() == {"ok": True, "logged": True}

    written = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()} - {
        path.relative_to(tmp_path).as_posix() for path in files_before}
    assert {name for name in written if not name.startswith("logs/.append_jsonl_")} == {"logs/events.jsonl"}, (
        "the one row and its append lock: no chat row, no task, no hold")
    retained = (tmp_path / "logs" / "events.jsonl").read_bytes()
    assert secret.encode() not in retained and b"hunter2" not in retained, "redacted before its bytes are kept"
    (row,) = _rows(client)
    assert {key: row[key] for key in ("skill", "tab_id", "revision", "where", "kind", "line", "reported_by")} == {
        "skill": "kotokulator", "tab_id": "calculator", "revision": REVISION, "where": "message", "kind": "csp",
        "line": 0, "reported_by": "page"}
    assert row["message"].startswith("blocked with ") and "***REDACTED***" in row["message"]
    assert "suppressed_count" not in row and "task_id" not in row and "chat_id" not in row

    entries, _coverage = Memory(drive_root=tmp_path).read_task_recent("events.jsonl", "task-1", 200)
    assert entries == [], "no task's own events window reads a task-less row"
    assert list(ui_holds._NAMED) == holds_before, "a fault is not a start hold"


def test_the_live_sink_pushes_the_row_unaddressed_and_nothing_else(tmp_path, client):
    from ouroboros import utils
    from supervisor.log_addressing import make_server_log_sink

    pushed = []

    class Bridge:
        def push_log(self, event):
            pushed.append(event)

    prior = utils._log_sink
    utils.set_log_sink(make_server_log_sink(Bridge(), tmp_path, running={}))
    try:
        assert client.post("/api/widgets/fault", json=FAULT).json()["logged"] is True
    finally:
        utils.set_log_sink(prior)
    assert [event["type"] for event in pushed] == ["widget_fault"]
    assert not {"task_id", "chat_id", "parent_task_id", "root_task_id"} & set(pushed[0]), (
        "an unaddressed log frame: the page's card and notification readers act only on task frames")


@pytest.mark.parametrize("change", [
    {"kind": "script"}, {"where": "panel"}, {"skill": "../x"}, {"skill": "kotokulator\n"}, {"tab_id": ""},
    {"revision": "r" * 129}, {"revision": 7}, {"message": "m" * 501}, {"source": "s" * 201},
    {"line": True}, {"line": 1.5}, {"line": -1}, {"line": 10_000_001}, {"line": "17"}, {"message": None},
])
def test_a_report_of_another_shape_answers_400_and_writes_nothing(tmp_path, client, change):
    response = client.post("/api/widgets/fault", json={**FAULT, **change})
    assert response.status_code == 400, response.text
    assert not (tmp_path / "logs" / "events.jsonl").exists()


@pytest.mark.parametrize("body", [b"not json", b"[]", json.dumps({k: v for k, v in FAULT.items() if k != "line"}).encode()])
def test_a_body_that_is_not_a_report_answers_400(tmp_path, client, body):
    response = client.post("/api/widgets/fault", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert not (tmp_path / "logs" / "events.jsonl").exists()


def test_the_parent_character_limits_fit_even_when_json_escapes_every_character(client):
    body = {**FAULT, "message": "\x00" * 500, "source": "\x00" * 200}
    response = client.post("/api/widgets/fault", json=body)
    assert response.status_code == 200, response.text
    (row,) = _rows(client)
    assert row["message"] == body["message"] and row["source"] == body["source"]


@pytest.mark.parametrize("text, retained", [
    ("x" * 499 + "\ud83d", "x" * 499 + "\ufffd"),
    ("\ude00 middle \ud83d", "\ufffd middle \ufffd"),
    ("intact \U0001f600", "intact \U0001f600"),
])
def test_utf16_clipping_preserves_the_report_and_valid_emoji(client, text, retained):
    # JSON.stringify escapes a lone surrogate that String.slice cut from an emoji.
    body = {**FAULT, "message": text, "source": "\ud83d", "kind": "rejection"}
    response = client.post("/api/widgets/fault", content=json.dumps(body),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 200, response.text
    (row,) = _rows(client)
    assert row["message"] == retained and row["source"] == "\ufffd"


def test_a_body_past_eight_kilobytes_answers_413_and_writes_nothing(tmp_path, client):
    padded = json.dumps({**FAULT, "padding": "p" * 9000})
    response = client.post("/api/widgets/fault", content=padded, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert not (tmp_path / "logs" / "events.jsonl").exists()


def test_a_widget_window_bounds_rows_replenishes_and_the_next_row_counts_what_was_lost(client, clock):
    answers = [client.post("/api/widgets/fault", json={**FAULT, "message": f"fault {i}"}).json()["logged"]
               for i in range(widgets.FAULT_ROWS_PER_WIDGET + 3)]
    assert answers == [True] * widgets.FAULT_ROWS_PER_WIDGET + [False] * 3
    other = client.post("/api/widgets/fault", json={**FAULT, "tab_id": "clock"})
    assert other.json()["logged"] is True, "another widget has its own window"
    clock[0] += widgets.FAULT_WINDOW_SEC - 1
    assert client.post("/api/widgets/fault", json=FAULT).json()["logged"] is False
    clock[0] += 1
    assert client.post("/api/widgets/fault", json=FAULT).json()["logged"] is True, "the window replenished"
    rows = [row for row in _rows(client) if row["tab_id"] == "calculator"]
    assert len(rows) == widgets.FAULT_ROWS_PER_WIDGET + 1
    assert rows[-1]["suppressed_count"] == 4 and all("suppressed_count" not in row for row in rows[:-1])
    assert client.post("/api/widgets/fault", json=FAULT).json()["logged"] is True
    assert "suppressed_count" not in _rows(client)[-1], "a count is carried once"


def test_a_new_revision_is_the_same_widget_for_the_window(client):
    for i in range(widgets.FAULT_ROWS_PER_WIDGET):
        client.post("/api/widgets/fault", json={**FAULT, "revision": f"rev{i}"})
    assert client.post("/api/widgets/fault", json={**FAULT, "revision": "rev-next"}).json()["logged"] is False


def test_all_widgets_together_are_bounded_per_window(client, clock):
    logged = [client.post("/api/widgets/fault", json={**FAULT, "tab_id": f"w{i}"}).json()["logged"]
              for i in range(widgets.FAULT_ROWS_PER_WINDOW + 5)]
    assert logged.count(True) == widgets.FAULT_ROWS_PER_WINDOW and logged[-5:] == [False] * 5
    clock[0] += widgets.FAULT_WINDOW_SEC
    assert client.post("/api/widgets/fault", json={**FAULT, "tab_id": "late"}).json()["logged"] is True


def test_the_widgets_remembered_are_bounded(client, monkeypatch):
    monkeypatch.setattr(widgets, "FAULT_ROWS_PER_WINDOW", 10_000)
    for i in range(widgets._FAULT_WIDGETS_MAX + 40):
        client.post("/api/widgets/fault", json={**FAULT, "tab_id": f"w{i}"})
    assert len(widgets._FAULT_WINDOWS) == widgets._FAULT_WIDGETS_MAX + 1, "the widgets plus the shared window"
    assert "" in widgets._FAULT_WINDOWS and "kotokulator:w0" not in widgets._FAULT_WINDOWS


def test_an_unwritten_row_answers_503(client, monkeypatch):
    from ouroboros import utils

    monkeypatch.setattr(utils, "append_jsonl", lambda *_args, **_kwargs: False)
    response = client.post("/api/widgets/fault", json=FAULT)
    assert response.status_code == 503
