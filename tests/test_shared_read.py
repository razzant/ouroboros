"""One message shown in two rooms is read once (DESIGN "Unread"; owner choice A).

A Project's notice to Main is one row counted unread in both rooms (``room_revisions``). Showing
it whole in one room records it read in each room it counted in (``read_ahead_revisions``)
without moving any room's cursor: the room's other unseen messages stay unread, and an unread
room no longer lands on it. The record is lossless and host-owned; only a room's own cursor
passing a revision drops it there. Read is neither an answer nor done.
"""
from __future__ import annotations

import queue
import types

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import dialogue_store as ds
from ouroboros.gateway.router import collect_routes
from ouroboros.tools.control_runtime import _send_user_message
from tests.test_main_notice import host as host_fixture
from tests.test_room_appearances import _born_in, _ctx, _notice, _project, _read

host = host_fixture  # the real supervisor handler, bus and store


def _unread(visible, seen, spans):
    """``web/modules/read_ahead.js::unreadCount``: revisions past the cursor, less those read elsewhere."""
    top, floor = max(0, visible), max(0, seen or 0)
    return max(0, top - floor - sum(max(0, min(top, hi) - max(floor + 1, lo) + 1) for lo, hi in spans))


@pytest.fixture
def rooms(tmp_path, host):
    nova = _project(tmp_path, "nova")
    _born_in(tmp_path, "nova-root", nova)
    for text in ("p1: first finding.", "p2: second finding."):
        q = queue.Queue()
        assert _send_user_message(_ctx(tmp_path, "nova-root", nova, q), text).startswith("OK:")
        host.handle(q.get_nowait())
    m3 = _notice(tmp_path, host, "nova-root", nova, "m3: the Titan series are done.")
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        yield types.SimpleNamespace(root=tmp_path, nova=nova, m3=m3, client=client, host=host)


def _prefs(rooms, **body):
    response = rooms.client.post("/api/ui/preferences", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _revision(root, chat):
    return int((ds.room(root, chat) or {}).get("revision") or 0)


def test_reading_m3_in_main_leaves_p1_p2_unread_and_the_project_lands_on_p2(rooms):
    nova = str(rooms.nova)
    assert rooms.m3["room_revisions"] == {"1": 1, nova: 3}
    assert (_revision(rooms.root, rooms.nova), _revision(rooms.root, 1)) == (3, 1)
    # Before any reading the Main copy is not silent: unread in Main and in Nova.
    assert _read(rooms.root, rooms.nova)[1]["window"]["latest_message"]["history_id"]
    prefs = _prefs(rooms, shared_read=[rooms.m3["row_address"]])
    assert prefs["read_ahead_revisions"] == {"1": [[1, 1]], nova: [[3, 3]]}
    assert prefs["project_seen_revision"] == {} and prefs["main_seen_revision"] is None  # no cursor moved
    spans = prefs["read_ahead_revisions"][nova]
    assert _unread(3, 0, spans) == 2  # p1 and p2 stay unread in Nova
    # Opening Nova lands on its newest unread message, p2, never again on m3.
    page = _read(rooms.root, rooms.nova)[1]
    by_id = {row.get("history_id"): row.get("text") for row in page["messages"]}
    assert by_id[page["window"]["latest_message"]["history_id"]] == "p2: second finding."
    # Read is not an answer: the work keeps running, nothing is delivered to it.
    from ouroboros.task_results import load_task_result
    assert load_task_result(rooms.root, "nova-root")["status"] == "running"


def test_reading_m3_in_the_project_reads_it_in_main_without_moving_mains_cursor(rooms, host):
    q = queue.Queue()
    later = _ctx(rooms.root, "main-turn", 1, q)
    later.task_metadata = {"origin_message_ref": {"chat_id": 1, "client_message_id": "m1"}}
    _prefs(rooms, main_seen_revision=0)
    ds.append(rooms.root, "chat", {"ts": "2026-10-28T07:00:00Z", "direction": "out", "chat_id": 1,
                                   "task_id": "main-turn", "text": "An unrelated Main reply."}, feed=True)
    prefs = _prefs(rooms, shared_read=[rooms.m3["row_address"]])
    assert prefs["main_seen_revision"] == 0 and prefs["read_ahead_revisions"]["1"] == [[1, 1]]
    assert _unread(_revision(rooms.root, 1), prefs["main_seen_revision"], prefs["read_ahead_revisions"]["1"]) == 1


def test_read_ahead_is_lossless_and_only_a_rooms_own_cursor_drops_it(rooms, host):
    """More distinct read facts than any bound (225, never two adjacent revisions), posted in request-sized
    batches with a stale client and repeats between them: none is evicted or merged away, no cursor moves,
    p1 and p2 stay unread, and only a room's own cursor drops what it covers."""
    nova = str(rooms.nova)
    addresses = [_notice(rooms.root, host, "nova-root", rooms.nova, f"notice {index}")["row_address"] for index in range(450)]
    read = addresses[1::2]  # 225 distinct shared messages; their revisions in each room are pairwise non-adjacent
    for start in range(0, len(read), 50):
        _prefs(rooms, shared_read=read[start:start + 50])
        stale = _prefs(rooms, project_seen_revision={"nova": 0})  # another client's older cursor in between
        assert stale["project_seen_revision"].get("nova", 0) == 0
        _prefs(rooms, shared_read=read[max(0, start - 30):start + 10])  # repeats of facts already held
    prefs = _prefs(rooms, shared_read=read[-3:])
    held = prefs["read_ahead_revisions"][nova]
    assert len(held) == 225 and all(lo == hi for lo, hi in held)  # every distinct fact kept, none merged
    assert len(prefs["read_ahead_revisions"]["1"]) == 225  # the same messages in Main, Main's cursor unset
    assert prefs["project_seen_revision"].get("nova", 0) == 0 and prefs["main_seen_revision"] is None
    visible = _revision(rooms.root, rooms.nova)
    assert visible == 3 + 450 and _unread(visible, 0, held) == visible - 225  # p1, p2, m3 and 225 notices unread
    cursor = held[100][0]
    after = _prefs(rooms, project_seen_revision={"nova": cursor})["read_ahead_revisions"]
    assert after[nova] == [span for span in held if span[0] > cursor] and len(after[nova]) == 124
    assert after["1"] == prefs["read_ahead_revisions"]["1"]  # Main's facts are Main's cursor's to drop
    from ouroboros.gateway.ui_preferences import _stored_preferences
    stored = _stored_preferences(rooms.root / "state" / "ui_preferences.json")
    assert stored["read_ahead_revisions"] == after  # the record survives a re-read


@pytest.mark.parametrize("body, reason", [
    ({"read_ahead_revisions": {"1": [[1, 9]]}}, "recorded by the host"),
    ({"shared_read": ["not-an-address"]}, "row addresses"),
    ({"shared_read": ["row:1@2026-10-28T00:00:00Z#abcdefabcdef"] * 65}, "at most 64"),
])
def test_read_facts_are_host_owned(rooms, body, reason):
    response = rooms.client.post("/api/ui/preferences", json=body)
    assert response.status_code == 400 and reason in response.text
    assert rooms.client.get("/api/ui/preferences").json()["read_ahead_revisions"] == {}


def test_a_message_counted_in_one_room_only_records_no_other_room(rooms):
    q = queue.Queue()
    assert _send_user_message(_ctx(rooms.root, "nova-root", rooms.nova, q), "p4: only in Nova.").startswith("OK:")
    rooms.host.handle(q.get_nowait())
    own = next(f for f in rooms.host.frames if f.get("content") == "p4: only in Nova.")
    assert "room_revisions" not in own  # ordinary speech: the room's own cursor reads it
    assert _prefs(rooms, shared_read=[own["row_address"]])["read_ahead_revisions"] == {}


def test_an_unreadable_store_never_fails_a_cursor_ack_or_drops_read_facts(rooms, monkeypatch):
    held = _prefs(rooms, shared_read=[rooms.m3["row_address"]])["read_ahead_revisions"]
    from ouroboros.store import StoreError

    def unreadable(*_args, **_kwargs):
        raise StoreError("dialogue store busy")
    monkeypatch.setattr(ds, "rooms", unreadable)
    acked = rooms.client.post("/api/ui/preferences", json={"project_seen_revision": {"nova": 3}})
    assert acked.status_code == 200 and acked.json()["read_ahead_revisions"] == held  # kept, not pruned blind
    assert acked.json()["project_seen_revision"]["nova"] == 3
    refused = rooms.client.post("/api/ui/preferences", json={"shared_read": [rooms.m3["row_address"]]})
    assert refused.status_code == 503  # a new read fact needs the store: the client offers it again


def test_a_torn_projects_registry_never_fails_mains_cursor_ack_or_drops_read_facts(rooms):
    from ouroboros import projects_registry as registry

    held = _prefs(rooms, shared_read=[rooms.m3["row_address"]])["read_ahead_revisions"]
    registry._registry_path(rooms.root).write_text("{torn", encoding="utf-8")
    acked = rooms.client.post("/api/ui/preferences", json={"main_seen_revision": 1})
    assert acked.status_code == 200, acked.text  # Main's own ACK never needed the registry
    assert acked.json()["main_seen_revision"] == 1 and acked.json()["read_ahead_revisions"] == held
