"""The owner-notification route of the Host Service (``POST /notify``).

Its own module beside the chat/presence suites: a notice is neither a chat
injection nor a presence receipt — one durable ``owner_notification`` events
row (never a chat row), the ``owner.notification`` topic, and with ``at``/
``cron`` a ``kind: "notify"`` row of the one schedule table.
"""

from __future__ import annotations

import pathlib

from starlette.testclient import TestClient

from ouroboros.gateway.host_service import create_host_service_app
from tests.test_host_service_api import _seed_token

def _events_rows(data_dir: pathlib.Path) -> list[dict]:
    import json

    path = data_dir / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _notify_client(tmp_path: pathlib.Path, *, granted: bool = True):
    _seed_token(
        tmp_path, skill="cal", token="tok",
        permissions=["notify_owner"] if granted else [],
        manifest_permissions=["notify_owner"],
    )
    app = create_host_service_app(tmp_path)
    return TestClient(app), app


def test_identity_advertises_the_notify_contract(tmp_path: pathlib.Path) -> None:
    client, _app = _notify_client(tmp_path)
    resp = client.get("/identity", headers={"X-Skill-Token": "tok"})
    assert resp.status_code == 200 and resp.json()["notify_version"] == 1


def test_notify_delivers_one_owner_notification_without_a_chat_row(tmp_path: pathlib.Path) -> None:
    from ouroboros import event_bus

    bus = event_bus.init_global_event_bus()
    seen: list[dict] = []
    bus.subscribe("telegram", event_bus.OWNER_NOTIFICATION, seen.append)
    try:
        client, _app = _notify_client(tmp_path)
        resp = client.post("/notify", headers={"X-Skill-Token": "tok"},
                           json={"text": "⏰ Meeting with Ivan in 15 min", "key": "cal:evt-1", "source": "evil"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True and body["chat_id"] == 1 and body["ts"]
        rows = [row for row in _events_rows(tmp_path) if row.get("type") == "owner_notification"]
        assert len(rows) == 1
        assert rows[0]["source"] == "skill:cal", "the host stamps the source, never the skill"
        assert rows[0]["key"] == "cal:evt-1" and rows[0]["text"] == "⏰ Meeting with Ivan in 15 min"
        assert rows[0]["ts"] == body["ts"] and "task_id" not in rows[0]
        assert not (tmp_path / "logs" / "chat.jsonl").exists()
        assert len(seen) == 1 and seen[0]["topic"] == "owner.notification"
    finally:
        event_bus.init_global_event_bus()


def test_notify_requires_the_notify_owner_grant(tmp_path: pathlib.Path) -> None:
    client, _app = _notify_client(tmp_path, granted=False)
    resp = client.post("/notify", headers={"X-Skill-Token": "tok"}, json={"text": "hi"})
    assert resp.status_code == 403 and "notify_owner" in resp.json()["error"]
    assert client.post("/notify", json={"text": "hi"}).status_code == 403
    assert [r for r in _events_rows(tmp_path) if r.get("type") == "owner_notification"] == []


def test_notify_refuses_bad_bodies_and_the_rate_limit(tmp_path: pathlib.Path) -> None:
    client, app = _notify_client(tmp_path)
    headers = {"X-Skill-Token": "tok"}
    assert client.post("/notify", headers=headers, content=b"not json").status_code == 400
    assert client.post("/notify", headers=headers, json=["x"]).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "   "}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "x" * 1001}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "hi", "key": 5}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "hi", "key": "k" * 129}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "hi", "cancel": "yes"}).status_code == 400
    assert [r for r in _events_rows(tmp_path) if r.get("type") == "owner_notification"] == []
    app.state.host_service_context.rate_limiter.allow = lambda key: False
    assert client.post("/notify", headers=headers, json={"text": "hi"}).status_code == 429


def test_notify_reports_a_failed_durable_write_as_503(tmp_path: pathlib.Path, monkeypatch) -> None:
    from ouroboros import utils

    client, _app = _notify_client(tmp_path)
    monkeypatch.setattr(utils, "append_jsonl", lambda *a, **k: False)
    resp = client.post("/notify", headers={"X-Skill-Token": "tok"}, json={"text": "hi"})
    assert resp.status_code == 503 and "retry" in resp.json()["error"]


def test_notify_with_at_schedules_a_notify_row_that_a_key_moves_and_cancels(tmp_path: pathlib.Path) -> None:
    from supervisor import queue

    queue.init(tmp_path)
    client, _app = _notify_client(tmp_path)
    headers = {"X-Skill-Token": "tok"}
    first = client.post("/notify", headers=headers, json={
        "text": "Meeting with Ivan", "key": "cal:evt-1", "at": "2999-01-01T14:45:00+00:00"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["ok"] is True and body["scheduled"] is True and body["next_run_at"].startswith("2999-01-01T14:45")
    rows = queue.list_scheduled_tasks(tmp_path)["tasks"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == body["id"] and row["kind"] == "notify" and row["source"] == "skill:cal"
    assert row["notification"] == {"text": "Meeting with Ivan", "key": "cal:evt-1"}
    assert row["trigger"] == {"type": "once", "run_at": "2999-01-01T14:45:00+00:00"}
    assert "task" not in row and "skill" not in row, "a notify row is neither a task nor skill-managed"
    assert [r for r in _events_rows(tmp_path) if r.get("type") == "owner_notification"] == [], "nothing fired yet"
    # The same key moves the reminder: one row, the new instant, the new text.
    moved = client.post("/notify", headers=headers, json={
        "text": "Meeting with Ivan (moved)", "key": "cal:evt-1", "at": "2999-01-02T10:00:00+00:00"})
    assert moved.status_code == 200 and moved.json()["id"] == body["id"]
    rows = queue.list_scheduled_tasks(tmp_path)["tasks"]
    assert len(rows) == 1 and rows[0]["trigger"]["run_at"].startswith("2999-01-02T10:00")
    assert rows[0]["notification"]["text"] == "Meeting with Ivan (moved)"
    # Cancel by key removes it through the audited delete; an unknown key is 404.
    gone = client.post("/notify", headers=headers, json={"text": "x", "key": "cal:evt-1", "cancel": True})
    assert gone.status_code == 200 and gone.json()["cancelled"] is True
    assert queue.list_scheduled_tasks(tmp_path)["tasks"] == []
    assert client.post("/notify", headers=headers, json={"text": "x", "key": "cal:evt-1", "cancel": True}).status_code == 404
    # A cron reminder from a skill rides the same row kind.
    cron = client.post("/notify", headers=headers, json={"text": "Standup", "key": "standup", "cron": "0 9 * * 1-5", "timezone": "Europe/Moscow"})
    assert cron.status_code == 200 and cron.json()["scheduled"] is True
    row = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert row["trigger"] == {"type": "cron", "expr": "0 9 * * 1-5"} and row["timezone"] == "Europe/Moscow"


def test_notify_scheduling_validation(tmp_path: pathlib.Path) -> None:
    from supervisor import queue

    queue.init(tmp_path)
    client, _app = _notify_client(tmp_path)
    headers = {"X-Skill-Token": "tok"}
    assert client.post("/notify", headers=headers, json={"text": "x", "at": "soon"}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "x", "at": "2999-01-01T00:00:00Z", "cron": "* * * * *"}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "x", "cron": "bad"}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "x", "cron": "* * * * *", "timezone": "Mars/Olympus"}).status_code == 400
    assert client.post("/notify", headers=headers, json={"text": "x", "cancel": True}).status_code == 400, "cancel needs a key"
    assert client.post("/notify", headers=headers, json={"text": "x", "key": "k", "cancel": True, "at": "2999-01-01T00:00:00Z"}).status_code == 400
    assert queue.list_scheduled_tasks(tmp_path)["tasks"] == []
    # Without a key a row is fire-and-forget: two posts, two rows.
    for _ in range(2):
        assert client.post("/notify", headers=headers, json={"text": "x", "at": "2999-01-01T00:00:00Z"}).status_code == 200
    assert len(queue.list_scheduled_tasks(tmp_path)["tasks"]) == 2


def test_events_websocket_allows_manifest_declared_owner_notification_without_grant(tmp_path: pathlib.Path) -> None:
    """The host's own owner notifications are not conversation content: a
    companion that declares the topic may subscribe without an owner grant,
    exactly like skill.lifecycle."""
    from ouroboros import event_bus

    _seed_token(tmp_path, skill="mirror", token="mtok", permissions=[], subscribe_events=["owner.notification"])
    app = create_host_service_app(tmp_path)
    client = TestClient(app)
    event_bus.init_global_event_bus()
    try:
        with client.websocket_connect("/events", headers={"X-Skill-Token": "mtok"}) as ws:
            ws.send_json({"type": "subscribe", "topic": "owner.notification"})
            assert ws.receive_json()["type"] == "subscribed"
            event_bus.publish_event("owner.notification", {"text": "hello", "source": "skill:cal"})
            message = ws.receive_json()
        assert message["type"] == "event" and message["topic"] == "owner.notification"
        assert message["data"]["text"] == "hello"
    finally:
        event_bus.init_global_event_bus()


def test_owner_disable_or_delete_of_a_notify_row_survives_the_skill_reposting_its_key(tmp_path: pathlib.Path) -> None:
    from supervisor import queue

    queue.init(tmp_path)
    client, _app = _notify_client(tmp_path)
    headers = {"X-Skill-Token": "tok"}
    body = {"text": "Meeting", "key": "cal:evt-9", "at": "2999-01-01T14:45:00+00:00"}
    first = client.post("/notify", headers=headers, json=body)
    assert first.status_code == 200 and first.json()["scheduled"] is True
    schedule_id = first.json()["id"]

    # The owner switches the reminder off: the row keeps a marker, the skill's
    # repeat (a moved time, a new text) is answered honestly and changes nothing.
    outcome = queue.mutate_scheduled_task("disable", schedule_id, reason="owner: not this one", actor="owner:gateway", drive_root=tmp_path)
    assert outcome["ok"] is True and outcome["status"] == "updated"
    again = client.post("/notify", headers=headers, json={**body, "text": "Meeting (moved)", "at": "2999-01-02T10:00:00+00:00"})
    assert again.status_code == 200 and again.json() == {"ok": True, "scheduled": False, "id": schedule_id, "status": "suppressed"}
    row = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert row["enabled"] is False and row["manual_override"] == "disabled"
    assert row["notification"]["text"] == "Meeting" and row["trigger"]["run_at"].startswith("2999-01-01T14:45")
    assert queue.schedule_lifecycle_status(row) == "suppressed"

    # Only the owner's restore lifts it; then the skill's next post moves it again.
    restored = queue.mutate_scheduled_task("restore", schedule_id, reason="owner: changed my mind", actor="owner:gateway", drive_root=tmp_path)
    assert restored["ok"] is True and restored["status"] == "updated"
    row = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert row["enabled"] is True and "manual_override" not in row
    moved = client.post("/notify", headers=headers, json={**body, "at": "2999-01-03T10:00:00+00:00"})
    assert moved.status_code == 200 and moved.json()["scheduled"] is True
    assert queue.list_scheduled_tasks(tmp_path)["tasks"][0]["trigger"]["run_at"].startswith("2999-01-03T10:00")

    # A delete is retained as a suppressed record for the same reason; the
    # skill's own cancel still removes what it may.
    deleted = queue.mutate_scheduled_task("delete", schedule_id, reason="owner: gone", actor="owner:gateway", drive_root=tmp_path)
    assert deleted["ok"] is True and deleted["status"] == "suppressed"
    assert client.post("/notify", headers=headers, json=body).json()["status"] == "suppressed"
    rows = queue.list_scheduled_tasks(tmp_path)["tasks"]
    assert len(rows) == 1 and rows[0]["manual_override"] == "deleted" and rows[0]["enabled"] is False
    # A second owner delete of the suppressed record removes it for good.
    gone = queue.mutate_scheduled_task("delete", schedule_id, reason="owner: clean up", actor="owner:gateway", drive_root=tmp_path)
    assert gone["ok"] is True and gone["status"] == "deleted"
    assert queue.list_scheduled_tasks(tmp_path)["tasks"] == []


def test_scheduled_notify_upsert_refuses_a_row_owned_by_another_source(tmp_path: pathlib.Path) -> None:
    from supervisor import queue

    queue.init(tmp_path)
    client, _app = _notify_client(tmp_path)
    from ouroboros.gateway.host_service import _notify_schedule_id

    queue.upsert_scheduled_task({
        "id": _notify_schedule_id("cal", "shared"), "name": "someone else's", "kind": "notify",
        "source": "skill:other", "enabled": True,
        "trigger": {"type": "once", "run_at": "2999-01-01T00:00:00+00:00"},
        "notification": {"text": "theirs", "key": "shared"},
    })
    resp = client.post("/notify", headers={"X-Skill-Token": "tok"},
                       json={"text": "mine", "key": "shared", "at": "2999-01-01T00:00:00+00:00"})
    assert resp.status_code == 409
    assert queue.list_scheduled_tasks(tmp_path)["tasks"][0]["notification"]["text"] == "theirs"
