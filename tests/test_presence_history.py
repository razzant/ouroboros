"""History-only Presence ingress at the real Host writer and the real room readers.

A synthetic connector posts earlier messages to ``POST /presence/history``; the checks read
back through the canonical dialogue store, ``/presence/turn``, room history, the Places rooms
read, the wake observation, reentry wording and memory facts. No provider, model or network.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import dialogue_store as ds
from ouroboros import places as p
from ouroboros.gateway import host_service
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.place_arrivals import reading_key
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_runner import _stable_numeric_id, presence_turn_task_id

KEY = conversation_key("chat", "a", "r", "")
CHAT = _stable_numeric_id("presence-conversation", KEY)
OLD = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)


def at(minutes: float) -> str:
    return (OLD + timedelta(minutes=minutes)).isoformat()


def event(n, text=None, *, actor="person", sent=None, message=None, conversation=None, source_event_id=None):
    return {"source_event_id": source_event_id or f"chat:a:r:0:{n}", "provider": "chat", "account_id": "a",
            "conversation_id": "r", "thread_id": "", "conversation_key": KEY,
            "actor": {"platform_actor_id": actor, "display_name": actor.title()},
            "conversation": {"title": "Room R"} if conversation is None else conversation,
            "message": {"message_id": str(n), "sent_at": sent or at(n), **(message or {})},
            "text": f"message {n}" if text is None else text}


def record(n, *, authorship="participant", sent=None, **kw):
    return {"event": event(n, sent=sent, **kw), "authorship": authorship, "source_timestamp": sent or at(n)}


def body(*records, import_id="imp_test"):
    return {"import_id": import_id, "records": list(records)}


@pytest.fixture
def room(tmp_path, monkeypatch):
    place = p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r",
                            wakes=["mention"])
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")

    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    runs = []

    def run(**kw):
        runs.append(kw["event"].source_event_id)
        return SimpleNamespace(outcome="message", text="Hello", task_id="focus", work_ref="")

    def client():
        return TestClient(host_service.create_host_service_app(tmp_path, presence_runner=run))
    state = SimpleNamespace(root=tmp_path, place=place, runs=runs, client=client, http=client())
    yield state
    state.http.close()


def rows(root):
    return [entry.row for entry in ds.iter_rows(root, "chat")]


def live(state, n, *, facts=("mention",), **kw):
    payload = {"event": {**event(n, **kw), "facts": list(facts)}}
    return state.http.post("/presence/turn", json=payload)


def files(root):
    return {path.relative_to(root).as_posix(): path.stat().st_mtime_ns for path in Path(root).rglob("*") if path.is_file()}


def history_client(root):
    return TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(root))]))


def test_capability_and_presence_grant_use_the_real_token_boundary(tmp_path):
    from tests.test_host_service_api import _seed_token
    _seed_token(tmp_path, skill="bridge", token="tok", permissions=["presence"], manifest_permissions=["presence"])
    _seed_token(tmp_path, skill="ungranted", token="other", permissions=[], manifest_permissions=["presence"])
    p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r")
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        assert client.get("/identity", headers={"x-skill-token": "tok"}).json()["presence_history_import_version"] == 1
        refused = client.post("/presence/history", headers={"x-skill-token": "other"}, json=body(record(1)))
        assert refused.status_code == 403 and refused.json()["code"] == "presence_auth_blocked"
        assert refused.json()["disposition"] == "blocked"
        missing = client.post("/presence/history", json=body(record(1)))
        assert missing.status_code == 403
        assert not rows(tmp_path)
        accepted = client.post("/presence/history", headers={"x-skill-token": "tok"}, json=body(record(1)))
        assert accepted.status_code == 200, accepted.text
    assert [row["text"] for row in rows(tmp_path)] == ["message 1"]


@pytest.mark.parametrize("change,field", [
    (lambda b: b["records"][0]["event"].update(facts=["mention"]), "records[0].event"),
    (lambda b: b["records"][0]["event"].update(identity={"sends_as": {"id": "w", "label": "W", "kind": "user"}}),
     "records[0].event"),
    (lambda b: b["records"][0]["event"]["actor"].update(authenticated="yes"), "records[0].event.actor.authenticated"),
    (lambda b: b["records"][0].update(staged_files=["x.jpg"]), "records[0].staged_files"),
    (lambda b: b["records"][0].update(authorship="owner"), "records[0].authorship"),
    (lambda b: b["records"][0].update(source_timestamp="2025-03-01T10:00:00"), "records[0].source_timestamp"),
    (lambda b: b["records"][0].update(source_timestamp=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()),
     "records[0].source_timestamp"),
    (lambda b: b["records"][0]["event"]["message"].update(sent_at=at(9)), "records[0].source_timestamp"),
    (lambda b: b["records"][0]["event"]["actor"].pop("platform_actor_id"), "records[0].event.actor.platform_actor_id"),
    (lambda b: b["records"].append(record(2, source_event_id="chat:a:other:0:2") | {"event": {
        **event(2), "conversation_id": "other"}}), "records[1].event"),
    (lambda b: b.update(records=[record(n) for n in range(101)]), "records"),
    (lambda b: b.update(import_id=" padded "), "import_id"),
    (lambda b: b.update(extra=True), "payload"),
    # Instants that leave the calendar once moved to UTC, and text that is no Unicode: never a retryable outage.
    (lambda b: b["records"][0].update(source_timestamp="0001-01-01T00:00:00+05:00"), "records[0].source_timestamp"),
    (lambda b: b["records"][0].update(source_timestamp="9999-12-31T23:59:59-05:00"), "records[0].source_timestamp"),
    (lambda b: b["records"][0]["event"]["message"].update(sent_at="0001-01-01T00:00:00+05:00"),
     "records[0].source_timestamp"),
    (lambda b: b["records"][0]["event"].update(text="half \ud800 a pair"), "records[0].event"),
    (lambda b: b["records"][0]["event"]["actor"].update(display_name="\udfff"), "records[0].event"),
    (lambda b: b.update(import_id="imp\ud83d"), "import_id"),
])
def test_malformed_requests_record_nothing(room, change, field):
    payload = body(record(1))
    change(payload)
    # JSON text as a connector sends it (a lone surrogate travels as its ``\ud800`` escape).
    response = room.http.post("/presence/history", content=json.dumps(payload).encode(),
                              headers={"content-type": "application/json"})
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "presence_history_invalid" and response.json()["disposition"] == "rejected"
    assert response.json()["field"] == field and "results" not in response.json()
    assert not rows(room.root)


def test_history_takes_the_actor_fact_a_live_event_takes(room):
    """``actor.authenticated`` is a boolean on both doors (CREATING_SKILLS); an imported row keeps it as sent."""
    imported = record(1)
    imported["event"]["actor"]["authenticated"] = False
    assert room.http.post("/presence/history", json=body(imported)).status_code == 200
    assert [row["transport"]["actor"].get("authenticated") for row in rows(room.root)] == [False]


@pytest.mark.parametrize("raw", [b'{"import_id": "\xff\xfe", "records": []}', b"\xff\xfe{",
                                 b"[" * 100_000 + b"]" * 100_000], ids=["not-utf8", "utf16-bom", "nested"])
def test_a_body_that_is_not_json_text_in_utf8_is_refused_not_retried(room, raw):
    """A connector resends a ``retry`` answer unchanged, forever: a body no resend can repair answers 400."""
    response = room.http.post("/presence/history", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400, response.text
    assert (response.json()["code"], response.json()["disposition"], response.json()["field"]) == (
        "presence_history_invalid", "rejected", "payload")
    assert not rows(room.root)


def test_an_edit_time_outside_the_calendar_compares_as_its_text(room):
    """``message.edited_at`` is a compared source fact; one that leaves the calendar in UTC stays its own text,
    so the record lands, its resend answers duplicate and a different text conflicts, never a 503."""
    odd = record(1, message={"edited_at": "0001-01-01T00:00:00+05:00"})
    assert room.http.post("/presence/history", json=body(odd)).status_code == 200
    again = room.http.post("/presence/history", json=body(odd))
    assert again.status_code == 200 and again.json()["results"][0]["duplicate"] is True
    other = room.http.post("/presence/history", json=body(record(1, message={"edited_at": "0001-01-01T00:00:00+06:00"})))
    assert other.status_code == 409 and other.json()["code"] == "presence_event_identity_conflict"
    assert len(rows(room.root)) == 1


def test_place_and_binding_refusals_record_nothing(room):
    other = p.connect_place(room.root, connector="elsewhere", provider="chat", account_id="z", conversation_id="q")
    outside = body(record(1) | {"event": {**event(1), "conversation_id": "unbound"}})
    response = room.http.post("/presence/history", json=outside)
    assert response.status_code == 409 and response.json()["code"] == "place_not_connected"
    assert response.json()["disposition"] == "blocked"
    foreign = room.http.post("/presence/history", json={**body(record(1)), "binding_id": other.place_id})
    assert foreign.status_code == 409 and foreign.json()["code"] == "presence_binding_wrong_transport"
    assert foreign.json()["disposition"] == "rejected"
    mismatch = room.http.post("/presence/history", json={**outside, "binding_id": room.place.place_id})
    assert mismatch.status_code == 403 and mismatch.json()["code"] == "presence_origin_mismatch"
    assert not rows(room.root)


def test_imported_rows_keep_dates_and_authors_and_cause_nothing_else(room):
    from ouroboros.dialogue_provenance import dialogue_author
    from ouroboros.owner_mailbox import _mailbox_path
    # The connected account's own messages are platform history even when that account is
    # also confirmed as the owner's: history never carries owner identity or a command.
    p.edit_place(room.root, room.place.place_id, "confirm_owner_account", author="owner", platform_actor_id="a",
                 account_id="a")
    before = files(room.root)
    response = room.http.post("/presence/history", json=body(
        record(1), record(2, authorship="platform_account", actor="a", message={"reply_to_message_id": 1})))
    assert response.status_code == 200, response.text
    first, second = response.json()["results"]
    assert first["custody"] == second["custody"] == "history" and not first["duplicate"]
    assert first["ts"] == at(1) and first["received_at"] > first["ts"]
    assert first["turn_ref"] == presence_turn_task_id(room.place.place_id, "chat:a:r:0:1")
    imported = rows(room.root)
    assert [row["ts"] for row in imported] == [at(1), at(2)]
    for row, authorship, actor in zip(imported, ("participant", "platform_account"), ("person", "a")):
        assert row["direction"] == "in" and row["type"] == "place_arrival" and row["chat_id"] == CHAT
        assert row["sender_session_id"] == actor and row["source"] == "presence:chat"
        assert row["historical_import"]["authorship"] == authorship
        assert row["historical_import"]["import_id"] == "imp_test" and row["historical_import"]["received_at"]
        assert row["client_message_id"] == f"chat:a:r:0:{row['transport']['message']['message_id']}"
    assert "sent from the connected account itself" in dialogue_author(imported[1])
    # Only the dialogue store (and its locks) changed: no task, mailbox, outbox, notice or state.
    changed = {name for name, stamp in files(room.root).items() if before.get(name) != stamp}
    assert changed and all(name.startswith("state/") and ("dialog" in name or "presence_turn_gate" in name)
                           for name in changed), changed
    with ds.transaction(room.root, write=False) as conn:
        stored = conn.execute("SELECT dispatch, dispatch_input FROM dialogue WHERE stream='chat'").fetchall()
        revision = conn.execute("SELECT revision FROM rooms WHERE chat_id=?", (CHAT,)).fetchone()[0]
    assert {(row[0], row[1]) for row in stored} == {("none", None)} and revision == 0
    assert not ds.pending_dispatches(room.root) and not room.runs
    assert not _mailbox_path(room.root, first["turn_ref"]).exists()
    with history_client(room.root) as client:
        page = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
    shown = {row["text"]: row for row in page["messages"]}
    for text, authorship in (("message 1", "participant"), ("message 2", "platform_account")):
        facts = shown[text]["place_history"]
        assert shown[text]["role"] == "user" and facts["historical_import"]["authorship"] == authorship
        assert "owner_confirmed" not in facts and "attention" not in facts


def test_retries_conflicts_and_display_enrichment(room):
    assert room.http.post("/presence/history", json=body(record(1))).status_code == 200
    stored = rows(room.root)
    same = room.http.post("/presence/history", json=body(record(1), import_id="imp_other"))
    assert same.status_code == 200 and same.json()["results"][0]["duplicate"] is True
    assert same.json()["results"][0]["custody"] == "history"
    enriched = record(1, conversation={"title": "Renamed"}, message={"username": "ann", "media": {
        "kind": "photo", "media_id": "9", "status": "staged"}})
    enriched["event"]["actor"]["display_name"] = "Ann Renamed"
    media = record(3, message={"media": {"kind": "photo", "media_id": "9", "status": "not_downloaded"}})
    assert room.http.post("/presence/history", json=body(media)).status_code == 200
    restaged = record(3, message={"media": {"kind": "photo", "media_id": "9", "status": "staged", "note": "x"}})
    assert room.http.post("/presence/history", json=body(restaged)).json()["results"][0]["duplicate"] is True
    assert room.http.post("/presence/history", json=body(enriched)).status_code == 409  # media appeared: a source fact
    enriched["event"]["message"].pop("media")
    assert room.http.post("/presence/history", json=body(enriched)).json()["results"][0]["duplicate"] is True
    for conflict in (record(1, text="changed"), record(1, message={"reply_to_message_id": 7}),
                     record(1, authorship="platform_account"), record(1, sent=at(30)),
                     record(1, message={"event_kind": "edit", "revision": "r2"}), record(1, actor="someone")):
        response = room.http.post("/presence/history", json=body(conflict))
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "presence_event_identity_conflict"
        result, = response.json()["results"]
        assert result["ok"] is False and result["disposition"] == "rejected"
    assert rows(room.root)[0] == stored[0] and len(rows(room.root)) == 2


def test_mixed_batch_conflict_is_409_with_ordered_partial_acceptance(room):
    assert room.http.post("/presence/history", json=body(record(1))).status_code == 200
    response = room.http.post("/presence/history", json=body(record(2), record(1, text="changed"), record(3)))
    assert response.status_code == 409
    data = response.json()
    assert data["ok"] is False and data["disposition"] == "rejected" and data["import_id"] == "imp_test"
    assert [(r["source_event_id"], r["ok"]) for r in data["results"]] == [
        ("chat:a:r:0:2", True), ("chat:a:r:0:1", False), ("chat:a:r:0:3", True)]
    assert [row["text"] for row in rows(room.root)] == ["message 1", "message 2", "message 3"]


def test_unconfirmed_write_is_503_and_the_unchanged_retry_completes(room, monkeypatch):
    assert room.http.post("/presence/history", json=body(record(1))).status_code == 200
    real, calls = ds.append, []

    def flaky(*args, **kwargs):
        calls.append(args[2].get("client_message_id"))
        if len(calls) == 2:
            raise ds.StoreBusy("dialogue store busy", reason="contention")
        return real(*args, **kwargs)
    monkeypatch.setattr(ds, "append", flaky)
    batch = body(record(1, text="changed"), record(2), record(3), record(4))
    response = room.http.post("/presence/history", json=batch)
    assert response.status_code == 503
    data = response.json()
    assert data["code"] == "presence_history_unwritable" and data["disposition"] == "retry"
    assert [(r["ok"], r.get("code"), r.get("disposition")) for r in data["results"]] == [
        (False, "presence_event_identity_conflict", "rejected"), (True, None, None),
        (False, "presence_history_unwritable", "retry"), (False, "presence_history_unwritable", "retry")]
    assert "not confirmed" in data["results"][2]["error"] and "not attempted" in data["results"][3]["error"]
    assert calls == ["chat:a:r:0:2", "chat:a:r:0:3"]
    monkeypatch.setattr(ds, "append", real)
    retry = room.http.post("/presence/history", json=body(record(2), record(3), record(4)))
    assert retry.status_code == 200
    assert [r["duplicate"] for r in retry.json()["results"]] == [True, False, False]
    assert [row["text"] for row in rows(room.root)] == ["message 1", "message 2", "message 3", "message 4"]


def test_import_of_a_live_capture_keeps_live_custody(room):
    assert live(room, 1).json()["outcome"] == "message" and room.runs == ["chat:a:r:0:1"]
    captured = rows(room.root)
    duplicate = room.http.post("/presence/history", json=body(record(1)))
    assert duplicate.status_code == 200
    result, = duplicate.json()["results"]
    assert (result["ok"], result["duplicate"], result["custody"]) == (True, True, "live")
    assert result["turn_ref"] == captured[0]["task_id"] and result["ts"] == result["received_at"] == captured[0]["ts"]
    # The live row's receipt time is never its send time; its recorded send time is compared.
    assert room.http.post("/presence/history", json=body(record(1, sent=at(45)))).status_code == 409
    assert room.http.post("/presence/history", json=body(record(1, text="other"))).status_code == 409
    assert rows(room.root) == captured and room.runs == ["chat:a:r:0:1"]
    unknown = {"event": {**event(2), "facts": ["mention"], "message": {"message_id": "2"}}}
    assert room.http.post("/presence/turn", json=unknown).status_code == 200
    stored = rows(room.root)[-1]
    imported = room.http.post("/presence/history", json=body(record(2) | {"event": {**event(2), "message": {
        "message_id": "2"}}}))
    assert imported.json()["results"][0]["custody"] == "live"  # unknown send time stays unknown
    assert rows(room.root)[-1] == stored and "historical_import" not in stored


def test_live_replay_of_an_imported_revision_is_silent_before_attention(room):
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_results import load_task_result
    assert room.http.post("/presence/history", json=body(record(1))).status_code == 200
    imported = rows(room.root)
    before = files(room.root)
    replay = live(room, 1)  # Active place, matching wake, consciousness on, allowance available
    assert replay.status_code == 200 and replay.json()["outcome"] == "silent", replay.text
    assert room.runs == [] and rows(room.root) == imported
    assert load_task_result(room.root, imported[0]["task_id"]) is None
    changed = {name for name, stamp in files(room.root).items() if before.get(name) != stamp}
    assert all(name.startswith("state/") and ("dialog" in name or "presence_turn_gate" in name)
               for name in changed), changed  # no task, mailbox, outbox, notice or attention record
    lease = PresenceTurnGate(state_root=room.root / "state").acquire(KEY)
    try:  # a current author would otherwise receive the arrival as context mail
        lease.bind_author("current-author")
        lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=threading.RLock(),
                                             _accepting_owner_messages=True, _busy=True,
                                             _current_task_id="current-author")
        with room.client() as restarted:  # a new Host context: the durable marker alone answers
            again = restarted.post("/presence/turn", json={"event": {**event(1), "facts": ["mention"]}})
        assert again.status_code == 200 and again.json()["outcome"] == "silent"
        assert not _mailbox_path(room.root, "current-author").exists()
    finally:
        lease.release()
    changed = live(room, 1, text="different words")
    assert changed.status_code == 409 and changed.json()["code"] == "presence_event_identity_conflict"
    edit = live(room, 1, text="edited words", source_event_id="chat:a:r:0:1:edit:r2",
                message={"event_kind": "edit", "revision": "r2"})
    assert edit.status_code == 200 and edit.json()["outcome"] == "message"
    fresh = live(room, 5)
    assert fresh.json()["outcome"] == "message"
    assert room.runs == ["chat:a:r:0:1:edit:r2", "chat:a:r:0:5"]
    assert [row["text"] for row in rows(room.root)] == ["message 1", "edited words", "message 5"]
    # History of a later revision beside its live original: a new record, then a silent replay.
    revision = record(5, text="final words", source_event_id="chat:a:r:0:5:edit:r9",
                      message={"event_kind": "edit", "revision": "r9", "edited_at": at(90)})
    stored = room.http.post("/presence/history", json=body(revision)).json()["results"][0]
    assert (stored["custody"], stored["duplicate"]) == ("history", False)
    replayed = room.http.post("/presence/turn", json={"event": {**revision["event"], "facts": ["mention"]}})
    assert replayed.json()["outcome"] == "silent" and room.runs == ["chat:a:r:0:1:edit:r2", "chat:a:r:0:5"]


def test_concurrent_identical_imports_store_one_row(room):
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_history import parse_history_request, record_history
    admission = admit_presence_turn(drive_root=room.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    request = parse_history_request(body(record(1)))
    gate, answers = threading.Barrier(2), []

    def post():
        gate.wait()
        answers.append(record_history(room.root, admission, request))
    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(status for status, _ in answers) == [200, 200]
    assert sorted(answer["results"][0]["duplicate"] for _, answer in answers) == [False, True]
    assert len(rows(room.root)) == 1


def test_a_live_capture_racing_its_import_keeps_one_row_and_one_custody(room):
    start = threading.Barrier(2)
    answers = {}

    def capture():  # each side has its own Host context, as two processes would
        with room.client() as client:
            start.wait()
            answers["live"] = client.post("/presence/turn", json={"event": {**event(7), "facts": ["mention"]}})

    def history():
        with room.client() as client:
            start.wait()
            answers["history"] = client.post("/presence/history", json=body(record(7)))
    threads = [threading.Thread(target=capture), threading.Thread(target=history)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    stored, = rows(room.root)
    result = answers["history"].json()["results"][0]
    if "historical_import" in stored:  # the import landed first: the live delivery was heard silent
        assert result["custody"] == "history" and answers["live"].json()["outcome"] == "silent" and not room.runs
    else:  # live capture landed first and keeps its custody
        assert result["custody"] == "live" and result["duplicate"] and room.runs == ["chat:a:r:0:7"]


def _pages(client, **params):
    pages, cursor = [], None
    while True:
        query = {"chat_id": CHAT, "n_human": 100, **params, **({"cursor": cursor} if cursor else {})}
        page = client.get("/api/chat/history", params=query).json()
        assert "error" not in page, page
        pages.append(page)
        if not page["has_more"]:
            return pages
        cursor = page["next_cursor"]


def _decode(cursor):
    return json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))


def _coordinates(pages):
    """Each delivered chat row's coverage coordinate, read off its page's span: a span holds exactly the
    rows its page delivered (the page reaching the beginning closes at 0), the newest just below its end."""
    found = {}
    for page in pages:
        span = page["coverage"]["spans"]["chat"]
        shown = [row["history_id"] for row in page["messages"] if row.get("history_id")]
        assert 0 <= span["from"] <= span["to"] and (span["from"] == 0 or span["to"] - span["from"] == len(shown)), span
        for step, history_id in enumerate(reversed(shown)):
            assert history_id not in found, history_id  # delivered once
            found[history_id] = span["to"] - 1 - step
    return found


def test_place_room_reads_chronologically_across_more_than_1000_imported_rows(room):
    now = datetime.now(timezone.utc)
    for n in range(50):  # recent live arrivals first
        assert live(room, 10_000 + n, sent=(now - timedelta(minutes=60 - n)).isoformat(), facts=()).status_code == 200
    for start in range(1100, -1, -100):  # then 1,200 earlier messages imported after them, newest first:
        batch = body(*(record(n) for n in range(start + 99, start - 1, -1)))  # stored order runs against time
        assert room.http.post("/presence/history", json=batch).status_code == 200
    for n in range(5):  # and newer live arrivals after the import
        sent = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        assert live(room, 20_000 + n, sent=sent, facts=()).status_code == 200
    stored = list(ds.iter_rows(room.root, "chat"))
    chronological = [entry.history_id for entry in sorted(stored, key=lambda entry: reading_key(entry.row, entry.seq))]
    end = ds.stream_end(room.root, "chat")
    with history_client(room.root) as client:
        pages = _pages(client)
        first = pages[0]
        newest = [row["history_id"] for row in first["messages"] if row.get("history_id")]
        assert set(newest) == set(chronological[-100:])
        assert [row["text"] for row in first["messages"]][-5:] == [f"message {20_000 + n}" for n in range(5)]
        delivered = [row["history_id"] for page in reversed(pages) for row in page["messages"] if row.get("history_id")]
        assert delivered == chronological and len(delivered) == 1255
        spans = [page["coverage"]["spans"]["chat"] for page in pages]
        # Coverage counts the frozen rows down from a fixed horizon in reading order: each span holds its rows.
        assert first["coverage"]["upper"]["chat"] == spans[0]["to"] == 1 << 52
        assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:])) and spans[-1]["from"] == 0
        assert len({span["chain"] for span in spans}) == 1 and spans[0]["chain"].endswith(f"-{end}")
        assert all(not span["gaps"] for span in spans)
        assert len(_coordinates(pages)) == 1255
        # Physical cursors stay physical: every page of the chain freezes the same stored end and continues
        # below a stored row's own key (its time and position), never a count or an offset.
        assert {_decode(page["next_cursor"])["upper"]["chat"] for page in pages[:-1]} == {end}
        for page in pages[1:-1]:
            oldest = page["messages"][0]
            assert _decode(page["next_cursor"])["chrono"]["before"] == [oldest["ts"], None, oldest["history_position"]["offset"]]
        assert all(row["history_id"].startswith("chat:") for row in first["messages"] if row.get("history_id"))
        replayed = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": pages[1]["page_cursor"]}).json()
        assert [row["history_id"] for row in replayed["messages"]] == [row["history_id"] for row in pages[1]["messages"]]
        middle = stored[50 + 600].row
        around = client.get("/api/chat/history", params={"chat_id": CHAT, "around": "task:" + middle["task_id"]}).json()
        assert around["anchor"]["state"] == "found"
        texts = [row["text"] for row in around["messages"]]
        assert texts == sorted(texts, key=lambda text: int(text.split()[1])) and middle["text"] in texts
        # New live rows continue the open window; newly imported earlier ones rebuild it.
        assert live(room, 30_000, facts=()).status_code == 200
        arrived = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": first["after_cursor"]}).json()
        assert [row["text"] for row in arrived["messages"] if row.get("history_id")] == ["message 30000"]
        assert room.http.post("/presence/history", json=body(record(5000))).status_code == 200
        reset = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": arrived["after_cursor"]})
        assert reset.status_code == 409 and reset.json()["reason_code"] == "history_delta_reset"
    if target := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        out = Path(target)
        out.mkdir(parents=True, exist_ok=True)
        (out / "presence-history-chronological-pages.json").write_text(json.dumps({
            "synthetic": True, "stream_end": end, "rows": len(stored),
            "pages": [{"has_more": page["has_more"], "coverage": page["coverage"],
                       "first": page["messages"][0]["ts"], "last": page["messages"][-1]["ts"],
                       "count": len(page["messages"])} for page in pages]}, indent=2))


BULK_CHAT = (1 << 40) + 4242
SHARED = "2025-01-01T00:00:00+00:00"  # a provider time of one second, as many imported messages share


def _bulk_room(root, size, shared=lambda _n: False):
    """One place room of ``size`` rows from a live chain file (one cutover): 50 live arrivals, then earlier
    messages imported newest first ("old 0" the oldest), then 5 newer live arrivals; every row has its own
    turn. Imported rows ``shared`` picks all carry ``SHARED``, the oldest time, stored oldest first."""
    def arrival(text, ts, **extra):
        return {"ts": ts, "direction": "in", "chat_id": BULK_CHAT, "type": "place_arrival", "text": text,
                "source": "presence:chat", "client_message_id": text, "task_id": "turn-" + text.replace(" ", "-"),
                "place_id": "place", "sender_session_id": "person", **extra}
    live_rows = [arrival(f"live {n}", f"2026-10-26T08:{n // 60:02d}:{n % 60:02d}.{n + 1:06d}+00:00") for n in range(55)]
    imported = [arrival(f"old {n}", SHARED if shared(n) else (OLD + timedelta(minutes=n)).isoformat(),
                        historical_import={"version": 1, "authorship": "participant",
                                           "received_at": "2026-10-26T09:00:00+00:00"})
                for n in [*(n for n in range(size - 55) if shared(n)), *(n for n in reversed(range(size - 55))
                                                                        if not shared(n))]]
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in
                                                      live_rows[:50] + imported + live_rows[50:]))
    ds.ensure(root)


def _reading_order(root):
    return [entry.history_id for entry in sorted(ds.iter_rows(root, "chat"), key=lambda entry: reading_key(entry.row, entry.seq))]


def _read(root, quota, cursor=None, around=None):
    from ouroboros.gateway import history
    return json.loads(history._assemble_history_response(root, BULK_CHAT, quota, 0, cursor, around))


def _chain(root, quota, around=None):
    """A first page (the newest, or one around ``around``) and every older page after it."""
    pages = [_read(root, quota, around=around)]
    while pages[-1]["has_more"]:
        pages.append(_read(root, quota, pages[-1]["next_cursor"]))
    return pages


def _delivered(pages):
    return [row["history_id"] for page in reversed(pages) for row in page["messages"] if row.get("history_id")]


def _metered(monkeypatch, read):
    """Statements and SQLite VM instructions of one read (``tests/test_dialogue_store.py`` counts the same)."""
    from ouroboros import store
    counts, real = {"statements": 0, "steps": 0}, store.connect

    def connect(*args, **kwargs):
        conn = real(*args, **kwargs)
        conn.set_trace_callback(lambda _statement: counts.__setitem__("statements", counts["statements"] + 1))
        conn.set_progress_handler(lambda: counts.__setitem__("steps", counts["steps"] + 1) or 0, 1)
        return conn
    store.close_connections()
    with monkeypatch.context() as patch:
        patch.setattr(store, "connect", connect)
        result = read()
    store.close_connections()
    return counts, result


@pytest.mark.parametrize("shared", [False, True], ids=["distinct-times", "one-shared-time"])
def test_a_time_ordered_page_costs_its_own_rows_newest_and_deep(tmp_path, monkeypatch, shared):
    """The keyset reader seeks ``dialogue_room_ts`` from a row key: a 20,000-row room's newest page, a page
    around a row 300 from its beginning and the older page after it cost what the 2,000-row room's do,
    within the logarithmic allowance of a deeper index, never a share of the room -- also when every
    imported row shares one time, so the key's own time is a group the seek must not step through."""
    import math

    from ouroboros.gateway import history
    measured = {}
    for size in (2_000, 20_000):
        root = tmp_path / f"rows{size}" / "data"
        _bulk_room(root, size, shared=lambda _n: shared)

        def read(cursor=None, around=None):
            return json.loads(history._assemble_history_response(root, BULK_CHAT, 100, 60, cursor, around))
        read()  # one-time index builds and schema reads are no page's cost
        newest_cost, newest = _metered(monkeypatch, read)
        around_cost, around = _metered(monkeypatch, lambda: read(around="task:turn-old-300"))
        older_cost, older = _metered(monkeypatch, lambda: read(cursor=around["next_cursor"]))
        assert len([row for row in newest["messages"] if row.get("history_id")]) == 100
        assert around["anchor"]["state"] == "found" and "old 300" in [row["text"] for row in around["messages"]]
        assert [row["text"] for row in older["messages"]][-1] == "old " + str(int(around["messages"][0]["text"][4:]) - 1)
        measured[size] = {"newest": newest_cost, "around": around_cost, "older": older_cost}
    allowance = 128 * math.ceil(math.log2(20_000 / 2_000))
    for read in ("newest", "around", "older"):
        small, large = measured[2_000][read], measured[20_000][read]
        assert large["statements"] == small["statements"], measured
        assert large["steps"] <= small["steps"] + allowance, measured
    if target := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / f"presence-history-page-cost{'-shared-time' if shared else ''}.json").write_text(json.dumps(
            {"synthetic": True, "allowance_steps": allowance, "measured": measured}, indent=2))


@pytest.mark.parametrize("quota", [1, 5, 100])
def test_rows_sharing_one_time_are_delivered_once_under_exact_spans(tmp_path, quota):
    """150 imported rows share one second, among 20 at their own times and live rows. Read ``quota`` at a time,
    every row arrives once in (time, stored position) order; every span holds exactly its page's rows, so
    chains of one frozen end give each row one coordinate whatever their quota; every page reads again
    by its ``page_cursor`` with the same rows, coverage and continuation."""
    root = tmp_path / "data"
    _bulk_room(root, 225, shared=lambda n: n < 150)
    expected = _reading_order(root)
    pages = _chain(root, quota)
    assert _delivered(pages) == expected
    spans = [page["coverage"]["spans"]["chat"] for page in pages]
    assert spans[0]["to"] == pages[0]["coverage"]["upper"]["chat"] and spans[-1]["from"] == 0
    assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:]))
    assert len({span["chain"] for span in spans}) == 1
    found = _coordinates(pages)
    assert [found[history_id] for history_id in expected] == sorted(found.values())
    assert found == _coordinates(_chain(root, 7))  # another quota over the same frozen rows
    for page in pages[::max(1, len(pages) // 6)]:
        again = _read(root, quota, page["page_cursor"])
        assert again["messages"] == page["messages"] and again["coverage"] == page["coverage"]
        assert (again["has_more"], again["next_cursor"]) == (page["has_more"], page["next_cursor"])


def test_the_parents_rows_at_one_instant_page_one_at_a_time_without_repeats(tmp_path):
    """The reported falsifier through the live append path: 120 imported rows stamped one instant, read one
    conversation row at a time, deliver 119, 118, 117 ... once each, under adjacent one-row spans."""
    from ouroboros.gateway import history
    root = tmp_path / "data"
    root.mkdir()
    chat = (1 << 40) + 928
    for n in range(120):
        ds.append(root, "chat", {"ts": SHARED, "direction": "in", "chat_id": chat, "type": "place_arrival",
                                 "text": str(n), "source": "presence:chat", "client_message_id": str(n),
                                 "task_id": f"turn{n}", "place_id": "place", "historical_import": {
                                     "version": 1, "authorship": "participant",
                                     "received_at": "2026-10-26T00:00:00+00:00"}})
    pages, cursor = [], None
    for _ in range(120):
        pages.append(json.loads(history._assemble_history_response(root, chat, 1, 0, cursor)))
        cursor = pages[-1]["next_cursor"]
    assert [row["text"] for page in pages for row in page["messages"] if row.get("history_id")] == [
        str(n) for n in reversed(range(120))]
    spans = [page["coverage"]["spans"]["chat"] for page in pages]
    assert all(span["to"] - span["from"] == 1 for span in spans[:-1]) and spans[-1]["from"] == 0
    assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:])) and not pages[-1]["has_more"]


def test_reads_around_a_row_sharing_its_time_begin_a_chain_of_their_own(tmp_path):
    """Around a task inside a group sharing one time: the page holds the rows nearest it in reading order
    and begins a chain of its own (its coordinates are not the newest page's), reads again identically,
    and its older pages deliver every older row once down to the beginning. Around the same exact row
    (the paging layer's row target: the endpoint lands exact rows only in Main or a Project), a page of
    another quota names the same chain and gives every row the same coordinate."""
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.gateway.history_paging import history_page_tokens, room_lens, select_history_page
    root = tmp_path / "data"
    _bulk_room(root, 225, shared=lambda n: n < 150)
    expected = _reading_order(root)
    newest_chain = _read(root, 5)["coverage"]["spans"]["chat"]["chain"]
    target = next(entry for entry in ds.iter_rows(root, "chat") if entry.row["text"] == "old 60")
    pages = _chain(root, 8, around="task:" + target.row["task_id"])
    first = pages[0]
    assert first["anchor"]["state"] == "found" and first["anchor"]["history_id"] == target.history_id
    shown = [row["history_id"] for row in first["messages"] if row.get("history_id")]
    start = expected.index(shown[0])
    assert shown == expected[start:start + 8] and start == expected.index(target.history_id) - 2
    assert _delivered(pages) == expected[:start + 8]
    spans = [page["coverage"]["spans"]["chat"] for page in pages]
    assert len({span["chain"] for span in spans}) == 1 and spans[0]["chain"] != newest_chain
    assert all(older["to"] == newer["from"] for newer, older in zip(spans, spans[1:])) and spans[-1]["from"] == 0
    by_task = _coordinates(pages)
    again = _read(root, 8, first["page_cursor"])
    assert again["messages"] == first["messages"] and again["coverage"] == first["coverage"]

    def select(cursor=None, **around):
        rule = lambda chat, _row: chat == BULK_CHAT  # noqa: E731
        page = select_history_page(root, BULK_CHAT, "view", cursor, {"human": 20, "progress": 0},
                                   {"chat": lambda _entry: True, "progress": lambda _entry: False},
                                   {"human": 1500, "progress": 600}, room_lens(root, BULK_CHAT, set(), [], {}),
                                   thread_filter=(rule, lambda chat, default: default if chat is None else int(chat)),
                                   **around)
        return page, history_page_tokens(page)
    page, tokens = select(around=format_address(row_address(target.row)), row_target={"source": "chat", "seq": target.seq})
    entries = page["selections"]["chat"][0]
    assert page["anchor"]["history_id"] == target.history_id and page["chains"]["chat"] == spans[0]["chain"]
    assert [entry["history_id"] for entry in entries] == expected[start - 3:start + 17]
    by_row = {entry["history_id"]: entry["_history_end"] - 1 for entry in entries}
    while tokens["has_more"]:
        page, tokens = select(tokens["next_cursor"])
        assert page["chains"]["chat"] == spans[0]["chain"]
        by_row.update((entry["history_id"], entry["_history_end"] - 1) for entry in page["selections"]["chat"][0])
    assert sorted(by_row, key=expected.index) == expected[:start + 17] and page["before"]["chat"] == 0
    assert {history_id: by_row[history_id] for history_id in by_task} == by_task


def test_the_last_time_ordered_page_names_the_beginning(tmp_path):
    """1,100 rows read 100 at a time: the tenth older page ends on the room's first row and says so
    (no press lands on an empty page), and the beginning closes the coverage at 0."""
    from ouroboros.gateway import history
    root = tmp_path / "data"
    _bulk_room(root, 1_100)
    pages, cursor = [], None
    while True:
        pages.append(json.loads(history._assemble_history_response(root, BULK_CHAT, 100, 60, cursor)))
        if not pages[-1]["has_more"]:
            break
        cursor = pages[-1]["next_cursor"]
    assert len(pages) == 11 and all(page["messages"] for page in pages)
    assert pages[-1]["messages"][0]["text"] == "old 0" and pages[-1]["coverage"]["spans"]["chat"]["from"] == 0


def test_deferral_moves_a_time_ordered_boundary_by_time_not_stored_position():
    """Quota deferral raises the boundary to the newest deferred row's ``_history_end``; the continuation reads
    below the cut above that row, named by the first row above it, even when stored positions run against
    time (``deferred_before``, ``_chrono_bound``); a boundary naming none of the page's rows is refused."""
    from ouroboros.gateway.history_paging import _chrono_bound, deferred_before
    from ouroboros.store import StoreError
    times = [at(1), at(2), at(3)]  # oldest first, stored at 900, 500 and 100
    entries = [{"history_id": f"chat:{seq}", "history_position": {"source": "chat", "offset": seq},
                "_history_end": mark + 1, "_chrono_key": [ts, None, seq]}
               for ts, seq, mark in zip(times, (900, 500, 100), (11, 12, 13))]
    shown = [entries[2]]
    before = deferred_before("chat", entries, entries, shown, 11)
    assert before == 13  # the newest deferred row (stored at 500), not the largest stored position (900)
    page = {"selections": {"chat": (entries, 11, set())}, "before": {"chat": before},
            "chrono": {"chat": {"lower": [times[0], None, 900], "top": None}}}
    assert _chrono_bound(page, "chat", before) == [times[2], None, 100]  # the shown row above the deferred one
    assert _chrono_bound(page, "chat", 14) is None  # the newest row deferred: the cut at the frozen present
    assert _chrono_bound(page, "chat", 11) == [times[0], None, 900] and _chrono_bound(page, "chat", 0) is None
    with pytest.raises(StoreError):
        _chrono_bound(page, "chat", 7)


def test_a_stored_order_cursor_after_the_first_import_asks_for_a_fresh_read(room):
    for n in range(3):
        assert live(room, n, sent=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), facts=()).status_code == 200
    with history_client(room.root) as client:
        stored_order = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
        assert room.http.post("/presence/history", json=body(record(1000))).status_code == 200
        for cursor in (stored_order["page_cursor"], stored_order["after_cursor"]):
            refused = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": cursor})
            assert refused.status_code == 409 and refused.json()["reason_code"] == "history_view_changed"
        fresh = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
        assert fresh["coverage"]["spans"]["chat"]["chain"].startswith("dialog-chrono-")
        assert [row["text"] for row in fresh["messages"]][0] == "message 1000"


def test_a_live_capture_under_an_earlier_binding_is_judged_by_its_frozen_identity(tmp_path, monkeypatch):
    """The Hub's case. A migrated place keeps its legacy binding id, so importing an event captured live
    under that binding answers ``duplicate, custody: live``. A room reconnected as a new place has a new
    id, so the import has a new turn id, the store's named-row dedup still finds the live row of that
    source event, and its frozen identity (the old binding) refuses the record: 409, nothing rewritten."""
    from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
    from tests.frozen_presence_bindings import save_presence_binding
    folder = tmp_path / "skills" / "external" / "legacy-helper"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: legacy-helper\ndescription: Old behavior.\nversion: 0.1\n"
                                     "type: instruction\npresence:\n  instructions: Answer briefly.\n---\n# Old\n")
    binding = "b" * 32
    endpoint = PresenceEndpoint("chat", "a", "r", "")
    save_presence_binding(tmp_path, PresenceBinding(binding, "bridge", "legacy-helper", endpoint, endpoint))
    migrated, = p.load_places(tmp_path)
    assert (migrated.place_id, migrated.legacy_binding_ids) == (binding, (binding,))

    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    with TestClient(host_service.create_host_service_app(tmp_path, presence_runner=lambda **_kw: pytest.fail(
            "an unmatched or imported event starts no turn"))) as client:
        assert client.post("/presence/turn", json={"event": {**event(1), "facts": []}}).json()["outcome"] == "silent"
        captured, = rows(tmp_path)
        assert captured["task_id"] == presence_turn_task_id(binding, "chat:a:r:0:1")
        same = client.post("/presence/history", json=body(record(1))).json()["results"][0]
        assert (same["custody"], same["duplicate"], same["turn_ref"]) == ("live", True, captured["task_id"])
        p.edit_place(tmp_path, binding, "disconnect", author="owner")
        reconnected = p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a",
                                      conversation_id="r")
        assert presence_turn_task_id(reconnected.place_id, "chat:a:r:0:1") != captured["task_id"]
        refused = client.post("/presence/history", json=body(record(1)))
        assert refused.status_code == 409, refused.text
        result, = refused.json()["results"]
        assert (result["code"], result["disposition"]) == ("presence_event_identity_conflict", "rejected")
    assert rows(tmp_path) == [captured]


def test_room_without_imports_keeps_stored_order_paging(room):
    for n in range(3):
        assert live(room, n, facts=()).status_code == 200
    with history_client(room.root) as client:
        page = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
    assert page["coverage"]["spans"]["chat"]["chain"] == "dialog"
    assert page["coverage"]["upper"]["chat"] == ds.stream_end(room.root, "chat")


def test_wake_counts_imported_history_without_listing_it(room):
    import time

    from ouroboros.consciousness_wake import observe_wake
    assert room.http.post("/presence/history", json=body(record(1), record(2), record(3))).status_code == 200
    assert live(room, 9, facts=(), text="a live correspondent").status_code == 200
    observation = observe_wake(room.root, boundary=None, since=0, now=time.time())
    kinds = [kind for kind, _offset, _line in observation.events]
    assert kinds.count("history_import") == 3 and kinds.count("correspondent_message") == 1
    text = observation.full_text()
    assert "3 imported earlier Presence messages" in text and "a live correspondent" in text
    assert "message 1" not in text and "message 2" not in text


def test_reentry_and_memory_read_imported_rows_as_history(room):
    from ouroboros.chat_chain import row_address
    from ouroboros.memory_inventory import place_activity, row_meta
    from ouroboros.presence_continuation import _row_fact
    assert room.http.post("/presence/history", json=body(
        record(1, authorship="platform_account", actor="a"))).status_code == 200
    assert live(room, 9, facts=()).status_code == 200
    imported, arrival = rows(room.root)
    fact = _row_fact(SimpleNamespace(task_id="author"), imported)
    assert "imported as history" in fact and "connected account itself" in fact and "not a new arrival" in fact
    assert "taken by" not in fact and "taken by" in _row_fact(SimpleNamespace(task_id="author"), arrival)
    entries = [(row_address(row), row_meta(row), index) for index, row in enumerate((imported, arrival))]
    assert place_activity(entries)["arrivals"] == 1


def test_real_pages_name_their_coverage_cuts_by_row_key(tmp_path):
    """The public ``keys`` of a room with imported history (CLIENT_CONTRACT §5) on real pages. Rows sharing one
    time paged 200 at a time name each cut by the first frozen row above it: adjacent spans share their cut from
    the beginning (``from: null``; an empty last page closes it) to the present (``to: null``) under one frozen
    end, and replays answer as read. A page around a row exactly adjacent to the newest page names the newest
    page's lower cut as its upper one; one row further names a lower cut, a row neither page delivered. A newer
    live arrival freezes a later read later and continues the ``after`` cursor; an import resets it."""
    from ouroboros.gateway.history_paging import HistoryCursorError
    root = tmp_path / "data"
    _bulk_room(root, 400, shared=lambda n: n < 300)
    expected = _reading_order(root)
    by_id = {entry.history_id: entry.row for entry in ds.iter_rows(root, "chat")}

    def keys(page):
        return page["coverage"]["spans"]["chat"]["keys"]

    def around(newest_rank, quota=20):  # the page around the row ``newest_rank`` rows below the newest
        return _read(root, quota, around="task:" + by_id[expected[-newest_rank]]["task_id"])
    chain, recent = _chain(root, 200), _read(root, 20)
    spans = [keys(page) for page in chain]
    assert [len(page["messages"]) for page in chain] == [200, 200, 0]
    assert spans[-1]["from"] is None and spans[-1]["to"] == spans[1]["from"]
    assert spans[0]["to"] is None and spans[1]["to"] == spans[0]["from"]
    assert len({span["frozen"] for span in spans}) == 1
    assert [_read(root, 200, page["page_cursor"])["coverage"] for page in chain] == [page["coverage"] for page in chain]
    adjacent, gapped = around(35), around(36)
    assert keys(adjacent)["to"] == keys(recent)["from"] and keys(gapped)["to"] < keys(recent)["from"]
    newest_live = {"ts": "2026-10-26T09:30:00.000001+00:00", "direction": "in", "chat_id": BULK_CHAT, "type": "place_arrival",
                   "text": "live late", "source": "presence:chat", "client_message_id": "live-late", "task_id": "turn-live-late",
                   "place_id": "place", "sender_session_id": "person"}
    ds.append(root, "chat", newest_live)
    arrived = _read(root, 20, recent["after_cursor"])
    assert [row["text"] for row in arrived["messages"] if row.get("history_id")] == ["live late"]
    assert keys(around(15))["frozen"] > keys(recent)["frozen"]
    ds.append(root, "chat", {**newest_live, "ts": at(-5), "text": "imported late", "client_message_id": "imported-late",
                             "task_id": "turn-imported-late", "historical_import": {
                                 "version": 1, "authorship": "participant", "received_at": "2026-10-26T09:31:00+00:00"}})
    with pytest.raises(HistoryCursorError) as reset:
        _read(root, 20, arrived["after_cursor"])
    assert reset.value.reason == "history_delta_reset"


@pytest.mark.parametrize("own_day", ["2026-10-25", "2026-10-23"], ids=["own-newest", "own-older"])
def test_rooms_read_and_account_suggestions_cost_their_own_bound_after_a_long_import(tmp_path, monkeypatch, own_day):
    """A room whose newest stored arrivals are thousands of imported rows of the connected account's own,
    newer or older by time than its live arrivals: the rooms read names the room's latest arrival by
    time, the suggestions read the newest 50 arrivals, both at one cost for 2,000 and 20,000 imported rows."""
    import math

    from ouroboros.place_arrivals import account_candidates, room_index
    place, measured = SimpleNamespace(place_id="place", legacy_binding_ids=()), {}
    for size in (2_000, 20_000):
        root = tmp_path / f"own{size}" / "data"

        def arrival(n, ts, **extra):
            return {"ts": ts, "direction": "in", "chat_id": BULK_CHAT, "type": "place_arrival", "text": f"m {n}",
                    "source": "presence:chat", "client_message_id": f"m-{n}", "task_id": f"turn-{n}",
                    "place_id": "place", "sender_session_id": f"person-{n}", "sender_label": f"Person {n}",
                    "transport": {"conversation": {"title": f"Room at {n}"}}, **extra}
        own = [arrival(n, f"{own_day}T{n // 3600 % 24:02d}:{n // 60 % 60:02d}:{n % 60:02d}+00:00",
                       sender_session_id="a", historical_import={"version": 1, "authorship": "platform_account",
                                                                 "received_at": "2026-10-26T09:00:00+00:00"})
               for n in range(size)]
        people = [arrival(100_000 + n, f"2026-10-24T08:00:{n:02d}.000001+00:00") for n in range(30)]
        (root / "logs").mkdir(parents=True)
        (root / "logs" / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in people + own))
        ds.ensure(root)
        room_index(root, place, details=True)  # one-time index builds are no read's cost
        rooms_cost, (listed,) = _metered(monkeypatch, lambda: room_index(root, place, details=True))
        latest = own[-1] if own_day > "2026-10-24" else people[-1]
        assert (listed["last_at"], listed["label"]) == (latest["ts"], latest["transport"]["conversation"]["title"])
        people_cost, suggested = _metered(monkeypatch, lambda: account_candidates(root, "place"))
        # People from the newest 50 arrivals; the account's own messages there suggest no one.
        assert [person["platform_actor_id"] for person in suggested] == (
            [] if own_day > "2026-10-24" else [f"person-{100_000 + n}" for n in reversed(range(10, 30))])
        measured[size] = {"rooms": rooms_cost, "suggestions": people_cost}
    allowance = 128 * math.ceil(math.log2(20_000 / 2_000))
    for read in ("rooms", "suggestions"):
        small, large = measured[2_000][read], measured[20_000][read]
        assert large["statements"] == small["statements"] and large["steps"] <= small["steps"] + allowance, measured
    if target := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / f"place-reads-cost-own-{'newest' if own_day > '2026-10-24' else 'older'}.json").write_text(
            json.dumps({"synthetic": True, "allowance_steps": allowance, "measured": measured}, indent=2))


def test_rooms_read_and_account_suggestions_follow_time_and_authorship(room, monkeypatch):
    from ouroboros.gateway import places_api
    from ouroboros.place_arrivals import account_candidates
    monkeypatch.setattr(places_api, "request_drive_root", lambda _request: room.root)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    assert live(room, 9, facts=()).status_code == 200
    assert room.http.post("/presence/history", json=body(
        record(1, conversation={}), record(2, authorship="platform_account", actor="a", conversation={}))).status_code == 200
    with TestClient(Starlette(routes=places_api.places_routes())) as client:
        listed, = client.get(f"/api/places/{room.place.place_id}/rooms").json()["rooms"]
    assert listed["label"] == "Room R" and listed["last_at"] == rows(room.root)[0]["ts"]
    assert {person["platform_actor_id"] for person in account_candidates(room.root, room.place.place_id)} == {"person"}


def test_a_page_around_work_that_only_narrated_reads_again_and_continues(room):
    """CLIENT_CONTRACT §5: ``page_cursor`` re-reads a page and older pages continue. Around a task whose only rows
    in a room holding imports are narration, the conversation holds none of its rows (a null span, never a
    guess), and the page's cursors still name the stream's empty row keys, so both decode and read on."""
    assert room.http.post("/presence/history", json=body(record(1), record(2))).status_code == 200
    assert live(room, 9, facts=()).status_code == 200
    for n in range(20):  # earlier work's narration: older pages remain
        ds.append(room.root, "progress", {"ts": at(10 + n), "chat_id": CHAT, "task_id": "earlier", "content": f"step {n}"})
    ds.append(room.root, "progress", {"ts": at(40), "chat_id": CHAT, "task_id": "narrator", "content": "only narration"})
    with history_client(room.root) as client:
        page = client.get("/api/chat/history", params={"chat_id": CHAT, "around": "task:narrator"}).json()
        assert page["anchor"]["state"] == "found" and page["coverage"]["spans"]["chat"] is None, page
        assert page["has_more"] and _decode(page["next_cursor"])["chrono"] == {"before": None}
        assert _decode(page["page_cursor"])["chrono"] == {"before": None, "lower": None}
        again = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": page["page_cursor"]})
        assert again.status_code == 200, again.text
        assert again.json()["messages"] == page["messages"]
        older = client.get("/api/chat/history", params={"chat_id": CHAT, "cursor": page["next_cursor"]})
        assert older.status_code == 200, older.text
        assert older.json()["messages"] and "only narration" not in json.dumps(older.json()["messages"])


def test_main_keeps_its_own_newest_page_after_a_bulk_import_into_a_place_room(tmp_path, monkeypatch):
    """DESIGN "History edges": a page counts the room's own messages, never other rooms' traffic. Main never shows
    a place arrival, so its conversation probes leave rooms holding them out: 11,000 imported rows stored after
    Main's own leave Main's newest page whole, at the cost a 2,000-row import leaves. What Main shows from
    elsewhere stays: a chat-less legacy row, another non-Project room's row, a private Main contact and a focus's
    narration stored under the place room."""
    import math

    from ouroboros.gateway import history
    from ouroboros.place_arrivals import _dialogue_indexes
    measured = {}
    for size in (2_000, 11_000):
        root = tmp_path / f"main{size}" / "data"
        main = [{"ts": f"2026-10-25T09:{n:02d}:00+00:00", "direction": "in", "chat_id": 1, "text": f"main {n}",
                 "client_message_id": f"owner-{n}"} for n in range(30)]
        elsewhere = [{"ts": "2026-10-25T08:00:00+00:00", "direction": "in", "text": "legacy without a room"},
                     {"ts": "2026-10-25T08:01:00+00:00", "direction": "in", "chat_id": 777, "text": "another room's row",
                      "transport": {"provider": "telegram", "actor": {"display_name": "Ann"}}},
                     {"ts": "2026-10-25T08:02:00+00:00", "direction": "out", "chat_id": 1, "task_id": "focus",
                      "text": "a private word for the owner", "presence_contact": {
                          "task_id": "focus", "binding_id": "place", "source_event_id": "chat:a:r:0:1"}}]
        imported = [{"ts": (OLD + timedelta(minutes=n)).isoformat(), "direction": "in", "chat_id": BULK_CHAT,
                     "type": "place_arrival", "text": f"old {n}", "source": "presence:chat", "client_message_id": f"old-{n}",
                     "task_id": f"turn-{n}", "place_id": "place", "sender_session_id": "person", "historical_import": {
                         "version": 1, "authorship": "participant", "received_at": "2026-10-26T09:00:00+00:00"}}
                    for n in reversed(range(size))]
        (root / "logs").mkdir(parents=True)
        (root / "logs" / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in elsewhere + main + imported))
        ds.ensure(root)
        _dialogue_indexes(root)  # as the import writer builds them before its rows land (``record_history``)
        ds.append(root, "progress", {"ts": "2026-10-26T10:00:00+00:00", "chat_id": BULK_CHAT, "task_id": "focus",
                                     "content": "focus narration under the place room"})

        def read():
            return json.loads(history._assemble_history_response(root, 1, 150, 60))
        read()  # one-time index builds are no read's cost
        cost, page = _metered(monkeypatch, read)
        shown = [str(row.get("text") or row.get("content") or "") for row in page["messages"]]
        assert [text for text in shown if text.startswith("main ")] == [f"main {n}" for n in range(30)], shown[:5]
        assert {"legacy without a room", "another room's row", "a private word for the owner",
                "focus narration under the place room"} <= set(shown), shown
        assert not any(text.startswith("old ") for text in shown) and page["has_more"] is False
        measured[size] = cost
    allowance = 128 * math.ceil(math.log2(11_000 / 2_000))
    assert measured[11_000]["statements"] == measured[2_000]["statements"], measured
    assert measured[11_000]["steps"] <= measured[2_000]["steps"] + allowance, measured
    if target := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "main-read-after-place-import.json").write_text(json.dumps(
            {"synthetic": True, "allowance_steps": allowance, "measured": measured}, indent=2))


def test_recall_reads_from_the_latest_message_by_time_after_an_import(room):
    """``chat_history`` pages from the latest message (``tools/control.py``). Earlier messages imported after newer
    live ones page at their send times: the first page holds the newest conversation, its count of older rows is
    exact, and the next page reads the imported history in time order."""
    from ouroboros.memory import Memory
    for n in range(3):
        assert live(room, 100 + n, facts=(), text=f"live words {n}").status_code == 200
    batch = body(*(record(n, text=f"earlier words {n}") for n in (3, 2, 1)))  # newest first, as the Hub sends
    assert room.http.post("/presence/history", json=batch).status_code == 200
    newest = Memory(room.root).chat_history(count=3)
    assert newest.startswith("Showing 3 of 6 messages; 3 older remain."), newest
    assert "earlier words" not in newest
    assert newest.index("live words 0") < newest.index("live words 1") < newest.index("live words 2")
    older = Memory(room.root).chat_history(count=3, offset=3)
    assert "live words" not in older
    assert older.index("earlier words 1") < older.index("earlier words 2") < older.index("earlier words 3")


def test_a_focus_view_of_a_room_with_imports_reads_and_folds_oldest_first_by_time(tmp_path):
    """A place focus reads its room's open rows (``memory_inventory.open_room_rows``): earlier messages imported
    after the live conversation sit at their send times, so the lane, its ``since``, the room's earliest and
    latest times and its latest place row follow time, and under window pressure the floor turns the
    years-old imported words into addresses before the fresh live ones (F7p, oldest first). Rows keep their
    physical addresses and positions."""
    from ouroboros import memory_floor
    from ouroboros import memory_view as mv
    from tests import _memory_inventory_shared as shared
    shared.world(tmp_path)
    stamp = {"place_id": "garden", "conversation_key": "garden:talk"}
    marker = {"version": 1, "authorship": "participant", "received_at": "2026-09-05T11:00:00+00:00"}
    words = " and then a long enough paragraph that an address is shorter than the words themselves" * 4
    live_rows = [shared.msg(f"2026-09-05T10:0{n}:00+00:00", f"LIVE_WORDS_{n}{words}", chat_id=444, sender_session_id="ann",
                            type="place_arrival", client_message_id=f"live-{n}",
                            presence_provenance={**stamp, "source_event_id": f"live-{n}"}) for n in range(3)]
    imported = [shared.msg(f"2024-0{n + 1}-01T10:00:00+00:00", f"OLD_WORDS_{n}{words}", chat_id=444, sender_session_id="ann",
                           type="place_arrival", client_message_id=f"old-{n}", historical_import=marker,
                           presence_provenance={**stamp, "source_event_id": f"old-{n}"}) for n in (2, 1, 0)]
    shared.append(tmp_path / "logs" / "chat.jsonl", *live_rows, *imported)
    focus = {"id": "focus1", "chat_id": 1, "metadata": {"usage_category": "consciousness", "wake_cause": "focus",
                                                         "focus": {"room_chat_id": 444, **stamp}}}
    view = mv.capture_memory_view(tmp_path, focus, mv.view_spec_for_task(focus, tmp_path))
    names = ["old-0", "old-1", "old-2", "live-0", "live-1", "live-2"]
    lane = [item for item in view.room["lane1"] if item["ref"].startswith("chat 444 /")]
    assert [item["ref"].rsplit("/ ", 1)[1] for item in lane] == names
    with history_client(tmp_path) as client:  # as the room's endpoint reads it
        page = client.get("/api/chat/history", params={"chat_id": 444}).json()
    assert [row["text"].split(" ", 1)[0] for row in page["messages"] if row.get("history_id")] == [
        "OLD_WORDS_0", "OLD_WORDS_1", "OLD_WORDS_2", "LIVE_WORDS_0", "LIVE_WORDS_1", "LIVE_WORDS_2"]
    positions = {item["ref"].rsplit("/ ", 1)[1]: item["pos"] for item in lane}
    assert positions["old-0"] > positions["live-2"]  # physical positions are kept, only the reading order moves
    facts = view.room["facts"]
    assert (facts["earliest"], facts["latest"]) == ("2024-01-01T10:00:00+00:00", "2026-09-05T10:02:00+00:00")
    assert view.room["since"].startswith("2024-01-01") and facts["place"]["latest"] == "2026-09-05T10:02:00+00:00"
    assert facts["place"]["last_address"] == lane[-1]["address"]
    walked = [ident for step, ident, *_ in memory_floor.floor_elements(view) if step == "F7p"]
    assert walked == [item["address"] for item in lane], walked


def test_a_frozen_older_page_steps_over_rows_imported_after_it_froze(tmp_path, monkeypatch):
    """Known cost, disclosed in ``history_chronology``: a frozen chain reads only what it froze, and rows a later
    import stores between its keys are stepped over in the time index, never read or delivered. The older page
    and its replay return the same rows after 2,000 later imports into its time range, with the same statements;
    their cost grows by a few VM steps per such row (bounded by those rows, never by the room)."""
    root = tmp_path / "data"
    _bulk_room(root, 1_000)
    newest = _read(root, 100)
    cursor = newest["next_cursor"]
    _read(root, 100, cursor)  # one-time index builds are no read's cost
    cost, page = _metered(monkeypatch, lambda: _read(root, 100, cursor))
    replay_cost, replay = _metered(monkeypatch, lambda: _read(root, 100, page["page_cursor"]))
    lowest, highest = (int(page["messages"][index]["text"].split()[1]) for index in (0, -1))
    later = 2_000
    for n in range(later):  # stored after the chain froze, timed between the older page's rows
        ds.append(root, "chat", {"ts": (OLD + timedelta(minutes=lowest + n % (highest - lowest), seconds=30)).isoformat(),
                                 "direction": "in", "chat_id": BULK_CHAT, "type": "place_arrival", "text": f"later {n}",
                                 "source": "presence:chat", "client_message_id": f"later-{n}", "task_id": f"turn-later-{n}",
                                 "place_id": "place", "sender_session_id": "person", "historical_import": {
                                     "version": 1, "authorship": "participant", "received_at": "2026-10-26T10:00:00+00:00"}})
    after_cost, after = _metered(monkeypatch, lambda: _read(root, 100, cursor))
    replay_after_cost, replay_after = _metered(monkeypatch, lambda: _read(root, 100, page["page_cursor"]))
    assert (after["messages"], after["coverage"], after["next_cursor"]) == (page["messages"], page["coverage"], page["next_cursor"])
    assert (replay_after["messages"], replay_after["coverage"]) == (replay["messages"], replay["coverage"])
    slopes = {}
    for name, (early, late) in {"older": (cost, after_cost), "replay": (replay_cost, replay_after_cost)}.items():
        assert late["statements"] == early["statements"], (name, early, late)
        slopes[name] = (late["steps"] - early["steps"]) / later
        assert slopes[name] <= 16, (name, early, late)  # measured 7 here; linear in the later rows, never the room
    if target := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "frozen-page-after-later-imports.json").write_text(json.dumps(
            {"synthetic": True, "later_rows": later, "slopes_steps_per_later_row": slopes,
             "measured": {"older": [cost, after_cost], "replay": [replay_cost, replay_after_cost]}}, indent=2))


# The provider's own order (``message.source_order``, ``presence_source_order_version``): one fact on both doors.
SECOND = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


def ordered(n, order, sent=SECOND, **kw):
    """A history record of message ``n`` sent at ``sent`` with the provider's order ``order``."""
    return record(n, sent=sent.isoformat(), message={"source_order": order}, **kw)


def _texts(pages):
    return [row["text"] for page in reversed(pages) for row in page["messages"] if row.get("history_id")]


def _triple(key):
    """A public reading key as the reader compares it (no order: before every order)."""
    return (key[0], -1 if key[1] is None else key[1], key[2])


def test_identity_advertises_the_source_order_beside_history_import_version_1(room):
    from tests.test_host_service_api import _seed_token
    _seed_token(room.root, skill="bridge", token="tok", permissions=["presence"], manifest_permissions=["presence"])
    with TestClient(host_service.create_host_service_app(room.root)) as client:
        identity = client.get("/identity", headers={"x-skill-token": "tok"}).json()
    assert (identity["presence_history_import_version"], identity["presence_source_order_version"]) == (1, 1)


@pytest.mark.parametrize("order", [True, -1, 1 << 53, 1.5, "7", None], ids=["bool", "negative", "2^53", "float", "text", "null"])
def test_a_source_order_that_is_no_order_is_refused_on_both_doors(room, order):
    """``message.source_order`` is an integer from 0 to 2^53-1 on the live and the history door alike: anything
    else is a malformed event (400, never a retry), and nothing is recorded."""
    refused = room.http.post("/presence/history", json=body(record(1, message={"source_order": order})))
    assert refused.status_code == 400 and refused.json()["field"] == "records[0].event.message.source_order", refused.text
    turn = live(room, 2, message={"source_order": order}, facts=())
    assert turn.status_code == 400 and (turn.json()["code"], turn.json()["disposition"]) == (
        "presence_payload_invalid", "rejected"), turn.text
    assert not rows(room.root) and not room.runs


def test_live_events_without_a_source_order_stay_valid_and_keep_their_send_time(room):
    assert live(room, 1, facts=()).status_code == 200
    unknown = {"event": {**event(2, message={"sent_at": None}), "facts": []}}
    assert room.http.post("/presence/turn", json=unknown).status_code == 200
    first, second = rows(room.root)
    assert first["source_order"] == {"time": at(1), "order": None} and "source_order" not in second


def test_a_source_order_is_enrichment_the_first_stored_value_stands(room):
    """A later record or live replay naming an order never conflicts with the row first stored: its block stands
    (no order stays no order), so an import before a connector learned the field resends as a duplicate."""
    assert room.http.post("/presence/history", json=body(record(1))).status_code == 200
    again = room.http.post("/presence/history", json=body(record(1, message={"source_order": 41})))
    assert again.status_code == 200 and again.json()["results"][0]["duplicate"] is True
    assert room.http.post("/presence/history", json=body(ordered(2, 7, sent=OLD))).status_code == 200
    other = room.http.post("/presence/history", json=body(ordered(2, 8, sent=OLD)))
    assert other.status_code == 200 and other.json()["results"][0]["duplicate"] is True
    replay = live(room, 2, sent=OLD.isoformat(), message={"source_order": 9})
    assert replay.status_code == 200 and replay.json()["outcome"] == "silent" and not room.runs
    assert [row.get("source_order") for row in rows(room.root)] == [
        {"time": at(1), "order": None}, {"time": OLD.isoformat(), "order": 7}]


def test_a_late_live_capture_reads_at_its_send_time_among_imported_history(room):
    """Example 1: message 105 is captured live late (received now, sent S+5, order 105) and 104 and 106 are imported:
    the room reads 104, 105, 106, and each page's cut is the explicit key ``[time, order, position]`` of its row."""
    assert live(room, 105, sent=(SECOND + timedelta(seconds=5)).isoformat(), message={"source_order": 105},
                facts=()).status_code == 200
    assert room.http.post("/presence/history", json=body(ordered(106, 106, SECOND + timedelta(seconds=6)),
                                                         ordered(104, 104, SECOND + timedelta(seconds=4)))).status_code == 200
    stored = {entry.row["text"]: entry for entry in ds.iter_rows(room.root, "chat")}
    with history_client(room.root) as client:
        pages = _pages(client, n_human=1)
    assert _texts(pages) == ["message 104", "message 105", "message 106"]
    cuts = [page["coverage"]["spans"]["chat"]["keys"]["from"] for page in pages]  # newest first; null: the beginning
    assert cuts == [[stored[text].row["source_order"]["time"], int(text.split()[1]), stored[text].seq]
                    for text in ("message 106", "message 105")] + [None]
    assert stored["message 105"].row["ts"] > stored["message 106"].row["ts"]  # the receipt stays the row's ts


def test_more_than_100_messages_of_one_second_read_in_source_order_across_requests(room):
    """Example 2: 130 album messages share one second (orders 300-429) and arrive newest first in requests of 100
    and 30; the room reads 300-429, and the two pages of 100 meet at one key, ``[S, 330, position]``."""
    newest_first = [ordered(300 + n, 300 + n) for n in reversed(range(130))]
    assert room.http.post("/presence/history", json=body(*newest_first[:100], import_id="imp_a")).status_code == 200
    assert room.http.post("/presence/history", json=body(*newest_first[100:], import_id="imp_b")).status_code == 200
    position = {entry.row["text"]: entry.seq for entry in ds.iter_rows(room.root, "chat")}
    with history_client(room.root) as client:
        pages = _pages(client)
    assert _texts(pages) == [f"message {300 + n}" for n in range(130)]
    newer, older = (page["coverage"]["spans"]["chat"]["keys"] for page in pages)
    assert newer["from"] == older["to"] == [SECOND.isoformat(), 330, position["message 330"]]


def test_a_second_split_by_a_cap_and_completed_later_reads_in_source_order(room):
    """Example 3: a capped plan stores orders 1050-1099 of one second; a later import of earlier dates appends
    1000-1049 of that second: the room reads 1000-1099."""
    capped = [ordered(1000 + n, 1000 + n) for n in reversed(range(50, 100))]
    assert room.http.post("/presence/history", json=body(*capped, import_id="imp_capped")).status_code == 200
    earlier = [ordered(1000 + n, 1000 + n) for n in reversed(range(50))]
    assert room.http.post("/presence/history", json=body(*earlier, import_id="imp_earlier")).status_code == 200
    with history_client(room.root) as client:
        assert _texts(_pages(client, n_human=30)) == [f"message {1000 + n}" for n in range(100)]


def test_a_later_overlapping_import_adds_the_missing_rows_once_in_source_order(room):
    """Example 4: a re-import of one second answers duplicate for the stored rows and adds the missing ones: each
    row reads once, in source order."""
    assert room.http.post("/presence/history", json=body(*(ordered(500 + n, 500 + n) for n in (4, 2, 0)))).status_code == 200
    overlap = room.http.post("/presence/history", json=body(*(ordered(500 + n, 500 + n) for n in reversed(range(6))),
                                                            import_id="imp_overlap"))
    assert overlap.status_code == 200
    assert [result["duplicate"] for result in overlap.json()["results"]] == [False, True, False, True, False, True]
    with history_client(room.root) as client:
        assert _texts(_pages(client)) == [f"message {500 + n}" for n in range(6)]


def test_a_page_around_a_row_inside_one_second_begins_its_chain_and_its_keys_compare(room):
    """Example 5: around a row inside a same-second group the page begins its own chain; its triple cuts compare
    with the newest page's: a near page reaches above the newest page's lower cut, one further down stops below
    it (rows neither page delivered); replays answer identically."""
    group = [ordered(700 + n, 700 + n) for n in reversed(range(40))]
    assert room.http.post("/presence/history", json=body(*group)).status_code == 200
    for n in range(3):
        assert live(room, 9000 + n, sent=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                    facts=()).status_code == 200
    task = {entry.row["text"]: entry.row["task_id"] for entry in ds.iter_rows(room.root, "chat")}
    with history_client(room.root) as client:
        def read(**params):
            return client.get("/api/chat/history", params={"chat_id": CHAT, **params}).json()
        newest = read(n_human=10)
        near, far = (read(n_human=8, around="task:" + task[f"message {order}"]) for order in (730, 720))
        again = read(cursor=near["page_cursor"])
    cut = newest["coverage"]["spans"]["chat"]["keys"]["from"]
    assert cut[:2] == [SECOND.isoformat(), 733]
    for page in (near, far):
        span = page["coverage"]["spans"]["chat"]
        assert page["anchor"]["state"] == "found" and "-at-" in span["chain"]
        assert span["chain"] != newest["coverage"]["spans"]["chat"]["chain"]
    assert _triple(near["coverage"]["spans"]["chat"]["keys"]["to"]) > _triple(cut)
    assert far["coverage"]["spans"]["chat"]["keys"]["to"][:2] == [SECOND.isoformat(), 726]
    assert _triple(far["coverage"]["spans"]["chat"]["keys"]["to"]) < _triple(cut)
    assert (again["messages"], again["coverage"]) == (near["messages"], near["coverage"])


def test_rows_without_a_source_order_keep_their_stored_position_before_ordered_ones(room):
    """Example 6, the honest fallback: imported rows without an order read by stored position, before the ordered
    rows of their second; a live row with an order but no send time reads at its receipt time."""
    assert room.http.post("/presence/history", json=body(record(1, sent=SECOND.isoformat()),
                                                         record(2, sent=SECOND.isoformat()))).status_code == 200
    assert room.http.post("/presence/history", json=body(ordered(4, 4), ordered(3, 3))).status_code == 200
    assert room.http.post("/presence/turn", json={"event": {**event(9, message={"sent_at": None, "source_order": 9}),
                                                            "facts": []}}).status_code == 200
    stored = {entry.row["text"]: entry for entry in ds.iter_rows(room.root, "chat")}
    with history_client(room.root) as client:
        pages = _pages(client, n_human=1)
    assert _texts(pages) == ["message 1", "message 2", "message 3", "message 4", "message 9"]
    keys = [page["coverage"]["spans"]["chat"]["keys"]["from"] for page in reversed(pages)]
    assert keys[0] is None and keys[1] == [SECOND.isoformat(), None, stored["message 2"].seq]  # the beginning, then 2
    assert keys[4] == [stored["message 9"].row["ts"], 9, stored["message 9"].seq]


def test_the_place_reader_projects_only_the_send_time_an_arrival_recorded(room):
    """``place_history.sent_at`` projects the send time a row's arrival recorded (its immutable ``source_order``):
    an import's is its ``ts`` (its saved time is ``historical_import.received_at``), a late live row's is earlier
    than its receipt; a live event with no send time and Ouroboros's reply have none. ``ts`` stays the row's own."""
    assert room.http.post("/presence/history", json=body(ordered(1, 1, sent=OLD))).status_code == 200
    late = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert live(room, 2, sent=late, facts=()).status_code == 200
    unknown = {"event": {**event(3, message={"sent_at": None}), "facts": []}}
    assert room.http.post("/presence/turn", json=unknown).status_code == 200
    ds.append(room.root, "chat", {"ts": datetime.now(timezone.utc).isoformat(), "direction": "out", "chat_id": CHAT,
                                  "text": "Hello", "source": "presence:chat", "place_id": room.place.place_id,
                                  "client_message_id": "chat:a:r:0:3", "task_id": "focus",
                                  "transport": {"delivery": {"state": "authored"}}})  # a reply as the runner writes it
    stored = {row["text"]: row for row in rows(room.root)}
    with history_client(room.root) as client:
        page = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
    shown = {row["text"]: row for row in page["messages"] if row.get("place_history")}
    assert shown["message 1"]["place_history"]["sent_at"] == shown["message 1"]["ts"] == OLD.isoformat()
    assert shown["message 1"]["place_history"]["historical_import"]["received_at"] == stored["message 1"]["historical_import"]["received_at"]
    assert shown["message 2"]["place_history"]["sent_at"] == late < shown["message 2"]["ts"] == stored["message 2"]["ts"]
    assert "sent_at" not in shown["message 3"]["place_history"] and "sent_at" not in shown["Hello"]["place_history"]


def test_memory_reads_a_room_holding_imports_in_the_rooms_own_order(room):
    """C1 (Hub b954 review). In a room holding imports every row reads by its reading key, live ones included, and
    memory follows the room: live 100 and 102 of one second, a later edit of 100 and an import of 101 read
    [100, edit of 100, 101, 102] in the room, in explicit recall and in the room's open memory rows. Recall never
    re-sorts Main: two Main rows stamped against their stored order stay in stored order."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory import Memory
    from ouroboros.memory_inventory import open_room_rows
    ChronicleStore(room.root).ensure_activated()  # rows written from here on are open
    sent = SECOND.isoformat()
    for n in (100, 102):
        assert live(room, n, sent=sent, message={"source_order": n}, text=f"live {n}", facts=()).status_code == 200
    edit = live(room, 100, sent=sent, text="edit of 100", facts=(), source_event_id="chat:a:r:0:100:edit:r2",
                message={"source_order": 100, "event_kind": "edit", "revision": "r2", "edited_at": at(1)})
    assert edit.status_code == 200
    assert room.http.post("/presence/history", json=body(ordered(101, 101, text="import 101"))).status_code == 200
    for text, stamp in (("main stamped later", "2026-10-26T10:00:00+00:00"), ("main stamped earlier", "2026-10-26T09:00:00+00:00")):
        ds.append(room.root, "chat", {"ts": stamp, "direction": "in", "chat_id": 1, "text": text, "client_message_id": text})
    expected = ["live 100", "edit of 100", "import 101", "live 102"]
    with history_client(room.root) as client:
        page = client.get("/api/chat/history", params={"chat_id": CHAT}).json()
    assert [row["text"] for row in page["messages"] if row.get("history_id")] == expected  # the room
    recall = Memory(room.root).chat_history(count=10)
    assert sorted(expected, key=recall.index) == expected, recall  # explicit recall
    assert recall.index("main stamped later") < recall.index("main stamped earlier"), recall  # Main as stored
    names = {"chat:a:r:0:100": "live 100", "chat:a:r:0:102": "live 102", "chat:a:r:0:100:edit:r2": "edit of 100",
             "chat:a:r:0:101": "import 101"}
    assert [names[meta["client_message_id"]] for _address, meta, _pos in open_room_rows(room.root, CHAT)] == expected


def _one_second_with_an_import(room, *, archive_actor="person"):
    """Live 100 and 102 of one second, a later edit of 100 and an import of 101 (by ``archive_actor``)."""
    sent = SECOND.isoformat()
    for n in (100, 102):
        assert live(room, n, sent=sent, message={"source_order": n}, text=f"LIVE-{n}", facts=()).status_code == 200
    edit = live(room, 100, sent=sent, text="LIVE-EDIT-100", facts=(), source_event_id="chat:a:r:0:100:edit:r2",
                message={"source_order": 100, "event_kind": "edit", "revision": "r2"})
    assert edit.status_code == 200
    archive = ordered(101, 101, text="ARCHIVE-101", actor=archive_actor)
    assert room.http.post("/presence/history", json=body(archive)).status_code == 200


def _a_room_without_imports(room):
    """A second connected room holding no import, whose two live rows were sent in the reverse of their arrival."""
    key = conversation_key("chat", "a", "s", "")
    p.connect_place(room.root, connector="bridge", provider="chat", account_id="a", conversation_id="s", wakes=["mention"])
    for n, second in ((1, 9), (2, 1)):
        arrival = {**event(n, text=f"OTHER-{n}", sent=(SECOND + timedelta(seconds=second)).isoformat(),
                           source_event_id=f"chat:a:s:0:{n}"), "conversation_id": "s", "conversation_key": key, "facts": []}
        assert room.http.post("/presence/turn", json={"event": arrival}).status_code == 200
    return _stable_numeric_id("presence-conversation", key)


def _room_texts(root, chat_id):
    with history_client(root) as client:
        page = client.get("/api/chat/history", params={"chat_id": chat_id}).json()
    return [row["text"] for row in page["messages"] if row.get("history_id")]


@pytest.mark.parametrize("narrowed", [{"search": "LIVE-"}, {"actor_id": "person"}], ids=["search", "actor"])
def test_a_recall_narrowed_away_from_a_rooms_imports_still_reads_as_the_room(room, narrowed):
    """C1 follow-up (nanny probe). Whether a room reads by time is a fact of the room, never of the rows a reader
    selected: recall narrowed by text or by actor to the room's live rows reads them as the room does (live 100,
    the edit of 100, live 102) though no imported row is among them. A room holding no import keeps its stored
    order, as its endpoint does, even when its rows' send times run against it."""
    from ouroboros.memory import Memory
    _one_second_with_an_import(room, archive_actor="archivist")
    other = _a_room_without_imports(room)
    recall = Memory(room.root).chat_history(count=20, **narrowed)
    assert "ARCHIVE-101" not in recall, recall
    expected = ["LIVE-100", "LIVE-EDIT-100", "LIVE-102"]
    assert sorted(expected, key=recall.index) == expected, recall
    assert [text for text in _room_texts(room.root, CHAT) if text in expected] == expected  # the room's own order
    assert _room_texts(room.root, other) == ["OTHER-1", "OTHER-2"]
    stored = Memory(room.root).chat_history(count=20, search="OTHER-")
    assert stored.index("OTHER-1") < stored.index("OTHER-2"), stored


def test_open_memory_rows_read_as_their_room_after_its_imports_are_sealed(room):
    """A page that seals a room's imported row leaves only live rows open. The room still reads by time, so its open
    rows, every room's open rows and its focus lane keep live 100, the edit of 100, live 102, with physical
    positions; the open rows of a room holding no import keep their stored order."""
    from ouroboros import memory_view as mv
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory_inventory import open_room_rows, open_rows_by_room
    store = ChronicleStore(room.root)
    store.ensure_activated()  # rows written from here on are open
    _one_second_with_an_import(room)
    other = _a_room_without_imports(room)
    imported = next(entry for entry in open_room_rows(room.root, CHAT) if entry[1].get("historical_import"))
    mind = {"kind": "mind", "task_id": "reader", "focus": {"role": "root", "task_id": "reader"}}
    sealed = store.publish_page(room_id=str(CHAT), text="The archive was read.", author=mind,
                                covers={"rows": [imported[0]["row_sha256"]], "stream_span": [imported[2], imported[2]]})
    assert sealed.ok, sealed
    expected = ["chat:a:r:0:100", "chat:a:r:0:100:edit:r2", "chat:a:r:0:102"]
    opened = open_room_rows(room.root, CHAT)
    assert [meta["client_message_id"] for _address, meta, _pos in opened] == expected
    assert [pos for _address, _meta, pos in opened] != sorted(pos for _address, _meta, pos in opened)  # physical
    by_room = open_rows_by_room(room.root)
    assert [meta["client_message_id"] for _address, meta, _pos in by_room[str(CHAT)]] == expected
    focus = {"id": "focus1", "chat_id": 1, "metadata": {"usage_category": "consciousness", "wake_cause": "focus", "focus": {
        "room_chat_id": CHAT, "place_id": room.place.place_id, "conversation_key": KEY}}}
    view = mv.capture_memory_view(room.root, focus, mv.view_spec_for_task(focus, room.root))
    lane = [item["ref"].rsplit("/ ", 1)[1] for item in view.room["lane1"] if item["ref"].startswith(f"chat {CHAT} /")]
    assert lane == expected, lane
    stored = ["chat:a:s:0:1", "chat:a:s:0:2"]
    assert [meta["client_message_id"] for _address, meta, _pos in open_room_rows(room.root, other)] == stored
    assert [meta["client_message_id"] for _address, meta, _pos in by_room[str(other)]] == stored


def test_memory_reads_a_rooms_imports_without_building_or_missing_their_index(room):
    """A memory reader writes nothing (``test_memory_fallback``). With the derived indexes gone, as a store cutover
    brings imported rows back without them, open rows, every room's open rows and recall still read the room by time
    from its place arrivals, never "no imports", and build nothing; the room's endpoint read builds them."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory import Memory
    from ouroboros.memory_inventory import open_room_rows, open_rows_by_room
    from ouroboros.place_arrivals import history_import_rooms
    ChronicleStore(room.root).ensure_activated()
    _one_second_with_an_import(room)
    other = _a_room_without_imports(room)
    derived = {"dialogue_place_rooms", "dialogue_place_recent", "dialogue_place_imported", "dialogue_room_reading"}
    with ds.transaction(room.root, write=True) as conn:
        for name in derived:
            conn.execute(f"DROP INDEX {name}")

    def built():
        with ds.transaction(room.root, write=False) as conn:
            return derived & {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    def database_bytes():
        # SQLite may use PERSIST or WAL. Compare both bytes and file presence without assuming either mode.
        names = ("dialog.sqlite", "dialog.sqlite-wal", "dialog.sqlite-journal")
        return {name: path.read_bytes() for name in names if (path := room.root / "state" / name).exists()}
    before = database_bytes()
    assert history_import_rooms(room.root, (CHAT, other, 1)) == {CHAT}
    expected = ["chat:a:r:0:100", "chat:a:r:0:100:edit:r2", "chat:a:r:0:101", "chat:a:r:0:102"]
    assert [meta["client_message_id"] for _address, meta, _pos in open_room_rows(room.root, CHAT)] == expected
    assert [meta["client_message_id"] for _address, meta, _pos in open_rows_by_room(room.root)[str(CHAT)]] == expected
    recall = Memory(room.root).chat_history(count=20, search="LIVE-")
    assert sorted(["LIVE-100", "LIVE-EDIT-100", "LIVE-102"], key=recall.index) == ["LIVE-100", "LIVE-EDIT-100", "LIVE-102"]
    assert built() == set()
    assert database_bytes() == before
    assert _room_texts(room.root, CHAT) == ["LIVE-100", "LIVE-EDIT-100", "ARCHIVE-101", "LIVE-102"]
    assert built() == derived


def test_a_room_read_by_time_is_bounded_by_its_rows_stored_times(room):
    """Review rl-20261026T173426 (critical). A live edit of a 2024 message, received now, reads at its send time,
    before a 2025 message imported after it. The room's open span, its lane's ``since`` and its place's latest time
    bound the rows' stored times (``source_time_span``) instead of taking the first and last rows read: open
    2025-06-01 → the edit's receipt, never the receipt → 2025-06-01, in the room's view and in the directory of live
    rooms. The rows, their reading order, times and addresses stay as stored; a room holding no import still spans
    its first and last rows."""
    from ouroboros import memory_view as mv
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory_inventory import open_room_rows
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext
    ChronicleStore(room.root).ensure_activated()  # rows written from here on are open
    edit = live(room, 100, sent="2024-03-01T10:00:00+00:00", text="EDIT-OF-2024", facts=(),
                source_event_id="chat:a:r:0:100:edit:r2", message={
                    "source_order": 100, "event_kind": "edit", "revision": "r2",
                    "edited_at": datetime.now(timezone.utc).isoformat()})
    assert edit.status_code == 200, edit.text
    assert room.http.post("/presence/history", json=body(ordered(101, 101, text="IMPORT-2025"))).status_code == 200
    other = _a_room_without_imports(room)
    before = rows(room.root)
    stamp = {row["text"]: row["ts"] for row in before}
    sent, received = stamp["IMPORT-2025"], stamp["EDIT-OF-2024"]
    assert sent == SECOND.isoformat() < received  # an import's time is its send time, a live row's its receipt
    opened = open_room_rows(room.root, CHAT)
    assert [meta["client_message_id"] for _address, meta, _pos in opened] == ["chat:a:r:0:100:edit:r2", "chat:a:r:0:101"]
    focus = {"id": "focus1", "chat_id": 1, "metadata": {"usage_category": "consciousness", "wake_cause": "focus", "focus": {
        "room_chat_id": CHAT, "place_id": room.place.place_id, "conversation_key": KEY}}}
    view = mv.capture_memory_view(room.root, focus, mv.view_spec_for_task(focus, room.root))
    facts, span = view.room["facts"], f"open {mv._minute(sent)} → {mv._minute(received)};"
    assert (facts["earliest"], facts["latest"], facts["place"]["latest"]) == (sent, received, received)
    lane = [item for item in view.room["lane1"] if item["ref"].startswith(f"chat {CHAT} /")]
    assert facts["place"]["last_address"] == lane[-1]["address"]  # the last row as the room reads it
    assert view.room["since"] == mv._minute(sent)
    text = mv.render_room(view)
    assert span in text and f"### Open conversation since {mv._minute(sent)} (" in text, text
    listed = _memory_read(ToolContext(repo_dir=room.root, drive_root=room.root, task_id="reader", current_chat_id=1),
                          room_id="*").split("\n")
    line = next(line for line in listed if f"memory_read(room_id='{CHAT}', rows=true)" in line)
    assert span in line and f"last activity {mv._minute(received)};" in line, line
    first, last = stamp["OTHER-1"], stamp["OTHER-2"]
    line = next(line for line in listed if f"memory_read(room_id='{other}', rows=true)" in line)
    assert f"open {mv._minute(first)} → {mv._minute(last)};" in line and f"last activity {mv._minute(last)};" in line
    assert rows(room.root) == before and open_room_rows(room.root, CHAT) == opened  # nothing restamped or moved


def test_memory_rows_keep_and_name_their_stored_order_in_a_room_read_by_time(room, monkeypatch):
    """Review rl-20261026T173426 (advisory), kept as this reader's stated exception. ``memory_read(rows=true)`` reads
    by address: from/to and its ``next: from=`` bound stored positions, so it reads stored order whatever order its
    room reads in, and every page and the schema say so. In the room [live 100, edit of 100, import 101, live 102] it
    pages live 100, live 102, edit of 100, import 101, whole rows over several pages, each once; a room holding no
    import pages the same way."""
    import re

    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.tools import chronicle
    from ouroboros.tools.registry import ToolContext
    ChronicleStore(room.root).ensure_activated()
    _one_second_with_an_import(room)
    other = _a_room_without_imports(room)
    assert _room_texts(room.root, CHAT) == ["LIVE-100", "LIVE-EDIT-100", "ARCHIVE-101", "LIVE-102"]  # the room's order
    ctx = ToolContext(repo_dir=room.root, drive_root=room.root, task_id="reader", current_chat_id=1)
    whole = chronicle._memory_read(ctx, room_id=str(CHAT), rows=True).split("\n")
    assert whole[:2] == [f"room {CHAT}; rows", "complete: no further rows (stored order)"], whole
    widest = max(len(line) for line in whole[2:])  # a page holds the head with its from= bound and a whole row
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _tool: len(whole[0]) + 242 + widest)

    def paged(chat):
        seen, start, pages = [], {}, 0
        while True:
            page = chronicle._memory_read(ctx, room_id=str(chat), rows=True, **start).split("\n")
            pages += 1
            assert len(page) > 2 and "(chars " not in page[2], page  # whole rows, never a character window
            seen += [line.split("] ", 1)[1] for line in page[2:]]
            if page[1] == "complete: no further rows (stored order)":
                return seen, pages
            found = re.fullmatch(r"next: from=(\S+) \(same room and filters; stored order\)", page[1])
            assert found, page
            start = {"from": found.group(1)}
    texts, pages = paged(CHAT)
    assert texts == ["LIVE-100", "LIVE-102", "LIVE-EDIT-100", "ARCHIVE-101"] and pages > 1
    assert paged(other)[0] == ["OTHER-1", "OTHER-2"]
    schema = next(entry.schema for entry in chronicle.chronicle_tools() if entry.name == "memory_read")
    rows_doc = schema["parameters"]["properties"]["rows"]["description"]
    assert "stored order" in schema["description"] and "stored order" in rows_doc and "chat_history" in rows_doc
