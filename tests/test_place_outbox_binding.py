"""Enqueue provenance survives disconnect; current account scopes still govern delivery."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import places, place_guard
from ouroboros.gateway import host_service
from tests.test_tool_effects import published_tool as published_tool


@pytest.fixture
def client(tmp_path, monkeypatch):
    async def authenticated(*_):
        return "connector", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    with TestClient(host_service.create_host_service_app(tmp_path)) as value:
        yield value


TARGET = {"provider": "service", "account_id": "C", "conversation_id": "thread"}


def bind(client, **values):
    return client.get("/presence/place", params={**TARGET, "operation": "bind", **values})


def check(client, binding, **values):
    return client.get("/presence/place", params={**TARGET, "operation": "check", "binding": json.dumps(binding), **values})


def test_unbound_capture_current_scope_and_legacy_lookup(client, tmp_path):
    other = places.connect_place(tmp_path, connector="connector", provider="service", account_id="A", state="silent")
    captured = bind(client).json()
    assert captured["allowed"] and captured["state"] == "unbound" and captured["binding"]["place_id"] == ""
    assert captured["binding"]["account_id"] == "C" and captured["binding"]["connector"] == "connector"
    assert check(client, captured["binding"]).json()["allowed"]
    assert client.get("/presence/place", params=TARGET).status_code == 404
    assert not bind(client, require_place="true").json()["allowed"]
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C", state="silent")
    assert check(client, captured["binding"]).json()["code"] == "place_silent"
    places.edit_place(tmp_path, own.place_id, "set_state", author="owner", state="active")
    assert check(client, captured["binding"]).json()["allowed"]
    assert places.get_place(tmp_path, other.place_id).state == "silent"


def test_bound_capture_cannot_become_unbound_when_disconnected(client, tmp_path):
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client, require_place="true").json()
    assert captured["allowed"] and captured["binding"]["place_id"] == own.place_id
    places.edit_place(tmp_path, own.place_id, "set_state", author="owner", state="paused")
    assert check(client, captured["binding"]).json()["code"] == "place_paused"
    places.edit_place(tmp_path, own.place_id, "disconnect", author="owner")
    held = check(client, captured["binding"]).json()
    assert not held["allowed"] and held["code"] == "place_disconnected"
    assert held["binding"] == captured["binding"]
    assert bind(client).json()["allowed"], "a newly chosen unbound operation is distinct from retained bound work"
    reconnected = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    assert reconnected.place_id == own.place_id
    assert check(client, captured["binding"]).json()["code"] == "place_disconnected"
    fresh = bind(client).json()
    assert fresh["allowed"] and fresh["binding"] != captured["binding"]
    places.edit_place(tmp_path, reconnected.place_id, "add_note", author="owner", text="Keep this connection.")
    assert check(client, fresh["binding"]).json()["allowed"], "ordinary edits do not replace a connection"


def test_legacy_place_keeps_observed_revision_until_reconnected(client, tmp_path):
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    path = places.places_path(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["connections"]
    path.write_text(json.dumps(document), encoding="utf-8")
    captured = bind(client).json()["binding"]
    assert captured["connected_revision"] == own.revision
    places.edit_place(tmp_path, own.place_id, "set_state", author="owner", state="silent")
    assert check(client, captured).json()["code"] == "place_silent"
    places.edit_place(tmp_path, own.place_id, "set_state", author="owner", state="active")
    assert check(client, captured).json()["allowed"]
    assert places.connect_place(tmp_path, connector="connector", provider="service", account_id="C").connected_revision == own.revision
    places.edit_place(tmp_path, own.place_id, "disconnect", author="owner")
    places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    assert check(client, captured).json()["code"] == "place_disconnected"


@pytest.mark.parametrize("older_action", ["edit", "disconnect_reconnect"])
def test_older_writer_cannot_reuse_stale_connection_metadata(client, tmp_path, older_action):
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client).json()["binding"]
    path = places.places_path(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    row = document["places"][own.place_id]
    witness = dict(document["connections"][own.place_id])
    # The older writer keeps unknown top-level metadata, but advances each
    # changed row's revision; a reconnect reuses the deterministic Place ID.
    if older_action == "disconnect_reconnect":
        del document["places"][own.place_id]
        document["revision"] += 1
        path.write_text(json.dumps(document), encoding="utf-8")
        assert check(client, captured).json()["code"] == "place_disconnected"
        document["places"][own.place_id] = row
    else:
        row["model"] = "older-body-choice"
    document["revision"] += 1
    row["revision"] = document["revision"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert document["connections"][own.place_id] == witness
    held = check(client, captured).json()
    assert not held["allowed"] and held["code"] == "place_disconnected" and held["binding"] == captured
    fresh = bind(client).json()
    assert fresh["allowed"] and fresh["binding"]["connected_revision"] == row["revision"]
    assert check(client, fresh["binding"]).json()["allowed"]
    places.edit_place(tmp_path, own.place_id, "add_note", author="owner", text="Current body observed this row.")
    assert check(client, fresh["binding"]).json()["allowed"]
    places.edit_place(tmp_path, own.place_id, "disconnect", author="owner")
    assert own.place_id not in json.loads(path.read_text(encoding="utf-8"))["connections"]


@pytest.mark.parametrize("state", ["silent", "paused"])
def test_new_wildcard_hold_matches_dispatch_for_queued_account_work(client, tmp_path, published_tool, state):
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client).json()["binding"]
    broad = places.connect_place(tmp_path, connector="connector", provider="service", account_id="*", state=state)
    name, tool = published_tool("message", target=["account_id", "to"], conversation="room", thread="thread")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    args = {"account_id": "C", "to": "recipient", "room": "thread"}
    assert state.title() in place_guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert bind(client).json()["code"] == f"place_{state}"
    held = check(client, captured).json()
    assert not held["allowed"] and held["code"] == f"place_{state}" and held["binding"] == captured
    # The old query remains an ingress-style lookup, with no binding fields.
    assert client.get("/presence/place", params=TARGET).json() == {
        "place_id": own.place_id, "label": own.label, "state": "active"}
    places.edit_place(tmp_path, broad.place_id, "set_state", author="owner", state="active")
    assert not place_guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert check(client, captured).json()["allowed"]


@pytest.mark.parametrize("broad_account,room,thread", [("", "", ""), ("*", "thread", ""), ("*", "thread", "topic")])
def test_outgoing_precedence_keeps_specific_active_scope(client, tmp_path, broad_account, room, thread):
    places.connect_place(tmp_path, connector="connector", provider="service", account_id=broad_account, state="silent")
    own = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C", conversation_id=room)
    if thread:
        own = places._write(tmp_path, own.place_id, lambda row: replace(row, thread_id=thread))
    target = {**TARGET, "thread_id": thread}
    resolved, known = places.place_for_target(tmp_path, connector="connector", provider="service",
        targets={key: [value] for key, value in target.items() if key != "provider" and value},
        conversation="conversation_id", thread="thread_id")
    assert known and resolved == own
    captured = bind(client, thread_id=thread).json()
    assert captured["allowed"] and captured["binding"]["place_id"] == own.place_id
    assert check(client, captured["binding"], thread_id=thread).json()["allowed"]


@pytest.mark.parametrize("account", ["", "*"])
def test_broad_scope_holds_unbound_account_and_known_thread(client, tmp_path, account):
    places.connect_place(tmp_path, connector="connector", provider="service", account_id="foreign")
    broad = places.connect_place(tmp_path, connector="connector", provider="service", account_id=account, state="silent")
    response = bind(client).json()
    assert not response["allowed"] and response["binding"]["place_id"] == broad.place_id
    assert response["code"] == "place_silent"
    places.edit_place(tmp_path, broad.place_id, "set_state", author="owner", state="active")
    assert check(client, response["binding"]).json()["allowed"]


def test_same_account_unknown_scope_stays_unknown(client, tmp_path):
    places.connect_place(tmp_path, connector="connector", provider="service", account_id="C", conversation_id="other", state="silent")
    response = bind(client).json()
    assert not response["allowed"] and response["binding"] is None
    assert response["code"] == "place_unattributable"
    assert bind(client, require_place="true").json()["code"] == "place_unattributable"
    opaque = bind(client, conversation_id="").json()
    assert not opaque["allowed"] and opaque["code"] == "place_silent"


@pytest.mark.parametrize("field,value", [("account_id", "Alias"), ("connector", "other"), ("provider", "other"),
    ("conversation_id", "other"), ("version", True), ("place_id", None)])
def test_snapshot_cannot_be_reused_for_another_target(client, field, value):
    captured = bind(client).json()["binding"]
    captured[field] = value
    response = check(client, captured)
    assert response.status_code == 400 and response.json()["code"] == "place_binding_invalid"


def test_missing_binding_auth_failure_and_unreadable_store_are_not_unbound(client, monkeypatch):
    for value in (None, {}, [], "unbound"):
        assert check(client, value).status_code == 400
    assert bind(client, require_place="yes").status_code == 400
    assert bind(client, account_id="").status_code == 400
    def unreadable(*_):
        raise places.PlaceError("places_unreadable")
    monkeypatch.setattr(places, "load_places", unreadable)
    assert bind(client).status_code == 503
    async def refused(*_):
        raise host_service.HostServiceAuthError("token no longer valid")
    monkeypatch.setattr(host_service, "_authenticated", refused)
    assert bind(client).status_code == 403


def test_existing_bound_scope_change_is_not_a_new_unbound_account(client, tmp_path, monkeypatch):
    original = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client).json()["binding"]
    monkeypatch.setattr(places, "load_places", lambda _: [replace(original, account_id="other")])
    assert check(client, captured).json()["code"] == "place_scope_changed"


def test_scope_disappearing_during_recheck_does_not_reuse_earlier_active_fact(client, tmp_path, monkeypatch):
    places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client).json()["binding"]
    monkeypatch.setattr(places, "outbox_place", lambda *_a, **_k: None)
    monkeypatch.setattr(places, "place_for_target", lambda *_a, **_k: (None, True))
    result = check(client, captured).json()
    assert not result["allowed"] and result["code"] == "place_scope_changed"
    assert result["binding"] == captured


@pytest.mark.parametrize("thread_id", ["", "topic"])
def test_destination_binding_rechecks_its_original_place_without_widening_ingress(client, tmp_path, thread_id):
    original = places.connect_place(tmp_path, connector="connector", provider="service", account_id="A", conversation_id="source")
    places._write(tmp_path, original.place_id, lambda row: replace(row, destination={
        "account_id": "C", "conversation_id": "thread", "thread_id": thread_id}))
    captured = bind(client, thread_id=thread_id, require_place="true").json()
    assert captured["allowed"] and captured["binding"]["place_id"] == original.place_id
    assert check(client, captured["binding"], thread_id=thread_id, require_place="true").json()["allowed"]
    assert places.resolve_place(tmp_path, connector="connector", **TARGET, thread_id=thread_id) is None
    places._write(tmp_path, original.place_id, lambda row: replace(row, destination={
        "account_id": "C", "conversation_id": "different", "thread_id": thread_id}))
    changed = check(client, captured["binding"], thread_id=thread_id).json()
    assert not changed["allowed"] and changed["code"] == "place_scope_changed"
    places._write(tmp_path, original.place_id, lambda row: replace(row, destination={
        "account_id": "C", "conversation_id": "thread", "thread_id": thread_id}))
    places.edit_place(tmp_path, original.place_id, "set_state", author="owner", state="silent")
    held = check(client, captured["binding"], thread_id=thread_id).json()
    assert not held["allowed"] and held["code"] == "place_silent"
    places.edit_place(tmp_path, original.place_id, "disconnect", author="owner")
    removed = check(client, captured["binding"], thread_id=thread_id).json()
    assert not removed["allowed"] and removed["code"] == "place_disconnected"
    assert removed["binding"] == captured["binding"]


def test_opaque_binding_stays_opaque_at_delivery(client, tmp_path):
    places.connect_place(tmp_path, connector="connector", provider="service", account_id="C", conversation_id="existing-room")
    captured = bind(client, conversation_id="").json()
    assert captured["allowed"] and captured["binding"]["conversation_id"] == ""
    assert check(client, captured["binding"], conversation_id="").json()["allowed"]
    changed = check(client, captured["binding"], conversation_id="generated-message-id")
    assert changed.status_code == 400 and changed.json()["code"] == "place_binding_invalid"


@pytest.mark.parametrize("state", ["silent", "paused"])
def test_opaque_primary_lookup_keeps_declared_destination_hold(client, tmp_path, published_tool, state):
    primary = places.connect_place(tmp_path, connector="connector", provider="service", account_id="C")
    captured = bind(client, conversation_id="").json()["binding"]
    assert captured["place_id"] == primary.place_id
    source = places.connect_place(tmp_path, connector="connector", provider="service", account_id="A",
                                  conversation_id="source", state=state)
    source = places._write(tmp_path, source.place_id, lambda row: replace(row, destination={"account_id": "C"}))
    name, tool = published_tool("message", target=["account_id", "to"])
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    args = {"account_id": "C", "to": "recipient"}
    assert state.title() in place_guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    # The legacy opaque lookup sees only the primary Active account; the
    # outgoing matcher also sees the other Place's declared destination.
    assert places.outbox_place(tmp_path, connector="connector", **{**TARGET, "conversation_id": ""}) == primary
    queued = bind(client, conversation_id="").json()
    assert not queued["allowed"] and queued["code"] == f"place_{state}"
    held = check(client, captured, conversation_id="").json()
    assert not held["allowed"] and held["code"] == f"place_{state}" and held["binding"] == captured
    places.edit_place(tmp_path, source.place_id, "set_state", author="owner", state="active")
    assert not place_guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert bind(client, conversation_id="").json()["allowed"]
    assert check(client, captured, conversation_id="").json()["allowed"]
