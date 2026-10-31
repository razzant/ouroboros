"""Enqueue provenance survives disconnect; current account scopes still govern delivery."""
from dataclasses import replace
import json

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway import host_service


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
