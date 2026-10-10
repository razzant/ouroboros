"""Engine maintenance wire: negotiation, exact envelopes and same-key recovery."""
from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest

from ouroboros.gateways import claudexor as wire

OPERATIONS = ["get:maintenance.harnesses", "post:maintenance.operations",
              "get:maintenance.operations.id", "post:maintenance.operations.id.cancel"]
REQUEST = {"harness": "agy", "target": {"kind": "latest"}}
OPERATION = {"id": "maintenance-one", "harness": "agy", "state": "running", "phase": "installing",
             "target": {"kind": "latest", "version": None}, "before": None, "after": None,
             "mutation": "unknown", "termination": "unconfirmed", "limitations": ["in_place_replacement"],
             "problem": None, "progress": ["Installing the vendor program"]}


@pytest.fixture
def client(monkeypatch):
    original = httpx.Client
    clients = []

    def build(respond, operations=OPERATIONS):
        seen = []

        def handle(request):
            seen.append(request)
            assert request.headers["Authorization"] == "Bearer fixture-maintenance-token"
            assert request.headers[wire.PROTOCOL_HEADER] == str(wire.CLAUDEXOR_PROTOCOL_MAJOR)
            if request.url.path == "/v2/operations":
                return httpx.Response(200, json={"operations": [{"id": value} for value in operations]})
            return respond(request)

        def factory(**kwargs):
            assert kwargs["trust_env"] is False
            value = original(**kwargs, transport=httpx.MockTransport(handle))
            clients.append(value)
            return value

        monkeypatch.setattr(wire.httpx, "Client", factory)
        return wire.ClaudexorGateway(wire.DaemonEndpoint("127.0.0.1", 1, "fixture-maintenance-token")), seen

    yield build
    for value in clients:
        value.close()


def test_inventory_keeps_generic_harness_facts_and_repeated_filters(client):
    payload = {"observedAt": "2026-10-09T00:00:00Z", "harnesses": [
        {"harness": "agy", "mechanism": "future_vendor_updater", "maintainable": True,
         "targets": ["latest"], "available": None, "operation": None}]}
    gateway, seen = client(lambda request: httpx.Response(200, json=payload))
    assert gateway.maintenance_harnesses(["agy", "future+cli"], fresh=True, check_latest=True) == payload
    assert seen[-1].url.params.get_list("harness") == ["agy", "future+cli"]
    assert dict(seen[-1].url.params)["checkLatest"] == "true"
    assert "fixture-maintenance-token" not in json.dumps(payload)


def test_old_engine_refuses_before_mutation_and_never_reinstalls(client):
    gateway, seen = client(lambda request: pytest.fail("No maintenance call on an old engine"), [])
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "request-one")
    assert caught.value.code == "maintenance_unavailable"
    assert caught.value.status_code == 503
    assert [request.url.path for request in seen] == ["/v2/operations"]


def test_control_problem_keeps_existing_operation_remedy(client):
    problem = {"code": "maintenance_already_active", "message": "An operation is active",
               "retryable": False, "context": {"operationId": "maintenance-old"},
               "requiredActions": ["inspect_operation"], "evidenceRefs": ["command:maintenance-old"]}
    gateway, seen = client(lambda request: httpx.Response(409, json=problem))
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "request-one")
    assert caught.value.problem == problem
    assert caught.value.status_code == 409
    assert seen[-1].headers["Idempotency-Key"] == "request-one"
    assert json.loads(seen[-1].content) == REQUEST


def test_lost_create_retains_key_and_explicit_rejoin_creates_no_second_operation(client):
    accepted = {}
    attempts = []

    def respond(request):
        key = request.headers["Idempotency-Key"]
        body = json.loads(request.content)
        attempts.append((key, body))
        first = key not in accepted
        accepted.setdefault(key, deepcopy(OPERATION))
        if first:
            raise httpx.ReadError("reply lost after acceptance", request=request)
        return httpx.Response(202, json=accepted[key])

    gateway, seen = client(respond)
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "stable-key")
    assert caught.value.problem["context"]["acceptance"] == "unknown"
    assert caught.value.problem["context"]["requestId"] == "stable-key"
    assert len(attempts) == 1  # Transport never retries automatically.
    result = gateway.maintenance_create(REQUEST, "stable-key")
    assert result == OPERATION and len(accepted) == 1
    assert attempts == [("stable-key", REQUEST), ("stable-key", REQUEST)]
    assert len([r for r in seen if r.method == "POST"]) == 2


def test_capability_read_failure_is_not_a_submitted_create(client, monkeypatch):
    gateway, seen = client(lambda request: pytest.fail("No POST"))
    monkeypatch.setattr(gateway, "operations", lambda: (_ for _ in ()).throw(
        wire.ClaudexorUnavailable("daemon_unreachable", "Unavailable before submission")))
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "request-one")
    assert not getattr(caught.value, "problem", {}).get("context", {}).get("acceptance")
    assert seen == []


def test_unreadable_success_is_unknown_and_status_cancel_never_invent_settlement(client):
    gateway, _ = client(lambda request: httpx.Response(202, json={}))
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "request-one")
    assert caught.value.problem["context"]["acceptance"] == "unknown"
    gateway, seen = client(lambda request: httpx.Response(200, json=OPERATION))
    assert gateway.maintenance_operation(OPERATION["id"]) == OPERATION
    assert gateway.maintenance_cancel(OPERATION["id"]) == OPERATION
    assert seen[-1].url.path.endswith("/maintenance-one/cancel")
    with pytest.raises(wire.ClaudexorUnavailable, match="identity"):
        gateway.maintenance_operation("another-operation")


@pytest.mark.parametrize("key", ["", " ", "bad\nkey", "x" * 257])
def test_invalid_key_never_contacts_engine(client, key):
    gateway, seen = client(lambda request: pytest.fail("Invalid key must not be sent"))
    with pytest.raises(ValueError):
        gateway.maintenance_create(REQUEST, key)
    assert not seen
