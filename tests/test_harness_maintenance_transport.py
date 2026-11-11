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
PROGRAM = {"path": "/vendor/cli", "realPath": "/vendor/releases/cli", "version": "1.2.3",
           "versionOutput": "Vendor CLI 1.2.3", "overrideEnv": None, "overrideValue": None}
ACTION = {"available": True, "reason": None, "commands": [
    {"kind": "exec", "command": "/vendor/cli", "args": ["update"], "env": {"CHANNEL": "stable"}}]}
ENTRY = {"harness": "future+cli", "selection": PROGRAM,
         "newest": {"status": "not_read", "source": "https://vendor.example/latest"},
         "updateAvailable": "cannot_tell", "releaseNotesUrl": None,
         "actions": {"install": {"available": False, "reason": "A program is selected.", "commands": []},
                     "update": ACTION, "exactVersion": ACTION},
         "operation": None, "observedAt": "2026-11-11T00:00:00Z"}
OPERATION = {"id": "maintenance-one", "harness": "agy", "state": "running", "phase": "installing",
             "target": {"kind": "latest", "version": None}, "before": PROGRAM, "after": None,
             "requester": {"source": "control-api", "requestId": "stable-key"},
             "installationHome": "/installation", "commands": [{"binary": "/bin/sh", "args": ["/tmp/vendor.sh"],
                 "env": {}, "script": {"url": "https://vendor.example/install.sh", "size": 52, "sha256": "a" * 64}}],
             "elsewhere": None, "activeRunsAtStart": 2, "exitCode": None, "outcome": "unknown",
             "mutation": "unknown", "termination": "unconfirmed", "limitations": ["The installer may still run."],
             "problem": None, "progress": ["Installing the vendor program"],
             "createdAt": "2026-11-11T00:00:00Z", "startedAt": "2026-11-11T00:00:01Z", "finishedAt": None}


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


@pytest.mark.parametrize("status,update_available", [("not_read", "cannot_tell"), ("available", "yes"),
                                                   ("available", "no"), ("unavailable", "cannot_tell")])
def test_inventory_keeps_generic_harness_facts_and_repeated_filters(client, status, update_available):
    row = deepcopy(ENTRY)
    row["updateAvailable"] = update_available
    row["newest"]["status"] = status
    if status != "not_read":
        row["newest"]["observedAt"] = row["observedAt"]
        row["newest"].update({"version": "opaque-vendor-version"} if status == "available" else {
            "problem": {"code": "source_unavailable", "message": "The vendor did not answer."}})
    row["operation"] = OPERATION
    payload = {"observedAt": row["observedAt"], "harnesses": [row]}
    gateway, seen = client(lambda request: httpx.Response(200, json=payload))
    assert gateway.maintenance_harnesses(["agy", "future+cli"], fresh=True, check_latest=True) == payload
    assert seen[-1].url.params.get_list("harness") == ["agy", "future+cli"]
    assert dict(seen[-1].url.params)["checkLatest"] == "true"
    assert "fixture-maintenance-token" not in json.dumps(payload)


@pytest.mark.parametrize("change", ["old_inventory", "old_selection", "missing_actions", "bad_action"])
def test_inventory_without_vendor_contract_is_unavailable_not_translated(client, change):
    row = deepcopy(ENTRY)
    if change == "old_inventory":
        row = {"harness": "future+cli", "maintainable": True, "targets": ["latest"],
               "selection": {"kind": "managed", "binary": "/old/cli", "version": "1.2.3"}}
    elif change == "old_selection":
        row["selection"] = {"binary": "/old/cli", "version": "1.2.3"}
    elif change == "missing_actions":
        del row["actions"]
    else:
        row["actions"]["update"]["available"] = "false"
    gateway, seen = client(lambda request: httpx.Response(200, json={"observedAt": "now", "harnesses": [row]}))
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_harnesses()
    assert caught.value.code == "maintenance_unavailable" and caught.value.status_code == 503
    assert all(request.method == "GET" for request in seen)


@pytest.mark.parametrize("removed", ["commands", "outcome", "requester"])
def test_old_receipt_after_create_keeps_acceptance_unknown(client, removed):
    old = deepcopy(OPERATION)
    del old[removed]
    gateway, _ = client(lambda request: httpx.Response(202, json=old))
    with pytest.raises(wire.ClaudexorUnavailable) as caught:
        gateway.maintenance_create(REQUEST, "stable-key")
    assert caught.value.problem["context"] == {
        "requestId": "stable-key", "acceptance": "unknown",
        "remedy": "Retry the same body with the same Idempotency-Key"}


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


@pytest.mark.parametrize("key", ["", "bad\nkey", "x" * 257])
def test_invalid_key_never_contacts_engine(client, key):
    gateway, seen = client(lambda request: pytest.fail("Invalid key must not be sent"))
    with pytest.raises(ValueError):
        gateway.maintenance_create(REQUEST, key)
    assert not seen


@pytest.mark.parametrize("cancel", [False, True])
def test_connect_receipt_reads_bound_catalog_and_operation_calls(client, cancel):
    gateway, seen = client(lambda _request: httpx.Response(200, json=OPERATION))
    call = gateway.maintenance_cancel if cancel else gateway.maintenance_operation
    assert call(OPERATION["id"], timeout_sec=0.75) == OPERATION
    assert all(request.extensions["timeout"] == dict(connect=0.75, read=0.75, write=0.75, pool=0.75)
               for request in seen)
