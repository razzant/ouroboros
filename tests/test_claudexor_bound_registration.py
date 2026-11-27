"""Bound project registration uses the engine's exact identities on the wire."""
import json
from urllib.parse import quote

import httpx
import pytest

from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, DaemonEndpoint
from ouroboros.gateways.claudexor_maintenance import bound_registration_supported

PID, STORE, ROOT = "project/1", "store &1", "/fixture/100% + папка"
IDENTITY = {"projectId": PID, "storeId": STORE, "root": ROOT}
FIELDS = {"expectedStoreId": STORE, "expectedRoot": ROOT}
RELEASE = {"method": "POST", "path": "/v2/projects/:id/release"}
INSPECT = {"method": "GET", "path": "/v2/projects/:id/registration"}


@pytest.fixture
def wire(monkeypatch):
    requests, replies = [], []
    client = httpx.Client

    def handle(request):
        requests.append(request)
        assert request.method != "DELETE"
        response = replies.pop(0)
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)

    monkeypatch.setattr(httpx, "Client", lambda **kw: client(**kw, transport=httpx.MockTransport(handle)))
    with ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "fixture")) as gateway:
        yield gateway, requests, replies
    assert not replies


@pytest.mark.parametrize("routes,capable", [([], False), ([RELEASE], False), ([INSPECT], False),
                                            ([RELEASE, INSPECT], True)])
def test_only_the_pair_identifies_support(wire, routes, capable):
    gateway, requests, replies = wire
    replies.append({"operations": routes})
    assert bound_registration_supported(gateway) is capable
    assert [(r.method, r.url.path) for r in requests] == [("GET", "/v2/operations")]


@pytest.mark.parametrize("created", [True, False, None])
def test_registration_preserves_the_returned_folder_and_optional_store(wire, created):
    gateway, requests, replies = wire
    replies.append({"id": PID, "root": ROOT, **({"created": created, "storeId": STORE} if created is not None else {})})
    receipt = gateway.project_registration_receipt("/fixture/alias")
    assert receipt == {"id": PID, "created": created, "registration_root": ROOT,
                       "registration_store_id": STORE if created is not None else None}
    [request] = requests
    assert (request.method, request.url.path) == ("POST", "/v2/projects")
    assert json.loads(request.content) == {"root": "/fixture/alias"}
    assert request.headers["Idempotency-Key"] and request.headers["X-Claudexor-Store-Identity"] == "1"


@pytest.mark.parametrize("release,state", [(True, "released"), (True, "absent"), (False, "present"), (False, "absent")])
def test_bound_registration_addresses_one_record(wire, release, state):
    gateway, requests, replies = wire
    answer = {**IDENTITY, "state": state}
    replies.append(answer)
    assert gateway.bound_registration(PID, STORE, ROOT, release=release) == answer
    [request] = requests
    path = "/v2/projects/" + quote(PID, safe="")
    assert request.url.raw_path.split(b"?")[0].decode() == path + ("/release" if release else "/registration")
    if release:
        assert request.method == "POST" and json.loads(request.content) == FIELDS and not request.url.query
    else:
        assert request.method == "GET" and dict(request.url.params) == FIELDS and not request.content


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("invalid", [{"projectId": "other"}, {"storeId": None}, {"root": ROOT + "/"}, {"state": "unknown"},
                                     {"registryRemoved": True, "state": None}])
def test_mismatched_or_legacy_success_cannot_settle(wire, release, invalid):
    gateway, _, replies = wire
    replies.append({**IDENTITY, "state": "absent", **invalid})
    with pytest.raises(ClaudexorUnavailable, match="addressed registration"):
        gateway.bound_registration(PID, STORE, ROOT, release=release)


@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("status,code", [(404, "project_not_found"), (409, "store_mismatch"),
                                         (409, "project_root_mismatch")])
def test_route_errors_never_become_absence(wire, release, status, code):
    gateway, _, replies = wire
    replies.append(httpx.Response(status, json={"code": code, "message": "fixture refusal"}))
    with pytest.raises(ClaudexorUnavailable) as problem:
        gateway.bound_registration(PID, STORE, ROOT, release=release)
    assert (problem.value.status_code, problem.value.code) == (status, code)


@pytest.mark.parametrize("refusal", ["store_mismatch", "project_root_mismatch"])
def test_guard_refusal_creates_no_removal_debt_that_blocks_admission(wire, tmp_path, monkeypatch, refusal):
    from ouroboros import delegate_custody_current as current, delegate_registration_policy as policy
    from tests._bound_registration import binding

    gateway, requests, replies = wire
    monkeypatch.setattr(type(gateway), "store_identity", property(lambda self: STORE))
    proof = binding(STORE, ROOT)
    assert policy.owe_registration(tmp_path, PID, proof)
    replies.extend([{"operations": [RELEASE, INSPECT]},
                    httpx.Response(409, json={"code": refusal, "message": "guard refused"})])
    policy.retire_owed(tmp_path, gateway)
    assert current.owed_registrations(tmp_path)[0]["registration_root"] == ROOT
    assert current.removal_status(tmp_path, PID, proof)[0] == ""
    assert [(r.method, r.url.path.rsplit("/", 1)[-1]) for r in requests] == [("GET", "operations"), ("POST", "release")]
    assert json.loads(requests[-1].content) == FIELDS


@pytest.mark.parametrize("missing", ["store", "root", "release", "inspect"])
def test_missing_binding_or_operation_keeps_the_registration_without_fencing_new_starts(wire, tmp_path, monkeypatch, missing):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, delegate_registration_policy as policy
    from tests._bound_registration import binding

    gateway, requests, replies = wire
    monkeypatch.setattr(type(gateway), "store_identity", property(lambda self: STORE))
    proof = binding(STORE, ROOT)
    if missing in ("store", "root"):
        proof.pop("registration_store_id" if missing == "store" else "registration_root")
    else:
        replies.append({"operations": [INSPECT if missing == "release" else RELEASE]})
    assert policy.owe_registration(tmp_path, PID, proof)
    policy.retire_owed(tmp_path, gateway)
    [kept] = [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.PROJECT_RETIRED]
    assert kept["project_kept"] and kept["reason"] == "registration_unproven"
    assert not current.owed_registrations(tmp_path) and current.removal_status(tmp_path, PID, proof)[0] == ""
    assert all(r.method == "GET" and r.url.path == "/v2/operations" for r in requests)
