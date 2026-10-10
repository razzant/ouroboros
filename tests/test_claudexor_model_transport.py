"""Exact model payload transport and explicit delivery custody over the real HTTP client."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import httpx
import pytest

from ouroboros.gateways import claudexor as cx


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _ref(data, resource_id="res-result"):
    return {"resourceId": resource_id, "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "sizeBytes": len(data)}


def _request():
    return {
        "source": "codex", "model": "exact-model", "account": {"mode": "pin", "profileId": "chosen"},
        "messages": [
            {"role": "system", "content": "Own SYSTEM\r\nКонституция 🐍"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": '{ "path": "a" }'}},
                {"id": "call-b", "type": "function", "function": {"name": "read", "arguments": '{"path":"b"}'}},
            ], "nativeContinuation": {
                "route": {"source": "codex", "credentialProfileId": "chosen", "accountFingerprint": "account", "model": "exact-model"},
                "format": "responses", "payload": [{"id": "rs_1", "encrypted_content": "opaque+/==\r\n"}],
            }},
            {"role": "tool", "tool_call_id": "call-a", "content": "sk-" + "TESTONLY" * 12},
            {"role": "tool", "tool_call_id": "call-b", "content": "\n\tcomplete result\n"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }}],
        "toolChoice": {"type": "function", "function": {"name": "read"}},
        "options": {"reasoningEffort": "high", "parallelToolCalls": True},
    }


@pytest.fixture
def gateway_factory(monkeypatch):
    clients = []
    factory = httpx.Client

    def make(handler):
        def client(**kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["headers"]["Authorization"] == "Bearer test-control-token"
            result = factory(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(result)
            return result

        monkeypatch.setattr(cx.httpx, "Client", client)
        return cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "test-control-token"))

    yield make
    for client in clients:
        client.close()


class ModelWire:
    """Stateful fake of the existing upload receipts and the new operation API.

    Lost replies occur AFTER their effect. The independent wire observations
    prove the client rejoins the handle rather than re-uploading or generating.
    """

    def __init__(self, lose=None):
        self.calls = []
        self.lose = lose
        self.upload = None
        self.create = None
        self.finalized = None
        self.payload = None
        self.operation = None
        self.upload_count = 0
        self.generation_count = 0
        self.result = _bytes({"outcome": "completed", "message": {"role": "assistant", "content": "done 🐍"}})

    def __call__(self, request):
        self.calls.append(request)
        assert request.headers["Authorization"] == "Bearer test-control-token"
        assert request.headers[cx.PROTOCOL_HEADER] == str(cx.CLAUDEXOR_PROTOCOL_MAJOR)
        stage = (request.method, request.url.path)
        response = self.respond(request)
        if self.lose == stage:
            self.lose = None
            raise httpx.ReadError("reply lost after effect", request=request)
        return response

    def respond(self, request):
        method, path = request.method, request.url.path
        if method == "POST" and path == "/v2/uploads":
            identity = (request.headers["Idempotency-Key"], request.content)
            if self.create is not None and identity != self.create:
                return httpx.Response(409, json={"code": "idempotency_conflict", "message": "changed request"})
            if self.create is None:
                self.create = identity
                size = json.loads(request.content)["sizeBytes"]
                self.upload = {"uploadId": "upl-one", "state": "open", "receivedBytes": 0, "expectedBytes": size}
                self.upload_count += 1
            # The engine returns its original create receipt on replay.
            return httpx.Response(201, json={"uploadId": "upl-one", "state": "open"})
        if method == "GET" and path == "/v2/uploads/upl-one":
            if self.upload is None:
                return httpx.Response(404, json={"code": "upload_not_found", "message": "no such upload"})
            return httpx.Response(200, json=self.upload)
        if method == "PUT" and path == "/v2/uploads/upl-one/bytes":
            assert self.upload["state"] == "open"
            assert request.headers["Content-Type"] == "application/octet-stream"
            self.payload = request.content
            assert len(self.payload) == self.upload["expectedBytes"]
            self.upload.update(state="uploaded", receivedBytes=len(self.payload))
            return httpx.Response(200, json=self.upload)
        if method == "POST" and path == "/v2/uploads/upl-one/finalize":
            identity = (request.headers["Idempotency-Key"], request.content)
            if self.finalized is not None:
                assert identity == self.finalized[0]
                return httpx.Response(201, json=self.finalized[1])
            assert self.upload["state"] == "uploaded"
            ref = _ref(self.payload, "res-request")
            assert json.loads(request.content) == {"expectedSha256": ref["sha256"]}
            resource = {**ref, "purpose": "model", "kind": "file", "mime": "application/json"}
            self.finalized = (identity, resource)
            self.upload = None
            return httpx.Response(201, json=resource)
        if method == "POST" and path == "/v2/model-operations":
            assert json.loads(request.content) == {"request": _ref(self.payload, "res-request")}
            identity = (request.headers["Idempotency-Key"], request.content)
            if self.operation is None:
                self.operation = identity
                self.generation_count += 1
            else:
                assert self.operation == identity
            return httpx.Response(202, json=self.detail())
        if method == "GET" and path == "/v2/model-operations/op-one/result":
            return httpx.Response(200, content=self.result, headers={"Content-Type": "application/json"})
        if method == "GET" and path == "/v2/model-operations/op-one":
            return httpx.Response(200, json=self.detail())
        if method == "POST" and path == "/v2/model-operations/op-one/ack":
            assert json.loads(request.content) == {"sha256": _ref(self.result)["sha256"]}
            return httpx.Response(200, json={**self.detail(), "response": {
                "state": "acknowledged", "ref": _ref(self.result), "releasedAt": "2026-09-06T00:01:00Z",
            }})
        raise AssertionError(f"Unexpected transport: {method} {path}")

    def detail(self):
        return {"id": "op-one", "state": "succeeded", "dispatch": {"state": "response_received"},
                "response": {"state": "ready", "ref": _ref(self.result)}}


def test_model_roundtrip_preserves_content_and_ack_is_separate(gateway_factory, caplog):
    wire = ModelWire()
    gateway = gateway_factory(wire)
    request = _request()
    original = deepcopy(request)
    ref = gateway.upload_model_request(request, idempotency_key="logical-invocation")
    assert wire.payload == _bytes(original)
    assert request == original
    assert ref == _ref(wire.payload, "res-request")
    metadata = json.loads(wire.create[1])
    assert metadata["purpose"] == "model"
    assert ref["sha256"].removeprefix("sha256:") in metadata["name"]
    assert "messages" not in metadata
    assert wire.generation_count == 0
    detail = gateway.create_model_operation(ref, idempotency_key="logical-invocation")
    assert wire.operation[0] == "logical-invocation"
    assert gateway.get_model_operation(detail["id"]) == detail
    result = gateway.get_model_result(detail["id"], expected_ref=detail["response"]["ref"])
    assert result == json.loads(wire.result)
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    acknowledged = gateway.acknowledge_model_result(detail["id"], detail["response"]["ref"]["sha256"])
    assert acknowledged["response"] == {"state": "acknowledged", "ref": _ref(wire.result), "releasedAt": "2026-09-06T00:01:00Z"}
    assert wire.generation_count == 1
    assert "test-control-token" not in caplog.text and "TESTONLY" not in caplog.text


@pytest.mark.parametrize("stage", [
    ("POST", "/v2/uploads"), ("PUT", "/v2/uploads/upl-one/bytes"),
    ("POST", "/v2/uploads/upl-one/finalize"), ("POST", "/v2/model-operations"),
])
def test_lost_reply_rejoins_same_upload_and_operation_only_on_explicit_retry(gateway_factory, stage):
    wire = ModelWire(lose=stage)
    gateway = gateway_factory(wire)

    def submit():
        ref = gateway.upload_model_request(_request(), idempotency_key="invocation")
        return gateway.create_model_operation(ref, idempotency_key="invocation")

    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        submit()
    assert raised.value.code == "daemon_unreachable"
    assert [(call.method, call.url.path) for call in wire.calls].count(stage) == 1
    assert submit()["id"] == "op-one"
    assert wire.upload_count == wire.generation_count == 1
    assert sum(call.method == "PUT" for call in wire.calls) == 1


@pytest.mark.parametrize("open_upload", [True, False])
def test_same_length_changed_request_conflicts_before_reusing_upload(gateway_factory, open_upload):
    wire = ModelWire(lose=("POST", "/v2/uploads") if open_upload else None)
    gateway = gateway_factory(wire)
    if open_upload:
        with pytest.raises(cx.ClaudexorUnavailable):
            gateway.upload_model_request({"text": "aaaa"}, idempotency_key="same")
    else:
        gateway.upload_model_request({"text": "aaaa"}, idempotency_key="same")
    previous_puts = sum(call.method == "PUT" for call in wire.calls)
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.upload_model_request({"text": "bbbb"}, idempotency_key="same")
    assert raised.value.code == "idempotency_conflict"
    assert sum(call.method == "PUT" for call in wire.calls) == previous_puts
    assert wire.upload_count == 1 and wire.generation_count == 0


@pytest.mark.parametrize("status,code", [(404, "http_404"), (404, "route_not_found"), (500, "upload_not_found")])
def test_only_typed_missing_upload_can_resume_finalization(gateway_factory, status, code):
    wire = ModelWire()

    def handler(request):
        if request.method == "GET":
            return httpx.Response(status, json={"code": code, "message": "not a finalized-upload receipt"})
        return wire(request)

    gateway = gateway_factory(handler)
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.upload_model_request(_request(), idempotency_key="one")
    assert raised.value.code == code
    assert len(wire.calls) == 1  # no PUT or finalize after the failed status read


@pytest.mark.parametrize("state", ["uploading", "cancelled", "unknown"])
def test_unfinished_upload_is_not_overwritten_or_finalized(gateway_factory, state):
    wire = ModelWire(lose=("POST", "/v2/uploads"))
    gateway = gateway_factory(wire)
    with pytest.raises(cx.ClaudexorUnavailable):
        gateway.upload_model_request(_request(), idempotency_key="one")
    wire.upload["state"] = state
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.upload_model_request(_request(), idempotency_key="one")
    assert raised.value.code == "model_upload_unavailable"
    assert not any(call.method == "PUT" or call.url.path.endswith("/finalize") for call in wire.calls)


@pytest.mark.parametrize("change,code", [
    ({"purpose": None}, "resource_purpose_mismatch"),
    ({"sha256": "sha256:" + "0" * 64}, "model_payload_integrity_error"),
    ({"sizeBytes": 0}, "model_payload_integrity_error"),
    ({"resourceId": ""}, "model_payload_ref_invalid"),
])
def test_finalized_upload_must_bind_the_model_bytes(gateway_factory, change, code):
    wire = ModelWire()

    def handler(request):
        response = wire(request)
        if request.url.path.endswith("/finalize"):
            return httpx.Response(201, json={**response.json(), **change})
        return response

    gateway = gateway_factory(handler)
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.upload_model_request(_request(), idempotency_key="one")
    assert raised.value.code == code
    assert wire.generation_count == 0


def test_lost_result_reply_keeps_same_operation_without_ack(gateway_factory):
    wire = ModelWire(lose=("GET", "/v2/model-operations/op-one/result"))
    gateway = gateway_factory(wire)
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.get_model_result("op-one", expected_ref=_ref(wire.result))
    assert raised.value.code == "daemon_unreachable"
    assert len(wire.calls) == 1
    assert gateway.get_model_result("op-one", expected_ref=_ref(wire.result)) == json.loads(wire.result)
    assert len(wire.calls) == 2 and all(call.method == "GET" for call in wire.calls)


@pytest.mark.parametrize("body", [None, [], {"bad": float("nan")}, {"bad": float("inf")}, {"bad": "\ud800"}])
def test_invalid_request_fails_before_any_http(gateway_factory, body):
    calls = []
    gateway = gateway_factory(lambda request: calls.append(request))
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.upload_model_request(body, idempotency_key="one")
    assert raised.value.code == "invalid_model_payload"
    assert calls == []


@pytest.mark.parametrize("key", [None, "", "   ", "x" * 257])
def test_model_invocation_requires_stable_key(gateway_factory, key):
    calls = []
    gateway = gateway_factory(lambda request: calls.append(request))
    for operation in (lambda: gateway.upload_model_request({}, idempotency_key=key),
                      lambda: gateway.create_model_operation(_ref(b"{}"), idempotency_key=key)):
        with pytest.raises(cx.ClaudexorUnavailable) as raised:
            operation()
        assert raised.value.code == "invalid_idempotency_key"
    assert calls == []


@pytest.mark.parametrize("data", [b'{"n":NaN}', b'{"n":Infinity}', b'{"n":-Infinity}', b'\xff', b'{"unterminated":', b'[]'])
def test_result_is_strict_utf8_json_object_after_integrity_verification(gateway_factory, data):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=data)

    gateway = gateway_factory(handler)
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.get_model_result("op-one", expected_ref=_ref(data))
    assert raised.value.code == "malformed_response"
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/v2/model-operations/op-one/result")]


@pytest.mark.parametrize("change", [{"sizeBytes": 999}, {"sha256": "sha256:" + "0" * 64}])
def test_result_integrity_precedes_decode(gateway_factory, change):
    data = b"\xffinvalid-json"
    gateway = gateway_factory(lambda request: httpx.Response(200, content=data))
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.get_model_result("op-one", expected_ref={**_ref(data), **change})
    assert raised.value.code == "model_payload_integrity_error"


def test_result_exceeds_agent_artifact_cap_without_redaction_or_ack(gateway_factory):
    content = "TESTONLY-sk-" + "x" * (4 * 1024 * 1024) + "\r\n末尾 🐍"
    result = {"message": {"content": content, "nativeContinuation": {"payload": "opaque+/=="}}}
    data = _bytes(result)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=data)

    gateway = gateway_factory(handler)
    assert gateway.get_model_result("op-one", expected_ref=_ref(data), timeout_sec=0.25) == result
    assert len(calls) == 1 and calls[0].method == "GET"
    assert calls[0].extensions["timeout"] == {"connect": 0.25, "read": 0.25, "write": 0.25, "pool": 0.25}


@pytest.mark.parametrize("ref", [{}, {**_ref(b"{}"), "sizeBytes": True}, {**_ref(b"{}"), "extra": "wrong"},
                                  {**_ref(b"{}"), "sha256": "missing-prefix"}, {**_ref(b"{}"), "sizeBytes": -1}])
def test_bad_reference_never_dispatches_or_reads(gateway_factory, ref):
    calls = []
    gateway = gateway_factory(lambda request: calls.append(request))
    for operation in (lambda: gateway.get_model_result("op-one", expected_ref=ref),
                      lambda: gateway.create_model_operation(ref, idempotency_key="one")):
        with pytest.raises(cx.ClaudexorUnavailable) as raised:
            operation()
        assert raised.value.code == "model_payload_ref_invalid"
    assert calls == []


@pytest.mark.parametrize("state", ["acknowledged", "expired"])
def test_released_custody_keeps_full_reference_and_result_refusal(gateway_factory, state):
    detail = {"id": "op-one", "state": "succeeded", "response": {
        "state": state, "ref": _ref(b"{}"), "releasedAt": "2026-09-06T00:00:00Z",
    }}
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/result"):
            return httpx.Response(410, json={"code": "model_result_released", "message": "result released"})
        return httpx.Response(200, json=detail)

    gateway = gateway_factory(handler)
    assert gateway.get_model_operation("op-one") == detail
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.get_model_result("op-one", expected_ref=detail["response"]["ref"])
    assert raised.value.code == "model_result_released"
    assert all(call.method == "GET" for call in calls)


def test_catalog_preserves_exact_profile_provenance_and_engine_auto(gateway_factory):
    sources = {"sources": [{"id": "codex+raw", "label": "Codex", "credentialHarness": "codex"}]}
    catalog = {"source": "codex+raw", "credentialProfileId": "team user+1", "accountFingerprint": None,
               "observedAt": "2026-09-06T00:00:00Z", "provenance": "route-metadata", "models": [{
                   "id": "exact-model", "contextWindow": 272000, "maxContextWindow": 872000,
                   "maxOutputTokens": None, "supportedOptions": [],
               }]}
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=sources if request.url.path == "/v2/model-sources" else catalog)

    gateway = gateway_factory(handler)
    assert gateway.list_model_sources() == sources
    assert gateway.list_source_models("codex+raw", "team user+1") == catalog
    assert gateway.list_source_models("codex+raw") == catalog
    assert calls[1].url.raw_path == b"/v2/model-sources/codex%2Braw/models?credentialProfileId=team+user%2B1"
    assert calls[2].url.query == b""
    assert gateway.list_source_models("codex+raw", requested_model="exact/model+1") == catalog
    assert calls[3].url.query == b"requestedModel=exact%2Fmodel%2B1"
    assert gateway.list_source_models("codex+raw", "team user+1", requested_model="exact-model") == catalog
    assert calls[4].url.query == b"credentialProfileId=team+user%2B1&requestedModel=exact-model"
    assert gateway.list_source_models("codex+raw", requested_model="future/model", include_admission=True) == catalog
    assert calls[5].url.query == b"requestedModel=future%2Fmodel&includeAdmission=true"


@pytest.mark.parametrize("reason", ["", "user_cancelled"])
def test_cancel_is_one_control_request_and_does_not_claim_settlement(gateway_factory, reason):
    detail = {"id": "op-one", "state": "running", "dispatch": {"state": "started"}, "response": {"state": "absent"}}
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=detail)

    gateway = gateway_factory(handler)
    assert gateway.cancel_model_operation("op-one", reason_code=reason) == detail
    assert len(calls) == 1
    assert calls[0].url.path == "/v2/model-operations/op-one/control"
    assert json.loads(calls[0].content) == {"action": "cancel", **({"reasonCode": reason} if reason else {})}


@pytest.mark.parametrize("body", [{}, {"id": "other"}, []])
def test_operation_identity_mismatch_is_not_an_accepted_response(gateway_factory, body):
    gateway = gateway_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(cx.ClaudexorUnavailable) as raised:
        gateway.get_model_operation("op-one", timeout_sec=0.3)
    assert raised.value.code == "malformed_response"


@pytest.mark.parametrize("capture_failure", [False, True])
@pytest.mark.parametrize("capture_effort", [False, True])
def test_evidence_opt_ins_preserve_create_body(gateway_factory, capture_failure, capture_effort):
    wire = ModelWire()
    gateway = gateway_factory(wire)
    ref = gateway.upload_model_request(_request(), idempotency_key="evidence")
    gateway.create_model_operation(ref, idempotency_key="evidence",
        capture_failure_evidence=capture_failure, capture_effort_evidence=capture_effort)
    request = wire.calls[-1]
    expected = {}
    if capture_failure:
        expected["captureFailureEvidence"] = "true"
    if capture_effort:
        expected["captureEffortEvidence"] = "true"
    assert dict(request.url.params) == expected
    assert json.loads(request.content) == {"request": ref}
    assert request.headers["Idempotency-Key"] == "evidence"
    assert wire.payload == _bytes(_request())
    assert wire.generation_count == 1


@pytest.mark.parametrize("parameter_name", ["captureFailureEvidence", "captureEffortEvidence"])
def test_query_negotiation_uses_exact_wire_descriptor(parameter_name):
    def supported(operations):
        return cx.operation_query_supported(operations, method="POST", path="/v2/model-operations",
                                            name=parameter_name, value="true")

    parameter = {"name": parameter_name, "location": "query", "enum": ["true", "false"]}
    operation = {"method": "POST", "path": "/v2/model-operations", "parameters": [parameter]}
    assert supported([operation])
    assert cx.model_failure_evidence_supported([operation]) is (parameter_name == "captureFailureEvidence")
    assert not supported([])
    for change in ({"method": "GET"}, {"path": "/v2/model-operations/:id"}, {"parameters": []}):
        assert not supported([{**operation, **change}])
    for change in ({"name": "other"}, {"location": "header"}, {"enum": ["false"]}, {"enum": "true"}):
        assert not supported([{**operation, "parameters": [{**parameter, **change}]}])
    accounts = {"method": "GET", "path": "/v2/model-sources", "parameters": [
        {"name": "view", "location": "query", "enum": ["accounts"]}]}
    assert cx.account_catalog_supported([accounts], accounts["path"])
    assert not cx.account_catalog_supported([operation], accounts["path"])
    assert not supported([accounts])
