"""Complete failed model evidence and known-terminal rejection share existing custody."""

import asyncio
import base64
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor as transport, usage_accounting as ua
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.request_wire_recovery import plan_next_wire_retry
from ouroboros.tools import vision_process
from ouroboros.transport_custody import ProviderNotDispatched
from tests.test_llm_claudexor import Gateway, MODEL, ROUTE, ledger, result, retained, setup as setup


CAPTURE_OPERATION = {"method": "POST", "path": "/v2/model-operations", "parameters": [
    {"name": "captureFailureEvidence", "location": "query", "enum": ["true", "false"]}]}
REJECTION = {"code": "response_rejected", "message": "The terminal response could not form a model message.",
             "retryable": False, "context": {"stage": "message", "requestId": "request-one"}}


def failure_evidence(body=b'\xffprivate-wire-marker\r\ndata: invalid JSON\n\n'):
    return {"bodyBase64": base64.b64encode(body).decode("ascii"), "receivedBytes": len(body),
            "bodyComplete": False, "stage": "message", "causeCycle": False,
            "errors": [{"name": "SyntaxError", "message": "private-error-marker", "stack": "private-stack-marker",
                        "code": None}]}


def call(client, asynchronous, **kwargs):
    value = (client.chat_async if asynchronous else client.chat)([], MODEL, **kwargs)
    return asyncio.run(value) if asynchronous else value


def capture_catalog(failure, effort):
    return [{**CAPTURE_OPERATION, "parameters": [
        {"name": name, "location": "query", "enum": ["true", "false"]}
        for name, enabled in (("captureFailureEvidence", failure), ("captureEffortEvidence", effort))
        if enabled]}]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure,effort", [(False, False), (True, False), (False, True), (True, True)])
def test_capture_is_negotiated_once_without_changing_provider_payload(setup, asynchronous, failure, effort):
    root, gateway, client = setup
    gateway.operation_catalog = capture_catalog(failure, effort)
    _, usage = call(client, asynchronous, model_role="main")
    assert gateway.catalog_reads == 1
    expected = {key: True for key, enabled in (
        ("capture_failure_evidence", failure), ("capture_effort_evidence", effort)) if enabled}
    assert gateway.capture_requests == [expected]
    payload = gateway.uploads[0][0]
    for key in ("captureFailureEvidence", "capture_failure_evidence", "captureEffortEvidence", "capture_effort_evidence"):
        assert key not in json.dumps(payload)
    assert usage.get("effort_resolution") is None  # Capability alone supplies no evidence.
    assert retained(root, "request") == payload
    manifests = list((root / "observability/calls/task-one").glob("*_model_request.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["capture_failure_evidence"] is failure
    assert manifest["capture_effort_evidence"] is effort
    assert manifest["operation_id"] == "op-0"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure,effort", [(False, False), (True, False), (False, True), (True, True)])
def test_lost_create_and_gateway_replacement_reuse_frozen_capture(setup, monkeypatch, asynchronous, failure, effort):
    root, gateway, client = setup
    gateway.operation_catalog = capture_catalog(failure, effort)
    gateway.lose_create = True
    replacement = Gateway()
    replacement.operation_catalog = capture_catalog(not failure, not effort)
    for name in ("accepted_operations", "creates", "capture_requests"):
        setattr(replacement, name, getattr(gateway, name))
    monkeypatch.setattr(transport, "read_owned_gateway", lambda: replacement)
    monkeypatch.setattr(transport, "current_model_wait", lambda: SimpleNamespace(
        tool_context=SimpleNamespace(task_id="task-one", is_direct_chat=False), control_reason=lambda: None))
    monkeypatch.setattr(transport.config, "NETWORK_WAIT_BACKOFF_START_SEC", 0.001)
    monkeypatch.setattr(transport.config, "NETWORK_WAIT_BACKOFF_MAX_SEC", 0.001)
    manifests = []

    def observe(receipt):
        manifests.append(json.loads(Path(receipt["request_manifest_ref"]["path"]).read_text()))

    answer, usage = call(client, asynchronous, model_operation_observer=observe)
    assert answer == result()["message"]
    assert gateway.catalog_reads == 1 and replacement.catalog_reads == 0
    expected = {key: True for key, enabled in (
        ("capture_failure_evidence", failure), ("capture_effort_evidence", effort)) if enabled}
    assert gateway.capture_requests == [expected] * 2
    assert len(manifests) == 3  # Before lost create, before rejoin, then accepted custody.
    assert all(item["capture_failure_evidence"] is failure and
               item["capture_effort_evidence"] is effort for item in manifests)
    assert len(gateway.uploads) == 1 and not replacement.uploads
    assert retained(root, "request") == gateway.uploads[0][0]
    assert len(gateway.creates) == 2 and len(set(gateway.creates)) == 1
    assert len(gateway.accepted_operations) == len(usage["ledger_attempt_ids"]) == 1
    assert gateway.closed == replacement.closed == 1
    # One current row per attempt; revision 3 = reserved, dispatched, settled.
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_catalog_read_failure_does_not_guess_unsupported_or_dispatch(setup, monkeypatch, asynchronous):
    root, gateway, client = setup

    def failed_catalog():
        raise ClaudexorUnavailable("daemon_unreachable", "Metadata connection lost")

    monkeypatch.setattr(gateway, "operations", failed_catalog)
    with pytest.raises(transport.ClaudexorModelError) as caught:
        call(client, asynchronous)
    assert caught.value.code == "daemon_unreachable"
    assert caught.value.physical_attempt_capture.state == "released"
    assert not gateway.uploads and not gateway.creates and gateway.closed == 1
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("released", 2)]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["completed", "incomplete"])
@pytest.mark.parametrize("has_problem", [False, True])
def test_known_terminal_null_message_settles_then_rejects_without_private_projection(
    setup, caplog, asynchronous, outcome, has_problem,
):
    root, gateway, client = setup
    gateway.operation_catalog = [deepcopy(CAPTURE_OPERATION)]
    evidence = failure_evidence()
    original_problem = deepcopy(REJECTION) if has_problem else None
    gateway.results = [{**result(outcome=outcome, problem=original_problem),
                        "message": None, "failureEvidence": evidence}]
    acknowledge = gateway.acknowledge_model_result

    def retained_first(*args):
        assert retained(root) == gateway.results[0]
        assert ledger(root)[-1]["state"] == "settled"
        return acknowledge(*args)

    gateway.acknowledge_model_result = retained_first
    with pytest.raises(transport.ClaudexorModelError) as caught:
        call(client, asynchronous, model_role="vision")
    error = caught.value
    assert type(error) is transport.ClaudexorModelError
    assert not isinstance(error, ProviderNotDispatched)
    assert error.code == "response_rejected" and error.stream_rejected and error.stream_incomplete
    assert error.operation_id == "op-0" and error.model_role == "vision" and error.route == ROUTE
    if has_problem:
        assert error.problem == original_problem
    assert error.physical_attempt_capture.state == "settled"
    assert error.usage["prompt_tokens"] == 20 and error.usage["completion_tokens"] == 7
    assert error.usage["cost"] is None and error.usage["cost_final"] is False
    assert error.usage["claudexor"]["outcome"] == outcome
    assert error.usage["claudexor"]["result_custody"]["state"] == "acknowledged"
    classified = classify_llm_exception(error)
    assert classified.kind == "provider_error" and not classified.retry_same_request
    assert plan_next_wire_retry({}, error=error) is None
    # One current row per attempt; revision 3 = reserved, dispatched, settled.
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]
    assert len(gateway.creates) == len(gateway.acks) == 1
    assert not hasattr(error, "model_result")
    public = json.dumps({"usage": error.usage, "ledger": ledger(root), "detail": gateway.detail(0)}) + caplog.text
    public += "".join(path.read_text() for path in (root / "logs").glob("*.jsonl"))
    for marker in (evidence["bodyBase64"], "private-error-marker", "private-stack-marker"):
        assert marker not in public
    assert retained(root)["failureEvidence"] == evidence


@pytest.mark.parametrize("effort", [None, {
    "requested": "ultra", "sent": {"options.reasoningEffort": "ultra"},
    "sent_state": "explicit", "sent_source": "host_candidate",
    "reported": None, "report_source": None,
}])
def test_response_rejection_survives_existing_vision_ipc_reconstruction(effort):
    capture = ua.PhysicalAttemptCapture("attempt-one", MODEL, "claudexor", "settled", "opaque", effort=effort)
    receipt = {"receipt_id": "receipt-one", "custody": None, "capture": asdict(capture),
               "kind": "model", "text": "", "usage": {"prompt_tokens": 20}, "ledger_attempt_ids": ["attempt-one"],
               "error": "", "problem": deepcopy(REJECTION), "operation_id": "operation-one", "model_role": "vision",
               "route": deepcopy(ROUTE), "unknown": False, "control_reason": "", "model_result": None}
    with pytest.raises(transport.ClaudexorModelError) as caught:
        vision_process._decode_terminal(json.loads(json.dumps(receipt)), "receipt-one")
    error = caught.value
    assert error.code == "response_rejected" and error.stream_rejected and error.stream_incomplete
    assert error.problem == REJECTION and error.operation_id == "operation-one" and error.route == ROUTE
    assert error.physical_attempt_capture.state == "settled" and error.usage == receipt["usage"]
    assert error.physical_attempt_capture.effort == effort
    assert classify_llm_exception(error).kind == "provider_error"
    assert not isinstance(error, ProviderNotDispatched)


def test_unknown_diagnostic_display_keeps_only_compact_response_context():
    error = transport.ClaudexorModelError({"code": "transport_unknown", "message": "Stream interrupted.",
        "context": {"stage": "read", "errorCode": "UND_ERR_SOCKET", "requestId": "request-one",
                    "vendorCode": "not-a-terminal-provider-fact", "stack": "private-stack-marker"}}, unknown=True)
    assert error.display_message == (
        "stage=read, cause=UND_ERR_SOCKET; model_outcome_unknown: Stream interrupted.")
    assert "private-stack-marker" not in error.display_message
    assert "request-one" not in error.display_message
    assert error.code == "model_outcome_unknown" and not error.retryable


def test_local_rejection_does_not_poison_the_next_account_preference(setup):
    _, gateway, client = setup
    gateway.results = [{**result(problem=deepcopy(REJECTION)), "message": None}, result()]
    gateway.dispatch = ["response_received"] * 2
    messages = [result()["message"]]
    with pytest.raises(transport.ClaudexorModelError):
        client.chat(messages, MODEL, cache_affinity="same-account-after-local-rejection")
    client.chat(messages, MODEL, cache_affinity="same-account-after-local-rejection")
    assert [payload["account"] for payload, _ in gateway.uploads] == [
        {"mode": "auto", "preferredProfileId": "account-a"}] * 2


def test_large_unknown_result_retains_exact_private_bytes_and_existing_pending_ack(setup):
    root, gateway, client = setup
    gateway.operation_catalog = [deepcopy(CAPTURE_OPERATION)]
    body = b'\xff\xc3\x28' + b"x" * (4 * 1024 * 1024) + b"unparsed-suffix\r\n"
    evidence = failure_evidence(body)
    gateway.results = [{**result(outcome="unknown"), "message": None, "failureEvidence": evidence}]
    gateway.dispatch = ["unknown"]
    with pytest.raises(transport.ClaudexorModelError) as caught:
        client.chat([], MODEL)
    assert caught.value.code == "model_outcome_unknown"
    assert not getattr(caught.value, "stream_rejected", False)
    stored = retained(root)["failureEvidence"]
    assert base64.b64decode(stored["bodyBase64"]) == body
    assert stored["receivedBytes"] == len(body)
    assert ledger(root)[-1]["state"] == "unresolved"
    assert len(gateway.creates) == 1 and not gateway.acks
