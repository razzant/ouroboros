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
from ouroboros.llm_messages import _MessageShapingMixin
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.loop_messages import _latch_final_answer_marker, _visible_round_text
from ouroboros.model_wait import model_wait_reason, task_model_wait_scope
from ouroboros.openai_responses import input_items
from ouroboros.request_wire_recovery import plan_next_wire_retry
from ouroboros.task_results import write_task_result
from ouroboros.tools import vision_process
from ouroboros.transport_custody import ProviderNotDispatched
from tests._model_replies import MIXED_REFUSAL_PARTS, MIXED_REFUSAL_TEXT
from tests._scripted_model_loop import walk_loop
from tests.test_llm_claudexor import Gateway, MODEL, ROUTE, ledger, result, retained, setup as setup


TYPED_REFUSALS = {"method": "POST", "path": "/v2/model-operations", "parameters": [
    {"name": "typedRefusals", "location": "query", "enum": ["true"]}]}
PRIVATE_DIAGNOSTIC = "synthetic-private-diagnostic"

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


@pytest.mark.parametrize(("failure", "effort", "asynchronous"), [
    (False, False, False),
    (True, False, True),
    (False, True, False),
    (True, True, True),
])
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


@pytest.mark.parametrize(("failure", "effort", "asynchronous"), [
    (False, False, False),
    (True, False, True),
    (False, True, False),
    (True, True, True),
])
def test_gateway_replacement_reuses_frozen_capture_after_create(setup, monkeypatch, asynchronous, failure, effort):
    root, gateway, client = setup
    gateway.operation_catalog = capture_catalog(failure, effort)
    gateway.pending = gateway.read_error = True
    replacement = Gateway()
    replacement.operation_catalog = capture_catalog(not failure, not effort)
    for name in ("accepted_operations", "creates", "capture_requests"):
        setattr(replacement, name, getattr(gateway, name))
    monkeypatch.setattr(transport, "read_owned_gateway", lambda: replacement)
    monkeypatch.setattr(transport, "current_model_wait", lambda: SimpleNamespace(
        tool_context=SimpleNamespace(task_id="task-one", is_direct_chat=False), control_reason=lambda: None))
    monkeypatch.setattr(transport.config, "NETWORK_WAIT_BACKOFF_START_SEC", 0.001)
    monkeypatch.setattr(transport.config, "NETWORK_WAIT_BACKOFF_MAX_SEC", 0.001)
    manifests, request_writes = [], []
    persist_call = transport.persist_call

    def persist(*args, **kwargs):
        receipt = persist_call(*args, **kwargs)
        if kwargs["call_type"] == "llm_claudexor_request":
            request_writes.append(json.loads(
                Path(receipt["manifest_ref"]["path"]).read_text(encoding="utf-8")))
        return receipt

    monkeypatch.setattr(transport, "persist_call", persist)

    def observe(receipt):
        manifests.append(json.loads(
            Path(receipt["request_manifest_ref"]["path"]).read_text(encoding="utf-8")))

    answer, usage = call(client, asynchronous, model_operation_observer=observe)
    assert answer == result()["message"]
    assert gateway.catalog_reads == 1 and replacement.catalog_reads == 0
    expected = {key: True for key, enabled in (
        ("capture_failure_evidence", failure), ("capture_effort_evidence", effort)) if enabled}
    assert gateway.capture_requests == [expected]
    # Phase observations may repeat a checkpoint, but cannot rewrite or rewind custody.
    versions = [item for index, item in enumerate(manifests)
                if index == 0 or item != manifests[index - 1]]
    assert [item.get("operation_id") for item in versions] == [None, "op-0"]
    assert [item.get("operation_id") for item in request_writes] == [None]  # acceptance merges receipt metadata only
    assert len({item["invocation_id"] for item in manifests}) == 1
    assert len({item["full_payload_ref"]["sha256"] for item in manifests}) == 1
    assert all(item["capture_failure_evidence"] is failure and
               item["capture_effort_evidence"] is effort for item in manifests)
    assert len(gateway.uploads) == 1 and not replacement.uploads
    assert retained(root, "request") == gateway.uploads[0][0]
    assert len(gateway.creates) == 1
    assert len(gateway.accepted_operations) == len(usage["ledger_attempt_ids"]) == 1
    assert gateway.closed == replacement.closed == 1
    # One current row per attempt; revision 3 = reserved, dispatched, settled.
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_catalog_read_failure_does_not_guess_unsupported_or_dispatch(setup, monkeypatch, asynchronous):
    root, gateway, client = setup

    def failed_catalog(**_bounds):
        raise ClaudexorUnavailable("daemon_unreachable", "Metadata connection lost")

    monkeypatch.setattr(gateway, "operations", failed_catalog)
    # The refusal is retried until the call's deadline and then reported as itself. The deadline runs on a
    # transport clock that advances only while the transport waits between reads: on the wall clock a loaded
    # runner can spend any deadline on gateway setup before the first read, which truthfully ends as
    # daemon_not_answering, a different case (#2101).
    clock = [0.0]
    monkeypatch.setattr(transport, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    with pytest.raises(transport.ClaudexorModelError) as caught:
        call(client, asynchronous, timeout=0.05)
    assert caught.value.code == "daemon_unreachable"
    assert caught.value.physical_attempt_capture.state == "released"
    assert not gateway.uploads and not gateway.creates and gateway.closed >= 1
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("released", 2)]


@pytest.mark.parametrize(("has_problem", "outcome", "asynchronous"), [
    (False, "completed", False),
    (True, "incomplete", True),
])
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("advertised", [False, True])
@pytest.mark.parametrize("accepts", [False, True])
def test_create_rejoin_and_completed_history(setup, asynchronous, advertised, accepts):
    root, gateway, client = setup
    gateway.operation_catalog = [TYPED_REFUSALS] if advertised else []
    opted = accepts and advertised
    gateway.lose_create = True
    value = result()
    value["message"].pop("nativeContinuation")
    value["message"]["content"] = deepcopy(MIXED_REFUSAL_PARTS) if opted else "legacy text"
    gateway.results = [value]
    reply, _ = call(client, asynchronous, accepts_typed_refusals=accepts)
    assert reply == value["message"] == retained(root)["message"]
    assert gateway.catalog_reads == len(set(gateway.creates)) == len(gateway.acks) == 1
    assert len(gateway.uploaded_bytes) == len(gateway.accepted_operations) == 1
    assert gateway.capture_requests == ([{"typed_refusals": True}] if opted else [{}]) * 2
    if opted:
        projected, = _MessageShapingMixin._normalize_system_message_placement([reply])
        assert projected["tool_calls"] == reply["tool_calls"]
        assert projected["content"] == [{"type": "text", "text": part[part["type"]]} for part in MIXED_REFUSAL_PARTS]
        assert input_items({}, [reply])[0] == {"role": "assistant", "content": MIXED_REFUSAL_TEXT}
        assert reply["content"] == MIXED_REFUSAL_PARTS
        assert _visible_round_text(MIXED_REFUSAL_PARTS) == MIXED_REFUSAL_TEXT
        trace = {}
        _latch_final_answer_marker(trace, MIXED_REFUSAL_PARTS)
        assert "best_valid_final_answer" not in trace


def test_negotiated_ordinary_text_stays_a_string(setup):
    root, gateway, client = setup
    gateway.operation_catalog = [TYPED_REFUSALS]
    reply, _ = call(client, False, accepts_typed_refusals=True)
    assert gateway.capture_requests == [{"typed_refusals": True}]
    assert isinstance(reply["content"], str) and reply == retained(root)["message"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("state", "outcome", "mixed", "error_code"), [
    ("succeeded", "completed", True, None),
    ("failed", "completed", True, "unexpected_refusal_content"),
    ("cancelled", "incomplete", True, "unexpected_refusal_content"),
    ("interrupted", "completed", True, "unexpected_refusal_content"),
    ("failed", "completed", False, "response_rejected"),
    ("failed", "incomplete", False, "response_rejected"),
    ("cancelled", "completed", False, "model_operation_cancelled"),
    ("interrupted", "incomplete", False, "response_rejected"),
    ("succeeded", "failed", False, "model_operation_failed"),
])
def test_warm_and_retained_receipt_agree_on_message_delivery(setup, monkeypatch, state, outcome, mixed, error_code, asynchronous):
    root, gateway, client = setup
    record = result(outcome=outcome, cash=0.17, knowledge="exact")
    record["message"] = {"role": "assistant", "content": deepcopy(MIXED_REFUSAL_PARTS) if mixed else "Sample note.",
                         "tool_calls": [{"id": "sample", "type": "function", "function": {"name": "label_sample", "arguments": "{}"}}]}
    gateway.operation_catalog, gateway.results = [TYPED_REFUSALS], [record]
    detail = gateway.detail
    monkeypatch.setattr(gateway, "detail", lambda index: {**detail(index), "state": state})

    def read_again():
        return transport.recover_chat_reply(root, "task-one", {
            "physical_attempt_id": gateway.creates[0], "operation_id": "op-0", "model": MODEL, "model_role": "main"}, None)

    def warm():
        return call(client, asynchronous, model_role="main", accepts_typed_refusals=True)

    for read in (warm, read_again):
        if error_code:
            with pytest.raises(transport.ClaudexorModelError) as stopped:
                read()
            assert stopped.value.code == error_code and not hasattr(stopped.value, "model_result")
            usage = stopped.value.usage
        else:
            reply, usage, *_ = read()
            assert reply == record["message"]
        assert (usage["claudexor"]["operation_state"], usage["claudexor"]["outcome"]) == (state, outcome)
        assert (usage["cost"], usage["prompt_tokens"], usage["completion_tokens"]) == (0.17, 20, 7)
    row, = ledger(root)
    assert row["state"] == "settled" and row["cost_usd"] == 0.17
    assert retained(root) == record and len(gateway.creates) == len(gateway.acks) == 1 and not gateway.reads


@pytest.mark.parametrize("asynchronous", [False, True])
def test_received_output_cut_is_not_a_one_shot_answer_and_survives_restore(setup, asynchronous):
    from ouroboros._usage_response import UnusableReplyError

    root, gateway, client = setup
    record = result(outcome="incomplete", cash=0.17, knowledge="exact", problem={
        "code": "provider_incomplete", "context": {"reason": "max_output_tokens"}})
    record["message"] = {"role": "assistant", "content": "A partial line"}
    gateway.results = [record]
    with pytest.raises(UnusableReplyError) as stopped:
        call(client, asynchronous, model_role="review")
    assert stopped.value.usability == "unfinished" and stopped.value.reply_message == record["message"]
    recovered, usage, _ = transport.recover_chat_reply(root, "task-one", {
        "physical_attempt_id": gateway.creates[0], "operation_id": "op-0", "model": MODEL, "model_role": "review"}, None)
    assert recovered == record["message"]
    for observed in (stopped.value.usage, usage):
        assert (observed["response_status"], observed["response_finish_reason"]) == ("incomplete", "length")
        assert observed["claudexor"]["operation_state"] == "interrupted" and observed["cost"] == 0.17
    assert retained(root) == record and len(ledger(root)) == len(gateway.creates) == len(gateway.acks) == 1
    assert not gateway.reads


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("typed", "advertised", "accepts"), [
    (False, False, False), (False, False, True), (False, True, False),  # legacy writer or no opt-in
    (True, True, True), (True, True, False), (True, False, True), (True, False, False),  # retain the explicit code
])
def test_settled_content_refusal_ends_request_with_safe_evidence(
        setup, monkeypatch, tmp_path, asynchronous, typed, advertised, accepts):
    root, gateway, client = setup
    gateway.operation_catalog = [TYPED_REFUSALS] if advertised else []
    problem = {"code": "provider_content_refused" if typed else "provider_failed", "retryable": False,
               "message": PRIVATE_DIAGNOSTIC, "context": {"vendorMessage": PRIVATE_DIAGNOSTIC}}
    if not typed:
        problem["context"]["vendorCode"] = "cyber_policy"
    failed = result(outcome="failed", cash=0.17, knowledge="exact", problem=problem)
    failed["message"] = None
    if advertised and accepts:
        failed["modelMismatch"] = {"requested": "fixture-model", "observed": "other-fixture-model"}
    gateway.results = [failed]

    def unexpected(*args, **kwargs):
        pytest.fail("A settled content refusal reached account retry or resource-wait classification")

    monkeypatch.setattr(transport.AccountRotation, "refused_call", unexpected)
    monkeypatch.setattr(transport.CapabilityRetry, "refused", unexpected)
    monkeypatch.setattr(transport.SubstitutionBudget, "admit", unexpected)
    write_task_result(root, "task-one", "running")
    with task_model_wait_scope(task={"id": "task-one"}, drive_root=root, event_queue=None,
                               worker_slot_held=False) as waiter:
        assert waiter.pre_dispatch_pause() is None
        with waiter.register_reprepare("main", lambda values: values, resource_reason=unexpected):
            with pytest.raises(transport.ClaudexorContentRefused) as raised:
                call(client, asynchronous, model_role="main", cache_affinity="case", accepts_typed_refusals=accepts)
    error = raised.value
    assert error.physical_attempt_capture.state == "settled"
    assert retained(root) == failed and ledger(root)[-1]["prompt_tokens"] == 20
    assert len(gateway.creates) == len(gateway.acks) == 1
    assert gateway.capture_requests == [{"typed_refusals": True} if advertised and accepts else {}]
    assert model_wait_reason(error) == ""
    assert PRIVATE_DIAGNOSTIC not in repr(error) + error.display_message + json.dumps(error.usage)
    assert classify_llm_exception(error).retry_same_request is False
    with pytest.raises(transport.ClaudexorContentRefused):
        transport.propagate_model_error(error)

    llm, text, usage, trace = walk_loop(tmp_path, monkeypatch, lambda: error)
    assert llm.calls == 1 and text == error.notice
    assert usage["reason_code"] == "provider_content_refused"
    assert usage["_last_llm_failure"]["operation_state"] == usage["_last_llm_failure"]["operation_outcome"] == "failed"
    assert (usage["cost"], usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"]) == (0.17, 20, 7, 12)
    assert usage["ledger_attempt_ids"] == [error.physical_attempt_capture.attempt_id]
    assert PRIVATE_DIAGNOSTIC not in json.dumps(trace) + (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_cold_reader_keeps_explicit_terminal_refusal_without_renegotiating(setup, monkeypatch):
    root, gateway, client = setup
    failed = result(outcome="failed", problem={"code": "provider_content_refused", "retryable": False,
                                               "message": PRIVATE_DIAGNOSTIC})
    failed["message"] = None
    gateway.results = [failed]
    with pytest.raises(transport.ClaudexorContentRefused):
        client.chat([], MODEL, model_role="main")
    monkeypatch.setattr(transport, "read_owned_gateway", lambda: pytest.fail("A retained result needs no discovery"))
    [row] = ledger(root)
    pending = {"physical_attempt_id": row["attempt_id"], "operation_id": "op-0", "model": MODEL, "model_role": "main"}
    with pytest.raises(transport.ClaudexorContentRefused) as again:
        transport.recover_chat_reply(root, "task-one", pending, None)
    assert again.value.problem == transport.ClaudexorContentRefused.terminal_problem()
    assert again.value.usage["claudexor"]["result_custody"]["state"] == "retained"
    assert gateway.capture_requests == [{}] and len(gateway.acks) == 1 and not gateway.reads
    assert ledger(root) == [row] and retained(root) == failed


@pytest.mark.parametrize(("code", "context", "source", "retryable"), [
    ("provider_failed", {"httpStatus": 403}, "codex", False),
    ("invalid_request", {"httpStatus": 403, "vendorCode": "other"}, "codex", False),
    ("provider_failed", {"vendorCode": "cyber_policy"}, "other", False),
    # The earlier producer shape is provider_failed, not retryable: the vendor code beside another fact proves nothing.
    ("invalid_request", {"vendorCode": "cyber_policy"}, "codex", False),
    ("provider_failed", {"vendorCode": "cyber_policy"}, "codex", True),
])
def test_similar_words_status_or_unbound_legacy_code_are_not_the_new_terminal(setup, code, context, source, retryable):
    _, gateway, client = setup
    gateway.results = [result(outcome="failed", route={"source": source}, problem={
        "code": code, "message": "request declined", "context": context, "retryable": retryable})]
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(client, False, accepts_typed_refusals=True)
    assert type(raised.value) is transport.ClaudexorModelError and raised.value.code == code


@pytest.mark.parametrize(("dispatch", "outcome", "retryable"), [
    ("response_received", "failed", True), ("response_received", "completed", False),
    ("unknown", "failed", False), ("not_started", "failed", False),
])
def test_inconsistent_explicit_code_cannot_resume_the_request(setup, monkeypatch, tmp_path, dispatch, outcome, retryable):
    root, gateway, client = setup
    failed = result(outcome=outcome, problem={"code": "provider_content_refused", "retryable": retryable,
                                              "message": PRIVATE_DIAGNOSTIC})
    gateway.results, gateway.dispatch = [failed], [dispatch]

    def unexpected(*args, **kwargs):
        pytest.fail("The inconsistent result cannot authorize another send decision")

    monkeypatch.setattr(transport.SubstitutionBudget, "admit", unexpected)
    with pytest.raises(transport.ClaudexorModelError) as raised:
        client.chat([], MODEL, model_role="main")
    error = raised.value
    assert not error.retryable and retained(root) == failed
    assert PRIVATE_DIAGNOSTIC not in error.display_message
    [row] = ledger(root)
    assert len(gateway.accepted_operations) == len(gateway.creates) == 1
    if dispatch == "response_received":
        assert type(error) is transport.ClaudexorRefusalError
        assert row["state"] == "settled" and len(gateway.acks) == 1
        llm, text, usage, _ = walk_loop(tmp_path, monkeypatch, lambda: error)
        assert llm.calls == 1 and text == error.notice and usage["reason_code"] == "unexpected_refusal_content"
    else:
        assert type(error) is transport.ClaudexorModelError and error.code == "model_outcome_unknown"
        assert error.same_operation_recoverable and error.route == {} and row["state"] == "unresolved" and not gateway.acks
        pending = {"physical_attempt_id": row["attempt_id"], "operation_id": "op-0", "model": MODEL, "model_role": "main"}
        monkeypatch.setattr(transport, "read_owned_gateway", lambda: pytest.fail("The retained body is already available"))
        assert transport.recover_chat_reply(root, "task-one", pending, None) is None
        assert ledger(root) == [row] and len(gateway.creates) == 1


def test_refusal_detail_without_body_cannot_release_the_attempt(setup, monkeypatch):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [result(outcome="failed", problem={
        "code": "provider_content_refused", "retryable": False, "message": PRIVATE_DIAGNOSTIC})], ["not_started"]
    detail = gateway.detail
    gateway.detail = lambda index: {**detail(index), "response": {"state": "absent"}}
    with pytest.raises(transport.ClaudexorModelError) as raised:
        client.chat([], MODEL, model_role="main")
    assert raised.value.code == "model_outcome_unknown" and raised.value.same_operation_recoverable
    [row] = ledger(root)
    assert row["state"] == "unresolved"
    pending = {"physical_attempt_id": row["attempt_id"], "operation_id": "op-0", "model": MODEL, "model_role": "main"}
    monkeypatch.setattr(transport, "read_owned_gateway", lambda: gateway)
    assert transport.recover_chat_reply(root, "task-one", pending, None) is None
    assert gateway.reads == ["op-0"] and not gateway.acks and len(gateway.creates) == 1
    assert ledger(root) == [row]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("advertised", "options"), [(True, {}), (False, {"accepts_typed_refusals": True})])
def test_unexpected_completed_refusal_retains_evidence_without_returning_a_verdict(
        setup, monkeypatch, asynchronous, advertised, options):
    root, gateway, client = setup
    gateway.operation_catalog = [TYPED_REFUSALS] if advertised else []
    reply = result()
    reply["message"]["content"] = [{"type": "refusal", "refusal": PRIVATE_DIAGNOSTIC}]
    gateway.results = [reply]

    def unexpected(*args, **kwargs):
        pytest.fail("Unexpected response content reached an alternate send decision")

    monkeypatch.setattr(transport.SubstitutionBudget, "admit", unexpected)
    monkeypatch.setattr(transport.AccountRotation, "refused_call", unexpected)
    with pytest.raises(transport.ClaudexorRefusalError) as raised:
        call(client, asynchronous, **options)
    error = raised.value
    assert type(error) is transport.ClaudexorRefusalError
    assert error.code == "unexpected_refusal_content" and error.usage["claudexor"]["outcome"] == "completed"
    assert error.physical_attempt_capture.state == "settled"
    assert PRIVATE_DIAGNOSTIC not in error.display_message + json.dumps(error.usage)
    assert gateway.capture_requests == [{}] and len(gateway.acks) == 1 and retained(root) == reply
    with pytest.raises(transport.ClaudexorRefusalError):
        transport.propagate_model_error(error)


def test_negotiated_malformed_refusal_retains_the_received_message(setup):
    root, gateway, client = setup
    gateway.operation_catalog = [TYPED_REFUSALS]
    reply = result()
    reply["message"]["content"] = [{"type": "text", "text": "Example."}, {"type": "refusal"}]
    gateway.results = [reply]
    with pytest.raises(transport.ClaudexorRefusalError) as raised:
        call(client, False, accepts_typed_refusals=True)
    assert raised.value.code == "unexpected_refusal_content"
    assert gateway.capture_requests == [{"typed_refusals": True}] and retained(root) == reply


@pytest.mark.parametrize("error_type,code", [
    (transport.ClaudexorContentRefused, "provider_content_refused"),
    (transport.ClaudexorRefusalError, "unexpected_refusal_content"),
])
def test_vision_child_preserves_only_a_settled_refusal(error_type, code):
    capture = asdict(ua.PhysicalAttemptCapture("fixture", "example", "claudexor", "settled", "opaque"))
    packet = {"kind": "content_refused", "receipt_id": "fixture", "capture": capture, "custody": None,
              "text": "", "usage": {}, "ledger_attempt_ids": ["fixture"], "error": "",
              "problem": {"code": code, "message": error_type.notice, "retryable": False}, "operation_id": "op",
              "model_role": "vision", "route": {}, "unknown": False, "control_reason": "", "model_result": None}
    prior = ua.last_physical_attempt_capture()
    try:
        with pytest.raises(error_type) as raised:
            vision_process._decode_terminal(packet, "fixture")
        assert type(raised.value) is error_type and raised.value.physical_attempt_capture.state == "settled"
        capture["state"] = "released"
        with pytest.raises(ValueError, match="settled"):
            vision_process._decode_terminal(packet, "fixture")
    finally:
        ua.adopt_physical_attempt_capture(prior)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("own", [True, False])  # the loop's own request (it accepts typed refusals), or a helper's
def test_a_control_read_after_a_received_refusal_keeps_its_exit_and_the_refusals_custody(setup, monkeypatch, asynchronous, own):
    """Every request's owner gets both facts: the control, and beside it the refusal its own request received."""
    root, gateway, client = setup
    failed = result(outcome="failed", problem={"code": "provider_content_refused", "retryable": False,
                                               "message": PRIVATE_DIAGNOSTIC})
    failed["message"] = None
    if own:
        failed["modelMismatch"] = {"requested": "fixture-model", "observed": "other-fixture-model"}
    gateway.results, received = [failed], []
    read = gateway.get_model_result

    def get_model_result(*args, **kwargs):
        received.append(args[0])
        return read(*args, **kwargs)

    gateway.get_model_result = get_model_result

    def no_alternative(*args, **kwargs):
        pytest.fail("The controlled refusal reached an alternate send decision")

    monkeypatch.setattr(transport.SubstitutionBudget, "admit", no_alternative)
    monkeypatch.setattr(transport.AccountRotation, "refused_call", no_alternative)
    monkeypatch.setattr(transport.CapabilityRetry, "refused", no_alternative)
    with pytest.raises(transport.ClaudexorModelError) as raised:
        call(client, asynchronous, model_poll_control=lambda: "finalize_requested" if received else None,
             **({"accepts_typed_refusals": True} if own else {}))
    error = raised.value
    assert error.code == "model_operation_interrupted" and error.control_reason == "finalize_requested"
    assert retained(root) == failed and ledger(root)[-1]["prompt_tokens"] == 20  # settled usage rides with it
    assert len(gateway.creates) == len(gateway.acks) == 1  # its custody was taken before the control was read
    refusal = error.received_refusal
    assert type(refusal) is transport.ClaudexorContentRefused and refusal.code == refusal.terminal_code
    assert refusal.physical_attempt_capture.state == "settled"
    assert PRIVATE_DIAGNOSTIC not in refusal.display_message + json.dumps(refusal.usage)


@pytest.mark.parametrize(("reason", "received"), [
    ("finalize_requested", True), ("finalize_requested", False), ("cancelled", True)])
def test_a_control_after_a_received_refusal_keeps_its_exit_and_never_sends_again(tmp_path, monkeypatch, reason, received):
    from ouroboros.model_wait import ModelWaitInterrupted

    refusal = transport.ClaudexorContentRefused(transport.ClaudexorContentRefused.terminal_problem())
    refusal.ledger_attempt_ids = ["sample-receipt"]
    refusal.usage = {"cost": 0.17, "cost_final": True, "prompt_tokens": 20, "completion_tokens": 7,
                     "claudexor": {"operation_state": "failed", "outcome": "failed"}}

    def interrupted():
        error = transport.ClaudexorModelError(
            {"code": "model_operation_interrupted", "message": "The caller interrupted model-result waiting.",
             "context": {"control_reason": reason, "cancellation": "requested"}}, model_role="main")
        error.control_reason = reason  # a Wrap up revoked before the loop read its mail, or a Stop
        if received:  # the receipt of the loop's own request, read at its model-call boundary
            error.received_refusal = refusal
            error.usage = refusal.usage
            error.ledger_attempt_ids = refusal.ledger_attempt_ids
        return error

    seen = {}
    if reason == "cancelled":
        with pytest.raises(ModelWaitInterrupted) as stopped:  # Stop keeps its own exit for the supervisor
            walk_loop(tmp_path, monkeypatch, interrupted, seen=seen)
        assert stopped.value.control_reason == "cancelled" and seen["llm"].calls == 1
        usage = seen["tools"]._ctx._accumulated_usage
    else:
        llm, text, usage, trace = walk_loop(tmp_path, monkeypatch, interrupted, seen=seen)
        if received:
            assert llm.calls == 1 and text == refusal.notice
            assert usage["reason_code"] == refusal.terminal_code and trace["refusal_reply_final_answer"] == ""
        else:
            assert llm.calls == 2  # an ordinary yielded call keeps its existing redial
    if received:
        assert (usage["cost"], usage["prompt_tokens"], usage["completion_tokens"]) == (0.17, 20, 7)
        assert usage["ledger_attempt_ids"] == ["sample-receipt"]
        assert usage["_last_llm_failure"]["operation_state"] == "failed"


def test_a_received_refusal_ends_one_request_where_its_caller_sends_nothing_more():
    from ouroboros.post_task_synthesis import propagate_paid_interruption

    for kind, code in ((transport.ClaudexorContentRefused, "provider_content_refused"),
                       (transport.ClaudexorRefusalError, "unexpected_refusal_content")):
        refusal = kind(kind.terminal_problem())
        assert kind.terminal_code == code == refusal.code and refusal.problem["retryable"] is False
        with pytest.raises(kind):  # a helper's route chain (Safety's) never asks another route
            transport.propagate_model_error(refusal)
        assert transport.propagate_model_error(refusal, single_request=True) is None
        assert propagate_paid_interruption(refusal) is None  # the later post-task stages still run
    control = transport.ClaudexorModelError({"code": "model_operation_interrupted", "message": "Stopped."})
    for propagate in (lambda error: transport.propagate_model_error(error, single_request=True),
                      propagate_paid_interruption):
        with pytest.raises(transport.ClaudexorModelError):
            propagate(control)  # a control keeps its meaning


def test_a_refused_caption_is_unavailable_for_the_task_and_asks_no_other_model(monkeypatch):
    from ouroboros import vision_routing

    models = iter(["first/vision-model", "second/vision-model"])
    monkeypatch.setattr(vision_routing, "prepare_caption_image", lambda *_args: (
        next(models), "data:image/png;base64,cGljbmlj", "prepared-digest", "", ""))
    asked = []

    def vision_query(*_args, **kwargs):
        asked.append(kwargs["model"])
        raise transport.ClaudexorContentRefused(transport.ClaudexorContentRefused.terminal_problem())

    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGljbmlj"}}
    usage, ctx = {}, SimpleNamespace(deadline_ts=None, task_attempt=None)
    llm = SimpleNamespace(vision_query=vision_query)
    first = vision_routing._caption_for_block(block, ctx=ctx, llm=llm, accumulated_usage=usage)
    again = vision_routing._caption_for_block(block, ctx=ctx, llm=llm, accumulated_usage=usage)
    assert first == again and first[0] == "" and transport.ClaudexorContentRefused.notice in first[1]
    assert asked == ["first/vision-model"]  # one send: a later round asks neither that model nor another


@pytest.mark.parametrize("wrapped", [True, False])  # in a task the model-wait seam surfaces the control wrapped
def test_a_caption_refusal_read_beside_a_control_is_kept_before_the_control_leaves(monkeypatch, wrapped):
    """Its refusal arrived, then a Wrap up revoked before the drain was read: the control leaves as itself, the
    caption's own refusal stays with the caption (asked once), and the request that carries on never takes it."""
    from ouroboros import vision_routing
    from ouroboros.model_wait import ModelWaitInterrupted, propagate_model_control

    monkeypatch.setattr(vision_routing, "prepare_caption_image", lambda *_args: (
        "caption/model", "data:image/png;base64,cGljbmlj", "prepared-digest", "", ""))
    asked = []

    def vision_query(*_args, **kwargs):  # the receipt ``finish`` raises: the control and, beside it, the refusal
        asked.append(kwargs["model"])
        control = transport.ClaudexorModelError({"code": "model_operation_interrupted", "message": "Wrap up.",
                                                 "context": {"control_reason": "finalize_requested"}}, model_role="vision")
        control.control_reason = "finalize_requested"
        control.received_refusal = transport.ClaudexorContentRefused(transport.ClaudexorContentRefused.terminal_problem())
        if wrapped:
            propagate_model_control(control)  # model_wait's call seam
        raise control

    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGljbmlj"}}
    usage, ctx = {}, SimpleNamespace(deadline_ts=None, task_attempt=None)
    llm = SimpleNamespace(vision_query=vision_query)
    with pytest.raises(ModelWaitInterrupted if wrapped else transport.ClaudexorModelError) as left:
        vision_routing._caption_for_block(block, ctx=ctx, llm=llm, accumulated_usage=usage)
    assert left.value.control_reason == "finalize_requested" and getattr(left.value, "received_refusal", None) is None
    again = vision_routing._caption_for_block(block, ctx=ctx, llm=llm, accumulated_usage=usage)  # the redialed round
    assert again[0] == "" and transport.ClaudexorContentRefused.notice in again[1] and asked == ["caption/model"]


def test_a_refused_extraction_keeps_the_review_unparsed_after_its_one_send():
    from ouroboros.review_verdict_extraction import canonicalize_session_verdict

    sends = []

    def chat(**kwargs):
        sends.append(kwargs["model"])
        raise transport.ClaudexorContentRefused(transport.ClaudexorContentRefused.terminal_problem())

    raw = "Review completed: the picnic menu lists two salads and no dessert."
    text, method, usage = canonicalize_session_verdict(raw, conformance_passed=False, llm=SimpleNamespace(chat=chat))
    assert (text, method, usage) == (raw, "unparsed", {}) and len(sends) == 1


def test_a_final_answer_is_read_only_inside_one_ordinary_run():
    from ouroboros._usage_response import assistant_answer_text

    trace = {"best_valid_final_answer": "older", "best_valid_final_answer_tools": 0}
    glued = [{"type": "text", "text": "FINAL ANS"}, {"type": "refusal", "refusal": ""}, {"type": "text", "text": "WER: picnic"}]
    _latch_final_answer_marker(trace, glued)
    assert trace["best_valid_final_answer"] == "older"  # no marker is formed across a refusal, an empty one included
    split = [{"type": "text", "text": "Menu:\n"}, {"type": "refusal", "refusal": "FINAL ANSWER: not this"},
             {"type": "text", "text": "\nFINAL ANSWER: two salads"}]
    _latch_final_answer_marker(trace, split)
    assert trace["best_valid_final_answer"] == "two salads"  # one run's own marker; the refusal's words select nothing
    assert assistant_answer_text(split) == "\nFINAL ANSWER: two salads"  # that one run: never stitched, never refused
    assert assistant_answer_text([{"type": "text", "text": "Menu."}, {"type": "refusal", "refusal": "x"},
                                  {"type": "text", "text": "Enjoy."}]) == ""  # two unmarked runs make no contract
    withdrawn = [{"type": "text", "text": "FINAL ANSWER:"}, {"type": "refusal", "refusal": ""},
                 {"type": "text", "text": "later words"}]
    _latch_final_answer_marker(trace, withdrawn)
    assert "best_valid_final_answer" not in trace  # an explicit empty selection in ordinary text withdraws the old
