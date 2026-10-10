"""Recover existing model-operation receipts without generating replacement work."""
import hashlib
import json

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros.observability import call_manifest_path, persist_call, read_call_payload


ROW = {"attempt_id": "attempt-one", "task_id": "task-one"}
RESULT = {"outcome": "completed", "message": {"content": "Ответ 🐍"},
          "usage": {"input_tokens": 20, "output_tokens": 4},
          "cost": {"knowledge": "exact", "cashUsd": .2}}


class Gateway:
    def __init__(self, *, state="succeeded", dispatch="response_received", ready=True):
        self.raw = json.dumps(RESULT, ensure_ascii=False).encode()
        self.ref = {"resourceId": "result-one", "sha256": "sha256:" + hashlib.sha256(self.raw).hexdigest(),
                    "sizeBytes": len(self.raw)}
        self.detail = {"id": "op-one", "state": state, "dispatch": {"state": dispatch},
                       "response": {"state": "ready" if ready else "expired", "ref": self.ref}}
        self.reads, self.results, self.acks = [], [], []
        self.closed = False

    def get_model_operation(self, operation_id, **kwargs):
        self.reads.append(operation_id)
        return self.detail

    def get_model_result(self, operation_id, *, expected_ref, raw_bytes, **kwargs):
        assert expected_ref == self.ref and raw_bytes is True
        self.results.append(operation_id)
        return self.raw

    def acknowledge_model_result(self, operation_id, digest):
        self.acks.append((operation_id, digest))

    def close(self):
        self.closed = True


def request(root, **manifest):
    persist_call(root, task_id=ROW["task_id"], call_id="attempt-one_model_request",
                 call_type="llm_claudexor_request", payload={"messages": []}, keep_raw=True,
                 manifest={"invocation_id": "attempt-one", **manifest})


def response(root, **manifest):
    persist_call(root, task_id=ROW["task_id"], call_id="attempt-one_model_response",
                 call_type="llm_claudexor_response",
                 payload={"result_json_utf8": json.dumps(RESULT, ensure_ascii=False)}, keep_raw=True,
                 manifest={"invocation_id": "attempt-one", "operation_id": "op-one", **manifest})


def test_retained_terminal_usage_requires_no_network(tmp_path):
    response(tmp_path, operation_state="succeeded", dispatch_state="response_received")
    def no_network():
        raise AssertionError("A retained terminal response must not query the daemon")
    disposition, usage, cost, final = transport.recover_model_attempt(tmp_path, ROW, gateway_factory=no_network)
    assert disposition == "settled" and cost == .2 and final is True
    assert usage["prompt_tokens"] == 20 and usage["completion_tokens"] == 4


def test_early_cancel_defers_until_existing_operation_ends(tmp_path):
    request(tmp_path, operation_id="op-one")
    gateway = Gateway(state="running", dispatch="started")
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: gateway) is None
    assert not call_manifest_path(tmp_path, ROW["task_id"], "attempt-one_model_response").exists()
    gateway.detail.update(state="succeeded", dispatch={"state": "response_received"})
    got = transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: gateway)
    assert got[0] == "settled" and got[2:] == (.2, True)
    manifest, payload, _ = read_call_payload(tmp_path, task_id=ROW["task_id"], call_id="attempt-one_model_response")
    assert payload["result_json_utf8"].encode() == gateway.raw
    assert manifest["dispatch_state"] == "response_received"
    assert gateway.results == ["op-one"] and len(gateway.acks) == 1 and not gateway.closed


def test_missing_operation_identity_never_creates_work(tmp_path):
    request(tmp_path)
    def no_network():
        raise AssertionError("No operation id grants a new generation")
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=no_network) is None


def test_legacy_response_rechecks_exact_dispatch_before_release(tmp_path):
    request(tmp_path, operation_id="op-one")
    response(tmp_path)
    gateway = Gateway(state="failed", dispatch="not_started")
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: gateway)[0] == "released"
    assert gateway.reads == ["op-one"] and not gateway.results


@pytest.mark.parametrize("proven", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_late_upload_proof_keeps_attempt_but_releases_only_definite_non_delivery(tmp_path, proven, ready):
    request(tmp_path, operation_id="op-one")
    gateway = Gateway(state="failed", dispatch="not_started" if proven else "unknown", ready=ready)
    gateway.detail["dispatch"].update(startedAt="2026-09-30T00:00:00Z", route={"source": "codex"})
    problem = {"code": "transport_not_delivered" if proven else "transport_unknown",
               "message": "Upload ended", "context": {"bodyBytes": 3_000_000,
               "handedToSocketBytes": 65_536 if proven else 3_000_000}}
    gateway.detail["problem"] = problem
    gateway.raw = json.dumps({"outcome": "failed" if proven else "unknown", "message": None,
                              "problem": problem, "usage": {}, "cost": {"knowledge": "unknown"}}).encode()
    got = transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: gateway)
    assert got == ("released" if proven else "abandoned", {}, None, False)
    assert gateway.reads == ["op-one"] and len(gateway.acks) == int(ready)
    # Once retained, reconciliation uses the same receipt without another engine read.
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: pytest.fail("new lookup")) == got


@pytest.mark.parametrize("dispatch", ["started", "unknown", "response_received"])
def test_terminal_operation_without_response_keeps_price_unknown(tmp_path, dispatch):
    request(tmp_path, operation_id="op-one")
    gateway = Gateway(state="failed", dispatch=dispatch, ready=False)
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: gateway) == ("abandoned", {}, None, False)
    assert not gateway.acks


def test_conflicting_operation_binding_is_not_recovered(tmp_path):
    request(tmp_path, operation_id="op-different")
    response(tmp_path, operation_state="succeeded", dispatch_state="response_received")
    assert transport.recover_model_attempt(tmp_path, ROW) is None


@pytest.mark.parametrize("retained", [True, False])
@pytest.mark.parametrize("dispatch,disposition", [
    ("response_received", "settled"), ("not_started", "released"), ("unknown", "abandoned")])
def test_interrupted_operation_is_terminal_offline_and_live(tmp_path, retained, dispatch, disposition):
    """The engine's ``interrupted`` is terminal (custody's TERMINAL_STATES): a received
    response settles, a proven non-start releases, an unknown dispatch is abandoned at an
    unknown price; none of them defers into the next pass's engine read."""
    gateway = Gateway(state="interrupted", dispatch=dispatch, ready=dispatch == "response_received")
    if retained:
        response(tmp_path, operation_state="interrupted", dispatch_state=dispatch)
    else:
        request(tmp_path, operation_id="op-one")
    got = transport.recover_model_attempt(
        tmp_path, ROW, gateway_factory=(lambda: pytest.fail("a retained terminal receipt needs no read"))
        if retained else (lambda: gateway))
    assert got is not None and got[0] == disposition
    if disposition == "settled":
        assert got[2:] == (.2, True) and got[1]["prompt_tokens"] == 20
    else:
        assert got == (disposition, {}, None, False)
    assert len(gateway.acks) == int(not retained and dispatch == "response_received")
    manifest, _payload, _ = read_call_payload(tmp_path, task_id=ROW["task_id"], call_id="attempt-one_model_response")
    assert (manifest["operation_state"], manifest["dispatch_state"]) == ("interrupted", dispatch)
    # The receipt now decides alone: the next maintenance pass makes no engine request.
    assert transport.recover_model_attempt(tmp_path, ROW, gateway_factory=lambda: pytest.fail("new lookup")) == got
