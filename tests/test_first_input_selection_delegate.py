"""Direct helper input selection and immutable replay; native reviewer packets stay exact."""
from __future__ import annotations

import json

import pytest

from tests._delegated_transport_shared import (  # noqa: F401 - installs the offline actor fixture
    _LiveRunStub,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)


@pytest.mark.parametrize("carrier", ["contract", "metadata_contract", "metadata", "explicit"])
def test_declared_direct_start_uses_only_selected_prompt_and_authority(tmp_path, monkeypatch, carrier):
    from ouroboros import chronicle_view
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools import delegate
    requests = []
    class Stub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append(request)
            return super().start_run(request, idempotency_key=idempotency_key)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=ordinary-model:high")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: Stub())
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("declared captured memory"))
    ctx = _nanny_ctx(tmp_path)
    contract = {"context": "PRIOR_CONTEXT", "notes": "PRIOR_NOTE", "constraints": "FULL_AUTHORITY"}
    ctx.task_contract = contract
    options = {}
    if carrier == "contract":
        contract["input_sources"] = "declared"
    elif carrier == "metadata_contract":
        ctx.task_contract = {}
        ctx.task_metadata["task_contract"] = {**contract, "input_sources": "declared"}
    elif carrier == "metadata":
        ctx.task_metadata["input_sources"] = "declared"
    else:
        options["input_sources"] = "declared"
    result = json.loads(delegate._delegate_start(ctx, "CHOSEN_EVIDENCE", **options).text)
    assert result["status"] == "started", result
    assert requests[0]["prompt"] == "CHOSEN_EVIDENCE"
    assert "FULL_AUTHORITY" in requests[0]["instructions"]
    assert "PRIOR_CONTEXT" not in requests[0]["instructions"]
    assert "PRIOR_NOTE" not in requests[0]["instructions"]
    assert "MEMORY REFERENCE" not in requests[0]["instructions"]


def test_direct_shared_memory_is_recorded_and_retry_never_recaptures(tmp_path, monkeypatch):
    from ouroboros import chronicle_view, delegate_custody
    from ouroboros.subagent_work_order import chosen_request_fingerprint
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools import delegate
    requests, captures = [], []
    class Stub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append((json.loads(json.dumps(request)), idempotency_key))
            if len(requests) == 1:
                raise gateway.ClaudexorUnavailable("daemon_unreachable", "lost response")
            return super().start_run(request, idempotency_key=idempotency_key)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=ordinary-model:high")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: Stub())
    def capture(*_a, **_kw):
        captures.append(True)
        return {"text": "ORIGINAL_SELECTED_MEMORY", "snapshot_sha256": "first", "facts": {
            "native_fit": "unobserved", "canonical_root": "/canonical/data",
            "derived_journal": "memory/chronicle/records.jsonl",
            "source_read": "Read immutable record by id with file tools, or request it through the existing parent channel."}}
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", capture)
    ctx = _nanny_ctx(tmp_path)
    lost = json.loads(delegate._delegate_start(ctx, "UNALTERED_PROMPT").text)
    token = lost["pending_invocation_id"]
    recorded = delegate_custody.invocation_record(tmp_path, token)
    assert recorded["work_order_fingerprint"] == chosen_request_fingerprint(requests[0][0])
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("retry reread memory"))
    resumed = json.loads(delegate._delegate_start(ctx, "UNALTERED_PROMPT", retry_of=token).text)
    assert resumed["status"] == "started"
    assert requests[0] == requests[1]
    assert len(captures) == 1
    assert requests[0][0]["prompt"] == "UNALTERED_PROMPT"
    assert requests[0][0]["instructions"].count("ORIGINAL_SELECTED_MEMORY") == 1
    assert "not task authority" in requests[0][0]["instructions"]
    assert "/canonical/data" in requests[1][0]["instructions"]
    assert "memory/chronicle/records.jsonl" in requests[1][0]["instructions"]
    assert "existing parent channel" in requests[1][0]["instructions"]


def test_native_reviewer_request_never_uses_helper_memory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import chronicle_view
    from ouroboros.review_session_preparation import prepare_review_session_request
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("review memory injection"))
    invocation = SimpleNamespace(timeout_sec=60, instructions="NATIVE_REVIEW_GOVERNANCE", use_thread=False,
                                 output_schema=None, source_delivery=None)
    route = SimpleNamespace(route_id="test-route", model="test-model", effort="high", profile_id="")
    request = prepare_review_session_request(invocation, route, prompt="EXACT_REVIEW_PACKET", root=str(tmp_path),
                                             thread_id="", schema_asked=False)
    assert request["prompt"] == "EXACT_REVIEW_PACKET"
    assert request["instructions"] == "NATIVE_REVIEW_GOVERNANCE"


@pytest.mark.parametrize("selection", [None, "shared", "declared"])
def test_ordinary_direct_start_and_retry_preserve_request_and_parent_context(tmp_path, monkeypatch, selection):
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools import delegate

    requests = []

    class RetryStub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append((json.loads(json.dumps(request)), idempotency_key))
            if len(requests) == 1:
                raise gateway.ClaudexorUnavailable("daemon_unreachable", "offline lost response")
            return super().start_run(request, idempotency_key=idempotency_key)

    stub = RetryStub()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=ordinary-model:high")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: stub)
    ctx = _nanny_ctx(tmp_path)
    ctx.task_contract = {"context": "PREVIOUS_CASE_CONTEXT", "constraints": "PARENT_AUTHORITY"}
    if selection is not None:
        ctx.task_contract["input_sources"] = selection
    prompt = "Continue ordinary work"

    lost = json.loads(delegate._delegate_start(ctx, prompt, max_seconds=120).text)
    assert lost["reason"] == "daemon_unreachable", lost
    token = lost["pending_invocation_id"]
    resumed = json.loads(delegate._delegate_start(ctx, prompt, retry_of=token).text)

    assert resumed["status"] == "started" and resumed["idempotent_recovery"] is True, resumed
    assert len(requests) == 2 and requests[0] == requests[1]
    request, key = requests[0]
    assert key == token and request["prompt"] == prompt
    assert (request["model"], request["effort"], request["maxSeconds"]) == ("ordinary-model", "high", 120)
    assert ("PREVIOUS_CASE_CONTEXT" in json.dumps(request)) == (selection != "declared")
    assert "PARENT_AUTHORITY" in json.dumps(request)


def test_registered_direct_selector_reaches_real_request(tmp_path, monkeypatch):
    from ouroboros import chronicle_view, config, safety
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools.registry import ToolRegistry
    requests = []
    class Stub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append(request)
            return super().start_run(request, idempotency_key=idempotency_key)
    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_SUBAGENTS": {"enabled": True, "items": [{
        "subagent_id": "session", "recommended_use": "Inspect sources",
        "route": {"kind": "agent_session", "target_id": "some-route=ordinary-model"}}]}})
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: Stub())
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("selector was dropped"))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx = _nanny_ctx(tmp_path)
    registry._ctx.task_contract = {"context": "AUTOMATIC_OLD_CONTEXT", "constraints": "KEEP_AUTHORITY"}
    result = registry.execute_result("delegate_start", {
        "prompt": "ONLY_CHOSEN_INPUT", "subagent_id": "session", "input_sources": "declared"})
    assert result.status == "ok", result.text
    assert requests[0]["prompt"] == "ONLY_CHOSEN_INPUT"
    assert "AUTOMATIC_OLD_CONTEXT" not in requests[0]["instructions"]
    assert "KEEP_AUTHORITY" in requests[0]["instructions"]
    assert "INPUT SOURCE SELECTION" in requests[0]["instructions"]


def test_recorded_source_request_never_gains_new_automatic_memory(tmp_path, monkeypatch):
    from ouroboros import chronicle_view
    from ouroboros.subagent_work_order import direct_start_instructions
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("historical source augmented"))
    ctx = _nanny_ctx(tmp_path)
    ctx.task_contract = {"constraints": "ORIGINAL_TASK_AUTHORITY"}
    assignment, reference = direct_start_instructions(ctx, {}, "shared", None, source_bound=True)
    assert "ORIGINAL_TASK_AUTHORITY" in assignment
    assert reference == ""


def test_declared_direct_parent_cannot_widen_selection_before_transport(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.tools import delegate
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: pytest.fail("widened selection sent"))
    ctx = _nanny_ctx(tmp_path)
    ctx.task_contract = {"input_sources": "declared"}
    result = json.loads(delegate._delegate_start(ctx, "question", input_sources="shared").text)
    assert result["reason"] == "INPUT_SOURCE_SELECTION_INVALID"
    assert "cannot widen" in result["detail"]
