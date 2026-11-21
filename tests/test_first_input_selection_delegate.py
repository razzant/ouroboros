"""Declared inputs reach a configured session through the same start path as shared
ones: a model-authored delegate_start and its retry replay send the complete normalized
contract (selection, declared context, inherited intent note) as authority; the nanny's
compiled work order (test_first_input_selection_context) carries the receipt. No route
is refused for its kind."""
from __future__ import annotations

import json

import pytest

from tests._delegated_transport_shared import (  # noqa: F401 - installs the offline actor fixture
    _LiveRunStub,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)


@pytest.mark.parametrize("carrier", ["contract", "metadata_contract", "metadata"])
def test_declared_direct_start_sends_the_normalized_contract_authority(tmp_path, monkeypatch, carrier):
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools import delegate

    requests = []

    class RecordingStub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append(json.loads(json.dumps(request)))
            return super().start_run(request, idempotency_key=idempotency_key)

    # A READ-ONLY child (the stub serves readonly only): technical access alone decides
    # the run shape now (#1609), so the read-only premise is stated, not inferred.
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=ordinary-model:high")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: RecordingStub())
    ctx = _nanny_ctx(tmp_path)
    contract = {"input_sources": "declared", "context": "COMMON_EVIDENCE", "constraints": "PARENT_AUTHORITY",
                "delegation_budget": {"intent_note": "ADVICE_NOTE"}}
    if carrier == "contract":
        ctx.task_contract = contract
    elif carrier == "metadata_contract":
        ctx.task_metadata["task_contract"] = contract
    else:
        ctx.task_metadata["input_sources"] = "declared"

    result = delegate._delegate_start(ctx, "Inspect the declared evidence", max_seconds=120)

    payload = json.loads(result.text)
    assert payload["status"] == "started", payload
    assert "INPUT_SOURCE_SELECTION_UNSUPPORTED" not in result.text
    [request] = requests
    assert request["prompt"] == "Inspect the declared evidence"
    sent = "\n".join(str(value) for value in request.values())
    if carrier != "metadata":
        # A direct start carries the complete normalized contract as authority: the leaf
        # hears the selection, the declared evidence and the inherited intent note.
        assert "HOST TASK CONTRACT AUTHORITY" in sent
        assert '"input_sources":"declared"' in sent
        assert "COMMON_EVIDENCE" in sent and "PARENT_AUTHORITY" in sent and "ADVICE_NOTE" in sent


def test_registered_direct_start_with_declared_selection_starts(tmp_path, monkeypatch):
    """The registered tool path (subagent_id → configured session) starts a declared leaf
    through work-order preparation and the gateway, like any shared start."""
    from ouroboros import safety
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.settings_integrity.runtime_catalog_settings", lambda *, settings_reader=None: {
        "OUROBOROS_SUBAGENTS": {"enabled": True, "items": [{
            "subagent_id": "session", "recommended_use": "Inspect the source.",
            "route": {"kind": "agent_session", "target_id": "some-route=weak"},
        }]},
    })
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_kw: _LiveRunStub())
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_id = "declared-child"
    registry._ctx.task_metadata = {"root_task_id": "declared-child", "parent_task_id": "declared-child"}
    registry._ctx.task_contract = {"input_sources": "declared", "context": "COMMON_EVIDENCE"}

    # The configured row captures `full`; this read-only leaf lowers it explicitly at the
    # call (#1609), the same choice a model makes for a read-only inspection.
    result = registry.execute_result("delegate_start", {
        "prompt": "Inspect declared evidence", "subagent_id": "session", "access": "readonly"})

    payload = json.loads(result.text)
    assert payload.get("status") == "started", result.text
    assert payload.get("reason") != "INPUT_SOURCE_SELECTION_UNSUPPORTED"


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
    _select_transport_access(monkeypatch, "readonly")   # a read-only child (#1609)
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
    assert "PREVIOUS_CASE_CONTEXT" in json.dumps(request)
    assert "PARENT_AUTHORITY" in json.dumps(request)
