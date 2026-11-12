"""Real registry/ledger consumers for image diagnostic and response-custody faults."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ouroboros.tools.registry import ToolRegistry
from tests.test_imagegen_tool import BODY, PNG, _summary, _transport, configured as configured, ctx as ctx
from tests.test_imagegen_subscription import CAPABILITY, ImageWire, engine as engine, gateway_factory as gateway_factory, invoke


def _registry(ctx):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry


@pytest.mark.parametrize("fault", [None, "busy", "store", "os"])
@pytest.mark.parametrize("settlement_failure", [False, True])
def test_received_direct_response_survives_manifest_fault_on_original_attempt(
        monkeypatch, ctx, fault, settlement_failure):
    from ouroboros import observability, store, usage_accounting as usage
    from ouroboros.artifacts import task_artifact_dir_path

    calls, failed_manifests = [], []
    result_body = {**BODY, "usage": {"input_tokens": 5, "output_tokens": 100},
                   "provider_metadata": {"label": "received — keep exact content"}}
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=result_body)
    _transport(monkeypatch, respond)
    write_manifest = observability.write_call_manifest
    failures = {"busy": store.StoreBusy, "store": store.StoreError, "os": OSError}
    def write(drive_root, **kwargs):
        if fault and kwargs["manifest"].get("call_type") == "image_api_response":
            failed_manifests.append(kwargs["call_id"])
            raise failures[fault]("fixture response manifest unavailable")
        return write_manifest(drive_root, **kwargs)
    monkeypatch.setattr(observability, "write_call_manifest", write)
    if settlement_failure:
        def fail_settlement(*args, **kwargs):
            raise usage.UsageAccountingError("fixture accounting unavailable")
        monkeypatch.setattr(usage, "settle_attempt", fail_settlement)

    registry = _registry(ctx)
    result = registry.execute_result("generate_image", {"prompt": "paint", "model": "openai::model-x"})
    assert result.code == ("IMAGE_RESULT_INCOMPLETE" if fault or settlement_failure else "OK"), result
    facts = _summary(result.text)
    rows = usage.read_usage_records(ctx.drive_root)
    assert len(rows) == len(calls) == 1 and calls[0].method == "POST"
    assert rows[0]["attempt_id"] == facts["request_id"]
    assert rows[0]["state"] == ("unresolved" if settlement_failure else "settled")
    if settlement_failure:
        assert "cost_usd" not in rows[0] and "cost_final" not in rows[0]
    else:
        assert rows[0]["cost_usd"] is None and not rows[0]["cost_final"]
    assert facts["actual_cost_usd"] is facts["estimated_cost_usd"] is None
    assert facts["cost_final"] is False and facts["generated"] == 1
    assert Path(facts["images"][0]["absolute_path"]).read_bytes() == PNG
    response = task_artifact_dir_path(ctx.drive_root, ctx.task_id) / facts["response"]["path"]
    assert response.read_bytes() == json.dumps(result_body, ensure_ascii=False).encode("utf-8")
    assert result.meta["artifact_registered"] is True and not ctx.pending_events
    if fault:
        assert failed_manifests == [f'{facts["request_id"]}_image_response']
        assert facts["failures"] == [{"stage": "private_response_retention", "error": failures[fault].__name__}]
        assert "response file" in result.text and "do not generate again" in result.text
        # Private manifest failure does not permit re-generation. Ordinary files
        # already expose the exact received material even if observation is unknown.
        assert registry.execute_result("recover_image", {"request_id": facts["request_id"]}).code == "IMAGE_OUTCOME_UNKNOWN"
    assert len(calls) == len(usage.read_usage_records(ctx.drive_root)) == 1


@pytest.mark.parametrize("kind", ["missing", "invalid_bytes", "invalid_reference"])
def test_input_refusal_is_typed_before_any_submission(monkeypatch, ctx, tmp_path, kind):
    from ouroboros import usage_accounting as usage

    (tmp_path / "outside.png").write_bytes(PNG)
    (ctx.workspace_root / "not-image.png").write_bytes(b"not a PNG")
    (ctx.workspace_root / "escape.png").symlink_to(tmp_path / "outside.png")
    refs = {
        "escape": {"root": "active_workspace", "path": "../outside.png"},
        "symlink_escape": {"root": "active_workspace", "path": "escape.png"},
        "unknown_root": {"root": "not_a_root", "path": "input.png"},
        "missing": {"root": "active_workspace", "path": "missing.png"},
        "invalid_bytes": {"root": "active_workspace", "path": "not-image.png"},
        "invalid_reference": {"root": "active_workspace", "path": ""},
    }
    _transport(monkeypatch, lambda request: pytest.fail("input refusal sent a provider request"))
    result = _registry(ctx).execute_result("generate_image", {
        "prompt": "edit", "model": "openai::model-x", "images": [refs[kind]]})
    expected = {"escape": "PATH_BLOCKED", "symlink_escape": "PATH_BLOCKED", "unknown_root": "TOOL_ARG_ERROR", "missing": "TOOL_ERROR",
                "invalid_bytes": "TOOL_ARG_ERROR", "invalid_reference": "TOOL_ARG_ERROR"}[kind]
    assert result.code == expected, result
    assert result.status == ("blocked" if expected == "PATH_BLOCKED" else "error")
    assert usage.read_usage_records(ctx.drive_root) == []
    assert not ctx.pending_events


@pytest.mark.parametrize("field,value", [
    ("id", None), ("id", ""), ("maxInputImages", None), ("maxInputImages", True),
    ("supportedOptions", None), ("models", None),
])
def test_malformed_engine_capability_is_unavailable_not_a_caller_error(engine, ctx, field, value):
    from ouroboros import usage_accounting as usage

    wire = ImageWire()
    original = wire.respond
    source = {"id": "codex", "imageGeneration": dict(CAPABILITY)}
    target = source if field == "id" else source["imageGeneration"]
    if value is None:
        target.pop(field)
    else:
        target[field] = value
    def respond(request):
        if request.url.path == "/v2/model-sources":
            return httpx.Response(200, json={"sources": [source]})
        return original(request)
    wire.respond = respond
    engine(wire)
    result = invoke(ctx)
    assert result.code == "CAPABILITY_UNAVAILABLE", result
    assert wire.upload_count == wire.generation_count == 0
    assert usage.read_usage_records(ctx.drive_root) == []
    assert '"request_id"' not in result.text
    from ouroboros.tools import imagegen
    rows, errors = imagegen._subscription_catalog()
    assert rows == [] and len(errors) == 1 and errors[0]["code"] == "CAPABILITY_UNAVAILABLE"
    assert wire.upload_count == wire.generation_count == 0


def test_unavailable_engine_does_not_advertise_an_unreserved_request(monkeypatch, ctx):
    from ouroboros import usage_accounting as usage
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def unavailable():
        raise ClaudexorUnavailable("fixture_offline", "No fixture engine")
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", unavailable)
    result = invoke(ctx)
    assert result.code == "CAPABILITY_UNAVAILABLE"
    assert '"request_id"' not in result.text
    assert usage.read_usage_records(ctx.drive_root) == []


def test_known_zero_edit_capability_remains_a_valid_generation_source(engine, ctx):
    wire = ImageWire()
    original = wire.respond
    capability = {**CAPABILITY, "maxInputImages": 0, "supportedOptions": []}
    def respond(request):
        if request.url.path == "/v2/model-sources":
            return httpx.Response(200, json={"sources": [{"id": "codex", "imageGeneration": capability}]})
        return original(request)
    wire.respond = respond
    engine(wire)
    from ouroboros.tools import imagegen
    rows, errors = imagegen._subscription_catalog()
    assert errors == [] and rows[0]["editing"] is False and rows[0]["parameters"] == []
    assert invoke(ctx).code == "OK"
    assert wire.upload_count == wire.generation_count == 1


@pytest.mark.parametrize("control", ["budget", "stop"])
def test_postresponse_custody_catch_does_not_intercept_send_controls(monkeypatch, ctx, control):
    from ouroboros import usage_accounting as usage
    from ouroboros.model_wait import ModelWaitInterrupted

    error = usage.BudgetExceeded("fixture budget rail") if control == "budget" else ModelWaitInterrupted("owner_stop")
    from ouroboros.gateways import image_api
    def send(*args, **kwargs):
        raise error
    monkeypatch.setattr(image_api, "send_image_request", send)
    with pytest.raises(type(error)) as caught:
        _registry(ctx).execute_result("generate_image", {"prompt": "paint", "model": "openai::model-x"})
    assert caught.value is error


def test_request_manifest_failure_never_reaches_the_provider(monkeypatch, ctx):
    from ouroboros import observability, store, usage_accounting as usage

    write_manifest = observability.write_call_manifest
    def fail_request(drive_root, **kwargs):
        if kwargs["manifest"].get("call_type") == "image_api_request":
            raise store.StoreBusy("fixture request manifest unavailable")
        return write_manifest(drive_root, **kwargs)
    monkeypatch.setattr(observability, "write_call_manifest", fail_request)
    _transport(monkeypatch, lambda request: pytest.fail("request custody failure dispatched"))
    result = _registry(ctx).execute_result("generate_image", {"prompt": "paint", "model": "openai::model-x"})
    assert result.code == "CAPABILITY_UNAVAILABLE", result
    rows = usage.read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["state"] == "released"
    assert not result.meta.get("artifact_registered") and not ctx.pending_events
