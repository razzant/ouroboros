"""Bounded review repairs: public result URLs and one-copy input custody."""
from __future__ import annotations

import ipaddress
import json

import httpx
import pytest

from ouroboros import browser_policy
from ouroboros.gateways import image_api
from ouroboros.tools import imagegen as ig
from tests.test_imagegen_tool import PNG, configured as configured, ctx as ctx, _transport
from tests.test_imagegen_subscription import ImageWire, engine as engine, invoke, gateway_factory as gateway_factory


@pytest.mark.parametrize(("redirect", "address"), [
    (False, "169.254.169.254"),
    (False, "0.0.0.0"),
    (False, "224.0.0.1"),
    (False, "fd00:ec2::254"),
    (False, "::ffff:169.254.169.254"),
    (True, "127.0.0.1"),
    (True, "10.0.0.1"),
    (True, "240.0.0.1"),
    (True, "::1"),
    (True, "fe80::1"),
])
def test_provider_result_never_fetches_nonpublic_destination(monkeypatch, address, redirect):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "http://private.example/image.png"})
    _transport(monkeypatch, handle)
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda host:
        {ipaddress.ip_address("8.8.8.8" if host == "cdn.example" else address)})
    url = "https://cdn.example/image.png" if redirect else "http://private.example/image.png"
    with pytest.raises(image_api.ImageBackendError) as caught:
        image_api.read_result_image({"url": url})
    assert caught.value.code == "IMAGE_RESULT_INVALID"
    assert len(calls) == int(redirect)


def test_public_cdn_redirect_preserves_download_without_credential(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        assert "authorization" not in request.headers
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "/final.png?signature=opaque"})
        return httpx.Response(200, content=PNG)
    _transport(monkeypatch, handle)
    assert image_api.read_result_image({"url": "https://cdn.example/image.png"}) == (PNG, "image/png")
    assert len(calls) == 2 and calls[1].url.path == "/final.png"


def test_redirect_loop_is_bounded_and_dns_failure_is_recoverable(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "/again.png"})
    _transport(monkeypatch, handle)
    with pytest.raises(image_api.ImageBackendError) as caught:
        image_api.read_result_image({"url": "https://cdn.example/image.png"})
    assert caught.value.code == "IMAGE_DOWNLOAD_FAILED" and len(calls) == 6
    def unavailable(_host):
        raise OSError("synthetic DNS failure")
    monkeypatch.setattr(browser_policy, "_resolved_addresses", unavailable)
    with pytest.raises(image_api.ImageBackendError) as caught:
        image_api.read_result_image({"url": "https://cdn.example/image.png"})
    assert caught.value.code == "IMAGE_DOWNLOAD_FAILED" and len(calls) == 6


def test_many_subscription_polls_persist_input_payload_once(engine, ctx, monkeypatch):
    from ouroboros.observability import read_call_payload

    class PollingWire(ImageWire):
        polls = 0
        def respond(self, request):
            if request.method == "GET" and request.url.path == "/v2/model-operations/op-one":
                self.polls += 1
                if self.polls < 20:
                    return httpx.Response(200, json={**self.detail(), "state": "running", "response": {"state": "absent"}})
            return super().respond(request)
    wire = engine(PollingWire()); wire.pending = True
    (ctx.workspace_root / "edit.png").write_bytes(PNG + b"original-edit-metadata" * 10000)
    persisted = []
    retain = ig._retain_call
    def spy(*args, **kwargs):
        persisted.append((args[3], args[4]))
        return retain(*args, **kwargs)
    monkeypatch.setattr(ig, "_retain_call", spy)
    monkeypatch.setattr(ig.time, "sleep", lambda _: None)
    result = invoke(ctx, images=[{"root": "active_workspace", "path": "edit.png"}])
    assert result.status == "ok" and wire.polls == 20 and wire.generation_count == 1
    requests = [payload for stage, payload in persisted if stage == "request"]
    assert requests == [json.loads(wire.payload)]
    manifest, original, _ = read_call_payload(ctx.drive_root, task_id=ctx.task_id,
                                            call_id=f"{wire.operation[0]}_model_request")
    assert original == requests[0]
    assert manifest["request_ref"]["sha256"] and manifest["operation_id"] == "op-one"
    observations = [payload for stage, payload in persisted if stage == "response" and "operation" in payload]
    assert len(observations) == 21 and all("images" not in payload for payload in observations)


def test_generated_artifact_explicitly_enters_native_vision(ctx, monkeypatch):
    import base64
    from pathlib import Path
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_imagegen_tool import _summary

    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
    _transport(monkeypatch, provider)
    ctx.messages = []
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    generated = registry.execute_result("generate_image", {"prompt": "fixture", "model": "openai::fixture-image"})
    assert generated.status == "ok" and ctx.messages == [] and ctx.pending_events == []
    image = _summary(generated.text)["images"][0]
    viewed = registry.execute_result("view_image", {"path": image["absolute_path"]})
    assert viewed.status == "ok", viewed
    blocks = [block for message in ctx.messages for block in message.get("content", [])
              if isinstance(block, dict) and block.get("type") == "image_url"]
    assert len(blocks) == 1
    assert base64.b64decode(blocks[0]["image_url"]["url"].split(",", 1)[1]) == PNG
    assert Path(blocks[0]["_source_path"]).read_bytes() == PNG
    assert Path(image["absolute_path"]).read_bytes() == PNG
    assert len(calls) == 1 and ctx.pending_events == []


def test_ready_not_started_receipt_waits_for_failed_release_and_recovery(engine, ctx, monkeypatch):
    from ouroboros import usage_accounting as usage
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_claudexor_model_transport import _bytes
    from tests.test_imagegen_tool import _summary

    wire = ImageWire(); wire.unsent = wire.ready_unsent = True
    wire.native.update(outcome="failed", images=[], problem={"code": "quota_unavailable", "message": "Image quota",
                       "retryable": True, "context": {"requestKind": "image"}})
    wire.result = _bytes(wire.native); engine(wire)
    transition = usage._transition
    def unavailable(reservation, state, **kwargs):
        if state == "released":
            raise OSError("synthetic release write unavailable")
        return transition(reservation, state, **kwargs)
    monkeypatch.setattr(usage, "_transition", unavailable)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root); registry.set_context(ctx)
    initial = invoke(ctx)
    assert initial.code == "IMAGE_NOT_DISPATCHED"
    request_id = _summary(initial.text)["request_id"]
    retry = registry.execute_result("recover_image", {"request_id": request_id})
    for result, warning in ((initial, "accounting release is pending"),
                            (retry, "Original accounting is pending; use recover_image before acknowledging custody.")):
        facts = _summary(result.text)
        assert facts["response"] and facts["acknowledgement"] == "pending"
        assert warning in " ".join(facts["warnings"])
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    rows = usage.read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["state"] == "unresolved"
    monkeypatch.setattr(usage, "_transition", transition)
    recovered = registry.execute_result("recover_image", {"request_id": request_id})
    assert recovered.code == "IMAGE_NOT_DISPATCHED"
    assert _summary(recovered.text)["acknowledgement"] == "acknowledged"
    assert usage.read_usage_records(ctx.drive_root)[0]["state"] == "released"
    assert len([call for call in wire.calls if call.url.path.endswith("/ack")]) == 1
    assert wire.generation_count == wire.upload_count == 1
