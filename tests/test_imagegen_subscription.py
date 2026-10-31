"""Engine image schema through real gateway, ledger, registry and artifact consumers."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft7Validator

from ouroboros.tools import imagegen as ig
from ouroboros.tools.registry import ToolRegistry
from tests.test_claudexor_model_transport import ModelWire, _bytes, _ref, gateway_factory as gateway_factory
from tests.test_imagegen_tool import PNG, configured as configured, ctx as ctx, _summary

CAPABILITY = {"models": ["gpt-image-2"], "modelsProvenance": "fixture upstream hint",
              "inputMimes": ["image/png", "image/jpeg", "image/webp"], "maxInputImages": 5,
              "maxInputImagesProvenance": "fixture tool cap",
              "supportedOptions": ["size", "quality", "background", "n"]}

# Exact generated schemas from engine patch 798a3faebdccdaea309c8179cfd3999b98037b6524a90045e21522c133b87d0e.
# Runtime release pinning remains separate. These fixtures establish the consumer wire.
SCHEMAS = {name: json.loads((Path(__file__).parent / "fixtures" / "claudexor_image_contract" /
                            f"{name}.schema.json").read_bytes()) for name in ("ImageCallRequest", "ImageCallResult")}


class ImageWire(ModelWire):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.native = {"outcome": "completed", "operation": "edit", "requestedModel": "requested-image",
            "images": [{"mime": "image/png", "sha256": _ref(PNG)["sha256"], "sizeBytes": len(PNG),
                        "dataBase64": base64.b64encode(PNG).decode(), "generationId": None} for _ in range(2)],
            "route": {"source": "codex", "credentialProfileId": "chosen", "accountFingerprint": "fixture", "model": None},
            "appliedOptions": {}, "usage": {"input_tokens": 10, "output_tokens": 20},
            "cost": {"cashUsd": None, "knowledge": "unknown", "billing": "unknown",
                     "source": "fixture", "provenance": ["fixture receipt"]},
            "providerRequestId": None, "problem": None}
        self.result = _bytes(self.native)
        self.pending = False
        self.before_ack = lambda: None
        self.poll_failure = False
        self.unsent = False
        self.lifecycle = "succeeded"
        self.dispatch = "response_received"
        self.ready_unsent = False

    def respond(self, request):
        method, path = request.method, request.url.path
        if path == "/v2/operations":
            return httpx.Response(200, json={"operations": [{"method": "POST", "path": "/v2/model-operations",
                "parameters": [{"name": "requestKind", "location": "query", "enum": ["image"]}]}]})
        if path == "/v2/model-sources":
            assert request.url.params["view"] == "accounts"
            return httpx.Response(200, json={"sources": [{"id": "codex", "credentialHarness": "codex", "imageGeneration": CAPABILITY}]})
        if method == "POST" and path == "/v2/model-operations":
            assert dict(request.url.params) == {"requestKind": "image"}
            Draft7Validator(SCHEMAS["ImageCallRequest"]).validate(json.loads(self.payload))
            result = super().respond(request)
            if self.pending:
                return httpx.Response(202, json={**self.detail(), "state": "running", "response": {"state": "absent"}})
            return result
        if path == "/v2/model-operations/op-one/control":
            assert json.loads(request.content) == {"action": "cancel", "reasonCode": "host_cancelled"}
            return httpx.Response(200, json={**self.detail(), "state": "running"})
        if path == "/v2/model-operations/op-one" and self.poll_failure:
            raise httpx.ConnectError("control disappeared", request=request)
        if path == "/v2/model-operations/op-one/result":
            Draft7Validator(SCHEMAS["ImageCallResult"]).validate(json.loads(self.result))
        if path.endswith("/ack"):
            self.before_ack()
        return super().respond(request)

    def detail(self):
        detail = super().detail()
        detail.update(state=self.lifecycle, dispatch={"state": self.dispatch})
        if self.unsent:
            detail.update(state="failed", dispatch={"state": "not_started"}, response={"state": "absent"},
                          problem={"code": "image_unavailable"})
            if self.ready_unsent:
                detail["response"] = {"state": "ready", "ref": _ref(self.result)}
        return detail


@pytest.fixture
def engine(monkeypatch, gateway_factory):
    def bind(wire):
        monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: gateway_factory(wire))
        monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: gateway_factory(wire))
        monkeypatch.setattr("ouroboros.llm_claudexor.read_owned_gateway", lambda: gateway_factory(wire))
        return wire
    return bind


def invoke(ctx, **kwargs):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result("generate_image", {"prompt": "paint", "model": "claudexor::codex=requested-image", **kwargs})


def test_subscription_edit_whole_result_ack_after_durable_retention(engine, ctx):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire())
    original = PNG + b"original metadata"
    (ctx.workspace_root / "input.png").write_bytes(original)
    def retained_before_ack():
        from ouroboros.observability import read_call_payload
        manifest, payload, _ = read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id=f"{wire.operation[0]}_model_response")
        assert payload["result_json_utf8"].encode() == wire.result
        assert manifest["operation_id"] == "op-one"
        ledger = read_usage_records(ctx.budget_drive_root or ctx.drive_root)
        assert ledger[0]["state"] == "settled" and ledger[0]["cost_usd"] is None
        files = task_artifact_dir_path(ctx.drive_root, ctx.task_id)
        assert (files / f"image-response-{wire.operation[0]}.json").read_bytes() == wire.result
        assert (files / f"image-{hashlib.sha256(PNG).hexdigest()}.png").read_bytes() == PNG
    wire.before_ack = retained_before_ack
    result = invoke(ctx, images=[{"root": "active_workspace", "path": "input.png"}],
                    options={"n": 2, "quality": "high"}, credential_profile_id="chosen")
    assert result.status == "ok", result
    sent = json.loads(wire.payload)
    assert sent == {"source": "codex", "model": "requested-image", "account": {"mode": "pin", "profileId": "chosen"},
        "prompt": "paint", "images": [{"mime": "image/png", "sha256": _ref(original)["sha256"],
                                      "dataBase64": base64.b64encode(original).decode()}], "options": {"n": 2, "quality": "high"}}
    facts = _summary(result.text)
    assert facts["generated"] == 2 and facts["route"]["model"] is None
    assert facts["requested_model"] == "requested-image" and facts["actual_cost_usd"] is None
    assert facts["acknowledgement"] == "acknowledged" and not ctx.pending_events
    assert wire.generation_count == 1
    assert all(Path(row["absolute_path"]).read_bytes() == PNG for row in facts["images"])
    assert base64.b64encode(PNG).decode() not in result.text
    ledger = read_usage_records(ctx.drive_root)
    assert len(ledger) == 1 and ledger[0]["cost_usd"] is None


@pytest.mark.parametrize("stage", [("POST", "/v2/model-operations"), ("GET", "/v2/model-operations/op-one/result")])
def test_lost_reply_retains_same_operation_identity_and_never_recreates(engine, ctx, stage):
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire(lose=stage))
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN", result
    assert wire.generation_count == 1
    from ouroboros.observability import read_call_payload
    manifest, _, _ = read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id=f"{wire.operation[0]}_model_request")
    assert manifest["invocation_id"] == wire.operation[0]
    if stage[0] == "GET":
        assert manifest["operation_id"] == "op-one"
    assert read_usage_records(ctx.drive_root)[0]["state"] == "unresolved"
    assert not any(call.url.path.endswith('/ack') for call in wire.calls)


def test_connect_failure_after_accepted_create_is_unknown_not_released(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire()); wire.pending = wire.poll_failure = True
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN", result
    assert _summary(result.text)["operation_id"] == "op-one"
    assert read_usage_records(ctx.drive_root)[0]["state"] == "unresolved"
    assert wire.generation_count == 1


def test_terminal_not_started_releases_attempt(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire()); wire.unsent = True
    assert invoke(ctx).code == "IMAGE_NOT_DISPATCHED"
    assert read_usage_records(ctx.drive_root)[0]["state"] == "released"


def test_artifact_failure_keeps_private_custody_and_can_recover(engine, ctx, monkeypatch):
    wire = engine(ImageWire())
    store = ig.store_task_artifact_bytes
    def fail(*args, **kwargs):
        if kwargs.get("kind") == "image_response":
            raise OSError("disk full")
        return store(*args, **kwargs)
    monkeypatch.setattr(ig, "store_task_artifact_bytes", fail)
    assert invoke(ctx).code == "IMAGE_RESULT_INCOMPLETE"
    assert wire.generation_count == 1 and not any(call.url.path.endswith('/ack') for call in wire.calls)
    monkeypatch.setattr(ig, "store_task_artifact_bytes", store)
    assert "OK: retained" in ig._recover_image(ctx, wire.operation[0])
    assert wire.generation_count == 1 and any(call.url.path.endswith('/ack') for call in wire.calls)


def test_ack_failure_keeps_successful_artifact(engine, ctx):
    wire = engine(ImageWire(lose=("POST", "/v2/model-operations/op-one/ack")))
    result = invoke(ctx)
    assert result.status == "ok" and _summary(result.text)["acknowledgement"] == "pending"
    assert wire.generation_count == 1


def test_image_digest_mismatch_keeps_exact_whole_result(engine, ctx):
    wire = ImageWire(); wire.native["images"][0]["sha256"] = "sha256:" + "0" * 64
    wire.result = _bytes(wire.native); engine(wire)
    assert invoke(ctx).code == "IMAGE_RESULT_INCOMPLETE"
    assert wire.generation_count == 1


def test_budget_refusal_never_uploads_or_creates(engine, ctx):
    from ouroboros import usage_accounting as ua
    wire = engine(ImageWire())
    with ua.usage_scope(ua.UsageScope(drive_root=ctx.drive_root, task_id=ctx.task_id, root_limit_usd=0)), pytest.raises(ua.BudgetExceeded):
        invoke(ctx)
    assert wire.upload_count == wire.generation_count == 0


def test_interrupt_requests_cancel_but_does_not_claim_terminal_cancellation(engine, ctx, monkeypatch):
    from ouroboros import model_wait
    wire = engine(ImageWire()); wire.pending = True
    class Waiter:
        def control_reason(self):
            return "owner_stop" if wire.generation_count else None
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: Waiter() if wire.generation_count else None)
    with pytest.raises(model_wait.ModelWaitInterrupted):
        invoke(ctx)
    assert wire.generation_count == 1
    assert any(call.url.path.endswith('/control') for call in wire.calls)


def test_discovery_preserves_engine_provenance_without_entitlement_claim(engine):
    engine(ImageWire())
    rows, errors = ig._subscription_catalog()
    assert not errors and rows[0]["model"] == "claudexor::codex=gpt-image-2"
    assert rows[0]["evidence"] == CAPABILITY and rows[0]["credential_configured"] is None


@pytest.mark.parametrize("options", [{"n": True}, {"n": 0}, {"n": 11}, {"quality": "max"}, {"output_format": "png"}])
def test_subscription_options_refused_before_upload(engine, ctx, options):
    wire = engine(ImageWire())
    assert invoke(ctx, options=options).code == "TOOL_ARG_ERROR"
    assert wire.upload_count == wire.generation_count == 0


def test_status_poll_reads_same_operation(engine, ctx):
    wire = engine(ImageWire()); wire.pending = True
    result = invoke(ctx)
    assert result.status == "ok"
    assert wire.generation_count == 1
    assert any(call.method == "GET" and call.url.path == "/v2/model-operations/op-one" for call in wire.calls)


def test_explicit_delivery_consumes_generated_artifact_without_regeneration(engine, ctx):
    from types import SimpleNamespace
    from supervisor.events_chat_delivery import _handle_send_photo
    # A real PNG with metadata clears the ordinary photo tool's minimum length.
    import struct
    import zlib
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    pixels = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 2, 2, 8, 2, 0, 0, 0))
              + chunk(b"tEXt", b"Description\0Synthetic generated image")
              + chunk(b"IDAT", zlib.compress(b"\x00" + b"\xff\x00\x00" * 2 + b"\x00" + b"\x00\xff\x00" * 2))
              + chunk(b"IEND", b""))
    wire = ImageWire()
    wire.native["images"] = [{"mime": "image/png", "sizeBytes": len(pixels), "sha256": _ref(pixels)["sha256"],
                              "dataBase64": base64.b64encode(pixels).decode(), "generationId": None}]
    wire.result = _bytes(wire.native)
    engine(wire)
    result = invoke(ctx)
    assert result.status == "ok", result
    image = _summary(result.text)["images"][0]
    assert ctx.pending_events == []
    ctx.current_chat_id = 123
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    sent = registry.execute_result("send_photo", {"file_path": image["absolute_path"], "caption": "Requested image"})
    assert sent.status == "ok", sent
    assert len(ctx.pending_events) == 1
    received = []
    class Bridge:
        def send_photo(self, chat_id, data, **kwargs):
            received.append((chat_id, data, kwargs))
            return True, ""
    _handle_send_photo(ctx.pending_events[0], SimpleNamespace(bridge=Bridge(), append_jsonl=lambda *a, **k: None, DRIVE_ROOT=None))
    assert received[0][0:2] == (123, pixels)
    assert received[0][2]["caption"] == "Requested image" and wire.generation_count == 1


def test_terminal_refusal_preserves_engine_cash_receipt_and_text_quota(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = ImageWire()
    wire.native["cost"].update(cashUsd=0.01, knowledge="exact")
    wire.native.update(outcome="failed", images=[],
                       problem={"code": "provider_failed", "message": "Limited", "retryable": True,
                                "context": {"httpStatus": 429}})
    wire.result = _bytes(wire.native); engine(wire)
    result = invoke(ctx)
    assert result.code == "IMAGE_RATE_LIMITED", result
    assert _summary(result.text)["actual_cost_usd"] == 0.01
    assert read_usage_records(ctx.drive_root)[0]["cost_usd"] == 0.01
    assert wire.generation_count == 1


def test_lost_result_recovery_uses_exact_get_and_original_ledger(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire(lose=("GET", "/v2/model-operations/op-one/result")))
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN"
    attempt = read_usage_records(ctx.drive_root)[0]
    wire.lose = None
    recovered = _summary(ig._recover_image(ctx, attempt["attempt_id"]))
    assert recovered["generated"] == 2 and recovered["recovered"] is True
    rows = read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["attempt_id"] == attempt["attempt_id"]
    assert rows[0]["state"] == "settled" and rows[0]["cost_usd"] is None
    assert wire.generation_count == wire.upload_count == 1
    ig._recover_image(ctx, attempt["attempt_id"])
    assert len(read_usage_records(ctx.drive_root)) == 1 and wire.generation_count == 1


def test_missing_create_identity_cannot_be_recovered_by_resubmission(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = engine(ImageWire(lose=("POST", "/v2/model-operations")))
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN"
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    recovered = registry.execute_result("recover_image", {"request_id": wire.operation[0]})
    assert recovered.code == "IMAGE_OUTCOME_UNKNOWN"
    assert len(read_usage_records(ctx.drive_root)) == wire.generation_count == wire.upload_count == 1


def test_partial_receipt_preserves_warning_good_sibling_and_cost(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = ImageWire()
    wire.native["cost"].update(cashUsd=0.03, knowledge="estimated")
    wire.native.update(outcome="failed", problem={"code": "response_rejected", "message": "Partial output",
        "retryable": False, "context": {"rejectedIndexes": [0], "warnings": ["Quality unavailable"]}})
    wire.native["images"][0]["sha256"] = "sha256:" + "0" * 64
    wire.result = _bytes(wire.native); engine(wire)
    result = invoke(ctx)
    facts = _summary(result.text)
    assert result.code == "IMAGE_RESULT_INCOMPLETE" and facts["generated"] == 1
    assert facts["problem"]["context"]["warnings"] == ["Quality unavailable"]
    assert facts["failures"][0]["index"] == 0 and facts["problem"]["code"] == "response_rejected"
    assert facts["actual_cost_usd"] is None and facts["estimated_cost_usd"] == 0.03
    assert "cost_usd" not in facts and result.meta["artifact_registered"] is True
    row = read_usage_records(ctx.drive_root)[0]
    assert row["state"] == "settled" and row["cost_usd"] == 0.03 and row["cost_final"] is False
    assert wire.generation_count == 1


def test_late_receipt_replaces_administrative_unknown_on_original_budget_drive(engine, ctx, tmp_path):
    from ouroboros import usage_accounting as usage
    from ouroboros.usage_accounting import read_usage_records
    wire = ImageWire(lose=("GET", "/v2/model-operations/op-one/result"))
    wire.native["cost"].update(knowledge="exact", cashUsd=0.07)
    wire.result = _bytes(wire.native); engine(wire)
    ctx.budget_drive_root = tmp_path / "original-budget"
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN"
    row = read_usage_records(ctx.budget_drive_root)[0]
    reservation = usage.AttemptReservation(row["attempt_id"], ctx.budget_drive_root, row["model"], row["provider"], None)
    usage.terminalize_abandoned_attempt(reservation, reason="fixture_owner_terminal", expected_revision=row["revision"])
    wire.lose = None
    recovered = _summary(ig._recover_image(ctx, row["attempt_id"]))
    assert recovered["actual_cost_usd"] == 0.07 and recovered["generated"] == 2
    rows = read_usage_records(ctx.budget_drive_root)
    assert len(rows) == 1 and rows[0]["cost_usd"] == 0.07 and rows[0]["cost_final"] is True
    assert read_usage_records(ctx.drive_root) == []
    assert Path(recovered["images"][0]["absolute_path"]).is_relative_to(ctx.drive_root)
    assert wire.generation_count == wire.upload_count == 1


def test_private_custody_failure_withholds_whole_ack_and_preserves_recovery_id(engine, ctx, monkeypatch):
    wire = engine(ImageWire())
    retain = ig._retain_call
    def fail(ctx, request_id, model, stage, payload, **facts):
        if "result_json_utf8" in payload:
            raise OSError("fixture disk full")
        return retain(ctx, request_id, model, stage, payload, **facts)
    monkeypatch.setattr(ig, "_retain_call", fail)
    result = invoke(ctx)
    assert result.code == "IMAGE_OUTCOME_UNKNOWN", result
    assert _summary(result.text)["operation_id"] == "op-one"
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    monkeypatch.setattr(ig, "_retain_call", retain)
    assert _summary(ig._recover_image(ctx, wire.operation[0]))["generated"] == 2
    assert wire.generation_count == 1


def test_existing_terminal_maintenance_consumes_image_receipt(engine, ctx, monkeypatch):
    from ouroboros import terminal_cost_reconciliation
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_accounting import read_usage_records
    from supervisor import queue
    wire = engine(ImageWire(lose=("GET", "/v2/model-operations/op-one/result")))
    assert invoke(ctx).code == "IMAGE_OUTCOME_UNKNOWN"
    original = read_usage_records(ctx.drive_root)[0]
    wire.lose = None
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **k: False)
    write_task_result(ctx.drive_root, ctx.task_id, "cancelled", result="Fixture stopped", budget_drive_root=str(ctx.drive_root))
    terminal_cost_reconciliation.reconcile_abandoned_usage(ctx.drive_root)
    rows = read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["attempt_id"] == original["attempt_id"]
    assert rows[0]["state"] == "settled" and rows[0]["cost_usd"] is None
    assert rows[0]["cost_final"] is False and wire.generation_count == 1
    # Maintenance reconciles money; only the image consumer takes ordinary custody.
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    ig._recover_image(ctx, original["attempt_id"])
    assert any(call.url.path.endswith("/ack") for call in wire.calls)


@pytest.mark.parametrize("lifecycle,outcome,dispatch", [
    ("failed", "failed", "response_received"),
    ("cancelled", "unknown", "unknown"),
    ("interrupted", "unknown", "started"),
])
def test_every_terminal_ready_receipt_retains_good_siblings_and_diagnostics(engine, ctx, lifecycle, outcome, dispatch):
    from ouroboros.usage_accounting import read_usage_records
    wire = ImageWire()
    wire.lifecycle, wire.dispatch = lifecycle, dispatch
    wire.native.update(outcome=outcome, problem={"code": "response_rejected", "message": "Partial output",
                                               "retryable": False, "context": {"rejectedIndexes": [1, 3]}})
    wire.result = _bytes(wire.native); engine(wire)
    result = invoke(ctx)
    facts = _summary(result.text)
    assert result.code == "IMAGE_RESULT_INCOMPLETE" and facts["generated"] == 2
    assert facts["problem"]["context"]["rejectedIndexes"] == [1, 3]
    assert facts["outcome"] == outcome and facts["actual_cost_usd"] is None
    assert facts["acknowledgement"] == "acknowledged"
    assert Path(facts["images"][0]["absolute_path"]).read_bytes() == PNG
    rows = read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["cost_usd"] is None and not rows[0]["cost_final"]
    assert wire.generation_count == 1 and not ctx.pending_events


def test_not_started_ready_refusal_is_retrieved_before_release_and_ack(engine, ctx):
    from ouroboros.usage_accounting import read_usage_records
    wire = ImageWire(); wire.unsent = wire.ready_unsent = True
    wire.native.update(outcome="failed", images=[], problem={"code": "quota_unavailable", "message": "Image quota",
                                                           "retryable": True, "context": {"requestKind": "image"}})
    wire.result = _bytes(wire.native); engine(wire)
    wire.before_ack = lambda: _assert_released(ctx)
    def _assert_released(ctx):
        assert read_usage_records(ctx.drive_root)[0]["state"] == "released"
    result = invoke(ctx)
    assert result.code == "IMAGE_NOT_DISPATCHED" and _summary(result.text)["response"]
    assert any(call.url.path.endswith("/result") for call in wire.calls)
    assert any(call.url.path.endswith("/ack") for call in wire.calls)
    assert not any("quota" in call.url.path or "catalog" in call.url.path for call in wire.calls)


def test_disclosed_five_edit_limit_is_checked_before_upload(engine, ctx):
    wire = engine(ImageWire())
    (ctx.workspace_root / "input.png").write_bytes(PNG)
    refs = [{"root": "active_workspace", "path": "input.png"}] * 6
    assert invoke(ctx, images=refs).code == "TOOL_ARG_ERROR"
    assert wire.generation_count == wire.upload_count == 0


def test_one_image_storage_failure_withholds_ack_until_exact_recovery(engine, ctx, monkeypatch):
    wire = engine(ImageWire())
    store = ig.store_task_artifact_bytes
    def fail(*args, **kwargs):
        if kwargs.get("kind") == "generated_image":
            raise OSError("fixture full disk")
        return store(*args, **kwargs)
    monkeypatch.setattr(ig, "store_task_artifact_bytes", fail)
    result = invoke(ctx)
    assert result.code == "IMAGE_RESULT_INCOMPLETE" and _summary(result.text)["generated"] == 0
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    monkeypatch.setattr(ig, "store_task_artifact_bytes", store)
    assert _summary(ig._recover_image(ctx, wire.operation[0]))["acknowledgement"] == "acknowledged"
    assert wire.generation_count == wire.upload_count == 1


@pytest.mark.parametrize("name,digest", [
    ("ImageCallRequest", "898c4fcbe295fb0eca6ea6a149151a397ce044701cb8e11d4d4d13dd90798e75"),
    ("ImageCallResult", "76751e0c7024baf7d4a5a498b0cdd7f0bd6a230d7c1f5764eba6992a37da9f22"),
])
def test_generated_schema_fixtures_match_published_engine_contract(name, digest):
    path = Path(__file__).parent / "fixtures" / "claudexor_image_contract" / f"{name}.schema.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    Draft7Validator.check_schema(SCHEMAS[name])


def test_explicit_count_ten_and_auto_account_use_published_request_schema(engine, ctx):
    wire = engine(ImageWire())
    assert invoke(ctx, options={"n": 10}).status == "ok"
    sent = json.loads(wire.payload)
    assert sent["options"] == {"n": 10} and sent["account"] == {"mode": "auto"}
    upload = next(call for call in wire.calls if call.method == "POST" and call.url.path == "/v2/uploads")
    assert json.loads(upload.content)["purpose"] == "model"


def test_accounting_failure_retains_outputs_but_withholds_ack_until_reconciled(engine, ctx, monkeypatch):
    from ouroboros import usage_accounting as usage
    wire = engine(ImageWire())
    settle = usage.settle_attempt
    def fail(*args, **kwargs):
        raise usage.UsageAccountingError("fixture ledger unavailable")
    monkeypatch.setattr(usage, "settle_attempt", fail)
    result = invoke(ctx)
    assert result.code == "IMAGE_RESULT_INCOMPLETE" and _summary(result.text)["generated"] == 2
    assert not any(call.url.path.endswith("/ack") for call in wire.calls)
    monkeypatch.setattr(usage, "settle_attempt", settle)
    recovered = _summary(ig._recover_image(ctx, wire.operation[0]))
    assert recovered["acknowledgement"] == "acknowledged"
    assert usage.read_usage_records(ctx.drive_root)[0]["state"] == "settled"
    assert wire.upload_count == wire.generation_count == 1


@pytest.mark.parametrize("receipt", ["settled", "released"])
@pytest.mark.parametrize("racing_state", ["unresolved", "abandoned", "terminal"])
def test_recovery_cas_loser_requires_durable_accounting_before_whole_ack(
        engine, ctx, tmp_path, monkeypatch, receipt, racing_state):
    import threading

    from ouroboros import usage_accounting as usage, usage_store
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.observability import read_call_payload
    from ouroboros.transport_custody import ProviderNotDispatched, release_pre_dispatch_attempt
    from ouroboros.usage_ledger import is_abandoned_settlement

    wire = ImageWire(lose=("GET", "/v2/model-operations/op-one/result"))
    if receipt == "released":
        wire.unsent = wire.ready_unsent = True
        wire.native.update(outcome="failed", images=[], problem={
            "code": "quota_unavailable", "message": "Image quota", "retryable": True,
            "context": {"requestKind": "image"}})
        wire.result = _bytes(wire.native)
    engine(wire)
    ctx.budget_drive_root = tmp_path / "original-budget"

    def original_row():
        with usage_store.read(ctx.budget_drive_root) as txn:
            return txn.attempt(wire.operation[0])

    transition = usage._transition
    raced, transitions, race_errors, late_write = [], [], [], []
    worker_waiting, release_worker, late_write_done, parked = (threading.Event() for _ in range(4))
    outcome, ack_rows = {}, []

    def checked(ok, message):
        # The product's accounting may catch an assertion raised here; the test body reads every miss (#1937).
        if not ok:
            race_errors.append(message)
            raise AssertionError(message)

    def late_worker_then_transition(reservation, state, **fields):
        transitions.append((state, threading.current_thread().name))
        if state == "unresolved":
            worker_waiting.set()
            parked.set()
            checked(release_worker.wait(5), "recovery did not release the late worker")
            try:
                result = transition(reservation, state, **fields)
                late_write.append("written")  # only once the real write returned
                return result
            except BaseException as exc:
                late_write.append(exc)
                raise
            finally:
                late_write_done.set()
        # Real ledger writes deterministically occupy the gap between recovery's
        # initial indexed read and its revision-CAS settlement/release.
        if not raced and fields.get("_expected_revision") is not None:
            raced.append(fields["_expected_revision"])
            release_worker.set()
            checked(late_write_done.wait(5) and late_write == ["written"],
                    f"late worker did not update the ledger: {late_write!r}")
            if racing_state == "abandoned":
                usage.terminalize_abandoned_attempt(reservation, reason="late_owner_terminal")
            elif racing_state == "terminal" and receipt == "released":
                checked(release_pre_dispatch_attempt(reservation, ProviderNotDispatched("never started")),
                        "the racing release did not commit")
            elif racing_state == "terminal":
                usage.settle_attempt(reservation, {"input_tokens": 10, "output_tokens": 20})
            checked(original_row()["revision"] > fields["_expected_revision"], "recovery's CAS was not raced")
        return transition(reservation, state, **fields)

    def run_worker():
        try:
            outcome["result"] = invoke(ctx)
        except BaseException as exc:  # an early failure is named at once, not after the startup wait
            outcome["error"] = exc
        finally:
            parked.set()

    def no_new_attempt_or_history(*args, **kwargs):
        pytest.fail("recovery must not reserve an attempt or scan usage history")

    def durable_before_ack():
        ack_rows.append(original_row())

    wire.before_ack = durable_before_ack
    # The stores' lifecycle import and the registry's imports precede the race window.
    for root in (ctx.drive_root, ctx.budget_drive_root):
        usage_store.migrate_from_journal(root)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr(usage, "_transition", late_worker_then_transition)
    worker = threading.Thread(target=run_worker, name="late-image-worker")
    worker.start()
    try:
        assert parked.wait(30) and worker_waiting.is_set(), f"late update missing: {transitions!r}; worker={outcome!r}"
        request_id = wire.operation[0]
        before = original_row()
        assert before["state"] == "dispatched"
        # The original sender is paused before its real unresolved write, not
        # replaced by a fake settlement API or a manually fabricated ledger row.
        monkeypatch.setattr(usage, "reserve_attempt", no_new_attempt_or_history)
        attempts = usage_store.Txn.attempts
        monkeypatch.setattr(usage_store.Txn, "attempts", no_new_attempt_or_history)
        monkeypatch.setattr(usage_store.Txn, "attempt_rows", no_new_attempt_or_history)
        monkeypatch.setattr(usage_store.Txn, "open_attempts", no_new_attempt_or_history)
        result = registry.execute_result("recover_image", {"request_id": request_id})
    finally:
        release_worker.set()
        worker.join(5)
    assert not worker.is_alive() and "error" not in outcome and not race_errors, (outcome, race_errors)
    assert outcome["result"].code == "IMAGE_OUTCOME_UNKNOWN"
    facts = _summary(result.text)
    assert raced == [before["revision"]]
    assert facts["recovered"] is True and facts["operation_id"] == "op-one"
    if receipt == "settled":
        assert facts["generated"] == 2
    assert (task_artifact_dir_path(ctx.drive_root, ctx.task_id) / facts["response"]["path"]).read_bytes() == wire.result
    manifest, payload, _ = read_call_payload(ctx.budget_drive_root, task_id=ctx.task_id,
                                            call_id=f"{request_id}_model_response")
    assert payload["result_json_utf8"].encode() == wire.result and manifest["operation_id"] == "op-one"
    assert all(Path(image["absolute_path"]).read_bytes() == PNG for image in facts.get("images", []))
    if racing_state == "terminal":
        assert facts["acknowledgement"] == "acknowledged"
    else:
        assert original_row()["state"] == ("settled" if racing_state == "abandoned" else "unresolved")
        assert is_abandoned_settlement(original_row()) == (racing_state == "abandoned")
        assert facts["acknowledgement"] == "pending" and facts["warnings"]
        assert "accounting" in " ".join(facts["warnings"]).lower()
        assert not any(call.url.path.endswith("/ack") for call in wire.calls)
        assert result.code == ("IMAGE_RESULT_INCOMPLETE" if receipt == "settled" else "IMAGE_NOT_DISPATCHED")
        # A later explicit recovery reuses this attempt/receipt and can now ACK.
        retry = registry.execute_result("recover_image", {"request_id": request_id})
        assert _summary(retry.text)["acknowledgement"] == "acknowledged"
    assert original_row()["state"] == receipt and not is_abandoned_settlement(original_row())
    assert wire.generation_count == wire.upload_count == 1
    assert sum(call.url.path.endswith("/ack") for call in wire.calls) == 1
    assert len(ack_rows) == 1 and ack_rows[0]["state"] == receipt and not is_abandoned_settlement(ack_rows[0])
    assert sum(call.method == "POST" and call.url.path == "/v2/model-operations" for call in wire.calls) == 1
    monkeypatch.setattr(usage_store.Txn, "attempts", attempts)
    assert len(usage.read_usage_records(ctx.budget_drive_root)) == 1
    assert usage.read_usage_records(ctx.drive_root) == []
