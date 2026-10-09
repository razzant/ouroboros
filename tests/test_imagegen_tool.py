"""Tests for the generate_image tool and its Claudexor client family."""

from __future__ import annotations

import base64
import json

import pytest

import ouroboros.tools.imagegen as ig
from ouroboros.usage_accounting import BudgetExceeded
from ouroboros.tools.imagegen import _generate_image, _sniff_mime
from ouroboros.gateways import claudexor_images
from ouroboros.gateways.claudexor_images import image_operation_supported


def _png_bytes(size=64):
    return b"\x89PNG\r\n\x1a\n" + b"0" * size


class _FakeGateway:
    def __init__(self, *, supported=True, result_body=None, fail_with=None, detail_sequence=None):
        self.supported = supported
        self.result_body = result_body or {}
        self.fail_with = fail_with
        self.detail_sequence = list(detail_sequence or [])
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True

    def operations(self):
        ops = []
        if self.supported:
            ops.append({"method": "POST", "path": claudexor_images.IMAGE_OPERATION_PATH, "parameters": []})
        return ops

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs.get("json_body"), kwargs.get("headers")))
        if self.fail_with is not None and method == "POST":
            raise self.fail_with
        if path.endswith("/result"):
            return self.result_body
        if method == "GET" and path.count("/") == 3:
            if self.detail_sequence:
                return self.detail_sequence.pop(0)
            return {"state": "succeeded"}
        return {"id": "img-op-1", "state": "queued"}

    # Compatibility with the module-level functions' gateway contract:
    # they call gateway._request directly, so _request above IS the seam.


class _Ctx:
    def __init__(self, tmp_path):
        self.task_id = "test-imagegen"
        self.drive_root = tmp_path
        self.budget_drive_root = tmp_path
        self.task_metadata = {}
        self.event_queue = None
        self.root_task_id = "test-imagegen"
        self.current_chat_id = 123
        self.repo_dir = str(tmp_path)
        self._sidecar = None
        self._current_review_tool_name = ""
        self.pending_events = []


def _patch_gateway(monkeypatch, gateway):
    monkeypatch.setattr(ig, "_gateway_for", lambda ctx: gateway)


def _patch_accounting(monkeypatch, captured):
    def fake_execute(request, send, **kwargs):
        captured.append({"request": request})
        return send()

    monkeypatch.setattr(ig, "execute_physical_attempt", fake_execute)


def _last_code(monkeypatch, ctx, call):
    captured_codes = []
    orig = ig._publish_tool_result

    def spy(c, result):
        captured_codes.append(result.code)
        return orig(c, result)

    monkeypatch.setattr(ig, "_publish_tool_result", spy)
    out = call()
    monkeypatch.setattr(ig, "_publish_tool_result", orig)
    return out, captured_codes[-1] if captured_codes else ""


class TestSniffMime:
    def test_png_jpeg_webp(self):
        assert _sniff_mime(_png_bytes()) == "image/png"
        assert _sniff_mime(b"\xff\xd8\xff" + b"0" * 8) == "image/jpeg"
        assert _sniff_mime(b"RIFF" + b"0" * 4 + b"WEBP") == "image/webp"
        assert _sniff_mime(b"nonsense") == ""


class TestImageOperationSupported:
    def test_present_and_absent(self):
        assert image_operation_supported([{"method": "POST", "path": claudexor_images.IMAGE_OPERATION_PATH}])
        assert not image_operation_supported([{"method": "GET", "path": "/v2/runs"}])
        assert not image_operation_supported([])


class TestGenerateImageTool:
    def test_typed_refusal_when_engine_lacks_route(self, monkeypatch, tmp_path):
        gw = _FakeGateway(supported=False)
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "CAPABILITY_UNAVAILABLE"
        assert captured == []  # no paid attempt before the capability check
        assert gw.closed  # attach-only gateway is ours to close even on refusal

    def test_typed_refusal_when_daemon_absent(self, monkeypatch, tmp_path):
        def _absent(ctx):
            raise ig._DaemonAbsent("daemon_not_discovered")

        monkeypatch.setattr(ig, "_gateway_for", _absent)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "CAPABILITY_UNAVAILABLE"

    def test_argument_validation(self, monkeypatch, tmp_path):
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, ""))
        assert code == "TOOL_ARG_ERROR"
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "x" * 32001))
        assert code == "TOOL_ARG_ERROR"
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "ok", n=0))
        assert code == "TOOL_ARG_ERROR"
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "ok", quality="non"))
        assert code == "TOOL_ARG_ERROR"

    def test_edit_inputs_keep_read_file_policy_and_preserve_in_root_edits(self, monkeypatch, tmp_path):
        png = _png_bytes(30)
        allowed = tmp_path / 'allowed.png'
        denied = tmp_path / 'denied.png'
        allowed.write_bytes(png)
        denied.write_bytes(png)
        gw = _FakeGateway(result_body={'data': [{'b64_json': base64.b64encode(png).decode()}]})
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        record = {'id': 'image_edit_black_box', 'role': 'black_box_reference',
                  'paths': [str(denied)], 'allow': ['execute']}
        ctx.task_contract = {'resource_policy': {'protected_artifacts': [record]}}
        ctx.task_metadata = {'task_contract': ctx.task_contract}

        _out, code = _last_code(monkeypatch, ctx,
                                lambda: _generate_image(ctx, 'edit', image_paths=[str(denied)], send=False))
        assert code == 'ACCESS_BLOCKED'
        assert not captured and not gw.calls  # refuse before upload or paid dispatch

        out = _generate_image(ctx, 'edit', image_paths=[str(allowed)], send=False)
        assert out.startswith('OK:')
        creates = [call for call in gw.calls if call[0] == 'POST' and call[1] == claudexor_images.IMAGE_OPERATION_PATH]
        assert len(creates) == 1
        assert creates[0][2]['images'][0]['dataUrl'] == 'data:image/png;base64,' + base64.b64encode(png).decode()

    def test_budget_refusal_is_not_a_dispatched_unknown_image(self, monkeypatch, tmp_path):
        _patch_gateway(monkeypatch, _FakeGateway())
        ctx = _Ctx(tmp_path)
        def refuse(_request, _send):
            raise BudgetExceeded('root budget exhausted')
        monkeypatch.setattr(ig, 'execute_physical_attempt', refuse)
        with pytest.raises(BudgetExceeded):
            _generate_image(ctx, 'castle', send=False)
        assert not (tmp_path / 'logs' / 'events.jsonl').exists()

    def test_one_attempt_no_retry_and_no_base64_in_result(self, monkeypatch, tmp_path):
        png = _png_bytes(256)
        body = {"data": [{"b64_json": base64.b64encode(png).decode()}], "usage": {"input_tokens": 10}}
        gw = _FakeGateway(result_body=body)
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out = _generate_image(ctx, "a castle", send=True)
        assert out.startswith("OK:") or "OK" in out
        creates = [c for c in gw.calls if c[0] == "POST" and c[1] == claudexor_images.IMAGE_OPERATION_PATH]
        assert len(creates) == 1  # ONE attempt, no retry
        summary = json.loads(out.split("\n", 1)[1])
        img = summary["images"][0]
        assert img["mime"] == "image/png" and img["size"] == len(png)
        assert set(img.keys()) == {"path", "sha256", "mime", "size"}
        assert "b64_json" not in json.dumps(summary)  # base64 never in tool text
        import pathlib
        assert pathlib.Path(img["path"]).exists()
        assert summary["delivery"] == {"requested": True, "submitted": 1, "failed_indices": []}
        assert len(ctx.pending_events) == 1
        assert ctx.pending_events[0]["type"] == "send_photo"
        assert ctx.pending_events[0]["chat_id"] == 123
        # The delivery event carries bytes; the model-visible result never does.

    def test_all_images_submitted_to_the_same_chat(self, monkeypatch, tmp_path):
        png = _png_bytes(128)
        b64 = base64.b64encode(png).decode()
        body = {"data": [{"b64_json": b64}, {"b64_json": b64}], "usage": {}}
        gw = _FakeGateway(result_body=body)
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out = _generate_image(ctx, "two castles", n=2, caption="castle", send=True)
        summary = json.loads(out.split("\n", 1)[1])
        assert summary["generated"] == 2
        assert summary["delivery"] == {"requested": True, "submitted": 2, "failed_indices": []}
        assert [ev["type"] for ev in ctx.pending_events] == ["send_photo", "send_photo"]
        assert [ev["chat_id"] for ev in ctx.pending_events] == [123, 123]
        assert [ev["caption"] for ev in ctx.pending_events] == ["castle", ""]
        # Drive the actual supervisor consumer without a Project binding. Without
        # chat_id this handler returns silently and no owner sees the images.
        from types import SimpleNamespace

        from supervisor import events_chat_delivery as delivery

        seen = []
        bridge = SimpleNamespace(send_photo=lambda chat_id, raw, **kw: (seen.append((chat_id, raw, kw)) or (True, "ok")))
        monkeypatch.setattr(delivery, "_bound_project_chat_id", lambda *_a: None)
        host = SimpleNamespace(bridge=bridge, DRIVE_ROOT=tmp_path)
        for event in ctx.pending_events:
            delivery._handle_send_photo(event, host)
        assert [item[0] for item in seen] == [123, 123]
        assert [item[1] for item in seen] == [png, png]

    def test_large_image_remains_artifact_without_a_false_delivery_claim(self, monkeypatch, tmp_path):
        png = _png_bytes(128)
        body = {"data": [{"b64_json": base64.b64encode(png).decode()}]}
        _patch_gateway(monkeypatch, _FakeGateway(result_body=body))
        _patch_accounting(monkeypatch, [])
        monkeypatch.setattr(ig, "_PHOTO_INLINE_CAP", 64)
        ctx = _Ctx(tmp_path)
        out = _generate_image(ctx, "large castle", send=True)
        summary = json.loads(out.split("\n", 1)[1])
        assert summary["delivery"] == {"requested": True, "submitted": 0, "failed_indices": [1]}
        assert not ctx.pending_events
        from pathlib import Path
        assert Path(summary["images"][0]["path"]).read_bytes() == png

    def test_send_failure_keeps_artifact_and_names_unsent_index(self, monkeypatch, tmp_path):
        png = _png_bytes(128)
        body = {"data": [{"b64_json": base64.b64encode(png).decode()}]}
        _patch_gateway(monkeypatch, _FakeGateway(result_body=body))
        _patch_accounting(monkeypatch, [])
        ctx = _Ctx(tmp_path)
        ctx.current_chat_id = None
        out = _generate_image(ctx, "unrouted castle", send=True)
        summary = json.loads(out.split("\n", 1)[1])
        assert summary["generated"] == 1
        assert summary["delivery"] == {"requested": True, "submitted": 0, "failed_indices": [1]}
        assert not ctx.pending_events

    def test_image_429_typed_refusal(self, monkeypatch, tmp_path):
        class Err(Exception):
            code = "image_generation_limit_reached"
            reset_at = "2026-10-08T12:00:00Z"

        gw = _FakeGateway(fail_with=Err("limit"))
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "IMAGE_RATE_LIMITED"
        assert "NOT parked" in out
        assert "2026-10-08T12:00:00Z" in out
        assert len(captured) == 1  # the attempt was made and accounted

    def test_image_429_typed_refusal_from_engine_detail_problem(self, monkeypatch, tmp_path):
        """The real engine shape: HTTP 202 → poll → terminal detail with
        state='failed' and the typed ControlProblem. The client must classify
        from problem.code/context.resetsAt — the detail has NO error/reason
        fields, so the old reader misclassified this as OUTCOME_UNKNOWN."""
        detail = {
            "id": "img-op-1",
            "state": "failed",
            "problem": {
                "code": "image_generation_limit_reached",
                "message": "Image limit reached; text allowance unchanged",
                "retryable": False,
                "context": {"resetsAt": "2026-10-09T00:00:00Z"},
            },
        }
        gw = _FakeGateway(detail_sequence=[detail])
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "IMAGE_RATE_LIMITED"
        assert "NOT parked" in out
        assert "2026-10-09T00:00:00Z" in out
        assert len(captured) == 1
        # A KNOWN failed terminal is not an unknown dispatch: no
        # image_outcome_unknown row may be written.
        assert not (tmp_path / "logs" / "events.jsonl").exists()

    def test_engine_interrupted_is_outcome_unknown_and_never_retried(self, monkeypatch, tmp_path):
        """Engine dispatch-unknown: state='interrupted'. The tool must surface
        IMAGE_OUTCOME_UNKNOWN with the durable row, and never POST twice."""
        detail = {"id": "img-op-1", "state": "interrupted", "problem": None}
        gw = _FakeGateway(detail_sequence=[detail])
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "IMAGE_OUTCOME_UNKNOWN"
        creates = [c for c in gw.calls if c[0] == "POST"]
        assert len(creates) == 1  # never retried
        events = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert "image_outcome_unknown" in events
        assert "engine_interrupted" in events

    def test_engine_failed_terminal_with_other_problem_is_image_error(self, monkeypatch, tmp_path):
        """A KNOWN failed terminal with a non-quota problem code is a typed
        IMAGE_ERROR carrying the engine's code — not OUTCOME_UNKNOWN."""
        detail = {
            "id": "img-op-1",
            "state": "failed",
            "problem": {"code": "image_upstream_error", "message": "provider refused",
                        "retryable": False, "context": {}},
        }
        gw = _FakeGateway(detail_sequence=[detail])
        _patch_gateway(monkeypatch, gw)
        _patch_accounting(monkeypatch, [])
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "IMAGE_ERROR"
        assert "image_upstream_error" in out
        assert "provider refused" in out
        assert not (tmp_path / "logs" / "events.jsonl").exists()

    def test_unknown_outcome_recorded_not_retried(self, monkeypatch, tmp_path):
        gw = _FakeGateway(fail_with=TimeoutError("image_operation_timeout:img-op-1"))
        _patch_gateway(monkeypatch, gw)
        captured = []
        _patch_accounting(monkeypatch, captured)
        ctx = _Ctx(tmp_path)
        out, code = _last_code(monkeypatch, ctx, lambda: _generate_image(ctx, "a castle"))
        assert code == "IMAGE_OUTCOME_UNKNOWN"
        creates = [c for c in gw.calls if c[0] == "POST"]
        assert len(creates) == 1  # never retried
        events = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert "image_outcome_unknown" in events


def test_registry_edit_upload_respects_execute_only_artifact_and_still_allows_plain_image(monkeypatch, tmp_path):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, data = tmp_path / 'workspace', tmp_path / 'data'
    repo.mkdir()
    data.mkdir()
    protected, ordinary = repo / 'protected.png', repo / 'ordinary.png'
    protected.write_bytes(_png_bytes(16))
    ordinary.write_bytes(_png_bytes(16))
    contract = build_task_contract({'resource_policy': {'protected_artifacts': [{
        'id': 'black_box_image', 'role': 'black_box_reference',
        'paths': [str(protected)], 'allow': ['execute'],
    }]}})
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id='image-test',
                                     task_contract=contract, task_metadata={'task_contract': contract}))
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *a, **k: (True, ''))
    png = _png_bytes(16)
    gateway = _FakeGateway(result_body={'data': [{'b64_json': base64.b64encode(png).decode()}]})
    _patch_gateway(monkeypatch, gateway)
    _patch_accounting(monkeypatch, [])
    forbidden = registry.execute_result('generate_image', {'prompt': 'edit', 'image_paths': [str(protected)], 'send': False})
    assert forbidden.code == 'ACCESS_BLOCKED'
    assert not gateway.calls
    accepted = registry.execute_result('generate_image', {'prompt': 'edit', 'image_paths': [ordinary.name], 'send': False})
    assert accepted.code == 'OK'
    assert len([call for call in gateway.calls if call[0] == 'POST' and call[1] == claudexor_images.IMAGE_OPERATION_PATH]) == 1


class TestClientFamily:
    """Client-family shape checks against the real module seam."""

    def test_create_image_operation_body_shape(self, monkeypatch):
        """B1 regression: the request body is the image request itself, NOT a
        model payload-ref; edit inputs carry their sniffed MIME (M3)."""
        seen = []

        class GW:
            def _request(self, method, path, **kwargs):
                seen.append((method, path, kwargs))
                return {"operationId": "op1"}

        claudexor_images.create_image_operation(
            GW(), {"model": "gpt-image-2", "prompt": "hi"},
            images=[(_png_bytes(16), "image/png")],
            idempotency_key="k-1",
        )
        method, path, kwargs = seen[0]
        assert method == "POST" and path == claudexor_images.IMAGE_OPERATION_PATH
        body = kwargs["json_body"]
        assert body["request"] == {"model": "gpt-image-2", "prompt": "hi"}  # plain request, no ref contract
        assert body["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        assert kwargs["headers"]["Idempotency-Key"] == "k-1"

    def test_ack_carries_retained_digest(self, monkeypatch):
        seen = []

        class GW:
            def _request(self, method, path, **kwargs):
                seen.append((method, path, kwargs))
                return {}

        claudexor_images.acknowledge_image_result(GW(), "op1", "abc123")
        method, path, kwargs = seen[0]
        assert path.endswith("/op1/ack") and kwargs["json_body"] == {"sha256": "abc123"}

    def test_module_functions_reachable(self):
        for name in ("create_image_operation", "get_image_operation",
                     "get_image_result", "acknowledge_image_result",
                     "image_operation_supported"):
            assert callable(getattr(claudexor_images, name)), name
