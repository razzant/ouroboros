"""Image dispatch authority and received-response accounting through real consumers."""
from pathlib import Path

import httpx
import pytest

from ouroboros import usage_accounting as usage, usage_store
from ouroboros.tools.registry import ToolRegistry
from tests.test_consciousness_authority import _wake_task
from tests.test_imagegen_tool import (
    BODY, PNG, _summary, _transport, configured as configured, ctx as ctx,
)


@pytest.mark.parametrize("accounting_pending", [False, True])
@pytest.mark.parametrize("provider_warning", [False, True])
def test_direct_image_discloses_pending_accounting_and_recovers_same_attempt(
        ctx, monkeypatch, accounting_pending, provider_warning):
    calls = []
    warnings = ["provider fixture warning"] if provider_warning else []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={**BODY, "usage": {"cost": 0.42}, "warnings": warnings})

    _transport(monkeypatch, provider)
    settle = usage.settle_attempt

    def unavailable(*args, **kwargs):
        raise usage.UsageAccountingError("synthetic direct-image settlement unavailable")

    if accounting_pending:
        monkeypatch.setattr(usage, "settle_attempt", unavailable)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    initial = registry.execute_result("generate_image", {"prompt": "fixture", "model": "openai::fixture-image"})
    facts = _summary(initial.text)
    request_id = facts["request_id"]
    with usage_store.read(ctx.drive_root) as txn:
        before = dict(txn.attempt(request_id))
    assert before["state"] == ("unresolved" if accounting_pending else "settled")
    assert facts["actual_cost_usd"] == 0.42 and facts["cost_final"] is True
    assert facts["generated"] == 1 and Path(facts["images"][0]["absolute_path"]).read_bytes() == PNG
    assert len(calls) == 1 and not ctx.pending_events

    # Exercise recovery even on the original RED: the hint must name a usable
    # operation on this very attempt, without another paid generation.
    monkeypatch.setattr(usage, "settle_attempt", settle)
    recovered = registry.execute_result("recover_image", {"request_id": request_id})
    recovered_facts = _summary(recovered.text)
    with usage_store.read(ctx.drive_root) as txn:
        after = dict(txn.attempt(request_id))
    assert after["state"] == "settled" and after["cost_usd"] == 0.42
    assert recovered_facts["request_id"] == request_id and recovered_facts["images"] == facts["images"]
    assert len(calls) == 1 and not ctx.pending_events
    assert recovered.code == ("IMAGE_RESULT_INCOMPLETE" if provider_warning else "OK")
    assert recovered_facts["warnings"] == warnings
    assert initial.code == ("IMAGE_RESULT_INCOMPLETE" if accounting_pending or provider_warning else "OK")
    assert all(warning in (facts["warnings"] or []) for warning in warnings)
    if accounting_pending:
        assert "post_response_accounting_failed" in str(before)
        assert "accounting" in " ".join(facts["warnings"]).lower()
        assert "recover_image" in " ".join(facts["warnings"])
    else:
        assert facts["warnings"] == warnings


@pytest.mark.parametrize("level", ["observe", "act", "normal"])
@pytest.mark.parametrize("tool", ["generate_image", "recover_image"])
def test_image_world_writes_follow_observe_dispatch_authority(ctx, monkeypatch, level, tool):
    calls = []

    def provider(request):
        calls.append(request)
        if request.method == "GET":
            assert request.url.path == "/api/v1/images/models"
            return httpx.Response(200, json={"data": [{"id": "fixture-image"}]})
        assert request.method == "POST" and request.url.path == "/v1/images/generations"
        return httpx.Response(200, json={**BODY, "usage": {"cost": 0.42}})

    _transport(monkeypatch, provider)
    ctx.is_direct_chat = True
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    arguments = {"prompt": "fixture", "model": "openai::fixture-image"}
    missing_image = None
    if tool == "recover_image":
        initial = registry.execute_result("generate_image", arguments)
        assert initial.code == "OK"
        facts = _summary(initial.text)
        arguments = {"request_id": facts["request_id"]}
        missing_image = Path(facts["images"][0]["absolute_path"])
        missing_image.unlink()  # recovery really has an artifact to restore

    ctx.task_metadata = _wake_task(level)["metadata"] if level != "normal" else {}
    registry.set_context(ctx)
    # Observe restrictions bind at dispatch, not by hiding schemas. Read-only
    # discovery remains usable, including its real fixture catalog transport.
    for name in ("generate_image", "recover_image", "list_image_models"):
        assert registry.get_schema_by_name(name) is not None
    catalog = registry.execute_result("list_image_models", {"model": "openrouter::fixture-image"})
    assert catalog.code == "OK"
    assert _summary(catalog.text)["models"][0]["model"] == "openrouter::fixture-image"
    assert len([call for call in calls if call.method == "GET"]) == 1
    before_posts = len([call for call in calls if call.method == "POST"])

    result = registry.execute_result(tool, arguments)
    after_posts = len([call for call in calls if call.method == "POST"])
    if level == "observe":
        assert result.status == "blocked" and result.code == "RESOURCE_CONSTRAINT_BLOCKED", result
        assert after_posts == before_posts
        assert missing_image is None or not missing_image.exists()
    else:
        assert result.code == "OK", result
        facts = _summary(result.text)
        assert Path(facts["images"][0]["absolute_path"]).read_bytes() == PNG
        assert after_posts == before_posts + int(tool == "generate_image")
        if tool == "recover_image":
            assert facts["request_id"] == arguments["request_id"]
    assert not ctx.pending_events
