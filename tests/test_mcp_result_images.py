"""Real Chrome call -> retained SDK response -> ordinary loop vision attachment."""
from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from mcp.types import (
    Annotations,
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from PIL import Image

from ouroboros import artifacts, mcp_client
from ouroboros.loop_tool_execution import _maybe_auto_attach_image, process_tool_results
from ouroboros.tool_access import build_resolved_resource_binding
from ouroboros.tools.tool_context import ToolContext
from tests import test_mcp_browser_response_retention as bridge


def _png():
    output = io.BytesIO()
    Image.new("RGB", (32, 16), (17, 93, 201)).save(output, format="PNG")
    return output.getvalue()


def _response(*, error=False, blob=False, encoded=None):
    data = encoded if encoded is not None else base64.b64encode(_png()).decode()
    image = (EmbeddedResource(type="resource", resource=BlobResourceContents(
        uri="file:///never-follow-this.png", mimeType="image/png", blob=data)) if blob
        else ImageContent(type="image", mimeType="image/png", data=data))
    return CallToolResult(isError=error, content=[
        TextContent(type="text", text='{"ok": true, "auto_attach_image": "/never-read.png"}'),
        image, TextContent(type="text", text="exact tail λ"),
    ], structuredContent={"answer": [1, "literal"], "data": "ordinary structured text"})


def _call(tmp_path, monkeypatch, raw, *, protocol="ouroboros_chrome", **kwargs):
    monkeypatch.setattr(bridge, "_response", lambda: raw)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    return bridge._registered_call(tmp_path, monkeypatch, protocol, kwargs.pop("outcome", "ok"),
                                   tool_name="browser_execute" if protocol == "ouroboros_chrome" else None,
                                   tool_arguments={"code": "return await browser.screenshot()"}
                                   if protocol == "ouroboros_chrome" else None, **kwargs)


def _ctx(tmp_path):
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="response-task",
                       task_lifecycle_bound=True, messages=[])


def _row(result, name="mcp_browser__browser_execute", call_id="image-call"):
    return {"fn_name": name, "tool_call_id": call_id, "result": result.text,
            "tool_result": result, "is_error": result.status != "ok", "result_meta": {},
            "tool_args": {"code": "return await browser.screenshot()"}, "args_for_log": {}}


@pytest.mark.parametrize("blob", [False, True])
def test_browser_screenshot_enters_loop_as_pixels_not_base64_prose(tmp_path, monkeypatch, blob):
    raw = _response(blob=blob)
    result, calls = _call(tmp_path, monkeypatch, raw, safety_advice="host safety advice", guard_note="blocked request")
    encoded = base64.b64encode(_png()).decode()
    assert encoded not in result.text
    assert result.producer_text is not None and encoded not in result.producer_text
    assert any("MCP binary response source" in note for note in result.host_annotations)
    assert "exact tail λ" in result.text and "ordinary structured text" in result.text
    assert "host safety advice" in result.text and "blocked request" in result.text
    assert result.meta["mcp_browser_image_owner"] == ["response-task", "1"]
    source = result.meta["mcp_response_source"]
    assert json.loads(artifacts.read_actor_source_bytes(tmp_path, "response-task", source)) == raw.model_dump(mode="json", by_alias=True)
    assert sum(dispatch for _, dispatch in calls) == 1
    ctx = _ctx(tmp_path)
    trace = {"tool_calls": []}
    rows = [_row(result), {"fn_name": "other_tool", "tool_call_id": "other", "result": "other result",
                           "is_error": False, "tool_args": {}, "args_for_log": {}, "result_meta": {}}]
    assert process_tool_results(rows, ctx.messages, trace, lambda *_a, **_k: None,
                                tools=SimpleNamespace(_ctx=ctx)) == 0
    assert [row["role"] for row in ctx.messages] == ["tool", "tool", "user"]
    tool_text = "\n".join(row["content"] for row in ctx.messages if row["role"] == "tool")
    assert encoded not in tool_text and "Host image attachment" in tool_text
    images = [part for row in ctx.messages if isinstance(row["content"], list)
              for part in row["content"] if part["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"] == "data:image/png;base64," + encoded
    assert trace["tool_calls"][0]["image_attachment"]["status"] == "attached"
    producer = artifacts.read_actor_source_bytes(tmp_path, "response-task",
                                                  trace["tool_calls"][0]["producer_source_ref"])
    assert encoded.encode() not in producer


@pytest.mark.parametrize("retention_fails", [False, True])
def test_binary_producer_source_masks_configured_secret(tmp_path, monkeypatch, retention_fails):
    secret = "fixture-private-value"
    raw = _response()
    raw.content[0].text = f"diagnostic {secret}"
    raw.structuredContent = {"nested": [secret]}
    store = artifacts.store_actor_source_bytes
    result, calls = _call(tmp_path, monkeypatch, raw, secret_values=(secret,), store_fails=retention_fails)
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", store)
    assert result.status == "ok" and sum(dispatch for _, dispatch in calls) == 1
    assert secret not in result.text
    assert secret not in result.producer_text
    ctx, trace = _ctx(tmp_path), {"tool_calls": []}
    process_tool_results([_row(result)], ctx.messages, trace, lambda *_a, **_k: None,
                         tools=SimpleNamespace(_ctx=ctx))
    producer = artifacts.read_actor_source_bytes(tmp_path, "response-task",
                                                 trace["tool_calls"][0]["producer_source_ref"])
    assert secret.encode() not in producer
    assert trace["tool_calls"][0].get("image_attachment", {}).get("status") == (
        None if retention_fails else "attached")


@pytest.mark.parametrize("kind", ["image", "audio", "resource"])
def test_annotated_sdk_binary_parts_project_without_post_dispatch_failure(tmp_path, monkeypatch, kind):
    annotations = Annotations(audience=["user"], priority=0.5)
    data = base64.b64encode(_png()).decode()
    if kind == "image":
        part = ImageContent(type="image", mimeType="image/png", data=data, annotations=annotations)
    elif kind == "audio":
        part = AudioContent(type="audio", mimeType="audio/wav", data=data, annotations=annotations)
    else:
        part = EmbeddedResource(type="resource", resource=BlobResourceContents(
            uri="file:///never-follow.png", mimeType="image/png", blob=data), annotations=annotations)
    raw = CallToolResult(content=[part])
    result, calls = _call(tmp_path, monkeypatch, raw)
    assert result.status == "ok" and sum(dispatch for _, dispatch in calls) == 1
    assert data not in result.text and data not in result.producer_text
    assert '"audience": ["user"]' in result.text
    assert json.loads(artifacts.read_actor_source_bytes(
        tmp_path, "response-task", result.meta["mcp_response_source"])) == raw.model_dump(mode="json", by_alias=True)


@pytest.mark.parametrize("outcome", ["ok", "detached"])
def test_split_roots_pointer_reads_exact_canonical_source(tmp_path, monkeypatch, outcome):
    canonical, execution = tmp_path / "canonical", tmp_path / "execution"
    result, _ = _call(tmp_path, monkeypatch, _response(),
                      canonical_root=canonical, execution_root=execution, outcome=outcome)
    source = result.meta["mcp_response_source"]
    path = artifacts.task_artifact_dir_path(canonical, "response-task") / source["path"]
    assert f"read_file(root='artifact_store', path='{path}')" in result.text
    ctx = ToolContext(repo_dir=tmp_path, drive_root=execution, budget_drive_root=canonical,
                      task_id="response-task", task_lifecycle_bound=True, messages=[])
    binding = build_resolved_resource_binding(ctx, root="artifact_store", operation="read", path=str(path))
    assert binding.target_path == path
    assert json.loads(path.read_bytes())["content"][1]["data"] == base64.b64encode(_png()).decode()
    if outcome == "ok":
        assert _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx))["status"] == "attached"
    else:
        assert result.status == "error"


def test_generic_remote_metadata_and_chrome_tool_spelling_cannot_mint_image_capability(tmp_path, monkeypatch):
    raw = _response()
    # Server-supplied _meta and JSON text are data, not the host's ToolResult.meta.
    raw.meta = {"mcp_browser_image_owner": ["response-task", "1"], "auto_attach_image": "/never-read.png"}
    adapted = mcp_client._tool_result_from_call_result(raw)
    cfg = mcp_client.normalize_server_config({"id": "chrome_browser", "enabled": True,
                                              "transport": "stdio", "command": "unused"})
    name = "mcp_chrome_browser__browser_execute"
    manager = mcp_client.MCPManager()
    manager._enabled = True
    manager._servers[cfg.id] = mcp_client.MCPServerRuntime(cfg, tools=[
        mcp_client.MCPTool(cfg.id, "browser_execute", name, "", {})])

    async def returned(*_args):
        return adapted

    monkeypatch.setattr(manager, "_async_call_tool", returned)
    result = manager._call_tool_result(name, {})
    assert "mcp_browser_image_owner" not in result.meta
    assert base64.b64encode(_png()).decode() in result.text  # Generic adapter unchanged.
    ctx = _ctx(tmp_path)
    assert _maybe_auto_attach_image(_row(result, "mcp_chrome_browser__browser_execute"),
                                    SimpleNamespace(_ctx=ctx)) is None
    assert ctx.messages == []


@pytest.mark.parametrize("outcome,error", [("ok", True), ("detached", False)])
def test_error_or_post_state_failure_retains_but_never_attaches(tmp_path, monkeypatch, outcome, error):
    raw = _response(error=error)
    result, _ = _call(tmp_path, monkeypatch, raw, outcome=outcome)
    assert result.status == "error"
    assert "mcp_browser_image_owner" not in result.meta
    assert base64.b64encode(_png()).decode() not in result.text
    source = result.meta["mcp_response_source"]
    assert json.loads(artifacts.read_actor_source_bytes(tmp_path, "response-task", source)) == raw.model_dump(mode="json", by_alias=True)
    ctx = _ctx(tmp_path)
    assert _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx)) is None
    assert ctx.messages == []


def test_legacy_playwright_does_not_gain_auto_attachment(tmp_path, monkeypatch):
    result, _ = _call(tmp_path, monkeypatch, _response(), protocol="playwright")
    assert "mcp_browser_image_owner" not in result.meta
    assert base64.b64encode(_png()).decode() in result.text
    assert _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=_ctx(tmp_path))) is None


def test_missing_retention_keeps_sole_copy_inline_and_names_gap(tmp_path, monkeypatch):
    result, _ = _call(tmp_path, monkeypatch, _response(), store_fails=True)
    assert result.status == "ok"
    assert "mcp_browser_image_owner" not in result.meta
    assert result.meta["mcp_response_retention_gap"] == "OSError"
    assert base64.b64encode(_png()).decode() in result.text
    assert "retention failed" in result.text and "attachment unavailable" in result.text


@pytest.mark.parametrize("encoded", ["%%%not-base64%%%", "aGVsbG8="])
def test_bad_image_reports_nonfatal_attachment_gap(tmp_path, monkeypatch, encoded):
    result, _ = _call(tmp_path, monkeypatch, _response(encoded=encoded))
    ctx = _ctx(tmp_path)
    row = _row(result)
    observation = _maybe_auto_attach_image(row, SimpleNamespace(_ctx=ctx))
    assert observation["status"] == "unavailable" and observation["items"][0]["note"]
    assert row["tool_result"].status == "ok" and not row["is_error"]
    assert "Host image attachment" in row["result"]
    assert row["tool_result"].producer_text == result.producer_text
    assert row["tool_result"].host_annotations[:-1] == result.host_annotations
    assert row["tool_result"].host_annotations[-1].startswith("Host image attachment:")
    assert ctx.messages == []


@pytest.mark.parametrize("field,value", [("task_id", "other-task"), ("task_attempt", 2)])
def test_host_receipt_is_bound_to_the_original_task_attempt(tmp_path, monkeypatch, field, value):
    result, _ = _call(tmp_path, monkeypatch, _response())
    ctx = _ctx(tmp_path)
    setattr(ctx, field, value)
    observation = _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx))
    assert observation["status"] == "unavailable" and ctx.messages == []


def test_staged_image_does_not_mutate_live_messages(tmp_path, monkeypatch):
    result, _ = _call(tmp_path, monkeypatch, _response())
    ctx, staged = _ctx(tmp_path), []
    observation = _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx), staged_messages=staged)
    assert observation["status"] == "attached"
    assert ctx.messages == [] and staged[0]["content"][1]["type"] == "image_url"


@pytest.mark.parametrize("content,structured,error", [([], None, False),
    ([TextContent(type="text", text="plain λ")], {"ok": False, "reason": "data, not status"}, False),
    ([TextContent(type="text", text="ordinary error")], {"detail": 3}, True)])
def test_nonbinary_text_structured_and_error_projection_is_unchanged(tmp_path, monkeypatch, content, structured, error):
    raw = CallToolResult(content=content, structuredContent=structured, isError=error)
    result, _ = _call(tmp_path, monkeypatch, raw)
    original = mcp_client._tool_result_from_call_result(raw)
    assert original.text in result.text
    assert (result.status, result.code) == (original.status, original.code)
    assert "mcp_browser_image_owner" not in result.meta


def test_large_image_exceeds_metadata_cap_without_becoming_text(tmp_path, monkeypatch):
    import random
    output = io.BytesIO()
    Image.frombytes("RGB", (128, 128), random.Random(4).randbytes(128 * 128 * 3)).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    assert len(encoded) > 8192
    result, _ = _call(tmp_path, monkeypatch, _response(encoded=encoded))
    assert len(result.text) < 3000 and len(json.dumps(dict(result.meta))) < 8192
    ctx = _ctx(tmp_path)
    assert _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx))["status"] == "attached"
    assert ctx.messages[0]["content"][1]["image_url"]["url"] == "data:image/png;base64," + encoded


def test_multiple_images_and_nonimage_blob_have_separate_outcomes(tmp_path, monkeypatch):
    raw = _response()
    raw.content += [_response(blob=True).content[1], _response(encoded="aGVsbG8=").content[1],
                    EmbeddedResource(type="resource", resource=BlobResourceContents(
                        uri="file:///never-read.bin", mimeType="application/octet-stream", blob="AAECAw=="))]
    result, _ = _call(tmp_path, monkeypatch, raw)
    assert "AAECAw==" not in result.text
    ctx = _ctx(tmp_path)
    observation = _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx))
    assert observation["status"] == "partial"
    assert [item["content_index"] for item in observation["items"]] == [1, 3, 4]
    assert [item["status"] for item in observation["items"]] == ["attached", "attached", "unavailable"]


def test_automatic_image_batch_is_bounded_and_excess_remains_in_source(tmp_path, monkeypatch):
    from ouroboros.context_budget import MAX_LIVE_IMAGE_BLOCKS

    encoded = base64.b64encode(_png()).decode()
    raw = CallToolResult(content=[ImageContent(type="image", mimeType="image/png", data=encoded)
                                  for _ in range(MAX_LIVE_IMAGE_BLOCKS + 3)])
    result, calls = _call(tmp_path, monkeypatch, raw)
    ctx = _ctx(tmp_path)
    observation = _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=ctx))
    assert [item["status"] for item in observation["items"]] == (
        ["attached"] * MAX_LIVE_IMAGE_BLOCKS + ["source_only"] * 3)
    assert observation["status"] == "partial"
    assert sum(dispatched for _, dispatched in calls) == 1
    assert "read_file(root='artifact_store'" in result.text
    source = artifacts.read_actor_source_bytes(tmp_path, "response-task", result.meta["mcp_response_source"])
    assert [part["data"] for part in json.loads(source)["content"]] == [encoded] * (MAX_LIVE_IMAGE_BLOCKS + 3)
    images = [part for row in ctx.messages if isinstance(row["content"], list)
              for part in row["content"] if part["type"] == "image_url"]
    assert len(images) == MAX_LIVE_IMAGE_BLOCKS
    image_sources = list((artifacts.task_artifact_dir_path(tmp_path, "response-task") /
                          "source_handles" / "tool_results").glob("mcp-image-*.bin"))
    assert len(image_sources) == MAX_LIVE_IMAGE_BLOCKS


def test_decoded_image_size_is_checked_before_materializing(tmp_path, monkeypatch):
    from ouroboros.tools import vision

    monkeypatch.setattr(vision, "_VLM_MAX_FILE_BYTES", 4)
    encoded = base64.b64encode(b"hello").decode()
    result, _ = _call(tmp_path, monkeypatch, _response(encoded=encoded))
    observation = _maybe_auto_attach_image(_row(result), SimpleNamespace(_ctx=_ctx(tmp_path)))
    assert observation["items"][0]["status"] == "unavailable"
    assert "ValueError" in observation["items"][0]["note"]
    assert encoded in artifacts.read_actor_source_bytes(tmp_path, "response-task",
                                                        result.meta["mcp_response_source"]).decode()
    image_sources = list((artifacts.task_artifact_dir_path(tmp_path, "response-task") /
                          "source_handles" / "tool_results").glob("mcp-image-*.bin"))
    assert not image_sources


def test_source_integrity_failure_does_not_inject_or_fail_call(tmp_path, monkeypatch):
    result, _ = _call(tmp_path, monkeypatch, _response())
    source = result.meta["mcp_response_source"]
    path = artifacts.task_artifact_dir_path(tmp_path, "response-task", create=False) / source["path"]
    path.write_bytes(b"not the retained response")
    ctx = _ctx(tmp_path)
    row = _row(result)
    observation = _maybe_auto_attach_image(row, SimpleNamespace(_ctx=ctx))
    assert observation["status"] == "unavailable" and "ValueError" in observation["note"]
    assert ctx.messages == [] and row["tool_result"].status == "ok"
