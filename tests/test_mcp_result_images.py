"""MCP protocol images reach the model as pixels through the ordinary staged attachment.

A real low-level MCP SDK server (memory streams) answers the registered call; the manager,
registry, ``process_tool_results`` and the send projection are production code. The
OpenAI-compatible SDK request crosses loopback HTTP to a recorder whose decoded image bytes
are compared with the PNG the server sent; the Anthropic request and the Claudexor upload
are checked at their payload builders. No vendor, paid or Chrome call is made. The registry
context and the sender share one synthetic task and attempt; a taskless context and the
text API, which own no task, keep the ordinary projection.
"""
from __future__ import annotations

import base64
import copy
import functools
import hashlib
import io
import json
import pathlib
import random
import re
import socket
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

mcp_types = pytest.importorskip("mcp.types")

from ouroboros import artifacts  # noqa: E402

TASK, ATTEMPT = "mcp-image-task", 2
CHILD, PARENT = "mcp-image-child", "mcp-image-parent"
TOOL = "mcp_shots__capture"


def _png(seed: int, size=(40, 24)) -> bytes:
    from PIL import Image

    rnd = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size[0] * size[1])])
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


PNG = _png(2457)
B64 = base64.b64encode(PNG).decode()
# A configured secret short enough to occur inside the base64: masking there would corrupt it.
SECRET = next(B64[i:i + 4] for i in range(len(B64)) if re.fullmatch(r"(?=.*\d)(?=.*[A-Z])[A-Za-z0-9]{4}", B64[i:i + 4]))


def _text(text):
    return mcp_types.TextContent(type="text", text=text)


def _image(data=B64, **extra):
    return mcp_types.ImageContent(type="image", mimeType="image/png", data=data, **extra)


def _blob(data=B64, mime="image/png"):
    return mcp_types.EmbeddedResource(type="resource", resource=mcp_types.BlobResourceContents(
        uri="file:///never-read.png", mimeType=mime, blob=data))


def _images_of(*shots):
    return mcp_types.CallToolResult(content=[_image(base64.b64encode(shot).decode()) for shot in shots])


class Peer:
    """The real low-level MCP SDK server behind the registered tool; answers ``self.result``."""

    def __init__(self):
        self.result = mcp_types.CallToolResult(content=[_text("captured"), _image()])
        self.calls, self.delay = 0, 0.0

    def transport(self):
        import anyio
        from mcp.server.lowlevel import Server
        from mcp.shared.memory import create_client_server_memory_streams

        server = Server("mcp-image-peer")

        @server.list_tools()
        async def tools():
            return [mcp_types.Tool(name="capture", description="capture", inputSchema={"type": "object"})]

        @server.call_tool(validate_input=False)
        async def call(_name, _arguments):
            self.calls += 1
            await anyio.sleep(self.delay)
            return self.result

        @asynccontextmanager
        async def transport(_cfg):
            async with create_client_server_memory_streams() as (client, (read, write)):
                async with anyio.create_task_group() as group:
                    group.start_soon(functools.partial(server.run, read, write, server.create_initialization_options()))
                    try:
                        yield client
                    finally:
                        group.cancel_scope.cancel()

        return transport


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    from ouroboros import config, mcp_client
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "catalogs")  # catalog writes stay private
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    peer = Peer()
    monkeypatch.setattr(mcp_client, "_transport_factory", peer.transport())
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [{
        "id": "shots", "name": "Shots", "enabled": True, "transport": "streamable_http",
        "url": "https://example.com/mcp", "auth_header": "Authorization", "auth_token": SECRET, "allowed_tools": []}]}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    mcp_client.reset_manager_for_tests()
    manager = mcp_client.get_manager()
    manager.reconfigure(settings)
    assert manager.refresh_server("shots")["ok"]
    (tmp_path / "repo").mkdir()
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_id, registry._ctx.task_attempt, registry._ctx.messages = TASK, ATTEMPT, []
    yield SimpleNamespace(peer=peer, registry=registry, ctx=registry._ctx, manager=manager, data=tmp_path / "data",
                          repo=tmp_path / "repo", cfg=manager._servers["shots"].config)
    mcp_client.reset_manager_for_tests()


def _ordinary(env, raw):
    """The projection every response had before protocol images were retained."""
    from ouroboros.mcp_client import _stringify_call_result
    from ouroboros.mcp_schema import _model_facing_result, _redact_error_text

    return _model_facing_result(env.cfg, "capture", _redact_error_text(_stringify_call_result(raw), env.cfg))


def _row(result, call_id="call-1", name=TOOL):
    return {"fn_name": name, "tool_call_id": call_id, "result": result.text, "is_error": result.status != "ok",
            "tool_result": result, "tool_args": {}, "args_for_log": {}, "result_meta": {}}


def _assistant(rows):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": row["tool_call_id"], "type": "function", "function": {"name": row["fn_name"], "arguments": "{}"}}
        for row in rows]}


def _batch(env, rows, history=None):
    from ouroboros.loop_tool_execution import process_tool_results

    messages = [{"role": "user", "content": "look"}] if history is None else history
    messages.append(_assistant(rows))
    trace = {"tool_calls": []}
    process_tool_results(rows, messages, trace, lambda *_a, **_k: None, tools=env.registry)
    return messages, trace


def _pixels(value):
    """Decoded bytes of every data-URL or base64-source image part anywhere in ``value``."""
    found = []

    def walk(item):
        if isinstance(item, dict):
            url = (item.get("image_url") or {}).get("url", "") if item.get("type") == "image_url" else ""
            if url.startswith("data:"):
                found.append(base64.b64decode(url.split(",", 1)[1]))
            if item.get("type") == "image" and (item.get("source") or {}).get("type") == "base64":
                found.append(base64.b64decode(item["source"]["data"]))
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return found


def _retained_paths(text):
    return re.findall(r"retained as (.+?)\]", text)


def _resource_paths(text):
    return re.findall(r"task artifact (.+?); sha256 [0-9a-f]{64}\]", text)


def _no_base64_text(text):
    """No long base64 run outside image data URLs; masking inside base64 cannot fool this oracle."""
    return not re.search(r"[A-Za-z0-9+/=]{200,}", re.sub(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+", "", text))


def _items(trace, index=0):
    observation = trace["tool_calls"][index]["image_attachment"]
    return observation["status"], [(item["content_index"], item["status"]) for item in observation["items"]]


class Recorder:
    """Loopback OpenAI-compatible endpoint: records each request body and answers "ok"."""

    def __init__(self):
        bodies = self.bodies = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"))
                data = json.dumps({"id": "r", "object": "chat.completion", "model": "fixture/vision",
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": "ok"}}],
                                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("mode", ["inline", "caption", "off"])
def test_registered_screenshot_reaches_the_physical_request_as_pixels(mcp, monkeypatch, mode):
    from ouroboros import pricing
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import _prepare_main_messages

    # No tariff for the synthetic model: reservation/settlement still run with unknown pricing.
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_a, **_k: {})
    remote_lookups = []
    resolve = socket.getaddrinfo

    def loopback_only(host, *args, **kwargs):
        if host not in ("127.0.0.1", "::1", "localhost"):
            remote_lookups.append(str(host))
            raise OSError(f"unexpected remote lookup in the image fixture: {host}")
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", loopback_only)
    result = mcp.registry.execute_result(TOOL, {})
    messages, trace = _batch(mcp, [_row(result)])
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", mode)
    with Recorder() as recorder:
        llm = LLMClient(api_key="fixture-key", base_url=recorder.base_url)
        sent = _prepare_main_messages(messages, model="openrouter::fixture/vision", llm=llm, accumulated_usage={},
                                      drive_root=mcp.data, task_id=TASK, task_attempt=ATTEMPT, event_queue=None,
                                      use_local=False)
        llm.chat(sent, model="openrouter::fixture/vision", tools=None, max_tokens=16, no_proxy=True, timeout=20)
    assert not remote_lookups, remote_lookups  # pricing failures may be handled while the image request succeeds
    assert len(recorder.bodies) == (2 if mode == "caption" else 1)  # the caption is also a real loopback request
    wire = recorder.bodies[-1]
    assert [hashlib.sha256(image).hexdigest() for image in _pixels(wire)] == (
        [hashlib.sha256(PNG).hexdigest()] if mode in {"inline", "auto"} else [])
    assert _no_base64_text(json.dumps(wire)), "protocol image bytes never travel as model-facing text"
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1 and "captured" in result.text
    assert _no_base64_text(result.text) and _no_base64_text(result.producer_text or "")
    assert result.meta["mcp_image_owner"] == [TASK, str(ATTEMPT)]
    assert _items(trace) == ("attached", [(1, "attached")])


def test_mcp_pixels_reach_the_anthropic_request_and_the_claudexor_upload(mcp):
    from ouroboros import llm_claudexor
    from ouroboros.llm import LLMClient

    messages, _ = _batch(mcp, [_row(mcp.registry.execute_result(TOOL, {}))])
    _system, anthropic = LLMClient(api_key="fixture-key")._build_anthropic_messages(copy.deepcopy(messages))
    assert _pixels(anthropic) == [PNG]
    payload = llm_claudexor._request({"source": "codex", "resolved_model": "fixture"}, copy.deepcopy(messages), None, {})
    upload = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    assert _pixels(json.loads(upload)) == [PNG] and _no_base64_text(upload)


@pytest.mark.parametrize("vision", [True, False])
def test_voice_engine_gets_mcp_pixels_or_the_saved_file_caption(mcp, vision):
    from ouroboros.realtime_setup import rows_for_engine

    messages, _ = _batch(mcp, [_row(mcp.registry.execute_result(TOOL, {}))])
    rows = rows_for_engine(messages, vision=vision, schemas=[{"function": {"name": "view_image"}}])
    assert _pixels(rows) == ([PNG] if vision else [])
    if not vision:
        caption = json.dumps(rows, ensure_ascii=False)
        assert "this voice engine does not see images" in caption and ".content-1-" in caption


def test_mixed_annotated_parts_keep_text_and_data_while_every_binary_leaves_the_text(mcp):
    second = _png(2)
    other = base64.b64encode(second).decode()
    mcp.peer.result = mcp_types.CallToolResult(content=[
        _text("head λ"), _image(annotations=mcp_types.Annotations(audience=["user"], priority=0.5)), _blob(other),
        mcp_types.AudioContent(type="audio", mimeType="audio/wav", data="UklGRgAAAAA="),
        _blob("AAECAw==", mime="application/octet-stream"), _text("tail")], structuredContent={"answer": [1, "x"]})
    result = mcp.registry.execute_result(TOOL, {})
    assert result.status == "ok" and _no_base64_text(result.text)
    assert "UklGRgAAAAA=" not in result.text and "AAECAw==" not in result.text
    assert all(part in result.text for part in ("head λ", "tail", '"answer": [1, "x"]', '"audience": ["user"]'))
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert retained == mcp.peer.result.model_dump(mode="json", by_alias=True)
    assert [pathlib.Path(path).read_bytes() for path in _retained_paths(result.text)] == [PNG, second]
    assert [pathlib.Path(path).read_bytes() for path in _resource_paths(result.text)] == [second, b"\x00\x01\x02\x03"]
    assert all(artifacts.registered_task_artifact(mcp.data, TASK, pathlib.Path(path).name)
               for path in _resource_paths(result.text))
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(1, "attached"), (2, "attached")]) and _pixels(messages) == [PNG, second]


@pytest.mark.parametrize("mime,payload,suffix", [
    ("application/pdf", b"%PDF-1.7\n\x00\xff", ".pdf"),
    ("text/csv", b"name,score\r\nA,1\r\n", ".csv"),
    ("application/zip", b"PK\x03\x04\x00\xff", ".zip"),
])
def test_binary_only_resource_is_registered_and_readable_after_disconnect(mcp, mime, payload, suffix):
    from ouroboros import mcp_client

    encoded = base64.b64encode(payload).decode()
    mcp.peer.result = mcp_types.CallToolResult(content=[_blob(encoded, mime=mime)])
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1
    assert encoded not in result.text and "mcp_image_owner" not in result.meta
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert retained["content"][0]["resource"]["blob"] == encoded
    (path,) = _resource_paths(result.text)
    path = pathlib.Path(path)
    assert path.suffix == suffix and path.read_bytes() == payload
    record = artifacts.registered_task_artifact(mcp.data, TASK, path.name)
    assert record["kind"] == "mcp_resource" and record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["immutable"] is True
    mcp_client.reset_manager_for_tests()  # the task artifact is independent of the MCP connection
    assert path.read_bytes() == payload
    assert artifacts.registered_task_artifact(mcp.data, TASK, path.name)["status"] == "ready"
    if mime == "text/csv":
        read = mcp.registry.execute_result("read_file", {"root": "artifact_store", "path": path.name})
        assert read.status == "ok" and "name,score" in read.text
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]


def test_mixed_image_and_resource_registers_blob_and_attaches_only_image(mcp):
    payload = b"PK\x03\x04\xff\x00"
    encoded = base64.b64encode(payload).decode()
    mcp.peer.result = mcp_types.CallToolResult(content=[_text("result"), _image(),
        _blob(encoded, mime="application/zip"), _text("end")])
    result = mcp.registry.execute_result(TOOL, {})
    assert encoded not in result.text and len(_resource_paths(result.text)) == 1
    assert pathlib.Path(_resource_paths(result.text)[0]).read_bytes() == payload
    assert artifacts.registered_task_artifact(mcp.data, TASK, pathlib.Path(_resource_paths(result.text)[0]).name)
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(1, "attached")]) and _pixels(messages) == [PNG]


def test_bad_resource_base64_names_gap_and_keeps_exact_source(mcp):
    mcp.peer.result = mcp_types.CallToolResult(content=[_blob("%%%bad-base64%%%", mime="application/pdf")])
    result = mcp.registry.execute_result(TOOL, {})
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert retained["content"][0]["resource"]["blob"] == "%%%bad-base64%%%"
    assert "resource data is not valid base64" in result.text and not _resource_paths(result.text)
    assert "mcp_image_owner" not in result.meta and not artifacts.collect_task_artifact_records(mcp.data, TASK)


def test_resource_storage_failure_keeps_success_and_retained_bytes(mcp, monkeypatch):
    encoded = base64.b64encode(b"%PDF-1.7\n").decode()
    mcp.peer.result = mcp_types.CallToolResult(content=[_blob(encoded, mime="application/pdf")])

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", refuse)
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1
    assert "resource artifact not saved (OSError)" in result.text and not _resource_paths(result.text)
    assert encoded not in result.text
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert retained["content"][0]["resource"]["blob"] == encoded


def _big_blob(mime="application/zip"):
    """A blob long enough for ``_no_base64_text`` to see it, beside the base64 of a real image."""
    payload = b"PK\x03\x04" + bytes(range(256)) * 4
    return payload, base64.b64encode(payload).decode()


def test_binary_only_source_retention_failure_describes_the_blob_never_inline(mcp, monkeypatch):
    payload, encoded = _big_blob()
    mcp.peer.result = mcp_types.CallToolResult(content=[_blob(encoded, mime="application/zip")])

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse)
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1
    assert encoded not in result.text and _no_base64_text(result.text)
    assert (f"[binary resource, about {len(payload)} bytes, not shown: response retention failed (OSError)]"
            in result.text) and "file:///never-read.png" in result.text and "application/zip" in result.text
    assert result.meta["mcp_response_retention_gap"] == "OSError"
    assert "mcp_response_source" not in result.meta and not _resource_paths(result.text)
    assert not artifacts.collect_task_artifact_records(mcp.data, TASK)


@pytest.mark.parametrize("error", [False, True], ids=["ok", "error"])
def test_a_blob_without_an_owning_task_is_described_and_its_uri_never_read(mcp, error):
    """No task store to keep it: the ordinary projection would print the whole base64 (Fable, 12 Nov 2026)."""
    from ouroboros.mcp_client import _stringify_call_result

    payload, encoded = _big_blob()
    planted = mcp.data / "planted.zip"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"LOCAL FILE SENTINEL")
    raw = mcp_types.CallToolResult(isError=error, content=[_text("export"), mcp_types.EmbeddedResource(
        type="resource", resource=mcp_types.BlobResourceContents(uri=planted.as_uri(), mimeType="application/zip",
                                                                 blob=encoded))])
    mcp.peer.result = raw
    assert encoded in _stringify_call_result(raw)  # the ordinary rendering this replaces
    contextless = mcp.manager.call_tool(TOOL, {})
    mcp.ctx.task_id = None
    result = mcp.registry.execute_result(TOOL, {})
    for text in (contextless, result.text):
        assert encoded not in text and _no_base64_text(text) and "LOCAL FILE SENTINEL" not in text, text
        assert f"about {len(payload)} bytes, not shown: no task owns this call" in text
        assert planted.as_uri() in text and "export" in text and text.count("MCP_TOOL_ERROR") == int(error)
    assert (result.status, result.code) == (("error", "MCP_ERROR") if error else ("ok", "OK"))
    assert result.meta["mcp_resource_gap"] == "no task owns this call" and "mcp_response_source" not in result.meta
    assert not list(mcp.data.rglob("source_handles")) and mcp.peer.calls == 2


@pytest.mark.parametrize("seam", ["_projected_body", "_protocol_view"], ids=["render", "render-and-retention"])
def test_a_projection_fault_never_falls_back_to_a_blob_s_base64(mcp, monkeypatch, seam):
    """The image fallback keeps the ordinary text; a blob's is a count with the error bit, never its base64."""
    from ouroboros import mcp_result_images

    _payload, encoded = _big_blob()
    mcp.peer.result = mcp_types.CallToolResult(content=[_text("export"), _blob(encoded, mime="application/zip")])

    def fault(*_args, **_kwargs):
        raise RuntimeError("fault")

    monkeypatch.setattr(mcp_result_images, seam, fault)
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1
    assert encoded not in result.text and _no_base64_text(result.text), result.text
    assert "projection failed (RuntimeError)" in result.text and result.meta["mcp_response_projection_gap"] == "RuntimeError"
    assert "an MCP result with 1 embedded binary resource(s), not shown" in result.text
    assert ("kept exactly in the retained response" in result.text) == (seam == "_projected_body")


def test_error_resource_is_registered_without_image_attachment(mcp):
    encoded = base64.b64encode(b"a,b\n1,2\n").decode()
    mcp.peer.result = mcp_types.CallToolResult(isError=True, content=[_text("failed"), _blob(encoded, mime="text/csv")])
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("error", "MCP_ERROR") and "failed" in result.text
    assert pathlib.Path(_resource_paths(result.text)[0]).read_bytes() == b"a,b\n1,2\n"
    assert "mcp_image_owner" not in result.meta
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]


def test_resource_protocol_masking_and_repeat_digest_reuse(mcp):
    payload = b"%PDF-1.7\nsecret\x00"
    encoded = base64.b64encode(payload).decode()
    planted = mcp.data / f"{SECRET}.pdf"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"unrelated local file")
    mcp.peer.result = mcp_types.CallToolResult(content=[_text(f"echo {SECRET}"),
        mcp_types.EmbeddedResource(type="resource", resource=mcp_types.BlobResourceContents(
            uri=planted.as_uri(), mimeType=f"application/pdf; name={SECRET}", blob=encoded))])
    first = mcp.registry.execute_result(TOOL, {})
    second = mcp.registry.execute_result(TOOL, {})
    assert first.meta["mcp_response_source"]["sha256"] == second.meta["mcp_response_source"]["sha256"]
    assert _resource_paths(first.text) == _resource_paths(second.text)
    (path,) = _resource_paths(first.text)
    assert pathlib.Path(path).read_bytes() == payload
    assert pathlib.Path(path).parent == artifacts.task_artifact_dir_path(mcp.data, TASK)
    assert SECRET not in first.text
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, first.meta["mcp_response_source"]))
    resource = retained["content"][1]["resource"]
    assert resource["blob"] == encoded and resource["mimeType"] == "application/pdf; name=<redacted:mcp-auth-token>"
    assert SECRET not in resource["uri"]
    assert artifacts.registered_task_artifact(mcp.data, TASK, pathlib.Path(path).name)["sha256"] == hashlib.sha256(payload).hexdigest()


def test_error_result_keeps_its_status_and_offers_its_image_only_by_path(mcp):
    from ouroboros.tools.vision import attach_local_image_to_context

    mcp.peer.result = mcp_types.CallToolResult(isError=True, content=[_text("element not found"), _image()])
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("error", "MCP_ERROR") and "⚠️ MCP_TOOL_ERROR: element not found" in result.text
    assert _no_base64_text(result.text) and "mcp_image_owner" not in result.meta and result.meta["mcp_response_source"]
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]
    viewer = copy.copy(mcp.ctx)
    viewer.messages = []
    (path,) = _retained_paths(result.text)
    assert attach_local_image_to_context(viewer, path)[0] and _pixels(viewer.messages) == [PNG]


def test_corrupt_image_data_keeps_the_text_and_the_valid_sibling(mcp):
    mcp.peer.result = mcp_types.CallToolResult(content=[
        _text("three shots"), _image("%%%not-base64%%%"), _image(), _image("aGVsbG8=")])
    result = mcp.registry.execute_result(TOOL, {})
    assert result.status == "ok" and "three shots" in result.text and "not valid base64" in result.text
    # Decoded bytes the shared loader cannot read are kept exactly; the loader names why they are not shown.
    assert [pathlib.Path(path).read_bytes() for path in _retained_paths(result.text)] == [PNG, b"hello"]
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("partial", [(1, "unavailable"), (2, "attached"), (3, "unavailable")])
    assert "IMAGE_FORMAT_UNSUPPORTED" in trace["tool_calls"][0]["image_attachment"]["items"][2]["note"]
    assert _pixels(messages) == [PNG]
    assert "Host image attachment" in next(row["content"] for row in messages if row["role"] == "tool")


@pytest.mark.parametrize("fmt,mime", [("ICO", "image/x-icon"), ("PPM", "image/x-portable-pixmap")])
def test_a_format_only_the_shared_decoder_knows_reaches_the_model_as_view_image_shows_it(mcp, fmt, mime):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (32, 32), (90, 110, 130)).save(out, format=fmt)
    raw = out.getvalue()
    control = mcp.repo / f"control.{fmt.lower()}"
    control.write_bytes(raw)
    assert mcp.registry.execute_result("view_image", {"path": str(control)}).status == "ok"
    expected, mcp.ctx.messages = _pixels(mcp.ctx.messages), []
    assert len(expected) == 1 and expected[0] != raw  # the registered reader converts these bytes
    mcp.peer.result = mcp_types.CallToolResult(content=[
        mcp_types.ImageContent(type="image", mimeType=mime, data=base64.b64encode(raw).decode())])
    result = mcp.registry.execute_result(TOOL, {})
    (path,) = _retained_paths(result.text)
    assert pathlib.Path(path).read_bytes() == raw and re.fullmatch(r".+-[0-9a-f]{64}\.[a-z0-9]+", path)
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(0, "attached")]) and _pixels(messages) == expected


@pytest.mark.parametrize("mode,keep", [("max", 5), ("nano", 2)])
def test_fifty_images_attach_the_mode_floor_and_keep_the_rest_as_files(mcp, mode, keep):
    shots = [_png(seed, size=(8, 8)) for seed in range(50)]
    mcp.peer.result = _images_of(*shots)
    mcp.ctx.context_fit_plan = SimpleNamespace(preferred_mode=mode)
    result = mcp.registry.execute_result(TOOL, {})
    messages, trace = _batch(mcp, [_row(result)])
    observation = trace["tool_calls"][0]["image_attachment"]
    assert [item["status"] for item in observation["items"]] == ["attached"] * keep + ["source_only"] * (50 - keep)
    assert _pixels(messages) == shots[:keep], "nothing attached was evicted again by its own batch"
    assert [pathlib.Path(item["path"]).read_bytes() for item in observation["items"][keep:]] == shots[keep:]
    assert [pathlib.Path(path).read_bytes() for path in _retained_paths(result.text)] == shots


def test_two_results_in_one_batch_report_each_image_s_final_state(mcp):
    shots = [_png(seed, size=(8, 8)) for seed in range(10)]
    results = []
    for group in (shots[:5], shots[5:]):
        mcp.peer.result = _images_of(*group)
        results.append(mcp.registry.execute_result(TOOL, {}))
    earlier = {"role": "user", "content": [{"type": "text", "text": "earlier"},
                                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + B64}}]}
    messages, trace = _batch(mcp, [_row(results[0], "call-a"), _row(results[1], "call-b")], [earlier])
    assert _items(trace, 0) == ("unavailable", [(index, "evicted") for index in range(5)])
    assert _items(trace, 1) == ("attached", [(index, "attached") for index in range(5)])
    assert _pixels(messages) == shots[5:]
    assert '"evicted"' in next(row["content"] for row in messages if row.get("tool_call_id") == "call-a")


def _context(env, drive, task_id, **metadata):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext

    ctx = ToolContext(repo_dir=env.repo, drive_root=drive, budget_drive_root=str(env.data), task_id=task_id,
                      task_lifecycle_bound=True, messages=[], task_metadata=metadata)
    registry = ToolRegistry(repo_dir=env.repo, drive_root=drive)
    registry.set_context(ctx)
    return SimpleNamespace(registry=registry, ctx=ctx, drive=drive)


def _delegated_child(env, attempt):
    """A read-only child on its own execution drive, placed as the host places one."""
    from ouroboros import headless
    from ouroboros.contracts.task_constraint import TaskConstraint

    child = _context(env, headless.prepare_task_drive(env.data, CHILD, "empty"), CHILD, delegation_role="subagent",
                     parent_task_id=PARENT, root_task_id=PARENT, budget_drive_root=str(env.data))
    child.ctx.task_attempt = attempt
    child.ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    return child


def _child_call(child, call_id, messages, trace):
    """One call through the loop's real tool handling, so its manifest is what retention transfers."""
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls

    call = {"id": call_id, "type": "function", "function": {"name": TOOL, "arguments": "{}"}}
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    handle_tool_calls([call], child.registry, child.drive / "logs", CHILD, StatefulToolExecutor(), messages, trace,
                      lambda *_a, **_k: None)
    return trace["tool_calls"][-1]


def _retire(env, drive, trace, **fields):
    """The host's whole terminal path: copy-back, ref promotion, settlement and trash removal."""
    from ouroboros import headless, task_custody
    from ouroboros.task_results import write_task_result

    write_task_result(drive, CHILD, "completed", result="done", artifact_status="ready", trace_refs={
        "tool_call_refs": [{"manifest_ref": row["trace_ref"]["manifest_ref"]} for row in trace["tool_calls"]]},
        **fields)
    headless.copy_child_task_result(env.data, {"id": CHILD, "drive_root": str(drive)})
    promoted = headless.retry_child_task_refs(env.data, drive, CHILD)
    assert promoted["child_ref_promotion"]["status"] == "complete", promoted["child_ref_promotion"]
    assert task_custody.settle_child_drive(env.data, CHILD, drive, live=lambda _: False)["status"] == "removed"
    assert task_custody.remove_custody_trash(env.data)["removed"] == 1
    assert not drive.exists()


def _zip(member="export/rows.csv", text="a,b\n1,2\n"):
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zipped:
        zipped.writestr(member, text)
    return out.getvalue()


def _split_export(env, content, error=False):
    """A delegated child, its parent and one call whose result carries ``content``, through the real loop."""
    from ouroboros.task_results import write_task_result

    child = _delegated_child(env, attempt=1)
    write_task_result(env.data, PARENT, "running")
    write_task_result(env.data, CHILD, "running", child_drive_root=str(child.drive))
    env.peer.result = mcp_types.CallToolResult(isError=error, content=content)
    trace = {"tool_calls": []}
    return child, trace, _child_call(child, "call-export", [{"role": "user", "content": "export"}], trace)["result"]


def _stored_child_result(env, child, trace):
    """The pipeline's own terminal store for the child, on its execution drive (``_store_task_result``)."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.task_results import load_task_result

    task = {"id": CHILD, "type": "task", "text": "export", "parent_task_id": PARENT, "root_task_id": PARENT}
    _store_task_result(SimpleNamespace(drive_root=child.drive, repo_dir=env.repo), task, "done",
                       {"rounds": 1, "cost": 0}, trace)
    return load_task_result(child.drive, CHILD)


@pytest.mark.parametrize("shape", ["zip", "error_csv", "image_blob", "empty"])
def test_a_split_drive_child_s_embedded_blob_is_one_deliverable_for_every_lifecycle_reader(mcp, shape):
    """A delegated child runs on its own execution drive. The blob it receives is a deliverable of that drive's store,
    where its own readers, the pipeline's stored row and sealed manifest, copy-back and settlement find it; the
    path it prints is the canonical twin of the same bytes and name, so that address, like the sources beside it,
    still reads after retirement, and the durable row names it, once, in both effective views."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.result_material import produced_material
    from ouroboros.task_finalization import build_sealed_final_package
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import effective_task_result
    from ouroboros.tools.core_artifacts import _delivery_path

    payload, mime = {"zip": (_zip(), "application/zip"), "error_csv": (b"a,b\n1,2\n", "text/csv"),
                     "image_blob": (PNG, "image/png"), "empty": (b"", "application/octet-stream")}[shape]
    child, trace, result = _split_export(
        mcp, [_text("export"), _blob(base64.b64encode(payload).decode(), mime=mime)], error=shape == "error_csv")
    (printed,) = _resource_paths(result)
    name = pathlib.Path(printed).name
    kept = task_artifact_dir_path(mcp.data, CHILD) / name  # the canonical twin the result prints
    assert pathlib.Path(printed) == kept and kept.read_bytes() == payload and "canonical copy not saved" not in result
    assert (task_artifact_dir_path(child.drive, CHILD) / name).read_bytes() == payload
    assert len(_retained_paths(result)) == (shape == "image_blob")
    for path in _retained_paths(result):  # an image stays a canonical source, as before
        assert pathlib.Path(path).is_relative_to(task_artifact_dir_path(mcp.data, CHILD) / "source_handles")

    # In the task: its own store lists it and answers by name, the printed path by itself; one production.
    assert name in child.registry.execute_result("list_files", {"root": "artifact_store", "path": "."}).text
    if shape == "zip":
        for arguments in ({"root": "artifact_store", "path": name}, {"path": printed}):
            assert "export/rows.csv" in child.registry.execute_result("read_archive", arguments).text, arguments
    if shape == "error_csv":
        for arguments in ({"root": "artifact_store", "path": name}, {"path": printed}):
            assert "1,2" in child.registry.execute_result("read_file", arguments).text, arguments
    assert _delivery_path(child.ctx, printed)[0].read_bytes() == payload  # send_file's file preparation
    produced = [row for row in produced_material(child.ctx)["files"] if row.get("kind") == "artifact"]
    assert [pathlib.Path(row["path"]).name for row in produced] == [name], produced

    stored = _stored_child_result(mcp, child, trace)
    rows = [row for row in stored["artifacts"] if row.get("name") == name]
    assert len(rows) == 1 and rows[0]["sha256"] == hashlib.sha256(payload).hexdigest(), stored["artifacts"]
    assert stored["artifact_bundle"]["status"] == "ready", stored["artifact_bundle"]
    sealed = build_sealed_final_package(stored, "done")
    assert [row["name"] for row in sealed["artifact_manifest"]].count(name) == 1, sealed
    live = effective_task_result(mcp.data, load_task_result(mcp.data, CHILD))  # the parent's view while it runs
    assert [pathlib.Path(str(row.get("path"))).name for row in live.get("artifacts") or []].count(name) == 1, live

    _retire(mcp, child.drive, trace, artifacts=stored["artifacts"])
    durable = load_task_result(mcp.data, CHILD)
    assert kept.read_bytes() == payload and not (task_artifact_dir_path(child.drive, CHILD) / name).exists()
    (row,) = [row for row in durable["artifacts"] if pathlib.Path(row["path"]).name == name]
    assert (row["path"], row["sha256"], row.get("immutable")) == (str(kept), rows[0]["sha256"], True)
    for materialize in (False, True):  # the listing cards and SSE read, and the detail/wait read
        view = effective_task_result(mcp.data, durable, materialize_artifacts=materialize)
        assert [item["path"] for item in view["artifacts"]].count(str(kept)) == 1, (materialize, view["artifacts"])
        assert view["artifact_bundle"]["status"] == "ready" and view.get("artifact_status") == "ready", view
    reopened = _delegated_child(mcp, attempt=2)
    for consumer in (reopened, _context(mcp, mcp.data, PARENT)):  # the printed path still reads after retirement
        assert _delivery_path(consumer.ctx, printed)[0].read_bytes() == payload
        if shape == "zip":
            assert "export/rows.csv" in consumer.registry.execute_result("read_archive", {"path": printed}).text


def test_a_split_drive_blob_whose_canonical_twin_fails_keeps_its_execution_copy(mcp, monkeypatch):
    """The execution store is the deliverable's authority: a canonical twin that cannot be written keeps the saved
    file, prints its execution path with that path's lifetime, never calls the server again, and the drive's custody
    then publishes the copy the durable row names."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import load_task_result

    real = artifacts.store_task_artifact_bytes

    def canonical_refused(drive_root, *args, **kwargs):
        if pathlib.Path(drive_root).resolve() == mcp.data.resolve():
            raise OSError("canonical store refused")
        return real(drive_root, *args, **kwargs)

    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", canonical_refused)
    child, trace, result = _split_export(mcp, [_blob("AAECAw==", mime="application/octet-stream")])
    assert mcp.peer.calls == 1
    (printed,) = _resource_paths(result)
    name = pathlib.Path(printed).name
    assert pathlib.Path(printed) == task_artifact_dir_path(child.drive, CHILD) / name
    assert pathlib.Path(printed).read_bytes() == b"\x00\x01\x02\x03" and "artifact not saved" not in result
    assert "canonical copy not saved (OSError): this execution-drive path lasts until the drive retires" in result
    assert not (task_artifact_dir_path(mcp.data, CHILD) / name).exists()
    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", real)
    stored = _stored_child_result(mcp, child, trace)
    _retire(mcp, child.drive, trace, artifacts=stored["artifacts"])
    (row,) = [row for row in load_task_result(mcp.data, CHILD)["artifacts"] if pathlib.Path(row["path"]).name == name]
    kept = task_artifact_dir_path(mcp.data, CHILD) / name
    assert row["path"] == str(kept) and kept.read_bytes() == b"\x00\x01\x02\x03" and mcp.peer.calls == 1


def test_a_split_drive_root_s_terminal_emit_names_its_mcp_blob_in_its_events_and_sealed_package(mcp, monkeypatch):
    """A root on its own execution drive (a headless split root) ends through the real ``emit_task_results``: the
    stored result, the ``task_done`` event and the sealed final package its post-task receives each name the blob it
    received once, ready. Nothing leaves the process: no event queue, and the post-task stage is held at its seam."""
    from ouroboros import agent_task_pipeline as pipeline, headless
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.task_results import load_task_result, write_task_result

    task_id, payload = "mcp-split-root", _zip()
    root = _context(mcp, headless.prepare_task_drive(mcp.data, task_id, "empty"), task_id, root_task_id=task_id)
    write_task_result(mcp.data, task_id, "running", child_drive_root=str(root.drive))
    mcp.peer.result = mcp_types.CallToolResult(content=[
        _text("export"), _blob(base64.b64encode(payload).decode(), mime="application/zip")])
    call = {"id": "call-export", "type": "function", "function": {"name": TOOL, "arguments": "{}"}}
    messages = [{"role": "user", "content": "export"}, {"role": "assistant", "content": "", "tool_calls": [call]}]
    trace = {"tool_calls": [], "reasoning_notes": []}
    handle_tool_calls([call], root.registry, root.drive / "logs", task_id, StatefulToolExecutor(), messages, trace,
                      lambda *_a, **_k: None)
    (printed,) = _resource_paths(trace["tool_calls"][-1]["result"])
    name = pathlib.Path(printed).name
    assert pathlib.Path(printed) == task_artifact_dir_path(mcp.data, task_id) / name  # the durable twin is printed
    sealed, pending = [], []
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **kw: sealed.append(kw["sealed_final"]))
    task = {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 1, "text": "export",
            "budget_drive_root": str(mcp.data)}
    pipeline.emit_task_results(SimpleNamespace(drive_root=root.drive, repo_dir=mcp.repo), None, None, pending, task,
                               "Exported.", {"rounds": 1}, trace, 0, root.drive / "logs", ctx=root.ctx)

    stored = load_task_result(root.drive, task_id)
    assert [row["name"] for row in stored["artifacts"]].count(name) == 1 and stored["artifact_bundle"]["status"] == "ready"
    (done,) = [event for event in pending if event["type"] == "task_done"]
    assert done["artifact_status"] == "ready", done
    assert [row["name"] for row in done["artifact_bundle"]["artifacts"]].count(name) == 1, done["artifact_bundle"]
    (send,) = [event for event in pending if event["type"] == "send_message"]
    assert send["text"] == "Exported." and send["task_id"] == task_id
    (package,) = sealed
    assert [row["name"] for row in package["artifact_manifest"]].count(name) == 1, package["artifact_manifest"]
    assert package["final_result_text"] == "Exported." and mcp.peer.calls == 1


def _printed_read(text):
    """The arguments of the ``read_file(...)`` call a result prints, exactly as a mind would pass them."""
    return dict(re.findall(r"(\w+)='([^']*)'", re.search(r"read_file\((.*?)\)", text).group(1)))


def test_every_delivered_locator_outlives_child_retirement_for_its_consumers(mcp):
    from ouroboros.context_images import evict_stale_image_blocks
    from ouroboros.loop_tool_execution import _maybe_auto_attach_image
    from ouroboros.maintenance_retirement import prune_agent_media_uploads
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_late_tool_result_delivery import _late_text

    shots, denied = [_png(seed, size=(8, 8)) for seed in range(7)], _png(99, size=(8, 8))
    child = _delegated_child(mcp, attempt=1)
    write_task_result(mcp.data, PARENT, "running")
    write_task_result(mcp.data, CHILD, "running", child_drive_root=str(child.drive))
    messages, trace = [{"role": "user", "content": "look"}], {"tool_calls": []}
    mcp.peer.result = _images_of(*shots)
    shown = _child_call(child, "call-shown", messages, trace)
    assert _items(trace)[1] == [(i, "attached") for i in range(5)] + [(i, "source_only") for i in (5, 6)]
    mcp.peer.result = mcp_types.CallToolResult(isError=True, content=[
        _text("denied"), _image(base64.b64encode(denied).decode())])
    failed = _child_call(child, "call-denied", messages, trace)
    assert failed["is_error"] and "image_attachment" not in failed

    blocks = [block for row in messages if isinstance(row.get("content"), list) for block in row["content"]
              if block.get("type") == "image_url"]
    evicted = copy.deepcopy(messages)
    assert evict_stale_image_blocks(evicted, incoming=10)
    placeholders = [re.search(r"re-view: view_image path=(.+?)\]", block["text"]).group(1) for row in evicted
                    if isinstance(row.get("content"), list) for block in row["content"] if block.get("_evicted_image")]
    captions = [re.search(r"original: (.+?)[;\]]", block["_caption"]).group(1) for block in blocks]
    # Attached blocks' own addresses first: viewing another copy of these bytes may not restore them.
    images = [*zip([block["_source_path"] for block in blocks], shots), *zip(placeholders, shots), *zip(captions, shots),
              *zip(_retained_paths(shown["result"]), shots), *zip(_retained_paths(failed["result"]), [denied])]
    assert len(images) == 5 + 5 + 5 + 7 + 1
    same_task_ref = (shown["tool_result_meta"]["mcp_response_source"]["read"]["arguments"], shown)

    def gaps_of(name, consumer, readers):
        gaps = []
        for arguments, row in readers:
            got = consumer.registry.execute_result("read_file", dict(arguments))
            target = (consumer.ctx.last_read_view or {}).get("target", "")
            digest = hashlib.sha256(pathlib.Path(target).read_bytes()).hexdigest() if target else ""
            if digest != row["tool_result_meta"]["mcp_response_source"]["sha256"]:
                gaps.append((name, "read_file", arguments, got.text[:200]))
        for path, expected in images:
            consumer.ctx.messages = []
            got = consumer.registry.execute_result("view_image", {"path": path})
            if _pixels(consumer.ctx.messages) != [expected]:
                gaps.append((name, "view_image", path, got.text[:200]))
        return gaps

    printed = [(_printed_read(shown["result"]), shown), (_printed_read(failed["result"]), failed)]
    gaps = gaps_of("live child", child, [*printed, same_task_ref])
    _retire(mcp, child.drive, trace)

    reopened, parent = _delegated_child(mcp, attempt=2), _context(mcp, mcp.data, PARENT)
    # A parent reaches the printed reader through the child's published retained calls, as it actually would.
    published = load_task_result(mcp.data, CHILD)["trace_refs"]["tool_call_refs"]
    fetched = [_late_text(parent.ctx, CHILD, row["manifest_ref"]) for row in published]
    assert [_printed_read(text) for text in fetched] == [arguments for arguments, _row in printed]
    assert _retained_paths(fetched[0]) == _retained_paths(shown["result"]) and _no_base64_text(fetched[0])
    for name, consumer, readers in (("reopened child", reopened, [*printed, same_task_ref]), ("parent", parent, printed)):
        prune_agent_media_uploads(mcp.data, retention_days=0, now=time.time() + 1)  # the age sweep of uploads/views
        gaps += gaps_of(name, consumer, readers)
    assert not gaps, json.dumps(gaps, indent=1, ensure_ascii=False)

    reopened.ctx.messages = []  # readable is not offered: a later attempt gets no automatic attachment
    stale = _maybe_auto_attach_image(
        _row(ToolResult(status="ok", code="OK", text=shown["result"], meta=shown["tool_result_meta"])),
        SimpleNamespace(_ctx=reopened.ctx), staged_messages=[])
    assert stale["status"] == "unavailable" and "another task attempt" in stale["note"] and reopened.ctx.messages == []


def test_server_text_meta_uris_and_names_grant_no_attachment(mcp, tmp_path):
    planted = mcp.data / "uploads" / "planted.png"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(PNG)
    offered = json.dumps({"ok": True, "auto_attach_image": str(planted)})
    mcp.peer.result = mcp_types.CallToolResult(content=[
        _text(offered),
        mcp_types.EmbeddedResource(type="resource", resource=mcp_types.TextResourceContents(
            uri=planted.as_uri(), mimeType="image/png", text="not a blob")),
        mcp_types.ResourceLink(type="resource_link", name="shot", uri=planted.as_uri(), mimeType="image/png"),
    ], **{"_meta": {"mcp_image_owner": [TASK, str(ATTEMPT)], "auto_attach_image": str(planted)}})
    result = mcp.registry.execute_result(TOOL, {})
    assert not {"mcp_image_owner", "mcp_response_source"} & set(result.meta)
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]
    # Control: the same offer from a first-party ext_ tool attaches, so the refusal above is the origin rule.
    extension = {"fn_name": "ext_1_r_fixture_shot", "tool_call_id": "ext", "result": offered, "is_error": False,
                 "tool_args": {}, "args_for_log": {}, "result_meta": {}}
    messages, _ = _batch(mcp, [extension])
    assert _pixels(messages) == [PNG]


def test_retention_failure_keeps_the_exact_sole_payload_inline_and_the_call_successful(mcp, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse)
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1
    assert SECRET in B64 and B64 in result.text, "the sole copy stays exact, unmasked inside its base64"
    assert "retention failed (OSError)" in result.text and result.meta["mcp_response_retention_gap"] == "OSError"
    assert "mcp_image_owner" not in result.meta
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]


@pytest.mark.parametrize("seam", ["_projected_body", "_protocol_view"], ids=["render", "render-and-retention"])
def test_a_projection_fault_after_the_remote_call_is_a_named_gap_never_a_failure(mcp, monkeypatch, seam):
    from ouroboros import mcp_result_images

    def fault(*_args, **_kwargs):
        raise RuntimeError("fault")

    monkeypatch.setattr(mcp_result_images, seam, fault)
    result = mcp.registry.execute_result(TOOL, {})
    assert (result.status, result.code) == ("ok", "OK") and mcp.peer.calls == 1 and "captured" in result.text
    assert "projection failed (RuntimeError)" in result.text and result.meta["mcp_response_projection_gap"] == "RuntimeError"
    assert "mcp_image_owner" not in result.meta
    if seam == "_projected_body":  # the exact bytes stay in the retained source and the printed reader names it
        retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
        assert retained["content"][1]["data"] == B64 and "read_file(path=" in result.text
    else:  # neither step held: the ordinary result is the sole copy, the retention gap named beside it
        assert result.meta["mcp_response_retention_gap"] == "RuntimeError" and "retention failed" in result.text
    messages, trace = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]


def test_only_an_image_or_embedded_blob_response_carries_its_sdk_value():
    from ouroboros.mcp_result_images import ReturnedResult, with_returned
    from ouroboros.tools.tool_result import ToolResult

    plain = ToolResult(status="ok", code="OK", text="adapted")
    for raw in (mcp_types.CallToolResult(content=[_text("plain")]), mcp_types.CallToolResult(content=[]),
                mcp_types.CallToolResult(content=[mcp_types.AudioContent(type="audio", mimeType="audio/wav", data="AAAA")])):
        assert with_returned(plain, raw) is plain
    for raw in (mcp_types.CallToolResult(content=[_image()]), mcp_types.CallToolResult(content=[_blob()]),
                mcp_types.CallToolResult(content=[_blob("AAECAw==", mime="application/octet-stream")])):
        carried = with_returned(plain, raw)
        assert type(carried) is ReturnedResult and carried.returned is raw and B64 not in repr(carried)


def test_configured_secret_is_masked_in_text_fields_never_inside_image_data(mcp):
    mcp.peer.result = mcp_types.CallToolResult(content=[_text(f"token {SECRET} echoed"), _image()],
                                               structuredContent={"echo": SECRET})
    result = mcp.registry.execute_result(TOOL, {})
    assert f"token {SECRET} echoed" not in result.text and "token <redacted:mcp-auth-token> echoed" in result.text
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert retained["content"][1]["data"] == B64
    assert retained["content"][0]["text"] == "token <redacted:mcp-auth-token> echoed"
    assert retained["structuredContent"] == {"echo": "<redacted:mcp-auth-token>"}
    messages, _ = _batch(mcp, [_row(result)])
    assert _pixels(messages) == [PNG]


def _with_secret(env, token):
    """``token`` becomes the server's configured secret."""
    from ouroboros import config

    settings = config.load_settings()
    settings["MCP_SERVERS"][0]["auth_token"] = token
    env.manager.reconfigure(settings)
    assert env.manager.refresh_server("shots")["ok"]


@pytest.mark.parametrize("token,embedded", [
    ("image", False), ("imag", False), ("mage", False), ("text", False), ("data", False),
    ("image", True), ("blob", True)])
def test_protocol_vocabulary_survives_a_colliding_secret_while_its_free_text_is_masked(mcp, token, embedded):
    _with_secret(mcp, token)
    mcp.peer.result = mcp_types.CallToolResult(content=[_text(f"echo {token}"), _blob() if embedded else _image()])
    result = mcp.registry.execute_result(TOOL, {})
    assert result.status == "ok" and f"echo {token}" not in result.text
    text, part = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))["content"]
    assert (text["type"], text["text"]) == ("text", "echo <redacted:mcp-auth-token>")
    holder = part["resource"] if embedded else part
    assert (part["type"], holder["mimeType"], holder["blob" if embedded else "data"]) == (
        "resource" if embedded else "image", "image/png", B64)
    assert token not in holder.get("uri", "")  # the resource's free text is still masked
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(1, "attached")]) and _pixels(messages) == [PNG]


def test_server_added_fields_stay_masked_data_and_classify_nothing(mcp):
    secret = "s3cr3t-value"
    _with_secret(mcp, secret)
    image = mcp_types.ImageContent(type="image", mimeType=f"image/png; name={secret}", data=B64,
                                   **{"_meta": {secret: secret}, f"x-{secret}": f"echo {secret}"})
    decoy = mcp_types.TextContent(type="text", text="decoy", **{"data": B64, "mimeType": "image/png"})
    mcp.peer.result = mcp_types.CallToolResult(content=[decoy, image], **{f"note-{secret}": secret})
    result = mcp.registry.execute_result(TOOL, {})
    retained = json.loads(artifacts.read_actor_source_bytes(mcp.data, TASK, result.meta["mcp_response_source"]))
    assert secret not in json.dumps(retained) and secret not in result.text  # keys and values a server added
    part = retained["content"][1]
    assert (part["type"], part["mimeType"], part["data"]) == ("image", "image/png; name=<redacted:mcp-auth-token>", B64)
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(1, "attached")]) and _pixels(messages) == [PNG]


def test_a_result_settling_after_the_outer_wait_is_published_once_and_never_attached(mcp, monkeypatch):
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.task_results import write_task_result
    from tests.test_late_tool_result_delivery import _late_text, _settled

    write_task_result(mcp.data, TASK, "running")
    monkeypatch.setattr(mcp.registry, "get_timeout", lambda _name: 1)
    monkeypatch.setattr("ouroboros.loop_tool_execution.load_settings", lambda: {})
    mcp.peer.delay = 2.5
    logs = mcp.data / "logs"  # the task's own drive logs, as the loop passes them
    call = {"id": "call-late", "type": "function", "function": {"name": TOOL, "arguments": "{}"}}
    messages = [{"role": "user", "content": "look"}, {"role": "assistant", "content": "", "tool_calls": [call]}]
    handle_tool_calls([call], mcp.registry, logs, TASK, StatefulToolExecutor(), messages, {"tool_calls": []},
                      lambda *_a, **_k: None)
    assert "TOOL_TIMEOUT" in messages[-1]["content"]
    before = copy.deepcopy(messages)
    late = _late_text(mcp.ctx, TASK, _settled(logs, tool=TOOL)["result_ref"])
    assert mcp.peer.calls == 1 and messages == before and _pixels(messages) == []
    assert "MCP response retained exactly" in late and _no_base64_text(late)
    assert [pathlib.Path(path).read_bytes() for path in _retained_paths(late)] == [PNG]


def test_taskless_and_contextless_calls_keep_the_ordinary_projection(mcp):
    expected = _ordinary(mcp, mcp.peer.result)
    assert mcp.manager.call_tool(TOOL, {}) == expected
    mcp.ctx.task_id = None
    result = mcp.registry.execute_result(TOOL, {})
    assert result.text == expected and "mcp_response_source" not in result.meta
    assert not list(mcp.data.rglob("source_handles"))


@pytest.mark.parametrize("raw", [
    mcp_types.CallToolResult(content=[_text("plain λ")]),
    mcp_types.CallToolResult(content=[_text("ordinary error")], isError=True),
    mcp_types.CallToolResult(content=[]),
], ids=["text", "error", "empty"])
def test_responses_without_protocol_images_keep_the_ordinary_projection(mcp, raw):
    mcp.peer.result = raw
    result = mcp.registry.execute_result(TOOL, {})
    assert result.text == _ordinary(mcp, raw) and result.producer_text is None and result.host_annotations == ()
    assert result.status == ("error" if raw.isError else "ok") and "mcp_response_source" not in result.meta
    assert not list(mcp.data.rglob("source_handles"))
