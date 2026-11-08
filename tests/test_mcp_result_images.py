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
    messages, trace = _batch(mcp, [_row(result)])
    assert _items(trace) == ("attached", [(1, "attached"), (2, "attached")]) and _pixels(messages) == [PNG, second]


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


def _retire(env, drive, trace):
    """The host's whole terminal path: copy-back, ref promotion, settlement and trash removal."""
    from ouroboros import headless, task_custody
    from ouroboros.task_results import write_task_result

    write_task_result(drive, CHILD, "completed", result="done", artifact_status="ready", trace_refs={
        "tool_call_refs": [{"manifest_ref": row["trace_ref"]["manifest_ref"]} for row in trace["tool_calls"]]})
    headless.copy_child_task_result(env.data, {"id": CHILD, "drive_root": str(drive)})
    promoted = headless.retry_child_task_refs(env.data, drive, CHILD)
    assert promoted["child_ref_promotion"]["status"] == "complete", promoted["child_ref_promotion"]
    assert task_custody.settle_child_drive(env.data, CHILD, drive, live=lambda _: False)["status"] == "removed"
    assert task_custody.remove_custody_trash(env.data)["removed"] == 1
    assert not drive.exists()


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


def test_only_an_image_bearing_response_carries_its_sdk_value():
    from ouroboros.mcp_result_images import ReturnedResult, with_returned
    from ouroboros.tools.tool_result import ToolResult

    plain = ToolResult(status="ok", code="OK", text="adapted")
    for raw in (mcp_types.CallToolResult(content=[_text("plain")]), mcp_types.CallToolResult(content=[]),
                mcp_types.CallToolResult(content=[_blob("AAECAw==", mime="application/octet-stream")]),
                mcp_types.CallToolResult(content=[mcp_types.AudioContent(type="audio", mimeType="audio/wav", data="AAAA")])):
        assert with_returned(plain, raw) is plain
    for raw in (mcp_types.CallToolResult(content=[_image()]), mcp_types.CallToolResult(content=[_blob()])):
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
