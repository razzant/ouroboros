"""A task session's protocol images reach the model exactly as a per-call server's do.

One real SDK stdio server script, configured held (``task_session``) and per call. Every call goes
through the real ToolRegistry, the loop's tool executor and its batch; the OpenAI-compatible request
crosses loopback HTTP to the image suite's recorder, whose decoded bytes are compared with the PNG the
server sent, and the Anthropic request and Claudexor upload are checked at their payload builders.
The per-call projection matrix lives in ``test_mcp_result_images.py``; here the held arm walks its
outcomes through one session, which outlives each of them and is never asked twice.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, mcp_result_images, mcp_task_session
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run
from tests.test_mcp_result_images import PNG, Recorder, _batch, _items, _no_base64_text, _pixels
from tests.test_mcp_task_session import _private_mcp, alive, rows, session_of, wire  # noqa: F401  (autouse fixture)


MODEL = "openrouter::fixture/vision"
SERVER = r'''
import base64, json, os, sys
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

LOG, IMAGE = sys.argv[1], base64.b64encode(open(sys.argv[2], "rb").read()).decode()
server = Server("held-image-fixture")
PARTS = [types.TextContent(type="text", text="captured"),
         types.ImageContent(type="image", mimeType="image/png", data=IMAGE)]


def note(**row):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), **row}) + "\n")


@server.list_tools()
async def tools():
    return [types.Tool(name=name, description=name, inputSchema={"type": "object"}) for name in ("capture", "refuse", "plain")]


@server.call_tool()
async def call(name, _arguments):
    note(event="call", tool=name)
    if name == "capture":
        return PARTS
    if name == "refuse":
        return types.CallToolResult(isError=True, content=PARTS)
    return [types.TextContent(type="text", text="no image here")]


async def main():
    note(event="start", task_scope=any(key.startswith("OURO_TASK_SCOPE_") for key in os.environ))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''


@pytest.fixture
def registry(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})  # usage reservation stays local
    yield registry
    mcp_task_session.close_context_sessions(registry._ctx)  # idempotent: a failed assertion leaves no server behind


def image_server(tmp_path, server_id, **extra):
    script, png, log = tmp_path / "held_image_server.py", tmp_path / "shot.png", tmp_path / f"{server_id}.jsonl"
    script.write_text(SERVER, encoding="utf-8")
    png.write_bytes(PNG)
    return {"id": server_id, "enabled": True, "transport": "stdio", "command": sys.executable,
            "args": [str(script), str(log), str(png)], "cwd": str(tmp_path), **extra}, log


def invoke(registry, tool, server_id="held"):
    return run(registry, name=f"mcp_{server_id}__{tool}", args={"n": 1}, timeout=40)


def physical_request(registry, messages, recorder):
    """The batch's history through the send projection and the real client, onto the recorder's socket."""
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import _prepare_main_messages

    ctx = registry._ctx
    llm = LLMClient(api_key="fixture-key", base_url=recorder.base_url)
    sent = _prepare_main_messages(messages, model=MODEL, llm=llm, accumulated_usage={}, drive_root=ctx.drive_root,
                                  task_id=ctx.task_id, task_attempt=ctx.task_attempt, event_queue=None, use_local=False)
    llm.chat(sent, model=MODEL, tools=None, max_tokens=16, no_proxy=True, timeout=20)
    return recorder.bodies[-1]


def test_held_and_per_call_images_reach_the_physical_request_as_the_same_pixels(tmp_path, monkeypatch, registry):
    from ouroboros import llm_claudexor
    from ouroboros.llm import LLMClient

    held_raw, held_log = image_server(tmp_path, "held", task_session=True)
    plain_raw, plain_log = image_server(tmp_path, "plain")
    wire(monkeypatch, registry, held_raw, plain_raw)
    env, sources = SimpleNamespace(registry=registry), {}
    with owner(registry), Recorder() as recorder:
        for server_id in ("held", "plain"):
            for _round in range(2):
                row = invoke(registry, "capture", server_id)
                result = row["tool_result"]
                assert (result.status, result.code) == ("ok", "OK") and "captured" in result.text
                assert _no_base64_text(result.text) and "retained as" in result.text
                assert result.meta["mcp_image_owner"] == [registry._ctx.task_id, "1"]
                messages, trace = _batch(env, [row])
                assert _items(trace) == ("attached", [(1, "attached")])
                wire_body = physical_request(registry, messages, recorder)
                assert [hashlib.sha256(image).hexdigest() for image in _pixels(wire_body)] == [hashlib.sha256(PNG).hexdigest()]
                assert _no_base64_text(json.dumps(wire_body)), "protocol image bytes never travel as model-facing text"
                sources.setdefault(server_id, result.meta["mcp_response_source"]["sha256"])
                _system, anthropic = LLMClient(api_key="fixture-key")._build_anthropic_messages(copy.deepcopy(messages))
                upload = llm_claudexor._request({"source": "codex", "resolved_model": "fixture"},
                                                copy.deepcopy(messages), None, {})
                encoded = json.dumps(upload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
                assert _pixels(anthropic) == [PNG] and _pixels(json.loads(encoded)) == [PNG] and _no_base64_text(encoded)
    assert sources["held"] == sources["plain"], "one response, one retained source, whichever arm carried it"
    held_calls, plain_calls = rows(held_log, "call"), rows(plain_log, "call")
    assert len(held_calls) == 2 and len({row["pid"] for row in held_calls}) == 1
    assert len(plain_calls) == 2 and len({row["pid"] for row in plain_calls}) == 2
    assert rows(held_log, "start")[-1]["task_scope"] is True and not rows(plain_log, "start")[-1]["task_scope"]
    assert not claims(tmp_path)  # every answered call discharged its MCP duty
    mcp_task_session.close_context_sessions(registry._ctx)
    assert session_of(registry, "held").closure == "closed" and not alive(held_calls[0]["pid"])


def test_each_held_image_outcome_keeps_its_answer_and_the_session(tmp_path, monkeypatch, registry):
    raw, log = image_server(tmp_path, "held", task_session=True)
    wire(monkeypatch, registry, raw)
    env = SimpleNamespace(registry=registry)

    def unattached(row):
        messages, trace = _batch(env, [row])
        assert _pixels(messages) == [] and "image_attachment" not in trace["tool_calls"][0]

    def fault(*_args, **_kwargs):
        raise OSError("disk full")

    with owner(registry):
        refused = invoke(registry, "refuse")  # a server-answered error keeps its status; its image is only a file
        result = refused["tool_result"]
        assert (result.status, result.code) == ("error", "MCP_ERROR") and _no_base64_text(result.text)
        assert "mcp_image_owner" not in result.meta and result.meta["mcp_response_source"] and "retained as" in result.text
        unattached(refused)

        plain = invoke(registry, "plain")  # no protocol image: the ordinary projection, nothing retained
        result = plain["tool_result"]
        assert result.status == "ok" and "no image here" in result.text and "mcp_response_source" not in result.meta
        assert result.producer_text is None and not hasattr(result, "returned")
        unattached(plain)

        with monkeypatch.context() as patch:  # the sole copy stays inline, the gap named; a completed call never fails
            patch.setattr(artifacts, "store_actor_source_bytes", fault)
            lost = invoke(registry, "capture")
        result = lost["tool_result"]
        assert (result.status, result.code) == ("ok", "OK") and result.meta["mcp_response_retention_gap"] == "OSError"
        assert "retention failed (OSError)" in result.text and not _no_base64_text(result.text)
        assert "mcp_image_owner" not in result.meta
        unattached(lost)

        with monkeypatch.context() as patch:  # a projection fault after the call is a named gap, outside the transport catch
            patch.setattr(mcp_result_images, "_projected_body", fault)
            broken = invoke(registry, "capture")
        result = broken["tool_result"]
        assert (result.status, result.code) == ("ok", "OK") and result.meta["mcp_response_projection_gap"] == "OSError"
        assert "mcp_call_not_sent" not in result.meta and "mcp_image_owner" not in result.meta
        unattached(broken)

        after = invoke(registry, "capture")  # the same session still answers, and now attaches
        messages, trace = _batch(env, [after])
        assert _items(trace) == ("attached", [(1, "attached")]) and _pixels(messages) == [PNG]
    calls = rows(log, "call")
    assert [row["tool"] for row in calls] == ["refuse", "plain", "capture", "capture", "capture"], "nothing was resent"
    assert len({row["pid"] for row in calls}) == 1 and session_of(registry, "held").usable()
    assert not claims(tmp_path)
    mcp_task_session.close_context_sessions(registry._ctx)
    assert session_of(registry, "held").closure == "closed" and not alive(calls[0]["pid"])
