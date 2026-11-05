"""The shared keyless model speaks the response protocol Main requests."""

from __future__ import annotations

import json

import pytest
from openai import OpenAI

from ouroboros.llm_stream import consume_stream
from tests.fixtures_e2e_cancellation import StubModelServer
from tests.system_e2e.harness import ScriptedStubModel


@pytest.mark.parametrize(("tool_call", "stream"), [(False, False), (True, True)])
def test_loopback_model_round_trips_through_the_real_sdk(stream, tool_call):
    script = [{"tool": "lookup", "arguments": {"query": "preserve both sides"}}] if tool_call else []
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    }}}]
    with ScriptedStubModel(script, final_answer="The complete final answer.") as model:
        with OpenAI(base_url=model.base_url, api_key="keyless-fixture", max_retries=0) as client:
            response = client.chat.completions.create(
                model="mock-model", messages=[{"role": "user", "content": "Execute the fixture."}],
                tools=tools, stream=stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            )
            result = consume_stream(response).model_dump() if stream else response.model_dump()
        assert len(model.calls) == 1 and model.calls[0][1]["stream"] is stream
        assert model.script_consumed()
    message = result["choices"][0]["message"]
    if tool_call:
        call = message["tool_calls"][0]
        assert call["id"] == "call_1" and call["type"] == "function"
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == {"query": "preserve both sides"}
    else:
        assert message["content"] == "The complete final answer."
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15


@pytest.mark.parametrize(("tool_call", "stream"), [(False, False), (True, True)])
def test_cancellation_model_round_trips_through_the_real_sdk(stream, tool_call):
    tools = [{"type": "function", "function": {"name": "list_files", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False,
    }}}]
    with StubModelServer(mode="keepalive" if tool_call else "finish") as model:
        with OpenAI(base_url=model.base_url, api_key="keyless-fixture", max_retries=0) as client:
            response = client.chat.completions.create(
                model="mock-model", messages=[{"role": "user", "content": "Execute the fixture."}],
                tools=tools, stream=stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
            )
            result = consume_stream(response).model_dump() if stream else response.model_dump()
        assert len(model.calls) == 1 and model.calls[0]["stream"] is stream
    message = result["choices"][0]["message"]
    if tool_call:
        call = message["tool_calls"][0]
        assert call["id"] == "call_1" and call["type"] == "function"
        assert call["function"]["name"] == "list_files"
        assert json.loads(call["function"]["arguments"]) == {"path": "."}
    else:
        assert message["content"] == "Done."
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
