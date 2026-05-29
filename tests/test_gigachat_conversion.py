"""Unit tests for the GigaChat message/tool/response converters in llm.py.

These exercise the pure conversion helpers (no network, no auth) that bridge
the OpenAI-shaped agent loop and the native ``gigachat`` library, including the
role mapping (tool→function), the tool_calls→function_call collapse, the
JSON-wrapped function results, the system-message demotion, and the response
normalization back to OpenAI shape.
"""

import json

from gigachat.models import ChatCompletion, Choices, FunctionCall, Messages, Usage

from ouroboros.llm import LLMClient


def test_gigachat_messages_role_and_tool_mapping():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "lookup", "arguments": '{"q": "x"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "result-text"},
    ]

    out = LLMClient._gigachat_messages(messages)

    assert out[0] == {"role": "system", "content": "sys"}
    # multipart content flattened to a plain string
    assert out[1] == {"role": "user", "content": "hi"}
    # assistant tool_calls collapsed to a single function_call with parsed args
    assert out[2]["role"] == "assistant"
    assert out[2]["function_call"] == {"name": "lookup", "arguments": {"q": "x"}}
    # tool result became a function-role message named after the originating call;
    # plain text is JSON-wrapped because GigaChat requires a valid JSON result.
    assert out[3]["role"] == "function"
    assert out[3]["name"] == "lookup"
    assert json.loads(out[3]["content"]) == {"result": "result-text"}


def test_gigachat_function_result_json_wrapping():
    # Plain text → wrapped as {"result": ...} (GigaChat parses results as JSON).
    assert json.loads(LLMClient._gigachat_function_result("plain text")) == {"result": "plain text"}
    # Already-valid JSON passes through unchanged.
    assert LLMClient._gigachat_function_result('{"a": 1}') == '{"a": 1}'
    assert LLMClient._gigachat_function_result("[1, 2]") == "[1, 2]"


def test_gigachat_messages_demotes_non_leading_system():
    # GigaChat rejects a system message that isn't first. The agent injects
    # system-reminders mid-conversation; these must be demoted to user.
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "do it"},
        {"role": "system", "content": "<system-reminder>be concise</system-reminder>"},
    ]
    out = LLMClient._gigachat_messages(messages)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "user"]
    # the demoted reminder keeps its content
    assert "be concise" in out[2]["content"]


def test_gigachat_messages_collapses_parallel_tool_calls_to_first():
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "first", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "second", "arguments": "{}"}},
        ],
    }]

    out = LLMClient._gigachat_messages(messages)

    assert out[0]["function_call"]["name"] == "first"


def test_gigachat_functions_strips_cache_control():
    tools = [{
        "type": "function",
        "function": {
            "name": "do",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "cache_control": {"type": "ephemeral"},
            },
        },
    }]

    functions = LLMClient._gigachat_functions(tools)

    assert functions == [{
        "name": "do",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
    }]


def test_gigachat_functions_adds_properties_to_bare_object_nodes():
    """GigaChat 422s on any ``object`` schema node missing ``properties`` (e.g.
    a free-form ``evidence`` object). The converter must inject ``properties: {}``
    recursively — top-level, nested, inside array items, and combinators."""
    tools = [{
        "type": "function",
        "function": {
            "name": "review",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence": {"type": "object", "description": "free-form blob"},
                    "items": {"type": "array", "items": {"type": "object"}},
                    "choice": {"anyOf": [{"type": "object"}, {"type": "string"}]},
                },
                "required": ["evidence"],
            },
        },
    }]

    params = LLMClient._gigachat_functions(tools)[0]["parameters"]

    assert params["properties"]["evidence"]["properties"] == {}
    assert params["properties"]["items"]["items"]["properties"] == {}
    assert params["properties"]["choice"]["anyOf"][0]["properties"] == {}
    # An object that already declares properties is left intact.
    assert params["properties"]["choice"]["anyOf"][1] == {"type": "string"}


def test_gigachat_sanitize_schema_drops_nested_cache_control():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "object", "cache_control": {"type": "ephemeral"}}},
    }
    out = LLMClient._gigachat_sanitize_schema(schema)
    assert "cache_control" not in out["properties"]["a"]
    assert out["properties"]["a"]["properties"] == {}


def _target():
    return {
        "provider": "gigachat",
        "usage_model": "gigachat/GigaChat-3-Ultra",
        "resolved_model": "GigaChat-3-Ultra",
    }


def test_normalize_text_response():
    completion = ChatCompletion(
        choices=[Choices(message=Messages(role="assistant", content="Hello"), index=0, finish_reason="stop")],
        created=1,
        model="GigaChat-3-Ultra",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, precached_prompt_tokens=4),
        object="chat.completion",
    )

    message, usage = LLMClient()._normalize_gigachat_response(completion, _target())

    assert message == {"role": "assistant", "content": "Hello"}
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["cached_tokens"] == 4
    assert usage["provider"] == "gigachat"
    assert usage["resolved_model"] == "gigachat/GigaChat-3-Ultra"
    assert "cost" in usage


def test_normalize_function_call_response():
    completion = ChatCompletion(
        choices=[Choices(
            message=Messages(role="assistant", content="", function_call=FunctionCall(name="do", arguments={"a": 1})),
            index=0,
            finish_reason="function_call",
        )],
        created=1,
        model="GigaChat-3-Ultra",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        object="chat.completion",
    )

    message, _ = LLMClient()._normalize_gigachat_response(completion, _target())

    assert message["content"] is None
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "do"
    # arguments re-encoded as a JSON string (round-trips back to the dict)
    assert json.loads(call["function"]["arguments"]) == {"a": 1}
