"""OpenRouter function tools carry an explicit ``strict: false`` (issue #1411).

OpenAI models behind OpenRouter are served by the Responses API, which normalizes a
function tool WITHOUT ``strict`` into strict mode and forces every optional property.
The OpenRouter lane sets ``false`` only where the key is absent, on its own copy of the
catalog, before request-wire identity is bound. Caller values, the server web-search
tool, tool-free requests, every other route and the explicit arguments a model returns
stay exactly as they were.
"""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros.llm import LLMClient
from ouroboros.request_wire_contract import build_request_wire_profile, function_strictness
from tests.provider_contract_catalog import shipped_builtin_tool_schemas
from tests.test_llm_provider_golden import _observe

_OPENAI_ON_OPENROUTER = "openai/gpt-5.6-luna"
_SERVER_WEB_TOOL = {"type": "openrouter:web_search", "parameters": {"max_total_results": 3}}


def _tool(name, **function_fields):
    return {"type": "function", "function": {
        "name": name,
        "description": f"{name} fixture",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "note": {"type": "string"},
                "verbose": {"type": "boolean", "default": False},
                "scope": {"type": "string", "enum": ["own_binding"]},
            },
            "required": ["task_id"],
        },
        **function_fields,
    }}


def _build(client, model, tools, **options):
    target = client._resolve_remote_target(model)
    return target, client._build_remote_kwargs(
        target, [{"role": "user", "content": "call once"}], "medium", 256, "auto", None,
        tools, skip_capability_fetch=True, **options,
    )


def _functions(payload):
    return {
        tool["function"]["name"]: tool["function"]
        for tool in payload.get("tools") or []
        if isinstance(tool.get("function"), dict)
    }


def test_openrouter_sets_false_only_where_strict_is_absent(monkeypatch):
    monkeypatch.setattr(
        LLMClient, "_openrouter_main_web_search_tool", staticmethod(lambda: dict(_SERVER_WEB_TOOL)),
    )
    tools = [
        {**_tool("read_task"), "cache_control": {"type": "ephemeral"}},
        _tool("explicit_true", strict=True),
        _tool("explicit_false", strict=False),
        _tool("explicit_null", strict=None),
    ]
    before = copy.deepcopy(tools)
    _target, payload = _build(
        LLMClient(api_key="test"), _OPENAI_ON_OPENROUTER, tools, allow_server_web_search=True,
    )

    assert tools == before, "the caller's catalog must never be mutated"
    functions = _functions(payload)
    assert functions["read_task"]["strict"] is False
    assert functions["explicit_true"]["strict"] is True
    assert functions["explicit_false"]["strict"] is False
    assert "strict" in functions["explicit_null"] and functions["explicit_null"]["strict"] is None
    for name, function in functions.items():
        source = next(tool["function"] for tool in before if tool["function"]["name"] == name)
        assert function["parameters"] == source["parameters"], name
        assert function is not source
    assert payload["tools"][-1] == _SERVER_WEB_TOOL
    assert function_strictness({"tools": before}) == "absent+false+true+malformed"
    assert function_strictness(payload) == "false+true+malformed"


@pytest.mark.parametrize("model", [
    "openai::gpt-5.6-terra",
    "deepseek::deepseek-v4-flash",
    "openai-compatible::local-model",
])
def test_other_routes_keep_their_function_tool_wire(model):
    tools = [_tool("read_task")]
    _target, payload = _build(LLMClient(api_key="test"), model, copy.deepcopy(tools))
    assert "strict" not in _functions(payload)["read_task"]
    assert function_strictness(payload) == "absent"


def test_tool_free_openrouter_request_is_unchanged():
    _target, payload = _build(LLMClient(api_key="test"), _OPENAI_ON_OPENROUTER, None)
    assert "tools" not in payload
    assert function_strictness(payload) == "none"


def test_full_registry_keeps_every_schema_and_explicit_argument_semantics():
    """Negative controls: only the wire flag changes, never a declared default, enum,
    required list or an explicit value the model returns (no neutral-value stripping)."""
    registry = shipped_builtin_tool_schemas()
    client = LLMClient(api_key="test")
    target, payload = _build(client, _OPENAI_ON_OPENROUTER, copy.deepcopy(registry))
    functions = _functions(payload)
    assert set(functions) == {tool["function"]["name"] for tool in registry}
    for tool in registry:
        sent = dict(functions[tool["function"]["name"]])
        assert sent.pop("strict") is False
        assert sent == client._sanitize_chat_completion_tools([tool])[0]["function"]
    assert functions["schedule_subagent"]["parameters"]["properties"]["may_fan_out"]["default"] is True
    assert "none" in functions["switch_model"]["parameters"]["properties"]["effort"]["enum"]

    explicit = {
        "schedule_subagent": {"may_fan_out": False, "allowed_origins": []},
        "compact_context": {"keep_unit_ids": []},
        "switch_model": {"effort": "none"},
        "skill_review": {"author_disposition": "accepted", "author_rationale": "reviewed"},
    }
    calls = [
        {"id": f"call-{name}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)}}
        for name, arguments in explicit.items()
    ]
    message, _usage = client._normalize_remote_response({
        "id": "gen-explicit",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": copy.deepcopy(calls),
        }}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }, target, skip_cost_fetch=True)
    returned = {call["function"]["name"]: call["function"]["arguments"] for call in message["tool_calls"]}
    assert returned == {call["function"]["name"]: call["function"]["arguments"] for call in calls}


def _response_step():
    return {"kind": "response", "body": {
        "id": "gen-strictness",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "ok", "refusal": None, "annotations": None,
        }}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 4, "cost": 0.001},
    }}


def _spec(method, transport, **kwargs):
    return {
        "env": {"OPENROUTER_API_KEY": "or-fixture-key"},
        "transport": transport,
        "call": {"kind": "method", "name": method, "kwargs": {
            "model": _OPENAI_ON_OPENROUTER,
            "messages": [{"role": "user", "content": "call once"}],
            "tools": [_tool("read_task"), _tool("explicit_true", strict=True)],
            **kwargs,
        }},
    }


def _assert_sealed_false_sends(observed, expected_sends):
    assert "raised" not in observed, observed.get("raised")
    sends, attempts = observed["sends"], observed["physical_attempts"]
    assert len(sends) == len(attempts) == expected_sends
    target = LLMClient(api_key="or-fixture-key")._resolve_remote_target(_OPENAI_ON_OPENROUTER)
    for send, attempt in zip(sends, attempts):
        functions = _functions(send["payload"])
        assert functions["read_task"]["strict"] is False
        assert functions["explicit_true"]["strict"] is True
        assert functions["read_task"]["parameters"] == _tool("read_task")["function"]["parameters"]
        # The bytes handed to the transport are the sealed physical candidate.
        assert send["payload_sha256"] == attempt["candidate_raw_sha256"]
    wire = observed["returned"]["usage"]["request_wire"]
    assert wire["candidate_sha256"] == sends[-1]["payload_sha256"]
    profile = build_request_wire_profile(
        target, sends[-1]["payload"], api_surface="chat.completions",
    )
    assert profile.function_strictness == "false+true"
    assert wire["source_profile_fingerprint"] == wire["accepted_profile_fingerprint"] == profile.fingerprint
    return wire


@pytest.mark.parametrize(("no_proxy", "method"), [(False, "chat"), (True, "chat_async")])
def test_physical_send_seals_explicit_false(method, no_proxy):
    observed = _observe(_spec(method, [_response_step()], no_proxy=no_proxy))
    wire = _assert_sealed_false_sends(observed, 1)
    assert wire["ladder_ordinal"] == 1 and wire["applied_actions"] == []


@pytest.mark.parametrize("method", ["chat", "chat_async"])
def test_wire_recovery_retry_keeps_explicit_false_sealed(method):
    rejection = {
        "kind": "error", "status_code": 400,
        "message": "temperature: unsupported parameter for this endpoint",
    }
    observed = _observe(_spec(method, [rejection, _response_step()], temperature=0.7))
    wire = _assert_sealed_false_sends(observed, 2)
    first, retried = (send["payload"] for send in observed["sends"])
    assert "temperature" in first and "temperature" not in retried
    assert [item["action"]["fields"] for item in wire["applied_actions"]] == [["temperature"]]
