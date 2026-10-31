"""Adversarial regressions for direct-OpenAI Chat Phase 2A seams."""

from __future__ import annotations

import json
import pathlib

import ouroboros.openai_chat_dispatch as dispatch
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.request_wire_receipts import (
    WireCandidateSpec,
    bind_wire_candidate,
)
from ouroboros.utils import estimate_tokens


def _target():
    return {
        "provider": "openai",
        "resolved_model": "future-model-without-prefix",
        "usage_model": "openai/future-model-without-prefix",
        "base_url": "https://api.openai.com/v1",
    }


def _read_tool():
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one allowed path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"const": "allowed"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def _custom_call(call_id, name, raw_input):
    return {
        "id": call_id,
        "type": "custom",
        "custom": {"name": name, "input": raw_input},
        "function": None,
    }


class _HistoryError(Exception):
    def __init__(self):
        message = (
            "A custom tool call in an assistant message is not compatible "
            "without a matching tool result"
        )
        super().__init__(message)
        self.status_code = 400
        self.code = "invalid_request_error"
        self.param = "messages[3].tool_call_id"
        self.body = {"error": {
            "status_code": 400,
            "code": self.code,
            "param": self.param,
            "message": message,
        }}


def _dispatch_source():
    return {
        "model": "future-model-without-prefix",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "reasoning_effort": "medium",
        "tool_choice": "auto",
        "tools": [_read_tool()],
    }


def _initial_custom_candidate():
    source = _dispatch_source()
    return bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "medium", "requested_wire_form",
        ),
        requested_effort="medium",
        ladder_ordinal=1,
    )


def test_history_exception_cannot_advance_the_dialect_ladder():
    candidate = _initial_custom_candidate()
    error = _HistoryError()
    assert dispatch.exact_tool_dialect_rejection(error, candidate) is False
    assert dispatch.plan_direct_openai_dialect_candidate(
        target=_target(),
        source_payload=_dispatch_source(),
        current=candidate,
        error=error,
        body_error=False,
    ) is None


def test_history_body_error_cannot_advance_the_dialect_ladder():
    candidate = _initial_custom_candidate()
    error = _HistoryError().body["error"]
    assert dispatch.plan_direct_openai_dialect_candidate(
        target=_target(),
        source_payload=_dispatch_source(),
        current=candidate,
        error=error,
        body_error=True,
    ) is None


def _serialized_projection(messages, tools):
    return json.dumps(
        {"messages": messages, "tools": tools},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _projection_measures(projections):
    serialized = [
        _serialized_projection(messages, tools)
        for messages, tools in projections
    ]
    return (
        [len(item.encode("utf-8")) for item in serialized],
        [estimate_tokens(item) for item in serialized],
    )


def test_localized_109_tools_max_each_admission_metric_independently():
    localized = [
        {
            "type": "function",
            "function": {
                "name": f"localized_{index:03d}",
                "description": "Run the localized operation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "я" * 315,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
        for index in range(109)
    ]
    messages = [{"role": "user", "content": "x"}]
    projections = dispatch.direct_openai_context_projections(
        messages,
        localized,
        provider="openai",
        reasoning_effort="medium",
    )
    byte_sizes, token_sizes = _projection_measures(projections)

    assert len(projections) == 2
    assert projections[0][1][0]["type"] == "function"
    assert projections[1][1][0]["type"] == "custom"
    assert byte_sizes[0] > byte_sizes[1]
    assert token_sizes[1] > token_sizes[0]
    assert dispatch.projected_context_size_bytes(
        messages,
        localized,
        provider="openai",
        reasoning_effort="medium",
    ) == max(byte_sizes)
    assert estimate_context_prompt_tokens(
        messages,
        localized,
        provider="openai",
        reasoning_effort="medium",
    ) == max(token_sizes)


def test_current_tool_registry_exposes_and_measures_both_projections(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_smoke import EXPECTED_TOOLS

    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    tools = ToolRegistry(repo_dir=repo_dir, drive_root=tmp_path).schemas()
    expected_names = set(EXPECTED_TOOLS)
    assert {tool["function"]["name"] for tool in tools} == expected_names
    assert len(tools) == len(expected_names)
    messages = [{"role": "user", "content": "x"}]
    projections = dispatch.direct_openai_context_projections(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    )
    byte_sizes, token_sizes = _projection_measures(projections)

    assert len(projections) == 2
    for _messages, projected_tools in projections:
        assert {tool[tool["type"]]["name"] for tool in projected_tools} == expected_names
        assert len(projected_tools) == len(expected_names)
    assert dispatch.projected_context_size_bytes(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    ) == max(byte_sizes)
    assert estimate_context_prompt_tokens(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    ) == max(token_sizes)


def test_context_sizing_uses_canonical_projection_for_unknown_historical_tool():
    tools = [_read_tool()]
    messages = [
        {"role": "user", "content": "Use a tool."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-unknown",
                "type": "function",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-unknown",
            "content": "Unknown tool: unknown_tool",
        },
    ]

    projections = dispatch.direct_openai_context_projections(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    )

    assert projections == ((messages, tools),)
    assert dispatch.projected_context_size_bytes(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    ) > 0
    assert estimate_context_prompt_tokens(
        messages,
        tools,
        provider="openai",
        reasoning_effort="medium",
    ) > 0
