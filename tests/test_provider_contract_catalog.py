"""Secretless full-registry provider-schema portability gate."""

from __future__ import annotations

import pytest

from tests.provider_contract_catalog import (
    ROOT_SCHEMA_COMBINATORS,
    assert_portable_tool_schemas,
    shipped_builtin_tool_schemas,
)


def _function_names(tools):
    return [tool["function"]["name"] for tool in tools]


def test_shipped_builtin_catalog_is_complete_sorted_and_deterministic():
    first = shipped_builtin_tool_schemas()
    second = shipped_builtin_tool_schemas()

    names = _function_names(first)
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert len(names) >= 100
    assert "delegate_start" in names
    assert first == second


def test_shipped_builtin_catalog_does_not_refresh_mcp_or_load_extensions(monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.mcp_client as mcp_client
    from ouroboros.tools.registry import ToolRegistry

    def _dynamic_discovery_forbidden(*_args, **_kwargs):
        raise AssertionError("dynamic extension/MCP discovery must not run")

    monkeypatch.setattr(ToolRegistry, "schemas", _dynamic_discovery_forbidden)
    monkeypatch.setattr(
        mcp_client,
        "ensure_configured_from_settings",
        _dynamic_discovery_forbidden,
    )
    monkeypatch.setattr(
        extension_loader,
        "is_extension_live",
        _dynamic_discovery_forbidden,
    )
    monkeypatch.setitem(extension_loader._tools, "contract_test_extension", {
        "name": "ext_contract_test",
        "description": "must not enter the shipped catalog",
        "schema": {"type": "object", "properties": {}},
        "skill": "contract_test_extension",
    })

    names = set(_function_names(shipped_builtin_tool_schemas()))

    assert "ext_contract_test" not in names


def test_every_shipped_builtin_schema_is_generally_valid_and_portable():
    assert_portable_tool_schemas(shipped_builtin_tool_schemas())


def test_root_combinator_failure_names_tool_and_keyword(subtests):
    for keyword in ROOT_SCHEMA_COMBINATORS:
        with subtests.test(keyword=keyword):
            bad = [{
                "type": "function",
                "function": {
                    "name": "bad_contract_tool",
                    "description": "bad",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        keyword: [{"required": ["selector"]}],
                    },
                },
            }]

            with pytest.raises(
                AssertionError,
                match=rf"bad_contract_tool: root schema keyword '{keyword}'",
            ):
                assert_portable_tool_schemas(bad)


def test_every_physical_provider_projection_preserves_the_complete_name_set():
    from ouroboros.llm import LLMClient
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom

    canonical = shipped_builtin_tool_schemas()
    expected = set(_function_names(canonical))

    openrouter_function = LLMClient._sanitize_chat_completion_tools(canonical)
    anthropic = LLMClient._build_anthropic_tools(canonical)
    gigachat = LLMClient._gigachat_functions(canonical)
    openai_custom = project_function_tools_to_openai_custom(openrouter_function)

    assert set(_function_names(openrouter_function)) == expected
    assert {tool["name"] for tool in anthropic} == expected
    assert {tool["name"] for tool in gigachat} == expected
    assert set(openai_custom.tool_names) == expected
    assert {
        tool["custom"]["name"] for tool in openai_custom.wire_tools()
    } == expected
