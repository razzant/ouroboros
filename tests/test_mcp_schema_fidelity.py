"""MCP discovery preserves full contracts and leaves JSON instance literals intact."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client
from ouroboros.mcp_schema import tool_metadata, wrap_schema_text


@pytest.mark.parametrize("keyword", ["default", "const", "enum", "examples", "x-extension"])
def test_instance_literals_are_not_schema_annotations(keyword):
    literal = {"title": "actual title", "description": "actual description",
               "properties": {"description": {"title": "also literal"}}}
    value = [literal] if keyword in {"enum", "examples"} else literal
    source = {"type": "object", "properties": {"value": {
        "type": "object", "description": "real annotation", keyword: value}}}
    original = deepcopy(source)
    result = wrap_schema_text(source, lambda text: text)
    assert result["properties"]["value"][keyword] == value
    assert result["properties"]["value"]["description"].endswith(" real annotation")
    assert source == original
    result["properties"]["value"][keyword] = None
    assert source == original


@pytest.mark.parametrize("container", ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"])
def test_schema_maps_preserve_property_names_and_wrap_schemas(container):
    source = {container: {"title": {"description": "  long " + "x" * 1600 + "\n"},
                          "description": {"title": "nested"}}}
    result = wrap_schema_text(source, lambda text: text)
    assert set(result[container]) == {"title", "description"}
    assert result[container]["title"]["description"].endswith(source[container]["title"]["description"])
    assert "untrusted" in result[container]["description"]["title"]


def test_subschemas_and_legacy_tuple_items_are_wrapped_but_dependencies_arrays_are_not():
    source = {"allOf": [{"description": "all"}], "items": [{"title": "item"}],
              "if": {"description": "conditional"}, "dependencies": {"field": ["title", "description"]},
              "additionalProperties": False}
    result = wrap_schema_text(source, lambda text: text)
    assert "untrusted" in result["allOf"][0]["description"]
    assert "untrusted" in result["items"][0]["title"]
    assert "untrusted" in result["if"]["description"]
    assert result["dependencies"] == source["dependencies"]
    assert result["additionalProperties"] is False


def test_manager_discovery_and_registry_preserve_full_text_and_metadata(monkeypatch):
    monkeypatch.setattr(mcp_client, "get_runtime_mode", lambda: "advanced")
    manager = mcp_client.MCPManager()
    config = {"id": "fidelity", "enabled": True, "transport": "streamable_http",
              "url": "https://example.test/mcp", "allowed_tools": ["read-record"]}
    description = "  " + "operation requirements " * 200 + "\n"
    annotation = "  " + "argument qualification " * 100 + "\n"
    literal = {"description": "instance value", "title": "instance title"}
    metadata = {"title": "Readable source", "annotations": {"readOnlyHint": False},
                "outputSchema": {"type": "object", "properties": {"value": {"const": literal}}}}

    async def listing(cfg, timeout):
        return [{"name": "read-record", "description": description,
                 "input_schema": {"type": "object", "properties": {
                     "value": {"description": annotation, "default": literal}}},
                 "metadata": deepcopy(metadata)}]

    manager._async_list_tools = listing
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [config]})
    refreshed = manager.refresh_server("fidelity")
    assert refreshed["tools"][0]["description"] == description
    assert manager.test_server(config)["tools"][0]["description"] == description
    assert manager.test_server(config)["tools"][0]["metadata"] == metadata
    row = manager.list_tools_for_registry()[0]
    assert row["raw_description"] == description
    assert description in row["description"]
    assert "[not read-only]" in row["description"] and "Server metadata: " in row["description"]
    assert '"title": "Readable source"' in row["description"] and row["description"].count("untrusted") == 1
    assert row["schema"]["properties"]["value"]["description"].endswith(annotation)
    assert row["schema"]["properties"]["value"]["default"] == literal
    assert row["metadata"] == metadata
    assert row["name"] == mcp_client.make_tool_name("fidelity", "read-record")
    assert manager.resolve_tool_name(row["name"]).raw_name == "read-record"
    # A readOnlyHint is advice: actual configured membership, not metadata, admits calls.
    config["allowed_tools"] = ["another"]
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [config]})
    manager.refresh_server("fidelity")
    assert manager.resolve_tool_name(row["name"]).status == "disallowed"


def test_sdk_metadata_is_json_and_independent():
    from mcp.types import Tool, ToolAnnotations

    tool = Tool(name="read", title="title", inputSchema={"type": "object"},
                annotations=ToolAnnotations(readOnlyHint=True), outputSchema={"type": "object"})
    result = tool_metadata(tool)
    assert result["annotations"]["readOnlyHint"] is True
    assert result["title"] == "title"
    result["outputSchema"]["type"] = "array"
    assert tool.outputSchema["type"] == "object"
    assert tool_metadata(SimpleNamespace(name="old")) == {}


@pytest.mark.parametrize("output_schema", [
    {"type": "object", "properties": {"result": {"const": "SECRET"}}},
    {"type": "object", "properties": {"SECRET": {"type": "string"}}},
    {"type": "object", "properties": {"result": {"default": "SECRET"}}},
    {"type": "object", "properties": {"result": {"enum": ["SECRET"]}}},
    {"type": "object", "properties": {"result": {"examples": ["SECRET"]}}},
    {"type": "object", "properties": {"result": {"x-extra": {"nested": "SECRET"}}}},
])
def test_output_schema_secret_literals_drop_metadata_without_withholding_tool(output_schema):
    import json

    secret = "synthetic-output-secret"
    secret_schema = json.loads(json.dumps(output_schema).replace("SECRET", secret))

    async def listing(_cfg, _timeout):
        return [
            {"name": "bad", "input_schema": {"type": "object"},
             "metadata": {"outputSchema": deepcopy(secret_schema)}},
            {"name": "good", "input_schema": {"type": "object"},
             "metadata": {"outputSchema": {"type": "object", "properties": {"result": {"const": "plain"}}}}},
        ]

    manager = mcp_client.MCPManager()
    manager._async_list_tools = listing
    config = {"id": "output", "enabled": True, "transport": "streamable_http",
              "url": "https://example.test/mcp", "auth_token": secret}
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [config]})
    refreshed = manager.refresh_server("output")
    tested = manager.test_server(config)
    rows = manager.list_tools_for_registry()
    assert [row["raw_name"] for row in rows] == ["bad", "good"]
    assert "outputSchema" not in rows[0]["metadata"]
    assert manager.resolve_tool_name("mcp_output__bad").status == "callable"
    assert tested["tool_count"] == 2
    assert "MCP_SCHEMA_OPTIONAL_DROPPED" in json.dumps(manager.status_payload())
    assert "MCP_SCHEMA_OPTIONAL_DROPPED" in json.dumps(tested)
    assert secret not in json.dumps([refreshed, tested, rows, manager.status_payload()])


def test_output_schema_annotations_and_metadata_keys_are_redacted_without_changing_literals():
    import json

    secret = "synthetic-output-secret"
    output_schema = {"type": "object", "description": "note " + secret,
                     "properties": {"result": {"const": "ordinary", "title": "title " + secret}}}

    async def listing(_cfg, _timeout):
        return [{"name": "good", "input_schema": {"type": "object"},
                 "metadata": {"outputSchema": deepcopy(output_schema),
                              "annotations": {"annotation-" + secret: "value " + secret}}}]

    manager = mcp_client.MCPManager()
    manager._async_list_tools = listing
    config = {"id": "output", "enabled": True, "transport": "streamable_http",
              "url": "https://example.test/mcp", "auth_token": secret}
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [config]})
    refreshed = manager.refresh_server("output")
    tested = manager.test_server(config)
    row = manager.list_tools_for_registry()[0]
    assert "outputSchema" not in row["metadata"]
    assert "annotation-<redacted:mcp-auth-token>" in row["metadata"]["annotations"]
    assert tested["tools"][0]["metadata"] == row["metadata"]
    assert secret not in json.dumps([refreshed, tested, row, manager.status_payload()])


@pytest.mark.serial
@pytest.mark.parametrize("transport", ["http", "sse"])
def test_real_sdk_to_registered_schema_preserves_complete_contract(tmp_path, monkeypatch, transport):
    """Cross the actual SDK/list/manager/registry path, not a synthetic list result."""
    import json
    from typing import Annotated

    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    from pydantic import Field

    from ouroboros import config
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_mcp_headers_loopback import start_asgi

    token = "synthetic-mcp-fidelity-secret-93"
    description = "  " + "Complete operation contract. " * 100 + "\n" + token
    argument = "  " + "Complete argument contract. " * 80 + "\n"
    server = FastMCP("fidelity", host="127.0.0.1", log_level="WARNING")

    def echo(text: str) -> dict[str, str]:
        return {"echo": text}

    echo.__annotations__["text"] = Annotated[str, Field(description=argument)]
    server.tool(description=description, title="Title " + token,
                annotations=ToolAnnotations(readOnlyHint=True))(echo)
    app = server.streamable_http_app() if transport == "http" else server.sse_app()
    running = []
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    mcp_client.reset_manager_for_tests()
    try:
        port = start_asgi(app, running, "mcp-fidelity")
        manager = mcp_client.get_manager()
        settings = {"MCP_ENABLED": True, "MCP_SERVERS": [{
            "id": "fidelity", "enabled": True,
            "transport": "streamable_http" if transport == "http" else "sse",
            "url": f"http://127.0.0.1:{port}/" + ("mcp" if transport == "http" else "sse"),
            "auth_token": token, "auth_header": "X-Token", "allowed_tools": ["echo"],
        }]}
        # Registry reads now adopt Settings and the published catalog.
        config.SETTINGS_PATH.write_text(json.dumps(settings))
        monkeypatch.setattr(config, "load_settings", lambda: settings)
        manager.reconfigure(settings)
        listing = manager.refresh_server("fidelity")
        assert listing["ok"], listing
        registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
        schema = registry.get_schema_by_name("mcp_fidelity__echo")["function"]
        assert description.replace(token, "<redacted:mcp-auth-token>") in schema["description"]
        assert argument in schema["parameters"]["properties"]["text"]["description"]
        # One frame per loaded tool: hints render as marks, other metadata stays complete under it.
        assert "[read-only]" in schema["description"] and "outputSchema" in schema["description"]
        assert schema["description"].count("untrusted") == 1
        assert token not in json.dumps(schema)
        status = manager.status_payload()["servers"][0]["tools"][0]
        assert status["metadata"]["annotations"]["readOnlyHint"] is True
        assert token not in json.dumps(status)
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
        result = registry.execute_result("mcp_fidelity__echo", {"text": "consumer"})
        assert result.status == "ok" and "consumer" in result.text
    finally:
        mcp_client.reset_manager_for_tests()
        for process, thread in running:
            process.should_exit = True
            thread.join(10)
            assert not thread.is_alive()


def test_nested_metadata_output_schema_is_annotation_data_not_a_schema():
    from ouroboros.mcp_schema import filter_schema_literals, redact_metadata

    secret = "synthetic-nested-metadata-secret"
    redact = lambda text: text.replace(secret, "<redacted>")
    metadata = {"annotations": {"outputSchema": {"const": secret}},
                "outputSchema": {"type": "string", "const": "ordinary"}}
    rows = [{"name": "safe", "input_schema": {"type": "object"}, "metadata": metadata}]
    kept, omitted = filter_schema_literals(rows, redact)
    assert kept == rows and omitted == []
    projected = redact_metadata(metadata, redact)
    assert projected["annotations"]["outputSchema"]["const"] == "<redacted>"
    assert projected["outputSchema"] == metadata["outputSchema"]
    assert metadata["annotations"]["outputSchema"]["const"] == secret


@pytest.mark.parametrize("schema", [
    {"enum": ["SECRET"]}, {"const": "SECRET"}, {"pattern": "SECRET"},
    {"required": ["SECRET"]}, {"properties": {"SECRET": {"type": "string"}}}, {"$ref": "SECRET"},
])
def test_irreducible_input_literals_withhold_only_their_descriptor(schema):
    from ouroboros.mcp_schema import filter_schema_literals
    rows = [{"name": "bad", "input_schema": schema}, {"name": "good", "input_schema": {"type": "object"}}]
    kept, notices = filter_schema_literals(rows, lambda text: text.replace("SECRET", "<redacted>"))
    assert [row["name"] for row in kept] == ["good"]
    assert notices == ["MCP_SCHEMA_LITERAL_SECRET: bad"]


@pytest.mark.parametrize("keyword", ["default", "examples"])
def test_optional_input_literals_drop_only_the_field(keyword):
    from ouroboros.mcp_schema import filter_schema_literals
    schema = {"properties": {"value": {"type": "string", keyword: "SECRET", "enum": ["ordinary"]}},
              "required": ["value"]}
    kept, notices = filter_schema_literals([{"name": "good", "input_schema": schema}],
                                           lambda text: text.replace("SECRET", "<redacted>"))
    assert kept[0]["input_schema"] == {"properties": {"value": {"type": "string", "enum": ["ordinary"]}},
                                       "required": ["value"]}
    assert keyword in notices[0] and "MCP_SCHEMA_OPTIONAL_DROPPED" in notices[0]
    assert schema["properties"]["value"][keyword] == "SECRET"
