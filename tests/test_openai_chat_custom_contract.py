"""Pure direct-OpenAI Chat custom-tool codec contracts."""

from __future__ import annotations

import copy
import json
import pathlib
import re

import pytest
from jsonschema import Draft7Validator, Draft202012Validator

from ouroboros.openai_chat_custom import (
    JSON_OBJECT_LARK_GRAMMAR,
    MAX_CUSTOM_DESCRIPTION_CHARS,
    CustomToolProjectionError,
    ProjectedCustomToolCatalog,
    custom_receipts_prove_wire_acceptance,
    normalize_openai_custom_tool_calls,
    project_function_tool_request_to_openai_custom,
    project_function_tools_to_openai_custom,
    project_messages_for_openai_custom,
    project_tool_choice_for_openai_custom,
    reencode_canonical_calls_for_openai_custom,
)
from ouroboros.request_wire_contract import canonical_json
from ouroboros.request_wire_receipts import WireCandidateSpec, bind_wire_candidate

_SCHEMA_CLAUSE = "Input must be one JSON object matching this schema: "


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "description": "Submit the marker.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "marker": {
                            "type": "string",
                            "enum": ["ok"],
                            "description": "Nested prose is not needed on wire.",
                        },
                    },
                    "required": ["marker"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "second",
                "description": "Return a number.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        },
    ]


def _wire_candidate(tools=None, *, dialect="openai_chat_custom"):
    source = {
        "model": "gpt-future",
        "messages": [{"role": "user", "content": "Use one tool."}],
        "reasoning_effort": "high",
        "tool_choice": "required",
        "tools": copy.deepcopy(tools if tools is not None else _tools()),
    }
    return bind_wire_candidate(
        target={
            "provider": "openai",
            "resolved_model": "gpt-future",
            "usage_model": "openai/gpt-future",
            "base_url": "https://api.openai.example/v1",
        },
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(dialect, "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )


def _projected_schema(catalog, name):
    tool = next(item for item in catalog.wire_tools() if item["custom"]["name"] == name)
    _, separator, schema_json = tool["custom"]["description"].rpartition(_SCHEMA_CLAUSE)
    assert separator == _SCHEMA_CLAUSE
    value = json.loads(schema_json)
    assert isinstance(value, dict)
    return value


def _assert_validator_parity(schema, cases, validator_class):
    tool = copy.deepcopy(_tools()[0])
    tool["function"]["parameters"] = schema
    compact = _projected_schema(project_function_tools_to_openai_custom([tool]), "probe")
    validator_class.check_schema(schema)
    validator_class.check_schema(compact)
    original_validator = validator_class(schema)
    compact_validator = validator_class(compact)
    for instance, expected_valid in cases:
        assert original_validator.is_valid(instance) is expected_valid
        assert compact_validator.is_valid(instance) is expected_valid
    return compact


def test_projection_is_deterministic_structural_and_schema_bound():
    tools = _tools()
    first = project_function_tools_to_openai_custom(tools)
    second = project_function_tools_to_openai_custom(list(reversed(tools)))
    assert first == second
    assert first.tool_names == ("probe", "second")
    assert first.serialized_bytes == len(first.wire_tools_json.encode("utf-8"))
    assert all(
        len(tool["custom"]["description"]) <= MAX_CUSTOM_DESCRIPTION_CHARS
        for tool in first.wire_tools()
    )
    probe_description = first.wire_tools()[0]["custom"]["description"]
    assert '"enum":["ok"]' in probe_description
    assert "Nested prose" not in probe_description
    assert first.schema_binding("probe").schema()["properties"]["marker"]["description"]


def test_compaction_preserves_literal_enum_and_const_objects_exactly():
    literal_enum = {"title": "literal", "description": "literal", "x-tag": "literal"}
    literal_const = {
        "properties": {"title": "literal"},
        "description": "literal",
        "x-tag": "literal",
    }
    schema = {
        "type": "object",
        "properties": {
            "enum_value": {"enum": [literal_enum]},
            "const_value": {"const": literal_const},
        },
        "required": ["enum_value", "const_value"],
        "additionalProperties": False,
        "description": "schema prose is removable",
    }
    compact = _assert_validator_parity(schema, [
        ({"enum_value": literal_enum, "const_value": literal_const}, True),
        ({"enum_value": {"title": "literal"}, "const_value": literal_const}, False),
        ({"enum_value": literal_enum, "const_value": {"description": "literal"}}, False),
    ], Draft202012Validator)

    assert compact["properties"]["enum_value"]["enum"] == [literal_enum]
    assert compact["properties"]["const_value"]["const"] == literal_const
    assert "description" not in compact


def test_compaction_preserves_draft7_dependency_behavior_and_nested_schemas():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "definitions": {
            "description": {
                "type": "string",
                "minLength": 2,
                "description": "nested schema prose",
            },
        },
        "properties": {
            "description": {"type": "string"},
            "title": {"$ref": "#/definitions/description"},
            "mode": {"type": "string"},
            "detail": {"type": "object"},
        },
        "patternProperties": {"^x-": {"type": "integer"}},
        "dependencies": {
            "description": ["title"],
            "mode": {
                "properties": {
                    "detail": {"const": {"description": "literal dependency value"}},
                },
                "required": ["detail"],
                "description": "dependency schema prose",
            },
        },
        "description": "root schema prose",
    }
    compact = _assert_validator_parity(schema, [
        ({}, True),
        ({"description": "present"}, False),
        ({"description": "present", "title": "ok"}, True),
        ({"description": "present", "title": "x"}, False),
        ({"mode": "on"}, False),
        ({"mode": "on", "detail": {"description": "literal dependency value"}}, True),
        ({"mode": "on", "detail": {}}, False),
        ({"x-count": 3}, True),
        ({"x-count": "3"}, False),
    ], Draft7Validator)

    assert compact["dependencies"]["description"] == ["title"]
    assert "description" not in compact["definitions"]["description"]
    assert "description" not in compact["dependencies"]["mode"]


def test_custom_json_number_grammar_matches_json_number_syntax():
    match = re.search(r"^NUMBER: /(.+)/$", JSON_OBJECT_LARK_GRAMMAR, re.MULTILINE)
    assert match is not None
    number_pattern = re.compile(match.group(1))
    for value in ("0", "-0", "17", "-17", "1.25", "-0.5", "1e9", "1E-9", "1.2e+3"):
        assert number_pattern.fullmatch(value), value
    for value in ("+1", "01", "-01", ".5", "1.", "1e", "1e+", "NaN", "Infinity"):
        assert not number_pattern.fullmatch(value), value
    assert "SIGNED_NUMBER" not in JSON_OBJECT_LARK_GRAMMAR


def test_custom_json_string_grammar_matches_json_escape_syntax():
    match = re.search(r"^ESCAPED_STRING: /(.+)/$", JSON_OBJECT_LARK_GRAMMAR, re.MULTILINE)
    assert match is not None
    string_pattern = re.compile(match.group(1))
    for value in (
        r'""',
        r'"plain"',
        r'"quote:\""',
        r'"solidus:\/"',
        r'"backslash:\\"',
        r'"\b\f\n\r\t"',
        r'"\u0041"',
    ):
        assert string_pattern.fullmatch(value), value
    for value in (
        r'"\q"',
        r'"\x41"',
        r'"\u123"',
        r'"\uZZZZ"',
        '"line\nbreak"',
        '"tab\tcharacter"',
    ):
        assert not string_pattern.fullmatch(value), value


def test_projection_rejects_mixed_duplicate_and_unrepresentable_catalogs():
    with pytest.raises(CustomToolProjectionError, match="at least one"):
        project_function_tools_to_openai_custom([])
    with pytest.raises(CustomToolProjectionError, match="function tools only"):
        project_function_tools_to_openai_custom([{"type": "web_search"}])
    with pytest.raises(CustomToolProjectionError, match="duplicate"):
        project_function_tools_to_openai_custom([_tools()[0], _tools()[0]])
    huge = _tools()[0]
    huge = copy.deepcopy(huge)
    huge["function"]["parameters"]["properties"]["marker"]["enum"] = [
        f"value-{index:04d}" for index in range(1000)
    ]
    with pytest.raises(CustomToolProjectionError, match="exceeds"):
        project_function_tools_to_openai_custom([huge])


@pytest.mark.parametrize("available", [1, 2, 3])
def test_description_limit_holds_at_truncation_marker_boundary(available):
    base_schema = {"type": "object", "properties": {"x": {"const": ""}}}
    base_clause_length = len(_SCHEMA_CLAUSE + canonical_json(base_schema))
    filler_length = (
        MAX_CUSTOM_DESCRIPTION_CHARS - 1 - available - base_clause_length
    )
    schema = {
        "type": "object",
        "properties": {"x": {"const": "x" * filler_length}},
    }
    tool = copy.deepcopy(_tools()[0])
    tool["function"]["description"] = "verbose prose"
    tool["function"]["parameters"] = schema

    description = project_function_tools_to_openai_custom([tool]).wire_tools()[0][
        "custom"
    ]["description"]
    schema_clause = _SCHEMA_CLAUSE + canonical_json(schema)

    assert len(description) <= MAX_CUSTOM_DESCRIPTION_CHARS
    assert description.endswith(schema_clause)
    if available < 3:
        assert description == schema_clause
    else:
        assert description == "...\n" + schema_clause


def test_tool_choice_and_assistant_call_round_trip_leave_canonical_history_unchanged():
    catalog = project_function_tools_to_openai_custom(_tools())
    assert project_tool_choice_for_openai_custom("required", catalog) == "required"
    assert project_tool_choice_for_openai_custom(
        {"type": "function", "function": {"name": "probe"}}, catalog
    ) == {"type": "custom", "custom": {"name": "probe"}}
    assert project_tool_choice_for_openai_custom(
        {"type": "tool", "name": "second"}, catalog
    ) == {"type": "custom", "custom": {"name": "second"}}

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "probe", "arguments": '{"marker":"ok"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"done":true}'},
    ]
    original = copy.deepcopy(messages)
    projected = project_messages_for_openai_custom(messages, catalog)
    assert messages == original
    assert projected[0]["tool_calls"] == [{
        "id": "call_1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }]
    assert projected[1] == messages[1]


def test_request_projection_narrows_allowed_custom_tools_and_preserves_other_choices():
    required = project_function_tool_request_to_openai_custom(_tools(), {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [{"type": "function", "name": "probe"}],
        },
    })
    assert required.catalog.tool_names == ("probe",)
    assert required.catalog.schema_binding("probe") is not None
    assert required.catalog.schema_binding("second") is None
    assert [tool["custom"]["name"] for tool in required.catalog.wire_tools()] == ["probe"]
    assert required.tool_choice == "required"

    auto = project_function_tool_request_to_openai_custom(_tools(), {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": ["second"]},
    })
    assert auto.catalog.tool_names == ("second",)
    assert auto.tool_choice == "auto"

    named = project_function_tool_request_to_openai_custom(
        _tools(),
        {"type": "function", "function": {"name": "probe"}},
    )
    assert named.catalog.tool_names == ("probe", "second")
    assert named.tool_choice == {"type": "custom", "custom": {"name": "probe"}}


@pytest.mark.parametrize("allowed_tools, message", [
    ({"mode": "auto", "tools": []}, "requires tool names"),
    ({"mode": "auto", "tools": ["missing"]}, "unknown tool"),
    ({"mode": "sometimes", "tools": ["probe"]}, "mode must be auto or required"),
])
def test_request_projection_rejects_invalid_allowed_custom_subsets(allowed_tools, message):
    with pytest.raises(CustomToolProjectionError, match=message):
        project_function_tool_request_to_openai_custom(_tools(), {
            "type": "allowed_tools",
            "allowed_tools": allowed_tools,
        })


@pytest.mark.parametrize("call", [None, []])
def test_canonical_replay_rejects_non_mapping_calls(call):
    catalog = project_function_tools_to_openai_custom(_tools())
    with pytest.raises(CustomToolProjectionError, match="must be a mapping"):
        reencode_canonical_calls_for_openai_custom([call], catalog)


@pytest.mark.parametrize("call_id", [None, "", "   ", 0])
def test_canonical_replay_rejects_missing_or_empty_call_ids(call_id):
    catalog = project_function_tools_to_openai_custom(_tools())
    with pytest.raises(CustomToolProjectionError, match="missing or empty"):
        reencode_canonical_calls_for_openai_custom([{
            "id": call_id,
            "type": "function",
            "function": {"name": "probe", "arguments": '{"marker":"ok"}'},
        }], catalog)


def test_canonical_replay_rejects_duplicate_call_ids_as_one_invalid_batch():
    catalog = project_function_tools_to_openai_custom(_tools())
    call = {
        "id": "call_duplicate",
        "type": "function",
        "function": {"name": "probe", "arguments": '{"marker":"ok"}'},
    }
    with pytest.raises(CustomToolProjectionError, match="duplicate"):
        reencode_canonical_calls_for_openai_custom([call, copy.deepcopy(call)], catalog)


@pytest.mark.parametrize(("call", "message"), [
    ({
        "id": " call_padded ",
        "type": "function",
        "function": {"name": "probe", "arguments": "{}"},
    }, "id has surrounding whitespace"),
    ({
        "id": "call_padded_name",
        "type": "function",
        "function": {"name": " probe ", "arguments": "{}"},
    }, "name has surrounding whitespace"),
    ({
        "id": "call_numeric_name",
        "type": "function",
        "function": {"name": 123, "arguments": "{}"},
    }, "name is missing or empty"),
    ({
        "id": "call_wrong_type",
        "type": "custom",
        "function": {"name": "probe", "arguments": "{}"},
    }, "type must be exactly function"),
])
def test_canonical_replay_rejects_coerced_or_noncanonical_identity(call, message):
    catalog = project_function_tools_to_openai_custom(_tools())
    with pytest.raises(CustomToolProjectionError, match=message):
        reencode_canonical_calls_for_openai_custom([call], catalog)


def test_custom_calls_normalize_and_full_schema_receipt_gates_execution():
    candidate = _wire_candidate()
    valid_calls, valid_receipts = normalize_openai_custom_tool_calls([{
        "id": "call_ok",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
        "function": None,
    }], candidate)
    assert valid_calls == [{
        "id": "call_ok",
        "type": "function",
        "function": {"name": "probe", "arguments": '{"marker":"ok"}'},
    }]
    assert valid_receipts[0].allows_execution
    assert custom_receipts_prove_wire_acceptance(valid_receipts)

    invalid_calls, invalid_receipts = normalize_openai_custom_tool_calls([{
        "id": "call_bad",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"wrong"}'},
    }], candidate)
    assert invalid_calls == [{
        "id": "call_bad",
        "type": "function",
        "function": {"name": "probe", "arguments": '{"marker":"wrong"}'},
    }]
    assert invalid_receipts[0].error_code == "schema_validation_failed"
    assert not invalid_receipts[0].allows_execution
    assert custom_receipts_prove_wire_acceptance(invalid_receipts)

    _, malformed = normalize_openai_custom_tool_calls([{
        "id": "call_malformed",
        "type": "custom",
        "custom": {"name": "probe", "input": "not-json"},
    }], candidate)
    assert not custom_receipts_prove_wire_acceptance(malformed)


def test_invalid_bound_schema_is_catalog_defect_but_still_proves_custom_wire():
    tools = copy.deepcopy(_tools())
    tools[0]["function"]["parameters"]["properties"]["marker"]["type"] = (
        "not-a-json-schema-type"
    )
    candidate = _wire_candidate(tools)
    canonical_calls, receipts = normalize_openai_custom_tool_calls([{
        "id": "call_invalid_schema",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }], candidate)

    assert canonical_calls[0]["function"]["arguments"] == '{"marker":"ok"}'
    assert receipts[0].decoded_object is True
    assert receipts[0].error_code == "invalid_schema"
    assert not receipts[0].allows_execution
    assert custom_receipts_prove_wire_acceptance(receipts)


def test_missing_local_schema_ref_returns_typed_non_executable_receipt():
    tools = copy.deepcopy(_tools())
    tools[0]["function"]["parameters"] = {
        "type": "object",
        "properties": {"marker": {"$ref": "#/$defs/missing"}},
    }
    candidate = _wire_candidate(tools)

    _, receipts = normalize_openai_custom_tool_calls([{
        "id": "call_missing_ref",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }], candidate)

    assert receipts[0].decoded_object is True
    assert receipts[0].error_code == "invalid_schema"
    assert not receipts[0].allows_execution
    assert custom_receipts_prove_wire_acceptance(receipts)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_custom_decoder_rejects_non_json_numeric_constants(constant):
    tools = copy.deepcopy(_tools())
    tools[0]["function"]["parameters"] = {
        "type": "object",
        "properties": {"marker": {"type": "number"}},
        "required": ["marker"],
    }
    candidate = _wire_candidate(tools)

    _, receipts = normalize_openai_custom_tool_calls([{
        "id": f"call_{constant}",
        "type": "custom",
        "custom": {"name": "probe", "input": f'{{"marker":{constant}}}'},
    }], candidate)

    assert receipts[0].error_code == "invalid_json"
    assert not receipts[0].allows_execution
    assert not custom_receipts_prove_wire_acceptance(receipts)


@pytest.mark.parametrize("call", [
    None,
    [],
])
def test_custom_response_rejects_non_mapping_calls_before_receipt_creation(call):
    candidate = _wire_candidate()
    with pytest.raises(CustomToolProjectionError, match="must be a mapping"):
        normalize_openai_custom_tool_calls([call], candidate)


@pytest.mark.parametrize("call_id", [None, "", "   ", 0])
def test_custom_response_rejects_missing_or_empty_call_ids(call_id):
    candidate = _wire_candidate()
    with pytest.raises(CustomToolProjectionError, match="missing or empty"):
        normalize_openai_custom_tool_calls([{
            "id": call_id,
            "type": "custom",
            "custom": {"name": "probe", "input": '{"marker":"ok"}'},
        }], candidate)


def test_custom_response_rejects_duplicate_call_ids_as_one_invalid_batch():
    candidate = _wire_candidate()
    call = {
        "id": "call_duplicate",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }
    with pytest.raises(CustomToolProjectionError, match="duplicate"):
        normalize_openai_custom_tool_calls(
            [call, copy.deepcopy(call)],
            candidate,
        )


@pytest.mark.parametrize(("call", "message"), [
    ({
        "id": " call_padded ",
        "type": "custom",
        "custom": {"name": "probe", "input": "{}"},
    }, "id has surrounding whitespace"),
    ({
        "id": "call_padded_name",
        "type": "custom",
        "custom": {"name": " probe ", "input": "{}"},
    }, "name has surrounding whitespace"),
    ({
        "id": "call_numeric_name",
        "type": "custom",
        "custom": {"name": 123, "input": "{}"},
    }, "name is missing or empty"),
    ({
        "id": "call_wrong_type",
        "type": "function",
        "custom": {"name": "probe", "input": "{}"},
    }, "type must be exactly custom"),
])
def test_custom_response_rejects_coerced_or_noncanonical_identity(call, message):
    candidate = _wire_candidate()
    with pytest.raises(CustomToolProjectionError, match=message):
        normalize_openai_custom_tool_calls(
            [call],
            candidate,
        )


def test_public_normalizer_cannot_accept_an_independent_permissive_catalog():
    candidate = _wire_candidate()
    owned_catalog = candidate.custom_catalog
    assert owned_catalog is not None
    permissive_tools = copy.deepcopy(_tools())
    permissive_tools[0]["function"]["parameters"] = {
        "type": "object",
        "properties": {"marker": {"type": "string"}},
    }
    permissive_catalog = project_function_tools_to_openai_custom(permissive_tools)
    counterfeit_catalog = ProjectedCustomToolCatalog(
        wire_tools_json=permissive_catalog.wire_tools_json,
        schemas=permissive_catalog.schemas,
        catalog_sha256=owned_catalog.catalog_sha256,
        serialized_bytes=permissive_catalog.serialized_bytes,
    )
    calls = [{
        "id": "call_counterfeit",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"wrong"}'},
    }]

    _, receipts = normalize_openai_custom_tool_calls(calls, candidate)
    assert receipts[0].error_code == "schema_validation_failed"
    assert not receipts[0].allows_execution
    assert counterfeit_catalog.catalog_sha256 == owned_catalog.catalog_sha256
    assert counterfeit_catalog.tool_names == owned_catalog.tool_names
    with pytest.raises(CustomToolProjectionError, match="factory-issued"):
        normalize_openai_custom_tool_calls(calls, counterfeit_catalog)
    with pytest.raises(TypeError, match="positional"):
        normalize_openai_custom_tool_calls(calls, candidate, counterfeit_catalog)


def test_public_normalizer_rejects_a_factory_issued_non_custom_candidate():
    function_candidate = _wire_candidate(dialect="function")
    with pytest.raises(CustomToolProjectionError, match="direct OpenAI Chat custom"):
        normalize_openai_custom_tool_calls([], function_candidate)


def test_full_builtin_registry_fits_custom_description_contract(tmp_path):
    from jsonschema import validators

    from ouroboros.tools.registry import ToolRegistry
    from tests.test_smoke import EXPECTED_TOOLS

    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    registry = ToolRegistry(repo_dir=repo_dir, drive_root=tmp_path)
    schemas = registry.schemas()
    expected_names = set(EXPECTED_TOOLS)
    assert {tool["function"]["name"] for tool in schemas} == expected_names
    assert len(schemas) == len(expected_names)
    catalog = project_function_tools_to_openai_custom(schemas)
    assert catalog.tool_names == tuple(sorted(expected_names))
    assert all(
        len(tool["custom"]["description"]) <= MAX_CUSTOM_DESCRIPTION_CHARS
        for tool in catalog.wire_tools()
    )
    assert json.loads(catalog.wire_tools_json)
    for tool in schemas:
        function = tool["function"]
        assert catalog.schema_binding(function["name"]).schema() == function["parameters"]
        compact = _projected_schema(catalog, function["name"])
        validator_class = validators.validator_for(compact)
        validator_class.check_schema(compact)


_ACTUAL_PROFILES = {
    "full": {}, "readonly": {"mode": "local_readonly_subagent"},
    "acting": {"mode": "acting_subagent", "surface": "self_worktree", "root_task_id": "projection"},
}


@pytest.mark.parametrize("profile", sorted(_ACTUAL_PROFILES))
def test_actual_profile_catalogs_carry_every_tool_lead_whole(tmp_path, profile):
    """Custom tools drop parameter prose, so a first-party lead clipped to fit beside its compact
    schema loses pre-call conditions nothing else on that wire carries (schedule_subagent's tail
    did, #502). Extension and MCP descriptions are external text, not covered here."""
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=tmp_path)
    registry._ctx.task_constraint = _ACTUAL_PROFILES[profile]
    schemas = registry.schemas()
    descriptions = {tool["custom"]["name"]: tool["custom"]["description"]
                    for tool in project_function_tools_to_openai_custom(schemas).wire_tools()}
    over, checked = {}, set()
    for tool in schemas:
        name = tool["function"]["name"]
        if name not in registry._entries:
            continue
        checked.add(name)
        lead = " ".join(str(tool["function"].get("description") or "").split())
        if not descriptions[name].startswith((lead + "\n" if lead else "") + _SCHEMA_CLAUSE):
            clause = descriptions[name].rpartition("\n")[2]
            over[name] = len(lead) - (MAX_CUSTOM_DESCRIPTION_CHARS - len(clause) - 1)
    assert "schedule_subagent" in checked, sorted(checked)
    assert over == {}, f"{profile}: leads clipped by the custom description bound (chars over): {over}"
