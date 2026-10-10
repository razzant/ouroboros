"""Chat name compatibility must leave additive payloads and raw wire intact."""

from __future__ import annotations

import base64
import json

import pytest

from ouroboros.llm import LLMClient
from ouroboros.llm_stream import ChatAccumulator, RejectedProviderStream
from ouroboros.observability import read_blob_ref
from tests.test_transport_b_stream_deadlines import (
    WireResponse, chunk, payload, rows, run_driver, sse, target,
)
from tests.test_transport_b_stream_deadlines import isolated as isolated


CALL_KINDS = ("function", "custom", "legacy")


def call_delta(kind, body, *, index=0):
    if kind == "legacy":
        return {"function_call": body}
    call = {"id": f"call-{index}", "type": kind, kind: body}
    if index is not None:
        call["index"] = index
    return {"tool_calls": [call]}


def assemble(*deltas):
    accumulator = ChatAccumulator()
    for delta in deltas:
        accumulator.accept("", json.dumps(chunk(delta)))
    accumulator.accept("", json.dumps(chunk(finish="tool_calls")))
    accumulator.accept("", "[DONE]")
    return accumulator


def call_body(message, kind):
    return message["function_call"] if kind == "legacy" else message["tool_calls"][0][kind]


@pytest.mark.parametrize("kind", CALL_KINDS)
@pytest.mark.parametrize("name", ("send_file", "send_user_message"))
def test_twelve_repeated_names_keep_every_argument_fragment(kind, name):
    field = "input" if kind == "custom" else "arguments"
    fragments = ['{"q":"', *(["x"] * 10), '"}']
    accumulator = assemble(*(call_delta(kind, {"name": name, field: part}) for part in fragments))

    body = call_body(accumulator.result()["choices"][0]["message"], kind)
    assert body == {"name": name, field: '{"q":"xxxxxxxxxx"}'}
    facts = accumulator.anomaly_facts()
    assert facts["count"] == 11
    assert len(facts["first"]) == 11
    assert all(note.endswith("equal name value not appended") for note in facts["first"])


@pytest.mark.parametrize("kind", CALL_KINDS)
@pytest.mark.parametrize("fragments, expected", (
    (("send_", "file"), "send_file"),
    (("ba", "na", "na", "banana"), "banana"),
    (("a", "a"), "a"),  # Accepted ambiguity: these could have meant additive "aa".
    (("abab",), "abab"),  # Never collapse repetition already inside a single value.
))
def test_name_assembly_compares_with_the_whole_accumulated_value(kind, fragments, expected):
    field = "input" if kind == "custom" else "arguments"
    accumulator = assemble(*(call_delta(kind, {"name": part, field: ""}) for part in fragments))

    body = call_body(accumulator.result()["choices"][0]["message"], kind)
    assert body["name"] == expected
    assert accumulator.anomaly_facts()["count"] == len(fragments) - 1


def test_composed_name_can_dispatch_to_a_registered_callable(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolEntry, ToolRegistry

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    called = []

    def handler(_ctx):
        called.append("banana")
        return "composed name ran"

    registry.register(ToolEntry(name="banana", schema={"type": "function", "function": {
        "name": "banana", "parameters": {"type": "object", "properties": {}},
    }}, handler=handler))
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    accumulator = assemble(*(call_delta("function", {"name": name, "arguments": args})
                            for name, args in (("ba", "{"), ("na", ""), ("na", "}"))))
    call = call_body(accumulator.result()["choices"][0]["message"], "function")

    result = registry.execute_result(call["name"], json.loads(call["arguments"]))
    assert result.status == "ok" and result.text == "composed name ran"
    assert called == ["banana"]
    assert all(note.endswith("different name values concatenated")
               for note in accumulator.anomaly_facts()["first"])


@pytest.mark.parametrize("kind", CALL_KINDS)
def test_missing_null_and_empty_names_do_not_erase_or_prevent_later_fragments(kind):
    field = "input" if kind == "custom" else "arguments"
    bodies = [
        {field: "{}"}, {"name": None}, {"name": ""}, {"name": "send_"},
        {}, {"name": None}, {"name": ""}, {"name": "file"},
        {"name": "send_file"}, {}, {"name": None}, {"name": ""},
    ]
    accumulator = assemble(*(call_delta(kind, body) for body in bodies))

    assert call_body(accumulator.result()["choices"][0]["message"], kind) == {
        "name": "send_file", field: "{}",
    }
    assert accumulator.anomaly_facts()["count"] == 2


@pytest.mark.parametrize("kind", ("function", "custom"))
def test_indexless_call_continuation_uses_the_same_name_policy(kind):
    field = "input" if kind == "custom" else "arguments"
    accumulator = assemble(*(call_delta(kind, {"name": part, field: ""}, index=None)
                            for part in ("send_", "file", "send_file")))
    message = accumulator.result()["choices"][0]["message"]
    assert len(message["tool_calls"]) == 1
    assert call_body(message, kind)["name"] == "send_file"


def test_interleaved_choices_and_call_indices_keep_separate_names():
    accumulator = ChatAccumulator(expected_choices=2)
    updates = [
        (1, call_delta("custom", {"name": "ba", "input": "{}"}, index=1)),
        (0, call_delta("function", {"name": "send_user_message", "arguments": '{"q":'}, index=1)),
        (1, call_delta("function", {"name": "send_", "arguments": "{}"})),
        (0, call_delta("function", {"name": "send_file", "arguments": "{}"})),
        (0, call_delta("function", {"name": "send_user_message", "arguments": "1}"}, index=1)),
        (1, call_delta("custom", {"name": "na"}, index=1)),
        (1, call_delta("function", {"name": "file"})),
        (0, call_delta("function", {"name": "send_file"})),
        (1, call_delta("custom", {"name": "na"}, index=1)),
    ]
    for index, delta in updates:
        accumulator.accept("", json.dumps(chunk(delta, index=index)))
    for index in (1, 0):
        accumulator.accept("", json.dumps(chunk(finish="tool_calls", index=index)))
    accumulator.accept("", "[DONE]")

    first, second = [choice["message"]["tool_calls"] for choice in accumulator.result()["choices"]]
    assert [call["function"]["name"] for call in first] == ["send_file", "send_user_message"]
    assert first[1]["function"]["arguments"] == '{"q":1}'
    assert second[0]["function"] == {"name": "send_file", "arguments": "{}"}
    assert second[1]["custom"] == {"name": "banana", "input": "{}"}


def test_name_policy_does_not_spread_to_content_reasoning_or_unrelated_nested_objects():
    unrelated = {
        "name": "na", "function": {"name": "na"}, "custom": {"name": "na"},
        "function_call": {"name": "na"},
        "tool_calls": [{"index": 0, "function": {"name": "na"}, "custom": {"name": "na"}}],
    }
    delta = {
        "name": "na", "content": "na", "reasoning": "na", "metadata": unrelated,
        "function": {"name": "na"}, "custom": {"name": "na"},
        "reasoning_details": [{"type": "reasoning.text", "text": "na", "name": "na"}],
        **call_delta("function", {"name": "send_file", "arguments": "", "metadata": unrelated}),
    }
    accumulator = assemble(delta, delta)
    message = accumulator.result()["choices"][0]["message"]

    assert message["name"] == message["content"] == message["reasoning"] == "nana"
    assert message["function"] == message["custom"] == {"name": "nana"}
    assert message["reasoning_details"] == [{"type": "reasoning.text", "text": "nana", "name": "nana"}]
    expected = {
        "name": "nana", "function": {"name": "nana"}, "custom": {"name": "nana"},
        "function_call": {"name": "nana"},
        "tool_calls": [{"index": 0, "function": {"name": "nana"}, "custom": {"name": "nana"}}],
    }
    assert message["metadata"] == expected
    assert call_body(message, "function") == {"name": "send_file", "arguments": "", "metadata": expected}
    assert accumulator.anomaly_facts()["count"] == 1


@pytest.mark.parametrize("kind", CALL_KINDS)
def test_later_name_shape_conflicts_keep_the_first_valid_text(kind):
    field = "input" if kind == "custom" else "arguments"
    accumulator = assemble(
        call_delta(kind, {"name": "send_file", field: "{}"}),
        *(call_delta(kind, {"name": value}) for value in ({"bad": "shape"}, ["bad"], 7)),
    )
    assert call_body(accumulator.result()["choices"][0]["message"], kind)["name"] == "send_file"
    assert accumulator.anomaly_facts()["count"] == 3


@pytest.mark.parametrize("kind", CALL_KINDS)
def test_initial_invalid_name_shape_stays_a_rejected_complete_response(kind):
    field = "input" if kind == "custom" else "arguments"
    accumulator = assemble(call_delta(kind, {"name": [], field: "{}"}),
                            call_delta(kind, {"name": "send_file"}))
    with pytest.raises(RejectedProviderStream):
        accumulator.result()
    assert accumulator.done is True
    assert call_body(accumulator.partial()["choices"][0]["message"], kind)["name"] == []
    assert accumulator.anomaly_facts()["count"] == 1


@pytest.mark.parametrize("asynchronous", (False, True))
def test_repeated_name_survives_physical_driver_normalization_and_private_receipt(isolated, asynchronous):
    fragments = ['{"q":"', *(["x"] * 10), '"}']
    wire = sse(
        *(chunk(call_delta("function", {"name": "send_file", "arguments": part})) for part in fragments),
        chunk(finish="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.25}),
    )
    response = WireResponse(wire)
    response.headers = {"x-generation-id": "gen-test"}
    sends = []

    def send(**kwargs):
        sends.append(kwargs)
        return response

    result = run_driver(send, payload(stream=True), target(), asynchronous=asynchronous).model_dump()
    message, _ = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert message["tool_calls"] == [{
        "id": "call-0", "type": "function",
        "function": {"name": "send_file", "arguments": '{"q":"xxxxxxxxxx"}'},
    }]
    assert len(sends) == 1
    assert response.closed
    assert [(row["state"], row["cost_usd"]) for row in rows(isolated)] == [("settled", 0.25)]

    receipt = result["_stream_receipt"]
    assert receipt["complete"] is True
    assert receipt["anomalies"]["count"] == 11
    assert len(receipt["anomalies"]["first"]) == 11
    manifest = json.loads((isolated / receipt["manifest_ref"]["path"]).read_text(encoding="utf-8"))
    projection = read_blob_ref(isolated, manifest["full_payload_ref"])
    evidence = read_blob_ref(isolated, projection["private_wire_ref"])
    assert base64.b64decode(evidence["wire_base64"]) == wire
    assert evidence["anomalies"] == receipt["anomalies"]
    assert evidence["partial_assembly"]["choices"][0]["message"]["tool_calls"] == message["tool_calls"]
