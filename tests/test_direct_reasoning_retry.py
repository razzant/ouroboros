"""Direct Anthropic recovery adopts only an answered omission of native reasoning."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from types import SimpleNamespace

import pytest
import requests

from ouroboros import request_wire_contract as wire
from ouroboros.anthropic_native_custody import (
    ANTHROPIC_NATIVE_RECEIPT_KEY,
    retain_native_assistant_content,
)
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import ProcessingNotStarted, ProviderPolicyRefusal
from ouroboros.llm_stream import ProviderStreamError
from ouroboros.reasoning_artifacts import note_replay_omission, pop_replay_omission
from ouroboros.send_clock import main_send_scope
from ouroboros.usage_accounting import UsageScope, usage_scope
from tests.test_anthropic_native_custody import _canonical_message, _raw_content, _target, _tool_results


class _Response:
    def __init__(self, status=200, message="The request was rejected.", *, code="invalid_request_error", invalid_json=False):
        self.status_code = status
        self.reason = "Bad Request" if status >= 400 else "OK"
        self.url = "https://api.anthropic.example/v1/messages"
        self.headers = {}
        self.invalid_json = invalid_json
        self.body = ({"error": {"type": "invalid_request_error", "code": code, "message": message}}
                     if status >= 400 else {"content": [{"type": "text", "text": "Recovered answer."}],
                                            "stop_reason": "end_turn", "usage": {}})
        self.text = json.dumps(self.body)

    def json(self):
        if self.invalid_json:
            raise ValueError("invalid response JSON")
        return copy.deepcopy(self.body)

    def close(self):
        pass


class _StreamErrorResponse(_Response):
    def iter_content(self, chunk_size):
        yield b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","code":400,"message":"Generic stream refusal."}}\n\n'


def _history(kind="thinking"):
    native = _canonical_message()
    raw = [block for block in _raw_content()
           if block["type"] not in {"thinking", "redacted_thinking"} or block["type"] == kind]
    native = retain_native_assistant_content(native, raw, _target())
    native.update(reasoning="flat derived thought", reasoning_content="foreign direct-route echo",
                  reasoning_details=[{"type": "reasoning.text", "format": "anthropic-claude-v1",
                                      "text": "derived thought", "signature": "derived-signature"}])
    return [{"role": "system", "content": "Keep the source facts."},
            {"role": "user", "content": "Inspect both sources."}, native, *_tool_results()]


def _thinking(payload):
    return [block for row in payload["messages"] for block in row["content"]
            if block.get("type") in {"thinking", "redacted_thinking"}]


@pytest.fixture
def lane(tmp_path, monkeypatch):
    wire_root, sent, builds, script = tmp_path / "wire", [], [], []
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: wire_root)
    client = LLMClient(api_key="unused")
    build = client._build_remote_candidate

    def build_candidate(target, messages, *args, **kwargs):
        builds.append(copy.deepcopy(messages))
        return build(target, messages, *args, **kwargs)

    def post(url, **kwargs):
        assert url == _target()["base_url"] + "/messages"
        sent.append(copy.deepcopy(kwargs["json"]))
        assert script, "The direct lane sent more requests than the test allowed."
        reply = script.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(client, "_build_remote_candidate", build_candidate)
    monkeypatch.setattr(requests, "post", post)
    pop_replay_omission()
    with usage_scope(UsageScope(drive_root=tmp_path / "ledger", task_id="direct-reasoning")):
        yield SimpleNamespace(client=client, target=_target(), script=script, sent=sent, builds=builds,
                              wire_state=wire_root / "state" / wire.REQUEST_WIRE_STATE_FILE)
    pop_replay_omission()


def _run(lane, messages, *, asynchronous=False, adopt=False, effort="high", stream=False):
    args = (lane.target, messages, None, effort, 64, "auto")

    def scope():
        return main_send_scope(None, messages) if adopt else contextlib.nullcontext()

    def take_omission():
        omitted = pop_replay_omission()
        if adopt:
            note_replay_omission(omitted)  # the surrounding Main scope consumes the same observed fact
        return omitted

    async def send_async():
        try:
            with scope():
                result = await lane.client._chat_anthropic_async(*args, stream=stream)
                omitted = take_omission()
            return result, omitted
        except Exception as error:
            return error, pop_replay_omission()

    if asynchronous:
        return asyncio.run(send_async())
    try:
        with scope():
            result = lane.client._chat_anthropic(*args, stream=stream)
            omitted = take_omission()
        return result, omitted
    except Exception as error:
        return error, pop_replay_omission()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["thinking", "redacted_thinking"])
def test_generic_400_rebuilds_once_without_native_receipt_and_main_keeps_the_omission(lane, asynchronous, kind):
    messages = _history(kind)
    expected = LLMClient._strip_openrouter_roundtrip_metadata(messages, keep_reasoning_content=True)
    lane.script[:] = [_Response(400), _Response(), _Response()]
    result, omitted = _run(lane, messages, asynchronous=asynchronous, adopt=True)
    assert not isinstance(result, BaseException), result
    assert result[0]["content"] == "Recovered answer." and omitted is True
    assert len(lane.sent) == 2 and len(lane.builds) == 2
    assert [block["type"] for block in _thinking(lane.sent[0])] == [kind]
    assert _thinking(lane.sent[1]) == []
    assert lane.builds[1] == messages == expected
    assert ANTHROPIC_NATIVE_RECEIPT_KEY not in messages[2]
    assert "reasoning" not in messages[2] and "reasoning_details" not in messages[2]
    assert messages[2]["reasoning_content"] == "foreign direct-route echo"
    assert "future_tool_field" in json.dumps(lane.sent[0])
    assert "future_tool_field" not in json.dumps(lane.sent[1]), "The retry must rebuild from the adopted canonical view."
    assert messages[-2:] == _tool_results()
    messages.extend([result[0], {"role": "user", "content": "Continue with the same sources."}])
    following, omitted_again = _run(lane, messages, asynchronous=asynchronous, adopt=True)
    assert not isinstance(following, BaseException), following
    assert len(lane.sent) == 3 and not lane.script and omitted_again is False
    assert _thinking(lane.sent[2]) == []
    assert lane.sent[2]["messages"][:len(lane.sent[1]["messages"])] == lane.sent[1]["messages"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_normal_success_does_not_report_or_adopt_an_omission(lane, asynchronous):
    messages = _history()
    original = copy.deepcopy(messages)
    lane.script[:] = [_Response()]
    result, omitted = _run(lane, messages, asynchronous=asynchronous, adopt=True)
    assert not isinstance(result, BaseException), result
    assert omitted is False and messages == original
    assert len(lane.sent) == 1 and len(lane.builds) == 1 and _thinking(lane.sent[0])


@pytest.mark.parametrize("scenario", ["text_overflow", "structured_overflow", "no_native_thinking", "policy", "processing",
                                      "stream", "status_429", "status_500"])
def test_other_failures_never_strip_or_adopt_replayed_reasoning(lane, scenario):
    messages = _history(None if scenario == "no_native_thinking" else "thinking")
    original = copy.deepcopy(messages)
    reply, error_type = _Response(400), requests.HTTPError
    if scenario == "text_overflow":
        reply = _Response(400, "prompt is too long: 1000794 tokens > 1000000 maximum")
    elif scenario == "structured_overflow":
        reply = _Response(400, "Opaque request refusal.", code="input_too_long")
    elif scenario == "policy":
        reply = ProviderPolicyRefusal("400 request blocked before provider dispatch")
        reply.status_code = 400
        error_type = ProviderPolicyRefusal
    elif scenario == "processing":
        reply = ProcessingNotStarted(requests.HTTPError("400 processing unavailable", response=_Response(400)),
                                     reason="unsupported")
        error_type = ProcessingNotStarted
    elif scenario == "stream":
        reply, error_type = _StreamErrorResponse(), ProviderStreamError
    elif scenario.startswith("status_"):
        reply = _Response(int(scenario.removeprefix("status_")))
    lane.script[:] = [reply]
    result, omitted = _run(lane, messages, adopt=True, stream=scenario == "stream")
    assert isinstance(result, error_type), result
    assert len(lane.sent) == len(lane.builds) == 1
    assert messages == original and omitted is False and not lane.wire_state.exists()
    assert bool(_thinking(lane.sent[0])) is (scenario != "no_native_thinking")
    if scenario in {"text_overflow", "structured_overflow"}:
        from ouroboros.loop_llm_call import classify_llm_exception

        assert classify_llm_exception(result).kind == "context_overflow"
    if scenario == "stream":
        assert result.stream_incomplete and result.status_code == 400


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_second_400_ends_the_call_without_a_second_strip_or_canonical_omission(lane, asynchronous):
    messages = _history()
    original = copy.deepcopy(messages)
    lane.script[:] = [_Response(400), _Response(400, "The stripped request was also rejected.")]
    result, omitted = _run(lane, messages, asynchronous=asynchronous, adopt=True)
    assert isinstance(result, requests.HTTPError), result
    assert "The stripped request was also rejected." in str(result)
    assert len(lane.sent) == len(lane.builds) == 2 and not lane.script
    assert _thinking(lane.sent[0]) and not _thinking(lane.sent[1])
    assert omitted is False and messages == original and not lane.wire_state.exists()


@pytest.mark.parametrize("answered", [False, True])
def test_wire_recovery_after_the_strip_learns_only_from_its_successful_candidate(lane, answered):
    messages = _history()
    original = copy.deepcopy(messages)
    lane.script[:] = [_Response(400), _Response(400, "thinking.type: Input should be 'adaptive'"),
                      _Response() if answered else _Response(400, "The final candidate was rejected.")]
    result, omitted = _run(lane, messages, adopt=True, effort="none")
    assert len(lane.sent) == 3 and len(lane.builds) == 2 and not lane.script
    assert _thinking(lane.sent[0]) and all(not _thinking(payload) for payload in lane.sent[1:])
    assert lane.sent[0]["thinking"] == lane.sent[1]["thinking"] == {"type": "disabled"}
    assert "thinking" not in lane.sent[2]
    assert omitted is answered
    if answered:
        assert not isinstance(result, BaseException), result
        assert result[1]["request_wire"]["applied_effort"] == "provider_default"
        state = json.loads(lane.wire_state.read_text())
        actions = [record["action"] for profile in state["profiles"].values() for record in profile["records"]]
        assert actions == [{"fields": ["thinking"], "kind": "drop_field", "reason_code": "provider_unsupported_field"}]
    else:
        assert isinstance(result, requests.HTTPError), result
        assert messages == original and not lane.wire_state.exists()


def test_invalid_json_from_the_stripped_retry_is_not_an_answered_omission(lane):
    messages = _history()
    original = copy.deepcopy(messages)
    lane.script[:] = [_Response(400), _Response(invalid_json=True)]
    result, omitted = _run(lane, messages, adopt=True)
    assert isinstance(result, ValueError) and "invalid response JSON" in str(result)
    assert len(lane.sent) == 2 and not lane.script
    assert omitted is False and messages == original and not lane.wire_state.exists()


def test_an_http_200_error_body_does_not_note_an_omission_or_change_native_normalization(lane):
    messages = _history()
    original = copy.deepcopy(messages)
    response = _Response()
    response.body = {"error": {"code": 502, "message": "Upstream unavailable."}}
    response.text = json.dumps(response.body)
    lane.script[:] = [_Response(400), response]
    result, omitted = _run(lane, messages, adopt=True)
    assert not isinstance(result, BaseException), result
    assert result[0]["content"] == ""  # the existing native normalizer still owns error-body behavior
    assert len(lane.sent) == 2 and not lane.script
    assert omitted is False and messages == original and not lane.wire_state.exists()
