"""The generic OpenAI-compatible lane sends prompt-cache hints and learns a server's refusal.

A compatible server uses the hints (OpenRouter or LiteLLM in front of Anthropic, DashScope),
ignores them (vLLM, SGLang, llama.cpp, Ollama) or refuses one with a 4xx that names it.
Unknown is not no: the lane sends the canonical block markers the OpenRouter ``anthropic/``
lane renders (at most four, the owner's TTL stamped by the send-time finalizer) and the
governance ``prompt_cache_key``. Request-wire recovery drops the narrowest named hint inside
the same call and keeps that fact for the exact route profile (endpoint, model, request
shape) for 14 days, and only once the repaired call succeeded.

The refusal bodies are recorded ones: Kimi K3 and GLM 5.2/5.3 behind one OpenAI-wire
gateway (FreePeak/onegw issue 156, probed 2026-09-29), OpenRouter's Claude 3 passthrough
(earendil-works/pi issue 7938, 2026-08-11, index ours); the ``extra_forbidden`` list is the
FastAPI/vLLM validation body, the ``prompt_cache_key`` line Azure OpenAI's unknown-argument
wording. Other direct lanes keep their strip, and policy, auth, quota and overflow
refusals never cost a hint.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import ProviderPolicyRefusal
from ouroboros.request_wire_recovery import (
    plan_wire_retry_from_exception,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
)
from ouroboros.usage_accounting import UsageScope, usage_scope

KIMI = {"error": {"message": "The parameter cache_control is not supported for the requested model",
                  "type": "invalid_request_error"}}
GLM = {"error": {"message": "Input should be a valid string, field: 'messages[0].content.str', value: "
                            "[{'type': 'text', ..., 'cache_control': ...}]; Extra inputs are not permitted",
                 "type": "invalid_request_error"}}
TTL_FORBIDDEN = {"object": "error", "type": "BadRequestError", "param": None, "code": 400, "message": (
    "[{'type': 'extra_forbidden', 'loc': ('body', 'messages', 0, 'content', 0, 'cache_control', 'ttl'), "
    "'msg': 'Extra inputs are not permitted', 'input': '1h'}]")}
TOOL_MARKER = {"error": {"message": "tools.1.cache_control: Extra inputs are not permitted",
                         "type": "invalid_request_error"}}
KEY_UNRECOGNIZED = {"error": {"message": "Unrecognized request argument supplied: prompt_cache_key",
                              "type": "invalid_request_error", "param": None, "code": None}}
# A strict validator refusing the one-part text list a marked tool row became, naming no field of ours.
UNATTRIBUTED = {"object": "error", "type": "BadRequestError", "param": None, "code": 400, "message": (
    "[{'type': 'string_type', 'loc': ('body', 'messages', 5, 'content', 'str'), "
    "'msg': 'Input should be a valid string', 'input': [{'type': 'text', 'text': 'second output'}]}]")}
UNRELATED = {"error": {"message": "Extra inputs are not permitted", "type": "invalid_request_error"}}

TOOLS = [{"type": "function", "function": {"name": name, "description": name,
                                           "parameters": {"type": "object", "properties": {}}}}
         for name in ("zeta_tool", "alpha_tool")]


class _Refusal(RuntimeError):
    """The OpenAI SDK's shape of a provider 4xx: ``status_code`` and the JSON ``body``."""

    def __init__(self, body, status=400):
        super().__init__(f"Error code: {status} - {json.dumps(body)}")
        self.status_code = status
        self.body = copy.deepcopy(body)


class _Response:
    def model_dump(self):
        return {"id": "resp-ok", "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.delenv("OUROBOROS_PROMPT_CACHE_TTL", raising=False)  # the shipped 1h tier
    return tmp_path


def _target(provider="openai-compatible", model="kimi-k3"):
    return {"provider": provider, "resolved_model": model, "usage_model": f"{provider}/{model}",
            "base_url": "https://gateway.example/v1", "supports_openrouter_extensions": False,
            "supports_generation_cost": False}


def _transcript(rounds=2):
    """Main's shape: two marked stable system blocks, the task, tool rounds, the rolling seal."""
    from ouroboros.loop import seal_task_transcript

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "governance prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "identity and story", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "dynamic tail"},
        ]},
        {"role": "user", "content": "task"},
    ]
    outputs = ["first output", [{"type": "text", "text": "second "}, {"type": "text", "text": "output"}],
               "third output"]
    for index in range(rounds):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{index}", "type": "function", "function": {"name": "zeta_tool", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{index}", "content": copy.deepcopy(outputs[index])})
    seal_task_transcript(messages, min_prefix_tokens=0)
    return messages


def _payload(target=None, messages=None):
    target = target or _target()
    client = LLMClient(api_key="unused")
    kwargs = client._build_remote_kwargs(copy.deepcopy(target), copy.deepcopy(messages or _transcript()),
                                         "high", 512, "auto", None, copy.deepcopy(TOOLS), skip_capability_fetch=True)
    return kwargs, client._normalize_payload_cache_ttl(target, kwargs)


def _markers(payload):
    return [holder["cache_control"] for holder in LLMClient._payload_cache_breakpoints(payload)]


def _send(tmp_path, label, refuses, refusal, *, target=None, sent=None):
    """One logical call through the production driver; returns (sends, usage)."""
    target = target or _target()
    kwargs = _payload(target)[0]
    sent = [] if sent is None else sent

    def create(**candidate):
        candidate.pop("timeout", None)
        sent.append(copy.deepcopy(candidate))
        if refuses(candidate):
            raise refusal
        return _Response()

    client = LLMClient(api_key="unused")
    with request_wire_call_scope(), usage_scope(UsageScope(drive_root=tmp_path / label, task_id=label)):
        response = client._create_chat_completion_with_retries(create, copy.deepcopy(kwargs), target)
        _msg, usage = client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True,
                                                        prompt_cache_ttl="1h")
    return sent, usage


def _learned(root):
    path = root / "state" / wire.REQUEST_WIRE_STATE_FILE
    store = json.loads(path.read_text()) if path.exists() else {"profiles": {}}
    return sorted(field for entry in store["profiles"].values() for record in entry["records"]
                  for field in record["action"].get("fields") or ())


def _has_markers(candidate):
    return "cache_control" in json.dumps(candidate)


# ---------------------------------------------------------------------------
# What the lane sends
# ---------------------------------------------------------------------------

def test_the_compatible_lane_sends_the_canonical_markers_and_the_governance_key(evidence_root):
    kwargs, ttl = _payload()
    messages = kwargs["messages"]
    ephemeral = {"type": "ephemeral", "ttl": "1h"}
    assert ttl == "1h" and _markers(kwargs) == [ephemeral] * 4  # A', B, the seal, the last schema
    assert [block.get("cache_control") for block in messages[0]["content"]] == [ephemeral, ephemeral, None]
    # Tool rows flatten as before; only the sealed row becomes a one-part list carrying the mark.
    assert messages[3] == {"role": "tool", "tool_call_id": "call_0", "content": "first output"}
    assert messages[5]["content"] == [{"type": "text", "text": "second output", "cache_control": ephemeral}]
    # A sealed transcript keeps user rows in one shape while the mark rolls.
    assert messages[1]["content"] == [{"type": "text", "text": "task"}]
    assert "cache_control" in kwargs["tools"][-1] and "cache_control" not in kwargs["tools"][0]
    assert re.fullmatch(r"ouroboros-[0-9a-f]{32}", kwargs["prompt_cache_key"])


def test_the_finalizer_owns_the_compatible_schema_marker_and_ttl(evidence_root):
    """The builder leaves tools unmarked; the send-time finalizer marks the last schema
    within the cap and stamps the owner's TTL, as on the other normalizing routes."""
    client = LLMClient(api_key="unused")
    review_pack = [{"role": "system", "content": [
        {"type": "text", "text": "stable governance", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "mutable evidence"}]}, {"role": "user", "content": "review this"}]
    kwargs = client._build_remote_kwargs(_target(), review_pack, "high", 512, "auto", None,
                                         copy.deepcopy(TOOLS), skip_capability_fetch=True)
    assert all("cache_control" not in tool for tool in kwargs["tools"])
    assert client._normalize_payload_cache_ttl(_target(), kwargs) == "1h"
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in kwargs["tools"][0]
    assert [block.get("cache_control") for block in kwargs["messages"][0]["content"]] == [
        {"type": "ephemeral", "ttl": "1h"}, None]
    assert kwargs["messages"][1]["content"] == "review this"  # an unsealed prompt keeps its shape


@pytest.mark.parametrize("setting,expected,report", [
    ("5m", {"type": "ephemeral", "ttl": "5m"}, "5m"),
    ("default", {"type": "ephemeral"}, "default"),
])
def test_the_owner_ttl_reaches_every_compatible_marker(evidence_root, monkeypatch, setting, expected, report):
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", setting)
    kwargs, ttl = _payload()
    assert ttl == report and _markers(kwargs) == [expected] * 4


def test_never_more_than_four_markers_and_no_schema_marker_without_a_slot(evidence_root):
    messages = _transcript()
    messages[0]["content"] = [{"type": "text", "text": f"block {index}", "cache_control": {"type": "ephemeral"}}
                              for index in range(4)]
    kwargs, _ttl = _payload(messages=messages)
    assert len(_markers(kwargs)) == 4
    assert all("cache_control" not in tool for tool in kwargs["tools"])
    assert "cache_control" not in json.dumps(kwargs["messages"][5])  # the cap keeps the earliest four


def test_the_next_round_extends_the_previous_prefix(evidence_root):
    """Rows before the previous boundary stay byte-identical; that row keeps its text and
    loses only the moved mark, as on the OpenRouter ``anthropic/`` lane."""
    before, _ = _payload(messages=_transcript(rounds=2))
    after, _ = _payload(messages=_transcript(rounds=3))
    boundary = 5
    assert after["messages"][:boundary] == before["messages"][:boundary]
    assert before["messages"][boundary]["content"][0]["text"] == after["messages"][boundary]["content"] == "second output"
    assert "cache_control" in json.dumps(after["messages"][-1])
    assert after["prompt_cache_key"] == before["prompt_cache_key"]


@pytest.mark.parametrize("provider", ["openai", "deepseek", "zai", "minimax", "cloudru"])
def test_the_other_direct_lanes_keep_their_strip(evidence_root, provider):
    target = _target(provider, "some-model")
    kwargs, ttl = _payload(target)
    assert ttl is None and _markers(kwargs) == [] and not _has_markers(kwargs)
    assert all(isinstance(m["content"], str) for m in kwargs["messages"] if m["role"] == "tool")
    assert kwargs["messages"][1]["content"] == "task"
    assert ("prompt_cache_key" in kwargs) is (provider == "openai")


def test_the_local_projection_keeps_its_strip():
    copied = LLMClient._copy_messages_with_cache_policy(
        _transcript(), allow_message_cache_control=False, flatten_tool_content_blocks=True)
    assert not _has_markers(copied) and copied[1]["content"] == "task"
    assert [m["content"] for m in copied if m["role"] == "tool"] == ["first output", "second output"]


# ---------------------------------------------------------------------------
# A recorded refusal is repaired in the same call and learned for that route
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body", [KIMI, GLM], ids=["kimi-k3", "glm"])
def test_a_named_cache_control_refusal_drops_every_marker_and_is_learned(evidence_root, tmp_path, body):
    sent, usage = _send(tmp_path, "named", _has_markers, _Refusal(body))
    assert len(sent) == 2 and _has_markers(sent[0]) and not _has_markers(sent[1])
    # The marker-free wire is the lane's former shape: tool and user rows are plain strings.
    assert sent[1]["messages"][5]["content"] == "second output" and sent[1]["messages"][1]["content"] == "task"
    assert sent[1]["prompt_cache_key"] == sent[0]["prompt_cache_key"]
    assert '"cache_control"' in json.dumps(usage["request_wire"]["applied_actions"])
    assert "prompt_cache_ttl" not in usage  # the terminal candidate carried no marker
    assert _learned(evidence_root) == ["cache_control"]


def test_a_refused_ttl_drops_only_the_ttl(evidence_root, tmp_path):
    sent, usage = _send(tmp_path, "ttl", lambda candidate: '"ttl"' in json.dumps(candidate), _Refusal(TTL_FORBIDDEN))
    assert len(sent) == 2 and _markers(sent[1]) == [{"type": "ephemeral"}] * 4
    assert usage["prompt_cache_ttl"] == "default"  # what the repaired candidate carried, not the 1h asked for
    assert _learned(evidence_root) == ["cache_control.ttl"]


def test_a_refused_tool_marker_drops_only_the_tool_marker(evidence_root, tmp_path):
    sent, _usage = _send(tmp_path, "tools", lambda c: any("cache_control" in tool for tool in c.get("tools") or []),
                         _Refusal(TOOL_MARKER))
    assert len(sent) == 2 and all("cache_control" not in tool for tool in sent[1]["tools"])
    assert len(_markers(sent[1])) == 3  # the message markers stay
    assert _learned(evidence_root) == ["tools.cache_control"]


def test_an_unrecognized_prompt_cache_key_drops_only_the_key(evidence_root, tmp_path):
    sent, _usage = _send(tmp_path, "key", lambda c: "prompt_cache_key" in c, _Refusal(KEY_UNRECOGNIZED))
    assert len(sent) == 2 and "prompt_cache_key" not in sent[1] and len(_markers(sent[1])) == 4
    assert _learned(evidence_root) == ["prompt_cache_key"]


def test_a_field_less_validation_refusal_gets_one_marker_free_retry(evidence_root, tmp_path):
    def refuses(candidate):
        return any(m["role"] == "tool" and isinstance(m["content"], list) for m in candidate["messages"])

    sent, _usage = _send(tmp_path, "unattributed", refuses, _Refusal(UNATTRIBUTED))
    assert len(sent) == 2 and not _has_markers(sent[1]) and not refuses(sent[1])
    assert _learned(evidence_root) == ["cache_control"]


def test_an_unrelated_refusal_costs_one_marker_free_retry_and_teaches_nothing(evidence_root, tmp_path):
    sent = []
    with pytest.raises(_Refusal):
        _send(tmp_path, "unrelated", lambda _candidate: True, _Refusal(UNRELATED), sent=sent)
    assert len(sent) == 2 and _has_markers(sent[0]) and not _has_markers(sent[1])
    assert _learned(evidence_root) == []  # the provider's own error stands


# ---------------------------------------------------------------------------
# The learned fact's scope and lifetime
# ---------------------------------------------------------------------------

def _accepted(_candidate):
    return False  # a server that answers everything


def test_a_learned_refusal_scopes_to_its_route_and_expires(evidence_root, tmp_path):
    _send(tmp_path, "learn", _has_markers, _Refusal(KIMI))
    accepted = _accepted
    sent, _usage = _send(tmp_path, "same-route", accepted, None)
    assert len(sent) == 1 and not _has_markers(sent[0])  # no refusal paid twice
    other = _target(model="kimi-k2.7")
    sent, _usage = _send(tmp_path, "other-model", accepted, None, target=other)
    assert _has_markers(sent[0])  # the same gateway's other model is judged on its own
    path = evidence_root / "state" / wire.REQUEST_WIRE_STATE_FILE
    store = json.loads(path.read_text())
    for entry in store["profiles"].values():
        for record in entry["records"]:
            record["observed_at"] = "2026-01-01T00:00:00+00:00"
    path.write_text(json.dumps(store))
    sent, _usage = _send(tmp_path, "expired", accepted, None)
    assert _has_markers(sent[0])  # after 14 days the hints return


# ---------------------------------------------------------------------------
# Refusals of another class never cost a hint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,body", [
    (401, {"error": {"message": "cache_control: Extra inputs are not permitted"}}),
    (402, {"error": {"message": "cache_control: Extra inputs are not permitted"}}),
    (403, {"error": {"message": "cache_control: Extra inputs are not permitted"}}),
    (429, {"error": {"message": "cache_control: Extra inputs are not permitted"}}),
    (400, {"error": {"code": "context_length_exceeded", "message": "cache_control: Extra inputs are not permitted"}}),
    (400, {"error": {"code": "content_policy_violation", "message": "cache_control is not supported"}}),
], ids=["401", "402", "403", "429", "overflow", "policy"])
def test_auth_quota_overflow_and_policy_refusals_never_strip_hints(evidence_root, status, body):
    kwargs, _ttl = _payload()
    with request_wire_call_scope():
        prepare_wire_payload_for_send(_target(), kwargs, api_surface="chat.completions")
        assert plan_wire_retry_from_exception(_Refusal(body, status)) is None
    with request_wire_call_scope():  # the same words on an ordinary 400 do repair: the guard is not vacuous
        prepare_wire_payload_for_send(_target(), kwargs, api_surface="chat.completions")
        repaired = plan_wire_retry_from_exception(_Refusal({"error": {"message": body["error"]["message"]}}))
    assert repaired is not None and not _has_markers(repaired)


def test_a_typed_policy_refusal_is_never_resent_without_hints(evidence_root, tmp_path):
    sent = []
    with pytest.raises(ProviderPolicyRefusal):
        _send(tmp_path, "policy", lambda _candidate: True, ProviderPolicyRefusal("connection not permitted"), sent=sent)
    assert len(sent) == 1 and _has_markers(sent[0]) and _learned(evidence_root) == []


def test_a_compatible_servers_cache_counters_reach_usage_and_the_ledger():
    from ouroboros._usage_response import usage_from_response

    body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {
        "prompt_tokens": 100, "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10}}}
    _msg, usage = LLMClient(api_key="unused")._normalize_remote_response(
        copy.deepcopy(body), _target(), skip_cost_fetch=True)
    assert (usage["cached_tokens"], usage["cache_write_tokens"]) == (80, 10)
    normalized, _cost, _final = usage_from_response(body)
    assert (normalized["cached_tokens"], normalized["cache_write_tokens"]) == (80, 10)


def test_settings_names_a_learned_refusal_on_its_endpoint_until_it_expires(evidence_root, tmp_path):
    from ouroboros.provider_connections import connections_meta

    base = "https://gateway.example/v1"
    assert wire.endpoint_cache_hint_refusal("openai-compatible", base) == ""
    _send(tmp_path, "learn", _has_markers, _Refusal(KIMI))
    learned = wire.endpoint_cache_hint_refusal("openai-compatible", base)
    assert learned
    assert wire.endpoint_cache_hint_refusal("openai-compatible", "https://GATEWAY.example/v1/") == learned
    assert wire.endpoint_cache_hint_refusal("openai-compatible", "https://other.example/v1") == ""
    assert wire.endpoint_cache_hint_refusal("openai", base) == ""
    rows = [{"kind": "openai-compatible", "name": "gateway", "base_url": base, "api_key": "fixture-key"},
            {"kind": "openai-compatible", "name": "local", "base_url": "http://127.0.0.1:8000/v1"}]
    settings = {"OUROBOROS_CONNECTIONS": rows, "OPENAI_COMPATIBLE_BASE_URL": base}
    meta = {entry["id"]: entry for entry in connections_meta(settings, settings)}
    assert meta["openai-compatible@gateway"]["cache_hints"] == {"refused": True, "learned_at": learned}
    assert meta["openai-compatible"]["cache_hints"] == {"refused": True, "learned_at": learned}  # same server
    assert "cache_hints" not in meta["openai-compatible@local"]
    path = evidence_root / "state" / wire.REQUEST_WIRE_STATE_FILE
    store = json.loads(path.read_text())
    records = [record for entry in store["profiles"].values() for record in entry["records"]]
    for record in records:
        record["observed_at"] = "2026-01-01T00:00:00+00:00"
    path.write_text(json.dumps(store))
    assert wire.endpoint_cache_hint_refusal("openai-compatible", base) == ""  # 14 days later it is forgotten
    records[0].update(observed_at=learned, action={"kind": "drop_field", "fields": ["temperature"],
                                                    "reason_code": "provider_unsupported_field"})
    path.write_text(json.dumps(store))
    assert wire.endpoint_cache_hint_refusal("openai-compatible", base) == ""  # another field taught nothing about hints
