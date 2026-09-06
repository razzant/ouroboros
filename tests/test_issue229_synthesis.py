"""Cross-stream regressions for the issue #229 synthesis driver."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import ouroboros.context_compaction as context_compaction
import ouroboros.llm_fallback as llm_fallback
import ouroboros.llm_observability as llm_observability
import ouroboros.request_wire_contract as wire_contract
from ouroboros.llm import LLMClient, add_usage
from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY
from ouroboros.request_wire_contract import (
    build_request_wire_profile,
    canonical_sha256,
    read_wire_action_records,
)
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture,
    PhysicalAttemptLimitExceeded,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return copy.deepcopy(self.payload)


class _Rejected(RuntimeError):
    def __init__(self, message: str, *, param: str):
        super().__init__(message)
        self.status_code = 400
        self.code = "unsupported_parameter"
        self.param = param
        self.body = {
            "error": {
                "code": self.code,
                "param": param,
                "message": message,
            },
        }


def _target():
    return {
        "provider": "openai",
        "resolved_model": "future-reasoning-model",
        "usage_model": "openai/future-reasoning-model",
        "base_url": "https://api.openai.com/v1",
        "contract_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Return the marker.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"marker": {"type": "string", "enum": ["ok"]}},
                "required": ["marker"],
                "additionalProperties": False,
            },
        },
    }]


def _custom_success(call_id="call-custom"):
    return _Response({
        "id": f"response-{call_id}",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "custom",
                    "custom": {"name": "probe", "input": '{"marker":"ok"}'},
                    "function": None,
                }],
            },
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 5},
    })


def _text_success():
    return _Response({
        "id": "response-text",
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 3},
    })


def _body_rejection(message, *, param):
    return _Response({
        "error": {
            "status_code": 400,
            "code": "unsupported_parameter",
            "param": param,
            "message": message,
        },
    })


def _install_transport(monkeypatch, responses, *, max_sends=None):
    sent = []
    queue = list(responses)
    capture = {"value": None}

    def create(**payload):
        sent.append(copy.deepcopy(payload))
        result = queue.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    client = LLMClient(api_key="test")
    remote = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: _target())
    monkeypatch.setattr(client, "_get_remote_client", lambda _target_value: remote)

    def execute(request, send, before_dispatch=None):
        del before_dispatch
        if max_sends is not None and len(sent) >= max_sends:
            raise PhysicalAttemptLimitExceeded("test physical attempt rail exhausted")
        response = send()
        attempt_id = f"attempt-{len(sent)}"
        capture["value"] = PhysicalAttemptCapture(
            attempt_id=attempt_id,
            model=request.model,
            provider=request.provider,
            state="settled",
            candidate_measurement_kind="canonical_json_v1",
            candidate_raw_sha256=request.candidate_raw_sha256,
            candidate_manifest_ref={
                "path": f"physical/{attempt_id}.json",
                "call_id": attempt_id,
                "sha256": canonical_sha256(attempt_id),
            },
        )
        return response

    # The v7 split moved the remote send drivers into llm_fallback.py, which is
    # where `_execute_candidate` / `last_physical_attempt_capture` are read on
    # the chat path; patching the names llm.py merely re-exports would be dead.
    monkeypatch.setattr(llm_fallback, "_execute_candidate", execute)
    monkeypatch.setattr(
        llm_fallback, "last_physical_attempt_capture", lambda: capture["value"],
    )
    return client, sent


def test_custom_first_generic_param_recovery_commits_bound_action(tmp_path, monkeypatch):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    client, sent = _install_transport(monkeypatch, [
        _Rejected("temperature is not supported", param="temperature"),
        _custom_success("call-first"),
        _custom_success("call-second"),
    ])

    _message, usage = client.chat(
        [{"role": "user", "content": "Use the probe."}],
        "openai::future-reasoning-model",
        tools=_tools(),
        tool_choice="required",
        reasoning_effort="medium",
        temperature=0.2,
    )
    assert [(item["tools"][0]["type"], item.get("reasoning_effort")) for item in sent] == [
        ("custom", "medium"),
        ("custom", "medium"),
    ]
    assert sent[0]["temperature"] == 0.2
    assert "temperature" not in sent[1]
    disclosure = usage["request_wire"]
    assert disclosure["ladder_ordinal"] == 1
    assert disclosure["applied_tool_dialect"] == "openai_chat_custom"
    assert disclosure["applied_effort"] == "medium"
    assert disclosure["applied_actions"][0]["source"] == "pending"

    client.chat(
        [{"role": "user", "content": "Use the probe again."}],
        "openai::future-reasoning-model",
        tools=_tools(),
        tool_choice="required",
        reasoning_effort="medium",
        temperature=0.2,
    )
    assert sent[2]["tools"][0]["type"] == "custom"
    assert "temperature" not in sent[2]


def test_custom_reject_function_same_rung_param_repair_never_reaches_none(
    tmp_path, monkeypatch,
):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    client, sent = _install_transport(monkeypatch, [
        _Rejected("custom tools are not supported", param="tools[0].type"),
        _Rejected("temperature is not supported", param="temperature"),
        _text_success(),
    ])

    _message, usage = client.chat(
        [{"role": "user", "content": "Use a tool if needed."}],
        "openai::future-reasoning-model",
        tools=_tools(),
        reasoning_effort="medium",
        temperature=0.2,
    )
    assert [(item["tools"][0]["type"], item["reasoning_effort"]) for item in sent] == [
        ("custom", "medium"),
        ("function", "medium"),
        ("function", "medium"),
    ]
    assert "temperature" not in sent[2]
    disclosure = usage["request_wire"]
    assert disclosure["ladder_ordinal"] == 2
    assert disclosure["reason_code"] == "provider_rejected_tool_dialect"
    assert disclosure["applied_effort"] == "medium"
    assert disclosure["task_local"] is False
    assert all(
        item["action"].get("to") != "none"
        for item in disclosure["applied_actions"]
    )
    rejected_profile = build_request_wire_profile(
        _target(), sent[1], api_surface="chat.completions",
    )
    assert any(
        record.action.get("kind") == "drop_field"
        and "temperature" in set(record.action.get("fields") or ())
        for record in read_wire_action_records(rejected_profile)
    )


def test_function_value_evidence_never_changes_next_custom_first_candidate(
    tmp_path, monkeypatch,
):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    client, sent = _install_transport(monkeypatch, [
        _Rejected("custom tools are not supported", param="tools[0].type"),
        _Rejected(
            "reasoning_effort value 'high' is not supported; use medium",
            param="reasoning_effort",
        ),
        _text_success(),
        _Rejected("custom tools are not supported", param="tools[0].type"),
        _text_success(),
    ])
    kwargs = {
        "messages": [{"role": "user", "content": "Use the probe."}],
        "model": "openai::future-reasoning-model",
        "tools": _tools(),
        "reasoning_effort": "high",
    }

    client.chat(**kwargs)
    client.chat(**kwargs)

    assert [
        (item["tools"][0]["type"], item.get("reasoning_effort"))
        for item in sent
    ] == [
        ("custom", "high"),
        ("function", "high"),
        ("function", "medium"),
        ("custom", "high"),
        ("function", "medium"),
    ]


@pytest.mark.parametrize("body_error", [False, True])
def test_custom_pending_repair_is_discarded_before_fresh_function(
    tmp_path, monkeypatch, body_error,
):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    reject = _body_rejection if body_error else _Rejected
    client, sent = _install_transport(monkeypatch, [
        reject("temperature is not supported", param="temperature"),
        reject("custom tools are not supported", param="tools[0].type"),
        _text_success(),
    ])

    _message, usage = client.chat(
        [{"role": "user", "content": "Use the probe."}],
        "openai::future-reasoning-model",
        tools=_tools(),
        reasoning_effort="medium",
        temperature=0.2,
    )

    assert [item["tools"][0]["type"] for item in sent] == [
        "custom", "custom", "function",
    ]
    assert "temperature" in sent[0]
    assert "temperature" not in sent[1]
    assert "temperature" in sent[2]
    assert usage["request_wire"]["ladder_ordinal"] == 2
    assert usage["request_wire"]["applied_actions"] == []


@pytest.mark.parametrize("body_error", [False, True])
def test_task_local_none_composes_nonlearning_same_rung_repair(
    tmp_path, monkeypatch, body_error,
):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    reject = _body_rejection if body_error else _Rejected
    client, sent = _install_transport(monkeypatch, [
        reject("custom tools are not supported", param="tools[0].type"),
        reject(
            "reasoning is not compatible with function tools; must use none",
            param="reasoning_effort",
        ),
        reject("temperature is not supported", param="temperature"),
        _text_success(),
    ])

    _message, usage = client.chat(
        [{"role": "user", "content": "Use a tool if needed."}],
        "openai::future-reasoning-model",
        tools=_tools(),
        reasoning_effort="medium",
        temperature=0.2,
    )

    assert [
        (item["tools"][0]["type"], item["reasoning_effort"], "temperature" in item)
        for item in sent
    ] == [
        ("custom", "medium", True),
        ("function", "medium", True),
        ("function", "none", True),
        ("function", "none", False),
    ]
    disclosure = usage["request_wire"]
    assert disclosure["ladder_ordinal"] == 3
    assert disclosure["task_local"] is True
    assert [item["source"] for item in disclosure["applied_actions"]] == [
        "task_local", "task_local",
    ]
    rejected_profile = build_request_wire_profile(
        _target(), sent[2], api_surface="chat.completions",
    )
    assert read_wire_action_records(rejected_profile) == ()


def test_task_local_same_rung_repair_still_obeys_physical_attempt_rail(
    tmp_path, monkeypatch,
):
    evidence_root = tmp_path / "wire-evidence"
    monkeypatch.setattr(
        wire_contract, "canonical_wire_evidence_root", lambda: evidence_root,
    )
    client, sent = _install_transport(monkeypatch, [
        _Rejected("custom tools are not supported", param="tools[0].type"),
        _Rejected(
            "reasoning is not compatible with function tools; must use none",
            param="reasoning_effort",
        ),
        _Rejected("temperature is not supported", param="temperature"),
        _text_success(),
    ], max_sends=3)

    with pytest.raises(PhysicalAttemptLimitExceeded):
        client.chat(
            [{"role": "user", "content": "Use a tool if needed."}],
            "openai::future-reasoning-model",
            tools=_tools(),
            reasoning_effort="medium",
            temperature=0.2,
        )

    assert len(sent) == 3
    rejected_profile = build_request_wire_profile(
        _target(), sent[2], api_surface="chat.completions",
    )
    assert read_wire_action_records(rejected_profile) == ()


def test_ordered_request_wire_history_survives_main_and_structured_compaction(
    tmp_path, monkeypatch,
):
    first = {
        "attempt_id": "attempt-main",
        "candidate_sha256": "a" * 64,
        "requested_effort": "medium",
        "applied_effort": "medium",
    }
    second = {
        "attempt_id": "attempt-compaction",
        "candidate_sha256": "b" * 64,
        "requested_effort": "medium",
        "applied_effort": "medium",
    }
    usage_first = {
        "request_wire": first,
        CUSTOM_RECEIPTS_USAGE_KEY: (object(),),
    }
    usage_second = {"request_wire": second}

    persisted = []
    monkeypatch.setattr(
        llm_observability,
        "persist_call",
        lambda *args, **kwargs: persisted.append(copy.deepcopy(kwargs)),
    )

    class _FakeLLM:
        def chat(self, **kwargs):
            del kwargs
            return {"role": "assistant", "content": "ok"}, usage_first

    _message, returned = llm_observability.chat_observed(
        _FakeLLM(),
        drive_root=tmp_path,
        task_id="synthesis",
        model="openai::future-reasoning-model",
    )
    response_payload = next(
        item["payload"] for item in persisted
        if item["call_type"].endswith("_response")
    )
    assert CUSTOM_RECEIPTS_USAGE_KEY not in response_payload["usage"]
    assert CUSTOM_RECEIPTS_USAGE_KEY in returned

    main_total = {}
    add_usage(main_total, usage_first)
    compaction_total = {}
    context_compaction._record_usage(compaction_total, usage_second)
    add_usage(main_total, compaction_total)
    add_usage(main_total, compaction_total)

    assert main_total["request_wire"] == second
    assert main_total["request_wire_history"] == [first, second]
