"""Direct-OpenAI Chat custom physical-send and consumer contracts."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

import ouroboros.context_compaction as context_compaction
import ouroboros.llm_fallback as llm_fallback
import ouroboros.openai_chat_dispatch as dispatch
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.llm import LLMClient
from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    _execute_single_tool,
    handle_tool_calls,
)
from ouroboros.openai_chat_custom import normalize_openai_custom_tool_calls
from ouroboros.request_wire_contract import (
    PendingWireAction,
    build_request_wire_profile,
    canonical_sha256,
)
from ouroboros.request_wire_receipts import (
    WireAppliedAction,
    WireCandidateSpec,
    bind_wire_candidate,
    direct_openai_tool_candidate_ladder,
)
from ouroboros.request_wire_recovery import (
    register_wire_candidate,
    request_wire_call_scope,
)
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture,
    PhysicalAttemptLimitExceeded,
)


def _target(model: str = "future-model-without-prefix"):
    return {
        "provider": "openai",
        "resolved_model": model,
        "usage_model": f"openai/{model}",
        "base_url": "https://api.openai.com/v1",
        "default_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "description": "Submit an exact marker.",
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"const": "ok"}},
                    "required": ["marker"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "second",
                "description": "Submit an integer.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        },
    ]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return copy.deepcopy(self.payload)


class _DialectError(Exception):
    def __init__(self, message: str, *, param: str):
        super().__init__(message)
        self.status_code = 400
        self.code = "unsupported_value"
        self.param = param
        self.body = {
            "error": {
                "code": "unsupported_value",
                "param": param,
                "message": message,
            }
        }


def _tool_response(*calls):
    return _Response({
        "id": "response-tools",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": list(calls),
            }
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 12},
    })


def _text_response(text="done"):
    return _Response({
        "id": "response-text",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 5},
    })


def _body_dialect_error():
    return _Response({
        "error": {
            "status_code": 400,
            "code": "unsupported_value",
            "param": "tools[0].type",
            "message": "custom tools are not supported",
        }
    })


def _body_parameter_error():
    return _Response({
        "error": {
            "status_code": 400,
            "code": "unsupported_parameter",
            "param": "temperature",
            "message": "Unsupported parameter: temperature",
        }
    })


def _custom_call(call_id, name, raw_input):
    return {
        "id": call_id,
        "type": "custom",
        "custom": {"name": name, "input": raw_input},
        "function": None,
    }


def _install_transport(monkeypatch, client, responses, *, no_proxy=False, max_sends=None):
    captured = []
    queue = list(responses)
    capture_holder = {"value": None}

    def create(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    endpoint = SimpleNamespace(create=create)
    remote = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: _target())
    closer = SimpleNamespace(close=lambda: None)
    if no_proxy:
        monkeypatch.setattr(client, "_make_no_proxy_client", lambda *_a, **_k: (remote, closer))
    else:
        monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)

    sends = {"count": 0}

    def execute(request, send, before_dispatch=None):
        if max_sends is not None and sends["count"] >= max_sends:
            raise PhysicalAttemptLimitExceeded("test physical attempt rail exhausted")
        sends["count"] += 1
        response = send()
        attempt_id = f"attempt-{sends['count']}"
        capture_holder["value"] = PhysicalAttemptCapture(
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
        llm_fallback,
        "last_physical_attempt_capture",
        lambda: capture_holder["value"],
    )
    return captured


def test_public_chat_model_agnostic_custom_normalize_and_continuation(monkeypatch):
    first = _tool_response(
        _custom_call("call-1", "probe", '{"marker":"ok"}'),
        _custom_call("call-2", "second", '{"value":2}'),
    )
    client = LLMClient(api_key="test")
    captured = _install_transport(monkeypatch, client, [first, _text_response()])
    initial = [{"role": "user", "content": "Use both tools."}]
    initial_snapshot = copy.deepcopy(initial)

    message, usage = client.chat(
        initial,
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
        tool_choice="required",
    )

    assert initial == initial_snapshot
    assert [tool["type"] for tool in captured[0]["tools"]] == ["custom", "custom"]
    assert captured[0]["reasoning_effort"] == "medium"
    assert captured[0]["max_completion_tokens"] == 65536
    assert "max_tokens" not in captured[0]
    assert [call["type"] for call in message["tool_calls"]] == ["function", "function"]
    disclosure = usage["request_wire"]
    assert disclosure["requested_effort"] == disclosure["applied_effort"] == "medium"
    assert disclosure["requested_tool_dialect"] == "function"
    assert disclosure["applied_tool_dialect"] == "openai_chat_custom"
    assert disclosure["reason_code"] == "requested_wire_form"
    assert disclosure["ladder_ordinal"] == 1
    receipts = dispatch.pop_custom_validation_receipts(usage, message["tool_calls"])
    assert len(receipts) == 2 and all(item.allows_execution for item in receipts)

    history = [
        *initial,
        message,
        {"role": "tool", "tool_call_id": "call-1", "content": "first-result"},
        {"role": "tool", "tool_call_id": "call-2", "content": "second-result"},
    ]
    canonical_snapshot = copy.deepcopy(history)
    final, final_usage = client.chat(
        history,
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
        tool_choice="none",
    )

    assert final["content"] == "done"
    assert history == canonical_snapshot
    physical_history = captured[1]["messages"]
    assert [call["type"] for call in physical_history[1]["tool_calls"]] == ["custom", "custom"]
    assert physical_history[2:] == history[2:]
    final_disclosure = final_usage["request_wire"]
    assert final_disclosure["requested_effort"] == "medium"
    assert final_disclosure["applied_effort"] == "medium"
    assert final_disclosure["applied_tool_dialect"] == "openai_chat_custom"
    assert final_disclosure["reason_code"] == "requested_wire_form"
    assert final_disclosure["applied_actions"] == []
    assert dispatch.CUSTOM_RECEIPTS_USAGE_KEY not in final_usage


def test_pending_custom_replacement_text_cannot_bind_compatibility():
    source = {
        "model": "future-model-without-prefix",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "reasoning_effort": "medium",
        "tool_choice": "auto",
        "tools": _tools(),
    }
    replacement = PendingWireAction(
        build_request_wire_profile(
            _target(),
            source,
            api_surface="chat.completions",
        ),
        {
            "kind": "replace_dialect",
            "axis": "tool",
            "from": "function",
            "to": "openai_chat_custom",
            "reason_code": "provider_rejected_tool_dialect",
        },
    )
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom",
            "medium",
            "provider_rejected_tool_dialect",
        ),
        requested_effort="medium",
        ladder_ordinal=2,
        applied_actions=(WireAppliedAction.pending(replacement),),
    )
    with request_wire_call_scope():
        register_wire_candidate(candidate, source_payload=source, target=_target())
        with pytest.raises(ValueError, match="requires a custom tool call"):
            dispatch.normalize_direct_openai_completion(
                {"role": "assistant", "content": "done"}, {}, None,
            )


def test_exact_rejections_preserve_custom_function_none_reason_order(monkeypatch):
    client = LLMClient(api_key="test")
    ladder = direct_openai_tool_candidate_ladder(
        "medium", remaining_physical_attempts=3,
    )
    assert [item.reason_code for item in ladder] == [
        "requested_wire_form",
        "provider_rejected_tool_dialect",
        "task_local_availability_fallback",
    ]
    captured = _install_transport(monkeypatch, client, [
        _DialectError("custom tools are not supported", param="tools[0].type"),
        _DialectError(
            "reasoning is not compatible with function tools; must use none",
            param="reasoning_effort",
        ),
        _text_response("degraded"),
    ])

    message, usage = client.chat(
        [{"role": "user", "content": "Use a tool."}],
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
    )

    assert message["content"] == "degraded"
    assert [(item["tools"][0]["type"], item.get("reasoning_effort")) for item in captured] == [
        ("custom", "medium"),
        ("function", "medium"),
        ("function", "none"),
    ]
    disclosure = usage["request_wire"]
    assert disclosure["reason_code"] == "task_local_availability_fallback"
    assert disclosure["task_local"] is True
    assert disclosure["ladder_ordinal"] == 3
    assert disclosure["requested_effort"] == "medium"
    assert disclosure["applied_effort"] == "none"
    assert len(disclosure["applied_actions"]) == 1
    assert disclosure["applied_actions"][0]["source"] == "task_local"
    assert (
        disclosure["applied_actions"][0]["action"]["reason_code"]
        == "task_local_availability_fallback"
    )


@pytest.mark.parametrize(
    "rejection",
    [
        _DialectError("custom tools are not supported", param="tools[0].type"),
        _body_dialect_error(),
    ],
)
def test_exact_custom_rejection_uses_function_with_same_effort(
    monkeypatch,
    rejection,
):
    client = LLMClient(api_key="test")
    captured = _install_transport(monkeypatch, client, [rejection, _text_response()])

    _message, usage = client.chat(
        [{"role": "user", "content": "Use a tool."}],
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
    )

    assert [(item["tools"][0]["type"], item["reasoning_effort"]) for item in captured] == [
        ("custom", "medium"),
        ("function", "medium"),
    ]
    disclosure = usage["request_wire"]
    assert disclosure["reason_code"] == "provider_rejected_tool_dialect"
    assert disclosure["ladder_ordinal"] == 2
    assert disclosure["applied_effort"] == "medium"


def test_body_parameter_recovery_stays_on_the_same_custom_rung(monkeypatch):
    client = LLMClient(api_key="test")
    captured = _install_transport(
        monkeypatch,
        client,
        [_body_parameter_error(), _text_response()],
    )

    _message, usage = client.chat(
        [{"role": "user", "content": "Use a tool."}],
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
        temperature=0.2,
    )

    assert [item["tools"][0]["type"] for item in captured] == ["custom", "custom"]
    assert captured[0]["temperature"] == 0.2
    assert "temperature" not in captured[1]
    assert usage["request_wire"]["reason_code"] == "requested_wire_form"
    assert usage["request_wire"]["ladder_ordinal"] == 1


def test_caller_physical_rail_blocks_none_without_reordering(monkeypatch):
    client = LLMClient(api_key="test")
    captured = _install_transport(monkeypatch, client, [
        _DialectError("custom tools are not supported", param="tools[0].type"),
        _DialectError(
            "reasoning is not compatible with function tools; must use none",
            param="reasoning_effort",
        ),
    ], max_sends=2)

    with pytest.raises(PhysicalAttemptLimitExceeded):
        client.chat(
            [{"role": "user", "content": "Use a tool."}],
            "openai::future-model-without-prefix",
            tools=_tools(),
            reasoning_effort="medium",
        )
    assert [item["tools"][0]["type"] for item in captured] == ["custom", "function"]


def test_no_proxy_uses_the_same_custom_physical_send(monkeypatch):
    client = LLMClient(api_key="test")
    captured = _install_transport(monkeypatch, client, [_text_response()], no_proxy=True)
    _message, usage = client.chat(
        [{"role": "user", "content": "Answer or call."}],
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
        no_proxy=True,
    )
    assert captured[0]["tools"][0]["type"] == "custom"
    assert usage["request_wire"]["applied_tool_dialect"] == "openai_chat_custom"


def test_public_async_api_still_rejects_tool_calls():
    client = LLMClient(api_key="test")
    with pytest.raises(ValueError, match="does not support tool calls"):
        asyncio.run(client.chat_async(
            [{"role": "user", "content": "Use a tool."}],
            "openai::future-model-without-prefix",
            tools=_tools(),
            reasoning_effort="medium",
        ))


@pytest.mark.parametrize(
    ("choice", "expected_choice", "expected_names"),
    [
        (
            {"type": "function", "function": {"name": "probe"}},
            {"type": "custom", "custom": {"name": "probe"}},
            ["probe", "second"],
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "function", "name": "probe"}],
                },
            },
            "required",
            ["probe"],
        ),
    ],
)
def test_public_chat_projects_named_and_allowed_choices(
    monkeypatch,
    choice,
    expected_choice,
    expected_names,
):
    client = LLMClient(api_key="test")
    captured = _install_transport(monkeypatch, client, [_tool_response(
        _custom_call("choice-call", "probe", '{"marker":"ok"}'),
    )])

    message, usage = client.chat(
        [{"role": "user", "content": "Choose exactly."}],
        "openai::future-model-without-prefix",
        tools=_tools(),
        reasoning_effort="medium",
        tool_choice=choice,
    )

    assert captured[0]["tool_choice"] == expected_choice
    assert [item["custom"]["name"] for item in captured[0]["tools"]] == expected_names
    assert usage["request_wire"]["applied_actions"] == []
    assert dispatch.pop_custom_validation_receipts(usage, message["tool_calls"])[0].allows_execution


def test_generic_effort_value_rejection_is_not_a_tool_dialect_rejection():
    source = {
        "model": "future-model-without-prefix",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "reasoning_effort": "medium",
        "tool_choice": "auto",
        "tools": _tools(),
    }
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "medium", "requested_wire_form",
        ),
        requested_effort="medium",
        ladder_ordinal=1,
    )
    error = _DialectError(
        "reasoning_effort value medium is unsupported for this model",
        param="reasoning_effort",
    )
    assert dispatch.exact_tool_dialect_rejection(error, candidate) is False
    linked = _DialectError(
        "reasoning is not compatible with tools",
        param="reasoning_effort",
    )
    assert dispatch.exact_tool_dialect_rejection(linked, candidate) is False
    exact = _DialectError(
        "reasoning is not compatible with custom tools",
        param="reasoning_effort",
    )
    assert dispatch.exact_tool_dialect_rejection(exact, candidate) is True


def _invalid_custom_exchange(tool=None, raw_input='{"marker":"wrong"}'):
    selected_tool = copy.deepcopy(tool or _tools()[0])
    tool_name = selected_tool["function"]["name"]
    source = {
        "model": "future-model-without-prefix",
        "messages": [{"role": "user", "content": f"Use {tool_name}."}],
        "reasoning_effort": "medium",
        "tool_choice": "required",
        "tools": [selected_tool],
    }
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "medium", "requested_wire_form"
        ),
        requested_effort="medium",
        ladder_ordinal=1,
    )
    calls, receipts = normalize_openai_custom_tool_calls([_custom_call(
        "call-invalid", tool_name, raw_input,
    )], candidate)
    return {"role": "assistant", "content": "", "tool_calls": calls}, receipts


def _receipt_usage(receipts):
    return {
        dispatch.REQUEST_WIRE_USAGE_KEY: {
            "candidate_sha256": receipts[0].candidate_sha256,
        },
        dispatch.CUSTOM_RECEIPTS_USAGE_KEY: receipts,
    }


class _FakeTools:
    CODE_TOOLS = set()

    def __init__(self):
        self.calls = []
        self._ctx = SimpleNamespace(task_metadata={})

    def get_timeout(self, _name):
        return 10

    def execute(self, name, args):
        self.calls.append((name, args))
        return "executed"

    def execute_result(self, name, args):
        # The loop reads the typed dispatch seam (D02); this fake adapts its
        # text exactly the way the real registry adapts a legacy handler.
        from ouroboros.tools.tool_result import LegacyTextResultAdapter

        return LegacyTextResultAdapter.from_text(name, self.execute(name, args))


def test_main_custom_schema_error_continues_without_handler(tmp_path):
    message, receipts = _invalid_custom_exchange()
    tools = _FakeTools()
    tools._ctx._request_wire_custom_receipts = receipts
    messages = [dict(message)]
    logs = tmp_path / "logs"
    logs.mkdir()

    errors = handle_tool_calls(
        message["tool_calls"],
        tools,
        logs,
        "task",
        StatefulToolExecutor(),
        messages,
        {"tool_calls": []},
        lambda _text: None,
    )

    assert errors == 1
    assert tools.calls == []
    assert messages[-1]["role"] == "tool"
    assert "TOOL_ARG_ERROR" in messages[-1]["content"]


def test_main_schema_valid_custom_call_reaches_existing_handler(tmp_path):
    message, receipts = _invalid_custom_exchange(raw_input='{"marker":"ok"}')
    tools = _FakeTools()
    tools._ctx._request_wire_custom_receipts = receipts
    logs = tmp_path / "logs"
    logs.mkdir()

    errors = handle_tool_calls(
        message["tool_calls"],
        tools,
        logs,
        "task",
        StatefulToolExecutor(),
        [dict(message)],
        {"tool_calls": []},
        lambda _text: None,
    )

    assert errors == 0
    assert tools.calls == [("probe", {"marker": "ok"})]


def test_custom_receipt_rejects_canonical_argument_tampering():
    message, receipts = _invalid_custom_exchange()
    message["tool_calls"][0]["function"]["arguments"] = '{"marker":"ok"}'
    usage = _receipt_usage(receipts)
    with pytest.raises(ValueError, match="differs from canonical calls"):
        dispatch.pop_custom_validation_receipts(usage, message["tool_calls"])

    usage = _receipt_usage(receipts)
    usage[dispatch.REQUEST_WIRE_USAGE_KEY]["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from physical candidate"):
        dispatch.pop_custom_validation_receipts(usage, _invalid_custom_exchange()[0]["tool_calls"])


def test_bounded_custom_error_continuation_answers_every_returned_call():
    source = {
        "model": "future-model-without-prefix",
        "messages": [{"role": "user", "content": "Use both."}],
        "reasoning_effort": "medium",
        "tool_choice": "required",
        "tools": _tools(),
    }
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "medium", "requested_wire_form"
        ),
        requested_effort="medium",
        ladder_ordinal=1,
    )
    calls, receipts = normalize_openai_custom_tool_calls([
        _custom_call("call-invalid", "probe", '{"marker":"wrong"}'),
        _custom_call("call-valid", "second", '{"value":2}'),
    ], candidate)

    continuation = dispatch.custom_tool_error_continuation(
        {"role": "assistant", "content": "", "tool_calls": calls},
        receipts,
    )

    assert [item["tool_call_id"] for item in continuation[1:]] == [
        "call-invalid",
        "call-valid",
    ]
    assert all("TOOL_ARG_ERROR" in item["content"] for item in continuation[1:])


def test_function_origin_remains_schema_tolerant(tmp_path):
    tools = _FakeTools()
    logs = tmp_path / "logs"
    logs.mkdir()
    result = _execute_single_tool(
        tools,
        {
            "id": "function-origin",
            "type": "function",
            "function": {"name": "probe", "arguments": '{"marker":"wrong"}'},
        },
        logs,
        "task",
    )
    assert result["is_error"] is False
    assert tools.calls == [("probe", {"marker": "wrong"})]


def test_background_custom_schema_error_never_reaches_registry():
    from ouroboros.consciousness import BackgroundConsciousness

    message, receipts = _invalid_custom_exchange()
    instance = object.__new__(BackgroundConsciousness)
    result = instance._execute_tool(message["tool_calls"][0], [], receipts[0])
    assert "TOOL_ARG_ERROR" in result


def test_background_two_round_custom_error_continuation(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros import consciousness
    from ouroboros.consciousness import BackgroundConsciousness

    read_tool = {
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
    invalid, receipts = _invalid_custom_exchange(
        read_tool,
        '{"path":"wrong"}',
    )
    observed_messages = []

    def fake_chat_observed(_client, **kwargs):
        observed_messages.append(copy.deepcopy(kwargs["messages"]))
        if len(observed_messages) == 1:
            return copy.deepcopy(invalid), {
                **_receipt_usage(receipts),
                "cost": 0.0,
            }
        return {"role": "assistant", "content": "corrected"}, {"cost": 0.0}

    registry_calls = []
    registry = SimpleNamespace(
        _ctx=SimpleNamespace(),
        get_timeout=lambda _name: 1,
        execute=lambda name, args: registry_calls.append((name, args)) or "executed",
    )
    instance = object.__new__(BackgroundConsciousness)
    instance._build_context = lambda: "context"
    instance._tool_schemas = lambda: [read_tool]
    instance._llm = SimpleNamespace(_resolve_remote_target=lambda _model: _target())
    instance._drive_root = tmp_path
    instance._max_bg_rounds = 2
    instance._paused = False
    instance._emit_live_log = lambda *_a, **_k: None
    instance._emit_progress = lambda _content: None
    instance._bg_spent_usd = 0.0
    instance._check_budget = lambda: True
    instance._event_queue = None
    instance._last_idle_reason = ""
    instance._next_wakeup_sec = 300
    instance._wakeup_max = 3600
    instance._owner_chat_id_fn = lambda: None
    instance._registry = registry
    instance._tool_executor = ThreadPoolExecutor(max_workers=1)
    (tmp_path / "logs").mkdir()

    monkeypatch.setattr(
        consciousness,
        "get_consciousness_model",
        lambda: "openai::future-model-without-prefix",
    )
    monkeypatch.setattr(consciousness, "resolve_effort", lambda _slot: "medium")
    monkeypatch.setattr(dispatch, "projected_context_size_bytes", lambda *_a, **_k: 1)
    monkeypatch.setattr(
        "ouroboros.llm_observability.chat_observed",
        fake_chat_observed,
    )
    try:
        assert instance._think_scoped() is True
    finally:
        instance._tool_executor.shutdown(wait=True)

    assert registry_calls == []
    assert len(observed_messages) == 2
    assert observed_messages[1][-1]["role"] == "tool"
    assert "TOOL_ARG_ERROR" in observed_messages[1][-1]["content"]


def test_background_admission_counts_physical_custom_projection(monkeypatch, tmp_path):
    from ouroboros import consciousness
    from ouroboros.consciousness import BackgroundConsciousness

    logs = tmp_path / "logs"
    logs.mkdir()
    instance = object.__new__(BackgroundConsciousness)
    instance._build_context = lambda: "context"
    instance._tool_schemas = _tools
    instance._llm = SimpleNamespace(_resolve_remote_target=lambda _model: _target())
    instance._drive_root = tmp_path
    instance._max_bg_rounds = 1
    instance._paused = False
    instance._last_idle_reason = ""
    observed = {}

    def oversized(messages, tools, **kwargs):
        observed.update(kwargs)
        assert messages[0]["content"] == "context"
        assert tools == _tools()
        return consciousness.BG_CONTEXT_MAX_CHARS + 1

    monkeypatch.setattr(dispatch, "projected_context_size_bytes", oversized)
    monkeypatch.setattr(consciousness, "resolve_effort", lambda _slot: "medium")
    monkeypatch.setattr(
        consciousness,
        "get_consciousness_model",
        lambda: "openai::future-model-without-prefix",
    )

    assert instance._think_scoped() is False
    assert instance._last_idle_reason == "context_overflow"
    assert observed == {"provider": "openai", "reasoning_effort": "medium"}


def test_structured_compaction_returns_one_bounded_tool_error_continuation(monkeypatch, tmp_path):
    message, receipts = _invalid_custom_exchange(
        context_compaction._CONTEXT_SUMMARIES_TOOL,
        '{"summaries":"wrong"}',
    )
    valid_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-valid",
            "type": "function",
            "function": {
                "name": "emit_context_summaries",
                "arguments": '{"summaries":[{"source_id":"source:0:4:x","summary":"ok"}]}',
            },
        }],
    }
    calls = []

    def fake_chat_observed(_client, **kwargs):
        calls.append(copy.deepcopy(kwargs["messages"]))
        if len(calls) == 1:
            return copy.deepcopy(message), _receipt_usage(receipts)
        return valid_message, {}

    import ouroboros.llm_observability as observed

    monkeypatch.setattr(observed, "chat_observed", fake_chat_observed)
    part = context_compaction._part("source", "text")
    expected_id = part.source_id
    valid_message["tool_calls"][0]["function"]["arguments"] = (
        '{"summaries":[{"source_id":"' + expected_id + '","summary":"ok"}]}'
    )
    result = context_compaction._call_summarizer(
        [part],
        drive_root=tmp_path,
        task_id="compaction",
        phase="map",
        spec={
            "model": "openai::future-model-without-prefix",
            "effort": "low",
            "output_budget": 100,
            "use_local": False,
        },
        summary_budgets={"source": 100},
        usage_total={},
    )
    assert result == {expected_id: "ok"}
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "tool"
    assert "TOOL_ARG_ERROR" in calls[1][-1]["content"]


def test_context_fit_counts_larger_direct_custom_projection():
    messages = [{"role": "user", "content": "x"}]
    canonical = estimate_context_prompt_tokens(messages, _tools())
    projected = estimate_context_prompt_tokens(
        messages,
        _tools(),
        provider="openai",
        reasoning_effort="medium",
    )
    assert projected > canonical
    assert estimate_context_prompt_tokens(
        messages,
        _tools(),
        provider="openrouter",
        reasoning_effort="medium",
    ) == canonical
