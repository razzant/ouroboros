from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient, _canonical_candidate_bytes
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", raising=False)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: 0.01)
    capture_token = ua._LAST_PHYSICAL_ATTEMPT.set(None)
    (root / "state").mkdir(parents=True)
    try:
        yield root
    finally:
        ua._LAST_PHYSICAL_ATTEMPT.reset(capture_token)


class _Response:
    def __init__(self, *, text: str = "ok") -> None:
        self._payload = {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 2,
                "cost": 0.0,
            },
        }

    def model_dump(self):
        return copy.deepcopy(self._payload)


def _rows(root: Path):
    return ledger_rows(root)


def _final_rows(root: Path):
    finals = {}
    for row in _rows(root):
        finals[row["attempt_id"]] = row
    return list(finals.values())


def _physical_context(route_fp: str = "route-main") -> ua.PhysicalAttemptContext:
    return ua.PhysicalAttemptContext(
        profile="owner_low",
        rendered_mode="low",
        measurement_basis="fresh_route_usage",
        route_fp=route_fp,
        round_id="round-7",
        target_total_tokens=200_000,
        capacity_total_tokens=500_000,
        context_target_miss=True,
        automatic_pass_used=True,
    )


def _scope(root: Path, task_id: str = "task-physical") -> ua.UsageScope:
    return ua.UsageScope(
        drive_root=root,
        task_id=task_id,
        root_task_id=task_id,
        category="task",
        source="test.physical",
    )


def _target():
    return {
        "provider": "openai",
        "usage_model": "openai/gpt-5.2",
        "resolved_model": "gpt-5.2",
    }


def _manifest(ref):
    path = Path(ref["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
    return json.loads(path.read_text())


def _has_capsule(value) -> bool:
    if isinstance(value, dict):
        return "_context_capsule" in value or any(_has_capsule(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_capsule(item) for item in value)
    return False


def test_remote_candidate_manifest_matches_exact_post_transform_send(data_root):
    from ouroboros.observability import read_blob_ref

    client = LLMClient(api_key="unused")
    sent = {}

    def create(**candidate):
        sent.update(copy.deepcopy(candidate))
        return _Response()

    payload = {
        "model": "gpt-5.2",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "OPENAI_API_KEY=sk-secret-physical-candidate",
                "_context_capsule": {"generation": 2},
            }],
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "inspect",
                "parameters": {
                    "type": "object",
                    "properties": {"_context_capsule": {"type": "string"}},
                },
            },
        }],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    context = _physical_context()
    with ua.usage_scope(_scope(data_root)), ua.bind_physical_attempt_context(context):
        result = client._create_chat_completion_with_retries(create, payload, _target())
    assert isinstance(result, _Response)
    assert not _has_capsule(sent["messages"])
    assert "_context_capsule" in sent["tools"][0]["function"]["parameters"]["properties"]

    raw = _canonical_candidate_bytes(sent)
    context_bytes = _canonical_candidate_bytes({
        key: sent[key] for key in ("system", "messages", "tools", "functions") if key in sent
    })
    rows = _rows(data_root)
    final = rows[-1]
    assert [row["state"] for row in rows] == ["settled"]
    assert all(row["candidate_raw_sha256"] == hashlib.sha256(raw).hexdigest() for row in rows)
    assert all(row["candidate_raw_size_bytes"] == len(raw) for row in rows)
    assert all(row["candidate_context_sha256"] == hashlib.sha256(context_bytes).hexdigest() for row in rows)
    assert all(row["candidate_context_size_bytes"] == len(context_bytes) for row in rows)
    assert all(row["candidate_measurement_kind"] == "canonical_json_v1" for row in rows)
    assert final["physical_context"] == context.__dict__

    manifest = _manifest(final["candidate_manifest_ref"])
    assert manifest["call_id"] == final["attempt_id"]
    assert manifest["candidate_raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["candidate_raw_digest_basis"] == "canonical_json_v1_pre_redaction"
    assert manifest["redacted_projection_digest_basis"] == (
        "observability_json_v1_post_default_redaction_cas"
    )
    assert manifest["full_payload_redacted"] is True
    assert manifest["redaction"]["redacted"] is True
    persisted = read_blob_ref(data_root, manifest["full_payload_ref"])
    assert "sk-secret-physical-candidate" not in json.dumps(persisted)
    assert manifest["full_payload_ref"]["sha256"] != final["candidate_raw_sha256"]

    capture = ua.last_physical_attempt_capture()
    assert capture is not None and capture.state == "settled"
    assert capture.attempt_id == final["attempt_id"]
    assert capture.candidate_manifest_ref == final["candidate_manifest_ref"]
    assert capture.physical_context == context


@pytest.mark.parametrize(
    "cache_write_key",
    ("cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"),
)
def test_openai_detail_cache_write_alias_reaches_physical_settlement(
    data_root, cache_write_key,
):
    client = LLMClient(api_key="unused")
    response = _Response()
    response._payload["usage"]["prompt_tokens_details"] = {cache_write_key: 5}

    with ua.usage_scope(_scope(data_root, f"task-cache-write-{cache_write_key}")):
        physical = client._create_chat_completion_with_retries(
            lambda **_candidate: response,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "cached"}]},
            _target(),
        )

    _message, returned_usage = client._normalize_remote_response(
        physical.model_dump(), _target(), skip_cost_fetch=True,
    )
    final = _rows(data_root)[-1]
    assert returned_usage["cache_write_tokens"] == 5
    assert final["state"] == "settled"
    assert final["cache_write_tokens"] == 5


def test_marker_free_candidate_does_not_invent_applied_cache_ttl(data_root):
    client = LLMClient(api_key="unused")
    with ua.usage_scope(_scope(data_root, "task-marker-free-ttl")):
        client._create_chat_completion_with_retries(
            lambda **_candidate: _Response(),
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "plain"}]},
            _target(),
        )

    final = _rows(data_root)[-1]
    assert final["state"] == "settled"
    assert final["prompt_cache_ttl"] == ""


def test_local_candidate_is_measured_after_existing_local_transform(data_root, monkeypatch):
    import ouroboros.local_model as local_model

    client = LLMClient(api_key="unused")
    sent = {}

    class _Completions:
        @staticmethod
        def create(**candidate):
            sent.update(copy.deepcopy(candidate))
            return _Response(text="local")

    monkeypatch.setattr(
        client,
        "_get_local_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())),
    )
    monkeypatch.setattr(
        local_model,
        "get_manager",

        lambda: SimpleNamespace(serving_context_evidence=lambda: {"context_window": 8192, "confirmed": True}, measure_prepared_input=lambda payload: {"supported": False}),
    )
    monkeypatch.setattr(
        client,
        "_prepare_messages_for_local_context",
        lambda messages, ctx_len, max_tokens: [{"role": "system", "content": "post-local-compactor"}],
    )
    tools = [{
        "type": "function",
        "function": {"name": "local_tool", "parameters": {"type": "object"}},
        "cache_control": {"type": "ephemeral"},
    }]
    with ua.usage_scope(_scope(data_root, "task-local")), ua.bind_physical_attempt_context(
        _physical_context("route-local")
    ):
        message, _usage = client._chat_local(
            [{"role": "user", "content": "pre-transform"}],
            tools,
            max_tokens=65_536,
            tool_choice="auto",
        )
    assert message["content"] == "local"
    assert sent["messages"] == [{"role": "system", "content": "post-local-compactor"}]
    assert sent["tools"] == [{
        "type": "function",
        "function": {"name": "local_tool", "parameters": {"type": "object"}},
    }]
    assert sent["max_tokens"] == 2048
    final = _rows(data_root)[-1]

    raw = _canonical_candidate_bytes(sent)
    assert final["candidate_raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert final["candidate_raw_size_bytes"] == len(raw)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("timeout", [33.0, None, 0.0, -1.0])
@pytest.mark.parametrize("measurement_supported", [False, True], ids=["estimated", "measured"])
def test_local_transport_timeout_is_separate_from_physical_candidate(
    data_root, monkeypatch, asynchronous, timeout, measurement_supported,
):
    from importlib import import_module

    import httpx
    from openai import OpenAI

    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.observability import read_blob_ref

    client = LLMClient(api_key="unused")
    sent, transport_timeouts, measured = [], [], []

    def handle(request):
        sent.append(json.loads(request.content))
        transport_timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json=_Response(text="local").model_dump())

    def measure(payload):
        measured.append(copy.deepcopy(payload))
        return {
            "supported": measurement_supported, "input_is_exact": measurement_supported,
            "input_tokens": 8, "context_window": 8192, "process_id": 123,
            "native_input_sha256": input_fingerprint(payload),
            "output_limit_enforced": True, "reasoning_included_in_limit": True,
        }

    monkeypatch.setattr(import_module("ouroboros.local_model"), "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=lambda: {"context_window": 8192, "confirmed": True, "process_id": 123},
        measure_prepared_input=measure,
    ))
    messages = [{"role": "user", "content": "hello", "_context_capsule": {"generation": 2}}]
    original = copy.deepcopy(messages)
    kwargs = dict(messages=messages, model="local", use_local=True, max_tokens=512, timeout=timeout)
    with OpenAI(
        base_url="http://127.0.0.1:8799/v1", api_key="local", max_retries=0, timeout=91.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as transport, ua.usage_scope(_scope(data_root)), ua.physical_attempt_limit(1):
        monkeypatch.setattr(client, "_get_local_client", lambda: transport)
        message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)

    assert message["content"] == "local" and len(sent) == 1
    # Assert the SDK's actual HTTP timeout, including its unchanged default
    # when the caller supplies no positive override, independently of kwargs.
    expected_timeout = timeout if timeout is not None and timeout > 0 else 91.0
    assert transport_timeouts == [dict.fromkeys(("connect", "read", "write", "pool"), expected_timeout)]
    physical = sent[0]
    assert measured == [physical] and "timeout" not in measured[0]
    assert not _has_capsule(physical) and messages == original
    raw = _canonical_candidate_bytes(physical)
    rows = _rows(data_root)
    assert [row["state"] for row in rows] == ["settled"]
    assert all(row["candidate_raw_sha256"] == hashlib.sha256(raw).hexdigest() for row in rows)
    assert all(row["candidate_raw_size_bytes"] == len(raw) for row in rows)
    assert rows[0]["reservation_upper_bound_usd"] == rows[-1]["cost_usd"] == 0.0
    assert usage["cost"] == 0.0 and usage["cost_final"] is True
    assert usage["ledger_attempt_ids"] == [rows[-1]["attempt_id"]]
    manifest = _manifest(rows[-1]["candidate_manifest_ref"])
    assert read_blob_ref(data_root, manifest["full_payload_ref"]) == physical


def test_local_read_timeout_after_dispatch_is_not_retried(data_root, monkeypatch):
    import httpx
    import ouroboros.local_model as local_model

    client = LLMClient(api_key="unused")
    calls = 0

    class _Completions:
        @staticmethod
        def create(**_candidate):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("local server stopped answering")

    monkeypatch.setattr(
        client,
        "_get_local_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())),
    )
    monkeypatch.setattr(
        local_model,
        "get_manager",

        lambda: SimpleNamespace(serving_context_evidence=lambda: {"context_window": 8192, "confirmed": True}, measure_prepared_input=lambda payload: {"supported": False}),
    )
    with ua.usage_scope(_scope(data_root, "task-local-timeout")), ua.bind_physical_attempt_context(
        _physical_context("route-local-timeout")
    ), pytest.raises(httpx.ReadTimeout):
        client._chat_local(
            [{"role": "user", "content": "do this once"}],
            None,
            max_tokens=1024,
            tool_choice="auto",
        )

    assert calls == 1
    assert _rows(data_root)[-1]["state"] == "unresolved"


def test_internal_retry_gets_distinct_attempt_but_stable_context_identity(data_root, monkeypatch):
    client = LLMClient(api_key="unused")
    sends = []

    class _Rejected(RuntimeError):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "type": "invalid_request_error"}}

    def create(**candidate):
        sends.append(copy.deepcopy(candidate))
        if len(sends) == 1:
            raise _Rejected("temperature unsupported")
        return _Response()

    def retry_without_temperature(candidate, model, exc):
        retry = copy.deepcopy(candidate)
        retry.pop("temperature", None)
        return retry

    monkeypatch.setattr(client, "_retry_without_optional_sampling", retry_without_temperature)
    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", lambda *args: None)
    payload = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "same context"}],
        "temperature": 0.7,
        "max_tokens": 32,
    }
    with ua.usage_scope(_scope(data_root, "task-retry")):
        client._create_chat_completion_with_retries(create, payload, _target())

    finals = _final_rows(data_root)
    assert len(sends) == len(finals) == 2
    assert finals[0]["attempt_id"] != finals[1]["attempt_id"]
    assert finals[0]["state"] == "unresolved"
    assert finals[1]["state"] == "settled"
    assert finals[0]["candidate_raw_sha256"] != finals[1]["candidate_raw_sha256"]
    assert finals[0]["candidate_context_sha256"] == finals[1]["candidate_context_sha256"]
    assert finals[0]["candidate_manifest_ref"] != finals[1]["candidate_manifest_ref"]


def test_provider_exception_carries_structured_physical_attempt_facts(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    class _Overflow(RuntimeError):
        status_code = 400
        body = {"error": {
            "code": "context_length_exceeded",
            "type": "invalid_request_error",
            "message": "input too long; OPENAI_API_KEY=sk-provider-secret",
        }}

    error = _Overflow("provider rejected request")

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        raise error

    target = {
        **_target(),
        "provider": "openrouter",
        "supports_openrouter_extensions": True,
    }
    with ua.usage_scope(_scope(data_root, "task-overflow")), pytest.raises(_Overflow) as caught:
        client._create_chat_completion_with_retries(
            create,
            {
                "model": "gpt-5.2",
                "messages": [{
                    "role": "assistant",
                    "content": "large",
                    "reasoning_details": [{"type": "reasoning", "text": "signed"}],
                }],
                "max_tokens": 123,
            },
            target,
        )
    assert provider_calls == 1
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None
    assert capture.state == "unresolved"
    assert capture.provider_status_code == 400
    assert capture.provider_code == "context_length_exceeded"
    assert capture.provider_error_type == "invalid_request_error"
    assert capture.max_completion_tokens == 123
    assert "sk-provider-secret" not in capture.provider_error
    assert capture.candidate_measurement_kind == "canonical_json_v1"
    assert capture.candidate_manifest_ref == _rows(data_root)[-1]["candidate_manifest_ref"]


def test_async_structured_overflow_bypasses_transport_recovery(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    class _Overflow(RuntimeError):
        status_code = 400
        body = {"error": {"code": "context_window_exceeded", "type": "invalid_request_error"}}

    async def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        raise _Overflow("provider rejected request")

    async def run():
        return await client._create_chat_completion_with_retries_async(
            create,
            {
                "model": "gpt-5.2",
                "messages": [{
                    "role": "assistant", "content": "large",
                    "reasoning_details": [{"type": "reasoning", "text": "signed"}],
                }],
                "max_tokens": 456,
            },
            {**_target(), "provider": "openrouter", "supports_openrouter_extensions": True},
        )

    with ua.usage_scope(_scope(data_root, "task-overflow-async")), pytest.raises(_Overflow) as caught:
        asyncio.run(run())
    assert provider_calls == 1
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None
    assert capture.provider_code == "context_window_exceeded"
    assert capture.max_completion_tokens == 456


def test_http_200_structured_overflow_body_is_not_retried(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    class _BodyOverflow(_Response):
        def __init__(self):
            self._payload = {"error": {
                "code": 400,
                "type": "context_length_exceeded",
                "message": "temperature must be between 0 and 2",
            }}

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        return _BodyOverflow()

    with ua.usage_scope(_scope(data_root, "task-overflow-body")):
        response = client._create_chat_completion_with_retries(
            create,
            {
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": "large"}],
                "temperature": 3.0,
            },
            {**_target(), "provider": "openrouter", "supports_openrouter_extensions": True},
        )
    assert provider_calls == 1
    assert response.model_dump()["error"]["type"] == "context_length_exceeded"


def test_precondition_releases_before_dispatch_and_does_not_claim_send(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        return _Response()

    with (
        ua.usage_scope(_scope(data_root, "task-precondition")),
        ua.bind_physical_attempt_context(_physical_context(), lambda request: False),
        ua.physical_attempt_limit(0),
        pytest.raises(ua.PhysicalAttemptPreconditionFailed) as caught,
    ):
        client._create_chat_completion_with_retries(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        )
    assert provider_calls == 0
    assert [row["state"] for row in _rows(data_root)] == ["released"]
    assert _rows(data_root)[-1]["candidate_manifest_ref"]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.state == "released"
    assert capture.candidate_manifest_ref == _rows(data_root)[-1]["candidate_manifest_ref"]
    assert ua.usage_breakdown(data_root)["physical_calls"] == 0


def test_attempt_limit_keeps_persisted_manifest_on_release_and_capture(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        return _Response()

    with (
        ua.usage_scope(_scope(data_root, "task-limit-manifest")),
        ua.physical_attempt_limit(0),
        pytest.raises(ua.PhysicalAttemptLimitExceeded) as caught,
    ):
        client._create_chat_completion_with_retries(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        )

    assert provider_calls == 0
    assert [row["state"] for row in _rows(data_root)] == ["released"]
    final = _rows(data_root)[-1]
    assert final["candidate_manifest_ref"]
    assert _manifest(final["candidate_manifest_ref"])["call_id"] == final["attempt_id"]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.state == "released"
    assert capture.candidate_manifest_ref == final["candidate_manifest_ref"]


def test_raising_host_predicate_keeps_persisted_manifest(data_root):
    client = LLMClient(api_key="unused")
    provider_calls = 0

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        return _Response()

    def predicate(_request):
        raise RuntimeError("host predicate failed")

    with (
        ua.usage_scope(_scope(data_root, "task-predicate-manifest")),
        ua.bind_physical_attempt_context(_physical_context(), predicate),
        pytest.raises(ua.PhysicalAttemptPreparationFailed) as caught,
    ):
        client._create_chat_completion_with_retries(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        )

    assert provider_calls == 0
    final = _rows(data_root)[-1]
    assert final["state"] == "released"
    assert final["candidate_manifest_ref"]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.state == "released"
    assert capture.candidate_manifest_ref == final["candidate_manifest_ref"]


def test_async_attempt_limit_keeps_persisted_manifest(data_root):
    client = LLMClient(api_key="unused")

    async def create(**candidate):
        return _Response()

    with (
        ua.usage_scope(_scope(data_root, "task-async-limit-manifest")),
        ua.physical_attempt_limit(0),
        pytest.raises(ua.PhysicalAttemptLimitExceeded) as caught,
    ):
        asyncio.run(client._create_chat_completion_with_retries_async(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        ))

    final = _rows(data_root)[-1]
    assert final["state"] == "released"
    assert final["candidate_manifest_ref"]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.candidate_manifest_ref == final["candidate_manifest_ref"]


def test_async_raising_predicate_keeps_persisted_manifest(data_root):
    client = LLMClient(api_key="unused")

    async def create(**candidate):
        return _Response()

    def predicate(_request):
        raise RuntimeError("async host predicate failed")

    with (
        ua.usage_scope(_scope(data_root, "task-async-predicate-manifest")),
        ua.bind_physical_attempt_context(_physical_context(), predicate),
        pytest.raises(ua.PhysicalAttemptPreparationFailed) as caught,
    ):
        asyncio.run(client._create_chat_completion_with_retries_async(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        ))

    final = _rows(data_root)[-1]
    assert final["state"] == "released"
    assert final["candidate_manifest_ref"]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.candidate_manifest_ref == final["candidate_manifest_ref"]


def test_candidate_persistence_failure_releases_without_send_or_limit_claim(
    data_root, monkeypatch,
):
    from ouroboros import observability

    client = LLMClient(api_key="unused")
    provider_calls = 0

    def create(**candidate):
        nonlocal provider_calls
        provider_calls += 1
        return _Response()

    monkeypatch.setattr(
        observability,
        "persist_physical_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("candidate store unavailable")),
    )
    with (
        ua.usage_scope(_scope(data_root, "task-persist-fail")),
        ua.bind_physical_attempt_context(_physical_context()),
        ua.physical_attempt_limit(0),
        pytest.raises(ua.PhysicalAttemptPreparationFailed) as caught,
    ):
        client._create_chat_completion_with_retries(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "same"}]},
            _target(),
        )
    assert provider_calls == 0
    assert [row["state"] for row in _rows(data_root)] == ["released"]
    assert "candidate_manifest_ref" not in _rows(data_root)[-1]
    capture = ua.physical_attempt_capture_from_exception(caught.value)
    assert capture is not None and capture.candidate_manifest_ref is None
    assert ua.usage_breakdown(data_root)["physical_calls"] == 0


def test_direct_anthropic_settlement_normalizes_cache_and_request_ttl(
    data_root, monkeypatch,
):
    import requests
    from ouroboros import config

    client = LLMClient(api_key="unused")
    monkeypatch.setattr(config, "resolve_prompt_cache_ttl", lambda: "1h")

    class _AnthropicResponse:
        status_code = 200
        reason = "OK"
        url = "https://anthropic.invalid/v1/messages"
        text = ""

        @staticmethod
        def json():
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 15_000,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 20_000,
                    "cache_creation_input_tokens": 10_000,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 4_000,
                        "ephemeral_1h_input_tokens": 6_000,
                    },
                },
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _AnthropicResponse())
    target = {
        "provider": "anthropic",
        "usage_model": "anthropic/claude-fable-5",
        "resolved_model": "claude-fable-5",
        "base_url": "https://anthropic.invalid/v1",
        "api_key": "unused",
    }
    tools = [{
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "inspect",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    with ua.usage_scope(_scope(data_root, "task-anthropic-native")):
        _message, usage = client._chat_anthropic(
            target,
            [{"role": "user", "content": "cached prompt"}],
            tools,
            "none",
            64,
            "auto",
        )

    assert usage["prompt_tokens"] == 45_000
    assert usage["cached_tokens"] == 20_000
    assert usage["cache_write_tokens"] == 10_000
    final = _rows(data_root)[-1]
    assert final["state"] == "settled"
    assert final["prompt_tokens"] == 45_000
    assert final["cached_tokens"] == 20_000
    assert final["cache_write_tokens"] == 10_000
    assert final["prompt_cache_ttl"] == "1h"


def test_direct_gigachat_cache_semantics_abstain_from_density(data_root, monkeypatch):
    from ouroboros.capability_evidence import get_token_density

    client = LLMClient(api_key="unused")

    class _GigaCompletion:
        choices = [SimpleNamespace(message=SimpleNamespace(content="ok", function_call=None))]
        usage = SimpleNamespace(
            prompt_tokens=15_000,
            completion_tokens=1,
            precached_prompt_tokens=30_000,
        )

        def model_dump(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 15_000,
                    "completion_tokens": 1,
                    "precached_prompt_tokens": 30_000,
                },
            }

    monkeypatch.setattr(
        client,
        "_get_gigachat_client",
        lambda *args, **kwargs: SimpleNamespace(chat=lambda _candidate: _GigaCompletion()),
    )
    target = {
        "provider": "gigachat",
        "usage_model": "gigachat/giga-test",
        "resolved_model": "giga-test",
    }

    with ua.usage_scope(_scope(data_root, "task-gigachat-native")):
        _message, usage = client._chat_gigachat(
            target,
            [{"role": "user", "content": "cached prompt"}],
            None,
            "none",
            64,
            "auto",
        )

    assert usage["cached_tokens"] == 30_000
    final = _rows(data_root)[-1]
    assert final["state"] == "settled"
    assert final["cached_tokens"] == 30_000
    assert get_token_density(data_root, "gigachat/giga-test") == 0.0


def test_async_candidate_uses_same_manifest_and_capture_seam(data_root):
    client = LLMClient(api_key="unused")
    sent = {}

    async def create(**candidate):
        sent.update(copy.deepcopy(candidate))
        return _Response(text="async")

    async def run_and_capture():
        result = await client._create_chat_completion_with_retries_async(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "async"}]},
            _target(),
        )
        return result, ua.last_physical_attempt_capture()

    with ua.usage_scope(_scope(data_root, "task-async")), ua.bind_physical_attempt_context(
        _physical_context("route-async")
    ):
        result, capture = asyncio.run(run_and_capture())
    assert isinstance(result, _Response)
    final = _rows(data_root)[-1]
    assert final["candidate_raw_sha256"] == hashlib.sha256(
        _canonical_candidate_bytes(sent)
    ).hexdigest()
    assert capture is not None and capture.attempt_id == final["attempt_id"]
    assert capture.state == "settled"


def test_capture_stays_dispatched_when_terminal_accounting_cannot_be_written(
    data_root, monkeypatch,
):
    response = _Response()
    monkeypatch.setattr(ua, "settle_attempt", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("settlement unavailable")
    ))
    monkeypatch.setattr(ua, "mark_unresolved", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("terminal transition unavailable")
    ))
    assert ua.execute_physical_attempt(
        ua.AttemptRequest(
            model="opaque/sdk", provider="opaque-sdk", reservation_usd=0.0,
            drive_root=data_root, task_id="task-open-attempt",
        ),
        lambda: response,
    ) is response
    assert _rows(data_root)[-1]["state"] == "dispatched"
    capture = ua.last_physical_attempt_capture()
    assert capture is not None and capture.state == "dispatched"


def test_opaque_attempts_remain_honest_and_candidate_validation_is_strict(data_root):
    request = ua.AttemptRequest(
        model="opaque/sdk",
        provider="opaque-sdk",
        reservation_usd=0.0,
        drive_root=data_root,
        task_id="task-opaque",
    )
    ua.execute_physical_attempt(request, lambda: _Response())
    rows = _rows(data_root)
    assert all(row["candidate_measurement_kind"] == "opaque" for row in rows)
    for row in rows:
        assert row["candidate_raw_sha256"] is None
        assert row["candidate_context_sha256"] is None
        assert "candidate_manifest_ref" not in row
    capture = ua.last_physical_attempt_capture()
    assert capture is not None and capture.candidate_measurement_kind == "opaque"

    with pytest.raises(ua.UsageLedgerCorrupt, match="invalid candidate_raw_sha256"):
        ua.reserve_attempt(replace(request, candidate_measurement_kind="canonical_json_v1"))
    with pytest.raises(ua.UsageLedgerCorrupt, match="opaque candidate claims identity"):
        ua.reserve_attempt(replace(request, candidate_raw_sha256="a" * 64))
