"""One-shot Settings provider-readiness probe."""

from __future__ import annotations

import asyncio
import copy
import json
import types

import httpx
import pytest

import ouroboros.gateway.models as provider_api
from ouroboros.llm import LLMClient
from ouroboros.usage_accounting import current_usage_scope


class _Request:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _Completion:
    def __init__(self, payload=None):
        self.payload = payload or {
            "choices": [{"message": {"role": "assistant", "content": "not literally OK"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.001},
        }

    def model_dump(self):
        return copy.deepcopy(self.payload)


class _Remote:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.timeouts = []
        self.closed = False
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create),
        )

    def with_options(self, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return self

    def create(self, **payload):
        self.calls.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def _post(payload):
    response = asyncio.run(provider_api.api_provider_test(_Request(payload)))
    return response.status_code, json.loads(response.body.decode("utf-8"))


def _bypass_accounting(monkeypatch):
    # The v7 split moved the candidate executor out of llm.py: `_execute_candidate`
    # lives in llm_attempt.py and reads `execute_physical_attempt` from THAT module,
    # so patching the name llm.py merely re-exports would be a dead patch that never
    # intercepts anything. Same seam, named at its owner.
    import ouroboros.llm_attempt as llm_attempt

    observed = []

    def execute(request, send):
        observed.append((request, current_usage_scope()))
        return send()

    monkeypatch.setattr(llm_attempt, "execute_physical_attempt", execute)
    return observed


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "JSON object"),
        ({}, "provider_id is required"),
        ({"provider_id": "openrouterr"}, "unknown provider id"),
        ({"provider_id": "openrouter", "overrides": []}, "overrides must be a JSON object"),
        ({"provider_id": "openrouter", "overrides": {"NOT_ALLOWED": "x"}}, "unsupported override"),
        ({"provider_id": "openrouter", "overrides": {"OPENROUTER_API_KEY": 7}}, "must be strings"),
    ],
)
def test_provider_test_rejects_contract_misuse_before_thread(monkeypatch, payload, error):
    monkeypatch.setattr(provider_api, "load_settings", lambda: {})
    monkeypatch.setattr(
        provider_api.asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    status, body = _post(payload)
    assert status == 400
    assert error in body["error"]
    assert "x" not in body["error"] or "override" in body["error"]


def test_provider_test_rejects_unparseable_json():
    class Broken:
        async def json(self):
            raise ValueError("bad")

    response = asyncio.run(provider_api.api_provider_test(Broken()))
    assert response.status_code == 400


def test_provider_test_runs_sync_work_behind_one_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *args):
        calls.append((fn, args))
        return {"ok": True}

    monkeypatch.setattr(
        provider_api,
        "load_settings",
        lambda: (_ for _ in ()).throw(AssertionError("event loop must not load settings")),
    )
    monkeypatch.setattr(provider_api.asyncio, "to_thread", fake_to_thread)
    status, body = _post({"provider_id": "openrouter"})
    assert status == 200 and body == {"ok": True}
    assert calls == [(provider_api._run_provider_test, ("openrouter", {}))]


def test_minimax_region_validation_runs_in_worker(monkeypatch):
    monkeypatch.setattr(provider_api, "load_settings", lambda: {})
    monkeypatch.setattr(
        provider_api,
        "_run_provider_test_with_settings",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    status, body = _post({
        "provider_id": "minimax",
        "overrides": {"MINIMAX_API_KEY": "x", "MINIMAX_REGION": "mars"},
    })
    assert status == 400
    assert body == {"error": "unknown MiniMax region"}


def test_saved_key_and_draft_url_reach_request_local_target(monkeypatch):
    saved = {
        "OPENAI_COMPATIBLE_API_KEY": "saved-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1",
        "OUROBOROS_MODEL": "openai-compatible::draft-model",
    }
    original = copy.deepcopy(saved)
    captured = {}

    def fake_probe(self, model, *, settings, timeout=20.0):
        captured.update(self._resolve_remote_target(model, settings=settings))
        return {"ok": True, "status_code": 200, "exception_type": ""}

    monkeypatch.setattr(provider_api, "load_settings", lambda: copy.deepcopy(saved))
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", fake_probe)
    request = {
        "provider_id": "openai-compatible",
        "overrides": {"OPENAI_COMPATIBLE_BASE_URL": "https://draft.example/v1"},
    }
    request_before = copy.deepcopy(request)
    status, body = _post(request)
    assert status == 200 and body == {"ok": True}
    assert captured["api_key"] == "saved-key"
    assert captured["base_url"] == "https://draft.example/v1"
    assert saved == original
    assert request == request_before


@pytest.mark.parametrize(
    "overrides,expected_key,expected_base_url",
    [
        (
            {"OPENAI_COMPATIBLE_BASE_URL": "https://draft.example/v1"},
            "legacy-saved-key",
            "https://draft.example/v1",
        ),
        (
            {"OPENAI_COMPATIBLE_API_KEY": "draft-key"},
            "draft-key",
            "https://legacy.example/v1",
        ),
    ],
)
def test_legacy_compatible_pair_survives_single_field_draft(
    monkeypatch, overrides, expected_key, expected_base_url,
):
    saved = {
        "OPENAI_API_KEY": "legacy-saved-key",
        "OPENAI_BASE_URL": "https://legacy.example/v1",
        "OPENAI_COMPATIBLE_API_KEY": "",
        "OPENAI_COMPATIBLE_BASE_URL": "",
        "OUROBOROS_MODEL": "openai-compatible::draft-model",
    }
    captured = {}

    def fake_probe(self, model, *, settings, timeout=20.0):
        captured["settings"] = copy.deepcopy(settings)
        captured.update(self._resolve_remote_target(model, settings=settings))
        return {"ok": True, "status_code": 200, "exception_type": ""}

    monkeypatch.setattr(provider_api, "load_settings", lambda: copy.deepcopy(saved))
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", fake_probe)
    status, body = _post({"provider_id": "openai-compatible", "overrides": overrides})

    assert status == 200 and body == {"ok": True}
    assert captured["api_key"] == expected_key
    assert captured["base_url"] == expected_base_url
    assert captured["settings"]["OPENAI_COMPATIBLE_API_KEY"] == expected_key
    assert captured["settings"]["OPENAI_COMPATIBLE_BASE_URL"] == expected_base_url


@pytest.mark.parametrize("legacy_key", ["", "stale-legacy-key"])
def test_compatible_discovery_and_probe_share_effective_mixed_pair(
    monkeypatch, legacy_key,
):
    saved = {
        "OPENAI_COMPATIBLE_API_KEY": "dedicated-key",
        "OPENAI_COMPATIBLE_BASE_URL": "",
        "OPENAI_API_KEY": legacy_key,
        "OPENAI_BASE_URL": "https://legacy.example/v1",
    }
    calls = []

    async def fake_catalog(_client, _provider_id, _label, api_key, base_url):
        calls.append(("catalog", api_key, base_url))
        return [{"value": "openai-compatible::catalog-model"}]

    def fake_probe(self, model, *, settings, timeout=20.0):
        target = self._resolve_remote_target(model, settings=settings)
        calls.append(("probe", target["api_key"], target["base_url"]))
        return {"ok": True, "status_code": 200, "exception_type": ""}

    monkeypatch.setattr(provider_api, "load_settings", lambda: copy.deepcopy(saved))
    monkeypatch.setattr(provider_api, "_fetch_openai_compatible_model_catalog", fake_catalog)
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", fake_probe)

    status, body = _post({"provider_id": "openai-compatible"})

    assert status == 200 and body == {"ok": True}
    expected = ("dedicated-key", "https://legacy.example/v1")
    assert calls == [("catalog", *expected), ("probe", *expected)]


def test_explicit_empty_draft_key_is_not_rehydrated(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    monkeypatch.setattr(provider_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "saved-key",
        "OUROBOROS_MODEL": "openai/test-model",
    })
    status, body = _post({
        "provider_id": "openrouter",
        "overrides": {"OPENROUTER_API_KEY": ""},
    })
    assert status == 200
    assert body == {"ok": False, "error": "Provider is not configured"}


def test_cleared_compatible_draft_does_not_restore_legacy_pair(monkeypatch):
    monkeypatch.setattr(provider_api, "load_settings", lambda: {
        "OPENAI_API_KEY": "saved-legacy-key",
        "OPENAI_BASE_URL": "https://legacy.example/v1",
        "OUROBOROS_MODEL": "openai-compatible::draft-model",
    })
    monkeypatch.setattr(
        LLMClient,
        "_new_remote_client",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    status, body = _post({
        "provider_id": "openai-compatible",
        "overrides": {
            "OPENAI_COMPATIBLE_API_KEY": "",
            "OPENAI_COMPATIBLE_BASE_URL": "",
        },
    })
    assert status == 200
    assert body == {"ok": False, "error": "Provider is not configured"}


@pytest.mark.parametrize(
    "model,settings,expected",
    [
        ("openai/test", {"OPENROUTER_API_KEY": ""}, {
            "provider": "openrouter", "api_key": "", "base_url": "https://openrouter.ai/api/v1",
        }),
        ("openai::gpt-5.6-terra", {"OPENAI_API_KEY": "oa"}, {
            "provider": "openai", "api_key": "oa", "base_url": "https://api.openai.com/v1",
        }),
        ("anthropic::claude-opus-5", {"ANTHROPIC_API_KEY": "ant"}, {
            "provider": "anthropic", "api_key": "ant", "base_url": "https://api.anthropic.com/v1",
        }),
        ("minimax::MiniMax-M3", {"MINIMAX_API_KEY": "mm", "MINIMAX_REGION": "cn_zh"}, {
            "provider": "minimax", "api_key": "mm", "base_url": "https://api.minimaxi.com/v1",
        }),
        ("cloudru::model", {
            "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloud",
            "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://cloud.example/v1",
        }, {"provider": "cloudru", "api_key": "cloud", "base_url": "https://cloud.example/v1"}),
        ("gigachat::GigaChat-2-Max", {
            "GIGACHAT_CREDENTIALS": "giga", "GIGACHAT_USER": "u", "GIGACHAT_PASSWORD": "p",
            "GIGACHAT_SCOPE": "GIGACHAT_API_CORP", "GIGACHAT_BASE_URL": "https://giga.example/v1",
            "GIGACHAT_VERIFY_SSL_CERTS": "false",
        }, {
            "provider": "gigachat", "api_key": "giga", "user": "u", "password": "p",
            "scope": "GIGACHAT_API_CORP", "base_url": "https://giga.example/v1",
            "verify_ssl_certs": False,
        }),
        ("openai-compatible::model", {
            "OPENAI_COMPATIBLE_API_KEY": "compat",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
            "OPENAI_API_KEY": "", "OPENAI_BASE_URL": "",
        }, {
            "provider": "openai-compatible", "api_key": "compat",
            "base_url": "https://compat.example/v1",
        }),
        ("openai-compatible::model", {
            "OPENAI_COMPATIBLE_API_KEY": "",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
            "OPENAI_API_KEY": "legacy-must-not-win",
            "OPENAI_BASE_URL": "https://legacy.example/v1",
        }, {
            "provider": "openai-compatible", "api_key": "",
            "base_url": "https://compat.example/v1",
        }),
    ],
)
def test_explicit_settings_mapping_is_authoritative(monkeypatch, model, settings, expected):
    for key in provider_api._PROVIDER_TEST_OVERRIDE_KEYS:
        monkeypatch.setenv(key, "environment-value")
    original = copy.deepcopy(settings)
    client = LLMClient(api_key="constructor-key", base_url="https://constructor.example/v1")
    target = client._resolve_remote_target(model, settings=settings)
    for key, value in expected.items():
        assert target[key] == value
    assert settings == original
    assert client._remote_clients == {}


@pytest.mark.parametrize(
    "provider,settings,expected",
    [
        ("openai", {"OUROBOROS_MODEL": "openai::main"}, "openai::main"),
        ("anthropic", {
            "OUROBOROS_MODEL": "openai::main",
            "OUROBOROS_MODEL_HEAVY": "anthropic::heavy",
            "OUROBOROS_MODEL_LIGHT": "anthropic::light",
            "CLAUDE_CODE_MODEL": "claude-name-only",
        }, "anthropic::light"),
        ("cloudru", {}, "cloudru::zai-org/GLM-4.7"),
        ("openrouter", {"CLAUDE_AGENT_SDK_MODEL": "opus"}, "google/gemini-3.8-flash"),
        ("openai-compatible", {
            "OUROBOROS_MODEL_FALLBACKS": "openai::other,openai-compatible::chosen",
        }, "openai-compatible::chosen"),
        ("openai-compatible", {}, ""),
    ],
)
def test_provider_test_model_selection(provider, settings, expected):
    assert provider_api._provider_test_model(provider, settings) == expected


def test_catalog_is_used_only_for_compatible_discovery(monkeypatch):
    discoveries = []
    probes = []

    def discover(provider, _settings):
        discoveries.append(provider)
        return "openai-compatible::catalog-first"

    def probe(self, model, *, settings, timeout=20.0):
        probes.append(model)
        return {"ok": True, "status_code": 200, "exception_type": ""}

    monkeypatch.setattr(provider_api, "_discover_provider_test_model", discover)
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", probe)
    assert provider_api._run_provider_test_with_settings(
        "openrouter", {"OPENROUTER_API_KEY": "key"},
    ) == {"ok": True}
    assert discoveries == []
    assert probes == ["google/gemini-3.8-flash"]
    assert provider_api._run_provider_test_with_settings("openai-compatible", {
        "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
    }) == {"ok": True}
    assert discoveries == ["openai-compatible"]
    assert probes[-1] == "openai-compatible::catalog-first"


@pytest.mark.parametrize(
    "discovery,error",
    [
        (lambda *_args: "", "Provider is not configured"),
        (lambda *_args: (_ for _ in ()).throw(httpx.ConnectError("secret raw body")),
         "Could not reach provider"),
    ],
)
def test_discovery_failure_stops_before_generation(monkeypatch, discovery, error):
    monkeypatch.setattr(provider_api, "_discover_provider_test_model", discovery)
    monkeypatch.setattr(
        LLMClient,
        "probe_provider_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
    )
    assert provider_api._run_provider_test_with_settings(
        "openai-compatible", {},
    ) == {"ok": False, "error": error}


@pytest.mark.parametrize(
    "model,settings,token_key",
    [
        ("openai/test", {"OPENROUTER_API_KEY": "key"}, "max_tokens"),
        ("openai::gpt-5.6-terra", {"OPENAI_API_KEY": "key"}, "max_completion_tokens"),
        ("openai-compatible::test", {
            "OPENAI_COMPATIBLE_API_KEY": "key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
            "OPENAI_API_KEY": "", "OPENAI_BASE_URL": "",
        }, "max_tokens"),
    ],
)
def test_openai_class_probe_is_one_minimal_attempt(monkeypatch, model, settings, token_key):
    observed = _bypass_accounting(monkeypatch)
    remote = _Remote([_Completion({
        "choices": [{"message": {"role": "assistant", "content": None, "reasoning": "done"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 0},
    })])
    client = LLMClient()
    monkeypatch.setattr(client, "_new_remote_client", lambda _target: remote)
    result = client.probe_provider_readiness(model, settings=settings)
    assert result["ok"] is True
    assert len(remote.calls) == len(observed) == 1
    payload = remote.calls[0]
    assert payload[token_key] == 16
    assert set(payload) <= {"model", "messages", token_key, "extra_body"}
    assert payload["messages"] == [{"role": "user", "content": "Reply OK"}]
    for forbidden in ("tools", "reasoning", "reasoning_effort", "temperature", "response_format", "web_search"):
        assert forbidden not in payload
    if model == "openai/test":
        assert payload["extra_body"] == {"provider": {"allow_fallbacks": False}}
    request, scope = observed[0]
    assert request.max_completion_tokens == 16
    assert request.source == "provider_test"
    assert (scope.task_id, scope.root_task_id, scope.category, scope.source) == (
        "system:provider_test", "system:provider_test", "provider_test", "provider_test",
    )
    assert client._remote_clients == {}
    assert remote.closed is True


def test_anthropic_probe_accepts_empty_completion(monkeypatch):
    observed = _bypass_accounting(monkeypatch)
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [], "stop_reason": "max_tokens", "usage": {"input_tokens": 2}}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    result = LLMClient().probe_provider_readiness(
        "anthropic::claude-opus-5", settings={"ANTHROPIC_API_KEY": "key"},
    )
    assert result["ok"] is True
    assert len(calls) == len(observed) == 1
    url, kwargs = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert kwargs["json"] == {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
    }


def test_gigachat_probe_uses_one_ephemeral_client(monkeypatch):
    import sys

    observed = _bypass_accounting(monkeypatch)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "stale-environment-credentials")
    monkeypatch.setenv("GIGACHAT_ACCESS_TOKEN", "stale-environment-token")
    constructed = {}
    completion = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="anything"))],
        usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )
    class FakeGigaChat:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.calls = []
            self.closed = False
            constructed["client"] = self

        def chat(self, payload):
            self.calls.append(payload)
            return completion

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "gigachat", types.SimpleNamespace(GigaChat=FakeGigaChat))
    client = LLMClient()
    result = client.probe_provider_readiness(
        "gigachat::GigaChat-2-Max",
        settings={
            "GIGACHAT_CREDENTIALS": "",
            "GIGACHAT_USER": "draft-user",
            "GIGACHAT_PASSWORD": "draft-password",
        },
    )
    remote = constructed["client"]
    assert result["ok"] is True
    assert len(remote.calls) == len(observed) == 1
    assert remote.calls[0]["max_tokens"] == 16
    assert constructed["credentials"] == ""
    assert constructed["user"] == "draft-user"
    assert constructed["password"] == "draft-password"
    assert constructed["access_token"] == ""
    assert constructed["max_retries"] == 0
    assert constructed["timeout"] == 20.0
    assert client._gigachat_clients == {}
    assert remote.closed is True


def test_failure_and_malformed_success_never_retry_or_learn(monkeypatch):
    observed = _bypass_accounting(monkeypatch)
    before_rejected = copy.deepcopy(LLMClient._REJECTED_PARAMS_CACHE)
    before_ceilings = copy.deepcopy(LLMClient._EFFORT_CEILING_CACHE)

    class Unauthorized(RuntimeError):
        status_code = 401

    for outcome, expected in (
        (Unauthorized("raw secret key=unit-test-secret"), "Invalid key"),
        (_Completion({"usage": {}}), "Model request failed"),
    ):
        remote = _Remote([outcome])
        client = LLMClient()
        monkeypatch.setattr(client, "_new_remote_client", lambda _target, remote=remote: remote)
        result = client.probe_provider_readiness(
            "openai/test", settings={"OPENROUTER_API_KEY": "key"},
        )
        assert result == {
            "ok": False,
            "error": expected,
            "status_code": 401 if isinstance(outcome, Unauthorized) else None,
            "exception_type": "Unauthorized" if isinstance(outcome, Unauthorized) else "ProbeEnvelopeError",
        }
        assert len(remote.calls) == 1
    assert len(observed) == 2
    assert LLMClient._REJECTED_PARAMS_CACHE == before_rejected
    assert LLMClient._EFFORT_CEILING_CACHE == before_ceilings


def test_typed_payment_required_maps_to_no_credits():
    from ouroboros.llm_probe import controlled_probe_error

    error = RuntimeError("raw provider billing body")
    error.status_code = 402
    assert controlled_probe_error(error)["error"] == "No credits"


def test_ephemeral_openai_client_disables_sdk_retries(monkeypatch):
    # Hermetic against proxied runners: with any proxy httpx would honor the
    # keepalive helper deliberately returns None (SDK default construction),
    # which would drop the http_client kwarg this test asserts on.
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})
    captured = {}
    http_clients = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeDefaultHttpxClient:
        def __init__(self, **kwargs):
            http_clients.append(kwargs)

    import sys

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(
            OpenAI=FakeOpenAI, DefaultHttpxClient=FakeDefaultHttpxClient,
        ),
    )
    LLMClient._new_remote_client({
        "api_key": "key", "base_url": "https://example.test/v1", "default_headers": {"X": "Y"},
    })
    http_client = captured.pop("http_client")
    assert isinstance(http_client, FakeDefaultHttpxClient)
    assert len(http_clients) == 1 and http_clients[0].get("transport") is not None
    assert captured == {
        "api_key": "key", "base_url": "https://example.test/v1",
        "default_headers": {"X": "Y"}, "max_retries": 0,
    }


def test_provider_test_attempts_are_physically_accounted_without_chat_side_effects(
    monkeypatch, tmp_path,
):
    import ouroboros.usage_accounting as accounting

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(accounting, "_reservation_cost", lambda _request: 0.01)

    class RateLimited(RuntimeError):
        status_code = 429

    remote = _Remote([_Completion(), RateLimited("raw provider body")])
    client = LLMClient()
    monkeypatch.setattr(client, "_new_remote_client", lambda _target: remote)
    settings = {"OPENROUTER_API_KEY": "key"}
    assert client.probe_provider_readiness("openai/test", settings=settings)["ok"] is True
    assert client.probe_provider_readiness("openai/test", settings=settings)["error"] == "Rate limited"

    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()
    ]
    by_attempt = {}
    for row in rows:
        if row.get("kind") == "attempt":
            by_attempt.setdefault(row["attempt_id"], []).append(row)
    assert len(by_attempt) == 2
    finals = [attempt_rows[-1] for attempt_rows in by_attempt.values()]
    assert sorted(row["state"] for row in finals) == ["settled", "unresolved"]
    for attempt_rows in by_attempt.values():
        assert [row["state"] for row in attempt_rows[:2]] == ["reserved", "dispatched"]
        assert attempt_rows[-1]["task_id"] == "system:provider_test"
        assert attempt_rows[-1]["root_task_id"] == "system:provider_test"
        assert attempt_rows[-1]["category"] == "provider_test"
        assert attempt_rows[-1]["source"] == "provider_test"
    events_path = tmp_path / "logs" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert not any(row.get("type") in {"llm_round", "llm_usage", "chat", "progress"} for row in events)
    assert not (tmp_path / "task_results").exists()
    assert not (tmp_path / "logs" / "chat.jsonl").exists()
    assert not (tmp_path / "logs" / "progress.jsonl").exists()


def test_response_log_and_accounting_expose_only_controlled_error(
    monkeypatch, caplog, tmp_path,
):
    import ouroboros.usage_accounting as accounting

    secret = "sk-or-unit-test-secret-value-123456789"
    encoded = "c2stb3ItdW5pdC10ZXN0LXNlY3JldC12YWx1ZS0xMjM0NTY3ODk="
    raw = RuntimeError(
        f"connect https://user:{secret}@host/v1?api_key={secret} Basic {encoded}"
    )
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(accounting, "_reservation_cost", lambda _request: 0.01)
    remote = _Remote([raw])
    monkeypatch.setattr(LLMClient, "_new_remote_client", lambda _self, _target: remote)
    monkeypatch.setattr(provider_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": secret,
        "OUROBOROS_MODEL": "openai/test",
    })
    with caplog.at_level("WARNING"):
        status, body = _post({"provider_id": "openrouter"})
    assert status == 200
    assert body == {"ok": False, "error": "Model request failed"}
    ledger = (tmp_path / "state" / "usage_attempts.jsonl").read_text()
    rendered = json.dumps(body) + caplog.text + ledger
    assert secret not in rendered
    assert encoded not in rendered
    assert "user:" not in rendered
    assert f"api_key={secret}" not in rendered
    assert f"Basic {encoded}" not in rendered
    final = json.loads(ledger.splitlines()[-1])
    assert final["state"] == "unresolved"
    assert "***REDACTED***" in final["reason"]


def test_provider_test_registry_is_derived_from_provider_defaults():
    assert provider_api._PROVIDER_TEST_KNOWN_IDS == {
        "openrouter", "openai", "anthropic", "cloudru", "gigachat", "minimax",
        "deepseek", "openai-compatible",
    }
    assert provider_api._PROVIDER_TEST_OVERRIDE_KEYS == provider_api.ALL_PROVIDER_CREDENTIAL_KEYS


def test_provider_test_and_active_model_enumeration_never_select_legacy_heavy():
    from ouroboros.provider_models import ACTIVE_MODEL_SETTING_KEYS, LEGACY_MODEL_SETTING_KEYS

    assert "OUROBOROS_MODEL_HEAVY" not in ACTIVE_MODEL_SETTING_KEYS
    assert LEGACY_MODEL_SETTING_KEYS == ("OUROBOROS_MODEL_HEAVY",)
    assert provider_api._provider_test_model("anthropic", {
        "OUROBOROS_MODEL_HEAVY": "anthropic::owner-legacy-heavy",
    }) == "anthropic::claude-opus-5"
