"""One-shot Settings provider-readiness probe."""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio
import copy
import json
import types

import httpx
import pytest

import ouroboros.gateway.models as provider_api
from ouroboros.llm import LLMClient
from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS
from ouroboros.usage_accounting import current_usage_scope
from tests._usage_store_testing import ledger_rows
from tests._safety_fixtures import settings_install as install_fixture

install = install_fixture


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


@pytest.mark.parametrize("slot", ["safety", "light", "main"])
@pytest.mark.parametrize("consumer", ["credential-plan", "provider-test"])
def test_safety_routes_reach_credential_plan_and_provider_test_without_losing_inheritance(install, monkeypatch, slot, consumer):
    from ouroboros import config
    from ouroboros.model_slots import get_safety_model
    from ouroboros.provider_models import declared_model_settings, provider_credential_plan
    from ouroboros.settings_integrity import TaskSettingsSnapshot, task_settings_scope

    route = "zai::owner-checker"
    document = {
        "OPENAI_API_KEY": "fixture-openai-key", "ZAI_API_KEY": "fixture-zai-key",
        "OUROBOROS_MODEL": "openai::owner-main", "OUROBOROS_MODEL_LIGHT": "openai::owner-light",
        "OUROBOROS_MODEL_SAFETY": "", "OUROBOROS_MODEL_FALLBACKS": "",
    }
    document[{"safety": "OUROBOROS_MODEL_SAFETY", "light": "OUROBOROS_MODEL_LIGHT", "main": "OUROBOROS_MODEL"}[slot]] = route
    if slot == "main":
        document["OUROBOROS_MODEL_LIGHT"] = ""
    install.write_text(json.dumps(document))
    loaded = config.load_settings()
    before = install.read_bytes()
    if consumer == "credential-plan":
        declared = declared_model_settings(loaded)
        plan = provider_credential_plan(loaded)
        assert plan["providers"]["zai"] == [route]
        assert "ZAI_API_KEY" in plan["planned_keys"] and plan["fail_open"] is False
        if slot == "safety":
            assert declared["OUROBOROS_MODEL_SAFETY"] == route
            without_pin = provider_credential_plan({**loaded, "OUROBOROS_MODEL_SAFETY": ""})
            assert "ZAI_API_KEY" not in without_pin["planned_keys"]
        else:
            assert "OUROBOROS_MODEL_SAFETY" not in declared
    projected = {}
    config.apply_settings_to_env(loaded, environ=projected)
    with task_settings_scope(TaskSettingsSnapshot(settings=loaded, environ=projected)):
        assert get_safety_model() == route

    if consumer == "credential-plan":
        return
    # Exercise the gateway, probe and request builder. Only the physical peer is fake.
    remote, targets = _Remote([_Completion()]), []
    def fake_remote(_client, target):
        targets.append(target)
        return remote
    monkeypatch.setattr(LLMClient, "_new_remote_client", fake_remote)
    _bypass_accounting(monkeypatch)
    status, body = _post({"provider_id": "zai"})
    assert status == 200 and body == {"ok": True}
    assert len(remote.calls) == len(targets) == 1
    assert targets[0]["provider"] == "zai" and targets[0]["api_key"] == "fixture-zai-key"
    assert remote.calls[0]["model"] == "owner-checker"
    assert install.read_bytes() == before


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "JSON object"),
        ({"provider_id": "openrouterr"}, "unknown provider id"),
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


def test_typed_key_and_draft_url_reach_request_local_target(monkeypatch):
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
        "overrides": {"OPENAI_COMPATIBLE_BASE_URL": "https://draft.example/v1",
                      "OPENAI_COMPATIBLE_API_KEY": "typed-key"},
    }
    request_before = copy.deepcopy(request)
    status, body = _post(request)
    assert status == 200 and body == {"ok": True}
    assert captured["api_key"] == "typed-key"
    assert captured["base_url"] == "https://draft.example/v1"
    assert saved == original
    assert request == request_before


@pytest.mark.parametrize(
    "overrides,expected_key,expected_base_url",
    [
        (
            # The same endpoint, respelled: the kept legacy key may still test it.
            {"OPENAI_COMPATIBLE_BASE_URL": "https://legacy.example/v1/"},
            "legacy-saved-key",
            "https://legacy.example/v1/",
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
def test_compatible_discovery_and_probe_share_the_legacy_pair_never_a_mixed_one(
    monkeypatch, legacy_key,
):
    """Without a dedicated base URL the legacy endpoint travels with its own key only: the
    dedicated key never rides it (owner decision D), and discovery and probe agree."""
    saved = {
        "OPENAI_COMPATIBLE_API_KEY": "dedicated-key",
        "OPENAI_COMPATIBLE_BASE_URL": "",
        "OPENAI_API_KEY": legacy_key,
        "OPENAI_BASE_URL": "https://legacy.example/v1",
    }
    calls = []

    async def fake_catalog(_client, _provider_id, _label, api_key, base_url, *, record_image_input=False):
        assert record_image_input is False
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
    expected = (legacy_key, "https://legacy.example/v1")
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
    for key in ALL_PROVIDER_CREDENTIAL_KEYS:
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
        ("openai-compatible", {}, ""),
    ],
)
def test_provider_test_model_selection(provider, settings, expected):
    assert provider_api._provider_test_model(provider, settings) == expected


def test_provider_test_of_an_added_key_takes_a_model_its_own_server_serves():
    """Two compatible servers list different models: Test on one never sends the other's; a fixed
    provider's ids serve on every key of it."""
    settings = {"OUROBOROS_MODEL": "openai-compatible@alpha::only-alpha",
                "OUROBOROS_MODEL_LIGHT": "openai-compatible@beta::only-beta",
                "OUROBOROS_MODEL_FALLBACKS": "anthropic::light"}
    pick = provider_api._provider_test_model
    assert pick("openai-compatible", settings, "beta") == "openai-compatible@beta::only-beta"
    assert pick("openai-compatible", settings, "alpha") == "openai-compatible@alpha::only-alpha"
    assert pick("openai-compatible", settings, "gamma") == ""  # its own catalog decides
    assert pick("anthropic", settings, "k2") == "anthropic@k2::light"


def test_catalog_is_used_only_for_compatible_discovery(monkeypatch):
    from ouroboros.settings_defaults import OPENROUTER_DEFAULTS

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
    assert probes == [OPENROUTER_DEFAULTS["main"]]  # a bare router key is probed on the shipped Main
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


@pytest.mark.parametrize("model", [
    "openai::gpt-6-astra",
    "deepseek::deepseek-v4-flash",
    "minimax::MiniMax-M3",
])
def test_provider_test_token_key_is_the_send_path_rule(monkeypatch, model):
    """Provider Test must send the token-limit carrier the real send path sends for the
    same route, for listed and unlisted model names alike."""
    from ouroboros import llm_probe

    client = LLMClient()
    target = client._resolve_remote_target(model)
    sent = client._build_remote_kwargs(
        dict(target), [{"role": "user", "content": "Reply OK"}], "medium", 16, "auto", None, None,
        skip_capability_fetch=True,
    )
    probe = llm_probe._probe_candidate(target)
    carriers = ("max_tokens", "max_completion_tokens")
    assert [key for key in carriers if key in probe] == [key for key in carriers if key in sent]


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

    # One current row per attempt; the failed probe also retains its failure before terminalization.
    finals = [row for row in ledger_rows(tmp_path) if row.get("kind") == "attempt"]
    assert len({row["attempt_id"] for row in finals}) == len(finals) == 2
    assert sorted(row["state"] for row in finals) == ["settled", "unresolved"]
    assert sorted((row["state"], row["revision"]) for row in finals) == [("settled", 3), ("unresolved", 4)]
    assert next(row for row in finals if row["state"] == "unresolved")["physical_failure"]
    for row in finals:
        assert row["task_id"] == "system:provider_test"
        assert row["root_task_id"] == "system:provider_test"
        assert row["category"] == "provider_test"
        assert row["source"] == "provider_test"
    events_path = tmp_path / "logs" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert not any(row.get("type") in {"llm_round", "llm_usage", "chat", "progress"} for row in events)
    assert not (tmp_path / "task_results").exists()
    assert not dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert not dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")


def test_a_provider_test_inside_a_task_is_that_tasks_spending(monkeypatch, tmp_path):
    """The Settings button has no task: its attempt is ``system:provider_test`` (above). A provider test a task runs
    itself (its own live qualification) stays the task's: its root's ledger, budget and Stop fences, tagged
    ``provider_test``, still one physical attempt; a system probe scope bound around it does not adopt it."""
    import ouroboros.usage_accounting as accounting

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(accounting, "_reservation_cost", lambda _request: 0.01)
    remote = _Remote([_Completion(), _Completion()])
    client = LLMClient()
    monkeypatch.setattr(client, "_new_remote_client", lambda _target: remote)
    settings = {"OPENROUTER_API_KEY": "key"}
    task = accounting.UsageScope(drive_root=tmp_path, task_id="qualify-1", root_task_id="qualify-1")
    with accounting.usage_scope(task), accounting.capture_attempt_ids() as attempt_ids:
        assert client.probe_provider_readiness("openai/test", settings=settings)["ok"] is True
    probe = accounting.UsageScope(drive_root=tmp_path, task_id="system:other", root_task_id="system:other",
                                  non_task_operation=True)
    with accounting.usage_scope(probe):
        assert client.probe_provider_readiness("openai/test", settings=settings)["ok"] is True
    rows = {row["attempt_id"]: row for row in ledger_rows(tmp_path) if row.get("kind") == "attempt"}
    assert len(attempt_ids) == 1 and len(rows) == 2
    owned = rows[attempt_ids[0]]
    assert (owned["task_id"], owned["root_task_id"], owned["source"], owned.get("non_task_operation")) == (
        "qualify-1", "qualify-1", "provider_test", None)
    [system] = [row for key, row in rows.items() if key != attempt_ids[0]]
    assert (system["task_id"], system["root_task_id"], system["non_task_operation"]) == (
        "system:provider_test", "system:provider_test", True)


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
    stored = ledger_rows(tmp_path)
    ledger = json.dumps(stored, default=str)
    rendered = json.dumps(body) + caplog.text + ledger
    assert secret not in rendered
    assert encoded not in rendered
    assert "user:" not in rendered
    assert f"api_key={secret}" not in rendered
    assert f"Basic {encoded}" not in rendered
    final = stored[-1]
    assert final["state"] == "unresolved"
    assert "***REDACTED***" in final["reason"]


def test_provider_test_registry_is_derived_from_provider_defaults():
    assert provider_api._PROVIDER_TEST_KNOWN_IDS == {
        "openrouter", "openai", "anthropic", "cloudru", "gigachat", "minimax",
        "deepseek", "zai", "openai-compatible",
    }


def test_provider_test_and_active_model_enumeration_never_select_legacy_heavy():
    from ouroboros.provider_models import ACTIVE_MODEL_SETTING_KEYS, LEGACY_MODEL_SETTING_KEYS

    assert "OUROBOROS_MODEL_HEAVY" not in ACTIVE_MODEL_SETTING_KEYS
    assert LEGACY_MODEL_SETTING_KEYS == ("OUROBOROS_MODEL_HEAVY",)
    assert provider_api._provider_test_model("anthropic", {
        "OUROBOROS_MODEL_HEAVY": "anthropic::owner-legacy-heavy",
    }) == "anthropic::claude-opus-5-5"


def test_provider_test_never_reads_the_web_search_model_as_a_chat_route():
    """A bare search id is OpenAI-native in search grammar (``search_routes``), never an OpenRouter chat id:
    testing a key picks the kind's chat routes or its maintained default, not the search slot."""
    from ouroboros.settings_defaults import OPENROUTER_DEFAULTS

    settings = {"OUROBOROS_MODEL": "openai::owner-main", "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.2"}
    assert provider_api._provider_test_model("openrouter", settings) == OPENROUTER_DEFAULTS["main"]
    assert provider_api._provider_test_model("openrouter", {**settings, "OUROBOROS_MODEL_LIGHT": "vendor/light"}) == "vendor/light"
    assert provider_api._provider_test_model("openai", settings) == "openai::owner-main"


@pytest.mark.parametrize("retired_key", [
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW", "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
])
def test_provider_test_and_declarations_ignore_retired_review_keys(retired_key):
    from ouroboros.provider_models import ACTIVE_MODEL_SETTING_KEYS, declared_model_settings

    settings = {retired_key: "anthropic::retired-review"}
    default_model = provider_api._provider_test_model("anthropic", {})
    assert default_model
    assert provider_api._provider_test_model("anthropic", settings) == default_model
    assert declared_model_settings(settings) == declared_model_settings({})
    assert retired_key not in ACTIVE_MODEL_SETTING_KEYS

    # An active fallback still participates in both consumers, even with stale
    # review bytes present. The provider resolver preserves comma-chain order.
    settings["OUROBOROS_MODEL_FALLBACKS"] = "openai::active-first,anthropic::active-fallback"
    assert provider_api._provider_test_model("anthropic", settings) == "anthropic::active-fallback"
    declared = declared_model_settings(settings)
    assert declared["OUROBOROS_MODEL_FALLBACKS"] == settings["OUROBOROS_MODEL_FALLBACKS"]
    assert retired_key not in declared


def _chat_reply(finish, *calls, content=None):
    choice = {"message": {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}}
    return {"choices": [{**choice, **({"finish_reason": finish} if finish else {})}]}


def _chat_call(arguments='{"ok":true}', **fields):
    """A Chat custom call: the form the Chat probe's custom tool is answered in."""
    return {"id": "c1", "type": "custom", "custom": {"name": "provider_test", "input": arguments}, **fields}


def _response(status, *items, **body):
    return {"status": status, "output": list(items), **body} if status else {"output": list(items), **body}


def _response_call(arguments='{"ok":true}', **fields):
    return {"type": "function_call", "call_id": "c1", "name": "provider_test", "arguments": arguments,
            "status": "completed", **fields}


_THOUGHT = {"type": "reasoning", "id": "rs_1", "summary": []}


def _openai_refusal(message, *, param, code="unsupported_value"):
    """The SDK's own HTTP 400 for a refused request form, as the direct OpenAI client raises it."""
    import openai

    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    response = httpx.Response(400, json={"error": body},
                              request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.BadRequestError(f"Error code: 400 - {json.dumps({'error': body})}", response=response, body=body)


def _chat_function_call(name, arguments):
    return _Completion({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}]}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 9}})


_CUSTOM_REFUSED = ("custom tools are not supported", "tools[0].type")
_FUNCTION_REFUSED = ("reasoning is not compatible with function tools; must use none", "reasoning_effort")


class _FormPeer(_Remote):
    """A direct OpenAI Chat peer answering by request form: the custom form, the function form with reasoning on
    (``refuse_reasoning``) and with reasoning ``none`` (``refuse_none``) get the exact typed refusals the runtime's
    ladder advances on; ``after`` scripts the answers to every later form; otherwise a call of the one tool."""

    def __init__(self, *, refuse_reasoning=False, refuse_none=False, after=()):
        super().__init__([])
        self.refuse_reasoning, self.refuse_none, self.after = refuse_reasoning, refuse_none, list(after)

    def create(self, **payload):
        self.calls.append(payload)
        tool, effort = payload["tools"][0], payload.get("reasoning_effort")
        if tool["type"] == "custom":
            raise _openai_refusal(_CUSTOM_REFUSED[0], param=_CUSTOM_REFUSED[1])
        if self.after:
            outcome = self.after.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        if (effort == "none" and self.refuse_none) or (effort != "none" and self.refuse_reasoning):
            raise _openai_refusal(_FUNCTION_REFUSED[0], param=_FUNCTION_REFUSED[1])
        name = tool["function"]["name"]
        return _chat_function_call(name, '{"ok":true}' if name == "provider_test" else '{"word":"tail"}')


def _chat_form(payload):
    return payload["tools"][0]["type"], payload.get("reasoning_effort")


@pytest.mark.parametrize(("peer", "forms", "error"), [
    # Custom refused: the function rung, at the provider's default reasoning.
    ({}, [("custom", None), ("function", None)], ""),
    # Reasoning refused beside function tools too: the runtime's task-local rung, functions with reasoning none.
    ({"refuse_reasoning": True}, [("custom", None), ("function", None), ("function", "none")], ""),
    # Every rung refused: tool calls, not the key or the model, failed on this API.
    ({"refuse_reasoning": True, "refuse_none": True}, [("custom", None), ("function", None), ("function", "none")],
     "Tool calls refused by this API"),
])
def test_a_chat_model_refusing_a_rung_is_tested_on_the_rung_its_tool_rounds_take(monkeypatch, peer, forms, error):
    """Counterexample: a saved direct OpenAI Chat model refuses the custom form, and maybe reasoning beside function
    tools, with the exact typed refusals the runtime's ladder answers (``openai_chat_dispatch``), and its real tool round
    still succeeds on the next rung. Provider Test walks the same rungs (``direct_openai_tool_candidate_ladder``), each
    its own accounted one-shot, never on another API. A pass on the reasoning-off rung names that rung in the result's
    ``applied`` mode: it confirms the tool call, not reasoning."""
    import functools

    lookup = [{"type": "function", "function": {"name": "lookup", "description": "Look up a word.", "parameters": {
        "type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}}}]
    settings = {"OPENAI_API_KEY": "key"}
    observed = _bypass_accounting(monkeypatch)
    client = LLMClient()
    monkeypatch.setattr(client, "_resolve_remote_target",
                        functools.partial(LLMClient._resolve_remote_target, client, settings=settings))
    if not error:  # the runtime itself, at its default effort, takes the same rung
        runtime = _FormPeer(**peer)
        monkeypatch.setattr(client, "_new_remote_client", lambda _target: runtime)
        message, _usage = client.chat([{"role": "user", "content": "Look up 'tail'."}], "openai::gpt-4.1",
                                      tools=lookup, tool_choice="required")
        assert [_chat_form(call)[0] for call in runtime.calls] == [form for form, _effort in forms]
        assert [call["function"]["name"] for call in message["tool_calls"]] == ["lookup"]
    observed.clear()
    probe = _FormPeer(**peer)
    monkeypatch.setattr(client, "_new_remote_client", lambda _target: probe)
    result = client.probe_provider_readiness("openai::gpt-4.1", settings=settings)
    assert (result["ok"], result.get("error", "")) == (not error, error)
    assert [_chat_form(call) for call in probe.calls] == forms
    assert all(call["max_completion_tokens"] == 4096 and not {"reasoning", "input"} & set(call) for call in probe.calls)
    assert all(call["tool_choice"] == "required" for call in probe.calls[1:])
    assert probe.calls[1]["tools"] == [{"type": "function", "function": {
        "name": "provider_test", "description": "Confirms that this connection can call a function.",
        "parameters": probe.calls[1]["tools"][0]["function"]["parameters"]}}]
    assert result["requested"] == {"api": "chat_completions", "tool_dialect": "openai_chat_custom",
                                   "effort": "provider_default", "rung": "requested_wire_form"}
    assert result["applied"] == {"api": "chat_completions", "tool_dialect": "function",
                                 "effort": forms[-1][1] or "provider_default",
                                 "rung": "task_local_availability_fallback" if forms[-1][1]
                                 else "provider_rejected_tool_dialect"}
    assert len(observed) == len(forms) and all(scope.task_id == "system:provider_test" for _request, scope in observed)


def test_provider_test_resends_nothing_but_an_exact_rung_refusal(monkeypatch):
    """Any other answer ends the test where it stands: a refusal of another field, a key refusal, a rate limit, a
    reasoning refusal that names no tool dialect, or a connection broken after the send is never followed by
    another form."""
    observed = _bypass_accounting(monkeypatch)
    for first in (_openai_refusal("Unsupported parameter: 'max_completion_tokens'", param="max_completion_tokens"),
                  _openai_refusal("Incorrect API key provided", param=None, code="invalid_api_key")):
        peer = _Remote([first])
        client = LLMClient()
        monkeypatch.setattr(client, "_new_remote_client", lambda _target, peer=peer: peer)
        result = client.probe_provider_readiness("openai::gpt-4.1", settings={"OPENAI_API_KEY": "key"})
        assert (result["ok"], result["error"], len(peer.calls)) == (False, "Model request failed", 1)
        assert result["applied"]["tool_dialect"] == "openai_chat_custom"

    class RateLimited(RuntimeError):
        status_code = 429

    after_custom = [
        (RateLimited("slow down"), "Rate limited"),
        (httpx.ReadTimeout("the connection broke after the send"), "Timed out"),
        (_openai_refusal("reasoning_effort value medium is unsupported for this model", param="reasoning_effort"),
         "Model request failed"),
    ]
    for outcome, error in after_custom:
        peer = _FormPeer(after=[outcome])
        client = LLMClient()
        monkeypatch.setattr(client, "_new_remote_client", lambda _target, peer=peer: peer)
        result = client.probe_provider_readiness("openai::gpt-4.1", settings={"OPENAI_API_KEY": "key"})
        assert (result["ok"], result["error"]) == (False, error)
        assert [_chat_form(call) for call in peer.calls] == [("custom", None), ("function", None)]  # no third form
        assert result["applied"] == {"api": "chat_completions", "tool_dialect": "function",
                                     "effort": "provider_default", "rung": "provider_rejected_tool_dialect"}
    assert len(observed) == 2 + 2 * len(after_custom)


def test_the_settings_button_says_works_without_claiming_reasoning(monkeypatch, caplog):
    """The public answer keeps its frozen shape (``ProviderTestResponse``: ok, error): a pass on the reasoning-off
    rung is "Works" for the tool call. Which form answered is in the ordinary provider_test log line, beside the one
    asked for, on the connection's own API; nothing claims reasoning was tested."""
    monkeypatch.setattr(provider_api, "load_settings", lambda: {
        "OPENAI_API_KEY": "fixture-openai-key", "OUROBOROS_MODEL": "openai::gpt-4.1"})
    _bypass_accounting(monkeypatch)
    peer = _FormPeer(refuse_reasoning=True)
    monkeypatch.setattr(LLMClient, "_new_remote_client", lambda _self, _target: peer)
    with caplog.at_level("INFO"):
        status, body = _post({"provider_id": "openai"})
    assert (status, body) == (200, {"ok": True})
    assert [_chat_form(call) for call in peer.calls] == [("custom", None), ("function", None), ("function", "none")]
    logged = next(record.getMessage() for record in caplog.records if "provider_test result=" in record.getMessage())
    assert "'applied': {'api': 'chat_completions', 'tool_dialect': 'function', 'effort': 'none'" in logged
    assert "'requested': {'api': 'chat_completions', 'tool_dialect': 'openai_chat_custom'" in logged
    assert "fixture-openai-key" not in logged


@pytest.mark.parametrize(("surface", "reply", "error"), [
    ("", _chat_reply("tool_calls", _chat_call()), ""),
    ("", _chat_reply("stop", content="I would rather not."), "No function call returned"),
    ("", _chat_reply("tool_calls", _chat_call(arguments='{"ok": false}')), "Function call malformed"),
    ("", _chat_reply("tool_calls", _chat_call(arguments="{not json")), "Function call malformed"),
    ("", _chat_reply("tool_calls", _chat_call(id="")), "Function call malformed"),
    ("", _chat_reply("length"), "Reply incomplete (length)"),  # the cap ended it: tools are not established
    ("", _chat_reply("content_filter", _chat_call()), "Reply incomplete (content_filter)"),
    ("", _chat_reply(None, _chat_call()), "Reply status unknown"),
    ("", _chat_reply("tool_calls", {"id": "c1", "type": "custom", "custom": {"name": "other", "input": '{"ok":true}'}}),
     "Function call malformed"),
    ("responses", _response("completed", _THOUGHT, _response_call()), ""),
    ("responses", _response("completed", {"type": "message", "phase": "commentary", "status": "completed",
                                          "content": [{"type": "output_text", "text": "Calling it next."}]}),
     "No function call returned"),
    ("responses", _response("completed", _response_call(arguments='{"ok": false}')), "Function call malformed"),
    ("responses", _response("completed", _response_call(status="in_progress")), "Function call malformed"),
    ("responses", _response("incomplete", _THOUGHT, incomplete_details={"reason": "max_output_tokens"}),
     "Reply incomplete (max_output_tokens)"),
    ("responses", _response("incomplete", _response_call(), incomplete_details={"reason": "content_filter"}),
     "Reply incomplete (content_filter)"),  # a call inside an unfinished reply is no pass
    ("responses", _response("incomplete", _THOUGHT), "Reply incomplete (no reason given)"),
    ("responses", _response("failed", error={"code": "server_error", "message": "boom"}), "Reply failed (server_error)"),
    ("responses", _response("", _response_call()), "Reply status unknown"),
])
def test_direct_openai_probe_is_one_attempt_whose_terminal_names_its_result(monkeypatch, surface, reply, error):
    """Direct OpenAI is tested with a required call of a harmless test function through the connection's own API,
    at the provider's default reasoning (no effort forced) and never as a text "Reply OK", a 16-token cap or
    tool_choice none. Chat asks in the Chat lane's own tool form, the runtime's custom projection with its JSON
    grammar, never plain functions beside default reasoning. One physical attempt decides: a pass needs a completed
    reply whose calls are complete calls of the test function with ``{"ok": true}``. A reply its cap or provider left
    unfinished (incomplete, failed, no terminal fact) says so as its own result, apart from a completed reply
    without the call."""
    observed = _bypass_accounting(monkeypatch)
    remote = _Remote([_Completion(reply)])
    remote.responses = types.SimpleNamespace(create=remote.create)
    client = LLMClient()
    monkeypatch.setattr(client, "_new_remote_client", lambda _target, remote=remote: remote)
    result = client.probe_provider_readiness(
        "openai::gpt-6-astra", settings={"OPENAI_API_KEY": "key", "OPENAI_API_SURFACE": surface})
    assert (result["ok"], result.get("error", "")) == (not error, error)
    [payload] = remote.calls
    token_key = "max_output_tokens" if surface else "max_completion_tokens"
    assert payload[token_key] == 4096 and payload["tool_choice"] == "required"
    assert [tool.get("name") or tool["custom"]["name"] for tool in payload["tools"]] == ["provider_test"]
    assert [tool["type"] for tool in payload["tools"]] == ["function" if surface else "custom"]
    if not surface:  # the exact projection the Chat lane sends, not a lookalike
        from ouroboros.openai_chat_custom import project_function_tool_request_to_openai_custom
        from ouroboros.llm_probe import _PROBE_FUNCTION

        lane = project_function_tool_request_to_openai_custom([{"type": "function", "function": _PROBE_FUNCTION}], "required")
        assert payload["tools"] == lane.catalog.wire_tools() and payload["tool_choice"] == lane.tool_choice
    assert ("input" in payload) is bool(surface) and not {"reasoning", "reasoning_effort", "include"} & set(payload)
    request, _scope = observed[0]
    assert len(observed) == 1 and request.max_completion_tokens == 4096 and request.source == "provider_test"
