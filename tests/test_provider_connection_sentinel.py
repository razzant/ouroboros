"""Sentinel: a route through a named connection never carries the built-in key.

The built-in OpenRouter key is SENT-A, the named ``openrouter@byok`` key SENT-B.
Main, Light, a reviewer row, web search, image generation, the window probe and
Provider Test through the named route build no client and send no header with
SENT-A. Every seam records what it was handed and stops before any network send.
The static guard keeps provider credential settings named only by the resolver.
"""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.llm import LLMClient
from ouroboros.provider_models import CONNECTIONS_KEY, ROUTE_CREDENTIAL_SETTING_KEYS

SENT_A, SENT_B = "sk-or-SENT-A-builtin-000000", "sk-or-SENT-B-named-111111"
COMPAT_A, COMPAT_B = "sk-SENT-A-compatible-2222", "sk-SENT-B-vllm-3333"
NAMED = "openrouter@byok::anthropic/claude-opus-5.5"
REPO = pathlib.Path(__file__).resolve().parents[1]


class _Stop(RuntimeError):
    """Raised by every recording seam: nothing reaches the network."""


@pytest.fixture()
def recorded(monkeypatch):
    for key in ROUTE_CREDENTIAL_SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", SENT_A)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", COMPAT_A)
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compat.example/v1")
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([
        {"kind": "openrouter", "name": "byok", "label": "Claude credits", "api_key": SENT_B},
        {"kind": "openai-compatible", "name": "vllm", "api_key": COMPAT_B, "base_url": "https://vllm.example/v1"},
    ]))
    calls: list = []

    def record(**facts):
        calls.append(facts)
        raise _Stop("recorded; no send")

    monkeypatch.setattr(LLMClient, "_new_remote_client", staticmethod(lambda target: record(target=dict(target))))
    from ouroboros import net_transport
    monkeypatch.setattr(net_transport, "web_search_openai_client", lambda **kwargs: record(**kwargs))
    return calls


def _no_builtin_key(calls) -> None:
    text = json.dumps(calls, default=str)
    assert calls and SENT_A not in text and COMPAT_A not in text


@pytest.mark.parametrize("role", ["main", "light", "reviewer:named-row"])
def test_chat_through_the_named_route_uses_its_key(recorded, role):
    with pytest.raises(Exception):
        LLMClient().chat([{"role": "user", "content": "hi"}], model=NAMED, model_role=role)
    _no_builtin_key(recorded)
    target = recorded[-1]["target"]
    assert target["api_key"] == SENT_B and target["connection"] == "byok"
    assert target["usage_model"] == NAMED and target["default_headers"]["X-OpenRouter-Metadata"] == "enabled"


def test_the_built_in_route_still_uses_the_built_in_key(recorded):
    with pytest.raises(Exception):
        LLMClient().chat([{"role": "user", "content": "hi"}], model="anthropic/claude-opus-5.5")
    assert recorded[-1]["target"]["api_key"] == SENT_A  # the twin: the guard does not hide the built-in key


def test_web_search_through_the_named_route_uses_its_key(recorded, monkeypatch):
    from ouroboros.tools import search

    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openrouter")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "openrouter@byok::openai/gpt-5.2")
    ctx = SimpleNamespace(task_id="", task_metadata={}, pending_events=[])
    search._web_search(ctx, "sentinel query")
    _no_builtin_key(recorded)
    assert recorded[-1]["api_key"] == SENT_B


def test_image_generation_through_the_named_route_uses_its_key(recorded):
    from ouroboros.gateways.image_api import resolve_image_target

    target = resolve_image_target("openrouter@byok::google/gemini-image")
    assert target["api_key"] == SENT_B and SENT_A not in json.dumps(target)


def test_the_window_probe_sends_no_built_in_key(recorded, monkeypatch, tmp_path):
    import httpx
    import requests

    from ouroboros import capability_evidence

    seen = []

    def fake_get(url, **kwargs):
        seen.append({"url": url, **kwargs})
        payload = {"data": [{"id": "anthropic/claude-opus-5.5", "context_length": 1000000},
                            {"id": "llama", "max_model_len": 131072}]}
        return SimpleNamespace(status_code=200, json=lambda: payload, raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(httpx, "get", fake_get)
    for name, value in (("_SUPPORTED_PARAMS_FETCHED", False), ("_CAPABILITIES_FETCH_OK", False),
                        ("_CONTEXT_LENGTH_CACHE", {}), ("_SUPPORTED_PARAMS_CACHE", {})):
        monkeypatch.setattr(LLMClient, name, value)
    named = capability_evidence.probe(tmp_path, provider="openrouter", model=NAMED, base_url="", allow_fetch=True)
    compatible = capability_evidence.probe(tmp_path, provider="openai-compatible", model="openai-compatible@vllm::llama",
                                           base_url="https://vllm.example/v1", allow_fetch=True)
    assert (named.window_tokens, compatible.window_tokens) == (1000000, 131072)
    _no_builtin_key(seen)
    assert seen[-1]["headers"] == {"Authorization": f"Bearer {COMPAT_B}"}


def test_provider_test_through_the_named_connection_uses_its_key(recorded):
    import ouroboros.gateway.models as models_api

    class _Request:
        async def json(self):
            return {"provider_id": "openrouter", "connection": "byok"}

    response = asyncio.run(models_api.api_provider_test(_Request()))
    assert json.loads(response.body.decode("utf-8"))["ok"] is False  # the recording seam stopped the send
    _no_builtin_key(recorded)
    assert recorded[-1]["target"]["api_key"] == SENT_B


# --- Static guard ------------------------------------------------------------------------

# The resolver and the kind table own the names; defaults and masking declare them.
OWNERS = {"ouroboros/provider_models.py", "ouroboros/llm_routing.py",
          "ouroboros/settings_defaults.py", "ouroboros/secret_masking.py"}
# Every other production module that names one, and why it may.
JUSTIFIED = {
    "ouroboros/contracts/plugin_api.py": "frozen skill ABI: FORBIDDEN_SKILL_SETTINGS declares the owner-granted keys",
    "ouroboros/colab_bootstrap.py": "settings writer: copies Colab secrets into the built-in flat settings",
    "ouroboros/settings_setup_contract.py": "onboarding writer: the wizard's fields are the built-in settings it saves",
    "supervisor/state.py": "OpenRouter ground-truth drift diagnostic of the built-in key; never a money owner",
    "scripts/run_external_review.py": "operator wrapper: provisions the built-in keys from the operator's own secrets",
    "scripts/reconcile_openrouter_cost.py": "explicit reconciliation: the saved OpenRouter key whose fingerprint bought "
                                            "the row, built-in or added; any other refuses as credential_mismatch, unsent",
    "scripts/qualify_linux_appimage.py": "packaging qualification: writes an endpoint into its own test settings",
}


def _named_settings(source: str) -> set[str]:
    tree = ast.parse(source)
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings and node.value in ROUTE_CREDENTIAL_SETTING_KEYS}


def test_only_the_resolver_names_provider_credential_settings():
    files = [path for root in ("ouroboros", "supervisor", "scripts") for path in (REPO / root).rglob("*.py")]
    sources = {path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8")
               for path in [*files, REPO / "server.py", REPO / "launcher.py"]}
    readers = {name for name, text in sources.items()
               if any(key in text for key in ROUTE_CREDENTIAL_SETTING_KEYS) and _named_settings(text)}
    assert readers - OWNERS == set(JUSTIFIED), (
        "a production module reads a provider credential setting by name; resolve the route instead "
        f"(llm_routing.resolve_target) or justify it here: unexpected={sorted(readers - OWNERS - set(JUSTIFIED))}, "
        f"stale={sorted(set(JUSTIFIED) - readers)}")


def test_the_guard_sees_a_literal_read_and_ignores_a_docstring():
    assert _named_settings('def f():\n    """Reads OPENROUTER_API_KEY."""\n    return 1\n') == set()
    assert _named_settings('import os\nkey = os.environ.get("OPENROUTER_API_KEY")\n') == {"OPENROUTER_API_KEY"}
    assert _named_settings('rows = settings["OUROBOROS_CONNECTIONS"]\n') == {"OUROBOROS_CONNECTIONS"}
