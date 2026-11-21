"""Isolated physical transport and usage ledger shared by transport regression tests."""
import copy
from types import SimpleNamespace

import pytest

from ouroboros import config, pricing, usage_accounting as ua
from ouroboros.llm import LLMClient


@pytest.fixture(name="transport")
def physical_transport(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.llm.transport_no_proxy", lambda: False)  # the spawner's transport policy fact
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a: None)
    sent = []

    class Response:
        status_code = 200

        def __init__(self, candidate):
            self.candidate = candidate

        def model_dump(self):
            if "speed" in self.candidate:
                return self.json()
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0},
                    "service_tier": self.candidate.get("service_tier")}

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1,
                              "speed": self.candidate.get("speed")}}

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        return Response(candidate)

    async def create_async(**candidate):
        return create(**candidate)

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))))

    def post(url, *, headers, json, **kwargs):
        sent.append({"payload": copy.deepcopy(json), "headers": copy.deepcopy(headers)})
        return Response(json)

    monkeypatch.setattr("requests.post", post)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="processing", root_task_id="processing")):
        yield root, client, sent
