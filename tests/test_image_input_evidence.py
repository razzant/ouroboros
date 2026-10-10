"""Image input as route evidence: the task-start window probe of a compatible gateway.

The probe already reads the gateway's /models; the same response states image input
for that exact route (provider, endpoint), valid for 24 hours. MiniMax's /models has no
modality field and is never a source (DEVELOPMENT "External facts: unknown is not no").
"""
from __future__ import annotations

import ouroboros.capability_evidence as ce


def _gateway_models(monkeypatch, items):
    import httpx

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": items}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())


def test_task_start_probe_records_a_compatible_gateways_image_input(tmp_path, monkeypatch):
    """The window probe already reads the gateway's /models; the same response
    states image input for the exact route, also when it publishes no window."""
    from ouroboros.provider_models import supports_vision

    base_url = "http://gateway.invalid/v1"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "k")
    _gateway_models(monkeypatch, [
        {"id": "giga-osa-glm53/glm-5.3-flash", "capabilities": {"supports_vision": True}},
        {"id": "giga-osa-glm53/glm-5.3", "capabilities": {"supports_vision": False}, "max_model_len": 131072},
        {"id": "plain-vllm-model", "max_model_len": 8192},
    ])
    ev = ce.probe(tmp_path, provider="openai-compatible", model="openai-compatible::giga-osa-glm53/glm-5.3-flash",
                  base_url=base_url, allow_fetch=True, api_key="k")
    assert ev.status == ce.STATUS_UNPROBEABLE  # no window: the image fact does not depend on it
    assert supports_vision("openai-compatible::giga-osa-glm53/glm-5.3-flash") is True
    assert supports_vision("openai-compatible::giga-osa-glm53/glm-5.3") is False
    assert supports_vision("openai-compatible::plain-vllm-model") is None  # publishes nothing: unknown
    # The statement belongs to that endpoint; the same slug elsewhere stays unknown.
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://another-gateway.invalid/v1")
    assert supports_vision("openai-compatible::giga-osa-glm53/glm-5.3") is None
    assert supports_vision("giga-osa-glm53/glm-5.3") is None


def test_minimax_models_listing_is_never_an_image_input_source(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _gateway_models(monkeypatch, [{"id": "MiniMax-M3", "supports_vision": False, "context_length": 200000}])
    assert ce._provider_metadata_window("minimax", "minimax::MiniMax-M3", "https://api.minimax.invalid/v1",
                                        allow_fetch=True, api_key="k") == 200000
    assert ce._load(tmp_path)["image_input"] == {}
    from ouroboros.provider_models import supports_vision
    assert supports_vision("minimax::MiniMax-M3") is None
