"""The model-route previews are a Python contract with real producers.

The browser has no JSDoc twin of ``ouroboros/gateway/model_route_contracts.py``.
These tests check that the real producers emit exactly the declared fields.
"""
from __future__ import annotations

import json
import typing
from dataclasses import asdict

from ouroboros.gateway import model_route_contracts as contracts


def _py_fields(cls) -> set[str]:
    return set(typing.get_type_hints(cls, include_extras=True))


def _required(cls) -> set[str]:
    # Postponed annotations hide NotRequired from ``__required_keys__``; the resolved hints keep it.
    hints = typing.get_type_hints(cls, include_extras=True)
    return {name for name, hint in hints.items() if typing.get_origin(hint) is not contracts.NotRequired}


def test_the_real_producers_emit_exactly_the_declared_fields(tmp_path, monkeypatch):
    from ouroboros.response_limits import ResponseLimit, record_response_ack, response_limit_preview
    from ouroboros.search_routes import resolve_web_search_route

    route = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "auto", "OPENROUTER_API_KEY": "synthetic",
                                               "OUROBOROS_WEBSEARCH_MODEL": "vendor/future"})
    assert _required(contracts.WebSearchRoute) <= set(route) <= _py_fields(contracts.WebSearchRoute)
    assert route["legs"] and all(set(leg) == _required(contracts.WebSearchLeg) for leg in route["legs"])
    assert route["unavailable"] and all(set(leg) == _required(contracts.WebSearchLeg) | {"unavailable_reason"}
                                        for leg in route["unavailable"])
    refused = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "openai",
                                                 "OUROBOROS_WEBSEARCH_MODEL": "anthropic::future"})
    assert "error" in refused and set(refused) <= _py_fields(contracts.WebSearchRoute)
    named = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "deepseek", "OUROBOROS_WEBSEARCH_MODEL": "deepseek@work::",
                                               "OUROBOROS_CONNECTIONS": [{"kind": "deepseek", "name": "work", "api_key": "k"}]})
    assert set(named["legs"][0]) == _py_fields(contracts.WebSearchLeg) - {"unavailable_reason"}  # every optional fact

    assert set(asdict(ResponseLimit())) == _py_fields(contracts.ResponseLimit)
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compatible.invalid/v1")
    preview = response_limit_preview(tmp_path, {"provider": "openai-compatible", "model": "openai-compatible::m",
                                                "base_url": "https://compatible.invalid/v1", "use_local": False})
    assert set(preview) == _py_fields(contracts.ResponseLimitPreview)
    assert set(preview["route"]) == _py_fields(contracts.ResponseLimitRoute)
    assert set(preview["response_limit"]) == _py_fields(contracts.ResponseLimit)
    ack = record_response_ack(tmp_path, provider="openai-compatible", model="openai-compatible::m",
                              base_url="https://compatible.invalid/v1", max_output_tokens=4096)
    assert set(ack) == _py_fields(contracts.ResponseLimit)
    json.dumps(preview)  # the wire shape is JSON as declared
