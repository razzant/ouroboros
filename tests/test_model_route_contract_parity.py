"""The model-route previews have a browser twin: every TypedDict in
ouroboros/gateway/model_route_contracts.py has a JSDoc typedef with the same fields in
web/modules/model_route_types.js (the twin pair beside gateway/contracts.py + api_types.js),
and the real producers emit exactly the declared fields."""
from __future__ import annotations

import json
import pathlib
import typing
from dataclasses import asdict

from ouroboros.gateway import model_route_contracts as contracts
from tests.test_gateway_parity import _js_typedef_fields

REPO = pathlib.Path(__file__).resolve().parents[1]
TWIN = REPO / "web" / "modules" / "model_route_types.js"


def _py_fields(cls) -> set[str]:
    return set(typing.get_type_hints(cls, include_extras=True))


def _required(cls) -> set[str]:
    # Postponed annotations hide NotRequired from ``__required_keys__``; the resolved hints keep it.
    hints = typing.get_type_hints(cls, include_extras=True)
    return {name for name, hint in hints.items() if typing.get_origin(hint) is not contracts.NotRequired}


def test_every_model_route_envelope_has_a_field_identical_browser_twin():
    text = TWIN.read_text(encoding="utf-8")
    assert contracts.__all__, "the twin pair lost its envelopes"
    for name in contracts.__all__:
        assert f"@typedef {{Object}} {name}\n" in text, f"model_route_types.js lacks {name}"
        assert _js_typedef_fields(text, name) == _py_fields(getattr(contracts, name)), name
    # The main mirror points at the twin, and the client types resolve against it.
    api_types = (REPO / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    assert "model_route_types.js" in api_types.splitlines()[0]
    client = (REPO / "web" / "modules" / "api_client.js").read_text(encoding="utf-8")
    for name in ("WebSearchRoute", "ResponseLimitPreview", "ResponseLimitAckResponse"):
        assert f"import('./model_route_types.js').{name}" in client, name


def test_the_real_producers_emit_exactly_the_declared_fields(tmp_path, monkeypatch):
    from ouroboros.response_limits import ResponseLimit, record_response_ack, response_limit_preview
    from ouroboros.search_routes import resolve_web_search_route

    route = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "auto", "OPENROUTER_API_KEY": "synthetic",
                                               "OUROBOROS_WEBSEARCH_MODEL": "vendor/future"})
    assert _required(contracts.WebSearchRoute) <= set(route) <= _py_fields(contracts.WebSearchRoute)
    assert route["legs"] and all(set(leg) == _py_fields(contracts.WebSearchLeg)
                                 for leg in route["legs"] + route["unavailable"])
    refused = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "openai",
                                                 "OUROBOROS_WEBSEARCH_MODEL": "anthropic::future"})
    assert "error" in refused and set(refused) <= _py_fields(contracts.WebSearchRoute)

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
