"""Local Main's configured route owns witnesses stored as the wire model local-model."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, context, loop
from ouroboros.context_fit import build_context_fit_plan
from tests._memory_view_context import world
from tests.test_context_comparable_density import _witness

MODEL = "local-fixture"


@pytest.fixture
def routes(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})
    local = ce.route_fingerprint(provider="local", base_url="", model=MODEL)
    remote = ce.route_fingerprint(provider="openai", base_url="", model=MODEL)
    foreign = ce.route_fingerprint(provider="local", base_url="", model="different-model")
    return local, remote, foreign


@pytest.mark.parametrize("raw,expected", [(300_000, 1.4), (600_000, 1.4), (1_200_000, 1.4),
                                         (299_999, 1.0), (1_200_001, 1.0)])
def test_exact_local_route_crosses_storage_keys_but_respects_size(routes, monkeypatch, tmp_path, raw, expected):
    local, remote, foreign = routes
    monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {
        "local-model": {"pairs": [_witness(600_000, 1.4, route=local), _witness(600_000, 2.0, route=foreign)]},
        MODEL: {"pairs": [_witness(600_000, 0.9, route=remote)]},
    }})
    ratio, basis = ce.resolve_main_token_density(tmp_path, local, MODEL, estimate_tokens=raw)
    assert ratio == expected
    assert basis == ("fresh_route_usage" if expected == 1.4 else "cold_estimate")


def test_foreign_local_model_and_stale_or_old_basis_witnesses_cannot_calibrate(routes, monkeypatch, tmp_path):
    local, _remote, foreign = routes
    monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {
        "local-model": {"pairs": [_witness(600_000, 1.4, route=foreign),
                                  _witness(600_000, 1.4, route=local, observed="2000-01-01T00:00:00Z"),
                                  _witness(600_000, 1.4, route=local, basis="raw")]},
    }})
    assert ce.resolve_main_token_density(tmp_path, local, MODEL, estimate_tokens=600_000) == (1.0, "cold_estimate")


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_local_witness_reaches_startup_rebind_and_first_round(routes, monkeypatch, tmp_path, mode):
    local, remote, _foreign = routes
    for model, route, ratio in (("local-model", local, 1.4), (MODEL, remote, 0.9)):
        ce.record_token_density(tmp_path, model, prompt_chars=600_000 * 4,
                                prompt_tokens=int(600_000 * ratio), route_fp=route, basis=ce.MAIN_DENSITY_BASIS,
                                view_kind="owner_turn")
    env, memory, _rooms = world(tmp_path / "world")
    task = {"type": "task", "text": "word " * 480_000, "id": "local-density", "chat_id": 1}
    core = context._capture_context_core(env, memory, task, None, None)
    evidence = SimpleNamespace(route_fp=local, status="confirmed", stale=False, window_tokens=2_000_000,
                               source="published_server_arguments", source_id="local")
    route = lambda *a, **kw: ({"model": MODEL, "provider": "local", "use_local": True}, evidence)
    monkeypatch.setattr(context, "_context_fit_route", route)
    plan = build_context_fit_plan(env, core, task, preferred_mode=mode, route_resolver=route, tool_schemas=[])
    assert plan.model == MODEL and plan.projection(mode).calibration_ratio == 1.4
    messages = plan.messages_for(mode)
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}, _accumulated_usage={}))
    # Start from a plan whose previous route used another density; rebind must look up local again.
    old = replace(plan, route_fp=remote, provider="openai")
    rebound, effective = loop._rebind_context_fit_plan(
        old, tools, messages, model=MODEL, use_local=True, tool_schemas=[], start_mode=mode)
    assert effective == mode and rebound.projection(mode).calibration_ratio == 1.4
    ctx = SimpleNamespace(context_fit_plan=rebound, active_context_mode=mode, active_model=MODEL,
                          messages=messages, tool_schemas=[], active_effort="", accumulated_usage={},
                          tools=tools, task_id="local-density", round_idx=1)
    measured = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert measured.measurement_basis == "fresh_route_usage" and measured.measurement_density == 1.4
    assert measured.anchor_input_tokens is None
    assert measured.estimated_input_tokens == math.ceil(measured.raw_input_tokens * 1.4)
