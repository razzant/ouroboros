"""First-round calibration uses same-model, same-basis, similarly sized input."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import context_fit, loop, loop_round_limits
from ouroboros.context_anchor import InputAnchor
from ouroboros.utils import utc_now_iso
from tests.test_context_economy_window import MODEL, ROUTE, _answer, _native, _run
from tests.test_context_economy_window import live_wait as wait_fixture
from tests.test_context_economy_window import main_call as main_fixture
from tests.test_context_economy_window import setup as gateway_fixture
from tests.test_context_fit_integration import _plan

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture
_OWNER_TEXT = "Owner asks for the complete text. " + "word " * 720_000


@pytest.fixture(autouse=True)
def isolated_density_memo(monkeypatch):
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})  # each fixture has its own evidence store


@pytest.mark.parametrize(("witness_size", "expected", "mode"), [
    (97_000, 1.0, "max"),
    (97_000, 1.0, "nano"),
    (600_000, 1.56, "low"),
])
def test_real_start_and_first_send_use_comparable_density(main_call, monkeypatch, tmp_path,
                                                         mode, witness_size, expected):
    from ouroboros.tool_policy import initial_tool_schemas

    ctx, gateway, tools, events, build, windows = _native(
        main_call, monkeypatch, tmp_path, task_text=_OWNER_TEXT, owner_mode=mode)
    windows["account-a"] = 2_200_000
    ce.record_token_density(ce.canonical_evidence_root(), ce._normalized_density_model(MODEL),
        prompt_chars=witness_size * 4, prompt_tokens=int(witness_size * 1.56),
        route_fp=_route("account-a"), basis=ce.MAIN_DENSITY_BASIS, source="dispatch_usage", view_kind="owner_turn")
    plan = build(windows["account-a"])
    assert plan.initial_mode == plan.preferred_mode == mode
    assert plan.projection(mode).calibration_ratio == expected
    ctx.messages[:] = plan.messages_for(mode)
    assert any(row.get("content") == _OWNER_TEXT for row in ctx.messages)
    ctx.context_fit_plan, ctx.active_context_mode = plan, mode
    ctx.tool_schemas = initial_tool_schemas(tools, context_mode=mode)
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert 900_000 <= fit.raw_input_tokens < 1_000_000  # owner text plus actual governance and schemas
    assert fit.measurement_density == expected and fit.anchor_input_tokens is None
    gateway.results = [_answer(ROUTE, {"role": "assistant", "content": "Read."})]
    gateway.dispatch = ["response_received"]
    text, usage, _trace = _run(ctx, tools, events, plan)
    assert text == "Read." and len(gateway.creates) == 1
    assert usage["_context_measurement_density"] == expected
    assert tools._ctx.active_context_mode == mode
    assert any(context_fit.extract_plain_text_from_content(row.get("content")) == _OWNER_TEXT
               for row in gateway.uploads[0][0]["messages"])


@pytest.mark.parametrize("witness_size,window,expected", [
    (97_000, 2_000_000, 1.0), (600_000, 2_000_000, 1.56),
    # A running input above the new route's window is compared at what that route can send (its usable input,
    # resolved before the ratio): a witness of that size calibrates it; the uncut input matched nothing (07.11).
    (300_000, 600_000, 1.56)])
def test_real_route_rebind_sizes_its_running_input(main_call, monkeypatch, tmp_path, witness_size, window, expected):
    from ouroboros.tool_policy import initial_tool_schemas

    ctx, _gateway, tools, _events, build, windows = _native(
        main_call, monkeypatch, tmp_path, owner_mode="nano")
    windows["account-a"], windows["account-b"] = 2_000_000, window
    plan = build(windows["account-a"])
    ctx.messages[:] = [*plan.messages_for("nano"), {"role": "user", "content": _OWNER_TEXT}]
    shown = list(ctx.messages)
    ce.record_token_density(ce.canonical_evidence_root(), ce._normalized_density_model(MODEL),
        prompt_chars=witness_size * 4, prompt_tokens=int(witness_size * 1.56),
        route_fp=_route("account-b"), basis=ce.MAIN_DENSITY_BASIS, source="dispatch_usage", view_kind="owner_turn")
    ctx.tool_schemas = initial_tool_schemas(tools, context_mode="nano")
    rebound, mode = loop._rebind_context_fit_plan(plan, tools, ctx.messages, model=MODEL, use_local=False,
        tool_schemas=ctx.tool_schemas, start_mode="nano", credential_profile_id="account-b")
    assert mode == rebound.preferred_mode == "nano" and rebound.projection(mode).calibration_ratio == expected
    if window < 2_000_000:
        return  # this transcript cannot fit that route: what it sends next is the refusal passes', not the rebind's
    assert ctx.messages == shown
    ctx.context_fit_plan, ctx.active_context_mode = rebound, mode
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert fit.measurement_density == expected and fit.anchor_input_tokens is None


def _route(account):
    return ce.route_fingerprint(provider="claudexor", model=MODEL, base_url="", options={
        "source_id": "codex", "credential_profile_id": account, "account_fingerprint": f"fingerprint-{account[-1]}"})


def _witness(raw, density=1.56, *, route="route-a", basis=ce.MAIN_DENSITY_BASIS, observed=None):
    return {"prompt_chars": raw * 4, "prompt_tokens": int(raw * density), "route_fp": route,
            "basis": basis, "observed_at": observed or utc_now_iso()}


@pytest.mark.parametrize("witness_size,model,basis,observed,expected", [
    (600_000, "openai/test-model", ce.MAIN_DENSITY_BASIS, None, 1.56),
    (97_000, "openai/test-model", ce.MAIN_DENSITY_BASIS, None, 1.0),
    (600_000, "openai/test-model", ce.MAIN_DENSITY_BASIS, "2000-01-01T00:00:00Z", 1.0),
])
def test_round_one_uses_only_comparable_same_model_evidence(
        tmp_path, monkeypatch, witness_size, model, basis, observed, expected):
    plan = _plan(preferred="max", window=1_000_000)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {
        model: {"pairs": [_witness(witness_size, basis=basis, observed=observed,
                                   route="route-a" if model == plan.model else "foreign-route")]}}})
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="max", active_model=plan.model,
        messages=plan.messages_for("max") + [{"role": "user", "content": "я" * 2_400_000}],
        tool_schemas=[], active_effort="", accumulated_usage={}, tools=SimpleNamespace(_ctx=SimpleNamespace()),
        task_id="turn", round_idx=1)
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert fit.measurement_density == expected
    assert fit.estimated_input_tokens == math.ceil(fit.raw_input_tokens * expected)
    assert fit.anchor_input_tokens is None
    assert (fit.capacity_deficit_tokens > 0) == (expected > 1)
    assert ctx.accumulated_usage["_context_measurement_density"] == expected


@pytest.mark.parametrize("witness_size,expected", [(30_000, 1.05), (300_000, 1.0)])
def test_a_start_compares_density_at_the_size_it_can_send(tmp_path, monkeypatch, witness_size, expected):
    """The start's whole captured request (391k estimated tokens here) is compared at what its route can send, its
    usable input (26,464 beside a 92k window and a 65k reply): a witness near that size calibrates it, one near the
    uncut size no longer does. On 07.11 an uncut 5.4 million beside a window under a million matched no witness."""
    from tests import _memory_scale_fixture as scale

    model = "openai/test-model"
    monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {model: {"pairs": [
        {**_witness(witness_size, 1.05), "view_kind": "owner_turn"}]}}})
    evidence = SimpleNamespace(window_tokens=92_000, status="confirmed", stale=False, route_fp="route-a",
                               source="fixture", response_limit={}, route_facts={"working_reserve_tokens": 10_000})
    plan = context_fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), scale.core(tmp_path),
        {"id": "start", "chat_id": 1, "_is_direct_chat": True}, preferred_mode="max", tool_schemas=scale.tools(),
        route_resolver=lambda *_a, **_kw: ({"provider": "openai", "model": model}, evidence))
    assert plan.window_record.usable == 26_464 and plan.projection(plan.initial_mode).calibration_ratio == expected


@pytest.mark.parametrize("model,view,basis,observed,expected", [
    ("claudexor::codex=exact-model", "owner_turn", ce.MAIN_DENSITY_BASIS, None, (1.25, "fresh_model_usage")),
    ("claudexor::codex=exact-model", "own_wake", ce.MAIN_DENSITY_BASIS, None, (1.0, "cold_estimate")),
    ("claudexor::codex=exact-model", "owner_turn", "raw", None, (1.0, "cold_estimate")),
    ("claudexor::codex=exact-model", "owner_turn", ce.MAIN_DENSITY_BASIS, "2000-01-01T00:00:00Z", (1.0, "cold_estimate")),
    ("openai/test-model", "owner_turn", ce.MAIN_DENSITY_BASIS, None, (1.0, "cold_estimate")),
])
def test_a_claudexor_model_lends_another_accounts_witness_of_the_same_view(tmp_path, monkeypatch, model, view, basis,
                                                                          observed, expected):
    """A start asks under the account the catalog advertises; witnesses are written under the account that served.
    A Claudexor model's accounts share one engine and tokenizer, so without an exact-route witness the newest
    comparable one of the same model, view and basis serves as its density (``fresh_model_usage``). Another view,
    basis, a stale witness or another provider's route never does, and the exact route still comes first."""
    other = {**_witness(600_000, 1.25, route="account-b", basis=basis, observed=observed), "view_kind": view}
    exact = {**_witness(600_000, 1.4, route="account-a"), "view_kind": "owner_turn"}
    for pairs, answer in (([other], expected), ([other, exact], (1.4, "fresh_route_usage"))):
        monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {model: {"pairs": pairs}}})
        assert ce.resolve_main_token_density(tmp_path, "account-a", model, estimate_tokens=600_000,
                                            view_kind="owner_turn") == answer


@pytest.mark.parametrize("route", ["route-a", "other-route"])
def test_comparability_is_symmetric_and_never_crosses_routes(monkeypatch, tmp_path, route):
    monkeypatch.setattr(ce, "_read", lambda root: {"token_density": {"openai/test-model": {"pairs": [
        _witness(97_000, 0.8), _witness(600_000, route=route),
    ]}}})
    assert ce.resolve_main_token_density(tmp_path, "route-a", "openai/test-model",
                                        estimate_tokens=600_000) == (
                                            (1.56, "fresh_route_usage") if route == "route-a" else (1.0, "cold_estimate"))
    for size, expected in [(300_000, 1.56), (1_200_000, 1.56), (1_200_001, 1.0)]:
        assert ce.resolve_main_token_density(tmp_path, "route-a", "openai/test-model",
                                            estimate_tokens=size)[0] == (expected if route == "route-a" else 1.0)


def test_task_anchor_outranks_even_comparable_foreign_measurements(monkeypatch):
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *a, **k: pytest.fail("anchor must win"))
    plan = _plan()
    fit = context_fit.measure_main_fit(plan, plan.messages_for("low"), [], profile="owner_low",
        rendered_mode="low", round_id="r2", anchor=InputAnchor(150_000, 100_000, 1))
    assert fit.measurement.measurement_density == 1.5


def test_memory_coarsening_uses_the_density_that_measured_the_saving(monkeypatch):
    plan = _plan()  # a real captured core without a required handbook: it adds nothing to the window
    plan = replace(plan, core=context_fit.ContextCore("", "", "", "", "", "", plan.user_content_json, False))
    fit = context_fit.measure_main_fit(plan, plan.messages_for("low"), [], profile="owner_low",
        rendered_mode="low", round_id="r2", anchor=InputAnchor(150_000, 100_000, 1))
    seen = []
    monkeypatch.setattr(context_fit.ContextFitPlan, "reproject_for_route",
                        lambda self, **kw: seen.append(kw) or self)
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="low", messages=plan.messages_for("low"),
                          tool_schemas=[])
    assert not loop_round_limits._coarsen_memory_view(ctx, fit, saving=30_000)
    assert seen[0]["ratio"] == 1.5 and seen[0]["refused_tokens"] == 30_000
