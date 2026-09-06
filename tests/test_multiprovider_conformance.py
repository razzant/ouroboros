"""CPL-6 conformance contracts for the multi-provider seams (plan §7.6).

One normative shared suite over the two seams every install crosses:

1. **LLM providers** — every provider lane of ``ouroboros/llm.py`` must pass
   the SAME parametrized contract: route-resolution form, ``(message, usage)``
   response shape, honest-only cost planes (an unknown cost is ``None``, never
   an invented number), typed error handling (a transport failure raises and
   never fabricates an answer; a typed policy refusal is permanent — exactly
   one physical send; an HTTP-200 body rate limit is a typed usage marker;
   ``finish_reason: null`` is surfaced observably), caller-timeout propagation,
   and one settled physical-attempt ledger row per successful send.
   The parametrization derives from the FACTUAL provider registry
   (``provider_models.PROVIDER_PREFIXES``) plus the local lane — a new
   provider registered without a conformance driver turns this suite red.
   Transports are the recording fakes of ``tests/test_llm_provider_golden.py``
   (reuse, not a copy): nothing touches the network and no fixture carries a
   real credential.

2. **The executor axis (native | harness)** — every value of the axis
   vocabulary (``subagents.SUBAGENT_EXECUTORS``) must pass the same typed
   rule-table contract across every route state (launch semantics), leave the
   durable delegation artifact when a run starts (artifact semantics), and
   refuse typed — never silently — when its substrate is missing (refusal
   semantics). A new axis point added to the vocabulary without a conformance
   row turns this suite red. Native-side task artifacts (task results,
   outcome receipts) are pinned by their own suites; here the native point
   pins the resolution row and the no-daemon-contact guarantee.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pytest

from ouroboros.provider_models import PROVIDER_PREFIXES
from tests.test_llm_provider_golden import _observe

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _HealthStub,
    _dispatch,
    _owned_gateway_uses_each_test_transport,
    _started_request,
)


# ===========================================================================
# 1. LLM provider conformance
# ===========================================================================

_MSGS = [
    {"role": "system", "content": "stable policy prefix"},
    {"role": "user", "content": "hello"},
]


def _choice_body(finish_reason: Any = "stop") -> Dict[str, Any]:
    return {
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": "ok"},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def _openai_success(finish_reason: Any = "stop") -> Dict[str, Any]:
    return {"kind": "response", "body": _choice_body(finish_reason)}


_ANTHROPIC_SUCCESS = {
    "kind": "response",
    "status_code": 200,
    "json": {
        "id": "msg_conformance", "type": "message", "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    },
}

_GIGACHAT_SUCCESS = {
    "kind": "response",
    "body": {"message": {"content": "ok"},
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
}


class ProviderDriver:
    """How one provider lane is exercised by the shared contract cases."""

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        env: Dict[str, str],
        success_step: Dict[str, Any],
        timeout_of: Callable[[Dict[str, Any]], Optional[float]],
        chat_kwargs: Optional[Dict[str, Any]] = None,
        spec_extra: Optional[Dict[str, Any]] = None,
        choice_shaped: bool = False,  # OpenAI choice family (finish_reason surface)
        free_lane: bool = False,      # cost is honestly 0.0 by contract (local)
        remote: bool = True,          # resolves through _resolve_remote_target
    ) -> None:
        self.provider = provider
        self.model = model
        self.env = dict(env)
        self.success_step = success_step
        self.timeout_of = timeout_of
        self.chat_kwargs = dict(chat_kwargs or {})
        self.spec_extra = dict(spec_extra or {})
        self.choice_shaped = choice_shaped
        self.free_lane = free_lane
        self.remote = remote

    def spec(self, transport: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        call_kwargs: Dict[str, Any] = {"messages": _MSGS, "model": self.model}
        call_kwargs.update(self.chat_kwargs)
        call_kwargs.update(kwargs.pop("chat_kwargs", {}) or {})
        spec: Dict[str, Any] = {
            "env": dict(self.env),
            "transport": list(transport),
            "pricing_estimate": None,
            "call": {"kind": "method", "name": "chat", "kwargs": call_kwargs},
        }
        spec.update(self.spec_extra)
        spec.update(kwargs)
        return spec


def _payload_timeout(send: Dict[str, Any]) -> Optional[float]:
    return (send.get("payload") or {}).get("timeout")


def _row_timeout(send: Dict[str, Any]) -> Optional[float]:
    return send.get("timeout")


def _client_timeout(send: Dict[str, Any]) -> Optional[float]:
    return (send.get("client") or {}).get("timeout")


def _openai_family(provider: str, model: str, env: Dict[str, str]) -> ProviderDriver:
    return ProviderDriver(
        provider, model=model, env=env, success_step=_openai_success(),
        timeout_of=_payload_timeout, choice_shaped=True,
    )


PROVIDER_DRIVERS: Dict[str, ProviderDriver] = {
    "openrouter": _openai_family(
        "openrouter", "some/model", {"OPENROUTER_API_KEY": "or-conformance-key"}),
    "openai": _openai_family(
        "openai", "openai::gpt-x", {"OPENAI_API_KEY": "openai-conformance-key"}),
    "openai-compatible": _openai_family(
        "openai-compatible", "openai-compatible::model-x",
        {"OPENAI_COMPATIBLE_API_KEY": "compatible-conformance-key",
         "OPENAI_COMPATIBLE_BASE_URL": "https://compatible.conformance.invalid/v1"}),
    "cloudru": _openai_family(
        "cloudru", "cloudru::model-x",
        {"CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloudru-conformance-key"}),
    "minimax": _openai_family(
        "minimax", "minimax::model-x", {"MINIMAX_API_KEY": "minimax-conformance-key"}),
    "deepseek": _openai_family(
        "deepseek", "deepseek::model-x", {"DEEPSEEK_API_KEY": "deepseek-conformance-key"}),
    "anthropic": ProviderDriver(
        "anthropic", model="anthropic::claude-x",
        env={"ANTHROPIC_API_KEY": "anthropic-conformance-key"},
        success_step=_ANTHROPIC_SUCCESS, timeout_of=_row_timeout,
    ),
    "gigachat": ProviderDriver(
        "gigachat", model="gigachat::GigaChat-X",
        env={"GIGACHAT_CREDENTIALS": "gigachat-conformance-key"},
        success_step=_GIGACHAT_SUCCESS, timeout_of=_client_timeout,
    ),
    "local": ProviderDriver(
        "local", model="local", env={"LOCAL_MODEL_PORT": "8799"},
        success_step=_openai_success(), timeout_of=_payload_timeout,
        chat_kwargs={"use_local": True},
        spec_extra={"local_context_length": 8192},
        # CPL6-F1 closed: the local lane now stamps usage provider/resolved_model
        # symmetrically with every remote lane, so no asymmetry flag remains.
        choice_shaped=False,  # local normalizes its own text/tool-call path
        free_lane=True, remote=False,
    ),
}

_DRIVERS = sorted(PROVIDER_DRIVERS)
_REMOTE = [name for name in _DRIVERS if PROVIDER_DRIVERS[name].remote]


def test_conformance_parametrization_derives_from_the_provider_registry():
    """Structural completeness: a provider registered in PROVIDER_PREFIXES
    without a conformance driver is RED — a new provider cannot ship without
    passing this suite. The local lane rides along explicitly."""
    registry = {provider for _prefix, provider in PROVIDER_PREFIXES}
    assert registry, "the provider registry parsed empty"
    missing = registry - set(PROVIDER_DRIVERS)
    assert not missing, (
        f"provider(s) {sorted(missing)} are registered in PROVIDER_PREFIXES but have "
        "no conformance driver — add a PROVIDER_DRIVERS entry and make the lane pass"
    )
    extra = set(PROVIDER_DRIVERS) - registry - {"local"}
    assert not extra, f"conformance drivers without a registry row: {sorted(extra)}"
    prefixes = [prefix for prefix, _provider in PROVIDER_PREFIXES]
    assert len(prefixes) == len(set(prefixes)), "provider prefixes must be unambiguous"


_TARGET_REQUIRED_KEYS = {
    "provider", "resolved_model", "usage_model", "api_key", "base_url",
    "default_headers", "supports_openrouter_extensions", "supports_generation_cost",
}


@pytest.mark.parametrize("name", _REMOTE)
def test_route_resolution_has_the_required_form(name):
    driver = PROVIDER_DRIVERS[name]
    observed = _observe(driver.spec(
        [], call={"kind": "resolve_target", "model": driver.model}))
    target = observed["returned"]["value"]
    assert _TARGET_REQUIRED_KEYS <= set(target), sorted(target)
    assert target["provider"] == driver.provider
    assert target["resolved_model"] and target["usage_model"]


def test_usage_model_attribution_is_unambiguous_across_providers():
    """Two providers may never account the same bare model under one name."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="conformance-key")
    usage_models = {}
    for prefix, provider in PROVIDER_PREFIXES:
        target = client._resolve_remote_target(f"{prefix}model-x")
        usage_models.setdefault(target["usage_model"], provider)
    assert len(usage_models) == len(PROVIDER_PREFIXES), usage_models


@pytest.mark.parametrize("name", _DRIVERS)
def test_success_has_the_shared_response_and_ledger_shape(name):
    driver = PROVIDER_DRIVERS[name]
    observed = _observe(driver.spec([driver.success_step]))
    assert "raised" not in observed, observed.get("raised")
    returned = observed["returned"]
    message, usage = returned["message"], returned["usage"]

    assert isinstance(message, dict)
    assert message.get("content") is None or isinstance(message.get("content"), str)
    assert int(usage.get("prompt_tokens") or 0) >= 0
    assert int(usage.get("completion_tokens") or 0) >= 0
    # EVERY lane stamps usage provenance — the local asymmetry was CPL6-F1.
    assert usage.get("provider") == driver.provider
    assert usage.get("resolved_model")

    # Honest cost planes: the key always exists; cost_final is a bool and may
    # be True only for a KNOWN, non-estimated cost.
    assert "cost" in usage
    assert usage["cost"] is None or isinstance(usage["cost"], (int, float))
    assert isinstance(usage.get("cost_final"), bool)
    if usage["cost_final"]:
        assert usage["cost"] is not None and not usage.get("cost_estimated")

    # One successful send = one settled physical-attempt ledger row.
    attempts = observed.get("physical_attempts") or []
    assert len(attempts) == 1 and returned.get("ledger_attempt_count") == 1
    assert attempts[0]["provider"] == driver.provider
    assert attempts[0]["states"] == ["reserved", "dispatched", "settled"]
    assert observed["unused_script_steps"] == 0


@pytest.mark.parametrize("name", _DRIVERS)
def test_unknown_cost_is_never_fabricated(name):
    """With no provider-reported cost and no catalog price, the cost plane
    stays honest: None (or the local lane's contractual 0.0) — never a made-up
    number, never cost_final=True over an estimate."""
    driver = PROVIDER_DRIVERS[name]
    observed = _observe(driver.spec([driver.success_step]))
    usage = observed["returned"]["usage"]
    if driver.free_lane:
        assert usage["cost"] == 0.0 and usage["cost_final"] is True
    else:
        assert usage["cost"] is None
        assert usage["cost_final"] is False
        assert not usage.get("cost_estimated")


@pytest.mark.parametrize("name", _DRIVERS)
def test_transport_failure_raises_instead_of_fabricating_an_answer(name):
    driver = PROVIDER_DRIVERS[name]
    error_step = {"kind": "error", "message": "conformance: transport down"}
    observed = _observe(driver.spec([dict(error_step)] * 4))
    assert "returned" not in observed
    raised = observed.get("raised") or {}
    assert raised and raised["type"] != "AssertionError", raised
    assert len(observed["sends"]) >= 1


@pytest.mark.parametrize("name", _DRIVERS)
def test_typed_policy_refusal_is_permanent_one_send_only(name):
    """D09: a typed provider policy refusal is permanent by class — no lane
    may re-attempt the refused candidate, whatever its private retry ladder."""
    driver = PROVIDER_DRIVERS[name]
    refusal = {"kind": "error", "code": "provider_policy_refusal",
               "message": "policy: this request is not permitted"}
    observed = _observe(driver.spec([refusal, driver.success_step]))
    assert "returned" not in observed
    assert observed.get("raised"), "the refusal must surface, not vanish"
    assert len(observed["sends"]) == 1, "a refused candidate was re-attempted"
    assert observed["unused_script_steps"] == 1


@pytest.mark.parametrize("name", _DRIVERS)
def test_caller_timeout_reaches_the_transport(name):
    driver = PROVIDER_DRIVERS[name]
    observed = _observe(driver.spec(
        [driver.success_step], chat_kwargs={"timeout": 33.0}))
    assert "raised" not in observed, observed.get("raised")
    assert driver.timeout_of(observed["sends"][0]) == 33.0


@pytest.mark.parametrize(
    "name", [n for n in _DRIVERS if PROVIDER_DRIVERS[n].choice_shaped])
def test_null_finish_reason_is_surfaced_observably(name):
    """An incomplete response (finish_reason: null) must be visible in usage —
    the key is present with a null marker, distinguishable from an absent
    field and from a provider body error."""
    driver = PROVIDER_DRIVERS[name]
    observed = _observe(driver.spec([_openai_success(finish_reason=None)]))
    usage = observed["returned"]["usage"]
    assert "response_finish_reason" in usage and usage["response_finish_reason"] is None
    assert "provider_error" not in usage

    observed = _observe(driver.spec([_openai_success(finish_reason="stop")]))
    assert observed["returned"]["usage"]["response_finish_reason"] == "stop"


def _body_error_lanes() -> List[str]:
    """Lanes whose resolved target declares the OpenRouter body-error
    extension — derived from the route facts, not a hardcoded list."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="conformance-key")
    lanes = []
    for name in _REMOTE:
        target = client._resolve_remote_target(PROVIDER_DRIVERS[name].model)
        if target.get("supports_openrouter_extensions"):
            lanes.append(name)
    return lanes


@pytest.mark.parametrize("name", _body_error_lanes())
def test_http_200_rate_limit_body_is_a_typed_marker_not_a_blank_answer(name):
    driver = PROVIDER_DRIVERS[name]
    body_429 = {"kind": "response", "body": {
        "id": "gen-conformance-429", "choices": None,
        "error": {"code": 429, "message": "Provider returned error: rate limit"},
    }}
    observed = _observe(driver.spec([body_429]))
    usage = observed["returned"]["usage"]
    marker = usage.get("provider_error") or {}
    assert marker.get("kind") == "rate_limit" and marker.get("code") == 429
    assert usage["cost"] is None and usage["cost_final"] is False


# ===========================================================================
# 2. Executor-axis conformance (native | harness)
# ===========================================================================

from ouroboros import subagents  # noqa: E402
from ouroboros.loop_llm_call import SUBSCRIPTION_WINDOW_EXHAUSTED  # noqa: E402

_ROUTE = subagents.DelegationRoute("some-route", "model-x", "low")
_RESOLUTION_EXECUTORS = {"native", "harness", "blocked"}

# The closed outcome table of the executor axis: requested value ->
# route state -> (resolved executor, expected reason, blocked). Route states:
#   not_configured  — no OUROBOROS_SUBAGENT_HARNESS route at all;
#   ready           — a configured, healthy route;
#   unavailable     — the daemon/route reports itself unusable;
#   spent           — every subscription window of the route is exhausted.
EXECUTOR_MATRIX: Dict[str, Dict[str, tuple]] = {
    "native": {
        "not_configured": ("native", "requested_native", False),
        "ready": ("native", "requested_native", False),
        "unavailable": ("native", "requested_native", False),
        "spent": ("native", "requested_native", False),
    },
    "auto": {
        "not_configured": ("native", "harness_not_configured", False),
        "ready": ("harness", "harness_ready", False),
        "unavailable": ("native", "daemon_unreachable", False),
        "spent": ("native", SUBSCRIPTION_WINDOW_EXHAUSTED, False),
    },
    "harness": {
        "not_configured": ("blocked", "harness_not_configured", True),
        "ready": ("harness", "harness_ready", False),
        "unavailable": ("blocked", "daemon_unreachable", True),
        "spent": ("blocked", SUBSCRIPTION_WINDOW_EXHAUSTED, True),
    },
}

_ROUTE_STATES: Dict[str, Dict[str, Any]] = {
    "not_configured": {"route": None},
    "ready": {"route": _ROUTE},
    "unavailable": {"route": _ROUTE, "unavailable_reason": "daemon_unreachable"},
    "spent": {"route": _ROUTE, "reset_at": "2030-01-01T00:00:00Z"},
}


def test_executor_axis_conformance_derives_from_the_vocabulary():
    """Structural completeness: a new executor value added to the axis
    vocabulary without a conformance row here is RED."""
    assert set(EXECUTOR_MATRIX) == set(subagents.SUBAGENT_EXECUTORS), (
        "SUBAGENT_EXECUTORS and EXECUTOR_MATRIX drifted apart — a new axis "
        "point must ship with its conformance outcome row"
    )
    for requested, rows in EXECUTOR_MATRIX.items():
        assert set(rows) == set(_ROUTE_STATES), requested


@pytest.mark.parametrize("requested", sorted(EXECUTOR_MATRIX))
@pytest.mark.parametrize("state", sorted(_ROUTE_STATES))
def test_rule_table_outcome_is_typed_and_closed(requested, state):
    resolution = subagents.resolve_subagent_executor(requested, **_ROUTE_STATES[state])
    executor, reason, blocked = EXECUTOR_MATRIX[requested][state]
    assert resolution.executor in _RESOLUTION_EXECUTORS
    assert (resolution.executor, resolution.blocked) == (executor, blocked)
    assert resolution.reason == reason and resolution.reason, (requested, state)
    if reason == SUBSCRIPTION_WINDOW_EXHAUSTED:
        # Whenever exhaustion IS the surfaced fact, the reset instant rides
        # along so waiting stays a visible option (requested_native ignores
        # route state entirely and legitimately carries none).
        assert resolution.reset_at == "2030-01-01T00:00:00Z"


def test_unknown_executor_is_refused_typed_at_both_seams():
    with pytest.raises(ValueError, match="auto, harness, native"):
        subagents.normalize_subagent_executor("magic")
    with pytest.raises(ValueError):
        subagents.resolve_subagent_executor("magic")


def test_schema_vocabulary_and_rule_table_cover_the_same_axis():
    for value in subagents.SUBAGENT_EXECUTORS:
        assert subagents.normalize_subagent_executor(value) == value


def test_dispatch_native_point_never_contacts_the_daemon(monkeypatch):
    res = _dispatch(
        "native", monkeypatch=monkeypatch,
        raises=AssertionError("the native point must not construct a gateway"),
    )
    assert (res.executor, res.reason) == ("native", "requested_native")


def test_dispatch_stale_unknown_executor_degrades_to_auto_not_a_crash(monkeypatch):
    res = _dispatch("warp-drive", monkeypatch=monkeypatch)
    assert res.executor in ("native", "harness") and not res.blocked


def test_dispatch_plain_task_is_not_subject_to_the_axis(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak:low")
    res = subagents.dispatch_executor_resolution({})
    assert res.executor in ("native", "harness")


def test_harness_point_refusal_without_a_route_is_typed(tmp_path, monkeypatch):
    from ouroboros.tools.delegate import _delegate_start
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    payload = json.loads(_delegate_start(ctx, "do a thing"))
    assert payload["status"] == "refused"
    assert payload["reason"] == "subagent_selection_required"


def test_harness_point_launch_produces_a_run_identity_and_route_facts(tmp_path, monkeypatch):
    request, payload = _started_request(tmp_path, acting=False, monkeypatch=monkeypatch)
    assert payload["status"] == "started" and payload.get("run_id")
    assert request is not None
    wire = json.dumps(request)
    assert "some-route" in wire  # the configured route reaches the wire request


def test_delegation_artifact_round_trip_is_durable_and_honest(tmp_path, monkeypatch):
    """The last-delegation projection: requested vs applied facts stay
    separate (a mismatch is disclosable, never rewritten)."""
    import ouroboros.config as config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    subagents.record_last_delegation(
        route="some-route", requested_model="model-x", applied_model="",
        run_id="run-conformance", requested_profile="pin-a", applied_profile="",
    )
    record = subagents.subagent_last_delegation()
    assert record["route"] == "some-route"
    assert record["requested_model"] == "model-x"
    assert record["applied_model"] == ""  # never dressed up as the applied one
    assert record["run_id"] == "run-conformance"
