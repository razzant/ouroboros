from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros.loop_model_call import _main_context_profile
from ouroboros.usage_ledger import UsageLedgerCorrupt, _validate_candidate_facts


def test_nano_uses_owner_nano_profile_and_rendered_mode():
    plan = SimpleNamespace(preferred_mode="nano")
    assert _main_context_profile(plan, "nano") == "owner_nano"


def _row(**context):
    # ``candidate_measurement_kind`` makes the validator read the row's candidate facts at all
    # (a row without it is a pre-feature legacy row and is never inspected).
    return {"candidate_measurement_kind": "opaque", "physical_context": {
        "profile": "owner_nano", "rendered_mode": "nano", "measurement_basis": "fresh_route_usage",
        "route_fp": "route", "round_id": "round", "target_total_tokens": None, "capacity_total_tokens": None,
        "context_target_miss": False, "automatic_pass_used": False, **context}}


def test_usage_ledger_accepts_nano_physical_context():
    _validate_candidate_facts(_row(), 1)  # a row written before the density travelled
    _validate_candidate_facts(_row(measurement_density=0.835284798376117), 1)
    _validate_candidate_facts(_row(measurement_density=None), 1)


@pytest.mark.parametrize("density", [0, -1.0, float("nan"), float("inf"), True, "1.0"])
def test_usage_ledger_rejects_an_impossible_measurement_density(density):
    with pytest.raises(UsageLedgerCorrupt):
        _validate_candidate_facts(_row(measurement_density=density), 1)


def test_the_physical_context_of_a_main_fit_carries_its_density():
    from ouroboros.context_fit import MainFitDisposition, MainFitMeasurement
    from ouroboros.loop_model_call import _physical_context_for_fit

    measurement = MainFitMeasurement(
        route_fp="route", round_id="x:round:1", profile="owner_nano", rendered_mode="nano",
        estimated_input_tokens=70_730, response_reserve_tokens=8_192, target_total_tokens=85_000,
        capacity_total_tokens=1_000_000, measurement_basis="fresh_route_usage",
        measurement_density=0.835284798376117, target_deficit_tokens=0, capacity_deficit_tokens=0,
        reclaim_goal_tokens=0)
    physical = _physical_context_for_fit(MainFitDisposition(measurement, "send", False, False))
    assert physical.measurement_density == 0.835284798376117
    _validate_candidate_facts({"candidate_measurement_kind": "opaque", "physical_context": asdict(physical)}, 1)


def test_the_mode_reaches_the_send_only_through_the_bound_physical_context():
    """No ``context_mode`` keyword travels the call chain: the bound PhysicalAttemptContext
    carries ``rendered_mode`` and the transport finalizer reads it from there."""
    import inspect

    from ouroboros.llm import LLMClient
    from ouroboros.llm_local import _LocalLaneMixin

    for function in (LLMClient.chat, LLMClient.chat_async, _LocalLaneMixin._chat_local, _LocalLaneMixin._build_local_candidate):
        assert "context_mode" not in inspect.signature(function).parameters, function.__qualname__
