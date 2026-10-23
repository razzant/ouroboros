"""Route identity and physical admission for the ordinary call host."""
import time
from dataclasses import replace



def wait_for(predicate, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("timed out waiting for consumer evidence")


def test_route_reserve_survives_nano_floor():
    from ouroboros.context_budget import context_mode_limits
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest, PhysicalAttemptContext
    physical = PhysicalAttemptContext("owner_nano", "nano", "cold_estimate", "route", "round",
        None, 90000, False, False, 1.0)
    request = AttemptRequest("model", "fixture_live", max_completion_tokens=32000,
        prompt_tokens_bounded_estimate=80001, physical_context=physical, working_reserve_tokens=10000)
    assert main_context_fit_refusal(request)["free_tokens"] == -1
    assert main_context_fit_refusal(replace(request, physical_context=replace(
        physical, response_reserve_tokens=8192)))["free_tokens"] == -1
    assert main_context_fit_refusal(replace(request, prompt_tokens_bounded_estimate=79999)) is None
    assert main_context_fit_refusal(replace(request, working_reserve_tokens=None)) is None
    assert context_mode_limits("nano", "nano", 32000, working_reserve_tokens=20000)[1] == 20000


def test_route_facts_store_acceptance_without_raising_window(tmp_path):
    from ouroboros.capability_evidence import record_route_facts, probe
    from ouroboros.contracts.realtime import RouteFacts
    route = dict(provider="fixture_live", model="fixture::voice", base_url="wss://example.invalid/live")
    facts = RouteFacts(80000, 231828, 10000, "estimator", "fixture retention probe", "2026-10-21")
    record_route_facts(tmp_path, **route, facts=facts)
    observed = probe(tmp_path, **route, allow_fetch=False)
    assert observed.window_tokens == 80000
    assert observed.route_facts["accepted_input_tokens"] == 231828
    assert observed.route_facts["basis"] == "estimator"
    record_route_facts(tmp_path, **route, facts=RouteFacts())
    assert probe(tmp_path, **route, allow_fetch=False).window_tokens == 0


def test_metadata_does_not_recreate_process_local_call_authority():
    from ouroboros.realtime_turn import recovery_refusal
    task = {"metadata": {"execution_transport": {"kind": "live_conversation", "call_id": "call"}}}
    assert recovery_refusal(task, None) == "realtime_restart_required"
    assert recovery_refusal({"metadata": {}}, None) == ""


def test_window_record_keeps_retention_and_estimator_limits_beside_acceptance(tmp_path):
    from ouroboros.capability_evidence import record_route_facts
    from ouroboros.contracts.realtime import RouteFacts
    from ouroboros.context_window import record_observation, resolve_window

    for basis, window in (("provider", 92_000), ("estimator", 80_000)):
        evidence = record_route_facts(tmp_path, provider="fixture_live", model=basis,
            base_url="wss://example.invalid/live", facts=RouteFacts(window, 231_828, 10_000, basis, "retention"))
        record_observation(tmp_path, evidence.route_fp, accepted=250_000)
        record = resolve_window(evidence, 10_000, root=tmp_path)
        assert record.accepted == 250_000 and record.usable == window - 10_000
        # A refusal above a retention bound cannot expand that smaller measured room.
        record_observation(tmp_path, evidence.route_fp, refused=120_000)
        assert resolve_window(evidence, 10_000, root=tmp_path).usable == window - 10_000
        record_observation(tmp_path, evidence.route_fp, refused=70_000)
        assert resolve_window(evidence, 10_000, root=tmp_path).usable == 60_000


def test_accepted_route_lower_bound_can_grow_without_inventing_reply_room(tmp_path):
    from ouroboros.capability_evidence import record_route_facts
    from ouroboros.contracts.realtime import RouteFacts
    from ouroboros.context_window import record_observation, resolve_window

    evidence = record_route_facts(tmp_path, provider="fixture_live", model="accepted",
        base_url="wss://example.invalid/live", facts=RouteFacts(133_554, 133_554, 10_000,
            source="accepted input lower bound", acceptance_can_raise=True))
    record_observation(tmp_path, evidence.route_fp, accepted=150_000)
    record = resolve_window(evidence, 10_000, root=tmp_path)
    assert record.input.tokens == record.usable == 150_000
    assert record.total is None and record.total_equivalent == 160_000
