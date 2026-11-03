"""Main measures its window on its own conversation (``context_anchor``).

17.10.2026: a 900k release task borrowed the token density of another task's 97k helper
request, was told ``free 171,453`` while its previous request was already 874,829 tokens,
stopped folding and died on a window refusal. These pins keep the estimate this
conversation's own and keep a stated window honest against what the model accepted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_window_store(tmp_path, monkeypatch):
    from ouroboros import capability_evidence
    monkeypatch.setattr(capability_evidence, "canonical_evidence_root", lambda: tmp_path)

from ouroboros.context_anchor import (
    ACCEPTED_INPUTS_KEY, INPUT_ANCHOR_KEY, InputAnchor, main_input_anchor, record_input_anchor,
)


def _plan(*, model: str = "claudexor::codex/gpt-6-astra", provider: str = "claudexor", window: int = 872_000):
    from ouroboros.context_fit import ContextFitPlan, ContextFitProjection

    projection = lambda mode: ContextFitProjection(mode=mode, system_content_json=json.dumps(mode),  # noqa: E731
                                                   estimated_tokens=10, calibrated_tokens=10, calibration_ratio=1.0)
    return ContextFitPlan(
        core_sha256="a" * 64, preferred_mode="max", initial_mode="max", model=model, provider=provider,
        route_fp="route-account-1", status="confirmed", stale=False, window_tokens=window,
        output_reserve_tokens=65_536, user_content_json=json.dumps("go"),
        max_projection=projection("max"), low_projection=projection("low"))


def _settled(usage: dict, *, prompt: int, estimate: int, provider: str = "claudexor", cached: int = 0,
             substituted: list | None = None) -> dict:
    usage["_last_round_cache_usage"] = {"round": 7, "model": "gpt-6-astra", "provider": provider,
                                        "prompt_tokens": prompt, "cached_tokens": cached, "cache_write_tokens": 0}
    usage["_context_raw_input_tokens"] = estimate
    usage["_model_substitutions"] = substituted or []
    return usage


def test_an_accepted_request_becomes_the_anchor_and_raises_the_floor():
    plan, usage = _plan(), {}
    record_input_anchor(_settled(usage, prompt=874_829, estimate=795_458, cached=873_216), plan, round_idx=1481)
    anchor, floor = main_input_anchor(usage, plan, rewrite_epoch=0)
    assert anchor == InputAnchor(input_tokens=874_829, estimate_tokens=795_458, round=1481, rewritten=False)
    assert floor == 874_829
    # A later, smaller accepted request moves the anchor but never lowers the floor.
    record_input_anchor(_settled(usage, prompt=600_000, estimate=550_000), plan, round_idx=1490)
    anchor, floor = main_input_anchor(usage, plan, rewrite_epoch=0)
    assert anchor.input_tokens == 600_000 and floor == 874_829


def test_an_account_rotation_and_another_model_have_no_anchor():
    plan, usage = _plan(), {}
    record_input_anchor(_settled(usage, prompt=500_000, estimate=480_000), plan, round_idx=3)
    rotated = SimpleNamespace(model=plan.model, provider=plan.provider)  # a new route fingerprint, same model
    assert main_input_anchor(usage, rotated, rewrite_epoch=0) == (None, None)
    other = SimpleNamespace(model="anthropic/claude-opus-5.5", provider="openrouter")
    assert main_input_anchor(usage, other, rewrite_epoch=0) == (None, None)


def test_each_model_keeps_its_own_anchor_so_a_return_after_a_fallback_stays_measured():
    """Both directions: a fallback's acceptance is its own anchor and never erases the primary's."""
    primary, usage = _plan(), {}
    fallback = _plan(model="claudexor::codex/gpt-6.1-sol")
    record_input_anchor(_settled(usage, prompt=874_829, estimate=795_458), primary, round_idx=10)
    record_input_anchor(_settled(usage, prompt=120_000, estimate=100_000), fallback, round_idx=11)
    assert main_input_anchor(usage, primary, rewrite_epoch=0)[0].input_tokens == 874_829
    assert main_input_anchor(usage, fallback, rewrite_epoch=0)[0].input_tokens == 120_000
    assert len(usage[INPUT_ANCHOR_KEY]) == 2


def test_a_substituted_answer_or_unknown_cache_semantics_leave_the_anchor_unchanged():
    plan, usage = _plan(), {}
    record_input_anchor(_settled(usage, prompt=400_000, estimate=380_000), plan, round_idx=2)
    record_input_anchor(_settled(usage, prompt=900_000, estimate=380_000,
                                 substituted=[{"requested": "a", "observed": "b"}]), plan, round_idx=3)
    assert main_input_anchor(usage, plan, rewrite_epoch=0)[0].input_tokens == 400_000  # a substitute teaches nothing
    unknown = _plan(model="gigachat/max", provider="gigachat")
    record_input_anchor(_settled(usage, prompt=50_000, estimate=40_000, provider="gigachat", cached=30_000),
                        unknown, round_idx=4)
    assert main_input_anchor(usage, unknown, rewrite_epoch=0) == (None, None)
    record_input_anchor(_settled(usage, prompt=50_000, estimate=40_000, provider="gigachat"), unknown, round_idx=5)
    assert main_input_anchor(usage, unknown, rewrite_epoch=0)[0].input_tokens == 50_000  # no cache: a total


def test_a_count_that_cannot_be_this_requests_input_teaches_nothing():
    plan, usage = _plan(), {}
    record_input_anchor(_settled(usage, prompt=20, estimate=20_000), plan, round_idx=1)  # a delta or a stub
    assert main_input_anchor(usage, plan, rewrite_epoch=0) == (None, None)
    record_input_anchor(_settled(usage, prompt=31_000, estimate=20_000), plan, round_idx=2)  # a dense tokenizer
    assert main_input_anchor(usage, plan, rewrite_epoch=0)[0].input_tokens == 31_000


def test_a_gateway_that_counts_the_cache_read_twice_teaches_nothing():
    """Both directions: 2 x cached (+-1) is the gateway's double count, an honest cached row is a total."""
    plan, usage = _plan(model="openrouter::google/gemini-3.6-flash", provider="openrouter"), {}
    record_input_anchor(_settled(usage, prompt=2 * 300_000 - 1, estimate=310_000, provider="openrouter",
                                 cached=300_000), plan, round_idx=4)
    assert main_input_anchor(usage, plan, rewrite_epoch=0) == (None, None)  # no anchor, no inflated floor
    record_input_anchor(_settled(usage, prompt=320_000, estimate=310_000, provider="openrouter", cached=300_000),
                        plan, round_idx=5)
    assert main_input_anchor(usage, plan, rewrite_epoch=0)[0].input_tokens == 320_000


def test_a_rewrite_since_the_anchor_is_named():
    from ouroboros.transcript_prefix import rewrite_epoch, sanction_rewrite

    plan, usage, slot = _plan(), {}, SimpleNamespace()
    record_input_anchor(_settled(usage, prompt=300_000, estimate=290_000), plan, round_idx=9,
                        rewrite_epoch=rewrite_epoch(slot))
    assert main_input_anchor(usage, plan, rewrite_epoch=rewrite_epoch(slot))[0].rewritten is False
    sanction_rewrite(slot, "compaction")
    assert main_input_anchor(usage, plan, rewrite_epoch=rewrite_epoch(slot))[0].rewritten is True


def test_measurement_uses_this_conversations_anchor_before_other_witnesses(monkeypatch, tmp_path):
    """Without a usable witness the estimate stays cold; a task anchor always wins."""
    from ouroboros import capability_evidence
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit

    monkeypatch.setattr(capability_evidence, "resolve_main_token_density",
                        lambda *_a, **_kw: (1.0, "cold_estimate"))
    plan = _plan(window=1_000_000)
    messages = plan.messages_for("max")
    raw = estimate_context_prompt_tokens(messages, [], provider=plan.provider)
    cold = measure_main_fit(plan, messages, [], profile="owner_max", rendered_mode="max", round_id="r")
    assert cold.measurement.estimated_input_tokens == raw  # cold chars/4, the witness is never read
    assert cold.measurement.measurement_basis == "cold_estimate" and cold.measurement.anchor_input_tokens is None
    anchor = InputAnchor(input_tokens=874_829, estimate_tokens=795_458, round=1481)
    measured = measure_main_fit(plan, messages, [], profile="owner_max", rendered_mode="max", round_id="r",
                                anchor=anchor)
    assert measured.measurement.estimated_input_tokens == -(-raw * 874_829 // 795_458)
    assert measured.measurement.measurement_basis == "fresh_route_usage"
    assert (measured.measurement.anchor_input_tokens, measured.measurement.anchor_round) == (874_829, 1481)


def test_an_accepted_input_raises_a_known_planning_window():
    from ouroboros.context_fit import measure_main_fit

    plan = _plan(window=872_000)
    messages = plan.messages_for("max")
    above = measure_main_fit(plan, messages, [], profile="owner_max", rendered_mode="max", round_id="r",
                             accepted_input_tokens=903_762)
    assert above.measurement.capacity_total_tokens == 903_762 + 65_536
    assert above.measurement.window_reading_tokens == 872_000
    below = measure_main_fit(plan, messages, [], profile="owner_max", rendered_mode="max", round_id="r",
                             accepted_input_tokens=640_000)
    assert below.measurement.capacity_total_tokens == 872_000  # a floor under the window cancels nothing


def test_the_facts_line_states_where_the_size_comes_from():
    from ouroboros.loop_messages import context_facts_line

    usage = {"_context_prompt_estimate": 880_000, "_context_reply_allowance_tokens": 65_536,
             "_context_capacity_total_tokens": None, "_context_window_reading_tokens": 872_000,
             "_context_accepted_input_tokens": 903_762, "_context_anchor_input_tokens": 874_829,
             "_context_anchor_round": 1481, "_context_anchor_rewritten": False}
    ctx = SimpleNamespace(accumulated_usage=usage, active_context_mode="max", active_model="codex", messages=[],
                          tool_schemas=[], context_fit_plan=None, round_idx=1482, tools=None)
    line = context_facts_line(ctx, money={})
    assert "round 1481 accepted 874,829, +5,171 since" in line
    assert "window unknown (largest accepted input 903,762)" in line  # no bound plan in this fixture
    assert "free unknown" in line
    usage.update(_context_capacity_total_tokens=1_000_000, _context_window_reading_tokens=1_000_000)
    assert "free ~54,464" in context_facts_line(ctx, money={})
    usage.update(_context_anchor_input_tokens=None, _context_anchor_round=None)
    cold = context_facts_line(ctx, money={})
    assert "cold chars/4 estimate" in cold and "predicted free ~54,464" in cold
    assert ACCEPTED_INPUTS_KEY not in usage  # the line reads facts; it records nothing


@pytest.mark.parametrize("reserve", [None, 8_192, 65_536])
def test_acceptance_records_the_planned_reserve_separately_from_the_wire(tmp_path, reserve):
    from ouroboros import capability_evidence as ce

    plan, usage = _plan(), {"_context_reply_allowance_tokens": 1_234}
    if reserve is not None:
        usage["_context_response_reserve_tokens"] = reserve
    record_input_anchor(_settled(usage, prompt=100_000, estimate=100_000), plan, round_idx=1)
    [row] = ce._load(tmp_path)["main_windows"][plan.route_fp]["observations"]
    assert row["accepted"] == 100_000 and row["transmitted"] == 1_234
    assert row["reserve"] == (plan.output_reserve_tokens if reserve is None else reserve)
