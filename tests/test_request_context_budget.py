"""One measured context frame: the memory allowance with and without working room.

``context_budget.request_context_budget`` is the single capacity calculation the
memory view's physical floor uses, and ``context_mode_limits`` names the owner
target and reply reserve of a rendered mode. Every rule is pinned in both
directions: it binds where it must and leaves the frame alone where it must not.
"""
from __future__ import annotations

import math

import pytest

from ouroboros import context_budget as cb


def _frame(**overrides):
    args = {"window_tokens": 872_000, "output_reserve_tokens": 65_536, "non_memory_tokens": 300_000}
    args.update(overrides)
    return cb.request_context_budget(**args)


def test_owner_targets_and_working_margins_are_the_ssot_values():
    assert cb.OWNER_LOW_TARGET_TOKENS == 250_000
    assert cb.OWNER_NANO_TARGET_TOKENS == 85_000
    assert cb.MEMORY_VIEW_WORKING_MARGINS == 2
    assert cb.MODE_TARGET_WORKING_MARGINS == 1
    # One level under the owner's targets, the full reply reserve (the arithmetic in the comment).
    low = cb.request_context_budget(window_tokens=1_050_000, output_reserve_tokens=65_536, non_memory_tokens=0,
                                    target_tokens=cb.OWNER_LOW_TARGET_TOKENS,
                                    margin_count=cb.MODE_TARGET_WORKING_MARGINS)
    nano = cb.request_context_budget(window_tokens=1_050_000, output_reserve_tokens=cb.NANO_MIN_HEADROOM_TOKENS,
                                     non_memory_tokens=0, target_tokens=cb.OWNER_NANO_TARGET_TOKENS,
                                     margin_count=cb.MODE_TARGET_WORKING_MARGINS)
    assert low["with_margin_tokens"] == 153_214 and low["without_margin_tokens"] == 184_464
    assert nano["with_margin_tokens"] == 66_183 and nano["without_margin_tokens"] == 76_808


@pytest.mark.parametrize("margin_count", [0, 1, 2])
def test_the_margins_differ_by_exactly_margin_count_low_water_levels(margin_count):
    frame = _frame(margin_count=margin_count)
    level = math.ceil(872_000 / cb.RECLAIM_LOW_WATER_DIVISOR)
    assert frame["working_margin_tokens"] == margin_count * level
    assert frame["without_margin_tokens"] == 872_000 - 65_536 - 300_000
    assert frame["without_margin_tokens"] - frame["with_margin_tokens"] == margin_count * level
    if margin_count:
        assert frame["with_margin_tokens"] < frame["without_margin_tokens"]
    else:
        assert frame["with_margin_tokens"] == frame["without_margin_tokens"]


def test_the_margin_reads_the_divisor_at_call_time(monkeypatch):
    before = _frame(margin_count=2)["working_margin_tokens"]
    monkeypatch.setattr(cb, "RECLAIM_LOW_WATER_DIVISOR", 4)
    after = _frame(margin_count=2)["working_margin_tokens"]
    assert before == 2 * math.ceil(872_000 / 8) and after == 2 * math.ceil(872_000 / 4)


def test_the_boundary_is_the_smaller_known_of_target_and_window():
    assert _frame(target_tokens=250_000)["boundary_tokens"] == 250_000  # the target binds
    assert _frame(window_tokens=200_000, target_tokens=250_000)["boundary_tokens"] == 200_000  # the window binds
    assert _frame(window_tokens=None, target_tokens=250_000)["boundary_tokens"] == 250_000  # target alone
    assert _frame()["boundary_tokens"] == 872_000  # window alone
    assert _frame(window_tokens=0)["boundary_tokens"] is None  # a zero window is not a known window


def test_an_unknown_frame_allows_everything_and_never_a_known_zero():
    frame = _frame(window_tokens=None)
    for key in ("boundary_tokens", "free_tokens", "with_margin_tokens", "without_margin_tokens",
                "target_deficit_tokens", "capacity_deficit_tokens"):
        assert frame[key] is None, key
    assert frame["working_margin_tokens"] == 0
    known = _frame()
    assert known["with_margin_tokens"] is not None and known["without_margin_tokens"] is not None


def test_calibration_scales_the_measured_input_both_ways():
    plain = _frame(margin_count=0)
    heavier = _frame(margin_count=0, calibration_ratio=1.25)
    lighter = _frame(margin_count=0, calibration_ratio=0.8)
    assert heavier["calibrated_input_tokens"] == 375_000 and lighter["calibrated_input_tokens"] == 240_000
    # Allowances stay in estimator tokens: floor((B - R) / ratio) - non_memory.
    assert heavier["without_margin_tokens"] == math.floor((872_000 - 65_536) / 1.25) - 300_000
    assert lighter["without_margin_tokens"] == math.floor((872_000 - 65_536) / 0.8) - 300_000
    assert heavier["without_margin_tokens"] < plain["without_margin_tokens"] < lighter["without_margin_tokens"]
    # A missing or zero ratio is the neutral 1.0, never a division by zero.
    assert _frame(margin_count=0, calibration_ratio=0)["without_margin_tokens"] == plain["without_margin_tokens"]


def test_an_allowance_never_goes_negative_and_free_space_can():
    frame = _frame(window_tokens=200_000, non_memory_tokens=300_000)
    assert frame["with_margin_tokens"] == 0 and frame["without_margin_tokens"] == 0
    assert frame["free_tokens"] == 200_000 - 65_536 - 300_000
    assert frame["capacity_deficit_tokens"] == 300_000 + 65_536 - 200_000


def test_deficits_are_measured_against_each_known_bound():
    frame = _frame(window_tokens=500_000, target_tokens=250_000, non_memory_tokens=200_000)
    assert frame["target_deficit_tokens"] == 200_000 + 65_536 - 250_000
    assert frame["capacity_deficit_tokens"] == 0
    assert _frame(non_memory_tokens=100_000)["target_deficit_tokens"] is None  # no target, no target deficit


def test_unknown_reply_space_is_an_optimistic_estimate_marked_unknown():
    frame = _frame(output_reserve_tokens=None, margin_count=0)
    assert frame["reserve_known"] is False
    assert frame["without_margin_tokens"] == 872_000 - 300_000
    assert _frame()["reserve_known"] is True


@pytest.mark.parametrize("mode,owner_mode,expected", [
    ("low", "low", (250_000, 65_536)),  # owner Low carries its target
    ("low", "max", (None, 65_536)),  # task-local Low keeps Max's window
    ("nano", "nano", (85_000, 8_192)),  # Nano: its target and its own minimum headroom
    ("nano", "max", (None, 8_192)),  # a Nano the window chose: its headroom, no owner target
    ("nano", "low", (None, 8_192)),
    ("max", "max", (None, 65_536)),
])
def test_context_mode_limits(mode, owner_mode, expected):
    assert cb.context_mode_limits(mode, owner_mode, 65_536) == expected


def test_nano_reserve_is_the_reply_floor_never_above_the_callers_ceiling():
    # A local lane's quarter window (4,096 on 16K) is its own ceiling: the floor and the send agree on it.
    assert cb.context_mode_limits("nano", "nano", 4_096) == (85_000, 4_096)
    assert cb.context_mode_limits("nano", "max", 2_048) == (None, 2_048)
    assert cb.context_mode_limits("low", "low", 4_096) == (250_000, 4_096)
