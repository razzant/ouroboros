"""The one reply allowance of a prepared Main candidate (``context_budget.reply_allowance_tokens``).

Owner decisions 2026-10-05 (1A 2A 3A): the Nano target sizes the input; the reply follows
the KNOWN route window; an unknown window is stood in for by the owner's Nano target; the
slack on an approximate count is an eighth of the window; the reply never goes below the
floor (8,192, or the caller's own ceiling when that is smaller) and never above the ceiling.
"""

import math

import pytest

from ouroboros.context_budget import (
    NANO_MIN_HEADROOM_TOKENS,
    OWNER_NANO_TARGET_TOKENS,
    RECLAIM_LOW_WATER_DIVISOR,
    exact_reply_shortfall,
    reply_allowance_tokens,
)

C = 65_536
INCIDENT_DENSITY = 0.835284798376117  # the fresh route witness of the Codex Cowork incident (scout S2)


def _nano(input_tokens, *, raw=None, window=None, owner=True, ceiling=C, exact=False):
    return reply_allowance_tokens(caller_max_tokens=ceiling, nano=True, owner_nano=owner, input_tokens=input_tokens,
                                  raw_input_tokens=raw, window_tokens=window, exact=exact)


def test_the_incident_requests_get_the_whole_ceiling_on_their_million_token_window():
    raw = 84_677
    calibrated = math.ceil(raw * INCIDENT_DENSITY)
    assert calibrated == 70_730  # what Main recorded for round 30 of the incident
    for growth in range(0, 6 * 28, 28):  # the six retries each stacked one clock line (+28 estimated tokens)
        assert _nano(calibrated + growth, raw=raw + growth, window=1_000_000) == C


def test_the_known_window_branch_is_continuous_and_monotone():
    window = 128_000
    assert _nano(62_464, window=window) == 49_536
    assert _nano(62_465, window=window) == 49_535
    assert _nano(46_464, window=window) == C and _nano(46_465, window=window) == C - 1
    values = [_nano(tokens, window=window) for tokens in range(0, window + 10_000, 997)]
    assert values == sorted(values, reverse=True)
    assert min(values) == NANO_MIN_HEADROOM_TOKENS and max(values) == C


def test_a_cold_install_on_a_128k_window_gets_the_room_the_window_leaves():
    assert _nano(91_677, raw=91_677, window=128_000) == 20_323  # density 1.0: 128,000 - 91,677 - 16,000


def test_the_raw_estimate_admits_when_it_is_the_larger_one():
    assert _nano(92_000, raw=110_000, window=128_000) == NANO_MIN_HEADROOM_TOKENS  # 128,000 - 110,000 - 16,000 < 8,192
    assert _nano(92_000, raw=80_000, window=128_000) == 20_000  # the calibrated count is the larger one


def test_an_exact_count_needs_no_slack_and_may_raise_an_over_cautious_estimate():
    window = 128_000
    assert _nano(80_000, raw=110_000, window=window, exact=True) == 48_000
    assert _nano(80_000, raw=110_000, window=window) == NANO_MIN_HEADROOM_TOKENS
    assert _nano(119_808, window=window, exact=True) == NANO_MIN_HEADROOM_TOKENS  # exactly the floor left
    assert not exact_reply_shortfall(caller_max_tokens=C, nano=True, input_tokens=119_808, window_tokens=window)
    assert exact_reply_shortfall(caller_max_tokens=C, nano=True, input_tokens=119_809, window_tokens=window)
    assert not exact_reply_shortfall(caller_max_tokens=C, nano=True, input_tokens=200_000, window_tokens=None)


def test_an_unknown_window_lets_the_owner_nano_target_stand_in():
    assert _nano(70_730) == 14_270
    assert _nano(91_677) == NANO_MIN_HEADROOM_TOKENS
    assert _nano(OWNER_NANO_TARGET_TOKENS - C - 1) == C  # the target has room for the whole ceiling
    assert _nano(70_730, owner=False) == C  # a Nano the window chose answers to the window alone


@pytest.mark.parametrize("ceiling", [4_096, 2_048, 256])
def test_a_ceiling_below_the_floor_is_never_raised(ceiling):
    assert _nano(10, ceiling=ceiling) == ceiling
    assert _nano(10, window=128_000, ceiling=ceiling) == ceiling
    assert _nano(127_000, window=128_000, ceiling=ceiling) == ceiling
    assert _nano(127_000, window=128_000, ceiling=ceiling, exact=True) == ceiling
    assert exact_reply_shortfall(caller_max_tokens=ceiling, nano=True, input_tokens=128_000 - ceiling + 1,
                                 window_tokens=128_000)
    assert not exact_reply_shortfall(caller_max_tokens=ceiling, nano=True, input_tokens=128_000 - ceiling,
                                     window_tokens=128_000)


@pytest.mark.parametrize("window", [None, 128_000, 1_000_000])
@pytest.mark.parametrize("input_tokens,raw", [(0, 0), (70_730, 84_677), (127_000, 127_000), (300_000, 400_000)])
def test_low_and_max_always_get_the_caller_ceiling(window, input_tokens, raw):
    for exact in (False, True):
        assert reply_allowance_tokens(caller_max_tokens=C, nano=False, owner_nano=False, input_tokens=input_tokens,
                                      raw_input_tokens=raw, window_tokens=window, exact=exact) == C


def test_the_slack_is_an_eighth_of_the_route_window_not_of_the_target():
    window = 128_000
    slack = math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
    assert slack == 16_000  # the reclaim landing helper would say ceil(min(85,000, W) / 8) = 10,625
    assert _nano(60_000, window=window) == window - 60_000 - slack
