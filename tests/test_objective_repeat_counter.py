"""C2 (BUG 3): the per-fingerprint cumulative objective-repeat counter.

The decisive property the prior consecutive-streak design FAILED: an objective that recurs
NON-consecutively (interleaved with other no_op work) must still accumulate to the pause
threshold. Replays the /tmp/b3 campaign e09bb702 sequence.
"""
from ouroboros.evolution_fingerprint import canonical_objective_fingerprint as fp
from supervisor.evolution_lifecycle import (
    _bump_objective_repeat_count,
    _clear_objective_repeat_count,
)

# Representative per-cycle objectives from campaign e09bb702.
APIKEY = "Add a pre-flight ANTHROPIC_API_KEY check at the start of commit_reviewed"
REGMAP = "Add a registration-map doc comment in registry.py"
REGISTRY = "Add a blocked_objective registry to the evolution cycle entry point"
DEVMD = "Create an Adding a New Tool checklist section in DEVELOPMENT.md"


def _tx(objective):
    """A transaction shaped like begin_evolution_transaction's output (only objective_fp matters)."""
    return {"objective_fp": fp(objective)}


def _counts(campaign):
    return campaign.get("objective_repeat_counts") or {}


def _dropped(campaign):
    return campaign.get("dropped_objective_fps") or []


def test_replay_tmp_b3_reaches_threshold_3_on_nonconsecutive_recurrence():
    """The API-KEY objective recurs at cyc 5,7,10,11 with DIFFERENT objectives at 6,8,9.
    A consecutive-streak counter peaks at 2; the cumulative map must reach 3 at cyc 10."""
    c = {}  # a campaign with no counter field yet (setdefault must tolerate it)

    # cyc4 ABSORBED (before the loop window): clearing an absent fp is a safe no-op
    _clear_objective_repeat_count(c, _tx(APIKEY))
    # cyc5 no_op API-KEY
    _bump_objective_repeat_count(c, _tx(APIKEY))
    assert _counts(c)[fp(APIKEY)] == 1
    # cyc6 no_op REG-MAP (interleaved — must NOT reset the API-KEY tally)
    _bump_objective_repeat_count(c, _tx(REGMAP))
    # cyc7 no_op API-KEY
    _bump_objective_repeat_count(c, _tx(APIKEY))
    assert _counts(c)[fp(APIKEY)] == 2
    # cyc8 no_op REGISTRY, cyc9 no_op DEVMD (interleaved)
    _bump_objective_repeat_count(c, _tx(REGISTRY))
    _bump_objective_repeat_count(c, _tx(DEVMD))
    # cyc10 no_op API-KEY -> reaches 3, the gate threshold, BEFORE cyc11 is spent
    _bump_objective_repeat_count(c, _tx(APIKEY))
    assert _counts(c)[fp(APIKEY)] == 3, "cumulative counter must catch the non-consecutive loop"
    # the interleaved objectives each sit at 1 and never interfere
    assert _counts(c)[fp(REGMAP)] == 1
    assert _counts(c)[fp(REGISTRY)] == 1
    assert _counts(c)[fp(DEVMD)] == 1


def test_absorb_clears_only_the_absorbed_fingerprint():
    c = {}
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _bump_objective_repeat_count(c, _tx(REGMAP))
    _clear_objective_repeat_count(c, _tx(APIKEY))  # genuine absorb on API-KEY
    assert fp(APIKEY) not in _counts(c)            # its tally is gone
    assert _counts(c)[fp(REGMAP)] == 1             # the other objective is untouched


def test_success_immunity_a_different_objective_does_not_clear_the_loop():
    """A non-failing/absorbed cycle on a DIFFERENT objective must NOT reset the looping one."""
    c = {}
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _clear_objective_repeat_count(c, _tx(REGMAP))  # some other objective absorbs
    assert _counts(c)[fp(APIKEY)] == 2             # the loop tally survives


def test_setdefault_tolerates_a_campaign_persisted_before_the_field_existed():
    c = {"status": "active", "objective": APIKEY}  # no objective_repeat_counts key
    _bump_objective_repeat_count(c, _tx(APIKEY))   # must not KeyError
    assert _counts(c)[fp(APIKEY)] == 1


def test_empty_fingerprint_is_never_bucketed():
    c = {}
    _bump_objective_repeat_count(c, {"objective_fp": ""})  # tx-less / empty objective
    _bump_objective_repeat_count(c, {})                    # missing key entirely
    assert _counts(c) == {}
    assert _dropped(c) == []


def test_layer_b_bump_records_dropped_fp_deduped():
    c = {}
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _bump_objective_repeat_count(c, _tx(APIKEY))  # again -> still a single entry
    _bump_objective_repeat_count(c, _tx(REGMAP))
    assert _dropped(c).count(fp(APIKEY)) == 1     # deduped, not appended twice
    assert fp(REGMAP) in _dropped(c)


def test_layer_b_absorb_undrops_only_the_absorbed_fp():
    c = {}
    _bump_objective_repeat_count(c, _tx(APIKEY))
    _bump_objective_repeat_count(c, _tx(REGMAP))
    _clear_objective_repeat_count(c, _tx(APIKEY))  # API-KEY landed -> no longer do-not-re-propose
    assert fp(APIKEY) not in _dropped(c)
    assert fp(REGMAP) in _dropped(c)               # the other stays dropped
