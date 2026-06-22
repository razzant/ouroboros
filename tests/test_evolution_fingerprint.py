"""C0 (BUG 3 SSOT): canonical_objective_fingerprint + _PLAN_REVIEW_SUFFIX hoist."""
from ouroboros.evolution_fingerprint import (
    _PLAN_REVIEW_SUFFIX,
    canonical_objective_fingerprint as fp,
)


def test_suffix_is_stripped_so_plan_review_framing_does_not_split_the_bucket():
    obj = "Add a pre-flight API key check at the start of commit_reviewed"
    assert fp(obj) == fp(obj + _PLAN_REVIEW_SUFFIX)


def test_whitespace_and_case_are_normalized():
    assert fp("Fix  X\n now") == fp("fix x now")
    assert fp("FIX X") == fp("fix x")


def test_distinct_objectives_get_distinct_fingerprints():
    assert fp("Add API key guard") != fp("Add a registration-map doc comment")


def test_empty_or_whitespace_objective_returns_empty_key():
    assert fp("") == ""
    assert fp("   \n  ") == ""
    assert fp(None) == ""
    # a bare suffix with no real objective text also collapses to the empty key
    assert fp(_PLAN_REVIEW_SUFFIX) == ""


def test_fingerprint_is_short_stable_hex():
    out = fp("some objective")
    assert isinstance(out, str) and len(out) == 16
    assert out == fp("some objective")  # deterministic
