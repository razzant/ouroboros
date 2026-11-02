"""Reviewer output contracts and the projection of a recorded run's failure.

What outlived the advisory pipeline (decision 3A retired it): the shared
predicate every reviewer's clean verdict is classified with, the two JSON
contracts and which of them may offer an all-clear, and how ``review_status``
names the cause of a recorded (legacy) run that did not parse.
"""
import pytest


class TestEmptyArrayIsVerifiedClean:
    """The shared predicate the review lanes classify with. A reviewer must not
    be able to opt out of the gate by emitting the sentinel word."""

    @pytest.mark.parametrize("raw,expected", [
        ("[]\nNO_FINDINGS", True),
        ("[]", True),
        ("[] NO_FINDINGS", True),          # sentinel must never make it worse
        ("[]\r\nNO_FINDINGS", True),        # CRLF
        ("[ ]\nNO_FINDINGS", True),
        ("[]\nNO_FINDINGS\nHope that helps!", False),
        ("```json\n[]\n```", True),
        ("```json\n[]\n```\nNO_FINDINGS", True),   # fencing model puts the sentinel after the fence
        ("```\n[]\n```\nNO_FINDINGS", True),
        ("```JSON\n[]\n```", True),               # tag case must not matter
        ("```json\n[]\n```\nI cannot review this", False),
        ("prose ```[]``` NO_FINDINGS", False),
        ("```json\n[1]\n```\nNO_FINDINGS", False),
        ("I cannot review this diff. NO_FINDINGS", False),
        ("NO_FINDINGS", False),
        ("I cannot review this. [] Please retry.", False),
        ("I cannot review this diff. []\nNO_FINDINGS", False),
        ("Everything checks out.\n[]\nNO_FINDINGS", False),
        ("[] NO_FINDINGS trailing prose", False),
        ('[{"item": "x", "verdict": "FAIL"}]\nNO_FINDINGS', False),
        ('[{"item": broken\nNO_FINDINGS', False),
        ("", False),
    ])
    def test_clean_verdict_requires_a_real_empty_array(self, raw, expected):
        from ouroboros.triad_review import empty_array_is_verified_clean
        assert empty_array_is_verified_clean(raw) is expected, repr(raw)


class TestReviewContractModes:
    """One findings-only contract for every seat asked one question (a packet change seat, a result
    seat, a skill seat): it offers the all-clear, and no second answer form exists beside it."""

    def test_the_findings_contract_offers_the_all_clear_and_no_matrix_form_exists(self):
        import ouroboros.triad_review as triad
        from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

        assert "NO_FINDINGS" in REVIEW_JSON_ARRAY_CONTRACT
        assert not hasattr(triad, "REVIEW_JSON_MATRIX_CONTRACT")
        assert not hasattr(triad, "parse_model_review_results")
