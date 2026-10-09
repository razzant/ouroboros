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


class TestReviewRunFailureReason:
    """`review_status` used to drop the per-run cause, so N identical
    deterministic failures read as N generic `parse_failure` rows."""

    def _run(self, status="parse_failure", raw=""):
        from types import SimpleNamespace
        return SimpleNamespace(status=status, raw_result=raw, items=[], snapshot_hash="h",
                               commit_message="m", ts="t", snapshot_summary="s", attempt=1,
                               bypass_reason="", repo_key="", tool_name="", task_id="",
                               model_used="opus", duration_sec=48.35, prompt_chars=786401)

    def test_rejected_clean_sentinel_is_named(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(self._run(raw="[]\nNO_FINDINGS"))
        assert data["failure_reason"] == "clean_sentinel_rejected"

    def test_sentinel_bearing_prose_is_not_called_a_rejected_clean_verdict(self):
        """The diagnostic asks the shared predicate, so refusal prose carrying
        the sentinel is reported as prose — not as a contract regression."""
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(
            self._run(raw="I cannot review this diff. []\nNO_FINDINGS"))
        assert data["failure_reason"] == "non_json_prose"

    def test_shapes_are_distinguished(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        assert _review_status_run_to_dict(self._run(raw=""))["failure_reason"] == "empty_response"
        assert _review_status_run_to_dict(self._run(raw="[{bad"))["failure_reason"] == "malformed_array"
        assert _review_status_run_to_dict(self._run(raw="sorry"))["failure_reason"] == "non_json_prose"

    def test_fresh_run_has_no_failure_reason_but_keeps_diagnostics(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(self._run(status="fresh", raw="[]"))
        assert data["failure_reason"] is None
        assert not any("raw" in k and k != "failure_reason" for k in data), (
            "the projection must not echo untrusted reviewer text to the model")
        assert data["duration_sec"] == 48.35
        assert data["model_used"] == "opus"
        assert data["prompt_chars"] == 786401


class TestReviewContractModes:
    """Findings-only mode offers an all-clear; required-matrix mode must not,
    because its parser rejects an empty array as missing every row."""

    def test_matrix_contract_has_no_all_clear_branch(self):
        from ouroboros.triad_review import (
            REVIEW_JSON_ARRAY_CONTRACT, REVIEW_JSON_MATRIX_CONTRACT,
        )
        assert "NO_FINDINGS" in REVIEW_JSON_ARRAY_CONTRACT
        assert "NO_FINDINGS" not in REVIEW_JSON_MATRIX_CONTRACT
        assert "one entry per required checklist item" in REVIEW_JSON_MATRIX_CONTRACT
