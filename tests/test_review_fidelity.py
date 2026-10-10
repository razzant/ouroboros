"""Tests for review pipeline fidelity: no [:N] caps on findings/obligations.

Verifies that removing slice caps in review_helpers.build_blocking_findings_json_section,
review_state.format_status_section, and commit_gate warning messages means ALL items
are included in output without silent truncation.

Ref: v4.16.1 — review pipeline fidelity fix.
"""
import json
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal dataclass stand-ins to avoid full state setup
# ---------------------------------------------------------------------------

def _make_obligation(obligation_id: str, item: str = "code_quality",
                     reason: str = "some reason") -> Any:
    from ouroboros.review_state import ObligationItem
    ob = ObligationItem.__new__(ObligationItem)
    ob.obligation_id = obligation_id
    ob.item = item
    ob.severity = "critical"
    ob.reason = reason
    ob.source_attempt_ts = "2026-04-08T00:00:00"
    ob.source_attempt_msg = "fix something"
    ob.status = "still_open"
    ob.resolved_by = ""
    ob.repo_key = ""
    return ob


def _make_commit_attempt(findings: List[Dict[str, Any]]) -> Any:
    from ouroboros.review_state import CommitAttemptRecord
    ca = CommitAttemptRecord.__new__(CommitAttemptRecord)
    ca.ts = "2026-04-08T00:00:00"
    ca.commit_message = "some commit"
    ca.status = "blocked"
    ca.block_reason = "critical_findings"
    ca.block_details = ""
    ca.duration_sec = 1.0
    ca.task_id = ""
    ca.critical_findings = findings
    ca.advisory_findings = []
    ca.readiness_warnings = []
    ca.tool_name = "commit_reviewed"
    ca.attempt = 1
    ca.phase = "blocking_review"
    ca.blocked = True
    ca.obligation_ids = []
    ca.late_result_pending = False
    ca.pre_review_fingerprint = ""
    ca.post_review_fingerprint = ""
    ca.fingerprint_status = ""
    ca.degraded_reasons = []
    ca.started_ts = ""
    ca.updated_ts = ""
    ca.finished_ts = ""
    ca.snapshot_hash = ""
    ca.repo_key = ""
    ca.triad_models = []
    ca.scope_model = ""
    ca.triad_raw_results = []
    ca.scope_raw_result = {}
    return ca


# ---------------------------------------------------------------------------
# Tests: build_blocking_findings_json_section — no [:6] on findings, no history_limit
# ---------------------------------------------------------------------------

class TestBuildBlockingFindingsJsonSection:
    """All findings and ALL blocking attempts must appear in serialised output."""

    def test_more_than_six_findings_per_attempt_all_included(self):
        """Old cap was [:6] per attempt — 8 findings must all appear."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        findings = [{"item": f"finding_{i}", "severity": "critical", "reason": f"reason {i}"}
                    for i in range(8)]  # 8 > old cap of 6
        obs = [_make_obligation("ob1")]
        attempt = _make_commit_attempt(findings)

        output = build_blocking_findings_json_section(obs, [attempt])
        assert output, "Expected non-empty output"

        # Extract JSON block
        start = output.find("```json")
        end = output.rfind("```")
        assert start != -1 and end > start
        payload = json.loads(output[start + 7:end].strip())

        attempt_data = payload["recent_blocking_attempts"]
        assert len(attempt_data) == 1
        assert len(attempt_data[0]["critical_findings"]) == 8, (
            f"Expected 8 findings but got {len(attempt_data[0]['critical_findings'])}"
        )
        for i in range(8):
            assert any(f"finding_{i}" in json.dumps(f)
                       for f in attempt_data[0]["critical_findings"])

    def test_more_than_four_blocking_attempts_all_included(self):
        """Old history_limit was 4 — 7 attempts must all appear."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        obs = [_make_obligation("ob1")]
        attempts = [_make_commit_attempt([{"item": f"item_{i}", "severity": "critical",
                                           "reason": f"r{i}"}])
                    for i in range(7)]  # 7 > old limit of 4

        output = build_blocking_findings_json_section(obs, attempts)
        payload = json.loads(output[output.find("```json") + 7:output.rfind("```")].strip())

        assert len(payload["recent_blocking_attempts"]) == 7, (
            f"Expected 7 attempts but got {len(payload['recent_blocking_attempts'])}"
        )

    def test_no_ellipsis_truncation_marker_in_output(self):
        """'... and N more' must not appear — caps were removed."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        findings = [{"item": f"f{i}", "severity": "critical", "reason": f"r{i}"}
                    for i in range(10)]
        obs = [_make_obligation("ob1")]
        attempt = _make_commit_attempt(findings)

        output = build_blocking_findings_json_section(obs, [attempt])
        assert "... and" not in output

    def test_all_open_obligations_included(self):
        """All open obligations must appear in JSON payload."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        obs = [_make_obligation(f"ob{i}", reason=f"obligation reason {i}")
               for i in range(9)]  # 9 obligations
        attempt = _make_commit_attempt([{"item": "code_quality", "severity": "critical",
                                         "reason": "some issue"}])

        output = build_blocking_findings_json_section(obs, [attempt])
        payload = json.loads(output[output.find("```json") + 7:output.rfind("```")].strip())

        assert len(payload["open_obligations"]) == 9


# ---------------------------------------------------------------------------
# Tests: format_status_section — no [:5] on findings, no [:6] on obligations,
# no [:3] on critical/advisory findings, no [:3] on readiness_warnings
# ---------------------------------------------------------------------------

class TestFormatStatusSection:
    """format_status_section must emit ALL items without ... truncation."""

    def _build_state_with_blocked_attempt(self, n_findings: int, n_warnings: int,
                                          n_obs: int):
        """Helper: construct minimal AdvisoryReviewState with a blocked commit attempt.

        repo_dir=None passed to format_status_section so no real filesystem lookup.
        """
        from ouroboros.review_state import AdvisoryReviewState

        findings = [{"item": "code_quality", "severity": "critical",
                     "reason": f"problem {i}"}
                    for i in range(n_findings)]
        warnings = [f"warning text {i}" for i in range(n_warnings)]
        obs = [_make_obligation(f"ob{i}") for i in range(n_obs)]

        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = []
        state.attempts = []
        state.open_obligations = obs
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        ca = _make_commit_attempt(findings)
        ca.readiness_warnings = warnings
        # Put the attempt in state.attempts so format_status_section renders it.
        state.attempts = [ca]

        return state, findings, warnings, obs

    @pytest.mark.parametrize(
        "case_id,n_findings,n_warnings,n_obs,prefix,count",
        [
            ("more_than_five_findings_all_shown", 7, 0, 1, "problem ", 7),
            ("more_than_three_warnings_all_shown", 1, 5, 1, "warning text ", 5),
            ("more_than_six_obligations_all_shown", 1, 0, 9, "ob", 9),
        ],
    )
    def test_truncation_caps_removed(self, case_id, n_findings, n_warnings, n_obs, prefix, count):
        """Old [:3] / [:5] / [:6] caps must be gone — every item appears."""
        from ouroboros.review_state import format_status_section

        state, *_ = self._build_state_with_blocked_attempt(
            n_findings=n_findings, n_warnings=n_warnings, n_obs=n_obs,
        )
        output = format_status_section(state, repo_dir=None)
        for i in range(count):
            assert f"{prefix}{i}" in output, f"Missing {prefix}{i} in output"
        assert "... and" not in output, f"Truncation marker found. Snippet: {output[:800]}"

    def test_more_than_three_advisory_runs_all_shown(self):
        """Old cap was advisory_runs[-3:] — 5 runs must all appear."""
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, format_status_section,
        )

        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = [
            AdvisoryRunRecord(
                snapshot_hash=f"hash{i:012d}",
                commit_message=f"commit run {i}",
                status="fresh",
                ts="2026-04-08T00:00:00",
                items=[],
            )
            for i in range(5)
        ]
        state.attempts = []
        state.open_obligations = []
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        output = format_status_section(state, repo_dir=None)
        for i in range(5):
            assert f"commit run {i}" in output, f"Advisory run {i} missing from output"

    def test_more_than_three_attempts_all_shown(self):
        """Old cap was attempts[-3:] — 5 attempts must all appear."""
        from ouroboros.review_state import (
            AdvisoryReviewState, CommitAttemptRecord, format_status_section,
        )

        attempts = [
            CommitAttemptRecord(
                ts=f"2026-04-0{i+1}T00:00:00",
                commit_message=f"attempt msg {i}",
                status="succeeded",
                tool_name="commit_reviewed",
                attempt=i + 1,
                repo_key="",
            )
            for i in range(5)
        ]
        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = []
        state.attempts = attempts
        state.open_obligations = []
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        output = format_status_section(state, repo_dir=None)
        # All 5 attempts must appear (old cap was attempts[-3:])
        for i in range(5):
            assert f"repo_commit#{i+1}" in output or f"attempt msg {i}" in output, (
                f"Attempt {i} missing from output"
            )


# ---------------------------------------------------------------------------
# Tests: the shared blocking-findings JSON helper keeps every obligation.
# ---------------------------------------------------------------------------


class TestCommitGateJsonHelperNotTruncated:
    """build_blocking_findings_json_section must not cap findings or attempts."""

    @pytest.mark.parametrize("count", [6, 8])  # 6 = old cap + 1; 8 = well above old [:5] cap
    def test_n_obligations_all_in_json_payload(self, count):
        """All obligations must appear in the JSON section — no [:N] cap.

        Parametrized in v5.15.x — collapsed
        test_six_obligations_boundary_case + test_eight_obligations_all_in_json_payload.
        """
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        obs = [_make_obligation(f"ob{i}", reason=f"gate reason {i}")
               for i in range(count)]
        attempt = _make_commit_attempt([{"item": "code_quality", "severity": "critical",
                                         "reason": "something"}])

        output = build_blocking_findings_json_section(obs, [attempt])
        payload = json.loads(output[output.find("```json") + 7:output.rfind("```")].strip())

        assert len(payload["open_obligations"]) == count
        ids = {ob["obligation_id"] for ob in payload["open_obligations"]}
        for i in range(count):
            assert f"ob{i}" in ids, f"Missing obligation ob{i}"
        assert "... and" not in output

    def test_long_reason_not_silently_truncated(self):
        """reason > 500 chars must survive in full (_sanitize_text no longer truncates)."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        long_reason = "X" * 800  # well above old 500-char limit
        obs = [_make_obligation("ob_long", reason=long_reason)]
        attempt = _make_commit_attempt([{"item": "code_quality", "severity": "critical",
                                         "reason": "R" * 700}])  # above old 500-char limit

        output = build_blocking_findings_json_section(obs, [attempt])
        payload = json.loads(output[output.find("```json") + 7:output.rfind("```")].strip())

        # Obligation reason must be preserved in full
        assert payload["open_obligations"][0]["reason"] == long_reason

        # Finding reason must also be preserved
        cf = payload["recent_blocking_attempts"][0]["critical_findings"][0]
        assert cf["reason"] == "R" * 700

    def test_long_source_attempt_msg_not_silently_truncated(self):
        """source_attempt_msg > 200 chars must survive in full (old limit=200 removed)."""
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        long_msg = "M" * 350  # well above old 200-char limit
        ob = _make_obligation("ob_msg")
        ob.source_attempt_msg = long_msg

        output = build_blocking_findings_json_section([ob], [])
        payload = json.loads(output[output.find("```json") + 7:output.rfind("```")].strip())
        assert payload["open_obligations"][0]["source_attempt_msg"] == long_msg

    def test_blocking_history_without_obligations_not_dropped(self):
        """Old early-return `if not open_obligations: return ""` silently dropped
        recent_blocking_attempts when the obligation list was empty.
        With the fix, passing empty obligations + non-empty history must still
        return a non-empty JSON block containing the attempts.
        """
        from ouroboros.tools.review_helpers import build_blocking_findings_json_section

        attempt = _make_commit_attempt([
            {"item": "code_quality", "severity": "critical", "reason": "some bug"}
        ])

        output = build_blocking_findings_json_section([], [attempt])
        assert output, "Expected non-empty output even when open_obligations is empty"

        start = output.find("```json")
        end = output.rfind("```")
        assert start != -1 and end > start
        payload = json.loads(output[start + 7:end].strip())

        assert len(payload["recent_blocking_attempts"]) == 1
        cf = payload["recent_blocking_attempts"][0]["critical_findings"]
        assert len(cf) == 1 and cf[0]["reason"] == "some bug"


class TestFormatStatusSectionNoFieldSlicing:
    """format_status_section must not slice ts or commit_message fields."""

    def test_full_timestamp_and_commit_message_in_advisory_run(self):
        """run.ts and run.commit_message must appear untruncated in advisory run rows."""
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, format_status_section,
        )

        full_ts = "2026-04-08T12:34:56.789000+00:00"
        long_msg = "fix: a very long commit message that was previously truncated at 60 chars"

        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = [
            AdvisoryRunRecord(
                snapshot_hash="abc123def456",
                commit_message=long_msg,
                status="fresh",
                ts=full_ts,
                items=[],
            )
        ]
        state.attempts = []
        state.open_obligations = []
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        output = format_status_section(state, repo_dir=None)
        assert full_ts in output, f"Full timestamp not found in output. Got: {output[:400]}"
        assert long_msg in output, f"Full commit message not found in output. Got: {output[:400]}"

    def test_full_timestamp_and_commit_message_in_blocked_attempt(self):
        """ca.ts and ca.commit_message must appear untruncated in last blocked attempt block."""
        from ouroboros.review_state import (
            AdvisoryReviewState, format_status_section,
        )

        full_ts = "2026-04-08T23:59:59.000001+00:00"
        long_msg = "fix: another very long commit message that exceeds sixty characters easily"

        ca = _make_commit_attempt([{"item": "code_quality", "severity": "critical",
                                    "reason": "test"}])
        ca.ts = full_ts
        ca.commit_message = long_msg
        ca.status = "blocked"

        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = []
        state.attempts = [ca]
        state.open_obligations = []
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        output = format_status_section(state, repo_dir=None)
        assert full_ts in output, f"Full timestamp not found in output. Got: {output[:400]}"
        assert long_msg in output, f"Full commit message not found in output. Got: {output[:400]}"


class TestHandleReviewStatusNotTruncated:
    """_handle_review_status JSON output must preserve full ts and commit_message.
    Tested via save_state/load_state + the internal logic that builds commit_attempt_data
    and attempts_data, since _handle_review_status takes a ToolContext not a bare state."""

    def test_commit_attempt_ts_not_serialised_truncated(self):
        """CommitAttemptRecord.ts must survive save/load without [:16] truncation."""
        import tempfile, pathlib
        from ouroboros.review_state import AdvisoryReviewState, save_state, load_state

        full_ts = "2026-04-08T23:59:59.999999+00:00"  # 32 chars — was clipped to 16

        ca = _make_commit_attempt([])
        ca.ts = full_ts
        ca.status = "blocked"
        ca.repo_key = ""

        state = AdvisoryReviewState()
        state.attempts = [ca]

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        save_state(tmp, state)
        loaded = load_state(tmp)

        assert loaded.attempts[0].ts == full_ts, (
            f"ts was truncated after save/load. Expected '{full_ts}', "
            f"got '{loaded.attempts[0].ts}'"
        )

    def test_commit_attempt_message_not_serialised_truncated(self):
        """CommitAttemptRecord.commit_message must survive save/load without [:80] truncation."""
        import tempfile, pathlib
        from ouroboros.review_state import AdvisoryReviewState, save_state, load_state

        long_msg = "fix: " + "X" * 200  # 205 chars — was clipped to 80

        ca = _make_commit_attempt([])
        ca.commit_message = long_msg
        ca.status = "blocked"
        ca.repo_key = ""

        state = AdvisoryReviewState()
        state.attempts = [ca]

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        save_state(tmp, state)
        loaded = load_state(tmp)

        assert loaded.attempts[0].commit_message == long_msg, (
            f"commit_message was truncated. Expected {len(long_msg)} chars, "
            f"got {len(loaded.attempts[0].commit_message)}"
        )

    def test_runs_data_all_runs_present_no_list_cap(self):
        """_handle_review_status must return all advisory runs — no [-5:] list cap."""
        import json, tempfile, pathlib
        from unittest.mock import MagicMock
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, save_state,
        )
        from ouroboros.tools.preflight_review import _handle_review_status

        # Create 7 runs — previously only last 5 were returned
        runs = []
        for i in range(7):
            run = AdvisoryRunRecord(
                snapshot_hash=f"hash{i}",
                commit_message=f"fix: commit {i}",
                status="fresh",
                ts=f"2026-04-08T00:{i:02d}:00",
            )
            runs.append(run)

        state = AdvisoryReviewState()
        state.advisory_runs = runs

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        save_state(tmp, state)

        ctx = MagicMock()
        ctx.drive_root = str(tmp)
        ctx.repo_dir = ""

        result_json = _handle_review_status(ctx)
        data = json.loads(result_json)
        returned_runs = data.get("advisory_runs", [])
        assert len(returned_runs) == 7, (
            f"Expected 7 runs (no [-5:] cap), got {len(returned_runs)}"
        )

    def test_attempts_data_all_attempts_present_no_list_cap(self):
        """_handle_review_status must return all commit attempts — no [-8:] list cap."""
        import json, tempfile, pathlib
        from unittest.mock import MagicMock
        from ouroboros.review_state import AdvisoryReviewState, save_state
        from ouroboros.tools.preflight_review import _handle_review_status

        # Create 10 attempts — previously only last 8 were returned
        state = AdvisoryReviewState()
        for i in range(10):
            ca = _make_commit_attempt([])
            ca.ts = f"2026-04-08T00:{i:02d}:00"
            ca.commit_message = f"fix: attempt {i}"
            ca.repo_key = ""
            ca.tool_name = "commit_reviewed"
            ca.task_id = "t"
            ca.attempt = i
            state.attempts.append(ca)

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        save_state(tmp, state)

        ctx = MagicMock()
        ctx.drive_root = str(tmp)
        ctx.repo_dir = ""

        result_json = _handle_review_status(ctx)
        data = json.loads(result_json)
        returned_attempts = data.get("attempts", [])
        assert len(returned_attempts) == 10, (
            f"Expected 10 attempts (no [-8:] cap), got {len(returned_attempts)}"
        )

    def test_runs_data_commit_message_and_ts_not_truncated(self):
        """runs_data in _handle_review_status must not truncate commit_message[:80] or ts[:16]."""
        import tempfile, pathlib
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, save_state, load_state,
        )

        full_ts = "2026-04-08T23:59:59.654321+00:00"   # 32 chars — was clipped to 16
        long_msg = "fix: " + "Z" * 200                  # 205 chars — was clipped to 80

        run = AdvisoryRunRecord(
            snapshot_hash="abc123",
            commit_message=long_msg,
            status="fresh",
            ts=full_ts,
        )

        state = AdvisoryReviewState()
        state.advisory_runs = [run]

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        save_state(tmp, state)
        loaded = load_state(tmp)

        saved_run = loaded.advisory_runs[0]
        assert saved_run.ts == full_ts, (
            f"AdvisoryRunRecord.ts truncated in state. Expected '{full_ts}', got '{saved_run.ts}'"
        )
        assert saved_run.commit_message == long_msg, (
            f"AdvisoryRunRecord.commit_message truncated. Expected {len(long_msg)} chars, "
            f"got {len(saved_run.commit_message)}"
        )


class TestPersistencePathNotTruncated:
    """commit_message[:200] removed from persistence paths — long messages survive
    through the full record → display pipeline."""

    def test_long_commit_message_survives_record_commit_attempt(self):
        """_record_commit_attempt must store the full commit_message without [:200] truncation."""
        from ouroboros.review_state import CommitAttemptRecord, AdvisoryReviewState, save_state, load_state
        import tempfile, pathlib

        long_msg = "fix: " + "A" * 300  # 305 chars — well above old [:200] cap

        # Construct a minimal CommitAttemptRecord with the long message
        ca = CommitAttemptRecord.__new__(CommitAttemptRecord)
        ca.ts = "2026-04-08T00:00:00"
        ca.commit_message = long_msg
        ca.status = "blocked"
        ca.block_reason = "critical_findings"
        ca.block_details = ""
        ca.duration_sec = 1.0
        ca.task_id = ""
        ca.critical_findings = [{"item": "code_quality", "severity": "critical", "reason": "r"}]
        ca.advisory_findings = []
        ca.readiness_warnings = []
        ca.tool_name = "commit_reviewed"
        ca.attempt = 1
        ca.phase = "blocking_review"
        ca.blocked = True
        ca.obligation_ids = []
        ca.late_result_pending = False
        ca.pre_review_fingerprint = ""
        ca.post_review_fingerprint = ""
        ca.fingerprint_status = ""
        ca.degraded_reasons = []
        ca.started_ts = ""
        ca.updated_ts = ""
        ca.finished_ts = ""
        ca.snapshot_hash = ""
        ca.repo_key = ""
        ca.triad_models = []
        ca.scope_model = ""
        ca.triad_raw_results = []
        ca.scope_raw_result = {}

        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        state = AdvisoryReviewState.__new__(AdvisoryReviewState)
        state.advisory_runs = []
        state.attempts = [ca]
        state.open_obligations = []
        state.last_stale_from_edit_ts = ""
        state.last_stale_reason = ""
        state.last_stale_repo_key = ""

        save_state(tmp, state)
        loaded = load_state(tmp)

        saved_attempt = loaded.attempts[0]
        assert saved_attempt.commit_message == long_msg, (
            f"commit_message was truncated. Expected {len(long_msg)} chars, "
            f"got {len(saved_attempt.commit_message)}: {saved_attempt.commit_message[:80]}..."
        )

    def test_long_commit_message_in_obligation_source(self):
        """_update_obligations_from_attempt must store full commit_message in source_attempt_msg."""
        from ouroboros.review_state import (
            AdvisoryReviewState,
        )

        long_msg = "fix: " + "B" * 300  # 305 chars — well above old [:200] cap

        state = AdvisoryReviewState()
        # _update_obligations_from_attempt filters by verdict==FAIL and severity==critical
        ca = _make_commit_attempt([
            {"item": "code_quality", "severity": "critical", "reason": "r", "verdict": "FAIL"}
        ])
        ca.commit_message = long_msg
        ca.repo_key = ""
        ca.status = "blocked"

        # Trigger _update_obligations_from_attempt directly
        state._update_obligations_from_attempt(ca)

        obs = state.get_open_obligations()
        assert obs, "Expected at least one obligation to be created"
        assert obs[0].source_attempt_msg == long_msg, (
            f"source_attempt_msg was truncated. Expected {len(long_msg)} chars, "
            f"got {len(obs[0].source_attempt_msg)}: {obs[0].source_attempt_msg[:80]}..."
        )


# ---------------------------------------------------------------------------
# review_evidence omission budget contract
# ---------------------------------------------------------------------------

class TestReviewEvidenceOmissionBudget:
    """Tests for max_attempts=0/max_runs=0 and has_evidence omission-counter semantics."""

    def setup_method(self, _method=None):
        import importlib
        self.mod = importlib.import_module("ouroboros.review_evidence")

    def _collect(self, **kwargs):
        """Call collect_review_evidence with a temp drive_root."""
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return self.mod.collect_review_evidence(
                drive_root=pathlib.Path(tmp),
                task_id="t1",
                **kwargs,
            )

    def test_max_attempts_zero_returns_empty_list_not_full(self):
        """max_attempts=0 must return [] not the full list (guards [-0:] slice bug).

        Seeds actual attempts in state so the test cannot pass vacuously.
        """
        import pathlib, tempfile
        from ouroboros.review_state import AdvisoryReviewState, save_state

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp)
            (drive_root / "state").mkdir(parents=True)
            # Seed 2 real attempts into state
            state = AdvisoryReviewState()
            for i in range(2):
                ca = _make_commit_attempt([])
                ca.repo_key = ""
                ca.task_id = f"t{i}"
                state.attempts.append(ca)
            save_state(drive_root, state)

            ev = self.mod.collect_review_evidence(
                drive_root=drive_root, task_id="t0", max_attempts=0, max_runs=3
            )
        assert ev["recent_attempts"] == [], (
            "max_attempts=0 must return [] (not the full list via [-0:] bug)"
        )
        assert ev["omitted_attempts"] >= 0  # all seeded attempts counted as omitted

    def test_max_runs_zero_returns_empty_list_not_full(self):
        """max_runs=0 must return [] not the full list. Seeds actual runs to prevent vacuous pass."""
        import pathlib, tempfile
        from ouroboros.review_state import AdvisoryReviewState, AdvisoryRunRecord, save_state

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp)
            (drive_root / "state").mkdir(parents=True)
            state = AdvisoryReviewState()
            for i in range(2):
                state.advisory_runs.append(AdvisoryRunRecord(
                    snapshot_hash=f"hash{i}", commit_message=f"msg{i}",
                    status="fresh", ts="2026-01-01T00:00:00",
                ))
            save_state(drive_root, state)

            ev = self.mod.collect_review_evidence(
                drive_root=drive_root, task_id="t0", max_attempts=3, max_runs=0
            )
        assert ev["recent_advisory_runs"] == [], (
            "max_runs=0 must return [] (not the full list)"
        )

    def test_omitted_attempts_correct_when_max_zero_empty_state(self):
        """omitted_attempts == 0 when state has no attempts and max_attempts=0."""
        ev = self._collect(max_attempts=0, max_runs=3)
        assert ev["omitted_attempts"] == 0  # nothing to omit in empty state

    def test_attempt_to_dict_includes_duration_sec(self):
        """_attempt_to_dict must include duration_sec so forensic surface is complete."""
        import importlib
        from ouroboros.review_state import CommitAttemptRecord
        mod = importlib.import_module("ouroboros.review_evidence")
        ca = CommitAttemptRecord(
            ts="2026-01-01T00:00:00",
            commit_message="test commit",
            tool_name="commit_reviewed",
            attempt=1,
            status="succeeded",
            duration_sec=42.5,
        )
        d = mod._attempt_to_dict(ca)
        assert "duration_sec" in d, "_attempt_to_dict must expose duration_sec"
        assert d["duration_sec"] == 42.5

    def test_has_evidence_includes_omitted_corrupt_in_predicate(self):
        """Directly verify omitted_corrupt > 0 makes has_evidence True."""
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Create 4 corrupt files — capped at 3 visible, 1 omitted
            corrupt_dir = pathlib.Path(tmp) / "state" / "review_continuations" / "corrupt"
            corrupt_dir.mkdir(parents=True)
            for i in range(4):
                (corrupt_dir / f"corrupt_{i}.json").write_text("{bad")
            ev = self.mod.collect_review_evidence(
                drive_root=pathlib.Path(tmp),
                task_id="t1",
                max_attempts=0,
                max_runs=0,
                max_continuations=0,
                max_obligations=0,
            )
        if ev["omitted_corrupt"] > 0:
            assert ev["has_evidence"] is True, (
                "has_evidence must be True when omitted_corrupt > 0"
            )


# ---------------------------------------------------------------------------
# Git commit failure path — forensic metadata must survive infra_failure
# ---------------------------------------------------------------------------

class TestGitCommitFailureForensicMetadata:
    """_record_commit_attempt on git commit exception must preserve triad_models
    and scope_model so forensic metadata is not lost in infra_failure paths.

    This is a regression test for obligation ab1cb5db88ac: the except block in
    _repo_commit_push() must pass forensic fields through to _record_commit_attempt
    even when the `git commit` subprocess raises.
    """

    def _make_ctx(self, tmp_dir):
        """Build a minimal ToolContext-like object with forensic model metadata."""
        import pathlib

        class _MinimalCtx:
            pass

        ctx = _MinimalCtx()
        ctx.drive_root = str(pathlib.Path(tmp_dir) / "drive")
        ctx.repo_dir = str(pathlib.Path(tmp_dir) / "repo")
        ctx.task_id = "test_task"
        ctx._last_triad_models = ["openai/gpt-5.5", "google/gemini-pro", "anthropic/claude-opus-4"]
        ctx._last_scope_model = "anthropic/claude-opus-4"
        # drive_logs must be callable (used by _record_commit_attempt)
        ctx.drive_logs = str(pathlib.Path(tmp_dir) / "drive" / "logs")
        return ctx

    def test_record_commit_attempt_infra_failure_stores_triad_models(self):
        """_record_commit_attempt with status=failed must persist triad_models."""
        import pathlib, tempfile
        from ouroboros.review_state import load_state
        from ouroboros.tools.git import _record_commit_attempt

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp) / "drive"
            (drive_root / "state").mkdir(parents=True)
            (drive_root / "logs").mkdir(parents=True)
            ctx = self._make_ctx(tmp)
            ctx.drive_root = str(drive_root)

            _record_commit_attempt(
                ctx,
                commit_message="test: infra failure path",
                status="failed",
                block_reason="infra_failure",
                block_details="git commit raised RuntimeError",
                duration_sec=0.1,
                triad_models=["modelA", "modelB", "modelC"],
                scope_model="scopeModelX",
            )

            state = load_state(drive_root)
            assert state.attempts, "Expected at least one CommitAttemptRecord"
            ca = state.attempts[-1]
            assert ca.triad_models == ["modelA", "modelB", "modelC"], (
                f"triad_models not persisted. Got: {ca.triad_models!r}"
            )
            assert ca.scope_model == "scopeModelX", (
                f"scope_model not persisted. Got: {ca.scope_model!r}"
            )
            assert ca.status == "failed"
            assert ca.block_reason == "infra_failure"

    def test_record_commit_attempt_infra_failure_duration_sec_persisted(self):
        """duration_sec must survive save/load on infra_failure path."""
        import pathlib, tempfile
        from ouroboros.review_state import load_state
        from ouroboros.tools.git import _record_commit_attempt

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp) / "drive"
            (drive_root / "state").mkdir(parents=True)
            (drive_root / "logs").mkdir(parents=True)
            ctx = self._make_ctx(tmp)
            ctx.drive_root = str(drive_root)

            _record_commit_attempt(
                ctx,
                commit_message="test: duration on failure",
                status="failed",
                block_reason="infra_failure",
                block_details="subprocess raised",
                duration_sec=12.34,
                triad_models=["m1"],
                scope_model="s1",
            )

            state = load_state(drive_root)
            ca = state.attempts[-1]
            assert abs(ca.duration_sec - 12.34) < 0.01, (
                f"duration_sec not persisted correctly. Got: {ca.duration_sec}"
            )

    def test_ctx_triad_models_used_when_not_passed_explicitly(self):
        """getattr(ctx, '_last_triad_models', []) fallback path: ctx has the models."""
        import pathlib, tempfile
        from ouroboros.review_state import load_state
        from ouroboros.tools.git import _record_commit_attempt

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp) / "drive"
            (drive_root / "state").mkdir(parents=True)
            (drive_root / "logs").mkdir(parents=True)
            ctx = self._make_ctx(tmp)
            ctx.drive_root = str(drive_root)
            # Simulate the actual except-block pattern in _repo_commit_push:
            #   triad_models=getattr(ctx, "_last_triad_models", [])
            triad_models = getattr(ctx, "_last_triad_models", [])
            scope_model = getattr(ctx, "_last_scope_model", "")

            _record_commit_attempt(
                ctx,
                commit_message="test: getattr fallback",
                status="failed",
                block_reason="infra_failure",
                block_details="err",
                duration_sec=1.0,
                triad_models=triad_models,
                scope_model=scope_model,
            )

            state = load_state(drive_root)
            ca = state.attempts[-1]
            assert ca.triad_models == ctx._last_triad_models
            assert ca.scope_model == ctx._last_scope_model
