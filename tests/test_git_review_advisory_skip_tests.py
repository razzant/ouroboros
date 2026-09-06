"""The advisory ``skip_tests`` parameter (v4.41.0).

Split verbatim out of ``tests/test_git_review_pipeline.py`` by theme. This
module owns when a commit may skip the test run and how that choice is
recorded, surfaced and constrained.
"""
import json
import os
import sys

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# Advisory skip_tests parameter (v4.41.0)
# ---------------------------------------------------------------------------

class TestAdvisorySkipTests:
    """Verify that advisory_pre_review runs tests before the SDK call and
    that skip_tests=True bypasses the test gate."""

    @pytest.fixture(autouse=True)
    def _explicit_enabled_advisory(self, monkeypatch):
        """Keep these unit tests independent of benchmark-authored process env."""
        monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
            "triad": [{
                "slot_id": "triad-test",
                "route": {"kind": "api_chat", "target_id": "openai/test"},
            }],
            "scope": [{
                "slot_id": "scope-test",
                "route": {"kind": "api_chat", "target_id": "openai/test"},
            }],
            "advisory": {
                "enabled": True,
                "route": {"kind": "api_chat", "target_id": "anthropic/test"},
            },
        }))

    def _make_advisory_ctx(self, tmp_path):
        """Minimal ToolContext-like mock for advisory handler tests."""
        from tests._shared import make_safe_mock_ctx
        fake_ctx = make_safe_mock_ctx(tmp_path, repo_dir=str(tmp_path))
        fake_ctx.task_id = "t-skiptest"
        return fake_ctx

    def _release_changed_files(self) -> str:
        return "\n".join([
            "M  VERSION",
            "M  pyproject.toml",
            "M  README.md",
            "M  docs/ARCHITECTURE.md",
        ])

    def test_tests_preflight_blocked_when_tests_fail(self, tmp_path, monkeypatch):
        """When tests fail and skip_tests=False, advisory returns
        status='tests_preflight_blocked' without calling the SDK."""
        import json as _json
        from ouroboros.tools import claude_advisory_review as adv

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *a, **kw: [])
        monkeypatch.setattr(adv, "_check_worktree_version_sync_shared", lambda *a, **kw: "")
        monkeypatch.setattr(adv, "compute_snapshot_hash", lambda *a, **kw: "hash-skip-test")
        monkeypatch.setattr(adv, "_get_changed_file_list", lambda *a, **kw: self._release_changed_files())
        monkeypatch.setattr(adv, "_release_metadata_preflight", lambda *a, **kw: None)

        # Simulate failing tests
        monkeypatch.setattr(adv, "_run_advisory_tests", lambda ctx: "FAILED: 3 failed, 10 passed")

        sdk_called = {"n": 0}
        def _fake_run_claude_advisory(*a, **kw):
            sdk_called["n"] += 1
            return [], "RESULT", "model", 100
        monkeypatch.setattr(adv, "_run_claude_advisory", _fake_run_claude_advisory)

        ctx = self._make_advisory_ctx(tmp_path)
        result_raw = adv._handle_advisory_pre_review(
            ctx, commit_message="test", skip_tests=False
        )
        result = _json.loads(result_raw)
        assert result["status"] == "tests_preflight_blocked"
        assert "TESTS_PREFLIGHT_BLOCKED" in result["message"]
        assert sdk_called["n"] == 0, "SDK should NOT be called when tests fail"

    def test_skip_tests_true_bypasses_test_gate(self, tmp_path, monkeypatch):
        """skip_tests=True skips the test gate and reaches the SDK call."""
        from ouroboros.tools import claude_advisory_review as adv

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *a, **kw: [])
        monkeypatch.setattr(adv, "_check_worktree_version_sync_shared", lambda *a, **kw: "")
        monkeypatch.setattr(adv, "compute_snapshot_hash", lambda *a, **kw: "hash-skip-test-2")
        monkeypatch.setattr(adv, "_get_changed_file_list", lambda *a, **kw: self._release_changed_files())
        monkeypatch.setattr(adv, "_release_metadata_preflight", lambda *a, **kw: None)

        # Even though tests "fail", skip_tests=True must bypass
        test_called = {"n": 0}
        def _fake_run_advisory_tests(ctx):
            test_called["n"] += 1
            return "FAILED: 1 failed"
        monkeypatch.setattr(adv, "_run_advisory_tests", _fake_run_advisory_tests)

        sdk_called = {"n": 0}
        def _fake_run_claude_advisory(*a, **kw):
            sdk_called["n"] += 1
            return [], "⚠️ ADVISORY_ERROR: fake error", "", 0
        monkeypatch.setattr(adv, "_run_claude_advisory", _fake_run_claude_advisory)

        ctx = self._make_advisory_ctx(tmp_path)
        adv._handle_advisory_pre_review(
            ctx, commit_message="test", skip_tests=True
        )
        assert test_called["n"] == 0, "_run_advisory_tests should not be called with skip_tests=True"
        assert sdk_called["n"] == 1, "SDK should be called when skip_tests=True"

    def test_passing_tests_proceed_to_sdk(self, tmp_path, monkeypatch):
        """When tests pass, advisory continues to the SDK call."""
        from ouroboros.tools import claude_advisory_review as adv

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *a, **kw: [])
        monkeypatch.setattr(adv, "_check_worktree_version_sync_shared", lambda *a, **kw: "")
        monkeypatch.setattr(adv, "compute_snapshot_hash", lambda *a, **kw: "hash-skip-test-3")
        monkeypatch.setattr(adv, "_get_changed_file_list", lambda *a, **kw: self._release_changed_files())
        monkeypatch.setattr(adv, "_release_metadata_preflight", lambda *a, **kw: None)

        monkeypatch.setattr(adv, "_run_advisory_tests", lambda ctx: None)  # tests pass

        sdk_called = {"n": 0}
        def _fake_run_claude_advisory(*a, **kw):
            sdk_called["n"] += 1
            return [], "⚠️ ADVISORY_ERROR: fake", "", 0
        monkeypatch.setattr(adv, "_run_claude_advisory", _fake_run_claude_advisory)

        ctx = self._make_advisory_ctx(tmp_path)
        adv._handle_advisory_pre_review(ctx, commit_message="test")
        assert sdk_called["n"] == 1, "SDK should be called when tests pass"

    def test_run_advisory_tests_respects_env_gate(self, tmp_path):
        """OUROBOROS_PRE_PUSH_TESTS=0 disables the test runner."""
        import os as _os
        from ouroboros.tools import claude_advisory_review as adv

        orig = _os.environ.get("OUROBOROS_PRE_PUSH_TESTS")
        try:
            _os.environ["OUROBOROS_PRE_PUSH_TESTS"] = "0"
            fake_ctx = type("C", (), {"repo_dir": str(tmp_path)})()
            result = adv._run_advisory_tests(fake_ctx)
            assert result is None, "Expected None when env gate disabled"
        finally:
            if orig is None:
                _os.environ.pop("OUROBOROS_PRE_PUSH_TESTS", None)
            else:
                _os.environ["OUROBOROS_PRE_PUSH_TESTS"] = orig

    def test_skip_tests_param_in_tool_schema(self):
        """advisory_pre_review tool schema must expose skip_tests parameter."""
        from ouroboros.tools.claude_advisory_review import get_tools
        tools = get_tools()
        advisory_tool = next(t for t in tools if t.name == "advisory_review")
        props = advisory_tool.schema["parameters"]["properties"]
        assert "skip_tests" in props, "skip_tests must be in advisory_pre_review schema"
        assert props["skip_tests"]["type"] == "boolean"

    def test_tests_preflight_blocked_persists_durable_record_and_review_status(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: _handle_advisory_pre_review with failing tests writes an
        AdvisoryRunRecord(status='tests_preflight_blocked'), and _handle_review_status
        surfaces it as non-fresh and the correct next-step guidance; after a hash
        mismatch (snapshot changes) it falls through to the stale path, not the
        tests-blocked path.
        """
        import json as _json
        from ouroboros.tools import claude_advisory_review as adv
        from ouroboros.review_state import load_state

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *a, **kw: [])
        monkeypatch.setattr(adv, "_check_worktree_version_sync_shared", lambda *a, **kw: "")
        monkeypatch.setattr(adv, "_get_changed_file_list", lambda *a, **kw: self._release_changed_files())
        monkeypatch.setattr(adv, "_release_metadata_preflight", lambda *a, **kw: None)

        call_count = {"n": 0}
        def _hash(repo_dir, commit_message, paths=None):
            call_count["n"] += 1
            return "snapshot-A" if call_count["n"] <= 4 else "snapshot-B"
        monkeypatch.setattr(adv, "compute_snapshot_hash", _hash)

        monkeypatch.setattr(adv, "_run_advisory_tests", lambda ctx: "FAILED: 2 tests")

        fake_ctx = type("C", (), {
            "repo_dir": str(tmp_path), "drive_root": tmp_path,
            "emit_progress_fn": lambda *a, **kw: None, "task_id": "t-e2e",
        })()

        # 1. Run advisory — tests fail
        result_raw = adv._handle_advisory_pre_review(fake_ctx, commit_message="test-commit")
        result = _json.loads(result_raw)
        assert result["status"] == "tests_preflight_blocked"

        # 2. Durable state must have the AdvisoryRunRecord
        state = load_state(tmp_path)
        matching = [r for r in state.advisory_runs if r.snapshot_hash == "snapshot-A"]
        assert len(matching) == 1
        assert matching[0].status == "tests_preflight_blocked"
        assert matching[0].commit_message == "test-commit"

        # 3. review_status must surface it (non-fresh + test-failure guidance)
        fake_ctx2 = type("C", (), {
            "repo_dir": str(tmp_path), "drive_root": tmp_path,
            "emit_progress_fn": lambda *a, **kw: None, "task_id": "t-e2e",
        })()
        status_raw = adv._handle_review_status(fake_ctx2)
        status = _json.loads(status_raw)
        assert status.get("repo_commit_ready") is False or status.get("repo_commit_ready") == "no"
        next_step = status.get("next_step", "")
        assert "test" in next_step.lower() or "skip_tests" in next_step.lower(), \
            f"Expected test-failure guidance in next_step, got: {next_step!r}"
        assert "Advisory is stale" not in next_step, \
            f"Fell through to generic stale message: {next_step!r}"

        # 4. After hash mismatch (snapshot-B), the next_step guidance must fall
        # to the stale/re-run path and NOT still say "fix failing tests" for
        # snapshot-A (that advice is only valid for the exact snapshot that failed).
        # hash_mismatch=True because tests_preflight_blocked is now in the status set.
        status_raw2 = adv._handle_review_status(fake_ctx2)
        status2 = _json.loads(status_raw2)
        next_step2 = status2.get("next_step", "")
        # The guidance must NOT still refer to the old tests_preflight_blocked path
        # after the snapshot changed — that block is now stale.
        # We accept "advisory is stale", "re-run", or similar stale-path messaging.
        # The _next_step_guidance tests_preflight_blocked branch fires only when
        # stale_from_edit=False AND hash matches — here hash diverged, so it won't.
        assert "advisory_review" in next_step2.lower() or "stale" in next_step2.lower() \
            or "re-run" in next_step2.lower() or "rerun" in next_step2.lower() \
            or "commit_reviewed" in next_step2.lower(), \
            f"Expected stale-path guidance after hash mismatch, got: {next_step2!r}"

    def test_next_step_guidance_tests_preflight_blocked(self):
        """_next_step_guidance must return a specific 'fix failing tests' message
        (not the generic stale-advisory fallback) when the latest advisory run
        has status='tests_preflight_blocked' and stale_from_edit=False."""
        from ouroboros.tools.claude_advisory_review import _next_step_guidance
        from ouroboros.review_state import AdvisoryRunRecord, AdvisoryReviewState

        latest = AdvisoryRunRecord(
            snapshot_hash="abc123",
            commit_message="test",
            status="tests_preflight_blocked",
            ts="2026-04-20T00:00:00Z",
            raw_result="⚠️ TESTS_PREFLIGHT_BLOCKED: 3 failed",
        )
        state = AdvisoryReviewState()
        guidance = _next_step_guidance(
            latest=latest,
            state=state,
            stale_from_edit=False,
            stale_from_edit_ts=None,
            open_obs=[],
            open_debts=[],
            effective_is_fresh=False,
        )
        assert "tests_preflight_blocked" not in guidance.lower() or "tests" in guidance.lower(), \
            "Guidance should reference test failures"
        assert "fix" in guidance.lower() or "pytest" in guidance.lower() or "tests" in guidance.lower(), \
            f"Expected test-failure guidance, got: {guidance!r}"
        # Must NOT be the generic stale-advisory fallback
        assert "Advisory is stale" not in guidance, \
            f"Fell through to generic stale message: {guidance!r}"
        assert "skip_tests" in guidance, \
            f"Guidance should mention skip_tests=True escape hatch: {guidance!r}"

    def test_next_step_guidance_stale_template_class_is_closed_by_the_projection(self):
        """The v6.74.5 stale-template class ("Last advisory run was blocked by
        SyntaxError" over a record from ANOTHER snapshot): the binding lives
        UPSTREAM — review_evidence's hash_mismatch sets stale_from_edit for a
        blocked record whose hash differs from the current tree, and the
        guidance then routes to the generic invalidated message, never
        asserting the problem class. Both production-reachable combinations
        are pinned here; the (record != current, stale_from_edit=False)
        combination is UNREACHABLE from build_review_projection by
        construction (find_by_hash matches exactly; a mismatching latest sets
        hash_mismatch)."""
        from ouroboros.review_state import AdvisoryReviewState, AdvisoryRunRecord
        from ouroboros.tools.claude_advisory_review import _next_step_guidance

        latest = AdvisoryRunRecord(
            snapshot_hash="abc123def4567890",
            commit_message="test",
            status="preflight_blocked",
            ts="2026-04-20T00:00:00Z",
            raw_result="SyntaxError: invalid syntax at foo.py:3",
            # H4 (capinv-447): the specific problem-class claim now requires the
            # typed cause; an untyped legacy record gets the generic wording.
            reason_kind="syntax",
        )
        state = AdvisoryReviewState()

        # Record from another snapshot -> the projection flags it stale
        # (hash_mismatch, review_evidence.py) -> generic message, no class.
        mismatched = _next_step_guidance(
            latest=latest, state=state,
            stale_from_edit=True, stale_from_edit_ts="now (hash mismatch)",
            open_obs=[], open_debts=[], effective_is_fresh=False,
        )
        assert "Last advisory run was blocked" not in mismatched, mismatched
        assert "SyntaxError" not in mismatched, mismatched
        assert "invalidated" in mismatched, mismatched
        assert "advisory_review" in mismatched, mismatched   # the actionable step

        # A record of the CURRENT snapshot keeps the specific, actionable claim.
        matched = _next_step_guidance(
            latest=latest, state=state,
            stale_from_edit=False, stale_from_edit_ts=None,
            open_obs=[], open_debts=[], effective_is_fresh=False,
        )
        assert "Last advisory run was blocked" in matched, matched
        assert "syntax preflight" in matched, matched

    def test_projection_flags_a_blocked_record_from_another_snapshot_as_stale(self, tmp_path):
        """The load-bearing upstream fact for the previous test: a
        preflight_blocked record whose hash differs from the CURRENT tree makes
        build_review_projection set stale_from_edit=True (hash_mismatch
        includes the blocked statuses), so the production guidance call
        (_handle_review_status passes projection fields verbatim) can never
        assert the recorded problem class over a tree the record never saw."""
        from ouroboros.review_evidence import build_review_projection
        from ouroboros.review_state import AdvisoryRunRecord, load_state, save_state
        from ouroboros.tools.claude_advisory_review import _next_step_guidance

        drive = tmp_path / "drive"
        (drive / "state").mkdir(parents=True)
        repo = tmp_path / "repo"
        repo.mkdir()
        state = load_state(drive)
        state.add_run(AdvisoryRunRecord(
            snapshot_hash="oldsnapshot00000",
            commit_message="test",
            status="preflight_blocked",
            ts="2026-04-20T00:00:00Z",
            raw_result="SyntaxError: invalid syntax at foo.py:3",
        ))
        save_state(drive, state)

        projection = build_review_projection(
            drive, repo_dir=repo,
            snapshot_hash_fn=lambda *_a, **_k: "newsnapshot11111",
        )
        assert projection["stale_from_edit"] is True
        guidance = _next_step_guidance(
            projection["guidance_run"], projection["state"],
            projection["stale_from_edit"], projection["stale_from_edit_ts"],
            projection["open_obligations"], projection["open_debts"],
            effective_is_fresh=projection["effective_is_fresh"],
        )
        assert "Last advisory run was blocked" not in guidance, guidance
        assert "SyntaxError" not in guidance, guidance
