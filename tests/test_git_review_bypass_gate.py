"""The bypass path: tests still run, and the route/slot-aware bypass gate.

Split verbatim out of ``tests/test_git_review_pipeline.py`` by theme. This
module owns what happens when review is bypassed — the test run that must
still happen, and the slot/route conditions under which the gate admits a
bypass at all.
"""
import json
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from tests._git_review_pipeline_shared import (
    _get_git_review_cycle_module,
)


def _make_staged_repo(tmp_path):
    """Repo helper with one staged change so the stage cycle reaches the test gate."""
    from ouroboros.tools.registry import ToolContext
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "logs").mkdir(parents=True)
    (drive / "locks").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), capture_output=True)
    (repo / "dummy.txt").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "branch", "-M", "ouroboros"], cwd=str(repo), capture_output=True)
    # One uncommitted change so `git status --porcelain` is non-empty after the stage
    # cycle runs `git add -A` internally.
    (repo / "new_change.txt").write_text("something", encoding="utf-8")
    return ToolContext(repo_dir=repo, drive_root=drive)


class TestBypassPathTestsRun:
    """When skip_advisory_pre_review=True, _run_reviewed_stage_cycle must run
    _run_review_preflight_tests before the expensive triad + scope review.

    This covers the new gate introduced when refactoring the test runner into
    review_helpers._run_review_preflight_tests — previously only the advisory
    path (claude_advisory_review._run_advisory_tests) ran tests.
    """

    def _make_staged_repo(self, tmp_path):
        """Repo helper with one staged change so the stage cycle reaches the test gate."""
        return _make_staged_repo(tmp_path)

    def test_bypass_runs_preflight_tests_and_blocks_on_failure(self, tmp_path, monkeypatch):
        """skip_advisory_pre_review=True → _run_review_preflight_tests is called,
        and a test failure blocks with reason='tests_preflight_blocked' BEFORE
        the parallel triad+scope review runs."""
        from ouroboros.tools import git as git_mod

        ctx = self._make_staged_repo(tmp_path)

        # Freshness check is irrelevant when bypass is in effect — stub to None.
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 2 failed, 5 passed"

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="bypass test",
            commit_start=0.0,
            skip_advisory_pre_review=True,
        )

        assert called["preflight"] == 1, "preflight tests must run in the bypass path"
        assert called["parallel"] == 0, "triad+scope must NOT run when preflight fails"
        assert outcome["status"] == "blocked"
        assert outcome["block_reason"] == "tests_preflight_blocked"
        assert "TESTS_PREFLIGHT_BLOCKED" in outcome["message"]

    def test_failed_bypass_preflight_stales_bypass_record(self, tmp_path, monkeypatch):
        """A failed bypass attempt must not leave a fresh bypass snapshot."""
        from ouroboros.review_state import load_state
        from ouroboros.tools import git as git_mod

        ctx = self._make_staged_repo(tmp_path)
        called = {"parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            return "FAILED: 2 failed, 5 passed"

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="bypass stale test",
            commit_start=0.0,
            skip_advisory_pre_review=True,
        )

        assert outcome["block_reason"] == "tests_preflight_blocked"
        assert called["parallel"] == 0
        state = load_state(tmp_path / "drive")
        matching = [
            run for run in state.advisory_runs
            if run.commit_message == "bypass stale test"
        ]
        assert matching, "bypass attempt should still be durably auditable"
        assert all(run.status not in ("fresh", "bypassed", "skipped") for run in matching)

    def test_bypass_preflight_pass_proceeds_to_review(self, tmp_path, monkeypatch):
        """When preflight passes in the bypass path, control reaches the
        parallel review. The review itself is stubbed (no LLM calls)."""
        from ouroboros.tools import git as git_mod

        ctx = self._make_staged_repo(tmp_path)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return None  # tests pass

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        # _aggregate_review_verdict returns (blocked, msg, reason, findings, scope_advisory)
        def _fake_aggregate(*a, **kw):
            # Simulate a clean verdict so the review passes through.
            return False, "", "", [], []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_aggregate_review_verdict", _fake_aggregate)

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="bypass test-pass",
            commit_start=0.0,
            skip_advisory_pre_review=True,
        )

        assert called["preflight"] == 1, "preflight must run in the bypass path"
        assert called["parallel"] == 1, (
            "triad+scope must run when preflight passes in the bypass path"
        )
        # outcome["status"] depends on downstream stages (commit/push) — the
        # invariant tested here is that the preflight gate does not block.
        assert outcome.get("block_reason") != "tests_preflight_blocked"

    def test_advisory_paths_include_rename_sources(self, tmp_path, monkeypatch):
        """Advisory freshness must see the same rename/copy source paths as
        protected-path classification, not only git diff --name-only output."""
        from ouroboros.tools import git as git_mod
        from ouroboros.tools.registry import ToolContext

        repo = tmp_path / "repo"
        drive = tmp_path / "drive"
        repo.mkdir()
        (drive / "logs").mkdir(parents=True)
        (drive / "locks").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True, capture_output=True)
        (repo / "old_name.txt").write_text("same\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)
        (repo / "old_name.txt").rename(repo / "new_name.txt")

        captured = {}

        def _fake_freshness(ctx, commit_message, skip_advisory_pre_review=False, *, paths=None):
            captured["paths"] = list(paths or [])
            return "blocked for test"

        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", _fake_freshness)

        outcome = git_mod._run_reviewed_stage_cycle(
            ToolContext(repo_dir=repo, drive_root=drive),
            commit_message="rename advisory paths",
            commit_start=0.0,
        )

        assert outcome["block_reason"] == "no_advisory"
        assert {"old_name.txt", "new_name.txt"} <= set(captured["paths"])

    def test_non_bypass_path_does_not_run_preflight_here(self, tmp_path, monkeypatch):
        """Without skip_advisory_pre_review, the stage cycle must NOT run the
        preflight tests — the advisory side already ran them, and the commit
        gate relies on advisory freshness instead.

        IMPORTANT: must set ANTHROPIC_API_KEY to a non-empty sentinel — and
        clear the reviewer-slot/route envs so the route/slot-aware gate
        (``advisory_gate_unavailable()``) reads the default enabled api route
        — so the gate reports AVAILABLE. Without this, CI environments (which
        have no key) silently fall into the bypass path and make the preflight
        run, causing the assert-0 below to fail even though
        ``skip_advisory_pre_review=False``.
        """
        from ouroboros.tools import git as git_mod

        # Simulate "normal" (non-bypass) path: advisory key is present on the
        # default (legacy, enabled, api) advisory configuration.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.delenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", raising=False)

        ctx = self._make_staged_repo(tmp_path)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 1 failed"

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        def _fake_aggregate(*a, **kw):
            return False, "", "", [], []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_aggregate_review_verdict", _fake_aggregate)

        git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="normal flow",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["preflight"] == 0, (
            "preflight must only run in the bypass path (non-bypass defers to "
            "the advisory-side runner)"
        )
        assert called["parallel"] == 1, (
            "triad+scope must run as normal in the non-bypass path"
        )

    def test_no_anthropic_key_auto_bypass_runs_preflight(self, tmp_path, monkeypatch):
        """When ANTHROPIC_API_KEY is absent on the default api advisory route
        (no reviewer-slot / advisory-route envs), _run_review_preflight_tests
        must still run in _run_reviewed_stage_cycle even though
        skip_advisory_pre_review is False. This covers the missing-key
        auto-bypass path of the route/slot-aware gate
        (``advisory_gate_unavailable()``): an api route without its key cannot
        run the advisory, so the compensating preflight must."""
        from ouroboros.tools import git as git_mod

        ctx = self._make_staged_repo(tmp_path)

        # Ensure no Anthropic key in environment, on the default api route.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.delenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", raising=False)

        # Advisory freshness check passes (advisory recorded a bypass run externally).
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 1 test error"  # tests fail

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="no-key auto-bypass test",
            commit_start=0.0,
            skip_advisory_pre_review=False,  # explicit False — gate must trigger via missing key
        )

        assert called["preflight"] == 1, (
            "preflight must run when ANTHROPIC_API_KEY is absent, "
            "even with skip_advisory_pre_review=False"
        )
        assert called["parallel"] == 0, "triad+scope must NOT run when preflight fails"
        assert outcome["status"] == "blocked"
        assert outcome["block_reason"] == "tests_preflight_blocked"


class TestRouteSlotAwareBypassGate:
    """#123: the stage cycle's bypass decision is route/slot-aware
    (``advisory_gate_unavailable()``), no longer a bare ANTHROPIC_API_KEY probe.

    Pins the behavior matrix: a disabled advisory slot bypasses (so the
    compensating hermetic preflight RUNS — defect (a)); the keyless delegated
    route does NOT bypass (no duplicate preflight, no false "Advisory
    bypassed" line — defect (b)); a malformed slots/route config fails closed
    INTO the preflight; and the bench keyless-legacy contract (bypass +
    OUROBOROS_PRE_PUSH_TESTS=0 preflight no-op) is preserved.
    """

    @staticmethod
    def _stub_cycle(monkeypatch, git_mod, called, *, preflight_result):
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return preflight_result

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        def _fake_aggregate(*a, **kw):
            return False, "", "", [], []

        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_aggregate_review_verdict", _fake_aggregate)

    def test_slot_disabled_with_key_runs_compensating_preflight(self, tmp_path, monkeypatch):
        """Defect (a): advisory slot disabled + key present used to skip BOTH
        the advisory and the compensating hermetic pytest — triad+scope ran on
        untested code. A disabled slot is a bypass, so the preflight RUNS."""
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
        monkeypatch.delenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", raising=False)
        monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
            "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
            "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
            "advisory": {"enabled": False},
        }))
        called = {"preflight": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called,
                         preflight_result="FAILED: 1 failed")

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="slot-disabled bypass",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["preflight"] == 1, (
            "a disabled advisory slot must trigger the compensating preflight"
        )
        assert called["parallel"] == 0
        assert outcome["block_reason"] == "tests_preflight_blocked"

    def test_delegated_keyless_route_is_not_bypassed(self, tmp_path, monkeypatch):
        """Defect (b): the keyless delegated (agent_session) route needs no key
        — the advisory ran on the free route, so the stage cycle must NOT run a
        duplicate preflight and must NOT emit the false "Advisory bypassed"
        progress line.

        The shared session route is set explicitly: availability of the
        delegated route means "a session route RESOLVES", not merely "the kind
        is agent_session" — an unroutable slot is the bypass corner pinned by
        ``test_unroutable_session_slot_is_bypassed_into_preflight``."""
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.setenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", "agent_session")
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude")
        progress: list = []
        ctx.emit_progress_fn = progress.append
        called = {"preflight": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called,
                         preflight_result="FAILED: must never run")

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="delegated keyless",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["preflight"] == 0, (
            "the delegated keyless route is not a bypass — no duplicate preflight"
        )
        assert called["parallel"] == 1
        assert outcome.get("block_reason") != "tests_preflight_blocked"
        assert not any("Advisory bypassed" in str(line) for line in progress)

    def test_unroutable_session_slot_is_bypassed_into_preflight(self, tmp_path, monkeypatch):
        """Triad a4 follow-up to #123: an ENABLED agent_session advisory whose
        delegated route resolves NOWHERE (no row target, no shared
        review/subagent route) cannot run — ``run_delegated_review_session``
        refuses that exact state with ``ReviewRouteUnavailable``. The gate must
        treat it as a bypass and run the compensating preflight; otherwise a
        fresh audited bypass record (e.g. an explicit skip) would reach
        triad+scope with neither advisory nor tests. The key is present to
        prove the decision is route-resolution-driven, not key-driven."""
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.setenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", "agent_session")
        monkeypatch.delenv("OUROBOROS_REVIEW_SESSION_ROUTE", raising=False)
        monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
        called = {"preflight": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called,
                         preflight_result="FAILED: 1 failed")

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="unroutable session slot",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["preflight"] == 1, (
            "an unroutable delegated advisory is a bypass — the compensating "
            "preflight must run"
        )
        assert called["parallel"] == 0
        assert outcome["block_reason"] == "tests_preflight_blocked"

    def test_malformed_config_fails_closed_into_preflight(self, tmp_path, monkeypatch):
        """A malformed advisory route token must not escape as an exception:
        the gate fails CLOSED into the compensating preflight."""
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.setenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", "cursor")  # unknown token
        called = {"preflight": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called,
                         preflight_result="FAILED: 1 failed")

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="malformed route token",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["preflight"] == 1, "malformed config must fail closed into the preflight"
        assert called["parallel"] == 0
        assert outcome["block_reason"] == "tests_preflight_blocked"

    def test_bench_keyless_legacy_env_reaches_review_without_pytest(self, tmp_path, monkeypatch):
        """Bench contract (e1v2/CLB): no key, no slot/route envs,
        OUROBOROS_PRE_PUSH_TESTS=0 → the gate still bypasses (legacy api route,
        no key) AND the preflight no-ops INSIDE _run_review_preflight_tests, so
        the flow reaches parallel review with zero real pytest spawn."""
        from ouroboros import preflight_runner
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.delenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", raising=False)
        monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
        monkeypatch.setattr(_get_git_review_cycle_module(), "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"parallel": 0, "pytest": 0}

        def _no_pytest(*a, **kw):
            called["pytest"] += 1
            raise AssertionError("real pytest must not spawn under OUROBOROS_PRE_PUSH_TESTS=0")

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        # The REAL _run_review_preflight_tests stays in place: its env gate
        # must return before ever reaching the hermetic runner.
        monkeypatch.setattr(preflight_runner, "run_hermetic_pytest", _no_pytest)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(_get_git_review_cycle_module(), "_aggregate_review_verdict",
                            lambda *a, **kw: (False, "", "", [], []))

        outcome = git_mod._run_reviewed_stage_cycle(
            ctx,
            commit_message="bench keyless legacy env",
            commit_start=0.0,
            skip_advisory_pre_review=False,
        )

        assert called["pytest"] == 0, "zero real pytest spawn in the bench env"
        assert called["parallel"] == 1, "the flow must reach parallel review"
        assert outcome.get("block_reason") != "tests_preflight_blocked"
