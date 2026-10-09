"""Reviewed commits apply the test policy before the pool panel in the system repository.

The advisory bypass this module used to pin is gone with the advisory (decision 3A): the
deterministic checks and the tests run ahead of the review pool panel. The suite runs
before any commit to the body, a documentation-only diff included (owner answer A,
2026-10-08); ``skip_tests`` is the one exemption, and a managed resolution pays the suite
even then. Skipping the look does not waive the tests.
"""
import subprocess

import pytest


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


class TestEveryCommitRunsTheTests:
    @staticmethod
    def _stub_cycle(monkeypatch, git_mod, called, *, tests_result):
        def _fake_tests(ctx, **kw):
            called["tests"] += 1
            return tests_result

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, None, "", []

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_tests)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict", lambda *a, **kw: (False, "", "", [], []))

    @pytest.mark.parametrize("skip", [False, True])
    def test_a_failing_suite_blocks_before_the_panel(self, tmp_path, monkeypatch, skip):
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        called = {"tests": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called, tests_result="FAILED: 2 failed, 5 passed")
        outcome = git_mod._run_reviewed_stage_cycle(
            ctx, commit_message="failing suite", commit_start=0.0, skip_advisory_pre_review=skip)
        assert called == {"tests": 1, "parallel": 0}
        assert outcome["status"] == "blocked" and outcome["block_reason"] == "tests_preflight_blocked"
        assert "TESTS_PREFLIGHT_BLOCKED" in outcome["message"]

    @pytest.mark.parametrize("skip", [False, True])
    def test_a_passing_suite_reaches_the_panel(self, tmp_path, monkeypatch, skip):
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        called = {"tests": 0, "parallel": 0}
        self._stub_cycle(monkeypatch, git_mod, called, tests_result=None)
        outcome = git_mod._run_reviewed_stage_cycle(
            ctx, commit_message="passing suite", commit_start=0.0, skip_advisory_pre_review=skip)
        assert called == {"tests": 1, "parallel": 1}
        assert outcome.get("block_reason") != "tests_preflight_blocked"
        assert ctx._commit_preflight["status"] == ("skipped" if skip else "not_performed")

    def test_the_bench_env_reaches_review_without_pytest(self, tmp_path, monkeypatch):
        """Bench contract (e1v2/CLB): OUROBOROS_PRE_PUSH_TESTS=0 no-ops the preflight INSIDE
        _run_review_preflight_tests, so the flow reaches the panel with zero pytest spawn."""
        from ouroboros import preflight_runner
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
        called = {"parallel": 0, "pytest": 0}

        def _no_pytest(*a, **kw):
            called["pytest"] += 1
            raise AssertionError("real pytest must not spawn under OUROBOROS_PRE_PUSH_TESTS=0")

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, None, "", []

        monkeypatch.setattr(preflight_runner, "run_hermetic_pytest", _no_pytest)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict", lambda *a, **kw: (False, "", "", [], []))
        outcome = git_mod._run_reviewed_stage_cycle(ctx, commit_message="bench env", commit_start=0.0)
        assert called == {"parallel": 1, "pytest": 0}
        assert outcome.get("block_reason") != "tests_preflight_blocked"

    def test_rename_sources_reach_the_gate(self, tmp_path, monkeypatch):
        """The gate classifies the same rename/copy source paths as protected-path
        classification does, not only `git diff --name-only` output."""
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        repo = ctx.repo_dir
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "add"], cwd=str(repo), check=True, capture_output=True)
        (repo / "dummy.txt").rename(repo / "renamed.txt")
        captured = {}

        def _gate(ctx, commit_message, commit_start, *, classification_paths, **kw):
            captured["paths"] = list(classification_paths)
            return {"status": "blocked", "message": "blocked for test", "block_reason": "preflight"}

        monkeypatch.setattr(git_mod, "_preflight_and_tests_gate", _gate)
        outcome = git_mod._run_reviewed_stage_cycle(ctx, commit_message="rename", commit_start=0.0)
        assert outcome["block_reason"] == "preflight"
        assert {"dummy.txt", "renamed.txt"} <= set(captured["paths"])


class TestCommitReviewedLandsInTheSystemRepository:
    """commit_reviewed is the landing in Ouroboros's own body: any other root is
    refused before anything is staged or reviewed, and points at review_change."""

    def _state(self, repo):
        return (subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True).stdout,
                subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True).stdout)

    def test_another_root_is_refused_toward_review_change(self, tmp_path, monkeypatch):
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        before = self._state(ctx.repo_dir)

        def _no_review(*a, **kw):
            raise AssertionError("a refused root must not reach the review")

        monkeypatch.setattr(git_mod, "_run_reviewed_stage_cycle", _no_review)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _no_review)
        # The registered handler names this call's review record on every outcome; a
        # refusal before any staging, review or record is ID-less, even when an
        # earlier call's id still sits on the context.
        ctx._current_review_record_id = "rl-stale-from-an-earlier-call"
        for root in ("active_workspace", str(tmp_path / "project"), "user_files"):
            result = git_mod._commit_reviewed(ctx, commit_message="land the project", root=root)
            assert result.startswith("⚠️ TOOL_ARG_ERROR: commit_reviewed lands in the system repository")
            assert "review_change" in result and "ordinary git" in result
            assert "review_record_id" not in result and "rl-stale" not in result
        assert self._state(ctx.repo_dir) == before

    def test_the_dispatcher_delivers_the_refusal_for_both_names(self, tmp_path, monkeypatch):
        from ouroboros.tools import git as git_mod
        from ouroboros.tools.registry import ToolRegistry

        ctx = _make_staged_repo(tmp_path)
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
        monkeypatch.setattr(git_mod, "_run_reviewed_stage_cycle",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not review")))
        registry = ToolRegistry(ctx.repo_dir, ctx.drive_root)
        registry.set_context(ctx)
        for name in ("commit_reviewed", "vcs_commit_reviewed"):
            root_schema = registry.get_schema_by_name(name)["function"]["parameters"]["properties"]["root"]
            assert root_schema["enum"] == ["system_repo"]
            result = registry.execute(name, {"commit_message": "land the project", "root": "active_workspace"})
            assert "commit_reviewed lands in the system repository" in result and "review_change" in result

    def test_no_root_and_the_system_repo_keep_todays_path(self, tmp_path):
        from ouroboros.tools import git as git_mod

        ctx = _make_staged_repo(tmp_path)
        answers = {repr(kwargs): git_mod._commit_reviewed(ctx, commit_message="", **kwargs)
                   for kwargs in ({}, {"root": ""}, {"root": "system_repo"})}
        assert len(set(answers.values())) == 1, answers
        assert "commit_message must be non-empty" in next(iter(answers.values()))
