"""Tests for the review readiness gate (cheap deterministic pre-advisory checks)."""

import subprocess
from unittest.mock import patch, MagicMock


from ouroboros.tools.review_helpers import (
    check_worktree_readiness,
    check_worktree_version_sync,
)


class TestCheckWorktreeReadiness:
    """Tests for check_worktree_readiness()."""

    def test_clean_worktree_returns_no_changes_warning(self, tmp_path):
        """A clean git worktree should produce a 'no changes' warning."""
        with patch("ouroboros.tools.review_helpers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            warnings = check_worktree_readiness(tmp_path)
            assert any("no uncommitted changes" in w.lower() for w in warnings)

    def test_dirty_worktree_no_warnings(self, tmp_path):
        """A worktree with changes and no obvious issues should return empty."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = " M README.md\n"
            elif "diff" in cmd:
                result.stdout = "some small diff"
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            assert warnings == []

    def test_py_without_tests_warning(self, tmp_path):
        """Modified .py in ouroboros/ without test changes should warn."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = " M ouroboros/loop.py\n"
            elif "diff" in cmd:
                result.stdout = "small diff"
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            assert any("test" in w.lower() for w in warnings)

    def test_py_with_tests_no_warning(self, tmp_path):
        """Modified .py in ouroboros/ WITH test changes should not warn."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = " M ouroboros/loop.py\n M tests/test_loop.py\n"
            elif "diff" in cmd:
                result.stdout = "small diff"
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            assert not any("test" in w.lower() for w in warnings)

    def test_large_diff_warning(self, tmp_path):
        """Very large diffs should produce a size warning."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = " M bigfile.py\n"
            elif "diff" in cmd:
                result.stdout = "x" * 500_000  # 500K chars
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            assert any("large" in w.lower() or "size" in w.lower() for w in warnings)

    def test_version_sync_warning_included(self, tmp_path):
        """Version sync issues from check_worktree_version_sync should be included."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = " M VERSION\n"
            elif "diff" in cmd:
                result.stdout = "small diff"
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            with patch("ouroboros.tools.review_helpers.check_worktree_version_sync",
                       return_value="VERSION mismatch: 1.0 vs 2.0"):
                warnings = check_worktree_readiness(tmp_path)
                assert any("version" in w.lower() or "mismatch" in w.lower() for w in warnings)

    def test_stale_uv_lock_root_version_is_reported(self, tmp_path):
        (tmp_path / "VERSION").write_text("4.99.0\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text('version = "4.99.0"\n', encoding="utf-8")
        (tmp_path / "uv.lock").write_text(
            '[[package]]\nname = "ouroboros"\nversion = "4.98.0"\n'
            'source = { editable = "." }\n',
            encoding="utf-8",
        )

        warning = check_worktree_version_sync(tmp_path)

        assert "uv.lock" in warning

    def test_git_error_does_not_crash(self, tmp_path):
        """If git subprocess fails, the gate should not crash."""
        with patch("ouroboros.tools.review_helpers.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10)):
            warnings = check_worktree_readiness(tmp_path)
            # Should return empty or a graceful warning, not raise
            assert isinstance(warnings, list)

    def test_paths_scoping(self, tmp_path):
        """When paths are provided, only those paths should be checked."""
        selected = ["ouroboros/loop.py", "ouroboros/config.py"]
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                if "--" in cmd:
                    result.stdout = " M ouroboros/loop.py\n"
                else:
                    result.stdout = " M ouroboros/loop.py\n M tests/test_loop.py\n"
            elif "diff" in cmd:
                assert cmd[cmd.index("--") + 1:] == selected
                result.stdout = "small diff"
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path, paths=selected)
            # With path scoping, only ouroboros/loop.py is visible → tests/ not in scope → warning
            assert any("test" in w.lower() for w in warnings)

    def test_returns_list_type(self, tmp_path):
        """check_worktree_readiness must always return a list, never a generator."""
        with patch("ouroboros.tools.review_helpers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=" M foo.py\n", stderr="")
            result = check_worktree_readiness(tmp_path)
            assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_size_ratchet_findings_surface_as_official_ci_warnings(self, tmp_path, monkeypatch):
        """Size-ratchet validator findings surface as warn-only readiness lines
        naming the enforcing surface (the official repository CI), never as a
        local block. Real scratch git checkout; the validator is stubbed to a
        deterministic finding."""
        import ouroboros.review as review_module

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / "dirty.py").write_text("x = 1\n", encoding="utf-8")

        finding = "byte debt grew: ouroboros/example.py 200001 -> 200002"
        monkeypatch.setattr(review_module, "validate_size_ratchet", lambda *_a, **_k: [finding])

        warnings = check_worktree_readiness(repo)

        assert f"official CI will enforce: {finding}" in warnings


class TestReadinessGateBlocksBeforeAlreadyFresh:
    """Regression: readiness gate must block on clean worktree even if a prior fresh run exists."""

    def test_clean_worktree_blocked_even_with_prior_fresh(self, tmp_path):
        """No uncommitted changes should return error even if already_fresh would match."""
        def side_effect(cmd, **kwargs):
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = ""  # clean worktree
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            # Must include "no uncommitted changes" warning
            assert any("no uncommitted changes" in w.lower() for w in warnings)

    def test_no_uncommitted_changes_is_first_warning_and_blocks(self, tmp_path):
        """The 'no uncommitted changes' warning should cause early return (no further checks)."""
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock(returncode=0, stderr="")
            if "status" in cmd and "--porcelain" in cmd:
                result.stdout = ""  # clean
            else:
                result.stdout = ""
            return result

        with patch("ouroboros.tools.review_helpers.subprocess.run", side_effect=side_effect):
            warnings = check_worktree_readiness(tmp_path)
            # Should have stopped early — only the initial status check ran
            assert any("no uncommitted changes" in w.lower() for w in warnings)
            # Should NOT have a large diff warning or test-related warnings
            assert not any("test" in w.lower() for w in warnings)
            assert not any("large" in w.lower() for w in warnings)


class TestSharedGitReviewHelpers:
    """Regression coverage for shared review/git parsing helpers."""

    def test_name_status_preflight_preserves_rename_and_copy_semantics(self):
        from ouroboros.tools.review_helpers import (
            format_name_status_for_preflight,
            paths_from_name_status,
        )

        raw = (
            "R100\touroboros/old.py\ttools/new.py\n"
            "C075\touroboros/base.py\touroboros/new_copy.py\n"
            "D\tREADME.md\n"
        )

        assert format_name_status_for_preflight(raw) == (
            "D  ouroboros/old.py\n"
            "A  tools/new.py\n"
            "A  ouroboros/new_copy.py\n"
            "D  README.md"
        )
        assert paths_from_name_status(raw) == [
            "ouroboros/old.py",
            "tools/new.py",
            "ouroboros/base.py",
            "ouroboros/new_copy.py",
            "README.md",
        ]

    def test_porcelain_line_helper_can_return_current_or_both_paths(self):
        from ouroboros.tools.review_helpers import paths_from_porcelain_line

        line = "R  docs/old.py -> ouroboros/new.py"

        assert paths_from_porcelain_line(line) == ["docs/old.py", "ouroboros/new.py"]
        assert paths_from_porcelain_line(
            line,
            include_sources_for_renames=False,
        ) == ["ouroboros/new.py"]

    def test_porcelain_z_helper_can_include_rename_sources(self):
        from ouroboros.tools.review_helpers import parse_changed_paths_from_porcelain_z

        raw = b"R  docs/new.py\0docs/old.py\0C  copy.py\0base.py\0M  keep.py\0"

        assert parse_changed_paths_from_porcelain_z(raw) == [
            "docs/new.py",
            "copy.py",
            "keep.py",
        ]
        assert parse_changed_paths_from_porcelain_z(
            raw,
            include_sources_for_renames=True,
        ) == [
            "docs/new.py",
            "docs/old.py",
            "copy.py",
            "base.py",
            "keep.py",
        ]

    def test_snapshot_hash_uses_same_rename_paths_as_commit_gate(self, tmp_path):
        import subprocess

        from ouroboros.review_state import compute_snapshot_hash

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True, capture_output=True)
        (repo / "old_name.txt").write_text("same\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)
        (repo / "old_name.txt").rename(repo / "new_name.txt")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)

        assert compute_snapshot_hash(repo, "rename") == compute_snapshot_hash(
            repo,
            "rename",
            paths=["old_name.txt", "new_name.txt"],
        )
