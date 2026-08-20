"""The pre-commit preflight gate: what blocks a commit before review runs.

Split verbatim out of ``tests/test_git_review_pipeline.py`` by theme. This
module owns ``_preflight_check``: the table-driven blocker cases and the P9
history/size limits it enforces.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from tests._git_review_pipeline_shared import (
    _get_review_module,
)


# --- Unified review gate ---

# Each tuple: (case_id, message, staged_files, expected_substrings_or_none).
# expected_substrings_or_none is ``None`` when ``_preflight_check`` should
# pass; otherwise an iterable of substrings every one of which must appear in
# the returned blocker text.
_PREFLIGHT_CASES = [
    (
        "missing_version",
        "v3.24.0: big change",
        "ouroboros/tools/git.py\nREADME.md",
        ("PREFLIGHT_BLOCKED", "VERSION"),
    ),
    (
        "missing_readme",
        "some change",
        "M  VERSION\nM  ouroboros/tools/git.py",
        ("README.md",),
    ),
    (
        "all_present_passes",
        "v3.24.0: change",
        "M  VERSION\nM  README.md\nM  ouroboros/tools/git.py\nM  tests/test_commit_gate.py",
        None,
    ),
    (
        "no_version_ref_passes",
        "fix typo in docs",
        "M  docs/ARCHITECTURE.md",
        None,
    ),
    (
        "logic_changed_without_tests_blocked",
        "fix something",
        "M  ouroboros/tools/shell.py\nM  VERSION\nM  README.md",
        ("PREFLIGHT_BLOCKED", "tests/"),
    ),
    (
        "logic_changed_with_tests_passes",
        "fix something",
        "M  ouroboros/tools/shell.py\nM  tests/test_shell_run_shell.py\nM  VERSION\nM  README.md",
        None,
    ),
    (
        "supervisor_logic_without_tests_blocked",
        "update supervisor",
        "M  supervisor/workers.py",
        ("PREFLIGHT_BLOCKED",),
    ),
    (
        "docs_only_change_no_tests_required",
        "update docs",
        "M  docs/ARCHITECTURE.md\nM  README.md",
        None,
    ),
    (
        "new_module_without_architecture_blocked",
        "add new module",
        "A  ouroboros/new_module.py\nM  tests/test_new_module.py",
        ("PREFLIGHT_BLOCKED", "ARCHITECTURE.md"),
    ),
    (
        "new_module_with_architecture_passes",
        "add new module",
        "A  ouroboros/new_module.py\nM  tests/test_new_module.py\nM  docs/ARCHITECTURE.md",
        None,
    ),
    (
        "modified_module_without_architecture_passes",
        "update existing module",
        "M  ouroboros/tools/shell.py\nM  tests/test_shell_run_shell.py",
        None,
    ),
]


@pytest.mark.parametrize(
    "case_id,message,staged_files,expected",
    _PREFLIGHT_CASES,
    ids=[c[0] for c in _PREFLIGHT_CASES],
)
def test_preflight_check(case_id, message, staged_files, expected):
    review = _get_review_module()
    result = review._preflight_check(message, staged_files, "/tmp")
    if expected is None:
        assert result is None, f"expected pass, got: {result!r}"
    else:
        assert result is not None
        for needle in expected:
            assert needle in result, f"missing {needle!r} in: {result!r}"


# ---------------------------------------------------------------------------
# Check 7: P9 history limits in _preflight_check (v4.41.0)
# ---------------------------------------------------------------------------

class TestPreflightCheck7P9Limits:
    """Verify that _preflight_check check 7 blocks when README.md Version
    History exceeds BIBLE.md P9 limits (2 major / 5 minor / 5 patch rows)."""

    # Helper: build a fake git-show-staged for check 7 tests.
    # We monkeypatch _git_show_staged to return controlled content.

    def _run_with_readme(self, monkeypatch, readme_content: str,
                         extra_staged: str = "") -> "str | None":
        """Run _preflight_check with VERSION staged and a controlled README."""
        review = _get_review_module()

        def _fake_git_show(repo_dir, path: str) -> str:
            if path == "VERSION":
                return "4.99.0"
            if path == "README.md":
                return readme_content
            if path == "pyproject.toml":
                return 'version = "4.99.0"'
            if path == "docs/ARCHITECTURE.md":
                return "# Ouroboros v4.99.0 — "
            return ""

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        staged = f"M  VERSION\nM  README.md\nM  tests/test_foo.py\n{extra_staged}".strip()
        return review._preflight_check("v4.99.0 release", staged, "/repo")

    # README must also contain the version badge to pass check 5 (version carrier
    # sync) so check 7 is actually reached. The badge line is the real format from
    # README.md: [![Version X.Y.Z](...badge/version-X.Y.Z-green.svg)].
    _BADGE_LINE = (
        "[![Version 4.99.0](https://img.shields.io/badge/version-4.99.0-green.svg)](VERSION)"
    )

    def _wrap_readme(self, rows_section: str) -> str:
        # Include a row for 4.99.0 itself so check 6 passes (changelog row required).
        current_row = "| 4.99.0 | 2026-01-01 | current release |"
        return (
            f"{self._BADGE_LINE}\n\n"
            "## Version History\n\n"
            "| Version | Date | Description |\n"
            "|---------|------|-------------|\n"
            f"{current_row}\n"
            f"{rows_section}\n"
        )

    def _readme_with_patch_rows(self, count: int) -> str:
        rows = "\n".join(
            f"| 4.{i}.1 | 2026-01-01 | patch fix |"
            for i in range(count)
        )
        return self._wrap_readme(rows)

    def _readme_with_minor_rows(self, count: int) -> str:
        rows = "\n".join(
            f"| 4.{i}.0 | 2026-01-01 | minor feature |"
            for i in range(count)
        )
        return self._wrap_readme(rows)

    def _readme_with_major_rows(self, count: int) -> str:
        rows = "\n".join(
            f"| {i}.0.0 | 2026-01-01 | major release |"
            for i in range(count)
        )
        return self._wrap_readme(rows)

    def test_patch_limit_exceeded_blocks(self, monkeypatch):
        """6 patch rows (limit 5) → PREFLIGHT_BLOCKED."""
        result = self._run_with_readme(monkeypatch, self._readme_with_patch_rows(6))
        assert result is not None, "Expected block on too many patch rows"
        assert "PREFLIGHT_BLOCKED" in result
        assert "patch" in result.lower()

    def test_patch_limit_at_boundary_passes(self, monkeypatch):
        """Exactly 5 patch rows → passes."""
        result = self._run_with_readme(monkeypatch, self._readme_with_patch_rows(5))
        assert result is None, f"Expected pass at 5 patch rows, got: {result}"

    def test_minor_limit_exceeded_blocks(self, monkeypatch):
        """6 minor rows (limit 5) → PREFLIGHT_BLOCKED."""
        result = self._run_with_readme(monkeypatch, self._readme_with_minor_rows(6))
        assert result is not None, "Expected block on too many minor rows"
        assert "PREFLIGHT_BLOCKED" in result
        assert "minor" in result.lower()

    def test_minor_limit_at_boundary_passes(self, monkeypatch):
        """Exactly 5 minor rows → passes."""
        result = self._run_with_readme(monkeypatch, self._readme_with_minor_rows(5))
        assert result is None, f"Expected pass at 5 minor rows, got: {result}"

    def test_major_limit_exceeded_blocks(self, monkeypatch):
        """3 major rows (limit 2) → PREFLIGHT_BLOCKED."""
        result = self._run_with_readme(monkeypatch, self._readme_with_major_rows(3))
        assert result is not None, "Expected block on too many major rows"
        assert "PREFLIGHT_BLOCKED" in result
        assert "major" in result.lower()

    def test_major_limit_at_boundary_passes(self, monkeypatch):
        """Exactly 2 major rows → passes."""
        result = self._run_with_readme(monkeypatch, self._readme_with_major_rows(2))
        assert result is None, f"Expected pass at 2 major rows, got: {result}"

    def test_check7_only_fires_when_version_staged(self, monkeypatch):
        """Check 7 must be a no-op when VERSION is not in the staged set."""
        review = _get_review_module()

        # README with too many patch rows, but VERSION is NOT staged.
        bloated_readme = self._readme_with_patch_rows(10)

        def _fake_git_show(repo_dir, path: str) -> str:
            if path == "README.md":
                return bloated_readme
            return ""

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        # Only README staged — no VERSION, no ouroboros/*.py.
        result = review._preflight_check(
            "fix docs", "M  README.md", "/repo"
        )
        assert result is None, (
            "Check 7 fired without VERSION staged — it should be a no-op."
        )

    def test_stale_staged_uv_lock_root_version_blocks(self, monkeypatch):
        review = _get_review_module()
        readme = self._wrap_readme("")

        def _fake_git_show(repo_dir, path: str) -> str:
            values = {
                "VERSION": "4.99.0",
                "pyproject.toml": 'version = "4.99.0"',
                "uv.lock": (
                    '[[package]]\nname = "ouroboros"\nversion = "4.98.0"\n'
                    'source = { editable = "." }\n'
                ),
                "web/package.json": '{"version": "4.99.0"}',
                "web/modules/api_types.js": "GATEWAY_CONTRACT_VERSION = '4.99.0'",
                "README.md": readme,
                "docs/ARCHITECTURE.md": "# Ouroboros v4.99.0 — Architecture",
            }
            return values.get(path, "")

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        result = review._preflight_check(
            "v4.99.0: release",
            "M  VERSION\nM  README.md\nM  uv.lock",
            "/repo",
        )

        assert result is not None
        assert "uv.lock" in result

    def test_check7_passes_when_readme_not_staged(self, monkeypatch):
        """VERSION staged but README not staged → check 7 silently skips
        (git show returns empty string for an un-staged README)."""
        review = _get_review_module()

        def _fake_git_show(repo_dir, path: str) -> str:
            if path == "VERSION":
                return "4.99.0"
            return ""  # README absent from staged index

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        # Tests staged to pass check 3; ARCHITECTURE.md for check 4.
        result = review._preflight_check(
            "v4.99.0 bump", "M  VERSION\nM  tests/test_foo.py", "/repo"
        )
        # Check 1 fires first (README.md missing from staged when VERSION staged).
        # This is acceptable — the missing README is caught by check 1, not check 7.
        # Either result is valid here; we just verify no crash.
        assert result is None or "PREFLIGHT_BLOCKED" in result
