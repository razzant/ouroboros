"""The pre-commit preflight gate: what blocks a commit before review runs.

``_preflight_check`` (split verbatim out of ``tests/test_git_review_pipeline.py``):
the table-driven blocker cases and the P9 history/size limits it enforces. Then
the commit gate's preflight (decision 3A): the free deterministic checks and the
tests block, the author's ONE named row only looks (``review_change`` with
``surface=preflight``), and the commit record states the fact either way. The
``candidate`` fixture is the prepared commit candidate the gate tests share.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from tests._git_review_pipeline_shared import (
    _get_review_module,
)
from ouroboros import commit_admission
from ouroboros.review_state import load_state
from ouroboros.tools import commit_gate, git, review_helpers
from ouroboros.tools.registry import ToolContext


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    """One staged change in a fresh repository, its release-metadata and readiness checks
    and the managed-proof probe stubbed, under blocking enforcement."""
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    # Match the frozen-checkout fixture's EOL contract before the first index
    # write; custody tests must reach the review cycle on Windows as well.
    for args in (("init",), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid"),
                 ("config", "core.autocrlf", "false")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "change.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "change.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (repo / "change.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "change.py"], cwd=repo, check=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="inline-task", emit_progress_fn=lambda *a: None)
    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(commit_admission, "release_metadata_preflight", lambda *a, **kw: "")
    monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    return ctx


def _make_agent_repo(tmp_path):
    """A minimal agent-repo layout (``ouroboros/__init__.py`` present)."""
    (tmp_path / "ouroboros").mkdir(exist_ok=True)
    (tmp_path / "ouroboros" / "__init__.py").write_text("")
    return tmp_path


def _init_git_repo(repo):
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


def _stub_preflight_lanes(repo, monkeypatch):
    """Keep runner/checkout/proof real; stub only lane execution in composition tests."""
    from ouroboros import preflight_runner as pr
    from tests.test_preflight_test_proof import _suite

    _suite(repo)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(pr, "_verify_preflight_plugins", lambda *a: [])
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    monkeypatch.setattr(git, "_consecutive_test_failures", 0)
    lanes = []

    def green_lane(python, worktree, temp_root, args, timeout):
        assert worktree != repo and (worktree / "candidate.txt").read_text(encoding="utf-8") == "tested"
        lanes.append((worktree, tuple(args)))
        return 0, "green fixture lane", ""

    monkeypatch.setattr(pr, "_execute_pytest_pass", green_lane)
    return lanes


def _write_release_files(repo, *, version, pyproject_version=None, minor_rows=1):
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "VERSION").write_text(version + "\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "ouroboros"\nversion = "' + (pyproject_version or version.replace("-rc.", "rc")) + '"\n',
        encoding="utf-8",
    )
    rows = [f"| 5.{idx}.0 | 2026-05-15 | row {idx} |" for idx in range(30, 30 - minor_rows, -1)]
    if not any(row.startswith(f"| {version} |") for row in rows):
        rows.insert(0, f"| {version} | 2026-05-15 | current row |")
    (repo / "README.md").write_text(
        f"[![Version {version}](https://img.shields.io/badge/version-{version.replace('-', '--')}-green.svg)](VERSION)\n\n"
        "## Version History\n\n| Version | Date | Description |\n|---------|------|-------------|\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    (repo / "docs" / "ARCHITECTURE.md").write_text(f"# Ouroboros v{version} — Architecture & Reference\n", encoding="utf-8")


# --- Unified review gate ---

# Each tuple: (case_id, message, staged_files, expected_substrings_or_none).
# expected_substrings_or_none is ``None`` when ``_preflight_check`` should
# pass; otherwise an iterable of substrings every one of which must appear in
# the returned blocker text.
_PREFLIGHT_CASES = [
    # Regression (#447): the commit-message version-reference heuristic was
    # removed — a version-shaped message no longer demands VERSION staged.
    (
        "version_ref_message_without_version_passes",
        "v3.24.0: big change",
        "ouroboros/tools/git.py\nREADME.md",
        None,
    ),
    # Regression (#447): the old "version" substring test matched "conversion"
    # and told an unrelated commit to bump VERSION.
    (
        "conversion_message_does_not_demand_version",
        "add unit conversion helper",
        "M  ouroboros/units.py",
        None,
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
    # Regression (#447): the tests-required predicate was removed — a .py
    # change under ouroboros/ (e.g. comment-only) without staged tests is no
    # longer refused; CHECKLISTS.md item 4 (tests_affected) owns coverage.
    (
        "logic_without_tests_passes",
        "fix something",
        "M  ouroboros/tools/shell.py\nM  VERSION\nM  README.md",
        None,
    ),
    (
        "supervisor_logic_without_tests_passes",
        "update supervisor",
        "M  supervisor/workers.py",
        None,
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
def test_preflight_check(case_id, message, staged_files, expected, monkeypatch, tmp_path):
    review = _get_review_module()
    values = {"VERSION": "3.24.0", "README.md":
              "[![Version 3.24.0](https://img.shields.io/badge/version-3.24.0-green.svg)]\n| 3.24.0 | release |"}
    monkeypatch.setattr(review, "_git_show_staged", lambda repo, path: values.get(path))
    from ouroboros import commit_admission

    monkeypatch.setattr(commit_admission, "_head_text", lambda repo, path: values.get(path))  # HEAD == index
    result = review._preflight_check(message, staged_files, tmp_path)
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
            return None

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
                return bloated_readme + "\nClarified docs prose.\n"
            return None

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        # The over-limit history is already HEAD's; this docs commit only edits prose,
        # so the docs-only carrier comparison finds every span byte-identical.
        from ouroboros import commit_admission
        compared = []
        monkeypatch.setattr(commit_admission, "_head_text",
                            lambda repo, path: compared.append(path) or {"README.md": bloated_readme}.get(path))
        # Only README staged — no VERSION, no ouroboros/*.py.
        result = review._preflight_check(
            "fix docs", "M  README.md", "/repo"
        )
        assert result is None, (
            "Check 7 fired without VERSION staged — it should be a no-op."
        )
        assert compared == ["README.md"]

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
            return values.get(path)

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        result = review._preflight_check(
            "v4.99.0: release",
            "M  VERSION\nM  README.md\nM  uv.lock",
            "/repo",
        )

        assert result is not None
        assert "uv.lock" in result

    def test_stale_staged_web_package_lock_root_version_blocks(self, monkeypatch):
        """MAJOR-1 (rc.15 review): the staged lockfile is read like the sibling
        carriers (the staged blob, not the worktree), so a root-entry desync
        blocks the commit gate naming web/package-lock.json."""
        review = _get_review_module()
        readme = self._wrap_readme("")
        seen = []

        def _fake_git_show(repo_dir, path: str) -> str:
            seen.append(path)
            values = {
                "VERSION": "4.99.0",
                "pyproject.toml": 'version = "4.99.0"',
                "uv.lock": (
                    '[[package]]\nname = "ouroboros"\nversion = "4.99.0"\n'
                    'source = { editable = "." }\n'
                ),
                "web/package.json": '{"version": "4.99.0"}',
                "web/package-lock.json": (
                    '{\n  "name": "ouroboros-web",\n  "version": "4.99.0",\n  "lockfileVersion": 3,\n'
                    '  "packages": {\n    "": {\n      "name": "ouroboros-web",\n      "version": "4.98.0"\n'
                    '    }\n  }\n}\n'
                ),
                "web/modules/api_types.js": "GATEWAY_CONTRACT_VERSION = '4.99.0'",
                "README.md": readme,
                "docs/ARCHITECTURE.md": "# Ouroboros v4.99.0 — Architecture",
            }
            return values.get(path)

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        result = review._preflight_check(
            "v4.99.0: release",
            "M  VERSION\nM  README.md\nM  web/package-lock.json",
            "/repo",
        )

        assert "web/package-lock.json" in seen, "the lockfile must be read from the staged index"
        assert result is not None and "PREFLIGHT_BLOCKED" in result
        assert 'web/package-lock.json (expected both root "version" entries = "4.99.0")' in result

    def test_missing_readme_reports_staging_and_source_problems(self, monkeypatch):
        """Missing indexed README retains the staging finding beside source unavailability."""
        review = _get_review_module()

        def _fake_git_show(repo_dir, path: str) -> str:
            if path == "VERSION":
                return "4.99.0"
            return None  # README absent from staged index

        monkeypatch.setattr(review, "_git_show_staged", _fake_git_show)
        result = review._preflight_check(
            "v4.99.0 bump", "M  VERSION\nM  tests/test_foo.py", "/repo"
        )
        assert result is not None and "Missing from staged: README.md" in result
        assert "PREFLIGHT_UNAVAILABLE" in result


# ---------------------------------------------------------------------------
# The free deterministic checks every commit runs ahead of any paid review
# ---------------------------------------------------------------------------

class TestSyntaxPreflightHelper:
    def test_non_agent_repo_skipped(self, tmp_path):
        """Target repos without `ouroboros/__init__.py` bypass the gate entirely."""
        (tmp_path / "broken.py").write_text("def foo(:\n")
        assert commit_admission.syntax_preflight_staged_py_files(tmp_path, ["broken.py"]) is None

    def test_agent_repo_valid_passes(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "good.py").write_text("def foo():\n    return 1\n")
        assert commit_admission.syntax_preflight_staged_py_files(repo, ["good.py"]) is None

    def test_agent_repo_syntax_error_blocks(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "broken.py").write_text("def foo(:\n")
        out = commit_admission.syntax_preflight_staged_py_files(repo, ["broken.py"])
        assert out is not None and "PREFLIGHT_BLOCKED" in out and "broken.py:1:" in out

    def test_multiple_errors_all_reported(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "a.py").write_text("def a(:\n")
        (repo / "b.py").write_text("x = (\n")
        out = commit_admission.syntax_preflight_staged_py_files(repo, ["a.py", "b.py"])
        assert "a.py" in (out or "") and "b.py" in (out or "")

    def test_non_py_files_ignored(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "README.md").write_text("# bad:\ndef foo(:")
        (repo / "data.json").write_text("not even python")
        assert commit_admission.syntax_preflight_staged_py_files(repo, ["README.md", "data.json"]) is None

    def test_staged_deletion_tolerated(self, tmp_path):
        """A staged deletion has no on-disk file — must not raise."""
        repo = _make_agent_repo(tmp_path)
        assert commit_admission.syntax_preflight_staged_py_files(repo, ["deleted.py"]) is None

    def test_no_pycache_created(self, tmp_path):
        """compile() with dont_inherit=True must not materialise __pycache__."""
        repo = _make_agent_repo(tmp_path)
        (repo / "good.py").write_text("x = 1\n")
        commit_admission.syntax_preflight_staged_py_files(repo, ["good.py"])
        assert not any("__pycache__" in str(p) for p in repo.rglob("*"))

    def test_empty_path_list_passes(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        assert commit_admission.syntax_preflight_staged_py_files(repo, []) is None

    def test_mixed_valid_and_broken(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "ok.py").write_text("x = 1\n")
        (repo / "bad.py").write_text("def f(:\n")
        out = commit_admission.syntax_preflight_staged_py_files(repo, ["ok.py", "bad.py"])
        assert out is not None and "bad.py" in out and "- ok.py" not in out

    def test_null_byte_source_is_blocked_as_preflight(self, tmp_path):
        """`compile()` raises ValueError (not SyntaxError) on null bytes; the check
        still blocks with an actionable PREFLIGHT_BLOCKED naming the file."""
        repo = _make_agent_repo(tmp_path)
        (repo / "null.py").write_bytes(b"x = 1\x00\n")
        out = commit_admission.syntax_preflight_staged_py_files(repo, ["null.py"])
        assert out is not None and "null.py" in out and "PREFLIGHT_BLOCKED" in out


class TestReleaseMetadataPreflight:
    def test_stale_uv_lock_blocks(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        _init_git_repo(repo)
        _write_release_files(repo, version="5.99.0-rc.1")
        (repo / "uv.lock").write_text(
            '[[package]]\nname = "ouroboros"\nversion = "5.98.0"\nsource = { editable = "." }\n', encoding="utf-8")
        result = commit_admission.release_metadata_preflight(repo, "v5.99.0-rc.1: release", ["VERSION"])
        assert result is not None and "uv.lock" in result

    def test_stale_web_package_lock_blocks(self, tmp_path):
        """The lockfile is a release carrier the sync writes; a root-entry desync is a
        typed PREFLIGHT_BLOCKED naming the file (also the SM1 stand check's carrier gate)."""
        repo = _make_agent_repo(tmp_path)
        _init_git_repo(repo)
        _write_release_files(repo, version="5.99.0-rc.1")
        (repo / "web").mkdir()
        (repo / "web" / "package.json").write_text(
            '{\n  "name": "ouroboros-web",\n  "version": "5.99.0-rc.1"\n}\n', encoding="utf-8")
        (repo / "web" / "package-lock.json").write_text(
            '{\n  "name": "ouroboros-web",\n  "version": "5.98.0",\n  "lockfileVersion": 3,\n'
            '  "packages": {\n    "": {\n      "name": "ouroboros-web",\n      "version": "5.98.0"\n'
            '    }\n  }\n}\n', encoding="utf-8")
        result = commit_admission.release_metadata_preflight(repo, "v5.99.0-rc.1: release", ["VERSION"])
        assert result is not None and "PREFLIGHT_BLOCKED" in result
        assert 'web/package-lock.json (expected both root "version" entries = "5.99.0-rc.1")' in result

    def test_doc_only_carve_is_the_commit_gate_classifier(self, tmp_path):
        """One detector: the release-metadata carve is the one remaining reader of
        ``_diff_is_doc_only`` (the tests rule runs the suite on a documentation diff too).
        A code file, a mixed diff and a doc under ``tests/`` keep the block; the carve never
        touches the carrier-coherence checks once VERSION is in scope."""
        from ouroboros.tools.git_review_cycle import _diff_is_doc_only

        repo = _make_agent_repo(tmp_path)
        _init_git_repo(repo)
        _write_release_files(repo, version="5.99.0-rc.1")
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
        (repo / "docs" / "NOTES.md").write_text("# notes\n", encoding="utf-8")
        (repo / "ouroboros" / "feature.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "tests").mkdir(exist_ok=True)
        (repo / "tests" / "NOTES.md").write_text("# notes\n", encoding="utf-8")

        def _preflight(paths):
            return commit_admission.release_metadata_preflight(repo, "m", paths)

        assert _preflight(["docs/NOTES.md"]) is None
        assert _diff_is_doc_only(["docs/NOTES.md"]) is True
        for scope in (["ouroboros/feature.py"], ["docs/NOTES.md", "ouroboros/feature.py"], ["tests/NOTES.md"]):
            assert _diff_is_doc_only(scope) is False, scope
            blocked = _preflight(scope)
            assert blocked is not None and "VERSION is not in scope" in blocked, scope
        (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "5.98.0"\n', encoding="utf-8")
        stale = _preflight(["VERSION", "docs/NOTES.md"])
        assert stale is not None and "pyproject.toml" in stale

    @pytest.mark.parametrize("case", ["p9_history", "readme_not_staged"])
    def test_the_commit_gate_blocks_on_the_staged_index_before_any_look(self, tmp_path, monkeypatch, case):
        """``commit_gate.deterministic_preflight`` reads the STAGED index (the prepared lane of
        the retired advisory, as is: the version-neutral form stays allowed there), so a P9
        overflow or a numbered release without its staged README blocks before the tests and
        before any paid look."""
        (tmp_path / "repo").mkdir()
        repo = _make_agent_repo(tmp_path / "repo")
        _init_git_repo(repo)
        if case == "p9_history":
            _write_release_files(repo, version="5.99.0-rc.1", minor_rows=6)
            message, paths, needle = "v5.99.0-rc.1: test", None, "Version History exceeds"
        else:
            _write_release_files(repo, version="5.99.0-rc.1")
            subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
            (repo / "VERSION").write_text("5.99.0-rc.2\n", encoding="utf-8")
            message, paths, needle = "v5.99.0-rc.2: bump", ["VERSION"], "Missing from staged: README.md"
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
        monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **kw: [])
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="t-det", emit_progress_fn=lambda *a: None)
        assert needle in commit_gate.deterministic_preflight(ctx, message, paths)


class TestTestsPreflightProofBinding:
    """Admission forwards only runner-minted evidence to managed telemetry.

    A successful return without execution (including no suite) is not proof.
    """

    def _ctx(self, tmp_path):
        from types import SimpleNamespace
        return SimpleNamespace(task_id="t", task_metadata={}, repo_dir=str(tmp_path),
                               emit_progress_fn=lambda *_a, **_k: None)

    def test_red_run_returns_error_and_records_no_proof(self, tmp_path):
        ctx = self._ctx(tmp_path)
        err = commit_admission.run_tests_preflight_with_proof(ctx, runner=lambda c: "FAILED: 2 failed")
        assert err == "FAILED: 2 failed"
        assert not getattr(ctx, "_preflight_test_proof", None)
        assert ctx._preflight_tests_passed is False

    def test_green_run_records_the_managed_proof(self, tmp_path, monkeypatch):
        from ouroboros.tools.review_helpers import _run_review_preflight_tests
        from tests.test_update_merge_assisted import _init_repo
        import supervisor.update_merge as um

        repo, _ = _init_repo(tmp_path)
        lanes = _stub_preflight_lanes(repo, monkeypatch)
        ctx = self._ctx(repo)
        recorded = []
        monkeypatch.setattr(um, "record_managed_tests_proof",
                            lambda c: recorded.append((c, c._preflight_test_proof)) or c._preflight_test_proof.tree)
        assert commit_admission.run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
        assert len(lanes) == 2 and len({worktree for worktree, _ in lanes}) == 1
        assert recorded == [(ctx, ctx._preflight_test_proof)]
        assert commit_admission.preflight_test_proof_matches(ctx, repo)

    def test_none_without_runner_receipt_records_no_proof(self, tmp_path, monkeypatch):
        ctx = self._ctx(tmp_path)
        monkeypatch.setattr("supervisor.update_merge.record_managed_tests_proof",
                            lambda c: pytest.fail("None is not an execution receipt"))
        assert commit_admission.run_tests_preflight_with_proof(ctx, runner=lambda c: None) is None
        assert ctx._preflight_test_proof is None
        assert ctx._preflight_tests_passed is False


# ---------------------------------------------------------------------------
# The commit gate's preflight (decision 3A)
# ---------------------------------------------------------------------------

def _roster(monkeypatch, *, enabled=True, subagent_id="api-scout"):
    """The catalog: the review pool (``tests.review_pool_rosters.mixed_pool_rows``) plus one
    enabled UNMARKED row the author may name for a look — a reviewer only when named."""
    from tests.review_pool_rosters import mixed_pool_rows, pool_roster, set_review_pool

    scout = {"subagent_id": subagent_id, "name": "API scout", "recommended_use": "An early look.",
             "route": {"kind": "api_model", "target_id": "openai/fake-reviewer"}, "effort": "high", "enabled": enabled}
    set_review_pool(monkeypatch, pool_roster(*mixed_pool_rows(), scout))
    return subagent_id


def _gate(ctx, paths=("change.py",), **kwargs):
    return git._preflight_and_tests_gate(ctx, "candidate", 0, classification_paths=list(paths), **kwargs)


def _suite(monkeypatch, calls, error=None):
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda ctx, **kw: calls.append("tests") or error)


def _look(monkeypatch, calls):
    def look(ctx, reviewer, **kw):
        calls.append(("look", reviewer, kw["review_rebuttal"], kw["goal"], kw["scope"]))
        return {"status": "performed", "record_id": "rl-look", "reviewer": reviewer, "aggregate": "PASS"}

    monkeypatch.setattr(commit_gate, "run_commit_preflight", look)


def test_the_system_prompt_says_which_look_the_commit_record_counts():
    """The model's own prompt states the record rule the tests below pin: only the row
    named in the ``commit_reviewed`` call itself is the commit's preflight; a separate
    ``preflight_review`` is an early look of its own; no name records not performed."""
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    phrase = " ".join(text.split())
    assert ("the commit's record counts an early look only when that call itself names the row "
            "(`commit_reviewed(preflight_reviewer=…)`)") in phrase
    assert "a separate `preflight_review` is an early look of its own" in phrase
    assert "a commit that names no row records its preflight as not performed" in phrase


@pytest.mark.parametrize("skip, status", [(False, "not_performed"), (True, "skipped")])
def test_without_a_named_row_the_record_states_the_fact_and_nothing_looks(candidate, monkeypatch, skip, status):
    calls = []
    _suite(monkeypatch, calls)
    monkeypatch.setattr(commit_gate, "run_commit_preflight", lambda *a, **kw: pytest.fail("no row was named"))
    assert _gate(candidate, skip_advisory_pre_review=skip) is None
    assert calls == ["tests"], "the deterministic checks and the tests run either way"
    assert candidate._commit_preflight == {"status": status, "record_id": ""}
    assert commit_gate._review_preflight_facts(candidate) == {"status": status, "record_id": ""}


def test_a_named_row_looks_after_the_checks_and_the_tests(candidate, monkeypatch):
    calls = []
    _suite(monkeypatch, calls)
    _look(monkeypatch, calls)
    assert _gate(candidate, preflight_reviewer="api-scout", review_rebuttal="new evidence", goal="g", scope="s") is None
    assert calls == ["tests", ("look", "api-scout", "new evidence", "g", "s")]
    assert commit_gate._review_preflight_facts(candidate) == {
        "status": "performed", "record_id": "rl-look", "reviewer": "api-scout", "aggregate": "PASS"}


@pytest.mark.parametrize("named", [False, True])
def test_a_doc_only_diff_pays_the_suite_like_any_other_diff(candidate, monkeypatch, named):
    """Documentation has tests too (version rows, canon tables, inventories): a ``.md``-only
    diff runs the suite whether or not a row is named — the retired advisory's rule for every
    commit; ``skip_tests`` is the one exemption."""
    calls = []
    _suite(monkeypatch, calls)
    _look(monkeypatch, calls)
    assert _gate(candidate, paths=("docs/notes.md",), preflight_reviewer="api-scout" if named else "") is None
    assert calls.count("tests") == 1
    calls.clear()
    assert _gate(candidate, paths=("docs/notes.md",), skip_tests=True) is None
    assert calls.count("tests") == 0


def test_every_site_states_the_one_tests_rule_for_a_documentation_diff():
    """Owner answer A (2026-10-08): the suite runs before any commit to the body, a
    documentation-only diff included; ``skip_tests`` is the one exemption. The gate's
    condition, its comments, the checklist line and the test modules that once pinned the
    retired carve say the same thing, so no site describes a doc-only diff as exempt."""
    import inspect
    import pathlib

    import tests.test_git_review_tests_gate as tests_gate
    import tests.test_skip_tests_doc_only as classifier_tests

    repo = pathlib.Path(git.__file__).resolve().parents[2]
    checklist = (repo / "docs" / "CHECKLISTS.md").read_text(encoding="utf-8")
    sites = {
        "git._preflight_and_tests_gate": inspect.getsource(git._preflight_and_tests_gate),
        "git._managed_candidate_needs_proof": inspect.getsource(git._managed_candidate_needs_proof),
        "docs/CHECKLISTS.md": checklist,
        "tests/test_git_review_tests_gate.py": tests_gate.__doc__ or "",
        "tests/test_skip_tests_doc_only.py": classifier_tests.__doc__ or "",
        "this module": TestReleaseMetadataPreflight.test_doc_only_carve_is_the_commit_gate_classifier.__doc__ or "",
    }
    retired = ("exempt from the suite", "preflight bypass", "skip_tests/doc-only",
               "the tests rule applies", "diff-aware")
    for name, text in sites.items():
        for phrase in retired:
            assert phrase not in text, (name, phrase)
    assert "if not message and (not skip_tests or _managed_needs_proof):" in sites["git._preflight_and_tests_gate"]
    assert "documentation has tests too" in sites["git._preflight_and_tests_gate"]
    assert "a documentation-only diff included" in checklist
    assert "a documentation-only diff included" in sites["tests/test_git_review_tests_gate.py"]


def test_a_syntax_error_blocks_before_the_tests_and_the_look(candidate, monkeypatch):
    calls = []
    _suite(monkeypatch, calls)
    _look(monkeypatch, calls)
    _make_agent_repo(candidate.repo_dir)
    (candidate.repo_dir / "change.py").write_text("value = (\n")
    subprocess.run(["git", "add", "change.py"], cwd=candidate.repo_dir, check=True)
    result = _gate(candidate, preflight_reviewer="api-scout")
    assert result["status"] == "blocked" and result["block_reason"] == "preflight"
    assert "syntax errors:" in result["message"] and "change.py:1" in result["message"]
    assert calls == [] and candidate._commit_preflight["status"] == "not_performed"


@pytest.mark.parametrize("choice, status", [
    ({"preflight_reviewer": "api-scout"}, "not_performed"),
    ({}, "not_performed"),
    ({"skip_advisory_pre_review": True}, "skipped"),
])
def test_a_failed_suite_blocks_every_commit_before_any_look(candidate, monkeypatch, choice, status):
    calls = []
    _suite(monkeypatch, calls, error="failed targeted suite")
    _look(monkeypatch, calls)
    result = _gate(candidate, **choice)
    assert result["block_reason"] == "tests_preflight_blocked" and calls == ["tests"]
    assert candidate._commit_preflight["status"] == status


@pytest.mark.parametrize("enabled, selector", [(True, "api-scout"), (False, "api-scout"), (None, "nobody")])
def test_the_named_row_is_an_enabled_catalog_row_or_an_argument_error(candidate, monkeypatch, enabled, selector):
    if enabled is not None:
        _roster(monkeypatch, enabled=enabled)
    assert (commit_gate.preflight_reviewer_error(selector) == "") is bool(enabled)
    if not enabled:
        result = git._commit_reviewed(candidate, "candidate", preflight_reviewer=selector)
        assert "TOOL_ARG_ERROR" in result and "Nothing was staged, reviewed or recorded" in result
        assert load_state(candidate.drive_root).attempts == []


@pytest.mark.parametrize("extra", [
    {"skip_advisory_review": True},
    {"review_reference": {"review_record_id": "rl-x"},
     "author_disposition": {"disposition": "accepted", "rationale": "Known tradeoff."}},
])
def test_a_named_row_beside_a_skip_or_an_author_continuation_is_refused(candidate, monkeypatch, extra):
    _roster(monkeypatch)
    result = git._commit_reviewed(candidate, "candidate", preflight_reviewer="api-scout", **extra)
    assert "TOOL_ARG_ERROR" in result and "preflight_reviewer" in result
    assert load_state(candidate.drive_root).attempts == []


def _panel_rows():
    models = ("openai/gpt-5", "anthropic/claude-x", "google/gemini")
    return [{"slot_id": f"s{i}", "model_id": model, "status": "responded", "raw_text": "[]", "parsed_items": [],
             "cost_usd": 0.01} for i, model in enumerate(models, 1)]


def test_a_performed_preflight_of_the_same_tree_never_answers_the_commit_panel(candidate, monkeypatch):
    """The real cycle: the named row's ``surface=preflight`` wave reads the worktree, then the
    panel still runs over the same tree and writes its own record, which names the look."""
    import hashlib

    from ouroboros import review_ledger as rl
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools import review_change as rc
    from tests.test_review_change_tool import Wave

    _roster(monkeypatch)
    monkeypatch.setattr(commit_gate, "review_max_cycles", lambda: None)
    look = Wave()
    monkeypatch.setattr(rc, "run_parallel_review", look)
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: None)
    panel = []

    def reviewer(ctx, message, **kw):
        # The panel's one wave: the first seat retrieves and answers both parts, the
        # others read the packet (contract A); the ledger's facts the gate records.
        panel.append(message)
        rows = _panel_rows()
        texts = {hashlib.sha256(b"BRIEF").hexdigest(): "BRIEF", hashlib.sha256(b"PACKET").hexdigest(): "PACKET"}
        plan = []
        for index, row in enumerate(rows):
            retrieves = index == 0
            parts = ["change", "coupling"] if retrieves else ["change"]
            plan.append({"slot_id": row["slot_id"], "model": row["model_id"], "route": "api_chat", "effort": "high",
                         "parts": parts, "retrieves": retrieves,
                         "brief_sha": hashlib.sha256(b"BRIEF" if retrieves else b"PACKET").hexdigest()})
            if retrieves:
                row["parts"] = parts
                row["answers"] = {part: {"status": "responded", "verdict": "PASS", "findings": [], "critical": 0,
                                         "coverage": "complete"} for part in parts}
        ctx._last_triad_raw_results = rows
        ctx._last_review_structured = {"rows": plan, "brief_texts": texts, "quorum": rl._quorum_for(len(plan)),
                                       "started_ts": "2026-10-07T00:00:00+00:00"}
        return None, CouplingOutcome(verdict="PASS", blocked=False, status="responded"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    git._reset_commit_review_state(candidate)
    result = git._run_reviewed_stage_cycle(candidate, "candidate", 0, require_release_tag=False,
                                           preflight_reviewer="api-scout")
    assert [(call.triad, call.coupling) for call in look.calls] == [(["api-scout"], [])]
    assert panel == ["candidate"], "the panel runs after the look, over the same tree"
    fact = candidate._commit_preflight
    assert fact["status"] == "performed" and fact["record_id"] and fact["reviewer"] == "api-scout"
    preflight = rl.load_record(rl.ledger_root(candidate), fact["record_id"])
    gate = rl.load_record(rl.ledger_root(candidate), result["review_record_id"])
    assert (preflight["surface"], gate["surface"]) == ("preflight", "commit_gate")
    assert [seat["seat_id"] for seat in preflight["rows"]] == ["api-scout"]
    assert preflight["fingerprints"]["reuse_key"] != gate["fingerprints"]["reuse_key"]
    assert gate["preflight"] == fact and result["status"] == "passed"
