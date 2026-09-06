"""Advisory syntax preflight regression tests (Phase 2.4).

Verify that `_syntax_preflight_staged_py_files` short-circuits the expensive
Claude SDK advisory call when any staged `.py` file has a SyntaxError, and
passes through cleanly in all other cases. No `__pycache__`, no subprocess
side-effects.
"""

from __future__ import annotations

import pathlib
import subprocess
import unittest.mock as mock

import pytest

import ouroboros.tools.claude_advisory_review as advisory
from ouroboros.tools.claude_advisory_review import (
    _syntax_preflight_staged_py_files,
)


@pytest.fixture(autouse=True)
def _explicit_enabled_advisory(monkeypatch):
    """These tests exercise the advisory path, independent of ambient review slots."""
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.delenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", raising=False)


def _make_agent_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal fake agent-repo layout (ouroboros/__init__.py present)."""
    (tmp_path / "ouroboros").mkdir()
    (tmp_path / "ouroboros" / "__init__.py").write_text("")
    return tmp_path


def _init_git_repo(repo: pathlib.Path) -> None:
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


def _write_release_files(repo: pathlib.Path, *, version: str, pyproject_version: str | None = None, minor_rows: int = 1) -> None:
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "VERSION").write_text(version + "\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "ouroboros"\nversion = "' + (pyproject_version or version.replace("-rc.", "rc")) + '"\n',
        encoding="utf-8",
    )
    rows = [
        f"| 5.{idx}.0 | 2026-05-15 | row {idx} |"
        for idx in range(30, 30 - minor_rows, -1)
    ]
    if not any(row.startswith(f"| {version} |") for row in rows):
        rows.insert(0, f"| {version} | 2026-05-15 | current row |")
    (repo / "README.md").write_text(
        f"[![Version {version}](https://img.shields.io/badge/version-{version.replace('-', '--')}-green.svg)](VERSION)\n\n"
        "## Version History\n\n| Version | Date | Description |\n|---------|------|-------------|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        f"# Ouroboros v{version} — Architecture & Reference\n",
        encoding="utf-8",
    )


class TestSyntaxPreflightHelper:
    def test_non_agent_repo_skipped(self, tmp_path):
        """Target repos without `ouroboros/__init__.py` bypass the gate entirely."""
        (tmp_path / "broken.py").write_text("def foo(:\n")
        assert _syntax_preflight_staged_py_files(tmp_path, ["broken.py"]) is None

    def test_agent_repo_valid_passes(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "good.py").write_text("def foo():\n    return 1\n")
        assert _syntax_preflight_staged_py_files(repo, ["good.py"]) is None

    def test_agent_repo_syntax_error_blocks(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "broken.py").write_text("def foo(:\n")
        out = _syntax_preflight_staged_py_files(repo, ["broken.py"])
        assert out is not None
        assert "PREFLIGHT_BLOCKED" in out
        assert "broken.py:1:" in out

    def test_multiple_errors_all_reported(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "a.py").write_text("def a(:\n")
        (repo / "b.py").write_text("x = (\n")
        out = _syntax_preflight_staged_py_files(repo, ["a.py", "b.py"])
        assert "a.py" in (out or "")
        assert "b.py" in (out or "")

    def test_non_py_files_ignored(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "README.md").write_text("# bad:\ndef foo(:")
        (repo / "data.json").write_text("not even python")
        assert _syntax_preflight_staged_py_files(
            repo, ["README.md", "data.json"]
        ) is None

    def test_staged_deletion_tolerated(self, tmp_path):
        """A staged deletion has no on-disk file — must not raise."""
        repo = _make_agent_repo(tmp_path)
        assert _syntax_preflight_staged_py_files(repo, ["deleted.py"]) is None

    def test_no_pycache_created(self, tmp_path):
        """compile() with dont_inherit=True must not materialise __pycache__."""
        repo = _make_agent_repo(tmp_path)
        (repo / "good.py").write_text("x = 1\n")
        _syntax_preflight_staged_py_files(repo, ["good.py"])
        assert not any("__pycache__" in str(p) for p in repo.rglob("*"))

    def test_empty_path_list_passes(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        assert _syntax_preflight_staged_py_files(repo, []) is None

    def test_mixed_valid_and_broken(self, tmp_path):
        repo = _make_agent_repo(tmp_path)
        (repo / "ok.py").write_text("x = 1\n")
        (repo / "bad.py").write_text("def f(:\n")
        out = _syntax_preflight_staged_py_files(repo, ["ok.py", "bad.py"])
        assert out is not None
        assert "bad.py" in out
        # ok.py must not appear in the error list
        assert "- ok.py" not in out

    def test_null_byte_source_is_blocked_as_preflight(self, tmp_path):
        """`compile()` raises ValueError (not SyntaxError) when a source file
        contains null bytes. The preflight helper must still treat that as
        a blocking preflight so the SDK call is skipped and the agent gets
        an actionable PREFLIGHT_BLOCKED message instead of an opaque
        ADVISORY_ERROR. Regression for final-round Advisory #1."""
        repo = _make_agent_repo(tmp_path)
        (repo / "null.py").write_bytes(b"x = 1\x00\n")
        out = _syntax_preflight_staged_py_files(repo, ["null.py"])
        assert out is not None, (
            "null-byte source must trip the preflight; got None"
        )
        assert "null.py" in out
        assert "PREFLIGHT_BLOCKED" in out


class TestPreflightGatesBeforeSDK:
    """Verify the preflight short-circuit fires BEFORE the paid critic call
    (native episode), saving the spend when staged `.py` files cannot compile.
    (The retired Claude-SDK skipif is gone: the successor path needs no SDK.)"""

    def test_syntax_error_returns_before_sdk(self, tmp_path, monkeypatch):
        """End-to-end: a staged syntactically broken .py file returns
        PREFLIGHT_BLOCKED without `run_readonly` being called."""
        from ouroboros.tools import claude_advisory_review as adv

        # Fake agent-repo layout.
        repo = _make_agent_repo(tmp_path)
        (repo / "broken.py").write_text("def foo(:\n")

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")

        monkeypatch.setattr(
            adv, "_get_staged_diff",
            lambda repo_dir, paths=None: "diff --git a/broken.py b/broken.py",
        )
        monkeypatch.setattr(
            adv, "_get_changed_file_list",
            lambda repo_dir, paths=None: "M  broken.py",
        )
        monkeypatch.setattr(
            adv, "build_advisory_changed_context",
            lambda repo_dir, changed_files_text, paths=None, exclude_paths=None:
                (["broken.py"], "(touched pack)", []),
        )

        sdk_called = {"n": 0}

        def _fake_run_readonly(*args, **kwargs):
            sdk_called["n"] += 1
            raise AssertionError("SDK should NOT be called when preflight blocks")

        monkeypatch.setattr(
            advisory, "_run_advisory_native",
            lambda prompt, repo_dir, ctx_, slot, model, **_: (
                _fake_run_readonly(), model,
            ),
        )

        fake_ctx = mock.MagicMock()
        items, raw, model, prompt_chars = adv._run_claude_advisory(
            repo_dir=repo,
            commit_message="test",
            ctx=fake_ctx,
            goal="",
            scope="",
            paths=None,
        )

        assert sdk_called["n"] == 0, "SDK was unexpectedly invoked"
        assert items == []
        assert "PREFLIGHT_BLOCKED" in raw
        assert "broken.py" in raw
        assert model == ""
        assert prompt_chars == 0

    def test_valid_python_does_not_short_circuit(self, tmp_path, monkeypatch):
        """Clean .py files must pass through preflight and reach the SDK step
        (we still short-circuit there via a fake SDK to avoid network calls)."""
        from ouroboros.tools import claude_advisory_review as adv

        repo = _make_agent_repo(tmp_path)
        (repo / "good.py").write_text("def foo():\n    return 1\n")

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")

        monkeypatch.setattr(
            adv, "_get_staged_diff",
            lambda repo_dir, paths=None: "diff --git a/good.py b/good.py",
        )
        monkeypatch.setattr(
            adv, "_get_changed_file_list",
            lambda repo_dir, paths=None: "M  good.py",
        )
        monkeypatch.setattr(
            adv, "build_advisory_changed_context",
            lambda repo_dir, changed_files_text, paths=None, exclude_paths=None:
                (["good.py"], "(touched pack)", []),
        )
        monkeypatch.setattr(
            adv, "_build_advisory_prompt",
            lambda *args, **kwargs: "(tiny prompt)",
        )

        # Fake SDK that returns a canned empty-items response to confirm
        # we reached the SDK path.
        sdk_called = {"n": 0}

        class _Result:
            success = False  # triggers early return via `_format_advisory_error`
            error = "fake"
            stderr_tail = ""
            session_id = ""
            cost_usd = 0.0
            usage = {}
            result_text = ""

        def _fake_run_readonly(*args, **kwargs):
            sdk_called["n"] += 1
            return _Result()

        # run_readonly is imported INSIDE _run_claude_advisory, not at module top,
        # so patch the source symbol.
        monkeypatch.setattr(
            advisory, "_run_advisory_native",
            lambda prompt, repo_dir, ctx_, slot, model, **_: (
                _fake_run_readonly(), model,
            ),
        )

        fake_ctx = mock.MagicMock()
        items, raw, model, prompt_chars = adv._run_claude_advisory(
            repo_dir=repo,
            commit_message="test",
            ctx=fake_ctx,
            goal="",
            scope="",
            paths=None,
        )

        assert sdk_called["n"] == 1, "SDK should have been invoked for valid .py"
        assert "PREFLIGHT_BLOCKED" not in raw


class TestHandleAdvisoryPreReviewSurfacesPreflightBlocked:
    """End-to-end integration test for the tool entrypoint (v4.38.0 fix).

    `_handle_advisory_pre_review` must special-case the `PREFLIGHT_BLOCKED`
    sentinel returned by `_run_claude_advisory`. Without this, the sentinel
    falls through to the `parse_failure` branch and the agent sees a
    generic "advisory output could not be parsed" message instead of the
    concrete SyntaxError location — hiding the real bug.
    """

    def test_preflight_blocked_surfaces_as_explicit_status(self, tmp_path, monkeypatch):
        """Feeding a broken .py into the full tool handler produces
        status='preflight_blocked' with the raw error text in the response."""
        from ouroboros.tools import claude_advisory_review as adv
        import json as _json

        # The handler short-circuits with `bypassed` if ANTHROPIC_API_KEY is
        # missing — set a fake key so we reach the preflight branch under test.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")

        # Patch the low-level `_run_claude_advisory` to return the exact
        # sentinel the preflight helper produces. This isolates the
        # `_handle_advisory_pre_review` routing logic under test.
        preflight_sentinel = (
            "⚠️ PREFLIGHT_BLOCKED: syntax errors:\n"
            "- broken.py:1: invalid syntax\n\n"
            "Fix the syntax error(s) above and re-run advisory_pre_review. "
            "Claude SDK advisory was skipped to save budget."
        )
        monkeypatch.setattr(
            adv, "_run_claude_advisory",
            lambda repo_dir, commit_message, ctx, **kwargs:
                ([], preflight_sentinel, "", 0),
        )

        # Stub out all the durable-state plumbing so the handler can run
        # without a real drive_root layout.
        monkeypatch.setattr(
            adv, "_get_staged_diff",
            lambda repo_dir, paths=None: "diff --git a/broken.py b/broken.py",
        )
        monkeypatch.setattr(
            adv, "_get_changed_file_list",
            lambda repo_dir, paths=None: "M  broken.py",
        )
        monkeypatch.setattr(
            adv, "check_worktree_readiness",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            adv, "_check_worktree_version_sync_shared",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(
            adv, "compute_snapshot_hash", lambda *args, **kwargs: "deadbeef",
        )

        # Minimal ToolContext stub with the attributes the handler reads.
        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = str(tmp_path)
        fake_ctx.drive_root = tmp_path
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-test"

        # Call via the module-level convenience signature. If the registry
        # decorator name differs, fall back to calling the handler directly.
        result_raw = adv._handle_advisory_pre_review(
            fake_ctx, commit_message="test commit"
        )

        result = _json.loads(result_raw)
        assert result.get("status") == "preflight_blocked", (
            f"Expected status=preflight_blocked, got {result.get('status')!r}. "
            f"Full result: {result!r}"
        )
        assert "PREFLIGHT_BLOCKED" in (result.get("error") or "")
        assert "broken.py" in (result.get("error") or "")
        # The generic parse_failure message MUST NOT appear — that would mean
        # the sentinel was misclassified.
        assert "parse_failure" not in result.get("status", "")
        assert "parseable checklist items" not in (result.get("error") or "")

    def test_p9_history_limit_blocks_before_sdk(self, tmp_path, monkeypatch):
        from ouroboros.tools import claude_advisory_review as adv
        import json as _json

        repo_root = tmp_path / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        repo = _make_agent_repo(repo_root)
        _init_git_repo(repo)
        (repo / "logs").mkdir(exist_ok=True)
        _write_release_files(repo, version="5.99.0-rc.1", minor_rows=6)
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *args, **kwargs: [])

        def forbidden_sdk(*args, **kwargs):
            raise AssertionError("Claude SDK should not run when P9 preflight blocks")

        monkeypatch.setattr(adv, "_run_claude_advisory", forbidden_sdk)

        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = repo
        fake_ctx.drive_root = repo
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-p9"

        result_raw = adv._handle_advisory_pre_review(
            fake_ctx,
            commit_message="v5.99.0-rc.1: test",
            skip_tests=True,
        )
        result = _json.loads(result_raw)
        assert result["status"] == "preflight_blocked"
        assert "Version History exceeds" in result["error"]

    def test_release_metadata_preflight_blocks_stale_uv_lock(self, tmp_path):
        from ouroboros.tools import claude_advisory_review as adv

        repo = _make_agent_repo(tmp_path)
        _write_release_files(repo, version="5.99.0-rc.1")
        (repo / "uv.lock").write_text(
            '[[package]]\nname = "ouroboros"\nversion = "5.98.0"\n'
            'source = { editable = "." }\n',
            encoding="utf-8",
        )

        result = adv._release_metadata_preflight(
            repo,
            "v5.99.0-rc.1: release",
            ["VERSION"],
        )

        assert result is not None
        assert "uv.lock" in result

    def test_release_metadata_preflight_blocks_stale_web_package_lock(self, tmp_path):
        """MAJOR-1 (rc.15 review): the lockfile is a release carrier the sync
        writes; the admission preflight must read it like its siblings, so a
        root-entry desync is a typed PREFLIGHT_BLOCKED naming the file (this
        preflight is also the SM1 stand check's carrier gate)."""
        from ouroboros.tools import claude_advisory_review as adv

        repo = _make_agent_repo(tmp_path)
        _write_release_files(repo, version="5.99.0-rc.1")
        (repo / "web").mkdir()
        (repo / "web" / "package.json").write_text(
            '{\n  "name": "ouroboros-web",\n  "version": "5.99.0-rc.1"\n}\n', encoding="utf-8",
        )
        (repo / "web" / "package-lock.json").write_text(
            '{\n  "name": "ouroboros-web",\n  "version": "5.98.0",\n  "lockfileVersion": 3,\n'
            '  "packages": {\n    "": {\n      "name": "ouroboros-web",\n      "version": "5.98.0"\n'
            '    }\n  }\n}\n',
            encoding="utf-8",
        )

        result = adv._release_metadata_preflight(repo, "v5.99.0-rc.1: release", ["VERSION"])

        assert result is not None and "PREFLIGHT_BLOCKED" in result
        assert 'web/package-lock.json (expected both root "version" entries = "5.99.0-rc.1")' in result

    def test_changed_diff_without_version_blocks_before_sdk(self, tmp_path, monkeypatch):
        from ouroboros.tools import claude_advisory_review as adv
        import json as _json

        repo_root = tmp_path / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        repo = _make_agent_repo(repo_root)
        _init_git_repo(repo)
        (repo / "logs").mkdir(exist_ok=True)
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "VERSION").write_text("5.99.0-rc.1\n", encoding="utf-8")
        (repo / "README.md").write_text("## Version History\n", encoding="utf-8")
        (repo / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v5.99.0-rc.1\n", encoding="utf-8")
        (repo / "pyproject.toml").write_text('[project]\nversion = "5.99.0rc1"\n', encoding="utf-8")
        (repo / "ouroboros" / "feature.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *args, **kwargs: [])

        def forbidden_sdk(*args, **kwargs):
            raise AssertionError("Claude SDK should not run when VERSION is missing from changed diff")

        monkeypatch.setattr(adv, "_run_claude_advisory", forbidden_sdk)

        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = repo
        fake_ctx.drive_root = repo
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-missing-version"

        result_raw = adv._handle_advisory_pre_review(
            fake_ctx,
            commit_message="feat: add feature",
            paths=["ouroboros/feature.py"],
            skip_tests=True,
        )
        result = _json.loads(result_raw)
        assert result["status"] == "preflight_blocked"
        assert "VERSION is not in scope" in result["error"]

    def test_doc_only_diff_without_version_reaches_the_critic(self, tmp_path, monkeypatch):
        """Doc-only carve (finding W3A-F1). The commit gate already exempts a
        doc-only diff from its compensating preflight, while this admission
        blocked the very same diff — so a doc-only change could never obtain a
        FRESH advisory verdict on any install and the standard flow degraded to
        the audited bypass. The critic must now actually be dispatched."""
        from ouroboros.tools import claude_advisory_review as adv
        import json as _json

        repo_root = tmp_path / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        repo = _make_agent_repo(repo_root)
        _init_git_repo(repo)
        (repo / "logs").mkdir(exist_ok=True)
        _write_release_files(repo, version="5.99.0-rc.1")
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
        (repo / "docs" / "NOTES.md").write_text("# notes\n", encoding="utf-8")
        subprocess.run(["git", "add", "docs/NOTES.md"], cwd=str(repo), check=True)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *args, **kwargs: [])

        dispatched: list = []

        def fake_run(repo_dir, commit_message, ctx, **kwargs):
            dispatched.append(commit_message)
            return [], "[]", "fake-model", 2

        monkeypatch.setattr(adv, "_run_claude_advisory", fake_run)

        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = repo
        fake_ctx.drive_root = repo
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-doc-only"

        result = _json.loads(adv._handle_advisory_pre_review(
            fake_ctx,
            commit_message="docs: notes",
            paths=["docs/NOTES.md"],
            skip_tests=True,
        ))
        assert result["status"] != "preflight_blocked", result
        assert dispatched == ["docs: notes"]

    def test_doc_only_carve_is_the_commit_gate_classifier(self, tmp_path):
        """One detector, not two: the carve is decided by the SAME
        ``_diff_is_doc_only`` the commit gate applies, so the gates cannot
        drift. A code file, a mixed diff and a doc under ``tests/`` all keep
        the block; the carve never touches the carrier-coherence checks, which
        still run the moment VERSION IS in scope."""
        from ouroboros.commit_admission import release_metadata_preflight
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
            return release_metadata_preflight(repo, "m", paths)

        assert _preflight(["docs/NOTES.md"]) is None
        assert _diff_is_doc_only(["docs/NOTES.md"]) is True

        for scope in (["ouroboros/feature.py"],
                      ["docs/NOTES.md", "ouroboros/feature.py"],
                      ["tests/NOTES.md"]):
            assert _diff_is_doc_only(scope) is False, scope
            blocked = _preflight(scope)
            assert blocked is not None and "VERSION is not in scope" in blocked, scope

        # VERSION in scope: the carve is out of the way and the coherence
        # checks own the answer again (stale carrier -> still blocked).
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "ouroboros"\nversion = "5.98.0"\n', encoding="utf-8")
        stale = _preflight(["VERSION", "docs/NOTES.md"])
        assert stale is not None and "pyproject.toml" in stale

    def test_advisory_auto_syncs_release_metadata_when_version_staged(self, tmp_path, monkeypatch):
        from ouroboros.tools import claude_advisory_review as adv

        repo_root = tmp_path / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        repo = _make_agent_repo(repo_root)
        _init_git_repo(repo)
        (repo / "logs").mkdir(exist_ok=True)
        _write_release_files(repo, version="5.99.0-rc.1", pyproject_version="5.22.0rc1", minor_rows=1)
        subprocess.run(["git", "add", "VERSION"], cwd=str(repo), check=True)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *args, **kwargs: [])

        def fake_run(repo_dir, commit_message, ctx, **kwargs):
            return [], "[]", "fake-model", 2

        monkeypatch.setattr(adv, "_run_claude_advisory", fake_run)

        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = repo
        fake_ctx.drive_root = repo
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-sync"

        adv._handle_advisory_pre_review(
            fake_ctx,
            commit_message="v5.99.0-rc.1: test",
            skip_tests=True,
        )
        pyproject_text = (repo / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "5.99.0rc1"' in pyproject_text
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "pyproject.toml" in status
        events = (repo / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert "release_metadata_auto_synced" in events


class TestPreflightBlockedPersistence:
    """A preflight_blocked outcome must be recorded as a durable
    AdvisoryRunRecord (v4.39.0). Without this, `review_status` and the
    `Review Continuity` context cannot report the block reason after a
    restart, and the next advisory call has no history of the preflight
    skip. Regression for Pass 2 advisory #5.
    """

    def test_preflight_blocked_persists_durable_run(self, tmp_path, monkeypatch):
        from ouroboros.tools import claude_advisory_review as adv
        import json as _json

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        preflight_sentinel = (
            "⚠️ PREFLIGHT_BLOCKED: syntax errors:\n"
            "- broken.py:1: invalid syntax\n"
        )
        monkeypatch.setattr(
            adv, "_run_claude_advisory",
            lambda repo_dir, commit_message, ctx, **kwargs:
                ([], preflight_sentinel, "", 0),
        )
        monkeypatch.setattr(
            adv, "_get_staged_diff",
            lambda repo_dir, paths=None: "diff",
        )
        monkeypatch.setattr(
            adv, "_get_changed_file_list",
            lambda repo_dir, paths=None: "M  broken.py",
        )
        monkeypatch.setattr(
            adv, "check_worktree_readiness",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            adv, "_check_worktree_version_sync_shared",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(
            adv, "compute_snapshot_hash",
            lambda *args, **kwargs: "preflight-test-hash",
        )

        fake_ctx = mock.MagicMock()
        fake_ctx.repo_dir = str(tmp_path)
        fake_ctx.drive_root = tmp_path
        fake_ctx.emit_progress_fn = lambda *a, **kw: None
        fake_ctx.task_id = "t-persist"

        result_raw = adv._handle_advisory_pre_review(
            fake_ctx, commit_message="test"
        )
        result = _json.loads(result_raw)
        assert result["status"] == "preflight_blocked"

        # Durable state must now contain an AdvisoryRunRecord with
        # status="preflight_blocked" for this snapshot hash.
        from ouroboros.review_state import load_state
        state = load_state(tmp_path)
        matching = [r for r in state.advisory_runs
                    if r.snapshot_hash == "preflight-test-hash"]
        assert len(matching) == 1, (
            f"Expected exactly one durable run for the preflight snapshot; "
            f"got {len(matching)}. All runs: "
            f"{[(r.snapshot_hash, r.status) for r in state.advisory_runs]}"
        )
        rec = matching[0]
        assert rec.status == "preflight_blocked"
        assert "PREFLIGHT_BLOCKED" in (rec.raw_result or "")
        assert rec.items == []  # SDK never called → no items
        assert rec.prompt_chars == 0
        assert rec.model_used == ""

    def test_preflight_blocked_record_excluded_from_fresh(self, tmp_path, monkeypatch):
        """`is_fresh` must return False for a preflight_blocked record so a
        subsequent `repo_commit` does NOT proceed with the SDK-skipped state;
        the user must fix the syntax error and re-run advisory first."""
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, save_state, load_state,
        )

        state = AdvisoryReviewState()
        state.add_run(AdvisoryRunRecord(
            snapshot_hash="pf-hash",
            commit_message="m",
            status="preflight_blocked",
            ts="2026-04-18T00:00:00Z",
            raw_result="⚠️ PREFLIGHT_BLOCKED: syntax errors",
        ))
        save_state(tmp_path, state)
        reloaded = load_state(tmp_path)

        assert reloaded.is_fresh("pf-hash") is False, (
            "A preflight_blocked record must NOT count as fresh — that would "
            "let repo_commit proceed without the SDK ever seeing the code."
        )


class TestTestsPreflightProofBinding:
    """The commit-admission SSOT owns the green-run -> Q10 proof coupling:
    a green preflight ALWAYS records the managed proof (else the managed gate
    pays a second identical full run), and the proof is only ever recorded off
    a green run (else it forges admission evidence)."""

    def _ctx(self, tmp_path):
        from types import SimpleNamespace
        return SimpleNamespace(
            task_id="t", task_metadata={}, repo_dir=str(tmp_path),
            emit_progress_fn=lambda *_a, **_k: None,
        )

    def test_red_run_returns_error_and_records_no_proof(self, tmp_path):
        from ouroboros.commit_admission import run_tests_preflight_with_proof

        ctx = self._ctx(tmp_path)
        err = run_tests_preflight_with_proof(ctx, runner=lambda c: "FAILED: 2 failed")
        assert err == "FAILED: 2 failed"
        assert not getattr(ctx, "_managed_tests_proof_trees", None)

    def test_green_run_records_the_managed_proof(self, tmp_path, monkeypatch):
        from ouroboros.commit_admission import run_tests_preflight_with_proof
        import supervisor.update_merge as um

        ctx = self._ctx(tmp_path)
        recorded = []
        monkeypatch.setattr(um, "record_managed_tests_proof",
                            lambda c: recorded.append(c) or "tree-sha")
        assert run_tests_preflight_with_proof(ctx, runner=lambda c: None) is None
        assert recorded == [ctx]
