"""Behavioral tests for the git+review pipeline.

Renamed in v5.15.x from ``test_phase7_pipeline.py`` — the file is the
canonical behavioral suite for the modern commit pipeline + operational
resilience, not a one-shot migration test. The previous name pinned a
historical migration phase that has long since shipped.

Tests:
- repo_write single-file and multi-file modes
- repo_write + repo_commit workflow
- Unified pre-commit review gate (preflight, parse, quorum)
- Blocked review leaves files on disk but unstaged
- review_rebuttal parameter
- configure_remote failure surfacing
- configure_remote credential-helper wiring
- Auto-rescue only reports committed when commit actually happened
- repo_write in CORE_TOOL_NAMES
- Review history building
"""
import importlib
import inspect
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


import re as _re


def _critical_triad_items():
    """Parse critical triad checklist item ids from the frozen CHECKLISTS.md.

    Used to parametrize the NW-2 advisory-downgrade guardrail over EVERY
    critical item (not just ``code_quality``), so a per-item always-block
    hardcode against owner-chosen advisory enforcement (the 58a52c4 class)
    fails the suite. Falls back to a known critical pair if parsing fails so
    the guardrail never silently degrades to zero cases.
    """
    try:
        review = importlib.import_module("ouroboros.tools.review")
        section = review._load_checklist_section()
        items = []
        for line in section.splitlines():
            m = _re.match(r"^\s*\|\s*\d+\s*\|\s*([a-z0-9_]+)\s*\|.*\|\s*critical\s*\|\s*$", line)
            if m:
                items.append(m.group(1))
        # version_bump (item 8) is the incident's triad item; ensure it's present.
        if "version_bump" in items and len(items) >= 5:
            return items
    except Exception:
        pass
    return ["bible_compliance", "code_quality", "version_bump", "security_issues"]


def _get_git_module():
    return importlib.import_module("ouroboros.tools.git")


def _get_review_module():
    return importlib.import_module("ouroboros.tools.review")


def _get_registry_module():
    return importlib.import_module("ouroboros.tools.registry")


def _get_git_ops_module():
    return importlib.import_module("supervisor.git_ops")


def _make_ctx(tmp_path):
    """Create a minimal ToolContext with a temporary git repo."""
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
    return ToolContext(repo_dir=repo, drive_root=drive)


@pytest.fixture
def git_ctx(tmp_path):
    """Yield ``(git_module, ToolContext)`` — the canonical git pipeline setup."""
    return _get_git_module(), _make_ctx(tmp_path)


@pytest.fixture
def review_ctx(tmp_path):
    """Yield ``(review_module, ToolContext)``."""
    return _get_review_module(), _make_ctx(tmp_path)


def test_managed_resolver_stages_tracked_binary_from_official_merge(tmp_path, monkeypatch):
    git_mod = _get_git_module()
    ctx = _make_ctx(tmp_path)
    binary = pathlib.Path(ctx.repo_dir) / "native.so"
    binary.write_bytes(b"old\x00payload")
    subprocess.run(["git", "add", "-f", "native.so"], cwd=ctx.repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "track binary"], cwd=ctx.repo_dir, check=True)
    binary.write_bytes(b"official\x00payload")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda _ctx: True)

    _paths, _advisory_paths, error = git_mod._stage_candidate_for_review(
        ctx,
        "review assisted update",
        0.0,
        paths=None,
        came_from_detached_checkout=False,
    )

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ctx.repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert error is None
    assert "native.so" in staged


# --- repo_write tool registration ---

class TestRepoWriteRegistration:
    def test_repo_write_registered(self):
        from ouroboros.tools import core as core_mod
        names = [t.name for t in core_mod.get_tools()]
        assert "write_file" in names

    def test_repo_write_in_core_tool_names(self):
        registry = _get_registry_module()
        assert "write_file" in registry.CORE_TOOL_NAMES

    def test_repo_write_schema_has_files_param(self):
        from ouroboros.tools import core as core_mod
        tools = core_mod.get_tools()
        rw = next(t for t in tools if t.name == "write_file")
        props = rw.schema["parameters"]["properties"]
        assert "files" in props
        assert props["files"]["type"] == "array"

    def test_repo_commit_has_review_rebuttal(self):
        git_mod = _get_git_module()
        tools = git_mod.get_tools()
        rc = next(t for t in tools if t.name == "commit_reviewed")
        props = rc.schema["parameters"]["properties"]
        assert "review_rebuttal" in props


# --- repo_write behavioral tests ---

class TestRepoWriteSingleFile:
    def test_single_file_write(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, path="hello.py", content="print('hello')")
        assert "Written 1 file" in result
        assert "NOT committed" in result
        assert (ctx.repo_dir / "hello.py").read_text() == "print('hello')"

    def test_single_file_creates_directories(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, path="deep/nested/file.py", content="x = 1")
        assert "Written 1 file" in result
        assert (ctx.repo_dir / "deep" / "nested" / "file.py").exists()

    def test_rejects_empty_args(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx)
        assert "WRITE_ERROR" in result

    def test_rejects_compaction_marker(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, path="x.py", content="<<CONTENT_OMITTED something")
        assert "WRITE_ERROR" in result
        assert "compaction marker" in result


class TestRepoWriteMultiFile:
    def test_multi_file_write(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, files=[
            {"path": "a.py", "content": "# a"},
            {"path": "b.py", "content": "# b"},
        ])
        assert "Written 2 file" in result
        assert (ctx.repo_dir / "a.py").read_text() == "# a"
        assert (ctx.repo_dir / "b.py").read_text() == "# b"

    def test_multi_file_rejects_empty_path(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, files=[{"path": "", "content": "x"}])
        assert "WRITE_ERROR" in result

    def test_multi_file_blocks_safety_critical(self, git_ctx, monkeypatch):
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(ctx, files=[
            {"path": "ok.py", "content": "x"},
            {"path": "BIBLE.md", "content": "hacked"},
        ])
        assert "CORE_PROTECTION_BLOCKED" in result

    def test_files_param_takes_priority(self, git_ctx):
        git_mod, ctx = git_ctx
        result = git_mod._repo_write(
            ctx, path="ignored.py", content="ignored",
            files=[{"path": "used.py", "content": "used"}],
        )
        assert "Written 1 file" in result
        assert (ctx.repo_dir / "used.py").exists()
        assert not (ctx.repo_dir / "ignored.py").exists()


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
    # longer refused; CHECKLISTS.md item 6 (tests_affected) owns coverage.
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
def test_preflight_check(case_id, message, staged_files, expected):
    review = _get_review_module()
    result = review._preflight_check(message, staged_files, "/tmp")
    if expected is None:
        assert result is None, f"expected pass, got: {result!r}"
    else:
        assert result is not None
        for needle in expected:
            assert needle in result, f"missing {needle!r} in: {result!r}"


_PARSE_REVIEW_JSON_CASES = [
    (
        "plain_json",
        '[{"item":"x","verdict":"PASS","severity":"critical","reason":"ok"}]',
        lambda r: r is not None and len(r) == 1,
    ),
    (
        "markdown_fenced",
        '```json\n[{"item":"x","verdict":"FAIL","severity":"advisory","reason":"bad"}]\n```',
        lambda r: r is not None and r[0]["verdict"] == "FAIL",
    ),
    (
        "text_around_json",
        'Here is my review:\n[{"item":"x","verdict":"PASS","severity":"critical","reason":"ok"}]\nDone.',
        lambda r: r is not None,
    ),
    (
        "invalid_json",
        "not json at all",
        lambda r: r is None,
    ),
]


@pytest.mark.parametrize(
    "case_id,data,predicate",
    _PARSE_REVIEW_JSON_CASES,
    ids=[c[0] for c in _PARSE_REVIEW_JSON_CASES],
)
def test_parse_review_json(case_id, data, predicate):
    review = _get_review_module()
    assert predicate(review._parse_review_json(data))


class TestReviewHistoryBuilding:
    def test_empty_history(self):
        review = _get_review_module()
        result = review._build_review_history_section([])
        assert result == ""

    def test_history_with_entries(self):
        review = _get_review_module()
        history = [{
            "attempt": 1,
            "commit_message": "test commit",
            "critical": ["[model] item: reason"],
            "advisory": [],
        }]
        result = review._build_review_history_section(history)
        assert "Round 1" in result
        assert "test commit" in result
        assert "CRITICAL" in result


class TestReviewQuorumLogic:
    # ``test_review_models_configured`` was removed in v5.8.3-rc.5 — the
    # ``len(get_review_models()) >= 2`` quorum assertion is already covered
    # in ``tests/test_settings_effort.py`` (3 cases). This class keeps the
    # checklist-path / loader smoke tests below which are unique to the
    # phase-7 pipeline contract.

    def test_checklist_path_exists(self):
        review = _get_review_module()
        assert review._CHECKLISTS_PATH.exists()

    def test_load_checklist_succeeds(self):
        review = _get_review_module()
        section = review._load_checklist_section()
        assert "bible_compliance" in section
        assert "code_quality" in section


class TestReviewEnforcementModes:
    @staticmethod
    def _fake_result(*review_texts):
        return json.dumps({
            "results": [
                {
                    "model": f"model-{idx}",
                    "verdict": "PASS",
                    "text": text,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_estimate": 0.0,
                }
                for idx, text in enumerate(review_texts, start=1)
            ]
        })

    @staticmethod
    def _mock_staged(monkeypatch, review_mod, changed_files="x.py", diff_text="diff --cached",
                     name_status_files=None):
        """Mock git commands for _run_unified_review.

        name_status_files: if provided, used as the --name-status output.
        Defaults to converting changed_files lines to "M  path" format.
        """
        if name_status_files is None:
            # Convert plain filenames to M\tpath format (what git --name-status emits)
            name_status_files = "\n".join(
                f"M\t{f.strip()}" for f in changed_files.splitlines() if f.strip()
            )

        def _fake_run_cmd(cmd, cwd=None):
            cmd = list(cmd)
            if cmd[:5] == ["git", "diff", "--cached", "--name-status"]:
                return name_status_files
            if cmd[:4] == ["git", "diff", "--cached", "--name-only"]:
                return changed_files
            if cmd[:3] == ["git", "diff", "--cached"]:
                return diff_text
            return ""
        monkeypatch.setattr(review_mod, "run_cmd", _fake_run_cmd)
        # The triad now reads its change evidence through the hardened
        # capture_staged_diff seam (imported function-locally), not run_cmd.
        import ouroboros.tools.review_binary_context as _rbc
        monkeypatch.setattr(_rbc, "capture_staged_diff",
                            lambda _repo, *, unified=3: diff_text)

    def test_blocking_mode_blocks_critical_findings(self, review_ctx, monkeypatch):
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"code_quality","verdict":"FAIL","severity":"critical","reason":"broken"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is not None
        assert "REVIEW_BLOCKED" in result

    def test_advisory_mode_downgrades_critical_findings(self, review_ctx, monkeypatch):
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"code_quality","verdict":"FAIL","severity":"critical","reason":"broken"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            isinstance(w, str) and "critical review findings did not block commit" in w.lower()
            for w in ctx._review_advisory
        )
        assert any(
            (isinstance(w, dict) and w.get("reason") == "broken")
            or (isinstance(w, str) and "broken" in w)
            for w in ctx._review_advisory
        )
        # Anti-thrashing state survives an advisory pass-through of critical
        # findings: repeats on the next attempt must still be recognized.
        assert ctx._review_iteration_count == 1

    @pytest.mark.parametrize("failure", ["nonzero_rc", "non_utf8_rc"])
    def test_uncapturable_staged_diff_blocks_instead_of_reviewing_a_placeholder(
        self, review_ctx, monkeypatch, failure
    ):
        """The triad's change evidence is the staged diff, and the old ``run_cmd``
        capture fell back to a ``(failed to get staged diff)`` STRING that a full,
        authoritative review then ran against — findings about a diff nobody has.
        It now goes through the hardened ``capture_staged_diff``; when git cannot
        produce the diff the review fails closed in blocking mode (no reviewer is
        dispatched), exactly like a checklist-load or reviewer-config infra
        failure."""
        review, ctx = review_ctx
        # name-status / name-only still answer so we reach the diff capture, but
        # the content capture is what fails.
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        import ouroboros.tools.review_binary_context as rbc

        def broken(_repo, *, unified=3):
            detail = "fatal: bad object" if failure == "nonzero_rc" else "fatal: \udcffbad"
            raise rbc.StagedDiffUnavailable(f"staged diff capture failed (rc 128): {detail}")

        monkeypatch.setattr(rbc, "capture_staged_diff", broken)
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        dispatched = []
        monkeypatch.setattr(
            review, "_handle_multi_model_review",
            lambda *a, **k: dispatched.append(True) or self._fake_result("[]", "[]"),
        )

        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)

        assert result is not None and "REVIEW_BLOCKED" in result
        assert "staged diff" in result.lower()
        assert "failed to get staged diff" not in result  # no placeholder anywhere
        assert ctx._last_review_block_reason == "infra_failure"
        assert dispatched == [], "no reviewer may run without the staged diff"

    def test_uncapturable_staged_diff_is_advisory_skip_not_placeholder_review(
        self, review_ctx, monkeypatch
    ):
        """Advisory counterpart: review is non-blocking, so an infra failure to
        capture the diff skips the triad with a durable warning instead of feeding
        a placeholder into it. The commit proceeds (``None``) and the skip is
        recorded, never a review of ``(failed to get staged diff)``."""
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        import ouroboros.tools.review_binary_context as rbc
        monkeypatch.setattr(
            review, "_handle_multi_model_review",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("triad must not run")),
        )

        def broken(_repo, *, unified=3):
            raise rbc.StagedDiffUnavailable("staged diff capture failed: boom")

        monkeypatch.setattr(rbc, "capture_staged_diff", broken)
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)

        assert result is None
        assert ctx._last_review_block_reason == "infra_failure"
        assert any(
            isinstance(w, str) and "staged diff capture failed" in w.lower()
            for w in ctx._review_advisory
        )

    def test_triad_one_pass_fit_removes_only_duplicated_context(self, review_ctx, monkeypatch):
        """Oversized triad evidence is compacted before its single dispatch."""
        review, ctx = review_ctx
        huge_diff = "diff --git a/x.py b/x.py\n" + ("+changed line\n" * 190_000)
        compact_diff = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+new\n"

        def fake_run_cmd(cmd, cwd=None):
            cmd = list(cmd)
            if cmd == ["git", "diff", "--cached", "--name-status"]:
                return "M\tx.py"
            if cmd == ["git", "diff", "--cached", "--name-only"]:
                return "x.py"
            if cmd == ["git", "diff", "--cached", "-U0"]:
                return compact_diff
            if cmd == ["git", "diff", "--cached"]:
                return huge_diff
            return ""

        captured = {}
        monkeypatch.setattr(review, "run_cmd", fake_run_cmd)
        import ouroboros.tools.review_binary_context as _rbc
        monkeypatch.setattr(
            _rbc, "capture_staged_diff",
            lambda _repo, *, unified=3: compact_diff if unified == 0 else huge_diff,
        )
        monkeypatch.setattr(
            review, "build_touched_file_pack",
            lambda *_a, **_k: ("FULL SNAPSHOT\n" + ("x = 1\n" * 400_000), []),
        )
        monkeypatch.setattr(review._cfg, "get_review_models", lambda: [
            "openai/gpt-5.5", "google/gemini-3.5-flash", "anthropic/claude-fable-5",
        ])

        def fake_review(*_args, **kwargs):
            captured["prompt"] = kwargs["prompt"]
            return self._fake_result(
                '[{"item":"code_quality","verdict":"PASS","severity":"advisory","reason":"ok"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"advisory","reason":"ok"}]',
            )

        monkeypatch.setattr(review, "_handle_multi_model_review", fake_review)

        assert review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir) is None
        prompt = captured["prompt"]
        assert "TRIAD FIT NOTE" in prompt
        assert "FULL SNAPSHOT" not in prompt
        assert compact_diff in prompt
        assert huge_diff not in prompt
        assert review.estimate_tokens(prompt) <= review.calibrated_input_token_limit(
            "anthropic/claude-fable-5",
            context_window=1_000_000,
            output_reserve=review._review_output_budget(),
            tokenizer_margin=50_000,
            budget_cap=review.REVIEW_PROMPT_TOKEN_BUDGET,
        )

    def test_triad_compact_rung_uses_hardened_capture_not_raw_run_cmd(self, review_ctx, monkeypatch):
        """The oversized ladder's compact rung called a RAW ``run_cmd(git diff
        --cached -U0)`` that inherits diff config/env and text decode, while only
        the primary diff used the hardened capture. The compact rung must use
        ``capture_staged_diff(unified=0)`` and never issue the raw ``-U0``
        command."""
        review, ctx = review_ctx
        huge_diff = "diff --git a/x.py b/x.py\n" + ("+changed line\n" * 190_000)
        compact_diff = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+new\n"

        run_cmd_calls = []

        def fake_run_cmd(cmd, cwd=None):
            run_cmd_calls.append(list(cmd))
            cmd = list(cmd)
            if cmd == ["git", "diff", "--cached", "--name-status"]:
                return "M\tx.py"
            if cmd == ["git", "diff", "--cached", "--name-only"]:
                return "x.py"
            return ""

        monkeypatch.setattr(review, "run_cmd", fake_run_cmd)

        capture_calls = []
        import ouroboros.tools.review_binary_context as _rbc

        def fake_capture(_repo, *, unified=3):
            capture_calls.append(unified)
            return compact_diff if unified == 0 else huge_diff

        monkeypatch.setattr(_rbc, "capture_staged_diff", fake_capture)
        monkeypatch.setattr(
            review, "build_touched_file_pack",
            lambda *_a, **_k: ("FULL SNAPSHOT\n" + ("x = 1\n" * 400_000), []))
        monkeypatch.setattr(review._cfg, "get_review_models", lambda: [
            "openai/gpt-5.5", "google/gemini-3.5-flash", "anthropic/claude-fable-5"])

        captured = {}

        def fake_review(*_args, **kwargs):
            captured["prompt"] = kwargs["prompt"]
            return self._fake_result(
                '[{"item":"code_quality","verdict":"PASS","severity":"advisory","reason":"ok"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"advisory","reason":"ok"}]')

        monkeypatch.setattr(review, "_handle_multi_model_review", fake_review)

        assert review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir) is None
        assert 0 in capture_calls, "compact rung must call capture_staged_diff(unified=0)"
        assert ["git", "diff", "--cached", "-U0"] not in run_cmd_calls, run_cmd_calls
        assert compact_diff in captured["prompt"]

    def test_advisory_mode_downgrades_quorum_failure(self, review_ctx, monkeypatch):
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                "Error: timeout",
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            "only 1 of 2 review models responded successfully" in w.lower()
            or "review enforcement=advisory" in w.lower()
            for w in ctx._review_advisory
        )

    def test_advisory_mode_keeps_preflight_as_warning(self, review_ctx, monkeypatch):
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="VERSION")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"version_bump","verdict":"PASS","severity":"critical","reason":"ok"}]',
                '[{"item":"readme_changelog","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "version update", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            isinstance(w, str) and "preflight warning did not block commit" in w.lower()
            for w in ctx._review_advisory
        )

    @pytest.mark.parametrize("item_id", _critical_triad_items())
    def test_advisory_downgrades_every_critical_item(self, item_id, review_ctx, monkeypatch):
        """NW-2 guardrail (58a52c4 class): advisory enforcement must downgrade a
        critical LLM finding for EVERY checklist item, with no per-item exception.

        The 58a52c4 incident added ``_ALWAYS_BLOCKING_ITEMS = {version_bump,
        forgotten_touchpoints}`` so those items blocked even under owner-chosen
        advisory mode. The pre-existing advisory test only used item
        ``code_quality``, so the hardcode passed the suite. This item-agnostic
        parametrization fails the moment any single item is special-cased to
        block under advisory.
        """
        review, ctx = review_ctx
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                f'[{{"item":"{item_id}","verdict":"FAIL","severity":"critical","reason":"broken"}}]',
                f'[{{"item":"{item_id}","verdict":"PASS","severity":"critical","reason":"looks ok to me"}}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is None, (
            f"advisory mode must NOT block critical item {item_id!r}; "
            "a per-item always-block hardcode (58a52c4 class) would fail here"
        )

    def test_new_module_triggers_architecture_preflight_through_run_unified_review(self, tmp_path, monkeypatch):
        """Check 4 (architecture_doc) fires through the real _run_unified_review caller.

        This proves the name-status conversion in _run_unified_review feeds
        _preflight_check correctly, so added files are detected.
        """
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        # Simulate: new ouroboros module added + tests staged, but ARCHITECTURE.md absent
        # name-status format: git emits "A\tpath" for added files
        self._mock_staged(
            monkeypatch, review,
            changed_files="ouroboros/new_module.py\ntests/test_new_module.py",
            name_status_files="A\touroboros/new_module.py\nA\ttests/test_new_module.py",
        )
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        result = review._run_unified_review(ctx, "add new module", repo_dir=ctx.repo_dir)
        # Should be blocked by preflight because ARCHITECTURE.md is not staged
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_rename_out_of_ouroboros_without_tests_passes(self):
        """Regression (#447): a rename/deletion of a .py file out of
        ouroboros/ without staged tests is no longer refused — the lexical
        tests-required predicate was removed (CHECKLISTS.md item 6 owns
        coverage semantically)."""
        review = _get_review_module()
        result = review._preflight_check(
            "move module out of ouroboros",
            "D  ouroboros/old.py\nR  docs/old.py",  # src deleted, dest not in ouroboros/
            "/tmp",
        )
        assert result is None

    def test_rename_into_ouroboros_triggers_architecture_check(self):
        """Renaming a .py file INTO ouroboros/ without ARCHITECTURE.md triggers check 4."""
        review = _get_review_module()
        # Destination becomes "A ouroboros/new_module.py" → triggers new-module check
        result = review._preflight_check(
            "move module into ouroboros",
            "D  docs/old_module.py\nA  ouroboros/new_module.py\nM  tests/test_new.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_rename_into_ouroboros_with_architecture_passes(self):
        """Renaming a .py file into ouroboros/ + staging ARCHITECTURE.md passes check 4."""
        review = _get_review_module()
        result = review._preflight_check(
            "move module into ouroboros",
            "D  docs/old_module.py\nA  ouroboros/new_module.py\nM  tests/test_new.py\nM  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    def test_rename_lines_parsed_correctly_by_preflight(self, tmp_path, monkeypatch):
        """Rename entries (R100\told\tnew) use the destination path for preflight checks."""
        review = _get_review_module()
        # Direct unit test of _preflight_check with a rename line
        # Renamed VERSION to VERSIONX — preflight should not care (it's not "VERSION")
        result = review._preflight_check(
            "rename version file",
            "R  VERSIONX",
            "/tmp",
        )
        # VERSIONX is not the VERSION carrier, so no preflight block expected
        assert result is None

    def test_rename_of_readme_counts_as_present(self, tmp_path, monkeypatch):
        """If README.md appears as a rename destination, preflight sees it as staged."""
        review = _get_review_module()
        # Simulate: VERSION staged + README.md arrived via rename
        result = review._preflight_check(
            "v1.0.0: rename readme",
            "M  VERSION\nR  README.md",
            "/tmp",
        )
        # Both VERSION and README.md present → no check 1 block
        assert result is None

    def test_copied_module_without_architecture_blocked(self):
        """Copied .py file in ouroboros/ (status C) triggers architecture-doc preflight."""
        review = _get_review_module()
        # C status means a new file that was copied from somewhere else — still a new module
        result = review._preflight_check(
            "add copied module",
            "C  ouroboros/new_copy.py\nM  tests/test_new_copy.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_copied_module_with_architecture_passes(self):
        """Copied .py file in ouroboros/ + ARCHITECTURE.md staged → passes."""
        review = _get_review_module()
        result = review._preflight_check(
            "add copied module",
            "C  ouroboros/new_copy.py\nM  tests/test_new_copy.py\nM  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    def test_logic_change_with_deleted_test_passes(self):
        """Regression (#447): a modified logic file plus a deleted test file
        is no longer refused for missing tests — the lexical tests-required
        predicate was removed."""
        review = _get_review_module()
        result = review._preflight_check(
            "refactor module",
            "M  ouroboros/some_module.py\nD  tests/test_old.py",
            "/tmp",
        )
        assert result is None

    def test_deleted_logic_file_without_tests_passes(self):
        """Regression (#447): a deletion-only diff in ouroboros/ without
        staged tests is no longer refused (removed tests-required predicate)."""
        review = _get_review_module()
        result = review._preflight_check(
            "remove old module",
            "D  ouroboros/old_module.py",
            "/tmp",
        )
        assert result is None

    def test_deleted_architecture_does_not_satisfy_check4(self):
        """Deleting ARCHITECTURE.md does not count as 'architecture doc staged'."""
        review = _get_review_module()
        result = review._preflight_check(
            "add new module",
            "A  ouroboros/new_module.py\nM  tests/test_new.py\nD  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_deleted_readme_does_not_satisfy_check1(self):
        """Deleting README.md while VERSION is staged triggers check 1."""
        review = _get_review_module()
        result = review._preflight_check(
            "v1.0.0: bump version",
            "M  VERSION\nD  README.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "README.md" in result

    def test_copied_module_triggers_via_run_unified_review(self, tmp_path, monkeypatch):
        """Check 4 fires for C-status copy via _run_unified_review, but source NOT treated as deleted."""
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        # Copy from ouroboros/base.py to ouroboros/new_copy.py.
        # The source (ouroboros/base.py) is unchanged — only the destination is new.
        # Architecture doc is absent → check 4 should fire.
        self._mock_staged(
            monkeypatch, review,
            changed_files="ouroboros/new_copy.py\ntests/test_new_copy.py",
            name_status_files="C100\touroboros/base.py\touroboros/new_copy.py\nA\ttests/test_new_copy.py",
        )
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        result = review._run_unified_review(ctx, "add copied module", repo_dir=ctx.repo_dir)
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    # ``test_copy_source_not_treated_as_deletion`` was removed with the
    # tests-required preflight heuristic (#447): a false D entry for a copy
    # source no longer has any observable preflight effect to pin.


# --- Unified review wired into commit functions ---

class TestReviewInCommitPipeline:
    # ``test_repo_commit_calls_unified_review`` was removed in
    # v5.8.3-rc.5 — it is a strict subset of
    # ``tests/test_scope_review.py::TestScopeReview::test_scope_review_wired_in_commit``
    # which additionally verify ``run_scope_review`` is reached and the
    # ``ThreadPoolExecutor`` parallelism contract holds.

    def test_blocked_review_unstages(self):
        """When review blocks, git reset HEAD must be called."""
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._run_reviewed_stage_cycle)
        assert 'git", "reset", "HEAD"' in source

    def test_review_rebuttal_forwarded(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        assert "review_rebuttal" in source


# --- Auto-push and last_push_succeeded ---

class TestAutoPushBehavior:
    def test_auto_push_exists(self):
        git_mod = _get_git_module()
        assert hasattr(git_mod, "_auto_push")
        assert callable(git_mod._auto_push)

    def test_auto_push_is_best_effort(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._auto_push)
        assert "except Exception" in source
        assert "non-fatal" in source.lower() or "non_fatal" in source.lower()

    def test_managed_tests_gate_before_tag_and_ordinary_tests_keep_prior_order(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        # The managed BLOCKING gate (extracted helper) runs before tagging;
        # ordinary warning-only tests run after the tag and before the push.
        managed_tests_pos = source.index("_managed_post_commit_tests_gate(")
        tag_pos = source.index("tag_info =")
        ordinary_tests_pos = source.index("_post_commit_result(ctx, commit_message")
        push_pos = source.rindex("push_status = _auto_push")
        assert managed_tests_pos < tag_pos < ordinary_tests_pos < push_pos

    def test_failed_managed_post_commit_gate_rolls_back_or_stays_blocked(
        self, monkeypatch
    ):
        git_mod = _get_git_module()
        from supervisor import update_merge

        marked = []
        monkeypatch.setattr(
            update_merge,
            "rollback_managed_update",
            lambda reason: (False, f"rollback failed for {reason}"),
        )
        monkeypatch.setattr(
            update_merge,
            "mark_update_tx_gate_blocked",
            # The real helper returns True when the valid tx was re-phased.
            lambda reason, detail: marked.append((reason, detail)) or True,
        )

        result = git_mod._managed_commit_gate_failure("post_tests", "tests failed")

        assert "MANAGED_UPDATE_GATE_BLOCKED" in result
        assert marked == [("post_tests", "rollback failed for post_tests")]


# --- configure_remote failure surfacing ---

class TestRemoteConfigSurfacing:
    def test_server_logs_remote_failure(self):
        """gateway.settings must check the personal remote provisioning result."""
        source = (pathlib.Path(REPO) / "ouroboros" / "gateway" / "settings.py").read_text(encoding="utf-8")
        assert "configure_personal_remote" in source
        assert "remote_ok, remote_msg, resolved_slug" in source
        assert "Remote configuration failed" in source

    def test_settings_save_returns_warnings(self):
        """api_settings_post must surface remote config failures."""
        source = (pathlib.Path(REPO) / "ouroboros" / "gateway" / "settings.py").read_text(encoding="utf-8")
        assert '"warnings"' in source

    def test_remote_credentials_migration_not_wired_at_startup(self):
        """Legacy token-in-URL migration is no longer run on startup."""
        server_path = pathlib.Path(REPO) / "server.py"
        source = server_path.read_text(encoding="utf-8")
        assert "migrate_remote_credentials" not in source


# --- credential helper safety (legacy migration retired) ---

class TestRemoteCredentialConfiguration:
    def test_legacy_migrator_retired(self):
        git_ops = _get_git_ops_module()
        assert not hasattr(git_ops, "migrate_remote_credentials")

    def test_configure_remote_uses_local_credential_helper(self):
        git_ops = _get_git_ops_module()
        configure_source = inspect.getsource(git_ops.configure_remote)
        helper_source = inspect.getsource(git_ops._configure_credential_helper)
        assert "_configure_credential_helper" in configure_source
        assert ".git/credentials" in helper_source

    def test_startup_setup_only_configures_current_settings_token(self):
        server_runtime = importlib.import_module("ouroboros.server_runtime")
        source = inspect.getsource(server_runtime.setup_remote_if_configured)
        assert "migrate_remote_credentials" not in source
        assert "configure_personal_remote" in source


# --- ToolContext review state ---

class TestToolContextReviewState:
    def test_review_fields_exist(self):
        from ouroboros.tools.registry import ToolContext
        ctx = ToolContext(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        assert hasattr(ctx, "_review_advisory")
        assert hasattr(ctx, "_review_iteration_count")
        assert hasattr(ctx, "_review_history")
        assert ctx._review_advisory == []
        assert ctx._review_iteration_count == 0
        assert ctx._review_history == []


# --- Registry sandbox covers repo_write ---

class TestSandboxCoversRepoWrite:
    def test_sandbox_mentions_repo_write(self):
        registry = _get_registry_module()
        source = inspect.getsource(registry.ToolRegistry.execute)
        assert "_ROOT_ARG_REPO_WRITE_TOOLS" in source
        assert "write_file" in registry._ROOT_ARG_REPO_WRITE_TOOLS

    def test_sandbox_checks_files_param(self):
        """Sandbox must check files array for safety-critical paths."""
        registry = _get_registry_module()
        assert registry._payload_write_paths(
            "write_file",
            {"files": [{"path": "BIBLE.md", "content": "x"}]},
        ) == ["BIBLE.md"]


# --- index-full instruction fix ---

class TestIndexFullInstruction:
    def test_system_md_warns_against_index_full(self):
        system_md = pathlib.Path(REPO) / "prompts" / "SYSTEM.md"
        content = system_md.read_text(encoding="utf-8")
        assert "Do NOT call" in content or "reserved internal name" in content
        assert "knowledge_list" in content


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
        result = review._preflight_check(
            "v4.99.0 bump", "M  VERSION\nM  tests/test_foo.py", "/repo"
        )
        # Check 1 fires first (README.md missing from staged when VERSION staged).
        # This is acceptable — the missing README is caught by check 1, not check 7.
        # Either result is valid here; we just verify no crash.
        assert result is None or "PREFLIGHT_BLOCKED" in result


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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 2 failed, 5 passed"

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)

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

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)

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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

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

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict", _fake_aggregate)

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

        monkeypatch.setattr(git_mod, "_check_advisory_freshness", _fake_freshness)

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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 1 failed"

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        def _fake_aggregate(*a, **kw):
            return False, "", "", [], []

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict", _fake_aggregate)

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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

        called = {"preflight": 0, "parallel": 0}

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return "FAILED: 1 test error"  # tests fail

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)

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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

        def _fake_preflight(ctx, *, timeout=120):
            called["preflight"] += 1
            return preflight_result

        def _fake_parallel(*a, **kw):
            called["parallel"] += 1
            return None, {}, "", []

        def _fake_aggregate(*a, **kw):
            return False, "", "", [], []

        monkeypatch.setattr(git_mod, "_run_review_preflight_tests", _fake_preflight)
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict", _fake_aggregate)

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
        monkeypatch.setattr(git_mod, "_check_advisory_freshness", lambda *a, **kw: None)

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
        monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
        monkeypatch.setattr(git_mod, "_aggregate_review_verdict",
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
