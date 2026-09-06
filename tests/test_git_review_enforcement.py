"""Review verdict parsing, history, quorum and enforcement modes.

Split verbatim out of ``tests/test_git_review_pipeline.py`` by theme. This
module owns how a review verdict is read and applied: JSON parsing, the
history the reviewers see, quorum arithmetic, and what blocking vs advisory
enforcement does to critical findings.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from tests._git_review_pipeline_shared import (
    _critical_triad_items,
    _get_review_module,
    _make_ctx,
)


@pytest.fixture
def review_ctx(tmp_path):
    """Yield ``(review_module, ToolContext)``."""
    return _get_review_module(), _make_ctx(tmp_path)


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
        # No version-ref in commit message, so no preflight block expected
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
        # No ouroboros .py → no check 3 block
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


# --- the triad call site's pack exclusions (review economics, D-06a) ----------


def _mock_triad_gates(review, monkeypatch, *, changed=("uv.lock", "VERSION")):
    """Deterministic gates around the touched-pack seam of _prepare_unified_review."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    name_status = "\n".join(f"M\t{p}" for p in changed)
    name_only = "\n".join(changed)

    def _run_cmd(cmd, cwd=None):
        cmd = list(cmd)
        if cmd[:5] == ["git", "diff", "--cached", "--name-status"]:
            return name_status
        if cmd[:4] == ["git", "diff", "--cached", "--name-only"]:
            return name_only
        if cmd[:3] == ["git", "diff", "--cached"]:
            return "diff --cached"
        return ""

    monkeypatch.setattr(review, "run_cmd", _run_cmd)
    import ouroboros.tools.review_binary_context as _rbc
    monkeypatch.setattr(_rbc, "capture_staged_diff", lambda _repo, *, unified=3: "diff --cached")
    monkeypatch.setattr(review, "_preflight_check", lambda *a, **k: None)
    monkeypatch.setattr(review, "_load_checklist_section", lambda: "## checklist")
    monkeypatch.setattr(review, "load_governance_doc", lambda repo, rel, **k: f"{rel} PREFIX TEXT")
    captured = {}

    def _fake_review(*_a, **kw):
        captured["prompt"] = kw["prompt"]
        return json.dumps({"results": [{
            "model": "m", "verdict": "PASS", "tokens_in": 1, "tokens_out": 1, "cost_estimate": 0.0,
            "text": '[{"item":"x","verdict":"PASS","severity":"advisory","reason":"ok"}]',
        }]})

    monkeypatch.setattr(review, "_handle_multi_model_review", _fake_review)
    return captured


def test_triad_pack_exclusions_reach_the_pack_and_the_prompt(review_ctx, monkeypatch):
    """The call site computes the two disclosed exclusion classes from the
    touched paths and the SAME prefix texts it inlines, hands them to the
    touched pack through the advisory seam's ``exclude_paths`` shape, and
    appends the disclosure note AFTER the builder's OMISSION NOTE."""
    review, ctx = review_ctx
    captured = _mock_triad_gates(review, monkeypatch)
    seen = {}

    def _exclusions(repo_dir, paths, *, prefix_texts):
        seen["paths"] = list(paths)
        seen["prefix_texts"] = dict(prefix_texts)
        return {"uv.lock"}, "PACK-EXCLUSION-NOTE-SENTINEL"

    def _pack(repo_dir, paths, **kw):
        seen["exclude_paths"] = kw.get("exclude_paths")
        return "### uv.lock\n\n*(omitted — withheld)*\n\n### VERSION\n```\n1.0.1\n```\n", ["uv.lock"]

    monkeypatch.setattr(review, "triad_pack_exclusions", _exclusions)
    monkeypatch.setattr(review, "build_touched_file_pack", _pack)

    review._run_unified_review(ctx, "release: 1.0.1", repo_dir=ctx.repo_dir)

    assert seen["paths"] == ["uv.lock", "VERSION"]
    assert seen["prefix_texts"] == {
        "docs/DEVELOPMENT.md": "docs/DEVELOPMENT.md PREFIX TEXT",
        "docs/DESIGN.md": "docs/DESIGN.md PREFIX TEXT",
        "docs/ARCHITECTURE.md": "docs/ARCHITECTURE.md PREFIX TEXT",
    }
    assert seen["exclude_paths"] == {"uv.lock"}
    prompt = captured["prompt"]
    omission = prompt.index("⚠️ OMISSION NOTE: 1 file(s) omitted from direct context: uv.lock")
    assert prompt.index("PACK-EXCLUSION-NOTE-SENTINEL") > omission
    assert prompt.index("## Staged diff") > prompt.index("PACK-EXCLUSION-NOTE-SENTINEL")


def test_a_managed_subject_keeps_every_full_text(review_ctx, monkeypatch):
    """A managed resolution's reviewed delta is M0→staged, not HEAD→staged, so
    the call site never computes the cut for it: the pack receives no
    ``exclude_paths`` and no exclusion note is appended."""
    from types import SimpleNamespace

    review, ctx = review_ctx
    captured = _mock_triad_gates(review, monkeypatch)
    import ouroboros.tools.review_subject as _subject
    fake_subject = SimpleNamespace(
        render_prompt_diff=lambda unified=3: "diff --managed", touched_paths=lambda: ["uv.lock"],
        m0_tree="", staged_tree="", name_status=[("M", "uv.lock")],
    )
    monkeypatch.setattr(_subject, "managed_review_subject", lambda ctx_, repo: fake_subject)
    monkeypatch.setattr(review, "triad_pack_exclusions",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no cut on a managed subject")))
    seen = {}

    def _pack(repo_dir, paths, **kw):
        seen["exclude_paths"] = kw.get("exclude_paths")
        seen["represent_binary"] = kw.get("represent_binary")
        return "### uv.lock\n```\nfull text\n```\n", []

    monkeypatch.setattr(review, "build_touched_file_pack", _pack)

    review._run_unified_review(ctx, "managed update", repo_dir=ctx.repo_dir)

    assert seen["exclude_paths"] == set() and seen["represent_binary"] is True
    assert "PACK EXCLUSION NOTE" not in captured["prompt"]
    assert "full text" in captured["prompt"]
