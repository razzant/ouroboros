"""The scope-review gate fails closed.

Split by theme out of the original ``tests/test_scope_review.py`` giant. This
module owns the gate itself: the fail-closed refusals of ``run_scope_review``
(unparseable verdicts, provider errors, missing packs, budget refusals), the
structured round history, and the fail-closed pack-assembly guards.
"""

import json

import pytest

from tests._scope_review_shared import _get_module

# ---------------------------------------------------------------------------
# Scope review module tests
# ---------------------------------------------------------------------------

class TestScopeFailClosed:
    """Runtime tests for fail-closed scope review behavior."""

    def test_build_scope_prompt_deletion_not_blocked(self, tmp_path):
        """_build_scope_prompt must NOT block on deletion-only diffs.
        
        Deletion-only diffs are valid: the HEAD snapshot shows old content,
        and the current_files_section has a deletion placeholder.
        This test verifies the correct new behavior after the Phase 3 fix.
        """
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text("## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        # Commit a file, then stage its deletion
        (tmp_path / "gone.py").write_text("CONTENT_BEFORE_DELETION", encoding="utf-8")
        subprocess.run(["git", "add", "gone.py"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T",
             "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "gone.py").unlink()
        subprocess.run(["git", "add", "gone.py"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, omitted = mod._build_scope_prompt(tmp_path, "test msg")
        # Deletion-only diffs must NOT block — omitted should be None
        assert omitted is None
        # HEAD snapshot must show old content
        assert "CONTENT_BEFORE_DELETION" in prompt
        # Current files section must note the deletion
        assert "DELETED" in prompt

    def test_build_scope_prompt_blocks_on_partial_omission(self, tmp_path):
        """_build_scope_prompt returns _TouchedContextStatus(status='omitted') when some files are binary."""
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text("## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "good.py").write_text("print('ok')", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        # Stage both files
        (tmp_path / "good.py").write_text("print('v2')", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 200)
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, context_status = mod._build_scope_prompt(tmp_path, "test msg")
        # Returns (None, _TouchedContextStatus) on fail-closed
        assert prompt is None
        assert context_status is not None
        assert context_status.status == "omitted"
        assert "image.png" in context_status.omitted_paths

    def test_build_scope_prompt_clean_when_all_readable(self, tmp_path):
        """_build_scope_prompt returns None omitted when all files are readable."""
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text("## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("bbb", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, omitted = mod._build_scope_prompt(tmp_path, "test msg")
        assert omitted is None
        assert "bbb" in prompt

    def test_scope_prompt_deduplicates_touched_tests_and_canonical_docs(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n",
            encoding="utf-8",
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "docs" / "ARCHITECTURE.md").write_text("architecture v1\n", encoding="utf-8")
        (tmp_path / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / "test_example.py").write_text("def test_old(): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )

        (tmp_path / "tests" / "test_example.py").write_text("def test_new(): pass\n", encoding="utf-8")
        (tmp_path / "docs" / "ARCHITECTURE.md").write_text("architecture v2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "test msg")

        assert status is None
        assert prompt is not None
        assert "## docs/ARCHITECTURE.md" in prompt
        assert "architecture v2" in prompt
        assert "CURRENT FILE CONTEXT DEDUPLICATION NOTE" in prompt
        assert "tests/test_example.py" in prompt
        assert "docs/ARCHITECTURE.md" in prompt
        assert "def test_new" in prompt  # visible via staged diff


def test_scope_history_keeps_all_rounds_and_structured_ids():
    mod = _get_module("ouroboros.tools.scope_review")
    history = [
        {
            "attempt": idx,
            "critical": [{
                "item": f"bug_{idx}",
                "severity": "critical",
                "reason": f"bug {idx}",
                "obligation_id": f"obl-00{idx}",
            }],
            "advisory": [{
                "item": f"advice_{idx}",
                "severity": "advisory",
                "reason": f"advice {idx}",
            }],
        }
        for idx in range(1, 5)
    ]
    out = mod._build_review_history_section(history, open_obligations=None)
    assert "Round 1" in out
    assert "Round 4" in out
    assert "⚠️ OMISSION NOTE" not in out
    assert "obligation=obl-001" in out


class TestRunScopeReviewFailClosed:
    """End-to-end fail-closed tests that execute run_scope_review()."""

    def test_run_scope_review_blocks_on_binary_files(self, tmp_path):
        """run_scope_review() must return SCOPE_REVIEW_BLOCKED for binary touched files."""
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text("## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "ok.py").write_text("print(1)", encoding="utf-8")
        (tmp_path / "bin.png").write_bytes(b"\x89PNG" + b"\x00" * 200)
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "ok.py").write_text("print(2)", encoding="utf-8")
        (tmp_path / "bin.png").write_bytes(b"\x89PNG" + b"\x00" * 300)
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        # Create a minimal mock ToolContext
        class MockCtx:
            repo_dir = str(tmp_path)
        ctx = MockCtx()

        mod = _get_module("ouroboros.tools.scope_review")
        result = mod.run_scope_review(
            ctx, "test commit",
            goal="test goal", scope="test scope",
        )
        assert result.blocked
        assert "SCOPE_REVIEW_BLOCKED" in result.block_message
        assert "bin.png" in result.block_message

    def test_build_scope_prompt_deletion_not_blocked_e2e(self, tmp_path):
        """_build_scope_prompt must NOT signal empty for deletion-only diffs.
        
        After the Phase 3 fix, deletion-only commits reach the scope reviewer.
        The prompt-builder must return omitted=None (not '__empty__') so
        run_scope_review proceeds to the LLM instead of short-circuiting.
        """
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text("## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "gone.py").write_text("CONTENT_X", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T",
             "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "gone.py").unlink()
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, omitted = mod._build_scope_prompt(tmp_path, "delete gone.py")
        # Deletion-only must NOT trigger fail-closed (omitted=None means "proceed to LLM")
        assert omitted is None, f"Expected omitted=None for deletion-only, got: {omitted!r}"
        # HEAD snapshot must show old content
        assert "CONTENT_X" in prompt
        # Current files section must note the deletion
        assert "DELETED" in prompt

    def test_build_scope_prompt_retries_compact_atlas_after_budget_overflow(self, tmp_path, monkeypatch):
        """Atlas budget overflow should retry once with compact manifest mode."""
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n",
            encoding="utf-8",
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "ok.py").write_text("print(2)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        calls = []

        def fake_gather(_repo_dir, _paths, **kwargs):
            calls.append(bool(kwargs.get("compact")))
            if not kwargs.get("compact"):
                raise mod._ScopeAtlasNotAssembled({"estimated_total_tokens": 900_000})
            return "COMPACT ATLAS"

        monkeypatch.setattr(scope_pack, "_gather_scope_packs", fake_gather)

        prompt, omitted = mod._build_scope_prompt(tmp_path, "test commit")

        assert omitted is None
        assert calls == [False, True]
        assert "COMPACT ATLAS" in prompt

    def test_build_scope_prompt_irreducible_overflow_fails_closed(self, tmp_path, monkeypatch):
        """Guaranteed-fit ladder: when even full touched degradation + compact
        atlas cannot fit (mocked oversize estimator), the result is the
        fail-closed fixed_overflow status — NOT a skippable budget_exceeded."""
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n",
            encoding="utf-8",
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "ok.py").write_text("print(2)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        calls = []

        def fake_gather(_repo_dir, _paths, fixed_prompt_tokens=0, compact=False, **_kw):
            calls.append(compact)
            if compact:
                raise mod._ScopeAtlasNotAssembled({"estimated_total_tokens": 900_001})
            return "OVERSIZED ATLAS"

        monkeypatch.setattr(scope_pack, "_gather_scope_packs", fake_gather)
        monkeypatch.setattr(scope_pack, "estimate_tokens", lambda _text: 800_000)

        prompt, status = mod._build_scope_prompt(tmp_path, "test commit")

        assert prompt is None
        assert status.status == "fixed_overflow"
        assert status.token_count > 0
        # First pass full atlas, then compact retries while the ladder degrades.
        assert calls[0] is False
        assert all(calls[1:])

    def test_build_scope_prompt_uses_zero_context_diff_before_overflow(self, tmp_path, monkeypatch):
        """The last fit step removes unchanged context, not changed lines or calls."""
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
        )
        (tmp_path / "tiny.py").write_text("old\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "tiny.py").write_text("new\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        large_diff = "diff --git a/tiny.py b/tiny.py\n" + (" unchanged context\n" * 30_000)
        compact_diff = "diff --git a/tiny.py b/tiny.py\n@@ -1 +1 @@\n-old\n+new\n"
        compact_calls = []

        def fake_capture(_repo_dir, *, unified=3):
            if unified == 0:
                compact_calls.append(True)
                return compact_diff
            return large_diff

        monkeypatch.setattr(scope_pack, "capture_staged_diff", fake_capture)
        monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 100_000)
        monkeypatch.setattr(scope_pack, "_gather_scope_packs", lambda *_a, **_k: "COMPACT ATLAS")

        prompt, status = mod._build_scope_prompt(tmp_path, "test commit")

        assert status is None
        assert compact_calls == [True]
        assert compact_diff in prompt
        assert large_diff not in prompt
        assert "COMPACT ATLAS" in prompt

    def test_sub_floor_windows_get_scaled_reserves_not_zero_limit(self, tmp_path, monkeypatch):
        """Provider Independence: the absolute 1M reserves must not swallow a
        small window whole. GigaChat (131K) and any known sub-1M slot keep a
        positive input limit via window-scaled reserves; the >=1M reviewer
        keeps the absolute reserves unchanged."""
        mod = _get_module("ouroboros.tools.scope_review")
        # Isolated evidence: every scope route (the designated default included) is
        # probed now, so an ambient store would otherwise decide this arithmetic.
        monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
        monkeypatch.setattr(
            "ouroboros.capability_evidence.probe",
            lambda *a, **k: mod.ReviewerWindow(0, "unprobeable"),
        )

        giga = mod._effective_scope_input_limit(scope_model="gigachat::GigaChat-3-Ultra")
        assert giga > 50_000, f"gigachat input limit must be workable, got {giga}"

        out, margin = mod._window_scaled_reserves(131_072)
        assert out == 32_768 and margin == 16_384
        # >=1M windows keep the absolute reserves (byte-identical behavior).
        assert mod._window_scaled_reserves(1_000_000) == (
            mod._SCOPE_MAX_TOKENS, mod._SCOPE_OUTPUT_MARGIN_TOKENS
        )
        # The designated >=1M reviewer keeps the ABSOLUTE reserves (not the
        # window-scaled ones); its cap is the family-calibrated absolute limit.
        from ouroboros.tools.review_helpers import calibrated_input_token_limit

        full = mod._effective_scope_input_limit(scope_model=mod._SCOPE_MODEL_DEFAULT)
        assert full == calibrated_input_token_limit(
            mod._SCOPE_MODEL_DEFAULT,
            context_window=mod._SCOPE_MODEL_CONTEXT_WINDOW,
            output_reserve=mod._SCOPE_MAX_TOKENS,
            tokenizer_margin=mod._SCOPE_OUTPUT_MARGIN_TOKENS,
        )
        assert full > 400_000  # a real 1M window keeps a large workable cap

    def test_irreducible_overflow_terminal_split_by_authority(self, tmp_path, monkeypatch):
        """Ladder terminal: the >=1M blocking reviewer fails CLOSED
        (fixed_overflow); a KNOWN sub-floor reviewer (advisory-only authority)
        emits the disclosed budget_exceeded fit signal, which run_scope_review
        later blocks unless the owner explicitly selected the advisory floor."""
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
        )
        (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "ok.py").write_text("print(2)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        monkeypatch.setattr(
            scope_pack, "_gather_scope_packs",
            lambda *_a, **_k: (_ for _ in ()).throw(
                mod._ScopeAtlasNotAssembled({"estimated_total_tokens": 999_999})
            ),
        )
        monkeypatch.setattr(scope_pack, "estimate_tokens", lambda _text: 800_000)
        # Capability Evidence: gigachat KNOWN sub-floor (131K), fable-5 >=1M.
        monkeypatch.setattr(
            scope_pack, "_scope_window",
            lambda m, **_k: mod.ReviewerWindow(
                131_072 if "gigachat" in str(m).lower() else 1_000_000, "confirmed",
            ),
        )

        _, status_full = mod._build_scope_prompt(
            tmp_path,
            "test commit",
            context=mod._ScopePromptContext(scope_model="anthropic/claude-fable-5"),
        )
        assert status_full.status == "fixed_overflow"

        _, status_sub = mod._build_scope_prompt(
            tmp_path,
            "test commit",
            context=mod._ScopePromptContext(scope_model="gigachat::GigaChat-3-Ultra"),
        )
        assert status_sub.status == "budget_exceeded"

    def test_build_scope_prompt_degrades_touched_files_to_fit(self, tmp_path, monkeypatch):
        """Guaranteed-fit ladder: a touched file too large for the budget is
        degraded to diff-only with an explicit disclosed note, and the prompt
        then fits and is returned (scope review actually runs)."""
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n",
            encoding="utf-8",
        )
        (tmp_path / "big.py").write_text("x = 1\n" * 40_000, encoding="utf-8")  # ~60K tokens
        (tmp_path / "small.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        # Tiny CHANGE inside a huge file: the staged diff stays small (irreducible
        # part fits) while the full post-change snapshot (~60K tokens) does not.
        (tmp_path / "big.py").write_text("y = 0\n" + "x = 1\n" * 39_999, encoding="utf-8")
        (tmp_path / "small.py").write_text("print(2)\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        monkeypatch.setattr(scope_pack, "_gather_scope_packs", lambda *_a, **_k: "TINY ATLAS")
        monkeypatch.setattr(
            scope_pack, "_effective_scope_input_limit", lambda **_kw: 30_000
        )

        prompt, status = mod._build_scope_prompt(tmp_path, "test commit")

        assert status is None
        assert prompt is not None
        assert "TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt
        assert "- big.py" in prompt
        # The small file keeps its full snapshot; the big one is diff-only.
        assert "### small.py" in prompt

    def test_run_scope_review_blocks_incomplete_scope_matrix(self, tmp_path, monkeypatch):
        """A parseable but incomplete scope checklist is a reviewer failure."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-contract-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        raw = json.dumps([
            {
                "item": "intent_alignment",
                "verdict": "PASS",
                "severity": "advisory",
                "reason": "Checked the staged intent against the changed review gate path.",
            }
        ])
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (raw, {"prompt_tokens": 10, "completion_tokens": 5}, None),
        )

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "parse_failure"
        assert "missing required items" in result.block_message
        assert result.parsed_items[0]["item"] == "intent_alignment"

    def test_run_scope_review_blocks_bare_pass_and_invalid_severity(self, tmp_path, monkeypatch):
        """Scope output contract rejects weak PASS reasons and bad severities."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-contract-negative-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        raw_items = [
            {
                "item": item_id,
                "verdict": "PASS",
                "severity": "advisory",
                "reason": f"Checked {item_id} against the staged review-gate fixture.",
            }
            for item_id in sorted(mod._SCOPE_REQUIRED_ITEMS)
        ]
        raw_items[0]["reason"] = "PASS"
        raw_items[1]["severity"] = "blocker"
        # FAIL without severity stays fail-closed (severity decides blocking);
        # PASS without severity is deliberately legal (defaulted to advisory).
        raw_items[2]["verdict"] = "FAIL"
        raw_items[2].pop("severity")
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (json.dumps(raw_items), {"prompt_tokens": 10, "completion_tokens": 5}, None),
        )

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "parse_failure"
        assert "PASS reason is too terse" in result.block_message
        assert "missing or invalid severity 'blocker'" in result.block_message
        assert "missing or invalid severity ''" in result.block_message

    @pytest.mark.parametrize("crit_item", sorted(_get_module("ouroboros.tools.scope_review")._SCOPE_REQUIRED_ITEMS))
    def test_advisory_downgrades_every_scope_critical_item(self, crit_item, tmp_path, monkeypatch):
        """NW-2 guardrail (58a52c4 class): under owner-chosen advisory enforcement,
        a critical scope finding for ANY required item must NOT block.

        ``forgotten_touchpoints`` is the scope-side item the 58a52c4 incident
        hardcoded to always block. The only pre-existing advisory-mode scope
        test hand-built a ScopeReviewResult and never exercised the real
        enforcement branch at run_scope_review level; this parametrization runs
        the real branch for every required item with a complete matrix (one
        critical FAIL + the rest valid PASS) so a per-item always-block hardcode
        fails here.
        """
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-advisory-guardrail-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        raw_items = []
        for item_id in sorted(mod._SCOPE_REQUIRED_ITEMS):
            if item_id == crit_item:
                raw_items.append({
                    "item": item_id,
                    "verdict": "FAIL",
                    "severity": "critical",
                    "reason": f"Staged diff violates {item_id} per the review-gate fixture.",
                })
            else:
                raw_items.append({
                    "item": item_id,
                    "verdict": "PASS",
                    "severity": "advisory",
                    "reason": f"Checked {item_id} against the staged review-gate fixture.",
                })
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (json.dumps(raw_items), {"prompt_tokens": 10, "completion_tokens": 5}, None),
        )
        # Treat the fake reviewer as >=1M so this test isolates the ADVISORY-ENFORCEMENT
        # downgrade (its target) from the separate sub-floor downgrade: an off-default
        # model with no Capability Evidence now fail-closes to the sub-floor (v6.46.0 fix),
        # which would empty critical_findings via the sub-floor path instead.
        monkeypatch.setattr(mod, "_scope_window",
                            lambda m, **_k: mod.ReviewerWindow(1_000_000, "confirmed"))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is False, (
            f"advisory mode must NOT block critical scope item {crit_item!r}; "
            "a per-item always-block hardcode (58a52c4 class) would fail here"
        )
        assert result.status == "responded"
        assert any(f.get("item") == crit_item for f in result.critical_findings)

    def test_provider_oversize_400_blocks_default_floor(self, tmp_path, monkeypatch):
        """A physical oversize cannot satisfy the default blocking scope floor."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-oversize-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        oversize_error = (
            "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer (anthropic/claude-fable-5) failed — commit blocked.\n"
            "Error: BadRequestError: Error code: 400 - prompt is too long: 1166914 tokens > 1000000 maximum\n"
            "Retry the commit, or check API key and network connectivity."
        )
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(mod, "_call_scope_llm", lambda *a, **k: ("", None, oversize_error))
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(1_000_000, "confirmed"))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="anthropic/claude-fable-5")

        assert result.blocked is True
        assert result.status == "fixed_overflow"
        assert result.advisory_findings[0]["item"] == "scope_review_skipped"
        assert "prompt is too long" in result.advisory_findings[0]["reason"]

    def test_sub_floor_scope_reviewer_is_advisory_only(self, tmp_path, monkeypatch):
        """BIBLE P3 floor: a scope model with a KNOWN <1M window gets a
        right-sized pack (window-aware cap) and can respond, but its critical
        findings must be delivered ADVISORY-ONLY — only a >=1M reviewer may act
        as the blocking scope gate. A >=1M reviewer keeps full blocking
        authority under the same enforcement."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-sub-floor-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        raw_items = []
        for item_id in sorted(mod._SCOPE_REQUIRED_ITEMS):
            if item_id == "intent_alignment":
                raw_items.append({
                    "item": item_id,
                    "verdict": "FAIL",
                    "severity": "critical",
                    "reason": "Staged diff contradicts the declared intent per the fixture.",
                })
            else:
                raw_items.append({
                    "item": item_id,
                    "verdict": "PASS",
                    "severity": "advisory",
                    "reason": f"Checked {item_id} against the staged review-gate fixture.",
                })
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (json.dumps(raw_items), {"prompt_tokens": 10, "completion_tokens": 5}, None),
        )

        # Capability Evidence sources the reviewer window (no static table, v6.33.0):
        # treat opus-4.8 / gigachat as KNOWN sub-floor (<1M), fable-5 as >=1M.
        monkeypatch.setattr(
            mod, "_scope_window",
            lambda m, **_k: mod.ReviewerWindow(
                200_000 if ("opus" in str(m) or "gigachat" in str(m).lower()) else 1_000_000,
                "confirmed",
            ),
        )

        result = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="anthropic/claude-opus-4.8"
        )
        assert result.blocked is True
        assert result.status == "sub_floor"
        assert result.critical_findings == []
        assert any(f.get("item") == "scope_review_sub_floor" for f in result.advisory_findings)
        assert any("[sub-floor scope reviewer]" in str(f.get("reason")) for f in result.advisory_findings)

        # GigaChat direct-provider form — documented below the floor.
        result_giga = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="gigachat::GigaChat-3-Ultra"
        )
        assert result_giga.blocked is True
        assert result_giga.status == "sub_floor"
        assert result_giga.critical_findings == []
        assert any(f.get("item") == "scope_review_sub_floor" for f in result_giga.advisory_findings)

        # The >=1M reviewer (fable-5 pin) keeps blocking authority.
        result_full = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="anthropic/claude-fable-5"
        )
        assert result_full.blocked is True
        assert any(f.get("item") == "intent_alignment" for f in result_full.critical_findings)

    def test_clean_sub_floor_pass_cannot_satisfy_blocking_floor(self, tmp_path, monkeypatch):
        """A clean sub-floor response is useful evidence, not an authoritative verdict."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-clean-sub-floor-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        clean_items = [
            {
                "item": item_id,
                "verdict": "PASS",
                "severity": "advisory",
                "reason": f"Checked {item_id} against the complete staged evidence.",
            }
            for item_id in sorted(mod._SCOPE_REQUIRED_ITEMS)
        ]
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(200_000, "confirmed"))
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (
                json.dumps(clean_items),
                {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.02},
                None,
            ),
        )

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="unknown/reviewer")

        assert result.blocked is True
        assert result.status == "sub_floor"
        assert result.cost_usd == 0.02
        assert any(f.get("item") == "scope_review_sub_floor" for f in result.advisory_findings)

    def test_oversize_always_fails_closed_after_floor_removal(self, tmp_path, monkeypatch):
        """v6.80.0: the owner advisory-floor escape is GONE — a provider oversize
        rejection has no authoritative verdict and therefore always fails CLOSED, in
        every context mode where scope review runs at all."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-advisory-oversize-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        error = "Error code: 400 - prompt is too long: 300000 tokens > 200000 maximum"
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(200_000, "confirmed"))
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(mod, "_call_scope_llm", lambda *a, **k: ("", None, error))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="small/reviewer")

        assert result.blocked is True
        assert result.status == "fixed_overflow"
        assert result.advisory_findings[0]["item"] == "scope_review_skipped"

    def test_irreducible_sub_floor_prompt_blocks_default_floor(self, monkeypatch):
        """A pre-dispatch sub-floor fit failure cannot silently satisfy P3."""
        mod = _get_module("ouroboros.tools.scope_review")
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(200_000, "confirmed"))

        result = mod._handle_prompt_signals(
            None,
            mod._TouchedContextStatus(status="budget_exceeded", token_count=180_000),
            input_limit=120_000,
            scope_model="small/reviewer",
        )

        assert result is not None
        assert result.blocked is True
        assert result.status == "sub_floor"

    def test_generic_provider_error_stays_fail_closed(self, tmp_path, monkeypatch):
        """B3 guard: non-oversize provider errors keep blocking (fail-closed)."""
        mod = _get_module("ouroboros.tools.scope_review")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-generic-error-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        generic_error = (
            "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer (test-scope) failed — commit blocked.\n"
            "Error: APIConnectionError: Connection error.\n"
            "Retry the commit, or check API key and network connectivity."
        )
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(mod, "_call_scope_llm", lambda *a, **k: ("", None, generic_error))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "error"

    def test_gateway_provider_error_400_oversize_blocks_default_floor(self, tmp_path, monkeypatch):
        """F2: an openai-compatible/OpenRouter oversize 400 returns an EMPTY body +
        usage['provider_error']{code:400} with llm_error='' (NOT a raised text error),
        so the text-marker oversize branch never fires and the empty body would
        otherwise hard-block as empty_response. With independent size evidence it must
        produce the same visible evidence while still blocking the default floor."""
        mod = _get_module("ouroboros.tools.scope_review")
        scope_budget = _get_module("ouroboros.tools.scope_review_budget")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-gateway-400-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        # 32 chars ~= 8 estimated tokens: below the effective input limit (10), but
        # near enough to be independent size evidence for an opaque gateway 400.
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("x" * 32, None))
        monkeypatch.setattr(
            mod, "_call_scope_llm",
            lambda *a, **k: ("", {"prompt_tokens": 0, "completion_tokens": 0,
                                  "provider_error": {"code": 400, "kind": "provider_error", "message": ""}}, ""),
        )
        monkeypatch.setattr(scope_budget, "_effective_scope_input_limit", lambda *a, **k: 10)

        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(1_000_000, "confirmed"))
        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="anthropic/claude-fable-5")

        assert result.blocked is True
        assert result.status == "fixed_overflow"
        assert result.advisory_findings[0]["item"] == "scope_review_skipped"

    def test_gateway_provider_error_400_non_oversize_stays_fail_closed(self, tmp_path, monkeypatch):
        """F2 guard: a NON-size gateway 400 (auth/param/policy — same code, same empty
        body) must NOT downgrade. No size evidence (small prompt, large window) keeps the
        fail-closed empty_response block, so a misconfiguration never silently skips the
        blocking scope review."""
        mod = _get_module("ouroboros.tools.scope_review")
        scope_budget = _get_module("ouroboros.tools.scope_review_budget")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-gateway-auth400-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("x" * 32, None))
        monkeypatch.setattr(
            mod, "_call_scope_llm",
            lambda *a, **k: ("", {"prompt_tokens": 0, "completion_tokens": 0,
                                  "provider_error": {"code": 400, "kind": "provider_error", "message": "invalid api key"}}, ""),
        )
        # Even a large prompt near the resolved window must stay fail-closed when the
        # provider gives a concrete non-size message. Size proximity is reserved for
        # opaque/empty gateway 400 bodies.
        monkeypatch.setattr(scope_budget, "_effective_scope_input_limit", lambda *a, **k: 10)

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "empty_response"

    def test_effective_scope_limit_uses_real_window_for_small_window_reviewer(self, monkeypatch):
        """B2: a KNOWN sub-1M reviewer window (Capability Evidence) replaces the
        assumed 1M, so the pack overflows into the visible budget_exceeded skip
        instead of a deterministic provider 400. The limit is computed PER CALL from
        the measured/cold density (v6.80.0), never from an import-time constant."""
        mod = _get_module("ouroboros.tools.scope_review")
        scope_budget = _get_module("ouroboros.tools.scope_review_budget")
        from ouroboros.tools.review_helpers import calibrated_input_token_limit
        # opus-4.8 KNOWN sub-floor (200K) via evidence; everything else >=1M.
        monkeypatch.setattr(
            scope_budget, "_scope_window",
            lambda m, **_k: mod.ReviewerWindow(
                200_000 if "opus" in str(m) else 1_000_000, "confirmed",
            ),
        )

        full = calibrated_input_token_limit(
            "anthropic/claude-fable-5",
            context_window=mod._SCOPE_MODEL_CONTEXT_WINDOW,
            output_reserve=mod._SCOPE_MAX_TOKENS,
            tokenizer_margin=mod._SCOPE_OUTPUT_MARGIN_TOKENS,
        )
        assert mod._effective_scope_input_limit(scope_model="anthropic/claude-fable-5") == full
        # opus-4.8 (200K window): cap shrinks far below the 1M-based one.
        small = mod._effective_scope_input_limit(scope_model="anthropic/claude-opus-4.8")
        assert 0 <= small < full
        # A cold-start non-Claude 1M reviewer is never LOOSER than the historical cap.
        assert mod._effective_scope_input_limit(scope_model="openai/gpt-5.5") <= mod._SCOPE_INPUT_TOKEN_LIMIT

    def test_run_scope_review_preserves_pass_rows_in_actor_record(self, tmp_path, monkeypatch):
        """scope_raw_result.parsed_items must keep PASS rows for audit coverage."""
        mod = _get_module("ouroboros.tools.scope_review")
        helpers = _get_module("ouroboros.tools.review_helpers")

        class MockCtx:
            repo_dir = str(tmp_path)
            task_id = "scope-pass-audit-test"
            pending_events = []

            def drive_logs(self):
                return tmp_path

        raw_items = [
            {
                "item": item_id,
                "verdict": "PASS",
                "severity": "advisory",
                "reason": f"Checked {item_id} against the staged review-gate fixture.",
            }
            for item_id in sorted(mod._SCOPE_REQUIRED_ITEMS)
        ]
        monkeypatch.setattr(mod, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
        monkeypatch.setattr(
            mod,
            "_call_scope_llm",
            lambda *a, **k: (json.dumps(raw_items), {"prompt_tokens": 10, "completion_tokens": 5}, None),
        )
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(1_000_000, "confirmed"))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")
        record = helpers.build_scope_actor_record(result, fallback_model_id="fallback-scope")

        assert result.blocked is False
        assert result.critical_findings == []
        assert result.advisory_findings == []
        assert len(result.parsed_items) == len(mod._SCOPE_REQUIRED_ITEMS)
        assert record["parsed_items"] == result.parsed_items
        assert {item["verdict"] for item in record["parsed_items"]} == {"PASS"}
