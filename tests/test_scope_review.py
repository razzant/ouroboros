"""Tests for the review stack upgrade: scope review, review_helpers, enriched triad.

Verifies:
- Checklist section loader extracts exact sections
- Goal/scope precedence: goal > scope > commit_message > fallback
- Touched-file pack builds correctly
- Scope review module structure
- Broader repo pack excludes touched files
- Path-aware freshness
- Stale marking lifecycle
- repo_commit doesn't bypass the new stack
- review_helpers imports cleanly (no circular deps)
"""

import importlib
import inspect
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_module(name):
    sys.path.insert(0, REPO)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# review_helpers tests
# ---------------------------------------------------------------------------

class TestChecklistSectionLoader:
    def test_loads_repo_commit_section(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.load_checklist_section("Repo Commit Checklist")
        assert "## Repo Commit Checklist" in section
        assert "bible_compliance" in section
        # Must NOT contain scope checklist
        assert "Intent / Scope Review Checklist" not in section

    def test_loads_scope_section(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.load_checklist_section("Intent / Scope Review Checklist")
        assert "## Intent / Scope Review Checklist" in section
        assert "intent_alignment" in section
        # Must NOT contain repo commit checklist items
        assert "## Repo Commit Checklist" not in section

    def test_raises_on_missing_section(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        with pytest.raises(ValueError):
            mod.load_checklist_section("Nonexistent Section")


class TestGoalSection:
    def test_goal_section_has_source(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_goal_section(goal="fix bug", scope="", commit_message="msg")
        assert "Source: goal" in section
        assert "fix bug" in section

    def test_scope_section_empty_when_no_scope(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_scope_section()
        assert section == ""

    def test_scope_section_present_when_scope(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_scope_section(scope="only review.py")
        assert "only review.py" in section
        assert "IMPORTANT" in section


class TestTouchedFilePack:
    def test_reads_existing_files(self, tmp_path):
        (tmp_path / "a.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "b.md").write_text("# readme", encoding="utf-8")
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["a.py", "b.md"])
        assert "a.py" in pack
        assert "print('hello')" in pack
        assert "b.md" in pack
        assert omitted == []

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["image.png"])
        assert "image.png" in omitted
        assert "```" not in pack or "image.png" not in pack.split("```")[1] if "```" in pack else True

    def test_omits_large_files(self, tmp_path):
        # _FILE_SIZE_LIMIT is now 1MB; write a file slightly above that threshold
        (tmp_path / "huge.py").write_bytes(b"x" * (1_048_576 + 1))
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["huge.py"])
        assert "huge.py" in omitted
        assert "omitted" in pack.lower()


class TestBroaderRepoPack:
    def test_excludes_touched_files(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "a.py").write_text("AAA", encoding="utf-8")
        (tmp_path / "b.py").write_text("BBB", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_full_repo_pack(tmp_path, exclude_paths={"a.py"})
        assert "BBB" in pack
        assert "AAA" not in pack
        assert "a.py" not in omitted


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
        calls = []

        def fake_gather(_repo_dir, _paths, **kwargs):
            calls.append(bool(kwargs.get("compact")))
            if not kwargs.get("compact"):
                raise mod._ScopeAtlasBudgetExceeded({"estimated_total_tokens": 900_000})
            return "COMPACT ATLAS"

        monkeypatch.setattr(mod, "_gather_scope_packs", fake_gather)

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
        calls = []

        def fake_gather(_repo_dir, _paths, fixed_prompt_tokens=0, compact=False, **_kw):
            calls.append(compact)
            if compact:
                raise mod._ScopeAtlasBudgetExceeded({"estimated_total_tokens": 900_001})
            return "OVERSIZED ATLAS"

        monkeypatch.setattr(mod, "_gather_scope_packs", fake_gather)
        monkeypatch.setattr(mod, "estimate_tokens", lambda _text: 800_000)

        prompt, status = mod._build_scope_prompt(tmp_path, "test commit")

        assert prompt is None
        assert status.status == "fixed_overflow"
        assert status.token_count > 0
        # First pass full atlas, then compact retries while the ladder degrades.
        assert calls[0] is False
        assert all(calls[1:])

    def test_sub_floor_windows_get_scaled_reserves_not_zero_limit(self):
        """Provider Independence: the absolute 1M reserves must not swallow a
        small window whole. GigaChat (131K) and any known sub-1M slot keep a
        positive input limit via window-scaled reserves; the >=1M reviewer
        keeps the absolute reserves unchanged."""
        mod = _get_module("ouroboros.tools.scope_review")

        giga = mod._effective_scope_input_limit(scope_model="gigachat::GigaChat-3-Ultra")
        assert giga > 50_000, f"gigachat input limit must be workable, got {giga}"

        out, margin = mod._window_scaled_reserves(131_072)
        assert out == 32_768 and margin == 16_384
        # >=1M windows keep the absolute reserves (byte-identical behavior).
        assert mod._window_scaled_reserves(1_000_000) == (
            mod._SCOPE_MAX_TOKENS, mod._SCOPE_OUTPUT_MARGIN_TOKENS
        )
        full = mod._effective_scope_input_limit(scope_model="openai/gpt-5.5")
        assert full == mod._SCOPE_INPUT_TOKEN_LIMIT

    def test_irreducible_overflow_terminal_split_by_authority(self, tmp_path, monkeypatch):
        """Ladder terminal: the >=1M blocking reviewer fails CLOSED
        (fixed_overflow); a KNOWN sub-floor reviewer (advisory-only authority)
        routes to the disclosed non-blocking budget_exceeded skip so a
        sub-floor-only install keeps committing on triad authority."""
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
        monkeypatch.setattr(
            mod, "_gather_scope_packs",
            lambda *_a, **_k: (_ for _ in ()).throw(
                mod._ScopeAtlasBudgetExceeded({"estimated_total_tokens": 999_999})
            ),
        )
        monkeypatch.setattr(mod, "estimate_tokens", lambda _text: 800_000)
        # Capability Evidence: gigachat KNOWN sub-floor (131K), fable-5 >=1M.
        monkeypatch.setattr(
            mod, "_scope_reviewer_window",
            lambda m: 131_072 if "gigachat" in str(m).lower() else 1_000_000,
        )

        _, status_full = mod._build_scope_prompt(
            tmp_path, "test commit", scope_model="anthropic/claude-fable-5"
        )
        assert status_full.status == "fixed_overflow"

        _, status_sub = mod._build_scope_prompt(
            tmp_path, "test commit", scope_model="gigachat::GigaChat-3-Ultra"
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
        monkeypatch.setattr(mod, "_gather_scope_packs", lambda *_a, **_k: "TINY ATLAS")
        monkeypatch.setattr(
            mod, "_effective_scope_input_limit", lambda **_kw: 30_000
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
        monkeypatch.setattr(mod, "_scope_reviewer_window", lambda m: 1_000_000)

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is False, (
            f"advisory mode must NOT block critical scope item {crit_item!r}; "
            "a per-item always-block hardcode (58a52c4 class) would fail here"
        )
        assert result.status == "responded"
        assert any(f.get("item") == crit_item for f in result.critical_findings)

    def test_provider_oversize_400_downgrades_to_nonblocking_skip(self, tmp_path, monkeypatch):
        """B3: a provider context-length 400 during the scope call (estimate gate
        passed, real tokenizer denser) is the SAME failure class as the pre-call
        budget_exceeded skip — non-blocking, visible advisory. Any other provider
        error stays fail-closed (next test pins that)."""
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

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="anthropic/claude-fable-5")

        assert result.blocked is False
        assert result.status == "budget_exceeded"
        assert result.advisory_findings
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
            mod, "_scope_reviewer_window",
            lambda m: 200_000 if ("opus" in str(m) or "gigachat" in str(m).lower()) else 1_000_000,
        )

        result = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="anthropic/claude-opus-4.8"
        )
        assert result.blocked is False
        assert result.critical_findings == []
        assert any(f.get("item") == "scope_review_sub_floor" for f in result.advisory_findings)
        assert any("[sub-floor scope reviewer]" in str(f.get("reason")) for f in result.advisory_findings)

        # GigaChat direct-provider form — documented below the floor.
        result_giga = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="gigachat::GigaChat-3-Ultra"
        )
        assert result_giga.blocked is False
        assert result_giga.critical_findings == []
        assert any(f.get("item") == "scope_review_sub_floor" for f in result_giga.advisory_findings)

        # The >=1M reviewer (fable-5 pin) keeps blocking authority.
        result_full = mod.run_scope_review(
            MockCtx(), "test commit", scope_model="anthropic/claude-fable-5"
        )
        assert result_full.blocked is True
        assert any(f.get("item") == "intent_alignment" for f in result_full.critical_findings)

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

    def test_gateway_provider_error_400_oversize_downgrades_to_nonblocking_skip(self, tmp_path, monkeypatch):
        """F2: an openai-compatible/OpenRouter oversize 400 returns an EMPTY body +
        usage['provider_error']{code:400} with llm_error='' (NOT a raised text error),
        so the text-marker oversize branch never fires and the empty body would
        otherwise hard-block as empty_response. With independent size evidence it must
        downgrade to the SAME non-blocking budget_exceeded advisory (the v6.46.0
        OpenRouter scope-discard variant)."""
        mod = _get_module("ouroboros.tools.scope_review")

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
        monkeypatch.setattr(mod, "_effective_scope_input_limit", lambda *a, **k: 10)

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="anthropic/claude-sonnet-4.5")

        assert result.blocked is False
        assert result.status == "budget_exceeded"
        assert result.advisory_findings
        assert result.advisory_findings[0]["item"] == "scope_review_skipped"

    def test_gateway_provider_error_400_non_oversize_stays_fail_closed(self, tmp_path, monkeypatch):
        """F2 guard: a NON-size gateway 400 (auth/param/policy — same code, same empty
        body) must NOT downgrade. No size evidence (small prompt, large window) keeps the
        fail-closed empty_response block, so a misconfiguration never silently skips the
        blocking scope review."""
        mod = _get_module("ouroboros.tools.scope_review")

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
        monkeypatch.setattr(mod, "_effective_scope_input_limit", lambda *a, **k: 10)

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "empty_response"

    def test_effective_scope_limit_uses_real_window_for_small_window_reviewer(self, monkeypatch):
        """B2: a KNOWN sub-1M reviewer window (Capability Evidence) replaces the
        assumed 1M, so the pack overflows into the visible budget_exceeded skip
        instead of a deterministic provider 400. Unknown windows keep the 1M
        constants (the default reviewer is >=1M)."""
        mod = _get_module("ouroboros.tools.scope_review")
        # opus-4.8 KNOWN sub-floor (200K) via evidence; everything else >=1M.
        monkeypatch.setattr(
            mod, "_scope_reviewer_window",
            lambda m: 200_000 if "opus" in str(m) else 1_000_000,
        )

        # fable-5 (1M, anthropic family): calibrated 1M-based constant.
        assert mod._effective_scope_input_limit(scope_model="anthropic/claude-fable-5") == mod._ANTHROPIC_SCOPE_INPUT_TOKEN_LIMIT
        # opus-4.8 (200K window): cap shrinks far below the 1M-based one.
        small = mod._effective_scope_input_limit(scope_model="anthropic/claude-opus-4.8")
        assert 0 <= small < mod._ANTHROPIC_SCOPE_INPUT_TOKEN_LIMIT
        # gpt-5.5 (1M, non-anthropic): unchanged constant.
        assert mod._effective_scope_input_limit(scope_model="openai/gpt-5.5") == mod._SCOPE_INPUT_TOKEN_LIMIT

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

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")
        record = helpers.build_scope_actor_record(result, fallback_model_id="fallback-scope")

        assert result.blocked is False
        assert result.critical_findings == []
        assert result.advisory_findings == []
        assert len(result.parsed_items) == len(mod._SCOPE_REQUIRED_ITEMS)
        assert record["parsed_items"] == result.parsed_items
        assert {item["verdict"] for item in record["parsed_items"]} == {"PASS"}


class TestScopeReviewModule:
    # test_scope_review_imports removed in v5.15.x — pure callable-existence
    # check. The fail-closed test below already imports the module, and the
    # behavioral integration tests exercise run_scope_review end-to-end.

    def test_scope_review_fail_closed_design(self):
        """run_scope_review must be fail-closed: errors return blocking strings."""
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod.run_scope_review)
        assert "SCOPE_REVIEW_BLOCKED" in source
        assert "fail" in source.lower() or "block" in source.lower()

    def test_scope_review_uses_opus(self):
        mod = _get_module("ouroboros.tools.scope_review")
        assert "gpt-5.5" in mod._SCOPE_MODEL_DEFAULT
        # Verify the getter returns opus when no override env var is set
        import os
        if not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL"):
            assert "gpt-5.5" in mod._get_scope_model()
        # else: env override is active — default check not applicable in this env

    def test_scope_review_model_configurable_via_env(self):
        """OUROBOROS_SCOPE_REVIEW_MODEL env overrides the default."""
        mod = _get_module("ouroboros.tools.scope_review")
        import os
        old = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL")
        old_plural = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS")
        try:
            os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODELS", None)
            os.environ["OUROBOROS_SCOPE_REVIEW_MODEL"] = "google/gemini-2.5-pro"
            assert mod._get_scope_model() == "google/gemini-2.5-pro"
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODEL", None)
            else:
                os.environ["OUROBOROS_SCOPE_REVIEW_MODEL"] = old
            if old_plural is None:
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODELS", None)
            else:
                os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] = old_plural

    def test_scope_review_effort_configurable(self):
        """OUROBOROS_EFFORT_SCOPE_REVIEW should resolve via resolve_effort."""
        from ouroboros.config import resolve_effort
        import os
        old = os.environ.get("OUROBOROS_EFFORT_SCOPE_REVIEW")
        try:
            os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = "low"
            assert resolve_effort("scope_review") == "low"
            assert resolve_effort("scope-review") == "low"
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_EFFORT_SCOPE_REVIEW", None)
            else:
                os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = old

    def test_scope_prompt_includes_scope_checklist(self):
        """_build_scope_prompt must load the scope checklist, not the repo checklist."""
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod._build_scope_prompt)
        assert "Intent / Scope Review Checklist" in source

    def test_scope_prompt_includes_generated_scope_atlas(self):
        # scope_review now uses the bounded generated Atlas instead of the legacy full pack.
        # The call is in _gather_scope_packs which _build_scope_prompt delegates to.
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod._gather_scope_packs)
        assert "compile_review_context_atlas" in source
        assert "ReviewContextAtlasRequest" in source
        assert "fixed_prompt_tokens" in source

    def test_scope_prompt_fails_closed_on_atlas_inventory_error(self, tmp_path, monkeypatch):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@o", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("bbb", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        monkeypatch.setattr(
            mod,
            "compile_review_context_atlas",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("inventory failed")),
        )
        with pytest.raises(RuntimeError, match="inventory failed"):
            mod._build_scope_prompt(tmp_path, "test msg")

    def test_scope_prompt_keeps_literal_atlas_placeholder_in_touched_content(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@o", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("print('__GENERATED_SCOPE_ATLAS_PENDING__')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "test msg")
        assert status is None
        current_section = prompt[prompt.index("## Current touched files"):prompt.index("## Wider repository context")]
        assert "__GENERATED_SCOPE_ATLAS_PENDING__" in current_section


# ---------------------------------------------------------------------------
# review_state path-aware freshness
# ---------------------------------------------------------------------------

class TestPathAwareFreshness:
    def test_snapshot_hash_stable_without_message(self, tmp_path):
        """Snapshot hash should NOT change when only commit_message changes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        rs = _get_module("ouroboros.review_state")
        h1 = rs.compute_snapshot_hash(tmp_path, "message A")
        h2 = rs.compute_snapshot_hash(tmp_path, "message B")
        # Hash now based on code only — should be SAME for different messages
        assert h1 == h2

    def test_snapshot_hash_changes_with_file_content(self, tmp_path):
        """Snapshot hash must change when file content changes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "file.py").write_text("v1", encoding="utf-8")
        subprocess.run(["git", "add", "file.py"], cwd=str(tmp_path), capture_output=True)
        rs = _get_module("ouroboros.review_state")
        h1 = rs.compute_snapshot_hash(tmp_path, "msg")
        # Modify file
        (tmp_path / "file.py").write_text("v2", encoding="utf-8")
        h2 = rs.compute_snapshot_hash(tmp_path, "msg")
        assert h1 != h2

    def test_path_scoped_hash(self, tmp_path):
        """When paths= is provided, only those files affect the hash."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.py").write_text("bbb", encoding="utf-8")
        rs = _get_module("ouroboros.review_state")
        h_a = rs.compute_snapshot_hash(tmp_path, paths=["a.py"])
        h_b = rs.compute_snapshot_hash(tmp_path, paths=["b.py"])
        assert h_a != h_b

    def test_stale_lifecycle(self):
        """add_run marks previous non-matching fresh runs as stale."""
        rs = _get_module("ouroboros.review_state")
        state = rs.AdvisoryReviewState()
        run1 = rs.AdvisoryRunRecord(
            snapshot_hash="hash1", commit_message="m1",
            status="fresh", ts="2026-01-01T00:00:00",
        )
        state.add_run(run1)
        assert state.advisory_runs[0].status == "fresh"

        run2 = rs.AdvisoryRunRecord(
            snapshot_hash="hash2", commit_message="m2",
            status="fresh", ts="2026-01-01T01:00:00",
        )
        state.add_run(run2)
        assert state.advisory_runs[0].status == "stale"  # hash1 became stale
        assert state.advisory_runs[1].status == "fresh"   # hash2 is fresh


# ---------------------------------------------------------------------------
# Triad review enrichment
# ---------------------------------------------------------------------------

class TestTriadReviewEnriched:
    def test_triad_prompt_has_touched_files_placeholder(self):
        """The review prompt template must include current_files_section."""
        mod = _get_module("ouroboros.tools.review")
        assert "{current_files_section}" in mod._REVIEW_PROMPT_TEMPLATE

    def test_triad_prompt_has_goal_section(self):
        """The review prompt template must include goal_section."""
        mod = _get_module("ouroboros.tools.review")
        assert "{goal_section}" in mod._REVIEW_PROMPT_TEMPLATE

    def test_run_unified_review_accepts_goal_scope(self):
        """_run_unified_review must accept goal and scope keyword args."""
        mod = _get_module("ouroboros.tools.review")
        sig = inspect.signature(mod._run_unified_review)
        assert "goal" in sig.parameters
        assert "scope" in sig.parameters


# ---------------------------------------------------------------------------
# git.py wiring
# ---------------------------------------------------------------------------

class TestGitWiring:
    def test_repo_commit_schema_has_goal_scope(self):
        git = _get_module("ouroboros.tools.git")
        tools = git.get_tools()
        commit = next(t for t in tools if t.name == "commit_reviewed")
        props = commit.schema["parameters"]["properties"]
        assert "goal" in props
        assert "scope" in props

    def test_repo_commit_push_accepts_goal_scope(self):
        git = _get_module("ouroboros.tools.git")
        sig = inspect.signature(git._repo_commit_push)
        assert "goal" in sig.parameters
        assert "scope" in sig.parameters

    def test_scope_review_wired_in_commit(self):
        """The shared reviewed stage must call the parallel review helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_reviewed_stage_cycle)
        assert "_run_parallel_review" in source
        # The parallel helper must contain both triad and scope review
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "run_scope_review" in parallel_source
        assert "_run_unified_review" in parallel_source
        # ThreadPoolExecutor must be used for parallel execution
        assert "ThreadPoolExecutor" in parallel_source

    def test_repo_commit_not_bypass_scope(self):
        """repo_commit must reach scope review via the shared stage helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._repo_commit_push)
        assert "_run_reviewed_stage_cycle" in source
        shared_source = inspect.getsource(git._run_reviewed_stage_cycle)
        assert "_check_advisory_freshness" in shared_source
        assert "_run_parallel_review" in shared_source
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "run_scope_review" in parallel_source
        assert "ThreadPoolExecutor" in parallel_source

    def test_parallel_execution_both_always_run(self):
        """Both triad and scope futures are always submitted regardless of each other's result."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_parallel_review)
        # Both submissions must be present before any result() call
        submit_triad = source.find("triad_fut = pool.submit")
        submit_scope = source.find("scope_fut = pool.submit")
        result_triad = source.find("triad_fut.result()")
        result_scope = source.find("scope_fut.result()")
        # Both must be submitted, and submissions must precede result() calls
        assert submit_triad > 0
        assert submit_scope > 0
        assert result_triad > 0
        assert result_scope > 0
        # Both submitted before any result() is collected
        assert submit_triad < result_triad
        assert submit_scope < result_scope

    def test_aggregated_verdict_both_blockers_shown(self):
        """When both triad and scope block, both messages must appear in combined output."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        triad_error = "⚠️ REVIEW_BLOCKED: triad finding"
        scope_blocked = scope_mod.ScopeReviewResult(
            blocked=True,
            block_message="⚠️ SCOPE_REVIEW_BLOCKED: scope finding",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "scope blocked", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                triad_error, scope_blocked, "critical_findings", [], ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        assert "triad finding" in combined_msg
        assert "scope finding" in combined_msg
        assert "Both triad review AND scope review" in combined_msg
        assert len(findings) == 1

    def test_triad_advisory_included_when_scope_blocks(self):
        """When triad passes but has advisory findings and scope blocks, all findings appear."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_blocked = scope_mod.ScopeReviewResult(
            blocked=True,
            block_message="⚠️ SCOPE_REVIEW_BLOCKED: scope critical finding",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "scope blocked", "model": "test"}],
        )
        triad_advisory = [{"item": "context_building", "reason": "advisory note"}]
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_blocked, "scope_blocked", triad_advisory, ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        assert "scope critical finding" in combined_msg
        assert "advisory note" in combined_msg
        assert len(findings) == 1

    def test_advisory_mode_scope_criticals_not_in_blocking_findings(self):
        """Advisory-mode scope critical findings must NOT be added to _combined_findings."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Triad blocks; scope does NOT block but has critical findings (advisory enforcement)
        triad_error = "⚠️ REVIEW_BLOCKED: triad issue"
        scope_advisory_crit = scope_mod.ScopeReviewResult(
            blocked=False,  # advisory mode — not blocked
            block_message="",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "advisory-only scope note", "model": "test"}],
            advisory_findings=[],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                triad_error, scope_advisory_crit, "critical_findings", [], ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        # Advisory-mode scope criticals must NOT appear in durable blocking findings
        assert all(f.get("item") != "intent_alignment" for f in findings), \
            "Advisory-mode scope criticals must not be recorded as blocking findings"
        # But should appear in scope_advisory_items for visibility
        assert any(
            (isinstance(item, dict) and item.get("item") == "intent_alignment")
            or (isinstance(item, str) and "intent_alignment" in item)
            for item in scope_adv
        )

    def test_scope_advisory_visible_on_successful_commit(self):
        """Non-blocking scope advisory findings must be returned even when commit is not blocked."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Scope passes (not blocked) but has advisory findings
        scope_advisory = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[],
            advisory_findings=[{"verdict": "PASS", "item": "architecture_fit",
                                "severity": "advisory", "reason": "minor concern", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        # Should NOT block
        assert not blocked
        assert combined_msg is None
        # But scope advisory items must be returned for caller to surface
        assert len(scope_adv) > 0
        assert any(
            (isinstance(item, dict) and item.get("item") == "architecture_fit")
            or (isinstance(item, str) and "architecture_fit" in item)
            for item in scope_adv
        )

    @pytest.mark.parametrize("crit_item", sorted(_get_module("ouroboros.tools.scope_review")._SCOPE_REQUIRED_ITEMS))
    def test_aggregation_does_not_block_on_advisory_scope_criticals(self, crit_item):
        """NW-2 guardrail (aggregation seam): a 58a52c4-class hardcode could be
        re-introduced downstream in aggregate_review_verdict instead of in
        scope_review.py. With no triad error and a non-blocked scope result that
        merely CARRIES a critical finding (advisory pass-through), the aggregator
        must NOT flip to blocked for ANY item id.
        """
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_advisory_crit = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[{"verdict": "FAIL", "item": crit_item,
                                "severity": "critical", "reason": "advisory-only scope note", "model": "test"}],
            advisory_findings=[],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory_crit, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        assert not blocked, (
            f"aggregation must NOT block on an advisory-pass-through scope critical "
            f"for item {crit_item!r}; a per-item always-block hardcode would fail here"
        )
        assert combined_msg is None

    def test_scope_review_skipped_surfaces_through_aggregation_path(self):
        """Budget-skip advisories must survive aggregation and caller-side surfacing."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_advisory = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[],
            advisory_findings=[{
                "verdict": "FAIL",
                "item": "scope_review_skipped",
                "severity": "advisory",
                "reason": "⚠️ SCOPE_REVIEW_SKIPPED: Full scope-review prompt exceeds budget.",
                "model": "scope_reviewer",
            }],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory, "", [], ctx, "test commit", 0.0, ctx.repo_dir)

        if scope_adv:
            ctx._review_advisory.extend(scope_adv)

        assert not blocked
        assert combined_msg is None
        assert findings == []
        assert any(
            (isinstance(item, dict) and item.get("item") == "scope_review_skipped")
            or (isinstance(item, str) and "scope_review_skipped" in item)
            for item in scope_adv
        )
        assert any(
            (isinstance(item, dict) and item.get("item") == "scope_review_skipped")
            or (isinstance(item, str) and "scope_review_skipped" in item)
            for item in ctx._review_advisory
        )

    def test_triad_crash_resets_stale_findings(self):
        """If triad crashes, stale ctx findings from prior attempt must not bleed into current run."""
        import types
        import unittest.mock as mock
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Seed stale fields from a previous attempt
        ctx = types.SimpleNamespace(
            repo_dir=None,
            _last_review_block_reason="critical_findings",
            _last_review_critical_findings=[
                {"verdict": "FAIL", "item": "secrets_check", "severity": "critical",
                 "reason": "stale from prior run", "model": "old-model"}
            ],
            _review_advisory=[],
            _review_history=[],
            _scope_review_history={},
        )
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._run_unified_review",
                            side_effect=RuntimeError("triad crashed")):
                with mock.patch("ouroboros.tools.scope_review.run_scope_review") as mock_scope:
                    from ouroboros.tools.scope_review import ScopeReviewResult
                    mock_scope.return_value = ScopeReviewResult(blocked=False)
                    review_err, scope_result, triad_block_reason, _ = pr_mod.run_parallel_review(
                        ctx, "test commit")
        # Triad crash must yield infra_failure reason, not the stale critical_findings
        assert triad_block_reason == "infra_failure"
        # Stale findings must be cleared — no bleed-through to aggregate
        assert ctx._last_review_critical_findings == []
        assert "crashed" in review_err

    def test_scope_crash_resets_stale_actor_records(self):
        """If scope crashes, current raw evidence must not reuse previous scope actors."""
        import types
        import unittest.mock as mock
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        ctx = types.SimpleNamespace(
            repo_dir=None,
            _last_review_block_reason="",
            _last_review_critical_findings=[],
            _review_advisory=[],
            _review_history=[],
            _scope_review_history={},
            _last_scope_raw_results=[
                {"slot_id": "stale", "model_id": "old-scope", "status": "responded"}
            ],
        )
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._run_unified_review", return_value=None):
                with mock.patch.object(pr_mod, "run_scope_review", side_effect=RuntimeError("scope crashed")):
                    review_err, scope_result, triad_block_reason, _ = pr_mod.run_parallel_review(
                        ctx, "test commit")

        assert review_err is None
        assert triad_block_reason == ""
        assert scope_result.blocked is True
        assert scope_result.status == "error"
        assert ctx._last_scope_raw_results
        assert ctx._last_scope_raw_results[0]["status"] == "error"
        assert ctx._last_scope_raw_results[0]["slot_id"] == "scope_slot_error"
        assert ctx._last_scope_raw_results[0]["model_id"] != "old-scope"
        assert ctx._last_scope_raw_result["raw_results"][0]["status"] == "error"

    def test_advisory_freshness_path_aware(self):
        """_check_advisory_freshness must accept paths parameter."""
        git = _get_module("ouroboros.tools.git")
        sig = inspect.signature(git._check_advisory_freshness)
        assert "paths" in sig.parameters


# ---------------------------------------------------------------------------
# HEAD snapshot section tests (Phase 3, item 5)
# ---------------------------------------------------------------------------

class TestHeadSnapshotSection:
    def _git_commit(self, cwd, message, allow_empty=False):
        """Helper to commit with identity configured for CI/clean machines."""
        cmd = ["git", "-c", "user.email=test@ouroboros", "-c", "user.name=TestBot", "commit", "-m", message]
        if allow_empty:
            cmd.append("--allow-empty")
        subprocess.run(cmd, cwd=str(cwd), capture_output=True)

    def test_new_file_shows_no_head_snapshot(self, tmp_path):
        """New files (not in HEAD) should note 'File is new — no HEAD snapshot'."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "empty init", allow_empty=True)
        # Add a new file (not committed yet)
        (tmp_path / "newfile.py").write_text("print('new')", encoding="utf-8")

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["newfile.py"])
        assert "File is new" in result
        assert "no HEAD snapshot" in result

    def test_existing_file_shows_old_content(self, tmp_path):
        """Modified files should show the HEAD (old) content in the snapshot."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "existing.py").write_text("OLD_CONTENT_V1", encoding="utf-8")
        subprocess.run(["git", "add", "existing.py"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        # Modify the file
        (tmp_path / "existing.py").write_text("NEW_CONTENT_V2", encoding="utf-8")

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["existing.py"])
        assert "OLD_CONTENT_V1" in result
        assert "NEW_CONTENT_V2" not in result  # HEAD snapshot, not current

    def test_deleted_file_shows_old_content(self, tmp_path):
        """Deleted files should show their old HEAD content."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "deleted.py").write_text("CONTENT_BEFORE_DELETE", encoding="utf-8")
        subprocess.run(["git", "add", "deleted.py"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        (tmp_path / "deleted.py").unlink()

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["deleted.py"])
        assert "CONTENT_BEFORE_DELETE" in result

    def test_new_file_not_confused_with_git_error(self, tmp_path, monkeypatch):
        """git show non-zero for a new file must say 'File is new', not 'error'."""
        import subprocess as sp_module

        class FakeNewFileResult:
            returncode = 128
            stdout = ""
            stderr = "fatal: path 'newfile.py' does not exist in 'HEAD'"

        original_run = sp_module.run
        def mock_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and "show" in cmd:
                return FakeNewFileResult()
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(sp_module, "run", mock_run)

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["newfile.py"])
        assert "File is new" in result
        assert "no HEAD snapshot" in result
        # Must NOT render as a git error
        assert "HEAD snapshot error" not in result

    def test_real_git_error_not_mislabeled_as_new_file(self, tmp_path, monkeypatch):
        """Real git failures (bad object, corrupt repo) must render as 'HEAD snapshot error',
        not silently as 'File is new — no HEAD snapshot'.
        """
        import subprocess as sp_module

        class FakeGitErrorResult:
            returncode = 128
            stdout = ""
            stderr = "fatal: bad object HEAD"

        original_run = sp_module.run
        def mock_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and "show" in cmd:
                return FakeGitErrorResult()
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(sp_module, "run", mock_run)

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["existing.py"])
        # Must render as an error, not as a new file
        assert "HEAD snapshot error" in result
        assert "File is new" not in result

    def test_binary_file_omitted_cleanly(self, tmp_path):
        """Binary files (e.g. .png) must produce an omission note, not garbage bytes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\xff" * 100)
        subprocess.run(["git", "add", "logo.png"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        (tmp_path / "logo.png").unlink()

        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, ["logo.png"])
        # Must produce an omission note, not binary garbage
        assert "omitted" in result.lower() or "binary" in result.lower()
        # Must not contain raw binary bytes
        assert "\x00" not in result
        assert "\xff" not in result

    def test_empty_paths_returns_placeholder(self, tmp_path):
        """Empty paths list returns a placeholder."""
        mod = _get_module("ouroboros.tools.review_helpers")
        result = mod.build_head_snapshot_section(tmp_path, [])
        assert "no touched files" in result

    def test_scope_prompt_omits_head_snapshots_section(self, tmp_path):
        """v4.33.0: _build_scope_prompt MUST NOT include a separate 'Pre-change snapshots' section.

        The staged diff already shows every removed line via `-`, and the full
        repo pack covers cross-module context. Removing the separate section
        saves ~164K tokens (~21% of the scope budget) on a typical repo.
        """
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n"
        , encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("ORIGINAL", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        (tmp_path / "a.py").write_text("MODIFIED", encoding="utf-8")
        subprocess.run(["git", "add", "a.py"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, _ = mod._build_scope_prompt(tmp_path, "test commit")
        # The dedicated HEAD snapshot section is gone in v4.33.0
        assert "Pre-change snapshots" not in prompt
        # New content must still appear in current files section
        assert "MODIFIED" in prompt
        # The old (`ORIGINAL`) content is still observable through the staged
        # diff's `-` lines — we don't assert on its presence because some
        # helper test setups may produce minimal diff context.

    def test_scope_prompt_does_not_import_head_snapshot_helper(self):
        """v4.33.0: scope_review.py no longer imports build_head_snapshot_section.

        The helper itself is kept in review_helpers.py for plan_task (which
        has no diff to draw from), but scope_review has no legitimate use
        for it anymore — the assertion guards against accidental reintroduction.

        The check looks for actual use (import or call-site), not bare
        mentions — a comment referring to the helper by name is
        informational cross-reference, not a regression.
        """
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod)
        # No import line referencing the helper
        assert "import build_head_snapshot_section" not in source
        assert "    build_head_snapshot_section," not in source
        # No call-site
        assert "build_head_snapshot_section(" not in source

    def test_scope_prompt_inlines_deleted_file_content(self, tmp_path):
        """Deleted files must still appear in 'Current touched files' with DELETED marker.

        Without the separate HEAD snapshots section we'd lose visibility into
        what was removed. _inline_deleted_file_pack restores it by embedding
        HEAD content right inside Current touched files.
        """
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n"
        , encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "removed.py").write_text("ORIGINAL_DELETED_CONTENT", encoding="utf-8")
        (tmp_path / "keep.py").write_text("keep_me", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        # Delete one file, keep the other — ensure scope prompt builds & shows both
        (tmp_path / "removed.py").unlink()
        subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "delete removed.py")
        assert prompt is not None, f"scope prompt build failed with status={status}"
        assert "DELETED" in prompt
        assert "ORIGINAL_DELETED_CONTENT" in prompt

    def test_deleted_sensitive_file_content_suppressed(self, tmp_path):
        """Deleting a tracked `.env` must not inline its HEAD content (v4.33.0).

        Defense-in-depth — the staged diff itself still shows removed lines,
        but `_inline_deleted_file_pack` MUST NOT duplicate sensitive content
        into the scope prompt. A `*(DELETED — sensitive ...; content
        suppressed)*` marker replaces the fenced HEAD block.
        """
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n",
            encoding="utf-8",
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / ".env").write_text("SECRET_TOKEN=sk-abc-DEADBEEF", encoding="utf-8")
        (tmp_path / "keep.py").write_text("keep_me", encoding="utf-8")
        # `-f` forces add even if a global gitignore excludes `.env`
        subprocess.run(["git", "add", "-f", ".env", "keep.py", "docs"],
                        cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")

        (tmp_path / ".env").unlink()
        subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "remove .env")
        assert prompt is not None, f"scope prompt build failed with status={status}"
        assert "DELETED" in prompt
        assert "sensitive" in prompt.lower()
        assert "content suppressed" in prompt.lower()
        # _inline_deleted_file_pack must NOT echo the secret payload. Note:
        # the staged diff below it still shows `-SECRET_TOKEN=...` through
        # git's own output — but the inline-pack copy is the only layer we
        # control in scope_review, and that copy must be clean.
        inline_header = "## Current touched files"
        diff_header = "## Staged diff"
        inline_start = prompt.index(inline_header)
        diff_start = prompt.index(diff_header)
        inline_section = prompt[inline_start:diff_start]
        assert "DEADBEEF" not in inline_section
        assert "SECRET_TOKEN" not in inline_section

    def test_deletion_only_diff_not_blocked(self, tmp_path):
        """Deletion-only diffs must reach scope reviewer, not be fail-closed."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T",
             "commit", "--allow-empty", "-m", "empty init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n"
        , encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "to_delete.py").write_text("CONTENT_TO_DELETE", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T",
             "commit", "-m", "add file"],
            cwd=str(tmp_path), capture_output=True,
        )
        # Stage a deletion
        (tmp_path / "to_delete.py").unlink()
        subprocess.run(["git", "add", "to_delete.py"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, omitted = mod._build_scope_prompt(tmp_path, "delete to_delete.py")
        # Must NOT be blocked (omitted should be None for deletion-only)
        assert omitted is None
        # HEAD snapshot must show old content
        assert "CONTENT_TO_DELETE" in prompt
        # Current files section must note the deletion
        assert "DELETED" in prompt

    def test_renamed_file_shows_old_head_content(self, tmp_path):
        """Renamed files must show old HEAD content (from old path), not 'File is new'."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n"
        , encoding="utf-8")
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "old_name.py").write_text("ORIGINAL_RENAME_CONTENT", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=T",
             "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        # Rename the file
        (tmp_path / "old_name.py").rename(tmp_path / "new_name.py")
        subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, omitted = mod._build_scope_prompt(tmp_path, "rename old_name to new_name")
        # Omission must be None — rename is handled correctly
        assert omitted is None
        # Old content must appear in HEAD snapshot (from old_name.py HEAD)
        assert "ORIGINAL_RENAME_CONTENT" in prompt


# ---------------------------------------------------------------------------
# LLM routing validation (Phase 3, item 6)
# ---------------------------------------------------------------------------

class TestSharedLLMRouting:
    def test_triad_review_uses_llm_client(self):
        """Triad review (_query_model) must use LLMClient, not ad-hoc HTTP."""
        mod = _get_module("ouroboros.tools.review")
        source = inspect.getsource(mod._query_model)
        assert "LLMClient" in source or "llm_client" in source.lower()
        # Must NOT use requests or httpx directly
        assert "requests.post" not in source
        assert "httpx" not in source

    def test_triad_emits_llm_usage_events(self):
        """Triad review must use the shared review usage emitter."""
        mod = _get_module("ouroboros.tools.review")
        source = inspect.getsource(mod._multi_model_review_async)
        assert "emit_review_usage" in source
        helper = inspect.getsource(_get_module("ouroboros.tools.review_helpers").emit_review_usage)
        assert "llm_usage" in helper
        assert "emit_review_event" in helper

    def test_scope_review_uses_llm_client(self):
        """Scope review must use LLMClient for its model call.

        LLMClient is used in _call_scope_llm (called by run_scope_review),
        so we check the whole module for its presence rather than just
        the top-level run_scope_review function.
        """
        mod = _get_module("ouroboros.tools.scope_review")
        # LLMClient is instantiated in _call_scope_llm which run_scope_review delegates to
        source = inspect.getsource(mod._call_scope_llm)
        assert "LLMClient" in source

    def test_scope_review_emits_usage(self):
        """Scope review must emit llm_usage event for cost tracking."""
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod)
        assert 'emit_review_usage(ctx, model=scope_model_id, usage=_usage, source="scope_review")' in source
        helper = inspect.getsource(_get_module("ouroboros.tools.review_helpers").emit_review_usage)
        assert "llm_usage" in helper
        assert "emit_review_event" in helper


# ---------------------------------------------------------------------------
# Advisory schema enrichment
# ---------------------------------------------------------------------------

class TestAdvisorySchemaEnriched:
    def test_advisory_schema_has_goal_scope_paths(self):
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        tools = adv.get_tools()
        adv_tool = next(t for t in tools if t.name == "advisory_review")
        props = adv_tool.schema["parameters"]["properties"]
        assert "goal" in props
        assert "scope" in props
        assert "paths" in props

    def test_advisory_prompt_uses_section_loader(self):
        """Advisory prompt builder must use precise section loader, not full CHECKLISTS.md."""
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(adv._build_advisory_prompt)
        assert "load_checklist_section" in source

    def test_advisory_no_blind_truncation(self):
        """Advisory must not silently truncate raw_result."""
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(adv._handle_advisory_pre_review)
        assert "raw_result[:4000]" not in source


class TestScopePromptMatrixContract:
    """v4.34.0: scope prompt requires full 8-item matrix + anti-pattern-lock guard.

    Regression-pins two behavioural contracts added in v4.34.0:
    (1) scope reviewer must emit one entry per Intent/Scope checklist item
        (not only FAILs as before), with mandatory PASS justification;
    (2) scope prompt carries an explicit Anti pattern-lock guard asking
        the reviewer to do a second focused pass on a different concern
        class whenever exactly one FAIL is surfaced.
    """

    def _get_scope_prompt(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@o", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("bbb", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "test")
        assert prompt is not None, f"unexpected non-None status: {status}"
        return prompt

    def test_full_matrix_contract_is_present(self, tmp_path):
        """Scope prompt must require coverage for every checklist item."""
        prompt = self._get_scope_prompt(tmp_path)
        assert "cover every checklist item" in prompt
        assert "Skipping an item is not allowed" in prompt
        assert "multiple distinct concrete problems" in prompt

    def test_pass_justification_is_mandatory(self, tmp_path):
        """PASS entries must require 1-2 sentences of justification.

        Guard: without this, reviewers can return bare `PASS` for items
        they never actually reviewed, defeating the matrix contract.
        """
        prompt = self._get_scope_prompt(tmp_path)
        # Some form of mandatory justification language must be present.
        assert "stating WHY this item passes" in prompt
        # And the bare-PASS anti-pattern must be called out explicitly.
        assert "bare" in prompt.lower()
        assert "reviewer failure" in prompt.lower()

    def test_anti_pattern_lock_guard_is_present(self, tmp_path):
        """Scope prompt must carry the Anti pattern-lock guard section."""
        prompt = self._get_scope_prompt(tmp_path)
        assert "Anti pattern-lock guard" in prompt
        assert "exactly one FAIL" in prompt
        # The guard must instruct a second pass on a different concern class.
        # Normalize whitespace before checking so a reflow of the prompt
        # wrapping doesn't break the contract.
        import re
        flat = re.sub(r"\s+", " ", prompt)
        assert "SECOND pass" in flat
        assert "DIFFERENT concern class" in flat

    def test_anti_pattern_lock_pairings_cover_checklist_items(self, tmp_path):
        """Concrete pairings must reference real Intent/Scope checklist item names.

        Without real item names the guidance is generic and models fall
        back to pattern-locking; the prompt has to name pairings by
        actual checklist identifiers.
        """
        prompt = self._get_scope_prompt(tmp_path)
        # At least the four most common concern classes must appear as
        # "if FAIL was in X, re-examine Y" pairings.
        for item in (
            "intent_alignment",
            "forgotten_touchpoints",
            "cross_surface_consistency",
            "regression_surface",
        ):
            assert item in prompt, f"Anti-pattern-lock pairing for `{item}` missing"


class TestTriadPromptAntiPatternLock:
    """v4.34.0: triad pre-commit review prompt now also carries the
    Anti pattern-lock guard. Scope and triad must stay symmetric so
    single-FAIL pattern-lock is guarded on both review surfaces.
    """

    def test_triad_template_has_anti_pattern_lock_guard(self):
        mod = _get_module("ouroboros.tools.review")
        tpl = mod._REVIEW_PROMPT_TEMPLATE
        assert "Anti pattern-lock guard" in tpl
        assert "exactly one FAIL" in tpl
        # Normalize whitespace so prompt reflow doesn't break the contract.
        import re
        flat = re.sub(r"\s+", " ", tpl)
        # Accept any casing — "different concern class" / "DIFFERENT concern class"
        assert "concern class" in flat.lower()
        assert "second pass" in flat.lower()


def test_scope_reviewer_window_fail_closed_on_absent_evidence(monkeypatch, tmp_path):
    """claudexor B4 + v6.46.0 false-1M fix: with NO capability evidence, ONLY the
    SHIPPED designated reviewer (_SCOPE_MODEL_DEFAULT) is granted the 1M blocking
    sentinel under blocking_1m. An OFF-DEFAULT reviewer (e.g. an
    OUROBOROS_SCOPE_REVIEW_MODEL pin) fails closed to the sub-floor under BOTH floors —
    so the P3 authority downgrades it visibly instead of silently treating a 200K model
    as 1M and overflowing its real window into a provider 400. A non-default >=1M
    reviewer must be owner-acked to regain 1M."""
    from ouroboros.tools import scope_review as sr
    from ouroboros import capability_evidence
    from types import SimpleNamespace

    # Isolated, empty evidence -> no model gets Capability Evidence.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        capability_evidence,
        "probe",
        lambda *a, **k: SimpleNamespace(window_tokens=0),
    )

    # An OFF-DEFAULT reviewer with no evidence fails closed under advisory...
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_FLOOR", "advisory")
    w_adv = sr._scope_reviewer_window("gigachat::GigaChat-3-Ultra")
    assert 0 < w_adv < sr._SCOPE_MODEL_CONTEXT_WINDOW, w_adv  # fail-closed sub-floor

    # ...AND under blocking_1m (the v6.46.0 bug: a pinned off-default 200K model that
    # used to be wrongly trusted as 1M now fails closed instead of overflowing).
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_FLOOR", "blocking_1m")
    w_offdefault = sr._scope_reviewer_window("anthropic/claude-sonnet-4.5")
    assert w_offdefault == sr._SCOPE_FAILCLOSED_WINDOW, w_offdefault

    # The SHIPPED designated reviewer keeps the 1M sentinel under blocking_1m.
    w_designated = sr._scope_reviewer_window(sr._SCOPE_MODEL_DEFAULT)
    assert w_designated == sr._SCOPE_MODEL_CONTEXT_WINDOW, w_designated

    # Direct-provider and explicit OpenRouter spellings of the same shipped reviewer
    # are also the designated default. Regression guard for openai::gpt-5.5 being
    # reduced to bare "gpt-5.5" and then misclassified as off-default.
    w_direct = sr._scope_reviewer_window("openai::gpt-5.5")
    assert w_direct == sr._SCOPE_MODEL_CONTEXT_WINDOW, w_direct
    w_openrouter = sr._scope_reviewer_window("openrouter::openai/gpt-5.5")
    assert w_openrouter == sr._SCOPE_MODEL_CONTEXT_WINDOW, w_openrouter


def test_scope_reviewer_window_uses_scope_slot_route_not_main(monkeypatch, tmp_path):
    """Capability Evidence for scope review must use the scope slot's route.

    A local-routed main lane (`USE_LOCAL_MAIN=true`) must not turn a remote direct
    OpenAI scope reviewer into a local route lookup.
    """
    from types import SimpleNamespace
    from ouroboros import capability_evidence, config
    from ouroboros.tools import scope_review as sr

    captured = {}

    def fake_probe(drive_root, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(window_tokens=333_333)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda: {
            "USE_LOCAL_MAIN": True,
            "OPENAI_BASE_URL": "https://api.openai.test/v1",
        },
    )
    monkeypatch.setattr(capability_evidence, "probe", fake_probe)

    assert sr._scope_reviewer_window("openai::gpt-5.5") == 333_333
    assert captured["provider"] == "openai"
    assert captured["model"] == "openai::gpt-5.5"
    assert captured["base_url"] == "https://api.openai.test/v1"
    assert captured["use_local"] is False
