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
import pathlib
import subprocess
import sys
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_module(name):
    sys.path.insert(0, REPO)
    return importlib.import_module(name)


def test_review_thoroughness_is_count_free_and_evidence_bound():
    helpers = _get_module("ouroboros.tools.review_helpers")
    block = helpers.REVIEW_THOROUGHNESS_BLOCK

    assert "5 bugs" not in block
    assert "zero, one, or many findings are all valid" in block
    assert "Never invent a finding to increase the count" in block


def test_scope_review_uses_active_subject_and_system_governance(tmp_path, monkeypatch):
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    governance = tmp_path / "system"
    subject = tmp_path / "subject"
    drive = tmp_path / "data"
    governance.mkdir()
    subject.mkdir()
    drive.mkdir()
    captured = {}

    def fake_build(repo_dir, _message, **kwargs):
        captured["subject"] = pathlib.Path(repo_dir)
        captured["governance"] = pathlib.Path(kwargs["context"].governance_repo_dir)
        return None, mod._TouchedContextStatus(status="empty")

    monkeypatch.setattr(mod, "_build_scope_prompt", fake_build)
    ctx = registry.ToolContext(
        repo_dir=governance,
        system_repo_dir=governance,
        workspace_root=subject,
        workspace_mode="external",
        drive_root=drive,
    )

    mod.run_scope_review(ctx, "review external subject", scope_model="test-scope")

    assert captured == {
        "subject": subject.resolve(),
        "governance": governance.resolve(),
    }


def test_scope_review_refuses_ambiguous_workspace_root(tmp_path):
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    system = tmp_path / "system"
    subject = tmp_path / "subject"
    drive = tmp_path / "data"
    system.mkdir()
    subject.mkdir()
    drive.mkdir()
    ctx = registry.ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        workspace_root=subject,
        workspace_mode="",
        drive_root=drive,
    )

    result = mod.run_scope_review(ctx, "must not inspect the wrong repo")

    assert result.blocked is True
    assert result.status == "error"
    assert "workspace_root is set without workspace_mode" in result.block_message


def test_managed_resolver_enables_binary_metadata_context(tmp_path, monkeypatch):
    """SUPERSESSION (lane L-review, Δ4): represent_binary now follows the managed
    REVIEW SUBJECT (predicate + authorized tx artifact), not the raw predicate
    alone — the same production condition, established one seam deeper. The
    subject itself must also reach the prompt builder."""
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    admission = _get_module("ouroboros.tools.review_admission")
    subject_mod = _get_module("ouroboros.tools.review_subject")
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    captured = {}

    def fake_build(_repo_dir, _message, **kwargs):
        captured["represent_binary"] = kwargs["context"].represent_binary
        captured["managed_subject"] = kwargs["context"].managed_subject
        return None, mod._TouchedContextStatus(status="empty")

    monkeypatch.setattr(mod, "_build_scope_prompt", fake_build)
    fake_subject = object()
    monkeypatch.setattr(
        subject_mod, "managed_review_subject", lambda _ctx, _repo: fake_subject
    )
    assert admission  # the prepare half resolves the seams patched above
    ctx = registry.ToolContext(repo_dir=repo, drive_root=drive, task_id="resolver")

    result = mod.run_scope_review(ctx, "review assisted update", scope_model="test")

    assert result.blocked is True
    assert captured == {"represent_binary": True, "managed_subject": fake_subject}


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

    def test_represents_binary_with_exact_git_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@ouroboros"],
            cwd=str(tmp_path), check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "TestBot"],
            cwd=str(tmp_path), check=True,
        )
        binary = tmp_path / "native.so"
        binary.write_bytes(b"old\x00payload")
        subprocess.run(["git", "add", "-f", "native.so"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        binary.write_bytes(b"new\x00payload")
        subprocess.run(["git", "add", "native.so"], cwd=str(tmp_path), check=True)

        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["native.so"], represent_binary=True
        )

        assert omitted == []
        assert "staged blob" in pack
        assert "pre-merge HEAD blob" in pack
        assert "official MERGE_HEAD blob" in pack
        assert "unknown" not in pack

    def test_binary_metadata_without_stage_zero_stays_omitted(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        (tmp_path / "native.so").write_bytes(b"unstaged\x00payload")

        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["native.so"], represent_binary=True
        )

        assert omitted == ["native.so"]
        assert "no readable stage-0" in pack

    def test_staged_binary_deletion_has_exact_parent_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@ouroboros"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True)
        binary = tmp_path / "logo.png"
        binary.write_bytes(b"png\x00payload")
        subprocess.run(["git", "add", "logo.png"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "rm", "logo.png"], cwd=str(tmp_path), check=True)

        helpers = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = helpers.build_touched_file_pack(
            tmp_path, ["logo.png"], represent_binary=True
        )
        scope = _get_module("ouroboros.tools.scope_review")
        scope_pack = scope._inline_deleted_file_pack(
            "", ["logo.png"], tmp_path, represent_binary=True
        )

        assert omitted == []
        assert "staged blob: `absent (deletion)`" in pack
        assert "pre-merge HEAD:" in pack
        assert "staged blob: `absent (deletion)`" in scope_pack

    def test_extensionless_binary_deletion_has_exact_parent_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@ouroboros"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True)
        binary = tmp_path / "firmware"
        binary.write_bytes(b"firmware\x00payload")
        subprocess.run(["git", "add", "firmware"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "rm", "firmware"], cwd=str(tmp_path), check=True)

        helpers = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = helpers.build_touched_file_pack(
            tmp_path, ["firmware"], represent_binary=True
        )
        scope = _get_module("ouroboros.tools.scope_review")
        scope_pack = scope._inline_deleted_file_pack(
            "", ["firmware"], tmp_path, represent_binary=True
        )

        assert omitted == []
        assert "staged blob: `absent (deletion)`" in pack
        assert "pre-merge HEAD:" in pack
        assert "staged blob: `absent (deletion)`" in scope_pack

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
        """#284 successor: compact coverage IS the atlas form — every gather is
        compact from the first call and there is no fuller form to retry
        from (the old full->compact retry rung is deleted by design)."""
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
            return "COMPACT ATLAS"

        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        monkeypatch.setattr(scope_pack, "_gather_scope_packs", fake_gather)

        prompt, omitted = mod._build_scope_prompt(tmp_path, "test commit")

        assert omitted is None
        assert calls and all(calls)  # compact-only, from the very first gather
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
            raise mod._ScopeAtlasNotAssembled({"estimated_total_tokens": 900_001})

        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        monkeypatch.setattr(scope_pack, "_gather_scope_packs", fake_gather)
        monkeypatch.setattr(mod, "estimate_tokens", lambda _text: 800_000)

        prompt, status = mod._build_scope_prompt(tmp_path, "test commit")

        assert prompt is None
        assert status.status == "fixed_overflow"
        assert status.token_count > 0
        # Compact is the only atlas form: every ladder gather asked for it.
        assert calls and all(calls)

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
        large_diff = "diff --git a/tiny.py b/tiny.py\n" + (" unchanged context\n" * 30_000)
        compact_diff = "diff --git a/tiny.py b/tiny.py\n@@ -1 +1 @@\n-old\n+new\n"
        compact_calls = []

        def fake_capture(_repo_dir, *, unified=3):
            if unified == 0:
                compact_calls.append(True)
                return compact_diff
            return large_diff

        monkeypatch.setattr(mod, "capture_staged_diff", fake_capture)
        monkeypatch.setattr(mod, "_effective_scope_input_limit", lambda **_kw: 100_000)
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
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
        monkeypatch.setattr(
            mod, "_gather_scope_packs",
            lambda *_a, **_k: (_ for _ in ()).throw(
                mod._ScopeAtlasNotAssembled({"estimated_total_tokens": 999_999})
            ),
        )
        monkeypatch.setattr(mod, "estimate_tokens", lambda _text: 800_000)
        # Capability Evidence: gigachat KNOWN sub-floor (131K), fable-5 >=1M.
        monkeypatch.setattr(
            mod, "_scope_window",
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
            lambda *a, **k: (raw, {
                "prompt_tokens": 10, "completion_tokens": 5,
                "operation_id": "review-op", "operation_state": "late_settled",
            }, None),
        )

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="test-scope")

        assert result.blocked is True
        assert result.status == "parse_failure"
        assert "missing required items" in result.block_message
        assert result.parsed_items[0]["item"] == "intent_alignment"
        assert result.operation_id == "review-op"
        assert result.operation_state == "late_settled"

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
        monkeypatch.setattr(mod, "_call_scope_llm", lambda *a, **k: (
            "", {"operation_id": "oversize-op", "late_result_pending": True}, oversize_error,
        ))
        monkeypatch.setattr(mod, "_scope_window",
                            lambda _m, **_k: mod.ReviewerWindow(1_000_000, "confirmed"))

        result = mod.run_scope_review(MockCtx(), "test commit", scope_model="anthropic/claude-fable-5")

        assert result.blocked is True
        assert result.status == "fixed_overflow"
        assert result.operation_id == "oversize-op"
        assert result.late_result_pending is True
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
        instead of a deterministic provider 400. The limit is computed PER CALL from
        the measured/cold density (v6.80.0), never from an import-time constant."""
        mod = _get_module("ouroboros.tools.scope_review")
        from ouroboros.tools.review_helpers import calibrated_input_token_limit
        # opus-4.8 KNOWN sub-floor (200K) via evidence; everything else >=1M.
        monkeypatch.setattr(
            mod, "_scope_window",
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

    def test_scope_review_default_is_terra(self):
        mod = _get_module("ouroboros.tools.scope_review")
        assert "gpt-5.6-terra" in mod._SCOPE_MODEL_DEFAULT
        # Verify the getter returns the shipped default when no override env var is set
        import os
        if not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL"):
            assert "gpt-5.6-terra" in mod._get_scope_model()
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
        """The dynamic review prompt template must include current_files_section."""
        mod = _get_module("ouroboros.tools.review")
        assert "{current_files_section}" in mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC

    def test_triad_prompt_has_goal_section(self):
        """The dynamic review prompt template must include goal_section (the
        per-commit tail; the stable prefix carries the cache marker)."""
        mod = _get_module("ouroboros.tools.review")
        assert "{goal_section}" in mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC
        assert "{goal_section}" not in mod._REVIEW_PROMPT_TEMPLATE_STABLE

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
        # (Q25-A two-phase contract: assembly, then dispatch).
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "_prepare_unified_review" in parallel_source
        assert "_dispatch_unified_review" in parallel_source
        # scope dispatch lives one seam deeper since the Q25-A split
        from ouroboros.tools import parallel_review as _pr
        assert "run_scope_review" in inspect.getsource(_pr._run_scope)
        # ThreadPoolExecutor must be used for parallel execution
        assert "ThreadPoolExecutor" in parallel_source

    def test_repo_commit_not_bypass_scope(self):
        """repo_commit must reach scope review via the shared stage helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._repo_commit_push)
        assert "_run_reviewed_stage_cycle" in source
        shared_source = inspect.getsource(git._run_reviewed_stage_cycle)
        # The advisory-freshness check lives in the extracted gate helper the
        # stage cycle calls before any paid dispatch.
        assert "_advisory_and_tests_gate" in shared_source
        assert "_check_advisory_freshness" in inspect.getsource(git._advisory_and_tests_gate)
        assert "_run_parallel_review" in shared_source
        parallel_source = inspect.getsource(git._run_parallel_review)
        from ouroboros.tools import parallel_review as _pr
        assert "run_scope_review" in inspect.getsource(_pr._run_scope)
        assert "ThreadPoolExecutor" in parallel_source

    def test_parallel_execution_both_always_run(self):
        """SUPERSESSION (lane L-review, Q25=A): the retired contract was
        "both futures always submitted regardless of each other's result" — it
        let one side SPEND while the other failed assembly deterministically.
        The ratified ordering: BOTH packets are assembled first, and both
        dispatches are submitted to the pool only past the admission; each
        submission still precedes any result() collection."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_parallel_review)
        prepare_triad = source.find("_prepare_unified_review(")
        prepare_scope = source.find("_prepare_scope_rows(")
        # Each submission runs under a copy of the admitting context (one fence
        # for admission and reservation); the anchors name the submitted seam.
        submit_triad = source.find("copy_context().run, _dispatch_unified_review")
        submit_scope = source.find("copy_context().run, _run_scope")
        result_triad = source.find("triad_fut.result()")
        result_scope = source.find("scope_fut.result()")
        for position in (prepare_triad, prepare_scope, submit_triad, submit_scope,
                         result_triad, result_scope):
            assert position > 0
        # Assembly of BOTH sides precedes ANY dispatch submission...
        assert prepare_triad < submit_triad and prepare_triad < submit_scope
        assert prepare_scope < submit_triad and prepare_scope < submit_scope
        # ...and both submissions precede their result() collection.
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
        fake_slot = types.SimpleNamespace(
            model="new-scope", slot_id="scope_slot_1", route=None, effort="",
            session_target="", session_profile="")
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._prepare_unified_review",
                            return_value=(None, None, True)):
                with mock.patch.object(
                        pr_mod, "_prepare_scope_rows",
                        return_value=[{"slot": fake_slot, "prepared": {"p": 1}, "final": None}]):
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
        result, included = mod.build_head_snapshot_section(tmp_path, ["newfile.py"])
        assert "File is new" in result
        assert "no HEAD snapshot" in result
        assert "newfile.py" not in included  # no snapshot text -> not claimable

    def test_existing_file_shows_old_content(self, tmp_path):
        """Modified files should show the HEAD (old) content in the snapshot."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "existing.py").write_text("OLD_CONTENT_V1", encoding="utf-8")
        subprocess.run(["git", "add", "existing.py"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        # Modify the file
        (tmp_path / "existing.py").write_text("NEW_CONTENT_V2", encoding="utf-8")

        mod = _get_module("ouroboros.tools.review_helpers")
        result, included = mod.build_head_snapshot_section(tmp_path, ["existing.py"])
        assert "OLD_CONTENT_V1" in result
        assert "existing.py" in included  # full snapshot present -> claimable
        assert "NEW_CONTENT_V2" not in result  # HEAD snapshot, not current

    def test_current_payload_snapshot_uses_collision_safe_fence(self, tmp_path):
        """Fenced examples inside SKILL.md must not escape the snapshot block."""
        payload = tmp_path / "SKILL.md"
        payload.write_bytes(b"Example:\n```python\nprint('safe')\n```\n")

        mod = _get_module("ouroboros.tools.review_helpers")
        result, included = mod.build_head_snapshot_section(
            tmp_path,
            ["data/skills/external/alpha/SKILL.md"],
            current_snapshots={"data/skills/external/alpha/SKILL.md": payload},
        )

        assert "````md\nExample:" in result
        assert "\n````\n" in result
        assert "```python\nprint('safe')\n```" in result
        assert "data/skills/external/alpha/SKILL.md" in included

    def test_deleted_file_shows_old_content(self, tmp_path):
        """Deleted files should show their old HEAD content."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "deleted.py").write_text("CONTENT_BEFORE_DELETE", encoding="utf-8")
        subprocess.run(["git", "add", "deleted.py"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        (tmp_path / "deleted.py").unlink()

        mod = _get_module("ouroboros.tools.review_helpers")
        result, included = mod.build_head_snapshot_section(tmp_path, ["deleted.py"])
        assert "CONTENT_BEFORE_DELETE" in result
        assert "deleted.py" in included

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
        result, included = mod.build_head_snapshot_section(tmp_path, ["newfile.py"])
        assert "File is new" in result
        assert "no HEAD snapshot" in result
        # Must NOT render as a git error
        assert "HEAD snapshot error" not in result
        assert "newfile.py" not in included

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
        result, included = mod.build_head_snapshot_section(tmp_path, ["existing.py"])
        # Must render as an error, not as a new file
        assert "HEAD snapshot error" in result
        assert "File is new" not in result
        assert "existing.py" not in included

    def test_binary_file_omitted_cleanly(self, tmp_path):
        """Binary files (e.g. .png) must produce an omission note, not garbage bytes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\xff" * 100)
        subprocess.run(["git", "add", "logo.png"], cwd=str(tmp_path), capture_output=True)
        self._git_commit(tmp_path, "init")
        (tmp_path / "logo.png").unlink()

        mod = _get_module("ouroboros.tools.review_helpers")
        result, included = mod.build_head_snapshot_section(tmp_path, ["logo.png"])
        # Must produce an omission note, not binary garbage
        assert "omitted" in result.lower() or "binary" in result.lower()
        # Must not contain raw binary bytes
        assert "\x00" not in result
        assert "\xff" not in result
        assert "logo.png" not in included

    def test_empty_paths_returns_placeholder(self, tmp_path):
        """Empty paths list returns a placeholder."""
        mod = _get_module("ouroboros.tools.review_helpers")
        result, included = mod.build_head_snapshot_section(tmp_path, [])
        assert "no touched files" in result
        assert included == frozenset()

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

    def test_triad_emits_llm_usage_once_via_substrate(self):
        """Triad usage is emitted exactly ONCE, by the shared review substrate.

        The former job-level re-emit in _multi_model_review_async doubled every
        triad call in llm_usage telemetry and mislabelled a delegated session's
        provider: the substrate per-slot emission is the single source, the
        same shape scope review already received.
        """
        source = inspect.getsource(_get_module("ouroboros.tools.review"))
        assert 'source="review"' not in source  # no job-level re-emit
        substrate = inspect.getsource(_get_module("ouroboros.review_substrate"))
        assert 'source=f"review_substrate:{request.surface}"' in substrate
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

    def test_scope_review_emits_usage_once_via_substrate(self):
        """Scope usage is emitted exactly ONCE, by the shared review substrate.

        The former job-level re-emit in run_scope_review duplicated every scope
        call in llm_usage telemetry without ledger_attempt_ids (v6.69.0 dedup):
        the substrate per-slot emission is the single telemetry source.
        """
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod)
        assert 'source="scope_review")' not in source  # no job-level re-emit
        substrate = inspect.getsource(_get_module("ouroboros.review_substrate"))
        assert 'source=f"review_substrate:{request.surface}"' in substrate
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
        class without imposing a numeric finding quota.
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
        assert "exactly one FAIL" not in prompt
        # The guard must instruct a second pass on a different concern class.
        # Normalize whitespace before checking so a reflow of the prompt
        # wrapping doesn't break the contract.
        import re
        flat = re.sub(r"\s+", " ", prompt)
        assert "zero or one FAIL is valid" in flat
        assert "numeric finding quota" in flat
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
    semantic breadth is guarded without pressuring either surface to invent findings.
    """

    def test_triad_template_has_anti_pattern_lock_guard(self):
        mod = _get_module("ouroboros.tools.review")
        tpl = mod._REVIEW_PROMPT_TEMPLATE_STABLE + mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC
        assert "Anti pattern-lock guard" in tpl
        assert "exactly one FAIL" not in tpl
        guard = mod.REPO_ANTI_PATTERN_LOCK_GUARD
        # Normalize whitespace so prompt reflow doesn't break the contract.
        import re
        flat = re.sub(r"\s+", " ", f"{tpl}\n{guard}")
        assert "zero or one FAIL is valid" in flat
        assert "numeric finding quota" in flat
        # Accept any casing — "different concern class" / "DIFFERENT concern class"
        assert "concern class" in flat.lower()
        assert "second pass" in flat.lower()


def test_scope_reviewer_window_fail_closed_on_absent_evidence(monkeypatch, tmp_path):
    """claudexor B4 + v6.46.0 false-1M fix: with NO capability evidence an OFF-DEFAULT
    reviewer (e.g. an OUROBOROS_SCOPE_REVIEW_MODEL pin) fails closed to the conservative
    sub-floor SIZE, instead of silently treating a 200K model as 1M and overflowing its
    real window into a provider 400. The SHIPPED designated reviewer keeps the 1M
    sentinel as a SIZE so the review is still dispatched — but NEITHER carries blocking
    authority, because a model acquires no authority from its name (BIBLE P3: a window
    that cannot be established by sourced Capability Evidence is treated as too small)."""
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

    # An OFF-DEFAULT reviewer with no evidence fails closed to the sub-floor...
    w_adv = sr._scope_window("gigachat::GigaChat-3-Ultra")
    assert 0 < w_adv.window_tokens < sr._SCOPE_MODEL_CONTEXT_WINDOW, w_adv

    # ...as does a pinned off-default 200K model (the v6.46.0 bug: it used to be
    # wrongly trusted as 1M and overflowed).
    w_offdefault = sr._scope_window("anthropic/claude-sonnet-4.5")
    assert w_offdefault.window_tokens == sr._SCOPE_FAILCLOSED_WINDOW, w_offdefault

    # The SHIPPED designated reviewer keeps the 1M sentinel as a SIZING number...
    w_designated = sr._scope_window(sr._SCOPE_MODEL_DEFAULT)
    assert w_designated.window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW, w_designated

    # Direct-provider and explicit OpenRouter spellings of the same shipped reviewer
    # are also the designated default. Regression guard for a provider spelling
    # (openai::/openrouter::) being misclassified as off-default.
    for spelling in ("openai::gpt-5.6-terra", "openrouter::openai/gpt-5.6-terra"):
        assert sr._scope_window(spelling).window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW

    # ...and NONE of them — the designated default least of all — may block a commit
    # on that invented number. Authority is computed from the evidence, not the name.
    for model in (
        "gigachat::GigaChat-3-Ultra", "anthropic/claude-sonnet-4.5",
        sr._SCOPE_MODEL_DEFAULT, "openai::gpt-5.6-terra",
        "openrouter::openai/gpt-5.6-terra",
    ):
        assert sr._scope_window(model).blocking_authority_allowed is False, model


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

    assert sr._scope_window("openai::gpt-5.5").window_tokens == 333_333
    assert captured["provider"] == "openai"
    assert captured["model"] == "openai::gpt-5.5"
    assert captured["base_url"] == "https://api.openai.test/v1"
    assert captured["use_local"] is False


def test_parallel_commit_scope_is_one_substantive_call(monkeypatch, tmp_path):
    """P3 wrapper must not fan a budget result into a second degraded call."""
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools.scope_review import ScopeReviewResult

    calls = []

    def fake_scope(_ctx, _message, **kwargs):
        calls.append((kwargs.get("scope_model"), kwargs.get("degraded", False)))
        return ScopeReviewResult(
            blocked=False,
            status="budget_exceeded",
            model_id=str(kwargs.get("scope_model") or ""),
        )

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        task_id="one-pass-scope",
        _review_history=[],
        _review_advisory=[],
        _scope_review_history={},
    )
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(parallel_review, "run_scope_review", fake_scope)
    monkeypatch.setattr(config, "get_scope_review_models", lambda: ["scope/model"])
    monkeypatch.setattr(
        review, "_prepare_unified_review", lambda *_a, **_k: (None, None, True)
    )
    from ouroboros.tools import review_admission
    monkeypatch.setattr(
        review_admission, "prepare_scope_review",
        lambda *_a, **_k: ({"packet": 1}, None),
    )

    parallel_review.run_parallel_review(ctx, "test commit")

    assert calls == [("scope/model", False)]


# --- v6.80.0: scope review follows the owner-only context mode -----------------

def test_low_context_mode_skips_scope_review_with_a_typed_evidence_row(monkeypatch, tmp_path):
    """RS2: in owner-selected `low` mode no reviewer is called, the commit is not
    gated on scope, and the skip leaves a TYPED durable row on the same
    review-evidence surface that carries fail-closed results — so a low-mode commit
    is never forensically confusable with "scope review silently failed" (BIBLE P1).

    The one-window provenance tombstone must be explicit `false` here: bare env
    Low remains effective sizing Low but resolves owner intent fail-closed to Max."""
    from ouroboros import config
    from ouroboros.tools import review_helpers
    from ouroboros.tools import scope_review as sr

    class _Ctx:
        repo_dir = str(tmp_path)
        task_id = "low-mode-skip"
        pending_events = []

        def drive_logs(self):
            return tmp_path

    called = []
    monkeypatch.setattr(sr, "_call_scope_llm", lambda *a, **k: called.append(1) or ("", None, ""))
    monkeypatch.setattr(sr, "_build_scope_prompt", lambda *a, **k: called.append(1) or ("p", None))
    monkeypatch.setattr(config, "get_context_mode", lambda: "low")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")

    result = sr.run_scope_review(_Ctx(), "test commit", scope_model="anthropic/claude-fable-5")

    assert called == [], "low mode must not call the reviewer or even assemble a prompt"
    assert result.blocked is False
    assert result.status == "skipped_low_context_mode"
    assert any(
        f.get("item") == "scope_review_skipped_low_context_mode"
        for f in result.advisory_findings
    )
    record = review_helpers.build_scope_actor_record(result, fallback_model_id="x")
    assert record["status"] == "skipped_low_context_mode"
    assert record["prompt_chars_source"] == "not_assembled"

    # max mode (the unchanged DEFAULT) still assembles and calls.
    monkeypatch.setattr(config, "get_context_mode", lambda: "max")
    sr.run_scope_review(_Ctx(), "test commit", scope_model="anthropic/claude-fable-5")
    assert called, "max mode must still run scope review"


def test_default_context_mode_is_max_and_agent_cannot_lower_it(monkeypatch):
    """RS2 anti-regression: the DEFAULT behaviour is unchanged (max ⇒ blocking scope
    gate), and the agent still cannot reach the setting that now also switches scope
    review off — on the settings merge, the shell guard, or the browser guard."""
    from ouroboros import config
    from ouroboros.gateway.settings import _merge_settings_payload
    from ouroboros.tools.browser import _blocks_context_mode_self_lowering_js
    from ouroboros.tools.registry import _detect_context_mode_self_lowering

    assert config.SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"] == "max"
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE", raising=False)
    assert config.get_context_mode() == "max"

    merged = _merge_settings_payload({"OUROBOROS_CONTEXT_MODE": "max"},
                                     {"OUROBOROS_CONTEXT_MODE": "low"})
    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"
    assert _detect_context_mode_self_lowering(
        "save_settings({'ouroboros_context_mode': 'low'})"
    ) is True
    assert _blocks_context_mode_self_lowering_js(
        "fetch('/api/owner/context-mode', {body: JSON.stringify({mode: 'low'})})"
    ) is True


def test_window_provenance_wording_is_five_way():
    """RS5: the cases must read differently — a conservative fallback must not be
    reported with the same words as a confirmed measurement, and an EXPIRED record
    must not be reported with the same words as a live one."""
    from ouroboros.tools import scope_review as sr

    phrases = {
        sr._window_provenance_phrase(200_000, sr._WINDOW_CONFIRMED),
        sr._window_provenance_phrase(200_000, sr._WINDOW_ASSERTED),
        sr._window_provenance_phrase(200_000, sr._WINDOW_UNKNOWN),
        sr._window_provenance_phrase(1_000_000, sr._WINDOW_STALE),
        sr._window_provenance_phrase(1_000_000, sr._WINDOW_SENTINEL),
    }
    assert len(phrases) == 5
    assert "confirmed" in sr._window_provenance_phrase(200_000, sr._WINDOW_CONFIRMED)
    assert "owner-asserted" in sr._window_provenance_phrase(200_000, sr._WINDOW_ASSERTED)
    assert "unknown window" in sr._window_provenance_phrase(200_000, sr._WINDOW_UNKNOWN)
    assert "designated-default" in sr._window_provenance_phrase(1_000_000, sr._WINDOW_SENTINEL)
    assert "EXPIRED" in sr._window_provenance_phrase(1_000_000, sr._WINDOW_STALE)

    # The label is read off the EVIDENCE, so a stale 1M record can never be labelled
    # (or worded) as a confirmed one just because its number clears the floor.
    stale = sr.ReviewerWindow(1_000_000, "confirmed", stale=True)
    assert sr._scope_window_provenance(stale) == sr._WINDOW_STALE
    assert sr._scope_window_provenance(sr.ReviewerWindow(250_000)) == sr._WINDOW_UNKNOWN


def test_ladder_steps_are_recorded_once_aggregated(tmp_path, monkeypatch):
    """RS5: the guaranteed-fit ladder leaves ONE aggregated field in the existing
    context manifest — not an event per step, and not silence."""
    from ouroboros.tools import scope_review as sr

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(sr, "run_cmd", lambda cmd, cwd=None: (
        "M\ta.py" if "--name-status" in cmd else "diff --git a/a.py b/a.py\n+x = 1\n"
    ))
    monkeypatch.setattr(sr, "capture_staged_diff",
                        lambda _repo, **_k: "diff --git a/a.py b/a.py\n+x = 1\n")
    monkeypatch.setattr(sr, "_gather_scope_packs", lambda *a, **k: "ATLAS")
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_k: 900_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt
    manifest = sr._current_scope_context_manifest()
    steps = manifest.get("ladder_steps")
    assert isinstance(steps, list) and len(steps) == 1
    assert steps[0]["step"] == "compact_atlas"  # compact is the only atlas form (#284)
    assert set(steps[0]) >= {"tokens_before", "tokens_after", "diff_only_files", "deficit"}


def _repo_with_oversized_required_prompt(tmp_path, required_bytes=935_000):
    """A repo whose UNCHANGED `prompts/` artifact cannot fit any atlas budget."""
    import subprocess

    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
    (tmp_path / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    (tmp_path / "prompts").mkdir(exist_ok=True)
    # Force-included by prefix => `required`, and never touched by this commit.
    (tmp_path / "prompts" / "huge.md").write_text("x" * required_bytes, encoding="utf-8")
    (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    (tmp_path / "ok.py").write_text("print(2)\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)


def test_unassembled_required_terminal_names_the_artifact_not_a_phantom_overflow(
    tmp_path, monkeypatch,
):
    """BIBLE P1/P3. The ladder terminates on TWO different failures. When it ends
    because a REQUIRED artifact never assembled, the owner-facing block must say
    so — not reuse the irreducible-prompt story, whose own quoted token count
    contradicts the budget it claims to exceed, and whose remedy ("split the
    staged diff") cannot shrink an UNCHANGED artifact. The refusal is also a
    ladder STEP: a terminal with an empty trace explains nothing after the fact."""
    from ouroboros.tools import scope_review as sr

    _repo_with_oversized_required_prompt(tmp_path)
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 200_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    assert status.unassembled_required == ["prompts/huge.md"]  # cause carried
    # The trace records the refusal steps, naming what did not assemble.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert [s for s in steps if s["step"] == "atlas_refused"], steps
    assert steps[0]["unassembled_required"] == ["prompts/huge.md"]

    result = sr._handle_prompt_signals(prompt, status, input_limit=200_000, scope_model="")
    assert "prompts/huge.md" in result.block_message
    # The false cause and its self-contradicting comparison are gone.
    assert "irreducible scope prompt" not in result.block_message
    assert f"({200_000})" not in result.block_message
    assert "Split the commit into smaller staged diffs" not in result.block_message


def test_sub_floor_terminal_reports_the_same_cause_as_the_1m_terminal(monkeypatch):
    """The twin: `budget_exceeded` (sub-floor reviewer) is the other authority
    branch of the same terminal and made the identical false claim. Fixing one
    branch and leaving its sibling is the defect, not the fix. The genuine
    overflow wording must survive on both."""
    from ouroboros.tools import scope_review as sr

    monkeypatch.setattr(
        sr, "_scope_window", lambda _m, **_k: sr.ReviewerWindow(window_tokens=200_000, status="confirmed"),
    )
    missing = sr._TouchedContextStatus(
        status="budget_exceeded", token_count=3_672,
        unassembled_required=["prompts/huge.md"],
    )
    result = sr._handle_prompt_signals(None, missing, input_limit=200_000, scope_model="m/x")
    assert "prompts/huge.md" in result.block_message
    assert "prompts/huge.md" in result.advisory_findings[0]["reason"]
    assert "cannot fit the irreducible scope prompt" not in result.block_message
    assert "Full scope-review prompt" not in result.advisory_findings[0]["reason"]

    # The half that must NOT change: a real overflow still reports an overflow.
    overflow = sr._TouchedContextStatus(status="budget_exceeded", token_count=990_000)
    plain = sr._handle_prompt_signals(None, overflow, input_limit=200_000, scope_model="m/x")
    assert "irreducible scope prompt" in plain.block_message
    assert "~990000 estimated tokens" in plain.advisory_findings[0]["reason"]


def test_mixed_terminal_reports_both_causes_and_the_mixed_remedy(tmp_path, monkeypatch):
    """The MIXED terminal: the refusal that dropped a required artifact was itself
    a hard-budget overflow (even the content-free manifest did not fit beside the
    fixed prompt). Reporting only the missing artifact prescribes
    ATLAS_MISSING_ARTIFACT_REMEDY — "narrowing the reviewed change cannot help" —
    which is false for, and cannot resolve, the overflow half. Both causes ride
    the terminal, the trace, and the owner-facing block."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools.review_context_atlas import ATLAS_MIXED_ASSEMBLY_REMEDY

    _repo_with_oversized_required_prompt(tmp_path)
    # An input budget so small the atlas hard allowance is zero: ANY rendered
    # manifest overflows, while required prompts/huge.md was already dropped.
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 6_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    assert status.unassembled_required == ["prompts/huge.md"]  # cause 1 carried
    assert status.atlas_overflowed is True                     # cause 2 carried
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    refused = [s for s in steps if s["step"] == "atlas_refused"]
    assert refused, steps
    assert refused[0]["atlas_overflowed"] is True

    result = sr._handle_prompt_signals(prompt, status, input_limit=6_000, scope_model="")
    # Both causes are rendered — neither shadows the other…
    assert "prompts/huge.md" in result.block_message
    assert "content-free atlas manifest" in result.block_message
    # …and the remedy is the mixed one, not the single-cause half-truth that
    # cannot resolve the overflow.
    assert ATLAS_MIXED_ASSEMBLY_REMEDY in result.block_message
    assert "narrowing the reviewed change cannot help" not in result.block_message


def test_mixed_sub_floor_terminal_reports_the_same_two_causes(monkeypatch):
    """The sub-floor authority branch is the twin surface of the same terminal:
    it must render the identical mixed cause and remedy."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools.review_context_atlas import ATLAS_MIXED_ASSEMBLY_REMEDY

    monkeypatch.setattr(
        sr, "_scope_window", lambda _m, **_k: sr.ReviewerWindow(window_tokens=200_000, status="confirmed"),
    )
    mixed = sr._TouchedContextStatus(
        status="budget_exceeded", token_count=3_672,
        unassembled_required=["prompts/huge.md"], atlas_overflowed=True,
    )
    result = sr._handle_prompt_signals(None, mixed, input_limit=200_000, scope_model="m/x")
    for text in (result.block_message, result.advisory_findings[0]["reason"]):
        assert "prompts/huge.md" in text
        assert "content-free atlas manifest" in text
    assert ATLAS_MIXED_ASSEMBLY_REMEDY in result.advisory_findings[0]["reason"]
    assert "narrowing the reviewed change cannot help" not in result.advisory_findings[0]["reason"]


def test_diff_only_degradation_is_not_reported_as_fully_included(tmp_path, monkeypatch):
    """P1. When the ladder drops a touched file's full snapshot, the durable
    coverage manifest must say so. `already_included` was re-derived from ALL
    touched paths instead of the surviving `kept` set that `_render_touched_section`
    owns, so a file whose snapshot had just been removed was still recorded as
    "included in fixed prompt context" — a claim the prompt itself contradicts."""
    import subprocess

    from ouroboros.tools import scope_review as sr

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    # Two equal-sized big files: the ladder degrades only the first.
    (tmp_path / "big_a.py").write_text("x = 1\n" * 40_000, encoding="utf-8")
    (tmp_path / "big_b.py").write_text("y = 1\n" * 40_000, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny change inside huge files: the diff fits, the snapshots do not.
    (tmp_path / "big_a.py").write_text("z = 0\n" + "x = 1\n" * 39_999, encoding="utf-8")
    (tmp_path / "big_b.py").write_text("z = 0\n" + "y = 1\n" * 39_999, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 120_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt is not None
    assert "TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt and "- big_a.py" in prompt
    assert "### big_b.py" in prompt  # this one kept its snapshot
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    # The degraded file is disclosed as diff-only, the intact one is not.
    assert "full snapshot omitted" in rows["big_a.py"]["reason"]
    assert rows["big_b.py"]["reason"] == "included in fixed prompt context"


def test_design_skipped_touched_test_is_not_claimed_as_fully_included(tmp_path):
    """XG-1R4.1 / P1. `_gather_scope_packs` used to derive `already_included` from
    the touched LIST (`all_touched_paths`), while `_build_scope_prompt` omits the
    full snapshots of touched TESTS by design (`current_skipped_by_design`). The
    durable row for a touched test therefore read "included in fixed prompt
    context" while NO full snapshot of it existed anywhere in the pack — the same
    false-coverage-claim class as XR-4/XG-1R.4, on the last surface still
    deriving the claim instead of being told it.

    `already_included` is now the CONSERVATIVE set the fixed part really carries,
    so the touched test falls through to the atlas, where (being an anchor, hence
    related to the change and not excludable under BIBLE P3) it is supplied in
    FULL exactly once. FULL is the spacious-budget DEFAULT, not a guarantee:
    under budget pressure the guaranteed-fit ladder may degrade a touched test
    to diff-only (the constrained sibling below) — constitutionally sound
    because the test's complete changes ride the staged diff. The invariant
    under test is the general one: a coverage row may never claim content the
    pack does not contain."""
    import subprocess

    from ouroboros.tools import scope_review as sr

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    # An INDENTED marker far from the change: unreachable through the staged diff
    # (outside any -U3 hunk, and never picked as git's hunk-header funcname), so
    # finding it in the prompt proves a real full snapshot.
    body = ["def test_a():", "    UNCHANGED_TEST_BODY_MARKER_ZZZ = 1"]
    body += [f"    filler_{idx} = {idx}" for idx in range(40)]
    body += ["    assert True"]
    (tmp_path / "tests" / "test_thing.py").write_text(
        "\n".join(body) + "\n", encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    (tmp_path / "tests" / "test_thing.py").write_text(
        "\n".join(body) + "\n\n\ndef test_b():\n    assert True\n", encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    row = rows["tests/test_thing.py"]
    # The false claim is gone…
    assert row["reason"] != "included in fixed prompt context"
    assert row["disposition"] != "already_included"
    # …and the pack really carries what the row now says: the atlas supplied the
    # touched test in full, so the unchanged body reached the reviewer.
    assert row["disposition"] == "full"
    assert "UNCHANGED_TEST_BODY_MARKER_ZZZ" in prompt
    # The dedup note still explains the non-duplication in the fixed part.
    assert "DEDUPLICATION NOTE" in prompt and "- tests/test_thing.py" in prompt
    # The touched non-test keeps its true `already_included` claim.
    assert rows["mod.py"]["reason"] == "included in fixed prompt context"
    # Spacious budget: the ladder never reached for the test — no degradation.
    assert "TOUCHED FILE BUDGET DEGRADATION NOTE" not in prompt


def _ladder_repo(tmp_path, files: dict, changes: dict):
    """Init a git repo with ``files``, then stage ``changes``.

    Values: str -> text file, bytes -> binary file, None -> delete the path
    (``git add .`` stages removals too)."""
    import subprocess

    def _put(rel, content):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            path.unlink()
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    for rel, content in files.items():
        _put(rel, content)
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    for rel, content in changes.items():
        _put(rel, content)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)


# ~90K estimated tokens of test body; the INDENTED marker is unreachable through
# the staged diff (outside any -U3 hunk of an end-of-file change, and never a
# hunk-header funcname), so its absence from the prompt proves the full
# snapshot is really gone from the whole pack.
_BIG_TEST_BODY = "\n".join(
    ["def test_big():", "    UNCHANGED_BIG_TEST_MARKER_QQQ = 1"]
    + ["    filler = 1"] * 24_000
    + ["    assert True"]
) + "\n"
_BIG_TEST_CHANGED = _BIG_TEST_BODY + "\n\ndef test_added():\n    assert True\n"


def test_constrained_budget_degrades_touched_test_to_diff_only(tmp_path, monkeypatch):
    """Phase L, the constrained sibling of the spacious pin above. A touched
    test is filtered out of the fixed part's snippets by design, yet the atlas
    is owed it as a FULL anchor — and the ladder built its degradable set only
    from `current_context_paths`, so one oversized touched test structurally
    sank pack assembly with `required_artifact_omitted` although its complete
    changes sat in the staged diff. Touched tests now ride the ladder's free
    tier: under pressure they degrade to diff-only via the existing
    `diff_only_included` mechanism, assembly SUCCEEDS, and the manifest row
    carries the disclosure."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={"tests/test_big.py": _BIG_TEST_BODY, "mod.py": "x = 1\n"},
        changes={"tests/test_big.py": _BIG_TEST_CHANGED, "mod.py": "x = 2\n"},
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    # Assembly SUCCEEDS — the oversized touched test no longer sinks the pack.
    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The ladder degraded the test, disclosed in the prompt…
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    # …and the full snapshot is truly gone from the whole pack.
    assert "UNCHANGED_BIG_TEST_MARKER_QQQ" not in prompt
    # The dedup note no longer lists it: that would claim an atlas snapshot.
    assert "CURRENT FILE CONTEXT DEDUPLICATION NOTE" not in prompt
    # The durable coverage row carries the diff-only disclosure — not a false
    # full-inclusion claim, not a required omission.
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    row = rows["tests/test_big.py"]
    assert row["disposition"] == "already_included"
    assert "changes included" in row["reason"]
    assert "full snapshot omitted" in row["reason"]
    # The small touched module keeps its true full-snapshot claim.
    assert rows["mod.py"]["reason"] == "included in fixed prompt context"


def test_touched_test_degrades_before_the_required_tier_and_zero_context_diff(
    tmp_path, monkeypatch,
):
    """Ordering. Touched tests join the FREE tier of the two-tier ladder sort
    (`atlas_required_beyond_diff` is False for tests/), so under deficit the
    big touched test degrades BEFORE the ladder reaches for the -U0 rung or
    the required tier: the required-beyond-diff artifact keeps its full
    snapshot in the fixed part and no zero-context-diff step is recorded."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={
            "tests/test_big.py": _BIG_TEST_BODY,
            # ~10K tokens: fits the fixed part; owed in full regardless of size.
            "prompts/mini_prompt.md": "word here\n" * 4_000,
        },
        changes={
            "tests/test_big.py": _BIG_TEST_CHANGED,
            "prompts/mini_prompt.md": "CHANGED\n" + "word here\n" * 3_999,
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The required-beyond-diff artifact kept its full snapshot in the fixed part…
    assert "### prompts/mini_prompt.md" in prompt
    # …the degradation note names the test, and only the test…
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "prompts/mini_prompt.md" not in note
    # …and the ladder never needed the -U0 rung: the free tier covered it.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert steps and not any(s.get("zero_context_diff") for s in steps), steps
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/mini_prompt.md"]["reason"] == "included in fixed prompt context"
    assert "full snapshot omitted" in rows["tests/test_big.py"]["reason"]


def test_canonical_doc_is_never_ladder_degraded_to_diff_only(tmp_path, monkeypatch):
    """The boundary of Phase L: only tests/ paths joined the degradable set. A
    touched CANONICAL doc is owed in full through the fixed part's
    canonical-docs section (`atlas_required_beyond_diff` is True), so when it
    alone overflows the budget the ladder exhausts its free rungs (including
    -U0) and fails CLOSED — it never hands the doc to diff-only."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={
            # ~84K tokens injected whole into the canonical-docs section.
            "docs/ARCHITECTURE.md": "arch doc line\n" * 24_000,
            "mod.py": "x = 1\n",
        },
        changes={
            "docs/ARCHITECTURE.md": "CHANGED\n" + "arch doc line\n" * 23_999,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"
    # The canonical doc never became a required omission via diff-only…
    assert status.unassembled_required == []
    assert status.atlas_overflowed is True
    # …its coverage row keeps the truthful fixed-part claim…
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["docs/ARCHITECTURE.md"]["reason"] == "included in fixed prompt context"
    # …and the ladder really exhausted the free rungs (-U0 attempted) first.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert any(s.get("zero_context_diff") for s in steps), steps


def test_binary_test_fixture_is_never_degraded_to_diff_only(tmp_path, monkeypatch):
    """A binary fixture under tests/ has NO changes in the staged text diff
    (`git diff --cached` renders "Binary files differ"), so the diff-only
    disclosure "changes included in the fixed staged diff" would be a false
    claim. Binary staged test paths (`staged_path_is_binary`) stay out of the
    degradable tier; the fixture remains the atlas's business (typed
    `binary_media` row). The fixture makes the binary the LARGEST touched test,
    so a candidates list without the binary filter would degrade the binary
    first and fail the note assertions below."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={
            "tests/fixture.bin": bytes(range(256)) * 1_600,  # ~400KB, biggest
            "tests/test_big.py": _BIG_TEST_BODY,
        },
        changes={
            "tests/fixture.bin": b"\x00CHANGED" + bytes(range(256)) * 1_600,
            "tests/test_big.py": _BIG_TEST_CHANGED,
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The TEXT test rode the diff-only rung; the binary fixture did not.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "tests/fixture.bin" not in note
    # The binary fixture stays honestly delegated to the atlas (typed row,
    # never the diff-only "changes included" claim).
    dedup = prompt.split("## CURRENT FILE CONTEXT DEDUPLICATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/fixture.bin" in dedup
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["tests/fixture.bin"]["disposition"] == "binary_media"
    assert "full snapshot omitted" not in rows["tests/fixture.bin"]["reason"]
    assert "full snapshot omitted" in rows["tests/test_big.py"]["reason"]


def test_deleted_text_test_degrades_to_diff_only_under_pressure(tmp_path, monkeypatch):
    """A deleted TEXT test inlines its whole HEAD snapshot into the fixed part
    (`_inline_deleted_file_pack`) and the ladder had no rung for it — pure
    pressure with no relief, although a text deletion's complete content is
    already the staged diff's own minus-lines. Deleted text tests now join the
    same degradable tier: under pressure the HEAD inline is replaced by a
    disclosed omission marker. Binary deletions never qualify (their content is
    not in the text diff). NB: mod.py co-degrades here because the
    refusal-branch deficit has a pre-existing 50K floor that outsizes the
    deleted test — a ladder property, not an effect of this fix."""
    from ouroboros.tools import scope_review as sr

    # ~30K tokens of deleted test: inline + its diff minus-lines both ride the
    # fixed part (~65K total), overflowing the 45K budget; dropping the inline
    # (~30K) brings the prompt back under it.
    gone_body = "\n".join(
        ["def test_gone():", "    DELETED_TEST_HEAD_MARKER_WWW = 1"]
        + ["    filler = 1"] * 8_000
        + ["    assert True"]
    ) + "\n"
    _ladder_repo(
        tmp_path,
        files={
            "tests/test_gone.py": gone_body,
            "tests/gone.bin": bytes(range(256)) * 4,  # small binary deletion
            "mod.py": "x = 1\n",
        },
        changes={
            "tests/test_gone.py": None,
            "tests/gone.bin": None,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The deleted text test was degraded: no HEAD inline, disclosed marker
    # instead, and the degradation note names it.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_gone.py" in note
    assert "full HEAD snapshot omitted" in prompt
    assert "*(DELETED — content from HEAD)*" not in prompt
    # The binary deletion did NOT ride the diff-only rung: it keeps its own
    # typed suppression marker (its content is not in the text diff).
    assert "tests/gone.bin" not in note
    assert "### tests/gone.bin\n\n*(DELETED — " in prompt
    assert "content suppressed" in prompt


def test_degraded_test_gets_no_false_atlas_delegation_phrase(tmp_path, monkeypatch):
    """The dedup note used to promise unconditionally that a touched test's
    "full snapshot appears once in the generated atlas" — false the moment the
    ladder degrades that test to diff-only. The reviewer-facing phrase is now
    conditional and per-file: full-delegated tests stay listed in the dedup
    note, budget-degraded ones move to the degradation note, and the
    unconditional phrase is gone."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={
            "tests/test_huge.py": _BIG_TEST_BODY,
            "tests/test_tiny.py": "def test_tiny():\n    TINY_KEPT_MARKER_JJJ = 1\n",
        },
        changes={
            "tests/test_huge.py": _BIG_TEST_CHANGED,
            "tests/test_tiny.py": (
                "def test_tiny():\n    TINY_KEPT_MARKER_JJJ = 1\n    assert True\n"
            ),
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The unconditional promise is gone from the reviewer-facing prompt…
    assert "appears once in the generated atlas" not in prompt
    # …the conditional wording rides the dedup note, which lists ONLY the
    # test still delegated in full…
    dedup = prompt.split("## CURRENT FILE CONTEXT DEDUPLICATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_tiny.py" in dedup
    assert "tests/test_huge.py" not in dedup
    assert "move to the degradation note instead" in prompt
    # …and each file's actual disposition backs the wording: tiny delegated in
    # full by the atlas, huge disclosed as diff-only.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_huge.py" in note
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["tests/test_tiny.py"]["disposition"] == "full"
    assert "full snapshot omitted" in rows["tests/test_huge.py"]["reason"]


def test_deleted_non_test_file_is_never_degraded(tmp_path, monkeypatch):
    """Boundary pin for the deleted branch: ONLY tests/ deletions join the
    degradable tier. A deleted ordinary module keeps its HEAD inline even under
    pressure (here it alone overflows the budget, so the ladder exhausts its
    rungs and fails CLOSED) — a mutation dropping the tests/-filter from the
    deleted branch would degrade it and assemble, flipping every assert below.
    The named `diff_only_paths` ladder trace proves it was never degraded."""
    from ouroboros.tools import scope_review as sr

    gone_body = "\n".join(["def helper():"] + ["    filler = 1"] * 8_000) + "\n"
    _ladder_repo(
        tmp_path,
        files={"mod_big.py": gone_body, "mod.py": "x = 1\n"},
        changes={"mod_big.py": None, "mod.py": "x = 2\n"},
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert steps
    for step in steps:
        assert "mod_big.py" not in (step.get("diff_only_paths") or []), step


def test_deleted_test_token_estimate_orders_largest_first(tmp_path, monkeypatch):
    """The cat-file size fallback in `_touched_token_estimate` is load-bearing:
    deleted paths have no worktree stat, and a fallback that returned 0 would
    (a) sort every deleted test LAST instead of largest-first and (b) count 0
    freed tokens per pop, so the loop would drain the whole tier. With honest
    estimates the LARGER deleted test alone covers the deficit and the smaller
    one keeps its HEAD inline."""
    from ouroboros.tools import scope_review as sr

    big_gone = "\n".join(["def test_gone_big():"] + ["    filler = 1"] * 16_000) + "\n"
    small_gone = "\n".join(["def test_gone_small():"] + ["    filler = 1"] * 2_100) + "\n"
    _ladder_repo(
        tmp_path,
        files={
            "tests/test_gone_big.py": big_gone,
            "tests/test_gone_small.py": small_gone,
            "mod.py": "x = 1\n",
        },
        changes={
            "tests/test_gone_big.py": None,
            "tests/test_gone_small.py": None,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 135_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # Guarded extraction: a missing note must fail as an assert, not IndexError.
    assert "## TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    # Largest-first: the big deleted test degraded, the small one did not.
    assert "- tests/test_gone_big.py" in note
    assert "tests/test_gone_small.py" not in note
    assert "### tests/test_gone_big.py\n\n*(DELETED — full HEAD snapshot omitted" in prompt
    assert "### tests/test_gone_small.py\n\n*(DELETED — content from HEAD)*" in prompt


def test_oversized_deleted_test_keeps_suppressed_marker_not_diff_only(
    tmp_path, monkeypatch,
):
    """A deleted test over `_DELETED_INLINE_MAX_BYTES` is ALREADY a suppressed
    marker — its inline never weighed on the budget, so degrading it would free
    phantom tokens (the HEAD-blob estimate, ~277K here) and misattribute the
    relief. The size guard keeps it out of the degradable tier: pressure is
    relieved by the genuine candidate (the big CURRENT test), and the oversized
    deletion keeps its own typed suppression marker."""
    from ouroboros.tools import scope_review as sr

    huge_gone = "\n".join(["def test_huge_gone():"] + ["    filler = 1"] * 74_000) + "\n"
    assert len(huge_gone.encode()) > 1_048_576  # over the inline cap
    _ladder_repo(
        tmp_path,
        files={"tests/test_huge_gone.py": huge_gone, "tests/test_big.py": _BIG_TEST_BODY},
        changes={"tests/test_huge_gone.py": None, "tests/test_big.py": _BIG_TEST_CHANGED},
    )
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 330_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The oversized deletion kept its typed suppression marker…
    assert "content > 1024 KB; suppressed" in prompt
    assert "## TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    # …the genuine candidate carried the degradation, not the phantom one.
    assert "- tests/test_big.py" in note
    assert "tests/test_huge_gone.py" not in note
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    for step in steps:
        assert "tests/test_huge_gone.py" not in (step.get("diff_only_paths") or []), step


def test_a_renamed_test_fixture_is_not_degraded(tmp_path, monkeypatch):
    """Conservative rename guard: a renamed path's staged diff may carry only a
    rename header (no content hunks), so degrading it to diff-only could hide
    its content entirely. Renamed touched tests keep their snapshot; the plain
    modified test still rides the diff-only rung."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(
        tmp_path,
        files={"tests/old_name.bin": bytes(range(256)) * 1_600,
               "tests/test_big.py": _BIG_TEST_BODY},
        changes={"tests/test_big.py": _BIG_TEST_CHANGED},
    )
    subprocess.run(["git", "mv", "tests/old_name.bin", "tests/new_name.bin"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "new_name.bin" not in note and "old_name.bin" not in note


def test_staged_diff_capture_survives_non_utf8_text(tmp_path, monkeypatch):
    """Git calls NUL-free non-UTF-8 content TEXT, so those bytes ride ordinary diff
    lines. The old strict-UTF-8 text capture raised on them and the review continued
    on a "(failed to get staged diff)" placeholder — with the ladder able to degrade
    a touched test to diff-only, that placeholder can be a file's only evidence. The
    bytes now arrive reversibly escaped and the pack assembles."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(tmp_path, files={"tests/test_bytes.py": "x = 1\n"}, changes={"mod.py": "y = 1\n"})
    (tmp_path / "tests" / "test_bytes.py").write_bytes(b"x = 1  # latin caf\xe9\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert "\\xe9" in prompt  # reversible escape, not a U+FFFD flattening
    assert "�" not in prompt
    assert "failed to get staged diff" not in prompt


def test_unavailable_staged_diff_blocks_instead_of_reviewing_a_placeholder(tmp_path, monkeypatch):
    """When the canonical staged diff cannot be captured at all, prompt assembly
    fails with a RuntimeError — the type `scope_review`'s own caller already turns
    into a blocked, fail-closed result — instead of sending an authoritative review
    a placeholder that says the evidence is missing."""
    from ouroboros.tools import review_binary_context as rbc
    from ouroboros.tools import scope_review as sr

    _ladder_repo(tmp_path, files={"mod.py": "x = 1\n"}, changes={"mod.py": "x = 2\n"})

    def broken(*_a, **_k):
        raise rbc.StagedDiffUnavailable("staged diff capture failed (rc 128): fatal")

    monkeypatch.setattr(sr, "capture_staged_diff", broken)

    assert issubclass(rbc.StagedDiffUnavailable, RuntimeError)
    with pytest.raises(RuntimeError):
        sr._build_scope_prompt(tmp_path, "test commit")


def test_ladder_cannot_degrade_a_required_beyond_diff_artifact_to_diff_only(
    tmp_path, monkeypatch,
):
    """XR-4 end-to-end, #284 successor. An artifact owed in full regardless of
    the change (here a `prompts/` file) NEVER enters the diff-only tier: the
    self-defeating rung is gone, the artifact stays delivered whole (fixed
    prompt context), ordinary files degrade and -U0 is attempted, and an
    irreducible misfit closes with the typed fixed_overflow — with NOTHING
    omitted, because refusing to degrade is exactly what kept it whole."""
    import subprocess

    from ouroboros.tools import scope_review as sr

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "prompts").mkdir()
    # The prompts/ artifact is the LARGEST touched file: degraded first.
    (tmp_path / "prompts" / "big_prompt.md").write_text(
        "word here\n" * 45_000, encoding="utf-8",
    )
    (tmp_path / "big_b.py").write_text("y = 1\n" * 40_000, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny changes inside huge files: the diff fits, the snapshots do not.
    (tmp_path / "prompts" / "big_prompt.md").write_text(
        "CHANGED\n" + "word here\n" * 44_999, encoding="utf-8",
    )
    (tmp_path / "big_b.py").write_text("z = 0\n" + "y = 1\n" * 39_999, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 120_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    # #284 successor contract: the required-beyond-diff artifact is NEVER
    # handed to the diff-only tier (the atlas would refuse such a pack by
    # design, so that rung could only manufacture a refusal). It stays owed
    # in full — here delivered via the fixed prompt context — and nothing is
    # omitted: the typed overflow says the configured input limit is simply
    # too small for the owed-in-full content.
    assert status.unassembled_required == []
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    for step in steps:
        assert "prompts/big_prompt.md" not in (step.get("diff_only_paths") or []), step
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/big_prompt.md"]["disposition"] == "already_included"
    # The ordinary file DID ride the disclosed diff-only rung, and -U0 was
    # attempted before the ladder closed.
    assert any(step.get("diff_only_files") for step in steps), steps
    assert any(step.get("zero_context_diff") for step in steps), steps


def test_ladder_degrades_ordinary_files_before_a_required_artifact(tmp_path, monkeypatch):
    """Defect B. The ladder sorted ALL touched paths by size with no requiredness
    filter, so the LARGEST file was degraded first even when it was an artifact
    owed in full. Degrading one of those can never buy a fitting pack — the atlas
    turns it into an assembly refusal (`required_artifact_omitted`), which the
    ladder then reads as a further deficit and degrades further. The deficit was
    manufactured: the ordinary files alone covered it.

    The fixture is deliberately shaped so a size-only sort CANNOT pass it: the
    required artifact is the LARGEST touched file (~40K tokens), while the three
    ordinary files are individually smaller (~20K each) but collectively cover
    the deficit. Pre-fix this reaches the terminal with
    `status == "fixed_overflow"` and `unassembled_required ==
    ["prompts/large_prompt.md"]`; post-fix the three ordinary files degrade, the
    prompt's full snapshot survives, and the pack assembles."""
    import subprocess

    from ouroboros.tools import scope_review as sr

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "prompts").mkdir()
    # ~40K touched tokens: the LARGEST touched file, and owed in full.
    (tmp_path / "prompts" / "large_prompt.md").write_text(
        "word here\n" * 16_000, encoding="utf-8",
    )
    ordinary = ["a_mod.py", "b_mod.py", "c_mod.py"]
    for name in ordinary:
        # ~20K touched tokens each: individually smaller than the artifact,
        # together (~60K) more than the ~50K deficit.
        (tmp_path / name).write_text("y = 1\n" * 13_334, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny changes inside big files: the diff fits, the snapshots do not.
    (tmp_path / "prompts" / "large_prompt.md").write_text(
        "CHANGED\n" + "word here\n" * 15_999, encoding="utf-8",
    )
    for name in ordinary:
        (tmp_path / name).write_text("z = 0\n" + "y = 1\n" * 13_333, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_kw: 90_000)
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    # The pack assembles: the deficit was always coverable by optional content.
    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The required artifact keeps its full snapshot in the fixed part…
    assert "### prompts/large_prompt.md" in prompt
    # …and the disclosed degradation names the ordinary files, and only those.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    for name in ordinary:
        assert f"- {name}" in note, note
    assert "prompts/large_prompt.md" not in note, note
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/large_prompt.md"]["disposition"] == "already_included"


def test_cold_start_sizes_down_and_passes_instead_of_400ing(tmp_path, monkeypatch):
    """RS4 anti-regression: with NO observation for an unknown model the cold-start
    density must make the cap SMALLER than the historical optimistic one (pack passes),
    never larger (pack draws a deterministic provider 400)."""
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density
    from ouroboros.tools import scope_review as sr

    _DENSITY_MEMO.clear()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(1_000_000, "confirmed"))

    cold = sr._effective_scope_input_limit(scope_model="unknown/brand-new-model")
    assert 0 < cold <= sr._SCOPE_INPUT_TOKEN_LIMIT, (
        "a cold start must never be LOOSER than the historical absolute-margin cap"
    )
    # A pack sized at the cold cap still fits the reviewer's real window even at the
    # conservative density, so the first call is not a guaranteed 400.
    from ouroboros.capability_evidence import COLD_START_TOKEN_DENSITY
    assert int(cold * COLD_START_TOKEN_DENSITY) + sr._SCOPE_MAX_TOKENS <= 1_000_000

    # A later genuine measurement is what changes the number — and it is disclosed as
    # `measured` provenance rather than silently replacing an assumption.
    record_token_density(
        tmp_path, "unknown/brand-new-model", prompt_chars=4_000_000, prompt_tokens=1_100_000,
    )
    from ouroboros.capability_evidence import resolve_token_density
    assert resolve_token_density(tmp_path, "unknown/brand-new-model")[1] == "measured"


# --- scope-slot identity: one owner, one id per configured row ----------------


def _run_scope_fanout(monkeypatch, tmp_path, models):
    """Run the parallel scope fan-out over ``models`` and collect every id surface.

    Returns (substrate_ids, actor_record_ids, manifest_ids): the ids the review
    substrate physically ran the rows under (sorted — the rows run concurrently,
    so completion order is not meaningful), the ids stamped on the durable actor
    records, and the ids in the scope context manifest.
    """
    from types import SimpleNamespace

    from ouroboros import config, review_substrate
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools import scope_review as sr

    rows = [
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Concrete scope artifact was checked and passes.",
        }
        for item in sorted(sr._SCOPE_REQUIRED_ITEMS)
    ]
    substrate_ids: list = []
    lock = threading.Lock()

    def fake_run_review_request(request, *, slots, drive_root, llm, usage_ctx=None):
        with lock:
            substrate_ids.extend(slot.slot_id for slot in slots)
        return SimpleNamespace(actors=[{
            "slot_id": slots[0].slot_id,
            "model": slots[0].model,
            "status": "ok",
            "raw_text": json.dumps(rows),
            "usage": {},
            "prompt_ref": {},
            "response_ref": {},
        }])

    monkeypatch.setattr(config, "get_scope_review_models", lambda: list(models))
    monkeypatch.setattr(review_substrate, "run_review_request", fake_run_review_request)
    monkeypatch.setattr(sr, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _model, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(
        review, "_prepare_unified_review", lambda *_a, **_k: (None, None, True)
    )

    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-slot-identity",
        pending_events=[], _review_history=[], _review_advisory=[], _scope_review_history={},
    )
    parallel_review.run_parallel_review(ctx, "identity commit")
    actor_ids = [str(r.get("slot_id") or "") for r in (ctx._last_scope_raw_results or [])]
    manifest = (ctx._last_scope_raw_result or {}).get("context_manifest") or {}
    manifest_ids = [str(a.get("slot_id") or "") for a in (manifest.get("actors") or [])]
    return sorted(substrate_ids), actor_ids, manifest_ids


def test_scope_rows_sharing_a_model_keep_distinct_identities(tmp_path, monkeypatch):
    """Duplicate model ids are valid independent slots (review_substrate contract,
    and get_scope_review_models preserves them on purpose). Naming a row after its
    model collapsed both rows onto one receipt id."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/a"]
    )
    assert len(set(substrate_ids)) == 2, substrate_ids
    assert len(set(actor_ids)) == 2, actor_ids
    assert len(set(manifest_ids)) == 2, manifest_ids


def test_scope_rows_whose_models_sanitize_alike_keep_distinct_identities(tmp_path, monkeypatch):
    """Two DIFFERENT models can normalize to the same token (``openai::gpt-5`` and
    ``openai/gpt/5`` both sanitize to ``openai_gpt_5``), which merged two rows."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["openai::gpt-5", "openai/gpt/5"]
    )
    assert len(set(substrate_ids)) == 2, substrate_ids
    assert len(set(actor_ids)) == 2, actor_ids
    assert len(set(manifest_ids)) == 2, manifest_ids


def test_scope_row_identity_survives_editing_that_row_model(tmp_path, monkeypatch):
    """Editing a slot's model in the settings UI must not re-identify the slot:
    its receipts have to keep lining up with its own history."""
    before_substrate, before_actors, _ = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    after_substrate, after_actors, _ = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/EDITED"]
    )
    assert before_substrate == after_substrate, (before_substrate, after_substrate)
    assert before_actors == after_actors, (before_actors, after_actors)


def test_scope_actor_records_and_substrate_agree_on_one_identity(tmp_path, monkeypatch):
    """The durable actor record, the context manifest, and the substrate call that
    produced the prompt/response refs must name the SAME row. They were derived
    independently — positionally in the coordinator, from the model in the reviewer —
    so one row carried two disagreeing identities."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    assert sorted(substrate_ids) == sorted(actor_ids) == sorted(manifest_ids), (
        substrate_ids, actor_ids, manifest_ids
    )
    # Pinned spelling: durable records written before v6.87.21 already carry these
    # ids, so historical receipts line up with new ones without a translation table.
    assert actor_ids == ["scope_slot_1", "scope_slot_2"], actor_ids


def test_scope_row_ids_come_from_the_one_mint(tmp_path, monkeypatch):
    """The coordinator must READ the row's id, not re-derive an identical string.

    parallel_review stamped ``scope_slot_{idx + 1}`` on the actor record and the
    manifest — byte-identical to the mint's output today, so nothing could tell
    the two apart. Repointing the ONE mint separates them: a surface that reads it
    follows, a surface that spells its own literal does not.
    """
    from ouroboros import review_substrate

    monkeypatch.setattr(
        review_substrate, "slot_id_for_row",
        lambda index, *, prefix=review_substrate.SLOT_ID_PREFIX: f"{prefix}_row{int(index)}",
    )
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    expected = ["scope_slot_row1", "scope_slot_row2"]
    assert substrate_ids == expected, substrate_ids
    assert actor_ids == expected, actor_ids
    assert manifest_ids == expected, manifest_ids

# --- Blocking scope authority is a property of the EVIDENCE (v6.87.44) ----------

def _seed_scope_evidence(monkeypatch, tmp_path, model, *, window, status, ts, use_ack=False):
    """Write one Capability-Evidence record for ``model``'s real scope route."""
    import json as _json
    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import reviewer_route

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    provider, base_url = reviewer_route(model)
    fp = ce.route_fingerprint(provider=provider, base_url=base_url, model=model)
    store = tmp_path / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    key = "owner_acks" if use_ack else "probes"
    store.write_text(_json.dumps({key: {fp: {
        "window_tokens": window, "status": status, "source": "provider_metadata",
        "route_fp": fp, "model": model, "provider": provider, "ts": ts,
    }}}), encoding="utf-8")
    return fp


def test_stale_evidence_cannot_authorize_a_blocking_scope_verdict(monkeypatch, tmp_path):
    """BIBLE P3: blocking authority turns on SOURCED Capability Evidence, and an
    EXPIRED record that the probe could not re-verify is a dated impression, not a
    source. Before the typed result, `(window, status)` dropped `stale` on the floor,
    so a five-day-old 1M record kept across a provider outage read as `confirmed 1M`
    and signed the blocking verdict."""
    import datetime

    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window
    from ouroboros.tools import scope_review as sr

    model = "anthropic/claude-fable-5"
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_000_000,
                         status="confirmed", ts=old)
    # The provider is unreachable now, so `probe` keeps the prior record — as STALE.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)

    resolved = resolve_reviewer_window(model)
    assert resolved.window_tokens == 1_000_000 and resolved.status == "confirmed"
    assert resolved.stale is True, "the outage-carried record must arrive marked stale"
    assert resolved.observed_at == old, "the observation time must survive the hand-off"
    assert resolved.blocking_authority_allowed is False

    # ...and the scope gate acts on it: criticals are preserved but demoted.
    critical = [{"item": "architecture_fit", "verdict": "FAIL",
                 "severity": "critical", "reason": "r"}]
    crit_out, adv_out, result = sr._apply_scope_authority(
        critical, [], scope_model_id=model, result_kwargs={},
    )
    assert crit_out == [] and result is not None and result.blocked is True
    assert result.status == "sub_floor"
    # The owner is told the window EXPIRED, not that it was "confirmed" — and WHEN it
    # was last confirmed, which is the difference between a blip and a dead route.
    assert "EXPIRED" in result.block_message
    assert f"last confirmed {old}" in result.block_message
    assert any("EXPIRED" in str(f.get("reason", "")) for f in adv_out)

    # A CURRENT record for the same route authorises normally — the fix rejects
    # staleness, not the route.
    fresh = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_000_000,
                         status="confirmed", ts=fresh)
    assert resolve_reviewer_window(model).blocking_authority_allowed is True


def test_designated_default_gets_no_authority_from_its_name(monkeypatch, tmp_path):
    """A designated model does not acquire blocking authority from being designated.

    The sentinel still SIZES an unevidenced default at 1M (so the review is dispatched
    rather than declined before it starts), but sizing is not signing: with no sourced
    evidence the scope verdict is advisory, exactly as for any other unevidenced route.
    The same name-check used to disable the ONE lazy probe that could source the
    default's window, which is why it could never stop being invented."""
    from types import SimpleNamespace

    from ouroboros import capability_evidence as ce
    from ouroboros.tools import scope_review as sr

    fetches = []

    def fake_probe(_drive_root, **kw):
        fetches.append(bool(kw.get("allow_fetch")))
        return SimpleNamespace(window_tokens=0, status="unprobeable", source="none",
                               route_fp="fp", stale=False, ts="")

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)

    resolved = sr._scope_window(sr._SCOPE_MODEL_DEFAULT)
    assert resolved.window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW  # sizing survives
    assert sr._scope_window_provenance(resolved) == sr._WINDOW_SENTINEL
    assert resolved.blocking_authority_allowed is False
    assert fetches == [True], "the default route must get the lazy probe like any other"

    critical = [{"item": "architecture_fit", "verdict": "FAIL",
                 "severity": "critical", "reason": "r"}]
    crit_out, _adv, result = sr._apply_scope_authority(
        critical, [], scope_model_id=sr._SCOPE_MODEL_DEFAULT, result_kwargs={},
    )
    assert crit_out == [] and result is not None and result.blocked is True

    # Owner-acking that exact route is what restores authority — evidence, not name.
    ce.record_owner_ack(tmp_path, provider="openrouter", model=sr._SCOPE_MODEL_DEFAULT,
                        window_tokens=1_050_000, note="test")
    monkeypatch.undo()
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    assert sr._scope_window(sr._SCOPE_MODEL_DEFAULT).blocking_authority_allowed is True


def test_concurrent_resolution_of_one_route_shares_one_probe(monkeypatch, tmp_path):
    """parallel_review runs the triad and the scope slots concurrently. Without the
    per-route lock two slots on the SAME route both reach the provider for a window
    the first one is already fetching; with it the second enters after the evidence
    has been stored and reads it back, so one route costs one metadata fetch."""
    import threading
    from types import SimpleNamespace

    from ouroboros import capability_evidence as ce
    from ouroboros.tools import scope_review as sr

    model = "anthropic/claude-fable-5"
    in_probe, release = threading.Event(), threading.Event()
    store: dict = {}   # stands in for capability_evidence.json, which the real probe writes
    fetches: list = []

    def fake_probe(_drive_root, **kw):
        # `probe` serves a CURRENT record straight from its cache without touching the
        # network whatever `allow_fetch` says; only an absent/expired one goes out.
        if "ev" in store:
            return store["ev"]
        if not kw.get("allow_fetch"):
            return SimpleNamespace(window_tokens=0, status="unprobeable", stale=False, ts="")
        fetches.append(str(kw.get("model") or ""))
        in_probe.set()
        release.wait(10)              # the network probe is still in flight
        store["ev"] = SimpleNamespace(
            window_tokens=1_000_000, status="confirmed", stale=False,
            ts="2026-08-02T00:00:00+00:00")
        return store["ev"]

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)
    monkeypatch.setattr("ouroboros.reviewer_window._LAZY_ROUTE_LOCKS", {})

    out = {}
    threads = [
        threading.Thread(target=lambda k=k: out.__setitem__(k, sr._scope_window(model)))
        for k in ("a", "b")
    ]
    threads[0].start()
    assert in_probe.wait(10), "the first thread never reached the probe"
    threads[1].start()
    threads[1].join(0.5)
    assert threads[1].is_alive(), (
        "the second thread must WAIT for the in-flight probe on its route"
    )
    release.set()
    for thread in threads:
        thread.join(10)

    assert fetches == [model], (
        f"one route must cost ONE metadata fetch; got {len(fetches)}"
    )
    assert out["a"].window_tokens == out["b"].window_tokens == 1_000_000
    assert out["a"].blocking_authority_allowed is out["b"].blocking_authority_allowed is True


def test_expired_evidence_is_re_sourced_instead_of_wedging_the_process(monkeypatch, tmp_path):
    """A long-lived process must be able to RE-confirm its scope reviewer.

    The lazy probe used to be memoised for the lifetime of the process while the
    evidence it produced expired after 24h, so a healthy, connected install that
    stayed up past the TTL read its own reviewer as EXPIRED on every later
    resolution: `blocking_authority_allowed` went False and stayed False, and
    `_apply_scope_authority` blocked EVERY commit for the rest of the process's
    life. How often a route may be re-probed is `capability_evidence.probe`'s TTL to
    decide — a second, never-expiring rate limit here could only ever wedge."""
    import datetime

    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window
    from ouroboros.tools import scope_review as sr

    model = "openai/gpt-5.6-terra"
    now = datetime.datetime.now(datetime.timezone.utc)
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_050_000,
                         status="confirmed", ts=now.isoformat())
    # The provider is up the whole time: a metadata read returns the real window.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 1_050_000)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: False)

    assert resolve_reviewer_window(model).blocking_authority_allowed is True

    # ...25 hours later, in the SAME process: the one stored record has aged past the
    # 24h confirmed TTL. Nothing about the install changed.
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_050_000, status="confirmed",
                         ts=(now - datetime.timedelta(hours=25)).isoformat())

    resolved = resolve_reviewer_window(model)
    assert resolved.stale is False, "an expired record must be RE-SOURCED, not read as expired"
    assert resolved.blocking_authority_allowed is True
    _crit, _adv, result = sr._apply_scope_authority(
        [{"item": "architecture_fit", "verdict": "FAIL",
          "severity": "critical", "reason": "r"}],
        [], scope_model_id=model, result_kwargs={},
    )
    assert result is None, "a healthy install must not block its own commits after 24h"


class TestTriadPackExclusions:
    """The triad pack's disclosed exclusion classes (review economics, D-06a).

    The builder takes the advisory seam's ``exclude_paths`` shape and marks an
    excluded path ONCE; ``triad_pack_exclusions`` names exactly two classes the
    host can back — span-only release carriers on a VERSION-staged commit
    (``release_sync`` carrier SSOT) and governance docs byte-identical to the
    inlined prefix copy — and returns the disclosure note the caller appends."""

    def test_exclude_paths_withhold_the_text_with_one_marker(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_helpers")
        # Oversize AND excluded: the exclusion marker wins, never two markers.
        (tmp_path / "uv.lock").write_bytes(b"x" * (1_048_576 + 1))
        (tmp_path / "a.py").write_text("print('kept')", encoding="utf-8")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["uv.lock", "a.py"], exclude_paths={"uv.lock"})
        assert omitted == ["uv.lock"]
        assert pack.count("### uv.lock") == 1
        assert "withheld by the caller's exclusion note" in pack
        assert "byte limit" not in pack and "xxxx" not in pack
        assert "print('kept')" in pack
        # The default is byte-identical to the pre-exclusion builder.
        pack_default, omitted_default = mod.build_touched_file_pack(tmp_path, ["a.py"])
        assert omitted_default == [] and "print('kept')" in pack_default

    @staticmethod
    def _carrier_repo(tmp_path, *, with_lock=True):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
        (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8")
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8")
        if with_lock:
            (repo / "uv.lock").write_text(_uv_lock_text("1.0.0"), encoding="utf-8")
        (repo / "docs").mkdir()
        (repo / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v1.0.0 — Architecture\n\nArchitecture body.\n", encoding="utf-8")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body.\n", encoding="utf-8")
        (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
        return repo

    @staticmethod
    def _staged_paths(repo):
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(repo),
                             check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if line]

    def test_span_only_carriers_and_prefix_duplicates_are_cut_on_a_version_bump(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path)
        (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8")
        # pyproject: version bump PLUS a dependency edit outside its span.
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "ouroboros"\nversion = "1.0.1"\ndependencies = ["httpx"]\n',
            encoding="utf-8")
        (repo / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v1.0.1 — Architecture\n\nArchitecture body.\n", encoding="utf-8")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body, revised.\n", encoding="utf-8")
        (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        paths = self._staged_paths(repo)
        dev_text = (repo / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

        excluded, note = mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": dev_text, "docs/DESIGN.md": ""})

        assert excluded == {"VERSION", "uv.lock", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md"}
        assert "pyproject.toml" not in excluded and "app.py" not in excluded
        assert note.startswith("⚠️ PACK EXCLUSION NOTE: full text withheld for 4 touched file(s)")
        assert "VERSION_CARRIER_SPANS" in note and "version_carrier_desyncs" in note
        assert "uv.lock" in note and "byte-identical" in note and "docs/DEVELOPMENT.md" in note
        # The pack renders the cut through the builder's own marker + omitted list.
        pack, omitted = mod.build_touched_file_pack(repo, paths, exclude_paths=excluded)
        assert set(omitted) == excluded
        assert "httpx" in pack and "x = 2" in pack  # kept texts
        assert "Handbook body, revised." not in pack and "editable" not in pack  # withheld texts

    def test_without_version_staged_carriers_keep_their_text(self, tmp_path):
        """The carrier class is a release-bump mechanism: no VERSION staged, no
        carrier cut (the preflight carrier gate did not run); the prefix-dedup
        class is independent of it."""
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path)
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body, revised.\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        paths = self._staged_paths(repo)
        dev_text = (repo / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

        excluded, note = mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": dev_text})
        assert excluded == {"docs/DEVELOPMENT.md"}
        assert "release carrier" not in note and "byte-identical" in note
        # A prefix copy with DIFFERENT bytes (or none) keeps the doc's full text.
        assert mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": "# DEV\n\nOther bytes.\n"}) == (set(), "")
        assert mod.triad_pack_exclusions(repo, paths, prefix_texts={}) == (set(), "")

    def test_a_carrier_new_at_head_keeps_its_text(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path, with_lock=False)
        (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        excluded, _note = mod.triad_pack_exclusions(
            repo, self._staged_paths(repo), prefix_texts={})
        assert excluded == {"VERSION"}


def _uv_lock_text(version):
    return (
        'version = 1\n\n[[package]]\nname = "ouroboros"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "httpx"\nversion = "0.27.0"\n'
    )


def test_scope_pack_applies_the_carrier_cut_over_its_own_staged_pair(tmp_path, monkeypatch):
    """Owner decision (F3 Q4 = A): the scope pack cuts span-only release carriers
    over the HEAD→index pair it reviews — no snapshot, named in the dedup note, a
    typed `already_included` row with the by-design reason in the durable
    manifest, the ladder's first entry — while the canonical docs (the governance
    prefix) and a carrier edited outside its span keep every byte. A managed
    subject and an artifact the atlas owes in full are never cut."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as pack

    def _lock(v):  # an unchanged tail marker far outside every -U3 hunk
        return _uv_lock_text(v) + "".join(
            f'\n[[package]]\nname = "filler{i}"\nversion = "1.0.0"\n' for i in range(6)
        ) + '\n[[package]]\nname = "UNCHANGED_LOCK_TAIL_MARKER"\nversion = "9.9.9"\n'

    repo = TestTriadPackExclusions._carrier_repo(tmp_path)
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
    (repo / "uv.lock").write_text(_lock("1.0.0"), encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "lock"], cwd=str(repo), check=True)
    (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
    (repo / "uv.lock").write_text(_lock("1.0.1"), encoding="utf-8")
    (repo / "pyproject.toml").write_text(  # version bump PLUS an edit outside its span
        '[project]\nname = "ouroboros"\nversion = "1.0.1"\ndependencies = ["httpx"]\n', encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros v1.0.1 — Architecture\n\nArchitecture body.\n", encoding="utf-8")
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)

    prompt, status = sr._build_scope_prompt(repo, "release: 1.0.1")

    assert status is None and prompt
    dedup = prompt.split("## CURRENT FILE CONTEXT DEDUPLICATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- uv.lock" in dedup and "- VERSION" in dedup and "VERSION_CARRIER_SPANS" in dedup
    assert "- docs/ARCHITECTURE.md" in dedup and "pyproject.toml" not in dedup
    assert "UNCHANGED_LOCK_TAIL_MARKER" not in prompt  # no snapshot anywhere in the pack
    assert 'version = "1.0.1"' in prompt  # …while the staged diff carries the change
    assert "httpx" in prompt and "Architecture body." in prompt  # kept: outside-span carrier; prefix copy
    manifest = sr._current_scope_context_manifest()
    rows = {r["path"]: r for r in manifest["coverage"]}
    assert rows["uv.lock"]["disposition"] == "already_included"
    assert "omitted by design" in rows["uv.lock"]["reason"] and "VERSION_CARRIER_SPANS" in rows["uv.lock"]["reason"]
    assert rows["pyproject.toml"]["reason"] == "included in fixed prompt context"
    steps = manifest["ladder_steps"]
    assert steps[0]["step"] == "carrier_span_only_omitted" and sorted(steps[0]["paths"]) == ["VERSION", "uv.lock"]
    assert steps[1]["step"] == "compact_atlas" and "TOUCHED FILE BUDGET DEGRADATION NOTE" not in prompt
    # The seam's two refusals: a managed subject, and an artifact owed in full.
    assert pack._carrier_span_only_paths(repo, ["VERSION", "uv.lock", "app.py"], object()) == []
    monkeypatch.setattr(sr, "atlas_required_beyond_diff", lambda rel: rel == "uv.lock")
    assert pack._carrier_span_only_paths(repo, ["VERSION", "uv.lock", "app.py"], None) == ["VERSION"]
