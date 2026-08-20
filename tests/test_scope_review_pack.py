"""What goes into the scope-review pack and prompt.

Split by theme out of the original ``tests/test_scope_review.py`` giant. This
module owns the pack assembly: the checklist section loader, the goal/scope
precedence, the touched-file and broader-repo packs, the HEAD snapshot section,
the scope prompt matrix contract and the triad prompt anti-pattern lock.
"""

import inspect
import subprocess

import pytest

from tests._scope_review_shared import _get_module

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
