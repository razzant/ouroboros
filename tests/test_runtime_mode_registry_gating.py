"""Runtime-mode gating of the tool registry: light blanket, advanced protected, pro notice.

Split verbatim out of ``tests/test_runtime_mode_core.py`` by theme. This module owns
the light-mode blanket block on repo mutation and the cognitive-memory redirects it
applies instead, the advanced-mode block on protected core, contract and release
surfaces, the pro-mode CORE_PATCH_NOTICE, and the commit/restore/revert gates that read
the same protected categories from staged paths.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros.runtime_mode_policy import protected_path_category
from ouroboros.tools.registry import ToolRegistry

from tests._runtime_mode_core_shared import _git_repo, _registry


# ===========================================================================
# Part 6: ToolRegistry runtime-mode gating
# ===========================================================================


class _CommitCtx:
    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self.repo_dir = repo_dir
        self.drive_root = drive_root
        self.task_id = "runtime-mode-test"
        self._review_advisory = []
        self._last_triad_models = []
        self._last_scope_model = ""
        self._last_triad_raw_results = []
        self._last_scope_raw_result = {}
        self._review_degraded_reasons = []
        self._current_review_tool_name = "commit_reviewed"
        self._scope_review_history = {}
        self._review_history = []

    def emit_progress_fn(self, *_args, **_kwargs):
        return None

    def drive_logs(self):
        path = pathlib.Path(self.drive_root) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


# ----- Light mode blanket block -----


@pytest.mark.parametrize(("tool_name", "args"), [
    ("write_file", {"path": "README.md", "content": "changed\n"}),
    ("commit_reviewed", {"commit_message": "test"}),
    ("edit_text", {"path": "README.md", "old_str": "ok", "new_str": "changed"}),
    ("vcs_revert", {"sha": "HEAD"}),
    ("vcs_pull_ff", {}),
    ("vcs_restore", {}),
    ("vcs_rollback", {"target": "HEAD"}),
    ("promote_to_stable", {"reason": "test"}),
])
def test_light_mode_blocks_repo_mutation_tools(tool_name, args, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute(tool_name, args)
    assert "LIGHT_MODE_BLOCKED" in result, result[:200]


def test_light_mode_still_allows_read_only_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("read_file", {"path": "README.md"})
    assert "LIGHT_MODE_BLOCKED" not in result


def test_light_mode_redirects_cognitive_memory_write(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {"root": "runtime_data", "path": "memory/identity.md", "content": "x" * 60},
    )
    assert "COGNITIVE_TOOL_REQUIRED" in result, result[:200]
    assert "update_identity" in result
    assert "LIGHT_MODE_BLOCKED" not in result


def test_light_mode_redirects_windows_style_cognitive_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {"root": "runtime_data", "path": "memory\\identity.md", "content": "x" * 60},
    )
    assert "COGNITIVE_TOOL_REQUIRED" in result, result[:200]


def test_light_mode_redirects_absolute_home_path_to_user_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    home_path = str(pathlib.Path.home() / "Desktop" / "ouro_root_required_test.html")
    result = reg.execute("write_file", {"path": home_path, "content": "<html></html>"})
    assert "ROOT_REQUIRED_USER_FILES" in result, result[:200]
    assert "user_files" in result


def test_light_mode_does_not_block_skill_exec_at_registry_layer(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("skill_exec", {})
    assert "LIGHT_MODE_BLOCKED" not in result
    assert "SKILL_EXEC_BLOCKED" not in result


# ----- Advanced mode: protected core/contract/release surfaces -----


@pytest.mark.parametrize("path", [
    "ouroboros/safety.py",
    "ouroboros/contracts/plugin_api.py",
    "ouroboros/runtime_mode_policy.py",
    ".github/workflows/ci.yml",
])
def test_advanced_mode_blocks_protected_write(path, tmp_path, monkeypatch):
    """One parametrized test replaces three near-identical
    test_advanced_mode_blocks_{safety_critical,frozen_contract,
    runtime_policy_guardrail,release_invariant}_write variants."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {"path": path, "content": "x"},
    )
    assert "CORE_PROTECTION_BLOCKED" in result


def test_dot_github_workflow_is_release_invariant():
    assert protected_path_category(".github/workflows/ci.yml") == "release-invariant"
    assert protected_path_category("./.github/workflows/ci.yml") == "release-invariant"


def test_advanced_mode_allows_non_critical_write_calls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {"path": "docs/README.md", "content": "x"},
    )
    assert "CORE_PROTECTION_BLOCKED" not in result
    assert "LIGHT_MODE_BLOCKED" not in result


# ----- Pro mode: protected edits allowed with CORE_PATCH_NOTICE -----


def test_pro_mode_allows_protected_write_with_core_patch_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    reg = _registry(tmp_path)
    result = reg.execute(
        "write_file",
        {"path": "ouroboros/safety.py", "content": "x"},
    )
    assert "CORE_PROTECTION_BLOCKED" not in result
    assert "CORE_PATCH_NOTICE" in result


def test_pro_mode_edit_text_emits_core_patch_notice(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    (repo / "ouroboros" / "contracts").mkdir(parents=True)
    (repo / "ouroboros" / "contracts" / "plugin_api.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "contracts"], cwd=repo, check=True, capture_output=True)

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path)

    result = reg.execute(
        "edit_text",
        {
            "path": "ouroboros/contracts/plugin_api.py",
            "old_str": "old",
            "new_str": "new",
        },
    )

    assert "Replaced" in result
    assert "CORE_PATCH_NOTICE" in result
    assert "ouroboros/contracts/plugin_api.py" in result


def test_advanced_commit_blocks_protected_staged_paths(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_mod

    repo = _git_repo(tmp_path)
    (repo / "BIBLE.md").write_text("changed\n", encoding="utf-8")
    ctx = _CommitCtx(repo, tmp_path / "drive")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")

    result = git_mod._run_reviewed_stage_cycle(
        ctx,
        "test protected commit",
        0.0,
        paths=["BIBLE.md"],
        skip_advisory_pre_review=True,
    )

    assert result["status"] == "blocked"
    assert result["block_reason"] == "core_protection_blocked"
    assert "CORE_PROTECTION_BLOCKED" in result["message"]


def test_advanced_commit_blocks_rename_from_protected_path(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_mod

    repo = _git_repo(tmp_path)
    subprocess.run(["git", "mv", "BIBLE.md", "BIBLE2.md"], cwd=repo, check=True)
    ctx = _CommitCtx(repo, tmp_path / "drive")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")

    result = git_mod._run_reviewed_stage_cycle(
        ctx,
        "rename protected file",
        0.0,
        skip_advisory_pre_review=True,
    )

    assert result["status"] == "blocked"
    assert result["block_reason"] == "core_protection_blocked"
    assert "BIBLE.md" in result["message"]


def test_pro_commit_uses_normal_review_for_protected_paths(tmp_path, monkeypatch):
    from ouroboros.tools import git_review_cycle as git_mod

    repo = _git_repo(tmp_path)
    (repo / "BIBLE.md").write_text("changed\n", encoding="utf-8")
    ctx = _CommitCtx(repo, tmp_path / "drive")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")

    calls = {"review": 0}

    def fake_review(*_args, **_kwargs):
        calls["review"] += 1
        return None, None, "", []

    monkeypatch.setattr(git_mod, "_run_parallel_review", fake_review)
    monkeypatch.setattr(git_mod, "_aggregate_review_verdict", lambda *a, **k: (False, None, "", [], []))

    result = git_mod._run_reviewed_stage_cycle(
        ctx,
        "test protected commit",
        0.0,
        paths=["BIBLE.md"],
        skip_advisory_pre_review=True,
    )

    assert result["status"] == "passed"
    assert calls == {"review": 1}


def test_restore_to_head_blocks_release_invariant_path(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_mod

    repo = _git_repo(tmp_path)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "ci"], cwd=repo, check=True, capture_output=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: changed\n", encoding="utf-8")

    ctx = _CommitCtx(repo, tmp_path / "drive")
    result = git_mod._restore_to_head(ctx, confirm=True, paths=[".github/workflows/ci.yml"])

    assert "RESTORE_BLOCKED" in result
    assert ".github/workflows/ci.yml" in result


def test_restore_to_head_blocks_protected_rename_source(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_mod

    repo = _git_repo(tmp_path)
    subprocess.run(["git", "mv", "BIBLE.md", "BIBLE2.md"], cwd=repo, check=True)

    ctx = _CommitCtx(repo, tmp_path / "drive")
    result = git_mod._restore_to_head(ctx, confirm=True)

    assert "RESTORE_BLOCKED" in result
    assert "BIBLE.md" in result


def test_revert_commit_blocks_protected_contract_path(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_mod

    repo = _git_repo(tmp_path)
    (repo / "ouroboros" / "contracts").mkdir(parents=True)
    (repo / "ouroboros" / "contracts" / "plugin_api.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "contract"], cwd=repo, check=True, capture_output=True)
    target_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    ctx = _CommitCtx(repo, tmp_path / "drive")
    result = git_mod._revert_commit(ctx, target_sha, confirm=True)

    assert "REVERT_BLOCKED" in result
    assert "ouroboros/contracts/plugin_api.py" in result
