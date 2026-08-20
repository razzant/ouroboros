"""Behavioral tests for the git+review commit pipeline.

Renamed in v5.15.x from ``test_phase7_pipeline.py`` — the file is the
canonical behavioral suite for the modern commit pipeline + operational
resilience, not a one-shot migration test. The previous name pinned a
historical migration phase that has long since shipped.

The preflight gate, verdict enforcement, advisory ``skip_tests`` and bypass
halves were split verbatim into
``tests/test_git_review_preflight_gate.py``,
``tests/test_git_review_enforcement.py``,
``tests/test_git_review_advisory_skip_tests.py`` and
``tests/test_git_review_bypass_gate.py``.

Tests:
- repo_write single-file and multi-file modes
- repo_write + repo_commit workflow
- Unified review wired into the commit functions
- configure_remote failure surfacing
- configure_remote credential-helper wiring
- Auto-rescue only reports committed when commit actually happened
- repo_write in CORE_TOOL_NAMES
"""
import importlib
import inspect
import os
import pathlib
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from tests._git_review_pipeline_shared import (
    _get_git_module,
    _get_git_ops_module,
    _get_git_review_cycle_module,
    _get_registry_module,
    _make_ctx,
)


@pytest.fixture
def git_ctx(tmp_path):
    """Yield ``(git_module, ToolContext)`` — the canonical git pipeline setup."""
    return _get_git_module(), _make_ctx(tmp_path)


def test_managed_resolver_stages_tracked_binary_from_official_merge(tmp_path, monkeypatch):
    git_mod = _get_git_module()
    ctx = _make_ctx(tmp_path)
    binary = pathlib.Path(ctx.repo_dir) / "native.so"
    binary.write_bytes(b"old\x00payload")
    subprocess.run(["git", "add", "-f", "native.so"], cwd=ctx.repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "track binary"], cwd=ctx.repo_dir, check=True)
    binary.write_bytes(b"official\x00payload")
    monkeypatch.setattr(
        _get_git_review_cycle_module(), "_authorized_managed_update_resolver", lambda _ctx: True
    )

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
        from ouroboros.tool_capabilities import CORE_TOOL_NAMES

        assert "write_file" in CORE_TOOL_NAMES

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


# --- Unified review wired into commit functions ---

class TestReviewInCommitPipeline:
    # ``test_repo_commit_calls_unified_review`` was removed in
    # v5.8.3-rc.5 — it is a strict subset of
    # ``tests/test_scope_review_wiring.py::TestGitWiring::test_scope_review_wired_in_commit``
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
        from ouroboros.tools import tool_resolution

        source = inspect.getsource(registry.ToolRegistry._execute_legacy_text)
        assert "_ROOT_ARG_REPO_WRITE_TOOLS" in source
        assert "write_file" in tool_resolution._ROOT_ARG_REPO_WRITE_TOOLS

    def test_sandbox_checks_files_param(self):
        """Sandbox must check files array for safety-critical paths."""
        from ouroboros.tools import tool_resolution

        assert tool_resolution._payload_write_paths(
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
