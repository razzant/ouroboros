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
    _get_registry_module,
    _make_ctx,
)
from tests.test_commit_index import _advisory_review, _baseline, _committed, _continue, _git, task  # noqa: F401 - fixture


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


@pytest.mark.parametrize("resumed", [False, True])
def test_an_unreadable_staged_binding_blocks_before_any_reviewer(git_ctx, monkeypatch, resumed):
    """The staged subject's binding cannot be read: a resumed wave refuses with its own fact and records nothing, any
    other call records the revalidation failure; both return the one blocked shape and buy no reviewer."""
    from ouroboros.tools import git_review_cycle

    git_mod, ctx = git_ctx
    (pathlib.Path(ctx.repo_dir) / "change.txt").write_text("product change\n", encoding="utf-8")
    unreadable, recorded = {"ok": False, "fingerprint": "", "status": "unavailable", "reason": "probe"}, []
    monkeypatch.setattr(git_mod, "_fingerprint_staged_diff", lambda repo: unreadable)
    monkeypatch.setattr(git_mod, "_handle_revalidation_failure", lambda *a, **kw: recorded.append(kw) or "RECORDED_FAILURE")
    monkeypatch.setattr(git_mod, "_run_parallel_review", lambda *a, **kw: pytest.fail("an unbound subject buys no reviewer"))
    ctx._review_resume_pending = resumed

    outcome = git_review_cycle._reviewed_stage_cycle(ctx, "candidate", 0.0, paths=["change.txt"])

    message = outcome.pop("message")
    assert outcome == {"status": "blocked", "block_reason": "fingerprint_unavailable", "pre_fingerprint": unreadable,
                       "post_fingerprint": {}}
    if resumed:
        assert message.startswith("⚠️ REVIEW_BINDING_UNAVAILABLE") and recorded == [], message
    else:
        assert message == "RECORDED_FAILURE" and recorded == [{"pre_fingerprint": unreadable, "kind": "fingerprint_unavailable"}]


# --- #2680: an author continuation stages its reviewed subject's exact entries, by Git's own evidence ----------------

def _nested_repository(path):
    for args in (("init", "-q"), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid"),
                 ("add", "-A"), ("commit", "-qm", "nested")):
        _git(path, *args)


@pytest.mark.parametrize("shape", ["marker_file", "empty_gitdir", "dangling_gitfile"])
def test_a_reviewed_file_replaced_by_a_directory_lands_by_gits_evidence_never_its_marker(task, monkeypatch, shape):  # noqa: F811
    """Independent review: the reviewed file own.txt is replaced by a directory of local QA holding a ``.git`` entry.
    Its shape decides nothing; Git's evidence does. A marker file, an empty directory or a gitfile that resolves
    nowhere is no repository, so omitted paths land the reviewed file's deletion and nothing the directory holds. A
    real nested repository is the one entry Git stages there (a gitlink), never its files."""
    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("reviewed change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    (repo / "own.txt").unlink()
    (repo / "own.txt").mkdir()
    (repo / "own.txt" / "local-qa.txt").write_text("unreviewed local evidence\n", encoding="utf-8")
    marker = repo / "own.txt" / ".git"
    {"marker_file": lambda: marker.write_text("", encoding="utf-8"), "empty_gitdir": marker.mkdir,
     "dangling_gitfile": lambda: marker.write_text("gitdir: ../nowhere\n", encoding="utf-8"),
     "nested_repository": lambda: _nested_repository(repo / "own.txt")}[shape]()

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert _committed(repo) == {"own.txt": "T" if shape == "nested_repository" else "D"}, result
    assert (repo / "own.txt" / "local-qa.txt").is_file() and "local-qa" not in _git(repo, "ls-files"), result


def test_a_reviewed_gitlink_whose_checkout_lost_its_repository_lands_its_pointer(task, monkeypatch):  # noqa: F811
    """Entry evidence: the reviewed submodule pointer is staged again as reviewed, and its checkout then loses its
    repository. The index's gitlink is still the one entry Git stages there, so omitted paths land that pointer,
    never its deletion, and nothing the plain directory holds."""
    from ouroboros.force_remove import force_rmtree

    repo, sub = task.repo_dir, task.repo_dir / "sub"
    sub.mkdir()
    (sub / "f.txt").write_text("one\n", encoding="utf-8")
    _nested_repository(sub)
    _git(repo, "add", "sub")
    _git(repo, "commit", "-qm", "track sub")
    _baseline(task)
    (sub / "f.txt").write_text("two\n", encoding="utf-8")
    _git(sub, "commit", "-qam", "two")
    reference = _advisory_review(task, monkeypatch, ["sub"])
    _git(repo, "add", "sub")
    pointer = _git(repo, "ls-files", "-s", "sub").split()[1]
    force_rmtree(sub / ".git")
    assert not (sub / ".git").exists()

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert _committed(repo) == {"sub": "M"}, result
    assert _git(repo, "ls-tree", "HEAD", "sub").split()[:3] == ["160000", "commit", pointer], result


def test_unreadable_entry_evidence_refuses_the_continuation_before_staging(task, monkeypatch):  # noqa: F811
    """Git's entry evidence for a reviewed name that now holds a directory cannot be read at staging: the call
    refuses with the staging error before anything is staged, never guessing a removal or a recursive add."""
    from ouroboros.tools import review_collection

    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("reviewed change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    (repo / "own.txt").unlink()
    (repo / "own.txt").mkdir()
    (repo / "own.txt" / "local-qa.txt").write_text("unreviewed local evidence\n", encoding="utf-8")

    def unreadable(_repo, names):
        if names:
            raise RuntimeError("index unreadable")
        return []
    monkeypatch.setattr(review_collection, "_directories_without_entry", unreadable)
    head = _git(repo, "rev-parse", "HEAD")

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert "GIT_ERROR (staged-status)" in result and "index unreadable" in result, result
    assert _git(repo, "rev-parse", "HEAD") == head and _git(repo, "diff", "--cached", "--name-only") == "", result


@pytest.mark.parametrize("baseline", [True, False])
def test_a_reviewed_file_over_a_tracked_directory_now_a_nested_repository_stages_nothing_below(task, monkeypatch,  # noqa: F811
                                                                                               baseline):
    """The review turned the tracked directory pkg into a file. pkg is a directory again, its tracked file restored
    beside local QA, inside a repository. The index still holds pkg as a directory, so Git would walk into it: the
    reviewed name is no entry there and nothing reviewed differs from HEAD, with or without task attribution."""
    import shutil

    repo = task.repo_dir
    (repo / "pkg").mkdir()
    (repo / "pkg" / "x.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "pkg")
    _git(repo, "commit", "-qm", "track pkg")
    if baseline:
        _baseline(task)
    shutil.rmtree(repo / "pkg")
    (repo / "pkg").write_text("reviewed file\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["pkg", "pkg/x.txt"])
    (repo / "pkg").unlink()
    (repo / "pkg").mkdir()
    (repo / "pkg" / "x.txt").write_text("tracked\n", encoding="utf-8")
    (repo / "pkg" / "qa.txt").write_text("unreviewed local evidence\n", encoding="utf-8")
    _nested_repository(repo / "pkg")
    head = _git(repo, "rev-parse", "HEAD")

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert "GIT_NO_CHANGES" in result and "pkg, pkg/x.txt" in result, result
    assert _git(repo, "rev-parse", "HEAD") == head and "qa.txt" not in _git(repo, "ls-files"), result


@pytest.mark.parametrize("operation", ["status", "resolve", "settle"])
def test_a_blocked_review_does_not_offer_author_continuation_or_rewrite_on_read(task, monkeypatch, operation):  # noqa: F811
    """A rejection a Blocking install recorded before Blocking left is not an author handback. Reading its status or
    refusing its reference must not offer an unusable continuation or rewrite its closed seats just to tag them; the
    same rejection recorded today is a handback whose status offers the continuation."""
    import dataclasses
    import json

    from ouroboros import review_state
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.tools.commit_gate import resolve_commit_review_reference
    from ouroboros.tools.preflight_review import _handle_review_status
    from ouroboros.tools.review_collection import reference_for
    from ouroboros.tools.review_change_custody import repair_settled_attempt
    from tests.test_commit_index import _rejected_review
    from tests.test_unwaited_review_settlement import _state_writes

    _baseline(task)
    (task.repo_dir / "own.txt").write_text("change\n", encoding="utf-8")
    _rejected_review(task, monkeypatch)
    state = review_state.load_state(task.drive_root)
    assert (state.attempts[-1].status, state.attempts[-1].phase) == ("reviewed", "review_only")
    assert "review_reference" in json.loads(_handle_review_status(task, task_id=task.task_id))
    # The row a Blocking install wrote for the same rejection: its refusal, never a handback.
    state.attempts[-1] = dataclasses.replace(state.attempts[-1], status="blocked", phase="blocking_review")
    review_state.save_state(task.drive_root, state)
    before = review_state.load_state(task.drive_root).attempts[-1]
    assert (before.status, before.phase) == ("blocked", "blocking_review")
    record = load_record(ledger_root(task), before.review_record_id)
    reference = reference_for(record, before)
    writes = _state_writes(monkeypatch)

    if operation == "status":
        assert "review_reference" not in json.loads(_handle_review_status(task, task_id=task.task_id))
    elif operation == "resolve":
        with pytest.raises(ValueError, match="missing or no longer"):
            resolve_commit_review_reference(task, reference)
    else:
        repair_settled_attempt(task.drive_root, ledger_root(task), before)
    assert writes == [] and review_state.load_state(task.drive_root).attempts[-1] == before


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


# --- Unified review wired into commit functions ---

class TestReviewInCommitPipeline:
    # ``test_repo_commit_calls_unified_review`` was removed in
    # v5.8.3-rc.5 — it is a strict subset of
    # ``tests/test_review_helpers_pack.py::TestGitWiring::test_one_wave_wired_in_commit``
    # which additionally verifies the one wave (assembly → admission → dispatch)
    # is reached and the ``ThreadPoolExecutor`` parallelism contract holds.

    def test_blocked_review_unstages(self, git_ctx, monkeypatch):
        """A refused cycle unstages what it staged (real Git; a protected path under Advanced)."""
        git_mod, ctx = git_ctx
        repo = pathlib.Path(ctx.repo_dir)
        (repo / "BIBLE.md").write_text("protected change\n", encoding="utf-8")
        monkeypatch.setattr(git_mod, "_current_runtime_mode", lambda: "advanced")
        git_mod._reset_commit_review_state(ctx)

        result = git_mod._run_reviewed_stage_cycle(ctx, "protected change", 0.0, paths=["BIBLE.md"])

        assert result["block_reason"] == "core_protection_blocked", result
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo,
                                check=True, capture_output=True, text=True).stdout
        assert staged == ""

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


    def test_managed_tests_gate_before_tag_and_ordinary_tests_keep_prior_order(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        # The managed BLOCKING gate (extracted helper) runs before tagging;
        # ordinary warning-only tests run after the tag and before the push.
        managed_tests_pos = source.index("_managed_post_commit_tests_gate(")
        tag_pos = source.index("tag_info =")
        ordinary_tests_pos = source.index("run_operation(ctx, _post_commit_result, ctx, commit_message")
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


# --- credential helper safety (legacy migration retired) ---

class TestRemoteCredentialConfiguration:
    def test_legacy_migrator_retired(self):
        git_ops = _get_git_ops_module()
        assert not hasattr(git_ops, "migrate_remote_credentials")


    def test_startup_setup_only_configures_current_settings_token(self):
        server_runtime = importlib.import_module("ouroboros.server_runtime")
        source = inspect.getsource(server_runtime.setup_remote_if_configured)
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
        source = inspect.getsource(registry.ToolRegistry._execute_admitted_text)
        assert "_ROOT_ARG_REPO_WRITE_TOOLS" in source
        assert "write_file" in registry._ROOT_ARG_REPO_WRITE_TOOLS

    def test_sandbox_checks_files_param(self):
        """Sandbox must check files array for safety-critical paths."""
        registry = _get_registry_module()
        assert registry._payload_write_paths(
            "write_file",
            {"files": [{"path": "BIBLE.md", "content": "x"}]},
        ) == ["BIBLE.md"]


# ---------------------------------------------------------------------------
# Check 7: P9 history limits in _preflight_check (v4.41.0)
# ---------------------------------------------------------------------------
# Advisory skip_tests parameter (v4.41.0)
# ---------------------------------------------------------------------------
