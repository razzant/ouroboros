"""Owner Pause on the real managed sibling of commit_reviewed, then rejoin its tail."""
from types import SimpleNamespace

import pytest

from ouroboros import owner_pause
from ouroboros.task_results import write_task_result
from ouroboros.tools import git
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import (
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)
from supervisor import update_merge
from tests import test_update_merge_assisted as assisted


@pytest.fixture
def managed_commit(tmp_path, monkeypatch):
    repo, branch, plan, tx = assisted._materialized_conflict_tx(tmp_path, monkeypatch)
    (repo / "VERSION").write_text("1.0.0\n")
    assert assisted._git(repo, "add", "-A").returncode == 0
    drive = tmp_path / "data"
    write_task_result(drive, "resolver", "running")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, branch_dev=branch,
                      task_id="resolver", task_metadata=assisted._authority_metadata(tx))
    ctx.current_task_type = "task"
    ctx.task_lifecycle_bound = True
    effects, records = [], []
    native = git.run_cmd
    mode = {"pause": "commit", "red": ""}

    def command(argv, **kwargs):
        result = native(argv, **kwargs)
        if argv[1:2] == ["commit"]:
            effects.append("commit")
            if mode["pause"] == "commit":
                owner_pause.install_fence(drive, "resolver", request_id="after-commit")
        return result

    def stage(*_a, **_kw):
        effects.append("review")
        fingerprint = git._fingerprint_staged_diff(repo)
        assert fingerprint["ok"], fingerprint
        return {"status": "passed", "pre_fingerprint": fingerprint, "post_fingerprint": fingerprint}

    def tests(*_a, **_kw):
        effects.append("tests")
        if mode["pause"] == "tests":
            owner_pause.install_fence(drive, "resolver", request_id="after-tests")
        return "red tests" if mode["red"] == "tests" else None

    def smoke():
        effects.append("smoke")
        if mode["pause"] == "smoke":
            owner_pause.install_fence(drive, "resolver", request_id="during-smoke")
        return {"ok": mode["red"] != "smoke", "stderr": "red smoke", "returncode": 1}

    monkeypatch.setattr(git, "run_cmd", command)
    monkeypatch.setattr(git, "_check_overlapping_review_attempt", lambda *_a: "")
    monkeypatch.setattr(git, "_run_reviewed_stage_cycle", stage)
    monkeypatch.setattr(git, "_git_commit_with_tests", tests)
    monkeypatch.setattr(git, "_log_test_failure", lambda *_a: None)
    monkeypatch.setattr(git, "_record_commit_attempt", lambda _ctx, _msg, status, **kw: records.append({"status": status, **kw}))
    monkeypatch.setattr(git, "record_bound_commit_success", lambda *_a: records.append({"status": "committed"}))
    monkeypatch.setattr(update_merge, "update_restart_smoke", smoke)
    assisted._stub_worker_gates(monkeypatch)
    return SimpleNamespace(repo=repo, plan=plan, tx=tx, ctx=ctx, drive=drive,
                           effects=effects, records=records, mode=mode)


def _typed_commit(ctx):
    """The text and the typed result one commit_reviewed invocation publishes."""
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        return git._repo_commit_push(ctx, "reviewed managed resolution"), _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)


@pytest.mark.parametrize("boundary", ["commit", "tests"])
def test_managed_pause_retains_commit_and_resumes_unfinished_gates(managed_commit, boundary):
    f = managed_commit
    f.mode["pause"] = boundary
    result, typed = _typed_commit(f.ctx)
    assert (typed.status, typed.code, typed.text) == ("blocked", "LEGACY_BLOCKED", result)
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    assert commit != f.plan["base_sha"]
    assert "OWNER_PAUSE" in result and commit in result, result
    assert not result.startswith("OK")
    assert f.effects == (["review", "commit"] if boundary == "commit" else ["review", "commit", "tests"])
    durable = update_merge.read_update_tx()
    assert durable["merge_commit"] == commit
    assert durable["phase"] == "committing_assisted"
    assert not any(row["status"] in {"failed", "committed"} for row in f.records)
    boot = update_merge._recover_assisted_on_boot(durable, supervisor_ready=False)
    assert boot == {"finalized": False, "reason": "assisted_postcommit_paused"}, boot
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == commit
    # A retry under the same closed fence neither runs gates nor duplicates the commit.
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    # Recreate the context to prove the durable tx supplies the unfinished boundary.
    resumed = ToolContext(repo_dir=f.repo, drive_root=f.drive, branch_dev=f.ctx.branch_dev,
                          task_id="resolver", task_metadata=f.ctx.task_metadata)
    resumed.current_task_type = "task"
    resumed.task_lifecycle_bound = True
    result = git._repo_commit_push(resumed, "reviewed managed resolution")
    assert result.startswith("OK: committed"), result
    assert f.effects.count("commit") == f.effects.count("review") == 1
    assert f.effects[-2:] == ["tests", "smoke"]
    assert update_merge.read_update_tx()["phase"] == "pending_boot_smoke"
    assert update_merge.read_update_tx()["pre_restart_smoke"] == "passed"
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == commit


@pytest.mark.parametrize("phase", ["tests", "smoke"])
@pytest.mark.parametrize("pause_during", [False, True])
def test_real_failed_managed_gates_keep_existing_rollback(managed_commit, phase, pause_during):
    f = managed_commit
    f.mode.update(pause=phase if pause_during else "", red=phase)
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert not result.startswith("OK"), result
    assert "rolled back" in result.lower(), result
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert not update_merge.read_update_tx()
    assert f.effects == (["review", "commit", "tests"] if phase == "tests"
                         else ["review", "commit", "tests", "smoke"])
    assert any(row["status"] == "failed" for row in f.records)
    assert not any(row["status"] == "committed" for row in f.records)


@pytest.mark.parametrize("change", ["dirty", "head", "authority", "binding"])
def test_managed_resume_does_not_authorize_changed_subject_or_other_resolver(managed_commit, change):
    f = managed_commit
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    if change == "dirty":
        (f.repo / "a.txt").write_text("later unrelated work\n")
    elif change == "head":
        assert assisted._git(f.repo, "commit", "--allow-empty", "-m", "later work").returncode == 0
    elif change == "authority":
        f.ctx.task_metadata = {}
    else:
        tx = update_merge.read_update_tx()
        tx["postcommit_resume"]["post_fingerprint"]["binding"]["tree_sha"] = "bad-tree"
        update_merge.write_update_tx(tx)
    result, typed = _typed_commit(f.ctx)
    assert "MANAGED_UPDATE" in result and not result.startswith("OK"), result
    if change != "authority":  # A changed retained subject is a failure, never a successful call.
        assert (typed.status, typed.code, typed.text) == ("error", "LEGACY_TOOL_ERROR", result)
        assert "MANAGED_UPDATE_POSTCOMMIT_CHANGED" in result
    assert f.effects == ["review", "commit"]
    assert update_merge.read_update_tx()["merge_commit"] == commit


def test_explicit_stop_of_paused_managed_resolver_keeps_existing_rollback(managed_commit):
    f = managed_commit
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    result = update_merge.abort_orphaned_assisted_tx("resolver", f.ctx.task_metadata)
    assert result.get("rolled_back"), result
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert not update_merge.read_update_tx()


def test_managed_pause_before_commit_keeps_precommit_resolution(managed_commit, monkeypatch):
    f = managed_commit
    native_phase = git._managed_committing_phase_error
    def pause_before_commit(tx):
        result = native_phase(tx)
        owner_pause.install_fence(f.drive, "resolver", request_id="before-commit")
        return result
    monkeypatch.setattr(git, "_managed_committing_phase_error", pause_before_commit)
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert "OWNER_PAUSE_NOT_STARTED" in result
    assert f.effects == ["review"]
    assert not update_merge.read_update_tx().get("postcommit_resume")
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert update_merge.managed_assisted_precommit_verify(update_merge.read_update_tx())[0]
