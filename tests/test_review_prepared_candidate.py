"""The paid review boundary observes one fully prepared, retained Git index."""

import subprocess
import time

import pytest

from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state
from ouroboros.tools import git
from tests.test_advisory_inline_freshness import candidate  # noqa: F401


def _git(ctx, *args):
    return subprocess.check_output(["git", *args], cwd=ctx.repo_dir, text=True).strip()


def _stage(ctx):
    return git._run_reviewed_stage_cycle(ctx, "candidate", time.time(), paths=["change.py"], require_release_tag=False)


@pytest.fixture
def stage_context(candidate, monkeypatch):  # noqa: F811 - imported pytest fixtures
    candidate._last_triad_raw_results = []
    candidate._last_scope_raw_result = {}
    candidate._review_degraded_reasons = []
    monkeypatch.setattr(git, "commit_review_contract_fingerprint", lambda: "contract")
    monkeypatch.setattr(git, "_advisory_and_tests_gate", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: (None, None, "", []))
    return candidate


def test_mechanical_files_are_staged_before_fingerprint_and_preflight(stage_context, monkeypatch):
    ctx = stage_context
    calls = []

    def prepare(_ctx, repo, drive, paths):
        calls.append("prepare")
        (repo / "generated.txt").write_text("prepared carrier\n")
        return ["generated.txt"]

    def preflight(_ctx, message, started, **kwargs):
        calls.append("preflight")
        assert _git(ctx, "show", ":generated.txt") == "prepared carrier"
        assert set(kwargs["advisory_paths"]) == {"change.py", "generated.txt"}
        assert git._fingerprint_staged_diff(ctx.repo_dir)["fingerprint"] == ctx._current_review_binding_test

    def free_gate(_ctx, message, started, **kwargs):
        calls.append("admission")
        ctx._current_review_binding_test = kwargs["pre_fingerprint"]["fingerprint"]
        ctx._current_review_retry_key = "prepared-review"

    monkeypatch.setattr("ouroboros.commit_admission.auto_sync_release_metadata_if_needed", prepare)
    monkeypatch.setattr(git, "_free_cycle_gate", free_gate)
    monkeypatch.setattr(git, "_advisory_and_tests_gate", preflight)
    result = _stage(ctx)
    assert result["status"] == "passed"
    assert calls == ["prepare", "admission", "preflight"]
    assert result["pre_fingerprint"]["fingerprint"] == result["post_fingerprint"]["fingerprint"]


@pytest.mark.parametrize("stage_new_bytes", [False, True])
def test_preflight_mutation_cannot_reach_triad(stage_context, monkeypatch, stage_new_bytes):
    ctx = stage_context

    def preflight(*args, **kwargs):
        (ctx.repo_dir / "change.py").write_text("value = 3\n")
        if stage_new_bytes:
            _git(ctx, "add", "change.py")

    monkeypatch.setattr(git, "_advisory_and_tests_gate", preflight)
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: pytest.fail("changed material must not dispatch"))
    assert _stage(ctx)["block_reason"] == "revalidation_failed"


def test_budget_ceiling_refuses_before_preflight(stage_context, monkeypatch):
    ctx = stage_context
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    update_state(ctx.drive_root, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-09-06T00:00:00Z", commit_message="previous", status="succeeded",
        repo_key=make_repo_key(ctx.repo_dir), task_id=ctx.task_id,
        root_task_id=ctx.task_id, paid=True, attempt=1,
    )))
    monkeypatch.setattr(git, "_advisory_and_tests_gate", lambda *a, **kw: pytest.fail("no paid preflight before admission"))
    assert _stage(ctx)["block_reason"] == "review_cycles_exhausted"


@pytest.mark.parametrize("change_index", [False, True])
def test_pending_retry_never_prepares_or_restages(stage_context, monkeypatch, change_index):
    ctx = stage_context
    calls = []

    def wave(inner, *args, **kwargs):
        from ouroboros.review_custody import prepare_frozen_review_reconciliation

        calls.append(bool(inner._review_reconcile_only))
        if inner._review_reconcile_only:
            prepare_frozen_review_reconciliation(inner, inner._pending_review_attempt)
        else:
            inner._review_paid_stamp()
        inner._last_triad_raw_results = [{
            "slot_id": "triad-one", "operation_id": "pending-operation",
            "model_id": "test/model", "status": "error",
            "operation_state": "in_flight", "late_result_pending": True,
            "pending_invocation_id": "existing-invocation", "error": "still pending",
        }]
        return "review pending", None, "review_late_result_pending", []

    monkeypatch.setattr(git, "_run_parallel_review", wave)
    first = _stage(ctx)
    assert first["block_reason"] == "review_late_result_pending"
    original_index = _git(ctx, "write-tree")
    assert _git(ctx, "diff", "--cached", "--name-only") == "change.py"
    assert load_state(ctx.drive_root).attempts[-1].late_result_pending
    # A user can make further unstaged edits while the old reviewer is live;
    # exact reconciliation must continue to own the original index only.
    (ctx.repo_dir / "change.py").write_text("value = 99\n")
    (ctx.repo_dir / "untracked.txt").write_text("keep me\n")
    if change_index:
        _git(ctx, "reset", "HEAD")
    before_retry = _git(ctx, "write-tree")
    assert git._check_overlapping_review_attempt(ctx) is None
    assert ctx._review_resume_pending
    monkeypatch.setattr("ouroboros.commit_admission.auto_sync_release_metadata_if_needed", lambda *a: pytest.fail("pending must not prepare"))
    monkeypatch.setattr(git, "_advisory_and_tests_gate", lambda *a, **kw: pytest.fail("pending must not rerun preflight"))
    second = _stage(ctx)
    assert _git(ctx, "write-tree") == before_retry
    assert (ctx.repo_dir / "change.py").read_text() == "value = 99\n"
    assert (ctx.repo_dir / "untracked.txt").read_text() == "keep me\n"
    if change_index:
        assert second["block_reason"] == "overlap_guard"
        assert calls == [False]
    else:
        assert before_retry == original_index
        assert second["block_reason"] == "review_late_result_pending"
        assert calls == [False, True]
    assert sum(row.paid for row in load_state(ctx.drive_root).attempts) == 1
