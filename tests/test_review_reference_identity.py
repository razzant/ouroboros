"""A paid reference is an exact author/attempt identity, never new dispatch authority."""
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as ledger
from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state
from ouroboros.tools.review_collection import reference_for, resolve_reference, prepare_commit_collection
from tests.test_review_ledger_settlement import _record


def _source(tmp_path, *, commit=False, **fields):
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path / "repo")
    ctx.repo_dir.mkdir()
    record = _record(tmp_path)
    record = ledger.revise_record(tmp_path, record["record_id"],
                                  lambda now: {**now, "surface": "commit_gate" if commit else "change"})
    row = CommitAttemptRecord(ts="2026-10-21T00:00:00Z", commit_message="original", status="reviewing",
        task_id=ctx.task_id, repo_key=make_repo_key(ctx.repo_dir), tool_name="commit_reviewed" if commit else "review_change",
        attempt=1, paid=True, review_record_id=record["record_id"], pre_review_fingerprint="original-bytes")
    for key, value in fields.items():
        setattr(row, key, value)
    update_state(tmp_path, lambda state: state.record_attempt(row))
    return ctx, row, record, reference_for(record, row)


@pytest.mark.parametrize("fault", ["none", "foreign_author", "not_paid", "stripped", "identity", "tool", "repository"])
def test_identity_rejects_only_the_uncollectible_source(tmp_path, fault):
    commit = fault == "repository"
    fields = {"paid": False} if fault == "not_paid" else {"raw_stripped": True} if fault == "stripped" else {}
    ctx, row, record, ref = _source(tmp_path, commit=commit, **fields)
    if fault == "foreign_author":
        ctx.task_id = "another-author"
    elif fault == "identity":
        ref["attempt"] = row.attempt + 1
    elif fault == "repository":
        ctx.repo_dir = tmp_path / "another-repo"
    elif fault == "tool":
        commit = True
    if fault == "none":
        root, recovered, loaded = resolve_reference(ctx, ref)
        assert root == tmp_path and recovered.review_record_id == record["record_id"] and loaded == record
    else:
        with pytest.raises(ValueError):
            resolve_reference(ctx, ref, commit=commit)


def test_compacted_attempt_still_collects_its_settled_record(tmp_path):
    ctx, _, record, ref = _source(tmp_path, raw_stripped=True)
    ledger.revise_record(tmp_path, record["record_id"], lambda now: {**now, "state": "settled"})
    assert resolve_reference(ctx, ref)[2]["state"] == "settled"


@pytest.mark.parametrize("fault", ["none", "already_committed", "message", "brief"])
def test_commit_resume_cannot_repeat_an_effect_or_change_intent(tmp_path, fault):
    ctx, row, _, ref = _source(tmp_path, commit=True, status="succeeded" if fault == "already_committed" else "reviewing")
    kwargs = {"commit_message": "other"} if fault == "message" else {"mutations": ("other brief",)} if fault == "brief" else {}
    if fault == "none":
        assert prepare_commit_collection(ctx, ref, **kwargs).attempt == row.attempt
        assert ctx._review_resume_pending and ctx._current_review_attempt_number == row.attempt
    else:
        with pytest.raises(ValueError):
            prepare_commit_collection(ctx, ref, **kwargs)
