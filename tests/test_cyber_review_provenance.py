"""A commit author record counts only after a review of that exact attempt.

Recovered first-review and exact-author evidence regression from the preserved Ouroboros candidate.
"""
import pytest

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("case", ["same", "no_first", "stale", "blocking", "post_changed"])
def test_commit_author_record_requires_prior_review_of_exact_attempt(tmp_path, monkeypatch, case):
    from types import SimpleNamespace
    from ouroboros.review_records import build_author_disposition
    from ouroboros.review_state import load_state
    from ouroboros.tools.commit_gate import _record_commit_attempt

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking" if case == "blocking" else "advisory")
    ctx = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, task_id="",
                          _review_advisory=[], _current_review_attempt_number=0)
    raw = [{"slot_id": "critic", "text": "Raw unresolved finding"}]
    if case != "no_first":
        _record_commit_attempt(ctx, commit_message="fixture", status="reviewing",
                               paid=True, pre_review_fingerprint="a" * 64, triad_raw_results=raw)
    author = build_author_disposition(
        disposition="deferred", rationale="The remaining finding is deferred.",
        subject_hash=("b" if case == "stale" else "a") * 64, enforcement="advisory",
    )
    _record_commit_attempt(ctx, commit_message="fixture", status="blocked",
                           pre_review_fingerprint="a" * 64,
                           post_review_fingerprint=("b" if case == "post_changed" else "a") * 64,
                           author_disposition=author)
    row = load_state(tmp_path).attempts[-1]
    assert row.status == "blocked"
    assert bool(row.author_disposition) is (case == "same")
    if case != "no_first":
        assert row.triad_raw_results == raw
