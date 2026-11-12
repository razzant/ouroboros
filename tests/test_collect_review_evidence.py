"""``collect_review_evidence`` scoping.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme. Covers task-scoped recent attempts, the repo-scoped stale marker, and
that a task's evidence inherits no obligation or debt the retired obligation
layer stored for its checkout (step 5).

The advisory-run ATTRIBUTION cases (own versus ``foreign_advisory_runs``, legacy
rows with no creator) retired with the run lists themselves: the retired advisory
lane is an archive the evidence no longer projects, so another task's old rows
cannot be narrated as this task's (``tests/test_advisory_run_archive.py``).
"""


def test_collect_review_evidence_keeps_recent_attempts_task_scoped(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()

    state = AdvisoryReviewState()
    state.record_attempt(CommitAttemptRecord(
        ts="2026-04-07T10:00:00+00:00",
        commit_message="other task attempt",
        status="blocked",
        repo_key=make_repo_key(repo_dir),
        tool_name="commit_reviewed",
        task_id="task-other",
        attempt=1,
        block_reason="critical_findings",
    ))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(
        tmp_path,
        task_id="task-current",
        repo_dir=repo_dir,
    )

    assert evidence["recent_attempts"] == []

def test_collect_review_evidence_scopes_the_stale_marker_to_repo(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        compute_snapshot_hash,
        make_repo_key,
        save_state,
    )

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    (repo_a / ".git").mkdir()
    (repo_b / ".git").mkdir()
    (repo_a / "tracked.py").write_text("print('repo a')\n", encoding="utf-8")
    (repo_b / "tracked.py").write_text("print('repo b')\n", encoding="utf-8")

    repo_a_key = make_repo_key(repo_a)
    repo_b_key = make_repo_key(repo_b)
    state = AdvisoryReviewState()
    state.advisory_runs.append({  # an archived row of the retired lane: never readiness
        "snapshot_hash": compute_snapshot_hash(repo_a), "commit_message": "repo a ready",
        "status": "fresh", "ts": "2026-04-07T10:00:00+00:00", "repo_key": repo_a_key,
    })
    state.record_attempt(CommitAttemptRecord(
        ts="2026-04-07T10:01:00+00:00",
        commit_message="repo b blocked",
        status="blocked",
        repo_key=repo_b_key,
        tool_name="commit_reviewed",
        task_id="task-b",
        attempt=1,
        block_reason="critical_findings",
        critical_findings=[{
            "item": "foreign_issue",
            "reason": "other repo only",
            "severity": "critical",
            "verdict": "FAIL",
        }],
    ))
    state.last_stale_from_edit_ts = "2026-04-07T10:02:00+00:00"
    state.last_stale_reason = "repo-b mutation"
    state.last_stale_repo_key = repo_b_key
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, repo_dir=repo_a)

    assert evidence["current_repo"]["stale_reason"] == ""
    assert evidence["current_repo"]["stale_ts"] == ""
    assert "foreign_issue" not in str(evidence)


def test_task_evidence_and_its_ledger_inherit_no_obligation_the_checkout_stored(tmp_path):
    """Home (step 5): over a state file that still holds an open obligation and a debt for the checkout, a task's
    review evidence carries neither (nor an omission counter for them), so it has no evidence until the task records
    something of its own, and the task's verification ledger fails on nothing it did not record."""
    import json

    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    repo_key = make_repo_key(repo_dir)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "advisory_review.json").write_text(json.dumps({  # what a body with Blocking left
        "state_version": 3, "attempts": [], "next_obligation_seq": 2, "next_commit_readiness_debt_seq": 2,
        "open_obligations": [{"obligation_id": "obl-0001", "item": "tests_affected", "severity": "critical",
                              "reason": "STORED-OBLIGATION coverage", "status": "still_open", "repo_key": repo_key}],
        "commit_readiness_debts": [{"debt_id": "crd-0001", "category": "readiness_warning", "status": "queued",
                                    "summary": "STORED-DEBT warning", "repo_key": repo_key}]}), encoding="utf-8")

    def ledger(evidence):
        return build_verification_ledger(task={"id": "task-now"}, loop_outcome={"outcome_axes": {}}, llm_trace={},
                                         artifact_bundle={}, review_evidence=evidence)

    quiet = collect_review_evidence(tmp_path, task_id="task-now", repo_dir=repo_dir)
    for key in ("open_obligations", "omitted_obligations", "commit_readiness_debts"):
        assert key not in quiet, key
    assert quiet["has_evidence"] is False and "STORED-" not in json.dumps(quiet)
    assert not [row for row in ledger(quiet)["entries"] if row.get("kind") == "review"]

    update_state(tmp_path, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-04-07T10:01:00+00:00", commit_message="mine", status="reviewed", repo_key=repo_key,
        task_id="task-now", attempt=1)))
    own = collect_review_evidence(tmp_path, task_id="task-now", repo_dir=repo_dir)
    assert own["has_evidence"] is True and [row["status"] for row in own["recent_attempts"]] == ["reviewed"]
    assert "STORED-" not in json.dumps(own)
