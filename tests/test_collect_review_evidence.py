"""``collect_review_evidence`` scoping.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers task-scoped recent attempts,
repo-scoped open obligations, and commit-readiness debt extraction.
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

def test_collect_review_evidence_scopes_open_obligations_to_repo(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import (
        AdvisoryReviewState,
        AdvisoryRunRecord,
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
    state.add_run(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(repo_a),
        commit_message="repo a ready",
        status="fresh",
        ts="2026-04-07T10:00:00+00:00",
        repo_key=repo_a_key,
    ))
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

    assert evidence["current_repo"]["repo_commit_ready"] is True
    assert evidence["current_repo"]["stale_reason"] == ""
    assert evidence["current_repo"]["stale_ts"] == ""
    assert evidence["open_obligations"] == []
    assert evidence["commit_readiness_debts"] == []

def test_collect_review_evidence_includes_commit_readiness_debt(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "tracked.py").write_text("print('hi')\n", encoding="utf-8")

    repo_key = make_repo_key(repo_dir)
    state = AdvisoryReviewState()
    for idx, reason in enumerate(["missing tests", "coverage still missing"], start=1):
        state.record_attempt(CommitAttemptRecord(
            ts=f"2026-04-07T10:0{idx}:00+00:00",
            commit_message=f"blocked {idx}",
            status="blocked",
            repo_key=repo_key,
            tool_name="commit_reviewed",
            task_id=f"task-{idx}",
            attempt=idx,
            block_reason="critical_findings",
            critical_findings=[{
                "item": "tests_affected",
                "reason": reason,
                "severity": "critical",
                "verdict": "FAIL",
            }],
            readiness_warnings=["Start retry from review debt."],
        ))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, repo_dir=repo_dir)

    assert evidence["current_repo"]["repo_commit_ready"] is False
    assert len(evidence["commit_readiness_debts"]) >= 1
    assert evidence["commit_readiness_debts"][0]["category"] in {"obligation_repeat", "readiness_warning"}
