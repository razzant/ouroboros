"""``collect_review_evidence`` scoping.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers task-scoped recent attempts,
repo-scoped open obligations, and commit-readiness debt extraction.

Also covers advisory-run ATTRIBUTION. Repository readiness is repository-scoped
by design, but the run lists are not: a root's reflection received another
task's failed advisory rows with their identity stripped and narrated them as
its own failures. The split is a projection over existing records, so it is the
same under advisory and blocking review enforcement; only ``repo_commit_ready``
reads enforcement at all.
"""


def _repo_with_history(tmp_path, name="repo"):
    """A checkout with a tracked file, so ``compute_snapshot_hash`` is stable."""
    repo_dir = tmp_path / name
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "tracked.py").write_text("print('hello')\n", encoding="utf-8")
    return repo_dir


_FOREIGN_FAILURE = "deleted the release notes instead of editing them"


def _advisory_run(*, snapshot_hash, repo_key, task_id, status="fresh", failing=""):
    from ouroboros.review_state import AdvisoryRunRecord

    return AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=f"advisory for {task_id or '(legacy)'}",
        status=status,
        ts="2026-08-08T10:00:00+00:00",
        repo_key=repo_key,
        task_id=task_id,
        attempt=1,
        phase="advisory",
        items=[{"verdict": "FAIL", "severity": "critical", "item": "destructive_edit",
                "reason": failing}] if failing else [],
    )


def test_another_tasks_advisory_failures_are_not_rendered_as_this_tasks_own(tmp_path):
    """The reproduced incident: A reflects, and B's FAIL rows are B's, labelled."""
    from ouroboros.review_evidence import collect_review_evidence, format_review_evidence_for_prompt
    from ouroboros.review_state import AdvisoryReviewState, make_repo_key, save_state

    repo_dir = _repo_with_history(tmp_path)
    repo_key = make_repo_key(repo_dir)
    state = AdvisoryReviewState()
    state.add_run(_advisory_run(snapshot_hash="b-snapshot", repo_key=repo_key,
                                task_id="task-b", failing=_FOREIGN_FAILURE))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, task_id="task-a", repo_dir=repo_dir)

    assert evidence["recent_advisory_runs"] == []
    assert evidence["omitted_advisory_runs"] == 0
    assert [row["task_id"] for row in evidence["foreign_advisory_runs"]] == ["task-b"]
    assert evidence["foreign_advisory_runs"][0]["findings"][0]["reason"] == _FOREIGN_FAILURE
    assert evidence["foreign_advisory_runs"][0]["snapshot_hash"] == "b-snapshot"
    assert evidence["has_evidence"] is True

    rendered = format_review_evidence_for_prompt(evidence, max_chars=8000)
    heading = rendered.find("ADVISORY RUNS OF OTHER TASKS")
    assert heading >= 0
    assert "task-b" in rendered[heading:]
    # The failure text exists ONLY after the attributing heading: nothing above
    # it can be read as this task's own record.
    assert rendered.find(_FOREIGN_FAILURE) > heading
    assert _FOREIGN_FAILURE not in rendered[:heading]


def test_another_tasks_exact_snapshot_advisory_still_answers_repository_readiness(tmp_path):
    """Attribution is not scoping: the checkout is still covered by B's review."""
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import (
        AdvisoryReviewState, compute_snapshot_hash, make_repo_key, save_state,
    )

    repo_dir = _repo_with_history(tmp_path)
    repo_key = make_repo_key(repo_dir)
    state = AdvisoryReviewState()
    state.add_run(_advisory_run(snapshot_hash=compute_snapshot_hash(repo_dir),
                                repo_key=repo_key, task_id="task-b"))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, task_id="task-a", repo_dir=repo_dir)

    assert evidence["current_repo"]["advisory_status"] == "fresh"
    assert evidence["current_repo"]["repo_commit_ready"] is True
    assert evidence["recent_advisory_runs"] == []
    assert [row["task_id"] for row in evidence["foreign_advisory_runs"]] == ["task-b"]


def test_a_run_without_a_task_id_stays_unknown_and_is_never_re_attributed(tmp_path):
    """A legacy row predates the field; it is not evidence that it is mine."""
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, make_repo_key, save_state

    repo_dir = _repo_with_history(tmp_path)
    state = AdvisoryReviewState()
    state.add_run(_advisory_run(snapshot_hash="legacy-snapshot",
                                repo_key=make_repo_key(repo_dir), task_id=""))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, task_id="task-a", repo_dir=repo_dir)

    assert evidence["foreign_advisory_runs"] == []
    assert [row["task_id"] for row in evidence["recent_advisory_runs"]] == [""]
    assert evidence["recent_advisory_runs"][0]["snapshot_hash"] == "legacy-snapshot"


def test_an_empty_repo_key_does_not_pull_another_tasks_runs_into_this_task(tmp_path):
    """No repo_dir widens the candidate list to the whole drive; identity still holds."""
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, make_repo_key, save_state

    repo_a = _repo_with_history(tmp_path, "repo-a")
    repo_b = _repo_with_history(tmp_path, "repo-b")
    state = AdvisoryReviewState()
    state.add_run(_advisory_run(snapshot_hash="a-snapshot", repo_key=make_repo_key(repo_a),
                                task_id="task-a"))
    state.add_run(_advisory_run(snapshot_hash="b-snapshot", repo_key=make_repo_key(repo_b),
                                task_id="task-b", failing=_FOREIGN_FAILURE))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, task_id="task-a", repo_dir=None)

    assert evidence["repo_key"] == ""
    assert [row["task_id"] for row in evidence["recent_advisory_runs"]] == ["task-a"]
    assert [row["task_id"] for row in evidence["foreign_advisory_runs"]] == ["task-b"]


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


def test_the_foreign_section_takes_a_bounded_share_of_the_prompt_budget():
    """Three long foreign advisory runs must not squeeze the task's OWN evidence out
    of a bounded prompt: the attributing section is capped at a quarter of the bound,
    and the own body keeps the rest (WP-D repair, owner Q2A)."""
    from ouroboros.review_evidence import format_review_evidence_for_prompt

    own_reason = "OWN-FINDING " * 600  # ~7,200 chars of this task's own record
    foreign = [
        {"task_id": f"task-{n}", "findings": [{"reason": "FOREIGN-FINDING " * 250}]}
        for n in ("b", "c", "d")
    ]
    evidence = {
        "task_id": "task-a", "has_evidence": True,
        "recent_advisory_runs": [{"task_id": "task-a", "findings": [{"reason": own_reason}]}],
        "foreign_advisory_runs": foreign,
    }
    rendered = format_review_evidence_for_prompt(evidence, max_chars=8000)
    heading = rendered.find("ADVISORY RUNS OF OTHER TASKS")
    assert heading > 0
    own_body, foreign_body = rendered[:heading], rendered[heading:]
    # The own record keeps at least half of the bound; the foreign section at most a quarter
    # (plus its omission marker).
    assert own_body.count("OWN-FINDING") >= 300
    assert len(foreign_body) <= 8000 // 4 + 200
    assert "OMISSION NOTE" in foreign_body
    assert "FOREIGN-FINDING" not in own_body

