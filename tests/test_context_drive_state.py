"""The drive-state projection, the review ledger and the settled continuations.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the typed
drive-state section and its pointer, the review ledger that caps runs and attempts with
omission notes, the continuations that retire after their age window, and the one with
open obligations that must survive that retirement.
"""

from __future__ import annotations






def test_drive_state_section_is_typed_projection_with_pointer(tmp_path):
    """W3 adjacent (a): the Drive state section projects the fields the agent
    reasons about and NAMES the omitted internal caches with an on-demand
    pointer (P1: disclosed omission) instead of dumping state.json wholesale —
    the budget narrative stays with the usage-accounting authority in the
    Runtime section."""
    import json

    from ouroboros.context import _drive_state_section

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({
        "session_id": "abc123",
        "current_branch": "ouroboros",
        "evolution_mode_enabled": False,
        "budget_drift_alert": True,
        "budget_drift_pct": 48.05,
        "spent_usd": 1699.3,
        "managed_update_cache": {"latest_sha": "x" * 40, "latest_message": "big"},
        "usage_accounting": {"settled_usd": 1633.1},
        "openrouter_last_check_call": 5750,
    }), encoding="utf-8")

    section = _drive_state_section(FakeEnv())

    assert section.startswith("## Drive state")
    assert '"session_id": "abc123"' in section
    assert '"budget_drift_alert": true' in section
    # Internal caches / duplicated spend narrative are OMITTED but NAMED.
    assert '"managed_update_cache"' not in section
    assert '"usage_accounting"' not in section
    assert '"spent_usd"' not in section
    for named in ("managed_update_cache", "usage_accounting", "spent_usd", "openrouter_last_check_call"):
        assert named in section  # named in the omission note
    assert "read_file(root='runtime_data', path='state/state.json')" in section

    # Missing/empty file: still a valid section, no omission note needed.
    (tmp_path / "state" / "state.json").unlink()
    empty = _drive_state_section(FakeEnv())
    assert empty.startswith("## Drive state")
    assert "read_file" not in empty


def test_review_ledger_caps_runs_and_attempts_with_omission_notes(tmp_path):
    """W3 adjacent (b): the historical review ledger rides into EVERY task's
    context — cap runs/attempts at the 5 most recent with EXPLICIT omission
    notes (the continuation pattern) and truncate commit messages; the full
    ledger stays behind review_status."""
    from ouroboros.review_state import (
        AdvisoryReviewState,
        AdvisoryRunRecord,
        CommitAttemptRecord,
        format_status_section,
    )

    state = AdvisoryReviewState()
    long_msg = "feat: " + ("y" * 2000)
    for i in range(8):
        state.add_run(AdvisoryRunRecord(
            snapshot_hash=f"hash{i:04d}00000000",
            commit_message=long_msg if i == 7 else f"commit {i}",
            status="fresh",
            ts=f"2026-01-0{i + 1}T00:00:00",
        ))
    for i in range(8):
        state.record_attempt(CommitAttemptRecord(
            status="succeeded",
            commit_message=f"attempt commit {i}",
            ts=f"2026-01-0{i + 1}T01:00:00",
            attempt=i + 1,
        ))

    section = format_status_section(state)

    assert "3 older advisory run(s) omitted" in section
    assert "3 older attempt(s) omitted" in section
    assert "review_status" in section
    assert "hash0007" in section       # newest kept
    assert "hash0000" not in section   # oldest omitted
    assert "attempt commit 7" in section
    assert "attempt commit 0" not in section
    # The 2000-char commit message is display-truncated with the explicit notice.
    assert "y" * 2000 not in section
    assert "truncated at 300 chars" in section


def test_settled_continuations_retire_after_age_window(tmp_path):
    """W3 adjacent (b): a continuation whose owning task SETTLED and that sat
    un-resumed past the age window is archived (durable move, never deleted);
    fresh settled records stay — they are the designed cross-task resume
    pointer."""
    from ouroboros.task_continuation import (
        ReviewContinuation,
        archived_continuation_dir,
        continuation_path,
        list_review_continuations,
        retire_settled_continuations,
        save_review_continuation,
    )

    old = save_review_continuation(tmp_path, ReviewContinuation(
        task_id="oldtask", source="commit_blocked", stage="review"))
    # Age the record past the window (rewrite the stored timestamps).
    import json as _json
    path = continuation_path(tmp_path, "oldtask")
    data = _json.loads(path.read_text(encoding="utf-8"))
    data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
    path.write_text(_json.dumps(data), encoding="utf-8")

    save_review_continuation(tmp_path, ReviewContinuation(
        task_id="freshtask", source="commit_blocked", stage="review"))

    settled = {"oldtask": True, "freshtask": True, "runningtask": False}
    retired = retire_settled_continuations(tmp_path, is_settled=lambda tid: settled.get(tid, False))

    assert retired == ["oldtask"]
    assert not continuation_path(tmp_path, "oldtask").exists()
    assert (archived_continuation_dir(tmp_path) / "oldtask.json").exists()  # durable, not deleted
    remaining, _corrupt = list_review_continuations(tmp_path)
    assert [c.task_id for c in remaining] == ["freshtask"]

    # An old continuation of a NON-settled task stays put.
    save_review_continuation(tmp_path, ReviewContinuation(
        task_id="runningtask", source="commit_blocked", stage="review"))
    path = continuation_path(tmp_path, "runningtask")
    data = _json.loads(path.read_text(encoding="utf-8"))
    data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
    path.write_text(_json.dumps(data), encoding="utf-8")
    assert retire_settled_continuations(tmp_path, is_settled=lambda tid: settled.get(tid, False)) == []
    assert continuation_path(tmp_path, "runningtask").exists()
    assert old.task_id == "oldtask"


def test_settled_continuation_with_open_obligations_survives_age_retirement(tmp_path):
    """A settled FAILED task whose continuation records obligations that are
    STILL open in the review ledger is genuinely unresolved review work: age
    must not archive it out of context (P1/P3). A same-age settled sibling with
    no open markers still retires — the noise-reduction path stays."""
    import json as _json

    from ouroboros.agent_task_pipeline import build_review_context
    from ouroboros.review_state import (
        AdvisoryReviewState,
        ObligationItem,
        make_repo_key,
        save_state,
    )
    from ouroboros.task_continuation import (
        ReviewContinuation,
        archived_continuation_dir,
        continuation_path,
        save_review_continuation,
    )

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

        def repo_path(self, p):
            return tmp_path / "repo" / p

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path

    env = FakeEnv()
    (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "tracked.py").write_text("print('hi')\n", encoding="utf-8")
    repo_key = make_repo_key(tmp_path / "repo")

    def _aged_continuation(task_id, obligation_ids):
        save_review_continuation(tmp_path, ReviewContinuation(
            task_id=task_id, source="commit_blocked", stage="review",
            block_reason="critical_findings", obligation_ids=obligation_ids))
        path = continuation_path(tmp_path, task_id)
        data = _json.loads(path.read_text(encoding="utf-8"))
        data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
        path.write_text(_json.dumps(data), encoding="utf-8")

    _aged_continuation("unresolvedtask", ["obl-open-1"])
    _aged_continuation("closedtask", ["obl-long-gone"])
    task_results = tmp_path / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    for tid in ("unresolvedtask", "closedtask"):
        (task_results / f"{tid}.json").write_text(
            _json.dumps({"id": tid, "status": "failed"}), encoding="utf-8")

    state = AdvisoryReviewState(open_obligations=[
        ObligationItem(
            obligation_id="obl-open-1",
            item="tests_affected",
            severity="critical",
            reason="Coverage still missing",
            source_attempt_ts="2026-01-01T00:00:00+00:00",
            source_attempt_msg="blocked commit",
            repo_key=repo_key,
            fingerprint="finding:tests_affected:abc123",
        )
    ])
    save_state(tmp_path, state)

    dynamic_text = build_review_context(env)

    # Unresolved work survives the age window and stays in cognitive context.
    assert continuation_path(tmp_path, "unresolvedtask").exists()
    assert "task=unresolvedtask" in dynamic_text
    # The provably-closed sibling still rides the age path (durable, disclosed).
    assert not continuation_path(tmp_path, "closedtask").exists()
    assert (archived_continuation_dir(tmp_path) / "closedtask.json").exists()
    assert "closedtask" in dynamic_text  # transient archive disclosure line
