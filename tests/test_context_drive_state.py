"""The drive-state projection, the review ledger and the settled continuations.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the typed
drive-state section and its pointer, the review ledger that caps runs and attempts with
omission notes, the continuations that retire after their age window, and the ones an
earlier obligation mark keeps until a commit lands in their checkout.
"""

from __future__ import annotations
from tests._store_shared import seed_result






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
    (tmp_path / "state" / "state.initialized.json").write_text(  # a completed initialization proves its controls
        json.dumps({"initialization_id": "init-1", "phase": "complete"}), encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text(json.dumps({
        "initialization_id": "init-1",
        "session_id": "abc123",
        "current_branch": "ouroboros",
        "evolution_mode_enabled": False,
        "bg_consciousness_enabled": False,
        "bg_consciousness_stop_source": "owner",  # the mind sees who holds the clock off
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
    assert '"bg_consciousness_stop_source": "owner"' in section
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


def test_review_ledger_caps_attempts_with_omission_notes(tmp_path):
    """W3 adjacent (b): the historical review ledger rides into EVERY task's
    context — cap attempts at the 5 most recent with EXPLICIT omission
    notes (the continuation pattern) and truncate commit messages; the full
    ledger stays behind review_status. The retired advisory lane's archived
    rows never render there."""
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        format_status_section,
    )

    state = AdvisoryReviewState()
    long_msg = "feat: " + ("y" * 2000)
    for i in range(8):
        state.advisory_runs.append({"snapshot_hash": f"hash{i:04d}00000000", "commit_message": f"archived run {i}",
                                    "status": "fresh", "ts": f"2026-01-0{i + 1}T00:00:00"})
    for i in range(8):
        state.record_attempt(CommitAttemptRecord(
            status="succeeded",
            commit_message=long_msg if i == 6 else f"attempt commit {i}",
            ts=f"2026-01-0{i + 1}T01:00:00",
            attempt=i + 1,
        ))

    section = format_status_section(state)

    assert "3 older attempt(s) omitted" in section
    assert "review_status" in section
    assert "hash000" not in section and "archived run" not in section  # the archive never renders
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


def test_a_settled_continuation_stays_while_an_earlier_obligation_mark_awaits_a_landing(tmp_path):
    """An aged continuation of a settled task stays in context while it carries an obligation mark
    written before obligations retired (its ids or its snapshot) and no commit has landed in its checkout since it
    was written. A landing after it, findings alone, or no record of work at all leave it to the age rule, which
    archives it (durable, disclosed). The rule reads the continuation and the landings, never an obligation table."""
    import json as _json

    from ouroboros.agent_task_pipeline import build_review_context
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state
    from ouroboros.task_continuation import (
        ReviewContinuation,
        archived_continuation_dir,
        continuation_path,
        save_review_continuation,
    )

    class FakeEnv:
        repo_dir = tmp_path / "repo"
        drive_root = tmp_path

    (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
    repo_key = make_repo_key(tmp_path / "repo")
    finding = {"item": "tests_affected", "reason": "Coverage still missing", "severity": "critical", "verdict": "FAIL"}
    snapshot = [{"obligation_id": "obl-0002", "fingerprint": "finding:tests_affected:abc", "status": "still_open"}]

    def _aged(task_id, written, **fields):
        save_review_continuation(tmp_path, ReviewContinuation(
            task_id=task_id, source="blocked_review", stage="blocking_review", repo_key=repo_key, **fields))
        path = continuation_path(tmp_path, task_id)
        data = _json.loads(path.read_text(encoding="utf-8"))
        data["created_ts"] = data["updated_ts"] = written
        path.write_text(_json.dumps(data), encoding="utf-8")
        seed_result(tmp_path, task_id, {"_schema_version": 1, "id": task_id, "status": "failed"})

    _aged("findingstask", "2026-01-10T00:00:00+00:00", critical_findings=[finding], review_record_id="rv-1")
    _aged("idstask", "2026-01-10T00:00:00+00:00", obligation_ids=["obl-0001"])
    _aged("snapshottask", "2026-01-10T00:00:00+00:00", open_obligations=snapshot)
    _aged("landedtask", "2026-01-01T00:00:00+00:00", obligation_ids=["obl-0003"], critical_findings=[finding])
    _aged("emptytask", "2026-01-10T00:00:00+00:00")
    update_state(tmp_path, lambda state: state.record_attempt(CommitAttemptRecord(  # lands between the two stamps
        ts="2026-01-05T00:00:00+00:00", finished_ts="2026-01-05T00:00:00+00:00", commit_message="landed",
        status="succeeded", repo_key=repo_key, task_id="other", attempt=1)))

    dynamic_text = build_review_context(FakeEnv())

    for kept in ("idstask", "snapshottask"):
        assert continuation_path(tmp_path, kept).exists(), kept
        assert f"task={kept}" in dynamic_text
    for released in ("findingstask", "landedtask", "emptytask"):
        assert not continuation_path(tmp_path, released).exists(), released
        assert (archived_continuation_dir(tmp_path) / f"{released}.json").exists(), released  # durable
        assert released in dynamic_text and f"task={released}" not in dynamic_text  # the archive line names it
