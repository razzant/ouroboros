"""The retired advisory pre-review lane is an archive: its rows are kept, never read as guidance.

Decision 3A retired the advisory gate and its writer; the owner then chose to keep the old
advisory pre-review only as an archive. What a former install recorded — the
``advisory_runs`` rows and the readiness debts they minted (``source=advisory_review`` AND
a ``readiness_warning:advisory:`` fingerprint) — survives every ordinary save exactly as
stored and is named where it lives (``review_status.advisory_run_archive``, the store
actually read), while the context, the ``review_status`` projection and the task evidence
read as if it were absent. An ``advisory_stale`` debt copy of the stale marker is archived
the same way; the preflight look and its marker stay live beside it, and live debts keep
their ordinary lifecycle.

Retired with that decision, because their subject no longer exists: rendering old runs in
the historical ledger, the ``review_status`` run list and its run-derived freshness
(``latest_advisory_hash``, failure reasons, the redacted rejoin intent), the task evidence's
own/foreign run lists and ``current_repo.advisory_status``, run invalidation on edit, and
debt minted from an old run's readiness warnings.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ouroboros import review_ledger
from ouroboros.review_state import (
    _STATE_RELPATH,
    CommitAttemptRecord,
    load_state,
    make_repo_key,
    update_state,
)


def _checkout(root: pathlib.Path) -> pathlib.Path:
    repo = root / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    return repo


def _archive_rows(repo_key: str) -> list:
    """Rows a former install wrote, in shapes the retired loader rewrote or refused."""
    return [
        {"snapshot_hash": "a" * 32, "commit_message": "старый коммит", "status": "fresh",
         "ts": "2026-08-01T10:00:00+00:00", "repo_key": repo_key, "task_id": "task-old", "attempt": 2,
         "items": [{"verdict": "FAIL", "severity": "critical", "item": "archive_finding",
                    "reason": "ARCHIVE-SENTINEL finding"}],
         "raw_result": "полный ответ рецензента, ARCHIVE-SENTINEL\n" * 200,
         "bypass_reason": "ARCHIVE-SENTINEL bypass", "readiness_warnings": ["ARCHIVE-SENTINEL warning"],
         "prompt_chars": 786401, "duration_sec": 48.35,
         "execution": {"pending_invocation_id": "inv-old", "intent": {"goal": "an old goal"}},
         "future_field": {"nested": [1, 2.5, True, None]}},
        {"snapshot_hash": "b", "status": "bypassed", "ts": "2026-08-02T10:00:00Z", "repo_key": repo_key,
         "task_id": None, "execution": "custody text", "attempt": "3"},
        {"status": "skipped", "repo_key": repo_key},
        "a row that is not an object",
    ]


def _archived_debts(repo_key: str) -> list:
    """Debts the lane minted, as a former install stored them, in shapes the typed debt
    record rewrites: an unknown nested field, more than five (and repeated) evidence lines,
    empty and missing timestamps, mixed value types, a row missing most fields; the last is the
    ``advisory_stale`` copy of the stale marker a former reviewed attempt minted."""
    return [
        {"debt_id": "crd-0005", "category": "readiness_warning", "summary": "ARCHIVE-DEBT-SENTINEL open",
         "status": "queued", "repo_key": repo_key, "fingerprint": "readiness_warning:advisory:0123456789ab",
         "title": "Readiness warning debt", "source": "advisory_review",
         "source_obligation_ids": ["obl-0007", "obl-0007", None],
         "evidence": [f"ARCHIVE-DEBT-SENTINEL line {n}" for n in range(7)] + ["ARCHIVE-DEBT-SENTINEL line 0", 42],
         "first_seen_at": "", "last_seen_at": "", "verified_at": None,
         "occurrence_count": "3", "consecutive_observations": None,
         "legacy_note": {"nested": [1, 2.5, True, None], "text": "kept as written"}},
        {"debt_id": "crd-0902", "category": "readiness_warning", "summary": "ARCHIVE-DEBT-SENTINEL resolved",
         "severity": "warning", "status": "verified", "repo_key": repo_key,
         "fingerprint": "readiness_warning:advisory:fedcba987654", "title": "Readiness warning debt",
         "source": "advisory_review", "evidence": [], "first_seen_at": "2026-08-01T10:00:00+00:00",
         "updated_at": "2026-08-04T10:00:00+00:00", "verified_at": "2026-08-04T10:00:00+00:00",
         "occurrence_count": 1.0, "consecutive_observations": 0},
        {"debt_id": "crd-0903", "source": "advisory_review", "status": "detected",
         "fingerprint": "readiness_warning:advisory:777777777777"},
        {"debt_id": "crd-0003", "category": "advisory_stale", "summary": "ARCHIVE-DEBT-SENTINEL stale",
         "status": "queued", "repo_key": repo_key, "fingerprint": "advisory_stale", "source": "review_state",
         "occurrence_count": 1, "consecutive_observations": 1},
    ]


def _live_debts(repo_key: str) -> list:
    """Live debts beside the archive; the first two carry only one half of its predicate."""
    def row(debt_id, category, status, fingerprint, source):
        return {"debt_id": debt_id, "category": category, "summary": f"live {debt_id}", "status": status,
                "repo_key": repo_key, "fingerprint": fingerprint, "source": source,
                "occurrence_count": 1, "consecutive_observations": 1 if status == "queued" else 0}

    return [row("crd-0001", "readiness_warning", "queued", "readiness_warning:attempt:aaaaaaaaaaaa", "advisory_review"),
            row("crd-0002", "readiness_warning", "queued", "readiness_warning:advisory:bbbbbbbbbbbb", "review_state"),
            row("crd-0004", "obligation_repeat", "verified", "obligation_repeat:obl-0004", "review_state")]


def _inject_archive(drive: pathlib.Path, runs: list, debts: list) -> dict:
    """Write rows straight into the state file, independent of the store's encoder."""
    path = drive / _STATE_RELPATH
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"state_version": 3}
    data["advisory_runs"] = runs
    data["commit_readiness_debts"] = [*data.get("commit_readiness_debts", []), *debts]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _saved(drive: pathlib.Path) -> dict:
    return json.loads((drive / _STATE_RELPATH).read_text(encoding="utf-8"))


def _blocked(repo_key: str, attempt: int, **extra) -> CommitAttemptRecord:
    return CommitAttemptRecord(
        ts=f"2026-10-28T10:0{attempt}:00+00:00", commit_message="modern change", status="blocked",
        repo_key=repo_key, tool_name="commit_reviewed", task_id="task-new", attempt=attempt,
        block_reason="critical_findings", **extra)


def _succeeded(repo_key: str, attempt: int) -> CommitAttemptRecord:
    return CommitAttemptRecord(
        ts=f"2026-10-28T10:0{attempt}:00+00:00", commit_message="modern change", status="succeeded",
        repo_key=repo_key, tool_name="commit_reviewed", task_id="task-new", attempt=attempt)


def _look(drive: pathlib.Path, repo: pathlib.Path) -> dict:
    """The author's preflight look: a settled ``surface=preflight`` ledger record of the checkout."""
    return review_ledger.write_record(drive, {
        "record_id": review_ledger.new_record_id(), "state": "settled", "surface": "preflight",
        "ts": "2026-01-01T00:00:00+00:00", "task_id": "task-look", "verdict": {"aggregate": "PASS"},
        "subject": {"root": str(repo)}, "rows": []})


@pytest.fixture
def ctx(tmp_path):
    from ouroboros import store, store_import
    from ouroboros.tools.registry import ToolContext

    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _: {"status": "completed", "result": "first"})
    return ToolContext(repo_dir=_checkout(tmp_path), drive_root=tmp_path, task_id="caller")


def test_archive_rows_survive_ordinary_attempt_and_commit_saves_exactly_as_stored(tmp_path):
    repo_key = make_repo_key(_checkout(tmp_path))
    drive = tmp_path / "drive"
    live, archived = _live_debts(repo_key), _archived_debts(repo_key)
    # Stored interleaved, as a former install could have left them.
    original = _inject_archive(drive, _archive_rows(repo_key), [live[0], archived[0], live[1], archived[1],
                                                                archived[2], live[2], archived[3]])

    def assert_archive_as_stored():
        saved = _saved(drive)
        # The exact original subtrees, byte for byte: unknown fields, the whole raw_result, more
        # than five evidence lines, empty and missing timestamps, null and odd-typed values.
        assert _compact(saved["advisory_runs"]) == _compact(original["advisory_runs"])
        assert _compact(saved["commit_readiness_debts"][-len(archived):]) == _compact(archived), (
            "kept as stored, after the live")
        assert [row["debt_id"] for row in saved["commit_readiness_debts"]].count("crd-0005") == 1
        return saved

    # The ordinary production writes: reviewed attempts under the locked read-modify-write.
    update_state(drive, lambda state: state.record_attempt(_blocked(
        repo_key, 1, readiness_warnings=["modern warning"],
        critical_findings=[{"item": "tests_affected", "reason": "add a test", "severity": "critical",
                            "verdict": "FAIL"}])))
    saved = assert_archive_as_stored()
    live_rows = {row["debt_id"]: row for row in saved["commit_readiness_debts"][:-len(archived)]}
    # The modern warning's debt takes the next free id: the archived crd-0005 is never reused.
    assert live_rows["crd-0006"]["fingerprint"].startswith("readiness_warning:attempt:")
    # Each half of the archive predicate alone is live debt, verified when no longer observed.
    assert {live_rows[debt_id]["status"] for debt_id in ("crd-0001", "crd-0002")} == {"verified"}
    assert [debt.debt_id for debt in load_state(drive).get_open_commit_readiness_debts(repo_key=repo_key)] == ["crd-0006"]

    update_state(drive, lambda state: state.record_attempt(_succeeded(repo_key, 2)))
    assert {row["status"] for row in assert_archive_as_stored()["commit_readiness_debts"][:-len(archived)]} == {
        "verified"}
    update_state(drive, lambda state: state.on_successful_commit(repo_key=None))
    assert_archive_as_stored()
    assert load_state(drive).get_open_commit_readiness_debts() == []


def test_review_status_names_the_store_it_read_and_existing_readers_return_the_rows(ctx):
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.preflight_review import _handle_review_status
    from ouroboros.tools.query_data import _query_data

    repo_key = make_repo_key(ctx.repo_dir)
    original = _inject_archive(pathlib.Path(ctx.drive_root), _archive_rows(repo_key), _archived_debts(repo_key))

    status = json.loads(_handle_review_status(ctx))
    archive = status["advisory_run_archive"]
    assert archive["path"] == str((pathlib.Path(ctx.drive_root) / _STATE_RELPATH).resolve())
    assert (archive["advisory_runs"], archive["advisory_debts"]) == (4, 4)
    assert "history only" in archive["note"].lower() and "readiness_warning:advisory:" in archive["note"]
    assert "advisory_stale" in archive["note"]
    assert "advisory_runs" not in status and "latest_advisory_hash" not in status
    assert status["latest_advisory_status"] == "none" and status["commit_readiness_debts"] == []

    rows = json.loads(_query_data(ctx, op="json", kind="advisory_review", select="advisory_runs"))
    # query_data answers canonical (sorted-key) JSON; every row and value is there.
    assert json.dumps(rows["data"], sort_keys=True) == json.dumps(original["advisory_runs"], sort_keys=True)
    page = _read_file(ctx, archive["path"])
    assert "ARCHIVE-SENTINEL finding" in page and '"legacy_note"' in page


def test_context_status_and_evidence_read_as_if_the_archive_were_absent(ctx):
    from ouroboros.agent_task_pipeline import build_review_context
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import format_status_section
    from ouroboros.tools.preflight_review import _handle_review_status

    drive, repo = pathlib.Path(ctx.drive_root), pathlib.Path(ctx.repo_dir)
    repo_key = make_repo_key(repo)

    def modern(state):  # owed work the archive must neither hide nor join
        state.record_attempt(_blocked(repo_key, 1, readiness_warnings=["modern warning"], critical_findings=[
            {"item": "tests_affected", "reason": "add a test", "severity": "critical", "verdict": "FAIL"}]))
        state.mark_look_stale("2026-01-01T00:00:00+00:00", reason_ts="2026-10-28T11:00:00+00:00",
                              reason="edit_text mutated the worktree", stale_repo_key=repo_key, stale_task_id="task-a")

    _look(drive, repo)  # the look the marker stales: a marker no look precedes is history
    update_state(drive, modern)

    def surfaces():
        return (build_review_context(types.SimpleNamespace(drive_root=drive, repo_dir=repo)),
                format_status_section(load_state(drive), repo_dir=repo),
                collect_review_evidence(drive, task_id="caller", repo_dir=repo),
                json.loads(_handle_review_status(ctx)))

    context, ledger, evidence, status = surfaces()
    _inject_archive(drive, _archive_rows(repo_key), _archived_debts(repo_key))
    archived_context, archived_ledger, archived_evidence, archived_status = surfaces()

    assert (archived_context, archived_ledger, archived_evidence) == (context, ledger, evidence)
    assert {key: value for key, value in archived_status.items() if key != "advisory_run_archive"} == status
    assert "advisory_run_archive" not in status
    assert "ARCHIVE" not in _compact([archived_context, archived_ledger, archived_evidence,
                                      archived_status["commit_readiness_debts"]])
    for retired in ("recent_advisory_runs", "foreign_advisory_runs", "advisory_status", "snapshot_hash"):
        assert retired not in archived_evidence and retired not in archived_evidence["current_repo"], retired
    # The modern owed work and the look's stale marker still reach every surface.
    assert "open_obligations=1" in context and "stale_marker=" in context and "invalidated_by=task task-a" in context
    assert evidence["current_repo"]["stale_task_id"] == "task-a" and evidence["open_obligations"]
    assert status["stale_from_edit"] is True and status["commit_readiness_debts_count"] == 1


def test_live_debts_keep_their_lifecycle_and_bound_beside_the_archive(tmp_path):
    repo_key = make_repo_key(_checkout(tmp_path))
    drive = tmp_path / "drive"
    archived = _archived_debts(repo_key)
    closed = [{"debt_id": f"crd-{n:04d}", "category": "obligation_repeat", "summary": f"closed {n}",
               "status": "verified", "repo_key": repo_key, "fingerprint": f"obligation_repeat:obl-{n:04d}",
               "source": "review_state", "updated_at": f"2026-07-01T00:{n % 60:02d}:00+00:00"}
              for n in range(100, 160)]
    _inject_archive(drive, [], [*archived, *_live_debts(repo_key), *closed])

    def debts():
        saved = _saved(drive)["commit_readiness_debts"]
        assert _compact(saved[-len(archived):]) == _compact(archived), "the archive is never resolved or pruned"
        return {row["debt_id"]: row for row in saved[:-len(archived)]}

    assert [debt.debt_id for debt in load_state(drive).get_open_commit_readiness_debts(repo_key=repo_key)] == [
        "crd-0001", "crd-0002"]
    # Nothing observes the live debts now: they are verified, and the retention bound
    # prunes only live closed rows — the archive rides outside it.
    update_state(drive, lambda state: state.record_attempt(_blocked(repo_key, 1)))
    live = debts()
    assert len(live) == 50 and {live[debt_id]["status"] for debt_id in ("crd-0001", "crd-0002")} == {"verified"}
    # The stale marker is the one freshness fact: it mints no debt, and the archived
    # ``advisory_stale`` copy is neither reopened nor verified.
    def stale_then_review(state):
        state.mark_look_stale("2026-01-01T00:00:00+00:00", reason_ts="2026-10-28T11:00:00+00:00", reason="an edit",
                              stale_repo_key=repo_key, stale_task_id="task-a")
        state.record_attempt(_blocked(repo_key, 2))

    update_state(drive, stale_then_review)
    assert "crd-0003" not in debts() and load_state(drive).get_open_commit_readiness_debts() == []
    update_state(drive, lambda state: state.on_successful_commit(repo_key=None))
    debts()
    assert load_state(drive).get_open_commit_readiness_debts() == []


def test_a_preflight_look_still_stales_on_edit_beside_fresh_archive_rows(ctx):
    from ouroboros.tools.commit_gate import _invalidate_advisory
    from ouroboros.tools.preflight_review import _handle_review_status

    drive, repo = pathlib.Path(ctx.drive_root), pathlib.Path(ctx.repo_dir)
    original = _inject_archive(drive, _archive_rows(make_repo_key(repo)), [])
    record = _look(drive, repo)

    fresh = json.loads(_handle_review_status(ctx))
    assert (fresh["stale_from_edit"], fresh["latest_advisory_status"]) == (False, "fresh")
    assert fresh["preflight"]["record_id"] == record["record_id"]

    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _invalidate_advisory(ctx, changed_paths=["tracked.py"], source_tool="edit_text")

    stale = json.loads(_handle_review_status(ctx))
    assert (stale["stale_from_edit"], stale["latest_advisory_status"]) == (True, "stale")
    assert stale["preflight"]["marked"] is True and "edit_text mutated the worktree" in stale["stale_reason"]
    assert (stale["stale_task_id"], stale["stale_attribution"]) == ("caller", "this_task")
    # The rows the retired lane would have invalidated stay exactly as written.
    raw = (drive / _STATE_RELPATH).read_text(encoding="utf-8")
    assert '"advisory_runs":' + _compact(original["advisory_runs"]) in raw


def test_a_split_child_names_its_own_archive_while_canonical_change_attempts_stay_visible(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.tools.preflight_review import _handle_review_status
    from ouroboros.tools.registry import ToolContext

    repo = _checkout(tmp_path)
    repo_key = make_repo_key(repo)
    parent, child = tmp_path / "parent-drive", tmp_path / "child-drive"
    # The canonical root holds the child's review_change wave, and an archive of its own.
    update_state(parent, lambda state: state.attempts.append(CommitAttemptRecord(
        ts="2026-10-28T09:00:00+00:00", commit_message="child wave", status="reviewed", repo_key=repo_key,
        tool_name="review_change", task_id="task-child", attempt=1)))
    _inject_archive(parent, _archive_rows(repo_key)[:1], [])
    parent_bytes = (parent / _STATE_RELPATH).read_bytes()
    # The child's execution drive keeps its local gate row and its own, larger archive.
    update_state(child, lambda state: state.attempts.append(CommitAttemptRecord(
        ts="2026-10-28T08:00:00+00:00", commit_message="child-local gate", status="blocked", repo_key=repo_key,
        tool_name="commit_reviewed", task_id="task-child", attempt=1, block_reason="critical_findings")))
    _inject_archive(child, _archive_rows(repo_key), _archived_debts(repo_key))
    child_bytes = (child / _STATE_RELPATH).read_bytes()

    ctx = ToolContext(repo_dir=repo, drive_root=child, task_id="task-child", budget_drive_root=str(parent))
    status = json.loads(_handle_review_status(ctx))
    archive = status["advisory_run_archive"]
    assert archive["path"] == str((child / _STATE_RELPATH).resolve()) != str((parent / _STATE_RELPATH).resolve())
    assert (archive["advisory_runs"], archive["advisory_debts"]) == (4, 4), "the child's store, not the parent's"
    assert sorted((row["tool_name"], row["status"]) for row in status["attempts"]) == [
        ("commit_reviewed", "blocked"), ("review_change", "reviewed")]
    assert "canonical_attempts_unreadable" not in status
    evidence = collect_review_evidence(child, task_id="task-child", repo_dir=repo, canonical_root=parent)
    assert sorted(row["tool_name"] for row in evidence["recent_attempts"]) == ["commit_reviewed", "review_change"]
    assert "ARCHIVE" not in _compact(evidence)
    # Reading moved nothing in either store.
    assert (parent / _STATE_RELPATH).read_bytes() == parent_bytes
    assert (child / _STATE_RELPATH).read_bytes() == child_bytes
