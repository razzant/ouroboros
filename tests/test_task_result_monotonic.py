"""Monotonic lifecycle guard for write_task_result (v6.7.0-rc.1; cancel redesign
phase A, 2026-08-11).

Pins the "ghost subagent" / status-corruption protections:
- a stale scheduled/running mirror cannot overwrite a terminal or a legacy latch
- a terminal status is sticky against a *different* terminal status — including
  cancelled-over-completed: natural completion WINS (owner 4=A), there is no
  explicit-cancellation override any more
- a LEGACY ``cancel_requested`` latch (pre-intent files) yields to ANY terminal,
  including a racing ``completed``
- normal forward progress and same-status enrichment are unaffected
"""

import pytest

from ouroboros import task_results as tr


@pytest.fixture()
def drive(tmp_path):
    return tmp_path


def _status(drive, tid):
    return tr.load_task_result(drive, tid)["status"]


def test_terminal_not_regressed_by_running(drive):
    tr.write_task_result(drive, "t", tr.STATUS_CANCELLED, result="cancelled")
    tr.write_task_result(drive, "t", tr.STATUS_RUNNING, result="stale mirror")
    assert _status(drive, "t") == tr.STATUS_CANCELLED


def test_terminal_is_sticky_against_other_terminal(drive):
    tr.write_task_result(drive, "t", tr.STATUS_CANCELLED)
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED, result="late completion")
    assert _status(drive, "t") == tr.STATUS_CANCELLED

    tr.write_task_result(drive, "u", tr.STATUS_COMPLETED)
    tr.write_task_result(drive, "u", tr.STATUS_FAILED)
    assert _status(drive, "u") == tr.STATUS_COMPLETED


def test_terminal_sticky_against_unknown_status(drive):
    # A typo / future / unranked status must NOT overwrite a terminal one.
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED, result="done")
    tr.write_task_result(drive, "t", "weird_unranked_status")
    assert _status(drive, "t") == tr.STATUS_COMPLETED


def test_same_terminal_status_enrichment_allowed(drive):
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED, result="first")
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED, result="enriched", trace_summary="trace")
    data = tr.load_task_result(drive, "t")
    assert data["status"] == tr.STATUS_COMPLETED
    assert data["result"] == "enriched"
    assert data["trace_summary"] == "trace"


def test_replica_projector_keeps_terminal_lifecycle_guard(drive):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    tr.write_task_result(drive, "t", tr.STATUS_CANCELLED, result="cancelled")
    result = tr.write_task_result(
        drive,
        "t",
        tr.STATUS_COMPLETED,
        _field_projector=project_replica_task_result_fields,
        result="late child",
    )
    assert result["status"] == tr.STATUS_CANCELLED
    assert result["result"] == "cancelled"


def test_writer_selects_current_review_before_locked_projector(drive):
    panel = {"surface": "task_acceptance", "panel_id": "p", "panel_index": 0,
             "task_attempt": 1, "publication_revision": 2, "superseded": True,
             "aggregate_signal": "FAIL", "applied_source_ref": {"path": "original"}}
    tr.write_task_result(drive, "review", tr.STATUS_COMPLETED, review_projection={"panels": [panel]})
    stale = {**panel, "superseded": False, "aggregate_signal": "PASS",
             "applied_source_ref": {"path": "stale"}}
    calls = []

    def project(current, fields):
        assert (drive / "task_results" / "review.json.lock").exists()
        assert fields["review_projection"] == current["review_projection"] == {"panels": [panel]}
        calls.append(True)
        return {**fields, "status": current["status"], "review_projection": {
            "panels": [{**fields["review_projection"]["panels"][0],
                        "applied_source_ref": {"path": "promoted"}}]}}

    saved = tr.write_task_result(drive, "review", tr.STATUS_RUNNING,
                                 review_projection={"panels": [stale]}, _field_projector=project)
    assert calls == [True]
    assert saved["status"] == tr.STATUS_COMPLETED
    assert saved["review_projection"] == {"panels": [{**panel, "applied_source_ref": {"path": "promoted"}}]}


def test_cancel_requested_blocks_running_but_allows_cancelled(drive):
    tr.write_task_result(drive, "t", tr.STATUS_CANCEL_REQUESTED)
    tr.write_task_result(drive, "t", tr.STATUS_RUNNING)
    assert _status(drive, "t") == tr.STATUS_CANCEL_REQUESTED
    tr.write_task_result(drive, "t", tr.STATUS_CANCELLED, result="done")
    assert _status(drive, "t") == tr.STATUS_CANCELLED


def test_legacy_cancel_latch_yields_to_natural_completion(drive):
    # Phase A (owner 4=A): natural completion WINS. A worker finishing after a
    # LEGACY cancel latch flips the task to "completed" and keeps its result —
    # cancel means "stop spending", never "discard the result".
    tr.write_task_result(drive, "t", tr.STATUS_CANCEL_REQUESTED)
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED, result="late success")
    assert _status(drive, "t") == tr.STATUS_COMPLETED
    # ...and once completed, a late cancelled write is refused (sticky terminal).
    tr.write_task_result(drive, "t", tr.STATUS_CANCELLED)
    assert _status(drive, "t") == tr.STATUS_COMPLETED
    # A legacy latch still advances to the real teardown outcome when no natural
    # completion raced it.
    tr.write_task_result(drive, "u", tr.STATUS_CANCEL_REQUESTED)
    tr.write_task_result(drive, "u", tr.STATUS_CANCELLED)
    assert _status(drive, "u") == tr.STATUS_CANCELLED


def test_completed_result_survives_a_late_cancellation(drive):
    # Phase A: the explicit-cancellation completed-overwrite is REMOVED. A late
    # cancel must neither flip the status nor strip the completed payload
    # (discarding a kept result is a separate explicit parent action).
    tr.write_task_result(
        drive,
        "won-race",
        tr.STATUS_COMPLETED,
        result="real child result",
        final_answer="real answer",
        trace_summary="real trace",
        artifacts=[{"name": "real.txt"}],
        artifact_bundle={"status": "ready"},
        outcome_axes={"objective": {"status": "solved"}},
        review_evidence={"verdict": "PASS"},
        root_phase_checkpoint={"post_task_synthesis": "completed"},
        cost_usd=1.25,
        parent_task_id="parent",
    )
    tr.write_task_result(
        drive, "won-race", tr.STATUS_CANCELLED, result="owner cancelled",
    )
    kept = tr.load_task_result(drive, "won-race")
    assert kept["status"] == tr.STATUS_COMPLETED
    assert kept["result"] == "real child result"
    assert kept["final_answer"] == "real answer"
    assert kept["artifacts"] == [{"name": "real.txt"}]
    # ABI-3 fix-round-2: stored under the honest name only (the write above
    # used the legacy kwarg — deprecated-wins honored it, then stripped it).
    assert kept["accounted_upper_bound_usd"] == 1.25
    assert "cost_usd" not in kept

    tr.write_task_result(drive, "failed", tr.STATUS_FAILED, result="real failure")
    tr.write_task_result(drive, "failed", tr.STATUS_CANCELLED)
    assert _status(drive, "failed") == tr.STATUS_FAILED


def test_normal_forward_progress_and_retry(drive):
    tr.write_task_result(drive, "t", tr.STATUS_SCHEDULED)
    tr.write_task_result(drive, "t", tr.STATUS_RUNNING)
    tr.write_task_result(drive, "t", tr.STATUS_INTERRUPTED)  # pre-requeue
    tr.write_task_result(drive, "t", tr.STATUS_RUNNING)      # retry
    tr.write_task_result(drive, "t", tr.STATUS_COMPLETED)
    assert _status(drive, "t") == tr.STATUS_COMPLETED


def test_updated_at_is_written(drive):
    tr.write_task_result(drive, "t", tr.STATUS_SCHEDULED)
    assert tr.load_task_result(drive, "t").get("updated_at")


def test_lock_timeout_never_falls_back_to_an_unlocked_authoritative_write(
    drive, monkeypatch,
):
    """A timeout preserves prior lifecycle truth instead of accepting stale state."""
    tr.write_task_result(drive, "locked", tr.STATUS_RUNNING, result="working")

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("held by concurrent writer")

    monkeypatch.setattr(tr, "update_json_locked", _timeout)
    with pytest.raises(TimeoutError, match="concurrent writer"):
        tr.write_task_result(drive, "locked", tr.STATUS_COMPLETED, result="done")

    stored = tr.load_task_result(drive, "locked")
    assert stored["status"] == tr.STATUS_RUNNING
    assert stored["result"] == "working"


def test_llm_project_name_uses_cleaned_model_title():
    """v6.40: the real LLM naming path returns the model's title run through clean_model_title
    (lexical clean — strips wrapping quotes, P5), not the raw model string."""
    from ouroboros import project_naming

    class _FakeClient:
        def chat(self, **kw):
            return ({"content": '"Cyber Racing Arena"'}, {"cost": 0.0})

    name = project_naming.llm_project_name(
        "build me a top-down neon racing game", llm_client=_FakeClient(),
    )
    assert name == "Cyber Racing Arena", f"expected cleaned title, got {name!r}"


def test_read_paths_do_not_create_task_results_dir(tmp_path):
    """v6.40.0: a READ/LIST scan of a never-provisioned root must NOT materialise the
    ``task_results`` directory (regression: an unguarded scan created stray dirs)."""
    root = tmp_path / "never_provisioned"
    assert tr.list_task_results(root) == []
    assert tr.load_task_result(root, "missing") is None
    assert not (root / "task_results").exists(), "read must not create the dir"
    # WRITE still provisions it.
    tr.write_task_result(root, "t", tr.STATUS_SCHEDULED)
    assert (root / "task_results").is_dir()


def test_read_with_stub_root_leaks_no_cwd_dir(tmp_path, monkeypatch):
    """The exact pollution repro: a MagicMock-derived root (``MagicMock/mock``) reaching a
    READ scan must not create a ``MagicMock`` tree in the cwd."""
    import pathlib
    from unittest.mock import MagicMock

    monkeypatch.chdir(tmp_path)
    stub_root = pathlib.Path(MagicMock()).parent  # == Path("MagicMock/mock")
    assert tr.list_task_results(stub_root) == []
    assert tr.load_task_result(stub_root, "x") is None
    leaked = [p.name for p in pathlib.Path(".").iterdir() if "MagicMock" in p.name]
    assert leaked == [], f"read scan leaked mock-named paths: {leaked}"
