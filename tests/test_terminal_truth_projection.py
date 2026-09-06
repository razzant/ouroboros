"""Canonical/replica custody regressions for terminal task-result truth."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros.post_task_checkpoint import (
    project_replica_task_result_fields,
    project_root_post_task_checkpoint_fields,
)
from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result


def _seed_atomic_race(tmp_path):
    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "atomic-root"
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "pending_once",
        },
        accounted_upper_bound_usd=1.0,
        cost_final=False,
        cost_with_children_partial=True,
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="accepted child answer",
        producer_exit_code=23,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "running",
        },
        accounted_upper_bound_usd=7.0,
        cost_final=False,
        cost_with_children_partial=True,
        total_rounds=7,
    )
    return data, child, task_id


def _seed_split_result(
    tmp_path,
    *,
    canonical_post_task="completed",
    child_post_task="running",
    canonical_cost_final=True,
):
    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "terminal-root"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="child answer",
        review_status={"status": "fail", "source": "acceptance"},
        trace_summary="child trace",
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "degraded",
            "pass_index": 3,
            "post_task_synthesis": child_post_task,
            "post_task_stop_reason": "stale_child_reason",
        },
        accounted_upper_bound_usd=7.0,
        accounted_upper_bound_usd_with_children=8.0,
        cost_final=not canonical_cost_final,
        cost_with_children_partial=canonical_cost_final,
        non_final_rows=4,
        total_rounds=17,
        prompt_tokens=700,
        completion_tokens=70,
        ts="2026-08-19T00:00:02+00:00",
    )
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        result="canonical placeholder",
        child_drive_root=str(child),
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": canonical_post_task,
            "post_task_stop_reason": f"canonical_{canonical_post_task}_reason",
        },
        accounted_upper_bound_usd=91.0,
        accounted_upper_bound_usd_with_children=99.0,
        cost_final=canonical_cost_final,
        cost_with_children_partial=not canonical_cost_final,
        non_final_rows=0 if canonical_cost_final else 2,
        total_rounds=41,
        prompt_tokens=4100,
        completion_tokens=410,
        ts="2026-08-19T00:00:01+00:00",
    )
    return data, child, task_id


def _assert_terminal_projection(
    result,
    *,
    canonical_post_task,
    canonical_cost_final=True,
):
    checkpoint = result["root_phase_checkpoint"]
    assert checkpoint == {
        "phase": "task_acceptance",
        "status": "degraded",
        "pass_index": 3,
        "post_task_synthesis": canonical_post_task,
        "post_task_stop_reason": f"canonical_{canonical_post_task}_reason",
    }
    assert result["result"] == "child answer"
    assert result["review_status"] == {"status": "fail", "source": "acceptance"}
    assert result["trace_summary"] == "child trace"
    assert result["accounted_upper_bound_usd"] == 91.0
    assert result["accounted_upper_bound_usd_with_children"] == 99.0
    assert result["cost_final"] is canonical_cost_final
    assert result["cost_with_children_partial"] is (not canonical_cost_final)
    assert result["non_final_rows"] == (0 if canonical_cost_final else 2)
    assert result["total_rounds"] == 41
    assert result["prompt_tokens"] == 4100
    assert result["completion_tokens"] == 410


def test_replica_field_projector_is_pure_and_updated_at_is_metadata_only():
    canonical = {
        "updated_at": "2026-08-19T00:00:03+00:00",
        "ts": "canonical-ts",
    }
    replica = {
        "updated_at": "2026-08-19T00:00:02+00:00",
        "ts": "replica-ts",
        "result": "replica answer",
    }
    canonical_before = deepcopy(canonical)
    replica_before = deepcopy(replica)

    projected = project_replica_task_result_fields(canonical, replica)

    assert projected["updated_at"] == canonical["updated_at"]
    assert projected["ts"] == "replica-ts"
    assert projected["result"] == "replica answer"
    assert canonical == canonical_before
    assert replica == replica_before

    projected = project_replica_task_result_fields(
        {"updated_at": "2026-08-19T00:00:01+00:00"},
        {"updated_at": "2026-08-19T00:00:04+00:00"},
    )
    assert projected["updated_at"] == "2026-08-19T00:00:04+00:00"


@pytest.mark.parametrize("stale_status", ["running", "degraded"])
def test_root_patch_keeps_terminal_state_and_accounting_sticky(stale_status):
    canonical = {
        "root_phase_checkpoint": {
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "completed",
            "post_task_stop_reason": "terminal_writer",
        },
        "accounted_upper_bound_usd": 99.0,
        "cost_final": True,
    }
    patch = {
        "root_phase_checkpoint": {"post_task_synthesis": stale_status},
        "accounted_upper_bound_usd": 7.0,
        "cost_final": False,
    }

    projected = project_root_post_task_checkpoint_fields(canonical, patch)
    merged = {**canonical, **projected}

    assert merged == canonical


@pytest.mark.parametrize("canonical_post_task", ["completed", "degraded"])
@pytest.mark.parametrize("child_post_task", ["pending_once", "running"])
@pytest.mark.parametrize("materialize_artifacts", [False, True])
def test_effective_terminal_truth_wins_open_replica_cartesian(
    tmp_path,
    canonical_post_task,
    child_post_task,
    materialize_artifacts,
):
    data, _child, task_id = _seed_split_result(
        tmp_path,
        canonical_post_task=canonical_post_task,
        child_post_task=child_post_task,
    )
    canonical = load_task_result(data, task_id)

    effective = load_effective_task_result(
        data,
        task_id,
        materialize_artifacts=materialize_artifacts,
    )

    _assert_terminal_projection(
        effective,
        canonical_post_task=canonical_post_task,
    )
    assert effective["updated_at"] == canonical["updated_at"]
    # Creation/sort metadata keeps the pre-existing child-overlay behavior.
    assert effective["ts"] == "2026-08-19T00:00:02+00:00"


@pytest.mark.parametrize(
    ("child_has_checkpoint", "child_checkpoint"),
    [(False, None), (True, "legacy-non-dict-checkpoint")],
)
def test_terminal_canonical_acceptance_survives_missing_child_checkpoint(
    tmp_path,
    child_has_checkpoint,
    child_checkpoint,
):
    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "checkpoint-fallback"
    child_fields = {"result": "child answer"}
    if child_has_checkpoint:
        child_fields["root_phase_checkpoint"] = child_checkpoint
    write_task_result(child, task_id, STATUS_COMPLETED, **child_fields)
    canonical_checkpoint = {
        "phase": "task_acceptance",
        "status": "not_required",
        "pass_index": 0,
        "post_task_synthesis": "completed",
        "post_task_stop_reason": "canonical_reason",
    }
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        child_drive_root=str(child),
        root_phase_checkpoint=canonical_checkpoint,
    )

    effective = load_effective_task_result(data, task_id, materialize_artifacts=False)

    assert effective["result"] == "child answer"
    assert effective["root_phase_checkpoint"] == canonical_checkpoint


def test_absent_canonical_stop_reason_preserves_child_reason(tmp_path):
    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "child-stop-reason"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "running",
            "post_task_stop_reason": "child_reason",
        },
    )
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        child_drive_root=str(child),
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "completed",
        },
    )

    effective = load_effective_task_result(data, task_id, materialize_artifacts=False)

    checkpoint = effective["root_phase_checkpoint"]
    assert checkpoint["phase"] == "task_acceptance"
    assert checkpoint["status"] == "pass"
    assert checkpoint["pass_index"] == 1
    assert checkpoint["post_task_synthesis"] == "completed"
    assert checkpoint["post_task_stop_reason"] == "child_reason"


@pytest.mark.parametrize("materialize_artifacts", [False, True])
def test_effective_terminal_partial_accounting_stays_canonical(
    tmp_path,
    materialize_artifacts,
):
    data, _child, task_id = _seed_split_result(
        tmp_path,
        canonical_cost_final=False,
    )

    effective = load_effective_task_result(
        data,
        task_id,
        materialize_artifacts=materialize_artifacts,
    )

    _assert_terminal_projection(
        effective,
        canonical_post_task="completed",
        canonical_cost_final=False,
    )


@pytest.mark.parametrize(
    "canonical_checkpoint",
    [None, {"post_task_synthesis": "pending_once", "status": "not_required"}],
)
def test_open_and_legacy_canonical_results_keep_provisional_child_overlay(
    tmp_path,
    canonical_checkpoint,
):
    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "open-root"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="provisional child answer",
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "running",
        },
        accounted_upper_bound_usd=12.5,
        cost_final=False,
        total_rounds=12,
        ts="2026-08-19T00:00:02+00:00",
    )
    canonical_fields = {
        "result": "canonical placeholder",
        "child_drive_root": str(child),
        "accounted_upper_bound_usd": 1.0,
        "cost_final": True,
        "total_rounds": 1,
        "ts": "2026-08-19T00:00:01+00:00",
    }
    if canonical_checkpoint is not None:
        canonical_fields["root_phase_checkpoint"] = canonical_checkpoint
    write_task_result(data, task_id, STATUS_COMPLETED, **canonical_fields)

    effective = load_effective_task_result(data, task_id, materialize_artifacts=False)

    assert effective["result"] == "provisional child answer"
    assert effective["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    assert effective["root_phase_checkpoint"]["status"] == "pass"
    assert effective["accounted_upper_bound_usd"] == 12.5
    assert effective["cost_final"] is False
    assert effective["total_rounds"] == 12
    assert effective["ts"] == "2026-08-19T00:00:02+00:00"


def test_terminal_truth_is_stable_before_and_after_physical_copyback(
    tmp_path,
    monkeypatch,
):
    from ouroboros.headless import copy_child_task_result
    import ouroboros.task_results as task_results

    data, child, task_id = _seed_split_result(
        tmp_path,
        canonical_cost_final=False,
    )

    before = load_effective_task_result(data, task_id, materialize_artifacts=False)
    _assert_terminal_projection(
        before,
        canonical_post_task="completed",
        canonical_cost_final=False,
    )

    copyback_updated_at = "2099-08-19T00:01:00+00:00"
    monkeypatch.setattr(task_results, "utc_now_iso", lambda: copyback_updated_at)
    copied = copy_child_task_result(data, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    assert (child / "task_results" / f"{task_id}.json").is_file()
    retained_child = load_task_result(child, task_id)
    assert retained_child["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    _assert_terminal_projection(
        copied,
        canonical_post_task="completed",
        canonical_cost_final=False,
    )
    assert copied["ts"] == "2026-08-19T00:00:02+00:00"
    assert copied["updated_at"] == copyback_updated_at
    after = load_effective_task_result(data, task_id, materialize_artifacts=False)
    _assert_terminal_projection(
        after,
        canonical_post_task="completed",
        canonical_cost_final=False,
    )
    assert after["root_phase_checkpoint"] == before["root_phase_checkpoint"]
    assert after["updated_at"] == copyback_updated_at


def test_delayed_copyback_projects_against_terminal_current_record(
    tmp_path,
    monkeypatch,
):
    """A copyback paused after its early read must reduce the later terminal record."""
    import ouroboros.headless as headless

    data, child, task_id = _seed_atomic_race(tmp_path)
    entered_write = threading.Event()
    release_write = threading.Event()
    real_write = headless.write_task_result
    outcome = {}

    def delayed_write(*args, **kwargs):
        entered_write.set()
        assert release_write.wait(5)
        return real_write(*args, **kwargs)

    def copyback():
        try:
            outcome["result"] = headless.copy_child_task_result(
                data, {"id": task_id, "drive_root": str(child)}
            )
        except BaseException as exc:  # surfaced in the parent test thread
            outcome["error"] = exc

    monkeypatch.setattr(headless, "write_task_result", delayed_write)
    thread = threading.Thread(target=copyback, name="delayed-copyback")
    thread.start()
    assert entered_write.wait(5)

    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "completed",
            "post_task_stop_reason": "terminal_writer",
        },
        accounted_upper_bound_usd=99.0,
        cost_final=True,
        cost_with_children_partial=False,
        total_rounds=99,
    )
    release_write.set()
    thread.join(5)

    assert not thread.is_alive()
    assert "error" not in outcome
    stored = load_task_result(data, task_id)
    assert outcome["result"] == stored
    assert stored["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "completed",
        "post_task_stop_reason": "terminal_writer",
    }
    assert stored["producer_exit_code"] == 23
    assert stored["accounted_upper_bound_usd"] == 99.0
    assert stored["cost_final"] is True
    assert stored["cost_with_children_partial"] is False
    assert stored["total_rounds"] == 99


def test_delayed_terminal_checkpoint_merges_current_acceptance_and_stays_terminal(
    tmp_path,
    monkeypatch,
):
    """A terminal writer paused after its read keeps a racing child acceptance."""
    import ouroboros.headless as headless
    import ouroboros.post_task_checkpoint as checkpoint
    import ouroboros.usage_accounting as usage_accounting
    import supervisor.state as supervisor_state

    data, child, task_id = _seed_atomic_race(tmp_path)
    entered_write = threading.Event()
    release_write = threading.Event()
    real_write = checkpoint.write_task_result
    outcome = {}

    def delayed_write(*args, **kwargs):
        entered_write.set()
        assert release_write.wait(5)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "write_task_result", delayed_write)
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *args, **kwargs: {
            "accounted_upper_bound_usd": 99.0,
            "cost_final": True,
            "cost_with_children_partial": False,
            "total_rounds": 99,
        },
    )
    monkeypatch.setattr(
        usage_accounting,
        "usage_breakdown",
        lambda *args, **kwargs: {"accounted_usd": 99.0, "cost_final": True},
    )
    task = {
        "id": task_id,
        "root_task_id": task_id,
        "budget_drive_root": str(data),
        "status": STATUS_COMPLETED,
    }

    def finalize_checkpoint():
        try:
            checkpoint.set_root_post_task_checkpoint(
                SimpleNamespace(drive_root=data), task, "completed"
            )
        except BaseException as exc:  # surfaced in the parent test thread
            outcome["error"] = exc

    thread = threading.Thread(
        target=finalize_checkpoint,
        name="delayed-terminal-checkpoint",
    )
    thread.start()
    assert entered_write.wait(5)

    copied = headless.copy_child_task_result(
        data, {"id": task_id, "drive_root": str(child)}
    )
    assert copied["root_phase_checkpoint"]["status"] == "pass"
    release_write.set()
    thread.join(5)

    assert not thread.is_alive()
    assert "error" not in outcome
    stored = load_task_result(data, task_id)
    assert stored["root_phase_checkpoint"]["status"] == "pass"
    assert stored["root_phase_checkpoint"]["pass_index"] == 1
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
    assert stored["producer_exit_code"] == 23
    assert stored["accounted_upper_bound_usd"] == 99.0
    assert stored["cost_final"] is True
    assert stored["cost_with_children_partial"] is False

    # A stale/open root writer cannot reopen the terminal checkpoint or replace
    # the snapshot with its provisional cost markers.
    checkpoint.set_root_post_task_checkpoint(
        SimpleNamespace(drive_root=data), task, "running"
    )
    after_open = load_task_result(data, task_id)
    assert after_open["root_phase_checkpoint"] == stored["root_phase_checkpoint"]
    assert after_open["accounted_upper_bound_usd"] == 99.0
    assert after_open["cost_final"] is True
    assert after_open["cost_with_children_partial"] is False


def test_terminal_checkpoint_uses_current_lifecycle_status_before_regression_guard(
    tmp_path,
    monkeypatch,
):
    """A stale pre-lock status must not abort the terminal checkpoint projection."""
    import ouroboros.headless as headless
    import ouroboros.post_task_checkpoint as checkpoint
    import ouroboros.usage_accounting as usage_accounting
    import supervisor.state as supervisor_state

    data = tmp_path / "data"
    child = tmp_path / "child"
    data.mkdir()
    child.mkdir()
    task_id = "status-race-root"
    write_task_result(
        data,
        task_id,
        "scheduled",
        root_task_id=task_id,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "running",
        },
        accounted_upper_bound_usd=1.0,
        cost_final=False,
        cost_with_children_partial=True,
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="accepted child answer",
        producer_exit_code=23,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "running",
        },
        accounted_upper_bound_usd=7.0,
        cost_final=False,
        cost_with_children_partial=True,
        total_rounds=7,
    )

    stale_read_complete = threading.Event()
    release_stale_reader = threading.Event()
    real_load = checkpoint.load_task_result
    outcome = {}

    def delayed_load(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        if threading.current_thread().name == "stale-status-checkpoint":
            stale_read_complete.set()
            assert release_stale_reader.wait(5)
        return loaded

    monkeypatch.setattr(checkpoint, "load_task_result", delayed_load)
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *args, **kwargs: {
            "accounted_upper_bound_usd": 99.0,
            "cost_final": True,
            "cost_with_children_partial": False,
            "total_rounds": 99,
        },
    )
    monkeypatch.setattr(
        usage_accounting,
        "usage_breakdown",
        lambda *args, **kwargs: {"accounted_usd": 99.0, "cost_final": True},
    )

    def finalize_checkpoint():
        try:
            checkpoint.set_root_post_task_checkpoint(
                SimpleNamespace(drive_root=data),
                {
                    "id": task_id,
                    "root_task_id": task_id,
                    "budget_drive_root": str(data),
                    "status": "scheduled",
                },
                "completed",
            )
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(
        target=finalize_checkpoint,
        name="stale-status-checkpoint",
    )
    thread.start()
    assert stale_read_complete.wait(5)

    copied = headless.copy_child_task_result(
        data,
        {"id": task_id, "drive_root": str(child)},
    )
    assert copied["status"] == STATUS_COMPLETED
    assert copied["root_phase_checkpoint"]["status"] == "pass"
    assert copied["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    release_stale_reader.set()
    thread.join(5)

    assert not thread.is_alive()
    assert "error" not in outcome
    stored = load_task_result(data, task_id)
    assert stored["status"] == STATUS_COMPLETED
    assert stored["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "completed",
    }
    assert stored["producer_exit_code"] == 23
    assert stored["accounted_upper_bound_usd"] == 99.0
    assert stored["cost_final"] is True
    assert stored["cost_with_children_partial"] is False
    assert stored["total_rounds"] == 99
    events = [
        json.loads(line)
        for line in (data / "logs" / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    finalized = events[-1]
    assert finalized["type"] == "task_cost_finalized"
    assert finalized["post_task_status"] == "completed"
    assert finalized["accounted_upper_bound_usd"] == 99.0
    assert finalized["cost_final"] is True
    assert finalized["total_rounds"] == 99


def test_startup_recovery_merges_child_acceptance_after_stale_scan(
    tmp_path,
    monkeypatch,
):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.headless as headless
    import ouroboros.task_results as task_results
    import ouroboros.usage_accounting as usage_accounting
    import supervisor.state as supervisor_state

    data, child, task_id = _seed_atomic_race(tmp_path)
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        root_task_id=task_id,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "running",
        },
    )
    scan_complete = threading.Event()
    release_scan = threading.Event()
    real_list = task_results.list_task_results
    outcome = {}

    def delayed_list(root):
        rows = real_list(root)
        scan_complete.set()
        assert release_scan.wait(5)
        return rows

    monkeypatch.setattr(task_results, "list_task_results", delayed_list)
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *args, **kwargs: {"accounted_upper_bound_usd": 99.0, "cost_final": True},
    )
    monkeypatch.setattr(
        usage_accounting,
        "usage_breakdown",
        lambda *args, **kwargs: {"accounted_usd": 99.0, "cost_final": True},
    )

    def recover():
        try:
            outcome["count"] = pipeline.recover_pending_root_post_task_synthesis(
                data, tmp_path / "repo"
            )
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=recover, name="delayed-startup-recovery")
    thread.start()
    assert scan_complete.wait(5)
    copied = headless.copy_child_task_result(
        data, {"id": task_id, "drive_root": str(child)}
    )
    assert copied["root_phase_checkpoint"]["status"] == "pass"
    release_scan.set()
    thread.join(5)

    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["count"] == 1
    stored = load_task_result(data, task_id)
    assert stored["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "degraded",
        "post_task_stop_reason": "restart_indeterminate_running",
    }
    assert stored["accounted_upper_bound_usd"] == 99.0
    assert stored["cost_final"] is True


def test_finalized_event_projects_actual_stored_terminal_truth(tmp_path, monkeypatch):
    import ouroboros.post_task_checkpoint as checkpoint
    import ouroboros.usage_accounting as usage_accounting
    import supervisor.state as supervisor_state

    data = tmp_path / "data"
    data.mkdir()
    task_id = "event-truth"
    canonical_checkpoint = {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "completed",
        "post_task_stop_reason": "canonical_terminal",
    }
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        root_task_id=task_id,
        root_phase_checkpoint=canonical_checkpoint,
        accounted_upper_bound_usd=99.0,
        accounted_upper_bound_usd_with_children=99.0,
        cost_final=True,
        cost_with_children_partial=False,
        total_rounds=99,
    )
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *args, **kwargs: {
            "accounted_upper_bound_usd": 7.0,
            "cost_final": True,
            "total_rounds": 7,
        },
    )
    monkeypatch.setattr(
        usage_accounting,
        "usage_breakdown",
        lambda *args, **kwargs: {"accounted_usd": 7.0, "cost_final": True},
    )

    persisted = checkpoint.set_root_post_task_checkpoint(
        SimpleNamespace(drive_root=data),
        {"id": task_id, "root_task_id": task_id, "status": STATUS_COMPLETED},
        "degraded",
        stop_reason="stale_degraded_attempt",
    )

    stored = load_task_result(data, task_id)
    assert persisted == stored
    assert stored["root_phase_checkpoint"] == canonical_checkpoint
    assert stored["accounted_upper_bound_usd"] == 99.0
    events = [
        json.loads(line)
        for line in (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    finalized = events[-1]
    assert finalized["type"] == "task_cost_finalized"
    assert finalized["post_task_status"] == "completed"
    assert finalized["accounted_upper_bound_usd"] == 99.0
    assert finalized["accounted_upper_bound_usd_with_children"] == 99.0
    assert finalized["cost_final"] is True
    assert finalized["total_rounds"] == 99


def test_failed_checkpoint_write_emits_no_finalized_event(tmp_path, monkeypatch):
    import ouroboros.post_task_checkpoint as checkpoint

    data = tmp_path / "data"
    data.mkdir()
    task_id = "failed-write"
    write_task_result(
        data,
        task_id,
        STATUS_COMPLETED,
        root_task_id=task_id,
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )
    appended = []
    monkeypatch.setattr(
        checkpoint,
        "write_task_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    monkeypatch.setattr(checkpoint, "append_jsonl", lambda *args: appended.append(args))

    persisted = checkpoint.set_root_post_task_checkpoint(
        SimpleNamespace(drive_root=data),
        {"id": task_id, "root_task_id": task_id, "status": STATUS_COMPLETED},
        "completed",
    )

    assert persisted is None
    assert appended == []


def test_startup_recovery_surfaces_failed_running_checkpoint_persistence(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.post_task_checkpoint as checkpoint

    task_id = "failed-recovery-write"
    write_task_result(
        tmp_path,
        task_id,
        STATUS_COMPLETED,
        root_task_id=task_id,
        root_phase_checkpoint={
            "phase": "task_acceptance", "status": "pass", "post_task_synthesis": "running"
        },
    )

    monkeypatch.setattr(checkpoint, "write_task_result", Mock(side_effect=TimeoutError))

    with pytest.raises(RuntimeError, match="did not persist a terminal checkpoint"):
        pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path / "repo")

    stored = load_task_result(tmp_path, task_id)
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    assert "post_task_stop_reason" not in stored["root_phase_checkpoint"]


def _write_history_rows(data, task_id):
    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-19T00:00:03+00:00",
            "type": "send_message",
            "task_id": task_id,
            "is_progress": True,
            "direction": "out",
            "chat_id": 1,
            "user_id": 1,
            "text": "delivered answer",
            "content": "delivered answer",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text("", encoding="utf-8")


def test_history_and_task_detail_project_terminal_canonical_accounting(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import api_task_get

    data, _child, task_id = _seed_split_result(tmp_path)
    _write_history_rows(data, task_id)

    detail_request = SimpleNamespace(
        path_params={"task_id": task_id},
        app=SimpleNamespace(state=SimpleNamespace(drive_root=data)),
    )
    detail = json.loads(asyncio.run(api_task_get(detail_request)).body.decode("utf-8"))

    # ProgramBench polls this exact task-detail surface before deciding whether
    # cost is still partial; the stale child must not restart its bounded wait.
    assert detail["status"] == STATUS_COMPLETED
    assert detail["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
    assert detail["root_phase_checkpoint"]["post_task_stop_reason"] == (
        "canonical_completed_reason"
    )
    assert detail["accounted_upper_bound_usd_with_children"] == 99.0
    assert detail["cost_final"] is True
    assert detail["cost_with_children_partial"] is False

    endpoint = make_chat_history_endpoint(data)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "20"})))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    progress = next(
        row
        for row in messages
        if row.get("is_progress") and row.get("task_id") == task_id
    )
    assert progress["task_terminal_status"] == STATUS_COMPLETED
    assert progress.get("task_phase") != "finalizing"
    assert progress["accounted_upper_bound_usd_with_children"] == 99.0
    assert progress["cost_final"] is True
    assert progress["cost_with_children_partial"] is False
