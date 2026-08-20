"""The effective task status a reader sees, and the reconciliation that repairs it.

This module owns the effective-status projection over a durable row plus the live queue,
the reconciliation that durably finalizes an orphaned running task, the outcome contract
recent tasks carry, and the child-task discovery those readers depend on.

The scheduling and cancel tools, the result reader, the wait tools, duplicate admission,
subagent admission and the subagent lifecycle were split verbatim into
``tests/test_task_status_scheduling.py``, ``tests/test_task_status_results.py``,
``tests/test_task_status_wait_tools.py``, ``tests/test_task_status_duplicates.py``,
``tests/test_task_status_subagent_admission.py`` and
``tests/test_task_status_subagent_lifecycle.py``.
"""

import json
import time
from types import SimpleNamespace


def test_recent_tasks_includes_outcome_contract_and_ledger(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.recent_tasks import _handle_recent_tasks

    write_task_result(
        tmp_path,
        "recent1",
        STATUS_COMPLETED,
        result="done",
        task_contract={"schema_version": 1, "objective": "Do work"},
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        artifact_bundle={"schema_version": 1, "status": "ready_no_changes", "artifacts": [], "errors": []},
        verification_ledger={"schema_version": 2, "entries": [{"kind": "objective_outcome"}], "summary": {"entry_count": 1}},
    )

    payload = json.loads(_handle_recent_tasks(SimpleNamespace(drive_root=tmp_path), limit=1))
    record = payload["tasks"][0]

    assert record["outcome_axes"]["execution"]["status"] == "ok"
    assert record["task_contract"]["objective"] == "Do work"
    assert record["artifact_bundle"]["status"] == "ready_no_changes"
    assert record["verification_ledger"]["entry_count"] == 1


def test_effective_status_keeps_workspace_finalization_nonterminal_without_child_drive(tmp_path):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks

    write_task_result(
        tmp_path,
        "workspace1",
        STATUS_COMPLETED,
        workspace_root=str(tmp_path / "workspace"),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        result="worker finished but artifacts are still pending",
    )

    effective = load_effective_task_result(tmp_path, "workspace1")
    waited = wait_for_effective_tasks(tmp_path, ["workspace1"], timeout_sec=0)

    assert effective["status"] == STATUS_RUNNING
    assert effective["child_status"] == STATUS_COMPLETED
    assert effective["artifact_status"] == ARTIFACT_STATUS_FINALIZING
    assert waited["all_terminal"] is False
    assert waited["timed_out"] is True


def test_effective_status_repairs_stale_running_infra_failure_when_queue_empty(tmp_path):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING, ARTIFACT_STATUS_FAILED
    from ouroboros.task_results import STATUS_FAILED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path,
        "providerfail",
        STATUS_RUNNING,
        workspace_root=str(tmp_path / "workspace"),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider error",
        artifact_bundle={
            "status": ARTIFACT_STATUS_FINALIZING,
            "artifacts": [
                {"name": "deck.html", "status": ARTIFACT_STATUS_FINALIZING, "errors": []},
            ],
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "providerfail")

    assert effective["status"] == STATUS_FAILED
    assert effective["status_reconciled_from"] == STATUS_RUNNING
    assert effective["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["artifacts"][0]["status"] == ARTIFACT_STATUS_FAILED
    assert "task ended before artifact finalization" in effective["artifact_bundle"]["artifacts"][0]["errors"]


def test_effective_status_does_not_repair_running_when_queue_snapshot_missing(tmp_path):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path,
        "providerfail",
        STATUS_RUNNING,
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider error",
    )

    effective = load_effective_task_result(tmp_path, "providerfail")

    assert effective["status"] == STATUS_RUNNING
    assert effective["queue_reconciliation_warning"] == "queue snapshot missing or invalid"


def test_effective_status_repairs_orphan_running_after_worker_restart(tmp_path, monkeypatch):
    from ouroboros.headless import ARTIFACT_STATUS_FINALIZING, ARTIFACT_STATUS_FAILED
    from ouroboros.task_results import STATUS_FAILED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    write_task_result(
        tmp_path,
        "cc4db6fa",
        STATUS_RUNNING,
        result="Task is running.",
        ts="2026-05-28T00:00:00+00:00",
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        artifact_bundle={
            "status": ARTIFACT_STATUS_FINALIZING,
            "artifacts": [
                {"name": "presentation.html", "status": ARTIFACT_STATUS_FINALIZING, "errors": []},
            ],
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    events = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": "cc4db6fa"})
    append_jsonl(events, {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})

    effective = load_effective_task_result(tmp_path, "cc4db6fa")

    assert effective["status"] == STATUS_FAILED
    assert effective["status_reconciled_from"] == STATUS_RUNNING
    assert effective["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert effective["reason_code"] == "orphaned_running_after_worker_restart"
    assert "TASK_ORPHAN_RECONCILED" in effective["result"]
    assert effective["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert effective["artifact_bundle"]["artifacts"][0]["status"] == ARTIFACT_STATUS_FAILED
    assert "task interrupted before artifact finalization" in effective["artifact_bundle"]["artifacts"][0]["errors"]


def test_reconcile_durably_finalizes_orphaned_running_task(tmp_path, monkeypatch):
    # C5: the durable sweep persists what the read projection already decides, so
    # a headless/no-UI run that never re-reads the result no longer keeps a zombie
    # `running` record on disk.
    from ouroboros.task_results import (
        STATUS_FAILED,
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    write_task_result(
        tmp_path, "orphan1", STATUS_RUNNING,
        result="Task is running.", ts="2026-05-28T00:00:00+00:00",
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    events = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": "orphan1"})
    append_jsonl(events, {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})

    healed = reconcile_orphaned_running_tasks(tmp_path)

    assert healed == 1
    on_disk = load_task_result(tmp_path, "orphan1")
    assert on_disk["status"] == STATUS_FAILED
    assert on_disk["reason_code"] == "orphaned_running_after_worker_restart"


def test_best_effort_outcome_is_not_a_terminal_failure(tmp_path):
    # ...and the effective-status projection must NOT flip a best_effort
    # completion to failed: it is the documented non-failed, non-clean shelf.
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.task_status import load_effective_task_result

    write_task_result(
        tmp_path, "besteffort1", STATUS_COMPLETED,
        result="Partial best-effort answer.",
        outcome_axes={
            "execution": {"status": "best_effort", "reason_code": "round_limit_reached"},
            "objective": {"status": "not_evaluated"},
        },
    )
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "besteffort1")

    assert effective["status"] == STATUS_COMPLETED  # never reconciled to failed
    assert effective["outcome_axes"]["execution"]["status"] == "best_effort"


def test_reconcile_skips_running_when_queue_snapshot_missing(tmp_path):
    # Liveness gate: a missing/invalid queue snapshot means we cannot prove the
    # task is orphaned, so the sweep must leave the durable `running` untouched.
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    write_task_result(tmp_path, "live1", STATUS_RUNNING, result="still running")

    healed = reconcile_orphaned_running_tasks(tmp_path)

    assert healed == 0
    assert load_task_result(tmp_path, "live1")["status"] == STATUS_RUNNING


def test_find_child_tasks_does_not_regress_terminal_or_running_from_stale_queue_snapshot(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import find_child_tasks, load_effective_task_result

    write_task_result(
        tmp_path,
        "childdone",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="terminal handoff",
    )
    write_task_result(
        tmp_path,
        "childrun",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="still working",
    )
    snapshot = {
        "pending": [
            {"id": "childdone", "task": {"id": "childdone", "parent_task_id": "parent1", "root_task_id": "parent1", "delegation_role": "subagent"}},
            {"id": "childrun", "task": {"id": "childrun", "parent_task_id": "parent1", "root_task_id": "parent1", "delegation_role": "subagent"}},
        ],
        "running": [],
    }
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    effective_done = load_effective_task_result(tmp_path, "childdone")
    effective_running = load_effective_task_result(tmp_path, "childrun")
    children = {row["task_id"]: row for row in find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1")}

    assert effective_done["status"] == STATUS_COMPLETED
    assert effective_running["status"] == STATUS_RUNNING
    assert children["childdone"]["status"] == STATUS_COMPLETED
    assert children["childrun"]["status"] == STATUS_RUNNING


def test_effective_status_preserves_parent_retry_status_over_stale_child_running(tmp_path):
    from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from ouroboros.task_status import load_effective_task_result

    child_drive = tmp_path / "state" / "headless_tasks" / "childretry" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "childretry",
        STATUS_INTERRUPTED,
        child_drive_root=str(child_drive),
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="parent marked retry",
        error="worker interrupted",
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        child_drive,
        "childretry",
        STATUS_RUNNING,
        result="stale child still running",
        error="",
        ts="2026-01-01T00:00:01Z",
    )
    snapshot = {
        "pending": [
            {
                "id": "childretry",
                "task": {
                    "id": "childretry",
                    "parent_task_id": "parent1",
                    "root_task_id": "parent1",
                    "delegation_role": "subagent",
                },
            }
        ],
        "running": [],
    }
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    effective = load_effective_task_result(tmp_path, "childretry")

    assert effective["status"] == STATUS_SCHEDULED
    assert effective["result"] == "parent marked retry"
    assert effective["error"] == "worker interrupted"


def test_find_child_tasks_requires_subagent_role_and_can_exclude_current_task(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import find_child_tasks, format_handoff_message

    write_task_result(
        tmp_path,
        "forgedroot",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="root",
        result="should not be treated as child",
    )
    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="x" * 2000,
        trace_summary="trace" * 500,
    )

    children = find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1")
    excluded = find_child_tasks(tmp_path, parent_task_id="parent1", root_task_id="parent1", exclude_task_id="child1")
    handoff = format_handoff_message(children)

    assert [row["task_id"] for row in children] == ["child1"]
    assert excluded == []
    assert "should not be treated as child" not in handoff
    assert len(handoff) < 1200
    assert "Use get_task_result" in handoff
    assert "result_chars" in handoff
