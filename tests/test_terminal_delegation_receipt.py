"""Terminal delegation receipt replay and canonical/replica custody."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.post_task_checkpoint import project_replica_task_result_fields
from ouroboros.task_results import STATUS_COMPLETED, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.task_result_schema import stamp_task_result_schema  # v7 ABI-2: unstamped rows are quarantined


def _execution_evidence(
    started: int,
    settled: int,
    cost: float | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "delegated_runs_started": started,
        "delegated_runs_settled": settled,
        "delegated_runs_succeeded": settled,
        "delegated_runs_failed": 0,
        "subscription_cost_usd": cost,
        **extra,
    }


def _delegation_envelope(
    started: int,
    settled: int,
    cost: float | None,
    substrate: str,
    **evidence_extra: Any,
) -> dict[str, Any]:
    return {
        "executor_route": "claude=opus",
        "actual_substrate": substrate,
        "native_contribution": "unknown",
        "execution_evidence": _execution_evidence(
            started,
            settled,
            cost,
            **evidence_extra,
        ),
    }


def _completed_task_result(task_id: str, cost: float) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "completed",
        "parent_task_id": "root",
        "root_task_id": "root",
        "delegation_role": "subagent",
        "executor_route": "claude=opus",
        "actual_substrate": "harness_used",
        "subagent_envelope": _delegation_envelope(
            1,
            1,
            cost,
            "harness_used",
        ),
    }


def test_history_rehydrates_receipt_on_latest_terminal_progress_consumer(tmp_path):
    """Summary, stale, absent, and retry directions share one replay seam."""
    logs = tmp_path / "logs"
    logs.mkdir()
    progress_rows = [
        {
            "task_id": "summary",
            "content": "older route only",
            "ts": "2026-09-01T17:00:00Z",
        },
        {
            "task_id": "summary",
            "content": "latest route only",
            "ts": "2026-09-01T17:00:30Z",
        },
        {
            "task_id": "stale",
            "content": "stale receipt",
            "ts": "2026-09-01T17:01:00Z",
            "actual_substrate": "harness_attempted",
            "execution_evidence": _execution_evidence(1, 0, None),
        },
        {
            "task_id": "absent",
            "content": "surviving receipt",
            "ts": "2026-09-01T17:02:00Z",
            "actual_substrate": "harness_used",
            "execution_evidence": _execution_evidence(1, 1, 1.25),
        },
        {
            "task_id": "original",
            "content": "retry route only",
            "ts": "2026-09-01T17:03:00Z",
        },
    ]
    for row in progress_rows:
        row.update({"is_progress": True, "executor_route": "claude=opus"})
    (logs / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in progress_rows) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-09-01T17:04:00Z",
                "direction": "system",
                "chat_id": 1,
                "type": "task_summary",
                "task_id": "summary",
                "text": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = tmp_path / "task_results"
    results.mkdir()
    for task_id, cost in (("summary", 0.0), ("stale", 1.25), ("retry", 2.5)):
        (results / f"{task_id}.json").write_text(
            json.dumps(stamp_task_result_schema(_completed_task_result(task_id, cost))),
            encoding="utf-8",
        )
    absent = _completed_task_result("absent", 1.25)
    absent.pop("actual_substrate")
    absent["subagent_envelope"] = {"executor_route": "claude=opus"}
    (results / "absent.json").write_text(json.dumps(stamp_task_result_schema(absent)), encoding="utf-8")
    (results / "original.json").write_text(
        json.dumps(
            stamp_task_result_schema(
                {
                    "task_id": "original",
                    "status": "interrupted",
                    "retry_task_id": "retry",
                }
            )
        ),
        encoding="utf-8",
    )

    response = asyncio.run(
        make_chat_history_endpoint(tmp_path)(
            SimpleNamespace(query_params={"n_human": "10", "n_progress": "20"})
        )
    )
    progress = [
        row for row in json.loads(response.body)["messages"] if row.get("is_progress")
    ]
    latest = {
        task_id: max(
            (row for row in progress if row.get("task_id") == task_id),
            key=lambda row: row["ts"],
        )
        for task_id in ("summary", "stale", "absent", "original")
    }
    for row in latest.values():
        assert row["task_terminal_status"] == "completed"
        assert row["actual_substrate"] == "harness_used"
    assert latest["summary"]["execution_evidence"] == _execution_evidence(1, 1, 0.0)
    assert latest["stale"]["execution_evidence"] == _execution_evidence(1, 1, 1.25)
    assert latest["absent"]["execution_evidence"] == _execution_evidence(1, 1, 1.25)
    assert latest["original"]["execution_evidence"] == _execution_evidence(1, 1, 2.5)

    summary_rows = [row for row in progress if row.get("task_id") == "summary"]
    assert len(summary_rows) == 2
    older_summary = min(summary_rows, key=lambda row: row["ts"])
    assert older_summary["text"] == "older route only"
    assert latest["summary"]["text"] == "latest route only"
    assert "execution_evidence" not in older_summary


def test_replica_projector_orders_receipts_and_preserves_disclosure_authority():
    first = _delegation_envelope(1, 1, 1.25, "harness_used")
    projected = project_replica_task_result_fields(
        {"subagent_envelope": {"executor_route": "claude=opus"}},
        {"actual_substrate": "harness_used", "subagent_envelope": first},
    )
    assert projected["subagent_envelope"] == first

    positive_replica = {
        "actual_substrate": "harness_used",
        "subagent_envelope": first,
    }
    for canonical_envelope in (
        _delegation_envelope(0, 0, None, "native_only"),
        {
            "executor_route": "claude=opus",
            "execution_evidence": {"evidence_read_failed": True},
        },
    ):
        projected = project_replica_task_result_fields(
            {"subagent_envelope": canonical_envelope},
            positive_replica,
        )
        assert projected["actual_substrate"] == "harness_used"
        assert (
            projected["subagent_envelope"]["execution_evidence"]
            == first["execution_evidence"]
        )

    enriched = _delegation_envelope(
        1,
        1,
        3.75,
        "harness_used",
        applied_access_profiles=["workspace_write"],
    )
    replica = {
        "actual_substrate": "harness_attempted",
        "delegated_runs_settled": 0,
        "subagent_envelope": {
            **_delegation_envelope(1, 1, None, "harness_attempted"),
            "marker": "keep-replica",
        },
    }
    canonical = {
        "actual_substrate": "harness_used",
        "subagent_envelope": enriched,
        "delegated_runs_unreconciled": [],
        "delegate_terminal_reconciliation": {"trigger": "boot_backfill"},
    }
    projected = project_replica_task_result_fields(
        canonical,
        {
            **replica,
            "delegated_runs_unreconciled": ["stale"],
            "delegate_terminal_reconciliation": {"trigger": "terminal_write"},
        },
    )
    evidence = projected["subagent_envelope"]["execution_evidence"]
    assert evidence["subscription_cost_usd"] == 3.75
    assert evidence["applied_access_profiles"] == ["workspace_write"]
    assert projected["actual_substrate"] == "harness_used"
    assert projected["delegated_runs_settled"] == 0
    assert projected["subagent_envelope"]["marker"] == "keep-replica"
    assert "delegated_runs_unreconciled" not in projected
    assert "delegate_terminal_reconciliation" not in projected

    first_disclosure = {
        "delegated_runs_unreconciled": ["first"],
        "delegate_terminal_reconciliation": {"trigger": "terminal_write"},
    }
    assert project_replica_task_result_fields({}, first_disclosure) == first_disclosure
    assert project_replica_task_result_fields(canonical, {})["subagent_envelope"] == enriched


def test_receipt_order_matches_effective_read_and_physical_copyback(tmp_path):
    from ouroboros.headless import copy_child_task_result

    for case in ("heal", "first"):
        data = tmp_path / case / "data"
        child = tmp_path / case / "child"
        task_id = f"{case}-task"
        data.mkdir(parents=True)
        child.mkdir()
        child_receipt = _delegation_envelope(
            2 if case == "heal" else 1,
            1,
            1.0,
            "harness_attempted" if case == "heal" else "harness_used",
        )
        canonical_receipt = (
            _delegation_envelope(2, 2, 2.5, "harness_used")
            if case == "heal"
            else {"executor_route": "claude=opus"}
        )
        write_task_result(
            child,
            task_id,
            STATUS_COMPLETED,
            actual_substrate=child_receipt["actual_substrate"],
            subagent_envelope=child_receipt,
            delegated_runs_unreconciled=["stale"],
            delegate_terminal_reconciliation={"trigger": "terminal_write"},
        )
        canonical_fields: dict[str, Any] = {
            "child_drive_root": str(child),
            "subagent_envelope": canonical_receipt,
        }
        if case == "heal":
            canonical_fields.update(
                {
                    "actual_substrate": "harness_used",
                    "delegated_runs_unreconciled": [],
                    "delegate_terminal_reconciliation": {
                        "trigger": "boot_backfill"
                    },
                }
            )
        write_task_result(data, task_id, STATUS_COMPLETED, **canonical_fields)

        expected = canonical_receipt if case == "heal" else child_receipt
        for result in (
            load_effective_task_result(data, task_id, materialize_artifacts=False),
            copy_child_task_result(
                data,
                {"id": task_id, "drive_root": str(child)},
            ),
        ):
            assert (
                result["subagent_envelope"]["execution_evidence"]
                == expected["execution_evidence"]
            )
            assert result["actual_substrate"] == "harness_used"
            if case == "heal":
                assert result["delegated_runs_unreconciled"] == []
                assert (
                    result["delegate_terminal_reconciliation"]["trigger"]
                    == "boot_backfill"
                )
            else:
                assert result["delegated_runs_unreconciled"] == ["stale"]
                assert (
                    result["delegate_terminal_reconciliation"]["trigger"]
                    == "terminal_write"
                )
