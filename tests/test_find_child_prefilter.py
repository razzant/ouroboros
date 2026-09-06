"""find_child_tasks filters raw rows BEFORE paying for the projection.

The materializing effective projection is not a read: it copies files,
rewrites artifact manifests and re-hashes every artifact. Running it over
EVERY stored task result to then filter by lineage made a read-shaped API
mutate unrelated trees' filesystems (and one finalization ran it 4-5x).
The lineage fields the filter needs are already on the raw disk row, so
non-matching rows must never reach the projection; the two classes that
cannot be judged raw (retry pointers, queue-live lineage-less rows) keep
their paths.
"""

import json
import pathlib

import ouroboros.task_status as task_status
from ouroboros.task_status import find_child_tasks, wait_for_effective_tasks


def _write_result(drive_root: pathlib.Path, row: dict) -> None:
    from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
    from ouroboros.task_result_schema import TASK_RESULT_SCHEMA_VERSION

    d = drive_root / "task_results"
    d.mkdir(parents=True, exist_ok=True)
    # Campaign ABI 7.0: readers QUARANTINE an unstamped row.
    row = {SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, **row}
    (d / f"{row['task_id']}.json").write_text(json.dumps(row))


def test_non_matching_rows_never_reach_the_projection(tmp_path, monkeypatch):
    _write_result(tmp_path, {
        "task_id": "t-child", "delegation_role": "subagent",
        "parent_task_id": "t-parent", "root_task_id": "t-parent",
        "status": "completed", "ts": "2026-08-30T00:00:01+00:00",
    })
    _write_result(tmp_path, {
        "task_id": "t-unrelated", "delegation_role": "",
        "parent_task_id": "", "root_task_id": "",
        "status": "completed", "ts": "2026-08-30T00:00:02+00:00",
    })
    _write_result(tmp_path, {
        "task_id": "t-other-tree", "delegation_role": "subagent",
        "parent_task_id": "t-elsewhere", "root_task_id": "t-elsewhere",
        "status": "completed", "ts": "2026-08-30T00:00:03+00:00",
    })
    projected = []
    real = task_status.effective_task_result

    def _spy(drive_root, item, **kwargs):
        projected.append(str(item.get("task_id") or ""))
        return real(drive_root, item, **kwargs)

    monkeypatch.setattr(task_status, "effective_task_result", _spy)
    rows = find_child_tasks(tmp_path, parent_task_id="t-parent", scope="direct")
    assert [r["task_id"] for r in rows] == ["t-child"]
    assert projected == ["t-child"], (
        "non-matching raw rows must be filtered before the materializing "
        f"projection, got projections for: {projected}"
    )


def test_retry_pointer_rows_are_admitted_past_the_prefilter(tmp_path):
    """A raw row whose lineage does not match may still project a RETRY row
    whose lineage does — the retry chain is followed inside the projection,
    so rows carrying a retry pointer must reach it."""
    _write_result(tmp_path, {
        "task_id": "t-old", "delegation_role": "subagent",
        "parent_task_id": "t-nobody", "root_task_id": "t-nobody",
        "superseded_by": "t-new",
        "status": "failed", "ts": "2026-08-30T00:00:01+00:00",
    })
    _write_result(tmp_path, {
        "task_id": "t-new", "delegation_role": "subagent",
        "parent_task_id": "t-parent", "root_task_id": "t-parent",
        "status": "completed", "ts": "2026-08-30T00:00:02+00:00",
    })
    rows = find_child_tasks(tmp_path, parent_task_id="t-parent", scope="direct")
    ids = {r["task_id"] for r in rows}
    assert "t-new" in ids


def test_queue_live_lineage_less_row_arrives_via_overlay(tmp_path):
    """A running child whose DISK row has no lineage yet is re-discovered by
    the queue-snapshot overlay (the raw prefilter may skip the disk row)."""
    _write_result(tmp_path, {
        "task_id": "t-live", "delegation_role": "",
        "parent_task_id": "", "root_task_id": "",
        "status": "running", "ts": "2026-08-30T00:00:01+00:00",
        "cost_usd": 1.75, "result": "partial work so far",
        "artifacts": [{"path": "a.txt"}],
    })
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "queue_snapshot.json").write_text(json.dumps({
        "pending": [],
        "running": [{
            "id": "t-live",
            "task": {
                "id": "t-live", "delegation_role": "subagent",
                "parent_task_id": "t-parent", "root_task_id": "t-parent",
            },
        }],
    }))
    rows = find_child_tasks(tmp_path, parent_task_id="t-parent", scope="direct")
    by_id = {r["task_id"]: r for r in rows}
    assert "t-live" in by_id
    # The overlay must carry the DISK row's content, not a thin queue stub —
    # a stub-only overlay zeroes the cost rollup and loses the partial result.
    live = by_id["t-live"]
    assert live["cost_usd"] == 1.75
    assert live["result"] == "partial work so far"
    assert live["artifacts"] == [{"path": "a.txt"}]
    assert live["status"] == "running"


def test_wait_polls_without_materializing_and_finishes_with_one_full_read(tmp_path, monkeypatch):
    _write_result(tmp_path, {
        "task_id": "t-done", "status": "completed",
        "ts": "2026-08-30T00:00:01+00:00",
    })
    calls = []
    real = task_status.load_effective_task_result

    def _spy(drive_root, tid, materialize_artifacts=True):
        calls.append(bool(materialize_artifacts))
        return real(drive_root, tid, materialize_artifacts=materialize_artifacts)

    monkeypatch.setattr(task_status, "load_effective_task_result", _spy)
    out = wait_for_effective_tasks(tmp_path, ["t-done"], timeout_sec=1)
    assert out["all_terminal"] is True
    # Every in-loop poll is a projection-only read; exactly the final read
    # (after the loop, on every exit path) materializes.
    assert calls, "no reads recorded"
    assert calls[-1] is True
    assert all(flag is False for flag in calls[:-1])
