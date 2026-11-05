"""Legacy snapshots retain boot recovery without becoming live ownership proof."""

import datetime as dt
import json

import pytest

from ouroboros import server_maintenance as maintenance, task_status
from ouroboros.task_results import load_task_result
from ouroboros.utils import atomic_write_json
from supervisor import queue
from tests.test_startup_terminal_files import _interrupted_running_row, _recovery, _restore_rows, _started_split_root
from tests.test_startup_terminal_files import roots as roots


def _snapshot(root, *, structural_age, stamp_age, generation, pending=()):
    now = dt.datetime.now(dt.timezone.utc)
    snapshot = {"ts": (now - dt.timedelta(seconds=structural_age)).isoformat(),
                "pending": list(pending), "running": []}
    if generation is not None:
        snapshot["generation"] = "snapshot-generation"
    atomic_write_json(root / "state/queue_snapshot.json", snapshot)
    if stamp_age is not None:
        atomic_write_json(root / "state/queue.alive", {
            "generation": generation, "ts": (now - dt.timedelta(seconds=stamp_age)).isoformat(),
            "heartbeats": {},
        })


@pytest.mark.parametrize("structural_age,stamp_age,generation,restored", [
    (1, None, None, 1), (901, None, None, 0),
    (1, 901, "snapshot-generation", 0),
    (1, None, "snapshot-generation", 0), (1, 1, "foreign-generation", 0),
])
def test_boot_restore_uses_legacy_time_only_without_generation(
    roots, monkeypatch, structural_age, stamp_age, generation, restored,
):
    root, _repo = roots
    task = {"id": "ordinary", "type": "task", "chat_id": 1, "text": "retained work"}
    _snapshot(root, structural_age=structural_age, stamp_age=stamp_age,
              generation=generation, pending=[{"task": task}])
    monkeypatch.setattr(queue, "PRIOR_DIRECT_ROOTS", {})
    monkeypatch.setattr(queue, "enqueue_task", lambda task, **kwargs: queue.PENDING.append(task))
    assert queue.restore_pending_from_snapshot(max_age_sec=900) == restored
    assert [row["id"] for row in queue.PENDING] == (["ordinary"] if restored else [])


def test_periodic_reconcile_and_status_keep_legacy_liveness_unknown(roots):
    root, repo = roots
    _started_split_root(root, canonical_status="running")
    observed = task_status.queue_snapshot_observation(task_status._load_queue_snapshot(root))
    assert observed["freshness"] == "unknown" and not observed["fresh"]
    assert task_status.task_has_live_queue_ownership(root, "split")
    assert task_status.effective_task_result(root, load_task_result(root, "split"))["status"] == "running"
    maintenance._periodic_zombie_reconcile()
    assert load_task_result(root, "split")["status"] == "running"
    report = _recovery(root, repo)
    assert report["errors"] == report["unresolved"] == []
    assert load_task_result(root, "split")["reason_code"] == "orphaned_running_after_worker_restart"


@pytest.mark.parametrize("structural_age,stamp_age,generation,settled", [
    (1, None, None, True), (901, None, None, False),
    (1, 901, "snapshot-generation", False),
    (1, None, "snapshot-generation", False), (1, 1, "foreign-generation", False),
])
def test_split_root_boot_recovery_obeys_the_snapshot_protocol(
    roots, monkeypatch, structural_age, stamp_age, generation, settled,
):
    root, repo = roots
    _started_split_root(root)
    _snapshot(root, structural_age=structural_age, stamp_age=stamp_age, generation=generation)
    monkeypatch.setattr("ouroboros.agent.run_llm_loop", lambda *a, **k: pytest.fail("no task replay"))
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis", lambda *a, **k: None)
    report = _recovery(root, repo)
    assert report.get("rebound", []) == (["split"] if settled else [])
    assert report["errors"] == report["unresolved"] == []
    assert load_task_result(root, "split")["status"] == ("failed" if settled else "scheduled")


@pytest.mark.parametrize(("malformed", "publication"), [
    (False, "legacy_stale"),
    (False, "stamp_missing"),
    (False, "fresh"),
    (True, "stamp_stale"),
    (True, "stamp_mismatch"),
])
def test_restore_records_invalid_fences_before_stale_empty_return(roots, publication, malformed):
    root, _repo = roots
    _interrupted_running_row(root, "fenced")
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc)
    old = (now - dt.timedelta(seconds=2000)).isoformat()
    snapshot["ts"] = old if publication in {"legacy_stale", "fresh"} else now.isoformat()
    if publication != "legacy_stale":
        snapshot["generation"] = "snapshot-generation"
        if publication != "stamp_missing":
            atomic_write_json(root / "state/queue.alive", {
                "generation": "foreign-generation" if publication == "stamp_mismatch" else "snapshot-generation",
                "ts": old if publication == "stamp_stale" else now.isoformat(), "heartbeats": {},
            })
    snapshot["acceptance_fences"] = ["not-a-fence-object"] if malformed else []
    atomic_write_json(queue.QUEUE_SNAPSHOT_PATH, snapshot)
    fenced = []
    assert queue.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["fenced"]
    assert bool(_restore_rows(root, "queue_restore_invalid_acceptance_fences")) is malformed
    recorded = _restore_rows(root, "queue_restored_from_snapshot")
    assert recorded[-1]["terminalized_running"] == ["fenced"]
