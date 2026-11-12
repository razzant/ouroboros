"""Queue context reports dated evidence without changing scheduling authority."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import task_status
from ouroboros.context_runtime_facts import _queue_context_fact
from ouroboros.engine_facts import engine_pool_line
from tests._shared import write_queue_document


def _snapshot(age=0):
    return {
        "ts": datetime.fromtimestamp(1000 - age, timezone.utc).isoformat(),
        "running": [{"id": "child", "task": {"delegation_role": "subagent", "root_task_id": "root"}}],
        "pending": [], "reaping_count": 1, "worker_total": 3, "assignable_idle_workers": 1,
    }


def _write(root, snapshot):
    path = root / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_queue_document(path, snapshot)
    return path


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(task_status.time, "time", lambda: 1000)


@pytest.mark.parametrize("age,fresh", [(0, True), (10.001, False), (900, False)])
def test_context_freshness_shares_ownership_boundary(age, fresh):
    snapshot = _snapshot(age)
    snapshot.update(generation="fixture-generation", _queue_alive={
        "generation": "fixture-generation", "ts": snapshot["ts"],
    })
    fact = task_status.queue_snapshot_observation(snapshot)
    assert fact["fresh"] is fresh
    assert task_status._snapshot_is_stale(snapshot) is not fresh
    assert fact["age_sec"] == age
    assert fact["freshness"] == ("fresh" if fresh else "stale")
    assert fact["ts"] == snapshot["ts"]


@pytest.mark.parametrize("snapshot", [{}, {"ts": "invalid"}, {"_snapshot_invalid": True}])
def test_unavailable_freshness_is_unknown(snapshot):
    fact = task_status.queue_snapshot_observation(snapshot)
    assert fact["freshness"] == "unknown"
    assert fact["age_sec"] is None
    assert not fact["fresh"]


def test_child_context_and_schedule_share_canonical_last_known_snapshot(tmp_path, monkeypatch):
    canonical, child = tmp_path / "canonical", tmp_path / "child"
    path = _write(canonical, _snapshot(60))
    _write(child, {**_snapshot(), "running": [], "assignable_idle_workers": 100})
    before = path.read_bytes()
    monkeypatch.setattr("ouroboros.config.DATA_DIR", child)
    task = {"budget_drive_root": str(canonical)}
    fact = _queue_context_fact(task)
    line = engine_pool_line(canonical)
    assert fact["freshness"] == "stale"
    assert fact["running_count"] == 1
    assert fact["free_worker_slots"] == 1
    assert fact["free_worker_slots_basis"] == "recorded_assignable_idle_workers"
    assert fact["source_root"] == str(canonical)
    assert "total 3" in line and "assignable idle 1" in line
    assert "assignable idle 100" not in line
    assert path.read_bytes() == before
    # A frozen context remains that dated observation even when the source advances.
    _write(canonical, {**_snapshot(), "assignable_idle_workers": 2})
    assert fact["freshness"] == "stale" and fact["free_worker_slots"] == 1
    assert _queue_context_fact(task)["free_worker_slots"] == 2


def test_legacy_capacity_remains_an_estimate(tmp_path, monkeypatch):
    snapshot = _snapshot()
    snapshot.pop("assignable_idle_workers")
    snapshot.pop("worker_total")
    _write(tmp_path, snapshot)
    monkeypatch.setattr("ouroboros.config.get_max_workers", lambda: 10)
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert fact["free_worker_slots"] == 8
    assert fact["free_worker_slots_basis"] == "legacy_estimate_from_configured_limit"
    assert fact["worker_total"] is None


def test_a_recorded_null_is_unknown_never_a_legacy_spare(tmp_path, monkeypatch):
    """Only a snapshot from before the pool was published estimates; a published null stays unknown."""
    snapshot = {**_snapshot(), "assignable_idle_workers": None}
    _write(tmp_path, snapshot)
    monkeypatch.setattr("ouroboros.config.get_max_workers", lambda: 40)
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert (fact["free_worker_slots"], fact["free_worker_slots_basis"]) == (None, "unknown")
    assert fact["acting_worker_count"] is None
    line = engine_pool_line(tmp_path)
    assert "assignable idle not observed" in line and "acting not observed" in line


def test_a_model_wait_is_not_worker_saturation(tmp_path):
    """A running task waiting for its model keeps its slot; the pool still says what is idle and assignable."""
    snapshot = {**_snapshot(), "max_workers": 40, "worker_total": 2, "active_worker_count": 2, "busy_worker_count": 1,
                "acting_worker_count": 1, "parked_worker_count": 0,
                "assignable_idle_workers": 1, "pending_count": 0, "pool_observation_status": "observed",
                "running": [{"id": "child", "model_waits": {"w": {"state": "waiting", "task_attempt": 1}},
                             "task": {"delegation_role": "subagent", "root_task_id": "root"}}]}
    _write(tmp_path, snapshot)
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert (fact["running_count"], fact["free_worker_slots"], fact["acting_worker_count"]) == (1, 1, 1)
    line = engine_pool_line(tmp_path)
    assert ("with a task 1, acting 1, waiting on their process 0, booting not observed, reaping (incl. booting) 1, "
            "assignable idle 1, pending rows 0;") in line
    assert "model" not in line.split("project worker pool", 1)[1].split(";", 1)[0]


def test_pending_hold_projection_uses_stop_and_project_hold_owners(tmp_path):
    """Explicit pending holds count without treating ordinary queued rows as held."""
    snapshot = {"ts": datetime.fromtimestamp(1000, timezone.utc).isoformat(),
                "running": [], "pending": [
                    {"id": "owner", "task": {"id": "owner", "_owner_hold": {"reason": "owner"}}},
                    {"id": "project", "task": {"id": "project", "project_id": "p",
                                                   "_project_admission_restore_hold": {"reason": "restore"}}},
                    {"id": "budget", "task": {"id": "budget", "_budget_pause": {
                        "reason": "budget", "exact_continuation": True,
                        "status": "paused_exact_continuation"}}},
                    {"id": "queued", "task": {"id": "queued"}},
                ], "budget_root_fences": []}
    _write(tmp_path, snapshot)
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert (fact["pending_paused_or_held_count"], fact["pending_paused_or_held_status"]) == (3, "observed")


def test_runtime_renderer_exposes_dated_queue_without_live_claim(tmp_path, monkeypatch):
    from ouroboros import context

    _write(tmp_path, _snapshot(60))
    monkeypatch.setattr(context, "_runtime_budget_info", lambda *a: {})
    monkeypatch.setattr(context, "_delegation_capability_fact", lambda: None)
    monkeypatch.setattr(context, "official_update_projection", lambda *_: {})
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda p: tmp_path / p)
    rendered = context.build_runtime_section(env, {"budget_drive_root": str(tmp_path)})
    queue = json.loads(rendered.split("\n\n", 1)[1])["queue"]
    assert queue["freshness"] == "stale" and queue["age_sec"] == 60
    assert queue["running_count"] == 1 and queue["free_worker_slots"] == 1
    assert "live worker/queue load" not in rendered


@pytest.mark.parametrize("body", [None, "broken JSON"])
def test_missing_or_corrupt_snapshot_does_not_claim_empty_queue(tmp_path, body):
    if body is not None:
        path = _write(tmp_path, {})
        path.write_text(body, encoding="utf-8")
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert fact["freshness"] == "unknown"
    assert "running_count" not in fact and "free_worker_slots" not in fact
    assert "pool: not observed" in engine_pool_line(tmp_path)


def test_cancellation_observation_preserves_status_and_same_freshness(tmp_path, monkeypatch):
    from ouroboros import cancel_intents

    _write(tmp_path, _snapshot(60))
    monkeypatch.setattr(cancel_intents, "_validated_single_cancel_target", lambda *_: "child")
    monkeypatch.setattr(task_status, "load_task_result", lambda *a, **kw: {})
    fact = task_status.observe_cancellation_target(tmp_path, "child")["queue_snapshot"]
    assert fact["status"] == "unknown"
    assert fact["fresh"] is False and fact["freshness"] == "stale"
    assert fact["age_sec"] == 60
    # Existing cancellation admission stays fail-open toward possible live ownership.
    assert task_status.task_has_live_queue_ownership(tmp_path, "child")


def test_wait_dates_stale_worker_observation_without_changing_terminal_result(tmp_path, monkeypatch):
    _write(tmp_path, _snapshot(60))
    monkeypatch.setattr(task_status, "load_effective_task_result", lambda *a, **kw: {
        "task_id": "child", "status": "completed", "result": "finished",
    })
    result = task_status.wait_for_effective_tasks(tmp_path, ["child"], timeout_sec=0)
    assert result["all_terminal"] and not result["timed_out"]
    assert result["tasks"]["child"]["status"] == "completed"
    assert result["live_child_status"]["child"] == "running"
    assert result["queue_snapshot_observation"]["freshness"] == "stale"
    assert result["queue_snapshot_observation"]["age_sec"] == 60


@pytest.mark.parametrize("status, phase", [("awaiting", "requested"), ("accepted", "requested"), ("ready", "draining"),
                                           ("refused", None), (None, None)])
def test_a_pending_restart_is_a_dated_fact_only_while_it_is_pending(tmp_path, status, phase):
    """Audit 05.11, item 1: the acting focus learns that a restart request waits, whose it is and in whose words."""
    intent = {"status": status, "task_id": "requester", "reason": "adopt my commit"} if status else None
    _write(tmp_path, {**_snapshot(), **({"restart_intent": intent} if intent else {})})
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    expected = {"requested_by_task": "requester", "phase": phase, "requester_reason": "adopt my commit"} if phase else None
    assert fact.get("pending_restart") == expected
