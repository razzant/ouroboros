"""Structural queue writes, atomic dispatch selection and generation freshness."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status, usage_accounting as usage
from ouroboros.utils import atomic_write_json
from supervisor import direct_roots, queue, queue_snapshot


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(direct_roots, "collect_direct_roots", lambda: {"roots": [], "incomplete": False})
    monkeypatch.setattr(queue_snapshot, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)  # the fence's in-flight re-read
    return tmp_path, now


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def test_ticks_only_write_liveness_and_derive_durations_on_read(published, monkeypatch):
    root, now = published
    queue.RUNNING["task"] = {"task": {"id": "task", "text": "large task " * 10_000},
                             "started_at": now[0] - 100, "last_heartbeat_at": now[0] - 2,
                             "model_wait_quota_clock": {"active": True, "elapsed_sec": 3, "observed_at": now[0] - 10},
                             "budget_paused_sec": 7, "sleep_parked_at": now[0] - 5}
    writes, writer = [], queue.atomic_write_text
    def record(path, content):
        assert queue._queue_lock._is_owned()
        writes.append((path.name, len(content)))
        return writer(path, content)
    monkeypatch.setattr(queue, "atomic_write_text", record)
    assert queue.persist_queue_snapshot("main_loop")
    before = (root / "state/queue_snapshot.json").read_bytes()
    generation = _read(root, "queue.alive")["generation"]
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert not {"runtime_sec", "quota_wait_sec", "execution_sec", "heartbeat_lag_sec", "last_heartbeat_at"} & row.keys()
    writes.clear()
    now[0] += 60
    queue.RUNNING["task"]["last_heartbeat_at"] = now[0] - 1
    assert queue.persist_queue_snapshot("main_loop")
    assert [name for name, _ in writes] == ["queue.alive"] and writes[0][1] < 250
    assert (root / "state/queue_snapshot.json").read_bytes() == before
    assert _read(root, "queue.alive")["generation"] == generation
    observed = task_status._load_queue_snapshot(root)
    assert task_status.queue_snapshot_observation(observed)["fresh"]
    row = observed["running"][0]
    assert row["runtime_sec"] == 160 and row["quota_wait_sec"] == 73
    assert row["execution_sec"] == 15 and row["heartbeat_lag_sec"] == 1
    queue.RUNNING["task"]["task"]["text"] = "structural edit"
    assert queue.persist_queue_snapshot("hold")
    assert _read(root, "queue.alive")["generation"] != generation
    generation = _read(root, "queue.alive")["generation"]
    now[0] += 60
    assert queue.persist_queue_snapshot("pre_restart_exit")
    snapshot = _read(root, "queue_snapshot.json")
    assert snapshot["generation"] != generation
    assert snapshot["ts"] == _read(root, "queue.alive")["ts"]


@pytest.mark.parametrize("failed", ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json", "queue.alive"])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_publish_never_refreshes_liveness_and_retries(published, monkeypatch, failed, raises):
    root, _now = published
    assert queue.persist_queue_snapshot()
    old = copy.deepcopy(queue.QUEUE_PUBLICATION)
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}
    writer = queue.atomic_write_text
    def fail(path, content):
        if path.name == failed:
            if raises:
                raise OSError("fixture publication failure")
            return False
        return writer(path, content)
    with monkeypatch.context() as broken:
        broken.setattr(queue, "atomic_write_text", fail)
        assert queue.persist_queue_snapshot() is False
    assert queue.QUEUE_PUBLICATION == old
    assert not (root / "state/queue.alive").exists()
    assert task_status.queue_snapshot_observation(task_status._load_queue_snapshot(root))["freshness"] == "unknown"
    assert task_status.task_has_live_queue_ownership(root, "missing")
    with pytest.raises(usage.UsageAccountingError):
        usage._check_dispatch_fences(usage.UsageScope(task_id="child", root_task_id="root"), root)
    assert queue.persist_queue_snapshot()
    generation = _read(root, "queue.alive")["generation"]
    assert all(_read(root, name)["generation"] == generation for name in
               ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"))
    assert task_status.task_has_live_queue_ownership(root, "missing") is False


@pytest.mark.parametrize("busy_attempts", [0, 1, 2, 3])
def test_stamp_revocation_retries_before_writing_documents(published, monkeypatch, busy_attempts):
    root, _now = published
    assert queue.persist_queue_snapshot()
    scope = usage.UsageScope(task_id="child", root_task_id="root")
    usage._check_dispatch_fences(scope, root)
    old = copy.deepcopy(queue.QUEUE_PUBLICATION)
    stamp = root / "state/queue.alive"
    before = {name: (root / "state" / name).read_bytes() for name in
              ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json", "queue.alive")}
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}
    attempts, pauses, writes = [], [], []
    unlink, writer = Path.unlink, queue.atomic_write_text

    def busy(path, *args, **kwargs):
        if path == stamp:
            attempts.append(path)
            if len(attempts) <= busy_attempts:
                raise PermissionError("fixture sharing violation")
        return unlink(path, *args, **kwargs)

    def record(path, content):
        assert not stamp.exists(), "documents must wait until the old certificate is revoked"
        writes.append(path.name)
        return writer(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", busy)
        patch.setattr(queue, "atomic_write_text", record)
        patch.setattr(queue_snapshot, "time", SimpleNamespace(sleep=pauses.append))
        assert queue.persist_queue_snapshot() is (busy_attempts < 3)
    assert len(attempts) == min(busy_attempts + 1, 3)
    assert len(pauses) == min(busy_attempts, 2)
    assert all(0 < pause <= 0.05 for pause in pauses)
    if busy_attempts == 3:
        assert not writes and queue.QUEUE_PUBLICATION == old
        assert before == {name: (root / "state" / name).read_bytes() for name in before}
        usage._check_dispatch_fences(scope, root)  # disclosed residual: old fences remain in force
        assert queue.persist_queue_snapshot()
    else:
        assert writes == list(before)
    assert _read(root, "queue.alive")["generation"] != old[2]
    with pytest.raises(usage.BudgetExceeded):
        usage._check_dispatch_fences(scope, root)


@pytest.mark.parametrize("blocked", [{}, {"_budget_pause": {}},
                                    {"_budget_pause_hold": {"reason": "stale_queue_row_grant_consumed", "selected": False}},
                                    {"_continuation_prepared": True}, {"_budget_pause_resume": {"authority": "sleep"}}])
def test_dispatch_projection_preserves_selection_and_missing_member_refuses(published, blocked):
    root, _now = published
    fence = {"root_task_id": "root", "fence_id": "fence", "status": "active", "other_fact": {"kept": True}}
    queue.BUDGET_ROOT_FENCES["root"] = fence
    task = {"id": "child", "root_task_id": "root", "text": "not a dispatch field",
            "_budget_pause_resume": {"pause_id": "pause", "grant_id": "grant", "authority": "explicit_resume",
                                     "root_fence_id": "fence", "selected_by": "owner", "extra": "omit"}, **blocked}
    queue.PENDING.append(task)
    assert queue.persist_queue_snapshot()
    dispatch = _read(root, "queue_dispatch.json")
    assert dispatch["budget_root_fences"] == [fence]
    assert "text" not in dispatch["members"]["child"]
    assert "extra" not in dispatch["members"]["child"]["_budget_pause_resume"]
    scope = usage.UsageScope(task_id="child", root_task_id="root")
    if blocked:
        with pytest.raises(usage.BudgetExceeded):
            usage._check_dispatch_fences(scope, root)
    else:
        usage._check_dispatch_fences(scope, root)
    queue.PENDING.clear()
    assert queue.persist_queue_snapshot()
    with pytest.raises(usage.BudgetExceeded):
        usage._check_dispatch_fences(scope, root)


@pytest.mark.parametrize("stamp", [None, {"generation": "foreign", "ts": "2099-01-01T00:00:00+00:00"}])
def test_missing_or_mismatched_stamp_is_unknown_for_all_documents(published, stamp):
    root, _now = published
    assert queue.persist_queue_snapshot()
    path = root / "state/queue.alive"
    if stamp is None:
        path.unlink()
    else:
        atomic_write_json(path, stamp)
    for name in ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"):
        seen = task_status.queue_snapshot_observation(queue_observation.load_queue_document(root, name))
        assert seen["freshness"] == "unknown" and seen["age_sec"] is None
    with pytest.raises(usage.UsageAccountingError):
        usage._check_dispatch_fences(usage.UsageScope(task_id="child", root_task_id="root"), root)



def test_controls_are_checked_while_waiting_and_never_before_the_first_read(published, monkeypatch):
    from ouroboros import _usage_wait

    root, _now = published
    assert queue.persist_queue_snapshot()

    def refuse():
        raise RuntimeError("controls checked")

    monkeypatch.setattr(_usage_wait, "_check", refuse)
    def read():
        return _usage_wait.retry_presend(lambda: queue_observation.dispatch_probe(root), check=False)

    # A certified publication answers on its first read: the caller's controls are not consulted.
    assert read()["generation"]
    # A publication in flight is re-read, and every re-read honours the caller's controls.
    stamp = root / "state/queue.alive"
    saved = stamp.read_bytes()
    stamp.unlink()
    with pytest.raises(RuntimeError, match="controls checked"):
        read()
    stamp.write_bytes(saved)

@pytest.mark.parametrize("structural_age,stamp_age,matching,stale", [
    (2000, 1, True, False), (1, 901, True, True), (1, 1, False, True),
])
def test_restore_900_second_gate_uses_matching_stamp(published, monkeypatch, structural_age, stamp_age, matching, stale):
    root, now = published
    assert queue.persist_queue_snapshot()
    snapshot = _read(root, "queue_snapshot.json")
    snapshot["ts"] = datetime.fromtimestamp(now[0] - structural_age, timezone.utc).isoformat()
    atomic_write_json(root / "state/queue_snapshot.json", snapshot)
    stamp = _read(root, "queue.alive")
    stamp.update(ts=datetime.fromtimestamp(now[0] - stamp_age, timezone.utc).isoformat())
    if not matching:
        stamp["generation"] = "different-generation"
    atomic_write_json(root / "state/queue.alive", stamp)
    monkeypatch.setattr(queue_snapshot, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue, "PRIOR_DIRECT_ROOTS", {})
    observed = []
    def retain(*args, stale, **kwargs):
        observed.append(stale)
        return [], [], [], {}
    monkeypatch.setattr(queue_snapshot, "_retain_snapshot_pending", retain)
    queue.restore_pending_from_snapshot()
    assert observed == [stale]


def test_schedule_worker_uses_stamp_instead_of_structural_age(published, monkeypatch):
    from supervisor import queue_schedules

    root, now = published
    monkeypatch.setattr(queue_schedules, "in_worker_process", lambda: True)
    assert queue.persist_queue_snapshot()
    now[0] += 2000
    assert queue.persist_queue_snapshot()
    assert queue_schedules._schedule_running_or_queued("schedule", root) is False
    now[0] += queue_schedules._QUEUE_SNAPSHOT_MAX_AGE_SEC + 1
    assert queue_schedules._schedule_running_or_queued("schedule", root) is None
    assert queue.persist_queue_snapshot()
    assert queue_schedules._schedule_running_or_queued("schedule", root) is False
    stamp = _read(root, "queue.alive")
    stamp["generation"] = "different-generation"
    atomic_write_json(root / "state/queue.alive", stamp)
    assert queue_schedules._schedule_running_or_queued("schedule", root) is None
