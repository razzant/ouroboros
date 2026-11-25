"""Structural queue writes, atomic dispatch selection and generation freshness."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status, usage_accounting as usage
from ouroboros.utils import atomic_write_json
from supervisor import queue, queue_publication, queue_snapshot


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
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
    alive_writer = queue_publication._write_alive
    def record(path, content):
        assert not queue._queue_lock._is_owned(), "documents are written outside Q"
        writes.append((path.name, len(content)))
        return (alive_writer if path.name == "queue.alive" else writer)(path, content)
    monkeypatch.setattr(queue, "atomic_write_text", record)
    monkeypatch.setattr(queue_publication, "_write_alive", record)
    assert queue.persist_queue_snapshot("main_loop")
    before = (root / "state/queue_snapshot.json").read_bytes()
    generation = _read(root, "queue.alive")["generation"]
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert not {"runtime_sec", "quota_wait_sec", "execution_sec", "heartbeat_lag_sec", "last_heartbeat_at"} & row.keys()
    writes.clear()
    now[0] += 60
    queue.RUNNING["task"]["last_heartbeat_at"] = now[0] - 1
    assert queue.persist_queue_snapshot("main_loop")
    # The certificate alone, a few hundred bytes (generation, time, one heartbeat, the pool's counts), not the 100 KB row.
    assert [name for name, _ in writes] == ["queue.alive"] and writes[0][1] < 500
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


@pytest.mark.parametrize("failed", ["queue_snapshot.json", "queue_dispatch.json", "queue.alive"])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_publish_never_refreshes_liveness_and_retries(published, monkeypatch, failed, raises):
    root, _now = published
    assert queue.persist_queue_snapshot()
    old = copy.deepcopy(queue.QUEUE_PUBLICATION)
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}
    writer, alive_writer = queue.atomic_write_text, queue_publication._write_alive
    def fail(path, content):
        if path.name == failed:
            if raises:
                raise OSError("fixture publication failure")
            return False
        return (alive_writer if path.name == "queue.alive" else writer)(path, content)
    with monkeypatch.context() as broken:
        broken.setattr(queue, "atomic_write_text", fail)
        broken.setattr(queue_publication, "_write_alive", fail)
        assert queue.persist_queue_snapshot() is False
    assert queue.QUEUE_PUBLICATION == old
    assert not (root / "state/queue.alive").exists()
    assert task_status.queue_snapshot_observation(task_status._load_queue_snapshot(root))["freshness"] == "unknown"
    assert task_status.task_has_live_queue_ownership(root, "missing")
    scope = usage.UsageScope(task_id="child", root_task_id="root")
    if failed == "queue_dispatch.json":
        usage._check_dispatch_fences(scope, root)
    else:
        with pytest.raises(usage.BudgetExceeded):
            usage._check_dispatch_fences(scope, root)
    assert queue.persist_queue_snapshot()
    generation = _read(root, "queue.alive")["generation"]
    assert all(_read(root, name)["generation"] == generation for name in ("queue_snapshot.json", "queue_dispatch.json"))
    assert task_status.task_has_live_queue_ownership(root, "missing") is False


@pytest.mark.parametrize("busy_attempts", [0, 1, 3])
def test_stamp_revocation_retries_before_writing_documents(published, monkeypatch, busy_attempts):
    root, _now = published
    assert queue.persist_queue_snapshot()
    scope = usage.UsageScope(task_id="child", root_task_id="root")
    usage._check_dispatch_fences(scope, root)
    old = copy.deepcopy(queue.QUEUE_PUBLICATION)
    stamp = root / "state/queue.alive"
    before = {name: (root / "state" / name).read_bytes() for name in
              ("queue_snapshot.json", "queue_dispatch.json", "queue.alive")}
    queue.BUDGET_ROOT_FENCES["root"] = {"root_task_id": "root", "fence_id": "fence", "status": "active"}
    attempts, pauses, writes = [], [], []
    unlink, writer = Path.unlink, queue.atomic_write_text

    def busy(path, *args, **kwargs):
        if path == stamp:
            attempts.append(path)
            if len(attempts) <= busy_attempts:
                raise PermissionError("fixture sharing violation")
        return unlink(path, *args, **kwargs)

    alive_writer = queue_publication._write_alive

    def record(path, content):
        assert not stamp.exists(), "documents must wait until the old certificate is revoked"
        writes.append(path.name)
        return writer(path, content)

    def record_alive(path, content):
        writes.append(path.name)
        return alive_writer(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", busy)
        patch.setattr(queue, "atomic_write_text", record)
        patch.setattr(queue_publication, "_write_alive", record_alive)
        patch.setattr(queue_publication, "time", SimpleNamespace(sleep=pauses.append, monotonic=lambda: 0.0))
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
        assert writes == ["queue_dispatch.json", "queue_snapshot.json", "queue.alive"]
    assert _read(root, "queue.alive")["generation"] != old[2]
    with pytest.raises(usage.BudgetExceeded):
        usage._check_dispatch_fences(scope, root)


@pytest.mark.parametrize("blocked", [{}, {"_continuation_prepared": True}, {"_budget_pause_resume": {"authority": "sleep"}}])
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
def test_missing_or_mismatched_stamp_leaves_dispatch_readable(published, stamp):
    root, _now = published
    assert queue.persist_queue_snapshot()
    path = root / "state/queue.alive"
    if stamp is None:
        path.unlink()
    else:
        atomic_write_json(path, stamp)
    for name in ("queue_snapshot.json", "queue_dispatch.json"):
        seen = task_status.queue_snapshot_observation(queue_observation.load_queue_document(root, name))
        assert seen["freshness"] == "unknown" and seen["age_sec"] is None
    usage._check_dispatch_fences(usage.UsageScope(task_id="child", root_task_id="root"), root)


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


def test_a_pending_rows_lane_wait_is_published_beside_its_task_and_never_restored_into_one(published, monkeypatch):
    """Owner 21.11.2026 («5A»): the snapshot and the status census say who keeps a queued root's lane now
    (``dispatch_choice.lane_waits``, computed under Q), beside the row, never inside its task; when the holder pauses
    the next capture and census say nothing. A restored row, even an older build's with a wait saved in its task,
    carries none, and the mind's list reads only the published one."""
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.peer_roster import independent_roots
    from supervisor import workers

    root, _now = published
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setattr(workers, "RUNNING", queue.RUNNING)
    behind = {"lane": "project", "task_id": "x", "project_id": "p"}
    holder = {"id": "x", "type": "task", "project_id": "p", "root_task_id": "x", "title": "Add Ollama support"}
    queue.RUNNING["x"] = {"task": holder, "worker_id": 0, "started_at": 1.0}
    queue.PENDING.extend([{"id": "q", "type": "task", "project_id": "p", "root_task_id": "q", "title": "Check CI",
                           "_folder_wait": {"task_id": "gone"}},
                          {"id": "free", "type": "task", "title": "Other"}])
    assert queue.persist_queue_snapshot(reason="test")
    rows = {row["id"]: row for row in _read(root, "queue_snapshot.json")["pending"]}
    assert rows["q"]["lane_wait"] == behind and "lane_wait" not in rows["free"]
    assert not {"lane_wait", "_lane_wait", "_folder_wait"} & set(rows["q"]["task"]), "a derived fact, never the task's"
    facts = {}
    _chat_activities_snapshot_safe(root, {}, direct_turns=[], work_facts=facts)
    assert facts["q"]["lane_wait"] == behind and "lane_wait" not in facts.get("free", {})
    assert [row["lane_wait"] for row in independent_roots(root)["roots"] if row["task_id"] == "q"] == [behind]
    restored = queue_snapshot._saved_pending_task(rows["q"])
    assert "lane_wait" not in restored and "_lane_wait" not in restored
    # An older build's saved note inside the task (format 1 or 2) is dropped where the old row is read, so it never
    # rides to the worker after a Resume; the row's own fields stay.
    old_task = {**rows["q"]["task"], "_folder_wait": {"task_id": "x"}}
    old = queue_snapshot._saved_pending_task({**rows["q"], "task": old_task})
    assert "_folder_wait" not in old and old["title"] == "Check CI"
    # The holder pauses: its lane is free at once, whatever any walk saw.
    del queue.RUNNING["x"]
    queue.PENDING.insert(0, {**holder, "_budget_pause": {"reason": "owner"}})
    assert queue.persist_queue_snapshot(reason="test")
    assert "lane_wait" not in {row["id"]: row for row in _read(root, "queue_snapshot.json")["pending"]}["q"]
    facts = {}
    _chat_activities_snapshot_safe(root, {}, direct_turns=[], work_facts=facts)
    assert "lane_wait" not in facts["q"]
    assert all("lane_wait" not in row for row in independent_roots(root)["roots"])


@pytest.mark.parametrize("change", ["claim_taken", "claim_released", "folder_settled", "room_folder_refreshed"])
def test_a_wait_that_changes_without_a_transition_reaches_both_publications_at_the_next_tick(published, monkeypatch,
                                                                                            change):
    """Review round 3 (Astra B2, Opus A2/A3) and round 4: who holds a lane changes in memory without a queue
    transition (a launch claim taken or released, a moved row's folder read off Q, the chooser's own refresh after
    its room's folder changed). The tick compares the waits with the ones last captured and moves the revision,
    so the snapshot the mind reads and the cached status body (keyed by ``state_publication._queue_revision``)
    follow; a quiet tick after it marks and writes nothing."""
    from ouroboros import project_placement
    from ouroboros.gateway.state_publication import _queue_revision
    from ouroboros.peer_roster import independent_roots
    from ouroboros.projects_registry import create_project, update_project
    from supervisor import worker_assignment, workers
    from tests.test_one_work_placement import place

    root, _now = published
    for name in ("PENDING", "RUNNING", "DRIVE_ROOT"):
        monkeypatch.setattr(workers, name, getattr(queue, name))
    monkeypatch.setattr(queue, "INITIALIZED", True, raising=False)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    shared = root.parent / f"shared-{change}"
    shared.mkdir()
    external = {"workspace_root": str(shared), "workspace_mode": "external"}
    holder = {"id": "x", "type": "task", "project_id": "p", "root_task_id": "x", "title": "Add Ollama support"}
    queued = {"id": "q", "type": "task", "project_id": "p", "root_task_id": "q", "title": "CI", "metadata": {}}
    if change.startswith("claim"):
        queue.PENDING.extend([{**holder, **({"_launch_claim": {"token": "t", "wid": 0}} if change == "claim_released"
                                            else {})}, queued])
        expected = {"lane": "project", "task_id": "x", "project_id": "p"}
    else:
        queue.RUNNING["x"] = {"task": {**holder, "project_id": "other", **external}, "worker_id": 0}
        intent = {"kind": "room_default", "project_id": "dest"} if change == "room_folder_refreshed" else None
        queue.PENDING.append({**queued, "project_id": "prior", "metadata": {"resource_intent": intent} if intent else {}})
        create_project(root, "dest", **({"working_dir": str(shared)} if change == "folder_settled" else {}))
        place(root, "q", "dest", workers)
        expected = {"lane": "folder", "task_id": "x", "project_id": "other", "folder": str(shared)}
        if change == "room_folder_refreshed":
            update_project(root, "dest", working_dir=str(shared))
    queue_publication.publish_queue_changes()

    def said():
        return [row.get("lane_wait") for row in independent_roots(root)["roots"] if row["task_id"] == "q"]

    before, key = said(), _queue_revision()
    with queue._queue_lock:  # the memory change itself, as its producer makes it: no mark
        row = next(task for task in queue.PENDING if task["id"] == "q")
        if change == "claim_taken":
            queue.PENDING[0]["_launch_claim"] = {"token": "t", "wid": 0}
        elif change == "folder_settled":
            assert project_placement.refresh_pending_placement(root, row, lane=True) and row["_placement_folder"] == ""
            basis = project_placement.pending_resource(row)
            assert basis is not None and project_placement.settle_resource(row, basis, False, False)
        elif change == "room_folder_refreshed":
            assert project_placement.refresh_pending_placement(root, row, lane=True)
    if change == "claim_released":
        worker_assignment._release_launch(queue, queue.PENDING[0], SimpleNamespace(busy_task_id="x", reaping=False), "t")
    assert _queue_revision() == key, "the change itself marks nothing"
    queue_publication.publish_queue_changes()
    after = [None] if change == "claim_released" else [expected]
    assert before == ([expected] if change == "claim_released" else [None]) and said() == after
    assert _queue_revision() != key, "the cached status body is produced again"
    revision, snapshot = queue_publication.queue_revision(), (root / "state" / "queue_snapshot.json").read_bytes()
    queue_publication.publish_queue_changes()
    assert queue_publication.queue_revision() == revision, "a quiet tick marks nothing"
    assert (root / "state" / "queue_snapshot.json").read_bytes() == snapshot, "and writes nothing"


@pytest.mark.parametrize("order", ["taken_then_released", "released_then_taken"])
def test_a_claim_between_two_ticks_never_leaves_a_card_body_built_from_waits_the_tick_did_not_see(published,
                                                                                                 monkeypatch, order):
    """Review round 4 (Astra B1): a launch claim taken and released (or released and taken again) between two
    ticks, with a status card body built in between, cached under the current queue revision. The census reads its
    waits through the publisher's noting (``queue_publication.note_lane_waits``), so the waits that body was built
    from are the last ones read: the next tick sees the change, and the next card body and the mind's snapshot say
    the same holder. Reading again with unchanged waits builds nothing; a quiet tick marks and writes nothing."""
    from ouroboros.gateway.state_publication import Publisher
    from ouroboros.peer_roster import independent_roots
    from supervisor import worker_assignment, workers
    from tests._work_projection_shared import state_fields

    root, _now = published
    for name in ("PENDING", "RUNNING", "DRIVE_ROOT"):
        monkeypatch.setattr(workers, name, getattr(queue, name))
    monkeypatch.setattr(queue, "INITIALIZED", True, raising=False)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    claim = {"token": "t", "wid": 0}
    holder = {"id": "x", "type": "task", "project_id": "p", "root_task_id": "x", "title": "Add Ollama support"}
    queue.PENDING.extend([{**holder, **({"_launch_claim": claim} if order == "released_then_taken" else {})},
                          {"id": "q", "type": "task", "project_id": "p", "root_task_id": "q", "title": "CI",
                           "metadata": {}}])
    queue_publication.publish_queue_changes()
    publisher = Publisher(str(root))

    def card():
        body = json.loads(publisher.refresh(lambda direct: {"fields": state_fields(root, {}, direct_turns=[]),
                                                            "task_bindings": {}}).body)
        return [reason.get("behind", {}).get("task_id") for work in body["works"] if work["work_id"] == "q"
                for reason in work["because"]]

    def mind():
        return [row.get("lane_wait") for row in independent_roots(root)["roots"] if row["task_id"] == "q"]

    def flip():
        if queue.PENDING[0].get("_launch_claim"):
            worker_assignment._release_launch(queue, queue.PENDING[0], SimpleNamespace(busy_task_id="x",
                                                                                         reaping=False), "t")
        else:
            with queue._queue_lock:
                queue.PENDING[0]["_launch_claim"] = dict(claim)

    flip()  # between the ticks: the claim taken (or released) ...
    assert card() == (["x"] if order == "taken_then_released" else [None]), "a card body built in between"
    flip()  # ... and released (or taken again) before the next tick
    queue_publication.publish_queue_changes()
    expected = [None] if order == "taken_then_released" else ["x"]
    assert card() == expected and [(wait or {}).get("task_id") for wait in mind()] == expected
    revision, built = queue_publication.queue_revision(), publisher.productions
    assert card() == expected and (queue_publication.queue_revision(), publisher.productions) == (revision, built), (
        "reading again with unchanged waits moves nothing")
    snapshot = (root / "state" / "queue_snapshot.json").read_bytes()
    queue_publication.publish_queue_changes()
    assert queue_publication.queue_revision() == revision, "a quiet tick marks nothing"
    assert (root / "state" / "queue_snapshot.json").read_bytes() == snapshot, "and writes nothing"


def test_the_ticks_wait_mark_keeps_a_pending_reason_and_a_barrier_records_the_waits_it_captured(published,
                                                                                                monkeypatch):
    """Round 4: the tick's wait mark never replaces the reason of a transition still unpublished (an export
    decides how that capture is written), names ``lane_wait`` only when nothing else is pending, marks nothing on
    an empty queue, and a barrier capture records the waits it captured so the next tick does not mark them again."""
    from supervisor import workers

    root, _now = published
    for name in ("PENDING", "RUNNING", "DRIVE_ROOT"):
        monkeypatch.setattr(workers, name, getattr(queue, name))
    monkeypatch.setattr(queue, "INITIALIZED", True, raising=False)
    monkeypatch.setitem(queue_publication._STATE, "waits", {})  # as a new queue generation starts (``reset``)
    revision = queue_publication.queue_revision()
    queue_publication.publish_queue_changes()
    assert queue_publication.queue_revision() == revision, "an empty queue's first tick marks nothing for waits"

    def reason():
        return _read(root, "queue_snapshot.json")["reason"]

    holder = {"task": {"id": "x", "type": "task", "project_id": "p", "root_task_id": "x"}, "worker_id": 0}
    queue.PENDING.append({"id": "q", "type": "task", "project_id": "p", "root_task_id": "q"})
    with queue._queue_lock:
        queue.RUNNING["x"] = holder
        queue_publication.mark_dirty_locked("pre_restart_exit")
    queue_publication.publish_queue_changes()
    assert reason() == "pre_restart_exit", "the pending export keeps its reason"
    del queue.RUNNING["x"]
    queue_publication._publish_now("barrier")
    revision = queue_publication.queue_revision()
    queue_publication.publish_queue_changes()
    assert queue_publication.queue_revision() == revision and reason() == "barrier", "the barrier's waits recorded"
    queue.RUNNING["x"] = holder
    queue_publication.publish_queue_changes()
    assert reason() == "lane_wait" and queue_publication.queue_revision() == revision + 1
