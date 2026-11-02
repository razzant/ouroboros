"""Queue publication, post-merge follow-up 4 (RV_MAIN3 R2/R3-A, R3-B, R5, R6).

A failed publication leaves the bundle on disk unknown (torn or old) with its certificate
revoked. From then on the publisher owes a fresh capture and a COMPLETE rewrite of every
document under a new generation, whatever the bytes say against the last cache, and the
failure clears only when that write lands; a quiet tick retries it, so one transient failure
whose change reverts never wedges the loop (R2/R3-A). An already-running quiet refresh never
resurrects a certificate a failed barrier revoked (R3-B). The Pause latch's barrier outside Q
keeps its typed refusal (R5). A kept saved queue is skipped before any row is copied (R6).
The reviewers' probes live here as the permanent pins.
"""

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import direct_roots, queue, queue_publication, queue_snapshot


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(direct_roots, "collect_direct_roots", lambda: {"roots": [], "incomplete": False})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _task(task_id, **extra):
    return {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "text": f"task {task_id}", **extra}


def _seed(published):
    root, now = published
    queue.RUNNING["r"] = {"task": _task("r", model_waits={}), "attempt": 1, "worker_id": 3,
                          "started_at": now[0] - 100, "last_heartbeat_at": now[0] - 1,
                          "budget_paused_sec": 0.0, "soft_sent": False}
    assert queue.persist_queue_snapshot("seed") is True
    queue_publication.tick_publisher_started()
    return root, now


def _recording_writer(monkeypatch):
    writes, real = [], queue.atomic_write_text
    monkeypatch.setattr(queue, "atomic_write_text", lambda *a, **k: writes.append(a[0].name) or real(*a, **k))
    return writes


# ---- R2 / R3-A: a failed write is retried with a complete rewrite, even when the change reverted ----

def test_a_reverted_change_after_a_failed_tick_write_is_rewritten_by_the_next_quiet_tick(published, monkeypatch):
    """Opus's R2 probe, inverted: the wedge is gone. A model wait starts, the tick's write fails, the
    wait resolves (an exact census revert) and the disk recovers: the next quiet tick rewrites the
    whole bundle under a new generation and certifies it; the ticks after it are quiet again."""
    root, now = _seed(published)
    assert queue_publication.publish_queue_changes() is True
    generation = _read(root, "queue.alive")["generation"]
    with queue._queue_lock:
        queue.RUNNING["r"]["task"]["model_waits"] = {"w1": {"state": "waiting", "revision": 1, "task_attempt": 1}}
    with monkeypatch.context() as broken:
        broken.setattr(queue, "atomic_write_text", lambda *_a: False)
        assert queue_publication.publish_queue_changes() is False
    assert not (root / "state/queue.alive").exists()
    with queue._queue_lock:
        queue.RUNNING["r"]["task"]["model_waits"]["w1"]["state"] = "resolved"
    writes = _recording_writer(monkeypatch)
    answers = []
    for _ in range(4):
        now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC * 2
        answers.append(queue_publication.publish_queue_changes())
    assert answers == [True, True, True, True], "one transient failure never wedges the loop"
    assert writes[:3] == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"], "a complete rewrite"
    assert len(writes) == 3, "once: the ticks after it are quiet"
    alive = _read(root, "queue.alive")
    assert alive["generation"] != generation, "a new generation, never the old one re-certified"
    assert all(_read(root, name)["generation"] == alive["generation"]
               for name in ("queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"))
    assert queue_publication._STATE["write_failed"] is False


def test_a_failed_census_publication_recovers_when_the_fact_reverts(published, monkeypatch):
    """Astra's probe: an owner wait that ended during the failed write."""
    root, now = _seed(published)
    queue.RUNNING["r"]["owner_wait"] = {"kind": "review", "state": "waiting"}
    with monkeypatch.context() as fail:
        fail.setattr(queue, "atomic_write_text", lambda *_a, **_kw: False)
        assert queue_publication.publish_queue_changes() is False
    queue.RUNNING["r"].pop("owner_wait")
    writes = _recording_writer(monkeypatch)
    assert [queue_publication.publish_queue_changes() for _ in range(3)] == [True, True, True]
    assert writes == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"]
    assert (root / "state/queue.alive").exists() and queue_publication._STATE["write_failed"] is False


def test_a_real_review_wait_that_resolves_after_a_failed_publication_recovers(published, monkeypatch):
    """Astra's real-handler probe: a review's wait is published on its root but does not move the
    author's quota clock; its resolution after a failed write is an exact census revert."""
    from supervisor.task_model_wait import handle_task_model_wait

    root, now = _seed(published)
    task = queue.RUNNING["r"]["task"]
    task["model_wait_owner_id"] = "author-owner"
    task["model_waits"] = {"previous": {"wait_id": "previous", "state": "resolved", "revision": 2, "task_attempt": 1}}
    assert queue.publish_queue_snapshot("prior_wait_settled") is True
    revision = queue_publication.queue_revision()
    ctx = SimpleNamespace(RUNNING=queue.RUNNING, DRIVE_ROOT=root, bridge=SimpleNamespace(push_log=lambda *_a: None))
    event = {"type": "task_model_wait", "task_id": "r", "wait_id": "review-w1", "task_attempt": 1, "revision": 1,
             "state": "waiting", "model_wait_owner_id": "review-owner", "reason": "quota",
             "quota_clock": {"revision": 1, "active": True}}
    handle_task_model_wait(event, ctx)
    assert queue_publication.queue_revision() == revision
    with monkeypatch.context() as fail:
        fail.setattr(queue, "atomic_write_text", lambda *_a: False)
        assert queue_publication.publish_queue_changes() is False
    handle_task_model_wait({**event, "revision": 2, "state": "resolved", "quota_clock": {"revision": 2, "active": False}}, ctx)
    assert queue_publication.queue_revision() == revision
    writes = _recording_writer(monkeypatch)
    assert [queue_publication.publish_queue_changes() for _ in range(3)] == [True, True, True]
    assert writes == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"]
    assert (root / "state/queue.alive").exists()


def test_a_partial_bundle_is_rewritten_by_a_barrier_even_when_the_facts_match_the_last_success(published, monkeypatch):
    """Astra's partial-bundle probe: the snapshot landed, the dispatch document did not, the facts
    reverted; an explicit barrier must never compare equal to the old cache and answer True over an
    unrepaired bundle."""
    from ouroboros.queue_observation import load_queue_document, queue_stamp

    root, now = _seed(published)
    queue.RUNNING["r"]["owner_wait"] = {"kind": "review", "state": "waiting"}
    real = queue.atomic_write_text
    with monkeypatch.context() as fail:
        fail.setattr(queue, "atomic_write_text",
                     lambda path, text: False if path.name == "queue_dispatch.json" else real(path, text))
        assert queue_publication.publish_queue_changes() is False
    queue.RUNNING["r"].pop("owner_wait")
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC * 2
    writes = _recording_writer(monkeypatch)
    assert queue.publish_queue_snapshot("explicit_repair") is True
    assert writes == ["queue_snapshot.json", "queue_dispatch.json", "direct_roots.json"], "the whole bundle, again"
    document = load_queue_document(root, fresh=True)
    assert queue_stamp(document), "one generation across the bundle, certified"
    assert document["running"][0]["owner_wait"] is None, "the reverted fact is what the repaired bundle says"
    assert _read(root, "queue_dispatch.json")["generation"] == document["generation"]
    assert queue_publication._STATE["write_failed"] is False


# ---- R3-B: an in-flight quiet refresh never resurrects a certificate a failed barrier revoked ------

def test_an_inflight_quiet_refresh_cannot_recertify_after_a_failed_barrier(published, monkeypatch):
    root, now = _seed(published)
    original = _read(root, "queue.alive")["generation"]
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC * 2
    reached, proceed = threading.Event(), threading.Event()
    due = queue_publication._alive_due

    def paused_due(heartbeats):
        if threading.current_thread().name == "quiet-tick":
            reached.set()
            assert proceed.wait(5)
        return due(heartbeats)

    monkeypatch.setattr(queue_publication, "_alive_due", paused_due)
    answers, errors = [], []

    def refresh():
        try:
            answers.append(queue_publication.publish_queue_changes())
        except BaseException as exc:  # noqa: BLE001 - the thread's failure is the assertion's evidence
            errors.append(repr(exc))

    thread = threading.Thread(target=refresh, name="quiet-tick")
    thread.start()
    try:
        assert reached.wait(5)
        with queue._queue_lock:
            queue.RUNNING["r"]["owner_wait"] = {"kind": "review", "state": "waiting"}
        with monkeypatch.context() as fail:
            fail.setattr(queue, "atomic_write_text", lambda *_a, **_kw: False)
            assert queue.publish_queue_snapshot("barrier") is False
        assert not (root / "state/queue.alive").exists()
    finally:
        proceed.set()
        thread.join(5)
    assert not thread.is_alive() and errors == []
    assert not (root / "state/queue.alive").exists(), "the revoked certificate stays revoked"
    assert answers == [False], "the refresh answers what it did: nothing certified"
    # The next tick repairs the bundle and certifies a new generation.
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] != original
    assert _read(root, "queue_snapshot.json")["running"][0]["owner_wait"] == {"kind": "review", "state": "waiting"}


