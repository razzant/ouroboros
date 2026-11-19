"""Reserve real attempts across blocked and failed queue publications."""
import errno
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ouroboros import budget_pause, queue_observation
from tests.test_budget_pause_monetary_selection import CHILD, FENCED, ROOT, SIBLING, _child_under_latch, _reserve, _tree


def no_publication_wait(monkeypatch):
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("reservation waited for the snapshot"),
                        raising=False)


@pytest.mark.parametrize("tighten", [False, True])
def test_reserve_while_snapshot_write_is_blocked(tmp_path, monkeypatch, tighten):
    queue, _, _ = _tree(tmp_path, monkeypatch, sibling=True)
    assert queue.publish_queue_snapshot()
    assert isinstance(_reserve(tmp_path, CHILD), str)
    queue.RUNNING[CHILD]["task"]["text"] = "changed snapshot"
    if tighten:
        queue.BUDGET_ROOT_FENCES[ROOT] = {"root_task_id": ROOT, "fence_id": "new", "status": "active"}
    entered, release = threading.Event(), threading.Event()
    write = queue.atomic_write_text

    def blocked(path, data):
        if path == queue.QUEUE_SNAPSHOT_PATH:
            entered.set()
            assert release.wait(10)
        return write(path, data)

    monkeypatch.setattr(queue, "atomic_write_text", blocked)
    no_publication_wait(monkeypatch)
    with ThreadPoolExecutor(1) as pool:
        publishing = pool.submit(queue.publish_queue_snapshot)
        try:
            assert entered.wait(10)
            answer = _reserve(tmp_path, CHILD)
            assert answer == FENCED if tighten else isinstance(answer, str)
        finally:
            release.set()
        assert publishing.result()


@pytest.mark.parametrize("failed", ["queue_snapshot.json", "queue_dispatch.json"])
def test_failed_resume_keeps_prior_fence_and_single_child_selection(tmp_path, monkeypatch, failed):
    queue, _, _, _, _ = _child_under_latch(tmp_path, monkeypatch, sibling=True)
    path = tmp_path / "state/queue_dispatch.json"
    old = path.read_bytes()
    write, unlink = queue.atomic_write_text, Path.unlink

    def fail(path, text):
        if path.name == failed:
            raise OSError(errno.ENOSPC, "volume full")
        return write(path, text)

    def sharing_violation(target, *args, **kw):
        assert target != path, "a reader holds the destination on Windows"
        return unlink(target, *args, **kw)

    no_publication_wait(monkeypatch)
    monkeypatch.setattr(Path, "unlink", sharing_violation)
    with monkeypatch.context() as fault:
        fault.setattr(queue, "atomic_write_text", fail)
        answer = queue.resume_budget_paused_task(CHILD)
    assert answer.get("error") == "snapshot_not_persisted"
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["state"] == "paused"
    assert path.read_bytes() == old
    assert _reserve(tmp_path, CHILD) == _reserve(tmp_path, SIBLING) == FENCED
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    assert isinstance(_reserve(tmp_path, CHILD), str) and _reserve(tmp_path, SIBLING) == FENCED


def test_two_snapshot_failures_leave_unchanged_dispatch_readable(tmp_path, monkeypatch):
    from ouroboros.usage_ledger import ControlAuthorityUnavailable

    queue, _, _ = _tree(tmp_path, monkeypatch, sibling=True)
    assert queue.publish_queue_snapshot()
    queue.RUNNING[CHILD]["task"]["text"] = "new"
    write = queue.atomic_write_text
    monkeypatch.setattr(queue, "atomic_write_text",
                        lambda path, data: False if path == queue.QUEUE_SNAPSHOT_PATH else write(path, data))
    no_publication_wait(monkeypatch)
    for _ in range(2):
        assert queue.publish_queue_snapshot() is False
        assert isinstance(_reserve(tmp_path, CHILD), str)
    dispatch = tmp_path / "state/queue_dispatch.json"
    for contents in (b'{}', b'{', None):
        dispatch.unlink() if contents is None else dispatch.write_bytes(contents)
        with pytest.raises(ControlAuthorityUnavailable):
            _reserve(tmp_path, CHILD)


def test_failed_barrier_supersedes_an_older_capture_with_spend_permission(tmp_path, monkeypatch):
    from supervisor import queue_publication as pub

    queue, _, _, _, _ = _child_under_latch(tmp_path, monkeypatch, sibling=True)
    old_publish = queue.publish_queue_snapshot
    overtaken = []

    def barrier(reason=""):
        with queue._queue_lock:
            overtaken.append(pub._capture(reason))
        return old_publish(reason=reason)

    with monkeypatch.context() as fault:
        fault.setattr(queue, "publish_queue_snapshot", barrier)
        write = queue.atomic_write_text
        fault.setattr(queue, "atomic_write_text", lambda path, data: False if path == queue.QUEUE_SNAPSHOT_PATH
                      else write(path, data))
        assert queue.resume_budget_paused_task(CHILD)["error"] == "snapshot_not_persisted"
    assert pub._publish(overtaken[0]) is False
    assert _reserve(tmp_path, CHILD) == FENCED
    assert queue.resume_budget_paused_task(CHILD)["ok"] and isinstance(_reserve(tmp_path, CHILD), str)


def test_mixed_member_selection_closes_old_before_opening_new(tmp_path, monkeypatch):
    queue, _, _, _, _ = _child_under_latch(tmp_path, monkeypatch, sibling=True)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    pending = {task["id"]: task for task in queue.PENDING}
    assert isinstance(_reserve(tmp_path, CHILD), str) and _reserve(tmp_path, SIBLING) == FENCED
    pending[SIBLING]["_budget_pause_resume"] = pending[CHILD].pop("_budget_pause_resume")
    pending[SIBLING].pop("_budget_pause", None)
    during = []
    write = queue.atomic_write_text

    def observe(path, data):
        if path == queue.QUEUE_SNAPSHOT_PATH:
            during.append((_reserve(tmp_path, CHILD), _reserve(tmp_path, SIBLING)))
        return write(path, data)

    monkeypatch.setattr(queue, "atomic_write_text", observe)
    assert queue.publish_queue_snapshot() and during == [(FENCED, FENCED)]
    assert _reserve(tmp_path, CHILD) == FENCED and isinstance(_reserve(tmp_path, SIBLING), str)


def test_old_reader_export(tmp_path, monkeypatch):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from tests._budget_pause_exact_helpers import _install_queue
    from ouroboros.queue_observation import SHARED_KEY
    from supervisor import queue_publication as pub
    from supervisor.work_stop_migration import FORMAT_KEY
    from tests.test_queue_publication_revision import _task

    q, _, _ = _install_queue(tmp_path, monkeypatch)
    q.PENDING.extend([_task("a"), _task("b")])
    assert q.publish_queue_snapshot("admit")
    entered, release = threading.Event(), threading.Event()
    original = pub._documents

    def ordered(capture):
        if capture["reason"] == "pre_restart_exit":
            entered.set()
            assert release.wait(12)
        return original(capture)

    monkeypatch.setattr(pub, "_documents", ordered)
    with ThreadPoolExecutor(1) as pool:
        forced = pool.submit(q.publish_queue_snapshot, "pre_restart_exit")
        try:
            assert entered.wait(12)
            q.PENDING[0]["text"] = "newest input"
            assert q.publish_queue_snapshot("admit")
        finally:
            release.set()
        reply = forced.result()
    doc = json.loads(q.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert reply is True and doc[FORMAT_KEY] == 1 and SHARED_KEY not in doc
    assert doc["pending"][0]["task"]["text"] == "newest input"


@pytest.mark.parametrize("bad", ["generation", "members", "valid"])
def test_repair_order(tmp_path, monkeypatch, bad):
    import errno
    import json
    from ouroboros import queue_observation as obs
    from tests.test_budget_pause_monetary_selection import _tree, ROOT

    q, _, _ = _tree(tmp_path, monkeypatch, sibling=True)
    assert q.publish_queue_snapshot("before")
    path = tmp_path / "state/queue_dispatch.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if bad == "generation":
        doc["generation"] = 7
    if bad == "members":
        doc["members"]["other"] = []
    path.write_text(json.dumps(doc))

    def accepted():
        view = obs.dispatch_probe(tmp_path)
        return view is not None and obs.valid_dispatch_facts(view)

    before = accepted()
    q.BUDGET_ROOT_FENCES[ROOT] = {"root_task_id": ROOT, "fence_id": "branch", "status": "active"}
    write = q.atomic_write_text

    def full(path, data):
        if path == q.QUEUE_SNAPSHOT_PATH:
            raise OSError(errno.ENOSPC, "disk full")
        return write(path, data)

    monkeypatch.setattr(q, "atomic_write_text", full)
    reply = q.publish_queue_snapshot("tighten_one_tree")
    assert reply is False and accepted() == before
