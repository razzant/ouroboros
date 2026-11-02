"""An answered tree's Resume publishes its grant and each child's choice together."""
import copy

import pytest

from ouroboros import budget_pause, owner_pause
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.events_budget import install_exact_budget_pause, live_root_resume_grant
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_pause import _live_tree, _owner_park


def _park_tree(tmp_path, monkeypatch):
    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _live_tree(tmp_path, workers)
    workers.RUNNING.pop("root-1")
    write_task_result(tmp_path, "root-1", "completed")
    assert request_owner_pause("root-1", request_id="tree-publication")["ok"]
    saved = _owner_park(tmp_path, monkeypatch, "child-run", [])
    install_exact_budget_pause(workers, "child-run", budget_pause.exact_pause_marker(saved)["checkpoint"])
    assert q.persist_queue_snapshot(reason="before_failure")
    return q, workers


def _boot(q, workers, tmp_path, monkeypatch):
    workers.PENDING.clear()
    workers.RUNNING.clear()
    q.BUDGET_ROOT_FENCES.clear()
    monkeypatch.setattr(owner_pause, "_CACHE", {})
    assert q.restore_pending_from_snapshot() >= 1
    assert any(t["id"] == "child-run" for t in workers.PENDING)
    assert load_task_result(tmp_path, "root-1")["status"] == "completed"


@pytest.mark.parametrize("failure", ["false", "exception", "partial", "fence_write", "rebound", "none"])
def test_tree_resume_publication_and_retry_after_real_boot(tmp_path, monkeypatch, failure):
    q, workers = _park_tree(tmp_path, monkeypatch)
    if failure == "rebound":
        assert q.resume_budget_paused_task("root-1")["ok"]
        assert request_owner_pause("root-1", request_id="second-pause")["ok"]
        assert q.persist_queue_snapshot(reason="before_failure")
    before = copy.deepcopy(workers.PENDING)
    latch = copy.deepcopy(q.BUDGET_ROOT_FENCES)
    fence = owner_pause.read_fence(tmp_path, "root-1")
    persist = q.persist_queue_snapshot

    def publish(**kw):
        if kw.get("reason") == "owner_pause_tree_resumed" and failure in {"false", "exception", "rebound"}:
            if failure == "exception":
                raise OSError("injected queue publication failure")
            return False
        return persist(**kw)

    with monkeypatch.context() as fault:
        fault.setattr(q, "persist_queue_snapshot", publish)
        if failure == "partial":
            write = q.atomic_write_text
            fault.setattr(q, "atomic_write_text", lambda path, body: False
                          if path.name == "queue_dispatch.json" else write(path, body))
        if failure == "fence_write":
            def fail_release(*a, **kw):
                raise OSError("injected fence publication failure")
            fault.setattr(owner_pause, "set_fence_state", fail_release)
        first = q.resume_budget_paused_task("root-1")
    restored_in_memory = (workers.PENDING == before and q.BUDGET_ROOT_FENCES == latch
                          and owner_pause.read_fence(tmp_path, "root-1") == fence)
    _boot(q, workers, tmp_path, monkeypatch)
    retry = q.resume_budget_paused_task("root-1")
    assert retry["ok"], retry
    assert live_root_resume_grant(q, "root-1", tmp_path)
    assert q.resume_budget_paused_task("child-run", selected_by="root-1")["ok"]
    assert all(t["id"] != "root-1" for t in workers.PENDING)
    assert next(t for t in workers.PENDING if t["id"] == "child-new")["_budget_pause_hold"]
    if failure == "none":
        assert first["ok"] and first["snapshot_persisted"]
    else:
        assert first["ok"] is False
        assert first["error"] == ("grant_not_recorded" if failure == "fence_write" else "snapshot_not_persisted")
        assert restored_in_memory, "rollback must restore the fence, latch and child carriers"


def test_old_released_tree_grant_does_not_ack_a_contradicting_restored_latch(tmp_path, monkeypatch):
    q, workers = _park_tree(tmp_path, monkeypatch)
    publications = {path: path.read_bytes() for path in (tmp_path / "state").iterdir()
                    if path.name in {"queue_snapshot.json", "queue_dispatch.json", "direct_roots.json", "queue.alive"}}
    assert q.resume_budget_paused_task("root-1")["ok"]
    # The old implementation could release the durable fence without publishing Q.
    for path, content in publications.items():
        path.write_bytes(content)
    _boot(q, workers, tmp_path, monkeypatch)
    assert q.BUDGET_ROOT_FENCES["root-1"]
    assert not live_root_resume_grant(q, "root-1", tmp_path)
    assert q.resume_budget_paused_task("root-1")["ok"]
    assert live_root_resume_grant(q, "root-1", tmp_path)
    assert q.resume_budget_paused_task("child-run", selected_by="root-1")["ok"]
