"""The ``/api/state`` census reads a task body only where a fact lives nowhere else, and only once per change.

The work projection re-reads only a stored result whose row signature moved since the last projection; the
room history's current-work question reads hot rows only (its one fact, the post-task phase, is a hot column).
"""
import pytest

from ouroboros import store, task_results, task_store
from ouroboros.gateway import state as gateway_state, work_projection as wp
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_hot_facts import _task


@pytest.fixture(autouse=True)
def _fresh_results():
    wp._RESULTS.clear()
    yield
    wp._RESULTS.clear()


def _seed_open_run(root):
    with store.open_store(root, write=True) as txn:
        _task(txn, "episode-1", status="completed", title="The episode")
        _task(txn, "worker-1", status="running", root_task_id="episode-1", parent_task_id="episode-1")
        txn.put_custody("run-1", {"run_id": "run-1", "task_id": "worker-1", "root_task_id": "episode-1",
                                  "source": "delegated_subagent"}, state="held")


def _loads(monkeypatch):
    calls, real = [], task_results.load_task_result

    def counted(root, task_id, *args, **kwargs):
        calls.append(task_id)
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", counted)
    return calls


def test_an_unmoved_result_is_not_read_again_by_the_next_projection(tmp_path, monkeypatch):
    _seed_open_run(tmp_path)
    loads = _loads(monkeypatch)
    first = wp.works_snapshot(tmp_path, [], {})
    assert sorted(set(loads)) == ["episode-1", "worker-1"]
    assert [work["work_id"] for work in first["works"]] == ["episode-1"]
    loads.clear()
    second = wp.works_snapshot(tmp_path, [], {})
    assert loads == [], "an unchanged row was read again"
    assert second == first
    with store.open_store(tmp_path, write=True) as txn:  # one document moves
        _task(txn, "episode-1", status="completed", title="Renamed episode")
    third = wp.works_snapshot(tmp_path, [], {})
    assert loads == ["episode-1"], "only the moved document is read again"
    assert third["works"][0]["name"] == "Renamed episode"


def test_a_vanished_result_is_not_served_from_the_last_projection(tmp_path, monkeypatch):
    _seed_open_run(tmp_path)
    wp.works_snapshot(tmp_path, [], {})
    with store.open_store(tmp_path, write=True) as txn:
        txn.delete_task("episode-1")
    loads = _loads(monkeypatch)
    wp.works_snapshot(tmp_path, [], {})
    assert "episode-1" in loads, "a row without a signature is read, never taken from memory"


def test_the_tree_census_has_one_call_site(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(wp, "tree_model_handoffs", lambda root, root_id: seen.append(root_id) or [])
    obs = wp._Observations(tmp_path)
    assert obs.model_handoffs("paused-root") == [] and obs.model_handoffs("paused-root") == []
    assert seen == ["paused-root"], "one census per root per projection, through the seam"


def test_every_paused_root_shares_one_read_of_open_attempts(tmp_path, monkeypatch):
    from ouroboros import owner_pause

    batched, single = [], []
    monkeypatch.setattr(owner_pause, "trees_model_requests",
                        lambda root, ids: batched.append(sorted(ids)) or {str(i): [] for i in ids})
    monkeypatch.setattr(owner_pause, "tree_model_requests", lambda root, task_id: single.append(task_id) or [])
    owners = [{"activity_id": f"paused-{n}", "phase": "budget_paused", "kind": "managed_task"} for n in range(3)]
    owners.append({"activity_id": "running-1", "phase": "running", "kind": "managed_task"})
    wp._project(tmp_path, owners, {}, True, wp._Observations(tmp_path), {})
    assert batched == [["paused-0", "paused-1", "paused-2"]], "one read answers every paused root"
    assert single == [], "no paused root reads its open attempts alone"


def test_the_room_history_current_work_question_reads_no_task_body(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "asking", chat_id=7)
        _task(txn, "answered", chat_id=7, root_phase_checkpoint={"post_task_synthesis": "running"})
    workers.PENDING[:] = [{"id": "asking", "chat_id": 7}]
    workers.RUNNING["answered"] = {"task": {"id": "answered", "chat_id": 7}, "attempt": 1, "started_at": 1}
    monkeypatch.setattr("ouroboros.post_task_checkpoint.post_task_model_waits", lambda _root: [])
    bodies, real = [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda *a, **k: bodies.append(a[1]) or real(*a, **k))
    gateway_state._FINALIZING_MEMO.clear()
    current = gateway_state.current_work_in_room(tmp_path, {}, lambda chat_id, _row: chat_id == 7)
    assert current == ["asking"], "the answered root with open late work is not current conversation"
    assert bodies == [], "the post-task phase is a hot column: no body is read for it"


def test_hot_facts_answer_the_phase_and_wait_without_a_body(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "waiting", owner_wait={"state": "waiting", "reason": "owner_pause", "quiz_id": "q"},
              root_phase_checkpoint={"post_task_synthesis": "running"})
    bodies, real = [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document", lambda *a, **k: bodies.append(a[1]) or real(*a, **k))
    facts = gateway_state._task_hot_facts(tmp_path, "waiting")
    assert facts == {"status": "running", "late_phase": "running", "finalizing": True,
                     "owner_wait": {"state": "waiting", "reason": "owner_pause"}}
    assert gateway_state._task_hot_facts(tmp_path, "absent") == {}
    assert bodies == []
    full = gateway_state._task_activity_facts(tmp_path, "waiting")
    assert (full["finalizing"], full["late_phase"]) == (True, "running")
    assert full["owner_wait"]["quiz_id"] == "q", "the waited quiz's id lives in the body alone"


def test_a_paused_root_review_questions_share_one_stored_read(tmp_path, monkeypatch):
    from ouroboros import review_operation

    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "paused-root", status="completed")
    asked = []
    monkeypatch.setattr(review_operation, "task_has_live_review_operation",
                        lambda root, task_id, sent_only=False, result_loader=None, **_: asked.append(
                            (task_id, sent_only, bool(result_loader(task_id)))) or False)
    loads = _loads(monkeypatch)
    reviews = wp.tree_review_reads(tmp_path, "paused-root")
    assert reviews() is False and reviews(sent_only=True) is False and reviews() is False
    assert asked == [("paused-root", False, True), ("paused-root", True, True)], "each question once, at the seam"
    assert loads == ["paused-root"], "both questions over one stored read"
