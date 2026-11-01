from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest
from tests._store_shared import result_exists, result_signature


def _isolated_queue(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod

    pending = []
    running = {}
    seq_counter = {"value": 0}
    # Bound for this test only (an in-place clear left its last seal in the module map for
    # the next queue test's snapshot); init_queue_refs still empties the admission maps.
    monkeypatch.setattr(queue_mod, "PENDING", pending)
    monkeypatch.setattr(queue_mod, "RUNNING", running)
    monkeypatch.setattr(queue_mod, "QUEUE_SEQ_COUNTER_REF", seq_counter)
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {})
    queue_mod.init_queue_refs(pending, running, seq_counter)
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
    return queue_mod, pending


def _write_restore_snapshot(tmp_path, tasks, fences):
    from ouroboros.utils import utc_now_iso

    path = tmp_path / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": task} for task in tasks],
            "running": [],
            "acceptance_fences": fences,
        }),
        encoding="utf-8",
    )


def _leftover_seal():
    """The row ``test_refused_begin_is_typed_refused_with_its_reason`` once left in the module map."""
    return {"token": "a" * 32, "root_task_id": "root-1", "task_id": "root-1",
            "status": "sealed", "outcome": "terminal"}


def test_isolated_queue_gives_the_module_queue_back_untouched(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod

    names = ("ACCEPTANCE_FENCES", "PENDING", "RUNNING", "QUEUE_SEQ_COUNTER_REF")

    def between_tests():
        return {"ACCEPTANCE_FENCES": {"root-1": _leftover_seal()}, "PENDING": [{"id": "queued"}],
                "RUNNING": {"running": {}}, "QUEUE_SEQ_COUNTER_REF": {"value": 7}}

    before = between_tests()
    for name, value in before.items():
        monkeypatch.setattr(queue_mod, name, value)
    with pytest.MonkeyPatch.context() as patch:
        q, pending = _isolated_queue(patch, tmp_path)
        inside = {name: getattr(q, name) for name in names}
        q.transition_acceptance_fence(action="begin", token="b" * 32, root_task_id="root-2", task_id="root-2")
        q.transition_acceptance_fence(action="end", token="b" * 32, outcome="terminal")
    # An in-place clear once left this test's seal in the module map for the next queue test.
    assert {name: getattr(queue_mod, name) for name in names} == between_tests()
    assert all(getattr(queue_mod, name) is before[name] for name in names)
    assert all(inside[name] is not before[name] for name in names) and inside["PENDING"] is pending
    assert list(inside["ACCEPTANCE_FENCES"]) == ["root-2"]
    assert inside["ACCEPTANCE_FENCES"]["root-2"]["status"] == "sealed"


def test_install_queue_never_snapshots_a_seal_an_earlier_test_left(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod
    from tests._budget_pause_exact_helpers import _install_queue

    leftover = {"root-1": _leftover_seal()}
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", leftover)
    with pytest.MonkeyPatch.context() as patch:
        q, _state, _workers = _install_queue(tmp_path, patch)
        inside = q.ACCEPTANCE_FENCES
        assert q.persist_queue_snapshot(reason="fixture_isolation")
    # A saved seal fences its root at restore, which cancels the root's never-started children.
    snapshot = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["acceptance_fences"] == []
    assert inside == {} and inside is not leftover
    assert queue_mod.ACCEPTANCE_FENCES is leftover and leftover == {"root-1": _leftover_seal()}


def test_acceptance_fence_atomically_blocks_then_releases_descendant(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)

    begun = queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1",
    )
    assert begun["status"] == "active"

    blocked = queue_mod.enqueue_task({
        "id": "child-1",
        "type": "task",
        "root_task_id": "root-1",
        "parent_task_id": "root-1",
        "delegation_role": "subagent",
    })
    assert blocked["_admission_blocked"] == "task_acceptance_fence"
    assert pending == []

    released = queue_mod.transition_acceptance_fence(
        action="end", token="a" * 32, outcome="revision",
    )
    assert released["status"] == "released"
    admitted = queue_mod.enqueue_task({
        "id": "child-2",
        "type": "task",
        "root_task_id": "root-1",
        "parent_task_id": "root-1",
        "delegation_role": "subagent",
    })
    assert admitted.get("_admission_blocked") is None
    assert [task["id"] for task in pending] == ["child-2"]


def test_terminal_acceptance_fence_stays_sealed_until_task_done(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-2", task_id="root-2",
    )
    sealed = queue_mod.transition_acceptance_fence(
        action="end", token="b" * 32, outcome="terminal",
    )
    assert sealed["status"] == "sealed"
    assert queue_mod.enqueue_task({
        "id": "late-child", "root_task_id": "root-2", "delegation_role": "subagent",
    })["_acceptance_fence_status"] == "sealed"
    assert not pending

    assert queue_mod.clear_acceptance_fence_for_root("root-2") is True
    queue_mod.enqueue_task({
        "id": "new-run-child", "root_task_id": "root-2", "delegation_role": "subagent",
    })
    assert [task["id"] for task in pending] == ["new-run-child"]


def test_acceptance_fence_is_visible_in_queue_snapshot(monkeypatch, tmp_path):
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="c" * 32, root_task_id="root-3", task_id="root-3",
    )
    payload = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    assert payload["acceptance_fences"] == [{
        "token": "c" * 32,
        "root_task_id": "root-3",
        "task_id": "root-3",
        "status": "active",
        "opened_at": payload["acceptance_fences"][0]["opened_at"],
        "owner_message_generation": 0,
    }]


def test_terminal_fence_generation_mismatch_releases_instead_of_sealing(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    begun = queue_mod.transition_acceptance_fence(
        action="begin", token="e" * 32, root_task_id="root-5", task_id="root-5",
    )
    assert begun["owner_message_generation"] == 0
    with queue_mod._queue_lock:
        queue_mod.ACCEPTANCE_FENCES["root-5"]["owner_message_generation"] += 1

    ended = queue_mod.transition_acceptance_fence(
        action="end", token="e" * 32, outcome="terminal", expected_generation=0,
    )
    assert ended["status"] == "released"
    assert ended["generation_mismatch"] is True
    assert ended["owner_message_generation"] == 1
    assert "root-5" not in queue_mod.ACCEPTANCE_FENCES
    queue_mod.enqueue_task({
        "id": "post-followup-child", "root_task_id": "root-5", "delegation_role": "subagent",
    })
    assert [task["id"] for task in pending] == ["post-followup-child"]


def test_acceptance_fence_reports_live_queue_descendants_until_quiescent(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    pending.append({
        "id": "pending-child",
        "root_task_id": "root-4",
        "parent_task_id": "root-4",
    })
    queue_mod.RUNNING["running-child"] = {
        "task": {
            "id": "running-child",
            "root_task_id": "root-4",
            "parent_task_id": "root-4",
        },
    }

    begun = queue_mod.transition_acceptance_fence(
        action="begin", token="d" * 32, root_task_id="root-4", task_id="root-4",
    )
    assert {(row["task_id"], row["status"]) for row in begun["queue_descendants"]} == {
        ("pending-child", "pending"),
        ("running-child", "running"),
    }

    pending.clear()
    queue_mod.RUNNING.clear()
    inspected = queue_mod.transition_acceptance_fence(action="inspect", token="d" * 32)
    assert inspected["status"] == "active"
    assert inspected["queue_descendants"] == []


@pytest.mark.parametrize("fence_status", ["active", "sealed"])
def test_restart_does_not_resurrect_descendant_behind_acceptance_fence(
    monkeypatch, tmp_path, fence_status
):
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    child = {
        "id": "late-child",
        "type": "task",
        "text": "late",
        "chat_id": 1,
        "root_task_id": "root-reviewing",
        "parent_task_id": "root-reviewing",
    }
    unrelated = {
        "id": "unrelated",
        "type": "task",
        "text": "safe",
        "chat_id": 1,
        "root_task_id": "unrelated",
    }
    _write_restore_snapshot(
        tmp_path,
        [child, unrelated],
        [{
            "token": "f" * 32,
            "root_task_id": "root-reviewing",
            "task_id": "root-reviewing",
            "status": fence_status,
        }],
    )

    assert queue_mod.restore_pending_from_snapshot() == 1
    assert [task["id"] for task in pending] == ["unrelated"]
    assert load_task_result(tmp_path, "late-child")["status"] == STATUS_CANCELLED
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()
    ]
    assert any(event.get("type") == "queue_restore_skipped_acceptance_fence" for event in events)


def test_malformed_acceptance_fence_snapshot_fails_closed_and_terminalizes(monkeypatch, tmp_path):
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    task = {"id": "uncertain", "type": "task", "text": "x", "chat_id": 1}
    _write_restore_snapshot(tmp_path, [task], {"not": "a list"})

    assert queue_mod.restore_pending_from_snapshot() == 0
    assert pending == []
    assert load_task_result(tmp_path, "uncertain")["status"] == STATUS_CANCELLED
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    diagnostic = next(event for event in events if event.get("type") == "queue_restore_invalid_acceptance_fences")
    assert diagnostic["action"] == "fail_closed_no_restore"
    assert diagnostic["affected_task_ids"] == ["uncertain"]


def test_restore_does_not_count_enqueue_admission_rejection(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    task = {"id": "blocked", "type": "task", "text": "x", "chat_id": 1}
    _write_restore_snapshot(tmp_path, [task], [])
    monkeypatch.setattr(
        queue_mod,
        "enqueue_task",
        lambda incoming, **_kwargs: {**incoming, "_admission_blocked": "project_routing_fence"},
    )

    assert queue_mod.restore_pending_from_snapshot() == 0
    assert pending == []
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()
    ]
    restored = next(event for event in events if event.get("type") == "queue_restored_from_snapshot")
    assert restored["restored_pending"] == 0
    assert restored["blocked_admission"] == ["blocked"]


def test_queue_rejects_negative_depth_before_admission_side_effects(monkeypatch, tmp_path):
    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)

    for index, raw_depth in enumerate((-1, -0.5, "-1", "not-a-depth")):
        task_id = f"invalid-queue-depth-{index}"
        token = f"token-{index}"
        queue_mod.ADMISSION_RESERVATIONS[task_id] = token
        admitted = queue_mod.enqueue_task({
            "id": task_id, "type": "task", "depth": raw_depth,
            "_admission_token": token,
        })
        assert admitted["_admission_blocked"] == "invalid_task_depth"
        assert task_id not in queue_mod.ADMISSION_RESERVATIONS
        assert pending == []


def test_queue_unique_id_wins_over_malformed_depth_replay(monkeypatch, tmp_path):
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, write_task_result

    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    task_id = "queue-live-replay"
    write_task_result(
        tmp_path,
        task_id,
        STATUS_SCHEDULED,
        root_task_id=task_id,
        delegation_role="root",
        result="keep live work",
    )
    original = result_signature(tmp_path, task_id)

    admitted = queue_mod.enqueue_task({
        "id": task_id,
        "type": "task",
        "depth": -1,
        "_require_unique_task_id": True,
    })

    assert admitted["_admission_blocked"] == "duplicate_task_id"
    assert result_signature(tmp_path, task_id) == original
    assert load_task_result(tmp_path, task_id)["status"] == STATUS_SCHEDULED
    assert pending == []


def test_restore_terminalizes_invalid_depth_at_task_budget_root(monkeypatch, tmp_path):
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    budget_root = tmp_path / "budget-root"
    task = {
        "id": "restore-invalid-depth",
        "type": "task",
        "text": "x",
        "chat_id": 1,
        "depth": -1,
        "budget_drive_root": str(budget_root),
    }
    _write_restore_snapshot(tmp_path, [task], [])

    assert queue_mod.restore_pending_from_snapshot() == 0
    assert pending == []
    result = load_task_result(budget_root, task["id"])
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "invalid_task_depth"
    assert result["depth"] == 0
    assert result["raw_task_depth"] == -1
    assert not result_exists(tmp_path, task['id'])
    snapshot = json.loads(queue_mod.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot["pending"] == []
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()
    ]
    restored = next(event for event in events if event.get("type") == "queue_restored_from_snapshot")
    assert restored["invalid_task_depth"] == [task["id"]]


def test_acceptance_ack_sidecar_compacts_stale_and_bounds_rows(monkeypatch, tmp_path):
    from supervisor import events, queue

    ack_dir = tmp_path / "state" / "acceptance_fence_acks"
    ack_dir.mkdir(parents=True)
    old = time.time() - 7200
    for index in range(260):
        path = ack_dir / f"{index:064x}.json"
        path.write_text('{}')
        os.utime(path, (old, old))
    monkeypatch.setattr(
        queue, "transition_acceptance_fence",
        lambda **_kwargs: {"ok": True, "status": "active"},
    )
    token, req = "f" * 64, "e" * 32
    events._handle_acceptance_fence(
        {"token": token, "req": req, "action": "begin", "root_task_id": "r", "task_id": "r"},
        SimpleNamespace(DRIVE_ROOT=tmp_path),
    )

    rows = list(ack_dir.glob("*.json"))
    assert len(rows) <= 256
    assert (ack_dir / f"{token}.{req}.json").is_file()
    assert all(path.stat().st_mtime > old for path in rows)


def test_split_drive_worker_reads_acceptance_ack_from_budget_root(tmp_path, monkeypatch):
    import queue as stdqueue

    from ouroboros import runtime_limits
    from ouroboros.agent import Env, OuroborosAgent

    monkeypatch.setattr(runtime_limits, "get_acceptance_fence_ack_wait_sec", lambda: 0.2)
    canonical = tmp_path / "canonical-data"
    child = canonical / "state" / "headless_tasks" / "root-1" / "data"
    repo = tmp_path / "repo"
    child.mkdir(parents=True)
    repo.mkdir()
    token, req = "a" * 32, "b" * 32
    payload = {"ok": True, "status": "active", "token": token}
    ack = canonical / "state" / "acceptance_fence_acks" / f"{token}.{req}.json"
    ack.parent.mkdir(parents=True)
    ack.write_text(json.dumps(payload), encoding="utf-8")

    # Avoid constructor-side LLM/Memory setup: this test exercises only the
    # production worker's one-shot acknowledgement reader.
    agent = object.__new__(OuroborosAgent)
    agent.env = Env(repo_dir=repo, drive_root=child)
    agent._current_task_metadata = {"budget_drive_root": str(canonical)}
    agent._event_queue = stdqueue.Queue()

    event = {"type": "acceptance_fence", "action": "inspect", "token": token, "req": req}
    assert agent._send_fence_event(event) == payload
    assert not ack.exists()
    child_ack = child / "state" / "acceptance_fence_acks" / f"{token}.{req}.json"
    assert not child_ack.exists()
