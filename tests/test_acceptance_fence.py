from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest


def _isolated_queue(monkeypatch, tmp_path):
    from supervisor import state as state_mod
    from supervisor import queue as queue_mod

    pending = []
    running = {}
    queue_mod.init_queue_refs(pending, running, {"value": 0})
    queue_mod.ACCEPTANCE_FENCES.clear()
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
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
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()
    ]
    assert events[-1]["type"] == "queue_restore_invalid_acceptance_fences"
    assert events[-1]["action"] == "fail_closed_no_restore"


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
    token = "f" * 64
    events._handle_acceptance_fence(
        {"token": token, "action": "begin", "root_task_id": "r", "task_id": "r"},
        SimpleNamespace(DRIVE_ROOT=tmp_path),
    )

    rows = list(ack_dir.glob("*.json"))
    assert len(rows) <= 256
    assert (ack_dir / f"{token}.json").is_file()
    assert all(path.stat().st_mtime > old for path in rows)


def test_split_drive_worker_reads_acceptance_ack_from_budget_root(tmp_path):
    from ouroboros.agent import Env, OuroborosAgent

    canonical = tmp_path / "canonical-data"
    child = canonical / "state" / "headless_tasks" / "root-1" / "data"
    repo = tmp_path / "repo"
    child.mkdir(parents=True)
    repo.mkdir()
    token = "a" * 32
    payload = {"ok": True, "status": "active", "token": token}
    ack = canonical / "state" / "acceptance_fence_acks" / f"{token}.json"
    ack.parent.mkdir(parents=True)
    ack.write_text(json.dumps(payload), encoding="utf-8")

    # Avoid constructor-side LLM/Memory setup: this test exercises only the
    # production worker's one-shot acknowledgement reader.
    agent = object.__new__(OuroborosAgent)
    agent.env = Env(repo_dir=repo, drive_root=child)
    agent._current_task_metadata = {"budget_drive_root": str(canonical)}

    assert agent._await_acceptance_fence_ack(token, timeout_sec=0.1) == payload
    assert not ack.exists()
    child_ack = child / "state" / "acceptance_fence_acks" / f"{token}.json"
    assert not child_ack.exists()
