from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest


def _isolated_queue(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod
    from supervisor import task_reaper

    pending = []
    running = {}
    queue_mod.init_queue_refs(pending, running, {"value": 0})
    queue_mod.ACCEPTANCE_FENCES.clear()
    task_reaper._REAPING_TASK_IDS.clear()
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
    result_path = tmp_path / "task_results" / f"{task_id}.json"
    original = result_path.read_bytes()

    admitted = queue_mod.enqueue_task({
        "id": task_id,
        "type": "task",
        "depth": -1,
        "_require_unique_task_id": True,
    })

    assert admitted["_admission_blocked"] == "duplicate_task_id"
    assert result_path.read_bytes() == original
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
    assert not (tmp_path / "task_results" / f"{task['id']}.json").exists()
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


def test_begin_with_lost_token_readopts_existing_live_fence(monkeypatch, tmp_path):
    """CyberGym full1507: the ack timed out AFTER the supervisor activated the
    fence, so the worker held no token while the fence stayed active. A re-begin
    with a fresh token must adopt the existing fence, not spin on rejections."""
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.RUNNING["root-1"] = {"task": {"id": "root-1", "root_task_id": "root-1"}}

    begun = queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1",
    )
    assert begun["status"] == "active"

    readopted = queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-1", task_id="root-1",
    )
    assert readopted["ok"] is True
    assert readopted["status"] == "active"
    assert readopted["token"] == "a" * 32
    assert readopted["re_adopted"] is True
    # The fence row is unchanged: same token, same owner, still one fence.
    assert queue_mod.ACCEPTANCE_FENCES["root-1"]["token"] == "a" * 32

    # The adopted token drives inspect/end exactly like the original one.
    inspected = queue_mod.transition_acceptance_fence(action="inspect", token="a" * 32)
    assert inspected["status"] == "active"
    ended = queue_mod.transition_acceptance_fence(
        action="end", token="a" * 32, outcome="terminal",
    )
    assert ended["status"] == "sealed"


def test_begin_reconciles_fence_whose_owner_is_dead(monkeypatch, tmp_path):
    """Dead-owner reconcile: a fence whose owner left RUNNING and PENDING is
    handed over to the new begin instead of rejecting it forever."""
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    begun = queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1",
    )
    assert begun["status"] == "active"
    # No RUNNING/PENDING entry for root-1: the owner is provably gone.

    reconciled = queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-1", task_id="root-1-retry",
    )
    assert reconciled["ok"] is True
    assert reconciled["status"] == "active"
    assert reconciled["token"] == "b" * 32
    assert reconciled["reconciled_dead_owner"] is True
    row = queue_mod.ACCEPTANCE_FENCES["root-1"]
    assert row["token"] == "b" * 32
    assert row["task_id"] == "root-1-retry"

    # The dead owner's token no longer drives the fence.
    orphaned = queue_mod.transition_acceptance_fence(action="inspect", token="a" * 32)
    assert orphaned["ok"] is False
    ended = queue_mod.transition_acceptance_fence(
        action="end", token="b" * 32, outcome="terminal",
    )
    assert ended["status"] == "sealed"


def test_sealed_fence_is_never_readopted(monkeypatch, tmp_path):
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.RUNNING["root-2"] = {"task": {"id": "root-2", "root_task_id": "root-2"}}
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-2", task_id="root-2",
    )
    sealed = queue_mod.transition_acceptance_fence(
        action="end", token="a" * 32, outcome="terminal",
    )
    assert sealed["status"] == "sealed"

    rejected = queue_mod.transition_acceptance_fence(
        action="begin", token="c" * 32, root_task_id="root-2", task_id="root-2",
    )
    assert rejected["ok"] is False
    assert "already sealed" in rejected["error"]
    assert queue_mod.ACCEPTANCE_FENCES["root-2"]["status"] == "sealed"


def test_gc_sweep_drops_dead_owner_fence_and_spares_live(monkeypatch, tmp_path):
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-dead", task_id="root-dead",
    )
    queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-live", task_id="root-live",
    )
    queue_mod.transition_acceptance_fence(
        action="begin", token="c" * 32, root_task_id="root-queued", task_id="root-queued",
    )
    queue_mod.RUNNING["root-live"] = {"task": {"id": "root-live", "root_task_id": "root-live"}}
    queue_mod.PENDING.append({"id": "root-queued", "root_task_id": "root-queued"})

    cleared = queue_mod.gc_acceptance_fences_for_dead_owners()
    assert cleared == ["root-dead"]
    assert set(queue_mod.ACCEPTANCE_FENCES) == {"root-live", "root-queued"}

    # The cleared root's subtree is admissible again; the spared fences still block.
    admitted = queue_mod.enqueue_task({
        "id": "child-of-dead", "root_task_id": "root-dead", "delegation_role": "subagent",
    })
    assert admitted.get("_admission_blocked") is None
    blocked = queue_mod.enqueue_task({
        "id": "child-of-live", "root_task_id": "root-live", "delegation_role": "subagent",
    })
    assert blocked["_admission_blocked"] == "task_acceptance_fence"


def test_enforce_task_timeouts_sweeps_dead_owner_fence_on_idle_queue(monkeypatch, tmp_path):
    """The watchdog sweep runs even when RUNNING is empty (no early return)."""
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-dead", task_id="root-dead",
    )
    queue_mod.enforce_task_timeouts()
    assert "root-dead" not in queue_mod.ACCEPTANCE_FENCES


def test_release_acceptance_fence_for_dead_owner_matches_owner_not_root(monkeypatch, tmp_path):
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1",
    )
    # A reaped CHILD of the fenced root must not drop its reviewing root's fence.
    assert queue_mod.release_acceptance_fence_for_dead_owner("child-1") is False
    assert "root-1" in queue_mod.ACCEPTANCE_FENCES
    # The confirmed-dead owner releases its own fence.
    assert queue_mod.release_acceptance_fence_for_dead_owner("root-1") is True
    assert "root-1" not in queue_mod.ACCEPTANCE_FENCES


def test_worker_adopts_existing_fence_token_from_begin_response():
    """Loop-side half of the idempotent begin: after a lost ack (begin raised,
    no token stored), the retry stores the EXISTING fence's token/generation."""
    from ouroboros.loop import _begin_task_acceptance_fence

    ctx = SimpleNamespace(task_metadata={"root_task_id": "root-1"})

    def failing_callback(**_kwargs):
        raise TimeoutError("supervisor did not acknowledge acceptance fence 1234")

    ctx.begin_acceptance_fence = failing_callback
    ok, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert (ok, token) == (False, None)
    assert getattr(ctx, "_task_acceptance_fence_token", None) is None

    def readopt_callback(**_kwargs):
        return {
            "ok": True,
            "status": "active",
            "token": "existing-token",
            "owner_message_generation": 2,
            "queue_descendants": [],
            "re_adopted": True,
        }

    ctx.begin_acceptance_fence = readopt_callback
    ok, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert ok is True
    assert token == "existing-token"
    assert ctx._task_acceptance_fence_token == "existing-token"
    assert ctx._task_acceptance_fence_generation == 2


def test_fence_wait_terminalizes_infra_failed_after_bounded_rounds(monkeypatch, tmp_path):
    """The fence wait must not burn paid rounds until the deadline: after the
    configured cap of consecutive fence-unavailable rounds the review exits
    with the typed infra-failure decision and the loop-side flag."""
    import ouroboros.loop as loop_mod
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", "3")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root-1"
    registry._ctx.task_metadata = {
        "task_id": "root-1",
        "root_task_id": "root-1",
        "parent_task_id": "",
        "delegation_role": "root",
    }
    registry._ctx.task_contract = {}
    monkeypatch.setattr(
        loop_mod, "_begin_task_acceptance_fence", lambda *_args, **_kwargs: (False, None),
    )

    def run_once(trace):
        return loop_mod._run_task_acceptance_review_once(
            tools=registry,
            content="answer",
            task_id="root-1",
            task_type="task",
            llm_trace=trace,
            drive_root=tmp_path,
            messages=[],
            emit_progress=lambda _message: None,
        )

    for expected_failures in (1, 2):
        trace = {"tool_calls": []}
        assert run_once(trace) is True
        assert trace["review_decision"]["eligibility"] == "acceptance_fence_failed"
        assert registry._ctx._task_acceptance_fence_failures == expected_failures
        assert not getattr(registry._ctx, "_task_acceptance_fence_infra_failed", False)

    trace = {"tool_calls": []}
    assert run_once(trace) is False
    assert registry._ctx._task_acceptance_fence_infra_failed is True
    decision = trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "acceptance_fence_unavailable"


def test_reaping_owner_is_not_dead_until_kill_confirmed(monkeypatch, tmp_path):
    """Sol race: enforce_task_timeouts pops the owner from RUNNING and hands it
    to the reaper; while the kill is unconfirmed (wedged hold) the worker may
    still be alive, so the sweep and the begin-reconcile must KEEP its fence.
    Only confirmed death releases it."""
    from supervisor import task_reaper

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-reap", task_id="root-reap",
    )
    # Owner popped from RUNNING into the reaper, kill not yet confirmed.
    task_reaper.note_task_reaping("root-reap")
    try:
        assert queue_mod.gc_acceptance_fences_for_dead_owners() == []
        assert "root-reap" in queue_mod.ACCEPTANCE_FENCES
        # A re-begin from the same owner re-adopts; it must NOT dead-owner-reconcile.
        readopted = queue_mod.transition_acceptance_fence(
            action="begin", token="b" * 32, root_task_id="root-reap", task_id="root-reap",
        )
        assert readopted["ok"] is True
        assert readopted.get("re_adopted") is True
        assert "reconciled_dead_owner" not in readopted
        assert queue_mod.ACCEPTANCE_FENCES["root-reap"]["token"] == "a" * 32
    finally:
        task_reaper._forget_task_reaping("root-reap")
    # Confirmed death: the reaper forgets the id and releases the fence.
    assert queue_mod.gc_acceptance_fences_for_dead_owners() == ["root-reap"]
    assert "root-reap" not in queue_mod.ACCEPTANCE_FENCES


def test_fence_wait_counter_resets_on_successful_begin(monkeypatch, tmp_path):
    """The bounded wait counts CONSECUTIVE failures: a successful begin between
    failures resets the counter, so fail/succeed/fail/fail must NOT terminalize
    at the cap of 3 — only the third consecutive failure does."""
    import ouroboros.loop as loop_mod
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", "3")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root-1"
    registry._ctx.task_metadata = {
        "task_id": "root-1",
        "root_task_id": "root-1",
        "parent_task_id": "",
        "delegation_role": "root",
    }
    registry._ctx.task_contract = {}
    begin_results = iter([(False, None), (True, "tok"), (False, None), (False, None), (False, None)])
    monkeypatch.setattr(
        loop_mod, "_begin_task_acceptance_fence", lambda *_args, **_kwargs: next(begin_results),
    )
    # The successful begin proceeds to a non-quiescent subtree wait (returns True).
    monkeypatch.setattr(
        loop_mod, "_task_acceptance_subtree_snapshot", lambda *_args, **_kwargs: (False, []),
    )

    def run_once(trace):
        return loop_mod._run_task_acceptance_review_once(
            tools=registry,
            content="answer",
            task_id="root-1",
            task_type="task",
            llm_trace=trace,
            drive_root=tmp_path,
            messages=[],
            emit_progress=lambda _message: None,
        )

    trace = {"tool_calls": []}
    assert run_once(trace) is True  # failure 1
    assert registry._ctx._task_acceptance_fence_failures == 1

    trace = {"tool_calls": []}
    assert run_once(trace) is True  # successful begin resets the counter
    assert registry._ctx._task_acceptance_fence_failures == 0
    assert trace["review_decision"]["eligibility"] == "waiting_for_quiescence"

    for expected_failures in (1, 2):
        trace = {"tool_calls": []}
        assert run_once(trace) is True  # failures 1 and 2 after the reset
        assert registry._ctx._task_acceptance_fence_failures == expected_failures
        assert not getattr(registry._ctx, "_task_acceptance_fence_infra_failed", False)

    trace = {"tool_calls": []}
    assert run_once(trace) is False  # third CONSECUTIVE failure terminalizes
    assert registry._ctx._task_acceptance_fence_infra_failed is True
    assert trace["acceptance_decision"]["reason"] == "acceptance_fence_unavailable"


def test_acceptance_fence_config_getters_defaults_and_clamps(monkeypatch):
    """The config SSOT getters ship 120s/3-round defaults, clamp to
    [5, 900] / [1, 50], and fall back to the default on a malformed value."""
    from ouroboros.config import (
        get_acceptance_fence_ack_timeout_sec,
        get_acceptance_fence_wait_max_rounds,
    )

    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_FENCE_ACK_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", raising=False)
    assert get_acceptance_fence_ack_timeout_sec() == 120.0
    assert get_acceptance_fence_wait_max_rounds() == 3

    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_ACK_TIMEOUT_SEC", "2")
    assert get_acceptance_fence_ack_timeout_sec() == 5.0
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_ACK_TIMEOUT_SEC", "5000")
    assert get_acceptance_fence_ack_timeout_sec() == 900.0
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", "0")
    assert get_acceptance_fence_wait_max_rounds() == 1
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", "999")
    assert get_acceptance_fence_wait_max_rounds() == 50

    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_ACK_TIMEOUT_SEC", "not-a-number")
    assert get_acceptance_fence_ack_timeout_sec() == 120.0
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_FENCE_WAIT_MAX_ROUNDS", "junk")
    assert get_acceptance_fence_wait_max_rounds() == 3


def test_direct_chat_owner_survives_gc_while_turn_busy(monkeypatch, tmp_path):
    """The direct-chat lane runs in-process and never enters RUNNING/PENDING:
    a fence whose owner is the currently-busy chat turn is ALIVE and must
    survive both the sweep and the enforce_task_timeouts tick."""
    from supervisor import workers

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-dc", task_id="root-dc",
    )
    monkeypatch.setattr(workers, "chat_turn_liveness", lambda: (True, "root-dc", 123.0))
    assert queue_mod.gc_acceptance_fences_for_dead_owners() == []
    assert "root-dc" in queue_mod.ACCEPTANCE_FENCES
    queue_mod.enforce_task_timeouts()
    assert "root-dc" in queue_mod.ACCEPTANCE_FENCES
    # A begin from the same live owner re-adopts; no dead-owner reconcile.
    readopted = queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-dc", task_id="root-dc",
    )
    assert readopted["ok"] is True
    assert readopted.get("re_adopted") is True
    assert "reconciled_dead_owner" not in readopted


def test_direct_chat_owner_collected_after_turn_goes_idle(monkeypatch, tmp_path):
    """Once the chat turn is no longer busy — or after a restart, when the
    in-process lane is gone — the same fence is collectable garbage."""
    from supervisor import workers

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-dc", task_id="root-dc",
    )
    monkeypatch.setattr(workers, "chat_turn_liveness", lambda: (False, None, None))
    assert queue_mod.gc_acceptance_fences_for_dead_owners() == ["root-dc"]
    assert "root-dc" not in queue_mod.ACCEPTANCE_FENCES


def test_bounded_fence_wait_terminalizes_via_forced_fallback(monkeypatch, tmp_path):
    """The bounded-wait flag set at the consecutive-failure cap must terminalize
    the no-tool finalization path as infra_failed through the host-salvage seam
    instead of finalizing past a review that never ran."""
    import queue

    import ouroboros.loop as loop_mod
    from ouroboros.outcomes import RESULT_INFRA_FAILED
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root-1"
    registry._ctx.task_metadata = {"root_task_id": "root-1", "budget_drive_root": str(tmp_path)}
    registry._ctx._task_acceptance_fence_infra_failed = True
    trace = {"tool_calls": [], "reasoning_notes": []}
    limit_ctx = loop_mod._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        1,
        tmp_path / "logs",
        "root-1",
        2,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
        incoming_messages=None,
        owner_msg_seen=set(),
    )
    loop_mod._finalize_limit_ctx(limit_ctx, registry, trace)
    monkeypatch.setattr(loop_mod, "_resolve_delivery_control", lambda content, *_args: ("fresh", content))
    monkeypatch.setattr(loop_mod, "_enforce_swarm_actions", lambda *_args: False)
    monkeypatch.setattr(loop_mod, "_compute_subagent_handoff", lambda *_args: None)
    monkeypatch.setattr(loop_mod, "_maybe_enforce_child_absorption_gate", lambda *_args: None)
    monkeypatch.setattr(loop_mod, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(loop_mod, "_finalize_task_services", lambda *_args: False)
    monkeypatch.setattr(loop_mod, "_run_task_acceptance_review_once", lambda **_kwargs: False)

    # Empty content: no live delivery candidate, so the seam returns the fallback text.
    text, usage, _trace = loop_mod._no_tool_final_answer(
        "", limit_ctx, trace, registry, queue.Queue(), set(), lambda _msg: None,
    )
    assert usage["execution_status"] == RESULT_INFRA_FAILED
    assert usage["reason_code"] == "acceptance_fence_unavailable"
    assert "could not start its acceptance review" in text


def test_forced_children_rail_preserves_fence_infra_failure(monkeypatch, tmp_path):
    import ouroboros.loop as loop_mod
    from ouroboros.outcomes import RESULT_INFRA_FAILED
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root-forced"
    registry._ctx.task_metadata = {
        "root_task_id": "root-forced",
        "budget_drive_root": str(tmp_path),
    }
    trace = {"tool_calls": [], "reasoning_notes": []}
    messages = [{"role": "user", "content": "task"}]
    limit_ctx = loop_mod._RoundLimitContext(
        messages,
        SimpleNamespace(),
        "test-model",
        "medium",
        1,
        tmp_path / "logs",
        "root-forced",
        2,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
        incoming_messages=None,
        owner_msg_seen=set(),
    )
    loop_mod._finalize_limit_ctx(limit_ctx, registry, trace)
    monkeypatch.setattr(loop_mod, "_undispositioned_children", lambda *_args: [])

    def exhausted(**_kwargs):
        registry._ctx._task_acceptance_fence_infra_failed = True
        return False

    monkeypatch.setattr(
        loop_mod,
        "_run_task_acceptance_review_once",
        exhausted,
    )
    result = loop_mod._run_forced_children_acceptance(
        registry,
        limit_ctx,
        "forced answer",
        messages,
        lambda _message: None,
        trace,
    )
    assert result is not None
    text, usage, _trace = result
    assert usage["execution_status"] == RESULT_INFRA_FAILED
    assert usage["reason_code"] == "acceptance_fence_unavailable"
    assert "could not start its acceptance review" in text


class _DeadProc:
    pid = 0

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def test_enforce_reap_confirmed_death_forgets_and_releases(monkeypatch, tmp_path):
    """Real wiring: the enforce hand-off registers the owner as not-provably-dead
    (its fence survives the sweep), and the reaper's confirmed death forgets the
    registry id and releases the fence itself."""
    from supervisor import task_reaper
    from supervisor import workers as workers_mod

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1",
    )
    monkeypatch.setattr(queue_mod, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_mod, "load_state", lambda: {})
    monkeypatch.setattr(queue_mod, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_mod, "_reap_queue", queue_mod._stdqueue.Queue())
    monkeypatch.setattr(workers_mod, "WORKERS", {
        1: SimpleNamespace(busy_task_id="root-1", proc=_DeadProc(), reaping=False),
    })
    monkeypatch.setattr(workers_mod, "respawn_worker", lambda worker_id: None)
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: SimpleNamespace(put=lambda *a, **k: None))
    monkeypatch.setattr("supervisor.task_reaper.send_with_budget", lambda *a, **k: None)
    now = time.time()
    queue_mod.RUNNING["root-1"] = {
        "task": {"id": "root-1", "type": "task", "deadline_at": "2000-01-01T00:00:00Z"},
        "started_at": now - 30, "last_heartbeat_at": now - 30, "worker_id": 1, "attempt": 1,
    }

    queue_mod.enforce_task_timeouts()

    # Popped from RUNNING into the reaper: the registry protects the fence.
    assert "root-1" not in queue_mod.RUNNING
    assert task_reaper.task_reaping_in_progress("root-1")
    assert queue_mod.gc_acceptance_fences_for_dead_owners() == []
    assert "root-1" in queue_mod.ACCEPTANCE_FENCES

    while not queue_mod._reap_queue.empty():
        queue_mod._reap_timed_out_task(queue_mod._reap_queue.get_nowait())

    # Confirmed death: the reaper forgot the id and released the fence.
    assert not task_reaper.task_reaping_in_progress("root-1")
    assert "root-1" not in queue_mod.ACCEPTANCE_FENCES


def test_wedged_owner_orphan_heal_forgets_and_releases(monkeypatch, tmp_path):
    """A wedged (kill-unconfirmed) owner keeps its fence while possibly alive;
    when the orphaned-running sweep later terminalizes the provably-dead task,
    the registry id is forgotten and the fence is released."""
    from supervisor import task_reaper
    from ouroboros.task_results import (
        STATUS_FAILED,
        STATUS_RUNNING,
        load_task_result,
        write_task_result,
    )
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-w", task_id="root-w",
    )
    task_reaper.note_task_reaping("root-w")  # wedged hold: kill never confirmed
    try:
        assert queue_mod.gc_acceptance_fences_for_dead_owners() == []
        assert "root-w" in queue_mod.ACCEPTANCE_FENCES

        monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
        write_task_result(
            tmp_path, "root-w", STATUS_RUNNING,
            result="held reaping, task left running", ts="2026-05-28T00:00:00+00:00",
        )
        (tmp_path / "state").mkdir(exist_ok=True)
        (tmp_path / "state" / "queue_snapshot.json").write_text(
            '{"pending": [], "running": []}', encoding="utf-8",
        )
        events = tmp_path / "logs" / "events.jsonl"
        append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": "root-w"})
        append_jsonl(events, {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})

        healed = reconcile_orphaned_running_tasks(tmp_path)

        assert healed == 1
        assert load_task_result(tmp_path, "root-w")["status"] == STATUS_FAILED
        assert not task_reaper.task_reaping_in_progress("root-w")
        assert "root-w" not in queue_mod.ACCEPTANCE_FENCES
    finally:
        task_reaper._forget_task_reaping("root-w")


def test_reaper_exception_path_preserves_fence_custody(monkeypatch, tmp_path):
    """An early reaper escape cannot make a potentially-live owner look dead."""
    from supervisor import task_reaper

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    task_reaper.note_task_reaping("root-x")
    queue_mod.ACCEPTANCE_FENCES["root-x"] = {
        "task_id": "root-x",
        "root_task_id": "root-x",
        "token": "fence-x",
        "status": "active",
    }
    try:
        # Invalid worker_id raises before any kill can prove the worker dead.
        task_reaper.reap_queue.put(
            {"worker_id": "not-an-int", "task_id": "root-x"}
        )
        task_reaper.ensure_reaper_started()
        deadline = time.time() + 10
        while time.time() < deadline and task_reaper.reap_queue.unfinished_tasks:
            time.sleep(0.05)
        assert task_reaper.task_reaping_in_progress("root-x")
        assert queue_mod.gc_acceptance_fences_for_dead_owners() == []
        assert "root-x" in queue_mod.ACCEPTANCE_FENCES
    finally:
        task_reaper._forget_task_reaping("root-x")


def test_chat_turn_liveness_form(monkeypatch):
    """None agent -> not live; busy agent -> live with its task id; idle -> not live."""
    from supervisor import workers

    monkeypatch.setattr(workers, "_chat_agent", None)
    assert workers.chat_turn_liveness() == (False, None, None)
    monkeypatch.setattr(
        workers, "_chat_agent",
        SimpleNamespace(_busy=True, _current_task_id="root-1", _last_activity_ts=42.0),
    )
    assert workers.chat_turn_liveness() == (True, "root-1", 42.0)
    workers._chat_agent._busy = False
    assert workers.chat_turn_liveness() == (False, None, None)
