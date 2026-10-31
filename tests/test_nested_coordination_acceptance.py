"""Production-shaped deterministic checks for host-visible depth-three trees.

The real control plane schedules and assigns a depth-three chain; the root's host-visible
subtree and its depth summary (``depth_evidence.build_depth_summary``, which the swarm
rollup embeds) read the canonical results; a saturated pool waits for a worker slot; a real
depth refusal reads as a capability reduction; split-root receipts reconcile in host time
order. The per-root acceptance-review wallet, its capacity projection and the acceptance
subtree snapshot are retired with task acceptance.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from ouroboros import task_tree_ledger
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.depth_evidence import build_depth_summary
from ouroboros.headless import copy_child_task_result
from ouroboros.outcome_receipt_store import merge_verification_receipts
from ouroboros.outcomes import latest_unreconciled_failed_receipt
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SCHEDULED,
    load_task_result,
    write_task_result,
)
from ouroboros.task_status import find_child_tasks
from ouroboros.tools import control
from ouroboros.tools.registry import ToolContext
from supervisor import (
    events,
    queue as queue_module,
    state as state_module,
    workers,
)
from tests.worker_pool_support import affirm_ready


def _subtree(drive_root, root_task_id="root"):
    """The root's host-visible subtree from the canonical results (the swarm rollup's read)."""
    return find_child_tasks(drive_root, parent_task_id=root_task_id, root_task_id=root_task_id,
                            exclude_task_id=root_task_id, scope="subtree", materialize_artifacts=False)


def test_depth3_control_plane_reaches_the_root_depth_summary(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    settings = {
        "OUROBOROS_SUBAGENTS": json.dumps(
            {
                "enabled": True,
                "items": [
                    {
                        "subagent_id": "api-depth",
                        "name": "Depth actor",
                        "recommended_use": (
                            "Deterministic nested control-plane fixture."
                        ),
                        "route": {
                            "kind": "api_model",
                            "target_id": "openai/gpt-5.6-sol",
                        },
                        "effort": "high",
                    }
                ],
            }
        )
    }

    pending = []
    running = {}
    delivered = []
    emitted = []

    class WorkerQueue:
        def put(self, task):
            # Freeze what the worker really received. A shallow alias could make
            # later mutation hide an assignment/copy-back regression.
            if task.get("type") != "worker_quiescence":  # an idle-surplus probe is no task
                delivered.append(copy.deepcopy(task))

    worker_map = {
        worker_id: affirm_ready(SimpleNamespace(
            wid=worker_id,
            busy_task_id=None,
            reaping=False,
            in_q=WorkerQueue(),
        ))
        for worker_id in (1, 2, 3)
    }

    class SupervisorContext:
        DRIVE_ROOT = tmp_path
        PENDING = pending
        RUNNING = running
        WORKERS = worker_map

        def load_state(self):
            return {}

        def enqueue_task(self, task):
            return queue_module.enqueue_task(task)

        def persist_queue_snapshot(self, reason=""):
            return queue_module.persist_queue_snapshot(reason=reason)

        def send_with_budget(self, *_args, **_kwargs):
            return None

    supervisor_context = SupervisorContext()

    class ImmediateEventQueue:
        def put_nowait(self, event):
            # Preserve the real schedule_subagent event construction. Only the
            # asynchronous transport is collapsed into this deterministic call.
            emitted.append(copy.deepcopy(event))
            events._handle_schedule_task(event, supervisor_context)

    event_queue = ImmediateEventQueue()

    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "6")
    monkeypatch.setattr(control, "load_settings", lambda: settings)
    monkeypatch.setattr(task_tree_ledger, "DATA_DIR", tmp_path)

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(workers, "WORKERS", worker_map)
    monkeypatch.setattr(workers, "load_state", lambda: {})

    monkeypatch.setattr(
        state_module,
        "budget_remaining",
        lambda *_args, **_kwargs: 100.0,
    )
    for key, value in {"DRIVE_ROOT": tmp_path, "PENDING": pending, "RUNNING": running,
                       "QUEUE_SNAPSHOT_PATH": tmp_path / "state/queue_snapshot.json",
                       "QUEUE_SEQ_COUNTER_REF": {"value": 0}, "ADMISSION_RESERVATIONS": {},
                       "ACCEPTANCE_FENCES": {}}.items():
        monkeypatch.setattr(queue_module, key, value)
    monkeypatch.setattr(queue_module, "BUDGET_ROOT_FENCES", {})

    root_contract = build_task_contract(
        {
            "delegation_budget": {
                "depth_remaining": 3,
            }
        }
    )
    write_task_result(
        tmp_path,
        "root",
        STATUS_RUNNING,
        root_task_id="root",
        delegation_role="root",
        task_contract=root_contract,
    )

    def actor_context(
        task_id,
        depth,
        task_contract,
        drive_root,
        task=None,
    ):
        ctx = ToolContext(
            repo_dir=repo,
            drive_root=Path(drive_root),
            budget_drive_root=str(tmp_path),
        )
        ctx.task_id = task_id
        ctx.task_depth = depth
        ctx.event_queue = event_queue
        ctx.task_contract = task_contract

        if task is not None:
            ctx.task_constraint = normalize_task_constraint(
                task.get("task_constraint")
            )

        ctx.task_metadata = {
            "root_task_id": "root",
            "parent_task_id": str(
                (task or {}).get("parent_task_id") or ""
            ),
            "delegation_role": (
                "root" if depth == 0 else "subagent"
            ),
            "budget_drive_root": str(tmp_path),
            "task_contract": task_contract,
            "configured_subagent": copy.deepcopy(
                (task or {}).get("configured_subagent") or {}
            ),
        }
        return ctx

    def schedule_and_assign(
        parent_id,
        parent_depth,
        parent_contract,
        parent_drive,
        parent_task=None,
    ):
        previous_event_count = len(emitted)
        previous_delivery_count = len(delivered)

        result = control._schedule_task(
            actor_context(
                parent_id,
                parent_depth,
                parent_contract,
                parent_drive,
                parent_task,
            ),
            subagent_id="api-depth",
            objective=f"depth-{parent_depth + 1}",
            expected_output="typed handoff",
            memory_mode="empty",
        )

        assert "Subagent request queued" in result
        assert len(emitted) == previous_event_count + 1

        task_id = emitted[-1]["task_id"]
        admitted = load_task_result(tmp_path, task_id)

        assert admitted["status"] == STATUS_SCHEDULED
        assert (
            admitted["depth_provenance"]["achieved_depth"]
            is None
        )

        workers.assign_tasks()

        assert len(delivered) == previous_delivery_count + 1
        assigned = delivered[-1]
        assert assigned["id"] == task_id
        return assigned

    l1 = schedule_and_assign(
        "root",
        0,
        root_contract,
        tmp_path,
    )
    l2 = schedule_and_assign(
        l1["id"],
        1,
        l1["task_contract"],
        l1["drive_root"],
        l1,
    )
    l3 = schedule_and_assign(
        l2["id"],
        2,
        l2["task_contract"],
        l2["drive_root"],
        l2,
    )

    assert [task["depth"] for task in (l1, l2, l3)] == [1, 2, 3]
    assert [
        task["parent_task_id"] for task in (l1, l2, l3)
    ] == [
        "root",
        l1["id"],
        l2["id"],
    ]
    assert {
        task["root_task_id"] for task in (l1, l2, l3)
    } == {"root"}

    for depth, task in enumerate((l1, l2, l3), start=1):
        expected_provenance = {
            "requested_depth": 3,
            "permitted_depth": 3,
            "attempted_depth": depth,
            "achieved_depth": depth,
        }

        assert task["depth_provenance"] == expected_provenance
        assert (
            task["task_contract"]["delegation_budget"][
                "depth_provenance"
            ]
            == expected_provenance
        )
        assert (
            task["task_contract"]["delegation_budget"][
                "depth_remaining"
            ]
            == 3 - depth
        )

        canonical = load_task_result(tmp_path, task["id"])
        assert canonical["status"] == STATUS_RUNNING
        assert canonical["depth_provenance"] == expected_provenance
        assert (
            canonical["task_contract"]["delegation_budget"][
                "depth_provenance"
            ]
            == expected_provenance
        )

    assert (
        l3["task_contract"]["delegation_budget"]["may_delegate"]
        is False
    )

    def terminal_copy(task):
        write_task_result(
            Path(task["drive_root"]),
            task["id"],
            STATUS_COMPLETED,
            parent_task_id=task["parent_task_id"],
            root_task_id="root",
            delegation_role="subagent",
            task_contract=task["task_contract"],
            depth_provenance=task["depth_provenance"],
            result=f"done-{task['id']}",
        )

        assert copy_child_task_result(tmp_path, task) is not None

        canonical = load_task_result(tmp_path, task["id"])
        assert canonical["status"] == STATUS_COMPLETED
        assert canonical["depth_provenance"] == (
            canonical["task_contract"]["delegation_budget"][
                "depth_provenance"
            ]
        )

    terminal_copy(l3)

    early = {row["task_id"]: row["status"] for row in _subtree(tmp_path)}
    assert early == {
        l1["id"]: STATUS_RUNNING,
        l2["id"]: STATUS_RUNNING,
        l3["id"]: STATUS_COMPLETED,
    }

    terminal_copy(l2)
    terminal_copy(l1)
    running.clear()
    for worker in worker_map.values():
        worker.busy_task_id = None

    terminal_rows = _subtree(tmp_path)
    assert {row["task_id"] for row in terminal_rows} == {l1["id"], l2["id"], l3["id"]}
    assert all(row["status"] == STATUS_COMPLETED for row in terminal_rows)
    assert build_depth_summary(root_contract, terminal_rows) == {
        "requested_depth": 3,
        "permitted_depth": 3,
        "attempted_depth": 3,
        "achieved_depth": 3,
        "status": "achieved",
        "host_visible_only": True,
    }


def test_depth3_waits_for_worker_slot_then_uses_active_cap_reservation(
    tmp_path, monkeypatch,
):
    delivered = []

    class WorkerQueue:
        def put(self, task):
            if task.get("type") != "worker_quiescence":  # an idle-surplus probe is no task
                delivered.append(copy.deepcopy(task))

    # Root + two ancestors + unrelated work occupy a four-worker install.
    worker_map = {
        idx: affirm_ready(SimpleNamespace(
            wid=idx,
            busy_task_id=task_id,
            reaping=False,
            in_q=WorkerQueue(),
        ))
        for idx, task_id in enumerate(("root", "l1", "l2", "other"))
    }
    running = {
        "root": {"task": {
            "id": "root", "root_task_id": "root", "delegation_role": "root",
        }},
        "l1": {"task": {
            "id": "l1", "root_task_id": "root",
            "parent_task_id": "root", "delegation_role": "subagent",
        }},
        "l2": {"task": {
            "id": "l2", "root_task_id": "root",
            "parent_task_id": "l1", "delegation_role": "subagent",
        }},
        "other": {"task": {
            "id": "other", "root_task_id": "other", "delegation_role": "root",
        }},
    }
    provenance = {
        "requested_depth": 3,
        "permitted_depth": 3,
        "attempted_depth": 3,
        "achieved_depth": None,
    }
    contract = build_task_contract({"delegation_budget": {
        "depth_remaining": 0,
        "depth_provenance": provenance,
    }})
    l3 = {
        "id": "l3",
        "type": "task",
        "chat_id": 0,
        "description": "depth three",
        "objective": "depth three",
        "expected_output": "typed result",
        "depth": 3,
        "root_task_id": "root",
        "parent_task_id": "l2",
        "delegation_role": "subagent",
        "drive_root": str(tmp_path),
        "child_drive_root": str(tmp_path),
        "budget_drive_root": str(tmp_path),
        "task_contract": contract,
        "depth_provenance": provenance,
    }
    pending = [l3]
    write_task_result(
        tmp_path,
        "l3",
        STATUS_SCHEDULED,
        parent_task_id="l2",
        root_task_id="root",
        delegation_role="subagent",
        task_contract=contract,
        depth_provenance=provenance,
        result="Subagent accepted and scheduled.",
    )

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "4")
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "2")
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    monkeypatch.setattr(workers, "MAX_WORKERS", 4)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(workers, "WORKERS", worker_map)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(
        state_module, "budget_remaining", lambda *_args, **_kwargs: 100.0,
    )
    monkeypatch.setattr(queue_module, "PENDING", pending)
    monkeypatch.setattr(queue_module, "RUNNING", running)
    monkeypatch.setattr(queue_module, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")

    # Saturation preserves attempted-but-not-achieved evidence.
    workers.assign_tasks()
    assert [task["id"] for task in pending] == ["l3"]
    assert delivered == []
    scheduled = load_task_result(tmp_path, "l3")
    assert scheduled["status"] == STATUS_SCHEDULED
    assert scheduled["depth_provenance"]["achieved_depth"] is None
    assert queue_module._has_pending_descendant("root") is True
    assert queue_module._has_pending_descendant("l2") is True

    # Only an explicit unrelated-task settlement creates capacity; this test
    # does not promise progress for a permanently saturated ancestry-only pool.
    running.pop("other")
    worker_map[3].busy_task_id = None
    workers.assign_tasks()

    assert pending == []
    assert delivered[-1]["id"] == "l3"
    assert delivered[-1]["depth_provenance"]["achieved_depth"] == 3
    assigned = load_task_result(tmp_path, "l3")
    assert assigned["status"] == STATUS_RUNNING
    assert assigned["depth_provenance"]["achieved_depth"] == 3
    assert worker_map[3].busy_task_id == "l3"


def test_real_over_cap_refusal_reaches_the_root_depth_summary(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "2")
    monkeypatch.setattr(
        control,
        "load_settings",
        lambda: {
            "OUROBOROS_SUBAGENTS": json.dumps({
                "enabled": True,
                "items": [{
                    "subagent_id": "api-depth",
                    "name": "Depth actor",
                    "recommended_use": "Typed depth refusal fixture.",
                    "route": {
                        "kind": "api_model",
                        "target_id": "openai/gpt-5.6-sol",
                    },
                    "effort": "high",
                }],
            }),
        },
    )
    root_contract = build_task_contract({
        "delegation_budget": {"depth_remaining": 3},
    })
    parent_provenance = {
        "requested_depth": 3,
        "permitted_depth": 2,
        "attempted_depth": 2,
        "achieved_depth": 2,
    }
    parent_contract = build_task_contract({
        "parent_task_id": "depth-one",
        "root_task_id": "root",
        "delegation_role": "subagent",
        "delegation_budget": {
            "may_delegate": False,
            "depth_remaining": 0,
            "depth_provenance": parent_provenance,
        },
    })
    write_task_result(
        tmp_path,
        "depth-one",
        STATUS_COMPLETED,
        parent_task_id="root",
        root_task_id="root",
        delegation_role="subagent",
        depth_provenance={
            **parent_provenance,
            "attempted_depth": 1,
            "achieved_depth": 1,
        },
        result="depth one complete",
    )
    write_task_result(
        tmp_path,
        "depth-two",
        STATUS_RUNNING,
        parent_task_id="depth-one",
        root_task_id="root",
        delegation_role="subagent",
        task_contract=parent_contract,
        depth_provenance=parent_provenance,
    )
    parent_ctx = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
    )
    parent_ctx.task_id = "depth-two"
    parent_ctx.task_depth = 2
    parent_ctx.task_contract = parent_contract
    parent_ctx.task_metadata = {
        "root_task_id": "root",
        "parent_task_id": "depth-one",
        "delegation_role": "subagent",
        "budget_drive_root": str(tmp_path),
        "task_contract": parent_contract,
    }

    refused = control._schedule_task(
        parent_ctx,
        subagent_id="api-depth",
        objective="attempt depth three",
        expected_output="typed refusal",
    )
    assert "subtask_depth_limit" in refused
    refused_id = refused.split("task_id=", 1)[1].split(";", 1)[0]
    assert load_task_result(tmp_path, refused_id)["status"] == STATUS_FAILED
    write_task_result(
        tmp_path,
        "depth-two",
        STATUS_COMPLETED,
        parent_task_id="depth-one",
        root_task_id="root",
        delegation_role="subagent",
        task_contract=parent_contract,
        depth_provenance=parent_provenance,
        result="depth two complete",
    )

    statuses = _subtree(tmp_path)
    assert {row["task_id"] for row in statuses} == {
        "depth-one", "depth-two", refused_id,
    }
    assert build_depth_summary(root_contract, statuses) == {
        "requested_depth": 3,
        "permitted_depth": 2,
        "attempted_depth": 3,
        "achieved_depth": 2,
        "status": "capability_reduced",
        "host_visible_only": True,
    }


def test_split_root_receipts_reconcile_in_host_timestamp_order():
    old_pass = {
        "criterion_id": "claim_1", "status": "pass",
        "contract_kind": "explicit_command", "ts": "2026-01-01T00:00:01+00:00",
    }
    new_fail = {
        "criterion_id": "claim_1", "status": "fail",
        "contract_kind": "explicit_command", "ts": "2026-01-01T00:00:02+00:00",
    }
    merged = merge_verification_receipts([new_fail], [old_pass])
    assert merged == [old_pass, new_fail]
    assert latest_unreconciled_failed_receipt(merged) == new_fail

    old_fail = {**new_fail, "ts": "2026-01-01T00:00:01+00:00"}
    new_pass = {**old_pass, "ts": "2026-01-01T00:00:02+00:00"}
    merged = merge_verification_receipts([new_pass], [old_fail])
    assert merged == [old_fail, new_pass]
    assert latest_unreconciled_failed_receipt(merged) is None
