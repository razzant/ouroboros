"""The subagent lifecycle after admission: assignment, mirroring, timeout and restart.

Split out of ``tests/test_task_status_flow.py`` by theme: the artifacts a readonly
workspace subagent does not get, the contract fields the queue snapshot preserves, the
assignment that mirrors running status to the parent drive and honors caps and depth
reservations, the hard-timeout retry, the absolute deadline, and the restart latch.
"""

import json
import pathlib
import time
from types import SimpleNamespace


def test_handle_task_done_skips_workspace_readonly_subagent_artifacts(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.headless as headless
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    calls = []

    def fake_copy(root, task):
        calls.append(("copy", task["id"]))
        return write_task_result(pathlib.Path(root), task["id"], STATUS_COMPLETED, result="child handoff")

    monkeypatch.setattr(headless, "copy_child_task_result", fake_copy)

    def fake_finalize(root, task):
        calls.append(("finalize", task["id"]))
        write_task_result(
            pathlib.Path(root),
            task["id"],
            STATUS_COMPLETED,
            result="done",
            artifact_status="failed",
            artifact_bundle={"status": "failed", "artifacts": []},
        )

    monkeypatch.setattr(headless, "finalize_task_artifacts", fake_finalize)
    pushed = []

    worker = SimpleNamespace(busy_task_id="workspace-child")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "workspace-child": {
                "task": {
                    "id": "workspace-child",
                    "chat_id": 1,
                    "delegation_role": "subagent",
                    "role": "workspace-reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "workspace_root": str(tmp_path / "workspace"),
                    "task_constraint": {"mode": "local_readonly_subagent"},
                }
            }
        },
        WORKERS={3: worker},
        bridge=SimpleNamespace(push_log=lambda payload: pushed.append(payload)),
        send_with_budget=lambda *args, **kwargs: None,
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "workspace-child", "worker_id": 3, "task_type": "task"}, ctx)

    assert ("copy", "workspace-child") in calls
    assert ("finalize", "workspace-child") not in calls
    assert pushed[-1]["status"] == STATUS_COMPLETED
    assert pushed[-1]["artifact_status"] is None


def test_queue_snapshot_preserves_subagent_contract_fields(tmp_path, monkeypatch):
    from supervisor import state as state_mod
    from supervisor import queue as queue_module

    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)

    queue_module.PENDING.append(
        {
            "id": "sub1",
            "type": "task",
            "chat_id": 1,
            "text": "subagent prompt",
            "description": "Review shared surface",
            "objective": "Review shared surface",
            "expected_output": "Distinct handoff table",
            "constraints": "No writes",
            "role": "security reviewer",
            "context": "same context",
            "parent_task_id": "parent1",
            "root_task_id": "root1",
            "session_id": "sess1",
            "actor_id": "subagent:security",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "allowed_resources": {"web": False, "network": False},
            "deadline_at": "2026-06-04T12:00:00Z",
            "task_contract": {
                "schema_version": 1,
                "objective": "Review shared surface",
                "allowed_resources": {"web": False, "network": False},
                "resource_policy": {
                    "protected_artifacts": [
                        {
                            "id": "reference",
                            "role": "black_box_reference",
                            "paths": ["reference.bin"],
                            "allow": ["execute"],
                        }
                    ]
                },
                "deadline_at": "2026-06-04T12:00:00Z",
            },
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "sub1" / "data"),
            "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
        }
    )

    queue_module.persist_queue_snapshot(reason="test")
    saved = json.loads(snapshot_path.read_text(encoding="utf-8"))["pending"][0]["task"]
    assert saved["objective"] == "Review shared surface"
    assert saved["expected_output"] == "Distinct handoff table"
    assert saved["constraints"] == "No writes"
    assert saved["role"] == "security reviewer"
    assert saved["allowed_resources"] == {"web": False, "network": False}
    assert saved["deadline_at"] == "2026-06-04T12:00:00Z"
    assert saved["task_contract"]["allowed_resources"] == {"web": False, "network": False}
    assert saved["task_contract"]["resource_policy"]["protected_artifacts"][0]["id"] == "reference"
    assert pathlib.Path(saved["child_drive_root"]).parts[-4:] == ("state", "headless_tasks", "sub1", "data")
    assert saved["task_constraint"]["mode"] == "local_readonly_subagent"

    queue_module.PENDING.clear()
    assert queue_module.restore_pending_from_snapshot(max_age_sec=900) == 1
    restored = queue_module.PENDING[0]
    assert restored["objective"] == "Review shared surface"
    assert restored["expected_output"] == "Distinct handoff table"
    assert restored["constraints"] == "No writes"
    assert restored["role"] == "security reviewer"
    assert restored["allowed_resources"] == {"web": False, "network": False}
    assert restored["deadline_at"] == "2026-06-04T12:00:00Z"
    assert restored["task_contract"]["allowed_resources"] == {"web": False, "network": False}
    assert restored["task_contract"]["resource_policy"]["protected_artifacts"][0]["paths"] == ["reference.bin"]
    assert pathlib.Path(restored["child_drive_root"]).parts[-4:] == ("state", "headless_tasks", "sub1", "data")
    assert restored["task_constraint"]["mode"] == "local_readonly_subagent"


def test_assign_tasks_mirrors_running_subagent_status_to_parent_drive(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, load_task_result
    from supervisor import queue as queue_module
    from supervisor import state as state_module
    from supervisor import workers as workers_module

    child_drive = tmp_path / "state" / "headless_tasks" / "childrun" / "data"
    child_drive.mkdir(parents=True)
    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(dict(task))

    task = {
        "id": "childrun",
        "type": "task",
        "chat_id": 1,
        "description": "Inspect handoff",
        "objective": "Inspect handoff",
        "expected_output": "Findings",
        "parent_task_id": "parent123",
        "root_task_id": "root123",
        "session_id": "sess123",
        "actor_id": "subagent:reviewer",
        "delegation_role": "subagent",
        "role": "reviewer",
        "memory_mode": "forked",
        "drive_root": str(child_drive),
        "child_drive_root": str(child_drive),
        "budget_drive_root": str(tmp_path),
        "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
        "metadata": {"root_task_id": "root123"},
    }
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", [task])
    monkeypatch.setattr(workers_module, "RUNNING", {})
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    parent_result = load_task_result(tmp_path, "childrun")
    assert parent_result["status"] == STATUS_RUNNING
    assert parent_result["child_drive_root"] == str(child_drive)
    assert parent_result["result"] == "Subagent assigned to a worker."
    assert delivered and delivered[0]["id"] == "childrun"


def test_assign_tasks_leaves_subagent_pending_when_running_cap_full(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from supervisor import state as state_module

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(task)

    pending = [{
        "id": "child2",
        "type": "task",
        "chat_id": 1,
        "description": "Wait",
        "root_task_id": "root123",
        "delegation_role": "subagent",
        "budget_drive_root": str(tmp_path),
    }]
    running = {
        "child1": {
            "task": {
                "id": "child1",
                "root_task_id": "root123",
                "delegation_role": "subagent",
            }
        }
    }
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "1")
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", pending)
    monkeypatch.setattr(workers_module, "RUNNING", running)
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    assert pending and pending[0]["id"] == "child2"
    assert delivered == []


def test_assign_tasks_honors_depth_reservation_for_first_grandchild(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from supervisor import state as state_module

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(task)

    pending = [{
        "id": "grandchild1",
        "type": "task",
        "chat_id": 1,
        "description": "Reserved depth child",
        "root_task_id": "root123",
        "parent_task_id": "child1",
        "delegation_role": "subagent",
        "budget_drive_root": str(tmp_path),
    }]
    running = {
        "child1": {
            "task": {
                "id": "child1",
                "root_task_id": "root123",
                "delegation_role": "subagent",
            }
        }
    }
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "1")
    monkeypatch.setattr(workers_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_module, "PENDING", pending)
    monkeypatch.setattr(workers_module, "RUNNING", running)
    monkeypatch.setattr(workers_module, "WORKERS", {1: SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())})
    monkeypatch.setattr(workers_module, "load_state", lambda: {})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _state, **_kwargs: 100.0)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)

    workers_module.assign_tasks()

    assert delivered and delivered[0]["id"] == "grandchild1"
    assert "grandchild1" in workers_module.RUNNING


def test_override_delegation_constraint_requires_parent_lineage(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.join_ledger import _override_delegation_constraint
    from ouroboros.tools.registry import ToolContext
    import ouroboros.task_tree_ledger as ledger

    monkeypatch.setattr(ledger, "DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "child1", STATUS_RUNNING, parent_task_id="parent1", root_task_id="root1", delegation_role="subagent")
    ledger.tree_ledger_append(
        "root1",
        "delegation_constraint",
        "child asks parent to stop fanout",
        task_id="child1",
        role="scout",
        payload={"constraint_id": "c1", "directive": "halt_fanout", "scope": {}, "rationale": "wait for evidence"},
    )
    sibling = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="sibling", task_metadata={"root_task_id": "root1"})
    child = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child1", task_metadata={"root_task_id": "root1"})
    parent = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="parent1", task_metadata={"root_task_id": "root1"})

    assert "only the parent" in _override_delegation_constraint(child, "c1", "self-clear")
    assert "only the parent" in _override_delegation_constraint(sibling, "c1", "not my constraint")
    assert _override_delegation_constraint(parent, "c1", "I gathered the evidence").startswith("OK:")
    assert ledger.open_delegation_constraints("root1") == []


def test_subagent_hard_timeout_retry_preserves_task_id(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from ouroboros.task_results import STATUS_INTERRUPTED, load_task_result

    class FakeProc:
        pid = 12345

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("already dead")

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    # Activity model: a "timed out" task is one with no real progress for the idle
    # window AND no progressing subtree (heartbeat alone is not progress). Variant A:
    # run the heavy teardown reaper synchronously (no daemon) for a deterministic test.
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(queue_module, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue_module, "get_per_call_timeout_ceiling_sec", lambda: 1)
    worker = SimpleNamespace(busy_task_id="childtimeout", proc=FakeProc(), reaping=False)
    monkeypatch.setattr(workers_module, "WORKERS", {9: worker})
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)
    child_drive = tmp_path / "child-drive"
    service_dir = child_drive / "services" / "childtimeout"
    service_dir.mkdir(parents=True)
    (service_dir / "devserver.log").write_text("READY\n", encoding="utf-8")

    queue_module.RUNNING["childtimeout"] = {
        "task": {
            "id": "childtimeout",
            "type": "task",
            "chat_id": 1,
            "delegation_role": "subagent",
            "drive_root": str(child_drive),
            "child_drive_root": str(child_drive),
            "_attempt": 1,
        },
        # idle for ~1000s, far beyond the monkeypatched idle window max(1, 1+120)=121s,
        # with no progressing subtree -> activity-based stop.
        "started_at": time.time() - 1000,
        "last_heartbeat_at": time.time() - 1000,
        "worker_id": 9,
        "attempt": 1,
    }

    queue_module.enforce_task_timeouts()
    # Drain the off-loop reaper synchronously (kill/archive/respawn).
    while not queue_module._reap_queue.empty():
        queue_module._reap_timed_out_task(queue_module._reap_queue.get_nowait())

    assert queue_module.PENDING
    retried = queue_module.PENDING[0]
    assert retried["id"] == "childtimeout"
    assert retried["_attempt"] == 2
    assert retried["timeout_retry_from"] == "childtimeout"
    assert load_task_result(tmp_path, "childtimeout")["status"] == STATUS_INTERRUPTED
    assert "childtimeout" not in queue_module.RUNNING
    assert not service_dir.exists()


def test_absolute_deadline_does_not_retry_expired_task(tmp_path, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import workers as workers_module
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    class FakeProc:
        pid = 12345

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("already dead")

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(queue_module, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue_module, "get_per_call_timeout_ceiling_sec", lambda: 1)
    worker = SimpleNamespace(busy_task_id="deadline1", proc=FakeProc(), reaping=False)
    monkeypatch.setattr(workers_module, "WORKERS", {9: worker})
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)

    queue_module.RUNNING["deadline1"] = {
        "task": {
            "id": "deadline1",
            "type": "task",
            "chat_id": 1,
            "deadline_at": "2000-01-01T00:00:00Z",
            "_attempt": 1,
        },
        # Past deadline AND idle (no progress for ~1000s): the deadline is gated through
        # idle/subtree-liveness, so an expired-but-idle task is stopped without retry.
        "started_at": time.time() - 1000,
        "last_heartbeat_at": time.time() - 1000,
        "worker_id": 9,
        "attempt": 1,
    }

    queue_module.enforce_task_timeouts()
    # Variant A: the terminal write + retry decision now happen in the off-loop reaper.
    while not queue_module._reap_queue.empty():
        queue_module._reap_timed_out_task(queue_module._reap_queue.get_nowait())

    assert queue_module.PENDING == []
    result = load_task_result(tmp_path, "deadline1")
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "deadline"
    assert result["outcome_axes"]["execution"]["reason_code"] == "deadline"


def test_handle_text_response_keeps_full_reasoning_note():
    from ouroboros.loop import _handle_text_response

    content = "A" * 500
    llm_trace = {"reasoning_notes": [], "tool_calls": []}
    _, _, updated = _handle_text_response(content, llm_trace, {})

    assert updated["reasoning_notes"] == [content]


def test_request_restart_latches_reason_until_task_end(tmp_path, monkeypatch):
    from ouroboros.tools import control_runtime as control_module

    monkeypatch.setattr(control_module, "run_cmd", lambda *args, **kwargs: "value")
    written = {}
    monkeypatch.setattr(
        control_module,
        "atomic_write_json",
        lambda path, payload: written.setdefault(str(path), payload),
    )

    class _Ctx:
        current_task_type = "task"
        last_push_succeeded = True
        pending_events = []
        pending_restart_reason = None
        repo_dir = tmp_path

        def drive_path(self, rel):
            return tmp_path / rel

    ctx = _Ctx()
    result = control_module._request_restart(ctx, "reload runtime")

    assert "Restart requested" in result
    assert ctx.pending_events == []
    assert ctx.pending_restart_reason == "reload runtime"
    assert written
