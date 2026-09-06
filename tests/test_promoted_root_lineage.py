"""Chat-promoted roots retain host lineage through the ordinary writers."""

from types import SimpleNamespace

from ouroboros.agent import OuroborosAgent
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.task_results import load_task_result, resolve_task_lineage, write_task_result
from supervisor import workers


def test_promoted_payload_survives_running_terminal_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    enqueued = []
    ctx = SimpleNamespace(enqueue_task=lambda task: enqueued.append(task) or task,
                          persist_queue_snapshot=lambda **kwargs: True,
                          load_state=lambda: {"owner_chat_id": 1})
    outcome = workers.promote_chat_to_task({"task_id": "promoted", "objective": "Write the report",
                                          "workspace": "none", "chat_id": 1}, ctx)
    assert outcome["status"] == "scheduled"
    task = enqueued[0]
    assert task["root_task_id"] == "promoted" and task["delegation_role"] == "root"
    assert task.get("actor_id") is None
    write_task_result(tmp_path, "promoted", "scheduled", root_task_id="promoted", delegation_role="root")
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    OuroborosAgent._persist_running_record(SimpleNamespace(env=env), task)
    running = load_task_result(tmp_path, "promoted")
    assert running["root_task_id"] == "promoted" and running["delegation_role"] == "root"
    _store_task_result(env, task, "The report is ready.", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    terminal = load_task_result(tmp_path, "promoted")
    assert terminal["status"] == "completed"
    assert terminal["root_task_id"] == "promoted" and terminal["delegation_role"] == "root"
    # The reaper copies the admitted task and changes the physical identity.
    retry = {**task, "id": "retried", "original_task_id": "promoted", "timeout_retry_from": "promoted"}
    lineage = resolve_task_lineage(retry["id"], root_task_id=retry["root_task_id"],
                                  delegation_role=retry["delegation_role"],
                                  original_task_id=retry["original_task_id"], timeout_retry_from=retry["timeout_retry_from"])
    assert lineage["is_retry_root_attempt"] is True and lineage["is_root_task"] is True
