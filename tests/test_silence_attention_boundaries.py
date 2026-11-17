"""Silence fixtures, and real child work answers its source's silence (no host role is minted)."""

from types import SimpleNamespace

import pytest

from ouroboros.task_results import write_task_result
from supervisor import queue, workers
from tests.test_silence_attention import attention as _attention, host as _host, observation

attention = _attention
host = _host


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This test permits no network, subprocess or process signal")

    for path in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
                 "subprocess.Popen", "os.kill", "os.killpg"):
        monkeypatch.setattr(path, forbidden, raising=False)


def test_missing_platform_signal_symbol_is_still_forbidden(monkeypatch):
    import os

    monkeypatch.delattr(os, "killpg", raising=False)  # Windows has no such symbol.
    no_external_effects.__wrapped__(monkeypatch)
    with pytest.raises(AssertionError, match="no network, subprocess or process signal"):
        os.killpg(0, 0)


def _child(attention):
    """An ordinary helper of the silent root: real child work, not a host role."""
    attention.tick()
    attention.flush()
    return {"id": "child-work", "type": "task", "parent_task_id": "root", "root_task_id": "root",
            "delegation_role": "subagent", "chat_id": 7, "metadata": {}}


def _running(attention, task):
    if task in queue.PENDING:
        queue.PENDING.remove(task)
    attention.running[task["id"]] = {
        "task": task, "attempt": 1, "worker_id": 1,
        "started_at": 1000.0, "last_progress_at": 1000.0,
    }


@pytest.mark.parametrize("retired", [True, False])
def test_child_settlement_rearms_source_for_real_work(attention, monkeypatch, retired):
    from supervisor.events_task_done import _handle_task_done
    from ouroboros.utils import append_jsonl

    child = _child(attention)
    episode = observation(attention)["episode_id"]
    _running(attention, child)
    write_task_result(attention.root, child["id"], "completed", metadata=child["metadata"],
                      parent_task_id="root", root_task_id="root", delegation_role="subagent",
                      result="Inspection finished; source custody remains.")
    if retired:
        attention.running.pop(child["id"])
    delivered = []
    ctx = SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root,
                          append_jsonl=append_jsonl, send_with_budget=lambda *a, **k: True,
                          persist_queue_snapshot=queue.persist_queue_snapshot,
                          bridge=SimpleNamespace(push_log=delivered.append))
    monkeypatch.setattr("ouroboros.headless.terminal_task_files_ready", lambda *a, **k: True)
    _handle_task_done({"type": "task_done", "task_id": child["id"], "worker_id": 1,
                       "status": "completed", "task_type": "task", "_files_prepared_attempt": 1}, ctx)
    assert child["id"] not in attention.running
    assert any(row.get("type") == "task_done" and row["task_id"] == child["id"] for row in delivered)
    assert attention.meta["last_progress_at"] == attention.clock[0]
    attention.tick()
    attention.flush()
    attention.clock[0] += 4000
    attention.tick()
    attention.flush()
    assert observation(attention)["episode_id"] != episode
    assert queue.PENDING == []


def test_pending_and_folded_child_activity_keep_their_own_progress(attention, monkeypatch):
    from supervisor import cognitive_operations, events_budget, events_chat_delivery

    child = _child(attention)
    _running(attention, child)
    # No prior observation: the outstanding child's event must not hide new source silence.
    attention.meta.pop("silence_attention")
    attention.bus.put({"type": "send_message", "task_id": child["id"], "chat_id": 7,
                       "text": "Inspection progress", "is_progress": True})
    attention.tick()
    assert not attention.meta.get("silence_attention")
    for module in (cognitive_operations, events_budget, events_chat_delivery):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: attention.clock[0]))
    attention.drain()
    assert attention.running[child["id"]]["last_progress_at"] == attention.clock[0]
    assert attention.meta["last_progress_at"] == 1000.0
    assert queue._subtree_progressing("root", attention.clock[0], 900) is True


