"""The real post-task producer keeps the supervisor's existing event endpoint."""
from __future__ import annotations

import json
import multiprocessing
import pathlib
import queue
from types import SimpleNamespace

import pytest
from multiprocessing.managers import SyncManager
from tests._cancel_intents_shared import DeliveryBus


class _BusManager(SyncManager):
    """A manager whose Queue is the membership-aware fixture bus (what lane A's bus exposes)."""


_BusManager.register("Queue", DeliveryBus)

pytestmark = pytest.mark.serial


def _finish_in_worker(root_text, task, event_queue, report_queue, attach):
    from ouroboros import agent_task_pipeline, post_task_evolution
    from supervisor import event_observation, terminal_delivery, workers

    manager_calls = []

    class NoManager:
        def Manager(self):
            manager_calls.append("create")
            raise AssertionError("a producer tried to create the supervisor bus")

    def no_event_manager(*args, **kwargs):
        manager_calls.append("create-event-manager")
        raise AssertionError("a producer tried to create the observed supervisor bus")

    workers._EVENT_Q = None
    workers._EVENT_Q_MANAGER = None
    workers._get_ctx = lambda: NoManager()
    event_observation.EventQueueManager = no_event_manager
    root = pathlib.Path(root_text)
    env = SimpleNamespace(drive_root=root, repo_dir=pathlib.Path(__file__).parents[1],
                          drive_path=lambda rel: root / rel)
    # The paid stages do nothing here: exercise the actual final callback without
    # asking a provider or replacing the checkpoint/projection/outbox writers.
    agent_task_pipeline._run_memory_fallback_draft = lambda *args, **kwargs: ""
    post_task_evolution.maybe_promote = lambda *args, **kwargs: None
    agent_task_pipeline._run_post_task_processing_async(
        env, task, {}, {}, root / "logs", blocking=True,
        event_queue=event_queue if attach else None,
    )
    report_queue.put((terminal_delivery.pending_deliveries(root), manager_calls))


def _seed_project(root):
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    project = create_project(root, "delivery", name="Delivery")
    bind_task_to_project(root, "root-producer", project["id"], project["chat_id"],
                         origin={"absent": "system"})
    task = {"id": "root-producer", "root_task_id": "root-producer", "title": "Retain answer",
            "project_id": project["id"], "chat_id": project["chat_id"]}
    row = write_task_result(root, task["id"], "completed", root_task_id=task["id"],
                            project_id=project["id"], chat_id=project["chat_id"],
                            result="The complete answer survives the worker.",
                            terminal_origin="model_final",
                            root_phase_checkpoint={"post_task_synthesis": "running"})
    append_terminal_task_projection(root, task["id"], task, row,
                                    {"status": "completed", "chat_id": project["chat_id"]})
    return task


@pytest.mark.parametrize("attach", [True, False])
def test_daemon_post_task_uses_injected_bus_or_keeps_delivery_owed(tmp_path, monkeypatch, attach):
    from supervisor import terminal_delivery as td

    task = _seed_project(tmp_path)
    context = multiprocessing.get_context("spawn")
    # The real bus (supervisor.event_observation) carries delivery membership; replay admits a
    # copy only through it, so the injected fixture bus serves the same protocol (DeliveryBus).
    with _BusManager(ctx=context) as manager:
        event_queue, report_queue = manager.Queue(), manager.Queue()
        worker = context.Process(target=_finish_in_worker,
                                 args=(str(tmp_path), task, event_queue, report_queue, attach))
        worker.daemon = True
        worker.start()
        try:
            owed, manager_calls = report_queue.get(timeout=30)
            worker.join(timeout=10)
            assert worker.exitcode == 0
            assert manager_calls == []
            assert len(owed) == 1
            expected = owed[0]
            assert expected["delivery_id"] == "project-completion:root-producer"
            assert expected["progress_meta"]["completion_answer"] == "The complete answer survives the worker."
            if attach:
                sent = event_queue.get(timeout=5)
                assert sent["delivery_id"] == expected["delivery_id"]
                assert sent["progress_meta"] == expected["progress_meta"]
            else:
                with pytest.raises(queue.Empty):
                    event_queue.get(timeout=0.05)
                # No attached bus means unknown membership, not a failed put. Cache debt
                # remains discoverable and replay can offer it immediately on a known bus.
                from ouroboros.task_results import load_task_result
                ready = load_task_result(tmp_path, task["id"])["canonical_terminal_projection_ready"]
                assert ready["main_delivery"]["attempts"] == 0
                assert not ready["main_delivery"].get("last_attempt_at")
                assert td.replay_pending_deliveries(tmp_path, event_queue=event_queue) == []
                sent = event_queue.get(timeout=5)
                assert sent["progress_meta"] == expected["progress_meta"]
            registry = json.loads((tmp_path / "state/terminal_deliveries.json").read_text(encoding="utf-8"))
            assert expected["delivery_id"] in registry["pending"]
            assert expected["delivery_id"] not in registry["delivered"]
            td.register_delivery(tmp_path, expected["delivery_id"])
            assert td.replay_pending_deliveries(tmp_path, event_queue=event_queue) == []
            with pytest.raises(queue.Empty):
                event_queue.get(timeout=0.05)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)


def test_a_producer_can_use_an_existing_bus_without_creating_it(monkeypatch):
    from supervisor import workers

    endpoint = object()
    monkeypatch.setattr(workers, "_EVENT_Q", endpoint)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    assert workers.get_event_q(create=False) is endpoint
    monkeypatch.setattr(workers, "_EVENT_Q", None)
    with pytest.raises(RuntimeError, match="not attached"):
        workers.get_event_q(create=False)
