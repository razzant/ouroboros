"""Landed model-wait controls keep concurrent native actors independently owned."""

from contextlib import ExitStack
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, model_wait, owner_mailbox, server_restart
from supervisor import active_activity, workers
from supervisor.task_model_wait import handle_task_model_wait
from tests.test_model_wait import _action_for, _decision_clients
from tests.test_post_task_model_wait import phase as post_phase_fixture, until

phase = post_phase_fixture


def test_native_wait_controls_and_manual_restart_address_all_registered_actors(tmp_path, monkeypatch):
    from ouroboros import delegate_custody
    from ouroboros.task_results import write_task_result
    from supervisor import queue as task_queue

    registry = active_activity.get_direct_activity_registry()
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(task_queue, "RUNNING", {})
    monkeypatch.setattr(task_queue, "PENDING", [])
    monkeypatch.setattr(delegate_custody, "reconcile_orphaned_runs", lambda *a, **kw: [])
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda *a: None)
    monkeypatch.setattr(server_restart, "_managed_update_pending_kwargs", lambda: {})
    events, published, cancelled = queue.Queue(), [], []
    context = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, consciousness=None,
        append_jsonl=lambda *a: None, bridge=SimpleNamespace(push_log=published.append),
        kill_workers=lambda **kw: cancelled.append(kw) or True,
    )
    controllers = []
    with ExitStack() as stack:
        for task_id, chat_id in (("native-main", 1), ("native-project", 7)):
            task = {"id": task_id, "chat_id": chat_id, "_is_direct_chat": True,
                    "metadata": {"project_id": "room"} if chat_id == 7 else {}}
            actor = SimpleNamespace(
                _busy=True, _accepting_owner_messages=True,
                _owner_message_admission_lock=threading.Lock(),
                _current_task_id=task_id, _current_chat_id=chat_id,
                _current_task_metadata=task["metadata"], _current_task_text="Continue work",
            )
            write_task_result(tmp_path, task_id, "running", chat_id=chat_id)
            registry.register(task_id, chat_id, actor=actor)
            stack.callback(registry.unregister, task_id)
            controller = stack.enter_context(model_wait.task_model_wait_scope(
                task=task, drive_root=tmp_path, event_queue=events, worker_slot_held=False))
            row = {"wait_id": "wait-" + task_id, "state": "waiting", "role": "main",
                   "task_attempt": 1, "auto_continue": True}
            controller.waits[row["wait_id"]] = row
            controller._publish(row)
            controllers.append(controller)
            handle_task_model_wait(events.get_nowait(), context)
        assert [row["chat_id"] for row in published] == [1, 7]
        assert {row["id"] for row in workers.direct_chat_turns()} == {"native-main", "native-project"}
        assert set(server_restart._live_running_task_ids(context)) == {"native-main", "native-project"}
        with _decision_clients(tmp_path) as clients:
            body = _action_for(published[1], "auto_continue", auto_continue=False)
            reply = clients["web"](body)
            assert reply.status_code == 202, reply.json()
            assert clients["host"](body).json()["duplicate"] is True
            for controller in controllers:
                controller._drain_controls()
            assert controllers[0].waits["wait-native-main"]["auto_continue"] is True
            assert controllers[1].waits["wait-native-project"]["auto_continue"] is False
            assert not owner_mailbox._mailbox_path(tmp_path, "native-main").exists()
            # Reuse the new manual Restart owner, without touching a real
            # process or daemon. Each actual controller sees its own Stop.
            server_restart._stop_owned_work(context)
            assert len(cancelled) == 1 and cancelled[0]["reconcile_delegate_custody"] is False
            assert all(cancel_intents.cancel_pending(tmp_path, owner.task_id) for owner in controllers)
            assert [owner.control_reason() for owner in controllers] == ["cancelled", "cancelled"]
            denied = clients["host"](_action_for(published[0], "retry"))
            assert denied.status_code == 409 and denied.json()["reason_code"] == "cancel_pending"
    assert registry.snapshot() == []
    assert all(owner.closed for owner in controllers)


@pytest.mark.parametrize("project_id", ["", "room"])
@pytest.mark.parametrize("action", ["switch", "stop", "stop_settled"])
def test_native_post_task_wait_remains_addressable_after_dialogue_closes(phase, monkeypatch, project_id, action):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.post_task_checkpoint import post_task_model_wait
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import message_bus, queue as task_queue
    from tests.test_direct_chat_turn_owner_control import _client
    from tests.test_llm_claudexor import MODEL

    f = phase
    monkeypatch.setattr(workers, "DRIVE_ROOT", f.root)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "get_event_q", lambda: f.events)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **kw: None))
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", f.root)
    monkeypatch.setattr(task_queue, "RUNNING", {})
    monkeypatch.setattr(task_queue, "PENDING", [])
    observed = {}

    class Actor:
        def handle_task(self, task):
            f.task.clear()
            f.task.update(task)
            self._busy, self._accepting_owner_messages = True, False
            self._current_task_id = task["id"]
            # The real loop_delivery seam closes ordinary dialogue before
            # entering post-task cognition; wait decisions still need an owner.
            write_task_result(f.root, task["id"], "completed", result="Already answered",
                              root_phase_checkpoint={"post_task_synthesis": "pending_once"})
            pending = [{"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "Already answered"},
                       {"type": "task_done", "task_id": task["id"], "status": "completed"}]
            with model_wait.task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=self._event_queue,
                                                  worker_slot_held=False) as owner:
                observed["owner"] = owner
                pipeline._dispatch_root_post_task(f.env, f.task, "Already answered", self._event_queue, pending,
                    {"rounds": 3}, {}, {}, f.root / "logs", budget_drive_root="", split_drive=False,
                    project_scoped=bool(project_id), project_task=False, parent_env=None, parent_task=None)
            return pending

    actor = Actor()
    thread = threading.Thread(target=workers._run_chat_task, args=(actor, 7 if project_id else 1, "Already answered"), kwargs={
        "task_metadata": {"project_id": project_id, "origin_suppressed": True},
    })
    release_stop, stop_observed = threading.Event(), threading.Event()
    thread.start()
    try:
        until(lambda: "owner" in observed and any(row.get("credential_harness") for row in observed["owner"].waits.values()))
        owner, task_id = observed["owner"], f.task["id"]
        row = next(row for row in owner.waits.values() if row["state"] == "waiting")
        assert workers.direct_chat_turn(task_id) is None  # ordinary dialogue remains closed
        assert active_activity.get_direct_activity_registry().get(task_id).actor is actor
        assert post_task_model_wait(f.root, task_id) is owner
        assert workers.drain_repo_writers(0) == [task_id]
        activity = next(item for item in _chat_activities_snapshot_safe(f.root) if item["activity_id"] == task_id)
        assert activity["kind"] == "direct_chat" and activity["phase"] == "finalizing"
        assert activity["model_waits"][row["wait_id"]]["state"] == "waiting"
        assert not row["worker_slot_held"]
        events = list(f.events.queue)
        assert any(event["type"] == "send_message" for event in events)
        assert not any(event["type"] == "task_done" for event in events)
        if action == "switch":
            body = _action_for({"task_id": task_id, **row}, "switch", model=MODEL,
                               credential_profile_id="replacement", use_local=False, persist_role=False)
            with _decision_clients(f.root) as clients:
                response = clients["web"](body)
                assert response.status_code == 202, response.json()
        else:
            if action == "stop":
                real_control_reason = owner.control_reason

                def hold_observed_stop():
                    reason = real_control_reason()
                    if reason == "cancelled":
                        stop_observed.set()
                        assert release_stop.wait(5), "test did not release the observed Stop"
                    return reason

                # Keep paid post-task custody alive until HTTP reports its
                # existing still-live response; this is a schedule, not a fake stop.
                monkeypatch.setattr(owner, "control_reason", hold_observed_stop)
            else:
                real_request_cancel = cancel_intents.request_cancel

                def settle_after_durable_stop(*args, **kwargs):
                    assert post_task_model_wait(f.root, task_id) is owner
                    intent = real_request_cancel(*args, **kwargs)
                    assert intent["request_id"] and intent["state"] == "requested"
                    assert intent["source"] == "http_single"
                    assert not intent.get("already_settled")
                    assert f.done.wait(5)
                    until(lambda: owner.closed and post_task_model_wait(f.root, task_id) is None)
                    assert row["resolution"] == "cancelled"
                    observed["accepted_stop"] = intent["request_id"]
                    return intent

                # The real wait can consume the durable intent before the next
                # custody call. A finished post phase has the legacy 404 envelope.
                monkeypatch.setattr(cancel_intents, "request_cancel", settle_after_durable_stop)
            with _client(f.root) as client:
                response = client.post(f"/api/tasks/{task_id}/cancel", json={"stop_policy": "immediate"})
                assert response.status_code == (503 if action == "stop" else 404), response.json()
            if action == "stop":
                assert stop_observed.wait(5)
                assert post_task_model_wait(f.root, task_id) is owner and not owner.closed
                intent = cancel_intents.active_intent(f.root, task_id)
                assert intent["source"] == "http_single" and intent["state"] == "requested"
                assert load_task_result(f.root, task_id)["root_phase_checkpoint"]["post_task_synthesis"] == "running"
                release_stop.set()
            else:
                assert observed["accepted_stop"]
                assert not cancel_intents.active_intent(f.root, task_id)
        thread.join(5)
        assert not thread.is_alive() and owner.closed
        stored = load_task_result(f.root, task_id)
        assert stored["status"] == "completed" and stored["result"] == "Already answered"
        assert stored["root_phase_checkpoint"]["post_task_synthesis"] == ("completed" if action == "switch" else "degraded")
        assert len(f.engine.creates) == (2 if action == "switch" else 1)
        if action == "switch":
            assert f.engine.uploads[-1][0]["account"] == {"mode": "pin", "profileId": "replacement"}
        else:
            assert row["resolution"] == "cancelled"
            assert f.stages == ["chat", "scratch", "summary", "reflection"]
        assert workers.drain_repo_writers(0) == []
        assert post_task_model_wait(f.root, task_id) is None
    finally:
        release_stop.set()
        f.task["_skip_post_task_synthesis"] = True
        f.ready.set()
        thread.join(5)
        assert not thread.is_alive()
