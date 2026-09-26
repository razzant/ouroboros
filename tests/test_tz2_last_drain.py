"""A finalizing worker cannot consume a new owner turn after its last loop drain."""

import pytest

from tests.test_project_routing_v664 import _ctx


@pytest.mark.parametrize("split", [False, True])
def test_settled_project_root_does_not_accept_unread_owner_mail(tmp_path, monkeypatch, split):
    from ouroboros.owner_mailbox import drain_owner_entries, _mailbox_path
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from ouroboros.task_results import write_task_result
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    project = create_project(tmp_path, "last-drain")
    chat_id = int(project["chat_id"])
    actor_root = tmp_path / "child-drive" if split else tmp_path
    actor_root.mkdir(exist_ok=True)
    task = {"id": "last-drain-root", "chat_id": chat_id, "root_task_id": "last-drain-root",
            "delegation_role": "root", "drive_root": str(actor_root)}
    # The answer has settled, but the pooled worker is still RUNNING for post-work.
    # Split-root copyback need not have published the terminal row in the canonical root yet.
    write_task_result(actor_root, task["id"], "completed", result="answer",
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    ctx = _ctx(tmp_path, running={task["id"]: {"task": task}})
    monkeypatch.setattr("ouroboros.server_owner_routing._addressable_root_tasks",
                        lambda *_: [{"task_id": task["id"], "project_id": project["id"]}])

    routed = _route_project_chat_to_running_task(ctx, chat_id, "late owner clarification", "late-1")
    assert routed == ""  # fall through to a fresh decision turn; no false delivered receipt
    assert drain_owner_entries(tmp_path, task["id"]) == []
    if not split:
        write_task_result(tmp_path, task["id"], "completed",
                          root_phase_checkpoint={"post_task_synthesis": "completed"})
        cleanup_settled_owner_mailbox(tmp_path, task["id"], task)
    assert not _mailbox_path(actor_root, task["id"]).exists()


def test_running_project_root_still_receives_owner_mail(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "live-loop")
    chat_id = int(project["chat_id"])
    task = {"id": "live-loop-root", "chat_id": chat_id, "root_task_id": "live-loop-root",
            "delegation_role": "root", "drive_root": str(tmp_path)}
    write_task_result(tmp_path, task["id"], "running", result="")
    ctx = _ctx(tmp_path, running={task["id"]: {"task": task}})
    monkeypatch.setattr("ouroboros.server_owner_routing._addressable_root_tasks",
                        lambda *_: [{"task_id": task["id"], "project_id": project["id"]}])
    assert _route_project_chat_to_running_task(ctx, chat_id, "continue", "live-1") == task["id"]
    assert [row["text"] for row in drain_owner_entries(tmp_path, task["id"])] == ["continue"]


def test_settled_project_followup_enters_new_decision_turn(tmp_path, monkeypatch):
    import server
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "new-turn")
    chat_id = int(project["chat_id"])
    task = {"id": "finished-loop", "chat_id": chat_id, "root_task_id": "finished-loop",
            "delegation_role": "root", "drive_root": str(tmp_path)}
    write_task_result(tmp_path, task["id"], "completed", result="answered",
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    calls = []
    ctx = _ctx(tmp_path, running={task["id"]: {"task": task}},
               direct=lambda *_a, **_k: calls.append("new_turn"))
    monkeypatch.setattr("ouroboros.server_owner_routing._addressable_root_tasks",
                        lambda *_: [{"task_id": task["id"], "project_id": project["id"]}])
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": 1},
                    "text": "new question after final answer", "source": "web",
                    "client_message_id": "post-work-1"}}]

        def send_routing_ack(self, *args, **kwargs):
            calls.append((args, kwargs))

        def broadcast(self, _payload):
            return None

    server._process_bridge_updates(Bridge(), 0, ctx)
    assert "new_turn" in calls
    assert not any(isinstance(call, tuple) and call[1].get("action") == "mailbox_delivery"
                   for call in calls)
    assert drain_owner_entries(tmp_path, task["id"]) == []


def test_decision_manifest_and_steer_refuse_settled_post_work_root(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.server_routing_context import _addressable_root_tasks
    from ouroboros.task_results import write_task_result
    from supervisor.steering import _handle_steer_task
    from ouroboros.project_dialogue import latest_chat_annotations

    project = create_project(tmp_path, "settled-steer")
    chat_id = int(project["chat_id"])
    task = {"id": "settled-steer-root", "chat_id": chat_id, "root_task_id": "settled-steer-root",
            "delegation_role": "root", "drive_root": str(tmp_path)}
    ctx = _ctx(tmp_path, running={task["id"]: {"task": task}})
    write_task_result(tmp_path, task["id"], "running")
    assert [row["task_id"] for row in _addressable_root_tasks(ctx, chat_id)] == [task["id"]]
    write_task_result(tmp_path, task["id"], "completed", result="answer",
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    assert _addressable_root_tasks(ctx, chat_id) == []
    # A stale decision turn may still call steer_task: the delivery owner must
    # independently refuse, not just rely on the manifest it saw at start.
    _handle_steer_task({"target_task_id": task["id"], "message": "late owner input",
                        "chat_id": chat_id, "client_message_id": "late-steer-1",
                        "routing_token": "t1", "issuer": {"kind": "owner_turn"}}, ctx)
    assert drain_owner_entries(tmp_path, task["id"]) == []
    assert latest_chat_annotations(tmp_path)["late-steer-1"]["reason"] == "target_finished"
