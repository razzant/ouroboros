"""#2012 across processes: the owner's public Stop reaches a parked author in another Host process.

The child is the existing isolated Host (``tests/presence_lifecycle_host.py``): a production
agent that reviewed its own result, left that review for an addressed message and parked,
lending its conversation. This process plays the owner's server on the same data root: the
public route (``gateway.tasks.api_task_cancel``) records the intent and drives custody, which
has no local registry entry for the turn and must find it through the cross-process author
census. The child's own runner observes the intent and writes the terminal; this process's
watchdog settles the intent. No ``/fixture/stop`` shortcut is used.
"""
from __future__ import annotations

import os
import time

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_presence_continuation_process import Host, eventually, host_result
from tests.test_presence_continuation_process import tmp_path as _short_socket_tmp_path

tmp_path = _short_socket_tmp_path  # the child Host's macOS-safe short AF_UNIX root

pytestmark = [pytest.mark.serial, pytest.mark.skipif(
    os.name != "posix", reason="The child Host fixture qualifies POSIX process-group custody only")]


def test_public_stop_reaches_a_parked_author_in_another_process(tmp_path, monkeypatch):
    from ouroboros import cancel_intents as ci
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.conversation_lease import live_authors
    from ouroboros.presence_runner import presence_turn_is_live
    from supervisor import queue, workers
    from supervisor.task_lifecycle import sweep_cancel_intents

    host = Host(tmp_path, 1, "blocking")
    try:
        first = host.turn().json()
        assert first.get("status") == "continuing", first
        ref = first["turn_ref"]
        data = tmp_path / "data"
        eventually(lambda: (host_result(data, ref).get("owner_wait") or {}).get("state") == "waiting")
        # This process is the owner's server for the child's install (``as_install``).
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
        monkeypatch.setattr(queue, "DRIVE_ROOT", data)
        monkeypatch.setattr(queue, "PENDING", [])
        monkeypatch.setattr(queue, "RUNNING", {})
        monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
        # No local stack: only the cross-process census witnesses the parked author.
        assert not presence_turn_is_live(ref, drive_root=data)
        assert ref in live_authors(drive_root=data)
        app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
        app.state.drive_root = data
        response = TestClient(app).post(f"/api/tasks/{ref}/cancel", json={
            "cascade": True, "stop_policy": "immediate", "stop_action_id": "owner-stop-process"})
        assert response.status_code == 202, response.text
        assert response.json()["cancel_state"] == "pending"
        # The child's runner ends the parked author on its own control check: no new model call.
        final = eventually(lambda: (reply if (reply := host.work(ref)).status_code == 200 else None))
        assert final.json()["status"] == "cancelled" and final.json()["text"] == ""
        assert final.json()["code"] == "presence_turn_cancelled"
        assert len(host.inputs()) == 2
        eventually(lambda: host.client.get("/fixture/live").json() == [])
        row = host_result(data, ref)
        assert row["status"] == "cancelled"
        assert row["cancel_origin"]["request_id"] == response.json()["request_id"]
        assert row["cancel_origin"]["source"] == "http_cascade"
        # The tree Stop stays pending while the author's own paid review still runs (custody counts it).
        eventually(lambda: ref not in live_authors(drive_root=data))
        assert sweep_cancel_intents(now=time.time() + 3600) == {ref: "failed"}
        assert ci.active_intent(data, ref)
        # Once that reviewer returns, this process's watchdog settles the intent from the child's terminal.
        assert host.client.post("/fixture/release-review").status_code == 200
        eventually(lambda: sweep_cancel_intents(now=time.time() + 3600) == {ref: "cancelled"})
        assert ci.active_intents(data) == {}
        assert host_result(data, ref)["status"] == "cancelled"
    finally:
        host.close()



def test_public_wrap_up_reaches_a_parked_author_in_another_process(tmp_path, monkeypatch):
    """Wrap up instead of Stop now, the same parked author in another Host process. This process's
    shared owner finds the turn only through the author census, records the graceful intent and arms
    the existing episode on the child's own mailbox; the child wakes, drains it and takes its one
    tool-less ``[OWNER_STOP]`` round. This fixture's model answers that round with a tool call a
    tool-less round cannot take, so the honest ending is a Wrap up without an answer: nothing is said."""
    import queue as queues
    import threading

    from ouroboros import cancel_intents as ci
    from ouroboros.conversation_lease import live_authors
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.presence_runner import presence_turn_is_live
    from supervisor import queue, workers
    from supervisor.owner_stop import owner_stop_ending
    from supervisor.task_lifecycle import sweep_cancel_intents

    host = Host(tmp_path, 1, "blocking")
    try:
        first = host.turn().json()
        assert first.get("status") == "continuing", first
        ref = first["turn_ref"]
        data = tmp_path / "data"
        eventually(lambda: (host_result(data, ref).get("owner_wait") or {}).get("state") == "waiting")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
        monkeypatch.setattr(queue, "DRIVE_ROOT", data)
        monkeypatch.setattr(queue, "PENDING", [])
        monkeypatch.setattr(queue, "RUNNING", {})
        monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
        bus = queues.Queue()
        monkeypatch.setattr(workers, "get_event_q", lambda *_a, **_k: bus)
        assert not presence_turn_is_live(ref, drive_root=data) and ref in live_authors(drive_root=data)
        assert len(host.inputs()) == 2
        app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
        app.state.drive_root = data
        response = TestClient(app).post(f"/api/tasks/{ref}/cancel", json={
            "cascade": True, "stop_policy": "finalize_then_cancel", "stop_action_id": "owner-wrap-up-process"})
        for thread in threading.enumerate():  # the ingress arms the episode off its request thread
            if thread.name.startswith("owner-stop-"):
                thread.join(30)
        assert response.status_code == 202, response.text
        assert response.json()["cancel_state"] == "pending" and response.json()["stop_policy"] == "finalize_then_cancel"
        request_id = ci.active_intent(data, ref)["request_id"]
        assert [(entry["kind"], entry["msg_id"]) for entry in drain_owner_entries(data, ref, set())] == [
            ("finalize_now", f"ownerstop:{request_id}")]
        # The child's own loop drained it and took the one bounded round: no further call, nothing said.
        final = eventually(lambda: (reply if (reply := host.work(ref)).status_code == 200 else None))
        assert final.json()["status"] == "failed" and final.json()["text"] == "" and final.json()["outcome"] == "silent"
        inputs = host.inputs()
        assert len(inputs) == 3 and inputs[-1]["messages"][-1]["content"].startswith("[OWNER_STOP]")
        assert ci.active_intent(data, ref)["control_drained_at"]
        row = host_result(data, ref)
        assert row["status"] == "failed" and row["reason_code"] == "owner_requested_finalization"
        assert row["cancel_origin"]["stop_policy"] == "finalize_then_cancel"
        assert row["cancel_origin"]["request_id"] == request_id
        assert owner_stop_ending(row) == ("wrap_up", "stop")
        assert [(event["type"], event["system_type"]) for event in iter(lambda: bus.get_nowait() if not bus.empty() else None, None)] == [
            ("send_message", "finalization_notice")]
        # As for Stop now, the tree stays pending while the author's own paid review still runs.
        eventually(lambda: ref not in live_authors(drive_root=data))
        assert sweep_cancel_intents(now=time.time() + 3600) == {ref: "failed"}
        assert host.client.post("/fixture/release-review").status_code == 200
        eventually(lambda: sweep_cancel_intents(now=time.time() + 3600) == {ref: "cancelled"})
        assert ci.active_intents(data) == {} and host_result(data, ref)["status"] == "failed"
    finally:
        host.close()
