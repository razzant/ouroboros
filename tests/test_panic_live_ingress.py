"""Live authenticated emergency ingress, exclusively in disposable subprocesses."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("surface", ["http", "ws", "ws_chat", "external", "external_correlated"])
def test_panic_reaches_emergency_owner_with_stalled_normal_consumer(tmp_path, surface):
    from ouroboros.test_environment import isolated_environment

    repo = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_panic_live_ingress", surface], cwd=repo,
        env=isolated_environment(tmp_path, repo), capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PANIC_INGRESS_OK" in result.stdout


def _exercise(surface):
    import threading
    from types import SimpleNamespace

    from starlette.applications import Starlette
    from starlette.routing import Route, WebSocketRoute
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import server
    from ouroboros import config, server_auth, server_control
    from ouroboros.gateway import control, host_service, ws
    from supervisor import message_bus, state
    from tests.test_host_service_api import _seed_token

    root = pathlib.Path(config.DATA_DIR)
    state.init(root)
    state.init_state()
    known = state.update_state(lambda st: st.update(owner_external_id=17, owner_external_chat_id=23))
    bridge = message_bus.LocalChatBridge()
    bridge.panic = server_control.PanicIngress(server._startup_owner_command("/panic"))
    bridge.panic.observe_owner(known)
    message_bus.init(drive_root=root, chat_bridge=bridge, total_budget_limit=10, budget_report_every=10)
    # No thread drains the inbox. Ordinary consumer and state/chat writes cannot
    # help the emergency path; mark the supervisor live to cover startup routing.
    server._supervisor_thread = SimpleNamespace(is_alive=lambda: True)
    request_made, completed = threading.Event(), threading.Event()
    mp = pytest.MonkeyPatch()
    prior_threads = set(threading.enumerate())

    def forbidden(*_a, **_k):
        raise AssertionError("emergency request reached ordinary state/chat I/O")

    def request_only(**kwargs):
        if kwargs.get("request_only"):
            request_made.set()
        return []

    from ouroboros.startup_historical_audit import audit
    mp.setattr(audit, "stop", lambda: None)
    mp.setattr("ouroboros.local_model.get_manager", lambda **_kw: None)
    mp.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **_kw: None)
    mp.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", request_only)
    mp.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *_a, **_k: [])
    mp.setattr("ouroboros.tools.services.kill_all_services", lambda *_a, **_k: [])
    mp.setattr("ouroboros.extension_companion.panic_kill_all", lambda **_kw: [])
    mp.setattr("multiprocessing.active_children", lambda: [])
    mp.setattr("ouroboros.platform_layer.kill_process_on_port", lambda *_a: None)
    mp.setattr(server_control, "_persist_panic_controls", lambda *_a: None)
    mp.setattr(server_control, "_record_unconfirmed_daemon_stop", lambda *_a: None)
    mp.setattr(server_control, "_write_panic_flag", lambda *_a: request_made.wait(1))
    mp.setattr(server_control.os, "_exit", lambda *_a: completed.set())
    mp.setattr(state, "load_state", forbidden)
    mp.setattr(state, "init", forbidden)
    mp.setattr(message_bus, "log_chat", forbidden)
    mp.setattr(host_service, "_chat_rows", forbidden)
    try:
        if surface.startswith("external"):
            _seed_token(root, permissions=["inject_chat"])
            app = host_service.create_host_service_app(root, bridge_getter=lambda: bridge)
            with TestClient(app) as client:
                payload = {"text": "/panic", "user_id": 17, "chat_id": 23}
                if surface.endswith("correlated"):
                    payload["client_message_id"] = "panic-correlated"
                assert client.post("/chat/inject", json=payload).status_code == 403
                assert not server_control._restart_stop_requested
                for user, chat in [(0, 0), (99, 23), (17, 99)]:
                    response = client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                                           json={"text": "/panic", "user_id": user, "chat_id": chat})
                    assert response.status_code == 202, response.text
                    assert not request_made.is_set()
                    assert not server_control._restart_stop_requested
                response = client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=payload)
                assert response.status_code == 202
        else:
            mp.setattr(server_auth, "get_configured_network_password", lambda: "test-only-password")
            app = server_auth.NetworkAuthGate(Starlette(routes=[
                Route("/api/command", control.api_command, methods=["POST"]), WebSocketRoute("/ws", ws.ws_endpoint)]))
            with TestClient(app) as client:
                assert client.post("/api/command", json={"cmd": "/panic"}).status_code == 401
                with pytest.raises(WebSocketDisconnect):
                    with client.websocket_connect("/ws"):
                        pass
                assert not request_made.is_set()
                assert not server_control._restart_stop_requested
                headers = {"Authorization": "Bearer test-only-password"}
                if surface == "http":
                    response = client.post("/api/command", headers=headers, json={"cmd": "/panic"})
                    assert response.status_code == 200
                elif surface == "ws_chat":
                    # The composer sends slash text as type=chat, not type=command.
                    # Keep an earlier acceptance on this authenticated socket held:
                    # typed Panic must reach the SAME emergency owner before it.
                    entered, release = threading.Event(), threading.Event()
                    original_send = bridge.ui_send

                    def held_send(text, **kwargs):
                        if text == "ordinary held chat":
                            entered.set()
                            assert release.wait(5), "held acceptance was never released"
                        return original_send(text, **kwargs)

                    bridge.ui_send = held_send
                    with client.websocket_connect("/ws", headers=headers) as socket:
                        try:
                            socket.send_json({"type": "chat", "content": "ordinary held chat"})
                            assert entered.wait(3)
                            socket.send_json({"type": "chat", "content": " /PaNiC ",
                                              "client_message_id": "typed-panic"})
                            assert request_made.wait(3), "typed Panic waited behind ordinary chat acceptance"
                            assert not release.is_set()
                        finally:
                            release.set()
                else:
                    with client.websocket_connect("/ws", headers=headers) as socket:
                        socket.send_json({"type": "command", "cmd": "/panic"})
                        assert request_made.wait(3), "live WS Panic remained in the ordinary inbox"
        assert request_made.wait(3), "live Panic remained in the ordinary inbox"
        assert completed.wait(5)
        # Other commands retain their ordinary routing.
        bridge.ui_send("/restart", broadcast=False)
        bridge.ui_send("/status", broadcast=False)
        queued = [item["message"]["text"] for item in bridge.get_updates(0, timeout=0)]
        assert queued[-2:] == ["/restart", "/status"]
        assert queued.count("/panic") == (3 if surface.startswith("external") else 0)
        print("PANIC_INGRESS_OK", surface)
    finally:
        for thread in set(threading.enumerate()) - prior_threads:
            if thread.name.startswith("panic-"):
                thread.join(5)
                assert not thread.is_alive()
        mp.undo()


def test_pinned_owner_needs_positive_identity_and_reset_cannot_rebind():
    from ouroboros.server_control import PanicIngress
    from supervisor.state import STATE_READ_KEY

    door = PanicIngress()
    known = {"initialization_id": "i", "owner_external_id": 17, "owner_external_chat_id": 23}
    for invalid in ({}, {**known, "initialization_id": ""}, {**known, "owner_external_id": 0},
                    {**known, STATE_READ_KEY: {"quality": "unavailable"}}):
        door.observe_owner(invalid)
        assert door._owner[1] is None
    door.observe_owner(known)
    cell = door._owner
    door.observe_owner({STATE_READ_KEY: {"quality": "unavailable"}})
    assert door._owner[1] == (17, 23)
    door.invalidate_owner()
    cell[1] = (17, 23)  # even a late old reader can only publish into the retired cell
    door.observe_owner(known)
    assert door._owner == [False, None]


@pytest.mark.parametrize("source", ["web", "telegram"])
def test_accepted_panic_claims_termination_before_starting_callback(monkeypatch, source):
    import threading
    from types import SimpleNamespace

    from ouroboros import server_control

    started = []
    stop = lambda: pytest.fail("callback must remain unscheduled in this test")
    door = server_control.PanicIngress(stop)
    door.observe_owner({"initialization_id": "i", "owner_external_id": 17, "owner_external_chat_id": 23})

    def thread(*, target, name, daemon):
        assert target is stop and name == "panic-ingress" and daemon is True

        def start():
            assert server_control._restart_stop_requested is True
            started.append(True)

        return SimpleNamespace(start=start)

    monkeypatch.setattr(threading, "Thread", thread)
    assert door.request(" /PaNiC ", source=source, user_id=17, chat_id=23)
    assert started == [True] and server_control._restart_stop_requested is True


@pytest.mark.parametrize("text,source,user_id,chat_id,owner,has_stop", [
    ("/restart", "web", 0, 0, "known", True),
    ("/status", "web", 0, 0, "known", True),
    ("/panic", "web", 0, 0, "known", False),
    ("/panic", "telegram", 99, 23, "known", True),
    ("/panic", "telegram", 17, 99, "known", True),
    ("/panic", "telegram", 17, 23, "unknown", True),
    ("/panic", "telegram", 17, 23, "reset", True),
])
def test_rejected_panic_does_not_claim_termination(monkeypatch, text, source, user_id, chat_id, owner, has_stop):
    import threading

    from ouroboros import server_control

    stop = lambda: pytest.fail("rejected request must not call the stop owner")
    door = server_control.PanicIngress(stop if has_stop else None)
    if owner != "unknown":
        door.observe_owner({"initialization_id": "i", "owner_external_id": 17, "owner_external_chat_id": 23})
    if owner == "reset":
        door.invalidate_owner()
    monkeypatch.setattr(threading, "Thread", lambda **_: pytest.fail("rejected request started a thread"))
    assert not door.request(text, source=source, user_id=user_id, chat_id=chat_id)
    assert server_control._restart_stop_requested is False


if __name__ == "__main__":
    _exercise(sys.argv[1])
