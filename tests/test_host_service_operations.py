"""Host Service A2A operation correlation (#667) and the WS relay burst reserve.

An injected message that carries a ``client_message_id`` becomes an addressable
operation: ``/chat/inject`` answers with its ``operation_ref``, a repeated SAME
message rejoins instead of enqueueing again, ``/chat/operations/{ref}`` joins the
existing records (inbound chat row, routing receipt, live turn registry, task
result, outbound answer row) into one typed view scoped to the injecting skill,
and ``/chat/cancel`` runs the existing cancellation owner and reports only what
it proved. The WS relay lane is a 60-message token bucket refilling one per
second; refusals are aggregated per burst.
"""
from __future__ import annotations

import json
import pathlib

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.host_service import (
    WS_RELAY_BURST,
    create_host_service_app,
    operation_ref,
)
from ouroboros import dialogue_store
from ouroboros.utils import utc_now_iso
from tests.test_host_service_api import FakeBridge, _seed_token

SKILL = "a2a"
TOKEN = "a2a-token"
CHAT = -42
MSG = "a2a:task-1:msg-1"


def _client(tmp_path: pathlib.Path, bridge=None, **kwargs) -> TestClient:
    _seed_token(tmp_path, skill=SKILL, token=TOKEN, permissions=["inject_chat"])
    bridge = bridge or FakeBridge()
    return TestClient(create_host_service_app(tmp_path, bridge_getter=lambda: bridge, **kwargs))


def _chat_row(tmp_path: pathlib.Path, direction: str, text: str, *, chat_id: int = CHAT,
              client_message_id: str = "", source: str = f"skill:{SKILL}", task_id: str = "",
              session_id: str = "s1", record_type: str = "") -> None:
    """The canonical row ``supervisor.message_bus.log_chat`` writes (same keys)."""
    row = {
        "ts": utc_now_iso(), "session_id": session_id, "direction": direction, "chat_id": chat_id,
        "user_id": 0, "text": text, "format": "", "source": source, "sender_label": "",
        "sender_session_id": "", "client_message_id": client_message_id, "transport": {},
        "task_id": task_id,
    }
    if record_type:
        row["type"] = record_type
    dialogue_store.append(tmp_path, "chat", row)


def _inbound(tmp_path: pathlib.Path, text: str = "hello", **kwargs) -> None:
    _chat_row(tmp_path, "in", text, client_message_id=MSG, **kwargs)


def _headers() -> dict:
    return {"X-Skill-Token": TOKEN}


def _origin_ref(tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.message_bus import accepted_chat_message

    row = accepted_chat_message(tmp_path, CHAT, MSG)
    return build_owner_message_ref(chat_id=CHAT, client_message_id=MSG, ts=row["ts"], text=row["text"])


def _answer(tmp_path, text, task_id="40cc86d9"):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, task_id, "completed", result=text, origin_message_ref=_origin_ref(tmp_path))
    _chat_row(tmp_path, "out", text, task_id=task_id)


def _receipt(tmp_path: pathlib.Path, action: str, status: str, target: str) -> None:
    from ouroboros.project_dialogue import append_chat_annotation

    assert append_chat_annotation(tmp_path, MSG, action=action, target=target, status=status)


# --- inject: correlation and rejoin -----------------------------------------


def test_inject_returns_operation_ref_only_when_the_caller_names_the_message(tmp_path):
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    plain = client.post("/chat/inject", headers=_headers(), json={"text": "hi", "chat_id": CHAT})
    assert plain.status_code == 202 and plain.json() == {"ok": True, "status": "queued"}
    named = client.post("/chat/inject", headers=_headers(),
                        json={"text": "hi", "chat_id": CHAT, "client_message_id": MSG})
    assert named.status_code == 202
    assert named.json() == {"ok": True, "status": "queued", "operation_ref": operation_ref(CHAT, MSG)}
    assert bridge.messages[1]["client_message_id"] == MSG
    _answer(tmp_path, "reply from host")
    waited = client.post("/chat/inject", headers=_headers(), json={
        "text": "hi", "chat_id": CHAT, "client_message_id": MSG, "wait_for_response": True, "timeout_sec": 5,
    })
    assert waited.status_code == 200
    assert waited.json()["operation_ref"] == operation_ref(CHAT, MSG)


def test_inject_wait_expiry_carries_the_operation_ref(tmp_path):
    class SilentBridge(FakeBridge):
        def enqueue_local_message(self, text, **kwargs):
            self.messages.append({"text": text, **kwargs})

    client = _client(tmp_path, SilentBridge())
    response = client.post("/chat/inject", headers=_headers(), json={
        "text": "slow", "chat_id": CHAT, "client_message_id": MSG, "wait_for_response": True, "timeout_sec": 1,
    })
    assert response.status_code == 504
    assert response.json() == {
        "ok": False, "error": "timed out waiting for response", "operation_ref": operation_ref(CHAT, MSG),
    }


def test_same_message_rejoins_and_a_different_message_is_refused(tmp_path):
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    _inbound(tmp_path, "hello")
    rejoin = client.post("/chat/inject", headers=_headers(),
                         json={"text": " hello ", "chat_id": CHAT, "client_message_id": MSG})
    assert rejoin.status_code == 202
    assert rejoin.json() == {
        "ok": True, "status": "accepted", "rejoined": True, "operation_ref": operation_ref(CHAT, MSG),
    }
    assert bridge.messages == [], "a rejoin must not enqueue a second message"
    reused = client.post("/chat/inject", headers=_headers(),
                         json={"text": "something else", "chat_id": CHAT, "client_message_id": MSG})
    assert reused.status_code == 409
    assert "different message" in reused.json()["error"]
    assert bridge.messages == []


def test_rejoin_of_an_answered_message_returns_the_late_answer(tmp_path):
    bridge = FakeBridge()
    client = _client(tmp_path, bridge)
    _inbound(tmp_path, "hello")
    _answer(tmp_path, "late but real answer")
    response = client.post("/chat/inject", headers=_headers(), json={
        "text": "hello", "chat_id": CHAT, "client_message_id": MSG, "wait_for_response": True, "timeout_sec": 5,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "late but real answer" and body["status"] == "completed" and body["rejoined"] is True
    assert bridge.messages == []


def test_rejoin_refuses_a_message_id_bound_to_another_source(tmp_path):
    client = _client(tmp_path, FakeBridge())
    _inbound(tmp_path, "hello", source="skill:telegram")
    response = client.post("/chat/inject", headers=_headers(),
                           json={"text": "hello", "chat_id": CHAT, "client_message_id": MSG})
    assert response.status_code == 409
    assert "another source" in response.json()["error"]


# --- operations read ----------------------------------------------------------


def test_operation_read_is_scoped_to_the_injecting_skill(tmp_path):
    client = _client(tmp_path)
    ref = operation_ref(CHAT, MSG)
    assert client.get(f"/chat/operations/{ref}", headers=_headers()).status_code == 404
    _inbound(tmp_path, "hello", source="skill:telegram")
    assert client.get(f"/chat/operations/{ref}", headers=_headers()).status_code == 404
    assert client.get("/chat/operations/not-a-ref", headers=_headers()).status_code == 400
    assert client.get(f"/chat/operations/{ref}").status_code == 403


def test_operation_read_reports_pending_then_the_durable_answer(tmp_path):
    client = _client(tmp_path)
    ref = operation_ref(CHAT, MSG)
    _chat_row(tmp_path, "out", "an older answer in the same chat")  # before the message: never its answer
    _inbound(tmp_path, "hello")
    pending = client.get(f"/chat/operations/{ref}", headers=_headers()).json()
    assert pending["status"] == "pending" and pending["cancel_supported"] is False
    assert pending["chat_id"] == CHAT and pending["client_message_id"] == MSG
    _chat_row(tmp_path, "out", "", record_type="routing_options")  # host-authored, not an answer
    _answer(tmp_path, "the answer")
    done = client.get(f"/chat/operations/{ref}", headers=_headers()).json()
    assert done["status"] == "completed" and done["text"] == "the answer" and done["task_id"] == "40cc86d9"


def test_operation_read_reports_a_live_direct_turn(tmp_path, monkeypatch):
    _isolate_queue(monkeypatch, tmp_path, [])
    from supervisor.active_activity import get_direct_activity_registry

    client = _client(tmp_path)
    _inbound(tmp_path, "hello")
    registry = get_direct_activity_registry()
    registry.clear()
    try:
        registry.register("40cc86d9", CHAT, client_message_id=MSG, kind="direct_chat", origin_message_ref=_origin_ref(tmp_path))
        state = client.get(f"/chat/operations/{operation_ref(CHAT, MSG)}", headers=_headers()).json()
        assert state["phase"] == "direct_chat" and state["cancel_supported"] is True
    finally:
        registry.clear()


def test_operation_read_follows_the_routing_receipt_to_the_promoted_task(tmp_path, monkeypatch):
    _isolate_queue(monkeypatch, tmp_path, [])
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result

    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "task-abc")
    write_task_result(tmp_path, "task-abc", STATUS_SCHEDULED, description="do the thing", origin_message_ref=_origin_ref(tmp_path))
    state = client.get(f"/chat/operations/{operation_ref(CHAT, MSG)}", headers=_headers()).json()
    assert state["task_id"] == "task-abc" and state["phase"] == "managed_task"
    assert state["status"] == STATUS_SCHEDULED and state["cancel_supported"] is True
    write_task_result(tmp_path, "task-abc", STATUS_COMPLETED, result="final answer")
    state = client.get(f"/chat/operations/{operation_ref(CHAT, MSG)}", headers=_headers()).json()
    assert state["status"] == STATUS_COMPLETED and state["text"] == "final answer"
    assert state["cancel_supported"] is False


def test_operation_read_reports_lost_after_a_host_restart(tmp_path):
    from ouroboros.utils import atomic_write_json

    client = _client(tmp_path)
    _inbound(tmp_path, "hello", session_id="old-session")
    atomic_write_json(tmp_path / "state" / "state.json", {"session_id": "new-session"})
    state = client.get(f"/chat/operations/{operation_ref(CHAT, MSG)}", headers=_headers()).json()
    assert state["status"] == "lost" and state["reason"] == "host_restarted_before_answer"


# --- cancel -------------------------------------------------------------------


def _isolate_queue(monkeypatch, tmp_path, tasks):
    from supervisor import queue, workers

    pending = [dict(task) for task in tasks]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "_chat_agent", None, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue, pending


def test_cancel_before_any_addressable_work_is_disclosed_not_faked(tmp_path, monkeypatch):

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    ref = operation_ref(CHAT, MSG)
    assert client.post("/chat/cancel", headers=_headers(), json={"operation_ref": ref}).status_code == 404
    _inbound(tmp_path, "hello")
    queued = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": ref})
    assert queued.status_code == 409
    assert queued.json()["outcome"] == "cancel_unsupported" and queued.json()["reason"] == "not_started"
    assert client.post("/chat/cancel", headers=_headers(), json={"operation_ref": "junk"}).status_code == 400
    assert client.post("/chat/cancel", headers=_headers(), json=["not", "an", "object"]).status_code == 400


def test_cancel_of_a_message_steered_into_a_foreign_task_is_unsupported(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    _isolate_queue(monkeypatch, tmp_path, [{"id": "foreign", "chat_id": 1, "root_task_id": "foreign"}])
    client = _client(tmp_path)
    _inbound(tmp_path, "please also do X")
    _receipt(tmp_path, "steer_task", "delivered", "foreign")
    write_task_result(tmp_path, "foreign", STATUS_RUNNING, description="owner work")
    response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
    assert response.status_code == 409
    assert response.json()["outcome"] == "cancel_unsupported"
    assert response.json()["reason"] == "not_started"
    assert "task_id" not in response.json(), "a presentation hint must not expose unrelated task state"


def test_cancel_runs_the_real_cancellation_owner_on_the_promoted_task(tmp_path, monkeypatch):
    """A real queued task (PENDING, no worker) bound to the message by the
    promotion receipt is cancelled through the durable intent + custody path;
    the answer is read back from the durable result, and a second cancel says
    ``already_terminal``."""
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_SCHEDULED, load_task_result, write_task_result

    queue, pending = _isolate_queue(monkeypatch, tmp_path, [
        {"id": "task-abc", "chat_id": CHAT, "root_task_id": "task-abc"},
    ])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "task-abc")
    write_task_result(tmp_path, "task-abc", STATUS_SCHEDULED, description="do the thing", origin_message_ref=_origin_ref(tmp_path))
    pending[0]["origin_message_ref"] = _origin_ref(tmp_path)
    response = client.post("/chat/cancel", headers=_headers(),
                           json={"operation_ref": operation_ref(CHAT, MSG), "reason": "peer gave up"})
    assert response.status_code == 200, response.json()
    assert response.json()["outcome"] == "cancelled" and response.json()["status"] == STATUS_CANCELLED
    assert response.json()["task_id"] == "task-abc"
    assert pending == []
    assert load_task_result(tmp_path, "task-abc")["status"] == STATUS_CANCELLED
    intents = json.loads((tmp_path / "state" / "cancel_intents.json").read_text(encoding="utf-8"))
    assert "task-abc" not in (intents.get("intents") or {}), "the intent settled"
    again = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
    assert again.status_code == 200 and again.json()["outcome"] == "already_terminal"
    state = client.get(f"/chat/operations/{operation_ref(CHAT, MSG)}", headers=_headers()).json()
    assert state["status"] == STATUS_CANCELLED


def test_cancel_reports_unresolved_when_custody_does_not_settle(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks as gateway_tasks
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from supervisor import queue

    _isolate_queue(monkeypatch, tmp_path, [])
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda task_id: True)
    monkeypatch.setattr(queue, "task_subtree_is_live", lambda task_id, **_kw: True)
    monkeypatch.setattr(gateway_tasks, "_run_cascade_cancel", lambda task_id: False)
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "task-abc")
    write_task_result(tmp_path, "task-abc", STATUS_RUNNING, description="do the thing", origin_message_ref=_origin_ref(tmp_path))
    response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
    assert response.status_code == 503
    assert response.json()["outcome"] == "unresolved" and response.json()["ok"] is False
    assert load_task_result(tmp_path, "task-abc")["status"] == STATUS_RUNNING
    intents = json.loads((tmp_path / "state" / "cancel_intents.json").read_text(encoding="utf-8"))
    assert "task-abc" in (intents.get("intents") or {}), "the durable intent stays open for the watchdog"


def test_cancel_refuses_when_the_durable_intent_cannot_be_recorded(tmp_path, monkeypatch):
    from ouroboros import cancel_intents
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import queue

    _isolate_queue(monkeypatch, tmp_path, [])
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda task_id: True)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cancel_intents, "request_cancel", _boom)
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "task-abc")
    write_task_result(tmp_path, "task-abc", STATUS_RUNNING, description="do the thing", origin_message_ref=_origin_ref(tmp_path))
    response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
    assert response.status_code == 503
    assert response.json()["outcome"] == "refused" and response.json()["reason"] == "cancel_intent_write_failed"


# --- WS relay burst reserve --------------------------------------------------


def test_ws_relay_burst_reserve_refuses_with_diagnostics_then_refills(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.gateway import host_service

    clock = [0.0]
    # Only this Host module's limiter sees the controlled clock; ASGI keeps time.
    monkeypatch.setattr(host_service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    _seed_token(tmp_path, skill="wsskill", token="tok", manifest_permissions=["ws_handler"])
    sent: list[dict] = []
    app = create_host_service_app(tmp_path, ws_broadcaster_getter=lambda: sent.append)
    client = TestClient(app)
    payload = {"message_type": "progress", "data": {"pct": 1}}
    for _ in range(WS_RELAY_BURST):
        assert client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json=payload).status_code == 202
    assert len(sent) == WS_RELAY_BURST
    first = client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json=payload)
    second = client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json=payload)
    assert first.status_code == 429 and second.status_code == 429
    assert first.json()["dropped_in_burst"] == 1 and second.json()["dropped_in_burst"] == 2
    assert 0 < first.json()["retry_after_sec"] <= 1.0
    assert first.headers["Retry-After"] == "1"
    assert len(sent) == WS_RELAY_BURST, "a refused relay reaches no browser client"
    # One second later exactly one token exists again; the burst summary is
    # recorded ONCE, durably, with the aggregate dropped count.
    clock[0] += 1.0
    assert client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json=payload).status_code == 202
    assert client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json=payload).status_code == 429
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    dropped = [row for row in rows if row.get("type") == "host_service_ws_relay_dropped"]
    assert [(row["skill"], row["dropped"]) for row in dropped] == [("wsskill", 2)]


def test_in_process_send_ws_message_is_not_bounded_by_the_host_bucket(tmp_path):
    """The bucket replaces only the Host Service WS lane; the in-process
    broadcast path (``PluginAPIImpl.send_ws_message``) keeps no bound."""
    from ouroboros import extension_plugin_api as plugin_api
    from ouroboros.extension_loader import extension_surface_name

    received: list[dict] = []
    previous = plugin_api._ws_broadcaster
    plugin_api._ws_broadcaster = received.append
    try:
        api = plugin_api.PluginAPIImpl.__new__(plugin_api.PluginAPIImpl)
        api._skill = "burst_skill"
        api._permissions = {"ws_handler"}
        api._runtime_closing = False
        api._runtime_closed = False
        api._api_lock = __import__("threading").RLock()
        for index in range(WS_RELAY_BURST * 3):
            api.send_ws_message("progress", {"n": index})
    finally:
        plugin_api._ws_broadcaster = previous
    assert len(received) == WS_RELAY_BURST * 3
    assert received[-1]["type"] == extension_surface_name("burst_skill", "progress")


@pytest.mark.parametrize("value", ["", ":", "abc", "12:"])
def test_operation_ref_parsing_rejects_malformed_refs(value):
    from ouroboros.gateway.host_service import _parse_operation_ref

    with pytest.raises(ValueError):
        _parse_operation_ref(value)


@pytest.mark.parametrize("status", ["running", "scheduled", "completed", "cancelled"])
def test_unowned_cancel_reports_only_the_observed_terminal_state(tmp_path, monkeypatch, status):
    from ouroboros import cancel_intents
    from ouroboros.task_results import write_task_result, load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "accepted work")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "unowned")
    write_task_result(tmp_path, "unowned", status, origin_message_ref=_origin_ref(tmp_path))
    def no_new_cancel(*args, **kwargs):
        pytest.fail("absent physical ownership must not mint a second cancellation attempt")
    monkeypatch.setattr(cancel_intents, "request_cancel", no_new_cancel)
    response = client.post("/chat/cancel", headers=_headers(),
                           json={"operation_ref": operation_ref(CHAT, MSG)})
    body = response.json()
    assert body["status"] == load_task_result(tmp_path, "unowned")["status"] == status
    if status in SETTLED_STATUSES:
        assert response.status_code == 200 and body["ok"] is True
        assert body["outcome"] == "already_terminal"
    else:
        assert response.status_code == 503 and body["ok"] is False
        assert body["outcome"] == "unresolved"
        assert body["reason"] == "cancellation_did_not_settle"


def test_cancel_never_targets_a_different_queue_root(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.gateway import tasks
    host, other = tmp_path / "host", tmp_path / "other"
    client = _client(host)
    _inbound(host, "own request")
    _receipt(host, "promote_chat_to_task", "scheduled", "same-id")
    write_task_result(host, "same-id", "scheduled", origin_message_ref=_origin_ref(host))
    write_task_result(other, "same-id", "scheduled", description="unrelated")
    _isolate_queue(monkeypatch, other, [{"id": "same-id", "chat_id": 1}])
    monkeypatch.setattr(tasks, "_run_cascade_cancel", lambda *_a: pytest.fail("foreign cancellation"))
    view = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert view["task_id"] == "same-id" and view["cancel_supported"] is False
    response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": f"{CHAT}:{MSG}"})
    assert response.status_code == 409
    assert response.json()["reason"] == "cancel_owner_unavailable"
    assert not (host / "state/cancel_intents.json").exists()
    assert not (other / "state/cancel_intents.json").exists()


def test_direct_operation_with_a_different_cancel_owner_is_explicit(tmp_path, monkeypatch):
    from supervisor.active_activity import get_direct_activity_registry
    host, other = tmp_path / "host", tmp_path / "other"
    client = _client(host)
    _inbound(host, "own request")
    _isolate_queue(monkeypatch, other, [])
    registry = get_direct_activity_registry()
    registry.clear()
    try:
        registry.register("direct", CHAT, client_message_id=MSG, kind="direct_chat", origin_message_ref=_origin_ref(host))
        state = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
        assert state["phase"] == "direct_chat" and state["cancel_supported"] is False
        assert state["reason"] == "cancel_owner_unavailable"
        response = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": f"{CHAT}:{MSG}"})
        assert response.status_code == 409 and response.json()["reason"] == "cancel_owner_unavailable"
    finally:
        registry.clear()
