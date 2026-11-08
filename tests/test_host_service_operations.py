"""Host Service A2A operation correlation (#667) and the WS relay burst reserve.

An injected message that carries a ``client_message_id`` becomes an addressable
operation: ``/chat/inject`` answers with its ``operation_ref``, a repeated SAME
message rejoins instead of enqueueing again, ``/chat/operations/{ref}`` joins the
message's own records (the accepted row, tasks whose origin names it, the live turn
registry, origin-indexed answer rows) into one typed view scoped to the injecting skill,
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
              session_id: str = "s1", record_type: str = "", **meta) -> None:
    """The canonical row ``supervisor.message_bus.log_chat`` writes (same keys; ``meta``: its
    ``message_meta`` facts such as ``origin_message_ref`` and ``task_terminal_status``)."""
    row = {
        "ts": utc_now_iso(), "session_id": session_id, "direction": direction, "chat_id": chat_id,
        "user_id": 0, "text": text, "format": "", "source": source, "sender_label": "",
        "sender_session_id": "", "client_message_id": client_message_id, "transport": {},
        "task_id": task_id, **meta,
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


def test_operation_read_finds_the_promoted_task_by_its_origin(tmp_path, monkeypatch):
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


@pytest.mark.parametrize("value", ["", "abc", "12:"])
def test_operation_ref_parsing_rejects_malformed_refs(value):
    from ouroboros.gateway.host_service import _parse_operation_ref

    with pytest.raises(ValueError):
        _parse_operation_ref(value)


@pytest.mark.parametrize("status", ["running", "completed", "cancelled"])
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


# --- addressed sources: the message's own records, never the room's history --------


def _view(client, ref=None):
    return client.get(f"/chat/operations/{ref or operation_ref(CHAT, MSG)}", headers=_headers()).json()


def _final(tmp_path, ref, text, turn="turn-1", status="completed"):
    """A direct turn's final row as its host-operation reply writes it (``host_operation_reply_kwargs``)."""
    _chat_row(tmp_path, "out", text, task_id=turn, origin_message_ref=ref, task_terminal_status=status)


@pytest.mark.parametrize("answer", ["managed", "direct"])
def test_an_accepted_message_behind_ten_thousand_newer_rows_keeps_its_operation(tmp_path, monkeypatch, answer):
    """The window a room probe reads ends 10,000 rows back; the accepted row and its answer do not."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from tests._dialogue import seed

    inbound = {"ts": "2026-10-01T00:00:00+00:00", "session_id": "s1", "direction": "in", "chat_id": CHAT,
               "user_id": 0, "text": "original work", "source": f"skill:{SKILL}", "client_message_id": MSG,
               "transport": {}, "task_id": ""}
    ref = build_owner_message_ref(chat_id=CHAT, client_message_id=MSG, ts=inbound["ts"], text=inbound["text"])
    final = [{"ts": "2026-10-01T00:00:01+00:00", "direction": "out", "chat_id": CHAT, "text": "original answer",
              "task_id": "turn-1", "origin_message_ref": ref, "task_terminal_status": "completed"}]
    newer = [{"ts": "2026-10-01T00:00:02+00:00", "direction": "out", "chat_id": CHAT, "text": "unrelated",
              "task_id": f"other-{index:06d}"} for index in range(10_001)]
    seed(tmp_path, live={"chat": [inbound, *(final if answer == "direct" else []), *newer]})
    dialogue_store.ensure(tmp_path)  # the production cutover stores the whole chain, oldest first
    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    if answer == "managed":
        write_task_result(tmp_path, "task-old", "completed", result="original answer", origin_message_ref=ref)
    state = _view(client)
    assert state["status"] == "completed" and state["text"] == "original answer"
    assert state.get("task_id") == ("task-old" if answer == "managed" else None)
    cancel = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
    assert cancel.status_code == 200 and cancel.json()["outcome"] == "already_terminal"


def test_a_live_direct_turn_answers_as_the_turn_beside_its_own_running_record(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor.active_activity import get_direct_activity_registry

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "hello")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "turn-1", "running", origin_message_ref=ref, _is_direct_chat=True)
    registry = get_direct_activity_registry()
    registry.clear()
    try:
        registry.register("turn-1", CHAT, client_message_id=MSG, kind="direct_chat", origin_message_ref=ref)
        state = _view(client)
        assert (state["phase"], state["status"], state["task_id"], state["cancel_supported"]) == (
            "direct_chat", "running", "turn-1", True)
    finally:
        registry.clear()
    _final(tmp_path, ref, "the turn's answer")
    write_task_result(tmp_path, "turn-1", "completed", result="the turn's answer")
    state = _view(client)
    assert (state["status"], state["text"], state["task_id"], state["cancel_supported"]) == (
        "completed", "the turn's answer", "turn-1", False)


@pytest.mark.parametrize("annotation", ["missing", "overwritten"])
def test_a_direct_handoff_never_masks_the_project_task_it_started(tmp_path, monkeypatch, annotation):
    """The turn's own terminal is no answer while the managed root it promoted (whose rows live in
    its Project room) still runs, whatever the routing annotation says or lost."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "build it")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "turn-1", "completed", result="Started a project task.",
                      origin_message_ref=ref, _is_direct_chat=True)
    _final(tmp_path, ref, "Started a project task.")
    write_task_result(tmp_path, "proj-task", STATUS_RUNNING, origin_message_ref=ref, project_id="room",
                      chat_id=1001, description="build it")
    _chat_row(tmp_path, "out", "working on it", chat_id=1001, task_id="proj-task")
    if annotation == "overwritten":
        _receipt(tmp_path, "promote_chat_to_task", "scheduled", "proj-task")
        _receipt(tmp_path, "steer_task", "delivered", "foreign")
    state = _view(client)
    assert (state["task_id"], state["phase"], state["status"], state["cancel_supported"]) == (
        "proj-task", "managed_task", STATUS_RUNNING, True)


def test_a_retried_operation_answers_through_its_root_and_a_queued_successor_first(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    _, pending = _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "first", "failed", origin_message_ref=ref, retry_task_id="second", result="boom")
    write_task_result(tmp_path, "second", "running", origin_message_ref=ref, original_task_id="first")
    state = _view(client)
    assert (state["task_id"], state["status"], state["cancel_supported"]) == ("first", "running", True)
    pending.append({"id": "third", "chat_id": CHAT, "root_task_id": "third", "origin_message_ref": ref})
    state = _view(client)  # live queue payloads are the current source, before any persisted record
    assert (state["task_id"], state["status"]) == ("third", "pending")


@pytest.mark.parametrize("damage", ["successor_status", "successor_missing", "off_the_chain"])
def test_a_retry_chain_read_through_an_unreadable_record_is_unknown(tmp_path, monkeypatch, damage):
    """A damaged successor publishes no status of its own ("5"), a missing one never its failed
    predecessor's; a broken row the root's retry-aware read never went through leaves its answer."""
    from ouroboros import store
    from ouroboros.task_results import write_task_result

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "first", "failed", origin_message_ref=ref, retry_task_id="second", result="boom")
    write_task_result(tmp_path, "second", "running", origin_message_ref=ref, original_task_id="first")
    if damage == "off_the_chain":
        write_task_result(tmp_path, "stray", "running", origin_message_ref=ref)
    with store.open_store(tmp_path, write=True) as txn:
        if damage == "successor_status":
            txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='second'", (json.dumps(
                {"task_id": "second", "status": 5, "origin_message_ref": ref, "original_task_id": "first"}),))
        elif damage == "successor_missing":
            txn.conn.execute("DELETE FROM task_bodies WHERE task_id='second'")
        else:
            txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='stray'")
    state = _view(client)
    if damage == "off_the_chain":
        assert (state["task_id"], state["status"], state["cancel_supported"]) == ("first", "running", True)
    else:
        assert (state["status"], state.get("reason"), "text" in state, "task_id" in state) == (
            "pending", "task_record_unreadable", False, False)


def test_a_managed_task_answers_from_the_queue_then_its_record_across_a_restart(tmp_path, monkeypatch):
    """Queued before any worker persisted it, then persisted after the queue let it go, then read by a
    restarted host (another session): the task's own record keeps answering, never ``lost``."""
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import atomic_write_json

    _, pending = _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing", session_id="before")
    ref = _origin_ref(tmp_path)
    pending.append({"id": "task-q", "chat_id": CHAT, "root_task_id": "task-q", "origin_message_ref": ref})
    state = _view(client)
    assert (state["task_id"], state["status"], state["cancel_supported"]) == ("task-q", "pending", True)
    pending.clear()
    write_task_result(tmp_path, "task-q", "completed", origin_message_ref=ref, result="done")
    atomic_write_json(tmp_path / "state" / "state.json", {"session_id": "after"})
    _isolate_queue(monkeypatch, tmp_path, [])  # the restarted process's empty queue
    state = _view(client)
    assert (state["task_id"], state["status"], state["text"]) == ("task-q", "completed", "done")


def test_a_completed_root_with_live_children_stays_cancellable_and_no_child_answers(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    _, pending = _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "fan out")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "root", "completed", origin_message_ref=ref, result="delivered early")
    write_task_result(tmp_path, "child", "running", origin_message_ref=ref, parent_task_id="root",
                      root_task_id="root", delegation_role="subagent")
    pending.append({"id": "child", "parent_task_id": "root", "root_task_id": "root", "delegation_role": "subagent",
                    "origin_message_ref": ref})
    state = _view(client)
    assert (state["task_id"], state["status"], state["text"], state["cancel_supported"]) == (
        "root", "completed", "delivered early", True)


def test_a_dispatch_an_ended_process_took_is_lost_until_its_task_answers(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus, message_ingress

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path, message_bus.LocalChatBridge())
    assert client.post("/chat/inject", headers=_headers(), json={
        "text": "do work", "chat_id": CHAT, "client_message_id": MSG}).status_code == 202
    monkeypatch.setattr(message_ingress, "_process", lambda: "a-later-process")
    assert _view(client)["status"] == "lost"
    write_task_result(tmp_path, "task-x", "completed", origin_message_ref=_origin_ref(tmp_path), result="answered")
    state = _view(client)
    assert (state["task_id"], state["status"], state["text"]) == ("task-x", "completed", "answered")


def test_an_inbound_still_importing_is_found_and_never_called_lost(tmp_path):
    from ouroboros.dialogue_import import run_dialogue_import
    from ouroboros.utils import atomic_write_json
    from tests._dialogue import seed

    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [{
        "ts": "2026-10-01T00:00:00+00:00", "session_id": "before", "direction": "in", "chat_id": CHAT, "user_id": 0,
        "text": "older work", "source": f"skill:{SKILL}", "client_message_id": MSG, "transport": {}, "task_id": ""}]})
    atomic_write_json(tmp_path / "state" / "state.json", {"session_id": "after"})
    client = _client(tmp_path)
    state = _view(client)
    assert (state["status"], state["reason"], state["accepted_at"]) == (
        "pending", "history_importing", "2026-10-01T00:00:00+00:00")
    run_dialogue_import(tmp_path)
    assert _view(client)["status"] == "lost"


@pytest.mark.parametrize("damage", ["derived", "status", "missing"])
def test_an_unreadable_record_linked_to_the_message_is_unknown_never_a_direct_final_or_lost(tmp_path, monkeypatch,
                                                                                         damage):
    from ouroboros import store
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import atomic_write_json

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "build it", session_id="before")
    atomic_write_json(tmp_path / "state" / "state.json", {"session_id": "after"})
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "proj-task", "running", origin_message_ref=ref)
    _final(tmp_path, ref, "Started a project task.")
    with store.open_store(tmp_path, write=True) as txn:  # a torn body that still names the message
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='proj-task'", (
            json.dumps({"task_id": "proj-task", "status": 5, "origin_message_ref": ref}) if damage == "status"
            else json.dumps({"origin_message_ref": ref})[:-12],))
        if damage == "underived":
            txn.conn.execute("UPDATE tasks SET origin=NULL WHERE task_id='proj-task'")
        if damage == "missing":  # the origin index still selects the row; its body is gone
            txn.conn.execute("DELETE FROM task_bodies WHERE task_id='proj-task'")
    state = _view(client)
    assert (state["status"], state.get("reason"), "text" in state, "task_id" in state) == (
        "pending", "task_record_unreadable", False, False)


def test_an_operation_read_answers_on_a_store_before_the_origin_step(tmp_path, monkeypatch):
    from ouroboros import store
    from ouroboros.task_results import write_task_result
    from tests.test_hot_facts import schema_nine

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    write_task_result(tmp_path, "task-old", "completed", origin_message_ref=_origin_ref(tmp_path), result="answered")
    schema_nine(tmp_path)
    state = _view(client)
    assert (state["task_id"], state["status"], state["text"]) == ("task-old", "completed", "answered")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == 9, "a read never upgrades"


def test_an_incomplete_origin_is_no_authority(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "hello")
    ref = _origin_ref(tmp_path)
    for name, partial in (("no-ts", {**ref, "ts": ""}), ("no-digest", {k: v for k, v in ref.items() if k != "text_sha256"}),
                          ("unnamed", {**ref, "client_message_id": ""})):
        write_task_result(tmp_path, name, "completed", origin_message_ref=partial, result="foreign answer")
    _final(tmp_path, {**ref, "ts": ""}, "a final row with an incomplete origin")
    state = _view(client)
    assert state["status"] == "pending" and "text" not in state and "task_id" not in state


def test_an_older_writers_replace_leaves_the_origin_underived_and_still_found(tmp_path, monkeypatch):
    from ouroboros import store, store_hot_facts
    from ouroboros.task_results import write_task_result

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "do the thing")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "promoted", "running", origin_message_ref=ref)
    # A continuation carrying the message only in its metadata: no top-level origin, no authority.
    write_task_result(tmp_path, "foreign", "running", metadata={"origin_message_ref": ref})
    with store.open_store(tmp_path, write=True) as txn:  # an older body's whole-row REPLACE: its own column list
        for task_id in ("promoted", "foreign"):
            columns = store.typed_columns(txn.load(task_id))
            columns.pop("origin")
            columns["write_seq"] = txn.row(task_id, "write_seq") + 1
            names = ("task_id", *columns)
            txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                             (task_id, *columns.values()))
        assert txn.row("promoted", "origin") is txn.row("foreign", "origin") is None
    state = _view(client)
    assert (state["task_id"], state["status"], state["cancel_supported"]) == ("promoted", "running", True)
    assert store_hot_facts.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert (txn.row("promoted", "origin"), txn.row("foreign", "origin")) == (operation_ref(CHAT, MSG), "")
    assert _view(client)["task_id"] == "promoted"
