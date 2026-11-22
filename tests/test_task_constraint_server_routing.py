from types import SimpleNamespace
import asyncio
import json

import server
from ouroboros.gateway import control as gateway_control


class FakeBridge:
    def get_updates(self, offset, timeout=1):
        return [{
            "update_id": 1,
            "message": {
                "chat": {"id": 1},
                "from": {"id": 1},
                "text": "repair skill",
                "task_constraint": {"mode": "skill_repair", "skill_name": "alpha", "payload_root": "skills/external/alpha"},
                "suppress_chat_log": True,
            },
        }]

    def broadcast(self, payload):
        pass


def test_constrained_repair_promotes_managed_task_before_busy_direct_lane(monkeypatch):
    calls = {"inject": 0, "direct": [], "promote": [], "sent": []}
    agent = SimpleNamespace(_busy=True, inject_message=lambda *a, **k: calls.__setitem__("inject", calls["inject"] + 1))
    ctx = SimpleNamespace(
        load_state=lambda: {"owner_id": 1},
        save_state=lambda st: None,
        update_state=lambda mutator: (lambda st: (mutator(st), st)[1])({"owner_id": 1}),
        consciousness=SimpleNamespace(inject_observation=lambda *_: None, pause=lambda: None, resume=lambda: None),
        get_chat_agent=lambda: agent,
        send_with_budget=lambda chat_id, text, **kw: calls["sent"].append((chat_id, text)),
        handle_chat_direct=lambda cid, txt, img, task_constraint=None, task_metadata=None: calls["direct"].append(task_constraint),
    )
    monkeypatch.setattr(
        "supervisor.events._handle_promote_chat_to_task",
        lambda event, _ctx: (
            calls["promote"].append(event)
            or {"status": "scheduled", "task_id": event["task_id"]}
        ),
    )
    monkeypatch.setattr(
        server,
        "_reserved_project_for_chat",
        lambda *_: (_ for _ in ()).throw(AssertionError("repair must route before project lookup")),
    )
    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=False):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
        def start(self):
            self.target(*self.args, **self.kwargs)
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)

    server._process_bridge_updates(FakeBridge(), 0, ctx)

    assert calls["inject"] == 0
    assert calls["direct"] == []
    assert len(calls["promote"]) == 1
    event = calls["promote"][0]
    assert event["type"] == "promote_chat_to_task"
    assert event["task_constraint"] == {
        "mode": "skill_repair",
        "skill_name": "alpha",
        "payload_root": "skills/external/alpha",
    }
    assert event["origin_suppressed"] is True
    # The skill card issued this promote: the handler's publication boundary
    # owns any refusal notice, so the event says so.
    assert event["host_initiated"] is True
    # A landed repair publishes no row: the card's own "Repair task queued" receipt
    # and the task's card carry it (a refusal still gets its typed row, below).
    assert calls["sent"] == []


_REPAIR_INCOMING = {
    "chat_id": 1,
    "text": "repair skill",
    "client_message_id": "repair-1",
    "task_constraint": {
        "mode": "skill_repair",
        "skill_name": "alpha",
        "payload_root": "skills/external/alpha",
    },
}


def test_constrained_repair_refusal_sends_no_untyped_bubble(monkeypatch):
    """The handler's own publication boundary tells the owner (one typed System
    row); the routing lane adds no «⚠️ Repair task was not started» bubble."""
    sent, events = [], []
    ctx = SimpleNamespace(
        consciousness=SimpleNamespace(inject_observation=lambda *_: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
    )
    monkeypatch.setattr(
        "supervisor.events._handle_promote_chat_to_task",
        lambda event, _ctx: events.append(event) or {
            "status": "needs_manual_target",
            "reason": "skill_repair_payload_missing",
            "task_id": event["task_id"],
        },
    )

    server._route_owner_message(FakeBridge(), ctx, dict(_REPAIR_INCOMING))

    assert sent == []
    assert events[0]["host_initiated"] is True


def test_constrained_repair_outer_failure_sends_one_typed_not_started_row(monkeypatch):
    """R13: `repair_promotion_failed` is minted OUTSIDE the handler, so the lane
    calls the same notice helper — one typed row in the owner's chat, bound to
    the task id that never started, with the host's sentence."""
    sent = []
    ctx = SimpleNamespace(
        consciousness=SimpleNamespace(inject_observation=lambda *_: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
    )

    def _crash(event, _ctx):
        raise RuntimeError("handler crashed")

    monkeypatch.setattr("supervisor.events._handle_promote_chat_to_task", _crash)

    server._route_owner_message(FakeBridge(), ctx, dict(_REPAIR_INCOMING))

    [(chat_id, text, kwargs)] = sent
    assert chat_id == 1
    assert text == "repair skill · Not started: the skill repair request could not be started"
    assert kwargs["role"] == "system" and kwargs["system_type"] == "task_not_started"
    assert kwargs["task_id"]


def test_ordinary_busy_message_uses_native_lane(monkeypatch):
    calls = {"direct": []}
    bridge = FakeBridge()
    bridge.get_updates = lambda offset, timeout=1: [{
        "update_id": 2,
        "message": {
            "chat": {"id": 1},
            "from": {"id": 1},
            "text": "ordinary follow-up",
            "suppress_chat_log": True,
        },
    }]
    ctx = SimpleNamespace(
        load_state=lambda: {"owner_id": 1},
        save_state=lambda st: None,
        update_state=lambda mutator: (lambda st: (mutator(st), st)[1])({"owner_id": 1}),
        consciousness=SimpleNamespace(inject_observation=lambda *_: None, pause=lambda: None, resume=lambda: None),
        get_chat_agent=lambda: SimpleNamespace(_busy=True),
        handle_chat_direct=lambda *args, **kwargs: calls["direct"].append((args, kwargs)),
    )

    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=False):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}

        def start(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)

    server._process_bridge_updates(bridge, 0, ctx)

    assert len(calls["direct"]) == 1


def test_visible_repair_command_is_deduped(monkeypatch):
    calls = []

    class Request:
        async def json(self):
            return {
                "cmd": "repair",
                "visible_text": "Repair task queued",
                "visible_task_id": "skill_repair_alpha",
                "task_constraint": {"mode": "skill_repair", "skill_name": "alpha", "payload_root": "skills/external/alpha"},
            }

    class Bridge:
        def ui_send(self, text, **kwargs):
            calls.append((text, kwargs))

    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    first = asyncio.run(gateway_control.api_command(Request()))
    second = asyncio.run(gateway_control.api_command(Request()))

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 1


def test_failed_visible_repair_command_does_not_poison_dedupe(monkeypatch):
    calls = []
    bridges = []

    class Request:
        async def json(self):
            return {
                "cmd": "repair",
                "visible_text": "Repair task queued",
                "visible_task_id": "skill_repair_alpha",
                "task_constraint": {"mode": "skill_repair", "skill_name": "alpha", "payload_root": "skills/external/alpha"},
            }

    class FailingBridge:
        def ui_send(self, text, **kwargs):
            raise RuntimeError("bus down")

    class HealthyBridge:
        def ui_send(self, text, **kwargs):
            calls.append((text, kwargs))

    bridges.extend([FailingBridge(), HealthyBridge()])
    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: bridges.pop(0))
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    first = asyncio.run(gateway_control.api_command(Request()))
    second = asyncio.run(gateway_control.api_command(Request()))

    assert first.status_code == 400
    assert second.status_code == 200
    assert len(calls) == 1


def _selected_reply_command(sent):
    class Request:
        app = SimpleNamespace(state=SimpleNamespace(drive_root="."))

        async def json(self):
            return {"cmd": "explain", "visible_text": "Explanation queued", "visible_task_id": "explain_passage",
                    "reply_to": "row:1@2026-11-20T10:00:00Z#0123456789ab",
                    "reply_excerpt": {"text": "this passage", "start_utf16": 4}}

    class Bridge:
        def ui_send(self, text, **kwargs):
            sent.append((text, kwargs))

    return Request, Bridge


def test_selected_reply_command_is_queued_once_when_its_repeat_overlaps_the_reply_check(monkeypatch):
    """The reply check runs in a worker thread, so a repeat of the same visible command can arrive while the first
    is still being checked: one correlation id is one queued command."""
    import time

    sent, checked = [], []
    Request, Bridge = _selected_reply_command(sent)

    def slow_check(_root, body):
        checked.append(body["visible_task_id"])
        time.sleep(0.05)  # long enough for the other request to be at its own check meanwhile
        return {"address": body["reply_to"], "excerpt": body["reply_excerpt"]}

    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr("ouroboros.chat_references.ingress_reply", slow_check)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    async def both():
        return await asyncio.gather(gateway_control.api_command(Request()), gateway_control.api_command(Request()))

    first, second = asyncio.run(both())

    assert len(checked) == 2, "both requests were in their check at once"
    assert len(sent) == 1, "and still only one command was queued"
    assert sorted(response.status_code for response in (first, second)) == [200, 200]
    assert b"deduped" in (first.body + second.body)
    assert sent[0][1]["task_metadata"]["reply_excerpt"] == {"text": "this passage", "start_utf16": 4}


def test_refused_or_failed_selected_reply_command_leaves_its_id_free_to_retry(monkeypatch):
    sent, outcomes = [], iter([ValueError("The quoted message is unavailable."), RuntimeError("worker lost"), None])
    Request, Bridge = _selected_reply_command(sent)

    def check(_root, body):
        outcome = next(outcomes)
        if outcome:
            raise outcome
        return {"address": body["reply_to"], "excerpt": body["reply_excerpt"]}

    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr("ouroboros.chat_references.ingress_reply", check)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    statuses = [asyncio.run(gateway_control.api_command(Request())).status_code for _ in range(3)]

    assert statuses == [400, 400, 200] and len(sent) == 1, "a refused or failed command is no queued one"
    assert "explain_passage" in gateway_control._RECENT_VISIBLE_COMMANDS


def test_a_repeat_of_a_queued_selected_reply_command_is_deduped_though_its_source_is_gone(monkeypatch):
    """The earlier success stands: a repeat inside the dedupe window is answered before any source is read, so a source
    that has become unavailable since cannot turn it into a refusal, and nothing is queued twice."""
    sent, reads = [], []
    Request, Bridge = _selected_reply_command(sent)
    outcomes = iter([None, ValueError("The quoted message is unavailable. Retry or clear the quote.")])

    def check(_root, body):
        reads.append(body["visible_task_id"])
        outcome = next(outcomes)
        if outcome:
            raise outcome
        return {"address": body["reply_to"], "excerpt": body["reply_excerpt"]}

    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr("ouroboros.chat_references.ingress_reply", check)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    first = asyncio.run(gateway_control.api_command(Request()))
    second = asyncio.run(gateway_control.api_command(Request()))

    assert (first.status_code, second.status_code) == (200, 200)
    assert json.loads(second.body) == {"ok": True, "deduped": True, "task_id": "explain_passage"}
    assert len(sent) == 1 and reads == ["explain_passage"], "the repeat read no source and queued nothing"


def test_visible_repair_command_can_retry_after_short_dedupe_window(monkeypatch):
    calls = []
    now = {"value": 100.0}

    class Request:
        async def json(self):
            return {
                "cmd": "repair",
                "visible_text": "Repair task queued",
                "visible_task_id": "skill_repair_alpha",
                "task_constraint": {"mode": "skill_repair", "skill_name": "alpha", "payload_root": "skills/external/alpha"},
            }

    class Bridge:
        def ui_send(self, text, **kwargs):
            calls.append((text, kwargs))

    monkeypatch.setattr(gateway_control, "_RECENT_VISIBLE_COMMANDS", {})
    monkeypatch.setattr(gateway_control.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    monkeypatch.setattr(gateway_control, "broadcast_ws_sync", lambda payload: None)

    first = asyncio.run(gateway_control.api_command(Request()))
    now["value"] += gateway_control._VISIBLE_COMMAND_DEDUPE_SEC + 0.1
    second = asyncio.run(gateway_control.api_command(Request()))

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 2


def test_scoped_task_metadata_derives_project_from_chat_id():
    """chat_id is the SSOT for thread→project (full project awareness, v6.32.0):
    a registered project chat scopes task_metadata to its OWN project, OVERRIDING
    any client-supplied project_id; a non-project chat DROPS an untrusted client
    project_id; metadata is otherwise preserved (None stays None)."""
    # Registered project chat: override a mismatched client project_id.
    assert server._scoped_task_metadata("proj_a", {"project_id": "proj_b", "x": 1}) == {"project_id": "proj_a", "x": 1}
    # Registered project chat with no client value: set it.
    assert server._scoped_task_metadata("proj_a", None) == {"project_id": "proj_a"}
    # Non-project chat: drop an untrusted client project_id, keep the rest.
    assert server._scoped_task_metadata("", {"project_id": "proj_b", "x": 1}) == {"x": 1}
    # Non-project chat, nothing to scope: unchanged (None preserved).
    assert server._scoped_task_metadata("", None) is None
