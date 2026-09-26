from __future__ import annotations

import pytest


class Bridge:
    def __init__(self, messages):
        self._messages = list(messages)
    def get_updates(self, offset=0, timeout=1):
        return [{"update_id": offset + idx, "message": msg} for idx, msg in enumerate(self._messages)]
    def broadcast(self, _payload):
        pass

class Ctx:
    def __init__(self, state):
        self.state = dict(state)
        self.sent = []
        self.consciousness = None
        self.kill_workers = None
    def load_state(self):
        return dict(self.state)
    def save_state(self, state):
        self.state = dict(state)
    def update_state(self, mutator):
        st = dict(self.state)
        mutator(st)
        self.state = st
        return st
    def send_with_budget(self, chat_id, text, **_kwargs):
        self.sent.append((chat_id, text))

def test_external_first_slash_binds_external_owner_without_executing(monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    called = []
    bridge = Bridge([{"chat": {"id": 42}, "from": {"id": 7}, "text": "/panic", "source": "skill:telegram-bridge"}])
    ctx = Ctx({})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *args, **kwargs: called.append(True))
    server._process_bridge_updates(bridge, 0, ctx)
    # Global owner binds for outbound routing, external owner binds for slash auth.
    assert ctx.state["owner_id"] == 7
    assert ctx.state["owner_chat_id"] == 42
    assert ctx.state["owner_external_id"] == 7
    assert ctx.state["owner_external_chat_id"] == 42
    assert ctx.state["owner_external_bound_at"]
    assert called == []
    assert ctx.sent == [(42, "✅ Owner chat registered. Send the command again to execute it.")]

def test_external_non_owner_slash_is_ignored(monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    called = []
    bridge = Bridge([{"chat": {"id": 99}, "from": {"id": 8}, "text": "/panic", "source": "skill:telegram-bridge"}])
    # An external owner is already bound to a different chat.
    ctx = Ctx({"owner_id": 7, "owner_chat_id": 42, "owner_external_id": 7, "owner_external_chat_id": 42})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *args, **kwargs: called.append(True))
    server._process_bridge_updates(bridge, 0, ctx)
    assert called == []
    assert ctx.sent == [(99, "⚠️ Command ignored: this transport is not the bound owner chat.")]

def test_desktop_web_owner_does_not_lock_out_telegram(monkeypatch):
    # Regression: on desktop the web UI binds owner=1/1 first; a real Telegram
    # owner must still be able to register (TOFU) and then execute slash commands.
    import server
    import supervisor.message_bus as message_bus
    called = []
    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *args, **kwargs: called.append(True))
    # First Telegram slash binds the external owner and asks for a resend.
    server._process_bridge_updates(
        Bridge([{"chat": {"id": 42}, "from": {"id": 7}, "text": "/panic", "source": "skill:telegram-bridge"}]),
        0, ctx,
    )
    assert called == []
    assert ctx.state["owner_id"] == 1 and ctx.state["owner_chat_id"] == 1
    assert ctx.state["owner_external_id"] == 7 and ctx.state["owner_external_chat_id"] == 42
    assert ctx.sent[-1] == (42, "✅ Owner chat registered. Send the command again to execute it.")
    # Resend from the bound external owner now executes.
    server._process_bridge_updates(
        Bridge([{"chat": {"id": 42}, "from": {"id": 7}, "text": "/panic", "source": "skill:telegram-bridge"}]),
        0, ctx,
    )
    assert called == [True]

def test_external_negative_id_cannot_bind_or_execute(monkeypatch):
    # Negative (A2A/synthetic) ids fail the chat_id>0 and user_id>0 gate, so they
    # can neither bind the external owner nor execute a slash command.
    import server
    import supervisor.message_bus as message_bus
    called = []
    bridge = Bridge([{"chat": {"id": -1001}, "from": {"id": -1001}, "text": "/panic", "source": "skill:a2a"}])
    ctx = Ctx({})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *args, **kwargs: called.append(True))
    server._process_bridge_updates(bridge, 0, ctx)
    assert called == []
    assert "owner_external_id" not in ctx.state
    assert ctx.sent == [(-1001, "⚠️ Command ignored: this transport did not provide owner identity.")]

def test_external_zero_identity_cannot_bind_owner_or_execute_on_retry(monkeypatch):
    import server
    import supervisor.message_bus as message_bus
    from supervisor.message_bus import LocalChatBridge
    called = []
    bridge = LocalChatBridge()
    bridge.enqueue_local_message("/panic", chat_id=0, user_id=0, source="skill:bridge")
    bridge.enqueue_local_message("/panic", chat_id=0, user_id=0, source="skill:bridge")
    ctx = Ctx({})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *args, **kwargs: called.append(True))
    server._process_bridge_updates(bridge, 0, ctx)
    server._process_bridge_updates(bridge, 1, ctx)
    assert called == []
    assert "owner_id" not in ctx.state
    assert "owner_external_id" not in ctx.state
    assert ctx.sent == [(0, "⚠️ Command ignored: this transport did not provide owner identity."), (0, "⚠️ Command ignored: this transport did not provide owner identity.")]


def test_accepted_restart_ends_current_batch_without_processing_later_command(monkeypatch):
    import server
    from supervisor import message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = Bridge([
        {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart", "source": "web"},
        {"chat": {"id": 1}, "from": {"id": 1}, "text": "/status", "source": "web"},
    ])
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_perform_owner_restart", lambda *_a: (True, ""))
    assert server._process_bridge_updates(bridge, 0, ctx) == 1
    assert ctx.sent == [(1, "♻️ Restarting.")]


def _real_bridge(*texts, chat_id=1, user_id=1, source="web"):
    """The real queue-backed bridge with a batch already waiting."""
    from supervisor.message_bus import LocalChatBridge

    bridge = LocalChatBridge()
    for index, text in enumerate(texts):
        bridge.enqueue_local_message(text, chat_id=chat_id, user_id=user_id, source=source,
                                     client_message_id=f"cmid-{index}")
    return bridge


def test_panic_never_calls_replay_before_the_hard_exit(monkeypatch):
    """Panic takes the direct hard-stop path; volatile replay cannot outlive it.
    Even a raising or blocking bridge hand-back is never invoked before stop."""
    import server
    import supervisor.message_bus as message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = _real_bridge("/panic", "after one", "/status")
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(bridge, "requeue_updates", _broken_hand_back)
    stops = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *_a, **_k: stops.append(True))
    assert server._process_bridge_updates(bridge, 0, ctx) == 2
    assert ctx.sent == [(1, "🛑 PANIC: killing everything. App will close.")], "nothing behind Panic ran"
    assert stops == [True]


def _broken_hand_back(*_a, **_k):
    raise RuntimeError("replay store exploded")


def test_accepted_restart_completes_when_the_hand_back_raises(monkeypatch, caplog):
    """A failed hand-back after an accepted /restart is a logged loss, not a
    loop crash: the batch still ends and the restart already requested stands."""
    import logging

    import server
    from supervisor import message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = _real_bridge("/restart", "/status")
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_perform_owner_restart", lambda *_a: (True, ""))
    monkeypatch.setattr(bridge, "requeue_updates", _broken_hand_back)
    with caplog.at_level(logging.ERROR):
        assert server._process_bridge_updates(bridge, 0, ctx) == 2
    assert ctx.sent == [(1, "♻️ Restarting.")]
    messages = [r.getMessage() for r in caplog.records]
    assert any("hand-back raised" in m and "replay store exploded" in m for m in messages)
    assert any("cannot take back 1 unprocessed update(s)" in m for m in messages)


def test_accepted_restart_hands_the_tail_back_and_the_next_read_serves_it(monkeypatch):
    """TZ-1: an accepted /restart ends the batch at once, but the messages
    dequeued behind it stay on the real bridge (their accepted rows durable),
    so a generation that is still reading (a cancelled or revived restart)
    handles them with their original ids instead of finding them erased."""
    import server
    from supervisor import message_bus, state

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    ctx.WORKERS, ctx.PENDING, ctx.RUNNING = {}, [], {}
    bridge = _real_bridge("/restart", "/status")
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_perform_owner_restart", lambda *_a: (True, ""))
    monkeypatch.setattr(state, "status_text", lambda *_a: "Runtime status")
    assert server._process_bridge_updates(bridge, 0, ctx) == 2
    assert ctx.sent == [(1, "♻️ Restarting.")], "the stop was not delayed by the tail"
    assert server._process_bridge_updates(bridge, 2, ctx) == 3
    assert ctx.sent == [(1, "♻️ Restarting."), (1, "Runtime status")]
    assert bridge.get_updates(offset=3, timeout=0) == []


def test_a_crash_mid_batch_hands_the_later_messages_back_for_the_next_tick(monkeypatch, caplog):
    """TZ-1: the batch reader drains several queued messages at once. One
    failing handler is the loop's crash to account for (it still raises); the
    messages dequeued behind it were never handled and come back on the next
    read instead of vanishing with the batch."""
    import logging

    import server
    from supervisor import message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = _real_bridge("one", "two", "three")
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    routed = []

    def route(_bridge, _ctx, message):
        if message["text"] == "two":
            raise RuntimeError("router exploded")
        routed.append(message["text"])

    monkeypatch.setattr(server, "_route_owner_message", route)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="router exploded"):
        server._process_bridge_updates(bridge, 0, ctx)
    assert routed == ["one"]
    assert any("Bridge update 2 failed" in r.getMessage() and "1 later update(s) handed back" in r.getMessage()
               for r in caplog.records)
    assert server._process_bridge_updates(bridge, 0, ctx) == 4
    assert routed == ["one", "three"], "the failing message is the crash; the one behind it is not lost"
    assert bridge.get_updates(offset=4, timeout=0) == []


def test_a_crash_mid_batch_stays_the_reported_crash_when_the_hand_back_fails(monkeypatch, caplog):
    """The hand-back is best effort: when it fails too, the loop still sees the
    handler's own crash (not the hand-back's), and the loss is logged."""
    import logging

    import server
    from supervisor import message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = _real_bridge("one", "two", "three")
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(bridge, "requeue_updates", _broken_hand_back)
    routed = []

    def route(_bridge, _ctx, message):
        if message["text"] == "two":
            raise RuntimeError("router exploded")
        routed.append(message["text"])

    monkeypatch.setattr(server, "_route_owner_message", route)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="router exploded"):
        server._process_bridge_updates(bridge, 0, ctx)
    assert routed == ["one"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("hand-back raised" in m and "replay store exploded" in m for m in messages)
    assert any("cannot take back 1 unprocessed update(s)" in m for m in messages)
    assert any("Bridge update 2 failed" in m and "0 later update(s) handed back" in m for m in messages)


def test_a_bridge_without_a_hand_back_is_told_what_it_loses(monkeypatch, caplog):
    """A transport-shaped fake without ``requeue_updates`` keeps the old
    behaviour (the batch ends) and the loss is logged, never silent."""
    import logging

    import server
    from supervisor import message_bus

    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    bridge = Bridge([
        {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart", "source": "web"},
        {"chat": {"id": 1}, "from": {"id": 1}, "text": "/status", "source": "web"},
    ])
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_perform_owner_restart", lambda *_a: (True, ""))
    with caplog.at_level(logging.ERROR):
        assert server._process_bridge_updates(bridge, 0, ctx) == 1
    assert ctx.sent == [(1, "♻️ Restarting.")]
    assert any("cannot take back 1 unprocessed update(s)" in r.getMessage() for r in caplog.records)


def test_command_voice_survives_live_delivery_and_history(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    import server
    from supervisor import message_bus, state
    from ouroboros.gateway.history import make_chat_history_endpoint

    (tmp_path / "logs").mkdir()
    live = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1, "bg_consciousness_enabled": True})
    ctx.consciousness = SimpleNamespace(start=lambda: "Background consciousness enabled",
                                        stop=lambda: "Background consciousness disabled")
    ctx.send_with_budget = message_bus.send_with_budget
    ctx.WORKERS, ctx.PENDING, ctx.RUNNING = {}, [], {}
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", ctx.load_state)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(state, "status_text", lambda *args: "Runtime status <plain>")
    monkeypatch.setattr(server, "_describe_bg_consciousness_state",
                        lambda enabled: {"status": "enabled", "detail": "waiting"})
    # Exercise the real command dispatcher; only effects/observations are isolated.
    for i, command in enumerate(("/bg start", "/bg status", "/bg stop", "/status")):
        server._process_bridge_updates(Bridge([{
            "chat": {"id": 1}, "from": {"id": 1}, "text": command,
            "source": "web", "client_message_id": f"voice-command-{i}",
        }]), i, ctx)
    rows = [r for r in live if r.get("type") == "chat"]
    assert len(rows) == 4
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"limit": "20"})))
    replay = [r for r in json.loads(response.body)["messages"]
              if r.get("system_type") == "command_reply"]
    assert len(replay) == 4
    for collection in (rows, replay):
        assert all(r["role"] == "system" and r["system_type"] == "command_reply"
                   and not r.get("markdown") for r in collection)
    assert [r["content"] for r in rows] == [r["text"] for r in replay]
