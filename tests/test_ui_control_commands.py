"""A press of a host-drawn control is my human's act; the command it composed is not their words (#2778).

Each producer that composes a command from a control stamps ``ui_control``; the host then writes no "You" row for
it, and the command's one existing System receipt (``command_reply``) or the work it starts is all that shows. Words
my human typed INTO a control (an Evolution objective) stay theirs, byte for byte. Typed chat and CLI commands keep
being their words; Panic keeps its first place on every door.
"""
from __future__ import annotations

import asyncio
import ast
import json
import pathlib
from types import SimpleNamespace

import pytest

import server
from tests.test_transport_commands import Ctx

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A real bridge, the real ingress writer and history reader, with command effects observed."""
    from supervisor import message_bus, state

    (tmp_path / "logs").mkdir()
    bridge = message_bus.LocalChatBridge()
    live: list = []
    bridge._broadcast_fn = live.append
    ctx = Ctx({"owner_id": 1, "owner_chat_id": 1, "owner_external_id": 17, "owner_external_chat_id": 23,
               "bg_consciousness_enabled": False})
    ctx.consciousness = SimpleNamespace(start=lambda: "Background consciousness enabled",
                                        stop=lambda source="": ctx.stops.append(source) or "Background consciousness disabled")
    ctx.stops = []  # the door each stop named: every owner surface sends /bg, the owner's door
    ctx.send_with_budget = message_bus.send_with_budget
    ctx.WORKERS, ctx.PENDING, ctx.RUNNING = {}, [], {}
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", ctx.load_state)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: bridge)
    monkeypatch.setattr(state, "status_text", lambda *args: "Runtime status")
    return SimpleNamespace(root=tmp_path, bridge=bridge, ctx=ctx, live=live)


def _drain(host):
    server._process_bridge_updates(host.bridge, 0, host.ctx)


def _history(root):
    from ouroboros.gateway.history import make_chat_history_endpoint

    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"limit": "50"})))
    return json.loads(response.body)["messages"]


def _app(*routes):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    return TestClient(Starlette(routes=list(routes)))


def test_an_http_press_leaves_only_its_receipt_and_a_typed_command_stays_words(host):
    from starlette.routing import Route
    from ouroboros.gateway import control

    client = _app(Route("/api/command", control.api_command, methods=["POST"]))
    assert client.post("/api/command", json={"cmd": "/bg start", "ui_control": {"name": "setup_consciousness"}}).status_code == 200
    _drain(host)
    assert client.post("/api/command", json={"cmd": "/bg stop"}).status_code == 200  # the CLI's typed command
    _drain(host)
    rows = _history(host.root)
    assert [row["text"] for row in rows if row.get("role") == "user"] == ["/bg stop"], "only typed words are a You row"
    receipts = [row for row in rows if row.get("system_type") == "command_reply"]
    assert [row["text"] for row in receipts] == ["🧠 Background consciousness enabled", "🧠 Background consciousness disabled"]
    assert all(row["role"] == "system" for row in receipts), "one existing System receipt per command, no click journal"
    assert host.ctx.stops == ["owner"], "/bg stop is the owner's stop: only the owner's start lifts it"


def test_a_ws_press_writes_no_owner_row_while_the_composer_s_slash_text_does(host):
    from starlette.routing import WebSocketRoute
    from ouroboros.gateway import ws

    client = _app(WebSocketRoute("/ws", ws.ws_endpoint))
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"type": "command", "cmd": "/bg start", "ui_control": {"name": "consciousness_toggle"}})
        socket.send_json({"type": "command", "cmd": "/bg stop", "ui_control": {"name": ""}})  # a producer's bug
        notice = socket.receive_json()
        while notice.get("type") != "log":
            notice = socket.receive_json()
        assert "command refused: ui_control must be" in notice["data"]["message"]
        socket.send_json({"type": "chat", "content": "/bg status", "client_message_id": "typed-1"})
    # Leaving the socket settles its accepted chat frame (row -> queue -> echo) before the endpoint returns.
    assert any(frame.get("client_message_id") == "typed-1" and frame.get("role") == "user" for frame in host.live)
    _drain(host)
    rows = _history(host.root)
    assert [row["text"] for row in rows if row.get("role") == "user"] == ["/bg status"]
    assert [row["text"] for row in rows if row.get("system_type") == "command_reply"][0] == "🧠 Background consciousness enabled"
    assert not any("/bg stop" in str(row.get("text")) for row in rows), "a refused press runs nothing"


def test_the_objective_typed_into_evolution_stays_my_human_s_exact_words(host, monkeypatch):
    from starlette.routing import Route
    from ouroboros.gateway import control
    from ouroboros.owner_words import owner_row

    started: list = []
    monkeypatch.setattr("supervisor.events_runtime_controls.owner_evolution_start",
                        lambda objective, **_kw: started.append(objective) or "")
    entered = "  Make the test suite faster —\nwithout skipping any check. "
    client = _app(Route("/api/command", control.api_command, methods=["POST"]))
    body = {"cmd": f"/evolve on {entered.strip()}", "ui_control": {"name": "evolution_start", "owner_text": entered}}
    assert client.post("/api/command", json=body).status_code == 200
    _drain(host)
    assert started == [entered.strip()], "the command keeps its meaning"
    from ouroboros.chat_chain import room_rows

    [row] = [entry.row for entry in room_rows(host.root, 1) if entry.row.get("direction") == "in"]
    assert row["text"] == entered and row["ui_control"] == {"name": "evolution_start"}, "byte for byte, without the command"
    assert owner_row(row, host.root), "my human's words, as any words they type"
    history = _history(host.root)
    assert [item["text"] for item in history if item.get("role") == "user"] == [entered]
    assert [item["text"] for item in history if item.get("system_type") == "command_reply"] == ["🧬 Evolution campaign: ON"]


def test_a_malformed_press_is_refused_and_panic_still_goes_first(host, monkeypatch):
    from starlette.routing import Route
    from ouroboros import server_control
    from ouroboros.gateway import control

    asked: list = []
    monkeypatch.setattr(server_control, "request_panic", lambda bridge, text, **_kw: asked.append(text) or True)
    client = _app(Route("/api/command", control.api_command, methods=["POST"]))
    refused = client.post("/api/command", json={"cmd": "/bg start", "ui_control": "toggle"})
    assert refused.status_code == 400 and refused.json()["accepted"] is False
    assert host.bridge.get_updates(0, timeout=0) == [], "nothing was queued"
    assert client.post("/api/command", json={"cmd": "/panic", "ui_control": "toggle"}).status_code == 200
    assert asked == ["/panic"], "Panic is answered before any other field is read"


def test_a_skill_s_inline_button_press_writes_no_owner_row(host, monkeypatch):
    from ouroboros.gateway import host_service
    from starlette.testclient import TestClient
    from tests.test_host_service_api import _seed_token

    _seed_token(host.root, permissions=["inject_chat"])
    client = TestClient(host_service.create_host_service_app(host.root, bridge_getter=lambda: host.bridge))
    sent = {"text": "/bg start", "chat_id": 23, "user_id": 17, "source": "telegram", "sender_label": "Telegram (owner)"}
    headers = {"X-Skill-Token": "token"}
    assert client.post("/chat/inject", headers=headers, json={**sent, "ui_control": {"name": "telegram_mind_panel"}}).status_code == 202
    _drain(host)
    assert client.post("/chat/inject", headers=headers, json={**sent, "text": "/status"}).status_code == 202  # typed
    _drain(host)
    assert client.post("/chat/inject", headers=headers, json={**sent, "ui_control": []}).status_code == 400
    correlated = {**sent, "ui_control": {"name": "telegram_mind_panel"}, "client_message_id": "press-1"}
    assert client.post("/chat/inject", headers=headers, json=correlated).status_code == 400, "a press is no owner message"
    inbound = [row for row in _history(host.root) if row.get("role") == "user"]
    assert [row["text"] for row in inbound] == ["/status"], "the button's command never speaks as my human"
    assert any(text == "🧠 Background consciousness enabled" for _chat, text in host.ctx.sent) or any(
        row.get("text") == "🧠 Background consciousness enabled" for row in _history(host.root))


def test_the_telegram_buttons_mark_their_command_as_a_press():
    """The producer stamp, on the inline consciousness buttons' one inject (``skills/telegram/plugin.py``)."""
    tree = ast.parse((REPO / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8"))
    payloads = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                and any(isinstance(key, ast.Constant) and key.value == "text" for key in node.keys)
                and any(isinstance(value, ast.Name) and value.id == "translated" for value in node.values)]
    assert len(payloads) == 1
    stamp = dict(zip([getattr(key, "value", None) for key in payloads[0].keys], payloads[0].values))["ui_control"]
    assert ast.literal_eval(stamp) == {"name": "telegram_mind_panel"}


@pytest.mark.parametrize(("path", "frame"), [
    ("web/modules/activity.js", "cmd: `/bg ${on ? 'stop' : 'start'}`, ui_control: { name: 'consciousness_toggle' }"),
    ("web/modules/consciousness_dashboard.js", "cmd: `/bg ${toggling}`, ui_control: { name: 'consciousness_toggle' }"),
    ("web/modules/consciousness_status.js", "cmd: `/bg ${pending}`, ui_control: { name: 'consciousness_toggle' }"),
    ("web/modules/onboarding_consciousness.js", "ui_control: { name: 'setup_consciousness' }"),
    ("web/modules/chat_activity.js", "cmd: '/restart', ui_control: { name: 'restart' }"),
    ("web/modules/updates.js", "cmd: '/restart', ui_control: { name: 'restart' }"),
    ("web/modules/review_command.js", "ui_control: { name: 'deep_review' }"),
    ("web/modules/evolution.js", "{ name: 'evolution_start', ...(asked.objective ? { owner_text: asked.entered } : {}) }"),
    ("web/modules/skills.js", "ui_control: { name: 'skill_repair' }"),
])
def test_every_web_control_that_composes_a_command_marks_it(path, frame):
    assert frame in (REPO / path).read_text(encoding="utf-8")


# --- Skills › Repair: the press is the request; the prompt is the host's ---------------------------------------

def test_a_repair_press_starts_its_task_on_the_press_never_on_quoted_words(monkeypatch):
    calls: list = []
    monkeypatch.setattr("supervisor.events._handle_promote_chat_to_task", lambda event, _ctx: calls.append(event))
    press = {"name": "skill_repair", "pressed_at": "2026-11-09T09:00:00+00:00"}
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": "skills/external/alpha",
                  "allow_enable": False, "allow_review": True}
    server._route_owner_message(SimpleNamespace(), SimpleNamespace(), {
        "chat_id": 1, "text": "Repair and run the installed Ouroboros skill selected in the Skills UI.",
        "client_message_id": "host-1", "task_constraint": constraint, "task_metadata": {"ui_control": press}})
    [event] = calls
    assert event["origin_suppressed"] is True, "my human's own act started it"
    assert event["objective_author"] == {"kind": "ui_control", "control": "skill_repair"}
    assert event["ui_control"] == press


def _repair_ctx(tmp_path, press):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tools.tool_context import ToolContext

    metadata = {"origin_suppressed": True, "objective_author": {"kind": "ui_control", "control": "skill_repair"},
                **({"ui_control": press} if press else {})}
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="repair-1", task_metadata=metadata,
                       task_constraint=normalize_task_constraint({"mode": "normal", "skill_name": "alpha",
                                                                  "payload_root": "skills/external/alpha"}))


def test_the_repair_task_reads_its_prompt_as_the_host_s_and_enables_on_the_press_alone(tmp_path, monkeypatch):
    from ouroboros import skill_lifecycle_actions as actions
    from ouroboros.context import build_user_content
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.tools.skill_source_tools import _owner_instruction_error

    monkeypatch.setattr(actions, "active_tool_profile", lambda _ctx: "workspace_task")
    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *_a: False)
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result", lambda *_a, **_k: {})
    monkeypatch.setattr("ouroboros.tools.skill_source_tools._cyber", lambda: False)
    press = {"name": "skill_repair", "pressed_at": "2026-11-09T09:00:00+00:00"}
    ctx = _repair_ctx(tmp_path, press)
    prompt = "Repair and run the installed Ouroboros skill selected in the Skills UI."
    first = build_user_content({"text": prompt, "metadata": ctx.task_metadata})
    assert first.startswith("[OBJECTIVE_AUTHOR] The objective below was the host's prompt for my human's press of skill_repair")
    _initialize_owner_directives(ctx, [{"role": "user", "content": first}])
    assert [row["source"] for row in ctx._owner_directives] == ["initial_text"], "never quoted as my human's words"
    # Enabling the repaired skill rests on the press, dated by the door (a later owner disable still wins).
    assert actions._caller_action_error(ctx, "enable", "", None) == ("", {
        "kind": "control", "control": "skill_repair", "ts": press["pressed_at"], "text": ""})
    # The press authorizes the chosen repair only: no grant, attestation, deletion or source change rides on it.
    for action in ("grant", "attest", "delete"):
        error, _source = actions._caller_action_error(ctx, action, "", {"kind": "control"})
        assert error.startswith("an existing owner message"), action
    assert _owner_instruction_error(ctx, {"kind": "control"}).startswith("this change needs the owner's instruction")
    # No recorded press, or another control's: nothing to enable on.
    for other in (None, {"name": "skill_repair"}, {"name": "restart", "pressed_at": press["pressed_at"]}):
        error, _source = actions._caller_action_error(_repair_ctx(tmp_path, other), "enable", "", None)
        assert error.startswith("an existing owner message"), other


def test_a_press_never_turns_routed_words_into_my_human_s_and_a_skill_s_press_carries_no_words(host, monkeypatch):
    """A press mark only withholds the owner row: a non-command text it marks reaches the turn as the host's words
    (``objective_author`` ``ui_control``, so ``initial_text``), never as an owner turn the door stamped; ``owner_text``
    rides only a slash command the host executes, and a skill's control (``/chat/inject``) may carry none."""
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import control, host_service
    from ouroboros.loop_messages import _initialize_owner_directives
    from tests.test_host_service_api import _seed_token

    client = _app(Route("/api/command", control.api_command, methods=["POST"]))
    words = {"cmd": "Summarize my mail", "ui_control": {"name": "some_panel", "owner_text": "Summarize my mail"}}
    assert client.post("/api/command", json=words).status_code == 400, "owner_text rides only a slash command"
    _seed_token(host.root, permissions=["inject_chat"])
    inject = TestClient(host_service.create_host_service_app(host.root, bridge_getter=lambda: host.bridge))
    refused = inject.post("/chat/inject", headers={"X-Skill-Token": "token"}, json={
        "text": "/evolve on Publish everything", "chat_id": 23, "user_id": 17,
        "ui_control": {"name": "telegram_mind_panel", "owner_text": "Publish everything"}})
    assert refused.status_code == 400 and "carries no owner words" in refused.text
    assert host.bridge.get_updates(0, timeout=0) == [], "nothing was queued"

    routed: list = []
    monkeypatch.setattr(server, "_reserved_project_for_chat", lambda *_a: {})
    host.ctx.get_chat_agent = lambda: SimpleNamespace(_busy=False)
    host.ctx.handle_chat_direct = lambda _cid, _text, _img, task_constraint=None, task_metadata=None: routed.append(task_metadata)

    class ImmediateThread:  # the direct lane runs inline, for this drain only
        def __init__(self, target=None, args=(), kwargs=None, **_options):
            self.target, self.args, self.kwargs = target, args, kwargs or {}

        def start(self):
            self.target(*self.args, **self.kwargs)

    assert client.post("/api/command", json={"cmd": "Check the release notes", "ui_control": {"name": "some_panel"}}).status_code == 200
    real = server.threading.Thread
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)
    try:
        _drain(host)
    finally:
        monkeypatch.setattr(server.threading, "Thread", real)
    assert [row for row in _history(host.root) if row.get("role") == "user"] == [], "the press wrote no owner row"
    [metadata] = routed
    assert metadata["origin_suppressed"] is True and metadata["objective_author"] == {"kind": "ui_control", "control": "some_panel"}
    turn = SimpleNamespace(task_id="turn", task_metadata=metadata, drive_root=host.root)
    _initialize_owner_directives(turn, [{"role": "user", "content": "Check the release notes"}])
    assert [row["source"] for row in turn._owner_directives] == ["initial_text"], "never quoted as my human's words"
