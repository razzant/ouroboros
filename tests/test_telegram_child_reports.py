"""A helper's own final report never reaches Telegram; what the owner is told still does.

The owner chose (9 Nov 2026) to stop mirroring helpers' full internal reports as ordinary
chat answers: the web keeps a child's final inside its task card and the root answers the
owner. Real producers (the agent finalizer, the completion-wins builder, the owed-outbox
replay, a helper's ``send_user_message``), the supervisor's delivery seam and message bus,
the shipped bridge handlers and ``TelegramClient`` run here; only Telegram's HTTP is
answered locally, and every Bot API request is recorded. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import urllib.parse
from collections import deque
from types import SimpleNamespace

import httpx
import pytest

from tests.test_telegram_proxy import _Api, _TOKEN, _load

# Long enough for several Telegram messages: the bridge used to split one report into many.
_REPORT = "\n\n".join(f"Finding {index}: the helper read module_{index}.py and checked its contract. " * 6
                      for index in range(40))


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """The shipped bridge behind the real message bus, Telegram answered on a mock transport."""
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus

    plugin, telegram_api, _notifier = _load()
    state = tmp_path / "state"
    state.mkdir()
    settings = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "all",
                "TELEGRAM_SUBAGENT_CARDS": "off", "TELEGRAM_MIRROR_PROGRESS": "off"}
    (state / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    api = _Api(state, {"TELEGRAM_BOT_TOKEN": _TOKEN})
    posts: list[tuple[str, dict]] = []
    real_client = httpx.AsyncClient

    def answer(request):
        method = request.url.path.rsplit("/", 1)[-1]
        body = request.content.decode("utf-8", "replace")
        form = {key: values[0] for key, values in urllib.parse.parse_qs(body).items()} if "=" in body else {"raw": body}
        posts.append((method, form))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 500 + len(posts),
                                                                "chat": {"id": 42}, "date": 0}})

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(answer), **kwargs))
    handle = plugin._make_outbound(api)
    outbound, frames = [], []

    def publish(_topic, event):
        outbound.append(dict(event))
        asyncio.run(handle(event))

    bus = message_bus.LocalChatBridge({})
    bus._broadcast_fn = frames.append
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bus)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", publish)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl,
                           send_with_budget=message_bus.send_with_budget)
    return SimpleNamespace(plugin=plugin, api=api, posts=posts, outbound=outbound, frames=frames,
                           host=host, root=root, delivery=delivery, tmp=tmp_path)


def _sent_texts(world) -> list[str]:
    return [form.get("text", "") for method, form in world.posts if method == "sendMessage"]


def _finish_turn(world, monkeypatch, task, answer):
    """The real loop and finalizer end one task; its terminal sends go through the real seam."""
    from ouroboros import agent_task_pipeline as pipeline, loop
    from ouroboros.tools.registry import ToolRegistry

    child = task.get("delegation_role") == "subagent"
    drive = world.tmp / "drive" if child else world.root
    logs = drive / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: True)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                        lambda *a, **kw: pytest.fail("no daemon in this test"))

    class LLM:
        def default_model(self):
            return "openai-compatible::test"

        def chat(self, **kwargs):
            return {"content": answer, "tool_calls": []}, {"cost": 0.0}

    queued = []

    class Events:
        def put(self, event):
            queued.append(json.loads(json.dumps(event)))

    registry = ToolRegistry(repo_dir=world.tmp, drive_root=drive)
    registry._ctx.task_metadata = dict(task)
    registry._ctx.task_attempt = 1
    registry._ctx.owner_message_admission_lock = threading.RLock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(_accepting_owner_messages=True)
    text, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": task["text"]}], tools=registry, llm=LLM(),
        drive_logs=logs, emit_progress=lambda *a, **kw: None, incoming_messages=queue.Queue(),
        task_id=task["id"], drive_root=drive,
    )
    pending = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=drive, repo_dir=world.tmp), None, None, pending,
                               task, text, usage, trace, 0.0, logs, ctx=registry._ctx, event_queue=Events())
    for event in [*queued, *pending]:
        if event.get("type") == "send_message":
            world.delivery._handle_send_message(event, world.host)


@pytest.mark.parametrize("child", [True, False], ids=["child", "root-control"])
def test_a_natural_final_reaches_telegram_only_from_a_root(bridge, monkeypatch, child):
    task = {"id": "kid-1" if child else "root-1", "type": "task", "chat_id": 1, "text": "Inspect.",
            "_attempt": 1, "budget_drive_root": str(bridge.root)}
    if child:
        task.update(parent_task_id="root-1", root_task_id="root-1", delegation_role="subagent")
    _finish_turn(bridge, monkeypatch, task, _REPORT)
    finals = [event for event in bridge.outbound if event.get("task_id") == task["id"] and not event.get("is_progress")]
    # The report reached the bus and the web (its card shows it) either way ...
    assert finals and finals[-1]["text"] == _REPORT
    assert any(frame.get("content") == _REPORT for frame in bridge.frames)
    if child:
        # ... typed by the finalizer as the child's own words, and the bridge sends nothing.
        assert finals[-1]["delegation_role"] == "subagent" and not finals[-1].get("system_type")
        assert bridge.posts == []
    else:
        sent = _sent_texts(bridge)
        assert len(sent) >= 3 and "Finding 0" in sent[0] and "Finding 39" in sent[-1]


@pytest.mark.parametrize("child", [True, False], ids=["child", "root-control"])
def test_completion_wins_and_its_owed_replay_follow_the_same_rule(bridge, monkeypatch, child):
    """The kept result of a killed worker, first live (RUNNING lineage), then the owed outbox
    row replayed after a restart (lineage recovered from the stored result)."""
    from ouroboros.task_results import write_task_result
    from supervisor import terminal_delivery as td

    tid = "kid-2" if child else "root-2"
    lineage = {"parent_task_id": "root-9", "root_task_id": "root-9", "delegation_role": "subagent"} if child else {}
    row = {"id": tid, "type": "task", "chat_id": 1, **lineage}
    stored = {"result": _REPORT, "status": "completed", **lineage}
    bridge.host.RUNNING = {tid: {"task": row}}
    event = td.build_completed_result_event(bridge.root, row, tid, stored)
    assert not (event.get("progress_meta") or {}).get("delegation_role")  # lineage comes from the seam
    bridge.delivery._handle_send_message(dict(event), bridge.host)
    live = len(_sent_texts(bridge))

    replay_id = tid + "-replayed"
    write_task_result(bridge.root, replay_id, "completed", result=_REPORT + " (kept)", **lineage)
    owed = td.build_completed_result_event(bridge.root, {**row, "id": replay_id}, replay_id,
                                           {**stored, "result": _REPORT + " (kept)"})
    assert td.register_pending_delivery(bridge.root, owed)
    bridge.host.RUNNING = {}
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
    replayed = []  # the bus as the replay sees it: membership observable, nothing outstanding
    replay_bus = SimpleNamespace(put_unless_outstanding=lambda event: replayed.append(event) or True,
                                 delivery_outstanding=lambda _did: False)
    assert td.replay_pending_deliveries(bridge.root, event_queue=replay_bus) == [owed["delivery_id"]]
    for replay in replayed:
        assert replay["_delivery_replay"] is True and "delegation_role" not in (replay.get("progress_meta") or {})
        bridge.delivery._handle_send_message(replay, bridge.host)

    finals = [entry for entry in bridge.outbound if entry.get("task_id") in (tid, replay_id)]
    assert [entry["task_id"] for entry in finals] == [tid, replay_id]
    if child:
        assert all(entry.get("delegation_role") == "subagent" for entry in finals)
        assert bridge.posts == []
    else:
        assert live >= 3 and len(_sent_texts(bridge)) >= 2 * live


def test_what_a_helper_tells_the_owner_still_arrives(bridge, monkeypatch):
    """A helper's deliberate message, its file and question, and the host's notices about it."""
    from ouroboros.task_results import task_result_path
    from ouroboros.tools.control_runtime import _send_user_message
    from tests.test_telegram_quiz_answers import _EVENT

    lineage = {"parent_task_id": "root-3", "root_task_id": "root-3", "delegation_role": "subagent"}
    bridge.host.RUNNING = {"kid-3": {"task": {"id": "kid-3", "type": "task", "chat_id": 1, **lineage}}}
    events = queue.Queue()
    tool_ctx = SimpleNamespace(current_chat_id=1, task_id="kid-3", task_metadata=dict(lineage),
                               event_queue=events, pending_events=[], drive_root=bridge.root,
                               drive_logs=lambda: bridge.tmp / "kid-logs")
    assert "accepted for delivery to owner chat" in _send_user_message(tool_ctx, "Anton, the build is green.")
    bridge.delivery._handle_send_message(events.get_nowait(), bridge.host)
    told = bridge.outbound[-1]
    assert told["system_type"] == "proactive_message" and told["delegation_role"] == "subagent"
    assert _sent_texts(bridge) == ["Anton, the build is green."]

    # The host's own notice about the child (a budget stop) is a System row and goes.
    bridge.delivery._handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "kid-3",
                                          "text": "The helper stopped at its budget.", "role": "system"},
                                         bridge.host)
    assert _sent_texts(bridge)[-1] == "The helper stopped at its budget."
    # A child's typed incident waits for its root, then goes, as before.
    incident = {"type": "send_message", "chat_id": 1, "task_id": "kid-3", "role": "system",
                "system_type": "terminal_incident", "text": "The helper's run ended unexpectedly."}
    data = bridge.tmp / "state"  # the bridge's data dir (``_Api.get_runtime_info``)
    bridge.delivery._handle_send_message(dict(incident), bridge.host)
    assert "The helper's run ended unexpectedly." not in _sent_texts(bridge)
    task_result_path(data, "root-3").parent.mkdir(parents=True, exist_ok=True)
    task_result_path(data, "root-3").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    asyncio.run(bridge.plugin._make_outbound(bridge.api)({**incident, **lineage}))
    assert _sent_texts(bridge)[-1] == "The helper's run ended unexpectedly."

    before = len(bridge.posts)
    asyncio.run(bridge.plugin._make_document(bridge.api)({
        "chat_id": 1, "task_id": "kid-3", **lineage, "file_base64": base64.b64encode(b"report").decode(),
        "filename": "report.txt", "mime": "text/plain"}))
    asyncio.run(bridge.plugin._make_quiz(bridge.api)({**_EVENT, "task_id": "kid-3", **lineage}))
    methods = [method for method, _form in bridge.posts[before:]]
    assert "sendDocument" in methods
    assert any(method == "sendMessage" and "Which db?" in form.get("text", "")
               for method, form in bridge.posts[before:])


@pytest.mark.parametrize("cards", ["on", "off"])
def test_lifecycle_cards_keep_their_own_toggle(bridge, cards):
    """The opt-in card is the helper's place here; its final report is not a second copy."""
    settings_path = bridge.tmp / "state" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings_path.write_text(json.dumps({**settings, "TELEGRAM_SUBAGENT_CARDS": cards}), encoding="utf-8")
    handle = bridge.plugin._make_outbound(bridge.api)
    lineage = {"subagent_task_id": "kid-4", "parent_task_id": "root-4", "root_task_id": "root-4",
               "delegation_role": "subagent", "subagent_role": "researcher", "task_id": "kid-4"}
    asyncio.run(handle({"text": "Subagent kid-4 completed", "is_progress": True,
                        "subagent_event": "completed", **lineage}))
    asyncio.run(handle({"text": _REPORT, "is_progress": False, **lineage}))
    sent = _sent_texts(bridge)
    assert len(sent) == (1 if cards == "on" else 0)
    if sent:
        assert sent[0].startswith("✅ researcher — done")


def test_the_mind_panel_no_longer_offers_recent_thoughts(bridge):
    """Recent thoughts is retired; a button from an earlier panel just shows the panel."""
    for mode in ("full_access", "safe_commands"):
        for active in (True, False):
            header, keyboard = bridge.plugin._build_menu_mind(mode, "en", active)
            data = [button["callback_data"] for row in keyboard for button in row]
            assert "cmd_act:bg_thoughts" not in data and data[-1] == "nav:menu"
            assert "Recent thoughts" not in header
    assert not hasattr(bridge.plugin, "_load_recent_thoughts")


def test_an_earlier_panels_thoughts_tap_settles_and_shows_the_current_panel(tmp_path, monkeypatch):
    """A Recent thoughts button left on an old panel answers its spinner and shows today's panel."""
    import tests.test_telegram_quiz_answers as quiz_tests

    plugin = quiz_tests._load_plugin()
    quiz_tests._settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access")
    clients = []

    class Tapped(quiz_tests.Client):
        def __init__(self, token, **kwargs):
            super().__init__(token, **kwargs)
            clients.append(self)

    monkeypatch.setattr(quiz_tests, "Client", Tapped)  # the poller harness installs this client
    monkeypatch.setattr(Tapped, "updates", [{"update_id": 60, "callback_query": {
        "id": "cb60", "data": "cmd_act:bg_thoughts", "from": {"id": 42},
        "message": {"message_id": 77, "chat": {"id": 42, "type": "private"}}}}])
    monkeypatch.setattr(plugin, "_is_bg_consciousness_active", lambda _api: True)
    posts = []
    assert quiz_tests._run_poller(plugin, quiz_tests.Api(tmp_path), monkeypatch, posts) == [] and posts == []
    assert [toast for client in clients for toast in client.toasts] == [("cb60", "")]
    (edit,) = [edit for client in clients for edit in client.edits]
    assert edit[:2] == (42, 77) and "Background Consciousness" in edit[2]
    assert [button["callback_data"] for row in edit[3] for button in row] == ["cmd_act:bg_stop", "nav:menu"]
