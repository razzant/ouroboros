"""Unified Telegram skill tests: owner dual-lane DM, 4 mirror modes, /mode, /main, and Presence transport."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ouroboros import extension_contributions, icon_geometry
from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.extension_ui_validation import validate_contributions
from ouroboros.skill_loader import parse_skill_manifest_text


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_ROOT = _REPO_ROOT / "skills" / "telegram"


def _load_plugin(pkg_name: str = "tg_unified_test"):
    package = types.ModuleType(pkg_name)
    package.__path__ = [str(_SKILL_ROOT)]
    sys.modules[pkg_name] = package
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.plugin", _SKILL_ROOT / "plugin.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSkillToken:
    def __init__(self, value: str = "skill-tok-123"):
        self._value = value

    def use_in_request(self) -> str:
        return self._value


class _FakeApi:
    def __init__(self, state_dir: Path, data_dir: Path | None = None, token: str = "123456:TESTTOKEN"):
        self.state_dir = state_dir
        self.data_dir = data_dir or state_dir.parent
        self.token = token
        self.logs: List[tuple[str, str]] = []
        self.tasks: Dict[str, Any] = {}
        self.routes: Dict[str, tuple[Any, tuple[str, ...]]] = {}
        self.tools: Dict[str, tuple[Any, str, dict]] = {}
        self.subscriptions: Dict[str, Any] = {}
        self.sections: Dict[str, dict] = {}
        self.unload_hooks: List[Any] = []

    def get_state_dir(self) -> str: return str(self.state_dir)
    def get_runtime_info(self) -> dict: return {"data_dir": str(self.data_dir)}
    def get_settings(self, keys): return {k: self.token for k in keys if k == "TELEGRAM_BOT_TOKEN" and self.token}
    def get_skill_token(self): return _FakeSkillToken()
    def log(self, level: str, message: str) -> None: self.logs.append((level, message))
    def register_supervised_task(self, name: str, coro_fn, **_kwargs) -> None: self.tasks[name] = coro_fn
    def register_route(self, path: str, handler, methods=("GET",)) -> None: self.routes[path] = (handler, tuple(methods))
    def register_tool(self, name: str, fn, *, description: str = "", schema: dict | None = None) -> None: self.tools[name] = (fn, description, schema or {})
    def subscribe_event(self, event_name: str, handler) -> None: self.subscriptions[event_name] = handler
    def register_settings_section(self, section_id: str, *, title: str, schema: dict) -> None: self.sections[section_id] = {"title": title, "schema": schema}
    def on_unload(self, callback) -> None: self.unload_hooks.append(callback)


class _RecordingClient:
    def __init__(self, token: str = "123456:TESTTOKEN", bot_id: str = "123456"):
        self.token, self.bot_id = token, bot_id
        self.calls: List[tuple[str, Any]] = []
        self.sent: List[tuple[int, str, Any]] = []
        self.edited: List[tuple[int, int, str, Any]] = []
        self.answers: List[tuple[str, str]] = []

    async def call(self, method: str, data=None, **_kwargs):
        self.calls.append((method, dict(data) if isinstance(data, dict) else data))
        if method == "getMe": return {"ok": True, "result": {"id": int(self.bot_id), "username": "ouro_test_bot"}}
        if method == "getUpdates": return {"ok": True, "result": []}
        return {"ok": True, "result": True}

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML", **_kwargs) -> int:
        self.sent.append((chat_id, text, None)); return len(self.sent)
    async def send_message_with_inline_keyboard(self, chat_id: int, text: str, keyboard, **_kwargs) -> int:
        self.sent.append((chat_id, text, keyboard)); return len(self.sent)
    async def edit_message_text_with_inline_keyboard(self, chat_id: int, message_id: int, text: str, keyboard, **_kwargs) -> bool:
        self.edited.append((chat_id, message_id, text, keyboard)); return True
    async def answer_callback_query(self, cb_id: str, *, text: str = "") -> None:
        self.answers.append((cb_id, text))


class _FakeSubmitter:
    def __init__(self, *, place_state: str = "active", places_version: int = 1):
        self.places_version = places_version
        self.delivery_reporting_status = "supported"
        self.delivery_reporting_version = 1
        self.presence_wake_signals = 1
        self.presence_identity = 1
        self.presence_reaction_context = 1
        self.presence_explicit_completion = 1
        self.continuation_version = 2
        self.connector_identity: Dict[str, Any] = {}
        self.place_state = place_state
        self.submitted: List[tuple[dict, list]] = []
        self.polled: List[str] = []
        self.reports: List[dict] = []
        self.next_submission = None
        self.next_poll = None

    async def discover_delivery_support(self) -> int: return self.delivery_reporting_version
    async def place_status(self, account_id: str, conversation_id: str, thread_id: str = "") -> Dict[str, Any]:
        return {"state": self.place_state, "place_id": f"place:{conversation_id}", "label": "Owner DM"}
    async def place_hold(self, account_id: str, conversation_id: str, thread_id: str = "") -> str:
        return "" if self.place_state in {"active", "legacy"} else f"Held: Owner DM is {self.place_state}"
    async def submit(self, event: Dict[str, Any], staged_files):
        self.submitted.append((dict(event), list(staged_files))); assert self.next_submission is not None; return self.next_submission
    async def poll(self, work_ref: str, binding_id: str):
        self.polled.append(work_ref); assert self.next_poll is not None; return self.next_poll
    async def report_delivery(self, payload: Dict[str, Any]) -> None: self.reports.append(dict(payload))


class _FakePresenceBotClient:
    download_limit = 20 * 1024 * 1024

    def __init__(self):
        self.sent_chunks: List[tuple[str, str, dict]] = []
        self.sent_media: List[tuple[str, str, str, dict]] = []
        self.operations: List[tuple[str, dict]] = []
        self.moderations: List[tuple[str, dict]] = []
        self.chat_actions: List[tuple[str, str, int | None]] = []
        self._msg_seq = 500

    async def send_text_chunk(self, chat_id: str, text: str, **kwargs) -> Dict[str, Any]:
        self._msg_seq += 1; self.sent_chunks.append((chat_id, text, kwargs))
        return {"message_id": self._msg_seq, "chat": {"id": int(chat_id), "type": "private"}, "text": text}
    async def send_media(self, kind: str, chat_id: str, file_path: Path, **kwargs) -> Dict[str, Any]:
        self._msg_seq += 1; self.sent_media.append((kind, chat_id, str(file_path), kwargs))
        return {"message_id": self._msg_seq, "chat": {"id": int(chat_id), "type": "private"}}
    async def operation(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        self.operations.append((method, dict(parameters))); return {"ok": True}
    async def moderate(self, action: str, parameters: Dict[str, Any]) -> bool:
        self.moderations.append((action, dict(parameters))); return True
    async def send_chat_action(self, chat_id: str, action: str = "typing", *, topic_id: int | None = None) -> bool:
        self.chat_actions.append((chat_id, action, topic_id)); return True


def _dm_update(update_id: int, text: str, *, chat_id: int = 42, message_id: int = 10, reply_to: dict | None = None) -> dict:
    msg: Dict[str, Any] = {
        "message_id": message_id,
        "date": 1_900_000_000,
        "chat": {"id": chat_id, "type": "private", "first_name": "Owner", "username": "owner"},
        "from": {"id": chat_id, "is_bot": False, "first_name": "Owner", "username": "owner"},
        "text": text,
    }
    if reply_to:
        msg["reply_to_message"] = reply_to
    return {"update_id": update_id, "message": msg}


def test_manifest_icon_contributions_and_registration(tmp_path, monkeypatch):
    manifest_text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    manifest = parse_skill_manifest_text(manifest_text)
    assert manifest.name == "telegram"
    assert manifest.version == "1.5.0"
    assert manifest.raw_extra.get("icon") == "ui/icon.svg"
    assert {"net", "fs", "tool", "presence", "inject_chat", "companion_process"} <= set(manifest.permissions)
    raw_tools = manifest.raw_extra.get("tools") or []
    assert {t["name"] for t in raw_tools} == {
        "telegram_send", "telegram_moderate", "telegram_operation",
        "telegram_react", "telegram_typing", "telegram_receipt",
    }
    raw_svg = (_SKILL_ROOT / "ui" / "icon.svg").read_bytes()
    sanitized = icon_geometry.sanitize_svg(raw_svg)
    assert "<svg" in sanitized

    entries, problems = validate_contributions(manifest, skill_dir=_SKILL_ROOT)
    assert problems == []
    assert len(entries) == 1
    projected = extension_contributions.project_contributions(
        "telegram", entries, "rev-150", "telegram",
        sources={"ui/icon.svg": sanitized.encode("utf-8")},
        tools=raw_tools,
    )
    place_entry = projected[0]
    assert place_entry["provides"] == "place"
    assert place_entry["destination"] == "vendor:Telegram"
    assert place_entry["source"] == {"url": "/api/extensions/telegram/status", "refresh_s": 15}
    assert place_entry["settings"] == "telegram"
    assert len(place_entry["tools"]) == 6

    plugin = _load_plugin("tg_unified_reg_test")
    monkeypatch.setattr(plugin, "register_miniapp", lambda _api: False)
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    api = _FakeApi(state_dir, data_dir=tmp_path)
    plugin.register(api)

    assert {"poller", "presence", "notifier", "typing_cue", "outbox", "token_watch"} <= set(api.tasks)
    assert "status" in api.routes and api.routes["status"][1] == ("GET",)
    assert "settings/save" in api.routes
    assert set(api.tools) == {
        "telegram_send", "telegram_moderate", "telegram_operation",
        "telegram_react", "telegram_typing", "telegram_receipt",
    }
    assert len(api.unload_hooks) == 1
    api.unload_hooks[0]()


def test_validate_bot_single_getme_scopes_commands_and_populates_presence(tmp_path):
    plugin = _load_plugin("tg_unified_validate_test")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _FakeApi(state_dir, data_dir=tmp_path)
    client = _RecordingClient()

    # 1) Unbound install: _validate_bot makes only getMe (never blocks getUpdates behind setMyCommands);
    #    _sync_bot_commands clears public default scope and installs NO owner commands globally.
    asyncio.run(plugin._validate_bot(api, client, "full_access", "en"))
    assert [m for m, _ in client.calls] == ["getMe"]
    asyncio.run(plugin._sync_bot_commands(api, client, "full_access", "en", ""))
    assert [m for m, _ in client.calls] == ["getMe", "setMyCommands"]
    default_call = client.calls[1][1]
    assert json.loads(default_call["scope"]) == {"type": "default"}
    assert json.loads(default_call["commands"]) == []

    # 2) Unbound stranger DM is rejected (no first-DM TOFU); Ouroboros-issued one-time pairing binds and scopes commands
    client.calls.clear()
    assert asyncio.run(plugin._authorized_owner(api, client, _dm_update(1, "hi", chat_id=42), "", "en")) is None
    pair_code = plugin.issue_owner_pairing_code(api)["code"]
    bound = asyncio.run(plugin._authorized_owner(api, client, _dm_update(2, f"/pair {pair_code}", chat_id=42), "", "en"))
    assert bound == "42"
    cmd_calls = [data for method, data in client.calls if method == "setMyCommands"]
    assert len(cmd_calls) == 2
    assert json.loads(cmd_calls[0]["scope"]) == {"type": "default"} and json.loads(cmd_calls[0]["commands"]) == []
    assert json.loads(cmd_calls[1]["scope"]) == {"type": "chat", "chat_id": 42}
    cmd_names = [row["command"] for row in json.loads(cmd_calls[1]["commands"])]
    assert cmd_names[0] == "menu"
    assert {"mode", "main", "panic", "bg", "status"} <= set(cmd_names)

    # 3) Bound validation makes single getMe + populates Presence identity; _sync_bot_commands clears default + sets chat scope
    client.calls.clear()
    asyncio.run(plugin._validate_bot(api, client, "full_access", "en"))
    assert [m for m, _ in client.calls] == ["getMe"]
    asyncio.run(plugin._sync_bot_commands(api, client, "full_access", "en", "42"))
    assert [m for m, _ in client.calls] == ["getMe", "setMyCommands", "setMyCommands"]
    rt = plugin._presence_runtime(api)
    assert rt.recorded_bot() == "123456"
    assert rt.submitter.connector_identity == {"sends_as": {"id": "123456", "label": "@ouro_test_bot", "kind": "bot"}}


def test_owner_dm_presence_mode_routes_chat_to_custody_and_keeps_commands_live(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_unified_dual_lane_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"bg_consciousness_enabled": False}), encoding="utf-8")
    (state_dir / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_OWNER_CHAT_MODE": "presence"}),
        encoding="utf-8",
    )
    api = _FakeApi(state_dir, data_dir=tmp_path)
    rt = plugin._presence_runtime(api)
    rt.submitter = _FakeSubmitter(place_state="not_connected")
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")
    client = _RecordingClient()
    injected: List[dict] = []

    async def fake_inject(_api, payload):
        injected.append(dict(payload))

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    # 1) Ordinary DM message in presence mode -> Presence custody (fail-closed, NO /chat/inject) + throttled hint
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(101, "hello circles", message_id=1), 101, "full_access", "en"))
    assert injected == []
    snap = rt.store.status_snapshot()
    assert snap["inbox_waiting"] == 1
    assert len(client.sent) == 1 and "Background Consciousness" in client.sent[0][1]

    # 2) Second burst message within 60s -> enters custody, hint is throttled (no spam)
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(102, "second thought", message_id=2), 102, "full_access", "en"))
    assert injected == []
    assert rt.store.status_snapshot()["inbox_waiting"] == 2
    assert len(client.sent) == 1

    # 3) Owner commands STAY LIVE in presence mode: /menu, /mode, /main, /status, /bg start
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(103, "/menu", message_id=3), 103, "full_access", "en"))
    assert client.sent[-1][2] is not None  # inline keyboard sent
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(104, "/mode", message_id=4), 104, "full_access", "en"))
    assert any(btn["callback_data"] == "set_preset:presence_control" for row in client.sent[-1][2] for btn in row)
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(105, "/status", message_id=5), 105, "full_access", "en"))
    assert [p["text"] for p in injected] == ["/status"]
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(106, "/main check deploy now", message_id=6), 106, "full_access", "en"))
    assert [p["text"] for p in injected] == ["/status", "check deploy now"]
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(107, "/m quick note", message_id=7), 107, "full_access", "en"))
    assert injected[-1]["text"] == "quick note"

    # 4) /main /restart in strict mode is STILL blocked by strict command policy
    before_inject = len(injected)
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(108, "/main /restart", message_id=8), 108, "strict", "en"))
    assert len(injected) == before_inject

    # 5) Switch back to direct mid-flight -> existing 2 Presence rows untouched; new DM goes to Main
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(109, "/mode direct", message_id=9), 109, "full_access", "en"))
    assert json.loads((state_dir / "settings.json").read_text(encoding="utf-8"))["TELEGRAM_OWNER_CHAT_MODE"] == "direct"
    asyncio.run(plugin._handle_owner_message(api, client, _dm_update(110, "back in main", message_id=10), 110, "full_access", "en"))
    assert injected[-1]["text"] == "back in main"
    assert rt.store.status_snapshot()["inbox_waiting"] == 2  # mid-flight Presence rows preserved!
    assert rt.store.offered_groups("123456") == [{"account_id": "123456", "conversation_id": "42", "label": "Owner @owner (42)"}]


def test_mode_callbacks_and_presets_persist_settings(tmp_path):
    plugin = _load_plugin("tg_unified_cb_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    (state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = _FakeApi(state_dir, data_dir=tmp_path)
    client = _RecordingClient()
    confirmed: List[bool] = []

    def cb(data: str) -> dict:
        return {
            "update_id": 200,
            "callback_query": {
                "id": f"cb-{data}",
                "data": data,
                "from": {"id": 42},
                "message": {"message_id": 99, "chat": {"id": 42, "type": "private"}},
            },
        }

    asyncio.run(plugin._handle_poll_callback(api, client, cb("set_preset:presence_control"), "42", "full_access", "en", lambda: confirmed.append(True)))
    saved = json.loads((state_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["TELEGRAM_OWNER_CHAT_MODE"] == "presence"
    assert saved["TELEGRAM_MIRROR_MODE"] == "control_only"
    assert len(confirmed) == 1

    asyncio.run(plugin._handle_poll_callback(api, client, cb("cmd_act:toggle_quizzes"), "42", "full_access", "en", lambda: confirmed.append(True)))
    assert json.loads((state_dir / "settings.json").read_text(encoding="utf-8"))["TELEGRAM_MIRROR_QUIZZES"] == "off"

    asyncio.run(plugin._handle_poll_callback(api, client, cb("set_preset:direct_main"), "42", "full_access", "en", lambda: confirmed.append(True)))
    saved = json.loads((state_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["TELEGRAM_OWNER_CHAT_MODE"] == "direct"
    assert saved["TELEGRAM_MIRROR_MODE"] == "main"


def test_four_mirror_modes_and_presence_room_exclusion():
    plugin = _load_plugin("tg_unified_mirror_test")
    proj_room = project_chat_id("proj_demo_123")
    presence_room = (1 << 40) + 777
    web = {"kind": "web"}
    tg = {"kind": "telegram", "conversation_id": "42"}

    events = {
        "web_main": {"chat_id": 1, "role": "assistant", "transport": web, "text": "Main answer"},
        "web_main_user": {"chat_id": 1, "role": "user", "transport": web, "text": "Owner on web"},
        "web_project": {"chat_id": proj_room, "role": "assistant", "transport": web, "text": "Project chatter"},
        "tg_project": {"chat_id": proj_room, "role": "assistant", "transport": tg, "text": "Reply to TG project turn"},
        "subagent": {"chat_id": 1, "role": "system", "subagent_event": "running", "transport": web},
        "proj_summary": {"chat_id": 1, "role": "system", "system_type": "project_completion_summary", "transport": web},
        "reminder": {"chat_id": 1, "role": "system", "system_type": "reminder", "transport": web},
        "main_notice": {"chat_id": 1, "role": "system", "system_type": "main_notice", "task_type": "presence", "transport": web},
        "quiz_web": {"chat_id": proj_room, "quiz_id": "q1", "task_id": "t1", "transport": web},
        "quiz_tg": {"chat_id": proj_room, "quiz_id": "q2", "task_id": "t2", "transport": tg},
        "presence_room": {"chat_id": presence_room, "role": "assistant", "task_type": "presence", "transport": web},
        "presence_room_tg": {"chat_id": presence_room, "role": "assistant", "task_type": "presence", "transport": tg},
    }

    for mode in ("all", "main", "telegram_only", "control_only"):
        s = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": mode}
        assert plugin._target_chat(s, events["presence_room"]) == 0, f"Presence room must never mirror in {mode}"
        assert plugin._target_chat(s, events["presence_room_tg"]) == 0, f"Presence room must never mirror in {mode}"

    # main mode: Main replies, main_notice, reminder, quizzes, TG-originated; NO web project/subagent/user/summary
    s_main = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "main"}
    assert plugin._target_chat(s_main, events["web_main"]) == 42
    assert plugin._target_chat(s_main, events["main_notice"]) == 42
    assert plugin._target_chat(s_main, events["reminder"]) == 42
    assert plugin._target_chat(s_main, events["tg_project"]) == 42
    assert plugin._target_chat(s_main, events["quiz_web"]) == 42
    assert plugin._target_chat(s_main, events["web_project"]) == 0
    assert plugin._target_chat(s_main, events["web_main_user"]) == 0
    assert plugin._target_chat(s_main, events["subagent"]) == 0
    assert plugin._target_chat(s_main, events["proj_summary"]) == 0

    # control_only: TG-originated, reminder, main_notice, quizzes (toggleable); NO web Main or web Project
    s_ctrl = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "control_only", "TELEGRAM_MIRROR_QUIZZES": "on"}
    assert plugin._target_chat(s_ctrl, events["tg_project"]) == 42
    assert plugin._target_chat(s_ctrl, events["reminder"]) == 42
    assert plugin._target_chat(s_ctrl, events["main_notice"]) == 42
    assert plugin._target_chat(s_ctrl, events["quiz_web"]) == 42
    assert plugin._target_chat(s_ctrl, events["web_main"]) == 0
    assert plugin._target_chat(s_ctrl, events["web_project"]) == 0

    s_ctrl_no_quiz = {**s_ctrl, "TELEGRAM_MIRROR_QUIZZES": "off"}
    assert plugin._target_chat(s_ctrl_no_quiz, events["quiz_web"]) == 0
    assert plugin._target_chat(s_ctrl_no_quiz, events["quiz_tg"]) == 42


def test_non_owner_chats_topics_reactions_discovery_and_status_route(tmp_path):
    plugin = _load_plugin("tg_unified_non_owner_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    (state_dir / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_OWNER_CHAT_MODE": "direct", "words": ["ouro", "alert"]}),
        encoding="utf-8",
    )
    api = _FakeApi(state_dir, data_dir=tmp_path)
    rt = plugin._presence_runtime(api)
    rt.submitter = _FakeSubmitter(places_version=1)
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")
    client = _RecordingClient()

    updates = [
        # Group addition discovery (my_chat_member)
        {
            "update_id": 301,
            "my_chat_member": {
                "chat": {"id": -100555, "type": "supergroup", "title": "Engineering", "is_forum": True},
                "date": 1_900_000_010,
                "new_chat_member": {"status": "member", "user": {"id": 123456, "is_bot": True}},
            },
        },
        # Forum topic message with signal.words match
        {
            "update_id": 302,
            "message": {
                "message_id": 22,
                "date": 1_900_000_020,
                "chat": {"id": -100555, "type": "supergroup", "title": "Engineering", "is_forum": True},
                "from": {"id": 9001, "is_bot": False, "first_name": "Alice", "username": "alice"},
                "message_thread_id": 77,
                "is_topic_message": True,
                "text": "Hey ouro check this",
            },
        },
        # Another person's private DM
        {
            "update_id": 303,
            "message": {
                "message_id": 33,
                "date": 1_900_000_030,
                "chat": {"id": 7001, "type": "private", "first_name": "Mira", "last_name": "Lee", "username": "mira"},
                "from": {"id": 7001, "is_bot": False, "first_name": "Mira", "last_name": "Lee", "username": "mira"},
                "text": "Hi bot!",
            },
        },
        # Owner DM edit in direct mode -> MUST NOT enter Presence inbox (record_only!)
        {
            "update_id": 304,
            "edited_message": {
                "message_id": 10,
                "date": 1_900_000_000,
                "edit_date": 1_900_000_040,
                "chat": {"id": 42, "type": "private", "first_name": "Owner", "username": "owner"},
                "from": {"id": 42, "is_bot": False, "first_name": "Owner", "username": "owner"},
                "text": "edited in direct mode",
            },
        },
        # Reaction on group message
        {
            "update_id": 305,
            "message_reaction": {
                "chat": {"id": -100555, "type": "supergroup", "title": "Engineering", "is_forum": True},
                "message_id": 22,
                "user": {"id": 9001, "is_bot": False, "first_name": "Alice"},
                "date": 1_900_000_050,
                "old_reaction": [],
                "new_reaction": [{"type": "emoji", "emoji": "🔥"}],
            },
        },
    ]

    for upd in updates:
        asyncio.run(plugin._handle_non_owner_update(api, client, upd, "42"))

    # 3 events queued: forum topic message, Mira DM, group reaction (NOT my_chat_member, NOT owner direct-mode edit)
    assert rt.store.status_snapshot()["inbox_waiting"] == 3
    first = rt.store.claim_inbox()
    assert first is not None
    assert first.payload["conversation_id"] == "-100555"
    assert first.payload["thread_id"] == "77"
    assert "signal.words" in first.payload["facts"]

    # Check GET status route payload
    rt._run_task = object()  # simulate active supervised runtime for confirmed_bot()
    plugin.binding_for(api).read_now(api)
    status_handler = plugin._make_presence_status(api)
    resp = asyncio.run(status_handler(None))
    body = json.loads(resp.body)
    assert body["owner_chat_mode"] == "direct"
    assert body["mirror_mode"] == "all"
    assert body["identity"]["sends_as"]["label"] == "@ouro_test_bot"
    offered_ids = {item["conversation_id"] for item in body["connect"]}
    assert offered_ids == {"-100555", "7001", "42"}
    assert "bridge" in body and "mini_app" in body


def test_presence_worker_and_tools_delivery_receipt_and_place_hold(tmp_path, monkeypatch):
    _load_plugin("tg_unified_tools_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    api = _FakeApi(state_dir, data_dir=tmp_path)
    submitter = _FakeSubmitter(place_state="active")
    fake_bot = _FakePresenceBotClient()

    from skills.telegram.lib.presence.host import PresenceSubmission, PresenceWorkResult
    from skills.telegram.lib.presence.runtime import PresenceRuntime
    from skills.telegram.lib.presence.tools import register_presence_tools

    rt = PresenceRuntime(
        api=api,
        state_dir=state_dir,
        submitter=submitter,
        client_factory=lambda: fake_bot,
    )
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")
    register_presence_tools(api, rt)

    # 1) End-to-end v2 continuation: inbox -> submit(continuing) -> poll(completed+output) -> outbox -> delivery report
    submitter.next_submission = PresenceSubmission(
        status="continuing", outcome="deferred", text="", turn_ref="turn-1",
        work_ref="turn-1", binding_id="", delivery_reporting_version=1, continuation_ref="turn-1",
    )
    submitter.next_poll = PresenceWorkResult(
        status="completed", outcome="message", text="Hello from Presence!", work_ref="turn-1",
        delivery_reporting_version=1, output_ref="out-1",
        outputs=({"output_ref": "out-1", "outcome": "message", "text": "Hello from Presence!", "reply_anchor": None},),
    )
    asyncio.run(rt.ingest_update(_dm_update(401, "hi presence", chat_id=7001, message_id=55), account_id="123456"))
    assert asyncio.run(rt.process_one_inbox()) is True
    assert asyncio.run(rt.process_one_work()) is True
    assert asyncio.run(rt.process_one_outbox()) is True
    assert asyncio.run(rt.process_one_delivery_report()) is True
    assert fake_bot.sent_chunks[0][0] == "7001"
    assert fake_bot.sent_chunks[0][1] == "Hello from Presence!"
    assert len(submitter.reports) == 1

    # 2) Tools: JSON string return contract + real extension_dispatch classification + PlaceHeld + resume
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    img = uploads / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    owner_ctx = types.SimpleNamespace(drive_root=tmp_path, task_id="task-owner-photo", task_metadata={})
    send_fn, react_fn, receipt_fn, mod_fn = (
        api.tools["telegram_send"][0], api.tools["telegram_react"][0],
        api.tools["telegram_receipt"][0], api.tools["telegram_moderate"][0],
    )
    queued_raw = send_fn(owner_ctx, chat_id="7001", kind="photo", file_path=str(img), caption="Look", request_id="req-photo-1")
    assert isinstance(queued_raw, str) and json.loads(queued_raw)["state"] == "queued"
    img.unlink()  # frozen copy must survive source deletion!

    # Place silent -> holds outbox without consuming retries
    submitter.place_state = "silent"
    assert asyncio.run(rt.process_one_outbox()) is True
    rec_held = json.loads(receipt_fn(owner_ctx, request_id="req-photo-1", operation="send"))
    assert rec_held["state"] == "held" and rec_held["attempts"] == 0
    assert all("source_path" not in f and "copy_path" not in f for f in rec_held.get("files") or [])

    # Place active -> delivers frozen copy
    submitter.place_state = "active"
    with rt.store._connect() as conn:
        conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-send:req-photo-1'")
    assert asyncio.run(rt.process_one_outbox()) is True
    rec_done = json.loads(receipt_fn(owner_ctx, request_id="req-photo-1", operation="send"))
    assert rec_done["state"] == "delivered" and len(fake_bot.sent_media) == 1

    # Reaction tool -> setMessageReaction
    r_res = json.loads(react_fn(owner_ctx, action="set", chat_id="7001", message_id=55, emoji="👍", request_id="req-react-1"))
    assert r_res["ok"] is True
    assert asyncio.run(rt.process_one_outbox()) is True
    assert fake_bot.operations[-1][0] == "setMessageReaction"
    assert json.loads(receipt_fn(owner_ctx, request_id="req-react-1", operation="operation"))["state"] == "delivered"

    # Real extension_dispatch must classify refusal JSON as TOOL_REPORTED_FAILURE and authorized receipt as OK
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    send_err = _dispatch_extension_tool_result(
        owner_ctx, "telegram_send", {"handler": send_fn}, {"chat_id": "bad", "kind": "message", "text": "x"}
    )
    assert send_err.status == "error" and send_err.code == "TOOL_REPORTED_FAILURE"
    mod_err = _dispatch_extension_tool_result(
        owner_ctx, "telegram_moderate", {"handler": mod_fn}, {"action": "ban_member", "chat_id": "bad", "request_id": "r1"}
    )
    assert mod_err.status == "error" and mod_err.code == "TOOL_REPORTED_FAILURE"
    rec_ok = _dispatch_extension_tool_result(
        owner_ctx, "telegram_receipt", {"handler": receipt_fn}, {"request_id": "req-photo-1", "operation": "send"}
    )
    assert rec_ok.status == "ok" and rec_ok.code == "OK" and json.loads(rec_ok.text)["state"] == "delivered"

    # Non-numeric topic_id refused upfront; editMessageText stores string topic_id preserving per-topic FIFO
    bad_topic = json.loads(send_fn(chat_id="-100555", kind="message", text="x", topic_id="bad"))
    assert bad_topic["ok"] is False
    op_fn = api.tools["telegram_operation"][0]
    with rt.store._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO telegram_messages"
            "(account_id, chat_id, message_id, topic_id, chat_forum, author_id, author_name, is_self, text, media_group_id, album_origin_known) "
            "VALUES ('123456', '-100555', 9, '77', 1, '123456', '', 1, 'orig', '', 0)"
        )
    assert json.loads(send_fn(chat_id="-100555", kind="message", text="first", topic_id="77", request_id="t77-send"))["ok"] is True
    assert json.loads(op_fn(method="editMessageText", chat_id="-100555", message_id=9, text="edit", topic_id=77, request_id="t77-edit"))["ok"] is True
    assert rt.store.outbox_payload("telegram-operation:t77-edit")["topic_id"] == "77"
    with rt.store._connect() as conn:
        conn.execute("UPDATE outbox SET available_at=9999999999 WHERE delivery_id='telegram-send:t77-send'")
    assert rt.store.claim_outbox() is None  # editMessageText MUST NOT overtake earlier pending send in topic 77!


def test_settings_save_validates_unified_fields_and_round_trips_words(tmp_path):
    plugin = _load_plugin("tg_unified_settings_test")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _FakeApi(state_dir, data_dir=tmp_path)
    handler = plugin._make_settings_save(api)

    class _Req:
        def __init__(self, method: str, payload=None, host: str = "127.0.0.1"):
            self.method = method
            self._payload = payload
            self.headers = {}
            self.client = types.SimpleNamespace(host=host)

        async def json(self):
            return self._payload

    # Invalid owner chat mode rejected
    bad = asyncio.run(handler(_Req("POST", {"TELEGRAM_OWNER_CHAT_MODE": "bogus"})))
    assert bad.status_code == 400
    # Invalid mirror mode rejected
    bad_m = asyncio.run(handler(_Req("POST", {"TELEGRAM_MIRROR_MODE": "unknown"})))
    assert bad_m.status_code == 400
    # Invalid management_group_id rejected (must be negative numeric)
    bad_g = asyncio.run(handler(_Req("POST", {"management_group_id": "12345"})))
    assert bad_g.status_code == 400

    # Valid save persists normalized words list + modes
    ok = asyncio.run(handler(_Req("POST", {
        "TELEGRAM_OWNER_CHAT_MODE": "presence",
        "TELEGRAM_MIRROR_MODE": "control_only",
        "TELEGRAM_MIRROR_QUIZZES": "off",
        "words": " alpha \n\n beta \n",
        "management_group_id": "-100123456",
    })))
    assert ok.status_code == 200
    raw = json.loads((state_dir / "settings.json").read_text(encoding="utf-8"))
    assert raw["TELEGRAM_OWNER_CHAT_MODE"] == "presence"
    assert raw["TELEGRAM_MIRROR_MODE"] == "control_only"
    assert raw["TELEGRAM_MIRROR_QUIZZES"] == "off"
    assert raw["words"] == ["alpha", "beta"]
    assert raw["management_group_id"] == "-100123456"

    # GET hydrates words back as newline-separated string
    hydrated = json.loads(asyncio.run(handler(_Req("GET"))).body)
    assert hydrated["words"] == "alpha\nbeta"
    assert hydrated["TELEGRAM_OWNER_CHAT_MODE"] == "presence"
    assert hydrated["TELEGRAM_MIRROR_MODE"] == "control_only"


def test_edited_secret_reply_and_main_quiz_reply_guards(tmp_path, monkeypatch):
    plugin = _load_plugin("tg_unified_secret_edit_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    (state_dir / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_OWNER_CHAT_MODE": "presence"}),
        encoding="utf-8",
    )
    api = _FakeApi(state_dir, data_dir=tmp_path)
    rt = plugin._presence_runtime(api)
    rt.submitter = _FakeSubmitter(place_state="active")
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")
    client = _RecordingClient()

    # Remember one secret-request card (msg 80) and one ordinary quiz card (msg 81)
    plugin.telegram_quiz.remember_quiz(
        api, "123456", "tok-sec",
        {"task_id": "t1", "quiz_id": "q-sec", "chat_id": 42, "message_id": 80, "options": [], "text": "Enter secret", "secret": True},
    )
    plugin.telegram_quiz.remember_quiz(
        api, "123456", "tok-q",
        {"task_id": "t2", "quiz_id": "q-1", "chat_id": 42, "message_id": 81, "options": ["Yes"], "text": "Deploy?"},
    )

    # 1) Edited reply to a secret-request card in presence mode MUST NOT enter Presence custody
    edited_secret = {
        "update_id": 601,
        "edited_message": {
            "message_id": 90,
            "date": 1_900_000_000,
            "edit_date": 1_900_000_005,
            "chat": {"id": 42, "type": "private", "first_name": "Owner"},
            "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
            "reply_to_message": {"message_id": 80},
            "text": "sk-super-secret-value",
        },
    }
    asyncio.run(plugin._handle_non_owner_update(api, client, edited_secret, "42"))
    assert rt.store.status_snapshot()["inbox_waiting"] == 0
    assert any(client.sent)  # local secret-refusal notice delivered

    # 2) Edited slash command in owner DM MUST stay record_only (never enters Presence inbox)
    edited_cmd = {
        "update_id": 602,
        "edited_message": {
            "message_id": 91,
            "date": 1_900_000_000,
            "edit_date": 1_900_000_006,
            "chat": {"id": 42, "type": "private", "first_name": "Owner"},
            "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
            "text": "/main sensitive command",
        },
    }
    asyncio.run(plugin._handle_non_owner_update(api, client, edited_cmd, "42"))
    assert rt.store.status_snapshot()["inbox_waiting"] == 0

    # 3) Replying to a quiz card with `/main <text>` routes stripped `<text>` to Main, NOT as a quiz answer
    injected, quiz_posts = [], []
    async def fake_inject(_api, payload):
        injected.append(dict(payload))
    async def fake_post(_api, path, payload):
        quiz_posts.append((path, dict(payload)))
        return 200, {"ok": True}
    monkeypatch.setattr(plugin, "_inject", fake_inject)
    monkeypatch.setattr(plugin, "_host_post", fake_post)

    reply_main = _dm_update(603, "/main route to main instead", chat_id=42, message_id=92, reply_to={"message_id": 81})
    asyncio.run(plugin._handle_owner_message(api, client, reply_main, 603, "full_access", "en"))
    assert [p["text"] for p in injected] == ["route to main instead"]
    assert quiz_posts == []


def test_token_fencing_cross_bot_hold_and_status_account_mismatch(tmp_path):
    from skills.telegram.lib.presence.tools import register_presence_tools
    plugin = _load_plugin("tg_unified_fence_test")
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    api = _FakeApi(state_dir, data_dir=tmp_path, token="111111:TOKEN_A")
    binding = plugin.binding_for(api); binding.read_now(api)
    rt = plugin._presence_runtime(api); rt.submitter = _FakeSubmitter(place_state="active")
    # Worker-process runtime (no token_watch, _run_task=None) starts with missing token before setup
    w_api = _FakeApi(state_dir, data_dir=tmp_path, token="")
    plugin.binding_for(w_api).read_now(w_api)
    w_rt = plugin._presence_runtime(w_api); w_rt.submitter = _FakeSubmitter(place_state="active"); register_presence_tools(w_api, w_rt)

    assert rt._build_client() is None  # unvalidated token MUST refuse
    _, gen_a = binding.snapshot(); binding.mark_validated(gen_a)
    rt.on_bot_validated("111111:TOKEN_A", "111111", "bot_a")
    assert rt._build_client() is not None
    # Worker picks up initial token live: recorded_bot(), group react without topic_id, and account_id on send
    grp_upd = {"update_id": 9, "message": {"message_id": 44, "date": 1_900_000_000, "chat": {"id": -100555, "type": "supergroup", "title": "Eng"}, "from": {"id": 9, "is_bot": False, "first_name": "A"}, "text": "hi"}}
    assert asyncio.run(rt.ingest_update(grp_upd, account_id="111111")) is True
    w_api.token = "111111:TOKEN_A"
    assert w_rt.recorded_bot() == "111111"
    assert json.loads(w_api.tools["telegram_react"][0](action="set", chat_id="-100555", message_id=44, emoji="👍", request_id="w-react-1"))["ok"] is True
    assert json.loads(w_api.tools["telegram_send"][0](chat_id="7001", kind="message", text="w", request_id="w-send-1"))["ok"] is True
    assert rt.store.outbox_payload("telegram-send:w-send-1")["account_id"] == "111111"
    assert asyncio.run(rt.ingest_update(_dm_update(10, "hi bot A", chat_id=7001, message_id=1), account_id="111111")) is True

    # Token changes to Bot B (before validation) -> Presence clears identity & refuses client
    api.token = "222222:TOKEN_B"; binding.read_now(api); rt.on_token_changed()
    assert rt._build_client() is None and rt.submitter.connector_identity == {}

    # Bot B validates -> worker refreshes live to 222222; Bot A rows hold/defer without deadlocking Bot B
    _, gen_b = binding.snapshot(); binding.mark_validated(gen_b)
    rt.on_bot_validated("222222:TOKEN_B", "222222", "bot_b")
    w_api.token = "222222:TOKEN_B"
    assert w_rt.recorded_bot() == "222222"
    assert asyncio.run(rt.ingest_update(_dm_update(10, "hi bot B", chat_id=7001, message_id=1), account_id="222222")) is True
    assert asyncio.run(rt.process_one_inbox()) is True and asyncio.run(rt.process_one_inbox()) is True and rt.submitter.submitted == []
    claimed = rt.store.claim_inbox()
    assert claimed is not None and claimed.event_id == "telegram:222222:m:7001:1"
    fake_b = _FakePresenceBotClient(); rt._custom_client_factory = lambda: fake_b
    rt.store.enqueue_outbox("telegram-send:bot-b-row", {"kind": "message", "chat_id": "7001", "text": "from B", "account_id": "222222", "markdown": False})
    assert all(asyncio.run(rt.process_one_outbox()) for _ in range(3))  # holds w-react-1 & w-send-1, delivers bot-b-row!
    rec = rt.store.delivery_receipt("telegram-send:w-send-1")
    assert rec["state"] == "held" and "111111" in rec["error"]
    assert rt.store.delivery_receipt("telegram-send:bot-b-row")["state"] == "delivered" and len(fake_b.sent_chunks) == 1

    # GET status?account_id=111111 while Bot B (222222) is active hides Bot B's identity/connect
    rt._run_task = object()
    resp = json.loads(asyncio.run(plugin._make_presence_status(api)(types.SimpleNamespace(query_params={"account_id": "111111"}))).body)
    assert resp["account_id"] == "222222" and "identity" not in resp and resp["connect"] is None

    # Permanent rejection clears identity & blocks client creation
    rt._custom_client_factory = None
    rt.on_token_rejected("222222:TOKEN_B", "Unauthorized")
    assert rt._build_client() is None and rt.submitter.connector_identity == {}


def test_unsent_request_retries_on_reactions_and_operations(tmp_path):
    from skills.telegram.lib.presence.runtime import PresenceRuntime
    from skills.telegram.lib.presence.transport import TelegramRequestNotSent

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _FakeApi(state_dir, data_dir=tmp_path)
    submitter = _FakeSubmitter(place_state="active")

    class _FailOnceClient(_FakePresenceBotClient):
        def __init__(self):
            super().__init__()
            self.fail_methods = {"setMessageReaction", "editMessageText"}

        async def operation(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            if method in self.fail_methods:
                self.fail_methods.remove(method)
                raise TelegramRequestNotSent(method, "ConnectError")
            return await super().operation(method, parameters)

    fake_bot = _FailOnceClient()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter, client_factory=lambda: fake_bot)
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")

    # 1) Unsent reaction retries cleanly instead of settling uncertain
    rt.store.enqueue_outbox("telegram-operation:react-retry", {
        "kind": "operation", "method": "setMessageReaction", "chat_id": "7001", "account_id": "123456",
        "parameters": {"chat_id": "7001", "message_id": 5, "reaction": [{"type": "emoji", "emoji": "👍"}], "is_big": False},
        "_reporting": {"version": -1, "origin": {}},
    })
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:react-retry")["state"] == "pending"
    with rt.store._connect() as conn:
        conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-operation:react-retry'")
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:react-retry")["state"] == "delivered"

    # 2) Unsent editMessageText retries cleanly instead of settling uncertain
    rt.store.enqueue_outbox("telegram-operation:edit-retry", {
        "kind": "operation", "method": "editMessageText", "chat_id": "7001", "account_id": "123456",
        "text": "updated", "parameters": {"chat_id": "7001", "message_id": 6, "text": "updated"},
        "_reporting": {"version": -1, "origin": {}},
    })
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:edit-retry")["state"] == "pending"
    with rt.store._connect() as conn:
        conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-operation:edit-retry'")
    assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:edit-retry")["state"] == "delivered"


def test_ingest_update_retries_discovery_preserves_places_and_defers_on_sqlite_busy(tmp_path, monkeypatch):
    import sqlite3
    from skills.telegram.lib.presence.runtime import PresenceRuntime
    from skills.telegram.lib.telegram_poll_binding import InjectNotAccepted

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _FakeApi(state_dir, data_dir=tmp_path)

    class _FlakySubmitter(_FakeSubmitter):
        def __init__(self):
            super().__init__(places_version=0)
            self.delivery_reporting_status = "unavailable"
            self.calls = 0

        async def discover_delivery_support(self) -> int:
            self.calls += 1
            if self.calls == 1:
                self.delivery_reporting_status = "unavailable"
                self.places_version = 0
                return 0
            self.delivery_reporting_status = "supported"
            self.places_version = 1
            return 1

    submitter = _FlakySubmitter()
    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter)
    channel = {"id": -100999, "type": "channel", "title": "News"}
    upd1 = {
        "update_id": 701,
        "channel_post": {
            "message_id": 12,
            "date": 1_900_000_000,
            "chat": channel,
            "sender_chat": channel,
            "text": "Release posted during startup",
        },
    }
    # 1) While Host discovery is still unavailable, startup channel_post is STILL captured with places=True (never dropped!)
    assert asyncio.run(rt.ingest_update(upd1, account_id="123456")) is True
    assert submitter.calls == 1
    lease = rt.store.claim_inbox()
    assert lease is not None and lease.payload["conversation_id"] == "-100999" and "facts" in lease.payload

    # 2) After retry cooldown, discovery retries from "unavailable" and settles places_version=1
    rt._next_discover_at = 0.0
    upd2 = {
        "update_id": 702,
        "channel_post": {
            "message_id": 13,
            "date": 1_900_000_010,
            "chat": channel,
            "sender_chat": channel,
            "text": "Second post after Host ready",
        },
    }
    assert asyncio.run(rt.ingest_update(upd2, account_id="123456")) is True
    assert submitter.calls == 2 and submitter.places_version == 1

    # 3) Transient SQLite lock contention raises InjectNotAccepted so handle_poll_batch rolls back offset and retries
    def _busy(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rt.store, "commit_update", _busy)
    with pytest.raises(InjectNotAccepted):
        asyncio.run(rt.ingest_update({**upd2, "update_id": 703}, account_id="123456"))


def test_notifier_excludes_presence_rooms_and_runtime_stop_is_thread_safe(tmp_path, monkeypatch):
    import threading
    from skills.telegram.lib.presence.outbox_worker import MAX_OUTBOX_ATTEMPTS
    from skills.telegram.lib.presence.runtime import PresenceRuntime
    from skills.telegram.lib.presence.transport import TelegramRequestNotSent
    from skills.telegram.lib.telegram_notifier import _announce_summary

    proj_room = project_chat_id("proj_demo_999")
    presence_room = (1 << 40) + 42
    base = {"summary_kind": "terminal_root_projection", "quiet_effective": False}

    assert _announce_summary({**base, "chat_id": 1}, notify_all=True) is True
    assert _announce_summary({**base, "chat_id": proj_room}, notify_all=True) is True
    assert _announce_summary({**base, "chat_id": presence_room}, notify_all=True) is False
    assert _announce_summary({**base, "chat_id": proj_room, "presence_provenance": {"place_id": "p1"}}, notify_all=True) is False

    # Exhausting MAX_OUTBOX_ATTEMPTS on proven-unsent editMessageText settles "failed", not "uncertain"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    api = _FakeApi(state_dir, data_dir=tmp_path)
    submitter = _FakeSubmitter(place_state="active")

    class _AlwaysUnsent(_FakePresenceBotClient):
        async def operation(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            raise TelegramRequestNotSent(method, "ConnectError")

    rt = PresenceRuntime(api=api, state_dir=state_dir, submitter=submitter, client_factory=lambda: _AlwaysUnsent())
    rt.on_bot_validated(api.token, "123456", "ouro_test_bot")
    rt.store.enqueue_outbox("telegram-operation:edit-exhaust", {
        "kind": "operation", "method": "editMessageText", "chat_id": "7001", "account_id": "123456",
        "text": "updated", "parameters": {"chat_id": "7001", "message_id": 6, "text": "updated"},
        "_reporting": {"version": -1, "origin": {}},
    })
    for _ in range(MAX_OUTBOX_ATTEMPTS):
        with rt.store._connect() as conn:
            conn.execute("UPDATE outbox SET available_at=0 WHERE delivery_id='telegram-operation:edit-exhaust'")
        assert asyncio.run(rt.process_one_outbox()) is True
    assert rt.store.delivery_receipt("telegram-operation:edit-exhaust")["state"] == "failed"

    # PresenceRuntime.stop() called from an unload daemon thread cleanly wakes and stops run() on the loop thread
    async def _run_and_stop_from_other_thread():
        import contextlib
        task = asyncio.create_task(rt.run())
        await asyncio.sleep(0.05)
        stopper = threading.Thread(target=rt.stop)
        stopper.start()
        stopper.join(timeout=5.0)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(_run_and_stop_from_other_thread())
    assert rt._runtime_state == "stopped"

    # TelegramSettingsError in handle_poll_batch rolls back unconfirmed update and propagates instead of swallowing
    from skills.telegram.lib.telegram_poll_binding import PollCursor, handle_poll_batch
    from skills.telegram.scripts.telegram_settings import TelegramSettingsError

    cur = PollCursor(tmp_path / "cursor.json", "https://api.telegram.org", "123456")
    binding = types.SimpleNamespace(current=lambda _g: True)
    async def _auth(*_a, **_kw): return "42"
    async def _boom(*_a, **_kw): raise TelegramSettingsError("corrupt settings.json")
    with pytest.raises(TelegramSettingsError):
        asyncio.run(handle_poll_batch(
            api, _RecordingClient(), [_dm_update(801, "hi", chat_id=42, message_id=7)], cur, binding, 1, "42", 10,
            "full_access", "en", authorize=_auth, is_panic=lambda _u: False, on_message=_boom, on_callback=_boom,
        ))
    assert cur.offset == 0

    # Presence Host transport bypasses ambient HTTP_PROXY and refuses 302 redirects (never leaks X-Skill-Token)
    import http.server
    from skills.telegram.lib.presence.host import PresenceHostHTTPError, UrllibPresenceHostTransport

    seen_tokens: List[str] = []
    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen_tokens.append(self.headers.get("X-Skill-Token", ""))
            if self.path == "/redir":
                self.send_response(302); self.send_header("Location", "http://example.invalid/"); self.end_headers(); return
            body = b'{"ok":true}'
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_a): pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1"); monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        port = srv.server_address[1]
        assert asyncio.run(UrllibPresenceHostTransport().request_json("GET", f"http://127.0.0.1:{port}/ok", headers={"X-Skill-Token": "tok"}, payload=None, timeout_sec=5.0)) == {"ok": True}
        assert seen_tokens == ["tok"]
        with pytest.raises(PresenceHostHTTPError) as exc_info:
            asyncio.run(UrllibPresenceHostTransport().request_json("GET", f"http://127.0.0.1:{port}/redir", headers={"X-Skill-Token": "tok"}, payload=None, timeout_sec=5.0))
        assert exc_info.value.status_code == 302
    finally:
        srv.shutdown(); srv.server_close(); t.join(timeout=5.0)

    # Distinct same-second reactions produce distinct update_keys (no silent replay drop)
    from skills.telegram.lib.telegram_poll_binding import update_key
    r_base = {"chat": {"id": -1005}, "message_id": 9, "date": 1_900_000_000, "user": {"id": 7}}
    assert update_key({"message_reaction": {**r_base, "new_reaction": [{"type": "emoji", "emoji": "👍"}]}}) != update_key({"message_reaction": {**r_base, "new_reaction": []}})
