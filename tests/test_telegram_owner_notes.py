"""The Telegram mirror carries the owner's notes and skill notices in every mirror mode.

``telegram_only`` forwards only conversations that came from Telegram; a
``reminder`` (a note Ouroboros left) or a ``skill_notice`` (a granted skill's word)
is written for the owner, so it is forwarded to the pinned chat too, as its own
plain message. Everything else keeps the mode's filter.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    package = types.ModuleType("tg_owner_notes_test")
    package.__path__ = [str(root)]
    sys.modules["tg_owner_notes_test"] = package
    spec = importlib.util.spec_from_file_location("tg_owner_notes_test.plugin", root / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    return plugin


class _Api:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def log(self, *_args, **_kwargs):
        pass


class _Client:
    sent: list = []

    def __init__(self, _token, **_kwargs):
        pass

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        _Client.sent.append((chat_id, text, parse_mode))
        return len(_Client.sent)

    async def send_chat_action(self, chat_id, action="typing"):
        pass


@pytest.mark.parametrize("mode", ["telegram_only", "all"])
def test_notes_and_skill_notices_reach_the_pinned_chat(tmp_path, monkeypatch, mode):
    plugin = _load_plugin()
    (tmp_path / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": mode}), encoding="utf-8")
    monkeypatch.setattr(plugin, "TelegramClient", _Client)
    _Client.sent = []
    handle = plugin._make_outbound(_Api(tmp_path))
    web = {"kind": "web"}
    note = {"chat_id": 1, "role": "system", "system_type": "reminder", "markdown": False, "transport": web,
            "text": "Reminder · Ouroboros · written Oct 3 14:05 · for Oct 3 15:00 (UTC+3)\nCall mother"}
    notice = {**note, "system_type": "skill_notice", "text": "Notice · calendar\nMeeting in 15 min"}
    safety = {**note, "system_type": "safety_notice", "text": "Safety Supervisor · not Ouroboros's words\nCall: x"}
    other = {**note, "system_type": "command_reply", "text": "Restarting."}
    reply = {"chat_id": 1, "role": "assistant", "markdown": False, "transport": web, "text": "Hello"}
    for event in (note, notice, safety, other, reply):
        asyncio.run(handle(event))
    forwarded = [text for _chat, text, _mode in _Client.sent]
    assert forwarded[:3] == [note["text"], notice["text"], safety["text"]]
    assert all(chat == 42 and parse_mode == "" for chat, _text, parse_mode in _Client.sent), "plain, to the pinned chat"
    if mode == "telegram_only":
        assert forwarded == [note["text"], notice["text"], safety["text"]], "the mode's filter still holds for the rest"
    else:
        assert forwarded == [note["text"], notice["text"], safety["text"], "Restarting.", "Hello"]


def test_the_settings_say_telegram_only_also_carries_reminders_and_skill_notices(tmp_path, monkeypatch):
    """The mode's explanation and its option name the exception the mirror makes above."""
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "register_miniapp", lambda _api: None)  # its own runtime, not this text
    sections = []

    class _Registrar(_Api):
        def __getattr__(self, _name):  # every other registration is irrelevant here
            return lambda *_args, **_kwargs: None

        def register_settings_section(self, _section_id, title, schema):
            sections.append(schema)

    plugin.register(_Registrar(tmp_path))
    [schema] = sections
    explanation = " ".join(c.get("text", "") for c in schema["components"] if c.get("type") == "markdown")
    fields = [f for c in schema["components"] if c.get("type") == "form" for f in c.get("fields", [])]
    [mode] = [field for field in fields if field.get("name") == "TELEGRAM_MIRROR_MODE"]
    option = {row["value"]: row["label"] for row in mode["options"]}["telegram_only"]
    told = explanation[explanation.index("*Telegram only*"):].split("\n", 1)[0]
    for words in (told, option):
        assert "Telegram" in words and "reminders" in words and "skill notices" in words, words
        assert "Safety notices" in words, words
