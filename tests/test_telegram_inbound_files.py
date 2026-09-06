"""Inbound Telegram files ride the host's shared attachment path (#668)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    package = types.ModuleType("telegram_inbound_test")
    package.__path__ = [str(root)]
    sys.modules["telegram_inbound_test"] = package
    spec = importlib.util.spec_from_file_location("telegram_inbound_test.plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Api:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: "skill-token")

    def log(self, level, message, **fields):
        self.logs.append((level, message))


class Client:
    updates: list = []
    instances: list = []

    def __init__(self, token, **_kwargs):
        self.sent, self.downloads = [], []
        Client.instances.append(self)

    async def call(self, method, **kwargs):
        return {"ok": True, "result": {}}

    async def get_updates(self, offset):
        return list(Client.updates)

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        self.sent.append((chat_id, text))
        return 1

    async def download_file(self, file_id):
        self.downloads.append(file_id)
        return b"%PDF-1.4 bytes"

    async def download_photo(self, file_id):
        raise AssertionError("documents never take the photo path")


def _run(plugin, tmp_path, monkeypatch, message, *, seen_at_inject=None):
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    Client.updates = [{"update_id": 1, "message": {
        "message_id": 5, "chat": {"id": 42, "type": "private"}, "from": {"id": 42, "first_name": "Anton"},
        **message,
    }}]
    Client.instances = []
    injected = []

    async def fake_inject(_api, payload):
        for item in payload.get("attachments") or []:
            path = Path(item["path"])
            if seen_at_inject is not None:
                seen_at_inject[str(path)] = path.read_bytes() if path.is_file() else None
        injected.append(payload)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "TelegramClient", Client)
    monkeypatch.setattr(plugin, "_inject", fake_inject)
    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    api = Api(tmp_path)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    return injected, Client.instances[-1], api


def test_document_with_caption_is_parked_relayed_and_cleaned_up(tmp_path, monkeypatch):
    plugin = _load_plugin()
    seen = {}
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "caption": "please review",
        "document": {"file_id": "f1", "file_name": "report.pdf", "mime_type": "application/pdf", "file_size": 1234},
    }, seen_at_inject=seen)

    assert client.downloads == ["f1"]
    (payload,) = injected
    assert payload["text"] == "please review"
    assert payload["image_caption"] == "please review"
    assert payload["image_base64"] == ""
    (spec,) = payload["attachments"]
    parked = Path(spec["path"])
    assert parked.parent == tmp_path / "inbox", "parked inside this skill's own state root"
    assert parked.name.endswith("_report.pdf")
    assert spec["name"] == "report.pdf" and spec["mime"] == "application/pdf"
    assert seen[str(parked)] == b"%PDF-1.4 bytes", "the file existed when the host was asked to copy it"
    assert not parked.exists(), "the parked copy is removed once the host answered"
    assert payload["transport"]["kind"] == "telegram"
    assert client.sent == []


def test_file_without_caption_is_a_text_less_message(tmp_path, monkeypatch):
    plugin = _load_plugin()
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "voice": {"file_id": "v1", "mime_type": "audio/ogg", "duration": 3, "file_size": 4000},
    })
    (payload,) = injected
    assert payload["text"] == ""
    assert payload["attachments"][0]["name"] == "voice_v1.ogg"
    assert payload["attachments"][0]["mime"] == "audio/ogg"
    assert client.sent == []


def test_parked_file_is_removed_even_when_the_host_refuses(tmp_path, monkeypatch):
    plugin = _load_plugin()
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    Client.updates = [{"update_id": 1, "message": {
        "message_id": 5, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "document": {"file_id": "f2", "file_name": "notes.txt", "mime_type": "text/plain", "file_size": 10},
    }}]
    Client.instances = []

    async def failing_inject(_api, payload):
        raise RuntimeError("Host inject returned HTTP 413")

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "TelegramClient", Client)
    monkeypatch.setattr(plugin, "_inject", failing_inject)
    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    api = Api(tmp_path)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    assert list((tmp_path / "inbox").iterdir()) == []
    assert any("Could not deliver" in text for _chat, text in Client.instances[-1].sent)


def test_oversize_file_is_refused_before_download(tmp_path, monkeypatch):
    plugin = _load_plugin()
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "document": {"file_id": "big", "file_name": "video.mov", "mime_type": "video/quicktime",
                     "file_size": 11 * 1024 * 1024},
    })
    assert injected == []
    assert client.downloads == []
    assert client.sent == [(42, "This file is 11.0 MiB; this integration accepts files up to 10 MiB. "
                                "Send a smaller file or a link.")]
    assert not (tmp_path / "inbox").exists()


def test_unsupported_kinds_get_an_explicit_notice(tmp_path, monkeypatch):
    plugin = _load_plugin()
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "sticker": {"file_id": "s1", "emoji": "👍"},
    })
    assert injected == []
    assert client.sent == [(42, "This kind of message isn't supported — send text, a photo, or a file "
                                "(document, video, audio, voice).")]


def test_inbound_file_descriptor_shapes():
    plugin = _load_plugin()
    inbound = plugin.telegram_inbound
    assert inbound.inbound_file({"text": "hi"}) is None
    doc = inbound.inbound_file({"document": {"file_id": "f", "file_name": "../../evil.sh", "mime_type": "text/x-sh"}})
    assert doc["name"] == "evil.sh" and doc["refusal"] == ""
    note = inbound.inbound_file({"video_note": {"file_id": "abcdefgh12345678", "file_size": 5}})
    assert note["name"] == "video_note_12345678.mp4" and note["mime"] == "video/mp4"
    assert inbound.inbound_file({"audio": {"file_id": "", "file_name": "x.mp3"}}) is None
    assert inbound.refusal_text({"size": 12 * 1024 * 1024}, "ru").startswith("Файл весит 12.0 МиБ")
