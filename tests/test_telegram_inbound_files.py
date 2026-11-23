"""Inbound Telegram files ride the host's shared attachment path (#668)."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
import uuid

import pytest
from pathlib import Path

PDF = b"%PDF-1.4 bytes"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4rkAAAAASUVORK5CYII="


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
    bot_id = ""
    updates: list = []
    instances: list = []

    def __init__(self, token, **_kwargs):
        self.token = token
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
        return PDF

    async def download_photo(self, file_id):
        self.downloads.append(file_id)
        return PNG, "image/png"


def _run(plugin, tmp_path, monkeypatch, message, *, seen_at_inject=None, use_host=False, update_id=1):
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    Client.updates = [{"update_id": update_id, "message": {
        "message_id": 5, "chat": {"id": 42, "type": "private"}, "from": {"id": 42, "first_name": "Anton"},
        **message,
    }}]
    Client.instances = []
    injected = []

    async def fake_inject(_api, payload):
        for item in payload.get("attachments") or []:
            if seen_at_inject is not None and "path" in item:
                path = Path(item["path"])
                seen_at_inject[str(path)] = path.read_bytes() if path.is_file() else None
        injected.append(payload)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "TelegramClient", Client)
    if not use_host:
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
        "document": {"file_id": "f1", "file_name": "report.pdf", "mime_type": "application/pdf", "file_size": len(PDF)},
    }, seen_at_inject=seen)

    assert client.downloads == ["f1"]
    (payload,) = injected
    assert payload["text"] == "please review"
    assert payload["image_caption"] == "please review"
    assert payload["image_base64"] == ""
    (spec,) = payload["attachments"]
    parked = Path(spec["path"])
    assert parked.parent == tmp_path / "inbox", "parked inside this skill's own state root"
    assert uuid.UUID(hex=parked.name).hex == parked.name
    assert spec["name"] == "report.pdf" and spec["mime"] == "application/pdf"
    assert seen[str(parked)] == b"%PDF-1.4 bytes", "the file existed when the host was asked to copy it"
    assert not parked.exists(), "the parked copy is removed once the host answered"
    assert payload["transport"]["kind"] == "telegram"
    assert client.sent == []


def test_file_without_caption_is_a_text_less_message(tmp_path, monkeypatch):
    plugin = _load_plugin()
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "voice": {"file_id": "v1", "mime_type": "audio/ogg", "duration": 3, "file_size": len(PDF)},
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
        "document": {"file_id": "f2", "file_name": "report.pdf", "mime_type": "application/pdf", "file_size": len(PDF)},
    }}]
    Client.instances = []

    async def failing_inject(_api, payload):
        assert Path(payload["attachments"][0]["path"]).read_bytes() == PDF
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
    assert len(Client.instances[-1].sent) == 1
    assert "Could not confirm" in Client.instances[-1].sent[0][1]


@pytest.mark.parametrize("caption", ["  please review\n ", ""])
def test_oversize_file_is_refused_before_download(tmp_path, monkeypatch, caption):
    injected, client, _api = _run(_load_plugin(), tmp_path, monkeypatch, {
        **({"caption": caption} if caption else {}),
        "document": {"file_id": "big", "file_name": "video.mov", "mime_type": "video/quicktime",
                     "file_size": 11 * 1024 * 1024},
    })
    (payload,) = injected
    assert payload["text"] == payload["image_caption"] == caption
    assert payload["attachments"] == [{"name": "video.mov", "mime": "video/quicktime", "size": 11 * 1024 * 1024,
                                       "unavailable": "too_large", "limit_bytes": 10 * 1024 * 1024}]
    assert client.downloads == []
    assert client.sent == []
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
    inbound = _load_plugin().telegram_inbound
    assert inbound.inbound_file({"text": "hi"}) is None
    doc = inbound.inbound_file({"document": {"file_id": "f", "file_name": "report.pdf", "mime_type": "application/pdf"}})
    assert doc == {"kind": "document", "file_id": "f", "name": "report.pdf", "mime": "application/pdf", "size": None}
    note = inbound.inbound_file({"video_note": {"file_id": "abcdefgh12345678", "file_size": 5}})
    assert note["name"] == "video_note_12345678.mp4" and note["mime"] == "video/mp4"
    assert inbound.inbound_file({"audio": {"file_id": "", "file_name": "x.mp3"}}) is None
    assert inbound.photo_media({"file_id": "photo", "file_size": 10})["size"] == 10


def test_photo_still_rides_inline(tmp_path, monkeypatch):
    injected, client, _api = _run(_load_plugin(), tmp_path, monkeypatch, {
        "caption": "  look here  ", "photo": [{"file_id": "small"}, {"file_id": "photo", "file_size": len(base64.b64decode(PNG))}],
    })
    (payload,) = injected
    assert client.downloads == ["photo"] and not client.sent
    assert payload["text"] == payload["image_caption"] == "  look here  "
    assert payload["image_base64"] == PNG and payload["image_mime"] == "image/png"
    assert "attachments" not in payload


@pytest.mark.parametrize("known_size", [False, True])
def test_failed_download_keeps_caption_and_only_the_facts_telegram_announced(tmp_path, monkeypatch, known_size):
    plugin = _load_plugin()
    size = {"file_size": len(PDF) + 2} if known_size else {}
    if not known_size:
        async def interrupted(self, file_id):
            self.downloads.append(file_id)
            raise RuntimeError("download interrupted")

        monkeypatch.setattr(Client, "download_file", interrupted)
    injected, client, _api = _run(plugin, tmp_path, monkeypatch, {
        "caption": "  notes  ", "document": {"file_id": "doc", "file_name": "report.pdf", "mime_type": "application/pdf", **size},
    })
    (payload,) = injected
    assert client.downloads == ["doc"] and not client.sent
    assert payload["text"] == "  notes  "
    assert payload["attachments"] == [{"name": "report.pdf", "mime": "application/pdf", "unavailable": "download_failed",
                                       **({"size": size["file_size"]} if known_size else {})}]
    assert not (tmp_path / "inbox").exists()


@pytest.mark.parametrize("receipt_found", [True, False])
def test_unknown_intake_reads_once_and_never_repeats_the_post(tmp_path, monkeypatch, receipt_found):
    plugin = _load_plugin()
    calls = []

    async def host(_api, path, payload, *, method="POST"):
        calls.append((method, path))
        if method == "POST":
            assert payload["text"] == "Read A"
            raise TimeoutError("No reply observed")
        return (200, {"ok": True, "status": "pending"}) if receipt_found else (404, {"ok": False})

    monkeypatch.setattr(plugin, "_host_post", host)
    _, client, _api = _run(plugin, tmp_path, monkeypatch, {"text": "Read A"}, use_host=True)
    assert calls == [("POST", "/chat/inject"), ("GET", "/chat/operations/42:tg:42:5")]
    assert client.sent == ([] if receipt_found else [(42, "Could not confirm that Ouroboros received that Telegram update.")])
    # After a poller restart, Telegram can renumber the update, but this is
    # still the same message and must not trigger another read or send.
    _, replay_client, _api = _run(plugin, tmp_path, monkeypatch, {"text": "Read A"}, use_host=True, update_id=11)
    assert calls == [("POST", "/chat/inject"), ("GET", "/chat/operations/42:tg:42:5")]
    assert replay_client.sent == []


@pytest.mark.parametrize("filename", [r"x\..\..\..\escape.txt", r"C:\outside\data.pdf", "report.pdf"])
def test_parked_bytes_use_native_safe_identity_and_preserve_display_name(tmp_path, monkeypatch, filename):
    import ntpath

    plugin = _load_plugin()
    seen = {}
    injected, _client, _api = _run(plugin, tmp_path, monkeypatch, {
        "document": {"file_id": "named", "file_name": filename, "mime_type": "application/pdf"},
    }, seen_at_inject=seen)
    spec = injected[0]["attachments"][0]
    path = Path(spec["path"])
    assert path.parent == tmp_path / "inbox"
    assert uuid.UUID(hex=path.name).hex == path.name
    assert ntpath.dirname(ntpath.normpath(ntpath.join(r"C:\skill\inbox", path.name))) == r"C:\skill\inbox"
    assert spec["name"] == filename and spec["mime"] == "application/pdf"
    assert seen[str(path)] == b"%PDF-1.4 bytes" and not path.exists()
