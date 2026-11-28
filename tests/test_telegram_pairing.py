"""Who becomes the Telegram bridge's owner: the setup wizard's one-time link, else the first private writer.

The shipped poller and owner gate run on synthetic updates (``tests/test_telegram_bridge_safety.py`` doubles); the
settings route runs on a request stand-in from the computer running Ouroboros or from the Mini App. Nothing reaches
Telegram, the core's Host Service or the live data root.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import types
import urllib.error
from pathlib import Path

import pytest

from tests.test_telegram_bridge_safety import FakeApi, FakeTelegramClient, _load_plugin

CODE = "Ab_-" + "x9" * 14  # 32 characters: the shape the wizard mints
DIGEST = hashlib.sha256(CODE.encode("utf-8")).hexdigest()


def _pairing_module():
    """The module as the host loads the card's function: by path, outside the skill's package."""
    path = Path(__file__).resolve().parents[1] / "skills" / "telegram" / "lib" / "telegram_pairing.py"
    spec = importlib.util.spec_from_file_location("telegram_pairing_by_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _private(update_id, chat, text):
    return {"update_id": update_id, "message": {"chat": {"id": chat, "type": "private"}, "from": {"id": chat},
                                                "text": text, "date": 2_000_000_000}}


def _run_poller(tmp_path, monkeypatch, settings, updates):
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, **settings}),
                                            encoding="utf-8")
    FakeTelegramClient.updates = updates
    FakeTelegramClient.instances = []
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "_inject", fake_inject)
    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    try:
        asyncio.run(plugin._make_poller(FakeApi(tmp_path))())
    except asyncio.CancelledError:
        pass
    texts = plugin._LOCALIZED_TEXTS["en"]
    sent = [item for client in FakeTelegramClient.instances for item in client.sent]
    return json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")), injected, sent, texts


def test_start_payload_reads_only_a_start_and_only_a_minted_code():
    start_payload = _pairing_module().start_payload

    assert start_payload(f"/start {CODE}") == CODE and start_payload(f"/start@Some_bot {CODE}") == CODE
    assert start_payload("/start") == "" and start_payload("/START@Some_bot") == ""
    assert start_payload("/start short") == "" and start_payload(f"/start {CODE}!") == ""
    assert start_payload("hello") is None and start_payload("/started") is None and start_payload(None) is None


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_a_waiting_link_binds_only_its_own_start_and_the_bot_says_connected(tmp_path, monkeypatch, mode):
    settings, injected, sent, texts = _run_poller(tmp_path, monkeypatch,
                                                  {"TELEGRAM_COMMAND_MODE": mode, "TELEGRAM_PAIRING_DIGEST": DIGEST}, [
        _private(1, 99, "hello, who are you"),  # someone who found the bot first
        _private(2, 77, "/start"),  # a bare Start
        _private(3, 55, "/start " + "Z" * 32),  # someone else's link
        _private(4, 42, f"/start {CODE}"),  # the owner's link
        _private(5, 42, "привет"),
        _private(6, 99, "let me in"),
    ])
    assert settings["TELEGRAM_CHAT_ID"] == "42" and not settings["TELEGRAM_PAIRING_DIGEST"]
    assert [(item["chat_id"], item["text"]) for item in injected] == [(42, "привет")]  # Start itself never reaches Main
    assert (42, texts["connected"]) in sent
    waits = {chat: texts.format("pairing_waits", chat_id=chat) for chat in (99, 77, 55)}
    assert {chat for chat, text in sent if text == waits.get(chat)} == {99, 77, 55}
    assert "99" in waits[99] and "Telegram Chat ID" in waits[99], "the way out without the link names this chat's ID"
    assert not [item for item in sent if item[0] == 99 and item[1] != waits[99]]


def test_a_link_still_held_by_setup_binds_before_its_post_lands(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.onboarding_connections.awaited_code_digest",
                        lambda drive_root, skill: DIGEST if skill == "telegram" else None)
    settings, injected, sent, texts = _run_poller(tmp_path, monkeypatch, {"TELEGRAM_COMMAND_MODE": "strict"}, [
        _private(1, 99, "first!"),
        _private(2, 42, f"/start {CODE}"),
    ])
    assert settings["TELEGRAM_CHAT_ID"] == "42" and injected == []
    assert (99, texts.format("pairing_waits", chat_id=99)) in sent and (42, texts["connected"]) in sent


def test_where_no_link_was_shown_the_first_private_writer_is_the_owner(tmp_path, monkeypatch):
    settings, injected, sent, texts = _run_poller(tmp_path, monkeypatch, {"TELEGRAM_COMMAND_MODE": "strict"}, [
        _private(1, 99, "hello"),
        _private(2, 42, f"/start {CODE}"),  # a link nobody armed binds nobody once an owner exists
    ])
    assert settings["TELEGRAM_CHAT_ID"] == "99" and [item["chat_id"] for item in injected] == [99]
    assert not any("waits for its owner's link" in text for _chat, text in sent)


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_start_from_the_bound_owner_is_answered_by_the_bridge_and_never_a_command(tmp_path, monkeypatch, mode):
    settings, injected, sent, texts = _run_poller(tmp_path, monkeypatch,
                                                  {"TELEGRAM_COMMAND_MODE": mode, "TELEGRAM_CHAT_ID": "42"},
                                                  [_private(1, 42, "/start")])
    assert injected == [] and sent == [(42, texts["connected"])]


class _Request:
    """A settings route request: from the computer running Ouroboros, or carrying the Mini App's mark."""

    def __init__(self, body, *, host="127.0.0.1", miniapp=False, method="POST"):
        self.method, self._body = method, body
        self.client = types.SimpleNamespace(host=host)
        self.headers = {"X-Ouroboros-Telegram-MiniApp": "1"} if miniapp else {}

    async def json(self):
        return self._body


def _save(tmp_path, body, **request):
    plugin = _load_plugin(tmp_path)
    route = plugin._make_settings_save(FakeApi(tmp_path))
    answer = asyncio.run(route(_Request(body, **request)))
    stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) if (tmp_path / "settings.json").exists() else {}
    return json.loads(answer.body), stored, plugin, route


def test_the_setup_post_arms_the_link_as_a_digest_only_from_this_computer(tmp_path):
    answer, stored, _plugin, route = _save(tmp_path, {"TELEGRAM_PAIRING_CODE": CODE, "TELEGRAM_NOTIFY_TASKS": "on"})
    assert answer["ok"] is True and answer["owner_ignored"] is False
    assert stored["TELEGRAM_PAIRING_DIGEST"] == DIGEST and stored["TELEGRAM_NOTIFY_TASKS"] == "on"
    assert CODE not in json.dumps(stored) and "TELEGRAM_PAIRING_CODE" not in stored
    form = json.loads(asyncio.run(route(_Request({}, method="GET"))).body)
    assert "TELEGRAM_PAIRING_DIGEST" not in form and CODE not in json.dumps(form)


@pytest.mark.parametrize("request_kind", [{"miniapp": True}, {"host": "192.168.1.5"}])
def test_a_link_cannot_be_armed_from_the_mini_app_or_another_computer(tmp_path, request_kind):
    answer, stored, *_ = _save(tmp_path, {"TELEGRAM_PAIRING_CODE": CODE}, **request_kind)
    assert answer["owner_ignored"] is True and "TELEGRAM_PAIRING_DIGEST" not in stored


def test_a_link_posted_after_the_owner_is_bound_arms_nothing(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    answer, stored, *_ = _save(tmp_path, {"TELEGRAM_PAIRING_CODE": CODE, "TELEGRAM_NOTIFY_TASKS": "on"})
    assert answer["ok"] is True and answer["owner_ignored"] is False
    assert "TELEGRAM_PAIRING_DIGEST" not in stored and stored["TELEGRAM_NOTIFY_TASKS"] == "on"


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_cards_token_check_names_the_bot_and_never_the_token(monkeypatch):
    telegram_pairing = _pairing_module()
    asked = []

    class _Opener:
        def __init__(self, answer):
            self.answer = answer

        def open(self, url, timeout):
            asked.append(url)
            if isinstance(self.answer, Exception):
                raise self.answer
            return _Answer(json.dumps(self.answer).encode("utf-8"))

    def opener(answer):
        monkeypatch.setattr(telegram_pairing.urllib.request, "build_opener", lambda *handlers: _Opener(answer))

    opener({"ok": True, "result": {"id": 1, "username": "My_test_bot"}})
    assert telegram_pairing.bot_identity("123:SECRET") == {"ok": True, "username": "My_test_bot"}
    opener(urllib.error.HTTPError("https://example.invalid", 401, "Unauthorized", {}, None))
    assert telegram_pairing.bot_identity("123:SECRET") == {"ok": False, "cause": "refused"}
    opener(urllib.error.URLError("no route"))
    refused = telegram_pairing.bot_identity("123:SECRET")
    assert refused == {"ok": False, "cause": "unreachable"} and "SECRET" not in json.dumps(refused)
    asked.clear()
    assert telegram_pairing.bot_identity("not a token") == {"ok": False, "cause": "refused"} and asked == []


def test_a_code_moving_from_the_wizard_to_the_skill_between_two_reads_is_never_missed(tmp_path, monkeypatch):
    """The wizard's post writes the skill's settings before it leaves the wizard's record: reading the record first
    cannot see neither copy, so a stranger is never bound by the first-writer rule while a shown link waits."""
    def moved(drive_root, skill):  # the post lands right after the record is read: the record is already empty
        stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        (tmp_path / "settings.json").write_text(json.dumps({**stored, "TELEGRAM_PAIRING_DIGEST": DIGEST}),
                                                encoding="utf-8")
        return None

    monkeypatch.setattr("ouroboros.onboarding_connections.awaited_code_digest", moved)
    settings, injected, sent, texts = _run_poller(tmp_path, monkeypatch, {"TELEGRAM_COMMAND_MODE": "strict"}, [
        _private(1, 99, "hello"),
    ])
    assert "TELEGRAM_CHAT_ID" not in settings and injected == []


def test_the_cards_check_asks_the_bot_api_server_the_skill_names(tmp_path):
    saved_api_base = _pairing_module().saved_api_base
    assert saved_api_base(tmp_path) is None
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": "http://127.0.0.1:8081/"}), encoding="utf-8")
    assert saved_api_base(tmp_path) == "http://127.0.0.1:8081"
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": "http://telegram_bot_api:8081/"}),
                                            encoding="utf-8")
    assert saved_api_base(tmp_path) == "http://telegram_bot_api:8081"  # a service name the poller accepts
    for refused in ("file:///etc", "https://user:pw@example.org", "https://example.org/path", "https://example.org/?q",
                    "https://bot.пример.рф"):  # a name in another script: the check asks the cloud, which knows the bot
        (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": refused}), encoding="utf-8")
        assert saved_api_base(tmp_path) is None, refused
