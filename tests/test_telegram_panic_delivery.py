import asyncio
from unittest.mock import AsyncMock
import pytest

from tests.test_telegram_bridge_safety import _load_plugin, FakeApi, FakeTelegramClient


def test_retained_telegram_panic_is_rejected_with_visible_reply(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *args: None)
    monkeypatch.setattr(plugin, '_POLL_GENERATION_STARTED', 200)
    inject = AsyncMock()
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
        'message': {'text': '/panic', 'date': 100, 'chat': {'id': 7}, 'from': {'id': 8}}}, 1, 'full_access', 'en'))
    inject.assert_not_awaited()
    assert 'not executed again' in client.sent[0][1]
    assert 'previous outcome is unknown' in client.sent[0][1]


def test_fresh_telegram_panic_failure_is_not_retried(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *args: None)
    monkeypatch.setattr(plugin, '_POLL_GENERATION_STARTED', 100)
    inject = AsyncMock(side_effect=OSError('fake offline'))
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
        'message': {'text': '/panic', 'date': 200, 'chat': {'id': 7}, 'from': {'id': 8}}}, 1, 'full_access', 'en'))
    inject.assert_awaited_once()
    assert 'not confirmed' in client.sent[0][1]


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_genuine_owner_panic_bypasses_ordinary_command_mode(tmp_path, monkeypatch, mode):
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *args: None)
    monkeypatch.setattr(plugin, '_POLL_GENERATION_STARTED', 100)
    inject = AsyncMock()
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
        'message': {'text': '/panic', 'date': 200, 'chat': {'id': 7}, 'from': {'id': 8}}}, 1, mode, 'en'))
    inject.assert_awaited_once()
    assert not client.sent
    assert [row["command"] for row in plugin._bot_commands(mode, "en")].count("panic") == 1


@pytest.mark.parametrize('sent_at', [200, 201, 10000])
def test_panic_accepts_start_boundary_second_without_normal_path_expiry(tmp_path, monkeypatch, sent_at):
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *args: None)
    monkeypatch.setattr(plugin, '_POLL_GENERATION_STARTED', 200.95)
    inject = AsyncMock()
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
        'message': {'text': '/panic', 'date': sent_at, 'chat': {'id': 7}, 'from': {'id': 8}}}, 1, 'full_access', 'en'))
    inject.assert_awaited_once()


@pytest.mark.parametrize('host_start,telegram_date,accepted', [
    (200.95, 101, False),  # Telegram time at import was 100: host ahead rejects a fresh command.
    (90.95, 99, True),    # Host behind admits a command predating that same import.
    (100.95, 100, True),  # Whole-second boundary remains indistinguishable.
])
def test_panic_generation_filter_depends_on_host_clock(tmp_path, monkeypatch, host_start, telegram_date, accepted):
    monkeypatch.setattr('time.time', lambda: host_start)
    plugin = _load_plugin(tmp_path)
    assert plugin._POLL_GENERATION_STARTED == host_start
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *a: None)
    inject = AsyncMock()
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
        'message': {'text': '/panic', 'date': telegram_date, 'chat': {'id': 7}, 'from': {'id': 8}}},
        1, 'strict', 'en'))
    assert inject.await_count == int(accepted)
    if accepted:
        assert client.sent == []
    else:
        assert 'not executed again' in client.sent[0][1]
        assert 'previous outcome is unknown' in client.sent[0][1]


@pytest.mark.parametrize("old", [False, True])
def test_panic_replies_use_the_installs_translation_memory(tmp_path, monkeypatch, old):
    from tests.test_telegram_i18n import _seed

    plugin = _load_plugin(tmp_path)
    root = tmp_path / "translations"
    plugin.telegram_i18n.configure(root)
    _seed(root, "ru", {
        "code:tg.menu.panic_earlier": {"text": "Предыдущее сообщение Паники не выполнено повторно."},
        "code:tg.menu.panic_unconfirmed": {"text": "Паника не подтверждена. Повтора не будет."},
    })
    monkeypatch.setattr(plugin, '_owner_message_seen', lambda *args: None)
    monkeypatch.setattr(plugin, '_POLL_GENERATION_STARTED', 200)
    inject = AsyncMock(side_effect=OSError('fixture transport lost'))
    monkeypatch.setattr(plugin, '_inject', inject)
    client = FakeTelegramClient('fake')
    try:
        asyncio.run(plugin._handle_owner_message(FakeApi(tmp_path), client, {
            'message': {'text': '/panic', 'date': 100 if old else 300,
                        'chat': {'id': 7}, 'from': {'id': 8}}}, 1, 'strict', 'ru'))
    finally:
        plugin.telegram_i18n.configure(None)
    assert inject.await_count == (0 if old else 1)
    assert client.sent[0][1] == ("Предыдущее сообщение Паники не выполнено повторно."
                                if old else "Паника не подтверждена. Повтора не будет.")
