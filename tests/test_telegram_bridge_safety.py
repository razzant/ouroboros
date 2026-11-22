import asyncio
import hashlib
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest


def _park_stops(plugin, monkeypatch):
    """A permanent refusal parks the poller; the parked wait here ends the test run."""
    parked = []

    async def park(_api, fingerprint, _binding=None):
        parked.append(fingerprint)
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "park_until_settings_change", park)
    return parked


def _load_plugin(tmp_path):
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    package = types.ModuleType("telegram_bridge_test")
    package.__path__ = [str(root)]
    sys.modules["telegram_bridge_test"] = package
    spec = importlib.util.spec_from_file_location("telegram_bridge_test.plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeApi:
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
        self.logs.append((level, message, fields))


class FakeTelegramClient:
    bot_id = ""
    instances = []

    def __init__(self, token, **_kwargs):
        self.token = token
        self.sent = []
        self.edited = []
        self.panels = []
        self.edit_result = True
        FakeTelegramClient.instances.append(self)

    async def call(self, method, **kwargs):
        return {"ok": True, "result": {}}

    async def get_updates(self, offset):
        return list(self.updates)

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))

    async def send_message_with_inline_keyboard(self, chat_id, text, keyboard):
        self.panels.append((chat_id, text, keyboard))
        self.sent.append((chat_id, text))

    async def edit_message_text_with_inline_keyboard(self, chat_id, message_id, text, keyboard):
        self.edited.append((chat_id, message_id, text, keyboard))
        return self.edit_result

    async def answer_callback_query(self, callback_query_id, *, text=""):
        self.sent.append(("cb_answer", callback_query_id, text))


def test_existing_corrupt_settings_refuse_owner_rebinding(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text("{", encoding="utf-8")

    def forbidden_client(_token, **_kwargs):
        raise AssertionError("Telegram client must not start with corrupt owner state")

    monkeypatch.setattr(plugin, "TelegramClient", forbidden_client)
    api = FakeApi(tmp_path)

    try:
        asyncio.run(plugin._make_poller(api)())
    except plugin.TelegramSettingsError as exc:
        assert str(exc) == "Telegram settings are invalid."
    else:
        raise AssertionError("corrupt existing settings must fail closed")

    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "settings_invalid",
    }
    assert plugin._bridge_status(api) == {
        "state": "error",
        "owner_bound": False,
        "poller": "failed",
        "command_mode": "strict",
        "mirror_mode": "all",
        "reason_code": "settings_invalid",
    }
    assert any(
        message == "Telegram settings are invalid; owner binding is closed."
        for _, message, _ in api.logs
    )


def test_slash_messages_are_not_injected(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    # Explicitly strict (the default is now full_access); strict must still block slashes.
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, "TELEGRAM_COMMAND_MODE": "strict"}), encoding="utf-8")
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/restart"}}
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []
    monkeypatch.setattr(plugin, "_inject", lambda api, payload: injected.append(payload))

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    poller = plugin._make_poller(FakeApi(tmp_path))

    try:
        asyncio.run(poller())
    except asyncio.CancelledError:
        pass

    assert injected == []
    assert FakeTelegramClient.instances[-1].sent


def test_full_access_injects_raw_slash_commands(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text(
        json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, "TELEGRAM_COMMAND_MODE": "full_access", "TELEGRAM_CHAT_ID": "42"}),
        encoding="utf-8",
    )
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic", "date": int(plugin._POLL_GENERATION_STARTED) + 1}}
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    poller = plugin._make_poller(FakeApi(tmp_path))

    try:
        asyncio.run(poller())
    except asyncio.CancelledError:
        pass

    assert len(injected) == 1
    assert injected[0]["text"] == "/panic"
    assert injected[0]["transport"]["kind"] == "telegram"


def test_full_access_first_chat_pins_silently_and_forwards(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text(
        json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, "TELEGRAM_COMMAND_MODE": "full_access"}),
        encoding="utf-8",
    )
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic", "date": int(plugin._POLL_GENERATION_STARTED) + 1}}
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []
    monkeypatch.setattr(plugin, "_inject", lambda api, payload: injected.append(payload))

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    poller = plugin._make_poller(FakeApi(tmp_path))

    try:
        asyncio.run(poller())
    except asyncio.CancelledError:
        pass

    # First chat is pinned (inbound filter), but pinning is SILENT: the message
    # flows straight through and the raw slash is forwarded. The single
    # "send the command again" confirmation is owned by the core owner-external
    # TOFU (server._process_bridge_updates), not duplicated by the skill.
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["TELEGRAM_CHAT_ID"] == "42"
    assert len(injected) == 1
    assert injected[0]["text"] == "/panic"
    # The skill did not emit its own registration prompt.
    assert all("registered" not in str(s).lower() for s in FakeTelegramClient.instances[-1].sent)


def test_poller_caps_update_batch_and_adds_transport(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    # Pinned owner chat (42) so the batch-cap + transport assertions are tested
    # within the owner-binding regime — all inbound is from the one bound chat
    # (messages from other chats are now correctly rejected by TOFU binding).
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 2, "TELEGRAM_COMMAND_MODE": "strict", "TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42, "username": "alice"}, "text": "one"}},
        {"update_id": 2, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "two"}},
        {"update_id": 3, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "three"}},
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    poller = plugin._make_poller(FakeApi(tmp_path))

    try:
        asyncio.run(poller())
    except asyncio.CancelledError:
        pass

    assert len(injected) == 2
    assert injected[0]["transport"] == {
        "kind": "telegram",
        "conversation_id": "42",
        "sender_label": "Telegram (alice)",
    }
    assert "telegram_chat_id" not in injected[0]


def test_poller_isolates_updates_notifies_only_authorized_owner_and_saves_highest_offset(
    tmp_path, monkeypatch,
):
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text(
        json.dumps({
            "TELEGRAM_MAX_UPDATES_PER_POLL": 20,
            "TELEGRAM_COMMAND_MODE": "full_access",
            "TELEGRAM_CHAT_ID": "42",
        }),
        encoding="utf-8",
    )
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": "bad", "type": "private"}, "from": {"id": 42}, "text": "untrusted"}},
        {"update_id": 2, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "one"}},
        {"update_id": 3, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "two"}},
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    attempts = []

    async def fake_inject(_api, payload):
        attempts.append(payload["text"])
        if payload["text"] == "one":
            raise RuntimeError("sensitive provider detail")

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    api = FakeApi(tmp_path)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass

    assert attempts == ["one", "two"]
    assert json.loads((tmp_path / "poll_offset.json").read_text(encoding="utf-8")) == {
        "offset": 4, "base": "https://api.telegram.org", "bot": "", "handled": {"https://api.telegram.org": []},
    }
    client = FakeTelegramClient.instances[-1]
    notices = [item for item in client.sent if item[0] == 42 and "Could not deliver" in item[1]]
    assert len(notices) == 1
    assert any("ValueError" in message for _, message, _ in api.logs)
    assert any("RuntimeError" in message for _, message, _ in api.logs)
    assert all("sensitive provider detail" not in message for _, message, _ in api.logs)


def test_poller_surfaces_permanent_telegram_rejection_and_records_failure(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)

    class RejectedClient(FakeTelegramClient):
        async def get_updates(self, _offset):
            raise plugin.TelegramRequestRejected(
                "Telegram API rejected getUpdates.",
                status_code=401,
            )

    monkeypatch.setattr(plugin, "TelegramClient", RejectedClient)
    api = FakeApi(tmp_path)
    parked = _park_stops(plugin, monkeypatch)

    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    assert parked == [("token", ("https://api.telegram.org", None))]

    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "telegram_rejected",
    }
    assert plugin._bridge_status(api) == {
        "state": "error",
        "token": "configured",
        "owner_bound": False,
        "poller": "failed",
        "command_mode": "full_access",
        "mirror_mode": "all",
        "reason_code": "telegram_rejected",
    }
    assert any(message == "Telegram polling was permanently rejected." for _, message, _ in api.logs)


def test_poller_backoff_grows_monotonically_to_cap_on_typed_transport_failure(tmp_path, monkeypatch):
    """Typed transient failures keep the poller alive with monotone backoff up
    to the cap; a cancellation during the backoff sleep still stops it."""
    plugin = _load_plugin(tmp_path)

    class OfflineClient(FakeTelegramClient):
        async def get_updates(self, _offset):
            raise plugin.TelegramTransportError("Telegram transport is unavailable.")

    monkeypatch.setattr(plugin, "TelegramClient", OfflineClient)
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 8:
            raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", record_sleep)
    try:
        asyncio.run(plugin._make_poller(FakeApi(tmp_path))())
    except asyncio.CancelledError:
        pass

    assert sleeps[0] == 5
    assert all(later >= earlier for earlier, later in zip(sleeps, sleeps[1:]))
    assert max(sleeps) == 60
    assert sleeps[-2:] == [60, 60]
    # A working bridge that loses Telegram records it once (never Connected while every poll fails).
    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "degraded",
        "reason_code": "telegram_unreachable",
    }


def test_poller_backoff_resets_after_any_successful_poll_and_logs_transitions(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    outcomes = iter([
        plugin.TelegramTransportError("offline one"),
        plugin.TelegramTransportError("offline two"),
        [],  # empty batch still counts as success and resets the backoff
        plugin.TelegramTransportError("offline again"),
    ])

    class FlappingClient(FakeTelegramClient):
        async def get_updates(self, _offset):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(plugin, "TelegramClient", FlappingClient)
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", record_sleep)
    api = FakeApi(tmp_path)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass

    # backoff, grown backoff, healthy pacing sleep, then backoff reset to base
    assert sleeps[0] == 5
    assert sleeps[1] > sleeps[0]
    assert sleeps[3] == 5
    degraded = [m for _lvl, m, _f in api.logs if "degraded" in m]
    recovered = [m for _lvl, m, _f in api.logs if "recovered" in m]
    assert len(degraded) == 2, degraded
    assert "TelegramTransportError" in degraded[0]
    assert len(recovered) == 1, recovered


def test_startup_transient_getme_does_not_kill_poller(tmp_path, monkeypatch):
    """A dead network during startup validation must not raise into the
    supervised-restart budget; validation is retried from the polling loop."""
    plugin = _load_plugin(tmp_path)
    failures = {"count": 0}

    class LateStartClient(FakeTelegramClient):
        updates = []

        async def call(self, method, **kwargs):
            if method == "getMe" and failures["count"] < 1:
                failures["count"] += 1
                raise plugin.TelegramTransportError("Telegram API timed out during getMe.")
            return {"ok": True, "result": {}}

    monkeypatch.setattr(plugin, "TelegramClient", LateStartClient)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    api = FakeApi(tmp_path)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass

    assert failures["count"] == 1
    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "ready",
        "reason_code": "",
    }
    assert any("Telegram poller started" in m for _lvl, m, _f in api.logs)


def test_startup_deferred_validation_with_network_still_down_paces_with_backoff(tmp_path, monkeypatch):
    """After a deferred (transient) startup getMe, a network that STAYS down
    keeps the poller alive: every in-loop revalidation failure paces with the
    monotone 5→60 backoff instead of hot-looping or raising."""
    plugin = _load_plugin(tmp_path)
    failures = {"count": 0}

    class StillDownClient(FakeTelegramClient):
        async def call(self, method, **kwargs):
            if method == "getMe":
                failures["count"] += 1
                raise plugin.TelegramTransportError("Telegram API timed out during getMe.")
            return {"ok": True, "result": {}}

        async def get_updates(self, _offset):
            raise AssertionError("polling must not start before validation succeeds")

    monkeypatch.setattr(plugin, "TelegramClient", StillDownClient)
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 8:
            raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", record_sleep)
    try:
        asyncio.run(plugin._make_poller(FakeApi(tmp_path))())
    except asyncio.CancelledError:
        pass

    assert failures["count"] >= 8  # startup + one revalidation per backoff sleep
    assert sleeps[0] == 5
    assert all(later >= earlier for earlier, later in zip(sleeps, sleeps[1:]))
    assert max(sleeps) == 60
    assert sleeps[-2:] == [60, 60]
    # #376: while validation stays deferred the durable status is truthful —
    # neither `ready` nor an error, and never absent.
    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "degraded",
        "reason_code": "telegram_startup_deferred",
    }


def test_startup_permanent_rejection_parks_with_startup_labeling(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)

    class RejectedStartClient(FakeTelegramClient):
        async def call(self, method, **kwargs):
            if method == "getMe":
                raise plugin.TelegramRequestRejected(
                    "Telegram API getMe returned HTTP 401.", status_code=401,
                )
            return {"ok": True, "result": {}}

    monkeypatch.setattr(plugin, "TelegramClient", RejectedStartClient)
    api = FakeApi(tmp_path)
    parked = _park_stops(plugin, monkeypatch)

    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    assert parked == [("token", ("https://api.telegram.org", None))]

    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "telegram_startup_failed",
    }
    assert any("Telegram token validation failed" in m for _lvl, m, _f in api.logs)


def test_inloop_revalidation_permanent_rejection_uses_startup_labeling(tmp_path, monkeypatch):
    """A permanent getMe rejection during in-loop revalidation is the startup
    token-validation failure deferred by a dead network: it must route through
    the startup labeling (telegram_startup_failed + token-validation log), not
    'telegram_rejected'."""
    plugin = _load_plugin(tmp_path)
    calls = {"getme": 0}

    class DeferredRejectClient(FakeTelegramClient):
        updates = []

        async def call(self, method, **kwargs):
            if method == "getMe":
                calls["getme"] += 1
                if calls["getme"] == 1:
                    raise plugin.TelegramTransportError("network down during startup getMe.")
                raise plugin.TelegramRequestRejected(
                    "Telegram API getMe returned HTTP 401.", status_code=401,
                )
            return {"ok": True, "result": {}}

    monkeypatch.setattr(plugin, "TelegramClient", DeferredRejectClient)
    api = FakeApi(tmp_path)
    parked = _park_stops(plugin, monkeypatch)

    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    assert len(parked) == 1

    assert calls["getme"] == 2
    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "telegram_startup_failed",
    }
    assert any("Telegram token validation failed" in m for _lvl, m, _f in api.logs)
    assert all(
        m != "Telegram polling was permanently rejected." for _lvl, m, _f in api.logs
    )


def test_poller_surfaces_local_failure_to_supervisor(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)

    class BrokenClient(FakeTelegramClient):
        async def get_updates(self, _offset):
            raise RuntimeError("local poller defect")

    monkeypatch.setattr(plugin, "TelegramClient", BrokenClient)
    api = FakeApi(tmp_path)

    try:
        asyncio.run(plugin._make_poller(api)())
    except RuntimeError as exc:
        assert str(exc) == "local poller defect"
    else:
        raise AssertionError("local poller failures must reach the supervisor")

    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "poller_failed",
    }
    assert any(message == "Telegram poller failed (RuntimeError)." for _, message, _ in api.logs)


def test_required_media_subscription_failure_stops_skill_registration(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)

    class RegistrationApi(FakeApi):
        def __init__(self, state_dir, rejected_topic):
            super().__init__(state_dir)
            self.rejected_topic = rejected_topic
            self.subscriptions = []

        def register_supervised_task(self, *_args, **_kwargs):
            pass

        def subscribe_event(self, topic, _handler):
            self.subscriptions.append(topic)
            if topic == self.rejected_topic:
                raise RuntimeError(f"{topic} subscription unavailable")

        def register_route(self, *_args, **_kwargs):
            raise AssertionError("registration must stop before routes")

    def forbidden_miniapp(_api):
        raise AssertionError("registration must stop before Mini App")

    monkeypatch.setattr(plugin, "register_miniapp", forbidden_miniapp)
    expected = {
        "chat.video": ["chat.outbound", "chat.typing", "chat.photo", "chat.video"],
        "chat.document": [
            "chat.outbound",
            "chat.typing",
            "chat.photo",
            "chat.video",
            "chat.document",
        ],
        "chat.links": [
            "chat.outbound",
            "chat.typing",
            "chat.photo",
            "chat.video",
            "chat.document",
            "chat.links",
        ],
    }

    for topic, subscriptions in expected.items():
        api = RegistrationApi(tmp_path, topic)
        try:
            plugin.register(api)
        except RuntimeError as exc:
            assert str(exc) == f"{topic} subscription unavailable"
        else:
            raise AssertionError("required media subscription failure must propagate")

        assert api.subscriptions == subscriptions


def test_strict_mode_with_no_pin_binds_first_chat_and_rejects_others(tmp_path, monkeypatch):
    # Security regression (inject_chat_minimization): in strict/safe mode with no
    # TELEGRAM_CHAT_ID, TOFU binding must pin the FIRST chat and reject all others
    # — arbitrary chats must NOT reach _inject.
    plugin = _load_plugin(tmp_path)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, "TELEGRAM_COMMAND_MODE": "strict"}), encoding="utf-8")
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "hello from owner"}},
        {"update_id": 2, "message": {"chat": {"id": 99, "type": "private"}, "from": {"id": 42}, "text": "intruder"}},
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)

    try:
        asyncio.run(plugin._make_poller(FakeApi(tmp_path))())
    except asyncio.CancelledError:
        pass

    # Only the first (bound) chat's plain text injects; the intruder is rejected.
    assert [item["chat_id"] for item in injected] == [42]
    # The first chat is pinned via TOFU even in strict mode.
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["TELEGRAM_CHAT_ID"] == "42"


def test_ambient_env_chat_id_is_ignored(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    FakeTelegramClient.updates = [
        {"update_id": 1, "message": {"chat": {"id": 99, "type": "private"}, "from": {"id": 99}, "text": "first"}},
        {"update_id": 2, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "env owner"}},
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    try:
        asyncio.run(plugin._make_poller(FakeApi(tmp_path))())
    except asyncio.CancelledError:
        pass

    assert [item["chat_id"] for item in injected] == [99]
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["TELEGRAM_CHAT_ID"] == "99"


def test_markdown_to_telegram_html_placeholder_ordering(tmp_path):
    _load_plugin(tmp_path)
    from telegram_bridge_test.lib.telegram_api import markdown_to_telegram_html
    # Generate a string with more than 10 backtick blocks
    text = " ".join(f"`block{i}`" for i in range(12))
    result = markdown_to_telegram_html(text)
    # Ensure all placeholders replaced correctly without trailing "0" or "1" on any block
    for i in range(12):
        assert f"<code>block{i}</code>" in result
    assert "CODEPLACEHOLDER" not in result


def _run_callback(plugin, monkeypatch, tmp_path, settings, cb_data):
    (tmp_path / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    FakeTelegramClient.updates = [
        {"update_id": 1, "callback_query": {
            "id": "cb", "data": cb_data,
            "message": {"message_id": 5, "chat": {"id": 42, "type": "private"}}, "from": {"id": 42},
        }}
    ]
    monkeypatch.setattr(plugin, "TelegramClient", FakeTelegramClient)
    injected = []

    async def fake_inject(api, payload):
        injected.append(payload)

    monkeypatch.setattr(plugin, "_inject", fake_inject)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    poller = plugin._make_poller(FakeApi(tmp_path))
    try:
        asyncio.run(poller())
    except asyncio.CancelledError:
        pass
    return injected


def test_panel_edit_failure_logs_and_sends_fresh_panel(tmp_path):
    plugin = _load_plugin(tmp_path)
    api = FakeApi(tmp_path)
    client = FakeTelegramClient("token")
    client.edit_result = False
    keyboard = [[{"text": "Back", "callback_data": "nav:menu"}]]

    asyncio.run(plugin._edit_panel(api, client, 42, 5, "Panel", keyboard))

    assert client.edited == [(42, 5, "Panel", keyboard)]
    assert client.panels == [(42, "Panel", keyboard)]
    assert api.logs == [
        ("warning", "Telegram panel edit failed; sending a fresh panel.", {}),
    ]


def test_dead_cmd_callback_is_unknown_and_never_injected(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    settings = {
        "TELEGRAM_MAX_UPDATES_PER_POLL": 20,
        "TELEGRAM_COMMAND_MODE": "full_access",
        "TELEGRAM_CHAT_ID": "42",
    }

    assert _run_callback(plugin, monkeypatch, tmp_path, settings, "cmd:status") == []
    assert not hasattr(plugin, "_CALLBACK_MAP")
    assert (
        "cb_answer",
        "cb",
        plugin._LOCALIZED_TEXTS["en"]["unknown_command"],
    ) in FakeTelegramClient.instances[-1].sent


def test_model_and_budget_buttons_defer_to_miniapp(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    settings = {
        "TELEGRAM_MAX_UPDATES_PER_POLL": 20,
        "TELEGRAM_COMMAND_MODE": "full_access",
        "TELEGRAM_CHAT_ID": "42",
    }
    assert _run_callback(plugin, monkeypatch, tmp_path, settings, "set_model:0") == []
    assert _run_callback(plugin, monkeypatch, tmp_path, settings, "set_budget:100") == []


# --- Pairing by one-time code; the first /panic after any owner binding (lib/telegram_owner.py) ---------------

CODE = "Pair-Code_0123456789abcdefghijkl"
GREETING = "✅ Connected: this chat now talks with Ouroboros. /menu opens the controls; /panic is the emergency stop."
LINK_INVALID = "This link is no longer valid. Open the Telegram card in Ouroboros for a fresh QR code."
LINK_BLOCKED = ("This bot's emergency stop belongs to another Telegram account. Its owner can move it to this chat "
                "only by resetting Ouroboros.")
LINK_RETRY = "Ouroboros could not confirm this connection just now. Open the link again in a moment."
TOKEN = "123:secret-token"


class PairingApi(FakeApi):
    """The bridge's API: the bot token from core Settings; the owner chat and the pairing code live in the skill's
    own settings file (``state_dir``), as the card's ``settings/save`` leaves them."""

    def __init__(self, state_dir, data_dir):
        super().__init__(state_dir)
        self.data_dir = Path(data_dir)
        self.routes, self.tasks = {}, {}

    def get_runtime_info(self):
        return {"data_dir": str(self.data_dir)}

    def get_settings_state(self, keys):
        return {"status": "ok", "values": {key: TOKEN for key in keys if key == "TELEGRAM_BOT_TOKEN"}, "missing": []}

    def register_route(self, path, handler, methods=("GET",)):
        self.routes[path] = (handler, methods)

    def register_supervised_task(self, name, factory, **_kwargs):
        self.tasks[name] = factory


def _owner_mod():
    return sys.modules["telegram_bridge_test.lib.telegram_owner"]


async def _cancel(task):
    """Cancel ``task`` until it has ended: Python 3.10's ``asyncio.wait_for`` loses a cancellation that arrives as
    its inner call completes, so one cancel alone may leave a loop running."""
    while not task.done():
        task.cancel()
        await asyncio.wait({task}, timeout=0.5)
    if not task.cancelled() and task.exception() is not None:
        raise task.exception()


def _update(update_id, chat, text, *, sender=None, chat_type="private", date=0, caption=False):
    message = {"message_id": 100 + update_id, "date": date, "chat": {"id": chat, "type": chat_type},
               "from": {"id": chat if sender is None else sender}}
    message.update({"photo": [{"file_id": "p1"}], "caption": text} if caption else {"text": text})
    return {"update_id": update_id, "message": message}


def _drive(plugin, monkeypatch, api, updates, until, *, post=None, refuse=None):
    """Run the shipped poller over ``updates`` (served once) until ``until(client)``, then let the gate's replies
    finish; returns that client. ``refuse(chat, text)`` may raise or wait inside a send."""
    built = []

    class Client(FakeTelegramClient):
        bot_id = "123"
        served = False

        def __init__(self, token, **kwargs):
            super().__init__(token, **kwargs)
            built.append(self)

        async def call(self, method, **kwargs):
            return {"ok": True, "result": {"id": 123, "username": "pair_bot"} if method == "getMe" else True}

        async def get_updates(self, offset):
            if type(self).served:
                await asyncio.sleep(0.01)
                return []
            type(self).served = True
            return list(updates)

        async def send_message(self, chat_id, text):
            if refuse is not None:
                await refuse(chat_id, text)
            await super().send_message(chat_id, text)

        async def download_photo(self, _file_id):
            return "", ""

    monkeypatch.setattr(plugin, "TelegramClient", Client)
    if post is not None:
        monkeypatch.setattr(plugin, "_host_post", post)

    async def main():
        task = asyncio.create_task(plugin._poller(api))
        deadline = time.monotonic() + 20
        try:
            while not (built and until(built[-1])) and not task.done():
                assert time.monotonic() < deadline, "the poller did not reach the expected state"
                await asyncio.sleep(0.01)
            replies = _owner_mod()._load(api).replies
            while replies:
                await asyncio.gather(*list(replies), return_exceptions=True)
        finally:
            await _cancel(task)

    asyncio.run(main())
    return built[-1]


def _settings(tmp_path, name="bridge", **values):
    bridge = tmp_path / name
    bridge.mkdir(exist_ok=True)
    (bridge / "settings.json").write_text(json.dumps({"TELEGRAM_MAX_UPDATES_PER_POLL": 20, **values}), encoding="utf-8")
    return bridge


def _saved(bridge):
    return json.loads((bridge / "settings.json").read_text(encoding="utf-8"))


def _owner(bridge):
    return _saved(bridge).get("TELEGRAM_CHAT_ID")


def _handled(bridge):
    return lambda _client: (bridge / "poll_offset.json").exists()


def _recording_inject(plugin, monkeypatch):
    injected = []

    async def fake_inject(_api, payload):
        injected.append(payload["text"])

    monkeypatch.setattr(plugin, "_inject", fake_inject)
    return injected


@pytest.fixture
def host_world(tmp_path, monkeypatch):
    """The real Host Service (its /owner/external and /chat/inject) over an isolated, initialized core state and
    the real emergency door; ordinary injected messages are only recorded."""
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.server_control import PanicIngress
    from supervisor import state
    from tests.test_host_service_api import FakeBridge, _seed_token

    data = tmp_path / "data"
    data.mkdir()
    # init/init_state rebind these process-wide (and pin the work-root basis): this test gives them back.
    for name in ("DRIVE_ROOT", "STATE_PATH", "STATE_LAST_GOOD_PATH", "STATE_LOCK_PATH", "TOTAL_BUDGET_LIMIT",
                 "_OPENROUTER_DIAGNOSTIC"):
        monkeypatch.setattr(state, name, getattr(state, name))
    monkeypatch.delenv("OUROBOROS_WORK_ROOT_BASIS", raising=False)
    state.init(data)
    state.init_state()
    _seed_token(data, skill="telegram", token="skill-token", permissions=["inject_chat"])
    bridge = FakeBridge()
    bridge.stopped = threading.Event()
    bridge.panic = PanicIngress(bridge.stopped.set)
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)
    world = types.SimpleNamespace(data=data, bridge=bridge, state=state, injected=[], asked=[], published=bridge)
    world.app = create_host_service_app(data, bridge_getter=lambda: world.published)  # None: the core still starting

    async def post(api, path, payload):
        world.asked.append((path, dict(payload)))
        if path == "/chat/inject" and not str(payload.get("text") or "").strip().lower().startswith("/panic"):
            world.injected.append(payload)  # ordinary intake is not under test here
            return 202, {"ok": True}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=world.app), base_url="http://host") as client:
            response = await client.post(path, headers={"X-Skill-Token": api.get_skill_token().use_in_request()},
                                         json=payload)
        return response.status_code, response.json()

    world.post = post
    yield world
    for thread in threading.enumerate():
        if thread.name == "panic-ingress":
            thread.join(timeout=3)


def _external_pair(world):
    current = world.state.load_state()
    return current.get("owner_external_id"), current.get("owner_external_chat_id")


def _bind_core(world, chat):
    """The core's emergency stop already bound to ``chat`` (as a registration before this test left it)."""
    from ouroboros.gateway.host_owner import bind_external_owner

    assert bind_external_owner(lambda: world.bridge, chat, chat) == "bound"


def _paths(world):
    return [path for path, _payload in world.asked]


def test_a_pending_code_binds_only_the_chat_that_opens_its_link_and_both_slots_at_once(tmp_path, monkeypatch,
                                                                                       host_world):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, host_world.data)
    updates = [_update(1, 77, "hello, I found this bot"), _update(2, 77, "/start"),
               _update(3, 77, f"/start {CODE[:-1]}x"), _update(4, 42, f"/start {CODE}")]
    client = _drive(plugin, monkeypatch, api, updates, lambda client: (42, GREETING) in client.sent,
                    post=host_world.post)

    # The stranger who wrote first bound nothing and reached nobody; the chat that opened the link owns the bridge
    # AND the core's emergency stop (asked first), and its /start never reached the Main chat.
    assert _owner(bridge) == "42" and _paths(host_world) == ["/owner/external"]
    assert _external_pair(host_world) == (42, 42) and host_world.bridge.panic.observed_owner() == (42, 42)
    assert host_world.injected == [] and host_world.bridge.messages == []
    assert sorted(client.sent) == sorted([(77, LINK_INVALID), (77, LINK_INVALID), (42, GREETING)])
    saved = _saved(bridge)  # one write: the owner, the code cleared, its digest kept, so the link never binds again
    pairing = sys.modules["telegram_bridge_test.lib.telegram_pairing"]
    assert saved["TELEGRAM_PAIRING_CODE"] == "" and saved["TELEGRAM_PAIRING_CONSUMED"] == pairing.code_digest(CODE)
    assert saved["TELEGRAM_PAIRING_CLAIM"] == ""  # the claim made before the core was asked is settled
    assert CODE not in (bridge / "settings.json").read_text(encoding="utf-8")
    assert ("info", "Telegram owner bound by the pairing link.", {}) in api.logs


def test_a_group_chat_and_a_used_or_wrong_code_bind_nothing(tmp_path, monkeypatch):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="strict", TELEGRAM_PAIRING_CODE=CODE)
    asked = []

    async def post(_api, path, payload):
        asked.append((path, payload))
        return 200, {"ok": True, "owner": "bound"}

    updates = [_update(1, -100123, f"/start@pair_bot {CODE}", sender=42, chat_type="supergroup"),
               _update(2, 55, f"/start {CODE}", chat_type="group"),  # a group never binds, whatever its id
               _update(3, 42, f"/start {CODE}"), _update(4, 43, f"/start {CODE}"), _update(5, 44, "/start wrong")]
    client = _drive(plugin, monkeypatch, PairingApi(bridge, tmp_path / "data"), updates, _handled(bridge), post=post)

    assert _owner(bridge) == "42"
    # The code is spent: with an owner bound and no code waiting, someone else's /start gets no answer.
    assert client.sent == [(42, GREETING)]
    assert asked == [("/owner/external", {"user_id": 42, "chat_id": 42})]


@pytest.mark.parametrize("saved,binds", [
    ({}, True),                                              # no code saved: the first private writer binds
    ({"TELEGRAM_PAIRING_CODE": "1234"}, True),               # a typed short value is no code
    ({"TELEGRAM_PAIRING_CODE": CODE}, False),                # a pending code: nothing but its link binds
    ({"TELEGRAM_PAIRING_CODE": CODE, "TELEGRAM_PAIRING_CONSUMED": "spent"}, False),  # not its digest: still waits
    ({"TELEGRAM_PAIRING_CODE": CODE, "TELEGRAM_PAIRING_CONSUMED": hashlib.sha256(CODE.encode()).hexdigest()}, True),
])
def test_only_a_pending_code_stops_the_first_private_writer_from_binding(tmp_path, monkeypatch, saved, binds):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", **saved)
    injected = _recording_inject(plugin, monkeypatch)
    client = _drive(plugin, monkeypatch, PairingApi(bridge, tmp_path / "data"), [_update(1, 42, "hello")],
                    _handled(bridge))

    assert (_owner(bridge) == "42") is binds
    assert injected == (["hello"] if binds else []) and client.sent == []


def test_unreadable_settings_bind_nothing_and_ask_the_core_nothing(tmp_path):
    _load_plugin(tmp_path)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "settings.json").write_text("{", encoding="utf-8")
    asked = []

    async def post(*args):
        asked.append(args)
        return 200, {"owner": "bound"}

    async def main():
        client = FakeTelegramClient(TOKEN)
        answers = [await _owner_mod().owner_gate(PairingApi(bridge, tmp_path / "data"), client, update, "", "en", post)
                   for update in (_update(1, 42, f"/start {CODE}"), _update(2, 42, "hello"))]
        return answers, client

    (start, hello), client = asyncio.run(main())
    assert start == (None, True) and hello == (None, False) and asked == [] and client.sent == []
    assert (bridge / "settings.json").read_text(encoding="utf-8") == "{"


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_start_is_the_bridges_own_in_every_command_mode(tmp_path, monkeypatch, mode):
    """The owner's bare or addressed /start is greeted again in every mode; a /start whose payload binds nothing
    is told the link is no longer valid, never "Connected". None is forwarded and no binding moves."""
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE=mode, TELEGRAM_CHAT_ID="42")
    injected = _recording_inject(plugin, monkeypatch)
    client = _drive(plugin, monkeypatch, PairingApi(bridge, tmp_path / "data"),
                    [_update(1, 42, "/start"), _update(2, 42, "/start@pair_bot"), _update(3, 42, "/start old-code"),
                     _update(4, 42, "after")], lambda _client: bool(injected))

    assert sorted(client.sent) == sorted([(42, GREETING), (42, GREETING), (42, LINK_INVALID)])
    assert injected == ["after"] and _owner(bridge) == "42"


def test_an_owner_whose_stale_link_binds_nothing_is_told_so_and_the_waiting_code_still_pairs(tmp_path, monkeypatch,
                                                                                             host_world):
    plugin = _load_plugin(tmp_path)
    _bind_core(host_world, 42)
    bridge = _settings(tmp_path, TELEGRAM_CHAT_ID="42", TELEGRAM_PAIRING_CODE=CODE)
    client = _drive(plugin, monkeypatch, PairingApi(bridge, host_world.data),
                    [_update(1, 42, f"/start {CODE[:-1]}x"), _update(2, 42, f"/start {CODE}")],
                    lambda client: (42, GREETING) in client.sent, post=host_world.post)

    assert sorted(client.sent) == sorted([(42, LINK_INVALID), (42, GREETING)])
    assert _saved(bridge)["TELEGRAM_PAIRING_CODE"] == "" and _paths(host_world) == ["/owner/external"]


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_the_first_panic_after_pairing_stops_in_every_command_mode(tmp_path, monkeypatch, host_world, mode):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE=mode, TELEGRAM_PAIRING_CODE=CODE)
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    client = _drive(plugin, monkeypatch, PairingApi(bridge, host_world.data),
                    [_update(1, 42, f"/start {CODE}"), _update(2, 42, "/panic", date=fresh)],
                    lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post)

    assert host_world.bridge.stopped.wait(3)
    assert _paths(host_world) == ["/owner/external", "/chat/inject"]  # registered once, then one /panic
    assert host_world.asked[-1][1]["text"] == "/panic"
    assert all(text != plugin._LOCALIZED_TEXTS["en"]["panic_unconfirmed"] for _chat, text in client.sent)


def _registered(api, world, until):
    """Run the bridge's own registration task (``keep_owner_registered``) until ``until()``."""
    async def main():
        task = asyncio.create_task(api.tasks["owner_registration"]())
        deadline = time.monotonic() + 10
        try:
            while not until():
                assert time.monotonic() < deadline, "the owner was never registered"
                await asyncio.sleep(0.01)
        finally:
            await _cancel(task)

    asyncio.run(main())


@pytest.mark.parametrize("mode", ["strict", "safe_commands", "full_access"])
def test_an_owner_bound_earlier_is_registered_at_load_and_their_first_panic_is_one_request(tmp_path, monkeypatch,
                                                                                           host_world, mode):
    """An install bound before 1.5.0 whose core slot is still empty: the bridge registers the owner from its load
    on, so the first /panic stops with one request, in every mode. A core slot another chat holds is never taken
    over: that bridge's owner keeps its traffic, its /panic is sent once and refused (the core, asked again after
    the refusal, still names another chat), and the bridge logs why once."""
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(_owner_mod(), "_RETRY_SEC", 0.01)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE=mode, TELEGRAM_CHAT_ID="42")
    api = PairingApi(bridge, host_world.data)
    plugin.register_pairing(api, host_world.post)
    _registered(api, host_world, lambda: host_world.bridge.panic.observed_owner() == (42, 42))
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    _drive(plugin, monkeypatch, api, [_update(1, 42, "/panic", date=fresh)],
           lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post)
    assert host_world.bridge.stopped.wait(3) and _paths(host_world) == ["/owner/external", "/chat/inject"]

    host_world.asked.clear()
    other = _settings(tmp_path, "other", TELEGRAM_COMMAND_MODE=mode, TELEGRAM_CHAT_ID="43")
    api = PairingApi(other, host_world.data)
    plugin.register_pairing(api, host_world.post)
    _registered(api, host_world, lambda: len(host_world.asked) >= 1 and 43 in _owner_mod()._load(api).settled)
    client = _drive(plugin, monkeypatch, api, [_update(1, 43, "/panic", date=fresh)],
                    lambda client: bool(client.sent), post=host_world.post)
    assert _external_pair(host_world) == (42, 42)  # never replaced
    assert _paths(host_world) == ["/owner/external", "/chat/inject", "/owner/external"]  # "other": not resent
    assert client.sent == [(43, plugin._LOCALIZED_TEXTS["en"]["panic_unconfirmed"])]
    assert sum("emergency stop is bound to another chat" in message for _level, message, _f in api.logs) == 1


def test_the_owner_registration_asks_again_until_settled_and_ends_with_the_load(tmp_path, monkeypatch):
    """While the core cannot settle the registration it is asked again after each pause, with one log line for the
    episode; once settled nothing more is asked; closing the load's token binding (unload, Off) ends the loop by
    itself, so a cancellation lost in ``wait_for`` never leaves it running."""
    plugin = _load_plugin(tmp_path)
    owner_mod = _owner_mod()
    monkeypatch.setattr(owner_mod, "_RETRY_SEC", 0.01)
    api = PairingApi(_settings(tmp_path, TELEGRAM_CHAT_ID="42"), tmp_path / "data")
    answers = iter([(503, {"ok": False}), (200, {"ok": True, "owner": "unknown"}), (200, {"ok": True, "owner": "bound"})])
    asked = []

    async def post(_api, path, payload):
        asked.append((path, payload))
        return next(answers)

    async def main():
        task = asyncio.create_task(owner_mod.keep_owner_registered(api, post))
        deadline = time.monotonic() + 10
        while len(asked) < 3:
            assert time.monotonic() < deadline and not task.done()
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert len(asked) == 3 and not task.done()  # settled: asked no more, still watching for a new owner
        plugin.binding_for(api).close()
        await asyncio.wait_for(task, 2)

    asyncio.run(main())
    assert asked == [("/owner/external", {"user_id": 42, "chat_id": 42})] * 3
    assert sum("could not register its owner" in message for _level, message, _f in api.logs) == 1


def test_a_panic_the_door_refused_before_the_owner_was_registered_is_sent_once_more_after_it(tmp_path, monkeypatch,
                                                                                            host_world):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="strict", TELEGRAM_CHAT_ID="42")
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    client = _drive(plugin, monkeypatch, PairingApi(bridge, host_world.data), [_update(1, 42, "/panic", date=fresh)],
                    lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post)

    assert host_world.bridge.stopped.wait(3)
    assert _paths(host_world) == ["/chat/inject", "/owner/external", "/chat/inject"]
    assert client.sent == []


def test_a_panic_refused_just_before_the_loads_own_registration_settled_is_sent_once_more(tmp_path, monkeypatch,
                                                                                          host_world):
    """The door refuses a legacy owner's first /panic, and the bridge's own registration beside the poll loop
    settles that owner before the refusal is handled. The resend follows this /panic's refusal and the door's answer
    now, never what the other task recorded meanwhile: the /panic stops."""
    from ouroboros.gateway.host_owner import bind_external_owner

    plugin = _load_plugin(tmp_path)
    api = PairingApi(_settings(tmp_path, TELEGRAM_COMMAND_MODE="strict", TELEGRAM_CHAT_ID="42"), host_world.data)
    load = _owner_mod()._load(api)

    async def post(api_, path, payload):
        answer = await host_world.post(api_, path, payload)
        if path == "/chat/inject" and answer[0] == 503 and 42 not in load.settled:
            assert bind_external_owner(lambda: host_world.bridge, 42, 42) == "bound"  # the load's registration
            load.settled.add(42)  # settled before this refusal reaches the bridge
        return answer

    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    _drive(plugin, monkeypatch, api, [_update(1, 42, "/panic", date=fresh)],
           lambda _client: host_world.bridge.stopped.is_set(), post=post)
    assert host_world.bridge.stopped.wait(3)
    assert _paths(host_world) == ["/chat/inject", "/owner/external", "/chat/inject"]


def test_a_scan_while_the_core_is_still_starting_binds_the_owner_the_core_saved(tmp_path, monkeypatch, host_world):
    """The core saves the scanner's pair while it has no live door yet (it is starting) and answers ``pending``: the
    bridge binds that same owner at once, and once the core is up the bridge's next pass leaves its door holding the
    pair, so the owner's /panic stops. Published from the start the core answers ``bound`` (the first pairing test
    above): the slot and the bridge's owner are the same chat either way."""
    plugin = _load_plugin(tmp_path)
    monkeypatch.setattr(_owner_mod(), "_RETRY_SEC", 0.01)
    bridge = _settings(tmp_path, TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, host_world.data)
    plugin.register_pairing(api, host_world.post)
    host_world.published = None
    _drive(plugin, monkeypatch, api, [_update(1, 42, f"/start {CODE}")], lambda client: (42, GREETING) in client.sent,
           post=host_world.post)
    assert _owner(bridge) == "42" and _external_pair(host_world) == (42, 42)
    assert host_world.bridge.panic.observed_owner() is None and 42 not in _owner_mod()._load(api).settled

    host_world.published = host_world.bridge
    _registered(api, host_world, lambda: host_world.bridge.panic.observed_owner() == (42, 42))
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    _drive(plugin, monkeypatch, api, [_update(2, 42, "/panic", date=fresh)],
           lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post)
    assert host_world.bridge.stopped.wait(3)


@pytest.mark.parametrize("fault", ["lost_answer", "busy_settings"])
def test_a_pairing_the_core_took_is_bound_at_the_next_pass_when_its_answer_or_the_write_fails(tmp_path, monkeypatch,
                                                                                             host_world, fault):
    """The core saved the scanner's pair, but its answer never reached the bridge, or the bridge's own write failed
    after it: the scanner is asked to open the link again, the claim stays, and the bridge's next pass binds that
    same owner, so the core's slot and the bridge's owner never name different chats. Opening the link that bound
    them then greets the owner."""
    plugin = _load_plugin(tmp_path)
    owner_mod = _owner_mod()
    monkeypatch.setattr(owner_mod, "_RETRY_SEC", 0.01)
    bridge = _settings(tmp_path, TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, host_world.data)
    plugin.register_pairing(api, host_world.post)
    faults, real_merge = [fault], owner_mod.merge_settings

    async def post(api_, path, payload):
        answer = await host_world.post(api_, path, payload)
        if path == "/owner/external" and "lost_answer" in faults:
            faults.remove("lost_answer")
            raise httpx.ReadTimeout("the answer was lost on the way")
        return answer

    def merge(state_dir, patch):
        if getattr(patch, "func", None) is owner_mod.pairing.paired and "busy_settings" in faults:
            faults.remove("busy_settings")
            raise owner_mod.TelegramSettingsError("Telegram settings are busy.")
        return real_merge(state_dir, patch)

    monkeypatch.setattr(owner_mod, "merge_settings", merge)
    client = _drive(plugin, monkeypatch, api, [_update(1, 42, f"/start {CODE}")], lambda client: bool(client.sent),
                    post=post)
    assert client.sent == [(42, LINK_RETRY)] and _external_pair(host_world) == (42, 42)
    assert _owner(bridge) is None and _saved(bridge)["TELEGRAM_PAIRING_CLAIM"].startswith("42:")

    _registered(api, host_world, lambda: _owner(bridge) == "42")
    assert _saved(bridge)["TELEGRAM_PAIRING_CLAIM"] == "" and _saved(bridge)["TELEGRAM_PAIRING_CODE"] == ""
    client = _drive(plugin, monkeypatch, api, [_update(2, 42, f"/start {CODE}")], lambda client: bool(client.sent),
                    post=post)
    assert client.sent == [(42, GREETING)]


@pytest.mark.parametrize("saved,closed", [("-1001234567890", True), ("0", True), ("none", True), ("", False)])
def test_any_saved_owner_chat_keeps_the_bot_closed_to_a_strangers_start(tmp_path, monkeypatch, saved, closed):
    """A saved owner chat that is no private chat's id (a group or channel used as a feed, a typed value) closes
    first-writer binding for a /start as for any message, as the bridge always kept it; an empty one does not."""
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", TELEGRAM_CHAT_ID=saved)
    injected = _recording_inject(plugin, monkeypatch)
    client = _drive(plugin, monkeypatch, PairingApi(bridge, tmp_path / "data"),
                    [_update(1, 77, "/start"), _update(2, 77, "do something for me")], _handled(bridge))
    if closed:
        assert _owner(bridge) == saved and client.sent == [] and injected == []
    else:
        assert _owner(bridge) == "77" and client.sent == [(77, GREETING)] and injected == ["do something for me"]


def test_pairing_another_account_while_the_emergency_stop_is_bound_keeps_the_owner_and_both_controls(tmp_path,
                                                                                                    monkeypatch,
                                                                                                    host_world):
    """The core's emergency stop belongs to the bridge owner 42; the QR opened by account 43 is asked of the core
    first, which refuses: nothing changes (42 keeps its traffic and its /panic, the code keeps waiting), 43 is told
    why, and pairing/status says blocked. The same account pairing again succeeds."""
    plugin = _load_plugin(tmp_path)
    _bind_core(host_world, 42)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", TELEGRAM_CHAT_ID="42",
                       TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, host_world.data)
    plugin.register_pairing(api, host_world.post)
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    client = _drive(plugin, monkeypatch, api,
                    [_update(1, 43, f"/start {CODE}"), _update(2, 42, "still here"),
                     _update(3, 42, "/panic", date=fresh)],
                    lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post)

    assert host_world.bridge.stopped.wait(3) and _external_pair(host_world) == (42, 42)
    assert _owner(bridge) == "42" and _saved(bridge)["TELEGRAM_PAIRING_CODE"] == CODE
    assert [payload["text"] for payload in host_world.injected] == ["still here"]
    assert client.sent == [(43, LINK_BLOCKED)]
    handler, _methods = api.routes["pairing/status"]
    status = json.loads(asyncio.run(handler(None)).body)
    assert status["pending"] is True and status["blocked"] == "emergency_stop_owner"

    again = _settings(tmp_path, "again", TELEGRAM_CHAT_ID="42", TELEGRAM_PAIRING_CODE=CODE)
    client = _drive(plugin, monkeypatch, PairingApi(again, host_world.data), [_update(1, 42, f"/start {CODE}")],
                    lambda client: (42, GREETING) in client.sent, post=host_world.post)
    assert _saved(again)["TELEGRAM_PAIRING_CODE"] == "" and _owner(again) == "42"


def test_a_failed_greeting_never_changes_who_the_owner_is(tmp_path, monkeypatch, host_world):
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, host_world.data)

    async def refuse(chat, _text):
        if chat == 42:
            raise OSError("Telegram did not take the greeting")

    _drive(plugin, monkeypatch, api, [_update(1, 42, f"/start {CODE}"), _update(2, 77, "hello")], _handled(bridge),
           post=host_world.post, refuse=refuse)

    assert _owner(bridge) == "42" and host_world.injected == []  # 77 wrote next in the same batch: it binds nothing
    assert ("warning", "Telegram could not send its pairing reply (OSError).", {}) in api.logs


def test_a_reply_to_someone_else_never_holds_the_owners_panic(tmp_path, monkeypatch, host_world):
    """A stranger's stale link while a code waits is answered beside the update loop: here Telegram takes that
    reply only after the owner's /panic stopped Ouroboros, so a reply sent in line would hold the /panic for good.
    With no code waiting, a stranger's /start gets no answer at all."""
    plugin = _load_plugin(tmp_path)
    _bind_core(host_world, 42)
    bridge = _settings(tmp_path, TELEGRAM_CHAT_ID="42", TELEGRAM_PAIRING_CODE=CODE)
    fresh = int(plugin._POLL_GENERATION_STARTED) + 1
    monkeypatch.setattr(_owner_mod(), "_REPLY_WAIT_SEC", 60.0)  # longer than _drive waits for the stop

    async def stall(chat, _text):
        while chat == 77 and not host_world.bridge.stopped.is_set():
            await asyncio.sleep(0.01)

    client = _drive(plugin, monkeypatch, PairingApi(bridge, host_world.data),
                    [_update(1, 77, "/start stale-link"), _update(2, 42, "/panic", date=fresh)],
                    lambda _client: host_world.bridge.stopped.is_set(), post=host_world.post, refuse=stall)
    assert host_world.bridge.stopped.wait(3) and client.sent == [(77, LINK_INVALID)]

    quiet = _settings(tmp_path, "quiet", TELEGRAM_CHAT_ID="42")
    client = _drive(plugin, monkeypatch, PairingApi(quiet, host_world.data), [_update(1, 77, "/start stale-link")],
                    _handled(quiet), post=host_world.post)
    assert client.sent == [] and _owner(quiet) == "42"


def test_a_captioned_start_with_the_code_is_the_bridges_own_and_never_reaches_the_main_chat(tmp_path, monkeypatch,
                                                                                           host_world):
    plugin = _load_plugin(tmp_path)
    _bind_core(host_world, 42)
    bridge = _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access", TELEGRAM_CHAT_ID="42",
                       TELEGRAM_PAIRING_CODE=CODE)
    client = _drive(plugin, monkeypatch, PairingApi(bridge, host_world.data),
                    [_update(1, 42, f"/start {CODE}", caption=True)], lambda client: (42, GREETING) in client.sent,
                    post=host_world.post)

    assert _saved(bridge)["TELEGRAM_PAIRING_CODE"] == "" and host_world.injected == []
    assert all(CODE not in json.dumps(payload) for _path, payload in host_world.asked)
    assert client.sent == [(42, GREETING)]


def test_a_first_writers_start_consumes_the_code_the_card_saves_after_it(tmp_path, monkeypatch):
    """The setup wizard saves the code just after the bridge loads; the owner's /start <code> may already be
    waiting in Telegram. It binds by the first-writer rule, and the code, saved afterwards, waits for nothing."""
    plugin = _load_plugin(tmp_path)
    bridge = _settings(tmp_path)
    api = PairingApi(bridge, tmp_path / "data")
    client = _drive(plugin, monkeypatch, api, [_update(1, 42, f"/start {CODE}")], _handled(bridge))
    assert _owner(bridge) == "42" and client.sent == [(42, GREETING)]

    saved = _saved(bridge)
    (bridge / "settings.json").write_text(json.dumps({**saved, "TELEGRAM_PAIRING_CODE": CODE}), encoding="utf-8")
    assert _owner_mod().waiting_code(api) is False
    fresh = _settings(tmp_path, "fresh", TELEGRAM_CHAT_ID="42", TELEGRAM_PAIRING_CODE=CODE)
    assert _owner_mod().waiting_code(PairingApi(fresh, tmp_path / "data")) is True  # a code no /start bound waits


def test_pairing_status_names_the_bot_and_a_waiting_code_never_the_code_or_token(tmp_path):
    plugin = _load_plugin(tmp_path)
    owner_mod = _owner_mod()
    bridge = _settings(tmp_path, TELEGRAM_PAIRING_CODE=CODE)
    api = PairingApi(bridge, tmp_path / "data")
    plugin.register_pairing(api, None)
    handler, methods = api.routes["pairing/status"]
    assert methods == ("GET",) and "owner_registration" in api.tasks

    def status():
        response = asyncio.run(handler(None))
        return response.status_code, json.loads(response.body)

    assert status() == (200, {"bot_username": None, "owner_bound": False, "pending": True, "blocked": None})
    owner_mod.note_identity(api, TOKEN, {"ok": True, "result": {"id": 123, "username": "pair_bot"}})
    code, answer = status()
    assert answer["bot_username"] == "pair_bot" and CODE not in json.dumps(answer) and "secret" not in json.dumps(answer)
    owner_mod.note_identity(api, TOKEN, {"ok": True, "result": {"id": 999, "username": "another_bot"}})
    assert status()[1]["bot_username"] is None  # a getMe naming another bot forgets the name
    owner_mod.note_identity(api, TOKEN, {"ok": True, "result": {"id": 123, "username": "pair_bot"}})
    plugin.binding_for(api).observe({"status": "ok", "values": {"TELEGRAM_BOT_TOKEN": "123:replaced-secret"}})
    assert status()[1]["bot_username"] is None  # a replaced token of the same bot shows no name until its own getMe
    (bridge / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    assert status() == (200, {"bot_username": None, "owner_bound": True, "pending": False, "blocked": None})
    (bridge / "settings.json").write_text("{", encoding="utf-8")
    assert status()[0] == 409


def test_the_connection_form_says_what_binds_while_a_code_waits(tmp_path):
    """While a pairing code waits a private message binds nothing, so the status no longer asks for one."""
    plugin = _load_plugin(tmp_path)

    def fields(**saved):
        api = PairingApi(_settings(tmp_path, **saved), tmp_path / "data")
        binding = plugin.binding_for(api)
        binding.read_now(api)
        generation = binding.snapshot()[1]
        binding.mark_validated(generation)
        binding.note(generation, "ready")
        return plugin._connection_fields(api)

    waiting, unpaired = fields(TELEGRAM_PAIRING_CODE=CODE), fields(TELEGRAM_PAIRING_CODE="")
    assert waiting["bridge"]["state"] == unpaired["bridge"]["state"] == "waiting_owner"
    assert waiting["bridge_status"] == ("Connected; waiting for the pairing link (the QR code on the Telegram card) "
                                        "to be opened")
    assert unpaired["bridge_status"] == "Connected; send the bot a private message to bind its owner"


def test_a_notification_never_goes_to_a_former_owner(tmp_path, monkeypatch):
    """A pairing that moved the owner while a notifier cycle was under way: what that cycle addressed to the former
    owner is not sent (a quiet wait, as for a replaced token); the owner in force still receives its notices."""
    _load_plugin(tmp_path)
    notifier = sys.modules["telegram_bridge_test.lib.telegram_notifier"]
    sent = []

    class Client:
        def __init__(self, _token, **_kwargs):
            pass

        async def send_message(self, chat_id, text, parse_mode=""):
            sent.append((chat_id, text))

    monkeypatch.setattr(notifier, "TelegramClient", Client)
    api = PairingApi(_settings(tmp_path, TELEGRAM_CHAT_ID="43"), tmp_path / "data")
    outcome, error = asyncio.run(notifier._push_notification(api, 42, "done"))
    assert outcome == "transient" and isinstance(error, notifier.TelegramBindingChanged) and sent == []
    assert asyncio.run(notifier._push_notification(api, 43, "done")) == ("sent", None) and sent == [(43, "done")]
    cleared = PairingApi(_settings(tmp_path, "cleared", TELEGRAM_CHAT_ID=""), tmp_path / "data")
    assert asyncio.run(notifier._push_notification(cleared, 43, "done"))[0] == "transient"  # the owner disconnected
    never = PairingApi(_settings(tmp_path, "never"), tmp_path / "data")  # a file that never named an owner
    assert asyncio.run(notifier._push_notification(never, 43, "done")) == ("sent", None)
    assert sent == [(43, "done"), (43, "done")]
