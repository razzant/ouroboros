import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


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
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}
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
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}
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
        {"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "/panic"}}
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
    assert json.loads((tmp_path / "poll_offset.json").read_text(encoding="utf-8")) == {"offset": 4}
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

    try:
        asyncio.run(plugin._make_poller(api)())
    except plugin.TelegramRequestRejected as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("permanent Telegram rejection must stop the poller")

    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "error",
        "reason_code": "telegram_rejected",
    }
    assert plugin._bridge_status(api) == {
        "state": "error",
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
    assert json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8")) == {
        "state": "ready",
        "reason_code": "",
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


def test_startup_permanent_rejection_keeps_raise_semantics(tmp_path, monkeypatch):
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

    try:
        asyncio.run(plugin._make_poller(api)())
    except plugin.TelegramRequestRejected as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("permanent startup rejection must stop the poller")

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

    try:
        asyncio.run(plugin._make_poller(api)())
    except plugin.TelegramRequestRejected as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("permanent revalidation rejection must stop the poller")

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


def test_manifest_declares_route_permission():
    manifest = Path(__file__).resolve().parents[1] / "skills" / "telegram" / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    assert "route" in text.split("permissions:", 1)[1].split("]", 1)[0]


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
