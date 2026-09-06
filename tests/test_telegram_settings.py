from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import telegram_settings as telegram_settings_module  # noqa: E402
import sidecar as sidecar_module  # noqa: E402
from telegram_settings import (  # noqa: E402
    MINIAPP_MARKER_HEADER,
    TelegramSettingsError,
    TelegramSettingsObserver,
    load_settings,
    merge_settings,
    miniapp_enabled,
    owner_chat_id,
    request_may_change_owner,
)


def _request(host: str, marker: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers={MINIAPP_MARKER_HEADER: marker},
    )


def _load_plugin():
    root = Path(__file__).parents[1] / "skills" / "telegram"
    package = types.ModuleType("telegram_settings_test")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RouteRequest:
    def __init__(self, payload, *, host=None, marker="") -> None:
        self.payload = payload
        self.headers = {MINIAPP_MARKER_HEADER: marker}
        self.client = None if host is None else SimpleNamespace(host=host)

    async def json(self):
        return self.payload


class _RouteApi:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}


class _InvalidJsonRequest(_RouteRequest):
    async def json(self):
        raise ValueError("malformed body")


def test_settings_merge_is_atomic_and_preserves_unrelated_keys(tmp_path: Path) -> None:
    merge_settings(tmp_path, {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_LANGUAGE": "en"})
    merge_settings(tmp_path, {"TELEGRAM_LANGUAGE": "ru"})
    assert load_settings(tmp_path) == {
        "TELEGRAM_CHAT_ID": "42",
        "TELEGRAM_LANGUAGE": "ru",
    }
    assert owner_chat_id(tmp_path) == 42
    assert miniapp_enabled(tmp_path) is True
    observer = TelegramSettingsObserver(tmp_path, 8765)
    assert observer.safe_for_exposure(42) is True
    merge_settings(tmp_path, {"TELEGRAM_MINIAPP_ENABLED": "off"})
    assert observer.safe_for_exposure(42) is False


@pytest.mark.parametrize("raw", ["{", "[]", "null"])
def test_existing_invalid_settings_fail_closed(tmp_path: Path, raw: str) -> None:
    (tmp_path / "settings.json").write_text(raw, encoding="utf-8")

    with pytest.raises(TelegramSettingsError, match="Telegram settings are invalid"):
        load_settings(tmp_path)


def test_existing_unreadable_settings_fail_closed(tmp_path: Path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == settings:
            raise OSError("private filesystem detail")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(TelegramSettingsError) as captured:
        load_settings(tmp_path)

    assert str(captured.value) == "Could not read Telegram settings."
    assert "private filesystem detail" not in str(captured.value)


def test_existing_nonfile_settings_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "settings.json").mkdir()

    with pytest.raises(TelegramSettingsError, match="Telegram settings are invalid"):
        load_settings(tmp_path)


def test_settings_persistence_error_is_bounded(tmp_path: Path, monkeypatch) -> None:
    def fail_replace(_source, _target):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(telegram_settings_module.os, "replace", fail_replace)
    with pytest.raises(TelegramSettingsError) as captured:
        merge_settings(tmp_path, {"TELEGRAM_LANGUAGE": "ru"})
    assert str(captured.value) == "Could not persist Telegram settings."
    assert "private filesystem detail" not in str(captured.value)


def test_only_unmarked_loopback_request_may_reset_owner() -> None:
    assert request_may_change_owner(_request("127.0.0.1")) is True
    assert request_may_change_owner(_request("::1")) is True
    assert request_may_change_owner(_request("127.0.0.1", "1")) is False
    assert request_may_change_owner(_request("203.0.113.7")) is False


@pytest.mark.parametrize(
    ("host", "marker"),
    [
        ("127.0.0.1", "1"),
        ("203.0.113.7", ""),
        (None, ""),
    ],
)
def test_untrusted_settings_route_ignores_owner_but_saves_ordinary_fields(
    tmp_path: Path, host: str | None, marker: str
) -> None:
    plugin = _load_plugin()
    merge_settings(tmp_path, {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_LANGUAGE": "en"})
    request = _RouteRequest(
        {"TELEGRAM_CHAT_ID": "99", "TELEGRAM_LANGUAGE": "ru"},
        host=host,
        marker=marker,
    )
    response = asyncio.run(plugin._make_settings_save(_RouteApi(tmp_path))(request))
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["ok"] is True and body["owner_ignored"] is True
    assert load_settings(tmp_path)["TELEGRAM_CHAT_ID"] == "42"
    assert load_settings(tmp_path)["TELEGRAM_LANGUAGE"] == "ru"


def test_unmarked_loopback_route_may_change_and_reset_owner(tmp_path: Path) -> None:
    plugin = _load_plugin()
    merge_settings(tmp_path, {"TELEGRAM_CHAT_ID": "42"})
    handler = plugin._make_settings_save(_RouteApi(tmp_path))
    changed = asyncio.run(handler(_RouteRequest({"TELEGRAM_CHAT_ID": "99"}, host="127.0.0.1")))
    assert json.loads(changed.body)["owner_ignored"] is False
    assert load_settings(tmp_path)["TELEGRAM_CHAT_ID"] == "99"
    observer = TelegramSettingsObserver(tmp_path, 8765)
    gateway = sidecar_module.TelegramProxySidecar(
        "12345678:" + "A" * 35,
        99,
        8765,
        seams=sidecar_module.SidecarSeams(exposure_guard=lambda: observer.safe_for_exposure(99)),
    )
    gateway.set_public_url("https://owner-reset.trycloudflare.com/")
    token, _session = gateway._issue_session()
    assert gateway._lookup_session(token) is not None
    reset = asyncio.run(handler(_RouteRequest({"TELEGRAM_CHAT_ID": ""}, host="::1")))
    assert json.loads(reset.body)["owner_ignored"] is False
    assert load_settings(tmp_path)["TELEGRAM_CHAT_ID"] == ""
    assert gateway._exposure_is_authorized() is False
    assert gateway.public_url is None
    assert gateway._lookup_session(token) is None


@pytest.mark.parametrize("route_request", [_InvalidJsonRequest({}), _RouteRequest([])])
def test_settings_route_rejects_invalid_payload(tmp_path: Path, route_request) -> None:
    plugin = _load_plugin()
    response = asyncio.run(plugin._make_settings_save(_RouteApi(tmp_path))(route_request))
    assert response.status_code == 400
    assert json.loads(response.body) == {
        "ok": False,
        "message": "Invalid Telegram settings payload.",
    }


def test_settings_route_returns_bounded_conflict_for_busy_store(tmp_path: Path, monkeypatch) -> None:
    plugin = _load_plugin()

    def busy(_state_dir, _payload):
        raise plugin.TelegramSettingsError("Telegram settings are busy.")

    monkeypatch.setattr(plugin, "merge_settings", busy)
    response = asyncio.run(
        plugin._make_settings_save(_RouteApi(tmp_path))(
            _RouteRequest({"TELEGRAM_LANGUAGE": "ru"})
        )
    )
    assert response.status_code == 409
    assert json.loads(response.body) == {
        "ok": False,
        "message": "Telegram settings are busy.",
    }


def test_bridge_status_reports_deferred_validation_as_degraded(tmp_path: Path) -> None:
    """#376: a dead network at startup defers getMe; the status must say so
    instead of `ready`/absent, in the SAME two-section body the UI renders."""
    plugin = _load_plugin()
    merge_settings(tmp_path, {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_COMMAND_MODE": "full_access"})
    (tmp_path / "bridge_status.json").write_text(
        json.dumps({"state": "degraded", "reason_code": "telegram_startup_deferred"}), encoding="utf-8",
    )
    response = asyncio.run(plugin._make_status(_RouteApi(tmp_path))(None))
    body = json.loads(response.body)
    assert set(body) == {"bridge", "mini_app"}
    assert body["bridge"] == {
        "state": "degraded",
        "owner_bound": True,
        "poller": "degraded",
        "command_mode": "full_access",
        "mirror_mode": "all",
        "reason_code": "telegram_startup_deferred",
    }
    # Validation later succeeds → the poller writes `ready` and the overlay lifts.
    plugin._save_bridge_status(_RouteApi(tmp_path), "ready")
    body = json.loads(asyncio.run(plugin._make_status(_RouteApi(tmp_path))(None)).body)
    assert body["bridge"]["state"] == "ready" and "reason_code" not in body["bridge"]


def test_combined_status_has_only_bounded_bridge_and_mini_app_sections(tmp_path: Path) -> None:
    plugin = _load_plugin()
    merge_settings(tmp_path, {
        "TELEGRAM_CHAT_ID": "42",
        "TELEGRAM_COMMAND_MODE": "full_access",
        "TELEGRAM_MIRROR_MODE": "all",
    })
    response = asyncio.run(plugin._make_status(_RouteApi(tmp_path))(None))
    body = json.loads(response.body)
    assert set(body) == {"bridge", "mini_app"}
    assert body["bridge"] == {
        "state": "ready",
        "owner_bound": True,
        "poller": "configured",
        "command_mode": "full_access",
        "mirror_mode": "all",
    }
    assert set(body["mini_app"]) <= {
        "state", "message", "public_url", "cloudflared_version", "instance_id",
        "platform", "reason_code", "updated_at_epoch", "last_ready_at_epoch",
        "attempt", "next_retry_at_epoch", "security",
    }
