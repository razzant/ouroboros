from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
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
    telegram_api_base,
    telegram_files_dir,
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
    def __init__(self, payload, *, host=None, marker="", method="POST") -> None:
        self.payload = payload
        self.method = method
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
    merge_settings(tmp_path, {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_COMMAND_MODE": "full_access"})
    request = _RouteRequest(
        {"TELEGRAM_CHAT_ID": "99", "TELEGRAM_COMMAND_MODE": "strict"},
        host=host,
        marker=marker,
    )
    response = asyncio.run(plugin._make_settings_save(_RouteApi(tmp_path))(request))
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["ok"] is True and body["owner_ignored"] is True
    assert load_settings(tmp_path)["TELEGRAM_CHAT_ID"] == "42"
    assert load_settings(tmp_path)["TELEGRAM_COMMAND_MODE"] == "strict"


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


def test_settings_route_get_hydrates_only_stored_form_keys(tmp_path: Path) -> None:
    """The Settings form is rendered from THIS read, so it must carry exactly
    the form's own stored keys — never the token, never a foreign key, and
    never an absent key (whose runtime default is the schema's first option)."""
    plugin = _load_plugin()
    merge_settings(tmp_path, {
        "TELEGRAM_CHAT_ID": "42",
        "TELEGRAM_COMMAND_MODE": "strict",
        "TELEGRAM_MAX_UPDATES_PER_POLL": 20,
        "TELEGRAM_BOT_TOKEN": "12345678:" + "A" * 35,
        "UNRELATED_KEY": "keep-private",
    })
    handler = plugin._make_settings_save(_RouteApi(tmp_path))
    response = asyncio.run(handler(_RouteRequest(None, method="GET")))

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "TELEGRAM_CHAT_ID": "42",
        "TELEGRAM_MAX_UPDATES_PER_POLL": "20",
        "TELEGRAM_COMMAND_MODE": "strict",
    }
    assert b"TELEGRAM_BOT_TOKEN" not in response.body
    assert b"keep-private" not in response.body
    # A read is a read: the stored document is untouched and the owner binding
    # survives, exactly as before any form is submitted.
    assert load_settings(tmp_path)["TELEGRAM_CHAT_ID"] == "42"
    assert load_settings(tmp_path)["UNRELATED_KEY"] == "keep-private"


def test_settings_route_get_on_empty_store_is_an_empty_document(tmp_path: Path) -> None:
    plugin = _load_plugin()
    response = asyncio.run(
        plugin._make_settings_save(_RouteApi(tmp_path))(_RouteRequest(None, method="GET"))
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {}
    assert not (tmp_path / "settings.json").exists()


def test_settings_route_get_reports_an_unreadable_store(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text("{", encoding="utf-8")
    plugin = _load_plugin()
    response = asyncio.run(
        plugin._make_settings_save(_RouteApi(tmp_path))(_RouteRequest(None, method="GET"))
    )
    assert response.status_code == 409
    assert json.loads(response.body)["ok"] is False


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

    monkeypatch.setattr(sys.modules[plugin._make_settings_save.__module__], "merge_settings", busy)
    response = asyncio.run(
        plugin._make_settings_save(_RouteApi(tmp_path))(
            _RouteRequest({"TELEGRAM_COMMAND_MODE": "strict"})
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


@pytest.mark.parametrize("host,marker", [("127.0.0.1", ""), ("127.0.0.1", "1"), ("203.0.113.7", "")])
def test_proxy_form_is_masked_preserves_empty_and_explicitly_clears(tmp_path, host, marker):
    plugin = _load_plugin()
    handler = plugin._make_settings_save(_RouteApi(tmp_path))
    proxy = "socks5h://owner:proxy-secret@127.0.0.1:1080"
    def request(payload=None, method="POST"):
        return asyncio.run(handler(_RouteRequest(payload, host=host, marker=marker, method=method)))
    assert request({"TELEGRAM_PROXY": proxy}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_PROXY"] == proxy
    hydrated = request(method="GET")
    assert b"TELEGRAM_PROXY" not in hydrated.body and b"proxy-secret" not in hydrated.body
    assert request({"TELEGRAM_PROXY": "", "TELEGRAM_COMMAND_MODE": "strict"}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_PROXY"] == proxy
    assert load_settings(tmp_path)["TELEGRAM_COMMAND_MODE"] == "strict"
    # The language is the install's, not the bridge's: the form neither shows nor stores it.
    assert request({"TELEGRAM_LANGUAGE": "ru"}).status_code == 200
    assert "TELEGRAM_LANGUAGE" not in load_settings(tmp_path)
    refused = request({"TELEGRAM_PROXY": "socks5://owner:proxy-secret@bad-host"})
    assert refused.status_code == 400
    failure = json.loads(refused.body)
    assert failure["error"] == failure["message"]
    assert "TELEGRAM_PROXY must be" in failure["error"]
    assert b"proxy-secret" not in refused.body and b"bad-host" not in refused.body
    assert load_settings(tmp_path)["TELEGRAM_PROXY"] == proxy
    assert request({"clear_telegram_proxy": True, "TELEGRAM_PROXY": ""}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_PROXY"] == ""
    assert "clear_telegram_proxy" not in load_settings(tmp_path)


def test_api_base_validator_canonicalizes_ipv6_with_brackets():
    plugin = _load_plugin()
    settings = sys.modules[plugin._make_settings_save.__module__]
    validate = settings.validate_telegram_api_base
    # IPv6 literals keep their brackets through canonicalization so the
    # origin round-trips settings persistence and client revalidation.
    assert validate("http://[::1]:8081") == "http://[::1]:8081"
    assert validate("https://[2001:db8::1]:8443") == "https://[2001:db8::1]:8443"
    # ...and the canonical form revalidates (no drift on reload).
    assert validate(validate("http://[::1]:8081")) == "http://[::1]:8081"
    # Bracket-less IPv6 with a port is malformed and rejected.
    with pytest.raises(ValueError):
        validate("http://::1:8081")


def test_api_base_form_keeps_what_it_shows_and_an_emptied_field_returns_to_the_cloud(tmp_path):
    plugin = _load_plugin()
    handler = plugin._make_settings_save(_RouteApi(tmp_path))
    base = "http://192.168.1.48:18085"

    def request(payload=None, method="POST"):
        return asyncio.run(handler(_RouteRequest(payload, host="127.0.0.1", method=method)))

    def message(payload):
        reply = request(payload)
        assert reply.status_code == 200
        return json.loads(reply.body)["message"]

    assert f"Bot API server: {base}." in message({"TELEGRAM_API_BASE": base})
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == base
    # The field itself is hydrated with the saved value, so an untouched Save sends it back.
    assert json.loads(request(method="GET").body)["TELEGRAM_API_BASE"] == base
    assert "Bot API server" not in message({"TELEGRAM_API_BASE": base})
    # A request without the key leaves the saved value alone.
    assert "Bot API server" not in message({"TELEGRAM_COMMAND_MODE": "strict"})
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == base
    # A fragment/query/base-with-path is rejected, and the stored value stands.
    for bad in ("https://host#", "https://host?x=1", "https://host/bot", "ftp://host"):
        refused = request({"TELEGRAM_API_BASE": bad})
        assert refused.status_code == 400, bad
        assert "TELEGRAM_API_BASE" in json.loads(refused.body)["message"]
        assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == base
    # Canonicalization: trailing slash and :443-on-https normalize away.
    assert "Bot API server" not in message({"TELEGRAM_API_BASE": base + "/"})
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == base
    assert "Bot API server: https://host.example." in message({"TELEGRAM_API_BASE": "https://host.example:443"})
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == "https://host.example"
    # An emptied field returns to the cloud server, and the form then shows it empty.
    assert "Bot API server: https://api.telegram.org." in message({"TELEGRAM_API_BASE": "  "})
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == ""
    assert json.loads(request(method="GET").body)["TELEGRAM_API_BASE"] == ""


def test_files_dir_form_keeps_a_canonical_path_and_an_emptied_field_clears_it(tmp_path):
    plugin = _load_plugin()
    handler = plugin._make_settings_save(_RouteApi(tmp_path))
    # Native absolute path for THIS platform (a POSIX literal would be
    # rejected at save time on Windows — see the validator tests below).
    store = str(tmp_path / "telegram-bot-api")
    (tmp_path / "telegram-bot-api").mkdir()

    def request(payload=None, method="POST"):
        return asyncio.run(handler(_RouteRequest(payload, host="127.0.0.1", method=method)))

    assert request({"TELEGRAM_API_FILES_DIR": store + os.sep}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_API_FILES_DIR"] == store
    assert json.loads(request(method="GET").body)["TELEGRAM_API_FILES_DIR"] == store
    refused = request({"TELEGRAM_API_FILES_DIR": "relative/dir"})
    assert refused.status_code == 400
    assert "TELEGRAM_API_FILES_DIR" in json.loads(refused.body)["message"]
    assert request({"TELEGRAM_COMMAND_MODE": "strict"}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_API_FILES_DIR"] == store
    assert request({"TELEGRAM_API_FILES_DIR": ""}).status_code == 200
    assert load_settings(tmp_path)["TELEGRAM_API_FILES_DIR"] == ""


def test_registered_form_offers_the_server_fields_as_plain_text(tmp_path):
    plugin = _load_plugin()

    class _RegisterApi(_RouteApi):
        sections: list = []

        def get_runtime_info(self):
            return {"data_dir": str(self.state_dir)}

        def register_settings_section(self, _section_id, **kwargs):
            self.sections.append(kwargs)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    api = _RegisterApi(tmp_path)
    plugin.register_miniapp = lambda _api: None
    plugin.register(api)
    fields = {field["name"]: field for component in api.sections[0]["schema"]["components"]
              if component.get("type") == "form" for field in component["fields"]}
    assert fields["TELEGRAM_API_BASE"]["type"] == fields["TELEGRAM_API_FILES_DIR"]["type"] == "text"
    assert not {"telegram_api_base_status", "clear_telegram_api_base", "clear_telegram_api_files_dir"} & set(fields)
    assert set(telegram_settings_module._SETTINGS_FORM_KEYS) <= set(fields)


def test_files_dir_validator_accepts_native_absolute_paths():
    plugin = _load_plugin()
    settings = sys.modules[plugin._make_settings_save.__module__]
    validate = settings.validate_telegram_files_dir
    # Only spellings NATIVE to the running platform are accepted (round-6 A1):
    # on POSIX a Windows drive/UNC spelling is a relative name to pathlib and
    # would silently resolve against the working directory, so it is refused;
    # on Windows a POSIX spelling normalizes to a backslash root and is
    # refused as well.
    if os.name == "posix":
        assert validate("/store") == "/store"
        assert validate("/store/") == "/store"
        for foreign in ("C:\\store", "C:/store", "\\\\server\\share\\store"):
            with pytest.raises(ValueError):
                validate(foreign)
    else:
        assert validate("C:\\store") == "C:\\store"
        assert validate("C:/store") == "C:\\store"
        assert validate("\\\\server\\share\\store") == "\\\\server\\share\\store"
        for foreign in ("/store", "/store/"):
            with pytest.raises(ValueError):
                validate(foreign)
    assert validate("") is None
    assert validate(None) is None
    for bad in ("/", "C:\\", "//server/share", "store/path", "not a path!"):
        if os.name == "posix" and bad == "//server/share":
            continue  # legal double-slash POSIX path, not a foreign spelling
        with pytest.raises(ValueError):
            validate(bad)
    # Root aliases that only normalize down to a root are refused too
    # ("/tmp/.." -> "/" on POSIX, "C:/store/.." -> "C:\\" on Windows), and an
    # accepted value revalidates to itself (reload idempotency).
    if os.name == "posix":
        for alias in ("/tmp/..", "/tmp/../.."):
            with pytest.raises(ValueError):
                validate(alias)
        accepted = os.path.normpath("/store")
        assert validate(accepted) == accepted
    else:
        for alias in ("C:/store/..", "C:\\store\\..\\.."):
            with pytest.raises(ValueError):
                validate(alias)
        accepted = os.path.normpath("C:\\store")
        assert validate(accepted) == accepted


@pytest.mark.parametrize("host,marker", [("127.0.0.1", "1"), ("::1", "telegram"), ("203.0.113.7", ""), (None, "")])
def test_only_the_local_settings_page_may_change_the_bot_api_server(tmp_path: Path, host, marker) -> None:
    # The saved server supplies the updates and the sender IDs the owner check trusts.
    plugin = _load_plugin()
    store = str(tmp_path / "store")
    merge_settings(tmp_path, {"TELEGRAM_API_BASE": "http://127.0.0.1:8081", "TELEGRAM_API_FILES_DIR": store})
    handler = plugin._make_settings_save(_RouteApi(tmp_path))

    def save(payload, **request):
        reply = asyncio.run(handler(_RouteRequest(payload, **request)))
        assert reply.status_code == 200
        return json.loads(reply.body)

    refused = save({"TELEGRAM_API_BASE": "http://198.51.100.9:8081", "TELEGRAM_API_FILES_DIR": "",
                    "TELEGRAM_COMMAND_MODE": "strict"}, host=host, marker=marker)
    assert refused["ok"] is True and refused["server_ignored"] is True
    assert "change them from the Settings page on the computer running Ouroboros" in refused["message"]
    saved = load_settings(tmp_path)
    assert (saved["TELEGRAM_API_BASE"], saved["TELEGRAM_API_FILES_DIR"]) == ("http://127.0.0.1:8081", store)
    assert saved["TELEGRAM_COMMAND_MODE"] == "strict"
    # The remote form sends back what it was shown: unchanged values pass without a notice.
    quiet = save({"TELEGRAM_API_BASE": "http://127.0.0.1:8081/", "TELEGRAM_API_FILES_DIR": store},
                 host=host, marker=marker)
    assert quiet["server_ignored"] is False and "Settings page" not in quiet["message"]
    local = save({"TELEGRAM_API_BASE": "http://198.51.100.9:8081"}, host="127.0.0.1")
    assert local["server_ignored"] is False
    assert load_settings(tmp_path)["TELEGRAM_API_BASE"] == "http://198.51.100.9:8081"


def test_server_notes_agree_on_timing_and_the_move_with_ouroboros_quit(tmp_path):
    # SKILL.md, the form help and the Save message describe one transition. A move is made with
    # Ouroboros quit: any request after logOut would launch the bot on that server again.
    plugin = _load_plugin()
    skill = (SCRIPTS.parent / "SKILL.md").read_text(encoding="utf-8")
    book = (SCRIPTS.parents[2] / "docs" / "architecture" / "13-external-skills-layer.md").read_text(encoding="utf-8")

    class _RegisterApi(_RouteApi):
        sections: list = []

        def get_runtime_info(self):
            return {"data_dir": str(self.state_dir)}

        def register_settings_section(self, _section_id, **kwargs):
            self.sections.append(kwargs)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    api = _RegisterApi(tmp_path)
    plugin.register_miniapp = lambda _api: None
    plugin.register(api)
    help_text = next(field["help"] for component in api.sections[0]["schema"]["components"]
                     if component.get("type") == "form" for field in component["fields"]
                     if field["name"] == "TELEGRAM_API_BASE")
    reply = asyncio.run(plugin._make_settings_save(_RouteApi(tmp_path))(
        _RouteRequest({"TELEGRAM_API_BASE": "http://127.0.0.1:8081"}, host="127.0.0.1")))
    message = json.loads(reply.body)["message"]
    for surface in (" ".join(skill.split()), help_text, message):
        assert "retry wait (up to about a minute)" in surface
        assert "25 seconds" not in surface
    assert "quit first" in help_text and "needs logOut" not in help_text
    assert "another address of the same server" in message and "Ouroboros quit" in message
    assert "logOut or close first" not in message
    runbook = " ".join(skill.split())
    steps = ["Copy `<data>/state/skills/telegram/settings.json` to a backup", "Quit Ouroboros so that none",
             "Stop every other client of this token", "Call one method on the old server",
             "With Ouroboros still quit, edit only these two keys", "Start Ouroboros."]
    positions = [runbook.index(step) for step in steps]
    assert positions == sorted(positions)
    for retired in ("Keep this skill enabled throughout", "save the new base immediately", "empty the field immediately"):
        assert retired not in runbook
    for direction in ("`logOut` on `https://api.telegram.org`", "`close` on the old one",
                      "`logOut` (or `close`) on your server", "treat them as lost"):
        assert direction in runbook
    assert "`NO_PROXY`" in runbook
    # The documented edit is what the skill reads, keeps the other keys, and empty means the cloud.
    example = re.search(r"```json\n(.*?)```", skill, re.S).group(1)
    edited = tmp_path / "edited"
    merge_settings(edited, {"TELEGRAM_CHAT_ID": "42"})
    merge_settings(edited, json.loads("{" + example + "}"))
    assert telegram_api_base(edited) == "http://127.0.0.1:8081"
    assert telegram_files_dir(edited) == "/absolute/path/to/server-file-store"
    assert load_settings(edited)["TELEGRAM_CHAT_ID"] == "42"
    merge_settings(edited, {"TELEGRAM_API_BASE": "", "TELEGRAM_API_FILES_DIR": ""})
    assert telegram_api_base(edited) == "https://api.telegram.org" and telegram_files_dir(edited) is None
    row = next(line for line in book.splitlines() if line.startswith("`skills/telegram/`"))
    assert "`TELEGRAM_API_BASE`" in row and "`TELEGRAM_API_FILES_DIR`" in row and "server forms" in row
