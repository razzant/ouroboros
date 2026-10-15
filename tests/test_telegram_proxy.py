"""TELEGRAM_PROXY: the owner's Telegram-only egress for the in-process bridge (#1149).

Hermetic: a loopback fake proxy stands in for the owner's proxy and refuses every tunnel,
so no request leaves the machine; no Telegram, Cloudflare, installed skill or settings
file outside ``tmp_path`` is touched.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import shutil
import sys
import types
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "skills" / "telegram"
_PACKAGE = "telegram_proxy_test"
_TOKEN = "123456:SECRET-BOT-TOKEN"
_PROXY = "socks5://owner:proxy-secret@127.0.0.1:1080"


def _load():
    for name in [key for key in sys.modules if key == _PACKAGE or key.startswith(f"{_PACKAGE}.")]:
        sys.modules.pop(name, None)
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_ROOT)]
    sys.modules[_PACKAGE] = package
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.plugin", _ROOT / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    return plugin, sys.modules[f"{_PACKAGE}.lib.telegram_api"], sys.modules[f"{_PACKAGE}.lib.telegram_notifier"]


class _Api:
    """PluginAPI stand-in: ``get_settings`` answers only what was granted and asked for."""

    def __init__(self, state_dir, granted):
        self.state_dir = Path(state_dir)
        self.granted = dict(granted)
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {key: self.granted[key] for key in keys if key in self.granted}

    def get_runtime_info(self):
        return {"data_dir": str(self.state_dir)}

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: "skill-token")

    def log(self, level, message, **_fields):
        self.logs.append((level, message))


@pytest.mark.parametrize("value", [
    "socks5://owner:proxy-secret@proxy.example:1080",
    "socks5h://proxy.example:1080",
    "http://proxy.example",
    "https://owner:proxy-secret@proxy.example:8443",
    "  http://proxy.example:3128\n",
])
def test_supported_proxy_urls_are_kept_verbatim(value):
    _plugin, telegram_api, _nt = _load()
    assert telegram_api.TelegramClient(_TOKEN, proxy=value).proxy == value.strip()


@pytest.mark.parametrize("value", [
    "proxy.example:3128",
    "ftp://owner:proxy-secret@proxy.example:21",
    "socks4://owner:proxy-secret@proxy.example:1080",
    "socks5://owner:proxy-secret@proxy.example",
    "http://owner:proxy-secret@:3128",
    "http://owner:proxy-secret@proxy.example:99999",
    "http://owner:proxy-secret@proxy.example/path",
    "http://owner:proxy-secret@proxy.example?route=bad",
    "socks5://owner:proxy-secret@proxy.example:0",
])
def test_malformed_proxy_is_refused_without_echoing_it(value):
    _plugin, telegram_api, _nt = _load()
    with pytest.raises(ValueError) as caught:
        telegram_api.TelegramClient(_TOKEN, proxy=value)
    assert "TELEGRAM_PROXY" in str(caught.value)
    assert "proxy-secret" not in str(caught.value) and "proxy.example" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize("proxy", [None, "", "   "])
def test_without_a_proxy_both_transports_stay_direct_and_never_follow_redirects(monkeypatch, proxy):
    _plugin, telegram_api, _nt = _load()
    real_async_client = httpx.AsyncClient
    seen, urls = [], []

    def handler(request):
        urls.append(str(request.url))
        if request.url.host != "api.telegram.org":
            return httpx.Response(200, json={"ok": True, "result": {}})
        return httpx.Response(302, headers={"Location": "https://attacker.invalid/steal"})

    def client_factory(**kwargs):
        seen.append(dict(kwargs))
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient", client_factory)
    client = telegram_api.TelegramClient(_TOKEN, proxy=proxy)
    assert client.proxy is None
    with pytest.raises(telegram_api.TelegramTransportError, match="invalid JSON during getMe"):
        asyncio.run(client.call("getMe"))
    assert asyncio.run(client._download_bytes("documents/file_1.pdf")) == b""

    assert [(kw["proxy"], kw["follow_redirects"], kw["trust_env"]) for kw in seen] == [(None, False, False)] * 2
    # Each 302 stayed an answer from Telegram; the redirect target was never requested.
    assert len(urls) == 2 and all(url.startswith("https://api.telegram.org/") for url in urls)


def _serve(handler):
    return asyncio.start_server(handler, "127.0.0.1", 0)


@pytest.mark.serial
def test_http_proxy_tunnels_to_telegram_and_never_sees_the_token(monkeypatch):
    """The explicit proxy wins over an ambient one; CONNECT names only the host, the
    credentials ride Proxy-Authorization, and refusals echo neither secret."""
    _plugin, telegram_api, _nt = _load()
    heads: list[str] = []

    async def refuse_tunnel(reader, writer):
        heads.append((await reader.readuntil(b"\r\n\r\n")).decode("latin-1"))
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    async def scenario():
        server = await _serve(refuse_tunnel)
        port = server.sockets[0].getsockname()[1]
        client = telegram_api.TelegramClient(
            _TOKEN, trust_env=True, proxy=f"http://owner:proxy-secret@127.0.0.1:{port}",
        )
        errors = []
        for attempt in (client.call("getMe"), client._download_bytes("documents/file_1.pdf")):
            try:
                await attempt
            except (telegram_api.TelegramTransportError, RuntimeError) as exc:
                errors.append(exc)
        server.close()
        await server.wait_closed()
        return errors

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    errors = asyncio.run(scenario())

    assert [str(exc) for exc in errors] == [
        "Telegram API transport failed during getMe (ProxyError).",
        "Telegram file download transport failed (ProxyError).",
    ]
    assert all(exc.__cause__ is None for exc in errors)
    credentials = base64.b64encode(b"owner:proxy-secret").decode("ascii")
    assert len(heads) == 2
    for head in heads:
        assert head.startswith("CONNECT api.telegram.org:443 HTTP/1.1\r\n")
        assert f"proxy-authorization: basic {credentials}".lower() in head.lower()
        assert "SECRET-BOT-TOKEN" not in head and "/bot" not in head


@pytest.mark.serial
@pytest.mark.parametrize("scheme", ["socks5", "socks5h"])
def test_socks_proxy_receives_the_telegram_host_name_and_credentials(scheme):
    _plugin, telegram_api, _nt = _load()
    seen: list[tuple] = []

    async def refuse_connect(reader, writer):
        _version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        writer.write(b"\x05\x02")
        await writer.drain()
        _auth_version, user_length = await reader.readexactly(2)
        user = await reader.readexactly(user_length)
        password = await reader.readexactly((await reader.readexactly(1))[0])
        writer.write(b"\x01\x00")
        await writer.drain()
        _version, command, _reserved, address_type = await reader.readexactly(4)
        host = await reader.readexactly((await reader.readexactly(1))[0]) if address_type == 3 else b""
        port = int.from_bytes(await reader.readexactly(2), "big")
        seen.append((methods, user, password, command, address_type, host, port))
        writer.write(b"\x05\x05\x00\x01" + bytes(6))
        await writer.drain()
        writer.close()

    async def scenario():
        server = await _serve(refuse_connect)
        port = server.sockets[0].getsockname()[1]
        client = telegram_api.TelegramClient(_TOKEN, proxy=f"{scheme}://owner:proxy-secret@127.0.0.1:{port}")
        try:
            with pytest.raises(telegram_api.TelegramTransportError) as caught:
                await client.call("sendMessage", data={"chat_id": "42", "text": "hi"})
        finally:
            server.close()
            await server.wait_closed()
        return caught.value

    error = asyncio.run(scenario())
    assert str(error) == "Telegram API transport failed during sendMessage (ProxyError)."
    # Username/password auth, CONNECT (1) to a domain name (3): the proxy resolves Telegram.
    assert seen == [(b"\x02", b"owner", b"proxy-secret", 1, 3, b"api.telegram.org", 443)]


class _RecordingClient:
    built: list = []

    def __init__(self, token, **kwargs):
        _RecordingClient.built.append((token, kwargs))

    async def call(self, _method, **_kwargs):
        return {"ok": True, "result": {}}

    def __getattr__(self, _name):
        async def sent(*_args, **_kwargs):
            return 555
        return sent


def _bridge_events():
    quiz = {"quiz_id": "q1", "task_id": "task-1", "question": "Which db?", "transport": {},
            "options": [{"label": "sqlite"}, {"label": "postgres"}]}
    return [
        ("_make_outbound", {"text": "hello", "transport": {}}),
        ("_make_typing", {"transport": {}}),
        ("_make_photo", {"image_base64": base64.b64encode(b"png").decode(), "transport": {}}),
        ("_make_video", {"video_base64": base64.b64encode(b"mp4").decode(), "transport": {}}),
        ("_make_document", {"file_base64": base64.b64encode(b"doc").decode(), "filename": "a.txt"}),
        ("_make_links", {"title": "Links", "actions": [{"label": "Docs", "url": "https://example.org"}]}),
        ("_make_quiz", quiz),
        ("_make_quiz_state", {"quiz_id": "q1", "task_id": "task-1", "state": "answered", "answered_index": 1}),
    ]


@pytest.mark.parametrize("local_proxy", [None, _PROXY])
def test_every_bridge_client_carries_exactly_the_local_proxy(tmp_path, monkeypatch, local_proxy):
    """Poller, each chat event handler, the quiz lifecycle and the notifier: one client
    factory per path, each built with the local proxy, independently of global Secrets."""
    plugin, _telegram_api, notifier = _load()
    _RecordingClient.built = []
    monkeypatch.setattr(plugin, "TelegramClient", _RecordingClient)
    monkeypatch.setattr(notifier, "TelegramClient", _RecordingClient)
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_PROXY": local_proxy}), encoding="utf-8")
    granted = {"TELEGRAM_BOT_TOKEN": _TOKEN, "TELEGRAM_PROXY": "http://ignored.invalid"}
    api = _Api(tmp_path, granted)

    asyncio.run(plugin._start_poller(api))
    for factory, event in _bridge_events():
        asyncio.run(getattr(plugin, factory)(api)(event))
    assert asyncio.run(notifier._push_notification(api, 42, "done", trust_env=False)) == ("sent", None)

    assert [(level, message) for level, message in api.logs if level != "info"] == []
    assert _RecordingClient.built == [(_TOKEN, {"trust_env": False, "proxy": local_proxy,
                                                 "api_base_url": "https://api.telegram.org",
                                                 "files_dir": None})] * 10


def test_a_malformed_proxy_stops_the_bridge_without_leaking_it(tmp_path):
    plugin, _telegram_api, _nt = _load()
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_PROXY": "socks5://owner:proxy-secret@proxy"}), encoding="utf-8")
    api = _Api(tmp_path, {"TELEGRAM_BOT_TOKEN": _TOKEN})

    with pytest.raises(ValueError, match="TELEGRAM_PROXY"):
        asyncio.run(plugin._start_poller(api))
    asyncio.run(plugin._make_outbound(api)({"text": "hello", "transport": {}}))

    status = json.loads((tmp_path / "bridge_status.json").read_text(encoding="utf-8"))
    assert status == {"state": "error", "reason_code": "telegram_startup_failed"}
    errors = [message for level, message in api.logs if level == "error"]
    assert len(errors) == 2 and all("TELEGRAM_PROXY must be" in message for message in errors)
    assert not any("proxy-secret" in message or "SECRET-BOT-TOKEN" in message for message in errors)


def test_global_proxy_secret_is_not_requested_or_exposed(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros import extension_loader
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_loader import requested_core_setting_keys

    manifest = parse_skill_manifest_text((_ROOT / "SKILL.md").read_text(encoding="utf-8"))
    settings = {"TELEGRAM_BOT_TOKEN": _TOKEN, "TELEGRAM_PROXY": _PROXY}
    monkeypatch.setattr(config, "load_settings", lambda: dict(settings))
    assert requested_core_setting_keys(manifest.env_from_settings) == ["TELEGRAM_BOT_TOKEN"]
    api = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="telegram", permissions=list(manifest.permissions),
        env_allowlist=list(manifest.env_from_settings), state_dir=tmp_path,
        settings_reader=lambda: dict(settings), granted_keys=["TELEGRAM_BOT_TOKEN"],
    ))
    assert api.get_settings(["TELEGRAM_BOT_TOKEN", "TELEGRAM_PROXY"]) == {"TELEGRAM_BOT_TOKEN": _TOKEN}


@pytest.mark.parametrize("owner_set_proxy", [False, True])
def test_upgrade_keeps_token_grant_and_local_proxy(
    tmp_path, monkeypatch, owner_set_proxy,
):
    """The proxy is skill-local state: reseeding changes payload, not credentials or grants."""
    import ouroboros.config as config
    from ouroboros.launcher_bootstrap import _per_skill_version_resync
    from ouroboros.skill_loader import (
        load_skill,
        load_skill_grants,
        requested_skill_permissions,
        save_skill_grants,
    )

    drive = tmp_path / "data"
    native = drive / "skills" / "native"
    installed = native / "telegram"
    shutil.copytree(_ROOT, installed)
    manifest = (installed / "SKILL.md").read_text(encoding="utf-8")
    manifest = manifest.replace("version: 1.2.14", "version: 1.2.11")
    (installed / "SKILL.md").write_text(manifest, encoding="utf-8")
    (installed / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    settings = {"TELEGRAM_BOT_TOKEN": _TOKEN, **({"TELEGRAM_PROXY": _PROXY} if owner_set_proxy else {})}
    monkeypatch.setattr(config, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    local_state = drive / "state" / "skills" / "telegram"
    local_state.mkdir(parents=True)
    local_settings = local_state / "settings.json"
    local_settings.write_text(json.dumps({"TELEGRAM_PROXY": _PROXY if owner_set_proxy else ""}), encoding="utf-8")
    settings_before = local_settings.read_bytes()
    old = load_skill(installed, drive)
    assert old is not None and not old.load_error
    permissions = requested_skill_permissions(list(old.manifest.permissions), list(old.manifest.subscribe_events))
    save_skill_grants(drive, "telegram", ["TELEGRAM_BOT_TOKEN"], content_hash=old.content_hash,
                      requested_keys=["TELEGRAM_BOT_TOKEN"], granted_permissions=permissions,
                      requested_permissions=permissions)

    log = logging.getLogger("tests.telegram_proxy")
    assert _per_skill_version_resync(_ROOT.parent, native, log, drive_root=drive) == 1
    new = load_skill(installed, drive)
    grants = load_skill_grants(drive, "telegram")
    assert new.manifest.version == "1.2.14" and new.content_hash != old.content_hash
    assert grants["content_hash"] == new.content_hash
    assert local_settings.read_bytes() == settings_before
    assert grants["granted_keys"] == ["TELEGRAM_BOT_TOKEN"]
