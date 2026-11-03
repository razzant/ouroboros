"""Real HTTPX+socksio traffic; every socket is confined to a loopback TLS bot.

Only the fake Telegram origin destination and its test CA are substituted. HTTPX,
CONNECT, SOCKS authentication, TLS, serialization and skill consumers are real.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import ipaddress
import json
import logging
import socket
import ssl
import sys
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from tests.test_telegram_proxy import _Api, _TOKEN, _bridge_events, _load

pytestmark = pytest.mark.serial
SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import companion  # noqa: E402
import telegram_menu  # noqa: E402


def _tls_contexts(root):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "api.telegram.org")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=True,
                                        data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                        crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("api.telegram.org")]), False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = root / "cert.pem", root / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    client = ssl.create_default_context(cafile=str(cert_path))
    return server, client


class _LoopbackBot:
    def __init__(self):
        self.calls = []
        self.tunnels = []
        self.tasks = set()
        self.errors = []
        self.polled = asyncio.Event()
        self.button = {"type": "commands"}
        self.origin_port = 0

    def accept(self, handler):
        def spawn(reader, writer):
            task = asyncio.create_task(handler(reader, writer))
            self.tasks.add(task)
            def finished(done):
                self.tasks.discard(done)
                if not done.cancelled() and done.exception():
                    self.errors.append(done.exception())
            task.add_done_callback(finished)
        return spawn

    async def origin(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            first, *lines = head.decode().split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines if ": " in line)
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            method, path, _version = first.split(" ")
            self.calls.append((method, path, headers, body))
            if path.startswith(f"/file/bot{_TOKEN}/"):
                payload, content_type = b"loopback file bytes", "application/octet-stream"
            else:
                action = path.rsplit("/", 1)[-1]
                result = {"message_id": 555}
                if action == "getMe":
                    result = {"id": 1, "is_bot": True, "username": "loopback_bot"}
                elif action == "getChat":
                    result = {"id": 42, "type": "private"}
                elif action == "getUpdates":
                    result = []
                    self.polled.set()
                elif action == "getFile":
                    result = {"file_path": "documents/file.txt"}
                elif action == "getChatMenuButton":
                    result = self.button
                elif action == "setChatMenuButton":
                    self.button = json.loads(body)["menu_button"]
                    result = True
                payload, content_type = json.dumps({"ok": True, "result": result}).encode(), "application/json"
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def proxy(self, reader, writer, scheme):
        peer = None
        try:
            if scheme == "http":
                head = await reader.readuntil(b"\r\n\r\n")
                auth = base64.b64encode(b"owner:proxy-secret")
                assert head.startswith(b"CONNECT api.telegram.org:443 HTTP/1.1\r\n")
                assert b"Proxy-Authorization: Basic " + auth in head
                assert _TOKEN.encode() not in head
                self.tunnels.append((scheme, "api.telegram.org", 443, "owner"))
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                version, count = await reader.readexactly(2)
                assert version == 5 and 2 in await reader.readexactly(count)
                writer.write(b"\x05\x02")
                await writer.drain()
                version, length = await reader.readexactly(2)
                user = await reader.readexactly(length)
                password = await reader.readexactly((await reader.readexactly(1))[0])
                assert (version, user, password) == (1, b"owner", b"proxy-secret")
                writer.write(b"\x01\x00")
                await writer.drain()
                assert await reader.readexactly(4) == b"\x05\x01\x00\x03"
                host = await reader.readexactly((await reader.readexactly(1))[0])
                port = int.from_bytes(await reader.readexactly(2), "big")
                assert (host, port) == (b"api.telegram.org", 443)
                self.tunnels.append((scheme, host.decode(), port, user.decode()))
                writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
            await writer.drain()
            upstream, peer = await asyncio.open_connection("127.0.0.1", self.origin_port)
            async def pipe(source, target):
                try:
                    while chunk := await source.read(65536):
                        target.write(chunk)
                        await target.drain()
                except (ConnectionResetError, BrokenPipeError):
                    pass  # HTTPX may close after the complete HTTP response.
                finally:
                    target.close()
            await asyncio.gather(pipe(reader, peer), pipe(upstream, writer))
        finally:
            for stream in (writer, peer):
                if stream is not None:
                    stream.close()
                    with contextlib.suppress(ConnectionError):
                        await stream.wait_closed()


@pytest.mark.parametrize("scheme", ["direct", "http", "socks5", "socks5h"])
def test_all_telegram_consumers_succeed_through_real_transport(tmp_path, monkeypatch, caplog, scheme):
    plugin, _telegram_api, notifier = _load()
    caplog.set_level(logging.DEBUG, logger="httpcore")
    server_tls, client_tls = _tls_contexts(tmp_path)
    real_client, real_dns = httpx.AsyncClient, socket.getaddrinfo
    options = []
    bot = _LoopbackBot()
    def dns(host, port, *args, **kwargs):
        if host in ("api.telegram.org", b"api.telegram.org"):
            # No external DNS, including the direct control case.
            host, port = "127.0.0.1", bot.origin_port
        assert host in ("127.0.0.1", b"127.0.0.1", "::1", b"::1", None)
        return real_dns(host, port, *args, **kwargs)
    def client_factory(**kwargs):
        options.append(dict(kwargs))
        return real_client(verify=client_tls, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    # An explicit skill proxy must beat ambient routing; unset stays direct here.
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "*")
    state = tmp_path / "state"
    state.mkdir()
    api = _Api(state, {"TELEGRAM_BOT_TOKEN": _TOKEN, "TELEGRAM_PROXY": "http://ignored.invalid"})

    async def scenario():
        loop = asyncio.get_running_loop()
        real_connect = loop.sock_connect
        async def connect(sock, address):
            if isinstance(address, tuple):
                assert ipaddress.ip_address(address[0]).is_loopback, address
                if address[1] == 443:
                    # AnyIO retains the URL port; Proactor bypasses socket.connect.
                    address = (address[0], bot.origin_port, *address[2:])
            return await real_connect(sock, address)
        monkeypatch.setattr(loop, "sock_connect", connect)
        origin = await asyncio.start_server(bot.accept(bot.origin), "127.0.0.1", 0, ssl=server_tls)
        bot.origin_port = origin.sockets[0].getsockname()[1]
        loop = asyncio.get_running_loop()
        real_create_connection = loop.create_connection
        async def create_connection(protocol_factory, host=None, port=None, **kwargs):
            if host is not None:
                assert ipaddress.ip_address(host).is_loopback, host
                if port == 443:
                    # AnyIO retains 443 after DNS; Windows Proactor bypasses socket.connect.
                    port = bot.origin_port
            return await real_create_connection(protocol_factory, host, port, **kwargs)
        monkeypatch.setattr(loop, "create_connection", create_connection)
        proxy = await asyncio.start_server(bot.accept(lambda r, w: bot.proxy(r, w, scheme)), "127.0.0.1", 0)
        proxy_url = None if scheme == "direct" else f"{scheme}://owner:proxy-secret@127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
        (state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_PROXY": proxy_url}), encoding="utf-8")
        try:
            poller = asyncio.create_task(plugin._poller(api))
            try:
                await asyncio.wait_for(bot.polled.wait(), 5)
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            for factory, event in _bridge_events():
                await getattr(plugin, factory)(api)(event)
            await plugin._typing_cue(api)._send(42)  # the typing refresher's own sends
            assert await notifier._push_notification(api, 42, "done", trust_env=False) == ("sent", None)
            client = plugin._telegram_client(api)
            await client.send_audio(42, b"audio", "sample.mp3")
            await client.answer_callback_query("cb-1")
            assert await client.download_file("file-id") == b"loopback file bytes"
            encoded, _mime = await client.download_photo("photo-id")
            assert base64.b64decode(encoded) == b"loopback file bytes"
            # Standalone manager owns an HTTPX client; run() owns and injects another.
            manager = telegram_menu.TelegramMenuManager(_TOKEN, 42, state)
            assert await manager.verify_private_owner() == "@loopback_bot"
            assert await manager.install("https://loopback.trycloudflare.com/")
            assert await manager.restore()
            (state / "runtime_config.json").write_text(json.dumps({
                "schema": 2, "core_port": 8765, "owner_chat_id": 42,
                "button_text": "Ouroboros", "tunnel": "cloudflare_quick",
            }), encoding="utf-8")
            original_wait = companion._wait_for_owner
            reached = []
            async def verified_owner(*args):
                owner, menu = await original_wait(*args)
                reached.append(owner)
                assert await menu.install("https://loopback.trycloudflare.com/")
                assert await menu.restore()
                # Stop before the separate Cloudflare/public lifecycle starts.
                raise companion.ShutdownRequested("Loopback Telegram qualification complete.")
            monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
            monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _TOKEN)
            monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
            monkeypatch.setattr(companion, "start_parent_lifeline", lambda *_args: (None, None))
            monkeypatch.setattr(companion, "_wait_for_owner", verified_owner)
            assert await companion.run() == 0
            assert reached == [42]
            assert bot.button == {"type": "commands"}
            assert not (state / "menu_button_snapshot.json").exists()
            assert options and all(option["proxy"] == proxy_url for option in options)
            assert all(option["follow_redirects"] is False for option in options)
            assert all(option["trust_env"] is False for option in options)
            # A real form clear restores both independently-owned transports.
            from tests.test_telegram_settings import _RouteRequest
            reply = await plugin._make_settings_save(api)(_RouteRequest({"clear_telegram_proxy": True}))
            assert reply.status_code == 200
            tunnels_before = len(bot.tunnels)
            await plugin._telegram_client(api).call("getMe")
            assert await telegram_menu.TelegramMenuManager(_TOKEN, 42, state).current() == {"type": "commands"}
            assert len(bot.tunnels) == tunnels_before
            assert options[-1]["proxy"] is None and options[-2]["proxy"] is None
        finally:
            proxy.close()
            origin.close()
            await proxy.wait_closed()
            await origin.wait_closed()
            if bot.tasks:
                await asyncio.wait_for(asyncio.gather(*bot.tasks), 5)
        assert not bot.errors

    asyncio.run(scenario())
    actions = {path.rsplit("/", 1)[-1] for _method, path, _headers, _body in bot.calls}
    assert {"getMe", "setMyCommands", "getUpdates", "sendMessage", "sendChatAction", "sendPhoto",
            "sendVideo", "sendDocument", "sendAudio", "editMessageText", "answerCallbackQuery",
            "getFile", "file.txt", "getChat", "getChatMenuButton", "setChatMenuButton"} <= actions
    assert all("proxy-authorization" not in headers for _, _, headers, _ in bot.calls)
    assert bool(bot.tunnels) == (scheme != "direct")
    assert all("proxy-secret" not in message for _level, message in api.logs)
    assert not [row for row in api.logs if row[0] in ("error", "warning")]
    assert "proxy-secret" not in caplog.text


def test_local_host_and_miniapp_clients_ignore_telegram_proxy(tmp_path, monkeypatch):
    plugin, _telegram_api, _notifier = _load()
    import sidecar
    import platform_support
    (tmp_path / "settings.json").write_text(json.dumps({
        "TELEGRAM_CHAT_ID": "42", "TELEGRAM_PROXY": "socks5://owner:proxy-secret@127.0.0.1:1",
    }), encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_PROXY", "socks5://owner:proxy-secret@127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    seen, built = [], []
    real_client = httpx.AsyncClient
    def factory(**kwargs):
        built.append(kwargs)
        return real_client(**kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    async def serve(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            seen.append(head)
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{"status":"ok"}')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    async def scenario():
        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", str(port))
        api = _Api(tmp_path, {"TELEGRAM_BOT_TOKEN": _TOKEN})
        monkeypatch.setattr(api, "get_runtime_info", lambda: {"server_port": port})
        gateway = sidecar.TelegramProxySidecar(_TOKEN, 42, port)
        try:
            assert await plugin._host_post(api, "/chat/inject", {}) == (200, {"status": "ok"})
            assert await plugin._load_runtime_state(api) == {"status": "ok"}
            assert await companion.probe_core(port)
            client = await gateway._get_http_client()
            assert (await client.get(f"http://127.0.0.1:{port}/")).json() == {"status": "ok"}
        finally:
            await gateway.aclose()
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())
    assert len(seen) == len(built) == 4
    assert all("proxy" not in options and options["trust_env"] is False for options in built)
    assert all(b"proxy-secret" not in head and b"Proxy-Authorization" not in head for head in seen)
    assert not any("PROXY" in key for key in platform_support.minimal_process_environment(tmp_path))
