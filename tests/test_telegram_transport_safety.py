import asyncio
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path

import httpx
import pytest


_PACKAGE = "telegram_transport_safety_test"


def _load_skill():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    for name in [key for key in sys.modules if key == _PACKAGE or key.startswith(f"{_PACKAGE}.")]:
        sys.modules.pop(name, None)
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(root)]
    sys.modules[_PACKAGE] = package
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.plugin", root / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    return plugin, sys.modules[f"{_PACKAGE}.lib.telegram_api"]


class _Token:
    def __init__(self):
        self.reveals = 0

    def use_in_request(self):
        self.reveals += 1
        return "skill-token"


class _Api:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.token = _Token()

    def get_state_dir(self):
        return str(self.state_dir)

    def get_skill_token(self):
        return self.token

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def log(self, _level, _message, **_fields):
        pass


@pytest.mark.parametrize("raw_port", ["8767@evil.example", "not-a-port", "0", "65536"])
def test_host_inject_rejects_invalid_port_before_revealing_token(tmp_path, monkeypatch, raw_port):
    plugin, _telegram_api = _load_skill()
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = _Api(tmp_path)
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", raw_port)

    def forbidden_client(**_kwargs):
        raise AssertionError("network client must not be constructed for an invalid port")

    monkeypatch.setattr(plugin.httpx, "AsyncClient", forbidden_client)
    with pytest.raises(RuntimeError, match="Host Service port is invalid"):
        asyncio.run(plugin._inject(api, {"chat_id": 42, "text": "hello"}))
    assert api.token.reveals == 0


def test_host_inject_uses_exact_loopback_url_for_valid_port(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = _Api(tmp_path)
    observed = {}
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", "9123")
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid:8080")

    class Client:
        def __init__(self, **kwargs):
            observed["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, *, headers, json):
            observed.update(url=url, headers=headers, json=json)
            return types.SimpleNamespace(status_code=202)

    monkeypatch.setattr(plugin.httpx, "AsyncClient", Client)
    asyncio.run(plugin._inject(api, {"chat_id": 42, "text": "hello"}))

    assert observed["url"] == "http://127.0.0.1:9123/chat/inject"
    assert observed["headers"] == {"X-Skill-Token": "skill-token"}
    assert observed["client_kwargs"] == {
        "timeout": 60,
        "trust_env": False,
        "follow_redirects": False,
    }
    assert api.token.reveals == 1


def _install_mock_transport(monkeypatch, telegram_api, handler):
    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient", client_factory)


def test_client_ignores_ambient_telegram_endpoint_overrides(monkeypatch):
    _plugin, telegram_api = _load_skill()
    monkeypatch.setenv("TELEGRAM_API_BASE", "http://attacker.invalid/api")
    monkeypatch.setenv("TELEGRAM_FILE_BASE", "http://attacker.invalid/file")

    client = telegram_api.TelegramClient("test-token")

    assert client.api_base == "https://api.telegram.org/bottest-token"
    assert client.file_base == "https://api.telegram.org/file/bottest-token"


def test_telegram_call_hides_token_bearing_transport_error(monkeypatch):
    _plugin, telegram_api = _load_skill()
    token = "12345678:" + "A" * 35

    def handler(request):
        raise httpx.ConnectError(f"failed to reach {request.url}", request=request)

    _install_mock_transport(monkeypatch, telegram_api, handler)
    with pytest.raises(telegram_api.TelegramTransportError) as captured:
        asyncio.run(telegram_api.TelegramClient(token).call("getMe"))

    message = str(captured.value)
    assert token not in message
    assert "api.telegram.org" not in message
    assert message == "Telegram API transport failed during getMe (ConnectError)."


def test_telegram_call_hides_token_bearing_description(monkeypatch):
    _plugin, telegram_api = _load_skill()
    token = "12345678:" + "B" * 35

    def handler(request):
        return httpx.Response(
            400,
            json={"ok": False, "description": f"Bad Request: reflected {request.url}"},
        )

    _install_mock_transport(monkeypatch, telegram_api, handler)
    with pytest.raises(RuntimeError) as captured:
        asyncio.run(telegram_api.TelegramClient(token).call("sendMessage"))

    message = str(captured.value)
    assert isinstance(captured.value, telegram_api.TelegramRequestRejected)
    assert token not in message
    assert "api.telegram.org" not in message
    assert message == "Telegram API sendMessage returned HTTP 400."


def test_send_message_retries_only_rejected_current_html_chunk_as_plain_text():
    _plugin, telegram_api = _load_skill()
    client = telegram_api.TelegramClient("token")
    first_chunk = "a" * 4085
    second_chunk = "**later**"
    calls = []

    async def call(_method, *, data, **_kwargs):
        calls.append(dict(data))
        if len(calls) == 2:
            raise telegram_api.TelegramRequestRejected(
                "Telegram API rejected sendMessage.",
                status_code=400,
                plain_retry_safe=True,
            )
        return {"result": {"message_id": len(calls)}}

    client.call = call
    message_id = asyncio.run(client.send_message(42, f"{first_chunk}\n\n{second_chunk}"))

    assert message_id == 3
    assert calls == [
        {"chat_id": "42", "text": first_chunk + "\n\n", "parse_mode": "HTML"},
        {"chat_id": "42", "text": "<b>later</b>", "parse_mode": "HTML"},
        {"chat_id": "42", "text": "later"},
    ]


def test_send_message_does_not_retry_ambiguous_transport_failure(monkeypatch):
    _plugin, telegram_api = _load_skill()
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("ambiguous delivery", request=request)

    _install_mock_transport(monkeypatch, telegram_api, handler)
    with pytest.raises(telegram_api.TelegramTransportError, match="timed out during sendMessage"):
        asyncio.run(telegram_api.TelegramClient("token").send_message(42, "hello"))
    assert attempts == 1


@pytest.mark.parametrize("response_kind", ["invalid_json", "non_object"])
def test_invalid_telegram_protocol_response_is_retryable(monkeypatch, response_kind):
    _plugin, telegram_api = _load_skill()

    def handler(_request):
        if response_kind == "invalid_json":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(200, json=[])

    _install_mock_transport(monkeypatch, telegram_api, handler)
    with pytest.raises(telegram_api.TelegramTransportError):
        asyncio.run(telegram_api.TelegramClient("token").call("getUpdates"))


@pytest.mark.parametrize(
    ("status_code", "transient"),
    [(400, False), (401, False), (409, False), (429, True), (500, True), (503, True)],
)
def test_telegram_rejection_classifies_only_retryable_statuses(status_code, transient):
    _plugin, telegram_api = _load_skill()
    exc = telegram_api.TelegramRequestRejected("rejected", status_code=status_code)
    assert exc.transient is transient


def test_telegram_rejection_uses_structured_error_code(monkeypatch):
    _plugin, telegram_api = _load_skill()

    def handler(_request):
        return httpx.Response(200, json={"ok": False, "error_code": 429})

    _install_mock_transport(monkeypatch, telegram_api, handler)
    with pytest.raises(telegram_api.TelegramRequestRejected) as captured:
        asyncio.run(telegram_api.TelegramClient("token").call("getUpdates"))

    assert captured.value.status_code == 429
    assert captured.value.transient is True


def test_edit_message_preserves_safe_not_modified_signal(monkeypatch):
    _plugin, telegram_api = _load_skill()
    token = "12345678:" + "C" * 35

    def handler(request):
        return httpx.Response(
            400,
            json={
                "ok": False,
                "description": (
                    "Bad Request: message is not modified: reflected "
                    f"{request.url}"
                ),
            },
        )

    _install_mock_transport(monkeypatch, telegram_api, handler)
    edited = asyncio.run(telegram_api.TelegramClient(token).edit_message_text(42, 7, "same"))
    assert edited is True


def test_inline_keyboard_edit_accepts_only_exact_not_modified_signal():
    _plugin, telegram_api = _load_skill()
    client = telegram_api.TelegramClient("token")

    async def exact_not_modified(_method, **_kwargs):
        raise RuntimeError("Telegram API editMessageText: message is not modified.")

    async def similar_but_not_exact(_method, **_kwargs):
        raise RuntimeError("Telegram API editMessageText: message is not modified. retry")

    async def real_failure(_method, **_kwargs):
        raise RuntimeError("Telegram API editMessageText returned HTTP 400.")

    client.call = exact_not_modified
    assert asyncio.run(client.edit_message_text_with_inline_keyboard(42, 7, "same", [])) is True
    client.call = similar_but_not_exact
    assert asyncio.run(client.edit_message_text_with_inline_keyboard(42, 7, "same", [])) is False
    client.call = real_failure
    assert asyncio.run(client.edit_message_text_with_inline_keyboard(42, 7, "same", [])) is False


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks
        self.reads = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        pass


def _file_handler(stream, *, content_length=None, file_path="payload.bin"):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"ok": True, "result": {"file_path": file_path}})
        headers = {} if content_length is None else {"Content-Length": str(content_length)}
        return httpx.Response(200, headers=headers, stream=stream)

    return handler


def test_photo_download_rejects_announced_oversize_without_reading(monkeypatch):
    _plugin, telegram_api = _load_skill()
    monkeypatch.setattr(telegram_api, "_MAX_TELEGRAM_DOWNLOAD_BYTES", 4)
    stream = _ChunkStream(b"x")
    _install_mock_transport(
        monkeypatch,
        telegram_api,
        _file_handler(stream, content_length=5, file_path="photo.jpg"),
    )

    with pytest.raises(RuntimeError, match="10 MiB download limit"):
        asyncio.run(telegram_api.TelegramClient("token").download_photo("file-id"))
    assert stream.reads == 0


def test_file_download_rejects_streamed_oversize_without_content_length(monkeypatch):
    _plugin, telegram_api = _load_skill()
    monkeypatch.setattr(telegram_api, "_MAX_TELEGRAM_DOWNLOAD_BYTES", 4)
    stream = _ChunkStream(b"1234", b"5")
    _install_mock_transport(monkeypatch, telegram_api, _file_handler(stream))

    with pytest.raises(RuntimeError, match="10 MiB download limit"):
        asyncio.run(telegram_api.TelegramClient("token").download_file("file-id"))
    assert stream.reads == 2


@pytest.mark.parametrize("content", [b"1234", b"ok"])
def test_file_download_accepts_boundary_and_small_responses(monkeypatch, content):
    _plugin, telegram_api = _load_skill()
    monkeypatch.setattr(telegram_api, "_MAX_TELEGRAM_DOWNLOAD_BYTES", 4)
    stream = _ChunkStream(content)
    _install_mock_transport(
        monkeypatch,
        telegram_api,
        _file_handler(stream, content_length=len(content)),
    )

    downloaded = asyncio.run(telegram_api.TelegramClient("token").download_file("file-id"))
    assert downloaded == content


def test_photo_download_accepts_small_response(monkeypatch):
    _plugin, telegram_api = _load_skill()
    monkeypatch.setattr(telegram_api, "_MAX_TELEGRAM_DOWNLOAD_BYTES", 4)
    stream = _ChunkStream(b"img")
    _install_mock_transport(
        monkeypatch,
        telegram_api,
        _file_handler(stream, content_length=3, file_path="photo.jpg"),
    )

    encoded, mime = asyncio.run(telegram_api.TelegramClient("token").download_photo("file-id"))
    assert base64.b64decode(encoded) == b"img"
    assert mime == "image/jpeg"


@pytest.mark.parametrize("trust_env", [None, True])
def test_telegram_http_clients_follow_the_constructor_trust_env(monkeypatch, trust_env):
    """Both TelegramClient transports (API calls and file downloads) take
    ``trust_env`` from the constructor — pinned False by default, so ambient
    HTTP(S)_PROXY and SSL_CERT_FILE/SSL_CERT_DIR (an env-injected MITM CA)
    never reach either client unless the caller opted in; the library itself
    reads no environment and imports nothing from the core."""
    _plugin, telegram_api = _load_skill()
    real_async_client = httpx.AsyncClient
    seen: list[dict] = []

    def handler(request):
        if "/file/" in str(request.url):
            return httpx.Response(200, content=b"bytes")
        return httpx.Response(200, json={"ok": True, "result": {}})

    def client_factory(**kwargs):
        seen.append(dict(kwargs))
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient", client_factory)
    kwargs = {} if trust_env is None else {"trust_env": trust_env}
    client = telegram_api.TelegramClient("token", **kwargs)

    asyncio.run(client.call("getMe"))
    asyncio.run(client._download_bytes("photos/file_1.jpg"))

    assert len(seen) == 2
    assert all(kwargs.get("trust_env") is bool(trust_env) for kwargs in seen)


@pytest.mark.parametrize(
    ("proxied", "worker", "expected"),
    [(False, False, False), (True, True, False), (False, True, False), (True, False, True)],
)
def test_plugin_honours_env_proxies_only_in_a_proxy_routed_server_process(
    tmp_path, monkeypatch, proxied, worker, expected,
):
    """The proxy decision is made once, at plugin import in the server process:
    True only when the install routes through a proxy (``env_proxies_configured``)
    AND this is not a supervisor worker (``in_worker_process``)."""
    import ouroboros.net_transport as net_transport
    import ouroboros.utils as utils

    monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: proxied)
    monkeypatch.setattr(utils, "in_worker_process", lambda: worker)
    plugin, _telegram_api = _load_skill()

    assert plugin._HONOR_ENV_PROXIES is expected

    # The decision is only worth making if it is wired: an outbound handler must
    # hand it to every TelegramClient it builds.
    built: list[dict] = []

    class _Recorder:
        def __init__(self, _token, **kwargs):
            built.append(kwargs)

        async def send_message(self, *_args, **_kwargs):
            return 1

    monkeypatch.setattr(plugin, "TelegramClient", _Recorder)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    quiz = plugin._make_quiz(_Api(tmp_path))
    asyncio.run(quiz({"question": "q?", "options": [{"label": "a"}, {"label": "b"}]}))

    assert built == [{"trust_env": expected}]
