"""``OUROBOROS_EXTRA_CA_BUNDLE``: the owner's extra CA rides every first-party client, additively.

Both directions are pinned: without the setting the clients are built exactly as
before (no ``verify`` at all), and with it a throwaway CA is trusted by the
no-proxy transport while certifi's anchors stay in the merged bundle.
"""
from __future__ import annotations

import http.server
import pathlib
import socketserver
import ssl
import sys
import threading
import types

import re

import pytest

from ouroboros import net_transport
from tests._throwaway_ca import throwaway_ca, throwaway_crl


@pytest.fixture(autouse=True)
def _isolated_bundle(monkeypatch, tmp_path):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_EXTRA_CA_BUNDLE", raising=False)
    net_transport._merged_bundle_cache.clear()
    yield
    net_transport._merged_bundle_cache.clear()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib callback name
        body = b"trusted"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


class _QuietTlsServer(socketserver.TCPServer):
    allow_reuse_address = True

    def handle_error(self, request, client_address):  # a refused handshake is the expected negative case
        return


@pytest.fixture
def tls_server(tmp_path):
    ca_pem, cert_pem, key_pem = throwaway_ca(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_pem), str(key_pem))
    server = _QuietTlsServer(("127.0.0.1", 0), _Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_address[1]}/", ca_pem
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_unset_setting_leaves_every_client_as_before(monkeypatch):
    import httpx

    assert net_transport.extra_ca_bundle() is None
    assert net_transport.verify_kwargs() == {} and net_transport.requests_verify_kwargs() == {}
    seen: dict = {}

    class _Transport:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(httpx, "HTTPTransport", _Transport)
    net_transport.remote_httpx_transport(trust_env=False)
    assert "verify" not in seen and seen["trust_env"] is False


def test_setting_merges_the_owner_file_over_certifi_and_rotates_with_its_content(monkeypatch, tmp_path):
    import certifi

    extra, _cert, _key = throwaway_ca(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))

    merged = pathlib.Path(net_transport.extra_ca_bundle())
    assert merged.parent == tmp_path / "data" / "state" / "extra-ca-bundle"
    assert re.fullmatch(r"[0-9a-f]{12}\.pem", merged.name), merged.name
    body = merged.read_bytes()
    assert body.startswith(pathlib.Path(certifi.where()).read_bytes().rstrip(b"\n"))
    assert body.endswith(extra.read_bytes().rstrip(b"\n") + b"\n")
    assert net_transport.requests_verify_kwargs() == {"verify": str(merged)}
    context = net_transport.verify_kwargs()["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert net_transport.trust_ssl_context() is context, "one context per merged bundle"

    stamp = merged.stat().st_mtime_ns
    net_transport._merged_bundle_cache.clear()
    assert net_transport.extra_ca_bundle() == str(merged)
    assert merged.stat().st_mtime_ns == stamp, "unchanged content must not be rewritten"

    second, _cert2, _key2 = throwaway_ca(tmp_path / "second")
    extra.write_bytes(second.read_bytes())
    rotated = pathlib.Path(net_transport.extra_ca_bundle())
    assert rotated != merged and rotated.read_bytes().endswith(second.read_bytes().rstrip(b"\n") + b"\n")
    assert merged.exists(), "a fresh sibling stays: a task still holding the earlier setting may use it"
    import os as _os
    import time as _time

    _os.utime(merged, (_time.time() - 2 * 86400, _time.time() - 2 * 86400))
    net_transport._merged_bundle_cache.clear()
    assert net_transport.extra_ca_bundle() == str(rotated)
    assert not merged.exists(), "a sibling older than a day is pruned"
    assert net_transport.trust_ssl_context() is not context, "a new owner file rotates the SSL context"
    assert net_transport.verify_kwargs()["verify"] is net_transport.trust_ssl_context()


def test_a_replacement_under_the_same_path_is_read_although_it_keeps_size_and_modification_time(monkeypatch, tmp_path):
    import os

    first, _cert, _key = throwaway_ca(tmp_path / "first")
    second, _cert2, _key2 = throwaway_ca(tmp_path / "second")
    owner = tmp_path / "owner.pem"
    owner.write_bytes(first.read_bytes() + second.read_bytes())
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(owner))
    before = net_transport.extra_ca_bundle()
    kept = owner.stat()
    staged = tmp_path / "staged.pem"  # the same certificates in the other order: the same size, other bytes
    staged.write_bytes(second.read_bytes() + first.read_bytes())
    os.utime(staged, ns=(kept.st_atime_ns, kept.st_mtime_ns))
    os.replace(staged, owner)
    assert (owner.stat().st_size, owner.stat().st_mtime_ns) == (kept.st_size, kept.st_mtime_ns)

    after = net_transport.extra_ca_bundle()  # the same process, its cache intact
    assert after != before
    assert pathlib.Path(after).read_bytes().endswith(second.read_bytes() + first.read_bytes())


def test_a_changed_file_identity_during_loading_is_refused_by_both_consumers(monkeypatch, tmp_path):
    owner, _cert, _key = throwaway_ca(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(owner))
    original = net_transport._ca_file_identity(owner)
    for load in (net_transport.extra_ca_bundle, net_transport.extra_ca_context):
        observations = iter([original, (*original[:-1], original[-1] + 1)])
        with monkeypatch.context() as observed:
            observed.setattr(net_transport, "_ca_file_identity", lambda _path: next(observations))
            with pytest.raises(net_transport.ExtraCaBundleError, match="changed while it was read"):
                load()


@pytest.mark.parametrize("content", [None, b"not a certificate\n", b"-----BEGIN CERTIFICATE-----\nMIIBogus\n-----END CERTIFICATE-----\n"])
def test_an_unusable_file_is_a_loud_error_not_a_silent_fallback(monkeypatch, tmp_path, content):
    extra = tmp_path / "extra.pem"
    if content is not None:
        extra.write_bytes(content)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))
    with pytest.raises(net_transport.ExtraCaBundleError):
        net_transport.extra_ca_bundle()
    with pytest.raises(net_transport.ExtraCaBundleError):
        net_transport.remote_httpx_transport(trust_env=False)


def test_a_revocation_list_without_a_certificate_is_refused_by_both_consumers(monkeypatch, tmp_path):
    crl = throwaway_crl(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(crl))
    for load in (net_transport.extra_ca_bundle, net_transport.extra_ca_context):
        with pytest.raises(net_transport.ExtraCaBundleError, match="holds no certificate to trust"):
            load()


def test_an_owner_root_already_in_httpx_trust_still_counts_as_a_certificate(monkeypatch, tmp_path):
    ca_pem, _, _ = throwaway_ca(tmp_path)
    mixed = tmp_path / "mixed.pem"
    owner_bytes = ca_pem.read_bytes() + throwaway_crl(tmp_path).read_bytes() + ca_pem.read_bytes()
    mixed.write_bytes(owner_bytes)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(mixed))
    assert pathlib.Path(net_transport.extra_ca_bundle()).read_bytes().endswith(owner_bytes)
    checked = net_transport.extra_ca_context()
    assert checked.cert_store_stats()["x509"] == 1
    assert checked.verify_mode == ssl.CERT_REQUIRED and checked.check_hostname


@pytest.mark.parametrize("shape", ["der", "home", "body"])
def test_a_file_openssl_cannot_read_as_pem_is_typed_at_the_setting(monkeypatch, tmp_path, shape):
    ca_pem, _cert, _key = throwaway_ca(tmp_path / "ca")
    pem = ca_pem.read_bytes()
    extra = tmp_path / "extra.cer"
    lines = pem.split(b"\n")
    extra.write_bytes({"der": ssl.PEM_cert_to_DER_cert(pem.decode()),  # a binary certificate, not PEM text
                       "body": b"\n".join([lines[0], "Ж".encode() + lines[1], *lines[2:]]),  # a damaged body stays so
                       "home": b""}[shape])
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", "~synthetic-nobody-e22094a0/x.pem" if shape == "home" else str(extra))
    for read in (net_transport.extra_ca_bundle, net_transport.extra_ca_context):
        with pytest.raises(net_transport.ExtraCaBundleError) as refused:
            read()
        assert ("convert it to PEM" in str(refused.value)) is (shape == "der"), refused.value


def test_text_around_the_pem_blocks_is_read_as_openssl_reads_its_own_files(monkeypatch, tmp_path):
    ca_pem, _cert, _key = throwaway_ca(tmp_path / "ca")
    pem = ca_pem.read_bytes()
    extra = tmp_path / "extra.pem"
    extra.write_bytes(b"\xef\xbb\xbf# " + "Корневой сертификат".encode() + b"\n" + pem + "# конец\n".encode())
    reference = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    reference.load_verify_locations(cafile=str(extra))  # OpenSSL's own file reader accepts it
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))
    import certifi

    base = ssl.create_default_context(cafile=certifi.where()).cert_store_stats()["x509_ca"]  # httpx's own base here
    assert net_transport.extra_ca_context().cert_store_stats()["x509_ca"] == base + 1
    merged = pathlib.Path(net_transport.extra_ca_bundle())
    assert not merged.read_bytes().count(b"\xef\xbb\xbf") and pem in merged.read_bytes()
    assert net_transport.trust_ssl_context().cert_store_stats()["x509_ca"] == base + 1


def test_proxy_routed_installs_keep_sdk_defaults_unless_a_bundle_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: True)
    assert net_transport.keepalive_http_client() is None
    extra, _cert, _key = throwaway_ca(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))
    openai = pytest.importorskip("openai")
    client = net_transport.keepalive_http_client()
    assert isinstance(client, openai.DefaultHttpxClient)
    client.close()


def test_gigachat_client_receives_the_bundle_only_when_configured(monkeypatch, tmp_path):
    from ouroboros.llm_gigachat import _GigaChatLaneMixin

    seen: list = []
    module = types.ModuleType("gigachat")

    class _GigaChat:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    module.GigaChat = _GigaChat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigachat", module)
    target = {"api_key": "k", "scope": "GIGACHAT_API_PERS", "verify_ssl_certs": True}

    _GigaChatLaneMixin._new_gigachat_client(target)
    assert "ca_bundle_file" not in seen[-1]

    extra, _cert, _key = throwaway_ca(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))
    _GigaChatLaneMixin._new_gigachat_client(target)
    assert seen[-1]["ca_bundle_file"] == net_transport.extra_ca_bundle()
    assert seen[-1]["verify_ssl_certs"] is True


def test_no_proxy_transport_trusts_the_owner_ca_only_when_configured(monkeypatch, tls_server):
    import httpx

    url, ca_pem = tls_server
    with httpx.Client(transport=net_transport.remote_httpx_transport(trust_env=False), timeout=10) as client:
        with pytest.raises(httpx.ConnectError):
            client.get(url)

    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(ca_pem))
    net_transport._merged_bundle_cache.clear()
    with httpx.Client(transport=net_transport.remote_httpx_transport(trust_env=False), timeout=10) as client:
        assert client.get(url).text == "trusted"


def test_provider_test_names_the_trust_bundle_failure():
    """Provider Test must show the owner's trust-bundle error, not a generic request failure."""
    from ouroboros.llm_probe import controlled_probe_error

    result = controlled_probe_error(net_transport.ExtraCaBundleError("OUROBOROS_EXTRA_CA_BUNDLE holds no loadable PEM certificate: /x.pem"))
    assert result["ok"] is False
    assert "OUROBOROS_EXTRA_CA_BUNDLE" in result["error"]


def test_openrouter_ground_truth_verifies_against_the_owner_bundle(monkeypatch, tmp_path):
    """The supervisor's OpenRouter usage check is a provider call too: it must carry the trust context."""
    import io
    import urllib.request

    from supervisor.state import check_openrouter_ground_truth

    seen: dict = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp(b'{"data": {"usage": 1.5, "usage_daily": 0.5}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert check_openrouter_ground_truth() == {"total_usd": 1.5, "daily_usd": 0.5}
    assert seen["context"] is None

    extra, _cert, _key = throwaway_ca(tmp_path)
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(extra))
    assert check_openrouter_ground_truth() == {"total_usd": 1.5, "daily_usd": 0.5}
    assert seen["context"] is net_transport.trust_ssl_context()
