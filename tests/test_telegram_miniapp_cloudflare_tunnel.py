from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
MODULE_PATH = SCRIPTS_DIR / "cloudflare_tunnel.py"
SPEC = importlib.util.spec_from_file_location("telegram_poc_cloudflare_tunnel", MODULE_PATH)
assert SPEC and SPEC.loader
cloudflare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cloudflare
SPEC.loader.exec_module(cloudflare)


def _archive(payload: bytes, *, name: str = "cloudflared", kind: str = "file") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.mode = 0o755
        if kind == "file":
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        elif kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../evil"
            bundle.addfile(member)
        else:
            raise AssertionError(kind)
    return buffer.getvalue()


def _fixture_spec(payload: bytes, archive: bytes) -> Any:
    return cloudflare._AssetSpec(
        platform_id="darwin-arm64",
        arch="arm64",
        filename="cloudflared",
        kind="tgz",
        download_size=len(archive),
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-darwin-arm64.tgz"
        ),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        binary_sha256=hashlib.sha256(payload).hexdigest(),
    )


@pytest.mark.skipif(cloudflare.IS_WINDOWS, reason="POSIX watchdog uses a pipe lifeline")
def test_nested_watchdog_starts_with_exact_production_minimal_environment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        home = tmp_path / "home"
        home.mkdir()
        tunnel = cloudflare.QuickTunnel(tmp_path / "unused-cloudflared", tmp_path, 43123)
        tunnel._home = home
        environment = cloudflare._version_environment(home)
        assert "PYTHONPATH" not in environment

        await tunnel._start_watchdog(environment)
        assert tunnel._watchdog_process is not None
        assert tunnel._watchdog_process.returncode is None
        await tunnel._disarm_watchdog()
        assert tunnel._watchdog_process.returncode == 0

    asyncio.run(scenario())


def test_pinned_release_has_exact_cross_platform_assets() -> None:
    assert set(cloudflare._ASSETS) == {
        ("Darwin", "arm64"),
        ("Darwin", "amd64"),
        ("Linux", "arm64"),
        ("Linux", "amd64"),
        ("Windows", "amd64"),
    }
    arm = cloudflare._ASSETS[("Darwin", "arm64")]
    amd = cloudflare._ASSETS[("Darwin", "amd64")]
    assert arm.url.endswith("/2026.7.2/cloudflared-darwin-arm64.tgz")
    assert amd.url.endswith("/2026.7.2/cloudflared-darwin-amd64.tgz")
    assert arm.archive_sha256 == "2086e51c61d6565781d84117a5007d0c826d03ffdc74acb91c08c167f9f8cd7c"
    assert arm.binary_sha256 == "0588df58494a6cadd38b9deb6078908a5054063c80784d92fdb8d4a5f3de1c67"
    assert amd.archive_sha256 == "4ee0d3b48a990a2f9b5faec5838f73ec1f400aa8e0a4864be576adfafec406cb"
    assert amd.binary_sha256 == "a5afb0ba3da859da47bebc9a918d5b196bf7e4aec23589419b46356731bcc75f"
    for spec in cloudflare._ASSETS.values():
        assert spec.download_size > 0
        assert len(spec.archive_sha256) == 64
        assert len(spec.binary_sha256) == 64
    assert cloudflare._ASSETS[("Windows", "amd64")].filename == "cloudflared.exe"


@pytest.mark.parametrize(
    ("system", "machine", "platform_id"),
    [
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-amd64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Linux", "amd64", "linux-amd64"),
        ("Windows", "AMD64", "windows-amd64"),
    ],
)
def test_platform_matrix(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    platform_id: str,
) -> None:
    monkeypatch.setattr(cloudflare.platform, "system", lambda: system)
    monkeypatch.setattr(cloudflare.platform, "machine", lambda: machine)
    assert cloudflare._current_asset().platform_id == platform_id


@pytest.mark.parametrize(("system", "machine"), [("Windows", "arm64"), ("Linux", "mips64")])
def test_platform_matrix_rejects_unpinned_targets(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    monkeypatch.setattr(cloudflare.platform, "system", lambda: system)
    monkeypatch.setattr(cloudflare.platform, "machine", lambda: machine)
    with pytest.raises(cloudflare.CloudflaredError, match="Unsupported"):
        cloudflare._current_asset()


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/cloudflare/cloudflared",
        "https://github.com.evil.example/file",
        "https://user@github.com/file",
        "https://github.com:8443/file",
        "https://objects.githubusercontent.com/file",
        "file:///tmp/cloudflared",
    ],
)
def test_download_redirect_allowlist_is_exact(url: str) -> None:
    with pytest.raises(cloudflare.CloudflaredError):
        cloudflare._validate_download_url(url)
    cloudflare._validate_download_url("https://github.com/file")
    cloudflare._validate_download_url("https://release-assets.githubusercontent.com/file?sig=x")


class _FakeDownloadResponse:
    def __init__(self, body: bytes, url: str, content_length: str | None = None) -> None:
        self.body = body
        self.url = url
        self.offset = 0
        self.headers = {} if content_length is None else {"Content-Length": content_length}

    def __enter__(self) -> "_FakeDownloadResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _FakeDownloadOpener:
    def __init__(self, response: _FakeDownloadResponse) -> None:
        self.response = response

    def open(self, _request: Any, timeout: int) -> _FakeDownloadResponse:
        assert timeout == 30
        return self.response


def test_download_is_offline_fixture_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"pinned archive fixture"
    response = _FakeDownloadResponse(
        body,
        "https://release-assets.githubusercontent.com/signed-asset?sig=fixture",
        str(len(body)),
    )
    monkeypatch.setattr(
        cloudflare.urllib.request,
        "build_opener",
        lambda *_args: _FakeDownloadOpener(response),
    )
    spec = cloudflare._AssetSpec(
        platform_id="fixture",
        arch="arm64",
        filename="cloudflared",
        kind="raw",
        download_size=len(body),
        url="https://github.com/pinned-asset",
        archive_sha256=hashlib.sha256(body).hexdigest(),
        binary_sha256="0" * 64,
    )
    destination = tmp_path / "asset.tgz"
    cloudflare._download_asset(spec, destination)
    assert destination.read_bytes() == body


def test_download_rejects_untrusted_final_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _FakeDownloadResponse(b"x", "https://attacker.example/asset", "1")
    monkeypatch.setattr(
        cloudflare.urllib.request,
        "build_opener",
        lambda *_args: _FakeDownloadOpener(response),
    )
    spec = cloudflare._AssetSpec(
        platform_id="fixture",
        arch="arm64",
        filename="cloudflared",
        kind="raw",
        download_size=1,
        url="https://github.com/pinned-asset",
        archive_sha256=hashlib.sha256(b"x").hexdigest(),
        binary_sha256="0" * 64,
    )
    with pytest.raises(cloudflare.CloudflaredError, match="allowlist"):
        cloudflare._download_asset(spec, tmp_path / "asset.tgz")


def test_download_enforces_stream_limit_without_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _FakeDownloadResponse(
        b"12345",
        "https://release-assets.githubusercontent.com/asset",
    )
    monkeypatch.setattr(cloudflare, "MAX_ASSET_BYTES", 4)
    monkeypatch.setattr(
        cloudflare.urllib.request,
        "build_opener",
        lambda *_args: _FakeDownloadOpener(response),
    )
    spec = cloudflare._AssetSpec(
        platform_id="fixture",
        arch="arm64",
        filename="cloudflared",
        kind="raw",
        download_size=5,
        url="https://github.com/pinned-asset",
        archive_sha256=hashlib.sha256(b"12345").hexdigest(),
        binary_sha256="0" * 64,
    )
    with pytest.raises(cloudflare.CloudflaredError, match="70 MiB"):
        cloudflare._download_asset(spec, tmp_path / "asset.tgz")


def test_archive_extracts_exactly_one_regular_named_file(tmp_path: Path) -> None:
    payload = b"fixture-cloudflared"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    archive = tmp_path / "asset.tgz"
    destination = tmp_path / "cloudflared"
    archive.write_bytes(archive_bytes)
    cloudflare._extract_verified_binary(spec, archive, destination)
    assert destination.read_bytes() == payload
    if sys.platform != "win32":
        assert destination.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("../cloudflared", "file"),
        ("nested/cloudflared", "file"),
        ("cloudflared", "symlink"),
    ],
)
def test_archive_rejects_traversal_nested_and_links(
    tmp_path: Path, name: str, kind: str
) -> None:
    payload = b"fixture"
    archive_bytes = _archive(payload, name=name, kind=kind)
    spec = _fixture_spec(payload, archive_bytes)
    archive = tmp_path / "asset.tgz"
    archive.write_bytes(archive_bytes)
    with pytest.raises(cloudflare.CloudflaredError, match="unsafe member"):
        cloudflare._extract_verified_binary(spec, archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_archive_rejects_extra_member(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name in ("cloudflared", "README"):
            member = tarfile.TarInfo(name)
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
    archive = tmp_path / "asset.tgz"
    archive.write_bytes(buffer.getvalue())
    spec = _fixture_spec(b"x", buffer.getvalue())
    with pytest.raises(cloudflare.CloudflaredError, match="exactly one"):
        cloudflare._extract_verified_binary(spec, archive, tmp_path / "out")


def test_archive_rejects_wrong_extracted_digest(tmp_path: Path) -> None:
    archive_bytes = _archive(b"actual")
    spec = cloudflare._AssetSpec(
        platform_id="fixture",
        arch="arm64",
        filename="cloudflared",
        kind="tgz",
        download_size=len(archive_bytes),
        url="https://github.com/file",
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        binary_sha256=hashlib.sha256(b"different").hexdigest(),
    )
    archive = tmp_path / "asset.tgz"
    archive.write_bytes(archive_bytes)
    with pytest.raises(cloudflare.CloudflaredError, match="Extracted"):
        cloudflare._extract_verified_binary(spec, archive, tmp_path / "out")


def test_ensure_reuses_only_verified_cache_and_checks_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-binary"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    root = cloudflare._safe_state_root(tmp_path / "state")
    target = cloudflare._installed_path(root, spec)
    target.write_bytes(payload)
    versions: list[Path] = []
    monkeypatch.setattr(
        cloudflare,
        "_verify_version",
        lambda binary, _home: versions.append(binary),
    )
    monkeypatch.setattr(
        cloudflare,
        "_download_asset",
        lambda _spec, _path: pytest.fail("valid cache must not download"),
    )
    assert asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state")) == target
    assert versions == [target]


def test_failed_replacement_preserves_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"new-valid-binary"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_version", lambda *_args: None)
    root = cloudflare._safe_state_root(tmp_path / "state")
    target = cloudflare._installed_path(root, spec)
    target.write_bytes(b"old-tampered-binary")

    def bad_download(_spec: Any, path: Path) -> None:
        path.write_bytes(b"wrong archive")

    monkeypatch.setattr(cloudflare, "_download_asset", bad_download)
    with pytest.raises(cloudflare.CloudflaredError):
        asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state"))
    assert target.read_bytes() == b"old-tampered-binary"


def test_raw_windows_asset_uses_exe_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned-windows-binary"
    spec = cloudflare._AssetSpec(
        platform_id="windows-amd64",
        arch="amd64",
        filename="cloudflared.exe",
        kind="raw",
        download_size=len(payload),
        url="https://github.com/cloudflare/cloudflared/releases/download/2026.7.2/cloudflared.exe",
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        binary_sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_version", lambda *_args: None)
    monkeypatch.setattr(
        cloudflare,
        "_download_asset",
        lambda _spec, destination: destination.write_bytes(payload),
    )
    target = asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state"))
    assert target.name == "cloudflared.exe"
    assert target.read_bytes() == payload


def test_version_output_requires_exact_pinned_release() -> None:
    assert cloudflare._VERSION_RE.fullmatch(
        "cloudflared version 2026.7.2 (built 2026-07-15-13:30 UTC)\n"
    )
    for output in (
        "cloudflared version 2026.7.20 (built 2026-07-15-13:30 UTC)\n",
        "prefix cloudflared version 2026.7.2 (built 2026-07-15-13:30 UTC)\n",
        "cloudflared version 2026.7.2 extra\n",
        "cloudflared version 2026.7.2 (built attacker\n",
    ):
        assert not cloudflare._VERSION_RE.fullmatch(output)
    assert cloudflare._VERSION_RE.fullmatch(
        "cloudflared version 2026.7.2 (built 2026-07-15-13:31 UTC)\r\n"
    )
    assert cloudflare._VERSION_RE.fullmatch("cloudflared version 2026.7.2\n")


@pytest.mark.parametrize(
    "value",
    [
        "http://three-words.trycloudflare.com",
        "https://three-words.trycloudflare.com/path",
        "https://three-words.trycloudflare.com/?query=1",
        "https://three-words.trycloudflare.com:443/",
        "https://three-words.trycloudflare.com.evil.example/",
        "https://a.b.trycloudflare.com/",
        "https://user@three-words.trycloudflare.com/",
        "https://-bad.trycloudflare.com/",
    ],
)
def test_quick_tunnel_url_rejects_non_root_or_noncanonical(value: str) -> None:
    assert cloudflare._quick_tunnel_url(value) is None


def test_quick_tunnel_url_accepts_only_exact_root() -> None:
    text = "INF Your quick Tunnel has been created! Visit https://three-words.trycloudflare.com"
    assert cloudflare._quick_tunnel_url(text) == "https://three-words.trycloudflare.com/"


def test_bounded_log_keeps_only_tail() -> None:
    log = cloudflare._BoundedLog(maximum=12)
    assert log.add(b"a" * 40) is None
    assert log.text() == "a" * 12


def test_safe_log_detail_redacts_urls_and_credential_shapes() -> None:
    detail = cloudflare._safe_log_detail(
        'ERR url=http://api.trycloudflare.com/tunnel '
        '{"token":"json-secret"}, tunnelSecret=field-secret, '
        'authorization: Bearer header-secret'
    )
    assert detail.count("[url]") == 1
    assert detail.count("[redacted]") == 3
    for hidden in (
        "api.trycloudflare.com",
        "json-secret",
        "field-secret",
        "header-secret",
    ):
        assert hidden not in detail


class _FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        await asyncio.sleep(0)
        return self.chunks.pop(0) if self.chunks else b""


class _FakeProcess:
    def __init__(self, stderr: list[bytes]) -> None:
        self.pid = 43210
        self.stdout = _FakeStream([])
        self.stderr = _FakeStream(stderr)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._done = asyncio.Event()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()


class _FakeWindowsJob:
    def __init__(self) -> None:
        self.assigned: list[int] = []
        self.closed = False

    def assign(self, pid: int) -> None:
        self.assigned.append(pid)

    def close(self) -> None:
        self.closed = True


def test_quick_tunnel_exact_argv_url_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        spec = _fixture_spec(b"binary", _archive(b"binary"))
        monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
        monkeypatch.setattr(cloudflare, "_verify_binary", lambda *_args: None)
        root = cloudflare._safe_state_root(tmp_path / "state")
        binary = cloudflare._installed_path(root, spec)
        binary.write_bytes(b"binary")
        process = _FakeProcess(
            [b"INF https://blue-bird-7.trycloudflare.com\n"],
        )
        windows_job = _FakeWindowsJob()
        observed: dict[str, Any] = {}

        async def fake_spawn(*argv: str, **kwargs: Any) -> _FakeProcess:
            observed["argv"] = list(argv)
            observed["kwargs"] = kwargs
            return process

        async def fake_watchdog(_self: Any, _env: dict[str, str]) -> None:
            return None

        monkeypatch.setattr(cloudflare.asyncio, "create_subprocess_exec", fake_spawn)
        monkeypatch.setattr(cloudflare.QuickTunnel, "_start_watchdog", fake_watchdog)
        monkeypatch.setattr(cloudflare, "WindowsKillJob", lambda: windows_job)
        tunnel = cloudflare.QuickTunnel(binary, tmp_path / "state", 43123)
        assert tunnel.returncode is None
        await tunnel.start()
        assert tunnel.returncode is None
        assert await tunnel.wait_url(1) == "https://blue-bird-7.trycloudflare.com/"
        assert observed["argv"] == [
            str(binary.resolve()),
            "tunnel",
            "--no-autoupdate",
            "--url",
            "http://127.0.0.1:43123",
        ]
        assert observed["kwargs"]["env"]["HOME"].startswith(str(tmp_path / "state"))
        assert "TELEGRAM_BOT_TOKEN" not in observed["kwargs"]["env"]
        isolated_home = Path(observed["kwargs"]["env"]["HOME"])
        assert isolated_home.is_dir()
        await tunnel.stop()
        assert process.terminated is True
        assert process.killed is False
        assert tunnel.returncode == 0
        assert not isolated_home.exists()
        if cloudflare.IS_WINDOWS:
            assert windows_job.assigned == [process.pid]
            assert windows_job.closed is True

    asyncio.run(scenario())


def test_quick_tunnel_reports_death_before_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        spec = _fixture_spec(b"binary", _archive(b"binary"))
        monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
        monkeypatch.setattr(cloudflare, "_verify_binary", lambda *_args: None)
        root = cloudflare._safe_state_root(tmp_path / "state")
        binary = cloudflare._installed_path(root, spec)
        binary.write_bytes(b"binary")
        process = _FakeProcess(
            [
                b"ERR allocation failed url=https://api.trycloudflare.com/tunnel "
                b"token=secret-value\n"
            ]
        )
        process.returncode = 17
        process._done.set()
        windows_job = _FakeWindowsJob()

        async def fake_spawn(*_args: Any, **_kwargs: Any) -> _FakeProcess:
            return process

        async def fake_watchdog(_self: Any, _env: dict[str, str]) -> None:
            return None

        monkeypatch.setattr(cloudflare.asyncio, "create_subprocess_exec", fake_spawn)
        monkeypatch.setattr(cloudflare.QuickTunnel, "_start_watchdog", fake_watchdog)
        monkeypatch.setattr(cloudflare, "WindowsKillJob", lambda: windows_job)
        tunnel = cloudflare.QuickTunnel(binary, tmp_path / "state", 43123)
        await tunnel.start()
        isolated_home = tunnel._home
        assert isolated_home is not None and isolated_home.is_dir()
        with pytest.raises(cloudflare.CloudflaredError, match="exit 17") as captured:
            await tunnel.wait_url(1)
        message = str(captured.value)
        assert "allocation failed" in message
        assert "[url]" in message
        assert "[redacted]" in message
        assert "api.trycloudflare.com" not in message
        assert "secret-value" not in message
        assert await tunnel.wait() == 17
        assert not isolated_home.exists()
        if cloudflare.IS_WINDOWS:
            assert windows_job.assigned == [process.pid]
            assert windows_job.closed is True

    asyncio.run(scenario())


def test_cancelled_wait_wrapper_does_not_cancel_owned_watcher(tmp_path: Path) -> None:
    async def scenario() -> None:
        tunnel = cloudflare.QuickTunnel(Path("/not-used"), tmp_path / "state", 43123)
        process = _FakeProcess([])
        process.returncode = 0
        process._done.set()
        release_watch = asyncio.Event()
        watch_task = asyncio.create_task(release_watch.wait())
        tunnel._process = process
        tunnel._watch_task = watch_task

        waiter = asyncio.create_task(tunnel.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not watch_task.cancelled()

        release_watch.set()
        await tunnel.stop()
        assert watch_task.done()

    asyncio.run(scenario())


def test_simultaneous_watcher_and_waiter_cancellation_preserves_caller_cancel(
    tmp_path: Path,
) -> None:
    async def one_race() -> None:
        tunnel = cloudflare.QuickTunnel(Path("/not-used"), tmp_path / "state", 43123)
        process = _FakeProcess([])
        process.returncode = 0
        process._done.set()
        watch_task = asyncio.create_task(asyncio.Event().wait())
        tunnel._process = process
        tunnel._watch_task = watch_task

        waiter = asyncio.create_task(tunnel.wait())
        await asyncio.sleep(0)
        watch_task.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.gather(watch_task, return_exceptions=True)

    async def scenario() -> None:
        for _attempt in range(20):
            await one_race()

    asyncio.run(scenario())


def test_stop_recovers_already_cancelled_owned_watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        tunnel = cloudflare.QuickTunnel(Path("/not-used"), tmp_path / "state", 43123)
        process = _FakeProcess([])
        process.returncode = 0
        process._done.set()
        watch_task = asyncio.create_task(asyncio.Event().wait())
        tunnel._process = process
        tunnel._watch_task = watch_task
        watch_task.cancel()
        await asyncio.gather(watch_task, return_exceptions=True)
        disarmed = 0

        async def disarm() -> None:
            nonlocal disarmed
            disarmed += 1

        monkeypatch.setattr(tunnel, "_disarm_watchdog", disarm)
        await tunnel.stop()
        assert disarmed == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("port", [0, 65536, -1, True, "8765"])
def test_quick_tunnel_accepts_only_a_numeric_sidecar_port(port: Any) -> None:
    with pytest.raises(cloudflare.CloudflaredError):
        cloudflare.QuickTunnel(Path("/not-used"), Path("/not-used"), port)


def test_download_is_bounded_by_a_whole_download_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-read socket timeout bounds one stalled read, not a trickling
    server; the whole download is bounded too, so the private install lock
    (and the worker thread an async caller cannot cancel) is held at most
    DOWNLOAD_DEADLINE_SEC."""
    response = _FakeDownloadResponse(
        b"12345",
        "https://release-assets.githubusercontent.com/asset",
    )
    monkeypatch.setattr(cloudflare, "DOWNLOAD_DEADLINE_SEC", -1)
    monkeypatch.setattr(
        cloudflare.urllib.request,
        "build_opener",
        lambda *_args: _FakeDownloadOpener(response),
    )
    spec = cloudflare._AssetSpec(
        platform_id="fixture",
        arch="arm64",
        filename="cloudflared",
        kind="raw",
        download_size=5,
        url="https://github.com/pinned-asset",
        archive_sha256=hashlib.sha256(b"12345").hexdigest(),
        binary_sha256="0" * 64,
    )
    with pytest.raises(cloudflare.CloudflaredError, match="deadline"):
        cloudflare._download_asset(spec, tmp_path / "asset.tgz")
    assert not (tmp_path / "asset.tgz").exists() or (tmp_path / "asset.tgz").stat().st_size <= 5


def _install_lock_is_free(lock_path: Path) -> bool:
    """Probe the private install lock from a second descriptor (same process)."""
    try:
        fd = cloudflare.acquire_file_lock(lock_path, timeout_sec=0.0)
    except cloudflare.PlatformSupportError:
        return False
    cloudflare.release_file_lock(fd)
    return True


def _leftover_candidates(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith((".asset-", ".binary-")))


def test_download_runs_outside_the_install_lock_and_promotion_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install lock serializes only the atomic promote. The download (bounded
    by DOWNLOAD_DEADLINE_SEC) used to run UNDER it, so a concurrent installer
    waited out somebody else's whole download before it could even re-verify."""
    payload = b"fresh-verified-binary"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_version", lambda *_args: None)
    root = cloudflare._safe_state_root(tmp_path / "state")
    target = cloudflare._installed_path(root, spec)
    lock_path = target.parent / ".install.lock"
    seen: list[str] = []

    def download(_spec: Any, destination: Path) -> None:
        seen.append("download:" + ("free" if _install_lock_is_free(lock_path) else "held"))
        destination.write_bytes(archive_bytes)

    real_fsync = cloudflare.fsync_directory

    def fsync(directory: Path) -> None:
        # Runs right after os.replace(candidate, target): the promote step.
        seen.append("promote:" + ("free" if _install_lock_is_free(lock_path) else "held"))
        real_fsync(directory)

    monkeypatch.setattr(cloudflare, "_download_asset", download)
    monkeypatch.setattr(cloudflare, "fsync_directory", fsync)
    assert asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state")) == target
    assert target.read_bytes() == payload
    assert seen == ["download:free", "promote:held"]
    assert _leftover_candidates(target.parent) == []
    assert _install_lock_is_free(lock_path)


def test_verified_install_promoted_during_the_download_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the download outside the lock, a sibling installer can promote its
    own verified copy meanwhile: the promote re-verifies under the lock and
    never replaces a verified install; the unused candidate is discarded."""
    payload = b"sibling-verified-binary"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_version", lambda *_args: None)
    root = cloudflare._safe_state_root(tmp_path / "state")
    target = cloudflare._installed_path(root, spec)

    def download(_spec: Any, destination: Path) -> None:
        target.write_bytes(payload)  # the sibling's verified install lands
        destination.write_bytes(archive_bytes)

    monkeypatch.setattr(cloudflare, "_download_asset", download)
    monkeypatch.setattr(
        cloudflare, "fsync_directory", lambda _d: pytest.fail("a verified install must not be replaced")
    )
    assert asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state")) == target
    assert target.read_bytes() == payload
    assert _leftover_candidates(target.parent) == []


def test_failed_download_leaves_no_candidate_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"never-installed"
    archive_bytes = _archive(payload)
    spec = _fixture_spec(payload, archive_bytes)
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_version", lambda *_args: None)
    root = cloudflare._safe_state_root(tmp_path / "state")
    target = cloudflare._installed_path(root, spec)

    def download(_spec: Any, destination: Path) -> None:
        destination.write_bytes(b"partial")
        raise cloudflare.CloudflaredError("Cloudflared download exceeded the deadline.")

    monkeypatch.setattr(cloudflare, "_download_asset", download)
    with pytest.raises(cloudflare.CloudflaredError, match="deadline"):
        asyncio.run(cloudflare.ensure_cloudflared(tmp_path / "state"))
    assert not target.exists()
    assert _leftover_candidates(target.parent) == []
    assert _install_lock_is_free(target.parent / ".install.lock")
