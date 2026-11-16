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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_wait_for = asyncio.wait_for

    async def disarm_wait(awaitable, timeout):
        # Test the stop protocol, not how quickly a loaded host starts Python.
        # Production disarm still writes S, closes the pipe and awaits the live child.
        return await real_wait_for(awaitable, timeout=60 if timeout == 1.0 else timeout)

    monkeypatch.setattr(cloudflare.asyncio, "wait_for", disarm_wait)

    async def scenario() -> None:
        home = tmp_path / "home"
        home.mkdir()
        tunnel = cloudflare.QuickTunnel(tmp_path / "unused-cloudflared", tmp_path, 43123)
        tunnel._home = home
        environment = cloudflare._version_environment(home)
        assert "PYTHONPATH" not in environment

        await tunnel._start_watchdog(environment)
        process = tunnel._watchdog_process
        assert process is not None and process.returncode is None
        await real_wait_for(tunnel._disarm_watchdog(), timeout=65)
        assert process.returncode == 0 and tunnel._watchdog_write_fd is None
        assert not tunnel._watchdog_failed

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


@pytest.mark.parametrize("build,arch", [("win-amd64", "amd64"), ("win-arm64", "arm64"), ("win32", "x86"), ("unknown", "")])
def test_empty_windows_machine_uses_interpreter_build(monkeypatch, tmp_path, build, arch):
    import json
    import platform_support
    from runtime_status import RuntimeStatus

    monkeypatch.setattr(cloudflare.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cloudflare.platform, "machine", lambda: "")
    monkeypatch.setattr(platform_support.sysconfig, "get_platform", lambda: build)
    assert platform_support.machine_architecture() == arch
    if arch == "amd64":
        assert cloudflare._current_asset().platform_id == "windows-amd64"
    else:
        with pytest.raises(cloudflare.CloudflaredError, match="Unsupported"):
            cloudflare._current_asset()
    RuntimeStatus(tmp_path, cloudflared_version="test").publish()
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["platform"] == "windows-" + arch


@pytest.mark.parametrize(
    "system,machine,build,arch,platform_id",
    [
        ("Windows", "AMD64", "win32", "amd64", "windows-amd64"),
        ("Windows", "ARM64", "win-amd64", "arm64", None),
        ("Windows", "x86", "win-amd64", "x86", None),
        ("Windows", "mystery", "win-amd64", "mystery", None),
        ("Darwin", "arm64", "win32", "arm64", "darwin-arm64"),
        ("Linux", "aarch64", "win32", "aarch64", "linux-arm64"),
        ("Darwin", "", "win-amd64", "", None),
        ("Linux", "", "win-amd64", "", None),
    ],
)
def test_build_fallback_preserves_known_machine_and_other_os(
    monkeypatch, system, machine, build, arch, platform_id,
):
    import platform_support

    monkeypatch.setattr(platform_support.sysconfig, "get_platform", lambda: build)
    monkeypatch.setattr(cloudflare.platform, "system", lambda: system)
    monkeypatch.setattr(cloudflare.platform, "machine", lambda: machine)
    assert platform_support.machine_architecture() == arch
    if platform_id is None:
        with pytest.raises(cloudflare.CloudflaredError, match="Unsupported"):
            cloudflare._current_asset()
    else:
        assert cloudflare._current_asset().platform_id == platform_id


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/cloudflare/cloudflared",
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
        "https://three-words.trycloudflare.com.evil.example/",
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


def test_safe_log_detail_reads_a_bounded_head_and_shows_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 64 KiB line reaches the quadratic patterns only as its first 2 x 180 characters. A quoted credential
    value the cut leaves open stays redacted although delimiters inside it end an unquoted value, and the cut
    shows as "…" even when what is left after redaction is short."""
    class _Bounded:
        def __init__(self, pattern: Any) -> None:
            self.pattern = pattern

        def sub(self, replacement: str, text: str) -> str:
            assert len(text) <= 360, f"a pattern read {len(text)} characters"
            return self.pattern.sub(replacement, text)

    monkeypatch.setattr(cloudflare, "_LOG_URL_RE", _Bounded(cloudflare._LOG_URL_RE))
    monkeypatch.setattr(cloudflare, "_LOG_CREDENTIAL_RE", _Bounded(cloudflare._LOG_CREDENTIAL_RE))
    assert cloudflare._safe_log_detail("z" * 65_536) == "z" * 179 + "…"
    # The URL shrinks to [url], so the cut (inside the quoted value) falls within the 180 characters shown.
    line = "ERR https://" + "h" * 300 + ' secret="' + "s1,s2;" * 100 + '" tail'
    assert cloudflare._safe_log_detail("an earlier line\n" + line) == "ERR [url] secret=[redacted]…"


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


@pytest.mark.parametrize("port", [0, -1, "8765"])
def test_quick_tunnel_accepts_only_a_numeric_sidecar_port(port: Any) -> None:
    with pytest.raises(cloudflare.CloudflaredError):
        cloudflare.QuickTunnel(Path("/not-used"), Path("/not-used"), port)


# cloudflared 2026.7.2's stderr when Cloudflare refuses the tunnel request, written from its quick_tunnel.go (not
# captured from a live run): the non-JSON answer is logged with its HTTP status, then the process exits on a generic
# line that names neither the status nor Cloudflare's code.
_REQUESTED = b"2026-11-14T09:12:01Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
_REFUSAL = (
    b"2026-11-14T09:12:02Z ERR Error unmarshaling QuickTunnel response: error code: 1015 "
    b"error=\"invalid character 'e' looking for beginning of value\" status_code=\"429 Too Many Requests\"\n"
)
_GENERIC_LAST = b"failed to unmarshal quick Tunnel: invalid character 'e' looking for beginning of value\n"
# What the companion matched in the message before (PR #3533): the error's last line carries none of them.
_MESSAGE_MARKERS = ("error code: 1015", 'status_code="429', "too many requests")
# zerolog's console writer puts the body raw in the message, then the record's own fields on its last line:
# error first, the other fields sorted after it, each value quoted on one line.
_MARKER = b"Error unmarshaling QuickTunnel response: "
_FIELDS_429 = b" error=\"invalid character 'e' looking for beginning of value\" status_code=\"429 Too Many Requests\"\n"
_FIELDS_503 = b" error=\"invalid character '<' looking for beginning of value\" status_code=\"503 Service Unavailable\"\n"


def _refusal_with(body: bytes, fields: bytes = _FIELDS_429) -> bytes:
    return b"2026-11-14T09:12:02Z ERR " + _MARKER + body + fields


def _reads(data: bytes, size: int = 4096) -> list[bytes]:
    return [data[index:index + size] for index in range(0, len(data), size)]


def _reads_splitting(data: bytes, *tokens: bytes) -> list[bytes]:
    """Reads of at most 4096 bytes, one boundary falling inside each of ``tokens``."""
    cuts = sorted({data.index(token) + len(token) // 2 for token in tokens} | set(range(4096, len(data), 4096)))
    reads = [data[start:end] for start, end in zip([0, *cuts], [*cuts, len(data)]) if end > start]
    assert max(map(len, reads)) <= 4096 and not any(token in read for read in reads for token in tokens)
    return reads


def _tunnel_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: _FakeProcess) -> Any:
    spec = _fixture_spec(b"binary", _archive(b"binary"))
    monkeypatch.setattr(cloudflare, "_current_asset", lambda: spec)
    monkeypatch.setattr(cloudflare, "_verify_binary", lambda *_args: None)
    binary = cloudflare._installed_path(cloudflare._safe_state_root(tmp_path / "state"), spec)
    binary.write_bytes(b"binary")

    async def fake_spawn(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        return process

    async def fake_watchdog(_self: Any, _env: dict[str, str]) -> None:
        return None

    monkeypatch.setattr(cloudflare.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(cloudflare.QuickTunnel, "_start_watchdog", fake_watchdog)
    monkeypatch.setattr(cloudflare, "WindowsKillJob", _FakeWindowsJob)
    return cloudflare.QuickTunnel(binary, tmp_path / "state", 43123)


def _exited(stderr: list[bytes]) -> _FakeProcess:
    process = _FakeProcess(stderr)
    process.returncode = 1
    process._done.set()
    return process


_NOISE = b"2026-11-14T09:12:02Z DBG " + b"n" * 1000 + b"\n"


@pytest.mark.parametrize("stderr", [
    pytest.param([_REQUESTED + _REFUSAL + _GENERIC_LAST], id="one-read"),
    pytest.param(_reads(_REQUESTED + _REFUSAL + _GENERIC_LAST, 7), id="seven-byte-reads"),
    # More output than the bounded log keeps follows the refusal, so only a reading made as it arrived saw it.
    pytest.param(_reads(_REQUESTED + _REFUSAL + _NOISE * 70 + _GENERIC_LAST), id="evicted-from-the-log"),
    # A line longer than the 16 KiB scan window comes first.
    pytest.param(_reads(b"y" * 20_000 + b"\n" + _REQUESTED + _REFUSAL + _GENERIC_LAST), id="oversized-line-before"),
    # A body line with a level-like word but no RFC 3339 time stays inside the record.
    pytest.param([_REQUESTED + _refusal_with(b"<html>\nNotice INF limits apply\n</html>") + _GENERIC_LAST],
                 id="level-word-inside-the-body"),
])
def test_a_refused_tunnel_request_is_read_from_the_stream_not_the_last_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: list[bytes]
) -> None:
    async def scenario() -> None:
        tunnel = _tunnel_on(tmp_path, monkeypatch, _exited(stderr))
        await tunnel.start()
        with pytest.raises(cloudflare.CreationRateLimited) as captured:
            await tunnel.wait_url(1)
        assert isinstance(captured.value, cloudflare.CloudflaredError)
        message = str(captured.value)
        assert message.startswith("Cloudflare refused a new Quick Tunnel with HTTP 429.") and "(exit 1)" in message
        last = cloudflare._safe_log_detail(tunnel.stderr_tail)
        assert last == _GENERIC_LAST.decode().strip() and last in message
        assert not any(marker in last.lower() for marker in _MESSAGE_MARKERS)
        if len(tunnel.stderr_tail) < len(b"".join(stderr)):
            assert "status_code" not in tunnel.stderr_tail
        await tunnel.stop()

    asyncio.run(scenario())


def test_a_refusal_read_after_the_exit_is_still_the_exit_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process ends before its last output is read: the error waits for the readers, so it is still typed."""
    async def scenario() -> None:
        process = _FakeProcess([])

        class _LateStream:
            def __init__(self) -> None:
                self.reads = 0

            async def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return _REQUESTED
                if self.reads == 2:
                    process.returncode = 1
                    process._done.set()
                    for _ in range(5):  # the watcher sees the exit while this read still holds the refusal
                        await asyncio.sleep(0)
                    return _REFUSAL + _GENERIC_LAST
                return b""

        process.stderr = _LateStream()
        tunnel = _tunnel_on(tmp_path, monkeypatch, process)
        await tunnel.start()
        with pytest.raises(cloudflare.CreationRateLimited, match=r"exit 1"):
            await tunnel.wait_url(1)
        await tunnel.stop()

    asyncio.run(scenario())


class _OpenStream:
    """Gives its reads, then stays open until its process ends, as a pipe does."""

    def __init__(self, process: _FakeProcess, reads: list[bytes]) -> None:
        self.process, self.reads = process, list(reads)

    async def read(self, _size: int) -> bytes:
        await asyncio.sleep(0)
        if self.reads:
            return self.reads.pop(0)
        await self.process._done.wait()
        return b""


@pytest.mark.parametrize("ending", ["exit", "url_timeout"])
@pytest.mark.parametrize("body", [
    # One unbroken run, the last line on the URL-timeout path: _safe_log_detail reads only a bounded head of it
    # (read whole, the credential pattern spent seconds on such a run).
    pytest.param(b"error code: 1015 " + b"z" * 20_800, id="20-KiB-body-on-one-line"),
    pytest.param(b"<html>\n" + (b"<p>" + b"y" * 200 + b"</p>\n") * 100 + b"</html>", id="21-KiB-body-on-many-lines"),
])
def test_an_oversized_refusal_record_keeps_its_own_status_across_split_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes, ending: str
) -> None:
    """More than the 16 KiB scan window separates the record's message from its own status, and reads of at most
    4096 bytes split both the message and the status token: the refusal is typed when the process exits and,
    with its stream still open, when the URL wait times out."""
    async def scenario() -> None:
        record = _refusal_with(body)
        assert record.index(b'status_code="429') - record.index(_MARKER) > 16_384
        stream = _REQUESTED + record + (_GENERIC_LAST if ending == "exit" else b"")
        reads = _reads_splitting(stream, _MARKER, b'status_code="429')
        process = _exited(reads) if ending == "exit" else _FakeProcess([])
        if ending == "url_timeout":
            process.stderr = _OpenStream(process, reads)
        tunnel = _tunnel_on(tmp_path, monkeypatch, process)
        await tunnel.start()
        with pytest.raises(cloudflare.CreationRateLimited, match="exit 1" if ending == "exit" else "Timed out"):
            await tunnel.wait_url(1 if ending == "exit" else 0.2)
        await tunnel.stop()

    asyncio.run(scenario())


def test_a_refusal_record_is_read_in_bounded_memory() -> None:
    """A one-line record of 1 MiB: what the reading keeps of it never grows past its fixed bounds."""
    log = cloudflare._BoundedLog()
    answer = log.answer
    log.add(b"2026-11-14T09:12:02Z ERR " + _MARKER)
    for _ in range(256):
        log.add(b"z" * 4096)
        assert len(answer._head) <= 256 and len(answer._tail) <= 1024 and answer._own == ""
    log.add(_FIELDS_429)
    assert (answer.now(), answer.status) == ("429", "")  # the open record shows it; its end decides it
    answer.finish()
    assert answer.status == "429"


@pytest.mark.parametrize("refused", [True, False])
def test_a_url_timeout_carries_a_refusal_already_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    async def scenario() -> None:
        process = _FakeProcess([_REQUESTED, _REFUSAL] if refused else [_REQUESTED])  # it does not exit
        tunnel = _tunnel_on(tmp_path, monkeypatch, process)
        await tunnel.start()
        with pytest.raises(cloudflare.CloudflaredError, match="Timed out waiting") as captured:
            await tunnel.wait_url(0.05)
        assert isinstance(captured.value, cloudflare.CreationRateLimited) is refused
        await tunnel.stop()
        assert process.terminated

    asyncio.run(scenario())


@pytest.mark.parametrize("stderr", [
    # Cloudflare's refusal as JSON parses, so cloudflared logs no status: an ordinary failure, nothing invented.
    pytest.param([_REQUESTED, b"failed to parse quick Tunnel ID: invalid UUID length: 0\n"], id="json-answer"),
    # The record's own status decides, not a 429 printed after it.
    pytest.param([_REQUESTED, b"2026-11-14T09:12:02Z ERR Error unmarshaling QuickTunnel response: upstream error "
                  b"error=\"invalid character 'u' looking for beginning of value\" "
                  b"status_code=\"503 Service Unavailable\"\n",
                  b'2026-11-14T09:12:02Z WRN later status_code="429 Too Many Requests"\n', _GENERIC_LAST],
                 id="non-json-503"),
    pytest.param([b'2026-11-14T09:12:01Z ERR Failed to fetch features status_code="429 Too Many Requests"\n',
                  b"failed to request quick Tunnel: dial tcp: i/o timeout\n"], id="429-outside-the-request"),
    pytest.param([_REQUESTED, b"2026-11-14T09:12:02Z ERR error code: 1015 Too Many Requests\n",
                  b"failed to request quick Tunnel: EOF\n"], id="code-without-the-record"),
    # The record's own fields are missing, so its status is unavailable; the next record's 429 is not its own.
    pytest.param([_REQUESTED, b"2026-11-14T09:12:02Z ERR " + _MARKER + b"error code: 1015\n",
                  b'2026-11-14T09:12:02Z ERR Failed to fetch features error="EOF" status_code="429 Too Many Requests"\n',
                  _GENERIC_LAST], id="no-own-status-then-an-unrelated-429"),
    pytest.param([_REQUESTED, b"2026-11-14T09:12:02Z ERR " + _MARKER + b"error code: 1015\n", _GENERIC_LAST],
                 id="no-own-status-at-all"),
    # 429 text inside the untrusted body, the record's own status a 503 at the end of its last line.
    pytest.param([_REQUESTED, _refusal_with(b'<p>status_code="429 Too Many Requests"</p>', _FIELDS_503), _GENERIC_LAST],
                 id="body-429-text-then-own-503"),
    pytest.param([_REQUESTED, _refusal_with(b'<html>\n<p>x error="a" status_code="429 Too Many Requests"\n</html>',
                                            _FIELDS_503), _GENERIC_LAST], id="body-line-shaped-as-own-429-then-own-503"),
])
def test_other_failures_before_a_url_stay_ordinary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: list[bytes]
) -> None:
    async def scenario() -> None:
        tunnel = _tunnel_on(tmp_path, monkeypatch, _exited(stderr))
        await tunnel.start()
        with pytest.raises(cloudflare.CloudflaredError, match=r"exit 1") as captured:
            await tunnel.wait_url(1)
        assert type(captured.value) is cloudflare.CloudflaredError
        await tunnel.stop()

    asyncio.run(scenario())


class _Clocks:
    def __init__(self) -> None:
        self.wall, self.mono = 1_800_000_000.0, 5_000.0

    def step(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


def _pacing(clocks: _Clocks) -> Any:
    return cloudflare.CreationPacing(wall=lambda: clocks.wall, monotonic=lambda: clocks.mono)


def test_creation_pacing_waits_an_hour_then_ninety_minutes_until_a_url() -> None:
    clocks = _Clocks()
    pacing = _pacing(clocks)
    assert pacing.remaining() == 0 and pacing.until_epoch == 0
    pacing.limit_confirmed()
    assert (pacing.limits, pacing.remaining(), pacing.until_epoch) == (1, 3600, 1_800_003_600)
    clocks.step(3599)
    assert pacing.remaining() == 1
    clocks.step(1)
    assert pacing.remaining() == 0
    for _further in range(3):  # every further refusal before a URL is the repeated class: 2, the longer window
        pacing.limit_confirmed()
        assert (pacing.limits, pacing.remaining()) == (2, 5400)
        clocks.step(5400)
    pacing.url_allocated()
    assert (pacing.limits, pacing.remaining(), pacing.until_epoch) == (0, 0, 0)
    pacing.limit_confirmed()
    assert pacing.remaining() == 3600


@pytest.mark.parametrize("wall_step,mono_step,left", [
    pytest.param(1000, 1000, 2600, id="both-clocks-run"),
    pytest.param(3600, 0, 0, id="computer-slept"),  # the wall clock ran on: the deadline is reached
    pytest.param(-7200, 1000, 2600, id="wall-clock-set-back"),  # the monotonic deadline still ends it on time
    pytest.param(7200, 1000, 0, id="wall-clock-set-forward"),  # one early request; no provider reset is claimed
])
def test_creation_pacing_ends_on_either_clock(wall_step: float, mono_step: float, left: float) -> None:
    clocks = _Clocks()
    pacing = _pacing(clocks)
    pacing.limit_confirmed()
    published = pacing.until_epoch
    clocks.wall += wall_step
    clocks.mono += mono_step
    assert pacing.remaining() == left
    assert pacing.until_epoch == published  # status keeps naming the same deadline


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
