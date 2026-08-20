from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import stat
import tarfile
import zipfile

import pytest

from ouroboros import betterleaks_runtime as runtime

_LICENSE = b"fixture license\n"
_README = b"fixture readme\n"
_BINARY = b"fixture executable\n"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact(path: pathlib.Path, *, kind: str, binary_member: str) -> runtime.BetterleaksArtifact:
    return runtime.BetterleaksArtifact(
        platform_key="windows-x64" if kind == "zip" else "linux-x64",
        version=runtime.BETTERLEAKS_VERSION,
        release_commit=runtime.BETTERLEAKS_RELEASE_COMMIT,
        archive_url=f"https://example.invalid/{path.name}",
        archive_name=path.name,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        archive_kind=kind,
        binary_member=binary_member,
        license_member=runtime.LICENSE_MEMBER,
        license_sha256=_sha(_LICENSE),
    )


def _tar(
    path: pathlib.Path,
    *,
    entries: list[tuple[str, bytes, str]] | None = None,
) -> pathlib.Path:
    rows = entries or [
        (runtime.LICENSE_MEMBER, _LICENSE, "file"),
        (runtime.README_MEMBER, _README, "file"),
        ("betterleaks", _BINARY, "file"),
    ]
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, kind in rows:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "betterleaks"
                bundle.addfile(member)
                continue
            member.size = len(payload)
            member.mode = 0o755 if name == "betterleaks" else 0o644
            bundle.addfile(member, io.BytesIO(payload))
    return path


def _zip(
    path: pathlib.Path,
    *,
    entries: list[tuple[str, bytes, str]] | None = None,
) -> pathlib.Path:
    rows = entries or [
        (runtime.LICENSE_MEMBER, _LICENSE, "file"),
        (runtime.README_MEMBER, _README, "file"),
        ("betterleaks.exe", _BINARY, "file"),
    ]
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload, kind in rows:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (
                (stat.S_IFLNK | 0o777) if kind == "symlink" else (stat.S_IFREG | 0o755)
            ) << 16
            bundle.writestr(info, payload)
    return path


def _install_fixture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str = "tar.gz",
) -> tuple[pathlib.Path, runtime.BetterleaksArtifact, pathlib.Path]:
    if kind == "zip":
        archive = _zip(tmp_path / "fixture.zip")
        member = "betterleaks.exe"
    else:
        archive = _tar(tmp_path / "fixture.tar.gz")
        member = "betterleaks"
    artifact = _artifact(archive, kind=kind, binary_member=member)
    target = tmp_path / runtime.STANDALONE_DIRNAME
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)
    state = runtime.install_betterleaks(
        build_output=target,
        archive_path=archive,
        artifact=artifact,
    )
    assert state.ready
    return archive, artifact, target


def test_pin_matrix_is_immutable_complete_and_bound_to_official_v181_release():
    assert set(runtime.BETTERLEAKS_ARTIFACTS) == {
        "darwin-arm64",
        "darwin-x64",
        "linux-arm64",
        "linux-x64",
        "windows-arm64",
        "windows-x64",
    }
    with pytest.raises(TypeError):
        runtime.BETTERLEAKS_ARTIFACTS["extra"] = next(  # type: ignore[index]
            iter(runtime.BETTERLEAKS_ARTIFACTS.values())
        )
    expected_checksums = {
        "darwin-arm64": "8e80f33b5f2a7426b390347b9fd466033723cb94b6bdffa7572632e2eaec964e",
        "darwin-x64": "6abc37df76f881cffae406aa2cec72bea6e6ae64b4e771b3ed21b4aac472ed10",
        "linux-arm64": "bbb578b12a2f65d7082ab436abf37724232bc71d8a078e3c41336574420f1b48",
        "linux-x64": "efa407244e1ea8e35f582b8a42becdeac08bdead04f68eb752adda722d583c2a",
        "windows-arm64": "aa12beb9ce1f6a911da91e1d0d8a72d7e68daf56a52a53f930038fd81f10f0ba",
        "windows-x64": "94310d028285a1bcce7f160bc19eb62f87de6460c95bfd4319151ef5b501ed3f",
    }
    for key, artifact in runtime.BETTERLEAKS_ARTIFACTS.items():
        artifact.validate()
        assert artifact.version == "1.8.1"
        assert artifact.release_commit == "5eab48332cc48565864514e3bc6de89df091a7c4"
        assert artifact.archive_url == (
            "https://github.com/betterleaks/betterleaks/releases/download/v1.8.1/"
            + artifact.archive_name
        )
        assert artifact.archive_sha256 == expected_checksums[key]
        assert artifact.license_sha256 == runtime.BETTERLEAKS_LICENSE_SHA256
        assert artifact.expected_members == {
            artifact.binary_member,
            runtime.LICENSE_MEMBER,
            runtime.README_MEMBER,
        }


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-x64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Linux", "amd64", "linux-x64"),
        ("Windows", "ARM64", "windows-arm64"),
        ("Windows", "AMD64", "windows-x64"),
        ("FreeBSD", "x86_64", ""),
    ],
)
def test_platform_key_maps_only_the_six_supported_targets(system, machine, expected):
    assert runtime.platform_key(system=system, machine=machine) == expected


@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_installer_extracts_only_binary_license_and_writes_bound_metadata(
    tmp_path, monkeypatch, kind
):
    _, artifact, target = _install_fixture(tmp_path, monkeypatch, kind=kind)

    binary_name = "betterleaks.exe" if kind == "zip" else "betterleaks"
    assert sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()) == [
        "LICENSE",
        f"bin/{binary_name}",
        runtime.INSTALL_METADATA_FILENAME,
    ]
    assert not (target / runtime.README_MEMBER).exists()
    metadata = json.loads((target / runtime.INSTALL_METADATA_FILENAME).read_text())
    assert metadata["version"] == runtime.BETTERLEAKS_VERSION
    assert metadata["release_commit"] == runtime.BETTERLEAKS_RELEASE_COMMIT
    assert metadata["archive_sha256"] == artifact.archive_sha256
    assert metadata["binary_sha256"] == _sha(_BINARY)
    assert metadata["license_sha256"] == _sha(_LICENSE)
    assert metadata["install_kind"] == "build"


def test_existing_valid_target_wins_without_reading_a_new_archive(tmp_path, monkeypatch):
    _, artifact, target = _install_fixture(tmp_path, monkeypatch)
    missing = tmp_path / "missing.tar.gz"

    state = runtime.install_betterleaks(
        build_output=target,
        archive_path=missing,
        artifact=artifact,
    )

    assert state.ready
    assert pathlib.Path(state.binary_path).read_bytes() == _BINARY


def test_managed_resolver_uses_exact_versioned_location_and_never_path(
    tmp_path, monkeypatch
):
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    monkeypatch.setattr(runtime, "current_artifact", lambda: artifact)
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)
    random_path_binary = tmp_path / "path-bin" / "betterleaks"
    random_path_binary.parent.mkdir()
    random_path_binary.write_bytes(b"random")
    monkeypatch.setenv("PATH", str(random_path_binary.parent))
    monkeypatch.setattr(
        runtime,
        "_download_archive",
        lambda *_args, **_kwargs: pytest.fail("read-only resolution must never download"),
    )

    missing = runtime.resolve_betterleaks(
        data_root=tmp_path / "data", bundle_bases=[]
    )
    assert missing.status == "missing"

    installed = runtime.install_betterleaks(
        data_root=tmp_path / "data",
        archive_path=archive,
        artifact=artifact,
    )
    resolved = runtime.resolve_betterleaks(
        data_root=tmp_path / "data", bundle_bases=[]
    )
    assert resolved.ready and resolved.source == "managed"
    assert pathlib.Path(resolved.binary_path) == pathlib.Path(installed.binary_path)
    assert pathlib.Path(resolved.binary_path).is_relative_to(
        tmp_path / "data" / "state" / "betterleaks" / "v1.8.1"
    )
    assert pathlib.Path(resolved.binary_path) != random_path_binary


def test_bundled_runtime_precedes_managed_runtime(tmp_path, monkeypatch):
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    monkeypatch.setattr(runtime, "current_artifact", lambda: artifact)
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)
    bundle_root = tmp_path / "bundle"
    bundled = runtime.install_betterleaks(
        build_output=bundle_root / runtime.STANDALONE_DIRNAME,
        archive_path=archive,
        artifact=artifact,
    )
    runtime.install_betterleaks(
        data_root=tmp_path / "data", archive_path=archive, artifact=artifact
    )

    resolved = runtime.resolve_betterleaks(
        data_root=tmp_path / "data", bundle_bases=[bundle_root]
    )

    assert resolved.ready and resolved.source == "bundled"
    assert resolved.binary_path == bundled.binary_path


def test_managed_resolver_rejects_binary_digest_drift(tmp_path, monkeypatch):
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    monkeypatch.setattr(runtime, "current_artifact", lambda: artifact)
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)
    installed = runtime.install_betterleaks(
        data_root=tmp_path / "data", archive_path=archive, artifact=artifact
    )
    pathlib.Path(installed.binary_path).write_bytes(b"changed executable\n")

    resolved = runtime.resolve_betterleaks(
        data_root=tmp_path / "data", bundle_bases=[]
    )

    assert resolved.status == "corrupt"
    assert resolved.reason_code == "runtime_binary_mismatch"


def test_packaged_macos_accepts_post_staging_codesign_digest_change(
    tmp_path, monkeypatch
):
    _, artifact, target = _install_fixture(tmp_path, monkeypatch)
    pathlib.Path(target / "bin" / "betterleaks").write_bytes(b"signed Mach-O bytes\n")
    monkeypatch.setattr(runtime.platform_layer, "IS_MACOS", True)

    with pytest.raises(runtime.BetterleaksRuntimeError) as staging_error:
        runtime._validate_runtime_root(target, artifact, source="bundled")
    assert staging_error.value.code == "runtime_binary_mismatch"

    state = runtime._validate_runtime_root(
        target,
        artifact,
        source="bundled",
        allow_signed_macos_digest=True,
    )

    assert state.ready


@pytest.mark.parametrize(
    "bad_name",
    ["../escape", "/absolute", "C:/drive", "folder\\member"],
)
def test_tar_extraction_rejects_traversal_and_absolute_members(
    tmp_path, bad_name
):
    archive = _tar(
        tmp_path / "unsafe.tar.gz",
        entries=[
            (runtime.LICENSE_MEMBER, _LICENSE, "file"),
            (runtime.README_MEMBER, _README, "file"),
            ("betterleaks", _BINARY, "file"),
            (bad_name, b"bad", "file"),
        ],
    )
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    with pytest.raises(runtime.BetterleaksRuntimeError, match="unsafe") as excinfo:
        runtime._extract_archive(archive, tmp_path / "out", artifact)
    assert excinfo.value.code == "archive_unsafe"


@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_extraction_rejects_links(tmp_path, kind):
    binary = "betterleaks.exe" if kind == "zip" else "betterleaks"
    entries = [
        (runtime.LICENSE_MEMBER, _LICENSE, "file"),
        (runtime.README_MEMBER, _README, "file"),
        (binary, b"target", "symlink"),
    ]
    archive = (
        _zip(tmp_path / "link.zip", entries=entries)
        if kind == "zip"
        else _tar(tmp_path / "link.tar.gz", entries=entries)
    )
    artifact = _artifact(archive, kind=kind, binary_member=binary)
    with pytest.raises(runtime.BetterleaksRuntimeError) as excinfo:
        runtime._extract_archive(archive, tmp_path / "out", artifact)
    assert excinfo.value.code == "archive_unsafe"


@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_extraction_rejects_duplicate_members(tmp_path, kind):
    binary = "betterleaks.exe" if kind == "zip" else "betterleaks"
    entries = [
        (runtime.LICENSE_MEMBER, _LICENSE, "file"),
        (runtime.README_MEMBER, _README, "file"),
        (binary, _BINARY, "file"),
        (runtime.LICENSE_MEMBER, _LICENSE, "file"),
    ]
    if kind == "zip":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = _zip(tmp_path / "duplicate.zip", entries=entries)
    else:
        archive = _tar(tmp_path / "duplicate.tar.gz", entries=entries)
    artifact = _artifact(archive, kind=kind, binary_member=binary)
    with pytest.raises(runtime.BetterleaksRuntimeError) as excinfo:
        runtime._extract_archive(archive, tmp_path / "out", artifact)
    assert excinfo.value.code == "archive_duplicate"


def test_extraction_rejects_wrong_member_set(tmp_path):
    archive = _tar(
        tmp_path / "wrong.tar.gz",
        entries=[
            (runtime.LICENSE_MEMBER, _LICENSE, "file"),
            (runtime.README_MEMBER, _README, "file"),
            ("wrong-binary", _BINARY, "file"),
        ],
    )
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    with pytest.raises(runtime.BetterleaksRuntimeError) as excinfo:
        runtime._extract_archive(archive, tmp_path / "out", artifact)
    assert excinfo.value.code == "archive_members_mismatch"


def test_archive_checksum_is_verified_before_extraction(tmp_path):
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    archive.write_bytes(archive.read_bytes() + b"drift")

    with pytest.raises(runtime.BetterleaksRuntimeError) as excinfo:
        runtime.verify_archive(archive, artifact)

    assert excinfo.value.code == "archive_checksum_mismatch"


def test_download_path_reuses_a_verified_durable_cache_without_network(
    tmp_path, monkeypatch
):
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("verified cache must win before network"),
    )

    assert runtime._download_archive(artifact, archive) == archive


def test_installer_rejects_license_and_version_mismatch_without_replacing_target(
    tmp_path, monkeypatch
):
    target = tmp_path / runtime.STANDALONE_DIRNAME
    target.mkdir()
    sentinel = target / "old-tree"
    sentinel.write_bytes(b"keep")
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    wrong_license = runtime.BetterleaksArtifact(
        **{**artifact.__dict__, "license_sha256": "0" * 64}
    )
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)

    with pytest.raises(runtime.BetterleaksRuntimeError) as license_error:
        runtime.install_betterleaks(
            build_output=target,
            archive_path=archive,
            artifact=wrong_license,
        )
    assert license_error.value.code == "runtime_license_mismatch"
    assert sentinel.read_bytes() == b"keep"

    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: "9.9.9")
    with pytest.raises(runtime.BetterleaksRuntimeError) as version_error:
        runtime.install_betterleaks(
            build_output=target,
            archive_path=archive,
            artifact=artifact,
        )
    assert version_error.value.code == "runtime_version_mismatch"
    assert sentinel.read_bytes() == b"keep"


def test_mid_promotion_failure_restores_prior_target(tmp_path, monkeypatch):
    target = tmp_path / runtime.STANDALONE_DIRNAME
    target.mkdir()
    sentinel = target / "old-tree"
    sentinel.write_bytes(b"keep")
    archive = _tar(tmp_path / "fixture.tar.gz")
    artifact = _artifact(archive, kind="tar.gz", binary_member="betterleaks")
    monkeypatch.setattr(runtime, "_probe_version", lambda _binary: runtime.BETTERLEAKS_VERSION)
    real_replace = os.replace

    def fail_staging_promote(source, destination):
        source_path = pathlib.Path(source)
        if source_path.name.startswith(f".{target.name}.tmp.") and pathlib.Path(destination) == target:
            raise OSError("simulated promote failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_staging_promote)
    with pytest.raises(runtime.BetterleaksRuntimeError) as excinfo:
        runtime.install_betterleaks(
            build_output=target,
            archive_path=archive,
            artifact=artifact,
        )

    assert excinfo.value.code == "runtime_install_failed"
    assert sentinel.read_bytes() == b"keep"
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".betterleaks")] == []


def test_module_cli_exposes_only_explicit_install_command():
    parser = runtime.build_parser()
    args = parser.parse_args(
        [
            "install",
            "--build-output",
            "betterleaks-standalone",
            "--cache-dir",
            "/tmp/cache",
        ]
    )
    assert args.command == "install"
    assert args.build_output == pathlib.Path("betterleaks-standalone")
    assert runtime.BETTERLEAKS_INSTALL_COMMAND == (
        "python -m ouroboros.betterleaks_runtime install"
    )
