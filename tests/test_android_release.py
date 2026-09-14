"""Android source delivery and release-proof boundaries, without device access."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from ouroboros.tools.release_sync import (
    DESKTOP_DOWNLOAD_IDS,
    RELEASE_ASSET_TEMPLATES,
    VERSION_CARRIER_SPANS,
    release_asset_download_url,
    release_asset_name,
    version_carrier_desyncs,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("android_release", REPO / "scripts/build_android_release.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_android_assets_share_the_release_registry_without_changing_desktop_carriers():
    assert release_asset_name("android-arm64", "7.1.0-rc.1") == "Ouroboros-7.1.0-rc.1-android-arm64.tar.gz"
    assert release_asset_name("android-apk", "7.1.0-rc.1") == "Ouroboros-7.1.0-rc.1-android.apk"
    assert len(RELEASE_ASSET_TEMPLATES) == 9
    assert len(DESKTOP_DOWNLOAD_IDS) == 7
    assert not any("android" in span.carrier_id for span in VERSION_CARRIER_SPANS)
    references = "".join(
        f"[download-{key}]: {release_asset_download_url(key, '7.1.0')}\n"
        for key in DESKTOP_DOWNLOAD_IDS
    )
    assert version_carrier_desyncs("7.1.0", download_readme_text=references) == []


def _archive_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    output = tmp_path / "dist"
    for path in (source, stage, output):
        path.mkdir()
    for name in builder.REQUIRED_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first-party source\n", encoding="utf-8")
    # A local credential/cache is not part of git's tracked source inventory.
    (source / "android/private.keystore").write_bytes(b"never publish")
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "run", lambda *_args, **_kwargs: "\0".join(builder.REQUIRED_FILES))

    def bundle(_root, bundle_path, manifest_path, **kwargs):
        assert kwargs["source_branch"] == "ouroboros"
        bundle_path.write_bytes(b"git bundle fixture")
        manifest_path.write_text(json.dumps({
            "source_sha": "a" * 40, "release_tag": "v7.1.0",
            "bundle_sha256": builder.RELEASE["sha256_file"](bundle_path),
        }))

    monkeypatch.setitem(builder.BUNDLE, "build_bundle", bundle)
    apk = output / "Ouroboros-7.1.0-android.apk"
    apk.write_bytes(b"verified APK fixture")
    identity = {"packageName": "ai.ouroboros.android", "versionName": "7.1.0",
                "versionCode": 12, "signerSha256": "b" * 64}
    args = argparse.Namespace(out=output, apk=apk, source_branch="ouroboros",
                              commit="a" * 40, tag="v7.1.0", version_code=12)
    archive = builder.create_archive(args, stage, "7.1.0", identity)
    return args, archive, identity


def test_archive_carries_only_tracked_source_and_binds_every_delivered_byte(tmp_path, monkeypatch):
    args, archive, identity = _archive_fixture(tmp_path, monkeypatch)
    with tarfile.open(archive) as handle:
        names = handle.getnames()
        assert not any("private.keystore" in name for name in names)
        assert "Ouroboros-Android/docs/ANDROID_RECOVERY.md" in names
        manifest = json.load(handle.extractfile("Ouroboros-Android/android_release_manifest.json"))
    assert set(manifest["files"]) == set(builder.REQUIRED_FILES) | {"repo.bundle", "repo_bundle_manifest.json"}
    assert manifest["sourceCommit"] == args.commit
    assert manifest["referenceApk"] == {"name": args.apk.name, **builder.file_record(args.apk), **identity}


@pytest.mark.parametrize("tamper", [False, True])
def test_final_archive_inspection_checks_bundle_and_installer_before_receipt(tmp_path, monkeypatch, tamper):
    args, archive, identity = _archive_fixture(tmp_path, monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return args.commit + "\n" if argv[:2] == ["git", "rev-parse"] else ""

    monkeypatch.setattr(builder, "run", run)
    monkeypatch.setattr(builder, "apk_identity", lambda *_args: identity)
    if tamper:
        args.apk.write_bytes(b"different APK after packaging")
        with pytest.raises(ValueError, match="Final Android APK differs"):
            builder.inspect_archive(args, archive, tmp_path / "extracted")
    else:
        assert builder.inspect_archive(args, archive, tmp_path / "extracted") == identity
        assert any(argv[:2] == ["git", "clone"] for argv in calls)
        assert any(
            Path(argv[1]).name == "install.py"
            and Path(argv[1]).parent.name == "android"
            and argv[-1] == "--help"
            for argv in calls
        )


def test_release_builder_refuses_missing_key_before_source_or_compiler_work(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.sys, "argv", [
        "build_android_release.py", "--sdk", str(tmp_path), "--java-home", str(tmp_path),
        "--keystore", str(tmp_path / "absent.keystore"),
        "--keystore-pass-file", str(tmp_path / "absent.password"),
        "--version-code", "12", "--source-branch", "ouroboros",
        "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"),
    ])
    monkeypatch.setattr(builder, "run", lambda *_args, **_kwargs: pytest.fail("no build may run without signing input"))
    with pytest.raises(SystemExit) as exc:
        builder.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_android_ci_is_fork_safe_and_required_for_publication():
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    validation = workflow.split("  android-test:", 1)[1].split("  android-build:", 1)[0]
    release = workflow.split("  android-build:", 1)[1].split("  release-preflight:", 1)[0]
    publication = workflow.split("  release:\n", 1)[1]
    assert "secrets." not in validation
    assert "--create-development-key" in validation
    assert "python -m pytest android/tests tests/test_android_release.py" in validation
    assert "--create-development-key" not in release
    assert "if: startsWith(github.ref, 'refs/tags/v')" in release
    assert "needs: [android-test, android-emulator-smoke, release-preflight]" in release
    assert "publisher signing credentials are required" in release
    assert "android-build" in next(line for line in publication.splitlines() if "needs:" in line)
    assert "secrets." not in release.split("- name: Generate Android source", 1)[1]
    for suffix in ("android-arm64.tar.gz", "android.apk"):
        assert f"release-artifacts/Ouroboros-*-{suffix}" in publication
    assert "draft: true" in publication


def test_android_ci_has_representative_emulator_matrix_without_calling_it_device_qualification():
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    smoke = workflow.split("  android-emulator-smoke:", 1)[1].split("  # The publisher key", 1)[0]
    assert "api-level: [26, 29, 30, 33, 36]" in smoke
    assert "adb install -r" in smoke
    assert "dumpsys package ai.ouroboros.android" in smoke
    assert "SELinux" not in smoke


def test_android_smoke_requirements_do_not_claim_a_device_was_tested():
    checks = builder.RELEASE["REQUIRED_SMOKE_CHECKS"]
    assert checks["android-arm64"] == {"embedded_repo_bundle", "android_source_manifest", "usb_installer_help"}
    assert checks["android-apk"] == {"apk_signature", "apk_package_version"}


def test_default_host_build_uses_the_shared_root_asset_path():
    source = (REPO / "android/host/build.py").read_text(encoding="utf-8")
    assert 'source.parents[1] / "assets" / "icon_1024.png"' in source
    assert (REPO / "assets/icon_1024.png").is_file()
