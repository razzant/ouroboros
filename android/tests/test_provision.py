"""Pinned downloads, archive boundaries, and browser installer ownership."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

import pytest


PROVISION = Path(__file__).resolve().parents[1] / "provision"
SPEC = importlib.util.spec_from_file_location("archive_identity", PROVISION / "archive_identity.py")
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def archive(path, *, mtime=0, text=b"source", link=None):
    with tarfile.open(path, "w:gz") as handle:
        member = tarfile.TarInfo("file.txt")
        member.mode = 0o644
        member.mtime = mtime
        member.size = len(text)
        handle.addfile(member, io.BytesIO(text))
        if link:
            member = tarfile.TarInfo("header.h")
            member.type = tarfile.SYMTYPE
            member.linkname = link
            handle.addfile(member)
    return path


def runtime(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "archive_identity", identity)
    spec = importlib.util.spec_from_file_location("android_runtime_provision_test", PROVISION / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.APP = tmp_path / "installation"
    module.APP.mkdir()
    module.PROVISION = module.APP / "provision"
    module.PROVISION.mkdir()
    return module


def test_gitiles_identity_ignores_container_dates_but_detects_source_changes(tmp_path):
    first = archive(tmp_path / "first.tar.gz", mtime=100)
    second = archive(tmp_path / "second.tar.gz", mtime=200)
    changed = archive(tmp_path / "changed.tar.gz", mtime=200, text=b"different")
    expected = identity.archive_content_sha256(first)
    assert identity.file_sha256(first) != identity.file_sha256(second)
    assert identity.archive_content_sha256(second) == expected
    assert identity.archive_content_sha256(changed) != expected
    assert identity.verify_artifact(second, {"archive_content_sha256": expected})


def test_incomplete_cached_tar_is_retryable_not_a_verified_dependency(tmp_path):
    path = tmp_path / "download.partial"
    path.write_bytes(b"unfinished")
    assert not identity.verify_artifact(path, {"archive_content_sha256": "0" * 64})


def test_aosp_sibling_links_work_inside_the_installation_root(monkeypatch, tmp_path):
    module = runtime(monkeypatch, tmp_path)
    source = archive(tmp_path / "headers.tar.gz", link="../libcutils/header.h")
    destination = module.APP / "toolchain-source/vendor/core/include"
    module.extract(source, destination)
    assert (destination / "file.txt").read_bytes() == b"source"
    assert (destination / "header.h").is_symlink()
    assert (destination / "header.h").readlink() == Path("../libcutils/header.h")


def test_dependency_cannot_extract_a_link_outside_installation(monkeypatch, tmp_path):
    module = runtime(monkeypatch, tmp_path)
    source = archive(tmp_path / "headers.tar.gz", link="../../../../outside")
    with pytest.raises(tarfile.FilterError):
        module.extract(source, module.APP / "vendor/include")


@pytest.mark.serial
def test_browser_cli_reads_exact_cached_input_and_local_server_is_closed(monkeypatch, tmp_path):
    module = runtime(monkeypatch, tmp_path)
    cached = tmp_path / "browser.zip"
    cached.write_bytes(b"verified browser fixture")
    urls = []

    def run(*argv, env):
        assert argv == ("python", "-m", "playwright", "install", "chromium")
        url = env["PLAYWRIGHT_DOWNLOAD_HOST"] + "/builds/chromium/1234/browser.zip"
        with urllib.request.urlopen(url, timeout=2) as result:
            assert result.read() == cached.read_bytes()
        urls.append(url)

    monkeypatch.setattr(module, "run", run)
    module.install_browser("python", [{"cached_file": str(cached), "playwright_path": "builds/chromium/1234/browser.zip"}])
    with pytest.raises(OSError):
        urllib.request.urlopen(urls[0], timeout=1)
    assert list(module.PROVISION.iterdir()) == []


def test_manifest_has_no_private_paths_and_node_matches_the_common_pin():
    pins = json.loads((PROVISION / "artifacts.json").read_text())
    common = json.loads((PROVISION.parents[1] / "ouroboros/claudexor_runtime_pin.json").read_text())
    node = next(item for item in pins if item["name"] == "node")
    expected = common["release"]["node_artifacts"]["linux-arm64"]
    assert node["url"] == expected["archive_url"]
    assert node["sha256"] == expected["sha256"]
    assert node["size_bytes"] == expected["size_bytes"]
    for item in pins:
        assert item["url"].startswith("https://")
        assert "/Users/" not in json.dumps(item)
        assert bool(item.get("sha256")) != bool(item.get("archive_content_sha256"))


@pytest.mark.parametrize('lookup', ['absent', 'installed', 'error'])
def test_first_key_creation_requires_confirmed_package_absence(monkeypatch, tmp_path, lookup):
    import subprocess
    module = runtime(monkeypatch, tmp_path)
    repo = module.APP / 'repo'
    tools = module.APP / 'tools'
    calls = []

    def installed_package(path):
        assert path == tools / 'android-exec'
        if lookup == 'error':
            raise subprocess.CalledProcessError(1, ['pm', 'path'], '', 'service unavailable')
        return None if lookup == 'absent' else {'path': '/data/app/base.apk', 'version_code': 1}

    def load(path):
        assert path == str(repo / 'android/bootstrap/update-host')
        return {'installed_package': installed_package}

    def run(*argv, **kwargs):
        calls.append(argv)
        key = Path(argv[argv.index('-keystore') + 1])
        key.write_bytes(b'first installation signing identity')

    monkeypatch.setattr(module.runpy, 'run_path', load)
    monkeypatch.setattr(module, 'run', run)
    if lookup == 'absent':
        module.ensure_personal_signing(repo, tools)
        assert (module.APP / 'signing/host.keystore').read_bytes() == b'first installation signing identity'
        assert (module.APP / 'signing/host-password').is_file()
        assert len(calls) == 1
    else:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            module.ensure_personal_signing(repo, tools)
        assert not calls and not (module.APP / 'signing/host.keystore').exists()
