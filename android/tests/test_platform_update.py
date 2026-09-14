"""Current source recipes share initial provisioning without recreating identity."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "provision"


@pytest.fixture
def platform(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("platform_update_fixture", SOURCE / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.APP = tmp_path / "installation"
    module.PROVISION = module.APP / "provision"
    module.PROVISION.mkdir(parents=True)
    module.OS_RELEASE = tmp_path / "os-release"
    module.OS_RELEASE.write_text('VERSION_CODENAME=noble\n')
    module.APT_SOURCES = tmp_path / "ubuntu.sources"
    repo = module.APP / "repo"
    shutil.copytree(SOURCE, repo / "android/provision", ignore=shutil.ignore_patterns('__pycache__'))
    (repo / "ouroboros").mkdir()
    for name in ("ouroboros/claudexor_runtime_pin.json", "requirements-runtime.lock"):
        shutil.copy2(SOURCE.parents[1] / name, repo / name)
    signing = module.APP / "signing"
    signing.mkdir()
    (signing / "host.keystore").write_bytes(b"permanent private fixture")
    (signing / "host-password").write_text("persistent fixture password")
    calls = []
    monkeypatch.setattr(module, "download", lambda pin, cache: calls.append(("download", pin["name"])) or cache / pin["name"])
    monkeypatch.setattr(module, "run", lambda *argv, **kwargs: calls.append(("run", argv)))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: b"fixture-package\t1\tarm64\n")
    for name in ("install_node", "install_sdk", "install_browser"):
        monkeypatch.setattr(module, name, lambda *args, _name=name, **kwargs: calls.append((_name, args)))
    receipt = module.APP / "android-sdk/installation.json"
    receipt.parent.mkdir()
    for name in ('platforms/android-36/android.jar', 'build-tools/36.0.0/lib/d8.jar',
                 'build-tools/36.0.0/lib/apksigner.jar', 'build-tools/36.0.0/aapt2', 'build-tools/36.0.0/zipalign'):
        output = receipt.parent / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'existing fixture tool')
    receipt.write_text(json.dumps({"platform_inputs": module.platform_inputs(repo)}))
    return module, repo, receipt, calls


def change_pin(repo, name, suffix):
    path = repo / "android/provision/artifacts.json"
    pins = json.loads(path.read_text())
    pin = next(item for item in pins if item['name'] == name)
    pin['url'] += suffix
    path.write_text(json.dumps(pins))


def test_unchanged_recipe_skips_all_dependency_operations(platform):
    module, repo, receipt, calls = platform
    before = receipt.read_bytes()
    assert module.ensure_platform(repo) is False
    assert not calls and receipt.read_bytes() == before


def test_sdk_pin_change_prepares_from_current_repo_and_keeps_identity(platform):
    module, repo, receipt, calls = platform
    old = json.loads(receipt.read_text())['platform_inputs']
    key = (module.APP / 'signing/host.keystore').read_bytes()
    change_pin(repo, 'android-platform', '?new-pinned-release')
    assert not module.platform_current(repo)[0]
    assert module.ensure_platform(repo) is True
    assert [name for name, _ in calls if name.startswith('install_')] == ['install_sdk']
    downloaded = [name for action, name in calls if action == 'download']
    assert 'android-platform' in downloaded and 'ubuntu-base' not in downloaded
    assert 'node' not in downloaded and 'playwright-chromium' not in downloaded
    installed = json.loads(receipt.read_text())
    assert installed['platform_inputs']['sdk'] != old['sdk']
    assert installed['platform_inputs']['node'] == old['node']
    assert 'platform_preparing' not in installed and module.platform_current(repo)[0]
    assert all(pin['name'] != 'ubuntu-base' for pin in installed['artifacts'])
    assert (module.APP / 'signing/host.keystore').read_bytes() == key
    calls.clear()
    assert not module.ensure_platform(repo) and not calls


@pytest.mark.parametrize('path,group', [('packages.sh', 'packages'), ('ubuntu.sources', 'packages'),
                                     ('patches/new.patch', 'aapt'), ('runtime.py', 'sdk')])
def test_tracked_recipe_and_patch_changes_are_inputs(platform, path, group):
    module, repo, receipt, calls = platform
    old = module.platform_inputs(repo)
    file = repo / 'android/provision' / path
    file.parent.mkdir(exist_ok=True)
    file.write_text((file.read_text() if file.exists() else '') + '\n# changed recipe\n')
    assert module.platform_inputs(repo)[group] != old[group]


def test_package_recipe_update_uses_current_snapshot_without_rootfs_overlay(platform):
    module, repo, receipt, calls = platform
    sources = repo / 'android/provision/ubuntu.sources'
    sources.write_text(sources.read_text().replace('20260911T000000Z', '20260912T000000Z'))
    assert module.ensure_platform(repo)
    assert module.APT_SOURCES.read_bytes() == sources.read_bytes()
    assert calls == [('run', ('/bin/sh', repo / 'android/provision/packages.sh'))]


def test_ubuntu_base_is_an_immutable_install_seed(platform):
    module, repo, receipt, calls = platform
    change_pin(repo, 'ubuntu-base', '?next-seed')
    assert module.platform_current(repo)[0]
    assert not module.ensure_platform(repo) and not calls


def test_interrupted_sdk_update_remains_stale_after_git_source_rollback(platform, monkeypatch):
    module, repo, receipt, calls = platform
    manifest = repo / 'android/provision/artifacts.json'
    original = manifest.read_bytes()
    change_pin(repo, 'android-platform', '?new-pinned-release')

    def fail(*args, **kwargs):
        raise RuntimeError('compiler was interrupted')

    monkeypatch.setattr(module, 'install_sdk', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.ensure_platform(repo)
    assert json.loads(receipt.read_text())['platform_preparing'] == ['sdk']
    manifest.write_bytes(original)
    assert not module.platform_current(repo)[0]
    rebuilt = []
    monkeypatch.setattr(module, 'install_sdk', lambda *args, **kwargs: rebuilt.append(args))
    assert module.ensure_platform(repo) and len(rebuilt) == 1
    assert module.platform_current(repo)[0]


def test_source_drift_during_preparation_does_not_advance_completed_receipt(platform, monkeypatch):
    module, repo, receipt, calls = platform
    previous = json.loads(receipt.read_text())['platform_inputs']
    change_pin(repo, 'android-platform', '?candidate')
    monkeypatch.setattr(module, 'install_sdk', lambda *a, **kw: change_pin(repo, 'android-platform', '?concurrent-edit'))
    with pytest.raises(RuntimeError, match='source changed'):
        module.ensure_platform(repo)
    assert json.loads(receipt.read_text())['platform_inputs'] == previous
    assert not module.platform_current(repo)[0]


def test_common_node_pin_changes_refresh_node_group(platform):
    module, repo, receipt, calls = platform
    file = repo / 'ouroboros/claudexor_runtime_pin.json'
    pin = json.loads(file.read_text())
    pin['release']['node_artifacts']['linux-arm64']['archive_url'] += '?next'
    file.write_text(json.dumps(pin))
    assert module.ensure_platform(repo)
    assert [name for name, _ in calls if name.startswith('install_')] == ['install_node']
    node_args = next(args for name, args in calls if name == 'install_node')
    assert node_args[-1][0]['url'].endswith('?next')


def test_browser_pin_change_uses_pinned_installer_and_cache(platform):
    module, repo, receipt, calls = platform
    change_pin(repo, 'playwright-chromium', '?next')
    assert module.ensure_platform(repo)
    assert [name for name, _ in calls if name.startswith('install_')] == ['install_browser']
    assert ('run', (module.APP / 'venv/bin/python', '-m', 'playwright', 'install-deps', 'chromium')) in calls


def test_first_install_uses_same_recipe_without_key_generation(platform):
    module, repo, receipt, calls = platform
    receipt.unlink()
    assert module.ensure_platform(repo)
    assert [name for name, _ in calls if name.startswith('install_')] == ['install_node', 'install_sdk', 'install_browser']
    assert module.platform_current(repo)[0]
    assert (module.APP / 'signing/host.keystore').read_bytes() == b'permanent private fixture'


def test_common_node_manager_replaces_owned_executable_links(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('node_link_fixture', SOURCE / 'runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.APP = tmp_path
    tools = tmp_path / 'tools'
    tools.mkdir()
    for version in ('old', 'new'):
        folder = tmp_path / version
        folder.mkdir()
        for name in ('node', 'npm', 'npx'):
            (folder / name).write_text(version)
    for name in ('node', 'npm', 'npx'):
        (tools / name).symlink_to(tmp_path / 'old' / name)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: json.dumps([str(tmp_path / 'new/node')]).encode())
    monkeypatch.setattr(module, 'run', lambda *a: None)
    module.install_node(tmp_path / 'repo', 'python', tools, [])
    assert all((tools / name).read_text() == 'new' for name in ('node', 'npm', 'npx'))


def test_actual_sdk_jar_replacement_skips_native_source_compile(platform, tmp_path, monkeypatch):
    module, repo, receipt, calls = platform
    # Execute the real SDK extraction function with two ordinary upstream-shaped
    # jars. Native AAPT is unchanged and must not compile on a Java-only update.
    spec = importlib.util.spec_from_file_location('sdk_extract_fixture', SOURCE / 'runtime.py')
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    real.APP = module.APP
    monkeypatch.setattr(real, 'run', lambda *a, **kw: pytest.fail('unexpected native source build'))
    source = tmp_path / 'new-platform.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('android-36/android.jar', b'new verified platform fixture')
    tools = tmp_path / 'new-build-tools.zip'
    with zipfile.ZipFile(tools, 'w') as archive:
        archive.writestr('android-36/lib/d8.jar', b'verified d8 fixture')
        archive.writestr('android-36/lib/apksigner.jar', b'verified signer fixture')
    before = (receipt.parent / 'build-tools/36.0.0/aapt2').read_bytes()
    monkeypatch.setattr(module, 'install_sdk', real.install_sdk)
    monkeypatch.setattr(module, 'download', lambda pin, cache: source if pin['name'] == 'android-platform' else tools)
    change_pin(repo, 'android-platform', '?new')
    assert module.ensure_platform(repo)
    actual = receipt.parent / 'platforms/android-36/android.jar'
    assert actual.read_bytes() == b'new verified platform fixture'
    assert json.loads(receipt.read_text())['outputs']['platforms/android-36/android.jar'] == module.sha(actual)
    assert (receipt.parent / 'build-tools/36.0.0/aapt2').read_bytes() == before
