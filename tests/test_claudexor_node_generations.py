"""Update preparation publishes new Node/npm generations, never edits live roots."""
from __future__ import annotations

import io
import json
import pathlib
import shutil
import tarfile
from dataclasses import replace

import pytest

from ouroboros import claudexor_runtime as runtime
from tests.test_claudexor_runtime_delivery import NODE_VERSION, _archive, _data_plane, _node_artifacts, _pin


def _fixture(tmp_path, monkeypatch):
    _data_plane(monkeypatch, tmp_path)
    archive = tmp_path / f"node-v{NODE_VERSION}-linux-x64.tar.gz"
    prefix = f"node-v{NODE_VERSION}-linux-x64"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in [("bin/node", b"exact node"),
                              ("lib/node_modules/npm/bin/npm-cli.js", b"npm fixture"),
                              ("lib/node_modules/npm/package.json", b'{"version":"11.13.0"}')]:
            member = tarfile.TarInfo(f"{prefix}/{name}")
            member.size, member.mode = len(content), 0o755
            bundle.addfile(member, io.BytesIO(content))
    pin = _pin(_archive(tmp_path / "closure.tar.gz"),
               node_artifacts=_node_artifacts(exact_key="linux-x64", exact_archive=archive))
    import ouroboros.platform_layer as platform
    monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [])
    monkeypatch.setattr(platform, "node_distribution_platform", lambda: "linux-x64")
    monkeypatch.setattr(platform, "embedded_node_candidates", lambda root: [
        pathlib.Path(root) / "node-standalone/bin/node"])
    monkeypatch.setattr(platform, "probe_node_version", lambda path: (
        NODE_VERSION if pathlib.Path(path).is_file() else ""))
    fetches = []

    def fetch(artifact, destination):
        fetches.append(str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(archive, destination)
        return runtime.verify_node_archive(destination, artifact)

    monkeypatch.setattr(runtime, "fetch_node_archive", fetch)
    return pin, fetches


def _legacy(pin, *, schema=2):
    root = runtime.managed_node_dir(pin, "linux-x64")
    node = root / "node-standalone/bin/node"
    node.parent.mkdir(parents=True)
    node.write_bytes(b"legacy exact node")
    npm = runtime.ClaudexorRuntimeManager._managed_npm_cli(node)
    npm.parent.mkdir(parents=True)
    npm.write_bytes(b"legacy npm")
    artifact = pin.node_artifacts["linux-x64"]
    meta = {"schema_version": schema, "version": pin.node_version, "platform": "linux-x64",
            "archive_url": artifact.archive_url, "archive_sha256": artifact.sha256,
            "archive_size": artifact.size_bytes, "archive_executable": artifact.executable,
            "archive_npm_cli": runtime.ClaudexorRuntimeManager._archive_npm_cli(artifact, "linux-x64")}
    (root / "managed-node.json").write_text(json.dumps(meta), encoding="utf-8")
    return root, node, npm


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("legacy_schema", [None, 1, 2])
def test_ready_full_runtime_beats_npm_less_bundle_for_both_readers(tmp_path, monkeypatch, legacy_schema):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    import ouroboros.platform_layer as platform
    bundle = tmp_path / "bundle"
    bundled_node = bundle / "node-standalone/bin/node"
    bundled_node.parent.mkdir(parents=True)
    bundled_node.write_bytes(b"bundled exact node without npm")
    monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [bundle])
    manager = runtime.ClaudexorRuntimeManager(pin)
    # The preserved reader must not depend on the next engine's Node pin.
    preserved = runtime.ClaudexorRuntimeManager(replace(pin, node_version="25.0.0"))
    legacy = _legacy(pin, schema=legacy_schema) if legacy_schema else None
    old_bytes = _bytes(legacy[0]) if legacy else None
    before = str(legacy[1]) if legacy_schema else str(bundled_node)  # a managed Node-only one outlives the app
    assert manager._resolve_node(pin) == before
    assert preserved._resolve_preserved_node(NODE_VERSION) == str(pathlib.Path(before).resolve())
    assert not manager._managed_npm_cli(bundled_node).exists()

    prepared = pathlib.Path(manager.prepare_node_toolchain())
    assert prepared != bundled_node
    assert manager._resolve_node(pin) == str(prepared)
    assert preserved._resolve_preserved_node(NODE_VERSION) == str(prepared.resolve())
    assert manager._managed_npm_cli(prepared).is_file()
    assert len(fetches) == (0 if legacy_schema == 2 else 1)
    assert bundled_node.read_bytes() == b"bundled exact node without npm"
    assert not manager._managed_npm_cli(bundled_node).exists()
    if legacy:
        assert _bytes(legacy[0]) == old_bytes


@pytest.mark.parametrize("damage", ["schema1", "invalid_json", "missing_npm", "other_pin"])
def test_prepare_preserves_every_legacy_byte_and_selects_an_exact_new_generation(tmp_path, monkeypatch, damage):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    root, old_node, npm = _legacy(pin, schema=1 if damage == "schema1" else 2)
    if damage == "invalid_json":
        (root / "managed-node.json").write_text("{broken", encoding="utf-8")
    elif damage == "missing_npm":
        npm.unlink()
    elif damage == "other_pin":
        metadata = json.loads((root / "managed-node.json").read_text())
        metadata["archive_sha256"] = "f" * 64
        (root / "managed-node.json").write_text(json.dumps(metadata))
    before = _bytes(root)
    manager = runtime.ClaudexorRuntimeManager(pin)
    node = pathlib.Path(manager.prepare_node_toolchain())
    assert node != old_node and node.parents[2].parent == root.parent
    assert node.parents[2].name.startswith(root.name + ".g-")
    assert _bytes(root) == before  # No rename, additive repair, deletion or metadata rewrite.
    assert manager._resolve_node(pin) == str(node)
    assert runtime.ClaudexorRuntimeManager(None)._resolve_preserved_node(NODE_VERSION) == str(node.resolve())
    assert manager.prepare_node_toolchain() == str(node)
    assert len(fetches) == 1
    assert not runtime.managed_runtime_dir(pin).exists(), "Node preparation does not reinstall the engine"


def test_ready_exact_legacy_toolchain_is_reused_without_download_or_generation(tmp_path, monkeypatch):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    root, node, _ = _legacy(pin)
    manager = runtime.ClaudexorRuntimeManager(pin)
    before = _bytes(root)
    assert manager.prepare_node_toolchain() == str(node)
    assert manager._resolve_node(pin) == str(node)
    assert not fetches and _bytes(root) == before
    assert not list(root.parent.glob(root.name + ".g-*"))


def test_corrupt_generation_is_retained_when_a_new_complete_one_is_published(tmp_path, monkeypatch):
    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    old = pathlib.Path(manager.prepare_node_toolchain())
    runtime.ClaudexorRuntimeManager._managed_npm_cli(old).unlink()
    before = _bytes(old.parents[2])
    new = pathlib.Path(manager.prepare_node_toolchain())
    assert new != old and _bytes(old.parents[2]) == before
    assert manager._resolve_node(pin) == str(new)
    # A later release pin does not hide the preserved engine's old Node generation.
    later = runtime.ClaudexorRuntimeManager(replace(pin, node_version="25.0.0"))
    assert later._resolve_preserved_node(NODE_VERSION) == str(new.resolve())


def test_pin_change_preserves_prior_generation_and_selects_matching_archive(tmp_path, monkeypatch):
    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    old = pathlib.Path(manager.prepare_node_toolchain())
    before = _bytes(old.parents[2])
    artifact = replace(pin.node_artifacts["linux-x64"], archive_url="https://mirror.example.test/node.tar.gz")
    new_pin = replace(pin, node_artifacts={**pin.node_artifacts, "linux-x64": artifact})
    other = runtime.ClaudexorRuntimeManager(new_pin)
    new = pathlib.Path(other.prepare_node_toolchain())
    assert new != old and _bytes(old.parents[2]) == before
    assert manager._resolve_node(pin) == str(old)
    assert other._resolve_node(new_pin) == str(new)


def test_failed_promotion_keeps_all_roots_and_removes_only_owned_staging(tmp_path, monkeypatch):
    pin, _ = _fixture(tmp_path, monkeypatch)
    root, _, _ = _legacy(pin, schema=1)
    before = _bytes(root)
    monkeypatch.setattr(runtime.ClaudexorRuntimeManager, "_extract_node_archive", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(runtime.ClaudexorRuntimeError("fixture_failure", "negative"))))
    with pytest.raises(runtime.ClaudexorRuntimeError, match="negative"):
        runtime.ClaudexorRuntimeManager(pin).prepare_node_toolchain()
    assert _bytes(root) == before
    assert not list(root.parent.glob(".tmp-*")) and not list(root.parent.glob(root.name + ".g-*"))
