"""The engine daemon runs on a Node this installation keeps (``ouroboros/claudexor_node_seed.py``).

The daemon outlives the app and spawns its children from its own Node. A managed exact Node, complete or Node-only,
therefore wins over the bundled one. With none, the foreground install owner copies the bundled bytes once into a
``.g-*`` generation recorded as ``bundle_seed``, and the daemon runs on that copy; a failed copy leaves the bundled
Node in use, logged. The read-only resolver never copies, and a seed never serves the npm toolchain.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib

import pytest

from ouroboros import claudexor_runtime as runtime
from tests.test_claudexor_node_generations import _fixture, _legacy
from tests.test_claudexor_runtime_delivery import NODE_VERSION


def _installed(tmp_path, monkeypatch, *, legacy_schema=None):
    """An installed engine, the app's exact Node in its bundle, and optionally a managed Node of that version."""
    pin, fetches = _fixture(tmp_path, monkeypatch)
    import ouroboros.platform_layer as platform

    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    bundled = tmp_path / "bundle" / "node-standalone/bin/node"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled exact node without npm")
    monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [tmp_path / "bundle"])
    engine = runtime.managed_runtime_dir(pin)
    (engine / "dist").mkdir(parents=True)
    (engine / pin.entrypoint).write_text("fixture\n", encoding="utf-8")
    (engine / "managed-runtime.json").write_text(json.dumps({
        "schema_version": 1, "version": pin.version, "build_sha": pin.build_sha,
        "protocol_major": pin.protocol_major, "archive_sha256": pin.sha256, "archive_size": pin.size_bytes,
        "node_version": pin.node_version, "entrypoint": pin.entrypoint, "archive_source": "cache"}), encoding="utf-8")
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(manager, "_probe", lambda _command, _pin: NODE_VERSION)
    legacy = _legacy(pin, schema=legacy_schema) if legacy_schema else None
    return pin, manager, bundled, legacy, fetches


def _generations():
    return sorted((runtime.managed_runtime_root() / "node").glob(f"{NODE_VERSION}-linux-x64.g-*"))


def test_a_managed_node_only_runtime_beats_the_bundled_node_for_the_daemon(tmp_path, monkeypatch):
    _pin, manager, _bundled, (_root, node, _npm), fetches = _installed(tmp_path, monkeypatch, legacy_schema=1)
    assert manager.resolve_command()[0] == str(node), "the daemon ran on the app's Node and lost it with the app"
    assert manager.ensure()[0] == str(node)
    assert runtime.ClaudexorRuntimeManager(None)._resolve_preserved_node(NODE_VERSION) == str(node.resolve())
    assert not _generations() and not fetches, "a kept Node is neither copied nor downloaded"


def test_a_complete_managed_runtime_still_wins_and_nothing_is_copied(tmp_path, monkeypatch):
    pin, manager, _bundled, (_root, node, _npm), fetches = _installed(tmp_path, monkeypatch, legacy_schema=2)
    assert manager.ensure()[0] == str(node) == manager._resolve_node_toolchain(pin)
    assert not _generations() and not fetches


def test_without_a_managed_node_the_install_owner_copies_the_bundled_one_once(tmp_path, monkeypatch):
    pin, manager, bundled, _none, fetches = _installed(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_APP_VERSION", "6.1.0")
    assert manager.resolve_command()[0] == str(bundled) and not _generations(), "the read-only resolver never copies"

    command = manager.ensure()
    seeded = pathlib.Path(command[0])
    assert [seeded.parents[2]] == _generations(), "the daemon was left on the app's Node"
    assert not seeded.is_symlink() and seeded.read_bytes() == bundled.read_bytes()
    assert json.loads((seeded.parents[2] / "managed-node.json").read_text(encoding="utf-8")) == {
        "schema_version": 1, "version": NODE_VERSION, "platform": "linux-x64", "source": "bundle_seed",
        "binary_sha256": hashlib.sha256(bundled.read_bytes()).hexdigest(), "app_version": "6.1.0"}
    assert manager.ensure() == command == manager.resolve_command()
    assert len(_generations()) == 1 and not fetches, "copied once, never downloaded"
    assert manager._resolve_node_toolchain(pin) == "", "a seed never serves npm"
    bundled.unlink()  # the app moves away; the daemon's Node stays
    assert manager.resolve_command() == command
    assert runtime.ClaudexorRuntimeManager(None)._resolve_preserved_node(NODE_VERSION) == str(seeded.resolve())


@pytest.mark.parametrize("tamper", ["archive_field", "no_digest", "other_source"])
def test_the_reader_takes_a_seed_record_only_as_what_it_is(tmp_path, monkeypatch, tamper):
    pin, manager, _bundled, _none, _fetches = _installed(tmp_path, monkeypatch)
    record_file = pathlib.Path(manager.ensure()[0]).parents[2] / "managed-node.json"
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if tamper == "archive_field":
        record["archive_sha256"] = pin.node_artifacts["linux-x64"].sha256
    elif tamper == "no_digest":
        del record["binary_sha256"]
    else:
        record["source"] = "download"
    record_file.write_text(json.dumps(record), encoding="utf-8")
    assert manager._resolve_node(pin, bundled=False) == ""


def test_a_failed_copy_leaves_the_daemon_on_the_bundled_node_and_says_so(tmp_path, monkeypatch, caplog):
    _pin, manager, bundled, _none, fetches = _installed(tmp_path, monkeypatch)
    import ouroboros.platform_layer as platform

    nodes = runtime.managed_runtime_root() / "node"
    monkeypatch.setattr(platform, "probe_node_version", lambda path: (  # the copy does not run
        NODE_VERSION if pathlib.Path(path).is_file() and nodes not in pathlib.Path(path).parents else ""))
    with caplog.at_level(logging.WARNING, logger="ouroboros.claudexor_node_seed"):
        assert manager.ensure()[0] == str(bundled)
    assert not list(nodes.glob("*")) and not fetches, "no generation, no staging, no download"
    assert f"runs on the bundled Node {bundled}" in caplog.text


def test_the_seed_module_never_imports_the_runtime_back():
    """claudexor_runtime imports the seed module when it loads; the seed receives its record's file name instead."""
    import ast

    from ouroboros import claudexor_node_seed

    tree = ast.parse(pathlib.Path(claudexor_node_seed.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "ouroboros.claudexor_runtime" not in imported
