"""Engine CLI delivery publishes new Node/npm generations, never edits live roots."""
from __future__ import annotations

import io
import json
import os
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

    selected = pathlib.Path(manager._ensure_node_toolchain(pin))
    assert selected != bundled_node
    assert manager._resolve_node(pin) == str(selected)
    assert preserved._resolve_preserved_node(NODE_VERSION) == str(selected.resolve())
    assert manager._managed_npm_cli(selected).is_file()
    assert len(fetches) == (0 if legacy_schema == 2 else 1)
    assert bundled_node.read_bytes() == b"bundled exact node without npm"
    assert not manager._managed_npm_cli(bundled_node).exists()
    if legacy:
        assert _bytes(legacy[0]) == old_bytes


@pytest.mark.parametrize("damage", ["schema1", "invalid_json", "other_pin"])
def test_delivery_preserves_every_legacy_byte_and_selects_an_exact_new_generation(tmp_path, monkeypatch, damage):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    root, old_node, npm = _legacy(pin, schema=1 if damage == "schema1" else 2)
    if damage == "invalid_json":
        (root / "managed-node.json").write_text("{broken", encoding="utf-8")
    elif damage == "missing_npm":
        npm.unlink()
    elif damage == "other_pin":
        metadata = json.loads((root / "managed-node.json").read_text(encoding="utf-8"))
        metadata["archive_sha256"] = "f" * 64
        (root / "managed-node.json").write_text(json.dumps(metadata), encoding="utf-8")
    before = _bytes(root)
    manager = runtime.ClaudexorRuntimeManager(pin)
    node = pathlib.Path(manager._ensure_node_toolchain(pin))
    assert node != old_node and node.parents[2].parent == root.parent
    assert node.parents[2].name.startswith(root.name + ".g-")
    assert _bytes(root) == before  # No rename, additive repair, deletion or metadata rewrite.
    assert manager._resolve_node(pin) == str(node)
    assert runtime.ClaudexorRuntimeManager(None)._resolve_preserved_node(NODE_VERSION) == str(node.resolve())
    assert manager._ensure_node_toolchain(pin) == str(node)
    assert len(fetches) == 1


def test_ready_exact_legacy_toolchain_is_reused_without_download_or_generation(tmp_path, monkeypatch):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    root, node, _ = _legacy(pin)
    manager = runtime.ClaudexorRuntimeManager(pin)
    before = _bytes(root)
    assert manager._ensure_node_toolchain(pin) == str(node)
    assert manager._resolve_node(pin) == str(node)
    assert not fetches and _bytes(root) == before
    assert not list(root.parent.glob(root.name + ".g-*"))


def test_corrupt_generation_is_retained_when_a_new_complete_one_is_published(tmp_path, monkeypatch):
    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    old = pathlib.Path(manager._ensure_node_toolchain(pin))
    runtime.ClaudexorRuntimeManager._managed_npm_cli(old).unlink()
    before = _bytes(old.parents[2])
    new = pathlib.Path(manager._ensure_node_toolchain(pin))
    assert new != old and _bytes(old.parents[2]) == before
    assert manager._resolve_node(pin) == str(new)
    # A later release pin does not hide the preserved engine's old Node generation.
    later = runtime.ClaudexorRuntimeManager(replace(pin, node_version="25.0.0"))
    assert later._resolve_preserved_node(NODE_VERSION) == str(new.resolve())


def test_pin_change_preserves_prior_generation_and_selects_matching_archive(tmp_path, monkeypatch):
    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    old = pathlib.Path(manager._ensure_node_toolchain(pin))
    before = _bytes(old.parents[2])
    artifact = replace(pin.node_artifacts["linux-x64"], archive_url="https://mirror.example.test/node.tar.gz")
    new_pin = replace(pin, node_artifacts={**pin.node_artifacts, "linux-x64": artifact})
    other = runtime.ClaudexorRuntimeManager(new_pin)
    new = pathlib.Path(other._ensure_node_toolchain(new_pin))
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
        runtime.ClaudexorRuntimeManager(pin)._ensure_node_toolchain(pin)
    assert _bytes(root) == before
    assert not list(root.parent.glob(".tmp-*")) and not list(root.parent.glob(root.name + ".g-*"))


def test_reviewed_node_prepares_node_only_and_a_readonly_lookup_never_installs(tmp_path, monkeypatch):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    with pytest.raises(runtime.ClaudexorRuntimeError) as missing:
        manager.reviewed_node()
    assert missing.value.code == "runtime_node_unavailable" and fetches == []
    node = manager.reviewed_node(allow_install=True)
    assert manager.reviewed_node() == node and len(fetches) == 1  # the generation is reused
    assert pathlib.Path(node).is_relative_to(runtime.managed_runtime_root() / "node")
    assert not runtime.managed_runtime_dir(pin).exists()  # Node only: no engine closure, no daemon


def test_reviewed_node_waits_for_no_engine_install_and_accepts_a_peer_node_only_install(tmp_path, monkeypatch):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    import ouroboros.platform_layer as platform
    manager = runtime.ClaudexorRuntimeManager(pin)
    _root, node, _npm = _legacy(pin, schema=1)  # an exact Node-only generation, no npm toolchain record
    with manager._lock:  # an engine install holding the manager for minutes does not block a lookup
        assert manager.reviewed_node(allow_install=True) == str(node)
    # Another process holds the install lock: an exact Node-only result is enough for a Node-only intent.
    monkeypatch.setattr(platform, "acquire_exclusive_file_lock", lambda *args, **kwargs: None)
    manager._install(pin, node_only=True)
    with pytest.raises(runtime.ClaudexorRuntimeError) as busy:
        manager._install(pin, require_npm=True, node_only=True)  # npm still needs its own toolchain
    assert busy.value.code == "runtime_install_in_progress" and fetches == []


def _bytes_of(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_a_failed_preparation_names_its_real_destination_and_recovery_publishes_exactly_there(tmp_path, monkeypatch):
    pin, fetches = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    # An earlier, unusable generation of the same archive (metadata of another platform) and a legacy root stay.
    old = runtime.managed_node_dir(pin, "linux-x64").parent / (runtime.managed_node_dir(pin, "linux-x64").name
                                                               + f".g-{pin.node_artifacts['linux-x64'].sha256[:12]}-00000000")
    (old / "node-standalone/bin").mkdir(parents=True)
    (old / "node-standalone/bin/node").write_bytes(b"older generation still loaded by a live process")
    (old / "managed-node.json").write_text(json.dumps({"schema_version": 1, "version": "0.0.1"}), encoding="utf-8")
    kept = _bytes_of(old)
    planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
    assert pathlib.Path(planned).parts[-4].endswith("-00000001")  # the first free name, never the occupied one
    real_fetch = runtime.fetch_node_archive

    def offline(artifact, destination):
        fetches.append("offline")
        raise runtime.ClaudexorRuntimeError("runtime_node_download_failed", "managed Node download failed: offline")

    monkeypatch.setattr(runtime, "fetch_node_archive", offline)
    with pytest.raises(runtime.ClaudexorRuntimeError) as failed:
        manager.reviewed_node(allow_install=True)
    assert failed.value.code == "runtime_node_download_failed" and failed.value.planned_node == planned
    assert not os.path.lexists(pathlib.Path(planned).parents[2])  # nothing published, nothing marked installed
    with pytest.raises(runtime.ClaudexorRuntimeError) as lookup:
        manager.reviewed_node()  # a lookup names the same destination
    assert lookup.value.code == "runtime_node_unavailable" and lookup.value.planned_node == planned
    monkeypatch.setattr(runtime, "fetch_node_archive", real_fetch)
    assert manager.reviewed_node(allow_install=True) == planned  # recovery publishes exactly where it was named
    assert pathlib.Path(planned).read_bytes() == b"exact node" and _bytes_of(old) == kept


def test_concurrent_preparations_share_the_install_lock_and_one_download(tmp_path, monkeypatch):
    import threading

    pin, fetches = _fixture(tmp_path, monkeypatch)
    planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
    real_fetch, entered, release = runtime.fetch_node_archive, threading.Event(), threading.Event()

    def slow(artifact, destination):
        entered.set()
        release.wait(5)
        return real_fetch(artifact, destination)

    monkeypatch.setattr(runtime, "fetch_node_archive", slow)
    results = []

    def prepare():
        results.append(runtime.ClaudexorRuntimeManager(pin).reviewed_node(allow_install=True))

    first = threading.Thread(target=prepare)
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=prepare)  # another process's manager: only the install lock is shared
    second.start()
    threading.Timer(0.5, release.set).start()
    first.join(10)
    second.join(10)
    assert results == [planned, planned] and len(fetches) == 1


@pytest.mark.parametrize("peer_after_lookup", [1, 2])
def test_a_peer_publishing_during_a_missed_lookup_is_what_the_child_receives(tmp_path, monkeypatch, peer_after_lookup):
    from ouroboros import node_runtime

    pin, fetches = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(node_runtime.PLAYWRIGHT_NODE_ENV, raising=False)
    reader = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: reader)
    real_resolve, lookups, published = reader._resolve_node, [], []

    def resolve(selected_pin, **kwargs):  # another process's install publishes right after this lookup misses
        result = real_resolve(selected_pin, **kwargs)
        lookups.append(result)
        if len(lookups) == peer_after_lookup and not result:
            published.append(runtime.ClaudexorRuntimeManager(pin).reviewed_node(allow_install=True))
        return result

    monkeypatch.setattr(reader, "_resolve_node", resolve)
    child = {}
    reason = node_runtime.forward_playwright_node(child)
    assert len(published) == 1 and len(fetches) == 1  # the lookup itself never installed
    assert child[node_runtime.PLAYWRIGHT_NODE_ENV] == published[0] and pathlib.Path(published[0]).is_file()
    if peer_after_lookup == 1:  # published before the miss was reported: the child gets the ready Node
        assert reason == "" and lookups == ["", published[0]]
    else:  # published after the last lookup: the named destination is exactly the peer's generation
        assert "becomes valid" in reason and lookups == ["", ""]
    assert reader.reviewed_node() == published[0]


def _child_envs(tmp_path, monkeypatch):
    """The six environment builders that forward Playwright's driver Node: skill, shell, companion, both service
    baselines and a stdio MCP server."""
    from types import SimpleNamespace

    from ouroboros import extension_companion, mcp_client
    from ouroboros.tools import skill_exec
    from ouroboros.tools.shell_process import _shell_env_for_cwd
    from ouroboros.workspace_executor import _executor_service_env, service_env

    captured = {}
    monkeypatch.setattr(mcp_client, "_stdio_with_diagnostics", lambda params, cfg, launch: captured.update(
        env=dict(params.env or {})))
    monkeypatch.setattr("ouroboros.execution_domain.attached_launch", lambda *a, **k: None)
    mcp_client._transport_factory(mcp_client.MCPServerConfig(
        id="fixture", name="fixture", enabled=True, transport="stdio", url="", command="python3", args=[],
        auth_header="", auth_token="", allowed_tools=[], env={}))
    return [skill_exec._scrub_env([], tmp_path / "skill-state", "fixture-skill"),
            _shell_env_for_cwd(SimpleNamespace(repo_dir=str(tmp_path), task_metadata={}), tmp_path),
            extension_companion._companion_base_env(), service_env(), _executor_service_env(), captured["env"]]


def _forbid_processes(monkeypatch):
    """From here on any spawn or ``node --version`` fails the test; returns the attempts it saw."""
    import subprocess

    import ouroboros.platform_layer as platform

    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(repr(args[1:2] or args[:1]))
        raise AssertionError(f"a child environment started a process: {attempts[-1]}")

    monkeypatch.setattr(subprocess.Popen, "_execute_child", refuse)  # every spawn, subclasses included
    monkeypatch.setattr(platform, "probe_node_version", refuse)
    return attempts


def _tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []


@pytest.mark.parametrize("state", ["empty", "exact_bundle", "downloaded", "seeded"])
def test_building_a_child_environment_runs_nothing_writes_nothing_and_names_the_managed_node(
        tmp_path, monkeypatch, state):
    from ouroboros import node_runtime
    import ouroboros.platform_layer as platform

    pin, fetches = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(node_runtime.PLAYWRIGHT_NODE_ENV, raising=False)
    manager = runtime.ClaudexorRuntimeManager(pin)  # a fresh manager: nothing memoized by an earlier lookup
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    if state in ("exact_bundle", "seeded"):
        bundled = tmp_path / "bundle/node-standalone/bin/node"
        bundled.parent.mkdir(parents=True)
        bundled.write_bytes(b"bundled exact node")
        monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [tmp_path / "bundle"])
    planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
    ready = manager.reviewed_node(allow_install=True) if state in ("downloaded", "seeded") else ""
    assert len(fetches) == (state == "downloaded")  # the exact bundle is copied, never downloaded
    if ready:
        assert ready == planned  # a copy and a download both publish where children were already pointed
    state_root = runtime.managed_runtime_root()
    before = _tree(state_root)
    attempts = _forbid_processes(monkeypatch)
    envs = _child_envs(tmp_path, monkeypatch)
    assert [env[node_runtime.PLAYWRIGHT_NODE_ENV] for env in envs] == [ready or planned] * 6  # never the bundle
    assert attempts == [] and _tree(state_root) == before  # no probe, no spawn, no lock, no directory
    assert node_runtime.PLAYWRIGHT_NODE_ENV not in os.environ
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: pytest.fail("an explicit Node needs no lookup"))
    child = {node_runtime.PLAYWRIGHT_NODE_ENV: "/operator/node"}
    assert node_runtime.forward_playwright_node(child) == "" and child == {node_runtime.PLAYWRIGHT_NODE_ENV:
                                                                            "/operator/node"}


def test_a_record_means_prepared_while_daemon_and_cli_lookups_keep_their_probes(tmp_path, monkeypatch):
    import ouroboros.platform_layer as platform

    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    node = manager.reviewed_node(allow_install=True)
    probes = []
    monkeypatch.setattr(platform, "probe_node_version", lambda path: probes.append(path) or "")  # it no longer runs
    assert manager.reviewed_node(prepared=True) == node and probes == []  # the record states preparation, not health
    assert manager._resolve_node(pin) == "" and probes  # the daemon/CLI resolver still runs what it selects
    with pytest.raises(runtime.ClaudexorRuntimeError):
        manager.reviewed_node()  # the probed lookup of host consumers still refuses a Node that does not run


def _exact_bundle(tmp_path, monkeypatch, version=None):
    import ouroboros.platform_layer as platform

    bundled = tmp_path / "bundle/node-standalone/bin/node"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled node")
    bundled.chmod(0o755)
    monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [tmp_path / "bundle"])
    if version:  # an older bundle kept after an update: it runs, as another version
        monkeypatch.setattr(platform, "probe_node_version", lambda path: (
            version if pathlib.Path(path) == bundled else NODE_VERSION if pathlib.Path(path).is_file() else ""))
    return bundled


def _offline(monkeypatch, fetches):
    def offline(artifact, destination):
        fetches.append("offline")
        raise runtime.ClaudexorRuntimeError("runtime_node_download_failed", "managed Node download failed: offline")

    monkeypatch.setattr(runtime, "fetch_node_archive", offline)


@pytest.mark.parametrize("bundle", ["exact", "older"])
def test_the_offline_boot_step_copies_an_exact_bundle_to_the_issued_path_and_never_claims_an_older_one(
        tmp_path, monkeypatch, bundle):
    from ouroboros import node_runtime, server_boot

    pin, fetches = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(node_runtime.PLAYWRIGHT_NODE_ENV, raising=False)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    bundled = _exact_bundle(tmp_path, monkeypatch, version=None if bundle == "exact" else "24.0.0")
    _offline(monkeypatch, fetches)
    monkeypatch.delenv(node_runtime.SELECTED_DRIVER_NODE_ENV, raising=False)
    issued = {}
    node_runtime.forward_playwright_node(issued)  # a child built before the boot step ran
    planned = issued[node_runtime.PLAYWRIGHT_NODE_ENV]
    assert planned == runtime.ClaudexorRuntimeManager.planned_node(pin) and not pathlib.Path(planned).exists()
    assert server_boot.prepare_pinned_node(tmp_path) is None  # local only (no thread under pytest), nothing copied
    assert not pathlib.Path(planned).exists() and fetches == []
    # This process's own Playwright (an in-process extension): the exact bundle at once, else the planned path.
    assert os.environ[node_runtime.PLAYWRIGHT_NODE_ENV] == (str(bundled) if bundle == "exact" else planned)
    if bundle == "exact":
        assert server_boot.finish_pinned_node(tmp_path) == planned  # no network, no engine: the bundle's bytes
        assert fetches == [] and pathlib.Path(planned).read_bytes() == bundled.read_bytes()
        record = json.loads((pathlib.Path(planned).parents[2] / "managed-node.json").read_text(encoding="utf-8"))
        assert record["source"] == "bundle_seed" and not any(key.startswith("archive_") for key in record)
        assert manager._resolve_node_toolchain(pin) == ""  # a seed never serves npm
        assert os.environ[node_runtime.PLAYWRIGHT_NODE_ENV] == planned  # now the managed copy, which outlives the app
    else:
        assert server_boot.finish_pinned_node(tmp_path) == ""  # the older shipped Node is neither copied nor ready
        assert fetches == ["offline"] and not pathlib.Path(planned).exists()
        assert os.environ[node_runtime.PLAYWRIGHT_NODE_ENV] == planned  # never the older bundle
        child = {"PATH": "/usr/bin"}
        assert "becomes valid" in node_runtime.forward_playwright_node(child)
        assert child == {"PATH": "/usr/bin", node_runtime.PLAYWRIGHT_NODE_ENV: planned,
                         node_runtime.SELECTED_DRIVER_NODE_ENV: planned}  # generic PATH untouched
    assert not runtime.managed_runtime_dir(pin).exists()  # Node only, never the engine


@pytest.mark.parametrize("driver", ["automatic", "operator"])
def test_a_server_booted_inside_a_boundary_keeps_the_pinned_node_it_carried_in(tmp_path, monkeypatch, driver):
    """R4 (Fable): the boot step cleared the selection a safe_test or candidate boundary carried in before the lookup
    that reads it, so that server selected a planned path in its scratch store and handed it to every later child; an
    operator's explicit driver dropped the carried Node too. The lookup now runs first and an explicit value rewrites
    nothing; preparation is decided by the store's fact, never by the driver's value. Plan rev12: the carried
    executable is its own fact (``CARRIED_NODE_ENV``); an operator's driver is never marked automatic beside it."""
    import ouroboros.config as config
    from ouroboros import node_runtime

    pin, fetches = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    carried = manager.reviewed_node(allow_install=True)  # prepared by the installation the boundary isolates
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "scratch")  # the boundary's own data root: nothing prepared
    names = (node_runtime.PLAYWRIGHT_NODE_ENV, node_runtime.SELECTED_DRIVER_NODE_ENV, node_runtime.CARRIED_NODE_ENV)
    for name in names:
        monkeypatch.setenv(name, "")
    boundary = {} if driver == "automatic" else {node_runtime.PLAYWRIGHT_NODE_ENV: "/operator/node"}
    assert node_runtime.carry_pinned_node(boundary, tmp_path / "data") == carried
    for name, value in boundary.items():
        monkeypatch.setenv(name, value)  # what the boundary's command, a server, starts with
    driver_pair = ({node_runtime.PLAYWRIGHT_NODE_ENV: "/operator/node"} if driver == "operator" else
                   {node_runtime.PLAYWRIGHT_NODE_ENV: carried, node_runtime.SELECTED_DRIVER_NODE_ENV: carried})
    expected = {**driver_pair, node_runtime.CARRIED_NODE_ENV: carried}
    assert boundary == expected
    assert node_runtime.select_local_playwright_node() is False  # prepared, through what it carried: no copy, no download
    assert {name: os.environ[name] for name in expected} == expected
    assert os.environ[node_runtime.SELECTED_DRIVER_NODE_ENV] == ("" if driver == "operator" else carried)
    child = {}
    assert node_runtime.forward_playwright_node(child) == ""
    assert child == driver_pair  # a child gets the driver; the carried fact stays with the environment it was given
    assert not (tmp_path / "scratch").exists() and len(fetches) == 1  # the one download was the isolated installation's
    monkeypatch.setenv(node_runtime.CARRIED_NODE_ENV, "")  # negative control: nothing carried, nothing prepared
    if driver == "automatic":
        monkeypatch.setenv(node_runtime.PLAYWRIGHT_NODE_ENV, "")
        monkeypatch.setenv(node_runtime.SELECTED_DRIVER_NODE_ENV, "")
    assert node_runtime.select_local_playwright_node() is True
    planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
    assert pathlib.Path(planned).is_relative_to(tmp_path / "scratch") and not pathlib.Path(planned).exists()
    assert os.environ[node_runtime.PLAYWRIGHT_NODE_ENV] == ("/operator/node" if driver == "operator" else planned)


def test_a_failed_copy_or_a_busy_install_lock_never_counts_a_bundled_node_as_prepared(tmp_path, monkeypatch):
    import ouroboros.platform_layer as platform
    from ouroboros import server_boot

    pin, fetches = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    bundled = _exact_bundle(tmp_path, monkeypatch)
    nodes = runtime.managed_runtime_root() / "node"
    real_lock = platform.acquire_exclusive_file_lock
    monkeypatch.setattr(platform, "acquire_exclusive_file_lock", lambda *args, **kwargs: None)  # a peer installs
    with pytest.raises(runtime.ClaudexorRuntimeError) as busy:
        manager.reviewed_node(allow_install=True)
    assert busy.value.code == "runtime_install_in_progress"
    assert server_boot.finish_pinned_node(tmp_path) == ""  # the boot's preparation says so in the log, not ready
    monkeypatch.setattr(platform, "acquire_exclusive_file_lock", real_lock)
    monkeypatch.setattr(platform, "probe_node_version", lambda path: (  # the copy does not run
        NODE_VERSION if pathlib.Path(path).is_file() and nodes not in pathlib.Path(path).parents else ""))
    with pytest.raises(runtime.ClaudexorRuntimeError, match="not prepared yet") as failed:
        manager.reviewed_node(allow_install=True)  # the daemon may stay on the bundle; this is not a success
    assert failed.value.planned_node == runtime.ClaudexorRuntimeManager.planned_node(pin)
    assert server_boot.finish_pinned_node(tmp_path) == ""
    assert not [p for p in nodes.glob("*") if p.is_dir()] and fetches == [] and bundled.is_file()


def test_an_older_kept_bundle_yields_only_a_pinned_copy_that_runs_while_naming_stays_unrun(tmp_path, monkeypatch):
    """Host consumers of ``resolve_bundled_node`` keep an execution probe (one per file and process, the health memo):
    past a kept older bundle they get the pinned managed copy only while it runs as the pin, never unverified bytes.
    Only an environment that names a Node (``discovered_host_node``) takes the copy by its record, running nothing."""
    from ouroboros import node_runtime
    import ouroboros.platform_layer as platform

    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    node = manager.reviewed_node(allow_install=True)
    _exact_bundle(tmp_path, monkeypatch)
    probes = []
    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", lambda path, timeout_sec=10: (
        probes.append(path) or ((NODE_VERSION, "") if path == node else ("24.0.0", ""))))  # the kept bundle is older
    assert platform.resolve_bundled_node() == node and node in probes  # the copy ran as the pin
    node_runtime._NODE_HEALTH_MEMO.clear()
    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", lambda path, timeout_sec=10: (
        probes.append(path) or (("", "signal:SIGKILL") if path == node else ("24.0.0", ""))))  # it no longer runs
    monkeypatch.setattr(platform, "probe_node_version", lambda path: probes.append(path) or "")  # nor in the owner's
    probes.clear()
    assert platform.resolve_bundled_node() is None and node in probes  # refused after running it, not by its record
    probes.clear()
    assert node_runtime.discovered_host_node() == node and probes == []


def _first_node_on(path):
    """``shutil.which("node", path=path)`` with POSIX semantics on every OS: the fixture models linux-x64, whose Node
    is the extensionless ``bin/node`` that Windows ``which`` (PATHEXT names only, the current directory first) never
    finds. The first PATH entry holding that executable file, or None."""
    for entry in path.split(os.pathsep):
        candidate = os.path.join(entry, "node")
        if entry and os.path.isfile(candidate) and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return candidate
    return None


@pytest.mark.parametrize("bundle,prepared,lane,named", [
    (None, False, None, None),
    ("exact", False, "bundle", None),  # naming cannot tell this copy's version without running it: none is named
    ("exact", True, "bundle", "managed"),  # same version: naming cannot run the bundle; the lane passes its own
    ("older", False, None, None),  # Carbon R2: never the older copy as found; the lane refuses it too
    ("older", True, "managed", "managed")])
def test_the_gate_environment_names_the_lane_node_without_running_it_and_the_lane_passes_the_one_it_runs(
        tmp_path, monkeypatch, bundle, prepared, lane, named):
    from ouroboros import node_runtime, preflight_node, preflight_runner

    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    empty = tmp_path / "no-node-on-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    managed = manager.reviewed_node(allow_install=True) if prepared else None  # a download: no bundle yet
    bundled = _exact_bundle(tmp_path, monkeypatch, version="24.0.0" if bundle == "older" else None) if bundle else None
    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", lambda path, timeout_sec=10: (
        ("24.0.0" if bundle == "older" and path == str(bundled) else NODE_VERSION), ""))
    nodes = {None: None, "bundle": str(bundled), "managed": managed}
    selected = preflight_node.resolve_node()  # the web lane's execution choice, probes included
    assert selected == nodes[lane]
    attempts = _forbid_processes(monkeypatch)
    env = preflight_runner._preflight_env(tmp_path / "gate", tmp_path / "gate" / "repo", create=False, workers=1)
    assert _first_node_on(env["PATH"]) == nodes[named]  # planning, or passes no lane ran: named, unrun
    lane_env = preflight_runner._preflight_env(tmp_path / "gate", tmp_path / "gate" / "repo", create=False, workers=1,
                                               node=selected)  # the lane's children and the passes after it
    assert _first_node_on(lane_env["PATH"]) == (selected or nodes[named])
    assert attempts == [] and not (tmp_path / "gate").exists()


@pytest.mark.skipif(os.name == "nt", reason="the generations' executables are POSIX shell scripts")
@pytest.mark.parametrize("shipped", ["older", "none"])
def test_a_recorded_generation_that_does_not_run_as_the_pin_falls_back_to_the_next_healthy_one(
        tmp_path, monkeypatch, shipped):
    """Parent review of 1550a9fe: the record lookup names the newest generation whose record is valid; when that one
    does not run as the pin, the probed lookup of the owner (``reviewed_node``) tries the other generations before
    any shipped copy. Real records, real executables, real probes: the newest generation runs as another version,
    the older one as the pin; a shipped copy is older or absent."""
    import ouroboros.platform_layer as platform
    from ouroboros import node_runtime

    pin, _ = _fixture(tmp_path, monkeypatch)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    for key in (node_runtime.CARRIED_NODE_ENV, node_runtime.SELECTED_DRIVER_NODE_ENV):
        monkeypatch.delenv(key, raising=False)
    healthy_root = pathlib.Path(manager.reviewed_node(allow_install=True)).parents[2]  # a verified fixture archive
    monkeypatch.setattr(platform, "probe_node_version", node_runtime.probe_node_version)  # from here probes run files
    broken_root = healthy_root.with_name(healthy_root.name[:-8] + "00000001")  # newer name: tried first by record
    shutil.copytree(healthy_root, broken_root)

    def script(root, version):
        node = root / "node-standalone/bin/node"
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_text(f"#!/bin/sh\necho v{version}\n", encoding="utf-8")
        node.chmod(0o755)
        return str(node)

    healthy, broken = script(healthy_root, NODE_VERSION), script(broken_root, "1.0.0")
    if shipped == "older":
        script(tmp_path / "bundle", "24.0.0")
        monkeypatch.setattr(platform, "bundled_resource_bases", lambda: [tmp_path / "bundle"])
    assert manager.reviewed_node(prepared=True) == broken  # the record alone cannot tell
    assert platform.resolve_bundled_node() == healthy  # never the generation that runs as another version
    assert node_runtime.select_skill_node_runtime() == (healthy, "bundled")
