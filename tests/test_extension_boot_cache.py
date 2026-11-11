"""Reusable imports and explicitly declared catalog inputs across reload/boot."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import types

import pytest

from ouroboros import extension_import_staging as staging, extension_loader as loader
from ouroboros.skill_loader import find_skill, save_review_state, SkillReviewState, skill_state_dir
from tests._extension_loader_shared import (
    _prepare_extension, _mark_isolated_deps_installed,
    _clear_loader_state,  # noqa: F401
)


def test_import_tree_reused_and_invalidated_by_code_and_dependencies(tmp_path, monkeypatch):
    skill, repo, drive = _prepare_extension(tmp_path, "reuse", "def register(api): pass\n", [])
    copies, removals = [], []
    copytree, rmtree = staging.shutil.copytree, staging.shutil.rmtree
    monkeypatch.setattr(staging.shutil, "copytree", lambda *a, **kw: (copies.append(a), copytree(*a, **kw))[1])
    monkeypatch.setattr(staging.shutil, "rmtree", lambda *a, **kw: (removals.append(a), rmtree(*a, **kw))[1])

    def reload():
        assert loader.reload_all(drive, lambda: {}, repo_path=str(repo)) == {"reuse": None}
        return pathlib.Path(loader._extensions["reuse"].import_root)

    first = reload()
    assert len(copies) == 1
    assert first.name.startswith("sha256-") and len(first.name) == 71
    # An empty trusted domain still requires the checked-byte identity variant.
    from ouroboros import deployment_policy
    with monkeypatch.context() as trusted:
        trusted.setattr(deployment_policy, "trusted_extensions", lambda: {})
        assert staging.extension_import_identity(skill) != first.name
    loader.unload_extension("reuse")  # a fresh process starts with empty registries too
    copies.clear()
    assert reload() == first
    assert copies == removals == []
    assert reload() == first
    assert copies == removals == []

    env = skill.skill_dir / ".ouroboros_env"
    env.mkdir()
    (env / "fingerprint.json").write_text('{"installed": "changed"}', encoding="utf-8")
    second = reload()
    assert second != first and len(copies) == 1
    (skill.skill_dir / "plugin.py").write_text("def register(api): return None\n", encoding="utf-8")
    changed = find_skill(drive, "reuse", repo_path=str(repo))
    # Cache never lends review authority to changed bytes.
    assert loader.reload_all(drive, lambda: {}, repo_path=str(repo))["reuse"] is not None
    save_review_state(drive, "reuse", SkillReviewState(status="pass", content_hash=changed.content_hash))
    assert reload() not in (first, second)
    assert len(copies) == 2


def test_shared_tree_sweep_requires_no_current_hash_and_dead_owners(tmp_path, monkeypatch):
    skill, _, drive = _prepare_extension(tmp_path, "owners", "def register(api): pass\n", [])
    state = skill_state_dir(drive, skill.name)
    tree, _ = staging._stage_extension_import_tree(skill, state_dir=state, entry_path=skill.skill_dir / "plugin.py")
    staging.release_extension_import_tree(tree)
    peer = 888888
    (tree / "owners" / str(peer)).touch()
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid == peer)
    staging._sweep_stale_extension_imports(drive, skill.name)
    assert tree.exists()  # creator dead, but a different worker uses these bytes
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: False)
    staging._sweep_stale_extension_imports(drive, skill.name, keep_hashes=[skill.content_hash])
    assert tree.exists()  # retain the current payload across boots
    staging._sweep_stale_extension_imports(drive, skill.name)
    assert not tree.exists()


def test_trees_under_the_earlier_two_digest_name_are_not_reused_and_leave_by_the_existing_sweep(tmp_path, monkeypatch):
    """Earlier generations' ``sha256-<content>-<runtime>`` trees are never taken for the one-digest name; the sweep
    keeps each while its payload is current or one of its owners lives, then reclaims it: no migration or reaper."""
    skill, repo, drive = _prepare_extension(tmp_path, "renamed", "def register(api): pass\n", [])
    dead = {777001, 777003}  # the earlier server, and the worker that used only the idle tree

    def earlier(runtime: bytes, worker: int) -> pathlib.Path:
        tree = (skill_state_dir(drive, skill.name) / "__extension_imports"
                / f"sha256-{skill.content_hash}-{hashlib.sha256(runtime).hexdigest()}")
        staging._copy_extension_tree(skill, tree)  # the unchanged publisher, under the earlier name
        complete = json.loads((tree / "complete.json").read_text(encoding="utf-8"))
        (tree / "complete.json").write_text(json.dumps({**complete, "owner_pid": 777001}), encoding="utf-8")
        (tree / "owners").mkdir()
        (tree / "owners" / str(worker)).touch()
        return tree

    held, idle = earlier(b"an earlier runtime", 777002), earlier(b"an older runtime", 777003)
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid not in dead)
    copies, copytree = [], staging.shutil.copytree
    monkeypatch.setattr(staging.shutil, "copytree", lambda *a, **kw: (copies.append(a), copytree(*a, **kw))[1])

    def reload():
        assert loader.reload_all(drive, lambda: {}, repo_path=str(repo)) == {"renamed": None}
        return pathlib.Path(loader._extensions["renamed"].import_root)

    current = reload()
    assert current.name == staging.extension_import_identity(skill) and len(copies) == 1  # neither was reused
    assert len(held.name) - len(current.name) == 65 and held.is_dir() and idle.is_dir()  # the payload is current
    (skill.skill_dir / "plugin.py").write_text("def register(api): return None\n", encoding="utf-8")
    changed = find_skill(drive, "renamed", repo_path=str(repo))
    save_review_state(drive, "renamed", SkillReviewState(status="pass", content_hash=changed.content_hash))
    assert reload() != current and held.is_dir() and not idle.exists()  # only a live owner keeps one now
    dead.add(777002)
    reload()
    assert not held.exists() and current.is_dir()  # nothing holds it; ``current``'s creator (this process) lives


def test_catalog_reload_counts_and_declared_input_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")  # only a worker takes the record; declared inputs narrow its key
    plugin = (
        "import os\n"
        "def register(api):\n"
        "    label = api.get_settings(['LABEL']).get('LABEL', 'unset')\n"
        "    api.register_tool('ping', lambda ctx: 'ok', description=label + os.environ.get('LANG', ''), schema={})\n"
    )
    # Each label is an ordinary setting the skill reads without a grant (a custom key would be a custom secret).
    label_keys = {"first": "OUROBOROS_UPDATE_CHANNEL", "second": "OUROBOROS_UI_LANGUAGE"}
    for name in ("first", "second"):
        skill, repo, drive = _prepare_extension(
            tmp_path, name, plugin.replace("LABEL", label_keys[name]),
            ["tool", "read_settings"], env_from_settings=[label_keys[name]],
            extra_frontmatter=f"dependencies: [dummy_pkg]\ncatalog_inputs:\n  settings: [{label_keys[name]}]\n  environment: [LANG]\n",
        )
        _mark_isolated_deps_installed(drive, skill)
    children = []
    popen = subprocess.Popen

    def counted(cmd, *args, **kwargs):
        if "ouroboros.extension_process_runner" in cmd:
            payload = json.loads(pathlib.Path(cmd[-1]).read_text(encoding="utf-8"))
            children.append(payload["skill_name"])
        return popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", counted)
    settings = {label_keys["first"]: "one", label_keys["second"]: "one"}

    def reload():
        assert loader.reload_all(drive, lambda: settings, repo_path=str(repo)) == {"first": None, "second": None}

    reload()
    assert children == ["first", "second"]
    trees = set((drive / "state" / "skills").glob("*/__extension_imports/sha256-*"))
    assert len(trees) == 2
    children.clear()
    for name in ("first", "second"):
        loader.unload_extension(name)
    with monkeypatch.context() as mp:
        mp.setattr(staging.shutil, "copytree", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("copy")))
        mp.setattr(staging.shutil, "rmtree", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("remove")))
        reload()
    assert children == []
    assert set((drive / "state" / "skills").glob("*/__extension_imports/sha256-*")) == trees

    # Change just one payload, keeping version constant; only its child runs.
    source = repo / "first" / "plugin.py"
    source.write_text(plugin.replace("LABEL", label_keys["first"]) + "# revised payload\n", encoding="utf-8")
    skill = find_skill(drive, "first", repo_path=str(repo))
    save_review_state(drive, "first", SkillReviewState(status="pass", content_hash=skill.content_hash))
    reload()
    assert children == ["first"]
    children.clear()
    settings[label_keys["first"]] = "two"
    reload()
    assert children == ["first"]
    for name, label in (("first", "two"), ("second", "one")):
        assert loader.get_tool(loader.extension_surface_name(name, "ping"))["description"].startswith(label)
    children.clear()
    monkeypatch.setenv("LANG", "changed-language")
    reload()
    assert children == ["first", "second"]
    children.clear()
    fingerprint = repo / "first" / ".ouroboros_env" / "fingerprint.json"
    deps = json.loads(fingerprint.read_text(encoding="utf-8"))
    fingerprint.write_text(json.dumps({**deps, "installed_at": "new-install"}), encoding="utf-8")
    reload()
    assert children == ["first"]


def _birth(drive, repo, settings, name):
    """One worker birth's reload of ``name``: a fresh process's registries and memories, its own measured window
    (``worker_main``); the fixture's files count as settled (``_RACY_NS`` is zeroed by the caller)."""
    from ouroboros import context_timing, skill_discovery_reuse as reuse, skill_identity

    loader.unload_extension(name)
    for memo in (reuse._INVENTORIES, reuse._SKILLS, reuse._DIGESTS, skill_identity._DIGESTS):
        memo.clear()
    with context_timing.measured_sections():
        results = loader.reload_all(drive, lambda: settings, repo_path=str(repo))
        return results[name], context_timing.section_spans()["work"]


@pytest.fixture
def settled(monkeypatch):
    from ouroboros import skill_discovery_reuse as reuse, skill_identity

    monkeypatch.setattr(reuse, "_RACY_NS", 0)
    monkeypatch.setattr(skill_identity, "_RACY_NS", 0)
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")  # ``_birth`` is a worker's


def _on_the_server(monkeypatch, drive, repo, settings, name):
    """One load of ``name`` in the server's role (boot, reconcile, enable, reload): not a worker's process."""
    with monkeypatch.context() as server:
        server.delenv("OUROBOROS_IN_WORKER")
        return _birth(drive, repo, settings, name)


def _catalog_children(monkeypatch) -> list:
    """The catalog children spawned from now on, by skill."""
    children, popen = [], subprocess.Popen

    def counted(cmd, *args, **kwargs):
        if "ouroboros.extension_process_runner" in cmd:
            children.append(json.loads(pathlib.Path(cmd[-1]).read_text(encoding="utf-8"))["skill_name"])
        return popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", counted)
    return children


@pytest.mark.parametrize("change_keys", ["platform", "absent"])
def test_undeclared_catalog_is_one_dry_run_per_source_identity(tmp_path, monkeypatch, settled, change_keys):
    """K5 law: with unchanged inputs a second worker runs zero catalog children and publishes the same surfaces,
    hashing each payload once where its change key is trusted (``skill_discovery_reuse.keeps``); where none is
    (Windows, a filesystem the store does not call local; forced here on every platform) the load hashes the payload
    again, that platform's repeat. A value the skill's PluginAPI cannot read is not an input, one it reads is."""
    from ouroboros import skill_discovery_reuse as reuse, skill_identity

    if change_keys == "absent":
        monkeypatch.setattr(skill_identity, "os", types.SimpleNamespace(name="nt"))  # the one name ``_change_key`` reads
    plugin = (
        "def register(api):\n"
        "    label = api.get_settings(['OUROBOROS_UI_LANGUAGE']).get('OUROBOROS_UI_LANGUAGE', 'unset')\n"
        "    api.register_tool('ping', lambda ctx: 'ok', description=label, schema={})\n"
    )
    skill, repo, drive = _prepare_extension(tmp_path, "dynamic", plugin, ["tool", "read_settings"],
                                            env_from_settings=["OUROBOROS_UI_LANGUAGE"],
                                            extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    children = _catalog_children(monkeypatch)
    settings = {"OUROBOROS_UI_LANGUAGE": "one", "OUROBOROS_UPDATE_CHANNEL": "one"}

    def described():
        return loader.get_tool(loader.extension_surface_name("dynamic", "ping"))["description"]

    keeps = reuse.keeps(skill.skill_dir)  # the rule ``reused_digest`` itself follows
    assert change_keys == "platform" or not keeps

    def hashes(forms: int) -> dict:  # each digest form once; the load takes discovery's, or hashes it again unkept
        return ({"skill_hash": forms, "skill_hash_reused": 1} if keeps
                else {"skill_hash": forms + 1, "skill_hash_again": 1})

    # The first birth runs the child; its Host Service token binds a second digest form of the payload.
    assert _birth(drive, repo, settings, "dynamic") == (None, {"skills": 1, **hashes(2), "catalog_child": 1})
    first = described()
    assert _birth(drive, repo, settings, "dynamic") == (None, {"skills": 1, **hashes(1), "catalog_reused": 1})
    assert (described(), children) == (first, ["dynamic"]) and first == "one"
    settings["OUROBOROS_UPDATE_CHANNEL"] = "two"  # outside its allowlist: not what its registration can read
    assert _birth(drive, repo, settings, "dynamic")[1].get("catalog_child", 0) == 0
    settings["OUROBOROS_UI_LANGUAGE"] = "two"
    assert _birth(drive, repo, settings, "dynamic")[1]["catalog_child"] == 1
    assert (described(), children) == ("two", ["dynamic", "dynamic"])


def test_a_load_on_the_server_runs_registration_anew_and_workers_take_its_result(tmp_path, monkeypatch, settled):
    """A registration may read its own state, which no key holds. A worker with an unchanged record runs no child and
    publishes what the server published; a load on the server runs the child and replaces the record, so the worker
    born after it publishes the new catalog."""
    plugin = (
        "from pathlib import Path\n"
        "def register(api):\n"
        "    path = Path(api.get_state_dir()) / 'selected.txt'\n"
        "    selected = path.read_text() if path.exists() else 'first'\n"
        "    api.register_tool(selected, lambda ctx: 'ok', description='Selected capability', schema={})\n"
    )
    skill, repo, drive = _prepare_extension(tmp_path, "stateful", plugin, ["tool"],
                                            extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    children = _catalog_children(monkeypatch)

    def published():
        return [tool for tool in ("first", "second") if loader.get_tool(loader.extension_surface_name("stateful", tool))]

    assert _on_the_server(monkeypatch, drive, repo, {}, "stateful")[0] is None
    assert (published(), children) == (["first"], ["stateful"])
    assert _birth(drive, repo, {}, "stateful")[0] is None  # a worker on the unchanged record: no child, the same catalog
    assert (published(), children) == (["first"], ["stateful"])
    (skill_state_dir(drive, "stateful") / "selected.txt").write_text("second", encoding="utf-8")
    assert _birth(drive, repo, {}, "stateful")[0] is None  # nothing has run registration since the state changed
    assert (published(), children) == (["first"], ["stateful"])
    assert _on_the_server(monkeypatch, drive, repo, {}, "stateful")[0] is None
    assert (published(), children) == (["second"], ["stateful", "stateful"])  # the reload: a child and the new catalog
    assert _birth(drive, repo, {}, "stateful")[0] is None  # the worker born after it publishes that catalog
    assert (published(), children) == (["second"], ["stateful", "stateful"])


def test_a_server_load_that_fails_leaves_no_earlier_catalog_for_a_worker(tmp_path, monkeypatch, settled):
    """A dry run that fails on the server records nothing and takes the earlier result away: the next worker runs its
    own child instead of publishing what no longer loads."""
    from ouroboros import extension_process_runner as runner

    skill, repo, drive = _prepare_extension(tmp_path, "fragile", "def register(api): pass\n", [],
                                            extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    record = skill_state_dir(drive, "fragile") / "extension_catalog.json"
    runs, child = [], {"fails": False}

    def run_child(payload, **_kw):
        runs.append(payload["skill_name"])
        if child["fails"]:
            raise runner.ExtensionProcessError("register() raised")
        return {"ok": True, "content_hash": skill.content_hash}

    monkeypatch.setattr(runner, "_run_child", run_child)
    assert _on_the_server(monkeypatch, drive, repo, {}, "fragile")[0] is None and record.exists() and len(runs) == 1
    assert _birth(drive, repo, {}, "fragile")[0] is None and len(runs) == 1
    with monkeypatch.context() as in_use:  # a record the platform will not delete yet does not fail the load
        unlink = pathlib.Path.unlink
        in_use.setattr(pathlib.Path, "unlink", lambda self, *a, **kw: unlink(self, *a, **kw) if self != record else (
            _ for _ in ()).throw(PermissionError("in use")))
        assert _on_the_server(monkeypatch, drive, repo, {}, "fragile")[0] is None and len(runs) == 2
    child["fails"] = True
    assert _on_the_server(monkeypatch, drive, repo, {}, "fragile")[0] is not None and not record.exists()
    assert _birth(drive, repo, {}, "fragile")[0] is not None and len(runs) == 4  # the worker's own child, not a record


def test_catalog_record_follows_its_inputs_and_never_holds_a_failure(tmp_path, monkeypatch, settled):
    """A changed payload, grant, dependency set or interpreter runs the child again; a failed dry run leaves the
    skill not ready and records nothing, so the next load runs it again; a child that loaded other bytes than the
    key's is used but not recorded."""
    from ouroboros import extension_process_runner as runner
    from ouroboros.skill_loader import save_skill_grants

    skill, repo, drive = _prepare_extension(tmp_path, "keyed", "def register(api): pass\n", [],
                                            extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    record = skill_state_dir(drive, "keyed") / "extension_catalog.json"
    runs, child = [], {"fails": True, "loaded": None}

    def run_child(payload, **_kw):  # the child's protocol answer: the bytes it loaded, or its failure
        runs.append(payload["catalog_settings"])
        if child["fails"]:
            raise runner.ExtensionProcessError("register() raised")
        return {"ok": True, "content_hash": child["loaded"] or find_skill(drive, "keyed", repo_path=str(repo)).content_hash}

    monkeypatch.setattr(runner, "_run_child", run_child)

    def birth():
        return _birth(drive, repo, {}, "keyed")[0]

    assert birth() is not None and birth() is not None  # not ready, twice: the failure was never recorded
    assert (len(runs), record.exists(), loader.is_extension_live("keyed", drive, repo_path=str(repo))) == (2, False, False)
    child.update(fails=False, loaded="other bytes")
    assert birth() is None and not record.exists()
    child["loaded"] = None
    assert birth() is None and birth() is None and len(runs) == 4 and record.exists()
    save_skill_grants(drive, "keyed", [], content_hash=skill.content_hash, requested_keys=[])  # a new grant record
    assert birth() is None and len(runs) == 5
    fingerprint = skill.skill_dir / ".ouroboros_env" / "fingerprint.json"
    fingerprint.write_text(json.dumps({**json.loads(fingerprint.read_text(encoding="utf-8")), "at": "new"}),
                           encoding="utf-8")
    assert birth() is None and len(runs) == 6
    interpreter = tmp_path / "another-python"
    interpreter.write_bytes(b"#!another interpreter\n")
    with monkeypatch.context() as mp:
        mp.setattr(sys, "executable", str(interpreter))
        assert birth() is None and len(runs) == 7
    assert birth() is None and len(runs) == 8
    (skill.skill_dir / "plugin.py").write_text("def register(api): return None\n", encoding="utf-8")
    changed = find_skill(drive, "keyed", repo_path=str(repo))
    save_review_state(drive, "keyed", SkillReviewState(status="pass", content_hash=changed.content_hash))
    assert birth() is None and birth() is None and len(runs) == 9
    assert runs[-1] == {}  # the catalog child reads exactly what the key names
