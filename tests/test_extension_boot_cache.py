"""Reusable imports and explicitly declared catalog inputs across reload/boot."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

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


def test_catalog_reload_counts_and_declared_input_invalidation(tmp_path, monkeypatch):
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


def test_undeclared_catalog_is_one_dry_run_per_source_identity(tmp_path, monkeypatch, settled):
    """K5 law: with unchanged inputs a second worker runs zero catalog children and publishes the same surfaces,
    hashing each payload once; a value the skill's PluginAPI cannot read is not an input, one it reads is."""
    plugin = (
        "def register(api):\n"
        "    label = api.get_settings(['OUROBOROS_UI_LANGUAGE']).get('OUROBOROS_UI_LANGUAGE', 'unset')\n"
        "    api.register_tool('ping', lambda ctx: 'ok', description=label, schema={})\n"
    )
    skill, repo, drive = _prepare_extension(tmp_path, "dynamic", plugin, ["tool", "read_settings"],
                                            env_from_settings=["OUROBOROS_UI_LANGUAGE"],
                                            extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    children = []
    popen = subprocess.Popen

    def counted(cmd, *args, **kwargs):
        if "ouroboros.extension_process_runner" in cmd:
            children.append(json.loads(pathlib.Path(cmd[-1]).read_text(encoding="utf-8"))["skill_name"])
        return popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", counted)
    settings = {"OUROBOROS_UI_LANGUAGE": "one", "OUROBOROS_UPDATE_CHANNEL": "one"}

    def described():
        return loader.get_tool(loader.extension_surface_name("dynamic", "ping"))["description"]

    # The first birth runs the child; its Host Service token binds a second digest form of the payload.
    assert _birth(drive, repo, settings, "dynamic") == (None, {"skills": 1, "skill_hash": 2, "skill_hash_reused": 1,
                                                             "catalog_child": 1})
    first = described()
    assert _birth(drive, repo, settings, "dynamic") == (None, {"skills": 1, "skill_hash": 1, "skill_hash_reused": 1,
                                                             "catalog_reused": 1})
    assert (described(), children) == (first, ["dynamic"]) and first == "one"
    settings["OUROBOROS_UPDATE_CHANNEL"] = "two"  # outside its allowlist: not what its registration can read
    assert _birth(drive, repo, settings, "dynamic")[1].get("catalog_child", 0) == 0
    settings["OUROBOROS_UI_LANGUAGE"] = "two"
    assert _birth(drive, repo, settings, "dynamic")[1]["catalog_child"] == 1
    assert (described(), children) == ("two", ["dynamic", "dynamic"])


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
