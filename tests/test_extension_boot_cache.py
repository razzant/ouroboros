"""Reusable imports and explicitly declared catalog inputs across reload/boot."""
from __future__ import annotations

import json
import pathlib
import subprocess

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
    for name in ("first", "second"):
        skill, repo, drive = _prepare_extension(
            tmp_path, name, plugin.replace("LABEL", name.upper() + "_LABEL"),
            ["tool", "read_settings"], env_from_settings=[name.upper() + "_LABEL"],
            extra_frontmatter=f"dependencies: [dummy_pkg]\ncatalog_inputs:\n  settings: [{name.upper()}_LABEL]\n  environment: [LANG]\n",
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
    settings = {"FIRST_LABEL": "one", "SECOND_LABEL": "one"}

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
        reload()
    assert children == []
    assert set((drive / "state" / "skills").glob("*/__extension_imports/sha256-*")) == trees

    # Change just one payload, keeping version constant; only its child runs.
    source = repo / "first" / "plugin.py"
    source.write_text(plugin.replace("LABEL", "FIRST_LABEL") + "# revised payload\n", encoding="utf-8")
    skill = find_skill(drive, "first", repo_path=str(repo))
    save_review_state(drive, "first", SkillReviewState(status="pass", content_hash=skill.content_hash))
    reload()
    assert children == ["first"]
    children.clear()
    settings["FIRST_LABEL"] = "two"
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


def test_undeclared_catalog_inputs_always_run_child_after_unload(tmp_path, monkeypatch):
    skill, repo, drive = _prepare_extension(tmp_path, "dynamic", "def register(api): pass\n", [],
                                          extra_frontmatter="dependencies: [dummy_pkg]\n")
    _mark_isolated_deps_installed(drive, skill)
    from ouroboros import extension_process_runner as runner
    calls = []
    monkeypatch.setattr(runner, "_run_child", lambda *a, **kw: (calls.append(a), {"ok": True})[1])
    for _ in range(2):
        assert loader.reload_all(drive, lambda: {}, repo_path=str(repo)) == {"dynamic": None}
        loader.unload_extension("dynamic")
    assert len(calls) == 2
    assert not (skill_state_dir(drive, skill.name) / "extension_catalog.json").exists()
