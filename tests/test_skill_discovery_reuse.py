"""Exact reuse of installed-skill discovery (cost law L13): one inventory and one load per content identity.

``skill_discovery_reuse`` keeps the location inventory and each package's load with the change keys of exactly
what they read. Each law compares the reused discovery with a fresh one of the same install, field for field,
after a change that moves the catalogue (so a stale reuse would differ), and counts what an unchanged install
costs: no walk, no load. ``_RACY_NS`` is zero where a law needs reuse at once (a fixture's files are seconds old);
the racy rule itself keeps its own law. Windows (its ``st_ctime`` is the creation time) and a filesystem the store
does not call local keep nothing, as the module documents: there each law states a full read every time, and one law
forces that rule on every platform.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import types

import pytest

from ouroboros import skill_discovery_reuse as reuse
from ouroboros import skill_loader as loader
from ouroboros.skill_loader import SkillReviewState, discover_skills, save_enabled, save_review_state
from ouroboros.utils import atomic_write_json

KEEPS = os.name != "nt"  # where change keys are trusted discovery is reused; on Windows it reads everything every time


def _manifest(name, description="Check the weather.", scripts="fetch.py"):
    return (f"---\nname: {name}\ndescription: {description}\nversion: 0.1.0\ntype: script\nruntime: python3\n"
            f"scripts:\n  - name: {scripts}\n    description: Fetch.\n---\n# {name}\n")


def _skill(data, name, bucket="external", **manifest):
    return _package(data / "skills" / bucket, name, **manifest)


def _package(parent, name, **manifest):
    folder = parent / name
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_text(_manifest(name, **manifest), encoding="utf-8")
    (folder / "scripts" / "fetch.py").write_text("print('one')\n", encoding="utf-8")
    (folder / "scripts" / "helper.py").write_text("print('one')\n", encoding="utf-8")  # payload no manifest names
    return folder


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Three packages; counts of the inventory walks and package loads discovery performs."""
    data = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.setattr(reuse, "_RACY_NS", 0)
    for name in ("alpha", "beta", "gamma"):
        _skill(data, name)
    counts = {"walks": 0, "loads": 0}
    walk, load = loader._walk_skill_packages, loader.load_skill

    def walked(*args, **kwargs):
        counts["walks"] += 1
        return walk(*args, **kwargs)

    def loaded(*args, **kwargs):
        counts["loads"] += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(loader, "_walk_skill_packages", walked)
    monkeypatch.setattr(loader, "load_skill", loaded)
    reuse._INVENTORIES.clear()
    reuse._SKILLS.clear()
    yield data, counts
    reuse._INVENTORIES.clear()
    reuse._SKILLS.clear()


def _dump(skills):
    return json.dumps([dataclasses.asdict(skill) for skill in skills], sort_keys=True, default=str)


def _fresh(data):
    held = (dict(reuse._INVENTORIES), dict(reuse._SKILLS))
    reuse._INVENTORIES.clear()
    reuse._SKILLS.clear()
    try:
        return _dump(discover_skills(data))
    finally:
        reuse._INVENTORIES.clear()
        reuse._SKILLS.clear()
        reuse._INVENTORIES.update(held[0])
        reuse._SKILLS.update(held[1])


def test_an_unchanged_install_is_discovered_once(install):
    data, counts = install
    first = _dump(discover_skills(data))
    assert counts == {"walks": 1, "loads": 3}  # one root, three packages
    assert _dump(discover_skills(data)) == first and loader.find_skill(data, "beta").name == "beta"
    # a repeat and a lookup read nothing; Windows reads it all again: a discovery walks once and loads three, a lookup one
    assert counts == ({"walks": 1, "loads": 3} if KEEPS else {"walks": 3, "loads": 7})
    assert _fresh(data) == first
    assert counts == ({"walks": 2, "loads": 6} if KEEPS else {"walks": 4, "loads": 10})  # the fresh reference read again


def _edit_same_length(path):
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("one", "two"), encoding="utf-8")  # an equal-size edit: bytes, not size, moved


CHANGES = {
    "install": lambda data, tmp: _skill(data, "delta"),
    "uninstall": lambda data, tmp: shutil.rmtree(data / "skills" / "external" / "gamma"),
    "payload_edit": lambda data, tmp: _edit_same_length(data / "skills" / "external" / "beta" / "scripts" / "helper.py"),
    "manifest_edit": lambda data, tmp: (data / "skills" / "external" / "beta" / "SKILL.md").write_text(
        _manifest("beta", description="Check the weather twice."), encoding="utf-8"),
    "review": lambda data, tmp: save_review_state(data, "beta", SkillReviewState(
        status="pass", content_hash=loader.find_skill(data, "beta").content_hash)),
    "enable": lambda data, tmp: save_enabled(data, "beta", True),
    "owner_edit": lambda data, tmp: atomic_write_json(
        data / "state" / "skills" / "beta" / "owner_edit.json", {"digest": "f" * 64}),
    "self_authored": lambda data, tmp: _self_authored(data, "beta"),
    "new_root": lambda data, tmp: _checkout(tmp),
    "unrelated_state": lambda data, tmp: [atomic_write_json(data / "state" / "skills" / "beta" / name, {"at": 1})
                                          for name in ("health.json", "review_job.json", "grants.json")],
}


def _self_authored(data, name):
    facts = {"schema_version": 1, "origin": "self_authored", "task_id": "t1", "created_at": "2026-11-02T00:00:00Z"}
    (data / "skills" / "external" / name / ".self_authored.json").write_text(json.dumps(facts), encoding="utf-8")
    atomic_write_json(data / "state" / "skills" / name / "self_authored.json", facts)


def _checkout(tmp):
    _package(tmp / "checkout" / "group", "omega")
    os.environ["OUROBOROS_SKILLS_REPO_PATH"] = str(tmp / "checkout")  # the test's monkeypatch restores it


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_each_change_is_seen_and_the_reuse_equals_a_fresh_discovery(install, tmp_path, monkeypatch, change):
    """Install, update, review, enablement, owner and author facts, an external edit or a new root move the
    catalogue, and the discovery after them equals a fresh one; the state a load never reads moves nothing."""
    data, counts = install
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    before = _dump(discover_skills(data))
    CHANGES[change](data, tmp_path)
    held = dict(counts)
    after = _dump(discover_skills(data))
    assert (counts == held) is (KEEPS and change == "unrelated_state")  # no walk, no load only while nothing read moved
    assert after == _fresh(data)
    assert (after == before) is (change == "unrelated_state")


def test_a_manifest_written_in_a_connected_folders_standard_folder_is_seen(install, tmp_path, monkeypatch):
    """A connected folder's ``.agents/skills`` (``skill_sources.STANDARD_SKILL_FOLDERS``) is read one level below
    the two the walk lists: a manifest written into a package folder already there moves the inventory."""
    data, counts = install
    checkout = tmp_path / "checkout"
    _package(checkout / "group", "omega")
    manifest = _package(checkout / ".agents" / "skills", "sigma") / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    manifest.unlink()  # its folder is there before its manifest, as an author or a checkout writes it
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    before = {skill.name for skill in discover_skills(data)}
    walks = counts["walks"]
    assert {skill.name for skill in discover_skills(data)} == before and (counts["walks"] == walks) is KEEPS  # reused
    manifest.write_text(text, encoding="utf-8")
    after = _dump(discover_skills(data))
    assert "omega" in before and "sigma" not in before
    assert "sigma" in {skill["name"] for skill in json.loads(after)} and after == _fresh(data)


def test_a_declared_entry_under_a_skipped_cache_name_is_part_of_the_identity(install):
    """The hash re-adds a manifest-declared script under ``node_modules``; its edit is seen though the tree skips it."""
    data, _counts = install
    folder = _skill(data, "nodey", scripts="node_modules/tool.py")
    (folder / "node_modules").mkdir()
    (folder / "node_modules" / "tool.py").write_text("print('one')\n", encoding="utf-8")
    before = {skill.name: skill.content_hash for skill in discover_skills(data)}
    _edit_same_length(folder / "node_modules" / "tool.py")
    after = {skill.name: skill.content_hash for skill in discover_skills(data)}
    assert after["nodey"] != before["nodey"] and json.dumps(after, sort_keys=True) == json.dumps(
        {s["name"]: s["content_hash"] for s in json.loads(_fresh(data))}, sort_keys=True)


def test_a_load_is_the_same_in_every_runtime_mode(install, monkeypatch):
    """No load fact depends on the mode (a credential-like payload name included): switching to Cyber Pro reuses
    the load and equals a fresh discovery; the gate reads the mode when it is asked, not at load."""
    data, counts = install
    (data / "skills" / "external" / "beta" / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    plain = _dump(discover_skills(data))
    assert not loader.find_skill(data, "beta").load_error
    held = dict(counts)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    cyber = _dump(discover_skills(data))
    assert (counts == held) is KEEPS  # the switch read nothing again where change keys are trusted
    assert cyber == plain == _fresh(data)


def test_callers_get_their_own_copy(install):
    """Callers assign a skill's fields and its review's and manifest's (``skill.enabled = …``, ``skill.review = …``):
    each gets its own objects, the kept skill never moves."""
    data, counts = install
    pristine = _dump([loader.find_skill(data, "beta")])
    beta = loader.find_skill(data, "beta")
    beta.enabled, beta.source, beta.review.status, beta.manifest.description = True, "x", "pass", "changed by a caller"
    other = loader.find_skill(data, "beta")
    other.review = SkillReviewState(status="pass")
    assert _dump([loader.find_skill(data, "beta")]) == pristine
    assert counts["loads"] == (1 if KEEPS else 4)  # one kept load; Windows loads at each of the four lookups


def test_a_just_written_package_is_read_again_until_its_change_time_settles(install, monkeypatch):
    """Within ``_RACY_NS`` of a change a coarse clock could hide a later write: nothing is kept."""
    data, counts = install
    monkeypatch.setattr(reuse, "_RACY_NS", 10 ** 15)  # every fixture file is "just written"
    discover_skills(data)
    discover_skills(data)
    assert counts == {"walks": 2, "loads": 6}


@pytest.mark.parametrize("cause", ["windows", "not_local"])
def test_where_no_change_key_can_be_trusted_nothing_is_kept(install, monkeypatch, cause):
    """Windows and a filesystem the store does not call local answer no change key, and then every discovery walks and
    loads again, so an equal-size edit is seen at once. Forced here, so the rule is exercised on every platform and
    not only where the platform imposes it."""
    from ouroboros import skill_identity, store

    if cause == "windows":
        monkeypatch.setattr(skill_identity, "os", types.SimpleNamespace(name="nt"))  # the one name ``_change_key`` reads
    else:
        monkeypatch.setattr(skill_identity, "_LOCAL_DEVICES", {})
        monkeypatch.setattr(store, "_filesystem_type", lambda path: "nfs")
    data, counts = install
    helper = data / "skills" / "external" / "beta" / "scripts" / "helper.py"
    assert skill_identity._change_key(helper) is None
    first = _dump(discover_skills(data))
    _edit_same_length(helper)
    after = _dump(discover_skills(data))
    assert after != first and after == _fresh(data)
    assert counts == {"walks": 3, "loads": 9}  # the first, the edited and the fresh reference each read all of it


def test_a_load_reads_only_the_listed_owner_state_files(install, monkeypatch):
    """``STATE_FILES`` names every owner-state file a load reads; a load that starts reading another one fails here."""
    data, _counts = install
    state = data / "state" / "skills" / "beta"
    for name in (*reuse.STATE_FILES, "health.json", "review_job.json", "grants.json", "auth_token.json"):
        atomic_write_json(state / name, {"schema_version": 1})
    touched = set()
    original_stat, original_open = pathlib.Path.stat, open

    def stat(self, *args, **kwargs):
        if state in pathlib.Path(self).parents:
            touched.add(pathlib.Path(self).name)
        return original_stat(self, *args, **kwargs)

    def opened(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and state in pathlib.Path(file).parents:
            touched.add(pathlib.Path(file).name)
        return original_open(file, *args, **kwargs)

    candidate = next(c for c in loader._skill_location_inventory(data) if c.name == "beta")
    with monkeypatch.context() as patch:  # only while the one load runs
        patch.setattr(pathlib.Path, "stat", stat)
        patch.setattr("builtins.open", opened)
        patch.setattr("io.open", opened)
        if hasattr(pathlib, "_NormalAccessor"):  # Python 3.10: Path.open reads through the accessor
            patch.setattr(pathlib._NormalAccessor, "open", staticmethod(opened))
        assert loader._load_candidate(candidate, data, False) is not None
    assert touched and touched <= set(reuse.STATE_FILES), touched - set(reuse.STATE_FILES)


def test_host_service_authentication_reads_the_catalogue_and_each_change(install, tmp_path):
    """Auth resolves the token's skill from the catalogue: a repeated request loads nothing, and a review taken
    back or an edited payload is refused at once, exactly as a fresh read refuses it."""
    from ouroboros.gateway.host_service import HostServiceAuthError, HostServiceContext
    from tests.test_host_service_api import _seed_token

    data, counts = install
    _seed_token(data, skill="hooked", token="tok", permissions=["inject_chat"])
    context = HostServiceContext(data)
    assert context.authenticate_token_payload("tok")[0] == "hooked"
    loads = counts["loads"]
    context.require_permission("hooked", {}, "inject_chat")
    assert context.authenticate_token_payload("tok")[0] == "hooked" and (counts["loads"] == loads) is KEEPS
    save_review_state(data, "hooked", SkillReviewState(status="pending"))
    with pytest.raises(HostServiceAuthError, match="no current executable review authority"):
        context.authenticate_token_payload("tok")
    _seed_token(data, skill="hooked", token="tok", permissions=["inject_chat"])
    assert context.authenticate_token_payload("tok")[0] == "hooked"
    (data / "skills" / "external" / "hooked" / "plugin.py").write_text("def register(api): return\n", encoding="utf-8")
    with pytest.raises(HostServiceAuthError, match="no current executable review authority"):  # its review went stale
        context.authenticate_token_payload("tok")
