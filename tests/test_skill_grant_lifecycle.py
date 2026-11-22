"""The owner's grants across the doors that accept, update and edit a skill (C10 п.55, owner 1A 21.11; review RV1).

Each test runs a real door (installer, re-seed, owner action, review acceptance, startup import, Host Service) on
fixture payloads and settings, and pins both directions: what the door must carry or keep, and what it must not
give. The scope rules themselves live in ``ouroboros/skill_grant_scope.py`` and their home tests in
``tests/test_skill_loader.py``; this file is where deleting a door's call is caught (RV1 S17).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import config
from ouroboros.skill_acceptance import carry_if_accepted, record_owner_edit
from ouroboros.skill_grant_scope import install_identity
from ouroboros.skill_loader import (
    SkillReviewState, find_skill, grant_status_for_skill, load_skill_grants, save_enabled, save_review_state,
    save_skill_grants,
)


def manifest(folder, *, version="1.0.0", keys=(), perms=()):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: local fixture\nversion: {version}\ntype: extension\n"
        f'entry: plugin.py\nplugin_api: "2.0"\npermissions: [{", ".join(perms)}]\n'
        f'env_from_settings: [{", ".join(keys)}]\n---\nFixture.\n', encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: False)
    return drive


def settings(drive, **entries):
    (drive / "settings.json").write_text(json.dumps(entries), encoding="utf-8")


def grant(drive, *, keys=(), perms=(), name="demo"):
    skill = find_skill(drive, name)
    save_review_state(drive, name, SkillReviewState(status="clean", content_hash=skill.content_hash))
    save_skill_grants(drive, name, list(keys), content_hash=skill.content_hash, requested_keys=list(keys),
                      granted_permissions=list(perms), requested_permissions=list(perms),
                      install_identity=install_identity(drive, skill))
    return skill


def state(drive, name="demo"):
    return grant_status_for_skill(drive, find_skill(drive, name))


def accept(drive, name="demo"):
    current = find_skill(drive, name)
    save_review_state(drive, name, SkillReviewState(status="clean", content_hash=current.content_hash))
    return carry_if_accepted(drive, find_skill(drive, name))


@pytest.mark.parametrize("policy", ["trusted", "review"])
def test_a_builtin_reseed_carries_only_under_a_trusted_source_policy(root, monkeypatch, policy):
    """S1 (05.10 D-A3): under a «review each» built-in policy the new release waits, before its bytes land, and gets
    neither the key nor the permission until a recorded check; a trusted policy carries both as before."""
    from ouroboros.skill_builtin import install_builtin_skill, reseed_changed_builtins
    from ouroboros.skill_install_record import read_install_record

    seeds = root.parent / "seeds"
    settings(root, FIXTURE_KEY="example")
    manifest(seeds / "demo", keys=("FIXTURE_KEY",), perms=("notify_owner",))
    assert install_builtin_skill("demo", drive_root=root, seed_dir=seeds)["basis"] == "trusted"
    accepted = grant(root, keys=("FIXTURE_KEY",), perms=("notify_owner",)).content_hash
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true" if policy == "trusted" else "false")
    manifest(seeds / "demo", version="2.0.0", keys=("FIXTURE_KEY",), perms=("notify_owner",))
    assert reseed_changed_builtins(root, seed_dir=seeds) == 1
    current = find_skill(root, "demo")
    assert current.content_hash != accepted
    status = state(root)
    if policy == "trusted":
        assert (status["granted_keys"], status["granted_permissions"]) == (["FIXTURE_KEY"], ["notify_owner"])
        return
    record = read_install_record(root, "demo")
    assert record["basis"] == "review" and record["pending_update"]["digest"] == current.content_hash
    assert load_skill_grants(root, "demo")["content_hash"] == accepted, "nothing was written for the new release"
    assert (status["granted_keys"], status["granted_permissions"]) == ([], [])
    assert carry_if_accepted(root, current) is False  # the release is not accepted yet
    assert accept(root) is True  # a recorded check of these exact bytes admits the release; the grants follow
    assert "pending_update" not in read_install_record(root, "demo")
    status = state(root)
    assert (status["granted_keys"], status["granted_permissions"]) == (["FIXTURE_KEY"], ["notify_owner"])


def test_only_a_recorded_basis_accepts_bytes_never_cyber_pro_alone(root, monkeypatch):
    """S3 (18.10): an author record on the current bytes in Cyber Pro, without a verdict about them, grants nothing;
    a finished verdict about these bytes does."""
    from ouroboros.skill_grant_scope import bytes_accepted
    from ouroboros.skill_loader import auto_grant_if_enabled

    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    settings(root, FIXTURE_KEY="example")
    folder = root / "skills" / "external" / "demo"
    manifest(folder, keys=("FIXTURE_KEY",), perms=("inject_chat",))
    skill = find_skill(root, "demo")
    author = {"action": "finish", "subject_hash": skill.content_hash, "enforcement": "advisory",
              "actor": "ouroboros", "reason": "fixture", "recorded_at": "2026-11-22T00:00:00Z"}
    save_review_state(root, "demo", SkillReviewState(status="pending", content_hash="older", author_disposition=author))
    skill = find_skill(root, "demo")
    assert skill.review.gate_for(skill.content_hash)["executable_review"], "Cyber Pro runs it"
    assert bytes_accepted(skill) is False and auto_grant_if_enabled(root, skill).granted is False
    save_review_state(root, "demo", SkillReviewState(status="clean", content_hash=skill.content_hash,
                                                     author_disposition=author))
    skill = find_skill(root, "demo")
    assert bytes_accepted(skill) is True and auto_grant_if_enabled(root, skill).granted is True


def test_an_owner_skip_of_a_pending_update_carries_the_grant_to_it(root, monkeypatch):
    """S9: Skip of a review-each update clears the mark, then carries; auto-grant off."""
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_source_catalog import tree_release_digest

    folder = root.parent / "source" / "demo"
    manifest(folder, perms=("notify_owner",))
    row = {"id": "team", "name": "Team", "kind": "hub", "url": "https://example.invalid/team",
           "trust": "trusted", "confirmed": True}
    item = {"id": "team/demo", "name": "demo", "version": "1.0.0", "digest": tree_release_digest(folder)}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([row]))
    monkeypatch.setattr(installer, "find_item", lambda *a, **kw: (row, item, {}))
    monkeypatch.setattr(installer, "entry_payload_dir", lambda *a, **kw: folder)
    assert installer.install_item(root, item["id"], rows=[row], turn_on_when_ready=False)["ok"]
    grant(root, perms=("notify_owner",))
    row["trust"] = "review"
    manifest(folder, version="2.0.0", perms=("notify_owner",))
    item.update(version="2.0.0", digest=tree_release_digest(folder))
    assert installer.update_installed(root, "demo", mode="replace", rows=[row])["needs_review"]
    assert state(root)["granted_permissions"] == [], "the unaccepted release waits"
    digest = find_skill(root, "demo").content_hash
    ctx = SimpleNamespace(drive_root=root, repo_dir=root.parent, messages=[])
    decision = run_skill_action(ctx, "demo", "attest", expected_content_hash=digest, _owner_actor="owner_ui",
                                repo_path="")
    assert decision["ok"], decision
    assert "pending_update" not in read_install_record(root, "demo")
    status = state(root)
    assert (status["granted_permissions"], status["waiting_permissions"]) == (["notify_owner"], [])


def test_acceptance_keeps_a_consent_settings_lost_and_retires_a_key_the_manifest_dropped(root):
    """S6: a declared key's consent survives a moment without its value; an accepted manifest without the key retires
    it, and a later version that declares it again does not revive it."""
    folder = root / "skills" / "external" / "demo"
    settings(root, FIXTURE_KEY="example")
    manifest(folder, keys=("FIXTURE_KEY",), perms=("notify_owner",))
    grant(root, keys=("FIXTURE_KEY",), perms=("notify_owner",))
    settings(root)  # the owner clears the field during the update
    manifest(folder, version="2.0.0", keys=("FIXTURE_KEY",), perms=("notify_owner",))
    assert accept(root) is True
    settings(root, FIXTURE_KEY="restored")
    status = state(root)
    assert (status["granted_keys"], status["pending_keys"], status["granted_permissions"]) == (
        ["FIXTURE_KEY"], [], ["notify_owner"])
    manifest(folder, version="3.0.0", perms=("notify_owner",))
    assert accept(root) is True
    assert load_skill_grants(root, "demo")["granted_keys"] == []
    manifest(folder, version="4.0.0", keys=("FIXTURE_KEY",), perms=("notify_owner",))
    assert state(root)["granted_keys"] == []


@pytest.mark.parametrize("door", ["trusted_builtin", "owner_edit", "existing_install", "review"])
def test_every_admission_door_records_a_decision_so_a_first_saved_key_stops_nothing(root, door):
    """S7: an accepted extension that declares a key nobody saved yet keeps running when the key is saved later, at
    every door that accepts it; a deleted record (a revocation) still stops it."""
    from ouroboros.skill_acceptance import record_missing_decisions
    from ouroboros.skill_builtin import install_builtin_skill
    from ouroboros.skill_loader import discover_skills

    settings(root)
    if door == "trusted_builtin":
        seeds = root.parent / "seeds"
        manifest(seeds / "demo", keys=("FIXTURE_KEY",))
        assert install_builtin_skill("demo", drive_root=root, seed_dir=seeds)["basis"] == "trusted"
    else:
        folder = root / "skills" / "external" / "demo"
        manifest(folder, keys=("FIXTURE_KEY",))
        installed = find_skill(root, "demo")
        if door == "owner_edit":
            assert record_owner_edit(root, folder, task_id="owner-repair") == installed.content_hash
        else:
            save_review_state(root, "demo", SkillReviewState(status="clean", content_hash=installed.content_hash))
            if door == "review":
                carry_if_accepted(root, find_skill(root, "demo"))
            else:  # accepted before decisions were recorded: the startup migration records it once
                assert record_missing_decisions(root, discover_skills(root)) == ["demo"]
    save_enabled(root, "demo", True, actor="owner_ui")
    assert state(root)["usable"]
    settings(root, FIXTURE_KEY="first-example")
    status = state(root)
    assert (status["granted_keys"], status["usable"], status["pending_keys"]) == ([], True, ["FIXTURE_KEY"])
    (root / "state" / "skills" / "demo" / "grants.json").unlink()  # the owner revokes
    assert state(root)["blocked"] is True


def test_a_key_only_owner_grant_keeps_the_last_accepted_permission(root, monkeypatch):
    """S8: granting one key to edited bytes leaves the permission the accepted version holds, still waiting; the
    next acceptance carries both."""
    from ouroboros.skill_lifecycle_actions import run_skill_action

    folder = root / "skills" / "external" / "demo"
    settings(root, FIXTURE_KEY="example")
    manifest(folder, perms=("notify_owner",))
    grant(root, perms=("notify_owner",))
    manifest(folder, version="2.0.0", keys=("FIXTURE_KEY",), perms=("notify_owner",))
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    assert state(root)["waiting_permissions"] == ["notify_owner"]
    current = find_skill(root, "demo")
    ctx = SimpleNamespace(drive_root=root, repo_dir=root.parent, messages=[])
    result = run_skill_action(ctx, "demo", "grant", expected_content_hash=current.content_hash,
                              items=["FIXTURE_KEY"], _owner_actor="owner_ui", repo_path="")
    assert result["ok"], result
    status = state(root)
    assert (status["granted_keys"], status["waiting_permissions"]) == (["FIXTURE_KEY"], ["notify_owner"])
    assert accept(root) is True
    status = state(root)
    assert (status["granted_keys"], status["granted_permissions"]) == (["FIXTURE_KEY"], ["notify_owner"])


def test_one_release_in_two_installations_shares_no_grant(root):
    """S5: identical bytes under another installation inherit nothing; a reinstall of the same installation with the
    same bytes keeps its grant (D-C16), an edit after it does not; a partial grant there grants only itself."""
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_uninstall_state import write_uninstall_tombstone

    settings(root, FIXTURE_KEY="example", OTHER_KEY="other")
    folder = root / "skills" / "ouroboroshub" / "demo"
    manifest(folder, keys=("FIXTURE_KEY", "OTHER_KEY"))
    write_install_record(root, "demo", {"source": {"id": "hub-a", "kind": "hub"}, "release": {"digest": "a" * 64},
                                        "basis": "trusted"})
    grant(root, keys=("FIXTURE_KEY",))
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]
    write_install_record(root, "demo", {"source": {"id": "mirror", "kind": "hub"}, "release": {"digest": "a" * 64},
                                        "basis": "trusted"})
    assert state(root)["granted_keys"] == [], "the same release from another source is another installation"
    write_install_record(root, "demo", {"source": {"id": "hub-a", "kind": "hub"}, "release": {"digest": "a" * 64},
                                        "basis": "trusted"})
    write_uninstall_tombstone(root, "demo", source="hub")
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"], "a reinstall of the same bytes keeps its grant"
    manifest(folder, version="1.0.1", keys=("FIXTURE_KEY", "OTHER_KEY"))
    assert state(root)["granted_keys"] == [], "the ended installation does not follow an edit"
    write_install_record(root, "demo", {"source": {"id": "mirror", "kind": "hub"}, "release": {"digest": "b" * 64},
                                        "basis": "trusted"})
    skill = find_skill(root, "demo")
    save_skill_grants(root, "demo", ["OTHER_KEY"], content_hash=skill.content_hash,
                      requested_keys=["FIXTURE_KEY", "OTHER_KEY"], install_identity=install_identity(root, skill))
    assert state(root)["granted_keys"] == ["OTHER_KEY"], "a later partial grant does not cement the old one"


def test_startup_converts_sidecar_installs_before_binding_their_grants(root):
    """S10 and S13: the startup binding runs after sidecar-era installs got their source record, keeps the original
    record bytes, and reads the connected skills folder too."""
    import hashlib
    from ouroboros import skill_builtin
    from ouroboros.skill_install_record import read_install_record

    folder = root / "skills" / "ouroboroshub" / "demo"
    manifest(folder, keys=("FIXTURE_KEY",))
    settings(root, FIXTURE_KEY="example")
    files = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in folder.iterdir()]
    (folder / ".ouroboroshub.json").write_text(json.dumps({
        "source": "ouroboroshub", "slug": "demo", "version": "1.0.0", "files": files,
        "catalog_url": "https://raw.githubusercontent.com/example/skills/main/catalog.json"}), encoding="utf-8")
    digest = find_skill(root, "demo").content_hash
    save_skill_grants(root, "demo", ["FIXTURE_KEY"], content_hash=digest, requested_keys=["FIXTURE_KEY"])
    original = (root / "state" / "skills" / "demo" / "grants.json").read_bytes()
    (root.parent / "bundle" / "skills").mkdir(parents=True)
    skill_builtin.ensure_builtin_skills(root, repo_dir=root.parent / "bundle", dependency_pass=False)  # the door
    source = read_install_record(root, "demo")["source"]["id"]
    assert load_skill_grants(root, "demo")["install_identity"] == {"source": source, "location": "data:ouroboroshub/demo"}
    assert source, "the binding names the converted source, never an empty one"
    assert (root / "state" / "skills" / "demo" / "grants.before-installation.json").read_bytes() == original
    manifest(folder, version="1.0.1", keys=("FIXTURE_KEY",))
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]


def test_a_skill_in_the_connected_folder_is_bound_at_startup(root, monkeypatch):
    """S13: the startup import resolves skills of the connected folder (``OUROBOROS_SKILLS_REPO_PATH``)."""
    from ouroboros import skill_builtin

    repo = root.parent / "my-skills"
    manifest(repo / "demo", keys=("FIXTURE_KEY",))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    settings(root, FIXTURE_KEY="example")
    skill = find_skill(root, "demo", repo_path=str(repo))
    save_skill_grants(root, "demo", ["FIXTURE_KEY"], content_hash=skill.content_hash, requested_keys=["FIXTURE_KEY"])
    skill_builtin._bind_grants(root, None)
    assert load_skill_grants(root, "demo")["install_identity"]["location"].startswith("path:")



def test_a_finished_review_carries_the_installation_grants_and_a_wave_without_verdict_does_not(root, monkeypatch):
    """S17: the review door (``skill_review._persist_projection``) is an acceptance event: a verdict about the edited
    bytes carries the waiting permission; a wave that formed no verdict carries nothing."""
    from ouroboros import skill_review

    folder = root / "skills" / "external" / "demo"
    manifest(folder, perms=("notify_owner",))
    grant(root, perms=("notify_owner",))
    manifest(folder, version="2.0.0", perms=("notify_owner",))
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    current = find_skill(root, "demo")
    assert state(root)["waiting_permissions"] == ["notify_owner"]
    pending = skill_review.SkillReviewOutcome(skill_name="demo", status="pending", content_hash=current.content_hash)
    skill_review._persist_projection(SimpleNamespace(), current, root, pending)
    assert state(root)["waiting_permissions"] == ["notify_owner"], "no verdict: nothing is accepted"
    verdict = skill_review.SkillReviewOutcome(skill_name="demo", status="clean", content_hash=current.content_hash)
    skill_review._persist_projection(SimpleNamespace(), find_skill(root, "demo"), root, verdict)
    status = state(root)
    assert (status["granted_permissions"], status["waiting_permissions"]) == (["notify_owner"], [])


def test_a_key_granted_later_reaches_an_extension_already_loaded_in_process(tmp_path, monkeypatch):
    """S12: a key saved later is not passed and unloads nothing (B″); once the owner grants it, the next reconcile
    reloads the in-process extension, whose ``get_settings`` then sees the key."""
    from ouroboros import extension_loader
    from tests._extension_loader_shared import _prepare_extension, extension_tool_answer
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    plugin = ("def register(api):\n    api.register_tool('token', lambda ctx: "
              "api.get_settings(['MAILER_TOKEN']).get('MAILER_TOKEN', 'none'), description='token', schema={})\n")
    loaded, repo, drive = _prepare_extension(tmp_path, "mailer", plugin, ["tool", "read_settings"],
                                             env_from_settings=["MAILER_TOKEN"])
    values: dict = {}
    monkeypatch.setattr(config, "load_settings", lambda *_a, **_k: dict(values))
    try:
        assert carry_if_accepted(drive, loaded) is False  # the acceptance records its (empty) decision
        reconcile = lambda: extension_loader.reconcile_extension(  # noqa: E731
            "mailer", drive, lambda: dict(values), repo_path=str(repo))
        assert reconcile()["live_loaded"]
        values["MAILER_TOKEN"] = "fixture-secret"
        assert reconcile()["live_loaded"], "a key saved later unloads nothing"
        assert extension_tool_answer(drive, "mailer", "token") == "none", "and it is not passed"
        current = find_skill(drive, "mailer", repo_path=str(repo))
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=current.content_hash,
                          requested_keys=["MAILER_TOKEN"], install_identity=install_identity(drive, current))
        assert reconcile()["live_loaded"]
        assert extension_tool_answer(drive, "mailer", "token") == "fixture-secret", "the later grant reached it"
    finally:
        clean_extension_runtime_state()
