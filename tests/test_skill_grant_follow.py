"""A key grant follows a local edit of the installed skill that holds it, and nothing else does (owner 22.11 «Упростить»).

Home test of ``ouroboros/skill_grant_follow.py`` at the seam every gate reads (``grant_status_for_skill``): a local
edit keeps the key while a permission waits (use-time) or blocks (load-time); a release awaiting its check, a retained
or landing update, an accepted update, an unreadable record or a key the manifest dropped do not follow; a grant
decided for edited files follows the next edit; a skill made here counts its own decisions. B″: a key saved after the
last decision is not passed and stops nothing, a key the decision covered and did not grant still stops it. A write for
edited files keeps what follows and does not erase a waiting permission.
"""

from __future__ import annotations

import json
import time

import pytest

from ouroboros import config
from ouroboros.skill_install_record import update_backup_path, update_install_record, write_install_record
from ouroboros.skill_loader import find_skill, grant_status_for_skill, load_skill_grants, save_skill_grants

HUB = {"id": "hub-fixture", "kind": "hub", "name": "Fixture hub"}


def payload(folder, *, keys=("FIXTURE_KEY",), perms=(), version="1.0.0"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: local fixture\nversion: {version}\ntype: extension\n"
        f'entry: plugin.py\nplugin_api: "2.0"\npermissions: [{", ".join(perms)}]\n'
        f'env_from_settings: [{", ".join(keys)}]\n---\nFixture.\n', encoding="utf-8")


def edit(folder, text="edited"):
    (folder / "notes.md").write_text(f"{text}\n", encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: False)
    settings(drive, FIXTURE_KEY="example")
    return drive


def settings(drive, **entries):
    (drive / "settings.json").write_text(json.dumps(entries), encoding="utf-8")


def grant(drive, keys=("FIXTURE_KEY",), perms=(), name="demo"):
    skill = find_skill(drive, name)
    save_skill_grants(drive, name, list(keys), content_hash=skill.content_hash, requested_keys=list(keys),
                      granted_permissions=list(perms), requested_permissions=list(perms))
    return skill


def installed(drive, *, keys=("FIXTURE_KEY",), perms=(), granted=None):
    """A hub payload with its install record (trusted release = its bytes) and the owner's grant of that release."""
    folder = drive / "skills" / "ouroboroshub" / "demo"
    payload(folder, keys=keys, perms=perms)
    digest = find_skill(drive, "demo").content_hash
    write_install_record(drive, "demo", {"source": dict(HUB), "release": {"version": "1.0.0", "digest": digest},
                                         "basis": "trusted", "official": False, "installed_by": "owner",
                                         "turn_on_when_ready": False})
    grant(drive, keys if granted is None else granted, perms)
    return folder


def state(drive, name="demo"):
    return grant_status_for_skill(drive, find_skill(drive, name))


@pytest.mark.parametrize(("perm", "blocks"), [("notify_owner", False), ("page_module", True)])
def test_a_local_edit_keeps_the_key_and_the_permission_waits_or_blocks(root, perm, blocks):
    folder = installed(root, perms=(perm,))
    unedited = state(root)
    assert (unedited["granted_keys"], unedited["granted_permissions"], unedited["blocked"]) == (
        ["FIXTURE_KEY"], [perm], False)
    record = (root / "state" / "skills" / "demo" / "grants.json").read_bytes()
    edit(folder)
    edited = state(root)
    assert edited["granted_keys"] == ["FIXTURE_KEY"], "the key follows the local edit"
    assert edited["granted_permissions"] == [] and edited["missing_permissions"] == [perm], "a permission never follows"
    assert edited["earlier_grant"] == {"keys": [], "permissions": [perm]}
    # A use-time permission waits and stops nothing; one acting through what the load publishes blocks, as before.
    assert edited["waiting_permissions"] == ([] if blocks else [perm]) and edited["blocked"] is blocks
    assert (root / "state" / "skills" / "demo" / "grants.json").read_bytes() == record, "reading wrote nothing"


@pytest.mark.parametrize("case", ["pending_update", "update_backup", "rollback_copy", "update_landed",
                                  "unreadable_record", "undeclared"])
def test_nothing_that_installs_or_waits_is_a_local_edit(root, case):
    folder = installed(root)
    edit(folder)
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]  # the same edit follows before the case
    (folder.parent / ".rollback").mkdir()
    (folder.parent / ".rollback" / "demo.previous").mkdir()  # a finished landing's kept payload is no landing
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]
    if case == "pending_update":  # a release of a source that reviews each skill awaits its check (D-A3)
        update_install_record(root, "demo", pending_update={"version": "2.0.0", "digest": "f" * 64})
    elif case == "update_backup":  # retained before an update replaces any byte, kept through an interrupted landing
        update_backup_path(root, "demo").write_text("{}", encoding="utf-8")
    elif case == "rollback_copy":  # the installer's copy of the payload made before new files are published
        (folder.parent / ".rollback" / f"demo.{'a' * 32}").mkdir()
    elif case == "update_landed":  # an accepted update lands after the grant: the grant is not this release's
        time.sleep(0.01)
        write_install_record(root, "demo", {"source": dict(HUB), "release": {"version": "2.0.0", "digest": "f" * 64},
                                            "basis": "trusted", "official": False, "installed_by": "owner",
                                            "turn_on_when_ready": False})
    elif case == "unreadable_record":
        (root / "state" / "skills" / "demo" / "install.json").write_text("{not json", encoding="utf-8")
    else:  # the manifest no longer declares the key
        payload(folder, keys=("OTHER_KEY",))
    status = state(root)
    assert status["granted_keys"] == [] and "FIXTURE_KEY" not in status["pending_keys"]


def test_a_grant_decided_for_edited_files_follows_the_next_edit(root):
    """Edit, check, edit: a grant written for the edited files (the owner's, or automatic after Ouroboros's review)
    is this installation's decision, so the next edit keeps the key."""
    folder = installed(root)
    edit(folder, "first edit")
    grant(root)  # decided for the edited bytes, after the release landed
    edit(folder, "second edit")
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]


def test_a_skill_made_here_counts_its_own_decisions(root):
    folder = root / "skills" / "external" / "demo"
    payload(folder)
    grant(root)
    edit(folder)
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"], "an own skill without a marker: its state folder is its own"
    # A self-authored skill created after that grant (a new skill reusing the name) does not inherit it ...
    mark = {"schema_version": 1, "origin": "self_authored", "task_id": "t1",
            "created_at": "2099-01-01T00:00:00+00:00"}
    for path in (folder / ".self_authored.json", root / "state" / "skills" / "demo" / "self_authored.json"):
        path.write_text(json.dumps(mark), encoding="utf-8")
    assert find_skill(root, "demo").source == "self_authored"
    assert state(root)["granted_keys"] == []
    # ... while a grant decided after the marker follows its edits.
    mark["created_at"] = "2001-01-01T00:00:00+00:00"
    for path in (folder / ".self_authored.json", root / "state" / "skills" / "demo" / "self_authored.json"):
        path.write_text(json.dumps(mark), encoding="utf-8")
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"]
    # A payload in a source's bucket without an install record binds the exact bytes, as before.
    hub = root / "skills" / "ouroboroshub" / "other"
    payload(hub)
    grant(root, name="other")
    edit(hub)
    assert state(root, "other")["granted_keys"] == []


def test_a_key_saved_after_the_decision_stops_nothing_and_a_covered_one_still_does(root):
    """B″ across a restart: the stored ``requested_keys`` of the last decision decide, nothing in memory."""
    from ouroboros.skill_readiness import skill_readiness_for_execution

    folder = installed(root, keys=("FIXTURE_KEY", "LATER_KEY", "OPENAI_API_KEY"), granted=("FIXTURE_KEY",))
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    for _restart in range(2):  # a fresh discovery each time: the rule reads only the stored decision
        status = state(root)
        assert status["granted_keys"] == ["FIXTURE_KEY"] and status["pending_keys"] == ["LATER_KEY"]
        assert status["missing_keys"] == ["LATER_KEY", "OPENAI_API_KEY"]
        assert status["blocked"] is True, "a core key requested and not granted still blocks"
    payload(folder, keys=("FIXTURE_KEY", "LATER_KEY"))
    status = state(root)
    assert (status["pending_keys"], status["blocked"], status["all_granted"]) == (["LATER_KEY"], False, False)
    notes = skill_readiness_for_execution(root, find_skill(root, "demo"), require_enabled=False).notes
    assert notes and notes[0].startswith("key LATER_KEY not granted")
    save_skill_grants(root, "demo", ["FIXTURE_KEY"], content_hash=find_skill(root, "demo").content_hash,
                      requested_keys=["FIXTURE_KEY", "LATER_KEY"])  # a decision covering LATER_KEY, not granting it
    assert "LATER_KEY" in load_skill_grants(root, "demo")["requested_keys"]
    status = state(root)
    assert (status["pending_keys"], status["blocked"]) == ([], True), "a covered key not granted blocks, as before"
    (root / "state" / "skills" / "demo" / "grants.json").unlink()
    status = state(root)
    assert status["pending_keys"] == ["FIXTURE_KEY", "LATER_KEY"] and status["granted_keys"] == []
    assert status["blocked"] is False, "without a decision a newly present custom key stops nothing either"


def test_a_grant_write_for_edited_files_keeps_what_follows(root):
    folder = installed(root, keys=("FIXTURE_KEY", "LATER_KEY"), perms=("notify_owner",))
    accepted = find_skill(root, "demo").content_hash
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    edit(folder)
    current = find_skill(root, "demo").content_hash
    # The owner grants the pending key alone (``_grant``'s call): the record of the accepted bytes stays.
    save_skill_grants(root, "demo", ["LATER_KEY"], content_hash=current, requested_keys=["FIXTURE_KEY", "LATER_KEY"],
                      granted_permissions=[], requested_permissions=["notify_owner"])
    stored = load_skill_grants(root, "demo")
    assert stored["content_hash"] == accepted and stored["granted_permissions"] == ["notify_owner"]
    status = state(root)
    assert status["granted_keys"] == ["FIXTURE_KEY", "LATER_KEY"] and status["waiting_permissions"] == ["notify_owner"]
    # Granting the permission for the edited files writes their record and keeps the keys that follow.
    save_skill_grants(root, "demo", [], content_hash=current, requested_keys=["FIXTURE_KEY", "LATER_KEY"],
                      granted_permissions=["notify_owner"], requested_permissions=["notify_owner"])
    assert load_skill_grants(root, "demo")["content_hash"] == current
    status = state(root)
    assert (status["granted_keys"], status["granted_permissions"], status["blocked"]) == (
        ["FIXTURE_KEY", "LATER_KEY"], ["notify_owner"], False)
    # Where nothing follows (an update is retained), a write for other bytes keeps nothing of the old record.
    update_backup_path(root, "demo").write_text("{}", encoding="utf-8")
    edit(folder, "again")
    save_skill_grants(root, "demo", [], content_hash=find_skill(root, "demo").content_hash,
                      requested_keys=["FIXTURE_KEY", "LATER_KEY"], granted_permissions=[], requested_permissions=[])
    assert load_skill_grants(root, "demo")["granted_keys"] == []


def test_only_what_blocks_refuses_a_script():
    from ouroboros.tools.skill_exec import _grant_refusal

    waiting = {"missing_keys": ["LATER_KEY"], "pending_keys": ["LATER_KEY"],
               "missing_permissions": ["notify_owner"], "waiting_permissions": ["notify_owner"]}
    assert _grant_refusal("demo", waiting) == ""
    blocking = {**waiting, "missing_keys": ["LATER_KEY", "OPENAI_API_KEY"]}
    refusal = _grant_refusal("demo", blocking)
    assert "SKILL_EXEC_GRANT_REQUIRED" in refusal and "OPENAI_API_KEY" in refusal and "LATER_KEY" not in refusal


def test_a_key_granted_later_reaches_an_extension_already_loaded_in_process(tmp_path, monkeypatch):
    """A key saved later is not passed and unloads nothing (B″); once the owner grants it, the next reconcile reloads
    the in-process extension, whose ``get_settings`` then sees the key."""
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
        reconcile = lambda: extension_loader.reconcile_extension(  # noqa: E731
            "mailer", drive, lambda: dict(values), repo_path=str(repo))
        assert reconcile()["live_loaded"]
        values["MAILER_TOKEN"] = "fixture-secret"
        assert reconcile()["live_loaded"], "a key saved later unloads nothing"
        assert extension_tool_answer(drive, "mailer", "token") == "none", "and it is not passed"
        current = find_skill(drive, "mailer", repo_path=str(repo))
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=current.content_hash,
                          requested_keys=["MAILER_TOKEN"])
        assert reconcile()["live_loaded"]
        assert extension_tool_answer(drive, "mailer", "token") == "fixture-secret", "the later grant reached it"
    finally:
        clean_extension_runtime_state()


def test_the_secret_card_says_whether_a_reader_keeps_running_without_the_key():
    """The readers sentence follows the host's ``blocked``: a key the last decision never covered leaves the enabled
    skill running without it; a missing key that stops it says so."""
    from ouroboros.secret_requests import _consumer_line

    row = {"name": "mailer", "enabled": True, "reads": ["MAILER_TOKEN"], "granted": []}
    assert "keeps running without them until they are granted" in _consumer_line({"consumers": [{**row, "blocked": False}]})
    assert "cannot run until the keys are granted" in _consumer_line({"consumers": [{**row, "blocked": True}]})
