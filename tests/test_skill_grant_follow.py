"""Grant binding and B″, through ordinary skill operations with synthetic settings.

A grant names the files it covers. Accepting a version without automatic grants does not
grant its keys. On accepted current files a later-saved custom key is not passed and does
not block by itself; a covered but denied key still blocks.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import config
from ouroboros.skill_install_record import update_install_record, write_install_record
from ouroboros.skill_loader import (SkillReviewState, find_skill, grant_status_for_skill, load_skill_grants,
                                    save_review_state, save_skill_grants)

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


def record(drive, digest, *, basis="trusted", installed_at="2026-06-01T00:00:00+00:00", **extra):
    """The install record of the landed release ``digest`` (times explicit)."""
    write_install_record(drive, "demo", {"source": dict(HUB), "release": {"version": "1.0.0", "digest": digest},
                                         "basis": basis, "official": False, "installed_by": "owner",
                                         "turn_on_when_ready": False, "installed_at": installed_at, **extra})


def installed(drive, *, keys=("FIXTURE_KEY",), perms=(), granted=None):
    """A hub payload with its install record (trusted release = its bytes) and the owner's grant of that release."""
    folder = drive / "skills" / "ouroboroshub" / "demo"
    payload(folder, keys=keys, perms=perms)
    record(drive, find_skill(drive, "demo").content_hash)
    grant(drive, keys if granted is None else granted, perms)
    return folder


def state(drive, name="demo"):
    return grant_status_for_skill(drive, find_skill(drive, name))


def accept(drive, name="demo"):
    """A review's pass of the skill's current files."""
    save_review_state(drive, name, SkillReviewState(status="clean", content_hash=find_skill(drive, name).content_hash))


@pytest.mark.parametrize("location", ["external", "ouroboroshub"])
def test_an_edit_without_a_new_grant_keeps_settings_but_not_the_grant(root, location):
    perm = "notify_owner"
    if location == "ouroboroshub":
        folder = installed(root, perms=(perm,))
    else:
        folder = root / "skills" / location / "demo"
        payload(folder, perms=(perm,))
        accept(root)
        grant(root, perms=(perm,))
    before = state(root)
    assert (before["granted_keys"], before["granted_permissions"], before["blocked"]) == (
        ["FIXTURE_KEY"], [perm], False)
    grant_file = root / "state" / "skills" / "demo" / "grants.json"
    grant_bytes, values = grant_file.read_bytes(), (root / "settings.json").read_bytes()
    (folder / "plugin.py").write_text("def register(api):\n    return None\n", encoding="utf-8")
    for checked in (False, True):
        if checked:
            accept(root)
        edited = state(root)
        assert edited["granted_keys"] == edited["granted_permissions"] == []
        assert edited["pending_keys"] == [] and edited["blocked"] is True
        assert edited["earlier_grant"] == {"keys": ["FIXTURE_KEY"], "permissions": [perm]}
    assert grant_file.read_bytes() == grant_bytes, "reading never moved the old grant"
    assert (root / "settings.json").read_bytes() == values, "the Settings value was not removed"
    grant(root, perms=(perm,))
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"] and state(root)["granted_permissions"] == [perm]
    assert state(root)["blocked"] is False


def test_a_strict_update_that_adds_a_key_refuses_until_its_release_is_accepted(root, monkeypatch):
    """A01 (D-A3), Cyber Pro: a release of a source that reviews each skill adds a key already saved for another
    integration. Until its check (or the owner's Skip) the key blocks, as on the base; after it, B″ applies."""
    from ouroboros.tools.skill_exec import _grant_refusal

    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    folder = root / "skills" / "ouroboroshub" / "demo"
    payload(folder, keys=())
    record(root, find_skill(root, "demo").content_hash, basis="review")
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved for another integration")
    payload(folder, keys=("LATER_KEY",), version="2.0.0")
    landed = find_skill(root, "demo").content_hash
    record(root, landed, basis="review", installed_at="2026-07-01T00:00:00+00:00",
           pending_update={"version": "2.0.0", "digest": landed})
    status = state(root)
    assert (status["pending_keys"], status["blocked"], status["usable"]) == ([], True, False)
    assert "LATER_KEY" in _grant_refusal("demo", status)
    update_install_record(root, "demo", pending_update=None)  # landed, not yet reviewed or skipped
    assert (state(root)["pending_keys"], state(root)["blocked"]) == ([], True), "a landed update before its check"
    update_install_record(root, "demo", basis="owner_skip")  # the owner's Skip for these bytes
    status = state(root)
    assert (status["pending_keys"], status["blocked"], status["usable"]) == (["LATER_KEY"], False, True)
    assert _grant_refusal("demo", status) == ""


@pytest.mark.parametrize("acceptance", ["review", "skip"])
def test_a_reseeded_builtin_with_a_new_key_refuses_until_its_new_version_is_accepted(root, tmp_path, monkeypatch,
                                                                                    acceptance):
    """R2-A01, Cyber Pro: an app update re-seeds an accepted built-in without keys to a version declaring a custom key
    already saved for another integration («review» built-in policy, automatic grants off). Until the new version's
    review or the owner's Skip the key blocks, as on the base; after either, B″: it runs without the key."""
    from ouroboros.skill_builtin import install_builtin_skill, reseed_changed_builtins
    from ouroboros.tools.skill_exec import _grant_refusal

    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    settings(root, LATER_KEY="saved for another integration", OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS="false",
             OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS="false")
    seeds = tmp_path / "seeds"
    payload(seeds / "demo", keys=())
    assert install_builtin_skill("demo", drive_root=root, seed_dir=seeds)["basis"] == "review"
    accept(root)
    payload(seeds / "demo", keys=("LATER_KEY",), version="2.0.0")
    assert reseed_changed_builtins(root, seed_dir=seeds) == 1
    status = state(root)
    assert (status["missing_keys"], status["pending_keys"], status["blocked"], status["usable"]) == (
        ["LATER_KEY"], [], True, False)
    assert "LATER_KEY" in _grant_refusal("demo", status)
    if acceptance == "review":
        accept(root)
    else:
        update_install_record(root, "demo", basis="owner_skip")
    status = state(root)
    assert (status["granted_keys"], status["pending_keys"], status["blocked"], status["usable"]) == (
        [], ["LATER_KEY"], False, True), "accepted: it runs, and the key is not passed"
    assert _grant_refusal("demo", status) == ""


@pytest.mark.parametrize("basis", ["review", "trusted"])
def test_a_landed_update_blocks_a_new_key_until_it_is_granted(root, basis):
    """The earlier release's grant does not reach the update. Its new key blocks until the update is accepted (a
    trusted source's release is accepted as it lands; then B″: the new key alone stops nothing)."""
    folder = installed(root)
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    payload(folder, keys=("FIXTURE_KEY", "LATER_KEY"), version="2.0.0")
    record(root, find_skill(root, "demo").content_hash, basis=basis, installed_at="2099-01-01T00:00:00+00:00")
    status = state(root)
    assert (status["granted_keys"], status["pending_keys"], status["blocked"]) == (
        [], [] if basis == "review" else ["LATER_KEY"], True), "the earlier decision's key blocks in both"
    grant(root, keys=("FIXTURE_KEY", "LATER_KEY"))  # the owner's grant of the landed release
    status = state(root)
    assert (status["granted_keys"], status["blocked"]) == (["FIXTURE_KEY", "LATER_KEY"], False)


def test_a_skill_loaded_for_review_keeps_its_place_and_its_key(root, tmp_path, monkeypatch):
    """A04: the review's pinned binding sees the selected folder's location and exact grant, not earlier grants."""
    from ouroboros.skill_review_runner import _load_binding_skill
    from ouroboros.tool_access_types import ResolvedResourceBinding

    repo = tmp_path / "my-skills"
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))

    def bound(path, source, name):
        return _load_binding_skill(ResolvedResourceBinding(
            profile="self_modification", root="skill_payload", operation="review", base_path=path,  # type: ignore[arg-type]
            target_path=path, source=source, skill_name=name, state_drive_root=root))

    for folder, source in ((repo / "demo", "user_repo"), (root / "skills" / "ouroboroshub" / "other", "ouroboroshub")):
        payload(folder)
        grant(root, name=folder.name)
        loaded = bound(folder, source, folder.name)
        assert loaded.location == source
        assert grant_status_for_skill(root, loaded)["granted_keys"] == ["FIXTURE_KEY"]
        edit(folder)
        loaded = bound(folder, source, folder.name)
        assert loaded.location == source
        assert grant_status_for_skill(root, loaded)["granted_keys"] == []


def test_a_key_saved_after_the_decision_stops_nothing_and_a_covered_one_still_does(root):
    """B″ across a restart: the stored ``requested_keys`` of the last decision decide, nothing in memory."""
    from ouroboros.skill_readiness import skill_readiness_for_execution

    folder = installed(root, keys=("FIXTURE_KEY", "LATER_KEY"), granted=("FIXTURE_KEY",))
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    for _restart in range(2):  # a fresh discovery each time: the rule reads only the stored decision
        status = state(root)
        assert status["granted_keys"] == ["FIXTURE_KEY"] and status["pending_keys"] == ["LATER_KEY"]
        assert status["missing_keys"] == ["LATER_KEY"]
        assert (status["blocked"], status["all_granted"]) == (False, False)
    notes = skill_readiness_for_execution(root, find_skill(root, "demo"), require_enabled=False).notes
    assert notes and "LATER_KEY is saved in Settings but not granted" in notes[0]
    save_skill_grants(root, "demo", ["FIXTURE_KEY"], content_hash=find_skill(root, "demo").content_hash,
                      requested_keys=["FIXTURE_KEY", "LATER_KEY"])  # a decision covering LATER_KEY, not granting it
    assert "LATER_KEY" in load_skill_grants(root, "demo")["requested_keys"]
    status = state(root)
    assert (status["pending_keys"], status["blocked"]) == ([], True), "a covered key not granted blocks, as before"
    edit(folder)
    (root / "state" / "skills" / "demo" / "grants.json").unlink()
    status = state(root)
    assert (status["pending_keys"], status["blocked"]) == ([], True), "files no one accepted keep the base rule"
    accept(root)
    status = state(root)
    assert status["pending_keys"] == ["FIXTURE_KEY", "LATER_KEY"] and status["granted_keys"] == []
    assert status["blocked"] is False, "accepted files without a decision: a newly present custom key stops nothing"


def test_enable_with_the_offered_key_leaves_a_key_saved_later_pending(tmp_path, monkeypatch):
    """R2-A03: an owner grant decision records as requested only the keys it decided. Enable on a disabled reviewed
    skill offers the one key that blocks (a core key); confirming it turns the skill on, while the key saved later
    stays pending and is not passed until the owner grants it as well."""
    from ouroboros import skill_lifecycle_actions as actions
    from ouroboros.skill_grant_follow import blocking_lists
    from ouroboros.skill_loader import load_enabled
    from tests._extension_loader_shared import _write_ext_skill, extension_tool_answer
    from tests._shared import clean_extension_runtime_state

    drive = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    plugin = ("def register(api):\n    api.register_tool('token', lambda ctx: ('present' if api.get_settings(['LATER_KEY']).get('LATER_KEY') == 'saved later'"
              " else 'absent'), description='token', schema={})\n")
    _write_ext_skill(drive / "skills" / "external", "demo", plugin_body=plugin, permissions=["tool", "read_settings"],
                     env_from_settings=["FIXTURE_KEY", "OPENAI_API_KEY", "LATER_KEY"])
    settings(drive, FIXTURE_KEY="example")
    accept(drive)
    grant(drive)  # an earlier decision for these bytes: FIXTURE_KEY
    settings(drive, FIXTURE_KEY="example", LATER_KEY="saved later")
    status = state(drive)
    assert status["pending_keys"] == ["LATER_KEY"] and blocking_lists(status) == (["OPENAI_API_KEY"], [])
    clean_extension_runtime_state()
    try:
        result = actions._grant(find_skill(drive, "demo"), drive, "", ["OPENAI_API_KEY"], enable_after_grant=True)
        assert result["ok"] and result.get("enabled") is True and not result.get("setup_error"), result
        assert load_enabled(drive, "demo") is True
        assert load_skill_grants(drive, "demo")["requested_keys"] == ["FIXTURE_KEY", "OPENAI_API_KEY"]
        status = state(drive)
        assert (status["granted_keys"], status["pending_keys"], status["blocked"]) == (
            ["FIXTURE_KEY", "OPENAI_API_KEY"], ["LATER_KEY"], False)
        assert extension_tool_answer(drive, "demo", "token") == "absent", "the key saved later is not passed"
        actions._grant(find_skill(drive, "demo"), drive, "", ["LATER_KEY"])  # the owner's Grant access for it
        assert state(drive)["granted_keys"] == ["FIXTURE_KEY", "OPENAI_API_KEY", "LATER_KEY"]
        assert extension_tool_answer(drive, "demo", "token") == "present", "granted, it is passed"
    finally:
        clean_extension_runtime_state()


def test_a_partial_grant_for_edited_files_records_only_the_approved_items(root):
    folder = installed(root, keys=("FIXTURE_KEY", "LATER_KEY"), perms=("notify_owner",))
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    edit(folder)
    current = find_skill(root, "demo").content_hash
    accept(root)
    # A decision on edited files records those bytes, not the installation's old key or permission.
    save_skill_grants(root, "demo", ["LATER_KEY"], content_hash=current, requested_keys=["FIXTURE_KEY", "LATER_KEY"],
                      granted_permissions=[], requested_permissions=["notify_owner"])
    stored = load_skill_grants(root, "demo")
    assert (stored["content_hash"], stored["granted_keys"], stored["granted_permissions"]) == (
        current, ["LATER_KEY"], [])
    status = state(root)
    assert status["granted_keys"] == ["LATER_KEY"] and status["missing_permissions"] == ["notify_owner"]
    save_skill_grants(root, "demo", ["FIXTURE_KEY"], content_hash=current, requested_keys=["FIXTURE_KEY", "LATER_KEY"],
                      granted_permissions=["notify_owner"], requested_permissions=["notify_owner"])
    status = state(root)
    assert (status["granted_keys"], status["granted_permissions"], status["blocked"]) == (
        ["FIXTURE_KEY", "LATER_KEY"], ["notify_owner"], False)


@pytest.mark.parametrize("auto_grant", [False, True])
def test_a_grant_on_edited_files_reaches_those_files_only(root, monkeypatch, auto_grant):
    """A02: the owner's Grant access on B records B, as on the base. A reinstall of exact A does not get the key
    granted only to B (automatic grants, when on, decide for A by themselves); an install of exact B keeps it."""
    from ouroboros.skill_loader import auto_grant_if_enabled

    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true" if auto_grant else "false")
    folder = installed(root, keys=("FIXTURE_KEY", "LATER_KEY"))  # A: FIXTURE_KEY granted, LATER_KEY not yet saved
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    edit(folder)
    edited = find_skill(root, "demo").content_hash
    save_skill_grants(root, "demo", ["LATER_KEY"], content_hash=edited, requested_keys=["FIXTURE_KEY", "LATER_KEY"],
                      granted_permissions=[], requested_permissions=[])
    assert load_skill_grants(root, "demo")["content_hash"] == edited
    record(root, edited, installed_at="2099-01-01T00:00:00+00:00")  # exact B installed again
    assert state(root)["granted_keys"] == ["LATER_KEY"]
    (folder / "notes.md").unlink()  # exact A installed again
    record(root, find_skill(root, "demo").content_hash, installed_at="2099-02-01T00:00:00+00:00")
    auto_grant_if_enabled(root, find_skill(root, "demo"))
    status = state(root)
    assert status["granted_keys"] == (["FIXTURE_KEY", "LATER_KEY"] if auto_grant else [])
    assert load_skill_grants(root, "demo")["content_hash"] == (find_skill(root, "demo").content_hash if auto_grant
                                                               else edited)


def test_an_installed_builtin_rebinds_once_and_reinstalling_exact_bytes_keeps_the_grant_file(root, tmp_path):
    from ouroboros.skill_builtin import install_builtin_skill

    settings(root, FIXTURE_KEY="example", OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS="true",
             OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS="false")
    seeds = tmp_path / "seeds"
    payload(seeds / "demo", perms=("notify_owner",))
    assert install_builtin_skill("demo", drive_root=root, seed_dir=seeds)["ok"]
    grant(root, perms=("notify_owner",))
    payload(seeds / "demo", perms=("notify_owner",), version="2.0.0")
    first = install_builtin_skill("demo", drive_root=root, seed_dir=seeds)
    assert first["ok"] and first["grants_carried"]
    stored = root / "state" / "skills" / "demo" / "grants.json"
    before = (stored.read_bytes(), stored.stat().st_mtime_ns)
    repeated = install_builtin_skill("demo", drive_root=root, seed_dir=seeds)
    assert repeated["ok"] and not repeated["grants_carried"]
    assert (stored.read_bytes(), stored.stat().st_mtime_ns) == before
    assert state(root)["granted_keys"] == ["FIXTURE_KEY"] and state(root)["granted_permissions"] == ["notify_owner"]


def test_a_pending_key_never_reaches_a_registration_child(root):
    """A06: the declared catalog inputs (``catalog_inputs.settings``) a registration child receives pass the same grant
    decision as the PluginAPI: a pending key stays out until the owner grants it; a public setting goes as it reads."""
    import pathlib

    from ouroboros.extension_process_runner import _catalog_context

    folder = root / "skills" / "external" / "demo"
    payload(folder, keys=("LATER_KEY",))
    manifest = (folder / "SKILL.md").read_text(encoding="utf-8").replace(
        "\n---\n", "\ncatalog_inputs:\n  settings: [LATER_KEY, OUROBOROS_UI_LANGUAGE]\n  environment: []\n---\n", 1)
    (folder / "SKILL.md").write_text(manifest, encoding="utf-8")
    settings(root, FIXTURE_KEY="example", LATER_KEY="harmless-value", OUROBOROS_UI_LANGUAGE="en")
    accept(root)
    repo = pathlib.Path(__file__).resolve().parents[1]
    sent = _catalog_context(find_skill(root, "demo"), root, repo)[2]
    assert state(root)["pending_keys"] == ["LATER_KEY"]
    assert sent == {"OUROBOROS_UI_LANGUAGE": "en"}, "the pending key stays out of the child's request"
    grant(root, keys=("LATER_KEY",))
    assert _catalog_context(find_skill(root, "demo"), root, repo)[2] == {
        "LATER_KEY": "harmless-value", "OUROBOROS_UI_LANGUAGE": "en"}


def test_a_registration_child_holds_public_rows_without_any_saved_secret(root):
    """Both registration inputs get public identity/status rows and no structured credential values."""
    import pathlib

    from ouroboros.extension_process_runner import _catalog_context

    secrets = ["tok-AAAA1111", "hdr-BBBB2222", "env-CCCC3333", "usr-DDDD4444", "key-EEEE5555"]
    mcp = [{"id": "remote", "name": "Remote", "transport": "streamable_http", "auth_token": secrets[0],
            "url": f"https://fixture:{secrets[3]}@mcp.example.test/sse", "headers": {"X-Fixture": secrets[1]}},
           {"id": "local", "name": "Local", "transport": "stdio", "command": "fixture-server",
            "env": {"FIXTURE_ENV": secrets[2]}}]
    rows = [{"kind": "openai-compatible", "name": "lab", "api_key": secrets[4], "base_url": "https://llm.example.test/v1"}]
    settings(root, FIXTURE_KEY="example", OUROBOROS_UI_LANGUAGE="en", MCP_SERVERS=mcp, OUROBOROS_CONNECTIONS=rows)
    folder = root / "skills" / "external" / "demo"
    payload(folder, keys=("OUROBOROS_UI_LANGUAGE", "MCP_SERVERS", "OUROBOROS_CONNECTIONS"), perms=("read_settings",))
    accept(root)
    grant(root, keys=("OUROBOROS_CONNECTIONS",))
    repo = pathlib.Path(__file__).resolve().parents[1]
    readable = _catalog_context(find_skill(root, "demo"), root, repo)[2]
    manifest = (folder / "SKILL.md").read_text(encoding="utf-8").replace("\n---\n", (
        "\ncatalog_inputs:\n  settings: [OUROBOROS_UI_LANGUAGE, MCP_SERVERS, OUROBOROS_CONNECTIONS]\n"
        "  environment: []\n---\n"), 1)
    (folder / "SKILL.md").write_text(manifest, encoding="utf-8")
    accept(root)
    grant(root, keys=("OUROBOROS_CONNECTIONS",))
    declared = _catalog_context(find_skill(root, "demo"), root, repo)[2]
    for sent in (readable, declared):
        assert not any(secret[:8] in json.dumps(sent) for secret in secrets), "not even a prefix of a saved secret"
        assert sent["OUROBOROS_UI_LANGUAGE"] == "en"
        assert sent["MCP_SERVERS"] == [
            {"id": "remote", "name": "Remote", "transport": "streamable_http", "auth_configured": True},
            {"id": "local", "name": "Local", "transport": "stdio", "auth_configured": False},
        ]
        assert sent["OUROBOROS_CONNECTIONS"] == [{"kind": "openai-compatible", "name": "lab"}]


def test_structured_settings_without_a_public_projection_are_omitted():
    from ouroboros.skill_grant_follow import registration_view

    assert registration_view({"OUROBOROS_UI_LANGUAGE": "en", "PRIVATE_FIXTURE": {"value": "example"},
                              "OTHER_FIXTURE": ["example"], "MCP_SERVERS": {}}) == {"OUROBOROS_UI_LANGUAGE": "en"}
    assert registration_view({"MCP_SERVERS": []}) == {"MCP_SERVERS": []}


def test_grant_continuity_intersects_current_grants_and_requests():
    from ouroboros.skill_grant_follow import _continuing_keys

    receipt = {"digest": "a" * 64, "keys": ["KEPT", "REVOKED", "NO_LONGER_REQUESTED"]}
    grants = {"content_hash": receipt["digest"], "granted_keys": ["KEPT", "NEW", "NO_LONGER_REQUESTED"]}
    requested = ["KEPT", "REVOKED", "NEW"]
    assert _continuing_keys(receipt, grants, requested) == ["KEPT"]
    assert _continuing_keys(receipt, {**grants, "content_hash": "b" * 64}, requested) == []


def test_the_grant_receipt_is_the_hosts_and_follows_only_its_installation(root):
    """The receipt of an explicit grant lives outside the skill's state directory, closed to ordinary agent writes,
    names only keys saved for these files, continues only from its last digest on its folder, and a local delete
    ends it."""
    from ouroboros.contracts.skill_payload_policy import skill_grant_receipts_path, skill_state_dir_path
    from ouroboros.skill_grant_follow import (RECEIPTS_FILENAME, extend_verified_edit, installation_of, load_receipt,
                                              receipt_tip, record_explicit_grant)
    from ouroboros.skill_uninstall_state import delete_local_skill

    payload(root / "skills" / "external" / "demo")
    skill = find_skill(root, "demo")
    record_explicit_grant(root, skill, ["FIXTURE_KEY"], via="owner_ui")
    assert load_receipt(root, "demo") == {}, "a grant not saved for these files is no receipt"
    grant(root)
    record_explicit_grant(root, skill, ["FIXTURE_KEY", "NOT_GRANTED_KEY"], via="owner_ui")
    installation = installation_of(root, "demo", skill.skill_dir)
    receipt = load_receipt(root, "demo")
    assert (receipt["keys"], receipt["digest"], receipt["installation"]) == (["FIXTURE_KEY"], skill.content_hash,
                                                                            installation)
    assert skill_grant_receipts_path(root).is_file()
    assert not (skill_state_dir_path(root, "demo") / RECEIPTS_FILENAME).exists()
    first, second, third = skill.content_hash, "b" * 64, "c" * 64
    elsewhere = {**installation, "directory": [installation["directory"][0], installation["directory"][1] + 1]}
    assert not extend_verified_edit(root, "demo", elsewhere, first, second), "another folder"
    assert not extend_verified_edit(root, "demo", installation, second, third), "not from the receipt's last digest"
    assert extend_verified_edit(root, "demo", installation, first, second)
    assert receipt_tip(root, "demo", installation) == second
    assert not extend_verified_edit(root, "demo", installation, first, third), "the chain moved on"
    assert extend_verified_edit(root, "demo", installation, second, first) and "tip" not in load_receipt(root, "demo")
    assert delete_local_skill(root, find_skill(root, "demo"))["ok"]
    assert load_receipt(root, "demo") == {}, "the installation ended with its folder"


@pytest.mark.parametrize("perm,earlier,held", [("notify_owner", True, True), ("notify_owner", False, False),
                                               ("page_module", True, False)],
                         ids=["held-notice", "never-granted", "other-permission"])
def test_verified_repairs_hold_only_a_use_checked_permission_the_earlier_files_had(root, perm, earlier, held):
    """Two verified Repairs of an extension rebind the explicit key and grant no permission. A ``notify_owner`` the
    first files had is held along the receipt's chain: it stops nothing and stays ungranted. One never granted, or
    another permission, blocks as before; a withdrawn decision ends the hold."""
    from ouroboros.skill_grant_follow import (
        extend_verified_edit,
        finish_local_edit_grants,
        installation_of,
        record_explicit_grant,
    )

    folder = root / "skills" / "ouroboroshub" / "demo"
    payload(folder, perms=(perm,))
    record(root, find_skill(root, "demo").content_hash)
    before = grant(root, perms=(perm,) if earlier else ())
    record_explicit_grant(root, before, ["FIXTURE_KEY"], via="owner_ui")
    installation, before = installation_of(root, "demo", folder), before.content_hash
    for text in ("first repair", "second repair"):
        edit(folder, text)
        after = find_skill(root, "demo").content_hash
        assert extend_verified_edit(root, "demo", installation, before, after)
        accept(root)
        finish_local_edit_grants(root, find_skill(root, "demo"))
        status, before = state(root), after
        assert (status["granted_keys"], status["granted_permissions"]) == (["FIXTURE_KEY"], [])
        assert (status["pending_permissions"], status["blocked"]) == (([perm], False) if held else ([], True))
    record_explicit_grant(root, find_skill(root, "demo"), ["FIXTURE_KEY"], via="owner_ui")  # adds, withdraws nothing
    assert state(root)["pending_permissions"] == ([perm] if held else [])
    (root / "state" / "skills" / "demo" / "grants.json").unlink()  # the owner's decision withdrawn
    withdrawn = state(root)
    assert (withdrawn["granted_permissions"], withdrawn["pending_permissions"], withdrawn["blocked"]) == ([], [], True)


def test_a_loaded_extension_reports_the_key_saved_but_not_granted(root):
    """F2: the extension's liveness carries the same neutral fact a run does, and nothing once the key is granted."""
    from ouroboros.extension_liveness import runtime_state_for_loaded_skill

    payload(root / "skills" / "external" / "demo", keys=("FIXTURE_KEY", "LATER_KEY"))
    accept(root)
    grant(root)
    settings(root, FIXTURE_KEY="example", LATER_KEY="saved later")
    note = runtime_state_for_loaded_skill(find_skill(root, "demo"), root)["grant_note"]
    assert note == "key LATER_KEY is saved in Settings but not granted to this skill; it was not passed"
    grant(root, keys=("FIXTURE_KEY", "LATER_KEY"))
    assert runtime_state_for_loaded_skill(find_skill(root, "demo"), root)["grant_note"] == ""


def test_only_what_blocks_refuses_a_script():
    from ouroboros.tools.skill_exec import _grant_refusal

    pending = {"missing_keys": ["LATER_KEY"], "pending_keys": ["LATER_KEY"]}
    assert _grant_refusal("demo", pending) == ""
    blocking = {**pending, "missing_keys": ["LATER_KEY", "OPENAI_API_KEY"], "missing_permissions": ["notify_owner"]}
    refusal = _grant_refusal("demo", blocking)
    assert all(word in refusal for word in ("SKILL_EXEC_GRANT_REQUIRED", "OPENAI_API_KEY", "notify_owner"))
    assert "LATER_KEY" not in refusal


def test_a_key_granted_later_reaches_an_extension_already_loaded_in_process(tmp_path, monkeypatch):
    """A key saved later is not passed and unloads nothing (B″); once the owner grants it, the next reconcile reloads
    the in-process extension, whose ``get_settings`` then sees the key; a revocation narrows the running API."""
    from ouroboros import extension_loader
    from tests._extension_loader_shared import _prepare_extension, extension_tool_answer
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    plugin = ("def register(api):\n    api.register_tool('token', lambda ctx: "
              "{None: 'absent', 'fixture-secret': 'first', 'replaced-secret': 'replacement'}"
              ".get(api.get_settings(['MAILER_TOKEN']).get('MAILER_TOKEN'), 'unexpected'),"
              " description='token', schema={})\n")
    loaded, repo, drive = _prepare_extension(tmp_path, "mailer", plugin, ["tool", "read_settings"],
                                             env_from_settings=["MAILER_TOKEN"])
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    values: dict = {}
    monkeypatch.setattr(config, "load_settings", lambda *_a, **_k: dict(values))
    try:
        reconcile = lambda: extension_loader.reconcile_extension(  # noqa: E731
            "mailer", drive, lambda: dict(values), repo_path=str(repo))
        assert reconcile()["live_loaded"]
        values["MAILER_TOKEN"] = "fixture-secret"
        assert reconcile()["live_loaded"], "a key saved later unloads nothing"
        assert extension_tool_answer(drive, "mailer", "token") == "absent", "and it is not passed"
        current = find_skill(drive, "mailer", repo_path=str(repo))
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=current.content_hash,
                          requested_keys=["MAILER_TOKEN"])
        assert reconcile()["live_loaded"]
        assert extension_tool_answer(drive, "mailer", "token") == "first", "the later grant reached it"
        # A05: a revocation reaches the running PluginAPI at the next reconcile, also for a value replaced after it.
        (drive / "state" / "skills" / "mailer" / "grants.json").unlink()
        assert reconcile()["live_loaded"], "a revoked custom key unloads nothing (B″)"
        assert extension_tool_answer(drive, "mailer", "token") == "absent", "and it is no longer readable"
        values["MAILER_TOKEN"] = "replaced-secret"
        assert extension_tool_answer(drive, "mailer", "token") == "absent"
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=current.content_hash,
                          requested_keys=["MAILER_TOKEN"])
        assert reconcile()["live_loaded"]
        assert extension_tool_answer(drive, "mailer", "token") == "replacement", "an explicit grant delivers it again"
    finally:
        clean_extension_runtime_state()
