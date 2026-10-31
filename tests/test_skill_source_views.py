"""Source-view regressions: local execution, exact endorsements, withdrawals and shared git caches."""
from __future__ import annotations

import json
import os
import types

import pytest

from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_git as transport
from ouroboros import skill_sources as sources
from ouroboros.skill_install_record import write_install_record
from tests._skill_source_fixtures import HubFixture, builtin_repo, commit_all, git, write_skill


def test_connected_local_folders_load_in_place_through_review(tmp_path, monkeypatch):
    from ouroboros import skill_loader as loader
    from ouroboros.skill_source_install import SkillInstallError, plan_install
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.skill_exec import _handle_skill_exec
    from tests.test_skill_exec import _build_skill

    drive = tmp_path / "data"
    folder = _build_skill(tmp_path / "owner", "local_skill")
    rows, row = sources.with_added_source([], kind="local", path=str(folder.parent), name="Mine", trust="trusted")
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    with monkeypatch.context() as inventory_check:
        inventory_check.setattr("ouroboros.config.load_settings", lambda: pytest.fail("inventory reread settings"))
        inventory = loader._skill_location_inventory(drive)
    assert [(s.name, s.location, s.skill_dir) for s in inventory] == [("local_skill", "user_repo", folder.resolve())]
    item = catalog.catalog_view(drive, rows=rows)["items"][0]
    assert item["installed"]["state"] == "installed"
    loaded = loader.find_skill(drive, "local_skill")
    assert loaded.source == "user_repo" and not loaded.available_for_execution
    loader.save_enabled(drive, "local_skill", True)
    assert loader.list_available_for_execution(drive) == []
    loader.save_review_state(drive, "local_skill", loader.SkillReviewState(status="clean", content_hash=loaded.content_hash))
    assert [s.name for s in loader.list_available_for_execution(drive)] == ["local_skill"]
    result = _handle_skill_exec(ToolContext(repo_dir=tmp_path, drive_root=drive), skill="local_skill", script="hello.py")
    assert "hello from skill" in result
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(sources.without_source(rows, row["id"])))
    assert loader.discover_skills(drive) == [] and folder.is_dir()
    # Settings can connect its own folder even after a source list has been saved.
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(folder.parent))
    assert loader.find_skill(drive, "local_skill").skill_dir == folder.resolve()
    monkeypatch.delenv(sources.SOURCES_SETTING)
    assert loader.find_skill(drive, "local_skill").skill_dir == folder.resolve()
    # Existing no-adopt protection also applies to connected folders.
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    hub = HubFixture(tmp_path / "remote")
    hub.add_hosted("local_skill")
    hub.commit()
    hub_rows = hub.rows()[1:]
    assert not catalog.refresh_source(drive, hub_rows[0]).get("error")
    with pytest.raises(SkillInstallError) as refused:
        plan_install(drive, "ouroboros-community/local_skill", rows=hub_rows, replace=True)
    assert refused.value.code == "name_taken" and folder.is_dir()


@pytest.mark.parametrize("saved_folder", [False, True])
def test_settings_folder_follows_current_path_with_a_saved_list(tmp_path, monkeypatch, saved_folder):
    from ouroboros import skill_loader as loader

    old = write_skill(tmp_path / "old", "old_skill").parent
    new = write_skill(tmp_path / "new", "new_skill").parent
    extra = write_skill(tmp_path / "extra", "extra_skill").parent
    rows, _ = sources.with_added_source([], kind="local", path=str(extra), name="Extra", trust="review")
    if saved_folder:
        rows.append({"id": sources.LOCAL_ID, "name": "Private", "kind": "local", "path": str(old),
                     "trust": "trusted", "confirmed": False})
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(new))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: pytest.fail("inventory reread settings"))
    drive = tmp_path / "data"
    assert {s.name for s in loader.discover_skills(drive)} == {"new_skill", "extra_skill"}
    assert loader.find_skill(drive, "new_skill").skill_dir == (new / "new_skill").resolve()
    effective = sources.load_sources(dict(os.environ))
    linked = sources.source_by_id(effective, sources.LOCAL_ID)
    assert linked["path"] == str(new)
    if saved_folder:
        assert (linked["name"], linked["trust"], linked["confirmed"]) == ("Private", "trusted", False)
    assert {s.name for s in loader.discover_skills(drive, repo_path=str(old))} == {"old_skill", "extra_skill"}
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    assert {s.name for s in loader.discover_skills(drive)} == {"extra_skill"}


def test_local_source_collision_and_duplicate_roots_keep_global_identity(tmp_path, monkeypatch):
    from ouroboros.skill_loader import _skill_location_inventory, skill_identity_collision_names

    first = write_skill(tmp_path / "one", "same").parent
    second = write_skill(tmp_path / "two", "same").parent
    rows = [{"id": name, "name": name, "kind": "local", "path": str(path), "trust": "review"}
            for name, path in (("one", first), ("alias", first), ("two", second))]
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    assert len(_skill_location_inventory(tmp_path / "data")) == 2
    assert skill_identity_collision_names(tmp_path / "data") == frozenset({"same"})


def test_third_party_hub_keeps_entries_without_official_or_recommended(tmp_path):
    hub = HubFixture(tmp_path)
    hub.add_hosted("endorsed", official=True, recommended=True)
    hub.commit()
    snapshot = catalog.parse_hub_tree(hub.work)
    public = hub.rows()[1]
    assert catalog.items_for(public, snapshot)[0]["official"] is True
    assert catalog.items_for(public, snapshot)[0]["recommended"] is True
    third_party = {**public, "id": "third-party"}
    third_party.pop("part")
    item = catalog.items_for(third_party, snapshot)[0]
    assert item["name"] == "endorsed"
    assert item["official"] is False and item["recommended"] is False
    snapshot["official"]["endorsed"]["digest"] = "a" * 64
    assert catalog.items_for(public, snapshot) == []


def test_bundled_releases_are_official_with_their_declared_recommendations(tmp_path):
    snapshot = catalog.builtin_snapshot(builtin_repo(tmp_path))
    row = sources.derived_sources({})[0]
    items = catalog.items_for(row, snapshot)
    assert all(item["official"] for item in items)
    assert {item["name"] for item in items if item["recommended"]} == {"telegram"}


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("snapshot_state", ["endorsed", "other_digest", "no_endorsement", "no_snapshot", "third_party"])
@pytest.mark.parametrize("modified", [False, True])
def test_installed_official_means_exact_current_endorsement(tmp_path, stored, snapshot_state, modified):
    digest = "1" * 64
    release = {"digest": digest, "version": "one"}
    source = {"id": "ouroboros-community", "kind": "hub", "url": "https://github.com/example/hub"}
    write_install_record(tmp_path, "skill", {"source": source, "entry": "skill", "release": release,
                                           "base_digest": digest, "official": stored, "basis": "review"})
    row = {**source, "id": "ouroboros", "part": "official"}
    snapshot = {"entries": [{"name": "skill", **release}], "official": {"skill": {
        "digest": digest if snapshot_state != "other_digest" else "2" * 64}}}
    if snapshot_state == "no_endorsement":
        snapshot["official"] = {}
    if snapshot_state == "no_snapshot":
        snapshot = None
    if snapshot_state == "third_party":
        row.pop("part")
    loaded = types.SimpleNamespace(name="skill", content_hash="3" * 64 if modified else digest)
    result = catalog.install_listing(tmp_path, loaded, {row["id"]: (row, snapshot)})["install"]
    assert result["official"] is (not modified and (stored if snapshot_state == "no_snapshot" else
                                                    snapshot_state == "endorsed"))
    assert result["basis"] == "review"


def test_per_release_withdrawal_is_visible_only_to_holders(tmp_path):
    hub = HubFixture(tmp_path / "remote")
    current = hub.add_hosted("skill", version="two")
    old = "1" * 64
    hub.entries["skill"]["withdrawn_releases"] = [{"digest": old, "version": "one", "reason": "broken"}]
    hub.commit()
    snapshot = catalog.parse_hub_tree(hub.work)
    row = hub.rows()[2]
    drive = tmp_path / "data"
    for digest, withdrawn in ((old, {"version": "one", "reason": "broken"}), (current, None)):
        write_install_record(drive, "skill", {"source": row, "entry": "skill", "basis": "review",
                                             "release": {"digest": digest}, "base_digest": digest})
        loaded = types.SimpleNamespace(name="skill", content_hash=digest)
        facts = catalog.install_listing(drive, loaded, {row["id"]: (row, snapshot)})["install"]
        assert facts["withdrawn"] == withdrawn
    assert catalog.items_for(row, snapshot)[0]["withdrawn"] is None
    snapshot["entries"][0]["withdrawn"] = {"reason": "all versions", "at": "today"}
    assert catalog.install_listing(drive, loaded, {row["id"]: (row, snapshot)})["install"]["withdrawn"] == {
        "reason": "all versions", "at": "today"}


def test_two_paths_share_a_clone_but_keep_distinct_snapshots(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    git("init", "-q", cwd=work)
    write_skill(work / "first", "alpha")
    write_skill(work / "second", "beta")
    commit_all(work)
    drive = tmp_path / "data"
    rows = [{"id": name, "name": name, "kind": "git", "url": str(work), "ref": "main", "path": name,
             "trust": "review", "confirmed": True} for name in ("first", "second")]
    for row in rows:
        assert not catalog.refresh_source(drive, row).get("error")
    assert transport.clone_dir(drive, rows[0]) == transport.clone_dir(drive, rows[1])
    assert [{e["name"] for e in catalog.read_snapshot(drive, row)["entries"]} for row in rows] == [{"alpha"}, {"beta"}]
    for row, name in zip(rows, ("alpha", "beta")):
        snapshot = catalog.read_snapshot(drive, row)
        assert catalog.entry_payload_dir(drive, row, snapshot, name) == transport.clone_dir(drive, row) / row["path"] / name
    shared = {**rows[0], "id": "another", "part": "community"}
    assert catalog.read_snapshot(drive, shared) == catalog.read_snapshot(drive, rows[0])


def test_plain_repository_does_not_silently_stop_at_500_skills(tmp_path):
    for i in range(501):
        write_skill(tmp_path, f"skill_{i:03}")
    snapshot = catalog.parse_plain_tree(tmp_path)
    assert len(snapshot["entries"]) == 501 and snapshot["problems"] == []


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_remote_http_probe_discloses_encryption_without_refusing(tmp_path, monkeypatch, scheme):
    def fetch(target, url, selector):
        write_skill(target, "skill")
        return "1" * 40

    monkeypatch.setattr(transport, "fetch_tree", fetch)
    url = f"{scheme}://git.example.invalid/owner/repo"
    assert sources.validate_git_url(url) == url
    answer = catalog.probe_source(tmp_path, url=url)
    assert answer["kind"] == "git" and answer["found"] == ["skill"]
    assert answer.get("warning") == ("not_encrypted" if scheme == "http" else None)


@pytest.mark.parametrize("url", ["http://u:p@host/repo", "https://u:p@host/repo", "https://u@host/repo",
                                  "http://u@host/repo", "ssh://git:p@host/repo"])
def test_git_addresses_still_refuse_embedded_credentials(url):
    with pytest.raises(sources.SkillSourceError, match="Remove the credentials"):
        sources.validate_git_url(url)


@pytest.mark.parametrize("trust", ["trusted", "review"])
def test_readding_default_public_hub_restores_the_two_source_rows(trust):
    defaults = sources.derived_sources({})
    public = next(row for row in defaults if row.get("part") == "official")
    remaining = [row for row in defaults if row["kind"] != "hub"]
    rows, added = sources.with_added_source(remaining, kind="hub", url=public["url"], name="Hub", trust=trust)
    hubs = [row for row in rows if row["kind"] == "hub"]
    assert {row["part"] for row in hubs} == {"official", "community"}
    assert {(row["part"], row["trust"], row["confirmed"]) for row in hubs} == {
        ("official", "trusted", False), ("community", "review", False)}
    assert added in hubs
    ordinary, _ = sources.with_added_source(remaining, kind="hub", url="https://github.com/another/hub",
                                            name="Another", trust=trust)
    assert len(ordinary) == len(remaining) + 1 and "part" not in ordinary[-1]
    assert ordinary[-1]["trust"] == trust and ordinary[-1]["confirmed"] is True
    # Reconnecting one missing row preserves the surviving row's explicit choice.
    chosen = {**public, "name": "Chosen", "trust": "review", "confirmed": True}
    restored, _ = sources.with_added_source([*remaining, chosen], kind="hub", url=public["url"], name="Hub", trust=trust)
    assert sources.source_by_id(restored, chosen["id"]) == chosen
    assert sources.source_by_id(restored, sources.HUB_COMMUNITY_ID) == next(r for r in hubs if r["part"] == "community")


def test_legacy_converted_hub_install_gains_the_current_endorsement(tmp_path):
    import hashlib
    import shutil
    from ouroboros.skill_install_record import legacy_hub_record
    from ouroboros.skill_loader import find_skill

    hub = HubFixture(tmp_path / "remote")
    digest = hub.add_hosted("skill", official=True)
    hub.commit()
    rows = hub.rows()[1:]
    payload = hub.work / "skills" / "skill"
    record = legacy_hub_record({"source": "ouroboroshub", "slug": "skill", "version": "1.0.0",
                                "catalog_url": str(hub.bare), "files": [{"path": "SKILL.md", "sha256":
                                hashlib.sha256((payload / "SKILL.md").read_bytes()).hexdigest()}]}, rows, payload)
    assert record["official"] is False and record["release"]["digest"] == digest
    drive = tmp_path / "data"
    shutil.copytree(payload, drive / "skills" / "ouroboroshub" / "skill")
    write_install_record(drive, "skill", record)
    snapshot = catalog.parse_hub_tree(hub.work)
    result = catalog.install_listing(drive, find_skill(drive, "skill"), {r["id"]: (r, snapshot) for r in rows})
    assert result["install"]["official"] is True


def test_builtin_install_listing_endorsement_tracks_the_bundled_bytes(tmp_path, monkeypatch):
    from ouroboros.skill_source_install import install_item
    from ouroboros.skill_loader import find_skill

    repo = builtin_repo(tmp_path)
    drive = tmp_path / "data"
    row = sources.derived_sources({})[0]
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps([row]))
    install_item(drive, "builtin/telegram", rows=[row], repo_dir=repo)
    snapshots = catalog.cached_snapshots(drive, [row], repo_dir=repo)
    loaded = find_skill(drive, "telegram")
    assert catalog.install_listing(drive, loaded, snapshots)["install"]["official"] is True
    (loaded.skill_dir / "SKILL.md").write_text("---\nname: telegram\ndescription: changed\n---\nOwner edit", encoding="utf-8")
    assert catalog.install_listing(drive, find_skill(drive, "telegram"), snapshots)["install"]["official"] is False


def test_listing_serves_check_and_update_outcomes_without_private_details(tmp_path, monkeypatch):
    """The listing joins the record's public check/update facts; task ids and rollback paths stay private."""
    from ouroboros.skill_loader import find_skill

    drive = tmp_path / "data"
    write_skill(drive / "skills" / "ouroboroshub", "notes")
    loaded = find_skill(drive, "notes")
    digest = loaded.content_hash
    write_install_record(drive, "notes", {
        "source": {"id": "elsewhere", "kind": "hub", "name": "Elsewhere"}, "entry": "notes", "basis": "self_check",
        "release": {"version": "1", "digest": digest}, "base_digest": digest,
        "check": {"by": "ouroboros", "notes": "read every file", "task_id": "t-1", "at": "2026-10-18T00:00:00Z"},
        "pending_update": {"from_version": "0", "from_digest": "a" * 64, "version": "2", "digest": "b" * 64,
                           "requested_at": "2026-10-18T00:00:00Z"},
        "update_rejected": {"version": "3", "digest": "c" * 64, "reason": "review_blockers",
                            "at": "2026-10-18T00:00:00Z"},
    })
    install = catalog.install_listing(drive, find_skill(drive, "notes"), {})["install"]
    assert install["check"] == {"by": "ouroboros", "notes": "read every file", "at": "2026-10-18T00:00:00Z"}
    assert install["pending_update"] == {"version": "2", "digest": "b" * 64}
    assert install["update_rejected"] == {"version": "3", "digest": "c" * 64, "reason": "review_blockers",
                                          "at": "2026-10-18T00:00:00Z"}
    assert "task_id" not in install["check"] and "from_digest" not in install["pending_update"]

    write_install_record(drive, "notes", {
        "source": {"id": "elsewhere", "kind": "hub"}, "entry": "notes", "basis": "review",
        "release": {"version": "1", "digest": digest}, "base_digest": digest})
    plain = catalog.install_listing(drive, find_skill(drive, "notes"), {})["install"]
    assert not {"check", "pending_update", "update_rejected"} & set(plain)

