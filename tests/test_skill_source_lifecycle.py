"""Sources → git transport → catalog → install / update / uninstall, against real local repositories."""

from __future__ import annotations

import json
import os
import stat

import pytest

from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_install as installer
from ouroboros.skill_install_record import read_install_record, write_install_record
from ouroboros.skill_loader import SkillReviewState, save_enabled, save_skill_grants, skill_state_dir
from tests._skill_source_fixtures import (
    HubFixture, author_repo, builtin_repo, commit_all, git, join_source_refreshes, publish, write_skill,
)


@pytest.fixture
def drive(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "skills").mkdir(parents=True)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    yield root
    join_source_refreshes()


@pytest.fixture
def hub(tmp_path):
    fixture = HubFixture(tmp_path / "remote")
    fixture.add_hosted("weather", "0.3.2", official=True, recommended=True)
    fixture.add_hosted("notes", "1.0.0", category=["writing"])
    fixture.commit("first")
    return fixture


def _refresh(drive, rows):
    for row in rows:
        if row["kind"] in {"hub", "git"}:
            meta = catalog.refresh_source(drive, row)
            assert not meta.get("error"), meta


def _ids(view):
    return {item["id"] for item in view["items"]}


def test_hub_rows_split_official_and_community_from_one_clone(drive, hub, tmp_path):
    rows = hub.rows()
    _refresh(drive, rows[1:2])  # one fetch serves both rows: they read one repository
    view = catalog.catalog_view(drive, rows=rows, repo_dir=builtin_repo(tmp_path))
    assert {"ouroboros/weather", "ouroboros-community/notes"} <= _ids(view)
    assert "ouroboros-community/weather" not in _ids(view) and "ouroboros/notes" not in _ids(view)
    weather = next(i for i in view["items"] if i["id"] == "ouroboros/weather")
    assert weather["official"] and weather["recommended"] and weather["trust"] == "trusted"
    assert weather["signals"] is None  # no signals branch: no data, never zero
    assert {c["id"] for c in view["categories"]} == {"city", "writing"}
    assert {"builtin/telegram", "builtin/unix_computer_use"} <= _ids(view)
    assert all(s["state"] == "ready" for s in view["sources"] if s["kind"] != "registry")


def test_an_endorsement_of_other_bytes_is_not_official(drive, hub):
    hub.official[0]["digest"] = "0" * 64
    hub.commit("stale endorsement")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    ids = _ids(catalog.catalog_view(drive, rows=rows[1:]))
    assert "ouroboros-community/weather" in ids and "ouroboros/weather" not in ids


def test_signals_branch_feeds_votes_and_missing_means_unknown(drive, hub):
    hub.publish_signals({"weather": {"votes": 25, "comments": 3, "discussion_url": "https://d/1"}})
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    items = {i["name"]: i for i in catalog.catalog_view(drive, rows=rows[1:])["items"]}
    assert items["weather"]["signals"]["votes"] == 25
    assert items["notes"]["signals"] is None


def test_catalog_reads_never_wait_for_the_network(drive, hub, monkeypatch):
    started = []
    monkeypatch.setattr(catalog, "refresh_in_background", lambda d, row, fn: started.append(row["id"]) or True)
    view = catalog.catalog_view(drive, rows=hub.rows()[1:])
    assert view["items"] == [] and started  # no cache yet: reading, with a background fetch started
    assert {s["state"] for s in view["sources"]} == {"reading"}


def test_trusted_install_lands_records_and_turns_on(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    result = installer.install_item(drive, "ouroboros/weather", rows=rows, reason="forecasts", actor="ouroboros")
    assert result["basis"] == "trusted" and result["acceptance"]["turn_on"] == "enabled", result
    target = drive / "skills" / "ouroboroshub" / "weather"
    assert (target / "SKILL.md").is_file() and not (target / ".ouroboroshub.json").exists()
    record = read_install_record(drive, "weather")
    assert record["basis"] == "trusted" and record["official"] is True and record["installed_by"] == "ouroboros"
    assert record["reason"] == "forecasts" and record["release"]["commit"] and record["release"]["path"] == "skills/weather"
    assert record["base_digest"] == record["release"]["digest"] == hub.entries["weather"]["digest"]
    assert json.loads((skill_state_dir(drive, "weather") / "enabled.json").read_text())["enabled"] is True


def test_review_install_waits_for_review_and_blockers_keep_it_off(drive, hub, monkeypatch):
    from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready
    from ouroboros.skill_loader import find_skill, save_review_state

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    result = installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    assert result["needs_review"] and result["basis"] == "review" and "acceptance" not in result
    assert reconcile_turn_on_when_ready(drive, "notes", repo_path="")["missing"] == ["acceptance"]
    loaded = find_skill(drive, "notes")
    # From a source that reviews each skill, blockers keep it off even under Advisory enforcement.
    save_review_state(drive, "notes", SkillReviewState(status="blockers", content_hash=loaded.content_hash))
    assert reconcile_turn_on_when_ready(drive, "notes", repo_path="")["action"] == "not_ready"
    from ouroboros.skill_review import review_skill
    from ouroboros.tools.registry import ToolContext
    from tests.test_skill_review_persist_guard import _clean_wave

    sends = _clean_wave(monkeypatch)  # the golden pool's seats answer clean through the real custody layer
    outcome = review_skill(ToolContext(repo_dir=drive, drive_root=drive), "notes", reviewers=["t1"], reason="verify the repair")
    assert outcome.status == "clean" and outcome.panel["assigned"] == ["t1"]
    assert [send["slot_id"] for send in sends] == ["t1"], "a re-review after blockers obeys ordinary Advisory composition"
    assert reconcile_turn_on_when_ready(drive, "notes", repo_path="")["action"] == "enabled"


def test_owner_off_is_never_overridden(drive, hub):
    from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready

    rows = hub.rows()
    _refresh(drive, rows[1:2])
    save_enabled(drive, "weather", False, actor="owner_ui")
    result = installer.install_item(drive, "ouroboros/weather", rows=rows)
    assert result["acceptance"]["accepted"] and result["acceptance"]["turn_on"] == "owner_choice"
    assert json.loads((skill_state_dir(drive, "weather") / "enabled.json").read_text())["enabled"] is False
    assert reconcile_turn_on_when_ready(drive, "weather", repo_path="")["action"] == "owner_choice"


def test_a_failed_preflight_keeps_a_trusted_release_off(drive, hub, monkeypatch):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    monkeypatch.setattr("ouroboros.skill_review.run_acceptance_preflight",
                        lambda *_a, **_k: {"ok": False, "error": "controlled preflight failure"})
    result = installer.install_item(drive, "ouroboros/weather", rows=rows)
    assert result["acceptance"] == {"accepted": False, "granted_keys": [], "granted_permissions": [],
                                    "deps_status": "", "turn_on": "not_ready",
                                    "preflight_error": "controlled preflight failure",
                                    "setup_error": "Its files did not pass the check; it remains off."}
    assert not (skill_state_dir(drive, "weather") / "enabled.json").exists()


def test_digest_mismatch_is_refused_and_nothing_lands(drive, hub):
    hub.entries["notes"]["digest"] = "1" * 64
    hub.commit("wrong digest")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    with pytest.raises(installer.SkillInstallError) as refused:
        installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    assert refused.value.code == "digest_mismatch"
    assert not (drive / "skills" / "ouroboroshub" / "notes").exists() and read_install_record(drive, "notes") is None


def test_withdrawn_and_unknown_items_are_refused(drive, hub):
    hub.add_hosted("old", withdrawn={"reason": "superseded", "at": "2026-10-17"})
    hub.commit()
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    for item_id, code in (("ouroboros-community/old", "withdrawn"), ("ouroboros-community/nope", "not_found"),
                          ("elsewhere/x", "not_found")):
        with pytest.raises(installer.SkillInstallError) as refused:
            installer.install_item(drive, item_id, rows=rows)
        assert refused.value.code == code


def test_same_name_from_elsewhere_needs_confirmation_and_keeps_the_previous_copy(drive, hub):
    write_skill(drive / "skills" / "external", "notes", "9.9.9", body="mine")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    with pytest.raises(installer.SkillInstallError) as refused:
        installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    assert refused.value.code == "name_taken" and refused.value.facts["occupant"]["locations"] == ["external"]
    result = installer.install_item(drive, "ouroboros-community/notes", rows=rows, replace=True)
    assert result["replaced"] and not (drive / "skills" / "external" / "notes").exists()
    kept = drive / "skills" / "external" / ".rollback" / "notes.previous" / "SKILL.md"
    assert "mine" in kept.read_text()


def test_reinstalling_the_same_release_is_a_no_op_and_a_new_one_says_update(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    assert installer.install_item(drive, "ouroboros-community/notes", rows=rows)["already_installed"]
    hub.add_hosted("notes", "1.1.0", category=["writing"])
    hub.commit()
    _refresh(drive, rows[1:2])
    with pytest.raises(installer.SkillInstallError) as refused:
        installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    assert refused.value.code == "update_instead"


def test_pointer_entry_fetches_the_author_repository_at_its_commit(drive, hub, tmp_path):
    bare, commit, digest = author_repo(tmp_path / "remote", "scholar")
    hub.add_pointer("scholar", str(bare), commit, digest)
    hub.commit("pointer")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    item = next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "scholar")
    assert item["author_repo"] == str(bare)
    installer.install_item(drive, item["id"], rows=rows)
    record = read_install_record(drive, "scholar")
    assert record["release"]["commit"] == commit and record["release"]["repo"] == str(bare)
    detail = catalog.item_detail(drive, item["id"], rows=rows)
    assert "About it." in detail["about"] and detail["links"]["author_repo"] == str(bare)


def test_plain_git_repository_lists_every_manifest_folder(drive, tmp_path):
    work = tmp_path / "plain"
    work.mkdir()
    git("init", "-q", cwd=work)
    write_skill(work, "alpha")
    write_skill(work / "group", "beta")
    commit_all(work)
    bare = tmp_path / "plain.git"
    publish(work, bare)
    row = {"id": "mine", "name": "Mine", "kind": "git", "url": str(bare), "ref": "main", "trust": "review",
           "confirmed": True}
    _refresh(drive, [row])
    items = {i["name"]: i for i in catalog.catalog_view(drive, rows=[row])["items"]}
    assert set(items) == {"alpha", "beta"} and len(items["beta"]["digest"]) == 64
    installer.install_item(drive, "mine/beta", rows=[row])
    assert read_install_record(drive, "beta")["release"]["path"] == "group/beta"


def test_update_detection_replace_keep_and_stale_candidates(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows)
    from ouroboros.skill_loader import find_skill

    snapshots = catalog.cached_snapshots(drive, rows)
    assert catalog.install_listing(drive, find_skill(drive, "weather"), snapshots)["install"]["update_available"] is None
    new_digest = hub.add_hosted("weather", "0.4.0", official=True, recommended=True)
    hub.commit("release")
    _refresh(drive, rows[1:2])
    listing = catalog.install_listing(drive, find_skill(drive, "weather"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["update_available"] == {"version": "0.4.0", "digest": new_digest}
    assert listing["install"]["modified"] is False and listing["category"] == ["city"]
    with pytest.raises(installer.SkillInstallError) as stale:
        installer.update_installed(drive, "weather", mode="replace", digest="2" * 64, rows=rows)
    assert stale.value.code == "candidate_changed"
    installer.update_installed(drive, "weather", mode="keep", digest=new_digest, rows=rows)
    listing = catalog.install_listing(drive, find_skill(drive, "weather"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["update_available"] is None
    result = installer.update_installed(drive, "weather", mode="replace", digest=new_digest, rows=rows)
    assert result["version"] == "0.4.0" and read_install_record(drive, "weather")["base_digest"] == new_digest
    assert (drive / "skills" / "ouroboroshub" / ".rollback" / "weather.previous" / "SKILL.md").is_file()


def test_modified_left_source_and_withdrawn_states(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    from ouroboros.skill_loader import find_skill

    (drive / "skills" / "ouroboroshub" / "notes" / "extra.md").write_text("local", encoding="utf-8")
    listing = catalog.install_listing(drive, find_skill(drive, "notes"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["modified"] is True
    hub.entries["notes"]["withdrawn"] = {"reason": "unsafe", "at": "2026-10-17"}
    hub.commit()
    _refresh(drive, rows[1:2])
    listing = catalog.install_listing(drive, find_skill(drive, "notes"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["withdrawn"]["reason"] == "unsafe"
    hub.remove("notes")
    hub.commit()
    _refresh(drive, rows[1:2])
    listing = catalog.install_listing(drive, find_skill(drive, "notes"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["left_source"] is True and listing["install"]["update_available"] is None


def test_a_trust_change_never_rewrites_an_existing_install(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    trusted_rows = hub.rows(community_trust="trusted")
    catalog.cached_snapshots(drive, trusted_rows)
    assert read_install_record(drive, "notes")["basis"] == "review"


def test_uninstall_is_one_path_and_keeps_grants(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    save_skill_grants(drive, "notes", [], content_hash="h", requested_keys=[])
    result = installer.uninstall_installed(drive, "notes")
    assert result == {"ok": True, "name": "notes", "source": "ouroboros-community"}
    state = skill_state_dir(drive, "notes")
    assert not (drive / "skills" / "ouroboroshub" / "notes").exists() and read_install_record(drive, "notes") is None
    assert (state / "uninstalled.json").is_file() and (state / "grants.json").is_file()
    write_skill(drive / "skills" / "external", "mine")
    with pytest.raises(installer.SkillInstallError) as refused:
        installer.uninstall_installed(drive, "mine")
    assert refused.value.code == "not_from_source"


def test_builtin_installs_through_the_builtin_source_and_is_not_resurrected(drive, tmp_path):
    from ouroboros.skill_builtin import reseed_changed_builtins

    repo = builtin_repo(tmp_path)
    rows = [{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}]
    result = installer.install_item(drive, "builtin/telegram", rows=rows, repo_dir=repo, reason="starter skills")
    native = drive / "skills" / "native"
    assert (native / "telegram" / ".seed-origin").is_file() and result["acceptance"]["turn_on"] == "enabled"
    record = read_install_record(drive, "telegram")
    assert record["source"]["kind"] == "builtin" and record["basis"] == "trusted"
    assert record["reason"] == "starter skills" and record["base_digest"] == record["release"]["digest"]
    installer.uninstall_installed(drive, "telegram")
    assert not (native / "telegram").exists()
    assert reseed_changed_builtins(drive, seed_dir=repo / "skills") == 0 and not (native / "telegram").exists()


def test_owner_merge_goes_through_the_owner_door_and_ouroboros_merges_in_its_own_task(drive, hub, monkeypatch):
    sent = []

    class _Bridge:
        def ui_send(self, text, **kwargs):
            sent.append((text, kwargs))

    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: _Bridge())
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    (drive / "skills" / "ouroboroshub" / "notes" / "SKILL.md").write_text(
        (drive / "skills" / "ouroboroshub" / "notes" / "SKILL.md").read_text() + "\nlocal edit\n")
    hub.add_hosted("notes", "2.0.0", category=["writing"])
    hub.commit()
    _refresh(drive, rows[1:2])
    result = installer.update_installed(drive, "notes", mode="merge", rows=rows)
    assert result["base_available"] is True and result["task"] == "queued"
    prompt, kwargs = sent[0]
    assert "three-way" in prompt and kwargs["task_constraint"]["skill_name"] == "notes"
    assert kwargs["task_constraint"]["allow_enable"] is False
    record = read_install_record(drive, "notes")
    assert record["merge"]["version"] == "2.0.0" and record["skipped_release_digest"] == record["merge"]["digest"]
    assert record["merge"]["requested_by"] == "owner"
    own = installer.update_installed(drive, "notes", mode="merge", rows=rows, actor="ouroboros")
    assert len(sent) == 1  # Ouroboros asking for itself never poses as the owner's request
    assert own["task"] == "current" and own["local_tree"] == "skills/ouroboroshub/notes"
    assert (drive / own["new_tree"] / "SKILL.md").is_file() and (drive / own["base_tree"] / "SKILL.md").is_file()
    assert own["instructions"].startswith("Merge the update") and "skill_review" in own["instructions"]


def test_legacy_record_without_base_says_original_unknown(drive, hub):
    write_skill(drive / "skills" / "clawhub", "legacy")
    write_install_record(drive, "legacy", {"source": {"id": "clawhub", "kind": "registry", "name": "ClawHub"},
                                           "release": {"version": "1", "digest": ""}, "legacy": True,
                                           "basis": "review"})
    from ouroboros.skill_loader import find_skill

    listing = catalog.install_listing(drive, find_skill(drive, "legacy"), {})
    assert listing["install"]["base_known"] is False and listing["install"]["modified"] is None


def test_copy_refuses_symlinks_and_oversized_trees(tmp_path):
    from ouroboros.marketplace.fetcher import FetchError, copy_tree_checked

    source = write_skill(tmp_path, "linked")
    (source / "escape").symlink_to(tmp_path)
    with pytest.raises(FetchError):
        copy_tree_checked(source, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()
    (source / "escape").unlink()
    copy_tree_checked(source, tmp_path / "stage2")
    assert (tmp_path / "stage2" / "SKILL.md").is_file()


def test_tree_digest_covers_declared_dependencies_and_ignores_install_notes(tmp_path):
    plain = write_skill(tmp_path / "a", "dep")
    declared = write_skill(tmp_path / "b", "dep", extra="dependencies: [ddgs]\n")
    assert catalog.tree_release_digest(plain) != catalog.tree_release_digest(declared)
    before = catalog.tree_release_digest(plain)
    (plain / ".ouroboroshub.json").write_text("{}", encoding="utf-8")
    # The note of the destination bucket stays outside; elsewhere a file of that name is payload.
    assert catalog.tree_release_digest(plain, bucket="ouroboroshub") == before
    assert catalog.tree_release_digest(plain) != before
    (plain / "SKILL.md").write_text((plain / "SKILL.md").read_text() + "!", encoding="utf-8")
    assert catalog.tree_release_digest(plain, bucket="ouroboroshub") != before


def _commit_at(monkeypatch, when):
    monkeypatch.setenv("GIT_AUTHOR_DATE", when)
    monkeypatch.setenv("GIT_COMMITTER_DATE", when)


def _dates(drive, row):
    return {item["name"]: item.get("updated_at") for item in catalog.catalog_view(drive, rows=[row])["items"]}


def test_entries_are_dated_by_the_newest_commit_touching_their_release(drive, tmp_path, monkeypatch):
    from ouroboros import skill_source_git

    hub = HubFixture(tmp_path / "dated")
    _commit_at(monkeypatch, "2026-10-01T10:00:00+00:00")
    hub.add_hosted("weather")
    hub.add_hosted("notes")
    hub.commit("first")
    _commit_at(monkeypatch, "2026-10-05T10:00:00+00:00")
    hub.add_hosted("notes", "1.1.0")
    hub.commit("notes release")
    bare, commit, digest = author_repo(tmp_path, "scholar")
    _commit_at(monkeypatch, "2026-10-09T10:00:00+00:00")
    hub.add_hosted("weather", "1.1.0")
    hub.add_pointer("scholar", str(bare), commit, digest)  # a pointer's release is its entry file
    hub.commit("weather release, scholar listed")
    community = hub.rows()[2]
    monkeypatch.setattr(skill_source_git, "HISTORY_DEPTH", 2)  # the fetched history ends at the notes release
    _refresh(drive, [community])
    assert _dates(drive, community) == {"notes": None, "scholar": "2026-10-09T10:00:00+00:00",
                                        "weather": "2026-10-09T10:00:00+00:00"}, "a change at the cut-off may be older"
    monkeypatch.setattr(skill_source_git, "HISTORY_DEPTH", 100)
    _refresh(drive, [community])
    assert _dates(drive, community)["notes"] == "2026-10-05T10:00:00+00:00"


def test_plain_repository_skills_are_dated_by_their_own_folders(drive, tmp_path, monkeypatch):
    work = tmp_path / "plain"
    work.mkdir()
    git("init", "-q", cwd=work)
    _commit_at(monkeypatch, "2026-09-01T08:00:00+00:00")
    write_skill(work, "alpha")
    write_skill(work / "group", "beta")
    commit_all(work)
    _commit_at(monkeypatch, "2026-09-03T08:00:00+00:00")
    (work / "group" / "beta" / "notes.md").write_text("more", encoding="utf-8")
    (work / "README.md").write_text("not a skill", encoding="utf-8")
    commit_all(work)
    publish(work, tmp_path / "plain.git")
    plain = {"id": "mine", "name": "Mine", "kind": "git", "url": str(tmp_path / "plain.git"), "ref": "main",
             "trust": "review", "confirmed": True}
    single = {**plain, "id": "one", "url": str(author_repo(tmp_path, "solo")[0])}  # the repository root is the skill
    _refresh(drive, [plain, single])
    assert _dates(drive, plain) == {"alpha": "2026-09-01T08:00:00+00:00", "beta": "2026-09-03T08:00:00+00:00"}
    assert _dates(drive, single) == {"solo": "2026-09-03T08:00:00+00:00"}


def test_registry_hits_name_their_author_and_update_time(monkeypatch):
    from ouroboros.marketplace import clawhub

    records = [{"slug": "weather", "ownerHandle": "steipete", "updatedAt": 1789594554485, "summary": "Forecasts."},
               {"slug": "plain", "publisher": {"handle": "someone"}, "summary": "No update time."},
               {"slug": "bare", "updatedAt": "yesterday", "summary": "Nothing said."}]
    page = {"results": [clawhub._summary_from_record(record) for record in records]}
    monkeypatch.setattr(clawhub, "search", lambda *_a, **_k: page)
    row = {"id": "clawhub", "name": "ClawHub", "kind": "registry", "trust": "review", "confirmed": True}
    hits = {item["name"]: item for item in catalog.registry_items(row, "weather")}
    assert (hits["weather"]["owner"], hits["weather"]["updated_at"]) == ("steipete", "2026-09-16T21:35:54.485000+00:00")
    assert hits["plain"]["owner"] == "someone" and "updated_at" not in hits["plain"]
    assert hits["bare"]["owner"] == "" and "updated_at" not in hits["bare"]
    # Shared catalog consumes the adapter's normalized facts, not its raw wire keys.
    monkeypatch.setattr(clawhub.ClawHubSkillSummary, "catalog_facts", lambda self: {"owner": "adapter"})
    assert {item["owner"] for item in catalog.registry_items(row, "weather")} == {"adapter"}


def test_the_listing_says_why_and_in_which_task_ouroboros_installed(drive, hub):
    from ouroboros.skill_loader import find_skill

    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows, actor="ouroboros",
                           reason="the owner asked for a forecast", task_id="t-1")
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    snapshots = catalog.cached_snapshots(drive, rows)
    told = catalog.install_listing(drive, find_skill(drive, "weather"), snapshots)["install"]
    assert told["installed_by"] == "ouroboros"
    assert (told["reason"], told["task_id"]) == ("the owner asked for a forecast", "t-1")
    owners = catalog.install_listing(drive, find_skill(drive, "notes"), snapshots)["install"]
    assert owners["installed_by"] == "owner" and "reason" not in owners and "task_id" not in owners


def test_a_build_marks_its_rows_managed_and_nothing_else_does():
    from ouroboros.skill_sources import normalize_source, repository_owner, with_added_source, with_changed_source

    shipped = normalize_source({"id": "team", "name": "Team", "kind": "git", "url": "https://git.example/t/skills",
                                "trust": "trusted", "managed": True})
    assert shipped["managed"] is True
    rows = with_changed_source([shipped], "team", name="Team skills", trust="review")
    assert rows[0]["managed"] is True and rows[0]["trust"] == "review", "an edit keeps the mark and is not refused"
    rows, added = with_added_source(rows, kind="git", url="https://git.example/me/more", name="More", trust="review")
    assert "managed" not in added
    assert "managed" not in normalize_source({**shipped, "managed": "yes"}), "only a real true marks a row"
    assert [repository_owner(url) for url in (
        "https://github.com/kuznetsoffandrey/ru-skills", "git@github.com:razzant/OuroborosHub.git",
        "ssh://git@git.example/group/sub/repo.git", "https://git.example/repo", "/srv/skills", "file:///srv/skills",
    )] == ["kuznetsoffandrey", "razzant", "group/sub", "", "", ""]


def test_probe_reads_a_hub_a_plain_repo_and_a_folder(drive, hub, tmp_path):
    probe = catalog.probe_source(drive, url=str(hub.bare))
    assert probe["kind"] == "hub" and probe["suggested_name"] == "TestHub" and "weather" in probe["found"]
    assert probe["owner"] == "m", "a hub names its maintainer"
    plain = catalog.probe_source(drive, url=str(author_repo(tmp_path, "solo")[0]))
    assert plain["kind"] == "git" and "owner" not in plain, "a folder path names no account"
    folder = write_skill(tmp_path / "folder", "x").parent
    assert catalog.probe_source(drive, path=str(folder)) == {"kind": "local", "suggested_name": "folder", "found": ["x"]}
    assert "error" in catalog.probe_source(drive, url="https://user:secret@example.invalid/r")
    assert "error" in catalog.probe_source(drive, url=str(tmp_path / "missing.git"))
    assert not list((drive / "state" / "skill_sources" / ".scratch").iterdir())


def test_a_probe_leaves_no_clone_where_read_only_files_refuse_deletion(drive, hub, monkeypatch):
    """Windows refuses to delete a read-only file, and a clone's git objects are read-only: the
    probe's scratch clone still goes instead of staying in the scratch folder forever."""
    real_unlink = os.unlink

    def windows_unlink(path, *, dir_fd=None):
        if not os.lstat(path, dir_fd=dir_fd).st_mode & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied", path)
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", windows_unlink)
    monkeypatch.setattr(os, "remove", windows_unlink)
    assert catalog.probe_source(drive, url=str(hub.bare))["kind"] == "hub"
    assert not list((drive / "state" / "skill_sources" / ".scratch").iterdir())


def test_a_failed_landing_restores_the_installed_release(drive, hub, monkeypatch):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    target = drive / "skills" / "ouroboroshub" / "notes"
    before = (target / "SKILL.md").read_text(encoding="utf-8")
    new_digest = hub.add_hosted("notes", "2.0.0", category=["writing"])
    hub.commit()
    _refresh(drive, rows[1:2])

    def unwritable(*_args, **_kwargs):
        raise OSError("controlled record failure")

    monkeypatch.setattr(installer, "write_install_record", unwritable)
    with pytest.raises(OSError, match="controlled record failure"):
        installer.update_installed(drive, "notes", mode="replace", digest=new_digest, rows=rows)
    assert (target / "SKILL.md").read_text(encoding="utf-8") == before
    assert read_install_record(drive, "notes")["release"]["version"] == "1.0.0"
    assert not list((target.parent / ".staging").iterdir())


def test_partial_state_restore_never_deletes_restored_payload(tmp_path, monkeypatch):
    from ouroboros.marketplace import install as rollback

    drive = tmp_path / "data"
    target = drive / "skills" / "ouroboroshub" / "demo"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("old", encoding="utf-8")
    snapshot = rollback.snapshot_payload_state(drive, "demo", target)
    (target / "payload.txt").write_text("new", encoding="utf-8")
    restore = rollback.restore_lifecycle_state
    monkeypatch.setattr(rollback, "restore_lifecycle_state", lambda *args: ["controlled state failure"])
    with pytest.raises(OSError, match="controlled state failure"):
        rollback.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text(encoding="utf-8") == "old"
    monkeypatch.setattr(rollback, "restore_lifecycle_state", restore)
    rollback.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text(encoding="utf-8") == "old"


def test_missing_snapshot_refuses_before_removing_current_payload(tmp_path):
    from ouroboros.marketplace import install as rollback

    drive = tmp_path / "data"
    target = drive / "skills" / "ouroboroshub" / "demo"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("keep", encoding="utf-8")
    snapshot = rollback.PayloadRollbackSnapshot(drive, "demo", target, tmp_path / "missing")
    with pytest.raises(OSError, match="rollback payload is missing"):
        rollback.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text(encoding="utf-8") == "keep"
