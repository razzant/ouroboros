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
from tests._extension_loader_shared import extension_tool_answer, version_plugin
from tests._skill_source_fixtures import (
    HubFixture, author_repo, builtin_repo, commit_all, entry, git, join_source_refreshes, publish, write_skill,
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
    assert not {"type", "needs_keys", "permissions"} & set(item), "a catalog read never fetches a pointer's manifest"
    installer.install_item(drive, item["id"], rows=rows)
    record = read_install_record(drive, "scholar")
    assert record["release"]["commit"] == commit and record["release"]["repo"] == str(bare)
    detail = catalog.item_detail(drive, item["id"], rows=rows)
    assert "About it." in detail["about"] and detail["links"]["author_repo"] == str(bare)
    assert detail["item"]["type"] == "instruction" and detail["item"]["permissions"] == [], "the pinned release's own"


_ALMANAC = "title: Almanac Pro\nicon: globe\npermissions: [net]\nui_tab:\n  tab_id: board\n  title: Almanac Board\n"


def test_the_detail_names_an_entry_as_its_card_and_takes_from_the_release_what_its_catalog_leaves_out(
        drive, hub, tmp_path):
    # Hub words for people (a title the manifest lacks, services and an icon of a pointer entry) are the entry's
    # presentation; the files the detail reads add the manifest's type, keys and permissions, and an icon or services
    # only where the catalog names none.
    bare, commit, digest = author_repo(tmp_path / "remote", "scholar", extra="icon: book\n")
    hub.add_pointer("scholar", str(bare), commit, digest)
    hub.entries["scholar"].update(title="Reading notes", pitch="Keeps sources at hand.", connects=["Mail"], icon="mail")
    # A pointer with no catalog title: its list shows the id, and so does its detail, though the files the detail
    # reads title the skill and its widget otherwise; what the release can use, and its icon, come from them.
    bare, commit, digest = author_repo(tmp_path / "remote", "almanac", extra=_ALMANAC)
    hub.add_pointer("almanac", str(bare), commit, digest)
    hub.entries["notes"]["title"] = "Account limits"
    hub.commit("titled entries")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    listed = {i["name"]: i for i in catalog.catalog_view(drive, rows=rows[1:])["items"]}
    assert (listed["almanac"]["title"], listed["almanac"]["widget_title"], listed["almanac"]["icon"]) == ("", "", {})
    for name, permissions in (("scholar", []), ("notes", []), ("almanac", ["net"])):
        shown = catalog.item_detail(drive, listed[name]["id"], rows=rows)["item"]
        for key in ("title", "widget_title", "search_names", "pitch", "connects", "description", "digest", "version",
                    "official"):
            assert shown.get(key) == listed[name].get(key), (name, key)
        assert (shown["type"], shown["needs_keys"], shown["permissions"]) == ("instruction", [], permissions), name
        if name != "almanac":
            assert shown["icon"] == listed[name]["icon"], name
    assert listed["scholar"]["icon"] == {"kind": "host", "name": "mail"}, "the catalog's icon, not its release's book"
    assert catalog.item_detail(drive, listed["almanac"]["id"], rows=rows)["item"]["icon"] == {
        "kind": "host", "name": "globe"}, "the catalog names none: the release's own"
    assert listed["scholar"]["title"] == "Reading notes" and listed["notes"]["title"] == "Account limits"


def test_an_installed_skill_is_named_by_its_own_source_entry_only_and_by_the_widget_it_declares(drive, hub, tmp_path):
    from ouroboros.skill_loader import find_skill

    hub.add_hosted("quotas", extra="ui_tab:\n  tab_id: board\n  title: Quota Board\n")
    hub.entries["quotas"]["title"] = "Account limits"
    hub.commit("titled entry")
    other = HubFixture(tmp_path / "other")  # another source offers a skill of the same name, titled otherwise
    other.add_hosted("notes")
    other.entries["notes"]["title"] = "Borrowed title"
    other.commit()
    rows = [*hub.rows(), {"id": "other", "name": "Other", "kind": "hub", "url": str(other.bare), "ref": "main",
                          "part": "community", "trust": "review", "confirmed": True}]
    _refresh(drive, [rows[1], rows[3]])
    for item_id in ("ouroboros-community/quotas", "ouroboros-community/notes"):
        installer.install_item(drive, item_id, rows=rows)
    snapshots = catalog.cached_snapshots(drive, rows)
    quotas = catalog.install_listing(drive, find_skill(drive, "quotas"), snapshots)
    assert (quotas["title"], quotas["catalog_title"], quotas["widget_title"]) == ("", "Account limits", "Quota Board")
    assert quotas["catalog_id"] == "ouroboros-community/quotas", "its detail sheet's address"
    notes = catalog.install_listing(drive, find_skill(drive, "notes"), snapshots)
    assert "catalog_title" not in notes and notes["widget_title"] == "", "its own source's entry has no title"
    assert notes["catalog_id"] == "ouroboros-community/notes", "never the other source's entry of the same name"
    uncached = catalog.install_listing(drive, find_skill(drive, "quotas"), {})
    assert "catalog_title" not in uncached and uncached["catalog_id"] is None, "no cached source: none"


def test_an_installed_skills_catalog_id_is_the_item_that_lists_its_entry_now(drive, hub):
    from ouroboros.skill_loader import find_skill

    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows)

    def catalog_id():
        listing = catalog.install_listing(drive, find_skill(drive, "weather"), catalog.cached_snapshots(drive, rows))
        return listing["catalog_id"]

    assert catalog_id() == "ouroboros/weather"
    hub.add_hosted("weather", "0.4.0")  # a release its maintainer has not endorsed: the Community view lists it now
    hub.commit("unendorsed release")
    _refresh(drive, rows[1:2])
    assert catalog_id() == "ouroboros-community/weather", "the detail sheet the catalog can open"


def _hosted_connector(hub, name, **manifest):
    """A hosted connector: an extension with the ``presence`` permission, and what ``manifest`` adds."""
    from ouroboros.skill_source_catalog import tree_release_digest

    folder = hub.work / "skills" / name
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "skill.json").write_text(json.dumps({
        "name": name, "description": f"{name} relays chats", "version": "1.0.0", "type": "extension",
        "entry": "plugin.py", "permissions": ["presence"], **manifest}), encoding="utf-8")
    hub.entries[name] = entry(name, tree_release_digest(folder))


def test_a_catalog_title_keeps_every_name_a_connectors_manifest_gives_it_and_its_installed_row_says_them(drive, hub):
    from ouroboros.skill_loader import find_skill

    places = [{"id": "chats", "provides": "place", "label": "Relay chats", "destination": "vendor:Relay"},
              {"id": "desk", "provides": "place", "label": "Relay Transport", "destination": "vendor:Relay"}]
    _hosted_connector(hub, "relay_bridge", title="Relay Courier", contributes=places,
                      ui_tab={"tab_id": "desk", "title": "Relay Transport"})
    hub.entries["relay_bridge"]["title"] = "Relay for others"
    hub.commit("connector")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    item = next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "relay_bridge")
    # The card prints the hub's title; the manifest's own title and place label stay names it is found by, and a
    # label its widget already prints is not said twice.
    assert (item["title"], item["widget_title"], item["search_names"]) == (
        "Relay for others", "Relay Transport", ["Relay Courier", "Relay chats"])
    installer.install_item(drive, item["id"], rows=rows, turn_on_when_ready=False)
    row = catalog.install_listing(drive, find_skill(drive, "relay_bridge"), catalog.cached_snapshots(drive, rows))
    assert (row["title"], row["widget_title"], row["search_names"], row["catalog_title"]) == (
        "Relay Courier", "Relay Transport", ["Relay chats"], "Relay for others")
    assert (row["connects"], row["catalog_id"]) == (["Relay"], item["id"])
    weather = next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "weather")
    assert "search_names" not in weather, "a skill with no other names has none"


def test_an_installed_unmodified_copy_of_a_pointers_release_lends_its_names_as_search_names_and_no_other_does(
        drive, hub, tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill

    bare, commit, digest = author_repo(tmp_path / "remote", "almanac", extra=_ALMANAC)
    hub.add_pointer("almanac", str(bare), commit, digest)
    hub.commit("pointer")
    rows = hub.rows()
    _refresh(drive, rows[1:2])

    def listed():
        return next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "almanac")

    assert (listed()["widget_title"], listed().get("search_names")) == ("", None), "a catalog read never fetches it"
    installer.install_item(drive, "ouroboros-community/almanac", rows=rows)
    real, digested = catalog.tree_release_digest, []
    monkeypatch.setattr(catalog, "tree_release_digest",
                        lambda folder, *a, **k: digested.append(os.path.basename(folder)) or real(folder, *a, **k))
    item = listed()
    # Search names only: the card keeps the catalog's title, empty here as before the install.
    assert (item["title"], item["widget_title"], item["search_names"]) == ("", "Almanac Board", ["Almanac Pro"])
    assert digested == ["almanac"], "the one installed pointer's digest, nothing for the other skills"
    installed = find_skill(drive, "almanac").skill_dir / "SKILL.md"
    original = installed.read_text(encoding="utf-8")
    installed.write_text(original + "Edited here.\n", encoding="utf-8")
    assert (listed()["widget_title"], listed().get("search_names")) == ("", None), "a modified copy lends nothing"
    installed.write_text(original, encoding="utf-8")
    assert listed()["search_names"] == ["Almanac Pro"], "the release's bytes again"
    bare, commit, digest = author_repo(tmp_path / "remote-next", "almanac", "2.1.0",
                                       extra=_ALMANAC.replace("Almanac Pro", "Almanac Next"))
    hub.add_pointer("almanac", str(bare), commit, digest)
    hub.commit("next release")
    _refresh(drive, rows[1:2])
    digested.clear()
    assert (listed()["widget_title"], listed().get("search_names"), digested) == ("", None, []), \
        "the copy of another release names neither this one nor costs a read"


def test_a_reused_source_id_of_another_repository_presents_an_older_record_by_none_of_its_words(drive, hub, tmp_path):
    from ouroboros.skill_loader import find_skill

    hub.entries["notes"]["title"] = "Field notes"
    hub.commit("titled")
    rows = hub.rows()
    _refresh(drive, rows[2:3])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    record = read_install_record(drive, "notes")
    for key in ("ref", "commit", "path"):  # recorded before sources had views: it matches by id or address
        record["source"].pop(key)
    write_install_record(drive, "notes", record)
    own = catalog.install_listing(drive, find_skill(drive, "notes"), catalog.cached_snapshots(drive, rows))
    assert (own["catalog_id"], own["catalog_title"], own["category"]) == (
        "ouroboros-community/notes", "Field notes", ["writing"])
    assert own["source_key"] is None, "display metadata does not prove the recorded branch"
    # The owner removes the hub and adds another repository under the community row's name: the id comes back.
    other = HubFixture(tmp_path / "other")
    newer = other.add_hosted("notes", "2.0.0", category=["city"])
    other.entries["notes"]["title"] = "Borrowed title"
    other.commit()
    reused = {**rows[2], "url": str(other.bare)}
    _refresh(drive, [reused])

    def listing(*more):
        snapshots = catalog.cached_snapshots(drive, [reused, *more])
        return catalog.install_listing(drive, find_skill(drive, "notes"), snapshots)

    alone = listing({**rows[2], "id": "plain", "kind": "git"})  # its own address, read as another kind
    assert alone["install"]["source"]["id"] == "ouroboros-community", "the older record still matches by id"
    assert alone["install"]["update_available"] == {"version": "2.0.0", "digest": newer}, "updates as before"
    assert (alone["catalog_id"], alone.get("catalog_title"), alone["category"]) == (None, None, [])
    # Its own repository connected again under another name presents it, while updates still follow the id.
    again = listing({**rows[2], "id": "old-notes", "name": "Old notes"})
    assert again["install"]["source"]["id"] == "ouroboros-community" and again["install"]["update_available"]
    assert (again["catalog_id"], again["catalog_title"], again["category"]) == (
        "old-notes/notes", "Field notes", ["writing"])


def _publish_widgets(monkeypatch, name, folder, revision, *titles, unloading=False):
    """A publication of ``name`` this process holds from ``folder`` at ``revision`` whose widget tabs are headed by
    ``titles`` (the registry the loader writes; nothing is loaded or started)."""
    from ouroboros import extension_registry_state as registry

    keys = [f"{name}:tab{index}" for index in range(len(titles))]
    monkeypatch.setitem(registry._extensions, name, registry._ExtensionRegistrations(
        ui_tabs=keys, skill_dir=str(folder.resolve()), content_hash=revision))
    for key, title in zip(keys, titles):
        monkeypatch.setitem(registry._ui_tabs, key, {"skill": name, "tab_id": key.split(":")[1], "title": title})
    if unloading:
        monkeypatch.setattr(registry, "_unloading", {name})


@pytest.mark.parametrize("case", ["this_release", "held_older_revision", "other_folder", "unloading", "no_tab"])
def test_live_widget_headers_name_only_their_own_folders_copy_and_the_entry_of_exactly_its_release(
        drive, hub, tmp_path, monkeypatch, case):
    """The headers an installed copy's live widgets are registered under find its Installed row while that very folder
    holds the publication (a call may still hold an older revision of it), and find its catalog entry only while that
    publication is exactly the entry's release; another folder's publication of the same name, one unloading, or none
    lends neither anything. Other skills gain nothing."""
    from ouroboros.skill_loader import find_skill

    hub.add_hosted("quotas", extra="ui_tab:\n  tab_id: board\n  title: Quota Board\n")
    hub.entries["quotas"]["title"] = "Account limits"
    hub.commit("titled entry")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    for item_id in ("ouroboros-community/quotas", "ouroboros-community/notes"):
        installer.install_item(drive, item_id, rows=rows)
    loaded = find_skill(drive, "quotas")
    folder, revision = loaded.skill_dir, loaded.content_hash
    if case == "other_folder":  # a namesake loaded from elsewhere, say the copy a replacement left
        folder = tmp_path / "elsewhere" / "quotas"
        folder.mkdir(parents=True)
    elif case == "held_older_revision":
        revision = "0" * 64
    if case != "no_tab":
        _publish_widgets(monkeypatch, "quotas", folder, revision, "Claudexor Quotas", unloading=case == "unloading")

    snapshots = catalog.cached_snapshots(drive, rows)
    installed = catalog.install_listing(drive, find_skill(drive, "quotas"), snapshots)
    listed = {i["name"]: i for i in catalog.catalog_view(drive, rows=rows[1:])["items"]}
    own_folder, exact = case in {"this_release", "held_older_revision"}, case == "this_release"
    assert installed.get("search_names") == (["Claudexor Quotas"] if own_folder else None)
    assert listed["quotas"].get("search_names") == (["Claudexor Quotas"] if exact else None)
    assert catalog.matches_words(listed["quotas"], "claudexor_quotas") is exact
    assert catalog.matches_words(listed["quotas"], "Account limits") and catalog.matches_words(
        listed["quotas"], "quota board"), "its own names find it in every case"
    assert "search_names" not in catalog.install_listing(drive, find_skill(drive, "notes"), snapshots)
    assert "search_names" not in listed["notes"] and not catalog.matches_words(listed["notes"], "Claudexor Quotas")
    # The listing's one read gives the same answer as a row's own.
    once = catalog.live_widget_titles()
    assert catalog.install_listing(drive, find_skill(drive, "quotas"), snapshots, live=once) == installed


def test_live_widget_headers_leave_an_entry_of_another_release_and_a_name_held_by_two_folders(
        drive, hub, monkeypatch):
    from ouroboros.skill_loader import find_skill

    hub.add_hosted("quotas", extra="ui_tab:\n  tab_id: board\n  title: Quota Board\n")
    hub.commit("quotas")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/quotas", rows=rows)
    loaded = find_skill(drive, "quotas")
    _publish_widgets(monkeypatch, "quotas", loaded.skill_dir, loaded.content_hash, "Claudexor Quotas")

    def listed():
        return next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "quotas")

    assert listed()["search_names"] == ["Claudexor Quotas"]
    hub.add_hosted("quotas", "0.2.0", extra="ui_tab:\n  tab_id: board\n  title: Quota Board\n")
    hub.commit("next release")
    _refresh(drive, rows[1:2])
    item = listed()
    assert item["digest"] != loaded.content_hash and "search_names" not in item, "another release's entry: none"
    assert catalog.install_listing(drive, loaded, catalog.cached_snapshots(drive, rows))["search_names"] == [
        "Claudexor Quotas"], "its own Installed row still is found by them"
    index = catalog.installed_index(drive)
    index["quotas"]["locations"].append("user_repo")  # two folders hold the name: which one published is unknown
    current = {i["name"]: i for i in catalog.items_for(rows[2], catalog.read_snapshot(drive, rows[2]))}
    hub_entry = current["quotas"]
    hub_entry["digest"] = loaded.content_hash
    catalog._join_installed([hub_entry], index, {rows[2]["id"]: rows[2]})
    assert "search_names" not in hub_entry


def test_an_older_record_of_a_repository_is_presented_by_its_top_folder_only(drive, hub):
    """A record written before rows had views came from its repository's top folder: a row reading a subfolder of the
    same address lends it neither words nor a detail sheet, while updates still match every view as before."""
    from ouroboros.skill_loader import find_skill

    hub.entries["notes"]["title"] = "Field notes"
    hub.commit("titled")
    rows = hub.rows()
    _refresh(drive, rows[2:3])
    installer.install_item(drive, "ouroboros-community/notes", rows=rows)
    record = read_install_record(drive, "notes")
    for key in ("ref", "commit", "path"):
        record["source"].pop(key, None)
    write_install_record(drive, "notes", record)
    cached = catalog.cached_snapshots(drive, rows)[rows[2]["id"]][1]
    sub = {**rows[2], "id": "notes-sub", "name": "Notes subfolder", "path": "skills"}
    root = {**rows[2], "id": "notes-root", "name": "Notes top"}
    loaded = find_skill(drive, "notes")
    beside = catalog.install_listing(drive, loaded, {"notes-sub": (sub, cached)})
    assert beside["install"]["source"]["id"] == "notes-sub", "the update matcher still reads every view"
    assert (beside["catalog_id"], beside.get("catalog_title"), beside["category"]) == (None, None, [])
    top = catalog.install_listing(drive, loaded, {"notes-sub": (sub, cached), "notes-root": (root, cached)})
    assert (top["catalog_id"], top["catalog_title"]) == ("notes-root/notes", "Field notes")
    assert top["source_key"] is None
    assert catalog.item_detail(drive, top["catalog_id"], rows=[root])["item"]["title"] == "Field notes"

    other = {**root, "id": "notes-next", "ref": "next"}
    newer = {**cached, "entries": [
        {**entry, "digest": "f" * 64} if entry["name"] == "notes" else entry for entry in cached["entries"]]}
    assert catalog.install_listing(drive, loaded, {other["id"]: (other, newer)})["catalog_id"] is None
    assert catalog.install_listing(drive, loaded, {
        root["id"]: (root, cached), other["id"]: (other, cached)})["catalog_id"] is None, "two candidate views"

    modern = read_install_record(drive, "notes")
    modern["source"].update(ref="main", path="")  # a record of the top view, written since views
    write_install_record(drive, "notes", modern)
    assert catalog.install_listing(drive, loaded, {
        sub["id"]: (sub, cached), other["id"]: (other, cached)})["catalog_id"] is None
    assert catalog.install_listing(drive, loaded, {
        root["id"]: (root, cached), other["id"]: (other, cached)})["catalog_id"] == "notes-root/notes"


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


# --- an accepted update and the running extension: the published tool, never the enabled bit, is the proof ---

@pytest.fixture
def live_extensions(monkeypatch):
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


_PINGER = ('---\nname: pinger\ndescription: pinger answers its release\nversion: {version}\ntype: extension\n'
           'entry: plugin.py\nplugin_api: "2.0"\npermissions: [tool]\n{extra}---\nAbout it.\n')  # 2.0: acceptance needs it


def _release_pinger(drive, hub, rows, version, *, plugin="", extra=""):
    """``pinger`` ``version`` as an Official (trusted) extension of the hub, read by this install."""
    folder = hub.work / "skills" / "pinger"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(_PINGER.format(version=version, extra=extra), encoding="utf-8")
    (folder / "plugin.py").write_text(plugin or version_plugin(version), encoding="utf-8")
    digest = catalog.tree_release_digest(folder)
    hub.entries["pinger"] = entry("pinger", digest, version=version)
    hub.official = [*[item for item in hub.official if item["name"] != "pinger"],
                    {"name": "pinger", "digest": digest, "recommended": False}]
    hub.commit(f"pinger {version}")
    _refresh(drive, rows[1:2])
    return digest


def _owner(drive, action):
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.tools.registry import ToolContext

    result = run_skill_action(ToolContext(repo_dir=drive, drive_root=drive), "pinger", action, _owner_actor="owner_ui")
    assert result.get("ok") is not False, result


def _choice(drive):
    return (skill_state_dir(drive, "pinger") / "enabled.json").read_bytes()


@pytest.fixture
def hub_calendar(drive, tmp_path, live_extensions, monkeypatch):
    """A prepared local Hub snapshot; use real install, review and extension effects, no remote transport. Like the
    real calendar it declares ``notify_owner``, and its read tool returns the meetings of its own local store for the
    calendars its granted fields name. All values are synthetic; nothing here sends a notice."""
    from pathlib import Path
    from types import SimpleNamespace

    from ouroboros import config, extension_loader
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.tools.extension_dispatch import dispatch_extension_tool
    from ouroboros.tools.registry import ToolContext
    from tests._extension_loader_shared import _write_ext_skill

    name, primary, other = "fixture-calendar", "FIXTURE_CALENDAR_PRIMARY", "FIXTURE_CALENDAR_SECONDARY"
    settings = drive / "settings.json"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    settings.write_text(json.dumps({primary: "example-primary", other: "example-secondary"}))
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "catalog"
    row = {"kind": "hub", "id": "fixture-hub", "name": "Fixture Hub", "trust": "trusted", "url": root.as_uri()}
    snapshot = {}
    monkeypatch.setattr(catalog, "snapshot_for", lambda *_a, **_k: (snapshot, {"state": "ready"}))
    monkeypatch.setattr(catalog, "checkout_at_commit", lambda *_a, **_k: root)

    def offer(label):
        folder = _write_ext_skill(root, name, permissions=["tool", "read_settings", "notify_owner"],
                                  env_from_settings=[primary, other],
                                  extra_frontmatter='plugin_api: "2.0"\ntools:\n  - name: events\n    effect: read\n',
                                  plugin_body=(
                                      "import json\nimport pathlib\n\n\n"
                                      "def register(api):\n"
                                      "    def events(ctx):\n"
                                      f"        keys = ({primary!r}, {other!r})\n"
                                      "        fields = api.get_settings(list(keys))\n"
                                      "        store = json.loads((pathlib.Path(api.get_state_dir()) / 'meetings.json')"
                                      ".read_text(encoding='utf-8'))\n"
                                      "        found = [m['start'] + ' ' + m['title'] for key in keys\n"
                                      "                 for m in store.get(fields.get(key) or '', [])]\n"
                                      f"        return str(({label!r}, found))\n"
                                      "    api.register_tool('events', events, description='Read meetings',"
                                      " schema={})\n"))
        digest = catalog.tree_release_digest(folder)
        snapshot.update(commit="1" * 40, entries=[{"name": name, "digest": digest, "version": "0.1.0",
                                                  "source": {"kind": "hosted", "path": name}}])
        return digest

    initial = offer("before")
    installed = installer.run_in_lane(drive, "install", name, lambda: installer.install_item(
        drive, f"{row['id']}/{name}", rows=[row], repo_dir=repo, turn_on_when_ready=False))
    assert installed["digest"] == initial and installed["acceptance"]["accepted"]
    # The calendar's local data in its state folder, keyed by the calendar each fictitious field names.
    (skill_state_dir(drive, name) / "meetings.json").write_text(json.dumps({
        "example-primary": [{"start": "2026-11-25T10:00+03:00", "title": "Demo standup"}],
        "example-secondary": [{"start": "2026-11-25T15:00+03:00", "title": "Demo review"}]}), encoding="utf-8")
    owner = ToolContext(repo_dir=repo, system_repo_dir=repo, drive_root=drive)
    granted = run_skill_action(owner, name, "grant", expected_content_hash=initial, items=[primary, "notify_owner"],
                               _owner_actor="owner_ui", _enable_after_grant=True)
    assert granted["ok"] and granted["granted_keys"] == [primary] and granted["enabled"]
    assert granted["granted_permissions"] == ["notify_owner"]

    def events():
        tool_name = extension_loader.extension_surface_name(name, "events")
        tool = extension_loader.get_tool(tool_name)
        assert tool is not None
        return dispatch_extension_tool(owner, tool_name, tool, {})

    return SimpleNamespace(name=name, primary=primary, other=other, initial=initial, drive=drive,
                           events=events, offer=offer, repo=repo, row=row, owner=owner)


@pytest.mark.parametrize("owner_off,kept", [(False, False), (True, False), (False, True)],
                         ids=["keeps-on", "keeps-off", "after-keep"])
def test_hub_repair_keeps_only_the_explicit_field_at_the_final_reviewed_digest(hub_calendar, monkeypatch, owner_off,
                                                                                kept):
    from ouroboros import extension_loader
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateway.host_service import HostServiceAuthError, HostServiceContext
    from ouroboros.skill_lifecycle_actions import run_skill_action
    from ouroboros.skill_loader import find_skill, grant_status_for_skill, load_enabled
    from ouroboros.skill_repair_admission import record_repair_admission
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.message_ingress import ui_control_fact
    from tests.test_skill_review_persist_guard import _clean_wave

    h = hub_calendar
    assert h.events() == "('before', ['2026-11-25T10:00+03:00 Demo standup'])"
    if kept:  # Keep skips the upstream release: the installation the receipt names stays
        upstream = h.offer("upstream replacement")
        installer.run_in_lane(h.drive, "update", h.name, lambda: installer.update_installed(
            h.drive, h.name, mode="keep", digest=upstream, rows=[h.row], repo_dir=h.repo))
    if owner_off:
        assert run_skill_action(h.owner, h.name, "disable", _owner_actor="owner_ui")["ok"]
    sends = _clean_wave(monkeypatch)  # Only the external reviewer answer is synthetic.
    registry = ToolRegistry(repo_dir=h.repo, drive_root=h.drive)
    repair = registry._ctx
    repair.system_repo_dir = h.repo
    repair.task_id = "fixture-calendar-repair"
    repair.task_constraint = TaskConstraint(skill_name=h.name, payload_root=f"skills/ouroboroshub/{h.name}")
    repair.task_metadata = {"origin_suppressed": True, "ui_control": ui_control_fact(
        {"name": "skill_repair"}, pressed_at="2026-11-21T12:00:00Z")}
    record_repair_admission(h.drive, h.name, task_id=repair.task_id, base_content_hash=h.initial, owner_request=True)
    hashes = [h.initial]
    for before, after in [("before", "intermediate"), ("intermediate", "final")]:
        result = registry.execute("edit_text", {"root": "skill_payload", "path": "plugin.py",
                                               "old_str": repr(before), "new_str": repr(after)})
        assert "Replaced" in result, result
        current = find_skill(h.drive, h.name)
        hashes.append(current.content_hash)
        assert grant_status_for_skill(h.drive, current)["granted_keys"] == []  # edits do not grant
    assert len(set(hashes)) == 3
    reviewed = run_skill_review_lifecycle_blocking(repair, h.name, reviewers=["t1"], reason="fixture extension")
    assert (reviewed["status"], reviewed["content_hash"]) == ("clean", hashes[-1]), reviewed
    assert len(sends) == 1 and not reviewed.get("auto_granted_keys")
    grants = grant_status_for_skill(h.drive, find_skill(h.drive, h.name))
    assert (grants["granted_keys"], grants["granted_permissions"], grants["pending_permissions"]) == (
        [h.primary], [], ["notify_owner"]) and not grants["blocked"], "the field follows; the notice is only held"
    with pytest.raises(HostServiceAuthError, match="notify_owner"):  # the host's decision, no notice is sent
        HostServiceContext(h.drive).require_permission(h.name, {}, "notify_owner")
    if owner_off:
        assert not load_enabled(h.drive, h.name) and not extension_loader.is_extension_live(h.name, h.drive)
    else:  # the repaired files read the granted field's meeting while the notice still waits for its own grant
        assert h.events() == "('final', ['2026-11-25T10:00+03:00 Demo standup'])"
        assert run_skill_action(h.owner, h.name, "grant", expected_content_hash=hashes[-1], items=["notify_owner"],
                                _owner_actor="owner_ui")["ok"]
        HostServiceContext(h.drive).require_permission(h.name, {}, "notify_owner")  # its own grant admits it
        (skill_state_dir(h.drive, h.name) / "grants.json").unlink()  # the owner's decision withdrawn
        with pytest.raises(HostServiceAuthError, match="notify_owner"):
            HostServiceContext(h.drive).require_permission(h.name, {}, "notify_owner")


@pytest.mark.parametrize("automatic", [False, True])
def test_hub_update_keeps_auto_grant_a_separate_decision(hub_calendar, monkeypatch, automatic):
    from ouroboros.skill_loader import find_skill, grant_status_for_skill

    h = hub_calendar
    before = read_install_record(h.drive, h.name)
    digest = h.offer("upstream replacement")
    if automatic:
        monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    outcome = installer.run_in_lane(h.drive, "update", h.name, lambda: installer.update_installed(
        h.drive, h.name, mode="replace", digest=digest, local_digest=h.initial, rows=[h.row], repo_dir=h.repo))
    after = read_install_record(h.drive, h.name)
    assert before["release"]["digest"] == h.initial != after["release"]["digest"] == digest
    expected = [h.primary, h.other] if automatic else []
    assert outcome["acceptance"]["accepted"] and outcome["acceptance"]["granted_keys"] == expected
    status = grant_status_for_skill(h.drive, find_skill(h.drive, h.name))
    assert status["granted_keys"] == expected and status["pending_permissions"] == []  # a landing holds nothing
    assert status["blocked"] is not automatic
    if automatic:  # both fields granted by the policy: both local calendars are read
        assert h.events() == ("('upstream replacement', ['2026-11-25T10:00+03:00 Demo standup', "
                              "'2026-11-25T15:00+03:00 Demo review'])")


def test_hub_keep_continues_the_grant_receipt_and_replace_ends_it(hub_calendar):
    """The explicit grant's receipt names this installation: Keep skips a release and is no landing; Replace lands a
    new installation, and the receipt does not follow it."""
    from ouroboros.skill_grant_follow import installation_of, load_receipt
    from ouroboros.skill_loader import find_skill

    h = hub_calendar
    receipt = load_receipt(h.drive, h.name)
    assert (receipt["keys"], receipt["digest"], receipt["via"]) == ([h.primary], h.initial, "owner_ui")
    upstream = h.offer("upstream replacement")
    kept = installer.run_in_lane(h.drive, "update", h.name, lambda: installer.update_installed(
        h.drive, h.name, mode="keep", digest=upstream, rows=[h.row], repo_dir=h.repo))
    assert kept["mode"] == "keep" and load_receipt(h.drive, h.name) == receipt
    assert installation_of(h.drive, h.name, find_skill(h.drive, h.name).skill_dir) == receipt["installation"]
    installer.run_in_lane(h.drive, "update", h.name, lambda: installer.update_installed(
        h.drive, h.name, mode="replace", digest=upstream, local_digest=h.initial, rows=[h.row], repo_dir=h.repo))
    assert load_receipt(h.drive, h.name) == {}, "a new landing is a new installation"


@pytest.mark.parametrize("turn_on_when_ready", [True, False])
def test_an_accepted_update_brings_an_extension_that_is_on_back_at_its_new_release(drive, hub, live_extensions,
                                                                                    turn_on_when_ready):
    rows = hub.rows()
    _release_pinger(drive, hub, rows, "1.0.0")
    installed = installer.install_item(drive, "ouroboros/pinger", rows=rows, turn_on_when_ready=turn_on_when_ready)
    if not turn_on_when_ready:  # the install never asked to turn on: the owner's own Enable did
        assert installed["acceptance"]["turn_on"] == "not_requested"
        _owner(drive, "enable")
    assert extension_tool_answer(drive, "pinger") == "1.0.0"
    choice = _choice(drive)
    _release_pinger(drive, hub, rows, "2.0.0")
    acceptance = installer.update_installed(drive, "pinger", mode="replace", rows=rows)["acceptance"]
    assert extension_tool_answer(drive, "pinger") == "2.0.0", acceptance  # no manual reconcile, no restart
    assert acceptance["turn_on"] == ("already_on" if turn_on_when_ready else "not_requested")
    assert acceptance["extension_action"] == "extension_loaded"
    assert _choice(drive) == choice
    assert read_install_record(drive, "pinger")["turn_on_when_ready"] is turn_on_when_ready


@pytest.mark.parametrize(("held", "reason"), [("owner_off", "disabled"), ("new_key", "missing_grants")])
def test_an_accepted_update_leaves_an_extension_held_off_where_it_stands(drive, hub, live_extensions, held, reason):
    rows = hub.rows()
    _release_pinger(drive, hub, rows, "1.0.0")
    installer.install_item(drive, "ouroboros/pinger", rows=rows)
    if held == "owner_off":
        _owner(drive, "disable")
    choice = _choice(drive)
    _release_pinger(drive, hub, rows, "2.0.0", extra="env_from_settings: [GITHUB_TOKEN]\n" if held == "new_key" else "")
    acceptance = installer.update_installed(drive, "pinger", mode="replace", rows=rows)["acceptance"]
    assert extension_tool_answer(drive, "pinger") is None
    assert _choice(drive) == choice and read_install_record(drive, "pinger")["release"]["version"] == "2.0.0"
    assert acceptance["extension_reason"] == reason and "setup_error" not in acceptance, acceptance


def test_a_start_failure_after_an_accepted_update_keeps_the_files_and_the_owner_choice(drive, hub, live_extensions):
    import asyncio
    import time

    from ouroboros.gateway.skill_sources import _start_job
    from ouroboros.skill_lifecycle_queue import queue_snapshot

    rows = hub.rows()
    _release_pinger(drive, hub, rows, "1.0.0")
    installer.install_item(drive, "ouroboros/pinger", rows=rows)
    choice = _choice(drive)
    broken = _release_pinger(drive, hub, rows, "2.0.0",
                             plugin="def register(api):\n    raise RuntimeError('cannot start 2.0.0')\n")
    seen = {}

    async def owner_update():  # the Skills page's Update: a lifecycle job whose one line the page shows
        job_id = await _start_job("update", "pinger", "Updating pinger", lambda progress: seen.setdefault(
            "result", installer.update_installed(drive, "pinger", mode="replace", rows=rows, progress=progress)))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            events = [item for item in queue_snapshot()["events"] if item["id"] == job_id]
            if events and events[-1]["status"] not in {"queued", "running"}:
                return events[-1]
            await asyncio.sleep(0.1)
        pytest.fail("the update job did not finish")

    event = asyncio.run(owner_update())
    result = seen["result"]
    assert result["ok"] and read_install_record(drive, "pinger")["release"]["digest"] == broken  # the files landed
    assert event["status"] == "succeeded" and event["error"] == ""
    assert event["message"].startswith("update pinger: done; It could not start: "), event
    assert "cannot start 2.0.0" in event["message"]
    acceptance = result["acceptance"]
    assert acceptance["extension_action"] == "extension_load_error" and "cannot start 2.0.0" in acceptance["load_error"]
    assert event["message"] == f"update pinger: done; {acceptance['setup_error']}"
    assert extension_tool_answer(drive, "pinger") is None and _choice(drive) == choice  # still on: no revert
    _release_pinger(drive, hub, rows, "3.0.0")  # so the fixed release starts without a new Enable
    assert installer.update_installed(drive, "pinger", mode="replace", rows=rows)["acceptance"]["extension_action"] == (
        "extension_loaded")
    assert extension_tool_answer(drive, "pinger") == "3.0.0"


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


def test_credential_like_files_survive_git_install_update_and_merge_trees(drive, hub):
    """The Git copy owner refuses no credential-like name: install and replace land a release's `.env`,
    `credentials.json` and `id_rsa` byte for byte under its digest, and a merge lays out the new and the
    base release whole."""
    files = {".env": b"TOKEN=synthetic-first\n", "config/prod.env": b"MODE=fixture\n",
             "credentials.json": b'{"fixture": true}\n', "id_rsa": b"SYNTHETIC KEY FIXTURE\n"}

    def release(version):
        for rel, body in files.items():
            path = hub.work / "skills" / "keys" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        digest = hub.add_hosted("keys", version, official=True)
        hub.commit(f"keys {version}")
        _refresh(drive, rows[1:2])
        return digest

    rows = hub.rows()
    first = release("1.0.0")
    installer.install_item(drive, "ouroboros/keys", rows=rows)
    target = drive / "skills" / "ouroboroshub" / "keys"
    assert {rel: (target / rel).read_bytes() for rel in files} == files
    assert read_install_record(drive, "keys")["release"]["digest"] == first
    files[".env"] = b"TOKEN=synthetic-second\n"
    second = release("1.1.0")
    installer.update_installed(drive, "keys", mode="replace", digest=second, rows=rows)
    assert {rel: (target / rel).read_bytes() for rel in files} == files
    assert read_install_record(drive, "keys")["base_digest"] == second
    base = dict(files)
    (target / "SKILL.md").write_text((target / "SKILL.md").read_text() + "\nlocal edit\n")
    files["id_rsa"] = b"SYNTHETIC KEY FIXTURE TWO\n"
    release("2.0.0")
    merged = installer.update_installed(drive, "keys", mode="merge", rows=rows, actor="ouroboros")
    assert merged["base_available"] is True
    assert {rel: (drive / merged["new_tree"] / rel).read_bytes() for rel in files} == files
    assert {rel: (drive / merged["base_tree"] / rel).read_bytes() for rel in base} == base


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


def test_a_registry_browse_is_one_page_in_its_order_with_search_shaped_items(monkeypatch, tmp_path):
    from ouroboros.marketplace import clawhub

    asked = []

    def fake_search(query, **kwargs):
        asked.append((query, kwargs))
        return {"results": [clawhub._summary_from_record({"name": "self-improving-agent", "ownerHandle": "pskoett",
                                                          "summary": "Learns.", "stats": {"downloads": 491927}})],
                "next_cursor": "page-2"}

    monkeypatch.setattr(clawhub, "search", fake_search)
    row = {"id": "clawhub", "name": "ClawHub", "kind": "registry", "trust": "review", "confirmed": True}
    page = catalog.registry_browse(row)
    assert page["next_cursor"] == "page-2"
    assert asked[0][0] == "" and (asked[0][1]["sort"], asked[0][1]["cursor"]) == ("popular", None)
    assert asked[0][1]["limit"] == catalog.REGISTRY_BROWSE_PAGE
    item = page["items"][0]
    assert (item["id"], item["registry_slug"], item["installed"]) == ("clawhub/self-improving-agent", "self-improving-agent", None)
    assert (item["owner"], item["registry_stats"]) == ("pskoett", {"downloads": 491927})
    # The install path resolves exactly this id.
    assert catalog.find_item(tmp_path, item["id"], rows=[row])[1]["registry_slug"] == "self-improving-agent"
    catalog.registry_browse(row, sort="updated", cursor="abc")
    assert (asked[1][1]["sort"], asked[1][1]["cursor"]) == ("updated", "abc")
    catalog.registry_browse(row, sort="stars")
    assert asked[2][1]["sort"] == "popular", "an order the registry cannot give falls back to its default"


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


# --- the source_key equality seam between a catalog view and an installed copy ---


def test_source_key_names_the_exact_origin(drive, hub, tmp_path):
    from ouroboros.skill_sources import source_key

    official, community = hub.rows()[1:3]
    assert source_key(official) == source_key(community), "the two hub parts of one repository share a key"
    renamed = {**official, "id": "other-id", "name": "Renamed", "trust": "review"}
    assert source_key(renamed) == source_key(official), "id, name and trust are not the origin"
    assert source_key({**official, "ref": "elsewhere"}) != source_key(official)
    assert source_key({**official, "path": "skills"}) != source_key(official)
    assert source_key({"kind": "builtin"}) == "builtin"
    registry = {"kind": "registry", "url": "https://registry.clawhub.ai"}
    assert source_key(registry) == source_key({"kind": "registry", "url": "https://registry.clawhub.ai/"})
    assert source_key(registry) != source_key({"kind": "registry"})
    assert source_key({"kind": "unknown"}) is None
    folder = tmp_path / "skills-folder"
    key = source_key({"kind": "local", "path": str(folder)})
    assert key and key.startswith("local:") and str(folder) not in key, "the key never carries the path"
    assert key == source_key({"kind": "local", "path": os.path.join(str(tmp_path), "skills-folder")})
    assert key != source_key({"kind": "local", "path": str(tmp_path)})


def test_catalog_items_and_installed_rows_share_the_source_key_across_a_refresh(drive, hub, tmp_path):
    from ouroboros.gateway.management_contracts import InstalledSkill
    from ouroboros.gateway.skill_source_contracts import SkillCatalogItem
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_sources import source_key

    assert "source_key" in SkillCatalogItem.__optional_keys__ and "source_key" in InstalledSkill.__optional_keys__
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    view = catalog.catalog_view(drive, rows=rows[1:], repo_dir=builtin_repo(tmp_path))
    item = next(i for i in view["items"] if i["id"] == "ouroboros/weather")
    assert item["source_key"] == source_key(rows[1])
    installer.install_item(drive, "ouroboros/weather", rows=rows)
    listing = catalog.install_listing(drive, find_skill(drive, "weather"), catalog.cached_snapshots(drive, rows))
    assert listing["source_key"] == item["source_key"]
    hub.add_hosted("weather", "0.4.0", official=True, recommended=True)
    hub.commit("new release")
    _refresh(drive, rows[1:2])
    refreshed = next(i for i in catalog.catalog_view(drive, rows=rows[1:], repo_dir=builtin_repo(tmp_path))["items"]
                    if i["id"] == "ouroboros/weather")
    assert refreshed["digest"] != item["digest"]
    assert refreshed["source_key"] == item["source_key"], "the key is stable across a refresh"
    again = catalog.install_listing(drive, find_skill(drive, "weather"), catalog.cached_snapshots(drive, rows))
    assert again["source_key"] == item["source_key"]
    assert again["install"]["update_available"] == {"version": "0.4.0", "digest": refreshed["digest"]}


def test_only_the_installed_copy_of_the_same_release_lends_its_connector_facts(drive, hub):
    places = [{"id": "chats", "provides": "place", "label": "Relay chats", "destination": "vendor:Relay"}]
    _hosted_connector(hub, "relay_bridge", title="Relay Courier", contributes=places)
    hub.commit("connector 1.0")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    item = next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "relay_bridge")
    assert item["connects"] == ["Relay"]
    installer.install_item(drive, item["id"], rows=rows, turn_on_when_ready=False)
    folder = hub.work / "skills" / "relay_bridge"
    (folder / "skill.json").write_text(json.dumps({
        "name": "relay_bridge", "description": "relay_bridge relays chats", "version": "2.0.0", "type": "extension",
        "entry": "plugin.py", "permissions": []}), encoding="utf-8")
    hub.entries["relay_bridge"] = entry("relay_bridge", catalog.tree_release_digest(folder), version="2.0.0")
    hub.commit("connector 2.0")
    _refresh(drive, rows[1:2])
    new = next(i for i in catalog.catalog_view(drive, rows=rows[1:])["items"] if i["name"] == "relay_bridge")
    assert new["digest"] != item["digest"]
    assert new["connects"] == [], "no connector hints borrowed from the installed older release"
    assert new["installed"]["state"] == "installed"
    assert new["installed"]["source_key"] == item["source_key"] == new["source_key"]


def test_a_recordless_user_repo_copy_is_matched_by_its_enumerated_skill_folder(drive, tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_sources import source_key

    root = tmp_path / "my-skills"
    write_skill(root, "qa-city-guide")  # an ordinary child package of the connected root
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(root))
    row = {"id": "my-folder", "name": "My folder", "kind": "local", "path": str(root), "trust": "review",
           "confirmed": True}
    view = catalog.catalog_view(drive, rows=[row])
    item = next(i for i in view["items"] if i["name"] == "qa-city-guide")
    assert item["installed"]["state"] == "installed", "the enumerated child folder is the installed copy"
    assert item["installed"]["source_key"] == item["source_key"] == source_key(row)
    listing = catalog.install_listing(drive, find_skill(drive, "qa-city-guide", repo_path=str(root)),
                                      catalog.cached_snapshots(drive, [row]))
    assert listing["catalog_id"] == "my-folder/qa-city-guide" and listing["source_key"] == source_key(row)
    # A namesake package under another connected root is a different origin and claims nothing.
    other = tmp_path / "elsewhere"
    write_skill(other, "qa-city-guide", "9.9.9")
    other_row = {"id": "other", "name": "Other", "kind": "local", "path": str(other), "trust": "review",
                 "confirmed": True}
    view2 = catalog.catalog_view(drive, rows=[row, other_row])
    other_item = next(i for i in view2["items"] if i["source_id"] == "other")
    assert other_item["source_key"] != source_key(row)
    assert (other_item["installed"] or {}).get("state") != "installed"
    mine = next(i for i in view2["items"] if i["source_id"] == "my-folder")
    assert mine["installed"]["state"] == "installed", "the connected root still claims its own copy"


def test_a_converted_legacy_hub_record_keys_with_the_sidecars_branch_for_presentation(drive, hub):
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_sources import source_key

    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows)
    loaded = find_skill(drive, "weather")
    record = read_install_record(drive, "weather")
    for key in ("ref", "commit", "path"):
        record["source"].pop(key, None)
    record["source"]["url"] = "https://github.com/razzant/OuroborosHub"
    record["converted_from"] = ".ouroboroshub.json"
    write_install_record(drive, "weather", record)
    (loaded.skill_dir / ".ouroboroshub.json").write_text(json.dumps({
        "source": "ouroboroshub",
        "catalog_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json",
        "slug": "weather", "version": "0.3.2", "files": []}), encoding="utf-8")
    row = {"id": "hub-x", "name": "Hub X", "kind": "hub", "url": "https://github.com/razzant/OuroborosHub",
           "ref": "main", "trust": "trusted", "confirmed": True}
    snapshot = catalog.read_snapshot(drive, rows[1])
    main = {**snapshot, "entries": [{**e, "title": "Main title"} for e in snapshot["entries"]]}
    next_row = {**row, "id": record["source"]["id"], "ref": "next"}
    other = {**snapshot, "entries": [{**e, "title": "Next title", "digest": "f" * 64} for e in snapshot["entries"]]}
    snapshots = {next_row["id"]: (next_row, other), row["id"]: (row, main)}
    listing = catalog.install_listing(drive, loaded, snapshots)
    assert listing["source_key"] == source_key(row)
    assert (listing["catalog_id"], listing["catalog_title"]) == ("hub-x/weather", "Main title")
    assert listing["install"]["update_available"]["digest"] == "f" * 64, "legacy update selection is unchanged"
    items = [next(item for item in catalog.items_for(source, snap) if item["name"] == "weather")
             for source, snap in snapshots.values()]
    catalog._join_installed(items, catalog.installed_index(drive), {source["id"]: source for source, _ in snapshots.values()})
    assert [item["installed"]["state"] for item in items] == ["name_taken", "installed"]
    assert all(item["installed"]["source_key"] == source_key(row) for item in items)
    (loaded.skill_dir / ".ouroboroshub.json").unlink()
    without = catalog.install_listing(drive, loaded, snapshots)
    assert without["source_key"] is None, "without the sidecar's view the origin is unknown"
    assert without["catalog_id"] is None
