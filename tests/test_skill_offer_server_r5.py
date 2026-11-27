"""R5 server-side fixes for selected-offer identity, on top of R4 66f1db9a: pinned install selectors
(``expected_source_key``/``expected_digest``/registry ``version``), registry origin from the recorded
provenance, real local package entry names, the cold-cache detail address, lending only from unmodified
copies, case-insensitive registry prefix, and the marketplace reuse origin guard."""

from __future__ import annotations

import json

import pytest

from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_install as installer
from ouroboros.marketplace import install as marketplace_install
from ouroboros.marketplace.provenance import write_provenance
from ouroboros.skill_install_record import read_install_record, write_install_record
from ouroboros.skill_loader import find_skill
from ouroboros.skill_sources import effective_registry_url, resolved_folder, source_key
from tests._skill_source_fixtures import HubFixture, entry, join_source_refreshes
from ouroboros.skill_source_catalog import tree_release_digest


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
    fixture.add_hosted("notes", "1.0.0")
    fixture.commit("first")
    return fixture


def _refresh(drive, rows):
    for row in rows:
        if row["kind"] in {"hub", "git"}:
            meta = catalog.refresh_source(drive, row)
            assert not meta.get("error"), meta


REGISTRY_ROW = {"id": "clawhub", "name": "ClawHub", "kind": "registry", "trust": "review", "confirmed": True}


def test_plan_install_checks_the_cards_source_key_and_digest(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    item_id = "ouroboros/weather"
    plan = installer.plan_install(drive, item_id, rows=rows)
    item = plan["item"]
    # matching selectors pass; absent fields keep the old behavior
    installer.plan_install(drive, item_id, rows=rows)
    installer.plan_install(drive, item_id, rows=rows, expected_source_key=item["source_key"],
                           expected_digest=item["digest"])
    with pytest.raises(installer.SkillInstallError) as bad_digest:
        installer.plan_install(drive, item_id, rows=rows, expected_digest="f" * 64)
    assert bad_digest.value.code == "candidate_changed" and bad_digest.value.status == 409
    assert bad_digest.value.facts["candidate"] == {"version": item["version"], "digest": item["digest"],
                                                    "source_key": item["source_key"]}
    with pytest.raises(installer.SkillInstallError) as bad_key:
        installer.plan_install(drive, item_id, rows=rows, expected_source_key="git:" + "0" * 32)
    assert bad_key.value.code == "candidate_changed"


def test_plan_install_checks_a_registry_rows_source_key_and_ignores_digest_at_plan_time(drive):
    plan = installer.plan_install(drive, "clawhub/ns/slug", rows=[REGISTRY_ROW],
                                  expected_source_key=source_key(REGISTRY_ROW), expected_digest="d" * 64)
    assert plan["row"]["kind"] == "registry"  # the stub carries no digest; landing checks it
    with pytest.raises(installer.SkillInstallError) as bad_key:
        installer.plan_install(drive, "clawhub/ns/slug", rows=[REGISTRY_ROW],
                               expected_source_key="registry:" + "0" * 32)
    assert bad_key.value.code == "candidate_changed"


def test_registry_version_and_digest_reach_the_marketplace_selector(drive, monkeypatch):
    captured = {}

    class _Result:
        ok = True
        sanitized_name = "slug"
        error = ""
        error_status = 0

    def fake_install_skill(*args, **kwargs):
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(marketplace_install, "install_skill", fake_install_skill)
    installer.install_item(drive, "clawhub/ns/slug", rows=[REGISTRY_ROW], actor="ouroboros",
                           version="1.2.3", expected_digest="d" * 64, expected_source_key=source_key(REGISTRY_ROW))
    assert captured["version"] == "1.2.3"
    assert captured["expected_digest"] == "d" * 64
    assert captured["registry_url"] is None
    installer.install_item(drive, "clawhub/ns/slug", rows=[REGISTRY_ROW], actor="ouroboros")  # legacy call
    assert captured["version"] is None and captured["expected_digest"] == ""


def test_a_default_registry_record_keys_from_the_provenances_recorded_origin(drive):
    default_origin = effective_registry_url(None)
    write_install_record(drive, "quotas", {"source": {"id": "clawhub", "kind": "registry", "name": "ClawHub"},
                                           "entry": "quotas", "basis": "review", "release": {"digest": "3" * 64}})
    write_provenance(drive, "quotas", {"source": "clawhub", "slug": "quotas", "registry_url": default_origin})
    record = read_install_record(drive, "quotas")
    fields = catalog._record_source_fields(record, "", local_name="quotas", drive_root=drive)
    assert catalog._record_key(record, fields) == source_key({"kind": "registry"}), \
        "the provenance's origin matches the connected default row"
    legacy = {**record, "converted_from": ".clawhub.json"}
    assert catalog._record_key(legacy, catalog._record_source_fields(legacy, "", local_name="quotas",
                                                                     drive_root=drive)) is None
    write_install_record(drive, "nopro", {"source": {"id": "clawhub", "kind": "registry"}, "entry": "nopro",
                                         "basis": "review", "release": {"digest": "4" * 64}})
    nopro = read_install_record(drive, "nopro")
    assert catalog._record_key(nopro, catalog._record_source_fields(nopro, "", local_name="nopro",
                                                                    drive_root=drive)) is None, \
        "without a recorded origin it stays unknown, never today's setting"
    assert catalog._record_key(nopro, catalog._record_source_fields(nopro, "", drive_root=drive)) is None, \
        "without an actual local name nothing is read and nothing is lent"
    # A custom registry row id (any id, not `clawhub`) with the same ClawHub provider provenance
    # lends the recorded URL just as the default row does; a different provider label lends nothing.
    write_install_record(drive, "tools", {"source": {"id": "tools", "kind": "registry"}, "entry": "toolbelt",
                                       "basis": "review", "release": {"digest": "5" * 64}})
    write_provenance(drive, "tools", {"source": "clawhub", "slug": "toolbelt", "registry_url": default_origin})
    tools = read_install_record(drive, "tools")
    fields = catalog._record_source_fields(tools, "", local_name="tools", drive_root=drive)
    assert str(fields.get("url") or "") == default_origin, \
        "the provenance's provider label names the ClawHub registry, whatever the row id is"
    assert catalog._record_key(tools, fields) == source_key({"kind": "registry", "url": default_origin})
    write_provenance(drive, "tools", {"source": "otherhub", "slug": "toolbelt", "registry_url": default_origin})
    other = read_install_record(drive, "tools")
    assert not str(catalog._record_source_fields(other, "", local_name="tools", drive_root=drive).get("url") or ""), \
        "a provenance recorded by a different provider lends no URL"


def test_a_single_package_root_is_named_by_its_manifest_not_its_folder(tmp_path):
    pkg = tmp_path / "renamed-folder"
    pkg.mkdir()
    (pkg / "SKILL.md").write_text(
        "---\nname: logical_name\ndescription: d\nversion: 1.0.0\ntype: instruction\n---\nbody\n", encoding="utf-8")
    row = {"id": "local1", "name": "Local", "kind": "local", "path": str(pkg), "trust": "review", "confirmed": True}
    snap = catalog.parse_plain_tree(pkg)
    assert [e["name"] for e in snap["entries"]] == ["logical_name"]
    assert catalog._local_entry_dir(row, snap, "logical_name") == resolved_folder(pkg)
    assert catalog._local_entry_name(row, snap, pkg, "renamed-folder") == "logical_name"
    assert catalog._local_entry_name(row, snap, tmp_path / "elsewhere", "renamed-folder") == "renamed-folder"


def test_presenting_carries_the_detail_address_when_only_the_cache_is_cold(drive, hub):
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows, actor="ouroboros")
    record = read_install_record(drive, "weather")
    loaded = find_skill(drive, "weather")
    fields = catalog._record_source_fields(record, loaded.skill_dir, drive_root=drive)
    key = catalog._record_key(record, fields)
    current = catalog.installed_release_digest(loaded)
    warm = catalog.cached_snapshots(drive, rows)
    shown, listed = catalog._presenting(record, warm, fields, key, current)
    assert listed == "ouroboros/weather" and shown is not None
    cold = {r["id"]: (r, None) for r in rows}
    shown_cold, listed_cold = catalog._presenting(record, cold, fields, key, current)
    assert (shown_cold, listed_cold) == (None, "ouroboros/weather"), \
        "the detail address is carried; no title or offer is invented from the cold cache"
    assert catalog._presenting(record, cold, fields, "git:" + "0" * 32, current) == (None, None), \
        "a mismatched or unknown source stays null"


def test_join_installed_lends_connector_facts_only_from_the_unmodified_copy(drive, hub):
    places = [{"id": "chats", "provides": "place", "label": "Relay chats", "destination": "vendor:Relay"}]
    folder = hub.work / "skills" / "relay_bridge"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "skill.json").write_text(json.dumps({
        "name": "relay_bridge", "description": "relays", "version": "1.0.0", "type": "extension",
        "entry": "plugin.py", "permissions": ["presence"], "title": "Relay Courier", "contributes": places,
        "ui_tab": {"tab_id": "desk", "title": "Relay Transport"}}), encoding="utf-8")
    hub.entries["relay_bridge"] = entry("relay_bridge", tree_release_digest(folder))
    hub.commit("connector")
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros-community/relay_bridge", rows=rows, turn_on_when_ready=False)
    record = read_install_record(drive, "relay_bridge")
    skill_dir = str((drive / "skills" / "ouroboroshub" / "relay_bridge").resolve())
    fields = catalog._record_source_fields(record, skill_dir, drive_root=drive)
    key = catalog._record_key(record, fields)
    index = catalog.installed_index(drive)
    snapshots = catalog.cached_snapshots(drive, rows)
    rows_by_id = {r["id"]: r for r in rows}

    def fresh_item():
        return {"name": "relay_bridge", "source_id": "ouroboros-community", "source_key": key,
                "digest": record["release"]["digest"], "version": "1.0.0", "title": "",
                "widget_title": "", "connects": [], "installed": None, "withdrawn": None}

    fresh = fresh_item()
    catalog._join_installed([fresh], index, rows_by_id, snapshots, drive_root=drive)
    assert fresh["installed"]["state"] == "installed"
    assert fresh["connects"] == ["Relay"], "the unmodified copy lends its manifest's connector"

    (drive / "skills" / "ouroboroshub" / "relay_bridge" / "plugin.py").write_text(
        "def register(api):\n    pass\n# edited\n", encoding="utf-8")
    edited = fresh_item()
    catalog._join_installed([edited], index, rows_by_id, snapshots, drive_root=drive)
    assert edited["installed"]["state"] == "installed"
    assert edited["connects"] == [], "an edited copy borrows nothing"


def test_registry_prefix_strips_case_insensitively_and_keeps_the_querys_case(monkeypatch):
    from ouroboros.marketplace import clawhub

    asked = []
    monkeypatch.setattr(clawhub, "search", lambda q, **k: asked.append(q) or {"results": []})
    catalog.registry_items(REGISTRY_ROW, "CLAWHUB/ns/Slug")
    assert asked[-1] == "ns/Slug"
    catalog.registry_items(REGISTRY_ROW, "clawhub/ns/Other")
    assert asked[-1] == "ns/Other"
    catalog.registry_items(REGISTRY_ROW, "weather")
    assert asked[-1] == "weather"


def test_existing_clawhub_provenance_also_checks_the_recorded_origin(tmp_path):
    target = tmp_path / "skills" / "clawhub" / "demo"
    target.mkdir(parents=True)
    (target / ".clawhub.json").write_text(
        json.dumps({"source": "clawhub", "slug": "demo", "sanitized_name": "demo"}), encoding="utf-8")
    write_provenance(tmp_path, "demo", {"source": "clawhub", "slug": "demo", "sanitized_name": "demo",
                                        "registry_url": "https://registry-a.invalid"})
    valid = marketplace_install._valid_existing_clawhub_provenance
    assert valid(tmp_path, "demo", target, slug="demo", route_url="https://registry-a.invalid") is not None
    assert valid(tmp_path, "demo", target, slug="demo", route_url="https://registry-b.invalid") is None, \
        "a provenance naming another registry is a same-slug lookalike"
    write_provenance(tmp_path, "demo", {"source": "clawhub", "slug": "demo", "sanitized_name": "demo"})
    assert valid(tmp_path, "demo", target, slug="demo", route_url="https://registry-b.invalid") is not None, \
        "an absent recorded origin keeps the ordinary compatibility when truly unknown"
    assert valid(tmp_path, "demo", target, slug="demo") is not None, "a legacy call without a route is unguarded"


def test_pinned_registry_origin_prefers_the_provenance_and_a_conflict_pins_none():
    pinned = marketplace_install._pinned_registry_origin
    assert pinned({"source": {"url": "https://a.invalid"}}, {"registry_url": "https://b.invalid"}) is None, \
        "a record naming another registry than the provenance is not followed"
    assert pinned({"source": {"url": "https://a.invalid/"}}, {"registry_url": "https://a.invalid"}) == "https://a.invalid"
    assert pinned({}, {"registry_url": "https://b.invalid"}) == "https://b.invalid"
    assert pinned({"source": {"url": "https://a.invalid"}}, None) == "https://a.invalid"
    assert pinned({}, None) == ""


def test_a_follow_up_claims_only_the_selected_release_and_repairs_only_this_offer(tmp_path):
    from ouroboros.skill_install_record import write_install_record

    target = tmp_path / "skills" / "clawhub" / "demo"
    target.mkdir(parents=True)
    origin = effective_registry_url(None)
    (target / ".clawhub.json").write_text(json.dumps({"source": "clawhub", "slug": "demo", "sanitized_name": "demo",
                                                      "version": "1.0.0"}), encoding="utf-8")
    write_provenance(tmp_path, "demo", {"source": "clawhub", "slug": "demo", "sanitized_name": "demo",
                                        "version": "1.0.0", "sha256": "abc", "registry_url": origin})
    write_install_record(tmp_path, "demo", {"source": {"id": "clawhub", "kind": "registry"}, "entry": "demo",
                                            "basis": "review", "release": {"version": "1.0.0", "digest": "1" * 64}})
    valid = marketplace_install._valid_existing_clawhub_provenance
    assert valid(tmp_path, "demo", target, slug="demo", release=("1.0.0", "abc", "")) is not None
    assert valid(tmp_path, "demo", target, slug="demo", release=("1.0.0", "abc", "1" * 64)) is not None
    for other in (("2.0.0", "abc", ""), ("1.0.0", "def", ""), ("1.0.0", "abc", "2" * 64)):
        assert valid(tmp_path, "demo", target, slug="demo", release=other) is None, other
    repairable = marketplace_install._has_repairable_clawhub_partial
    assert repairable(tmp_path, "demo", target, slug="demo", version="1.0.0")
    assert not repairable(tmp_path, "demo", target, slug="demo", version="2.0.0"), "another release stays installed"
    assert not repairable(tmp_path, "demo", target, slug="acme/demo", version="1.0.0"), "another slug is not replaced"
    assert not repairable(tmp_path, "demo", target, slug="demo", route_url="https://registry-b.invalid",
                          version="1.0.0"), "another registry is not replaced"
    assert repairable(tmp_path, "demo", target, slug="demo", version="1.0.0", sha256="abc", digest="1" * 64)
    assert repairable(tmp_path, "demo", target, slug="demo", version="1.0.0"), "unsupplied evidence stays compatible"
    assert not repairable(tmp_path, "demo", target, slug="demo", version="1.0.0", sha256="def"), "other archive hash"
    assert not repairable(tmp_path, "demo", target, slug="demo", version="1.0.0", digest="2" * 64), "other digest"
    sidecar = {"source": "clawhub", "slug": "demo", "sanitized_name": "demo", "version": "1.0.0"}
    (target / ".clawhub.json").write_text(json.dumps({**sidecar, "sha256": "abc"}), encoding="utf-8")
    assert valid(tmp_path, "demo", target, slug="demo", release=("1.0.0", "abc", "")) is not None
    assert repairable(tmp_path, "demo", target, slug="demo", version="1.0.0", sha256="abc"), "same release metadata"
    (target / ".clawhub.json").write_text(json.dumps({**sidecar, "sha256": "def"}), encoding="utf-8")
    assert valid(tmp_path, "demo", target, slug="demo", release=("1.0.0", "abc", "")) is None, "sidecar hash differs"
    assert not repairable(tmp_path, "demo", target, slug="demo", version="1.0.0", sha256="abc"), "sidecar hash differs"


def test_presenting_invents_no_detail_address_for_a_registry_or_another_rows_cold_cache(drive, hub):
    url = effective_registry_url(None)
    record = {"source": {"id": "clawhub", "kind": "registry", "name": "ClawHub", "url": url}, "entry": "acme/notes",
              "release": {"digest": "5" * 64}}
    fields = dict(record["source"])
    key = catalog._record_key(record, fields)
    assert key == source_key(REGISTRY_ROW)
    assert catalog._presenting(record, {"clawhub": (REGISTRY_ROW, None)}, fields, key, "") == (None, None)
    rows = hub.rows()
    _refresh(drive, rows[1:2])
    installer.install_item(drive, "ouroboros/weather", rows=rows, actor="ouroboros")
    weather = read_install_record(drive, "weather")
    loaded = find_skill(drive, "weather")
    hub_fields = catalog._record_source_fields(weather, loaded.skill_dir, drive_root=drive)
    hub_key = catalog._record_key(weather, hub_fields)
    others = {r["id"]: (r, None) for r in rows if r["id"] != "ouroboros"}
    assert catalog._presenting(weather, others, hub_fields, hub_key,
                               catalog.installed_release_digest(loaded)) == (None, None), \
        "a cold row of the same key that is not the record's own may read another part"


def test_an_uncached_directly_attached_package_is_named_by_its_manifest_and_an_empty_path_walks_nothing(tmp_path):
    pkg = tmp_path / "renamed-folder"
    pkg.mkdir()
    (pkg / "SKILL.md").write_text(
        "---\nname: logical_name\ndescription: d\nversion: 1.0.0\ntype: instruction\n---\nbody\n", encoding="utf-8")
    row = {"id": "local1", "name": "Local", "kind": "local", "path": str(pkg), "trust": "review", "confirmed": True}
    assert catalog._same_local_copy(row, pkg, None, "renamed-folder", {})
    assert catalog._local_entry_name(row, None, pkg, "renamed-folder") == "logical_name"
    assert catalog._local_entry_name(row, None, tmp_path / "elsewhere", "renamed-folder") == "renamed-folder"
    # The join: the copy loaded from the attached folder under its folder's name is that entry's; a namesake of the
    # manifest name loaded from a separate root stays that root's.
    other = tmp_path / "separate-root"
    (other / "logical_name").mkdir(parents=True)
    (other / "logical_name" / "SKILL.md").write_text(
        "---\nname: logical_name\ndescription: d\nversion: 2.0.0\ntype: instruction\n---\nbody\n", encoding="utf-8")
    other_row = {**row, "id": "local2", "path": str(other)}
    index = {"renamed-folder": {"name": "renamed-folder", "skill_dir": resolved_folder(pkg), "locations": ["user_repo"]},
             "logical_name": {"name": "logical_name", "skill_dir": resolved_folder(other / "logical_name"),
                              "locations": ["user_repo"]}}
    snaps = {"local1": catalog.parse_plain_tree(pkg), "local2": catalog.parse_plain_tree(other)}
    mine, = catalog.items_for(row, snaps["local1"])
    theirs, = catalog.items_for(other_row, snaps["local2"])
    catalog._join_installed([mine, theirs], index, {"local1": row, "local2": other_row}, snaps)
    assert mine["installed"]["state"] == "installed" and mine["installed"]["source_key"] == source_key(row)
    assert theirs["installed"]["state"] == "installed" and theirs["installed"]["source_key"] == source_key(other_row)
    del index["renamed-folder"]
    alone, = catalog.items_for(row, snaps["local1"])
    catalog._join_installed([alone], index, {"local1": row}, snaps)
    assert alone["installed"]["state"] != "installed", "a namesake under another root is not the attached copy"
    bare = {**row, "path": ""}
    assert not catalog._same_local_copy(bare, pkg, None, "renamed-folder", {}), "no folder walks nothing"
    assert catalog._local_entry_dir(bare, catalog.parse_plain_tree(pkg), "logical_name") == ""


def test_registry_record_reads_provenance_under_the_installed_local_name(drive):
    slug, local = "acme/notes", "acme__notes"
    payload = drive / "skills" / "clawhub" / local
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text(
        "---\nname: acme__notes\ndescription: d\nversion: 1.0.0\ntype: instruction\n---\nbody\n", encoding="utf-8")
    write_install_record(drive, local, {"source": {"id": "clawhub", "kind": "registry", "name": "ClawHub"},
                                        "entry": slug, "basis": "review", "release": {"digest": "5" * 64}})
    write_provenance(drive, local, {"source": "clawhub", "slug": slug, "sanitized_name": local,
                                     "registry_url": effective_registry_url(None)})
    record = read_install_record(drive, local)
    fields = catalog._record_source_fields(record, str(payload), local_name=local, drive_root=drive)
    assert str(fields.get("url") or "") == effective_registry_url(None), \
        "provenance is read under the installed local name, not the namespaced slug"
    assert catalog._record_key(record, fields) == source_key({"kind": "registry"})
    nolocal = catalog._record_source_fields(record, str(payload), drive_root=drive)
    assert not str(nolocal.get("url") or "") and catalog._record_key(record, nolocal) is None, \
        "no actual local name: the origin stays unknown, with no slug fallback and no scan"
    write_provenance(drive, local, {"source": "clawhub", "slug": "acme/other", "sanitized_name": local,
                                     "registry_url": effective_registry_url(None)})
    namesake = catalog._record_source_fields(record, str(payload), local_name=local, drive_root=drive)
    assert not str(namesake.get("url") or "") and catalog._record_key(record, namesake) is None, \
        "a provenance naming another record never lends this one its URL"
