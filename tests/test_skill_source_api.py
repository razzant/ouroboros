"""HTTP surface of skill sources, the catalog and the install lifecycle, checked against its published shapes."""

from __future__ import annotations

import json
import time

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from ouroboros.skill_install_record import read_install_record
from tests._skill_source_fixtures import HubFixture, builtin_repo, join_source_refreshes, write_skill
from tests.test_client_contract_bindings import assert_bound


@pytest.fixture
def env(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    from ouroboros.skill_source_catalog import refresh_source

    drive = tmp_path / "data"
    (drive / "skills").mkdir(parents=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    hub = HubFixture(tmp_path / "remote")
    hub.add_hosted("weather", "0.3.2", official=True, recommended=True)
    hub.add_hosted("notes", category=["writing"])
    hub.commit()
    rows = hub.rows()
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    refresh_source(drive, rows[1])
    app = Starlette(routes=collect_routes(data_dir=drive))
    app.state.drive_root = drive
    app.state.repo_dir = builtin_repo(tmp_path)
    try:
        with TestClient(app) as client:
            yield client, drive, hub, tmp_path
    finally:
        join_source_refreshes()


def _wait(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_sources_list_probe_add_change_and_remove(env):
    client, _drive, hub, tmp_path = env
    listed = client.get("/api/skill-sources")
    assert assert_bound("GET /api/skill-sources", listed) == "SkillSourcesResponse"
    assert [row["id"] for row in listed.json()["sources"]] == ["builtin", "ouroboros", "ouroboros-community"]
    probe = client.post("/api/skill-sources/probe", json={"url": str(hub.bare)})
    assert assert_bound("POST /api/skill-sources/probe", probe)
    assert (probe.json()["kind"], probe.json()["owner"]) == ("hub", "m")
    folder = write_skill(tmp_path / "mine", "local_one").parent
    added = client.post("/api/skill-sources", json={"path": str(folder), "name": "Mine", "trust": "review"})
    assert assert_bound("POST /api/skill-sources", added) == "SkillSourceChangeResponse", added.json()
    assert added.json()["source"]["kind"] == "local"
    saved = json.loads((tmp_path / "settings.json").read_text())["OUROBOROS_SKILL_SOURCES"]
    assert [row["id"] for row in json.loads(saved)][-1] == "mine"
    changed = client.post("/api/skill-sources/ouroboros-community", json={"trust": "trusted", "confirmed": True})
    assert assert_bound("POST /api/skill-sources/{source_id}", changed)
    community = next(r for r in changed.json()["sources"] if r["id"] == "ouroboros-community")
    assert community["trust"] == "trusted" and community["confirmed"] is True
    removed = client.post("/api/skill-sources/mine/remove")
    assert assert_bound("POST /api/skill-sources/{source_id}/remove", removed)
    assert "mine" not in [row["id"] for row in removed.json()["sources"]]
    bad = client.post("/api/skill-sources", json={"url": "https://u:p@example.invalid/r", "name": "x", "trust": "review"})
    assert bad.status_code == 400 and assert_bound("POST /api/skill-sources", bad)
    refresh = client.post("/api/skill-sources/ouroboros/refresh")
    assert assert_bound("POST /api/skill-sources/{source_id}/refresh", refresh)


def test_a_row_a_build_supplied_keeps_its_managed_mark_through_an_edit(env, monkeypatch):
    client, _drive, hub, _tmp = env
    rows = hub.rows()
    rows[1]["managed"] = True
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    listed = client.get("/api/skill-sources")
    assert assert_bound("GET /api/skill-sources", listed) == "SkillSourcesResponse"
    assert [row.get("managed") for row in listed.json()["sources"]] == [None, True, None]
    changed = client.post("/api/skill-sources/ouroboros", json={"trust": "review", "managed": False})
    assert assert_bound("POST /api/skill-sources/{source_id}", changed)
    row = changed.json()["source"]
    assert row["trust"] == "review" and row["managed"] is True, "the mark locks nothing and no edit clears it"


def test_removing_settings_folder_clears_path_in_the_same_write(env, monkeypatch):
    import os
    from ouroboros import skill_loader, skill_sources
    from ouroboros.gateway import owner_settings

    client, drive, _hub, tmp_path = env
    folder = write_skill(tmp_path / "private", "private_skill").parent
    extra = write_skill(tmp_path / "extra", "extra_skill").parent
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(folder))
    added = client.post("/api/skill-sources", json={"path": str(extra), "name": "Extra", "trust": "review"})
    assert added.status_code == 200
    snapshots = []
    original = owner_settings._owner_update_settings

    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots.append(json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")))
        return result

    monkeypatch.setattr(owner_settings, "_owner_update_settings", write)
    removed_extra = client.post("/api/skill-sources/extra/remove")
    assert removed_extra.status_code == 200
    assert os.environ["OUROBOROS_SKILLS_REPO_PATH"] == str(folder)
    # Keep another folder present while removing the Settings-linked row.
    assert client.post("/api/skill-sources", json={"path": str(extra), "name": "Extra", "trust": "review"}).status_code == 200
    snapshots.clear()
    removed = client.post(f"/api/skill-sources/{skill_sources.LOCAL_ID}/remove")
    assert removed.status_code == 200, removed.text
    assert len(snapshots) == 1
    assert snapshots[0]["OUROBOROS_SKILLS_REPO_PATH"] == ""
    assert skill_sources.LOCAL_ID not in {row["id"] for row in json.loads(snapshots[0][skill_sources.SOURCES_SETTING])}
    assert os.environ["OUROBOROS_SKILLS_REPO_PATH"] == ""
    assert {s.name for s in skill_loader.discover_skills(drive)} == {"extra_skill"}
    # The saved document and the projected list agree after a reload; payloads stay in place.
    assert skill_sources.source_by_id(skill_sources.load_sources(snapshots[0]), skill_sources.LOCAL_ID) is None
    assert folder.is_dir() and extra.is_dir()


def test_catalog_search_and_detail(env):
    client, _drive, _hub, _tmp = env
    view = client.get("/api/skill-catalog")
    assert assert_bound("GET /api/skill-catalog", view) == "SkillCatalogResponse"
    ids = {item["id"] for item in view.json()["items"]}
    assert {"ouroboros/weather", "ouroboros-community/notes", "builtin/telegram"} <= ids
    found = client.get("/api/skill-catalog/search", params={"q": "weather"})
    assert assert_bound("GET /api/skill-catalog/search", found)
    assert [item["id"] for item in found.json()["items"]] == ["ouroboros/weather"]
    detail = client.get("/api/skill-catalog/item/ouroboros/weather")
    assert assert_bound("GET /api/skill-catalog/item/{source_id}/{name}", detail)
    assert "About it." in detail.json()["about"]
    missing = client.get("/api/skill-catalog/item/ouroboros/nope")
    assert missing.status_code == 404


def test_install_answers_the_job_at_once_and_refuses_a_taken_name(env, monkeypatch):
    client, drive, _hub, _tmp = env
    reviews = []
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", lambda _d, name, **_k: reviews.append(name))
    answer = client.post("/api/skill-catalog/install", json={"id": "ouroboros-community/notes"})
    assert assert_bound("POST /api/skill-catalog/install", answer) == "SkillCatalogInstallResponse"
    body = answer.json()
    assert body["ok"] and body["job_id"].startswith("skill-job-") and body["basis"] == "review"
    assert _wait(lambda: read_install_record(drive, "notes") is not None)
    assert _wait(lambda: reviews == ["notes"])  # a review install starts its review after landing
    write_skill(drive / "skills" / "external", "weather")
    taken = client.post("/api/skill-catalog/install", json={"id": "ouroboros/weather"})
    assert taken.status_code == 409 and taken.json()["code"] == "name_taken"
    assert assert_bound("POST /api/skill-catalog/install", taken)
    unknown = client.post("/api/skill-catalog/install", json={"id": "nope/x"})
    assert unknown.status_code == 404 and assert_bound("POST /api/skill-catalog/install", unknown)


def test_update_keep_then_uninstall(env):
    client, drive, hub, _tmp = env
    from ouroboros.skill_source_catalog import refresh_source
    from ouroboros.skill_source_install import install_item

    rows = hub.rows()
    install_item(drive, "ouroboros-community/notes", rows=rows)
    digest = hub.add_hosted("notes", "2.0.0", category=["writing"])
    hub.commit()
    refresh_source(drive, rows[2])
    stale = client.post("/api/skills/notes/update", json={"mode": "replace", "digest": "3" * 64})
    assert stale.status_code == 409 and stale.json()["code"] == "candidate_changed"
    kept = client.post("/api/skills/notes/update", json={"mode": "keep", "digest": digest})
    assert assert_bound("POST /api/skills/{skill}/update", kept) and kept.json()["skipped"]["digest"] == digest
    assert read_install_record(drive, "notes")["skipped_release_digest"] == digest
    gone = client.post("/api/skills/notes/uninstall")
    assert assert_bound("POST /api/skills/{skill}/uninstall", gone) and gone.json()["ok"]
    again = client.post("/api/skills/notes/uninstall")
    assert again.status_code == 404 and assert_bound("POST /api/skills/{skill}/uninstall", again)


def test_listing_rows_carry_install_facts_without_network(env, monkeypatch):
    client, drive, hub, _tmp = env
    from ouroboros import skill_source_git
    from ouroboros.skill_source_install import install_item

    install_item(drive, "ouroboros-community/notes", rows=hub.rows())
    monkeypatch.setattr(skill_source_git, "update_clone", lambda *a, **k: pytest.fail("the listing fetched"))
    listing = client.get("/api/extensions")
    assert assert_bound("GET /api/extensions", listing, closed=False) == "ExtensionsIndexResponse"
    row = next(r for r in listing.json()["skills"] if r["name"] == "notes")
    assert row["install"]["source"]["id"] == "ouroboros-community" and row["install"]["basis"] == "review"
    assert row["install"]["modified"] is False and row["category"] == ["writing"]


def test_starter_skills_for_onboarding(env):
    client, _drive, _hub, _tmp = env
    answer = client.get("/api/onboarding/starter-skills")
    assert assert_bound("GET /api/onboarding/starter-skills", answer) == "StarterSkillsResponse"
    groups = answer.json()["groups"]
    items = {item["id"]: item for group in groups for item in group["items"]}
    assert items["builtin/telegram"]["recommended"] and not items["builtin/unix_computer_use"]["recommended"]
    assert items["ouroboros/weather"]["recommended"] and items["ouroboros/weather"]["official"]
    assert "ouroboros-community/notes" not in items  # the offer is built-in and Official skills
    assert any(item["builtin"] for item in groups[0]["items"])
    assert {row["id"] for row in answer.json()["hub_rows"]} == {"ouroboros", "ouroboros-community"}


def test_retired_hub_routes_are_gone_and_the_receipt_clear_moved(env):
    client, drive, _hub, _tmp = env
    for path in ("/api/marketplace/ouroboroshub/catalog", "/api/marketplace/ouroboroshub/preview/notes"):
        assert client.get(path).status_code == 404
    assert client.post("/api/marketplace/ouroboroshub/install", json={"slug": "x"}).status_code == 404
    from ouroboros.marketplace.provenance import write_publication_record

    shown = {"slug": "demo", "version": "1", "content_hash": "a" * 64, "repository": "o/r", "pr_number": 3,
             "pr_url": "https://github.com/o/r/pull/3", "published_at": "2026-10-17T00:00:00Z"}
    write_publication_record(drive, "demo", shown)
    cleared = client.post("/api/skills/demo/publication/clear", json={"expected_published": shown})
    assert assert_bound("POST /api/skills/{skill}/publication/clear", cleared) and cleared.json()["publication_cleared"]


def test_publish_preflight_binds_the_destination_the_owner_chose(env, monkeypatch):
    client, drive, hub, _tmp = env
    rows = hub.rows() + [{"id": "mine", "name": "Mine", "kind": "git", "url": "https://github.com/alice/skills",
                          "ref": "main", "trust": "trusted", "confirmed": True}]
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    write_skill(drive / "skills" / "external", "demo")
    chosen = client.post("/api/skills/demo/publish-preflight", json={"destination": {"source_id": "mine",
                                                                                     "mode": "entry"}})
    assert assert_bound("POST /api/skills/{skill}/publish-preflight", chosen) == "SkillPublishPreflightResponse"
    assert chosen.json()["repository"] == "alice/skills"
    unknown = client.post("/api/skills/demo/publish-preflight", json={"destination": {"source_id": "nope",
                                                                                      "mode": "entry"}})
    assert unknown.json()["state"] == "hard_block" and unknown.json()["reason_code"] == "destination_unknown"
    assert assert_bound("POST /api/skills/{skill}/publish-preflight", unknown)


def test_frozen_hub_installed_endpoint_includes_hub_records_and_legacy_bucket(env):
    from ouroboros.marketplace.provenance import write_provenance
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_source_install import install_item

    client, drive, hub, _tmp = env
    install_item(drive, "ouroboros-community/notes", rows=hub.rows())
    write_skill(drive / "skills" / "ouroboroshub", "legacy")
    write_skill(drive / "skills" / "external", "other_hub")
    write_install_record(drive, "other_hub", {"source": {"id": "elsewhere", "kind": "hub"},
                                             "entry": "other_hub", "basis": "review", "release": {"digest": "1" * 64}})
    write_skill(drive / "skills" / "external", "local")
    # Plain git installs share the historical hub bucket but have an explicit non-hub origin.
    for name, kind in (("plain_git", "git"), ("recorded_registry", "registry")):
        write_skill(drive / "skills" / "ouroboroshub", name)
        write_install_record(drive, name, {"source": {"id": name, "kind": kind}, "entry": name,
                                         "basis": "review", "release": {"digest": "2" * 64}})
    registry_dir = write_skill(drive / "skills" / "clawhub", "registry")
    (registry_dir / ".clawhub.json").write_text('{"source": "clawhub"}', encoding="utf-8")
    write_provenance(drive, "registry", {"source": "clawhub", "slug": "registry"})
    response = client.get("/api/marketplace/ouroboroshub/installed")
    assert response.status_code == 200
    assert assert_bound("GET /api/marketplace/ouroboroshub/installed", response) == "MarketplaceInstalledResponse"
    assert response.json()["count"] == 3
    assert {s["name"] for s in response.json()["skills"]} == {"notes", "legacy", "other_hub"}
    registry = client.get("/api/marketplace/clawhub/installed").json()["skills"][0]
    assert set(response.json()["skills"][0]) == set(registry) - {"provenance"}


def test_http_source_probe_exposes_the_encryption_warning(env, monkeypatch):
    client, _drive, _hub, _tmp = env
    monkeypatch.setattr("ouroboros.skill_source_git.fetch_tree", lambda target, *_a: write_skill(target, "sample"))
    response = client.post("/api/skill-sources/probe", json={"url": "http://git.example.invalid/team/skills"})
    assert response.status_code == 200
    assert assert_bound("POST /api/skill-sources/probe", response) == "SkillSourceProbeResponse"
    assert response.json()["warning"] == "not_encrypted"


def test_update_without_review_is_owner_http_replace_only(env, monkeypatch):
    from ouroboros.skill_source_install import install_item

    client, drive, hub, _tmp = env
    reviews = []
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", lambda *a, **k: reviews.append(a))
    install_item(drive, "ouroboros-community/notes", rows=hub.rows())
    for body in ({"mode": "keep", "skip_review": True}, {"mode": "merge", "skip_review": True},
                 {"mode": "replace", "skip_review": "false"}):
        assert client.post("/api/skills/notes/update", json=body).status_code == 400
    result = client.post("/api/skills/notes/update", json={"mode": "replace", "skip_review": True})
    assert result.status_code == 200 and assert_bound("POST /api/skills/{skill}/update", result)
    assert _wait(lambda: (read_install_record(drive, "notes") or {}).get("basis") == "owner_skip")
    assert not reviews
