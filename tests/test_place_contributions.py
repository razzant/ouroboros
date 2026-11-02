"""Places are real providers, with no speech-engine or host-page requirement."""
from dataclasses import replace

import pytest

from ouroboros.extension_ui_validation import validate_contributions
from ouroboros.extension_contributions import project_contributions
from ouroboros import skill_source_catalog as catalog
from tests.test_skill_contributions import manifest, TEMPLATE
from tests.test_extension_icons import PLACE, staged_icon


def test_place_has_own_contract_and_full_projection():
    entry = {**PLACE, "scope": "conversation", "wakes": ["mention", "reply_to_me"], "icon": "mail",
             "settings": "connection", "read_only": "provider", "source": {"route": "status", "refresh_s": 60}}
    clean, errors = validate_contributions(manifest([entry], kind="extension", permissions=["presence"]))
    assert not errors and clean[0]["wakes"] == entry["wakes"]
    rows = project_contributions("mail", clean, "revision", tools=[{"name": "send", "effect": "message", "target": "to"}])
    assert rows[0]["provides"] == "place" and "runtime" not in rows[0] and "route_url" not in rows[0]
    assert rows[0]["source"]["url"] == "/api/extensions/mail/status"
    assert rows[0]["icon_spec"] == {"kind": "host", "name": "mail"}
    assert rows[0]["tools"] == [{"name": "ext_6_r_mail_send", "effect": "message", "target": "to"}]


@pytest.mark.parametrize("patch", [{"destination": "host"}, {"destination": "vendor:"}, {"scope": "workspace"},
    {"wakes": ["a keyword"]}, {"wakes": [{}]}, {"wakes": "mention"}, {"settings": "../x"},
    {"read_only": True}, {"read_only": "host"}, {"source": {"route": "../status", "refresh_s": 60}},
    {"source": {"route": "status", "refresh_s": 1}}, {"runtime": "server"}, {"route": "send"}])
def test_invalid_place_drops_only_it_and_valid_place_still_works(patch):
    invalid = {**PLACE, **patch, "id": "invalid"}
    clean, errors = validate_contributions(manifest([PLACE, invalid, TEMPLATE], kind="extension", permissions=["presence"]))
    assert [row["id"] for row in clean] == ["place", "simpler"]
    assert len(errors) == 1 and "contributes.invalid" in errors[0]


@pytest.mark.parametrize("kind,permissions", [("instruction", ["presence"]), ("extension", [])])
def test_place_requires_extension_and_presence_grant_declaration(kind, permissions):
    assert validate_contributions(manifest([PLACE], kind=kind, permissions=permissions))[1]
    assert not validate_contributions(manifest([PLACE], kind="extension", permissions=["presence"]))[1]


def test_installed_connector_derived_without_hub_and_catalog_tolerates_optional_fields(tmp_path, monkeypatch):
    skill, staged = staged_icon(tmp_path)
    monkeypatch.setattr("ouroboros.skill_install_record.read_install_record", lambda *a: None)
    listing = catalog.install_listing(tmp_path, skill, {})
    assert listing["connects"] == ["Service"]
    assert catalog._manifest_facts(replace(skill.manifest, contributes=[]))["connects"] == []
    snapshot = catalog.parse_plain_tree(skill.skill_dir)
    row = {"id": "local", "kind": "local", "name": "Local", "trust": "review"}
    assert catalog.items_for(row, snapshot)[0]["icon"]["kind"] == "svg"
    assert catalog.items_for(row, snapshot)[0]["connects"] == ["Service"]
    snapshot["entries"][0].pop("connects")
    snapshot["entries"][0].pop("icon")
    legacy = catalog.items_for(row, snapshot)[0]
    assert legacy["connects"] == [] and legacy["icon"] == {}
    catalog._join_installed([legacy], {skill.name: {"skill_dir": str(skill.skill_dir), "locations": ["user_repo"]}},
                            {row["id"]: row})
    assert legacy["connects"] == ["Service"] and legacy["installed"]["state"] == "installed"
    snapshot["entries"][0].update(connects=["Service"], icon={"kind": "svg", "svg": "<svg><script/></svg>"})
    assert catalog.items_for(row, snapshot)[0]["icon"] == {}


def test_hub_pointer_optional_fields_do_not_change_schema(tmp_path):
    entry = {"schema_version": 2, "name": "connector", "digest": "a" * 64,
             "source": {"kind": "git", "repo": "https://github.com/example/connector", "commit": "b" * 40}}
    assert catalog._hub_entry(entry, "connector", tmp_path)["connects"] == []
    marked = catalog._hub_entry({**entry, "connects": ["Mail"], "icon": "mail"}, "connector", tmp_path)
    assert marked["connects"] == ["Mail"] and marked["icon"] == {"kind": "host", "name": "mail"}
    # A bare boolean names no service: the Hub index writes the list of services.
    assert catalog._hub_entry({**entry, "connects": True}, "connector", tmp_path)["connects"] == []
    assert catalog._hub_entry({**entry, "connects": "true", "icon": "https://example.test/icon.svg"},
                              "connector", tmp_path)["icon"] == {}


@pytest.mark.parametrize("facts,wakes,labels", [
    (["mention", {"id": "signal.words", "label": "Words you chose appear"}], ["mention", "signal.words"],
     {"signal.words": "Words you chose appear"}),
    ([{"id": "direct_message", "label": "Mail addressed to you"}], ["direct_message"],
     {"direct_message": "Mail addressed to you"}),
    ([], [], {}), (["any_message", "from_bot"], ["from_bot"], {}),
])
def test_declared_facts_and_signal_labels_project_unchanged(facts, wakes, labels):
    clean, errors = validate_contributions(manifest(
        [{**PLACE, "facts": facts, "wakes": wakes}], kind="extension", permissions=["presence"]))
    assert not errors
    row, = project_contributions("connector", clean, "reviewed")
    assert row["facts"] == facts and row["wakes"] == wakes
    assert row["labels"] == labels


@pytest.mark.parametrize("patch", [
    {"facts": ["mention"], "wakes": ["reply_to_me"]},
    {"facts": [], "wakes": ["mention"]}, {"wakes": ["signal.words"]},
    {"facts": ["signal.words"]}, {"facts": ["unknown"]}, {"facts": "mention"},
    {"facts": ["mention", "mention"]},
    *[{"facts": [{"id": value, "label": "Words"}]} for value in
      ("unknown", "signal.", "signal.Upper", "signal.a.b", "signal." + "a" * 41)],
    {"facts": [{"id": "signal.words", "label": ""}]},
    {"facts": [{"id": "mention", "label": "x" * 121}]},
    {"facts": [{"id": "signal.words", "label": "Words", "pattern": "text"}]},
])
def test_invalid_facts_drop_only_invalid_contribution(patch):
    good = {**PLACE, "wakes": ["mention"]}  # no facts keeps existing core vocabulary
    clean, errors = validate_contributions(manifest(
        [good, {**PLACE, "id": "invalid", **patch}], kind="extension", permissions=["presence"]))
    assert len(errors) == 1 and [row["id"] for row in clean] == ["place"]
