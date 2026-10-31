"""The client contract is regenerated from stored wake facts, never handwritten rows."""
import json
from pathlib import Path
import time
from unittest.mock import patch

from ouroboros import skill_source_catalog as catalog
from scripts.generate_consciousness_fixture import generate


def _fixture_handles():
    """Cached store handles left under the generator's temporary folders: Windows cannot remove an open file."""
    from ouroboros import store

    with store._CONNECTION_GUARD:
        return [name for pool in list(store._CONNECTION_POOLS) for name in pool.entries if "wake-fixture-" in name]


def test_browser_fixture_matches_the_producers():
    expected = json.loads((Path(__file__).parents[1] / "web/tests/fixtures/consciousness_contract.json").read_text(encoding="utf-8"))
    if not hasattr(time, "tzset"):  # Windows cannot change the process zone; the generator pins the producer's own
        assert generate() == expected and not _fixture_handles()
        return
    try:
        for zone in ("UTC", "Europe/Moscow"):
            with patch.dict("os.environ", TZ=zone):
                time.tzset()
                assert generate() == expected
                assert not _fixture_handles(), _fixture_handles()
    finally:
        time.tzset()


def test_catalog_passes_optional_titles_without_renaming_ids():
    rows = catalog.items_for({"id": "hub", "name": "Hub", "kind": "hub", "trust": "review"},
                             {"entries": [{"name": "email", "title": "Email tools"}, {"name": "plain"}]})
    assert [(row["id"], row["name"], row["title"]) for row in rows] == [
        ("hub/email", "email", "Email tools"), ("hub/plain", "plain", "")]


def test_installed_manifest_supplies_title_and_connector_label(tmp_path):
    for title, label, expected in [("Email tools", "Mail", "Email tools"), ("", "Mailbox", "Mailbox"), ("", "", "")]:
        text = "---\nname: mail\ndescription: Read mail\ntype: extension\npermissions: [presence]\n"
        if title:
            text += f"title: {title}\n"
        if label:
            text += f"contributes:\n  - id: mail\n    provides: place\n    label: {label}\n    destination: vendor:Mail\n    icon: mail\n"
        (tmp_path / "SKILL.md").write_text(text + "---\nMail tools.\n", encoding="utf-8")
        assert catalog._connector_facts(catalog.read_manifest(tmp_path))["title"] == expected


def test_inventory_lists_translated_fixed_doors_and_disclosures_only():
    from scripts.ui_control_inventory import scan_controls

    rows, dynamic = scan_controls("""<summary class="ui-hit-area">${escapeHtml(tr('more', 'more'))}</summary>
        <button class="door">${escapeHtml(tr('main', 'Open in Main'))} ↗</button>
        <button>${fmt('{n} places', {n: count})}</button>""")
    assert rows == [("more", "ui-hit-area"), ("Open in Main ↗", "door")]
    assert dynamic == 1
