"""Bundled instruction contracts and reachable payload references."""

import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import pytest

from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

REPO = Path(__file__).resolve().parents[1]
PAYLOADS = {
    "ui_design": ("SKILL.md", "references/craft.md", "references/browser.md", "references/measure.js",
                  "references/freeze.js", "references/brief.md", "assets/compare.html"),
}

@pytest.mark.parametrize("name", PAYLOADS)
def test_design_manifest_and_visible_metadata(tmp_path, monkeypatch, name):
    from ouroboros.context import _build_installed_skills_section
    from ouroboros.skill_builtin import install_builtin_skill
    from ouroboros.skill_catalogue import session_skill_index
    from ouroboros.skill_loader import save_enabled

    root = REPO / "skills" / name
    text = (root / "SKILL.md").read_text(encoding="utf-8")
    manifest = parse_skill_manifest_text(text)
    assert manifest.name == name and manifest.version
    assert manifest.type == "instruction" and manifest.os == "any"
    assert manifest.permissions == [] and not manifest.entry and not manifest.scripts
    assert not re.search(r"^(entry|scripts)\s*:", text.split("---", 2)[1], re.MULTILINE)
    assert 0 < len(manifest.description) <= 260 and 0 < len(manifest.when_to_use) <= 260
    assert manifest.model_experience.get("what_model_sees") and manifest.model_experience.get("token_effect")
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    assert install_builtin_skill(name, drive_root=tmp_path, seed_dir=REPO / "skills")["ok"]
    save_enabled(tmp_path, name, True, actor="owner_ui")
    ctx = SimpleNamespace(repo_dir=REPO, drive_root=tmp_path)
    for projection in (_build_installed_skills_section(ctx), session_skill_index(ctx)):
        assert manifest.description in projection and manifest.when_to_use in projection


@pytest.mark.parametrize("name", PAYLOADS)
def test_referenced_design_files_are_in_the_bundled_payload(name):
    root = REPO / "skills" / name
    for relative in PAYLOADS[name]:
        assert (root / relative).is_file(), relative
    text = (root / "SKILL.md").read_text(encoding="utf-8")
    paths = set(re.findall(
        r"(?:skills/|(?:\.\./)*)(?:ui_design/)?(?:references|assets)/[\w./-]+", text))
    paths.update(re.findall(r"(?:\.\./)+ui_design/SKILL\.md", text))
    for relative in paths:
        target = (REPO / relative if relative.startswith("skills/") else root / relative).resolve()
        assert target.is_relative_to((REPO / "skills").resolve()) and target.is_file(), relative


def test_design_catalogue_rows_are_recommended_default_on_and_categorised(tmp_path):
    from ouroboros.skill_builtin import default_on_names
    from ouroboros.skill_source_catalog import builtin_snapshot

    seeds = tmp_path / "skills"
    for name in PAYLOADS:
        shutil.copytree(REPO / "skills" / name, seeds / name)
    shutil.copyfile(REPO / "skills/builtin.json", seeds / "builtin.json")
    rows = {row["name"]: row for row in json.loads((seeds / "builtin.json").read_text())["skills"]}
    catalog = {row["name"]: row for row in builtin_snapshot(tmp_path)["entries"]}
    for name in PAYLOADS:
        assert rows[name] == {"name": name, "category": ["design"], "recommended": True, "default_on": True}
        assert catalog[name]["category"] == ["design"] and catalog[name]["recommended"] is True
    assert set(default_on_names(seeds)) == set(PAYLOADS)
