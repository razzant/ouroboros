"""Real bundled speech payloads are ordinary built-ins (ouroboros/skill_builtin.py).

The built-in source installs them trusted and the owner's auto-grant setting grants
`page_module`; skills/builtin.json turns Templates and Device voice on by default, while
Browser speech stays off until the owner installs it from Browse (the dictation chooser
sends them there) because its audio may leave the device.
"""
import importlib.util
from pathlib import Path

import pytest

from ouroboros.extension_ui_validation import validate_contributions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def drive(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    data = tmp_path / "data"
    (data / "skills" / "native").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", data / "settings.json")
    return data


@pytest.mark.parametrize("name,kind", [("device_voice", "tts"), ("browser_speech", "stt")])
def test_actual_builtin_installs_trusted_and_page_module_follows_the_auto_grant_setting(drive, monkeypatch, name, kind):
    import ouroboros.config as config
    from ouroboros.skill_builtin import install_builtin_skill
    from ouroboros.skill_loader import find_skill, grant_status_for_skill

    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: False)
    result = install_builtin_skill(name, drive_root=drive, seed_dir=ROOT / "skills")
    assert result["ok"] and result["basis"] == "trusted"
    assert result["auto_granted_permissions"] == [], "the owner's setting is off: nothing is granted"
    assert grant_status_for_skill(drive, find_skill(drive, name))["all_granted"] is False

    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: True)
    result = install_builtin_skill(name, drive_root=drive, seed_dir=ROOT / "skills")
    assert "page_module" in result["auto_granted_permissions"]
    path = drive / "skills" / "native" / name
    skill = find_skill(drive, name)
    assert skill.manifest.plugin_api["version"] == "2.0"
    gate = skill.review.gate_for(skill.content_hash)
    assert gate["executable_review"] is True and gate["blocking_reason"] == "trusted_source"
    assert grant_status_for_skill(drive, skill)["all_granted"] is True
    entries, errors = validate_contributions(skill.manifest, skill_dir=path)
    assert not errors and len(entries) == 1
    assert entries[0]["provides"] == kind and entries[0]["runtime"] == "client"
    assert entries[0]["render"]["kind"] == "host"
    assert "page_module" in skill.manifest.permissions
    assert ("net" in skill.manifest.permissions) is (name == "browser_speech")
    if name == "browser_speech":
        # The chooser shows this before the engine runs: every browser's data path in the skill's own words.
        assert all(vendor in entries[0]["title"] for vendor in ("Free", "Google", "Microsoft", "Apple"))
    assert skill.manifest.model_experience and skill.manifest.when_to_use
    spec = importlib.util.spec_from_file_location(f"test_{name}_plugin", path / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    assert plugin.register(object()) is None


def test_templates_and_device_voice_are_on_by_default_and_browser_speech_is_not():
    from ouroboros.skill_builtin import default_on_names

    names = default_on_names(ROOT / "skills")
    assert {"templates", "device_voice"} <= set(names)
    assert "browser_speech" not in names, "audio may leave the device: the owner installs it"


def test_an_uninstalled_builtin_engine_is_offered_and_an_installed_one_is_not_offered_twice(drive, monkeypatch):
    from ouroboros.gateway.contributions import _uninstalled_builtin_offers

    monkeypatch.setattr("ouroboros.skill_builtin.bundled_skills_dir", lambda: ROOT / "skills")
    offers = _uninstalled_builtin_offers(set())
    assert [(offer["skill"], offer["installed"]) for offer in offers] == [("browser_speech", False)], \
        "only dictation engines are offered; Device voice speaks, it does not dictate"
    assert offers[0]["destination"].startswith("vendor:")
    assert _uninstalled_builtin_offers({"browser_speech"}) == [], "an installed copy is offered by the catalog itself"
