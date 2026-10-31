"""The room's Subagents choice is saved independently of catalog availability."""
import json
import os

import pytest

from ouroboros import config, projects_registry as projects
from ouroboros.composer_cognition import read_cognition
from ouroboros.settings_integrity import effective_subagents_mode, live_subagents_mode
from ouroboros.subagent_runtime import subagents_availability
from tests.test_project_cognition import room_settings as cognition_room_settings

room_settings = cognition_room_settings


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
def test_main_and_project_mode_roundtrip_and_new_project_copy(room_settings, mode):
    root, client = room_settings
    projects.create_project(root, "existing")
    before = read_cognition(root)
    response = client.post("/api/owner/subagents-mode", json={"mode": mode})
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "project_id": "", "subagents_mode": mode,
                               "cognition": {**before, "subagents_mode": mode}}
    assert live_subagents_mode() == mode
    assert read_cognition(root, "existing")["subagents_mode"] == "auto"
    assert projects.create_project(root, "copied")["cognition"]["subagents_mode"] == mode
    stored, env = config.SETTINGS_PATH.read_bytes(), dict(os.environ)
    response = client.post("/api/owner/subagents-mode", json={"project_id": "existing", "mode": mode})
    assert response.status_code == 200 and response.json()["cognition"]["subagents_mode"] == mode
    assert config.SETTINGS_PATH.read_bytes() == stored and dict(os.environ) == env
    assert read_cognition(root, "existing") == {**before, "subagents_mode": mode}


@pytest.mark.parametrize("legacy_cognition", [False, True])
def test_legacy_room_missing_mode_reads_auto_even_when_main_is_swarm(room_settings, legacy_cognition):
    root, client = room_settings
    projects.create_project(root, "legacy")
    path = root / "state/projects.json"
    rows = json.loads(path.read_text())
    if legacy_cognition:
        rows["projects"][0]["cognition"].pop("subagents_mode")
    else:
        rows["projects"][0].pop("cognition")
    path.write_text(json.dumps(rows))
    assert client.post("/api/owner/subagents-mode", json={"mode": "swarm"}).status_code == 200
    assert read_cognition(root)["subagents_mode"] == "swarm"
    assert read_cognition(root, "legacy")["subagents_mode"] == "auto"
    assert projects.create_project(root, "new")["cognition"]["subagents_mode"] == "swarm"


@pytest.mark.parametrize("body,status,code", [
    ({"mode": "invalid"}, 400, "subagents_mode_invalid"),
    ({"mode": None}, 400, "subagents_mode_invalid"),
    ([], 400, "subagents_mode_invalid"),
    ({"mode": "auto", "project_id": "../bad"}, 400, "subagents_mode_invalid"),
    ({"mode": "auto", "project_id": "missing"}, 404, "project_unavailable"),
])
def test_invalid_mode_or_room_is_unsaved(room_settings, body, status, code):
    _root, client = room_settings
    before = config.SETTINGS_PATH.read_bytes()
    response = client.post("/api/owner/subagents-mode", json=body)
    assert response.status_code == status
    assert response.json()["code"] == code and response.json()["saved"] is False
    assert config.SETTINGS_PATH.read_bytes() == before


@pytest.mark.parametrize("enabled,availability", [(True, "available"), (False, "disabled"), (None, "unsaved")])
def test_settings_availability_controls_effective_mode_without_changing_saved_choice(room_settings, monkeypatch, enabled, availability):
    _root, client = room_settings
    assert client.post("/api/owner/subagents-mode", json={"mode": "swarm"}).status_code == 200
    values = {"OUROBOROS_SUBAGENTS_MODE": "swarm", "OUROBOROS_MODEL": "openai::probe", "OPENAI_API_KEY": "test-key"}
    if enabled is None:
        values = {"OUROBOROS_SUBAGENTS_MODE": "swarm", "OUROBOROS_MODEL_HEAVY": "owner/unsaved-candidate"}
    else:
        values["OUROBOROS_SUBAGENTS"] = json.dumps({"enabled": enabled, "items": []})
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    assert subagents_availability(config.load_settings()) == availability
    assert live_subagents_mode() == "swarm"
    assert effective_subagents_mode() == ("swarm" if enabled else "off")


def test_mode_writer_respects_settings_integrity_pin(room_settings, monkeypatch):
    import hashlib

    _root, client = room_settings
    before = config.SETTINGS_PATH.read_bytes()
    monkeypatch.setenv("OUROBOROS_SETTINGS_SHA256", hashlib.sha256(before).hexdigest())
    from ouroboros.settings_integrity import SettingsIntegrityError

    with pytest.raises(SettingsIntegrityError, match="immutable"):
        client.post("/api/owner/subagents-mode", json={"mode": "off"})
    assert config.SETTINGS_PATH.read_bytes() == before
