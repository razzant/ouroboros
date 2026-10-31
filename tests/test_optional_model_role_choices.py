"""A saved Vision or Consciousness value, blank included, is the owner's choice; an absent one is not.

Blank inherits Main and beats an operator's variable through GET, an unrelated Settings save, an
incidental owner write and the task start; an absent role keeps following that variable, and a save
that posts it back as read does not author it. Fixtures are the defaults-provenance suite's.
"""

from __future__ import annotations

import json

import pytest

from tests.test_defaults_provenance import ANTHROPIC, SLOTS, _settings_client, _task_start, _write
from tests.test_defaults_provenance import install as provenance_install

install = provenance_install


@pytest.mark.parametrize("key, getter", [("OUROBOROS_MODEL_VISION", "get_vision_model"),
                                         ("OUROBOROS_MODEL_CONSCIOUSNESS", "get_consciousness_model")])
def test_a_saved_blank_vision_or_consciousness_keeps_inheriting_main_against_an_operator_variable(
        install, monkeypatch, key, getter):
    """Blank Vision/Consciousness is the owner's "use Main", kept by GET, unrelated saves and the
    task start; an operator's variable still sets the role while the document does not."""
    from ouroboros import model_slots
    from ouroboros.gateway.owner_settings import _owner_update_settings
    from ouroboros.settings_integrity import task_settings_scope

    monkeypatch.setenv(key, "anthropic::claude-operator-role")
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key", key: ""})
    client = _settings_client(monkeypatch)
    shown = client.get("/api/settings").json()
    assert shown[key] == ""
    saved = client.post("/api/settings", json={**{k: shown[k] for k in (*SLOTS, key)}, "TOTAL_BUDGET": "25"})
    assert saved.status_code == 200, saved.text
    _owner_update_settings(lambda current: {**current, "OUROBOROS_UI_LANGUAGE": "ru"})
    assert json.loads(install.read_text(encoding="utf-8"))[key] == ""
    snapshot = _task_start()
    assert snapshot.settings[key] == ""
    with task_settings_scope(snapshot):
        assert getattr(model_slots, getter)() == ANTHROPIC[0]
    monkeypatch.setenv(key, "anthropic::claude-operator-role")
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key", key: "anthropic::claude-saved-role"})
    assert _task_start().settings[key] == "anthropic::claude-saved-role", "a saved name beat it already"
    monkeypatch.setenv(key, "anthropic::claude-operator-role")
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    assert _task_start().settings[key] == "anthropic::claude-operator-role"


@pytest.mark.parametrize("key", ["OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS"])
def test_an_unrelated_save_leaves_an_absent_vision_or_consciousness_to_the_environment(install, monkeypatch, key):
    """Absent is not a saved blank: Settings and an incidental owner write posting the role back as
    read do not author it, so an operator's variable keeps choosing; an edit authors it, blank included."""
    from ouroboros.gateway.owner_settings import _owner_update_settings

    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    client = _settings_client(monkeypatch)
    for operator in ("", "anthropic::claude-operator-role"):
        if operator:
            monkeypatch.setenv(key, operator)
        shown = client.get("/api/settings").json()
        assert shown[key] == operator
        saved = client.post("/api/settings", json={**{k: shown[k] for k in (*SLOTS, key)}, "TOTAL_BUDGET": "25"})
        assert saved.status_code == 200, saved.text
        _owner_update_settings(lambda current: {**current, "OUROBOROS_UI_LANGUAGE": "ru"})
        assert key not in json.loads(install.read_text(encoding="utf-8"))
    monkeypatch.setenv(key, "anthropic::claude-later-role")
    assert _task_start().settings[key] == "anthropic::claude-later-role"
    shown = client.get("/api/settings").json()
    saved = client.post("/api/settings", json={**{k: shown[k] for k in SLOTS}, key: ""})
    assert saved.status_code == 200, saved.text
    assert json.loads(install.read_text(encoding="utf-8"))[key] == ""
    monkeypatch.setenv(key, "anthropic::claude-later-role")
    assert _task_start().settings[key] == ""
