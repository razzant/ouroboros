"""One settings write leaves one record of what it changed: before → after, with the origin its caller declared.

The settings writer returns the change it committed, read under its own lock; the generic save and the Safety
owner endpoints (here) and a model-wait card's saved choice (``test_model_wait_controls``) record that receipt as
one ``settings_changed`` event. A save the app did not declare also puts
one quiet System row in Main. Secrets and structured values keep only their names; a failed record never
unsaves the settings.
"""

from __future__ import annotations

import json
import os
import re
import threading

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests._safety_fixtures import settings_install as install_fixture

install = install_fixture
APP = {"X-Ouroboros-Surface": "app", "X-Ouroboros-Actor": "owner"}
ALLOWANCE = "OUROBOROS_CONSCIOUSNESS_DAILY_USD"


@pytest.fixture
def client(install, monkeypatch):
    """Both HTTP writers over a real settings file; Main rows are captured, never delivered."""
    import supervisor.message_bus as bus
    import supervisor.state as state
    from ouroboros import config
    from ouroboros.gateway import safety_settings, settings

    rows = []
    monkeypatch.setattr(bus, "send_with_budget", lambda *args, **kwargs: rows.append((args, kwargs)))
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 77})
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    monkeypatch.setattr(settings, "_apply_settings_save_side_effects", lambda *a, **kw: None)
    monkeypatch.setattr(settings, "_start_supervisor_if_needed_for_request", lambda *a, **kw: False)
    install.write_text(json.dumps({"OPENROUTER_API_KEY": "fixture-router", ALLOWANCE: 20,
                                   "OUROBOROS_MODEL": "openrouter/old-main"}), encoding="utf-8")
    app = Starlette(routes=[Route("/api/settings", settings.api_settings_post, methods=["POST"]),
                            Route("/api/owner/safety-model", safety_settings.api_owner_safety_model, methods=["POST"])])
    app.state.drive_root = app.state.repo_dir = install.parent
    with TestClient(app) as test_client:
        yield test_client, install, rows


def records(install):
    path = install.parent / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("action") == "settings_changed"]


def save(test_client, entry, value, headers=None):
    if entry == "safety_model":
        response = test_client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_MODEL": value}}, headers=headers)
    else:
        response = test_client.post("/api/settings", json={ALLOWANCE: value}, headers=headers)
    assert response.status_code == 200, response.text
    return response


@pytest.mark.parametrize("entry, value, key, old, new, shown", [
    ("safety_model", "openrouter/new-main", "OUROBOROS_MODEL", "openrouter/old-main", "openrouter/new-main",
     ("openrouter/old-main", "openrouter/new-main")),
    ("generic", 40, ALLOWANCE, 20.0, 40.0, ("20", "40")),
])
def test_both_writers_record_old_and_new_and_an_undeclared_save_tells_main_once(client, entry, value, key, old, new, shown):
    from ouroboros.gateway.owner_settings import settings_change_text

    test_client, install, rows = client
    save(test_client, entry, value)
    [record] = records(install)
    assert record["changes"] == {key: {"old": old, "new": new}}
    assert (record["surface"], record["actor"]) == ("api", "not declared") and "actor_task_id" not in record
    [((chat, text), kwargs)] = rows
    assert (chat, kwargs) == (77, {"role": "system", "system_type": "settings_changed"})
    assert text == settings_change_text(record["changes"], record) and "\n" not in text  # one line of the record
    assert all(value in text for value in shown), text  # what changed, before and after


@pytest.mark.parametrize("entry, value", [("safety_model", "openrouter/new-main"), ("generic", 40)])
def test_the_apps_own_save_is_recorded_without_a_main_row(client, entry, value):
    test_client, install, rows = client
    save(test_client, entry, value, headers=APP)
    [record] = records(install)
    assert (record["surface"], record["actor"]) == ("app", "owner") and not rows


_SCRIPT = {"X-Ouroboros-Surface": "Script", "X-Ouroboros-Actor": "ouroboros", "X-Ouroboros-Task-Id": "task-19"}


def _words(text):
    return set(re.findall(r"[A-Za-z][\w-]*", text))


def test_the_row_says_who_changed_it_and_the_record_keeps_where_it_came_from(client):
    """Main's row names who changed the setting in words (Ouroboros, you, or neither when nobody declared);
    the surface, the declaration and the task are the record's facts and never reach the row (DESIGN principle 7)."""
    from ouroboros.gateway.owner_settings import settings_change_text

    test_client, install, rows = client
    for value, headers in ((40, _SCRIPT), (50, {"X-Ouroboros-Actor": "owner"}), (60, {})):
        save(test_client, "generic", value, headers=headers)
    recorded = records(install)
    assert [(record["surface"], record["actor"], record.get("actor_task_id")) for record in recorded] == [
        ("script", "ouroboros", "task-19"), ("api", "owner", None), ("api", "not declared", None)]
    texts = [text for ((_chat, text), _kwargs) in rows]
    assert texts == [settings_change_text(record["changes"], record) for record in recorded]
    by_ouroboros, by_owner, undeclared = map(_words, texts)
    assert "Ouroboros" in by_ouroboros and "you" not in by_ouroboros
    assert "you" in by_owner and "Ouroboros" not in by_owner
    assert not {"Ouroboros", "you"} & undeclared
    for text in texts:
        assert not {"script", "Script", "api", "declared", "ouroboros", "owner"} & _words(text), text
        assert "task-19" not in text and "\n" not in text
    # One change reads alike whoever made it: the words for who changed it are all that differ.
    said = [settings_change_text({ALLOWANCE: {"old": 20.0, "new": 40.0}}, {"surface": "api", "actor": actor})
            for actor in ("ouroboros", "owner", "not declared")]
    shared = os.path.commonprefix(said)
    assert len(set(said)) == 3 and "20" in shared and "40" in shared


@pytest.mark.parametrize("entry, value", [("safety_model", "openrouter/old-main"), ("generic", 20)])
def test_a_save_that_changes_nothing_records_nothing(client, entry, value):
    test_client, install, rows = client
    save(test_client, entry, value)
    assert not records(install) and not rows


def test_secrets_and_structures_keep_only_their_names(client):
    test_client, install, rows = client
    secret, custom = "sk-or-v1-" + "s" * 40, "owner-private-token"
    servers = [{"id": "demo", "name": "Demo", "enabled": True, "transport": "streamable_http", "allowed_tools": [],
                "url": "https://e.example/mcp", "auth_header": "Authorization", "auth_token": "Bearer mcp-private-token"}]
    response = test_client.post("/api/settings", json={"OPENROUTER_API_KEY": secret, "SERVICE_TOKEN": custom,
                                                       "MCP_SERVERS": servers, ALLOWANCE: 30})
    assert response.status_code == 200, response.text
    [record] = records(install)
    assert record["changes"]["OPENROUTER_API_KEY"] == record["changes"]["SERVICE_TOKEN"] == {"withheld": "secret"}
    assert record["changes"]["MCP_SERVERS"] == {"withheld": "structure"}
    assert record["changes"][ALLOWANCE] == {"old": 20.0, "new": 30.0}
    [((_chat, text), _kwargs)] = rows
    assert all(key in text for key in ("OPENROUTER_API_KEY", "SERVICE_TOKEN", "MCP_SERVERS", "30")), text
    written = (install.parent / "logs" / "events.jsonl").read_text(encoding="utf-8") + text
    assert all(value not in written for value in (secret, custom, "mcp-private-token"))


def test_a_text_of_several_lines_is_named_in_main_and_kept_in_the_record():
    from ouroboros.gateway.owner_settings import settings_change_text

    rules = {"old": "Ask before paying.", "new": "Ask before paying.\nNever post publicly."}
    text = settings_change_text({"OUROBOROS_SAFETY_OWNER_RULES": rules, ALLOWANCE: {"new": 5.5}},
                                {"surface": "api", "actor": "not declared"})
    assert "\n" not in text and "OUROBOROS_SAFETY_OWNER_RULES" in text and "5.5" in text
    assert "Ask before paying" not in text and "Never post publicly" not in text


def test_two_saves_in_a_row_record_the_exact_steps(client):
    test_client, install, _rows = client
    save(test_client, "generic", 40)
    save(test_client, "generic", 60)
    assert [record["changes"][ALLOWANCE] for record in records(install)] == [
        {"old": 20.0, "new": 40.0}, {"old": 40.0, "new": 60.0}]


def test_concurrent_writers_each_receive_the_step_they_committed(install):
    """The receipt is taken under the writer's lock: two racing writers chain, never share an old value."""
    from ouroboros.gateway.owner_settings import _owner_update_settings

    install.write_text(json.dumps({ALLOWANCE: 20}), encoding="utf-8")
    receipts, start = [], threading.Barrier(2)

    def write(value):
        start.wait()
        receipts.append(_owner_update_settings(lambda current: {**current, ALLOWANCE: value}, authored_keys=(ALLOWANCE,)))

    threads = [threading.Thread(target=write, args=(value,)) for value in (40, 60)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first, second = (receipt[ALLOWANCE] for receipt in receipts)
    assert first["old"] == 20.0 and second["old"] == first["new"] and {first["new"], second["new"]} == {40.0, 60.0}


def test_a_failed_record_or_row_never_unsaves_the_settings(client, monkeypatch):
    import supervisor.message_bus as bus
    from ouroboros.gateway import owner_settings

    test_client, install, _rows = client
    monkeypatch.setattr(owner_settings, "append_jsonl", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(bus, "send_with_budget", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat")))
    assert save(test_client, "generic", 40).json()["status"] == "saved"
    monkeypatch.setattr(owner_settings, "_withheld", lambda *a: (_ for _ in ()).throw(ValueError("projection")))
    assert test_client.post("/api/owner/safety-model", json={"settings": {"OUROBOROS_MODEL": "openrouter/x"}}).json() == {"ok": True, "receipt": "unavailable"}
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert (stored[ALLOWANCE], stored["OUROBOROS_MODEL"]) == (40.0, "openrouter/x")


@pytest.mark.parametrize("headers, expected", [
    ({}, {"surface": "api", "actor": "not declared"}),
    ({"X-Ouroboros-Surface": "app", "X-Ouroboros-Actor": "OWNER"}, {"surface": "app", "actor": "owner"}),
    ({"X-Ouroboros-Actor": "ouroboros", "X-Ouroboros-Task-Id": "abc123"},
     {"surface": "api", "actor": "ouroboros", "actor_task_id": "abc123"}),
])
def test_a_declaration_is_read_never_inferred(headers, expected):
    from types import SimpleNamespace

    from ouroboros.gateway.owner_settings import declared_settings_origin

    request = SimpleNamespace(headers=headers, client=SimpleNamespace(host="127.0.0.1"))
    assert declared_settings_origin(request) == expected


def test_the_mind_is_told_how_its_own_settings_write_declares_itself(tmp_path, monkeypatch):
    """A blocked file write to settings.json sends the mind to /api/settings and names the two headers that make
    its change read as Ouroboros's, with its task (CLIENT_CONTRACT "Declared settings origin"); the file is kept."""
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    drive, repo = tmp_path / "drive", tmp_path / "repo"
    drive.mkdir()
    repo.mkdir()
    (drive / "settings.json").write_text('{"TOTAL_BUDGET": 10}\n', encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(cfg, "DATA_DIR", drive)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", drive / "settings.json")
    result = ToolRegistry(repo_dir=repo, drive_root=drive).execute(
        "write_file", {"root": "runtime_data", "path": "settings.json", "content": "{}\n"})
    assert "DATA_WRITE_BLOCKED" in result and "/api/settings" in result, result
    assert "X-Ouroboros-Actor: ouroboros" in result and "X-Ouroboros-Task-Id: <your task id>" in result, result
    assert (drive / "settings.json").read_text(encoding="utf-8") == '{"TOTAL_BUDGET": 10}\n'
