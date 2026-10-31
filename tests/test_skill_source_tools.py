"""Ouroboros's own skill tools: search and install by catalog id, review rows review, owner-only changes."""

from __future__ import annotations

import json

import pytest

from ouroboros.skill_install_record import read_install_record
from ouroboros.tools import skill_source_tools as tools
from ouroboros.tools.registry import ToolContext
from tests._skill_source_fixtures import HubFixture, builtin_repo, join_source_refreshes


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    import ouroboros.config as cfg

    drive = tmp_path / "data"
    (drive / "skills").mkdir(parents=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    hub = HubFixture(tmp_path / "remote")
    hub.add_hosted("weather", official=True, recommended=True)
    hub.add_hosted("notes", category=["writing"])
    hub.commit()
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(hub.rows()))
    monkeypatch.setattr(cfg, "REPO_DIR", builtin_repo(tmp_path))
    reviews = []
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", lambda d, name, **k: reviews.append(name))
    context = ToolContext(repo_dir=tmp_path, drive_root=drive)
    context.task_id = "task-1"
    yield context, drive, reviews, hub
    join_source_refreshes()


def _call(tool, ctx, **kwargs):
    return _typed_call(tool, ctx, **kwargs).text


def _typed_call(tool, ctx, **kwargs):
    from ouroboros.tools.tool_result import (
        ToolResult, _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )

    entry = next(e for e in tools.get_tools() if e.name == tool)
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = entry.handler(ctx, **kwargs)
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert isinstance(published, ToolResult), f"{tool} did not publish a typed result"
    assert published.text == text
    return published


@pytest.mark.parametrize("tool, arguments, code, status", [
    ("skill_sources", {"action": "unknown"}, "TOOL_ARG_ERROR", "argument_error"),
    ("skill_sources", {"action": "remove", "source_id": "builtin"}, "TOOL_ERROR", "error"),
    ("skill_install", {"id": "https://example.test/skill", "reason": "r"}, "TOOL_ARG_ERROR", "argument_error"),
    ("skill_install", {"id": "builtin/skill", "reason": " "}, "TOOL_ARG_ERROR", "argument_error"),
    ("skill_install", {"id": "builtin/skill", "reason": "r", "replace": True}, "TOOL_ERROR", "error"),
    ("skill_update", {"name": "skill", "mode": "later"}, "TOOL_ARG_ERROR", "argument_error"),
    ("skill_update", {"name": "", "mode": "replace"}, "TOOL_ARG_ERROR", "argument_error"),
    ("skill_update", {"name": "absent", "mode": "replace"}, "TOOL_ERROR", "error"),
])
def test_refusals_publish_argument_or_host_errors(tmp_path, monkeypatch, tool, arguments, code, status):
    from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata

    monkeypatch.setattr(tools, "_cyber", lambda: False)
    context = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    result = _typed_call(tool, context, **arguments)
    assert result.status == "error" and result.code == code
    assert _typed_execution_failure(True, result)
    assert _typed_result_metadata(tool, result.text, True, result)["status"] == status


@pytest.mark.parametrize("tool", ["skill_install", "skill_update"])
@pytest.mark.parametrize("http_status", [400, 502])
@pytest.mark.parametrize("error_code, expected", [("bad_check", "TOOL_ARG_ERROR"),
                                                  ("registry_host_failed", "TOOL_ERROR")])
def test_installer_error_classification_uses_code_not_http_status(tmp_path, monkeypatch, tool, http_status,
                                                                error_code, expected):
    from ouroboros import skill_source_install as installer

    def fail(*args, **kwargs):
        raise installer.SkillInstallError(error_code, "injected producer failure", http_status)

    monkeypatch.setattr(installer, "install_item" if tool == "skill_install" else "update_installed", fail)
    arguments = {"id": "clawhub/demo", "reason": "r"} if tool == "skill_install" else {
        "name": "demo", "mode": "replace"}
    result = _typed_call(tool, ToolContext(repo_dir=tmp_path, drive_root=tmp_path), **arguments)
    assert (result.status, result.code) == ("error", expected)
    assert f'"code": "{error_code}"' in result.text


@pytest.mark.parametrize("failure", ["probe", "source"])
def test_source_host_failures_publish_tool_error(ctx, monkeypatch, failure):
    context, _drive, _reviews, _hub = ctx
    monkeypatch.setattr(tools, "_cyber", lambda: True)
    if failure == "probe":
        monkeypatch.setattr("ouroboros.skill_source_catalog.probe_source",
                            lambda *a, **kw: {"error": "Repository is unavailable."})
        arguments = {"action": "add", "url": "https://example.test/skills.git"}
    else:
        arguments = {"action": "remove", "source_id": "no-longer-connected"}
    result = _typed_call("skill_sources", context, **arguments)
    assert (result.status, result.code) == ("error", "TOOL_ERROR")


@pytest.mark.parametrize("tool", ["skill_sources", "skill_search", "skill_install", "skill_update"])
def test_successful_calls_publish_ok(ctx, tool):
    context, _drive, _reviews, _hub = ctx
    _call("skill_search", context)
    if tool == "skill_update":
        _call("skill_install", context, id="ouroboros/weather", reason="forecast")
    arguments = {"skill_sources": {}, "skill_search": {"query": "weather"},
                 "skill_install": {"id": "ouroboros/weather", "reason": "forecast"},
                 "skill_update": {"name": "weather", "mode": "keep"}}[tool]
    result = _typed_call(tool, context, **arguments)
    assert (result.status, result.code) == ("ok", "OK")
    assert isinstance(json.loads(result.text), dict)


@pytest.mark.parametrize("tool", ["skill_install", "skill_update"])
def test_invalid_self_check_is_an_argument_error(ctx, tool):
    context, drive, _reviews, _hub = ctx
    _call("skill_search", context)
    arguments = {"id": "ouroboros-community/notes", "reason": "notes"}
    if tool == "skill_update":
        _call("skill_install", context, **arguments)
        arguments = {"name": "notes", "mode": "replace"}
    result = _typed_call(tool, context, **arguments, check="self", check_notes="")
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert '"code": "bad_check"' in result.text
    if tool == "skill_install":
        assert read_install_record(drive, "notes") is None


def test_the_four_tools_are_registered_with_schemas():
    names = {entry.name: entry for entry in tools.get_tools()}
    assert set(names) == {"skill_sources", "skill_search", "skill_install", "skill_update"}
    assert names["skill_install"].schema["parameters"]["required"] == ["id", "reason"]
    assert {name: entry.safety_policy for name, entry in names.items()} == {
        "skill_sources": "check", "skill_search": "skip", "skill_install": "check", "skill_update": "check",
    }


def test_search_reads_a_new_source_once_and_returns_catalog_ids(ctx):
    context, _drive, _reviews, _hub = ctx
    result = json.loads(_call("skill_search", context, query="weather"))
    assert [item["id"] for item in result["items"]] == ["ouroboros/weather"]
    assert result["items"][0]["trust"] == "trusted" and result["items"][0]["official"] is True
    everything = json.loads(_call("skill_search", context, category="writing"))
    assert [item["id"] for item in everything["items"]] == ["ouroboros-community/notes"]


def test_install_from_a_review_row_starts_the_review_and_records_the_reason(ctx):
    context, drive, reviews, _hub = ctx
    _call("skill_search", context)
    out = json.loads(_call("skill_install", context, id="ouroboros-community/notes", reason="the owner's notes task"))
    assert out["needs_review"] and reviews == ["notes"] and "reviewed before it runs" in out["message"]
    record = read_install_record(drive, "notes")
    assert record["installed_by"] == "ouroboros" and record["reason"] == "the owner's notes task"
    assert record["task_id"] == "task-1" and record["basis"] == "review"


@pytest.mark.parametrize("tool", ["skill_install", "skill_update"])
@pytest.mark.parametrize("enforcement,expected", [("blocking", ["a", "b"]), ("advisory", ["a"])])
def test_skill_tools_carry_the_admitted_panel_before_mutation(tmp_path, monkeypatch, tool, enforcement, expected):
    from ouroboros import skill_source_install as installer
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
    set_review_pool(monkeypatch, pool_roster(pool_seat("a", "openai/a"), pool_seat("b", "openai/b")))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    called, panels = [], []
    def install(*_args, **kwargs):
        called.append(kwargs)
        return {"name": "notes", "needs_review": True, "source": {"name": "community"}}
    monkeypatch.setattr(installer, "install_item" if tool == "skill_install" else "update_installed", install)
    monkeypatch.setattr(installer, "start_review", lambda *_a, **kw: panels.append(kw["panel"]))
    args = ({"id": "community/notes", "reason": "notes for the trip"} if tool == "skill_install"
            else {"name": "notes", "mode": "replace"})
    why = "review_reason" if tool == "skill_install" else "reason"  # the install reason never explains reviewers
    context = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    invalid = _typed_call(tool, context, **args, reviewers=["a"], **{why: 42})
    assert invalid.code == "TOOL_ARG_ERROR" and f"{why} must be a string" in invalid.text
    assert called == panels == []
    if enforcement == "advisory":
        refused = _typed_call(tool, context, **args, reviewers=["a"])
        assert refused.code == "TOOL_ARG_ERROR" and "REVIEW_PANEL_REASON_REQUIRED" in refused.text
        assert f"Supply {why} or omit reviewers" in refused.text
        assert called == panels == []
    result = _typed_call(tool, context, **args, reviewers=["a"], **{why: "one area changed"})
    assert result.code == "OK" and len(called) == len(panels) == 1
    assert panels[0].facts["assigned"] == expected and panels[0].facts["reason"] == "one area changed"
    assert panels[0].facts["chosen_by"] == "ouroboros"
    assert called[0].get("reason", "") == ("notes for the trip" if tool == "skill_install" else "")


def test_install_from_a_trusted_row_is_accepted_at_once(ctx):
    context, drive, reviews, _hub = ctx
    _call("skill_search", context)
    out = json.loads(_call("skill_install", context, id="ouroboros/weather", reason="forecast for a trip"))
    assert out["basis"] == "trusted" and "acceptance" in out and reviews == []
    assert "installs without review" in out["message"] and "Keys granted:" in out["message"]
    assert read_install_record(drive, "weather")["basis"] == "trusted"


def test_install_refuses_urls_missing_reasons_and_unconfirmed_replacement(ctx):
    context, drive, _reviews, _hub = ctx
    assert "never a URL" in _call("skill_install", context, id="https://github.com/x/y", reason="r")
    assert "reason is required" in _call("skill_install", context, id="ouroboros/weather", reason=" ")
    refused = _call("skill_install", context, id="ouroboros/weather", reason="r", replace=True)
    assert refused.startswith("⚠️ SKILL_INSTALL_REFUSED") and "owner's instruction" in refused
    assert read_install_record(drive, "weather") is None


def test_changing_sources_needs_the_owner_outside_cyber_pro(ctx, tmp_path, monkeypatch):
    context, _drive, _reviews, _hub = ctx
    listed = json.loads(_call("skill_sources", context))
    assert [row["id"] for row in listed["sources"]] == ["builtin", "ouroboros", "ouroboros-community"]
    folder = tmp_path / "folder"
    folder.mkdir()
    refused = _call("skill_sources", context, action="add", path=str(folder), name="Mine")
    assert refused.startswith("⚠️ SKILL_SOURCES_REFUSED")
    monkeypatch.setattr(tools, "_cyber", lambda: True)
    added = json.loads(_call("skill_sources", context, action="add", path=str(folder), name="Mine", trust="review"))
    assert added["sources"][-1]["id"] == "mine" and added["sources"][-1]["trust"] == "review"


def test_update_keep_records_the_skipped_release(ctx):
    context, drive, _reviews, _hub = ctx
    _call("skill_search", context)
    _call("skill_install", context, id="ouroboros-community/notes", reason="r")
    assert "SKILL_UPDATE_ERROR" in _call("skill_update", context, name="notes", mode="later")
    hub_digest = json.loads(_call("skill_search", context, query="notes"))["items"][0]
    out = json.loads(_call("skill_update", context, name="notes", mode="keep"))
    assert out["mode"] == "keep" and read_install_record(drive, "notes")["skipped_release_digest"]
    assert hub_digest["installed"]["state"] == "installed"


def test_a_merge_ouroboros_asks_for_is_its_own_task_never_the_owner_door(ctx, monkeypatch):
    from ouroboros.skill_source_catalog import refresh_source

    context, drive, _reviews, hub = ctx
    _call("skill_search", context)
    _call("skill_install", context, id="ouroboros-community/notes", reason="r")
    manifest = drive / "skills" / "ouroboroshub" / "notes" / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")
    hub.add_hosted("notes", "2.0.0", category=["writing"])
    hub.commit()
    refresh_source(drive, hub.rows()[2])

    def owner_door():
        raise AssertionError("Ouroboros never speaks through the owner's door")

    monkeypatch.setattr("supervisor.message_bus.get_bridge", owner_door)
    out = json.loads(_call("skill_update", context, name="notes", mode="merge"))
    assert out["task"] == "current" and out["base_available"] is True
    assert out["local_tree"] == "skills/ouroboroshub/notes" and "skill_review" in out["instructions"]
    assert read_install_record(drive, "notes")["merge"]["requested_by"] == "ouroboros"


def test_search_serves_release_capabilities_and_current_local_digest(ctx):
    from ouroboros.skill_loader import find_skill

    context, drive, _reviews, _hub = ctx
    first = json.loads(_call("skill_search", context, query="notes"))["items"][0]
    assert first["digest"] and first["type"] == "instruction"
    assert first["needs_keys"] == [] and first["permissions"] == []
    assert "local_digest" not in first
    _call("skill_install", context, id=first["id"], reason="r")
    loaded = find_skill(drive, "notes")
    (loaded.skill_dir / "edit.txt").write_text("local", encoding="utf-8")
    item = json.loads(_call("skill_search", context, query="notes"))["items"][0]
    assert item["digest"] == first["digest"]
    assert item["local_digest"] == find_skill(drive, "notes").content_hash != item["digest"]
    refused = _call("skill_update", context, name="notes", mode="replace", local_digest=first["digest"])
    assert "local_changed" in refused
    assert (loaded.skill_dir / "edit.txt").exists()


@pytest.fixture
def cyber_pro():
    import ouroboros.config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    cfg.initialize_runtime_mode_baseline("cyber_pro")
    yield
    cfg.reset_runtime_mode_baseline_for_tests()


@pytest.mark.parametrize("tool", ["skill_install", "skill_update"])
def test_self_check_outside_cyber_pro_is_refused_with_the_rule(ctx, tool):
    context, drive, reviews, _hub = ctx
    _call("skill_search", context)
    arguments = {"id": "ouroboros-community/notes", "reason": "notes"}
    if tool == "skill_update":
        _call("skill_install", context, **arguments)
        arguments = {"name": "notes", "mode": "replace"}
    result = _typed_call(tool, context, **arguments, check="self", check_notes="Read every file.")
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert "only in Cyber Pro" in result.text and "only the owner skips one" in result.text
    record = read_install_record(drive, "notes")
    assert record is None if tool == "skill_install" else record["basis"] == "review"


def test_tools_record_self_check_without_starting_panel(ctx, cyber_pro):
    context, drive, reviews, _hub = ctx
    _call("skill_search", context)
    installed = json.loads(_call("skill_install", context, id="ouroboros-community/notes", reason="r",
                                 check="self", check_notes="Read this release; it only describes note formatting."))
    assert installed["basis"] == "self_check" and not reviews
    updated = json.loads(_call("skill_update", context, name="notes", mode="replace",
                               check="self", check_notes="Read the exact offered release."))
    assert updated["basis"] == "self_check" and not reviews
    assert read_install_record(drive, "notes")["check"]["task_id"] == "task-1"
    schemas = {e.name: e.schema["parameters"]["properties"] for e in tools.get_tools()}
    assert schemas["skill_install"]["check"]["enum"] == schemas["skill_update"]["check"]["enum"] == ["review", "self"]
    assert "skip_review" not in schemas["skill_update"]
