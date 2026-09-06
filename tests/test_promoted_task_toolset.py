"""F6 (2026-08-10 amendments): the promote/router turn sees the LIVE toolset.

The router turn authors objectives/contracts for a task it will never run. The
The projection asks the real registry so credential-gated omissions remain
truthful. Workspace focus changes the default target, not this ordinary
top-level catalog.
"""

import json

from types import SimpleNamespace

import pytest


def _env(tmp_path):
    return SimpleNamespace(repo_dir=str(tmp_path / "repo"), drive_root=tmp_path)


def _toolset(tmp_path):
    from ouroboros.context import build_runtime_section

    task = {"id": "t1", "_ephemeral_turn": True, "metadata": {"force_plan": True}}
    section = build_runtime_section(_env(tmp_path), task)
    payload = json.loads(section.split("\n\n", 1)[1])
    assert "promoted_task_toolset" in payload, "the swarm-router turn must carry F6"
    return payload["promoted_task_toolset"]


@pytest.fixture()
def _github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")


def test_credential_gated_tool_is_not_advertised_when_unavailable(tmp_path, monkeypatch):
    # web_search is credential-gated behind live backends; with none available
    # the router must not be able to demand it — it moves to the TYPED omission
    # list instead of silently disappearing.
    import ouroboros.tools.search as search

    monkeypatch.setattr(search, "_available_web_search_backends", lambda: [])
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr("ouroboros.tools.github.github_cli_configured", lambda: False)
    toolset = _toolset(tmp_path)
    assert "web_search" not in toolset["top_level_tools"]
    assert "missing_credential" in toolset["unavailable_builtin_tools"]["web_search"]
    # No GitHub credentials from env, settings or CLI: the same typed omission.
    assert "get_github_issue" not in toolset["top_level_tools"]
    assert "missing_credential" in toolset["unavailable_builtin_tools"]["get_github_issue"]


def test_live_toolset_contains_the_complete_ordinary_top_level_catalog(tmp_path, monkeypatch, _github_token):
    import ouroboros.tools.search as search

    monkeypatch.setattr(search, "_available_web_search_backends", lambda: ["ddgs"])
    toolset = _toolset(tmp_path)
    names = set(toolset["top_level_tools"])
    assert "read_file" in names
    assert "delegate_start" in names
    assert "get_github_issue" in names
    assert "commit_reviewed" in names
    # With its credential present the gated tool is advertised normally.
    assert "web_search" in names
    assert "web_search" not in toolset.get("unavailable_builtin_tools", {})
    assert set(toolset) <= {"top_level_tools", "unavailable_builtin_tools", "rule"}


def test_non_router_turns_do_not_pay_for_the_projection(tmp_path):
    from ouroboros.context import build_runtime_section

    section = build_runtime_section(_env(tmp_path), {"id": "t1", "type": "task"})
    assert "promoted_task_toolset" not in section
