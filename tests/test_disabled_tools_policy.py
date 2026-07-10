"""Pins the declarative `disabled_tools` tool-policy (TB2.1 methodology fix).

The benchmark adapter withholds the agent's own web/search/VLM tools while leaving
shell network egress (git clone/pip) intact. This is done via a `disabled_tools`
list on the task contract — NOT via `allowed_resources` — so it never trips the
web<->network cross-implication in the registry resource gate. These tests pin:
normalization, contract propagation, gateway (/api/tasks) pass-through, registry
hiding (schemas / core-only / get_schema_by_name / available_tools) + execute block,
and subagent inheritance via the parent-contract spread.
"""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.contracts.task_contract import (
    build_task_contract,
    normalize_disabled_tools,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry

WEB_TOOLS = ["web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query"]


def test_normalize_disabled_tools():
    assert normalize_disabled_tools(["a", " a ", "b", "", None]) == ["a", "b"]
    assert normalize_disabled_tools("web_search") == ["web_search"]
    assert normalize_disabled_tools(None) == []
    assert normalize_disabled_tools(42) == []
    assert normalize_disabled_tools(("x", "x", "y")) == ["x", "y"]


def test_build_task_contract_carries_disabled_tools():
    c = build_task_contract({"description": "x", "disabled_tools": WEB_TOOLS})
    assert c["disabled_tools"] == WEB_TOOLS
    # absent -> empty, and allowed_resources stays empty (no web<->network entanglement)
    c2 = build_task_contract({"description": "x"})
    assert c2["disabled_tools"] == []
    assert c2["allowed_resources"] == {}


def test_api_tasks_create_carries_disabled_tools(tmp_path, monkeypatch):
    """The /api/tasks gateway must thread top-level `disabled_tools` into the task
    dict, metadata, and the attached contract — otherwise the registry block never
    activates for API-created (benchmark) tasks."""
    from ouroboros.gateway.tasks import api_tasks_create

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)

    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    resp = TestClient(app).post("/api/tasks", json={"description": "x", "disabled_tools": WEB_TOOLS})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    assert captured and captured[0]["disabled_tools"] == WEB_TOOLS
    assert captured[0]["metadata"]["disabled_tools"] == WEB_TOOLS
    assert captured[0]["task_contract"]["disabled_tools"] == WEB_TOOLS
    # survives to the persisted task result contract too
    result = json.loads((data / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
    assert result["task_contract"]["disabled_tools"] == WEB_TOOLS


def test_api_tasks_create_carries_acceptance_claims(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import api_tasks_create

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    claim = {"id": "answer", "claim": "final answer is exact", "support": "exact receipt"}
    resp = TestClient(app).post("/api/tasks", json={"description": "x", "acceptance_claims": [claim]})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    assert captured[0]["acceptance_claims"][0]["id"] == "answer"
    assert captured[0]["metadata"]["acceptance_claims"][0]["claim"] == "final answer is exact"
    result = json.loads((data / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
    assert result["task_contract"]["acceptance_claims"][0]["support"] == "exact receipt"


def test_registry_hides_and_blocks_disabled_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    # sanity: web_search exists by default
    assert "web_search" in reg.available_tools()

    contract = build_task_contract({"description": "x", "disabled_tools": WEB_TOOLS})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract}))

    avail = set(reg.available_tools())
    schema_names = {s["function"]["name"] for s in reg.schemas()}
    core_names = {s["function"]["name"] for s in reg.schemas(core_only=True)}
    for tool in WEB_TOOLS:
        assert tool not in avail, f"{tool} should be withheld from available_tools"
        assert tool not in schema_names, f"{tool} should be hidden from schemas()"
        assert tool not in core_names, f"{tool} should be hidden from schemas(core_only=True)"
        assert reg.get_schema_by_name(tool) is None, f"{tool} should not be rediscoverable"
        blocked = reg.execute(tool, {})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked and "disabled_tools" in blocked

    # A non-disabled core tool is unaffected.
    assert "read_file" in avail
    assert reg.get_schema_by_name("read_file") is not None
    # `view_image` is intentionally NOT in the web-tool denylist: it is a LOCAL image-to-model
    # tool (outside _WEB_TOOLS), so a legitimate local-vision affordance survives web-tools-off.
    assert "view_image" not in WEB_TOOLS
    assert "view_image" in avail and reg.get_schema_by_name("view_image") is not None


def test_registry_hides_missing_credential_tools(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="task-missing-creds"))

    assert "claude_code_edit" not in reg.available_tools()
    assert reg.get_schema_by_name("claude_code_edit") is None
    blocked = reg.execute("claude_code_edit", {"prompt": "edit"})
    assert "CAPABILITY_UNAVAILABLE" in blocked
    assert "ANTHROPIC_API_KEY" in blocked

    assert reg.get_schema_by_name("create_github_issue") is None
    assert reg.get_schema_by_name("submit_skill_to_hub") is None
    assert reg.get_schema_by_name("generate_evolution_stats") is None
    assert "CAPABILITY_UNAVAILABLE" in reg.execute("submit_skill_to_hub", {"skill": "x"})
    assert "CAPABILITY_UNAVAILABLE" in reg.execute("generate_evolution_stats", {})
    reg.schemas()
    omissions = reg.capability_omissions()
    assert any(
        item.get("surface") == "tools"
        and item.get("reason") == "missing_credential"
        and "claude_code_edit" in item.get("tools", [])
        for item in omissions
    )


def test_registry_arg_aliases_and_public_tool_arg_errors(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)

    seen = {}

    def _private_search_code(ctx, query="", max_results=0, **_kwargs):
        seen["query"] = query
        seen["max_results"] = max_results
        return "ok"

    reg.override_handler("search_code", _private_search_code)
    assert reg.execute("search_code", {"query": "needle", "max_entries": 2}) == "ok"
    assert seen["query"] == "needle"
    assert seen["max_results"] == 2

    def _private_vcs_status(ctx, path="", max_chars=0):
        seen["vcs_status"] = (path, max_chars)
        return "status-ok"

    reg.override_handler("vcs_status", _private_vcs_status)
    assert reg.execute("vcs_status", {"root": "system_repo", "path": "."}) == "status-ok"
    assert seen["vcs_status"] == (".", 0)

    result = reg.execute("search_code", {"dir": "."})
    assert "TOOL_ARG_ERROR (search_code)" in result
    assert "Accepted parameters:" in result
    assert "_private_search_code" not in result
    assert "unexpected keyword" not in result

    def _internal_type_error(ctx, query=""):
        raise TypeError("internal math failed")

    reg.override_handler("search_code", _internal_type_error)
    result = reg.execute("search_code", {"query": "needle"})
    assert "TOOL_ERROR (search_code)" in result
    assert "internal math failed" in result

    result = reg.execute("commit_reviewed", {"commit_message": "x", "skip_advisory_pre_review": True})
    assert "TOOL_ARG_ERROR (commit_reviewed)" in result
    assert "skip_advisory_review" in result
    assert "skip_advisory_pre_review" not in result

    result = reg.execute("list_skills", {"foo": "bar"})
    assert "TOOL_ARG_ERROR (list_skills)" in result
    assert "Accepted parameters: none" in result
    assert "_kwargs" not in result


def test_subagent_inherits_disabled_tools():
    """control.py builds the child contract by spreading the parent contract into
    metadata.task_contract; disabled_tools must survive that spread so a subagent
    cannot use a tool the root disabled."""
    parent = build_task_contract({"description": "root", "disabled_tools": WEB_TOOLS})
    child = build_task_contract({
        "id": "child1",
        "description": "sub",
        "objective": "sub",
        "metadata": {"task_contract": {**parent, "source": "parent_delegation", "objective": "sub"}},
    })
    assert set(child["disabled_tools"]) == set(WEB_TOOLS)
