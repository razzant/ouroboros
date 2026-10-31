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

import pytest


from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.contracts.task_contract import (
    build_task_contract,
    normalize_disabled_tools,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._store_shared import read_result

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
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
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
    result = read_result(data, task_id)
    assert result["task_contract"]["disabled_tools"] == WEB_TOOLS


def test_api_tasks_create_carries_acceptance_claims(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import api_tasks_create

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    captured = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
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
    result = read_result(data, task_id)
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
    monkeypatch.setattr("ouroboros.tools.github.github_cli_configured", lambda: False)
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

    assert "github_write" not in reg.available_tools()
    assert reg.get_schema_by_name("github_write") is None
    assert reg.get_schema_by_name("submit_skill_to_hub") is None
    assert reg.get_schema_by_name("generate_evolution_stats") is not None
    blocked = reg.execute("submit_skill_to_hub", {"skill": "x"})
    assert "CAPABILITY_UNAVAILABLE" in blocked
    assert "GITHUB_TOKEN" in blocked
    assert "generate_evolution_stats" in reg.available_tools()
    reg.schemas()
    omissions = reg.capability_omissions()
    assert any(
        item.get("surface") == "tools"
        and item.get("reason") == "missing_credential"
        and "submit_skill_to_hub" in item.get("tools", [])
        for item in omissions
    )


def test_registry_arg_aliases_and_public_tool_arg_errors(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)

    seen = {}

    def _private_search_code(
        ctx, query="", max_results=0, _resolved_binding=None, **_kwargs
    ):
        seen["query"] = query
        seen["max_results"] = max_results
        seen["binding"] = _resolved_binding
        return "ok"

    reg.override_handler("search_code", _private_search_code)
    assert reg.execute("search_code", {"query": "needle", "max_entries": 2}) == "ok"
    assert seen["query"] == "needle"
    assert seen["max_results"] == 2
    assert seen["binding"] is not None

    def _private_vcs_read(
        ctx, action="status", path="", max_chars=0, root="system_repo", _resolved_binding=None,
    ):
        seen["vcs_read"] = (path, max_chars, root, _resolved_binding)
        return "status-ok"

    reg.override_handler("vcs_read", _private_vcs_read)
    assert reg.execute("vcs_read", {"action": "status", "root": "system_repo", "path": "."}) == "status-ok"
    assert seen["vcs_read"][:3] == (".", 0, "system_repo")
    assert seen["vcs_read"][3].root == "system_repo"

    result = reg.execute("search_code", {"dir": "."})
    assert "TOOL_ARG_ERROR (search_code)" in result
    assert "Accepted parameters:" in result
    assert "_private_search_code" not in result
    assert "unexpected keyword" not in result

    def _internal_type_error(ctx, query="", _resolved_binding=None):
        raise TypeError("internal math failed")

    reg.override_handler("search_code", _internal_type_error)
    result = reg.execute("search_code", {"query": "needle"})
    assert "TOOL_ERROR (search_code)" in result
    assert "internal math failed" in result

    result = reg.execute("commit_reviewed", {"commit_message": "x", "skip_advisory_pre_review": True})
    assert "TOOL_ARG_ERROR (commit_reviewed)" in result
    assert "skip_advisory_review" in result
    assert "unsupported argument(s): skip_advisory_pre_review" in result

    result = reg.execute("list_skills", {"foo": "bar"})
    assert "TOOL_ARG_ERROR (list_skills)" in result
    assert "Accepted parameters: name, offset, snapshot" in result
    assert "_kwargs" not in result


def test_legacy_claude_code_edit_contract_also_disables_delegate_start(tmp_path, monkeypatch):
    """D10 compatibility shim (registry `_disabled_tools`): a saved contract that
    carried disabled_tools=["claude_code_edit"] — the retired external coding
    gateway — must keep withholding the SUCCESSOR delegated-coding verb
    `delegate_start`. The dead name itself stays in the disabled set too, so old
    contracts round-trip and the dead name is blocked as disabled, not surfaced
    as an unknown-tool surprise."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    # sanity: the successor exists by default
    assert "delegate_start" in reg.available_tools()

    contract = build_task_contract({"description": "x", "disabled_tools": ["claude_code_edit"]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract}))

    avail = set(reg.available_tools())
    schema_names = {s["function"]["name"] for s in reg.schemas()}
    assert "delegate_start" not in avail
    assert "delegate_start" not in schema_names
    assert reg.get_schema_by_name("delegate_start") is None
    blocked = reg.execute("delegate_start", {"prompt": "x"})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked and "disabled_tools" in blocked
    # The legacy name is inert but still reported as withheld-by-contract.
    dead = reg.execute("claude_code_edit", {"prompt": "x"})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in dead and "disabled_tools" in dead

    # A contract that does NOT name claude_code_edit leaves delegate_start alone.
    contract2 = build_task_contract({"description": "x", "disabled_tools": ["web_search"]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract2}))
    assert "delegate_start" in set(reg.available_tools())


def test_retired_run_ci_tests_has_no_shim_and_no_table_row(tmp_path, monkeypatch):
    """`run_ci_tests` pushed a branch and dispatched a workflow; its successor
    `github_read(method="checks")` only reads. A saved contract that withheld the
    retired tool therefore withholds nothing of the successor, and the dead name
    stays inert: blocked as disabled where a contract names it, unknown elsewhere."""
    import importlib.util

    from ouroboros.safety import TOOL_POLICY
    from ouroboros.tool_capabilities import OBSERVE_WORLD_MUTATION_TOOLS
    from ouroboros.tools import github
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS

    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    assert importlib.util.find_spec("ouroboros.tools.ci") is None
    assert "ci" not in ToolRegistry._FROZEN_TOOL_MODULES and "github" in ToolRegistry._FROZEN_TOOL_MODULES
    for table in (TOOL_POLICY, OBSERVE_WORLD_MUTATION_TOOLS, _GITHUB_TOKEN_TOOLS):
        assert "run_ci_tests" not in table
    assert "github_read" in TOOL_POLICY and "github_read" in _GITHUB_TOKEN_TOOLS

    contract = build_task_contract({"description": "x", "disabled_tools": ["run_ci_tests"]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract}))
    dead = reg.execute("run_ci_tests", {})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in dead and "disabled_tools" in dead
    assert "github_read" in set(reg.available_tools())
    assert reg.get_schema_by_name("github_read") is not None
    assert "RESOURCE_CONSTRAINT_BLOCKED" not in reg.execute("github_read", {"method": "checks"})  # reaches the reader

    # A contract that names the successor does withhold it; without one the dead name is unknown.
    contract = build_task_contract({"description": "x", "disabled_tools": ["github_read"]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract}))
    assert "github_read" not in set(reg.available_tools())
    assert reg.execute("run_ci_tests", {}).startswith("⚠️ Unknown tool")


_RETIRED_GITHUB_CASES = [
    # retired name, successor, a call it covered (blocked), a call it did not cover (reaches the handler)
    ("list_github_issues", "github_read", {"method": "list", "kind": "issue"}, {"method": "list", "kind": "pr", "limit": 0}),
    ("get_github_issue", "github_read", {"method": "get", "kind": "issue", "target": 7}, {"method": "get", "kind": "pr", "target": 0}),
    ("list_github_prs", "github_read", {"method": "list", "kind": "pr"}, {"method": "list", "kind": "issue", "limit": 0}),
    ("get_github_pr", "github_read", {"method": "get", "kind": "pr", "target": 7}, {"method": "get", "kind": "issue", "target": 0}),
    ("get_github_issue", "github_read", {"method": "list", "kind": "comment", "target": 7}, {"method": "list", "kind": "comment", "limit": 0}),
    ("get_github_pr", "github_read", {"method": "list", "kind": "comment", "target": " 7 "}, {"method": "list", "kind": "comment", "target": "", "limit": 0}),
    ("get_github_pr", "github_read", {"method": "list", "kind": "review_comment", "target": "7"}, {"method": "list", "kind": "review_comment", "target": 0, "limit": 0}),
    ("get_github_pr", "github_read", {"method": "list", "kind": "review", "target": 7}, {"method": "get", "kind": "issue", "target": 0}),
    ("get_github_checks", "github_read", {"method": "checks", "target": 7}, {"method": "get", "kind": "issue", "target": 0}),
    ("comment_on_issue", "github_write", {"method": "comment", "target": 7, "body": "x"}, {"method": "update", "target": 0, "title": "t"}),
    ("comment_on_pr", "github_write", {"method": "review", "target": 7, "event": "APPROVE"}, {"method": "create_issue", "title": " "}),
    ("close_github_issue", "github_write", {"method": "update", "target": 7, "state": "closed"}, {"method": "update", "target": 0, "title": "t"}),
    ("create_github_issue", "github_write", {"method": "create_issue", "title": "T"}, {"method": "comment", "target": 0, "body": "x"}),
    ("create_github_pr", "github_write", {"method": " create_pr ", "title": "T"}, {"method": "create_issue", "title": " "}),
    ("update_github_pr", "github_write", {"method": "update", "kind": "pr", "target": 7, "title": "T"}, {"method": "update", "kind": "pr", "target": 0, "draft": False}),
    ("update_github_pr", "github_write", {"method": "update", "kind": "issue", "target": 7, "body": ""}, {"method": "update", "target": 0, "state": "closed"}),
    ("update_github_pr", "github_write", {"method": "update", "target": 7, "base": "main"}, {"method": "comment", "target": 0}),
    ("set_github_pr_ready", "github_write", {"method": " update ", "kind": " pr ", "target": 7, "draft": False}, {"method": "update", "kind": "pr", "target": 0, "body": ""}),
    ("set_github_pr_ready", "github_write", {"method": "update", "kind": "pr", "target": 7, "draft": True}, {"method": "create_pr"}),
    ("get_github_run_log", "github_read", {"method": " get ", "kind": " run ", "target": 11, "include": ["log"]}, {"method": "get", "kind": "run", "target": 0}),
    ("list_github_run_artifacts", "github_read", {"method": "list", "kind": "artifact", "target": 11}, {"method": "download", "kind": "artifact", "target": 0}),
    ("download_github_artifact", "github_read", {"method": "download", "kind": "artifact", "target": 22}, {"method": "list", "kind": "artifact", "target": 0}),
]


@pytest.mark.parametrize("retired,successor,blocked_call,allowed_call", _RETIRED_GITHUB_CASES)
def test_a_retired_github_name_withholds_its_successor_methods(tmp_path, monkeypatch, retired, successor, blocked_call, allowed_call):
    """The narrow GitHub tools became methods of github_read / github_write; a saved contract that
    withheld one keeps withholding the method(s) it covered, plus the api hatch, and nothing else."""
    from ouroboros.tools import github
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS, _RETIRED_GITHUB_TOOLS

    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    assert retired not in _GITHUB_TOKEN_TOOLS and _RETIRED_GITHUB_TOOLS[retired][0] == successor
    contract = build_task_contract({"description": "x", "disabled_tools": [retired]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id="task-compat", task_metadata={"task_contract": contract}))

    # The successor stays advertised: only the covered method(s) are withheld, by name of the retired tool.
    assert successor in set(reg.available_tools()) and reg.get_schema_by_name(successor) is not None
    blocked = reg.execute(successor, blocked_call)
    assert blocked.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED") and retired in blocked and successor in blocked, blocked
    hatch = reg.execute(successor, {"method": "api", "route": "repos/{owner}/{repo}/issues"} if successor == "github_read"
                        else {"method": "api", "route": "repos/{owner}/{repo}/issues", "input": {"title": "T"}})
    assert hatch.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED") and retired in hatch, hatch
    allowed = reg.execute(successor, allowed_call)
    assert allowed.startswith("⚠️ TOOL_ARG_ERROR"), allowed  # the handler itself answered: the guard let it through
    dead = reg.execute(retired, {})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in dead and "disabled_tools" in dead

    # The other direction: naming the successor withholds the whole tool; no contract leaves the dead name unknown.
    contract = build_task_contract({"description": "x", "disabled_tools": [successor]})
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"task_contract": contract}))
    assert successor not in set(reg.available_tools()) and reg.get_schema_by_name(successor) is None
    assert reg.execute_result(successor, allowed_call).status == "blocked"
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data))
    assert reg.execute(retired, {}).startswith("⚠️ Unknown tool")
    assert successor in set(reg.available_tools())


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
