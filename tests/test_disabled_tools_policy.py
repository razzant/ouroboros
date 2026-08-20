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
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.contracts.task_contract import (
    build_task_contract,
    normalize_disabled_tools,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import ToolResult

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
    result = json.loads((data / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
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

    assert "create_github_issue" not in reg.available_tools()
    assert reg.get_schema_by_name("create_github_issue") is None
    assert reg.get_schema_by_name("submit_skill_to_hub") is None
    assert reg.get_schema_by_name("generate_evolution_stats") is None
    blocked = reg.execute("submit_skill_to_hub", {"skill": "x"})
    assert "CAPABILITY_UNAVAILABLE" in blocked
    assert "GITHUB_TOKEN" in blocked
    assert "CAPABILITY_UNAVAILABLE" in reg.execute("generate_evolution_stats", {})
    reg.schemas()
    omissions = reg.capability_omissions()
    assert any(
        item.get("surface") == "tools"
        and item.get("reason") == "missing_credential"
        and "submit_skill_to_hub" in item.get("tools", [])
        for item in omissions
    )


def test_capability_resource_guard_owner_facades_preserve_identity():
    from ouroboros.tools import registry, registry_guards

    for name in (
        "_WEB_TOOLS",
        "_resource_allowed",
        "_disabled_tools",
        "_GITHUB_TOKEN_TOOLS",
        "_builtin_tool_availability",
    ):
        assert getattr(registry, name) is getattr(registry_guards, name)


@pytest.mark.parametrize(
    (
        "name",
        "args",
        "contract",
        "expected_status",
        "expected_code",
        "legacy_status",
        "expected_text",
    ),
    (
        (
            "create_github_issue",
            {},
            {
                "disabled_tools": ["create_github_issue"],
                "allowed_resources": {"network": False},
            },
            "blocked",
            "RESOURCE_CONSTRAINT_BLOCKED",
            "resource_constraint_blocked",
            (
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.disabled_tools "
                "withholds 'create_github_issue' for this task."
            ),
        ),
        (
            "create_github_issue",
            {},
            {},
            "unavailable",
            "CAPABILITY_UNAVAILABLE",
            "unavailable",  # T1 §A.18
            (
                "⚠️ CAPABILITY_UNAVAILABLE: 'create_github_issue' is unavailable: "
                "missing_credential (GITHUB_TOKEN)."
            ),
        ),
        (
            "web_search",
            {"query": "x"},
            {"allowed_resources": {"web": False}},
            "unavailable",
            "CAPABILITY_UNAVAILABLE",
            "unavailable",  # T1 §A.18
            (
                "⚠️ CAPABILITY_UNAVAILABLE: 'web_search' is unavailable: "
                "missing_credential (web_search_backend)."
            ),
        ),
        (
            "vlm_query",
            {"image_url": "https://example.test/image.png"},
            {"allowed_resources": {"web": False}},
            "blocked",
            "RESOURCE_CONSTRAINT_BLOCKED",
            "resource_constraint_blocked",
            (
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: remote image_url for vlm_query "
                "requires allowed_resources.web/network."
            ),
        ),
        (
            "youtube_transcript",
            {},
            {"allowed_resources": {"web": False}},
            "blocked",
            "RESOURCE_CONSTRAINT_BLOCKED",
            "resource_constraint_blocked",
            (
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.web=false "
                "blocks 'youtube_transcript'."
            ),
        ),
        (
            "vcs_pull_ff",
            {},
            {"allowed_resources": {"network": False}},
            "blocked",
            "RESOURCE_CONSTRAINT_BLOCKED",
            "resource_constraint_blocked",
            (
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false "
                "blocks 'vcs_pull_ff'."
            ),
        ),
        (
            "delegate_start",
            {"prompt": "x"},
            {
                "disabled_tools": ["claude_code_edit"],
                "allowed_resources": {"network": False},
            },
            "blocked",
            "RESOURCE_CONSTRAINT_BLOCKED",
            "resource_constraint_blocked",
            (
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.disabled_tools "
                "withholds 'delegate_start' for this task."
            ),
        ),
    ),
)
def test_builtin_capability_resource_guards_are_native_and_never_dispatch(
    name,
    args,
    contract,
    expected_status,
    expected_code,
    legacy_status,
    expected_text,
    tmp_path,
    monkeypatch,
):
    import ouroboros.loop_tool_execution as execution
    from ouroboros.tools import search

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(search, "_available_web_search_backends", lambda: [])
    calls = {"handler": 0, "safety": 0}

    def _physical(*_args, **_kwargs):
        calls["handler"] += 1
        raise AssertionError("a denied built-in tool must not dispatch")

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    assert name in registry._entries
    registry.override_handler(name, _physical)
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_a, **_k: calls.__setitem__("safety", calls["safety"] + 1) or (True, ""),
    )
    monkeypatch.setattr(execution, "persist_call", lambda *_a, **_k: {})
    registry.set_context(ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_id="task-capability-resource",
        task_metadata={"task_contract": contract},
    ))
    logs = tmp_path / "logs"
    logs.mkdir()

    typed = registry.execute_result(name, args)
    assert typed == ToolResult(
        status=expected_status,
        code=expected_code,
        text=expected_text,
    )
    assert registry.execute(name, args) == expected_text
    row = execution._execute_single_tool(
        registry,
        {"id": "call-resource", "function": {"name": name, "arguments": json.dumps(args)}},
        logs,
        "task-capability-resource",
    )
    assert calls == {"handler": 0, "safety": 0}
    assert row["result"] == expected_text
    assert row["is_error"] is True
    assert row["result_meta"] == {
        "status": legacy_status,
        "tool_result_status": expected_status,
        "tool_result_code": expected_code,
        "tool_result_meta": {},
    }


def test_capability_resource_guard_allows_an_admitted_call():
    from ouroboros.tools.registry_guards import _capability_resource_guard_result

    ctx = SimpleNamespace(
        task_id="task-admitted",
        task_metadata={
            "task_contract": {"allowed_resources": {"web": True, "network": True}},
        },
        task_contract={},
    )
    assert _capability_resource_guard_result(ctx, "vcs_pull_ff", {}) is None


@pytest.mark.parametrize("failure", (ImportError("missing"), RuntimeError("broken")))
def test_builtin_availability_probe_fail_open(failure, monkeypatch):
    from ouroboros.tools import search
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    def _fail():
        raise failure

    monkeypatch.setattr(search, "_available_web_search_backends", _fail)
    ctx = SimpleNamespace(task_id="task-probe", task_metadata={}, task_contract={})
    assert _builtin_tool_availability("web_search", ctx) == (True, "", "")


def test_builtin_availability_bare_registry_skips_runtime_probes(monkeypatch):
    from ouroboros.tools import search
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    monkeypatch.setattr(
        search,
        "_available_web_search_backends",
        lambda: pytest.fail("bare structural inventory must not probe credentials"),
    )
    ctx = SimpleNamespace(task_id="", task_metadata={}, task_contract={})
    assert _builtin_tool_availability("web_search", ctx) == (True, "", "")


@pytest.mark.parametrize(
    ("provider", "ephemeral", "expected_code", "legacy_error", "legacy_status"),
    (
        # T1 §A.4: the ephemeral-turn denial is a denial, not a clean call.
        ("extension", True, "ACCESS_BLOCKED", True, "blocked"),
        ("extension", False, "RESOURCE_CONSTRAINT_BLOCKED", True, "resource_constraint_blocked"),
        ("mcp", False, "RESOURCE_CONSTRAINT_BLOCKED", True, "resource_constraint_blocked"),
    ),
)
def test_external_resource_guard_precedes_child_policy_and_never_dispatches(
    provider,
    ephemeral,
    expected_code,
    legacy_error,
    legacy_status,
    tmp_path,
    monkeypatch,
):
    import ouroboros.loop_tool_execution as execution
    from ouroboros import extension_loader, mcp_client
    from ouroboros.tools import extension_dispatch

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    calls = {"discovery": 0, "physical": 0, "safety": 0}

    def _physical(*_args, **_kwargs):
        calls["physical"] += 1
        raise AssertionError("a denied external tool must not dispatch")

    name = "ext_4_demo_ping" if provider == "extension" else "mcp_demo__ping"
    if provider == "extension":
        descriptor = {"name": name, "skill": "demo", "handler": _physical}

        def _get_tool(candidate):
            calls["discovery"] += 1
            return descriptor if candidate == name else None

        monkeypatch.setattr(extension_loader, "get_tool", _get_tool)
        monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    else:
        monkeypatch.setattr(
            mcp_client,
            "ensure_configured_from_settings",
            lambda **_kwargs: calls.__setitem__("discovery", calls["discovery"] + 1),
        )
        monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda candidate: candidate == name)
        monkeypatch.setattr(extension_dispatch, "_dispatch_mcp_tool_result", _physical)

    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_a, **_k: calls.__setitem__("safety", calls["safety"] + 1) or (True, ""),
    )
    monkeypatch.setattr(execution, "persist_call", lambda *_a, **_k: {})
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_id="task-external-resource",
        task_metadata={"task_contract": {"allowed_resources": {"network": False}}},
        task_constraint=TaskConstraint(
            mode="acting_subagent",
            allow_enable=False,
            surface="external_workspace",
        ),
    )
    ctx.is_ephemeral_turn = ephemeral
    registry.set_context(ctx)
    logs = tmp_path / "logs"
    logs.mkdir()

    row = execution._execute_single_tool(
        registry,
        {"id": "call-resource", "function": {"name": name, "arguments": "{}"}},
        logs,
        "task-external-resource",
    )

    assert calls["discovery"] > 0
    assert calls["physical"] == 0
    assert calls["safety"] == 0
    assert row["is_error"] is legacy_error
    assert row["result_meta"] == {
        "status": legacy_status,
        "tool_result_status": "blocked",
        "tool_result_code": expected_code,
        "tool_result_meta": {},
    }
    if expected_code == "RESOURCE_CONSTRAINT_BLOCKED":
        assert row["result"] == (
            "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false "
            f"blocks external tool {name!r}."
        )
    else:
        assert row["result"].startswith("⚠️ EPHEMERAL_TURN_RESTRICTED: external tool ")


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_code", "expected_meta"),
    (
        ("extension", "unavailable", "EXTENSION_UNAVAILABLE", {"dynamic_provider": True}),
        ("mcp", "error", "UNKNOWN_TOOL", {}),
    ),
)
def test_external_discovery_failure_does_not_fabricate_a_resource_denial(
    provider,
    expected_status,
    expected_code,
    expected_meta,
    tmp_path,
    monkeypatch,
):
    import ouroboros.loop_tool_execution as execution
    from ouroboros import extension_loader, mcp_client
    from ouroboros.tools import extension_dispatch

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    calls = {"discovery": 0, "physical": 0, "safety": 0}
    name = "ext_4_demo_ping" if provider == "extension" else "mcp_demo__ping"

    def _physical(*_args, **_kwargs):
        calls["physical"] += 1
        raise AssertionError("an unavailable external tool must not dispatch")

    if provider == "extension":
        descriptor = {"name": name, "skill": "demo", "handler": _physical}

        def _get_tool(candidate):
            calls["discovery"] += 1
            return descriptor if candidate == name else None

        monkeypatch.setattr(extension_loader, "get_tool", _get_tool)
        monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: False)
        monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_k: None)
        monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda _candidate: False)
    else:
        def _configuration_failure(**_kwargs):
            calls["discovery"] += 1
            raise RuntimeError("configuration unavailable")

        monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", _configuration_failure)
        monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda candidate: candidate == name)
    monkeypatch.setattr(extension_dispatch, "_dispatch_mcp_tool_result", _physical)
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_a, **_k: calls.__setitem__("safety", calls["safety"] + 1) or (True, ""),
    )
    monkeypatch.setattr(execution, "persist_call", lambda *_a, **_k: {})
    registry.set_context(ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_id="task-external-unavailable",
        task_metadata={"task_contract": {"allowed_resources": {"network": False}}},
    ))
    logs = tmp_path / "logs"
    logs.mkdir()

    row = execution._execute_single_tool(
        registry,
        {"id": "call-unavailable", "function": {"name": name, "arguments": "{}"}},
        logs,
        "task-external-unavailable",
    )
    expected_text = f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(registry._entries))}"

    assert calls["discovery"] > 0
    assert calls["physical"] == 0
    assert calls["safety"] == 0
    # T1 §A.1/§A.3: an unavailable extension and a tool that does not exist are both
    # honest failures of the call; the text alone carried no marker, so both used to
    # be recorded as clean.
    assert row["result"] == expected_text
    assert row["is_error"] is True
    assert row["result_meta"] == {
        "status": "unavailable" if expected_status == "unavailable" else "unknown_tool",
        "tool_result_status": expected_status,
        "tool_result_code": expected_code,
        "tool_result_meta": expected_meta,
    }


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

    def _private_vcs_status(
        ctx, path="", max_chars=0, root="system_repo", _resolved_binding=None,
    ):
        seen["vcs_status"] = (path, max_chars, root, _resolved_binding)
        return "status-ok"

    reg.override_handler("vcs_status", _private_vcs_status)
    assert reg.execute("vcs_status", {"root": "system_repo", "path": "."}) == "status-ok"
    assert seen["vcs_status"][:3] == (".", 0, "system_repo")
    assert seen["vcs_status"][3].root == "system_repo"

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
    assert "skip_advisory_pre_review" not in result

    result = reg.execute("list_skills", {"foo": "bar"})
    assert "TOOL_ARG_ERROR (list_skills)" in result
    assert "Accepted parameters: none" in result
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
