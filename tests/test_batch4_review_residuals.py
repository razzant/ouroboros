"""Batch4 exact-delta review R1-R2, driven through their real consumers.

R1: a GitHub refusal before the invocation's FIRST gh launch settles its own
    launch claim; one after an earlier launch, or any unknown outcome, keeps it.
R2: a task's MCP discovery passes that task's launch admission before EACH
    transport entry; Settings probes carry no task authority.
"""
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_full_repair import _park
from tests.test_batch4_producer_custody import _registry
from tests.test_batch4_repair_compositions import _running

pytestmark = pytest.mark.serial


def _github_task(tmp_path, monkeypatch, target):
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")  # the credential gate precedes the handler
    ctx = registry._ctx
    ctx.task_lifecycle_bound = True
    if target == "folderless":
        ctx.workspace_mode, ctx.workspace_root, ctx.project_id = "", None, "project-folderless"
    launched, real_run = [], subprocess.run

    def run(argv, *args, **kwargs):
        if argv[:1] != ["gh"]:
            return real_run(argv, *args, **kwargs)
        launched.append(argv[1:3])
        if target == "cli_missing":
            raise FileNotFoundError(2, "No such file or directory", "gh")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return registry, queue, workers, launched


@pytest.mark.parametrize("tool,args", [
    ("github_read", {"method": "get", "kind": "issue", "target": 7, "repo": ""}),
    ("github_write", {"method": "update", "target": 7, "state": "closed", "repo": ""}),
    ("github_write", {"method": "create_issue", "title": "T", "body": "B", "labels": ["bug"], "repo": ""}),
    ("github_read", {"method": "list", "kind": "pr"}),
])
@pytest.mark.parametrize("target", ["folderless", "unavailable", "invalid", "cli_missing"])
def test_refusal_before_the_first_launch_settles_its_claim(tmp_path, monkeypatch, tool, args, target):
    from ouroboros.task_results import load_task_result

    registry, queue, workers, launched = _github_task(tmp_path, monkeypatch, target)
    if target == "unavailable":
        registry._ctx.task_metadata = {"_project_room_note": "Project registry unavailable"}
    if target == "invalid":
        args = {**args, "repo": None}

    result = registry.execute_result(tool, args)

    assert result.status != "ok" and result.meta.get("operation_outcome") == "completed_no_effect", result
    first = {"get": ["api", "repos/{owner}/{repo}/issues/7"], "update": ["api", "--method"],
             "create_issue": ["api", "--method"], "list": ["api", "--include"]}[args["method"]]
    assert launched == ([first] if target == "cli_missing" else [])  # exec of `gh` itself failed
    expected = {"folderless": "GH_TARGET_REQUIRED", "unavailable": "GH_TARGET_UNAVAILABLE",
                "invalid": "GH_TARGET_INVALID", "cli_missing": "`gh` CLI not found"}[target]
    assert expected in result.text
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"] is True


@pytest.mark.parametrize("second", ["unavailable", "folderless"])
def test_later_refusal_keeps_prior_effect_unknown_but_closes_local_invocation(tmp_path, monkeypatch, second):
    """A second subcall's target refusal cannot claim no effect for the already-launched first one."""
    import json

    from ouroboros.task_results import load_task_result

    registry, queue, workers, launched = _github_task(tmp_path, monkeypatch, "workspace")
    ctx, workspace = registry._ctx, registry._ctx.workspace_root
    ctx.task_metadata = {}
    fake = subprocess.run

    def read_then_lose_target(argv, *args, **kwargs):
        if argv[:1] == ["gh"] and argv[1] == "api":  # the pull request read; the diff subcall then finds no target
            launched.append(argv[1:3])
            if second == "unavailable":
                workspace.rmdir()
            else:
                ctx.workspace_mode, ctx.workspace_root, ctx.project_id = "", None, "project-folderless"
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                {"number": 7, "title": "x", "state": "open", "html_url": "https://github.example/owner/selected/pull/7"}))
        return fake(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", read_then_lose_target)
    result = registry.execute_result("github_read", {"method": "get", "kind": "pr", "target": 7, "include": ["diff"], "repo": ""})

    assert launched == [["api", "repos/{owner}/{repo}/pulls/7"]]
    assert "GH_TARGET_" in result.text and result.status != "ok", result
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_returned_unknown_gh_outcome_preserves_error_without_ghost_invocation(tmp_path, monkeypatch, failure):
    from ouroboros.task_results import load_task_result

    registry, queue, workers, launched = _github_task(tmp_path, monkeypatch, "workspace")
    real_run = subprocess.run

    def unknown(argv, *args, **kwargs):
        if argv[:1] != ["gh"]:
            return real_run(argv, *args, **kwargs)
        launched.append(argv[1:3])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        return SimpleNamespace(returncode=1, stdout="", stderr="HTTP 502: Bad Gateway (https://api.github.com)")

    monkeypatch.setattr(subprocess, "run", unknown)
    result = registry.execute_result("github_write", {"method": "comment", "target": 7, "body": "text", "repo": ""})

    # The send, then exactly one lookup that also failed: the outcome stays unknown, nothing is resent.
    assert launched == [["api", "--method"], ["api", "repos/{owner}/{repo}/issues/7/comments?since=" + launched[1][1].split("since=")[1]]]
    assert result.status != "ok" and result.meta.get("operation_outcome") == "unknown", result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


def test_direct_handler_calls_attest_nothing(tmp_path, monkeypatch):
    """Only a public invocation scope proves "first launch"; internal callers do not."""
    from ouroboros.tools import github_read
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar)

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="t", project_id="project-folderless")
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        assert "GH_TARGET_REQUIRED" in github_read.run_read(ctx, method="get", kind="issue", target=7)
        assert "operation_outcome" not in _published_tool_result(ctx, sentinel).meta
        assert "TOOL_ARG_ERROR" in github_read.run_read(ctx, method="get", kind="issue", target=0)
        assert "operation_outcome" not in _published_tool_result(ctx, sentinel).meta
    finally:
        _restore_tool_result_sidecar(token)


# --- R2: MCP discovery --------------------------------------------------------------

def _discovery(tmp_path, monkeypatch, *, on_initialize=None):
    from ouroboros import config, mcp_client
    from ouroboros.tools.registry import ToolRegistry

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")
    servers = [{"id": sid, "enabled": True, "transport": "stdio", "command": "server"} for sid in ("alpha", "beta")]
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "load_settings", lambda: {"MCP_ENABLED": True, "MCP_SERVERS": servers})
    mcp_client.reset_manager_for_tests()
    monkeypatch.setattr(mcp_client, "_manager", None)  # teardown drops this test's manager
    events = []

    @asynccontextmanager
    async def transport(cfg):
        events.append(("transport", cfg.id))
        yield (cfg.id, None)

    class Session:
        def __init__(self, server_id, _write):
            self.server_id = server_id

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            pass

        async def initialize(self):
            events.append(("initialize", self.server_id))
            if on_initialize:
                on_initialize(self.server_id)

        async def list_tools(self, cursor=None):
            return SimpleNamespace(tools=[SimpleNamespace(name="lookup", description="", inputSchema={})],
                                   nextCursor=None)

    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.task_lifecycle_bound = True
    return registry, events


def _mcp_names(registry):
    return sorted(schema["function"]["name"] for schema in registry.schemas()
                  if schema["function"]["name"].startswith("mcp_"))


def test_task_never_discovers_even_when_not_paused(tmp_path, monkeypatch):
    registry, events = _discovery(tmp_path, monkeypatch)
    assert _mcp_names(registry) == []
    assert events == []


def test_pause_and_unpaused_task_both_use_published_catalog(tmp_path, monkeypatch):
    from ouroboros import mcp_client
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    registry, events = _discovery(tmp_path, monkeypatch)
    mcp_client.ensure_configured_from_settings()
    manager = mcp_client.get_manager()
    assert manager.refresh_all()["refreshed"]["beta"]["published"]
    expected = ["mcp_alpha__lookup", "mcp_beta__lookup"]
    assert len(events) == 4
    events.clear()
    assert request_owner_pause("root", request_id="pause-before")["ok"]
    assert _mcp_names(registry) == expected
    assert events == []
    write_task_result(tmp_path, "other", "running", root_task_id="other", chat_id=0)
    registry._ctx.task_id = registry._ctx.root_task_id = "other"
    assert _mcp_names(registry) == expected
    assert events == []


def test_pause_without_catalog_opens_nothing_and_keeps_gap(tmp_path, monkeypatch):
    from ouroboros import mcp_client
    from supervisor.owner_pause_control import request_owner_pause

    registry, events = _discovery(tmp_path, monkeypatch)
    assert request_owner_pause("root", request_id="pause-unlisted")["ok"]
    assert _mcp_names(registry) == [] and events == []
    status = mcp_client.get_manager().status_payload()["servers"]
    assert all(not row["last_attempted"] and row["listed_at"] is None for row in status)
    assert all("not listed yet" in row["detail"] for row in mcp_client.get_manager().catalog_notices())


def test_settings_probes_carry_no_task_authority(tmp_path, monkeypatch):
    from ouroboros import mcp_client
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause

    _registry_unused, events = _discovery(tmp_path, monkeypatch)
    assert request_owner_pause("root", request_id="pause-settings")["ok"]
    assert read_fence(tmp_path, "root")["state"] == "requested"
    mcp_client.ensure_configured_from_settings()
    manager = mcp_client.get_manager()

    assert manager.refresh_server("alpha")["ok"] is True
    probe = manager.test_server({"id": "gamma", "enabled": True, "transport": "stdio", "command": "server"})
    assert probe["ok"] is True and probe["tool_count"] == 1
    assert manager.refresh_all()["refreshed"]["beta"]["ok"] is True
    assert [kind for kind, _ in events].count("transport") == 4


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
