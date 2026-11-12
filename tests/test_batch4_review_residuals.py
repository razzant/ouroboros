"""Batch4 exact-delta review R1-R2, driven through their real consumers.

R1: a GitHub refusal before the invocation's FIRST gh launch settles its own
    launch claim; one after an earlier launch, or any unknown outcome, keeps it.
R2: a task's MCP discovery passes that task's launch admission before EACH
    transport entry; Settings probes carry no task authority.
"""
import json
import subprocess
from types import SimpleNamespace

import pytest

from tests.test_batch4_full_repair import _park
from tests.test_batch4_producer_custody import _registry



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
        if argv[1:3] == ["repo", "view"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nameWithOwner": "owner/selected", "url": "https://github.example/owner/selected"}), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return registry, queue, workers, launched


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
        if argv[:1] == ["gh"] and argv[1:3] == ["repo", "view"]:
            launched.append(argv[1:3])
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"nameWithOwner": "owner/selected", "url": "https://github.example/owner/selected"}))
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

    assert launched == [["repo", "view"], ["api", "repos/owner/selected/pulls/7"]]
    assert "GH_TARGET_" in result.text and result.status != "ok", result
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


# --- R2: MCP discovery --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
