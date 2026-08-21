"""Workspace-root compatibility and stable project identity for delegated starts."""

from __future__ import annotations

import json
import subprocess

import pytest

from ouroboros.config import CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION


@pytest.fixture(autouse=True)
def _transport_fixture(monkeypatch):
    from ouroboros import claudexor_daemon, delegate_custody, subagent_runtime, subagents
    from ouroboros.gateways import claudexor as gateway_module
    from ouroboros.tools import delegate

    monkeypatch.setattr(
        claudexor_daemon, "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )
    original_actor = delegate.prepare_delegate_start_actor

    def explicit_actor(ctx, drive_root, **kwargs):
        if kwargs.get("recovering"):
            return original_actor(ctx, drive_root, **kwargs)
        route = subagents.get_subagent_harness()
        if route is None:
            return original_actor(ctx, drive_root, **kwargs)
        token = subagent_runtime._EXACT_START_SELECTION.set({
            "snapshot": {
                "schema": 1, "selected_subagent_id": "workspace-root-fixture",
                "config_fingerprint": "workspace-root-fixture-v1",
                "route": {"kind": "agent_session", "target_id": route.route_id},
                "effort": route.effort,
            },
        })
        try:
            return original_actor(ctx, drive_root, **kwargs)
        finally:
            subagent_runtime._EXACT_START_SELECTION.reset(token)

    delegate_custody._CUSTODY.clear()
    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", explicit_actor)
    yield
    delegate_custody._CUSTODY.clear()


def _ctx(tmp_path, *, task_id="t-workspace-root"):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path.parent / f"wt-{tmp_path.name}"
    worktree.mkdir()
    subprocess.run(["git", "init"], cwd=worktree, capture_output=True, check=True)
    (worktree / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed"],
        cwd=worktree, capture_output=True, check=True,
    )
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(
            mode="acting_subagent", surface="self_worktree", write_root=str(worktree),
        ),
    )
    ctx.workspace_root = str(worktree)
    ctx.workspace_mode = "self_worktree"
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root"}
    return ctx


def _started_request(tmp_path, monkeypatch, *, engine_version):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway

    seen = {}

    class Stub:
        def handshake(self, **_kwargs):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"],
            }]}

        def quota_snapshots(self):
            return []

        def find_project_id(self, _root):
            return "prj-existing"

        def register_project(self, _root):
            raise AssertionError("the fixture registration should be reused")

        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            return {"runId": "run-write", "runDir": "/tmp/run-write"}

        def close(self):
            pass

    Stub.engine_version = engine_version
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **k: Stub())
    delegate._CUSTODY.clear()
    payload = json.loads(delegate._delegate_start(_ctx(tmp_path), "edit the README"))
    delegate._CUSTODY.clear()
    assert payload["status"] == "started", payload
    return seen["request"], payload


def test_pinned_old_engine_keeps_legacy_snapshot_scope_shape(tmp_path, monkeypatch):
    request, payload = _started_request(tmp_path, monkeypatch, engine_version="3.8.0")
    target = str(tmp_path.parent / f"wt-{tmp_path.name}")
    assert request["scope"]["root"] != target
    assert "workspaceRoot" not in request["execution"]
    assert payload["execution_root"] == request["scope"]["root"]


def test_future_engine_uses_stable_target_and_private_execution_root(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody

    request, payload = _started_request(
        tmp_path, monkeypatch, engine_version=CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION,
    )
    target = str(tmp_path.parent / f"wt-{tmp_path.name}")
    assert request["scope"]["root"] == target
    assert request["execution"]["workspaceRoot"] != target
    assert custody.replay(tmp_path)["run-write"].project_persistent is True
    assert payload["authority_target_root"] == target


def test_stable_project_registration_survives_settlement(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway

    removed = []

    class StableGateway:
        engine_version = CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION

        def handshake(self, **_kwargs): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True,
                                    "status": "ok",
                                    "accessProfilesSupported": ["readonly", "workspace_write"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, _root): return ""
        def register_project(self, _root): return "prj-stable"
        def start_run(self, _request, *, idempotency_key=""): return {"runId": "run-stable"}
        def remove_project(self, project_id): removed.append(project_id)
        def close(self): pass

    stub = StableGateway()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **k: stub)
    payload = json.loads(delegate._delegate_start(_ctx(tmp_path, task_id="t-stable"), "edit"))
    assert payload["status"] == "started"
    run = custody.replay(tmp_path)["run-stable"]
    run.ledger_recorded = True
    settled = custody.settle_run(tmp_path, stub, run,
                                 {"summary": {"state": "succeeded", "spendUsd": 0.0}})
    assert settled["project_retired"] is False
    assert removed == []


def test_retry_replays_stable_scope_and_workspace_root_verbatim(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway

    seen = []

    class RetryGateway:
        engine_version = CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION

        def handshake(self, **_kwargs): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                    "accessProfilesSupported": ["readonly", "workspace_write"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, _root): return "prj-existing"
        def register_project(self, _root): raise AssertionError("stable registration is reused")
        def start_run(self, request, *, idempotency_key=""):
            seen.append(json.loads(json.dumps(request)))
            if len(seen) == 1:
                raise gateway.ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)
            return {"runId": "run-retried"}
        def close(self): pass

    stub = RetryGateway()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **k: stub)
    ctx = _ctx(tmp_path, task_id="t-retry")
    lost = json.loads(delegate._delegate_start(ctx, "edit"))
    retried = json.loads(delegate._delegate_start(ctx, "edit", retry_of=lost["pending_invocation_id"]))
    assert retried["status"] == "started"
    assert seen[1] == seen[0]


def test_new_shape_retry_refuses_old_engine_before_post(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway

    class StaleGateway:
        engine_version = CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION

        def __init__(self): self.starts = 0
        def handshake(self, **_kwargs): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                    "accessProfilesSupported": ["readonly", "workspace_write"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, _root): return "prj-existing"
        def register_project(self, _root): raise AssertionError("registration is reused")
        def start_run(self, *_args, **_kwargs):
            self.starts += 1
            raise gateway.ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)
        def close(self): pass

    stub = StaleGateway()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **k: stub)
    ctx = _ctx(tmp_path, task_id="t-stale")
    lost = json.loads(delegate._delegate_start(ctx, "edit"))
    stub.engine_version = "3.8.0"
    stale = json.loads(delegate._delegate_start(ctx, "edit", retry_of=lost["pending_invocation_id"]))
    assert stale["reason"] == "engine_rejects_delegated_workspace_root"
    assert stub.starts == 1


def _orphan(tmp_path, request, *, persistent=True):
    import ouroboros.delegate_custody as custody

    assert custody.record_start_requested(
        tmp_path, invocation_id="inv-recovery", task_id="t-recovery", request=request,
        route="some-route", project_id="prj-stable", project_owned=True,
        project_persistent=persistent,
    )
    return custody.pending_invocations(tmp_path)[0]


def test_recovery_blocks_future_shape_on_old_engine_without_post(tmp_path):
    import ouroboros.delegate_custody as custody

    record = _orphan(tmp_path, {
        "mode": "agent", "access": "workspace_write",
        "scope": {"kind": "project", "root": "/target"},
        "execution": {"isolation": "live", "delegated": True,
                       "workspaceRoot": "/snapshot"},
    })

    class OldGateway:
        engine_version = "3.8.0"
        def start_run(self, *_args, **_kwargs): raise AssertionError("must not POST")

    result = custody._recover_pending_invocation(tmp_path, OldGateway(), record)
    assert result["reason"] == "engine_rejects_delegated_workspace_root"
    assert custody.pending_invocations(tmp_path)[0]["invocation_id"] == "inv-recovery"


def test_recovery_keeps_legacy_pending_on_future_workspace_floor(tmp_path):
    import ouroboros.delegate_custody as custody
    from ouroboros.gateways import claudexor as gateway

    record = _orphan(tmp_path, {
        "mode": "agent", "access": "workspace_write",
        "scope": {"kind": "project", "root": "/target"},
        "execution": {"isolation": "live", "delegated": True},
    })

    class FutureGateway:
        engine_version = CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION
        def __init__(self): self.calls = 0
        def start_run(self, *_args, **_kwargs):
            self.calls += 1
            raise gateway.ClaudexorUnavailable(
                "execution_workspace_required", "legacy body", status_code=400,
            )
        def remove_project(self, *_args, **_kwargs): raise AssertionError("persistent project")

    stub = FutureGateway()
    result = custody._recover_pending_invocation(tmp_path, stub, record)
    assert stub.calls == 1
    assert result["reason"] == "legacy_workspace_root_requires_compatible_retry"
    assert custody.pending_invocations(tmp_path)[0]["invocation_id"] == "inv-recovery"
    assert not any(row.get("type") == custody.START_FAILED
                   for row in custody._iter_rows(custody.event_log_path(tmp_path)))


def test_future_recovery_attempts_unknown_legacy_replay(tmp_path):
    import ouroboros.delegate_custody as custody
    from ouroboros.gateways import claudexor as gateway

    record = _orphan(tmp_path, {
        "mode": "agent", "access": "workspace_write",
        "scope": {"kind": "project", "root": "/target"},
        "execution": {"isolation": "live", "delegated": True},
    })

    class FutureGateway:
        engine_version = CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION
        def start_run(self, *_args, **_kwargs):
            raise gateway.ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)

    result = custody._recover_pending_invocation(tmp_path, FutureGateway(), record)
    assert result["action"] == "recovery_unreachable"
    assert custody.pending_invocations(tmp_path)[0]["invocation_id"] == "inv-recovery"


def test_pending_recovery_retains_persistent_project_marker(tmp_path):
    import ouroboros.delegate_custody as custody

    assert custody.record_start_requested(
        tmp_path, invocation_id="inv-persistent", task_id="t-persistent",
        request={"mode": "agent", "scope": {"root": "/target"}},
        project_id="prj-stable", project_owned=True, project_persistent=True,
    )
    assert custody.pending_invocations(tmp_path)[0]["project_persistent"] is True
