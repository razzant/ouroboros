"""Version-negotiated Claudexor execution-workspace wire contract."""

from types import SimpleNamespace

from ouroboros.subagents import (
    DelegationRoute,
    delegated_execution_workspace_root,
    delegated_run_shape,
)
from ouroboros.tools.delegate import _start_request


def _request(version: str, *, acting: bool) -> dict:
    shape = delegated_run_shape(acting)
    stable_root = "/tmp/stable-project"
    snapshot_root = "/tmp/private-execution-snapshot" if acting else stable_root
    gateway = SimpleNamespace(engine_version=version)
    execution_root = delegated_execution_workspace_root(gateway, shape, snapshot_root)
    scope_root = stable_root if execution_root else snapshot_root
    return _start_request(
        SimpleNamespace(), DelegationRoute("codex"), shape, scope_root,
        "do the work", 300, "host instructions", execution_root,
    )


def test_legacy_strict_schema_keeps_the_byte_compatible_execution_shape():
    request = _request("3.8.0", acting=True)
    assert request["scope"]["root"] == "/tmp/private-execution-snapshot"
    assert request["execution"] == {"isolation": "live", "delegated": True}


def test_new_schema_receives_the_private_snapshot_as_execution_workspace():
    request = _request("3.8.1", acting=True)
    assert request["scope"]["root"] == "/tmp/stable-project"
    assert request["execution"] == {
        "isolation": "live",
        "delegated": True,
        "workspaceRoot": "/tmp/private-execution-snapshot",
    }


def test_readonly_shape_never_sends_a_live_execution_workspace():
    request = _request("99.0.0", acting=False)
    assert request["mode"] == "ask" and "execution" not in request


def test_retry_binding_keeps_snapshot_as_host_execution_root(tmp_path, monkeypatch):
    import ouroboros.subagent_worktrees as worktrees
    import ouroboros.tools.delegate_integration as integration

    stable_root = tmp_path / "stable-project"
    snapshot_root = tmp_path / "private-snapshot"
    stable_root.mkdir()
    snapshot_root.mkdir()
    request = {
        "primaryHarness": "codex",
        "model": "gpt-5.6-sol",
        "effort": "high",
        "access": "workspace_write",
        "mode": "agent",
        "maxSeconds": 300,
        "scope": {"kind": "project", "root": str(stable_root)},
        "execution": {
            "isolation": "live",
            "delegated": True,
            "workspaceRoot": str(snapshot_root),
        },
    }
    record = {
        "request": request,
        "project_id": "project-stable",
        "project_owned": False,
        "idempotency_key": "stored-key",
        "snapshot_id": "snapshot-1",
        "target_root": str(stable_root),
        "baseline_sha": "a" * 40,
        "authority_source": "acting_constraint",
    }
    monkeypatch.setattr(
        integration, "_validated_invocation",
        lambda *_args, **_kwargs: (record, None),
    )
    monkeypatch.setattr(
        integration, "_retry_binding_refusal", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        integration, "_mutation_authority",
        lambda *_args, **_kwargs: ({"target_root": str(stable_root)}, None),
    )
    monkeypatch.setattr(
        worktrees, "find_execution_snapshot",
        lambda _snapshot_id: {"path": str(snapshot_root)},
    )

    binding, refusal = integration._resolve_retry_invocation(
        SimpleNamespace(task_id="task-a"), tmp_path, "invocation-a", "same prompt",
    )

    assert refusal is None
    assert binding is not None
    assert binding.request_body is request
    assert binding.root == str(snapshot_root)
    assert binding.target_root == str(stable_root)
