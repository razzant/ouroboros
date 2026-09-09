from __future__ import annotations

import pytest

from ouroboros.copilot_acp_policy import (
    CopilotPermissions, copilot_child_env, runtime_options, validate_runtime_settings, validate_runtime_task,
)
from ouroboros.server_runtime import has_startup_ready_provider


def permission(kind="edit", **raw):
    return {
        "toolCall": {"kind": kind, "rawInput": raw},
        "options": [{"optionId": "always", "kind": "allow_always"}, {"optionId": "once", "kind": "allow_once"}],
    }


def test_workspace_policy_allows_scoped_edits_and_explicit_shell(tmp_path):
    policy = CopilotPermissions(tmp_path, "workspace")
    assert policy.decide(permission(path="new.py")) == {"outcome": {"outcome": "selected", "optionId": "once"}}
    assert policy.decide(permission("execute", command="pytest -q", cwd=str(tmp_path)))["outcome"]["optionId"] == "once"
    assert policy.decide(permission("read", path=str(tmp_path / "new.py")))["outcome"]["outcome"] == "selected"


@pytest.mark.parametrize("permission_params", [
    {}, {"toolCall": []}, permission([], path="file.py"), permission(path="../outside.py"),
    permission(path=""), permission(paths=[None]), permission(paths=42),
    permission("edit"), permission("fetch", url="https://example.com"),
    {"toolCall": {"kind": "edit", "rawInput": {}, "locations": [None]}, "options": []},
    {**permission(path="a.py"), "options": [{"optionId": "always", "kind": "allow_always"}]},
])
def test_unknown_or_outside_permissions_deny(tmp_path, permission_params):
    assert CopilotPermissions(tmp_path, "workspace").decide(permission_params) == {"outcome": {"outcome": "cancelled"}}


def test_symlink_escape_denies_but_new_in_workspace_path_works(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available")
    policy = CopilotPermissions(workspace, "workspace")
    assert policy.decide(permission(path="link/file.py"))["outcome"]["outcome"] == "cancelled"
    assert policy.decide(permission(path="new/file.py"))["outcome"]["outcome"] == "selected"


def test_read_only_tools_do_not_grant_edits_shell_or_other_features(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda path: "/installed/copilot")
    policy = CopilotPermissions(tmp_path, "read_only")
    assert policy.decide(permission("read", path="source.py"))["outcome"]["outcome"] == "selected"
    assert policy.decide(permission(path="source.py"))["outcome"]["outcome"] == "cancelled"
    assert policy.decide(permission("execute", command="echo write"))["outcome"]["outcome"] == "cancelled"
    command = policy.command("chosen-model", executable="copilot")
    assert command[-2:] == ["--model", "chosen-model"]
    assert "--available-tools=view,glob,grep" in command
    assert "--acp" in command and "--disallow-temp-dir" in command
    assert not any(flag in command for flag in ("--allow-all", "--allow-all-tools", "--allow-all-paths", "--yolo"))


def test_child_environment_preserves_login_discovery_without_host_secrets_or_byok():
    env = copilot_child_env({
        "PATH": "/bin", "HOME": "/home/owner", "COPILOT_HOME": "/home/owner/copilot",
        "https_proxy": "http://proxy", "NODE_EXTRA_CA_CERTS": "ca.pem",
        "COPILOT_GITHUB_TOKEN": "explicit-copilot-auth",
        "GITHUB_TOKEN": "stale", "GH_TOKEN": "unrelated",
        "ANTHROPIC_API_KEY": "paid-key", "OPENAI_API_KEY": "paid-key",
        "COPILOT_PROVIDER_BASE_URL": "https://byok.example", "COPILOT_PROVIDER_API_KEY": "paid-key",
        "COPILOT_SESSION_ID": "parent", "COPILOT_RESUME": "parent",
        "COPILOT_ALLOW_ALL": "true", "COPILOT_CUSTOM_INSTRUCTIONS_DIRS": "parent-instructions",
        "HOST_SERVICE_TOKEN": "host", "OUROBOROS_SETTINGS_PATH": "owner-state",
        "CLAUDECODE": "parent", "CODEX_THREAD_ID": "parent",
    })
    assert set(env) == {"PATH", "HOME", "COPILOT_HOME", "https_proxy", "NODE_EXTRA_CA_CERTS", "COPILOT_GITHUB_TOKEN"}


def test_explicit_runtime_and_model_snapshot_survive_later_settings_changes():
    captured = runtime_options({
        "execution_backend": "copilot_acp", "copilot_model": "selected-model",
        "copilot_permission_policy": "workspace",
    }, {})
    assert runtime_options({"metadata": captured}, {
        "OUROBOROS_TASK_BACKEND": "native", "OUROBOROS_COPILOT_MODEL": "other-model",
    }) == captured
    assert runtime_options({"execution_backend": "native"}, {"OUROBOROS_TASK_BACKEND": "copilot_acp"}) == {"execution_backend": "native"}


@pytest.mark.parametrize("task", [
    {"_is_direct_chat": True}, {"type": "evolution"}, {"type": "deep_self_review"},
    {"delegation_role": "subagent"}, {"_ephemeral_turn": True}, {"_presence_turn": True},
])
def test_copilot_default_never_replaces_native_governance_or_chat(task):
    assert runtime_options(task, {"OUROBOROS_TASK_BACKEND": "copilot_acp"}) == {"execution_backend": "native"}
    with pytest.raises(ValueError, match="managed root"):
        runtime_options({**task, "execution_backend": "copilot_acp"}, {})


@pytest.mark.parametrize("extra", [
    {"disabled_tools": ["run_command"]}, {"allowed_resources": {"network": False}},
    {"resource_policy": {"protected_artifacts": [{"paths": ["secret"]}]}},
    {"executor_ref": {"type": "docker_exec"}}, {"attachments": [{"path": "input.txt"}]},
    {"metadata": {"force_plan": True}}, {"service_teardown": "keep"},
])
def test_incompatible_contracts_are_refused_not_ignored(extra):
    with pytest.raises(ValueError, match="does not support|unsupported"):
        validate_runtime_task({"execution_backend": "copilot_acp", "workspace_root": "workspace", **extra})
    validate_runtime_task({"execution_backend": "native", **extra})


def test_keyless_readiness_is_explicit_and_does_not_invent_provider_access():
    assert not has_startup_ready_provider({})
    assert has_startup_ready_provider({"OUROBOROS_TASK_BACKEND": "copilot_acp"})
    assert not has_startup_ready_provider({"OUROBOROS_TASK_BACKEND": "unknown"})
    validate_runtime_settings({"OUROBOROS_TASK_BACKEND": "copilot_acp", "OUROBOROS_COPILOT_PERMISSION_POLICY": "workspace"})
    with pytest.raises(ValueError):
        validate_runtime_settings({"OUROBOROS_TASK_BACKEND": "unknown"})
    with pytest.raises(ValueError):
        validate_runtime_settings({"OUROBOROS_COPILOT_BIN": "copilot\n--allow-all"})


def test_missing_workspace_and_cli_are_actionable(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="external workspace_root"):
        validate_runtime_task({"execution_backend": "copilot_acp"})
    monkeypatch.setattr("shutil.which", lambda path: None)
    with pytest.raises(ValueError, match="copilot login"):
        CopilotPermissions(tmp_path, "workspace").command()


@pytest.mark.parametrize("model", ["--allow-all", "model & echo unexpected", "%COMSPEC%", "model\nnext"])
def test_model_is_an_identifier_not_windows_shell_syntax(tmp_path, model):
    with pytest.raises(ValueError):
        runtime_options({"execution_backend": "copilot_acp", "copilot_model": model}, {})
    with pytest.raises(ValueError):
        validate_runtime_settings({"OUROBOROS_COPILOT_MODEL": model})
    with pytest.raises(ValueError):
        CopilotPermissions(tmp_path, "workspace").command(model)
