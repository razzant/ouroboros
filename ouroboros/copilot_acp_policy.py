"""Explicit local Copilot launch and one-shot ACP permission policy.

Permission semantics adapted from Q00/ouroboros (Copyright 2025 Q00, MIT);
full notice: ``ouroboros/gateways/copilot_acp.py``. Workspace approval is
NOT an OS sandbox: an approved shell executes as the local user.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
from dataclasses import dataclass
from typing import Any, Mapping

from ouroboros.config import COPILOT_PERMISSION_POLICIES, SETTINGS_DEFAULTS, TASK_EXECUTION_BACKENDS

_BASE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "COPILOT_HOME",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "GH_HOST", "COPILOT_GH_HOST", "COPILOT_GITHUB_TOKEN",
})
_READ_TOOLS = ("view", "glob", "grep")
_WORKSPACE_TOOLS = (*_READ_TOOLS, "edit", "create", "bash", "read_bash", "stop_bash", "list_bash")


def _model_id(value: Any) -> str:
    """Keep model ids opaque but safe for Windows npm command shims too."""
    if not isinstance(value, str):
        raise ValueError("copilot_model must be a single model id")
    model = value.strip()
    if model and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", model):
        raise ValueError("copilot_model must be a model id, not a command line")
    return model


def runtime_options(task: Mapping[str, Any], settings: Mapping[str, Any] | None = None) -> dict:
    """Snapshot a managed root's runtime, independently of every raw-model slot."""
    settings = os.environ if settings is None else settings
    eligible = (
        str(task.get("type") or "task") == "task"
        and not task.get("_is_direct_chat") and not task.get("_ephemeral_turn")
        and not task.get("_presence_turn")
        and str(task.get("delegation_role") or "root") != "subagent"
        and not task.get("parent_task_id")
    )
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    backend = task.get("execution_backend", metadata.get("execution_backend", settings.get(
        "OUROBOROS_TASK_BACKEND", SETTINGS_DEFAULTS["OUROBOROS_TASK_BACKEND"],
    ) if eligible else "native"))
    if backend not in TASK_EXECUTION_BACKENDS:
        raise ValueError("execution_backend must be native or copilot_acp")
    if backend != "native" and not eligible:
        raise ValueError("Copilot ACP runs managed root tasks, not native chat, subagents or self-evolution")
    if backend == "native":
        if task.get("copilot_model") or task.get("copilot_permission_policy"):
            raise ValueError("Copilot options require execution_backend=copilot_acp")
        return {"execution_backend": "native"}
    model = task.get("copilot_model", metadata.get("copilot_model", settings.get("OUROBOROS_COPILOT_MODEL", "")))
    policy = task.get("copilot_permission_policy", metadata.get("copilot_permission_policy", settings.get(
        "OUROBOROS_COPILOT_PERMISSION_POLICY", SETTINGS_DEFAULTS["OUROBOROS_COPILOT_PERMISSION_POLICY"],
    )))
    model = _model_id(model)
    if policy not in COPILOT_PERMISSION_POLICIES:
        raise ValueError("copilot_permission_policy must be read_only or workspace")
    return {"execution_backend": backend, "copilot_model": model, "copilot_permission_policy": policy}


def validate_runtime_task(task: Mapping[str, Any]) -> None:
    """Refuse contracts the external CLI cannot enforce rather than discard them."""
    if task.get("execution_backend") == "native":
        return
    if not str(task.get("workspace_root") or "").strip():
        raise ValueError("Copilot ACP requires an external workspace_root; it cannot edit Ouroboros itself")
    sources = [task, task.get("metadata") or {}, task.get("task_contract") or {}]
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Copilot ACP task metadata and contract must be objects")
        for key in (
            "executor_ref", "workspace_executor", "allowed_resources", "resource_policy",
            "disabled_tools", "attachments", "attachment_manifest", "attachment_manifest_ref",
            "force_plan", "context_requires_self_body_docs",
        ):
            if source.get(key):
                raise ValueError(f"Copilot ACP does not support {key}; use the native backend for this contract")
        if source.get("service_teardown") == "keep":
            raise ValueError("Copilot ACP closes all task processes; service_teardown=keep is unsupported")


def validate_runtime_settings(settings: Mapping[str, Any]) -> None:
    """Validate owner settings without launching, authenticating, or choosing a model."""
    for key, choices in (
        ("OUROBOROS_TASK_BACKEND", TASK_EXECUTION_BACKENDS),
        ("OUROBOROS_COPILOT_PERMISSION_POLICY", COPILOT_PERMISSION_POLICIES),
    ):
        if key in settings and settings[key] not in choices:
            raise ValueError(f"{key} must be one of: {', '.join(choices)}")
    for key in ("OUROBOROS_COPILOT_BIN", "OUROBOROS_COPILOT_MODEL"):
        if key in settings and (
            not isinstance(settings[key], str) or any(char in settings[key] for char in ("\0", "\r", "\n"))
        ):
            raise ValueError(f"{key} must be a single string, not a command line")
    if "OUROBOROS_COPILOT_MODEL" in settings:
        _model_id(settings["OUROBOROS_COPILOT_MODEL"])


def copilot_child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Keep OS/auth discovery, not host secrets, BYOK overrides or parent sessions.

Use the CLI's stored login by default. Explicit automation credentials belong
in COPILOT_GITHUB_TOKEN; ambient GH_TOKEN/GITHUB_TOKEN are not forwarded.
"""
    return {
        key: value for key, value in (os.environ if base is None else base).items()
        if key.upper() in _BASE_ENV_KEYS
    }


@dataclass(frozen=True)
class CopilotPermissions:
    workspace: pathlib.Path
    policy: str

    def command(self, model: str = "", executable: str = "") -> list[str]:
        if self.policy not in COPILOT_PERMISSION_POLICIES:
            raise ValueError("Unknown Copilot permission policy")
        model = _model_id(model)
        candidate = executable or os.environ.get("OUROBOROS_COPILOT_BIN") or SETTINGS_DEFAULTS["OUROBOROS_COPILOT_BIN"]
        binary = shutil.which(str(pathlib.Path(candidate).expanduser()))
        if not binary:
            raise ValueError("Copilot CLI not found. Install it, run copilot login, or set OUROBOROS_COPILOT_BIN.")
        tools = _READ_TOOLS if self.policy == "read_only" else _WORKSPACE_TOOLS
        command = [
            binary, "--acp", "--no-auto-update", "--no-ask-user", "--no-color",
            "--no-bash-env", "--disable-builtin-mcps", "--disallow-temp-dir",
            "--no-remote", "--no-remote-export", f"--available-tools={','.join(tools)}",
        ]
        if model:
            command.extend(["--model", model])
        return command

    def decide(self, params: dict) -> dict:
        denied = {"outcome": {"outcome": "cancelled"}}
        if self.policy not in COPILOT_PERMISSION_POLICIES:
            return denied
        call, options = params.get("toolCall"), params.get("options")
        if not isinstance(call, dict) or not isinstance(options, list):
            return denied
        kinds = {"read", "search"} if self.policy == "read_only" else {"read", "search", "edit", "execute"}
        if not isinstance(call.get("kind"), str) or call["kind"] not in kinds:
            return denied
        raw, locations = call.get("rawInput", {}), call.get("locations", [])
        if not isinstance(raw, dict) or not isinstance(locations, list):
            return denied
        paths = []
        for location in locations:
            if not isinstance(location, dict):
                return denied
            paths.append(location.get("path"))
        paths.extend(raw[key] for key in ("path", "file_path", "fileName", "cwd", "workingDirectory", "working_directory") if key in raw)
        extra = raw.get("paths", [])
        if isinstance(extra, str):
            extra = [extra]
        if not isinstance(extra, list):
            return denied
        paths.extend(extra)
        if call.get("kind") == "edit" and not paths:
            return denied
        if call["kind"] == "execute" and not (
            isinstance(raw.get("command"), str) and raw["command"].strip()
        ):
            return denied
        if call["kind"] in {"read", "search"} and not paths and not (
            isinstance(raw.get("pattern"), str) and raw["pattern"].strip()
        ):
            return denied
        # ACP session/new supplies cwd. Copilot's bash/glob/grep requests may
        # omit a path; that means the already-bound session workspace, not a
        # request for an arbitrary root. Shell effects still are not sandboxed.
        if not paths:
            paths.append(str(self.workspace))
        for value in paths:
            if not isinstance(value, str) or not value.strip():
                return denied
            try:
                path = pathlib.Path(value).expanduser()
                resolved = (path if path.is_absolute() else self.workspace / path).resolve()
                if not resolved.is_relative_to(self.workspace.resolve()):
                    return denied
            except (OSError, ValueError, RuntimeError):
                return denied
        for option in options:
            if isinstance(option, dict) and option.get("kind") == "allow_once" and isinstance(option.get("optionId"), str) and option["optionId"]:
                return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
        return denied
