"""The access VOCABULARY: the three axes, and the matrix they index.

A profile asks to perform an operation on a resource root, and `_POLICY` is the table
that answers. Everything else in `tool_access` — resolving a path, refusing a cwd,
building the affordance map a prompt shows — is written in these terms, which is why
they live apart: this module depends on nothing above it, and everything above it
depends on this. That direction is the whole point. A vocabulary that imported its own
users would make every question about access circular.

`ResolvedResourceBinding` is here too, though it is a RESULT rather than an axis: it
is the sealed answer to one dispatch's question — which logical root, which exact
path — and it is passed to guards instead of letting each re-derive it. Keeping it
beside the axes is what makes it obvious that a binding is an ANSWER in this
vocabulary and not a fourth axis.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal


# Deferral 1: orchestrator-visible READ-ONLY roots — durable subagent (genesis) projects
# and the unnamed-deliverables container. Only ever granted {read,list,search}; NEVER
# write/edit/shell/vcs (no mutation, no shell-cwd — deliberately absent from
# resolve_shell_cwd candidates) and NEVER to acting/readonly subagents (a child must not
# read sibling projects). operator_control is capped to read-only on these too.
_READONLY_RESOURCE_ROOTS: frozenset[str] = frozenset({"subagent_projects", "deliverables"})

_TOP_LEVEL_PRINCIPAL_POLICY: dict[str, set[str]] = {
    "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "system_repo": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "runtime_data": {"read", "list", "search", "write", "edit"},
    "task_drive": {"read", "list", "write", "edit", "shell", "service"},
    "skill_payload": {"read", "list", "search", "write", "edit", "review", "shell"},
    "artifact_store": {"read", "list", "write", "shell", "service"},
    "user_files": {"read", "list", "search", "write", "edit", "shell", "service"},
    "subagent_projects": {"read", "list", "search"},
    "deliverables": {"read", "list", "search"},
}


ToolProfile = Literal[
    "self_modification",
    "workspace_task",
    "external_workspace_task",
    "acting_subagent",
    "skill_repair",
    "local_readonly_subagent",
    "operator_control",
]


ResourceRoot = Literal[
    "active_workspace",
    "system_repo",
    "runtime_data",
    "task_drive",
    "skill_payload",
    "artifact_store",
    "user_files",
    "subagent_projects",
    "deliverables",
]


Operation = Literal[
    "read",
    "list",
    "search",
    "write",
    "edit",
    "shell",
    "vcs",
    "review",
    "delegate",
    "service",
]


@dataclass(frozen=True)
class ToolAccessDecision:
    allow: bool
    reason: str = ""
    guard: str = ""


@dataclass(frozen=True)
class ResolvedResourceBinding:
    """One dispatch-selected logical root and its exact physical target."""

    profile: ToolProfile
    root: ResourceRoot
    operation: Operation
    base_path: pathlib.Path
    target_path: pathlib.Path
    source: str
    skill_name: str
    state_drive_root: pathlib.Path


_ALL_ROOTS: frozenset[str] = frozenset({
    "active_workspace",
    "system_repo",
    "runtime_data",
    "task_drive",
    "skill_payload",
    "artifact_store",
    "user_files",
    "subagent_projects",
    "deliverables",
})


_TOP_LEVEL_PRINCIPAL_PROFILES: frozenset[str] = frozenset({
    "workspace_task",
    "external_workspace_task",
    "self_modification",
})


_READ_OPS = frozenset({"read", "list", "search"})


_POLICY: dict[str, dict[str, set[str]]] = {
    "local_readonly_subagent": {
        # Read-only child VCS names still need their target binding to resolve.
        "active_workspace": set(_READ_OPS) | {"vcs"},
        "system_repo": set(_READ_OPS) | {"vcs"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
        # v6.70.0 (owner-approved): read-only scouts sent to review a skill were
        # structurally blind to its payload — a scout literally reported
        # "reviewing blind", and a correct "skill does not exist" answer was
        # indistinguishable from an access block. Payloads are skill CODE
        # (data/skills/...); grants/secrets live in data/state/skills, which
        # stays invisible to this profile.
        "skill_payload": {"read", "list", "search"},
    },
    "skill_repair": {
        "skill_payload": {"read", "list", "search", "write", "edit", "review"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
    },
    # Top-level preset names remain observable, but workspace focus never narrows
    # the ordinary principal. Independent path/credential/child/runtime guards
    # still apply after this shared operation matrix.
    "workspace_task": _TOP_LEVEL_PRINCIPAL_POLICY,
    "external_workspace_task": _TOP_LEVEL_PRINCIPAL_POLICY,
    # Mutative (acting) subagents write only inside their isolated active
    # workspace (self_worktree / external_workspace / genesis). No vcs-commit /
    # review here; the parent integrates and commits. self_worktree additionally
    # keeps protected-path discipline active in the registry (it is the system
    # repo). runtime_data stays read-only.
    "acting_subagent": {
        # Acting children write ONLY inside their isolated surface (active_workspace =
        # the self_worktree / external_workspace / genesis). task_drive / artifact_store
        # are read-only here (no extra write surface); the deliverable is a workspace.patch.
        "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "service"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
    },
    "self_modification": _TOP_LEVEL_PRINCIPAL_POLICY,
    # operator_control gets full authority on every mutable root, but the orchestrator
    # read-only roots stay read-only even here (they are deliverables/durable projects,
    # not a control surface).
    "operator_control": {
        **{root: {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "delegate", "service"}
           for root in _ALL_ROOTS if root not in _READONLY_RESOURCE_ROOTS},
        **{root: {"read", "list", "search"} for root in _READONLY_RESOURCE_ROOTS},
    },
}


_SUBAGENT_CAPABILITY_TO_OPERATION: dict[str, Operation] = {
    "write": "write",
    "edit": "edit",
    "shell": "shell",
    "vcs": "vcs",
    "review": "review",
    "delegate": "delegate",
    "service": "service",
}


__all__ = [
    "ToolProfile",
    "ResourceRoot",
    "Operation",
    "ToolAccessDecision",
    "ResolvedResourceBinding",
    "_ALL_ROOTS",
    "_TOP_LEVEL_PRINCIPAL_PROFILES",
    "_READ_OPS",
    "_POLICY",
    "_SUBAGENT_CAPABILITY_TO_OPERATION",
]
