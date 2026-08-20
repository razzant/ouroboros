"""Tool API v2 vocabulary: profiles, roots, operations, and the policy matrix.

The closed enums a caller may name (tool profile, resource root, operation,
subagent capability), the two frozen decision/binding records the access
surface returns, and the profile x root x operation matrix every consumer of
the access decision reads. Data and types only: the decision itself, the
projections over it, and the physical resolution all live with their own
owners.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal


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
SubagentCapability = Literal[
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

# Deferral 1: orchestrator-visible READ-ONLY roots — durable subagent (genesis) projects
# and the unnamed-deliverables container. Only ever granted {read,list,search}; NEVER
# write/edit/shell/vcs (no mutation, no shell-cwd — deliberately absent from
# resolve_shell_cwd candidates) and NEVER to acting/readonly subagents (a child must not
# read sibling projects). operator_control is capped to read-only on these too.
_READONLY_RESOURCE_ROOTS: frozenset[str] = frozenset({"subagent_projects", "deliverables"})
_TOP_LEVEL_PRINCIPAL_PROFILES: frozenset[str] = frozenset({
    "workspace_task",
    "external_workspace_task",
    "self_modification",
})

_READ_OPS = frozenset({"read", "list", "search"})


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
SUBAGENT_CAPABILITIES: tuple[str, ...] = tuple(_SUBAGENT_CAPABILITY_TO_OPERATION.keys())
