"""The closed access vocabulary and the profile x root x operation policy matrix.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    import pathlib


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
    logical_base_path: pathlib.Path | None = None


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


_READONLY_RESOURCE_ROOTS: frozenset[str] = frozenset({"subagent_projects", "deliverables"})


_TOP_LEVEL_PRINCIPAL_PROFILES: frozenset[str] = frozenset({
    "workspace_task",
    "external_workspace_task",
    "self_modification",
})


_READ_OPS = frozenset({"read", "list", "search"})
# Operations that can MUTATE a root. "vcs" is deliberately not write-like on its
# own: read-only children carry {read,list,search,vcs} only so their status/diff
# bindings resolve (the registry tool allowlist exposes no mutating vcs names to
# them), and every profile that can actually mutate through vcs also holds
# write/edit/shell on the same root — property-pinned by the projection test.
_WRITE_LIKE_OPS: frozenset[str] = frozenset({"write", "edit", "shell", "service"})


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
