"""The closed access vocabulary and the profile x root x operation policy matrix.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
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
    "operator_control",
})


_READ_OPS = frozenset({"read", "list", "search"})
# Operations that can MUTATE a root. "vcs" is deliberately not write-like on its
# own: read-only children carry {read,list,search,vcs} only so their status/diff
# bindings resolve (the registry tool allowlist exposes no mutating vcs names to
# them), and every profile that can actually mutate through vcs also holds
# write/edit/shell on the same root — property-pinned by the projection test.
_WRITE_LIKE_OPS: frozenset[str] = frozenset({"write", "edit", "shell", "service"})

# The matrix is CLOSED under two implications (TZ-1 cluster E): a root a profile
# may ``read`` it may also ``list`` and ``search`` (both only read the same bytes
# through the same per-file guards, so withholding them bought no protection and
# sent the model into a probe loop over roots it could already read file by
# file), and a root it may ``write`` it may also ``edit`` (an exact replacement
# is a narrower write). Neither implication adds a write-like operation to a
# read-only row, so every read-only ceiling — the read-only child, the
# orchestrator read-only roots — holds byte for byte. The literals below are
# written closed; ``_close_operations`` keeps a future row honest.
_READ_IMPLIES: frozenset[str] = frozenset({"list", "search"})
_WRITE_IMPLIES: frozenset[str] = frozenset({"edit"})


def _close_operations(ops: set[str]) -> set[str]:
    """``ops`` closed under read⇒list,search and write⇒edit (a new set)."""
    closed = set(ops)
    if "read" in closed:
        closed |= _READ_IMPLIES
    if "write" in closed:
        closed |= _WRITE_IMPLIES
    return closed


_TOP_LEVEL_PRINCIPAL_POLICY: dict[str, set[str]] = {
    "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "system_repo": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "runtime_data": {"read", "list", "search", "write", "edit"},
    "task_drive": {"read", "list", "search", "write", "edit", "shell", "service"},
    "skill_payload": {"read", "list", "search", "write", "edit", "review", "shell"},
    "artifact_store": {"read", "list", "search", "write", "edit", "shell", "service"},
    "user_files": {"read", "list", "search", "write", "edit", "shell", "service"},
    "subagent_projects": {"read", "list", "search"},
    "deliverables": {"read", "list", "search"},
}


_POLICY: dict[str, dict[str, set[str]]] = {
    "local_readonly_subagent": {
        # Read-only child VCS names still need their target binding to resolve.
        "active_workspace": set(_READ_OPS) | {"vcs"},
        "system_repo": set(_READ_OPS) | {"vcs"},
        # Read⇒search closure: the search tool applies the same per-file secret /
        # owner-state guards and match masking a child's read_file does.
        "runtime_data": set(_READ_OPS),
        "task_drive": set(_READ_OPS),
        "artifact_store": set(_READ_OPS),
        # v6.70.0 (owner-approved): read-only scouts sent to review a skill were
        # structurally blind to its payload — a scout literally reported
        # "reviewing blind", and a correct "skill does not exist" answer was
        # indistinguishable from an access block. Payloads are skill CODE
        # (data/skills/...); grants/secrets live in data/state/skills, which
        # stays invisible to this profile.
        "skill_payload": {"read", "list", "search"},
        # Owner T4=A (#1105): the owner-visible Deliverables container is readable
        # by a read-only child; `subagent_projects` stays top-level only.
        "deliverables": {"read", "list", "search"},
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
        "runtime_data": set(_READ_OPS),
        "task_drive": set(_READ_OPS),
        "artifact_store": set(_READ_OPS),
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

# Structural closure, in place: the row objects (and the one shared top-level
# dict) keep their identity, so profile aliasing stays observable; a literal
# that forgets an implied operation is corrected here, never silently narrower
# than the rule above says. Idempotent over the shared matrix.
for _matrix in _POLICY.values():
    for _ops in _matrix.values():
        _ops |= _close_operations(_ops)
del _matrix, _ops


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
