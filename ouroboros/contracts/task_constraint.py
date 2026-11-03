"""Structured per-task execution constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional


_LOCAL_READONLY_SUBAGENT_MODE = "local_readonly_subagent"
_ACTING_SUBAGENT_MODE = "acting_subagent"

# Valid write surfaces for mutative (acting) subagents. SSOT shared with
# tool_access.active_tool_profile so the fail-closed floor and the schema agree.
# - self_worktree:     isolated current-tree copy of the selected Git source.
#                      Host task metadata binds its source and own-body identity.
# - external_workspace: a pre-existing external git working tree.
# - genesis:           a from-scratch project the supervisor provisions as a new
#                      empty git repo under the durable projects root (game/site/
#                      new Ouroboros). Like external_workspace it is NOT the system
#                      repo, so protected-path discipline does not apply; the parent
#                      is still the sole committer of the live body.
VALID_WRITE_SURFACES: frozenset[str] = frozenset({
    "self_worktree",
    "external_workspace",
    "genesis",
})


@dataclass(frozen=True)
class TaskConstraint:
    mode: str = "normal"
    skill_name: str = ""
    payload_root: str = ""
    allow_enable: bool = True
    allow_review: bool = True
    extra_allowlist: tuple[str, ...] = ()
    # Acting (mutative) subagent authority envelope. Populated only for
    # mode == "acting_subagent"; ignored otherwise. Machine-enforced; the
    # parent stays the sole committer (parent_only_commit) and children return
    # a workspace.patch (return_kind).
    surface: str = ""
    write_root: str = ""
    base_sha: str = ""
    protected_paths_grant: bool = False
    external_tool_grants: tuple[str, ...] = ()
    parent_only_commit: bool = True
    return_kind: str = "workspace_patch"

    @property
    def has_selected_skill(self) -> bool:
        """An ordinary task's payload selection, including persisted Repair rows."""
        return bool(
            self.mode in {"normal", "skill_repair"}
            and self.skill_name and self.payload_root
        )


def normalize_task_constraint(value: Any) -> Optional[TaskConstraint]:
    if isinstance(value, TaskConstraint):
        if value.mode == _LOCAL_READONLY_SUBAGENT_MODE:
            return TaskConstraint(mode=_LOCAL_READONLY_SUBAGENT_MODE, allow_enable=False, allow_review=False)
        if value.mode == _ACTING_SUBAGENT_MODE:
            # These flags govern tool expansion and review/commit gates. The
            # separately admitted review_change capability only asks independent
            # reviewers about assigned work; the parent remains the committer.
            return replace(value, allow_enable=False, allow_review=False, parent_only_commit=True)
        return value
    if not isinstance(value, Mapping):
        return None
    extra = value.get("extra_allowlist") or ()
    if not isinstance(extra, (list, tuple)):
        extra = ()
    mode = str(value.get("mode") or "normal").strip() or "normal"
    if mode == _LOCAL_READONLY_SUBAGENT_MODE:
        return TaskConstraint(mode=_LOCAL_READONLY_SUBAGENT_MODE, allow_enable=False, allow_review=False)
    if mode == _ACTING_SUBAGENT_MODE:
        return _normalize_acting_constraint(value, extra)
    return TaskConstraint(
        mode=mode,
        skill_name=str(value.get("skill_name") or "").strip(),
        payload_root=str(value.get("payload_root") or "").strip().replace("\\", "/").strip("/"),
        allow_enable=_coerce_bool(value.get("allow_enable", True), default=True),
        allow_review=_coerce_bool(value.get("allow_review", True), default=True),
        extra_allowlist=tuple(str(item) for item in extra if str(item).strip()),
    )


def _normalize_acting_constraint(value: Mapping, extra: Any) -> TaskConstraint:
    """Build the machine-enforced authority envelope for an acting subagent."""
    surface = str(value.get("surface") or "").strip().lower()
    if surface not in VALID_WRITE_SURFACES:
        surface = ""
    grants_raw = value.get("external_tool_grants") or ()
    if not isinstance(grants_raw, (list, tuple)):
        grants_raw = ()
    return TaskConstraint(
        mode=_ACTING_SUBAGENT_MODE,
        skill_name="",
        payload_root="",
        allow_enable=False,
        allow_review=False,
        extra_allowlist=tuple(str(item) for item in extra if str(item).strip()),
        surface=surface,
        write_root=str(value.get("write_root") or "").strip(),
        base_sha=str(value.get("base_sha") or "").strip(),
        protected_paths_grant=_coerce_bool(value.get("protected_paths_grant", False), default=False),
        external_tool_grants=tuple(str(g).strip() for g in grants_raw if str(g).strip()),
        parent_only_commit=True,
        return_kind=str(value.get("return_kind") or "workspace_patch").strip() or "workspace_patch",
    )


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return default


def resolve_payload_path(drive_root: Path, constraint: TaskConstraint, path_text: str) -> Path:
    from ouroboros.contracts.skill_payload_policy import resolve_constrained_payload_path

    return resolve_constrained_payload_path(drive_root, constraint, path_text)
