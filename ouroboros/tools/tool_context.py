"""Concrete per-task context shared by tool handlers and the registry facade."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.artifacts import task_id_for_artifacts
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import normalize_root_relative, workspace_mode_block_reason
from ouroboros.utils import safe_relpath


@dataclass
class BrowserState:
    """Per-task Playwright lifecycle state."""

    pw_instance: Any = None
    browser: Any = None
    page: Any = None
    last_screenshot_b64: Optional[str] = None


@dataclass
class ToolContext:
    """Tool execution context passed from the agent."""

    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = "ouroboros"
    system_repo_dir: Optional[pathlib.Path] = None
    workspace_root: Optional[pathlib.Path] = None
    workspace_mode: str = ""
    memory_mode: str = ""
    budget_drive_root: str = ""
    # Per-project facts scope (Phase 3b): when set, knowledge reads/writes target
    # the per-project store under the canonical data dir instead of memory/knowledge.
    project_id: str = ""
    task_metadata: Dict[str, Any] = field(default_factory=dict)
    executor_ref: Dict[str, Any] = field(default_factory=dict)
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    current_chat_id: Optional[int] = None
    current_task_type: Optional[str] = None
    pending_restart_reason: Optional[str] = None
    last_push_succeeded: bool = False
    last_reviewed_commit_sha: str = ""
    emit_progress_fn: Callable[[str], None] = field(default=lambda _: None)

    # LLM-driven model/effort switch.
    active_model_override: Optional[str] = None
    active_effort_override: Optional[str] = None
    active_use_local_override: Optional[bool] = None
    task_model_override: Optional[str] = None
    task_use_local_override: Optional[bool] = None
    # CW2 (v6.34.0): the loop publishes the effective context mode each round so
    # switch_model can refuse switching to a sub-1M route while the transcript is max-sized.
    active_context_mode: str = ""

    # Per-task browser state.
    browser_state: BrowserState = field(default_factory=BrowserState)

    # Budget tracking for usage events.
    event_queue: Optional[Any] = None
    task_id: Optional[str] = None

    # Conversation messages for safety checks.
    messages: Optional[List[Dict[str, Any]]] = None

    # Structured task constraints, e.g. skill repair payload confinement.
    task_constraint: Optional[TaskConstraint] = None
    task_contract: Dict[str, Any] = field(default_factory=dict)

    # Task depth for fork-bomb protection.
    task_depth: int = 0

    # True inside handle_chat_direct, not a queued worker task.
    is_direct_chat: bool = False
    # CW3 (v6.34.0): a SHORT-LIVED same-route "decision" turn (run while the chat
    # agent is busy). It may answer / route / spawn / steer, but is barred from
    # durable cognitive-memory / evolution / settings / control-plane mutators
    # (the WS10 ephemeral contract) — enforced in schemas()/execute().
    is_ephemeral_turn: bool = False

    # Pre-commit review state.
    _review_advisory: List[Any] = field(default_factory=list)
    _review_iteration_count: int = 0
    _review_history: list = field(default_factory=list)

    def active_repo_dir(self) -> pathlib.Path:
        if self.is_workspace_mode():
            return pathlib.Path(self.workspace_root)
        return pathlib.Path(self.repo_dir)

    def is_workspace_mode(self) -> bool:
        return (
            self.workspace_root is not None
            and bool(str(self.workspace_mode or "").strip())
            and not workspace_mode_block_reason(self)
        )

    def repo_path(self, rel: str) -> pathlib.Path:
        root = self.active_repo_dir()
        # Accept the paths an agent naturally writes against a workspace root:
        # an absolute path already INSIDE the root (e.g. /app/out.txt under a
        # workspace rooted at /app — otherwise re-nested as /app/app/out.txt) and
        # a redundant root-basename prefix ('app/out.txt'). normalize_root_relative
        # only ever returns a relative string; paths not under the root fall
        # through to safe_relpath (kept inside) and the boundary check below.
        rel_str = normalize_root_relative(root, str(rel))
        resolved = (root / safe_relpath(rel_str)).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"Path escapes repo_dir boundary: {rel}")
        return resolved

    def drive_path(self, rel: str) -> pathlib.Path:
        resolved = (self.drive_root / safe_relpath(rel)).resolve()
        try:
            resolved.relative_to(self.drive_root.resolve())
        except ValueError:
            raise ValueError(f"Path escapes drive_root boundary: {rel}")
        return resolved

    def drive_logs(self) -> pathlib.Path:
        return (self.drive_root / "logs").resolve()

    def task_drive_root(self) -> pathlib.Path:
        return (pathlib.Path(self.drive_root).resolve(strict=False) / "task_drives" / task_id_for_artifacts(self)).resolve(strict=False)

    def workspace_executor_ref(self) -> Dict[str, Any]:
        if isinstance(self.executor_ref, dict) and self.executor_ref:
            return dict(self.executor_ref)
        if isinstance(self.task_metadata, dict) and isinstance(self.task_metadata.get("executor_ref"), dict):
            return dict(self.task_metadata["executor_ref"])
        return {}


__all__ = ["BrowserState", "ToolContext"]
