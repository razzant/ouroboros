"""Concrete per-task context shared by tool handlers and the registry facade.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
"""

from __future__ import annotations

import pathlib

from dataclasses import dataclass
from dataclasses import field

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.llm_claudexor import ModelTurnState
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import List
    from typing import Optional


def _registry():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import registry

    return registry


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
    # Set only while a body candidate is bound (``body_candidate.bind``): the checkout
    # the server actually runs, when ``repo_dir``/``system_repo_dir`` name the candidate.
    serving_repo_dir: Optional[pathlib.Path] = None
    workspace_root: Optional[pathlib.Path] = None
    workspace_mode: str = ""
    memory_mode: str = ""
    budget_drive_root: str = ""
    # Per-project facts scope (Phase 3b): when set, knowledge reads/writes target
    # the per-project store under the canonical data dir instead of memory/knowledge.
    project_id: str = ""
    task_metadata: Dict[str, Any] = field(default_factory=dict)
    # The latest owner message this turn actually DRAINED from its mailbox
    # (``{"msg_id", "client_message_id", "text", "ts"}``), stamped by the loop's
    # drain seam; ``None`` while the turn still acts only on the message that
    # started it.
    last_owner_delivery: Optional[Dict[str, Any]] = None
    executor_ref: Dict[str, Any] = field(default_factory=dict)
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    current_chat_id: Optional[int] = None
    current_task_type: Optional[str] = None
    pending_restart_reason: Optional[str] = None
    last_push_succeeded: bool = False
    last_reviewed_commit_sha: str = ""
    # The real binder accepts keyword facts (``narration=True`` for a tool that relays
    # the model's own words); an unbound context must swallow them the same way.
    emit_progress_fn: Callable[[str], None] = field(default=lambda _text, **_kw: None)

    # LLM-driven model/effort switch.
    active_model_override: Optional[str] = None
    active_effort_override: Optional[str] = None
    active_use_local_override: Optional[bool] = None
    task_model_override: Optional[str] = None
    task_use_local_override: Optional[bool] = None
    # CW2 (v6.34.0): the loop publishes the effective context mode each round so
    # switch_model can refuse switching to a sub-1M route while the transcript is max-sized.
    active_context_mode: str = ""

    # The active-turn transport slot every model call of ONE loop invocation
    # shares (ordinary rounds, fallback candidates and forced finalization).
    # `run_llm_loop` mints a fresh empty one at entry, so a next loop — and a
    # cold restart — begins a new turn even where the transcript is identical.
    # Opaque transport only: mechanism in `llm_claudexor.ModelTurnState`.
    model_turn_state: Optional[ModelTurnState] = None

    # Per-task browser state.
    browser_state: BrowserState = field(default_factory=BrowserState)

    # Budget tracking for usage events.
    event_queue: Optional[Any] = None
    task_id: Optional[str] = None
    # The agent binds lifecycle authority; standalone tool invocations only
    # carry attribution. Missing managed authority must never become standalone.
    task_lifecycle_bound: bool = False

    # Conversation messages for safety checks.
    messages: Optional[List[Dict[str, Any]]] = None

    # Borrowed loop trace; restored when the owning loop exits.
    _execution_trace: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # Structured task constraints, e.g. skill repair payload confinement.
    task_constraint: Optional[TaskConstraint] = None
    task_contract: Dict[str, Any] = field(default_factory=dict)

    # Task depth for fork-bomb protection.
    task_depth: int = 0

    # True inside handle_chat_direct, not a queued worker task.
    is_direct_chat: bool = False
    # Pre-commit review state.
    _review_advisory: List[Any] = field(default_factory=list)
    _review_iteration_count: int = 0
    _review_history: list = field(default_factory=list)

    def active_repo_dir(self) -> pathlib.Path:
        if self.is_workspace_mode():
            return pathlib.Path(self.workspace_root)
        from ouroboros.tool_access import project_room_lens_dir

        room = project_room_lens_dir(self)
        if room is not None:
            return room
        from ouroboros.tool_access import folderless_scratch_dir

        return folderless_scratch_dir(self) or pathlib.Path(self.repo_dir)

    def is_workspace_mode(self) -> bool:
        return (
            self.workspace_root is not None
            and bool(str(self.workspace_mode or "").strip())
            and not _registry().workspace_mode_block_reason(self)
        )

    def repo_path(self, rel: str) -> pathlib.Path:
        from ouroboros.tool_access import _resolve_target_in_selected_base

        return _resolve_target_in_selected_base(
            self, root="active_workspace", base_path=self.active_repo_dir(),
            path=str(rel), operation="read",
        )

    def drive_path(self, rel: str) -> pathlib.Path:
        from ouroboros.tool_access import _resolve_target_in_selected_base

        return _resolve_target_in_selected_base(
            self, root="runtime_data", base_path=self.drive_root,
            path=str(rel), operation="read",
        )

    def drive_logs(self) -> pathlib.Path:
        return (self.drive_root / "logs").resolve()

    def task_drive_root(self) -> pathlib.Path:
        return (pathlib.Path(self.drive_root).resolve(strict=False) / "task_drives" / _registry().task_id_for_artifacts(self)).resolve(strict=False)

    def workspace_executor_ref(self) -> Dict[str, Any]:
        if isinstance(self.executor_ref, dict) and self.executor_ref:
            return dict(self.executor_ref)
        if isinstance(self.task_metadata, dict) and isinstance(self.task_metadata.get("executor_ref"), dict):
            return dict(self.task_metadata["executor_ref"])
        return {}
