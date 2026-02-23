"""
Ouroboros — Tool registry (SSOT).

Plugin architecture: each module in tools/ exports get_tools().
ToolRegistry collects all tools, provides schemas() and execute().
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import safe_relpath


@dataclass
class BrowserState:
    """Per-task browser lifecycle state (Playwright). Isolated from generic ToolContext."""

    pw_instance: Any = None
    browser: Any = None
    page: Any = None
    last_screenshot_b64: Optional[str] = None


@dataclass
class ToolContext:
    """Tool execution context — passed from the agent before each task."""

    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = "ouroboros"
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    current_chat_id: Optional[int] = None
    current_task_type: Optional[str] = None
    last_push_succeeded: bool = False
    emit_progress_fn: Callable[[str], None] = field(default=lambda _: None)

    # LLM-driven model/effort switch (set by switch_model tool, read by loop.py)
    active_model_override: Optional[str] = None
    active_effort_override: Optional[str] = None

    # Per-task browser state
    browser_state: BrowserState = field(default_factory=BrowserState)

    # Budget tracking (set by loop.py for real-time usage events)
    event_queue: Optional[Any] = None
    task_id: Optional[str] = None

    # Task depth for fork bomb protection
    task_depth: int = 0

    # True when running inside handle_chat_direct (not a queued worker task)
    is_direct_chat: bool = False

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()

    def drive_logs(self) -> pathlib.Path:
        return (self.drive_root / "logs").resolve()


@dataclass
class ToolEntry:
    """Single tool descriptor: name, schema, handler, metadata."""

    name: str
    schema: Dict[str, Any]
    handler: Callable  # fn(ctx: ToolContext, **args) -> str
    is_code_tool: bool = False
    timeout_sec: int = 120


CORE_TOOL_NAMES = {
    "repo_read", "repo_list", "repo_write_commit", "repo_commit_push",
    "drive_read", "drive_list", "drive_write",
    "run_shell", "claude_code_edit",
    "git_status", "git_diff",
    "schedule_task", "wait_for_task", "get_task_result",
    "update_scratchpad", "update_identity",
    "chat_history", "web_search",
    "send_owner_message", "switch_model",
    "request_restart", "promote_to_stable",
    "knowledge_read", "knowledge_write",
    "browse_page", "browser_action", "analyze_screenshot",
}


class ToolRegistry:
    """Ouroboros tool registry (SSOT).

    To add a tool: create a module in ouroboros/tools/,
    export get_tools() -> List[ToolEntry].
    """

    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self._entries: Dict[str, ToolEntry] = {}
        self._ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
        self._load_modules()

    def _load_modules(self) -> None:
        """Auto-discover tool modules in ouroboros/tools/ that export get_tools()."""
        import importlib
        import pkgutil
        import ouroboros.tools as tools_pkg
        for _importer, modname, _ispkg in pkgutil.iter_modules(tools_pkg.__path__):
            if modname.startswith("_") or modname == "registry":
                continue
            try:
                mod = importlib.import_module(f"ouroboros.tools.{modname}")
                if hasattr(mod, "get_tools"):
                    for entry in mod.get_tools():
                        self._entries[entry.name] = entry
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load tool module %s", modname, exc_info=True)

    def set_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    def register(self, entry: ToolEntry) -> None:
        """Register a new tool (for extension by Ouroboros)."""
        self._entries[entry.name] = entry

    # --- Contract ---

    def available_tools(self) -> List[str]:
        return [e.name for e in self._entries.values()]

    def schemas(self, core_only: bool = False) -> List[Dict[str, Any]]:
        if not core_only:
            return [{"type": "function", "function": e.schema} for e in self._entries.values()]
        # Core tools + meta-tools for discovering/enabling extended tools
        result = []
        for e in self._entries.values():
            if e.name in CORE_TOOL_NAMES or e.name in ("list_available_tools", "enable_tools"):
                result.append({"type": "function", "function": e.schema})
        return result

    def list_non_core_tools(self) -> List[Dict[str, str]]:
        """Return name+description of all non-core tools."""
        result = []
        for e in self._entries.values():
            if e.name not in CORE_TOOL_NAMES:
                desc = e.schema.get("description", "No description")
                result.append({"name": e.name, "description": desc})
        return result

    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the full schema for a specific tool."""
        entry = self._entries.get(name)
        if entry:
            return {"type": "function", "function": entry.schema}
        return None

    def get_timeout(self, name: str) -> int:
        """Return timeout_sec for the named tool (default 120)."""
        entry = self._entries.get(name)
        return entry.timeout_sec if entry is not None else 120

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        entry = self._entries.get(name)
        if entry is None:
            return f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(self._entries.keys()))}"
        try:
            return entry.handler(self._ctx, **args)
        except TypeError as e:
            return f"⚠️ TOOL_ARG_ERROR ({name}): {e}"
        except Exception as e:
            return f"⚠️ TOOL_ERROR ({name}): {e}"

    def override_handler(self, name: str, handler) -> None:
        """Override the handler for a registered tool (used for closure injection)."""
        entry = self._entries.get(name)
        if entry:
            self._entries[name] = ToolEntry(
                name=entry.name,
                schema=entry.schema,
                handler=handler,
                timeout_sec=entry.timeout_sec,
            )

    @property
    def CODE_TOOLS(self) -> frozenset:
        return frozenset(e.name for e in self._entries.values() if e.is_code_tool)
