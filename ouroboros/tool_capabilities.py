"""Single source of truth for tool visibility, parallelism, and result limits."""

from __future__ import annotations

CORE_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "write_file", "edit_text",
    "search_code", "query_code", "plan_task",
    "run_command", "claude_code_edit", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "vcs_status", "vcs_diff", "vcs_commit_reviewed", "commit_reviewed",
    "vcs_restore", "vcs_revert", "vcs_pull_ff", "vcs_rollback",
    "schedule_subagent", "integrate_subagent_patch", "compare_subagent_patches",
    "wait_task", "wait_tasks", "get_task_result",
    # D#7 soft-join child controls (siblings of steer_task): inspect/decide a child's fate
    # before finalizing (peek = pure read, discard = explicit abandon, cancel = real stop).
    "cancel_task", "peek_task", "discard_child_result",
    # Task-tree coordination must be in the round-one envelope so a parent can publish the
    # shared frame BEFORE fanning out interdependent children (no enable_tools detour).
    "tree_note", "tree_read",
    # Main-chat routing capabilities the SYSTEM.md decision turn relies on
    # (kept in the core envelope so the anti-freeze ephemeral turn never needs an
    # enable_tools detour to route — though initial_tool_schemas exposes the full
    # set today, this makes the coupling explicit).
    "list_projects", "route_to_project", "promote_chat_to_task", "steer_task",
    "ensure_project_scope",
    "update_scratchpad", "update_identity",
    "chat_history", "recent_tasks",
    "knowledge_read", "knowledge_write", "knowledge_list",
    "web_search",
    "browse_page", "browser_action", "analyze_screenshot", "view_image",
    "send_user_message", "send_photo", "send_video",
    "switch_model",
    "request_restart", "promote_to_stable",
    "advisory_review", "review_status", "task_acceptance_review",
    # Heal mode blocks enable_tools, so repair/review tools must be core.
    "list_skills", "skill_review", "skill_preflight",
    "submit_skill_to_hub",
})

# Meta-tools: always visible alongside core tools
META_TOOL_NAMES: frozenset[str] = frozenset({
    "list_available_tools", "enable_tools",
})

LOCAL_READONLY_SUBAGENT_MODE: str = "local_readonly_subagent"

# V1 subagents are read-only against local Ouroboros state. Browser interaction
# remains available by explicit product decision, so this mode is not a remote
# website sandbox.
LOCAL_READONLY_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "chat_history", "recent_tasks", "get_task_result", "wait_task", "wait_tasks",
    "schedule_subagent",
    # Task-tree coordination: a child reads the shared frame and raises beacons. tree_note
    # is a bounded local coordination write (no repo/control-plane mutation), so it is
    # allowed even for read-only subagents — same class as emitting progress.
    "tree_note", "tree_read",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
})

ACTING_SUBAGENT_MODE: str = "acting_subagent"

# Mutative ("acting") subagents may write inside an isolated write root
# (self_worktree / external_workspace) and run shell/services there.
# They explicitly CANNOT commit the live body (commit_reviewed /
# vcs_commit_reviewed), run runtime control, touch the skills lifecycle, enable
# tools, or write cognitive memory (update_identity/update_scratchpad/
# knowledge_write). The parent integrates and is the sole committer. Extension /
# MCP tools are denied unless explicitly granted per-child via
# TaskConstraint.external_tool_grants.
ACTING_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "write_file", "edit_text",
    "run_command", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "integrate_subagent_patch", "compare_subagent_patches",
    "schedule_subagent", "wait_task", "wait_tasks", "get_task_result",
    "knowledge_read", "knowledge_list",
    "tree_note", "tree_read",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
    "list_available_tools",
})

READ_ONLY_PARALLEL_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_files",
    "search_code", "query_code", "recent_tasks",
    "web_search", "chat_history",
    "vcs_status", "vcs_diff", "service_status", "service_logs",
    "get_task_result", "list_projects",
})

# Enqueue-only tools safe to emit in parallel within one tool-call round.
# schedule_subagent is fire-and-forget: it writes a `requested` task result and
# does event_queue.put_nowait(...) with no blocking LLM/RPC on the parent path.
# Parent-side shared ctx state touched during emission is guarded by
# _SCHEDULE_EMIT_LOCK in tools/control.py; the supervisor still drains EVENT_Q
# serially, so cap/dedup/enqueue remain single-threaded and safe.
PARALLEL_SAFE_ENQUEUE_TOOLS: frozenset[str] = frozenset({"schedule_subagent"})

# Stateful browser tools need the thread-sticky executor.
STATEFUL_BROWSER_TOOLS: frozenset[str] = frozenset({
    "browse_page", "browser_action",
})

# Full outputs are semantic (review verdicts, advisory findings, status).
UNTRUNCATED_TOOL_RESULTS: frozenset[str] = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
    "plan_task",
    "task_acceptance_review",
    "advisory_review",
    "skill_review",
    "review_status",
    "get_task_result",
    "wait_task",
    "wait_tasks",
})

# Cognitive artifacts must not be truncated.
UNTRUNCATED_REPO_READ_PATHS: frozenset[str] = frozenset({
    "BIBLE.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CHECKLISTS.md",
    "docs/DEVELOPMENT.md",
})

# Per-tool char caps; omitted tools use DEFAULT_TOOL_RESULT_LIMIT.
TOOL_RESULT_LIMITS: dict[str, int] = {
    "read_file": 80_000,
    "recent_tasks": 80_000,
    "knowledge_read": 80_000,
    "claude_code_edit": 80_000,
    "run_command": 80_000,
    "run_script": 80_000,
    "search_code": 80_000,
    "query_code": 80_000,
    "service_logs": 80_000,
    # Best-of-N patch comparison shows several candidate diffs side by side; the
    # default 15k cap would truncate after the first one and defeat the tool.
    "compare_subagent_patches": 80_000,
    # skill_exec wraps stdout/stderr; keep the full capped payload visible.
    "skill_exec": 300_000,
    # tree_read returns the shared task-tree coordination tail (up to 200 entries); the 15k
    # default would truncate the swarm blackboard and defeat the coordination contract.
    "tree_read": 80_000,
}

DEFAULT_TOOL_RESULT_LIMIT: int = 15_000

# Reviewed mutative tools must not end with ambiguous executor timeouts.
REVIEWED_MUTATIVE_TOOLS: frozenset[str] = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
})

# Foreground mutative tools may keep editing files after Python future timeout;
# the loop must wait for terminal completion instead of returning while they run.
FOREGROUND_MUTATIVE_TOOLS: frozenset[str] = frozenset({
    "claude_code_edit",
})
