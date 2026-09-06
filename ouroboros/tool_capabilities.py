"""Single source of truth for tool visibility, parallelism, and result limits."""

from __future__ import annotations

# The delegation_role value background consciousness stamps on its shared tool
# context before every tool call. Owner-delivery gating keys on it, so both
# sides import this one name instead of repeating the literal.
BACKGROUND_DELEGATION_ROLE: str = "background"

OWNER_DELIVERY_TOOL_NAMES: frozenset[str] = frozenset({
    "send_user_message", "send_photo", "send_video", "send_file", "send_links",
})

CORE_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "write_file", "edit_text",
    "apply_patch", "edit_batch",
    "search_code", "query_code", "plan_task",
    "run_command", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "vcs_status", "vcs_diff", "vcs_commit_reviewed", "commit_reviewed",
    "vcs_restore", "vcs_revert", "vcs_pull_ff", "vcs_rollback",
    # One-shot deferred follow-up (W=A): core so a root task facing a typed wait
    # instant (e.g. a structurally unreachable review quorum with a known reset)
    # can register it without an enable_tools detour. Deliberately absent from
    # the subagent profiles below — a child may not mint future root tasks.
    "schedule_followup",
    "schedule_subagent", "integrate_subagent_patch", "compare_subagent_patches",
    "integrate_delegated_patch",
    "wait_task", "wait_tasks", "get_task_result",
    # D#7 soft-join child controls (siblings of steer_task): inspect/decide a child's fate
    # before finalizing (peek = pure read, discard = explicit abandon, cancel = real stop).
    "cancel_task", "peek_task", "discard_child_result", "override_delegation_constraint",
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
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
    *OWNER_DELIVERY_TOOL_NAMES,
    "escalate",
    "switch_model",
    "request_restart", "promote_to_stable",
    "preflight_review", "advisory_review", "review_status", "task_acceptance_review", "verify_and_record",
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
    # switch_model changes COGNITIVE POWER, not authority: a child that started on
    # the cheap lane and finds the work harder raises itself instead of failing or
    # asking the parent to respawn it (BIBLE P5). Nothing about the sandbox changes.
    "switch_model",
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "knowledge_read", "knowledge_list",
    "chat_history", "recent_tasks", "get_task_result", "wait_task", "wait_tasks",
    "escalate",
    "forward_to_worker", "peek_task", "cancel_task", "discard_child_result",
    "schedule_subagent",
    # Task-tree coordination: a child reads the shared frame and raises beacons. tree_note
    # is a bounded tree-scoped write; its tagged child-result disposition branch also
    # updates the existing child result through join_ledger's lineage/hash authority.
    # It has no repo/control-plane effect, so remains valid for read-only subagents.
    "tree_note", "tree_read", "override_delegation_constraint",
    # Nanny verbs. The child gets no shell — it gets the right to ASK the host to run a
    # session, and the host derives the access profile from THIS task's authority, so a
    # read-only child can only ever host a read-only session. delegate_answer speaks
    # only to a run this task already owns (custody-gated like cancel).
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
    # Bounded media projection: writes derived frames only under artifact_store/video_frames.
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
})

ACTING_SUBAGENT_MODE: str = "acting_subagent"

# Mutative ("acting") subagents may write inside their assigned write root
# (isolated self_worktree / shared external_workspace) and run shell/services there.
# They explicitly CANNOT commit the live body (commit_reviewed /
# vcs_commit_reviewed), run runtime control, touch the skills lifecycle, enable
# tools, or write cognitive memory (update_identity/update_scratchpad/
# knowledge_write). The parent integrates and is the sole committer. Extension /
# MCP tools are denied unless explicitly granted per-child via
# TaskConstraint.external_tool_grants.
ACTING_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({
    # switch_model changes COGNITIVE POWER, not authority: a child that started on
    # the cheap lane and finds the work harder raises itself instead of failing or
    # asking the parent to respawn it (BIBLE P5). Nothing about the sandbox changes.
    "switch_model",
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "write_file", "edit_text",
    "apply_patch", "edit_batch",
    "run_command", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "integrate_subagent_patch", "compare_subagent_patches",
    "schedule_subagent", "wait_task", "wait_tasks", "get_task_result",
    "escalate",
    "forward_to_worker", "peek_task", "cancel_task", "discard_child_result",
    "verify_and_record",
    "knowledge_read", "knowledge_list",
    "tree_note", "tree_read", "override_delegation_constraint",
    # Same nanny verbs, same host-derived profile — an acting child hosts a
    # workspace_write session confined to a private snapshot of its own write
    # root, and explicitly integrates the captured diff (C1).
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "integrate_delegated_patch",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
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
    "preflight_review",
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
    # apply_patch results carry per-hunk diagnostics, edit_batch per-edit ones
    # (an aborted batch reports EVERY failed edit so one retry can fix them all);
    # write_file appends the overwrite diff.
    "apply_patch": 80_000,
    "edit_batch": 80_000,
    "write_file": 80_000,
}

DEFAULT_TOOL_RESULT_LIMIT: int = 15_000


def tool_result_limit(tool_name: str) -> int:
    """The char budget a tool's result is delivered under.

    Read by the truncator AND by producers that must fit inside it: a tool whose payload
    is structured JSON has to bound itself, because outer head-truncation cuts mid-string
    and destroys the document. Both sides asking the same function is what keeps a
    producer's idea of "small enough" from drifting away from the cap actually applied.
    """
    return TOOL_RESULT_LIMITS.get(str(tool_name or ""), DEFAULT_TOOL_RESULT_LIMIT)


# Reviewed mutative tools must not end with ambiguous executor timeouts.
REVIEWED_MUTATIVE_TOOLS: frozenset[str] = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
})

# Foreground mutative tools may keep editing files after Python future timeout;
# the loop must wait for terminal completion instead of returning while they run.
# D10 retired the SDK edit gateway; publication is now the foreground mutator
# whose remote branch/commit/PR effects must settle before control returns.
FOREGROUND_MUTATIVE_TOOLS: frozenset[str] = frozenset({
    "submit_skill_to_hub",
})
