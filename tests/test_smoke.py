[...lines 1-88...]
EXPECTED_TOOLS = [
    "repo_read", "repo_write_commit", "repo_list", "repo_commit_push",
    "drive_read", "drive_write", "drive_list",
    "git_status", "git_diff",
    "run_shell", "claude_code_edit",
    "browse_page", "browser_action",
    "web_search", "search_agent",
    "chat_history", "update_scratchpad", "update_identity",
    "request_restart", "promote_to_stable", "request_review",
    "schedule_task", "cancel_task",
    "switch_model", "toggle_evolution", "toggle_consciousness",
    "send_owner_message", "send_photo",
    "codebase_digest", "codebase_health",
    "knowledge_read", "knowledge_write", "knowledge_list",
    "multi_model_review",
    # GitHub Issues
    "list_github_issues", "get_github_issue", "comment_on_issue",
    "close_github_issue", "create_github_issue",
    "summarize_dialogue",
    # Task decomposition
    "get_task_result", "wait_for_task",
    "generate_evolution_stats",
    # VLM / Vision
    "analyze_screenshot", "vlm_query",
    # Message routing
    "forward_to_worker",
    # Context management
    "compact_context",
    "list_available_tools",
    "enable_tools",
]
[...rest of file unchanged...]