"""Migration-ledger membership: every extraction the v7 branch performed is a row.

Split out of tests/test_v7_prologue_evidence.py so that module stays inside the size
ratchet as the ledger grows stream by stream. The assertions are unchanged.
"""

from __future__ import annotations

import importlib.util
import pathlib
import tests._v7_ledger_inventories as _inv

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "v7_evidence.py"
SPEC = importlib.util.spec_from_file_location("v7_evidence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
v7_evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v7_evidence)
v7_migration = v7_evidence._migration


def test_migration_table_is_valid_and_uses_only_spec_approved_pending_owners():
    assert v7_evidence.validate_migration(REPO) == []
    rows = v7_evidence._parse_migration(REPO / "MIGRATION_v7.md")
    assert len({row["old path/symbol"] for row in rows}) == len(rows)
    implemented = {
        "ouroboros/tools/registry.py::BrowserState": "ouroboros/tools/tool_context.py::BrowserState",
        "ouroboros/tools/registry.py::ToolContext": "ouroboros/tools/tool_context.py::ToolContext",
        "ouroboros/tools/registry.py::ToolEntry": "ouroboros/tools/tool_catalog.py::ToolEntry",
        "ouroboros/tools/registry.py::_compose_execute_result":
            "ouroboros/tools/tool_result.py::_compose_execute_result",
        "ouroboros/tools/registry.py::_coerce_real_path":
            "ouroboros/tools/tool_resolution.py::_coerce_real_path",
        "ouroboros/tools/registry.py::active_repo_dir_for":
            "ouroboros/tools/tool_resolution.py::active_repo_dir_for",
        "ouroboros/tools/registry.py::system_repo_dir_for":
            "ouroboros/tools/tool_resolution.py::system_repo_dir_for",
        "ouroboros/tools/registry.py::_PATH_NORMALIZED_TOOLS":
            "ouroboros/tools/tool_resolution.py::_PATH_NORMALIZED_TOOLS",
        "ouroboros/tools/registry.py::_normalize_dispatch_path_args":
            "ouroboros/tools/tool_resolution.py::_normalize_dispatch_path_args",
        "ouroboros/tools/registry.py::_GENERIC_VCS_TARGET_TOOLS":
            "ouroboros/tools/tool_resolution.py::_GENERIC_VCS_TARGET_TOOLS",
        "ouroboros/tools/registry.py::_TARGET_BINDING_OPERATIONS":
            "ouroboros/tools/tool_resolution.py::_TARGET_BINDING_OPERATIONS",
        "ouroboros/tools/registry.py::_SKILL_LIFECYCLE_TARGET_TOOLS":
            "ouroboros/tools/tool_resolution.py::_SKILL_LIFECYCLE_TARGET_TOOLS",
        "ouroboros/tools/registry.py::_PROCESS_TARGET_TOOLS":
            "ouroboros/tools/tool_resolution.py::_PROCESS_TARGET_TOOLS",
        "ouroboros/tools/registry.py::_VERIFY_RUN_KINDS":
            "ouroboros/tools/tool_resolution.py::_VERIFY_RUN_KINDS",
        "ouroboros/tools/registry.py::_target_binding_operation":
            "ouroboros/tools/tool_resolution.py::_target_binding_operation",
        "ouroboros/tools/registry.py::_build_builtin_target_binding":
            "ouroboros/tools/tool_resolution.py::_build_builtin_target_binding",
        "ouroboros/tools/registry.py::_binding_items":
            "ouroboros/tools/tool_resolution.py::_binding_items",
        "ouroboros/tools/registry.py::_binding_set_targets_system_repo":
            "ouroboros/tools/tool_resolution.py::_binding_set_targets_system_repo",
        "ouroboros/tools/registry.py::_binding_set_is_light_restricted":
            "ouroboros/tools/tool_resolution.py::_binding_set_is_light_restricted",
        "ouroboros/tools/registry.py::_binding_state_drive_root":
            "ouroboros/tools/tool_resolution.py::_binding_state_drive_root",
        "ouroboros/tools/registry.py::_detect_runtime_mode_elevation":
            "ouroboros/tools/registry_guard_process.py::_detect_runtime_mode_elevation",
        "ouroboros/tools/registry.py::_SUBAGENT_SHELL_SECRET_MARKERS":
            "ouroboros/tools/registry_guard_process.py::_SUBAGENT_SHELL_SECRET_MARKERS",
        "ouroboros/tools/registry.py::_subagent_shell_targets_secret":
            "ouroboros/tools/registry_guard_process.py::_subagent_shell_targets_secret",
        "ouroboros/tools/registry.py::_detect_mutative_toggle_self_change":
            "ouroboros/tools/registry_guard_process.py::_detect_mutative_toggle_self_change",
        "ouroboros/tools/registry.py::_detect_evolution_owner_control_self_change":
            "ouroboros/tools/registry_guard_process.py::_detect_evolution_owner_control_self_change",
        "ouroboros/tools/registry.py::_detect_context_mode_self_lowering":
            "ouroboros/tools/registry_guard_process.py::_detect_context_mode_self_lowering",
        "ouroboros/tools/registry.py::_READ_ONLY_INSPECTION_COMMANDS":
            "ouroboros/tools/registry_guard_process.py::_READ_ONLY_INSPECTION_COMMANDS",
        "ouroboros/tools/registry.py::_COMMAND_HEAD_WRAPPERS":
            "ouroboros/tools/registry_guard_process.py::_COMMAND_HEAD_WRAPPERS",
        "ouroboros/tools/registry.py::_READ_ONLY_GIT_SUBCOMMANDS":
            "ouroboros/tools/registry_guard_process.py::_READ_ONLY_GIT_SUBCOMMANDS",
        "ouroboros/tools/registry.py::_SEARCH_TOOL_EXEC_OPTIONS":
            "ouroboros/tools/registry_guard_process.py::_SEARCH_TOOL_EXEC_OPTIONS",
        "ouroboros/tools/registry.py::_DENIED_READ_OPTIONS":
            "ouroboros/tools/registry_guard_process.py::_DENIED_READ_OPTIONS",
        "ouroboros/tools/registry.py::_TRUSTED_EXECUTABLE_DIRS":
            "ouroboros/tools/registry_guard_process.py::_TRUSTED_EXECUTABLE_DIRS",
        "ouroboros/tools/registry.py::_trusted_read_head":
            "ouroboros/tools/registry_guard_process.py::_trusted_read_head",
        "ouroboros/tools/registry.py::_denied_read_option":
            "ouroboros/tools/registry_guard_process.py::_denied_read_option",
        "ouroboros/tools/registry.py::_NESTED_EXECUTION_MARKERS":
            "ouroboros/tools/registry_guard_process.py::_NESTED_EXECUTION_MARKERS",
        "ouroboros/tools/registry.py::_NESTED_EXECUTION_TOKENS":
            "ouroboros/tools/registry_guard_process.py::_NESTED_EXECUTION_TOKENS",
        "ouroboros/tools/registry.py::_is_pure_read_inspection":
            "ouroboros/tools/registry_guard_process.py::_is_pure_read_inspection",
        "ouroboros/tools/registry.py::_detect_scope_review_floor_self_lowering":
            "ouroboros/tools/registry_guard_process.py::_detect_scope_review_floor_self_lowering",
        "ouroboros/tools/registry.py::_detect_safety_mode_self_lowering":
            "ouroboros/tools/registry_guard_process.py::_detect_safety_mode_self_lowering",
        "ouroboros/tools/registry.py::_detect_owner_skill_attest_self_call":
            "ouroboros/tools/registry_guard_process.py::_detect_owner_skill_attest_self_call",
        "ouroboros/tools/registry.py::_SKILL_OWNER_STATE_STEMS":
            "ouroboros/tools/registry_guard_process.py::_SKILL_OWNER_STATE_STEMS",
        "ouroboros/tools/registry.py::_DETACHED_PROCESS_MARKERS":
            "ouroboros/tools/registry_guard_process.py::_DETACHED_PROCESS_MARKERS",
        "ouroboros/tools/registry.py::_mentions_skill_owner_state":
            "ouroboros/tools/registry_guard_process.py::_mentions_skill_owner_state",
        "ouroboros/tools/registry.py::_mentions_detached_process":
            "ouroboros/tools/registry_guard_process.py::_mentions_detached_process",
        "ouroboros/tools/registry.py::ToolRegistry._run_shell_safety_check":
            "ouroboros/tools/registry_guard_process.py::_run_shell_safety_check",
        "ouroboros/tools/registry.py::_light_repo_snapshot":
            "ouroboros/tools/registry_guard_process.py::_light_repo_snapshot",
        "ouroboros/tools/registry.py::_format_light_repo_write_block":
            "ouroboros/tools/registry_guard_process.py::_format_light_repo_write_block",
        "ouroboros/tools/registry.py::_git_ref_snapshot":
            "ouroboros/tools/registry_guard_process.py::_git_ref_snapshot",
        "ouroboros/tools/registry.py::ToolRegistry._snapshot_owner_files":
            "ouroboros/tools/registry_guard_process.py::_snapshot_owner_files",
        "ouroboros/tools/registry.py::ToolRegistry._restore_owner_files":
            "ouroboros/tools/registry_guard_process.py::_restore_owner_files",
        "ouroboros/tools/registry.py::ToolRegistry._run_shell_post_checks":
            "ouroboros/tools/registry_guard_process.py::_run_shell_post_checks",
        "tests/test_skill_exec.py::test_run_shell_restores_obfuscated_self_authored_state_marker":
            "tests/test_registry_guard_process.py::test_run_shell_restores_obfuscated_self_authored_state_marker",
        "ouroboros/tools/registry.py::SKILL_OWNER_STATE_FILENAMES":
            "ouroboros/contracts/skill_payload_policy.py::SKILL_OWNER_STATE_FILENAMES",
        "ouroboros/tools/registry.py::parse_porcelain_paths":
            "ouroboros/tools/shell_guards.py::parse_porcelain_paths",
        "ouroboros/tools/registry.py::safe_relpath":
            "ouroboros/utils.py::safe_relpath",
        "ouroboros/tools/registry.py::LIGHT_SHELL_WRITER_COMMANDS":
            "ouroboros/tools/shell_guards.py::LIGHT_SHELL_WRITER_COMMANDS",
        "ouroboros/tools/registry.py::SKILL_OWNER_STATE_STEMS":
            "ouroboros/contracts/skill_payload_policy.py::SKILL_OWNER_STATE_STEMS",
        "ouroboros/tools/registry.py::build_resolved_resource_binding":
            "ouroboros/tool_access.py::build_resolved_resource_binding",
        "ouroboros/tools/registry.py::interpreter_family":
            "ouroboros/tools/shell_guards.py::interpreter_family",
        "ouroboros/tools/registry.py::light_shell_repo_mutation":
            "ouroboros/tools/shell_guards.py::light_shell_repo_mutation",
        "ouroboros/tools/registry.py::protected_artifact_shell_block_reason":
            "ouroboros/protected_artifacts.py::shell_block_reason",
        "ouroboros/tools/registry.py::runtime_data_guard_targets":
            "ouroboros/tools/shell_guards.py::runtime_data_guard_targets",
        "ouroboros/tools/registry.py::shell_command_string":
            "ouroboros/shell_parse.py::shell_command_string",
        "ouroboros/tools/registry.py::strip_leading_env_assignments":
            "ouroboros/shell_parse.py::strip_leading_env_assignments",
        "ouroboros/tools/registry.py::sudo_noninteractive_violation":
            "ouroboros/shell_parse.py::sudo_noninteractive_violation",
        "ouroboros/tools/registry.py::unwrap_env_argv":
            "ouroboros/shell_parse.py::unwrap_env_argv",
        "ouroboros/tools/registry.py::workspace_executor_state_write_block":
            "ouroboros/tools/shell_guards.py::workspace_executor_state_write_block",
        "ouroboros/tools/registry.py::writer_target_tokens":
            "ouroboros/tools/shell_guards.py::writer_target_tokens",
        "ouroboros/tools/registry.py::_EPHEMERAL_ALLOWED_TOOLS":
            "ouroboros/tools/registry_guards.py::_EPHEMERAL_ALLOWED_TOOLS",
        "ouroboros/tools/registry.py::_WEB_TOOLS":
            "ouroboros/tools/registry_guards.py::_WEB_TOOLS",
        "ouroboros/tools/registry.py::_resource_allowed":
            "ouroboros/tools/registry_guards.py::_resource_allowed",
        "ouroboros/tools/registry.py::_disabled_tools":
            "ouroboros/tools/registry_guards.py::_disabled_tools",
        "ouroboros/tools/registry.py::_GITHUB_TOKEN_TOOLS":
            "ouroboros/tools/registry_guards.py::_GITHUB_TOKEN_TOOLS",
        "ouroboros/tools/registry.py::_builtin_tool_availability":
            "ouroboros/tools/registry_guards.py::_builtin_tool_availability",
        "ouroboros/tools/registry.py::ToolRegistry._ephemeral_block":
            "ouroboros/tools/registry_guards.py::_ephemeral_block_result",
        "ouroboros/tools/registry.py::ToolRegistry._subagent_and_update_gate":
            "ouroboros/tools/registry_guards.py::_subagent_and_update_guard_result",
        "ouroboros/tools/registry.py::_managed_update_code_tool_block":
            "ouroboros/tools/registry_guards.py::_managed_update_code_tool_block",
        "ouroboros/tools/registry.py::_HEAL_MODE_ALLOWED_TOOLS":
            "ouroboros/tools/registry_guards.py::_HEAL_MODE_ALLOWED_TOOLS",
        "ouroboros/tools/registry.py::_task_constraint_path_allowed":
            "ouroboros/tools/registry_guards.py::_task_constraint_path_allowed",
        "ouroboros/tools/registry.py::_heal_protected_payload_sidecar":
            "ouroboros/tools/registry_guards.py::_heal_protected_payload_sidecar",
        "ouroboros/tools/registry.py::ToolRegistry._heal_mode_block":
            "ouroboros/tools/registry_guards.py::_heal_mode_guard_result",
        "ouroboros/tools/registry.py::_executor_backend_candidate_allowed":
            "ouroboros/tools/registry_guards.py::_executor_backend_candidate_allowed",
        "ouroboros/tools/registry.py::_command_mentions_protected_root":
            "ouroboros/tools/registry_guards.py::_command_mentions_protected_root",
        "ouroboros/tools/registry.py::_authorized_managed_update_resolver":
            "ouroboros/tools/registry_guards.py::_authorized_managed_update_resolver",
        "ouroboros/tools/registry.py::_light_mode_payload_mutation_allowed":
            "ouroboros/tools/registry_guards.py::_light_mode_payload_mutation_allowed",
        "ouroboros/tools/registry.py::ToolRegistry._protected_shell_block":
            "ouroboros/tools/registry_guards.py::_protected_shell_block",
        "ouroboros/tools/registry.py::ToolRegistry._git_protected_roots":
            "ouroboros/tools/registry_guards.py::_git_protected_roots",
        "ouroboros/tools/registry.py::ToolRegistry._resolved_shell_cwd":
            "ouroboros/tools/registry_guards.py::_resolved_shell_cwd",
        "ouroboros/tools/registry.py::ToolRegistry._external_workspace_git_block":
            "ouroboros/tools/registry_guards.py::_external_workspace_git_block",
        "ouroboros/tools/registry.py::ToolRegistry._external_runtime_protected_paths":
            "ouroboros/tools/registry_guards.py::_external_runtime_protected_paths",
        "ouroboros/tools/registry.py::ToolRegistry._external_shell_runtime_or_secret_block":
            "ouroboros/tools/registry_guards.py::_external_shell_runtime_or_secret_block",
        "ouroboros/tools/registry.py::ToolRegistry._workspace_shell_write_block":
            "ouroboros/tools/registry_guards.py::_workspace_shell_write_block",
        "ouroboros/tools/registry.py::ToolRegistry._shell_git_and_runtime_block":
            "ouroboros/tools/registry_guards.py::_shell_git_and_runtime_block",
        "tests/test_external_workspace_access.py::_command_mentions_protected_root":
            "ouroboros/tools/registry_guards.py::_command_mentions_protected_root",
        "ouroboros/tools/registry.py::PROTECTED_RUNTIME_PATHS":
            "ouroboros/runtime_mode_policy.py::PROTECTED_RUNTIME_PATHS",
        "ouroboros/tools/registry.py::task_artifact_dir_path":
            "ouroboros/artifacts.py::task_artifact_dir_path",
        "ouroboros/tools/registry.py::task_id_for_artifacts":
            "ouroboros/artifacts.py::task_id_for_artifacts",
        "ouroboros/tools/registry.py::run_shell_git_block_reason":
            "ouroboros/git_shell_policy.py::run_shell_git_block_reason",
        "ouroboros/tools/registry.py::workspace_git_safety_violation":
            "ouroboros/git_shell_policy.py::workspace_git_safety_violation",
        "ouroboros/tools/registry.py::is_absolute_path_text":
            "ouroboros/shell_parse.py::is_absolute_path_text",
        "ouroboros/tools/registry.py::path_text_is_inside":
            "ouroboros/shell_parse.py::path_text_is_inside",
        "ouroboros/tools/registry.py::shell_argv":
            "ouroboros/shell_parse.py::shell_argv",
        "ouroboros/tools/registry.py::shell_argv_with_path_tokens":
            "ouroboros/shell_parse.py::shell_argv_with_path_tokens",
        "ouroboros/tools/registry.py::PROTECTED_RUNTIME_PATHS_LOWER":
            "ouroboros/tools/shell_guards.py::PROTECTED_RUNTIME_PATHS_LOWER",
        "ouroboros/tools/registry.py::shell_has_write_indicator":
            "ouroboros/tools/shell_guards.py::shell_has_write_indicator",
        "ouroboros/tools/registry.py::shell_writer_targets_protected":
            "ouroboros/tools/shell_guards.py::shell_writer_targets_protected",
        "ouroboros/tools/registry.py::is_external_workspace":
            "ouroboros/tool_access.py::is_external_workspace",
        "ouroboros/tools/registry.py::normalize_root":
            "ouroboros/tool_access.py::normalize_root",
        "ouroboros/tools/registry.py::resolve_shell_cwd":
            "ouroboros/tool_access.py::resolve_shell_cwd",
        "ouroboros/tools/registry.py::SKILL_PAYLOAD_CONTROL_DIRNAMES":
            "ouroboros/contracts/skill_payload_policy.py::SKILL_PAYLOAD_CONTROL_DIRNAMES",
        "ouroboros/tools/registry.py::is_skill_payload_path":
            "ouroboros/contracts/skill_payload_policy.py::is_skill_payload_path",
        "ouroboros/tools/registry.py::resolve_skill_payload_target":
            "ouroboros/contracts/skill_payload_policy.py::resolve_skill_payload_target",
    }
    registry_core_symbols = """ToolRegistry log _PROCESS_COMMAND_TOOLS
        _SHELL_GUARDED_TOOLS _REPO_MUTATION_TOOLS
        _SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS""".split()
    registry_core_rows = {
        f"ouroboros/tools/registry.py::{symbol}":
            f"ouroboros/tools/registry_core.py::{symbol}"
        for symbol in registry_core_symbols
    }
    registry_resolution_symbols = """_ROOT_ARG_REPO_WRITE_TOOLS _payload_write_paths
        _TOOL_ARG_ALIASES _IGNORE_ROOT_ARG_TOOLS _handler_public_params
        _entry_public_params _entry_has_public_param_schema _normalize_tool_call_args
        _prepare_public_builtin_args _light_binding_failure_redirect _binding_error_text
        _format_tool_arg_error""".split()
    registry_resolution_rows = {
        f"ouroboros/tools/registry.py::{symbol}":
            f"ouroboros/tools/tool_resolution.py::{symbol}"
        for symbol in registry_resolution_symbols
    }
    registry_guard_rows = {
        "ouroboros/tools/registry.py::_stray_skill_payload_failsoft":
            "ouroboros/tools/registry_guards.py::_stray_skill_payload_failsoft",
        "ouroboros/tools/registry.py::_payload_dispatch_constraint":
            "ouroboros/tools/registry_guards.py::_payload_dispatch_constraint",
    }
    registry_dispatch_method_rows = {
        "ouroboros/tools/registry.py::ToolRegistry._dispatch_mcp_tool":
            "ouroboros/tools/extension_dispatch.py::_dispatch_mcp_tool_result",
        "ouroboros/tools/registry.py::ToolRegistry._dispatch_extension_tool":
            "ouroboros/tools/extension_dispatch.py::_dispatch_extension_tool_result",
        "ouroboros/tools/registry.py::ToolRegistry._resolve_python_predispatch":
            "ouroboros/tools/tool_resolution.py::_resolve_python_predispatch",
    }
    dependency_symbols_by_owner = _inv.dependency_symbols_by_owner
    registry_dependency_owners = {
        f"ouroboros/tools/registry.py::{symbol}": f"{owner}::{symbol}"
        for owner, symbols in dependency_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    core_extraction_rows = {f"ouroboros/tools/core.py::{symbol}": f"ouroboros/tools/{'core_artifacts.py' if symbol in '_MAX_PHOTO_FILE_BYTES _detect_image_mime _send_photo _MAX_VIDEO_FILE_BYTES _detect_video_mime _send_video _MAX_DOCUMENT_FILE_BYTES _detect_document_mime _send_file'.split() else 'core_file_tools.py'}::{symbol}" for symbol in "_SKILL_OWNER_STATE_FILENAMES _direct_resource_binding _render_line_slice _coerce_start_char _coerce_line_window _is_cognitive_data_path _is_skill_owner_state_target _ListingFailure _list_dir _list_user_files_dir _SUBAGENT_SECRET_FILE_NAMES is_restricted_subagent_profile _is_subagent_secret_data_path _is_subagent_secret_repo_path _is_subagent_secret_repo_target _filter_subagent_secret_repo_listing _filter_subagent_secret_listing _MEMORY_AT_DRIVE_MEMORY _repo_read _repo_list _normalize_data_read_path _data_read _data_list _profile_roots_hint _access_or_block _local_readonly_resource_block _root_display_path _annotate_reread _read_file _list_files _MAX_PHOTO_FILE_BYTES _detect_image_mime _send_photo _MAX_VIDEO_FILE_BYTES _detect_video_mime _send_video _MAX_DOCUMENT_FILE_BYTES _detect_document_mime _send_file".split()} | {"ouroboros/tools/core.py::_code_search": "ouroboros/tools/core.py::_code_search"} | {"ouroboros/tools/core.py::_filter_out_project_store": "ouroboros/project_facts.py::filter_out_project_store", "ouroboros/tools/core.py::_policy_is_skill_owner_state_target": "ouroboros/contracts/skill_payload_policy.py::is_skill_owner_state_target", "ouroboros/tools/core.py::active_repo_dir_for": "ouroboros/tools/tool_resolution.py::active_repo_dir_for", "ouroboros/tools/core.py::active_tool_profile": "ouroboros/tool_access.py::active_tool_profile", "ouroboros/tools/core.py::build_resolved_resource_binding": "ouroboros/tool_access.py::build_resolved_resource_binding", "ouroboros/tools/core.py::decide_tool_access": "ouroboros/tool_access.py::decide_tool_access", "ouroboros/tools/core.py::normalize_root": "ouroboros/tool_access.py::normalize_root", "ouroboros/tools/core.py::normalize_runtime_data_path": "ouroboros/tool_access.py::normalize_runtime_data_path", "ouroboros/tools/core.py::read_text": "ouroboros/utils.py::read_text", "ouroboros/tools/core.py::SKILL_OWNER_STATE_FILENAMES": "ouroboros/contracts/skill_payload_policy.py::SKILL_OWNER_STATE_FILENAMES", "ouroboros/tools/browser.py::_readonly_subagent": "ouroboros/tools/core_file_tools.py::is_restricted_subagent_profile", "tests/test_filesystem_root_observability.py::_read_file": "ouroboros/tools/core_file_tools.py::_read_file", "tests/test_headless_cli.py::_repo_read": "ouroboros/tools/core_file_tools.py::_repo_read", "tests/test_send_file.py::_MAX_DOCUMENT_FILE_BYTES": "ouroboros/tools/core_artifacts.py::_MAX_DOCUMENT_FILE_BYTES", "tests/test_send_file.py::_detect_document_mime": "ouroboros/tools/core_artifacts.py::_detect_document_mime", "tests/test_send_file.py::_send_file": "ouroboros/tools/core_artifacts.py::_send_file", "tests/test_send_photo.py::_MAX_PHOTO_FILE_BYTES": "ouroboros/tools/core_artifacts.py::_MAX_PHOTO_FILE_BYTES", "tests/test_send_photo.py::_detect_image_mime": "ouroboros/tools/core_artifacts.py::_detect_image_mime", "tests/test_send_photo.py::_send_photo": "ouroboros/tools/core_artifacts.py::_send_photo", "tests/test_send_video.py::_MAX_VIDEO_FILE_BYTES": "ouroboros/tools/core_artifacts.py::_MAX_VIDEO_FILE_BYTES", "tests/test_send_video.py::_detect_video_mime": "ouroboros/tools/core_artifacts.py::_detect_video_mime", "tests/test_send_video.py::_send_video": "ouroboros/tools/core_artifacts.py::_send_video"}
    git_extraction_symbols_by_owner = _inv.git_extraction_symbols_by_owner
    git_extraction_rows = {
        f"ouroboros/tools/git.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in git_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    shell_extraction_symbols_by_owner = _inv.shell_extraction_symbols_by_owner
    shell_extraction_rows = {
        f"ouroboros/tools/shell.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in shell_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    headless_extraction_symbols_by_owner = _inv.headless_extraction_symbols_by_owner
    headless_extraction_rows = {
        f"ouroboros/headless.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in headless_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    tool_access_extraction_symbols_by_owner = {
        "tool_access_types.py": "ToolProfile ResourceRoot Operation SubagentCapability ToolAccessDecision ResolvedResourceBinding _ALL_ROOTS _READONLY_RESOURCE_ROOTS _TOP_LEVEL_PRINCIPAL_PROFILES _READ_OPS _TOP_LEVEL_PRINCIPAL_POLICY _POLICY _SUBAGENT_CAPABILITY_TO_OPERATION SUBAGENT_CAPABILITIES",
        "tool_access_paths.py": "_user_files_root _deliverables_root normalize_root path_is_relative_to normalize_root_relative _path_is_relative_to_casefold paths_overlap_casefold workspace_mode_block_reason canonical_data_root normalize_runtime_data_path",
        "tool_access_roots.py": "_is_subagent_ctx is_external_workspace active_tool_profile predicted_subagent_profile project_room_lens_dir load_bound_skill _skill_payload_base resource_root_path binding_targets_system_repo",
        "tool_access_user_files.py": "_USER_FILES_SECRET_COMPONENTS _USER_FILES_SECRET_NAMES _USER_FILES_SECRET_RE _USER_FILES_ALLOWED_DOTNAMES _subagent_projects_read_hint user_files_path_block_reason UserFilesPathBlockedError resolve_user_file_path",
    }
    tool_access_extraction_rows = {
        f"ouroboros/tool_access.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in tool_access_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    # v7 stream S, lane S1: config.py splits into the settings vocabulary, the closed
    # scales, the model slots, the reviewer routes and the numeric knobs. The parent keeps
    # the settings-file lifecycle, the path roots and the owner-only ratchets.
    config_extraction_symbols_by_owner = _inv.config_extraction_symbols_by_owner
    config_extraction_rows = {
        f"ouroboros/config.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in config_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    # v7 stream S lane S2: the server.py composition split. Every moved name keeps
    # its server.py facade, so no row belongs to the no-facade set.
    server_extraction_symbols_by_owner = _inv.server_extraction_symbols_by_owner
    server_extraction_rows = {
        f"server.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in server_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    # v7 stream W periphery extractions. Owner -> symbols, one row per symbol.
    w_stream_owners = {
        ("skills/unix_computer_use/plugin.py", "skills/unix_computer_use/lib/cu_runtime.py"):
            "_TIMEOUT_SEC _MAX_IMAGE_W _MAX_IMAGE_H _CONNECTIONS_FILE _ACTIVE_CONNECTION_FILE _REMOTE_BACKENDS _MAX_REMOTE_SHOT_BYTES _OSWORLD_PKGS_PREFIX _osworld_result_ok _png_dimensions _png_intact _json _run",
        ("skills/unix_computer_use/plugin.py", "skills/unix_computer_use/lib/cu_connections.py::_ConnectionRegistryMixin"):
            "_ComputerUse._connections_path _ComputerUse._active_connection_path _ComputerUse._read_connections _ComputerUse._atomic_write _ComputerUse._write_connections _ComputerUse._active_connection _ComputerUse._disabled_connection_error _ComputerUse._active_backend_name _ComputerUse._is_remote _ComputerUse.list_connections _ComputerUse.add_connection _ComputerUse.activate_connection _ComputerUse.use_local _ComputerUse.clear_active_connection _ComputerUse.test_connection",
        ("skills/unix_computer_use/plugin.py", "skills/unix_computer_use/lib/cu_remote_backends.py::_RemoteBackendMixin"):
            "_ComputerUse._connection_target _ComputerUse._osworld_execute _ComputerUse._ssh_macos_key_name _ComputerUse._ssh_macos_cliclick_for_pyautogui _ComputerUse._remote_pyautogui _ComputerUse._remote_screenshot_result _ComputerUse._osworld_screenshot _ComputerUse._test_osworld _ComputerUse._ssh_destination _ComputerUse._ssh_scp_source _ComputerUse._ssh_run _ComputerUse._ssh_macos_screenshot _ComputerUse._test_ssh_macos",
        ("devtools/benchmarks/osworld/run_cu_bridge_agent.py", "devtools/benchmarks/osworld/cu_bridge_runtime.py"):
            "SKILL_NAME _api _text_declares_infeasible _terminal_answer_text _final_answer_declares_infeasible",
        ("devtools/benchmarks/osworld/run_cu_bridge_agent.py", "devtools/benchmarks/osworld/cu_bridge_prompts.py"):
            "GATE_PREAMBLE GATE_SUFFIX OSWORLD_PREAMBLE _ACCEPTANCE_CLAIMS",
        ("devtools/benchmarks/osworld/run_cu_bridge_agent.py", "devtools/benchmarks/osworld/cu_bridge_tool_policy.py"):
            "_ALLOWED_CORE_TOOLS _core_tool_names _host_denied_tools _GUI_ACTION_TOOLS _DENIED_SKILL_EXT_TOOLS _effective_disabled_tools _COMPUTER_USE_SHORT_TOOLS",
        ("devtools/benchmarks/osworld/run_cu_bridge_agent.py", "devtools/benchmarks/osworld/cu_bridge_gate.py"):
            "_gate_window_sec _gate_claim_window_sec _gate_verdict _DesktopEnvLogCapture ResetUnverified _reset_verified _live_policy_turns _policy_turns _await_gate_task _gate_round _GATE_TURN_RESERVE _GUEST_DOWN_GRACE_SEC _guest_endpoint_healthy _gate_cancel_unconfirmed _gate_tool_trace _gate_turn_budget",
        ("devtools/benchmarks/osworld/run_cu_bridge_agent.py", "devtools/benchmarks/osworld/cu_bridge_budget.py"):
            "_effective_max_rounds _step_budget _official_evaluate_cwd _worker_round_cap _publish_worker_round_cap _proxy_trace_shows_exhaustion _verify_setup_effect _task_scoped_proxy_config _proxy_config_is_live _refuse_wrong_dataset_commit _refuse_uncapped_step_claim _audit_step_budget _collect_budget_counters",
        ("devtools/benchmarks/osworld/run_step_agent.py", "devtools/benchmarks/osworld/step_agent_common.py"):
            "StepAgentConfig TaskRecordConfig PreflightConfig _safe_slug _http_json",
        ("devtools/benchmarks/osworld/run_step_agent.py", "devtools/benchmarks/osworld/step_agent_env.py"):
            "VMWARE_FUSION_PATHS ALIGNED_UPSTREAM SUPPORTED_PROVIDERS osworld_checkout_info provider_preflight_failures _install_optional_dependency_stubs _ensure_vmrun_on_path _DEFAULT_DESKTOP_PORT _LOOPBACK_HOSTS _is_default_desktop_server _teardown_partial_desktop_env construct_desktop_env",
        ("devtools/benchmarks/osworld/run_step_agent.py", "devtools/benchmarks/osworld/step_agent_claims.py"):
            "ClaimDirNotConfined confined_claims_dir task_claim_key claim_stale_sec acquire_task_claim UNCONFIRMED_SCORE_SUFFIX ClaimMarkerNotDurable record_unconfirmed_score mark_task_scored scored_claim_state task_already_scored release_task_claim",
        ("devtools/benchmarks/osworld/run_step_agent.py", "devtools/benchmarks/osworld/step_agent_actions.py"):
            "SPECIAL_ACTIONS _json_from_text _shell_action _click_action _type_action _hotkey_action _wait_action _normalize_structured_action",
        ("devtools/benchmarks/osworld/run_step_agent.py", "devtools/benchmarks/osworld/step_agent_policy.py"):
            "_initial_observation_with_retries OuroborosStepAgent",
        ("web/tests/harness_accounts.test.js", "web/tests/harness_accounts_helpers.js"):
            "fakeResponse",
        ("web/tests/harness_accounts.test.js", "web/tests/harness_accounts_cards.test.js"):
            "cardWithUrl fakeCodeInput fakeCardHost",
        ("web/tests/harness_accounts.test.js", "web/tests/harness_accounts_custody.test.js"):
            "storeWithReads",
        ("web/tests/harness_accounts.test.js", "web/tests/harness_accounts_panel.test.js"):
            "fakeElement mountSection captureCardControls WAKE_STILL_DOWN WAKE_UP",
    }
    w_stream_rows = {
        f"{old}::{symbol}": (f"{owner}.{symbol.split('.', 1)[1]}" if "::" in owner
                             else f"{owner}::{symbol}")
        for (old, owner), symbols in w_stream_owners.items() for symbol in symbols.split()
    }
    shell_extraction_symbols_by_owner = _inv.shell_extraction_symbols_by_owner
    shell_extraction_rows = {
        f"ouroboros/tools/shell.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in shell_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    headless_extraction_symbols_by_owner = _inv.headless_extraction_symbols_by_owner
    headless_extraction_rows = {
        f"ouroboros/headless.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in headless_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    tool_access_extraction_symbols_by_owner = {
        "tool_access_types.py": "ToolProfile ResourceRoot Operation SubagentCapability ToolAccessDecision ResolvedResourceBinding _ALL_ROOTS _READONLY_RESOURCE_ROOTS _TOP_LEVEL_PRINCIPAL_PROFILES _READ_OPS _TOP_LEVEL_PRINCIPAL_POLICY _POLICY _SUBAGENT_CAPABILITY_TO_OPERATION SUBAGENT_CAPABILITIES",
        "tool_access_paths.py": "_user_files_root _deliverables_root normalize_root path_is_relative_to normalize_root_relative _path_is_relative_to_casefold paths_overlap_casefold workspace_mode_block_reason canonical_data_root normalize_runtime_data_path",
        "tool_access_roots.py": "_is_subagent_ctx is_external_workspace active_tool_profile predicted_subagent_profile project_room_lens_dir load_bound_skill _skill_payload_base resource_root_path binding_targets_system_repo",
        "tool_access_user_files.py": "_USER_FILES_SECRET_COMPONENTS _USER_FILES_SECRET_NAMES _USER_FILES_SECRET_RE _USER_FILES_ALLOWED_DOTNAMES _subagent_projects_read_hint user_files_path_block_reason UserFilesPathBlockedError resolve_user_file_path",
    }
    tool_access_extraction_rows = {
        f"ouroboros/tool_access.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in tool_access_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    # Verbatim theme splits of the v7 T-stream giant test modules: source module -> {owner path: moved symbols}.
    # A moved test, fixture or helper is owned by the sibling module that now hosts it; a test-private import
    # binding keeps the canonical production provider, which these splits never moved.
    test_split_symbols_by_owner = {
        "tests/test_headless_cli.py": {"ouroboros/gateway/tasks.py": "_compose_task_text _resolve_workspace_root api_task_artifact api_task_events api_task_get api_tasks_create api_tasks_list iter_task_events", "ouroboros/headless.py": "ARTIFACT_STATUS_FAILED ARTIFACT_STATUS_FINALIZING ARTIFACT_STATUS_READY ARTIFACT_STATUS_READY_WITH_CHANGES _incidental_lockfile_excludes build_memory_export build_workspace_patch finalize_task_artifacts prune_headless_task_drives prune_task_drives task_artifacts_dir write_workspace_patch_artifacts", "ouroboros/task_results.py": "write_task_result", "ouroboros/tools/registry.py": "ToolContext ToolRegistry", "ouroboros/workspace_preflight.py": "_infer_tools_from_manifests", "tests/_headless_cli_shared.py": "_init_repo_with_file _managed_worker_pool_available", "tests/test_headless_task_api.py": "test_api_tasks_create_rejects_internal_task_types test_api_tasks_create_requires_description_not_legacy_aliases test_compose_task_text_extends_existing_headless_workspace_block test_resolve_workspace_root_blocks_case_variant_control_plane test_task_api_admission_refusal_is_terminal_not_scheduled_phantom test_task_api_enqueue_workspace_creates_child_drive test_task_api_preserves_top_level_actor_id_after_metadata_sanitization test_task_api_refuses_when_durable_queue_snapshot_fails test_task_api_rejects_external_lineage_forgery test_task_api_rejects_forged_subagent_without_child_drive_side_effect test_task_api_rejects_unsafe_task_id_and_system_workspace test_task_api_releases_reservation_when_payload_composition_fails", "tests/test_headless_task_artifacts.py": "test_child_copyback_preserves_acceptance_verdict_and_terminal_post_task_marker test_copy_child_result_cannot_overwrite_finalized_accounting test_copy_child_result_merges_cost_before_finalization test_effective_result_preserves_workspace_artifact_status_with_child_drive test_effective_result_preserves_workspace_patch_kind_with_child_drive test_external_child_task_budget_uses_parent_drive_state test_finalize_task_artifacts_preserves_existing_artifact_axis_fields test_memory_export_includes_nested_memory_files test_startup_prune_removes_only_old_terminal_child_drives test_startup_prune_removes_only_old_terminal_task_scratch test_startup_prune_uses_effective_terminal_status test_task_artifact_endpoint_rebases_child_drive_artifact_after_status_repair test_task_artifact_endpoint_rejects_metadata_name_path_mismatch test_task_artifact_endpoint_serves_manifest_artifact_after_status_repair test_task_artifact_endpoint_serves_only_declared_artifacts", "tests/test_headless_task_events.py": "test_effective_child_completion_waits_for_artifacts test_effective_child_failure_waits_for_artifacts test_effective_task_result_preserves_parent_terminal_status test_logs_tail_parent_filter_includes_child_lineage_events test_public_task_result_strips_nested_legacy_result_status test_task_event_replay_parent_includes_child_lineage_events test_task_event_replay_uses_existing_logs_and_result test_task_list_filters_on_effective_child_status test_task_sse_emits_final_result_after_cursor_saw_scheduled_result test_workspace_event_replay_suppresses_task_done_until_artifacts_terminal", "tests/test_headless_workspace_patch.py": "test_failed_refinalization_drops_stale_workspace_patch_metadata test_finalize_workspace_patch_allows_external_workspace_head_changed test_finalize_workspace_patch_exception_manifest_keeps_base_fields test_workspace_patch_allows_benign_tokenizer_json test_workspace_patch_allows_external_workspace_first_commit test_workspace_patch_excludes_binary_junk_and_oversize test_workspace_patch_fails_on_common_credential_paths test_workspace_patch_fails_on_invalid_head_not_unborn test_workspace_patch_fails_on_sensitive_untracked_file test_workspace_patch_fails_on_sensitive_untracked_file_inside_excluded_dir test_workspace_patch_fails_when_acting_base_sha_head_changed test_workspace_patch_includes_tracked_and_untracked_files test_workspace_patch_lockfile_without_manifest_is_incidental_only_with_code_changes test_workspace_patch_manifest_excludes_env_cache_dirs test_workspace_patch_preserves_lockfile_when_other_changes_are_junk test_workspace_patch_preserves_untracked_paths_with_whitespace test_workspace_patch_supports_unborn_git_worktree test_workspace_patch_supports_unborn_sha256_git_worktree test_workspace_patch_uses_acting_base_sha_without_preflight_metadata", "tests/test_headless_workspace_shell.py": "test_external_workspace_shell_allows_task_local_git test_workspace_context_routes_project_files_and_keeps_system_tools_reachable test_workspace_preflight_infers_binaries_from_script_commands test_workspace_run_shell_allows_absolute_cwd_under_workspace_and_child_drive test_workspace_run_shell_cwd_allows_scratch_and_explicit_system test_workspace_shell_allows_nested_relative_write_paths test_workspace_shell_blocks_nested_symlink_escape_absolute_path test_workspace_shell_blocks_windows_absolute_redirects_before_shell_execution test_workspace_shell_git_ls_remote_requires_network_contract test_workspace_shell_keeps_symlinked_workspace_absolute_paths_allowed test_workspace_shell_safe_stdio_redirects_are_not_write_like test_workspace_shell_sudo_and_pro_passthrough_policy"},
        "tests/test_git_review_pipeline.py": {"tests/_git_review_pipeline_shared.py": "_critical_triad_items _get_git_module _get_git_ops_module _get_registry_module _get_review_module _make_ctx", "tests/test_git_review_advisory_skip_tests.py": "TestAdvisorySkipTests", "tests/test_git_review_bypass_gate.py": "TestBypassPathTestsRun TestRouteSlotAwareBypassGate _make_staged_repo", "tests/test_git_review_enforcement.py": "TestReviewEnforcementModes TestReviewHistoryBuilding TestReviewQuorumLogic _PARSE_REVIEW_JSON_CASES review_ctx test_parse_review_json", "tests/test_git_review_preflight_gate.py": "TestPreflightCheck7P9Limits _PREFLIGHT_CASES test_preflight_check"},
        "tests/test_tool_capabilities.py": {"tests/test_tool_capabilities_black_box_policy.py": "test_protected_black_box_artifact_policy_blocks_introspection test_protected_black_box_recursive_policy_maps_executor_backend_paths test_runtime_data_write_blocks_workspace_executor_control_state", "tests/test_tool_capabilities_readonly_subagent.py": "test_allowed_resources_block_web_and_external_tools test_local_readonly_subagent_allows_enabled_extension_tool test_local_readonly_subagent_data_read_denies_secret_files test_local_readonly_subagent_execute_blocks_forbidden_tools test_local_readonly_subagent_repo_read_denies_secret_files test_local_readonly_subagent_task_drive_and_skill_payload_filters", "tests/test_tool_capabilities_search_code.py": "_make_ctx _populate_repo test_code_search_empty_query test_code_search_include_filter test_code_search_invalid_regex test_code_search_literal test_code_search_max_results test_code_search_no_matches test_code_search_regex test_code_search_scoped_path test_code_search_skips_binaries test_code_search_skips_cache_dirs test_search_code_does_not_follow_symlink_outside_root test_search_code_has_result_limit test_search_code_in_core_tools test_search_code_in_initial_schemas test_search_code_is_parallel_safe test_search_code_registered test_search_code_ripgrep_fallback_when_unavailable test_search_code_ripgrep_path_filters_protected_files", "tests/test_tool_capabilities_subagent_scheduling.py": "test_capability_omission_manifest_surfaces_extension_discovery_failure test_get_task_result_in_core test_local_readonly_subagent_initial_schemas_are_allowlisted test_schedule_subagent_available_in_registry test_schedule_subagent_in_core test_schedule_subagent_in_initial_schemas test_schedule_subagent_inherits_workspace_executor_ref test_schedule_subagent_required_capabilities_fail_fast_for_readonly test_schedule_subagent_required_delegate_capability_is_satisfied_for_readonly test_schedule_subagent_required_vcs_capability_is_satisfied_for_readonly test_wait_task_in_core test_workspace_focus_does_not_turn_top_level_cancel_into_child_only test_workspace_parent_keeps_the_ordinary_top_level_control_surface"},
    }
    test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    test_split_facade_rows = {"tests/test_headless_cli.py::_managed_worker_pool_available",
                              "tests/test_git_review_pipeline.py::_get_git_module",
                              "tests/test_git_review_pipeline.py::_get_git_ops_module",
                              "tests/test_git_review_pipeline.py::_get_registry_module",
                              "tests/test_git_review_pipeline.py::_make_ctx"}
    web_extractions = {f"web/modules/chat.js::{symbol}": f"web/modules/{owner}::{symbol}" for owner, symbols in {"chat_card_state.js": "liveLineRowToggleKey clearStickyCardState COLLAPSED_ACTIVITY_MAX boundActivityPreview projectCollapsedActivity isTerminalTaskPhase", "chat_controls.js": "shouldFirePanic confirmAndSendPanic", "chat_render_batch.js": "insertTimelineNode", "costs.js": "headerBudgetPresentation taskCostMeta taskCostProjection mergeStickyCostMeta", "utils.js": "rawTimestampEpoch", "chat_card_actions.js": "projectIdFromTask", "chat_attachments.js": "MAX_PENDING_ATTACHMENTS MAX_ATTACHMENT_FILE_BYTES MAX_PENDING_ATTACHMENT_BYTES", "chat_history_sync.js": "CHAT_STORAGE_KEY"}.items() for symbol in symbols.split()}
    # createChatInstance closure helpers moved into per-instance factories (no facade: they were never exported)
    web_extractions.update({f"web/modules/chat.js::createChatInstance.{symbol}": f"web/modules/{owner}::{factory}.{symbol}" for owner, factory, symbols in (
        ("chat_timeline_anchor.js", "createTimelineAnchors", "NEAR_BOTTOM_THRESHOLD_PX isNearBottom captureVisibleTimelineAnchor restoreVisibleTimelineAnchor"),
        ("chat_message_identity.js", "createMessageIdentity", "buildMessageKey rememberMessageKey formatMsgTime stampNodeTimestamp getSenderLabel"),
        ("chat_document_bubble.js", "createDocumentBubbles", "buildDocumentBubble documentMessageKey appendDocumentBubble"),
        ("chat_subagent_routing.js", "createSubagentRouting", "setSubagentParent summarizeSubagentCardFrame updateSubagentCardFromEvent routeSubagentProgressToCard routeSubagentFinalMessageToCard routeSubagentTerminalToCard"),
    ) for symbol in symbols.split()})
    # v7 wave C, lane W3: the remaining createChatInstance closure clusters, each moved
    # whole into a per-instance factory of its own sibling owner. No facade: none of
    # these helpers was ever exported, so the ledger identity is the only address.
    w3_chat_extraction_symbols_by_owner = (
        ("chat_task_ui_state.js", "createTaskUiStateTracker", "isBackgroundTaskId shouldAlwaysShowTaskCard isForegroundLiveCard createTaskUiState getTaskUiState scheduleTaskUiCleanup bufferLiveUpdate markTaskToolCall forceTaskCard markAssistantReply markTaskComplete"),
        ("chat_card_actions.js", "createCardActions", "turnTaskIntoProject ensureLiveActionsEl syncCancelRunButton syncCancelRunButtonMutation markLiveCardCancelPending captureLiveCardPhase restoreLiveCardPhase reconcileCancelCardFromDetail cancelRunFromCard markTaskCancelable markCardConverted markCardConvertedMutation"),
        ("chat_live_card_view.js", "createLiveCardView", "applySuggestedName applySuggestedNameMutation renderCollapsedActivity ensureSubagentContainer setLiveCardTypingVisible formatLiveCardPhaseLabel setLiveCardExpanded isLiveLineExpandable syncLiveCardToggle directSubagentCount buildTimelineItemHtml isTimelinePinnedToBottom deferCollapsedTimeline renderLiveCardTimeline appendTimelineItem patchLastTimelineItem patchTimelineItemAt renderLiveCardMeta"),
        ("chat_message_annotations.js", "createMessageAnnotations", "routingAnnotationText renderRoutingAnnotation updateMessageAnnotation clearTransientRoutingAnnotations markPendingDelivered"),
        ("chat_composer.js", "createComposer", "resizeChatInput swarmArmed setSwarm setSendBusy scrollToBottom updateScrollButton updateMessagesPadding"),
        ("chat_header_controls.js", "createHeaderControls", "syncHeaderControlState refreshHeaderControlState"),
        ("chat_frame_routing.js", "createFrameRouting", "isKnownProjectFrame incrementUnreadIfNeeded isProjectMirrorFrame isMyThread"),
        # v7 W3 wave D (chat.js size campaign): the remaining per-instance closure
        # clusters — attachment staging, the live-card store, the task-frame router
        # and the history/feed owner — each moved whole into its own factory.
        ("chat_attachments.js", "createChatAttachments", "pendingAttachmentBytes updateAttachmentPreview stagePendingFiles cleanupUploadedAttachments setAttachmentUploadState isFileDrag setFileDragActive"),
        ("chat_live_cards.js", "createChatLiveCards", "registerEphemeralDecisionFrame registerEphemeralDecisionFrameMutation reanchorTaskCard reanchorVisibleTaskCard revealBufferedCardIfNeeded revealBufferedCardMutation queueTaskLiveUpdate queueTaskLiveUpdateMutation createLiveCardRecord getLiveCardRecord getSubagentCardRecord getSubagentCardRecordMutation resetLiveCardRecord ensureLiveCardVisible updateLiveCardCount syncLiveCardLayout fetchFullLineOutput applyLiveCardState applyLiveCardStateMutation finishLiveCard finishLiveCardMutation"),
        ("chat_task_frames.js", "createTaskFrames", "appendTaskSummaryToLiveCard updateLiveCardFromProgressMessage updateLiveCardFromLogEvent"),
        ("chat_history_sync.js", "createChatHistorySync", "readPendingReconnectBanner clearPendingReconnectBanner persistVisibleHistory insertMessageNode addMessage ensureWelcomeMessage awaitInitialHydration MAIN_HYDRATION_MAX_DEFER_MS waitForHydrationWindow finalizeRebuildBatch syncHistory cancelHistoryPaint refreshHistory scheduleHistorySync historyResyncScheduler syncLoadOlderControl loadOlderHistory"),
    )
    # The same wave's chat.js module-scope primitives: three closure helpers with no
    # closure reads at all became plain top-level owners, and the cost presentation
    # joined the existing costs owner.
    w3_chat_primitive_rows = {
        "web/modules/chat.js::withTaskCostMeta": "web/modules/costs.js::withTaskCostMeta",
        "web/modules/chat.js::shownIncidentToastKeys": "web/modules/chat_notices.js::shownIncidentToastKeys",
        "web/modules/chat.js::showTaskIncidentToast": "web/modules/chat_notices.js::showTaskIncidentToast",
        "web/modules/chat.js::showContextFitToast": "web/modules/chat_notices.js::showContextFitToast",
    }
    web_extractions.update(w3_chat_primitive_rows)
    web_extractions.update({f"web/modules/chat.js::createChatInstance.{symbol}": f"web/modules/{owner}::{factory}.{symbol}" for owner, factory, symbols in w3_chat_extraction_symbols_by_owner for symbol in symbols.split()})
    implemented.update(registry_core_rows)
    implemented.update(registry_resolution_rows)
    implemented.update(registry_guard_rows)
    implemented.update(registry_dispatch_method_rows)
    implemented.update(registry_dependency_owners | core_extraction_rows)
    implemented.update(git_extraction_rows)
    implemented.update(shell_extraction_rows)
    implemented.update(headless_extraction_rows)
    implemented.update(tool_access_extraction_rows)
    implemented.update(server_extraction_rows)
    registry_extraction_no_facade_rows = (
        set(registry_core_rows) - {"ouroboros/tools/registry.py::ToolRegistry"}
    ) | set(registry_resolution_rows) | set(registry_guard_rows) | set(
        registry_dispatch_method_rows
    ) | set(registry_dependency_owners) | set(core_extraction_rows)
    # _ComputerUse methods move into mixin classes: the class inherits the exact
    # same function object, so the compatibility contract is inheritance, not a
    # module-level re-export, and the facade cell is "-".
    registry_extraction_no_facade_rows |= {
        identity for identity in w_stream_rows if "::_ComputerUse." in identity
    }
    # Node test fixtures move to a sibling *.test.js discovered by the same glob;
    # a test module has no re-export contract, so those facade cells stay "-".
    registry_extraction_no_facade_rows |= {
        identity for identity in w_stream_rows
        if identity.startswith("web/tests/") and "fakeResponse" not in identity
    }
    retired_current = {
        "ouroboros/tools/registry.py::_HEAL_PROTECTED_PAYLOAD_FILENAMES":
            "retired:unused payload-control alias removed with registry core extraction",
        "tests/test_commit_gate.py::_get_registry_module": "retired:test-only registry import helper removed when CORE_TOOL_NAMES characterization moved to its canonical owner",
    }
    retired_current["ouroboros/review.py::_git_source_snapshot"] = (
        "retired:ref inventories read blobs directly through _iter_ref_gated_blobs "
        "and reuse them by blob id"
    )
    retired_current.update({"web/modules/chat.js::optionalFiniteNumber": "web/modules/costs.js::optionalFiniteNumber", "ouroboros/loop_tool_execution.py::PLAN_REVIEW_CONTROL_PREFIX": "retired:loop no longer imports the display-only plan footer prefix"})
    # D02 (lane d02): the typed plan-result seam re-applied on the upstream-rewritten
    # engine. The loop's textual parser and its vocabulary re-home to the grammar owner
    # beside the emitter (the loop reads native metadata only, no re-export, facade "-");
    # the wrapper and the engine coroutine change in place (facade "-"); the typed cap
    # result moves to the runtime-seam owner behind an import-alias facade.
    d02_plan_seam_rows = {
        "ouroboros/loop_tool_execution.py::_parse_plan_review_control":
            "ouroboros/tools/plan_render.py::_parse_plan_review_control",
        "ouroboros/loop_tool_execution.py::_PLAN_REVIEW_OUTCOMES":
            "ouroboros/tools/plan_render.py::_PLAN_REVIEW_OUTCOMES",
        "ouroboros/tools/plan_review.py::_handle_plan_task":
            "ouroboros/tools/plan_review.py::_handle_plan_task",
        "ouroboros/tools/plan_review.py::_run_plan_review_async":
            "ouroboros/tools/plan_review.py::_run_plan_review_async",
        "ouroboros/tools/plan_review.py::_cycles_exhausted":
            "ouroboros/tools/plan_review_runtime.py::plan_review_cycles_exhausted",
    }
    implemented.update(d02_plan_seam_rows)
    # The vacuity predicate moves verbatim and takes back its pre-rewrite public
    # name; the engine keeps the old private binding as a facade (delta none).
    implemented["ouroboros/tools/plan_review.py::_vacuous_disposition"] = (
        "ouroboros/tools/plan_review_runtime.py::vacuous_review_disposition"
    )
    retired_current.update({
        "ouroboros/tools/plan_review.py::PLAN_REVIEW_CONTROL_PREFIX":
            "ouroboros/tools/review_synthesis.py::PLAN_REVIEW_CONTROL_PREFIX",
        "ouroboros/tools/plan_review.py::current_plan_review_wave":
            "ouroboros/task_results.py::current_plan_review_wave",
    })
    # T1: two retirements that DO carry a semantic delta — the loop's ordered
    # families and generic markers move into the single classifier rather than
    # disappearing, so their rows name the spec 4.3.3 delta instead of "none".
    retired_current.update({
        "ouroboros/loop_tool_execution.py::_FAILURE_PREFIXES":
            "retired:the loop no longer classifies result text; the single classifier owns every family",
        "ouroboros/loop_tool_execution.py::_FAILURE_MARKERS":
            "retired:the generic marker fallbacks live once, in the single classifier",
    })
    # T3: the last two text scans of a process result. Their only reader moved to
    # typed meta at the cutover, so they carry the same 4.3.3 delta rather than
    # retiring as observable-identical dead code.
    retired_current.update({
        "ouroboros/loop_tool_execution.py::_EXIT_CODE_RE":
            "retired:the process exit code is a producer fact carried in ToolResult.meta, "
            "never scraped from stdout",
        "ouroboros/loop_tool_execution.py::_SIGNAL_RE":
            "retired:the terminating signal is a producer fact carried in ToolResult.meta, "
            "never scraped from stdout",
    })
    retired_current["ouroboros/launcher_onboarding.py::save_settings"] = (
        "retired:the launcher persists nothing at startup; the pre-server provider "
        "normalization is applied to the environment and re-derived by every reader"
    )
    retired_delta_ids = {
        "ouroboros/launcher_onboarding.py::save_settings": "D03",
        # Panel finding (fable seat, 2026-08-20): the snapshot-path shadow
        # collapse is harness-observable (state.init required; queue.init alone
        # no longer redirects), so the retired row carries the D18 queue
        # single-authority id instead of the observable-identical "none".
        "supervisor/queue.py::QUEUE_SNAPSHOT_PATH": "D18",
        "ouroboros/loop_tool_execution.py::_FAILURE_PREFIXES": "D02",
        "ouroboros/loop_tool_execution.py::_FAILURE_MARKERS": "D02",
        "ouroboros/loop_tool_execution.py::_EXIT_CODE_RE": "D02",
        "ouroboros/loop_tool_execution.py::_SIGNAL_RE": "D02",
        # S3, spec 4.3.8: the safety module's import-time supervisor edge.
        "ouroboros/safety.py::update_budget_from_usage": "D05",
        # S3, spec 4.3.6: three worker-pool globals nothing read.
        "supervisor/workers.py::SOFT_TIMEOUT_SEC": "D04",
        "supervisor/workers.py::HARD_TIMEOUT_SEC": "D04",
        "supervisor/workers.py::TOTAL_BUDGET_LIMIT": "D04",
    }
    retired_current.update({
        "supervisor/workers.py::SOFT_TIMEOUT_SEC":
            "retired:no rail reads it; the queue raises the deprecation notice and discards the value",
        "supervisor/workers.py::HARD_TIMEOUT_SEC":
            "retired:no rail reads it; the queue raises the deprecation notice and discards the value",
        "supervisor/workers.py::TOTAL_BUDGET_LIMIT":
            "retired:a third copy of a limit nothing read; supervisor.state is the budget authority",
    })
    retired_current["ouroboros/safety.py::update_budget_from_usage"] = (
        "retired:the ledger writer is injected by the context, or reached at call time"
    )
    existing_process_owner_rows = {
        "tests/test_skill_exec.py::test_run_shell_restores_obfuscated_self_authored_state_marker",
        "ouroboros/tools/registry.py::SKILL_OWNER_STATE_FILENAMES",
        "ouroboros/tools/registry.py::parse_porcelain_paths",
        "ouroboros/tools/registry.py::safe_relpath",
        "ouroboros/tools/registry.py::LIGHT_SHELL_WRITER_COMMANDS",
        "ouroboros/tools/registry.py::SKILL_OWNER_STATE_STEMS",
        "ouroboros/tools/registry.py::build_resolved_resource_binding",
        "ouroboros/tools/registry.py::interpreter_family",
        "ouroboros/tools/registry.py::light_shell_repo_mutation",
        "ouroboros/tools/registry.py::protected_artifact_shell_block_reason",
        "ouroboros/tools/registry.py::runtime_data_guard_targets",
        "ouroboros/tools/registry.py::shell_command_string",
        "ouroboros/tools/registry.py::strip_leading_env_assignments",
        "ouroboros/tools/registry.py::sudo_noninteractive_violation",
        "ouroboros/tools/registry.py::unwrap_env_argv",
        "ouroboros/tools/registry.py::workspace_executor_state_write_block",
        "ouroboros/tools/registry.py::writer_target_tokens",
        "ouroboros/tools/registry.py::PROTECTED_RUNTIME_PATHS",
        "ouroboros/tools/registry.py::task_artifact_dir_path",
        "ouroboros/tools/registry.py::task_id_for_artifacts",
        "ouroboros/tools/registry.py::run_shell_git_block_reason",
        "ouroboros/tools/registry.py::workspace_git_safety_violation",
        "ouroboros/tools/registry.py::is_absolute_path_text",
        "ouroboros/tools/registry.py::path_text_is_inside",
        "ouroboros/tools/registry.py::shell_argv",
        "ouroboros/tools/registry.py::shell_argv_with_path_tokens",
        "ouroboros/tools/registry.py::PROTECTED_RUNTIME_PATHS_LOWER",
        "ouroboros/tools/registry.py::shell_has_write_indicator",
        "ouroboros/tools/registry.py::shell_writer_targets_protected",
        "ouroboros/tools/registry.py::is_external_workspace",
        "ouroboros/tools/registry.py::normalize_root",
        "ouroboros/tools/registry.py::resolve_shell_cwd",
        "ouroboros/tools/registry.py::SKILL_PAYLOAD_CONTROL_DIRNAMES",
        "ouroboros/tools/registry.py::is_skill_payload_path",
        "ouroboros/tools/registry.py::resolve_skill_payload_target",
    }
    # T1: _FAILURE_PREFIXES and _FAILURE_MARKERS are retired rather than re-owned;
    # the loop pair keeps its names as compatibility wrappers over the typed owners.
    implemented.update({name: name for name in ("ouroboros/_outcome_tool_errors.py::_BLOCKING_TOOL_STATUSES", "ouroboros/reflection.py::_ERROR_MARKERS")})
    implemented["ouroboros/loop_tool_execution.py::_extract_result_metadata"] = "ouroboros/loop_tool_execution.py::_typed_result_metadata"
    implemented["ouroboros/loop_tool_execution.py::_is_tool_execution_failure"] = "ouroboros/loop_tool_execution.py::_typed_execution_failure"
    implemented["ouroboros/loop_tool_execution.py::_structured_tool_failure"] = "ouroboros/tools/tool_result.py::_structured_failure"
    implemented["tests/test_tool_execution_classification.py::test_shell_and_claude_failures_are_treated_as_tool_failures"] = "tests/test_tool_execution_classification.py::test_shell_and_protected_failures_are_treated_as_tool_failures"
    existing_process_owner_rows.update({"ouroboros/tools/core.py::_code_search", "ouroboros/loop_tool_execution.py::_structured_tool_failure", "tests/test_tool_execution_classification.py::test_shell_and_claude_failures_are_treated_as_tool_failures", 'ouroboros/tools/core.py::_filter_out_project_store', 'ouroboros/tools/core.py::_policy_is_skill_owner_state_target', 'ouroboros/tools/core.py::active_repo_dir_for', 'ouroboros/tools/core.py::active_tool_profile', 'ouroboros/tools/core.py::build_resolved_resource_binding', 'ouroboros/tools/core.py::decide_tool_access', 'ouroboros/tools/core.py::normalize_root', 'ouroboros/tools/core.py::normalize_runtime_data_path', 'ouroboros/tools/core.py::read_text', 'ouroboros/tools/core.py::SKILL_OWNER_STATE_FILENAMES', "ouroboros/loop_tool_execution.py::_extract_result_metadata", "ouroboros/loop_tool_execution.py::_is_tool_execution_failure", "ouroboros/_outcome_tool_errors.py::_BLOCKING_TOOL_STATUSES", "ouroboros/reflection.py::_ERROR_MARKERS"})
    registry_extraction_no_facade_rows.update({"ouroboros/loop_tool_execution.py::_structured_tool_failure", "tests/test_tool_execution_classification.py::test_shell_and_claude_failures_are_treated_as_tool_failures", "ouroboros/loop_tool_execution.py::_extract_result_metadata", "ouroboros/loop_tool_execution.py::_is_tool_execution_failure", "ouroboros/_outcome_tool_errors.py::_BLOCKING_TOOL_STATUSES", "ouroboros/reflection.py::_ERROR_MARKERS"})
    # T1 fix batch: the self-reported-failure homing, the reflection ok-set, and the
    # two classifier pins that move out of the loop's wall module.
    t1_fix_rows = {
        "ouroboros/_outcome_tool_errors.py::_POLICY_DENIAL_STATUSES": "ouroboros/_outcome_tool_errors.py::_POLICY_DENIAL_STATUSES",
        "ouroboros/reflection.py::should_generate_reflection": "ouroboros/reflection.py::_trace_call_errored",
        "tests/test_loop_misc.py::test_a_tool_that_reports_its_own_failure_is_not_recorded_as_success": "tests/test_tool_execution_classification.py::test_a_tool_that_reports_its_own_failure_is_not_recorded_as_success",
        "tests/test_loop_misc.py::test_auto_attach_skips_a_result_that_declared_failure": "tests/test_tool_execution_classification.py::test_auto_attach_skips_a_result_that_declared_failure",
    }
    implemented.update(t1_fix_rows)
    existing_process_owner_rows.update(t1_fix_rows)
    existing_process_owner_rows.add("tests/test_repo_health_smoke.py::test_transition_rejects_function_swap_even_at_same_cardinality")
    registry_extraction_no_facade_rows.update(t1_fix_rows)
    # D02 (lane d02): the seam rows land in existing engine modules (plan_render.py,
    # plan_review.py, plan_review_runtime.py). The parser/vocabulary moves and the
    # in-place wrapper/engine rows carry NO facade; the cap-result and vacuity-
    # predicate moves keep the old private bindings as import-alias facades.
    existing_process_owner_rows.update(d02_plan_seam_rows)
    existing_process_owner_rows.add("ouroboros/tools/plan_review.py::_vacuous_disposition")
    registry_extraction_no_facade_rows.update(
        identity for identity in d02_plan_seam_rows
        if identity != "ouroboros/tools/plan_review.py::_cycles_exhausted"
    )
    # The closure-invariant harness keeps its module-level parser binding as a
    # facade re-export re-pointed at the T1 home (verbatim binding, delta none).
    implemented["tests/test_plan_spec.py::_parse_plan_review_control"] = (
        "ouroboros/tools/plan_render.py::_parse_plan_review_control"
    )
    existing_process_owner_rows.add("tests/test_plan_spec.py::_parse_plan_review_control")
    # v7 stream S3: supervisor/events.py split into per-family owner modules.
    s3_events_symbols_by_owner = _inv.s3_events_symbols_by_owner
    s3_events_rows = {
        f"supervisor/events.py::{symbol}": f"supervisor/{owner}::{symbol}"
        for owner, symbols in s3_events_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(s3_events_rows)
    existing_process_owner_rows.update(
        identity for identity in s3_events_rows
        if s3_events_rows[identity].startswith("supervisor/queue_transitions.py")
    )
    s3_custody_rows = {
        f"supervisor/task_lifecycle.py::{symbol}": f"supervisor/cancel_custody.py::{symbol}"
        for symbol in """_queue_module _durable_settled_status cancel_task_custody SETTLED_ALREADY
            _worker_possibly_alive _active_intent _reaping_owner_abandoned
            _recover_stranded_reaping_slot _claim_intent _settle_intent _release_intent_claim
            _intent_outcome_fields _restore_custody _finish_captured_pending
            _finish_captured_running _finalize_cancel_intent_on_miss""".split()
    }
    implemented.update(s3_custody_rows)
    # v7 stream S3, spec 4.3.12: the dispatch table and its miss path carry a
    # semantic delta at their own path — the retired key and the declared miss
    # disposition — so they are implemented rows with a D06 id, not moves.
    s3_semantic_delta_ids = {
        "supervisor/events.py::EVENT_HANDLERS": "D06",
        "supervisor/events.py::dispatch_event": "D06",
        # spec 4.3.6: the supervisor half of the three retired no-op settings keys.
        "supervisor/state.py::status_text": "D04",
        "supervisor/queue.py::init": "D04",
        "supervisor/workers.py::init": "D04",
        ("devtools/benchmarks/terminal_bench/harbor_installed_agent.py"
         "::OuroborosTerminalBenchAgent._container_env"): "D06",
    }
    implemented.update({name: name for name in s3_semantic_delta_ids})
    existing_process_owner_rows.update(s3_semantic_delta_ids)
    registry_extraction_no_facade_rows.update(s3_semantic_delta_ids)
    s3_worker_process_rows = {
        f"supervisor/workers.py::{symbol}": f"supervisor/worker_process.py::{symbol}"
        for symbol in """WORKER_LOG_SINK_SUPPRESSED_TYPES _current_custody_session_id
            _bind_worker_repo_root _prepare_worker_task_runtime worker_main
            _log_worker_crash""".split()
    }
    implemented.update(s3_worker_process_rows)
    # v7 stream L-A: the scope reviewer keeps the run; its prompt budget/window
    # authority and its pack assembly become owners. Every row is a facade row —
    # the parent re-exports the same objects under their historical private names.
    scope_review_extraction_symbols_by_owner = _inv.scope_review_extraction_symbols_by_owner
    scope_review_extraction_rows = {
        f"ouroboros/tools/scope_review.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in scope_review_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    # S3b: the queue's snapshot-path shadow. The ratchet-transition rename is the
    # wip's own row (D11), registered beside the other delta expectations.
    retired_current["supervisor/queue.py::QUEUE_SNAPSHOT_PATH"] = (
        "retired:supervisor.state owns the queue snapshot path; the queue reads it through the module at use time"
    )
    # S3b: the module-handle extraction of the queue (delta D18).
    s3b_queue_handle_symbols_by_owner = _inv.s3b_queue_handle_symbols_by_owner
    s3b_queue_handle_rows = {
        f"supervisor/queue.py::{symbol}": f"supervisor/{owner}::{symbol}"
        for owner, symbols in s3b_queue_handle_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(s3b_queue_handle_rows)
    s3b_queue_no_facade = {
        "supervisor/queue.py::_SKILL_SCHEDULE_SYNC_INTERVAL_SEC",
        "supervisor/queue.py::_last_skill_schedule_sync",
    }
    registry_extraction_no_facade_rows.update(s3b_queue_no_facade)
    # S3b: the module-handle extraction of the worker pool (delta D18).
    s3b_pool_handle_symbols_by_owner = _inv.s3b_pool_handle_symbols_by_owner
    s3b_pool_handle_rows = {
        f"supervisor/workers.py::{symbol}": f"supervisor/{owner}::{symbol}"
        for owner, symbols in s3b_pool_handle_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(s3b_pool_handle_rows)
    # S3b: the retired liveness knobs leave the signatures they were ferried through.
    s3b_retired_signature_rows = {
        "supervisor/queue.py::refresh_timeouts_from_settings":
            "supervisor/queue.py::refresh_timeouts_from_settings",
    }
    implemented.update(s3b_retired_signature_rows)
    existing_process_owner_rows.update(s3b_retired_signature_rows)
    registry_extraction_no_facade_rows.update(s3b_retired_signature_rows)
    s3_semantic_delta_ids["supervisor/queue.py::refresh_timeouts_from_settings"] = "D04"
    retired_current.update({
        "supervisor/queue.py::SOFT_TIMEOUT_SEC":
            "retired:no rail consulted it and its last reader, the owner status line, stopped printing it",
        "supervisor/queue.py::HARD_TIMEOUT_SEC":
            "retired:no rail consulted it and its last reader, the owner status line, stopped printing it",
    })
    retired_delta_ids["supervisor/queue.py::SOFT_TIMEOUT_SEC"] = "D04"
    retired_delta_ids["supervisor/queue.py::HARD_TIMEOUT_SEC"] = "D04"
    implemented.update(w_stream_rows)
    implemented.update(shell_extraction_rows)
    implemented.update(headless_extraction_rows)
    implemented.update(tool_access_extraction_rows)
    # v7 stream L-A: review_helpers keeps the shared plumbing; the reviewer vocabulary
    # and the reviewable-file classification/packs become owners. All facade rows.
    review_helpers_extraction_symbols_by_owner = _inv.review_helpers_extraction_symbols_by_owner
    review_helpers_extraction_rows = {
        f"ouroboros/tools/review_helpers.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in review_helpers_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(scope_review_extraction_rows)
    # v7 stream L-A: review_state keeps the durable store; its record rules and its
    # in-memory ledger become owners. All facade rows.
    review_state_extraction_symbols_by_owner = _inv.review_state_extraction_symbols_by_owner
    review_state_extraction_rows = {
        f"ouroboros/review_state.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in review_state_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(review_helpers_extraction_rows)
    implemented.update(review_state_extraction_rows)
    implemented.update(test_split_rows)
    implemented.update(web_extractions)
    implemented.update(config_extraction_rows)
    implemented["tests/test_repo_health_smoke.py::test_transition_rejects_function_swap_even_at_same_cardinality"] = "tests/test_repo_health_smoke.py::test_transition_allows_a_same_qualname_relocation_but_not_a_swap"
    # v7 stream S, lane S1: the spec 4.3.5 settings seam. One normalization for every
    # reader, one locked read-modify-write for the owner endpoints, one serializer for
    # the three writers, and the start-time mutator removed.
    settings_seam_rows = {
        "ouroboros/gateway/onboarding.py::_settings_fingerprint":
            "ouroboros/gateway/owner_settings.py::settings_document_digest",
        "ouroboros/config.py::load_settings_lock_held":
            "ouroboros/config.py::normalize_settings_raw",
        "ouroboros/gateway/owner_settings.py::_owner_write_settings":
            "ouroboros/gateway/owner_settings.py::_owner_update_settings",
        "ouroboros/config.py::save_settings": "ouroboros/config.py::serialize_settings",
        "ouroboros/packaged_cli.py::_save_settings": "ouroboros/packaged_cli.py::_save_settings",
        "ouroboros/launcher_onboarding.py::prepare_first_run_settings":
            "ouroboros/launcher_onboarding.py::prepare_first_run_settings",
        "tests/test_onboarding_host.py::test_pre_server_normalization_never_creates_the_settings_file":
            "tests/test_onboarding_host.py::test_pre_server_normalization_never_writes_the_settings_file",
        # Lane S2: the last start-time mutator, inside the server lifespan.
        "server.py::lifespan": "server.py::lifespan",
        "tests/test_onboarding_host.py::test_server_boot_normalization_carries_the_same_guard":
            "tests/test_onboarding_host.py::test_server_boot_never_writes_the_settings_file",
    }
    implemented.update(settings_seam_rows)
    # v7 stream T, lane T2: the write/edit/search/forward producers that never left
    # core.py publish their own result code. Same identity, same bytes, and the code
    # IS the answer the adapter already gave for those bytes, so the id is "none".
    t2_core_native_rows = {
        f"ouroboros/tools/core.py::{symbol}": f"ouroboros/tools/core.py::{symbol}"
        for symbol in "_data_write _write_file _edit_text _forward_to_worker".split()
    }
    implemented.update(t2_core_native_rows)
    existing_process_owner_rows.update(t2_core_native_rows)
    registry_extraction_no_facade_rows.update(t2_core_native_rows)
    # v7 lane T2b, owner item A.20 (batch #10): producers whose OBSERVABLE
    # classification the owner changed, so these rows carry the same tool-domain
    # delta id as the T1 cutover rather than "none".
    t2b_owner_delta_rows = {
        "ouroboros/tools/core.py::_send_photo",
        "ouroboros/tools/core.py::_send_video",
        "ouroboros/tools/core.py::_send_file",
        "ouroboros/tools/core.py::_write_file",
        "ouroboros/tools/core.py::_edit_text",
        "ouroboros/tools/core.py::_forward_to_worker",
        "ouroboros/tools/core.py::_data_read",
    }
    # v7 lane A21, owner item A.21 (batch #13): the control producers whose
    # OBSERVABLE classification the owner changed — refusals that reported ok — so
    # these rows carry the same tool-domain delta id as the T1 cutover, not "none".
    # The lane's other rows stay "none": publishing the code the adapter already
    # assigned to the same text moves nothing a consumer can see.
    a21_owner_delta_rows = {
        "ouroboros/tools/control.py::_promote_chat_to_task",
        "ouroboros/tools/control.py::_route_to_project",
        "ouroboros/tools/control.py::_steer_task",
        "ouroboros/tools/control.py::_request_deep_self_review",
        "ouroboros/tools/control.py::_update_scratchpad",
        "ouroboros/tools/control.py::_update_identity",
        "ouroboros/tools/control.py::_send_user_message",
        "ouroboros/tools/control.py::_switch_model",
        "ouroboros/tools/control.py::_schedule_task",
        "ouroboros/tools/control.py::_get_task_result",
    }
    # The ratchet relocation contract renamed its own pin in place (commit 73360232)
    # without recording the rename; the row is the ledger half of that change.
    ratchet_relocation_rename = {
        "tests/test_repo_health_smoke.py::test_transition_rejects_function_swap_even_at_same_cardinality":
            "tests/test_repo_health_smoke.py::test_transition_allows_a_same_qualname_relocation_but_not_a_swap",
    }
    implemented.update(ratchet_relocation_rename)
    existing_process_owner_rows.update(ratchet_relocation_rename)
    registry_extraction_no_facade_rows.update(ratchet_relocation_rename)
    existing_process_owner_rows.update(settings_seam_rows)
    implemented.update(server_extraction_rows)
    # v7 stream S lane S2, spec 4.3.11 (Emergency Stop 2A): execute_panic_stop keeps
    # its path, name and public identity; only how it learns the bound port changes.
    s2_panic_delta_rows = {
        "ouroboros/server_control.py::execute_panic_stop":
            "ouroboros/server_control.py::execute_panic_stop",
    }
    implemented.update(s2_panic_delta_rows)
    existing_process_owner_rows.update(s2_panic_delta_rows)
    # No facade row: the symbol keeps its own path and name, so there is nothing to
    # re-export — only its keyword surface gained an optional argument.
    # Integration fix (D13): supervisor/git_ops pre-init module defaults follow the
    # environment-aware roots from ouroboros.config instead of a hardcoded
    # ~/Ouroboros; in-place, no facade — same shape as the panic row above.
    git_ops_delta_rows = {
        "supervisor/git_ops.py::DRIVE_ROOT": "supervisor/git_ops.py::DRIVE_ROOT",
        "supervisor/git_ops.py::REPO_DIR": "supervisor/git_ops.py::REPO_DIR",
    }
    implemented.update(git_ops_delta_rows)
    existing_process_owner_rows.update(git_ops_delta_rows)
    # v7 stream L-A lane L2b: verbatim extraction of the review substrate's record,
    # verdict and projection owners out of ouroboros/review_substrate.py.
    l2b_review_extraction_symbols_by_owner = {
        "review_records.py": "ReviewSlot ReviewRequest ReviewActorRecord ReviewRunResult HARDNESS_ADVISORY_VISIBLE HARDNESS_LABEL_ONLY HARDNESS_HARD_GATE",
        "review_verdict.py": "_TIER_ORDER _CRITERION_STATUSES _criteria_have_supported_evidence _criteria_shape_valid _contributing_actors aggregate_outcome_tier task_acceptance_is_clean DIALOGUE_CONTINUE DIALOGUE_UNREACHABLE DIALOGUE_STABLE_DISAGREEMENT DIALOGUE_STATUS_VALUES _contract_valid_actors aggregate_dialogue_status _unresolved_evidence_ref_labels panel_reason dissent_findings build_improvement_capsule",
        "review_projection.py": "_transport_error_status _public_review_reason _review_actor_projection _response_ref_projection _review_enforcement_impact _review_panel_id build_review_binding compact_review_projection",
    }
    l2b_review_extraction_rows = {
        f"ouroboros/review_substrate.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in l2b_review_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(l2b_review_extraction_rows)
    l2b_evidence_extraction_symbols = (
        "collect_turn_diff _ACCEPT_RESULT_CAP _ACCEPT_ARGS_CAP _ACCEPT_NOTES_CAP "
        "_ACCEPT_TRAJECTORY_MAX_CALLS _ACCEPT_ARTIFACT_PREVIEW_CAP _ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES "
        "_ACCEPT_TOTAL_BUDGET _ACCEPT_OBLIGATIONS_MAX _ACCEPT_RETRIEVAL_URLS_MAX obligation_is_pending "
        "_accept_obligation_row task_acceptance_evidence_revision _accept_redact_cap _accept_task_contract "
        "_accept_protected_set _accept_verification_summary _accept_receipt_exhibits _accept_effective_claims "
        "_accept_claim_support_refs _accept_trajectory _accept_artifact_manifest _accept_enforce_budget "
        "_owner_content_projection _accept_owner_directives _ACCEPT_DELTA_CHILD_CAP _accept_capability_deltas"
    )
    l2b_evidence_extraction_rows = {
        f"ouroboros/review_evidence.py::{symbol}":
            f"ouroboros/review_evidence_sections.py::{symbol}"
        for symbol in l2b_evidence_extraction_symbols.split()
    }
    implemented.update(l2b_evidence_extraction_rows)
    l2b_skill_review_symbols_by_owner = {
        "skill_review_packs.py": "_SKILL_PACK_TOKEN_HEADROOM _skill_pack_token_budget _LOADABLE_BINARY_EXTENSIONS _SkillFileOverBudget _SkillFileUnreadable _SkillBinaryPayload _read_skill_text _build_skill_file_packs",
        "skill_review_rebuttals.py": "_review_history_path _accepted_rebuttals_path _load_accepted_rebuttals _persist_rebuttal_flips _fail_items_from_history_entry _record_accepted_rebuttal _build_skill_review_history_section _convergence_hint _render_accepted_rebuttals_section",
        "skill_review_prompt.py": "_SKILL_CHECKLIST_SECTION _SKILL_REVIEW_ITEMS _CRITICAL_ITEMS _load_governance_artifact _REPO_ROOT _build_review_prompt _emit_skill_advisory_warning _run_skill_advisory_pre_review _review_wave_budget_block _build_review_prompt_for_attempt",
        "skill_review_output.py": "render_skill_review_block _extract_actor_findings _parse_json_array _aggregate_status",
    }
    l2b_skill_review_rows = {
        f"ouroboros/skill_review.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in l2b_skill_review_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(l2b_skill_review_rows)
    # v7 stream S, lane S4: ouroboros/extension_loader.py split into owner leaves.
    # The loader keeps the extension lifecycle; the registries, the namespace
    # encoding, the child-catalog re-validation, the staged import trees, the
    # liveness projection and the PluginAPI object each get one owner.
    s4_extension_symbols_by_owner = _inv.s4_extension_symbols_by_owner
    s4_extension_rows = {
        f"ouroboros/extension_loader.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in s4_extension_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(s4_extension_rows)
    # v7 stream S, lane S4: ouroboros/tools/control.py split into owner leaves.
    # get_tools() stays with the catalog owner; every handler and helper it wires
    # gets one. The leaves carry the parent's hot-code label, nothing more.
    s4_control_symbols_by_owner = {
        "control_events.py": "_SCHEDULE_EMIT_LOCK _PROMOTE_CONFIRM_TIMEOUT_SEC _PROMOTE_CONFIRM_POLL_SEC _emit_control_event _promotion_pool_disabled_from_snapshot _routing_status_root _wait_for_promotion_admission _wait_for_routing_annotation _emit_and_wait_for_routing",
        "control_routing.py": "_attach_origin_from_metadata _attach_swarm_intent _cached_swarm_handoff _finish_swarm_handoff _promote_chat_to_task _list_projects _route_to_project _steer_task",
        "control_subagent_spec.py": "VALID_SUBTASK_MEMORY_MODES schedule_subagent_properties schedule_subagent_param_names _INTERNAL_SCHEDULE_OPTIONS _validated_schedule_fields RETIRED_SCHEDULE_PARAMS",
        "control_scheduling.py": "_record_scheduled_subagent _emit_swarm_fanout _subagent_slot_note _capability_mismatch_message _finalize_schedule_emission _build_acting_constraint _select_subagent_constraint _populate_subagent_event_extras _prepare_child_drive _earliest_deadline_at _build_child_subagent_contract _resolve_executor_ref _inherited_workspace_from_active_repo _schedule_task",
        "control_runtime.py": "_evolution_restart_block_reason _request_restart _set_tool_timeout _promote_to_stable _request_deep_self_review _chat_history _update_scratchpad _send_user_message _update_identity _toggle_evolution _toggle_consciousness _switch_model",
        "control_task_results.py": "disclosable_capability_delta _subtask_outcome_summary _get_task_result _wait_attention_poll cache_horizon_note _wait_for_task _count_live_sibling_children _UNMINTED_WAIT_GRACE_SEC _unminted_wait_ids _children_roster_projection _wait_for_tasks",
    }
    s4_control_rows = {
        f"ouroboros/tools/control.py::{symbol}": f"ouroboros/tools/{owner}::{symbol}"
        for owner, symbols in s4_control_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(s4_control_rows)
    # The broadcaster slot is REBOUND by set_ws_broadcaster, so a re-export would be
    # a snapshot that stops tracking its owner: the setter is the facade, the binding
    # is not.
    registry_extraction_no_facade_rows.add("ouroboros/extension_loader.py::_ws_broadcaster")
    existing_process_owner_rows.update(test_split_rows)
    registry_extraction_no_facade_rows.update(s2_panic_delta_rows)
    registry_extraction_no_facade_rows.update(git_ops_delta_rows)
    registry_extraction_no_facade_rows.update(set(test_split_rows) - test_split_facade_rows)
    registry_extraction_no_facade_rows.update(old for old in web_extractions if "::createChatInstance." in old)
    # W3: a module-private chat.js helper that moved with its only caller — never exported, so no facade.
    registry_extraction_no_facade_rows.add("web/modules/chat.js::projectIdFromTask")
    # W3 wave D: never-exported chat.js module constants that moved with their only consumers.
    registry_extraction_no_facade_rows.update({
        "web/modules/chat.js::MAX_PENDING_ATTACHMENTS",
        "web/modules/chat.js::MAX_ATTACHMENT_FILE_BYTES",
        "web/modules/chat.js::MAX_PENDING_ATTACHMENT_BYTES",
        "web/modules/chat.js::CHAT_STORAGE_KEY",
    })
    registry_extraction_no_facade_rows.update(w3_chat_primitive_rows)
    registry_extraction_no_facade_rows.add("tests/test_repo_health_smoke.py::test_transition_rejects_function_swap_even_at_same_cardinality")
    # S6 (delegation/cancellation targeted fixes): no symbol moves — every row
    # is the SAME identity with a stated behaviour delta, so the owner is the
    # old path and the facade cell is "-". D03 is the one class these rows
    # share: a durable-registry mutator that could not read its file stops
    # answering as if the record were absent.
    s6_delta_rows = {
        identity: identity for identity in (
            "ouroboros/cancel_intents.py::claim_intent",
            "ouroboros/cancel_intents.py::release_claim",
            "ouroboros/cancel_intents.py::settle_intent",
            "ouroboros/cancel_intents.py::mark_intent_scope",
            "ouroboros/cancel_intents.py::mark_finalize_control_drained",
            "ouroboros/cancel_intents.py::_load_intents",
            "ouroboros/subagent_worktrees.py::_load_registry",
            "ouroboros/subagent_worktrees.py::find_execution_snapshot",
            "ouroboros/subagent_worktrees.py::prune_execution_snapshots",
            "ouroboros/subagent_worktrees.py::prune_orphans",
            "ouroboros/subagent_worktrees.py::remove_worktree",
            "ouroboros/subagent_worktrees.py::remove_execution_snapshot",
            "ouroboros/subagent_worktrees.py::provision_worktree",
            "ouroboros/subagent_worktrees.py::provision_payload_snapshot",
            "ouroboros/subagent_worktrees.py::provision_execution_snapshot",
        )
    }
    s6_disclosure_rows = {
        identity: identity for identity in (
            "ouroboros/cancel_intents.py::_SCHEMA_VERSION",
            "ouroboros/subagent_worktrees.py::_KIND_DELEGATED_EXEC",
            "ouroboros/task_finalization.py::register_final_answer_owed",
        )
    }
    s6_rows = {**s6_delta_rows, **s6_disclosure_rows}
    implemented.update(s6_rows)
    existing_process_owner_rows.update(s6_rows)
    registry_extraction_no_facade_rows.update(s6_rows)
    # v7 stream S test-giant theme splits (lane S7a): source module -> {owner path: moved symbols}.
    # A moved test, fixture or stub is owned by the sibling module that now hosts it; a test-private
    # import binding keeps the canonical production provider, which these splits never moved.
    s7a_test_split_symbols_by_owner = _inv.s7a_test_split_symbols_by_owner
    s7a_test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in s7a_test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    s7a_test_split_facade_rows = {
        "tests/test_context.py::_make_health_env",
        "tests/test_delegated_run_isolation.py::_git",
        "tests/test_delegated_run_isolation.py::_isolated_entry",
        "tests/test_delegated_run_isolation.py::_nanny_ctx",
        "tests/test_delegated_run_isolation.py::_seed_target",
        "tests/test_delegated_subagent_transport.py::_gateway",
        "tests/test_delegated_subagent_transport.py::_owned_gateway_uses_each_test_transport",
        "tests/test_delivery_forced_finalization.py::_forced_test_context",
        "tests/test_extensions_api.py::_clean_extensions",
        "tests/test_extensions_api.py::_make_client",
        "tests/test_extensions_api.py::_stop_patches",
        "tests/test_extensions_api.py::_write_ext",
        "tests/test_promote_chat_flow.py::_isolated_projects_root",
        "tests/test_runtime_mode_elevation.py::_make_drive_ctx",
        "tests/test_runtime_mode_elevation.py::_seed_disk",
        "tests/test_workspace_executor.py::_init_repo",
    }
    implemented.update(s7a_test_split_rows)
    existing_process_owner_rows.update(s7a_test_split_rows)
    registry_extraction_no_facade_rows.update(set(s7a_test_split_rows) - s7a_test_split_facade_rows)
    # v7 stream S test-giant theme splits (lane S7b): source module -> {owner path: moved symbols}.
    # Same shape as the S7a block: the sibling module that hosts a test, fixture or helper owns it.
    # A symbol minted on the wip line after the merge base has no baseline identity, so it carries
    # no ledger row (tests/test_evolution_state_integrity_v3.py::_patch_commit_seam is the one case).
    s7b_test_split_symbols_by_owner = _inv.s7b_test_split_symbols_by_owner
    s7b_test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in s7b_test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    s7b_test_split_facade_rows = {
        "tests/test_skill_exec.py::_build_skill",
        "tests/test_skill_exec.py::_make_ctx",
        "tests/test_skill_exec.py::_mark_reviewed_and_enabled",
        "tests/test_skill_exec.py::_valid_script_manifest",
        "tests/test_skill_review.py::_NEW_SKILL_REVIEW_PASS_ITEMS",
        "tests/test_skill_review.py::_build_skill",
        "tests/test_skill_review.py::_make_actor",
        "tests/test_skill_review.py::_make_ctx",
        "tests/test_skill_review.py::_pass_array_for_script_skill",
        "tests/test_skill_review.py::_patch_review",
        "tests/test_skill_loader.py::_valid_script_manifest",
        "tests/test_skill_loader.py::_write_skill",
    }
    implemented.update(s7b_test_split_rows)
    existing_process_owner_rows.update(s7b_test_split_rows)
    registry_extraction_no_facade_rows.update(set(s7b_test_split_rows) - s7b_test_split_facade_rows)
    # v7 stream S test-giant theme splits (lane W5): source module -> {owner path: moved symbols}.
    # Every row is a relocation inside the test tree: a moved test, stub, fixture or argv builder is
    # owned by the sibling module that now hosts it. A facade cell appears only where the parent still
    # imports the moved helper by its old name; a test the parent no longer mentions carries "-".
    w5_test_split_symbols_by_owner = _inv.w5_test_split_symbols_by_owner
    w5_test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in w5_test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    w5_test_split_facade_rows = {
        "tests/test_devtools_benchmarks.py::REPO_ROOT",
        "tests/test_devtools_benchmarks.py::_git_commit_all",
        "tests/test_devtools_benchmarks.py::_git_repo",
        "tests/test_git_ops_recovery.py::_git",
        "tests/test_git_ops_recovery.py::_history_repo",
        "tests/test_preflight_runner.py::REPO_ROOT",
        "tests/test_preflight_runner.py::_PREFLIGHT_PLUGIN_PROBLEMS",
        "tests/test_preflight_runner.py::_REAL_SPAWN_SKIP_REASON",
        "tests/test_preflight_runner.py::_REQUIRE_PLUGINS_ENV",
        "tests/test_ui_smoke_playwright.py::_free_port",
        "tests/test_ui_smoke_playwright.py::_wait_health",
    }
    implemented.update(w5_test_split_rows)
    existing_process_owner_rows.update(w5_test_split_rows)
    registry_extraction_no_facade_rows.update(set(w5_test_split_rows) - w5_test_split_facade_rows)
    # v7 stream S test-giant theme splits (lane TS1): source module -> {owner path: moved symbols}.
    # Same shape as the S7b block: the sibling module that hosts a moved test, fixture or helper
    # owns it. A facade cell appears only where a helper keeps resolving under its old module
    # name for historical importers; a symbol the parent no longer mentions carries "-".
    ts1_test_split_symbols_by_owner = _inv.ts1_test_split_symbols_by_owner
    ts1_test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in ts1_test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    # The extension_loader parent re-exports the shared helper module's six symbols so the
    # five pre-existing importer suites keep resolving them under the old module name.
    ts1_test_split_facade_rows = {
        "tests/test_extension_loader.py::_add_fake_native_dep",
        "tests/test_extension_loader.py::_clear_loader_state",
        "tests/test_extension_loader.py::_isolated_site_packages_dir",
        "tests/test_extension_loader.py::_mark_isolated_deps_installed",
        "tests/test_extension_loader.py::_prepare_extension",
        "tests/test_extension_loader.py::_write_ext_skill",
    }
    implemented.update(ts1_test_split_rows)
    existing_process_owner_rows.update(ts1_test_split_rows)
    registry_extraction_no_facade_rows.update(set(ts1_test_split_rows) - ts1_test_split_facade_rows)
    # v7 lane TS2: the review-family test-giant theme splits: source module -> {owner path: moved
    # symbols}. Same shape as the S7a/S7b/W5 blocks: the sibling module that hosts a moved test,
    # fixture or helper owns it, and a facade cell appears only where the parent still imports
    # the moved helper by its old name. The facade sets and the import-binding side rows are
    # data and live beside the symbol maps in tests/_v7_ledger_inventories.py (byte-ratchet idiom).
    ts2_test_split_rows = {f"{source}::{symbol}": f"{owner}::{symbol}" for source, owners in _inv.ts2_test_split_symbols_by_owner.items() for owner, symbols in owners.items() for symbol in symbols.split()}
    ts2_rows = {**ts2_test_split_rows, **_inv.ts2_binding_rows}
    ts2_facade_rows = _inv.ts2_test_split_facade_rows | _inv.ts2_binding_facade_rows
    implemented.update(ts2_rows)
    existing_process_owner_rows.update(ts2_rows)
    registry_extraction_no_facade_rows.update(set(ts2_rows) - ts2_facade_rows)
    # v7 stream L: llm.py splits into ten owner leaves. Module-level names keep a
    # facade on llm.py; LLMClient members move into owner mixins the class
    # composes, so the class inherits the exact same function objects.
    llm_extraction_symbols_by_owner = {
        "llm_attempt.py": "_CACHE_TTL_SECONDS _VALID_CACHE_TTLS _applied_payload_cache_ttl _attempt_request _candidate_before_dispatch _canonical_candidate_bytes _execute_candidate _execute_candidate_async _is_structured_context_overflow_body _is_structured_context_overflow_exception _physical_candidate _route_normalizes_cache_breakpoints _structured_error_values cache_ttl_seconds supports_message_cache_control",
        "llm_capability_policy.py": "_MANDATORY_VALUE_MARKERS _OPTIONAL_DROPPABLE_PARAMS _OPTIONAL_SAMPLING_PARAMS normalize_reasoning_effort",
        "llm_routing.py": "_OR_PROVIDER_PRESETS _resolve_or_provider",
        "llm_messages.py": "_reasoning_signature_portable_across_or_providers",
        "llm_local.py": "LocalContextTooLargeError _LOCAL_COMPACTION_MODES _compact_local_text _compact_markdown_sections _estimate_message_chars _split_markdown_sections",
        "llm_openai_compatible.py": "_FALSE_LIKE_ENV_VALUES",
        "llm_pricing.py": "add_usage fetch_cloudru_pricing fetch_openrouter_pricing",
    }
    llm_extraction_rows = {
        f"ouroboros/llm.py::{symbol}": f"ouroboros/{owner}::{symbol}"
        for owner, symbols in llm_extraction_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    llm_mixin_symbols_by_owner = _inv.llm_mixin_symbols_by_owner
    llm_mixin_rows = {
        f"ouroboros/llm.py::LLMClient.{symbol}": f"ouroboros/{owner}::{mixin}.{symbol}"
        for (owner, mixin), symbols in llm_mixin_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(llm_extraction_rows)
    implemented.update(llm_mixin_rows)
    registry_extraction_no_facade_rows.update(llm_mixin_rows)
    # Shared delta registry: D09 is spec 4.3.2 (LLM local retry) and its
    # typed-refusal companion. The id rides the row of each symbol it changed.
    llm_semantic_delta_ids = {"ouroboros/llm.py::LLMClient._chat_local": "D09"}
    # The recovery ladder stops consuming a typed policy refusal (same id).
    llm_semantic_delta_ids.update({
        f"ouroboros/llm.py::LLMClient.{symbol}": "D09"
        for symbol in (
            "_create_chat_completion_with_retries",
            "_create_chat_completion_with_retries_async",
            "_retry_without_prompt_cache_parameter",
            "_openrouter_signature_retry_kwargs",
            "_retry_without_optional_sampling",
        )
    })
    # Upstream adoption through v6.104.0: the release carved web/modules/chat_activity.js
    # out of chat.js, so at the new merge base four identities DECLARE there. Only their
    # baseline address moves — owner, body and facade shape are what they already were.
    merge_adopt_rebased_declarations = {
        "web/modules/chat.js::createChatInstance.buildMessageKey": "web/modules/chat_activity.js::buildMessageKey",
        "web/modules/chat.js::createChatInstance.formatMsgTime": "web/modules/chat_activity.js::formatMsgTime",
        "web/modules/chat.js::createChatInstance.routingAnnotationText": "web/modules/chat_activity.js::routingAnnotationText",
    }
    for _old, _new in merge_adopt_rebased_declarations.items():
        web_extractions[_new] = web_extractions.pop(_old)
        implemented[_new] = implemented.pop(_old)
        registry_extraction_no_facade_rows.discard(_old)
        registry_extraction_no_facade_rows.add(_new)
    retired_current["web/modules/chat_activity.js::optionalFiniteNumber"] = retired_current.pop(
        "web/modules/chat.js::optionalFiniteNumber")
    # The eleven live-card projections upstream's chat_activity.js re-publishes keep the
    # domain owners this line already gave them; chat_activity.js re-exports each name, so
    # the facade cell is the old identity and no historical importer notices the move.
    merge_adopt_web_facade_rows = {
        f"web/modules/chat_activity.js::{symbol}": f"web/modules/{owner}::{symbol}"
        for owner, symbols in {
            "chat_card_state.js": "COLLAPSED_ACTIVITY_MAX boundActivityPreview clearStickyCardState isTerminalTaskPhase liveLineRowToggleKey projectCollapsedActivity",
            "costs.js": "headerBudgetPresentation mergeStickyCostMeta taskCostMeta taskCostProjection",
            "utils.js": "rawTimestampEpoch",
        }.items()
        for symbol in symbols.split()
    }
    # The shipped router profile follows the settings vocabulary it fills in: provider_models
    # imports that leaf, so the profile cannot be declared above it. provider_models and the
    # config facade both re-export the two names like every other settings-vocabulary member.
    merge_adopt_facade_rows = {
        **merge_adopt_web_facade_rows,
        "ouroboros/provider_models.py::OPENROUTER_DEFAULTS": "ouroboros/settings_defaults.py::OPENROUTER_DEFAULTS",
        "ouroboros/provider_models.py::OPENROUTER_REVIEW_DEFAULTS": "ouroboros/settings_defaults.py::OPENROUTER_REVIEW_DEFAULTS",
        "ouroboros/config.py::OPENROUTER_DEFAULTS": "ouroboros/settings_defaults.py::OPENROUTER_DEFAULTS",
        "ouroboros/config.py::OPENROUTER_REVIEW_DEFAULTS": "ouroboros/settings_defaults.py::OPENROUTER_REVIEW_DEFAULTS",
        "ouroboros/tools/plan_render.py::PLAN_REVIEW_CONTROL_PREFIX": "ouroboros/tools/review_synthesis.py::PLAN_REVIEW_CONTROL_PREFIX",
    }
    merge_adopt_no_facade_rows = {
        "tests/test_claudexor_owned_daemon.py::test_login_create_passes_the_daemon_400_verdict_through":
            "tests/test_claudexor_login_accounts.py::test_login_create_passes_the_daemon_400_verdict_through",
        "tests/test_heartbeat_presentation.py::test_retired_planning_heartbeat_key_is_silent_and_dropped_on_load":
            "tests/test_heartbeat_presentation.py::test_retired_planning_heartbeat_default_is_quiet_but_custom_value_is_loud",
    }
    merge_adopt_rows = {**merge_adopt_facade_rows, **merge_adopt_no_facade_rows}
    implemented.update(merge_adopt_rows)
    existing_process_owner_rows.update(merge_adopt_rows)
    registry_extraction_no_facade_rows.update(merge_adopt_no_facade_rows)
    # Upstream adoption through v6.105.1 (unified accounts, delegation substrate,
    # rotation visibility). Three kinds of row land together and are kept as data in
    # _v7_ledger_inventories: the base's OWN extraction of the executor-note pair,
    # which v7 answers by keeping its agent_dispatch home; the leaves v7 had to carve
    # when upstream's additions pushed subagents.py, context.py and the plan-review
    # engine suite back over the 1500-line ceiling; and the upstream test symbols that
    # landed in themed v7 suites instead of the parents they were written against.
    merge_adopt_v6105_facade_rows = dict(_inv.merge_adopt_v6105_facade_rows)
    merge_adopt_v6105_no_facade_rows = dict(_inv.merge_adopt_v6105_no_facade_rows)
    merge_adopt_v6105_rows = {**merge_adopt_v6105_facade_rows, **merge_adopt_v6105_no_facade_rows}
    implemented.update(merge_adopt_v6105_rows)
    existing_process_owner_rows.update(merge_adopt_v6105_rows)
    registry_extraction_no_facade_rows.update(merge_adopt_v6105_no_facade_rows)
    # The FINAL upstream cutoff (PR #257). Its only ledger-visible move is the one
    # the size gate forced: upstream's launcher.py growth crossed this branch's
    # 1500-line ceiling, so the Windows runtime preparation left launcher.py inside
    # the merge — verbatim, re-exported under the same names.
    merge_adopt_pr257_facade_rows = dict(_inv.merge_adopt_pr257_facade_rows)
    merge_adopt_pr257_no_facade_rows = dict(_inv.merge_adopt_pr257_no_facade_rows)
    merge_adopt_pr257_rows = {**merge_adopt_pr257_facade_rows, **merge_adopt_pr257_no_facade_rows}
    implemented.update(merge_adopt_pr257_rows)
    existing_process_owner_rows.update(merge_adopt_pr257_rows)
    registry_extraction_no_facade_rows.update(merge_adopt_pr257_no_facade_rows)
    # Lane followup (owner decision 2026-08-19, answer "B"): the adopted one-shot
    # follow-up tool joins the T1 typed-result cutover IN PLACE — same path, same
    # name, same sentences — so nothing is re-exported and the D02 id below carries
    # the owner-approved retyping of refusals that used to report ok (item A.22).
    followup_native_result_rows = dict(_inv.followup_native_result_rows)
    implemented.update(followup_native_result_rows)
    existing_process_owner_rows.update(followup_native_result_rows)
    registry_extraction_no_facade_rows.update(followup_native_result_rows)
    # D02: the upstream v6.103.0 silent-ignore pin for a vacuous disposition is
    # re-pointed at the ratified disclosure contract (note + preserved native meta);
    # the replacement is named in the ledger so the SSOT, not just the commit
    # message, discloses that an upstream pin was replaced.
    _d02_repinned_old = (
        "tests/test_plan_review.py::TestPlanReviewDispositionEnvelope"
        ".test_vacuous_disposition_beside_a_plan_is_ignored"
    )
    implemented[_d02_repinned_old] = (
        "tests/test_plan_review.py::TestPlanReviewDispositionEnvelope"
        ".test_vacuous_disposition_beside_a_plan_is_ignored_with_disclosure"
    )
    existing_process_owner_rows.add(_d02_repinned_old)
    registry_extraction_no_facade_rows.add(_d02_repinned_old)
    s3_semantic_delta_ids[_d02_repinned_old] = "D02"
    # D31 (owner decision 2026-08-19, superseding the spec 1.14-2 batch): the
    # contributor lane always executes the target base's review machinery, so the
    # per-proposal trust classifier — the hand-list, its anchors-plus-name-rule
    # successor and the base-flow import closure — retires whole, and so does the
    # boundary characterization that proved its membership.
    d31_rows = {
        "scripts/run_external_review.py::_REVIEW_SUBSTRATE_PATHS":
            "retired:the contributor lane always executes the target base's review machinery, so no diff is classified",
        "tests/test_external_review_script.py::_REVIEW_SUBSTRATE_PATHS":
            "retired:the boundary characterization retires with the classifier it proved",
        "tests/test_external_review_script.py::test_contributor_trust_boundary_covers_functional_review_dependencies":
            "retired:no boundary classifies the functional review dependencies any more",
        "tests/test_external_review_script.py::test_contributor_snapshot_flags_transitive_review_substrate_changes":
            "retired:the snapshot carries no substrate flag left to characterize",
    }
    retired_current.update(d31_rows)
    for _d31_old in d31_rows:
        retired_delta_ids[_d31_old] = "D31"
    # The receipt half of the fail-closed characterization survives the retired
    # trust half, under the name that now describes it.
    _d31_renamed = (
        "tests/test_external_review_script.py"
        "::test_contributor_outcome_fails_closed_on_receipt_or_trust_drift"
    )
    implemented[_d31_renamed] = (
        "tests/test_external_review_script.py"
        "::test_contributor_outcome_fails_closed_on_receipt_drift_only"
    )
    existing_process_owner_rows.add(_d31_renamed)
    registry_extraction_no_facade_rows.add(_d31_renamed)
    s3_semantic_delta_ids[_d31_renamed] = "D31"
    # D04 rides the heartbeat row: the retired knob is stripped from the settings document,
    # so the environment is the only surviving source and the test says which half is quiet.
    s3_semantic_delta_ids[
        "tests/test_heartbeat_presentation.py::test_retired_planning_heartbeat_key_is_silent_and_dropped_on_load"
    ] = "D04"
    # Both sides fixed the same squatted-literal defect; the per-test tmp_path isolation
    # already in this tree makes upstream's unique-name helper redundant.
    retired_current.update({
        f"tests/test_telegram_miniapp_{module}.py::_nonexistent_state_dir":
            "retired:each scenario takes its own pytest tmp_path state dir, so no shared-host name can be squatted"
        for module in ("companion", "lifecycle")
    })
    # v7 stream L lane L-B: loop.py splits into cohesive owner leaves. Handle rows
    # read rebindable loop globals through the call-time handle _loop() (delta D33,
    # set pinned in tests/test_module_handle_extraction.py); verbatim rows moved
    # byte-identical. Lane L3 then spent the TEMPORARY private half of that facade
    # (spec 4.3-15): a retired name carries facade "-" (any consumer left outside
    # the owning leaf imports the owner directly), and if the same edit removed
    # its last _loop() read it is a verbatim row again, because un-substituting
    # the handle restores the merge-base text exactly.
    lb_loop_verbatim_rows = {
        f"ouroboros/loop.py::{symbol}": f"{owner}::{symbol}"
        for owner, symbols in _inv.lb_loop_verbatim_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    lb_loop_handle_rows = {
        f"ouroboros/loop.py::{symbol}": f"{owner}::{symbol}"
        for owner, symbols in _inv.lb_loop_handle_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    lb_loop_l3_retired_rows = {
        f"ouroboros/loop.py::{symbol}": f"{owner}::{symbol}"
        for owner, symbols in _inv.lb_loop_l3_retired_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    assert lb_loop_l3_retired_rows.keys() <= (lb_loop_verbatim_rows | lb_loop_handle_rows).keys()
    # L3 also re-homed the loop-private TEST imports: the characterization binds the
    # leaf that defines the symbol instead of the loop re-export. The test module
    # still exports the name, so these rows keep their facade; only the provider moved.
    l3_repointed_test_rows = {
        f"{test}::{symbol}": f"{owner}::{symbol}"
        for test, owners in _inv.l3_repointed_test_import_owners.items()
        for owner, symbols in owners.items()
        for symbol in symbols.split()
    }
    implemented.update(lb_loop_verbatim_rows)
    implemented.update(lb_loop_handle_rows)
    implemented.update(l3_repointed_test_rows)
    existing_process_owner_rows.update(lb_loop_verbatim_rows)
    existing_process_owner_rows.update(lb_loop_handle_rows)
    existing_process_owner_rows.update(l3_repointed_test_rows)
    # v7 lane 1A: the update_merge planning/materialization cluster moves to
    # supervisor/update_merge_plan.py; every moved name keeps its update_merge
    # facade re-export. _git_run moved byte-identical (delta none); the three
    # bodies hosting the carrier-engine insertion points ride D34 — the
    # owner-ratified (batch №8 answer 6=A / spec §1.9-10) span-substitution
    # resolver applied before write-tree — and the two of them that read
    # monkeypatch-addressable parent bindings do so through the call-time
    # handle _um() (set pinned in tests/test_module_handle_extraction.py).
    lane1a_update_plan_rows = {
        f"supervisor/update_merge.py::{symbol}": f"supervisor/update_merge_plan.py::{symbol}"
        for symbol in (
            "_git_run", "_build_clean_merge_commit",
            "plan_managed_update_merge", "materialize_assisted_merge_live",
        )
    }
    lane1a_carrier_delta_rows = {
        "supervisor/update_merge.py::_build_clean_merge_commit",
        "supervisor/update_merge.py::plan_managed_update_merge",
        "supervisor/update_merge.py::materialize_assisted_merge_live",
    }
    implemented.update(lane1a_update_plan_rows)
    existing_process_owner_rows.update(lane1a_update_plan_rows)
    for _lane1a_old in lane1a_carrier_delta_rows:
        s3_semantic_delta_ids[_lane1a_old] = "D34"
    # v7 lane G1: the supervisor/git_ops.py ownership split — each cluster moves
    # to its own supervisor/git_ops_*.py owner and every moved name keeps its
    # git_ops facade re-export. Bodies reading rebindable/monkeypatch-addressable
    # git_ops globals do so through the call-time handle _go() and ride D35 (the
    # ratified §1.9-1 module-handle mechanism with the G1 stream's own id);
    # bodies with no such reads move verbatim (delta none).
    g1_git_ops_handle_rows = {
        f"supervisor/git_ops.py::{symbol}": f"supervisor/{owner}::{symbol}"
        for owner, symbols in _inv.g1_git_ops_handle_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    g1_git_ops_verbatim_rows = {
        f"supervisor/git_ops.py::{symbol}": f"supervisor/{owner}::{symbol}"
        for owner, symbols in _inv.g1_git_ops_verbatim_symbols_by_owner.items()
        for symbol in symbols.split()
    }
    implemented.update(g1_git_ops_handle_rows)
    implemented.update(g1_git_ops_verbatim_rows)
    existing_process_owner_rows.update(g1_git_ops_handle_rows)
    existing_process_owner_rows.update(g1_git_ops_verbatim_rows)
    # v7 lane DEL1: the delegate family splits into cohesive owner leaves; every
    # moved name keeps its parent facade re-export. Handle rows ride delta D36
    # (sets pinned in tests/test_module_handle_extraction.py); the rest verbatim.
    del1_verbatim_rows = {
        f"{parent}::{symbol}": f"{owner}::{symbol}"
        for parent, owners in _inv.del1_verbatim_symbols_by_parent.items()
        for owner, symbols in owners.items() for symbol in symbols.split()
    }
    del1_handle_rows = {
        f"{parent}::{symbol}": f"{owner}::{symbol}"
        for parent, owners in _inv.del1_handle_symbols_by_parent.items()
        for owner, symbols in owners.items() for symbol in symbols.split()
    }
    implemented.update(del1_verbatim_rows)
    implemented.update(del1_handle_rows)
    existing_process_owner_rows.update(del1_verbatim_rows)
    existing_process_owner_rows.update(del1_handle_rows)
    # v7 stream L lane L-C: review-stack owner leaves (parent-aware maps live in
    # _v7_ledger_inventories). Handle rows read rebindable parent facade bindings
    # through the call-time handles _rev()/_car() (delta D37, sets pinned in
    # tests/test_module_handle_extraction.py); verbatim rows moved byte-identical.
    def _lc_rows(maps):
        return {f"{parent}::{s}": f"{owner}::{s}" for parent, owners in maps.items()
                for owner, symbols in owners.items() for s in symbols.split()}

    lc_review_verbatim_rows = _lc_rows(_inv.lc_review_verbatim_symbols_by_owner)
    lc_review_handle_rows = _lc_rows(_inv.lc_review_handle_symbols_by_owner)
    for _lc in (lc_review_verbatim_rows, lc_review_handle_rows):
        implemented.update(_lc)
        existing_process_owner_rows.update(_lc)
    # v7 lane L-C2: one cohesive cluster of each of agent.py,
    # agent_task_pipeline.py and usage_accounting.py moves to a leaf owner;
    # every moved name keeps its parent facade re-export. Handle rows read
    # monkeypatch-addressable parent bindings through the call-time handles
    # _agent()/_usage() (delta D38, sets pinned in
    # tests/test_module_handle_extraction.py); verbatim rows moved byte-identical.
    lc2_verbatim_rows = {
        f"{parent}::{s}": f"{owner}::{s}"
        for parent, owners in _inv.lc2_verbatim_symbols_by_owner.items()
        for owner, symbols in owners.items() for s in symbols.split()
    }
    lc2_handle_rows = {
        f"{parent}::{s}": f"{owner}::{s}"
        for parent, owners in _inv.lc2_handle_symbols_by_owner.items()
        for owner, symbols in owners.items() for s in symbols.split()
    }
    implemented.update({**lc2_verbatim_rows, **lc2_handle_rows})
    existing_process_owner_rows.update({**lc2_verbatim_rows, **lc2_handle_rows})
    for _lc2_old in lc2_handle_rows:
        s3_semantic_delta_ids[_lc2_old] = "D38"
    for row in rows:
        delta = v7_evidence._migration_json(row["semantic delta"], ("id", "note"))
        upstream = v7_evidence._migration_json(row["upstream-transfer status/note"], ("status", "note"))
        assert upstream["note"]
        assert row["characterization test"] != "-"
        if row["old path/symbol"] in implemented:
            assert upstream["status"] == "pending"
            assert row["new owner/path"] == implemented[row["old path/symbol"]]
            owner_path = row["new owner/path"].split("::", 1)[0]
            if (
                row["old path/symbol"] in existing_process_owner_rows
                or row["old path/symbol"] in registry_dependency_owners | web_extractions
            ):
                assert (REPO / owner_path).is_file()
            else:
                assert owner_path in v7_evidence.APPROVED_PENDING_OWNERS
            if row["old path/symbol"] in settings_seam_rows:
                # In-place semantic changes: the old identity keeps working because it
                # is still implemented at the old path (a caller, a wrapper, or the same
                # function with a new body), not because a re-export forwards it.
                assert delta["id"] == "D03" and delta["note"]
                assert row["facade/public contract"] == "-"
                continue
            expected_delta = llm_semantic_delta_ids.get(row["old path/symbol"]) or (
                # spec 4.3.6: the settings vocabulary moved as-is, then the three no-op
                # knobs were retired from it (an approved observable delta).
                "D04"
                if row["old path/symbol"] in {
                    "ouroboros/config.py::SETTINGS_DEFAULTS",
                    "ouroboros/config.py::RETIRED_SETTING_KEYS",
                }
                # plan 1.9 batch 8: the ratchet-transition test renamed with its relaxed contract.
                else "D11"
                if row["old path/symbol"] == "tests/test_repo_health_smoke.py::test_transition_rejects_function_swap_even_at_same_cardinality"
                else "D07"
                if row["old path/symbol"] in s2_panic_delta_rows
                else "D13"
                if row["old path/symbol"] in git_ops_delta_rows
                else "D08"
                if row["old path/symbol"] in s6_delta_rows
                else "D02"
                if row["old path/symbol"] in t2b_owner_delta_rows | a21_owner_delta_rows | set(d02_plan_seam_rows) | set(followup_native_result_rows) | {
                    "ouroboros/tools/registry.py::ToolEntry",
                    "ouroboros/tools/registry.py::ToolRegistry",
                    # T1: the classification cutover is a spec 4.3.3 tool-domain delta.
                    "ouroboros/loop_tool_execution.py::_extract_result_metadata",
                    "ouroboros/loop_tool_execution.py::_is_tool_execution_failure",
                    "ouroboros/loop_tool_execution.py::_structured_tool_failure",
                    "ouroboros/_outcome_tool_errors.py::_BLOCKING_TOOL_STATUSES",
                    "ouroboros/_outcome_tool_errors.py::_POLICY_DENIAL_STATUSES",
                    "ouroboros/reflection.py::should_generate_reflection",
                    "tests/test_tool_execution_classification.py::test_shell_and_claude_failures_are_treated_as_tool_failures",
                }
                else "D18" if row["old path/symbol"] in (s3b_queue_handle_rows | s3b_pool_handle_rows)
                else "D33" if row["old path/symbol"] in lb_loop_handle_rows
                else "D35" if row["old path/symbol"] in g1_git_ops_handle_rows
                else "D36" if row["old path/symbol"] in del1_handle_rows
                else "D37" if row["old path/symbol"] in lc_review_handle_rows
                else s3_semantic_delta_ids.get(row["old path/symbol"], "none")
            )
            assert delta["id"] == expected_delta and delta["note"]
            expected_facade = (
                "-"
                if row["old path/symbol"] in (
                    set(_inv.facadeless_extraction_rows)
                    | registry_extraction_no_facade_rows
                    | lb_loop_l3_retired_rows.keys()
                )
                else row["old path/symbol"]
            )
            assert row["facade/public contract"] == expected_facade
        elif row["old path/symbol"] in retired_current:
            assert row["new owner/path"] == retired_current[row["old path/symbol"]]
            assert row["facade/public contract"] == "-"
            assert delta["id"] == retired_delta_ids.get(row["old path/symbol"], "none")
            assert delta["note"]
            assert upstream["status"] == "retired"
            assert "v7 WIP" in upstream["note"]
        else:
            assert upstream["status"] == "pending"
            expected_delta = (
                "D08"
                if row["old path/symbol"] in s6_delta_rows
                else "D02"
                if row["old path/symbol"] in {
                    "ouroboros/tools/registry.py::ToolRegistry",
                    # T1: the classification cutover is a spec 4.3.3 tool-domain delta.
                    "ouroboros/loop_tool_execution.py::_extract_result_metadata",
                    "ouroboros/loop_tool_execution.py::_is_tool_execution_failure",
                    "ouroboros/loop_tool_execution.py::_structured_tool_failure",
                    "ouroboros/_outcome_tool_errors.py::_BLOCKING_TOOL_STATUSES",
                    "ouroboros/_outcome_tool_errors.py::_POLICY_DENIAL_STATUSES",
                    "ouroboros/reflection.py::should_generate_reflection",
                    "tests/test_tool_execution_classification.py::test_shell_and_claude_failures_are_treated_as_tool_failures",
                }
                else "none"
            )
            assert delta["id"] == expected_delta and delta["note"]
            assert row["new owner/path"] in v7_evidence.APPROVED_PENDING_OWNERS
            assert row["facade/public contract"] == row["old path/symbol"]
    # Enumerated rows are pinned by MEMBERSHIP, not by a total: every name listed
    # above must still be in the ledger. A literal grand total would churn on
    # every extraction slice and says nothing about correctness — the real
    # contract is that no row escapes classification, asserted below.
    assert sum(row["old path/symbol"] in implemented for row in rows) == len(implemented)
    assert sum(row["old path/symbol"] in retired_current for row in rows) == len(retired_current)
    assert v7_migration.APPROVED_SEMANTIC_DELTAS == frozenset({"none", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D11", "D13", "D18", "D31", "D33", "D34", "D35", "D36", "D37", "D38"})
