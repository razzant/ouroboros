"""Compatibility facade for the tool registry and established registry imports."""

from __future__ import annotations

from ouroboros.tools.tool_catalog import ToolEntry  # noqa: F401
from ouroboros.tools.tool_context import BrowserState, ToolContext  # noqa: F401
from ouroboros.tools.tool_resolution import (  # noqa: F401
    _GENERIC_VCS_TARGET_TOOLS,
    _PATH_NORMALIZED_TOOLS,
    _PROCESS_TARGET_TOOLS,
    _SKILL_LIFECYCLE_TARGET_TOOLS,
    _TARGET_BINDING_OPERATIONS,
    _VERIFY_RUN_KINDS,
    _binding_items,
    _binding_set_is_light_restricted,
    _binding_set_targets_system_repo,
    _binding_state_drive_root,
    _build_builtin_target_binding,
    _coerce_real_path,
    _normalize_dispatch_path_args,
    _target_binding_operation,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.tools.tool_result import _compose_execute_result  # noqa: F401
from ouroboros.tools.registry_guards import (  # noqa: F401
    _EPHEMERAL_ALLOWED_TOOLS,
    _GITHUB_TOKEN_TOOLS,
    _HEAL_MODE_ALLOWED_TOOLS,
    _WEB_TOOLS,
    _authorized_managed_update_resolver,
    _builtin_tool_availability,
    _disabled_tools,
    _heal_protected_payload_sidecar,
    _managed_update_code_tool_block,
    _resource_allowed,
    _task_constraint_path_allowed,
)
from ouroboros.tools.registry_core import ToolRegistry  # noqa: F401
