"""Tool registry SSOT: load tool modules, expose schemas, execute safely."""

from __future__ import annotations

import copy  # noqa: F401 — historical facade surface
import hashlib  # noqa: F401 — historical facade surface
import inspect  # noqa: F401 — historical facade surface
import logging
import os  # noqa: F401 — historical facade surface
import pathlib  # noqa: F401 — historical facade surface
import re  # noqa: F401 — historical facade surface
import subprocess  # noqa: F401 — historical facade surface
from dataclasses import dataclass, field  # noqa: F401 — historical facade surface
from typing import Any, Callable, Dict, List, Optional  # noqa: F401 — historical facade surface

from ouroboros.runtime_mode_policy import (
    PROTECTED_RUNTIME_PATHS,  # noqa: F401 — historical facade surface
    mode_allows_protected_write,  # noqa: F401 — historical facade surface
    protected_paths_in,  # noqa: F401 — historical facade surface
    protected_write_block_message,  # noqa: F401 — historical facade surface
)
from ouroboros.tool_capabilities import (
    ACTING_SUBAGENT_MODE,  # noqa: F401 — historical facade surface
    ACTING_SUBAGENT_TOOL_NAMES,  # noqa: F401 — historical facade surface
    CORE_TOOL_NAMES,  # noqa: F401 — historical facade surface
    LOCAL_READONLY_SUBAGENT_MODE,  # noqa: F401 — historical facade surface
    LOCAL_READONLY_SUBAGENT_TOOL_NAMES,  # noqa: F401 — historical facade surface
    META_TOOL_NAMES,  # noqa: F401 — historical facade surface
)
from ouroboros.shell_parse import (
    directory_destination_child_name,  # noqa: F401 — historical facade surface
    is_absolute_path_text,  # noqa: F401 — historical facade surface
    sequential_effective_cwds,  # noqa: F401 — historical facade surface
    shell_argv,  # noqa: F401 — historical facade surface
    shell_command_string,  # noqa: F401 — historical facade surface
    strip_leading_env_assignments,  # noqa: F401 — historical facade surface
    sudo_noninteractive_violation,  # noqa: F401 — historical facade surface
    unwrap_env_argv,  # noqa: F401 — historical facade surface
)
from ouroboros.tools.shell_guards import (
    interpreter_family,  # noqa: F401 — historical facade surface
    interpreter_inline_code,  # noqa: F401 — historical facade surface
    parse_porcelain_paths,  # noqa: F401 — historical facade surface
    process_shell_guard_args,  # noqa: F401 — historical facade surface
    directory_destination_pairs,  # noqa: F401 — historical facade surface
    writer_target_tokens,  # noqa: F401 — historical facade surface
)
from ouroboros.tools.deliverables_shell import (
    lexical_user_files_block_reason,  # noqa: F401 — historical facade surface
)
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts  # noqa: F401 — historical facade surface
from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason  # noqa: F401 — historical facade surface
from ouroboros.git_shell_policy import run_shell_git_block_reason, workspace_git_safety_violation  # noqa: F401 — historical facade surface
from ouroboros.tool_access import (
    active_tool_profile,  # noqa: F401 — historical facade surface
    binding_targets_system_repo,  # noqa: F401 — historical facade surface
    build_resolved_resource_binding,  # noqa: F401 — historical facade surface
    canonical_repo_relative_path,  # noqa: F401 — historical facade surface
    decide_tool_access,  # noqa: F401 — historical facade surface
    _deliverables_root_lexical,  # noqa: F401 — historical facade surface
    _deliverables_root_lexical_alias,  # noqa: F401 — historical facade surface
    _lexical_path_is_relative_to_casefold,  # noqa: F401 — historical facade surface
    is_external_workspace,  # noqa: F401 — historical facade surface
    light_cognitive_or_root_redirect,  # noqa: F401 — historical facade surface
    normalize_root,  # noqa: F401 — historical facade surface
    normalize_root_relative,  # noqa: F401 — historical facade surface
    _path_is_relative_to_casefold,  # noqa: F401 — historical facade surface
    resource_root_path,  # noqa: F401 — historical facade surface
    resolve_shell_cwd,  # noqa: F401 — historical facade surface
    shell_cwd_block_message,  # noqa: F401 — historical facade surface
    UserFilesPathBlockedError,  # noqa: F401 — historical facade surface
    user_files_path_block_reason,  # noqa: F401 — historical facade surface
    workspace_mode_block_reason,  # noqa: F401 — historical facade surface
)
from ouroboros.process_interpreters import (  # noqa: F401 — historical facade surface
    interpreter_attestation,
    record_interpreter_resolution,
    resolve_node_postgates,
    resolve_process_python,
)
from ouroboros.utils import safe_relpath  # noqa: F401 — historical facade surface
from ouroboros.contracts.task_constraint import TaskConstraint, VALID_WRITE_SURFACES, normalize_task_constraint  # noqa: F401 — historical facade surface
from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,  # noqa: F401 — historical facade surface
    SKILL_OWNER_STATE_STEMS,  # noqa: F401 — historical facade surface
    SKILL_PAYLOAD_CONTROL_DIRNAMES,  # noqa: F401 — historical facade surface
    SKILL_PAYLOAD_CONTROL_FILENAMES,  # noqa: F401 — historical facade surface
    constraint_bucket_skill,  # noqa: F401 — historical facade surface
    cross_skill_redirect_error,  # noqa: F401 — historical facade surface
    decide_payload_short_form,  # noqa: F401 — historical facade surface
    is_skill_payload_control_filename,  # noqa: F401 — historical facade surface
    is_skill_payload_path,  # noqa: F401 — historical facade surface
    resolve_skill_payload_target,  # noqa: F401 — historical facade surface
    synthesize_payload_constraint,  # noqa: F401 — historical facade surface
)

# v7 D04 split: the owners below were extracted VERBATIM from this module
# (see each leaf's header); re-exported here so the 80+ historical importers
# and monkeypatch targets keep working unchanged.
from ouroboros.tools.tool_catalog import ToolEntry  # noqa: F401 — re-exported moved surface
from ouroboros.tools.tool_context import BrowserState, ToolContext  # noqa: F401 — re-exported moved surface
from ouroboros.tools.tool_resolution import (  # noqa: F401 — re-exported moved surface
    _GENERIC_VCS_TARGET_TOOLS,
    _IGNORE_ROOT_ARG_TOOLS,
    _PATH_NORMALIZED_TOOLS,
    _PROCESS_TARGET_TOOLS,
    _ROOT_ARG_REPO_WRITE_TOOLS,
    _SKILL_LIFECYCLE_TARGET_TOOLS,
    _TARGET_BINDING_OPERATIONS,
    _TOOL_ARG_ALIASES,
    _VERIFY_RUN_KINDS,
    _binding_error_text,
    _binding_items,
    _binding_set_is_light_restricted,
    _binding_set_targets_system_repo,
    _build_builtin_target_binding,
    _coerce_real_path,
    _entry_has_public_param_schema,
    _entry_public_params,
    _format_tool_arg_error,
    _handler_public_params,
    _light_binding_failure_redirect,
    _normalize_dispatch_path_args,
    _normalize_tool_call_args,
    _payload_write_paths,
    _prepare_public_builtin_args,
    _target_binding_operation,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.tools.registry_guards import (  # noqa: F401 — re-exported moved surface
    _GITHUB_TOKEN_TOOLS,
    _WEB_TOOLS,
    _authorized_managed_update_resolver,
    _builtin_tool_availability,
    _disabled_tools,
    _executor_backend_candidate_allowed,
    _light_mode_payload_mutation_allowed,
    _managed_update_code_tool_block,
    _payload_dispatch_constraint,
    _resource_allowed,
    _stray_skill_payload_failsoft,
)
from ouroboros.tools.registry_guard_process import (  # noqa: F401 — re-exported moved surface
    _format_light_repo_write_note,
    _git_ref_snapshot,
    _light_repo_snapshot,
)

log = logging.getLogger(__name__)

# v7 F3.1 typed-organ re-homing: the surfaces below moved out of this module
# (ToolRegistry and its module constants into registry_core, the composition
# helper into tool_result, the write-block message helpers and the executor
# backend path mapper into registry_guards); re-exported here so the historical
# importers and monkeypatch targets keep working unchanged.
from ouroboros.tools.tool_result import (  # noqa: F401 — re-exported moved surface
    LegacyTextResultAdapter,
    ToolResult,
    _compose_execute_result,
)
from ouroboros.tools.registry_guards import (  # noqa: F401 — re-exported moved surface
    _capability_resource_guard_result,
    _executor_backend_candidate_path,
    _managed_update_code_tool_block_result,
    _subagent_and_update_guard_result,
    _workspace_write_block_outside_root_message,
    _workspace_write_block_runtime_message,
)
from ouroboros.tools.registry_core import (  # noqa: F401 — re-exported moved surface
    ToolRegistry,
    _ACTING_NO_WORKSPACE_PROCESS_RESULT,
    _ACTING_NO_WORKSPACE_REPO_RESULT,
    _LIGHT_START_SERVICE_RESULT,
    _PROCESS_COMMAND_TOOLS,
    _REPO_MUTATION_TOOLS,
    _SHELL_GUARDED_TOOLS,
    _SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS,
    _configured_delegate_selector,
    _presence_binding_allowed,
    _presence_bound_args,
    _presence_tool_allowed,
    _protected_write_block_result,
    _shell_guard_required,
)




# verify_and_record runs the agent's declared `check` like a command, so it must clear the
# same PRE-EXECUTION shell guards (subagent-secret read, protected-artifact read, sudo,
# protected-root / workspace-state / light-mode writes) — that pre-exec filter is the
# security boundary and blocks a forbidden mutation BEFORE the handler runs, so a guarded
# check cannot mutate protected state and then leave a host-attested PASS receipt. It is
# deliberately NOT in _PROCESS_COMMAND_TOOLS: those POST-execution checks (light-repo
# diff, git-ref tripwire) run AFTER the handler has already written the
# receipt, so they would only annotate the returned text, not gate the durable receipt —
# adding them would give false assurance while the pre-exec guards already do the gating.
# Dispatch path normalization for path-bearing file tools, apply_patch patch text
# and edit_batch edits[] paths included, is owned by
# `tool_resolution._PATH_NORMALIZED_TOOLS` / `_normalize_dispatch_path_args_result`.

# Repo-lane write tools that take a top-level `root` arg. Every gate keyed to
# "a write that lands in the repo working tree" must judge the whole set, not
# the historical write_file/edit_text pair — a new editing primitive that misses
# one of these gates is a silently weaker lane, not a new capability.
