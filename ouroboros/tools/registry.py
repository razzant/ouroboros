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
    path_text_is_inside,  # noqa: F401 — historical facade surface
    sequential_effective_cwds,  # noqa: F401 — historical facade surface
    shell_argv,  # noqa: F401 — historical facade surface
    shell_argv_with_path_tokens,  # noqa: F401 — historical facade surface
    shell_command_string,  # noqa: F401 — historical facade surface
    strip_leading_env_assignments,  # noqa: F401 — historical facade surface
    sudo_noninteractive_violation,  # noqa: F401 — historical facade surface
    unwrap_env_argv,  # noqa: F401 — historical facade surface
)
from ouroboros.tools.shell_guards import (
    PROTECTED_RUNTIME_PATHS_LOWER,  # noqa: F401 — historical facade surface
    interpreter_family,  # noqa: F401 — historical facade surface
    interpreter_inline_code,  # noqa: F401 — historical facade surface
    interpreter_write_shape,  # noqa: F401 — historical facade surface
    light_shell_repo_mutation,  # noqa: F401 — historical facade surface
    non_interpreter_write_shape,  # noqa: F401 — historical facade surface
    parse_porcelain_paths,  # noqa: F401 — historical facade surface
    process_shell_guard_args,  # noqa: F401 — historical facade surface
    runtime_data_guard_targets,  # noqa: F401 — historical facade surface
    shell_writer_targets_protected,  # noqa: F401 — historical facade surface
    workspace_executor_state_write_block,  # noqa: F401 — historical facade surface
    directory_destination_pairs,  # noqa: F401 — historical facade surface
    writer_target_rows,  # noqa: F401 — historical facade surface
    writer_target_tokens,  # noqa: F401 — historical facade surface
)
from ouroboros.tools.deliverables_shell import (
    direct_deliverable_target_block,  # noqa: F401 — historical facade surface
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
from ouroboros.tools.write_shape import _workspace_write_candidates  # noqa: F401 — historical facade surface
from ouroboros.tools.registry_guards import (  # noqa: F401 — re-exported moved surface
    _EPHEMERAL_ALLOWED_TOOLS,
    _GITHUB_TOKEN_TOOLS,
    _HEAL_MODE_ALLOWED_TOOLS,
    _WEB_TOOLS,
    _authorized_managed_update_resolver,
    _builtin_tool_availability,
    _command_mentions_protected_root,
    _disabled_tools,
    _executor_backend_candidate_allowed,
    _heal_protected_payload_sidecar,
    _light_mode_payload_mutation_allowed,
    _managed_update_code_tool_block,
    _payload_dispatch_constraint,
    _resource_allowed,
    _stray_skill_payload_failsoft,
    _task_constraint_path_allowed,
)
from ouroboros.tools.registry_guard_process import (  # noqa: F401 — re-exported moved surface
    _COMMAND_HEAD_WRAPPERS,
    _DENIED_READ_OPTIONS,
    _DETACHED_PROCESS_MARKERS,
    _NESTED_EXECUTION_MARKERS,
    _NESTED_EXECUTION_TOKENS,
    _READ_ONLY_INSPECTION_COMMANDS,
    _SEARCH_TOOL_EXEC_OPTIONS,
    _SKILL_OWNER_STATE_STEMS,
    _TRUSTED_EXECUTABLE_DIRS,
    _denied_read_option,
    _detect_context_mode_self_lowering,
    _detect_evolution_owner_control_self_change,
    _detect_mutative_toggle_self_change,
    _detect_owner_skill_attest_self_call,
    _detect_runtime_mode_elevation,
    _detect_safety_mode_self_lowering,
    _format_light_repo_write_note,
    _git_ref_snapshot,
    _is_pure_read_inspection,
    _light_repo_snapshot,
    _mentions_detached_process,
    _mentions_skill_owner_state,
    _subagent_shell_targets_secret,
    _trusted_read_head,
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
    _ephemeral_block_result,
    _executor_backend_candidate_path,
    _heal_mode_guard_result,
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


def _owner_control_mention_blocks(text_lower: str, detected: bool, writeish: bool) -> bool:
    """Shared read-carve for the owner-control mention detectors.

    The scope-floor guard adjudicated this contract at v6.80.0 (that detector
    itself was retired with its setting in the 7.0 ABI window, owner Q10=A;
    the contract it established governs the whole surviving family):
    naming an owner key/endpoint
    blocks UNLESS the whole command line is demonstrably read-only inspection —
    ``grep OUROBOROS_RUNTIME_MODE data/settings.json`` and
    ``rg /api/owner/safety-mode ouroboros/gateway`` read and do not act, and the
    product's own reuse-first duty (grep callers of ``save_settings``) depends on
    them. The other six family members stayed read-blind, blocking those exact
    inspections in every runtime mode — the same hazard class at a different
    strictness. Fail-closed like the precedent: ``writeish`` (any write shape)
    disqualifies the exemption, ``_is_pure_read_inspection`` is a HEAD allowlist
    where any interpreter, HTTP client, wrapper-with-flags, or nested execution
    is NOT an inspection, and the default ``writeish=True`` keeps a caller that
    cannot supply the fact fail-closed."""
    if not detected:
        return False
    return writeish or not _is_pure_read_inspection(text_lower)


# Commands that can only READ. This is an ALLOWLIST on purpose: an unrecognised
# command head is treated as executable access, so the enumeration fails CLOSED.
# (A denylist of "write markers" fails OPEN — every new spelling of a POST walks
# around it, which is exactly the keyword-gate antipattern BIBLE P5 forbids.)
# Wrappers that do not themselves act: the real command head follows them.
# ``git`` reads only through these subcommands.
# Allowlist MEMBERSHIP IS NOT ENOUGH: several read heads execute or write through their
# own options. Per command, because short flags are not portable — ``grep -o`` prints
# matches, ``sort -o`` writes a file. Text reaching here is lowercased, so an upper-case
# spelling (``git grep -O``, ``fd -X``) collapses onto the same entry.
# The executable itself must be a bare name or live in a system bin: ``/tmp/evil/grep``
# and ``./grep`` are shadowing, not inspection.


# Spellings that make a shell run a command NESTED inside another one. The read exemption
# fails closed on all of them: the head-allowlist can only vouch for heads it actually sees,
# and a nested command's head is not one of them ("echo" vouching for the "curl -X POST" it
# interpolates). Refusing the CONSTRUCT rather than enumerating the payloads inside it is the
# point — no list of "what a write looks like" is ever complete (BIBLE P5).
# Bare tokens the lexer emits for the same constructs (and for a plain subshell). These used to
# be STRIPPED from the token list before the head was taken, which is precisely how the nested
# command escaped validation; they are refused instead.


# verify_and_record runs the agent's declared `check` like a command, so it must clear the
# same PRE-EXECUTION shell guards (subagent-secret read, protected-artifact read, sudo,
# protected-root / workspace-state / light-mode writes) — that pre-exec filter is the
# security boundary and blocks a forbidden mutation BEFORE the handler runs, so a guarded
# check cannot mutate protected state and then leave a host-attested PASS receipt. It is
# deliberately NOT in _PROCESS_COMMAND_TOOLS: those POST-execution checks (light-repo
# diff, git-ref tripwire) run AFTER the handler has already written the
# receipt, so they would only annotate the returned text, not gate the durable receipt —
# adding them would give false assurance while the pre-exec guards already do the gating.
# Path-bearing file tools whose active_workspace/system_repo path arg is normalized
# ONCE at dispatch (execute) so the handler AND every guard (protected-path,
# protected-artifact, shrink) resolve the identical target — no desync bypass.
# apply_patch/edit_batch are absent because they carry no top-level `path` arg
# (their paths live inside the patch text / edits[] entries), so this seam has
# nothing to rewrite. They are NOT exempt from the canonicalization itself: both
# the dispatch guards below and their handlers run every payload path through
# `canonical_repo_relative_path`, the same normalization this seam applies.

# Repo-lane write tools that take a top-level `root` arg. Every gate keyed to
# "a write that lands in the repo working tree" must judge the whole set, not
# the historical write_file/edit_text pair — a new editing primitive that misses
# one of these gates is a silently weaker lane, not a new capability.


# CW3 (v6.34.0): an ephemeral decision turn DECIDES (answer / route / spawn /
# steer) — it does NOT do durable work; that is the spawned task's job.
# Enforced as a DEFAULT-DENY ALLOWLIST, not a denylist (a denylist is
# whack-a-mole: it kept missing review/skill/publish/control mutators —
# advisory_review, skill_review, submit_skill_to_hub, skill_exec,
# toggle_skill, cancel_task, task_acceptance_review, ...). The turn may call
# only the read-only INSPECTION tools plus the route/spawn/steer/reply tools
# below; everything else — repo/git/cognitive/control/review/skill/publish
# mutators, run_command (shell is durable-capable), extension/MCP tools —
# is hidden from schemas() and fails closed in execute(). The allowlist is
# EXPLICITLY curated, not derived (deriving from
# LOCAL_READONLY_SUBAGENT_TOOL_NAMES leaked subagent-only tools:
# schedule_subagent spawns durable children, wait_task/wait_tasks BLOCK a
# short turn, browser_action INTERACTS with pages).
