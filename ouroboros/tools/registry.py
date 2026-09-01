"""Tool registry SSOT: load tool modules, expose schemas, execute safely."""

from __future__ import annotations

import copy
import hashlib
import inspect
import logging
import os
import pathlib
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.runtime_mode_policy import (
    PROTECTED_RUNTIME_PATHS,
    mode_allows_protected_write,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tool_capabilities import (
    ACTING_SUBAGENT_MODE,
    ACTING_SUBAGENT_TOOL_NAMES,
    CORE_TOOL_NAMES,
    LOCAL_READONLY_SUBAGENT_MODE,
    LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    META_TOOL_NAMES,
)
from ouroboros.shell_parse import (
    is_absolute_path_text,
    path_text_is_inside,
    shell_argv,
    shell_argv_with_path_tokens,
    shell_command_string,
    strip_leading_env_assignments,
    sudo_noninteractive_violation,
    unwrap_env_argv,
)
from ouroboros.tools.read_inspection import _is_pure_read_inspection
from ouroboros.tools.shell_guards import (
    PROTECTED_RUNTIME_PATHS_LOWER,
    interpreter_family,
    interpreter_write_shape,
    light_shell_repo_mutation,
    non_interpreter_write_shape,
    parse_porcelain_paths,
    process_shell_guard_args,
    runtime_data_guard_targets,
    shell_writer_targets_protected,
    workspace_executor_state_write_block,
    directory_destination_child_name,
    directory_destination_pairs,
    writer_target_tokens,
)
from ouroboros.tools.deliverables_shell import (
    direct_deliverable_target_block,
    lexical_user_files_block_reason,
)
from ouroboros.tools.result_envelope import annotate as _annotate_result, append_note as _append_result_note
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason
from ouroboros.git_shell_policy import run_shell_git_block_reason, workspace_git_safety_violation
from ouroboros.tool_access import (
    active_tool_profile,
    binding_targets_system_repo,
    build_resolved_resource_binding,
    canonical_repo_relative_path,
    decide_tool_access,
    _deliverables_root_lexical,
    _deliverables_root_lexical_alias,
    _lexical_path_is_relative_to_casefold,
    is_external_workspace,
    light_cognitive_or_root_redirect,
    normalize_root,
    normalize_root_relative,
    _path_is_relative_to_casefold,
    resource_root_path,
    resolve_shell_cwd,
    shell_cwd_block_message,
    UserFilesPathBlockedError,
    user_files_path_block_reason,
    workspace_mode_block_reason,
)
from ouroboros.process_interpreters import (
    interpreter_attestation,
    record_interpreter_resolution,
    resolve_node_postgates,
    resolve_process_python,
)
from ouroboros.utils import safe_relpath
from ouroboros.contracts.task_constraint import TaskConstraint, VALID_WRITE_SURFACES, normalize_task_constraint
from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_STEMS,
    SKILL_PAYLOAD_CONTROL_DIRNAMES,
    SKILL_PAYLOAD_CONTROL_FILENAMES,
    constraint_bucket_skill,
    cross_skill_redirect_error,
    decide_payload_short_form,
    is_skill_payload_control_filename,
    is_skill_payload_path,
    resolve_skill_payload_target,
    synthesize_payload_constraint,
)

log = logging.getLogger(__name__)
def _coerce_real_path(value: Any) -> pathlib.Path | None:
    if value is None or value.__class__.__module__.startswith("unittest.mock"):
        return None
    try:
        return pathlib.Path(os.fspath(value))
    except TypeError:
        return None
def active_repo_dir_for(ctx: Any) -> pathlib.Path:
    """Return the active repo/workspace root for real and lightweight test contexts."""
    active = getattr(ctx, "active_repo_dir", None)
    if callable(active):
        try:
            candidate = active()
        except Exception:
            candidate = None
        path = _coerce_real_path(candidate)
        if path is not None:
            return path

    workspace_root = getattr(ctx, "workspace_root", None)
    workspace_path = _coerce_real_path(workspace_root)
    if workspace_path is not None:
        workspace_mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
        if workspace_mode:
            return workspace_path

    return pathlib.Path(getattr(ctx, "repo_dir"))


def system_repo_dir_for(ctx: Any) -> pathlib.Path:
    """Return the Ouroboros system repo root, not an external active workspace."""

    return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir"))


def _executor_backend_candidate_allowed(ctx: Any, candidate: str, allowed_roots: List[pathlib.Path]) -> bool:
    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx as _executor_ref_from_ctx
        from ouroboros.workspace_executor import map_backend_path as _executor_map_backend_path

        executor_ref = _executor_ref_from_ctx(ctx)
        if executor_ref is None:
            return False
        resolved = _executor_map_backend_path(executor_ref, candidate)
        return any(
            resolved.is_relative_to(root) or _path_is_relative_to_casefold(resolved, root)
            for root in allowed_roots
        )
    except Exception:
        return False


def _executor_backend_candidate_path(ctx: Any, candidate: str) -> pathlib.Path | None:
    """Map one backend spelling lexically, preserving descendant symlinks."""
    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx as _executor_ref_from_ctx
        from ouroboros.workspace_executor import map_backend_path_lexical as _executor_map_backend_path_lexical

        executor_ref = _executor_ref_from_ctx(ctx)
        if executor_ref is None:
            return None
        return _executor_map_backend_path_lexical(executor_ref, candidate)
    except Exception:
        return None


def _owner_control_mention_blocks(text_lower: str, detected: bool, writeish: bool) -> bool:
    """Shared read-carve for the owner-control mention detectors.

    The scope-floor guard adjudicated this contract at v6.80.0
    (``_detect_scope_review_floor_self_lowering``): naming an owner key/endpoint
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


def _detect_runtime_mode_elevation(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script attempts to change ``OUROBOROS_RUNTIME_MODE``."""
    has_save = "save_settings" in text_lower
    has_mode_key = "ouroboros_runtime_mode" in text_lower
    has_dotted_path = "ouroboros.config.save_settings" in text_lower
    detected = (has_save and has_mode_key) or has_dotted_path
    return _owner_control_mention_blocks(text_lower, detected, writeish)


_SUBAGENT_SHELL_SECRET_MARKERS = (
    # Ouroboros owner secrets/control state. The relative form (no leading slash)
    # closes the interpreter-string bypass (CW4, v6.34.0): the whole-command
    # substring scan already catches "/data/settings.json" and "../../data/..",
    # but a bare "data/settings.json" (e.g. python -c "open('data/settings.json')"
    # from a workspace cwd) needs the slash-less marker too.
    "/data/settings.json", "data/settings.json", "ouroboros/data/settings", "file1.txt",
    # Universal credential/secret/control files (relative or absolute).
    # ouroboros-update-tx.json is the managed-update tx marker (.git/…): owner
    # control state, mirrored on .git/config. Subagent shell only — the
    # authorized resolver is the MAIN agent and the supervisor/host writers go
    # through supervisor.update_merge, so neither is affected (synthesis F3).
    ".env", ".git/config", ".git/credentials", "ouroboros-update-tx.json",
    "credentials.json", "tokens.json",
    "/.ssh/", ".ssh/", "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pgpass", ".aws/",
)


def _subagent_shell_targets_secret(cmd_path_lower: str) -> bool:
    """Deterministic guard: a shell command referencing Ouroboros secrets/credentials
    or owner-control state (settings.json, ssh keys, token/credential files)."""
    return any(marker in cmd_path_lower for marker in _SUBAGENT_SHELL_SECRET_MARKERS)


def _command_mentions_protected_root(cmd_path_lower: str, root_text: str) -> bool:
    """Boundary-aware path containment for the workspace shell guard.

    True only when ``root_text`` (a normalised, lower-cased protected root path)
    appears in the command as a whole path or a parent prefix at a real path
    boundary — NOT as an incidental substring of an unrelated path that merely
    shares the prefix (e.g. protected ``/x/data`` must not match ``/x/database``).
    Used as a coarse catch-all for runtime paths embedded in non-tokenised text
    (e.g. inside a ``python -c`` string); the precise per-token containment loop
    still does the authoritative active/protected classification.
    """
    if not root_text:
        return False
    norm = root_text.rstrip("/")
    if not norm:
        return False
    span = len(norm)
    limit = len(cmd_path_lower)
    start = 0
    while True:
        idx = cmd_path_lower.find(norm, start)
        if idx < 0:
            return False
        end = idx + span
        nxt = cmd_path_lower[end] if end < limit else ""
        # Boundary = end-of-string, a path separator (child path), or a shell
        # token delimiter (the exact path). A trailing path char (letter/digit/
        # ``.``/``-``/``_``) means a DIFFERENT sibling path → keep scanning.
        if nxt == "" or nxt == "/" or nxt in " \t\"')(;:,&|<>":
            return True
        start = end


def _workspace_write_block_runtime_message(path_text: Any = "") -> str:
    """Guard-B block for a write-shaped command reaching a protected runtime root.

    Names the resolved offending path and the sanctioned route (the light-lane
    message at the runtime_data guard is the in-repo exemplar): five return sites
    used to emit one byte-identical reasonless string, so the log could not even
    say WHICH path fired, and the agent had no route to self-correct.
    """
    path_note = f" Blocked path: {path_text}." if str(path_text or "").strip() else ""
    return (
        "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
        + path_note
        + " Use the gated read_file/write_file tools for runtime data (installed skill"
        " payloads: root=skill_payload with bucket/skill_name, or run the command with"
        " cwd=skill_payload), and keep shell writes inside the selected process root."
    )


def _workspace_write_block_outside_root_message(path_text: Any = "", work_dir: Any = "") -> str:
    """Guard-B block for a write-shaped command targeting outside the process root."""
    path_note = f" Blocked path: {path_text}." if str(path_text or "").strip() else ""
    root_note = f" Selected process root: {work_dir}." if str(work_dir or "").strip() else ""
    return (
        "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths"
        " outside the selected process root." + path_note + root_note
        + " Write inside the process root, or use the file tools with an explicit root"
        " (user_files / task_drive / artifact_store)."
    )


def _stray_skill_payload_failsoft(root_arg: str, workspace_mode: bool, task_constraint: Any) -> bool:
    """Whether stray bucket/skill_name on a write tool should be DROPPED rather than
    surfaced as SKILL_PAYLOAD_ARG_ERROR. Fail-soft ONLY for a WORKSPACE edit that is
    NOT skill-authoring: there bucket/skill_name are model noise (the B2 footgun —
    reflexive bucket="external" on an /app edit). In light/advanced non-workspace
    skill-authoring (or an explicit root=skill_payload / skill_repair) the specific
    error is the intended helpful signal."""
    skill_payload_intent = root_arg == "skill_payload" or bool(
        task_constraint and getattr(task_constraint, "mode", "") == "skill_repair"
    )
    return bool(workspace_mode and not skill_payload_intent)


def _detect_mutative_toggle_self_change(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script/CLI attempts to change the owner-only mutative-subagents toggle."""
    has_key = "ouroboros_allow_mutative_subagents" in text_lower
    has_write = (
        "save_settings" in text_lower
        or "settings.json" in text_lower
        or "/api/settings" in text_lower
        or "settings set" in text_lower  # `ouroboros settings set <key> <value>` CLI path
        or "ouroboros.cli" in text_lower
    )
    return _owner_control_mention_blocks(text_lower, has_key and has_write, writeish)


def _managed_update_code_tool_block(ctx: Any, name: str) -> str:
    """Block a repo-mutating code tool while a managed-update assisted merge is staged for
    ANOTHER task (P2/SC2). Returns a block message, or "" when allowed (this is the authorized
    resolution task, or no managed tx is active). A corrupt tx marker fails closed."""
    try:
        from supervisor.update_merge import managed_assisted_tx_for

        if managed_assisted_tx_for(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        )[1]:
            return (
                f"⚠️ MANAGED_UPDATE_IN_PROGRESS: {name!r} is blocked while a managed update merge "
                "is being resolved (only its authorized resolution task may write the repo). "
                "Retry after the update lands or is rolled back."
            )
    except Exception:
        return (
            f"⚠️ MANAGED_UPDATE_STATE_UNAVAILABLE: {name!r} is blocked because the managed "
            "update transaction state could not be verified. Retry after the update state is "
            "available or repaired."
        )
    return ""


def _authorized_managed_update_resolver(ctx: Any) -> bool:
    """Whether this task is the durable tx-authorized assisted resolver.

    Fail-closed bool for every authority consumer (False = no extra powers).
    The AUTHORITY-READ failure is additionally distinguished from an honest
    "not the resolver" via a typed ctx marker (``_managed_authority_read_error``:
    set on an unreadable read AND on a corrupt tx marker, cleared on every
    healthy evaluation), so the review-subject builder can fail LOUDLY instead
    of silently reviewing a possibly-managed candidate as an ordinary full
    staged capture."""
    try:
        from supervisor.update_merge import authorized_assisted_task_strict

        marker_status, tx = authorized_assisted_task_strict(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        )
        try:
            if marker_status == "corrupt":
                # A tx marker EXISTS but cannot be parsed: authority stays
                # False (fail-closed) for every bool consumer, but the loud
                # A4 channel must fire — clearing the marker here would let
                # the review subject silently treat a possibly-managed
                # candidate as an ordinary full staged diff.
                setattr(
                    ctx, "_managed_authority_read_error",
                    "update_tx_corrupt: the managed update transaction marker "
                    "exists but could not be parsed",
                )
            else:
                setattr(ctx, "_managed_authority_read_error", "")
        except Exception:
            pass
        return bool(tx)
    except Exception as exc:
        try:
            setattr(ctx, "_managed_authority_read_error", repr(exc))
        except Exception:
            pass
        return False


def _detect_evolution_owner_control_self_change(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script/CLI attempts to set the owner-only self-evolution controls:
    the post-task evolution toggle OR the persistent evolution-objective steer (which
    biases every evolution campaign, so it is owner-only like the toggle)."""
    has_key = (
        "ouroboros_post_task_evolution" in text_lower
        or "ouroboros_evolution_persistent_objective" in text_lower
    )
    has_write = (
        "save_settings" in text_lower
        or "settings.json" in text_lower
        or "/api/settings" in text_lower
        or "settings set" in text_lower
        or "ouroboros.cli" in text_lower
    )
    return _owner_control_mention_blocks(text_lower, has_key and has_write, writeish)


def _detect_context_mode_self_lowering(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script attempts to lower the owner-controlled context mode."""
    mentions_context_key = "ouroboros_context_mode" in text_lower
    mentions_owner_endpoint = "/api/owner/context-mode" in text_lower
    mentions_context_endpoint = "context-mode" in text_lower and "/api/owner" in text_lower
    mentions_context_cli = "context-mode" in text_lower and (
        "ouroboros settings" in text_lower
        or "ouroboros.cli" in text_lower
    )
    mentions_save = "save_settings" in text_lower or "settings.json" in text_lower
    mentions_owner_lowering_flag = "allow_context_lowering" in text_lower
    detected = (
        mentions_owner_endpoint
        or mentions_context_endpoint
        or mentions_context_cli
        or mentions_owner_lowering_flag
        or (mentions_context_key and mentions_save)
    )
    return _owner_control_mention_blocks(text_lower, detected, writeish)



def _detect_scope_review_floor_self_lowering(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script attempts to REACH the owner-controlled scope-review floor
    (CW1, v6.34.0). ``OUROBOROS_SCOPE_REVIEW_FLOOR`` is deprecated and enforcement-inert
    since v6.80.0 (scope-review applicability follows the owner context mode), but it is
    still an owner-only stored setting behind its dedicated audited endpoint, so the agent
    must not write it through any channel. Mirrors the context-mode guard.

    POLARITY (v6.80.0): naming the owner endpoint or the floor key in a settings context
    is blocked UNLESS the whole command line is demonstrably read-only inspection
    (``_is_pure_read_inspection``). The earlier shape — block only on a listed HTTP write
    marker — failed OPEN: ``python -c "httpx.request('POST', '.../api/owner/
    scope-review-floor', ...)"`` names the endpoint, matches no marker, and mutated the
    setting. No substring enumeration of "what a write looks like" is ever complete
    (BIBLE P5), so the enumeration was inverted to "what a read looks like", where an
    unrecognised entry is refused rather than admitted.

    Pure source inspection stays allowed: ``grep OUROBOROS_SCOPE_REVIEW_FLOOR
    data/settings.json`` and ``rg '/api/owner/scope-review-floor' ouroboros/gateway``
    read and do not act. ``writeish`` is the shell guard's own already-computed
    write-shape fact (redirects, writer commands, write-mode ``open()``); it disqualifies
    the read exemption, and its default ``True`` keeps a caller that cannot supply the
    fact fail-closed."""
    import urllib.parse

    decoded = urllib.parse.unquote(urllib.parse.unquote(text_lower)).lower()
    text = f"{text_lower} {decoded}"
    mentions_floor_key = "ouroboros_scope_review_floor" in text
    mentions_owner_endpoint = "/api/owner/scope-review-floor" in text
    mentions_floor_endpoint = "scope-review-floor" in text and "/api/owner" in text
    mentions_floor_cli = "scope-review-floor" in text and (
        "ouroboros settings" in text
        or "ouroboros.cli" in text
    )
    mentions_save = "save_settings" in text or "settings.json" in text or "/api/settings" in text
    reaches_floor = (
        mentions_owner_endpoint
        or mentions_floor_endpoint
        or mentions_floor_cli
        or (mentions_floor_key and mentions_save)
    )
    if not reaches_floor:
        return False
    return writeish or not _is_pure_read_inspection(text_lower)


def _compose_execute_result(result: str, route_note: str, safety_msg: str) -> str:
    """Assemble the final tool result.

    ALL host notes TRAIL the result (#447 В12/H1): failure classification
    (loop_tool_execution) reads the producer's typed result_meta first and the
    payload's FIRST line as fallback, so neither the auto-route note nor the
    safety warning may own line 1 — a leading SAFETY_WARNING used to reclassify
    a failed command (or an extension's ``{"ok": false}`` answer) as ok."""
    if route_note:
        result = _append_result_note(result, route_note)
    if safety_msg:
        result = _append_result_note(result, safety_msg)
    return result


def _detect_safety_mode_self_lowering(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script attempts to lower the owner-controlled LLM-safety coverage
    (v6.54.3). OUROBOROS_SAFETY_MODE gates the LLM safety supervisor layer; the agent
    must not switch it to ``light``/``off`` to remove its own supervision friction
    (BIBLE P3). Mirrors the context-mode/scope-floor self-lowering guards."""
    import urllib.parse

    # URL-decode so a percent-encoded endpoint (curl .../safety%2Dmode) cannot
    # slip the literal scan (review round 6; mirrors the attest-endpoint guard).
    decoded = urllib.parse.unquote(urllib.parse.unquote(text_lower)).lower()
    text = f"{text_lower} {decoded}"
    mentions_mode_key = "ouroboros_safety_mode" in text
    mentions_owner_endpoint = "/api/owner/safety-mode" in text
    mentions_mode_endpoint = "safety-mode" in text and "/api/owner" in text
    mentions_mode_cli = "safety-mode" in text and (
        "ouroboros settings" in text
        or "ouroboros.cli" in text
    )
    mentions_save = "save_settings" in text or "settings.json" in text or "/api/settings" in text
    detected = (
        mentions_owner_endpoint
        or mentions_mode_endpoint
        or mentions_mode_cli
        or (mentions_mode_key and mentions_save)
    )
    return _owner_control_mention_blocks(text_lower, detected, writeish)


def _detect_owner_skill_attest_self_call(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect agent attempts to loopback-call the OWNER-ONLY skill owner-attestation endpoint
    (C1, v6.39). Owner-attestation skips the expensive LLM skill review; it MUST be
    owner-issued, never agent self-callable — otherwise the agent could self-bypass the
    immune system's review for its own skill. Mirrors the context-mode/scope-floor guards.

    URL-DECODE first so a percent-encoded path (e.g. ``%61ttest-review`` / ``attest%2Dreview``)
    — which Starlette decodes back to ``attest-review`` before routing — cannot slip past the
    literal match (decode twice to catch double-encoding)."""
    import urllib.parse
    decoded = urllib.parse.unquote(urllib.parse.unquote(text_lower)).lower()
    text = f"{text_lower} {decoded}"
    detected = "/api/owner/skills/" in text and "attest-review" in text
    return _owner_control_mention_blocks(text_lower, detected, writeish)


def _task_constraint_path_allowed(path_text: str, constraint: Optional[TaskConstraint], drive_root: pathlib.Path) -> bool:
    return is_skill_payload_path(
        drive_root,
        path_text or "",
        constraint=constraint,
        allow_short_relative=True,
        allow_control_plane=True,
    )

def _light_mode_payload_mutation_allowed(
    *,
    ctx: Any,
    tool_name: str,
    args: Dict[str, Any],
    runtime_mode: str,
    effective_constraint: Optional[TaskConstraint],
    implicit_skill_cwd_allowed: bool,
    allow_short_relative: bool,
) -> bool:
    """Return True for light-mode data skill payload edits that do not touch repo files."""

    # apply_patch/edit_batch are DELIBERATELY absent: they refuse data-plane roots
    # entirely (repo lanes only), so they can never be a payload edit — in light
    # mode they stay under the generic repo-mutation block like any repo write.
    if runtime_mode != "light" or tool_name not in {"edit_text", "write_file"}:
        return False
    requested_root = str(args.get("root", "") or "active_workspace")
    try:
        requested_root = normalize_root(requested_root)
    except Exception:
        requested_root = str(args.get("root", "") or "active_workspace")
    if requested_root in {"task_drive", "artifact_store", "user_files"}:
        return True
    legacy_data_skill_edit = False
    if tool_name == "edit_text" and requested_root == "active_workspace":
        try:
            legacy_target = resolve_skill_payload_target(
                pathlib.Path(ctx.drive_root),
                str(args.get("path", "") or ""),
            )
            legacy_data_skill_edit = legacy_target.target_path.exists() and not legacy_target.control_plane
        except Exception:
            legacy_data_skill_edit = False
    if requested_root not in {"runtime_data", "skill_payload"} and not legacy_data_skill_edit:
        return False
    return is_skill_payload_path(
        pathlib.Path(ctx.drive_root),
        str(args.get("path", "") or ""),
        constraint=effective_constraint,
        allow_short_relative=allow_short_relative,
        allow_control_plane=False,
    )


_HEAL_MODE_ALLOWED_TOOLS = frozenset({
    "read_file",
    "list_files",
    "write_file",
    "edit_text",
    "list_skills",
    "skill_review", "skill_preflight",
})

_HEAL_PROTECTED_PAYLOAD_FILENAMES = SKILL_PAYLOAD_CONTROL_FILENAMES


_SKILL_OWNER_STATE_STEMS = SKILL_OWNER_STATE_STEMS
_DETACHED_PROCESS_MARKERS = ("start_new_session", "new_session", "setsid", "preexec_fn", "nohup")


def _mentions_skill_owner_state(text_lower: str, *, writeish: bool = True) -> bool:
    """Skill owner-state mention, with the family read-carve (#447 A2): the file
    plane explicitly allows reading review.json (core.py review-carve), so a pure
    read inspection naming it must not be refused here with a WRITE-named marker.
    Any write shape or non-inspection head still blocks, fail-closed."""
    if "state" not in text_lower or "skills" not in text_lower:
        return False
    detected = False
    for stem in _SKILL_OWNER_STATE_STEMS:
        if f"{stem}.json" in text_lower or (stem in text_lower and ".json" in text_lower):
            detected = True
            break
    return _owner_control_mention_blocks(text_lower, detected, writeish)


def _mentions_detached_process(text_lower: str) -> bool:
    return any(marker in text_lower for marker in _DETACHED_PROCESS_MARKERS)


def _heal_protected_payload_sidecar(path_text: str) -> bool:
    return is_skill_payload_control_filename(path_text)


_PROCESS_COMMAND_TOOLS = frozenset({"run_command", "run_script", "start_service"})
# verify_and_record runs the agent's declared `check` like a command, so it must clear the
# same PRE-EXECUTION shell guards (subagent-secret read, protected-artifact read, sudo,
# protected-root / workspace-state / light-mode writes) — that pre-exec filter is the
# security boundary and blocks a forbidden mutation BEFORE the handler runs, so a guarded
# check cannot mutate protected state and then leave a host-attested PASS receipt. It is
# deliberately NOT in _PROCESS_COMMAND_TOOLS: those POST-execution checks (light-repo
# diff, git-ref tripwire) run AFTER the handler has already written the
# receipt, so they would only annotate the returned text, not gate the durable receipt —
# adding them would give false assurance while the pre-exec guards already do the gating.
_SHELL_GUARDED_TOOLS = _PROCESS_COMMAND_TOOLS | {"verify_and_record"}


def _shell_guard_required(name: str, args: Dict[str, Any]) -> bool:
    """Keep the command-less actor zero-run receipt outside shell-CWD guards."""

    return name in _SHELL_GUARDED_TOOLS and not (
        name == "verify_and_record"
        and str(args.get("contract_kind") or "").strip() == "delegation_zero_run"
    )
# Path-bearing file tools whose active_workspace/system_repo path arg is normalized
# ONCE at dispatch (execute) so the handler AND every guard (protected-path,
# protected-artifact, shrink) resolve the identical target — no desync bypass.
# apply_patch/edit_batch are absent because they carry no top-level `path` arg
# (their paths live inside the patch text / edits[] entries), so this seam has
# nothing to rewrite. They are NOT exempt from the canonicalization itself: both
# the dispatch guards below and their handlers run every payload path through
# `canonical_repo_relative_path`, the same normalization this seam applies.
_PATH_NORMALIZED_TOOLS = frozenset({"read_file", "write_file", "edit_text", "list_files", "search_code", "query_code"})

# Repo-lane write tools that take a top-level `root` arg. Every gate keyed to
# "a write that lands in the repo working tree" must judge the whole set, not
# the historical write_file/edit_text pair — a new editing primitive that misses
# one of these gates is a silently weaker lane, not a new capability.
_ROOT_ARG_REPO_WRITE_TOOLS = frozenset({"write_file", "edit_text", "apply_patch", "edit_batch"})


def _payload_write_paths(name: str, args: Dict[str, Any]) -> List[str]:
    """Repo paths a write tool will touch, in the spelling its guards must judge.

    write_file/edit_text carry `path`/`files[]` and were already canonicalized by
    `_normalize_dispatch_path_args`. apply_patch addresses files inside the patch
    text (`*** Update File: <path>`) and edit_batch inside `edits[]`, so their
    paths reach this point RAW and are canonicalized here — otherwise a
    protected-path gate reads `repo/BIBLE.md` (not a protected-table member)
    while the write lands on `BIBLE.md`.
    """

    paths: List[str] = []
    if name == "write_file":
        if isinstance(args.get("path"), str) and args["path"]:
            paths.append(args["path"])
        for entry in args.get("files") or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
    elif name == "edit_text":
        if isinstance(args.get("path"), str):
            paths.append(args["path"])
    elif name == "edit_batch":
        for entry in args.get("edits") or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
    elif name == "apply_patch":
        # Derived from the REAL parser (lazy import: edit_ops imports this
        # module), so the gate can never drift from what apply_patch will do.
        # An unparseable patch yields no paths and is refused by the handler
        # before any write, so the gate has nothing to miss.
        from ouroboros.tools.edit_ops import patch_target_paths

        paths.extend(patch_target_paths(str(args.get("patch") or "")))
    return [p for p in paths if str(p or "").strip()]


def _normalize_dispatch_path_args(ctx: Any, name: str, args: Dict[str, Any]) -> str:
    """ROOT-FIX (v6.35.0): normalize an absolute / redundant-root-basename
    active_workspace|system_repo path arg IN PLACE at the dispatch boundary, so
    the handler AND every downstream guard (protected-path, protected-artifact,
    accidental-truncation shrink guard) resolve the SAME target. One authoritative
    normalization point is what makes a guard unable to desync from the operation.

    v6.54.3 root-label fix: returns a dispatch note ("" when nothing rerouted).
    When ``root='user_files'`` carries an ABSOLUTE path that resolves under the
    ACTIVE WORKSPACE root, the root label is wrong, not the intent: reads
    (read_file/list_files/search_code) are auto-routed to
    ``root='active_workspace'`` with a visible note appended AFTER the result
    (trailing, so first-line failure classification is never masked),
    and writes (write_file/edit_text) return an actionable
    ROOT_REQUIRED_ACTIVE_WORKSPACE redirect instead of a generic access denial.
    The destination root still passes every downstream gate (profile access
    decision, protected-path guards, subagent filters) — only the label is
    corrected, never the authority. ``query_code`` is excluded: its
    root=user_files external-target contract handles absolute paths natively."""
    if name not in _PATH_NORMALIZED_TOOLS:
        return ""
    root_arg = str(args.get("root") or "active_workspace")
    if root_arg in ("active_workspace", "system_repo"):
        try:
            norm_root = active_repo_dir_for(ctx) if root_arg == "active_workspace" else system_repo_dir_for(ctx)
            for _key in ("path", "dir"):
                if isinstance(args.get(_key), str) and args[_key]:
                    args[_key] = normalize_root_relative(norm_root, args[_key])
            if isinstance(args.get("files"), list):
                for _f in args["files"]:
                    if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                        _f["path"] = normalize_root_relative(norm_root, _f["path"])
        except Exception:
            pass
        return ""
    if root_arg != "user_files" or name == "query_code":
        return ""
    try:
        workspace = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    except Exception:
        return ""

    def _under_workspace(text: str) -> bool:
        if not is_absolute_path_text(text):
            return False
        try:
            pathlib.Path(text).expanduser().resolve(strict=False).relative_to(workspace)
            return True
        except (ValueError, OSError, RuntimeError):
            return False

    candidates: list[str] = []
    for _key in ("path", "dir"):
        if isinstance(args.get(_key), str) and args[_key]:
            candidates.append(args[_key])
    if isinstance(args.get("files"), list):
        for _f in args["files"]:
            if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                candidates.append(_f["path"])
    hits = [text for text in candidates if _under_workspace(text)]
    if not hits:
        return ""
    if name in ("write_file", "edit_text"):
        return (
            "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: absolute path "
            f"{hits[0]!r} is under the active workspace, but root='user_files' does not "
            "write there. Retry the same call with root='active_workspace' (the same "
            "path is accepted)."
        )
    args["root"] = "active_workspace"
    try:
        for _key in ("path", "dir"):
            if isinstance(args.get(_key), str) and args[_key]:
                args[_key] = normalize_root_relative(workspace, args[_key])
        if isinstance(args.get("files"), list):
            for _f in args["files"]:
                if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                    _f["path"] = normalize_root_relative(workspace, _f["path"])
    except Exception:
        pass
    return (
        "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: absolute path "
        f"{hits[0]!r} is under the active workspace; the call ran with "
        "root='active_workspace'. Pass root='active_workspace' directly for "
        "workspace paths."
    )


_WEB_TOOLS = frozenset({"web_search", "browse_page", "browser_action", "youtube_transcript"})
_REPO_MUTATION_TOOLS = frozenset({
    "write_file",
    "commit_reviewed",
    "vcs_commit_reviewed",
    "edit_text",
    "apply_patch",
    "edit_batch",
    "vcs_revert",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_rollback",
    "promote_to_stable",
    # PR integration tools mutate the local worktree/refs.
    "fetch_pr_ref",
    "create_integration_branch",
    "cherry_pick_pr_commits",
    "stage_adaptations",
    "stage_pr_merge",
})
_SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
    "vcs_rollback",
    "promote_to_stable",
    "fetch_pr_ref",
    "create_integration_branch",
    "cherry_pick_pr_commits",
    "stage_adaptations",
    "stage_pr_merge",
})


def _resource_allowed(ctx: Any, key: str) -> bool:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = getattr(ctx, "task_contract")
    resources = {}
    for source in (metadata, contract):
        raw = source.get("allowed_resources") if isinstance(source, dict) else None
        if isinstance(raw, dict):
            resources.update(raw)
    if not resources:
        return True
    for name in (key, f"allow_{key}"):
        value = resources.get(name)
        if isinstance(value, bool):
            return value
    if key == "web":
        for name in ("network", "allow_network", "internet", "external_network"):
            value = resources.get(name)
            if isinstance(value, bool) and not value:
                return False
    if key == "network":
        for name in ("web", "allow_web", "internet", "external_network"):
            value = resources.get(name)
            if isinstance(value, bool) and not value:
                return False
    return True


def _disabled_tools(ctx: Any) -> frozenset:
    """Tool names the task contract withholds (declarative tool policy).

    Independent of ``allowed_resources``: a caller can disable specific tools
    (e.g. the agent's web_search/browser/VLM tools for a faithful benchmark)
    WITHOUT setting web/network=false — so shell network egress (git/pip) stays
    available and the web<->network cross-implication in ``_resource_allowed``
    never fires.
    """
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = getattr(ctx, "task_contract")
    names: set = set()
    for source in (metadata, contract):
        raw = source.get("disabled_tools") if isinstance(source, dict) else None
        if isinstance(raw, (list, tuple)):
            names.update(str(n).strip() for n in raw if str(n).strip())
    # D10 compatibility: `claude_code_edit` was retired; saved contracts that
    # withheld the external coding gateway keep withholding its SUCCESSOR — the
    # delegated coding session's start verb. The dead name stays in the set
    # too (harmless: nothing registers it), so old contracts round-trip as-is.
    if "claude_code_edit" in names:
        names.add("delegate_start")
    # Q1 rename compatibility: contracts that withheld `advisory_review` keep
    # withholding the SAME organ under its new name, and vice versa (a new
    # contract naming only the new spelling must also silence the alias).
    if "advisory_review" in names:
        names.add("preflight_review")
    if "preflight_review" in names:
        names.add("advisory_review")
    return frozenset(names)


def _presence_tool_allowed(ctx: Any, name: str) -> bool:
    """Positive ceiling for host-admitted presence work; absent is byte-compatible."""

    from ouroboros.presence_authority import (
        presence_ceiling_allows_tool,
        presence_ceiling_from_context,
    )

    ceiling = presence_ceiling_from_context(ctx)
    if name in {"presence_finish", "presence_cancel_work"}:
        return ceiling is not None
    return ceiling is None or presence_ceiling_allows_tool(ceiling, name)


def _presence_binding_allowed(ctx: Any, binding: Any) -> bool:
    from ouroboros.presence_authority import (
        presence_ceiling_allows_binding,
        presence_ceiling_from_context,
    )

    ceiling = presence_ceiling_from_context(ctx)
    if ceiling is None or binding is None:
        return True
    items = binding if isinstance(binding, tuple) else (binding,)
    return bool(items) and all(presence_ceiling_allows_binding(ceiling, item) for item in items)


def _presence_bound_args(ctx: Any, name: str, args: Any) -> tuple[dict[str, Any], str]:
    try:
        from ouroboros.presence_authority import apply_presence_argument_bindings

        bound = apply_presence_argument_bindings(ctx, name, dict(args or {}))
        if not _presence_tool_allowed(ctx, name):
            return {}, (
                "⚠️ PRESENCE_CAPABILITY_BLOCKED: "
                f"{name!r} is outside this presence task's positive capability ceiling."
            )
        return bound, ""
    except Exception as exc:
        return {}, f"⚠️ PRESENCE_ARGUMENT_BINDING_BLOCKED: {exc}"


_GITHUB_TOKEN_TOOLS = frozenset({
    "list_github_prs",
    "get_github_pr",
    "comment_on_pr",
    "list_github_issues",
    "get_github_issue",
    "comment_on_issue",
    "close_github_issue",
    "create_github_issue",
    "run_ci_tests",
    "submit_skill_to_hub",
    "generate_evolution_stats",
})

_TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
    "*": {"max_entries": "max_results"},
}
_IGNORE_ROOT_ARG_TOOLS = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
})
_GENERIC_VCS_TARGET_TOOLS = frozenset({
    "vcs_status",
    "vcs_diff",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_revert",
})

_TARGET_BINDING_OPERATIONS = {
    "read_file": "read",
    "list_files": "list",
    "search_code": "search",
    "query_code": "search",
    "write_file": "write",
    "edit_text": "edit",
    "apply_patch": "edit",
    "edit_batch": "edit",
    **{name: "vcs" for name in _GENERIC_VCS_TARGET_TOOLS},
}
_SKILL_LIFECYCLE_TARGET_TOOLS = frozenset({
    "skill_review",
    "skill_preflight",
    "submit_skill_to_hub",
})
_PROCESS_TARGET_TOOLS = frozenset({"run_command", "run_script", "start_service"})
_VERIFY_RUN_KINDS = frozenset({
    "visible_verifier",
    "explicit_command",
    "explicit_metric",
})


def _configured_delegate_selector(ctx: Any, name: str, args: dict[str, Any]) -> bool:
    """Configured-actor validation PRECEDES generic target binding.

    A configured session child passing any exact-resource selector gets the
    handler's precise configured_actor_resource_mismatch (which also records
    the START_BLOCKED attempt row), not the generic payload-binding
    TOOL_ACCESS_BLOCKED that twice read as "workspace authority lost" and
    pushed the nanny into native rebuilds."""
    return (
        name == "delegate_start"
        and isinstance(getattr(ctx, "_configured_actor_bootstrap", None), dict)
        and not str(args.get("retry_of") or "").strip()
        and any(str(args.get(key) or "").strip()
                for key in ("root", "bucket", "skill_name"))
    )


def _target_binding_operation(name: str, args: dict[str, Any]) -> str | None:
    operation = _TARGET_BINDING_OPERATIONS.get(name)
    if operation is not None:
        return operation
    if name in _SKILL_LIFECYCLE_TARGET_TOOLS:
        return "review"
    if name in _PROCESS_TARGET_TOOLS:
        return "service" if name == "start_service" else "shell"
    if name == "verify_and_record" and str(args.get("contract_kind") or "") in _VERIFY_RUN_KINDS:
        return "shell"
    # CONDITIONAL, never a static map entry (R1 item 1): delegate_start becomes
    # target-bound only when it explicitly selects an exact skill payload; a
    # plain or retry call keeps its current active-workspace behavior untouched.
    # ONLY the known selector value binds here — any other root value falls
    # through to the handler's TYPED unsupported_root refusal instead of an
    # untyped ValueError from binding construction (gate fix 9).
    if (name == "delegate_start"
            and str(args.get("root") or "").strip() == "skill_payload"
            and not str(args.get("retry_of") or "").strip()):
        return "write"
    return None


def _builtin_tool_availability(name: str, ctx: Any = None) -> tuple[bool, str, str]:
    """Return ``(available, reason, detail)`` for built-in tool credential gates.

    Predicates are lazy to avoid registry import cycles and discovery-time side effects.
    """
    # A bare registry (unit tests, static policy inventory, import-time introspection)
    # is a structural surface, not a running task capability envelope.
    if not str(getattr(ctx, "task_id", "") or "").strip():
        metadata = getattr(ctx, "task_metadata", {}) if ctx is not None else {}
        contract = getattr(ctx, "task_contract", {}) if ctx is not None else {}
        if not metadata and not contract:
            return True, "", ""
    tool = str(name or "").strip()
    if tool == "web_search":
        try:
            from ouroboros.tools.search import _available_web_search_backends

            if not _available_web_search_backends():
                return False, "missing_credential", "web_search_backend"
        except ImportError:
            return True, "", ""
        except Exception:
            return True, "", ""
    if tool in _GITHUB_TOKEN_TOOLS and not os.environ.get("GITHUB_TOKEN", "").strip():
        return False, "missing_credential", "GITHUB_TOKEN"
    return True, "", ""


def _handler_public_params(handler: Callable[..., Any]) -> list[str]:
    try:
        params = list(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return []
    return [name for name in params if name not in {"ctx", "_resolved_binding"}]


def _entry_public_params(entry: "ToolEntry") -> list[str]:
    try:
        params = entry.schema.get("parameters") or {}
        props = params.get("properties")
        if isinstance(props, dict):
            return [str(name) for name in props]
    except Exception:
        pass
    return _handler_public_params(entry.handler)


def _entry_has_public_param_schema(entry: "ToolEntry") -> bool:
    try:
        params = entry.schema.get("parameters") or {}
        return isinstance(params.get("properties"), dict)
    except Exception:
        return False


def _normalize_tool_call_args(entry: "ToolEntry", args: dict[str, Any]) -> None:
    tool_name = entry.name
    accepted = set(_entry_public_params(entry))
    aliases: dict[str, str] = {}
    aliases.update(_TOOL_ARG_ALIASES.get("*", {}))
    aliases.update(_TOOL_ARG_ALIASES.get(tool_name, {}))
    for alias, canonical in aliases.items():
        if alias in args and canonical in accepted and alias not in accepted and canonical not in args:
            args[canonical] = args.pop(alias)
    if tool_name in _IGNORE_ROOT_ARG_TOOLS and "root" in args and "root" not in accepted:
        args.pop("root", None)


def _prepare_public_builtin_args(entry: "ToolEntry", args: dict[str, Any]) -> str:
    """Normalize and validate only the model-visible builtin argument surface.

    This runs after capability/lineage availability checks but before path
    normalization, target selection, Python predispatch, or target-sensitive
    guards. Private dispatch carriers therefore cannot be supplied by the model
    and invalid public calls cannot trigger target work before rejection.
    """

    _normalize_tool_call_args(entry, args)
    public_params = set(_entry_public_params(entry))
    # A handler may name a bounded set of execution-only legacy parameters.  They
    # remain absent from its model-visible schema and are therefore usable only by
    # callers replaying the former wire shape through this real registry path.  The
    # handler still owns deterministic migration/refusal; this generic seam neither
    # chooses a route nor special-cases a tool name.
    hidden_legacy = {
        str(name)
        for name in (getattr(entry.handler, "_hidden_legacy_params", ()) or ())
        if str(name)
    }
    accepted_params = public_params | hidden_legacy
    if _entry_has_public_param_schema(entry) and any(key not in accepted_params for key in args):
        return _format_tool_arg_error(entry)
    try:
        inspect.signature(entry.handler).bind(object(), **args)
    except TypeError:
        return _format_tool_arg_error(entry)
    return ""


def _build_builtin_target_binding(ctx: Any, name: str, args: dict[str, Any]) -> Any:
    """Build the one private physical-target carrier for a builtin call."""

    operation = _target_binding_operation(name, args)
    if operation is None:
        return None
    if name in _SKILL_LIFECYCLE_TARGET_TOOLS:
        return build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="review",
            path=".",
            skill_name=str(args.get("skill") or ""),
        )
    if name in _PROCESS_TARGET_TOOLS or name == "verify_and_record":
        return build_resolved_resource_binding(
            ctx,
            operation=operation,
            process_cwd=str(args.get("cwd") or ""),
            bucket=str(args.get("bucket") or ""),
            skill_name=str(args.get("skill_name") or ""),
        )
    if name == "delegate_start":
        return build_resolved_resource_binding(
            ctx,
            root=str(args.get("root") or ""),
            operation="write",
            path=".",
            bucket=str(args.get("bucket") or ""),
            skill_name=str(args.get("skill_name") or ""),
        )
    root = str(args.get("root") or "active_workspace")
    bucket = str(args.get("bucket") or "")
    skill_name = str(args.get("skill_name") or "")

    def _one(path: str) -> Any:
        return build_resolved_resource_binding(
            ctx,
            root=root,
            operation=operation,
            path=path or ".",
            bucket=bucket,
            skill_name=skill_name,
        )

    if name == "write_file" and args.get("files"):
        return tuple(
            _one(str(item.get("path") or ""))
            for item in args.get("files") or []
            if isinstance(item, dict)
        )
    if name == "apply_patch":
        from ouroboros.tools.edit_ops import patch_target_paths

        return tuple(_one(path) for path in patch_target_paths(str(args.get("patch") or "")))
    if name == "edit_batch":
        return tuple(
            _one(str(item.get("path") or ""))
            for item in args.get("edits") or []
            if isinstance(item, dict)
        )
    return _one(str(args.get("path") or "."))


def _binding_items(binding: Any) -> tuple[Any, ...]:
    if binding is None:
        return ()
    return binding if isinstance(binding, tuple) else (binding,)


def _binding_set_targets_system_repo(ctx: Any, binding: Any) -> bool:
    items = _binding_items(binding)
    return bool(items) and all(binding_targets_system_repo(ctx, item) for item in items)


def _binding_set_is_light_restricted(ctx: Any, binding: Any) -> bool:
    """Whether light mode must treat this file/VCS target as internal state."""
    items = _binding_items(binding)
    return bool(items) and all(
        binding_targets_system_repo(ctx, item)
        or (item.root == "runtime_data" and item.source == "runtime_data")
        for item in items
    )


def _light_binding_failure_redirect(name: str, args: dict[str, Any]) -> str:
    """Project an existing light-mode UX redirect after a failed target bind."""

    try:
        from ouroboros.config import get_runtime_mode

        if get_runtime_mode() == "light":
            return light_cognitive_or_root_redirect(name, args) or ""
    except Exception:
        pass
    return ""


def _binding_error_text(name: str, root: str, exc: Exception) -> str:
    detail = str(exc)
    if detail.startswith("SKILL_REDIRECT_BLOCKED:"):
        return f"⚠️ {detail}"
    if detail.startswith("profile=") and " cannot " in detail:
        return f"⚠️ TOOL_ACCESS_BLOCKED: {detail.rstrip('.')}."
    if isinstance(exc, UserFilesPathBlockedError) and name in {
        "read_file", "list_files", "search_code",
    }:
        return f"⚠️ USER_FILES_PATH_BLOCKED: {detail}"
    if root == "skill_payload" and name in {"write_file", "edit_text"}:
        return f"⚠️ SKILL_PAYLOAD_ARG_ERROR: {detail}"
    prefixes = {
        "read_file": "READ_FILE_ERROR",
        "list_files": "LIST_FILES_ERROR",
        "search_code": "SEARCH_ERROR",
        "query_code": "TOOL_ARG_ERROR (query_code)",
        "write_file": "WRITE_FILE_ERROR",
        "edit_text": "EDIT_TEXT_ERROR",
        "vcs_status": "GIT_ERROR",
        "vcs_diff": "GIT_ERROR",
        "vcs_pull_ff": "PULL_ERROR",
        "vcs_restore": "RESTORE_ERROR",
        "vcs_revert": "REVERT_ERROR",
        "skill_review": "SKILL_REVIEW_ERROR",
        "skill_preflight": "SKILL_PREFLIGHT_ERROR",
        "submit_skill_to_hub": "SUBMIT_BLOCKED",
        "run_command": "SHELL_CWD_BLOCKED",
        "run_script": "SCRIPT_CWD_BLOCKED",
        "start_service": "SHELL_CWD_BLOCKED",
        "verify_and_record": "VERIFY_ERROR",
    }
    return f"⚠️ {prefixes.get(name, 'TOOL_ERROR')}: {type(exc).__name__}: {detail}"


def _payload_dispatch_constraint(
    ctx: Any,
    *,
    name: str,
    args: dict[str, Any],
    task_constraint: Optional[TaskConstraint],
    workspace_mode: bool,
) -> tuple[Optional[TaskConstraint], str]:
    """Preserve repair selectors without letting stray selectors retarget work."""

    raw_bucket = str(args.get("bucket", "") or "")
    raw_skill_name = str(args.get("skill_name", "") or "")
    explicit_skill_root = str(args.get("root", "") or "").strip().lower() == "skill_payload"
    short_form_decision = None if explicit_skill_root else decide_payload_short_form(
        bucket=raw_bucket,
        skill_name=raw_skill_name,
        path_text=str(args.get("path", "") or "."),
        repo_dir=pathlib.Path(ctx.repo_dir),
        drive_root=pathlib.Path(ctx.drive_root),
    )
    if explicit_skill_root:
        # Binding selection already handled the explicit target. This legacy
        # constraint exists only for the light-mode data-payload carve-out.
        synthesized = synthesize_payload_constraint(raw_bucket, raw_skill_name)
    else:
        synthesized = (
            short_form_decision.constraint
            if short_form_decision is not None
            and task_constraint
            and task_constraint.mode == "skill_repair"
            else None
        )

    if (
        (raw_bucket or raw_skill_name)
        and short_form_decision is not None
        and short_form_decision.error
        and name in {"write_file", "edit_text"}
    ):
        root_arg = str(args.get("root", "") or "").strip().lower()
        if _stray_skill_payload_failsoft(root_arg, workspace_mode, task_constraint):
            log.info(
                "Ignoring stray bucket/skill_name on %s (workspace edit, root=%s): %s",
                name,
                root_arg or "active_workspace",
                short_form_decision.error[:80],
            )
            args.pop("bucket", None)
            args.pop("skill_name", None)
            synthesized = None
        else:
            return None, f"⚠️ SKILL_PAYLOAD_ARG_ERROR: {short_form_decision.error}"

    redirect_err = cross_skill_redirect_error(task_constraint, synthesized)
    if redirect_err and name in {"write_file", "edit_text"}:
        return None, f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
    if task_constraint and task_constraint.mode == "skill_repair":
        return task_constraint, ""
    return synthesized or task_constraint, ""


def _format_tool_arg_error(entry: "ToolEntry") -> str:
    params = _entry_public_params(entry)
    accepted = ", ".join(params) if params else "none"
    return (
        f"⚠️ TOOL_ARG_ERROR ({entry.name}): invalid arguments for {entry.name}. "
        f"Accepted parameters: {accepted}."
    )


def _light_repo_snapshot(repo_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Worktree tripwire for light-mode shell writes, not rollback machinery."""
    try:
        repo = pathlib.Path(repo_dir)
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if status.returncode != 0:
            return None
        unstaged = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        paths = parse_porcelain_paths(status.stdout)
        digest = hashlib.sha256()
        digest.update((status.stdout or "").encode("utf-8", errors="replace"))
        digest.update((unstaged.stdout if unstaged.returncode == 0 else "").encode("utf-8", errors="replace"))
        digest.update((staged.stdout if staged.returncode == 0 else "").encode("utf-8", errors="replace"))
        for rel in paths:
            try:
                target = (repo / safe_relpath(rel)).resolve(strict=False)
                target.relative_to(repo.resolve(strict=False))
                if target.is_file() and rel in (status.stdout or ""):
                    stat = target.stat()
                    digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8"))
            except Exception:
                continue
        return {"digest": digest.hexdigest(), "paths": paths}
    except Exception:
        return None


def _format_light_repo_write_note(before: Dict[str, Any], after: Dict[str, Any], tool_name: str = "run_command") -> str:
    """The light-lane tripwire NOTE, appended after the command payload (#447 В12)."""
    before_paths = set(before.get("paths") or [])
    after_paths = set(after.get("paths") or [])
    touched = sorted(after_paths | before_paths)
    listed = ", ".join(touched[:30]) if touched else "(status changed; no paths parsed)"
    if len(touched) > 30:
        listed += f", ... (+{len(touched) - 30} more)"
    return (
        "⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: runtime_mode=light detected "
        f"a mutation of the Ouroboros repository after {tool_name}. "
        "The command result is blocked and no automatic rollback was attempted "
        "to avoid overwriting concurrent human edits. "
        f"Affected/dirty paths: {listed}. Switch to advanced/pro for repo writes."
    )


def _git_ref_snapshot(repo_dir: pathlib.Path) -> Optional[Dict[str, str]]:
    try:
        repo = pathlib.Path(repo_dir)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        refs = subprocess.run(
            ["git", "show-ref", "--head", "--dereference"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0 or refs.returncode not in (0, 1):
            return None
        digest = hashlib.sha256()
        digest.update((head.stdout or "").encode("utf-8", errors="replace"))
        digest.update((refs.stdout or "").encode("utf-8", errors="replace"))
        return {"head": (head.stdout or "").strip(), "digest": digest.hexdigest()}
    except Exception:
        return None


@dataclass
class BrowserState:
    """Per-task Playwright lifecycle state."""

    pw_instance: Any = None
    browser: Any = None
    page: Any = None
    last_screenshot_b64: Optional[str] = None


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
_EPHEMERAL_ALLOWED_TOOLS = frozenset({
    # read / inspect
    "read_file", "query_code", "search_code", "list_files", "web_search", "browse_page",
    "chat_history", "recent_tasks", "get_task_result", "vcs_diff", "vcs_status",
    "analyze_screenshot", "vlm_query",
    # decide / route / spawn-owner-task / reply
    "route_to_project", "promote_chat_to_task", "steer_task", "list_projects", "send_photo",
})


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


@dataclass
class ToolEntry:
    """Single tool descriptor."""

    name: str
    schema: Dict[str, Any]
    handler: Callable  # fn(ctx: ToolContext, **args) -> str
    is_code_tool: bool = False
    timeout_sec: int = 360
    # Capability flag: tool can mutate the live repo worktree. The dispatcher
    # snapshots `git status --porcelain` around flagged tools and invalidates
    # advisory freshness when the worktree ACTUALLY changed — covering error
    # and timeout paths uniformly, and never invalidating for read-only runs.
    mutates_worktree: bool = False
    # Compatibility alias: a renamed tool's old public name. An alias entry is
    # CALLABLE (execute dispatches it like any entry) but never advertised —
    # schemas()/available_tools() skip it, so the public surface carries only
    # the canonical name while saved prompts, memories, and configs that still
    # use the old spelling keep working.
    alias_for: str = ""


class ToolRegistry:
    """Tool registry; modules export ``get_tools()``."""

    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self._entries: Dict[str, ToolEntry] = {}
        self._ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
        # Load-time omissions are FACTS OF THIS PROCESS (a tool module that failed
        # to import stays missing until restart); schemas() rebuilds start from
        # them instead of an empty list so the ledger never forgets them (H3).
        self._module_load_omissions: List[Dict[str, Any]] = []
        self._capability_omissions: List[Dict[str, Any]] = []
        self._load_modules()
        self._capability_omissions = [dict(item) for item in self._module_load_omissions]

    _FROZEN_TOOL_MODULES = [
        "browser", "ci", "claude_advisory_review", "compact_context", "control",
        "core", "delegate", "edit_ops", "evolution_stats", "followup", "git", "git_pr", "git_rollback", "github",
        "health", "join_ledger", "knowledge", "media", "memory_tools", "plan_review", "project_journal", "presence",
        "recent_tasks",
        "query_code", "review", "search", "services", "shell", "skill_exec", "skill_publish",
        "skill_preflight", "subagent_integration", "task_tree", "tool_discovery", "verify", "vision",
    ]

    def _load_modules(self) -> None:
        """Load frozen or package-discovered tool modules."""
        import importlib
        import logging
        import sys

        if getattr(sys, 'frozen', False):
            module_names = self._FROZEN_TOOL_MODULES
        else:
            import pkgutil
            import ouroboros.tools as tools_pkg
            module_names = [
                m for _, m, _ in pkgutil.iter_modules(tools_pkg.__path__)
                if not m.startswith("_") and m != "registry"
            ]

        for modname in module_names:
            try:
                mod = importlib.import_module(f"ouroboros.tools.{modname}")
                if hasattr(mod, "get_tools"):
                    for entry in mod.get_tools():
                        self._entries[entry.name] = entry
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to load tool module %s", modname, exc_info=True)
                # A failed module silently omits EVERY tool it exports; record it
                # in the durable capability ledger, not only the process log (H3).
                self._module_load_omissions.append({
                    "surface": "tools",
                    "reason": "module_load_failed",
                    "module": modname,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    def set_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    def register(self, entry: ToolEntry) -> None:
        """Register a new tool entry."""
        self._entries[entry.name] = entry

    # Contract.

    def _ctx_is_delegated_subagent(self) -> bool:
        for attr in ("task_metadata", "task_contract"):
            data = getattr(self._ctx, attr, None)
            if isinstance(data, dict) and str(data.get("delegation_role") or "").strip() == "subagent":
                return True
        return False

    def _is_local_readonly_subagent(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        if tc and tc.mode == LOCAL_READONLY_SUBAGENT_MODE:
            return True
        # Fail-closed (mirror active_tool_profile): a valid acting constraint is
        # acting; a malformed acting constraint, or any delegated subagent without
        # a valid acting constraint (incl. a missing constraint), resolves read-only.
        if self._is_acting_subagent():
            return False
        if tc and tc.mode == ACTING_SUBAGENT_MODE:
            return True
        return self._ctx_is_delegated_subagent()

    def _is_acting_subagent(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return bool(
            tc and tc.mode == ACTING_SUBAGENT_MODE
            and str(getattr(tc, "surface", "") or "") in VALID_WRITE_SURFACES
        )

    def _acting_self_worktree(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return bool(
            tc and getattr(tc, "mode", "") == ACTING_SUBAGENT_MODE
            and str(getattr(tc, "surface", "") or "") == "self_worktree"
        )

    def _deliverables_shell_target_allowed(
        self,
        candidate: pathlib.Path,
        *,
        lexical_candidate: pathlib.Path | None = None,
    ) -> bool:
        """Return whether a top-level user-files shell may write this target.

        The workspace shell guard owns the process-root boundary.  This narrow
        exception reuses the user-files policy and the configured Deliverables
        root for the one existing top-level profile that already has
        ``user_files:shell``.  Delegated children never inherit the carve-out.
        """
        if self._is_acting_subagent() or self._is_local_readonly_subagent():
            return False
        profile = active_tool_profile(self._ctx)
        if not decide_tool_access(
            profile=profile,
            root="user_files",
            operation="shell",
        ).allow:
            return False
        try:
            if lexical_user_files_block_reason(lexical_candidate or candidate):
                return False
            target = pathlib.Path(candidate).resolve(strict=False)
            deliverables = resource_root_path(self._ctx, "deliverables")
            # Validate the configured container itself before admitting a child.
            # A root that contains a protected repo/data drive is not a genuine
            # sibling; checking only the final file would otherwise turn its
            # harmless-looking sibling paths into a broad parent escape.
            if user_files_path_block_reason(self._ctx, deliverables):
                return False
            if not (
                target.is_relative_to(deliverables)
                or _path_is_relative_to_casefold(target, deliverables)
            ):
                return False
            try:
                deliverable_binding = build_resolved_resource_binding(
                    self._ctx,
                    root="user_files",
                    operation="shell",
                    path=str(target),
                )
            except (OSError, TypeError, ValueError, RuntimeError):
                return False
            if not _presence_binding_allowed(self._ctx, deliverable_binding):
                return False
            return not user_files_path_block_reason(self._ctx, target)
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def _acting_tool_grants(self) -> set:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return set(getattr(tc, "external_tool_grants", ()) or ()) if tc else set()

    def _readonly_tool_allowed(self, name: str) -> bool:
        """Return the read-only child allowlist, including the narrow actor receipt.

        ``verify_and_record`` is normally an acting/workspace code tool because its
        other contract kinds execute commands or inspect deliverables.  An
        actor-first configured session needs one exceptional, non-shell use: a
        typed ``delegation_zero_run`` receipt when its selected leaf is still
        pending.  Keep that exception bound to the private host bootstrap marker;
        the handler applies the same check again at execution time.
        """
        if name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
            return True
        if name != "verify_and_record" or not self._is_local_readonly_subagent():
            return False
        bootstrap = getattr(self._ctx, "_configured_actor_bootstrap", None)
        return (
            isinstance(bootstrap, dict)
            and not bool(bootstrap.get("physical_started"))
            and not bool(bootstrap.get("zero_run_receipt_recorded"))
            and (
                bool(bootstrap.get("exact_start_pending", True))
                or str(bootstrap.get("zero_run_evidence_status") or "") == "unknown"
            )
        )

    def initial_tool_names(self) -> frozenset[str]:
        if self._is_local_readonly_subagent():
            names = set(LOCAL_READONLY_SUBAGENT_TOOL_NAMES)
            if self._readonly_tool_allowed("verify_and_record"):
                names.add("verify_and_record")
            return frozenset(names)
        if self._is_acting_subagent():
            return ACTING_SUBAGENT_TOOL_NAMES
        return frozenset(set(self.available_tools()) | set(META_TOOL_NAMES))

    def available_tools(self) -> List[str]:
        acting_subagent = self._is_acting_subagent()
        local_readonly_subagent = self._is_local_readonly_subagent()
        disabled = _disabled_tools(self._ctx)
        return [
            e.name
            for e in self._entries.values()
            if not e.alias_for  # compat aliases are callable, never advertised
            if e.name not in disabled  # declarative tool policy (task_contract.disabled_tools)
            if _presence_tool_allowed(self._ctx, e.name)
            if _builtin_tool_availability(e.name, self._ctx)[0]
            if not local_readonly_subagent or self._readonly_tool_allowed(e.name)
            if not acting_subagent or e.name in ACTING_SUBAGENT_TOOL_NAMES
        ]

    def _schema_for_entry(self, entry: ToolEntry) -> Dict[str, Any]:
        schema = entry.schema
        if self._is_local_readonly_subagent():
            if entry.name == "verify_and_record" and self._readonly_tool_allowed(entry.name):
                # The read-only actor is allowed to mint exactly one kind of
                # host receipt.  Do not expose the general verification schema,
                # whose other kinds can execute shell commands or inspect files.
                schema = copy.deepcopy(schema)
                schema["description"] = (
                    "Record the typed zero-run decision for this configured actor-first "
                    "session. No physical leaf may have started. This receipt is the "
                    "only verification available to a read-only actor."
                )
                parameters = schema.setdefault("parameters", {})
                properties = parameters.setdefault("properties", {})
                parameters["properties"] = {
                    key: properties[key]
                    for key in ("contract_kind", "criterion_id", "zero_run_decision", "zero_run_basis")
                    if key in properties
                }
                if "contract_kind" in parameters["properties"]:
                    parameters["properties"]["contract_kind"]["enum"] = ["delegation_zero_run"]
                parameters["required"] = ["contract_kind", "zero_run_decision", "zero_run_basis"]
            elif entry.name in {"read_file", "list_files", "search_code", "query_code"}:
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                if entry.name == "search_code":
                    allowed = {"active_workspace", "system_repo", "skill_payload"}
                elif entry.name == "query_code":
                    # query_code itself rejects non-repo roots — do not advertise more.
                    allowed = {"active_workspace", "system_repo"}
                else:
                    allowed = {"active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store"}
                if isinstance(root_schema.get("enum"), list): root_schema["enum"] = [root for root in root_schema["enum"] if root in allowed]
            elif entry.name in {"browse_page", "browser_action"}:
                schema = copy.deepcopy(entry.schema)
                if entry.name == "browse_page":
                    schema["description"] = "Open an HTTP(S) URL (external, or localhost on non-Ouroboros ports) or a file:// path under your workspace in a headless browser. Returns page content as text, html, markdown, or screenshot (base64 PNG) — use it with analyze_screenshot to visually verify your own built apps. The Ouroboros API ports, private/link-local IPs, and other URL schemes are blocked for subagents. Use viewport to test mobile layouts (e.g. '375x812')."
                if entry.name == "browser_action":
                    schema["description"] = "Perform action on the current browser page (external HTTP(S), localhost on non-Ouroboros ports, or a file:// page under your workspace). Actions: click (selector), fill (selector + value), select (selector + value), screenshot (base64 PNG), scroll (value: up/down/top/bottom). JavaScript evaluate is unavailable to local-readonly subagents."
                    props = schema.get("parameters", {}).get("properties", {})
                    action_schema = props.get("action", {})
                    if isinstance((action_enum := action_schema.get("enum")), list):
                        action_schema["enum"] = [name for name in action_enum if name != "evaluate"]
                    if isinstance((value_schema := props.get("value", {})), dict): value_schema["description"] = "Value for fill/select or direction for scroll"
            elif entry.name == "schedule_subagent":
                # A read-only subagent may delegate read-only children only — hide the
                # acting (mutative) fields so it cannot spawn an acting grandchild.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("write_surface", "write_root", "protected_paths_grant", "external_tool_grants"):
                    props.pop(field, None)
            elif entry.name == "delegate_start":
                # Same selector trap the acting branch hides: a readonly child
                # can never start a skill-payload resource either.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("root", "bucket", "skill_name"):
                    props.pop(field, None)
        elif self._is_acting_subagent():
            # Advertise only what the acting profile can actually execute: writes go
            # ONLY to the isolated surface (active_workspace); reads use the read roots;
            # browser evaluate remains available on the current page; the browser
            # handler retains its owner/self-lowering checks.
            if entry.name == "delegate_start":
                # The exact-resource selector is a TRAP here: an acting child
                # can never write root=skill_payload, yet the schema advertised
                # it as root's only enum value - twice a nanny took the bait,
                # was refused by generic binding before the precise
                # configured_actor_resource_mismatch could answer, and rebuilt
                # the work natively (the two Minecraft-widget incidents). Hide
                # the selector triple; bound starts and retry_of need none of
                # it; the top-level principal's schema is untouched.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("root", "bucket", "skill_name"):
                    props.pop(field, None)
            elif entry.name in _ROOT_ARG_REPO_WRITE_TOOLS or entry.name in _GENERIC_VCS_TARGET_TOOLS:
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                if isinstance(root_schema.get("enum"), list):
                    root_schema["enum"] = [root for root in root_schema["enum"] if root == "active_workspace"]
            elif entry.name in {"read_file", "list_files", "search_code", "query_code"}:
                # Acting profile reads its own surface + data roots, NOT the live
                # system_repo (no system_repo in _POLICY['acting_subagent']).
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                allowed = {"active_workspace"} if entry.name in {"search_code", "query_code"} else {"active_workspace", "runtime_data", "task_drive", "artifact_store"}
                if isinstance(root_schema.get("enum"), list):
                    root_schema["enum"] = [root for root in root_schema["enum"] if root in allowed]
        return {"type": "function", "function": schema}

    def _schemas_for_entry(self, entry: ToolEntry) -> List[Dict[str, Any]]:
        return [self._schema_for_entry(entry)]

    def schemas(self, core_only: bool = False) -> List[Dict[str, Any]]:
        acting_subagent = self._is_acting_subagent()
        acting_grants = self._acting_tool_grants() if acting_subagent else set()
        local_readonly_subagent = self._is_local_readonly_subagent()
        ephemeral_turn = bool(getattr(self._ctx, "is_ephemeral_turn", False))
        disabled_tools = _disabled_tools(self._ctx)
        # Rebuild from the load-time facts, never from empty: a rebuilt schema
        # list must not erase module_load_failed omissions (H3, capinv-447).
        self._capability_omissions = [dict(item) for item in self._module_load_omissions]
        unavailable_tools = {
            entry.name: detail
            for entry in self._entries.values()
            for available, reason, detail in [_builtin_tool_availability(entry.name, self._ctx)]
            if not available and reason == "missing_credential" and entry.name not in disabled_tools
        }
        built_in = [
            schema
            for entry in self._entries.values()
            if not entry.alias_for  # compat aliases are callable, never advertised
            if entry.name not in disabled_tools  # declarative tool policy (task_contract.disabled_tools)
            if _presence_tool_allowed(self._ctx, entry.name)
            if entry.name not in unavailable_tools
            if not local_readonly_subagent or self._readonly_tool_allowed(entry.name)
            if not acting_subagent or entry.name in ACTING_SUBAGENT_TOOL_NAMES
            if not ephemeral_turn or entry.name in _EPHEMERAL_ALLOWED_TOOLS  # CW3: default-deny allowlist
            for schema in self._schemas_for_entry(entry)
        ]
        if disabled_tools:
            self._capability_omissions.append({"surface": "tools", "reason": "disabled_by_contract", "tools": sorted(disabled_tools)})
        if unavailable_tools:
            self._capability_omissions.append({
                "surface": "tools",
                "reason": "missing_credential",
                "tools": sorted(unavailable_tools),
                "details": {name: unavailable_tools[name] for name in sorted(unavailable_tools)},
            })
        # Include live extension tool schemas in normal tool discovery.
        extension_schemas: List[Dict[str, Any]] = []
        if ephemeral_turn:
            # CW3: a short decision turn answers/routes/spawns/steers only — it gets no
            # extension surfaces, which can have durable/reviewed side effects.
            self._capability_omissions.append({"surface": "extensions", "reason": "ephemeral_turn"})
        elif not _resource_allowed(self._ctx, "network"):
            self._capability_omissions.append({"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"})
        else:
            try:
                from ouroboros.extension_loader import (
                    _tools as _ext_tools,
                    _lock as _ext_lock,
                    is_extension_live as _ext_is_live,
                )
                meta = getattr(self._ctx, "task_metadata", {})
                capability_root = pathlib.Path((meta.get("budget_drive_root") if isinstance(meta, dict) else "") or getattr(self._ctx, "budget_drive_root", "") or getattr(self._ctx, "drive_root", "") or ".").resolve(strict=False)
                with _ext_lock:
                    extension_schemas = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool.get("description", ""),
                                "parameters": tool.get("schema", {"type": "object", "properties": {}}),
                            },
                        }
                        for tool in _ext_tools.values()
                        if _ext_is_live(str(tool.get("skill") or ""), capability_root, repo_path=str(tool.get("skills_repo_path") or "") or None)
                        and _presence_tool_allowed(self._ctx, tool["name"])
                        and (not acting_subagent or tool["name"] in acting_grants)
                    ]
            except Exception as exc:
                self._capability_omissions.append({"surface": "extensions", "reason": "discovery_error", "error": f"{type(exc).__name__}: {exc}"})

        if not core_only:
            mcp_schemas = []
            if ephemeral_turn:
                # CW3: MCP tools can have durable side effects — not for a decision turn.
                self._capability_omissions.append({"surface": "mcp", "reason": "ephemeral_turn"})
            elif not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"})
            else:
                try:
                    from ouroboros.mcp_client import ensure_configured_from_settings as _mcp_ensure_configured, get_manager as _mcp_get_manager
                    _mcp_ensure_configured(refresh=True)
                    _mgr = _mcp_get_manager()
                    mcp_schemas = [
                        {
                            "type": "function",
                            "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("schema", {"type": "object", "properties": {}})},
                        }
                        for tool in _mgr.list_tools_for_registry()
                        if _presence_tool_allowed(self._ctx, tool["name"])
                        if not acting_subagent or tool["name"] in acting_grants
                    ]
                    # D1: an enabled+configured server returning zero tools WITHOUT
                    # raising (unreachable/slow/auth-failed) is otherwise silent. Make
                    # the reason visible so the model/owner learns WHY an expected MCP
                    # server produced no tools, instead of "the agent can't see MCP".
                    # Checked unconditionally so a broken server is surfaced even when a
                    # co-located healthy server contributed tools (does not mask it).
                    _empty = _mgr.enabled_servers_without_tools()
                    if _empty:
                        self._capability_omissions.append({"surface": "mcp", "reason": "server_no_tools", "servers": _empty})
                except Exception as exc:
                    self._capability_omissions.append({"surface": "mcp", "reason": "discovery_error", "error": f"{type(exc).__name__}: {exc}"})
            combined = built_in + extension_schemas + mcp_schemas
            if disabled_tools:
                # Apply the declarative tool policy to dynamic extension/MCP schemas too, not just
                # built-ins, so a disabled name can never surface from any discovery source.
                combined = [
                    s for s in combined
                    if (s.get("function", {}) or {}).get("name") not in disabled_tools
                ]
            return combined
        # Core tools plus meta-tools for enabling extended tools.
        result = []
        for e in self._entries.values():
            if e.alias_for:  # compat aliases are callable, never advertised
                continue
            if e.name in disabled_tools:  # declarative tool policy (task_contract.disabled_tools)
                continue
            if not _presence_tool_allowed(self._ctx, e.name):
                continue
            if e.name in unavailable_tools:
                continue
            if local_readonly_subagent and not self._readonly_tool_allowed(e.name):
                continue
            if acting_subagent and e.name not in ACTING_SUBAGENT_TOOL_NAMES:
                continue
            if ephemeral_turn and e.name not in _EPHEMERAL_ALLOWED_TOOLS:
                continue  # CW3: the core/initial envelope is allowlisted too, not just schemas(core_only=False)
            if (
                (local_readonly_subagent and self._readonly_tool_allowed(e.name))
                or (acting_subagent and e.name in ACTING_SUBAGENT_TOOL_NAMES)
                or e.name in CORE_TOOL_NAMES
                or e.name in ("list_available_tools", "enable_tools")
            ):
                result.extend(self._schemas_for_entry(e))
        ext = extension_schemas
        if disabled_tools:
            ext = [s for s in ext if (s.get("function", {}) or {}).get("name") not in disabled_tools]
        return result + ext

    def capability_omissions(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._capability_omissions]

    def policy_hidden_reason(self, name: str) -> Optional[str]:
        """Why a REGISTERED built-in tool is invisible to THIS task, or None.

        Read-only companion to get_schema_by_name (same predicates, same order):
        it distinguishes "hidden by policy" from "does not exist" so discovery
        answers can stop reporting a policy-filtered tool as nonexistent (F3,
        2026-08-10 saga). None means visible OR unknown name — callers that got
        no schema and no reason may honestly say "not found".
        """
        requested = str(name or "").strip()
        if not requested:
            return None
        # BEFORE the registration check: the declarative contract policy applies
        # across ALL discovery sources (get_schema_by_name checks it first for the
        # same reason), so a contract-disabled extension/MCP name answers with its
        # reason instead of "not found" (2026-08-10 amendments). Deeper extension/
        # MCP policy reasons (grants, network) would need new plumbing — disclosed
        # residual, not built.
        if requested in _disabled_tools(self._ctx):
            return "disabled by this task's contract (disabled_tools)"
        if not _presence_tool_allowed(self._ctx, requested):
            return "outside this presence task's positive capability ceiling"
        if requested not in self._entries:
            return None
        if self._entries[requested].alias_for:
            # Mirrors get_schema_by_name: a compat alias is callable but never
            # advertised — discovery answers as for any non-public name.
            return None
        available, reason, _detail = _builtin_tool_availability(requested, self._ctx)
        if not available:
            return f"unavailable ({reason})"
        if getattr(self._ctx, "is_ephemeral_turn", False) and requested not in _EPHEMERAL_ALLOWED_TOOLS:
            return "hidden on this ephemeral decision turn (allowlist)"
        acting_subagent = self._is_acting_subagent()
        if self._is_local_readonly_subagent() and not self._readonly_tool_allowed(requested):
            return "hidden by the read-only subagent profile"
        if acting_subagent and requested not in ACTING_SUBAGENT_TOOL_NAMES:
            return "hidden by the acting subagent profile"
        return None

    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the full schema for a specific tool."""
        requested = str(name or "").strip()
        acting_subagent = self._is_acting_subagent()
        acting_grants = self._acting_tool_grants() if acting_subagent else set()
        local_readonly_subagent = self._is_local_readonly_subagent()
        # Declarative tool policy applies across ALL discovery sources (built-in, extension, MCP),
        # so enable_tools/discovery can never surface a disabled name — consistent with schemas()/execute().
        if requested in _disabled_tools(self._ctx):
            return None
        if not _presence_tool_allowed(self._ctx, requested):
            return None
        entry = self._entries.get(requested)
        if entry and entry.alias_for:
            # Compat aliases are callable, never advertised — discovery answers
            # as it does for any non-public name.
            return None
        if entry:
            available, reason, detail = _builtin_tool_availability(requested, self._ctx)
            if not available:
                if reason == "missing_credential":
                    self._capability_omissions.append({
                        "surface": "tools",
                        "reason": reason,
                        "tools": [requested],
                        "details": {requested: detail},
                    })
                return None
            if getattr(self._ctx, "is_ephemeral_turn", False) and requested not in _EPHEMERAL_ALLOWED_TOOLS:
                return None  # CW3: allowlist-consistent with schemas()/execute() (so enable_tools can't surface a denied tool)
            if local_readonly_subagent and not self._readonly_tool_allowed(requested):
                return None
            if acting_subagent and requested not in ACTING_SUBAGENT_TOOL_NAMES:
                return None
            return self._schema_for_entry(entry)
        try:
            from ouroboros.extension_loader import parse_extension_surface_name as _ext_parse_name
        except Exception:
            _ext_parse_name = None
        if _ext_parse_name and _ext_parse_name(name):
            if acting_subagent and requested not in acting_grants:
                return None
            if not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"})
                return None
            try:
                from ouroboros.extension_loader import get_tool as _ext_get_tool, is_extension_live as _ext_is_live
                ext_tool = _ext_get_tool(name)
                meta = getattr(self._ctx, "task_metadata", {})
                capability_root = pathlib.Path((meta.get("budget_drive_root") if isinstance(meta, dict) else "") or getattr(self._ctx, "budget_drive_root", "") or getattr(self._ctx, "drive_root", "") or ".").resolve(strict=False)
            except Exception:
                ext_tool = None
            if (
                ext_tool
                and _ext_is_live(str(ext_tool.get("skill") or ""), capability_root, repo_path=str(ext_tool.get("skills_repo_path") or "") or None)
            ):
                return {
                    "type": "function",
                    "function": {
                        "name": ext_tool["name"],
                        "description": ext_tool.get("description", ""),
                        "parameters": ext_tool.get("schema", {"type": "object", "properties": {}}),
                    },
                }
        try:
            from ouroboros.mcp_client import (
                ensure_configured_from_settings as _mcp_ensure_configured,
                get_manager as _mcp_get_manager,
                is_mcp_tool_name as _mcp_is_name,
            )
            _mcp_ensure_configured(refresh=False)
        except Exception:
            _mcp_get_manager = None
            _mcp_is_name = None
        if _mcp_get_manager and _mcp_is_name and _mcp_is_name(requested):
            if acting_subagent and requested not in acting_grants:
                return None
            if not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"})
                return None
            mcp_tool = _mcp_get_manager().get_tool(requested)
            if mcp_tool:
                return {
                    "type": "function",
                    "function": {
                        "name": mcp_tool["name"],
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get("schema", {"type": "object", "properties": {}}),
                    },
                }
        return None

    def get_timeout(self, name: str) -> int:
        """Return timeout_sec for the named tool (default 360)."""
        entry = self._entries.get(str(name or "").strip())
        if entry is not None:
            return entry.timeout_sec
        # Extension tools carry timeout_sec in the loader descriptor.
        try:
            from ouroboros.extension_loader import parse_extension_surface_name as _ext_parse_name
        except Exception:
            _ext_parse_name = None
        if _ext_parse_name and _ext_parse_name(name):
            try:
                from ouroboros.extension_loader import get_tool as _ext_get_tool
                ext_tool = _ext_get_tool(name)
            except Exception:
                ext_tool = None
            if ext_tool:
                # Add cleanup grace around the inner async wait_for.
                return int(ext_tool.get("timeout_sec") or 60) + 3
        try:
            from ouroboros.mcp_client import (
                ensure_configured_from_settings as _mcp_ensure_configured,
                get_manager as _mcp_get_manager,
                is_mcp_tool_name as _mcp_is_name,
            )
            _mcp_ensure_configured(refresh=False)
        except Exception:
            _mcp_get_manager = None
            _mcp_is_name = None
        if _mcp_get_manager and _mcp_is_name and _mcp_is_name(name):
            try:
                return int(_mcp_get_manager().tool_timeout_sec()) + 3
            except Exception:
                return 63
        return 360

    def _dispatch_extension_tool(self, name: str, ext_tool: Dict[str, Any], args: Optional[Dict[str, Any]]) -> str:
        """Dispatch live extension tools through the registry's helper module."""
        from ouroboros.tools.extension_dispatch import dispatch_extension_tool

        return dispatch_extension_tool(self._ctx, name, ext_tool, args)

    def _dispatch_mcp_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Run a provider-safe MCP tool after the normal safety supervisor."""
        from ouroboros.safety import check_safety as _mcp_check_safety
        is_safe, safety_msg = _mcp_check_safety(
            name,
            args,
            messages=getattr(self._ctx, "messages", None),
            ctx=self._ctx,
        )
        if not is_safe:
            return safety_msg
        try:
            from ouroboros.mcp_client import call_mcp_tool as _mcp_call
            result = _mcp_call(name, args or {})
        except Exception as exc:
            return f"⚠️ TOOL_ERROR ({name}): {exc}"
        # Warning trails the payload (#447 H1): a leading SAFETY_WARNING masked
        # a structured {"ok": false} MCP answer from failure classification.
        return _append_result_note(result, safety_msg) if safety_msg else result

    def _protected_shell_block(
        self, raw_cmd, cmd_path_lower, binding, acting_self_worktree, writeish,
    ) -> Optional[str]:
        """Apply payload/core write guards to the selected physical target."""
        items = _binding_items(binding)
        targets_skill = bool(items) and all(item.root == "skill_payload" for item in items)
        targets_system = (
            _binding_set_targets_system_repo(self._ctx, binding)
            or acting_self_worktree
        )
        if (targets_skill or targets_system) and any(
            name in cmd_path_lower
            for name in (
                *SKILL_PAYLOAD_CONTROL_FILENAMES,
                *(SKILL_PAYLOAD_CONTROL_DIRNAMES - {"__pycache__"}),
            )
        ) and writeish:
            return (
                "⚠️ SAFETY_VIOLATION: Shell command would modify a skill "
                "provenance / launcher seed / dependency marker (.clawhub.json, "
                ".ouroboroshub.json, .self_authored.json, SKILL.openclaw.md, .seed-origin, "
                ".ouroboros_env, node_modules). "
                "Use marketplace lifecycle flows or edit user-authored "
                "payload files instead."
            )
        if _authorized_managed_update_resolver(self._ctx):
            return None
        if targets_system and shell_writer_targets_protected(raw_cmd):
            return (
                "⚠️ CRITICAL SAFETY_VIOLATION: Shell command would modify "
                "a protected core/contract/release file. Protected: "
                + ", ".join(sorted(PROTECTED_RUNTIME_PATHS))
            )
        if targets_system:
            for cf in PROTECTED_RUNTIME_PATHS_LOWER:
                # The MODE-AWARE composition fact, not the coarse legacy scan: a
                # pure read that merely mentions a protected name (`grep -n delete
                # ouroboros/safety.py`, `sed -n 1,40p BIBLE.md`, a python open('r'))
                # is not a modification; every write shape still blocks here.
                if cf in cmd_path_lower and writeish:
                    return (
                        "⚠️ CRITICAL SAFETY_VIOLATION: Shell command would modify "
                        "a protected core/contract/release file. Protected: "
                        + ", ".join(sorted(PROTECTED_RUNTIME_PATHS))
                    )
        return None

    def _git_protected_roots(self) -> list:
        """Ouroboros runtime roots the target-aware git resolver protects, by
        enumeration: the system repo + EVERY data drive the task touches (parent
        drive plus any child / budget drive in task_metadata). Missing a child
        drive here would let git escape into the control plane. ONE enumeration
        for the external-workspace lane and the default (non-workspace) lane."""
        git_protected_roots = [
            pathlib.Path(getattr(self._ctx, "system_repo_dir", None) or self._ctx.repo_dir),
            pathlib.Path(self._ctx.repo_dir),
            pathlib.Path(self._ctx.drive_root),
        ]
        _meta = getattr(self._ctx, "task_metadata", {})
        if isinstance(_meta, dict):
            for _k in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
                if _meta.get(_k):
                    git_protected_roots.append(pathlib.Path(str(_meta.get(_k))))
        return git_protected_roots

    def _resolved_shell_cwd(self, args: Dict[str, Any], binding: Any = None) -> Any:
        """The command's working directory, resolved ONCE through the cwd SSOT.

        Returns a ``pathlib.Path``, or the typed cwd-block MESSAGE (a ``str``) when
        resolution fails. Every guard downstream takes this canonical path instead
        of re-resolving — or, worse, string-joining the raw cwd label onto a root,
        which is the D1 regression class (v6.74.0)."""
        items = _binding_items(binding)
        if items:
            return pathlib.Path(items[0].target_path)
        raw_cwd = str(args.get("cwd") or "")
        operation = "service" if str(args.get("__tool_name") or "") == "start_service" else "shell"
        try:
            work_dir, _cwd_root, _allowed = resolve_shell_cwd(self._ctx, raw_cwd, operation=operation)
        except Exception as exc:
            return shell_cwd_block_message(self._ctx, raw_cwd, operation=operation, error=exc)
        return pathlib.Path(work_dir)

    def _external_workspace_git_block(self, raw_cmd: Any, work_dir: pathlib.Path) -> Optional[str]:
        from ouroboros.git_shell_policy import external_workspace_git_violation

        # External-workspace git is no longer confined to the active workspace
        # (host scratch is legitimate); only the enumerated runtime roots are
        # protected. ``work_dir`` is the ALREADY-RESOLVED cwd from the one
        # resolve_shell_cwd call in _shell_git_and_runtime_block — passing it as
        # the base with cwd="" keeps the D1 rule (resolve once, through the SSOT,
        # never re-join a raw cwd label onto a root).
        git_violation = external_workspace_git_violation(
            raw_cmd,
            active_root=work_dir,
            cwd="",
            protected_roots=self._git_protected_roots(),
            allow_network=_resource_allowed(self._ctx, "network"),
        )
        if not git_violation:
            return None
        if git_violation.startswith("task_contract.allowed_resources"):
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}."
        return f"⚠️ WORKSPACE_GIT_BLOCKED: {git_violation}."

    def _external_runtime_protected_paths(
        self, binding: Any = None,
    ) -> tuple[list, list, list, list]:
        """Ouroboros runtime roots that an EXTERNAL-workspace task must not touch via
        shell (system repo + EVERY data drive incl child/budget + owner credential
        locations) plus the task's own exempt task_drive/artifact_store roots. Returns
        (protected_texts, allowed_texts, protected_paths, allowed_paths): the *_texts
        feed the embedded-string boundary check; the *_paths feed token resolution
        (relative->cwd, ~->home, symlink canonicalization) so relative/symlink bypasses
        are closed. SSOT for the read + write guards."""
        meta = getattr(self._ctx, "task_metadata", {}) if isinstance(getattr(self._ctx, "task_metadata", {}), dict) else {}
        protected_values = [getattr(self._ctx, "system_repo_dir", None) or getattr(self._ctx, "repo_dir", None),
                            getattr(self._ctx, "drive_root", None)]
        try:
            from ouroboros.config import DATA_DIR as _PARENT_DATA_DIR
            protected_values.append(_PARENT_DATA_DIR)
        except Exception:
            pass
        for _dk in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
            if meta.get(_dk):
                protected_values.append(meta.get(_dk))
        # Owner/runtime credential locations, as ABSOLUTE paths. Blocking by
        # absolute containment (not a substring marker) means the OWNER's personal
        # secrets (~/.ssh/id_rsa, ~/.aws, ~/file1.txt) are off-limits while a
        # project-relative file merely NAMED like a credential (site/.ssh/config, a
        # project .env) stays the task's own — and a non-path token like
        # "os.environ" can never spuriously match.
        try:
            _home = pathlib.Path.home()
            for _rel in (".ssh", ".aws", ".gnupg", ".netrc", ".pgpass", ".config/gcloud",
                         ".docker/config.json", ".kube/config", ".npmrc", "file1.txt"):
                protected_values.append(_home / _rel)
        except Exception:
            pass
        def _text_forms(value: Any) -> list:
            # Both the as-given and the symlink-resolved form, so a command using
            # /var/... matches a root resolved to /private/var/... (macOS) and vice
            # versa. In production ($HOME paths) the two coincide.
            out = []
            for variant in (value, None):
                try:
                    p = pathlib.Path(value)
                    if variant is None:
                        p = p.resolve(strict=False)
                    t = str(p).replace("\\", "/").lower().rstrip("/")
                    if t and t not in out:
                        out.append(t)
                except Exception:
                    continue
            return out

        def _resolved(value: Any):
            try:
                return pathlib.Path(value).resolve(strict=False)
            except Exception:
                return None

        protected_texts: list = []
        protected_paths: list = []
        for v in protected_values:
            if not v:
                continue
            for t in _text_forms(v):
                if t not in protected_texts:
                    protected_texts.append(t)
            rp = _resolved(v)
            if rp is not None and rp not in protected_paths:
                protected_paths.append(rp)
        allowed_texts: list = []
        allowed_paths: list = []
        task_id = task_id_for_artifacts(self._ctx)
        for data_root in (getattr(self._ctx, "drive_root", None), meta.get("drive_root"), meta.get("budget_drive_root")):
            if not data_root:
                continue
            for rp_src in (pathlib.Path(data_root) / "task_drives" / task_id, task_artifact_dir_path(pathlib.Path(data_root), task_id, create=False)):
                for t in _text_forms(rp_src):
                    if t not in allowed_texts:
                        allowed_texts.append(t)
                rp = _resolved(rp_src)
                if rp is not None and rp not in allowed_paths:
                    allowed_paths.append(rp)
        # An explicitly selected system repo or exact skill payload is an
        # authorized process target. Keep every other runtime/credential root
        # protected, but do not re-block that exact binding merely because the
        # task also has an external workspace focus.
        for item in _binding_items(binding):
            if item.root not in {"system_repo", "skill_payload"}:
                continue
            selected = pathlib.Path(item.base_path)
            for t in _text_forms(selected):
                if t not in allowed_texts:
                    allowed_texts.append(t)
            rp = _resolved(selected)
            if rp is not None and rp not in allowed_paths:
                allowed_paths.append(rp)
        return protected_texts, allowed_texts, protected_paths, allowed_paths

    def _external_shell_runtime_or_secret_block(
        self, raw_cmd: Any, cmd_path_lower: str, args: Dict[str, Any],
        work_dir: Optional[pathlib.Path] = None,
        binding: Any = None,
    ) -> Optional[str]:
        """External-workspace shell guard for READ and write commands alike: block any
        command that targets the Ouroboros runtime (system repo / any data drive) or an
        owner credential path. read_file/user_files already enforce this; raw shell
        (cat, python -c open(...), etc.) would otherwise bypass it. Two layers, because
        string matching alone is bypassable by relative paths and symlinks:
          (1) embedded-string boundary match of ABSOLUTE protected roots (catches a path
              literal inside e.g. python -c "open('/abs/data/settings.json')");
          (2) path-token RESOLUTION — every path-like arg is expanduser'd, joined to the
              command cwd when relative, and resolve()'d (canonicalizing symlinks + ..),
              then containment-checked. This closes a relative path passed as its own
              argv token (`cat ../../data/settings.json`) and a workspace-internal symlink
              to the data drive (round-2 review).
        Both layers are best-effort DEFENSE-IN-DEPTH, not the primary control: a relative
        path hidden INSIDE an interpreter one-liner string (e.g. node -e
        "readFileSync('../../data/settings.json')") is not a standalone token, so it is
        not extracted here — and that residual is deliberately NOT chased with a regex
        over code strings (an unwinnable arms race; BIBLE P5 / no-string-gate doctrine).
        The PRIMARY control is the gated read_file/user_files path, which fully resolves
        and containment-checks every read against the protected drives, plus the LLM
        safety supervisor judging intent on each shell call."""
        _BLOCK = (
            "⚠️ WORKSPACE_SHELL_BLOCKED: shell command targets the Ouroboros runtime "
            "(system repo / data drive) or an owner credential path. External-workspace "
            "tasks may not read or write those; use the gated read_file tool for any "
            "inspection you need. Run your command against the task's own surfaces "
            "instead: the active workspace root (e.g. /app) or scratch such as /tmp."
        )
        protected_texts, allowed_texts, protected_paths, allowed_paths = (
            self._external_runtime_protected_paths(binding)
        )
        # (1) embedded-string boundary match (absolute roots only — no substring secret
        # markers, which would false-block the task's own project files / "os.environ").
        for pt in protected_texts:
            if _command_mentions_protected_root(cmd_path_lower, pt) and not any(
                _command_mentions_protected_root(cmd_path_lower, t) for t in allowed_texts
            ):
                return _BLOCK
        # (2) path-token resolution (relative -> cwd, ~ -> home, symlinks canonicalized).
        # The cwd is resolved ONCE per safety check by the caller (D1); resolve here
        # only when this guard is used standalone.
        if work_dir is None:
            resolved_cwd = self._resolved_shell_cwd(args, binding)
            if isinstance(resolved_cwd, str):
                return resolved_cwd
            work_dir = pathlib.Path(resolved_cwd)
        work_dir = pathlib.Path(work_dir)

        def _within(child: pathlib.Path, parent: pathlib.Path) -> bool:
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        for tok in shell_argv_with_path_tokens(raw_cmd):
            tok_text = str(tok or "").strip()
            if not tok_text or tok_text.startswith("-") or tok_text in {"|", "&&", "||", ";", ">", ">>", "<", "<<", "&"}:
                continue
            try:
                p = pathlib.Path(tok_text).expanduser()
                resolved = p.resolve(strict=False) if p.is_absolute() else (work_dir / p).resolve(strict=False)
            except Exception:
                continue
            if any(_within(resolved, ap) for ap in allowed_paths):
                continue
            if any(_within(resolved, pp) for pp in protected_paths):
                return _BLOCK
        return None

    def _workspace_shell_write_block(
        self,
        args: Dict[str, Any],
        raw_cmd: Any,
        cmd_path_lower: str,
        explicit_write_targets: list[str],
        write_target_argvs: list[list[str]],
        executable_path_tokens: set[str],
        runtime_mode: str,
        acting_subagent: bool,
        binding: Any,
    ) -> Optional[str]:
        """Keep workspace writes inside the selected target plus task custody roots."""

        items = _binding_items(binding)
        if not items:
            return "⚠️ WORKSPACE_SHELL_BLOCKED: process target was not resolved."
        selected = items[0]
        work_dir = pathlib.Path(selected.target_path).resolve(strict=False)
        selected_base = pathlib.Path(selected.base_path).resolve(strict=False)
        allowed_relative_roots = list(dict.fromkeys((selected_base, work_dir)))
        allowed_data_roots: list[pathlib.Path] = []
        meta = (
            getattr(self._ctx, "task_metadata", {})
            if isinstance(getattr(self._ctx, "task_metadata", {}), dict)
            else {}
        )
        for data_root in (getattr(self._ctx, "drive_root", None), meta.get("budget_drive_root")):
            if not data_root:
                continue
            task_id = task_id_for_artifacts(self._ctx)
            for root_path in (
                pathlib.Path(data_root) / "task_drives" / task_id,
                task_artifact_dir_path(pathlib.Path(data_root), task_id, create=False),
            ):
                resolved_root = pathlib.Path(root_path).resolve(strict=False)
                if resolved_root not in allowed_data_roots:
                    allowed_data_roots.append(resolved_root)
        if selected.root in {"task_drive", "artifact_store"}:
            allowed_data_roots.append(selected_base)
        # Executor-backed commands use backend path spellings (for example
        # ``/deliverables/report.html``), while the policy roots above are host
        # paths. Keep the configured Deliverables root separate from the generic
        # allow-root list: every descendant must still pass the target-specific
        # user-files policy (hidden/credential/symlink checks) below.
        deliverables_root_lexical: pathlib.Path | None = None
        deliverables_root_lexical_alias: pathlib.Path | None = None
        deliverables_root_physical: pathlib.Path | None = None
        try:
            candidate_deliverables = resource_root_path(self._ctx, "deliverables")
            # Retain the configured spelling even when the root itself is
            # malformed or protected. Descendants must then take the
            # target-specific path and fail closed before a broader
            # workspace/data allow-root can accidentally admit them.
            deliverables_root_physical = pathlib.Path(candidate_deliverables).resolve(strict=False)
            deliverables_root_lexical = _deliverables_root_lexical()
            deliverables_root_lexical_alias = _deliverables_root_lexical_alias()
        except (OSError, TypeError, ValueError, RuntimeError):
            pass

        def _deliverables_target_decision(path: pathlib.Path) -> bool | None:
            """Decide Deliverables descendants before generic root admission.

            Deliverables can be configured inside the selected workspace.  A
            generic workspace-root fast path must not skip the user-files
            hidden/credential/symlink checks for such a target. ``None`` means
            that the candidate is outside Deliverables and may use the normal
            workspace-root checks.
            """
            if deliverables_root_lexical is None:
                return None
            try:
                lexical_path = pathlib.Path(path).expanduser()
                if not lexical_path.is_absolute():
                    lexical_path = pathlib.Path(os.path.abspath(lexical_path))
                in_deliverables = (
                    deliverables_root_lexical is not None
                    and (
                        lexical_path.is_relative_to(deliverables_root_lexical)
                        or _lexical_path_is_relative_to_casefold(lexical_path, deliverables_root_lexical)
                        or _lexical_path_is_relative_to_casefold(
                            lexical_path, deliverables_root_lexical_alias,
                        )
                        or _lexical_path_is_relative_to_casefold(
                            lexical_path, deliverables_root_physical,
                        )
                    )
                )
            except (OSError, TypeError, ValueError):
                return False
            resolved_path = pathlib.Path(path).resolve(strict=False)
            physically_in_deliverables = (
                deliverables_root_physical is not None
                and (
                    resolved_path.is_relative_to(deliverables_root_physical)
                    or _path_is_relative_to_casefold(
                        resolved_path, deliverables_root_physical,
                    )
                )
            )
            if not in_deliverables and not physically_in_deliverables:
                return None
            return self._deliverables_shell_target_allowed(
                resolved_path,
                lexical_candidate=lexical_path,
            )

        if direct_target_block := direct_deliverable_target_block(
            self._ctx,
            work_dir,
            write_target_argvs,
            deliverables_root_physical,
            _deliverables_target_decision,
        ):
            return direct_target_block

        # Acting subagents must write ONLY inside their isolated surface, so pro
        # mode does NOT grant them the outside-workspace absolute-path passthrough.
        pro_workspace_passthrough = (
            str(runtime_mode or "").strip().lower() == "pro" and not acting_subagent
        )
        protected_roots = [
            getattr(self._ctx, "system_repo_dir", None) or getattr(self._ctx, "repo_dir", None),
            getattr(self._ctx, "drive_root", None),
        ]
        try:
            from ouroboros.config import DATA_DIR as parent_data_dir

            protected_roots.append(parent_data_dir)
        except Exception:
            pass
        for key in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
            if meta.get(key):
                protected_roots.append(meta.get(key))
        allowed_texts = [
            str(root).replace("\\", "/").lower().rstrip("/")
            for root in [*allowed_relative_roots, *allowed_data_roots]
        ]
        protected_paths = []
        for root_value in protected_roots:
            try:
                root_path = pathlib.Path(root_value).resolve(strict=False)
            except Exception:
                continue
            protected_paths.append(root_path)
            if any(root_path.is_relative_to(root) for root in allowed_relative_roots):
                continue
            root_text = str(root_path).replace("\\", "/").lower()
            if _command_mentions_protected_root(cmd_path_lower, root_text) and not any(
                _command_mentions_protected_root(cmd_path_lower, text)
                for text in allowed_texts
            ):
                return _workspace_write_block_runtime_message(root_path)
        path_tokens = list(shell_argv_with_path_tokens(raw_cmd))
        path_tokens.extend(
            token
            for token in explicit_write_targets
            if token and token not in path_tokens
        )
        for token in path_tokens:
            token_text = str(token)
            if token_text in executable_path_tokens and token_text not in explicit_write_targets:
                continue
            candidates = [token_text] if is_absolute_path_text(token_text) else []
            if token_text.startswith(("./", "../")):
                candidates.append(token_text)
            elif (
                token_text
                and not token_text.startswith("-")
                and token_text not in {"|", "&&", "||", ";", ">", ">>", "<", "<<"}
                and (
                    token_text in explicit_write_targets
                    or "/" in token_text
                    or "\\" in token_text
                )
            ):
                candidates.append(token_text)
            for candidate in candidates:
                if candidate == "/dev/null":
                    continue
                if is_absolute_path_text(candidate):
                    mapped_executor_lexical = _executor_backend_candidate_path(self._ctx, candidate)
                    if mapped_executor_lexical is not None:
                        mapped_executor = mapped_executor_lexical.resolve(strict=False)
                        deliverables_decision = _deliverables_target_decision(mapped_executor_lexical)
                        if deliverables_decision is not None:
                            if deliverables_decision:
                                continue
                            return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed Deliverables path."
                        if any(mapped_executor.is_relative_to(root) for root in allowed_relative_roots):
                            continue
                        if any(mapped_executor.is_relative_to(root) for root in allowed_data_roots):
                            continue
                        for protected_path in protected_paths:
                            try:
                                mapped_executor.relative_to(protected_path)
                                return _workspace_write_block_runtime_message(mapped_executor)
                            except Exception:
                                pass
                    if _executor_backend_candidate_allowed(
                        self._ctx,
                        candidate,
                        [*allowed_relative_roots, *allowed_data_roots],
                    ):
                        continue
                    windows_drive_path = bool(re.match(r"^[A-Za-z]:[\\/]", candidate))
                    unc_path = candidate.startswith("\\\\")
                    # On the native Windows host, resolve drive paths exactly as
                    # POSIX paths are resolved below. This canonicalizes directory
                    # symlinks/junctions before containment: a workspace alias stays
                    # allowed, while an in-workspace spelling whose nested link exits
                    # the root is blocked. Keep lexical handling for foreign Windows
                    # spellings seen on POSIX and for UNC paths (which may require a
                    # network lookup merely to evaluate the guard).
                    if (not windows_drive_path and not unc_path) or (
                        os.name == "nt" and windows_drive_path
                    ):
                        try:
                            resolved = pathlib.Path(candidate).resolve(strict=False)
                        except Exception:
                            continue
                        # Keep the pre-resolution spelling so a symlink child
                        # cannot resolve into another allowed root and bypass
                        # the Deliverables policy.
                        deliverables_decision = _deliverables_target_decision(pathlib.Path(candidate))
                        if deliverables_decision is not None:
                            if deliverables_decision:
                                continue
                            return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed Deliverables path."
                        if any(resolved.is_relative_to(root) for root in allowed_relative_roots):
                            continue
                        if any(resolved.is_relative_to(root) for root in allowed_data_roots):
                            continue
                        for protected_path in protected_paths:
                            try:
                                resolved.relative_to(protected_path)
                                return _workspace_write_block_runtime_message(resolved)
                            except Exception:
                                pass
                        if not pro_workspace_passthrough:
                            return _workspace_write_block_outside_root_message(resolved, work_dir)
                        continue
                    deliverables_decision = _deliverables_target_decision(pathlib.Path(candidate))
                    if deliverables_decision is not None:
                        if deliverables_decision:
                            continue
                        return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed Deliverables path."
                    if any(path_text_is_inside(candidate, root) for root in allowed_relative_roots):
                        continue
                    if any(path_text_is_inside(candidate, root) for root in allowed_data_roots):
                        continue
                    for protected_path in protected_paths:
                        if path_text_is_inside(candidate, protected_path):
                            return _workspace_write_block_runtime_message(candidate)
                    if not pro_workspace_passthrough:
                        return _workspace_write_block_outside_root_message(candidate, work_dir)
                    continue
                resolved = (work_dir / pathlib.Path(candidate)).resolve(strict=False)
                # The lexical relative spelling is authoritative for detecting
                # a Deliverables-origin target; the helper then canonicalizes
                # it and rejects symlink escapes.
                deliverables_decision = _deliverables_target_decision(
                    work_dir / pathlib.Path(candidate)
                )
                if deliverables_decision is not None:
                    if deliverables_decision:
                        continue
                    return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed Deliverables path."
                if any(resolved.is_relative_to(root) for root in allowed_relative_roots):
                    continue
                if any(resolved.is_relative_to(root) for root in allowed_data_roots):
                    continue
                for protected_path in protected_paths:
                    try:
                        resolved.relative_to(protected_path)
                        return _workspace_write_block_runtime_message(resolved)
                    except Exception:
                        pass
                if not pro_workspace_passthrough:
                    return _workspace_write_block_outside_root_message(resolved, work_dir)
        return None

    def _run_shell_safety_check(
        self, args: Dict[str, Any], runtime_mode: str, binding: Any = None,
    ) -> Optional[str]:
        """Pre-execution run_command filter; returns a block message or ``None``."""
        raw_cmd = args.get("cmd", args.get("command", ""))
        if binding is None:
            operation = (
                "service"
                if str(args.get("__tool_name") or "") == "start_service"
                else "shell"
            )
            try:
                binding = build_resolved_resource_binding(
                    self._ctx,
                    operation=operation,
                    process_cwd=str(args.get("cwd") or ""),
                    bucket=str(args.get("bucket") or ""),
                    skill_name=str(args.get("skill_name") or ""),
                )
            except Exception as exc:
                return shell_cwd_block_message(
                    self._ctx,
                    str(args.get("cwd") or ""),
                    operation=operation,
                    error=exc,
                )
        workspace_mode = bool(getattr(self._ctx, "is_workspace_mode", lambda: False)())
        # self_worktree is a checkout of the system repo, so protected shell-write
        # guards must stay active for it even in workspace mode (acting children
        # must use write_file/edit_text, which apply the pro+grant gate).
        acting_self_worktree = self._acting_self_worktree()
        acting_subagent = self._is_acting_subagent()
        argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
        if sudo_noninteractive_violation(raw_cmd):
            return (
                "⚠️ SUDO_INTERACTIVE_BLOCKED: sudo must be noninteractive. Use sudo -n for commands that can run without a password; if sudo -n fails, report validation/install blocked by environment."
            )
        cmd_lower = (" ".join(str(x) for x in raw_cmd) if isinstance(raw_cmd, list) else str(raw_cmd)).lower()
        cmd_path_lower = cmd_lower.replace("\\", "/")
        while "//" in cmd_path_lower: cmd_path_lower = cmd_path_lower.replace("//", "/")
        # Subagents must not read owner secrets/credentials/control state via shell
        # (read_file already denies these). read_file is the gated inspection path.
        if (acting_subagent or self._is_local_readonly_subagent()) and _subagent_shell_targets_secret(cmd_path_lower):
            return (
                "⚠️ SUBAGENT_SECRET_READ_BLOCKED: subagents may not read Ouroboros secrets, "
                "credentials, or owner-control state via shell. Use the gated read_file tool "
                "(which denies secrets) for any inspection you actually need."
            )
        argv_for_write = argv
        argv_executable = pathlib.PurePath(argv_for_write[0]).name.lower().removesuffix(".exe") if argv_for_write else ""
        write_target_argvs = [argv_for_write] if argv_for_write else []
        inline_argv: list = []
        if argv_executable in {"sh", "bash", "zsh"}:
            inline_cmd = next((str(argv_for_write[idx + 1] or "") for idx, token in enumerate(argv_for_write[1:], start=1) if str(token or "") in {"-c", "--command"} and idx + 1 < len(argv_for_write)), "")
            if not inline_cmd:
                inline_cmd = shell_command_string(argv_for_write)
            inline_argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(inline_cmd)))
            if inline_argv:
                write_target_argvs.append(inline_argv)
        explicit_write_targets = list(dict.fromkeys(str(token) for target_argv in write_target_argvs for token in writer_target_tokens(target_argv) if str(token or "").strip()))
        # ``cp source Deliverables/`` (and the equivalent mv/ln form) writes a
        # child named after the source, while the ordinary writer-target parser
        # only sees the directory operand. Add those argv-visible child names to
        # the same target-first policy without attempting to parse inline code,
        # archive formats, or other deferred Q3 syntax.
        for target_argv in write_target_argvs:
            for command, destination, source in directory_destination_pairs(target_argv):
                source_name = directory_destination_child_name(command, target_argv, source)
                if source_name in {"", ".", ".."}:
                    continue
                explicit_write_targets.append(
                    destination.rstrip("/\\") + "/" + source_name
                )
        # A located -e/-E/-c inline CODE BODY is not a write target: the
        # generic fallback reported every non-flag operand of a writer command
        # (ruby/perl) - code string included - making every one-liner
        # write-shaped. The light fence and protected lane keep the unfiltered
        # SSOT (pinned XG-7B3.1); only THIS lane drops the bodies. FILE
        # operands stay write-suspect (`perl -pi -e s/a/b/ file` rewrites
        # `file`); literal in-code targets still arrive via inline extraction.
        from ouroboros.tools.shell_guards import interpreter_inline_code as _interp_inline_code
        inline_code_bodies: set = set()
        for target_argv in write_target_argvs:
            inline_code_bodies.update(_interp_inline_code([str(t) for t in target_argv]))
        if inline_code_bodies:
            explicit_write_targets = [t for t in explicit_write_targets if t not in inline_code_bodies]
        explicit_write_targets = list(dict.fromkeys(explicit_write_targets))
        executable_path_tokens = {str(target_argv[0]) for target_argv in write_target_argvs if target_argv}
        # Writer-command membership canonicalizes versioned interpreter spellings to
        # their family (`ruby3.2` is `ruby`), so a versioned basename is exactly as
        # write-suspect as the unversioned one (XG-2R.2).
        # Interpreter argv (direct or inside sh -c) takes the MODE-AWARE
        # write-shape classifier: the bare `open(` token classified read-only
        # `open(p, 'rb')` as a write ("the original GAIA class"). Write-mode
        # opens, pathlib `.open('w')`, save-APIs, opaque subprocess escapes
        # and shell-level indicators still classify as writes;
        # `writer_target_tokens` keeps covering literal targets below.
        write_shape_interpreter = bool(interpreter_family(argv_executable)) or (
            bool(inline_argv)
            and bool(interpreter_family(pathlib.PurePath(str(inline_argv[0])).name.lower().removesuffix(".exe")))
        )
        # ONE mode-aware write-shape seam (write_shape.py) for BOTH halves:
        # interpreter argv takes interpreter_write_shape; everything else takes
        # non_interpreter_write_shape, where unconditional writers keep the
        # membership floor, pure-filter utilities (sort/uniq/sed/tar/gzip) need a
        # real write channel, and prose words yield to the same read-carve the
        # owner-control detectors use. No guard below consumes a coarser fact.
        coarse_write_shape = (
            interpreter_write_shape(raw_cmd)
            if write_shape_interpreter
            else non_interpreter_write_shape(
                raw_cmd, argv_for_write, argv_executable, is_pure_read=_is_pure_read_inspection,
            )
        )
        writeish = coarse_write_shape or bool(explicit_write_targets)
        work_dir = self._resolved_shell_cwd(args, binding)
        if isinstance(work_dir, str):
            return work_dir
        if protected_artifact_block := protected_artifact_shell_block_reason(
            self._ctx,
            raw_cmd,
            cwd=str(work_dir),
            default_cwd=pathlib.Path(work_dir),
            binding=_binding_items(binding)[0] if _binding_items(binding) else None,
        ):
            return protected_artifact_block
        if writeish and (executor_state_block := workspace_executor_state_write_block(
            raw_cmd,
            drive_root=pathlib.Path(self._ctx.drive_root),
            cwd=str(work_dir),
            default_cwd=pathlib.Path(work_dir),
        )):
            return executor_state_block
        if workspace_mode and writeish:
            workspace_write_block = self._workspace_shell_write_block(
                args,
                raw_cmd,
                cmd_path_lower,
                explicit_write_targets,
                write_target_argvs,
                executable_path_tokens,
                runtime_mode,
                acting_subagent,
                binding,
            )
            if workspace_write_block:
                return workspace_write_block

        # Elevation pattern: blocked in all modes. Every owner-control mention
        # detector takes the shared read-carve (pure read-only inspection of the
        # key/endpoint names is allowed; the write shape or any non-inspection
        # head still blocks) — the scope-floor precedent applied family-wide.
        if _detect_runtime_mode_elevation(cmd_lower, writeish=writeish):
            return "⚠️ ELEVATION_BLOCKED: shell command pattern looks like an OUROBOROS_RUNTIME_MODE elevation attempt (mentions ``save_settings`` together with ``OUROBOROS_RUNTIME_MODE``, or invokes ``ouroboros.config.save_settings`` directly). Runtime mode is owner-controlled — change it by stopping the agent and editing settings.json directly, then restart."
        if _detect_context_mode_self_lowering(cmd_lower, writeish=writeish):
            return "⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to lower OUROBOROS_CONTEXT_MODE to low through settings.json or /api/owner/context-mode. Context mode is owner-controlled — ask the owner to change the Low/Max toggle or edit settings while the agent is stopped."
        if _detect_scope_review_floor_self_lowering(cmd_lower, writeish=writeish):
            return "⚠️ SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED: shell command pattern reaches OUROBOROS_SCOPE_REVIEW_FLOOR through settings.json, /api/settings, or /api/owner/scope-review-floor from something other than a pure read. The floor is a deprecated, enforcement-inert owner setting (BIBLE P3 scope-review applicability follows the owner context mode) — it stays owner-only, and the agent must not write owner settings through any channel. Ask the owner to change it via the dedicated /api/owner/scope-review-floor endpoint, or stop the agent and edit settings.json directly. Pure source inspection (grep/rg/cat/jq/git grep) is allowed; an interpreter or HTTP client naming the endpoint is not, whatever verb it spells."
        if _detect_safety_mode_self_lowering(cmd_lower, writeish=writeish):
            return "⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to change OUROBOROS_SAFETY_MODE (e.g. to ``light``/``off``) through settings.json, /api/settings, or /api/owner/safety-mode. LLM-safety coverage is owner-controlled (BIBLE P3) — the agent must not reduce its own supervision. Ask the owner to change it via the dedicated /api/owner/safety-mode endpoint, or stop the agent and edit settings.json directly."
        if _detect_owner_skill_attest_self_call(cmd_lower, writeish=writeish):
            return "⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: shell command pattern looks like an attempt to loopback-POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips the expensive LLM skill review and is OWNER-ONLY — the agent must not self-attest its own skill to bypass the immune system's review. Ask the owner to attest it from the Skills UI."
        if _detect_mutative_toggle_self_change(cmd_lower, writeish=writeish):
            return "⚠️ ELEVATION_BLOCKED: OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is owner-controlled (it grants subagents write power against the live body). Change it by stopping the agent and editing settings.json directly, then restart — the agent must not self-enable mutative subagents."
        if _detect_evolution_owner_control_self_change(cmd_lower, writeish=writeish):
            return "⚠️ ELEVATION_BLOCKED: the self-evolution controls (OUROBOROS_POST_TASK_EVOLUTION and OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE) are owner-controlled — they enable or steer self-modification cycles. Change them via the owner Settings UI, or stop the agent and edit settings.json directly — the agent must not self-set evolution controls."
        if _mentions_skill_owner_state(cmd_lower, writeish=writeish):
            return (
                "⚠️ SKILL_STATE_WRITE_BLOCKED: skill review, enablement, "
                "grants, and marketplace provenance are owner/review "
                "controlled state. Use skill_review, toggle_skill/the Skills "
                "UI, or the desktop launcher confirmation flow. Pure read-only "
                "inspection (grep/rg/cat/jq) of these names is allowed."
            )
        if "state" in cmd_lower and "skills" in cmd_lower and _mentions_detached_process(cmd_lower):
            return (
                "⚠️ SKILL_STATE_WRITE_BLOCKED: detached shell processes must "
                "not target skill state directories. Use the reviewed skill "
                "lifecycle tools instead."
            )

        # Light-mode checks follow the selected physical target, not whether a
        # project workspace happens to be attached.
        if runtime_mode == "light":
            if light_shell_repo_mutation(
                raw_cmd,
                repo_dir=system_repo_dir_for(self._ctx),
                cwd=str(args.get("cwd") or ""),
                work_dir=pathlib.Path(work_dir),
                # Inline-code inspection now reaches EVERY surface this check guards
                # (it defaults ON in the fence) — scoping it to `__tool_name ==
                # "run_script"` let run_command mutate the repo first (XG-7B3.1).
            ):
                return (
                    "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses "
                    "shell commands that mutate the Ouroboros repository. "
                    "For external deliverables, run with cwd under user_files "
                    "(for example /Users/<you>/Desktop), root=artifact_store, "
                    "or root=task_drive. Switch to advanced/pro only for "
                    "reviewed Ouroboros self-modification."
                )
            runtime_data_executable = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe") if argv else ""
            # Versioned interpreter basenames (python3.11, ruby3.2, php8.3,
            # perl5.38, node18) must trigger the runtime_data scan exactly like
            # their unversioned spellings. Classification is the shared structural
            # `interpreter_family` — the exact-set + `startswith("python")` pair
            # recognized versions of ONE family and let every other family's
            # versioned spelling bypass the guard (XG-2R.2).
            runtime_data_scan = (
                writeish
                or runtime_data_executable in {"sh", "bash", "zsh"}
                or bool(interpreter_family(runtime_data_executable))
            )
            if runtime_data_scan:
                own_task_drive = pathlib.Path(self._ctx.task_drive_root())
                own_artifact_dir = task_artifact_dir_path(
                    pathlib.Path(self._ctx.drive_root),
                    task_id_for_artifacts(self._ctx),
                    create=False,
                )
                allowed_runtime_roots = [own_task_drive, own_artifact_dir]
                for item in _binding_items(binding):
                    if item.root == "skill_payload" and item.source != "native":
                        allowed_runtime_roots.append(pathlib.Path(item.base_path))
                runtime_data_targets = runtime_data_guard_targets(
                    raw_cmd,
                    writeish=writeish,
                    drive_root=pathlib.Path(self._ctx.drive_root),
                    work_dir=pathlib.Path(work_dir),
                    allowed_roots=allowed_runtime_roots,
                )
                if runtime_data_targets:
                    action = "write under" if writeish else "write-indicating commands that mention"
                    # Name the REAL task roots: a mis-guessed absolute path used to
                    # produce this block with no way to self-correct (v6.54.3).
                    return (
                        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks process commands "
                        f"that {action} runtime_data paths outside this task's own roots. "
                        f"This task's real roots are: artifact_store={own_artifact_dir}, "
                        f"task_drive={own_task_drive} — staged attachments live under "
                        f"{own_artifact_dir / 'attachments'}. Use those absolute paths in scripts, "
                        "or root=artifact_store / root=task_drive / root=user_files in file tools. "
                        "Blocked paths: " + ", ".join(runtime_data_targets[:5])
                    )

        if protected_shell := self._protected_shell_block(
            raw_cmd, cmd_path_lower, binding, acting_self_worktree, writeish,
        ):
            return protected_shell

        # GitHub repo create/delete/auth — argv-positional, never substring (#447 A7).
        from ouroboros.git_shell_policy import gh_shell_block_reason

        if gh_block := gh_shell_block_reason(raw_cmd):
            return gh_block

        return self._shell_git_and_runtime_block(
            raw_cmd, args, cmd_path_lower, workspace_mode,
            acting_self_worktree, binding,
        )

    def _shell_git_and_runtime_block(
        self, raw_cmd: Any, args: Dict[str, Any], cmd_path_lower: str,
        workspace_mode: bool, acting_self_worktree: bool, binding: Any,
    ) -> Optional[str]:
        """Direct-git-via-shell policy + the external-workspace runtime/secret read
        guard. External workspaces AND the default (non-workspace) lane get full
        task-local git through ONE target-aware resolver — only the Ouroboros
        runtime is protected (Q4=A unwind, 2026-08-08) — while raw non-git shell
        in external workspaces still cannot read the runtime/secrets;
        self_worktree keeps the strict read-only git policy."""
        from ouroboros.git_shell_policy import is_readonly_git_command

        if not shell_argv(raw_cmd):
            return None
        if workspace_mode and not acting_self_worktree:
            work_dir = self._resolved_shell_cwd(args, binding)
            if isinstance(work_dir, str):  # a cwd block message, not a path
                return work_dir
            if git_block := self._external_workspace_git_block(raw_cmd, work_dir):
                return git_block
            # Even READ-only, non-git shell (cat/head/grep/python) must not
            # reach the runtime or secrets — the raw-shell bypass of the
            # user_files path guard stays closed (top-level external tasks).
            # READ-ONLY GIT IS EXEMPT (owner Q4=A "read-only everywhere"; the
            # f14baf8f false-block class): `git -C <system repo>
            # status|log|diff|show|rev-parse` is the vcs_status-equivalent
            # lane, already readable via gated read_file. The secret surface
            # stays closed: the exemption is ALL-or-nothing per segment
            # (`git status && cat <data>/settings.json` is not exempt) and
            # write-aware — `is_readonly_git_command` refuses `--output=` and
            # `--no-index`, so neither a runtime write nor a settings dump
            # can ride "read-only git".
            if is_external_workspace(self._ctx) and not is_readonly_git_command(raw_cmd):
                if ext_block := self._external_shell_runtime_or_secret_block(
                    raw_cmd, cmd_path_lower, args, work_dir=work_dir,
                    binding=binding,
                ):
                    return ext_block
            return None
        if workspace_mode:
            # Acting self_worktree: a checkout of the Ouroboros repo itself; the
            # acting-child contract (no commits anywhere — a moved HEAD fails patch
            # capture closed; patch integration) keeps the strict read-only git
            # policy, UNWEAKENED by the target-aware default lane below: both the
            # workspace-escape check and the blanket mutating-git text classifier
            # keep running for this lane.
            work_dir = self._resolved_shell_cwd(args, binding)
            if isinstance(work_dir, str):
                return work_dir
            binding_item = _binding_items(binding)[0]
            active_root = pathlib.Path(binding_item.base_path)
            try:
                binding_cwd = pathlib.Path(work_dir).relative_to(active_root).as_posix()
            except ValueError:
                binding_cwd = ""
            git_violation = workspace_git_safety_violation(
                raw_cmd,
                active_root=active_root,
                cwd=binding_cwd,
                allow_network=_resource_allowed(self._ctx, "network"),
            )
            if git_violation:
                if git_violation.startswith("task_contract.allowed_resources"):
                    return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}."
                return (
                    "⚠️ WORKSPACE_GIT_BLOCKED: run_command may only use read-only git "
                    f"operations inside the active workspace; blocked {git_violation}."
                )
            git_violation = run_shell_git_block_reason(
                raw_cmd,
                allow_network=_resource_allowed(self._ctx, "network"),
            )
            if git_violation:
                if git_violation.startswith("task_contract.allowed_resources"):
                    return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}."
                subcmd = git_violation.removeprefix("git ").strip() or git_violation
                return (
                    f"⚠️ GIT_VIA_SHELL_BLOCKED: `git {subcmd}` is blocked for acting "
                    "self_worktree children (no commits; the parent integrates the "
                    "returned patch and is the sole committer). For read-only git: "
                    "vcs_status, vcs_diff tools, or run_command with git "
                    "log/show/diff/status/rev-list/show-ref/for-each-ref/listing branch-tag forms."
                )
            return None
        # DEFAULT (non-workspace) lane. Q4=A (owner 2026-08-08): mutating git
        # is free EVERYWHERE outside the Ouroboros runtime. The argv-text
        # blanket is replaced by the SAME target-aware resolver the external
        # lane runs since v6.27: read-only git allowed even at a runtime
        # target; mutating git blocked only when it TARGETS the runtime
        # (bidirectional/casefold/symlink-resolved); the contract network
        # fence rides along. The cwd resolves EXACTLY ONCE via the shared
        # resolver, passed canonical — never re-join a raw label onto a root
        # (the v6.74.0 D1 class). Disclosed residual (no shell-parser arms
        # race): git via wrapper (nice/xargs) or interpreter code is not
        # classified here; the LLM safety layer still reviews intent, and the
        # light-mode post-exec dirtiness tripwire backstops.
        if "git" not in cmd_path_lower:
            return None
        work_dir = self._resolved_shell_cwd(args, binding)
        if isinstance(work_dir, str):  # a cwd block message, not a path
            return work_dir
        from ouroboros.git_shell_policy import external_workspace_git_violation

        git_violation = external_workspace_git_violation(
            raw_cmd,
            active_root=work_dir,
            cwd="",
            protected_roots=self._git_protected_roots(),
            allow_network=_resource_allowed(self._ctx, "network"),
        )
        if not git_violation:
            return None
        if git_violation.startswith("task_contract.allowed_resources"):
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}."
        return (
            f"⚠️ GIT_VIA_SHELL_BLOCKED: {git_violation}. Mutating git may not target "
            "the Ouroboros runtime (system repo / data drives): self-repo changes go "
            "through commit_reviewed, which enforces pre-commit checks and review. "
            "Read-only git (status/log/diff/show/rev-parse/branch- and tag-listing, "
            "or the vcs_status/vcs_diff tools) works everywhere, and mutating git is "
            "free in any tree OUTSIDE the runtime (e.g. ~/projects, /tmp, an attached "
            "project folder)."
        )

    @staticmethod
    def _owner_settings_snapshot() -> Optional[str]:
        """Text of the live settings.json, "" if absent, None if UNREADABLE.

        None disarms the tripwire below: the deleted restore recorded an OSError
        as "file absent" and could unlink the live settings.json — a baseline is
        either read successfully or not used at all."""
        from ouroboros import config as _cfg

        path = pathlib.Path(_cfg.SETTINGS_PATH)
        try:
            return path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError:
            return None

    def _run_shell_post_checks(
        self,
        result: str,
        *,
        light_repo_before: Optional[Dict[str, Any]],
        workspace_refs_before: Optional[Dict[str, str]],
        settings_before: Optional[str] = None,
        tool_name: str = "run_command",
    ) -> str:
        # The owner-state snapshot/restore (OWNER_STATE_RESTORED) that used to
        # run here was DELETED (issue #447, owner decision): it reverted ANY
        # post-command difference without proving the command caused it, so a
        # concurrent owner edit (Settings UI, grant click) was silently rolled
        # back and blamed on the command; and its snapshot recorded an OSError
        # while reading as "file absent", so the restore could UNLINK the live
        # settings.json after a transient read error. The light-lane guard
        # below refuses auto-rollback for exactly this reason. Pre-exec guards
        # keep skill owner state (SKILL_STATE_WRITE_BLOCKED on any writeish /
        # non-inspection mention — pure read inspection is carved, #447 A2);
        # the settings.json mention-gates are lexical, so an obfuscated
        # argument-level writer (`S=settings; cat > data/$S.json`) can pass
        # them — the tripwire below makes that LOUD (a typed ``tripwire`` fact
        # in result_meta plus an appended ⚠️ note) without re-introducing the
        # unsound rollback. Disclosed residual: the write itself is not
        # reverted; owner-surface restore is the remedy. Tripwire notes TRAIL
        # the payload (#447 В12): line 1 stays with the command output so its
        # own outcome is never masked; each fired tripwire appends its own
        # note, and the typed facts carry the classification.
        if settings_before is not None:
            settings_after = self._owner_settings_snapshot()
            if settings_after is not None and settings_after != settings_before:
                result = _annotate_result(_append_result_note(result, (
                    "⚠️ OWNER_SETTINGS_CHANGED: data/settings.json changed while this "
                    "command ran. Owner settings change only through save_settings / the "
                    "Settings UI; this write was NOT auto-reverted (a post-hoc rollback "
                    "can clobber a concurrent legitimate owner edit) — the owner surface "
                    "is the place to verify and restore."
                )), tripwire="owner_settings_changed")
        if light_repo_before is not None:
            light_repo_after = _light_repo_snapshot(system_repo_dir_for(self._ctx))
            if (
                light_repo_after is not None
                and light_repo_after.get("digest") != light_repo_before.get("digest")
            ):
                result = _annotate_result(
                    _append_result_note(result, _format_light_repo_write_note(light_repo_before, light_repo_after, tool_name=tool_name)),
                    status="light_mode_blocked", is_failure=True,
                )
        if workspace_refs_before is not None:
            workspace_refs_after = _git_ref_snapshot(active_repo_dir_for(self._ctx))
            if (
                workspace_refs_after is not None
                and workspace_refs_after.get("digest") != workspace_refs_before.get("digest")
            ):
                result = _annotate_result(_append_result_note(result, (
                    "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs "
                    "inside the external workspace. External workspace runs must leave "
                    "changes as files/patch artifacts, not commits/tags/resets."
                )), status="workspace_blocked", is_failure=True)
        return result

    def _heal_mode_block(self, name, args, task_constraint, ext_tool, is_mcp) -> Optional[str]:
        """skill_repair (heal) confinement: return a block message, or None to continue."""
        heal_skill = task_constraint.skill_name if task_constraint else ""
        if (
            name in {"read_file", "list_files", "write_file", "edit_text"}
            and str(args.get("root", "") or "") == "skill_payload"
        ):
            expected_bucket, expected_skill = constraint_bucket_skill(task_constraint)
            requested_bucket = str(args.get("bucket", "") or "").strip()
            requested_skill = str(args.get("skill_name", "") or "").strip()
            if (
                (requested_bucket and requested_bucket != expected_bucket)
                or (requested_skill and requested_skill != expected_skill)
            ):
                if name in {"write_file", "edit_text"}:
                    return (
                        "⚠️ SKILL_REDIRECT_BLOCKED: active skill_repair "
                        "task is scoped to the selected skill payload."
                    )
                return (
                    "⚠️ HEAL_MODE_BLOCKED: Repair payload access is limited "
                    "to the selected skill payload."
                )
        if name in {"read_file", "write_file"} and str(args.get("root", "") or "") == "skill_payload":
            payload_paths = []
            maybe_path = str(args.get("path", "") or "")
            if maybe_path:
                payload_paths.append(maybe_path)
            for f_entry in args.get("files") or []:
                if isinstance(f_entry, dict):
                    payload_paths.append(str(f_entry.get("path", "") or ""))
            for payload_path in payload_paths or ["."]:
                if not _task_constraint_path_allowed(payload_path, task_constraint, pathlib.Path(self._ctx.drive_root)):
                    return (
                        "⚠️ HEAL_MODE_BLOCKED: Repair data access is limited "
                        "to the selected skill payload under data/skills/external "
                        "data/skills/clawhub, or data/skills/ouroboroshub."
                    )
                if name == "write_file" and _heal_protected_payload_sidecar(payload_path):
                    return (
                        "⚠️ HEAL_MODE_BLOCKED: Repair may not edit marketplace "
                        "or official provenance sidecars (.clawhub.json, "
                        ".ouroboroshub.json, SKILL.openclaw.md, .seed-origin). "
                        "Edit the user-authored payload files instead."
                    )
        if name == "list_files" and str(args.get("root", "") or "") == "skill_payload":
            data_dir = str(args.get("path", "") or "")
            if not _task_constraint_path_allowed(data_dir, task_constraint, pathlib.Path(self._ctx.drive_root)):
                return (
                    "⚠️ HEAL_MODE_BLOCKED: Repair data listing is limited "
                    "to the selected skill payload under data/skills/external "
                    "data/skills/clawhub, or data/skills/ouroboroshub."
                )
        if name == "edit_text":
            edit_path = str(args.get("path", "") or "")
            if not _task_constraint_path_allowed(edit_path, task_constraint, pathlib.Path(self._ctx.drive_root)):
                return "⚠️ HEAL_MODE_BLOCKED: Repair edit_text is limited to the selected skill payload."
            if _heal_protected_payload_sidecar(edit_path):
                return (
                    "⚠️ HEAL_MODE_BLOCKED: Repair may not edit marketplace "
                    "or official provenance sidecars (.clawhub.json, "
                    ".ouroboroshub.json, SKILL.openclaw.md, .seed-origin). "
                    "Edit the user-authored payload files instead."
                )
        if name == "skill_review" and str(args.get("skill", "") or "").strip() != heal_skill:
            return "⚠️ HEAL_MODE_BLOCKED: Repair may only review the selected skill."
        if name == "skill_preflight" and str(args.get("skill", "") or "").strip() != heal_skill:
            return "⚠️ HEAL_MODE_BLOCKED: Repair may only preflight the selected skill."
        if ext_tool or is_mcp or name not in _HEAL_MODE_ALLOWED_TOOLS:
            return (
                "⚠️ HEAL_MODE_BLOCKED: Repair tasks may inspect/edit skill "
                "payloads and run skill_review only. Shell, browser automation, "
                "repo mutation, skill execution, extension tools, MCP tools, "
                "delegation, and enable/disable flows are unavailable. Use "
                "the Skills UI after a fresh executable review."
            )
        return None

    def _ephemeral_block(self, name: str, ext_tool: Any = None, is_mcp: bool = False) -> str:
        """CW3: a short ephemeral decision turn may call ONLY the allowlisted read/decision
        tools (_EPHEMERAL_ALLOWED_TOOLS); every other built-in (durable/control/review/skill
        mutator, run_command) AND all extension/MCP tools fail closed. Default-deny, so a new
        mutator can never silently become reachable. It answers inline or promote_chat_to_task's
        the durable work into a supervised task."""
        if not getattr(self._ctx, "is_ephemeral_turn", False):
            return ""
        if ext_tool or is_mcp:
            return (
                f"⚠️ EPHEMERAL_TURN_RESTRICTED: external tool '{name}' can have durable side "
                "effects, which a short same-route decision turn must not do. Answer inline, "
                "or promote_chat_to_task to do that work in a supervised task."
            )
        if name not in _EPHEMERAL_ALLOWED_TOOLS:
            return (
                f"⚠️ EPHEMERAL_TURN_RESTRICTED: '{name}' is not in the decision-turn allowlist "
                "(read/inspect + answer/route/spawn/steer only) — a short same-route turn must "
                "not do durable/control/review/skill work or run shell. Answer inline, or "
                "promote_chat_to_task to do it in a supervised task."
            )
        return ""

    def _subagent_and_update_gate(
        self, name, entry, ext_tool, is_mcp, local_readonly_subagent, acting_subagent, acting_tool_grants
    ) -> str:
        """Early dispatch gates that return a block message (or "" to allow): the read-only and
        acting subagent tool-name allowlists, and the managed-update merge write-exclusivity
        (P2/SC2 — only the authorized resolution task may run code tools while a merge is staged)."""
        if local_readonly_subagent and entry is not None and not self._readonly_tool_allowed(name):
            return (
                "⚠️ LOCAL_READONLY_SUBAGENT_BLOCKED: this subagent may inspect "
                "local repo/data/history plus web/browser surfaces and enabled "
                "external tools, but may not call first-party local tool "
                f"{name!r}. Parent tasks must perform writes, commits, review "
                "gates, tool expansion, runtime control, shell, and skills. "
                "Nested readonly delegation is allowed only through schedule_subagent "
                "within configured depth/cap limits."
            )
        if acting_subagent and entry is not None and name not in ACTING_SUBAGENT_TOOL_NAMES:
            return (
                "⚠️ ACTING_SUBAGENT_BLOCKED: this mutative subagent may read and "
                "write inside its isolated write root and run shell/services "
                f"there, but may not call first-party tool {name!r}. It cannot "
                "commit the live body, run review/runtime/skills lifecycle, enable "
                "tools, or write cognitive memory; the parent integrates the "
                "returned patch and is the sole committer."
            )
        if acting_subagent and entry is None and (ext_tool or is_mcp) and name not in acting_tool_grants:
            return (
                "⚠️ ACTING_SUBAGENT_TOOL_NOT_GRANTED: extension/MCP tool "
                f"{name!r} is not in this acting subagent's external_tool_grants. "
                "The parent must grant dynamic tools explicitly per child."
            )
        # Cover the full repo-mutating surface explicitly (CODE_TOOLS ∪ _REPO_MUTATION_TOOLS):
        # write_file/edit_text AND shell/process tools (run_command/run_script/
        # start_service) are all is_code_tool=True, but gating on the union makes the
        # "no OTHER task writes the repo while a merge is staged" contract robust to flag drift.
        if entry is not None and (name in self.CODE_TOOLS or name in _REPO_MUTATION_TOOLS):
            return _managed_update_code_tool_block(self._ctx, name)
        return ""

    def _resolve_python_predispatch(
        self,
        name: str,
        args: Dict[str, Any],
        runtime_mode: str,
        effective_constraint: Any,
        resolved_binding: Any = None,
    ) -> tuple[Dict[str, Any], Any, str]:
        """Resolve an exact python/python3 request ONCE, before the shell guard.

        Every downstream guard and the handler therefore see byte-identical
        argv; launchers must not select an interpreter after this boundary.
        Python never EXECUTES a candidate; node probes post-gates instead.
        """
        args, python_resolution = resolve_process_python(
            self._ctx,
            name,
            args,
            runtime_mode=runtime_mode,
            effective_constraint=effective_constraint,
            resolved_binding=resolved_binding,
        )
        record_interpreter_resolution(self._ctx, python_resolution)
        if python_resolution is not None and python_resolution.error_reason:
            if python_resolution.error_reason == "cwd_resolution_failed":
                # The failure is the CWD CONFINEMENT policy, not interpreter
                # provenance: python argv is resolved pre-dispatch, so without
                # this the same bad cwd that gets the self-healing
                # SHELL_CWD_BLOCKED root list from a non-python command got an
                # opaque interpreter message naming nothing (submarine waves
                # 1/3, `python3 -m http.server` in a coop tree). Emit the ONE
                # canonical cwd message (label=path root list); the
                # python_interpreter_resolution trace above keeps the true
                # reason, and the typed SHELL_CWD_BLOCKED status lands in the
                # policy-denial family instead of degrading execution.
                return args, python_resolution, shell_cwd_block_message(
                    self._ctx,
                    str((args or {}).get("cwd") or ""),
                    operation="service" if name == "start_service" else "shell",
                )
            return args, python_resolution, (
                "⚠️ PYTHON_INTERPRETER_UNAVAILABLE: Ouroboros could not prove "
                "the target interpreter for this launch surface "
                f"({python_resolution.error_reason}). The process was not started."
            )
        return args, python_resolution, ""


    def _invoke_builtin_handler(
        self,
        name: str,
        entry: Any,
        args: Dict[str, Any],
        resolved_binding: Any,
        interpreter_resolution: Any,
        worktree_before: Any,
    ) -> tuple[str | None, Any]:
        """Run one builtin handler under the scoped attestation."""
        try:
            with interpreter_attestation(self._ctx, interpreter_resolution):
                try:
                    handler_args = dict(args)
                    if resolved_binding is not None:
                        parameters = inspect.signature(entry.handler).parameters
                        if "_resolved_binding" not in parameters:
                            return (
                                f"⚠️ TOOL_INTERNAL_ERROR ({name}): target-sensitive handler "
                                "does not declare the private _resolved_binding keyword.",
                                None,
                            )
                        handler_args["_resolved_binding"] = resolved_binding
                    try:
                        inspect.signature(entry.handler).bind(self._ctx, **handler_args)
                    except TypeError:
                        return _format_tool_arg_error(entry), None
                    return None, entry.handler(self._ctx, **handler_args)
                except TypeError as e:
                    return f"⚠️ TOOL_ERROR ({name}): {e}", None
                except Exception as e:
                    return f"⚠️ TOOL_ERROR ({name}): {e}", None
        finally:
            # Central advisory invalidation by OBSERVED worktree diff: runs on
            # success, tool error, and exception paths alike (the per-tool
            # manual calls missed early-return/error paths), and skips
            # invalidation when a flagged tool ran read-only.
            if worktree_before is not None:
                self._invalidate_advisory_if_worktree_changed(name, worktree_before)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        name = str(name or "").strip()
        args, presence_arg_error = _presence_bound_args(self._ctx, name, args)
        if presence_arg_error:
            return presence_arg_error
        _route_note = ""
        task_constraint = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        local_readonly_subagent = self._is_local_readonly_subagent()
        acting_subagent = self._is_acting_subagent()
        acting_self_worktree = acting_subagent and str(getattr(task_constraint, "surface", "") or "") == "self_worktree"
        acting_protected_grant = acting_subagent and bool(getattr(task_constraint, "protected_paths_grant", False))
        acting_tool_grants = set(getattr(task_constraint, "external_tool_grants", ()) or ()) if acting_subagent else set()
        entry = self._entries.get(name)
        ext_tool = None
        try:
            from ouroboros.extension_loader import parse_extension_surface_name as _ext_parse_name
        except Exception:
            _ext_parse_name = None
        if entry is None and _ext_parse_name and _ext_parse_name(name):
            try:
                from ouroboros.extension_loader import get_tool as _ext_get_tool, is_extension_live as _ext_is_live
                ext_tool = _ext_get_tool(name)
                capability_root = pathlib.Path(((getattr(self._ctx, "task_metadata", {}) or {}).get("budget_drive_root") if isinstance(getattr(self._ctx, "task_metadata", {}), dict) else "") or getattr(self._ctx, "budget_drive_root", "") or getattr(self._ctx, "drive_root", "") or ".").resolve(strict=False)
                if ext_tool and not _ext_is_live(str(ext_tool.get("skill") or ""), capability_root, repo_path=str(ext_tool.get("skills_repo_path") or "") or None):
                    ext_tool = None
            except Exception:
                ext_tool = None
        _mcp_is_name = None
        if entry is None and ext_tool is None:
            try:
                from ouroboros.mcp_client import (
                    ensure_configured_from_settings as _mcp_ensure_configured,
                    is_mcp_tool_name as _mcp_is_name,
                )
                _mcp_ensure_configured(refresh=False)
            except Exception:
                _mcp_is_name = None
        is_mcp = bool(_mcp_is_name and _mcp_is_name(name))
        _eph = self._ephemeral_block(name, ext_tool, is_mcp)  # CW3: built-in deny set + extension/MCP
        if _eph:
            return _eph
        if name in _disabled_tools(self._ctx):
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.disabled_tools withholds {name!r} for this task."
        available, unavailable_reason, unavailable_detail = _builtin_tool_availability(name, self._ctx)
        if not available:
            suffix = f" ({unavailable_detail})" if unavailable_detail else ""
            return f"⚠️ CAPABILITY_UNAVAILABLE: {name!r} is unavailable: {unavailable_reason}{suffix}."
        if name == "vlm_query" and str(args.get("image_url") or "").strip() and (
            not _resource_allowed(self._ctx, "web") or not _resource_allowed(self._ctx, "network")
        ):
            return "⚠️ RESOURCE_CONSTRAINT_BLOCKED: remote image_url for vlm_query requires allowed_resources.web/network."
        if name in _WEB_TOOLS and not _resource_allowed(self._ctx, "web"):
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.web=false blocks {name!r}."
        if name == "vcs_pull_ff" and not _resource_allowed(self._ctx, "network"):
            return "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false blocks 'vcs_pull_ff'."
        if (is_mcp or ext_tool) and not _resource_allowed(self._ctx, "network"):
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false blocks external tool {name!r}."
        _gate = self._subagent_and_update_gate(
            name, entry, ext_tool, is_mcp, local_readonly_subagent, acting_subagent, acting_tool_grants
        )
        if _gate:
            return _gate
        workspace_block_reason = ""
        try:
            workspace_block_reason = workspace_mode_block_reason(self._ctx)
        except Exception as exc:
            workspace_block_reason = f"workspace metadata validation failed: {type(exc).__name__}: {exc}"
        if workspace_block_reason:
            return (
                "⚠️ WORKSPACE_MODE_BLOCKED: invalid external workspace metadata: "
                f"{workspace_block_reason}. Workspace tasks must not overlap the "
                "Ouroboros repo, runtime data, or control plane."
            )
        if entry is not None:
            public_arg_error = _prepare_public_builtin_args(entry, args)
            if public_arg_error:
                return public_arg_error
            _route_note = _normalize_dispatch_path_args(self._ctx, name, args)
            if _route_note.startswith("⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE"):
                return _route_note
        heal_no_enable = bool(task_constraint and task_constraint.mode == "skill_repair")
        if heal_no_enable:
            heal_block = self._heal_mode_block(name, args, task_constraint, ext_tool, is_mcp)
            if heal_block:
                return heal_block
        workspace_mode = bool(getattr(self._ctx, "is_workspace_mode", lambda: False)())
        effective_constraint = task_constraint
        if entry is not None and not (skip_binding := _configured_delegate_selector(self._ctx, name, args)):
            effective_constraint, payload_error = _payload_dispatch_constraint(
                self._ctx,
                name=name,
                args=args,
                task_constraint=task_constraint,
                workspace_mode=workspace_mode,
            )
            if payload_error:
                return payload_error
        resolved_binding = None
        if entry is not None and not skip_binding and _target_binding_operation(name, args) is not None:
            try:
                resolved_binding = _build_builtin_target_binding(self._ctx, name, args)
            except Exception as exc:
                redirect = _light_binding_failure_redirect(name, args)
                if redirect:
                    return redirect
                operation = _target_binding_operation(name, args)
                if operation in {"shell", "service"}:
                    return shell_cwd_block_message(
                        self._ctx,
                        str(args.get("cwd") or ""),
                        operation=operation,
                        error=exc,
                    )
                return _binding_error_text(
                    name,
                    str(args.get("root") or "active_workspace"),
                    exc,
                )
        if not _presence_binding_allowed(self._ctx, resolved_binding):
            return (
                "⚠️ PRESENCE_RESOURCE_BLOCKED: the resolved target is outside "
                "this presence task's positive resource ceiling."
            )
        # Fail-closed: an acting child WITHOUT a resolved isolated workspace would
        # have active_workspace/system_repo fall back to the LIVE repo. Confine it
        # to data roots and block shell/coding/service (whose default target is the repo).
        if acting_subagent and not workspace_mode:
            if name in _ROOT_ARG_REPO_WRITE_TOOLS and str(args.get("root", "") or "active_workspace") in ("active_workspace", "system_repo"):
                return (
                    "⚠️ ACTING_NO_WORKSPACE_BLOCKED: this acting subagent has no resolved isolated "
                    "workspace; write only to root=task_drive, root=artifact_store, or root=user_files. "
                    "active_workspace/system_repo map to the live Ouroboros repo and are blocked."
                )
            if name in ("run_command", "run_script", "start_service",
                        "integrate_subagent_patch", "integrate_delegated_patch"):
                return (
                    "⚠️ ACTING_NO_WORKSPACE_BLOCKED: shell/coding/service/integration tools need an "
                    "isolated workspace (their default target is the live repo). Schedule a self_worktree "
                    "/ external_workspace child for that work."
                )
        # Hardcoded sandbox: light blocks repo mutation; advanced protects
        # core/contracts/release; pro still relies on commit review.
        try:
            from ouroboros.config import get_runtime_mode as _get_runtime_mode
            _runtime_mode = _get_runtime_mode()
        except Exception:
            _runtime_mode = "advanced"
        if is_mcp:
            return self._dispatch_mcp_tool(name, args)
        if entry is None:
            if ext_tool and callable(ext_tool.get("handler")):
                return self._dispatch_extension_tool(name, ext_tool, args)
            return f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(n for n, e in self._entries.items() if not e.alias_for))}"
        args, interpreter_resolution, interpreter_block = self._resolve_python_predispatch(
            name, args, _runtime_mode, effective_constraint, resolved_binding,
        )
        if interpreter_block:
            return interpreter_block
        allow_short_relative = bool(
            effective_constraint and effective_constraint.mode == "skill_repair"
        )
        light_skill_scoped_str_replace = resolved_binding is None and (
            _light_mode_payload_mutation_allowed(
                ctx=self._ctx,
                tool_name=name,
                args=args,
                runtime_mode=_runtime_mode,
                effective_constraint=effective_constraint,
                implicit_skill_cwd_allowed=bool(
                    task_constraint and task_constraint.mode == "skill_repair"
                ),
                allow_short_relative=allow_short_relative,
            )
        )
        if resolved_binding is not None and name not in _SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS:
            light_targets_system = (
                _binding_set_is_light_restricted(self._ctx, resolved_binding)
                or acting_self_worktree
            )
        elif name in _SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS:
            light_targets_system = True
        else:
            light_targets_system = not workspace_mode or acting_self_worktree
        if (
            _runtime_mode == "light"
            and name in _REPO_MUTATION_TOOLS
            and light_targets_system
            and not light_skill_scoped_str_replace
            and not _authorized_managed_update_resolver(self._ctx)
        ):
            return light_cognitive_or_root_redirect(name, args) or (
                "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks Ouroboros "
                f"self-repo/control-plane mutation via {name!r}. For user-visible "
                "deliverables use root=user_files (for example Desktop/file.html), "
                "root=artifact_store for the canonical task artifact, or root=task_drive "
                "for scratch. Skill payload edits remain allowed only through "
                "root=skill_payload with bucket and skill_name "
                "(data/skills/<bucket>/<skill>/) or skill_repair constraints. "
                "Switch to advanced/pro only for reviewed Ouroboros self-modification."
            )
        protected_write_paths = []
        if name in _ROOT_ARG_REPO_WRITE_TOOLS:
            root_name = str(args.get("root", "") or "active_workspace")
            protected_write_paths = [
                canonical_repo_relative_path(self._ctx, root_name, p)
                for p in _payload_write_paths(name, args)
            ]
            if resolved_binding is not None:
                protected_target = (
                    _binding_set_targets_system_repo(self._ctx, resolved_binding)
                    or acting_self_worktree
                )
            else:
                protected_root = root_name in {"active_workspace", "system_repo"}
                protected_target = (
                    (not workspace_mode or acting_self_worktree) and protected_root
                )
            protected_matches = (
                protected_paths_in(protected_write_paths) if protected_target else []
            )
            allow_protected = _authorized_managed_update_resolver(self._ctx) or (
                mode_allows_protected_write(_runtime_mode)
                and (acting_protected_grant or not acting_subagent)
            )
            if protected_matches and not allow_protected:
                first = protected_matches[0]
                return protected_write_block_message(
                    path=first.path,
                    runtime_mode=_runtime_mode,
                    action=f"run tool {name!r} against",
                )

        if _shell_guard_required(name, args):
            if (
                name == "start_service"
                and _runtime_mode == "light"
                and (
                    _binding_set_targets_system_repo(self._ctx, resolved_binding)
                    or acting_self_worktree
                )
            ):
                return ("⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses start_service against the Ouroboros repository because long-running services can mutate after initial tool checks. For external services, set cwd under user_files, task_drive, or artifact_store; switch to advanced/pro only for reviewed Ouroboros self-modification.")
            block_msg = self._run_shell_safety_check(
                process_shell_guard_args(name, args, ctx=self._ctx, runtime_mode=_runtime_mode),
                _runtime_mode,
                resolved_binding,
            )
            if block_msg:
                return block_msg
        # LLM safety supervisor.
        from ouroboros.safety import check_safety
        is_safe, safety_msg = check_safety(
            name,
            args,
            messages=getattr(self._ctx, "messages", None),
            ctx=self._ctx,
            python_resolution=interpreter_resolution,
        )
        if not is_safe:
            return safety_msg
        light_repo_before = (
            _light_repo_snapshot(system_repo_dir_for(self._ctx))
            if (
                name in _PROCESS_COMMAND_TOOLS
                and _runtime_mode == "light"
                and (
                    _binding_set_targets_system_repo(self._ctx, resolved_binding)
                    or acting_self_worktree
                )
            )
            else None
        )
        workspace_refs_before = (
            _git_ref_snapshot(active_repo_dir_for(self._ctx))
            if name in _PROCESS_COMMAND_TOOLS and workspace_mode and acting_self_worktree
            else None
        )
        worktree_before = self._worktree_status_snapshot() if entry.mutates_worktree else None
        settings_before = self._owner_settings_snapshot() if name in _PROCESS_COMMAND_TOOLS else None
        if interpreter_resolution is None:  # node: post-gates (A-F4)
            args, interpreter_resolution = resolve_node_postgates(self._ctx, name, args, runtime_mode=_runtime_mode, effective_constraint=effective_constraint, resolved_binding=resolved_binding)
        early_error, result = self._invoke_builtin_handler(
            name, entry, args, resolved_binding, interpreter_resolution, worktree_before,
        )
        if name in _PROCESS_COMMAND_TOOLS:
            # Tripwires run on the TOOL_ERROR path too: two early_error returns
            # fire AFTER the process already ran (#447 B2).
            checked = self._run_shell_post_checks(
                early_error if early_error is not None else result,
                light_repo_before=light_repo_before,
                workspace_refs_before=workspace_refs_before,
                settings_before=settings_before,
                tool_name=name,
            )
            if early_error is not None:
                return checked
            result = checked
        if early_error is not None:
            return early_error
        return _compose_execute_result(result, _route_note, safety_msg)

    def _worktree_status_snapshot(self) -> str:
        try:
            from ouroboros.utils import run_cmd

            return run_cmd(["git", "status", "--porcelain"], cwd=self._ctx.repo_dir, timeout=20)
        except Exception:
            return "<status-unavailable>"

    def _invalidate_advisory_if_worktree_changed(self, tool_name: str, before: str) -> None:
        after = self._worktree_status_snapshot()
        if after == before:
            return
        try:
            from ouroboros.review_state import invalidate_advisory_after_mutation

            invalidate_advisory_after_mutation(
                pathlib.Path(self._ctx.drive_root),
                mutation_root=pathlib.Path(self._ctx.repo_dir),
                source_tool=tool_name,
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Central advisory invalidation failed for %s", tool_name, exc_info=True
            )

    def override_handler(self, name: str, handler) -> None:
        """Override the handler for a registered tool (used for closure injection)."""
        entry = self._entries.get(name)
        if entry:
            self._entries[name] = ToolEntry(
                name=entry.name,
                schema=entry.schema,
                handler=handler,
                is_code_tool=entry.is_code_tool,
                timeout_sec=entry.timeout_sec,
                mutates_worktree=entry.mutates_worktree,
            )

    @property
    def CODE_TOOLS(self) -> frozenset:
        return frozenset(e.name for e in self._entries.values() if e.is_code_tool)
