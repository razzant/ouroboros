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
from ouroboros.tools.shell_guards import (
    LIGHT_SHELL_WRITER_COMMANDS,
    PROCESS_COMMAND_TOOLS as _PROCESS_COMMAND_TOOLS,
    PROTECTED_RUNTIME_PATHS_LOWER,
    SHELL_GUARDED_TOOLS as _SHELL_GUARDED_TOOLS,
    interpreter_family,
    launches_a_command as _launches_a_command,
    light_shell_repo_mutation,
    parse_porcelain_paths,
    process_shell_guard_args,
    shell_has_write_indicator,
    runtime_data_guard_targets,
    shell_writer_targets_protected,
    workspace_executor_state_write_block,
    writer_target_tokens,
)
from ouroboros.tools.shell_guards import (
    _command_mentions_protected_root,
    _detect_context_mode_self_lowering,
    _detect_evolution_owner_control_self_change,
    _detect_mutative_toggle_self_change,
    _detect_owner_skill_attest_self_call,
    _detect_runtime_mode_elevation,
    _detect_safety_mode_self_lowering,
    _mentions_detached_process,
    _mentions_skill_owner_state,
    _subagent_shell_targets_secret,
)
from ouroboros.tools.shell_guards_runtime import external_shell_runtime_or_secret_block
from ouroboros.tools.shell_guards_target import native_shell_target, native_shell_write_block
from ouroboros.tools.tool_args import (
    _entry_has_public_param_schema,
    _entry_public_params,
    _format_tool_arg_error,
    _normalize_tool_call_args,
)
from ouroboros.tools.dispatch_args import project_dispatch_args
from ouroboros.tools.dispatch_execute import (
    execute_native_operation,
    withdraw_outstanding_prepare,
)
from ouroboros.tools.dispatch_policy import (
    filter_native_listing,
    script_interpreter_refusal,
    subagent_secret_path_refusal,
)
from ouroboros.tools.dispatch_prepare import (
    OutstandingPrepare,
    bind_execution_args,
    native_execution_cwd,
    prepare_operation,
    reconcile_target_args,
)
from ouroboros.remote_task_files import MEDIA_PATH_ARGS, remote_media_predispatch
from ouroboros.workspace_ref import (
    RemoteWorkspacePathError,
    SshWorkspaceRef,
    normalize_remote_root_relative,
    workspace_ref_for,
)
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason
from ouroboros.git_shell_policy import (
    run_shell_git_block_reason,
    target_native_git_violation,
    workspace_git_safety_violation,
)
from ouroboros.tool_access import (
    binding_targets_system_repo,
    build_resolved_resource_binding,
    canonical_repo_relative_path,
    is_external_workspace,
    light_cognitive_or_root_redirect,
    normalize_root,
    normalize_root_relative,
    resolve_shell_cwd,
    shell_cwd_block_message,
    UserFilesPathBlockedError,
    workspace_mode_block_reason,
)
from ouroboros.python_interpreter import record_python_resolution, resolve_process_python
from ouroboros.utils import safe_relpath
from ouroboros.contracts.task_constraint import TaskConstraint, VALID_WRITE_SURFACES, normalize_task_constraint
from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,
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
    """Return the active repo/workspace root for real and lightweight test contexts.

    (RWS v2 §3.1/P1) An SSH placement has NO Home path, so the refusal happens
    HERE, typed, instead of falling through to ``ctx.repo_dir``: that fallback
    would silently aim a remote task's active workspace at the live Ouroboros
    repo, which is precisely the "an SSH binding never degrades to system-repo
    execution" invariant. A malformed sealed placement fails loudly for the same
    reason — a durable record this build cannot honor must not be coerced.
    """
    if isinstance(workspace_ref_for(ctx), SshWorkspaceRef):
        raise RemoteWorkspacePathError(
            "active_workspace is target-native for an ssh placement and has no Home "
            "path; route the operation through the executor instead of resolving a path"
        )
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
    """Return the Ouroboros system repo root, not an external active workspace.

    Home-native under EVERY placement (root matrix, Q2а): the system repo is
    Home by definition, so an ssh placement changes nothing here."""

    return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir"))




_SUBAGENT_SHELL_SECRET_MARKERS = (
    # Ouroboros owner secrets/control state. The relative form (no leading slash)
    # closes the interpreter-string bypass (CW4, v6.34.0): the whole-command
    # substring scan already catches "/data/settings.json" and "../../data/..",
    # but a bare "data/settings.json" (e.g. python -c "open('data/settings.json')"
    # from a workspace cwd) needs the slash-less marker too.
    "/data/settings.json", "data/settings.json", "ouroboros/data/settings", "file1.txt",
    # Universal credential/secret/control files (relative or absolute).
    ".env", ".git/config", ".git/credentials", "credentials.json", "tokens.json",
    "/.ssh/", ".ssh/", "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pgpass", ".aws/",
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
    """Whether this task is the durable tx-authorized assisted resolver."""
    try:
        from supervisor.update_merge import authorized_assisted_task

        return bool(authorized_assisted_task(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        ))
    except Exception:
        return False




# Commands that can only READ. This is an ALLOWLIST on purpose: an unrecognised
# command head is treated as executable access, so the enumeration fails CLOSED.
# (A denylist of "write markers" fails OPEN — every new spelling of a POST walks
# around it, which is exactly the keyword-gate antipattern BIBLE P5 forbids.)
_READ_ONLY_INSPECTION_COMMANDS = frozenset({
    "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack", "ripgrep",
    "cat", "bat", "head", "tail", "less", "more", "nl", "strings",
    "ls", "find", "fd", "stat", "file", "wc", "sort", "uniq", "cut", "tr", "column",
    "basename", "dirname", "realpath", "readlink", "diff", "cmp", "jq", "yq",
    "echo", "printf", "true", "pwd", "date", "tree",
})
# Wrappers that do not themselves act: the real command head follows them.
_COMMAND_HEAD_WRAPPERS = frozenset({
    "sudo", "env", "command", "builtin", "exec", "nohup", "time", "nice", "ionice",
    "stdbuf", "\\",
})
# ``git`` reads only through these subcommands.
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    "grep", "log", "show", "diff", "blame", "cat-file", "ls-files", "ls-tree",
    "rev-parse", "status", "describe",
})
# Allowlist MEMBERSHIP IS NOT ENOUGH: several read heads execute or write through their
# own options. Per command, because short flags are not portable — ``grep -o`` prints
# matches, ``sort -o`` writes a file. Text reaching here is lowercased, so an upper-case
# spelling (``git grep -O``, ``fd -X``) collapses onto the same entry.
_SEARCH_TOOL_EXEC_OPTIONS = frozenset({"--pre", "--pre-glob", "--hostname-bin", "--pager"})
_DENIED_READ_OPTIONS: dict = {
    # find/fd run and delete: -exec/-execdir/-ok/-okdir/-x, -delete, and the -f* writers.
    "find": frozenset({
        "-exec", "-execdir", "-ok", "-okdir", "-delete",
        "-fls", "-fprint", "-fprint0", "-fprintf",
    }),
    "fd": frozenset({"-x", "--exec", "--exec-batch"}),
    "rg": _SEARCH_TOOL_EXEC_OPTIONS,
    "ripgrep": _SEARCH_TOOL_EXEC_OPTIONS,
    "ag": _SEARCH_TOOL_EXEC_OPTIONS,
    "ack": _SEARCH_TOOL_EXEC_OPTIONS,
    "sort": frozenset({"-o", "--output", "--compress-program"}),
    "less": frozenset({"-o", "--log-file", "-k", "--lesskey-file"}),
    "more": frozenset({"-o"}),
    "file": frozenset({"-c", "--compile"}),
    # git: external diff/textconv helpers execute a configured program, -o/--output and
    # git grep -O write or spawn a pager, --exec-path relocates the git binaries.
    "git": frozenset({
        "-c", "--config-env", "--exec-path", "--ext-diff", "--textconv",
        "-o", "--output", "--open-files-in-pager",
    }),
}
# The executable itself must be a bare name or live in a system bin: ``/tmp/evil/grep``
# and ``./grep`` are shadowing, not inspection.
_TRUSTED_EXECUTABLE_DIRS = frozenset({
    "/bin", "/usr/bin", "/usr/local/bin", "/sbin", "/usr/sbin", "/opt/homebrew/bin",
})


def _trusted_read_head(token: str) -> str:
    """The allowlist-comparable command name, or "" when the executable is untrusted."""
    if "\\" in token:
        return ""  # a windows/escaped path is not a form we can resolve — fail closed
    directory, sep, name = token.rpartition("/")
    if sep and directory not in _TRUSTED_EXECUTABLE_DIRS:
        return ""
    return name.removesuffix(".exe")


def _denied_read_option(token: str, denied: frozenset) -> bool:
    """True when an argument spells an execution/mutation option of its command."""
    if not token.startswith("-") or token in {"-", "--"}:
        return False
    name = token.split("=", 1)[0]
    if name in denied:
        return True
    if name.startswith("--"):
        return False
    return any(f"-{letter}" in denied for letter in name[1:])  # bundled short cluster


# Spellings that make a shell run a command NESTED inside another one. The read exemption
# fails closed on all of them: the head-allowlist can only vouch for heads it actually sees,
# and a nested command's head is not one of them ("echo" vouching for the "curl -X POST" it
# interpolates). Refusing the CONSTRUCT rather than enumerating the payloads inside it is the
# point — no list of "what a write looks like" is ever complete (BIBLE P5).
_NESTED_EXECUTION_MARKERS = ("$(", "`", "<(", ">(")
# Bare tokens the lexer emits for the same constructs (and for a plain subshell). These used to
# be STRIPPED from the token list before the head was taken, which is precisely how the nested
# command escaped validation; they are refused instead.
_NESTED_EXECUTION_TOKENS = frozenset({"$", "(", ")", "<(", ">(", "$("})


def _is_pure_read_inspection(text_lower: str) -> bool:
    """True when EVERY command in a shell line is a read-only source inspection.

    Structural, not keyword-based: the line is split into per-command segments with
    the shared lexer (``shell_parse.shell_segments``) and each segment's HEAD is
    matched against an allowlist. An unknown head — any interpreter, HTTP client,
    or shell — is not an inspection, whatever flags or payload spelling it carries.

    Head membership is NECESSARY, NOT SUFFICIENT (review round 2): an allowed head can
    still execute through its own options (``find -exec``, ``rg --pre``, git's external
    diff/textconv) or through what precedes it. So the options are validated per command
    (``_DENIED_READ_OPTIONS``), a leading environment assignment is REFUSED rather than
    dropped (``PATH=``/``LD_PRELOAD=``/``GIT_EXTERNAL_DIFF=`` change what actually runs),
    wrappers may not carry their own flags (``env -i``, ``sudo -e``), and the executable
    must resolve to a bare name or a system bin. Anything unrecognised stays fail-closed.

    NESTED EXECUTION IS REFUSED BEFORE ANY OF THAT (review round 3). Only the heads the lexer
    actually surfaces get validated, so a command substitution hid its command from every check
    above: ``echo "$(curl -X POST .../api/owner/scope-review-floor)"`` presented the allowlisted
    ``echo``, and the write-shape detector does not recognise an HTTP POST, so the exemption was
    granted to a line that existed to reach the owner-only endpoint. A quoted substitution is
    one opaque argument token to the lexer, which is why this is a check on the TEXT and on the
    tokens, not something the per-segment head walk could have caught.
    """
    from ouroboros.shell_parse import shell_segments

    if any(marker in text_lower for marker in _NESTED_EXECUTION_MARKERS):
        return False
    segments = shell_segments(text_lower)
    if not segments:
        return False
    for segment in segments:
        if any(token in _NESTED_EXECUTION_TOKENS for token in segment):
            return False
        tokens = [token for token in segment if token]
        while tokens and tokens[0] in _COMMAND_HEAD_WRAPPERS:
            tokens = tokens[1:]
            if tokens and tokens[0].startswith("-"):
                return False  # a wrapper's own options can rebuild the environment
        if not tokens:
            continue  # a bare wrapper executes nothing
        if "=" in tokens[0] and not tokens[0].startswith(("-", "=")):
            return False  # leading env assignment: never silently discarded
        head = _trusted_read_head(tokens[0])
        if head == "git":
            if len(tokens) < 2 or tokens[1] not in _READ_ONLY_GIT_SUBCOMMANDS:
                return False
        elif not head or head not in _READ_ONLY_INSPECTION_COMMANDS:
            return False
        denied = _DENIED_READ_OPTIONS.get(head)
        if denied and any(_denied_read_option(token, denied) for token in tokens[1:]):
            return False
    return True


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

    The auto-route note TRAILS the result: failure classification
    (loop_tool_execution) inspects the FIRST line, so a leading note would mask
    an underlying tool error on the auto-routed read path (review round 3). The
    safety warning keeps its historical leading position — its ``---`` separator
    is an established transcript convention the metadata scan already handles."""
    if route_note:
        result = f"{result}\n\n{route_note}"
    if safety_msg:
        return f"{safety_msg}\n\n---\n{result}"
    return result




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






def _heal_protected_payload_sidecar(path_text: str) -> bool:
    return is_skill_payload_control_filename(path_text)


# verify_and_record runs the agent's declared `check` like a command, so it must clear the
# same PRE-EXECUTION shell guards (subagent-secret read, protected-artifact read, sudo,
# protected-root / workspace-state / light-mode writes) — that pre-exec filter is the
# security boundary and blocks a forbidden mutation BEFORE the handler runs, so a guarded
# check cannot mutate protected state and then leave a host-attested PASS receipt. It is
# deliberately NOT in _PROCESS_COMMAND_TOOLS: those POST-execution checks (owner-file
# restore, light-repo diff, git-ref tripwire) run AFTER the handler has already written the
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


def _executor_backend_candidate_allowed(ctx: Any, candidate: str, allowed_roots: List[pathlib.Path]) -> bool:
    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx as _executor_ref_from_ctx
        from ouroboros.workspace_executor import map_backend_path as _executor_map_backend_path

        executor_ref = _executor_ref_from_ctx(ctx)
        if executor_ref is None:
            return False
        resolved = _executor_map_backend_path(executor_ref, candidate)
        return any(resolved.is_relative_to(root) for root in allowed_roots)
    except Exception:
        return False



def _root_relative_normalizer(ctx: Any, root_arg: str) -> Callable[[str], str]:
    """Path-arg normalizer for ``root_arg`` IN THAT ROOT'S NATIVE SPELLING SPACE.

    (RWS-05) For an ssh placement `active_workspace` normalizes against the
    sealed ref's target root with pure posix semantics; every other root — and
    every local/docker placement — keeps the Home resolver byte-for-byte.
    """
    if root_arg == "active_workspace":
        ref = workspace_ref_for(ctx)
        if isinstance(ref, SshWorkspaceRef):
            remote_root = ref.remote_root
            return lambda text: normalize_remote_root_relative(remote_root, text)
        root = active_repo_dir_for(ctx)
    else:
        root = system_repo_dir_for(ctx)
    return lambda text: normalize_root_relative(root, text)


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
            _norm = _root_relative_normalizer(ctx, root_arg)
            for _key in ("path", "dir"):
                if isinstance(args.get(_key), str) and args[_key]:
                    args[_key] = _norm(args[_key])
            if isinstance(args.get("files"), list):
                for _f in args["files"]:
                    if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                        _f["path"] = _norm(_f["path"])
        except Exception:
            pass
        return ""
    if root_arg != "user_files" or name == "query_code":
        return ""
    if isinstance(workspace_ref_for(ctx), SshWorkspaceRef):
        # `user_files` is Home-native and the workspace lives on the target, so a
        # Home absolute path can never be "under the active workspace" — there is
        # nothing to auto-route, and a Home path must never be compared against a
        # target spelling (root matrix, Q2а).
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
    return frozenset(names)


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


def _target_binding_operation(name: str, args: dict[str, Any]) -> str | None:
    operation = _TARGET_BINDING_OPERATIONS.get(name)
    if operation is not None:
        return operation
    if name in _SKILL_LIFECYCLE_TARGET_TOOLS:
        return "review"
    if name in _PROCESS_TARGET_TOOLS:
        return "service" if name == "start_service" else "shell"
    if name == "verify_and_record" and _launches_a_command(name, args):
        return "shell"
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




def _prepare_public_builtin_args(entry: "ToolEntry", args: dict[str, Any]) -> str:
    """Normalize and validate only the model-visible builtin argument surface.

    This runs after capability/lineage availability checks but before path
    normalization, target selection, Python predispatch, or target-sensitive
    guards. Private dispatch carriers therefore cannot be supplied by the model
    and invalid public calls cannot trigger target work before rejection.
    """

    _normalize_tool_call_args(entry, args)
    public_params = set(_entry_public_params(entry))
    if _entry_has_public_param_schema(entry) and any(key not in public_params for key in args):
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


def _authorized_process_roots(binding: Any) -> list[pathlib.Path]:
    """The binding roots that are an AUTHORIZED process target, as plain paths.

    An explicitly selected system repo or exact skill payload was chosen by the
    caller, so the runtime guard must not re-block it merely because the task also
    has an external workspace focus. Projecting it here keeps the guard itself free
    of the binding type — and free of any second cwd resolution (D1).
    """
    return [
        pathlib.Path(item.base_path)
        for item in _binding_items(binding)
        if item.root in {"system_repo", "skill_payload"}
    ]



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


def _binding_state_drive_root(ctx: Any, binding: Any) -> pathlib.Path:
    items = _binding_items(binding)
    if items:
        return pathlib.Path(items[0].state_drive_root)
    return pathlib.Path(ctx.drive_root)


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


def _format_light_repo_write_block(before: Dict[str, Any], after: Dict[str, Any], result: str, tool_name: str = "run_command") -> str:
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
        f"Affected/dirty paths: {listed}. Switch to advanced/pro for repo writes.\n\n"
        "Original command output:\n"
        f"{result}"
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


# CW3 (v6.34.0): tools a SHORT-LIVED ephemeral same-route decision turn must NOT
# call — durable cognitive memory, evolution/consciousness, model/timeout/settings
# control, and the release/restart control-plane. The ephemeral turn may still
# answer / steer_task / promote_chat_to_task / route_to_project and read freely;
# An ephemeral decision turn DECIDES (answer / route / spawn / steer); it does NOT do
# durable work — that is what the task it spawns is for. CW3 (v6.34.0) enforces this with
# a DEFAULT-DENY ALLOWLIST, not a denylist: a denylist is whack-a-mole (it kept missing
# review/skill/publish/control mutators — advisory_review, skill_review, submit_skill_to_hub,
# skill_exec, toggle_skill, cancel_task, task_acceptance_review, ...). The decision turn may
# only call the read-only INSPECTION tools (the LOCAL_READONLY_SUBAGENT_TOOL_NAMES SSOT —
# read_file/query_code/search_code/web_search/vcs_diff/...) plus the route/spawn/steer/reply
# tools below. Everything else — every repo/git/cognitive/control/review/skill/publish
# mutator, run_command (shell is durable-capable), and all extension/MCP tools (blocked
# separately) — is hidden from schemas()/get_schema_by_name() and fails closed in execute().
# EXPLICIT curated allowlist (not derived from another set — deriving from
# LOCAL_READONLY_SUBAGENT_TOOL_NAMES leaked subagent-only tools: schedule_subagent spawns
# durable child tasks, wait_task/wait_tasks BLOCK a short turn, browser_action INTERACTS
# with pages). A decision turn may only READ/INSPECT (no mutation, no spawning, no blocking
# wait, no page interaction) and answer/route/spawn-owner-task/steer/reply.
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
        # Home-local contract (§3.1): legal for local placements and materialized
        # snapshots only. An ssh placement has no Home path and refuses TYPED at the
        # ref seam rather than handing back the system repo.
        if isinstance(workspace_ref_for(self), SshWorkspaceRef):
            raise RemoteWorkspacePathError(
                "active_repo_dir is Home-local; an ssh workspace has no Home path — "
                "route the operation through the executor"
            )
        if self.is_workspace_mode():
            return pathlib.Path(self.workspace_root)
        return pathlib.Path(self.repo_dir)

    def is_workspace_mode(self) -> bool:
        """Whether this task runs in an EXTERNAL workspace (placement-blind flag).

        (RWS v2) The predicate reads the SEALED placement, not the Home path: an ssh
        placement has no ``workspace_root`` by construction, and answering False for it
        would put a remote task on the workspace-less ``self_modification`` profile over
        the live Ouroboros repo — exactly the degradation the placement contract forbids.
        ``workspace_mode_block_reason`` guards overlap with HOME roots, which is
        structurally impossible for a target-native root, so it stays local-only.
        """
        if not bool(str(self.workspace_mode or "").strip()):
            return False
        if isinstance(workspace_ref_for(self), SshWorkspaceRef):
            return True
        return self.workspace_root is not None and not workspace_mode_block_reason(self)

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


@dataclass(frozen=True)
class _DispatchGates:
    """What the placement-BLIND half of the dispatch decided.

    A frozen record rather than a tuple on purpose: eleven values cross this seam,
    and a positional unpack would let a twelfth silently shift every caller.
    """

    route_note: str
    task_constraint: Any
    entry: Any
    ext_tool: Any
    is_mcp: bool
    acting_subagent: bool
    acting_self_worktree: bool
    acting_protected_grant: bool
    workspace_mode: bool
    effective_constraint: Any
    runtime_mode: str


class ToolRegistry:
    """Tool registry; modules export ``get_tools()``."""

    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self._entries: Dict[str, ToolEntry] = {}
        self._ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
        self._capability_omissions: List[Dict[str, Any]] = []
        self._load_modules()

    _FROZEN_TOOL_MODULES = [
        "browser", "ci", "claude_advisory_review", "compact_context", "control",
        "core", "delegate", "edit_ops", "evolution_stats", "git", "git_pr", "git_rollback", "github",
        "health", "join_ledger", "knowledge", "media", "memory_tools", "plan_review", "project_journal",
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
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to load tool module %s", modname, exc_info=True)

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

    def _acting_tool_grants(self) -> set:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return set(getattr(tc, "external_tool_grants", ()) or ()) if tc else set()

    def initial_tool_names(self) -> frozenset[str]:
        if self._is_local_readonly_subagent():
            return LOCAL_READONLY_SUBAGENT_TOOL_NAMES
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
            if e.name not in disabled  # declarative tool policy (task_contract.disabled_tools)
            if _builtin_tool_availability(e.name, self._ctx)[0]
            if not local_readonly_subagent or e.name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
            if not acting_subagent or e.name in ACTING_SUBAGENT_TOOL_NAMES
        ]

    def _schema_for_entry(self, entry: ToolEntry) -> Dict[str, Any]:
        schema = entry.schema
        if self._is_local_readonly_subagent():
            if entry.name in {"read_file", "list_files", "search_code", "query_code"}:
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
        elif self._is_acting_subagent():
            # Advertise only what the acting profile can actually execute: writes go
            # ONLY to the isolated surface (active_workspace); reads use the read roots;
            # browser evaluate is unavailable (rejected at execute time).
            if entry.name in _ROOT_ARG_REPO_WRITE_TOOLS or entry.name in _GENERIC_VCS_TARGET_TOOLS:
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
            elif entry.name == "browser_action":
                schema = copy.deepcopy(entry.schema)
                props = schema.get("parameters", {}).get("properties", {})
                action_schema = props.get("action", {})
                if isinstance((action_enum := action_schema.get("enum")), list):
                    action_schema["enum"] = [name for name in action_enum if name != "evaluate"]
        return {"type": "function", "function": schema}

    def _schemas_for_entry(self, entry: ToolEntry) -> List[Dict[str, Any]]:
        return [self._schema_for_entry(entry)]

    def schemas(self, core_only: bool = False) -> List[Dict[str, Any]]:
        acting_subagent = self._is_acting_subagent()
        acting_grants = self._acting_tool_grants() if acting_subagent else set()
        local_readonly_subagent = self._is_local_readonly_subagent()
        ephemeral_turn = bool(getattr(self._ctx, "is_ephemeral_turn", False))
        disabled_tools = _disabled_tools(self._ctx)
        self._capability_omissions = []
        unavailable_tools = {
            entry.name: detail
            for entry in self._entries.values()
            for available, reason, detail in [_builtin_tool_availability(entry.name, self._ctx)]
            if not available and reason == "missing_credential" and entry.name not in disabled_tools
        }
        built_in = [
            schema
            for entry in self._entries.values()
            if entry.name not in disabled_tools  # declarative tool policy (task_contract.disabled_tools)
            if entry.name not in unavailable_tools
            if not local_readonly_subagent or entry.name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
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
            if e.name in disabled_tools:  # declarative tool policy (task_contract.disabled_tools)
                continue
            if e.name in unavailable_tools:
                continue
            if local_readonly_subagent and e.name not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
                continue
            if acting_subagent and e.name not in ACTING_SUBAGENT_TOOL_NAMES:
                continue
            if ephemeral_turn and e.name not in _EPHEMERAL_ALLOWED_TOOLS:
                continue  # CW3: the core/initial envelope is allowlisted too, not just schemas(core_only=False)
            if (
                (local_readonly_subagent and e.name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES)
                or (acting_subagent and e.name in ACTING_SUBAGENT_TOOL_NAMES)
                or e.name in CORE_TOOL_NAMES
                or e.name in ("list_available_tools", "enable_tools")
            ):
                result.extend(self._schemas_for_entry(e))
        ext = extension_schemas
        if disabled_tools:
            ext = [s for s in ext if (s.get("function", {}) or {}).get("name") not in disabled_tools]
        return result + ext

    def workspace_capability_manifest(
        self,
        *,
        repo_root: pathlib.Path,
    ) -> Dict[str, Any]:
        """Build the Home/execd capability contract from the unfiltered built-ins.

        Per-task visibility and dynamic filtering are bypassed ON PURPOSE: the
        manifest is a property of the BUILD, not of a task. If it were filtered,
        two tasks on the same server would compute different digests and a target
        admitted by one would be refused by the other. Every schema still comes
        from the one `_entries` SSOT, so the contract cannot drift from the tools
        that actually exist.

        Which is why the envelope is built HERE from `entry.schema` and not through
        `_schema_for_entry`. That method is the PER-TASK projection: it narrows `root`
        enums and drops fields for a local-readonly or an acting subagent
        (`_is_local_readonly_subagent`/`_is_acting_subagent` read the live `ToolContext`),
        so calling it made this "unfiltered" only for the contexts that happen not to
        filter. The same build could then produce two different manifests — and
        `manifest_sha256` is a Home↔execd compatibility identity that admission
        compares, so the drift would surface as a target admitted by one task and
        refused by another with no fact about the target having changed. The schemas are
        deep-copied because the manifest builder canonicalizes them and `_entries` is
        the SSOT the live tool surface reads.
        """

        from ouroboros.tool_capabilities import (
            WORKSPACE_TOOL_EXECUTION_AFFINITY,
            build_workspace_capability_manifest,
        )

        missing = sorted(set(WORKSPACE_TOOL_EXECUTION_AFFINITY) - set(self._entries))
        if missing:
            raise ValueError(
                "the workspace capability surface names tools this registry does "
                f"not register: {missing}"
            )
        public_schemas = [
            {"type": "function", "function": copy.deepcopy(self._entries[name].schema)}
            for name in sorted(WORKSPACE_TOOL_EXECUTION_AFFINITY)
        ]
        return build_workspace_capability_manifest(
            public_schemas,
            repo_root=pathlib.Path(repo_root),
        )

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
        if requested not in self._entries:
            return None
        available, reason, _detail = _builtin_tool_availability(requested, self._ctx)
        if not available:
            return f"unavailable ({reason})"
        if getattr(self._ctx, "is_ephemeral_turn", False) and requested not in _EPHEMERAL_ALLOWED_TOOLS:
            return "hidden on this ephemeral decision turn (allowlist)"
        acting_subagent = self._is_acting_subagent()
        if self._is_local_readonly_subagent() and requested not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
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
        entry = self._entries.get(requested)
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
            if local_readonly_subagent and requested not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
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
        return f"{safety_msg}\n\n---\n{result}" if safety_msg else result

    def _protected_shell_block(
        self, raw_cmd, cmd_path_lower, binding, acting_self_worktree,
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
        ) and shell_has_write_indicator(raw_cmd):
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
                if cf in cmd_path_lower and shell_has_write_indicator(raw_cmd):
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



    def _workspace_shell_write_block(
        self,
        args: Dict[str, Any],
        raw_cmd: Any,
        cmd_path_lower: str,
        explicit_write_targets: list[str],
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
                return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
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
                        if any(resolved.is_relative_to(root) for root in allowed_relative_roots):
                            continue
                        if any(resolved.is_relative_to(root) for root in allowed_data_roots):
                            continue
                        for protected_path in protected_paths:
                            try:
                                resolved.relative_to(protected_path)
                                return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
                            except Exception:
                                pass
                        if not pro_workspace_passthrough:
                            return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths outside the selected process root."
                        continue
                    if any(path_text_is_inside(candidate, root) for root in allowed_relative_roots):
                        continue
                    if any(path_text_is_inside(candidate, root) for root in allowed_data_roots):
                        continue
                    for protected_path in protected_paths:
                        if path_text_is_inside(candidate, protected_path):
                            return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
                    if not pro_workspace_passthrough:
                        return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths outside the selected process root."
                    continue
                resolved = (work_dir / pathlib.Path(candidate)).resolve(strict=False)
                if any(resolved.is_relative_to(root) for root in allowed_relative_roots):
                    continue
                if any(resolved.is_relative_to(root) for root in allowed_data_roots):
                    continue
                for protected_path in protected_paths:
                    try:
                        resolved.relative_to(protected_path)
                        return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
                    except Exception:
                        pass
                if not pro_workspace_passthrough:
                    return "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths outside the selected process root."
        return None

    def _run_shell_safety_check(
        self, args: Dict[str, Any], runtime_mode: str, binding: Any = None,
        target: Any = None,
    ) -> Optional[str]:
        """Pre-execution run_command filter; returns a block message or ``None``.

        ``target`` is the TARGET-NATIVE fact object, present for exactly the
        dispatches that will run on another machine. The placement-BLIND arms —
        every rule that reads command TEXT: elevation, self-lowering, skill-state,
        GitHub — run unchanged. The arms that resolve a HOME path are swapped for
        their target spelling, because a Home path is not a fact a target-native
        command can address."""
        raw_cmd = args.get("cmd", args.get("command", ""))
        if binding is None and target is None:
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
        if sudo_noninteractive_violation(argv):
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
        if argv_executable in {"sh", "bash", "zsh"}:
            inline_cmd = next((str(argv_for_write[idx + 1] or "") for idx, token in enumerate(argv_for_write[1:], start=1) if str(token or "") in {"-c", "--command"} and idx + 1 < len(argv_for_write)), "")
            if not inline_cmd:
                inline_cmd = shell_command_string(argv_for_write)
            inline_argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(inline_cmd)))
            if inline_argv:
                write_target_argvs.append(inline_argv)
        explicit_write_targets = list(dict.fromkeys(str(token) for target_argv in write_target_argvs for token in writer_target_tokens(target_argv) if str(token or "").strip()))
        executable_path_tokens = {str(target_argv[0]) for target_argv in write_target_argvs if target_argv}
        # Writer-command membership canonicalizes versioned interpreter spellings to
        # their family (`ruby3.2` is `ruby`), so a versioned basename is exactly as
        # write-suspect as the unversioned one (XG-2R.2).
        writeish = shell_has_write_indicator(raw_cmd) or (bool(argv_for_write) and (interpreter_family(argv_executable) or argv_executable) in LIGHT_SHELL_WRITER_COMMANDS) or bool(explicit_write_targets)
        work_dir = None
        if target is not None:
            # The three Home-path arms below, in ONE target-spelling arm: protected
            # artifacts and workspace write-containment are the rules that still mean
            # something where the command actually runs. The executor-state arm is
            # absent because it guards Home's own `data/state/...` process ledger,
            # which no target-native command can address.
            if native_block := native_shell_write_block(
                target, raw_cmd,
                writeish=writeish,
                explicit_write_targets=explicit_write_targets,
                executable_path_tokens=executable_path_tokens,
            ):
                return native_block
        else:
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
                    executable_path_tokens,
                    runtime_mode,
                    acting_subagent,
                    binding,
                )
                if workspace_write_block:
                    return workspace_write_block

        # Elevation pattern: blocked in all modes.
        if _detect_runtime_mode_elevation(cmd_lower):
            return "⚠️ ELEVATION_BLOCKED: shell command pattern looks like an OUROBOROS_RUNTIME_MODE elevation attempt (mentions ``save_settings`` together with ``OUROBOROS_RUNTIME_MODE``, or invokes ``ouroboros.config.save_settings`` directly). Runtime mode is owner-controlled — change it by stopping the agent and editing settings.json directly, then restart."
        if _detect_context_mode_self_lowering(cmd_lower):
            return "⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to lower OUROBOROS_CONTEXT_MODE to low through settings.json or /api/owner/context-mode. Context mode is owner-controlled — ask the owner to change the Low/Max toggle or edit settings while the agent is stopped."
        if _detect_scope_review_floor_self_lowering(cmd_lower, writeish=writeish):
            return "⚠️ SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED: shell command pattern reaches OUROBOROS_SCOPE_REVIEW_FLOOR through settings.json, /api/settings, or /api/owner/scope-review-floor from something other than a pure read. The floor is a deprecated, enforcement-inert owner setting (BIBLE P3 scope-review applicability follows the owner context mode) — it stays owner-only, and the agent must not write owner settings through any channel. Ask the owner to change it via the dedicated /api/owner/scope-review-floor endpoint, or stop the agent and edit settings.json directly. Pure source inspection (grep/rg/cat/jq/git grep) is allowed; an interpreter or HTTP client naming the endpoint is not, whatever verb it spells."
        if _detect_safety_mode_self_lowering(cmd_lower):
            return "⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to change OUROBOROS_SAFETY_MODE (e.g. to ``light``/``off``) through settings.json, /api/settings, or /api/owner/safety-mode. LLM-safety coverage is owner-controlled (BIBLE P3) — the agent must not reduce its own supervision. Ask the owner to change it via the dedicated /api/owner/safety-mode endpoint, or stop the agent and edit settings.json directly."
        if _detect_owner_skill_attest_self_call(cmd_lower):
            return "⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: shell command pattern looks like an attempt to loopback-POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips the expensive LLM skill review and is OWNER-ONLY — the agent must not self-attest its own skill to bypass the immune system's review. Ask the owner to attest it from the Skills UI."
        if _detect_mutative_toggle_self_change(cmd_lower):
            return "⚠️ ELEVATION_BLOCKED: OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is owner-controlled (it grants subagents write power against the live body). Change it by stopping the agent and editing settings.json directly, then restart — the agent must not self-enable mutative subagents."
        if _detect_evolution_owner_control_self_change(cmd_lower):
            return "⚠️ ELEVATION_BLOCKED: the self-evolution controls (OUROBOROS_POST_TASK_EVOLUTION and OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE) are owner-controlled — they enable or steer self-modification cycles. Change them via the owner Settings UI, or stop the agent and edit settings.json directly — the agent must not self-set evolution controls."
        if _mentions_skill_owner_state(cmd_lower):
            return (
                "⚠️ SKILL_STATE_WRITE_BLOCKED: skill review, enablement, "
                "grants, and marketplace provenance are owner/review "
                "controlled state. Use skill_review, toggle_skill/the Skills "
                "UI, or the desktop launcher confirmation flow."
            )
        if "state" in cmd_lower and "skills" in cmd_lower and _mentions_detached_process(cmd_lower):
            return (
                "⚠️ SKILL_STATE_WRITE_BLOCKED: detached shell processes must "
                "not target skill state directories. Use the reviewed skill "
                "lifecycle tools instead."
            )

        # Light-mode checks follow the selected physical target, not whether a
        # project workspace happens to be attached. `target is None` because this
        # whole arm is about the Ouroboros runtime's OWN repo and data drives: it
        # exists to keep the agent from mutating its own body, and a target-native
        # command cannot reach either — the body it could mutate is on Home.
        if runtime_mode == "light" and target is None:
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
            raw_cmd, cmd_path_lower, binding, acting_self_worktree,
        ):
            return protected_shell

        # GitHub repo create/delete/auth.
        cmd_words = re.sub(r"\s+", " ", cmd_lower)
        if "gh repo create" in cmd_words or "gh repo delete" in cmd_words:
            return "⚠️ SAFETY_VIOLATION: Creating/deleting GitHub repositories requires admin approval."
        if "gh auth" in cmd_words:
            return "⚠️ SAFETY_VIOLATION: Modifying GitHub authentication is not permitted."

        return self._shell_git_and_runtime_block(
            raw_cmd, args, cmd_path_lower, workspace_mode,
            acting_self_worktree, binding, target,
        )

    def _shell_git_and_runtime_block(
        self, raw_cmd: Any, args: Dict[str, Any], cmd_path_lower: str,
        workspace_mode: bool, acting_self_worktree: bool, binding: Any,
        target: Any = None,
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
        if target is not None:
            # Target-native. Every arm below protects HOME state — system repo, data
            # drives, owner credentials — none of which exists on the target, so
            # neither containment question has an answer there. What survives is the
            # resource contract: `allowed_resources.network=false` must not become
            # satisfiable by running the fetch on another host.
            git_violation = target_native_git_violation(
                raw_cmd, allow_network=_resource_allowed(self._ctx, "network")
            )
            return f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}." if git_violation else None
        if workspace_mode and not acting_self_worktree:
            work_dir = self._resolved_shell_cwd(args, binding)
            if isinstance(work_dir, str):  # a cwd block message, not a path
                return work_dir
            if git_block := self._external_workspace_git_block(raw_cmd, work_dir):
                return git_block
            # Even READ-only, non-git shell (cat/head/grep/python -c open(...)) must
            # not reach the runtime or secrets — close the raw-shell bypass of the
            # user_files path guard (scoped to top-level external tasks).
            #
            # READ-ONLY GIT IS EXEMPT (owner contract, Q4=A: "read-only everywhere",
            # and the f14baf8f false-block class). `git -C <system repo> status|log|
            # diff|show|rev-parse` is the vcs_status-equivalent inspection lane; the
            # runtime-read guard was catching it by path token and refusing it with a
            # WORKSPACE_SHELL_BLOCKED that named the wrong reason. The marginal
            # escalation is nil — the same history is already readable through the
            # gated read_file this very message points the agent at — while the
            # SECRET/credential surface stays closed because the exemption is
            # ALL-or-nothing per segment (`git status && cat <data>/settings.json`
            # is not exempt; every non-git shell still meets the full guard) AND
            # write-aware: `is_readonly_git_command` refuses the key to a read-only
            # subcommand carrying the file-truncating `--output=<file>` diff option
            # or `--no-index` (which reads arbitrary host files), so neither a
            # runtime write nor a settings.json dump can ride "read-only git".
            if is_external_workspace(self._ctx) and not is_readonly_git_command(raw_cmd):
                if ext_block := external_shell_runtime_or_secret_block(
                    self._ctx, raw_cmd, cmd_path_lower, args,
                    work_dir=work_dir,
                    authorized_roots=_authorized_process_roots(binding),
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
        # DEFAULT (non-workspace) lane — direct chat, light mode, self_modification-
        # profile tasks. Q4=A (owner, 2026-08-08): mutating git is free EVERYWHERE
        # outside the Ouroboros runtime, in every runtime mode and lane. The
        # argv-text blanket (blocked ANY mutating git with a commit_reviewed remedy
        # that is false for non-repo trees) is replaced by the SAME target-aware
        # resolver the external lane has run since v6.27: read-only git stays
        # allowed even at a runtime target, mutating git is blocked only when it
        # TARGETS the runtime (bidirectional/casefold/symlink-resolved containment),
        # and the contract network fence rides along. The cwd resolves EXACTLY ONCE
        # through the shared resolver and is passed as a canonical path — never
        # re-join a raw label onto a root (the v6.74.0 D1 regression class).
        # Disclosed residual (proportionality; no shell-parser arms race): git via
        # a transparent wrapper (nice/xargs) or interpreter code is not classified
        # here — the pre-flip text classifier never saw the interpreter form either,
        # and the LLM safety layer still reviews intent. The light-mode post-exec
        # system-repo dirtiness tripwire stays as the backstop.
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

    def _snapshot_owner_files(
        self, state_drive_root: pathlib.Path | None = None,
    ) -> Dict[pathlib.Path, Optional[str]]:
        from ouroboros import config as _cfg
        out: Dict[pathlib.Path, Optional[str]] = {}
        settings_path = pathlib.Path(_cfg.SETTINGS_PATH)
        try:
            out[settings_path] = settings_path.read_text(encoding="utf-8") if settings_path.is_file() else None
        except OSError:
            out[settings_path] = None
        root = pathlib.Path(state_drive_root or self._ctx.drive_root) / "state" / "skills"
        if not root.is_dir():
            return out
        for path in root.glob("*/*"):
            if path.name.lower() not in SKILL_OWNER_STATE_FILENAMES:
                continue
            try:
                out[path] = path.read_text(encoding="utf-8")
            except OSError:
                out[path] = None
        return out

    def _restore_owner_files(
        self,
        before: Dict[pathlib.Path, Optional[str]],
        state_drive_root: pathlib.Path | None = None,
    ) -> bool:
        from ouroboros import config as _cfg
        root = pathlib.Path(state_drive_root or self._ctx.drive_root) / "state" / "skills"
        current = set()
        if root.is_dir():
            current.update(
                path for path in root.glob("*/*")
                if path.name.lower() in SKILL_OWNER_STATE_FILENAMES
            )
        settings_path = pathlib.Path(_cfg.SETTINGS_PATH)
        current.add(settings_path)
        changed = False
        for path in current - set(before):
            try:
                path.unlink()
                changed = True
            except OSError:
                pass
        for path, content in before.items():
            try:
                if content is None:
                    if path.exists():
                        path.unlink()
                        changed = True
                    continue
                if not path.exists() or path.read_text(encoding="utf-8") != content:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    changed = True
            except OSError:
                pass
        return changed

    def _run_shell_post_checks(
        self,
        result: str,
        *,
        owner_snapshot: Dict[pathlib.Path, Optional[str]],
        state_drive_root: pathlib.Path,
        light_repo_before: Optional[Dict[str, Any]],
        workspace_refs_before: Optional[Dict[str, str]],
        tool_name: str = "run_command",
    ) -> str:
        import time

        restored_owner_state = False
        for _ in range(4):
            time.sleep(0.3)
            restored_owner_state = (
                self._restore_owner_files(owner_snapshot, state_drive_root)
                or restored_owner_state
            )
        if restored_owner_state:
            result = (
                f"{result}\n\n⚠️ OWNER_STATE_RESTORED: run_command attempted to "
                "change owner-only settings or skill trust state; protected files were restored."
            )
        if light_repo_before is not None:
            light_repo_after = _light_repo_snapshot(system_repo_dir_for(self._ctx))
            if (
                light_repo_after is not None
                and light_repo_after.get("digest") != light_repo_before.get("digest")
            ):
                result = _format_light_repo_write_block(light_repo_before, light_repo_after, result, tool_name=tool_name)
        if workspace_refs_before is not None:
            workspace_refs_after = _git_ref_snapshot(active_repo_dir_for(self._ctx))
            if (
                workspace_refs_after is not None
                and workspace_refs_after.get("digest") != workspace_refs_before.get("digest")
            ):
                result = (
                    "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs "
                    "inside the external workspace. External workspace runs must leave "
                    "changes as files/patch artifacts, not commits/tags/resets.\n\n"
                    "Original command output:\n"
                    f"{result}"
                )
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
        if local_readonly_subagent and entry is not None and name not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
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
        facts: Any = None,
    ) -> tuple[Dict[str, Any], Any, str]:
        """Resolve an exact python/python3 request ONCE, before the shell guard.

        Every downstream guard and the handler therefore see byte-identical
        argv; launchers must not select an interpreter after this boundary. The
        interpreter comes from the operation's PREPARE facts, so a non-local
        placement is answered by its target, never by a Home probe (RWS-02).
        """
        args, python_resolution = resolve_process_python(
            self._ctx,
            name,
            args,
            runtime_mode=runtime_mode,
            effective_constraint=effective_constraint,
            resolved_binding=resolved_binding,
            facts=facts,
        )
        record_python_resolution(self._ctx, python_resolution)
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
        python_resolution: Any,
        worktree_before: Any,
    ) -> tuple[str | None, Any]:
        """Run one builtin handler; returns (early_error_text, result).

        The launcher attestation lives exactly as long as the handler call:
        run_script consults it to accept the resolver-chosen interpreter.
        """
        missing = object()
        prior = getattr(self._ctx, "_active_python_resolution", missing)
        self._ctx._active_python_resolution = python_resolution
        try:
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
            except RemoteWorkspacePathError:
                # A PLACEMENT answer, not a tool error: it goes to the one
                # classifier in `execute` rather than being flattened into a generic
                # TOOL_ERROR with a Home-shaped sentence the model cannot act on.
                raise
            except TypeError as e:
                return f"⚠️ TOOL_ERROR ({name}): {e}", None
            except Exception as e:
                return f"⚠️ TOOL_ERROR ({name}): {e}", None
        finally:
            if prior is missing:
                try:
                    delattr(self._ctx, "_active_python_resolution")
                except AttributeError:
                    pass
            else:
                self._ctx._active_python_resolution = prior
            # Central advisory invalidation by OBSERVED worktree diff: runs on
            # success, tool error, and exception paths alike (the per-tool
            # manual calls missed early-return/error paths), and skips
            # invalidation when a flagged tool ran read-only.
            if worktree_before is not None:
                self._invalidate_advisory_if_worktree_changed(name, worktree_before)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch boundary (RWS v2 §3.1): run the whole pipeline inside ONE
        operation scope, so a prepared operation is always released.

        The catch is Appendix C-2's fourth classification: a consumer that asks a
        remote placement for a Home path gets a typed refusal at the seam, and if
        nothing above claimed it, that must still reach the model as a RESULT — an
        exception escaping the dispatch is a crash in the tool loop, not a policy
        answer. Local placements never raise it, so no Home bug can hide here.

        The `finally` is what makes the abort structural instead of a list of
        refusal sites to keep in sync with the pipeline: whatever the dispatch
        answered, and even if it raised, an operation the target prepared and Home
        did not execute is released here (see
        `dispatch_execute.withdraw_outstanding_prepare`)."""
        name = str(name or "").strip()
        args = dict(args or {})
        outstanding = OutstandingPrepare()
        try:
            return self._dispatch(name, args, outstanding)
        except RemoteWorkspacePathError as exc:
            return (
                f"⚠️ PLACEMENT_UNSUPPORTED_TOOL: {name!r} needs a Home path this task's "
                f"workspace does not have ({exc}). Use a tool that runs on the target "
                "(files, search, git, shell and services all do), or move the work to a "
                "task whose workspace is local."
            )
        finally:
            withdraw_outstanding_prepare(self._ctx, outstanding)

    def _placement_blind_gates(
        self, name: str, args: Dict[str, Any],
    ) -> "tuple[Optional[_DispatchGates], str]":
        """Everything decidable BEFORE we know which machine this runs on.

        Identity and contract gates, capability availability, the resource fences,
        workspace-metadata validation, the public-schema refusal and the two
        placement-blind refusals (`subagent_secret_path_refusal`,
        `script_interpreter_refusal`), then skill-repair and payload constraint.

        Not one of them ROUTES on the placement, and that is the whole reason they are
        a unit: a call this half refuses is refused identically on every host, so it
        must never reach `prepare_operation` and reserve a token on someone else's
        machine first. Returns ``(gates, "")`` to continue or ``(None, refusal)``.

        "Blind" is about the DECISION, not about the read. Two of these gates DO consult
        the sealed ref — path normalization, to pick which vocabulary a path argument is
        spelled in, and the workspace-mode predicate — and the flat claim that none of
        them reads it was false in a way a reader would have believed. What makes them
        blind is that neither answer can change WHERE the call executes. Every such read
        is enumerated, with the reason for each, in
        `test_dispatch_prepare.py::_DECLARED_PLACEMENT_READS`.
        """
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
            return None, _eph
        if name in _disabled_tools(self._ctx):
            return None, f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.disabled_tools withholds {name!r} for this task."
        available, unavailable_reason, unavailable_detail = _builtin_tool_availability(name, self._ctx)
        if not available:
            suffix = f" ({unavailable_detail})" if unavailable_detail else ""
            return None, f"⚠️ CAPABILITY_UNAVAILABLE: {name!r} is unavailable: {unavailable_reason}{suffix}."
        if name == "vlm_query" and str(args.get("image_url") or "").strip() and (
            not _resource_allowed(self._ctx, "web") or not _resource_allowed(self._ctx, "network")
        ):
            return None, "⚠️ RESOURCE_CONSTRAINT_BLOCKED: remote image_url for vlm_query requires allowed_resources.web/network."
        if name in _WEB_TOOLS and not _resource_allowed(self._ctx, "web"):
            return None, f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.web=false blocks {name!r}."
        if name == "vcs_pull_ff" and not _resource_allowed(self._ctx, "network"):
            return None, "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false blocks 'vcs_pull_ff'."
        if (is_mcp or ext_tool) and not _resource_allowed(self._ctx, "network"):
            return None, f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false blocks external tool {name!r}."
        _gate = self._subagent_and_update_gate(
            name, entry, ext_tool, is_mcp, local_readonly_subagent, acting_subagent, acting_tool_grants
        )
        if _gate:
            return None, _gate
        workspace_block_reason = ""
        try:
            workspace_block_reason = workspace_mode_block_reason(self._ctx)
        except Exception as exc:
            workspace_block_reason = f"workspace metadata validation failed: {type(exc).__name__}: {exc}"
        if workspace_block_reason:
            return None, (
                "⚠️ WORKSPACE_MODE_BLOCKED: invalid external workspace metadata: "
                f"{workspace_block_reason}. Workspace tasks must not overlap the "
                "Ouroboros repo, runtime data, or control plane."
            )
        if entry is not None:
            # The PUBLIC-SCHEMA refusal: ONE site for both routes, placement-BLIND,
            # and ahead of prepare so a malformed call never reserves a token on
            # another machine. It also canonicalizes argument aliases, which prepare
            # depends on — prepare tells the target which paths the operation is
            # about (rationale in `tools/tool_args`).
            public_arg_error = _prepare_public_builtin_args(entry, args)
            if public_arg_error:
                return None, public_arg_error
            _route_note = _normalize_dispatch_path_args(self._ctx, name, args)
            if _route_note.startswith("⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE"):
                return None, _route_note
            # The restricted-subagent secret/control denial (rationale in
            # `tools/dispatch_policy`): a PLACEMENT-BLIND decision — it answers the same
            # on every host, and it is taken before the placement is RESOLVED — because
            # the very same guard living inside the Home handler body was absent from
            # the native route entirely. (The line above it does read the sealed ref, to
            # choose a spelling space; see this method's docstring.)
            _secret_block = subagent_secret_path_refusal(self._ctx, name, args)
            if _secret_block:
                return None, _secret_block
            # Same class, same seam: `run_script`'s interpreter allowlist was a
            # handler-body rule, and the target takes the interpreter verbatim into
            # its argv.
            _interpreter_block = script_interpreter_refusal(name, args)
            if _interpreter_block:
                return None, _interpreter_block
        heal_no_enable = bool(task_constraint and task_constraint.mode == "skill_repair")
        if heal_no_enable:
            heal_block = self._heal_mode_block(name, args, task_constraint, ext_tool, is_mcp)
            if heal_block:
                return None, heal_block
        workspace_mode = bool(getattr(self._ctx, "is_workspace_mode", lambda: False)())
        effective_constraint = task_constraint
        if entry is not None:
            effective_constraint, payload_error = _payload_dispatch_constraint(
                self._ctx,
                name=name,
                args=args,
                task_constraint=task_constraint,
                workspace_mode=workspace_mode,
            )
            if payload_error:
                return None, payload_error
        # Fail-closed: an acting child WITHOUT a resolved isolated workspace would
        # have active_workspace/system_repo fall back to the LIVE repo. Confine it
        # to data roots and block shell/coding/service (whose default target is the repo).
        if acting_subagent and not workspace_mode:
            if name in _ROOT_ARG_REPO_WRITE_TOOLS and str(args.get("root", "") or "active_workspace") in ("active_workspace", "system_repo"):
                return None, (
                    "⚠️ ACTING_NO_WORKSPACE_BLOCKED: this acting subagent has no resolved isolated "
                    "workspace; write only to root=task_drive, root=artifact_store, or root=user_files. "
                    "active_workspace/system_repo map to the live Ouroboros repo and are blocked."
                )
            if name in ("run_command", "run_script", "start_service",
                        "integrate_subagent_patch", "integrate_delegated_patch"):
                return None, (
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
        return _DispatchGates(
            route_note=_route_note,
            task_constraint=task_constraint,
            entry=entry,
            ext_tool=ext_tool,
            is_mcp=is_mcp,
            acting_subagent=acting_subagent,
            acting_self_worktree=acting_self_worktree,
            acting_protected_grant=acting_protected_grant,
            workspace_mode=workspace_mode,
            effective_constraint=effective_constraint,
            runtime_mode=_runtime_mode,
        ), ""
    def _home_resolved_binding(
        self, name: str, args: Dict[str, Any], prepared: Any,
    ) -> "tuple[Any, str]":
        """The HOME path this operation names, or the typed reason there is none.

        Returns ``(binding, "")`` to continue or ``(None, refusal)``. A ``None`` binding
        with no refusal means the operation genuinely has no Home path — it runs on the
        target — and every guard below is written to read that.

        A resolved binding names a HOME path, and every guard keyed on it asks about
        Home. So the question this gate must ask is "does this operation address Home?"
        and NOT "is the task remote?" — a remote task calling against a HOME-native root
        (`system_repo`, `runtime_data`, `artifact_store`, the owner's files) keeps its
        Home handler by the ratified root matrix (`dispatch_prepare`) and needs the same
        single Home resolution a local task gets. Gating on the PLACEMENT skipped it for
        exactly those calls, and the `resolved_binding is None` fallbacks then read
        `workspace_mode` — true for any admitted remote task — so `runtime_mode=light`
        silently stopped protecting Home's own repo and data drive for every ssh task.
        One policy, two doors: the class this pipeline exists to close.

        `native_routed` is the honest question because it is the EXECUTE-phase fact —
        true only when the operation really runs on the target, where there is no Home
        path to resolve. It is read off the PREPARE, not from the sealed placement: the
        pipeline has exactly ONE placement-resolution site and this is not it. Asking
        `is_remote_workspace(ctx)` here would be a second door into the placement, and
        the whole point of the sealed read is that no guard gets its own. Pinned by
        `test_dispatch_prepare.py::test_the_dispatch_pipeline_has_exactly_one_placement_resolution_site`.
        """
        if prepared.native_routed or _target_binding_operation(name, args) is None:
            return None, ""
        try:
            return _build_builtin_target_binding(self._ctx, name, args), ""
        except RemoteWorkspacePathError:
            # The one honest "Home has no answer" case: a PROCESS cwd on a remote task
            # resolves on the machine. The tools whose cwd is target-native but which
            # the routing table gives no native counterpart — `verify_and_record`'s run
            # kinds — arrive here, and their equivalent target facts come from the
            # prepare. The binding stays None for this TYPED seam error only, never as a
            # blanket placement exemption.
            return None, ""
        except Exception as exc:
            redirect = _light_binding_failure_redirect(name, args)
            if redirect:
                return None, redirect
            operation = _target_binding_operation(name, args)
            if operation in {"shell", "service"}:
                return None, shell_cwd_block_message(
                    self._ctx,
                    str(args.get("cwd") or ""),
                    operation=operation,
                    error=exc,
                )
            return None, _binding_error_text(
                name, str(args.get("root") or "active_workspace"), exc,
            )

    def _dispatch(self, name: str, args: Dict[str, Any], outstanding: Any) -> str:
        gates, refusal = self._placement_blind_gates(name, args)
        if gates is None:
            return refusal
        _route_note = gates.route_note
        task_constraint = gates.task_constraint
        entry = gates.entry
        ext_tool = gates.ext_tool
        is_mcp = gates.is_mcp
        acting_subagent = gates.acting_subagent
        acting_self_worktree = gates.acting_self_worktree
        acting_protected_grant = gates.acting_protected_grant
        workspace_mode = gates.workspace_mode
        effective_constraint = gates.effective_constraint
        _runtime_mode = gates.runtime_mode

        if is_mcp:
            return self._dispatch_mcp_tool(name, args)
        if entry is None:
            if ext_tool and callable(ext_tool.get("handler")):
                return self._dispatch_extension_tool(name, ext_tool, args)
            return f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(self._entries.keys()))}"
        # PREPARE (§3.1 step 2) — the pipeline's ONE placement-resolution site. It
        # follows the placement-INDEPENDENT identity/contract gates above, whose
        # precedence is a fixed contract, and precedes every guard that judges a
        # path, a cwd or an interpreter.
        _prepared = prepare_operation(self._ctx, name, args, outstanding=outstanding)
        if not _prepared.available:
            return f"⚠️ REMOTE_EXECUTION_UNAVAILABLE: {_prepared.unavailable}"
        resolved_binding, binding_refusal = self._home_resolved_binding(name, args, _prepared)
        if binding_refusal:
            return binding_refusal
        # MEDIA IMPORT (D1: vision runs on Home, so the bytes come here first;
        # rationale in `remote_task_files`). Before every path-judging guard, so they
        # judge the Home path that will actually be opened.
        if _prepared.placement == "ssh" and name in MEDIA_PATH_ARGS:
            media_block = remote_media_predispatch(self._ctx, name, args)
            if media_block:
                return media_block
        args, python_resolution, python_block = self._resolve_python_predispatch(
            name, args, _runtime_mode, effective_constraint, resolved_binding,
            facts=_prepared.facts,
        )
        if python_block:
            return python_block
        # ARGV RECONCILIATION (§3.1 step 3 precondition; rationale in `dispatch_prepare`).
        _argv_note, _argv_block = reconcile_target_args(_prepared, args)
        if _argv_block:
            return _argv_block
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
        elif _prepared.native_routed:
            # The Ouroboros body lives on Home; a target-native operation cannot
            # reach it, so light mode has nothing to protect on this route.
            light_targets_system = False
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
            elif _prepared.native_routed:
                # Same reason as the light arm: the protected paths this guard knows
                # are Home's own, and no target-native write can address them.
                protected_target = False
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

        # `_launches_a_command` and not membership alone: `verify_and_record`'s
        # non-run contract kinds execute nothing, so judging their declared prose as
        # argv produced refusals about a command that does not exist.
        if name in _SHELL_GUARDED_TOOLS and _launches_a_command(name, args):
            if (
                name == "start_service"
                and _runtime_mode == "light"
                and (
                    _binding_set_targets_system_repo(self._ctx, resolved_binding)
                    or acting_self_worktree
                )
            ):
                return ("⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses start_service against the Ouroboros repository because long-running services can mutate after initial tool checks. For external services, set cwd under user_files, task_drive, or artifact_store; switch to advanced/pro only for reviewed Ouroboros self-modification.")
            # AUTHORIZE (§3.1 step 3) runs over the three NAMED projections the
            # prepared token is bound to, so the set that is authorized and the set
            # that is executed cannot be two look-alike objects one rename apart
            # (codex #2).
            _prepared = bind_execution_args(
                _prepared, self._ctx, args,
                runtime_mode=_runtime_mode, binding=resolved_binding,
            )
            block_msg = self._run_shell_safety_check(
                _prepared.projections.guard_args if _prepared.bound
                else process_shell_guard_args(name, args, ctx=self._ctx, runtime_mode=_runtime_mode),
                _runtime_mode,
                resolved_binding,
                native_shell_target(_prepared),
            )
            if block_msg:
                return block_msg
        elif _prepared.native_routed:
            # EVERY native operation leaves Home token-bound, not only the
            # shell-guarded ones: a remote read/write/vcs call crosses a wire too.
            # No local placement takes this branch, so the golden traces are intact.
            _prepared = bind_execution_args(
                _prepared, self._ctx, args,
                runtime_mode=_runtime_mode, binding=resolved_binding,
            )

        # LLM safety supervisor.
        from ouroboros.safety import check_safety
        is_safe, safety_msg = check_safety(
            name,
            args,
            messages=getattr(self._ctx, "messages", None),
            ctx=self._ctx,
            python_resolution=python_resolution,
        )
        if not is_safe:
            return safety_msg
        if _prepared.bound:
            # EXECUTE integrity (§3.1 step 4): the command about to run must still be
            # the one the guards authorized. Nothing between AUTHORIZE and here is
            # supposed to rewrite argv or cwd — and the prepared token is what makes
            # "supposed to" checkable instead of assumed. The remote target repeats
            # this same check on the token it receives.
            if not _prepared.binds(
                project_dispatch_args(
                    self._ctx, name, args, runtime_mode=_runtime_mode,
                    facts=_prepared.facts, target_cwd=native_execution_cwd(_prepared),
                    binding=resolved_binding,
                ).execution_args
            ):
                return (
                    "⚠️ PREPARED_CALL_BINDING_MISMATCH: the execution arguments changed "
                    "after authorization; the process was not started."
                )
        if _prepared.native_routed:
            # EXECUTE (§3.1 step 4). It lands HERE — after the whole guard pipeline
            # and the binding check — and REPLACES the local handler rather than
            # running beside it: that handler resolves `active_workspace` as a Home
            # path, so calling it for a remote operation would answer about the
            # Ouroboros checkout instead of the project the task was pointed at.
            # Nothing below applies either: the owner-file/light-repo/git-ref
            # snapshots and post-checks all guard HOME state, which a target-native
            # command cannot touch, and the target owns its own equivalents.
            return _compose_execute_result(
                filter_native_listing(
                    self._ctx,
                    name,
                    execute_native_operation(
                        self._ctx, _prepared, timeout_sec=getattr(entry, "timeout_sec", None), outstanding=outstanding,
                    ),
                ),
                "\n".join(part for part in (_route_note, _argv_note) if part),
                safety_msg,
            )
        state_drive_root = _binding_state_drive_root(self._ctx, resolved_binding)
        owner_snapshot = (
            self._snapshot_owner_files(state_drive_root)
            if name in _PROCESS_COMMAND_TOOLS else {}
        )
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
        worktree_before = (
            self._worktree_status_snapshot() if entry.mutates_worktree else None
        )
        early_error, result = self._invoke_builtin_handler(
            name, entry, args, resolved_binding, python_resolution, worktree_before,
        )
        if early_error is not None:
            return early_error
        if name in _PROCESS_COMMAND_TOOLS:
            result = self._run_shell_post_checks(
                result,
                owner_snapshot=owner_snapshot,
                state_drive_root=state_drive_root,
                light_repo_before=light_repo_before,
                workspace_refs_before=workspace_refs_before,
                tool_name=name,
            )

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
