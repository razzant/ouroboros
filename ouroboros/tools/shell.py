"""Process tools: run_command and run_script."""

from __future__ import annotations

import hashlib  # noqa: F401
from hashlib import sha256  # noqa: F401
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import tempfile
import signal  # noqa: F401
import stat  # noqa: F401
import subprocess
import threading  # noqa: F401
import time
import uuid
from typing import Dict, List  # noqa: F401

from ouroboros.owner_pause import OwnerPauseRefused
from ouroboros.tool_capabilities import RESULT_VIEW_PARAMS
from ouroboros.tools.tool_result import launch_refusal_result, _publish_tool_result
from ouroboros.artifacts import copy_directory_to_task_artifacts, copy_file_to_task_artifacts, record_task_scratch  # noqa: F401
from ouroboros.platform_layer import bootstrap_process_path, kill_process_tree, scrub_repo_from_pythonpath, subprocess_new_group_kwargs  # noqa: F401
from ouroboros.process_interpreters import (
    active_node_resolution,
    apply_env_path_prepend,
    interpreter_path_overlay,
)
from ouroboros.config import SETTINGS_DEFAULTS, load_settings  # noqa: F401
from ouroboros.runtime_mode_policy import (
    is_protected_runtime_path,  # noqa: F401
)
from ouroboros.tools.commit_gate import _invalidate_advisory
from ouroboros.shell_parse import POSIX_SHELL_HEADS, is_absolute_path_text, recover_stringified_argv  # noqa: F401
from ouroboros.tools.tool_result import _publish_process_result, _wrap_run_script_process_result
from ouroboros.tools.verify import check_exit_masking  # noqa: F401 -- ONE exit-masking sensor shared with verify_and_record (pinned here); its disclosure lives in shell_audit
from ouroboros.tools.registry import (
    ToolContext,
    ToolEntry,
)
from ouroboros.tools import shell_audit as _shell_audit
from ouroboros.tools.deliverables_shell import lexical_user_files_block_reason  # noqa: F401
from ouroboros.tools.shell_audit import (
    _UNDECLARED_OUTPUTS_MARKER,
    _disclose_output_audit_failure,
    _masked_green_disclosure,
    _mentioned_user_file_outputs_without_declaration,
    _presence_allows_user_output,  # noqa: F401
    _allowed_output_roots,  # noqa: F401
)
from ouroboros.tools.shell_process import (  # noqa: F401
    _RUN_SHELL_DEFAULT_TIMEOUT_SEC,
    _active_subprocesses,
    _describe_returncode,
    _executor_can_run_cwd,
    _format_process_output,
    _kill_process_group,
    _resolve_effective_timeout,
    _shell_env_for_cwd,
    _subprocess_lock,
    _tracked_subprocess_run,
    kill_all_tracked_subprocesses,
)
from ouroboros.tools.shell_effects import (  # noqa: F401
    _get_changed_files,
    _get_diff_stat,
    _protected_runtime_dirty_paths,
    _record_scratch_fingerprints,
    _resolve_git_root,
    _resolve_scratch_abs,
    _restore_protected_runtime_paths,
    _scratch_safety_reason,
    _shallow_listing,
    _status_snapshot,
    _tree_fingerprint,
    _user_files_run_had_effect,
)
from ouroboros.tools.shell_outputs import (  # noqa: F401
    _directory_fingerprint,
    _changed_path_covers,
    _directory_fingerprint_from_entries,
    _fingerprint_output,
    _protected_output_source_reason,
    _register_process_outputs,
    _resolve_declared_output,
    _scan_directory_output_members,
    _sensitive_output_component_reason,
    _snapshot_declared_outputs,
)

# Preserve private module attributes used by existing callers/tests while the
# implementation lives in the extracted audit module.
_EMBEDDED_OUTPUT_PATH_RE = _shell_audit._EMBEDDED_OUTPUT_PATH_RE
_OUTPUT_CALL_PATH_RE = _shell_audit._OUTPUT_CALL_PATH_RE
_OUTPUT_REDIRECT_PATH_RE = _shell_audit._OUTPUT_REDIRECT_PATH_RE
_OUTPUT_STAT_SLACK_SEC = _shell_audit._OUTPUT_STAT_SLACK_SEC
_USER_FILE_OPEN_WRITE_CALL_RE = _shell_audit._USER_FILE_OPEN_WRITE_CALL_RE
_USER_FILE_REDIRECT_RE = _shell_audit._USER_FILE_REDIRECT_RE
_USER_FILE_WRITE_CALL_RE = _shell_audit._USER_FILE_WRITE_CALL_RE
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    _deliverables_root_lexical,  # noqa: F401
    _deliverables_root_lexical_alias,  # noqa: F401
    _lexical_path_is_relative_to_casefold,  # noqa: F401
    _path_is_relative_to_casefold,  # noqa: F401
    build_resolved_resource_binding,
    path_is_relative_to,  # noqa: F401
    resource_root_path,  # noqa: F401
    shell_cwd_block_message,
    user_files_path_block_reason,  # noqa: F401
)
from ouroboros.utils import safe_relpath  # noqa: F401
from ouroboros.deadline_utils import deadline_remaining_sec  # noqa: F401
from ouroboros.workspace_executor import execute as executor_execute
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_backend_path as executor_map_backend_path  # noqa: F401
from ouroboros.workspace_executor import map_backend_path_lexical as executor_map_backend_path_lexical  # noqa: F401
from ouroboros.workspace_executor import map_host_path as executor_map_host_path

log = logging.getLogger(__name__)
# Tracked process groups let panic kill descendant trees too.
_CONTROL_DIR_BACKUP_MAX_BYTES = 5 * 1024 * 1024
# Typed process-facts channel (R5) seam: ouroboros/tools/process_facts.py.
# Historical private spellings stay as aliases for call sites and tests.
from ouroboros.tools.process_facts import (  # noqa: E402
    active_resolved_runtime as _active_resolved_runtime,
    process_environment_tool, record_runtime_selection, selected_process_environment,
    publish_process_facts as _publish_process_facts,  # noqa: F401 — historical private spelling for call sites and tests
)
from ouroboros.tools.shell_process import (  # noqa: E402
    _publish_finished_process_facts,
    _publish_unfinished_process_facts,
)



# v6.90.x (submarine unwind) — the DECLARATION-NUDGE marker, deliberately typed
# APART from the real ``ARTIFACT_OUTPUT_ERROR`` registration failure above. The
# command SUCCEEDED (exit_code=0) and this only asks for ``outputs=[...]`` to be
# declared, so its status lands in the v6.57.0 POLICY-DENIAL partition
# (``_outcome_tool_errors._POLICY_DENIAL_STATUSES``) instead of degrading execution
# to ``tool_failure``. The submarine wave-3 incident was exactly this: a moot nudge
# on an already-registered artifact fed the failure record. SSOT for both
# ``run_command`` and ``run_script`` so the two nudges cannot drift apart.


_SHELL_BUILTINS = frozenset([
    "cd", "source", ".", "export", "alias", "eval",
    "set", "unset", "pushd", "popd", "read", "ulimit",
])

_SHELL_OPERATORS = frozenset(["&&", "||", "|", ";", ">", ">>", "<", "<<"])
# A redirect GLUED into a single argv element ("2>/dev/null", "2>&1", ">out.log",
# "&>x") — the standalone-operator set above misses these. Anchored at the element
# START so a '>' inside a sed/awk/grep expression ("s/a>b/c/g") is NOT flagged.
# Output redirects keep a permissive glued tail. Input redirects are restricted to
# UNAMBIGUOUS shapes — heredoc/herestring ("<<EOF", "<<<s"), an fd-prefixed input
# ("0<f", "2<&1"), or a bare standalone "<" — because a plain "<word" element is
# indistinguishable from a legitimate literal angle-bracket arg (grep "<div>",
# "<stdin>"), and false-flagging those is worse than missing a rare glued "<file"
# input redirect. Pipes/control operators are deliberately NOT matched (a glued
# '|' is valid regex alternation, grep "a|b").
_GLUED_REDIRECT_RE = re.compile(
    r'^(?:(?:\d+>>?|>>?&?\d*|\d*>&\d*|&>>?)(?:\S.*)?|\d+<\S*|<<\S*|<)$'
)
_SHELL_INTERPRETERS = POSIX_SHELL_HEADS | frozenset({"fish", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"})
_ENV_REF_PATTERN = re.compile(r'\$(?:\{[A-Z][A-Z0-9_]*\}|[A-Z][A-Z0-9_]*)')


# Portable grep fix: GNU basic-regex "\|" fails on BSD grep in argv mode.
_GREP_TOOLS = frozenset(("grep", "egrep", "fgrep"))
_GREP_REGEX_MODE_FLAGS = frozenset((
    "-E", "--extended-regexp",
    "-P", "--perl-regexp",
    "-F", "--fixed-strings",
    "-G", "--basic-regexp",
))
_GREP_BACKSLASH_PIPE_PATTERN = re.compile(r'\\\|')
_NO_MATCH_EXIT_TOOLS = frozenset(("grep", "egrep", "fgrep", "rg", "ag", "ack"))


def _is_search_no_match(res: subprocess.CompletedProcess) -> bool:
    tool = pathlib.Path(str(res.args[0] if res.args else "")).name.lower()
    return (
        int(res.returncode) == 1
        and tool in _NO_MATCH_EXIT_TOOLS
        and not str(res.stderr or "").strip()
    )


def _grep_has_explicit_regex_mode(cmd: List[str]) -> bool:
    """Return whether grep argv already chooses regex/string flavor."""
    if not cmd:
        return False
    tool = pathlib.Path(cmd[0]).name.lower()
    if tool in ("egrep", "fgrep"):
        return True
    for arg in cmd[1:]:
        if not isinstance(arg, str):
            continue
        if arg in _GREP_REGEX_MODE_FLAGS:
            return True
        if arg.startswith("--"):
            continue
        # Short options may be clustered, e.g. `grep -rnE pattern path`.
        if arg.startswith("-") and any(flag in arg[1:] for flag in ("E", "P", "F", "G")):
            return True
    return False


def _maybe_autocorrect_grep_backslash_pipe(cmd: List[str]) -> tuple[List[str], str]:
    if not cmd or pathlib.Path(cmd[0]).name.lower() not in _GREP_TOOLS:
        return cmd, ""
    if _grep_has_explicit_regex_mode(cmd):
        return cmd, ""
    corrected = list(cmd)
    changed_args: list[str] = []
    for idx, arg in enumerate(corrected[1:], start=1):
        if isinstance(arg, str) and _GREP_BACKSLASH_PIPE_PATTERN.search(arg):
            corrected[idx] = _GREP_BACKSLASH_PIPE_PATTERN.sub("|", arg)
            changed_args.append(arg)
    if not changed_args:
        return cmd, ""
    corrected.insert(1, "-E")
    return corrected, (
        "⚠️ SHELL_REGEX_AUTO_CORRECTED: converted grep backslash-escaped "
        "alternation (\\|) to extended regex mode (`grep -E`) and rewrote "
        f"{changed_args!r} to use `|`.\n"
    )


def _literal_argv_notes(cmd: List[str]) -> str:
    """Disclosure notes for shell-syntax-looking bytes in direct argv (#447 A5).

    A commit message naming ``$HOME``, an awk ``|`` field separator, a
    ``2>/dev/null`` element — no shell runs for direct argv, so these are
    LITERAL DATA carrying no authority question. They used to be REFUSED as
    errors, blocking commands that would have worked; the in-file autocorrect
    precedent applies instead: run the command and DISCLOSE what was passed
    literally, so a genuinely mistaken spelling still explains its own cryptic
    program error.
    """
    def _note(subject: str, remedy: str) -> str:
        return (
            f"⚠️ SHELL_LITERAL_ARGV_NOTE: {subject} in the cmd array reached the "
            "program as LITERAL data (run_command executes argv directly; "
            f"subprocess interprets no shell syntax). {remedy}\n"
        )

    notes: list[str] = []
    if (pathlib.Path(cmd[0]).name.lower() if cmd else "") not in _SHELL_INTERPRETERS:
        env_ref = next(filter(None, (_ENV_REF_PATTERN.search(arg) for arg in cmd)), None)
        if env_ref:
            notes.append(_note(
                f'literal env reference "{env_ref.group(0)}"',
                'Use ["sh", "-c", "..."] if you intended shell expansion.'))
    if found_ops := _SHELL_OPERATORS.intersection(cmd):
        notes.append(_note(
            f'shell operator "{sorted(found_ops)[0]}"',
            'Use ["sh", "-c", "cmd1 && cmd2"] for pipes/chaining.'))
    # Glued redirects bypass the standalone-operator set but remain shell-looking.
    glued = next((arg for arg in cmd if _GLUED_REDIRECT_RE.match(arg)), "")
    if glued:
        notes.append(_note(
            f'redirect-looking argument "{glued}"',
            'Use ["sh", "-c", "..."] for real redirection.'))
    return "".join(notes)


def _pre_spawn_refusal(ctx: ToolContext, text: str, *, tool: str = "run_command") -> str:
    """This producer refused before submission; arbitrary errors carry no such fact."""
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, _replace_tool_result

    result = LegacyTextResultAdapter.from_text(tool, text)
    return _publish_tool_result(ctx, _replace_tool_result(
        result, meta_updates={"operation_outcome": "completed_no_effect"}))


@process_environment_tool
def _run_shell(
    ctx: ToolContext,
    cmd,
    cwd: str = "",
    outputs: List[str] | None = None,
    scratch: List[str] | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
    **kwargs,
) -> str:
    # Per-call timeout override (canonical timeout_sec; timeout accepted as alias).
    timeout_sec = kwargs.get("timeout_sec")
    timeout = kwargs.get("timeout")
    _timeout_override = timeout_sec if timeout_sec is not None else timeout
    bucket = str(kwargs.get("bucket") or "")
    skill_name = str(kwargs.get("skill_name") or "")
    if isinstance(cmd, str):
        # Shared recovery keeps run_command and verify argv semantics aligned.
        recovered = recover_stringified_argv(cmd)
        # Malformed structured literals are not shell commands; refuse explicitly.
        if recovered is None:
            stripped = cmd.lstrip()
            is_posix_test_cmd = stripped.startswith("[ ") and stripped.rstrip().endswith(" ]")
            # A `{ ...; }` brace group is valid shell, not malformed JSON.
            is_brace_group = stripped.startswith("{ ") and stripped.rstrip().endswith("}")
            if is_brace_group:
                return _pre_spawn_refusal(ctx, '⚠️ SHELL_CMD_ERROR: `{ ...; }` is a shell brace group, which run_command '
                    'cannot execute directly (it runs argv without a shell). Wrap it in a shell:\n'
                    '  run_command(cmd=["sh", "-c", "{ cmd1; cmd2; }"])')
            if stripped[:1] in ("[", "{") and not is_posix_test_cmd:
                return _pre_spawn_refusal(ctx, '⚠️ SHELL_ARG_ERROR: `cmd` looks like a JSON/Python list literal '
                    'but failed to parse cleanly (likely an escape or quote-mismatch '
                    'issue). Pass cmd as an actual array, not a stringified array.\n\n'
                    'Correct usage:\n'
                    '  run_command(cmd=["git", "log", "--oneline", "-10"])\n\n'
                    'Wrong usage (the failure that brought you here):\n'
                    '  run_command(cmd=\'["git", "log", "--oneline", "-10"]\')\n\n'
                    'For reading files, prefer `read_file`.\n'
                    'For searching code, prefer `search_code`.')
            try:
                parts = shlex.split(cmd)
                if parts:
                    recovered = parts
            except ValueError:
                pass
        if recovered is not None:
            cmd = recovered
        else:
            return _pre_spawn_refusal(ctx, '⚠️ SHELL_ARG_ERROR: `cmd` must be a JSON array of strings, not a plain string.\n\n'
                'Correct usage:\n'
                '  run_command(cmd=["grep", "-r", "pattern", "path/"])\n'
                '  run_command(cmd=["python", "-c", "print(1+1)"])\n\n'
                'Wrong usage:\n'
                '  run_command(cmd="grep -r pattern path/")\n\n'
                'For reading files, prefer `read_file`.\n'
                'For searching code, prefer `search_code`.')

    if not isinstance(cmd, list):
        return _pre_spawn_refusal(ctx, "⚠️ SHELL_ARG_ERROR: cmd must be a list of strings.")
    cmd = [str(x) for x in cmd]
    if not cmd:
        return _pre_spawn_refusal(ctx, "⚠️ SHELL_ARG_ERROR: cmd must not be empty.")

    if cmd and cmd[0] in _SHELL_BUILTINS:
        if cmd[0] == "cd":
            return _pre_spawn_refusal(ctx, '⚠️ SHELL_CMD_ERROR: "cd" is a shell builtin, not an executable. '
                'Use the "cwd" parameter instead: '
                'run_command(cmd=["git", "log"], cwd="/target/dir")')
        return _pre_spawn_refusal(ctx, f'⚠️ SHELL_CMD_ERROR: "{cmd[0]}" is a shell builtin and cannot '
            'be executed directly via subprocess. '
            'Use ["sh", "-c", "your command"] if you need shell builtins.')

    cmd, autocorrect_note = _maybe_autocorrect_grep_backslash_pipe(cmd)
    regex_autocorrected = bool(autocorrect_note)
    autocorrect_note += _literal_argv_notes(cmd)

    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, operation="shell", process_cwd=cwd, bucket=bucket, skill_name=skill_name,
        )
        work_dir = pathlib.Path(binding.target_path)
        cwd_root = binding.root
    except (OSError, ValueError) as exc:
        return _pre_spawn_refusal(ctx, shell_cwd_block_message(ctx, cwd, operation="shell", error=exc))
    if not work_dir.exists() or not work_dir.is_dir():
        return _pre_spawn_refusal(ctx, f"⚠️ SHELL_CWD_BLOCKED: cwd is not a directory: {work_dir}. "
            f"root={binding.root}, source={binding.source}.")
    # Disclose the room-lens default once; explicit cwd is already caller-visible.
    if not str(cwd or "").strip() and not getattr(ctx, "_room_cwd_noted", False):
        try:
            from ouroboros.tool_access import project_room_lens_dir

            _room = project_room_lens_dir(ctx)
        except Exception:
            _room = None
        if _room is not None and pathlib.Path(work_dir).resolve(strict=False) == _room:
            ctx._room_cwd_noted = True
            autocorrect_note += (
                f"[project-room cwd: this command ran in {_room} (the room's folder). "
                'The Ouroboros system repo needs an explicit cwd.]\n\n'
            )
    repo_root = _resolve_git_root(pathlib.Path(work_dir))
    before_changed = _status_snapshot(repo_root)
    # Bounded snapshot makes the user_files artifact nudge effect-based.
    before_listing = (
        _shallow_listing(pathlib.Path(work_dir))
        if (cwd_root == "user_files" and repo_root is None and not outputs)
        else None
    )
    before_outputs = _snapshot_declared_outputs(
        ctx,
        outputs,
        pathlib.Path(work_dir),
        cwd_root=cwd_root,
        changed_paths=set(before_changed or []),
        binding=binding,
    )

    # Scratch is confined/new/untracked, patch-excluded, and never an artifact.
    scratch_abs = _resolve_scratch_abs(scratch, work_dir)
    if scratch_abs:
        _scratch_reason = _scratch_safety_reason(ctx, scratch_abs, pathlib.Path(work_dir), repo_root)
        if _scratch_reason:
            return _pre_spawn_refusal(ctx, f"⚠️ SCRATCH_BLOCKED: {_scratch_reason}.")
    timeout_sec = _resolve_effective_timeout(_RUN_SHELL_DEFAULT_TIMEOUT_SEC, ctx, override_sec=_timeout_override)
    bootstrap_process_path()
    # Emergency bundled-node PATH prepend; None on every healthy path (env stays byte-identical).
    node_resolution = active_node_resolution(ctx)
    from ouroboros.workspace_executor import overlay_env
    selected_env = selected_process_environment()
    run_env = apply_env_path_prepend(overlay_env(_shell_env_for_cwd(ctx, pathlib.Path(work_dir)), selected_env), node_resolution)
    record_runtime_selection(ctx, cmd, work_dir, run_env)
    # Two clocks (D2-1): EPOCH feeds the st_mtime audit; MONOTONIC feeds durations.
    _command_start_epoch = time.time()
    _command_start_ts = time.monotonic()
    res = None
    try:
        if _executor_can_run_cwd(ctx, pathlib.Path(work_dir)):
            res = executor_execute(ctx, cmd, pathlib.Path(work_dir), timeout_sec,
                                   env_overlay=interpreter_path_overlay(node_resolution),
                                   **({"target_env": selected_env} if selected_env else {}))
            process_meta = {"operation_outcome": getattr(res, "operation_outcome", "unknown")}
        else:
            res = _tracked_subprocess_run(
                cmd, cwd=str(work_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_sec,
                **({"env": run_env} if run_env is not None else {}),
            )
            process_meta = {"operation_outcome": "completed"}
        _lived_ms = _publish_finished_process_facts(ctx, res, _command_start_ts)
        # Post-run hashes exclude scratch only while its exact bytes still match.
        _record_scratch_fingerprints(ctx, scratch_abs)
        if res.returncode != 0:
            executor_note = ""
            if getattr(res, "backend_trace", None):
                executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
            if _is_search_no_match(res):
                text = autocorrect_note + (
                    f"{_describe_returncode(res.returncode, cwd=work_dir, binding=binding)} (no matches)\n"
                    f"{_format_process_output(res.stdout or '', '')}"
                    f"{executor_note}"
                )
                return _publish_process_result(ctx, "SHELL_NO_MATCH", text, exit_code=res.returncode, shell_regex_auto_corrected=regex_autocorrected, meta=process_meta)
            text = autocorrect_note + f"⚠️ SHELL_EXIT_ERROR: command exited with {_describe_returncode(res.returncode, cwd=work_dir, binding=binding, lived_ms=_lived_ms, resolved_runtime=_active_resolved_runtime(ctx))}.\n\n{_format_process_output(res.stdout or '', res.stderr or '')}{executor_note}"
            return _publish_process_result(ctx, "SHELL_EXIT_ERROR", text, exit_code=res.returncode, shell_regex_auto_corrected=regex_autocorrected, meta=process_meta)
        after_changed = _status_snapshot(repo_root)
        if after_changed != before_changed:
            # This resolved cwd may be outside the live-repo dispatcher snapshot.
            _invalidate_advisory(
                ctx,
                changed_paths=after_changed or before_changed,
                mutation_root=repo_root,
                source_tool="run_command",
            )
        audit_error = ""
        try:
            undeclared_user_outputs = _mentioned_user_file_outputs_without_declaration(
                ctx, cmd, outputs, scratch_abs=scratch_abs,
                command_start_ts=_command_start_epoch, cwd=work_dir,
            )
        except Exception as exc:
            # The child already finished. This observational failure must not
            # reach the spawn-error handler and overwrite its measured facts.
            undeclared_user_outputs, audit_error = [], type(exc).__name__
        if undeclared_user_outputs:
            # Declaration NUDGE, not a failure — see _UNDECLARED_OUTPUTS_MARKER.
            text = (
                autocorrect_note
                + f"{_UNDECLARED_OUTPUTS_MARKER}: command appears to write user_files outputs "
                "without declaring outputs=[...]. Declare generated user-visible files so "
                "they are copied into the task artifact store before claiming completion. "
                f"Paths: {', '.join(undeclared_user_outputs[:5])}.\n\n"
                + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n"
                + _format_process_output(res.stdout or "", res.stderr or "")
            )
            return _masked_green_disclosure(ctx, _publish_process_result(ctx, "ARTIFACT_OUTPUT_UNDECLARED", text, exit_code=0, shell_regex_auto_corrected=regex_autocorrected, meta=process_meta), cmd)
        artifact_note, artifact_failed, artifact_registered = _register_process_outputs(
            ctx,
            outputs,
            pathlib.Path(work_dir),
            cwd_root=cwd_root,
            changed_paths=set(after_changed or []),
            before_outputs=before_outputs,
            binding=binding,
        )
        audit_note = ""
        if cwd_root == "user_files" and not outputs:
            # Remove scratch effects without hiding a simultaneous real deliverable.
            _after_for_audit = after_changed
            if scratch_abs and repo_root is not None:
                _repo = pathlib.Path(repo_root).resolve(strict=False)
                _scratch_rel: set[str] = set()
                for _sp in scratch_abs:
                    try:
                        _scratch_rel.add(_sp.resolve(strict=False).relative_to(_repo).as_posix())
                    except ValueError:
                        continue
                if _scratch_rel:
                    _after_for_audit = [p for p in (after_changed or []) if p not in _scratch_rel]
            if _user_files_run_had_effect(before_changed, _after_for_audit, before_listing, pathlib.Path(work_dir)):
                audit_note = (
                    "\n\n⚠️ ARTIFACT_AUDIT_GAP: command modified files in a user_files cwd without "
                    "outputs=[...]. If it created a deliverable, rerun/register the file "
                    "with outputs or write_file(root=artifact_store) before claiming it."
                )
        scratch_note = ""
        _scratch_remaining = [str(p) for p in scratch_abs if p.exists()]
        if _scratch_remaining:
            scratch_note = (
                "\n\n⚠️ SCRATCH_REMAINS: declared scratch still on disk after the command: "
                + ", ".join(_scratch_remaining[:5])
                + ". It is excluded from the workspace patch, but delete it before finishing so it does not linger."
            )
        if artifact_failed:
            text = (
                autocorrect_note
                + "⚠️ ARTIFACT_OUTPUT_ERROR: command succeeded but declared output registration failed. "
                + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n"
                + f"{_format_process_output(res.stdout or '', res.stderr or '')}"
                + artifact_note
            )
            result = _masked_green_disclosure(ctx, _publish_process_result(ctx, "ARTIFACT_OUTPUT_ERROR", text, exit_code=0, shell_regex_auto_corrected=regex_autocorrected, meta=process_meta), cmd)
            return _disclose_output_audit_failure(ctx, result, audit_error)
        executor_note = ""
        if getattr(res, "backend_trace", None):
            executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
        text = autocorrect_note + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n{_format_process_output(res.stdout or '', res.stderr or '')}{artifact_note}{audit_note}{scratch_note}{executor_note}"
        result = _masked_green_disclosure(ctx, _publish_process_result(ctx, "SHELL_REGEX_AUTO_CORRECTED" if regex_autocorrected else "OK", text, exit_code=0, artifact_registered=bool(artifact_registered and not artifact_failed), shell_regex_auto_corrected=regex_autocorrected, meta=process_meta), cmd)
        return _disclose_output_audit_failure(ctx, result, audit_error)
    except OwnerPauseRefused as exc:
        return _publish_tool_result(ctx, launch_refusal_result(str(exc), completed_no_effect=True))
    except subprocess.TimeoutExpired:
        _publish_unfinished_process_facts(ctx, _command_start_ts, timed_out=True)
        # Timeout-created scratch still needs its exclusion fingerprint.
        _record_scratch_fingerprints(ctx, scratch_abs)
        return (
            f"⚠️ TOOL_TIMEOUT (run_command): command exceeded the per-command timeout of {timeout_sec}s "
            f"and the host killed what it could still reach of its process tree; a background child may survive untracked (root={binding.root}, cwd={work_dir}). NOTE: this is the per-command "
            f"FOREGROUND timeout, NOT the task deadline. For genuinely long-running compute (training, "
            f"sampling, large builds/downloads), start it with start_service and poll "
            f"service_status/service_logs while you do other work, or pass an explicit timeout_sec=<seconds> "
            f"(up to the per-call ceiling) — and preserve a best-effort deliverable before the task deadline."
        )
    except Exception as e:
        if res is None:
            _publish_unfinished_process_facts(ctx, _command_start_ts, spawn_error=e)
        _record_scratch_fingerprints(ctx, scratch_abs)
        if res is None and getattr(e, "process_not_started", False) is True and isinstance(e, FileNotFoundError) and len(cmd) == 1:
            return _pre_spawn_refusal(ctx,
                "⚠️ SHELL_ARG_ERROR: the sole cmd element was treated as ONE executable name, "
                "and that executable was not found. Pass the program and each argument as "
                'separate array elements, e.g. ["git", "status", "--porcelain"]. For pipes, '
                'redirects or chaining, explicitly use ["sh", "-c", "..."] or run_script. '
                f"No command was started. root={binding.root}, cwd={work_dir}"
            )
        text = f"⚠️ SHELL_ERROR: {e}. root={binding.root}, cwd={work_dir}"
        return (_pre_spawn_refusal(ctx, text) if res is None and getattr(e, "process_not_started", False) is True
                else text)


@process_environment_tool
def _run_script(
    ctx: ToolContext,
    script: str,
    interpreter: str = "python3",
    args: List[str] | None = None,
    cwd: str = "",
    outputs: List[str] | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
    **kwargs,
) -> str:
    """Stage a temporary script and run it with one resolved process binding.

    Optional public fields ride in ``kwargs`` to keep the handler within the
    DEVELOPMENT parameter limit; dispatch validates them against the schema.
    """
    timeout_sec = kwargs.get("timeout_sec")
    timeout = kwargs.get("timeout")
    scratch = kwargs.get("scratch")
    bucket = str(kwargs.get("bucket") or "")
    skill_name = str(kwargs.get("skill_name") or "")
    interp = str(interpreter or "python3").strip()
    body = str(script or "")
    if not body.strip():
        return _pre_spawn_refusal(ctx, "⚠️ TOOL_ARG_ERROR (run_script): script is required.", tool="run_script")
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, operation="shell", process_cwd=cwd, bucket=bucket, skill_name=skill_name,
        )
    except (OSError, ValueError) as exc:
        return _pre_spawn_refusal(ctx, shell_cwd_block_message(ctx, cwd, operation="shell", error=exc), tool="run_script")
    # The undeclared-output audit of the script BODY (argv only carries the temp script path, so
    # _run_shell cannot see the body) is POST-exec (v6.56.0): the stat filter needs the files to
    # exist, and a pre-exec scan on not-yet-written paths would either be a no-op or false-flag
    # import strings. We resolve the body-audit scratch against the SAME effective cwd the script
    # executes in so a relatively-declared scratch path matches a user_files write in the body.
    resolved_workdir = pathlib.Path(binding.target_path)
    if not resolved_workdir.is_dir():
        return _pre_spawn_refusal(ctx, f"⚠️ SHELL_CWD_BLOCKED: cwd is not a directory: {resolved_workdir}.",
                                  tool="run_script")
    _scratch_abs_body = _resolve_scratch_abs(scratch, resolved_workdir)
    _body_start_epoch = time.time()  # st_mtime audit; monotonic below is for durations
    _body_start_ts = time.monotonic()
    executor_active = _executor_can_run_cwd(ctx, resolved_workdir)
    active_workspace_script = binding.root == "active_workspace"
    if active_workspace_script:
        # Workspace scripts stage in an owned per-run dir under <cwd>/.ouroboros/tmp_scripts: the run's
        # own .gitignore keeps it out of raw Git status, and workspace_patch_rules._TOP_LEVEL_EXCLUDE_DIRS
        # keeps ".ouroboros" out of patches. Task-drive runs stage under <task_drive>/tmp_scripts.
        root = resolved_workdir / ".ouroboros" / "tmp_scripts"
    else:
        try:
            root = pathlib.Path(ctx.task_drive_root()) / "tmp_scripts"
        except Exception:
            root = pathlib.Path(ctx.drive_root) / "tmp_scripts"
    suffix = ".py" if "python" in pathlib.PurePath(interp).name else ".sh"
    run_dir = None
    script_path = root / f"script_{uuid.uuid4().hex}{suffix}"
    shell_entered = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        if active_workspace_script:
            run_dir = pathlib.Path(tempfile.mkdtemp(prefix="script_", dir=root))
            # Ignore only this invocation's files, not neighbouring user work.
            (run_dir / ".gitignore").write_text("*\n", encoding="utf-8")
            script_path = run_dir / f"script{suffix}"
        script_path.write_text(body, encoding="utf-8")
        try:
            os.chmod(script_path, 0o600)
        except OSError:
            pass
        script_arg = str(script_path)
        if executor_active:
            executor = executor_ref_from_ctx(ctx)
            if executor is not None and executor.kind != "local":
                try:
                    script_arg = executor_map_host_path(executor, script_path)
                except Exception as exc:
                    return _pre_spawn_refusal(ctx, f"⚠️ RUN_SCRIPT_BLOCKED: executor-backed run_script could not map temp script path: {type(exc).__name__}: {exc}", tool="run_script")
        argv = [interp, script_arg, *[str(item) for item in (args or [])]]
        shell_entered = True
        result = _run_shell(
            ctx, argv, cwd=cwd, outputs=outputs, scratch=scratch,
            _resolved_binding=binding, timeout_sec=timeout_sec, timeout=timeout,
            env_from_settings=kwargs.get("env_from_settings"),
        )
    except (OSError, ValueError) as exc:
        if shell_entered:
            raise
        return _pre_spawn_refusal(ctx, f"⚠️ RUN_SCRIPT_BLOCKED: script preparation failed: {exc}", tool="run_script")
    finally:
        try:
            if run_dir is not None:
                shutil.rmtree(run_dir)
            else:
                script_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not remove run_script scratch %s (%s)", run_dir or script_path, type(exc).__name__)
        # These shared parents may contain another run or a user's file.
        for parent in (root, root.parent) if active_workspace_script else (root,):
            try:
                parent.rmdir()
            except OSError:
                pass
    if pathlib.PurePath(interp).name in {"sh", "bash"}:
        result = _masked_green_disclosure(ctx, result, [interp, "-c", body])
    # POST-exec body audit: stat-confirmed user_files writes performed by the script
    # body itself. Runs on EVERY exit path (parity with _record_scratch_fingerprints):
    # a script that writes an undeclared deliverable and then FAILS (raise/SystemExit/
    # timeout) still leaves that file on disk, so a `⚠️` result does NOT mean "no
    # deliverable to declare" — surface both the error and the output-guard note.
    audit_error = ""
    try:
        undeclared_user_outputs = _mentioned_user_file_outputs_without_declaration(
            ctx, [interp, "-c", body], outputs, scratch_abs=_scratch_abs_body,
            command_start_ts=_body_start_epoch, cwd=resolved_workdir,
        )
    except Exception as exc:
        undeclared_user_outputs, audit_error = [], type(exc).__name__
    audit_note = ""
    if undeclared_user_outputs:
        # Same declaration NUDGE class as run_command's — see _UNDECLARED_OUTPUTS_MARKER.
        audit_note = (
            f"{_UNDECLARED_OUTPUTS_MARKER}: run_script wrote user_files without declaring outputs: "
            + ", ".join(undeclared_user_outputs)
            + ". Re-run with outputs=[...] or write the canonical deliverable via root=artifact_store."
        )
    result = _wrap_run_script_process_result(ctx, result, audit_note, script_path)
    return _disclose_output_audit_failure(ctx, result, audit_error)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("run_command", {
            "name": "run_command",
            "description": (
                "Run a bounded foreground command in an allowed cwd; returns stdout+stderr and the resolved cwd. "
                "cmd MUST be an array of strings, never one shell string. A builtin as cmd[0] (cd, export, ...) "
                "is refused: use cwd= or [\"sh\", \"-c\", \"cd x && a | b\"] (also for pipes/chaining). "
                "A background child (&, nohup) is no service: holding stdout/stderr it stalls the call until "
                "timeout_sec, and once the call returns nothing tracks it (use start_service). "
                "Prefer read_file, search_code/query_code, write_file/edit_text to cat/head/sed, grep/find, redirects."
            ),
            "parameters": {"type": "object", "properties": {
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Argv as a JSON array of strings. Example: "
                        "[\"git\", \"log\", \"--oneline\", \"-10\"]. NEVER "
                        "pass a single string like \"git log\" or a "
                        "stringified array like '[\"git\", \"log\"]'."
                    ),
                },
	                "cwd": {"type": "string", "default": "", "description": "Omit for active_workspace; use system_repo[/subdir] for Ouroboros or skill_payload[/subdir] with bucket+skill_name for a skill. task_drive, artifact_store, user_files and authorized absolute cwd forms are also accepted."},
	                "env_from_settings": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Explicit environment variable → saved setting key mapping. Uses existing Settings-selection authority; secret values are masked in diagnostics."},
	                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "user_repo"], "description": "Physical skill location for cwd=skill_payload[/subdir]."},
	                "skill_name": {"type": "string", "description": "Exact skill identity for cwd=skill_payload[/subdir]."},
	                "outputs": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": "Generated paths copied/registered after success, e.g. outputs=['report.txt']. The managed artifact directory is created lazily; do not move files into an assumed physical store.",
	                },
	                "scratch": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": (
	                        "Transient in-repo verification files (e.g. a throwaway test you write, run, and "
	                        "delete to check your own work) — throwaway verification ONLY, never part of the "
	                        "solution. Each must be untracked and confined to the cwd: a NEW file, or an existing "
	                        "untracked file created earlier in THIS task (adopted by sha, so re-declaring is "
	                        "idempotent); tracked files and directories stay blocked. They are exempt "
	                        "from the deliverable-output guard, never registered as artifacts, and EXCLUDED "
	                        "from the workspace patch. Use outputs=[...] for real deliverables."
	                    ),
	                },
	                "timeout_sec": {
	                    "type": "integer",
	                    "description": (
	                        "Optional per-call timeout in seconds for long builds/tests (alias: timeout). "
	                        "Clamped to the remaining task-deadline budget. Omit for the default (deadline-capped)."
	                    ),
	                },
	                **RESULT_VIEW_PARAMS,
	            }, "required": ["cmd"]},
        }, _run_shell, is_code_tool=True, timeout_sec=_RUN_SHELL_DEFAULT_TIMEOUT_SEC, mutates_worktree=True),
        ToolEntry("run_script", {
            "name": "run_script",
            "description": (
                "Run a short task-scoped temporary script with a declared interpreter. "
                "Use for multi-line diagnostics or harness helpers; generated scripts use a private run directory inside the mapped workspace or the task drive. "
                "The underlying command result echoes the resolved cwd."
            ),
            "parameters": {"type": "object", "properties": {
                "script": {"type": "string"},
	                "interpreter": {"type": "string", "default": "python3", "description": "Installed executable name or path that accepts a script filename, such as python3, node, perl, zsh or lua. Receives the temporary script path followed by args. Use run_command for compiler or launcher subcommands."},
	                "args": {"type": "array", "items": {"type": "string"}, "default": []},
	                "cwd": {"type": "string", "default": "", "description": "Omit for active_workspace; use system_repo[/subdir] for Ouroboros or skill_payload[/subdir] with bucket+skill_name for a skill."},
	                "env_from_settings": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Explicit environment variable → saved setting key mapping. Uses existing Settings-selection authority; secret values are masked in diagnostics."},
	                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "user_repo"], "description": "Physical skill location for cwd=skill_payload[/subdir]."},
	                "skill_name": {"type": "string", "description": "Exact skill identity for cwd=skill_payload[/subdir]."},
	                "outputs": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": "Generated paths copied/registered after success, e.g. outputs=['report.txt']. The managed artifact directory is created lazily; do not move files into an assumed physical store.",
	                },
	                "scratch": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": (
	                        "Transient in-repo verification files (e.g. a throwaway test you write, run, and "
	                        "delete to check your own work) — throwaway verification ONLY, never part of the "
	                        "solution. Each must be untracked and confined to the cwd: a NEW file, or an existing "
	                        "untracked file created earlier in THIS task (adopted by sha, so re-declaring is "
	                        "idempotent); tracked files and directories stay blocked. They are exempt "
	                        "from the deliverable-output guard, never registered as artifacts, and EXCLUDED "
	                        "from the workspace patch. Use outputs=[...] for real deliverables."
	                    ),
	                },
	                "timeout_sec": {
	                    "type": "integer",
	                    "description": (
	                        "Optional per-call timeout in seconds for long scripts (alias: timeout). "
	                        "Clamped to the remaining task-deadline budget. Omit for the default (deadline-capped)."
	                    ),
	                },
	                **RESULT_VIEW_PARAMS,
	            }, "required": ["script"]},
        }, _run_script, is_code_tool=True, timeout_sec=_RUN_SHELL_DEFAULT_TIMEOUT_SEC, mutates_worktree=True),
    ]
