"""Pre- and post-execution process guards for shell-backed registry tools."""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
from typing import Any, Dict, Optional

import ouroboros.tools.registry_guards as registry_guards
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,
    SKILL_OWNER_STATE_STEMS,
)
from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason
from ouroboros.shell_parse import (
    shell_argv,
    shell_command_string,
    strip_leading_env_assignments,
    sudo_noninteractive_violation,
    unwrap_env_argv,
)
from ouroboros.tool_access import build_resolved_resource_binding, shell_cwd_block_message
from ouroboros.tools.shell_guards import (
    LIGHT_SHELL_WRITER_COMMANDS,
    interpreter_family,
    light_shell_repo_mutation,
    parse_porcelain_paths,
    runtime_data_guard_targets,
    shell_has_write_indicator,
    workspace_executor_state_write_block,
    writer_target_tokens,
)
from ouroboros.tools.tool_resolution import (
    _binding_items,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.tools.tool_result import ToolResult, _replace_tool_result
from ouroboros.utils import safe_relpath


def _detect_runtime_mode_elevation(text_lower: str) -> bool:
    """Detect shell/script attempts to change ``OUROBOROS_RUNTIME_MODE``."""
    has_save = "save_settings" in text_lower
    has_mode_key = "ouroboros_runtime_mode" in text_lower
    has_dotted_path = "ouroboros.config.save_settings" in text_lower
    return (has_save and has_mode_key) or has_dotted_path


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


def _subagent_shell_targets_secret(cmd_path_lower: str) -> bool:
    """Deterministic guard: a shell command referencing Ouroboros secrets/credentials
    or owner-control state (settings.json, ssh keys, token/credential files)."""
    return any(marker in cmd_path_lower for marker in _SUBAGENT_SHELL_SECRET_MARKERS)


def _detect_mutative_toggle_self_change(text_lower: str) -> bool:
    """Detect shell/script/CLI attempts to change the owner-only mutative-subagents toggle."""
    has_key = "ouroboros_allow_mutative_subagents" in text_lower
    has_write = (
        "save_settings" in text_lower
        or "settings.json" in text_lower
        or "/api/settings" in text_lower
        or "settings set" in text_lower  # `ouroboros settings set <key> <value>` CLI path
        or "ouroboros.cli" in text_lower
    )
    return has_key and has_write


def _detect_evolution_owner_control_self_change(text_lower: str) -> bool:
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
    return has_key and has_write


def _detect_context_mode_self_lowering(text_lower: str) -> bool:
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
    return (
        mentions_owner_endpoint
        or mentions_context_endpoint
        or mentions_context_cli
        or mentions_owner_lowering_flag
        or (mentions_context_key and mentions_save)
    )


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


def _detect_safety_mode_self_lowering(text_lower: str) -> bool:
    """Detect shell/script attempts to lower the owner-controlled LLM-safety coverage
    (v6.54.3). OUROBOROS_SAFETY_MODE gates the LLM safety supervisor layer; the agent
    must not switch it to ``light``/``off`` to remove its own supervision friction
    (BIBLE P3). Mirrors the context-mode/scope-floor self-lowering guards."""
    import urllib.parse

    # URL-decode so a percent-encoded endpoint (curl .../safety%2Dmode) cannot
    # slip the literal scan (review round 6; mirrors the attest-endpoint guard).
    decoded = urllib.parse.unquote(urllib.parse.unquote(text_lower)).lower()
    text_lower = f"{text_lower} {decoded}"
    mentions_mode_key = "ouroboros_safety_mode" in text_lower
    mentions_owner_endpoint = "/api/owner/safety-mode" in text_lower
    mentions_mode_endpoint = "safety-mode" in text_lower and "/api/owner" in text_lower
    mentions_mode_cli = "safety-mode" in text_lower and (
        "ouroboros settings" in text_lower
        or "ouroboros.cli" in text_lower
    )
    mentions_save = "save_settings" in text_lower or "settings.json" in text_lower or "/api/settings" in text_lower
    return (
        mentions_owner_endpoint
        or mentions_mode_endpoint
        or mentions_mode_cli
        or (mentions_mode_key and mentions_save)
    )


def _detect_owner_skill_attest_self_call(text_lower: str) -> bool:
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
    return "/api/owner/skills/" in text and "attest-review" in text


_SKILL_OWNER_STATE_STEMS = SKILL_OWNER_STATE_STEMS
_DETACHED_PROCESS_MARKERS = ("start_new_session", "new_session", "setsid", "preexec_fn", "nohup")


def _mentions_skill_owner_state(text_lower: str) -> bool:
    if "state" not in text_lower or "skills" not in text_lower:
        return False
    for stem in _SKILL_OWNER_STATE_STEMS:
        if f"{stem}.json" in text_lower:
            return True
        if stem in text_lower and ".json" in text_lower:
            return True
    return False


def _mentions_detached_process(text_lower: str) -> bool:
    return any(marker in text_lower for marker in _DETACHED_PROCESS_MARKERS)


def _run_shell_safety_check(
    self, args: Dict[str, Any], runtime_mode: str, binding: Any = None,
) -> ToolResult | None:
    """Pre-execution run_command filter; returns a native denial or ``None``."""
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
            return ToolResult(
                status="blocked",
                code="SHELL_CWD_BLOCKED",
                text=shell_cwd_block_message(
                    self._ctx,
                    str(args.get("cwd") or ""),
                    operation=operation,
                    error=exc,
                ),
            )
    workspace_mode = bool(getattr(self._ctx, "is_workspace_mode", lambda: False)())
    # self_worktree is a checkout of the system repo, so protected shell-write
    # guards must stay active for it even in workspace mode (acting children
    # must use write_file/edit_text, which apply the pro+grant gate).
    acting_self_worktree = self._acting_self_worktree()
    acting_subagent = self._is_acting_subagent()
    argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
    if sudo_noninteractive_violation(argv):
        return ToolResult(
            status="blocked",
            code="SUDO_INTERACTIVE_BLOCKED",
            text="⚠️ SUDO_INTERACTIVE_BLOCKED: sudo must be noninteractive. Use sudo -n for commands that can run without a password; if sudo -n fails, report validation/install blocked by environment.",
        )
    cmd_lower = (" ".join(str(x) for x in raw_cmd) if isinstance(raw_cmd, list) else str(raw_cmd)).lower()
    cmd_path_lower = cmd_lower.replace("\\", "/")
    while "//" in cmd_path_lower: cmd_path_lower = cmd_path_lower.replace("//", "/")
    # Subagents must not read owner secrets/credentials/control state via shell
    # (read_file already denies these). read_file is the gated inspection path.
    if (acting_subagent or self._is_local_readonly_subagent()) and _subagent_shell_targets_secret(cmd_path_lower):
        return ToolResult(
            status="blocked",
            code="SUBAGENT_SECRET_READ_BLOCKED",
            text=(
                "⚠️ SUBAGENT_SECRET_READ_BLOCKED: subagents may not read Ouroboros secrets, "
                "credentials, or owner-control state via shell. Use the gated read_file tool "
                "(which denies secrets) for any inspection you actually need."
            ),
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
    work_dir = registry_guards._resolved_shell_cwd(self, args, binding)
    if isinstance(work_dir, ToolResult):
        return work_dir
    if protected_artifact_block := protected_artifact_shell_block_reason(
        self._ctx,
        raw_cmd,
        cwd=str(work_dir),
        default_cwd=pathlib.Path(work_dir),
        binding=_binding_items(binding)[0] if _binding_items(binding) else None,
    ):
        return ToolResult(
            status="blocked",
            # protected_artifact_shell_block_reason emits only the resource-POLICY
            # refusal; the two resource blocks are distinct codes because only they
            # demote a block on a read-only tool to ignored telemetry.
            code="RESOURCE_POLICY_BLOCKED",
            text=protected_artifact_block,
        )
    if writeish and (executor_state_block := workspace_executor_state_write_block(
        raw_cmd,
        drive_root=pathlib.Path(self._ctx.drive_root),
        cwd=str(work_dir),
        default_cwd=pathlib.Path(work_dir),
    )):
        return ToolResult(
            status="blocked",
            code="WORKSPACE_BLOCKED",
            text=executor_state_block,
        )
    if workspace_mode and writeish:
        workspace_write_block = registry_guards._workspace_shell_write_block(
            self,
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
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: shell command pattern looks like an OUROBOROS_RUNTIME_MODE elevation attempt (mentions ``save_settings`` together with ``OUROBOROS_RUNTIME_MODE``, or invokes ``ouroboros.config.save_settings`` directly). Runtime mode is owner-controlled — change it by stopping the agent and editing settings.json directly, then restart.")
    if _detect_context_mode_self_lowering(cmd_lower):
        return ToolResult(status="blocked", code="CONTEXT_MODE_SELF_LOWERING_BLOCKED", text="⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to lower OUROBOROS_CONTEXT_MODE to low through settings.json or /api/owner/context-mode. Context mode is owner-controlled — ask the owner to change the Low/Max toggle or edit settings while the agent is stopped.")
    if _detect_scope_review_floor_self_lowering(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED", text="⚠️ SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED: shell command pattern reaches OUROBOROS_SCOPE_REVIEW_FLOOR through settings.json, /api/settings, or /api/owner/scope-review-floor from something other than a pure read. The floor is a deprecated, enforcement-inert owner setting (BIBLE P3 scope-review applicability follows the owner context mode) — it stays owner-only, and the agent must not write owner settings through any channel. Ask the owner to change it via the dedicated /api/owner/scope-review-floor endpoint, or stop the agent and edit settings.json directly. Pure source inspection (grep/rg/cat/jq/git grep) is allowed; an interpreter or HTTP client naming the endpoint is not, whatever verb it spells.")
    if _detect_safety_mode_self_lowering(cmd_lower):
        return ToolResult(status="blocked", code="SAFETY_MODE_SELF_LOWERING_BLOCKED", text="⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to change OUROBOROS_SAFETY_MODE (e.g. to ``light``/``off``) through settings.json, /api/settings, or /api/owner/safety-mode. LLM-safety coverage is owner-controlled (BIBLE P3) — the agent must not reduce its own supervision. Ask the owner to change it via the dedicated /api/owner/safety-mode endpoint, or stop the agent and edit settings.json directly.")
    if _detect_owner_skill_attest_self_call(cmd_lower):
        return ToolResult(status="blocked", code="OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED", text="⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: shell command pattern looks like an attempt to loopback-POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips the expensive LLM skill review and is OWNER-ONLY — the agent must not self-attest its own skill to bypass the immune system's review. Ask the owner to attest it from the Skills UI.")
    if _detect_mutative_toggle_self_change(cmd_lower):
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is owner-controlled (it grants subagents write power against the live body). Change it by stopping the agent and editing settings.json directly, then restart — the agent must not self-enable mutative subagents.")
    if _detect_evolution_owner_control_self_change(cmd_lower):
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: the self-evolution controls (OUROBOROS_POST_TASK_EVOLUTION and OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE) are owner-controlled — they enable or steer self-modification cycles. Change them via the owner Settings UI, or stop the agent and edit settings.json directly — the agent must not self-set evolution controls.")
    if _mentions_skill_owner_state(cmd_lower):
        return ToolResult(
            status="blocked",
            code="SKILL_STATE_WRITE_BLOCKED",
            text=(
                "⚠️ SKILL_STATE_WRITE_BLOCKED: skill review, enablement, "
                "grants, and marketplace provenance are owner/review "
                "controlled state. Use skill_review, toggle_skill/the Skills "
                "UI, or the desktop launcher confirmation flow."
            ),
        )
    if "state" in cmd_lower and "skills" in cmd_lower and _mentions_detached_process(cmd_lower):
        return ToolResult(
            status="blocked",
            code="SKILL_STATE_WRITE_BLOCKED",
            text=(
                "⚠️ SKILL_STATE_WRITE_BLOCKED: detached shell processes must "
                "not target skill state directories. Use the reviewed skill "
                "lifecycle tools instead."
            ),
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
            return ToolResult(
                status="blocked",
                code="LIGHT_MODE_BLOCKED",
                text=(
                    "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses "
                    "shell commands that mutate the Ouroboros repository. "
                    "For external deliverables, run with cwd under user_files "
                    "(for example /Users/<you>/Desktop), root=artifact_store, "
                    "or root=task_drive. Switch to advanced/pro only for "
                    "reviewed Ouroboros self-modification."
                ),
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
                return ToolResult(
                    status="blocked",
                    code="LIGHT_MODE_BLOCKED",
                    text=(
                        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks process commands "
                        f"that {action} runtime_data paths outside this task's own roots. "
                        f"This task's real roots are: artifact_store={own_artifact_dir}, "
                        f"task_drive={own_task_drive} — staged attachments live under "
                        f"{own_artifact_dir / 'attachments'}. Use those absolute paths in scripts, "
                        "or root=artifact_store / root=task_drive / root=user_files in file tools. "
                        "Blocked paths: " + ", ".join(runtime_data_targets[:5])
                    ),
                )

    if protected_shell := registry_guards._protected_shell_block(
        self, raw_cmd, cmd_path_lower, binding, acting_self_worktree,
    ):
        return protected_shell

    # GitHub repo create/delete/auth.
    cmd_words = re.sub(r"\s+", " ", cmd_lower)
    if "gh repo create" in cmd_words or "gh repo delete" in cmd_words:
        return ToolResult(status="blocked", code="SAFETY_VIOLATION", text="⚠️ SAFETY_VIOLATION: Creating/deleting GitHub repositories requires admin approval.")
    if "gh auth" in cmd_words:
        return ToolResult(status="blocked", code="SAFETY_VIOLATION", text="⚠️ SAFETY_VIOLATION: Modifying GitHub authentication is not permitted.")

    return registry_guards._shell_git_and_runtime_block(
        self, raw_cmd, args, cmd_path_lower, workspace_mode,
        acting_self_worktree, binding,
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
    result: str | ToolResult,
    *,
    owner_snapshot: Dict[pathlib.Path, Optional[str]],
    state_drive_root: pathlib.Path,
    light_repo_before: Optional[Dict[str, Any]],
    workspace_refs_before: Optional[Dict[str, str]],
    tool_name: str = "run_command",
) -> str | ToolResult:
    import time

    text = result.text if isinstance(result, ToolResult) else result
    typed = result if isinstance(result, ToolResult) else None

    restored_owner_state = False
    for _ in range(4):
        time.sleep(0.3)
        restored_owner_state = (
            _restore_owner_files(self, owner_snapshot, state_drive_root)
            or restored_owner_state
        )
    if restored_owner_state:
        text = (
            f"{text}\n\n⚠️ OWNER_STATE_RESTORED: run_command attempted to "
            "change owner-only settings or skill trust state; protected files were restored."
        )
        if typed is not None:
            typed = _replace_tool_result(
                typed,
                text=text,
                code="OWNER_STATE_RESTORED" if typed.status == "ok" else typed.code,
                meta_updates={"owner_state_restored": True},
            )
    if light_repo_before is not None:
        light_repo_after = _light_repo_snapshot(system_repo_dir_for(self._ctx))
        if (
            light_repo_after is not None
            and light_repo_after.get("digest") != light_repo_before.get("digest")
        ):
            text = _format_light_repo_write_block(
                light_repo_before,
                light_repo_after,
                text,
                tool_name=tool_name,
            )
            if typed is not None:
                typed = _replace_tool_result(
                    typed,
                    text=text,
                    code="LIGHT_MODE_REPO_WRITE_BLOCKED",
                    meta_updates={"light_repo_changed": True},
                )
    if workspace_refs_before is not None:
        workspace_refs_after = _git_ref_snapshot(active_repo_dir_for(self._ctx))
        if (
            workspace_refs_after is not None
            and workspace_refs_after.get("digest") != workspace_refs_before.get("digest")
        ):
            text = (
                "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs "
                "inside the external workspace. External workspace runs must leave "
                "changes as files/patch artifacts, not commits/tags/resets.\n\n"
                "Original command output:\n"
                f"{text}"
            )
            if typed is not None:
                typed = _replace_tool_result(
                    typed,
                    text=text,
                    code="WORKSPACE_GIT_REF_CHANGED",
                    meta_updates={"workspace_git_refs_changed": True},
                )
    return typed if typed is not None else text
