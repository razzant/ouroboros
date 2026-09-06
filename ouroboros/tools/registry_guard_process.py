"""Process/shell guard helpers: self-change tripwires, read-only inspection classification and light-mode repo snapshots.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess

from ouroboros.contracts.skill_payload_policy import SKILL_OWNER_STATE_STEMS

import ouroboros.tools.registry_guards as registry_guards
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _replace_tool_result,
)
from ouroboros.tools.write_shape import _directory_change_argv

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from typing import Any
    from typing import Dict
    from typing import Optional


def _registry():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import registry

    return registry


def _detect_runtime_mode_elevation(text_lower: str, *, writeish: bool = True) -> bool:
    """Detect shell/script attempts to change ``OUROBOROS_RUNTIME_MODE``."""
    has_save = "save_settings" in text_lower
    has_mode_key = "ouroboros_runtime_mode" in text_lower
    has_dotted_path = "ouroboros.config.save_settings" in text_lower
    detected = (has_save and has_mode_key) or has_dotted_path
    return _registry()._owner_control_mention_blocks(text_lower, detected, writeish)


def _subagent_shell_targets_secret(cmd_path_lower: str, *, ctx: Any = None, cwd: Any = None) -> bool:
    """Use the same physical read targets as file tools and the other shell lanes."""
    from ouroboros.tools.core_secret_paths import _is_subagent_secret_repo_target, restricted_data_roots
    from ouroboros.tools.shell_guards import shell_inspection_paths

    data_roots = restricted_data_roots(ctx) if ctx is not None else []
    repo_root = (_registry().active_repo_dir_for(ctx)
                 if getattr(ctx, "repo_dir", None) is not None else pathlib.Path(cwd or "."))
    work_dir = pathlib.Path(cwd or repo_root).resolve(strict=False)
    paths = shell_inspection_paths(
        cmd_path_lower, work_dir=work_dir,
        drive_root=data_roots[0] if data_roots else None,
    )
    return any(_is_subagent_secret_repo_target(target, repo_root, ctx=ctx) for target in paths)


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
    return _registry()._owner_control_mention_blocks(text_lower, has_key and has_write, writeish)


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
    return _registry()._owner_control_mention_blocks(text_lower, has_key and has_write, writeish)


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
    return _registry()._owner_control_mention_blocks(text_lower, detected, writeish)


_READ_ONLY_INSPECTION_COMMANDS = frozenset({
    "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack", "ripgrep",
    "cat", "bat", "head", "tail", "less", "more", "nl", "strings",
    "ls", "find", "fd", "stat", "file", "wc", "sort", "uniq", "cut", "tr", "column",
    "basename", "dirname", "realpath", "readlink", "diff", "cmp", "jq", "yq",
    "echo", "printf", "true", "pwd", "date", "tree",
})


_COMMAND_HEAD_WRAPPERS = frozenset({
    "sudo", "env", "command", "builtin", "exec", "nohup", "time", "nice", "ionice",
    "stdbuf", "\\",
})


# ``git`` read-only classification: the git_shell_policy SSOT
# (``_git_subcommand_is_readonly``), shared with the shell git guards and the
# affordance map — two divergent allowlists disagreed on the same line (#447 A7).


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
    # yq edits the named file in place with -i/--inplace; without this the family
    # read-carve exempted `yq -i '.OUROBOROS_SAFETY_MODE="off"' settings.json` as
    # "pure inspection" (jq has no in-place edit and stays a stdout-only read).
    "yq": frozenset({"-i", "--inplace"}),
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


_NESTED_EXECUTION_MARKERS = ("$(", "`", "<(", ">(")


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
            from ouroboros.git_shell_policy import (
                _git_output_file_args,
                _git_subcommand_and_args,
                _git_subcommand_is_readonly,
            )

            subcmd, sub_args = _git_subcommand_and_args(tokens)
            if not subcmd or not _git_subcommand_is_readonly(subcmd, sub_args):
                return False
            if _git_output_file_args(sub_args):
                return False  # `git log --output=<file>` truncates and writes <file>
        elif not head or head not in _READ_ONLY_INSPECTION_COMMANDS:
            return False
        denied = _DENIED_READ_OPTIONS.get(head)
        if denied and any(_denied_read_option(token, denied) for token in tokens[1:]):
            return False
        if head == "uniq" and sum(1 for t in tokens[1:] if t == "-" or not t.startswith("-")) >= 2:
            # uniq's SECOND positional operand is its output file ('-' is the
            # stdin operand, not a flag): `... | uniq - settings.json` writes.
            return False
    return True


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
    return _registry()._owner_control_mention_blocks(text_lower, detected, writeish)


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
    return _registry()._owner_control_mention_blocks(text_lower, detected, writeish)


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
    return _registry()._owner_control_mention_blocks(text_lower, detected, writeish)


def _mentions_detached_process(text_lower: str) -> bool:
    return any(marker in text_lower for marker in _DETACHED_PROCESS_MARKERS)


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
        paths = _registry().parse_porcelain_paths(status.stdout)
        digest = hashlib.sha256()
        digest.update((status.stdout or "").encode("utf-8", errors="replace"))
        digest.update((unstaged.stdout if unstaged.returncode == 0 else "").encode("utf-8", errors="replace"))
        digest.update((staged.stdout if staged.returncode == 0 else "").encode("utf-8", errors="replace"))
        for rel in paths:
            try:
                target = (repo / _registry().safe_relpath(rel)).resolve(strict=False)
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


def _lane_writer_targets(raw_cmd: Any) -> tuple[list, list, list[str], set[str]]:
    """The run_command lane's per-segment writer-target facts, derived ONCE from
    the raw command: ``(target_rows, write_target_argvs, explicit_write_targets,
    executable_path_tokens)``. Every guard below consumes these rows rather than
    re-tokenizing the command (the inline body stays a STRING so its own operator
    grammar survives — re-tokenizing it with plain shlex once glued ``2>/dev/null;``
    onto the following command and forged the path ``/dev/null;``)."""
    # ONE per-segment writer-target SSOT for this lane. The inline body is
    # carried as a STRING so its own operator grammar survives; re-tokenizing
    # it with plain shlex glued `2>/dev/null;` onto the following command and
    # forged the path `/dev/null;` out of a redirection.
    target_rows = _registry().writer_target_rows(raw_cmd)
    write_target_argvs = [list(row[0]) for row in target_rows]
    explicit_write_targets = list(dict.fromkeys(
        str(token)
        for row in target_rows
        for token in row[1]
        if not _directory_change_argv(row[0])
        if str(token or "").strip()
    ))
    # ``cp source Deliverables/`` (and the equivalent mv/ln form) writes a
    # child named after the source, while the ordinary writer-target parser
    # only sees the directory operand. Add those argv-visible child names to
    # the same target-first policy without attempting to parse inline code,
    # archive formats, or other deferred Q3 syntax.
    for row_index, target_argv in enumerate(write_target_argvs):
        for command, destination, source in _registry().directory_destination_pairs(target_argv):
            source_name = _registry().directory_destination_child_name(command, target_argv, source)
            if source_name in {"", ".", ".."}:
                continue
            derived = destination.rstrip("/\\") + "/" + source_name
            row_argv, row_targets, row_inline, row_unprovable = target_rows[row_index]
            target_rows[row_index] = (
                row_argv, [*row_targets, derived], row_inline, row_unprovable,
            )
    # A located -e/-E/-c inline CODE BODY is not a write target: the
    # generic fallback reported every non-flag operand of a writer command
    # (ruby/perl) - code string included - making every one-liner
    # write-shaped. The light fence and protected lane keep the unfiltered
    # SSOT (pinned XG-7B3.1); only THIS lane drops the bodies. FILE
    # operands stay write-suspect (`perl -pi -e s/a/b/ file` rewrites
    # `file`); literal in-code targets still arrive via inline extraction.
    inline_code_bodies: set = set()
    for target_argv in write_target_argvs:
        inline_code_bodies.update(_registry().interpreter_inline_code([str(t) for t in target_argv]))
    if inline_code_bodies:
        explicit_write_targets = [t for t in explicit_write_targets if t not in inline_code_bodies]
    explicit_write_targets = list(dict.fromkeys(explicit_write_targets))
    executable_path_tokens = {str(target_argv[0]) for target_argv in write_target_argvs if target_argv}
    return target_rows, write_target_argvs, explicit_write_targets, executable_path_tokens


def _run_shell_safety_check(
    self, args: Dict[str, Any], runtime_mode: str, binding: Any = None,
) -> ToolResult | None:
    """Pre-execution run_command filter; returns a native denial or ``None``."""
    from ouroboros.shell_parse import local_shell_subject

    raw_cmd = args.get("cmd", args.get("command", ""))
    if binding is None:
        operation = (
            "service"
            if str(args.get("__tool_name") or "") == "start_service"
            else "shell"
        )
        try:
            binding = _registry().build_resolved_resource_binding(
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
                text=_registry().shell_cwd_block_message(
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
    argv = _registry().strip_leading_env_assignments(_registry().unwrap_env_argv(_registry().shell_argv(raw_cmd)))
    if _registry().sudo_noninteractive_violation(raw_cmd):
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
    if (acting_subagent or self._is_local_readonly_subagent()) and _subagent_shell_targets_secret(
            raw_cmd, ctx=self._ctx, cwd=getattr(binding, "target_path", None)):
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
    inline_argv: list = []
    if argv_executable in {"sh", "bash", "zsh"}:
        inline_cmd = next((str(argv_for_write[idx + 1] or "") for idx, token in enumerate(argv_for_write[1:], start=1) if str(token or "") in {"-c", "--command"} and idx + 1 < len(argv_for_write)), "")
        if not inline_cmd:
            inline_cmd = _registry().shell_command_string(argv_for_write)
        inline_argv = _registry().strip_leading_env_assignments(_registry().unwrap_env_argv(_registry().shell_argv(inline_cmd)))
    # Only filesystem writer targets use SSH's local-effect projection.
    # Owner controls, credentials, safety and execution inspect the full argv.
    writer_cmd = local_shell_subject(raw_cmd)
    target_rows, write_target_argvs, explicit_write_targets, executable_path_tokens = _lane_writer_targets(writer_cmd)
    # Writer-command membership canonicalizes versioned interpreter spellings to
    # their family (`ruby3.2` is `ruby`), so a versioned basename is exactly as
    # write-suspect as the unversioned one (XG-2R.2).
    # Interpreter argv (direct or inside sh -c) takes the MODE-AWARE
    # write-shape classifier: the bare `open(` token classified read-only
    # `open(p, 'rb')` as a write ("the original GAIA class"). Write-mode
    # opens, pathlib `.open('w')`, save-APIs, opaque subprocess escapes
    # and shell-level indicators still classify as writes;
    # `writer_target_tokens` keeps covering literal targets below.
    write_shape_interpreter = bool(_registry().interpreter_family(argv_executable)) or (
        bool(inline_argv)
        and bool(_registry().interpreter_family(pathlib.PurePath(str(inline_argv[0])).name.lower().removesuffix(".exe")))
    )
    # ONE mode-aware write-shape seam (write_shape.py) for BOTH halves:
    # interpreter argv takes interpreter_write_shape; everything else takes
    # non_interpreter_write_shape, where unconditional writers keep the
    # membership floor, pure-filter utilities (sort/uniq/sed/tar/gzip) need a
    # real write channel, and prose words yield to the same read-carve the
    # owner-control detectors use. No guard below consumes a coarser fact.
    coarse_write_shape = (
        _registry().interpreter_write_shape(raw_cmd)
        if write_shape_interpreter
        else _registry().non_interpreter_write_shape(
            raw_cmd, argv_for_write, argv_executable, is_pure_read=_is_pure_read_inspection,
        )
    )
    writeish = (
        coarse_write_shape
        or bool(explicit_write_targets)
        or any(row[3] for row in target_rows)
    )
    work_dir = registry_guards._resolved_shell_cwd(self, args, binding)
    if isinstance(work_dir, ToolResult):
        return work_dir
    if protected_artifact_block := _registry().protected_artifact_shell_block_reason(
        self._ctx,
        raw_cmd,
        cwd=str(work_dir),
        default_cwd=pathlib.Path(work_dir),
        binding=_registry()._binding_items(binding)[0] if _registry()._binding_items(binding) else None,
    ):
        return ToolResult(
            status="blocked",
            # protected_artifact_shell_block_reason emits only the resource-POLICY
            # refusal; the two resource blocks are distinct codes because only they
            # demote a block on a read-only tool to ignored telemetry.
            code="RESOURCE_POLICY_BLOCKED",
            text=protected_artifact_block,
        )
    if writeish and (executor_state_block := _registry().workspace_executor_state_write_block(
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
            writer_cmd,
            (" ".join(str(x) for x in writer_cmd) if isinstance(writer_cmd, list) else str(writer_cmd)).lower().replace("\\", "/"),
            explicit_write_targets,
            target_rows,
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
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: shell command pattern looks like an OUROBOROS_RUNTIME_MODE elevation attempt (mentions ``save_settings`` together with ``OUROBOROS_RUNTIME_MODE``, or invokes ``ouroboros.config.save_settings`` directly). Runtime mode is owner-controlled — change it by stopping the agent and editing settings.json directly, then restart.")
    if _detect_context_mode_self_lowering(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="CONTEXT_MODE_SELF_LOWERING_BLOCKED", text="⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to lower OUROBOROS_CONTEXT_MODE to low through settings.json or /api/owner/context-mode. Context mode is owner-controlled — ask the owner to change the Low/Max toggle or edit settings while the agent is stopped.")
    if _detect_safety_mode_self_lowering(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="SAFETY_MODE_SELF_LOWERING_BLOCKED", text="⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: shell command pattern looks like an attempt to change OUROBOROS_SAFETY_MODE (e.g. to ``light``/``off``) through settings.json, /api/settings, or /api/owner/safety-mode. LLM-safety coverage is owner-controlled (BIBLE P3) — the agent must not reduce its own supervision. Ask the owner to change it via the dedicated /api/owner/safety-mode endpoint, or stop the agent and edit settings.json directly.")
    if _detect_owner_skill_attest_self_call(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED", text="⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: shell command pattern looks like an attempt to loopback-POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips the expensive LLM skill review and is OWNER-ONLY — the agent must not self-attest its own skill to bypass the immune system's review. Ask the owner to attest it from the Skills UI.")
    if _detect_mutative_toggle_self_change(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is owner-controlled (it grants subagents write power against the live body). Change it by stopping the agent and editing settings.json directly, then restart — the agent must not self-enable mutative subagents.")
    if _detect_evolution_owner_control_self_change(cmd_lower, writeish=writeish):
        return ToolResult(status="blocked", code="ELEVATION_BLOCKED", text="⚠️ ELEVATION_BLOCKED: the self-evolution controls (OUROBOROS_POST_TASK_EVOLUTION and OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE) are owner-controlled — they enable or steer self-modification cycles. Change them via the owner Settings UI, or stop the agent and edit settings.json directly — the agent must not self-set evolution controls.")
    if _mentions_skill_owner_state(cmd_lower, writeish=writeish):
        return ToolResult(
            status="blocked",
            code="SKILL_STATE_WRITE_BLOCKED",
            text=(
                "⚠️ SKILL_STATE_WRITE_BLOCKED: skill review, enablement, "
                "grants, and marketplace provenance are owner/review "
                "controlled state. Use skill_review, toggle_skill/the Skills "
                "UI, or the desktop launcher confirmation flow. Pure read-only "
                "inspection (grep/rg/cat/jq) of these names is allowed."
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
        if _registry().light_shell_repo_mutation(
            raw_cmd,
            repo_dir=_registry().system_repo_dir_for(self._ctx),
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
            or bool(_registry().interpreter_family(runtime_data_executable))
        )
        if runtime_data_scan:
            own_task_drive = pathlib.Path(self._ctx.task_drive_root())
            own_artifact_dir = _registry().task_artifact_dir_path(
                pathlib.Path(self._ctx.drive_root),
                _registry().task_id_for_artifacts(self._ctx),
                create=False,
            )
            allowed_runtime_roots = [own_task_drive, own_artifact_dir]
            for item in _registry()._binding_items(binding):
                if item.root == "skill_payload" and item.source != "native":
                    allowed_runtime_roots.append(pathlib.Path(item.base_path))
            runtime_data_targets = _registry().runtime_data_guard_targets(
                raw_cmd,
                writeish=writeish,
                drive_root=pathlib.Path(self._ctx.drive_root),
                work_dir=pathlib.Path(work_dir),
                allowed_roots=allowed_runtime_roots,
                target_rows=target_rows,
            )
            if runtime_data_targets:
                # Name the REAL task roots: a mis-guessed absolute path used to
                # produce this block with no way to self-correct (v6.54.3).
                return ToolResult(
                    status="blocked",
                    code="LIGHT_MODE_BLOCKED",
                    text=(
                        "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks this command's "
                        "access to runtime_data outside the permitted task roots. "
                        f"This task's real roots are: artifact_store={own_artifact_dir}, "
                        f"task_drive={own_task_drive} — staged attachments live under "
                        f"{own_artifact_dir / 'attachments'}. Use those absolute paths in scripts, "
                        "or root=artifact_store / root=task_drive / root=user_files in file tools. "
                        "Blocked paths: " + ", ".join(runtime_data_targets[:5])
                    ),
                )

    if protected_shell := registry_guards._protected_shell_block(
        self, raw_cmd, cmd_path_lower, binding, acting_self_worktree, writeish,
    ):
        return protected_shell

    # GitHub repo create/delete/auth — argv-positional, never substring (#447 A7).
    from ouroboros.git_shell_policy import gh_shell_block_reason

    if gh_block := gh_shell_block_reason(raw_cmd):
        return ToolResult(status="blocked", code="SAFETY_VIOLATION", text=gh_block)

    return registry_guards._shell_git_and_runtime_block(
        self, raw_cmd, args, cmd_path_lower, workspace_mode,
        acting_self_worktree, binding,
    )


def _owner_settings_snapshot() -> Optional[str]:
    """Text of the live settings.json, "" if absent, None if UNREADABLE.

    None disarms the tripwire below: the deleted restore recorded an OSError as
    "file absent" and could unlink the live settings.json — a baseline is either
    read successfully or not used at all."""
    from ouroboros import config as _cfg

    path = pathlib.Path(_cfg.SETTINGS_PATH)
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return None


def _run_shell_post_checks(
    self,
    result: str | ToolResult,
    *,
    light_repo_before: Optional[Dict[str, Any]],
    workspace_refs_before: Optional[Dict[str, str]],
    settings_before: Optional[str] = None,
    tool_name: str = "run_command",
) -> str | ToolResult:
    """Post-execution tripwires. They ANNOTATE, they never roll back.

    The owner-state snapshot/restore (OWNER_STATE_RESTORED) that used to run here
    was DELETED (issue #447, owner decision): it reverted ANY post-command
    difference without proving the command caused it, so a concurrent owner edit
    (Settings UI, grant click) was silently rolled back and blamed on the command;
    and its snapshot recorded an OSError while reading as "file absent", so the
    restore could UNLINK the live settings.json after a transient read error. The
    light-lane guard below refuses auto-rollback for exactly this reason. Pre-exec
    guards keep skill owner state (SKILL_STATE_WRITE_BLOCKED on any writeish /
    non-inspection mention — pure read inspection is carved, #447 A2); the
    settings.json mention-gates are lexical, so an obfuscated argument-level
    writer (``S=settings; cat > data/$S.json``) can pass them — the tripwire below
    makes that LOUD (a typed ``tripwire`` fact plus an appended note) without
    re-introducing the unsound rollback. Disclosed residual: the write itself is
    not reverted; the owner surface is the remedy. Every note TRAILS the payload
    (#447 В12/H1) so line 1 stays with the command's own outcome."""
    text = result.text if isinstance(result, ToolResult) else result
    typed = result if isinstance(result, ToolResult) else None

    def _typed_base() -> ToolResult:
        # A tripwire fact must survive even when the producer returned plain
        # text: since the notes TRAIL the payload (#447 H1), the marker no
        # longer owns line 1, so a text-only reader could not re-derive the
        # classification. Adapt once, through the ONE legacy adapter, so the
        # fact is carried typed instead of being lost in prose.
        nonlocal typed
        if typed is None:
            typed = LegacyTextResultAdapter.from_text(tool_name, text)
        return typed

    if settings_before is not None:
        settings_after = _owner_settings_snapshot()
        if settings_after is not None and settings_after != settings_before:
            text = (
                f"{text}\n\n⚠️ OWNER_SETTINGS_CHANGED: data/settings.json changed while "
                "this command ran. Owner settings change only through save_settings / "
                "the Settings UI; this write was NOT auto-reverted (a post-hoc rollback "
                "can clobber a concurrent legitimate owner edit) — the owner surface is "
                "the place to verify and restore."
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                meta_updates={"tripwire": "owner_settings_changed"},
            )
    if light_repo_before is not None:
        light_repo_after = _light_repo_snapshot(_registry().system_repo_dir_for(self._ctx))
        if (
            light_repo_after is not None
            and light_repo_after.get("digest") != light_repo_before.get("digest")
        ):
            text = (
                f"{text}\n\n"
                + _format_light_repo_write_note(
                    light_repo_before, light_repo_after, tool_name=tool_name
                )
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                code="LIGHT_MODE_REPO_WRITE_BLOCKED",
                meta_updates={"light_repo_changed": True},
            )
    if workspace_refs_before is not None:
        workspace_refs_after = _git_ref_snapshot(_registry().active_repo_dir_for(self._ctx))
        if (
            workspace_refs_after is not None
            and workspace_refs_after.get("digest") != workspace_refs_before.get("digest")
        ):
            text = (
                f"{text}\n\n"
                "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs "
                "inside the external workspace. External workspace runs must leave "
                "changes as files/patch artifacts, not commits/tags/resets."
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                code="WORKSPACE_GIT_REF_CHANGED",
                meta_updates={"workspace_git_refs_changed": True},
            )
    return typed if typed is not None else text
