"""Process tools: run_command and run_script."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import os
import pathlib
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from typing import List

from ouroboros.tools.declared_outputs import (  # noqa: F401 -- re-exported
    _OUTPUT_DIR_MAX_FILES,
    _OUTPUT_DIR_MAX_BYTES,
    _allowed_output_roots,
    _protected_output_source_reason,
    _changed_path_covers,
    _resolve_declared_output,
    _directory_fingerprint_from_entries,
    _bounded_directory_fingerprint,
    _fingerprint_output,
    _snapshot_declared_outputs,
    _scan_directory_output_members,
    _register_process_outputs,
    _tree_fingerprint,
    _sensitive_output_component_reason,
)
from ouroboros.artifacts import record_task_scratch
from ouroboros.platform_layer import bootstrap_process_path, kill_process_tree, scrub_repo_from_pythonpath, subprocess_new_group_kwargs
from ouroboros.config import SETTINGS_DEFAULTS, load_settings
from ouroboros.runtime_mode_policy import (
    is_protected_runtime_path,
)
from ouroboros.tools.commit_gate import _invalidate_advisory
from ouroboros.shell_parse import embedded_absolute_path_tokens, recover_stringified_argv, shell_argv_with_inline
from ouroboros.tools.registry import (
    ToolContext,
    ToolEntry,
    active_repo_dir_for,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    path_is_relative_to,
    resource_root_path,
    shell_cwd_block_message,
    user_files_path_block_reason,
)
from ouroboros.deadline_utils import deadline_remaining_sec
from ouroboros.workspace_executor import covers as executor_covers
from ouroboros.workspace_executor import ensure_execution_cwd
from ouroboros.workspace_executor import execute as executor_execute
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_host_path as executor_map_host_path

log = logging.getLogger(__name__)
# Tracked process groups let panic kill descendant trees too.
_active_subprocesses: set = set()
_subprocess_lock = threading.Lock()
_RUN_SHELL_DEFAULT_TIMEOUT_SEC = 360
_CONTROL_DIR_BACKUP_MAX_BYTES = 5 * 1024 * 1024


def _tracked_subprocess_run(cmd, **kwargs):
    """subprocess.run replacement with process-tree tracking. When capturing TEXT
    output, decode tolerantly (errors='replace') so binary stdout/stderr (a MIPS
    interpreter, a DOOM framebuffer, raw bytes) surfaces as readable text instead
    of raising UnicodeDecodeError and collapsing the whole call into a
    shell_error."""
    timeout = kwargs.pop("timeout", None)
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.update(subprocess_new_group_kwargs())
    proc = subprocess.Popen(cmd, **kwargs)
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait(timeout=5)
        raise
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)


def _kill_process_group(proc):
    """Kill a subprocess tree."""
    kill_process_tree(proc)


def kill_all_tracked_subprocesses():
    """Kill all tracked subprocess trees on panic."""
    with _subprocess_lock:
        procs = list(_active_subprocesses)
    for proc in procs:
        _kill_process_group(proc)
    with _subprocess_lock:
        _active_subprocesses.clear()


def _shell_env_for_cwd(ctx: ToolContext, work_dir: pathlib.Path) -> "dict | None":
    """For a command whose cwd is OUTSIDE the Ouroboros system repo (an external
    workspace / target project, e.g. SWE-bench dig-direct ``/app``), return an
    env copy with the repo dir scrubbed from ``PYTHONPATH`` so the target cannot
    shadow-import Ouroboros's own modules (R2). ``ctx.repo_dir`` stays pinned to
    the Ouroboros repo even in workspace mode, so this is the authoritative
    in-repo test. Returns ``None`` for commands inside the system repo (Ouroboros
    tooling legitimately imports itself) so they inherit ``os.environ``."""
    try:
        system_repo = pathlib.Path(getattr(ctx, "repo_dir")).resolve(strict=False)
        wd = pathlib.Path(work_dir).resolve(strict=False)
    except Exception:
        return None
    try:
        in_repo = wd == system_repo or wd.is_relative_to(system_repo)
    except AttributeError:  # pragma: no cover - py<3.9
        in_repo = str(wd) == str(system_repo) or str(wd).startswith(str(system_repo) + os.sep)
    if in_repo:
        return None
    return scrub_repo_from_pythonpath(dict(os.environ), system_repo)


def _resolve_effective_timeout(
    default_timeout_sec: int,
    ctx: ToolContext | None = None,
    override_sec: int | None = None,
) -> int:
    """Resolve the effective per-command timeout as ONE normalized pipeline:
    resolve the REQUESTED value from a single precedence chain (per-call
    ``override_sec`` > env ``OUROBOROS_TOOL_TIMEOUT_SEC`` > settings.json > config
    ``SETTINGS_DEFAULTS`` > the in-code last-resort ``default_timeout_sec``), then
    apply the per-call ceiling, then clamp toward the remaining task-deadline budget
    (60s floor when a deadline exists), then floor at 1s. The outer budget loop
    remains the hard deadline enforcer.

    Hygiene fix (SSOT): the prior code skipped an env/settings value EQUAL to the
    config default (``!= default_setting``), so ``OUROBOROS_TOOL_TIMEOUT_SEC=600``
    (= the SETTINGS_DEFAULTS value) silently fell through to the in-code 360s default.
    The configured value is now honored regardless of equality, and env/settings
    values no longer BYPASS the ceiling/deadline clamp. RELEASE NOTE: installs that
    relied on the buggy effective 360s now get the configured 600s — a foreground
    command may hold the task longer (still bounded by ceiling + task deadline).
    """
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    # 1. Resolve the REQUESTED timeout from a single precedence chain.
    requested: int | None = None
    if override_sec is not None:
        try:
            ov = int(override_sec)
        except (TypeError, ValueError):
            ov = 0
        if ov > 0:
            requested = ov
    if requested is None:
        raw = str(os.environ.get("OUROBOROS_TOOL_TIMEOUT_SEC", "") or "").strip()
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    requested = v
            except ValueError:
                pass
    if requested is None:
        try:
            settings_val = int(load_settings().get("OUROBOROS_TOOL_TIMEOUT_SEC") or 0)
            if settings_val > 0:
                requested = settings_val
        except Exception:
            pass
    if requested is None:
        cfg_default = int(SETTINGS_DEFAULTS.get("OUROBOROS_TOOL_TIMEOUT_SEC") or 0)
        requested = cfg_default if cfg_default > 0 else int(default_timeout_sec)

    # 2. Per-call ceiling.
    effective = min(requested, get_per_call_timeout_ceiling_sec())

    # 3. Clamp toward the remaining task-deadline budget (60s floor when a deadline exists).
    if ctx is not None:
        remaining = deadline_remaining_sec(ctx)
        if remaining > 0:
            effective = int(max(60, min(effective, remaining * 0.5)))

    # 4. Floor at 1s.
    return max(1, int(effective))


def _describe_returncode(returncode: int, *, cwd: pathlib.Path | str | None = None,
                         binding: ResolvedResourceBinding | None = None) -> str:
    """Render a return code with signal details when applicable."""
    suffix: list[str] = []
    if int(returncode) < 0:
        signal_num = abs(int(returncode))
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        suffix.append(f"signal={signal_name}")
    if cwd is not None:
        suffix.append(f"cwd={pathlib.Path(cwd).resolve(strict=False)}")
    rendered_suffix = f" ({', '.join(suffix)})" if suffix else ""
    target_suffix = ""
    if binding is not None:
        target = [f"root={binding.root}", f"source={binding.source}"]
        if binding.skill_name:
            target.append(f"skill={binding.skill_name}")
        target_suffix = "; " + ", ".join(target)
    return f"exit_code={returncode}{rendered_suffix}{target_suffix}"


def _format_process_output(stdout: str, stderr: str, *, limit: int = 50_000) -> str:
    """Render bounded stdout/stderr sections."""
    stdout_text = str(stdout or "")
    stderr_text = str(stderr or "")
    parts: List[str] = []
    if stdout_text.strip():
        parts.append(f"STDOUT:\n{stdout_text}")
    if stderr_text.strip():
        parts.append(f"STDERR:\n{stderr_text}")
    rendered = "\n\n".join(parts) if parts else "STDOUT:\n(empty)"
    if len(rendered) > limit:
        rendered = rendered[: limit // 2] + "\n...(truncated)...\n" + rendered[-limit // 2 :]
    return rendered






















# v6.90.x (submarine unwind) — the DECLARATION-NUDGE marker, deliberately typed
# APART from the real ``ARTIFACT_OUTPUT_ERROR`` registration failure above. The
# command SUCCEEDED (exit_code=0) and this only asks for ``outputs=[...]`` to be
# declared, so its status lands in the v6.57.0 POLICY-DENIAL partition
# (``_outcome_tool_errors._POLICY_DENIAL_STATUSES``) instead of degrading execution
# to ``tool_failure``. The submarine wave-3 incident was exactly this: a moot nudge
# on an already-registered artifact fed the failure record. SSOT for both
# ``run_command`` and ``run_script`` so the two nudges cannot drift apart.
_UNDECLARED_OUTPUTS_MARKER = "⚠️ ARTIFACT_OUTPUT_UNDECLARED"


def _resolve_git_root(path: pathlib.Path) -> pathlib.Path | None:
    try:
        from ouroboros.review_state import discover_repo_root
        root = discover_repo_root(path)
        if not (root / ".git").exists():
            return None
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return root if probe.returncode == 0 and probe.stdout.strip() == "true" else None
    except Exception:
        return None


def _status_snapshot(repo_dir: pathlib.Path | None) -> list[str]:
    if repo_dir is None:
        return []
    return sorted(_get_changed_files(repo_dir))


def _shallow_listing(work_dir: pathlib.Path, cap: int = 5000) -> dict:
    """Bounded immediate-children {name: (mtime_ns, size)} snapshot of a cwd. One
    directory level, capped — NOT a recursive filesystem monitor (R5). Used to
    detect a non-git user_files cwd actually producing a top-level deliverable."""
    out: dict = {}
    try:
        with os.scandir(work_dir) as it:
            for entry in it:
                if len(out) >= cap:
                    break
                try:
                    st = entry.stat(follow_symlinks=False)
                    out[entry.name] = (int(st.st_mtime_ns), int(st.st_size))
                except OSError:
                    continue
    except OSError:
        return {}
    return out


def _user_files_run_had_effect(
    before_changed: list[str],
    after_changed: list[str],
    before_listing: dict | None,
    work_dir: pathlib.Path,
) -> bool:
    """Effect-based gate for the ARTIFACT_AUDIT_GAP nudge (R5): warn only when the
    command produced an OBSERVABLE filesystem change in the cwd, not merely
    because it ran in a user_files cwd. Git-tracked cwd (e.g. dig-direct /app) →
    a status delta (modified or new untracked file). Non-git cwd → a bounded
    shallow immediate-children snapshot delta. A read-only command (ls/cat/grep)
    changes neither and is no longer falsely flagged."""
    if after_changed != before_changed:
        return True
    if before_listing is not None:
        return _shallow_listing(work_dir) != before_listing
    return False


def _protected_runtime_dirty_paths(repo_dir: pathlib.Path) -> list[str]:
    dirty: set[str] = set()
    for cmd in (["git", "diff", "--name-only"], ["git", "diff", "--cached", "--name-only"]):
        try:
            res = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                dirty.update(rel for rel in res.stdout.splitlines() if is_protected_runtime_path(rel))
        except Exception:
            pass
    return sorted(dirty)


def _restore_protected_runtime_paths(repo_dir: pathlib.Path, paths: list[str]) -> list[str]:
    restored: list[str] = []
    for rel in sorted(set(paths)):
        try:
            subprocess.run(
                ["git", "reset", "HEAD", "--", rel],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=5,
            )
            subprocess.run(
                ["git", "checkout", "--", rel],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=5,
            )
            restored.append(rel)
        except Exception:
            pass
    return restored




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
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh", "fish", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"})
_ENV_REF_PATTERN = re.compile(r'\$(?:\{[A-Z][A-Z0-9_]*\}|[A-Z][A-Z0-9_]*)')




_OUTPUT_CALL_PATH_RE = r"(?:~?/[^'\"]+|[A-Za-z]:[\\/][^'\"]+|\\\\[^'\"]+)"
_OUTPUT_REDIRECT_PATH_RE = r"(?:~?/[^\s;|&'\"]+|[A-Za-z]:[\\/][^\s;|&'\"]+|\\\\[^\s;|&'\"]+)"
_EMBEDDED_OUTPUT_PATH_RE = re.compile(_OUTPUT_CALL_PATH_RE)
_USER_FILE_WRITE_CALL_RE = re.compile(
    rf"(?:write_text|write_bytes)\s*\(\s*['\"](?P<path>{_OUTPUT_CALL_PATH_RE})['\"]",
    re.I,
)
_USER_FILE_OPEN_WRITE_CALL_RE = re.compile(
    rf"open\s*\(\s*['\"](?P<path>{_OUTPUT_CALL_PATH_RE})['\"]\s*,\s*['\"][^'\"]*[wax+][^'\"]*['\"]",
    re.I,
)
_USER_FILE_REDIRECT_RE = re.compile(
    rf"(?:^|\s)(?:>|>>|1>|2>|&>)\s*(?:['\"](?P<quoted>{_OUTPUT_REDIRECT_PATH_RE})['\"]|(?P<bare>{_OUTPUT_REDIRECT_PATH_RE}))"
)

# Undeclared-output stat filter (v6.56.0): a text-scan candidate counts as a real write only if it
# exists with mtime >= command_start - this slack (covers coarse FS mtime granularity, e.g. FAT 2s).
_OUTPUT_STAT_SLACK_SEC = 2.0

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


def _resolve_scratch_abs(scratch: List[str] | None, work_dir) -> list[pathlib.Path]:
    """Resolve declared ephemeral `scratch=[...]` paths to absolute host paths (relative ones
    against the command cwd). Blank entries dropped. (v6.52.2)"""
    base = pathlib.Path(work_dir).resolve(strict=False) if work_dir else None
    out: list[pathlib.Path] = []
    for raw in (scratch or []):
        text = str(raw or "").strip()
        if not text:
            continue
        p = pathlib.Path(text).expanduser()
        out.append((p if p.is_absolute() else ((base / p) if base is not None else p)).resolve(strict=False))
    return out


def _scratch_safety_reason(ctx: ToolContext, scratch_abs: list[pathlib.Path], work_dir, repo_root) -> str:
    """Pre-exec gate for declared scratch (v6.52.2; v6.56.0 adoptable): the cwd must be inside a git
    worktree (so the git-untracked proof is meaningful and the patch-exclusion contract applies), and
    each path must be CONFINED to the command cwd and git-UNTRACKED — so an ephemeral verification
    file can never mask a real TRACKED edit. Returns a refusal reason or ''.

    v6.56.0: a path is no longer blocked merely because it already EXISTS. Re-declaring the same
    throwaway across commands, or adopting an untracked file created earlier in THIS task (e.g. via
    write_file, or a prior command), is a normal verification loop — the git-tracked check still
    blocks masking a real edit, and headless patch exclusion stays sha-gated (a later real rewrite
    diverges the sha and is NOT dropped). On adoption we record the current sha through the SSOT
    writer so the manifest reflects the adopted state at declaration time."""
    if not scratch_abs:
        return ""
    if repo_root is None:
        # No git worktree at the cwd: we cannot prove a path is git-untracked, and there is no
        # workspace patch to exclude it from — so scratch is not meaningful here.
        return "scratch requires a git-worktree cwd (it is for in-repo verification); use outputs= for a deliverable"
    base = pathlib.Path(work_dir).resolve(strict=False) if work_dir else None
    tracked: set[str] = set()
    try:
        res = subprocess.run(["git", "ls-files"], cwd=str(repo_root), capture_output=True, text=True, timeout=20)
        if res.returncode == 0:
            root = pathlib.Path(repo_root).resolve(strict=False)
            tracked = {str((root / line.strip()).resolve(strict=False)) for line in (res.stdout or "").splitlines() if line.strip()}
    except Exception:
        tracked = set()
    adopt: dict = {}
    for cand in scratch_abs:
        if base is not None and not (cand == base or path_is_relative_to(cand, base)):
            return f"scratch path escapes the command cwd ({base}): {cand}"
        if str(cand) in tracked:
            return f"scratch path is git-tracked — not a throwaway (use outputs=, or edit it as a real change): {cand}"
        # A directory can neither be sha-fingerprinted nor excluded from the patch
        # file-by-file — silently adopting one would let its contents leak into the
        # deliverable while SCRATCH_REMAINS nags forever. Refuse explicitly.
        try:
            if cand.is_dir():
                return f"scratch path is a directory — declare the throwaway FILES, not their parent dir: {cand}"
        except OSError:
            pass
        # Adoptable: an existing untracked+confined file — record its current sha now so a
        # re-declaration is idempotent and the adopted state is captured at declaration.
        try:
            if cand.is_file():
                adopt[str(cand)] = sha256(cand.read_bytes()).hexdigest()
        except OSError:
            continue
    if adopt:
        record_task_scratch(ctx, adopt)
    return ""


def _record_scratch_fingerprints(ctx: ToolContext, scratch_abs: list[pathlib.Path]) -> None:
    """Record sha256 of declared scratch files that exist NOW (post-exec) so workspace patch
    capture can exclude them while they still match. Called on EVERY exit path — normal, nonzero,
    timeout, and exception — so a file created by a command that then times out is still managed
    (v6.52.2). Fail-soft; only records files that currently exist."""
    if not scratch_abs:
        return
    fingerprints: dict = {}
    for sp in scratch_abs:
        try:
            if sp.is_file():
                fingerprints[str(sp)] = sha256(sp.read_bytes()).hexdigest()
        except OSError:
            continue
    if fingerprints:
        record_task_scratch(ctx, fingerprints)


def _mentioned_user_file_outputs_without_declaration(
    ctx: ToolContext,
    cmd: List[str],
    outputs: List[str] | None,
    scratch_abs: list[pathlib.Path] | None = None,
    command_start_ts: float | None = None,
) -> list[str]:
    """Best-effort audit for commands that write absolute user_files without outputs. Declared
    ephemeral `scratch` paths (v6.52.2) are exempt.

    v6.56.0: the text scan only produces CANDIDATES; a candidate is confirmed a written deliverable
    only if it now exists on disk with a fresh mtime (>= command start). This grounds the guard in
    real filesystem effects instead of string shape, so import strings (`/http`, `/zap`), CLI flags
    (`-run TestX`), and heredoc bodies no longer trip a false ARTIFACT_OUTPUT_ERROR. Pass
    `command_start_ts` on the POST-exec call (run_command, and the run_script body audit); when it is
    None the stat filter is skipped (candidate list returned as before). Known limitations (advisory
    audit, both acceptable): (1) `cp -p` / `tar -x` preserve mtime, so such a copied deliverable is
    not flagged (false negative); (2) a file created by a PRIOR tool call within the ~2s mtime slack
    of this command's start and merely MENTIONED here can trip the mtime floor (false positive) — the
    slack is deliberate to cover coarse FS mtime granularity. In workspace mode, candidates under the
    active workspace are skipped — real /app edits are captured by the workspace patch, not undeclared
    user_files deliverables."""

    if outputs:
        return []
    scratch_set = {str(p) for p in (scratch_abs or [])}
    mtime_floor = (command_start_ts - _OUTPUT_STAT_SLACK_SEC) if command_start_ts is not None else None
    workspace_root: pathlib.Path | None = None
    if bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        try:
            workspace_root = active_repo_dir_for(ctx).resolve(strict=False)
        except Exception:
            workspace_root = None
    mentioned: list[str] = []
    for token in shell_argv_with_inline(cmd):
        token_text = str(token)
        token_lower = token_text.lower()
        redirect_paths = [
            match.group("quoted") or match.group("bare")
            for match in _USER_FILE_REDIRECT_RE.finditer(token_text)
        ]
        has_write_open = bool(_USER_FILE_OPEN_WRITE_CALL_RE.search(token_text))
        if not redirect_paths and not has_write_open and not any(marker in token_lower for marker in ("write_text", "write_bytes", ".write(", "writefile", "createwritestream")):
            continue
        candidates = embedded_absolute_path_tokens(str(token))
        candidates.extend(_EMBEDDED_OUTPUT_PATH_RE.findall(str(token)))
        candidates.extend(match.group("path") for match in _USER_FILE_WRITE_CALL_RE.finditer(str(token)))
        candidates.extend(match.group("path") for match in _USER_FILE_OPEN_WRITE_CALL_RE.finditer(str(token)))
        candidates.extend(redirect_paths)
        for candidate in candidates:
            try:
                path = pathlib.Path(candidate).expanduser().resolve(strict=False)
            except Exception:
                continue
            try:
                user_root = resource_root_path(ctx, "user_files")
            except Exception:
                continue
            if not path_is_relative_to(path, user_root):
                continue
            if user_files_path_block_reason(ctx, path):
                continue
            if workspace_root is not None and path_is_relative_to(path, workspace_root):
                continue  # real active-workspace edit — captured by the workspace patch, not a user_files deliverable
            path_text = str(path)
            if path_text in scratch_set:
                continue  # declared ephemeral scratch (v6.52.2) — not an undeclared deliverable
            if path_text in mentioned:
                continue
            if mtime_floor is not None:
                # Confirm a real filesystem write: the candidate must exist now with a fresh mtime.
                try:
                    if not (path.is_file() and path.stat().st_mtime >= mtime_floor):
                        continue
                except OSError:
                    continue
            mentioned.append(path_text)
    return mentioned


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
                return (
                    '⚠️ SHELL_CMD_ERROR: `{ ...; }` is a shell brace group, which run_command '
                    'cannot execute directly (it runs argv without a shell). Wrap it in a shell:\n'
                    '  run_command(cmd=["sh", "-c", "{ cmd1; cmd2; }"])'
                )
            if stripped[:1] in ("[", "{") and not is_posix_test_cmd:
                return (
                    '⚠️ SHELL_ARG_ERROR: `cmd` looks like a JSON/Python list literal '
                    'but failed to parse cleanly (likely an escape or quote-mismatch '
                    'issue). Pass cmd as an actual array, not a stringified array.\n\n'
                    'Correct usage:\n'
                    '  run_command(cmd=["git", "log", "--oneline", "-10"])\n\n'
                    'Wrong usage (the failure that brought you here):\n'
                    '  run_command(cmd=\'["git", "log", "--oneline", "-10"]\')\n\n'
                    'For reading files, prefer `read_file`.\n'
                    'For searching code, prefer `search_code`.'
                )
            try:
                parts = shlex.split(cmd)
                if parts:
                    recovered = parts
            except ValueError:
                pass
        if recovered is not None:
            cmd = recovered
        else:
            return (
                '⚠️ SHELL_ARG_ERROR: `cmd` must be a JSON array of strings, not a plain string.\n\n'
                'Correct usage:\n'
                '  run_command(cmd=["grep", "-r", "pattern", "path/"])\n'
                '  run_command(cmd=["python", "-c", "print(1+1)"])\n\n'
                'Wrong usage:\n'
                '  run_command(cmd="grep -r pattern path/")\n\n'
                'For reading files, prefer `read_file`.\n'
                'For searching code, prefer `search_code`.'
            )

    if not isinstance(cmd, list):
        return "⚠️ SHELL_ARG_ERROR: cmd must be a list of strings."
    cmd = [str(x) for x in cmd]

    executable_name = pathlib.Path(cmd[0]).name.lower() if cmd else ""
    if executable_name not in _SHELL_INTERPRETERS:
        for arg in cmd:
            match = _ENV_REF_PATTERN.search(arg)
            if match:
                return (
                    f'⚠️ SHELL_ENV_ERROR: Found literal env reference "{match.group(0)}" in cmd array. '
                    "run_command executes argv directly, so shell variables are not expanded. "
                    'Use ["sh", "-c", "..."] if you intentionally need shell expansion, '
                    "or read the environment variable inside the called program."
                )

    if cmd and cmd[0] in _SHELL_BUILTINS:
        if cmd[0] == "cd":
            return (
                '⚠️ SHELL_CMD_ERROR: "cd" is a shell builtin, not an executable. '
                'Use the "cwd" parameter instead: '
                'run_command(cmd=["git", "log"], cwd="/target/dir")'
            )
        return (
            f'⚠️ SHELL_CMD_ERROR: "{cmd[0]}" is a shell builtin and cannot '
            'be executed directly via subprocess. '
            'Use ["sh", "-c", "your command"] if you need shell builtins.'
        )

    cmd, autocorrect_note = _maybe_autocorrect_grep_backslash_pipe(cmd)

    found_ops = _SHELL_OPERATORS.intersection(cmd)
    if found_ops:
        op = sorted(found_ops)[0]
        return (
            f'⚠️ SHELL_CMD_ERROR: Shell operator "{op}" found in cmd array. '
            'Subprocess does not interpret shell syntax. '
            'Options: (1) Split into separate run_command calls. '
            '(2) For pipes/chaining: ["sh", "-c", "cmd1 && cmd2"]'
        )

    # Glued redirects bypass the standalone-operator set but remain shell syntax.
    for arg in cmd:
        if _GLUED_REDIRECT_RE.match(arg):
            return (
                f'⚠️ SHELL_CMD_ERROR: Shell redirection "{arg}" found in cmd array. '
                'Subprocess does not interpret shell syntax, so it reaches the '
                'program as a literal argument. '
                'Use ["sh", "-c", "your command with redirects"] for redirection.'
            )

    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, operation="shell", process_cwd=cwd, bucket=bucket, skill_name=skill_name,
        )
        work_dir = pathlib.Path(binding.target_path)
        cwd_root = binding.root
        # Materialization is the execution boundary's job, not the resolver's; the
        # executor owns it so a remote backend can create the cwd on the target.
        ensure_execution_cwd(executor_ref_from_ctx(ctx), work_dir, cwd_root=cwd_root)
    except (OSError, ValueError) as exc:
        return shell_cwd_block_message(ctx, cwd, operation="shell", error=exc)
    if not work_dir.exists() or not work_dir.is_dir():
        return (
            f"⚠️ SHELL_CWD_BLOCKED: cwd is not a directory: {work_dir}. "
            f"root={binding.root}, source={binding.source}."
        )
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
            return f"⚠️ SCRATCH_BLOCKED: {_scratch_reason}."

    timeout_sec = _resolve_effective_timeout(_RUN_SHELL_DEFAULT_TIMEOUT_SEC, ctx, override_sec=_timeout_override)
    bootstrap_process_path()
    _command_start_ts = time.time()
    try:
        if executor_covers(executor_ref_from_ctx(ctx), pathlib.Path(work_dir)):
            res = executor_execute(ctx, cmd, pathlib.Path(work_dir), timeout_sec)
        else:
            run_env = _shell_env_for_cwd(ctx, pathlib.Path(work_dir))
            res = _tracked_subprocess_run(
                cmd, cwd=str(work_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_sec,
                **({"env": run_env} if run_env is not None else {}),
            )
        # Post-run hashes exclude scratch only while its exact bytes still match.
        _record_scratch_fingerprints(ctx, scratch_abs)
        if res.returncode != 0:
            executor_note = ""
            if getattr(res, "backend_trace", None):
                executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
            if _is_search_no_match(res):
                return autocorrect_note + (
                    f"{_describe_returncode(res.returncode, cwd=work_dir, binding=binding)} (no matches)\n"
                    f"{_format_process_output(res.stdout or '', '')}"
                    f"{executor_note}"
                )
            return autocorrect_note + f"⚠️ SHELL_EXIT_ERROR: command exited with {_describe_returncode(res.returncode, cwd=work_dir, binding=binding)}.\n\n{_format_process_output(res.stdout or '', res.stderr or '')}{executor_note}"
        after_changed = _status_snapshot(repo_root)
        if after_changed != before_changed:
            # This resolved cwd may be outside the live-repo dispatcher snapshot.
            _invalidate_advisory(
                ctx,
                changed_paths=after_changed or before_changed,
                mutation_root=repo_root,
                source_tool="run_command",
            )
        undeclared_user_outputs = _mentioned_user_file_outputs_without_declaration(ctx, cmd, outputs, scratch_abs=scratch_abs, command_start_ts=_command_start_ts)
        if undeclared_user_outputs:
            # Declaration NUDGE, not a failure — see _UNDECLARED_OUTPUTS_MARKER.
            return (
                autocorrect_note
                + f"{_UNDECLARED_OUTPUTS_MARKER}: command appears to write user_files outputs "
                "without declaring outputs=[...]. Declare generated user-visible files so "
                "they are copied into the task artifact store before claiming completion. "
                f"Paths: {', '.join(undeclared_user_outputs[:5])}.\n\n"
                + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n"
                + _format_process_output(res.stdout or "", res.stderr or "")
            )
        artifact_note, artifact_failed = _register_process_outputs(
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
            return (
                autocorrect_note
                + "⚠️ ARTIFACT_OUTPUT_ERROR: command succeeded but declared output registration failed. "
                + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n"
                + f"{_format_process_output(res.stdout or '', res.stderr or '')}"
                + artifact_note
            )
        executor_note = ""
        if getattr(res, "backend_trace", None):
            executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
        return autocorrect_note + f"{_describe_returncode(0, cwd=work_dir, binding=binding)}\n{_format_process_output(res.stdout or '', res.stderr or '')}{artifact_note}{audit_note}{scratch_note}{executor_note}"
    except subprocess.TimeoutExpired:
        # Timeout-created scratch still needs its exclusion fingerprint.
        _record_scratch_fingerprints(ctx, scratch_abs)
        return (
            f"⚠️ TOOL_TIMEOUT (run_command): command exceeded the per-command timeout of {timeout_sec}s "
            f"and its subprocess tree was terminated (root={binding.root}, cwd={work_dir}). NOTE: this is the per-command "
            f"FOREGROUND timeout, NOT the task deadline. For genuinely long-running compute (training, "
            f"sampling, large builds/downloads), start it with start_service and poll "
            f"service_status/service_logs while you do other work, or pass an explicit timeout_sec=<seconds> "
            f"(up to the per-call ceiling) — and preserve a best-effort deliverable before the task deadline."
        )
    except Exception as e:
        _record_scratch_fingerprints(ctx, scratch_abs)
        return f"⚠️ SHELL_ERROR: {e}. root={binding.root}, cwd={work_dir}"


def _load_project_context(repo_dir: pathlib.Path) -> str:
    """Load governance docs for Claude Code system_prompt injection."""
    docs = [
        ("BIBLE.md", "CONSTITUTION"),
        ("docs/DEVELOPMENT.md", "DEVELOPMENT GUIDE"),
        ("docs/CHECKLISTS.md", "REVIEW CHECKLISTS"),
        ("docs/ARCHITECTURE.md", "ARCHITECTURE"),
    ]
    parts: list = []
    for relpath, label in docs:
        fpath = repo_dir / relpath
        if fpath.is_file():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                parts.append(f"## {label}\n\n{content}")
            except Exception:
                pass
    return "\n\n---\n\n".join(parts)


def _get_changed_files(repo_dir: pathlib.Path) -> list:
    """Return changed files after an edit."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return [line[3:].strip() for line in res.stdout.splitlines() if len(line) > 3 and line.strip()]
    except Exception:
        pass
    return []


def _get_diff_stat(repo_dir: pathlib.Path) -> str:
    """Return git diff --stat output."""
    try:
        res = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


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
    # ONE allowlist, not two that will disagree. The pipeline judges the RAW request so
    # the rule reaches the native route too (`tools/dispatch_policy.
    # script_interpreter_refusal`); this check stays as defence in depth for
    # non-dispatch callers, and it is the only one that sees the resolver's rewrite, so
    # it is the one that keeps the attestation clause.
    from ouroboros.tools.dispatch_policy import SCRIPT_INTERPRETERS as allowed

    resolver_attested = False
    try:
        from ouroboros.python_interpreter import PythonResolutionTrace

        resolution = getattr(ctx, "_active_python_resolution", None)
        resolver_attested = bool(
            isinstance(resolution, PythonResolutionTrace)
            and resolution.verified
            and resolution.tool == "run_script"
            and resolution.requested_interpreter in {"python", "python3"}
            and resolution.resolved_interpreter == interp
        )
    except Exception:
        resolver_attested = False
    if pathlib.PurePath(interp).name not in allowed and not resolver_attested:
        return f"⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of {sorted(allowed)}."
    body = str(script or "")
    if not body.strip():
        return "⚠️ TOOL_ARG_ERROR (run_script): script is required."
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, operation="shell", process_cwd=cwd, bucket=bucket, skill_name=skill_name,
        )
    except (OSError, ValueError) as exc:
        return shell_cwd_block_message(ctx, cwd, operation="shell", error=exc)
    # The undeclared-output audit of the script BODY (argv only carries the temp script path, so
    # _run_shell cannot see the body) is POST-exec (v6.56.0): the stat filter needs the files to
    # exist, and a pre-exec scan on not-yet-written paths would either be a no-op or false-flag
    # import strings. We resolve the body-audit scratch against the SAME effective cwd the script
    # executes in so a relatively-declared scratch path matches a user_files write in the body.
    resolved_workdir = pathlib.Path(binding.target_path)
    _scratch_abs_body = _resolve_scratch_abs(scratch, resolved_workdir)
    _body_start_ts = time.time()
    executor_active = executor_covers(executor_ref_from_ctx(ctx), resolved_workdir)
    active_workspace_script = binding.root == "active_workspace"
    if active_workspace_script:
        root = resolved_workdir / ".ouroboros" / "tmp_scripts"
    else:
        try:
            root = pathlib.Path(ctx.task_drive_root()) / "tmp_scripts"
        except Exception:
            root = pathlib.Path(ctx.drive_root) / "tmp_scripts"
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".py" if "python" in pathlib.PurePath(interp).name else ".sh"
    script_path = root / f"script_{uuid.uuid4().hex}{suffix}"
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
                script_path.unlink(missing_ok=True)
                return f"⚠️ RUN_SCRIPT_BLOCKED: executor-backed run_script could not map temp script path: {type(exc).__name__}: {exc}"
    argv = [interp, script_arg, *[str(item) for item in (args or [])]]
    try:
        result = _run_shell(
            ctx, argv, cwd=cwd, outputs=outputs, scratch=scratch,
            _resolved_binding=binding, timeout_sec=timeout_sec, timeout=timeout,
        )
    finally:
        try:
            script_path.unlink(missing_ok=True)
            script_path.parent.rmdir()
            if active_workspace_script:
                script_path.parent.parent.rmdir()
        except OSError:
            pass
    # POST-exec body audit: stat-confirmed user_files writes performed by the script
    # body itself. Runs on EVERY exit path (parity with _record_scratch_fingerprints):
    # a script that writes an undeclared deliverable and then FAILS (raise/SystemExit/
    # timeout) still leaves that file on disk, so a `⚠️` result does NOT mean "no
    # deliverable to declare" — surface both the error and the output-guard note.
    undeclared_user_outputs = _mentioned_user_file_outputs_without_declaration(
        ctx, [interp, "-c", body], outputs, scratch_abs=_scratch_abs_body, command_start_ts=_body_start_ts,
    )
    audit_note = ""
    if undeclared_user_outputs:
        # Same declaration NUDGE class as run_command's — see _UNDECLARED_OUTPUTS_MARKER.
        audit_note = (
            f"{_UNDECLARED_OUTPUTS_MARKER}: run_script wrote user_files without declaring outputs: "
            + ", ".join(undeclared_user_outputs)
            + ". Re-run with outputs=[...] or write the canonical deliverable via root=artifact_store."
        )
    if str(result).lstrip().startswith("⚠️"):
        tail = f"\n{audit_note}" if audit_note else ""
        return f"{result}{tail}\n# script_path={script_path}"
    if audit_note:
        return f"{audit_note}\n# script_path={script_path}"
    return f"# script_path={script_path}\n{result}"


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("run_command", {
            "name": "run_command",
            "description": (
                "Run a foreground bounded command in an allowed resource-root cwd. Returns stdout+stderr. "
                "Every result header echoes the resolved cwd. "
                "cmd MUST be an array of strings, never a single shell-style "
                "string. Use cwd= for working directory; cd is rejected. "
                "For pipes/chaining use [\"sh\", \"-c\", \"cmd1 && cmd2\"]. "
                "Prefer the dedicated tools where one fits: read_file (not cat/head/sed-as-reader), "
                "search_code/query_code (not grep/find-as-search), write_file/edit_text (not sed/echo-redirect)."
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
	                "cwd": {"type": "string", "default": "", "description": "Omit for active_workspace; use system_repo[/subdir] for Ouroboros or skill_payload[/subdir] with bucket+skill_name for a skill. Existing task_drive, artifact_store, user_files and authorized absolute cwd forms remain available; use cwd instead of the rejected cd builtin."},
	                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "user_repo"], "description": "Physical skill location for cwd=skill_payload[/subdir]."},
	                "skill_name": {"type": "string", "description": "Exact skill identity for cwd=skill_payload[/subdir]."},
	                "outputs": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": "Generated file paths to copy/register into the task artifact store after success.",
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
	                "timeout": {
	                    "type": "integer",
	                    "description": "Alias for timeout_sec (per-call timeout in seconds).",
	                },
	            }, "required": ["cmd"]},
        }, _run_shell, is_code_tool=True, timeout_sec=_RUN_SHELL_DEFAULT_TIMEOUT_SEC, mutates_worktree=True),
        ToolEntry("run_script", {
            "name": "run_script",
            "description": (
                "Run a short task-scoped temporary script with a declared interpreter. "
                "Use for multi-line diagnostics or harness helpers; generated script files live under the task drive. "
                "The underlying command result echoes the resolved cwd."
            ),
            "parameters": {"type": "object", "properties": {
                "script": {"type": "string"},
	                "interpreter": {"type": "string", "enum": ["python", "python3", "bash", "sh", "node", "ruby"], "default": "python3"},
	                "args": {"type": "array", "items": {"type": "string"}, "default": []},
	                "cwd": {"type": "string", "default": "", "description": "Omit for active_workspace; use system_repo[/subdir] for Ouroboros or skill_payload[/subdir] with bucket+skill_name for a skill."},
	                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "user_repo"], "description": "Physical skill location for cwd=skill_payload[/subdir]."},
	                "skill_name": {"type": "string", "description": "Exact skill identity for cwd=skill_payload[/subdir]."},
	                "outputs": {
	                    "type": "array",
	                    "items": {"type": "string"},
	                    "default": [],
	                    "description": "Generated file paths to copy/register into the task artifact store after success.",
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
	                "timeout": {
	                    "type": "integer",
	                    "description": "Alias for timeout_sec (per-call timeout in seconds).",
	                },
	            }, "required": ["script"]},
        }, _run_script, is_code_tool=True, timeout_sec=_RUN_SHELL_DEFAULT_TIMEOUT_SEC, mutates_worktree=True),
    ]
