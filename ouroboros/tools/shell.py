"""Process tools: run_command and run_script."""

from __future__ import annotations

import hashlib
from hashlib import sha256
import json
import logging
import os
import pathlib
import re
import shutil
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List

from ouroboros.artifacts import artifact_store_path_block_reason, copy_directory_to_task_artifacts, copy_file_to_task_artifacts, record_task_scratch
from ouroboros.platform_layer import bootstrap_process_path, kill_process_tree, scrub_repo_from_pythonpath, subprocess_new_group_kwargs
from ouroboros.config import SETTINGS_DEFAULTS, get_runtime_mode, load_settings
from ouroboros.runtime_mode_policy import (
    core_patch_notice,
    is_protected_runtime_path,
    mode_allows_protected_write,
    protected_paths_in,
)
from ouroboros.tools.commit_gate import _invalidate_advisory
from ouroboros.shell_parse import embedded_absolute_path_tokens, is_absolute_path_text, recover_stringified_argv, shell_argv_with_inline
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.tool_access import (
    active_tool_profile,
    decide_tool_access,
    path_is_relative_to,
    resource_root_path,
    resolve_shell_cwd,
    user_files_path_block_reason,
)
from ouroboros.utils import safe_relpath, utc_now_iso, run_cmd
from ouroboros.deadline_utils import deadline_remaining_sec
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.contracts.skill_payload_policy import (
    SKILL_PAYLOAD_CONTROL_DIRNAMES,
    SKILL_PAYLOAD_CONTROL_FILENAMES,
    SkillPayloadPathError,
    cross_skill_redirect_error,
    decide_payload_short_form,
    resolve_skill_payload_target,
)
from ouroboros.workspace_executor import execute as executor_execute
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_backend_path as executor_map_backend_path
from ouroboros.workspace_executor import map_host_path as executor_map_host_path

log = logging.getLogger(__name__)
# Tracked process groups let panic kill descendant trees too.
_active_subprocesses: set = set()
_subprocess_lock = threading.Lock()
_RUN_SHELL_DEFAULT_TIMEOUT_SEC = 360
_CONTROL_DIR_BACKUP_MAX_BYTES = 5 * 1024 * 1024
_OUTPUT_DIR_MAX_FILES = 1000
_OUTPUT_DIR_MAX_BYTES = 50 * 1024 * 1024

def _tracked_subprocess_run(cmd, **kwargs):
    """subprocess.run replacement with process-tree tracking. When capturing TEXT
    output, decode tolerantly (errors='replace') so binary stdout/stderr (a MIPS
    interpreter, a DOOM framebuffer, raw bytes) surfaces as readable text instead
    of raising UnicodeDecodeError and collapsing the whole call into a
    shell_error."""
    timeout = kwargs.pop("timeout", None)
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    kwargs.update(subprocess_new_group_kwargs())
    kwargs.setdefault("stdin", subprocess.DEVNULL)
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


def _describe_returncode(returncode: int, *, cwd: pathlib.Path | str | None = None) -> str:
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
    return f"exit_code={returncode}{rendered_suffix}"


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


def _allowed_output_roots(ctx: ToolContext, work_dir: pathlib.Path, cwd_root: str = "") -> list[tuple[str, pathlib.Path]]:
    roots: list[tuple[str, pathlib.Path]] = []
    root_label = str(cwd_root or "cwd").strip() or "cwd"
    roots.append((root_label, pathlib.Path(work_dir).resolve(strict=False)))
    profile = active_tool_profile(ctx)
    for label in ("task_drive", "artifact_store", "user_files"):
        # An output is a deliverable the command PRODUCED, so its root must be WRITABLE by the
        # profile — not merely readable. (v6.52.0: workspace_task gained user_files READ for
        # attachments; that must NOT make user_files a valid run_command output destination.)
        op = "write" if label == "user_files" else "read"
        if not decide_tool_access(profile=profile, root=label, operation=op).allow:  # type: ignore[arg-type]
            continue
        try:
            root_path = resource_root_path(ctx, label)  # type: ignore[arg-type]
        except Exception:
            continue
        if not any(path_is_relative_to(root_path, existing) and path_is_relative_to(existing, root_path) for _, existing in roots):
            roots.append((label, root_path))
    return roots


def _protected_output_source_reason(ctx: ToolContext, source: pathlib.Path, label: str, changed_paths: set[str]) -> str:
    """Return a block reason for protected/control-plane output sources."""

    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        protected_artifact_reason = block_reason_for_path(ctx, source, "copy")
        if protected_artifact_reason:
            return protected_artifact_reason
    except Exception:
        pass

    name_lower = source.name.lower()
    if (
        source.name.startswith(".")
        or name_lower in _SENSITIVE_OUTPUT_NAMES
        or name_lower.endswith(_SENSITIVE_OUTPUT_SUFFIXES)
        or any(marker in name_lower for marker in _SENSITIVE_OUTPUT_MARKERS)
    ):
        return f"credential-like output {source.name} is not a deliverable artifact"

    try:
        system_repo = pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")).resolve(strict=False)
    except Exception:
        system_repo = pathlib.Path(getattr(ctx, "repo_dir")).resolve(strict=False)
    if path_is_relative_to(source, system_repo):
        try:
            rel = source.relative_to(system_repo).as_posix()
        except ValueError:
            rel = source.name
        if is_protected_runtime_path(rel):
            return f"protected repo output {rel} is not a deliverable artifact"
        if label in {"active_workspace", "system_repo"} and not _changed_path_covers(rel, changed_paths):
            return f"unchanged repo output {rel} is not a generated deliverable"

    try:
        drive = pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)
        if path_is_relative_to(source, drive):
            task_drive = resource_root_path(ctx, "task_drive")
            artifact_store = resource_root_path(ctx, "artifact_store")
            if not (path_is_relative_to(source, task_drive) or path_is_relative_to(source, artifact_store)):
                return "runtime data output is not a user deliverable; use task_drive or artifact_store"
    except Exception:
        pass

    return ""


def _changed_path_covers(rel: str, changed_paths: set[str]) -> bool:
    clean = str(rel or "").strip().strip("/")
    if not clean:
        return False
    for item in changed_paths or set():
        path = str(item or "").strip().strip("/")
        if path == clean or path.startswith(clean + "/") or clean.startswith(path + "/"):
            return True
    return False


def _resolve_declared_output(
    ctx: ToolContext,
    raw_item: str,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
) -> tuple[pathlib.Path | None, str]:
    text = str(raw_item or "").strip()
    if not text:
        return None, "empty output path"
    raw = pathlib.Path(text).expanduser()
    executor_ref = executor_ref_from_ctx(ctx)
    # is_absolute_path_text (not Path.is_absolute) so a backend output path like
    # "/workspace/out.txt" maps through the executor on Windows too, where
    # Path.is_absolute() is False for drive-less roots.
    if executor_ref is not None and is_absolute_path_text(text) and not text.startswith("~"):
        try:
            source = executor_map_backend_path(executor_ref, text)
        except ValueError:
            source = raw.resolve(strict=False)
    elif is_absolute_path_text(text) or text.startswith("~"):
        source = raw.resolve(strict=False)
    else:
        source = (pathlib.Path(work_dir) / safe_relpath(text)).resolve(strict=False)
    changed = changed_paths or set()
    for label, root in _allowed_output_roots(ctx, work_dir, cwd_root):
        if not path_is_relative_to(source, root):
            continue
        if label == "user_files":
            reason = user_files_path_block_reason(ctx, source)
            if reason:
                return None, f"protected user_files output {text}: {reason}"
        protected_reason = _protected_output_source_reason(ctx, source, label, changed)
        if protected_reason:
            return None, protected_reason
        return source, ""
    allowed = ", ".join(f"{label}={root}" for label, root in _allowed_output_roots(ctx, work_dir, cwd_root))
    return None, f"output escapes allowed artifact roots: {text}; allowed_roots: {allowed}"


def _directory_fingerprint_from_entries(root: pathlib.Path, entries: list[tuple[str, os.stat_result, pathlib.Path]]) -> str:
    digest = hashlib.sha256()
    for rel, st, child in sorted(entries, key=lambda item: item[0]):
        digest.update(rel.encode("utf-8", errors="replace"))
        digest.update(str(st.st_mode).encode())
        digest.update(str(st.st_size).encode())
        digest.update(str(st.st_mtime_ns).encode())
        if stat.S_ISLNK(st.st_mode):
            try:
                digest.update(os.readlink(child).encode("utf-8", errors="replace"))
            except OSError:
                pass
    return digest.hexdigest()


def _bounded_directory_fingerprint(path: pathlib.Path) -> tuple[bool, int, str]:
    root = pathlib.Path(path).resolve(strict=False)
    total = 0
    entries: list[tuple[str, os.stat_result, pathlib.Path]] = []
    try:
        for child in root.rglob("*"):
            try:
                st = child.lstat()
            except OSError:
                continue
            try:
                rel = child.resolve(strict=False).relative_to(root).as_posix()
            except ValueError:
                rel = safe_relpath(str(child))
            entries.append((rel, st, child))
            if child.is_file() and not child.is_symlink():
                total += st.st_size
            if len(entries) > _OUTPUT_DIR_MAX_FILES:
                return True, total, f"too_many_entries:{_OUTPUT_DIR_MAX_FILES}"
            if total > _OUTPUT_DIR_MAX_BYTES:
                return True, total, f"too_many_bytes:{_OUTPUT_DIR_MAX_BYTES}"
        return True, total, _directory_fingerprint_from_entries(root, entries)
    except OSError:
        return False, -1, ""


def _fingerprint_output(path: pathlib.Path) -> tuple[bool, int, str]:
    try:
        if path.is_dir():
            return _bounded_directory_fingerprint(path)
        if not path.is_file():
            return False, -1, ""
        raw = path.read_bytes()
        return True, len(raw), sha256(raw).hexdigest()
    except OSError:
        return False, -1, ""


def _snapshot_declared_outputs(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
) -> Dict[str, tuple[bool, int, str]]:
    snapshots: Dict[str, tuple[bool, int, str]] = {}
    for raw_item in outputs or []:
        source, block_reason = _resolve_declared_output(
            ctx,
            str(raw_item or ""),
            work_dir,
            cwd_root=cwd_root,
            changed_paths=changed_paths,
        )
        if source is not None and not block_reason:
            snapshots[str(source)] = _fingerprint_output(source)
    return snapshots


def _scan_directory_output_members(
    ctx: ToolContext,
    source: pathlib.Path,
    *,
    label: str,
    changed_paths: set[str],
) -> tuple[list[pathlib.Path], int, str]:
    root = pathlib.Path(source).resolve(strict=False)
    members: list[pathlib.Path] = []
    dir_size = 0
    try:
        for child in root.rglob("*"):
            if child.is_symlink():
                continue
            if not child.is_file():
                continue
            members.append(child)
            try:
                dir_size += child.stat().st_size
            except OSError:
                pass
            try:
                rel_parts = child.resolve(strict=False).relative_to(root).parts
            except ValueError:
                rel_parts = child.parts
            component_reason = _sensitive_output_component_reason(rel_parts)
            if component_reason:
                return [], dir_size, f"{child}: {component_reason}"
            reason = _protected_output_source_reason(ctx, child.resolve(strict=False), label, changed_paths)
            if reason:
                return [], dir_size, f"{child}: {reason}"
            if len(members) > _OUTPUT_DIR_MAX_FILES:
                return [], dir_size, f"{source}: directory output has more than {_OUTPUT_DIR_MAX_FILES} files"
            if dir_size > _OUTPUT_DIR_MAX_BYTES:
                return [], dir_size, f"{source}: directory output exceeds {_OUTPUT_DIR_MAX_BYTES} bytes"
    except OSError as exc:
        return [], dir_size, f"{source}: {type(exc).__name__}: {exc}"
    return sorted(members, key=lambda item: item.as_posix()), dir_size, ""


def _register_process_outputs(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
    before_outputs: Dict[str, tuple[bool, int, str]] | None = None,
) -> tuple[str, bool]:
    """Copy declared command outputs into the task artifact store."""

    if not outputs:
        return "", False
    notes: list[str] = []
    failed = False
    registered = False  # at least one canonical artifact record was actually created
    for raw_item in outputs:
        text = str(raw_item or "").strip()
        source, block_reason = _resolve_declared_output(
            ctx,
            text,
            work_dir,
            cwd_root=cwd_root,
            changed_paths=changed_paths,
        )
        if block_reason:
            notes.append(block_reason)
            failed = True
            continue
        if source is None:
            notes.append(f"invalid output: {text}")
            failed = True
            continue
        if not source.exists():
            notes.append(f"missing output: {text}")
            failed = True
            continue
        before = (before_outputs or {}).get(str(source), (False, -1, ""))
        after = _fingerprint_output(source)
        if before[0] and before == after:
            # Present-but-unchanged is NOT a failure (a deterministic re-run, or a
            # command that re-verifies an existing artifact): note it cosmetically
            # and skip re-registration. "Did it actually work?" lives on the
            # objective/review axis, not the tool-execution axis (Bible P5). A
            # genuinely MISSING declared output above stays a blocking failure.
            notes.append(f"unchanged output (cosmetic): {text}")
            continue
        if source.is_file():
            try:
                record = copy_file_to_task_artifacts(ctx, source, kind="process_output")
            except OSError as exc:
                notes.append(f"failed output copy {text}: {type(exc).__name__}: {exc}")
                failed = True
                continue
            if record:
                registered = True
                notes.append(
                    f"registered output {source} -> artifact_store:{record.get('name')} "
                    f"sha256={str(record.get('sha256') or '')[:12]}"
                )
            else:
                notes.append(f"failed output copy {text}: source is not a regular file")
                failed = True
        elif source.is_dir():
            dir_members, _dir_size, blocked_member = _scan_directory_output_members(
                ctx,
                source,
                label=str(cwd_root or "cwd"),
                changed_paths=changed_paths or set(),
            )
            if blocked_member:
                notes.append(f"blocked directory output: {blocked_member}")
                failed = True
                continue
            try:
                records = copy_directory_to_task_artifacts(
                    ctx,
                    source,
                    kind="process_output_directory",
                    member_paths=dir_members,
                )
            except OSError as exc:
                notes.append(f"failed directory output copy {text}: {type(exc).__name__}: {exc}")
                failed = True
                continue
            if records:
                registered = True
                names = ", ".join(str(record.get("name") or "") for record in records)
                notes.append(f"registered directory output {source} -> artifact_store:{names}")
            else:
                notes.append(f"failed directory output copy {text}: no artifact records")
                failed = True
        else:
            notes.append(f"skipped non-file output: {text}")
            failed = True
    if not notes:
        return "", False
    # Distinguish a CANONICAL artifact registration from a cosmetic-only note (e.g.
    # an unchanged declared output): the downstream artifact_registered detector
    # (outcomes.py / loop_tool_execution.py) keys on the exact "ARTIFACT_OUTPUTS"
    # marker, so a cosmetic note must NOT borrow it — else an unchanged output reads
    # as a real registration / false recovery signal. "ARTIFACT_OUTPUT_NOTE" does
    # not contain the "ARTIFACT_OUTPUTS" substring, so it is correctly ignored.
    if failed:
        prefix = "⚠️ ARTIFACT_OUTPUT_ERROR"
    elif registered:
        prefix = "ARTIFACT_OUTPUTS"
    else:
        prefix = "ARTIFACT_OUTPUT_NOTE"
    return "\n\n" + prefix + ":\n" + "\n".join(f"- {note}" for note in notes), failed


def _executor_can_run_cwd(ctx: ToolContext, work_dir: pathlib.Path) -> bool:
    executor_ref = executor_ref_from_ctx(ctx)
    if executor_ref is None:
        return False
    try:
        executor_map_host_path(executor_ref, pathlib.Path(work_dir).resolve(strict=False))
        return True
    except Exception:
        return False


def _resolve_git_root(path: pathlib.Path) -> pathlib.Path | None:
    try:
        from ouroboros.review_state import discover_repo_root
        root = discover_repo_root(path)
        return root if (root / ".git").exists() else None
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


def _tree_fingerprint(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    root = pathlib.Path(path)
    if not root.exists():
        return ""
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            st = child.lstat()
        except OSError:
            continue
        try:
            rel = child.relative_to(root).as_posix()
        except ValueError:
            rel = safe_relpath(str(child))
        digest.update(rel.encode("utf-8", errors="replace"))
        digest.update(str(st.st_mode).encode())
        digest.update(str(st.st_size).encode())
        digest.update(str(st.st_mtime_ns).encode())
        if stat.S_ISLNK(st.st_mode):
            try:
                digest.update(os.readlink(child).encode("utf-8", errors="replace"))
            except OSError:
                pass
    return digest.hexdigest()


def _snapshot_skill_control_paths(payload_root: pathlib.Path) -> Dict[pathlib.Path, Any]:
    snapshots: Dict[pathlib.Path, Any] = {}
    root = pathlib.Path(payload_root).resolve(strict=False)
    control_file_names = set(SKILL_PAYLOAD_CONTROL_FILENAMES) | {"SKILL.openclaw.md"}
    existing_names: set[str] = set()
    try:
        existing_names = {child.name for child in root.iterdir() if child.name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES}
    except OSError:
        existing_names = set()
    for name in sorted(control_file_names | existing_names):
        path = root / name
        try:
            snapshots[path] = ("file", path.read_bytes() if path.exists() else None)
        except OSError:
            snapshots[path] = ("file", None)
    for name in SKILL_PAYLOAD_CONTROL_DIRNAMES:
        path = root / name
        backup = None
        if path.exists() and path.is_dir():
            before_fingerprint = _tree_fingerprint(path)
            try:
                total = 0
                for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
                    try:
                        total += child.lstat().st_size
                    except OSError:
                        continue
                    if total > _CONTROL_DIR_BACKUP_MAX_BYTES:
                        break
                if total <= _CONTROL_DIR_BACKUP_MAX_BYTES:
                    backup = pathlib.Path(
                        shutil.copytree(
                            path,
                            root.parent / f".ouroboros-control-backup-{uuid.uuid4().hex}" / name,
                            symlinks=True,
                        )
                    )
            except Exception:
                backup = None
            snapshots[path] = ("dir", True, before_fingerprint, backup)
        elif path.exists():
            try:
                snapshots[path] = ("dir_file", path.read_bytes())
            except OSError:
                snapshots[path] = ("dir_file", None)
        else:
            snapshots[path] = ("dir", False, "", None)
    return snapshots


def _restore_skill_control_changes(snapshots: Dict[pathlib.Path, Any]) -> list[str]:
    changed: list[str] = []
    for path, state in snapshots.items():
        kind = state[0]
        before = state[1:]
        name = path.name
        try:
            if kind == "file":
                before_bytes = before[0] if before else None
                after = path.read_bytes() if path.exists() else None
                if after != before_bytes:
                    if before_bytes is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_bytes(before_bytes)
                    changed.append(name)
            elif kind == "dir":
                existed, before_fingerprint, backup = before
                after_fingerprint = _tree_fingerprint(path) if path.exists() and path.is_dir() else None
                if not existed:
                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink(missing_ok=True)
                        changed.append(name)
                elif after_fingerprint != before_fingerprint:
                    if backup is not None and pathlib.Path(backup).exists():
                        if path.exists():
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink(missing_ok=True)
                        shutil.move(str(backup), str(path))
                    changed.append(name)
                if backup is not None:
                    try:
                        shutil.rmtree(pathlib.Path(backup).parent, ignore_errors=True)
                    except OSError:
                        pass
            elif kind == "dir_file":
                before_bytes = before[0] if before else None
                after = path.read_bytes() if path.exists() and path.is_file() else None
                if after != before_bytes:
                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink(missing_ok=True)
                    if before_bytes is not None:
                        path.write_bytes(before_bytes)
                    changed.append(name)
            elif kind == "dir_unmoved":
                before_fingerprint, temp_root = before
                after_fingerprint = _tree_fingerprint(path) if path.exists() else None
                if after_fingerprint != before_fingerprint:
                    changed.append(name)
                try:
                    shutil.rmtree(temp_root, ignore_errors=True)
                except OSError:
                    pass
        except OSError:
            changed.append(name)
    return sorted(set(changed))


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
_SENSITIVE_OUTPUT_NAMES = frozenset({".env", ".env.local", "credentials.json", "secrets.json", "token.json"})
_SENSITIVE_OUTPUT_SUFFIXES = (".key", ".pem", ".p12", ".pfx")
_SENSITIVE_OUTPUT_MARKERS = ("api_key", "apikey", "access_token", "bearer_token", "credential", "password", "refresh_token", "secret")
_SENSITIVE_OUTPUT_COMPONENT_NAMES = _SENSITIVE_OUTPUT_NAMES | frozenset({"secret", "secrets", "credential", "credentials", "token", "tokens"})


def _sensitive_output_component_reason(parts: tuple[str, ...]) -> str:
    for part in parts:
        text = str(part or "")
        if not text:
            continue
        low = text.lower()
        if text.startswith("."):
            return f"hidden/control output path component {text} is not a deliverable artifact"
        if low in _SENSITIVE_OUTPUT_COMPONENT_NAMES or low.endswith(_SENSITIVE_OUTPUT_SUFFIXES) or any(marker in low for marker in _SENSITIVE_OUTPUT_MARKERS):
            return f"credential-like output path component {text} is not a deliverable artifact"
    return ""
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
    timeout_sec: int | None = None,
    timeout: int | None = None,
) -> str:
    # Per-call timeout override (canonical timeout_sec; timeout accepted as alias).
    _timeout_override = timeout_sec if timeout_sec is not None else timeout
    if isinstance(cmd, str):
        # Recover common stringified argv mistakes before failing (shared SSOT with
        # verify_and_record via shell_parse.recover_stringified_argv — P7 DRY / P2 class-fix).
        recovered = recover_stringified_argv(cmd)
        # Malformed structured literals are not shell commands; refuse explicitly.
        if recovered is None:
            stripped = cmd.lstrip()
            is_posix_test_cmd = stripped.startswith("[ ") and stripped.rstrip().endswith(" ]")
            # A shell brace group `{ ...; }` starts with "{ " (brace + space, the
            # reserved word) — distinct from a JSON object `{"k":...}`. It is valid
            # shell, not a malformed list, so don't emit the misleading JSON error;
            # point at sh -c instead (run_command runs argv directly, no shell).
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

    # A redirect glued into one argv element (e.g. "2>/dev/null", "2>&1") slips
    # past the standalone-operator set above and reaches the program as a literal
    # arg — the program then dies cryptically ("find: 2>/dev/null: unknown
    # primary"). Surface the same actionable hint before subprocess runs.
    for arg in cmd:
        if _GLUED_REDIRECT_RE.match(arg):
            return (
                f'⚠️ SHELL_CMD_ERROR: Shell redirection "{arg}" found in cmd array. '
                'Subprocess does not interpret shell syntax, so it reaches the '
                'program as a literal argument. '
                'Use ["sh", "-c", "your command with redirects"] for redirection.'
            )

    active_repo_dir = active_repo_dir_for(ctx)
    active_root = pathlib.Path(active_repo_dir).resolve(strict=False)
    try:
        work_dir, cwd_root, allowed_roots = resolve_shell_cwd(ctx, cwd)
    except (OSError, ValueError) as exc:
        try:
            _, _, allowed_roots = resolve_shell_cwd(ctx, "")
        except Exception:
            allowed_roots = [("active_workspace", active_root)]
        roots = ", ".join(f"{name}={pathlib.Path(root).resolve(strict=False)}" for name, root in allowed_roots)
        return (
            f"⚠️ SHELL_CWD_BLOCKED: cwd escapes allowed roots: {exc}. "
            f"allowed_roots: {roots}. For user-visible files use an absolute/~/ cwd "
            "under user_files, root=artifact_store, or root=task_drive."
        )
    if not work_dir.exists() or not work_dir.is_dir():
        roots = ", ".join(f"{name}={pathlib.Path(root).resolve(strict=False)}" for name, root in allowed_roots)
        return f"⚠️ SHELL_CWD_BLOCKED: cwd is not a directory: {cwd or work_dir}. allowed_roots: {roots}"
    # Room-lens cwd disclosure (v6.61.3): the FIRST default-cwd command of a
    # folder-room chat names where it ran (the room folder is the default now)
    # so the surface change is never silent. One-shot per task; explicit cwd
    # calls are the caller's own choice and carry no note.
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
    # R5: for a non-git user_files cwd, take a bounded shallow snapshot so the
    # artifact-audit nudge can be effect-based (only when the command actually
    # produced a top-level deliverable), not fired on every read-only command.
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
    )

    # Ephemeral verification scratch (v6.52.2): a sanctioned channel for a throwaway in-workspace
    # file the agent writes, runs, and deletes (e.g. an in-package scratch test that MUST live in
    # the repo to compile). Pre-exec gate it (confined + NEW + untracked) so it cannot mask a real
    # edit, then record it so workspace patch capture EXCLUDES it. It is exempt from the
    # undeclared-output guard below and is never registered as a task artifact.
    scratch_abs = _resolve_scratch_abs(scratch, work_dir)
    if scratch_abs:
        _scratch_reason = _scratch_safety_reason(ctx, scratch_abs, pathlib.Path(work_dir), repo_root)
        if _scratch_reason:
            return f"⚠️ SCRATCH_BLOCKED: {_scratch_reason}."

    timeout_sec = _resolve_effective_timeout(_RUN_SHELL_DEFAULT_TIMEOUT_SEC, ctx, override_sec=_timeout_override)
    bootstrap_process_path()
    _command_start_ts = time.time()
    try:
        if _executor_can_run_cwd(ctx, pathlib.Path(work_dir)):
            res = executor_execute(ctx, cmd, pathlib.Path(work_dir), timeout_sec)
        else:
            run_env = _shell_env_for_cwd(ctx, pathlib.Path(work_dir))
            res = _tracked_subprocess_run(
                cmd, cwd=str(work_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_sec,
                **({"env": run_env} if run_env is not None else {}),
            )
        # Record scratch FINGERPRINTS (sha256 of declared scratch files that exist AFTER the command)
        # so workspace patch capture excludes a file ONLY while it still matches — a later real file at
        # the same path has a different sha and is NOT dropped (v6.52.2; runs regardless of exit code).
        _record_scratch_fingerprints(ctx, scratch_abs)
        if res.returncode != 0:
            executor_note = ""
            if getattr(res, "backend_trace", None):
                executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
            if _is_search_no_match(res):
                return autocorrect_note + (
                    f"{_describe_returncode(res.returncode, cwd=work_dir)} (no matches)\n"
                    f"{_format_process_output(res.stdout or '', '')}"
                    f"{executor_note}"
                )
            return autocorrect_note + f"⚠️ SHELL_EXIT_ERROR: command exited with {_describe_returncode(res.returncode, cwd=work_dir)}.\n\n{_format_process_output(res.stdout or '', res.stderr or '')}{executor_note}"
        after_changed = _status_snapshot(repo_root)
        if after_changed != before_changed:
            # Kept (nonstandard case): repo_root here is the RESOLVED cwd root,
            # which may be a workspace/skill repo the central live-repo
            # dispatcher check does not watch; this call passes precise paths.
            _invalidate_advisory(
                ctx,
                changed_paths=after_changed or before_changed,
                mutation_root=repo_root,
                source_tool="run_command",
            )
        undeclared_user_outputs = _mentioned_user_file_outputs_without_declaration(ctx, cmd, outputs, scratch_abs=scratch_abs, command_start_ts=_command_start_ts)
        if undeclared_user_outputs:
            return (
                autocorrect_note
                + "⚠️ ARTIFACT_OUTPUT_ERROR: command appears to write user_files outputs "
                "without declaring outputs=[...]. Declare generated user-visible files so "
                "they are copied into the task artifact store before claiming completion. "
                f"Paths: {', '.join(undeclared_user_outputs[:5])}.\n\n"
                + f"{_describe_returncode(0, cwd=work_dir)}\n"
                + _format_process_output(res.stdout or "", res.stderr or "")
            )
        artifact_note, artifact_failed = _register_process_outputs(
            ctx,
            outputs,
            pathlib.Path(work_dir),
            cwd_root=cwd_root,
            changed_paths=set(after_changed or []),
            before_outputs=before_outputs,
        )
        audit_note = ""
        if cwd_root == "user_files" and not outputs:
            # Audit only NON-scratch user_files effects: declared scratch is transient (not a
            # deliverable), but a command may ALSO create a real undeclared deliverable — so strip the
            # scratch paths from the change set rather than disabling the audit whenever scratch exists.
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
                + f"{_describe_returncode(0, cwd=work_dir)}\n"
                + f"{_format_process_output(res.stdout or '', res.stderr or '')}"
                + artifact_note
            )
        executor_note = ""
        if getattr(res, "backend_trace", None):
            executor_note = "\n\nEXECUTOR_TRACE:\n" + json.dumps(res.backend_trace, ensure_ascii=False, indent=2)
        return autocorrect_note + f"{_describe_returncode(0, cwd=work_dir)}\n{_format_process_output(res.stdout or '', res.stderr or '')}{artifact_note}{audit_note}{scratch_note}{executor_note}"
    except subprocess.TimeoutExpired:
        # A timed-out command may have created its declared scratch before the kill — fingerprint it
        # so headless still excludes it from the workspace patch (v6.52.2 leak-safety on the timeout path).
        _record_scratch_fingerprints(ctx, scratch_abs)
        return (
            f"⚠️ TOOL_TIMEOUT (run_command): command exceeded the per-command timeout of {timeout_sec}s "
            f"and its subprocess tree was terminated (cwd={work_dir}). NOTE: this is the per-command "
            f"FOREGROUND timeout, NOT the task deadline. For genuinely long-running compute (training, "
            f"sampling, large builds/downloads), start it with start_service and poll "
            f"service_status/service_logs while you do other work, or pass an explicit timeout_sec=<seconds> "
            f"(up to the per-call ceiling) — and preserve a best-effort deliverable before the task deadline."
        )
    except Exception as e:
        _record_scratch_fingerprints(ctx, scratch_abs)
        return f"⚠️ SHELL_ERROR: {e}. cwd={work_dir}"


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


def _run_validation(repo_dir: pathlib.Path) -> str:
    """Run basic post-edit validation."""
    agent_python = sys.executable or os.environ.get("OUROBOROS_AGENT_PYTHON") or "python3"
    try:
        res = subprocess.run(
            [agent_python, "-m", "pytest", "tests/", "--tb=line", "-q"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=60,
        )
        if res.returncode == 0:
            return "PASS: all tests passed"
        output = (res.stdout or "")[-500:]
        return f"FAIL: tests failed (exit {res.returncode})\n{output}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: validation exceeded 60s"
    except Exception as e:
        return f"ERROR: validation failed: {e}"


def _control_restore_note(restored: list[str]) -> str:
    if not restored:
        return ""
    return (
        "\n\n⚠️ SKILL_PAYLOAD_CONTROL_RESTORED: restored skill provenance/control-plane "
        "paths after claude_code_edit: "
        + ", ".join(sorted(set(restored)))
        + "."
    )


def _claude_code_executor_block_reason(ctx: ToolContext, work_dir_path: pathlib.Path) -> str:
    executor_ref = executor_ref_from_ctx(ctx)
    if executor_ref is None or executor_ref.kind != "docker_exec" or not _executor_can_run_cwd(ctx, work_dir_path):
        return ""
    return (
        "⚠️ CLAUDE_CODE_EDIT_BLOCKED: docker executor-backed workspaces route "
        "process execution through a backend, but claude_code_edit edits host "
        "paths directly. Use read/write/edit tools or run_command inside the "
        "mapped workspace until a reviewed backend-safe Claude Code path exists."
    )


def _room_default_cwd_edit_block(ctx: ToolContext, cwd: str) -> str:
    """Room guard (v6.61.3, same rule as write_file/edit_text): a DEFAULT-cwd SDK
    edit in a folder-room chat would silently edit the SYSTEM REPO while the room's
    reads show the project folder. Mutations go through promoted tasks; a
    deliberate target needs an explicit cwd. Returns "" when not a folder-room."""
    if str(cwd or "").strip():
        return ""
    try:
        from ouroboros.tool_access import project_room_lens_dir

        _room = project_room_lens_dir(ctx)
    except Exception:
        _room = None
    if _room is None:
        return ""
    return (
        f"⚠️ ROOM_WRITE_VIA_TASK: this room's files live in {_room} and are edited by "
        "PROMOTED tasks — call promote_chat_to_task (it inherits the room folder as its "
        "workspace). For a deliberate edit elsewhere, pass an explicit cwd."
    )


def _claude_code_edit(ctx: ToolContext, prompt: str, cwd: str = "", budget: float = 5.0, validate: bool = False, bucket: str = "", skill_name: str = "", outputs: List[str] | None = None) -> str:
    """Delegate SDK edits with cwd and protected-path safety hooks."""
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ CLAUDE_CODE_UNAVAILABLE: ANTHROPIC_API_KEY not set."
    if (_room_block := _room_default_cwd_edit_block(ctx, cwd)):
        return _room_block

    active_root = active_repo_dir_for(ctx).resolve(strict=False)
    system_repo_root = pathlib.Path(ctx.repo_dir).resolve(strict=False)
    existing_tc = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    workspace_mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
    workspace_task_mode = bool(workspace_mode)
    work_dir = str(active_root)
    work_dir_root = "active_workspace"
    skill_payload_root = None
    short_form_path_text = cwd if str(cwd or "").strip() else str(active_root)
    synth = None
    ignored_reason = ""
    if workspace_task_mode and not (existing_tc and existing_tc.mode == "skill_repair"):
        if str(bucket or "").strip() or str(skill_name or "").strip():
            return (
                "⚠️ CLAUDE_CODE_ERROR: skill payload short-form is unavailable in workspace mode. "
                "Use a workspace-relative cwd, or run a skill_repair task for data skill payload edits."
            )
    else:
        short_form = decide_payload_short_form(
            bucket=bucket,
            skill_name=skill_name,
            path_text=short_form_path_text,
            repo_dir=active_root,
            drive_root=pathlib.Path(ctx.drive_root),
        )
        if short_form.error:
            return f"⚠️ CLAUDE_CODE_ERROR: {short_form.error}"
        synth = short_form.constraint
        ignored_reason = short_form.ignored_reason
    redirect_err = cross_skill_redirect_error(existing_tc, synth)
    if redirect_err:
        return f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
    # Real skill_repair constraint wins; repair confinement is sticky.
    if existing_tc and existing_tc.mode == "skill_repair":
        task_constraint = existing_tc
    else:
        task_constraint = synth or existing_tc
    if task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            resolved_skill_target = resolve_skill_payload_target(
                pathlib.Path(ctx.drive_root),
                cwd or ".",
                constraint=task_constraint,
                allow_short_relative=True,
            )
            work_dir = str(resolved_skill_target.target_path)
            work_dir_root = "skill_payload"
            skill_payload_root = resolved_skill_target.payload_root
        except (SkillPayloadPathError, ValueError) as e:
            return f"⚠️ CLAUDE_CODE_ERROR: {e}"
    elif cwd and cwd.strip() not in ("", ".", "./"):
        raw_cwd = cwd.strip()
        if workspace_task_mode:
            try:
                candidate, cwd_root, allowed_roots = resolve_shell_cwd(ctx, raw_cwd)
            except (OSError, ValueError) as e:
                try:
                    _, _, allowed_roots = resolve_shell_cwd(ctx, "")
                except Exception:
                    allowed_roots = []
                allowed_text = ", ".join(f"{name}={pathlib.Path(root).resolve(strict=False)}" for name, root in allowed_roots)
                return f"⚠️ CLAUDE_CODE_ERROR: cwd escapes allowed workspace edit roots. {e}. allowed_roots: {allowed_text}. Use the active workspace, task_drive, or artifact_store for workspace tasks."
            if cwd_root not in {"active_workspace", "task_drive", "artifact_store", "user_files"}:
                return "⚠️ CLAUDE_CODE_ERROR: cwd root is unavailable for workspace task edits."
            work_dir_root = cwd_root
        else:
            try:
                resolved_skill_target = resolve_skill_payload_target(pathlib.Path(ctx.drive_root), raw_cwd)
                candidate = resolved_skill_target.target_path
                work_dir_root = "skill_payload"
                skill_payload_root = resolved_skill_target.payload_root
            except SkillPayloadPathError as exc:
                normalized_cwd = raw_cwd.replace("\\", "/").strip().lstrip("/")
                if normalized_cwd.startswith("data/skills/") or normalized_cwd.startswith("skills/"):
                    return f"⚠️ CLAUDE_CODE_ERROR: skill cwd is invalid: {exc}"
                try:
                    candidate, cwd_root, allowed_roots = resolve_shell_cwd(ctx, raw_cwd)
                    work_dir_root = cwd_root
                except (OSError, ValueError) as e:
                    try:
                        _, _, allowed_roots = resolve_shell_cwd(ctx, "")
                    except Exception:
                        allowed_roots = []
                    allowed_text = ", ".join(f"{name}={pathlib.Path(root).resolve(strict=False)}" for name, root in allowed_roots)
                    return f"⚠️ CLAUDE_CODE_ERROR: cwd escapes allowed edit roots. {e}. allowed_roots: {allowed_text}. Use an active repo/workspace cwd, an absolute/~/ user_files cwd, root=task_drive/artifact_store paths, or explicit data/skills/<bucket>/<skill>/... for skill payload edits."
        if not candidate.exists() or not candidate.is_dir():
            return f"⚠️ CLAUDE_CODE_ERROR: cwd not found or not a directory: {cwd}"
        work_dir = str(candidate)
    work_dir_path = pathlib.Path(work_dir).resolve()
    executor_block = _claude_code_executor_block_reason(ctx, work_dir_path)
    if executor_block:
        return executor_block
    skill_control_snapshots = {}
    sidecar_root = pathlib.Path(skill_payload_root).resolve() if skill_payload_root is not None else None
    if sidecar_root is not None:
        skill_control_snapshots = _snapshot_skill_control_paths(sidecar_root)

    target_repo_root = _resolve_git_root(work_dir_path)
    repo_mode = target_repo_root is not None
    if target_repo_root is None:
        target_repo_root = work_dir_path
    before_changed = _status_snapshot(target_repo_root)
    before_outputs = _snapshot_declared_outputs(ctx, outputs, work_dir_path, cwd_root=work_dir_root, changed_paths=set(before_changed or []))
    system_repo_mode = repo_mode and pathlib.Path(target_repo_root).resolve(strict=False) == system_repo_root
    runtime_mode = get_runtime_mode()
    if system_repo_mode and not mode_allows_protected_write(runtime_mode):
        protected_dirty_before = _protected_runtime_dirty_paths(target_repo_root)
        if protected_dirty_before:
            restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
            return (
                "⚠️ CORE_PROTECTION_BLOCKED: protected runtime files are already dirty; "
                "refusing claude_code_edit so existing human/operator changes are not overwritten. "
                "Resolve or commit them before delegating edits. Files: "
                + ", ".join(protected_dirty_before)
                + _control_restore_note(restored_sidecars)
            )
    invalidate_if_changed = lambda: (
        _invalidate_advisory(
            ctx,
            changed_paths=_status_snapshot(target_repo_root) or before_changed,
            mutation_root=target_repo_root,
            source_tool="claude_code_edit",
        )
        if repo_mode and _status_snapshot(target_repo_root) != before_changed
        else None
    )

    lock = _acquire_git_lock(ctx) if system_repo_mode else None
    try:
        if system_repo_mode:
            try:
                run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
            except Exception as e:
                restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
                return f"⚠️ GIT_ERROR (checkout): {e}" + _control_restore_note(restored_sidecars)

        ctx.emit_progress_fn("Delegating to Claude Agent SDK...")

        try:
            from ouroboros.gateways.claude_code import (
                DEFAULT_CLAUDE_CODE_MAX_TURNS,
                resolve_claude_code_model,
                run_edit,
            )
            model = resolve_claude_code_model()

            system_prompt = (
                f"STRICT: Only modify files inside {work_dir}. "
                f"Git branch: {ctx.branch_dev}. Do NOT commit or push.\n\n"
                + _load_project_context(system_repo_root)
            )
            write_path_blocker = None
            if work_dir_root == "user_files":
                write_path_blocker = lambda target: user_files_path_block_reason(ctx, pathlib.Path(target))
            elif work_dir_root == "artifact_store":
                write_path_blocker = lambda target: artifact_store_path_block_reason(pathlib.Path(target))

            result = run_edit(
                prompt=prompt,
                cwd=work_dir,
                model=model,
                max_turns=DEFAULT_CLAUDE_CODE_MAX_TURNS,
                budget=budget,
                system_prompt=system_prompt,
                repo_root=str(target_repo_root if repo_mode else work_dir_path),
                protect_runtime_paths=system_repo_mode,
                write_path_blocker=write_path_blocker,
            )

            result.changed_files = _get_changed_files(target_repo_root)
            result.diff_stat = _get_diff_stat(target_repo_root)

            if validate and result.success:
                result.validation_summary = _run_validation(target_repo_root)

            if result.cost_usd > 0:
                ctx.pending_events.append({
                    "type": "llm_usage",
                    "provider": "claude_agent_sdk",
                    "model": model,
                    "api_key_type": "anthropic",
                    "model_category": "claude_code",
                    "usage": result.usage or {"cost": result.cost_usd},
                    "cost": result.cost_usd,
                    "source": "claude_code_edit",
                    "ts": utc_now_iso(),
                    "category": "task",
                })

            if not result.success:
                restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
                invalidate_if_changed()
                return (
                    f"⚠️ CLAUDE_CODE_ERROR: {result.error}\n\n{result.result_text}"
                    + _control_restore_note(restored_sidecars)
                )

            restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
            if restored_sidecars:
                invalidate_if_changed()
                return (
                    "⚠️ SKILL_PAYLOAD_CONTROL_BLOCKED: claude_code_edit attempted to modify "
                    "skill provenance/control-plane paths: "
                    + ", ".join(sorted(set(restored_sidecars)))
                    + ". Created control paths and sidecar changes were reverted where possible; edit payload code files instead."
                )

            if system_repo_mode and not mode_allows_protected_write(runtime_mode):
                protected_dirty_after = _protected_runtime_dirty_paths(target_repo_root)
                if protected_dirty_after:
                    restored = _restore_protected_runtime_paths(target_repo_root, protected_dirty_after)
                    invalidate_if_changed()
                    return (
                        "⚠️ CORE_PROTECTION_BLOCKED: claude_code_edit attempted to modify "
                        "protected Ouroboros runtime files in non-pro mode. Reverted: "
                        + ", ".join(restored or protected_dirty_after)
                        + ". Switch to pro mode only after an explicit reviewed plan."
                    )

            after_changed = _status_snapshot(target_repo_root)
            if repo_mode and after_changed != before_changed:
                # Kept (nonstandard case): target_repo_root may be a skill
                # payload/workspace repo outside the central live-repo check,
                # and result.changed_files gives precise invalidation paths.
                _invalidate_advisory(
                    ctx,
                    changed_paths=result.changed_files or after_changed or before_changed,
                    mutation_root=target_repo_root,
                    source_tool="claude_code_edit",
                )

            output = result.to_tool_output()
            artifact_note, artifact_failed = _register_process_outputs(
                ctx,
                outputs,
                work_dir_path,
                cwd_root=work_dir_root,
                changed_paths=set(after_changed or []),
                before_outputs=before_outputs,
            )
            if artifact_failed:
                return (
                    "⚠️ ARTIFACT_OUTPUT_ERROR: claude_code_edit succeeded but declared "
                    "output registration failed.\n\n"
                    f"{output}"
                    f"{artifact_note}"
                    + _control_restore_note(restored_sidecars)
                )
            if artifact_note:
                output += artifact_note
            elif work_dir_root == "user_files" and not outputs:
                output += (
                    "\n\n⚠️ ARTIFACT_AUDIT_GAP: claude_code_edit ran in user_files cwd "
                    "without outputs=[...]. If it created a deliverable, register it "
                    "with outputs or write_file(root=artifact_store) before claiming it."
                )
            if system_repo_mode and mode_allows_protected_write(runtime_mode):
                protected_written = protected_paths_in(result.changed_files or after_changed)
                if protected_written:
                    output += "\n\n" + core_patch_notice(protected_written)
            if ignored_reason:
                output += f"\n\n⚠️ SKILL_SHORT_FORM_IGNORED: {ignored_reason}."
            return output

        except ImportError:
            restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
            return (
                "⚠️ CLAUDE_CODE_UNAVAILABLE: claude-agent-sdk not installed. "
                "Install: pip install 'ouroboros[claude-sdk]'"
                + _control_restore_note(restored_sidecars)
            )
        except Exception as e:
            restored_sidecars = _restore_skill_control_changes(skill_control_snapshots) if skill_control_snapshots else []
            invalidate_if_changed()
            import sys
            sdk_version = "(unknown)"
            try:
                import importlib.metadata
                sdk_version = importlib.metadata.version("claude-agent-sdk")
            except Exception:
                pass
            return (
                f"⚠️ CLAUDE_CODE_FAILED: {type(e).__name__}: {e}\n"
                f"Diagnostic: sdk_version={sdk_version}, python={sys.executable}"
                + _control_restore_note(restored_sidecars)
            )

    finally:
        if lock is not None:
            _release_git_lock(lock)


def _run_script(
    ctx: ToolContext,
    script: str,
    interpreter: str = "python3",
    args: List[str] | None = None,
    cwd: str = "",
    outputs: List[str] | None = None,
    scratch: List[str] | None = None,
    **kwargs,
) -> str:
    """Write a task-scoped temporary script and run it as a foreground command. The `timeout_sec`
    / `timeout` aliases ride in **kwargs to keep the signature within the <=8-parameter rule
    (DEVELOPMENT.md) after the `scratch` addition; both are forwarded to _run_shell unchanged."""
    timeout_sec = kwargs.get("timeout_sec")
    timeout = kwargs.get("timeout")
    interp = str(interpreter or "python3").strip()
    allowed = {"python", "python3", "bash", "sh", "node", "ruby"}
    if pathlib.PurePath(interp).name not in allowed:
        return f"⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of {sorted(allowed)}."
    body = str(script or "")
    if not body.strip():
        return "⚠️ TOOL_ARG_ERROR (run_script): script is required."
    # The undeclared-output audit of the script BODY (argv only carries the temp script path, so
    # _run_shell cannot see the body) is POST-exec (v6.56.0): the stat filter needs the files to
    # exist, and a pre-exec scan on not-yet-written paths would either be a no-op or false-flag
    # import strings. We resolve the body-audit scratch against the SAME effective cwd the script
    # executes in so a relatively-declared scratch path matches a user_files write in the body.
    _audit_cwd = str(cwd or "").strip()
    if not _audit_cwd and get_runtime_mode() == "light" and not bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        try:
            _audit_cwd = str(pathlib.Path(ctx.task_drive_root()).resolve(strict=False))
        except Exception:
            _audit_cwd = ""
    _scratch_abs_body = _resolve_scratch_abs(scratch, _audit_cwd or active_repo_dir_for(ctx))
    _body_start_ts = time.time()
    try:
        workdir, _cwd_root, _allowed = resolve_shell_cwd(ctx, cwd)
        resolved_workdir = pathlib.Path(workdir).resolve(strict=False)
    except Exception:
        if executor_ref_from_ctx(ctx) is not None:
            return f"⚠️ RUN_SCRIPT_BLOCKED: executor-backed run_script could not resolve mapped cwd {cwd!r}."
        resolved_workdir = pathlib.Path("")
    executor_active = _executor_can_run_cwd(ctx, resolved_workdir) if str(resolved_workdir) else False
    workspace_backed_script = False
    if executor_active:
        root = resolved_workdir / ".ouroboros" / "tmp_scripts"
        workspace_backed_script = True
    elif str(resolved_workdir) and bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        root = resolved_workdir / ".ouroboros" / "tmp_scripts"
        workspace_backed_script = True
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
    effective_cwd = str(cwd or "")
    if (
        not effective_cwd.strip()
        and get_runtime_mode() == "light"
        and not bool(getattr(ctx, "is_workspace_mode", lambda: False)())
    ):
        effective_cwd = str(pathlib.Path(ctx.task_drive_root()).resolve(strict=False))
    try:
        result = _run_shell(ctx, argv, cwd=effective_cwd, outputs=outputs, scratch=scratch, timeout_sec=timeout_sec, timeout=timeout)
    finally:
        if workspace_backed_script:
            try:
                script_path.unlink(missing_ok=True)
                script_path.parent.rmdir()
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
        audit_note = (
            "⚠️ ARTIFACT_OUTPUT_ERROR: run_script wrote user_files without declaring outputs: "
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
	                "cwd": {
	                    "type": "string", "default": "",
	                    "description": (
	                        "Working directory. Relative paths resolve under allowed task/workspace roots; "
	                        "absolute or ~ paths under user_files are allowed for external user deliverables. "
	                        "Use "
	                        "this instead of `cd` (which is a shell builtin "
	                        "and is rejected)."
	                    ),
	                },
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
        ToolEntry("claude_code_edit", {
            "name": "claude_code_edit",
            "description": (
                "Delegate a bounded code-editing task to the Claude Agent SDK. "
                "Use this as the strongest coding helper for substantial edits. "
                "It may edit files under cwd, never commits or pushes, and still "
                "runs through Ouroboros runtime-mode and review protections."
            ),
            "parameters": {"type": "object", "properties": {
                "prompt": {"type": "string", "description": "Precise coding task and constraints."},
	                "cwd": {
	                    "type": "string",
	                    "default": "",
	                    "description": (
	                        "Working directory under the active repo/workspace, task_drive, artifact_store, "
	                        "an external absolute/~/ user_files path, or an explicit "
	                        "data/skills/<bucket>/<skill> payload path for skill repair. "
	                        "For docker executor-backed external workspaces, mapped active_workspace cwd is "
	                        "blocked until a backend-safe Claude Code path exists; unmapped task_drive, "
	                        "artifact_store, and user_files cwd remain valid where runtime mode permits."
	                    ),
	                },
                "budget": {"type": "number", "default": 5.0},
                "validate": {"type": "boolean", "default": False},
                "bucket": {"type": "string", "default": "", "description": "Skill payload bucket — set ONLY for skill_payload edits; leave empty otherwise."},
                "skill_name": {"type": "string", "default": "", "description": "Skill slug — set ONLY for skill_payload edits; leave empty otherwise."},
                "outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Generated file paths to copy/register into the task artifact store after a successful edit.",
                },
            }, "required": ["prompt"]},
        }, _claude_code_edit, is_code_tool=True, timeout_sec=1200, mutates_worktree=True),
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
	                "cwd": {"type": "string", "default": ""},
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
