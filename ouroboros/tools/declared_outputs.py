"""DECLARED OUTPUTS: what a process said it would write, and what it actually wrote.

A task declares its outputs before it runs; this decides where those may live, takes a
bounded fingerprint of each before and after, and reports what changed. Two rules make
it more than bookkeeping.

The first is that a declared output is not a licence: the path is judged against the
export policy the same way any other read is, so declaring `~/.ssh` as an output does
not make its bytes readable. The second is the bound — a directory output is walked
with a file-count and byte cap, because an unbounded walk of a path the model chose is
a denial-of-service the task can spell in one argument.

Extracted from `tools/shell`, which is about RUNNING a process. This is about what the
process leaves behind, and the distinction matters because these guards must hold for
every producer, not only for the shell.
"""

from __future__ import annotations

import hashlib
from hashlib import sha256
import os
import pathlib
import stat
from typing import Dict, List
from ouroboros.artifacts import copy_directory_to_task_artifacts, copy_file_to_task_artifacts
from ouroboros.runtime_mode_policy import (
    is_protected_runtime_path,
)
from ouroboros.shell_parse import is_absolute_path_text
from ouroboros.tools.registry import (
    ToolContext,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    active_tool_profile,
    decide_tool_access,
    path_is_relative_to,
    resource_root_path,
    user_files_path_block_reason,
)
from ouroboros.utils import safe_relpath
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_backend_path as executor_map_backend_path


_OUTPUT_DIR_MAX_FILES = 1000


_OUTPUT_DIR_MAX_BYTES = 50 * 1024 * 1024


def _allowed_output_roots(
    ctx: ToolContext,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    binding: ResolvedResourceBinding | None = None,
) -> list[tuple[str, pathlib.Path]]:
    roots: list[tuple[str, pathlib.Path]] = []
    root_label = str(cwd_root or "cwd").strip() or "cwd"
    roots.append((root_label, pathlib.Path(work_dir).resolve(strict=False)))
    if binding is not None:
        base = pathlib.Path(binding.base_path).resolve(strict=False)
        if not any(
            path_is_relative_to(base, existing)
            and path_is_relative_to(existing, base)
            for _, existing in roots
        ):
            roots.append((binding.root, base))
    profile = active_tool_profile(ctx)
    for label in ("task_drive", "artifact_store", "user_files"):
        # A user_files output is a deliverable the command produced, so that root
        # must be WRITABLE by the active profile.  Task/artifact registration keeps
        # its existing host-owned read/copy semantics.
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


def _protected_output_source_reason(
    ctx: ToolContext,
    source: pathlib.Path,
    label: str,
    changed_paths: set[str],
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """Return a block reason for protected/control-plane output sources."""

    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        protected_artifact_reason = block_reason_for_path(
            ctx, source, "copy", binding,
        )
        if protected_artifact_reason:
            return protected_artifact_reason
    except Exception:
        pass

    # The same rule as the member check below it, and the same rule the remote
    # declared-output kernel applies — one document, asked about one name.
    if _sensitive_output_component_reason((source.name,)):
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
            if (
                binding is not None
                and binding.root == "skill_payload"
                and path_is_relative_to(source, binding.base_path)
            ):
                return ""
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
    binding: ResolvedResourceBinding | None = None,
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
    for label, root in _allowed_output_roots(ctx, work_dir, cwd_root, binding):
        if not path_is_relative_to(source, root):
            continue
        if label == "user_files":
            reason = user_files_path_block_reason(ctx, source)
            if reason:
                return None, f"protected user_files output {text}: {reason}"
        protected_reason = _protected_output_source_reason(
            ctx, source, label, changed, binding,
        )
        if protected_reason:
            return None, protected_reason
        return source, ""
    allowed = ", ".join(
        f"{label}={root}"
        for label, root in _allowed_output_roots(ctx, work_dir, cwd_root, binding)
    )
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
    binding: ResolvedResourceBinding | None = None,
) -> Dict[str, tuple[bool, int, str]]:
    snapshots: Dict[str, tuple[bool, int, str]] = {}
    for raw_item in outputs or []:
        source, block_reason = _resolve_declared_output(
            ctx,
            str(raw_item or ""),
            work_dir,
            cwd_root=cwd_root,
            changed_paths=changed_paths,
            binding=binding,
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
    binding: ResolvedResourceBinding | None = None,
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
            reason = _protected_output_source_reason(
                ctx, child.resolve(strict=False), label, changed_paths, binding,
            )
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
    binding: ResolvedResourceBinding | None = None,
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
            binding=binding,
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
                binding=binding,
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


def _sensitive_output_component_reason(parts: tuple[str, ...]) -> str:
    """The declared-output rule, asked of the ONE export-policy contract.

    This function used to carry a byte-for-byte copy of the rule tables that also
    lived in `workspace_payload_native` — which is exactly how a rule tightens on
    one placement and not the other. Cross-placement parity is now structural:
    both doors evaluate the same document, through the SAME evaluator.

    It used to ask `describe_component_exclusion`, one rule GROUP out of the ladder,
    and so did the target's declared-output door — so parity held and both were wrong
    the same way (`id_rsa` is caught by the credential-PREFIX rule the ladder applies
    and that group does not). Both now reach the same ladder, so the parity is over the
    whole document instead of over one of its groups.

    It asks `unaliased_exclusion` — the SPELLING door — because this guard judges a path
    parsed out of a command line on HOME, before anything runs and with no target
    filesystem to stat. The identity half is the target's job
    (`workspace_payload_native.collect_declared_outputs`), and a Home-side gate that
    pretended to answer it would be describing a filesystem it cannot see.
    """

    from ouroboros.export_policy_contract import (
        build_policy_document,
        unaliased_exclusion,
    )

    return unaliased_exclusion(
        "/".join(str(part) for part in parts if part),
        build_policy_document(channel="declared_output"),
    )[1]


__all__ = [
    "_OUTPUT_DIR_MAX_FILES",
    "_OUTPUT_DIR_MAX_BYTES",
    "_allowed_output_roots",
    "_protected_output_source_reason",
    "_changed_path_covers",
    "_resolve_declared_output",
    "_directory_fingerprint_from_entries",
    "_bounded_directory_fingerprint",
    "_fingerprint_output",
    "_snapshot_declared_outputs",
    "_scan_directory_output_members",
    "_register_process_outputs",
    "_tree_fingerprint",
    "_sensitive_output_component_reason",
]
