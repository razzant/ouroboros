"""Declared process outputs: resolution, fingerprints, and artifact registration.

Owns the protected and credential-like refusals that keep a control-plane or
secret path out of the deliverable store, the complete file/directory
fingerprints that decide whether a declared output actually changed, and the
copy into the task artifact store. The allowed artifact roots and the lexical
audit that notices an undeclared user_files deliverable live with the upstream
extraction in ``tools/shell_audit.py``; the handlers that call this surface
stay with ``tools/shell.py``.
"""

from __future__ import annotations

import logging

import hashlib
import os
import pathlib
import stat
from typing import Dict, List

from ouroboros.artifacts import copy_directory_to_task_artifacts, copy_file_to_task_artifacts, iter_artifact_tree, stream_artifact_file
from ouroboros.artifact_directories import iter_artifact_entries
from ouroboros.runtime_mode_policy import (
    is_protected_runtime_path,
)
from ouroboros.shell_parse import is_absolute_path_text
from ouroboros.tools.registry import (
    ToolContext,
)
from ouroboros.tools.deliverables_shell import lexical_user_files_block_reason
from ouroboros.tools.shell_audit import (
    _allowed_output_roots,
    _presence_allows_user_output,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    _deliverables_root_lexical,
    _deliverables_root_lexical_alias,
    _lexical_path_is_relative_to_casefold,
    _path_is_relative_to_casefold,
    path_is_relative_to,
    resource_root_path,
    user_files_path_block_reason,
)
from ouroboros.utils import normalized_path_text, safe_relpath, sanitize_tool_result_for_log
from ouroboros.workspace_executor import BackendPathNoMatch, executor_ref_from_ctx
from ouroboros.workspace_executor import map_backend_path as executor_map_backend_path
from ouroboros.workspace_executor import map_backend_path_lexical as executor_map_backend_path_lexical
from ouroboros.workspace_executor import map_host_path as executor_map_host_path
from ouroboros.workspace_executor_paths import (
    DockerOutputContext, DockerOutputPath, OutputResolutionUnavailable,
    prepare_docker_outputs, resolve_docker_output,
)


log = logging.getLogger(__name__)



def _protected_output_source_reason(
    ctx: ToolContext,
    source: pathlib.Path,
    label: str,
    changed_paths: set[str],
    binding: ResolvedResourceBinding | None = None,
    *,
    judge_changes: bool = True,
) -> str:
    """Return a block reason for protected/control-plane output sources.

    ``judge_changes=False`` is the pre-launch view: whether a repo output was
    generated is known only after the process, so only static refusals apply."""

    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    if mode_has_unrestricted_agency(get_runtime_mode()):
        return ""

    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        protected_artifact_reason = block_reason_for_path(
            ctx, source, "copy", binding,
        )
        if protected_artifact_reason:
            return protected_artifact_reason
    except Exception:
        pass

    # D4 (capinv-447): the credential-shape judgment is the workspace-patch SSOT
    # (permits .env.example and ordinary dotfiles like .gitignore); the exporter
    # previously blanket-blocked every dotfile and disagreed with patch policy.
    from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

    sensitive_reason = _sensitive_untracked_reason(source.name)
    if sensitive_reason:
        return f"credential-like output {source.name} is not a deliverable artifact ({sensitive_reason})"

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
        if judge_changes and label in {"active_workspace", "system_repo"} and not _changed_path_covers(rel, changed_paths):
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
    *,
    judge_changes: bool = True,
    resolved_output: DockerOutputPath | None = None,
) -> tuple[pathlib.Path | None, str]:
    text = str(raw_item or "").strip()
    if not text:
        return None, "empty output path"
    try:
        relative = normalized_path_text(text)
    except ValueError as exc:
        return None, f"declared output is not a usable path: {exc}"
    executor_ref = executor_ref_from_ctx(ctx)
    if executor_ref is not None and executor_ref.kind == "docker_exec" and not _backend_cwd(executor_ref, work_dir):
        executor_ref = None
    if not (is_absolute_path_text(text) or text.startswith("~")):
        # Outputs alone may spell ``..`` (#1797): the root walk below judges the
        # resolved location as for the absolute spelling, and the original spelling
        # is resolved, never a collapsed one, so a symlinked parent keeps its meaning.
        # A local executor runs in this host cwd (workspace_executor.execute), so the host resolves
        # its ``..`` as it does without an executor; only a backend with its own namespace remaps it.
        backend = executor_ref if executor_ref is not None and executor_ref.kind != "local" else None
        backend_cwd = _backend_cwd(backend, work_dir) if ".." in pathlib.PurePosixPath(relative).parts else ""
        if backend_cwd:
            # In a mapped cwd the backend resolves ``..``: the same spelling as an absolute backend path.
            text, relative = f"{backend_cwd.rstrip('/')}/{relative}", ""
    raw = pathlib.Path(text).expanduser()
    # is_absolute_path_text (not Path.is_absolute) so a backend output path like
    # "/workspace/out.txt" maps through the executor on Windows too, where
    # Path.is_absolute() is False for drive-less roots.
    if resolved_output is not None:
        lexical_source, source = resolved_output.lexical_source, resolved_output.source
    elif executor_ref is not None and is_absolute_path_text(text) and not text.startswith("~"):
        try:
            lexical_source = executor_map_backend_path_lexical(executor_ref, text)
            source = executor_map_backend_path(executor_ref, text)
        except BackendPathNoMatch:  # Only a genuine no-match keeps the host spelling.
            lexical_source, source = raw, raw.resolve(strict=False)
        except ValueError as exc:
            return None, f"declared output is not a usable backend path: {exc}"
    elif is_absolute_path_text(text) or text.startswith("~"):
        lexical_source, source = raw, raw.resolve(strict=False)
    else:
        lexical_source = pathlib.Path(work_dir) / relative
        source = lexical_source.resolve(strict=False)
    changed = changed_paths or set()

    # Deliverables may be configured inside the active workspace.  Check that
    # physical target before the generic cwd root, otherwise a declared
    # ``workspace/Deliverables/.env`` is labelled active_workspace and skips
    # the user-files hidden/credential and Presence policy.
    try:
        deliverables_root = resource_root_path(ctx, "deliverables")
        deliverables_root_lexical = _deliverables_root_lexical()
        deliverables_root_lexical_alias = _deliverables_root_lexical_alias()
    except Exception:
        deliverables_root = None
        deliverables_root_lexical = None
        deliverables_root_lexical_alias = None
    if deliverables_root is not None:
        try:
            in_deliverables = (
                deliverables_root_lexical is not None
                and (
                    path_is_relative_to(lexical_source, deliverables_root_lexical)
                    or _lexical_path_is_relative_to_casefold(lexical_source, deliverables_root_lexical)
                    or _lexical_path_is_relative_to_casefold(
                        lexical_source, deliverables_root_lexical_alias,
                    )
                    or _lexical_path_is_relative_to_casefold(
                        lexical_source, deliverables_root,
                    )
                )
            )
        except (OSError, TypeError, ValueError):
            in_deliverables = False
        if in_deliverables:
            physical_deliverables = pathlib.Path(deliverables_root).resolve(strict=False)
            if not (
                path_is_relative_to(source, physical_deliverables)
                or _path_is_relative_to_casefold(source, physical_deliverables)
            ):
                return None, (
                    f"Deliverables output escapes its resolved configured root: {text}"
                )
            lexical_reason = lexical_user_files_block_reason(lexical_source)
            if lexical_reason:
                return None, f"protected user_files output {text}: {lexical_reason}"
            reason = user_files_path_block_reason(ctx, source)
            if reason:
                return None, f"protected user_files output {text}: {reason}"
            if not _presence_allows_user_output(ctx, source):
                return None, (
                    "user_files output is outside this presence task's positive "
                    "resource ceiling"
                )
            protected_reason = _protected_output_source_reason(
                ctx, source, "user_files", changed, binding, judge_changes=judge_changes,
            )
            if protected_reason:
                return None, protected_reason
            return source, ""

    for label, root in _allowed_output_roots(ctx, work_dir, cwd_root, binding):
        if not (
            path_is_relative_to(source, root)
            or _path_is_relative_to_casefold(source, root)
        ):
            continue
        if label == "user_files":
            reason = user_files_path_block_reason(ctx, source)
            if reason:
                return None, f"protected user_files output {text}: {reason}"
            if not _presence_allows_user_output(ctx, source):
                return None, (
                    "user_files output is outside this presence task's positive "
                    "resource ceiling"
                )
        protected_reason = _protected_output_source_reason(
            ctx, source, label, changed, binding, judge_changes=judge_changes,
        )
        if protected_reason:
            return None, protected_reason
        return source, ""
    allowed = ", ".join(
        f"{label}={root}"
        for label, root in _allowed_output_roots(ctx, work_dir, cwd_root, binding)
    )
    return None, f"output escapes allowed artifact roots: {text}; allowed_roots: {allowed}"


def _backend_cwd(executor_ref, work_dir: pathlib.Path) -> str:
    """The executor's spelling of ``work_dir``, or ``""`` where no mapping covers it."""
    if executor_ref is None:
        return ""
    try:
        return executor_map_host_path(executor_ref, pathlib.Path(work_dir))
    except ValueError:
        return ""


def _prepare_declared_output_context(ctx: ToolContext, outputs, work_dir: pathlib.Path):
    """Resolve Docker addresses before effects; unreadable byte baselines stay separate."""
    from ouroboros.tools.tool_result import ToolResult

    try:
        return prepare_docker_outputs(executor_ref_from_ctx(ctx), work_dir, outputs), None
    except (OutputResolutionUnavailable, ValueError) as exc:
        unavailable = isinstance(exc, OutputResolutionUnavailable)
        code = "CAPABILITY_UNAVAILABLE" if unavailable else "RESOURCE_POLICY_BLOCKED"
        text = sanitize_tool_result_for_log(f"⚠️ {code}: declared output preflight failed; nothing ran. {exc}")
        return None, ToolResult(status="unavailable" if unavailable else "blocked", code=code, text=text,
                                meta={"operation_outcome": "completed_no_effect", "process_not_started": True})


def _declared_output_refusal(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    binding: ResolvedResourceBinding | None = None,
    output_context: DockerOutputContext | None = None,
) -> str:
    """The pre-launch answer for a declaration the resolver already refuses.

    Absent and not-yet-changed outputs are not refused here; the post-run
    registration re-resolves every output, symlinks included."""
    reasons = []
    for raw_item in outputs or []:
        try:
            _source, reason = _resolve_declared_output(
                ctx, str(raw_item or ""), work_dir, cwd_root=cwd_root, binding=binding, judge_changes=False,
                resolved_output=output_context.paths.get(str(raw_item).strip()) if output_context else None,
            )
            resolved = output_context.paths.get(str(raw_item).strip()) if output_context else None
            for rel, host in resolved.mounts if resolved else ():
                _source, mount_reason = _resolve_declared_output(
                    ctx, str(host), work_dir, cwd_root=cwd_root, binding=binding, judge_changes=False,
                    resolved_output=DockerOutputPath("", "", host, resolved.lexical_source / rel),
                )
                if mount_reason:
                    reason = mount_reason
                    break
        except (ValueError, RuntimeError) as exc:
            reason = sanitize_tool_result_for_log(f"declared output could not be resolved: {raw_item}: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 — unknown I/O; the baseline that follows discloses it
            continue
        if reason:
            reasons.append(reason)
    if not reasons:
        return ""
    return ("⚠️ RESOURCE_POLICY_BLOCKED: declared outputs refused before the process started; nothing ran. "
            "Correct or remove these outputs and call again:\n" + "\n".join(f"- {reason}" for reason in reasons))


def _directory_fingerprint_from_entries(root: pathlib.Path, entries: list[tuple[str, os.stat_result, pathlib.Path]],
                                       *, strict: bool = False) -> str:
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
                if strict:
                    raise
    return digest.hexdigest()


def _directory_fingerprint(path: pathlib.Path, *, strict: bool = False,
                           resolved_output: DockerOutputPath | None = None) -> tuple[bool, int, str]:
    root = pathlib.Path(path).resolve(strict=False)
    total = 0
    entries: list[tuple[str, os.stat_result, pathlib.Path]] = []
    try:
        children = (iter_artifact_entries(root, resolved_output.mounts) if resolved_output else
                    (("", child) for child in iter_artifact_tree(root)))
        for rel, child in children:
            st = child.lstat()
            if not resolved_output:
                try:
                    rel = child.resolve(strict=False).relative_to(root).as_posix()
                except ValueError:
                    rel = safe_relpath(str(child))
            entries.append((rel, st, child))
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
        return True, total, _directory_fingerprint_from_entries(root, entries, strict=strict)
    except OSError:
        if strict:
            raise
        return False, -1, ""


def _fingerprint_output(path: pathlib.Path, *, strict: bool = False,
                        resolved_output: DockerOutputPath | None = None) -> tuple[bool, int, str]:
    """``(exists, size, digest)``; ``(False, -1, "")`` is absence or, unless ``strict``, unreadability.

    ``strict`` is the pre-run baseline: only a path that does not exist is absent, so a
    stat, walk or read failure on an existing output raises instead of reading as new."""
    try:
        try:
            mode = path.stat().st_mode  # explicit: Path.is_dir() hides some stat failures as False
        except (FileNotFoundError, NotADirectoryError):
            return False, -1, ""
        if stat.S_ISDIR(mode):
            return _directory_fingerprint(path, strict=strict, resolved_output=resolved_output)
        if not stat.S_ISREG(mode):
            return False, -1, ""
        measured = stream_artifact_file(path)
        return True, measured["size"], measured["sha256"]
    except OSError:
        if strict:
            raise
        return False, -1, ""


def _snapshot_declared_outputs(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
    binding: ResolvedResourceBinding | None = None,
    output_context: DockerOutputContext | None = None,
) -> Dict[str, tuple[bool, int, str]]:
    snapshots: Dict[str, tuple[bool, int, str]] = {}
    for raw_item in outputs or []:
        resolved = output_context.paths.get(str(raw_item).strip()) if output_context else None
        # The static view, as the pre-launch refusal: a tracked file is judged
        # changed or unchanged only after the run, so it keeps its baseline.
        source, block_reason = _resolve_declared_output(
            ctx,
            str(raw_item or ""),
            work_dir,
            cwd_root=cwd_root,
            changed_paths=changed_paths,
            binding=binding,
            judge_changes=False,
            resolved_output=resolved,
        )
        if source is not None and not block_reason:
            snapshots[str(source)] = _fingerprint_output(source, strict=True, resolved_output=resolved)
    return snapshots


def _declared_output_baseline(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
    binding: ResolvedResourceBinding | None = None,
    output_context: DockerOutputContext | None = None,
) -> tuple[Dict[str, tuple[bool, int, str]], str]:
    """``(baseline, problem)``: a baseline that could not be taken is said before launch.

    Without it registration cannot tell an unchanged existing output from a new one,
    so the problem also rides in the result; no baseline evidence is invented."""
    try:
        return _snapshot_declared_outputs(
            ctx, outputs, work_dir, cwd_root=cwd_root, changed_paths=changed_paths, binding=binding,
            output_context=output_context,
        ), ""
    except Exception as exc:  # noqa: BLE001 — disclosed, never a clean empty baseline
        problem = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")[:500]
    if getattr(ctx, "emit_progress_fn", None):
        # The owner's row states the fact; the exception text rides in the result only.
        ctx.emit_progress_fn("⚠️ Output baseline unavailable: an existing output may be registered "
                             "without proof that this run changed it; details in the tool result.")
    return {}, problem


def _scan_directory_output_members(
    ctx: ToolContext,
    source: pathlib.Path,
    *,
    label: str,
    changed_paths: set[str],
    binding: ResolvedResourceBinding | None = None,
    resolved_output: DockerOutputPath | None = None,
) -> tuple[list, int, str, list[str]]:
    """Return ``(members, dir_size, whole_dir_block_reason, skipped_receipts)``.

    D4 (capinv-447): a policy-rejected MEMBER is skipped with a receipt instead
    of failing the whole declared directory (one ``.env`` no longer discards an
    otherwise-successful export). Missing or unreadable members refuse the
    directory as a whole; the complete member list has no inline byte/count cap.
    """
    root = pathlib.Path(source).resolve(strict=False)
    members = []
    skipped: list[str] = []
    dir_size = 0
    try:
        for rel, child in iter_artifact_entries(root, resolved_output.mounts if resolved_output else ()):
            if child.is_symlink():
                skipped.append(f"{child}: symlink members are not followed")
                continue
            if not child.is_file():
                continue
            rel_parts = pathlib.PurePosixPath(rel).parts
            component_reason = _sensitive_output_component_reason(rel_parts)
            if component_reason:
                skipped.append(f"{child}: {component_reason}")
                continue
            if resolved_output:
                # Nested binds carry no extra roots or Deliverables rights.
                _allowed, reason = _resolve_declared_output(
                    ctx, str(child), root, cwd_root=label, changed_paths=changed_paths, binding=binding,
                    resolved_output=DockerOutputPath("", "", child, root / rel),
                )
                if reason:
                    skipped.append(f"{child}: {reason}")
                    continue
            reason = _protected_output_source_reason(
                ctx, child.resolve(strict=False), label, changed_paths, binding,
            )
            if reason:
                skipped.append(f"{child}: {reason}")
                continue
            members.append((rel, child) if resolved_output else child)
            dir_size += child.stat().st_size
    except OSError as exc:
        return [], dir_size, f"{source}: {type(exc).__name__}: {exc}", skipped
    return sorted(members), dir_size, "", skipped


def _record_skipped_members(ctx: ToolContext, output: str, members: List[str]) -> None:
    """The COMPLETE skip list as one durable row in the task's ``events.jsonl`` (the rendered
    note is bounded); a context without a log root falls back to the process log."""
    from ouroboros.utils import append_jsonl, utc_now_iso

    row = {"ts": utc_now_iso(), "type": "directory_output_members_skipped",
           "task_id": str(getattr(ctx, "task_id", "") or ""), "output": str(output),
           "count": len(members), "members": list(members)}
    try:
        if not append_jsonl(ctx.drive_logs() / "events.jsonl", row):
            raise OSError("events.jsonl append refused")
    except Exception:
        log.info("task %s: directory output %s: full export skip list (%d): %s",
                 row["task_id"] or "?", output, len(members), "; ".join(members))


def _register_process_outputs(
    ctx: ToolContext,
    outputs: List[str] | None,
    work_dir: pathlib.Path,
    cwd_root: str = "",
    changed_paths: set[str] | None = None,
    before_outputs: Dict[str, tuple[bool, int, str]] | None = None,
    binding: ResolvedResourceBinding | None = None,
    output_context: DockerOutputContext | None = None,
) -> tuple[str, bool, bool]:
    """Copy declared command outputs into the task artifact store."""

    if not outputs:
        return "", False, False
    notes: list[str] = []
    failed = False
    registered = False  # at least one canonical artifact record was actually created
    for raw_item in outputs:
        text = str(raw_item or "").strip()
        try:
            previous = output_context.paths.get(text) if output_context else None
            resolved = resolve_docker_output(output_context.executor, previous.requested, must_exist=True) if previous else None
            source, block_reason = _resolve_declared_output(
                ctx, text, work_dir, cwd_root=cwd_root, changed_paths=changed_paths, binding=binding,
                resolved_output=resolved,
            )
            for rel, host in resolved.mounts if resolved else ():
                _, reason = _resolve_declared_output(
                    ctx, str(host), work_dir, cwd_root=cwd_root, binding=binding, judge_changes=False,
                    resolved_output=DockerOutputPath("", "", host, resolved.lexical_source / rel),
                )
                if reason:
                    block_reason = reason
                    break
        except Exception as exc:  # noqa: BLE001 — output failure must not replace the completed process result
            notes.append(sanitize_tool_result_for_log(f"output could not be resolved: {text}: {type(exc).__name__}: {exc}"))
            failed = True
            continue
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
        after = _fingerprint_output(source, resolved_output=resolved)
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
            dir_members, _dir_size, blocked_member, skipped_members = _scan_directory_output_members(
                ctx,
                source,
                label=str(cwd_root or "cwd"),
                changed_paths=changed_paths or set(),
                binding=binding,
                resolved_output=resolved,
            )
            if skipped_members:
                # Per-member receipt (D4): the export is PARTIAL, never silently
                # so. The rendered note is bounded; the COMPLETE list goes to
                # the task log so the omission stays resolvable (#447 P1).
                shown = "; ".join(skipped_members[:5])
                more = (
                    f" (+{len(skipped_members) - 5} more; full list: directory_output_members_skipped"
                    f" in the task's events.jsonl)"
                    if len(skipped_members) > 5 else ""
                )
                if len(skipped_members) > 5:
                    _record_skipped_members(ctx, text, skipped_members)
                notes.append(
                    f"skipped {len(skipped_members)} member(s) of directory output {text}: {shown}{more}"
                )
            if blocked_member:
                notes.append(f"blocked directory output: {blocked_member}")
                failed = True
                continue
            if not dir_members:
                notes.append(f"failed directory output copy {text}: no exportable members")
                failed = True
                continue
            try:
                records = copy_directory_to_task_artifacts(
                    ctx,
                    source,
                    kind="process_output_directory",
                    **({"member_sources": dir_members} if resolved else {"member_paths": dir_members}),
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
        return "", False, False
    # Only canonical registration gets ARTIFACT_OUTPUTS; cosmetic unchanged-output
    # notes use ARTIFACT_OUTPUT_NOTE and cannot forge artifact_registered recovery.
    if failed:
        prefix = "⚠️ ARTIFACT_OUTPUT_ERROR"
    elif registered:
        prefix = "ARTIFACT_OUTPUTS"
    else:
        prefix = "ARTIFACT_OUTPUT_NOTE"
    return "\n\n" + prefix + ":\n" + "\n".join(f"- {note}" for note in notes), failed, registered


def _sensitive_output_component_reason(parts: tuple[str, ...]) -> str:
    """Credential-shape check for every relative path component of an export.

    D4 (capinv-447): reuses the workspace-patch SSOT (`_sensitive_untracked_reason`)
    per component — a plain dotfile like `.gitignore` or `.github/...` is an
    ordinary deliverable, while `.env`, `id_rsa`, or `secrets.json` anywhere in
    the path still refuses with the exact rule that fired.
    """
    from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

    for part in parts:
        text = str(part or "")
        if not text:
            continue
        reason = _sensitive_untracked_reason(text)
        if reason:
            return f"credential-like output path component {text} is not a deliverable artifact ({reason})"
    return ""
