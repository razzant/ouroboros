"""Post-execution custody audit helpers for shell tools."""

from __future__ import annotations

import pathlib
import re
from typing import List, TYPE_CHECKING

from ouroboros.shell_parse import (
    collect_leading_env,
    embedded_absolute_path_tokens,
    is_absolute_path_text,
    shell_argv_with_inline,
    shell_command_string,
    shell_segments,
    split_redirections,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    _deliverables_root_lexical,
    _deliverables_root_lexical_alias,
    _lexical_path_is_relative_to_casefold,
    _path_is_relative_to_casefold,
    build_resolved_resource_binding,
    decide_tool_access,
    active_tool_profile,
    path_is_relative_to,
    resource_root_path,
    user_files_path_block_reason,
)
from ouroboros.tools.shell_guards import writer_target_tokens
from ouroboros.tools.deliverables_shell import lexical_user_files_block_reason
from ouroboros.tools.tool_result import (
    ToolResult,
    _publish_tool_result,
    _published_tool_result,
    _replace_tool_result,
)
from ouroboros.tools.verify import check_exit_masking
from ouroboros.workspace_executor import (
    executor_ref_from_ctx,
    map_backend_path,
    map_backend_path_lexical,
)

if TYPE_CHECKING:
    from ouroboros.tools.registry import ToolContext


_UNDECLARED_OUTPUTS_MARKER = "⚠️ ARTIFACT_OUTPUT_UNDECLARED"
_UNDECLARED_OUTPUT_SCAN_MAX_FILES = 5000
_UNDECLARED_OUTPUT_METADATA_COMMANDS = frozenset({"chmod", "chown", "mkdir", "rm"})
_SHELL_WRAPPER_COMMANDS = frozenset({"sh", "bash", "zsh"})


def _redirect_targets_for_audit(argv: list[str]) -> set[str]:
    """Return only syntactic redirect destinations from one command segment."""
    _command_argv, targets = split_redirections(argv)
    return set(targets)


def _writer_targets_for_output_audit(argv: list[str]) -> set[str]:
    """Find output-shaped writer operands without treating reads/metadata as writes."""
    targets: set[str] = set()
    for segment in shell_segments(argv):
        if not segment:
            continue
        try:
            _env, command_argv = collect_leading_env(segment)
        except Exception:
            command_argv = list(segment)
        command = pathlib.PurePath(command_argv[0]).name.lower().removesuffix(".exe") if command_argv else ""
        if command in _SHELL_WRAPPER_COMMANDS:
            body = shell_command_string(command_argv)
            if body:
                targets.update(_writer_targets_for_output_audit(shell_argv_with_inline(body)))
        segment_targets = set(writer_target_tokens(command_argv))
        if command in _UNDECLARED_OUTPUT_METADATA_COMMANDS:
            segment_targets = _redirect_targets_for_audit(segment)
        else:
            # sed's own second in-place matcher is gone: writer_target_tokens is
            # now channel-aware itself (in-place in any spelling, in-script
            # w/W/e, -f scripts), so the audit consumes the ONE parser and only
            # adds syntactic redirects — a non-writing sed reports no targets.
            segment_targets.update(_redirect_targets_for_audit(segment))
        targets.update(str(target) for target in segment_targets if str(target).strip())
    return targets


def _presence_allows_user_output(ctx: ToolContext, source: pathlib.Path) -> bool:
    """Apply the optional Presence ceiling to a physical user output."""
    from ouroboros.tools.registry import _presence_binding_allowed

    for operation in ("shell", "write"):
        try:
            output_binding = build_resolved_resource_binding(
                ctx, root="user_files", operation=operation, path=str(source),
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            continue
        if _presence_binding_allowed(ctx, output_binding):
            return True
    return False


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
_OUTPUT_STAT_SLACK_SEC = 2.0


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
        op = "write" if label == "user_files" else "read"
        if not decide_tool_access(profile=profile, root=label, operation=op).allow:  # type: ignore[arg-type]
            continue
        try:
            root_path = resource_root_path(ctx, label)  # type: ignore[arg-type]
        except Exception:
            continue
        if not any(
            path_is_relative_to(root_path, existing)
            and path_is_relative_to(existing, root_path)
            for _, existing in roots
        ):
            roots.append((label, root_path))
    if decide_tool_access(profile=profile, root="user_files", operation="write").allow:
        try:
            deliverables_root = resource_root_path(ctx, "deliverables")
        except Exception:
            deliverables_root = None
        if deliverables_root is not None and not any(
            path_is_relative_to(deliverables_root, existing)
            and path_is_relative_to(existing, deliverables_root)
            for _, existing in roots
        ):
            roots.append(("user_files", deliverables_root))
    return roots


def _mentioned_user_file_outputs_without_declaration(
    ctx: ToolContext,
    cmd: List[str],
    outputs: List[str] | None,
    scratch_abs: list[pathlib.Path] | None = None,
    command_start_ts: float | None = None,
    cwd: pathlib.Path | str | None = None,
) -> list[str]:
    """Best-effort audit for fresh absolute user_files writes without outputs."""
    if outputs:
        return []
    scratch_set = {str(p) for p in (scratch_abs or [])}
    mtime_floor = (
        command_start_ts - _OUTPUT_STAT_SLACK_SEC
        if command_start_ts is not None else None
    )
    workspace_root: pathlib.Path | None = None
    if bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        try:
            from ouroboros.tools.registry import active_repo_dir_for

            workspace_root = active_repo_dir_for(ctx).resolve(strict=False)
        except Exception:
            workspace_root = None
    mentioned: list[str] = []
    try:
        effective_cwd = (
            pathlib.Path(cwd).expanduser().resolve(strict=False)
            if cwd else pathlib.Path.cwd()
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        effective_cwd = pathlib.Path.cwd()
    executor_ref = executor_ref_from_ctx(ctx)
    user_output_roots: list[pathlib.Path] = []
    deliverables_root_lexical: pathlib.Path | None = None
    deliverables_root_lexical_alias: pathlib.Path | None = None
    deliverables_root_physical: pathlib.Path | None = None
    for resource_name in ("user_files", "deliverables"):
        try:
            root_path = resource_root_path(ctx, resource_name)
            user_output_roots.append(root_path)
            if resource_name == "deliverables":
                deliverables_root_physical = root_path
                try:
                    deliverables_root_lexical = _deliverables_root_lexical()
                    deliverables_root_lexical_alias = _deliverables_root_lexical_alias()
                except Exception:
                    deliverables_root_lexical = None
                    deliverables_root_lexical_alias = None
        except Exception:
            continue
    if not user_output_roots:
        return []
    argv_tokens = shell_argv_with_inline(cmd)
    writer_targets = _writer_targets_for_output_audit(argv_tokens)
    for token in argv_tokens:
        token_text = str(token)
        token_lower = token_text.lower()
        redirect_paths = [
            match.group("quoted") or match.group("bare")
            for match in _USER_FILE_REDIRECT_RE.finditer(token_text)
        ]
        has_write_open = bool(_USER_FILE_OPEN_WRITE_CALL_RE.search(token_text))
        is_writer_target = token_text in writer_targets
        if not redirect_paths and not has_write_open and not is_writer_target and not any(
            marker in token_lower
            for marker in ("write_text", "write_bytes", ".write(", "writefile", "createwritestream")
        ):
            continue
        candidates = embedded_absolute_path_tokens(str(token))
        candidates.extend(_EMBEDDED_OUTPUT_PATH_RE.findall(str(token)))
        candidates.extend(
            match.group("path") for match in _USER_FILE_WRITE_CALL_RE.finditer(str(token))
        )
        candidates.extend(
            match.group("path") for match in _USER_FILE_OPEN_WRITE_CALL_RE.finditer(str(token))
        )
        candidates.extend(redirect_paths)
        if is_writer_target:
            candidates.append(token_text)
        for candidate in candidates:
            try:
                candidate_text = str(candidate)
                raw_candidate = pathlib.Path(candidate_text).expanduser()
                if (
                    executor_ref is not None
                    and is_absolute_path_text(candidate_text)
                    and not candidate_text.startswith("~")
                ):
                    try:
                        path = map_backend_path(executor_ref, candidate_text)
                        lexical_path = map_backend_path_lexical(executor_ref, candidate_text)
                    except ValueError:
                        path = raw_candidate.resolve(strict=False)
                        lexical_path = raw_candidate
                elif raw_candidate.is_absolute() or candidate_text.startswith("~"):
                    path = raw_candidate.resolve(strict=False)
                    lexical_path = raw_candidate
                elif is_absolute_path_text(candidate_text):
                    continue
                else:
                    path = (effective_cwd / raw_candidate).resolve(strict=False)
                    lexical_path = effective_cwd / raw_candidate
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            paths_to_check: list[tuple[pathlib.Path, pathlib.Path]] = [(path, lexical_path)]
            if path.is_dir():
                try:
                    for child in path.iterdir():
                        if len(paths_to_check) >= _UNDECLARED_OUTPUT_SCAN_MAX_FILES:
                            break
                        if child.is_file():
                            paths_to_check.append((child, lexical_path / child.name))
                except OSError:
                    pass
            for output_path, lexical_output_path in paths_to_check:
                if not any(
                    path_is_relative_to(output_path, root)
                    or _path_is_relative_to_casefold(output_path, root)
                    for root in user_output_roots
                ):
                    continue
                lexical_reason = lexical_user_files_block_reason(lexical_output_path)
                if lexical_reason:
                    continue
                if user_files_path_block_reason(ctx, output_path):
                    continue
                if not _presence_allows_user_output(ctx, output_path):
                    continue
                in_deliverables = bool(
                    deliverables_root_lexical is not None
                    and (
                        path_is_relative_to(output_path, deliverables_root_lexical)
                        or _lexical_path_is_relative_to_casefold(
                            lexical_output_path, deliverables_root_lexical,
                        )
                        or _lexical_path_is_relative_to_casefold(
                            lexical_output_path, deliverables_root_lexical_alias,
                        )
                        or _lexical_path_is_relative_to_casefold(
                            lexical_output_path, deliverables_root_physical,
                        )
                    )
                )
                if (
                    workspace_root is not None
                    and path_is_relative_to(output_path, workspace_root)
                    and not in_deliverables
                ):
                    continue
                path_text = str(output_path)
                if path_text in scratch_set or path_text in mentioned:
                    continue
                if mtime_floor is not None:
                    try:
                        if not (
                            output_path.is_file()
                            and output_path.stat().st_mtime >= mtime_floor
                        ):
                            continue
                    except OSError:
                        continue
                mentioned.append(path_text)
    return mentioned


def _masked_green_disclosure(ctx: ToolContext, result: str, cmd) -> str:
    """Disclose an exit code the command's own shape laundered.

    RESULT AND TRACE ONLY: the published status, code and exit_code stay exactly
    as the producer set them, no receipt is written, and nothing here enters the
    verification ledger or receipt reconciliation. The note TRAILS the payload so
    line 1 still belongs to the producer's typed marker, and the reasons ride the
    published result's ``meta`` under ``exit_masking_reasons``. Same sensor as
    ``verify_and_record`` (``verify.check_exit_masking``), read a second time.
    """
    masked, reasons = check_exit_masking([str(part) for part in (cmd or [])])
    if not masked:
        return result
    text = (
        f"{result}\n\nEXIT_MASKING_NOTE: exit_code=0 belongs to the last stage "
        f"({', '.join(reasons)}); an upstream failure cannot change it — drop the "
        "filter, use `set -o pipefail`, or check each stage."
    )
    base = _published_tool_result(ctx, None)
    if (
        not isinstance(base, ToolResult)
        or base.meta.get("exit_code") != 0
        or base.text != str(result)
    ):
        # Only an exit-0 PROCESS is a laundered exit: the trusted process fact
        # decides, never the typed status (an undeclared-output nudge or an
        # artifact-registration error is still a green exit that the shape
        # laundered) and never an inspection of the text. A non-zero exit
        # already says what happened.
        return result
    # Republish the SAME typed result with the note and the reasons: status,
    # code and the trusted process facts (exit_code, signal) are carried
    # through untouched by `_replace_tool_result`.
    return _publish_tool_result(ctx, _replace_tool_result(
        base, text=text, meta_updates={"exit_masking_reasons": list(reasons)},
    ))
