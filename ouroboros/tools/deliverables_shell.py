"""Direct argv checks for writes that create entries in Deliverables."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Sequence

from ouroboros.tools.shell_guards import directory_destination_child_name, directory_destination_pairs
from ouroboros.credential_shapes import (
    BENIGN_DOT_NAMES,
    CREDENTIAL_COMPONENT_NAMES,
    CREDENTIAL_FILE_NAMES,
    CREDENTIAL_FILE_SUFFIXES,
    CREDENTIAL_NAME_RE,
)
from ouroboros.tool_access import (
    _path_is_relative_to_casefold,
    user_files_path_block_reason,
)
from ouroboros.workspace_executor import executor_ref_from_ctx, map_backend_path_lexical


_BLOCK = (
    "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell symlink target escapes "
    "the configured Deliverables path."
)
_TARGET_BLOCK = (
    "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed "
    "Deliverables path."
)
_SHORT_OPTIONS_WITH_ATTACHED_ARGS = frozenset({"S", "t"})


def _short_option_present(argv: Sequence[str], wanted: str) -> bool:
    """Find a short flag without mistaking an attached option argument for flags."""
    for token in argv[1:]:
        if not token.startswith("-") or token.startswith("--") or token == "-":
            continue
        for char in token[1:]:
            if char == wanted:
                return True
            if char in _SHORT_OPTIONS_WITH_ATTACHED_ARGS:
                break
    return False


def lexical_user_files_block_reason(candidate: pathlib.Path) -> str:
    """Retain hidden/credential semantics before a target symlink is resolved."""
    try:
        parts = pathlib.Path(candidate).expanduser().parts
    except (OSError, TypeError, ValueError):
        return "path could not be inspected"
    for part in parts:
        lower = part.lower()
        if not part or part in {"/", "\\"}:
            continue
        if lower in CREDENTIAL_COMPONENT_NAMES:
            return "path is hidden or credential-like (secret/credential directory)"
        if part.startswith(".") and lower not in BENIGN_DOT_NAMES:
            return "path is hidden or credential-like (non-allowlisted hidden component)"
    name = pathlib.PurePath(str(candidate)).name.lower()
    if (
        name in CREDENTIAL_FILE_NAMES
        or CREDENTIAL_NAME_RE.search(name)
        or name.endswith(CREDENTIAL_FILE_SUFFIXES)
    ):
        return "path name is credential-like"
    return ""


def _command_path(ctx, work_dir: pathlib.Path, token: str) -> pathlib.Path | None:
    try:
        raw = pathlib.Path(str(token or "")).expanduser()
        if raw.is_absolute():
            executor = executor_ref_from_ctx(ctx)
            if executor is not None:
                mapped = map_backend_path_lexical(executor, str(token))
                if mapped is not None:
                    return mapped
            return raw
        return work_dir / raw
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _source_link_kind(
    command: str,
    argv: Sequence[str],
    source_path: pathlib.Path | None,
) -> str:
    if command == "mv":
        try:
            return "preserved" if source_path is not None and source_path.is_symlink() else ""
        except OSError:
            return ""
    if command == "ln":
        return "ln" if any(
            token == "--symbolic"
            or token.startswith("--symbolic=")
            for token in argv[1:]
        ) or _short_option_present(argv, "s") else ""
    if command != "cp":
        return ""
    # GNU ``cp -s/--symbolic-link`` creates a new symlink even when the source
    # itself is an ordinary file. Treat that creation mode like ``ln -s`` so
    # its payload is checked against the destination container below.
    if any(
        token == "--symbolic-link"
        or token.startswith("--symbolic-link=")
        for token in argv[1:]
    ) or _short_option_present(argv, "s"):
        return "ln"
    try:
        if source_path is None or not source_path.is_symlink():
            return ""
    except OSError:
        return ""
    return "preserved" if any(
        token in {"--archive", "--no-dereference", "-d"}
        or _short_option_present(argv, "a")
        or _short_option_present(argv, "d")
        or _short_option_present(argv, "P")
        or (
            token.startswith("--preserve=")
            and any(name in token.split("=", 1)[1].split(",") for name in ("links", "all"))
        )
        for token in argv[1:]
    ) else ""


def direct_deliverable_target_block(
    ctx,
    work_dir: pathlib.Path,
    write_target_argvs: Sequence[list[str]],
    deliverables_root_physical: pathlib.Path | None,
    target_decision: Callable[[pathlib.Path], bool | None],
) -> str | None:
    """Check direct cp/mv/ln targets before generic workspace admission.

    The parser intentionally covers only argv-visible direct operations. Inline
    shell construction and recursive archive/copy semantics remain the documented
    residual rather than growing a second shell parser here.
    """
    for target_argv in write_target_argvs:
        for command, destination, source in directory_destination_pairs(target_argv):
            destination_path = _command_path(ctx, work_dir, destination)
            if destination_path is None:
                continue
            source_name = directory_destination_child_name(command, target_argv, source)
            if source_name in {"", ".", ".."}:
                destination_path = _command_path(ctx, work_dir, destination)
                if destination_path is not None:
                    decision = target_decision(destination_path)
                    if decision is not None:
                        return _TARGET_BLOCK
                continue
            destination_resolved = destination_path.resolve(strict=False)
            destination_is_dir = destination_resolved.is_dir()
            source_path = _command_path(ctx, work_dir, source)
            link_kind = _source_link_kind(command, target_argv, source_path)
            explicit_link = not destination_is_dir and link_kind in {"ln", "preserved"}
            if not destination_is_dir and not explicit_link:
                continue
            derived_target = destination_path if explicit_link else destination_path / source_name
            decision = target_decision(derived_target)
            if decision is not None and not decision:
                return _TARGET_BLOCK
            if decision is None or not link_kind:
                continue

            if link_kind == "preserved":
                if source_path is None:
                    return _BLOCK
                try:
                    payload = os.readlink(source_path)
                except (OSError, TypeError, ValueError):
                    return (
                        "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell symlink target "
                        "could not be resolved."
                    )
                payload_path = pathlib.Path(payload)
                if payload_path.is_absolute():
                    link_target = payload_path.resolve(strict=False)
                else:
                    link_parent = destination_resolved.parent if explicit_link else destination_resolved
                    link_target = (link_parent / payload_path).resolve(strict=False)
            else:
                source_arg = pathlib.Path(str(source or "")).expanduser()
                relative_link = command == "ln" and any(
                    token == "--relative"
                    or token.startswith("--relative=")
                    for token in target_argv[1:]
                ) or (command == "ln" and _short_option_present(target_argv, "r"))
                if relative_link:
                    # GNU ``ln --relative`` resolves the source from the
                    # command cwd and emits a payload relative to the new
                    # link. The resulting target is therefore the cwd-based
                    # source, not ``destination/source``.
                    if source_path is None:
                        return _BLOCK
                    link_target = source_path.resolve(strict=False)
                elif source_arg.is_absolute():
                    link_target = source_path.resolve(strict=False) if source_path is not None else source_arg.resolve(strict=False)
                else:
                    link_parent = destination_resolved.parent if explicit_link else destination_resolved
                    link_target = (link_parent / source_arg).resolve(strict=False)
            if (
                deliverables_root_physical is None
                or not (
                    link_target.is_relative_to(deliverables_root_physical)
                    or _path_is_relative_to_casefold(link_target, deliverables_root_physical)
                )
                or user_files_path_block_reason(ctx, link_target)
            ):
                return _BLOCK
    return None
