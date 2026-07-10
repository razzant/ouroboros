"""File/data tools plus code search and digest helpers."""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
import os
import pathlib
import re
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.artifacts import artifact_store_path_block_reason, copy_file_to_task_artifacts
from ouroboros.project_facts import filter_out_project_store as _filter_out_project_store
from ouroboros.project_facts import project_store_access_block as _project_store_access_block
from ouroboros.protected_artifacts import block_reason_for_path
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.tool_access import (
    decide_tool_access,
    active_tool_profile,
    normalize_root,
    normalize_root_relative,
    project_room_lens_dir,
    resolve_user_file_path,
    resolve_resource_path,
    resource_root_path,
    user_files_path_block_reason,
)
from ouroboros.utils import atomic_write_json, read_text, safe_relpath, utc_now_iso, write_text_atomic
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path
from ouroboros.contracts.skill_payload_policy import (
    SKILL_PAYLOAD_ALL_BUCKETS,
    SKILL_OWNER_STATE_FILENAMES,
    SkillPayloadPathError,
    SkillPayloadTarget,
    cross_skill_redirect_error,
    decide_payload_short_form,
    is_skill_control_plane_path as _policy_is_skill_control_plane_path,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _policy_is_skill_owner_state_target,
    is_skill_create_typo,
    resolve_skill_payload_target,
)

log = logging.getLogger(__name__)

_SKILL_OWNER_STATE_FILENAMES = SKILL_OWNER_STATE_FILENAMES

# Payload-local provenance sidecars are launcher/marketplace-owned, not
# skill-author-editable. Generic write/delete/upload paths must block them.
_SELF_AUTHORED_MARKER = ".self_authored.json"


def _render_line_slice(path: str, content: str, max_lines: int = 2000, start_line: int = 1) -> str:
    """Return a line-ranged file view with the shared read-tool header."""
    start_raw, max_raw = _coerce_line_window(start_line, max_lines)
    max_raw = max(1, max_raw)
    lines = content.splitlines(keepends=True)
    total = len(lines)
    start = max(1, min(start_raw, total + 1))
    end = min(start + max_raw - 1, total)
    result = "".join(lines[start - 1:end])
    header = f"# {path} — lines {start}\u2013{end} of {total}\n"
    return header + result


def _coerce_line_window(start_line: Any = 1, max_lines: Any = 2000) -> tuple[int, int]:
    try:
        start_raw = int(start_line)
    except (TypeError, ValueError):
        start_raw = 1
    try:
        max_raw = int(max_lines)
    except (TypeError, ValueError):
        max_raw = 2000
    return start_raw, max(1, max_raw)


def _is_cognitive_data_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").lstrip("./")
    return text.startswith("memory/") or text in _MEMORY_AT_DRIVE_MEMORY


def _skill_payload_parts(target: pathlib.Path, data_root: pathlib.Path) -> tuple[str, str, pathlib.Path] | None:
    """Return (bucket, skill, payload_root) for data/skills payload paths."""
    for candidate in (target, pathlib.Path(target).resolve(strict=False)):
        try:
            rel = candidate.relative_to(data_root)
        except (OSError, ValueError):
            continue
        parts = rel.parts
        if len(parts) < 3 or parts[0].lower() != "skills":
            continue
        bucket = parts[1]
        if bucket.lower() not in SKILL_PAYLOAD_ALL_BUCKETS:
            continue
        skill_name = parts[2]
        if not skill_name or skill_name in {".", ".."}:
            continue
        return bucket.lower(), skill_name, data_root / "skills" / bucket / skill_name
    return None


def _native_payload_without_seed(target: pathlib.Path, data_root: pathlib.Path) -> bool:
    payload = _skill_payload_parts(target, data_root)
    if payload is None:
        return False
    bucket, _skill_name, payload_root = payload
    return bucket == "native" and not (payload_root / ".seed-origin").is_file()


def _data_skill_target(path: str, drive_root: pathlib.Path) -> SkillPayloadTarget | None:
    """Single resolver for an explicit data-plane skills/<bucket>/<skill>/... write target (None when
    the path is not inside a skill payload). SSOT for both _data_skill_path and the _data_write
    manifest-first typo guard, so the payload resolution is never duplicated."""
    try:
        return resolve_skill_payload_target(pathlib.Path(drive_root), path)
    except SkillPayloadPathError:
        return None


def _data_skill_path(path: str, drive_root: pathlib.Path) -> pathlib.Path | None:
    target = _data_skill_target(path, drive_root)
    return target.target_path if target is not None else None


def _looks_like_serialized_tool_result(content: Any) -> bool:
    text = str(content or "").lstrip()
    if not (text.startswith("{'content'") or text.startswith('{"content"')):
        return False
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return False
    return isinstance(parsed, dict) and isinstance(parsed.get("content"), str)


def _is_skill_owner_state_target(target: pathlib.Path, data_root: pathlib.Path) -> bool:
    return _policy_is_skill_owner_state_target(target, data_root)


def is_skill_control_plane_path(target: pathlib.Path, data_root: pathlib.Path) -> bool:
    """Return True for skill owner/provenance files blocked from generic writes."""
    return _policy_is_skill_control_plane_path(target, data_root)


def _is_workspace_executor_control_state_path(target: pathlib.Path, data_root: pathlib.Path) -> bool:
    try:
        rel_parts = pathlib.Path(target).resolve(strict=False).relative_to(
            pathlib.Path(data_root).resolve(strict=False)
        ).parts
    except (OSError, ValueError):
        return False
    lowered = [str(part).casefold() for part in rel_parts]
    return "state" in lowered and "workspace_executor_processes" in lowered


class _ListingFailure(Exception):
    """A failed list_files state that must surface as a FIRST-CLASS tool error.

    v6.54.3 (review round 4): path-escape / not-found / not-a-directory used to
    return warning strings INSIDE an ok-shaped JSON list — the exact
    error-inside-success shape the TB2.1 post-mortem showed silently poisoning
    reasoning. _list_files renders this as a leading ⚠️ LIST_FILES_ERROR."""


def _list_dir(root: pathlib.Path, rel: str, max_entries: int = 500) -> List[str]:
    target = (root / safe_relpath(rel)).resolve()
    # CONFINE to the root before any iterdir: a resolved target that escapes (e.g. an
    # in-tree symlink pointing outside — common in untrusted child-created project /
    # deliverable trees behind the new read-only roots) is rejected, never listed.
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise _ListingFailure(f"Path escapes root: {rel}") from None
    if not target.exists():
        raise _ListingFailure(f"Directory not found: {rel}")
    if not target.is_dir():
        raise _ListingFailure(f"Not a directory: {rel}")
    items = []
    # A hard iterdir/permission/race failure PROPAGATES: _list_files renders it
    # as a first-class "⚠️ LIST_FILES_ERROR" tool error, never an ok-shaped JSON
    # listing carrying an error string inside (v6.54.3, review round 3).
    for entry in sorted(target.iterdir()):
        if len(items) >= max_entries:
            items.append(f"...(truncated at {max_entries})")
            break
        suffix = "/" if entry.is_dir() else ""
        items.append(str(entry.relative_to(root)) + suffix)
    return items


def _list_user_files_dir(ctx: ToolContext, root: pathlib.Path, target: pathlib.Path, max_entries: int = 500) -> List[str]:
    if not target.exists():
        raise _ListingFailure(f"Directory not found: {target}")
    if not target.is_dir():
        raise _ListingFailure(f"Not a directory: {target}")
    items: List[str] = []
    hidden = 0
    # A hard iterdir/permission/race failure PROPAGATES to the first-class
    # "⚠️ LIST_FILES_ERROR" path in _list_files (v6.54.3, review round 3).
    for entry in sorted(target.iterdir()):
        if user_files_path_block_reason(ctx, entry):
            hidden += 1
            continue
        if len(items) >= max_entries:
            items.append(f"...(truncated at {max_entries})")
            break
        suffix = "/" if entry.is_dir() else ""
        # An external-workspace listing outside the user_files home has no
        # home-relative form — render the absolute path instead of crashing
        # the whole listing on relative_to (v6.54.3: the TB2.1
        # "'/app/…' is not in the subpath of '/root'" class).
        try:
            rendered = str(entry.relative_to(root))
        except ValueError:
            rendered = str(entry)
        items.append(rendered + suffix)
    if hidden:
        items.append(f"⚠️ {hidden} hidden/control entr{'y' if hidden == 1 else 'ies'} omitted from user_files listing.")
    return items


_SUBAGENT_SECRET_FILE_NAMES = frozenset({
    ".env",
    ".netrc",
    "auth.json",
    "credentials",
    "credentials.json",
    "keys.json",
    "secret.json",
    "secrets.json",
    "settings.json",
    "settings.json.lock",
    "token.json",
    "tokens.json",
})


def is_restricted_subagent_profile(ctx: ToolContext) -> bool:
    # Fail-closed SSOT for subagent READ restrictions (secret/control denials):
    # read-only subagents, acting subagents, and delegated subagents with a
    # missing/invalid constraint are ALL barred from reading owner secrets/control
    # state. Acting children may WRITE their isolated surface but never read owner
    # secrets; the resource WRITE distinction lives in _local_readonly_resource_block.
    from ouroboros.tool_access import active_tool_profile
    return active_tool_profile(ctx) in ("local_readonly_subagent", "acting_subagent")


def _is_subagent_secret_data_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return False
    parts = [part.lower() for part in text.split("/") if part and part != "."]
    if not parts:
        return False
    if any(part in {"auth", "credentials", "secrets", "tokens"} for part in parts):
        return True
    name = parts[-1]
    normalized_names = {name, name.lstrip(".")}
    if name.lstrip(".") == "settings.tmp":
        normalized_names.add("settings.json")
    for protected_name in (_SUBAGENT_SECRET_FILE_NAMES | _SKILL_OWNER_STATE_FILENAMES):
        bare = name.lstrip(".")
        if bare.startswith(f"{protected_name}.tmp") or bare.startswith(f"{protected_name}.lock"):
            normalized_names.add(protected_name)
    if normalized_names & (_SUBAGENT_SECRET_FILE_NAMES | _SKILL_OWNER_STATE_FILENAMES):
        return True
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    if name.endswith((".key", ".pem", ".p12", ".pfx")):
        return True
    return bool(re.search(r"(?:^|[._-])(api[_-]?key|credential|password|secret|token)(?:[._-]|$)", name))


def _is_subagent_secret_repo_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    parts = [part.lower() for part in text.split("/") if part and part != "."]
    if ".git" in parts or any(part in {"auth", "credentials", "secrets", "tokens"} for part in parts):
        return True
    if not parts:
        return False
    name = parts[-1]
    if name in _SUBAGENT_SECRET_FILE_NAMES or name == "settings.tmp":
        return True
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    if name.endswith((".key", ".pem", ".p12", ".pfx")):
        return True
    if re.search(r"(?:^|[._-])(api[_-]?key|credential|password|secret|token)(?:[._-]|$)", name):
        suffix = pathlib.PurePosixPath(name).suffix.lower()
        return suffix in {"", ".json", ".env", ".key", ".pem", ".p12", ".pfx", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf"}
    return False


def _is_subagent_secret_repo_target(target: pathlib.Path, repo_root: pathlib.Path) -> bool:
    root = pathlib.Path(repo_root).resolve(strict=False)
    try:
        rel = str(pathlib.Path(target).resolve(strict=False).relative_to(root)).replace(os.sep, "/")
    except (OSError, ValueError):
        rel = str(target).replace(os.sep, "/")
    if _is_subagent_secret_repo_path(rel):
        return True
    secret_candidates = [
        root / ".git" / "credentials",
        root / ".git" / "config",
    ]
    try:
        secret_candidates.extend(
            candidate
            for candidate in root.iterdir()
            if candidate.is_file() and _is_subagent_secret_repo_path(candidate.name)
        )
    except OSError:
        pass
    return any(
        candidate.is_file()
        and target.exists()
        and target.samefile(candidate)
        for candidate in secret_candidates
    )


def _filter_subagent_secret_repo_listing(items: List[str], repo_root: pathlib.Path) -> List[str]:
    filtered: List[str] = []
    redacted = 0
    root = pathlib.Path(repo_root).resolve(strict=False)
    for item in items:
        marker = item.rstrip("/")
        if marker.startswith("⚠️") or marker.startswith("...("):
            filtered.append(item)
            continue
        if _is_subagent_secret_repo_path(marker) or _is_subagent_secret_repo_target(root / marker, root):
            redacted += 1
            continue
        filtered.append(item)
    if redacted:
        filtered.append(f"⚠️ {redacted} secret/control entr{'y' if redacted == 1 else 'ies'} hidden from this subagent.")
    return filtered


def _filter_subagent_secret_listing(items: List[str], data_root: pathlib.Path) -> List[str]:
    filtered: List[str] = []
    redacted = 0
    root = pathlib.Path(data_root).resolve(strict=False)
    for item in items:
        marker = item.rstrip("/")
        if marker.startswith("⚠️") or marker.startswith("...("):
            filtered.append(item)
            continue
        target = root / marker
        try:
            resolved_rel = str(pathlib.Path(target).resolve(strict=False).relative_to(root)).replace(os.sep, "/")
        except (OSError, ValueError):
            resolved_rel = marker
        if (
            _is_subagent_secret_data_path(marker)
            or _is_subagent_secret_data_path(resolved_rel)
            or _is_skill_owner_state_target(target, root)
            or is_skill_owner_state_alias(target, root)
            or any(
                candidate.is_file()
                and _is_subagent_secret_data_path(candidate.name)
                and target.exists()
                and target.samefile(candidate)
                for candidate in root.iterdir()
            )
        ):
            redacted += 1
            continue
        filtered.append(item)
    if redacted:
        filtered.append(f"⚠️ {redacted} secret/control entr{'y' if redacted == 1 else 'ies'} hidden from this subagent.")
    return filtered


_MEMORY_AT_DRIVE_MEMORY = frozenset({
    "identity.md", "scratchpad.md", "dialogue_summary.md",
    "dialogue_blocks.json", "registry.md", "deep_review.md",
    "WORLD.md",
})


def _repo_read(
    ctx: ToolContext,
    path: str,
    max_lines: int = 2000,
    start_line: int = 1,
    display_path: str | None = None,
) -> str:
    """Read a repo file; root-level memory names return a runtime_data read hint."""
    target = ctx.repo_path(path)
    if is_restricted_subagent_profile(ctx) and _is_subagent_secret_repo_target(target, active_repo_dir_for(ctx)):
        return "⚠️ REPO_READ_BLOCKED: this subagent cannot read repo secret or control files."
    try:
        content = read_text(target)
    except FileNotFoundError:
        norm = path.strip().lstrip("./").replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        if "/" not in norm and base in _MEMORY_AT_DRIVE_MEMORY:
            title = base.split('.')[0].title()
            return (
                f"⚠️ NOT_FOUND: '{path}' is not at the repo root.\n\n"
                f"This file lives at `data_root/memory/{base}`, not in the "
                f"git repo. Some memory artifacts are already summarized in "
                f"context as `## {title}`, but raw memory state must be read "
                f"from the data root. If you need the raw file, call "
                f"`read_file(root='runtime_data', path='memory/{base}')`."
            )
        raise
    return _render_line_slice(display_path or path, content, max_lines=max_lines, start_line=start_line)


def _repo_list(ctx: ToolContext, dir: str = ".", max_entries: int = 500) -> str:
    repo_root = active_repo_dir_for(ctx)
    target = ctx.repo_path(dir)
    if is_restricted_subagent_profile(ctx) and _is_subagent_secret_repo_target(target, repo_root):
        # First-class tool error, not an ok-shaped one-element JSON listing
        # (v6.54.3, review round 5 — the whole-call block IS the result).
        return "⚠️ REPO_LIST_BLOCKED: this subagent cannot list repo secret or control paths."
    # ctx.repo_path already normalized absolute/redundant-prefix dirs; pass the
    # resulting root-relative form so _list_dir doesn't re-nest the raw input.
    try:
        listed_rel = target.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        listed_rel = dir
    items = _list_dir(repo_root, listed_rel, max_entries)
    if is_restricted_subagent_profile(ctx):
        items = _filter_subagent_secret_repo_listing(items, repo_root)
    return json.dumps(items, ensure_ascii=False, indent=2)


def _normalize_data_read_path(ctx: ToolContext, path: str) -> str:
    """Normalize paths that redundantly include the drive root."""

    norm = str(path).strip().replace("\\", "/")
    if norm.startswith("./"):
        norm = norm[2:]
    drive_str = str(ctx.drive_root).rstrip("/")
    drive_no_lead = drive_str.lstrip("/")
    if drive_no_lead and norm.lstrip("/").startswith(drive_no_lead):
        stripped = norm.lstrip("/")
        norm = stripped[len(drive_no_lead):].lstrip("/")
    elif norm.startswith(".tmp-data-") or norm.lstrip("/").startswith(".tmp-data-"):
        candidate = norm.lstrip("/")
        first_slash = candidate.find("/")
        if first_slash > 0:
            after = candidate[first_slash + 1:]
            if after.startswith("data/"):
                norm = after[len("data/"):]
            else:
                norm = after
    return norm


def _data_read(
    ctx: ToolContext,
    path: str,
    max_lines: int = 2000,
    start_line: int = 1,
    display_path: str | None = None,
) -> str:
    """Read a drive text file; duplicate drive_root prefixes are stripped."""
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    norm = _normalize_data_read_path(ctx, path)
    if (b := _project_store_access_block(norm)):
        return b
    if is_restricted_subagent_profile(ctx) and _is_subagent_secret_data_path(norm):
        return "⚠️ DATA_READ_BLOCKED: this subagent cannot read secret or owner-control data files."
    if task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            target = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, norm)
        except ValueError as e:
            return f"⚠️ DATA_READ_BLOCKED: {e}"
    else:
        target = ctx.drive_path(norm)
    if is_restricted_subagent_profile(ctx):
        root = pathlib.Path(ctx.drive_root).resolve(strict=False)
        try:
            resolved_rel = str(pathlib.Path(target).resolve(strict=False).relative_to(root)).replace(os.sep, "/")
        except (OSError, ValueError):
            resolved_rel = norm
        if (
            _is_subagent_secret_data_path(resolved_rel)
            or _is_skill_owner_state_target(target, root)
            or is_skill_owner_state_alias(target, root)
            or any(
                candidate.is_file()
                and _is_subagent_secret_data_path(candidate.name)
                and pathlib.Path(target).exists()
                and pathlib.Path(target).samefile(candidate)
                for candidate in root.iterdir()
            )
        ):
            return "⚠️ DATA_READ_BLOCKED: this subagent cannot read secret or owner-control data files."
    if (
        _is_skill_owner_state_target(target, pathlib.Path(ctx.drive_root))
        and target.name.lower() != "review.json"
    ):
        return "DATA_READ_BLOCKED: skill owner state is not readable through generic data tools."
    try:
        content = read_text(target)
        start_raw, max_raw = _coerce_line_window(start_line, max_lines)
        if _is_cognitive_data_path(norm) and start_raw == 1 and max_raw == 2000:
            if display_path is None:
                return content
            full_line_count = max(1, len(content.splitlines()))
            return _render_line_slice(display_path, content, max_lines=full_line_count, start_line=1)
        return _render_line_slice(display_path or norm, content, max_lines=max_raw, start_line=start_raw)
    except FileNotFoundError:
        if norm.replace("\\", "/").startswith("memory/"):
            explanation = (
                "Memory artifacts under memory/ are created lazily on first "
                "write. Treat this as an empty/absent state and proceed with "
                "initialization if that is the task."
            )
        else:
            explanation = (
                "This path does not exist yet. Treat it as an empty/absent "
                "state. Lazy-creation is not guaranteed for paths outside "
                "memory/; if this path was expected to exist, verify it was "
                "written correctly."
            )
        return (
            f"⚠️ DATA_NOT_YET_CREATED: {path}\n\n"
            f"{explanation} Use list_files with root=runtime_data to confirm what currently exists."
        )


def _data_list(ctx: ToolContext, dir: str = ".", max_entries: int = 500) -> str:
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    norm_dir = _normalize_data_read_path(ctx, dir)
    # Whole-call block states are FIRST-CLASS tool errors, never ok-shaped
    # one-element JSON listings (v6.54.3, review round 5).
    if (b := _project_store_access_block(norm_dir)):
        return str(b)
    if is_restricted_subagent_profile(ctx) and _is_subagent_secret_data_path(norm_dir):
        return "⚠️ DATA_LIST_BLOCKED: this subagent cannot list secret or owner-control data paths."
    if is_restricted_subagent_profile(ctx):
        try:
            list_target = ctx.drive_path(norm_dir)
        except ValueError as e:
            return f"⚠️ DATA_LIST_BLOCKED: {e}"
        root = pathlib.Path(ctx.drive_root).resolve(strict=False)
        if _is_skill_owner_state_target(list_target, root) or is_skill_owner_state_alias(list_target, root):
            return "⚠️ DATA_LIST_BLOCKED: this subagent cannot list secret or owner-control data paths."
    if task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            root = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, dir)
        except ValueError as e:
            return f"⚠️ DATA_LIST_BLOCKED: {e}"
        items = _list_dir(root, ".", max_entries)
        return json.dumps(items, ensure_ascii=False, indent=2)
    # Drop any projects/<id> entry so a generic root listing never exposes the store.
    items = _filter_out_project_store(_normalize_data_read_path(ctx, dir), _list_dir(ctx.drive_root, dir, max_entries))
    if is_restricted_subagent_profile(ctx):
        items = _filter_subagent_secret_listing(items, pathlib.Path(ctx.drive_root))
    return json.dumps(items, ensure_ascii=False, indent=2)


def _str_match_replace(
    text: str, old_str: str, new_str: str, display_path: str, error_tag: str
):
    """Shared exact, byte-level, single-occurrence replacement for both str-replace
    editors — the repo editor (``git._str_replace_editor``) and the data-plane editor
    (``_edit_text``) — so they give IDENTICAL match feedback (deferral 4). Returns
    ``(new_text, None)`` on a unique match, else ``(None, error_message)`` with the
    count==0 file preview / count>1 positional hints. ``error_tag`` is the caller's
    error prefix (e.g. ``STR_REPLACE_ERROR`` / ``EDIT_TEXT_ERROR``)."""
    count = text.count(old_str)
    if count == 0:
        preview = text[:2000]
        return None, (
            f"⚠️ {error_tag}: old_str not found in {display_path}.\n"
            f"File preview (first 2000 chars):\n{preview}"
        )
    if count > 1:
        positions = []
        start = 0
        for _ in range(min(count, 5)):
            idx = text.index(old_str, start)
            positions.append(f"line {text[:idx].count(chr(10)) + 1}")
            start = idx + 1
        return None, (
            f"⚠️ {error_tag}: old_str found {count} times in {display_path} "
            f"(must be unique). Occurrences at: {', '.join(positions)}. "
            f"Include more surrounding context in old_str to make it unique."
        )
    return text.replace(old_str, new_str, 1), None


def _check_data_shrink_guard(
    target: pathlib.Path, new_content: str, force: bool = False
) -> "str | None":
    """Block likely accidental truncation of an EXISTING data-plane file on OVERWRITE,
    unless force=True (deferral 5). Mirrors the repo shrink-guard
    (``git._check_shrink_guard``) but WITHOUT the ``git ls-files`` tracking check — the
    data plane is not a git tree. Skips a non-existent target (a fresh create is any
    size) and appends (the caller only invokes this on overwrite). Never raises."""
    if force:
        return None
    try:
        if not target.exists():
            return None
        old_len = len(target.read_text(encoding="utf-8"))
        new_len = len(new_content)
        if old_len > 0 and new_len < old_len * 0.7:
            pct = round(new_len / old_len * 100)
            return (
                f"⚠️ WRITE_BLOCKED: new content for '{target.name}' is {pct}% of original "
                f"({old_len} -> {new_len} chars). This looks like accidental truncation. "
                f"Use edit_text for surgical edits, or pass force=true to confirm an "
                f"intentional rewrite."
            )
    except Exception:
        return None
    return None


def _data_write(
    ctx: ToolContext,
    path: str,
    content: str,
    mode: str = "overwrite",
    bucket: str = "",
    skill_name: str = "",
    display_root: str = "runtime_data",
    force: bool = False,
) -> str:
    if (b := _project_store_access_block(_normalize_data_read_path(ctx, path))):
        return b
    # bucket+skill_name synthesize a payload-confined skill_repair constraint.
    short_form = decide_payload_short_form(
        bucket=bucket,
        skill_name=skill_name,
        path_text=path,
        repo_dir=pathlib.Path(ctx.repo_dir),
        drive_root=pathlib.Path(ctx.drive_root),
    )
    if short_form.error:
        return f"⚠️ DATA_WRITE_ERROR: {short_form.error}"
    synth = short_form.constraint
    existing_tc = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    redirect_err = cross_skill_redirect_error(existing_tc, synth)
    if redirect_err:
        return f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
    # Real skill_repair confinement wins over synthesized short-form context.
    if existing_tc and existing_tc.mode == "skill_repair":
        task_constraint = existing_tc
    else:
        task_constraint = synth or existing_tc
    write_path = _normalize_data_read_path(ctx, path)
    # Resolved skills payload target (None unless this is an explicit skills/<bucket>/<skill> path).
    # The manifest-first typo guard runs LATER, AFTER the owner-state/control-plane/content blocks, so
    # those security blocks take precedence over a missing-payload typo.
    _skill_target = None
    if task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            p = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, path)
        except ValueError as e:
            return f"⚠️ DATA_WRITE_ERROR: {e}"
    else:
        # Resolve the skills target on the NORMALIZED write_path (the exact path the write uses below)
        # so the manifest-first typo guard can never be skipped by a redundant drive-root / .tmp-data-*
        # prefix that _normalize_data_read_path would later strip into a real skills/<bucket>/<skill>.
        _skill_target = _data_skill_target(write_path, pathlib.Path(ctx.drive_root))
        explicit_skill_target = _skill_target.target_path if _skill_target is not None else None
        p = explicit_skill_target if explicit_skill_target is not None else ctx.drive_path(write_path)
    # Defense-in-depth: settings.json is owner-only. Use inode-aware matching
    # for symlinks/hardlinks/case-insensitive APFS/NTFS, with a fallback for
    # not-yet-existing case variants.
    from ouroboros import config as _cfg
    target_path = pathlib.Path(p)
    settings_path = pathlib.Path(_cfg.SETTINGS_PATH)
    data_root = pathlib.Path(_cfg.DATA_DIR).resolve(strict=False)
    ctx_data_root = pathlib.Path(ctx.drive_root).resolve(strict=False)
    if task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        lexical_target = pathlib.Path(p).resolve(strict=False)
    else:
        lexical_target = pathlib.Path(ctx.drive_root).resolve(strict=False) / safe_relpath(write_path)
    suffix = pathlib.PurePosixPath(str(path or "")).suffix.lower()
    if suffix in {".py", ".md", ".json", ".sh"} and _looks_like_serialized_tool_result(content):
        return (
            "⚠️ DATA_WRITE_BLOCKED: content looks like a serialized tool result "
            "object (for example {'content': ...}) rather than file text. "
            "Extract the actual file body before calling write_file."
        )
    if _native_payload_without_seed(lexical_target, data_root) or _native_payload_without_seed(target_path, data_root):
        return (
            "⚠️ DATA_WRITE_BLOCKED: data/skills/native/<skill>/ is reserved "
            "for launcher-seeded skills that carry a .seed-origin marker. "
            "Write user- or agent-authored skill payloads under "
            "data/skills/external/<skill>/ instead."
        )
    skill_owner_state_path = (
        _is_skill_owner_state_target(lexical_target, data_root)
        or _is_skill_owner_state_target(target_path, data_root)
    )
    if not skill_owner_state_path:
        skill_owner_state_path = is_skill_owner_state_alias(target_path, data_root)
    if skill_owner_state_path:
        return (
            "⚠️ DATA_WRITE_BLOCKED: skill review, enablement, grants, and "
            "marketplace provenance are owner/review controlled state. Edit "
            "the skill payload under data/skills/ and use skill_review, the "
            "Skills UI toggle, or the desktop launcher grant flow."
        )
    # Block marketplace/launcher sidecars for every data_write path, not only heal mode.
    if is_skill_control_plane_path(lexical_target, data_root) or is_skill_control_plane_path(target_path, data_root):
        return (
            "⚠️ DATA_WRITE_BLOCKED: marketplace provenance and launcher "
            "seed markers (.clawhub.json, .ouroboroshub.json, "
            "SKILL.openclaw.md, .seed-origin) are owner/review controlled. "
            "Edit the payload's user-authored files instead and rerun skill_review."
        )
    if (
        _is_workspace_executor_control_state_path(lexical_target, ctx_data_root)
        or _is_workspace_executor_control_state_path(target_path, ctx_data_root)
        or _is_workspace_executor_control_state_path(lexical_target, data_root)
        or _is_workspace_executor_control_state_path(target_path, data_root)
    ):
        return (
            "⚠️ DATA_WRITE_BLOCKED: workspace executor process records are "
            "owner/runtime control-plane state. Use process/service lifecycle "
            "tools instead of writing state/workspace_executor_processes directly."
        )
    matches = False
    try:
        if target_path.exists() and settings_path.exists():
            matches = target_path.samefile(settings_path)
    except OSError:
        matches = False
    if not matches:
        try:
            same_parent = target_path.parent.resolve() == settings_path.parent.resolve()
        except OSError:
            same_parent = False
        if same_parent and target_path.name.lower() == settings_path.name.lower():
            matches = True
    if matches:
        return (
            "⚠️ DATA_WRITE_BLOCKED: settings.json is the canonical owner-edited "
            "file. Tool-level writes must route through /api/settings (which "
            "applies key-by-key policy — OUROBOROS_RUNTIME_MODE is owner-only "
            "and dropped on POST; other keys flow through normally). To change "
            "owner-only values, stop the agent, edit ~/Ouroboros/data/settings.json "
            "directly, then restart."
        )
    # Manifest-first typo guard (SSOT with the bucket/skill_name short-form via is_skill_create_typo),
    # applied AFTER the owner-state / control-plane / content DATA_WRITE_BLOCKED guards above so those
    # take precedence: an explicit runtime_data write into a NON-existent skills/<bucket>/<skill>
    # payload is a typo unless it is the root manifest of a NEW external skill — never silently mkdir a
    # bogus payload from a misspelled name (resolution ran on the normalized write_path).
    if _skill_target is not None and is_skill_create_typo(
        payload_root=_skill_target.payload_root,
        bucket=_skill_target.bucket,
        rel_within_payload=_skill_target.rel_path,
    ):
        return (
            f"⚠️ DATA_WRITE_ERROR: skill payload not found: "
            f"skills/{_skill_target.bucket}/{_skill_target.skill}. Use an existing skill; for a "
            "NEW skill write its manifest (SKILL.md/skill.json) at the payload root under "
            "bucket=external; this path looks like a typo into a missing payload."
        )
    marker_payload = _skill_payload_parts(lexical_target, data_root) or _skill_payload_parts(target_path, data_root)
    should_mark_self_authored = False
    marker_path: pathlib.Path | None = None
    if (
        mode == "overwrite"
        # A genuine NEW external skill is self-authored even when reached via the bucket+skill_name
        # short-form (which synthesizes a skill_repair constraint, so the old `not skill_repair`
        # guard wrongly suppressed provenance on create). Require BOTH the manifest AND the payload
        # directory to be new (`not marker_payload[2].exists()`, evaluated before the mkdir below) so
        # writing a SKILL.md into an ALREADY-EXISTING external skill is never mis-marked self-authored.
        and marker_payload is not None
        and marker_payload[0] == "external"
        and pathlib.PurePosixPath(str(path or "")).name.lower() in {"skill.md", "skill.json"}
        and not target_path.exists()
        and not marker_payload[2].exists()
    ):
        marker_path = marker_payload[2] / _SELF_AUTHORED_MARKER
        should_mark_self_authored = not marker_path.exists()

    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "overwrite":
        # Deferral 5: block likely-accidental truncation of an existing data-plane file
        # (e.g. settings.json, skill state) unless force=true. Append is exempt.
        if (shrink := _check_data_shrink_guard(p, content, force)):
            return shrink
        write_text_atomic(p, content)  # crash-safe full overwrite (G)
    else:
        with p.open("a", encoding="utf-8") as f:
            f.write(content)  # append is intentionally NOT atomized
    if should_mark_self_authored and marker_path is not None:
        from ouroboros.skill_loader import compute_content_hash

        marker_payload[2].mkdir(parents=True, exist_ok=True)
        try:
            initial_hash = compute_content_hash(marker_payload[2])
        except Exception:
            initial_hash = ""
        marker_payload_data = {
            "schema_version": 1,
            "origin": "self_authored",
            "created_at": utc_now_iso(),
            "chat_id": int(getattr(ctx, "current_chat_id", 0) or 0),
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "created_by_tool": "data_write",
            "initial_content_hash": initial_hash,
        }
        atomic_write_json(marker_path, marker_payload_data, trailing_newline=True)
        state_marker = pathlib.Path(ctx.drive_root) / "state" / "skills" / marker_payload[1] / "self_authored.json"
        state_marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_marker, marker_payload_data, trailing_newline=True)
    result = f"OK: wrote {mode} {_root_display_path(display_root, write_path)} ({len(content)} chars)"
    if short_form.ignored_reason:
        result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
    return result


def _access_or_block(ctx: ToolContext, root: str, operation: str) -> tuple[str, str]:
    try:
        normalized = normalize_root(root)
    except ValueError as exc:
        return "", f"⚠️ TOOL_ARG_ERROR: {exc}"
    profile = active_tool_profile(ctx)
    decision = decide_tool_access(profile=profile, root=normalized, operation=operation)  # type: ignore[arg-type]
    if not decision.allow:
        return "", f"⚠️ TOOL_ACCESS_BLOCKED: {decision.reason}"
    return normalized, ""


def _local_readonly_resource_block(
    ctx: ToolContext,
    normalized: str,
    target: pathlib.Path,
    base: pathlib.Path,
    *,
    action: str,
) -> str:
    # Resource (active_workspace/system_repo) restriction is for STRICT read-only
    # subagents only — acting children legitimately write their isolated surface.
    from ouroboros.tool_access import active_tool_profile
    if active_tool_profile(ctx) != "local_readonly_subagent":
        return ""
    if normalized in {"active_workspace", "system_repo"}:
        if _is_subagent_secret_repo_target(target, pathlib.Path(base)):
            return f"⚠️ {action}_BLOCKED: this subagent cannot access repo secret or control paths."
        return ""
    if normalized in {"runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files"}:
        root = pathlib.Path(base).resolve(strict=False)
        try:
            rel = pathlib.Path(target).resolve(strict=False).relative_to(root).as_posix()
        except (OSError, ValueError):
            rel = str(target).replace(os.sep, "/")
        data_root = pathlib.Path(ctx.drive_root).resolve(strict=False)
        if (
            _is_subagent_secret_data_path(rel)
            or _is_skill_owner_state_target(target, data_root)
            or is_skill_owner_state_alias(target, data_root)
        ):
            return f"⚠️ {action}_BLOCKED: this subagent cannot access secret or owner-control data files."
    return ""


def _root_display_path(root: str, path: str) -> str:
    rel = safe_relpath(str(path or "."))
    if rel.startswith("./"):
        rel = rel[2:]
    return f"{root}:{rel or '.'}"


def _join_write_results(results: List[str]) -> str:
    rendered = "\n".join(results) if results else "⚠️ TOOL_ARG_ERROR: files must contain {path, content} objects."
    if any(str(line).lstrip().startswith("⚠️") for result in results for line in str(result).splitlines()):
        return "⚠️ WRITE_FILE_BATCH_PARTIAL_FAILURE: one or more writes failed.\n" + rendered
    return rendered


def _protected_artifact_write_block(
    ctx: ToolContext,
    root: str,
    paths: List[str],
    *,
    bucket: str = "",
    skill_name: str = "",
    prefix: str,
) -> str:
    for rel_path in paths:
        if not str(rel_path or "").strip():
            continue
        try:
            target = resolve_resource_path(ctx, root=root, path=str(rel_path), bucket=bucket, skill_name=skill_name)
        except Exception:
            continue
        block_reason = block_reason_for_path(ctx, target, "write")
        if block_reason:
            return f"⚠️ {prefix}: protected artifact path blocked: {block_reason}"
    return ""


def _protected_artifact_list_block(
    ctx: ToolContext,
    root: str,
    path: str,
    *,
    bucket: str = "",
    skill_name: str = "",
) -> str:
    try:
        target = resolve_resource_path(ctx, root=root, path=path, bucket=bucket, skill_name=skill_name)
    except Exception:
        return ""
    direct_block = block_reason_for_path(ctx, target, "static_introspection")
    if direct_block:
        return direct_block
    return ""


def _annotate_reread(ctx: ToolContext, target: Any, start_line: int, max_lines: int, result: str) -> str:
    """Append an advisory hint when the SAME file slice is re-read unchanged.

    Per-task, key on (resolved path, slice); the change signal is (size, mtime).
    A repeat read of an unchanged slice is usually wasted budget — nudge the model
    to act on what it has. Advisory only (never blocks; different slices and
    changed files are not flagged)."""
    try:
        resolved = pathlib.Path(target).resolve(strict=False)
        st = resolved.stat()
    except (OSError, TypeError, ValueError):
        return result
    if not isinstance(result, str) or result.startswith("⚠️"):
        return result
    key = f"{resolved}|{int(start_line)}|{int(max_lines)}"
    sig = (st.st_size, st.st_mtime_ns)
    seen = getattr(ctx, "_read_file_seen", None)
    if not isinstance(seen, dict):
        seen = {}
        ctx._read_file_seen = seen
    prev = seen.get(key)
    seen[key] = sig
    if prev is not None and prev == sig:
        return (
            result
            + "\n\nℹ️ This exact view is unchanged since you already read it this task — "
            "re-reading is usually wasted budget; act on what you have."
        )
    return result


def _room_lens_target(ctx: ToolContext, path: str) -> Optional[pathlib.Path]:
    """Resolve ``path`` under the project-room lens root (direct-chat folder-room,
    v6.61.3), confined inside it. None when the lens is inactive."""
    room = project_room_lens_dir(ctx)
    if room is None:
        return None
    rel = normalize_root_relative(room, str(path or "."))
    try:
        resolved = (room / safe_relpath(rel)).resolve(strict=False)
        resolved.relative_to(room)
    except (ValueError, OSError):
        # Traversal/escape out of the room folder: confine to the room root
        # itself rather than silently reaching elsewhere.
        return room
    return resolved


def _read_file(
    ctx: ToolContext,
    path: str,
    root: str = "active_workspace",
    max_lines: int = 2000,
    start_line: int = 1,
    bucket: str = "",
    skill_name: str = "",
) -> str:
    normalized, block = _access_or_block(ctx, root, "read")
    if block:
        return block
    if normalized == "active_workspace":
        _room_target = _room_lens_target(ctx, path)
        if _room_target is not None:
            # Room lens: reads in a folder-room chat resolve to the PROJECT FOLDER,
            # disclosed via the absolute display path (self-repo reads stay
            # available through root="system_repo").
            protected_block = block_reason_for_path(ctx, _room_target, "read_bytes")
            if protected_block:
                return protected_block
            try:
                content = read_text(_room_target)
            except FileNotFoundError:
                return f"⚠️ NOT_FOUND: {_room_target} (project-room folder)"
            except Exception as exc:
                return f"⚠️ READ_FILE_ERROR: {type(exc).__name__}: {exc}"
            return _annotate_reread(ctx, _room_target, start_line, max_lines, _render_line_slice(
                f"{_room_target} (project room)", content, max_lines=max_lines, start_line=start_line,
            ))
        target = ctx.repo_path(path)
        protected_block = block_reason_for_path(ctx, target, "read_bytes")
        if protected_block:
            return protected_block
        return _annotate_reread(ctx, target, start_line, max_lines, _repo_read(
            ctx,
            path,
            max_lines=max_lines,
            start_line=start_line,
            display_path=_root_display_path(normalized, path),
        ))
    if normalized == "runtime_data":
        try:
            target = resolve_resource_path(ctx, root=normalized, path=path)
            protected_block = block_reason_for_path(ctx, target, "read_bytes")
            if protected_block:
                return protected_block
        except Exception:
            pass
        return _annotate_reread(ctx, locals().get("target"), start_line, max_lines, _data_read(
            ctx,
            path,
            max_lines=max_lines,
            start_line=start_line,
            display_path=_root_display_path(normalized, path),
        ))
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if normalized == "skill_payload" and not bucket and not skill_name and task_constraint and task_constraint.mode == "skill_repair":
        try:
            target = resolve_resource_path(ctx, root=normalized, path=path, bucket=bucket, skill_name=skill_name)
            protected_block = block_reason_for_path(ctx, target, "read_bytes")
            if protected_block:
                return protected_block
        except Exception:
            pass
        return _annotate_reread(ctx, locals().get("target"), start_line, max_lines, _data_read(ctx, path, max_lines=max_lines, start_line=start_line, display_path=_root_display_path(normalized, path)))
    try:
        base = resource_root_path(ctx, normalized, bucket=bucket, skill_name=skill_name)
        target = resolve_resource_path(ctx, root=normalized, path=path, bucket=bucket, skill_name=skill_name)
        protected_block = block_reason_for_path(ctx, target, "read_bytes")
        if protected_block:
            return protected_block
        block_msg = _local_readonly_resource_block(ctx, normalized, target, base, action="READ_FILE")
        if block_msg:
            return block_msg
        content = read_text(target)
        return _annotate_reread(ctx, target, start_line, max_lines, _render_line_slice(_root_display_path(normalized, path), content, max_lines=max_lines, start_line=start_line))
    except FileNotFoundError:
        return f"⚠️ NOT_FOUND: {_root_display_path(normalized, path)}"
    except Exception as exc:
        return f"⚠️ READ_FILE_ERROR: {type(exc).__name__}: {exc}"


def _list_files(
    ctx: ToolContext,
    path: str = ".",
    root: str = "active_workspace",
    max_entries: int = 500,
    bucket: str = "",
    skill_name: str = "",
) -> str:
    normalized, block = _access_or_block(ctx, root, "list")
    if block:
        return block
    protected_list_block = _protected_artifact_list_block(ctx, normalized, path, bucket=bucket, skill_name=skill_name)
    if protected_list_block:
        return protected_list_block
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    try:
        # Every listing branch runs inside this try: a hard iterdir/permission/
        # race failure from any helper becomes the first-class LIST_FILES_ERROR
        # below (v6.54.3, review round 3 — helpers no longer swallow it into an
        # ok-shaped listing).
        if normalized == "active_workspace":
            _room = project_room_lens_dir(ctx)
            if _room is not None:
                # Room lens: the folder-room chat lists the PROJECT FOLDER (the
                # robot incident: "." listed the system repo and the agent
                # narrated the wrong tree). Same JSON-array shape as every root.
                _rel = normalize_root_relative(_room, str(path or "."))
                return json.dumps(_list_dir(_room, _rel, max_entries), ensure_ascii=False, indent=2)
            return _repo_list(ctx, dir=path, max_entries=max_entries)
        if normalized == "runtime_data":
            return _data_list(ctx, dir=path, max_entries=max_entries)
        if normalized == "skill_payload" and not bucket and not skill_name and task_constraint and task_constraint.mode == "skill_repair":
            return _data_list(ctx, dir=path, max_entries=max_entries)
        base = resource_root_path(ctx, normalized, bucket=bucket, skill_name=skill_name)
        if normalized == "user_files":
            target = resolve_user_file_path(ctx, path, allow_protected_descendants=True)
            items = _list_user_files_dir(ctx, base, target, max_entries)
            return json.dumps(items, ensure_ascii=False, indent=2)
        # Normalize a redundant-prefix/absolute path only for the repo roots; the
        # protected-artifact list guard above (_protected_artifact_list_block) reads
        # the RAW path, so normalizing a non-repo root here would desync them.
        list_path = normalize_root_relative(base, path) if normalized in ("active_workspace", "system_repo") else path
        items = _list_dir(base, list_path, max_entries)
        if is_restricted_subagent_profile(ctx):
            if normalized == "system_repo":
                items = _filter_subagent_secret_repo_listing(items, base)
            elif normalized in {"task_drive", "skill_payload", "artifact_store", "user_files"}:
                items = _filter_subagent_secret_listing(items, base)
        return json.dumps(items, ensure_ascii=False, indent=2)
    except _ListingFailure as exc:
        return f"⚠️ LIST_FILES_ERROR: {exc}"
    except Exception as exc:
        # A hard failure is a first-class tool error, never a JSON "listing" that
        # reads as success with an error string inside (v6.54.3: that shape
        # silently poisoned reasoning in 63% of TB2.1 trials).
        return f"⚠️ LIST_FILES_ERROR ({type(exc).__name__}): {exc}"


def _write_file(
    ctx: ToolContext,
    path: str = "",
    content: str = "",
    files: List[Dict[str, str]] | None = None,
    root: str = "active_workspace",
    mode: str = "overwrite",
    force: bool = False,
    bucket: str = "",
    skill_name: str = "",
) -> str:
    normalized, block = _access_or_block(ctx, root, "write")
    if block:
        return block
    if normalized == "system_repo":
        try:
            from ouroboros.tool_access import resource_root_path

            active_root = resource_root_path(ctx, "active_workspace")
            system_root = resource_root_path(ctx, "system_repo")
            if active_root.resolve(strict=False) != system_root.resolve(strict=False):
                return "⚠️ WRITE_FILE_BLOCKED: root=system_repo writes require the active workspace to be the system repo."
        except Exception as exc:
            return f"⚠️ WRITE_FILE_BLOCKED: could not validate system_repo root: {type(exc).__name__}: {exc}"
    write_paths = [path]
    for item in files or []:
        if isinstance(item, dict):
            write_paths.append(str(item.get("path") or ""))
    protected_block = _protected_artifact_write_block(
        ctx,
        normalized,
        write_paths,
        bucket=bucket,
        skill_name=skill_name,
        prefix="WRITE_FILE_BLOCKED",
    )
    if protected_block:
        return protected_block
    if normalized == "active_workspace" and (_room := project_room_lens_dir(ctx)) is not None:
        # Room write-guard (v6.61.3): with the lens re-pointing reads at the room
        # folder, a default-root write silently landing in the SYSTEM REPO would be
        # a read/write split trap (read game.js from the folder, "fix" it into the
        # repo). Mutations belong to promoted tasks; deliberate self-repo writes
        # stay available via the explicit root.
        return (
            f"⚠️ ROOM_WRITE_VIA_TASK: this room's files live in {_room} and are edited by "
            "PROMOTED tasks — call promote_chat_to_task (it inherits the room folder as its "
            "workspace) for real work there. For a deliberate write to the Ouroboros system "
            'repo, pass root="system_repo" explicitly.'
        )
    if normalized in {"active_workspace", "system_repo"}:
        from ouroboros.tools.git import _repo_write

        return _repo_write(ctx, path=path, content=content, files=files or [], force=force, display_root=normalized)
    if normalized == "runtime_data":
        if files:
            results = []
            for item in files:
                if not isinstance(item, dict):
                    continue
                results.append(_data_write(
                    ctx,
                    str(item.get("path") or ""),
                    str(item.get("content") or ""),
                    mode=mode,
                    display_root=normalized,
                    force=force,
                ))
            return _join_write_results(results)
        return _data_write(ctx, path=path, content=content, mode=mode, display_root=normalized, force=force)
    if normalized == "skill_payload":
        if files:
            results = []
            for item in files:
                rel = str(item.get("path") or "") if isinstance(item, dict) else ""
                body = str(item.get("content") or "") if isinstance(item, dict) else ""
                results.append(_data_write(
                    ctx,
                    rel,
                    body,
                    mode=mode,
                    bucket=bucket,
                    skill_name=skill_name,
                    display_root=normalized,
                    force=force,
                ))
            return _join_write_results(results)
        return _data_write(ctx, path=path, content=content, mode=mode, bucket=bucket, skill_name=skill_name, display_root=normalized, force=force)
    try:
        if files:
            results = []
            for item in files:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "")
                target = resolve_resource_path(ctx, root=normalized, path=rel_path, bucket=bucket, skill_name=skill_name)
                if normalized == "artifact_store":
                    block_reason = artifact_store_path_block_reason(target)
                    if block_reason:
                        results.append(f"⚠️ WRITE_FILE_BLOCKED: artifact_store path blocked: {block_reason}")
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Deferral 5: batch items overwrite too — shrink-guard each (parity with the
                # single-file path), force=true bypasses.
                if (shrink := _check_data_shrink_guard(target, str(item.get("content") or ""), force)):
                    results.append(shrink)
                    continue
                write_text_atomic(target, str(item.get("content") or ""))  # crash-safe (G)
                result = f"OK: wrote {_root_display_path(normalized, rel_path)} ({len(str(item.get('content') or ''))} chars)"
                if normalized == "user_files":
                    record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
                    if record:
                        result += f"\nARTIFACT_OUTPUTS: registered user file -> artifact_store:{record.get('name')}"
                results.append(result)
            return _join_write_results(results)
        target = resolve_resource_path(ctx, root=normalized, path=path, bucket=bucket, skill_name=skill_name)
        if normalized == "artifact_store":
            block_reason = artifact_store_path_block_reason(target)
            if block_reason:
                return f"⚠️ WRITE_FILE_BLOCKED: artifact_store path blocked: {block_reason}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content)  # append is intentionally NOT atomized
        else:
            # Deferral 5: shrink-guard the full overwrite (e.g. active_workspace rewrites)
            # — force=true bypasses, matching the tool-schema `force` description.
            if (shrink := _check_data_shrink_guard(target, content, force)):
                return shrink
            write_text_atomic(target, content)  # crash-safe full overwrite (G)
        result = f"OK: wrote {_root_display_path(normalized, path)} ({len(content)} chars)"
        if normalized == "user_files":
            record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
            if record:
                result += f"\nARTIFACT_OUTPUTS: registered user file -> artifact_store:{record.get('name')}"
        return result
    except Exception as exc:
        return f"⚠️ WRITE_FILE_ERROR: {type(exc).__name__}: {exc}"


def _edit_text(
    ctx: ToolContext,
    path: str,
    old_str: str,
    new_str: str,
    root: str = "active_workspace",
    bucket: str = "",
    skill_name: str = "",
    force: bool = False,
) -> str:
    normalized, block = _access_or_block(ctx, root, "edit")
    if block:
        return block
    if normalized == "system_repo":
        try:
            from ouroboros.tool_access import resource_root_path

            active_root = resource_root_path(ctx, "active_workspace")
            system_root = resource_root_path(ctx, "system_repo")
            if active_root.resolve(strict=False) != system_root.resolve(strict=False):
                return "⚠️ EDIT_TEXT_BLOCKED: root=system_repo edits require the active workspace to be the system repo."
        except Exception as exc:
            return f"⚠️ EDIT_TEXT_BLOCKED: could not validate system_repo root: {type(exc).__name__}: {exc}"
    protected_block = _protected_artifact_write_block(
        ctx,
        normalized,
        [path],
        bucket=bucket,
        skill_name=skill_name,
        prefix="EDIT_TEXT_BLOCKED",
    )
    if protected_block:
        return protected_block
    if normalized == "active_workspace" and (_room := project_room_lens_dir(ctx)) is not None:
        # Room write-guard (v6.61.3) — same rule as write_file: room mutations go
        # through promoted tasks; explicit root="system_repo" for the self-repo.
        return (
            f"⚠️ ROOM_WRITE_VIA_TASK: this room's files live in {_room} and are edited by "
            "PROMOTED tasks — call promote_chat_to_task (it inherits the room folder as its "
            "workspace) for real work there. For a deliberate edit of the Ouroboros system "
            'repo, pass root="system_repo" explicitly.'
        )
    if normalized in {"active_workspace", "system_repo"}:
        from ouroboros.tools.git import _str_replace_editor

        result = _str_replace_editor(ctx, path=path, old_str=old_str, new_str=new_str, display_root=normalized)
        short_form = decide_payload_short_form(
            bucket=bucket,
            skill_name=skill_name,
            path_text=path,
            repo_dir=pathlib.Path(ctx.repo_dir),
            drive_root=pathlib.Path(ctx.drive_root),
        )
        if short_form.ignored_reason:
            result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
        return result
    if normalized == "skill_payload":
        from ouroboros.tools.git import _str_replace_editor

        # Deferral 5: skill payloads live under data/skills/ (not the repo git), so
        # git._str_replace_editor's git-ls-files shrink check never fires for them. Apply
        # the data-plane shrink guard here (pre-checking the prospective replacement with
        # the shared matcher) before delegating, so a payload edit can't silently truncate.
        try:
            _sp_target = resolve_resource_path(ctx, root=normalized, path=path, bucket=bucket, skill_name=skill_name)
            if _sp_target.exists():
                _sp_new, _sp_err = _str_match_replace(
                    _sp_target.read_text(encoding="utf-8"), old_str, new_str,
                    _root_display_path(normalized, path), "EDIT_TEXT_ERROR",
                )
                if _sp_err:
                    return _sp_err
                if (shrink := _check_data_shrink_guard(_sp_target, _sp_new, force)):
                    return shrink
        except Exception:
            log.debug("skill_payload shrink pre-check skipped", exc_info=True)
        return _str_replace_editor(
            ctx,
            path=path,
            old_str=old_str,
            new_str=new_str,
            bucket=bucket,
            skill_name=skill_name,
            display_root=normalized,
        )
    try:
        target = resolve_resource_path(ctx, root=normalized, path=path, bucket=bucket, skill_name=skill_name)
        if normalized == "runtime_data":
            if (b := _project_store_access_block(_normalize_data_read_path(ctx, path))):
                return b
            data_root = pathlib.Path(ctx.drive_root).resolve(strict=False)
            if _is_workspace_executor_control_state_path(target, data_root):
                return (
                    "⚠️ EDIT_TEXT_BLOCKED: workspace executor process records are "
                    "owner/runtime control-plane state. Use process/service lifecycle "
                    "tools instead of editing state/workspace_executor_processes directly."
                )
        if normalized == "artifact_store":
            block_reason = artifact_store_path_block_reason(target)
            if block_reason:
                return f"⚠️ EDIT_TEXT_BLOCKED: artifact_store path blocked: {block_reason}"
        text = target.read_text(encoding="utf-8")
        new_text, _match_err = _str_match_replace(
            text, old_str, new_str, _root_display_path(normalized, path), "EDIT_TEXT_ERROR"
        )
        if _match_err:
            return _match_err  # count==0 preview / count>1 positional hints (deferral 4)
        # Deferral 5: an exact replace that shrinks an existing data-plane file >30% is
        # likely accidental truncation — block unless force=true (matches the overwrite
        # paths; force lets a deliberate large surgical deletion through).
        if (shrink := _check_data_shrink_guard(target, new_text, force)):
            return shrink
        write_text_atomic(target, new_text)  # crash-safe edit (G)
        result = f"OK: edited {_root_display_path(normalized, path)}"
        if normalized == "user_files":
            record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
            if record:
                result += f"\nARTIFACT_OUTPUTS: registered user file -> artifact_store:{record.get('name')}"
        return result
    except FileNotFoundError:
        return f"⚠️ EDIT_TEXT_ERROR: file not found: {_root_display_path(normalized, path)}"
    except Exception as exc:
        return f"⚠️ EDIT_TEXT_ERROR: {type(exc).__name__}: {exc}"

_MAX_PHOTO_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def _detect_image_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'GIF8':
        return "image/gif"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    return "application/octet-stream"


def _send_photo(ctx: ToolContext, file_path: str = "", image_base64: str = "",
                caption: str = "") -> str:
    """Queue an owner-chat image from a file or legacy base64 payload."""
    if not ctx.current_chat_id:
        return "⚠️ No active chat — cannot send photo."

    actual_b64 = ""
    mime = "image/png"

    if file_path:
        fp = pathlib.Path(file_path).expanduser().resolve()
        if not fp.exists():
            return f"⚠️ File not found: {file_path}"
        if fp.stat().st_size > _MAX_PHOTO_FILE_BYTES:
            return f"⚠️ File too large ({fp.stat().st_size} bytes). Max: {_MAX_PHOTO_FILE_BYTES} bytes."
        try:
            raw = fp.read_bytes()
            mime = _detect_image_mime(raw)
            actual_b64 = __import__("base64").b64encode(raw).decode()
        except Exception as e:
            return f"⚠️ Failed to read image file: {e}"
    elif image_base64:
        if image_base64 == "__last_screenshot__":
            if not ctx.browser_state.last_screenshot_b64:
                return "⚠️ No screenshot stored. Take one first with browse_page(output='screenshot')."
            actual_b64 = ctx.browser_state.last_screenshot_b64
        else:
            actual_b64 = image_base64
    else:
        return "⚠️ Provide either file_path or image_base64."

    if not actual_b64 or len(actual_b64) < 100:
        return "⚠️ Image data is empty or too short."

    _photo_meta = getattr(ctx, "task_metadata", {})
    _photo_meta = _photo_meta if isinstance(_photo_meta, dict) else {}
    ctx.pending_events.append({
        "type": "send_photo",
        "chat_id": ctx.current_chat_id, "task_id": str(getattr(ctx, "task_id", "") or ""),  # task_id -> bound-task project-panel routing
        # Lineage so a SUBAGENT's photo routes to its root's project thread (C4.4) —
        # only the root is bound; the child carries parent/root on its task metadata.
        "parent_task_id": str(_photo_meta.get("parent_task_id") or ""),
        "root_task_id": str(_photo_meta.get("root_task_id") or ""),
        "image_base64": actual_b64,
        "mime": mime,
        "caption": caption or "",
    })
    return "OK: photo queued for delivery to owner."


_MAX_VIDEO_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


def _detect_video_mime(file_path: str, data: bytes) -> str:
    """Detect video MIME type from path extension or magic bytes."""
    if len(data) >= 8 and data[4:8] == b'ftyp':
        return "video/mp4"
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return "video/webm"
    mime, _ = __import__("mimetypes").guess_type(file_path)
    if mime and str(mime).lower().startswith("video/"):
        return mime
    return "video/mp4"


def _send_video(ctx: ToolContext, file_path: str = "", caption: str = "") -> str:
    """Queue an owner-chat video from a file."""
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None or chat_id == "":
        return "⚠️ No active chat — cannot send video."
    if not file_path:
        return "⚠️ Provide a file_path."

    fp = pathlib.Path(file_path).expanduser().resolve()
    if not fp.exists():
        return f"⚠️ File not found: {file_path}"
    if fp.stat().st_size > _MAX_VIDEO_FILE_BYTES:
        return f"⚠️ File too large ({fp.stat().st_size} bytes). Max: {_MAX_VIDEO_FILE_BYTES} bytes."

    try:
        raw = fp.read_bytes()
        mime = _detect_video_mime(str(fp), raw)
        actual_b64 = __import__("base64").b64encode(raw).decode()
    except Exception as e:
        return f"⚠️ Failed to read video file: {e}"

    _video_meta = getattr(ctx, "task_metadata", {})
    _video_meta = _video_meta if isinstance(_video_meta, dict) else {}
    ctx.pending_events.append({
        "type": "send_video",
        "chat_id": chat_id, "task_id": str(getattr(ctx, "task_id", "") or ""),  # task_id -> bound-task project-panel routing
        # Lineage so a SUBAGENT's video routes to its root's project thread (C4.4).
        "parent_task_id": str(_video_meta.get("parent_task_id") or ""),
        "root_task_id": str(_video_meta.get("root_task_id") or ""),
        "video_base64": actual_b64,
        "mime": mime,
        "caption": caption or "",
    })
    return "OK: video queued for delivery to owner."

_MAX_SEARCH_RESULTS = 200
# Search file-skip helper and caps live in ouroboros.code_search_rg (the search
# module SSOT); imported with the historical private names used by call sites.
from ouroboros.code_search_rg import (  # noqa: E402
    MAX_SEARCH_FILES_SCANNED as _MAX_SEARCH_FILES_SCANNED,
    _search_wall_clock_sec,
    is_search_skippable as _is_search_skippable,
)


def _code_search(ctx: ToolContext, query: str, path: str = ".",
                 regex: bool = False, max_results: int = 200,
                 include: str = "", root: str = "active_workspace",
                 bucket: str = "", skill_name: str = "") -> str:
    """Search repo text with optional regex, path, glob, and result cap."""
    if not query:
        return "⚠️ SEARCH_ERROR: query is required."
    normalized, block = _access_or_block(ctx, root, "search")
    if block:
        return block
    if normalized == "runtime_data" and (b := _project_store_access_block(_normalize_data_read_path(ctx, path))):
        return b

    max_results = min(max(1, max_results), _MAX_SEARCH_RESULTS)
    try:
        root_path = resource_root_path(ctx, normalized, bucket=bucket, skill_name=skill_name)
    except Exception as exc:
        return f"⚠️ SEARCH_ERROR: {type(exc).__name__}: {exc}"
    if normalized == "active_workspace":
        _room = project_room_lens_dir(ctx)
        if _room is not None:
            # Room lens: folder-room chat searches the PROJECT FOLDER (self-repo
            # searches stay available via root="system_repo").
            root_path = _room
    if normalized in ("active_workspace", "system_repo"):
        # Accept absolute/redundant-prefix paths inside the repo root (e.g. '/app/x'
        # or 'app/x' under a root at /app); confinement stays via safe_relpath below.
        # ONLY the repo roots: the runtime_data project-store guard above matches the
        # RAW path (via _normalize_data_read_path, which does not strip a bare
        # basename), so normalizing a non-repo root here would let
        # search_code(root='runtime_data', path='<drive_basename>/projects/...') slip
        # the guard and then search the normalized 'projects/...' store.
        path = normalize_root_relative(root_path, path)
    display_search_path = _root_display_path(normalized, path)
    try:
        search_root = (
            resolve_user_file_path(ctx, path, allow_protected_descendants=True)
            if normalized == "user_files"
            else (root_path / safe_relpath(path)).resolve()
        )
    except Exception as exc:
        return f"⚠️ SEARCH_ERROR: {type(exc).__name__}: {exc}"
    if not search_root.exists():
        return f"⚠️ SEARCH_ERROR: path not found: {display_search_path}"
    if normalized != "user_files":
        # Reject a search ROOT that escapes its resource root (e.g. the requested path is an
        # in-tree symlink pointing outside — untrusted child project/deliverable trees) BEFORE
        # any rg/os.walk. Parity with _list_dir + the per-file _path_allowed_for_rg guard.
        try:
            search_root.relative_to(root_path.resolve(strict=False))
        except ValueError:
            return f"⚠️ SEARCH_ERROR: path escapes root: {display_search_path}"
    protected_root_block = block_reason_for_path(ctx, search_root, "static_introspection")
    if protected_root_block:
        return protected_root_block
    protected_root_read_block = block_reason_for_path(ctx, search_root, "read_bytes")
    if protected_root_read_block and search_root.is_file():
        return protected_root_read_block
    subagent_readonly = is_restricted_subagent_profile(ctx)
    if subagent_readonly:
        block_msg = _local_readonly_resource_block(ctx, normalized, search_root, root_path, action="SEARCH")
        if block_msg:
            return block_msg
    root_resolved = root_path.resolve(strict=False)
    _rt_search_root = str(root_resolved) if normalized == "runtime_data" else ""

    def _path_allowed_for_rg(fp: pathlib.Path) -> bool:
        # Resolve, then CONFINE to the resource root: a path whose resolved target
        # escapes it (e.g. an in-root symlink to outside) is rejected — no leak.
        try:
            fp = pathlib.Path(fp).resolve(strict=False)
            rel_parts = fp.relative_to(root_resolved).parts
        except Exception:
            return False
        # runtime_data per-project store is reachable only via scoped knowledge tools.
        if normalized == "runtime_data" and rel_parts and str(rel_parts[0]).casefold() == "projects":
            return False
        return not (
            (subagent_readonly and _local_readonly_resource_block(ctx, normalized, fp, root_path, action="SEARCH"))
            or (normalized == "user_files" and user_files_path_block_reason(ctx, fp))
            or block_reason_for_path(ctx, fp, "read_bytes")
            or _is_search_skippable(fp)
        )

    # Validate a regex query UP FRONT so the invalid-regex contract holds for BOTH the
    # ripgrep path and the Python fallback. ripgrep accepts some malformed patterns
    # permissively (e.g. an unterminated '[' yields "no matches" instead of erroring),
    # so without this the rg path would silently swallow an invalid regex while only the
    # fallback rejected it. Non-regex queries are matched literally and need no check.
    # (Checked before the wall-clock budget below: an invalid regex returns immediately,
    # so there is no point starting the timer for it.)
    if regex:
        try:
            re.compile(query)
        except re.error as e:
            return f"⚠️ SEARCH_ERROR: invalid regex: {e}"

    import time as _time
    _search_t0 = _time.monotonic()  # start the wall-clock budget BEFORE rg, so a
    # subsequent fallback degradation shares ONE budget (not a fresh 2nd one).
    try:
        from ouroboros.code_search_rg import format_search_result, search_with_rg

        if search_root.is_dir():
            rg_result = search_with_rg(
                search_root, query, regex=bool(regex), include=include,
                max_results=max_results, path_allowed=_path_allowed_for_rg,
            )
            return format_search_result(
                display_path=display_search_path, root_name=normalized,
                root_path=root_path, query=query, regex=bool(regex),
                max_results=max_results, result=rg_result,
            )
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError, OSError) as e:
        # Degrade to the policy-aware Python scanner for rg absent/failed/timeout
        # AND OSError (wrong-arch/non-executable bundled rg -> 'Exec format
        # error'). MemoryError etc. still propagate rather than silently degrade.
        logging.getLogger(__name__).debug("search_code: ripgrep unavailable, using fallback: %s", e)

    try:
        if regex:
            pattern = re.compile(query)
        else:
            pattern = re.compile(re.escape(query))
    except re.error as e:
        return f"⚠️ SEARCH_ERROR: invalid regex: {e}"

    matches: List[str] = []
    files_searched = 0
    protected_omitted = 0
    truncated = False
    files_capped = False
    deadline_hit = False
    # search_code on a single FILE: os.walk yields nothing for a file path, which
    # would make the search a silent no-op. Feed the scanner a one-file "walk".
    if search_root.is_file():
        _walker = [(str(search_root.parent), [], [search_root.name])]
    else:
        _walker = os.walk(str(search_root))
    # Bound TOTAL (rg attempt + this fallback walk) to one budget, but always grant
    # the fallback a small floor so an rg that ate the budget still makes some progress.
    _search_deadline = max(_search_t0 + _search_wall_clock_sec(), _time.monotonic() + 5.0)
    for dirpath, dirnames, filenames in _walker:
        # Wall-clock cap: the file-count cap bounds memory but a walk over a very
        # large root (user_files == / under a bench HOME) can traverse for minutes.
        if _time.monotonic() > _search_deadline:
            deadline_hit = True  # ran out of TIME (distinct from the file-count cap)
            break
        # Prune skipped dirs in-place. For runtime_data, also prune the top-level
        # per-project store (reachable only via the scoped knowledge tools).
        from ouroboros.code_intelligence import SKIP_DIRS

        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIRS]
        if normalized == "runtime_data" and str(pathlib.Path(dirpath).resolve(strict=False)) == _rt_search_root:
            dirnames[:] = [d for d in dirnames if d.casefold() != "projects"]
        if normalized == "user_files":
            dirnames[:] = [
                d for d in dirnames
                if not user_files_path_block_reason(ctx, pathlib.Path(dirpath) / d)
            ]
        if subagent_readonly:
            dirnames[:] = [
                d for d in dirnames
                if not _local_readonly_resource_block(ctx, normalized, pathlib.Path(dirpath) / d, root_path, action="SEARCH")
            ]

        for fname in sorted(filenames):
            fp = pathlib.Path(dirpath) / fname

            if include and not fnmatch.fnmatch(fname, include):
                continue

            if subagent_readonly and _local_readonly_resource_block(ctx, normalized, fp, root_path, action="SEARCH"):
                continue
            if normalized == "user_files" and user_files_path_block_reason(ctx, fp):
                continue
            if block_reason_for_path(ctx, fp, "read_bytes"):
                protected_omitted += 1
                continue

            if _is_search_skippable(fp):
                continue

            # CONFINE to the root before reading (parity with the rg path's _path_allowed_for_rg
            # and _list_dir): a resolved file escaping the root — e.g. an in-tree symlink in an
            # untrusted child project/deliverable tree — must never have its target read out.
            try:
                fp.resolve(strict=False).relative_to(root_resolved)
            except (OSError, ValueError):
                continue

            if files_searched >= _MAX_SEARCH_FILES_SCANNED:
                files_capped = True
                break

            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            files_searched += 1
            rel = fp.relative_to(root_path).as_posix()

            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{_root_display_path(normalized, rel)}:{lineno}: {line.rstrip()}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break
        if truncated or files_capped or deadline_hit:
            break

    # A deadline cutoff means even a "no matches" may be INCOMPLETE (parity with the rg
    # path's deadline signal) — never let a timed-out fallback read as authoritative empty.
    deadline_note = (
        " Search stopped at the time budget before the whole tree was scanned — results "
        "may be incomplete; narrow the path or glob, or raise OUROBOROS_SEARCH_CODE_WALL_SEC."
        if deadline_hit else ""
    )
    if not matches:
        suffix = f" {protected_omitted} protected artifact file(s) omitted." if protected_omitted else ""
        cap_note = f" Scan stopped after {_MAX_SEARCH_FILES_SCANNED} files — narrow the path or glob." if files_capped else ""
        return f"No matches found for {'regex' if regex else 'literal'} `{query}` in {display_search_path} ({files_searched} files searched).{suffix}{cap_note}{deadline_note}"

    header = f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} in {display_search_path} ({files_searched} files searched)"
    if files_capped:
        header += f" — scan stopped at {_MAX_SEARCH_FILES_SCANNED} files (narrow the path or glob)"
    if truncated:
        header += f" — truncated at {max_results} results"
    if deadline_hit:
        header += " — stopped at the time budget (results may be incomplete)"
    if protected_omitted:
        header += f" — {protected_omitted} protected artifact file(s) omitted"
    return header + "\n\n" + "\n".join(matches)


def _forward_to_worker(ctx: ToolContext, task_id: str, message: str) -> str:
    """Forward a message to a running worker task's mailbox."""
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.task_results import STATUS_RUNNING, validate_task_id
    from ouroboros.task_status import FINAL_STATUSES, load_effective_task_result

    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (forward_to_worker): {exc}"
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = pathlib.Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    data = load_effective_task_result(status_drive_root, tid)
    status = str(data.get("status") or "").lower()
    if not data:
        return f"⚠️ TASK_NOT_FOUND: task {tid} is not registered."
    if status in FINAL_STATUSES:
        return f"⚠️ TASK_NOT_ACTIVE: task {tid} is already {status}."
    if status != STATUS_RUNNING:
        return f"⚠️ TASK_NOT_ACTIVE: task {tid} is {status or 'unknown'}, not running."
    current_task_id = str(getattr(ctx, "task_id", "") or "").strip()
    target_parent = str(data.get("parent_task_id") or "").strip()
    target_root = str(data.get("root_task_id") or "").strip()
    if not current_task_id:
        return "⚠️ TASK_FORBIDDEN: forward_to_worker requires an active task context."
    allowed = target_parent == current_task_id or target_root == current_task_id
    if not allowed:
        return f"⚠️ TASK_FORBIDDEN: task {tid} is not a child or descendant of the current task."
    child_drive = str(data.get("child_drive_root") or data.get("headless_child_drive_root") or data.get("drive_root") or "").strip()
    mailbox_drive = pathlib.Path(child_drive) if child_drive else pathlib.Path(ctx.drive_root)
    write_owner_message(mailbox_drive, message, task_id=tid, msg_id=uuid.uuid4().hex)
    return f"Message forwarded to task {tid}"

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("read_file", {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from a declared resource root. "
                "Default root=active_workspace (the user's workspace or the Ouroboros repo in self-modification tasks). "
                "Use max_lines (default 2000) and start_line (default 1) to read large files in chunks. "
                "The result header shows root:path and 'lines X\u2013Y of Z' so you know where and how much you read. "
                "Prefer this over cat/head/sed-as-reader in run_command; to locate code first use query_code "
                "(symbols/definitions/callers) or search_code (text/regex), then read_file the hit."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files", "subagent_projects", "deliverables"], "default": "active_workspace"},
                "max_lines": {"type": "integer", "default": 2000,
                              "description": "Maximum number of lines to return (default 2000)."},
                "start_line": {"type": "integer", "default": 1,
                               "description": "1-indexed line to start reading from (default 1 = beginning)."},
                "bucket": {"type": "string", "description": "Required only for root=skill_payload."},
                "skill_name": {"type": "string", "description": "Required only for root=skill_payload."},
            }, "required": ["path"]},
        }, _read_file),
        ToolEntry("list_files", {
            "name": "list_files",
            "description": "List files under a resource root directory.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "default": "."},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files", "subagent_projects", "deliverables"], "default": "active_workspace"},
                "max_entries": {"type": "integer", "default": 500},
                "bucket": {"type": "string", "description": "Required only for root=skill_payload."},
                "skill_name": {"type": "string", "description": "Required only for root=skill_payload."},
            }, "required": []},
        }, _list_files),
        ToolEntry("write_file", {
            "name": "write_file",
            "description": (
                "Write UTF-8 file(s) to a declared resource root. "
                "Default root=active_workspace. "
                "OK messages show root:path. "
                "Use mode='append' to write a large file in chunks across multiple calls "
                "(useful when the full content exceeds a single LLM output budget). "
                "Set bucket/skill_name ONLY for root=skill_payload (skill authoring); leave empty for normal file edits."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "files": {"type": "array", "items": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                }, "required": ["path", "content"]}},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files"], "default": "active_workspace"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
                "force": {"type": "boolean", "default": False, "description": "Bypass the shrink guard for an intentional full rewrite on any root where it applies (active_workspace via the repo guard; runtime_data/task_drive/skill_payload/artifact_store/user_files via the data-plane guard)."},
                "bucket": {
                    "type": "string",
                    "description": "Skill payload bucket — set ONLY when root=skill_payload (skill authoring); leave empty for normal file edits.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill slug — set ONLY when root=skill_payload; leave empty otherwise.",
                },
            }, "required": []},
        }, _write_file, is_code_tool=True),
        ToolEntry("edit_text", {
            "name": "edit_text",
            "description": (
                "Replace exactly one occurrence of old_str with new_str in a file. "
                "Default root=active_workspace. Result messages show root:path. "
                "Set bucket/skill_name ONLY for root=skill_payload (skill authoring); leave empty for normal edits."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files"], "default": "active_workspace"},
                "bucket": {"type": "string", "description": "Skill payload bucket — set ONLY when root=skill_payload; leave empty otherwise."},
                "skill_name": {"type": "string", "description": "Skill slug — set ONLY when root=skill_payload; leave empty otherwise."},
                "force": {"type": "boolean", "default": False, "description": "Bypass the shrink guard for a deliberate large data-plane deletion (>30% smaller)."},
            }, "required": ["path", "old_str", "new_str"]},
        }, _edit_text, is_code_tool=True),
        ToolEntry("send_photo", {
            "name": "send_photo",
            "description": (
                "Send an image to the owner's chat. "
                "Preferred: use file_path to send a local file. "
                "Legacy: use image_base64 with raw base64 or __last_screenshot__. "
                "Use after browse_page(output='screenshot') or browser_action(action='screenshot')."
            ),
            "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string", "description": "Local file path to image (preferred)"},
                "image_base64": {"type": "string", "description": "Base64-encoded image data or __last_screenshot__"},
                "caption": {"type": "string", "description": "Optional caption for the photo"},
            }, "required": []},
        }, _send_photo),
        ToolEntry("send_video", {
            "name": "send_video",
            "description": "Send a video to the owner's chat (e.g. an anime animation). Requires a local file_path.",
            "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string", "description": "Local file path to video (preferred)"},
                "caption": {"type": "string", "description": "Optional caption for the video"},
            }, "required": ["file_path"]},
        }, _send_video),
        ToolEntry("search_code", {
            "name": "search_code",
            "description": (
                "Search for a pattern in the repository code. "
                "Literal search by default; set regex=True for regular expressions. "
                "Scoped to path (default: entire active workspace). "
                "Skips binaries, caches, vendor dirs, and files >1MB. "
                "Returns up to max_results matches (default 200) with root:file:line context. "
                "Use this for plain-text/regex matches (prefer it over grep/find-as-search in run_command); "
                "for symbol-aware lookups (definitions, callers, references, impact) prefer query_code."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search pattern (literal or regex)"},
                "path": {"type": "string", "default": ".", "description": "Subdirectory to search (relative to repo root)"},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store", "user_files", "subagent_projects", "deliverables"], "default": "active_workspace"},
                "bucket": {"type": "string", "description": "Required only for root=skill_payload."},
                "skill_name": {"type": "string", "description": "Required only for root=skill_payload."},
                "regex": {"type": "boolean", "default": False, "description": "Treat query as a regular expression"},
                "max_results": {"type": "integer", "default": 200, "description": "Maximum number of matches to return (max 200)"},
                "include": {"type": "string", "default": "", "description": "Filter by glob pattern (e.g. '*.py')"},
            }, "required": ["query"]},
        }, _code_search),
        ToolEntry("forward_to_worker", {
            "name": "forward_to_worker",
            "description": (
                "Forward a message to a running worker task's mailbox. "
                "Use when my human sends a message during your active conversation "
                "that is relevant to a specific running background task. "
                "The worker will see it as [Message from my human] on its next LLM round."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "ID of the running task to forward to"},
                "message": {"type": "string", "description": "Message text to forward"},
            }, "required": ["task_id", "message"]},
        }, _forward_to_worker),
    ]
