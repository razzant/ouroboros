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
from typing import Any, Dict, List

from ouroboros.artifacts import artifact_store_path_block_reason, copy_file_to_task_artifacts
from ouroboros.project_facts import filter_out_project_store as _filter_out_project_store  # noqa: F401
from ouroboros.project_facts import project_store_access_block as _project_store_access_block
from ouroboros.protected_artifacts import block_reason_for_path
from ouroboros.secret_masking import mask_secret_bytes
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for  # noqa: F401
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,  # noqa: F401
    decide_tool_access,  # noqa: F401
    active_tool_profile,  # noqa: F401
    normalize_root,  # noqa: F401
    normalize_runtime_data_path,  # noqa: F401
    project_room_lens_dir,
    UserFilesPathBlockedError,
    user_files_path_block_reason,
)
from ouroboros.utils import atomic_write_json, read_text, safe_relpath, utc_now_iso, write_text_atomic  # noqa: F401
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path
from ouroboros.contracts.skill_payload_policy import (
    SKILL_PAYLOAD_ALL_BUCKETS,
    SKILL_PAYLOAD_CONTROL_DIRNAMES,
    SKILL_PAYLOAD_CONTROL_FILENAMES,
    SKILL_OWNER_STATE_FILENAMES,  # noqa: F401
    SkillPayloadPathError,
    SkillPayloadTarget,
    cross_skill_redirect_error,
    decide_payload_short_form,
    is_skill_control_plane_path as _policy_is_skill_control_plane_path,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _policy_is_skill_owner_state_target,  # noqa: F401
    is_skill_create_typo,
    resolve_skill_payload_target,
)
from ouroboros.tools.core_file_tools import (  # noqa: F401
    _ListingFailure,
    _MEMORY_AT_DRIVE_MEMORY,
    _SKILL_OWNER_STATE_FILENAMES,
    _SUBAGENT_SECRET_FILE_NAMES,
    _access_or_block,
    _annotate_reread,
    _coerce_line_window,
    _coerce_start_char,
    _data_list,
    _data_read,
    _direct_resource_binding,
    _filter_subagent_secret_listing,
    _filter_subagent_secret_repo_listing,
    _is_cognitive_data_path,
    _is_skill_owner_state_target,
    _is_subagent_secret_data_path,
    _is_subagent_secret_repo_path,
    _is_subagent_secret_repo_target,
    make_subagent_secret_target_check,
    _list_dir,
    _list_files,
    _list_user_files_dir,
    _local_readonly_resource_block,
    _normalize_data_read_path,
    _profile_roots_hint,
    _read_file,
    _render_line_slice,
    _repo_list,
    _repo_read,
    _root_display_path,
    is_restricted_subagent_profile,
)
from ouroboros.tools.core_artifacts import (  # noqa: F401
    _MAX_DOCUMENT_FILE_BYTES,
    _MAX_PHOTO_FILE_BYTES,
    _MAX_VIDEO_FILE_BYTES,
    _detect_document_mime,
    _detect_image_mime,
    _detect_video_mime,
    _send_file,
    _send_photo,
    _send_video,
    LinkActionsValidationError,
    QuizValidationError,
    _MAX_LINK_ACTIONS,
    _MAX_QUIZ_OPTIONS,
    _escalate,
    _send_links,
    validate_link_actions,
    validate_quiz_payload,
)

log = logging.getLogger(__name__)


# Payload-local provenance sidecars are launcher/marketplace-owned, not
# skill-author-editable. Generic write/delete/upload paths must block them.
_SELF_AUTHORED_MARKER = ".self_authored.json"


def _binding_skill_control_plane_path(binding: ResolvedResourceBinding) -> bool:
    """Apply the existing payload-sidecar vocabulary to any physical source."""

    try:
        relative = binding.target_path.relative_to(binding.base_path)
    except ValueError:
        return False
    lowered = tuple(part.lower() for part in relative.parts)
    return bool(
        (lowered and lowered[-1] in SKILL_PAYLOAD_CONTROL_FILENAMES)
        or any(part in SKILL_PAYLOAD_CONTROL_DIRNAMES for part in lowered)
    )


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


def _native_payload_mutation_block_reason(
    target: pathlib.Path, data_root: pathlib.Path,
) -> str:
    payload = _skill_payload_parts(target, data_root)
    if payload is None:
        return ""
    bucket, _skill_name, payload_root = payload
    if bucket != "native":
        return ""
    if (payload_root / ".seed-origin").is_file():
        return (
            "launcher-seeded data/skills/native/<skill>/ payloads are "
            "read/review only; edit their seed via root=system_repo"
        )
    if not payload_root.is_dir():
        return (
            "data/skills/native/<skill>/ is not a user-managed payload yet; "
            "create new skills under data/skills/external/<skill>/"
        )
    return ""


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
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    if (
        (_resolved_binding is None or _resolved_binding.root == "runtime_data")
        and (b := _project_store_access_block(_normalize_data_read_path(ctx, path)))
    ):
        return b
    # bucket+skill_name synthesize a payload-confined skill_repair constraint.
    short_form = None if _resolved_binding is not None else decide_payload_short_form(
        bucket=bucket, skill_name=skill_name, path_text=path,
        repo_dir=pathlib.Path(ctx.repo_dir), drive_root=pathlib.Path(ctx.drive_root),
    )
    if short_form is not None and short_form.error:
        return f"⚠️ DATA_WRITE_ERROR: {short_form.error}"
    synth = short_form.constraint if short_form is not None else None
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
    if _resolved_binding is not None:
        p = _resolved_binding.target_path
    elif task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
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
    data_root = (
        _resolved_binding.state_drive_root
        if _resolved_binding is not None
        else pathlib.Path(_cfg.DATA_DIR).resolve(strict=False)
    )
    ctx_data_root = pathlib.Path(ctx.drive_root).resolve(strict=False)
    if _resolved_binding is not None:
        lexical_target = pathlib.Path(p).resolve(strict=False)
    elif task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
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
    native_block = (
        _native_payload_mutation_block_reason(lexical_target, data_root)
        or _native_payload_mutation_block_reason(target_path, data_root)
    )
    if native_block:
        return f"⚠️ DATA_WRITE_BLOCKED: {native_block}."
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
    if (
        (_resolved_binding is not None and _binding_skill_control_plane_path(_resolved_binding))
        or is_skill_control_plane_path(lexical_target, data_root)
        or is_skill_control_plane_path(target_path, data_root)
    ):
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
    if (
        _resolved_binding is not None
        and _resolved_binding.source in {"external", "clawhub", "ouroboroshub"}
        and not _resolved_binding.base_path.exists()
        and (
            _resolved_binding.source != "external"
            or is_skill_create_typo(
                payload_root=_resolved_binding.base_path,
                bucket="external",
                rel_within_payload=_resolved_binding.target_path.relative_to(
                    _resolved_binding.base_path
                ).as_posix(),
            )
        )
    ):
        return (
            f"⚠️ DATA_WRITE_ERROR: skill payload not found: "
            f"skills/{_resolved_binding.source}/{_resolved_binding.skill_name}. Use an existing skill; for a "
            "NEW skill write its manifest (SKILL.md/skill.json) at the payload root under "
            "bucket=external; this path looks like a typo into a missing payload."
        )
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

    # X3 hash-bind: the ADMITTED repair task's payload writes CAS-check the
    # repair's own hash chain (covers overwrite AND append — an append drifts
    # the payload exactly like an overwrite). Foreign lanes pass through.
    _repair_cas_constraint = (
        task_constraint
        if task_constraint and task_constraint.mode == "skill_repair"
        and str(getattr(task_constraint, "skill_name", "") or "")
        else None
    )
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import repair_write_cas_error

        _cas = repair_write_cas_error(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            # The TASK's own constraint decides whether the binding is mandatory:
            # a short-form `bucket`+`skill_name` selector synthesizes the same
            # constraint shape for an ordinary payload edit, and that lane must
            # not need a repair admission.
            repair_task=bool(existing_tc and existing_tc.mode == "skill_repair"))
        if _cas:
            return _cas
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
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import advance_repair_expected_hash

        advance_repair_expected_hash(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""))
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
        state_marker_root = (
            _resolved_binding.state_drive_root
            if _resolved_binding is not None
            else pathlib.Path(ctx.drive_root)
        )
        state_marker = state_marker_root / "state" / "skills" / marker_payload[1] / "self_authored.json"
        state_marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_marker, marker_payload_data, trailing_newline=True)
    result = f"OK: wrote {mode} {_root_display_path(display_root, write_path)} ({len(content)} chars)"
    if _resolved_binding is not None:
        result += (
            f" (resolved_root={_resolved_binding.base_path}; "
            f"source={_resolved_binding.source})"
        )
    if short_form is not None and short_form.ignored_reason:
        result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
    return result


def _join_write_results(results: List[str]) -> str:
    rendered = "\n".join(results) if results else "⚠️ TOOL_ARG_ERROR: files must contain {path, content} objects."
    if any(str(line).lstrip().startswith("⚠️") for result in results for line in str(result).splitlines()):
        return "⚠️ WRITE_FILE_BATCH_PARTIAL_FAILURE: one or more writes failed.\n" + rendered
    return rendered


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
    _resolved_binding: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None = None,
) -> str:
    normalized, block = _access_or_block(ctx, root, "write")
    if block:
        return block
    try:
        if _resolved_binding is None and files:
            bindings: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] = tuple(
                _direct_resource_binding(
                    ctx, None, root=normalized, operation="write",
                    path=str(item.get("path") or ""), bucket=bucket, skill_name=skill_name,
                )
                for item in files if isinstance(item, dict)
            )
        else:
            bindings = _direct_resource_binding(
                ctx, _resolved_binding, root=normalized, operation="write", path=path,
                bucket=bucket, skill_name=skill_name,
            ) if _resolved_binding is None else _resolved_binding
    except Exception as exc:
        prefix = "SKILL_PAYLOAD_ARG_ERROR" if normalized == "skill_payload" else "WRITE_FILE_ERROR"
        return f"⚠️ {prefix}: {exc}"
    binding_items = bindings if isinstance(bindings, tuple) else (bindings,)
    protected_block = next((
        f"⚠️ WRITE_FILE_BLOCKED: protected artifact path blocked: {reason}"
        for item in binding_items
        if (reason := block_reason_for_path(ctx, item.target_path, "write", item))
    ), "")
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

        return _repo_write(
            ctx, path=path, content=content, files=files or [], mode=mode, force=force,
            display_root=normalized, _resolved_binding=bindings,
        )
    if normalized == "runtime_data":
        if files:
            results = []
            binding_iter = iter(binding_items)
            for item in files:
                if not isinstance(item, dict):
                    continue
                item_binding = next(binding_iter, None)
                if item_binding is None:
                    results.append("⚠️ TOOL_ARG_ERROR: files must contain {path, content} objects.")
                    continue
                results.append(_data_write(
                    ctx,
                    str(item.get("path") or ""),
                    str(item.get("content") or ""),
                    mode=mode,
                    display_root=normalized,
                    force=force,
                    _resolved_binding=item_binding,
                ))
            return _join_write_results(results)
        return _data_write(
            ctx, path=path, content=content, mode=mode, display_root=normalized,
            force=force, _resolved_binding=binding_items[0],
        )
    if normalized == "skill_payload":
        if files:
            results = []
            binding_iter = iter(binding_items)
            for item in files:
                rel = str(item.get("path") or "") if isinstance(item, dict) else ""
                body = str(item.get("content") or "") if isinstance(item, dict) else ""
                item_binding = next(binding_iter, None) if isinstance(item, dict) else None
                if item_binding is None:
                    results.append("⚠️ TOOL_ARG_ERROR: files must contain {path, content} objects.")
                    continue
                results.append(_data_write(
                    ctx,
                    rel,
                    body,
                    mode=mode,
                    bucket=bucket,
                    skill_name=skill_name,
                    display_root=normalized,
                    force=force,
                    _resolved_binding=item_binding,
                ))
            return _join_write_results(results)
        return _data_write(
            ctx, path=path, content=content, mode=mode, bucket=bucket,
            skill_name=skill_name, display_root=normalized, force=force,
            _resolved_binding=binding_items[0],
        )
    try:
        if files:
            results = []
            binding_iter = iter(binding_items)
            for item in files:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "")
                item_binding = next(binding_iter, None)
                if item_binding is None:
                    results.append("⚠️ TOOL_ARG_ERROR: files must contain {path, content} objects.")
                    continue
                target = item_binding.target_path
                if normalized == "artifact_store":
                    block_reason = artifact_store_path_block_reason(
                        target, base_path=item_binding.base_path,
                    )
                    if block_reason:
                        results.append(f"⚠️ WRITE_FILE_BLOCKED: artifact_store path blocked: {block_reason}")
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                body = str(item.get("content") or "")
                if mode == "append":
                    # Batch items honor the declared mode like the single-file path below:
                    # silently overwriting here destroyed every prior chunk of a chunked
                    # large-file write while reporting success (#447 D2).
                    with target.open("a", encoding="utf-8") as fh:
                        fh.write(body)  # append is intentionally NOT atomized
                else:
                    # Deferral 5: batch items overwrite too — shrink-guard each (parity with the
                    # single-file path), force=true bypasses.
                    if (shrink := _check_data_shrink_guard(target, body, force)):
                        results.append(shrink)
                        continue
                    write_text_atomic(target, body)  # crash-safe (G)
                result = f"OK: wrote {_root_display_path(normalized, rel_path)} ({len(body)} chars)"
                if normalized == "user_files":
                    record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
                    if record:
                        result += f"\nARTIFACT_OUTPUTS: registered user file -> artifact_store:{record.get('name')}"
                results.append(result)
            return _join_write_results(results)
        target = binding_items[0].target_path
        if normalized == "artifact_store":
            block_reason = artifact_store_path_block_reason(
                target, base_path=binding_items[0].base_path,
            )
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
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    normalized, block = _access_or_block(ctx, root, "edit")
    if block:
        return block
    try:
        binding = _direct_resource_binding(
            ctx, _resolved_binding, root=normalized, operation="edit", path=path,
            bucket=bucket, skill_name=skill_name,
        )
    except Exception as exc:
        prefix = "SKILL_PAYLOAD_ARG_ERROR" if normalized == "skill_payload" else "EDIT_TEXT_ERROR"
        return f"⚠️ {prefix}: {exc}"
    reason = block_reason_for_path(ctx, binding.target_path, "write", binding)
    protected_block = (
        f"⚠️ EDIT_TEXT_BLOCKED: protected artifact path blocked: {reason}" if reason else ""
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
    bound_skill_payload = bool(
        binding.skill_name
        and binding.source in {"external", "clawhub", "ouroboroshub", "native", "user_repo"}
    )
    if native_block := _native_payload_mutation_block_reason(
        binding.target_path, binding.state_drive_root,
    ):
        return f"⚠️ EDIT_TEXT_BLOCKED: {native_block}."
    if normalized in {"active_workspace", "system_repo"} and not bound_skill_payload:
        from ouroboros.tools.git import _str_replace_editor

        result = _str_replace_editor(
            ctx, path=path, old_str=old_str, new_str=new_str,
            display_root=normalized, force=force, _resolved_binding=binding,
        )
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
    if normalized == "skill_payload" or bound_skill_payload:
        try:
            target = binding.target_path
            if (
                _binding_skill_control_plane_path(binding)
                or is_skill_control_plane_path(target, binding.state_drive_root)
            ):
                return (
                    "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                    "marketplace, dependency, and self-authored markers are "
                    "control-plane state. Edit user-authored payload files instead."
                )
            text = target.read_text(encoding="utf-8")
            new_text, match_error = _str_match_replace(
                text, old_str, new_str, _root_display_path(normalized, path), "EDIT_TEXT_ERROR",
            )
            if match_error:
                return match_error
            if (shrink := _check_data_shrink_guard(target, new_text, force)):
                return shrink
            write_text_atomic(target, new_text)
            replacement_line = new_text[:new_text.index(new_str)].count("\n") + 1
            context_start = max(0, replacement_line - 3)
            context_lines = new_text.splitlines()[
                context_start:replacement_line + len(new_str.splitlines()) + 2
            ]
            context_preview = "\n".join(
                f"{context_start + index + 1:>4}| {line}"
                for index, line in enumerate(context_lines)
            )
            return (
                f"✅ Replaced in {_root_display_path(normalized, path)} "
                f"(line {replacement_line}; resolved_root={binding.base_path}; "
                f"source={binding.source}).\nContext:\n{context_preview}\n\n"
                "File is on disk but NOT committed.\n"
                "Run skill_review for this skill before enabling or declaring it ready."
            )
        except FileNotFoundError:
            return f"⚠️ EDIT_TEXT_ERROR: file not found: {_root_display_path(normalized, path)}"
        except Exception as exc:
            return f"⚠️ EDIT_TEXT_ERROR: {type(exc).__name__}: {exc}"
    try:
        target = binding.target_path
        if normalized == "runtime_data":
            if is_skill_control_plane_path(target, binding.state_drive_root):
                return (
                    "⚠️ EDIT_TEXT_BLOCKED: skill provenance, launcher seed, "
                    "marketplace, dependency, and self-authored markers are "
                    "control-plane state. Edit user-authored payload files instead."
                )
            if (b := _project_store_access_block(_normalize_data_read_path(ctx, path))):
                return b
            if (
                _is_workspace_executor_control_state_path(target, binding.base_path)
                or _is_workspace_executor_control_state_path(target, binding.state_drive_root)
            ):
                return (
                    "⚠️ EDIT_TEXT_BLOCKED: workspace executor process records are "
                    "owner/runtime control-plane state. Use process/service lifecycle "
                    "tools instead of editing state/workspace_executor_processes directly."
                )
        if normalized == "artifact_store":
            block_reason = artifact_store_path_block_reason(
                target, base_path=binding.base_path,
            )
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
        result = (
            f"OK: edited {_root_display_path(normalized, path)} "
            f"(resolved_root={binding.base_path}; source={binding.source})"
        )
        if normalized == "user_files":
            record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
            if record:
                result += f"\nARTIFACT_OUTPUTS: registered user file -> artifact_store:{record.get('name')}"
        return result
    except FileNotFoundError:
        return f"⚠️ EDIT_TEXT_ERROR: file not found: {_root_display_path(normalized, path)}"
    except Exception as exc:
        return f"⚠️ EDIT_TEXT_ERROR: {type(exc).__name__}: {exc}"

_MAX_SEARCH_RESULTS = 200
# Search file-skip helper and caps live in ouroboros.code_search_rg (the search
# module SSOT); imported with the historical private names used by call sites.
from ouroboros.code_search_rg import (  # noqa: E402
    MAX_SEARCH_FILES_SCANNED as _MAX_SEARCH_FILES_SCANNED,
    _search_wall_clock_sec,
    is_search_skippable as _is_search_skippable,  # noqa: F401 — re-exported for tests/call sites
    search_skip_reason as _search_skip_reason,
)


def _code_search(ctx: ToolContext, query: str, path: str = ".",
                 regex: bool = False, max_results: int = 200,
                 include: str = "", root: str = "active_workspace",
                 bucket: str = "", skill_name: str = "",
                 _resolved_binding: ResolvedResourceBinding | None = None) -> str:
    """Search repo text with optional regex, path, glob, and result cap."""
    if not query:
        return "⚠️ SEARCH_ERROR: query is required."
    normalized, block = _access_or_block(ctx, root, "search")
    if block:
        return block
    try:
        binding = _direct_resource_binding(
            ctx, _resolved_binding, root=normalized, operation="search", path=path,
            bucket=bucket, skill_name=skill_name,
        )
    except UserFilesPathBlockedError as exc:
        return f"⚠️ USER_FILES_PATH_BLOCKED: {exc}"
    except Exception as exc:
        return f"⚠️ SEARCH_ERROR: {type(exc).__name__}: {exc}"
    if normalized == "runtime_data" and (b := _project_store_access_block(_normalize_data_read_path(ctx, path))):
        return b

    max_results = min(max(1, max_results), _MAX_SEARCH_RESULTS)
    root_path = binding.base_path
    display_path = binding.target_path.relative_to(root_path).as_posix() if normalized in {"active_workspace", "system_repo"} else path
    display_search_path = _root_display_path(normalized, display_path)
    search_root = binding.target_path
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
    protected_root_block = block_reason_for_path(
        ctx, search_root, "static_introspection", binding
    )
    if protected_root_block:
        return protected_root_block
    protected_root_read_block = block_reason_for_path(
        ctx, search_root, "read_bytes", binding
    )
    if protected_root_read_block and search_root.is_file():
        return protected_root_read_block
    subagent_readonly = is_restricted_subagent_profile(ctx)
    secret_check = None
    if subagent_readonly:
        secret_repo = root_path if normalized in {"active_workspace", "system_repo"} else active_repo_dir_for(ctx)
        secret_check = make_subagent_secret_target_check(secret_repo, ctx=ctx)
        block_msg = _local_readonly_resource_block(ctx, normalized, search_root, root_path, action="SEARCH", secret_check=secret_check)
        if block_msg:
            return block_msg
    root_resolved = root_path.resolve(strict=False)
    _rt_search_root = str(root_resolved) if normalized == "runtime_data" else ""
    # Filter receipt (D3, capinv-447): every file present under the search root
    # but excluded before reading is COUNTED by typed reason, so "No matches"
    # can honestly distinguish a clean empty result from a filtered one.
    search_drops: dict[str, int] = {}

    def _drop(reason: str) -> bool:
        search_drops[reason] = search_drops.get(reason, 0) + 1
        return False

    def _path_allowed_for_rg(fp: pathlib.Path) -> bool:
        # Resolve, then CONFINE to the resource root: a path whose resolved target
        # escapes it (e.g. an in-root symlink to outside) is rejected — no leak.
        try:
            fp = pathlib.Path(fp).resolve(strict=False)
            rel_parts = fp.relative_to(root_resolved).parts
        except Exception:
            return _drop("escapes_root")
        # runtime_data per-project store is reachable only via scoped knowledge tools.
        if normalized == "runtime_data" and rel_parts and str(rel_parts[0]).casefold() == "projects":
            return _drop("project_store_scoped")
        if subagent_readonly and _local_readonly_resource_block(ctx, normalized, fp, root_path, action="SEARCH", secret_check=secret_check):
            return _drop("restricted_subagent")
        if normalized == "user_files" and user_files_path_block_reason(ctx, fp, operation="search"):
            return _drop("user_files_policy")
        if block_reason_for_path(ctx, fp, "read_bytes", binding):
            return _drop("protected_artifact")
        skip = _search_skip_reason(fp)
        if skip:
            return _drop(skip)
        return True

    def _mask_user_files_matches(result_text: str) -> str:
        # Same egress seam as _read_file (#447 В23): a search over the owner's
        # home surfaces file CONTENT in the match lines, so raw credential
        # bytes must be masked here too — on BOTH the rg path and the Python
        # fallback. Names/paths stay; values become ***.
        if normalized != "user_files" and not subagent_readonly:
            return result_text
        masked_text, masked = mask_secret_bytes(
            result_text, mask_opaque=normalized not in {"active_workspace", "system_repo"},
        )
        if masked:
            masked_text += (
                f"\n⚠️ SECRET_BYTES_MASKED: {masked} secret-shaped span(s) in these "
                "matches were replaced with ***; raw credentials never enter model "
                "context. Reference them by location, not value."
            )
        return masked_text

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
            return _mask_user_files_matches(format_search_result(
                display_path=display_search_path, root_name=normalized,
                root_path=root_path, query=query, regex=bool(regex),
                max_results=max_results, result=rg_result, dropped=search_drops,
            ))
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
    search_drops.clear()  # the fallback re-walks the tree; drop any partial rg counts
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

        # SKIP_DIRS pruning is a documented walk convention (parity with rg's
        # skip of .git/node_modules); POLICY prunes below are counted — a whole
        # subtree removed by policy must not read as "0 files searched" with no
        # receipt (#447 D3).
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIRS]
        if normalized == "runtime_data" and str(pathlib.Path(dirpath).resolve(strict=False)) == _rt_search_root:
            for d in dirnames:
                if d.casefold() == "projects":
                    _drop("project_store_scoped")
            dirnames[:] = [d for d in dirnames if d.casefold() != "projects"]
        if normalized == "user_files":
            kept_dirs = []
            for d in dirnames:
                if user_files_path_block_reason(ctx, pathlib.Path(dirpath) / d, operation="search"):
                    _drop("user_files_policy")
                else:
                    kept_dirs.append(d)
            dirnames[:] = kept_dirs
        if subagent_readonly:
            kept_dirs = []
            for d in dirnames:
                if _local_readonly_resource_block(ctx, normalized, pathlib.Path(dirpath) / d, root_path, action="SEARCH", secret_check=secret_check):
                    _drop("restricted_subagent")
                else:
                    kept_dirs.append(d)
            dirnames[:] = kept_dirs

        for fname in sorted(filenames):
            fp = pathlib.Path(dirpath) / fname

            if include and not fnmatch.fnmatch(fname, include):
                continue

            if subagent_readonly and _local_readonly_resource_block(ctx, normalized, fp, root_path, action="SEARCH"):
                _drop("restricted_subagent")
                continue
            if normalized == "user_files" and user_files_path_block_reason(ctx, fp, operation="search"):
                _drop("user_files_policy")
                continue
            if block_reason_for_path(ctx, fp, "read_bytes", binding):
                _drop("protected_artifact")
                continue

            if _skip := _search_skip_reason(fp):
                _drop(_skip)
                continue

            # CONFINE to the root before reading (parity with the rg path's _path_allowed_for_rg
            # and _list_dir): a resolved file escaping the root — e.g. an in-tree symlink in an
            # untrusted child project/deliverable tree — must never have its target read out.
            try:
                fp.resolve(strict=False).relative_to(root_resolved)
            except (OSError, ValueError):
                _drop("escapes_root")
                continue

            if files_searched >= _MAX_SEARCH_FILES_SCANNED:
                files_capped = True
                break

            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                _drop("read_error")
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
    # Typed disclosure, not invisibility (capinv-447): a restricted subagent is
    # TOLD how many secret/control files its search could not see, mirroring the
    # listing filters' hidden-entries marker.
    protected_omitted = search_drops.get("protected_artifact", 0)
    restricted_omitted = search_drops.get("restricted_subagent", 0)
    restricted_note = (
        f" {restricted_omitted} secret/control file(s) omitted from this subagent's search."
        if restricted_omitted else ""
    )
    # Filter receipt for the remaining ordinary exclusions (D3): oversized,
    # symlinks, excluded names, unreadable, policy-scoped — never silent.
    from ouroboros.code_search_rg import format_dropped_files_note

    other_dropped_note = format_dropped_files_note({
        key: count for key, count in search_drops.items()
        if key not in ("protected_artifact", "restricted_subagent")
    })
    if not matches:
        suffix = f" {protected_omitted} protected artifact file(s) omitted." if protected_omitted else ""
        cap_note = f" Scan stopped after {_MAX_SEARCH_FILES_SCANNED} files — narrow the path or glob." if files_capped else ""
        return f"No matches found for {'regex' if regex else 'literal'} `{query}` in {display_search_path} ({files_searched} files searched).{suffix}{restricted_note}{other_dropped_note}{cap_note}{deadline_note}"

    header = f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} in {display_search_path} ({files_searched} files searched)"
    if files_capped:
        header += f" — scan stopped at {_MAX_SEARCH_FILES_SCANNED} files (narrow the path or glob)"
    if truncated:
        header += f" — truncated at {max_results} results"
    if deadline_hit:
        header += " — stopped at the time budget (results may be incomplete)"
    if protected_omitted:
        header += f" — {protected_omitted} protected artifact file(s) omitted"
    if restricted_omitted:
        header += f" — {restricted_omitted} secret/control file(s) omitted from this subagent's search"
    if other_dropped_note:
        header += " —" + other_dropped_note.rstrip(".")
    return _mask_user_files_matches(header + "\n\n" + "\n".join(matches))


def _durable_descendant_of(
    drive_root: pathlib.Path,
    task_id: str,
    task: Dict[str, Any],
    ancestor_id: str,
    *,
    max_hops: int = 64,
) -> bool:
    """Follow the durable parent chain; shared-root labels are not ancestry proof."""

    from ouroboros.task_status import load_effective_task_result

    current_id = str(task_id or "")
    current = task if isinstance(task, dict) else {}
    seen = {current_id}
    for _hop in range(max_hops):
        parent_id = str(current.get("parent_task_id") or "").strip()
        if not parent_id:
            return False
        if parent_id == ancestor_id:
            return True
        if parent_id in seen:
            return False
        seen.add(parent_id)
        current = load_effective_task_result(drive_root, parent_id)
        if not current:
            return False
        current_id = parent_id
    return False


def _forward_to_worker(
    ctx: ToolContext, task_id: str, message: str, relayed_from_task_id: str = "",
) -> str:
    """Forward a message to a running worker task's mailbox."""
    from ouroboros.owner_mailbox import write_task_message
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
    # AR2-6: no NEW steering writes while a cancellation is pending. The
    # effective status honestly stays ``running`` (cancel_state=pending rides
    # beside it), so the checks above pass — consult the same predicate the
    # steer_task/mailbox routes use, covering BOTH carriers (durable intent +
    # legacy latch), and refuse typed.
    try:
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(status_drive_root, tid):
            return (
                f"⚠️ TASK_CANCEL_PENDING: task {tid} has a pending cancellation — the "
                "supervisor is tearing it down; the message was NOT delivered. Wait for "
                "the settled outcome or start a new task."
            )
    except Exception:
        log.debug("forward_to_worker cancel-pending check failed for %s", tid, exc_info=True)
    current_task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not current_task_id:
        return "⚠️ TASK_FORBIDDEN: forward_to_worker requires an active task context."
    if not _durable_descendant_of(status_drive_root, tid, data, current_task_id):
        return f"⚠️ TASK_FORBIDDEN: task {tid} is not a child or descendant of the current task."
    relayed_from = str(relayed_from_task_id or "").strip()
    provenance = "ancestor_task"
    if relayed_from:
        try:
            relayed_from = validate_task_id(relayed_from)
        except ValueError as exc:
            return f"⚠️ TOOL_ARG_ERROR (forward_to_worker): {exc}"
        source = load_effective_task_result(status_drive_root, relayed_from)
        if not source or not _durable_descendant_of(
            status_drive_root, relayed_from, source, current_task_id,
        ):
            return (
                f"⚠️ TASK_FORBIDDEN: relay source {relayed_from} is not a child or "
                "descendant of the current task."
            )
        provenance = "peer_via_ancestor"
    child_drive = str(data.get("child_drive_root") or data.get("headless_child_drive_root") or data.get("drive_root") or "").strip()
    mailbox_drive = pathlib.Path(child_drive) if child_drive else pathlib.Path(ctx.drive_root)
    written = write_task_message(
        mailbox_drive,
        message,
        task_id=tid,
        source_task_id=current_task_id,
        provenance=provenance,
        relayed_from_task_id=relayed_from,
        msg_id=uuid.uuid4().hex,
    )
    if not written:
        return f"⚠️ TASK_MESSAGE_UNWRITTEN: message to task {tid} was not persisted."
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
                "start_char": {"type": "integer", "default": 0,
                               "description": "Sub-line cursor: skip this many characters of the selected "
                                              "window before rendering. Use it to advance WITHIN a single "
                                              "line longer than the delivery limit (the result would "
                                              "otherwise be cut at the tool-result budget)."},
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
                "Overwriting an existing repo file returns the unified diff vs the "
                "previous version — CHECK IT; invalid .py/.json content is blocked "
                "before writing (force=true bypasses). "
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
                "For several edits at once, repeated identical replacements, or "
                "counted replace-all, prefer edit_batch (one atomic call). "
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
        ToolEntry("send_file", {
            "name": "send_file",
            "description": (
                "Send an arbitrary document/file to the owner's chat (e.g. a report, .md/.csv/.html, "
                "PDF, archive, or code file — anything that is not an image or video). "
                "Requires a local file_path; max 50 MB. Use this to deliver a finished file result "
                "the owner can download, rather than only describing it or sending a screenshot."
            ),
            "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string", "description": "Local file path to the document/file"},
                "caption": {"type": "string", "description": "Optional caption for the file"},
            }, "required": ["file_path"]},
        }, _send_file),
        ToolEntry("send_links", {
            "name": "send_links",
            "description": "Send one or more HTTP(S) links as prominent clickable buttons in the owner's chat.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Optional heading above the buttons"},
                "links": {"type": "array", "minItems": 1, "maxItems": _MAX_LINK_ACTIONS, "items": {
                    "type": "object", "properties": {
                        "label": {"type": "string"},
                        "url": {"type": "string"},
                    }, "required": ["label", "url"]}},
            }, "required": ["links"]},
        }, _send_links),
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
        ToolEntry("escalate", {
            "name": "escalate",
            "description": (
                "Escalate a decision up the responsibility chain instead of guessing. "
                "A root task asks the OWNER (a typed quiz card with option buttons); "
                "a subagent asks its PARENT task (a typed mailbox frame the parent "
                "answers with forward_to_worker or escalates higher, verbatim). "
                "Fire-and-continue: state the assumption you keep working under — "
                "the answer, if it comes, arrives in a later round, and the question "
                "expires when your task ends."
            ),
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "The decision being escalated (markdown renders in chat)"},
                "options": {"type": "array", "items": {"type": "object", "properties": {
                    "label": {"type": "string", "description": "Short option label (button text, max 120)"},
                    "detail": {"type": "string", "description": "Optional one-line consequence of this option (max 500)"},
                }, "required": ["label"]}, "description": "2-6 mutually exclusive options"},
                "stake": {"type": "string", "description": "What depends on this decision (optional, max 500)"},
                "assumption": {"type": "string", "description": "REQUIRED: the assumption you continue under until answered (max 500)"},
            }, "required": ["question", "options", "assumption"]},
        }, _escalate),
        ToolEntry("forward_to_worker", {
            "name": "forward_to_worker",
            "description": (
                "Send an addressed task-tree message to a running child or descendant. "
                "The mailbox preserves you as the ancestor sender; it never labels this "
                "message as owner dialogue. Arbitrary unrelated tasks remain unreachable."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "ID of the running task to forward to"},
                "message": {"type": "string", "description": "Message text to forward"},
                "relayed_from_task_id": {"type": "string", "description":
                    "Optional sibling/descendant task whose output this ancestor relays. "
                    "The recipient sees both peer and ancestor provenance."},
            }, "required": ["task_id", "message"]},
        }, _forward_to_worker),
    ]
