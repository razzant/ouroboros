"""Task-scoped file read authorization, receipt visibility, and idempotency fingerprints for Presence tools."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
from typing import Any, Dict, List, Tuple


def _safe_lineage_ids(ctx: Any) -> set[str]:
    ids: set[str] = set()
    tid = str(getattr(ctx, "task_id", "") or "").strip()
    if tid:
        ids.add(tid)
    meta = getattr(ctx, "task_metadata", None)
    if isinstance(meta, dict):
        for key in ("parent_task_id", "root_task_id"):
            val = str(meta.get(key) or "").strip()
            if val:
                ids.add(val)
    try:
        from ouroboros.tool_access import lineage_task_ids

        ids.update(str(x) for x in lineage_task_ids(ctx) if x)
    except Exception:
        pass
    return ids


def _path_under_exact(fp: pathlib.Path, root: pathlib.Path) -> bool:
    """Physical containment for positive admission (never case-folds on case-sensitive filesystems)."""
    try:
        from ouroboros.tool_access import path_is_relative_to

        return bool(path_is_relative_to(fp, root))
    except Exception:
        try:
            fp.relative_to(root)
            return True
        except ValueError:
            return False


def _path_under_deny(fp: pathlib.Path, root: pathlib.Path) -> bool:
    """Conservative containment for deny-lists (blocks both exact and case-folded spellings)."""
    try:
        from ouroboros.tool_access import _path_is_relative_to_casefold, path_is_relative_to

        return bool(path_is_relative_to(fp, root) or _path_is_relative_to_casefold(fp, root))
    except Exception:
        return _path_under_exact(fp, root)


_UPLOADABLE_PROFILE_ROOTS = frozenset({
    "active_workspace",
    "task_drive",
    "artifact_store",
    "deliverables",
    "subagent_projects",
    "user_files",
})


def _allowed_upload_entries(ctx: Any, *, include_user_files: bool = True) -> List[tuple[str, pathlib.Path, pathlib.Path]]:
    """Return ``(logical_root, physical_allowed_root, ceiling_base_root)`` tuples."""
    entries: List[tuple[str, pathlib.Path, pathlib.Path]] = []
    try:
        from ouroboros.tool_access import active_tool_profile, canonical_data_root, decide_tool_access

        if decide_tool_access(profile=active_tool_profile(ctx), root="runtime_data", operation="read").allow:
            base = canonical_data_root(ctx)
            if base is not None:
                canon = pathlib.Path(base).resolve(strict=False)
                entries.append(("runtime_data", (canon / "uploads").resolve(strict=False), canon))
    except Exception:
        pass
    try:
        from ouroboros.tool_access import active_tool_profile, decide_tool_access
        from ouroboros.tools.registry import active_repo_dir_for

        if decide_tool_access(profile=active_tool_profile(ctx), root="active_workspace", operation="read").allow:
            ws = pathlib.Path(active_repo_dir_for(ctx)).expanduser().resolve(strict=False)
            entries.append(("active_workspace", ws, ws))
    except Exception:
        pass
    try:
        from ouroboros.tool_access import lineage_read_roots, profile_readable_root_paths

        for label, path in profile_readable_root_paths(ctx):
            if label not in _UPLOADABLE_PROFILE_ROOTS:
                continue
            if include_user_files or label != "user_files":
                resolved = pathlib.Path(path).expanduser().resolve(strict=False)
                entries.append((label, resolved, resolved))
        for lr_name in ("task_drive", "artifact_store"):
            for base_lr in lineage_read_roots(ctx, lr_name):
                resolved = pathlib.Path(base_lr).expanduser().resolve(strict=False)
                entries.append((lr_name, resolved, resolved))
    except Exception:
        for name in ("artifact_store", "task_drive"):
            try:
                from ouroboros.tool_access import resource_root_path

                resolved = pathlib.Path(resource_root_path(ctx, name)).expanduser().resolve(strict=False)
                entries.append((name, resolved, resolved))
            except Exception:
                pass
    return list(dict.fromkeys(entries))


def _resolve_candidate(ctx: Any, raw: str) -> pathlib.Path:
    if raw.startswith("~"):
        try:
            from ouroboros.tool_access import resource_root_path, safe_relpath

            home = pathlib.Path(resource_root_path(ctx, "user_files")).resolve(strict=False)
            after = raw[1:]
            if after[:1] in ("/", "\\"):
                rel = after[1:]
            elif "/" in after or "\\" in after:
                rel = after.replace("\\", "/").split("/", 1)[1]
            else:
                rel = ""
            return (home / safe_relpath(rel)).resolve() if rel else home.resolve()
        except Exception as exc:
            raise PermissionError(f"file_path cannot be resolved under user_files: {exc}") from exc
    fp = pathlib.Path(raw)
    if not fp.is_absolute():
        try:
            from ouroboros.tools.registry import active_repo_dir_for

            fp = pathlib.Path(active_repo_dir_for(ctx)).expanduser() / fp
        except Exception as exc:
            raise PermissionError(
                f"Relative file_path {raw!r} has no active workspace to resolve against. Name an absolute path."
            ) from exc
    return fp.resolve()


def _open_confined_regular_fd(root: pathlib.Path, target: pathlib.Path) -> tuple[int, int]:
    """Open ``target`` under ``root`` without following symlinks/reparse points (POSIX + Windows)."""
    root_real = pathlib.Path(root).resolve(strict=True)
    rel = pathlib.Path(target).relative_to(root_real)
    parts = rel.parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise PermissionError("file_path must name a regular file inside an authorized root")
    if os.name != "nt" and os.open in getattr(os, "supports_dir_fd", ()):
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        step_flags = dir_flags | getattr(os, "O_NOFOLLOW", 0)
        leaf_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        cur_fd = os.open(root_real, dir_flags)
        try:
            for segment in parts[:-1]:
                nxt = os.open(segment, step_flags, dir_fd=cur_fd)
                os.close(cur_fd)
                cur_fd = nxt
            leaf_fd = os.open(parts[-1], leaf_flags, dir_fd=cur_fd)
        finally:
            os.close(cur_fd)
        try:
            st = os.fstat(leaf_fd)
            if not stat.S_ISREG(st.st_mode):
                raise PermissionError("file_path must name a regular file")
            return leaf_fd, int(st.st_size)
        except BaseException:
            os.close(leaf_fd)
            raise
    from ouroboros.confined_files import open_confined_file

    handle, st = open_confined_file(root_real, parts)
    try:
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError("file_path must name a regular file")
        return os.dup(handle.fileno()), int(st.st_size)
    finally:
        handle.close()


def _authorize_with_root(ctx: Any, file_path: Any, state_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Verify the calling task may read ``file_path`` and return ``(authorized_root, resolved_target)``."""
    from types import SimpleNamespace

    raw = str(file_path or "").strip()
    if not raw:
        raise PermissionError("file_path is required")
    if ctx is None:
        raise PermissionError("file upload requires an authorized task context; nothing was read or queued")
    fp = _resolve_candidate(ctx, raw)
    # Never allow exfiltrating this skill's own secrets, SQLite custody, or frozen copies.
    state_resolved = pathlib.Path(state_dir).resolve(strict=False)
    staging_root = (state_resolved / "staging").resolve(strict=False)
    if _path_under_deny(fp, state_resolved):
        if not _path_under_exact(fp, staging_root):
            raise PermissionError("file_path cannot read Telegram skill private state")
        # Even inside staging/, only a Presence turn may read attachments staged for its own source event.
        meta = getattr(ctx, "task_metadata", None) or {}
        event = (meta.get("presence") or {}).get("event") or {} if isinstance(meta, dict) else {}
        event_id = str(event.get("source_event_id") or "").strip() if isinstance(event, dict) else ""
        if not event_id:
            raise PermissionError("file_path cannot read Telegram skill staging without a matching source event")
        import re

        safe_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id)[:160] or "event"
        exact_dir = (staging_root / safe_dir).resolve(strict=False)
        if _path_under_exact(fp, exact_dir):
            return exact_dir, fp
        rel_staged = fp.relative_to(staging_root)
        if rel_staged.parts and rel_staged.parts[0].startswith(safe_dir + "--"):
            attempt_dir = (staging_root / rel_staged.parts[0]).resolve(strict=False)
            if _path_under_exact(fp, attempt_dir):
                return attempt_dir, fp
        raise PermissionError("file_path cannot read another event's staged attachment")

    # Enforce any frozen Presence capability ceiling on the task contract (fail-closed on tamper/error).
    try:
        from ouroboros.presence_authority import presence_ceiling_allows_binding, presence_ceiling_from_context

        ceiling = presence_ceiling_from_context(ctx)
    except Exception as exc:
        raise PermissionError(f"presence capability ceiling check failed: {exc}") from exc

    entries = _allowed_upload_entries(ctx, include_user_files=True)
    matched = [entry for entry in entries if _path_under_exact(fp, entry[1])]
    if ceiling is not None:
        user_files_root: pathlib.Path | None = None
        try:
            from ouroboros.tool_access import resource_root_path

            user_files_root = pathlib.Path(resource_root_path(ctx, "user_files")).resolve(strict=False)
        except Exception:
            user_files_root = None
        matched = [
            entry
            for entry in matched
            if presence_ceiling_allows_binding(
                ceiling,
                SimpleNamespace(
                    root="user_files" if entry[0] == "deliverables" else entry[0],
                    operation="read",
                    base_path=entry[2],
                    logical_base_path=user_files_root if entry[0] == "deliverables" else None,
                    target_path=fp,
                    source="",
                    skill_name="",
                ),
            )
        ]
    if not matched:
        raise PermissionError(
            "file_path is outside the roots this task may read (workspace, uploads, task_drive, "
            "artifact_store, deliverables, or permitted user_files). Nothing was read or queued."
        )
    # User-files-only admission still enforces secret/credential/runtime-overlap confinement (fail-closed).
    if all(label == "user_files" for label, _, _ in matched):
        try:
            from ouroboros.tool_access import user_files_path_block_reason

            reason = user_files_path_block_reason(ctx, fp, operation="read")
        except Exception as exc:
            raise PermissionError(f"user_files policy check failed: {exc}") from exc
        if reason:
            raise PermissionError(f"user_files path blocked: {reason}")
    # Runtime-data per-path guards (skill owner state, project stores, secrets) — fail-closed.
    try:
        from ouroboros.tools.core_file_tools import _runtime_data_read_block

        block = _runtime_data_read_block(ctx, fp)
    except Exception as exc:
        raise PermissionError(f"runtime_data policy check failed: {exc}") from exc
    if block:
        raise PermissionError(block)
    # Protected-artifact read_bytes / copy guards — fail-closed.
    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        for op in ("read_bytes", "copy"):
            art_block = block_reason_for_path(ctx, fp, op)
            if art_block:
                raise PermissionError(art_block)
    except PermissionError:
        raise
    except Exception as exc:
        raise PermissionError(f"protected_artifacts policy check failed: {exc}") from exc
    return matched[0][1], fp


def authorize_upload_path(ctx: Any, file_path: Any, state_dir: pathlib.Path) -> pathlib.Path:
    """Verify the calling task may read ``file_path`` BEFORE stat/open/copy (fail-closed)."""
    return _authorize_with_root(ctx, file_path, state_dir)[1]


def open_authorized_upload(ctx: Any, file_path: Any, state_dir: pathlib.Path) -> Tuple[int, pathlib.Path, int]:
    """Authorize ``file_path`` and open it pinned to its verified root via O_NOFOLLOW dir_fd traversal."""
    root, fp = _authorize_with_root(ctx, file_path, state_dir)
    fd, size = _open_confined_regular_fd(root, fp)
    return fd, fp, size


def caller_scope(ctx: Any) -> Dict[str, str]:
    """Capture the calling task/focus identity stamped onto queued outbox rows."""
    if ctx is None:
        return {}
    meta = getattr(ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    presence = meta.get("presence") if isinstance(meta.get("presence"), dict) else {}
    event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    scope: Dict[str, str] = {}
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if task_id:
        scope["task_id"] = task_id
    root_id = str(meta.get("root_task_id") or "").strip()
    if root_id:
        scope["root_task_id"] = root_id
    place_id = str(presence.get("place_id") or (meta.get("focus") or {}).get("place_id") or "").strip()
    if place_id:
        scope["place_id"] = place_id
    try:
        from ouroboros.presence_authority import presence_caller_binding

        binding_id = presence_caller_binding(ctx)
        if isinstance(binding_id, str) and binding_id.strip():
            scope["binding_id"] = binding_id.strip()
    except Exception:
        pass
    if event.get("provider") == "telegram":
        for key in ("account_id", "conversation_id", "thread_id", "conversation_key"):
            val = str(event.get(key) or "").strip()
            if val:
                scope[key] = val
    return scope


def receipt_visible_to(ctx: Any, payload: Dict[str, Any]) -> bool:
    """Whether ``ctx`` may read the delivery receipt for ``payload`` (fail-closed for scoped callers)."""
    scope = payload.get("_caller_scope") if isinstance(payload.get("_caller_scope"), dict) else {}
    origin = (payload.get("_reporting") or {}).get("origin") if isinstance(payload.get("_reporting"), dict) else {}
    row_task = str((origin or {}).get("task_id") or scope.get("task_id") or "").strip()
    if ctx is None:
        return not scope and not row_task
    meta = getattr(ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    presence = meta.get("presence") if isinstance(meta.get("presence"), dict) else {}
    focus = meta.get("focus") if isinstance(meta.get("focus"), dict) else {}
    lineage = _safe_lineage_ids(ctx)
    # Presence focus or delegated descendant of a Presence turn: restrict strictly to own conversation tuple,
    # lineage, and positive binding agreement whenever the caller carries a binding id.
    caller_binding = None
    try:
        from ouroboros.presence_authority import presence_caller_binding

        caller_binding = presence_caller_binding(ctx)
    except Exception:
        caller_binding = None
    if presence or focus or caller_binding is not None:
        event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
        focus_place = str(presence.get("place_id") or focus.get("place_id") or "").strip()
        row_place = str(scope.get("place_id") or (payload.get("_place_binding") or {}).get("place_id") or "").strip()
        row_binding = str(
            scope.get("binding_id")
            or (payload.get("_reporting") or {}).get("binding_id")
            or (payload.get("_place_binding") or {}).get("binding_id")
            or ""
        ).strip()
        if focus_place and row_place and focus_place != row_place:
            return False
        if isinstance(caller_binding, str) and caller_binding and row_binding != caller_binding:
            return False
        if event.get("provider") == "telegram":
            ev_acct = str(event.get("account_id") or "").strip()
            ev_chat = str(event.get("conversation_id") or "").strip()
            ev_topic = str(event.get("thread_id") or "").strip()
            row_acct = str(payload.get("account_id") or scope.get("account_id") or "").strip()
            row_chat = str(payload.get("chat_id") or scope.get("conversation_id") or "").strip()
            row_topic = str(payload.get("topic_id") or scope.get("thread_id") or "").strip()
            if not (ev_acct and ev_chat):
                return False
            return row_acct == ev_acct and row_chat == ev_chat and row_topic == ev_topic
        if not (row_task and row_task in lineage):
            return False
        if focus_place and row_place != focus_place:
            return False
        return True
    if row_task and row_task in lineage:
        return True
    # Restricted subagent: only rows from its own task lineage.
    try:
        from ouroboros.tool_access import active_tool_profile
        from ouroboros.tool_access_roots import _is_subagent_ctx

        if _is_subagent_ctx(ctx) or active_tool_profile(ctx) in {"local_readonly_subagent", "acting_subagent"}:
            return False
    except Exception:
        return False
    # Top-level owner / Main task may inspect receipts of its install.
    return True


def sanitize_tool_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Strip host local filesystem paths from tool-facing receipts."""
    cleaned = dict(receipt)
    files = cleaned.get("files")
    if isinstance(files, list):
        cleaned["files"] = [
            {k: v for k, v in item.items() if k not in {"source_path", "copy_path"}}
            for item in files
            if isinstance(item, dict)
        ]
    return cleaned


def idempotency_fingerprint(payload: Dict[str, Any]) -> str:
    """Canonical SHA-256 fingerprint of an outbox operation's caller scope, target, and content."""
    scope = payload.get("_caller_scope") if isinstance(payload.get("_caller_scope"), dict) else {}
    frozen = payload.get("_frozen") if isinstance(payload.get("_frozen"), list) else []
    files = [
        (str(item.get("sha256") or ""), int(item.get("size") or 0), pathlib.PurePath(str(item.get("path") or "")).name)
        for item in frozen
        if isinstance(item, dict)
    ]
    album_caps = [
        (str(item.get("kind") or ""), str(item.get("caption") or ""))
        for item in (payload.get("items") or [])
        if isinstance(item, dict)
    ]
    canonical = {
        "caller": {
            "task_id": str(scope.get("task_id") or ""),
            "place_id": str(scope.get("place_id") or ""),
            "conversation_key": str(scope.get("conversation_key") or ""),
        },
        "account_id": str(payload.get("account_id") or ""),
        "chat_id": str(payload.get("chat_id") or ""),
        "topic_id": str(payload.get("topic_id") or ""),
        "reply_to_message_id": str(payload.get("reply_to_message_id") or ""),
        "original_delivery_id": str(payload.get("original_delivery_id") or ""),
        "kind": str(payload.get("kind") or ""),
        "method": str(payload.get("method") or ""),
        "action": str(payload.get("action") or ""),
        "text": str(payload.get("text") or ""),
        "caption": str(payload.get("caption") or ""),
        "markdown": bool(payload.get("markdown", False)),
        "parameters": payload.get("parameters") or {},
        "files": files,
        "album_captions": album_caps,
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
