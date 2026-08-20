"""Owner-chat media and document delivery helpers."""

from __future__ import annotations

import pathlib

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


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
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text="⚠️ No active chat — cannot send photo."))

    actual_b64 = ""
    mime = "image/png"

    if file_path:
        fp = pathlib.Path(file_path).expanduser().resolve()
        if not fp.exists():
            return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File not found: {file_path}"))
        if fp.stat().st_size > _MAX_PHOTO_FILE_BYTES:
            return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File too large ({fp.stat().st_size} bytes). Max: {_MAX_PHOTO_FILE_BYTES} bytes."))
        try:
            raw = fp.read_bytes()
            mime = _detect_image_mime(raw)
            actual_b64 = __import__("base64").b64encode(raw).decode()
        except Exception as e:
            return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to read image file: {e}"))
    elif image_base64:
        if image_base64 == "__last_screenshot__":
            if not ctx.browser_state.last_screenshot_b64:
                return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text="⚠️ No screenshot stored. Take one first with browse_page(output='screenshot')."))
            actual_b64 = ctx.browser_state.last_screenshot_b64
        else:
            actual_b64 = image_base64
    else:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text="⚠️ Provide either file_path or image_base64."))

    if not actual_b64 or len(actual_b64) < 100:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text="⚠️ Image data is empty or too short."))

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
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text="OK: photo queued for delivery to owner."))


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
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text="⚠️ No active chat — cannot send video."))
    if not file_path:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text="⚠️ Provide a file_path."))

    fp = pathlib.Path(file_path).expanduser().resolve()
    if not fp.exists():
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File not found: {file_path}"))
    if fp.stat().st_size > _MAX_VIDEO_FILE_BYTES:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File too large ({fp.stat().st_size} bytes). Max: {_MAX_VIDEO_FILE_BYTES} bytes."))

    try:
        raw = fp.read_bytes()
        mime = _detect_video_mime(str(fp), raw)
        actual_b64 = __import__("base64").b64encode(raw).decode()
    except Exception as e:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to read video file: {e}"))

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
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text="OK: video queued for delivery to owner."))


_MAX_DOCUMENT_FILE_BYTES = 50 * 1024 * 1024  # 50 MB (Telegram bot sendDocument limit)


def _detect_document_mime(file_path: str) -> str:
    """Best-effort MIME for an arbitrary document/file from its extension."""
    mime, _ = __import__("mimetypes").guess_type(file_path)
    return mime or "application/octet-stream"


def _send_file(ctx: ToolContext, file_path: str = "", caption: str = "") -> str:
    """Queue an owner-chat document/file (report, archive, code, PDF, etc.) from a local path."""
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None or chat_id == "":
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text="⚠️ No active chat — cannot send file."))
    if not file_path:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text="⚠️ Provide a file_path."))

    fp = pathlib.Path(file_path).expanduser().resolve()
    if not fp.exists() or not fp.is_file():
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File not found: {file_path}"))
    if fp.stat().st_size > _MAX_DOCUMENT_FILE_BYTES:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File too large ({fp.stat().st_size} bytes). Max: {_MAX_DOCUMENT_FILE_BYTES} bytes."))

    try:
        raw = fp.read_bytes()
        mime = _detect_document_mime(str(fp))
        actual_b64 = __import__("base64").b64encode(raw).decode()
    except Exception as e:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to read file: {e}"))

    # Copy into the task's canonical artifact store so the delivered file stays
    # downloadable after reload even if the original path is temporary / GC'd,
    # and derive a loopback download URL from that DURABLE copy (WKWebView-safe
    # desktop download + base64-free history replay).
    download_url = ""
    try:
        from ouroboros.artifacts import copy_file_to_task_artifacts
        from ouroboros.gateway.files import download_url_for_local_file

        record = copy_file_to_task_artifacts(ctx, fp, kind="user_file")
        durable = pathlib.Path(str(record.get("path"))) if record and record.get("path") else fp
        download_url = download_url_for_local_file(durable)
    except Exception:
        download_url = ""  # non-fatal: fall back to base64 blob delivery

    _doc_meta = getattr(ctx, "task_metadata", {})
    _doc_meta = _doc_meta if isinstance(_doc_meta, dict) else {}
    ctx.pending_events.append({
        "type": "send_document",
        "chat_id": chat_id, "task_id": str(getattr(ctx, "task_id", "") or ""),  # task_id -> bound-task project-panel routing
        # Lineage so a SUBAGENT's file routes to its root's project thread (C4.4).
        "parent_task_id": str(_doc_meta.get("parent_task_id") or ""),
        "root_task_id": str(_doc_meta.get("root_task_id") or ""),
        "file_base64": actual_b64,
        "mime": mime,
        "filename": fp.name,
        "caption": caption or "",
        "download_url": download_url,
    })
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=f"OK: file '{fp.name}' queued for delivery to owner."))
