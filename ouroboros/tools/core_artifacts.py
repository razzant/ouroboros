"""The verbs a task uses to put something in front of a human.

Photo, video and document delivery to the owner chat; link buttons; and
``escalate``, the one question verb — an owner quiz card at the root, a
parent mailbox frame below it. The payload validators for the two structured
frames (link actions, quiz) live here with the verbs that emit them, because
they define those wire shapes; ``supervisor.message_bus`` re-validates
through the same functions on the delivery side.
"""

from __future__ import annotations

import base64
import ipaddress
import mimetypes
import pathlib
import uuid
from typing import Any, Dict, List

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
    """Send an owner-chat image from a file or legacy base64 payload."""
    _photo_chat_id = getattr(ctx, "current_chat_id", None)
    if _photo_chat_id is None or _photo_chat_id == "":  # 0 is a real hidden session
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
            actual_b64 = base64.b64encode(raw).decode()
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

    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_photo",
        "chat_id": _photo_chat_id,
        "image_base64": actual_b64,
        "mime": mime,
        "caption": caption or "",
    })
    text = "OK: photo sent to owner chat." if mode == "live" else "OK: photo queued for delivery to owner."
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text))


_MAX_VIDEO_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


def _detect_video_mime(file_path: str, data: bytes) -> str:
    """Detect video MIME type from path extension or magic bytes."""
    if len(data) >= 8 and data[4:8] == b'ftyp':
        return "video/mp4"
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return "video/webm"
    mime, _ = mimetypes.guess_type(file_path)
    if mime and str(mime).lower().startswith("video/"):
        return mime
    return "video/mp4"


def _send_video(ctx: ToolContext, file_path: str = "", caption: str = "") -> str:
    """Send an owner-chat video from a file."""
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
        actual_b64 = base64.b64encode(raw).decode()
    except Exception as e:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to read video file: {e}"))

    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_video",
        "chat_id": chat_id,
        "video_base64": actual_b64,
        "mime": mime,
        "caption": caption or "",
    })
    text = "OK: video sent to owner chat." if mode == "live" else "OK: video queued for delivery to owner."
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text))


_MAX_DOCUMENT_FILE_BYTES = 50 * 1024 * 1024  # 50 MB (Telegram bot sendDocument limit)


def _detect_document_mime(file_path: str) -> str:
    """Best-effort MIME for an arbitrary document/file from its extension."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def _send_file(ctx: ToolContext, file_path: str = "", caption: str = "") -> str:
    """Send an owner-chat document/file (report, archive, code, PDF, etc.) from a local path."""
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
        actual_b64 = base64.b64encode(raw).decode()
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

    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_document",
        "chat_id": chat_id,
        "file_base64": actual_b64,
        "mime": mime,
        "filename": fp.name,
        "caption": caption or "",
        "download_url": download_url,
    })
    text = (f"OK: file '{fp.name}' sent to owner chat." if mode == "live"
            else f"OK: file '{fp.name}' queued for delivery to owner.")
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text))


_MAX_LINK_ACTIONS = 12


class LinkActionsValidationError(ValueError):
    """Typed atomic refusal from the shared link-action validator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_link_actions(actions: Any) -> List[Dict[str, str]]:
    """Return one cleaned HTTP(S) action batch, or refuse the entire batch."""
    from urllib.parse import urlparse

    if not isinstance(actions, list) or not actions:
        raise LinkActionsValidationError(
            "SEND_LINKS_ARG_ERROR", "provide a non-empty links array."
        )
    if len(actions) > _MAX_LINK_ACTIONS:
        raise LinkActionsValidationError(
            "SEND_LINKS_TOO_MANY", f"maximum {_MAX_LINK_ACTIONS} links."
        )
    cleaned: List[Dict[str, str]] = []
    for item in actions:
        if not isinstance(item, dict):
            raise LinkActionsValidationError(
                "SEND_LINKS_ARG_ERROR", "each link must contain label and url."
            )
        label = str(item.get("label") or "")
        url = str(item.get("url") or "")
        invalid_url_char = any(
            ord(char) < 0x20 or ord(char) == 0x7F or char.isspace() for char in url
        )
        invalid_label_char = any(
            ord(char) < 0x20 or ord(char) == 0x7F
            or (char.isspace() and char != " ") for char in label
        )
        label = label.strip()
        if len(url) > 2048 or invalid_url_char:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED",
                "URL contains disallowed characters or exceeds 2048 characters.",
            )
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            port = parsed.port
            if "[" in parsed.netloc:
                ipaddress.ip_address(hostname or "")
        except ValueError as exc:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            ) from exc
        if parsed.scheme not in {"http", "https"}:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "only http:// and https:// URLs are allowed."
            )
        if not hostname or (port is not None and not 0 <= port <= 65535):
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            )
        if "\\" in parsed.netloc:
            raise LinkActionsValidationError("SEND_LINKS_URL_BLOCKED", "URL has an invalid authority.")
        invalid_reg_name = "[" not in parsed.netloc and any(
            not ((char.isascii() and (char.isalnum() or char in "._~-")) or
                 (not char.isascii() and char.isalpha())) for char in hostname
        )
        if "%" in hostname or invalid_reg_name:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            )
        if not label or invalid_label_char:
            raise LinkActionsValidationError(
                "SEND_LINKS_ARG_ERROR", "each link requires a label and absolute URL."
            )
        cleaned.append({"label": label[:120], "url": url})
    return cleaned


_MAX_QUIZ_OPTIONS = 6
_MAX_QUIZ_QUESTION_CHARS = 2000


class QuizValidationError(ValueError):
    """Typed atomic refusal from the shared quiz payload validator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_quiz_payload(
    question: Any, options: Any, stake: Any, assumption: Any,
) -> Dict[str, Any]:
    """Return one cleaned quiz payload, or refuse the entire card.

    Shared by the asking tool and the message bus (one validator, two
    callers — the LinksOutbound pattern). ``assumption`` is REQUIRED by
    owner decision 27=A: a quiz is fire-and-continue, so the card must name
    what the task keeps doing while the owner has not answered.
    """
    q_text = str(question or "").strip()
    if not q_text or len(q_text) > _MAX_QUIZ_QUESTION_CHARS:
        raise QuizValidationError(
            "QUIZ_QUESTION_INVALID",
            f"question must be 1..{_MAX_QUIZ_QUESTION_CHARS} characters.",
        )
    if not isinstance(options, list) or not 2 <= len(options) <= _MAX_QUIZ_OPTIONS:
        raise QuizValidationError(
            "QUIZ_OPTIONS_INVALID",
            f"provide 2..{_MAX_QUIZ_OPTIONS} options.",
        )
    cleaned: List[Dict[str, str]] = []
    for item in options:
        if isinstance(item, str):
            item = {"label": item}
        if not isinstance(item, dict):
            raise QuizValidationError(
                "QUIZ_OPTIONS_INVALID", "each option needs a label."
            )
        label = str(item.get("label") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if not label:
            raise QuizValidationError(
                "QUIZ_OPTIONS_INVALID", "each option needs a non-empty label."
            )
        option: Dict[str, str] = {"label": label[:120]}
        if detail:
            option["detail"] = detail[:500]
        cleaned.append(option)
    assumption_text = str(assumption or "").strip()
    if not assumption_text:
        raise QuizValidationError(
            "QUIZ_ASSUMPTION_REQUIRED",
            "state the assumption you continue under until the owner answers.",
        )
    return {
        "question": q_text,
        "options": cleaned,
        "stake": str(stake or "").strip()[:500],
        "assumption": assumption_text[:500],
    }


def _send_links(
    ctx: ToolContext,
    links: list | None = None,
    title: str = "",
) -> str:
    """Queue validated HTTP(S) links as first-class chat actions."""
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None or chat_id == "":
        return "⚠️ SEND_LINKS_NO_CHAT: no active chat."
    try:
        actions = validate_link_actions(links)
    except LinkActionsValidationError as exc:
        return f"⚠️ {exc.code}: {exc}"
    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_links",
        "chat_id": chat_id,
        "title": str(title or "")[:240],
        "actions": actions,
    })
    if mode == "live":
        return "OK: link buttons sent to owner chat."
    return "OK: link buttons queued for delivery to owner."


def _escalate(
    ctx: ToolContext,
    question: str,
    options: list | None = None,
    stake: str = "",
    assumption: str = "",
) -> str:
    """One escalation verb for the whole tree (owner decision 31 hierarchy).

    A ROOT task addresses the OWNER: a typed quiz card in the chat
    (fire-and-continue under the mandatory ``assumption`` — decision 27=A).
    A SUBAGENT addresses its PARENT: a typed mailbox frame the parent answers
    with ``forward_to_worker`` or raises higher by calling ``escalate``
    itself, forwarding the payload verbatim. The owner only ever sees what no
    ancestor was willing to answer. Expiry is structural only (decision
    30=A): the question dies with its author; there is no host deadline.
    """
    from ouroboros.tool_capabilities import BACKGROUND_DELEGATION_ROLE

    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    if str(meta.get("delegation_role") or "") == BACKGROUND_DELEGATION_ROLE:
        # Background cognition has no owner-interactive loop and no parent:
        # a card it can never collect an answer for would be a zombie.
        return ("⚠️ ESCALATE_UNAVAILABLE: background consciousness cannot escalate — "
                "record the open question in memory or scratchpad instead.")
    try:
        payload = validate_quiz_payload(question, options, stake, assumption)
    except QuizValidationError as exc:
        return f"⚠️ {exc.code}: {exc}"
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return "⚠️ ESCALATE_UNAVAILABLE: escalate requires an active task context."
    if bool(getattr(ctx, "is_direct_chat", False)):
        # A direct chat turn IS the owner conversation: it is not a queue
        # task the answer ingress can address, and a card would be a dead
        # end — ask the question directly in the reply instead.
        return ("⚠️ ESCALATE_UNAVAILABLE: this is a live owner conversation — "
                "ask the question directly in your reply instead of a card.")
    parent_task_id = str(meta.get("parent_task_id") or "").strip()
    delegation_role = str(meta.get("delegation_role") or "").strip()
    if delegation_role and not parent_task_id:
        # Fail-closed on corrupted lineage: a delegated context without its
        # parent id must never fall through to the OWNER card path (decision
        # 31 — the owner sees only what no ancestor answered).
        return ("⚠️ ESCALATE_UNAVAILABLE: delegated context without a parent "
                "task id — record the open question in your result instead.")

    if parent_task_id:
        # Upward hop: descendant -> nearest LIVE ancestor (the mirror of
        # forward_to_worker). A live subagent may legitimately OUTLIVE its
        # direct parent (the queue keeps descendants running past a settled
        # intermediate), so one settled/unknown/cancel-pending link is not a
        # dead end — the walk continues toward the root; only a chain with NO
        # live ancestor is the typed terminal (decision 31: the owner-facing
        # card still belongs to the ROOT alone, so an orphaned subtree keeps
        # the assumption path).
        from ouroboros.owner_mailbox import write_task_message
        from ouroboros.task_status import FINAL_STATUSES, load_effective_task_result

        status_drive_root = pathlib.Path(str(meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
        from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED

        root_task_id = str(meta.get("root_task_id") or "").strip()
        target_id, data = "", {}
        candidate, seen = parent_task_id, set()
        for _ in range(10):
            if not candidate or candidate in seen or candidate == task_id:
                break
            seen.add(candidate)
            row = load_effective_task_result(status_drive_root, candidate)
            status = str(row.get("status") or "").lower()
            # A scheduled ancestor is a legitimate addressee (its mailbox is
            # drained when it starts); unknown/empty status is not.
            alive = bool(row) and status not in FINAL_STATUSES \
                and status in {STATUS_RUNNING, STATUS_SCHEDULED}
            if alive:
                try:
                    from ouroboros.cancel_intents import cancel_pending

                    if cancel_pending(status_drive_root, candidate):
                        alive = False
                except Exception:
                    pass
            if alive:
                target_id, data = candidate, row
                break
            next_candidate = str(row.get("parent_task_id") or "").strip()
            if not next_candidate and candidate != root_task_id:
                next_candidate = root_task_id
            candidate = next_candidate
        if not target_id:
            return ("⚠️ ESCALATE_PARENT_SETTLED: no live ancestor is left to "
                    f"answer (walked up from parent {parent_task_id}) — proceed "
                    "under your stated assumption and record the open question "
                    "in your result.")
        parent_task_id = target_id
        lines = [f"ESCALATION (decision requested): {payload['question']}", "Options:"]
        lines += [
            f"{i + 1}. {row['label']}" + (f" — {row['detail']}" if row.get("detail") else "")
            for i, row in enumerate(payload["options"])
        ]
        if payload["stake"]:
            lines.append(f"At stake: {payload['stake']}")
        lines.append(f"I continue meanwhile under the assumption: {payload['assumption']}")
        lines.append(
            f"Answer with forward_to_worker(task_id={task_id}, message=...), or "
            "escalate this question yourself (verbatim) if it is above your authority."
        )
        parent_drive = str(data.get("child_drive_root") or data.get("headless_child_drive_root") or data.get("drive_root") or "").strip()
        written = write_task_message(
            pathlib.Path(parent_drive) if parent_drive else status_drive_root,
            "\n".join(lines),
            task_id=parent_task_id,
            source_task_id=task_id,
            provenance="descendant_task",
        )
        if not written:
            return f"⚠️ ESCALATE_UNWRITTEN: the escalation to parent {parent_task_id} was not persisted."
        return (f"OK: escalated to parent task {parent_task_id}; continuing under "
                f"assumption: {payload['assumption']}")

    # Root task: the owner gets a typed quiz card.
    from ouroboros.owner_quiz import record_asked

    quiz_id = uuid.uuid4().hex
    canonical_root = pathlib.Path(str(meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    asked = record_asked(
        canonical_root, task_id,
        quiz_id=quiz_id, question=payload["question"],
        options=[row["label"] for row in payload["options"]],
        stake=payload["stake"], assumption=payload["assumption"],
    )
    if asked.get("refused"):
        return ("⚠️ ESCALATE_UNAVAILABLE: this task already has the maximum "
                "number of unanswered owner questions open; proceed under your "
                f"stated assumption: {payload['assumption']}")
    from ouroboros.tools.owner_delivery import deliver_owner_event

    mode = deliver_owner_event(ctx, {
        "type": "send_quiz",
        "chat_id": getattr(ctx, "current_chat_id", None) or 0,
        "quiz_id": quiz_id,
        "question": payload["question"],
        "options": payload["options"],
        "stake": payload["stake"],
        "assumption": payload["assumption"],
        "state": "open",
        "task_id": task_id,
    })
    delivered = "delivered to the owner" if mode == "live" else "queued for the owner"
    return (f"OK: quiz {quiz_id} {delivered}; continuing under assumption: "
            f"{payload['assumption']}. The answer (if any) arrives as an owner "
            "quiz answer in a later round; the card expires when this task ends.")
