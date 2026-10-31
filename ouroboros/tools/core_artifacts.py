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
from typing import Any, Dict, List, Optional

from ouroboros.delegate_reply import REPLY_TO_SCHEMA, validate_reply_target, reply_instruction
from ouroboros.tools.arg_feedback import argument_refusal
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
    text = "OK: photo accepted for delivery to owner chat." if mode == "live" else "OK: photo queued for delivery to owner."
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
    text = "OK: video accepted for delivery to owner chat." if mode == "live" else "OK: video queued for delivery to owner."
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
    if not fp.exists():
        from ouroboros.evidence_packs import materialize_path
        materialize_path(fp)  # a packed checkpoint handle is written back beside its name first
    if not fp.exists() or not fp.is_file():
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ File not found: {file_path}"))
    mime = _detect_document_mime(str(fp))
    # Capture immutable, task-owned bytes before publishing a link. File delivery
    # stays available above the old inline limit without placing the file in RAM.
    try:
        from types import SimpleNamespace
        from urllib.parse import quote
        from ouroboros.artifacts import copy_file_to_task_artifacts, task_id_for_artifacts
        from ouroboros.gateway.files import download_url_for_local_file

        if getattr(ctx, "drive_root", None) is None:
            raise OSError("artifact context is unavailable")
        task_id = task_id_for_artifacts(ctx)
        record = copy_file_to_task_artifacts(ctx, fp, kind="user_file", immutable=True)
        if not record:
            raise ValueError("file was refused by the artifact store")
        metadata = getattr(ctx, "task_metadata", {}) or {}
        canonical = pathlib.Path(getattr(ctx, "budget_drive_root", None) or metadata.get("budget_drive_root") or ctx.drive_root)
        if canonical.resolve() != pathlib.Path(ctx.drive_root).resolve():
            record = copy_file_to_task_artifacts(SimpleNamespace(drive_root=canonical, task_id=task_id),
                                                 pathlib.Path(record["path"]), kind="user_file", immutable=True, expected=record)
        durable = pathlib.Path(record["path"])
        file_ref = {"kind": "task_artifact", "root": "artifact_store", "task_id": task_id,
                    "path": record["name"], "size": record["size"], "sha256": record["sha256"]}
        download_url = f"/api/tasks/{quote(task_id, safe='')}/artifacts/{quote(record['name'], safe='')}"
        compat_url = download_url_for_local_file(durable)
        # Existing transport subscribers can keep consuming bounded inline files.
        actual_b64 = base64.b64encode(durable.read_bytes()).decode() if record["size"] <= _MAX_DOCUMENT_FILE_BYTES else ""
    except (OSError, TypeError) as exc:
        # Preserve the historical bounded inline delivery when storage or an
        # older caller's artifact context is unavailable. No uncaptured URL or
        # reference is published, and source-read failures still refuse delivery.
        try:
            from ouroboros.store import open_for_read

            with open_for_read(fp) as source:
                raw = source.read(_MAX_DOCUMENT_FILE_BYTES + 1)
            if len(raw) > _MAX_DOCUMENT_FILE_BYTES:
                raise OSError(f"large file requires artifact capture: {exc}")
            actual_b64 = base64.b64encode(raw).decode()
            file_ref, download_url, compat_url = None, "", ""
        except OSError as read_error:
            return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to read or capture file: {read_error}"))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ Failed to capture file: {exc}"))

    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_document",
        "chat_id": chat_id,
        "file_base64": actual_b64,
        "mime": mime,
        "filename": fp.name,
        "caption": caption or "",
        "download_url": download_url,
        "download_url_compat": compat_url,
        "file_ref": file_ref,
    })
    text = (f"OK: file '{fp.name}' accepted for delivery to owner chat." if mode == "live"
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
# A label is button text; it is refused beyond this bound, never sliced.
_MAX_QUIZ_LABEL_CHARS = 120


class QuizValidationError(ValueError):
    """Typed atomic refusal from the shared quiz payload validator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_quiz_payload(
    question: Any, options: Any, stake: Any, assumption: Any,
    *, wait_for_answer: bool = False, max_wait_minutes: Any = None, task: Optional[dict] = None,
) -> Dict[str, Any]:
    """Return one cleaned quiz payload, or refuse the entire card.

    Shared by the asking tool and the message bus (one validator, two
    callers — the LinksOutbound pattern). Optional questions name an
    assumption; required waiting has no implied default answer.

    ``max_wait_minutes`` bounds a required wait only, and never past ``task``'s
    execution time limit (a wait counts as execution): beyond it that limit would
    end the task first, so a larger bound would be a promise the runtime cannot keep.

    The authored text (question, option details, stake, assumption) is kept
    whole: the card may be the only explanation its reader gets, so there is no
    quiz-specific length cap and nothing is silently cut (BIBLE P1). Only a
    label, which is button text, has a bound, and an over-long label refuses
    the card instead of being sliced.
    """
    q_text = str(question or "").strip()
    if not q_text:
        raise QuizValidationError("QUIZ_QUESTION_INVALID", "question must be non-empty.")
    if options is None:
        options = []
    if not isinstance(options, list) or len(options) > _MAX_QUIZ_OPTIONS:
        raise QuizValidationError(
            "QUIZ_OPTIONS_INVALID",
            f"provide at most {_MAX_QUIZ_OPTIONS} options.",
        )
    cleaned: List[Dict[str, Any]] = []
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
        if len(label) > _MAX_QUIZ_LABEL_CHARS:
            raise QuizValidationError(
                "QUIZ_OPTIONS_INVALID",
                f"option labels must be at most {_MAX_QUIZ_LABEL_CHARS} characters.",
            )
        option: Dict[str, Any] = {"label": label}
        if detail:
            option["detail"] = detail
        if item.get("recommended") is True:  # the asker's recommendation rides with its option
            option["recommended"] = True
        cleaned.append(option)
    if sum(1 for option in cleaned if option.get("recommended")) > 1:
        # One recommendation: the durable record keeps one index, so every surface
        # (live card, replay, Telegram, parent frame) must be able to show the same one.
        raise QuizValidationError("QUIZ_RECOMMENDED_INVALID", "mark at most one option as recommended.")
    assumption_text = str(assumption or "").strip()
    if not isinstance(wait_for_answer, bool):
        raise QuizValidationError("QUIZ_WAIT_INVALID", "wait_for_answer must be a boolean.")
    if not assumption_text and not wait_for_answer:
        raise QuizValidationError(
            "QUIZ_ASSUMPTION_REQUIRED",
            "state the assumption you continue under until the owner answers "
            "(what you do meanwhile, for example your recommended option).",
        )
    bound = _validate_wait_bound(max_wait_minutes, wait_for_answer=wait_for_answer, task=task)
    return {
        "question": q_text,
        "options": cleaned,
        "stake": str(stake or "").strip(),
        "assumption": assumption_text,
        **({"max_wait_minutes": bound} if bound is not None else {}),
    }


# The escalate catalog entry lives beside its validator and handler: the
# description and field texts define the same wire shape validate_quiz_payload
# enforces (core.get_tools registers it).
ESCALATE_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "escalate",
    "description": (
        "Escalate one decision up the responsibility chain instead of guessing. "
        "A root task asks its human through a quiz card; a subagent asks its parent "
        "through a typed mailbox frame, which the parent answers with forward_to_worker "
        "or escalates higher verbatim.\n\n"
        "The card may be the only thing your human sees. It may be read later, outside "
        "this room, by someone who remembers the purpose of the work without remembering "
        "its mechanisms or your earlier discussion. Issue, specification and question "
        "numbers, function names and reviewer names belong to your working context; they "
        "are not shared context by themselves. The card carries the explanation needed to "
        "understand this decision and its consequences.\n\n"
        "The source of the fork belongs in that explanation, in ordinary prose: the human's "
        "words, a document they supplied, your idea, or a helper's or reviewer's proposal. Human words "
        "the fork rests on are quoted exactly when available; missing wording is disclosed, "
        "and your paraphrase is identified as your interpretation. Your own proposal is a "
        "distinct option, not an assumed premise of every option.\n\n"
        "Offer 0-6 real alternatives for this decision: none for an open question the human "
        "answers in their own words; with options, mark your recommendation with "
        "recommended=true. A substantive position you or a helper set aside is one of them, "
        "stated at its strongest with its cost and why you set it aside. "
        "By default, state the assumption you continue under and keep "
        "working; the card stays answerable and a late answer still arrives. Set "
        "wait_for_answer=true on a live root, including an ordinary conversation, when the "
        "next step is irreversible or costly to redo, or the choice belongs to the human; "
        "your judgment decides. Waiting begins after the current tool batch, without model "
        "calls. Waiting questions in one batch share one wait, ending on the first incoming "
        "message, not necessarily an owner answer. A plain-text clarification ends this "
        "turn; a waited question keeps it alive. How many decisions to raise and when "
        "remains your judgment. Unanswered questions remain available from Main after the "
        "task ends. To explicitly retire a question, use action=close, quiz_id and a "
        "non-empty reason; question_task_id may identify a recorded Continue or retry predecessor, "
        "or another task that has already finished. Before you finish, close each of your own "
        "unanswered questions whose answer would no longer change anything, saying why; leave the "
        "rest open. Closing is your decision with an explanation, never the human's approval. "
        "Materials may name existing task artifacts needed for the decision."
    ),
    "parameters": {"type": "object", "properties": {
        "reply_to": REPLY_TO_SCHEMA,
        "request_id": {"type": "string", "description": "Stable ask retry key, at most 128 characters. Reuse it with identical question fields after uncertain delivery. After its wait ends, use a new request_id to wait again. With a retained Presence source, asks go privately to Main and omitted keys deduplicate identical asks from that work. Other roots mint a new ask when the key is omitted."},
        "action": {"type": "string", "enum": ["ask", "close"], "default": "ask"},
        "quiz_id": {"type": "string", "description": "Existing question to close; used only with action=close."},
        "question_task_id": {"type": "string", "description": "Question author for action=close; defaults to this task. Your own, a recorded Continue/retry predecessor's, or an already finished task's question may be closed; another live task's may not."},
        "reason": {"type": "string", "description": "Required explanation shown beside an explicitly closed question. Silence or task completion is not approval."},
        "materials": {"type": "array", "items": {"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Artifact's owning task; defaults to this task. Your task, descendants and recorded Continue/retry predecessors are supported."},
            "name": {"type": "string", "description": "Exact registered artifact name from the task result inventory; never a filesystem path."},
        }, "required": ["name"], "additionalProperties": False}, "description": "Existing artifacts needed to answer this question. Selected bytes are retained before publication; no automatic report attachment."},
        "question": {"type": "string", "description": (
            "The self-contained explanation of one decision, in the reader's language. "
            "Markdown renders in chat. No quiz-specific character limit.")},
        "options": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string", "description": (
                "Short name of the choice, understandable to the reader on a button "
                "(max 120 characters).")},
            "detail": {"type": "string", "description": (
                "What choosing this option changes for the human: what they gain and what "
                "they give up, including any relevant cost, delay or lost capability (optional).")},
            "recommended": {"type": "boolean", "description": (
                "True on the one option you recommend; omitted or false otherwise.")},
        }, "required": ["label"]}, "description": (
            "Optional 0-6 mutually exclusive alternatives for this decision; omit for an open "
            "question answered in the human's own words.")},
        "stake": {"type": "string", "description": (
            "What depends on this decision for the human or the work (optional).")},
        "assumption": {"type": "string", "description": (
            "What you will do while the question remains unanswered. Required when continuing "
            "without a wait; may be empty when wait_for_answer=true. It is your assumption, "
            "not the human's answer.")},
        "wait_for_answer": {"type": "boolean", "default": False, "description": (
            "Live roots: wait for addressed owner input before another model round. Default "
            "false; an unanswered card remains answerable either way.")},
        "max_wait_minutes": {"type": "integer", "description": (
            "Optional positive whole-minute bound for wait_for_answer, within the task's "
            "existing lifetime limit. On expiry, resume with a system notice; the card stays "
            "answerable. Silence is not an answer.")},
    }},
}


def _validate_wait_bound(max_wait_minutes: Any, *, wait_for_answer: bool, task: Optional[dict] = None) -> Optional[int]:
    """The optional wait bound in whole minutes, capped by the task's execution time limit."""
    if max_wait_minutes is None or not wait_for_answer:
        # On an optional question a bound only spells out the documented default (no wait):
        # it takes the omitted path, and the asker's receipt says so.
        return None
    if (isinstance(max_wait_minutes, bool) or not isinstance(max_wait_minutes, int)
            or max_wait_minutes < 1):
        raise QuizValidationError(
            "QUIZ_WAIT_BOUND_INVALID",
            "max_wait_minutes must be a positive integer; omit it for an unbounded wait.",
        )
    from ouroboros.config import resolve_task_execution_timeout_sec

    # The bound the supervisor enforces: an explicit task budget narrows the global ceiling.
    ceiling_sec = resolve_task_execution_timeout_sec(task if isinstance(task, dict) else {})
    if ceiling_sec is None:  # no task lifetime: any positive bound is within it
        return int(max_wait_minutes)
    ceiling_minutes = int(ceiling_sec) // 60
    if ceiling_minutes == 0:
        raise QuizValidationError(
            "QUIZ_WAIT_BOUND_INVALID",
            "The task's execution time limit is shorter than one minute; "
            "omit max_wait_minutes (the execution limit still applies).",
        )
    if max_wait_minutes > ceiling_minutes:
        raise QuizValidationError(
            "QUIZ_WAIT_BOUND_INVALID",
            f"max_wait_minutes must be at most {ceiling_minutes} "
            "(the task's execution time limit); omit it for an unbounded wait.",
        )
    return int(max_wait_minutes)


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
        # Validation rejects the whole batch before delivery. The adapter keeps
        # URL policy denials distinct from bad arguments, including TOO_MANY;
        # no links were sent on any of these branches.
        from ouroboros.tools.tool_result import LegacyTextResultAdapter

        return _publish_tool_result(ctx, LegacyTextResultAdapter.from_text(
            "send_links", f"⚠️ {exc.code}: {exc} No links were sent."))
    from ouroboros.tools.owner_delivery import deliver_owner_event
    mode = deliver_owner_event(ctx, {
        "type": "send_links",
        "chat_id": chat_id,
        "title": str(title or "")[:240],
        "actions": actions,
    })
    if mode == "live":
        return "OK: link buttons accepted for delivery to owner chat."
    return "OK: link buttons queued for delivery to owner."


def _quiz_host_facts(ctx: ToolContext, canonical_root: pathlib.Path, task_id: str, chat_id: int) -> str:
    """The host's one-sentence account under a root's owner card: which task asks,
    how its run started, and when the owner last wrote in this chat. Read from
    typed records only (the task record's ``run_origin`` provenance and the chat
    store), never from the question text; an unrecorded fact says unknown."""
    from ouroboros import dialogue_store as ds
    from ouroboros.consciousness_authority import CONSCIOUSNESS_INITIATOR
    from ouroboros.deadline_utils import parse_deadline_ts as moment, utc_now
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.followup import FOLLOWUP_SOURCE

    def shown(when: Any) -> str:
        return when.strftime("%Y-%m-%d %H:%M UTC")

    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    try:
        record = load_task_result(canonical_root, task_id) or {}
    except Exception:
        record = {}
    # The live metadata (which carries the owner door's stamp) laid over the
    # persisted record's own, as the post-task synthesis reads the same origin.
    metadata = {**(record.get("metadata") if isinstance(record.get("metadata"), dict) else {}), **meta}
    origin = run_origin({**record, "metadata": metadata})
    ref = metadata.get("origin_message_ref") or record.get("origin_message_ref")
    origin_task = str(origin.get("origin_task_id") or "")
    if origin.get("owner_ingress"):
        sent = moment(ref.get("ts")) if isinstance(ref, dict) else None
        started = "started by your message" + (f" of {shown(sent)}" if sent else "")
    elif origin.get("source") == FOLLOWUP_SOURCE and origin_task:
        started = f"started as a scheduled follow-up of task {origin_task}"
    elif origin.get("initiator") == CONSCIOUSNESS_INITIATOR:
        started = "started by background consciousness"
    elif origin.get("source") == "promote_chat_to_task":
        started = "started by promotion" + (f" from task {origin_task}" if origin_task else "")
    elif origin.get("schedule_id"):
        started = f"started by schedule {origin['schedule_id']}"
    elif origin_task:
        started = f"started from task {origin_task}"
    else:
        started = "origin unknown" + (f" (recorded source: {origin['source']})" if origin.get("source") else "")
    last = None
    coverage = ""
    try:
        before = None
        while (entry := ds.last_inbound(canonical_root, chat_id, before=before)) is not None:
            last = moment(entry.row.get("ts"))
            if last is not None:
                break
            before = entry.seq
        lower = entry.seq if entry else 0
        ds.require_imported(canonical_root, "chat", lower=lower)
        if ds.range_coverage(canonical_root, "chat", lower=lower)["gaps"]:  # a later message may be unreadable
            raise ds.StoreError("Dialogue history has recorded invalid intervals")
    except ds.StoreMigrating:
        last, coverage = None, " (history is still importing)"
    except ds.StoreError as exc:
        last, coverage = None, f" ({exc})"
    except Exception:
        last = None
    if last is None:
        seen = "your last message in this chat: unknown" + coverage
    else:
        minutes = max(0, int((utc_now() - last).total_seconds() // 60))
        seen = f"your last message in this chat: {shown(last)} ({minutes} minutes before this question)"
    return f"Asked by task {task_id}, {started}; {seen}."


def _escalate(
    ctx: ToolContext,
    question: str = "",
    options: list | None = None,
    stake: str = "",
    assumption: str = "",
    wait_for_answer: bool = False,
    max_wait_minutes: int | None = None,
    action: str = "ask",
    quiz_id: str = "",
    reason: str = "",
    question_task_id: str = "",
    materials: list | None = None,
    reply_to: dict | None = None,
    request_id: str = "",
) -> str:
    """One escalation verb for the whole tree (owner decision 31 hierarchy).

    A ROOT task addresses the OWNER: a typed quiz card in the chat
    (optional clarification continues under ``assumption``; required waiting
    preserves the task at the next completed-tool boundary, optionally only
    for ``max_wait_minutes`` before it resumes with a system notice).
    A SUBAGENT addresses its PARENT: a typed mailbox frame the parent answers
    with ``forward_to_worker`` or raises higher by calling ``escalate``
    itself, forwarding the payload verbatim. The owner only ever sees what no
    ancestor was willing to answer. Expiry stays structural (no host deadline):
    the task-done seam expires the active wait but preserves the unanswered card; a late answer is
    still accepted and reaches the chat as an ordinary owner message (В17a=A).
    A consciousness wake-up is an ordinary root turn here: it asks its owner
    through the same card and may wait for the answer like any other root.
    """
    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    if action == "close":
        return _close_owner_question(ctx, quiz_id, reason, question_task_id)
    if action not in ("", "ask"):
        return argument_refusal(ctx, "QUIZ_ACTION_INVALID", ["action must be ask or close."], effect="The quiz was not sent.")
    try:
        payload = validate_quiz_payload(question, options, stake, assumption,
                                        wait_for_answer=wait_for_answer,
                                        max_wait_minutes=max_wait_minutes, task=meta)
    except QuizValidationError as exc:
        # Typed, and it says what did NOT happen: a refusal that only restates the rule is retried unchanged.
        return argument_refusal(ctx, exc.code, [str(exc)], effect="The quiz was not sent.")
    ignored_bound = (" max_wait_minutes ignored: it applies only to wait_for_answer=true."
                     if max_wait_minutes is not None and not wait_for_answer else "")
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return "⚠️ ESCALATE_UNAVAILABLE: escalate requires an active task context."
    from ouroboros.presence_owner_contact import contact_origin, contact_identity
    # A child still escalates to its parent, never directly to Main.
    parent_task_id = str(meta.get("parent_task_id") or "").strip()
    try:
        origin = contact_origin(ctx, allow_unattributed=True) if not parent_task_id else None
    except (OSError, ValueError) as exc:
        return argument_refusal(ctx, "OWNER_CONTACT_UNAVAILABLE", [str(exc)], effect="The quiz was not sent.")
    if origin is None and bool(getattr(ctx, "is_direct_chat", False)) and (
            not callable(getattr(ctx, "owner_wait_callback", None))):
        # Native conversations with a live continuation owner are addressable
        # through the same decision ingress.
        return ("⚠️ ESCALATE_UNAVAILABLE: this is a live owner conversation — "
                "ask the question directly in your reply instead of a card.")
    parent_task_id = str(meta.get("parent_task_id") or "").strip()
    delegation_role = str(meta.get("delegation_role") or "").strip()
    wait_owner = callable(getattr(ctx, "owner_wait_callback", None)) or (
        origin is not None and callable(getattr(ctx, "review_wait_callback", None)))
    if wait_for_answer and (parent_task_id or not wait_owner):
        return "⚠️ ESCALATE_UNAVAILABLE: required owner waiting needs a root task with a live continuation owner."
    if wait_for_answer:
        from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

        if is_a2a_chat_id(getattr(ctx, "current_chat_id", None)):
            return "⚠️ ESCALATE_UNAVAILABLE: this machine-to-machine chat has no owner question delivery."
    if delegation_role not in ("", "root") and not parent_task_id:
        # A root has no parent. A child missing its lineage must not fall
        # through to the owner card path.
        return ("⚠️ ESCALATE_UNAVAILABLE: delegated context without a parent "
                "task id — record the open question in your result instead.")

    canonical_root = pathlib.Path(str(meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    try:
        reply_to = validate_reply_target(ctx, reply_to)
        material_refs = _question_materials(ctx, canonical_root, task_id, materials)
    except (ValueError, OSError) as exc:
        return argument_refusal(ctx, "QUIZ_MATERIAL_INVALID", [str(exc)], effect="The quiz was not sent.")

    if parent_task_id:
        # Upward hop: descendant -> nearest LIVE ancestor (the mirror of
        # forward_to_worker). A live subagent may legitimately OUTLIVE its
        # direct parent (the queue keeps descendants running past a settled
        # intermediate), so one settled/unknown/cancel-pending link is not a
        # dead end — the walk continues toward the root; only a chain with NO
        # live ancestor is the typed terminal (decision 31: the owner-facing
        # card still belongs to the ROOT alone, so an orphaned subtree keeps
        # the assumption path).
        from ouroboros.install_identity import StoreForeign
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
        lines = [f"ESCALATION (decision requested): {payload['question']}"]
        lines.append("Options:" if payload["options"] else "Open question — answer in your own words.")
        lines += [
            f"{i + 1}. {row['label']}" + (f" — {row['detail']}" if row.get("detail") else "")
            + (" [recommended]" if row.get("recommended") else "")
            for i, row in enumerate(payload["options"])
        ]
        if payload["stake"]:
            lines.append(f"At stake: {payload['stake']}")
        for material in material_refs:
            lines.append(f"Decision material: task_id={material['task_id']}, name={material['name']} ({material['label']})")
        lines.append(f"I continue meanwhile under the assumption: {payload['assumption']}")
        reply_task = reply_to["starter_task_id"] if reply_to else task_id
        lines.append(
            f"Answer with forward_to_worker(task_id={reply_task}, message=..., reply_to=<address if supplied>), or "
            "escalate this question yourself (verbatim) if it is above your authority."
        )
        parent_drive = str(data.get("child_drive_root") or data.get("headless_child_drive_root") or data.get("drive_root") or "").strip()
        try:
            written = write_task_message(
                pathlib.Path(parent_drive) if parent_drive else status_drive_root,
                "\n".join(lines),
                task_id=parent_task_id,
                source_task_id=task_id,
                provenance="descendant_task",
                **({"reply_to": reply_to} if reply_to else {}),
            )
        except StoreForeign as exc:
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="ACCESS_BLOCKED",
                text=f"⚠️ ESCALATE_UNWRITTEN: {exc.reason}: {exc}",
                meta={"reason": exc.reason, "path": str(exc.path)},
            ))
        if not written:
            return f"⚠️ ESCALATE_UNWRITTEN: the escalation to parent {parent_task_id} was not persisted."
        return (f"OK: escalated to parent task {parent_task_id}; continuing under "
                f"assumption: {payload['assumption']}"
                + ("\n" + reply_instruction(reply_to) if reply_to else "")
                + (f" ({ignored_bound.strip().rstrip('.')})" if ignored_bound else ""))

    # Root task: the owner gets a typed quiz card.
    from ouroboros.owner_quiz import record_asked

    fingerprint = ""
    if origin is not None or request_id:
        try:
            quiz_id, fingerprint = contact_identity(task_id, "quiz", {
                **payload, "wait_for_answer": wait_for_answer, "materials": material_refs,
                **({"reply_to": reply_to} if reply_to else {}),
            }, request_id)
        except ValueError as exc:
            return argument_refusal(ctx, "QUIZ_REQUEST_INVALID", [str(exc)], effect="The quiz was not sent.")
    else:
        quiz_id = uuid.uuid4().hex
    # The card's own chat, stored with the block: a late answer arriving after
    # this task is gone is delivered there, exactly where the card was shown.
    try:
        card_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        card_chat_id = 0
    if origin is not None:
        from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
        card_chat_id = WEB_UI_CHAT_ID
    host_facts = _quiz_host_facts(ctx, canonical_root, task_id, card_chat_id)
    question_text = payload["question"]
    asked = record_asked(
        canonical_root, task_id,
        quiz_id=quiz_id, question=question_text,
        options=[row["label"] for row in payload["options"]],
        option_details=[row.get("detail", "") for row in payload["options"]],
        recommended_index=next((i for i, row in enumerate(payload["options"]) if row.get("recommended")), None),
        stake=payload["stake"], assumption=payload["assumption"],
        wait_for_answer=wait_for_answer, chat_id=card_chat_id,
        max_wait_minutes=payload.get("max_wait_minutes"),
        host_facts=host_facts,
        materials=material_refs,
        **({"reply_to": reply_to} if reply_to else {}),
        presence_contact=origin,
        request_fingerprint=fingerprint,
    )
    if asked.get("refused"):
        if asked["refused"] in {"request_id_conflict", "wait_ended"}:
            problem = ("The previous wait ended; use a new request_id to wait again"
                       if asked["refused"] == "wait_ended" else "request_id already names different question fields")
            return argument_refusal(ctx, "QUIZ_REQUEST_CONFLICT", [problem],
                                    effect="The original question is unchanged; no new question or wait was selected.")
        if asked["refused"] == "owner_questions_unavailable":
            return argument_refusal(ctx, "QUIZ_MEMBERSHIP_UNAVAILABLE",
                ["The current-question address could not be registered; retry when storage is available."],
                effect="The question was not accepted or sent; no wait was selected.")
        if wait_for_answer:
            return ("⚠️ ESCALATE_UNAVAILABLE: the task already has the maximum number "
                    "of open owner questions. No default answer or wait was selected.")
        return ("⚠️ ESCALATE_UNAVAILABLE: this task already has the maximum "
                "number of unanswered owner questions open; proceed under your "
                f"stated assumption: {payload['assumption']}")
    from ouroboros.tools.owner_delivery import deliver_owner_event

    if asked.get("state") != "open":
        return f"OK: quiz {quiz_id} already {asked.get('state')}; no new question or wait was selected."
    mode = deliver_owner_event(ctx, {
        "type": "send_quiz",
        "chat_id": card_chat_id,
        "quiz_id": quiz_id,
        "question": question_text,
        "options": payload["options"],
        "stake": payload["stake"],
        "assumption": payload["assumption"],
        "state": "open",
        "task_id": task_id,
        "host_facts": host_facts,
        **({"presence_contact": origin} if origin else {}),
        **({"materials": material_refs} if material_refs else {}),
        **({"reply_to": reply_to} if reply_to else {}),
        **({"wait_for_answer": True} if wait_for_answer else {}),
    })
    delivered = "accepted for delivery" if mode == "live" else "queued for delivery"
    if wait_for_answer:
        bound = payload.get("max_wait_minutes")
        ctx._owner_wait_requested = quiz_id
        # One batch shares one wait: an earlier question's bound must not outlive it.
        ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "", 0
        if bound:
            # An ABSOLUTE stamp, not a countdown: the bound must survive a
            # planned restart instead of starting over on the warm resume.
            import datetime

            from ouroboros.deadline_utils import utc_now

            ctx._owner_wait_deadline_at = (
                utc_now() + datetime.timedelta(minutes=int(bound))).isoformat()
            ctx._owner_wait_max_minutes = int(bound)
        limit = (f"the task waits up to {int(bound)} minutes after this tool batch, then "
                 "continues with a notice" if bound else "the task waits after this tool batch")
        return (f"OK: quiz {quiz_id} {delivered}; {limit}, "
                "preserving its live browser and releasing active execution capacity. "
                "Any incoming mail or an owner hurry request ends the wait; only an owner answer "
                "answers the question. Other mail leaves the card open. Stop and task deadlines still apply.")
    return (f"OK: quiz {quiz_id} {delivered}; continuing under assumption: "
            f"{payload['assumption']}. The answer (if any) arrives as an owner "
            "quiz answer in a later round; the card stays answerable after this "
            f"task ends — a later answer reaches this chat as an ordinary owner message.{ignored_bound}")


def _close_owner_question(ctx: ToolContext, quiz_id: str, reason: str, question_task_id: str) -> str:
    """Explicit model closure through the existing canonical quiz and delivery seam."""
    from ouroboros.owner_quiz import record_closed
    from ouroboros.tools.owner_delivery import deliver_owner_event

    task_id = str(getattr(ctx, "task_id", "") or "")
    meta = getattr(ctx, "task_metadata", {}) or {}
    if not task_id or meta.get("parent_task_id") or meta.get("delegation_role") not in (None, "", "root"):
        return argument_refusal(ctx, "QUIZ_CLOSE_UNAVAILABLE", ["Only an active root may close an owner question."], effect="The question remains unchanged.")
    root = pathlib.Path(str(meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    source_task = str(question_task_id or task_id)
    outcome = record_closed(root, source_task, quiz_id=quiz_id, reason=reason, actor_task_id=task_id)
    if not outcome.get("ok"):
        return argument_refusal(ctx, "QUIZ_CLOSE_REFUSED", [str(outcome.get("error"))], effect="The question remains unchanged.")
    block = outcome["block"]
    if source_task == task_id and getattr(ctx, "_owner_wait_requested", "") == str(quiz_id):
        # A model may ask and explicitly close within the same tool batch;
        # its next boundary must not sleep on a question it just retired.
        ctx._owner_wait_requested = ""
        ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "", 0
    # Retries preserve the first explanation and still heal an unwritten close row.
    mode = deliver_owner_event(ctx, {
        "type": "send_quiz_state", "task_id": source_task, "quiz_id": str(quiz_id),
        "state": "closed", "closed_reason": block["closed_reason"],
        "chat_id": block.get("publication_chat_id", block.get("chat_id", 0)),
    })
    return f"OK: quiz {quiz_id} explicitly closed; explanation {('accepted' if mode == 'live' else 'queued')} for delivery. No owner answer or approval was recorded."


def _question_materials(ctx: ToolContext, root: pathlib.Path, task_id: str, materials: Any) -> list:
    """Retain explicitly named existing artifacts under this question author's custody."""
    if materials in (None, []):
        return []
    from types import SimpleNamespace
    from urllib.parse import quote

    from ouroboros.artifacts import copy_file_to_task_artifacts, registered_task_artifact
    from ouroboros.owner_quiz import question_task_owned_by
    from ouroboros.task_custody import task_artifact_stores
    from ouroboros.task_results import load_task_result, validate_task_id

    if not isinstance(materials, list):
        raise ValueError("materials must be a list of existing {task_id, name} artifact references")
    selected = []
    for ref in materials:
        if not isinstance(ref, dict) or set(ref) - {"task_id", "name"}:
            raise ValueError("materials accepts only task_id and name, never an arbitrary path")
        source_id = validate_task_id(str(ref.get("task_id") or task_id))
        name = str(ref.get("name") or "")
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("material name must be an exact registered artifact filename")
        row = load_task_result(root, source_id) or {}
        permitted = question_task_owned_by(root, source_id, task_id)
        seen, ancestor = set(), row
        while not permitted and ancestor:
            parent = str(ancestor.get("parent_task_id") or "")
            if not parent or parent in seen:
                break
            seen.add(parent)
            permitted = question_task_owned_by(root, parent, task_id)
            ancestor = load_task_result(root, parent) or {}
        if not permitted:
            raise ValueError("material task is outside your own, descendant or recorded Continue/retry task line")
        records = [item for item in row.get("artifacts", []) if isinstance(item, dict) and item.get("name") == name]
        for drive in dict.fromkeys([root, pathlib.Path(ctx.drive_root)] if source_id == task_id else [root]):
            registered = registered_task_artifact(drive, source_id, name)
            if registered:
                records.insert(0, registered)
        stores = task_artifact_stores(root, source_id)
        if source_id == task_id:
            stores += task_artifact_stores(pathlib.Path(ctx.drive_root), source_id)
        record = next((item for item in records if item.get("path")
                       and pathlib.Path(str(item["path"])).name == name and any(
            pathlib.Path(str(item["path"])).resolve().parent == store for store in stores)), None)
        if not record or record.get("status") not in (None, "", "ready") or record.get("errors"):
            raise ValueError(f"material {source_id}/{name} is not an available registered task artifact")
        retained = copy_file_to_task_artifacts(SimpleNamespace(drive_root=root, task_id=task_id),
                                              record["path"], immutable=True, expected={key: record[key] for key in ("size", "sha256", "immutable") if key in record})
        if not retained:
            raise ValueError(f"material {source_id}/{name} could not be retained")
        selected.append({"task_id": task_id, "name": retained["name"], "label": name,
                         "size": retained["size"], "sha256": retained["sha256"],
                         "download_url": f"/api/tasks/{quote(task_id, safe='')}/artifacts/{quote(retained['name'], safe='')}"})
    return selected
