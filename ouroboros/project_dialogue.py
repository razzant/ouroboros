"""Canonical Project dialogue projections and routing annotations.

Project conversion stores a reference to the original owner row on the immutable
task binding. A Project room projects that row instead of copying it into
``chat.jsonl``. Terminal task projections append to that same canonical biography;
``chat_annotations.jsonl`` is presentation-first and owns no Project state; its
token-bound ``needs_manual_target`` decision card is the one routing-authority
exception.
"""

from __future__ import annotations

from ouroboros.room_membership import room_membership  # noqa: F401 — public compatibility surface

import hashlib
import json
import logging
import os
import pathlib
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.review_records import recorded_author_stop
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
from ouroboros import dialogue_store
from ouroboros.utils import iter_jsonl_objects, jsonl_append_lock_path, replace_atomic, strip_markdown, utc_now_iso

_ANNOTATIONS_NAME = "chat_annotations.jsonl"
# Receipt id for a routing act that belongs to NO owner message — the agent's
# own steer of a task it already routed for, or of one it was told about after
# its origin. The act still needs a durable token-bound receipt (the tool waits
# on one through ``routing_wait``, and silence there reports a landed delivery
# as unconfirmed), but it must annotate no owner message: the id is the act's
# routing token, so nothing in any chat joins to it. Chat-row membership is
# therefore the wrong retention test for these rows — compaction applies the
# chat-retention rule only to ids that address a real message, and bounds these
# by the newest-N cap below. Per-id dedupe cannot bound them: every steer mints
# a fresh token, so each act has an id of its own. The colon shape cannot
# collide with a client id: the browser mints ``msg-<epoch_ms>-<n>`` and non-web
# ingress ``host-<uuid5>``.
AGENT_RECEIPT_ID_PREFIX = "agent-steer:"
# How many synthetic receipts survive a compaction. ``routing_wait`` polls for at
# most 15 seconds, so every live waiter's row is far inside this window; without a
# cap a long-lived install would keep the file permanently above the threshold and
# rewrite the whole of it on every append.
_RETAINED_AGENT_RECEIPTS = 64
_COMPACT_AT_BYTES = 800_000
log = logging.getLogger(__name__)


def _row_chat_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("chat_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


# The closed lifecycle vocabulary of a Project question, shared with
# web/modules/question_presentation.js: one leading word answers "is there an
# unanswered question for me?", the rest is context. Both sides are pinned on the
# rows this module actually emits by web/tests/fixtures/question_presentation_parity.json.
QUESTION_STATUS = {
    "waiting": "Waiting for your answer",
    "open": "Unanswered",
    "resumed": "Unanswered · task continued",
    "expired_terminal": "Unanswered · task finished",
    "answered": "You answered",
    "answer_pending": "Answer recorded · delivery pending",
    "answer_saved": "Answer saved · the task reads it when it continues",
    "answer_unknown": "Answer recorded · delivery unknown",
    "closure_pending": "Question closed · update pending",
    "superseded": "Replaced by a newer question",
    "closed": "Question closed",
    "unknown": "Status unavailable",
}
_QUIZ_LIFECYCLE = {"open", "answered", "expired_terminal", "superseded", "closed"}


def owner_wait_projection(quiz_id: str, owner_wait: Any, block: Any) -> Dict[str, Any]:
    """Wait facts for one quiz, from evidence only: the task's ``owner_wait`` record when
    it names this quiz (``waiting`` / ``resumed`` plus the timeout reason), a record that
    moved on to another quiz (this wait is over), and the block's closed bound."""
    waiting = owner_wait if isinstance(owner_wait, dict) else {}
    block = block if isinstance(block, dict) else {}
    facts: Dict[str, Any] = {}
    if waiting.get("quiz_id") == quiz_id and waiting.get("state"):
        facts["owner_wait_state"] = str(waiting["state"])
        if waiting.get("resume_reason"):
            facts["owner_wait_resume_reason"] = str(waiting["resume_reason"])
    elif (waiting.get("quiz_id") and str(waiting.get("quiz_id")) != quiz_id
          and (block.get("wait_for_answer") is True or block.get("wait_ended_at"))):
        # Only a question the task actually waited on can have been resumed.
        facts["owner_wait_state"] = "resumed"
    if block.get("wait_ended_at"):
        facts["wait_ended_at"] = str(block["wait_ended_at"])
    return facts


def question_status(state: str, facts: Dict[str, Any], wait_for_answer: bool) -> str:
    """One status sentence; waiting needs positive wait evidence (a live record, or the
    original required flag before any record exists), never an inference from silence."""
    if state not in _QUIZ_LIFECYCLE:
        return QUESTION_STATUS["unknown"]
    if state != "open":
        return QUESTION_STATUS[state]
    wait_state = str(facts.get("owner_wait_state") or "")
    resumed = wait_state == "resumed" or bool(facts.get("wait_ended_at"))
    waiting = not resumed and (wait_state == "waiting" or (not wait_state and wait_for_answer))
    return QUESTION_STATUS["waiting" if waiting else "resumed" if resumed else "open"]


def project_question_pointer(row: Dict[str, Any], block: Any, project: Any,
                             owner_wait: Any = None) -> Optional[Dict[str, Any]]:
    """Read projection of one Project question into Main; never another ask.

    The row is complete for display: question, option labels and details, stake, the recorded
    answer and the wait facts ride with the pointer, so the browser paints the Project's own form
    from history, the live frame or the census alone; task detail only opens the original."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.owner_quiz import CLOSURE_FIELDS, question_delivery_projection

    quiz = row.get("quiz") if isinstance(row.get("quiz"), dict) else row
    task_id, quiz_id = str(row.get("task_id") or ""), str(quiz.get("quiz_id") or "")
    block = block if isinstance(block, dict) else {}
    project = project if isinstance(project, dict) else {}
    if not task_id or not quiz_id or not project.get("id") or not project.get("chat_id"):
        return None
    state = str(block.get("state") or "")
    known = block.get("quiz_id") == quiz_id and state in _QUIZ_LIFECYCLE
    facts = owner_wait_projection(quiz_id, owner_wait, block or quiz)
    # The block drops its required flag when its bound closes; the durable row keeps it.
    still_required = bool(block.get("wait_for_answer")) if block else bool(quiz.get("wait_for_answer"))
    name = str(project.get("name") or "Project")
    lead = question_status(state if known else "unknown", facts, still_required)
    delivery = question_delivery_projection(block)
    if known and state in {"answered", "closed"} and delivery.get("state") == "owed":
        # An accepted late answer whose dispatch witness ended with its process: no retry knows more.
        reason = delivery.get("reason_code")
        lead = QUESTION_STATUS["closure_pending" if state == "closed" else "answer_unknown"
                               if reason == "late_answer_delivery_unknown" else "answer_saved"
                               if reason == "awaiting_recipient_read" else "answer_pending"]
    options = next((value for value in (quiz.get("options"), block.get("options")) if isinstance(value, list)), [])
    labels = [str(option.get("label") if isinstance(option, dict) else option or "") for option in options]
    # Aligned details from the row's option objects or the block's own list (a legacy ask has none).
    details = ([str(option.get("detail") or "") if isinstance(option, dict) else "" for option in options]
               if any(isinstance(option, dict) for option in options) else block.get("option_details"))
    details = [str(v or "") for v in details] if isinstance(details, list) and 0 < len(labels) == len(details) else None
    question = str(quiz.get("question") or row.get("text") or block.get("question") or "")
    assumption = str(quiz.get("assumption") or block.get("assumption") or "")
    stake = str(quiz.get("stake") or block.get("stake") or "")
    host_facts = str(quiz.get("host_facts") or block.get("host_facts") or "")
    recommended = block.get("recommended_index")
    if not isinstance(recommended, int) or isinstance(recommended, bool):
        recommended = next((i for i, option in enumerate(options)
                            if isinstance(option, dict) and option.get("recommended") is True), None)
    pointer: Dict[str, Any] = {
        "role": "system", "system_type": "project_question_pointer", "task_id": task_id,
        "quiz_id": quiz_id, "quiz_state": state if known else "unknown",
        "project_id": str(project["id"]), "project_name": name,
        "project_chat_id": int(project["chat_id"]), "chat_id": WEB_UI_CHAT_ID,
        "ts": str(block.get("asked_at") or row.get("ts") or ""),
        "text": f"{lead} in {name}", "is_progress": False, "markdown": False,
        # Display fields only when known: a narrower producer must never blank a complete row.
        **({"question": question} if question else {}),
        **({"options": labels} if isinstance(quiz.get("options"), list) or isinstance(block.get("options"), list) else {}),
        **({"option_details": details} if details else {}),
        **({"stake": stake} if stake else {}),
        **({"assumption": assumption} if assumption else {}),
        **({"host_facts": host_facts} if host_facts else {}),
        **({"recommended_index": recommended} if recommended is not None else {}),
        **facts,
        **({"source_status": "unavailable"} if not known else {}),
    }
    if still_required:
        pointer["wait_for_answer"] = True
    if isinstance(block.get("answered_index"), int) and not isinstance(block.get("answered_index"), bool):
        pointer["answered_index"] = int(block["answered_index"])
    if str(block.get("comment") or ""):
        pointer["comment"] = str(block["comment"])
    for key in (*CLOSURE_FIELDS, "materials"):
        if block.get(key):
            pointer[key] = block[key]
    if delivery:
        pointer["question_delivery"] = delivery
    return pointer


def _text_sha256(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_owner_message_ref(*, chat_id: int, client_message_id: str, ts: str, text: str) -> Dict[str, Any]:
    """The canonical owner-row identity, built AT INGRESS from host-known facts.

    Identity is captured where the host writes the canonical row and passed by
    value downstream (BIBLE P2/P5); it is never re-derived from content later.
    ``text_sha256`` rides along as an integrity checksum, not a lookup key."""
    return {
        "chat_id": int(chat_id or 0),
        "client_message_id": str(client_message_id or ""),
        "ts": str(ts or ""),
        "text_sha256": _text_sha256(text),
    }


def owner_message_ref_is_valid(ref: Any) -> bool:
    """Whether a source ref has the complete host-minted owner-row identity."""
    if not isinstance(ref, dict) or not {
        "chat_id", "client_message_id", "ts", "text_sha256",
    }.issubset(ref):
        return False
    digest = ref.get("text_sha256")
    return bool(
        isinstance(ref.get("chat_id"), int)
        and not isinstance(ref.get("chat_id"), bool)
        and isinstance(ref.get("client_message_id"), str)
        and isinstance(ref.get("ts"), str) and ref.get("ts")
        and isinstance(digest, str) and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def source_refs_for_project(drive_root: Any, project_chat_id: int, *, bindings=None) -> List[Dict[str, Any]]:
    """Canonical owner-row references held by bindings for one Project lens."""
    from ouroboros.projects_registry import project_task_bindings

    refs: List[Dict[str, Any]] = []
    for row in (project_task_bindings(drive_root) if bindings is None else bindings).values():
        if not isinstance(row, dict):
            continue  # raw ``bindings`` keep malformed rows; project_task_bindings drops them too
        try:
            same_chat = int(row.get("project_chat_id") or 0) == int(project_chat_id or 0)
        except (TypeError, ValueError):
            same_chat = False
        ref = row.get("source_ref")
        if same_chat and isinstance(ref, dict) and ref:
            refs.append(dict(ref))
    return refs


def project_origin_rows(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Origin rows a Project lens can SYNTHESIZE when the canonical row is gone.

    Only bindings that carry ``source_text`` qualify (cross-thread origins — the
    binding is the retention-proof copy of the message that started the project),
    with the attachment refs, ``source`` and placeholder mark recorded from its row.
    Deduplicated by complete origin identity so several bindings created from one
    owner message yield one row."""
    from ouroboros.projects_registry import project_task_bindings

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for row in project_task_bindings(drive_root).values():
        try:
            same_chat = int(row.get("project_chat_id") or 0) == int(project_chat_id or 0)
        except (TypeError, ValueError):
            same_chat = False
        ref = row.get("source_ref")
        text = row.get("source_text")
        if not (same_chat and isinstance(ref, dict) and ref and isinstance(text, str) and text):
            continue
        identity = (
            str(ref.get("chat_id") or ""),
            str(ref.get("client_message_id") or ""),
            str(ref.get("ts") or ""),
            str(ref.get("text_sha256") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        media = row.get("source_attachments")
        rows.append({"ref": dict(ref), "text": text, "origin_id": project_origin_identity(ref), **({
            "attachments": media, "channel": str(row.get("source_channel") or ""),
            "text_placeholder": row.get("source_text_placeholder") is True,
        } if isinstance(media, list) and media else {})})
    return rows


def bound_room_chat(rooms: Callable[..., int], row: Dict[str, Any]) -> int:
    """The Project room a row's task lives in by the one room rule (``rooms`` is
    ``projects_registry.RoomResolver``), 0 when it lives in none: a row that
    carries only its task id resolves through the task's own recorded lineage."""
    return int(rooms(row.get("task_id"), row.get("parent_task_id"), row.get("root_task_id")) or 0)



def _source_ref_identity(ref: Dict[str, Any]) -> Optional[tuple]:
    try:
        chat_id = int(ref.get("chat_id") or 0)
    except (TypeError, ValueError):
        return None
    return (
        chat_id, str(ref.get("client_message_id") or ""), str(ref.get("ts") or ""),
        str(ref.get("text_sha256") or ""),
    )


def _entry_source_identities(entry: Dict[str, Any]) -> set:
    if str(entry.get("direction") or "") != "in":
        return set()
    try:
        chat_id = int(entry.get("chat_id", 1))
    except (TypeError, ValueError):
        chat_id = 1
    client_id = str(entry.get("client_message_id") or "")
    ts = str(entry.get("ts") or "")
    text_hash = _text_sha256(entry.get("text"))
    return {
        (chat_id, client_id, ts, text_hash), (chat_id, "", ts, text_hash),
        (chat_id, client_id, "", text_hash), (chat_id, "", "", text_hash),
    }


def entry_matches_source_ref(entry: Dict[str, Any], refs: Iterable[Dict[str, Any]]) -> bool:
    """Whether ``entry`` is the original row identified by one binding ref."""
    ref_keys = {
        key for ref in refs if isinstance(ref, dict)
        if (key := _source_ref_identity(ref)) is not None
    }
    return bool(_entry_source_identities(entry) & ref_keys)


def project_origin_identity(ref: dict) -> str:
    """Same binding identity on retained context and its eventual physical row."""
    return _text_sha256(json.dumps(_source_ref_identity(ref), separators=(",", ":")))


def matching_project_origin(entry: dict, refs: list) -> str:
    keys = _entry_source_identities(entry)
    return next((project_origin_identity(ref) for ref in refs if _source_ref_identity(ref) in keys), "")


def resolve_owner_message_source(drive_root: Any, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the owner source by its stored identity, checking its original text hash."""
    from ouroboros import dialogue_store as ds

    ref_key = _source_ref_identity(ref)
    if ref_key is None:
        return None
    chat_id, client_id, ts, _digest = ref_key
    if client_id:
        candidates = ds.iter_probes(drive_root, "chat", [("client_message_id", client_id)], newest_first=True)
    elif ts:
        candidates = ds.rows_by_identity(drive_root, chat_id, ts)
        if chat_id == 1:  # the legacy owner-source convention for a missing/invalid chat id
            candidates += ds.rows_by_identity(drive_root, None, ts)
        candidates = sorted(candidates, key=lambda entry: entry.seq, reverse=True)
    else:
        # Pre-identity bindings named only a chat and a text checksum.
        keys = [("chat_id", chat_id)] + ([("chat_id", None)] if chat_id == 1 else [])
        candidates = ds.iter_probes(drive_root, "chat", keys, newest_first=True)
    for entry in candidates:
        if ref_key in _entry_source_identities(entry.row):
            return dict(entry.row)
    ds.require_imported(drive_root, "chat")
    return None


def _latest_annotations(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl_objects(path):
        message_id = str(row.get("client_message_id") or "")
        if message_id and row.get("type") == "chat_annotation":
            latest[message_id] = dict(row)
    return latest


def _latest_annotations_by_token(path: pathlib.Path) -> Dict[tuple, Dict[str, Any]]:
    """Latest row per (message, routing token): one owner message carries several
    routing acts -- a first promote, a later steer relaying the same message, a
    picker click -- and each act's receipt must stay readable by its own token
    while the message's LATEST row is what the UI paints."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for row in iter_jsonl_objects(path):
        message_id = str(row.get("client_message_id") or "")
        if message_id and row.get("type") == "chat_annotation":
            latest[(message_id, str(row.get("routing_token") or ""))] = dict(row)
    return latest


def latest_chat_annotations(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Latest presentation annotation per message; a torn tail is ignored."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    return _latest_annotations(path)


def chat_annotation_receipt(drive_root: Any, client_message_id: str, routing_token: str) -> Dict[str, Any]:
    """The latest annotation written for one routing attempt, read BY TOKEN.

    A later act under the same owner message no longer hides an earlier act's
    receipt: the waiter that minted the token polls for its own outcome, and the
    UI projection stays latest-per-message (``latest_chat_annotations``)."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    row = _latest_annotations_by_token(path).get(
        (str(client_message_id or ""), str(routing_token or "")),
    )
    return dict(row) if row else {}


def _compact_annotations_locked(drive_root: Any, path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size < _COMPACT_AT_BYTES:
        return
    from ouroboros import dialogue_store as ds

    # Incomplete history cannot prove an annotation's source is absent. Leave the
    # sidecar intact until its retained-message decisions can be made truthfully;
    # over an unreadable line (a gap) no source is proven absent, but the newest
    # act per message still replaces the older ones.
    coverage = ds.range_coverage(drive_root, "chat")
    if not coverage["imported"]:
        return
    latest = _latest_annotations(path)
    kept_receipts = set(sorted(
        (message_id for message_id in latest if message_id.startswith(AGENT_RECEIPT_ID_PREFIX)),
        key=lambda message_id: str(latest[message_id].get("ts") or ""),
    )[-_RETAINED_AGENT_RECEIPTS:])
    retained_ids = {message_id for message_id in latest if not message_id.startswith(AGENT_RECEIPT_ID_PREFIX)
                    and (coverage["gaps"] or ds.rows_by_client_id(drive_root, message_id, limit=1))}
    # Retained per (message, token): an older act's receipt under a message that
    # is still in the chat survives beside the message's newest act.
    rows = [
        row for (message_id, _token), row in _latest_annotations_by_token(path).items()
        if message_id in retained_ids or message_id in kept_receipts
    ]
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    replace_atomic(tmp, path)


# One owner-routing receipt's fields (#968): its row below, its live frame (`send_routing_ack`) and its history
# replay state this one set; the destination part names the room and the works its words reached.
ROUTING_DESTINATION_FIELDS = ("project_id", "project_chat_id", "project_name", "target_task_ids", "target_titles")
ROUTING_RECEIPT_FIELDS = ("action", "target", "target_label", "status", "routing_token", "cause", "options",
                          "attachment_manifest", "reasoning_effort", *ROUTING_DESTINATION_FIELDS)


def append_chat_annotation(
    drive_root: Any,
    client_message_id: str,
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    project_id: str = "",
    project_chat_id: int = 0,
    status: str,
    routing_token: str = "",
    reason: str = "",
    detail: str = "",
    cause: str = "",
    options: Any = None,
    attachment_manifest: Any = None,
    require_latest_status: Any = None,
    require_latest_token: Any = None,
    reasoning_effort: str = "",
    project_name: str = "", target_task_ids: Any = None, target_titles: Any = None,
) -> bool:
    """Append one compact UI annotation.

    Presentation-first with ONE named exception (#198): a routing refusal row
    (status=needs_manual_target) is also the picker's durable decision-card
    authority — its token+options validate the owner's click, and the
    dispatch_pending/closing rows carry the click's first-wins/idempotency
    facts. Routing STATE still lives in the supervisor receipts (task-result
    admission, mailbox); the sidecar only arbitrates the card.

    ``require_latest_status`` (a set of status strings) turns the append into
    a compare-and-append under the annotations lock: the row is written only
    while the message's CURRENT latest status is in the set — the first-wins
    claim seam of the routing picker (#198). Absent/None keeps plain append.

    """
    message_id = str(client_message_id or "").strip()
    if not message_id:
        return False
    row = {
        "ts": utc_now_iso(),
        "type": "chat_annotation",
        "client_message_id": message_id[:200],
        "action": str(action or "")[:80],
        "target": str(target or "")[:200],
        "status": str(status or "")[:80],
    }
    if target_task_ids is not None:
        row.update(project_name=project_name, target_task_ids=list(target_task_ids), target_titles=list(target_titles or []))
    if str(target_label or ""):
        row["target_label"] = str(target_label)[:200]
    if project_id and project_chat_id:
        row.update(project_id=str(project_id), project_chat_id=int(project_chat_id))
    if str(routing_token or ""):
        row["routing_token"] = str(routing_token)[:128]
    if str(reason or ""):
        row["reason"] = str(reason)[:200]
    if str(detail or ""):
        row["detail"] = str(detail)[:1000]
    if str(cause or ""):
        # Q3=A: the host-owned owner-facing sentence for a refused act; the
        # browser renders it verbatim on the receipt line (reason stays a code).
        row["cause"] = str(cause)[:200]
    if isinstance(options, list):
        row["options"] = [dict(item) for item in options[:100] if isinstance(item, dict)]
    if isinstance(attachment_manifest, list):
        row["attachment_manifest"] = [
            dict(item) for item in attachment_manifest if isinstance(item, dict)
        ]
    if str(reasoning_effort or ""):
        row["reasoning_effort"] = str(reasoning_effort)[:16]
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0, owner_aware_stale=True)
    if lock_fd is None:
        return False
    try:
        if require_latest_status is not None or require_latest_token is not None:
            latest = _latest_annotations(path).get(message_id)
            if latest is not None:
                latest_status = str(latest.get("status") or "")
                latest_token = str(latest.get("routing_token") or "")
                if require_latest_status is not None and latest_status not in set(require_latest_status):
                    return False  # lost the claim race — the caller reads the truth back
                if require_latest_token is not None and latest_token not in set(require_latest_token):
                    return False  # a NEWER routing attempt owns the card now
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        _compact_annotations_locked(drive_root, path)
        return True
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def routing_target_label(
    drive_root: Any, action: str, target: str, *, task: Any = None,
    project_id: str = "",
) -> str:
    """Resolve one deterministic event-time label for an existing raw target."""
    target = str(target or "").strip()
    if not target:
        return ""
    try:
        from ouroboros.project_facts import project_display_name
        from ouroboros.projects_registry import get_reserved_project, task_presentation_snapshot

        if action == "project_route":
            project = get_reserved_project(drive_root, target) or {}
            return project_display_name(project.get("name"), target) or "Project"
        return task_presentation_snapshot(
            drive_root, target, task=task, project_id=project_id,
        )["target_label"]
    except Exception:
        log.debug("Routing target label resolution failed for %s", target, exc_info=True)
        return "Task"


# Q3=A: the HOST owns the owner-facing sentence for a REFUSED routing act. One
# factual PHRASE per typed reason (what the producer branch observed: ≤ 60
# chars, lower-case start, no codes, no trailing period); the prefix comes from
# the ACT and its outcome in routing_refusal_cause, so one reason reads
# "Not started: …" on a promote and "Not moved: …" on a scope bind. The receipt
# under the owner's message, the host-initiated System row and the picker's
# 409 toast all read this table; the browser renders the sentence verbatim (no
# client table). Tool results to the agent read the same table: they keep the
# machine prefix and code, then append these words. A reason without a row stays raw ("Not started (<reason>)")
# so a new refusal is visible before it has words.
ROUTING_REFUSAL_CAUSES: Dict[str, str] = {
    "workspace_unusable": "the working folder can't be used",
    "workspace_provisioning_failed": "no working folder could be created for the project",
    "worker_pool_unavailable": "no worker is available right now",
    "worker_pool_state_unavailable": "the worker pool could not be checked",
    "duplicate_task_id": "this task already exists",
    "admission_reservation_owned": "another request already owns this task id",
    "admission_reservation_lost": "the task lost its place in the queue",
    "admission_reservation_failed": "the task could not be admitted",
    "admission_fence": "the task could not be admitted",
    "admission_rejected": "the task could not be admitted",
    "invalid_admission_reservation": "the task id or its token was missing",
    "task_id_lookup_failed": "the task record could not be read",
    "empty_objective": "the request was empty",
    "project_routing_fence": "the project no longer accepts new work",
    "project_routing_fence_changed": "the project changed while the task was being prepared",
    "project_routing_fence_lookup_failed": "the project state could not be checked",
    "project_binding_failed": "the project could not be set up",
    "project_registration_failed": "the project could not be set up",
    "ensure_project_scope_failed": "the project could not be set up",
    "project_source_error": "the project folder could not be attached",
    "attachment_admission_rejected": "the attachments could not be staged",
    "staging_unavailable": "the attachments could not be staged",
    "queue_snapshot_persist_unavailable": "the task queue could not be saved",
    "queue_snapshot_persist_failed": "the task queue could not be saved",
    "invalid_skill_repair_constraint": "the skill repair request was invalid",
    "skill_repair_payload_missing": "the skill's files are missing",
    "skill_repair_payload_unreadable": "the skill's files could not be read",
    "skill_repair_admission_unwritable": "the skill repair request could not be recorded",
    "repair_promotion_failed": "the skill repair request could not be started",
    "task_acceptance_fence": "the task tree is closing after its author stopped it",
    "invalid_task_depth": "the task depth was invalid",
    "promotion_persistence_failed": "the task record could not be saved",
    "routing_receipt_persist_failed": "the receipt could not be saved",
    "routing_annotation_persist_failed": "the receipt could not be saved",
    "source_continuation_publish_failed": "the source hand-off could not be published",
    "confirmation_timeout": "no confirmation arrived in time",
    "target_unknown": "that task is no longer running",
    "direct_chat_turn": "that reply has already been given",
    "subagent_target": "that task is a helper of another task",
    "cancel_pending": "that task is being stopped",
    "target_closed": "that task has already finished",
    "target_finished": "that task has already finished",
    "acceptance_fence_sealed": "that task has already finished",
    "mailbox_write_failed": "the message could not be saved",
    "project_scope_conflict": "the task already belongs to another project",
    "missing_task_or_project": "no task or project was named",
    "project_unavailable": "the project is no longer available",
    # Historical routing-card refusal codes, retained for receipt replay.
    "target_unspecified": "no destination was chosen",
    "invalid_project_id": "that project id is not valid",
    "target_not_found": "that project does not exist",
}

# Rows that stand alone in the chat that issued them, whatever the named task's room: host
# routing refusals and a relay's answer (also its work's speech: WORK_SPEECH_ELSEWHERE_TYPES).
# Never terminal task facts.
ORIGIN_ADDRESSED_NOTICE_TYPES = frozenset({"task_not_started", "task_start_unconfirmed", "steer_not_delivered", "relay_reply"})

# A root's deliberate notice to Main (send_user_message destination="main"); never a task's final answer.
MAIN_NOTICE_TYPE = "main_notice"
# Host rows about an act in Main, pinned there whatever the task's Project binding.
MAIN_PINNED_ROW_TYPES = frozenset({"project_started", "project_handoff", "project_completion_summary"})
# Model-authored work speech written to a fixed address (Main, a relay's source room): one row,
# also a member of its work's room (``room_membership.row_rooms``), so a Project keeps its whole
# conversation. Presence private contact stays Main-only.
WORK_SPEECH_ELSEWHERE_TYPES = frozenset({MAIN_NOTICE_TYPE, "relay_reply"})
# Rows whose written address is the producer's, never their task's room. The bus write seam
# (``message_bus.send_with_budget``) and the supervisor send handler read this one set;
# membership is ``room_membership.row_rooms``.
ROOM_FIXED_ROW_TYPES = (MAIN_PINNED_ROW_TYPES | ORIGIN_ADDRESSED_NOTICE_TYPES | WORK_SPEECH_ELSEWHERE_TYPES
                        | frozenset({"project_question_pointer"}))

# Statuses of an act that LANDED (or is still in flight): no cause sentence.
_LANDED_ROUTING_STATUSES = frozenset({"scheduled", "delivered", "read", "pending", "dispatch_pending", "accepted"})


def routing_refusal_cause(action: str, status: str, reason: str, options: Any = None) -> str:
    """The owner-facing sentence for one routing receipt; "" when the act landed
    or when the picker keeps «Choose a target» (a refusal WITH options).

    The prefix states only what the act's outcome proves: an UNCONFIRMED act
    reads "Not confirmed" whatever it was; a refused steer "Not delivered"; a
    refused scope bind "Not moved"; every other refused act (promote, route,
    skill repair) "Not started". An unconfirmed act with an unknown reason
    reads as the honest "may or may not have started"; any other unknown reason
    stays raw (``Not started (<reason>)`` — DESIGN sanctions raw over invented)."""
    status_text = str(status or "").strip()
    if status_text in _LANDED_ROUTING_STATUSES:
        return ""
    if status_text == "needs_manual_target" and isinstance(options, list) and options:
        return ""
    action_text = str(action or "").strip()
    if status_text == "unconfirmed":
        prefix = "Not confirmed"
    elif action_text == "steer_task":
        prefix = "Not delivered"
    elif action_text == "ensure_project_scope":
        prefix = "Not moved"
    else:
        prefix = "Not started"
    reason_text = str(reason or "").strip()
    phrase = ROUTING_REFUSAL_CAUSES.get(reason_text, "")
    if phrase:
        return f"{prefix}: {phrase}"
    if status_text == "unconfirmed":
        return "Not confirmed: the task may or may not have started"
    return f"{prefix} ({reason_text})" if reason_text else prefix


def routing_options_with_labels(drive_root: Any, options: Any) -> List[Dict[str, Any]]:
    """Stamp human labels on manual task choices while retaining their raw ids."""
    rows: List[Dict[str, Any]] = []
    for raw in list(options or [])[:100]:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        task_id = str(row.get("task_id") or "").strip()
        if task_id and not str(row.get("label") or "").strip():
            row["label"] = routing_target_label(
                drive_root, str(row.get("action") or "steer_task"), task_id,
                task=row, project_id=str(row.get("project_id") or ""),
            )
        rows.append(row)
    return rows


def routing_option_label(option: Any) -> str:
    """One human label per manual-routing option — the HOST SSOT (the durable
    routing_options history row and the Telegram skill both render through it;
    web mirrors it as chat_activity.routingOptionLabel)."""
    if not isinstance(option, dict):
        return ""
    if str(option.get("label") or "").strip():
        return str(option["label"]).strip()
    if str(option.get("action") or "") == "new_task_in_project":
        return f"New task in {str(option.get('project_name') or 'Project')}"
    if option.get("title") or option.get("project_name"):
        return str(option.get("title") or option.get("project_name"))
    return "Project" if option.get("project_id") and not option.get("task_id") else "Task"


OUTCOME_PHASE_HEADLINE = {"working": "Working", "done": "Done", "warn": "Done with warnings",
                          "error": "Failed", "cancelled": "Stopped"}
# A run that chose to finish without a message (catalog code task.headline.quiet).
QUIET_RUN_HEADLINE = "Done · no message"

# One sentence per typed cause, not status × reason. Host text keeps unknown codes;
# details-bearing views pair a neutral sentence with the exact codes, never a guess.
# Browser twin: task_reason_words.TASK_CAUSE_PHRASES; outcome_phase_parity.json pins both.
TASK_CAUSE_PHRASES = {
    # Reasons of a task-acceptance decision stored before the result review. A clean accepted decision renders no
    # clause (clean_pass, clean_pass_obligations_closed carry no sentence); one with a sentence states its cause.
    "previous_revision_accepted": "The reviewers approved an earlier version of this answer; the current version was not re-reviewed.",
    "admission_close_unconfirmed": "Reviewers approved this answer; the supervisor did not confirm that task admission was closed.",
    "author_stop": "Ouroboros stopped with unfinished work; no review approval was granted.",
    "review_outcome_received": "Ouroboros received the reviewers' outcome and finished on that.",
    "author_finish": "Ouroboros delivered this answer on its own judgement; the reviewers had not signed it off.",
    "review_degraded": "The reviewers did not reach a verdict on this answer.",
    "infra_failure": "The review could not run because of an infrastructure failure, so there is no verdict.",
    "acceptance_preparation_failed": "Ouroboros could not assemble the evidence for this answer's review, so this preparation attempt dispatched no new reviewers; the work itself is kept.",
    "dialogue_terminal": "The reviewers and Ouroboros could not agree, and both positions were kept.",
    "improvement_capsule": "The reviewers asked for one more pass and Ouroboros was given their notes.",
    "fence_reopen_failed": "The requested extra pass could not be started, so the answer stands as it was.",
    "review_cycles_exhausted": "The task used up its review rounds before the answer was signed off.",
    "open_obligations": "The answer was delivered with reviewer requests still open.",
    "improvement_window_closed": "There was no room left for another pass, so the answer stands as it was.",
    "capsule_spent": "The one allowed improvement pass was already used.",
    "reviewer_fail_no_capsule": "A reviewer rejected the answer and suggested nothing to change.",
    "no_actionable_changes": "The re-review was not clean and suggested nothing to change.",
    "identical_acceptance_refused": "Nothing had changed since the last review, so the recorded verdict stands.",
    "review_skipped_deadline_reserve": "There was not enough time left to review the answer.",
    "delivery_binding_superseded": "The answer or its evidence changed, so the earlier review no longer covered it.",
    "owner_followup": "A new message from you arrived, so the review was set aside for it.",
    "evidence_refresh": "The work changed after the review was frozen, so it no longer covered the answer.",
    "revision_unavailable_on_forced_rail": "The task had to stop, so the requested rework never happened.",
    "owner_hurry": "You asked me to hurry, so no further review was started.",
    "unspecified": "The answer was not signed off, and no cause was recorded.",
    # The rail that ended the task before an owed acceptance panel could run.
    "acceptance_bypassed_budget_exhausted": "The task ran out of budget before the answer could be reviewed.",
    "acceptance_bypassed_round_limit": "The task hit its round limit before the answer could be reviewed.",
    "acceptance_bypassed_deadline": "The task ran out of time before the answer could be reviewed.",
    "acceptance_bypassed_provider_unavailable": "The model provider was unavailable, so the answer was never reviewed.",
    "acceptance_bypassed_context_overflow": "The task outgrew its context before the answer could be reviewed.",
    "acceptance_bypassed_children_unabsorbed": "Some sub-tasks had not been folded in, so the answer was never reviewed.",
    # Execution reason codes, carried verbatim from the card's own old table.
    # The plan review's outcome CLASS at delivery (outcome_axes.execution.plan_review,
    # review_projection): one sentence per class; plan_review_advisory serves a
    # legacy row that recorded the open review without a class.
    "plan_review_unanswered": "Only some of the plan reviewers answered; the work went on with their notes.",
    "plan_review_none_answered": "None of the plan reviewers answered; the work went on without their notes.",
    "plan_review_answered_open": "The plan reviewers answered, but the review was never closed; the work went on with their notes.",
    "plan_review_advisory": "The plan review was never closed; the work went on with what the reviewers said.",
    "plan_review_awaiting": "Not every plan reviewer had answered when the task ended.",
    "plan_review_quorum_unreachable": "Too few plan reviewers could answer, so the work was held.",
    "plan_review_objections_open": "Objections to the plan were still open when this answer was delivered; the work went on with them disclosed.",
    "host_child_status_suffix": "A child task had not settled when the answer was delivered",
    "invalid_delivery_control_after_repair": "Ouroboros's final delivery instruction could not be read even after repair, so the answer stands as delivered.",
    "budget_exhausted": "The task ran out of budget before it could finish cleanly",
    "billing_authority_unreadable": "Ouroboros could not read this task's billing authority, so it admitted no further paid call; budget exhaustion was not established",
    "control_authority_unreadable": "Ouroboros could not read a control this task needs before a call, so it admitted none; budget exhaustion was not established",
    "billing_authority_refused": "This task's billing lineage was missing or not its own, so it admitted no further paid call; budget exhaustion was not established",
    # The other forced-finalization rails (outcomes.BEST_EFFORT_REASON_CODES, ACCEPTANCE_BYPASS_REASON_BY_RAIL keys):
    # the loop's typed reason_code when a limit ended it; each names only its limit, the status word says the rest.
    "round_limit": "The task hit its round limit before it could finish cleanly",
    "finalization_grace": "The task hit a time limit and had to wrap up before it could finish cleanly",
    "deadline_local": "The task reached its deadline before it could finish cleanly",
    "context_overflow": "The task outgrew its context before it could finish cleanly",
    "output_recovery_exhausted": "Empty length-limited replies exhausted recovery; saved work remains available",
    "children_unabsorbed": "Some sub-task results were never folded in, so the task had to wrap up",
    # The supervisor's timeout rails (queue_timeouts.TIMEOUT_TERMINAL_REASONS): the
    # reaper's task_done reason_code, spoken on its grace toast, its kill notice
    # and its salvage line through this same table, never as the code.
    "absolute_ceiling": "The task reached its maximum running time",
    "deadline": "The task reached its deadline",
    "idle_timeout": "The task made no progress for too long",
    # The reason codes outcomes.derive_loop_outcome stamps from typed terminal facts.
    "provider_failure": "The model provider failed to answer, so the task could not finish",
    "empty_final_text": "The task ended without a final answer",
    "deep_self_review_unavailable": "The deep self-review could not run",
    "deep_self_review_error": "The deep self-review stopped on an error",
    # #869: the provider-death rail's terminal words; the amount of retained text is
    # said by the notice, this clause only names why the task ended.
    "provider_unavailable": "The task could not get a usable model answer, so it could not finish",
    "accounting_wait_expired": "Accounting access did not recover within this turn’s wait window",
    "delivery_control_degraded": "Ouroboros's final delivery instruction could not be applied, so the answer stands as delivered.",
    "authoring_handover_incomplete": "The replacement model stopped before resuming tool work.",
    "delegated_custody_unreconciled": "Some delegated work was never reconciled.",
    # The one non-reason-code key: the task-result FIELD terminal_plan_review_open
    # (task_finalization.terminal_result_fields), a standing limitation of the
    # answer rather than the thing that ended the task.
    "terminal_plan_review_open": "The plan review was still open when this answer was delivered",
    "child_results_deferred": "Some sub-task results were deferred instead of being folded into this answer",
    "tool_failure": "A tool this task used failed and nothing recovered it",
    "receipt_absent": "No host verification receipt was recorded; this does not establish whether other checks ran.",
    "expected_output_ungrounded": "The declared output has no recorded tool work or structured final answer to support it.",
    "residual_tool_errors_without_review": "Tool errors remain in the record and the result has not been evaluated.",
    "artifacts_unavailable": "Some result files are missing or could not be preserved.",
    "partial_result": "The result is partial; remaining work is described in task details.",
    "task_exception": "The task stopped on an internal error",
    # Continue's technical endings: worker/reaper/recovery and model-call failures.
    "llm_api_error": "A model call ended in an error, so the task could not finish",
    "provider_rejected_tool_dialect": "The model provider rejected the tool format of this task's requests, so the task could not finish",
    "execution_deadline": "The task's execution time limit ran out",
    "reaper_wedged_worker_alive": "The task timed out and its worker could not be confirmed stopped",
    "workers_unavailable": "The worker pool was unavailable, so the task was not scheduled",
    "worker_pool_unavailable": "No worker was available to run the task",
    "worker_crash_signal": "The worker process running the task crashed",
    "worker_crash_retry_exhausted": "The worker process running the task kept crashing until its retries ran out",
    "worker_crash_owner_wait": "The worker process died while the task was waiting; completed actions were not retried",
    "worker_crash_budget_pausing": "The worker process died while a continuation was being saved; no exact continuation was available",
    "orphaned_running_after_worker_restart": "The workers restarted while the task was running, and it recorded no ending",
    "interrupted_retry_lost": "The task was interrupted, and its retry was lost when the workers restarted",
    # Recovery finalized a dead attempt from its already registered answer instead of a retry
    # (working_checkpoint.adopt_registered_answer): the answer stands, the packaging does not.
    "finalization_incomplete": "The answer was authored, but the task's finalization did not complete, so its review evidence, artifacts and cost accounting are incomplete.",
    # The Done card's delivery disposition (terminal_projection._project_row): not reason codes but the
    # row-level warning keys of a delivery the host could not confirm; terminal_delivery_clause adds the copy pointer.
    "terminal_delivery_exhausted": "Sending the final answer was not confirmed after the delivery attempts ran out",
    "terminal_delivery_unknown": "Delivery of the final answer could not be confirmed because the delivery registry could not be read; the result is in task details.",
    # Presentation only, not a producer's reason code; the exact code remains in details.
    "reason_undescribed": "The recorded reason has no description yet.",
}

def _cause_phrase(code: str, undescribed: Optional[List[str]] = None) -> str:
    """Raw host text, or a neutral sentence with exact codes collected for accessible details."""
    if not code or code in TASK_CAUSE_PHRASES or undescribed is None:
        return TASK_CAUSE_PHRASES.get(code, code)
    if code not in undescribed:
        undescribed.append(code)
    return TASK_CAUSE_PHRASES["reason_undescribed"]


def terminal_delivery_clause(state: str, preserved_path: str = "") -> str:
    """The card's one sentence about a delivery the host could not confirm (exhausted / unknown).

    An exhausted delivery names the saved copy when one exists and says so when it
    does not — never a claimed copy. The browser twin is terminalDeliveryClause.
    """
    sentence = TASK_CAUSE_PHRASES.get(f"terminal_delivery_{state}", "")
    if not sentence:
        return ""
    if state == "exhausted":
        sentence += (f"; a copy of the answer is saved at {preserved_path}." if preserved_path
                     else "; no copy of the answer could be saved.")
    return sentence


def outcome_phase(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    """The host mirror of the browser's terminality gate and severity fold.

    Durable host rows read exactly what ``log_events.js`` paints, over
    NORMALIZED axes; web/tests/fixtures/outcome_phase_parity.json pins both.
    """
    from ouroboros.outcomes import (REASON_OWNER_REQUESTED_FINALIZATION, current_custody_debt, normalize_outcome_axes,
                                    objective_warned)
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_status import FINAL_STATUSES
    from supervisor.owner_stop import owner_stop_ending

    record = {**event, **{key: value for key, value in result.items() if value not in (None, "")}}
    sources = [s.get("outcome_axes") for s in (event, result) if isinstance(s.get("outcome_axes"), dict)]
    record["outcome_axes"] = {k: v for source in sources for k, v in source.items() if isinstance(v, dict)}
    axes = normalize_outcome_axes(record)
    axis = {k: str(v.get("status") or "").lower() for k, v in axes.items() if isinstance(v, dict)}
    status = str(record.get("task_terminal_status") or record.get("status") or "").strip().lower()
    checkpoint = record.get("root_phase_checkpoint")
    synthesis = checkpoint.get("post_task_synthesis") if isinstance(checkpoint, dict) else ""
    if not (status in {"done", "cancel_requested"} or (status in FINAL_STATUSES and not (
            status == "completed" and post_task_synthesis_is_open(synthesis)))):
        return "working"
    lifecycle = axis.get("lifecycle") or status
    if lifecycle in {"cancelled", "cancel_requested"}:
        return "cancelled"
    command, ending = owner_stop_ending(record)  # Stop now is Stopped whatever it kept
    if command == "stop_now":
        return "cancelled"
    unanswered = ending == "stop"  # a Wrap up without an answer: its own failure is the stop
    author_finished = axis.get("objective") == "pass" and (axes.get("objective") or {}).get("source") == "author_acceptance"
    if (((lifecycle == "failed" or axis.get("execution") == "failed") and not unanswered)
            or axis.get("execution") == "infra_failed"
            or axis.get("objective") == "fail" or (axis.get("review") == "fail" and not author_finished)
            or {axis.get("artifacts"), str(record.get("artifact_status") or "").lower()} & {"failed", "missing"}):
        return "error"
    if unanswered:
        return "cancelled"
    if str(record.get("reason_code") or "") == REASON_OWNER_REQUESTED_FINALIZATION:
        return "done"
    debt = current_custody_debt(result, event)  # the current list, never a stamp (#1583)
    if (lifecycle == "rejected_duplicate" or objective_warned(axes.get("objective"), debt)
            or axis.get("execution") in {"degraded", "best_effort"}
            or axis.get("objective") in {"degraded", "best_effort"}
            or (axis.get("review") == "degraded" and not author_finished)):
        return "warn"
    return "done"


def completion_status_label(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    """The one owner-visible status word for a host-authored task row."""
    return OUTCOME_PHASE_HEADLINE[outcome_phase(result, event)]


def append_canonical_task_summary(drive_root: Any, row: Dict[str, Any], *, retained: bool = False) -> bool:
    """Append one task-summary row (the one chat writer seam). A root's projection row answers False
    on a busy store and its readiness appends it again (content dedup); a ``retained`` row (a child's
    terminal row, host task facts) has no such owner, so a busy store keeps it in the retry buffer."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.store import StoreError, acquisition_budget

    if not str(row.get("summary_id") or "").strip():
        return False
    if retained:
        return dialogue_store.append_quietly(pathlib.Path(drive_root), "chat", dict(row)) is not None
    try:
        with acquisition_budget(ACQUISITION_SLICE_SEC):
            return dialogue_store.append(pathlib.Path(drive_root), "chat", dict(row), idempotent=True) is not None
    except (StoreError, OSError, ValueError):
        return False


def canonical_task_summary_receipt(result: Dict[str, Any]) -> Dict[str, Any]:
    """The receipt proving this task's own terminal row reached the canonical chat.

    ``terminal_projection`` stamps it after the chat append (deduped by the
    readiness token on retry), so another composer can tell that a task already
    spoke for itself without scanning chat text (BIBLE P5). Empty when no row was
    appended for that task.
    """
    tid = str(result.get("task_id") or result.get("id") or "").strip()
    receipt = result.get("canonical_terminal_projection")
    if not tid or not isinstance(receipt, dict):
        return {}
    if str(receipt.get("summary_id") or "") != f"task-terminal:{tid}":
        return {}
    return dict(receipt)


def canonical_task_summary_reached_chat(result: Dict[str, Any], chat_id: Any) -> bool:
    """Did this task's own terminal row already reach THAT chat?

    A settled task whose row landed inline gets a receipt, so "a receipt exists" answers
    nothing: it is true of every completed, failed and cancelled child alike. A child's row
    the retry buffer lands later carries no receipt, so a composer may name that child once
    more. The question a second writer actually has is whether the reader it is about to
    address has already been told, and only the row's own chat answers that. A receipt
    written before the chat was recorded says nothing either, so it never
    silences anybody.
    """
    row_chat = canonical_task_summary_receipt(result).get("chat_id")
    if row_chat is None or chat_id is None:
        return False
    return str(row_chat) == str(chat_id)


def _narrative_result_ref_is_valid(value: Any, task_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        str(value.get("kind") or "") == "task_result"
        and str(value.get("task_id") or "") == str(task_id or "")
        and str(value.get("reader") or "") == "get_task_result"
    )


def continuation_narrative_is_valid(value: Any, task_id: str) -> bool:
    """Validate the small, authored summary persisted beside a task result."""
    if not isinstance(value, dict) or not str(value.get("text") or "").strip():
        return False
    tid = str(task_id or "").strip()
    if not tid or str(value.get("task_id") or "") != tid:
        return False
    if str(value.get("summary_kind") or "") != "authored_root_summary":
        return False
    if str(value.get("summary_id") or "") != f"task-narrative:{tid}":
        return False
    result_ref = value.get("result_ref")
    coverage = value.get("source_coverage")
    return bool(
        _narrative_result_ref_is_valid(result_ref, tid)
        and isinstance(coverage, dict)
        and _narrative_result_ref_is_valid(coverage.get("task_result"), tid)
        and coverage.get("task_result") == result_ref
    )


def resolve_legacy_continuation_narrative(
    drive_root: Any, task_id: str, result_ref: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find the newest authored row by exact task identity in the dialogue store."""
    from ouroboros import dialogue_store as ds

    tid = str(task_id or "").strip()
    expected_ref = dict(result_ref) if isinstance(result_ref, dict) else {}
    if not tid or not _narrative_result_ref_is_valid(expected_ref, tid):
        return None
    for entry in ds.iter_probes(drive_root, "chat", [("task_id", tid)], newest_first=True):
        row = entry.row
        if (
            str(row.get("type") or "") != "task_summary"
            or str(row.get("summary_kind") or "") != "authored_root_summary"
            or str(row.get("summary_id") or "") != f"task-narrative:{tid}"
            or str(row.get("task_id") or "") != tid
            or not _narrative_result_ref_is_valid(row.get("result_ref"), tid)
            or row.get("result_ref") != expected_ref
            or not isinstance(row.get("source_coverage"), dict)
            or row["source_coverage"].get("task_result") != expected_ref
            or not str(row.get("text") or "").strip()
        ):
            continue
        return {
            "text": str(row.get("text") or ""),
            "task_id": tid,
            "summary_id": f"task-narrative:{tid}",
            "summary_kind": "authored_root_summary",
            "result_ref": dict(expected_ref),
            "source_coverage": dict(row["source_coverage"]),
            "source": {
                "kind": "dialogue_row",
                "path": str(ds.path(drive_root)),
                "history_id": entry.history_id,
                "summary_id": str(row.get("summary_id") or ""),
            },
            "written_at": str(row.get("ts") or ""),
        }
    ds.require_imported(drive_root, "chat")
    return None


def _append_terminal_task_projection(
    drive_root: Any, task_id: str, task: Dict[str, Any], result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> bool:
    from ouroboros.terminal_projection import append_terminal_projection

    return append_terminal_projection(drive_root, task_id, task, task_done_event, result=result)


def historical_terminal_projection(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Small immutable lifecycle observation, never revived review/cost truth."""
    from ouroboros.task_status import SETTLED_STATUSES

    if (entry.get("type") != "task_summary"
            or entry.get("summary_kind") not in {"terminal_result_projection", "terminal_root_projection"}
            or entry.get("outcome_authority") != "canonical_task_result_after_finalization"
            or entry.get("outcome_final") is not True
            or not entry.get("task_id") or entry.get("status") not in SETTLED_STATUSES
            or entry.get("outcome_phase") not in {"done", "warn", "error", "cancelled"}):
        return None
    projection = {"status": entry["status"], "phase": entry["outcome_phase"],
                  "ts": str(entry.get("ts") or ""), "provenance": entry["outcome_authority"]}
    if isinstance(entry.get("model_execution"), dict):
        projection["model_execution"] = dict(entry["model_execution"])
    return projection


def append_terminal_task_projection(
    drive_root: Any, task_id: str, task: Dict[str, Any], result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> bool:
    """Fail-soft terminal projection; lifecycle cleanup must always continue."""
    try:
        return _append_terminal_task_projection(
            drive_root, task_id, task, result, task_done_event,
        )
    except Exception:
        log.warning("Failed to append canonical terminal projection for %s", task_id, exc_info=True)
        return False


SALVAGE_EXCERPT_LABEL = "Preserved intermediate output (not a final answer)"


def _stop_receipt_reached_chat(result: Dict[str, Any], chat_id: Any) -> bool:
    """Did the stop receipt publish these bytes into the chat THIS row targets?

    Only the successful sender records the destination: a task's admission chat
    or an enqueued receipt cannot prove delivery after lineage rebinding. Main
    keeps its excerpt unless it received that receipt itself. A legacy receipt
    without delivery evidence keeps the bytes too.
    """
    receipt = result.get("cancel_receipt")
    if not isinstance(receipt, dict) or not receipt or chat_id is None:
        return False
    lineage = receipt.get("delivered_chat_id")
    return lineage is not None and str(lineage) == str(chat_id)


def _completion_excerpt(result: Dict[str, Any], *, chat_id: Any = None,
                        salvage_only: bool = False) -> str:
    """One plain-text excerpt for BOTH lifecycle writers (event + task_summary).

    Markdown markers are stripped BEFORE whitespace flattening: the stripper's
    line-anchored heading/list patterns need the original newlines, and a
    flatten-first order would glue a ``##`` mid-line where no pattern (and no
    renderer) can treat it as markup again.

    Host-salvaged bytes are LABELLED, not hidden. They are real applied work, so
    a row that dropped them left a bare headline and a reason code over a task
    that had in fact produced something. The label says what the bytes are while
    the caller's own pointer keeps owning the untruncated copy. ``chat_id`` is
    the row's destination: only there can the stop receipt already have
    published the same text, and only there does the label stand alone.

    ``salvage_only`` is how the durable rows ask for that ONE excerpt and
    nothing else: a cut of the model's own answer is already in the room the
    row lives in, while salvaged bytes exist nowhere else.
    """
    salvaged = str(result.get("terminal_origin") or "") == TERMINAL_ORIGIN_HOST_SALVAGE
    if salvage_only and not salvaged:
        return ""
    body = ""
    for key in ("summary", "result", "error"):
        body = " ".join(strip_markdown(str(result.get(key) or "")).split())
        if body:
            break
    if not body:
        return ""
    excerpt = body if len(body) <= 240 else body[:239].rstrip() + "…"
    if not salvaged:
        return excerpt
    if _stop_receipt_reached_chat(result, chat_id):
        return f"{SALVAGE_EXCERPT_LABEL}."
    return f"{SALVAGE_EXCERPT_LABEL}: {excerpt}"


def _custody_debt_reason(reason: str, result: Dict[str, Any], event: Dict[str, Any]) -> tuple:
    """Current debt stands beside the execution cause, including a retained rail.

    The debt is the record's current list, never its reason (#1583), and healing
    rewrites only that list. Absence is not proof of reconciliation. An older row's
    custody stamp is read beneath, from its typed execution reason.
    """
    from ouroboros.outcomes import WARN_DELEGATED_CUSTODY_UNRECONCILED, current_custody_debt

    execution_reason = ""
    for source in (result, event):
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
        typed = str(execution.get("reason_code") or "")  # a stamp read back as the cause names none
        execution_reason = execution_reason or ("" if typed == WARN_DELEGATED_CUSTODY_UNRECONCILED else typed)
    return (execution_reason if reason == WARN_DELEGATED_CUSTODY_UNRECONCILED else reason,
            WARN_DELEGATED_CUSTODY_UNRECONCILED if current_custody_debt(result, event) else "")


# The open-review classes that state a standing limitation (never the merely awaited case).
PLAN_REVIEW_OPEN_CLASSES = frozenset({"plan_review_unanswered", "plan_review_none_answered", "plan_review_answered_open",
                                      "plan_review_objections_open"})


def _plan_review_key(result: Dict[str, Any], event: Dict[str, Any], fallback: str) -> str:
    """``plan_review_<class>`` when ``execution.plan_review`` names a class with a
    sentence, else the caller's fallback. The twin of ``planReviewKey``."""
    for source in (result, event):
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
        key = f"plan_review_{str(execution.get('plan_review') or '')}"
        if key in TASK_CAUSE_PHRASES:
            return key
    return fallback


def _terminal_limitations(result: Dict[str, Any], event: Dict[str, Any], reason: str,
                          *, held: bool = False) -> List[str]:
    """Standing limitations of the delivered answer, from facts already stored:
    deferred children, and a plan review still open at delivery. The plan clause
    is stated when the record carries the ``terminal_plan_review_open`` flag OR
    names an open-review class on ``execution.plan_review`` (the class rides the
    live event and the replayed row where the result-only flag does not); the
    class chooses the wording, and without one ``plan_review_advisory`` speaks
    when that is the recorded reason (so the join states it once). A HELD task
    states the hold as its primary cause and never a limitation of work that
    went on. The twin of ``terminalLimitations``."""
    deferred, flagged = False, False
    for source in (result, event):
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
        deferred = deferred or str(objective.get("deferred_count") or "0").strip() not in {"0", ""}
        flagged = flagged or source.get("terminal_plan_review_open") is True or (
            f"plan_review_{execution.get('plan_review') or ''}" in PLAN_REVIEW_OPEN_CLASSES)
    flagged = flagged and not held
    plan_key = reason if reason == "plan_review_advisory" else "terminal_plan_review_open"
    return [TASK_CAUSE_PHRASES["child_results_deferred"] if deferred else "",
            TASK_CAUSE_PHRASES[_plan_review_key(result, event, plan_key)] if flagged else ""]


def _join_cause_clauses(clauses: List[str]) -> str:
    """One line, one clause per distinct fact, separated by a middle dot.

    A clause that is not last drops its own full stop so the line reads as one
    statement rather than a row of stubs. The browser twin is joinCauseClauses.
    """
    kept = list(dict.fromkeys(clause for clause in clauses if clause))
    return " · ".join(c[:-1] if index < len(kept) - 1 and c.endswith(".") else c
                      for index, c in enumerate(kept))


def _completion_verdict(result: Dict[str, Any], event: Dict[str, Any], *, include_review_details: bool = True,
                        undescribed: Optional[List[str]] = None) -> str:
    """Every simultaneous cause this record holds, as ONE terminated host clause.

    Functional causes precede independent review caveats; neither replaces the other.
    The compact phone view leaves pure provenance in details, never rewriting axes.
    Unknown review obligations remain visible. ``undescribed`` is a details-bearing
    surface's list (``_cause_phrase``); authored prose is never touched. Python twin
    of ``taskReasonDetail`` and ``taskReasonCodes``.
    """
    from ouroboros.outcomes import (
        ACCEPTANCE_ACCEPTED, ACCEPTANCE_BYPASS_REASON_BY_RAIL, REASON_FINAL_MESSAGE, REASON_OWNER_REQUESTED_FINALIZATION,
        WARN_DELEGATED_CUSTODY_UNRECONCILED, WARN_RESIDUAL_TOOL_ERRORS_WITHOUT_REVIEW, plan_review_awaiting,
    )
    from ouroboros.tool_error_words import residual_tool_errors_phrase

    decision: Dict[str, Any] = {}
    veto: Dict[str, Any] = {}
    objective: Dict[str, Any] = {}
    completion: Dict[str, Any] = {}
    review: Dict[str, Any] = {}
    execution: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    for source in (event, result):
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else execution
        artifacts = axes.get("artifacts") if isinstance(axes.get("artifacts"), dict) else artifacts
        value = source.get("task_completion", execution.get("task_completion"))
        if isinstance(value, dict):
            completion = value
        objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else objective
        if isinstance(objective.get("receipt_veto"), dict):
            veto = objective["receipt_veto"]
        for holder in (source.get("review_status"), axes.get("review")):
            if isinstance(holder, dict):
                review = holder
                if isinstance(holder.get("acceptance_decision"), dict):
                    decision = holder["acceptance_decision"]
    status = str(decision.get("status") or "").strip()
    cause = str(decision.get("reason") or "")
    raw_reason = str(result.get("reason_code") or event.get("reason_code") or "")
    origin = result.get("cancel_origin") or event.get("cancel_origin")
    phase = outcome_phase(result, event)
    from supervisor.owner_stop import owner_stop_ending
    stopped = owner_stop_ending({"reason_code": raw_reason, "cancel_origin": origin,
                                 "outcome_axes": {"execution": execution}})[1] == "stop"
    reason, custody = _custody_debt_reason(raw_reason, result, event)
    typed = str(execution.get("reason_code") or "")
    if reason in {REASON_FINAL_MESSAGE, ""} and typed != WARN_DELEGATED_CUSTODY_UNRECONCILED:
        reason = typed or reason
    review_cause = (cause if status and (status != ACCEPTANCE_ACCEPTED or cause in TASK_CAUSE_PHRASES)
                    and (phase in {"done", "warn"} or recorded_author_stop(decision)
                         or (phase == "error" and status != ACCEPTANCE_ACCEPTED and not custody
                             and reason in {REASON_FINAL_MESSAGE, ""})) else "")
    # Author-accepted completion is not reviewer approval. Required mode, explicit
    # debts and unknown enforcement are not inferred optional from a reason string.
    trigger = str(review.get("trigger") or "").removesuffix("_after_agent_tool").removesuffix("_after_agent_advisory")
    author_closed = (objective.get("status") == "pass" and objective.get("source") == "author_acceptance"
                     and decision.get("enforcement") == "advisory"
                     and trigger in {"auto_effect", "auto_contract", "auto_agent_request"}
                     and not decision.get("open_obligations") and not review.get("acceptance_obligations"))
    if not include_review_details and (review_cause == "previous_revision_accepted"
                                      or (review_cause == "author_finish" and author_closed and review.get("status") != "fail")):
        review_cause = ""
    held = phase == "error" and str(objective.get("source") or "").startswith("plan_review_")
    if reason == "foreign_install_root" and (result.get("reason_detail") or event.get("reason_detail")):
        clause = str(result.get("reason_detail") or event["reason_detail"])
    elif raw_reason == REASON_OWNER_REQUESTED_FINALIZATION and not stopped:
        clause = review_cause = ""  # a Wrap up's answer carries its own marker
    elif (phase == "cancelled" or stopped) and isinstance(origin, dict) and origin:
        # The recorded cause and the relation the record PROVES (#1061).
        from supervisor.cancel_publication import cancel_cause_clauses

        clause = _join_cause_clauses(cancel_cause_clauses(origin, result, event))
    elif held:
        # A task HELD by a blocking plan review states the objective's own reason.
        clause = _cause_phrase(str(objective.get("reason") or ""), undescribed)
    elif reason in {REASON_FINAL_MESSAGE, ""}:
        clause = (TASK_CAUSE_PHRASES["author_stop"] if completion.get("action") == "stop" and phase != "cancelled"
                  else TASK_CAUSE_PHRASES["plan_review_awaiting"] if plan_review_awaiting(event, result) and phase == "done" else "")
    else:
        # A healed debt is never restored here: naming the code again would
        # state a debt the same record shows as empty. The recorded open-review
        # reason speaks through the wave's class when the record names one.
        detail = veto.get("detail") if veto.get("reason") == reason else ""
        key = _plan_review_key(result, event, reason) if reason == "plan_review_advisory" else reason
        clause = (" ".join(strip_markdown(str(detail)).split()) if detail
                  else _cause_phrase(key, undescribed))
    # The bypass sentence already names its rail; do not repeat the same cause.
    if review_cause and (reason == review_cause or ACCEPTANCE_BYPASS_REASON_BY_RAIL.get(reason) == review_cause):
        clause = ""
    warnings = list(objective.get("warnings") or []) if isinstance(objective.get("warnings"), list) else []
    warnings.append(objective.get("warning") or "")
    excluded = {"", "delegated_custody_unreconciled", "terminal_delivery_unknown", "terminal_delivery_exhausted"}
    if not include_review_details:
        excluded.add("receipt_absent")
    named = residual_tool_errors_phrase(execution)  # the failure the record holds, in owner words
    warning_clauses = [(str(w) == WARN_RESIDUAL_TOOL_ERRORS_WITHOUT_REVIEW and named) or TASK_CAUSE_PHRASES.get(str(w), str(w))
                       for w in warnings if str(w) not in excluded]
    record = {**event, **result}
    bundle = record.get("artifact_bundle") if isinstance(record.get("artifact_bundle"), dict) else {}
    artifact_status = bundle.get("status") or record.get("artifact_status") or artifacts.get("status")
    if artifact_status in {"failed", "missing"}:
        warning_clauses.append(TASK_CAUSE_PHRASES["artifacts_unavailable"])
    if decision.get("open_obligations"):
        warning_clauses.append(TASK_CAUSE_PHRASES["open_obligations"])
    if (objective.get("status") == "best_effort" and not clause and not objective.get("deferred_count")
            and raw_reason != REASON_OWNER_REQUESTED_FINALIZATION and review_cause in {"", "previous_revision_accepted"}):
        warning_clauses.append(TASK_CAUSE_PHRASES["partial_result"])
    remainder = (execution.get("mutation_attribution") or {}).get("uncommitted_remainder") or {}
    count = remainder.get("count")
    remainder_clause = ("Uncommitted files in the saved candidate could not be checked" if remainder.get("state") == "unknown"
                        else f"{count} uncommitted file{'s' if count != 1 else ''} remain{'s' if count == 1 else ''} in the saved candidate"
                        if remainder.get("state") == "known" and type(count) is int and count > 0 else "")
    line = _join_cause_clauses([clause, *warning_clauses, _cause_phrase(review_cause, undescribed),
                                _author_stop_rationale(decision, completion), remainder_clause,
                                *_incident_cause_clauses(decision, reason),
                                *_terminal_limitations(result, event, reason, held=held),
                                TASK_CAUSE_PHRASES.get(custody, custody) if custody else ""])
    return line if not line or line.endswith((".", "!", "?", "…", ")")) else line + "."


def _incident_cause_clauses(decision: Dict[str, Any], reason: str) -> List[str]:
    """An old record's unresolved local evidence-assembly failure states both facts: it and the rail."""
    from ouroboros.outcomes import REASON_FINAL_MESSAGE

    incident = decision.get("acceptance_incident")
    if not isinstance(incident, dict) or not incident or str(incident.get("status") or "") == "resolved":
        return []
    rail = TASK_CAUSE_PHRASES.get(reason, "") if reason not in {"", REASON_FINAL_MESSAGE} else ""
    return [TASK_CAUSE_PHRASES["acceptance_preparation_failed"], rail]


def _author_stop_rationale(decision: Dict[str, Any], completion: Optional[Dict[str, Any]] = None) -> str:
    """The agent's own reason for an explicit stop, beside the typed sentence (TZ-2 C4).

    Only the AUTHOR's recorded rationale reaches the row: the reviewer rationale
    stays in the card; a finish carries none. The twin of ``authorStopRationale``."""
    author = completion if (completion or {}).get("action") == "stop" else (
        decision.get("author_disposition") if recorded_author_stop(decision) else None)
    return " ".join(strip_markdown(str(author.get("rationale") or "")).split()) if isinstance(author, dict) else ""


def _run_lives_in_its_project(
    drive_root: Any, task_id: str, project_id: str, task: Dict[str, Any], result: Dict[str, Any],
) -> Optional[bool]:
    """Only explicit room addressing or a durable binding proves membership.

    A binding re-homes earlier rows too. Registration alone proves nothing: a
    hidden admission can predate its room, leaving that room empty of this run.
    """
    try:
        from ouroboros.projects_registry import get_reserved_project, project_binding_for_task

        chat_id = result.get("chat_id")
        if chat_id is None:
            chat_id = task.get("chat_id")
        project_chat = (get_reserved_project(drive_root, project_id, strict=True) or {}).get("chat_id")
        if chat_id is not None and project_chat is not None and int(chat_id) == int(project_chat):
            return True
        binding = project_binding_for_task(drive_root, task_id, strict=True) or {}
        return str(binding.get("project_id") or "") == str(project_id)
    except Exception:
        log.debug("project-room membership check failed for %s", task_id, exc_info=True)
        return None


# Typed answers of the Main-mirror seam (#1154): INELIGIBLE is a POSITIVE finding
# (this run owes Main nothing); UNKNOWN leaves the obligation owed for a retry.
MAIN_MIRROR_OWED, MAIN_MIRROR_INELIGIBLE, MAIN_MIRROR_UNKNOWN = "owed", "ineligible", "unknown"


def _enqueue_main_completion(drive_root: Any, tid: str, event: Dict[str, Any], *, event_queue: Any = None) -> tuple:
    from ouroboros.terminal_projection import retain_main_delivery, claim_main_delivery
    from supervisor.event_observation import delivery_outstanding
    from supervisor.terminal_delivery import enqueue_terminal_delivery, register_pending_delivery

    event = retain_main_delivery(drive_root, tid, event)
    if event.get("terminal_projection_token"):
        if event_queue is None:
            from supervisor.workers import get_event_q
            try:
                event_queue = get_event_q(create=False)
            except Exception:
                event_queue = None  # unattached daemon: cached debt still reaches boot/tick replay
        outstanding = delivery_outstanding(event_queue, event.get("delivery_id"))
        if outstanding is not False:  # queued, handler-inflight or unobservable: no copy or retry-clock spend
            if outstanding is None:
                register_pending_delivery(pathlib.Path(drive_root), event)  # boot/tick can discover unattached debt
            return (MAIN_MIRROR_OWED if outstanding else MAIN_MIRROR_UNKNOWN), False
    if not claim_main_delivery(drive_root, event):
        return MAIN_MIRROR_OWED, False
    if not register_pending_delivery(pathlib.Path(drive_root), event):
        return MAIN_MIRROR_UNKNOWN, False
    try:
        queued = bool(enqueue_terminal_delivery(drive_root, event, event_queue=event_queue))
    except Exception:
        log.warning("Main mirror remains owed after queue failure for %s", tid, exc_info=True)
        queued = False
    return MAIN_MIRROR_OWED, queued


def project_completion_delivery_outcome(
    drive_root: Any, evt: Dict[str, Any], task_id: str, task: Dict[str, Any],
    result: Dict[str, Any], task_done_event: Dict[str, Any],
    *, event_queue: Any = None, source_delivery: Optional[dict] = None,
) -> tuple:
    """Owe Main's answer for Project roots, including conversations moved from Main.

    A direct conversation born inside a Project stays there. Only its durable,
    ingress-bound source can prove that a direct turn was transferred from Main;
    current chat addressing and project existence cannot establish that origin.

    Answers ``(durability, live_send_queued)``: one of ``MAIN_MIRROR_OWED`` /
    ``MAIN_MIRROR_INELIGIBLE`` / ``MAIN_MIRROR_UNKNOWN``, resting on the OWED
    registration and never on the live queue (a registered row the queue refused
    is replayed from the outbox; a send the registry could not record is not
    available), plus the live-queue boolean. Canonical readiness retains the
    exact event, destination and retry timing until strict chat delivery returns;
    a bounded outbox eviction cannot retire it. External delivery is at-least-once.
    """
    tid = str(task_id or "").strip()
    task = task if isinstance(task, dict) else {}
    result = result if isinstance(result, dict) else {}
    if not tid:
        return MAIN_MIRROR_INELIGIBLE, False
    try:
        from ouroboros.projects_registry import mirrored_answer, project_binding_for_task, task_presentation_snapshot

        # Once publication was addressed, later Project deletion/rebinding cannot
        # invalidate the already accepted Main debt or change its exact content.
        ready = result.get("canonical_terminal_projection_ready")
        retained = ready.get("main_delivery") if isinstance(ready, dict) else None
        if isinstance(retained, dict) and isinstance(retained.get("event"), dict):
            return _enqueue_main_completion(drive_root, tid, retained["event"], event_queue=event_queue)

        if any(bool(row.get("_is_direct_chat")) for row in (evt, task, result, task_done_event)
               if isinstance(row, dict)):
            binding = project_binding_for_task(drive_root, tid, strict=True) or {}
            source = binding.get("source_ref")
            if not owner_message_ref_is_valid(source) or source["chat_id"] != 1:
                return MAIN_MIRROR_INELIGIBLE, False
        from ouroboros.task_results import resolve_task_lineage
        from ouroboros.task_status import SETTLED_STATUSES
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        lineage = resolve_task_lineage(
            tid, metadata=metadata,
            root_task_id=result.get("root_task_id") or task.get("root_task_id"),
            parent_task_id=result.get("parent_task_id") or task.get("parent_task_id"),
            delegation_role=result.get("delegation_role") or task.get("delegation_role"),
            original_task_id=result.get("original_task_id") or task.get("original_task_id"),
            timeout_retry_from=result.get("timeout_retry_from") or task.get("timeout_retry_from"),
        )
        status = str(result.get("status") or task_done_event.get("status") or "").lower()
        if not lineage["is_root_task"] or status not in SETTLED_STATUSES:
            return MAIN_MIRROR_INELIGIBLE, False
        snapshot = task_presentation_snapshot(
            drive_root, tid, task=task, result=result,
            project_id=str(result.get("project_id") or task.get("project_id") or ""),
            strict=True,
        )
        if not snapshot["project_id"] or not snapshot["project_routable"]:
            # Owner decision 3A: a run whose project id was DERIVED from a
            # workspace has no room, so Main stays silent instead of offering a
            # Project reference that lands in an empty duplicate of itself. The same
            # holds once a project is deleting or tombstoned.
            return MAIN_MIRROR_INELIGIBLE, False
        membership = _run_lives_in_its_project(drive_root, tid, snapshot["project_id"], task, result)
        if membership is None:
            return MAIN_MIRROR_UNKNOWN, False
        if not membership:
            # The room exists but holds none of this run's work: its id was only
            # registered AFTER admission, or a mid-flight bind failed fail-soft.
            # Offering "Open the Project" would reproduce the reported defect —
            # a Main row leading into an empty room.
            return MAIN_MIRROR_INELIGIBLE, False
        # Text preserves only the labelled salvage excerpt, never a cut of the authored answer (which rides whole
        # in its typed key or not at all). The Project reference is the sole door, so the text invites nothing;
        # promising the same action in prose would duplicate it. Salvaged bytes exist nowhere else in this row.
        excerpt = _completion_excerpt(result, chat_id=1, salvage_only=True)
        delivery = source_delivery or {}
        clause = terminal_delivery_clause(delivery.get("state", ""), delivery.get("preserved_path", ""))
        source_clause = f"Source Project: {clause}" if clause else ""
        verdict = _join_cause_clauses([_completion_verdict(result, task_done_event), source_clause])
        # Web details retain undescribed codes; canonical text and Telegram retain the full verdict.
        undescribed: List[str] = []
        cause = _join_cause_clauses([_completion_verdict(result, task_done_event, undescribed=undescribed), source_clause])
        detail = " ".join(part for part in (verdict, excerpt) if part)
        from ouroboros.terminal_time import terminal_time_fact
        from ouroboros.terminal_projection import completion_notice
        from ouroboros.project_completion_facts import completion_facts
        event = {
            "type": "send_message", "chat_id": 1, "task_id": tid,
            "text": (f"{snapshot['target_label']} · {completion_status_label(result, task_done_event)}"
                     + (f"\n{detail}" if detail else "")),
            "role": "system", "system_type": "project_completion_summary",
            "delivery_id": f"project-completion:{tid}",
            "progress_meta": {
                **completion_facts(drive_root, tid, task, result, task_done_event),
                "project_id": snapshot["project_id"],
                "project_name": snapshot["project_name"],
                "task_name": snapshot["task_name"],
                "target_label": snapshot["target_label"], "status": status,
                "terminal_time": terminal_time_fact(result),
                "completion_notice": {**completion_notice({**task_done_event, **result, "terminal_delivery": delivery}),
                                      "salvage": excerpt, "cause": cause, "undescribed": undescribed},
                **({"completion_reason": verdict or OUTCOME_PHASE_HEADLINE[outcome_phase(result, task_done_event)]}
                   if verdict or outcome_phase(result, task_done_event) != "done" else {}),
                **mirrored_answer(result, outcome_phase(result, task_done_event)),
            },
        }
        return _enqueue_main_completion(drive_root, tid, event, event_queue=event_queue)
    except Exception:
        log.warning("Failed to enqueue Project completion summary for %s", tid, exc_info=True)
        return MAIN_MIRROR_UNKNOWN, False


def enqueue_project_completion_summary(
    drive_root: Any, evt: Dict[str, Any], task_id: str, task: Dict[str, Any],
    result: Dict[str, Any], task_done_event: Dict[str, Any],
) -> bool:
    """Whether this call queued a live Main copy; the typed pair also reports durability."""
    return project_completion_delivery_outcome(drive_root, evt, task_id, task, result, task_done_event)[1]


def announce_project_started(
    drive_root: Any, project: Dict[str, Any], task_id: str, *, task: Any = None,
) -> bool:
    """Owe Main's one durable entry row when the AGENT starts a Project.

    Mirrors ``enqueue_project_completion_summary``'s delivery mechanics: the
    same terminal-delivery outbox, with the restart-surviving
    ``delivery_id=project-start:<project_id>`` dedupe as the ONLY dedupe.
    Called exclusively from the agent-initiated creation seams (owner decision
    2=A): the promote_chat_to_task bind and a REAL ensure_project_scope create
    (``created is True`` from ``create_project``). Owner HTTP/API creation and
    manual task-to-project conversion stay silent.
    """
    project = project if isinstance(project, dict) else {}
    pid = str(project.get("id") or "").strip()
    tid = str(task_id or "").strip()
    if not pid:
        return False
    try:
        from ouroboros.projects_registry import task_presentation_snapshot
        from supervisor.terminal_delivery import enqueue_terminal_delivery

        snapshot = task_presentation_snapshot(
            drive_root, tid, task=task if isinstance(task, dict) else None,
            project_id=pid,
        )
        event = {
            "type": "send_message", "chat_id": 1, "task_id": tid,
            "text": f"{snapshot['target_label']} · Started",
            "role": "system", "system_type": "project_started",
            "delivery_id": f"project-start:{pid}",
            "progress_meta": {
                "project_id": pid,
                "project_name": snapshot["project_name"],
                "target_label": snapshot["target_label"],
                "task_name": snapshot["task_name"],
            },
        }
        return bool(enqueue_terminal_delivery(drive_root, event))
    except Exception:
        log.warning("Failed to enqueue Project started row for %s", pid, exc_info=True)
        return False


__all__ = ["AGENT_RECEIPT_ID_PREFIX", "announce_project_started",
    "append_chat_annotation", "append_canonical_task_summary",
    "append_terminal_task_projection", "build_owner_message_ref", "chat_annotation_receipt",
    "entry_matches_source_ref", "latest_chat_annotations",
    "enqueue_project_completion_summary", "project_completion_delivery_outcome",
    "completion_status_label", "outcome_phase", "owner_message_ref_is_valid",
    "project_origin_rows", "project_question_pointer",
    "routing_options_with_labels", "routing_target_label", "resolve_owner_message_source",
    "source_refs_for_project",
]
