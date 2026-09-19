"""Canonical Project dialogue projections and routing annotations.

Project conversion stores a reference to the original owner row on the immutable
task binding. A Project room projects that row instead of copying it into
``chat.jsonl``. Terminal task projections append to that same canonical biography;
``chat_annotations.jsonl`` is presentation-first and owns no Project state; its
token-bound ``needs_manual_target`` decision card is the one routing-authority
exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
from ouroboros.utils import append_jsonl, iter_jsonl_objects, jsonl_append_lock_path, replace_atomic, strip_markdown, utc_now_iso

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
_RETAINED_ARCHIVES = 3
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
    "open": "Unanswered · an answer is still accepted",
    "resumed": "Unanswered · the task continued; an answer is still accepted",
    "expired_terminal": "Unanswered · the task finished; a late answer is accepted as your message",
    "answered": "You answered",
    "superseded": "Replaced by a newer question",
    "unknown": "Status unavailable",
}
_QUIZ_LIFECYCLE = {"open", "answered", "expired_terminal", "superseded"}


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
    options = next((value for value in (quiz.get("options"), block.get("options")) if isinstance(value, list)), [])
    labels = [str(option.get("label") if isinstance(option, dict) else option or "") for option in options]
    # Aligned details from the row's option objects or the block's own list (a legacy ask has none).
    details = ([str(option.get("detail") or "") if isinstance(option, dict) else "" for option in options]
               if any(isinstance(option, dict) for option in options) else block.get("option_details"))
    details = [str(v or "") for v in details] if isinstance(details, list) and 0 < len(labels) == len(details) else None
    question = str(quiz.get("question") or row.get("text") or block.get("question") or "")
    assumption = str(quiz.get("assumption") or block.get("assumption") or "")
    stake = str(quiz.get("stake") or block.get("stake") or "")
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
        **({"options": labels} if labels else {}),
        **({"option_details": details} if details else {}),
        **({"stake": stake} if stake else {}),
        **({"assumption": assumption} if assumption else {}),
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
    return pointer


def _chat_paths(drive_root: Any) -> List[pathlib.Path]:
    root = pathlib.Path(drive_root)
    archives = sorted(
        (root / "archive").glob("chat_*.jsonl"),
        key=lambda path: path.name,
        reverse=True,
    )[:_RETAINED_ARCHIVES]
    return [*reversed(archives), root / "logs" / "chat.jsonl"]


def _text_sha256(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_owner_message_ref(
    *,
    chat_id: int,
    client_message_id: str,
    ts: str,
    text: str,
) -> Dict[str, Any]:
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


def source_refs_for_project(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Canonical owner-row references held by bindings for one Project lens."""
    from ouroboros.projects_registry import project_task_bindings

    refs: List[Dict[str, Any]] = []
    for row in project_task_bindings(drive_root).values():
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
    binding is the retention-proof copy of the message that started the project).
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
        rows.append({"ref": dict(ref), "text": text})
    return rows


def bound_room_chat(bindings: Dict[str, int], row: Dict[str, Any]) -> int:
    """Resolve a row's immutable task binding in delivery lineage order."""
    for field in ("task_id", "parent_task_id", "root_task_id"):
        chat = bindings.get(str(row.get(field) or "").strip())
        if chat:
            return int(chat)
    return 0


def room_membership(chat_id: int, project_chat_ids: set, source_refs: list,
                    bindings: Dict[str, int]):
    """Canonical room membership shared by history and evidence readers.

    Presentation-only hiding and cross-room question pointers belong to the UI
    caller. A room source retains the actual cognitive result as well.
    """
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id

    def matches(entry_chat: int, entry: Optional[dict] = None) -> bool:
        row = entry if isinstance(entry, dict) else {}
        if is_a2a_chat_id(entry_chat):
            return False
        # A routing refusal belongs to the issuing chat, even when the target
        # is bound to another Project. Its lineage must not move the notice.
        bound = 0 if row.get("type") in ORIGIN_ADDRESSED_NOTICE_TYPES else bound_room_chat(bindings, row)
        lifecycle = row.get("type") in {"project_started", "project_completion_summary"}
        if chat_id in project_chat_ids:
            return not lifecycle and (bound == chat_id or entry_chat == chat_id
                                      or entry_matches_source_ref(row, source_refs))
        if chat_id != 1:
            return entry_chat == chat_id and not bound
        if entry_chat == HIDDEN_CHAT_ID:
            return False
        if lifecycle:
            return entry_chat not in project_chat_ids
        return entry_chat not in project_chat_ids and not bound

    return matches


def project_recent_dialogue(
    memory: Any, project_chat_id: int, max_entries: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Focused recent rows plus retention-proof cross-thread owner origins."""
    from ouroboros.projects_registry import all_task_bindings

    try:
        bound = all_task_bindings(memory.drive_root)
    except Exception:
        bound = {}
    refs = source_refs_for_project(memory.drive_root, project_chat_id)
    matches = room_membership(project_chat_id, {project_chat_id}, refs, bound)
    entries, coverage = memory.read_unconsolidated_chat(
        memory.load_dialogue_meta(), max_entries,
        predicate=lambda row: matches(_row_chat_id(row), row),
    )
    present_ref_keys = set()
    for entry in entries:
        present_ref_keys.update(_entry_source_identities(entry))
    retained: List[Dict[str, Any]] = []
    for origin in project_origin_rows(memory.drive_root, project_chat_id):
        ref = origin.get("ref") if isinstance(origin.get("ref"), dict) else {}
        if _source_ref_identity(ref) in present_ref_keys:
            continue
        retained.append({
            "chat_id": ref.get("chat_id"), "client_message_id": ref.get("client_message_id"),
            "ts": ref.get("ts"), "direction": "in", "text": origin.get("text"),
            "project_origin_projection": True,
        })
    return entries, coverage, retained


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
        chat_id = int(entry.get("chat_id", 1) or 1)
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


def resolve_owner_message_source(drive_root: Any, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Stream the exact named owner source across the durable generation chain."""
    from ouroboros.consolidator import _ordered_chat_generation_paths

    live = pathlib.Path(drive_root) / "logs" / "chat.jsonl"
    ref_key = _source_ref_identity(ref)
    if ref_key is None:
        return None
    for path in reversed(_ordered_chat_generation_paths(live)):
        try:
            for row in iter_jsonl_objects(path):
                if ref_key in _entry_source_identities(row):
                    return dict(row)
        except OSError:
            continue
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


def chat_annotation_receipt(
    drive_root: Any, client_message_id: str, routing_token: str,
) -> Dict[str, Any]:
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
    retained_ids = {
        str(row.get("client_message_id") or "")
        for chat_path in _chat_paths(drive_root)
        for row in iter_jsonl_objects(chat_path)
        if row.get("client_message_id")
    }
    latest = _latest_annotations(path)
    kept_receipts = set(sorted(
        (message_id for message_id in latest if message_id.startswith(AGENT_RECEIPT_ID_PREFIX)),
        key=lambda message_id: str(latest[message_id].get("ts") or ""),
    )[-_RETAINED_AGENT_RECEIPTS:])
    # Retained per (message, token): an older act's receipt under a message that
    # is still in the chat survives beside the message's newest act.
    rows = [
        row for (message_id, _token), row in _latest_annotations_by_token(path).items()
        if message_id in retained_ids or message_id in kept_receipts
    ]
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    replace_atomic(tmp, path)


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
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
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
        from ouroboros.projects_registry import get_reserved_project, task_presentation_snapshot

        if action == "project_route":
            project = get_reserved_project(drive_root, target) or {}
            name = str(project.get("name") or "").strip()
            return name if name and name != target else "Project"
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
# client table). A reason without a row stays raw ("Not started (<reason>)")
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
    "task_acceptance_fence": "the task tree is already being accepted",
    "invalid_task_depth": "the task depth was invalid",
    "promotion_persistence_failed": "the task record could not be saved",
    "routing_receipt_persist_failed": "the receipt could not be saved",
    "routing_annotation_persist_failed": "the receipt could not be saved",
    "source_continuation_publish_failed": "the source hand-off could not be published",
    "confirmation_timeout": "no confirmation arrived in time",
    "target_unknown": "that task is no longer running",
    "direct_chat_turn": "that reply has already been given",
    "subagent_target": "that task is a helper of another task",
    "chat_mismatch": "that task belongs to another chat",
    "cancel_pending": "that task is being stopped",
    "target_closed": "that task has already finished",
    "target_finished": "that task has already finished",
    "acceptance_fence_sealed": "that task has already finished",
    "mailbox_write_failed": "the message could not be saved",
    "project_scope_conflict": "the task already belongs to another project",
    "missing_task_or_project": "no task or project was named",
    "project_unavailable": "the project is no longer available",
    # route_to_project's typed abstention codes (control_routing._route_to_project)
    "target_unspecified": "no destination was chosen",
    "invalid_project_id": "that project id is not valid",
    "target_not_found": "that project does not exist",
}

# Host routing refusals stay in the issuing chat regardless of the target's
# Project binding (room_membership); they are never terminal task facts.
ORIGIN_ADDRESSED_NOTICE_TYPES = frozenset({"task_not_started", "task_start_unconfirmed", "steer_not_delivered"})

# Statuses of an act that LANDED (or is still in flight): no cause sentence.
_LANDED_ROUTING_STATUSES = frozenset({"scheduled", "delivered", "pending", "dispatch_pending", "accepted"})


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
                          "error": "Failed", "cancelled": "Cancelled"}

# One owner sentence per typed cause, for BOTH lifecycle writers and the card.
# Keyed on the CODE only — never on (status × reason): the status word already
# speaks, and a product of the two would be a matrix nobody maintains. A code
# with no sentence stays raw (docs/DESIGN.md "Status and chips"), and the raw
# code stays typed on the row. web/modules/log_events.js carries the twin;
# web/tests/fixtures/outcome_phase_parity.json pins both.
TASK_CAUSE_PHRASES = {
    # Acceptance-decision reasons. A clean accepted decision renders no clause,
    # so clean_pass and clean_pass_obligations_closed carry no sentence; an
    # accepted decision with a sentence here still states its cause.
    "previous_revision_accepted": "The reviewers approved the earlier version of this answer; it changed before they finished.",
    "author_stop": "Main stopped with unfinished work; no review approval was granted.",
    "review_outcome_received": "Main received the review outcome or recorded limitation.",
    "author_finish": "The answer was delivered on Main's own judgement; the reviewers had not signed it off.",
    "review_degraded": "No reviewer verdict was established for this answer.",
    "infra_failure": "A review infrastructure failure prevented a settled verdict.",
    "dialogue_terminal": "The reviewers and Main could not agree, and both positions were kept.",
    "improvement_capsule": "The reviewers asked for one more pass and Main was given their notes.",
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
    "plan_review_advisory": "Plan review never closed; the work continued under advisory enforcement",
    "host_child_status_suffix": "A child task had not settled when the answer was delivered",
    "invalid_delivery_control_after_repair": "The delivery control object was still malformed after repair",
    "budget_exhausted": "The task ran out of budget before it could finish cleanly",
    "delivery_control_degraded": "Delivery finished in a degraded control state",
    "delegated_custody_unreconciled": "Some delegated work was never reconciled.",
}


def outcome_phase(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    """The host mirror of the browser's terminality gate and severity fold.

    Durable host rows read exactly what ``log_events.js`` paints, over
    NORMALIZED axes; web/tests/fixtures/outcome_phase_parity.json pins both.
    """
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION, normalize_outcome_axes
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_status import FINAL_STATUSES

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
    author_finished = axis.get("objective") == "pass" and (axes.get("objective") or {}).get("source") == "author_acceptance"
    if (lifecycle == "failed" or axis.get("execution") in {"failed", "infra_failed"}
            or axis.get("objective") == "fail" or (axis.get("review") == "fail" and not author_finished)
            or {axis.get("artifacts"), str(record.get("artifact_status") or "").lower()} & {"failed", "missing"}):
        return "error"
    if str(record.get("reason_code") or "") == REASON_OWNER_REQUESTED_FINALIZATION:
        return "done"
    if (lifecycle == "rejected_duplicate" or bool((axes.get("objective") or {}).get("warning"))
            or axis.get("execution") in {"degraded", "best_effort"}
            or axis.get("objective") in {"degraded", "best_effort"}
            or (axis.get("review") == "degraded" and not author_finished)):
        return "warn"
    return "done"


def completion_status_label(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    """The one owner-visible status word for a host-authored task row."""
    return OUTCOME_PHASE_HEADLINE[outcome_phase(result, event)]


def append_canonical_task_summary(drive_root: Any, row: Dict[str, Any]) -> bool:
    """Append one task-summary row through the existing concurrent JSONL owner."""
    if not str(row.get("summary_id") or "").strip():
        return False
    path = pathlib.Path(drive_root) / "logs" / "chat.jsonl"
    return append_jsonl(path, dict(row))


def canonical_task_summary_receipt(result: Dict[str, Any]) -> Dict[str, Any]:
    """The receipt proving this task's own terminal row reached the canonical chat.

    ``_append_terminal_task_projection`` stamps it on the task result in the same
    write that appends the row, so another composer can tell that a task already
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

    Every settled task gets a receipt, so "a receipt exists" answers nothing: it
    is true of every completed, failed and cancelled child alike. The question a
    second writer actually has is whether the reader it is about to address has
    already been told, and only the row's own chat answers that. A receipt
    written before the chat was recorded says nothing either, so it never
    silences anybody.
    """
    row_chat = canonical_task_summary_receipt(result).get("chat_id")
    if row_chat is None or chat_id is None:
        return False
    return str(row_chat) == str(chat_id)


def append_authored_task_summary(
    canonical_root: Any, result_root: Any, row: Dict[str, Any], *, status: str = "",
) -> bool:
    """Append the authored row and persist its identical continuation narrative."""
    appended = append_canonical_task_summary(canonical_root, row)
    persist_continuation_narrative(
        result_root,
        str(row.get("task_id") or ""),
        str(row.get("text") or ""),
        summary_id=str(row.get("summary_id") or ""),
        summary_kind=str(row.get("summary_kind") or ""),
        result_ref=row.get("result_ref") if isinstance(row.get("result_ref"), dict) else {},
        source_coverage=row.get("source_coverage") if isinstance(row.get("source_coverage"), dict) else {},
        status=status,
    )
    return appended


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


def persist_continuation_narrative(
    drive_root: Any,
    task_id: str,
    text: str,
    *,
    summary_id: str,
    summary_kind: str,
    result_ref: Dict[str, Any],
    source_coverage: Dict[str, Any],
    status: str = "",
) -> bool:
    """Persist the exact authored summary through the task-result lock."""
    tid = str(task_id or "").strip()
    narrative = {
        "text": str(text or ""),
        "task_id": tid,
        "summary_id": str(summary_id or ""),
        "summary_kind": str(summary_kind or ""),
        "result_ref": dict(result_ref) if isinstance(result_ref, dict) else {},
        "source_coverage": dict(source_coverage) if isinstance(source_coverage, dict) else {},
        "written_at": utc_now_iso(),
    }
    if not tid or not continuation_narrative_is_valid(narrative, tid):
        return False
    try:
        from ouroboros.task_results import load_task_result, write_task_result

        existing = load_task_result(drive_root, tid) or {}
        if not existing and not str(status or "").strip():
            return False
        requested_status = str(status or existing.get("status") or "running")

        def _project(current: Dict[str, Any], _patch: Dict[str, Any]) -> Dict[str, Any]:
            current_narrative = current.get("continuation_narrative")
            if continuation_narrative_is_valid(current_narrative, tid):
                # The summary id is task-unique.  A second post-task worker must
                # not race a complete narrative with a partial/empty rewrite.
                return {
                    "status": str(current.get("status") or requested_status),
                    "continuation_narrative": dict(current_narrative),
                }
            return {
                "status": str(current.get("status") or requested_status),
                "continuation_narrative": dict(narrative),
            }

        write_task_result(
            drive_root, tid, requested_status, _field_projector=_project,
        )
        return True
    except Exception:
        log.warning("Failed to persist continuation narrative for %s", tid, exc_info=True)
        return False


def _bounded_chat_tail_rows(
    path: pathlib.Path, *, max_bytes: int, max_rows: int,
) -> List[Dict[str, Any]]:
    """Read only a bounded tail; never enter the unbounded archive resolver."""
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - max(1, int(max_bytes)))
            handle.seek(start)
            if start:
                handle.readline()  # discard the partial first JSONL row
            rows: List[Dict[str, Any]] = []
            for raw in handle:
                if len(rows) >= max(1, int(max_rows)):
                    break
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
            return rows
    except OSError:
        return []


def resolve_legacy_continuation_narrative(
    drive_root: Any, task_id: str, result_ref: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find one recent authored row by exact task identity, within fixed bounds."""
    from ouroboros.context_budget import (
        CONTINUATION_NARRATIVE_LEGACY_GENERATIONS,
        CONTINUATION_NARRATIVE_LEGACY_MAX_ROWS,
        CONTINUATION_NARRATIVE_LEGACY_TAIL_BYTES,
    )

    tid = str(task_id or "").strip()
    expected_ref = dict(result_ref) if isinstance(result_ref, dict) else {}
    if not tid or not _narrative_result_ref_is_valid(expected_ref, tid):
        return None
    paths = _chat_paths(drive_root)
    paths = paths[-max(1, int(CONTINUATION_NARRATIVE_LEGACY_GENERATIONS)):]
    for path in reversed(paths):
        rows = _bounded_chat_tail_rows(
            path,
            max_bytes=CONTINUATION_NARRATIVE_LEGACY_TAIL_BYTES,
            max_rows=CONTINUATION_NARRATIVE_LEGACY_MAX_ROWS,
        )
        for row in reversed(rows):
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
                    "kind": "chat_jsonl",
                    "path": str(path),
                    "summary_id": str(row.get("summary_id") or ""),
                },
                "written_at": str(row.get("ts") or ""),
            }
    return None


def _append_terminal_task_projection(
    drive_root: Any, task_id: str, task: Dict[str, Any], result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> bool:
    """Project one terminal child result into canonical cognition, without an LLM."""
    from ouroboros.task_results import resolve_task_lineage, write_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    tid = str(task_id or "").strip()
    task = task if isinstance(task, dict) else {}
    result = result if isinstance(result, dict) else {}
    event = task_done_event if isinstance(task_done_event, dict) else {}
    if not tid:
        return False
    lineage = resolve_task_lineage(
        tid,
        metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
        root_task_id=result.get("root_task_id") or task.get("root_task_id"),
        parent_task_id=result.get("parent_task_id") or task.get("parent_task_id"),
        delegation_role=result.get("delegation_role") or task.get("delegation_role"),
        original_task_id=result.get("original_task_id") or task.get("original_task_id"),
        timeout_retry_from=result.get("timeout_retry_from") or task.get("timeout_retry_from"),
    )
    status = str(result.get("status") or event.get("status") or "").strip().lower()
    if status not in SETTLED_STATUSES:
        return False
    is_root = bool(lineage["is_root_task"])
    summary_id = f"task-terminal:{tid}"
    summary_kind = "terminal_root_projection" if is_root else "terminal_result_projection"
    parent_id = str(lineage.get("parent_task_id") or "")
    root_id = str(lineage.get("root_task_id") or tid)
    from ouroboros.project_facts import resolve_project_id

    appended = False

    def _append_once(current: Dict[str, Any], _patch: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal appended
        existing_marker = current.get("canonical_terminal_projection")
        if isinstance(existing_marker, dict) and str(existing_marker.get("summary_id") or "") == summary_id:
            return {"status": str(current.get("status") or status)}
        checkpoint = current.get("root_phase_checkpoint")
        post_task_phase = (
            str(checkpoint.get("post_task_synthesis") or "")
            if isinstance(checkpoint, dict) else ""
        )
        if is_root and post_task_phase in {"pending_once", "running"}:
            ready = current.get("canonical_terminal_projection_ready")
            if isinstance(ready, dict) and str(ready.get("summary_id") or "") == summary_id:
                return {"status": str(current.get("status") or status)}
            return {
                "status": str(current.get("status") or status),
                "canonical_terminal_projection_ready": {
                    "summary_id": summary_id,
                    "task_done_ts": str(event.get("ts") or utc_now_iso()),
                    "chat_id": int(event.get("chat_id") or task.get("chat_id") or 0),
                },
            }
        effective = {**result, **current}
        project_id = resolve_project_id({**task, **effective})
        role = str(effective.get("role") or task.get("role") or ("root" if is_root else "child"))
        reason = str(effective.get("reason_code") or event.get("reason_code") or "")
        phase = outcome_phase(effective, event)
        outcome = OUTCOME_PHASE_HEADLINE[phase]
        row_chat_id = int(event.get("chat_id") or task.get("chat_id") or 0)
        # The room IS the project and ``result_ref`` IS the pointer, so the row
        # says in words only what the model cannot read off the typed fields:
        # ``memory._format_chat_line`` renders the text and drops every other
        # key, leaving lineage as the one fact that must stay prose.
        text = (f"{outcome}. Root task {tid}." if is_root
                else f"{outcome}. {role} (child {tid} of {parent_id or 'unknown'}).")
        verdict = _completion_verdict(effective, event)
        if verdict:
            text += f" {verdict}"
        excerpt = _completion_excerpt(effective, chat_id=row_chat_id, salvage_only=True)
        if excerpt:
            text += f" {excerpt}"
        result_ref = {"kind": "task_result", "task_id": tid, "reader": "get_task_result"}
        row = {
            "ts": str(event.get("ts") or effective.get("ts") or utc_now_iso()),
            "direction": "system", "type": "task_summary", "summary_kind": summary_kind,
            "summary_id": summary_id, "task_id": tid,
            "parent_task_id": parent_id, "root_task_id": root_id,
            "project_id": project_id,
            "chat_id": row_chat_id,
            "delegation_role": str(effective.get("delegation_role") or task.get("delegation_role") or ""),
            "role": role, "status": str(effective.get("status") or status),
            "outcome": outcome, "outcome_phase": phase, "outcome_final": True,
            "outcome_authority": "canonical_task_result_after_finalization",
            "outcome_axes": effective.get("outcome_axes") or event.get("outcome_axes") or {},
            "reason_code": reason, "result_ref": result_ref,
            "text": text,
        }
        if isinstance(effective.get("model_execution"), dict):
            row["model_execution"] = dict(effective["model_execution"])
        appended = append_canonical_task_summary(drive_root, row)
        if not appended:
            return {"status": str(current.get("status") or status)}
        return {
            "status": str(current.get("status") or status),
            "canonical_terminal_projection": {
                "summary_id": summary_id, "summary_kind": summary_kind,
                "written_at": row["ts"], "chat_id": row_chat_id,
            },
            "canonical_terminal_projection_ready": None,
        }

    write_task_result(
        drive_root, tid, status, _field_projector=_append_once,
    )
    return appended


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
    """Split a stored custody-debt code into (execution reason, custody clause).

    The custody overlay stamps ``delegated_custody_unreconciled`` as the row's
    reason_code, and the debt then HEALS from the write side while
    ``docs/ARCHITECTURE.md`` forbids that refresh rewriting reason_code. So the
    stored code outlives the fact: nine of fourteen terminal rows named a debt
    the same record showed as empty. Render time holds the only fresh truth, and
    the fresh truth is the row's own ``delegated_runs_unreconciled`` list.

    The debt is a WARNING BESIDE the rail cause, never a replacement: when both
    are real the caller states them in one line. Any other reason code passes
    through untouched."""
    from ouroboros.outcomes import WARN_DELEGATED_CUSTODY_UNRECONCILED

    if reason != WARN_DELEGATED_CUSTODY_UNRECONCILED:
        return reason, ""
    debt: Any = None
    execution_reason = ""
    for source in (result, event):
        if debt is None and isinstance(source.get("delegated_runs_unreconciled"), list):
            debt = source["delegated_runs_unreconciled"]
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
        execution_reason = execution_reason or str(execution.get("reason_code") or "")
    return execution_reason, (WARN_DELEGATED_CUSTODY_UNRECONCILED if debt else "")


def _completion_verdict(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    """One TERMINATED host clause for BOTH lifecycle rows.

    A host row must not present an unaccepted claim as the whole story: a
    non-accepted decision speaks through the owner sentence of its own typed
    reason, otherwise the execution reason speaks. The stored reviewer
    rationale never reaches the row — it stays in the card, the task result and
    Logs, which is the complete text this pointer resolves to. The Python twin
    of ``taskReasonDetail``; callers add no punctuation.
    """
    from ouroboros.outcomes import (
        ACCEPTANCE_ACCEPTED, REASON_FINAL_MESSAGE, REASON_OWNER_REQUESTED_FINALIZATION,
    )

    decision: Dict[str, Any] = {}
    veto: Dict[str, Any] = {}
    for source in (event, result):
        axes = source.get("outcome_axes") if isinstance(source.get("outcome_axes"), dict) else {}
        objective = axes.get("objective") or {}
        if isinstance(objective, dict) and isinstance(objective.get("receipt_veto"), dict):
            veto = objective["receipt_veto"]
        for holder in (source.get("review_status"), axes.get("review")):
            if isinstance(holder, dict) and isinstance(holder.get("acceptance_decision"), dict):
                decision = holder["acceptance_decision"]
    status = str(decision.get("status") or "").strip()
    cause = str(decision.get("reason") or "")
    reason = str(result.get("reason_code") or event.get("reason_code") or "")
    if (reason != REASON_OWNER_REQUESTED_FINALIZATION and status
            and (status != ACCEPTANCE_ACCEPTED or cause in TASK_CAUSE_PHRASES)
            and outcome_phase(result, event) in {"done", "warn"}):
        clause = TASK_CAUSE_PHRASES.get(cause, cause)
    elif reason in {REASON_OWNER_REQUESTED_FINALIZATION, REASON_FINAL_MESSAGE}:
        return ""
    else:
        # A healed debt is never restored here. The objective warning the
        # overlay froze keeps the headline and the refresh may not rewrite it,
        # but naming the code again would state a debt the same record shows as
        # empty. The debt is a warning BESIDE the rail cause, and a row with
        # neither states no cause and leaves the headline to its own axis.
        reason, custody = _custody_debt_reason(reason, result, event)
        detail = veto.get("detail") if veto.get("reason") == reason else ""
        clause = (" ".join(strip_markdown(str(detail)).split()) if detail
                  else TASK_CAUSE_PHRASES.get(reason, reason))
        if clause and custody:
            clause += f" ({TASK_CAUSE_PHRASES.get(custody, custody)})"
        elif custody:
            clause = TASK_CAUSE_PHRASES.get(custody, custody)
    if not clause:
        return ""
    return clause if clause.endswith((".", "!", "?", "…", ")")) else clause + "."


def _run_lives_in_its_project(
    drive_root: Any, task_id: str, project_id: str, task: Dict[str, Any], result: Dict[str, Any],
) -> bool:
    """Did this run's work actually go into that project's room?

    Two facts answer yes, and only these two. The run was ADDRESSED there —
    admission resolves a registered project's thread, and a scoped run cannot be
    addressed anywhere else. Or the run is BOUND to it, which is how a task that
    started unscoped joins a project mid-flight; a binding re-homes every row it
    already wrote, including the ones written before the project existed.

    Registration alone is not that fact. A run scoped to an id nobody had
    registered yet is admitted to the hidden partition; if that id is registered
    while the run is still going, a room appears its rows never entered, and
    answering "a room exists" is how the reported defect comes back.
    """
    try:
        from ouroboros.projects_registry import get_reserved_project, project_binding_for_task

        chat_id = result.get("chat_id")
        if chat_id is None:
            chat_id = task.get("chat_id")
        project_chat = (get_reserved_project(drive_root, project_id) or {}).get("chat_id")
        if chat_id is not None and project_chat is not None and int(chat_id) == int(project_chat):
            return True
        binding = project_binding_for_task(drive_root, task_id) or {}
        return str(binding.get("project_id") or "") == str(project_id)
    except Exception:
        log.debug("project-room membership check failed for %s", task_id, exc_info=True)
        return False


def enqueue_project_completion_summary(
    drive_root: Any, evt: Dict[str, Any], task_id: str, task: Dict[str, Any],
    result: Dict[str, Any], task_done_event: Dict[str, Any],
) -> bool:
    """Owe Main's compact row for a managed Project root, not a conversation."""
    tid = str(task_id or "").strip()
    task = task if isinstance(task, dict) else {}
    result = result if isinstance(result, dict) else {}
    if not tid or any(
        bool(row.get("_is_direct_chat"))
        for row in (evt, task, result, task_done_event) if isinstance(row, dict)
    ):
        return False
    try:
        from ouroboros.projects_registry import task_presentation_snapshot
        from ouroboros.task_results import resolve_task_lineage
        from ouroboros.task_status import SETTLED_STATUSES
        from supervisor.terminal_delivery import enqueue_terminal_delivery

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
            return False
        snapshot = task_presentation_snapshot(
            drive_root, tid, task=task, result=result,
            project_id=str(result.get("project_id") or task.get("project_id") or ""),
        )
        if not snapshot["project_id"] or not snapshot["project_routable"]:
            # Owner decision 3A: a run whose project id was DERIVED from a
            # workspace has no room, so Main stays silent instead of offering an
            # "Open Project" that lands in an empty duplicate of itself. The same
            # holds once a project is deleting or tombstoned.
            return False
        if not _run_lives_in_its_project(drive_root, tid, snapshot["project_id"], task, result):
            # The room exists but holds none of this run's work: its id was only
            # registered AFTER admission, or a mid-flight bind failed fail-soft.
            # Offering "Open the Project" would reproduce the reported defect —
            # a Main row leading into an empty room.
            return False
        # Only the salvage excerpt survives here: a cut of the model's own
        # answer repeats bytes the Project already holds, while salvaged bytes
        # exist nowhere else. This writer's only pointer is the invitation, so
        # a salvage may never displace it — preserved bytes named with no way
        # to reach them are worse than the plain invitation.
        excerpt = _completion_excerpt(result, chat_id=1, salvage_only=True)
        verdict = _completion_verdict(result, task_done_event)
        lead = f"{verdict} " if verdict else ""
        if excerpt:
            excerpt = f"{excerpt} Open the Project for details."
        event = {
            "type": "send_message", "chat_id": 1, "task_id": tid,
            "text": (f"{snapshot['target_label']} · "
                     f"{completion_status_label(result, task_done_event)}\n"
                     f"{lead}{excerpt or 'Open the Project for details.'}"),
            "role": "system", "system_type": "project_completion_summary",
            "delivery_id": f"project-completion:{tid}",
            "progress_meta": {
                "project_id": snapshot["project_id"],
                "project_name": snapshot["project_name"],
                "target_label": snapshot["target_label"], "status": status,
            },
        }
        return bool(enqueue_terminal_delivery(drive_root, event))
    except Exception:
        log.warning("Failed to enqueue Project completion summary for %s", tid, exc_info=True)
        return False


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
            },
        }
        return bool(enqueue_terminal_delivery(drive_root, event))
    except Exception:
        log.warning("Failed to enqueue Project started row for %s", pid, exc_info=True)
        return False


__all__ = [
    "AGENT_RECEIPT_ID_PREFIX",
    "announce_project_started",
    "append_authored_task_summary",
    "append_chat_annotation",
    "append_canonical_task_summary",
    "append_terminal_task_projection",
    "build_owner_message_ref",
    "chat_annotation_receipt",
    "entry_matches_source_ref",
    "latest_chat_annotations",
    "enqueue_project_completion_summary",
    "completion_status_label",
    "outcome_phase",
    "owner_message_ref_is_valid",
    "project_origin_rows",
    "project_question_pointer",
    "project_recent_dialogue",
    "routing_options_with_labels",
    "routing_target_label",
    "resolve_owner_message_source",
    "source_refs_for_project",
]
