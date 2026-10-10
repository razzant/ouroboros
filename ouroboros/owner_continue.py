"""The owner's Continue after a technical interruption: identity, eligibility, sources.

Owner Batch4 (1A/4A/8A): Continue is a NEW root in the same room and folder
that reads the interrupted work; it is never a resurrection of the old task,
never offered after the owner's own Stop, a Panic or a finished answer, and it
adopts no old children. The successor model decides whether and how to go on
(LLM-first) and asks in the existing conversation when unsure — the host
carries facts, not a policy. This module is the protocol's data half; the
queue admission is ``supervisor/continuation_admission.py``.

- IDENTITY: the successor id is derived from ``(verb, predecessor, action
  nonce)``; the nonce is the client's, kept across retry and reload. The id is
  a name, not the proof: the full binding (predecessor, nonce, successor,
  destination, group, token) is recorded and compared on every replay.
- CLAIM before any effect: the binding is written on the predecessor's result
  (``continued_by``, under its own lock) before the successor row, the queue or
  any workspace effect; a write that fails refuses with nothing done. One
  predecessor has one Continue successor: a later press with another nonce is
  answered with the accepted successor (a stale card points to it); new
  intentional work goes through the ordinary conversation.
- ELIGIBILITY from typed facts only (``continuation_eligibility``): a settled
  ROOT whose recorded cause is a technical interruption — the owner Restart's
  or a graceful shutdown's cancel origin (the stop door records it,
  ``worker_pool_lifecycle._write_failure_result``), a boot's restore fence, an
  infrastructure reason code, or a technical execution limit, including a
  forced finalization that still delivered a best-effort answer. An owner
  Stop, a Panic, a finished answer, the work's own hard deadline or money
  limit, a live task or a cause that was never recorded refuse — the host
  never guesses a cause from prose.
- SOURCES kept apart: the exact original owner message and every later
  owner-authored mailbox row (read or unread) are the owner's; the
  predecessor's own result is its authored note; unread mail from tasks is
  peer context. A required owner source that cannot be read refuses (a gap is
  named, never reconstructed as owner text).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

VERB = "continue"
BINDING_VERSION = 1
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

# Recorded stop sources that are the owner's explicit decision (never Continue)
# and the ones that record a technical interruption. Unknown sources refuse.
OWNER_STOP_SOURCES = frozenset({"http_single", "http_cascade", "http_graceful", "cascade_descendant",
                                "owner", "owner_stop", "panic"})
TECHNICAL_CANCEL_SOURCES = frozenset({"owner_restart", "snapshot_restore", "server_shutdown"})
# Terminal reason codes that name an infrastructure failure or a technical
# execution limit — not a decision about the work. Owner 1A: technical limits
# included; the successor keeps every explicit whole-work bound regardless.
TECHNICAL_REASON_CODES = frozenset({
    "provider_unavailable", "provider_failure", "llm_api_error", "provider_rejected_tool_dialect",
    "reaper_wedged_worker_alive", "round_limit", "execution_deadline", "absolute_ceiling",
    "task_exception", "workers_unavailable", "worker_pool_unavailable", "finalization_grace",
    "idle_timeout", "worker_crash_signal", "worker_crash_retry_exhausted",
    # A worker that died after an owner wait or with budget-continuation evidence
    # is not retried automatically (replaying would repeat effects), and its saved
    # source is retained: an explicit Continue is exactly its manual path (#1543).
    "worker_crash_owner_wait", "worker_crash_budget_pausing",
})
# The loop's forced-finalization rails that are technical limits: an extracted
# best-effort answer settles ``completed``, the host fallback ``failed``
# (``outcomes.derive_loop_outcome``), so the rail, not the status word, decides.
EXECUTION_LIMIT_RAILS = frozenset({"round_limit", "finalization_grace"})
# The work's own hard bounds (its deadline, its money): a new task id never
# extends them, so a task they ended is not offered Continue.
HARD_LIMIT_REASON_CODES = frozenset({"deadline", "deadline_local", "budget_exhausted"})
OWNER_SOURCE_KINDS = frozenset({"owner_text", "quiz_answer"})
# The retained owner corpus (``review_evidence.task_inputs``) labels each row
# by its recorder's typed ``source`` (``loop_messages._record_owner_directive``):
# owner-authored rows (the successor's seeding set), the run's first text (a
# projection of the original, never a later message) and a task's words
# (context). A row whose label is none of these is not promoted to owner text.
CORPUS_OWNER_SOURCES = frozenset({"owner_mailbox", "owner_quiz_answer", "owner_corpus", "direct_incoming"})
CORPUS_INITIAL_SOURCES = frozenset({"origin_message", "initial_user", "initial_text", "initial_text_transcript"})
CORPUS_TASK_SOURCES = frozenset({"principal_task_message", "relayed_peer_message"})


def valid_nonce(nonce: Any) -> str:
    text = str(nonce or "").strip()
    if not _NONCE_RE.fullmatch(text):
        raise ValueError("action_nonce must be 8-128 characters of [A-Za-z0-9_-]")
    return text


def successor_id(predecessor_task_id: str, nonce: str) -> str:
    digest = hashlib.sha256(f"{VERB}|{predecessor_task_id}|{nonce}".encode("utf-8")).hexdigest()
    return f"{str(predecessor_task_id)[:48]}-c{digest[:16]}"


def binding_sha(binding: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _is_root(result: Dict[str, Any], task_id: str) -> bool:
    root = str(result.get("root_task_id") or task_id)
    return root == str(task_id) and not str(result.get("parent_task_id") or "")


def continuation_eligibility(result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """``{"eligible": bool, "cause": str, "refusal": str}`` from typed facts only."""
    from ouroboros.task_status import SETTLED_STATUSES

    status = str(result.get("status") or "")
    if not result:
        return {"eligible": False, "cause": "", "refusal": "predecessor_missing"}
    if status not in SETTLED_STATUSES:
        return {"eligible": False, "cause": "", "refusal": "predecessor_live"}
    if not _is_root(result, task_id):
        return {"eligible": False, "cause": "", "refusal": "not_a_root_task"}
    reason = str(result.get("reason_code") or "")
    if reason in HARD_LIMIT_REASON_CODES:
        return {"eligible": False, "cause": reason, "refusal": "hard_limit_reached"}
    if status == "completed" and reason not in EXECUTION_LIMIT_RAILS:
        return {"eligible": False, "cause": "finished", "refusal": "author_finished"}
    origin = result.get("cancel_origin") if isinstance(result.get("cancel_origin"), dict) else {}
    source = str(origin.get("source") or "")
    if source in OWNER_STOP_SOURCES:
        return {"eligible": False, "cause": source, "refusal": "stopped_by_owner"}
    from ouroboros.deadline_utils import seconds_until

    # A reached hard deadline is one too (the supervisor's grace rail may have been it).
    carriers = (result, result.get("metadata"), result.get("task_contract"))
    deadline = next((row["deadline_at"] for row in carriers if isinstance(row, dict) and row.get("deadline_at")), "")
    if deadline and seconds_until(deadline) == 0.0:
        return {"eligible": False, "cause": "deadline", "refusal": "hard_limit_reached"}
    if source in TECHNICAL_CANCEL_SOURCES or str(origin.get("reason") or "") == "server_shutdown":
        return {"eligible": True, "cause": source or "server_shutdown", "refusal": ""}
    if not origin and reason in TECHNICAL_REASON_CODES:
        return {"eligible": True, "cause": reason, "refusal": ""}
    return {"eligible": False, "cause": source or reason, "refusal": "interruption_cause_unrecorded"}


def _mail_rows(drive_root: pathlib.Path, task_id: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Every owner-authored mailbox row (read AND unread) and peer rows, in order."""
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return [], True
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [], False
    rows: List[Dict[str, Any]] = []
    complete = not content or content.endswith("\n")
    for line in mailbox_lines(content):
        try:
            entry = json.loads(line)
        except ValueError:
            complete = False
            continue
        if isinstance(entry, dict):
            rows.append({**entry, "_exact_row": line})
        else:
            complete = False
    return rows, complete


def _mailbox_exists(drive_root: pathlib.Path, task_id: str) -> bool:
    from ouroboros.owner_mailbox import _mailbox_path

    return _mailbox_path(drive_root, task_id).exists()


def owner_sources(drive_root: Any, result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """The predecessor's owner sources, authored note pointer and peer context, apart.

    ``gaps`` names every required source that could not be read, and every row
    whose shape or provenance its producer never writes (a non-text body, an
    unknown source label): the admission refuses on any gap rather than
    reconstructing, promoting or silently dropping owner text.
    """
    root = pathlib.Path(drive_root)
    gaps: List[str] = []

    def container(parent, key, kind, gap):
        value = parent.get(key, kind())
        if not isinstance(value, kind):
            gaps.append(gap)
            return kind()
        return value

    metadata = container(result, "metadata", dict, "owner_source_metadata_malformed")
    continuation = container(metadata, "continuation", dict, "chained_owner_source_malformed")
    chained = container(continuation, "owner_sources", dict, "chained_owner_source_malformed")
    original: Optional[Dict[str, Any]] = None
    if "owner_sources" in continuation:
        original = container(chained, "original", dict, "chained_owner_source_malformed")
    elif result.get("origin_message_ref") or result.get("origin_message_text"):
        original = {"source": "origin_message", "content": result.get("origin_message_text"),
                    "origin_message_ref": result.get("origin_message_ref")}
    if original is None or not isinstance(original.get("content"), str) or not original["content"].strip():
        gaps.append("original_owner_message_text")
    if original and (original.get("source") != "origin_message" or
                     not isinstance(original.get("origin_message_ref", {}), dict)):
        gaps.append("original_owner_source_malformed")
    later: List[Dict[str, Any]] = []
    peers: List[Dict[str, Any]] = []
    seen: Dict[str, Tuple[str, bool]] = {}
    for gap in container(chained, "gaps", list, "chained_owner_source_malformed"):
        gaps.append(gap if isinstance(gap, str) and gap else "chained_owner_source_malformed")
    # Validate identities before deduplication: a dict comprehension would hide rivals.
    for key, target, text_key, source_key, sources in (
        ("later", later, "content", "source", CORPUS_OWNER_SOURCES),
        ("peer_context", peers, "text", "provenance", None),
    ):
        for row in container(continuation if key == "peer_context" else chained, key, list,
                             "chained_owner_source_malformed"):
            if not (isinstance(row, dict) and isinstance(row.get(text_key), str)
                    and all(isinstance(row.get(field, ""), str) for field in
                            (source_key, "msg_id", "source_task_id", "relayed_from_task_id"))):
                gaps.append("chained_owner_source_malformed")
                continue
            if sources is not None and row.get(source_key, "") not in sources:
                gaps.append("owner_source_provenance_unknown")
                continue
            msg_id, text = row.get("msg_id", ""), row[text_key]
            if msg_id and msg_id in seen:
                if seen[msg_id] != (text, target is later):
                    gaps.append("owner_message_identity_conflict")
                continue
            if msg_id:
                seen[msg_id] = (text, target is later)
            target.append(dict(row))
    late_framed: set = set()  # ids whose exact row carries typed late-answer provenance
    # Deferred history leaves the child's exact mailbox in custody. A canonical
    # terminal capture is not proof that this retained source has no later words.
    from ouroboros.task_custody import own_child_drives
    rows, complete, live = [], True, False
    try:
        for drive in [root, *own_child_drives(root, task_id)]:
            entries, whole = _mail_rows(drive, task_id)
            rows.extend(entries)
            complete = complete and whole
            live = live or _mailbox_exists(drive, task_id)
    except (OSError, ValueError):
        complete = False
    if not complete:
        gaps.append("owner_mailbox_unreadable")
    # The root's terminal captured its exact owner rows, ACKed ones included
    # (``task_custody.capture_owner_mail``): the copy that outlives the cleanup.
    captured_owner = container(result, "owner_mailbox", dict, "retained_mailbox_malformed")
    if not live and "owner_mailbox" not in result:
        gaps.append("owner_mailbox_uncaptured")
    elif not live and captured_owner.get("read_complete") is not True:
        gaps.append("owner_mailbox_capture_incomplete")
    unread = container(result, "unread_mailbox", dict, "retained_mailbox_malformed")
    held_rows = [row for held in (captured_owner, unread)
                 for row in container(held, "rows", list, "retained_mailbox_malformed")]
    live_rows, rows = rows, []
    for exact in held_rows:
        try:
            entry = json.loads(exact)
        except (TypeError, ValueError):
            gaps.append("retained_mailbox_row_malformed")
            continue
        if isinstance(entry, dict):
            rows.append({**entry, "_exact_row": exact})
        else:
            gaps.append("retained_mailbox_row_malformed")
    rows.extend(live_rows)
    exact_seen: set = set()
    for entry in rows:
        if any(not isinstance(entry.get(key, ""), str) for key in ("msg_id", "kind", "source_task_id")):
            gaps.append("owner_mailbox_row_malformed")
            continue
        msg_id = entry.get("msg_id", "")
        kind = entry.get("kind") or "owner_text"
        if entry["_exact_row"] in exact_seen:
            continue
        if kind in OWNER_SOURCE_KINDS | {"task_message"} and not isinstance(entry.get("text") or "", str):
            gaps.append("owner_mailbox_row_malformed")  # the writer stores text; never str() a body
            continue
        if msg_id and msg_id in seen:
            if seen[msg_id] != (str(entry.get("text") or ""), kind in OWNER_SOURCE_KINDS):
                gaps.append("owner_message_identity_conflict")
            continue
        exact_seen.add(entry["_exact_row"])
        if msg_id:
            seen[msg_id] = (str(entry.get("text") or ""), kind in OWNER_SOURCE_KINDS)
            if isinstance(entry.get("late_answer"), dict):
                late_framed.add(msg_id)
        if kind in OWNER_SOURCE_KINDS:
            later.append({"source": "owner_quiz_answer" if kind == "quiz_answer" else "owner_mailbox",
                          "content": str(entry.get("text") or ""), "msg_id": msg_id,
                          "ts": str(entry.get("ts") or ""), "exact_row": entry["_exact_row"]})
        elif kind == "task_message":
            peers.append({"msg_id": msg_id, "ts": str(entry.get("ts") or ""),
                          "source_task_id": str(entry.get("source_task_id") or ""),
                          "provenance": str(entry.get("provenance") or ""),
                          "text": str(entry.get("text") or "")})
    # The retained corpus keeps what the mailbox may no longer hold (a direct
    # turn's input, a row consumed before any capture) under the SAME identity
    # rules; an exact mailbox row outranks its projection (a late answer's
    # corpus row is the model-facing frame of that typed row, not a rival text).
    evidence = container(result, "review_evidence", dict, "retained_owner_corpus_malformed")
    inputs = container(evidence, "task_inputs", dict, "retained_owner_corpus_malformed")
    unavailable = container(inputs, "unavailable_sections", list, "retained_owner_corpus_malformed")
    if any(not isinstance(section, str) for section in unavailable):
        gaps.append("retained_owner_corpus_malformed")
    if "owner_requirements_and_decisions" in unavailable:
        gaps.append("owner_corpus_capture_unavailable")
    corpus = container(inputs, "owner_requirements_and_decisions", list, "retained_owner_corpus_malformed")
    texts = {True: {row["content"] for row in later} | {str((original or {}).get("content") or "")},
             False: {row["text"] for row in peers}}
    for row in corpus:
        # The recorder writes a non-empty text body, a source label and string ids.
        if not (isinstance(row, dict) and isinstance(row.get("content"), str) and row["content"].strip()
                and all(isinstance(row.get(key, ""), str)
                        for key in ("source", "msg_id", "source_task_id", "relayed_from_task_id"))):
            gaps.append("retained_owner_corpus_row_malformed")
            continue
        msg_id, source, text = str(row.get("msg_id") or ""), str(row.get("source") or ""), row["content"]
        if source not in CORPUS_INITIAL_SOURCES | CORPUS_OWNER_SOURCES | CORPUS_TASK_SOURCES:
            gaps.append("owner_source_provenance_unknown")
            continue
        if source in CORPUS_INITIAL_SOURCES:
            continue  # the run's first text is the original's own projection
        if msg_id and msg_id in seen:
            if (seen[msg_id][1] != (source in CORPUS_OWNER_SOURCES) or
                    (seen[msg_id][0] != text and msg_id not in late_framed)):
                gaps.append("owner_message_identity_conflict")
            continue
        if not msg_id and text in texts[source in CORPUS_OWNER_SOURCES]:
            continue
        if msg_id:
            seen[msg_id] = (text, source in CORPUS_OWNER_SOURCES)
        texts[source in CORPUS_OWNER_SOURCES].add(text)
        if source in CORPUS_OWNER_SOURCES:
            later.append({"source": source, "content": text, "msg_id": msg_id})
        elif source in CORPUS_TASK_SOURCES:
            peers.append({"msg_id": msg_id, "ts": "", "source_task_id": str(row.get("source_task_id") or ""),
                          "provenance": source, "text": text,
                          **({"relayed_from_task_id": str(row["relayed_from_task_id"])}
                             if row.get("relayed_from_task_id") else {})})
        else:
            gaps.append("owner_source_provenance_unknown")  # never promoted to owner text
    return {"original": original, "later": later, "peer_context": peers, "gaps": list(dict.fromkeys(gaps)),
            "predecessor_note": {"task_id": task_id, "reader": "get_task_result",
                                 "kind": "predecessor_authored_note"}}


def continuation_room(drive_root: Any, predecessor_task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """The room a FRESH Continue is admitted to: ``{"chat_id", "project_id"}``.

    "Turn into project" writes only the immutable task<->Project binding, the one
    truth about a task's room (owner decision B4=A); the interrupted worker's own
    result keeps the chat and project copy it started with. The binding therefore
    outranks that copy, and unbound work keeps its own row. Only the room moves:
    the original owner message, its ref and the prepared folder stay the
    predecessor's. An unreadable store or a malformed row raises, so the caller
    refuses before its claim instead of admitting the work to Main.
    """
    from ouroboros.projects_registry import project_binding_for_task

    bound = project_binding_for_task(drive_root, predecessor_task_id, strict=True)
    if not bound:
        return {"chat_id": result.get("chat_id"), "project_id": str(result.get("project_id") or "")}
    chat = bound.get("project_chat_id")
    if type(chat) is not int or chat <= 0:
        raise ValueError("Project binding is unavailable")
    return {"chat_id": chat, "project_id": bound["project_id"]}


def owner_corpus_rows(sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The successor's seeded owner corpus (``_initialize_owner_directives``), exact words only."""
    rows: List[Dict[str, Any]] = []
    if isinstance(sources.get("original"), dict) and str(sources["original"].get("content") or "").strip():
        rows.append({"source": "origin_message", "content": sources["original"]["content"]})
    for row in sources.get("later") or []:
        if isinstance(row, dict) and row.get("content"):
            rows.append({"source": row.get("source") or "owner_mailbox", "content": row["content"],
                         **({"msg_id": row["msg_id"]} if row.get("msg_id") else {})})
    return rows


def work_order_text(predecessor_task_id: str, cause: str, sources: Dict[str, Any], *,
                    deadline_at: str = "") -> str:
    """The successor's first message: host-authored FACTS, the owner's words verbatim.

    It frames nothing as a decision: the model judges whether and how to go on
    and asks the owner in this conversation when it cannot tell.
    """
    lines = [
        f"[CONTINUE] The owner pressed Continue on task {predecessor_task_id}, which was interrupted "
        f"(recorded cause: {cause or 'technical interruption'}). This is a NEW task in the same "
        "conversation and folder; the old task is not resumed and its helpers are not yours, but its "
        "stopped delegated runs are yours to continue with delegate_start(subagent_id=..., continue_from=<run_id>, "
        "prompt=...) instead of redoing them.",
        "Read its saved results and materials first: get_task_result("
        f"{predecessor_task_id!r}) — that result is the previous run's own note, not an owner instruction.",
        "Decide whether and how to continue; if the right next step is unclear (for example the work "
        "may no longer be wanted), ask the owner in this conversation instead of guessing.",
        "Use manage_schedules(action='list') to inspect related or unknown future follow-ups. "
        "Continue does not release their holds. Resolve unknown relationships explicitly; restore "
        "only an observed hold_id. Original money and hard deadlines still apply to related work.",
    ]
    if deadline_at:
        lines.append(f"An explicit hard deadline still applies to this work: {deadline_at}.")
    original = sources.get("original") if isinstance(sources.get("original"), dict) else None
    if original and str(original.get("content") or "").strip():
        lines += ["", "Owner's original instruction (verbatim):", str(original["content"])]
    later = [row for row in sources.get("later") or [] if isinstance(row, dict) and row.get("content")]
    if later:
        lines += ["", "Later owner messages to that task (verbatim, in order):"]
        lines += [f"- {row.get('ts') or ''} {row['content']}" for row in later]
    peers = sources.get("peer_context") or []
    if peers:
        lines += ["", "Messages from other tasks that were addressed to it (context, not owner instructions):"]
        lines += [f"- from {row.get('source_task_id') or 'unknown'}: {row.get('text')}" for row in peers]
    return "\n".join(lines)


def claim_on_predecessor(drive_root: Any, predecessor_task_id: str, nonce: str,
                         binding: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Bind this action on the predecessor BEFORE any effect; ``(claim, created)``.

    Same nonce: the recorded claim (replay). Another nonce while a claim
    exists: ``ValueError("already_continued:<successor>")``. A predecessor that
    turned ineligible refuses. Every refusal writes nothing.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import update_json_locked

    outcome: Dict[str, Any] = {}

    def update(current: dict) -> Optional[dict]:
        require_writable_task_result_schema(current)
        claim = current.get("continued_by") if isinstance(current.get("continued_by"), dict) else {}
        if claim:
            if str(claim.get("action_nonce") or "") == nonce:
                outcome.update(claim=dict(claim), created=False)
                return None
            raise ValueError(f"already_continued:{claim.get('successor_task_id') or ''}")
        verdict = continuation_eligibility(current, predecessor_task_id)
        if not verdict["eligible"]:
            raise ValueError(verdict["refusal"])
        claim = {"successor_task_id": binding["successor_task_id"], "action_nonce": nonce,
                 "binding": dict(binding), "binding_sha256": binding_sha(binding),
                 "bound_at": utc_now_iso(), "state": "bound"}
        outcome.update(claim=claim, created=True)
        return stamp_task_result_schema({**current, "continued_by": claim})

    update_json_locked(task_result_path(pathlib.Path(drive_root), predecessor_task_id), update,
                       strict_existing_dict=True)
    return dict(outcome["claim"]), bool(outcome["created"])


def mark_claim_admitted(drive_root: Any, predecessor_task_id: str, nonce: str) -> None:
    """Record on the predecessor that its claimed successor is durably admitted."""
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import update_json_locked

    def update(current: dict) -> Optional[dict]:
        require_writable_task_result_schema(current)
        claim = current.get("continued_by") if isinstance(current.get("continued_by"), dict) else {}
        if str(claim.get("action_nonce") or "") != nonce or claim.get("state") == "admitted":
            return None
        return stamp_task_result_schema(
            {**current, "continued_by": {**claim, "state": "admitted", "admitted_at": utc_now_iso()}})

    update_json_locked(task_result_path(pathlib.Path(drive_root), predecessor_task_id), update,
                       strict_existing_dict=True)


def recorded_continuation(drive_root: Any, predecessor_task_id: str, task_id: str) -> bool:
    """Is ``task_id`` a root the owner's Continue created from ``predecessor_task_id``?

    Read from the recorded claims only: each ``continued_by`` binding on an
    earlier root's own result names its successor (hash-verified), so a chain of
    Continues is followed claim by claim. Room, folder or root equality grants
    nothing, and an unreadable result is no binding (``delegate_continuation``).
    """
    from ouroboros.task_results import load_task_result

    current, target, seen = str(predecessor_task_id or ""), str(task_id or ""), set()
    while current and target and current not in seen:
        seen.add(current)
        try:
            result = load_task_result(drive_root, current, strict=True) or {}
        except Exception:
            return False
        claim = result.get("continued_by") if isinstance(result.get("continued_by"), dict) else {}
        binding = claim.get("binding") if isinstance(claim.get("binding"), dict) else {}
        successor = str(claim.get("successor_task_id") or "")
        if (not successor or binding.get("successor_task_id") != successor
                or claim.get("binding_sha256") != binding_sha(binding)):
            return False
        if successor == target:
            return True
        current = successor
    return False


def continuation_offer(result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """What a settled card may offer: Continue, or the successor it already has.

    A bound claim retries its exact stored action until admission is confirmed.
    Only then does the card point to the successor; new work uses conversation.
    """
    claim = result.get("continued_by") if isinstance(result.get("continued_by"), dict) else {}
    if claim.get("successor_task_id") and claim.get("state") == "bound":
        return {"eligible": False, "refusal": "continuation_unconfirmed", "state": "bound",
                "successor_task_id": str(claim["successor_task_id"]),
                "action_nonce": str(claim.get("action_nonce") or "")}
    if claim.get("successor_task_id"):
        return {"eligible": False, "refusal": "already_continued",
                "successor_task_id": str(claim["successor_task_id"]), "state": str(claim.get("state") or "")}
    return continuation_eligibility(result, task_id)
