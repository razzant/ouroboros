"""The owner's Continue after a technical interruption: identity, eligibility, sources.

Owner Batch4 (1A/4A/8A): Continue is a NEW root in the same room and folder
that reads the interrupted work; it is never a resurrection of the old task,
never offered after the owner's own Stop, a Panic or a finished answer, and it
adopts no old children: it may only decide a direct child's finished native
result (apply or verify it, accept a shared one as it stands, or reject it)
through ``integrate_subagent_patch`` under the recorded relation.
The successor model decides whether and how to go on
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
  named, never reconstructed as owner text). A root ``start_task`` opened
  without an owner message (its producer's ``origin_message_absent``) is
  continued from its drafted objective and the owner's answered cards it began
  from: the asking task's question beside the owner's exact words, never a
  stamp of owner authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso
from ouroboros.task_store import patch_document

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
    # A positively reconciled orphan is a technical interruption, not an
    # unrecorded cause. Continue still rechecks owner sources and live custody.
    "orphaned_running_after_worker_restart", "interrupted_retry_lost",
    # A billing or dispatch-fence authority that could not be read, where no exact pause was saved.
    "billing_authority_unreadable", "control_authority_unreadable", "output_recovery_exhausted",
    # A running direct turn's accounting wait outlived its idle window (``_usage_wait``): storage, not
    # the work, ended it; Continue is its manual path, never an automatic retry.
    "accounting_wait_expired",
})
# The loop's forced-finalization rails that are technical limits: an extracted
# best-effort answer settles ``completed``, the host fallback ``failed``
# (``outcomes.derive_loop_outcome``), so the rail, not the status word, decides.
EXECUTION_LIMIT_RAILS = frozenset({"round_limit", "finalization_grace"})
# The work's own hard bounds (its deadline, its money): a new task id never
# extends them, so a task they ended is not offered Continue.
HARD_LIMIT_REASON_CODES = frozenset({"deadline", "deadline_local", "budget_exhausted"})
OWNER_SOURCE_KINDS = frozenset({"owner_text", "quiz_answer", "quiz_closed", "card_input"})
# The retained owner corpus (``review_evidence.task_inputs``) labels each row
# by its recorder's typed ``source`` (``loop_messages._record_owner_directive``):
# owner-authored rows (the successor's seeding set), the run's first text (a
# projection of the original, never a later message) and a task's words
# (context). A row whose label is none of these is not promoted to owner text.
CORPUS_OWNER_SOURCES = frozenset({"owner_mailbox", "owner_quiz_answer", "owner_corpus", "direct_incoming"})
CORPUS_INITIAL_SOURCES = frozenset({"origin_message", "initial_user", "initial_text", "initial_text_transcript"})
CORPUS_TASK_SOURCES = frozenset({"principal_task_message", "relayed_peer_message"})
# The typed ids an owner-source row may carry, each a string when present: its message,
# a task message's sender and relaying sibling, an owner answer's card.
ROW_IDS = ("msg_id", "source_task_id", "relayed_from_task_id", "quiz_id")
# What a successor reads of an answered card beside the owner's words (``answered_card``).
CARD_FIELDS = ("quiz_id", "question", "options", "answered_index", "comment", "asked_at", "answered_at")


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
    """``{"eligible": bool, "cause": str, "refusal": str}`` from typed facts only.

    An older row's custody stamp is read beneath (``outcomes.recorded_cause``); a
    stamped row whose cause was never recorded still refuses, naming the stamp."""
    from ouroboros.outcomes import recorded_cause
    from ouroboros.task_status import SETTLED_STATUSES

    status = str(result.get("status") or "")
    if not result:
        return {"eligible": False, "cause": "", "refusal": "predecessor_missing"}
    if status not in SETTLED_STATUSES:
        return {"eligible": False, "cause": "", "refusal": "predecessor_live"}
    if not _is_root(result, task_id):
        return {"eligible": False, "cause": "", "refusal": "not_a_root_task"}
    if (result.get("metadata") or {}).get("execution_transport") is not None:
        return {"eligible": False, "cause": "realtime_transport", "refusal": "realtime_restart_required"}
    reason = recorded_cause(result)
    if reason in HARD_LIMIT_REASON_CODES:
        return {"eligible": False, "cause": reason, "refusal": "hard_limit_reached"}
    if reason == "billing_authority_refused":  # missing/foreign lineage: Continue cannot supply one
        return {"eligible": False, "cause": reason, "refusal": "billing_lineage_refused"}
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
    return {"eligible": False, "cause": source or reason or str(result.get("reason_code") or ""),
            "refusal": "interruption_cause_unrecorded"}


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


def answered_card(block: Any, words: str) -> Optional[Dict[str, Any]]:
    """What a successor reads of the card an owner answer replies to, or None.

    ``block`` must be an answered card whose recorded owner words are exactly ``words``.
    The question and options are the asking task's words; they ride beside the answer
    and never enter the owner corpus.
    """
    from ouroboros.owner_quiz import STATE_ANSWERED, late_answer_owner_text

    if not (isinstance(block, dict) and block.get("state") == STATE_ANSWERED
            and late_answer_owner_text(block) == words):
        return None
    return {key: copy.deepcopy(block[key]) for key in CARD_FIELDS if key in block}


def _drafted_start(root: pathlib.Path, result: Dict[str, Any], task_id: str, metadata: Dict[str, Any],
                   gaps: List[str]) -> Optional[Dict[str, Any]]:
    """The start ``start_task`` recorded without an owner message (``origin_message_absent``).

    The objective is the drafting task's, carried as such and never as owner words.
    Each owner answer in the start's corpus must resolve on the asking task's own card,
    named by the identity the producer carried. The card must record exactly these
    words, and the carried message must be that card's answer: its live delivery id,
    which survives a later re-route, or the delivery the card records. Otherwise the
    answer is a named gap. A drafted start without an answered card is autonomous work
    and is not admitted on this source.
    """
    from ouroboros.owner_source import answered_quiz_block
    from ouroboros.question_answers import answer_message_id

    author, objective = metadata.get("objective_author"), result.get("objective") or result.get("description")
    if not (isinstance(author, dict) and author.get("kind") == "task" and isinstance(author.get("task_id"), str)
            and author["task_id"].strip() and isinstance(objective, str) and objective.strip()):
        gaps.append("start_objective_unreadable")
        return None
    answers: List[Dict[str, Any]] = []
    for row in metadata.get("owner_corpus") if isinstance(metadata.get("owner_corpus"), list) else []:
        if not isinstance(row, dict) or row.get("source") != "owner_quiz_answer":
            continue  # the corpus pass names a malformed row; other owner words stay later sources
        asker, quiz_id, words, msg_id = (row.get(key) for key in ("source_task_id", "quiz_id", "content", "msg_id"))
        typed = all(isinstance(value, str) and value.strip() for value in (asker, quiz_id, words, msg_id))
        block = answered_quiz_block(root, asker, quiz_id) if typed and asker != task_id else {}
        delivered = typed and (msg_id == answer_message_id(asker, quiz_id)
                               or (block.get("delivery") or {}).get("msg_id") == msg_id)
        card = answered_card(block, words) if delivered else None
        if card is None:
            gaps.append("start_quiz_answer_unresolved")
            continue
        answers.append({"source": "owner_quiz_answer", "content": words, "msg_id": msg_id, "source_task_id": asker,
                        "quiz_id": quiz_id, "request_id": block["request_id"], "card": card})
    if not answers and "start_quiz_answer_unresolved" not in gaps:
        gaps.append("start_quiz_answer_absent")
    expected = result.get("expected_output")
    return {"source": "drafted_start", "task_id": task_id, "objective": objective,
            **({"expected_output": expected} if isinstance(expected, str) and expected.strip() else {}),
            "author": copy.deepcopy(author), "answers": answers}


def valid_start(start: Any) -> bool:
    """A chained start basis exactly as ``_drafted_start`` builds it."""
    answers = start.get("answers") if isinstance(start, dict) else None
    return (isinstance(start, dict) and start.get("source") == "drafted_start"
            and all(isinstance(start.get(key), str) and start[key].strip() for key in ("task_id", "objective"))
            and isinstance(start.get("author"), dict) and start["author"].get("kind") == "task"
            and isinstance(answers, list) and bool(answers)
            and all(isinstance(row, dict) and row.get("source") == "owner_quiz_answer" and isinstance(row.get("card"), dict)
                    and all(isinstance(row.get(key), str) and row[key].strip()
                            for key in ("content", "msg_id", "source_task_id", "quiz_id")) for row in answers))


def owner_sources(drive_root: Any, result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """The predecessor's owner sources, authored note pointer and peer context, apart.

    The opening is one of three recorded states: the owner's message (``original``);
    ``start``, which ``start_task`` recorded without one (``_drafted_start``); or neither,
    a lost or legacy source and a gap. Only the producer's own fact marks absence. An empty
    field or a false ``owner_ingress`` never stands in for it. ``gaps`` names every
    required source that could not be read, and every row whose shape or provenance its
    producer never writes (a non-text body, an unknown source label): the admission
    refuses on any gap rather than reconstructing, promoting or silently dropping owner text.
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
    from ouroboros.schedule_continuation import admission_basis, valid_basis
    schedule = chained.get("schedule") or admission_basis(result, task_id)
    if schedule is not None and not valid_basis(schedule):
        gaps.append("schedule_source_malformed")
        schedule = None
    chained_start = chained.get("start")
    original: Optional[Dict[str, Any]] = None
    if "owner_sources" in continuation and not ((schedule or chained_start is not None)
                                                and chained.get("original") is None):
        original = container(chained, "original", dict, "chained_owner_source_malformed")
    elif result.get("origin_message_ref") or result.get("origin_message_text"):
        original = {"source": "origin_message", "content": result.get("origin_message_text"),
                    "origin_message_ref": result.get("origin_message_ref")}
    absent = metadata.get("origin_message_absent") is True
    if absent and (result.get("origin_message_ref") or result.get("origin_message_text")
                   or metadata.get("origin_suppressed")):
        gaps.append("opening_owner_message_conflict")  # the producer never writes both
    start: Optional[Dict[str, Any]] = None
    if (original is None or not isinstance(original.get("content"), str) or not original["content"].strip()) \
            and not schedule:
        if chained_start is not None:  # a Continue of a Continue keeps the start it was admitted on
            start = chained_start if valid_start(chained_start) else None
            if start is None:
                gaps.append("start_source_malformed")
        elif absent:
            if "opening_owner_message_conflict" not in gaps:
                start = _drafted_start(root, result, task_id, metadata, gaps)
        else:
            gaps.append("original_owner_message_text")
    if original and (original.get("source") != "origin_message" or
                     not isinstance(original.get("origin_message_ref", {}), dict)):
        gaps.append("original_owner_source_malformed")
    later: List[Dict[str, Any]] = []
    peers: List[Dict[str, Any]] = []
    # The start's answers keep their one identity: a corpus copy is the same message.
    seen: Dict[str, Tuple[str, bool]] = {row["msg_id"]: (row["content"], True) for row in (start or {}).get("answers") or []}
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
                    and all(isinstance(row.get(field, ""), str) for field in (source_key, *ROW_IDS))):
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
    quizzes = result.get("owner_quiz") if isinstance(result.get("owner_quiz"), dict) else {}

    def corpus_pass(rows: List[Any], malformed: str) -> None:
        """A recorded corpus under the one identity rule: a msg_id names one text, the run's
        first text is the original's projection, a task's words are context and an unknown
        label is never promoted to owner text."""
        texts = {True: {row["content"] for row in later} | {str((original or {}).get("content") or "")},
                 False: {row["text"] for row in peers}}
        for row in rows:
            # The recorder writes a non-empty text body, a source label and string ids.
            if not (isinstance(row, dict) and isinstance(row.get("content"), str) and row["content"].strip()
                    and all(isinstance(row.get(key, ""), str) for key in ("source", *ROW_IDS))):
                gaps.append(malformed)
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
                ids = {key: row[key] for key in ROW_IDS[1:] if row.get(key)}
                # An answer to the predecessor's own card reads that card from its own record.
                card = (answered_card(quizzes.get(ids.get("quiz_id", "")), text)
                        if source == "owner_quiz_answer" and ids.get("source_task_id") == task_id else None)
                later.append({"source": source, "content": text, "msg_id": msg_id, **ids,
                              **({"card": card} if card else {})})
            elif source in CORPUS_TASK_SOURCES:
                peers.append({"msg_id": msg_id, "ts": "", "source_task_id": str(row.get("source_task_id") or ""),
                              "provenance": source, "text": text,
                              **({"relayed_from_task_id": str(row["relayed_from_task_id"])}
                                 if row.get("relayed_from_task_id") else {})})
            else:
                gaps.append("owner_source_provenance_unknown")  # never promoted to owner text

    # The start's own corpus (``metadata.owner_corpus``, which the producer persisted) keeps
    # the owner words this root began with even when no terminal corpus was captured.
    corpus_pass(container(metadata, "owner_corpus", list, "start_owner_corpus_malformed"),
                "start_owner_corpus_row_malformed")
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
        if kind == "quiz_answer" and (entry.get("quiz_delivery") or {}).get("answer_snapshot"):
            from ouroboros.owner_quiz import late_answer_owner_text
            entry = {**entry, "text": late_answer_owner_text(entry["quiz_delivery"]["answer_snapshot"])}
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
            row = {"source": "owner_quiz_answer" if kind == "quiz_answer" else "owner_mailbox",
                   "content": str(entry.get("text") or ""), "msg_id": msg_id,
                   "ts": str(entry.get("ts") or ""), "exact_row": entry["_exact_row"]}
            delivery = entry.get("quiz_delivery") if kind == "quiz_answer" else None
            card = answered_card(delivery.get("answer_snapshot"), row["content"]) if isinstance(delivery, dict) else None
            if card and all(isinstance(delivery.get(key), str) and delivery[key] for key in ("task_id", "quiz_id")):
                # The delivered card's snapshot: its question rides beside the owner's words.
                row.update(source_task_id=delivery["task_id"], quiz_id=delivery["quiz_id"], card=card)
            later.append(row)
        elif kind == "relay_entry":
            for row in entry.get("owner_rows") or []:
                from ouroboros.chat_chain import format_address, row_address
                from ouroboros.owner_relay import owner_text
                address = format_address(row_address(row))
                if address not in seen:
                    later.append({"source": "owner_mailbox", "content": owner_text(row), "msg_id": address})
                    seen[address] = (owner_text(row), True)
            peers.append({"msg_id": msg_id, "ts": str(entry.get("ts") or ""),
                          "source_task_id": str(entry.get("source_task_id") or ""),
                          "provenance": "independent_task", "text": str(entry.get("text") or "")})
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
    corpus_pass(container(inputs, "owner_requirements_and_decisions", list, "retained_owner_corpus_malformed"),
                "retained_owner_corpus_row_malformed")
    return {"original": original, "schedule": schedule, "start": start, "later": later, "peer_context": peers,
            "gaps": list(dict.fromkeys(gaps)),
            "open_relays": result.get("open_relays", chained.get("open_relays", {})),
            "held_owner_messages": {**chained.get("held_owner_messages", {}), **result.get("held_owner_messages", {})},
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
    """The successor's seeded owner corpus (``_initialize_owner_directives``): exact words
    with their typed ids, the start's answers first; a card's question never enters it."""
    rows: List[Dict[str, Any]] = []
    if isinstance(sources.get("original"), dict) and str(sources["original"].get("content") or "").strip():
        rows.append({"source": "origin_message", "content": sources["original"]["content"]})
    start = sources.get("start") if isinstance(sources.get("start"), dict) else {}
    for row in [*(start.get("answers") or []), *(sources.get("later") or [])]:
        if isinstance(row, dict) and row.get("content"):
            rows.append({"source": row.get("source") or "owner_mailbox", "content": row["content"],
                         **{key: row[key] for key in ROW_IDS if row.get(key)}})
    return rows


def _card_lines(row: Dict[str, Any]) -> List[str]:
    """One answered card for the successor: the asking task's question, then the owner's answer."""
    card = row.get("card") if isinstance(row.get("card"), dict) else {}
    options = card.get("options") if isinstance(card.get("options"), list) else []
    index, comment = card.get("answered_index"), str(card.get("comment") or "")
    answer = [f"chose option {index + 1}: {options[index]}"] if type(index) is int and 0 <= index < len(options) else []
    answer += [f"wrote (verbatim): {comment}"] if comment.strip() else []
    return [f"- Card {card.get('quiz_id') or row.get('quiz_id')} of task {row.get('source_task_id')}, "
            f"asked {card.get('asked_at') or 'at an unrecorded time'}: {card.get('question')}",
            f"  Options: {'; '.join(f'{n}. {label}' for n, label in enumerate(options, 1)) or 'none recorded'}",
            f"  The owner {'; '.join(answer) or 'answered (verbatim): ' + str(row.get('content') or '')}"
            f" (answered {card.get('answered_at') or 'at an unrecorded time'})."]


def recorded_system_checkout(result: Dict[str, Any]) -> Dict[str, str]:
    """The predecessor's recorded system-checkout observations, each dated by its own capture.

    The only per-task fact is the mutation baseline's ``system_repo`` surface:
    ``rev-parse HEAD`` at that surface's ``captured_at``. The task's own
    attributed commit or an accepted fast-forward re-anchors the surface (new
    SHA, new timestamp), so it is an observation at that moment — never the
    task's start, the code its process imported, or the checkout now. A
    retained terminal snapshot of the same surface is a second dated
    observation. Nothing recorded: ``{}``.
    """
    evidence = result.get("mutation_evidence") if isinstance(result.get("mutation_evidence"), dict) else {}
    baseline = evidence.get("baseline") if isinstance(evidence.get("baseline"), dict) else {}
    surface = next((row for row in baseline.get("surfaces") or [] if isinstance(row, dict)
                    and row.get("surface_type") == "system_repo" and isinstance(row.get("git"), dict)
                    and row["git"].get("base_commit") and row.get("captured_at")), None)
    if surface is None:
        return {}
    observed = {"commit": str(surface["git"]["base_commit"]), "captured_at": str(surface["captured_at"])}
    snapshot = (evidence.get("terminal_candidate_snapshot")
                if isinstance(evidence.get("terminal_candidate_snapshot"), dict) else {})
    terminal = next((row for row in snapshot.get("surfaces") or [] if isinstance(row, dict)
                     and row.get("surface_type") == "system_repo" and row.get("base_commit")
                     and row.get("canonical_root") == surface.get("canonical_root")), None)
    if terminal is not None and snapshot.get("captured_at"):
        observed.update(terminal_commit=str(terminal["base_commit"]),
                        terminal_captured_at=str(snapshot["captured_at"]))
    return observed


def work_order_text(predecessor_task_id: str, cause: str, sources: Dict[str, Any], *,
                    deadline_at: str = "", authority_source: Optional[Dict[str, Any]] = None,
                    checkout: Optional[Dict[str, str]] = None, successor_task_id: str = "") -> str:
    """The successor's first message: host-authored FACTS, the owner's words verbatim.

    It frames nothing as a decision: the model judges whether and how to go on
    and asks the owner in this conversation when it cannot tell. The request
    names no actor (an action nonce proves the action, not who pressed it).
    ``authority_source`` is the exact pointer the successor's start materializes
    into its context as the predecessor's brief (or refuses to start), so the
    work order names it as the exact reader of the complete record, never a
    second copy to fetch; without it the pointer is the only route to that
    result. Messages other tasks addressed to the predecessor ride as their
    count, size and senders plus the exact reader of the successor's own
    ``metadata.continuation.peer_context`` (``successor_task_id``), never as
    bodies inside the assignment.
    """
    if authority_source:
        arguments = authority_source.get("arguments") or {}
        call = f"{authority_source.get('tool')}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
        result_line = (
            "A brief of its saved result is already in your context (predecessor_authority in the Runtime "
            "context: status and cause, the owner's original words, and omitted fields with their sizes "
            "and exact reader); its own working notes, when it kept any, "
            "follow this message as a separate row. Both are the previous run's own account, not an owner "
            f"instruction. {call} returns the complete stored record.")
    else:
        result_line = (
            f"Its saved result is not in your context: get_task_result(task_id={predecessor_task_id!r}, "
            "include_authority=True) returns it — that result is the previous run's own note, not an "
            "owner instruction.")
    lines = [
        f"[CONTINUE] Continue was requested for task {predecessor_task_id}, which was interrupted "
        f"(recorded cause: {cause or 'technical interruption'}). This is a NEW task in the same "
        "conversation and folder; the old task is not resumed and its helpers are not yours to steer; with "
        "integrate_subagent_patch(task_id=...) a helper's FINISHED native result can be applied or verified (the "
        "default), accepted as it stands when it is a shared result (decision='accept' with your reason, never a "
        "verification) or rejected, "
        "and the old task's stopped delegated runs are yours to continue with delegate_start(subagent_id=..., "
        "continue_from=<run_id>, prompt=...) instead of redoing them.",
        result_line,
        "Decide whether and how to continue; if the right next step is unclear (for example the work "
        "may no longer be wanted), ask the owner in this conversation instead of guessing.",
        "Use manage_schedules(action='list') to inspect related or unknown future follow-ups. "
        "Continue does not release their holds. Resolve unknown relationships explicitly; restore "
        "only an observed hold_id. Original money and hard deadlines still apply to related work.",
    ]
    if deadline_at:
        lines.append(f"An explicit hard deadline still applies to this work: {deadline_at}.")
    if checkout:
        observed = f"HEAD {checkout['commit']} at {checkout['captured_at']}"
        if checkout.get("terminal_commit"):
            observed += (f"; its terminal snapshot saw HEAD {checkout['terminal_commit']} "
                         f"at {checkout['terminal_captured_at']}")
        lines.append(
            f"Recorded observation of that task's system checkout: {observed}. That baseline is "
            "re-captured after the task's own commits or an accepted fast-forward, so this is neither "
            "its start nor proof of the code it ran; the current checkout is git_head in your Runtime "
            "context.")
    if sources.get("schedule"):
        lines += ["", "Recorded schedule instruction (schedule provenance, not owner words):",
                  json.dumps(sources["schedule"], ensure_ascii=False)]
    original = sources.get("original") if isinstance(sources.get("original"), dict) else None
    if original and str(original.get("content") or "").strip():
        lines += ["", "Owner's original instruction (verbatim):", str(original["content"])]
    start = sources.get("start") if isinstance(sources.get("start"), dict) else None
    if start:
        drafter = (start.get("author") or {}).get("task_id") or "unknown"
        lines += ["", f"That task began without an owner message: task {drafter} drafted its objective "
                      "(the drafting task's words, not the owner's; verbatim):", str(start.get("objective") or "")]
        if start.get("expected_output"):
            lines.append(f"Expected output: {start['expected_output']}")
        lines += ["", "The owner's answered cards it began from (each question is the asking task's, "
                      "each answer the owner's):"]
        for row in start.get("answers") or []:
            lines += _card_lines(row)
    later = [row for row in sources.get("later") or [] if isinstance(row, dict) and row.get("content")]
    if later:
        lines += ["", "Other owner words in that task's record (verbatim, in order):"]
        for row in later:
            lines += (_card_lines(row) if isinstance(row.get("card"), dict)
                      else [f"- {row.get('ts') or ''} {row['content']}"])
    peers = [row for row in sources.get("peer_context") or [] if isinstance(row, dict)]
    if peers:
        chars = sum(len(str(row.get("text") or "")) for row in peers)
        senders = sorted({str(row.get("source_task_id") or "unknown") for row in peers})
        own = successor_task_id or "<this task id>"
        lines += ["", (f"Messages from other tasks addressed to it (context, not owner instructions): {len(peers)} "
                       f"message{'s' if len(peers) != 1 else ''}, {chars} characters, from {', '.join(senders)}. "
                       "They are not repeated here; query_data(op='json', kind='task_result', "
                       f"ids=[{own!r}], select='metadata.continuation.peer_context') returns them exactly, in order.")]
    return "\n".join(lines)


def claim_on_predecessor(drive_root: Any, predecessor_task_id: str, nonce: str,
                         binding: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Bind this action on the predecessor BEFORE any effect; ``(claim, created)``.

    Same nonce: the recorded claim (replay). Another nonce while a claim
    exists: ``ValueError("already_continued:<successor>")``. A predecessor that
    turned ineligible refuses. Every refusal writes nothing.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema,
    )

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
        from ouroboros.control_actor import control_actor
        claim = {"by": control_actor(), "successor_task_id": binding["successor_task_id"], "action_nonce": nonce,
                 "binding": dict(binding), "binding_sha256": binding_sha(binding),
                 "bound_at": utc_now_iso(), "state": "bound"}
        outcome.update(claim=claim, created=True)
        return stamp_task_result_schema({**current, "continued_by": claim})

    patch_document(pathlib.Path(drive_root), predecessor_task_id, update)
    return dict(outcome["claim"]), bool(outcome["created"])


def mark_claim_admitted(drive_root: Any, predecessor_task_id: str, nonce: str) -> None:
    """Record on the predecessor that its claimed successor is durably admitted."""
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema,
    )

    def update(current: dict) -> Optional[dict]:
        require_writable_task_result_schema(current)
        claim = current.get("continued_by") if isinstance(current.get("continued_by"), dict) else {}
        if str(claim.get("action_nonce") or "") != nonce or claim.get("state") == "admitted":
            return None
        return stamp_task_result_schema(
            {**current, "continued_by": {**claim, "state": "admitted", "admitted_at": utc_now_iso()}})

    patch_document(pathlib.Path(drive_root), predecessor_task_id, update)


def continuation_chain(drive_root: Any, predecessor_task_id: str, task_id: str) -> List[str]:
    """The verified roots from ``predecessor_task_id`` to ``task_id``, both included, or ``[]``.

    Read from the recorded claims only: each ``continued_by`` binding on an
    earlier root's own result names its successor (hash-verified), so a chain of
    Continues is followed claim by claim. Room, folder or root equality grants
    nothing, and an unreadable result is no binding (``delegate_continuation``).
    """
    from ouroboros.task_results import load_task_result

    current, target, seen = str(predecessor_task_id or ""), str(task_id or ""), set()
    chain: List[str] = []
    while current and target and current not in seen:
        seen.add(current)
        chain.append(current)
        try:
            result = load_task_result(drive_root, current, strict=True) or {}
        except Exception:
            return []
        claim = result.get("continued_by") if isinstance(result.get("continued_by"), dict) else {}
        binding = claim.get("binding") if isinstance(claim.get("binding"), dict) else {}
        successor = str(claim.get("successor_task_id") or "")
        if (not successor or binding.get("successor_task_id") != successor
                or claim.get("binding_sha256") != binding_sha(binding)):
            return []
        if successor == target:
            return [*chain, target]
        current = successor
    return []


def recorded_continuation(drive_root: Any, predecessor_task_id: str, task_id: str) -> bool:
    """The existing yes/no form of the verified explicit-Continue chain."""
    return bool(continuation_chain(drive_root, predecessor_task_id, task_id))


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
