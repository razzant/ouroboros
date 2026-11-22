"""The owner's Continue of interrupted or explicitly unfinished work: identity, eligibility, sources.

Owner Batch4 (1A/4A/8A): Continue is a NEW root in the same room and folder that reads the earlier work; it is
never a resurrection of the old task, never offered after the owner's own Stop, a Panic or a finished answer, and it
adopts no old children: it may only decide a direct child's finished native result (apply or verify it, accept a
shared one as it stands, or reject it) through ``integrate_subagent_patch`` under the recorded relation. The
successor model decides whether and how to go on (LLM-first) and asks in the existing conversation when unsure — the
host carries facts, not a policy. This module is the protocol's data half; the queue admission is
``supervisor/continuation_admission.py``.

- IDENTITY: the successor id is derived from ``(verb, predecessor, action nonce)``; the nonce is the client's, kept
  across retry and reload. The id is a name, not the proof: the full binding (predecessor, nonce, successor,
  destination, group, token) is recorded and compared on every replay.
- CLAIM before any effect: the binding is written on the predecessor's result (``continued_by``, under its own lock)
  before the successor row, the queue or any workspace effect; a write that fails refuses with nothing done. One
  predecessor has one Continue successor: a later press with another nonce is answered with the accepted successor (a
  stale card points to it); new intentional work goes through the ordinary conversation.
- ELIGIBILITY from typed facts only (``continuation_eligibility``): a settled
  ROOT whose recorded cause is a technical interruption — the owner Restart's
  or a graceful shutdown's cancel origin (the stop door records it,
  ``worker_pool_lifecycle._write_failure_result``), a boot's restore fence, an
  infrastructure reason code, or a technical execution limit, including a
  forced finalization that still delivered a best-effort answer; or the
  author's own ``finish_task(action="stop")`` (``review_records.recorded_stop_rationale``),
  only when asked, never automatically. An owner Stop, a Panic, a finished
  answer, the work's own hard deadline or money limit, a live task or a cause
  that was never recorded refuse — the host never guesses a cause from prose.
  An explicit API/CLI execution budget (#939) is the same work's: the successor
  keeps its total and starts from the execution its predecessor closed
  (``execution_carry``); a spent budget refuses like a deadline, an unrecorded
  total refuses as unknown, never as a new quota.
- SOURCES kept apart: the exact original owner message and every later owner-authored mailbox row (read or unread) are
  the owner's; the predecessor's own result is its authored note; unread mail from tasks is peer context. A required
  owner source that cannot be read refuses (a gap is named, never reconstructed as owner text). A root no owner
  message opened is continued from the opening its producer recorded, never as owner words (``owner_sources``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.answer_facts import (card_input, forwarded_bubble_matches, option_answer, press_record,
                                   retained_card_input, row_source, submitted_fields_line)
from ouroboros.owner_words import (OWNER_CARD_INPUT, OWNER_PRESS, OWNER_SOURCES, SCHEDULE_ITEM_IDS, SCHEDULE_TEXT, answered_card,
                                   answered_card_reply, attachments_line, carded, reply_target, scheduled_text_items)
from ouroboros.review_evidence_sections import renders_exactly
from ouroboros.utils import utc_now_iso
from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
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
    # is not retried automatically (replaying would repeat effects): an explicit
    # Continue is exactly its manual path (#1543), a new root from this one's notes,
    # never its saved state, which retires with the work (prune_worked_copies).
    "worker_crash_owner_wait", "worker_crash_budget_pausing",
    # A positively reconciled orphan is a technical interruption, not an
    # unrecorded cause. Continue still rechecks owner sources and live custody.
    "orphaned_running_after_worker_restart", "interrupted_retry_lost",
    # A billing or dispatch-fence authority that could not be read, where no exact pause was saved.
    "billing_authority_unreadable", "control_authority_unreadable", "output_recovery_exhausted",
    # A running direct turn's accounting wait outlived its idle window (``_usage_wait``): storage, not
    # the work, ended it; Continue is its manual path, never an automatic retry.
    "accounting_wait_expired",
    # A pooled start its own context capture refused: a known window that cannot hold it, or a handbook the task
    # itself requires that could not be read (the producer's own cause; other authority refusals keep theirs). The
    # successor keeps that requirement; a larger route, the Context mode or a new task without it is the caller's choice.
    "context_capacity", "required_reference_unavailable",
    # The size recovery's own forecast refused the last candidate (a provider's size refusal ends as llm_api_error,
    # already here): the window, not a decision about the work, ended it. The successor starts from the window the
    # route has learned since (P1, 15.11.2026); Continue stays the owner's explicit act, never an automatic retry.
    "context_overflow",
})
# The loop's forced-finalization rails that are technical limits: an extracted
# best-effort answer settles ``completed``, the host fallback ``failed``
# (``outcomes.derive_loop_outcome``), so the rail, not the status word, decides.
EXECUTION_LIMIT_RAILS = frozenset({"round_limit", "finalization_grace"})
AUTHOR_STOP_CAUSE = "author_stop"  # a completed row its own author stopped as unfinished (no rail or cancel ended it)
# The work's own hard bounds (its deadline, its money): a new task id never
# extends them, so a task they ended is not offered Continue.
HARD_LIMIT_REASON_CODES = frozenset({"deadline", "deadline_local", "budget_exhausted"})
OWNER_SOURCE_KINDS = frozenset({"owner_text", "quiz_answer", "quiz_closed", "card_input"})
# The retained owner corpus (``review_evidence.task_inputs``) labels each row by its recorder's typed
# ``source`` (``loop_messages._record_owner_directive``): my human's later words (the one owner list less
# the first text; the successor's seeding set), the run's first text (a projection of the original, never a
# later message), a task's words and a confirmed account's place words (context; an order only by cite). A row whose
# label is none of these is not promoted to owner text. The door's labels in words carried by value (a follow-up's, a
# drafted start's ``owner_corpus``) are my human's.
CORPUS_INITIAL_SOURCES = frozenset({"origin_message", "initial_user", "initial_text", "initial_text_transcript"})
CARD_ANSWER_SOURCES = frozenset({"owner_quiz_answer", OWNER_PRESS})  # an answer to a card: words, or a press (an act)
CORPUS_OWNER_SOURCES = OWNER_SOURCES - CORPUS_INITIAL_SOURCES
CARRIED_DOOR_SOURCES = frozenset({"origin_message", "initial_user"})
CORPUS_TASK_SOURCES = frozenset({"principal_task_message", "relayed_peer_message"})
CORPUS_PLACE_SOURCES = frozenset({"place_confirmed"})  # my human's confirmed account in a place: provenance, whole
# A schedule's text under its recorded supplier (``owner_words.SCHEDULE_TEXT``): the work's assignment and conditions,
# carried whole to every successor and from it to its helpers, never owner text.
CORPUS_SCHEDULE_SOURCES = frozenset({SCHEDULE_TEXT})
# The typed ids an owner-source row may carry, each a string when present: its message, a task message's sender and
# relaying sibling, an owner answer's card, the chat row it is and that id's room, a place message's provider
# identity (``owner_words.place_row_facts``); its ``reply_to`` rides beside them.
PLACE_ROW_IDS = ("conversation_id", "message_id")  # a confirmed place row's provider identity, read again at a cite
ROW_IDS = ("msg_id", "source_task_id", "relayed_from_task_id", "quiz_id", "client_message_id", "chat_id", "row_sha256", "relay_source_root", *PLACE_ROW_IDS)


def row_identity(row: Dict[str, Any]) -> str:
    """The verified row identity, or the delivery identity kept by an older carrier."""
    if row.get("row_sha256"):
        return f"row-sha256:{row['row_sha256']}"
    client = str(row.get("client_message_id") or "")
    return str(row.get("msg_id") or "") or (f"client {row.get('chat_id') or ''} / {client}" if client else "")


def _same_opening(row, ref, address, original, words, index, *, typed=False):
    if row.get("row_sha256") and ref.get("row_sha256"):
        return row["row_sha256"] == ref["row_sha256"]
    msg_id = row.get("msg_id") or ""
    if msg_id.startswith("row:"):
        return msg_id == address
    if row.get("client_message_id"):
        if typed and not ref.get("row_sha256"):
            return (row["client_message_id"] == ref.get("client_message_id")
                    and str(row.get("chat_id")) == str(ref.get("chat_id")))
        return _same_opening_client(row, ref, original, words)
    if msg_id:
        return msg_id == address
    # The start writer put its anonymous original first; a card never takes an anonymous identity.
    return not typed and not (row.get("carried") or index or row.get("source") != "origin_message")


def _same_opening_client(row, ref, original, words):
    """A client-only carrier needs a matching room, time and payload to stand for an exact opening."""
    if row["client_message_id"] != str(ref.get("client_message_id") or ""):
        return False
    room, stamp = row.get("chat_id"), row.get("ts")
    if ref.get("row_sha256"):
        return (room is not None and str(room) == str(ref.get("chat_id")) and bool(stamp) and stamp == ref.get("ts")
                and words == original.get("content")
                and ("attachments" not in row or row["attachments"] == (original.get("attachments") or [])))
    return (room is None or ref.get("chat_id") is None or str(room) == str(ref["chat_id"])) and stamp in (None, ref.get("ts"))


# The binding's copy of the work's explicit execution budget (``execution_carry``).
EXECUTION_CARRY_KEYS = ("execution_timeout_sec", "execution_consumed_sec")
WAKE_OPENING = "consciousness_wake"
API_OPENING = "api_request"
# The request words the API door saved on the root (``gateway/tasks._complete_api_task_admission``).
API_REQUEST_FIELDS = ("description", "context", "expected_output", "constraints")


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


def execution_carry(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The work's explicit execution budget for its Continue (#939), ``None`` when it has none. ``execution_consumed_sec``
    is the total the predecessor's last attempt closed (``model_wait.record_execution_close``; attempt 0, the opening
    its admission recorded, for a row that never started), or ``None`` when no close names that attempt: unknown,
    never zero, a fresh budget or no limit."""
    from ouroboros.model_wait import EXECUTION_CLOSE_KEY
    from ouroboros.runtime_limits import explicit_execution_timeout_sec

    total = explicit_execution_timeout_sec(result)
    if total is None:
        return None
    close = result.get(EXECUTION_CLOSE_KEY) if isinstance(result.get(EXECUTION_CLOSE_KEY), dict) else {}
    sec, attempt = close.get("sec"), close.get("attempt")
    try:
        last = int(result.get("task_attempt") or 0)
    except (TypeError, ValueError):
        last = -1
    known = (type(sec) in (int, float) and math.isfinite(sec) and sec >= 0
             and type(attempt) is int and attempt == last)
    return {"execution_timeout_sec": total, "execution_consumed_sec": float(sec) if known else None}


def recorded_task_type(result: Dict[str, Any]) -> str:
    """The kind of work the record's task contract names (``build_task_contract``: the running write and the final
    store carry it), else the admission metadata's copy; ``""`` when neither names one."""
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    for contract in (result.get("task_contract"), metadata.get("task_contract")):
        if isinstance(contract, dict) and contract.get("task_type"):
            return str(contract["task_type"])
    return ""


def continuation_eligibility(result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """``{"eligible": bool, "cause": str, "refusal": str}`` from typed facts only.

    An older row's custody stamp is read beneath (``outcomes.recorded_cause``); a
    stamped row whose cause was never recorded still refuses, naming the stamp."""
    from ouroboros.outcomes import recorded_cause
    from ouroboros.review_records import recorded_stop_rationale
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
    author_stopped = status == "completed" and recorded_stop_rationale(result) is not None
    if status == "completed" and reason not in EXECUTION_LIMIT_RAILS and not author_stopped:
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
    carry = execution_carry(result)
    consumed = (carry or {}).get("execution_consumed_sec")
    if consumed is not None and consumed >= carry["execution_timeout_sec"]:
        return {"eligible": False, "cause": "execution_deadline", "refusal": "hard_limit_reached"}
    if source in TECHNICAL_CANCEL_SOURCES or str(origin.get("reason") or "") == "server_shutdown":
        cause = source or "server_shutdown"
    elif not origin and reason in TECHNICAL_REASON_CODES:
        cause = reason
    elif author_stopped:
        cause = AUTHOR_STOP_CAUSE
    else:
        return {"eligible": False, "cause": source or reason or str(result.get("reason_code") or ""),
                "refusal": "interruption_cause_unrecorded"}
    if carry is not None and consumed is None:
        return {"eligible": False, "cause": cause, "refusal": "execution_budget_unknown"}
    if cause == AUTHOR_STOP_CAUSE and deadline and seconds_until(deadline) is None:
        return {"eligible": False, "cause": cause, "refusal": "deadline_unknown"}
    # Where Continue would be offered: a deep self-review saves no midpoint, so a successor would be an ordinary task
    # (a finished or owner-stopped one keeps its own refusal above).
    if recorded_task_type(result) == "deep_self_review":
        return {"eligible": False, "cause": "deep_self_review", "refusal": "deep_review_restart_required"}
    return {"eligible": True, "cause": cause, "refusal": ""}


def _exact_rows(lines: List[Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """The JSON-object rows of mailbox lines, each with its exact text, and whether every line was one."""
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(entry, dict):
            rows.append({**entry, "_exact_row": line})
    return rows, len(rows) == len(lines)


def _mail_rows(drive_root: pathlib.Path, task_id: str) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """Every owner and peer row of the mailbox (read AND unread) in order, whether the read was whole, whether it exists."""
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return [], True, False
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [], False, True
    rows, whole = _exact_rows(mailbox_lines(content))
    return rows, whole and (not content or content.endswith("\n")), True


def _retained_start_quote(root: pathlib.Path, drafter: str, address: str) -> Optional[dict]:
    """A drafter's exact addressed relay can outlive the room row, including a quote made from a carried field."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_relay import KIND_RELAY, owner_row_address
    from ouroboros.task_custody import own_child_drives
    from ouroboros.task_store import read_document

    held = ((read_document(root, drafter) or {}).get("held_owner_messages") or {}).get(address)
    if isinstance(held, dict) and owner_row_address(held) == address:
        return held
    for drive in [root, *own_child_drives(root, drafter)]:
        for entry in drain_owner_entries(drive, drafter, include_acknowledged=True, kinds={KIND_RELAY}):
            if not entry.get("host_carried"):
                for quoted in entry.get("owner_rows") or []:
                    if isinstance(quoted, dict) and owner_row_address(quoted) == address:
                        return quoted
    return None


def _start_answer_source(card: dict, row: dict) -> Optional[str]:
    """The type the accepted answer proves, including any citation carried on the answer."""
    from ouroboros.answer_facts import answer_source
    from ouroboros.chronicle_quotes import quote_in
    from ouroboros.owner_quiz import late_answer_owner_text

    words = row["content"]
    if late_answer_owner_text(card) != words:
        return None
    source = answer_source(card)
    if "cited" in row or row.get("basis") == "cited":
        cite = row.get("cited")
        if not isinstance(cite, str) or not cite.strip() or row.get("basis") != "cited":
            return None
        if source == OWNER_PRESS:
            if cite.strip() != str(card["options"][card["answered_index"]]).strip():
                return None
        elif not quote_in(cite, words):
            return None
    return source


def _start_answer(root: pathlib.Path, task_id: str, row: dict, drafter: str) -> Optional[dict]:
    """Read a relay by its exact chat address, or a direct answer by its asker's delivery."""
    from ouroboros.answer_facts import answer_source
    from ouroboros.chat_chain import resolve_row
    from ouroboros.owner_relay import owner_row_identity, quotable_owner_row
    from ouroboros.owner_source import answered_quiz_block
    from ouroboros.owner_words import answered_card_source
    from ouroboros.question_answers import answer_message_id
    from ouroboros.store import StoreError

    asker, quiz_id, words, msg_id = (row.get(key) for key in ("source_task_id", "quiz_id", "content", "msg_id"))
    if not isinstance(msg_id, str) or not isinstance(words, str):
        return None
    physical = row.get("row_sha256")
    if msg_id.startswith("row:"):
        source_root = row.get("relay_source_root") or root
        try:
            quoted, _ = resolve_row(source_root, msg_id)
            quoted = quoted or _retained_start_quote(root, drafter, msg_id)
        except (StoreError, OSError, ValueError, TypeError):
            return None
        if not quoted or not quotable_owner_row(quoted, source_root) or row_source(quoted, "owner_mailbox", source_root) != row["source"]:
            return None
        if physical and physical != owner_row_identity(quoted)["row_sha256"]:
            return None
        block, asker = answered_card_source(quoted, source_root)
        quiz_id = (block or {}).get("quiz_id")
        if not block and isinstance(carried := quoted.get("corpus_answer"), dict):
            asker, quiz_id = carried.get("source_task_id"), carried.get("quiz_id")
            block = answered_quiz_block(source_root, asker, quiz_id) if asker and quiz_id else {}
            if (not block or answer_source(block) != row["source"]
                    or answered_card_reply(block, asker) != carried.get("reply_to")):
                return None
    else:
        if not all(isinstance(x, str) and x.strip() for x in (asker, quiz_id, msg_id)) or asker == task_id:
            return None
        block = answered_quiz_block(root, asker, quiz_id)
        delivery = block.get("delivery") or {}
        if msg_id not in (answer_message_id(asker, quiz_id), delivery.get("msg_id")):
            return None
        if physical and (answer_row := delivery.get("answer_row")):
            if not isinstance(answer_row, dict) or physical != owner_row_identity(answer_row)["row_sha256"]:
                return None
    card = answered_card(block, words)
    if (not card or not asker or not quiz_id or row.get("source_task_id", asker) != asker
            or row.get("quiz_id", quiz_id) != quiz_id
            or "reply_to" in row and row["reply_to"] != answered_card_reply(block, asker)):
        return None
    source = _start_answer_source(card, row)
    if source is None:
        return None
    return {"source": source, "content": words, "msg_id": msg_id, "source_task_id": asker,
            **{key: row[key] for key in ("basis", "cited", "row_sha256") if row.get(key)},
            "quiz_id": quiz_id, "card": card, **({"request_id": block["request_id"]} if block.get("request_id") else {}),
            **({"relay_source_root": row["relay_source_root"]} if row.get("relay_source_root") else {})}


def _start_citations(rows: list) -> tuple[list, dict]:
    """Fold an older citation's display address into the unique native quiz row it names.
    The native row still has to prove its delivery or relay in _start_answer."""
    from ouroboros.chronicle_quotes import quote_in
    from ouroboros.owner_words import row_ref

    rows = [dict(row) if isinstance(row, dict) else row for row in rows]
    native = [row for row in rows if isinstance(row, dict) and row.get("source") == "owner_quiz_answer"
              and row.get("source_task_id") and row.get("quiz_id") and isinstance(row.get("msg_id"), str)]
    kept, aliases = [], {}
    for row in rows:
        if (isinstance(row, dict) and row.get("source") == "owner_quiz_answer" and row.get("basis") == "cited"
                and row.get("carried") is True and isinstance(row.get("cited"), str) and row["cited"].strip()
                and isinstance(row.get("content"), str) and quote_in(row["cited"], row["content"])
                and set(row) <= {"source", "content", "msg_id", "cited", "basis", "carried"}):
            matches = [item for item in native if row_ref(item) == row.get("msg_id")]
            if len(matches) == 1 and matches[0].get("content") == row["content"]:
                matches[0].update(basis="cited", cited=row["cited"])
                aliases[row["msg_id"]] = (row["content"], True)
                continue
        kept.append(row)
    return kept, aliases


def _drafted_start(root: pathlib.Path, result: Dict[str, Any], task_id: str, metadata: Dict[str, Any],
                   gaps: List[str]) -> Optional[Dict[str, Any]]:
    """The start ``start_task`` recorded without an owner message (``origin_message_absent``).

    The objective is the drafting task's, carried as such and never as owner words. Each owner answer in the start's
    corpus must resolve on the asking task's own card, named by the identity the producer carried: the card records
    exactly these words, and the carried message is that card's answer (its live delivery id, which survives a later
    re-route, or the delivery the card records); otherwise the answer is a named gap. A drafted start without an
    answered card opens on its drafted objective alone (#1376): the drafting task's words, never owner words.
    """
    author, objective = metadata.get("objective_author"), result.get("objective") or result.get("description")
    if not (isinstance(author, dict) and author.get("kind") == "task" and isinstance(author.get("task_id"), str)
            and author["task_id"].strip() and isinstance(objective, str) and objective.strip()):
        gaps.append("start_objective_unreadable")
        return None
    answers: List[Dict[str, Any]] = []
    for row in metadata.get("owner_corpus") if isinstance(metadata.get("owner_corpus"), list) else []:
        if not isinstance(row, dict) or row.get("source") not in CARD_ANSWER_SOURCES:
            continue  # the corpus pass names a malformed row; other owner words stay later sources
        reply = row.get("reply_to") or {}
        if (isinstance(reply, dict) and reply.get("card_id") and reply.get("pressed_label") == row.get("content")
                and not row.get("quiz_id") and not str(row.get("msg_id") or "").startswith("quiz_answer:")):
            continue  # a living-card submission rides by value in the corpus, outside the local quiz protocol
        answer = _start_answer(root, task_id, row, author["task_id"])
        if answer is None:
            gaps.append("start_quiz_answer_unresolved")
        elif answer not in answers:
            answers.append(answer)
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
            and isinstance(answers, list)
            and all(isinstance(row, dict) and row.get("source") in CARD_ANSWER_SOURCES
                    and isinstance(row.get("card"), dict) and all(isinstance(row.get(key), str) and row[key].strip()
                            for key in ("content", "msg_id", "source_task_id", "quiz_id")) for row in answers))


def _retained_start(start: Any, gaps: List[str]) -> Optional[Dict[str, Any]]:
    """Read a chained start, recovering each answer's source from its saved card."""
    if not valid_start(start):
        gaps.append("start_source_malformed")
        return None
    answers = []
    for row in start["answers"]:
        card = row["card"]
        source = _start_answer_source(card, row)
        if source is None or card.get("quiz_id") != row["quiz_id"]:
            gaps.append("start_quiz_answer_unresolved")
        else:
            answers.append({**row, "source": source})
    return {**start, "answers": answers}


def _wake_opening(metadata: Dict[str, Any], task_id: str, gaps: List[str]) -> Optional[Dict[str, Any]]:
    """The opening a consciousness wake's own row recorded, copied by value, or a named gap: own clock, its
    ``wake_task_metadata`` reason; place focus, ``focus_metadata``'s facts (sender, ``owner_confirmed`` as the place
    recorded it), the event's exact words (``metadata.presence.observed_text``), the room's context, and what the
    connector reported about that message (``reported``, its labelled block; a cut names this task's record). None is
    owner text here: an addressed event from my human's confirmed account is its own ``place_owner`` corpus row."""
    from ouroboros.consciousness_authority import (
        WAKE_CAUSE_FOCUS, WORK_KIND_CONSCIOUSNESS, focus_evidence, is_consciousness_origin, wake_cause, work_kind,
    )
    from ouroboros.presence_message_facts import connector_facts, event_reader

    cause = wake_cause(metadata)
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    opening = {"source": WAKE_OPENING, "task_id": task_id, "wake_cause": cause}
    if cause == WAKE_CAUSE_FOCUS:
        opening.update(focus=copy.deepcopy(focus_evidence(metadata.get("focus"))), observed_text=presence.get("observed_text"))
        focus = opening["focus"] if isinstance(opening["focus"], dict) else {}
        actor = focus.get("actor") if isinstance(focus.get("actor"), dict) else {}
        event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
        reported = connector_facts(event.get("message"), text=str(presence.get("observed_text") or ""), sender=str(
            actor.get("display_name") or actor.get("platform_actor_id") or ""),
            source=event_reader(task_id, "message.non_text"))
        if reported:
            opening["reported"] = reported
    else:
        opening["wake_reason"] = metadata.get("wake_reason")
    if not (is_consciousness_origin(metadata) and work_kind(metadata) == WORK_KIND_CONSCIOUSNESS
            and valid_wake(opening)):
        gaps.append("wake_opening_unreadable")
        return None
    return opening


def valid_wake(wake: Any) -> bool:
    """A wake opening exactly as ``_wake_opening`` builds it (also its chained copy)."""
    from ouroboros.consciousness_authority import WAKE_CAUSE_FOCUS, WAKE_CAUSE_OWN_CLOCK

    if not (isinstance(wake, dict) and wake.get("source") == WAKE_OPENING
            and isinstance(wake.get("task_id"), str) and wake["task_id"].strip()):
        return False
    if wake.get("wake_cause") == WAKE_CAUSE_OWN_CLOCK:
        return isinstance(wake.get("wake_reason"), str) and bool(wake["wake_reason"].strip())
    focus = wake.get("focus")
    return (wake.get("wake_cause") == WAKE_CAUSE_FOCUS and isinstance(wake.get("observed_text"), str)
            and isinstance(wake.get("reported", ""), str)
            and isinstance(focus, dict) and isinstance(focus.get("actor"), dict)
            and all(isinstance(focus.get(key), str) and focus[key].strip() for key in ("place_id", "conversation_key")))


def _api_opening(result: Dict[str, Any], metadata: Dict[str, Any], task_id: str,
                 gaps: List[str]) -> Optional[Dict[str, Any]]:
    """The request the API door saved with its accepted receipt, copied by value, or a named gap. The words are its
    caller's under the channel it declared (``client_surface``, the door's one stamp of it); the receipt marks the
    opening, it proves no caller identity. Its inputs are the work's declared inputs (``declared_inputs``)."""
    receipt = result.get("api_admission")
    surface = metadata.get("client_surface") if isinstance(metadata.get("client_surface"), dict) else {}
    opening = {"source": API_OPENING, "task_id": task_id, "channel": surface.get("channel"),
               **{key: result[key] for key in API_REQUEST_FIELDS if isinstance(result.get(key), str) and result[key]}}
    if not (isinstance(receipt, dict) and receipt.get("status") == "accepted" and valid_api(opening)):
        gaps.append("api_request_unreadable")
        return None
    return opening


def valid_api(api: Any) -> bool:
    """An API opening exactly as ``_api_opening`` builds it (also its chained copy)."""
    return (isinstance(api, dict) and api.get("source") == API_OPENING
            and all(isinstance(api.get(key), str) and api[key].strip() for key in ("task_id", "channel", "description"))
            and all(isinstance(api.get(key, ""), str) for key in API_REQUEST_FIELDS))


def declared_inputs(predecessor_task_id: str, result: Dict[str, Any], sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The work's inputs as its direct predecessor holds them, for the successor's own copy (``[]``: none): its own
    manifest (contract, canonical over the stored copy, else its row), then what its owner messages attached and its
    relays quoted (``owner_sources``), each verified where held: its own stores, canonical first, the one hop its lineage
    read admits (``artifacts.held_attachment_manifest``), and saved chat uploads (``owner_relay.quoted_inputs``). Bytes
    no longer held stay a ``source_missing`` gap; an unreadable manifest refuses ``continuation_inputs_unavailable``."""
    from ouroboros.artifacts import held_attachment_manifest
    from ouroboros.owner_relay import quoted_inputs

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    authority = next((source for source in (result.get("task_contract"), metadata.get("task_contract"), result) if
                      isinstance(source, dict) and (source.get("attachment_manifest") or source.get("attachment_manifest_ref"))), None)
    try:
        held = held_attachment_manifest(sources["drives"], predecessor_task_id, authority) if authority else []
        for entry in sources["attached"]:
            held += held_attachment_manifest(sources["drives"], predecessor_task_id, entry)
        return held + quoted_inputs(sources["drives"][0], sources["quoted"])  # the canonical root holds chat uploads
    except ValueError as exc:
        raise ValueError("continuation_inputs_unavailable") from exc


def inputs_text(predecessor_task_id: str, copied: List[Dict[str, Any]], authority: Dict[str, Any]) -> str:
    """The successor's own input block (``artifacts.render_attachment_lines``) and where it came from."""
    from ouroboros.artifacts import render_attachment_lines

    lost = "; ".join(f"{row.get('label')} ({row.get('size')} bytes, sha256 {row['sha256']})" for row in copied
                     if row.get("status") == "rejected" and row.get("reason") == "source_missing" and row.get("sha256"))
    return (f"The work's inputs, copied from task {predecessor_task_id} into this task's own artifact store; "
            f"each copy matched the identity its row records:\n[ATTACHMENTS]\n{render_attachment_lines(authority)}\n"
            "[END_ATTACHMENTS]" + (f"\nMissing (their recorded bytes were gone when a Continue copied this work's "
                                   f"inputs): {lost}." if lost else ""))


def _keep_schedule_text(kept: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    """One ``SCHEDULE_TEXT`` row per schedule, field and words, under the supplier the host recorded for it."""
    row = {"source": SCHEDULE_TEXT, "content": item["content"],
           **{key: str(item[key]) for key in SCHEDULE_ITEM_IDS if item.get(key)}}
    if not any((old.get("schedule_id"), old.get("field"), old["content"])
               == (row.get("schedule_id"), row.get("field"), row["content"]) for old in kept):
        kept.append(row)


def occurrence_schedule_text(result: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    """The text a scheduled occurrence was accepted with (its frozen admission receipt) as ``SCHEDULE_TEXT`` rows;
    empty for work no schedule admitted: a successor's copy rides its chained corpus (``owner_corpus``)."""
    from ouroboros.schedule_continuation import admission_basis, valid_basis
    kept: List[Dict[str, Any]] = []
    if valid_basis(admission_basis(result, task_id)):
        frozen = result["schedule_admission"]["task"]
        for item in scheduled_text_items(frozen.get("metadata"), frozen.get("task_contract")
                                         or (frozen.get("metadata") or {}).get("task_contract")):
            _keep_schedule_text(kept, item)
    return kept


def _restore_original(root, original, snapshot, held_messages):
    """Identify a retained opening without a client id. A verified file placeholder counts as an opening with no words."""
    from ouroboros.project_dialogue import (opening_input, opening_words, _text_sha256,
                                           owner_message_ref_is_valid, resolve_owner_message_source)
    from ouroboros.owner_relay import origin_files, owner_row_address
    from ouroboros.owner_words import _held_origin
    from ouroboros.store import StoreError

    if not original:
        return False, None
    ref = original.get("origin_message_ref")
    content = original.get("content")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    reply = snapshot.get("reply")
    address = original.get("msg_id") or (reply.get("address") if isinstance(reply, dict) else None)
    if owner_message_ref_is_valid(ref):
        row = _held_origin(ref, held_messages)
        if not row and (not address or not original.get("reply_to")):
            try:
                row = resolve_owner_message_source(root, ref)
            except (OSError, StoreError):
                pass  # retained text or a recorded placeholder still stands
        item = retained_card_input(row, row.get("_owner_source_root") or root) if row else {}
        if row:
            address = owner_row_address(row)
        if (not item or item["source"] == OWNER_CARD_INPUT) and original.get("input_source") in (None, OWNER_CARD_INPUT):
            if typed := opening_input(None, root, retained=snapshot.get("input")):
                item = opening_words({"ref": ref, "text": content, "source_input": typed})
        if item and (item["source"] != OWNER_CARD_INPUT or original.get("input_source") in (None, OWNER_CARD_INPUT)):
            original.update(input_source=item["source"], words=item["content"],
                            **{key: item[key] for key in ("reply_to", "source_task_id", "quiz_id") if key in item})
        if not ref["client_message_id"] and not original.get("msg_id"):
            if address:
                original["msg_id"] = address
        files = snapshot.get("attachments") or (origin_files(row, root) if row else [])
        if files:
            original["attachments"] = files
        placeholder = snapshot.get("placeholder_text") or ((row or {}).get("text") if (row or {}).get("text_placeholder") else None)
        if content in (None, "", placeholder) and isinstance(placeholder, str) and placeholder and ref["text_sha256"] == _text_sha256(placeholder):
            original.update(content="", placeholder_text=placeholder)
            return True, address
    return isinstance(content, str) and bool(content.strip()), address


def _predecessor_mail(root, task_id, result, container, gaps):
    """Read retained and live mailbox rows; keep the drives for later file recovery."""
    from ouroboros.task_custody import own_child_drives

    rows, complete, live, drives = [], True, False, [root]
    try:
        drives += own_child_drives(root, task_id)
        for drive in drives:
            entries, whole, present = _mail_rows(drive, task_id)
            rows.extend(entries)
            complete = complete and whole
            live = live or present
    except (OSError, ValueError):
        complete = False
    if not complete:
        gaps.append("owner_mailbox_unreadable")
    captured_owner = container(result, "owner_mailbox", dict, "retained_mailbox_malformed")
    if not live and "owner_mailbox" not in result:
        gaps.append("owner_mailbox_uncaptured")
    elif not live and captured_owner.get("read_complete") is not True:
        gaps.append("owner_mailbox_capture_incomplete")
    unread = container(result, "unread_mailbox", dict, "retained_mailbox_malformed")
    held, whole = _exact_rows([row for held in (captured_owner, unread)
                               for row in container(held, "rows", list, "retained_mailbox_malformed")])
    if not whole:
        gaps.append("retained_mailbox_row_malformed")
    # A terminal capture outlives cleanup, but deferred history may have received later mail.
    return held + rows, drives


def _chained_row(row: Any, text_key: str, source_key: str, sources: Optional[frozenset[str]], *,
                 later: bool) -> Tuple[Optional[dict], str]:
    """Copy a retained row, checking a later quiz answer against the accepted card saved with it."""
    if not (isinstance(row, dict) and isinstance(row.get(text_key), str)
            and all(isinstance(row.get(field, ""), str) for field in (source_key, "words", *ROW_IDS))):
        return None, "chained_owner_source_malformed"
    source = row.get(source_key, "")
    if sources is not None and source not in sources:
        return None, "owner_source_provenance_unknown"
    kept = dict(row)
    if later and "card" in row:
        card = row["card"]
        if (not isinstance(card, dict) or not row["content"].strip()
                or not row.get("quiz_id") or card.get("quiz_id") != row["quiz_id"]):
            return None, "chained_owner_source_malformed"
        proven = _start_answer_source(card, row)
        if proven is None:
            return None, "chained_owner_source_malformed"
        # A verified comment already typed as words keeps its recorded source.
        if proven == OWNER_PRESS or source == OWNER_PRESS:
            kept["source"] = proven
    return kept, ""


def owner_sources(drive_root: Any, result: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """The predecessor's owner sources, authored note pointer and peer context, apart.

    The opening is one recorded state: the owner's message (``original``), a frozen ``schedule``, ``start``
    (``start_task`` recorded none, ``_drafted_start``), ``wake`` (a consciousness wake's own clock or place focus,
    ``_wake_opening``), ``api`` (the request the task API saved, ``_api_opening``), or none: a lost or legacy source
    and a gap. Only the producer's own fact marks absence (never an empty field or a false ``owner_ingress``), and a
    wake, start or API opening never seeds owner words. ``gaps`` names every unreadable required source and every
    row its producer never writes (a non-text body, an unknown source label): admission refuses on any gap rather
    than reconstructing, promoting or silently dropping owner text.
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
    from ouroboros.consciousness_authority import wake_cause
    from ouroboros.schedule_continuation import admission_basis, valid_basis
    schedule = chained.get("schedule") or admission_basis(result, task_id)
    if schedule is not None and not valid_basis(schedule):
        gaps.append("schedule_source_malformed")
        schedule = None
    chained_start, chained_wake, chained_api = chained.get("start"), chained.get("wake"), chained.get("api")
    origin_ref = result.get("origin_message_ref", metadata.get("origin_message_ref", {}))
    origin_words = result.get("origin_message_text", metadata.get("origin_message_text"))
    original: Optional[Dict[str, Any]] = None
    if "owner_sources" in continuation and not ((schedule or any(opening is not None for opening in (
            chained_start, chained_wake, chained_api))) and chained.get("original") is None):
        original = dict(container(chained, "original", dict, "chained_owner_source_malformed"))
    elif origin_ref or origin_words:
        original = _opening_message(root, origin_ref, origin_words, metadata)
    absent = metadata.get("origin_message_absent") is True
    aliases = {}
    if absent and isinstance(metadata.get("owner_corpus"), list):
        corpus, aliases = _start_citations(metadata["owner_corpus"])
        metadata = {**metadata, "owner_corpus": corpus}
    if absent and (origin_ref or origin_words
                   or metadata.get("origin_suppressed")):
        gaps.append("opening_owner_message_conflict")  # the producer never writes both
    start: Optional[Dict[str, Any]] = None
    wake: Optional[Dict[str, Any]] = None
    api: Optional[Dict[str, Any]] = None
    snapshot = metadata.get("origin_message_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    if isinstance(origin_ref, dict) and origin_ref.get("row_sha256") and not snapshot.get("input"):
        from ouroboros.projects_registry import project_binding_for_task

        bound = project_binding_for_task(root, task_id) or {}
        if bound.get("source_ref") == origin_ref and bound.get("source_input"):
            snapshot = {**snapshot, "input": bound["source_input"]}
    has_opening, original_address = _restore_original(root, original, snapshot, result.get("held_owner_messages"))
    if not (schedule or has_opening):
        if chained_start is not None:  # a Continue of a Continue keeps the start it was admitted on
            start = _retained_start(chained_start, gaps)
        elif chained_wake is not None:  # ...and the wake and API openings likewise
            wake = chained_wake if valid_wake(chained_wake) else None
            if wake is None:
                gaps.append("wake_source_malformed")
        elif chained_api is not None:
            api = chained_api if valid_api(chained_api) else None
            if api is None:
                gaps.append("api_source_malformed")
        elif absent:
            if "opening_owner_message_conflict" not in gaps:
                start = _drafted_start(root, result, task_id, metadata, gaps)
        elif wake_cause(metadata):
            wake = _wake_opening(metadata, task_id, gaps)
        elif "api_admission" in result:
            api = _api_opening(result, metadata, task_id, gaps)
        else:
            gaps.append("original_owner_message_text")
    if original and (original.get("source") != "origin_message" or original.get("input_source") not in (None, OWNER_PRESS, OWNER_CARD_INPUT, "owner_mailbox", "owner_quiz_answer", "origin_message") or
                     not isinstance(original.get("origin_message_ref", {}), dict)):
        gaps.append("original_owner_source_malformed")
    later: List[Dict[str, Any]] = []
    peers, confirmed = [], []  # other tasks' words (counted, behind a reader); my human's confirmed place words (whole)
    attached, quoted = [], []  # an owner message's staged files; the chat uploads a relay quoted
    # The start's answers keep their one identity: a corpus copy is the same message.
    seen: Dict[str, Tuple[str, bool]] = {identity: (row["content"], True) for row in (start or {}).get("answers") or []
                                        for identity in (row["msg_id"], row_identity(row))}
    seen.update(aliases)
    opening_ref = (original or {}).get("origin_message_ref")
    if isinstance(opening_ref, dict) and opening_ref.get("row_sha256") and has_opening:
        seen[row_identity(opening_ref)] = (original.get("words", original["content"]), True)
    exact: Dict[str, str] = {}  # this task's own exact rows by msg_id: the mailbox pass fills it after reading drives
    delivery_rows: Dict[str, str] = {}

    schedule_text = occurrence_schedule_text(result, task_id)  # then the chained corpus's: one per schedule/field/words
    def named(identity: str, words: str, owner: bool, corpus: Optional[Dict[str, Any]] = None) -> Optional[bool]:
        """One message under one identity, compared by its words (#2941): None when first named (now recorded), True
        when its words and owner/peer side repeat, else False and a gap. A corpus row also stands for its exact twin
        as a late answer's frame or, recorded before ``words``, as the twin's own rendering (``renders_exactly``)."""
        # A physical row may also have an independent mailbox delivery id. The
        # mailbox pass reads that carrier by its own id, including staged files.
        delivery = (corpus or {}).get("msg_id") if (corpus or {}).get("source") in {"owner_mailbox", *CARD_ANSWER_SOURCES} else ""
        if (delivery and not delivery.startswith("row:") and identity != delivery
                and delivery_rows.setdefault(delivery, identity) == identity):
            if delivery in seen and identity not in seen:
                seen[identity] = seen[delivery]
            seen.setdefault(delivery, (words, owner))
            if delivery in exact:
                exact[identity] = exact[delivery]
        if not identity or identity not in seen:
            if identity:
                seen[identity] = (words, owner)
            return None
        if seen[identity] == (words, owner):
            return True
        twin = json.loads(exact[identity]) if corpus is not None and identity in exact else {}
        if twin and seen[identity][1] == owner:
            if isinstance(twin.get("late_answer"), dict) or ("words" not in corpus
                    and (renders_exactly(twin, corpus["content"], drives, task_id)
                         or twin.get("kind") == "card_input" and twin.get("text") == corpus["content"]
                         or forwarded_bubble_matches(twin, corpus["content"], root))):
                return True
        gaps.append("owner_message_identity_conflict")
        return False

    for gap in container(chained, "gaps", list, "chained_owner_source_malformed"):
        gaps.append(gap if isinstance(gap, str) and gap else "chained_owner_source_malformed")
    # Validate identities before deduplication: a dict comprehension would hide rivals.
    for key, target, text_key, source_key, sources in (
        ("later", later, "content", "source", CORPUS_OWNER_SOURCES | CARRIED_DOOR_SOURCES),
        ("confirmed_place", confirmed, "content", "source", CORPUS_PLACE_SOURCES),
        ("peer_context", peers, "text", "provenance", None),
    ):
        for row in container(continuation if key == "peer_context" else chained, key, list,
                             "chained_owner_source_malformed"):
            row, gap = _chained_row(row, text_key, source_key, sources, later=target is later)
            if gap:
                gaps.append(gap)
                continue
            if named(row_identity(row), row.get("words", row[text_key]), target is later, row) is None:
                target.append(row)
    quizzes = result.get("owner_quiz") if isinstance(result.get("owner_quiz"), dict) else {}

    def corpus_pass(rows: List[Any], malformed: str, *, carried: bool = False) -> None:
        """A recorded corpus under the one identity rule (a msg_id, else a client_message_id, names one text): the
        run's first text is the original's projection; a door row the start carried by value (``carried``) is my
        human's words unless it repeats that original; a task's words are context and an unknown label is never
        promoted to owner text."""
        texts = {True: {row.get("words", row["content"]) for row in later} | {str((original or {}).get("content") or "")},
                 False: {row["text"] for row in peers}}
        original_ref = opening_ref if isinstance(opening_ref, dict) else {}
        for index, row in enumerate(rows):
            # The recorder writes a non-empty text body, a source label, string ids and, beside a rendering, its words.
            if not (isinstance(row, dict) and isinstance(row.get("content"), str)
                    and (row["content"].strip() or row.get("attachments") and row.get("source") in CARRIED_DOOR_SOURCES | CORPUS_OWNER_SOURCES)
                    and all(isinstance(row.get(key, ""), str) for key in ("source", "words", *ROW_IDS))):
                gaps.append(malformed)
                continue
            msg_id, source, text = str(row.get("msg_id") or ""), str(row.get("source") or ""), row["content"]
            words = row.get("words", text)  # the exact text a rendered row was delivered as; the rendering stays
            client, identity = str(row.get("client_message_id") or ""), row_identity(row)
            if source in CORPUS_SCHEDULE_SOURCES:  # the assignment's text, carried whole beside the words, never theirs
                _keep_schedule_text(schedule_text, row)
                continue
            if source not in CORPUS_INITIAL_SOURCES | CORPUS_OWNER_SOURCES | CORPUS_TASK_SOURCES | CORPUS_PLACE_SOURCES:
                gaps.append("owner_source_provenance_unknown")
                continue
            door = source in CARRIED_DOOR_SOURCES and (carried or row.get("carried") or bool(client) or msg_id.startswith("row:"))
            if door and original is not None:  # a carried copy of the original's own message is that message
                door = not _same_opening(row, original_ref, original_address, original, words, index)
            # A typed form or answer may itself be the opening, but only under the opening's retained identity.
            form = isinstance(row.get("reply_to"), dict) and row["reply_to"].get("card_id")
            opening = bool(original and original_ref and (source in CARD_ANSWER_SOURCES | {OWNER_CARD_INPUT} or form)
                           and _same_opening(row, original_ref, original_address, original, words, index, typed=True))
            if source in CORPUS_INITIAL_SOURCES and not door or opening:
                if original is not None and isinstance(row.get("reply_to"), dict):
                    original.setdefault("reply_to", row["reply_to"])
                if opening and original is not None:
                    if original.get("input_source") == OWNER_CARD_INPUT and source != OWNER_CARD_INPUT:
                        original.update(input_source=source, words=words)
                        if row.get("reply_to"):
                            original["reply_to"] = row["reply_to"]
                    else:
                        if source in (OWNER_PRESS, OWNER_CARD_INPUT, "owner_mailbox", "owner_quiz_answer"):
                            original.setdefault("input_source", source)
                        original.setdefault("words", words)
                    for key in (*ROW_IDS, "basis", "cited"):
                        if row.get(key):
                            original.setdefault(key, row[key])
                continue
            owner = source in CORPUS_OWNER_SOURCES or door
            if named(identity, words, owner, row) is not None or not identity and words in texts[owner] and not door:
                continue
            texts[owner].add(words)
            if owner:
                ids = {key: row[key] for key in (*ROW_IDS[1:], "ts", "received", "reply_to", "attachments", "carried") if row.get(key)}
                # An answer to the predecessor's own card reads that card from its own record.
                card = (answered_card(quizzes.get(ids.get("quiz_id", "")), text)
                        if source in CARD_ANSWER_SOURCES and ids.get("source_task_id") == task_id else None)
                later.append({"source": source, "content": text, "msg_id": msg_id, **ids,
                              **({"card": card} if card else {}), **({"words": words} if "words" in row else {})})
            elif source in CORPUS_PLACE_SOURCES:  # carried whole, with its ids, provider identity and reply target
                confirmed.append({"source": source, "content": text, "msg_id": msg_id,
                                  **{key: row[key] for key in PLACE_ROW_IDS if row.get(key)},
                                  **({"reply_to": row["reply_to"]} if isinstance(row.get("reply_to"), dict) else {})})
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
                "start_owner_corpus_row_malformed", carried=True)
    rows, drives = _predecessor_mail(root, task_id, result, container, gaps)
    exact_seen: set = set()
    quoted_rows: set = set()
    from ouroboros.owner_quiz import late_answer_owner_text, secret_outcome_block
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
        exact_seen.add(entry["_exact_row"])
        if secret_outcome_block(entry) is not None:
            continue  # form outcomes remain recoverable task evidence, never fabricated owner words
        if kind == "quiz_answer" and (entry.get("quiz_delivery") or {}).get("answer_snapshot"):
            entry = {**entry, "text": late_answer_owner_text(entry["quiz_delivery"]["answer_snapshot"])}
        living = card_input(entry, root)
        if living:
            entry = {**entry, "text": living["content"]}
        files = kind in OWNER_SOURCE_KINDS and (entry.get("attachment_manifest") or "attachment_manifest_ref" in entry)
        if (same := named(msg_id, str(entry.get("text") or ""), kind in OWNER_SOURCE_KINDS)) is not None:
            if same and files and msg_id not in exact:  # a message an earlier source carried: its files are still ours
                exact[msg_id] = entry["_exact_row"]
                attached.append(entry)
            continue
        if msg_id:
            exact[msg_id] = entry["_exact_row"]  # a late answer's frame, or an earlier rendering, is proven against it
        if kind in OWNER_SOURCE_KINDS:
            row = {"source": row_source(entry, "owner_quiz_answer" if kind == "quiz_answer" else "owner_mailbox"),
                   "content": str(entry.get("text") or ""), "msg_id": msg_id,
                   "ts": str(entry.get("ts") or ""), "exact_row": entry["_exact_row"]}
            if living:
                row.update(living)
            if target := row.get("reply_to") or reply_target(root, entry.get("reply_reference")):  # the message these words reply to
                row["reply_to"] = target
            later.append(carded(row, entry, root))  # a live delivery's card, or a late answer's snapshot, beside the words
            if files:
                attached.append(entry)  # the files that message attached, staged in this task's own store
        elif kind == "relay_entry" and not entry.get("host_carried"):  # a carried row is the tree's, not this corpus
            for item, quoted_files in _relay_owner_inputs(root, entry):
                identity, address = row_identity(item), item["msg_id"]
                if address not in seen and identity not in seen and address != original_address:
                    later.append(item)
                    seen[address] = seen[identity] = (item["content"], True)
                if identity not in quoted_rows:
                    quoted_rows.add(identity)
                    quoted.extend(quoted_files)
        if kind in {"relay_entry", "task_message"} and not entry.get("host_carried"):  # a note and a task's words: peers
            peers.append({"msg_id": msg_id, "ts": str(entry.get("ts") or ""),
                          "source_task_id": str(entry.get("source_task_id") or ""),
                          "provenance": "independent_task" if kind == "relay_entry" else str(entry.get("provenance") or ""),
                          "text": str(entry.get("text") or "")})
    # The retained corpus keeps what the mailbox may no longer hold (a direct
    # turn's input, a row consumed before any capture) under the SAME identity
    # rules; an exact mailbox row outranks its projection (a rendered row's words, a
    # late answer's frame of that typed row or a pre-``words`` rendering: no rival text).
    evidence = container(result, "review_evidence", dict, "retained_owner_corpus_malformed")
    inputs = container(evidence, "task_inputs", dict, "retained_owner_corpus_malformed")
    unavailable = container(inputs, "unavailable_sections", list, "retained_owner_corpus_malformed")
    if any(not isinstance(section, str) for section in unavailable):
        gaps.append("retained_owner_corpus_malformed")
    if "owner_requirements_and_decisions" in unavailable:
        gaps.append("owner_corpus_capture_unavailable")
    corpus_pass(container(inputs, "owner_requirements_and_decisions", list, "retained_owner_corpus_malformed"),
                "retained_owner_corpus_row_malformed")
    export = next((str(row["path"]) for row in result.get("artifacts") or [] if isinstance(row, dict)
                   and row.get("kind") == "memory_export" and pathlib.Path(str(row.get("path") or "")).is_file()), "")
    from ouroboros.task_custody import task_artifact_stores
    readable = (*(drive / "task_drives" / task_id for drive in drives), *task_artifact_stores(root, task_id))
    return {"original": original, "schedule": schedule, "start": start, "wake": wake, "api": api, "later": later,
            "peer_context": peers, "confirmed_place": confirmed, "schedule_text": schedule_text,
            "gaps": list(dict.fromkeys(gaps)),
            "open_relays": result.get("open_relays", chained.get("open_relays", {})),
            "held_owner_messages": {**chained.get("held_owner_messages", {}), **result.get("held_owner_messages", {})},
            "attached": attached, "quoted": quoted, "drives": drives, "memory_export": export,
            "readable": ", ".join(str(path) for path in readable if path.is_dir()),
            "predecessor_note": {"task_id": task_id, "reader": "get_task_result",
                                 "kind": "predecessor_authored_note"}}


def _opening_message(root, ref, text, metadata: dict) -> dict:
    """Rebuild the opening from its own producer facts, including a submitted form or an answered quiz."""
    original = {"source": "origin_message", "content": text, "origin_message_ref": ref}
    if submitted := card_input(metadata, root, "origin_message"):
        if submitted["source"] != "origin_message":
            original["input_source"] = submitted["source"]
        original.update(words=submitted["content"], **{k: submitted[k] for k in ("reply_to", "source_task_id", "quiz_id") if k in submitted})
    return original


def _relay_owner_inputs(root, entry: dict):
    """Yield the new owner inputs and their uploads from an explicit relay, verified against each source store."""
    from ouroboros.owner_relay import origin_files, owner_row_address, owner_row_identity, owner_text, quotable_owner_row
    from ouroboros.owner_words import owner_place_accounts

    accounts_by_root = {}
    for row in entry.get("owner_rows") or []:
        source_root = row.get("_owner_source_root") or entry.get("relay_canonical_root") or root
        key = str(source_root)
        if key not in accounts_by_root:
            accounts_by_root[key] = owner_place_accounts(source_root)
        address = owner_row_address(row)
        if not quotable_owner_row(row, source_root, accounts=accounts_by_root[key]):
            continue
        item = card_input(row, source_root)
        if not item:
            carried = row.get("corpus_answer") or {}
            target = carried.get("reply_to") or reply_target(source_root, row.get("reply_to"))
            item = {"source": row_source(row, "owner_mailbox", source_root), "content": owner_text(row),
                    **{k: carried[k] for k in ("source_task_id", "quiz_id") if carried.get(k)},
                    **({"reply_to": target} if target else {})}
        files = [{**ref, "_owner_source_root": str(source_root)}
                 for ref in row.get("attachments") or [] if isinstance(ref, dict)]
        ids = {"msg_id": address, "ts": row.get("ts") or "", "row_sha256": owner_row_identity(row)["row_sha256"]}
        if refs := origin_files({**row, "_owner_source_root": source_root}, root):
            ids["attachments"] = refs
        yield carded({**item, **ids}, row, source_root), files


def continuation_room(drive_root: Any, predecessor_task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """The room a FRESH Continue is admitted to: ``{"chat_id", "project_id"}``.

    "Turn into project" writes only the immutable task<->Project binding, the one truth about a task's room (owner
    decision B4=A); the interrupted worker's own result keeps the chat and project copy it started with, so the
    binding outranks that copy and unbound work keeps its own row. Only the room moves: the original owner message,
    its ref and the prepared folder stay the predecessor's. An unreadable store or a malformed row raises, so the
    caller refuses before its claim instead of admitting the work to Main.
    """
    from ouroboros.projects_registry import project_binding_for_task

    bound = project_binding_for_task(drive_root, predecessor_task_id, strict=True)
    if not bound:
        return {"chat_id": result.get("chat_id"), "project_id": str(result.get("project_id") or "")}
    chat = bound.get("project_chat_id")
    if type(chat) is not int or chat <= 0:
        raise ValueError("Project binding is unavailable")
    return {"chat_id": chat, "project_id": bound["project_id"]}


def carried_execution_fields(binding: Dict[str, Any]) -> Dict[str, Any]:
    """The successor result's copy of the work's budget before any attempt of it starts (#939):
    a Continue of a successor that never ran hands the same closed total on."""
    if "execution_timeout_sec" not in binding:
        return {}
    from ouroboros.model_wait import execution_opening

    return {"execution_timeout_sec": binding["execution_timeout_sec"],
            **execution_opening(binding, binding["execution_consumed_sec"])}


def owner_corpus_rows(sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The successor's seeded owner corpus (``_initialize_owner_directives``): exact words with their typed ids, reply
    targets and a rendering's ``words``, the start's answers first; an answered card rides beside its answer as its reply target."""
    rows: List[Dict[str, Any]] = []
    original = sources.get("original") if isinstance(sources.get("original"), dict) else {}
    start = sources.get("start") if isinstance(sources.get("start"), dict) else {}
    ref = original.get("origin_message_ref") or {}
    opening = {**{key: str(ref[key]) for key in ("chat_id", "client_message_id", "ts", "row_sha256") if ref.get(key)},
               **original, "content": original.get("words", original.get("content")),
               "source": original.get("input_source", "origin_message")}
    for row in [opening, *(start.get("answers") or []), *(sources.get("later") or [])]:
        if isinstance(row, dict) and (str(row.get("content") or "").strip() or row.get("attachments")):
            card = answered_card_reply(row.get("card"), row.get("source_task_id"))
            rows.append({"source": row.get("source") or "owner_mailbox", "content": row["content"],  # a card wins
                         **{key: row[key] for key in (*ROW_IDS, "ts", "received", "reply_to", "attachments", "carried", "basis", "cited") if row.get(key)},
                         **({"reply_to": card} if card else {}),
                         **{key: row[key] for key in ("words",) if key in row}})  # exact, even empty
    for row in sources.get("schedule_text") or []:  # a schedule's text under its supplier, by value, never owner words
        if isinstance(row, dict) and str(row.get("content") or "").strip():
            rows.append({"source": SCHEDULE_TEXT, "content": row["content"],
                         **{key: row[key] for key in SCHEDULE_ITEM_IDS if row.get(key)}})
    for row in sources.get("confirmed_place") or []:  # provenance rows ride beside the owner's, outside the owner sources
        if isinstance(row, dict) and str(row.get("content") or "").strip():
            rows.append({"source": "place_confirmed", "content": row["content"],
                         **{key: row[key] for key in ("msg_id", *PLACE_ROW_IDS, "reply_to") if row.get(key)}})
    return rows


def _card_lines(row: Dict[str, Any]) -> List[str]:
    """One answered card for the successor: the asking task's question, then the owner's answer, a press as their
    act on that task's label (``answer_facts.option_answer``)."""
    card = row.get("card") if isinstance(row.get("card"), dict) else {}
    options = card.get("options") if isinstance(card.get("options"), list) else []
    index, comment = card.get("answered_index"), str(card.get("comment") or "")
    answer = (option_answer(card, index, comment, asker=row.get("source_task_id")) if type(index) is int
              else f"The owner wrote (verbatim): {comment}" if comment.strip()
              else f"The owner answered (verbatim): {row.get('content') or ''}")
    return [f"- Card {card.get('quiz_id') or row.get('quiz_id')} of task {row.get('source_task_id')}, "
            f"asked {card.get('asked_at') or 'at an unrecorded time'}: {card.get('question')}",
            f"  Options: {'; '.join(f'{n}. {label}' for n, label in enumerate(options, 1)) or 'none recorded'}",
            f"  {answer} (answered {card.get('answered_at') or 'at an unrecorded time'})."]


def recorded_system_checkout(result: Dict[str, Any]) -> Dict[str, str]:
    """The predecessor's recorded system-checkout observations, each dated by its own capture.

    The only per-task fact is the mutation baseline's ``system_repo`` surface: ``rev-parse HEAD`` at its
    ``captured_at``. The task's own attributed commit or an accepted fast-forward re-anchors it (new SHA, new time),
    so it is an observation at that moment — never the task's start, the code its process imported, or the checkout
    now. A retained terminal snapshot of the same surface is a second dated observation; a body candidate's row
    (``candidate``) is not the checkout (its base may be the copy's creation, not HEAD then). Nothing recorded: ``{}``.
    """
    evidence = result.get("mutation_evidence") if isinstance(result.get("mutation_evidence"), dict) else {}
    baseline = evidence.get("baseline") if isinstance(evidence.get("baseline"), dict) else {}
    surface = next((row for row in baseline.get("surfaces") or [] if isinstance(row, dict)
                    and row.get("surface_type") == "system_repo" and isinstance(row.get("git"), dict)
                    and row["git"].get("base_commit") and row.get("captured_at")
                    and not isinstance(row.get("candidate"), dict)), None)
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
                    checkout: Optional[Dict[str, str]] = None, successor_task_id: str = "",
                    drive_mode: str = "", stop_note: str = "", tree_note: str = "", work_name_open: bool = False) -> str:
    """The successor's first message: host-authored FACTS, the owner's words verbatim.

    It frames nothing as a decision: the model judges whether and how to go on and asks the owner in this
    conversation when it cannot tell; the request names no actor (an action nonce proves the action, not who
    pressed it). ``authority_source`` is the exact pointer the successor's start materializes as the predecessor's
    brief (or refuses to start), so the work order names it as the complete record's exact reader, never a second
    copy to fetch; without it the pointer is the only route to that result. Messages other tasks addressed to the
    predecessor ride as their count, size and senders plus the exact reader of the successor's own
    ``metadata.continuation.peer_context`` (``successor_task_id``), never as bodies inside the assignment.
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
    ended = "was explicitly stopped as unfinished by its author" if cause == AUTHOR_STOP_CAUSE else "was interrupted"
    lines = [
        f"[CONTINUE] Continue was requested for task {predecessor_task_id}, which {ended} "
        f"(recorded cause: {cause or 'technical interruption'}). This is a NEW task in the same "
        "conversation and folder; the old task is not resumed and its helpers are not yours to steer; with "
        "integrate_subagent_patch(task_id=...) a helper's FINISHED native result can be applied or verified (the "
        "default), accepted as it stands when it is a shared result (decision='accept' with your reason, never a "
        "verification) or rejected (a cancelled helper stays cancelled, but its ready retained capture is decided "
        "the same ways with your reason), "
        "and the old task's stopped delegated runs are yours to continue with delegate_start(subagent_id=..., "
        "continue_from=<run_id>, prompt=...) instead of redoing them.",
        *([tree_note] if tree_note else []),
        result_line,
        *([stop_note] if stop_note else []),
        "Decide whether and how to continue; if whether the work is still wanted is unclear, ask the "
        "owner in this conversation.",
        "Use manage_schedules(action='list') to inspect related or unknown future follow-ups. "
        "Continue does not release their holds. Resolve unknown relationships explicitly; restore "
        "only an observed hold_id. Original money and hard deadlines still apply to related work.",
    ]
    if deadline_at:
        lines.append(f"An explicit hard deadline still applies to this work: {deadline_at}.")
    if drive_mode:  # only what of the predecessor's exists and the lineage read reaches, by its real path
        export, folders = str(sources.get("memory_export") or ""), str(sources.get("readable") or "")
        memory = f"Its memory export is readable at {export}." if export else "It recorded no memory export."
        reach = f"its folders {folders}, until a prune removes them" if folders else "no folder of it now"
        lines.append(f"This task runs on its own {drive_mode} memory drive, as the interrupted task did; that task's "
                     f"drive files are not copied in. {memory} This task's lineage read (read_file, root task_drive or "
                     f"artifact_store) reaches {reach}, and never a drive's memory folder.")
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
        wording = original.get("words", original["content"])
        if original.get("input_source") == OWNER_PRESS:
            lines += ["", press_record(wording, original.get("source_task_id"), who="the owner")]
        elif original.get("input_source") == OWNER_CARD_INPUT:
            lines += ["", wording]
        else:
            lines += ["", "Owner's original instruction (verbatim):", str(wording)]
    elif original and original.get("content") == "":
        lines += ["", "The owner opened this work with a file and wrote no words."]
    if original:
        if files := attachments_line(original.get("attachments")):
            lines.append(files)
        target = original.get("reply_to") or {}
        if fields := submitted_fields_line(target):
            lines.append(fields)
        if isinstance(target, dict) and target.get("text") and str(target.get("address") or "").startswith(("quiz ", "card ")):
            lines += [f"Question from {target.get('author') or 'an unrecorded author'} ({target['address']}):", target["text"]]
    start = sources.get("start") if isinstance(sources.get("start"), dict) else None
    if start:
        drafter = (start.get("author") or {}).get("task_id") or "unknown"
        lines += ["", f"That task began without an owner message: task {drafter} drafted its objective "
                      "(the drafting task's words, not the owner's; verbatim):", str(start.get("objective") or "")]
        if start.get("expected_output"):
            lines.append(f"Expected output: {start['expected_output']}")
        if start.get("answers"):
            lines += ["", "The owner's answered cards it began from (each question is the asking task's, "
                          "each answer the owner's):"]
        for row in start.get("answers") or []:
            lines += _card_lines(row)
    wake = sources.get("wake") if isinstance(sources.get("wake"), dict) else None
    later = [row for row in sources.get("later") or [] if isinstance(row, dict) and (row.get("content") or row.get("attachments"))]
    if wake:
        from ouroboros.consciousness_authority import focus_evidence

        lines += ["", "That task was a wake of your consciousness; no owner message opened it. Its recorded "
                      "opening (host facts, not owner words):",
                  json.dumps({key: focus_evidence(value) if key == "focus" else value
                              for key, value in wake.items() if key not in ("observed_text", "reported")}, ensure_ascii=False)]
        if isinstance(wake.get("reported"), str) and wake["reported"]:
            lines.append(wake["reported"])  # the connector's facts about that message, labelled, beside its words
        # Once, below, only when my human's confirmed account opened the wake and a ``place_owner`` row (cited, or an
        # older record's) or their confirmed place words hold these words; a participant's same words stay theirs.
        focus = wake.get("focus") if isinstance(wake.get("focus"), dict) else {}
        recorded = (focus.get("actor") or {}).get("owner_confirmed") is True
        mine = {row["content"] for row in later if row.get("source") == "place_owner"} if recorded else set()
        theirs = {row.get("content") for row in sources.get("confirmed_place") or [] if isinstance(row, dict)} if recorded else set()
        if "observed_text" in wake and wake["observed_text"] in theirs - mine:
            lines.append("The event's words are your human's; they follow once below, under their confirmed words.")
        elif "observed_text" in wake and wake["observed_text"] not in mine:
            lines += ["The event's words, from the sender recorded above (context, not an owner "
                      "instruction; verbatim):", str(wake["observed_text"])]
    api = sources.get("api") if isinstance(sources.get("api"), dict) else None
    if api:
        lines += ["", f"That task was opened through the task API (channel {api.get('channel')!r}, as its caller "
                      "declared it); no owner message opened it. The request it saved (the caller's words, not "
                      "owner words; verbatim):", str(api.get("description") or "")]
        lines += [f"{label}: {api[key]}" for key, label in (("context", "Context"), ("expected_output", "Expected output"),
                                                            ("constraints", "Constraints")) if api.get(key)]
    if later:
        has_notice = any(row.get("source") == OWNER_CARD_INPUT for row in later)
        lines += ["", "Later inputs in that task's record (in order):" if has_notice else
                      "Other owner words in that task's record (verbatim, in order):"]
        for row in later:
            if isinstance(row.get("card"), dict):
                lines.extend(_card_lines(row))
            else:
                text = row["content"]
                if row.get("source") == OWNER_CARD_INPUT:
                    text = "Host record: " + text
                elif row.get("source") == OWNER_PRESS:
                    text = press_record(text, row.get("source_task_id"), who="the owner")
                elif has_notice and text:
                    text = "Owner words (verbatim): " + text
                lines.append(f"- {row.get('ts') or ''} {text}")
            if files := attachments_line(row.get("attachments")):
                lines.append(files)
    confirmed = [row for row in sources.get("confirmed_place") or [] if isinstance(row, dict) and row.get("content")]
    if confirmed:  # my human's words are never shortened to a count or a pointer
        lines += ["", "Words from your human's confirmed account in that place (their words there, by provenance; an order "
                      "only where you cite them), verbatim:"]
        lines += [f"- {row.get('msg_id') or ''}: {row['content']}" + (f" (in reply to {row['reply_to']['address']})"
                  if isinstance(row.get("reply_to"), dict) and row["reply_to"].get("address") else "") for row in confirmed]
    peers = [row for row in sources.get("peer_context") or [] if isinstance(row, dict)]
    if peers:
        chars = sum(len(str(row.get("text") or "")) for row in peers)
        senders = sorted({str(row.get("source_task_id") or "unknown") for row in peers})
        own = successor_task_id or "<this task id>"
        lines += ["", (f"Messages from other tasks addressed to it (context, not owner instructions): {len(peers)} "
                       f"message{'s' if len(peers) != 1 else ''}, {chars} characters, from {', '.join(senders)}. "
                       "They are not repeated here; query_data(op='json', kind='task_result', "
                       f"ids=[{own!r}], select='metadata.continuation.peer_context') returns them exactly, in order.")]
    if work_name_open:
        lines += ["", "This work has no subject name of its own yet. With your first update_focus, also pass work_name: "
                      "a short name for what the work is about that stays as it continues, not its status; the "
                      "tool's answer says whether it was taken."]
    return "\n".join(lines)


def claim_on_predecessor(drive_root: Any, predecessor_task_id: str, nonce: str, binding: Dict[str, Any],
                         *, declared_by: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
    """Bind this action on the predecessor BEFORE any effect; ``(claim, created)``.

    Same nonce: the recorded claim (replay). Another nonce while a claim
    exists: ``ValueError("already_continued:<successor>")``. A predecessor that
    turned ineligible refuses. Every refusal writes nothing.
    """
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
        if (execution_carry(current) or {}) != {key: binding[key] for key in EXECUTION_CARRY_KEYS if key in binding}:
            raise ValueError("execution_budget_changed")  # a later close: the same nonce rebinds
        from ouroboros.control_actor import control_actor
        claim = {"by": control_actor() or dict(declared_by or {}), "successor_task_id": binding["successor_task_id"],
                 "action_nonce": nonce,
                 "binding": dict(binding), "binding_sha256": binding_sha(binding),
                 "bound_at": utc_now_iso(), "state": "bound"}
        outcome.update(claim=claim, created=True)
        return stamp_task_result_schema({**current, "continued_by": claim})

    from ouroboros.task_custody import task_custody_lock

    # Take predecessor custody alone. Scratch release takes this same lock before
    # accepting deletion, then removes bytes off-lock.
    with task_custody_lock(drive_root, predecessor_task_id, timeout_sec=0) as locked:
        if not locked:
            raise ValueError("continuation_custody_busy")
        patch_document(pathlib.Path(drive_root), predecessor_task_id, update)
    return dict(outcome["claim"]), bool(outcome["created"])


def mark_claim_admitted(drive_root: Any, predecessor_task_id: str, nonce: str) -> None:
    """Record on the predecessor that its claimed successor is durably admitted."""
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


OFFER_VIEW_KEYS = ("eligible", "cause", "refusal", "successor_task_id", "state")


def offer_view(result: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """The offer a later reader of a ROOT's whole stored row is shown, recomputed on every read: the card's own
    ``continuation_offer`` without a claim's nonce. ``None`` for a non-root row, live or settled (a child's work
    is its root's): decided first, since eligibility names a live row ``predecessor_live`` before it asks."""
    if not _is_root(result, task_id):
        return None
    offer = continuation_offer(result, task_id)
    return {key: offer[key] for key in OFFER_VIEW_KEYS if key in offer}
