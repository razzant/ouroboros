"""The wake-up message, its observation and the wake's task metadata (Background Consciousness).

A wake-up is an ordinary Main turn nobody typed: ``prompts/CONSCIOUSNESS.md`` is its USER
message (system prompt, memory and tools are Main's own, owner decision В15).
``observe_wake`` captures what happened since the last ACCEPTED wake from three sources, with
no count cut or category priority: positions in the append-only chat chain (owner and
Presence-correspondent input by producer provenance, runner failures, and the chat rows that
announce a transition) and, by identity, the transitions the task results themselves record —
a task's terminal, a card's closed state — whether or not any
chat row announced them (an orphan sweep, a lost ``task_done``, an expired card) — and the
memory journal records written after the accepted sequence (``memory_inventory.memory_changes``:
pages, parts, drafts, notes, corrections, decisions, marks, mark releases; one line of text, its
author and an exact read each; what still stands open is the view's, not an event). The whole inventory of answerable
owner cards follows. What the wake accepts is a chain position plus the transition identities
of every observed task (``_transitions_since``) and the last observed journal sequence, not
the alarm's finish time, so a fact written during a wake, or written late with an older stamp,
reaches the next wake. A wake never activates the journal. ``bind_wake_observation`` stores the complete observation as an exact
source under the registered wake; when the actual request cannot fit it, the context fit
delivers composition, window, coverage and that pointer instead (``loop_model_call``).
``wake_task_metadata`` is the wake's origin/authority envelope for ``handle_wake_direct`` (``consciousness_authority``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.consciousness_authority import (
    CONSCIOUSNESS_CATEGORY,
    CONSCIOUSNESS_INITIATOR,
    WAKE_CAUSE_OWN_CLOCK,
    WORK_KIND_CONSCIOUSNESS,
    disabled_tools_for,
    is_consciousness_origin,
    normalize_level,
    runtime_mode_cap_for,
)
from ouroboros.context_health import safe_read
from ouroboros.dialogue_provenance import is_presence_task

log = logging.getLogger(__name__)

PROMPT_REL = pathlib.Path("prompts") / "CONSCIOUSNESS.md"
WAKE_OBSERVATION_KEY = "wake_observation"  # task metadata: the bound observation's source and projection
OBSERVATION_BOUNDARY_VERSION = 1
TRANSITIONS_VERSION = 2  # complete per-task transition inventory, including closed tasks
PLACEHOLDERS = ("reason", "last_wake_ago", "events", "level", "level_line", "withheld_tools",
                "spent_usd", "daily_usd", "running", "max_tasks", "interval", "spend_by_cause", "counted_roots", "places", "previous_wake")
LEVEL_LINES = {
    "observe": "On its own it studies, keeps its memory and tells you; it changes nothing outside.",
    "act": "Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself.",
    "full": "Also changes itself: its own code, its settings, connecting new places.",
}
_FALLBACK_TEMPLATE = "[Wake-up · {reason}] No one wrote to you: this turn is yours. Previous wake: {previous_wake}. Wake context (last wake {last_wake_ago}): {events}"


def wake_task_metadata(level: Any, reason: str, *, root_cost_ceiling_usd: Optional[float] = None) -> Dict[str, Any]:
    """The wake's ``task_metadata``: origin label, ledger category, level and its consequences.

    ``root_cost_ceiling_usd`` is the wake tree's GRACEFUL ceiling (what is left of the
    allowance, at most the per-task cap): ``task_pacing.resolve_cost_ceiling`` honors it for
    the root itself and the members inherit the resolved number, while the ledger fence stays
    at the owner's per-task cap. Only a strictly positive ceiling is stamped — a wake with
    nothing left of its allowance is skipped by the alarm, never started under a $0 ceiling.
    """
    normalized = normalize_level(level)
    metadata: Dict[str, Any] = {
        "initiator": CONSCIOUSNESS_INITIATOR, "usage_category": CONSCIOUSNESS_CATEGORY,
        "wake_cause": WAKE_CAUSE_OWN_CLOCK, "consciousness_cause": "own",
        "work_kind": WORK_KIND_CONSCIOUSNESS,
        "wake_reason": str(reason or "heartbeat"), "consciousness_autonomy": normalized,
        "model_role": "consciousness", "disabled_tools": disabled_tools_for(normalized),
        "runtime_mode_cap": runtime_mode_cap_for(normalized),
    }
    if root_cost_ceiling_usd is not None and float(root_cost_ceiling_usd) > 0:
        metadata["root_cost_ceiling_usd"] = float(root_cost_ceiling_usd)
    return metadata


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat()


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 86400:
        return f"{seconds // 86400} d {(seconds % 86400) // 3600} h ago"
    if seconds < 3600:
        return f"{max(1, seconds // 60)} min ago"
    return f"{seconds // 3600} h {(seconds % 3600) // 60} min ago"


def _parse_iso(value: Any) -> Optional[float]:
    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def previous_wake_text(drive_root: Any, task_id: str, *, now: float) -> str:
    """One previous own wake by primary key, with its whole authored note."""
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase
    from ouroboros.task_store import read_document
    from ouroboros.terminal_time import terminal_time_fact

    if not task_id:
        return "none"
    try:
        row = read_document(drive_root, task_id)
        if not row:
            return "unreadable"
        stamp = _parse_iso(terminal_time_fact(row).get("occurred_at"))
        ago = _ago(now - stamp) if stamp is not None else "at an unknown time"
        line = f"«{row['wake_summary']}»" if row.get("wake_summary") else "no line"
        outcome = "Interrupted" if row.get("status") == "interrupted" else OUTCOME_PHASE_HEADLINE[outcome_phase(row, {})]
        return (f"{task_id}, {outcome} {ago}; "
                f"its line: {line}; its note: «{row.get('result') or ''}»")
    except Exception:
        log.debug("consciousness: previous wake %s is unreadable", task_id, exc_info=True)
        return "unreadable"


def _clip_preview(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + f" …[{len(text) - (limit - 18)} chars omitted]"


def _trigger_line(reason: str) -> str:
    """The wake's cause line. The schedule alone starts a wake (#1603), so the
    heartbeat is the cause; what happened since is in the observed events."""
    raw = str(reason or "").strip()
    if not raw:
        return ""
    if raw == "heartbeat":
        return "- wake cause: the scheduled wake time; events never move it, they are listed as observed"
    return f"- wake cause: {raw}"


def _card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float, owner_wait: Any = None) -> str:
    from ouroboros.project_dialogue import owner_wait_projection, question_status

    state = str(block.get("state") or "open")
    facts = owner_wait_projection(quiz_id, owner_wait, block)
    label = question_status(state, facts, bool(block.get("wait_for_answer")))
    asked_at = str(block.get("asked_at") or "")
    asked_ts = _parse_iso(asked_at)
    age = f", {_ago(now - asked_ts)}" if asked_ts is not None else ""
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {label}{age}; {preview}"


def _answered_card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float) -> str:
    """One answered card of the window: stamps, the recorded answer, a question preview.

    The owner's own words are the answer itself, so the comment is rendered whole;
    only the question is a (named) preview. No verdict on what the answer meant.
    """
    parts = []
    for key, word in (("answered_at", "answered"), ("asked_at", "asked")):
        stamp = _parse_iso(block.get(key))
        parts.append(f"{word} {_ago(now - stamp)}" if stamp is not None else f"{word} at an unknown time")
    options = block.get("options") if isinstance(block.get("options"), list) else []
    index = block.get("answered_index")
    comment = str(block.get("comment") or "")
    if isinstance(index, int) and not isinstance(index, bool):
        label = str(options[index]) if 0 <= index < len(options) else "label unavailable"
        answer = f"chose option {index + 1}: {label}"
        if comment.strip():
            answer += f"; with the words: {comment}"
    elif comment.strip():
        answer = f"answered in own words: {comment}"
    else:
        answer = "answer text unavailable"
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {'; '.join(parts)}; {answer}; question: {preview}"




def _closed_card_line(task_id, quiz_id, block, *, now):
    stamp = _parse_iso(block.get("closed_at"))
    when = _ago(now - stamp) if stamp is not None else "at an unknown time"
    actor = str(block.get("closed_actor") or "unknown actor")
    if actor == "model" and block.get("closed_by_task_id"):
        actor += f" task {block['closed_by_task_id']}"
    return (f"- owner card {quiz_id} on task {task_id}: closed {when} by {actor}, without an owner answer; "
            f"reason: {block.get('closed_reason') or 'unavailable'}; "
            f"question: {_clip_preview(block.get('question') or 'question text unavailable')}")


def _chat_window(drive_root: pathlib.Path, boundary: Any, since: float,
                 gaps: set) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """Every row after the accepted SQL/legacy byte boundary, including late old timestamps.

    A wake requests all rooms. Only the initial time bootstrap surveys their retained
    rows; subsequent wakes query the accepted interval. Imports must cover the interval
    before its boundary can advance, and skipped intervals remain visible gaps.
    """
    from ouroboros import dialogue_store as ds, dialogue_positions as dp
    from ouroboros.chat_chain import source_row_id

    upper = ds.stream_end(drive_root, "chat")
    if not upper:
        ds.require_imported(drive_root, "chat")
        return [], {"lower": 0, "upper": 0, "basis": "empty_chain", "boundary": {
            "version": OBSERVATION_BOUNDARY_VERSION, "upper": 0, "segment_start": 0,
            "segment_sha256": "dialogue:empty"}}
    boundary = boundary if isinstance(boundary, dict) and "upper" in boundary else None
    lower, valid = 0, False
    if boundary and boundary.get("version") == OBSERVATION_BOUNDARY_VERSION:
        lower = int(boundary["upper"])
        start, signature = int(boundary.get("segment_start", 0)), str(boundary.get("segment_sha256") or "")
        if signature == "dialogue:empty":
            valid = lower <= upper and not ds.newest(drive_root, "chat", before=lower, limit=1)
        elif signature.startswith("dialogue:"):
            entry = ds.row_at(drive_root, "chat", start)
            valid = entry is not None and signature == "dialogue:" + source_row_id(entry.row) and entry.end <= lower <= upper
        else:
            ds.require_imported(drive_root, "chat", upper=lower)
            valid = any(unit["base"] == start and lower <= unit["base"] + unit["size"]
                        and (lower == start or not signature or unit["gen"] == signature)
                        for unit in ds.generation_metadata(drive_root, "chat"))
    basis = "accepted_boundary"
    if not valid:
        basis = "time_bootstrap" if boundary is None else "time_bootstrap_boundary_mismatch"
        if boundary:
            gaps.add("accepted_boundary_no_longer_matches_chat_chain")
        ds.require_imported(drive_root, "chat", upper=upper)
        lower = next((entry.seq for entry in ds.iter_rows(drive_root, "chat", upper=upper)
                      if str(entry.row.get("ts") or "") >= _iso(since)), upper)
    ds.require_imported(drive_root, "chat", lower=lower, upper=upper)
    for gap in dp.skipped_between(drive_root, "chat", lower if valid else 0, upper):
        gaps.add(f"{'jsonl_non_object' if gap.kind == 'non_object' else 'jsonl_malformed'}:state/dialog.sqlite:{gap.seq}")
    collected = [(entry.seq, entry.row) for entry in ds.iter_rows(drive_root, "chat", lower=lower, upper=upper)]
    last = ds.newest(drive_root, "chat", before=upper, limit=1)
    accepted = ({"version": OBSERVATION_BOUNDARY_VERSION, "upper": upper, "segment_start": last[0].seq,
                 "segment_sha256": "dialogue:" + source_row_id(last[0].row)} if last else {
                     "version": OBSERVATION_BOUNDARY_VERSION, "upper": upper, "segment_start": 0,
                     "segment_sha256": "dialogue:empty"})
    return collected, {"lower": lower, "upper": upper, "basis": basis, "boundary": accepted}


# A pre-fold ``acceptance_late_settlement`` row (the deleted task-acceptance machine's late
# verdict) is no wake event: it stays ordinary history of the chat it was written to.
_SYSTEM_ROW_KINDS = {"task_summary": "task_terminal", "quiz_answer": "card_answer", "quiz_closed": "card_state",
                     "task_error": "task_error"}


def _row_kind(row: Dict[str, Any]) -> str:
    """The provenance class of one chat row; "" for rows that are not wake events.

    Human input is classified by what its producer stamped, never by
    ``direction`` alone: A2A traffic is machine traffic, a Presence
    correspondent is not the owner, an Ouroboros-initiated Presence row is not
    a human, a late card answer is carried by its ``quiz_answer`` row, and an
    imported earlier message (``presence_history``) is counted, never listed as new.
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    kind, direction = str(row.get("type") or ""), str(row.get("direction") or "")
    if (kind == "task_summary" and row.get("summary_kind") == "host_task_facts"
            and isinstance(row.get("presence_recovery"), dict)):
        return "presence_observation"
    if kind in _SYSTEM_ROW_KINDS:
        return _SYSTEM_ROW_KINDS[kind]
    if direction != "in" or is_a2a_chat_id(row.get("chat_id")):
        return ""
    if str(row.get("client_message_id") or "").startswith("quiz_late_answer:"):
        return ""
    if isinstance(row.get("historical_import"), dict):
        return "history_import"  # an earlier message a connector recorded as history, not an arrival
    if str(row.get("source") or "").startswith("presence:"):
        actor = (row.get("transport") or {}).get("actor") if isinstance(row.get("transport"), dict) else None
        if isinstance(actor, dict) and actor.get("kind") == "proactive_initiation":
            return ""
        return "correspondent_message"
    return "owner_message"


_KIND_WORDS = {  # (one, many)
    "owner_message": ("owner message", "owner messages"),
    "correspondent_message": ("Presence correspondent message", "Presence correspondent messages"),
    "card_answer": ("card answer", "card answers"), "task_terminal": ("task terminal", "task terminals"),
    "direct_turn": ("own direct turn ended (counted, not listed: its exchange is the dialogue itself)",
                    "own direct turns ended (counted, not listed: their exchange is the dialogue itself)"),
    "direct_turn_summary": ("own direct turn ended with its line", "own direct turns ended with their lines"),
    "task_error": ("task runner failure", "task runner failures"),
    "card_state": ("card closed unanswered", "cards closed unanswered"),
    "outstanding_card": ("outstanding card", "outstanding cards"),
    "memory_change": ("memory change", "memory changes"),
    "presence_observation": ("retained Presence observation", "retained Presence observations"),
    "history_import": ("imported earlier Presence message (counted, not listed: room history, not an arrival)",
                       "imported earlier Presence messages (counted, not listed: room history, not arrivals)"),
}
# Kinds the wake counts without listing: their content is the room itself.
_COUNTED_ONLY = frozenset({"direct_turn", "history_import"})


def _presence_failure(result: Dict[str, Any]) -> bool:
    """Inline Presence shares the direct lane, but its host failure is absent from the dialogue."""
    from ouroboros.task_finalization import HOST_AUTHORED_TERMINAL_ORIGINS

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return is_presence_task(result) and not is_consciousness_origin(metadata) and (
        str(result.get("status") or "") in {"failed", "cancelled"}
        or result.get("terminal_origin") in HOST_AUTHORED_TERMINAL_ORIGINS)


def _event_line(kind: str, row: Dict[str, Any], results: Dict[str, Dict[str, Any]],
                rooms: Any, projects: Dict[str, str], now: float, *, when: str = "") -> str:
    """One chat-row event, or (``when`` given) a task terminal read from its result."""
    stamp = _parse_iso(row.get("ts"))
    when = when or (_ago(now - stamp) if stamp is not None else "at an unknown time")
    task_id = str(row.get("task_id") or "")
    if kind in {"owner_message", "correspondent_message"}:
        return (f"- {_KIND_WORDS[kind][0]} in {rooms.label(row)}, {when}: "
                f"\"{_clip_preview(row.get('text'), 240)}\"")
    if kind == "history_import":
        return f"- imported earlier message in {rooms.label(row)}, sent {when}"
    if kind == "card_answer":
        block = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
        return _answered_card_line(task_id, str(block.get("quiz_id") or "?"), block, now=now)
    if kind == "card_state":
        block = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
        return _closed_card_line(task_id, str(block.get("quiz_id") or "?"), block, now=now)
    if kind == "task_error":
        return f"- task {task_id} runner failed, {when}: {_clip_preview(row.get('text'), 160)}"
    if kind == "presence_observation":
        from ouroboros.dialogue_provenance import render_row_text

        return f"- retained Presence observation in {rooms.label(row)}, {when}: {render_row_text(row)}"
    result = results.get(task_id) or {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    status = str(row.get("status") or result.get("status") or "settled")
    cost = result.get("accounted_upper_bound_usd", result.get("cost_usd"))
    cost_text = f", ${float(cost):.2f}" if isinstance(cost, (int, float)) else ""
    title = _clip_preview(result.get("description") or result.get("text") or row.get("text"), 100)
    project = str(row.get("project_id") or result.get("project_id") or "")
    where = f" in project {projects.get(project, project)}" if project else ""
    summary = f": {result['wake_summary']}" if result.get("wake_summary") else ""
    if kind in {"direct_turn", "direct_turn_summary"}:
        return f"- {'wake' if is_consciousness_origin(metadata) else 'direct turn'} {task_id} {status}{where}, {when}{summary}"
    if is_presence_task(result):
        if isinstance(metadata.get("presence_unknown_outcome"), dict):
            from ouroboros.dialogue_provenance import render_row_text
            from ouroboros.presence_recovery import retained_turn_facts

            presence = metadata.get("presence") or {}
            retained = {"task_id": task_id, "type": "task_summary", "text": "",
                        "presence_recovery": retained_turn_facts(result, task_id),
                        "source_event": presence.get("event"), "source_text": presence.get("observed_text")}
            return f"- Presence turn {task_id} {status}{where}, {when}; {render_row_text(retained)}{summary}"
        outcome = str(metadata.get("presence_outcome") or "unknown")
        work = str(metadata.get("presence_work_ref") or "")
        return (f"- Presence turn {task_id} {status}{where}, {when}; Presence outcome={outcome}; see get_task_result"
                + (f"; deferred work={work}" if work else "") + summary)
    parent = str(row.get("parent_task_id") or result.get("parent_task_id") or "")
    role = str(row.get("delegation_role") or result.get("delegation_role") or "")
    noun = f"child task {task_id} of {parent}" if role == "subagent" and parent else f"task {task_id}"
    return f"- {noun} {status}{cost_text}{where}, {when}: {title}".rstrip(": ") + summary


# --- canonical transitions: identities in the task results, not chat rows or stamps --------


def _transition_facts(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every transition one task result records, by identity.

    ``key -> fact`` for a terminal of one attempt (the terminal-publication fence's
    own attempt facts) and each card's closed state. Stamps ride along for display
    and the first (time-bootstrapped) window only: a terminal's is its publication
    instant when recorded, else the last update, never presented as the completion
    time. Closed tasks remain in the inventory: a late owner answer can still add
    evidence.
    """
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.terminal_projection import _attempt

    task_id, facts = str(row.get("task_id") or ""), {}
    status = str(row.get("status") or "")
    quizzes = row.get("owner_quiz") if isinstance(row.get("owner_quiz"), dict) else {}
    if status in SETTLED_STATUSES:
        attempt = hashlib.sha256(json.dumps(_attempt(row), sort_keys=True, default=str).encode("utf-8")).hexdigest()
        proven = [_parse_iso(record[key]) for holder, key in (("canonical_terminal_projection", "written_at"),
                                                               ("canonical_terminal_projection_ready", "task_done_ts"))
                  if isinstance(record := row.get(holder), dict) and record.get(key)]
        proven = [stamp for stamp in proven if stamp is not None]
        facts[f"terminal:{task_id}:{status}:{attempt[:16]}"] = {
            "kind": "task_terminal", "task_id": task_id, "proven": bool(proven),
            "stamp": proven[0] if proven else _parse_iso(row.get("updated_at") or "")}
    for quiz_id, block in quizzes.items():
        state = str(block.get("state") or "") if isinstance(block, dict) else ""
        if state and state != "open":
            stamp = _parse_iso(block.get("closed_at") or block.get("answered_at") or block.get("reconciled_at") or "")
            facts[f"card:{task_id}:{quiz_id}:{state}"] = {
                "kind": "card_answer" if state == "answered" else "card_state", "task_id": task_id,
                "quiz_id": str(quiz_id), "block": block, "stamp": stamp, "proven": stamp is not None}
    return facts


def _transitions_since(transitions: Dict[str, Dict[str, Dict[str, Any]]], accepted: Any, *,
                       since: float, scan_at: float) -> Tuple[Dict[str, Dict[str, Any]], set, Dict[str, Any], str]:
    """Compare complete transition inventories; timestamps are display/bootstrap only.

    Keep every observed task's keys, including closed tasks. Absence from a
    complete accepted inventory then proves a newly observed result even if its
    first write copied an old timestamp. A key an earlier inventory holds and no
    task records any more (a pre-fold late-review key) is simply never fresh again.
    The existing wake boundary owns this state, with no new
    store or count cutoff. Unchanged tasks retain their accepted keys; their
    bodies are not needed for comparison.

    V1 recorded only open tasks. Unknown keys in that incomplete inventory are
    conservatively replayed once, with coverage disclosed by the caller.
    """
    valid = (isinstance(accepted, dict) and accepted.get("version") in {1, TRANSITIONS_VERSION}
             and isinstance(accepted.get("inventory"), dict))
    inventory = accepted["inventory"] if valid else {}
    known = set(accepted.get("observed") or []) if valid else set()
    known.update(key for keys in inventory.values() for key in keys)
    fresh, current = {}, {}
    for task_id, facts in transitions.items():
        current[task_id] = sorted(facts)
        for key, fact in facts.items():
            if valid:
                if key not in known:
                    fresh[key] = fact
            elif fact["stamp"] is not None and fact["stamp"] >= since:
                fresh[key] = fact
    state = {"version": TRANSITIONS_VERSION, "scan_at": _iso(scan_at), "inventory": current}
    basis = ("accepted_inventory" if accepted.get("version") == TRANSITIONS_VERSION else
             "partial_inventory_upgrade") if valid else "time_bootstrap"
    return fresh, known, state, basis


def _transition_line(fact: Dict[str, Any], results: Dict[str, Dict[str, Any]], rooms: Any,
                     now: float, projects: Dict[str, str]) -> Tuple[str, str]:
    """``(kind, line)`` for one canonical transition."""
    kind, task_id, stamp = fact["kind"], fact["task_id"], fact["stamp"]
    ago = _ago(now - stamp) if stamp is not None else "at an unknown time"
    if kind == "card_answer":
        return kind, _answered_card_line(task_id, fact["quiz_id"], fact["block"], now=now)
    if kind == "card_state":
        from ouroboros.project_dialogue import owner_wait_projection, question_status

        block = fact["block"]
        if block.get("state") == "closed":
            return kind, _closed_card_line(task_id, fact["quiz_id"], block, now=now)
        label = question_status(str(block.get("state") or ""), owner_wait_projection(
            fact["quiz_id"], (results.get(task_id) or {}).get("owner_wait"), block), False)
        when = f", closed {ago}" if stamp is not None else ""
        return kind, (f"- owner card {fact['quiz_id']} on task {task_id}: {label}{when}; "
                      f"{_clip_preview(block.get('question') or 'question text unavailable')}")
    result = results.get(task_id) or {}
    when = ago if fact["proven"] or stamp is None else f"completion time not recorded (last updated {ago})"
    if result.get("_is_direct_chat") and not _presence_failure(result):
        kind = "direct_turn_summary" if result.get("wake_summary") else "direct_turn"
    return kind, _event_line(kind, {"task_id": task_id}, results, rooms, projects, now, when=when)


_TRANSITION_BASIS = {
    "accepted_inventory": "transitions by identity since the accepted inventory",
    "time_bootstrap": "no accepted inventory yet, transitions stamped since the last wake",
    "partial_inventory_upgrade": "old inventory covered only open tasks; unseen identities replayed once",
    "unreadable_task_results": "unreadable, the accepted inventory is kept for the next wake",
    "unreadable_transition_source": "unknown: accepted inventory source unreadable; boundary unchanged",
}


@dataclass(frozen=True)
class WakeObservation:
    """One wake's complete compact observation: every event since the accepted boundary."""

    trigger: str
    # (kind, chat-chain offset or None, rendered line): chat-positioned events in append
    # order, then transitions no chat row of this window announced, by their own stamps,
    # then memory journal changes in publication order.
    events: Tuple[Tuple[str, Optional[int], str], ...]
    outstanding: Tuple[str, ...]
    window: Dict[str, Any]
    gaps: Tuple[str, ...]
    captured_at: str
    # The chat position, transition state and memory sequence a wake that accepts this observation persists.
    boundary: Optional[Dict[str, Any]] = None

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for kind, _offset, _line in self.events:
            counts[kind] = counts.get(kind, 0) + 1
        if self.outstanding:
            counts["outstanding_card"] = len(self.outstanding)
        return counts

    def composition(self) -> str:
        counts = self.counts()
        return ", ".join(f"{count} {_KIND_WORDS.get(kind, (kind, kind))[count != 1]}"
                         for kind, count in counts.items()) or "no events"

    def _coverage(self) -> str:
        window = self.window
        text = (f"chat log bytes {window.get('lower', 0)}–{window.get('upper', 0)} "
                f"({str(window.get('basis') or '').replace('_', ' ')}); task results: "
                f"{_TRANSITION_BASIS.get(str(window.get('transitions_basis') or ''), 'not read')}")
        memory = window.get("memory")
        if isinstance(memory, dict):
            text += (f"; memory records: sequence {memory.get('lower')}–{memory.get('upper', 'unknown')} "
                     f"({str(memory.get('basis') or '').replace('_', ' ')})")
            if memory.get("basis") == "initial_baseline":
                text += "; first observed sequence, earlier changes not inventoried"
                if memory.get("last_record_id"):
                    text += f"; baseline record: memory_read(node_id='{memory['last_record_id']}')"
                text += (f"; earlier records of a room: memory_read(room_id=..., after_seq=0), "
                         f"through sequence {memory.get('upper')}")
        return text + (f"; gaps: {', '.join(self.gaps)}" if self.gaps else "")

    def full_text(self) -> str:
        """The observation when it fits; only direct turns without an authored line stay counted alone."""
        lines = [self.trigger] if self.trigger else []
        if self.events:
            lines.append(f"Observed since the last accepted wake ({self.composition()}; {self._coverage()}):")
            unannounced = memory = False
            for kind, offset, line in self.events:
                if kind == "memory_change" and not memory:
                    memory = True
                    lines.append("Recorded in memory after the accepted sequence (source dates do not order publication):")
                elif offset is None and kind != "memory_change" and not unannounced:
                    unannounced = True
                    lines.append("Recorded in task results with no chat row in this window (by their own stamps):")
                if kind not in _COUNTED_ONLY:
                    lines.append(line)
        elif self.gaps or (self.window.get("memory") or {}).get("basis") == "initial_baseline":
            lines.append(f"Observation coverage: {self._coverage()}")
        if self.outstanding:
            lines.append(f"Outstanding owner cards ({len(self.outstanding)}):")
            lines.extend(self.outstanding)
        return ("\n" + "\n".join(lines)) if lines else "nothing new"

    def overview_text(self, source: Dict[str, Any]) -> str:
        """Composition, window and coverage plus the exact pointer; no event is pre-selected."""
        lines = [self.trigger] if self.trigger else []
        lines.append(
            "The complete observation of this wake does not fit this request beside the tools and the "
            "answer reserve, so it is stored whole as an exact source instead of shown here. "
            f"Composition: {self.composition()}. Window: {self._coverage()}, captured {self.captured_at}.")
        read = source.get("read") if isinstance(source.get("read"), dict) else {}
        lines.append(
            f"Source: {json.dumps(read, ensure_ascii=False, sort_keys=True)} — "
            f"{source.get('lines')} JSON lines (a header, then one event per line: chat-positioned in append "
            f"order, then transitions no chat row announced, then memory changes; then outstanding cards), sha256 "
            f"{source.get('sha256')}. Read the ranges you need; this pointer is neither a summary nor a "
            "priority order.")
        return "\n" + "\n".join(lines)

    def source_bytes(self) -> bytes:
        header = {"kind": "wake_observation", "captured_at": self.captured_at, "window": self.window,
                  "gaps": list(self.gaps), "composition": self.counts(), "trigger": self.trigger,
                  "boundary": self.boundary}
        rows = [header] + [{"kind": kind, "chat_offset": offset, "line": line} for kind, offset, line in self.events]
        rows += [{"kind": "outstanding_card", "line": line} for line in self.outstanding]
        return ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode("utf-8")


_CHAT_BOUNDARY_KEYS = ("version", "upper", "segment_start", "segment_sha256")


def _task_window(root, prior, chat_rows):
    """Capture typed signatures and only changed/question/chat-addressed bodies together.

    The accepted inventory already owns unchanged transition identities. Old
    boundaries without signatures scan once; timestamps alone cannot prove that
    a result carrying an old date is unchanged. NULL question flags are unknown.
    """
    from ouroboros import task_store

    complete = (isinstance(prior, dict) and prior.get("version") == TRANSITIONS_VERSION
                and isinstance(prior.get("inventory"), dict) and isinstance(prior.get("signatures"), dict))
    previous = prior["signatures"] if complete else {}
    with task_store.open_store(root, write=False) as txn:
        signatures = {tid: list(stamp) for tid, stamp in txn.write_seqs().items()}
        selected = {tid for tid, stamp in signatures.items()
                    if not complete or tid not in prior["inventory"] or previous.get(tid) != stamp}
        selected.update(str(row.get("task_id") or "") for _offset, row in chat_rows)
        selected.update(row[0] for row in txn.conn.execute(
            "SELECT task_id FROM tasks WHERE has_owner_questions=1 OR has_owner_questions IS NULL"))
        fetched = task_store._fetch(txn, "t.task_id IN (SELECT value FROM json_each(?))",
                                   (json.dumps(sorted(selected)),))
    rows = task_store._decode(fetched)
    retained = {tid: {key: None for key in prior["inventory"][tid]}
                for tid in signatures if complete and tid not in selected}
    return rows, signatures, retained


def observe_wake(drive_root: Any, *, boundary: Any, since: float, now: float, reason: str = "") -> WakeObservation:
    """Capture every fact since the accepted boundary, plus outstanding cards.

    No count cut and no category order. Chat events keep their physical append
    order; a canonical transition takes the position of the chat row that
    announced it, and one no row of this window announced follows. Answerable
    cards (``open``/``expired_terminal`` — В17a) are an inventory of the whole
    store, listed separately and in full. ``now`` is taken before this scan and
    anchors the next wake's transition window. A source that cannot be read is a
    disclosed gap, and what the wake accepts for it stays where it was.
    """
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.owner_quiz import STATE_EXPIRED_TERMINAL, STATE_OPEN
    from ouroboros.projects_registry import list_reserved_projects, PROJECT_ACTIVE

    root, gaps = pathlib.Path(drive_root), set()
    accepted = boundary if isinstance(boundary, dict) else {}
    prior = accepted.get("transitions")
    source_unreadable = False
    if isinstance(prior, dict) and "source_ref" in prior:
        from ouroboros.artifacts import read_actor_source_bytes

        try:
            ref = prior["source_ref"]
            raw = read_actor_source_bytes(root, ref["task_id"], ref)
            header = json.loads(raw.splitlines()[0])
            stored = header["boundary"]
            inventory = stored["transitions"]
            if (header.get("kind") != "wake_observation" or header.get("captured_at") != accepted.get("captured_at")
                    or any(stored.get(key) != accepted.get(key) for key in _CHAT_BOUNDARY_KEYS)
                    or stored.get("memory") != accepted.get("memory")
                    or inventory.get("version") != TRANSITIONS_VERSION
                    or not isinstance(inventory.get("inventory"), dict)
                    or any(not isinstance(keys, list) or not all(isinstance(key, str) for key in keys)
                           for keys in inventory["inventory"].values())):
                raise ValueError("wake source does not match accepted boundary")
            prior = inventory
        except Exception as exc:
            source_unreadable = True
            gaps.add(f"unreadable_transition_source: {type(exc).__name__}")
    try:
        chat_rows, window = _chat_window(root, accepted or None, since, gaps)
    except Exception as exc:
        chat_rows, window = [], {"lower": None, "upper": None, "basis": "chat_chain_unreadable", "boundary": None}
        gaps.add(f"chat log unreadable: {type(exc).__name__}")
        from ouroboros.store import StoreError
        if isinstance(exc, StoreError):
            gaps.add(str(exc))
    try:
        rows, signatures, transitions = _task_window(root, prior, chat_rows)
        scanned = True
    except Exception as exc:  # a disclosed gap beats a missing wake
        rows, signatures, transitions, scanned = [], {}, {}, False
        gaps.add(f"task_results unreadable: {type(exc).__name__}")
    results = {str(row.get("task_id") or ""): row for row in rows}
    transitions.update({task_id: _transition_facts(row) for task_id, row in results.items()})

    if source_unreadable:
        fresh, known, state, basis = {}, set(), None, "unreadable_transition_source"
    elif scanned:
        fresh, known, state, basis = _transitions_since(
            transitions, prior, since=since, scan_at=now)
        state["signatures"] = signatures
    else:  # nothing was read: the accepted state stays, so the next wake finds the same transitions
        fresh, known, state, basis = {}, set(), prior, "unreadable_task_results"
    window["transitions_basis"] = basis
    try:
        registry = list_reserved_projects(root)
    except Exception as exc:
        registry = []
        gaps.add(f"project names unreadable: {type(exc).__name__}")
    rooms = RoomLabelResolver(projects=registry)
    projects = {str(row["id"]): str(row.get("name") or row["id"]) for row in registry
                if row.get("lifecycle") == PROJECT_ACTIVE}
    if basis == "partial_inventory_upgrade":
        gaps.add("previous_transition_inventory_incomplete")
    events, announced = [], set()
    for offset, row in chat_rows:
        kind = _row_kind(row)
        if not kind:
            continue
        # The canonical transition this row announces, when its task result records it.
        facts = transitions.get(str(row.get("task_id") or ""), {})
        quiz = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
        key = next((key for key, fact in facts.items() if (
            kind == "task_terminal" and fact["kind"] == "task_terminal")
            or (kind == "card_answer" and key == f"card:{row.get('task_id')}:{quiz.get('quiz_id')}:answered")
            or (kind == "card_state" and key == f"card:{row.get('task_id')}:{quiz.get('quiz_id')}:closed")),
            None)
        if key is not None:
            if key in announced or (key in known and key not in fresh):
                continue  # announced earlier in this window, or already accepted by a previous wake
            # A row for a transition the identity window classed as old (a result first
            # written late under an older stamp) is still new to this position: at least once.
            fact = fresh.setdefault(key, facts[key])
            announced.add(key)
            kind, line = _transition_line(fact, results, rooms, now, projects)
            events.append((kind, offset, line))
            continue
        result = results.get(str(row.get("task_id") or "")) or {}
        if kind == "task_terminal" and result.get("_is_direct_chat") and not _presence_failure(result):
            kind = "direct_turn_summary" if result.get("wake_summary") else "direct_turn"
        events.append((kind, offset, _event_line(kind, row, results, rooms, projects, now)))
    for key, fact in sorted(((key, fact) for key, fact in fresh.items() if key not in announced),
                            key=lambda item: (item[1]["stamp"] is None, item[1]["stamp"] or 0.0, item[0])):
        kind, line = _transition_line(fact, results, rooms, now, projects)
        events.append((kind, None, line))
    memory_boundary = accepted.get("memory")
    if not source_unreadable:
        from ouroboros.memory_inventory import memory_changes

        changes, memory_boundary, memory_window = memory_changes(root, memory_boundary, gaps)
        events.extend(changes)
        if memory_window:
            window["memory"] = memory_window
    cards = []
    for task_id, row in results.items():
        quizzes = row.get("owner_quiz") if isinstance(row.get("owner_quiz"), dict) else {}
        for quiz_id, block in quizzes.items():
            if isinstance(block, dict) and not block.get("answered_at") and block.get("state") in (
                    STATE_OPEN, STATE_EXPIRED_TERMINAL):
                cards.append((str(block.get("asked_at") or ""), _card_line(
                    task_id, str(quiz_id), block, now=now, owner_wait=row.get("owner_wait"))))
    trigger = _trigger_line(reason)
    chat_boundary = window.get("boundary") or {key: accepted[key] for key in _CHAT_BOUNDARY_KEYS if key in accepted}
    state_to_accept = {**chat_boundary, **({"transitions": state} if isinstance(state, dict) else {}),
                       **({"memory": memory_boundary} if memory_boundary is not None else {})}
    return WakeObservation(
        trigger=trigger, events=tuple(events),
        outstanding=tuple(line for _stamp, line in sorted(cards, reverse=True)),
        window=window, gaps=tuple(sorted(gaps)), captured_at=_iso(now),
        boundary=None if source_unreadable else state_to_accept or None)


def bind_wake_observation(drive_root: Any, task: Dict[str, Any], observation: WakeObservation,
                          render: Any) -> Optional[Dict[str, Any]]:
    """Store the whole observation under the registered wake and name its projection.

    Runs after the wake is registered and before its thread starts. The task's
    text stays the complete observation (the original host input); the stored
    source and the overview message only let the context fit deliver it by
    reference when it cannot fit (``loop_model_call``). Only exact source readback
    permits accepting a compact boundary. Failure leaves full input and no advance.
    """
    from ouroboros.chat_chain import retain_memory_source
    from ouroboros.artifacts import read_actor_source_bytes
    from types import SimpleNamespace

    metadata = task.setdefault("metadata", {})
    accepted = None
    facts: Dict[str, Any] = {"captured_at": observation.captured_at, "window": observation.window,
                             "composition": observation.counts(), "gaps": list(observation.gaps)}
    try:
        data = observation.source_bytes()
        source = retain_memory_source(SimpleNamespace(drive_root=drive_root, task_id=str(task["id"])),
                                      "wake_observation", data, extension="jsonl")
        if read_actor_source_bytes(drive_root, str(task["id"]), source) != data:
            raise ValueError("wake source readback mismatch")
        source = {**source, "lines": data.count(b"\n")}
        facts.update(source=source, projection_text=render(observation.overview_text(source)))
        if observation.boundary is not None and (observation.boundary.get("transitions") or {}).get("version") == TRANSITIONS_VERSION:
            accepted = {key: observation.boundary[key] for key in _CHAT_BOUNDARY_KEYS if key in observation.boundary}
            accepted.update(transitions={"version": TRANSITIONS_VERSION, "source_ref": source},
                            task_id=str(task["id"]), captured_at=observation.captured_at)
            if "memory" in observation.boundary:
                accepted["memory"] = observation.boundary["memory"]
    except Exception as exc:
        facts["source_error"] = f"{type(exc).__name__}: {exc}"
    metadata[WAKE_OBSERVATION_KEY] = facts
    return accepted


def render_wake_message(repo_dir: Any, *, drive_root: Any, reason: str, last_wake_at: float, now: float, level: Any,
                        disabled_tools: List[str], spent_usd: Any, daily_usd: Any, running: int,
                        max_tasks: int, interval: int, events: str, spent_is_floor: bool = False,
                        spend_by_cause=None, counted_roots=None, places=None, previous_wake: str = "none") -> str:
    """Fill ``prompts/CONSCIOUSNESS.md`` for one wake; every placeholder is substituted."""
    template = safe_read(pathlib.Path(repo_dir) / PROMPT_REL) or _FALLBACK_TEMPLATE
    from ouroboros.consciousness_view import places_text, roots_text, spend_text
    normalized = normalize_level(level)
    spent = f"{float(spent_usd):.2f}" if isinstance(spent_usd, (int, float)) else "unknown"
    if spent_is_floor and spent != "unknown":
        spent = f"at least {spent}"  # unmetered rows in the window: the number is a floor
    facts = {
        "reason": str(reason or "heartbeat"),
        "last_wake_ago": _ago(now - last_wake_at) if last_wake_at else "no wake since this process started",
        "previous_wake": previous_wake,
        "events": events or "nothing new",
        "level": {"observe": "Advises", "act": "Acts", "full": "Evolves"}[normalized],
        "level_line": LEVEL_LINES[normalized],
        "withheld_tools": ", ".join(disabled_tools) if disabled_tools else "none",
        "spent_usd": spent, "daily_usd": f"{float(daily_usd):.2f}",
        "running": str(int(running)), "max_tasks": str(int(max_tasks)), "interval": str(int(interval)),
        "spend_by_cause": spend_text(spend_by_cause), "counted_roots": roots_text(counted_roots),
        "places": places_text(drive_root, places),
    }
    # One pass: a fact (a task title inside {events}) that happens to contain "{daily_usd}"
    # is never substituted again.
    return re.sub(r"\{(\w+)\}", lambda m: facts.get(m.group(1), m.group(0)), template)
