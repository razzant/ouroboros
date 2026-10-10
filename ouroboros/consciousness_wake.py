"""The wake-up message, its observation and the wake's task metadata (Background Consciousness).

A wake-up is an ordinary Main turn nobody typed: ``prompts/CONSCIOUSNESS.md`` is its USER
message (system prompt, memory and tools are Main's own, owner decision В15).
``observe_wake`` captures what happened since the last ACCEPTED wake from three sources, with
no count cut or category priority: positions in the append-only chat chain (owner and
Presence-correspondent input by producer provenance, runner failures, and the chat rows that
announce a transition) and, by identity, the transitions the task results themselves record —
a task's terminal, a card's closed state, a late acceptance settlement — whether or not any
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
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.consciousness_authority import (
    CONSCIOUSNESS_CATEGORY,
    CONSCIOUSNESS_INITIATOR,
    disabled_tools_for,
    is_consciousness_origin,
    normalize_level,
    runtime_mode_cap_for,
)
from ouroboros.context_health import safe_read
from ouroboros.dialogue_provenance import is_presence_task
from ouroboros.jsonl_tail import JsonlChainSnapshot

PROMPT_REL = pathlib.Path("prompts") / "CONSCIOUSNESS.md"
WAKE_OBSERVATION_KEY = "wake_observation"  # task metadata: the bound observation's source and projection
OBSERVATION_BOUNDARY_VERSION = 1
TRANSITIONS_VERSION = 2  # complete per-task transition inventory, including closed tasks
PLACEHOLDERS = ("reason", "last_wake_ago", "events", "level", "level_line", "withheld_tools",
                "spent_usd", "daily_usd", "running", "max_tasks", "interval")
LEVEL_LINES = {
    "observe": "research and internal work, memory, project notes, your own children and schedules, owner delivery; no shell, user-file, source, skill/settings or publication changes",
    "act": "everything your runtime mode allows except editing your own code/prompts, evolution, restart and settings",
    "full": "everything your runtime mode allows, including evolution",
}
_FALLBACK_TEMPLATE = "[Wake-up · {reason}] No one wrote to you: this turn is yours. Wake context (last wake {last_wake_ago}): {events}"


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


def _clip_preview(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + f" …[{len(text) - (limit - 18)} chars omitted]"


def _trigger_line(
    projects: Dict[str, str], reason: str, rows: List[Dict[str, Any]], *, now: float,
) -> tuple[str, str]:
    """Return ``(human line, task id pinned by the trigger)`` without inventing state."""
    from ouroboros.task_status import SETTLED_STATUSES

    raw = str(reason or "").strip()
    if not raw:
        return "", ""
    if raw.startswith("project_digest:"):
        parts = raw.split(":", 2)
        project_id = parts[1].strip() if len(parts) > 1 else ""
        trigger_task_id = parts[2].strip() if len(parts) > 2 else ""
        matches = [
            row for row in rows
            if str(row.get("project_id") or "").strip() == project_id
            and str(row.get("status") or "") in SETTLED_STATUSES
            and not row.get("_is_direct_chat")
            and str((row.get("metadata") or {}).get("initiator") or "") != "consciousness"
        ]
        if trigger_task_id:
            exact = [row for row in matches if str(row.get("task_id") or "") == trigger_task_id]
            if exact:
                matches = exact
        matches.sort(key=lambda row: str(row.get("updated_at") or row.get("ts") or ""), reverse=True)
        if matches:
            row = matches[0]
            task_id = str(row.get("task_id") or "")
            status = str(row.get("status") or "settled")
            title = _clip_preview(row.get("description") or row.get("text") or row.get("result"), 120)
            cost = row.get("accounted_upper_bound_usd", row.get("cost_usd"))
            cost_text = f", ${float(cost):.2f}" if isinstance(cost, (int, float)) else ""
            project = projects.get(project_id, project_id)
            stamp = _parse_iso(row.get("updated_at") or row.get("ts"))
            age_text = f", {_ago(now - stamp)}" if stamp is not None else ""
            detail = f"; {title}" if title else ""
            qualifier = "settled task" if trigger_task_id else "latest settled task"
            return f"- wake cause: project {project} {qualifier} {task_id} ({status}){cost_text}{age_text}{detail}", task_id
        return f"- wake cause: project {projects.get(project_id, project_id)} reported a settled task (details unavailable)", ""
    if raw.startswith("task_finished:"):
        parts = raw.split(":", 2)
        task_id, status = (parts[1:] + ["", ""])[:2]
        row = next((row for row in rows if str(row.get("task_id") or "") == task_id), None)
        title = _clip_preview((row or {}).get("description") or (row or {}).get("text") or (row or {}).get("result"), 120)
        detail = f"; {title}" if title else ""
        return f"- wake cause: task {task_id} finished ({status or 'settled'}){detail}", task_id
    if raw.startswith("orphans_healed:"):
        return f"- wake cause: {raw.replace('_', ' ')}", ""
    if raw == "heartbeat":
        return "- wake cause: scheduled heartbeat; no event reason is recorded for this wake", ""
    return f"- wake cause: {raw or 'unknown event'}", ""


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




class _ChatChain(JsonlChainSnapshot):
    """One captured pass over ``logs/chat.jsonl`` + its rotated archives (``jsonl_chain_handles``).

    Offsets count bytes across the chain; archives are never pruned, so an
    accepted boundary stays a valid position across rotations. Only complete
    rows are returned (``JsonlChainSnapshot.rows``): an unfinished live line
    belongs to the next wake.
    """

    def first_line_sha256(self, index: int) -> str:
        """Hash only a committed first nonempty line within this captured segment."""
        from ouroboros.utils import jsonl_chain_handles

        base, end = self.segment(index)
        if base < end:
            with jsonl_chain_handles(self.path, strict=True, start_offset=base, snapshot=self.snapshot) as handles:
                remaining = end - base
                while remaining:
                    raw = handles[0][1].readline(remaining)
                    if not raw.endswith(b"\n"):
                        return ""
                    remaining -= len(raw)
                    first = raw.decode("utf-8", errors="replace").strip()
                    if first:
                        return hashlib.sha256(first.encode("utf-8", errors="replace")).hexdigest()
        return ""


def _boundary_segment(chain: _ChatChain, boundary: Any) -> Optional[int]:
    """The segment an accepted boundary still denotes, or None when the chain changed."""
    if not isinstance(boundary, dict) or boundary.get("version") != OBSERVATION_BOUNDARY_VERSION:
        return None
    try:
        upper, seg_start = int(boundary["upper"]), int(boundary["segment_start"])
    except (KeyError, TypeError, ValueError):
        return None
    for index in range(len(chain.entries)):
        base, end = chain.segment(index)
        if base == seg_start and upper <= end:
            expected = str(boundary.get("segment_sha256") or "")
            actual = chain.first_line_sha256(index)
            # A zero-byte boundary never committed that segment's first line;
            # legacy cursors may have fingerprinted its then-unfinished bytes.
            return index if upper == base or not expected or expected == actual else None
    return None


def _chat_window(drive_root: pathlib.Path, boundary: Any, since: float,
                 gaps: set) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """Every complete chat-chain row appended after the accepted boundary.

    Position, not the row's own ``ts``, defines the window: a row written late
    with an older stamp is still new to this wake. Without a usable boundary
    (first wake after upgrade, or a replaced chain) the window starts at the
    first row stamped at/after ``since``, and says so.
    """
    chain = _ChatChain(drive_root / "logs" / "chat.jsonl")
    if not isinstance(boundary, dict) or "upper" not in boundary:
        boundary = None  # no accepted chat position (only transitions were ever accepted)
    if not chain.entries:
        return [], {"lower": 0, "upper": 0, "basis": "empty_chain", "boundary": None}
    start_index = _boundary_segment(chain, boundary)
    basis = "accepted_boundary"
    if start_index is not None:
        lower = int(boundary["upper"])
    else:
        basis = "time_bootstrap" if boundary is None else "time_bootstrap_boundary_mismatch"
        if boundary is not None:
            gaps.add("accepted_boundary_no_longer_matches_chat_chain")
        # Newest segment first; without a fresh row the window starts at the end
        # of the last COMPLETE line, so an unfinished live line is read later.
        since_iso, index = _iso(since), len(chain.entries) - 1
        rows, lower = chain.rows(index, chain.segment(index)[0], set())
        start_index = index
        while index >= 0:
            fresh = [offset for offset, row in rows if str(row.get("ts") or "") >= since_iso]
            if fresh:
                lower, start_index = fresh[0], index
            if rows and str(rows[0][1].get("ts") or "") < since_iso:
                break
            index -= 1
            rows = chain.rows(index, chain.segment(index)[0], set())[0] if index >= 0 else []
    collected: List[Tuple[int, Dict[str, Any]]] = []
    upper = lower
    for index in range(start_index, len(chain.entries)):
        rows, upper = chain.rows(index, max(lower, chain.segment(index)[0]), gaps)
        collected.extend(rows)
    # The boundary names the segment it sits in by start and first line: a later
    # pass proves the chain still holds that position before trusting it.
    seg_index = max(i for i in range(len(chain.entries)) if chain.segment(i)[0] <= upper)
    seg_start = chain.segment(seg_index)[0]
    accepted = {"version": OBSERVATION_BOUNDARY_VERSION, "upper": upper, "segment_start": seg_start,
                "segment_sha256": chain.first_line_sha256(seg_index)}
    return collected, {"lower": lower, "upper": upper, "basis": basis, "boundary": accepted}


_SYSTEM_ROW_KINDS = {"task_summary": "task_terminal", "quiz_answer": "card_answer",
                     "acceptance_late_settlement": "late_review", "task_error": "task_error"}


def _row_kind(row: Dict[str, Any]) -> str:
    """The provenance class of one chat row; "" for rows that are not wake events.

    Human input is classified by what its producer stamped, never by
    ``direction`` alone: A2A traffic is machine traffic, a Presence
    correspondent is not the owner, an Ouroboros-initiated Presence row is not
    a human, and a late card answer is carried by its ``quiz_answer`` row.
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    kind, direction = str(row.get("type") or ""), str(row.get("direction") or "")
    if kind in _SYSTEM_ROW_KINDS:
        return _SYSTEM_ROW_KINDS[kind]
    if direction != "in" or is_a2a_chat_id(row.get("chat_id")):
        return ""
    if str(row.get("client_message_id") or "").startswith("quiz_late_answer:"):
        return ""
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
    "late_review": ("late review", "late reviews"), "task_error": ("task runner failure", "task runner failures"),
    "card_state": ("card closed unanswered", "cards closed unanswered"),
    "outstanding_card": ("outstanding card", "outstanding cards"),
    "memory_change": ("memory change", "memory changes"),
}


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
    if kind == "card_answer":
        block = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
        return _answered_card_line(task_id, str(block.get("quiz_id") or "?"), block, now=now)
    if kind == "late_review":
        # A chat row whose canonical fact was not found: only what the row itself says.
        return f"- late review announced for task {task_id}, {when}: {_clip_preview(row.get('text'), 240)}"
    if kind == "task_error":
        return f"- task {task_id} runner failed, {when}: {_clip_preview(row.get('text'), 160)}"
    result = results.get(task_id) or {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    status = str(row.get("status") or result.get("status") or "settled")
    cost = result.get("accounted_upper_bound_usd", result.get("cost_usd"))
    cost_text = f", ${float(cost):.2f}" if isinstance(cost, (int, float)) else ""
    title = _clip_preview(result.get("description") or result.get("text") or row.get("text"), 100)
    project = str(row.get("project_id") or result.get("project_id") or "")
    where = f" in project {projects.get(project, project)}" if project else ""
    if kind == "direct_turn":
        return f"- {'wake' if is_consciousness_origin(metadata) else 'direct turn'} {task_id} {status}{where}, {when}"
    if is_presence_task(result):
        outcome = str(metadata.get("presence_outcome") or "unknown")
        work = str(metadata.get("presence_work_ref") or "")
        return (f"- Presence turn {task_id} {status}{where}, {when}; Presence outcome={outcome}; see get_task_result"
                + (f"; deferred work={work}" if work else ""))
    parent = str(row.get("parent_task_id") or result.get("parent_task_id") or "")
    role = str(row.get("delegation_role") or result.get("delegation_role") or "")
    noun = f"child task {task_id} of {parent}" if role == "subagent" and parent else f"task {task_id}"
    return f"- {noun} {status}{cost_text}{where}, {when}: {title}".rstrip(": ")


# --- canonical transitions: identities in the task results, not chat rows or stamps --------


def _transition_facts(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every transition one task result records, by identity.

    ``key -> fact`` for a terminal of one attempt (the terminal-publication fence's
    own attempt facts), each card's closed state, each late acceptance settlement
    (``acceptance_settlement.late_acceptance_facts``, its exact source ref consumed
    as published). Stamps ride along for display and the first (time-bootstrapped)
    window only: a terminal's is its publication instant when recorded, else the
    last update, never presented as the completion time. Closed tasks remain in
    the inventory: a new panel or a late owner answer can still add evidence.
    """
    from ouroboros.acceptance_settlement import late_acceptance_facts
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
            stamp = _parse_iso(block.get("answered_at") or block.get("reconciled_at") or "")
            facts[f"card:{task_id}:{quiz_id}:{state}"] = {
                "kind": "card_answer" if state == "answered" else "card_state", "task_id": task_id,
                "quiz_id": str(quiz_id), "block": block, "stamp": stamp, "proven": stamp is not None}
    try:
        late = late_acceptance_facts(row)
    except Exception:  # a malformed projection is a gap of that task, not of the wake
        late = []
    for fact in late:
        stamp = _parse_iso(fact.get("settled_at") or "")
        facts[f"late:{task_id}:{fact.get('panel_id')}:{fact.get('settled_at')}"] = {
            "kind": "late_review", "task_id": task_id, "fact": fact, "stamp": stamp, "proven": stamp is not None}
    return facts


def _transitions_since(transitions: Dict[str, Dict[str, Dict[str, Any]]], accepted: Any, *,
                       since: float, scan_at: float) -> Tuple[Dict[str, Dict[str, Any]], set, Dict[str, Any], str]:
    """Compare complete transition inventories; timestamps are display/bootstrap only.

    Keep every observed task's keys, including closed tasks. Absence from a
    complete accepted inventory then proves a newly observed result even if its
    first write copied an old timestamp. A new panel on an old task is likewise
    a new identity. The existing wake boundary owns this state, with no new
    store or count cutoff. Cost is one fact extraction per result per scan.

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
                     drive_root: pathlib.Path, now: float, projects: Dict[str, str]) -> Tuple[str, str]:
    """``(kind, line)`` for one canonical transition; a late review names its exact published source."""
    kind, task_id, stamp = fact["kind"], fact["task_id"], fact["stamp"]
    ago = _ago(now - stamp) if stamp is not None else "at an unknown time"
    if kind == "late_review":
        from ouroboros.artifacts import task_artifact_dir_path
        from ouroboros.task_finalization import review_source_reader

        record = fact["fact"]
        late = record.get("late_settlement") if isinstance(record.get("late_settlement"), dict) else {}
        emitted = late.get("emitted_answer") if isinstance(late.get("emitted_answer"), dict) else {}
        parts = [f"panel {record.get('panel_id') or '?'}", f"signal {record.get('aggregate_signal') or 'unknown'}",
                 f"reviewed version {late.get('reviewed_revision') or 'unknown'}",
                 f"emitted answer {emitted.get('state') or 'unknown'}",
                 f"{len(late.get('reviewer_outputs') or [])} reviewer outputs"]
        ref = record.get("source_ref") if isinstance(record.get("source_ref"), dict) else {}
        try:
            stored = task_artifact_dir_path(drive_root, task_id) / str(ref.get("path") or "")
            present = bool(ref.get("path")) and stored.is_file()
        except (ValueError, OSError):
            present = False
        read = review_source_reader(task_id, ref)
        parts.append(f"exact source {json.dumps(read, sort_keys=True)} sha256 {ref.get('sha256')}" if present else
                     "exact source unavailable" + (f" ({ref.get('path')})" if ref.get("path") else ""))
        first = str(late.get("note") or "").splitlines()[0] if late.get("note") else ""
        return kind, f"- late review settled for task {task_id}, {ago}: {_clip_preview(first, 240)}; " + "; ".join(parts)
    if kind == "card_answer":
        return kind, _answered_card_line(task_id, fact["quiz_id"], fact["block"], now=now)
    if kind == "card_state":
        from ouroboros.project_dialogue import owner_wait_projection, question_status

        block = fact["block"]
        label = question_status(str(block.get("state") or ""), owner_wait_projection(
            fact["quiz_id"], (results.get(task_id) or {}).get("owner_wait"), block), False)
        when = f", closed {ago}" if stamp is not None else ""
        return kind, (f"- owner card {fact['quiz_id']} on task {task_id}: {label}{when}; "
                      f"{_clip_preview(block.get('question') or 'question text unavailable')}")
    result = results.get(task_id) or {}
    when = ago if fact["proven"] or stamp is None else f"completion time not recorded (last updated {ago})"
    if result.get("_is_direct_chat") and not _presence_failure(result):
        kind = "direct_turn"
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
        """The whole observation, as the wake's own input when it fits."""
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
                if kind != "direct_turn":
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
    from ouroboros.task_results import list_task_results
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
        rows, scanned = list_task_results(root, strict=True), True
    except Exception as exc:  # a disclosed gap beats a missing wake
        rows, scanned = [], False
        gaps.add(f"task_results unreadable: {type(exc).__name__}")
    results = {str(row.get("task_id") or ""): row for row in rows}
    transitions = {task_id: _transition_facts(row) for task_id, row in results.items()}

    if source_unreadable:
        fresh, known, state, basis = {}, set(), None, "unreadable_transition_source"
    elif scanned:
        fresh, known, state, basis = _transitions_since(
            transitions, prior, since=since, scan_at=now)
    else:  # nothing was read: the accepted state stays, so the next wake finds the same transitions
        fresh, known, state, basis = {}, set(), prior, "unreadable_task_results"
    try:
        chat_rows, window = _chat_window(root, accepted or None, since, gaps)
    except Exception as exc:
        chat_rows, window = [], {"lower": None, "upper": None, "basis": "chat_chain_unreadable", "boundary": None}
        gaps.add(f"chat log unreadable: {type(exc).__name__}")
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
        retry = str(row.get("card_row_id") or "").removeprefix("acceptance-late:")
        key = next((key for key, fact in facts.items() if (
            kind == "task_terminal" and fact["kind"] == "task_terminal")
            or (kind == "card_answer" and key == f"card:{row.get('task_id')}:{quiz.get('quiz_id')}:answered")
            or (kind == "late_review" and fact["kind"] == "late_review" and retry and retry == str(
                ((fact["fact"].get("late_settlement") or {}).get("reviewed_subject") or {}).get("retry_key") or ""))),
            None)
        if key is not None:
            if key in announced or (key in known and key not in fresh):
                continue  # announced earlier in this window, or already accepted by a previous wake
            # A row for a transition the identity window classed as old (a result first
            # written late under an older stamp) is still new to this position: at least once.
            fact = fresh.setdefault(key, facts[key])
            announced.add(key)
            kind, line = _transition_line(fact, results, rooms, root, now, projects)
            events.append((kind, offset, line))
            continue
        result = results.get(str(row.get("task_id") or "")) or {}
        if kind == "task_terminal" and result.get("_is_direct_chat") and not _presence_failure(result):
            kind = "direct_turn"
        events.append((kind, offset, _event_line(kind, row, results, rooms, projects, now)))
    for key, fact in sorted(((key, fact) for key, fact in fresh.items() if key not in announced),
                            key=lambda item: (item[1]["stamp"] is None, item[1]["stamp"] or 0.0, item[0])):
        kind, line = _transition_line(fact, results, rooms, root, now, projects)
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
    trigger, _trigger_task = _trigger_line(projects, reason, rows, now=now)
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


def render_wake_message(repo_dir: Any, *, reason: str, last_wake_at: float, now: float, level: Any,
                        disabled_tools: List[str], spent_usd: Any, daily_usd: Any, running: int,
                        max_tasks: int, interval: int, events: str, spent_is_floor: bool = False) -> str:
    """Fill ``prompts/CONSCIOUSNESS.md`` for one wake; every placeholder is substituted."""
    template = safe_read(pathlib.Path(repo_dir) / PROMPT_REL) or _FALLBACK_TEMPLATE
    normalized = normalize_level(level)
    spent = f"{float(spent_usd):.2f}" if isinstance(spent_usd, (int, float)) else "unknown"
    if spent_is_floor and spent != "unknown":
        spent = f"at least {spent}"  # unmetered rows in the window: the number is a floor
    facts = {
        "reason": str(reason or "heartbeat"),
        "last_wake_ago": _ago(now - last_wake_at) if last_wake_at else "no wake since this process started",
        "events": events or "nothing new",
        "level": normalized, "level_line": LEVEL_LINES[normalized],
        "withheld_tools": ", ".join(disabled_tools) if disabled_tools else "none",
        "spent_usd": spent, "daily_usd": f"{float(daily_usd):.2f}",
        "running": str(int(running)), "max_tasks": str(int(max_tasks)), "interval": str(int(interval)),
    }
    # One pass: a fact (a task title inside {events}) that happens to contain "{daily_usd}"
    # is never substituted again.
    return re.sub(r"\{(\w+)\}", lambda m: facts.get(m.group(1), m.group(0)), template)
