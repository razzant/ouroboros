"""The wake-up message and the wake's task metadata (Background Consciousness redesign).

A wake-up is an ordinary Main turn nobody typed: ``prompts/CONSCIOUSNESS.md`` is its USER
message (system prompt, memory and tools are Main's own, owner decision В15).
``render_wake_message`` fills its placeholders from existing readers (tasks settled and owner
cards answered since the last wake, open owner quiz cards, the count of owner messages) and
truncates the event list with an explicit source pointer, never silently (BIBLE P1).
``wake_task_metadata`` is the wake's origin/authority envelope for ``handle_wake_direct`` (``consciousness_authority``).
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import re
from typing import Any, Dict, List, Optional

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
from ouroboros.utils import iter_jsonl_objects

PROMPT_REL = pathlib.Path("prompts") / "CONSCIOUSNESS.md"
EVENT_LINES_MAX = 10
CARD_LINES_MAX = 4  # unanswered cards never crowd out what settled since the last wake
CHAT_TAIL_BYTES = 512_000
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


def _project_name(drive_root: pathlib.Path, project_id: str) -> str:
    try:
        from ouroboros.projects_registry import get_project

        row = get_project(drive_root, project_id) or {}
        return str(row.get("name") or project_id)
    except Exception:
        return project_id


def _trigger_line(
    drive_root: pathlib.Path, reason: str, rows: List[Dict[str, Any]], *, now: float,
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
            project = _project_name(drive_root, project_id)
            stamp = _parse_iso(row.get("updated_at") or row.get("ts"))
            age_text = f", {_ago(now - stamp)}" if stamp is not None else ""
            detail = f"; {title}" if title else ""
            qualifier = "settled task" if trigger_task_id else "latest settled task"
            return f"- wake cause: project {project} {qualifier} {task_id} ({status}){cost_text}{age_text}{detail}", task_id
        return f"- wake cause: project {_project_name(drive_root, project_id)} reported a settled task (details unavailable)", ""
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


def wake_events(
    drive_root: Any, *, since: float, now: float, reason: str = "", exclude_task_id: str = "",
) -> List[str]:
    """Render a trigger-first, bounded view of fresh facts and outstanding cards.

    Settled task rows and answered cards are filtered by ``since`` (an answer is an
    event of the window, sorted with the settled facts by ``answered_at``). Answerable
    cards intentionally span the full store because ``expired_terminal`` still accepts
    a late owner answer (В17a), but they are rendered after the fresh trigger/facts and carry their semantic state.
    """
    from ouroboros.owner_quiz import STATE_ANSWERED, STATE_EXPIRED_TERMINAL, STATE_OPEN
    from ouroboros.task_results import list_task_results
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_finalization import HOST_AUTHORED_TERMINAL_ORIGINS

    root, since_iso, lines, read_errors = pathlib.Path(drive_root), _iso(since), [], []
    try:
        rows = list_task_results(root)
    except Exception as exc:  # a disclosed gap beats a missing wake
        rows, read_errors = [], [f"- task_results unreadable: {type(exc).__name__}"]
    cards, settled = [], []  # (stamp, line) / (stamp, task_id, line) pairs
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        quizzes = row.get("owner_quiz") if isinstance(row.get("owner_quiz"), dict) else {}
        for quiz_id, block in quizzes.items():
            if not isinstance(block, dict):
                continue
            if block.get("answered_at"):
                # An answer given inside the window is an event of the window: it sorts
                # with the settled facts by its own stamp (no task id, so the trigger's
                # de-duplication never hides it). Older answers are not news.
                answered_at = str(block.get("answered_at") or "")
                answered_ts = _parse_iso(answered_at)  # an unreadable stamp cannot be placed in the window
                if block.get("state") == STATE_ANSWERED and answered_ts is not None and answered_ts >= since:
                    settled.append((_iso(answered_ts), "", _answered_card_line(task_id, str(quiz_id), block, now=now)))
                continue
            if block.get("state") in (STATE_OPEN, STATE_EXPIRED_TERMINAL):
                cards.append((
                    str(block.get("asked_at") or ""),
                    _card_line(task_id, str(quiz_id), block, now=now, owner_wait=row.get("owner_wait")),
                ))
        status, stamp = str(row.get("status") or ""), str(row.get("updated_at") or row.get("ts") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        presence_failure = is_presence_task(row) and not is_consciousness_origin(metadata) and (
            status in {"failed", "cancelled"}
            or row.get("terminal_origin") in HOST_AUTHORED_TERMINAL_ORIGINS
        )
        # Inline Presence shares the direct-turn lane, but its host failure is
        # absent from external dialogue. Surface the existing task, not a retry.
        if task_id == exclude_task_id or (row.get("_is_direct_chat") and not presence_failure):
            continue
        if status in SETTLED_STATUSES and stamp >= since_iso:
            cost = row.get("accounted_upper_bound_usd", row.get("cost_usd"))
            cost_text = f", ${float(cost):.2f}" if isinstance(cost, (int, float)) else ""
            title = _clip_preview(row.get("description") or row.get("text") or row.get("result"), 80)
            detail = ""
            if presence_failure:
                outcome = str(metadata.get("presence_outcome") or "unknown")
                work = str(metadata.get("presence_work_ref") or "")
                detail = f"; Presence outcome={outcome}; see get_task_result"
                if work:
                    detail += f"; deferred work={work}"
            settled.append((stamp, task_id, f"- task {task_id} {status}{cost_text}: {title}".rstrip(": ") + detail))
    trigger, trigger_task_id = _trigger_line(root, reason, rows, now=now)
    if trigger:
        lines.append(trigger)
    lines += read_errors
    lines += [line for _stamp, task_id, line in sorted(settled, reverse=True)
              if not trigger_task_id or task_id != trigger_task_id]
    owner_messages = 0
    try:
        for entry in iter_jsonl_objects(root / "logs" / "chat.jsonl", tail_bytes=CHAT_TAIL_BYTES):
            if entry.get("direction") == "in" and str(entry.get("ts") or "") >= since_iso:
                owner_messages += 1
    except Exception:
        lines.append("- chat log unreadable")
    if owner_messages:
        lines.append(f"- {owner_messages} message(s) from your human (see Recent chat)")
    # Outstanding cards are intentionally bounded, but no longer outrank the fresh
    # event that caused this wake. Their canonical lifecycle wording makes late-answer
    # semantics visible without creating a second question-state vocabulary.
    lines += [line for _stamp, line in sorted(cards, reverse=True)[:CARD_LINES_MAX]]
    return lines


def render_wake_message(drive_root: Any, repo_dir: Any, *, reason: str, last_wake_at: float, since: float,
                        now: float, level: Any, disabled_tools: List[str], spent_usd: Any, daily_usd: Any,
                        running: int, max_tasks: int, interval: int, exclude_task_id: str = "",
                        spent_is_floor: bool = False) -> str:
    """Fill ``prompts/CONSCIOUSNESS.md`` for one wake; every placeholder is substituted."""
    template = safe_read(pathlib.Path(repo_dir) / PROMPT_REL) or _FALLBACK_TEMPLATE
    events = wake_events(drive_root, since=since, now=now, reason=reason, exclude_task_id=exclude_task_id)
    omitted = max(0, len(events) - EVENT_LINES_MAX)
    shown = events[:EVENT_LINES_MAX] + ([f"(+{omitted} more; see recent_tasks, get_task_result, and chat_history)"] if omitted else [])
    normalized = normalize_level(level)
    spent = f"{float(spent_usd):.2f}" if isinstance(spent_usd, (int, float)) else "unknown"
    if spent_is_floor and spent != "unknown":
        spent = f"at least {spent}"  # unmetered rows in the window: the number is a floor
    facts = {
        "reason": str(reason or "heartbeat"),
        "last_wake_ago": _ago(now - last_wake_at) if last_wake_at else "no wake since this process started",
        "events": ("\n" + "\n".join(shown)) if shown else "nothing new",
        "level": normalized, "level_line": LEVEL_LINES[normalized],
        "withheld_tools": ", ".join(disabled_tools) if disabled_tools else "none",
        "spent_usd": spent, "daily_usd": f"{float(daily_usd):.2f}",
        "running": str(int(running)), "max_tasks": str(int(max_tasks)), "interval": str(int(interval)),
    }
    # One pass: a fact (a task title inside {events}) that happens to contain "{daily_usd}"
    # is never substituted again.
    return re.sub(r"\{(\w+)\}", lambda m: facts.get(m.group(1), m.group(0)), template)
