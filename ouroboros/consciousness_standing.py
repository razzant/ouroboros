"""My human's answerable cards as every own-clock wake shows them, and the card lines its observation prints.

A wake's observation is a delta (``consciousness_wake.observe_wake``): each transition reaches one accepted wake.
After it every wake shows my human's answerable cards (``outstanding_cards``); the card line renderers live here,
also for the card transitions the observation prints.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Mapping, Optional, Tuple

from ouroboros.deadline_utils import ago_text


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


# --- my human's cards --------------------------------------------------------------------


def _card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float, owner_wait: Any = None,
               closed: Optional[str] = None) -> str:
    """One card's status line; ``closed`` (a transition's own stamp, or none) replaces its asked age and wait."""
    from ouroboros.project_dialogue import owner_wait_projection, question_status

    facts = owner_wait_projection(quiz_id, owner_wait, block)
    waits = closed is None and bool(block.get("wait_for_answer"))
    label = question_status(str(block.get("state") or "open"), facts, waits)
    age = closed
    if age is None:  # still open: its asked age
        asked_ts = _parse_iso(str(block.get("asked_at") or ""))
        age = f", {ago_text(now - asked_ts)}" if asked_ts is not None else ""
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {label}{age}; {preview}"


def _answered_card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float) -> str:
    """One answered card of the window: stamps, the recorded answer whole (my human's own words verbatim, so in the
    fallback form too; a press as their act on the asking task's label, ``answer_facts.option_answer``), a named
    question preview and no verdict on what the answer meant. A secret request's answer is the
    host's typed outcome: its event, plugin and field ids, never a value."""
    from ouroboros.answer_facts import option_answer
    from ouroboros.owner_quiz import secret_outcome_block, secret_outcome_summary

    parts = []
    for key, word in (("answered_at", "answered"), ("asked_at", "asked")):
        stamp = _parse_iso(block.get(key))
        parts.append(f"{word} {ago_text(now - stamp)}" if stamp is not None else f"{word} at an unknown time")
    index = block.get("answered_index")
    comment = str(block.get("comment") or "")
    if secret_outcome_block(block) is block:
        answer = f"host outcome {secret_outcome_summary(block)}"
    elif isinstance(index, int) and not isinstance(index, bool):
        answer = option_answer(block, index, comment, asker=task_id)
    elif comment.strip():
        answer = f"answered in own words: {comment}"
    else:
        answer = "answer text unavailable"
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {'; '.join(parts)}; {answer}; question: {preview}"


def _closed_card_line(task_id, quiz_id, block, *, now):
    stamp = _parse_iso(block.get("closed_at"))
    when = ago_text(now - stamp) if stamp is not None else "at an unknown time"
    actor = str(block.get("closed_actor") or "unknown actor")
    if actor == "model" and block.get("closed_by_task_id"):
        actor += f" task {block['closed_by_task_id']}"
    return (f"- owner card {quiz_id} on task {task_id}: closed {when} by {actor}, without an owner answer; "
            f"reason: {block.get('closed_reason') or 'unavailable'}; "
            f"question: {_clip_preview(block.get('question') or 'question text unavailable')}")


def outstanding_cards(results: Mapping[str, Dict[str, Any]], *, now: float) -> Tuple[str, ...]:
    """Every answerable card (``open``/``expired_terminal`` — В17a) of the rows read, newest asked first."""
    from ouroboros.owner_quiz import STATE_EXPIRED_TERMINAL, STATE_OPEN

    cards = []
    for task_id, row in results.items():
        quizzes = row.get("owner_quiz") if isinstance(row.get("owner_quiz"), dict) else {}
        for quiz_id, block in quizzes.items():
            if isinstance(block, dict) and not block.get("answered_at") and block.get("state") in (
                    STATE_OPEN, STATE_EXPIRED_TERMINAL):
                cards.append((str(block.get("asked_at") or ""), _card_line(
                    task_id, str(quiz_id), block, now=now, owner_wait=row.get("owner_wait"))))
    return tuple(line for _stamp, line in sorted(cards, reverse=True))
