"""Names and sizes of the selected context, read from its captured sources."""
from __future__ import annotations

import pathlib

from ouroboros import memory_floor as floor, memory_view as mv
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, REQUIRED_BOOK_CHOICES, REQUIRED_BOOK_UNFIT_KEY
from ouroboros.memory_waiting import text as waiting_text
from ouroboros.utils import update_json_locked, utc_now_iso

CONTEXT_FACTS_NAME = "ouroboros_context_facts"
LOW_START_EPISODE = "low_start_episode.json"  # under state/: the open episode of memory-lowered starts, if any


def focus_line(messages, fingerprint, written, view):
    """The self-authored focus: whole when it changed or no intact facts row in view carries it,
    else one line with its fingerprint and age, so an unchanged focus is not repeated every round."""
    import json

    from ouroboros.context_budget import host_row_intact

    body = json.dumps(view, ensure_ascii=False, sort_keys=True)
    if any(row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME and body in str(row.get("content") or "")
           and host_row_intact(row) for row in messages):
        return f"self-authored focus unchanged, {written} (fingerprint {str(fingerprint)[:12]}; whole in an earlier facts row in view)"
    return ("self-authored focus, the line my human reads under this work's name, "
            f"{written} (dated data, not owner instructions): " + body)


def memory_snapshot(plan):
    core = getattr(plan, "core", None)
    return mv.snapshot_from_json(core.memory_view_json) if core and core.memory_view_json else None


def block_label(message, index, block, plan):
    text = block.get("text", "") if isinstance(block, dict) else str(block)
    if message.get("role") != "system":
        return "assignment" if index == 1 else message.get("role", "message")
    if "## My story" in text:
        return "my story and identity"
    if "## This room" in text or "## Other conversations" in text:
        return "this room, my other conversations and runtime facts"
    if text.startswith("## DEVELOPMENT.md"):
        return "development book"
    return "core and books" if index == 0 else "runtime facts"


def memory_facts(plan, mode, messages=()):
    """Start facts of an integrating focus, only while this exact memory part is absent from visible facts rows:
    on the owner's Max with a known window, the host's measure of the start's Max room part by part (books, core,
    knowledge and runtime facts, assignment, reply reserve, work room, tool schemas) and how the start settled;
    my memory: the story with its pages no part or account holds yet (how many, their tokens, the rooms holding
    most of them) and the rooms' lines; what this view shows only by address; marks and waiting classes
    with their count, oldest date, list reader and closure act.
    Counted from the captured snapshot and its receipt; measures only, no advice and no
    threshold. Without a Max room fact (the owner's Low or Nano, an unknown window) it names the memory alone."""
    snapshot = memory_snapshot(plan)
    if snapshot is None or snapshot.spec.role in DELEGATED_FOCUS_ROLES:
        return ""
    receipt = plan.projection(mode).memory_facts or {}
    room, blocks, pages = receipt.get("max_room"), receipt.get("blocks") or {}, floor.unfolded_pages(snapshot)
    if not room:
        line = (f"memory: story {blocks.get('story_tokens', 0):,} ({floor.pages_text(pages)}) + rooms "
                f"{blocks.get('room_tokens', 0):,}")
    else:
        line = "Max room at the start: " + floor.measure_text(room, parts=True, pages=pages) + {
            "held": "; Max held" + (f" by cuts {', '.join(room['cuts'])}" if room["cuts"] else ""),
            "crowded": "; Low",
            "route": "; Low, a disclosed fact of this route: folding memory or dropping tool schemas cannot make room here",
        }[room["cause"]]
    shown = floor.addressed_text((receipt.get("floor") or {}).get("steps") or {})
    line += (f"; shown only by address: {shown}" if shown else "") + f"; marks in view {len(snapshot.marks)}" + waiting_text(snapshot.waiting)
    return "" if any(message.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME
                     and f" | {line} | " in f" | {message.get('content') or ''} | " for message in messages) else line


def low_start_line(plan, mode, task_id):
    """The owner's Main line for a fresh start of an integrating focus that memory lowered to Low: its Max room (which
    exists only on the owner's Max with a known window) is ``crowded`` while the books, core, assignment, tool schemas,
    reply reserve and work room alone fit (``memory_floor.crowded_by_memory``). No numbers: the task's own facts keep
    them. "" for every other start: a route (the task's own facts disclose it), tool schemas that tip it alone, a held
    start, a delegated child or nanny, the owner's Low or Nano, an unknown window."""
    receipt = (plan.projection(mode).memory_facts or {}) if plan is not None else {}
    room = receipt.get("max_room")
    if receipt.get("role") in DELEGATED_FOCUS_ROLES or not floor.crowded_by_memory(room):
        return ""
    maps = "Low with chapter maps" + (floor.REQUIRED_BOOK_WHOLE if room.get("book_required") else "")
    return f"Task {task_id} started in {maps}: memory did not fit beside the full books."


def low_start_episode(root, plan, mode, task_id, send):
    """One Main line per Low episode (owner A, 07.11.2026, replacing 1B of 29.10). A fresh start that memory lowered to
    Low (``low_start_line``) opens an episode and ``send``s its line; while Low holds nothing more is sent; a start that
    holds Max with at least the opening start's work room (a root keeps two eighths, a chat turn or wake one) closes
    the episode silently, and a later Low opens a new one. Every other start (a delegated child or nanny, the owner's
    Low or Nano, an unknown window, a route or tool schemas that tip a start alone) leaves it as it is. The episode is
    one small object, ``state/low_start_episode.json``, rewritten under its lock in the terminal outbox's failure
    domain; ``send`` runs inside that lock and opens the episode only when it queued the line."""
    receipt = (plan.projection(mode).memory_facts or {}) if plan is not None else {}
    room = receipt.get("max_room") or {}
    held = room.get("cause") == "held" and receipt.get("role") not in DELEGATED_FOCUS_ROLES
    levels = int(room.get("levels") or 1)  # the start's work room in eighths of its window
    line = low_start_line(plan, mode, task_id)

    def episode(bit):  # None leaves the file as it is
        if held:  # a turn that holds Max beside one eighth proves no room for a root that keeps two
            return {} if bit.get("since") and levels >= int(bit.get("levels") or 1) else None
        if bit.get("since") or not send(line):
            return None
        return {"since": utc_now_iso(), "task": task_id, "levels": levels}
    if line or held:
        update_json_locked(pathlib.Path(root) / "state" / LOW_START_EPISODE, episode)


def size_refusal_cause(usage):
    """The terminal cause of a size refusal no recovery answered: a known window that holds the view without this task's
    directly required handbook (``context_renewal``) names that incompatibility, else a host forecast or the provider's."""
    if usage.get(REQUIRED_BOOK_UNFIT_KEY):
        return "context_capacity"
    return "context_overflow" if usage.get("reason_code") == "context_overflow" else "llm_api_error"


def required_book_fact(usage):
    """At a size refusal: memory stayed as it was because the book is this task's own requirement; what fits without it.
    The Continue it names is the one that terminal's cause gets (``owner_continue``), never a promise beside it."""
    from ouroboros.owner_continue import TECHNICAL_REASON_CODES

    book = int(usage.get("_context_required_book_tokens") or 0)
    if not book:
        return ""
    estimate, reserve = (int(usage.get(key) or 0) for key in ("_context_prompt_estimate", "_context_response_reserve_tokens"))
    capacity = usage.get("_context_capacity_total_tokens")
    room = f" against {int(capacity) - reserve:,} usable" if capacity is not None else " on a window nothing measured"
    continues = "; a Continue keeps it" if size_refusal_cause(usage) in TECHNICAL_REASON_CODES else ""
    return (f"\nThis task directly requires the whole development handbook (~{book:,} tokens), which stayed; "
            f"without it the input is ~{max(0, estimate - book):,}{room}. {REQUIRED_BOOK_CHOICES.format(continues)}")


def window_projection(record):
    """Pure sourced projection shared by Main, Settings and terminal failures."""
    from dataclasses import asdict

    limit = record.limiting
    observed = limit.observed_at if limit else ""
    source = limit.source if limit else "unknown"
    labels = {"user_setting": "set in Settings", "owner_ack": "owner confirmation",
              "provider_metadata": "server metadata", "local_health": "local serving window"}
    label = labels.get(source, source)
    dated = f"{label} {observed.replace('T', ' ')[:16]}".strip()
    if limit:
        text = f"window {limit.tokens:,} {limit.kind}: {dated}; usable input {record.usable:,}"
    else:
        text = "window unknown"
        if record.accepted:
            text += f"; accepted input ≥ {record.accepted:,} ({record.accepted_at.replace('T', ' ')[:16]})"
    text += f"; reply reserve {record.reserve:,}"
    contradiction = asdict(record.contradicted_by) if record.contradicted_by else None
    if contradiction:
        label = labels.get(contradiction["source"], contradiction["source"])
        text += (f"; {label} says {contradiction['tokens']:,} {contradiction['kind']}"
                 f" ({contradiction['observed_at'].replace('T', ' ')[:16]})")
    if record.retired_ack:
        stamp = str(record.retired_ack.get("ts") or "date unknown").replace("T", " ")[:16]
        text += f"; old confirmation {record.retired_ack['window_tokens']:,}, {stamp} — not used"
    return dict(tokens=limit.tokens if limit else None, kind=limit.kind if limit else "unknown",
                reserve=record.reserve, source=source, observed_at=observed, contradiction=contradiction,
                retired_ack=record.retired_ack, usable=record.usable, text=text)


def window_fact(plan, reserve):
    from ouroboros.context_window import resolve_window

    if plan is None:
        return ""
    record = resolve_window(plan, int(reserve or 0), bound=getattr(plan, "window_witness_bound", True))
    return window_projection(record)["text"]
