"""Names and sizes of the selected context, read from its captured sources."""
from __future__ import annotations

from ouroboros import memory_floor as floor, memory_view as mv
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.memory_waiting import text as waiting_text

CONTEXT_FACTS_NAME = "ouroboros_context_facts"


def memory_snapshot(plan):
    core = getattr(plan, "core", None)
    return mv.snapshot_from_json(core.memory_view_json) if core and core.memory_view_json else None


def block_label(message, index, block, plan):
    text = block.get("text", "") if isinstance(block, dict) else str(block)
    if message.get("role") != "system":
        return "assignment" if index == 1 else message.get("role", "message")
    if "## My story" in text:
        return "my story and identity"
    if "## This room" in text or "## Live rooms" in text:
        return "this room, live rooms and runtime facts"
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
    """The owner's Main line for one fresh start of an integrating focus that memory lowered to Low (owner decision
    1B, 29.10): its Max room (which exists only on the owner's Max with a known window) is ``crowded`` while the
    books, core, assignment, tool schemas, reply reserve and work room alone fit (``memory_floor.crowded_by_memory``).
    The host's measure, no advice. "" for every other start: a route (the task's own facts disclose it), tool
    schemas that tip it alone, a held start, a delegated child or nanny, the owner's Low or Nano, an unknown window."""
    receipt = (plan.projection(mode).memory_facts or {}) if plan is not None else {}
    room = receipt.get("max_room")
    if receipt.get("role") in DELEGATED_FOCUS_ROLES or not floor.crowded_by_memory(room):
        return ""
    return (f"Task {task_id} started in Low with chapter maps: its memory ({room['story'] + room['room']:,} tokens) "
            f"did not fit the {room['allowance']:,} tokens left beside the full books.")


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
