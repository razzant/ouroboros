"""Room-isolated dialogue consolidation: partition, prompts, correction, assembly.

One logical dialogue block is summarized one room at a time from that room's
exact chronological source, and every successful summary unit is compared once
against the same complete bytes it was written from before anything is kept.
The host owns room identity: it partitions by the actual ``chat_id`` before any
model call, stamps the typed ``rooms`` sections and their deterministic
Markdown projection, and never parses generated text for labels. Episodic
claims stay grounded in that room's source; cumulative knowledge revisions
also require the correcting operation's own complete current-note read.
An era regroups the same recorded room across blocks, and a legacy record stays one explicitly
unknown-provenance section. The Light transport (route, fit, retained sources,
typed failures) stays with ``consolidator.py``; this module receives it as one
``call`` function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.dialogue_provenance import source_continuation_note

LEGACY_ROOM_ID = "legacy"
LEGACY_ROOM_LABEL = "Unknown provenance [legacy mixed record]"

# One ``call(prompt, label, *, fixed_prompt, input_limit, call_type)`` returns
# ``(content, usage, knowledge)``: content is empty on any typed failure, usage
# carries ``_consolidation_errors``, knowledge is the read context that made the
# call (None without a knowledge context) and binds its own nominations.
LightCall = Callable[..., Tuple[str, Dict[str, Any], Any]]

FIDELITY_RULES = """## Fidelity
- Keep every actor exact: who asked, decided, approved, reported. The owner's words stay the owner's; my proposals, options, questions and recommendations stay mine; a task, subagent or review report quoted in a message is evidence attributed to that task, never new owner authorization.
- "Agree with all recommendations" binds only to recommendations actually stated earlier in this source; name them from the source, never from memory.
- Retain concrete limits and obligations with their units and conditions: budgets, deadlines, checkpoints, publication/merge/live-update boundaries, pending independent reviews, unfinished deliveries, negative constraints.
- Distinguish proposed, approved, implemented, verified, published and rejected; infer no approval from silence; unknown stays unknown.
- This memory is my interpretation of what was said, never an authorization by itself."""

DRAFT_SOURCE_HEADING = "## Messages to summarize"
CORRECTION_SOURCE_HEADING = "## Complete source"


@dataclass
class RoomSource:
    """One room's exact share of a block: source order kept, bytes formatted once."""

    room_id: str
    label: str
    entries: List[Dict[str, Any]]
    text: str = ""
    spans: List[Tuple[int, int, str]] = field(default_factory=list)


def partition_entries(entries: List[Dict[str, Any]], resolver: Any) -> List[RoomSource]:
    """Group a block's messages by actual room before any model call.

    Rooms keep their order of first appearance; messages keep the chat log's
    own chronological order inside each room. Every message lands in exactly
    one room, so the rooms' sources concatenate to the whole block's source.
    """
    from ouroboros.consolidator import _format_entries_for_block

    rooms: Dict[str, RoomSource] = {}
    for entry in entries:
        room_id = resolver.room_id(entry)
        if room_id not in rooms:
            rooms[room_id] = RoomSource(room_id, resolver.label(entry), [])
        rooms[room_id].entries.append(entry)
    for room in rooms.values():
        room.text = _format_entries_for_block(
            room.entries, include_room_labels=True, room_resolver=resolver, source_spans=room.spans,
        )
    return list(rooms.values())


def block_range(first_ts: str, last_ts: str) -> str:
    first_date, last_date = first_ts[:10], last_ts[:10]
    first_time, last_time = first_ts[11:16], last_ts[11:16]
    if first_date == last_date:
        return f"{first_date} {first_time} - {last_time}"
    return f"{first_date} {first_time} - {last_date} {last_time}"


def _identity_section(identity_text: str) -> str:
    return f"\n## Identity context\n{identity_text}\n" if identity_text else ""


def room_draft_prompt(
    source: str, *, room_label: str, block_range_text: str, message_count: int,
    identity_text: str = "", continuation_note: str = "", knowledge_instruction: str = "",
) -> str:
    # Room labels are owner-authored project names: data, quoted as one JSON
    # string so a label cannot read as prompt structure or a header.
    room_label = json.dumps(str(room_label), ensure_ascii=False)
    return f"""{knowledge_instruction}You are the memory consolidator of Ouroboros, a self-modifying AI agent.
Write the episodic memory of one room's messages inside the dialogue block {block_range_text}.
Room: {room_label}. This room contributed {message_count} messages; other rooms of the block are written separately and the host assembles them.
The source may be one contiguous part of the room; the episodic summary covers only the supplied part. Existing knowledge may inform a cumulative note update, but must not become an event or approval in this episode.

## Rules
1. No block or room headers; the host writes them. Start with the memory itself.
2. Preserve: decisions, agreements, technical discoveries, emotional and personal moments (what someone felt, asked for, enjoyed or disliked — quote them), task outcomes, what worked/failed.
3. Compress: routine tool calls, repetitive back-and-forth.
4. Quote key phrases directly when important; include task_ids when referencing specific tasks.
5. First person as Ouroboros: "I did..."; call people by the names the messages give; when no name is known, describe the speaker honestly rather than inventing one.
6. Adapt length to content density; no fixed word range.
{FIDELITY_RULES}
{_identity_section(identity_text)}
{continuation_note}{DRAFT_SOURCE_HEADING}
{source}
"""


def correction_prompt(
    draft: str, source: str, *, room_label: str, scope: str,
    identity_text: str = "", continuation_note: str = "", knowledge_instruction: str = "",
) -> str:
    room_label = json.dumps(str(room_label), ensure_ascii=False)
    knowledge_check = ("If the draft has a `KNOWLEDGE_ENTRIES_JSON:` block, check those cumulative updates "
                       "against each complete current note you read YOURSELF and this episode. The draft is a "
                       "proposal, not a source. After the episodic memory, return only draft-nominated topics "
                       "(including proposed new notes), or drop them. Unsupported episode claims must not survive in a note; "
                       "independently established knowledge may remain without becoming an event or approval "
                       "in this episode.\n" + knowledge_instruction if knowledge_instruction else "")
    return f"""Compare this draft memory of Ouroboros against its complete source and return the corrected memory.
Scope: {scope}; room: {room_label}. The source below is complete for this episodic summary, not for cumulative knowledge; the host assembles rooms and headers separately.
Check sentence by sentence. Fix misattributed actors or approvals; decisions moved between rooms, tasks or people; invented, dropped or altered budgets, deadlines, checkpoints, boundaries and obligations; completion, review, verification or publication the source does not show; anything called approved that the source shows proposed, asked or rejected.
Keep the first-person Ouroboros voice, quotes, task_ids and everything the draft got right; adapt length to the content.
{FIDELITY_RULES}
Return only the corrected memory text: no headers, commentary or diff. If the draft is already faithful, return it unchanged.
{knowledge_check}
{_identity_section(identity_text)}
{continuation_note}## Draft memory
{draft}

{CORRECTION_SOURCE_HEADING}
{source}
"""


def era_room_prompt(
    sections: str, *, room_label: str, start_date: str, end_date: str,
    identity_text: str = "", legacy: bool = False,
) -> str:
    room_label = json.dumps(str(room_label), ensure_ascii=False)
    legacy_note = ("These sections come from legacy records whose messages' rooms were not recorded: "
                   "keep their provenance unknown and assign no decision to a named room.\n") if legacy else ""
    return f"""Compress these older memory blocks of one room into a single era section.
Room: {room_label}. Era: {start_date} to {end_date}. The sections below are this room's parts of consecutive blocks, oldest first; other rooms are compressed separately and the host assembles them.
Preserve: key decisions, personality discoveries, relationship moments, technical milestones, the latest status and open commitments.
Drop: debugging details, routine operations, redundant info.
No headers; one first-person Ouroboros voice; adapt length to the meaningful content.
{FIDELITY_RULES}
{legacy_note}
## Sections to compress

{sections}
{_identity_section(identity_text)}"""


def split_source_text(text: str, boundaries: Tuple[int, ...] = ()) -> Optional[Tuple[str, str]]:
    """Split a source payload near its midpoint without dropping any bytes."""
    if len(text) < 2:
        return None
    midpoint = len(text) // 2
    radius = max(1, len(text) // 4)
    # Only formatter-supplied boundaries are messages; blank lines can be body.
    candidates = [p for p in boundaries if 0 < p < len(text) and abs(p - midpoint) <= radius]
    split_at = min(candidates, key=lambda p: abs(p - midpoint)) if candidates else midpoint
    if not 0 < split_at < len(text):
        return None
    return text[:split_at], text[split_at:]


def _extract_nominations(content: str, knowledge: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Bind a trailing nomination list through the read context that made the call."""
    if knowledge is None:
        return content, []
    from ouroboros.reflection import _extract_trailing_json

    content, entries = _extract_trailing_json(content, "KNOWLEDGE_ENTRIES_JSON:")
    return content, knowledge.bind_entries(entries)


def summarize_source(
    call: LightCall, text: str, spans: List[Tuple[int, int, str]],
    draft_prompt: Callable[[str, str], str], correct_prompt: Callable[[str, str, str], str],
    *, input_limit: Optional[Dict[str, Any]] = None,
    on_refusal: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Draft and then correct one exact source, splitting only to fit its Light route.

    Each part is drafted, then compared against its own complete bytes in one
    correction call; the corrected text is what survives. A correction that does
    not fit splits the part exactly like a draft refusal would, so no correction
    ever sees a clipped source, and each split halves the part until the route's
    fixed prompt alone is the remaining excess — no other repair loop exists.
    Parts cover the source in order without clipping; any failed or empty
    response withholds the whole source. A real refusal lowers the same route's
    byte limit for remaining parts and the next cycle. Knowledge nominations
    are released only from the CORRECTED response: the draft's trailing block
    travels into the correction, where cumulative revisions use its own note
    reads alongside the episode rather than inheriting the draft's read credit.
    """
    pending, summaries, usages = [(0, len(text))], [], []
    entries: List[Dict[str, Any]] = []
    from ouroboros.consolidator import _merge_consolidation_usage

    def result(content: str) -> Tuple[str, Dict[str, Any]]:
        return content, {**_merge_consolidation_usage(*usages), "_consolidation_retry": input_limit,
                         **({"_knowledge_entries": entries} if entries else {})}

    def split(start: int, end: int, failure: Dict[str, Any], *, correction: bool = False) -> bool:
        nonlocal input_limit
        if not failure["preflight_only"] and "input_bytes" in failure:
            input_limit = {key: failure[key] for key in ("route_fp", "capacity_tokens", "output_reserve_tokens")}
            input_limit["input_bytes"] = failure["input_bytes"] - 1
            failure["byte_limit"] = input_limit["input_bytes"]
            if on_refusal is not None:
                on_refusal(input_limit)
        halves = split_source_text(text[start:end], tuple(a - start for a, _, _ in spans))
        # A correction's fixed prefix carries the whole draft, so "fixed alone
        # exceeds the limit" says nothing about re-drafting smaller halves; only
        # a DRAFT prefix that cannot fit makes the part unsplittable.
        if halves is None or (not correction and any(
                failure.get(limit) is not None and failure[fixed] > failure[limit]
                for fixed, limit in (("fixed_tokens", "input_limit"), ("fixed_bytes", "byte_limit")))):
            return False
        midpoint = start + len(halves[0])
        pending.extend([(midpoint, end), (start, midpoint)])
        # The refusal is answered by its halves, each accounted by its own row; the
        # attempt stays in the usage history, never read as an unresolved failure.
        failure["resolution"] = "split"
        return True

    while pending:
        start, end = pending.pop()
        part, note = text[start:end], source_continuation_note(spans, start, end)
        draft, usage, _draft_knowledge = call(draft_prompt(part, note), "Room summary", fixed_prompt=draft_prompt("", note),
                                             input_limit=input_limit, call_type="memory_consolidation")
        usages.append(usage)
        if draft.strip():
            corrected, usage, knowledge = call(
                correct_prompt(draft, part, note), "Room correction", fixed_prompt=correct_prompt(draft, "", note),
                input_limit=input_limit, call_type="memory_correction")
            usages.append(usage)
            if corrected.strip():
                # A part's nominations are the corrected response's, released
                # only with its corrected text: a draft whose correction failed
                # (and was then split) never entered the block, and a draft
                # nomination the correction dropped was never source-checked.
                # Keep this correction scoped to the draft's nominated topics.
                # Only its OWN complete reads can authorize existing-note updates;
                # draft read credit says nothing about the correction's evidence.
                from ouroboros.reflection import _extract_trailing_json

                _, draft_raw = _extract_trailing_json(draft, "KNOWLEDGE_ENTRIES_JSON:")
                draft_topics = {(str(e.get("topic") or ""), str(e.get("scope") or ""))
                                for e in (draft_raw if isinstance(draft_raw, list) else []) if isinstance(e, dict)}
                corrected, raw = _extract_trailing_json(corrected, "KNOWLEDGE_ENTRIES_JSON:")
                kept = [e for e in (raw if isinstance(raw, list) else []) if isinstance(e, dict)
                        and (str(e.get("topic") or ""), str(e.get("scope") or "")) in draft_topics]
                # Provenance is per nomination: the route that answered THIS part's
                # correction wrote these entries, whatever route the block's other
                # rooms or parts ran on (a wait may rebind between parts).
                from ouroboros.knowledge import observed_route_stamp
                route = observed_route_stamp(usage)
                entries.extend({**entry, "_nomination_route": route}
                               for entry in (knowledge.bind_entries(kept) if knowledge is not None and kept else []))
                summaries.append(corrected.strip())
                continue
        failure = usage["_consolidation_errors"][-1]
        if failure["kind"] != "context_overflow" or not split(start, end, failure, correction=bool(draft.strip())):
            return result("")
    return result("\n\n".join(summaries))


def render_sections(header: str, rooms: List[Dict[str, Any]]) -> str:
    """The deterministic one-biography projection consumers read; labels are host facts."""
    parts = [header]
    for room in rooms:
        parts.append(f"#### [room={room['label']}] · {room['message_count']} messages\n\n{room['content']}")
    return "\n\n".join(parts)


def summarize_block(
    call: LightCall, rooms: List[RoomSource], *, first_ts: str, last_ts: str, identity_text: str = "",
    knowledge_instruction: str = "", input_limit: Optional[Dict[str, Any]] = None,
    on_refusal: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Summarize every room of one logical block; any room failure withholds the block.

    Returns ``({"range", "content", "rooms"}, usage)`` or ``(None, usage)``. The
    usage always carries ``_consolidation_retry`` and, when any call nominated
    knowledge, ``_knowledge_entries`` — the caller applies nominations only
    for a block that was published.
    """
    from ouroboros.consolidator import _merge_consolidation_usage

    range_text = block_range(first_ts, last_ts)
    sections: List[Dict[str, Any]] = []
    usages: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []

    def finish(block: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        return block, {**_merge_consolidation_usage(*usages), "_consolidation_retry": input_limit,
                       **({"_knowledge_entries": entries} if entries else {})}

    for room in rooms:
        def draft(part: str, note: str, room: RoomSource = room) -> str:
            return room_draft_prompt(part, room_label=room.label, block_range_text=range_text,
                                     message_count=len(room.entries), identity_text=identity_text,
                                     continuation_note=note, knowledge_instruction=knowledge_instruction)

        def correct(draft_text: str, part: str, note: str, room: RoomSource = room) -> str:
            return correction_prompt(draft_text, part, room_label=room.label, scope=f"dialogue block {range_text}",
                                     identity_text=identity_text, continuation_note=note,
                                     knowledge_instruction=knowledge_instruction)

        content, usage = summarize_source(call, room.text, room.spans, draft, correct,
                                          input_limit=input_limit, on_refusal=on_refusal)
        usages.append(usage)
        input_limit = usage["_consolidation_retry"]
        entries.extend(usage.get("_knowledge_entries") or [])
        if not content.strip():
            return finish(None)
        sections.append({"room_id": room.room_id, "label": room.label,
                         "message_count": len(room.entries), "content": content.strip()})
    return finish({"range": range_text, "content": render_sections(f"### Block: {range_text}", sections),
                   "rooms": sections})


def room_sections(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Typed sections of a stored record; a record without them is one legacy mixed section.

    Legacy blocks and eras never receive guessed labels or a migration: their
    complete content stays one section of explicitly unknown provenance that
    later eras can still reduce under genuine pressure.
    """
    rooms = block.get("rooms")
    if isinstance(rooms, list) and rooms and all(
        isinstance(room, dict) and isinstance(room.get("room_id"), str) and isinstance(room.get("content"), str)
        for room in rooms
    ):
        return [{"room_id": room["room_id"], "label": str(room.get("label") or room["room_id"]),
                 "message_count": int(room.get("message_count") or 0), "content": room["content"]}
                for room in rooms]
    return [{"room_id": LEGACY_ROOM_ID, "label": LEGACY_ROOM_LABEL,
             "message_count": int(block.get("message_count") or 0), "content": str(block.get("content") or "")}]


def era_dates(blocks: List[Dict[str, Any]]) -> Tuple[str, str]:
    start_date = str(blocks[0].get("range", "unknown"))[:10]
    last_range = str(blocks[-1].get("range", "unknown"))
    end_date = last_range.split(" to ")[-1].strip()[:10] if " to " in last_range else last_range[:10]
    return start_date, end_date


def compress_blocks_to_era(
    call: LightCall, blocks: List[Dict[str, Any]], identity_text: str = "",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Compress a contiguous run of records room by room; any failure keeps the originals.

    The same recorded room is grouped across the run (latest label wins), each
    group is compressed in one call and corrected against its complete
    sections in another; no call ever mixes rooms. Returns ``({"range",
    "message_count", "content", "rooms"}, usage)`` or ``(None, usage)``.
    """
    from ouroboros.consolidator import _merge_consolidation_usage

    start_date, end_date = era_dates(blocks)
    groups: Dict[str, Dict[str, Any]] = {}
    for block in blocks:
        for section in room_sections(block):
            group = groups.setdefault(section["room_id"], {"message_count": 0, "sections": []})
            group["label"] = section["label"]
            group["message_count"] += section["message_count"]
            group["sections"].append(f"### {block.get('range', 'unknown')}\n{section['content']}")
    rooms: List[Dict[str, Any]] = []
    usages: List[Dict[str, Any]] = []
    scope = f"era {start_date} to {end_date}"
    for room_id, group in groups.items():
        combined = "\n\n---\n\n".join(group["sections"])
        prompt_for = lambda text, group=group, room_id=room_id: era_room_prompt(  # noqa: E731
            text, room_label=group["label"], start_date=start_date, end_date=end_date,
            identity_text=identity_text, legacy=room_id == LEGACY_ROOM_ID)
        content, usage, _knowledge = call(prompt_for(combined), "Era compression", fixed_prompt=prompt_for(""),
                                          call_type="era_compression")
        usages.append(usage)
        if not content.strip():
            return None, _merge_consolidation_usage(*usages)
        corrected, usage, _knowledge = call(
            correction_prompt(content, combined, room_label=group["label"], scope=scope, identity_text=identity_text),
            "Era correction", fixed_prompt=correction_prompt(content, "", room_label=group["label"], scope=scope,
                                                             identity_text=identity_text),
            call_type="era_correction")
        usages.append(usage)
        if not corrected.strip():
            return None, _merge_consolidation_usage(*usages)
        rooms.append({"room_id": room_id, "label": group["label"], "message_count": group["message_count"],
                      "content": corrected.strip()})
    era_range = f"{start_date} to {end_date}"
    return ({"range": era_range, "message_count": sum(int(b.get("message_count") or 0) for b in blocks),
             "content": render_sections(f"### Era: {era_range}", rooms), "rooms": rooms},
            _merge_consolidation_usage(*usages))


__all__ = [
    "LEGACY_ROOM_ID", "LEGACY_ROOM_LABEL", "FIDELITY_RULES", "RoomSource",
    "partition_entries", "block_range", "room_draft_prompt", "correction_prompt", "era_room_prompt",
    "split_source_text", "summarize_source", "summarize_block", "render_sections", "room_sections",
    "era_dates", "compress_blocks_to_era",
]
