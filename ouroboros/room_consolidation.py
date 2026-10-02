"""Source-grounded room episodes and attributed revisions on the chronicle store.

The existing Light transport owns complete-source retrieval, route fitting and
custody. The mind's original publishes immediately; helper corrections are
separate records with their own source-read credit, never a gate on the original.
Raw room identity and legacy section reading remain deterministic host facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


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
    """One room's exact source rows, in their original order."""

    room_id: str
    label: str
    entries: List[Dict[str, Any]]


def partition_entries(entries: List[Dict[str, Any]], resolver: Any) -> List[RoomSource]:
    """Group a block's messages by actual room before any model call.

    Rooms keep their order of first appearance; messages keep the chat log's
    own chronological order inside each room. Every message lands in exactly
    one room, so the rooms' sources concatenate to the whole block's source.
    """
    rooms: Dict[str, RoomSource] = {}
    for entry in entries:
        room_id = resolver.room_id(entry)
        if room_id not in rooms:
            rooms[room_id] = RoomSource(room_id, resolver.label(entry), [])
        rooms[room_id].entries.append(entry)
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
    identity_text: str = "", continuation_note: str = "", knowledge_instruction: str = "", helper: bool = False,
) -> str:
    # Room labels are owner-authored project names: data, quoted as one JSON
    # string so a label cannot read as prompt structure or a header.
    room_label = json.dumps(str(room_label), ensure_ascii=False)
    voice = ("Write an attributed helper reconstruction of Ouroboros's experience, not a first-person account "
             "authored by the acting mind." if helper else "First person as Ouroboros: \"I did...\";")
    return f"""{knowledge_instruction}You are the memory consolidator of Ouroboros, a self-modifying AI agent.
Write the episodic memory of one room's messages inside the dialogue block {block_range_text}.
Room: {room_label}. This room contributed {message_count} messages; other rooms of the block are written separately and the host assembles them.
The source may be one contiguous part of the room; the episodic summary covers only the supplied part. Existing knowledge may inform a cumulative note update, but must not become an event or approval in this episode.

## Rules
1. No block or room headers; the host writes them. Start with the memory itself.
2. Preserve: decisions, agreements, technical discoveries, emotional and personal moments (what someone felt, asked for, enjoyed or disliked — quote them), task outcomes, what worked/failed.
3. Compress: routine tool calls, repetitive back-and-forth.
4. Quote key phrases directly when important; include task_ids when referencing specific tasks.
5. {voice} Call people by the names the messages give; when no name is known, describe the speaker honestly rather than inventing one.
6. Adapt length to content density; no fixed word range.
{FIDELITY_RULES}
{_identity_section(identity_text)}
{continuation_note}{DRAFT_SOURCE_HEADING}
{source}
"""


def correction_prompt(
    draft: str, source: str, *, room_label: str, scope: str,
    identity_text: str = "", continuation_note: str = "", knowledge_instruction: str = "",
    authored: bool = False,
) -> str:
    room_label = json.dumps(str(room_label), ensure_ascii=False)
    knowledge_check = ("You may nominate source-grounded cumulative knowledge updates even when the author's episode nominated none. "
                       "Read each complete current note YOURSELF before revising it. Return any nominations after the corrected episode "
                       "as KNOWLEDGE_ENTRIES_JSON: [...].\n" + knowledge_instruction if authored and knowledge_instruction else
                       "If the draft has a `KNOWLEDGE_ENTRIES_JSON:` block, check those cumulative updates "
                       "against each complete current note you read YOURSELF and this episode. The draft is a "
                       "proposal, not a source. After the episodic memory, return only draft-nominated topics "
                       "(including proposed new notes), or drop them. Unsupported episode claims must not survive in a note; "
                       "independently established knowledge may remain without becoming an event or approval "
                       "in this episode.\n" + knowledge_instruction if knowledge_instruction else "")
    voice = ("This is your helper revision of an already published episode, not the acting mind's original words. "
             "Keep its perspective clearly attributed; your correction does not create owner authority."
             if authored else "Keep the first-person Ouroboros voice, quotes, task_ids and everything the draft got right; adapt length to the content.")
    return f"""Compare this draft memory of Ouroboros against its complete source and return the corrected memory.
Scope: {scope}; room: {room_label}. The source below is complete for this episodic summary, not for cumulative knowledge; the host assembles rooms and headers separately.
Check sentence by sentence. Fix misattributed actors or approvals; decisions moved between rooms, tasks or people; invented, dropped or altered budgets, deadlines, checkpoints, boundaries and obligations; completion, review, verification or publication the source does not show; anything called approved that the source shows proposed, asked or rejected.
{voice}
{FIDELITY_RULES}
Return only the corrected memory text: no headers, commentary or diff. If the draft is already faithful, return it unchanged.
{knowledge_check}
{_identity_section(identity_text)}
{continuation_note}## Draft memory
{draft}

{CORRECTION_SOURCE_HEADING}
{source}
"""


def _extract_nominations(content: str, knowledge: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Bind a trailing nomination list through the read context that made the call."""
    if knowledge is None:
        return content, []
    from ouroboros.reflection import _extract_trailing_json

    content, entries = _extract_trailing_json(content, "KNOWLEDGE_ENTRIES_JSON:")
    return content, knowledge.bind_entries(entries)


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


def consolidation_coverage(facts: Any) -> Dict[str, Any]:
    """What one consolidation run covered, from its per-unit facts; every ratio names its denominator.

    ``accepted`` chunks passed draft and correction (publication is the
    separate ``blocks_written``); a ``withheld`` chunk stays behind the cursor
    for the next cycle, and chunks after it were not attempted this run.
    ``split_attempts`` counts refusals answered by queueing a part's halves,
    whatever those halves then did; whether the chunk came through is its
    status. An era ``produced`` is kept only when ``shorter``.
    """
    rows = [row for row in (facts or []) if isinstance(row, dict)]

    def total(unit: str, status: str = "") -> Dict[str, int]:
        chosen = [row for row in rows if row.get("unit") == unit and (not status or row.get("status") == status)]
        keys = ("messages", "rooms", "source_chars", "output_chars") if unit == "block" else (
            "blocks", "messages", "source_chars", "output_chars")
        return {"count": len(chosen), **{key: sum(int(row.get(key) or 0) for row in chosen) for key in keys}}

    accepted = total("block", "accepted")
    eras = [row for row in rows if row.get("unit") == "era"]
    return {
        "unit": "chat chunk attempted this run",
        "attempted": total("block"), "accepted": accepted, "withheld": total("block", "withheld"),
        "split_attempts": sum(int(row.get("split_attempts") or 0) for row in rows if row.get("unit") == "block"),
        "accepted_output_to_source": {
            "ratio": round(accepted["output_chars"] / accepted["source_chars"], 4) if accepted["source_chars"] else None,
            "numerator": "output chars of accepted chunks", "denominator": "source chars of accepted chunks"},
        "eras": {**total("era"), "produced": sum(1 for row in eras if row.get("status") == "produced"),
                 "shorter": sum(1 for row in eras if row.get("status") == "produced"
                                and int(row.get("output_chars") or 0) < int(row.get("source_chars") or 0))},
    }


__all__ = [
    "LEGACY_ROOM_ID", "LEGACY_ROOM_LABEL", "FIDELITY_RULES", "RoomSource",
    "partition_entries", "block_range", "room_draft_prompt", "correction_prompt",
    "room_sections", "consolidation_coverage",
]


def chronicle_pending(store: Any, source_path: Any) -> bool:
    """Whether the existing maintenance rail has unseen dialogue or authored episodes."""
    from ouroboros import consolidator as c

    scan = store.scan_state()
    if store.pending_episodes(limit=1):
        return True
    if scan.get("raw_scan_unavailable"):
        return False
    if not source_path.exists():
        return False
    segments, offset, gap = c._resolve_generation_segments(scan, source_path)
    return gap or sum(len(c._read_chat_entries(path)) for path in segments) > offset


def _chronicle_author(usage: dict) -> dict:
    from ouroboros.knowledge import observed_route_stamp
    return {"kind": "helper", "route": observed_route_stamp(usage)}


def _chronicle_nominations(store: Any, entries: list, source_ref: dict, context: Any, usage: dict,
                           *, revision: Optional[dict] = None) -> None:
    """Retain nomination bytes and the existing positional debt before note effects."""
    if not entries:
        if revision is not None:
            store.publish([revision])
        return
    import hashlib
    from ouroboros import consolidator as c
    from ouroboros.memory_nomination_receipts import prepare, settle
    from ouroboros.utils import append_jsonl, utc_now_iso

    source_id = hashlib.sha256(json.dumps([source_ref, entries], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    scan = store.scan_state()
    ids = prepare(scan, source_id, [(None, entries)])
    # The correction and its nomination bytes/debt enter one transaction. Even
    # a later history or note write failure leaves the full proposal reachable.
    store.publish([revision] if revision is not None else [], scan_state=scan)
    if not append_jsonl(store.data_root / "memory/knowledge_history.jsonl", {
            "ts": utc_now_iso(), "type": "dialogue_knowledge_nominations", "entry_id": source_id,
            "source_ref": source_ref, "nominations": entries}, ensure_record_boundary=True, require_lock=True):
        raise OSError("chronicle knowledge nominations could not be retained")
    outcomes = c._write_knowledge_entries(store.data_root / "memory/knowledge", entries, context=context,
        stamp={"writer": "chronicle_correction", "route": c._route_stamp(usage), "writer_input_ref": source_ref})
    settle(scan, ids, outcomes)
    store.publish([], scan_state=scan)


def _correct_chronicle_episode(store: Any, episode: dict, source: str, call: LightCall,
                               context: Any, identity_text: str) -> dict:
    """Publish a labeled helper revision; failure never retracts the original."""
    from ouroboros import consolidator as c

    if store.get("correction:" + episode["id"]) is not None:
        return c._merge_consolidation_usage()

    def prompt(text):
        return correction_prompt(episode["text"], text, room_label=episode["room_id"], scope="authored episode",
            identity_text=identity_text, knowledge_instruction=c.KNOWLEDGE_MAINTENANCE_PROMPT, authored=True)
    corrected, usage, knowledge = call(prompt(source), "Episode correction", fixed_prompt=prompt(""),
        call_type="memory_correction", memory_operation={"kind": "correction", "target_id": episode["id"]})
    if corrected.strip():
        corrected, entries = _extract_nominations(corrected, knowledge)
        if corrected.strip():
            revision = {"id": "correction:" + episode["id"], "kind": "revision", "room_id": episode["room_id"],
                "target_id": episode["id"], "text": corrected.strip(), "author": _chronicle_author(usage),
                "status": "published", "metadata": {"source_refs": episode["source_refs"],
                    "corrected_record": episode["id"], "auto_correction": True, "knowledge_entries": entries}}
            _chronicle_nominations(store, entries, {"kind": "chronicle", "record_id": revision["id"],
                "source_refs": episode["source_refs"]}, context, usage, revision=revision)
    return usage


def _check_authored_episodes(store: Any, call: LightCall, context: Any, identity_text: str) -> list:
    """Source-check newly authored episodes on the ordinary post-task rail."""
    from ouroboros.chronicle_sources import read_chronicle_source

    usages = []
    pending = store.pending_episodes()
    current = {record["id"]: record for room in dict.fromkeys(e["room_id"] for e in pending)
               for record in store.room_records(room)}
    mind_sources = [(record["id"], set(record.get("metadata", {}).get("source_row_ids", [])))
                    for record in store.records(kinds=["episode"]) if (record.get("author") or {}).get("kind") == "mind"]
    for episode in pending:
        visible = current.get(episode["id"], {})
        source_ids = set(episode.get("metadata", {}).get("source_row_ids", []))
        # Explicit new mind-authored coverage resolves this memory need, not the
        # independent physical outcome of an older helper operation.
        if ((visible.get("correction") and (visible.get("current_author") or {}).get("kind") == "mind")
                or source_ids and any(key != episode["id"] and source_ids <= ids for key, ids in mind_sources)):
            continue
        try:
            sources = [read_chronicle_source(store.data_root, ref, context.task_id, require_complete=True)
                       for ref in episode.get("source_refs", [])]
            if not sources:
                raise ValueError("episode has no complete retained source")
        except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
            record_id = "source-gap:" + episode["id"]
            if store.get(record_id) is None:
                store.publish([{"id": record_id, "kind": "maintenance", "room_id": episode["room_id"],
                    "target_id": episode["id"], "status": "source_unavailable", "reason": str(exc)}])
        else:
            usage = _correct_chronicle_episode(store, episode, "\n\n".join(sources), call, context, identity_text)
            usages.append(usage)
            if any(e.get("kind") == "budget_exhausted" for e in usage.get("_consolidation_errors", [])):
                return usages
    return usages


def _closed_source_rows(root: Any, entries: list, completed_task: Optional[dict]) -> set:
    """Select exact finished task/turn sources; unrelated or ambiguous rows stay raw."""
    from ouroboros.chronicle_store import source_row_id
    from ouroboros.project_dialogue import entry_matches_source_ref, historical_terminal_projection, latest_chat_annotations
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_results import load_task_result

    boundaries, tasks = {}, {}
    for index, row in enumerate(entries):
        if historical_terminal_projection(row):
            key = str(row["task_id"])
            boundaries[key], tasks[key] = index, row
    # Only the post-task producer may supply this, never an inline fit pass or
    # an arbitrary current ToolContext. Prefer its real host-facts boundary.
    if completed_task:
        key = str(completed_task.get("id") or completed_task.get("task_id") or "")
        candidates = [index for index, row in enumerate(entries) if key and row.get("task_id") == key]
        host = [index for index in candidates if entries[index].get("type") == "task_summary"
                and entries[index].get("summary_kind") == "host_task_facts"]
        if candidates:
            boundaries[key], tasks[key] = max(host or candidates), completed_task
    # A host operation can settle without a model task. Its exact source ref
    # closes only that command; an acknowledgement leaves the owner arc open.
    origins = [(index, row["origin_message_ref"]) for index, row in enumerate(entries)
               if row.get("direction") == "system" and not row.get("task_id")
               and row.get("task_terminal_status") in SETTLED_STATUSES
               and isinstance(row.get("origin_message_ref"), dict)]
    for key, task in tasks.items():
        if task.get("delegation_role") == "subagent":
            continue  # Inherited owner sources do not close with a child.
        if task is not completed_task:
            task = load_task_result(root, key) or task
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        ref = task.get("origin_message_ref") or metadata.get("origin_message_ref")
        if isinstance(ref, dict):
            origins.append((boundaries[key], ref))
    annotations = latest_chat_annotations(root)
    chosen = set()
    for index, row in enumerate(entries):
        key = str(row.get("subagent_task_id") or row.get("task_id") or row.get("root_task_id") or "")
        delivery = annotations.get(str(row.get("client_message_id") or ""), {})
        delivered_end = boundaries.get(str(delivery.get("target") or ""))
        # Task-less host observations are complete records, not open task arcs.
        # Their words may still describe unfinished work; summarization retains it.
        host_fact = not key and row.get("direction") in {"system", "out"} and bool(row.get("type"))
        delivered = (delivery.get("status") == "delivered" and delivered_end is not None
                     and index <= delivered_end)
        if host_fact or delivered or (key in boundaries and index <= boundaries[key]) or any(
                index <= end and entry_matches_source_ref(row, [ref]) for end, ref in origins):
            chosen.add(source_row_id(row))
    return chosen


def consolidate_chronicle(store: Any, source_path: Any, llm: Any, identity_text: str, context: Any,
                           *, room_registry_root: Any = None, pressure_fits: Any = None,
                           completed_task: Optional[dict] = None, represented_only: bool = False,
                           fitting_demand: Optional[dict] = None) -> dict:
    """Use the existing source/Light rails while the new store owns publication.

    Each room publishes immediately against retained exact source bytes. The
    shared physical scan advances only after every room is represented; stable
    source identities retain successful rooms when another call is interrupted.
    The old blocks and old cursor are never written after activation.
    """
    import hashlib
    from ouroboros import consolidator as c
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.chronicle_store import source_row_id, source_time_span
    from ouroboros.tools.registry import ToolContext

    context = context or ToolContext(repo_dir=store.data_root, drive_root=store.data_root, task_id="chronicle")
    transport = c._light_call(llm, context, {})

    def call(prompt, label, **options):
        # The source and actual attempt IDs stay on the existing memory/custody
        # rails. Unknown paid work cannot become a retry on a later task/route.
        from ouroboros.transport_custody import outcome_unknown_on_chain, attempt_custody_event_fields
        operation = options.pop("memory_operation", {})
        scan = store.scan_state()
        pending = scan.get("pending_consolidation_outcomes", [])
        for fact in pending:
            if operation and fact.get("operation") == operation:
                # Historical custody remains in the store and rendered view.
                # No call occurred now, so it cannot interrupt this run's later
                # independent work as a newly unresolved physical attempt.
                return "", c._merge_consolidation_usage(), None
        source_ref = c.retain_memory_source(context, label, prompt.encode("utf-8"))
        from ouroboros.memory_guidance import remembering_guidance
        guidance_sha = hashlib.sha256(remembering_guidance(store.data_root).encode("utf-8")).hexdigest()
        refusal_key = "input-refusal:" + hashlib.sha256(json.dumps(
            [source_ref["sha256"], c._light_route(), guidance_sha], sort_keys=True).encode()).hexdigest()
        refused = store.get(scan.get("consolidation_input_limits", {}).get(refusal_key, refusal_key))
        if refused:
            options["input_limit"] = refused["input_limit"]
        options["source_ref"] = source_ref

        def retain_failure(fact):
            scan = store.scan_state()
            fact = {**fact, "source_ref": source_ref, "operation": operation}
            scan["last_consolidation_error"] = fact
            # A legacy/unbound error is evidence, not a veto. Only a named
            # physical attempt binds no-repeat to this exact memory operation.
            if fact.get("kind") == "provider_outcome_unknown" and operation and (
                    fact.get("physical_attempt_id") or fact.get("ledger_attempt_ids")):
                pending = scan.setdefault("pending_consolidation_outcomes", [])
                if not any(row.get("operation") == operation for row in pending):
                    pending.append(fact)
            store.publish([], scan_state=scan)
        try:
            content, usage, knowledge = transport(prompt, label, **options)
        except Exception as exc:
            if outcome_unknown_on_chain(exc):
                retain_failure({"kind": "provider_outcome_unknown", "label": label,
                    "ledger_attempt_ids": list(getattr(exc, "ledger_attempt_ids", []) or []),
                    **attempt_custody_event_fields(exc)})
            raise
        errors = usage.get("_consolidation_errors") or []
        if errors:
            retain_failure({**errors[-1], "ledger_attempt_ids": list(usage.get("ledger_attempt_ids") or [])})
            failure = errors[-1]
            if (failure.get("kind") == "context_overflow" and not failure.get("preflight_only")
                    and failure.get("input_bytes")):
                bound = {key: failure.get(key) for key in ("route_fp", "capacity_tokens", "output_reserve_tokens")}
                bound["input_bytes"] = int(failure["input_bytes"]) - 1
                refusal_key = "input-refusal:" + hashlib.sha256(json.dumps([source_ref["sha256"],
                    usage.get("_light_dispatch_binding") or c._light_route(),
                    usage.get("_remembering_guidance_sha256") or guidance_sha], sort_keys=True).encode()).hexdigest()
                receipt_id = refusal_key + ":" + hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
                scan = store.scan_state()
                scan.setdefault("consolidation_input_limits", {})[refusal_key] = receipt_id
                record = {"id": receipt_id, "kind": "input_refusal", "room_id": operation.get("room_id", ""),
                          "source_ref": source_ref, "input_limit": bound}
                store.publish([] if store.get(receipt_id) else [record], scan_state=scan)
        return content, usage, knowledge
    if represented_only:
        # First-turn preparation may only change already represented memory.
        # It shares the lock/transport/custody above, never the raw-history scan.
        usages = (compact_chronicle_rooms(store, call, context, identity_text, pressure_fits, fitting_demand=fitting_demand)
                  if pressure_fits is not None else [])
        usage = c._merge_consolidation_usage(*usages)
        usage["_blocks_written"] = sum(bool(row.get("record_id")) for row in usage["_coverage"])
        return usage
    usages = _check_authored_episodes(store, call, context, identity_text)
    if any(e.get("kind") == "budget_exhausted" for u in usages for e in u.get("_consolidation_errors", [])):
        return c._merge_consolidation_usage(*usages)
    scan = store.scan_state()
    written = 0
    if source_path.exists() and not scan.get("raw_scan_unavailable"):
        segments, offset, gap = c._resolve_generation_segments(scan, source_path)
        if gap:
            old = json.dumps(scan.get("chat_log_signature", {}), sort_keys=True)
            store.publish([{"id": "gap:" + hashlib.sha256(old.encode()).hexdigest(), "kind": "gap",
                "room_id": "legacy", "text": "[MEMORY GAP] The previous raw chat generation is unavailable.",
                "source_refs": [], "metadata": {"lost_cursor": scan}}])
            scan["last_consolidated_offset"] = 0
            scan["chat_log_signature"] = c._chat_log_signature(source_path)
            store.publish([], scan_state=scan)
        captured = c._capture_generation_window(store.data_root / "memory/dialogue_meta.json", source_path,
            segments, offset, load_cursor=store.scan_state)
        if captured is not None:
            segments, signatures, segment_entries, all_entries, offset = captured
            entries = all_entries[offset:]
            resolver = RoomLabelResolver(room_registry_root or store.data_root)
            authored_ids = {identifier for record in store.records(kinds=["episode"])
                            for identifier in record.get("metadata", {}).get("source_row_ids", [])}
            withheld_ids = {identifier for fact in store.scan_state().get("pending_consolidation_outcomes", [])
                            for identifier in fact.get("operation", {}).get("source_row_ids", [])} - authored_ids
            closed_ids = _closed_source_rows(store.data_root, all_entries, completed_task)
            rooms = partition_entries([row for row in entries if source_row_id(row) in closed_ids
                                       and source_row_id(row) not in authored_ids | withheld_ids], resolver)
            for room in rooms:
                exact = json.dumps(room.entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
                identity = hashlib.sha256(json.dumps([room.room_id, signatures, offset, len(all_entries)],
                    sort_keys=True).encode() + exact).hexdigest()
                record_id = "source:" + identity
                if store.get(record_id) is not None:
                    continue
                source_ref = c.retain_memory_source(context, record_id, exact, "json")
                def draft(text):
                    return (room_draft_prompt(text, room_label=room.label,
                        block_range_text=block_range(str(room.entries[0].get("ts", "")), str(room.entries[-1].get("ts", ""))),
                        message_count=len(room.entries), identity_text=identity_text, helper=True))
                operation = {"kind": "raw_episode", "room_id": room.room_id,
                             "source_row_ids": [source_row_id(row) for row in room.entries]}
                text, usage, _knowledge = call(draft(exact.decode("utf-8")), "Room episode", fixed_prompt=draft(""),
                                              memory_operation=operation)
                coverage = {"unit": "block", "representation": "episode", "status": "withheld", "rooms": 1,
                            "messages": len(room.entries), "source_chars": len(exact.decode("utf-8")),
                            "output_chars": 0, "split_attempts": 0}
                usage.setdefault("_coverage", []).append(coverage)
                usages.append(usage)
                if not text.strip():
                    if any(e.get("kind") == "budget_exhausted" for e in usage.get("_consolidation_errors", [])):
                        break
                    continue
                episode = store.append_episode(room.room_id, text.strip(), [source_ref], _chronicle_author(usage),
                    record_id=record_id, metadata={"label": room.label, "message_count": len(room.entries),
                        "source_span": source_time_span(row.get("ts") for row in room.entries),
                        "source_ref": source_ref, "source_row_ids": [source_row_id(row) for row in room.entries],
                        "task_ids": sorted({str(row[key]) for row in room.entries
                                            for key in ("task_id", "parent_task_id", "root_task_id") if row.get(key)}),
                        "source_range": {"generations": signatures, "start_offset": offset, "end_offset": len(all_entries),
                                         "coverage": "selected_rows"}})
                authored_ids.update(source_row_id(row) for row in room.entries)
                written += 1
                coverage.update(status="accepted", output_chars=len(text.strip()), record_id=episode["id"])
                usage = _correct_chronicle_episode(store, episode, exact.decode("utf-8"), call, context, identity_text)
                usages.append(usage)
                # Yield this ordinary source-publication unit only after its
                # correction attempt; advance the frontier and keep the separate
                # pressure stage eligible so new raw work cannot starve it.
                if ((fitting_demand or {}).get("ordinary_maintenance") or
                        any(e.get("kind") == "budget_exhausted" for e in usage.get("_consolidation_errors", []))):
                    break
            position = offset
            for row in entries:
                if source_row_id(row) not in authored_ids:
                    break
                position += 1
            if position > offset:
                scan = store.scan_state()
                c._advance_cursor(scan, segments, signatures, segment_entries, position)
                store.publish([], scan_state=scan)
    if pressure_fits is not None and not pressure_fits() and not any(u.get("_consolidation_errors") for u in usages):
        usages.extend(compact_chronicle_rooms(store, call, context, identity_text, pressure_fits, fitting_demand=fitting_demand))
    usage = c._merge_consolidation_usage(*usages)
    if not usage.get("_consolidation_errors"):
        scan = store.scan_state()
        if (not scan.get("pending_consolidation_outcomes")
                and (scan.get("last_consolidation_error") or {}).get("kind") != "provider_outcome_unknown"
                and scan.pop("last_consolidation_error", None) is not None):
            store.publish([], scan_state=scan)
    usage["_blocks_written"] = written
    return usage


def _digest_requirement(demand: Optional[dict]) -> dict:
    """Stable fitting boundary; purpose/mode labels remain observations, not paid needs."""
    return {key: demand[key] for key in ("requirement_tokens",)
            if demand and demand.get(key) is not None}


def _digest_requirement_attempted(prior: dict, requirement: dict, guidance_sha: str) -> bool:
    """Deduplicate a tried requirement; an attempt is not proof of successful fit."""
    if prior.get("guidance_sha256", guidance_sha) != guidance_sha:
        return False
    previous = prior.get("requirement", _digest_requirement(prior.get("fitting_demand")))
    for witness in ("refused_digest_ids", "unmet_digest_ids"):
        if requirement.get(witness) and previous.get(witness) != requirement[witness]:
            return False
    before, now = previous.get("requirement_tokens"), requirement.get("requirement_tokens")
    return now is None or (before is not None and now >= before)


def _digest_fulfilled_id(record_id, requirement, guidance_sha):
    import hashlib
    return "digest-fulfilled:" + hashlib.sha256(json.dumps(
        [record_id, _digest_requirement(requirement), guidance_sha], sort_keys=True).encode()).hexdigest()


def record_maintenance_fit(store, demand):
    """Accept only observed target satisfaction, naming the actual captured digest revisions."""
    from ouroboros.memory_guidance import remembering_guidance
    import hashlib
    if (not demand or not demand.get("ordinary_maintenance") or demand.get("target_miss") is not False
            or demand.get("maintenance_target_reachable") is False or demand.get("memory_budget_tokens") == 0):
        return
    guidance_sha = hashlib.sha256(remembering_guidance(store.data_root).encode("utf-8")).hexdigest()
    requirement, records = _digest_requirement(demand), []
    for record_id in dict.fromkeys(demand.get("selected_digest_ids") or []):
        key = _digest_fulfilled_id(record_id, requirement, guidance_sha)
        source = store.get(record_id)
        if source and store.get(key) is None:
            records.append({"id": key, "kind": "maintenance", "room_id": source["room_id"],
                "target_id": record_id, "target_fits": True, "requirement": requirement,
                "guidance_sha256": guidance_sha})
    if records:
        store.publish(records)  # One immutable transaction; no scan-state read/modify/write race.


def _unfulfilled_digest(store, parent, requirement, guidance_sha):
    record_id = (parent.get("correction") or parent)["id"]
    if (not requirement or record_id != parent["id"]
            or store.get(_digest_fulfilled_id(record_id, requirement, guidance_sha)) is not None):
        return False
    suffix = parent["id"].removeprefix("digest:")
    attempt = store.get("digest-attempt:" + suffix) or {}
    outcome = store.get("digest-fit:" + suffix) or {}
    return (outcome.get("published_progress") is True and outcome.get("target_fits") is False
            and attempt.get("guidance_sha256") == guidance_sha
            and _digest_requirement(attempt.get("requirement")) == requirement)


def _record_digest_fit(store, fingerprint, room_id, progress, observed, demand):
    """Keep publication, measured fit and still-unknown provider acceptance separate."""
    # Recovery's callback proves useful shrink, not a successful Main send.
    facts = demand or {}
    target_fits = not facts["target_miss"] if "target_miss" in facts else observed
    if facts.get("memory_budget_tokens") == 0 or facts.get("maintenance_target_reachable") is False:
        target_fits = False  # The callback may stop futile work without satisfying its target.
    outcome = {"published_progress": progress,
        "target_fits": target_fits if facts.get("purpose") != "actual_context_refusal" else None,
        "remaining_demand": dict(facts)}
    store.publish([{"id": "digest-fit:" + fingerprint, "kind": "maintenance", "room_id": room_id,
                   "target_id": "digest-attempt:" + fingerprint, **outcome}])
    if demand is not None:
        demand.update(published_progress=progress, target_fits=outcome["target_fits"])
    record_maintenance_fit(store, demand)


def _digest_source_groups(store, records, requirement, guidance_sha, refused, ordinary=False):
    """Whole stable peers and fixed children of an eligible existing interpretation."""
    from itertools import groupby
    from ouroboros.chronicle_view import _cuts

    # Recover heights/order for older digests too. Parents were published
    # after children; current revision ids alias the same immutable node.
    topology = {}
    for record in records:
        children = (record.get("metadata") or {}).get("covers_record_ids", [])
        child_nodes = [topology[cid] for cid in children if cid in topology]
        node = (1 + max((n[0] for n in child_nodes), default=0),
                min((n[1] for n in child_nodes), default=record["sequence"])) if record["kind"] == "digest" else (0, record["sequence"])
        topology[record["id"]] = topology[(record.get("correction") or record)["id"]] = node
    frontier = sorted(_cuts(records)[-1], key=lambda r: topology[r["id"]][1])
    unmet = {r["id"] for r in frontier if ordinary and r["kind"] == "digest"
             and _unfulfilled_digest(store, r, requirement, guidance_sha)}
    groups = [list(group) for _height, group in groupby(frontier, key=lambda r: topology[r["id"]][0])]
    groups = [group for group in groups if len(group) > 1 or group[0]["kind"] != "digest"]
    groups.sort(key=lambda group: topology[group[0]["id"]][0])
    # A changed requirement or guidance can need an alternative.
    # Read fixed children, never the parent's own previous wording.
    current = {(r.get("correction") or r)["id"]: r for r in records}
    for parent in frontier:
        children = (parent.get("metadata") or {}).get("covers_record_ids", [])
        if parent["kind"] == "digest" and children and all(cid in current for cid in children):
            published = store.get(parent["id"].replace("digest:", "digest-attempt:", 1)) or {}
            if ((parent.get("correction") or parent)["id"] in refused | unmet
                    or not _digest_requirement_attempted(published, requirement, guidance_sha)):
                groups.append([current[cid] for cid in children])
    return groups, unmet


def compact_chronicle_rooms(store: Any, call: LightCall, context: Any,
                            identity_text: str, fits: Callable[[], bool], *, fitting_demand: Optional[dict] = None) -> list:
    """Close new source groups, then join stable peers without rolling old history.

    A digest is never its own only input. Equal-height children can form a
    higher parent; a new tail cannot repeatedly wrap the whole older room.
    Height follows immutable child edges, with no calendar or depth limit.
    ``fits`` may refresh observations in the caller's ``fitting_demand`` dict.
    Only a changed source, guidance or genuine requirement buys a new attempt.
    Leftover memory allowance, account and request identity are observations.
    """
    import hashlib
    from ouroboros import consolidator as c
    from ouroboros.chronicle_store import source_time_span
    from ouroboros.memory_guidance import remembering_guidance
    from ouroboros.utils import estimate_tokens

    pending = [fact for fact in store.scan_state().get("pending_consolidation_outcomes", [])
               if fact.get("operation", {}).get("kind") == "digest"]
    held = {tuple(revision) for fact in pending for revision in fact["operation"].get("source_revisions", [])}
    usages = []  # Held custody is resident memory, not a new interruption.
    attempted = set()
    demand = dict(fitting_demand or {})
    base_requirement = _digest_requirement(demand)
    fits()  # Refresh the captured representation before recording any positive observation.
    record_maintenance_fit(store, fitting_demand)
    refused = set(demand.get("refused_digest_ids") or []) if demand.get("purpose") == "actual_context_refusal" else set()
    room_sizes = [(sum(len(r["current_text"]) for r in store.room_cover(room)), room)
                  for room in store.room_ids()]
    for _size, room_id in sorted(room_sizes, reverse=True):
        while not fits():
            guidance_sha = hashlib.sha256(remembering_guidance(store.data_root).encode("utf-8")).hexdigest()
            records = store.room_records(room_id)
            groups, unmet = _digest_source_groups(store, records, base_requirement, guidance_sha, refused,
                                                   ordinary=bool(demand.get("ordinary_maintenance")))
            sources = []
            attempts = store.records(room_id, kinds=["maintenance"])
            while groups:
                group = groups.pop(0)
                if len(group) == 1 and group[0]["kind"] == "digest":
                    continue
                source_keys = [[r["id"], (r.get("correction") or {}).get("id", r["id"])] for r in group]
                alternatives = [r for r in records if r["kind"] == "digest"
                    and (r.get("metadata") or {}).get("source_revisions") == source_keys]
                rejected = sorted((r.get("correction") or r)["id"] for r in alternatives
                                  if (r.get("correction") or r)["id"] in refused)
                unfinished = sorted(r["id"] for r in alternatives if r["id"] in unmet)
                requirement = {**base_requirement, **({"refused_digest_ids": rejected} if rejected else {}),
                               **({"unmet_digest_ids": unfinished} if unfinished else {})}
                if any(tuple(key) in held for key in source_keys):
                    # Keep this unresolved operation and its sources visible;
                    # it cannot freeze unrelated rooms or buy overlapping work.
                    continue
                fingerprint = hashlib.sha256(json.dumps([source_keys, guidance_sha, requirement],
                    sort_keys=True).encode()).hexdigest()
                matching = [prior for prior in attempts if prior.get("source_keys") == source_keys
                            and _digest_requirement_attempted(prior, requirement, guidance_sha)]
                if len(group) > 1 and any(prior.get("status") == "output_truncated" for prior in matching):
                    # A known terminal output failure justifies smaller existing
                    # source units; unknown physical custody never enters here.
                    middle = len(group) // 2
                    groups[:0] = [group[:middle], group[middle:]]
                    continue
                if not matching and fingerprint not in attempted and store.get("digest-attempt:" + fingerprint) is None:
                    attempted.add(fingerprint)
                    sources = group
                    break
            if not sources:
                break
            source_rows = [{"record_id": r["id"],
                "revision_id": (r.get("correction") or r)["id"], "author": r["current_author"],
                "text": r["current_text"], "source_refs": r.get("source_refs", []),
                "metadata": r.get("metadata", {}),
                **({"revision_source_span": r["correction"]["metadata"]["source_span"]}
                   if "source_span" in (r.get("correction") or {}).get("metadata", {}) else {})} for r in sources]
            source_text = "\n\n".join(json.dumps(row, ensure_ascii=False) for row in source_rows)
            source_ref = c.retain_memory_source(context, "digest:" + fingerprint, source_text.encode("utf-8"))
            # Keep every cognitive field; relocate only repeated host indexes and
            # locators. Unknown metadata stays inline, not an importance filter.
            projected = json.loads(json.dumps(source_rows, ensure_ascii=False))
            index_fields = ("children", "source_row_ids", "covers_record_ids", "source_revisions")
            locator_fields = ("canonical_root", "root", "path", "read", "sha256", "size")
            for row in projected:
                metadata = row["metadata"]
                for key in index_fields:
                    if isinstance(metadata.get(key), list):
                        metadata[key] = {"retained_items": len(metadata[key])}
                for ref in [*row["source_refs"], metadata.get("legacy_source_ref")]:
                    if isinstance(ref, dict):
                        for key in locator_fields:
                            ref.pop(key, None)
            previous_chars = min([len(r["current_text"]) for r in alternatives]
                                 + [sum(len(r["current_text"]) for r in sources)])
            observations = {**dict(fitting_demand or {}), "previous_representation_chars": previous_chars,
                "token_estimate_basis": "chars_div_4",
                "source_text_chars": sum(len(row["text"]) for row in source_rows),
                "source_text_estimated_tokens": estimate_tokens("".join(row["text"] for row in source_rows)),
                "source_contribution_basis": "Complete source bodies only; excludes record and locator overhead."}
            rendered = (fitting_demand or {}).get("rendered_memory_tokens")
            if rendered is not None:
                observations["rendered_memory_tokens"] = rendered
                if demand.get("memory_budget_tokens") is not None:
                    observations["memory_deficit_tokens"] = max(0, rendered - demand["memory_budget_tokens"])
            def prompt(text):
                return ("Write a compact shared-life account of these closed parts of a room's history. "
                    "These are immutable sibling sources, not a previous version of the digest you are writing. "
                    "This coarser account stays resident beside other rooms and the detailed current conversation. "
                    "Let a later self recognize the room's story, changes of intent, causal lessons and open lines. "
                    "Choose detail toward the supplied whole-memory budget, which also holds open conversations and marks; "
                    "it is not a separate allowance for this room. Dates do not determine importance. "
                    "Preserve supplied author distinctions and source gaps; do not impersonate the acting mind. "
                    "This changes representation only, all exact source revisions remain.\n"
                    + FIDELITY_RULES + _identity_section(identity_text)
                    + "\nApply source fidelity at this coarser resolution: keep exact constraints and decisions that still "
                    "govern action, uncertainty, personal meaning and material causal changes. Collapse the detailed "
                    "protocol of superseded steps into their outcome and lesson; exact intermediate identifiers and "
                    "test counters remain in the addressed children when unnecessary for this understanding. "
                    "Do not turn unsettled work into completion or replace its meaning with an address. "
                    "The model chooses the level of detail; no per-room word quota or fixed compression ratio is imposed. "
                    "When a published account was actually refused, choose a materially coarser useful account from "
                    "these original sources. An alternative must improve on the previous representation size, "
                    "not merely on the larger original sources; do not paraphrase the prior account."
                    + "\nMeasured whole-memory fitting demand (shared with other rooms; estimates, not a word quota): " + json.dumps(observations)
                    + "\nAll source text and authors follow. Host index arrays in metadata " + json.dumps(index_fields)
                    + " are shown as retained item counts. Locator fields " + json.dumps(locator_fields)
                    + " in source_refs and metadata.legacy_source_ref are not included in this view. "
                    "Their omission is not a read receipt. The complete group, including every metadata field, "
                    "remains readable through read_file: " + json.dumps(source_ref["read"], ensure_ascii=False)
                    + "\n## Source group: complete text, projected host metadata\n" + text)
            text, usage, _knowledge = call(prompt("\n\n".join(json.dumps(row, ensure_ascii=False) for row in projected)),
                "Room digest", fixed_prompt=prompt(""), call_type="era_compression",
                memory_operation={"kind": "digest", "room_id": room_id, "source_revisions": source_keys})
            usages.append(usage)
            applied_guidance = usage.get("_remembering_guidance_sha256") or guidance_sha
            fingerprint = hashlib.sha256(json.dumps([source_keys, applied_guidance, requirement],
                sort_keys=True).encode()).hexdigest()
            if errors := usage.get("_consolidation_errors"):
                if all(error.get("kind") == "output_truncated" for error in errors):
                    store.publish([{"id": "digest-attempt:" + fingerprint, "kind": "maintenance", "room_id": room_id,
                        "status": "output_truncated", "source_keys": source_keys, "source_ref": source_ref,
                        "requirement": requirement, "guidance_sha256": applied_guidance, "fitting_demand": demand}])
                    if demand.get("purpose") == "actual_context_refusal" and len(sources) > 1:
                        # Finish available source-part preparation before book fallback;
                        # only the caller's complete-frame measurement proves progress.
                        continue
                return usages
            shorter = bool(text.strip()) and len(text.strip()) < previous_chars
            coverage = {"unit": "era", "representation": "digest",
                "status": "accepted" if shorter else "not_shorter", "blocks": len(sources),
                "messages": sum(r.get("metadata", {}).get("message_count", 0) for r in sources),
                "source_chars": sum(len(r["current_text"]) for r in sources), "output_chars": len(text.strip())}
            usage.setdefault("_coverage", []).append(coverage)
            records = [{"id": "digest-attempt:" + fingerprint, "kind": "maintenance", "room_id": room_id,
                        "status": "compressed" if shorter else "not_shorter", "source_keys": source_keys,
                        "fitting_demand": demand, "requirement": requirement, "guidance_sha256": applied_guidance}]
            if shorter:
                timestamps, incomplete = [], False
                for source in sources:
                    metadata = source.get("metadata") or {}
                    revised = (source.get("correction") or {}).get("metadata") or {}
                    span = revised.get("source_span", metadata.get("source_span")) or {}
                    timestamps.extend([span.get("start"), span.get("end")])
                    incomplete |= span.get("incomplete", True)
                gap_sources = [r["id"] for r in sources if r["kind"] == "gap" or r.get("metadata", {}).get("source_gap")]
                records.append({"id": "digest:" + fingerprint, "kind": "digest", "room_id": room_id,
                    "text": text.strip(), "author": _chronicle_author(usage), "source_refs": [source_ref],
                    "metadata": {"covers_record_ids": [(r.get("correction") or r)["id"] for r in sources],
                        "source_span": source_time_span(timestamps, incomplete=incomplete),
                        "source_row_ids": list(dict.fromkeys(identifier for r in sources
                            for identifier in r.get("metadata", {}).get("source_row_ids", []))),
                        "task_ids": sorted({task_id for r in sources for task_id in r.get("metadata", {}).get("task_ids", [])}),
                        **({"source_gap": "Source gaps remain in covered records: " + ", ".join(gap_sources)
                            + ". Read their retained causes and sources before claiming complete history."} if gap_sources else {}),
                        "source_revisions": source_keys}})
                coverage["record_id"] = records[-1]["id"]
            store.publish(records)
            _record_digest_fit(store, fingerprint, room_id, shorter, bool(fits()), fitting_demand)
            if shorter and demand.get("ordinary_maintenance"):
                # A complete source-cover publication is the incremental unit,
                # even when its transport needed several physical source reads.
                # Return the still-measured need to the ordinary rail, rather
                # than making this task finish a whole-corpus conversion.
                return usages
    return usages
