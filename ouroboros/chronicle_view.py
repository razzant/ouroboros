"""Capture one biography and render its existing interpretations at a fitting resolution.

Sources and interpretations are captured once. The renderer performs no I/O or
paid work and never invents a shorter recollection: only complete published
digests may replace their exact children. Room membership is a projection over
source identity, so promoting Main work does not move or rewrite its history.
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

from ouroboros.chronicle_store import ChronicleStore, source_row_id
from ouroboros.utils import estimate_tokens
from ouroboros.context_budget import MEMORY_BEGIN, MEMORY_END, MEMORY_FACTS_PREFIX, canonical_context_json as _encoded


CHRONICLE_MARKER = "\n[CHRONICLE_VIEW]\n"
_TARGET_MISS_NOTICE = (
    "\n\n[Memory does not fit the requested remaining space. Complete available meanings were retained; "
    "the selected published views exceed this target. This is a target miss, not a demonstrated route refusal "
    "or permission to discard a period. Exact sources and revision history remain readable.]")


def _current_id(record: dict) -> str:
    return str((record.get("correction") or {}).get("id") or record["id"])


def _record_text(record: dict) -> str:
    author = record.get("current_author", record.get("author", {}))
    label = _encoded(author)
    ref = f"memory_read(node_id={record['id']!r})"
    corrected = " Helper/author revision; original and revision history remain readable." if record.get("correction") else ""
    metadata = record.get("metadata") or {}
    gap = metadata.get("source_gap")
    legacy = (f" Legacy {metadata['legacy_type']}; range: {metadata.get('range') or 'unknown'}; "
              "interpretation, not a grant or standing rule."
              if metadata.get("legacy_type") else "")
    if metadata.get("legacy_message_count") is not None:
        legacy += f" Legacy source messages: {metadata['legacy_message_count']}."
    revision = record.get("correction") or {}
    span = (revision.get("metadata") or {}).get("source_span", metadata.get("source_span"))
    period = ""
    if span is not None:
        period = f" Known source time bounds: {span.get('start') or 'unknown'} .. {span.get('end') or 'unknown'}."
        if span.get("incomplete"):
            period += " Some source times are unknown."
        period += " Bounds do not assert continuous coverage."
    elif not metadata.get("legacy_type") and record.get("ts"):
        period = f" Recorded: {revision.get('ts') or record['ts']}; source period unknown."
    return (f"[Memory; author={label}; source: {ref}.{legacy}{period}{corrected}]\n"
            + str(record.get("current_text", record.get("text", "")))
            + (f"\n[Source gap: {gap}]" if gap else ""))


def _cuts(records: list[dict]) -> list[list[dict]]:
    """Whole-source covers, finest first; a stale digest cannot hide a correction."""
    by_id = {_current_id(r): r for r in records}
    active = {_current_id(r): r for r in records if r.get("kind") != "digest" or r.get("_cover_root")}
    cuts = [list(active.values())]
    remaining = [r for r in records if r.get("kind") == "digest"]
    while remaining:
        choices = []
        for digest in remaining:
            children = (digest.get("metadata") or {}).get("covers_record_ids", [])
            if not children or not set(children).issubset(active):
                continue
            saved = sum(len(_record_text(active[c])) for c in children) - len(_record_text(digest))
            if saved > 0:
                choices.append((saved, digest, children))
        if not choices:
            break
        _, digest, children = max(choices, key=lambda choice: (choice[0], choice[1].get("sequence", 0)))
        for child in children:
            active.pop(child)
        active[_current_id(digest)] = by_id[_current_id(digest)]
        remaining.remove(digest)
        cuts.append(sorted(active.values(), key=lambda record: record.get("sequence", 0)))
    return cuts



def _snapshot_record(record: dict) -> dict:
    """Omit exact duplicate projection fields, retaining their existing fallbacks.

    This only packs a captured view; source records and revision history stay
    untouched in the append-only store.
    """
    packed = dict(record)
    for current, original in (("current_text", "text"), ("current_author", "author")):
        if current in packed and packed[current] == packed.get(original):
            packed.pop(current)
    if packed.get("revisions") == []:
        packed.pop("revisions")
    return packed


def _derived_rooms(store: ChronicleStore, focus: str, resolver: Any, bindings: dict):
    from ouroboros.project_dialogue import bound_room_chat

    room_ids = store.room_ids()
    rooms: dict[str, dict] = {}
    covered = set()
    for room_id in room_ids:
        # Capture published alternatives, not only today's smallest cover. The
        # immutable snapshot can then rebind upward without rereading sources.
        for record in store.room_records(room_id):
            meta = record.get("metadata") or {}
            task_ids = meta.get("task_ids", [(record.get("author") or {}).get("task_id")]) or []
            projected = {bound_room_chat(bindings, {"task_id": tid}) for tid in task_ids if tid}
            projected.discard(0)
            destination = str(next(iter(projected))) if len(projected) == 1 else room_id
            room = rooms.setdefault(destination, {"id": destination, "records": [], "label":
                resolver.label({"chat_id": destination}) if destination.lstrip("-").isdigit() else destination})
            room["records"].append(_snapshot_record(record))
            if destination == focus:
                covered.update(meta.get("source_row_ids") or [])
    # A Project can adopt rows that were written in Main. The provenance index
    # finds its exact interpretations without rekeying source history.
    present = {record["id"] for room in rooms.values() for record in room["records"]}
    for record in store.records_for_tasks([tid for tid, chat in bindings.items() if str(chat) == focus]):
        if record["id"] not in present:
            room = rooms.setdefault(focus, {"id": focus, "label": resolver.label({"chat_id": focus}), "records": []})
            room["records"].append({**_snapshot_record(record), "focus_expansion": True})
            present.add(record["id"])
            covered.update((record.get("metadata") or {}).get("source_row_ids") or [])
    return list(rooms.values()), covered


def capture_chronicle(memory: Any, task: dict, *, rendered_chars_budget: int | None = None) -> str:
    """Cold deterministic activation and a frozen input for every route projection.

    Called by ordinary turn preparation, never a UI polling reader. Missing or
    locked legacy sources keep their existing reader until activation succeeds.
    """
    store = ChronicleStore(memory.drive_root)
    activation = store.import_legacy()
    if not activation or activation.get("kind") != "activation":
        return ""
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.project_dialogue import bound_room_chat, room_membership, source_refs_for_project
    from ouroboros.projects_registry import all_task_bindings
    from ouroboros.consolidator import retain_memory_source

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    address = task.get("chat_id", metadata.get("chat_id", 1))
    focus_chat = int(1 if address is None else address)
    focus = str(focus_chat)
    resolver = RoomLabelResolver(memory.drive_root)
    bindings = all_task_bindings(memory.drive_root)
    matches = room_membership(focus_chat, resolver.project_chat_ids,
                              source_refs_for_project(memory.drive_root, focus_chat), bindings)
    from ouroboros.project_dialogue import _row_chat_id

    predicate = lambda row: matches(_row_chat_id(row), row)
    from ouroboros.chronicle_sources import capture_room, capture_pending_rows, retain_room_source

    rooms, covered = _derived_rooms(store, focus, resolver, bindings)
    raw, coverage = capture_room(memory, focus, rendered_chars_budget=rendered_chars_budget)
    from ouroboros.room_consolidation import _closed_source_rows
    from ouroboros.chronicle_sources import capture_covered_focus

    # Summary coverage is not completion. An oversized focus reads only its
    # covered sources and typed closure candidates from existing indexed rows.
    focused_sources = raw
    if covered and not coverage.get("full_room_in_view", True):
        focused_sources, source_gaps = capture_covered_focus(memory, store, coverage.get("row_locators", []), covered)
        coverage.setdefault("gaps", []).extend(source_gaps)
        if source_gaps:
            coverage.update(complete=False, snapshot_stable=False)
    closed = _closed_source_rows(memory.drive_root, focused_sources, None)
    historical_gaps, gap_ids = memory._durable_dialogue_gaps()
    coverage.setdefault("gaps", []).extend(historical_gaps)
    coverage["durable_gap_ids"] = gap_ids
    scan = store.scan_state()
    represented = {key for room in rooms for record in room["records"]
                   for key in (record.get("metadata") or {}).get("source_row_ids", [])}
    pending_all, pending_gaps = capture_pending_rows(memory, store, represented)
    coverage.setdefault("gaps", []).extend(pending_gaps)
    # Preserve the existing explicit fallback when the scan boundary is unknown.
    pending = raw if pending_all is None else [row for row in pending_all if predicate(row)]
    pending_all = pending_all or []
    # An authored interim account can cover an unfinished OTHER room, while
    # returning to that room restores its still-open source words. Unrepresented
    # closed sources also stay until a meaningful account exists.
    pending_by_id = {source_row_id(row): row for row in pending if source_row_id(row) not in covered}
    pending_by_id.update((source_row_id(row), row) for row in focused_sources
                         if source_row_id(row) in covered and source_row_id(row) not in closed)
    source_order = {locator["source_row_id"]: index for index, locator in enumerate(coverage.get("row_locators", []))}
    pending = sorted(pending_by_id.values(), key=lambda row: source_order.get(source_row_id(row), len(source_order)))
    other_open: dict[str, dict] = {}
    for row in pending_all:
        if predicate(row) or source_row_id(row) in represented:
            continue
        destination = str(bound_room_chat(bindings, row) or resolver.room_id(row))
        room = other_open.setdefault(destination, {"id": destination,
            "label": resolver.label({**row, "chat_id": destination}), "rows": []})
        room["rows"].append(row)
    context = SimpleNamespace(drive_root=memory.drive_root, task_id=str(task.get("id") or "chronicle-context"))
    locators = coverage.pop("row_locators", [])
    source_ref = retain_room_source(memory, context.task_id, raw, locators, coverage)
    other_ref = (retain_memory_source(context, "unconsolidated_rooms", _encoded(other_open).encode("utf-8"), "json")
                 if other_open else {})
    return _encoded({"schema": 1, "focus": focus, "rooms": rooms,
                     "raw_focus": raw, "open_focus": pending, "source_ref": source_ref,
                     "other_open_rooms": list(other_open.values()), "other_open_source": other_ref,
                     "coverage": coverage, "marks": store.active_marks(focus),
                     "maintenance": {"last_error": scan.get("last_consolidation_error"),
                                     "pending_operations": scan.get("pending_consolidation_outcomes", [])},
                     "is_child": str(task.get("delegation_role") or metadata.get("delegation_role") or "") == "subagent"})


def _raw_text(rows: list[dict], source_ref: dict | None) -> str:
    from ouroboros.chronicle_sources import format_source_row

    if not rows:
        return ""
    body = "\n\n".join(format_source_row(row) for row in rows)
    return ("[Full message text and recorded status; transport/UI bookkeeping stays in the exact retained source]\n"
            + body + ("\nComplete retained source: memory_read(source_ref=" + _encoded(source_ref) + ")."
                      if source_ref is not None else ""))


def render_memory(snapshot: dict, token_budget: int | None = None, *, shared_out: dict | None = None,
                  refusal_recovery: bool = False, refused_memory_bytes: int | None = None) -> tuple[str, dict]:
    """Choose complete published views, preserving meaning before lowering detail.

    A budget is an economic/physical fact, never an importance classifier. Other
    rooms start at their smallest already-authored cover; no source is dropped
    to make an arithmetic target appear satisfied.
    """
    if refusal_recovery and refused_memory_bytes is None:
        # Compatibility for callers without a captured view: compare with this
        # same snapshot's ordinary projection, never infer a route capacity.
        ordinary, _ = render_memory(snapshot, token_budget,
                                    shared_out={} if shared_out is not None else None)
        refused_memory_bytes = len(ordinary.encode("utf-8"))

    def reduction_needed(value):
        target_miss = token_budget is not None and estimate_tokens(value) > token_budget
        physical_view = value + (_TARGET_MISS_NOTICE if target_miss else "")
        return target_miss or (refusal_recovery and len(physical_view.encode("utf-8")) >= refused_memory_bytes)

    def refusal_shrinks(value):
        target_miss = token_budget is not None and estimate_tokens(value) > token_budget
        physical_view = value + (_TARGET_MISS_NOTICE if target_miss else "")
        return refusal_recovery and len(physical_view.encode("utf-8")) < refused_memory_bytes

    focus = str(snapshot.get("focus", "1"))
    rooms = snapshot.get("rooms") or []
    cuts = {room["id"]: _cuts(room.get("records") or []) for room in rooms}
    selected = {room: len(options) - 1 for room, options in cuts.items()}
    if focus in selected:
        selected[focus] = 0
    raw_full = bool(snapshot.get("raw_focus")) and (snapshot.get("coverage") or {}).get("full_room_in_view", True)
    shared_ids, shared_digest_ids, shared_parts, shared_focus = set(), set(), [], ""
    if shared_out is not None:
        # Closed published interpretations have the same words for every focus.
        # New episodes/open work stay in the mutable tail; no timer reseals them.
        for room in sorted(rooms, key=lambda item: item["id"]):
            stable = [row for row in cuts[room["id"]][-1] if row.get("kind") in {"digest", "legacy"}]
            if stable:
                room_text = f"### {room.get('label', room['id'])}\n\n" + "\n\n".join(_record_text(row) for row in stable)
                shared_parts.append(room_text)
                if room["id"] == focus:
                    shared_focus = room_text
                shared_ids.update(_current_id(row) for row in stable)
                shared_digest_ids.update(_current_id(row) for row in stable if row.get("kind") == "digest")
        shared_out["text"] = ("## Dialogue History\n\nShared closed history; finer views below retain the same sources.\n\n"
                              + "\n\n".join(shared_parts)) if shared_parts else ""

    def visible_rows(room):
        rid = room["id"]
        rows = [row for row in cuts[rid][selected[rid]] if _current_id(row) not in shared_ids]
        # Complete raw focus replaces duplicate narrative coverage, not its sources.
        if rid == focus and raw_full:
            rows = [row for row in rows if row.get("kind") == "gap" or (row.get("metadata") or {}).get("source_gap")
                    or (row.get("author") or {}).get("kind") == "mind"]
        return rows

    def compose() -> str:
        parts = ["## Dialogue History\n\nOne source-grounded life; room views are foci, not separate identities."]
        for room in rooms:
            rid = room["id"]
            rows = visible_rows(room)
            if rows:
                parts.append(f"### {room.get('label', rid)}\n\n" + "\n\n".join(_record_text(row) for row in rows))
        raw = snapshot.get("raw_focus" if raw_full else "open_focus") or []
        if raw:
            from ouroboros.context_runtime_facts import snapshot_labelled
            recent = "## Recent chat\n\n" + _raw_text(raw, snapshot.get("source_ref") or {})
            parts.append(snapshot_labelled(recent, snapshot["captured_at"]) if snapshot.get("captured_at") else recent)
        for room in snapshot.get("other_open_rooms") or []:
            parts.append(f"### Unconsolidated source — {room.get('label', room['id'])}\n"
                         + _raw_text(room.get("rows") or [], None))
        if any(room.get("rows") for room in snapshot.get("other_open_rooms") or []):
            parts.append("Complete retained source for the unconsolidated rooms above: memory_read(source_ref="
                         + _encoded(snapshot.get("other_open_source") or {}) + ").")
        if not raw_full and (snapshot.get("raw_focus") or (snapshot.get("coverage") or {}).get("matched_rows")):
            parts.append("[Focused conversation at reduced detail; exact original remains available]\n"
                         + _encoded(snapshot.get("source_ref") or {}))
        for mark in snapshot.get("marks") or []:
            text = f"[Significance mark {mark['id']}; author={_encoded(mark.get('author', {}))}]\n{mark.get('text', '')}"
            if mark.get("quote"):
                text += ("\nExact words: " + mark["quote"] if mark.get("visibility", "full") == "full" else
                         "\n[Exact words are not in this view; the authored meaning above remains. Read the source when exact wording matters.]")
            text += "\nSource: " + _encoded(mark.get("target_ref") or {})
            text += f"\nRead/release: memory_read(node_id={mark['id']!r}); memory_mark(release_id={mark['id']!r}, reason=...)."
            parts.append(text)
        if snapshot.get("mark_source_status"):
            parts.append("[Active mark source unavailable; previously captured marks remain, not an empty current set]\n"
                         + _encoded(snapshot["mark_source_status"]))
        gaps = (snapshot.get("coverage") or {}).get("gaps")
        if gaps:
            parts.append("[Recorded history gaps; complete coverage is not established]\n" + _encoded(gaps))
        if snapshot.get("maintenance"):
            parts.append("[Memory maintenance fact; inspect sources and judge what needs repair]\n" + _encoded(snapshot["maintenance"]))
        return "\n\n".join([*([shared_out["text"]] if shared_out is not None and shared_out["text"] else []), *parts])

    # Other rooms initially use their smallest published covers. Reduce focus
    # detail last, measuring the whole frame including its common cover. A
    # shorter digest may otherwise add bytes beside the fine focused records.
    text = compose()
    if reduction_needed(text):
        focus_words = shared_focus + _raw_text(
            snapshot.get("raw_focus" if raw_full else "open_focus") or [], snapshot.get("source_ref") or {})
        focus_words += "\n\n".join(_record_text(row) for room in rooms if room["id"] == focus
                                    for row in visible_rows(room))
        focus_fits = token_budget is None or estimate_tokens(focus_words) <= token_budget
        full_before = raw_full
        raw_full = False
        reduced = compose()
        if focus in cuts:
            while selected[focus] < len(cuts[focus]) - 1 and reduction_needed(reduced):
                selected[focus] += 1
                reduced = compose()
        # Do not lose independently fitting focus detail when reducing it
        # cannot resolve pressure from the rest of the biography.
        if len(reduced) < len(text) and (refusal_shrinks(reduced) or not focus_fits or
                                      (token_budget is not None and estimate_tokens(reduced) <= token_budget)):
            text = reduced
        else:
            raw_full = full_before
            if focus in selected:
                selected[focus] = 0
    # Keep a stable meaningful common base, then use remaining space for
    # available finer accounts. Traverse whole published alternatives in their
    # existing presentation order; this is fitting, not an importance score.
    for room in rooms:
        rid = room["id"]
        if rid == focus:
            continue
        while selected[rid] > 0:
            previous = selected[rid]
            selected[rid] -= 1
            expanded = compose()
            if reduction_needed(expanded):
                selected[rid] = previous
                break
            text = expanded
    visible = [row for room in rooms for row in visible_rows(room)]
    selected_ids = shared_ids | {_current_id(row) for row in visible}
    digest_ids = shared_digest_ids | {_current_id(row) for row in visible if row.get("kind") == "digest"}
    facts = {"requested_memory_tokens": token_budget, "rendered_memory_tokens": estimate_tokens(text),
             "selected_record_count": len(selected_ids), "selected_digest_ids": sorted(digest_ids),
             "shared_memory_tokens": estimate_tokens(shared_out["text"]) if shared_out and shared_out.get("text") else 0,
             "full_focused_room": raw_full,
             "legacy_transition": any(row.get("kind") == "legacy" and (row.get("author") or {}).get("kind") != "host"
                                      for options in cuts.values() for row in options[-1]),
             "verbatim_marks": all(mark.get("visibility", "full") == "full" for mark in snapshot.get("marks") or []),
             "levels": selected, "target_miss": token_budget is not None and estimate_tokens(text) > token_budget}
    if facts["target_miss"]:
        text += _TARGET_MISS_NOTICE
    facts.update(rendered_memory_chars=len(text), rendered_memory_bytes=len(text.encode("utf-8")),
                 rendered_memory_tokens=estimate_tokens(text),
                 selection_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    if refusal_recovery:
        facts["reduction_requested_after_refusal"] = True
    return text, facts


def _memory_allowance(system_content, snapshot, mode, window_tokens, calibration_ratio,
                      output_reserve_tokens, task):
    from ouroboros.context_budget import (
        NANO_MIN_HEADROOM_TOKENS, OWNER_LOW_TARGET_TOKENS, OWNER_NANO_TARGET_TOKENS,
        RECLAIM_LOW_WATER_DIVISOR,
    )
    target = {"low": OWNER_LOW_TARGET_TOKENS, "nano": OWNER_NANO_TARGET_TOKENS}.get(mode)
    if mode == "low" and task.get("owner_context_mode") == "max":
        target = None  # Overflow projection is not an owner choice of economy mode.
    boundary = min(target, window_tokens) if target and window_tokens else target or window_tokens
    ratio = max(float(calibration_ratio or 1), 0.01)
    remaining = None
    if boundary:
        reserve = NANO_MIN_HEADROOM_TOKENS if mode == "nano" else output_reserve_tokens
        direct = task.get("_is_direct_chat") or task.get("type") in {"consciousness", "consciousness_wake"} or snapshot.get("is_child")
        margins = 1 if direct else 2
        margin = (int(boundary) + RECLAIM_LOW_WATER_DIVISOR - 1) // RECLAIM_LOW_WATER_DIVISOR
        other = sum(estimate_tokens(str(block.get("text", "")).replace(CHRONICLE_MARKER, "")) for block in system_content)
        other += int(task.get("context_user_tokens") or 0)
        if task.get("context_non_memory_tokens") is not None:
            other = max(0, int(task["context_non_memory_tokens"]))
        remaining = max(0, int((int(boundary) - reserve - margins * margin) / ratio) - other)
    return remaining


def render_system_view(system_content: list[dict], snapshot_json: str, *, mode: str,
                       window_tokens: int, calibration_ratio: float, output_reserve_tokens: int,
                       task: dict, facts_out: dict | None = None) -> list[dict]:
    """Pure fitting hook; rebind uses the SAME source snapshot and unrendered blocks."""
    if not snapshot_json:
        return system_content
    snapshot = json.loads(snapshot_json)
    remaining = _memory_allowance(system_content, snapshot, mode, window_tokens,
                                 calibration_ratio, output_reserve_tokens, task)
    shared = {}
    text, facts = render_memory(snapshot, remaining, shared_out=shared,
        refusal_recovery=bool(task.get("memory_refusal_recovery")),
        refused_memory_bytes=task.get("refused_memory_bytes"))
    if facts_out is not None:
        facts_out.update(facts)
    rendered = copy.deepcopy(system_content)
    common = shared.get("text") or ""
    if common:
        text = text[len(common) + 2:]
        rendered.insert(1, {"type": "text", "text": MEMORY_BEGIN + common + MEMORY_END,
                            "cache_control": {"type": "ephemeral"}, "_shared_memory": True})
    for block in rendered:
        if isinstance(block.get("text"), str) and CHRONICLE_MARKER in block["text"]:
            block["text"] = block["text"].replace(CHRONICLE_MARKER,
                MEMORY_BEGIN + text + MEMORY_END + MEMORY_FACTS_PREFIX + _encoded(facts) + "\n")
    return rendered


def refresh_chronicle_snapshot(snapshot_json: str, data_root: Any) -> str:
    """Refresh interpretations after maintenance, retaining the captured raw sources."""
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.projects_registry import all_task_bindings

    snapshot = json.loads(snapshot_json)
    store = ChronicleStore(data_root)
    snapshot["rooms"], covered = _derived_rooms(store, snapshot["focus"],
        RoomLabelResolver(data_root), all_task_bindings(data_root))
    from ouroboros.room_consolidation import _closed_source_rows

    closed = _closed_source_rows(data_root, snapshot.get("raw_focus") or snapshot.get("open_focus", []), None)
    snapshot["open_focus"] = [row for row in snapshot.get("open_focus", [])
                              if source_row_id(row) not in covered or source_row_id(row) not in closed]
    represented = {key for room in snapshot["rooms"] for record in room["records"]
                   for key in (record.get("metadata") or {}).get("source_row_ids", [])}
    snapshot["other_open_rooms"] = [{**room, "rows": [row for row in room.get("rows", [])
        if source_row_id(row) not in represented]} for room in snapshot.get("other_open_rooms", [])]
    snapshot["marks"] = store.active_marks(snapshot["focus"])
    scan = store.scan_state()
    snapshot["maintenance"] = {"last_error": scan.get("last_consolidation_error"),
                               "pending_operations": scan.get("pending_consolidation_outcomes", [])}
    return _encoded(snapshot)


def helper_memory_reference(ctx: Any, task: dict | None = None) -> dict:
    """One retained parent-selected reference for ordinary external helpers.

    The existing work-order owner records these bytes; retries replay them.
    Native harness overhead is not observed by this compiler and is not claimed
    to fit merely because the parent's known projection does.
    """
    from ouroboros.contracts.task_contract import task_input_sources

    parent = {"task_contract": getattr(ctx, "task_contract", {}),
              "metadata": getattr(ctx, "task_metadata", {})}
    if task_input_sources(task or {}) == "declared" or task_input_sources(parent) == "declared":
        return {}
    plan = getattr(ctx, "context_fit_plan", None)
    source = getattr(plan, "chronicle_state_json", "")
    if not source:
        return {}
    mode = str(getattr(ctx, "active_context_mode", "") or plan.rendered_mode or plan.initial_mode)
    templates = plan.system_templates_json
    template = json.loads(templates.get("low") or templates.get(mode) or "[]")
    snapshot = json.loads(source)
    snapshot["is_child"] = True
    task_facts = {**plan.context_task, "delegation_role": "subagent"}
    remaining = _memory_allowance(template, snapshot, mode, plan.window_tokens,
        plan.projection(mode).calibration_ratio, plan.output_reserve_tokens, task_facts)
    text, facts = render_memory(snapshot, remaining,
        refusal_recovery=bool(task_facts.get("memory_refusal_recovery")),
        refused_memory_bytes=task_facts.get("refused_memory_bytes"))
    orientation = snapshot.get("shared_orientation", "")
    from ouroboros.tool_access import canonical_data_root

    source_hint = {"canonical_root": str(canonical_data_root(ctx)),
                   "chronicle_path": "memory/chronicle/records.jsonl",
                   "reader": "memory_read for Ouroboros; external file readers may locate transaction records by id, or request exact sources through the existing parent channel. Source availability is not proof of reading."}
    return {"text": "\n\n".join(part for part in (orientation, text, "Memory source access: " + _encoded(source_hint)) if part),
            "facts": {**facts, **source_hint, "basis": "parent-selected frozen memory view",
                      "external_native_context_fit": "unobserved"},
            "snapshot_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}


def maintenance_projection(env: Any, memory: Any, task: dict):
    """Evaluate the owner's current Main view on the existing post-task rail.

    The initial source capture is frozen; each fit check refreshes only derived
    interpretations and marks. This is a fresh post-task evaluation, not a claim
    about the exact last physical request or a new background scheduler.
    """
    from ouroboros.context import build_context_fit_plan
    from ouroboros.config import get_context_mode, get_owner_context_mode

    plan = build_context_fit_plan(env, memory, task, preferred_mode=get_context_mode())
    if not plan.chronicle_state_json:
        return None, {"evaluation": "legacy_view"}
    mode = plan.preferred_mode
    snapshot = json.loads(plan.chronicle_state_json)
    template = json.loads(plan.system_templates_json.get(mode) or "[]")
    budget = _memory_allowance(template, snapshot, mode, plan.window_tokens,
        plan.projection(mode).calibration_ratio, plan.output_reserve_tokens, plan.context_task)
    canonical_root = task.get("budget_drive_root") or getattr(env, "budget_drive_root", None) or memory.drive_root
    facts = {"evaluation": "post_task_main_view", "mode": mode, "memory_budget_tokens": budget,
             "maintenance_target_reachable": budget != 0,
             "owner_context_mode": get_owner_context_mode(), "window_tokens": plan.window_tokens,
             "route_fp": plan.route_fp, "context_source_sha256": plan.core_sha256}
    if budget == 0:
        facts["maintenance_target_reason"] = "non_memory_core_exhausts_target"

    def fits():
        current = json.loads(refresh_chronicle_snapshot(plan.chronicle_state_json, canonical_root))
        _text, projection = render_memory(current, budget, shared_out={})
        facts.update(projection)
        # A zero allowance means the rest of the context alone exhausts the
        # bound. Buying another summary cannot reach it, and must not loop.
        # True here stops futile work; target_miss still reports the real view.
        return budget == 0 or not projection["target_miss"]

    fits()
    return fits, facts
