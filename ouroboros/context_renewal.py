"""The floor of the refusal path: the same task continues in a fresh view of its own record.

Owner decision Q1=A (17.10.2026). When the model-free refusal passes and the configured
fallback could not bring a refused request under the window, the task does not end: its
conversation is rebuilt as a seed of what continuity needs, and everything else stays one
address away in an exact checkpoint of the whole transcript. The task id, attempt,
children, budget, route, pending questions and custody are those of the running loop, so
only the view changes. No model is called and nothing is summarized; the mind's own notes
are never dropped for age.

The seed keeps, in their original order: the system view and the assignment; every delivered
message (the owner's and principals' words typed at delivery, review feedback, notices, and
any row written before host state was bound to its text, which may hold merged owner words);
Ouroboros's outward dialogue; the mind's authored notes (actor capsules) and a predecessor's
notes; the current review decisions; the address records the refusal passes wrote since the
newest note; and the mind's own prose since its newest note, including the text it wrote beside
a tool call. Chosen read units and explicitly restored sources stay whole. The record lists each removed
tool unit since the newest note with its typed facts, shown or never shown. Unasked tool units, replies the notes
already cover, host state the host sends afresh (facts, clock, roster and skill snapshots,
bound and unchanged), superseded review copies and earlier address records leave the view;
one host record names the checkpoint that holds them exactly. Owner 29.10 Q1=A:
if that seed still exceeds the physical window, oldest whole rows of any kind may
leave by exact address; core, assignment, notes, newest owner turn and current tail
stay. An irreducible view, or a candidate that does not shrink, still refuses.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_ROW_KINDS, bound_host_row

RENEWAL_RECORD_NAME = "ouroboros_view_renewal"


def _is_note(row: Dict[str, Any]) -> bool:
    """An actor capsule: a note the mind authored and kept."""
    from ouroboros.context_compaction import _capsule_metadata

    capsule, meta = _capsule_metadata(row)
    return bool(capsule and meta and meta.get("authorship") == "actor")


def _newest_note(messages: Sequence[Dict[str, Any]]) -> int:
    return max((i for i, row in enumerate(messages) if _is_note(row)), default=-1)


def _restored_address(row: Dict[str, Any]) -> Dict[str, Any]:
    """The exact address of a source the mind restored (``restore_unit_refs``), else ``{}``.

    A restored source is a host source view built at the actor's request (``route_fp == "actor"``);
    the refusal passes' address records carry the refused route instead (review r2, 18.10)."""
    from ouroboros.context_compaction import _capsule_metadata

    capsule, meta = _capsule_metadata(row)
    if not (capsule and meta and meta.get("retention") == "source_view" and meta.get("route_fp") == "actor"):
        return {}
    return {"checkpoint_ref": meta["checkpoint_ref"], "unit_id": meta["unit_id"], "raw_sha256": meta["source_unit_sha256"]}


def restored_sources(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Addresses of the sources the mind restored in this view, in order (a restore made in the same call as
    a note sits before that note)."""
    return [address for row in messages if (address := _restored_address(row))]


def units_since_note(messages: Sequence[Dict[str, Any]], observation: Any) -> str:
    """Typed facts of every tool unit after the newest note, each named shown or never shown to me.

    Plan item 16 (review r1, 17.10): the record indexes what left the view since the newest note, so a
    renewal never hides which calls ran (a child started, a commit attempted) or which results the mind
    never saw. The facts are ``context_source_view._unit_facts``; exposure is the usable-send evidence
    (``_exposed_units``); unknown exposure reads as never shown.
    """
    from ouroboros import context_compaction as compaction
    from ouroboros.context_source_view import _exposed_units, _unit_facts

    newest = _newest_note(messages)
    units = [unit for unit in compaction.context_units(list(messages), scope="tool")
             if unit.start > newest and compaction.unit_kind(messages, unit) == "tool"]  # calls, not records
    from ouroboros.context_source_view import chosen_read_units
    retained = chosen_read_units(messages, units)
    units = [unit for unit in units if unit.unit_id not in retained]
    shown = _exposed_units(list(messages), units, observation if isinstance(observation, dict) else {})
    return "\n".join(f"{_unit_facts(messages, unit)}\n  {'shown to me' if unit.unit_id in shown else 'never shown to me'}"
                     for unit in units)


def renewal_seed(messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``(kept rows in order, counts of the rows that leave the view)``; pure, no I/O."""
    from ouroboros.context_budget import host_row_intact
    from ouroboros.context_compaction import _capsule_metadata, _plain_text
    from ouroboros.loop_messages import CONTEXT_FACTS_NAME
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
    from ouroboros.peer_roster import ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY, obsolete_review_index_positions
    from ouroboros.send_clock import CLOCK_ROW_NAME
    from ouroboros.skill_context_snapshot import SNAPSHOT_KIND
    from ouroboros import context_compaction as cc
    from ouroboros.context_source_view import chosen_read_units

    units = cc.context_units(list(messages), scope="tool")
    chosen = chosen_read_units(messages, units)
    keep_read = {i for unit in units if unit.unit_id in chosen for i in range(unit.start, unit.end + 1)}

    resent = {CONTEXT_FACTS_NAME, CLOCK_ROW_NAME, ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME, SNAPSHOT_KIND,
              RENEWAL_RECORD_NAME}

    note = _is_note
    first_user = next((i for i, row in enumerate(messages) if row.get("role") == "user"), -1)
    newest_note = _newest_note(messages)
    superseded = set(obsolete_review_index_positions(messages))
    kept: List[Dict[str, Any]] = []
    left = {"tool_rows": 0, "covered_replies": 0, "host_rows": 0, "restored_sources": 0}
    for i, row in enumerate(messages):
        role, kind = row.get("role"), row.get(HOST_CONTEXT_KIND_KEY)
        if (i in keep_read or _restored_address(row) or (i == 0 and role == "system") or i == first_user or kind in OWNER_ROW_KINDS or note(row)
                or kind == PREDECESSOR_NOTES_NAME or (row.get(REVIEW_CONTEXT_INDEX_KEY) and i not in superseded)):
            kept.append(row)
        elif role == "tool" or row.get("tool_calls"):
            left["tool_rows"] += 1
            prose = _plain_text(row.get("content")) if role == "assistant" and i > newest_note else None
            if prose and prose.strip():
                kept.append({"role": "assistant", "content": prose})  # my words beside the call stay
        elif (_capsule_metadata(row)[0] or kind == RENEWAL_RECORD_NAME) and i > newest_note:
            kept.append(row)  # address records and earlier renewal indexes since my newest note: what left, with facts
        elif (_capsule_metadata(row)[0] or row.get(REVIEW_CONTEXT_INDEX_KEY)
              or (kind in resent and host_row_intact(row))):
            left["host_rows"] += 1  # older addresses, superseded review copies, state the host sends afresh
        elif role == "assistant" and i <= newest_note:
            left["covered_replies"] += 1
        else:
            kept.append(row)  # every delivered message and my replies since my newest note
    return kept, left


def renew_after_refusal(limit_ctx: Any, tools: Any) -> bool:
    """Publish the renewal seed for a round that ended in a size refusal; False when renewal cannot help."""
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
    from ouroboros.loop_model_call import CONTEXT_REFUSAL_ROUND_KEY
    from ouroboros.loop_round_limits import _publish_reclaimed_transcript

    usage = limit_ctx.accumulated_usage
    # This round's own size refusal decides, not the kind a configured fallback failed with afterwards
    # (its resource refusal stays recorded beside the renewal; the next primary call clears it).
    if (usage.get(CONTEXT_REFUSAL_ROUND_KEY) != limit_ctx.round_idx
            or isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict) or usage.get("_pending_transport_outcome")):
        return False
    messages = limit_ctx.messages
    before = len(messages)
    root = limit_ctx.drive_root or limit_ctx.drive_logs.parent
    schemas = getattr(limit_ctx, "tool_schemas", None)
    density = usage.get("_context_measurement_density") or 1.0
    def measure(rows):
        return math.ceil(estimate_context_prompt_tokens(rows, schemas) * density)
    seed, record = renewal_view(messages, observation=getattr(tools._ctx, "_last_context_observation", None),
                                drive_root=root, task_id=limit_ctx.task_id, measure=measure)
    capacity = usage.get("_context_capacity_total_tokens")
    reserve = usage.get("_context_response_reserve_tokens") or 0
    if capacity is not None and measure(seed) + reserve > capacity:
        seed, cut = physical_window_cut(seed, drive_root=root, task_id=limit_ctx.task_id,
                                        measure=measure, usable=capacity - reserve)
        if cut:
            record = {**(record or {}), "physical_cut": cut}
    if measure(seed) >= measure(messages):
        return _cannot_help(limit_ctx)
    if not record:
        return _cannot_help(limit_ctx)
    _publish_reclaimed_transcript(SimpleNamespace(messages=messages, task_id=limit_ctx.task_id, tools=tools), seed,
                                  by="renewal")
    from ouroboros.loop import _emit_checkpoint_event

    _emit_checkpoint_event(limit_ctx.event_queue, limit_ctx.task_id, limit_ctx.drive_logs, {
        "checkpoint_kind": "context_view_renewed", "round": limit_ctx.round_idx, "owner_visible": True,
        "rows_before": before, "rows_after": len(seed), **record})
    return True


def renewal_view(messages, *, observation, drive_root, task_id, measure):
    """Pure selection plus exact custody; both drivers price their own final envelope.

    No round flags, dispatch, publication or trace pruning belongs here.
    """
    from ouroboros.artifacts import actor_source_reference, store_actor_source_bytes
    seed, left = renewal_seed(messages)
    if not (left["tool_rows"] or left["covered_replies"] or left["restored_sources"]):
        return messages, None
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    args = dict(category="context_checkpoints", source_id=hashlib.sha256(raw).hexdigest(), data=raw, extension="json")
    source = actor_source_reference(**args)
    index = units_since_note(messages, observation)
    seed.append(bound_host_row(RENEWAL_RECORD_NAME, (
        "[Host record: working view renewed during context-size recovery. Same task; "
        "nothing summarized; no model called.] Kept, in order: the assignment, every delivered message and "
        "outward dialogue, my notes, my replies since my newest note and the current review decisions. "
        f"Left the view: {left['tool_rows']} tool rows, {left['covered_replies']} replies before my newest note, "
        f"{left['host_rows']} host rows, {left['restored_sources']} sources I restored. The exact transcript "
        f"of {len(messages)} rows: {json.dumps(source, ensure_ascii=False, sort_keys=True)}."
        + (f"\nTool units since my newest note (they left the view; each stays exact there):\n{index}" if index else ""))))
    if measure(seed) >= measure(messages):
        return messages, None
    try:
        store_actor_source_bytes(drive_root, task_id, **args)
    except OSError:
        return messages, None
    return seed, {"source": source, **left}


def _current_feedback_positions(messages):
    # The current index keeps decisions whole. Delivery notices from superseded
    # rounds are ordinary rows; retain the newest notice for each review family.
    latest, unknown = {}, set()
    for i, row in enumerate(messages):
        feedback = row.get('review_feedback')
        if not feedback:
            continue
        if not isinstance(feedback, list) or not all(isinstance(item, dict) for item in feedback):
            unknown.add(i)
            continue
        for item in feedback:
            latest[(item.get('surface', ''), item.get('task_id', ''))] = i
    return unknown | set(latest.values())


def _cut_units(messages):
    """Oldest eligible atomic units; protection is independent of dialogue kind."""
    from ouroboros import context_compaction as cc
    from ouroboros.context_budget import OWNER_WORDS_KIND, host_row_intact
    from ouroboros.context_source_view import chosen_read_units
    from ouroboros.main_context_authority import LATE_ANSWER_NOTES_NAME, PREDECESSOR_NOTES_NAME
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY, obsolete_review_index_positions
    first = next((i for i, row in enumerate(messages) if row.get('role') == 'user'), -1)
    latest = max((i for i, row in enumerate(messages) if i > first
                  and row.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND), default=len(messages) - 1)
    while latest > first + 1 and messages[latest - 1].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND:
        latest -= 1
    obsolete = set(obsolete_review_index_positions(messages))
    feedback = _current_feedback_positions(messages)
    protected = {i for i, row in enumerate(messages) if i <= first or i >= latest or row.get('role') == 'system'
        or _is_note(row) or _restored_address(row)
        or row.get(HOST_CONTEXT_KIND_KEY) in {PREDECESSOR_NOTES_NAME, LATE_ANSWER_NOTES_NAME}
        or i in feedback or (row.get(REVIEW_CONTEXT_INDEX_KEY) and i not in obsolete)
        or row.get(HOST_CONTEXT_KIND_KEY) == 'context_physical_cut'}
    prior = {i for i, row in enumerate(messages) if first < i < latest
             and row.get(HOST_CONTEXT_KIND_KEY) == 'context_physical_cut' and host_row_intact(row)}
    units = cc.context_units(list(messages), scope='tool')
    chosen = chosen_read_units(messages, units)
    spans = {unit.start: unit for unit in units}
    groups, i = [], first + 1
    while i < latest:
        unit = spans.get(i)
        group = tuple(range(i, unit.end + 1)) if unit else (i,)
        if not protected.intersection(group) and not (unit and unit.unit_id in chosen):
            groups.append(group)
        i = group[-1] + 1
    return groups, prior


def _position_runs(positions):
    runs = []
    for position in positions:
        if runs and position == runs[-1][1] + 1:
            runs[-1][1] = position
        else:
            runs.append([position, position])
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in runs)


def physical_window_cut(messages, *, drive_root, task_id, measure, usable):
    """Cut the oldest whole units under physical pressure, never protected judgment.

    Search prefixes in order, bisecting only intervals with unchanged images.
    Batched image floors can raise the price when an image leaves. An expensive
    last prefix proves nothing about an earlier interval. Publish one exact cut.
    """
    from ouroboros.artifacts import actor_source_reference, store_actor_source_bytes
    before = measure(messages)
    if before <= usable:
        return messages, None
    groups, prior = _cut_units(messages)
    if not groups:
        return messages, None
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    args = dict(category='context_checkpoints', source_id=hashlib.sha256(raw).hexdigest(), data=raw, extension='json')
    source = actor_source_reference(**args)
    def candidate(count):
        positions = [i for group in groups[:count] for i in group]
        gone = set(positions) | prior
        address = bound_host_row('context_physical_cut', (
            "[Host record: this window's physical cut, not a summary or an authored fold.] "
            f"Oldest whole rows left the working view: zero-based positions {_position_runs(positions)}. "
            + (f"Earlier physical-cut address records at positions {_position_runs(sorted(prior))} remain exact in that checkpoint. " if prior else '')
            + "Newest owner turn, assignment, core, notes and current tail stay. "
            f"Read the exact original rows in this checkpoint JSON array: {json.dumps(source, sort_keys=True)}."))
        rows = [address if i == positions[0] else row for i, row in enumerate(messages)
                if i == positions[0] or i not in gone]
        return rows, positions, measure(rows)
    from ouroboros.context_images import image_positions
    image_rows = {i for i, _ in image_positions(messages)}
    starts = sorted({1, len(groups) + 1, *(n for n, group in enumerate(groups, 1) if image_rows.intersection(group))})
    best = None
    for start, end in zip(starts, starts[1:]):
        low, high = start, end - 1
        trial = candidate(high)
        if trial[2] > usable:
            continue
        best = trial
        while low < high:
            middle = (low + high) // 2
            trial = candidate(middle)
            if trial[2] <= usable:
                high, best = middle, trial
            else:
                low = middle + 1
        break
    if best is None:
        return messages, None
    try:
        store_actor_source_bytes(drive_root, task_id, **args)
    except OSError:
        return messages, None
    rows, positions, after = best
    return rows, {'source': source, 'positions': positions, 'previous_cuts': sorted(prior),
                  'before_tokens': before, 'after_tokens': after}


def _cannot_help(limit_ctx: Any) -> bool:
    """Name the frame's largest parts for the named refusal (``loop_transport``), then decline."""
    from ouroboros.loop_messages import _context_components

    view = SimpleNamespace(messages=limit_ctx.messages, tool_schemas=getattr(limit_ctx, "tool_schemas", None) or [],
                           accumulated_usage=limit_ctx.accumulated_usage,
                           context_fit_plan=getattr(limit_ctx, "context_fit_plan", None))
    try:
        limit_ctx.accumulated_usage["_context_refusal_frame"] = [list(part) for part in _context_components(
            view, measured=False)]
    except Exception:  # the refusal text is facts only; an unreadable frame stays unnamed
        pass
    return False
