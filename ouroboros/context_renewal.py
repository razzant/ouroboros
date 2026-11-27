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
newest note; and the mind's own prose since its newest note, including the visible text it wrote
beside a tool call. Chosen reads (their whole exchange) and explicitly restored sources stay whole. The record lists
each removed tool unit since the newest note with its typed facts, shown or never shown. Unasked tool units, replies
the notes already cover, host state the host sends afresh (facts, clock, roster and skill snapshots,
bound and unchanged), superseded review copies and earlier address records leave the view;
one host record names the checkpoint that holds them exactly. Owner 29.10 Q1=A:
if that seed still exceeds the physical window, the oldest whole rows may leave to an
exact copy by address and the work continues. Among them host records and tool results
leave first and the owner's words last (CHECKLISTS item 18; BIBLE P1: operative human
guidance stays whole; ``_cut_tier``); core, assignment, notes, current review decisions,
chosen reads, host state its producer would send again unchanged, the newest owner turn,
the newest exchange and what arrived after the last act stay, and a call leaves only with
all its results, never while its native replay is unconsumed (``_cut_units``). An
irreducible view, or a candidate that does not shrink, still refuses, and so does an
excess a directly required handbook alone causes (owner 1.11.2026).
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_ROW_KINDS, REQUIRED_BOOK_UNFIT_KEY, bound_host_row

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


def units_since_note(messages: Sequence[Dict[str, Any]], observation: Any, *, usable_input: Any = None,
                     density: float = 1.0) -> str:
    """Typed facts of every tool unit after the newest note, each named shown or never shown to me.

    Plan item 16 (review r1, 17.10): the record indexes what left the view since the newest note, so a
    renewal never hides which calls ran (a child started, a commit attempted) or which results the mind
    never saw. The facts are ``context_source_view._unit_facts`` without error texts (the physical floor stays
    lean; each reader keeps them one read away); exposure is the usable-send evidence
    (``_exposed_units``); unknown exposure reads as never shown.
    """
    from ouroboros import context_compaction as compaction
    from ouroboros.context_source_view import _exposed_units, _unit_facts

    newest = _newest_note(messages)
    units = [unit for unit in compaction.context_units(list(messages), scope="tool")
             if unit.start > newest and compaction.unit_kind(messages, unit) == "tool"]  # calls, not records
    from ouroboros.context_source_view import chosen_read_units
    retained = chosen_read_units(messages, units, usable_input=usable_input, density=density)
    units = [unit for unit in units if unit.unit_id not in retained]
    shown = _exposed_units(list(messages), units, observation if isinstance(observation, dict) else {})
    return "\n".join(f"{_unit_facts(messages, unit, error_text=False)}\n  {'shown to me' if unit.unit_id in shown else 'never shown to me'}"
                     for unit in units)


def renewal_seed(messages: Sequence[Dict[str, Any]], *, usable_input: Any = None,
                 density: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``(kept rows in order, counts of the rows that leave the view)``; pure, no I/O. A chosen read stays when it fits
    known usable input alone (every chosen read when that is unknown), as at the address rungs."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.context_budget import host_row_intact
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.llm_messages import DISPLAY_KEY, shown_content
    from ouroboros.loop_messages import _visible_round_text
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
    from ouroboros.review_history_view import REVIEW_CONTEXT_INDEX_KEY, obsolete_review_index_positions
    from ouroboros.context_source_view import chosen_read_rows
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME
    from ouroboros.send_clock import CLOCK_ROW_NAME

    keep_read = chosen_read_rows(messages, _protocol_spans(messages), usable_input=usable_input, density=density)
    resent = _restated_host_kinds() | {CONTEXT_FACTS_NAME, CLOCK_ROW_NAME, RENEWAL_RECORD_NAME}

    note = _is_note
    first_user = next((i for i, row in enumerate(messages) if row.get("role") == "user"), -1)
    newest_note = _newest_note(messages)
    superseded = set(obsolete_review_index_positions(messages))
    kept: List[Dict[str, Any]] = []
    left = {"tool_rows": 0, "covered_replies": 0, "host_rows": 0, "restored_sources": 0, "projected_replies": 0}
    for i, row in enumerate(messages):
        role, kind = row.get("role"), row.get(HOST_CONTEXT_KIND_KEY)
        tool_row = role == "tool" or bool(row.get("tool_calls"))
        projected_reply = (role == "assistant" and i > newest_note and isinstance(row.get(DISPLAY_KEY), dict)
                           and row[DISPLAY_KEY].get("text") != row.get("content")
                           and not any(custody_private_key(key) for key in row))
        if (i in keep_read or _restored_address(row) or (i == 0 and role == "system") or i == first_user or kind in OWNER_ROW_KINDS or note(row)
                or kind == PREDECESSOR_NOTES_NAME or (row.get(REVIEW_CONTEXT_INDEX_KEY) and i not in superseded)):
            kept.append(row)
        elif tool_row or projected_reply:
            left["tool_rows" if tool_row else "projected_replies"] += 1
            shown = shown_content(row) if role == "assistant" and i > newest_note else None  # a lane's view, maybe ""
            prose = shown if isinstance(shown, str) else _visible_round_text(shown)  # blocks: their visible text alone
            if prose and prose.strip():
                kept.append({"role": "assistant", "content": prose})  # words beside a call or its omitted fragments
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
    plan = getattr(tools._ctx, "context_fit_plan", None)
    def measure(rows):
        return math.ceil(estimate_context_prompt_tokens(rows, schemas, provider=getattr(plan, "provider", ""),
            reasoning_effort=getattr(limit_ctx, "active_effort", "")) * density)
    capacity = usage.get("_context_capacity_total_tokens")
    reserve = usage.get("_context_response_reserve_tokens") or 0
    book = int(usage.get("_context_required_book_tokens") or 0)
    # A required handbook is kept, never paid for: only what a view without it could not hold renews or cuts,
    # and an excess nothing measures is not guessed onto my memory. A known window that holds that view is the
    # task's typed incompatibility (``context_capacity``); an unknown one proves none.
    if book and (capacity is None or measure(messages) - book + reserve <= capacity):
        usage[REQUIRED_BOOK_UNFIT_KEY] = capacity is not None
        return _cannot_help(limit_ctx, plan)
    usable = capacity - reserve if capacity is not None else None
    seed, record = renewal_view(messages, observation=getattr(tools._ctx, "_last_context_observation", None),
                                drive_root=root, task_id=limit_ctx.task_id, measure=measure, usable_input=usable,
                                density=density)
    if capacity is not None and measure(seed) - book + reserve > capacity:
        seed, cut = physical_window_cut(seed, drive_root=root, task_id=limit_ctx.task_id,
                                        measure=lambda rows: measure(rows) - book, usable=usable, density=density)
        if cut:
            record = {**(record or {}), "physical_cut": cut}
    if measure(seed) >= measure(messages):
        return _cannot_help(limit_ctx, plan)
    if not record:
        return _cannot_help(limit_ctx, plan)
    _publish_reclaimed_transcript(SimpleNamespace(messages=messages, task_id=limit_ctx.task_id, tools=tools), seed,
                                  by="renewal")
    from ouroboros.loop import _emit_checkpoint_event

    _emit_checkpoint_event(limit_ctx.event_queue, limit_ctx.task_id, limit_ctx.drive_logs, {
        "checkpoint_kind": "context_view_renewed", "round": limit_ctx.round_idx, "owner_visible": True,
        "rows_before": before, "rows_after": len(seed), **record})
    return True


def renewal_view(messages, *, observation, drive_root, task_id, measure, usable_input=None, density=1.0):
    """Pure selection plus exact custody; both drivers price their own final envelope.

    No round flags, dispatch, publication or trace pruning belongs here.
    """
    from ouroboros.artifacts import actor_source_reference, store_actor_source_bytes
    seed, left = renewal_seed(messages, usable_input=usable_input, density=density)
    if not (left["tool_rows"] or left["covered_replies"] or left["restored_sources"] or left["projected_replies"]):
        return messages, None
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    args = dict(category="context_checkpoints", source_id=hashlib.sha256(raw).hexdigest(), data=raw, extension="json")
    source = actor_source_reference(**args)
    index = units_since_note(messages, observation, usable_input=usable_input, density=density)
    projected = left["projected_replies"]
    seed.append(bound_host_row(RENEWAL_RECORD_NAME, (
        "[Host record: working view renewed during context-size recovery. Same task; "
        "nothing summarized; no model called.] Kept, in order: the assignment, every delivered message and "
        "outward dialogue, my notes, my replies since my newest note and the current review decisions. "
        f"Left the view: {left['tool_rows']} tool rows, {left['covered_replies']} replies before my newest note, "
        f"{left['host_rows']} host rows, {left['restored_sources']} sources I restored. "
        + (f"Kept only visible prose from {projected} replies with omitted call fragments. " if projected else "")
        + f"The exact transcript of {len(messages)} rows: {json.dumps(source, ensure_ascii=False, sort_keys=True)}."
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


def _cut_tier(messages, group, superseded):
    """The order a physical cut spends a whole unit in, by who produced it as far as typed metadata its producer can
    still vouch for shows (never text): superseded host copies, then host rows still bound to their text and address
    records with valid metadata, then tool exchanges and recorded review disputes bound to their source, then the mind's
    prose, and outward dialogue with the owner's and principals' words last, together with any delivered row whose
    provenance cannot rule out human words: an untyped row, a typed row whose text no longer matches its binding (an
    older merge appended words to it), an address record or dispute body whose binding fails (CHECKLISTS item 18: the
    owner's words last; BIBLE P1: operative human guidance stays whole)."""
    from ouroboros.context_budget import OWNER_ROW_KINDS, REPLY_CITATION_KIND, RETURN_FACTS_KIND, host_row_intact
    from ouroboros.context_compaction import _capsule_metadata, _review_source_ref
    from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY

    rows = [messages[i] for i in group]
    kinds = {row.get(HOST_CONTEXT_KIND_KEY) for row in rows}
    if kinds & (OWNER_ROW_KINDS | {REPLY_CITATION_KIND, RETURN_FACTS_KIND}):
        return 4
    if all(i in superseded for i in group):
        return 0
    head = rows[0]
    capsule, meta = _capsule_metadata(head)
    if capsule:  # a host address record (the mind's notes and restored reads are protected)
        return 1 if meta is not None else 4
    if any(row.get('role') == 'tool' or row.get('tool_calls') for row in rows):
        return 2
    if head.get(REVIEW_HISTORY_MESSAGE_KEY):
        return 2 if _review_source_ref(head) is not None else 4
    if head.get('role') == 'assistant':
        return 3
    return 1 if head.get(HOST_CONTEXT_KIND_KEY) and host_row_intact(head) else 4


def _restated_host_kinds():
    """Host state its producer appends again unchanged whenever no visible copy states it: the roster chain and skill
    snapshots. The facts row and the clock are sampled afresh for every request instead."""
    from ouroboros.peer_roster import ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME
    from ouroboros.skill_context_snapshot import SNAPSHOT_KIND

    return frozenset({ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME, SNAPSHOT_KIND})


def _protocol_spans(messages):
    """``{call row: last result row}``: an assistant row with tool calls through the last tool row before the next
    assistant row, complete or not, with what the host delivered between them (an image a viewing call attached while
    the batch ran; the wire moves such rows after the results, ``llm_messages._order_complete_tool_episodes``), whether
    or not the mind may fold it (``context_units`` lists only foldable, contiguous ones)."""
    spans, i = {}, 0
    while i < len(messages):
        calls = messages[i].get('tool_calls')
        end = i
        if messages[i].get('role') == 'assistant' and isinstance(calls, list) and calls:
            cursor = i + 1
            while cursor < len(messages) and messages[cursor].get('role') != 'assistant':
                end = cursor if messages[cursor].get('role') == 'tool' else end
                cursor += 1
            spans[i] = end
        i = end + 1
    return spans


def _cut_units(messages, *, usable_input=None, density=1.0):
    """Eligible whole units in the order a physical cut spends them (``_cut_tier``), oldest first within a tier.

    Protected: the core and assignment, the mind's and a predecessor's notes, chosen reads with their whole exchange
    (when they fit known usable input alone) and restored reads, current review decisions and feedback, current host
    state its producer would send again unchanged (the roster chain, skill snapshots and mark changes no later row
    states anew: cutting them frees nothing; an earlier facts row or clock pays, the next request samples its own),
    earlier cut records (folded into the new one), the newest owner turn with the citation and return context beside
    it, the newest exchange (the span holding the mind's last reply or tool result, or the batch the host's bound
    record of an outward message closed) and every row after that act, a batch nobody answered yet even when the owner
    spoke after it; before any act, every row after the newest owner turn (with no owner turn either, the last row).
    A call and all its results, complete or not, are one unit (``_protocol_spans``), and an exchange whose native
    replay is still unconsumed stays whole wherever it sits. Work after the newest owner turn is not protected for its
    position: hours of later tool traffic leave before the owner's earlier words do."""
    from ouroboros.anthropic_native_custody import anthropic_tool_unit_active
    from ouroboros.context_budget import (HOST_TEXT_SHA_KEY, OWNER_WORDS_KIND, REPLY_CITATION_KIND,
                                         REPLY_SOURCE_SHA_KEY, RETURN_FACTS_KIND, host_row_intact)
    from ouroboros.context_source_view import _obsolete_host_rows, chosen_read_rows
    from ouroboros.main_context_authority import LATE_ANSWER_NOTES_NAME, PREDECESSOR_NOTES_NAME
    from ouroboros.mark_anchors import MARK_CHANGES_KIND
    from ouroboros.openai_responses_custody import responses_tool_unit_active
    from ouroboros.review_history_view import (REVIEW_CONTEXT_INDEX_KEY, REVIEW_HISTORY_MESSAGE_KEY,
                                               obsolete_review_index_positions)
    first = next((i for i, row in enumerate(messages) if row.get('role') == 'user'), -1)
    newest = max((i for i, row in enumerate(messages) if i > first
                  and row.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND), default=None)
    turn = set()
    if newest is not None:
        start = newest
        while start > first + 1 and messages[start - 1].get(HOST_CONTEXT_KIND_KEY) in {
                OWNER_WORDS_KIND, REPLY_CITATION_KIND, RETURN_FACTS_KIND}:  # context beside the newest owner turn
            start -= 1
        turn = set(range(start, newest + 1))
    obsolete = set(obsolete_review_index_positions(messages))
    feedback = _current_feedback_positions(messages)
    spans = _protocol_spans(messages)
    # The newest exchange: the span of the mind's last reply or call when it acted after the newest owner turn (else
    # the owner turn is the newest thing to act on), and every row that arrived after that act. Nobody has seen what
    # arrived before a first act.
    acts = [i for i, row in enumerate(messages) if i > first and row.get('role') in ('assistant', 'tool')]
    if acts:
        since = next((start for start, end in spans.items() if start <= acts[-1] <= end), acts[-1])
        unanswered = messages[acts[-1]].get('role') == 'tool'  # no reply of the mind after its batch's results yet
        if (since == acts[-1] and messages[since].get(HOST_CONTEXT_KIND_KEY) == 'owner_dialogue'
                and host_row_intact(messages[since])):
            # After a batch's results the host appends its own rows before any new reply: review updates, then the
            # record of an outward message (role assistant, bound to its text at ``tools/owner_delivery``), which
            # answers none of those results; so the batch it closes belongs to the newest exchange too. Completed
            # speech in a call carries no binding: it is the mind's answer, and the batch before it is older work.
            cursor = since
            while cursor - 1 > first and (messages[cursor - 1].get(HOST_CONTEXT_KIND_KEY) == 'owner_dialogue'
                                          or messages[cursor - 1].get(REVIEW_CONTEXT_INDEX_KEY)
                                          or messages[cursor - 1].get(REVIEW_HISTORY_MESSAGE_KEY)):
                cursor -= 1
            closed = next((start for start, end in spans.items() if end == cursor - 1), None)
            if closed is not None:
                since, unanswered = closed, True
        # An unanswered batch stays even when my human spoke after it: the mind has not yet seen its results.
        exchange = range(since if newest is None or since > newest or unanswered else acts[-1] + 1, len(messages))
    else:
        exchange = range(len(messages) - 1 if newest is None else newest + 1, len(messages))
    superseded = _obsolete_host_rows(messages)
    restated = _restated_host_kinds() | {MARK_CHANGES_KIND}
    protected = {i for i, row in enumerate(messages) if i <= first or i in turn or i in exchange or row.get('role') == 'system'
        or _is_note(row) or _restored_address(row)
        or row.get(HOST_CONTEXT_KIND_KEY) in {PREDECESSOR_NOTES_NAME, LATE_ANSWER_NOTES_NAME}
        or i in feedback or (row.get(REVIEW_CONTEXT_INDEX_KEY) and i not in obsolete)
        or row.get(HOST_CONTEXT_KIND_KEY) == 'context_physical_cut'
        or (row.get(HOST_CONTEXT_KIND_KEY) in restated and host_row_intact(row) and i not in superseded)}
    # A provider that still has to replay a call's native reasoning beside its results needs both in the wire request.
    protected.update(i for start, end in spans.items() if anthropic_tool_unit_active(messages, start, end)
                     or responses_tool_unit_active(messages, start, end) for i in range(start, end + 1))
    protected.update(chosen_read_rows(messages, spans, usable_input=usable_input, density=density))
    # A fresh selection may reuse a source delivered many turns ago. Its digest, rather than adjacency,
    # keeps that exact reading (including its files) beside the protected turn.
    needed = {messages[i][REPLY_SOURCE_SHA_KEY] for i in protected if messages[i].get(REPLY_SOURCE_SHA_KEY)}
    protected.update(i for i, row in enumerate(messages) if row.get(HOST_CONTEXT_KIND_KEY) == REPLY_CITATION_KIND
                     and row.get(HOST_TEXT_SHA_KEY) in needed and host_row_intact(row))
    prior = {i for i, row in enumerate(messages) if i > first
             and row.get(HOST_CONTEXT_KIND_KEY) == 'context_physical_cut' and host_row_intact(row)}
    groups, i = [], first + 1
    while i < len(messages):
        end = spans.get(i, i)
        if messages[i].get(HOST_CONTEXT_KIND_KEY) == REPLY_CITATION_KIND:
            words = i + 1
            if words < len(messages) and messages[words].get(HOST_CONTEXT_KIND_KEY) == RETURN_FACTS_KIND:
                words += 1  # the drain may insert a host fact before its first owner words
            if words < len(messages) and messages[words].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND:
                end = words  # retain or evict the addressed pair together
        group = tuple(range(i, end + 1))
        if not protected.intersection(group):
            groups.append((_cut_tier(messages, group, superseded), group))
        i = group[-1] + 1
    return [group for _tier, group in sorted(groups)], prior


def _position_runs(positions):
    runs = []
    for position in positions:
        if runs and position == runs[-1][1] + 1:
            runs[-1][1] = position
        else:
            runs.append([position, position])
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in runs)


def physical_cut_record(positions, prior, source):
    """The cut's one host record: which positions left, the owner's words last, and the exact checkpoint reader. Every
    candidate pays for its words, so naming the order costs no more than the older record's opening."""
    return bound_host_row('context_physical_cut', (
        "[Host record: this window's physical cut, not a summary or an authored fold.] "
        f"Rows left the view, owner's words last: zero-based positions {_position_runs(positions)}. "
        + (f"Earlier physical-cut address records at positions {_position_runs(sorted(prior))} remain exact in that checkpoint. " if prior else '')
        + "Newest owner turn, assignment, core, notes and last exchange stay. "
        f"Read the exact original rows in this checkpoint JSON array: {json.dumps(source, sort_keys=True)}."))


def physical_window_cut(messages, *, drive_root, task_id, measure, usable, density=1.0):
    """Cut whole units in ``_cut_units`` order under physical pressure, never protected judgment.

    Search prefixes in order, bisecting only intervals with unchanged images.
    Batched image floors can raise the price when an image leaves. An expensive
    last prefix proves nothing about an earlier interval. Publish one exact cut.
    """
    from ouroboros.artifacts import actor_source_reference, store_actor_source_bytes
    before = measure(messages)
    if before <= usable:
        return messages, None
    groups, prior = _cut_units(messages, usable_input=usable, density=density)
    if not groups:
        return messages, None
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    args = dict(category='context_checkpoints', source_id=hashlib.sha256(raw).hexdigest(), data=raw, extension='json')
    source = actor_source_reference(**args)
    def candidate(count):
        positions = sorted(i for group in groups[:count] for i in group)
        gone = set(positions) | prior
        address = physical_cut_record(positions, prior, source)
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


def _cannot_help(limit_ctx: Any, plan: Any) -> bool:
    """Name the frame's largest parts for the named refusal (``loop_transport``), then decline: raw estimates in the
    refused request's shape (the plan the renewal measured with and the round's active effort), kept as the
    ``[label, tokens]`` pairs a same-task checkpoint carries across versions."""
    from ouroboros.context_view_facts import largest_parts

    view = SimpleNamespace(messages=limit_ctx.messages, tool_schemas=getattr(limit_ctx, "tool_schemas", None) or [],
                           accumulated_usage=limit_ctx.accumulated_usage, context_fit_plan=plan,
                           active_effort=getattr(limit_ctx, "active_effort", ""))
    try:
        limit_ctx.accumulated_usage["_context_refusal_frame"] = largest_parts(view, measured=False)
    except Exception:  # the refusal text is facts only; an unreadable frame stays unnamed
        pass
    return False
