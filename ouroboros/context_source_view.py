"""Model-free source-address reconstruction after a real context refusal.

The compactor owns exact checkpoints, units and capsules. This leaf owns only
which complete sources an emergency view represents and the facts it shows.
It never writes a summary or claims that the mind understood an omitted body.
"""
from __future__ import annotations

import pathlib
from collections import Counter
from dataclasses import replace
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from ouroboros import context_compaction as compaction
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, ContextReclaimRequest, ContextReclaimReceipt, _AtomicUnit, _Selection, _SelectedUnit
from ouroboros.tool_result_record import read_tool_result_record

EmergencyRung = Literal["host_copies", "bodies", "unseen_bodies"]
_EMERGENCY_LABELS = {  # the record's honest heading: no summary, exact rows retained by checkpoint
    "host_copies": ("Emergency host record after the provider's context refusal: obsolete host rows "
                    "replaced by typed state and exact addresses; nothing summarized; exact rows retained by checkpoint"),
    "bodies": ("Emergency host record after the provider's context refusal: tool bodies previously exposed to a "
               "usable Main turn replaced by typed facts and exact addresses; exposure proves no understanding; "
               "nothing summarized; exact rows retained by checkpoint"),
    "unseen_bodies": ("Late source-only rescue after other same-model recovery: tool bodies replaced by typed facts "
                      "and exact addresses; usable Main exposure is UNCONFIRMED, not proof of understanding; "
                      "nothing summarized; exact rows retained by checkpoint")}
_FACT_ARGUMENTS_CHARS = 400


def _obsolete_host_rows(messages: Sequence[Mapping[str, Any]]) -> set[int]:
    """Only producer-labelled state replaced by a later full snapshot is obsolete.

    Legacy/unlabelled prose stays raw. A user protocol role alone says nothing
    about authorship, and neither an old warning nor a peer message is obsolete
    merely because the mind has already seen it.
    """
    from ouroboros.loop_messages import CONTEXT_FACTS_NAME
    from ouroboros.peer_roster import ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME
    from ouroboros.review_history_view import obsolete_review_index_positions

    obsolete = set()
    for full_name, names in ((CONTEXT_FACTS_NAME, {CONTEXT_FACTS_NAME}),
                            (ROSTER_SNAPSHOT_NAME, {ROSTER_SNAPSHOT_NAME, ROSTER_UPDATE_NAME})):
        latest = max((i for i, row in enumerate(messages)
                      if row.get("role") == "user" and row.get(HOST_CONTEXT_KIND_KEY) == full_name
                      and not row.get("tool_calls") and not row.get("function_call")
                      and compaction._plain_text(row.get("content"))), default=-1)
        obsolete.update(i for i, row in enumerate(messages[:max(0, latest)])
                        if row.get("role") == "user" and row.get(HOST_CONTEXT_KIND_KEY) in names)
    return obsolete | set(obsolete_review_index_positions(messages))


def _exposed_units(messages: list, units: Sequence[_AtomicUnit], observation: Mapping[str, Any]) -> set[str]:
    """Match an observed original unit, never its current position or a reused call ID.

    Cache presentation may change and host rows may leave. Complete normalized source
    bytes include the producer invocation identity; multiplicity prevents a copied row
    from borrowing another occurrence's exposure. Missing invocation custody stays unknown.
    """
    def key(rows):
        normalized = compaction._view_source_messages(rows)
        results = [row for row in normalized if row.get("role") == "tool"]
        if not results or any(read_tool_result_record(row)["state"] != "recorded" for row in results):
            return None
        return compaction.context_reclaim_transcript_sha256(normalized)

    original = observation.get("messages") or []
    declared = {(ref.get("unit_id"), ref.get("raw_sha256")) for ref in observation.get("exposed_units") or ()}
    remaining = Counter()
    for unit in compaction.context_units(original, scope="tool"):
        if (unit.unit_id, unit.raw_sha256) in declared and compaction.unit_kind(original, unit) == "tool":
            if identity := key(original[unit.start:unit.end + 1]):
                remaining[identity] += 1
    exposed = set()
    for unit in units:
        if identity := key(messages[unit.start:unit.end + 1]):
            if remaining[identity] > 0:
                remaining[identity] -= 1
                exposed.add(unit.unit_id)
    return exposed


def _unit_facts(messages: Sequence[Mapping[str, Any]], unit: _AtomicUnit) -> str:
    """Typed facts of one unit from its protocol rows and trace references, never a retelling."""
    rows = compaction._view_source_messages(messages[unit.start:unit.end + 1])
    lines = [f"Unit {unit.unit_id}: tool, {unit.context_size_tokens} tokens estimated"]
    results = {str(row.get("tool_call_id") or ""): row for row in rows if str(row.get("role") or "") == "tool"}
    for call in rows[0].get("tool_calls") or []:
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        call_id, arguments = str(call.get("id") or ""), str(function.get("arguments") or "")
        shown = (arguments if len(arguments) <= _FACT_ARGUMENTS_CHARS
                 else f"<arguments {len(arguments)} chars, sha256 {compaction._sha256(arguments)[:16]}>")
        result_row = results.get(call_id, {})
        record = read_tool_result_record(result_row)
        result = result_row.get("content")
        result_text = compaction._plain_text(result)
        result_chars = len(result_text) if result_text is not None else len(compaction._canonical_json(result))
        detail = f"; outcome {compaction._canonical_json(record['facts'])}"
        if record["state"] == "unknown":
            detail += f"; outcome binding unknown ({record['reason']})"
        else:
            detail += f"; invocation {record['invocation']['invocation_id']}"
        for label, key in (("trace", "trace_ref"), ("source", "source_ref")):
            if record.get(key):
                detail += f"; {label} {compaction._canonical_json(record[key])}"
        lines.append(f"- {function.get('name') or '?'} {shown} -> result {result_chars} chars{detail}")
    return "\n".join(lines)


def emergency_address_view(
    messages: list,
    request: ContextReclaimRequest,
    *,
    rung: EmergencyRung,
    protected_texts: Sequence[str] = (),
    observation: Optional[Mapping[str, Any]] = None,
    measure_candidate: Optional[Callable[[list], int]] = None,
    trace_refs_by_tool_call_id: Optional[Mapping[str, Any]] = None,
    drive_root: pathlib.Path,
    task_id: str,
) -> Tuple[list, ContextReclaimReceipt]:
    """Model-free pass after the provider's typed refusal of this very request.

    Obsolete host rows share one exact original-checkpoint pointer; review indexes
    have no invented selectable unit IDs. ``bodies`` takes confirmed exposed complete
    tool units oldest first, stopping at the measured goal. ``unseen_bodies`` is the
    separately disclosed late rescue of units whose exposure is unconfirmed. Exact
    body units remain restorable; no view claims understanding. ``measure_candidate``
    measures the complete final request, including the seal and fresh host facts.
    """
    trace_refs = trace_refs_by_tool_call_id or {}
    density = request.measurement_density
    before_sha = compaction.context_reclaim_transcript_sha256(messages)
    if str(request.transcript_sha256 or "") != before_sha:
        return messages, compaction._receipt("binding_mismatch", before_sha=before_sha)
    units = compaction.context_units(messages, scope="dialogue", trace_refs_by_tool_call_id=trace_refs,
                          measurement_density=density)
    protected = compaction.owner_protected_unit_ids(messages, units, protected_texts)
    obsolete = _obsolete_host_rows(messages) if rung == "host_copies" else set()
    first_user = next((i for i, row in enumerate(messages) if row.get("role") == "user"), -1)
    obsolete = {i for i in obsolete if i != first_user and not messages[i].get("tool_calls")
                and not messages[i].get("function_call")
                and (text := compaction._plain_text(messages[i].get("content"))) is not None
                and not any(word and word in text for word in protected_texts)}
    raw = [u for u in units if compaction.unit_kind(messages, u) == "tool" and u.generation == 0
           and u.unit_id not in protected]
    exposed = _exposed_units(messages, raw, observation or {})
    eligible = [u for u in raw if (u.unit_id in exposed) == (rung == "bodies")] if rung != "host_copies" else []
    if not eligible and not obsolete:
        return messages, compaction._receipt("no_eligible", before_sha=before_sha)
    fingerprint = compaction._sha256(compaction._canonical_bytes({"rung": rung, "transcript": before_sha,
                                           "units": [u.unit_id for u in eligible], "host_positions": sorted(obsolete)}))
    # One exact source write for the entire pass. This empty selection checkpoints the
    # original before materialization; only the receipt below names units actually removed.
    selection = _Selection((), fingerprint, 0)
    checkpoint_ref = compaction._persist_reclaim_checkpoint(messages, request, selection, drive_root=drive_root, task_id=task_id)
    if checkpoint_ref is None:
        return messages, compaction._receipt("checkpoint_failed", before_sha=before_sha, selection=selection)
    measure = measure_candidate or (lambda rows: compaction._context_tokens_for_messages(rows, density))
    before_tokens = measure(messages)
    if obsolete:
        source = [messages[i] for i in sorted(obsolete)]
        unit = compaction._unit_from_slice(source, 0, len(source) - 1, trace_refs_by_tool_call_id={},
                                           measurement_density=density)
        # This is the host record's identity, not a selectable unit of the original.
        unit = replace(unit, unit_id=f"host-copies:{fingerprint}")
        record, ref = compaction._capsule_message(_SelectedUnit(unit, 0, ""),
            f"{len(obsolete)} obsolete host rows replaced together; latest full snapshots retained.",
            [compaction._part(unit.unit_id, unit.source_text)], checkpoint_ref, request,
            retention="source_view", label=_EMERGENCY_LABELS[rung], address={"checkpoint_ref": dict(checkpoint_ref)})
        first = min(obsolete)
        rebuilt = [record if i == first else row for i, row in enumerate(messages) if i == first or i not in obsolete]
        reclaimed = before_tokens - measure(rebuilt)
        return (rebuilt, compaction._receipt("applied", before_sha=before_sha,
            after_sha=compaction.context_reclaim_transcript_sha256(rebuilt), selection=selection,
            reclaimed_tokens=reclaimed, goal_reached=reclaimed >= request.reclaim_goal_tokens,
            checkpoint_ref=checkpoint_ref, capsule_refs=[ref], source_refs=(checkpoint_ref,))
        ) if reclaimed > 0 else (messages, compaction._receipt("no_measurable_shrink", before_sha=before_sha,
                                                             selection=selection, checkpoint_ref=checkpoint_ref))
    # Rebuild each cumulative prefix with the real checkpoint address. A complete unit
    # may overshoot the goal; later units remain raw once the final request has reclaimed it.
    groups: List[List[_AtomicUnit]] = []
    replacements: Dict[int, tuple] = {}
    selected: List[_AtomicUnit] = []
    source_refs: List[Dict[str, Any]] = []
    rebuilt, capsule_refs, reclaimed = messages, [], 0
    for unit in eligible:
        if groups and groups[-1] and groups[-1][-1].end + 1 == unit.start:
            groups[-1].append(unit)
        else:
            groups.append([unit])
        group = groups[-1]
        start, end = group[0].start, group[-1].end
        combined = compaction._unit_from_slice(messages, start, end, trace_refs_by_tool_call_id=trace_refs,
                                    measurement_density=density)
        if combined is None:
            continue
        members = [{"unit_id": u.unit_id, "raw_sha256": u.raw_sha256} for u in group]
        member_refs = [{"checkpoint_ref": checkpoint_ref, **member} for member in members]
        combined = replace(
            combined,
            lineage_hashes=compaction._unique_strings([*combined.lineage_hashes, *(h for u in group for h in u.lineage_hashes)]),
            source_refs=compaction._unique_refs([*combined.source_refs, *(r for u in group for r in u.source_refs), *member_refs]))
        record, ref = compaction._capsule_message(
            _SelectedUnit(combined, 0, ""), "\n".join(_unit_facts(messages, u) for u in group),
            [compaction._part(combined.unit_id, combined.source_text)], checkpoint_ref, request, retention="source_view",
            label=_EMERGENCY_LABELS[rung], address={"checkpoint_ref": dict(checkpoint_ref), "units": members})
        proposal = {**replacements, start: (end, [record], [ref])}
        candidate, refs = compaction._materialize_replacements(messages, proposal)
        delta = before_tokens - measure(candidate)
        if delta <= reclaimed:
            group.pop()  # a non-economic unit stays raw and breaks adjacency
            continue
        replacements, rebuilt, capsule_refs, reclaimed = proposal, candidate, refs, delta
        selected.append(unit)
        source_refs.append(member_refs[-1])
        if reclaimed >= request.reclaim_goal_tokens:
            break
    selection = replace(selection, units=tuple(_SelectedUnit(u, 0, "") for u in selected),
                        predicted_reclaim_tokens=sum(u.predicted_reclaim_tokens for u in selected))
    if reclaimed <= 0:
        return messages, compaction._receipt("no_measurable_shrink", before_sha=before_sha, selection=selection,
                                  checkpoint_ref=checkpoint_ref)
    return rebuilt, compaction._receipt(
        "applied", before_sha=before_sha, after_sha=compaction.context_reclaim_transcript_sha256(rebuilt),
        selection=selection, reclaimed_tokens=reclaimed, goal_reached=reclaimed >= int(request.reclaim_goal_tokens),
        checkpoint_ref=checkpoint_ref, capsule_refs=capsule_refs,
        source_refs=compaction._unique_refs([*source_refs, checkpoint_ref]),
    )
