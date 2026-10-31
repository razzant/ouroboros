"""Source-addressed first-show delivery of tool results.

One completed call batch is projected as ONE candidate against the measured
context frame its caller supplies (``fit_candidate(messages, schemas)`` returns
the existing Main measurement fields: ``estimated_input_tokens``,
``response_reserve_tokens``, ``target_total_tokens``, ``capacity_total_tokens``,
``measurement_basis``, ``measurement_density``). Every result keeps its typed
facts and an exact source address before any body is allocated; the bodies then
share one physical allowance derived from the measurement, never from a tool's
name or a path. A body the frame cannot hold is shown head+tail with exact
character ranges over the same text basis ``read_file`` pages (universal
newlines), so the omitted middle is one bounded read away and never a rerun of
the tool. A whole delivery is byte-identical to what the tool produced.

Two policies share the machinery:

``accepted_prefix``
    The consolidator's existing contract: the callback's ``accepted`` verdict
    drives a prefix search and the view is ``[Tool result source view]``.

``measured_frame``
    Unasked bodies share the measured eighth. Chosen bodies arrive as asked,
    even beside a crowded transcript: Main's pre-send recovery narrows memory
    and host material first. Only a chosen body exceeding usable input by itself
    may be cut here. A resulting non-fitting candidate is named for recovery,
    never mistaken for an accepted provider request.

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.tool_capabilities import PAGEABLE_TOOL_RESULTS, requested_result_view

log = logging.getLogger(__name__)

FitCandidate = Callable[[list, list], Mapping[str, Any]]

RESULT_VIEW_BASIS = "unicode_text_universal_newlines"
RESULT_VIEW_MARKER = "RESULT_VIEW_JSON="
FULL_SOURCE_MARKER = "FULL_RESULT_SOURCE_JSON="
FULL_SOURCE_UNAVAILABLE = "FULL_RESULT_SOURCE_UNAVAILABLE=true"
SOURCE_UNAVAILABLE_NOTE = "Do not treat this partial result as complete; exact source persistence failed."
RECOVER_PAGEABLE = "Read the exact source above, or request another range using this tool's declared parameters."
RECOVER_OTHER = "Do not rerun this tool to recover omitted output. Read the exact source above."

# Re-measurements of the complete candidate after the first allocation. Each one
# shrinks the bodies over their bound by the measured overshoot; the last resort
# is their minimum view (status ``minimum_view_unfit`` when the window is still over).
_MAX_REMEASURES = 4


def reader_text(text: Any) -> str:
    """The text basis every range here addresses: what ``read_file`` pages."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def persist_result_source(
    drive_root: pathlib.Path | str, task_id: str, source_id: str, text: Any,
) -> Dict[str, Any]:
    """Write-once exact source for one result, verified by an exact read-back.

    ``{}`` when the source cannot be persisted or read back: the caller then
    discloses the gap and claims no readability.
    """
    from ouroboros.artifacts import persist_exact_text_source

    _, ref, issue = persist_exact_text_source(
        pathlib.Path(drive_root), str(task_id), source_id=str(source_id), text=str(text),
    )
    if issue or not ref:
        log.warning("Tool result source not persisted for %s: %s", source_id, (issue or {}).get("reason"))
        return {}
    return ref


def compact_verification(fact: Any) -> Dict[str, Any]:
    """A check's recorded criterion verdict, compact: pass/fail, kind, match mode, the
    evaluated match (absent when no comparison ran) and the claim it supports (#1585)."""
    if not isinstance(fact, Mapping):
        return {}
    return {key: fact[key] for key in ("status", "contract_kind", "expected_match", "matched", "criterion_id")
            if key in fact}


def typed_result_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Status, code, process facts and any check verdict the producer published; nothing read from the text."""
    from ouroboros.tools.process_facts import PROCESS_FACT_KEYS

    meta = row.get("result_meta") if isinstance(row.get("result_meta"), Mapping) else {}
    typed = row.get("tool_result")
    facts: Dict[str, Any] = {"is_error": row.get("is_error") if isinstance(row.get("is_error"), bool) else None}
    status = getattr(typed, "status", None) or meta.get("tool_result_status") or meta.get("status")
    code = getattr(typed, "code", None) or meta.get("tool_result_code")
    if status:
        facts["status"] = str(status)
    if code:
        facts["code"] = str(code)
    facts.update({key: meta[key] for key in PROCESS_FACT_KEYS if key in meta})
    if verification := compact_verification(meta.get("verification")):
        facts["verification"] = verification
    return facts


def prepare_producer_sources(results: Sequence[Mapping[str, Any]], drive_root: Any, task_id: str,
                             notes: Sequence[str] = ()) -> list:
    """Retain clean producer text before measuring its model-visible locator.

    A row the host notes only at delivery (``notes``, aligned with ``results``) keeps the
    note as ``delivery_note`` and its own result as that clean text.
    """
    rows = [dict(row) for row in results]
    for index, row in enumerate(rows):
        producer_text = getattr(row.get("tool_result"), "producer_text", None)
        note = notes[index] if index < len(notes) else ""
        if note:
            row["delivery_note"] = note
        if producer_text is not None or note:
            clean = row["result"] if producer_text is None else producer_text
            row["producer_source_ref"] = (persist_result_source(drive_root, task_id,
                str(row["tool_call_id"]) + ".producer", clean) if drive_root is not None else {})
    return rows


def with_delivery_note(row: Mapping[str, Any], text: str) -> str:
    """What the mind is shown: ``text`` with the row's delivery note trailing it. Measurement
    and the transcript use this; the host readers of a result keep ``text``."""
    note = row.get("delivery_note")
    return f"{text}\n\n{note}" if note else text


def with_producer_source(row: Mapping[str, Any], text: str) -> str:
    """The same complete delivery text for measurement and transcript publication."""
    typed = row.get("tool_result")
    if getattr(typed, "producer_text", None) is None:
        return text
    if text != str(row["result"]) and typed.host_annotations:
        text += "\n\n" + "\n\n".join(typed.host_annotations)
    ref = row.get("producer_source_ref") or {}
    return text + (
        "\nPRODUCER_RESULT_SOURCE_JSON=" + json.dumps(ref, ensure_ascii=False)
        + "\nUnannotated tool data for programmatic reading; host notes and outcome still apply."
        if ref else
        "\nPRODUCER_RESULT_SOURCE_UNAVAILABLE=true"
        "\nHost notes remain in the result; no clean producer file was retained.")


def _span_text(ranges: Sequence[Sequence[int]]) -> str:
    return " and ".join(f"{start}\u2013{end}" for start, end in ranges) or "nothing"


def render_result_view(
    text: str, *, head_chars: int, tail_chars: int, tool_name: str = "",
    source_ref: Optional[Mapping[str, Any]] = None, facts: Optional[Mapping[str, Any]] = None,
    delivery: Optional[Mapping[str, Any]] = None,
    reader_available: bool = True,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Head+tail view of ``text`` with exact ranges, or ``text`` itself.

    Returns ``(delivered, view)``. ``view`` is ``None`` for a whole delivery:
    when the requested head and tail cover the text, or when the view would not
    be smaller than the text (the anti-growth floor). Otherwise ``delivered`` is
    the head, one line naming the omitted range, the tail, then -- outside the
    omitted body -- the typed facts and exact ranges as ``RESULT_VIEW_JSON=``,
    the source address as ``FULL_RESULT_SOURCE_JSON=`` (or the disclosed
    ``FULL_RESULT_SOURCE_UNAVAILABLE=true``) and the recovery sentence.
    """
    total = len(text)
    head = max(0, min(int(head_chars or 0), total))
    tail = max(0, min(int(tail_chars or 0), total - head))
    if head + tail >= total:
        return text, None
    omitted = [head, total - tail]
    shown = [[lo, hi] for lo, hi in ([0, head], [total - tail, total]) if lo < hi]
    ref = dict(source_ref) if isinstance(source_ref, Mapping) and source_ref else {}
    if not reader_available:
        ref.pop("read", None)
    view: Dict[str, Any] = {
        "complete_chars": total,
        "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "shown_ranges": shown,
        "omitted_ranges": [omitted],
        "range_basis": RESULT_VIEW_BASIS,
        "facts": dict(facts or {}),
    }
    if delivery is not None:
        # Host allocation facts never become producer stdout or producer status.
        view["delivery"] = {**delivery, "source_status": "ready" if ref else "source_unavailable"}
    if ref and reader_available:
        view["read_omitted"] = {
            "tool": "read_file", "root": str(ref.get("root") or "artifact_store"), "path": ref.get("path"),
            "start_line": 1, "max_lines": max(1, len(text.splitlines())),
            "start_char": omitted[0], "max_chars": omitted[1] - omitted[0],
        }
    marker = (
        f"\n... (truncated from {total} chars: shown {_span_text(shown)}, "
        f"omitted {omitted[0]}\u2013{omitted[1]}; exact source and facts below)\n"
    )
    lines = [RESULT_VIEW_MARKER + json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))]
    if ref:
        lines.append(FULL_SOURCE_MARKER + json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        lines.append((RECOVER_PAGEABLE if tool_name in PAGEABLE_TOOL_RESULTS else RECOVER_OTHER)
                     if reader_available else "The exact source is retained, but no reader is loaded in this connection. "
                     + ("Another range is available using this tool's declared parameters."
                        if tool_name in PAGEABLE_TOOL_RESULTS else "Do not rerun this tool to recover omitted output."))
    else:
        lines.extend([FULL_SOURCE_UNAVAILABLE, SOURCE_UNAVAILABLE_NOTE])
    delivered = text[:head] + marker + text[total - tail:] + "\n" + "\n".join(lines)
    if len(delivered) >= total:
        return text, None
    view.update(text_chars=len(delivered), delivered_range=[0, head])
    return delivered, view


def split_allowance(allowance: int, request: Optional[Tuple[Optional[int], Optional[int]]], total: int) -> Tuple[int, int]:
    """Head and tail chars for one result from its allowance and the call's request.

    A head/tail request keeps its proportions when it must shrink; any other
    allocation is split half and half. ``allowance >= total`` means whole.
    """
    allowance = max(0, min(int(allowance), total))
    if allowance >= total:
        return total, 0
    if request and request[0] is not None and request[1] is not None and (request[0] + request[1]) > 0:
        head_want, tail_want = int(request[0]), int(request[1])
        if head_want + tail_want <= allowance:
            return head_want, tail_want
        head = int(round(allowance * head_want / (head_want + tail_want)))
        return head, allowance - head
    head = int(math.ceil(allowance / 2))
    return head, allowance - head


def _water_fill(wants: Dict[int, int], budget: int) -> Dict[int, int]:
    """Equal shares, the smaller wants satisfied first; nothing exceeds its want."""
    out: Dict[int, int] = {}
    pending = sorted(wants.items(), key=lambda item: item[1])
    remaining = max(0, int(budget))
    for position, (index, want) in enumerate(pending):
        share = remaining // (len(pending) - position)
        out[index] = max(0, min(int(want), share))
        remaining -= out[index]
    return out


def _positive(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if int(value) > 0 else None


def _tokens(measurement: Mapping[str, Any]) -> int:
    return max(0, int(measurement.get("estimated_input_tokens") or 0))


def frame_bounds(measurement: Mapping[str, Any]) -> Tuple[Optional[int], int, int]:
    """``(B, R, eighth)`` of one measurement: the smaller known positive of the
    owner target and the route capacity, the reply reserve, and the one
    structural eighth ``ceil(B / RECLAIM_LOW_WATER_DIVISOR)`` (0 when B is unknown)."""
    from ouroboros.context_fit import reclaim_low_water_margin

    target = _positive(measurement.get("target_total_tokens"))
    capacity = _positive(measurement.get("capacity_total_tokens"))
    known = [value for value in (target, capacity) if value]
    reserve = max(0, int(measurement.get("response_reserve_tokens") or 0))
    return (min(known) if known else None), reserve, reclaim_low_water_margin(target, capacity)


def _standin_source_ref(call_id: Any, text: str, digest: str) -> Dict[str, Any]:
    """An address-shaped stand-in of the real locator's size, for measuring the
    minimum request before any source is written. Never delivered."""
    path = f"source_handles/tool_results/{call_id}-{digest}.txt"
    return {"kind": "task_source", "root": "artifact_store", "path": path, "size": len(text.encode("utf-8")),
            "sha256": digest, "read": {"tool": "read_file", "arguments": {
                "root": "artifact_store", "path": path, "start_line": 1, "max_lines": 2000, "start_char": 0}}}


def project_tool_result_batch(
    results: List[Dict[str, Any]], messages: List[Dict[str, Any]], tool_schemas: list,
    *, drive_root: pathlib.Path, task_id: str, fit_candidate: FitCandidate,
    policy: str = "accepted_prefix",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retain full results before constructing a fitting multi-result view.

    The caller supplies one completed call batch (rows with ``tool_call_id`` and
    ``result``; ``fn_name``, ``tool_args``, ``tool_result``, ``result_meta`` and
    ``is_error`` inform the measured policy) and its actual prospective send
    measurement. No result is appended to the live transcript here; source
    persistence, projection and callback failures never imply a full read.
    Returns copies of the rows (the row dict and its ``result_meta`` are the
    only things this projection writes; a typed ``tool_result`` is shared, not
    copied) with ``result`` replaced by the delivered text and, for a partial
    row, ``result_partial``, ``result_source_ref``, ``result_source_status`` and
    ``result_source_view``; plus a receipt whose ``status`` is ``complete``,
    ``projected``, ``minimum_view_unfit`` or (the measured policy only)
    ``capacity_unknown``.
    """
    rows = [{**row, **({"result_meta": dict(row["result_meta"])} if isinstance(row.get("result_meta"), dict) else {})}
            for row in results]
    full = [str(row["result"]) for row in rows]

    def measure(contents: Sequence[str]) -> Dict[str, Any]:
        # A fresh list over the live message dicts: the callback measures and
        # must not mutate (``measure_main_fit`` only serializes its inputs).
        candidate = [*messages, *({"role": "tool", "tool_call_id": row["tool_call_id"], "content": text}
                                  for row, text in zip(rows, contents))]
        return dict(fit_candidate(candidate, list(tool_schemas or [])))

    if policy == "measured_frame":
        return _measured_frame(rows, full, measure, drive_root=drive_root, task_id=task_id)
    if policy != "accepted_prefix":
        raise ValueError(f"unknown tool result delivery policy: {policy!r}")
    return _accepted_prefix(rows, full, measure, drive_root=drive_root, task_id=task_id)


def _mark_partial(row: Dict[str, Any], source: Dict[str, Any], view: Dict[str, Any]) -> None:
    status = "ready" if source else "source_unavailable"
    row.update(result_partial=True, result_source_ref=source, result_source_status=status,
               result_source_view={"source_ref": source, "source_status": status, **view})
    if isinstance(row.get("result_meta"), dict) and "knowledge_source_complete" in row["result_meta"]:
        row["result_meta"]["knowledge_source_complete"] = False


def _accepted_prefix(
    rows: List[Dict[str, Any]], full: List[str], measure: Callable[[Sequence[str]], Dict[str, Any]],
    *, drive_root: pathlib.Path, task_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The consolidator's contract: the callback's ``accepted`` verdict drives a prefix search."""
    initial = measure(full)
    if initial.get("accepted") is True:
        return rows, {"status": "complete", "fit": initial}
    sources = [persist_result_source(drive_root, task_id, row["tool_call_id"], text) for row, text in zip(rows, full)]
    shown = [0] * len(rows)

    def render(index: int) -> str:
        text, source, count = full[index], sources[index], shown[index]
        info = {"source_ref": source, "source_status": "ready" if source else "source_unavailable",
                "complete_chars": len(text), "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "requested_range": [0, len(text)], "delivered_range": [0, count], "text_chars": count,
                "text_sha256": hashlib.sha256(text[:count].encode("utf-8")).hexdigest()}
        rows[index]["result"] = text[:count] + "\n[Tool result source view]\n" + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
        _mark_partial(rows[index], source, info)
        rows[index]["result_partial"] = count < len(text)
        return rows[index]["result"]

    # Reserve every call's result envelope and source locator before allocating
    # body text. A later result can never disappear because an earlier one grew.
    contents = [render(index) for index in range(len(rows))]
    minimum = measure(contents)
    if minimum.get("accepted") is not True:
        return rows, {"status": "minimum_view_unfit", "fit": minimum}
    for index, text in enumerate(full):
        low, high = 0, len(text)
        while low < high:
            shown[index] = (low + high + 1) // 2
            contents[index] = render(index)
            if measure(contents).get("accepted") is True:
                low = shown[index]
            else:
                high = shown[index] - 1
        shown[index] = low
        contents[index] = render(index)
    final = measure(contents)
    return rows, {"status": "projected" if final.get("accepted") is True else "minimum_view_unfit", "fit": final}


def _measured_frame(
    rows: List[Dict[str, Any]], full: List[str], measure: Callable[[Sequence[str]], Dict[str, Any]],
    *, drive_root: pathlib.Path, task_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Main's first show: one measured batch allowance, requested forms first."""

    count = len(rows)
    texts = [reader_text(text) for text in full]
    digests = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    names = [str(row.get("fn_name") or row.get("tool") or "") for row in rows]
    facts = [typed_result_facts(row) for row in rows]
    requests = [requested_result_view(row.get("tool_args")) for row in rows]

    whole = measure(full)
    boundary, reserve, eighth = frame_bounds(whole)
    target = _positive(whole.get("target_total_tokens"))
    capacity = _positive(whole.get("capacity_total_tokens"))
    limiters = [set() for _ in rows]

    def diagnostic(index: int) -> Dict[str, Any]:
        return {"selection": "automatic" if requests[index] is None else "explicit",
                "limiters": sorted(limiters[index]), "target_tokens": target, "capacity_tokens": capacity,
                "reserve_tokens": reserve, "batch_share_tokens": eighth,
                "unasked_results": sum(request is None for request in requests)}

    for i, request in enumerate(requests):
        if capacity is None:
            limiters[i].add("capacity_unknown")
        if request is not None and request != (None, None) and sum(request) < len(texts[i]):
            limiters[i].add("explicit_view")
    receipt: Dict[str, Any] = {"fit": whole, "measurements": 1, "boundary_tokens": boundary, "capacity_tokens": capacity,
                               "reserve_tokens": reserve, "results": count, "partial": []}
    if boundary is None:
        # Unknown capacity does not cancel the form the caller explicitly chose.
        # Preserve unrequested/full outputs, and retain exact sources for named views.
        for index, request in enumerate(requests):
            if request is None or request == (None, None):
                continue
            head, tail = split_allowance(sum(request), request, len(texts[index]))
            if head + tail >= len(texts[index]):
                continue
            source = persist_result_source(drive_root, task_id, rows[index]["tool_call_id"], full[index])
            delivered, view = render_result_view(texts[index], head_chars=head, tail_chars=tail,
                tool_name=names[index], source_ref=source, facts=facts[index], delivery=diagnostic(index))
            if view is not None:
                rows[index]["result"] = delivered
                _mark_partial(rows[index], source, {**view, "requested": True})
                receipt["partial"].append({"tool_call_id": rows[index]["tool_call_id"],
                    "shown_ranges": view["shown_ranges"], "source_status": rows[index]["result_source_status"]})
        final = measure([row["result"] for row in rows]) if receipt["partial"] else whole
        return rows, {**receipt, "status": "capacity_unknown", "fit": final,
                      "measurements": 2 if receipt["partial"] else 1}

    def envelope(index: int, source: Mapping[str, Any]) -> str:
        return render_result_view(texts[index], head_chars=0, tail_chars=0, tool_name=names[index],
                                  source_ref=source, facts=facts[index], delivery=diagnostic(index))[0]

    def form(index: int, chars: int) -> str:  # a body at ``chars``, on its stand-in address
        head, tail = split_allowance(chars, requests[index], len(texts[index]))
        if head + tail >= len(texts[index]):
            return full[index]
        return render_result_view(texts[index], head_chars=head, tail_chars=tail, tool_name=names[index],
                                  source_ref=standins[index], facts=facts[index], delivery=diagnostic(index))[0]

    standins = [_standin_source_ref(row["tool_call_id"], text, digest) for row, text, digest in zip(rows, texts, digests)]
    envelopes = [envelope(index, standins[index]) for index in range(count)]
    minimum = measure(envelopes)
    p_min, p_whole = _tokens(minimum), _tokens(whole)
    # Stand-ins size allocation only; minimum_view_unfit is measured below
    # with actual source outcomes (a failed persistence has a shorter envelope).
    free = boundary - reserve - p_min
    room = None if capacity is None else capacity - reserve - p_min  # what requested forms may use
    unsolicited_allowance = max(0, min(free, eighth))
    # Body chars beyond each minimum envelope.
    body_chars = [max(0, len(text) - len(envelopes[index])) for index, text in enumerate(texts)]
    # What each call asked to see: the whole produced form, or a head+tail view.
    wants = {i: (len(texts[i]) if requests[i] == (None, None) else min(len(texts[i]), requests[i][0] + requests[i][1]))
             for i in range(count) if requests[i] is not None}
    receipt.update(measurements=2, minimum_tokens=p_min, whole_tokens=p_whole, free_tokens=free,
                   requested_room_tokens=room, unsolicited_allowance_tokens=unsolicited_allowance)
    whole_fits = (room is None or p_whole + reserve < capacity) and all(wants[i] >= len(texts[i]) for i in wants)
    complete = "complete" if room is not None else "capacity_unknown"
    if whole_fits and p_whole + reserve <= boundary and p_whole - p_min <= unsolicited_allowance:
        return rows, {**receipt, "status": complete}  # every body together inside the unasked share
    # Every chosen form whole beside the minimum unasked envelopes, as ONE request: the
    # measured cost of what the calls chose on THIS transcript, and so of the unasked rest.
    chosen_contents = [form(i, wants[i]) if i in wants else envelopes[i] for i in range(count)]
    chosen = minimum if not wants else whole if chosen_contents == full else measure(chosen_contents)
    p_chosen = _tokens(chosen)
    requested_capacity = capacity
    body_costs = {}
    for i in wants:
        empty = list(envelopes)
        empty[i] = ""
        baseline = measure(empty)
        one = list(empty)
        one[i] = chosen_contents[i]
        body_costs[i] = max(0, _tokens(measure(one)) - _tokens(baseline))
    protected = {i for i in wants if capacity is None or body_costs[i] <= capacity - reserve}
    for i in wants.keys() - protected:
        facts[i] = {**facts[i], "cut_reason": "chosen_body_exceeds_usable_input"}
    receipt.update(measurements=2 + (chosen is not minimum and chosen is not whole) + 2 * len(wants),
                   chosen_tokens=p_chosen, chosen_body_tokens={rows[i]["tool_call_id"]: cost for i, cost in body_costs.items()})
    chosen_chars = sum(max(0, len(chosen_contents[i]) - len(envelopes[i])) for i in wants)
    requested_rate = max(p_chosen - p_min, 1) / max(chosen_chars, 1)
    # Initial estimate of unasked cost: price omitted requested text at its shown
    # rate. Different escaping in that omitted text can skew this estimate; the
    # corrective pass below uses the measured request without unasked bodies.
    unsolicited_tokens = max(0.0, p_whole - p_chosen - requested_rate * sum(
        len(full[i]) - len(chosen_contents[i]) for i in wants if chosen_contents[i] is not full[i]))
    unsolicited_rate = max(unsolicited_tokens, 1) / max(sum(body_chars[i] for i in range(count) if i not in wants), 1)
    # Unsolicited bodies stay inside their eighth AND inside the target room requested forms left.
    if (whole_fits and unsolicited_tokens <= unsolicited_allowance
            and (not unsolicited_tokens or p_whole + reserve <= boundary)):
        return rows, {**receipt, "status": complete}

    allocation = [wants.get(i, 0) if i in protected else 0 for i in range(count)]
    if room is None or p_chosen - p_min <= room:  # unknown window, or every chosen form fits it whole
        allocation = [wants.get(index, 0) for index in range(count)]
        requested_tokens = float(p_chosen - p_min)
    else:  # the window shares out among the chosen forms at their own measured rate
        for index, chars in _water_fill({i: want for i, want in wants.items() if i not in protected},
                int(max(0, room - sum(body_costs[i] for i in protected)) / requested_rate)).items():
            allocation[index] = chars
        requested_tokens = sum(allocation[i] for i in wants) * requested_rate
    left = min(free, float("inf") if room is None else room) - requested_tokens
    pool = int(max(0.0, min(left, float(unsolicited_allowance))) / unsolicited_rate)
    for index, chars in _water_fill({i: body_chars[i] for i in range(count) if requests[i] is None}, pool).items():
        allocation[index] = chars
    # Explain the bound that actually allocated this batch, not merely which
    # window was selected. A known window can leave ample room while the target
    # or the unsolicited eighth limits a view to nothing.
    unasked_bounds = {"unsolicited_share": float(eighth)}
    if target is not None:
        unasked_bounds["target_pressure"] = target - reserve - p_min - requested_tokens
    if capacity is not None:
        unasked_bounds["physical_pressure"] = capacity - reserve - p_min - requested_tokens
    binding = min(unasked_bounds.values())
    for i, request in enumerate(requests):
        if request is None:
            limiters[i].update(name for name, value in unasked_bounds.items() if value == binding)
        elif allocation[i] < wants[i]:
            limiters[i].add("physical_pressure")

    sources: Dict[int, Dict[str, Any]] = {}

    def render(index: int, allowance: Optional[int] = None, *, publish: bool = True) -> str:
        head, tail = split_allowance(allocation[index] if allowance is None else allowance,
                                     requests[index], len(texts[index]))
        if head + tail >= len(texts[index]):
            if publish:
                rows[index]["result"] = full[index]
                for key in ("result_partial", "result_source_ref", "result_source_status", "result_source_view"):
                    rows[index].pop(key, None)
            return full[index]
        if index not in sources:  # persisted once, before any partial view of it is handed out
            sources[index] = persist_result_source(drive_root, task_id, rows[index]["tool_call_id"], full[index])
        delivered, view = render_result_view(texts[index], head_chars=head, tail_chars=tail, tool_name=names[index],
                                             source_ref=sources[index], facts=facts[index], delivery=diagnostic(index))
        if view is None:  # the anti-growth floor made it whole after all
            if publish:
                rows[index]["result"] = full[index]
            return full[index]
        if publish:
            rows[index]["result"] = delivered
            _mark_partial(rows[index], sources[index], {**view, "requested": requests[index] is not None})
        return delivered

    attempt = 0
    while True:
        contents = [render(index) for index in range(count)]
        final = measure(contents)
        receipt["measurements"] += 1
        without_unsolicited = measure([render(i, 0, publish=False) if requests[i] is None else contents[i]
                                       for i in range(count)])
        receipt["measurements"] += 1
        body_tokens = max(0, _tokens(final) - _tokens(without_unsolicited))
        receipt["unsolicited_body_tokens"] = body_tokens
        # Only the window shrinks a requested form; the target binds unsolicited bodies alone.
        capacity_over = _tokens(final) + reserve - requested_capacity if capacity is not None else 0
        economy_over = max(body_tokens - unsolicited_allowance,
                           _tokens(final) + reserve - boundary if body_tokens else 0)
        over = max(capacity_over, economy_over)
        # An unasked overshoot cannot take space from chosen forms that fit
        # beside the minimum unasked envelopes. Both eligibility and cost use
        # that already measured baseline, including real source locators.
        shrink_chosen = capacity is not None and _tokens(without_unsolicited) + reserve > requested_capacity
        shrinking = [i for i in range(count) if (shrink_chosen and i not in protected) or requests[i] is None]
        shown_total = sum(min(allocation[i], len(texts[i])) for i in shrinking)
        if capacity_over > 0 and shown_total <= 0 and not any("minimum_view_unfit" in causes for causes in limiters):
            actual_minimum = measure([render(i, 0, publish=False) for i in range(count)])
            receipt["measurements"] += 1
            if _tokens(actual_minimum) + reserve > capacity:
                for causes in limiters:
                    causes.add("minimum_view_unfit")
                continue  # include the new diagnostic bytes in the delivered measurement
        if over <= 0 or shown_total <= 0 or attempt > _MAX_REMEASURES:
            break
        for i in shrinking:
            if capacity_over > 0:
                limiters[i].add("physical_pressure")
            if requests[i] is None and economy_over > 0:
                if body_tokens > unsolicited_allowance:
                    limiters[i].update(name for name, value in unasked_bounds.items() if value == binding)
                if _tokens(final) + reserve > boundary:
                    limiters[i].add("target_pressure" if target == boundary else "physical_pressure")
        attempt += 1
        if attempt > _MAX_REMEASURES:
            for i in shrinking:  # last resort: the minimum view of every body still over
                allocation[i] = 0
        else:  # chars at the shrinking bodies' own measured cost on this candidate
            spent = _tokens(final) - (p_min if shrink_chosen else _tokens(without_unsolicited))
            cut = min(shown_total, int(over * shown_total / max(spent, 1)) + 1)
            for i in shrinking:
                allocation[i] = int(allocation[i] * max(0.0, 1.0 - (cut * 1.1) / shown_total))
    status = ("capacity_unknown" if capacity is None
              else "projected" if _tokens(final) + reserve <= capacity else "minimum_view_unfit")
    receipt["recovery_required"] = capacity is not None and _tokens(final) + reserve > capacity
    for i in wants:
        if allocation[i] < wants[i] and rows[i].get("result_partial"):
            rows[i]["result_source_view"]["cut_reason"] = "chosen_body_exceeds_usable_input"
    receipt["target_excess_tokens"] = max(0, _tokens(final) + reserve - target) if target else None
    receipt["partial"] = [{"tool_call_id": rows[i]["tool_call_id"], "tool": names[i],
                           "shown_ranges": rows[i]["result_source_view"]["shown_ranges"],
                           "complete_chars": len(texts[i]), "source_status": rows[i]["result_source_status"]}
                          for i in range(count) if rows[i].get("result_partial")]
    return rows, {**receipt, "status": status, "fit": final}
