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
    Main's first show. With B the smaller known positive of the owner target
    and the route capacity, R the reply reserve and Pmin the measured minimum
    request (every envelope with status and address, no body), unsolicited
    bodies share ``max(0, min(B - R - Pmin, ceil(B / RECLAIM_LOW_WATER_DIVISOR)))``
    -- one allowance per batch, the reserve subtracted once -- while a form
    the call asked for (a head/tail view, a page, a window) may use the free
    room ``B - R - Pmin``. The complete candidate is re-measured before it is
    handed back. An unknown B is disclosed and the batch is delivered whole:
    no capacity is invented. The callback's ``accepted`` flag is never read as
    fit here.
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
RECOVER_PAGEABLE = "Read the exact source above, or page this tool (offset/limit) for the omitted range."
RECOVER_OTHER = "Do not rerun this tool to recover omitted output. Read the exact source above."

# Re-measurements of the complete candidate after the first allocation. Each one
# shrinks every shown body by the measured overshoot; the last resort is the
# minimum request itself (status ``minimum_view_unfit`` when even that is over).
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


def typed_result_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Status, code and process facts the producer published; nothing read from the text."""
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
    return facts


def prepare_producer_sources(results: Sequence[Mapping[str, Any]], drive_root: Any, task_id: str) -> list:
    """Retain clean producer text before measuring its model-visible locator."""
    rows = [dict(row) for row in results]
    for row in rows:
        producer_text = getattr(row.get("tool_result"), "producer_text", None)
        if producer_text is not None:
            row["producer_source_ref"] = (persist_result_source(drive_root, task_id,
                str(row["tool_call_id"]) + ".producer", producer_text) if drive_root is not None else {})
    return rows


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
    view: Dict[str, Any] = {
        "complete_chars": total,
        "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "shown_ranges": shown,
        "omitted_ranges": [omitted],
        "range_basis": RESULT_VIEW_BASIS,
        "facts": dict(facts or {}),
    }
    if ref:
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
        lines.append(RECOVER_PAGEABLE if tool_name in PAGEABLE_TOOL_RESULTS else RECOVER_OTHER)
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
    receipt: Dict[str, Any] = {"fit": whole, "measurements": 1, "boundary_tokens": boundary, "reserve_tokens": reserve,
                               "results": count, "partial": []}
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
                tool_name=names[index], source_ref=source, facts=facts[index])
            if view is not None:
                rows[index]["result"] = delivered
                _mark_partial(rows[index], source, {**view, "requested": True})
                receipt["partial"].append({"tool_call_id": rows[index]["tool_call_id"],
                    "shown_ranges": view["shown_ranges"], "source_status": rows[index]["result_source_status"]})
        return rows, {**receipt, "status": "capacity_unknown"}

    def envelope(index: int, source: Mapping[str, Any]) -> str:
        return render_result_view(texts[index], head_chars=0, tail_chars=0, tool_name=names[index],
                                  source_ref=source, facts=facts[index])[0]

    standins = [_standin_source_ref(row["tool_call_id"], text, digest) for row, text, digest in zip(rows, texts, digests)]
    minimum = measure([envelope(index, standins[index]) for index in range(count)])
    p_min, p_whole = _tokens(minimum), _tokens(whole)
    free = boundary - reserve - p_min
    unsolicited_allowance = max(0, min(free, eighth))
    # Body chars beyond each minimum envelope, and the measured cost of one of them
    # on THIS transcript (the two measurements differ exactly by the bodies).
    body_chars = [max(0, len(text) - len(envelope(index, standins[index]))) for index, text in enumerate(texts)]
    total_body = sum(body_chars)
    tokens_per_char = max(p_whole - p_min, 1) / max(total_body, 1)
    receipt.update(measurements=2, minimum_tokens=p_min, whole_tokens=p_whole, free_tokens=free,
                   unsolicited_allowance_tokens=unsolicited_allowance)
    # What each call asked to see: the whole produced form, or a head+tail view.
    wants = {i: (len(texts[i]) if requests[i] == (None, None) else min(len(texts[i]), requests[i][0] + requests[i][1]))
             for i in range(count) if requests[i] is not None}
    unsolicited_tokens = sum(body_chars[i] for i in range(count) if requests[i] is None) * tokens_per_char
    if (p_whole + reserve <= boundary and unsolicited_tokens <= unsolicited_allowance
            and all(wants[i] >= len(texts[i]) for i in wants)):
        return rows, {**receipt, "status": "complete"}

    def to_chars(tokens: float) -> int:
        return max(0, int(tokens / tokens_per_char))

    allocation = [0] * count
    if free > 0:
        for index, chars in _water_fill(wants, to_chars(free)).items():
            allocation[index] = chars
        remaining = free - sum(allocation[i] for i in wants) * tokens_per_char
        pool = to_chars(max(0.0, min(remaining, float(unsolicited_allowance))))
        for index, chars in _water_fill({i: body_chars[i] for i in range(count) if requests[i] is None}, pool).items():
            allocation[index] = chars

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
                                             source_ref=sources[index], facts=facts[index])
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
        capacity_over = _tokens(final) + reserve - boundary
        over = max(capacity_over, body_tokens - unsolicited_allowance)
        shrinking = [i for i in range(count) if capacity_over > 0 or requests[i] is None]
        shown_total = sum(min(allocation[i], len(texts[i])) for i in shrinking)
        if over <= 0 or shown_total <= 0 or attempt > _MAX_REMEASURES:
            break
        attempt += 1
        if attempt > _MAX_REMEASURES:
            allocation = [0] * count  # last resort: the minimum request itself
        else:
            cut = min(shown_total, to_chars(over) + 1)
            for i in shrinking:
                allocation[i] = int(allocation[i] * max(0.0, 1.0 - (cut * 1.1) / shown_total))
    status = "projected" if _tokens(final) + reserve <= boundary else "minimum_view_unfit"
    receipt["partial"] = [{"tool_call_id": rows[i]["tool_call_id"], "tool": names[i],
                           "shown_ranges": rows[i]["result_source_view"]["shown_ranges"],
                           "complete_chars": len(texts[i]), "source_status": rows[i]["result_source_status"]}
                          for i in range(count) if rows[i].get("result_partial")]
    return rows, {**receipt, "status": status, "fit": final}
