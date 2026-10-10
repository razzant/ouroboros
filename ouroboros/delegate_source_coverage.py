"""Custody helpers for exact source coverage of oversized work orders.

Snapshot adoption copies unresolved source bindings and their verified ranges
into ``work_order_source_request.inherited_sources`` beside the new request.
These are flat, independent copies: no predecessor history read is needed to
verify or answer a successor's sources. Each source keeps its original brief
fingerprint; the successor's own fingerprint still identifies its new request.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Tuple, Optional


def _strict_int(value: Any) -> int | None:
    return value if type(value) is int else None


def _merge_verified_source_range(entry: Any, start: Any, end: Any) -> None:
    """Union one already validated source interval into the custody memo."""

    left, right = _strict_int(start), _strict_int(end)
    if left is None or right is None or left < 0 or right <= left:
        return
    request = getattr(entry, "work_order_source_request", {}) or {}
    if str(getattr(entry, "work_order_coverage", "") or "") == "partial":
        total = _strict_int(request.get("complete_chars"))
        if total is None or right > total:
            return
    ranges = [tuple(pair) for pair in (getattr(entry, "verified_source_ranges", []) or [])]
    ranges.append((left, right))
    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for current_left, current_right in ranges:
        if merged and current_left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current_right))
        else:
            merged.append((current_left, current_right))
    entry.verified_source_ranges = merged


def _source_entries(entry: Any) -> List[Any]:
    request = getattr(entry, "work_order_source_request", {}) or {}
    return [entry, *(SimpleNamespace(**binding) for binding in request.get("inherited_sources", []))]


def inherit_source_obligations(previous: Any, binding: Dict[str, Any]) -> None:
    """Copy unresolved custody at snapshot adoption; add to the new brief, never replace it."""
    inherited = []
    for entry in _source_entries(previous):
        if _source_verification(entry)["can_authorize"]:
            continue
        request = dict(entry.work_order_source_request)
        request.pop("inherited_sources", None)
        inherited.append({
            "work_order_source_request": request,
            "work_order_fingerprint": entry.work_order_fingerprint,
            "work_order_coverage": entry.work_order_coverage,
            "verified_source_ranges": entry.verified_source_ranges,
        })
    if inherited:
        request = binding["work_order_source_request"]
        binding["work_order_source_request"] = copy.deepcopy({
            **request, "inherited_sources": [*request.get("inherited_sources", []), *inherited]})


def source_request_for_response(entry: Any, response: Any) -> Dict[str, Any]:
    """Select the bound source by its digest AND selector before exact byte validation."""
    if isinstance(response, Mapping):
        for source in _source_entries(entry):
            request = source.work_order_source_request
            if (request.get("complete_sha256") == response.get("complete_sha256")
                    and request.get("source") == response.get("source")):
                return request
    return entry.work_order_source_request  # The ordinary validator names the mismatch.


def work_order_source_verification(custody: Any) -> Dict[str, Any]:
    """The existing completeness gate covers every source carried by this patch."""
    sources = [_source_verification(entry) for entry in _source_entries(custody)]
    required = [source for source in sources if source["status"] != "not_required"]
    if not required:
        return sources[0]
    incomplete = next((source for source in required if not source["can_authorize"]), None)
    result = dict(incomplete or required[0])
    if len(required) > 1:
        result["sources"] = required
    return result


def _source_verification(custody: Any) -> Dict[str, Any]:
    """Return the durable, typed completeness fact for ONE oversized brief."""

    request = dict(getattr(custody, "work_order_source_request", {}) or {})
    request.pop("inherited_sources", None)
    if str(getattr(custody, "work_order_coverage", "") or "") != "partial" and not request:
        return {"status": "not_required", "can_authorize": True}
    total = _strict_int(request.get("complete_chars")) if isinstance(request, Mapping) else None
    ranges: List[Dict[str, int]] = []
    for raw_start, raw_end in (getattr(custody, "verified_source_ranges", []) or []):
        start, end = _strict_int(raw_start), _strict_int(raw_end)
        if total is not None and start is not None and end is not None and 0 <= start < end <= total:
            ranges.append({"start_char": start, "end_char": end})
    ranges.sort(key=lambda row: (row["start_char"], row["end_char"]))
    cursor = 0
    complete = total is not None and total >= 0
    for row in ranges:
        if row["start_char"] > cursor:
            complete = False
            break
        cursor = max(cursor, row["end_char"])
    complete = complete and cursor >= (total or 0)
    if complete:
        return {
            "status": "complete",
            "can_authorize": True,
            "complete_chars": total,
            "complete_sha256": str(request.get("complete_sha256") or ""),
            "verified_ranges": ranges,
        }
    return {
        "status": "cannot_verify",
        "reason": "source_incomplete",
        "can_authorize": False,
        "complete_chars": total,
        "complete_sha256": str(request.get("complete_sha256") or ""),
        "source": request.get("source") if isinstance(request.get("source"), dict) else {},
        "verified_ranges": ranges,
    }


def record_source_range_verified(
    drive_root: Any, custody: Any, *, start_char: int, end_char: int,
    complete_sha256: str, source: Dict[str, Any], text_sha256: str = "",
    text_chars: int = 0,
) -> bool:
    """Persist one bounded source interval after exact host validation."""

    if not _source_range_receipt_valid(
        custody,
        start_char=start_char,
        end_char=end_char,
        complete_sha256=complete_sha256,
        source=source,
        text_sha256=text_sha256,
        text_chars=text_chars,
    ):
        return False
    left, right = int(start_char), int(end_char)
    expected_sha = str(complete_sha256 or "")
    chars = int(text_chars)
    digest = str(text_sha256 or "")
    # Import lazily: custody re-exports this helper after defining its row machinery.
    from ouroboros import delegate_custody as custody_module

    payload = {
        "run_id": custody.run_id,
        "task_id": custody.task_id,
        "start_char": left,
        "end_char": right,
        "complete_sha256": expected_sha,
        "source": dict(source),
        "text_sha256": digest,
        "text_chars": chars,
    }
    landed = custody_module.emit(drive_root, custody_module.SOURCE_RANGE_VERIFIED, payload)
    if landed:
        apply_source_range_receipt(custody, payload)
    return landed


def apply_source_range_receipt(entry: Any, row: Mapping[str, Any]) -> None:
    """Merge a receipt only into the matching source, in live custody and replay."""
    facts = {key: row.get(key) for key in (
        "start_char", "end_char", "complete_sha256", "source", "text_sha256", "text_chars")}
    if _source_range_matches(entry, **facts):
        _merge_verified_source_range(entry, facts["start_char"], facts["end_char"])
    request = entry.work_order_source_request
    inherited = []
    for binding in request.get("inherited_sources", []):
        source = SimpleNamespace(**binding)
        if _source_range_matches(source, **facts):
            _merge_verified_source_range(source, facts["start_char"], facts["end_char"])
        inherited.append({**binding, "verified_source_ranges": source.verified_source_ranges})
    if inherited:
        entry.work_order_source_request = {**request, "inherited_sources": inherited}


def _source_delivery_record(interaction_id: str, source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "interaction_id": str(interaction_id or ""),
        "complete_sha256": str(source.get("complete_sha256") or ""),
        "source": source.get("source") or {},
        "start_char": _strict_int(source.get("start_char")),
        "end_char": _strict_int(source.get("end_char")),
        "text_sha256": str(source.get("text_sha256") or ""),
        "text_chars": _strict_int(source.get("text_chars")),
    }


def record_source_delivery_confirmed(
    drive_root: Any, custody: Any, *, interaction_id: str,
    verified_source: Mapping[str, Any],
) -> bool:
    """Persist positive delivery evidence before a source receipt is retried."""
    from ouroboros import delegate_custody as custody_module

    record = _source_delivery_record(interaction_id, verified_source)
    if not record["interaction_id"] or not _source_range_receipt_valid(
            custody, start_char=record["start_char"], end_char=record["end_char"],
            complete_sha256=record["complete_sha256"], source=record["source"],
            text_sha256=record["text_sha256"], text_chars=record["text_chars"]):
        return False
    landed = custody_module.emit(
        drive_root, custody_module.SOURCE_RANGE_DELIVERY_CONFIRMED,
        {"run_id": custody.run_id, "task_id": custody.task_id, **record},
    )
    if landed:
        confirmations = getattr(custody, "_source_delivery_confirmations", [])
        if record not in confirmations:
            confirmations.append(record)
        custody._source_delivery_confirmations = confirmations
    return landed


def source_delivery_confirmed(
    custody: Any, interaction_id: str, verified_source: Mapping[str, Any],
) -> bool:
    """True only for a durable confirmation of this exact interaction and bytes."""
    expected = {
        **_source_delivery_record(interaction_id, verified_source),
    }
    return expected in getattr(custody, "_source_delivery_confirmations", [])


def merge_source_delivery_confirmations(entry: Any, previous: Any) -> None:
    confirmations = list(getattr(entry, "_source_delivery_confirmations", []) or [])
    for record in getattr(previous, "_source_delivery_confirmations", []) or []:
        if record not in confirmations:
            confirmations.append(dict(record))
    entry._source_delivery_confirmations = confirmations


def apply_source_delivery_confirmation(entry: Any, row: Mapping[str, Any]) -> None:
    record = _source_delivery_record(str(row.get("interaction_id") or ""), row)
    if record["interaction_id"] and _source_range_receipt_valid(
            entry, start_char=record["start_char"], end_char=record["end_char"],
            complete_sha256=record["complete_sha256"], source=record["source"],
            text_sha256=record["text_sha256"], text_chars=record["text_chars"]):
        confirmations = getattr(entry, "_source_delivery_confirmations", [])
        if record not in confirmations:
            confirmations.append(record)
        entry._source_delivery_confirmations = confirmations


def _source_range_receipt_valid(
    custody: Any, **facts: Any,
) -> bool:
    return any(_source_range_matches(entry, **facts) for entry in _source_entries(custody))


def _source_range_matches(
    custody: Any, *, start_char: Any, end_char: Any, complete_sha256: Any,
    source: Any, text_sha256: Any, text_chars: Any,
) -> bool:
    """Validate a durable receipt again during replay, not only before emit."""

    request = getattr(custody, "work_order_source_request", {}) or {}
    expected_sha = str(request.get("complete_sha256") or "")
    left, right = _strict_int(start_char), _strict_int(end_char)
    total = _strict_int(request.get("complete_chars"))
    chars = _strict_int(text_chars)
    digest = str(text_sha256 or "")
    return bool(
        request
        and getattr(custody, "work_order_coverage", "") == "partial"
        and str(complete_sha256 or "") == expected_sha
        and str(getattr(custody, "work_order_fingerprint", "") or "") == expected_sha
        and source == request.get("source")
        and left is not None and right is not None and total is not None and chars is not None
        and 0 <= left < right <= total and chars == right - left
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    )


def record_started_custody(
    drive: Any, run_id: str, ctx: Any, route: Any, authority: Any, *,
    key: str, access: str, root: str, seconds: int, invocation_id: str,
    project_id: str, project_owned: bool, project_persistent: bool,
    selected_subagent_id: str,
    config_fingerprint: str, work_order_fingerprint: str, work_order_coverage: str,
    work_order_source_request: Dict[str, Any], authority_fingerprint: str,
    snapshot_id: str, execution_binding_fingerprint: str, target_root: str,
    baseline_sha: str, authority_source: str,
    resource_ref: Dict[str, Any], capture_mode: str, processing: Mapping[str, Any] | None = None,
    continuation_of: str = "", capture_id: str = "", snapshot_task_id: str = "", max_seconds_basis: str = "",
    row_effort: Optional[str] = None,
) -> bool:
    """Write the one STARTED custody row, including the source binding.

    ``continuation_of`` names the prior run this start explicitly continues
    (``delegate_continuation``); ``capture_id``/``snapshot_task_id`` bind a run
    that continues IN that run's snapshot. They ride the STARTED row so the
    lineage and the hand-over replay with every other start fact. ``max_seconds_basis``
    records HOW ``seconds`` was decided (``delegate_registration_policy.CAP_BASIS_*``)
    beside the cap itself, so a later expiry can be told apart from the nanny's
    own deadline or lifetime.
    """

    from ouroboros import delegate_custody as custody_module

    metadata = getattr(ctx, "task_metadata", {}) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    entry = custody_module.RunCustody(
        run_id=run_id,
        task_id=str(getattr(ctx, "task_id", "") or ""),
        route_id=route.route_id,
        model=route.model,
        profile_id=route.profile_id,
        effort=route.effort,
        processing_preference=(processing or {}).get("requested"),
        project_id=project_id,
        project_owned=project_owned,
        project_persistent=project_persistent,
        root_task_id=str(metadata.get("root_task_id") or ""),
        parent_task_id=str(metadata.get("parent_task_id") or ""),
        ledger_root=str(drive),
        idempotency_key=key,
        invocation_id=invocation_id,
        selected_subagent_id=selected_subagent_id,
        config_fingerprint=config_fingerprint,
        work_order_fingerprint=work_order_fingerprint,
        work_order_coverage=work_order_coverage,
        work_order_source_request=work_order_source_request,
        authority_fingerprint=authority_fingerprint,
        snapshot_id=snapshot_id,
        execution_root=(root if snapshot_id or (resource_ref.get("workspace_kind") == "directory"
                                               and resource_ref.get("strategy") == "direct") else ""),
        execution_binding_fingerprint=execution_binding_fingerprint,
        baseline_sha=baseline_sha,
        target_root=target_root,
        authority_source=authority_source,
        resource_ref=resource_ref,
        access=access,
        mode=authority.mode,
        isolation=authority.isolation,
        delegated=authority.delegated,
        continuation_of=str(continuation_of or ""),
        capture_id=str(capture_id or ""),
        snapshot_task_id=str(snapshot_task_id or ""),
    )
    return custody_module.record_started(
        drive,
        entry,
        shape={
            "effort": route.effort, "access": access, "mode": authority.mode,
            **({"row_effort": str(row_effort)} if row_effort is not None else {}),
            "isolation": authority.isolation, "delegated": authority.delegated,
            "root": root, "max_seconds": seconds, "max_seconds_basis": str(max_seconds_basis or ""),
            "capture_mode": capture_mode,
        },
    )


__all__ = [
    "_merge_verified_source_range",
    "add_terminal_source_verification",
    "apply_source_range_receipt",
    "inherit_source_obligations",
    "prepare_work_order_start_binding",
    "record_started_custody",
    "record_source_range_verified",
    "record_source_delivery_confirmed",
    "source_delivery_confirmed",
    "merge_source_delivery_confirmations",
    "apply_source_delivery_confirmation",
    "source_apply_refusal",
    "source_request_for_response",
    "work_order_source_verification",
]


def prepare_work_order_start_binding(
    ctx: Any, drive: Any, retry_token: str, canonical_fingerprint: str,
    prompt: str, supplied_request: Any,
) -> Dict[str, Any]:
    """Bind source custody metadata once, including replay of a pending request."""

    request = dict(supplied_request) if isinstance(supplied_request, dict) else {}
    recovering = bool(retry_token)
    if recovering and not request:
        from ouroboros import delegate_custody as custody_module

        recorded = custody_module.invocation_record(drive, retry_token) or {}
        stored = recorded.get("work_order_source_request")
        if isinstance(stored, dict) and stored:
            request = dict(stored)
    from ouroboros.subagent_work_order import start_binding_fingerprints

    prompt_fingerprint, authority_fingerprint = start_binding_fingerprints(ctx, prompt)
    durable_fingerprint = str(request.get("complete_sha256") or "").strip()
    selected_fingerprint = str(canonical_fingerprint or "").strip()
    return {
        "request": request,
        "coverage": "partial" if str(request.get("coverage") or "") == "partial" else "",
        "prompt_fingerprint": prompt_fingerprint,
        "authority_fingerprint": authority_fingerprint,
        "fingerprint": selected_fingerprint or durable_fingerprint or prompt_fingerprint,
        "recovering": recovering,
    }


def source_apply_refusal(entry: Any, run_id: str) -> str:
    """Refuse patch application until a partial brief has full source coverage."""

    from ouroboros import delegate_custody as custody_module

    if custody_module.work_order_source_verification(entry).get("status") != "cannot_verify":
        return ""
    return (
        f"⚠️ INTEGRATE_DELEGATED_SOURCE_UNRESOLVED: run {run_id}'s external "
        "work order was only partially delivered and its canonical source "
        "ranges are not fully verified. The patch was NOT applied and the "
        "snapshot is preserved. Answer the existing source question(s) with "
        "typed source_response ranges, wait for a complete verification "
        "receipt, or call integrate_delegated_patch(decision='reject') to "
        "discard it."
    )


def add_terminal_source_verification(payload: Dict[str, Any], entry: Any) -> None:
    """Attach the source completeness fact to a terminal host payload."""

    from ouroboros import delegate_custody as custody_module

    verification = custody_module.work_order_source_verification(entry)
    payload["work_order_verification"] = verification
    if verification.get("status") == "cannot_verify":
        payload["acceptance_status"] = "cannot_verify"
        payload["acceptance_note"] = (
            "The external work order was only partially delivered. This terminal "
            "run is evidence, not a PASS: obtain and verify the missing canonical "
            "source ranges before accepting or applying its work."
        )
