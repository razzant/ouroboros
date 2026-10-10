"""Staged output for delegated runs, and the durable receipt that it was READ.

Terminal payloads too large for the tool budget are written whole to the task drive.
The inline payload then carries bounded ``*_preview`` heads of the bulk fields while
the file carries the full bytes. D7's acknowledgement records which authorized reader
received every character of which content: it binds the reader, the content sha256 and
delivered coverage of every character to EOF. ``tools.delegate`` re-exports this
delivery surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from ouroboros import delegate_custody as custody
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

# Where the full terminal detail is staged under the task's own drive, to be read back
# with the ordinary read_file contract.
_ARTIFACT_SUBDIR = "delegated_runs"

# Room inside a self-bounding delivery budget for the JSON scaffold and the
# delivery block itself — shared by every producer that measures itself against
# `tool_result_limit` (the terminal payload, the waiting_on_user payload).
_PAYLOAD_ENVELOPE_HEADROOM = 2_000


def _safe_run_filename(run_id: str) -> str:
    """A filesystem-safe name for a daemon-supplied run id (never trusted verbatim)."""
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(run_id or "")).strip(".")[:80]
    return cleaned or hashlib.sha256(str(run_id or "").encode("utf-8", "replace")).hexdigest()[:16]


def _stage_full_output(ctx: ToolContext, run_id: str, text: str,
                       suffix: str = "") -> Optional[Dict[str, Any]]:
    """Write the WHOLE terminal detail to the task's own drive and describe it.

    Host-owned by construction: the host writes it, and every profile that can call a
    nanny verb has task_drive as READ-only or better, so the artifact is reachable
    through the ordinary ``read_file`` contract (which already carries a stable
    ``start_line`` cursor) instead of a parallel artifact system.

    ``suffix`` distinguishes a SECONDARY per-run artifact (e.g. ``.interactions``
    for a spilled question set) from the terminal result: the D7 read-receipt
    matcher keys on the exact ``<run>.json`` name, so a suffixed artifact is
    deliberately outside the consumed-receipt contract — its receipt is the
    sha256/size pair returned here.
    """
    from ouroboros.tool_access import resource_root_path

    # The bytes DECLARED here and the bytes WRITTEN must be one object: the sha256 below
    # becomes the artifact's identity (`custody.output_sha`), and the read receipt measures
    # the file with `read_bytes`. A text write translates "\n" to os.linesep, so on Windows
    # the declared hash described a file that never existed on disk and the D7
    # acknowledgement could never be recorded for any staged output.
    staged_bytes = text.encode("utf-8", "replace")
    tmp = None
    try:
        base = resource_root_path(ctx, "task_drive") / _ARTIFACT_SUBDIR
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{_safe_run_filename(run_id)}{suffix}.json"
        # UNIQUE tmp per write (F15): two concurrent stagings of the same target
        # (worker restart racing its predecessor, two waits over one run) must
        # not interleave bytes through a shared scratch name — each writes its
        # own and the atomic rename decides.
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(staged_bytes)
        tmp.replace(target)
    except Exception:
        log.warning("Failed to stage delegated run output for %s", run_id, exc_info=True)
        return None
    finally:
        # R2-9a: a failed write must not leave its unique scratch file behind;
        # after a successful replace the name is already gone (missing_ok).
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "root": "task_drive",
        "path": f"{_ARTIFACT_SUBDIR}/{target.name}",
        "abs_path": str(target),
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "bytes": len(staged_bytes),
        "sha256": hashlib.sha256(staged_bytes).hexdigest(),
    }


# Proven-coverage ledger for staged artifacts: merged, sorted character intervals
# actually SERVED to the reader, keyed by path, content hash and successor identity. A re-wait
# re-stages the identical payload and must not void an honest reader's partial proof,
# while changed content honestly resets it. Process-local by design: the DURABLE fact
# is the acknowledgement row itself; a restarted worker re-proves coverage by
# re-reading, it never inherits an unproven claim.
_READ_COVERAGE: Dict[str, List[List[int]]] = {}
_READ_COVERAGE_MAX_KEYS = 128


def record_output_consumed(drive_root: Any, custody: _RunCustody, *,
                           artifact: str, byte_length: int, sha256: str,
                           chars: int, lines: int, reader_task_id: str = "") -> bool:
    """Record the D7 acknowledgement for one fully read staged artifact."""

    if not custody.output_complete:
        return False
    if custody.output_sha and str(sha256 or "") != custody.output_sha:
        return False
    reader = str(reader_task_id or custody.task_id)
    if output_consumed_by_reader(custody, reader):
        return True
    from ouroboros import delegate_custody as custody_module

    landed = custody_module.emit(drive_root, custody_module.OUTPUT_CONSUMED, {
        "run_id": custody.run_id,
        "task_id": custody.task_id,
        "reader_task_id": reader,
        "artifact": str(artifact or ""),
        "bytes": int(byte_length),
        "sha256": str(sha256 or ""),
        "chars": int(chars),
        "lines": int(lines),
    })
    if landed:
        custody.output_consumed = True
        custody.output_reader_receipts = tuple(
            pair for pair in custody.output_reader_receipts if pair[0] != reader
        ) + ((reader, str(sha256 or "")),)
    return landed


def output_consumed_by_reader(entry: _RunCustody, reader_task_id: str) -> bool:
    """A predecessor's receipt never proves delivery to its retry successor."""
    receipts = dict(entry.output_reader_receipts)
    reader = str(reader_task_id or "")
    if reader in receipts:
        return bool(entry.output_complete and receipts[reader] == entry.output_sha)
    # Compatibility for pre-receipt in-process entries; replay attributes old
    # OUTPUT_CONSUMED rows to their immutable starter explicitly.
    return bool(not receipts and reader == entry.task_id and entry.output_consumed)


def output_disposition(custody: _RunCustody) -> Dict[str, Any]:
    """Return the staged-output facts carried by terminal custody rows."""

    if not custody.output_artifact:
        return {}
    return {
        "staged_output": custody.output_artifact,
        "staged_output_complete": custody.output_complete,
        "staged_output_consumed": custody.output_consumed,
    }


def _covered_whole(key: str, start: int, end: int, total: int) -> bool:
    """Merge one DELIVERED char range [start, end) into the ledger; True when
    [0, total) is covered.

    CONTINUOUS coverage of what was actually DELIVERED, not a reached cursor and not
    source-file line ranges: a head plus a tail with a skipped middle must never read
    as "read to EOF", and neither may a window the delivery layer cut — the model
    received the prefix the truncator kept, nothing more. Ranges may arrive in any
    order; only an unbroken union over every character is full reading.
    """
    if total <= 0:
        return True
    if key not in _READ_COVERAGE and len(_READ_COVERAGE) >= _READ_COVERAGE_MAX_KEYS:
        _READ_COVERAGE.pop(next(iter(_READ_COVERAGE)))
    intervals = _READ_COVERAGE.setdefault(key, [])
    if start < end:
        intervals.append([start, end])
        intervals.sort()
        merged: List[List[int]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        _READ_COVERAGE[key] = merged
        intervals = merged
    return bool(intervals) and intervals[0][0] <= 0 and intervals[0][1] >= total


_DEFERRED_READ_DELIVERY: ContextVar[bool] = ContextVar("staged_read_delivery_deferred", default=False)


@contextmanager
def staged_read_delivery_scope():
    """Defer receipts only inside this executor invocation's pending projection."""
    token = _DEFERRED_READ_DELIVERY.set(True)
    try:
        yield
    finally:
        _DEFERRED_READ_DELIVERY.reset(token)


def acknowledge_staged_output_read(ctx: ToolContext, read_view: Dict[str, Any], rendered: str) -> None:
    """A direct reader delivers this exact window; a scoped executor credits later."""
    if not _DEFERRED_READ_DELIVERY.get():
        _acknowledge_staged_output(ctx, read_view, rendered, {"result": rendered})


def _delivered_source_ranges(read_view: Dict[str, Any], original: str,
                             projected: Dict[str, Any], content: str,
                             shown_ranges: Optional[List[List[int]]] = None) -> Optional[List[List[int]]]:
    """Intersect the host's delivered result ranges with this reader's file body."""
    from ouroboros.tool_result_delivery import RESULT_VIEW_BASIS, reader_text

    original = reader_text(original)
    if (read_view.get("range_basis") != RESULT_VIEW_BASIS
            or read_view.get("source_masked") is not False
            or read_view.get("complete_chars") != len(content)
            or read_view.get("complete_sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest()):
        return None
    start, end = read_view.get("source_start_char"), read_view.get("source_end_char")
    body_start, body_chars = read_view.get("body_start"), read_view.get("body_chars")
    if (not all(isinstance(n, int) for n in (start, end, body_start, body_chars))
            or not 0 <= start <= end <= len(content) or body_start < 0
            or body_chars != end - start
            or original[body_start:body_start + body_chars] != content[start:end]):
        return None
    if shown_ranges is not None:
        shown = shown_ranges
    elif projected.get("result_partial"):
        view = projected.get("result_source_view")
        if (not isinstance(view, dict) or view.get("range_basis") != RESULT_VIEW_BASIS
                or view.get("complete_chars") != len(original)
                or view.get("complete_sha256") != hashlib.sha256(original.encode("utf-8")).hexdigest()
                or not isinstance(view.get("shown_ranges"), list)):
            return None
        shown = view["shown_ranges"]
    elif reader_text(projected.get("result")) == original:
        shown = [[0, len(original)]]
    else:
        return None
    ranges = []
    for span in shown:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not all(isinstance(n, int) for n in span)
                or not 0 <= span[0] <= span[1] <= len(original)):
            return None
        lo, hi = max(span[0], body_start), min(span[1], body_start + body_chars)
        if lo < hi:
            ranges.append([start + lo - body_start, start + hi - body_start])
    return ranges


def acknowledge_staged_output_delivery(ctx: ToolContext, exec_result: Dict[str, Any],
                                       projected: Dict[str, Any], *,
                                       shown_ranges: Optional[List[List[int]]] = None) -> None:
    """Credit exactly the source ranges in this invocation's final Main projection.

    read_view comes from the reader's invocation-local ToolResult, never the
    shared last_read_view slot. An outer reader with its own projection may pass
    its known shown_ranges explicitly. Coordinates are Unicode characters after universal
    newline decoding; the durable receipt still names the original staged BYTES.
    Unknown/mismatched views credit nothing and never block the read. Whole reads
    have no fixed cap; a head+tail projection credits both pieces, never its gap.
    """
    from ouroboros.tools.tool_result import ToolResult

    typed = exec_result.get("tool_result")
    read_view = typed.meta.get("read_view") if isinstance(typed, ToolResult) else None
    _acknowledge_staged_output(ctx, read_view, exec_result.get("result", ""), projected, shown_ranges)


def _acknowledge_staged_output(ctx: ToolContext, read_view: Any, original: str,
                              projected: Dict[str, Any],
                              shown_ranges: Optional[List[List[int]]] = None) -> None:
    try:
        from ouroboros.tool_access import resource_root_path

        if not isinstance(read_view, dict) or read_view.get("opened_root") != "task_drive":
            return
        path = pathlib.Path(read_view["target"])
        artifact_dir = (resource_root_path(ctx, "task_drive") / _ARTIFACT_SUBDIR).resolve(strict=False)
        resolved = path.resolve(strict=False)
        if resolved.parent != artifact_dir:
            return
        raw = path.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        if read_view.get("source_revision") != source_sha:
            return  # A replacement is not the source this invocation opened.
        content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        ranges = _delivered_source_ranges(read_view, original, projected, content, shown_ranges)
        if ranges is None or (not ranges and content):
            return
        task_id = str(getattr(ctx, "task_id", "") or "")
        wanted = path.name
        drive = custody.custody_root(ctx)

        def _match(candidates: Any) -> Optional[_RunCustody]:
            return next((c for c in candidates
                         if _safe_run_filename(c.run_id) + ".json" == wanted), None)

        entry = _match(list(custody._CUSTODY.values())) or _match(custody.replay(drive).values())
        if entry is None:
            return
        successor = entry.task_id != task_id
        if successor:
            from ouroboros.delegate_shared import retry_result_status

            status, entry, predecessor = retry_result_status(ctx, drive, entry.run_id)
            if status != custody.OWNED or entry is None or not predecessor:
                return
        identity = f"{resolved}|{source_sha}"
        if successor:
            identity += f"|reader:{task_id}"
        complete = not content
        for start, end in ranges:
            complete = _covered_whole(identity, start, end, len(content))
        if not complete or output_consumed_by_reader(entry, task_id):
            return
        custody.record_output_consumed(
            drive, entry, artifact=f"{_ARTIFACT_SUBDIR}/{wanted}",
            byte_length=len(raw), sha256=source_sha, chars=len(content),
            lines=len(content.splitlines(keepends=True)), reader_task_id=task_id,
        )
    except Exception:
        log.warning("coverage acknowledgement for a staged delegated output failed", exc_info=True)


# -- bounded delivery of a structured payload (moved with the size-gate split;
# `tools.delegate` re-exports these names) ------------------------------------

_PREVIEW_STEPS = (6_000, 3_000, 1_200, 400, 0)
_BULK_FIELDS = ("final_summary", "primary_output")
_STRUCTURED_FIELDS = ("outcome_banner", "outcome_facts", "output_conformance", "failure")


def _preview_payload(full: Dict[str, Any], text: str, artifact: Optional[Dict[str, Any]],
                     budget: int, consumed: bool = False, full_ok: bool = True,
                     full_note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Shrink the inline view until it FITS, and say so in typed fields.

    The bulk fields are renamed to ``*_preview`` rather than silently shortened: a
    consumer reading ``primary_output`` gets nothing instead of a cut string it would
    mistake for the whole answer.

    ``consumed`` is the DURABLE fact (the D7 acknowledgement row exists), never an
    assumption: on first delivery it is False, and a re-wait on the same terminal run
    reports True only after the artifact really was read whole. ``full_ok`` is whether
    the staged content is the VERIFIED full result; ``full_note`` is the typed
    disclosure of how the engine's bounded primary-output preview was (or was not)
    resolved to the full artifact.
    """
    delivery: Dict[str, Any] = {
        "complete": False,
        "consumed": bool(consumed),
        "inline_is_preview": True,
        "total_chars": len(text),
        "artifact": artifact,
        "read_next": ({"tool": "read_file", "root": artifact["root"], "path": artifact["path"],
                       "start_line": 1, "max_lines": 2000} if artifact else None),
        "note": (
            (("PARTIAL inline, but the staged artifact has already been read whole — the "
              "durable acknowledgement exists, so this result counts as obtained."
              if consumed else
              "PARTIAL. The inline fields are a bounded preview; the whole terminal payload "
              "is the artifact above. Read it in chunks with read_file(root=..., path=..., "
              "start_line=N, max_lines=M) — start_line is a stable cursor over an immutable "
              "file — until your reads have covered EVERY character, contiguously. Delivery "
              "is char-bounded: a window longer than the tool-result budget is cut at "
              "delivery, and the cut remainder only counts as read once you advance WITHIN "
              "it via start_char. A review or research result is NOT consumed, and must not "
              "be reported as its verdict, until the whole artifact has been read.")
             if full_ok else
             "PARTIAL and INCOMPLETE AT THE SOURCE: the engine reported its primary output "
             "as a bounded preview and the full artifact could not be matched to the size "
             "or the preview the run itself reported (see primary_output_full; the engine "
             "publishes no content hash for it, so that match is the whole of the check). "
             "Treat this result as incomplete evidence, not as "
             "the verdict; it can never be acknowledged as fully read.")
            if artifact else
            "PARTIAL and UNRECOVERABLE INLINE: the full payload could not be staged to the "
            "task drive. Treat this result as incomplete evidence, not as the verdict."
        ),
    }
    if full_note is not None:
        delivery["primary_output_full"] = full_note
    payload: Dict[str, Any] = {}
    for preview_chars in _PREVIEW_STEPS:
        payload = {key: value for key, value in full.items() if key not in _BULK_FIELDS}
        for field in _BULK_FIELDS:
            raw = full.get(field)
            if raw is None:
                continue
            body = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            payload[f"{field}_preview"] = body[:preview_chars]
        for field in _STRUCTURED_FIELDS:
            value = payload.get(field)
            if value is not None and len(json.dumps(value, ensure_ascii=False)) > preview_chars:
                payload[field] = {"omitted": "see output_delivery.artifact"}
        payload["output_delivery"] = delivery
        # Same threshold as the complete branch: the headroom covers the JSON scaffold
        # and the settlement block the caller appends afterwards.
        if len(json.dumps(payload, ensure_ascii=False, indent=2)) <= budget - _PAYLOAD_ENVELOPE_HEADROOM:
            return payload
    return payload


# Tolerance for the preview-prefix consistency check below: the engine redacts the
# preview over a bounded prefix window with a 1 KiB overlap, so a secret spanning the
# preview boundary may redact differently in the full serve than in the preview tail.
_PREVIEW_PREFIX_SLACK = 2_048


def _resolve_full_primary_output(gateway: Any, run_id: str,
                                 primary: Any) -> Tuple[Any, bool, Optional[Dict[str, Any]]]:
    """Resolve the engine's bounded primary-output preview to the verified FULL text.

    ``primaryOutput.text`` on the run detail is a 256 KiB PREVIEW (control-api
    ``PRIMARY_OUTPUT_PREVIEW_BYTES``), with ``bytes`` (on-disk size) and ``truncated``
    beside it. A truncated preview must NEVER be staged, delivered or acknowledged as
    the result: the full file is fetched from ``GET /v2/runs/:id/artifacts/<path>`` and
    verified against what the run reported before it may wear the plain name.

    The engine reports NO content hash for the primary output, so verification is what
    the contract actually offers: the served size equal to the reported ``bytes``
    (exact), or — because the artifact route serves text through ``redactSecrets``,
    which can legally change the length — the fetched text carrying the preview as its
    prefix (up to a bounded slack at the preview boundary, where the engine's own
    redaction overlap can differ). Anything less keeps the preview, marked incomplete,
    with a typed disclosure — never a partial result wearing a full one's name.

    Returns ``(primary_output, full_ok, disclosure)``; ``disclosure`` is None when the
    engine never reported a truncation.
    """
    if not isinstance(primary, dict) or primary.get("truncated") is not True:
        return primary, True, None
    path = str(primary.get("path") or "")
    reported_bytes = primary.get("bytes")
    preview_text = primary.get("text") if isinstance(primary.get("text"), str) else ""
    disclosure: Dict[str, Any] = {"requested": True, "fetched": False, "verified": "",
                                  "path": path, "reported_bytes": reported_bytes}
    if not path or gateway is None:
        disclosure["reason"] = "no_artifact_path" if not path else "no_transport"
        return primary, False, disclosure
    try:
        raw = gateway.get_run_artifact(run_id, path)
    except Exception as exc:
        disclosure["reason"] = truncate_review_artifact(
            f"{getattr(exc, 'code', type(exc).__name__)}: {exc}", 300)
        return primary, False, disclosure
    disclosure["fetched"] = True
    disclosure["fetched_bytes"] = len(raw)
    full_text = raw.decode("utf-8", errors="replace")
    if isinstance(reported_bytes, int) and not isinstance(reported_bytes, bool) \
            and len(raw) == reported_bytes:
        disclosure["verified"] = "size"
    else:
        prefix = preview_text[:max(0, len(preview_text) - _PREVIEW_PREFIX_SLACK)]
        if prefix and full_text.startswith(prefix) and len(full_text) >= len(preview_text):
            disclosure["verified"] = "preview_prefix"
        else:
            disclosure["reason"] = "verification_failed_size_and_prefix"
            return primary, False, disclosure
    resolved = {**primary, "text": full_text, "truncated": False,
                "full_fetched": True, "verified_by": disclosure["verified"]}
    return resolved, True, disclosure
