"""Staged output for delegated runs, and the durable receipt that it was READ.

Terminal payloads too large for the tool budget are retained whole in the staging
task's canonical source store (#1573), beyond drive cleanup and the engine's own GC.
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

# Where a SECONDARY per-run artifact (a spilled question set) is staged under the
# task's own drive, and where terminal payloads were staged before #1573 (legacy rows).
_ARTIFACT_SUBDIR = "delegated_runs"
# #1573: the terminal payload is retained ONCE in the staging task's own source store on
# the CANONICAL drive and read back through ``artifact_store`` by its absolute path: the
# existing lineage read admits that store for the staging task on its own or a forked
# drive, for its children and for a direct Continue. No new root, category or grant.
_STAGED_SOURCE_ROOT = "artifact_store"
_STAGED_SOURCE_CATEGORY = "tool_results"
_STAGED_SOURCE_PREFIX = "delegated-run-"

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
    """Write the WHOLE terminal detail where its reader can reach it, and describe it.

    The terminal payload (no ``suffix``) is retained by ``_retain_terminal_output``.
    Host-owned by construction either way: the artifact is reachable through the
    ordinary ``read_file`` contract (which already carries a stable ``start_line``
    cursor) instead of a parallel artifact system.

    ``suffix`` distinguishes a SECONDARY per-run artifact (e.g. ``.interactions``
    for a spilled question set), still written to the task's own drive (every
    nanny profile reads ``task_drive``): it is deliberately outside the
    consumed-receipt contract — its receipt is the sha256/size pair returned here.
    """
    from ouroboros.tool_access import resource_root_path

    # The bytes DECLARED here and the bytes WRITTEN must be one object: the sha256 below
    # becomes the artifact's identity (`custody.output_sha`), and the read receipt measures
    # the file with `read_bytes`. A text write translates "\n" to os.linesep, so on Windows
    # the declared hash described a file that never existed on disk and the D7
    # acknowledgement could never be recorded for any staged output.
    staged_bytes = text.encode("utf-8", "replace")
    if not suffix:
        return _retain_terminal_output(ctx, run_id, text, staged_bytes)
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


def _retain_terminal_output(ctx: ToolContext, run_id: str, text: str,
                            staged_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Retain the terminal payload once in the STAGING task's canonical source store.

    The staging task is the reader whose context staged it (a confirmed retry
    successor stages an old run into its own store), named on the returned
    ``source`` beside the standard ``task_source`` ref. Write-once and
    digest-named, so a re-wait of identical bytes rewrites nothing, and neither
    the execution drive's cleanup nor the engine's GC reaches it. The read
    address is the absolute canonical path under ``artifact_store``.
    """
    from ouroboros.artifacts import task_id_for_artifacts

    return retain_terminal_bytes(custody.custody_root(ctx), task_id_for_artifacts(ctx), run_id, text, staged_bytes)


def retain_terminal_bytes(drive: Any, task_id: str, run_id: str, text: str,
                          staged_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Canonical retention without a reader or any consumption/acceptance claim."""
    from ouroboros.artifacts import store_actor_source_bytes

    try:
        ref = store_actor_source_bytes(
            drive, task_id, category=_STAGED_SOURCE_CATEGORY,
            source_id=f"{_STAGED_SOURCE_PREFIX}{_safe_run_filename(run_id)}",
            data=staged_bytes, extension="json")
        source = {**{key: ref[key] for key in ("kind", "root", "path", "size", "sha256")},
                  "task_id": task_id}
        target = staged_source_path(drive, source)
    except Exception:
        log.warning("Failed to retain delegated run output for %s", run_id, exc_info=True)
        return None
    return {
        "root": _STAGED_SOURCE_ROOT,
        "path": str(target),
        "abs_path": str(target),
        "source": source,
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "bytes": len(staged_bytes),
        "sha256": source["sha256"],
    }


def staged_source_path(drive_root: Any, source: Dict[str, Any]) -> pathlib.Path:
    """The canonical file one recorded ``output_source`` names (raises when malformed)."""
    from ouroboros.artifacts import task_artifact_dir_path

    rel = pathlib.PurePosixPath(str(source.get("path") or ""))
    if rel.is_absolute() or not rel.parts or rel.parts[0] != "source_handles" or ".." in rel.parts:
        raise ValueError("staged output source has an invalid path")
    return task_artifact_dir_path(drive_root, str(source.get("task_id") or "")).joinpath(*rel.parts)


def staged_output_address(drive_root: Any, entry: _RunCustody) -> Dict[str, str]:
    """``read_file`` root/path of a run's CURRENT staged output ({} when none).

    A legacy row names its path relative to the staging task's own drive, as before."""
    if not entry.output_artifact:
        return {}
    if not entry.output_source:
        return {"root": "task_drive", "path": entry.output_artifact}
    try:
        return {"root": _STAGED_SOURCE_ROOT, "path": str(staged_source_path(drive_root, entry.output_source))}
    except ValueError:
        return {}


def project_output_readers(row: Dict[str, Any], publisher_task_id: str = "") -> Dict[str, Any]:
    """Prefer the receipt-bound reader, without changing retained receipt bytes.

    ``task_id`` on a source is its staging store, NOT its physical publisher.
    The enclosing task/hop supplies the publisher; Wrap up rows also record it
    as ``owner_task_id``. A legacy source_read is usable when neither is known.
    Direct staging reads remain available to the owner for D7 acknowledgement.
    No source is opened and no read grant or consumption receipt is created.
    """
    projected = dict(row)

    def source(ref: Any) -> Any:
        if not isinstance(ref, dict):
            return ref
        old = ref.get("source_read")
        args = old.get("arguments") if isinstance(old, dict) else None
        publisher = str(publisher_task_id or row.get("owner_task_id") or "")
        if (not publisher and isinstance(args, dict) and old.get("tool") == "get_task_result"
                and args.get("delegated_output_sha256") == ref.get("sha256")):
            publisher = str(args.get("task_id") or "")
        digest = str(ref.get("sha256") or "")
        if not publisher or len(digest) != 64 or digest.strip("0123456789abcdef"):
            return dict(ref)
        reader = {"tool": "get_task_result", "arguments": {
            "task_id": publisher, "delegated_output_sha256": digest}}
        direct = ref.get("direct_read") or ref.get("read")
        return {**ref, "read": reader, "source_read": reader,
                **({"direct_read": direct} if isinstance(direct, dict)
                   and direct.get("tool") == "read_file" else {})}

    for key in ("staged_output", "terminal_observation"):
        if key in row:
            projected[key] = source(row[key])
    for key in ("other_staged_outputs", "other_terminal_observations"):
        if isinstance(row.get(key), list):
            projected[key] = [source(ref) for ref in row[key]]
    return projected


def project_task_output_readers(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read projection for current, predecessor and Wrap up receipt locators."""
    def rows(value: Any, publisher: str) -> Any:
        return ([project_output_readers(row, publisher) if isinstance(row, dict) else row
                 for row in value] if isinstance(value, list) else value)

    def audit(value: Any, publisher: str) -> Any:
        if not isinstance(value, dict):
            return value
        out = dict(value)
        if "terminal_runs" in out:
            out["terminal_runs"] = rows(out["terminal_runs"], publisher)
        receipt = out.get("owner_stop_children")
        if isinstance(receipt, dict):
            receipt = dict(receipt)
            # Mixed predecessor runs carry their own owner_task_id; never use
            # the Wrap up root (or a source's staging id) for all of them.
            if "runs" in receipt:
                receipt["runs"] = rows(receipt["runs"], "")
            if isinstance(receipt.get("children"), list):
                receipt["children"] = [{**child, "custody": audit(child["custody"], str(child.get("task_id") or ""))}
                    if isinstance(child, dict) and isinstance(child.get("custody"), dict) else child
                    for child in receipt["children"]]
            out["owner_stop_children"] = receipt
        return out

    out = dict(data)
    if "delegate_terminal_reconciliation" in out:
        out["delegate_terminal_reconciliation"] = audit(
            out["delegate_terminal_reconciliation"], str(data.get("task_id") or data.get("id") or ""))
    if isinstance(data.get("retry_lineage"), list):
        out["retry_lineage"] = [{**hop, "terminal_runs": rows(hop["terminal_runs"], str(hop.get("task_id") or ""))}
            if isinstance(hop, dict) and "terminal_runs" in hop else hop for hop in data["retry_lineage"]]
    return out


def engine_locator(detail: Any) -> Dict[str, Any]:
    """The engine's OWN locators for a run's work, recorded on its SETTLED row.

    ``run_dir`` and the primary output's kind/path/bytes/truncated — never its
    text. Engine-owned: the engine may reclaim them, and they grant no read
    authority. Absent facts stay absent; nothing here is inferred or guessed.
    """
    detail = detail if isinstance(detail, dict) else {}
    locator: Dict[str, Any] = {}
    run_dir = custody.summary_of(detail).get("runDir")
    if isinstance(run_dir, str) and run_dir.strip():
        locator["run_dir"] = run_dir
    primary = detail.get("primaryOutput")
    if isinstance(primary, dict):
        facts: Dict[str, Any] = {key: primary[key] for key in ("kind", "path")
                                 if isinstance(primary.get(key), str) and primary[key]}
        if isinstance(primary.get("bytes"), int) and not isinstance(primary.get("bytes"), bool):
            facts["bytes"] = primary["bytes"]
        if isinstance(primary.get("truncated"), bool):
            facts["truncated"] = primary["truncated"]
        if facts:
            locator["primary_output"] = facts
    return locator


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
        **({"staged_output_source": dict(custody.output_source)} if custody.output_source else {}),
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
    """Intersect the host's delivered result ranges with this reader's file body.

    The body is read_file's numbered display: the renderer's own decoder maps
    result offsets to source characters, after its rows proved equal to the
    exact source window, so display numbers and a hidden middle credit nothing.
    """
    from ouroboros.tool_result_delivery import RESULT_VIEW_BASIS, reader_text
    from ouroboros.tools.core_file_tools import delivered_source_prefix, display_intervals_to_source

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
            or delivered_source_prefix(read_view, original, len(original)) != content[start:end]):
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
    for span in shown:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not all(isinstance(n, int) for n in span)
                or not 0 <= span[0] <= span[1] <= len(original)):
            return None
    ranges = display_intervals_to_source(read_view, original, [(lo, hi) for lo, hi in shown])
    return None if ranges is None else [[lo, hi] for lo, hi in ranges]


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
    if not read_view:
        _acknowledge_inline_terminal(ctx, projected)
    _acknowledge_staged_output(ctx, read_view, exec_result.get("result", ""), projected, shown_ranges)


def _acknowledge_inline_terminal(ctx: ToolContext, projected: Dict[str, Any]) -> None:
    """Credit body-retained bytes only in a complete actual tool-result view."""
    if projected.get("result_partial"):
        return
    try:
        payload = json.loads(projected.get("result") or "")
        delivery = payload.get("output_delivery") or {}
        if payload.get("status") != "terminal" or not delivery.get("complete") or delivery.get("inline_is_preview"):
            return
        from ouroboros.delegate_custody_current import run
        from ouroboros.artifacts import read_actor_source_bytes
        from ouroboros.delegate_shared import retry_result_status

        root, reader = custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or "")
        entry = run(root, str(payload.get("run_id") or ""))
        if entry is None or not entry.output_complete or not entry.output_source:
            return
        if reader != entry.task_id and retry_result_status(ctx, root, entry.run_id)[0] != custody.OWNED:
            return
        ref = entry.output_source
        delivered_sha = delivery.get("retained_sha256") or (delivery.get("artifact") or {}).get("sha256")
        if delivered_sha != entry.output_sha:
            return
        raw = read_actor_source_bytes(root, str(ref.get("task_id") or entry.task_id), ref)
        full = json.loads(raw)
        if any(payload.get(key) != value for key, value in full.items()):
            return
        text = raw.decode("utf-8")
        record_output_consumed(root, entry, artifact=entry.output_artifact, byte_length=len(raw),
            sha256=entry.output_sha, chars=len(text), lines=text.count("\n") + 1, reader_task_id=reader)
    except Exception:
        log.debug("Inline delegated output acknowledgement unavailable", exc_info=True)
        return  # A partial/malformed/unavailable projection never earns an EOF receipt.


def _staged_source_candidate(resolved: pathlib.Path) -> bool:
    """Read-free qualifier: only a delegated-run source-store file name can be one."""
    return (resolved.suffix == ".json" and resolved.name.startswith(_STAGED_SOURCE_PREFIX)
            and resolved.parent.name == _STAGED_SOURCE_CATEGORY
            and resolved.parent.parent.name == "source_handles")


def _names_staged_source(drive: Any, entry: _RunCustody, resolved: pathlib.Path) -> bool:
    if not entry.output_source:
        return False
    try:
        return staged_source_path(drive, entry.output_source).resolve(strict=False) == resolved
    except ValueError:
        return False


def _acknowledge_staged_output(ctx: ToolContext, read_view: Any, original: str,
                              projected: Dict[str, Any],
                              shown_ranges: Optional[List[List[int]]] = None) -> None:
    """The ONE D7 credit: actual delivered ranges of the exact staged source.

    A legacy staging is the reader's own ``task_drive`` ``delegated_runs/<run>.json``;
    a retained one (#1573) must be EXACTLY the canonical file its custody row
    recorded (staging task plus source path), read through ``artifact_store`` or
    ``runtime_data`` — a name, category or equal text is no evidence, and the name
    test runs before any read or custody replay so unrelated reads cost nothing.
    Only the run's starter or its confirmed retry successor earns a receipt.
    """
    try:
        from ouroboros.tool_access import resource_root_path

        if not isinstance(read_view, dict):
            return
        opened_root = read_view.get("opened_root")
        path = pathlib.Path(read_view["target"])
        resolved = path.resolve(strict=False)
        legacy = opened_root == "task_drive"
        if legacy:
            artifact_dir = (resource_root_path(ctx, "task_drive") / _ARTIFACT_SUBDIR).resolve(strict=False)
            if resolved.parent != artifact_dir:
                return
        elif opened_root not in {"artifact_store", "runtime_data"} or not _staged_source_candidate(resolved):
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
        drive = custody.custody_root(ctx)

        def _is(entry: _RunCustody) -> bool:
            if legacy:
                return not entry.output_source and _safe_run_filename(entry.run_id) + ".json" == path.name
            return _names_staged_source(drive, entry, resolved)

        def _match(candidates: Any) -> Optional[_RunCustody]:
            return next((c for c in candidates if _is(c)), None)

        # Memo first; a restarted worker (empty or unrelated memo) falls through to the
        # durable replay, the same authority every other custody question consults.
        entry = _match(list(custody._CUSTODY.values())) or _match(custody.replay(drive).values())
        if entry is None:
            return
        successor = entry.task_id != task_id
        if successor:
            from ouroboros.delegate_shared import retry_result_status

            status, entry, predecessor = retry_result_status(ctx, drive, entry.run_id)
            if status != custody.OWNED or entry is None or not predecessor or not _is(entry):
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
            drive, entry, artifact=entry.output_artifact if entry.output_source else f"{_ARTIFACT_SUBDIR}/{path.name}",
            byte_length=len(raw), sha256=source_sha, chars=len(content),
            lines=len(content.splitlines(keepends=True)), reader_task_id=task_id,
        )
    except Exception:
        log.warning("coverage acknowledgement for a staged delegated output failed", exc_info=True)


def published_output_ref(result: Any, sha256: str) -> tuple:
    """``(row, ref)`` of the one ``staged_output``, ``terminal_observation`` or ``other_*`` ref that
    ``result``'s own ``delegate_terminal_reconciliation.terminal_runs`` publishes under exactly this
    digest, else ``(None, None)``. The ref names its ACTUAL staging task; the caller reads it there."""
    audit = result.get("delegate_terminal_reconciliation") if isinstance(result, dict) else None
    rows = audit.get("terminal_runs") if isinstance(audit, dict) else None
    return next(((row, ref) for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
                 for others in [row.get("other_staged_outputs")]
                 for observations in [row.get("other_terminal_observations")]
                 for ref in (row.get("staged_output"), row.get("terminal_observation"),
                             *(others if isinstance(others, list) else ()),
                             *(observations if isinstance(observations, list) else ()))
                 if isinstance(ref, dict) and str(ref.get("sha256") or "") == sha256), (None, None))


def delegated_output_source_projection(drive_root: Any, result: Any, sha256: Any,
                                       start_char: Any = None, end_char: Any = None) -> Dict[str, Any]:
    """Read one retained full delegated output that ``result`` itself publishes (#1573).

    Only a ref ``published_output_ref`` finds in this task result's own receipts qualifies,
    selected by its exact sha256: no digest search, filesystem scan or engine call. The bytes come from the
    ACTUAL staging task's store on the canonical root, verified against the published
    size and hash, so a reader on any drive gets the same exact source. Discovery, never
    consumption: this read is not the D7 acknowledgement and records nothing.
    """
    from ouroboros.artifacts import read_actor_source_bytes, text_source_range_projection

    unavailable: Dict[str, Any] = {"schema": 1, "kind": "delegated_run_output", "status": "unavailable"}
    wanted = str(sha256 or "").strip().lower()
    if len(wanted) != 64 or wanted.strip("0123456789abcdef"):
        return {**unavailable, "reason": "source_ref_invalid"}
    row, staged = published_output_ref(result, wanted)
    if row is None:
        return {**unavailable, "reason": "source_not_published", "requested_sha256": wanted}
    facts = {"run_id": str(row.get("run_id") or ""), "staging_task_id": str(staged.get("task_id") or ""),
             "sha256": wanted, "source_complete": staged.get("complete") is True}
    if staged.get("content_kind") == "terminal_observation":
        facts["content_kind"] = "terminal_observation"  # exact read-back, not proof of full engine output
    ref = {key: staged.get(key) for key in ("kind", "root", "path", "size", "sha256")}
    try:
        raw = read_actor_source_bytes(drive_root, facts["staging_task_id"], ref)
        projection, reason = text_source_range_projection(
            raw.decode("utf-8"), unavailable["kind"], start_char, end_char)
    except ValueError as exc:
        reason = "source_identity_mismatch" if "verification" in str(exc) else "source_ref_invalid"
        return {**unavailable, **facts, "reason": reason}
    except (OSError, RuntimeError):
        return {**unavailable, **facts, "reason": "source_unavailable"}
    # A retained ENGINE PREVIEW stays an explicit gap: the bytes are exact, the result is not whole.
    status = "available" if facts["source_complete"] else "incomplete"
    return {**(projection or unavailable), "status": status, **facts, **({"reason": reason} if reason else {})}


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
            "PARTIAL and UNRECOVERABLE INLINE: the full payload could not be retained. "
            "Treat this result as incomplete evidence, not as the verdict."
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
        disclosure["retry"] = "changed_input" if not path else "transient"
        return primary, False, disclosure
    try:
        raw = gateway.get_run_artifact(run_id, path)
    except Exception as exc:
        from ouroboros.delegate_capture_attempt import failure_retry

        disclosure["retry"] = failure_retry(exc)
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
