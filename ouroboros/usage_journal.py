"""The retired usage JSONL journal: read ONCE by the store migration, written by the export.

``state/usage_attempts.jsonl`` was the monetary authority before the usage
store (docs/USAGE_STORE.md). Two explicit jobs still speak its format and no
ordinary path does: ``usage_store.migrate_from_journal`` reads the live journal
with the validated reader below (the last full read ever; a torn final row is
quarantined exactly as before) plus the pre-ledger telemetry an install that
never finished its legacy import still owes; ``export_journal`` writes, from
the store, a journal an older release accepts, offline, as the step before the
owner checks out such a release.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import logging
import lzma
import os
import pathlib
import threading
import uuid
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ouroboros._usage_money import LiteralFloat, amount, durable_literals
from ouroboros._usage_rows import BINDING_KEYS, _BINDING_CAPS, valid_work_cap
from ouroboros.usage_ledger import (
    ARCHIVE_SEGMENT_DIR_REL, LEDGER_REL, QUARANTINE_REL, UsageAccountingError, UsageLedgerCorrupt,
    _SHA256_RE, _number, is_one_shot_kind,
    late_receipt_eligible, valid_archive_rel, validate_row_fields, validate_transition,
)
from ouroboros.utils import append_jsonl, atomic_write_json, replace_atomic, utc_now_iso

log = logging.getLogger(__name__)

IMPORT_REL = pathlib.Path("state/usage_import_watermark.json")  # legacy telemetry import watermark
EXPORTED_WORK_CAPS = "exported_work_caps"


def _append_bytes_fsync(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short append to {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_atomic_fsync(
    path: pathlib.Path,
    payload: bytes,
    precondition: Optional[Callable[[], bool]] = None,
) -> bool:
    """Persist the exact bytes via a durable sibling temp file and an atomic
    rename; a ``False`` precondition leaves the destination untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")
    fd: Optional[int] = None
    try:
        # Windows defaults low-level descriptors to text mode, which would
        # expand LF bytes and break an immutable source hash.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(str(tmp), flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write to {tmp}")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        if not replace_atomic(tmp, path, precondition=precondition):
            tmp.unlink()
            return False
        return True
    except Exception:
        if fd is not None:
            os.close(fd)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _quarantine_tail(root: pathlib.Path, raw: bytes, offset: int, reason: str) -> None:
    ledger = root / LEDGER_REL
    row = {
        "ts": utc_now_iso(),
        "reason": reason,
        "source": str(ledger),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }
    _append_bytes_fsync(
        root / QUARANTINE_REL,
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )
    fd = os.open(str(ledger), os.O_RDWR)
    try:
        os.ftruncate(fd, offset)
        os.fsync(fd)
    finally:
        os.close(fd)
    log.error("Quarantined corrupt final usage-ledger row: %s", reason)
    try:
        append_jsonl(
            root / "logs" / "events.jsonl",
            {"type": "usage_ledger_tail_quarantined", "ts": utc_now_iso(), "reason": reason},
        )
    except Exception:
        log.exception("Failed to emit usage-ledger quarantine event")


def _validate_baseline_header(row: Dict[str, Any], sequence: int) -> None:
    """Provenance checks on a compaction stamp: a bounded archive path, a
    well-formed source hash, a positive epoch, and counts that close."""

    def _count(key: str, minimum: int) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise UsageLedgerCorrupt(f"invalid usage baseline {key} seq={sequence}")
        return value

    _count("compaction_epoch", 1)
    if not valid_archive_rel(row.get("archive_rel")):
        raise UsageLedgerCorrupt(f"invalid usage baseline archive_rel seq={sequence}")
    if not _SHA256_RE.fullmatch(str(row.get("source_sha256") or "")):
        raise UsageLedgerCorrupt(f"invalid usage baseline source_sha256 seq={sequence}")
    _count("source_size_bytes", 1)
    _count("folded_attempt_count", 1)
    _count("group_count", 1)
    source_rows = _count("source_row_count", 1)
    folded_rows = _count("folded_row_count", 1)
    retained_rows = _count("retained_row_count", 0)
    if _count("source_first_seq", 1) != 1 or _count("source_last_seq", 1) != source_rows:
        raise UsageLedgerCorrupt(f"usage baseline source range mismatch seq={sequence}")
    if folded_rows + retained_rows != source_rows:
        raise UsageLedgerCorrupt(f"usage baseline row counts do not sum seq={sequence}")


def _validate_records(records: Sequence[Dict[str, Any]]) -> None:
    """Validate a whole journal: dense sequence, the leading baseline block
    (one ``usage_baseline`` header at seq 1, ``usage_baseline_group`` rows
    joined by ``baseline_id``, counts closing with the header), retained-row
    ``pre_compaction_seq`` provenance, and every row's structural and
    transition rules (``usage_ledger.validate_row_fields`` /
    ``validate_transition``, the same rules the store applies on write)."""
    baseline_allowed = True
    baseline_id: Optional[str] = None
    baseline_header: Optional[Dict[str, Any]] = None
    baseline_groups = baseline_attempts = 0
    baseline_closed = pre_compaction_closed = False
    pre_compaction_seq = 0

    def _close_baseline_block() -> None:
        nonlocal baseline_closed
        if baseline_closed or baseline_header is None:
            return
        baseline_closed = True
        if baseline_groups != int(baseline_header.get("group_count") or 0):
            raise UsageLedgerCorrupt("usage baseline group_count does not match the block")
        if baseline_attempts != int(baseline_header.get("folded_attempt_count") or 0):
            raise UsageLedgerCorrupt("usage baseline folded_attempt_count does not match the block")

    states: Dict[str, str] = {}
    previous_rows: Dict[str, dict] = {}
    late: set = set()
    expected = 1
    for row in records:
        try:
            sequence = int(row.get("seq") or 0) if isinstance(row, dict) else 0
        except (TypeError, ValueError, OverflowError) as exc:
            raise UsageLedgerCorrupt(f"invalid usage ledger sequence at {expected}") from exc
        if not isinstance(row, dict) or sequence != expected:
            raise UsageLedgerCorrupt(f"usage ledger sequence mismatch at {expected}")
        expected += 1
        validate_row_fields(row, sequence)
        attempt_id = str(row.get("attempt_id") or "")
        state = str(row.get("state") or "")
        kind = str(row.get("kind") or "attempt")
        previous = states.get(attempt_id)
        if kind in {"usage_baseline", "usage_baseline_group"}:
            if not baseline_allowed or previous is not None:
                raise UsageLedgerCorrupt(f"baseline row outside the leading block at seq={sequence}")
            if kind == "usage_baseline":
                identity = row.get("baseline_id")
                if baseline_id is not None or sequence != 1 or state != "settled" or not (
                        isinstance(identity, str) and identity):
                    raise UsageLedgerCorrupt(f"invalid usage baseline header seq={sequence}")
                _validate_baseline_header(row, sequence)
                baseline_id, baseline_header = identity, row
            else:
                count = row.get("folded_attempt_count")
                if (baseline_id is None or row.get("baseline_id") != baseline_id
                        or isinstance(count, bool) or not isinstance(count, int) or count < 1
                        or state not in {"settled", "unresolved", "released"}):
                    raise UsageLedgerCorrupt(f"invalid usage baseline group seq={sequence}")
                baseline_groups += 1
                baseline_attempts += count
            states[attempt_id] = state
            continue
        baseline_allowed = False
        _close_baseline_block()
        carried = row.get("pre_compaction_seq")
        if carried is not None:
            if (baseline_header is None or pre_compaction_closed or isinstance(carried, bool)
                    or not isinstance(carried, int) or carried <= pre_compaction_seq
                    or carried < int(baseline_header.get("source_first_seq") or 0)
                    or carried > int(baseline_header.get("source_last_seq") or 0)):
                raise UsageLedgerCorrupt(f"invalid pre_compaction_seq in usage row seq={sequence}")
            pre_compaction_seq = carried
        else:
            pre_compaction_closed = True
        validate_transition(row, previous, attempt_id in late, sequence,
                            previous_row=previous_rows.get(attempt_id))
        previous_rows[attempt_id] = row
        states[attempt_id] = state
        if late_receipt_eligible(row):
            late.add(attempt_id)
        else:
            late.discard(attempt_id)
    _close_baseline_block()


def _decode_record(chunk: bytes) -> Optional[Dict[str, Any]]:
    """One JSONL grammar: only empty CR/LF lines are empty records; UTF-8
    objects; money literals retained (``LiteralFloat``)."""
    raw = chunk.rstrip(b"\r\n")
    if not raw:
        return None
    row = json.loads(raw.decode("utf-8"), parse_float=LiteralFloat)
    if not isinstance(row, dict):
        raise ValueError("row is not an object")
    return row


def read_journal_records(root: pathlib.Path) -> Tuple[list, bytes]:
    """Every validated row of the live journal plus the exact bytes they came
    from. A torn or structurally invalid FINAL row is quarantined (appended to
    the quarantine file, the journal truncated before it); corruption before
    the final row is fatal. The caller holds the money name lock."""
    path = root / LEDGER_REL
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], b""
    except OSError as exc:
        raise UsageAccountingError(f"cannot read usage ledger: {exc}") from exc
    records: list[Dict[str, Any]] = []
    record_locations: list[Tuple[int, bytes]] = []
    chunks = data.splitlines(keepends=True)
    nonempty = [index for index, chunk in enumerate(chunks) if chunk.rstrip(b"\r\n")]
    last_nonempty = nonempty[-1] if nonempty else -1
    offset = 0
    for index, chunk in enumerate(chunks):
        try:
            row = _decode_record(chunk)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if index == last_nonempty:
                _quarantine_tail(root, chunk, offset, f"{type(exc).__name__}: {exc}")
                data = data[:offset]
                break
            raise UsageLedgerCorrupt(f"corrupt usage ledger row before tail: {index + 1}") from exc
        if row is not None:
            records.append(row)
            record_locations.append((offset, chunk))
        offset += len(chunk)
    try:
        _validate_records(records)
    except UsageLedgerCorrupt:
        # A final row can be valid JSON yet structurally torn; the validated
        # history before it is preserved exactly as for a JSON-torn tail.
        if not records:
            raise
        _validate_records(records[:-1])
        bad_offset, bad_chunk = record_locations[-1]
        _quarantine_tail(root, bad_chunk, bad_offset, "structurally invalid final ledger row")
        records.pop()
        data = data[:bad_offset]
    return records, data


# ---- pre-ledger telemetry (an install whose legacy import never completed) ----

def completed_legacy_watermark(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Only a missing file is absent; a failed read must not re-import legacy spend or erase authority."""
    try:
        value = json.loads((root / IMPORT_REL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UsageAccountingError("cannot read the completed legacy usage watermark") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise UsageLedgerCorrupt("unreadable legacy usage watermark") from exc
    if not isinstance(value, dict) or value.get("completed") is not True:
        raise UsageLedgerCorrupt("invalid legacy usage watermark; expected a completed import")
    return value


def import_completion(watermark: Optional[dict]) -> Optional[dict]:
    """The already validated pre-ledger import completion, not a previous export's decision set."""
    return {key: value for key, value in watermark.items() if key != EXPORTED_WORK_CAPS} if watermark else None


def _archived_export_prefix(root, header, *, size, digest) -> bool:
    """Follow only the older writer's hash-bound predecessors; do not re-fold their money.

    This is a cold import check, and only for a watermarked export whose live prefix changed.
    """
    while header and header.get("kind") == "usage_baseline":
        _validate_baseline_header(header, 1)
        path = root / header["archive_rel"]
        if path.parent.resolve() != root.resolve() / ARCHIVE_SEGMENT_DIR_REL or path.is_symlink():
            raise UsageLedgerCorrupt("export ancestry archive escapes its retained directory")
        # The evidence packer can replace this exact segment with its .xz form.
        # The compactor's own header is the size/digest authority for either placement.
        packed = not path.is_file()
        if packed:
            path = path.with_name(path.name + ".xz")
        if path.is_symlink() or not path.is_file():
            raise UsageLedgerCorrupt("export ancestry archive is unavailable")
        full, prefix, count = hashlib.sha256(), hashlib.sha256(), 0
        with (lzma.open(path, "rb") if packed else path.open("rb")) as stream:
            first = stream.readline(header["source_size_bytes"] + 1)
            for part in itertools.chain((first,), iter(lambda: stream.read(1024 * 1024), b"")):
                full.update(part)
                if count < size:
                    prefix.update(part[:size - count])
                count += len(part)
                if count > header["source_size_bytes"]:
                    raise UsageLedgerCorrupt("export ancestry archive exceeds its recorded size")
        if count != header["source_size_bytes"] or full.hexdigest() != header["source_sha256"]:
            raise UsageLedgerCorrupt("export ancestry archive changed")
        if (count == size or 0 < size < count) and prefix.hexdigest() == digest:
            return True
        predecessor = _decode_record(first)
        if not predecessor or predecessor.get("kind") != "usage_baseline":
            return False
        _validate_baseline_header(predecessor, 1)
        if predecessor["compaction_epoch"] >= header["compaction_epoch"]:
            raise UsageLedgerCorrupt("export ancestry epochs do not decrease")
        header = predecessor
    return False


def _valid_cap_map(entries: Any) -> bool:
    return isinstance(entries, dict) and all(
        isinstance(group, str) and group and valid_work_cap(cap) and cap.get("billing_group_id") == group
        for group, cap in entries.items())


def work_caps_stamp(stamp: Any) -> Tuple[int, str, dict, dict]:
    """Validate the transmitted monetary decisions before restoring or replacing them."""
    if isinstance(stamp, dict):
        size, digest, caps = stamp.get("size"), stamp.get("sha256"), stamp.get("caps")
        unverified = stamp.get("unverified_caps", {})
        if (type(size) is int and size >= 0 and isinstance(digest, str) and _SHA256_RE.fullmatch(digest)
                and _valid_cap_map(caps) and _valid_cap_map(unverified)
                and (caps or unverified) and not (caps.keys() & unverified.keys())
                and ("unverified_caps" not in stamp or bool(unverified))):
            return size, digest, caps, unverified
    raise UsageLedgerCorrupt("invalid exported work caps in legacy usage watermark")


def split_work_caps(records: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate active work caps from carried unverified export records, failing closed on corruption."""
    verified, unverified = {}, {}
    for group, record in records.items():
        if isinstance(record, dict) and set(record) == {"unverified_export"}:
            unverified[group] = record["unverified_export"]
        else:
            verified[group] = record
    if not (_valid_cap_map(verified) and _valid_cap_map(unverified)):
        raise UsageLedgerCorrupt("invalid exported work caps in usage store")
    return verified, unverified


def work_caps_export_meta(watermark: Optional[dict], work_caps: dict, *,
                          restored: int, unverified: int, preserved: int = 0) -> Optional[Dict[str, Any]]:
    if not work_caps:
        return None
    stamp = watermark[EXPORTED_WORK_CAPS]
    status = "verified" if not unverified else ("partial" if restored else "unverified")
    return {"status": status, "size": stamp["size"], "sha256": stamp["sha256"],
            "restored": restored, "unverified": unverified, "preserved": preserved}


def plan_work_caps_export(root: pathlib.Path, provenance_legacy: Any, caps: Dict[str, Any], *,
                          journal_id: Optional[Dict[str, Any]] = None,
                          reconcile_pending: bool = False, had_export: bool = False) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any]:
    """Validate watermark and work-cap export prerequisites shared by preview and freeze."""
    watermark = completed_legacy_watermark(root)
    if reconcile_pending and watermark is None and had_export:
        raise UsageLedgerCorrupt("the legacy usage watermark is missing after an export; keep the export and its watermark together")
    legacy = import_completion(watermark) or provenance_legacy
    if not legacy:
        raise UsageAccountingError("usage export requires its completed legacy import watermark")
    prior = (watermark or {}).get(EXPORTED_WORK_CAPS)
    if EXPORTED_WORK_CAPS in (watermark or {}):
        _, _, prior_caps, prior_unverified = work_caps_stamp(prior)
        if not (prior_caps.keys() | prior_unverified.keys()) <= caps.keys():
            raise UsageLedgerCorrupt("the prior usage export has decisions not held in the active store")
    verified, unverified = split_work_caps(caps)
    export_caps = None
    if verified or unverified:
        export_caps = {**(journal_id or {"size": 0, "sha256": "0" * 64, "rows": 0}), "caps": verified,
                       **({"unverified_caps": unverified} if unverified else {})}
        work_caps_stamp(export_caps)
    return legacy, export_caps, prior


def _kept_journal_export_prefix(
    root: pathlib.Path, *, import_size: int, import_sha256: str, export_size: int, export_sha256: str,
) -> bool:
    """Verify the kept live journal's imported prefix matches ``meta.import.source`` and a nonempty export prefix."""
    if not (0 < export_size <= import_size):
        return False
    path = root / LEDGER_REL
    if path.is_symlink() or not path.is_file():
        return False
    imported, prefix, count = hashlib.sha256(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        while count < import_size:
            part = stream.read(min(1024 * 1024, import_size - count))
            if not part:
                return False
            imported.update(part)
            if count < export_size:
                prefix.update(part[:export_size - count])
            count += len(part)
    return count == import_size and imported.hexdigest() == import_sha256 and prefix.hexdigest() == export_sha256


def _resolve_work_caps(root: pathlib.Path, verified: dict, unverified: dict, *,
                       proven: bool, header: Optional[dict], size: int, digest: str) -> dict:
    """Apply proven verified caps; carry unproven or previously unverified amendments separately."""
    if verified and not proven and header and size > 0:
        try:
            proven = _archived_export_prefix(root, header, size=size, digest=digest)
        except (UsageLedgerCorrupt, OSError, ValueError, OverflowError, EOFError, lzma.LZMAError) as exc:
            log.warning("Cannot verify the usage export's ancestry: %s", exc)
    resolved = {group: cap if proven else {"unverified_export": cap} for group, cap in verified.items()}
    resolved.update({group: {"unverified_export": cap} for group, cap in unverified.items()})
    return resolved


def imported_work_caps(root: pathlib.Path, watermark: Optional[dict], import_meta: Optional[dict]) -> dict:
    """Resolve exported work caps against an already-completed store's recorded import provenance."""
    if not watermark or EXPORTED_WORK_CAPS not in watermark:
        return {}
    size, digest, verified, unverified = work_caps_stamp(watermark[EXPORTED_WORK_CAPS])
    provenance = import_meta if isinstance(import_meta, dict) else {}
    source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    header = provenance.get("header") if isinstance(provenance.get("header"), dict) else None
    import_size, import_sha256 = source.get("size"), source.get("sha256")
    valid_source = (source.get("present") is True and type(import_size) is int and import_size >= 0
                    and isinstance(import_sha256, str) and bool(_SHA256_RE.fullmatch(import_sha256)))
    proven = False
    if valid_source and verified:
        if import_size == size and import_sha256 == digest:
            proven = True
        elif 0 < size < import_size:
            try:
                proven = _kept_journal_export_prefix(
                    root, import_size=import_size, import_sha256=import_sha256,
                    export_size=size, export_sha256=digest)
            except OSError as exc:
                log.warning("Cannot read the kept usage journal to verify export lineage: %s", exc)
    return _resolve_work_caps(root, verified, unverified, proven=proven,
                              header=header if valid_source else None, size=size, digest=digest)


def plan_work_caps_reconcile(
    root: pathlib.Path, status: Dict[str, Any], *, had_export: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Read the legacy watermark and resolve exported work caps for an already-completed store.
    Returns ``(pending_state, watermark, work_caps)``; malformed ``exported_work_caps`` raises."""
    reason, watermark = "", None
    try:
        watermark = completed_legacy_watermark(root)
    except UsageAccountingError as exc:
        reason = "watermark_unreadable_after_export" if had_export else "watermark_unreadable"
        log.warning("Skipping work-cap export reconciliation on completed store (%s): %s", reason, exc)
    else:
        if watermark is None and had_export:
            reason = "watermark_missing_after_export"
            log.warning("Completed usage store has a retired export without %s", IMPORT_REL)
    pending = {"status": "pending", "reason": reason} if reason else None
    work_caps = {} if pending or not watermark else imported_work_caps(root, watermark, status)
    return pending, watermark, work_caps


def exported_work_caps(root: pathlib.Path, watermark: Optional[dict], records: Sequence[dict],
                       source: bytes, *, present: bool) -> dict:
    """Restore proven amendments. Preserve unverified ones separately from effective authority."""
    if not watermark or EXPORTED_WORK_CAPS not in watermark:
        return {}
    size, digest, verified, unverified = work_caps_stamp(watermark[EXPORTED_WORK_CAPS])
    if not present:
        raise UsageLedgerCorrupt("the live journal is missing beside the usage export watermark")
    proven = bool(verified) and (size == len(source) or 0 < size < len(source)) and (
        hashlib.sha256(memoryview(source)[:size]).hexdigest() == digest)
    return _resolve_work_caps(root, verified, unverified, proven=proven,
                              header=records[0] if records else None, size=size, digest=digest)


def write_exported_work_caps(root: pathlib.Path, value: Optional[dict], *, fallback: Optional[dict] = None) -> None:
    """Update only the export portion of the existing watermark, before retiring the store."""
    current = completed_legacy_watermark(root) or fallback
    if current is None:
        raise UsageAccountingError("usage export requires its completed legacy import watermark")
    current = dict(current)
    if value is None:
        current.pop(EXPORTED_WORK_CAPS, None)
    else:
        current[EXPORTED_WORK_CAPS] = value
    atomic_write_json(root / IMPORT_REL, current, trailing_newline=True, fsync=True)


def legacy_snapshot(root: pathlib.Path) -> Tuple[list, Dict[str, Any], Dict[str, str]]:
    """Snapshot and archive the pre-ledger usage sources (``llm_usage`` events
    across the rotated event chain, ``state.json``), hashing settings without
    copying them. STRICT: an unreadable segment is a typed incomplete view."""
    events_path = root / "logs" / "events.jsonl"
    state_path = root / "state" / "state.json"
    settings_path = pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH") or root / "settings.json")
    sources = {"events.jsonl": events_path, "state.json": state_path}
    snapshots: Dict[str, bytes] = {}
    try:
        from ouroboros.utils import jsonl_chain_handles

        chain_parts: list[bytes] = []
        with jsonl_chain_handles(events_path, strict=True) as handles:
            for _, handle in handles:
                chain_parts.append(handle.read())
        if chain_parts:
            snapshots["events.jsonl"] = b"".join(chain_parts)
    except OSError as exc:
        raise UsageAccountingError(f"cannot snapshot legacy usage source {events_path}: {exc}") from exc
    try:
        snapshots["state.json"] = state_path.read_bytes()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise UsageAccountingError(f"cannot snapshot legacy usage source {state_path}: {exc}") from exc
    hashes = {name: hashlib.sha256(snapshots[name]).hexdigest() if name in snapshots else "" for name in sources}
    try:
        hashes["settings.json"] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        hashes["settings.json"] = ""
    except OSError as exc:
        raise UsageAccountingError(f"cannot hash settings file {settings_path}: {exc}") from exc
    rows: list[Dict[str, Any]] = []
    try:
        for line_no, line in enumerate(snapshots.get("events.jsonl", b"").decode("utf-8").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("type") == "llm_usage":
                rows.append({**value, "_legacy_line": line_no})
    except UnicodeDecodeError:
        pass
    try:
        state = json.loads(snapshots.get("state.json", b"{}").decode("utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        state = {}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    archive = root / "archive" / "usage_import" / combined
    archive.mkdir(parents=True, exist_ok=True)
    for name, payload in snapshots.items():
        target = archive / name
        if target.exists():
            if target.read_bytes() != payload:
                raise UsageAccountingError(f"legacy usage archive mismatch: {target}")
        else:
            _write_bytes_atomic_fsync(target, payload)
            try:
                target.chmod(0o400)
            except OSError:
                pass
    atomic_write_json(archive / "sha256.json", hashes, trailing_newline=True, fsync=True)
    return rows, state, hashes


def legacy_candidates(snapshot: Tuple[list, Dict[str, Any], Dict[str, str]],
                      existing_ids: set) -> Tuple[list, Dict[str, Any]]:
    """The one-shot legacy rows (``legacy_usage`` per deduplicated event,
    ``legacy_metadata`` call delta, ``legacy_delta`` cost delta) of a
    ``legacy_snapshot`` that the journal does not hold yet, and the completed
    watermark that records them."""
    legacy_rows, state, hashes = snapshot
    legacy_rows = [dict(row) for row in legacy_rows]
    candidates: list[Dict[str, Any]] = []
    seen: set[str] = set()
    imported_cost, usage_count = 0.0, 0
    for event in legacy_rows:
        line_no = int(event.pop("_legacy_line", 0) or 0)
        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        fingerprint = hashlib.sha256(json.dumps(
            event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        task_id = str(event.get("task_id") or "")
        raw_cost = event.get("cost")
        if raw_cost is None:
            raw_cost = usage.get("cost", usage.get("total_cost"))
        amount(raw_cost)  # Refuse nonfinite evidence before float coercion or a completed watermark.
        cost = _number(raw_cost)

        def legacy_int(name: str, *aliases: str) -> int:
            for candidate in (name, *aliases):
                value = event.get(candidate)
                if value in (None, ""):
                    value = usage.get(candidate)
                try:
                    return max(0, int(float(value or 0)))
                except (TypeError, ValueError):
                    continue
            return 0

        prompt, completion = legacy_int("prompt_tokens", "input_tokens"), legacy_int("completion_tokens", "output_tokens")
        provider = str(event.get("provider") or event.get("api_key_type") or "unknown")
        if cost == 0 and (prompt or completion) and provider != "local":
            cost = None  # legacy zero may mean unknown pricing, never "free"
        usage_count += 1
        if cost is not None:
            imported_cost += cost
        candidates.append({
            "kind": "legacy_usage", "attempt_id": f"legacy-{fingerprint[:24]}", "state": "settled",
            "model": str(event.get("model") or ""), "provider": provider, "cost_usd": cost,
            "cost_final": bool(cost is not None and not event.get("cost_estimated")),
            "reservation_upper_bound_usd": None, "prompt_tokens": prompt, "completion_tokens": completion,
            "cached_tokens": legacy_int("cached_tokens", "cache_read_input_tokens"),
            "cache_write_tokens": legacy_int("cache_write_tokens", "cache_creation_input_tokens"),
            "prompt_cache_ttl": str(event.get("prompt_cache_ttl") or usage.get("prompt_cache_ttl") or ""),
            "task_id": task_id, "root_task_id": str(event.get("root_task_id") or task_id),
            "parent_task_id": str(event.get("parent_task_id") or ""),
            "category": str(event.get("category") or "legacy"), "source": "legacy_llm_usage", "legacy_line": line_no,
        })
    legacy_calls = max(0, int(state.get("spent_calls") or state.get("calls") or 0))
    metadata_count = max(0, legacy_calls - usage_count)
    if metadata_count:
        identity = hashlib.sha256(f"legacy-metadata:{metadata_count}:{hashes.get('state.json', '')}".encode()).hexdigest()
        candidates.append({
            "kind": "legacy_metadata", "attempt_id": f"legacy-{identity[:24]}", "state": "unresolved",
            "model": "", "provider": "legacy", "reservation_upper_bound_usd": None,
            "ambiguous_call_count": metadata_count, "task_id": "", "root_task_id": "", "parent_task_id": "",
            "category": "legacy", "source": "legacy_state_call_delta",
        })
    amount(state.get("spent_usd"))
    state_spent = _number(state.get("spent_usd")) or 0.0
    delta = round(max(0.0, state_spent - imported_cost), 6)
    if delta:
        identity = hashlib.sha256(f"legacy-delta:{delta:.6f}:{hashes.get('state.json', '')}".encode()).hexdigest()
        candidates.append({
            "kind": "legacy_delta", "attempt_id": f"legacy-{identity[:24]}", "state": "settled",
            "model": "", "provider": "legacy", "cost_usd": delta, "cost_final": False,
            "reservation_upper_bound_usd": None, "task_id": "", "root_task_id": "", "parent_task_id": "",
            "category": "legacy", "source": "legacy_state_delta",
        })
    missing = [row for row in candidates if row["attempt_id"] not in existing_ids]
    watermark = {
        "completed": True, "completed_at": utc_now_iso(), "source_sha256": hashes,
        "legacy_baseline_source": "state.json", "legacy_baseline_spent_usd": state_spent,
        "legacy_baseline_spent_calls": legacy_calls, "legacy_usage_count": usage_count,
        "legacy_metadata_count": metadata_count, "legacy_delta_usd": delta,
        # The legacy schema has no trustworthy typed test/operator bit.
        "quarantined_test_operator_rows": 0, "test_operator_quarantine_policy": "typed_evidence_only_no_inference",
        "events_exceed_state_calls": max(0, usage_count - legacy_calls),
        "events_exceed_state_usd": round(max(0.0, imported_cost - state_spent), 6),
        "rows_appended": len(missing),
    }
    return missing, watermark


# ---- downgrade export ----

# Fields a reserved/dispatched predecessor row never carried: settlement and
# outcome facts that only the attempt's later rows wrote.
_SETTLEMENT_FIELDS = frozenset({
    "settle_reason", "reason", "cost_usd", "cost_final", "prompt_tokens", "completion_tokens",
    "cached_tokens", "cache_write_tokens", "prompt_cache_ttl", "effort_resolution", "processing", "speed",
    "service_tier", "cost_basis", "cost_evidence", "input_token_usage", "attempt_execution", "provider_token_receipt",
})
_DISPATCH_FIELDS = frozenset({
    "launch_state", "transport_outcome", "candidate_manifest_ref", "local_answer_owner_pid",
    "local_answer_consumer_id", "local_answer_task_attempt", "local_answer_owner_birth", "local_answer_owner_proof",
})
_STORE_ONLY_FIELDS = ("revision", "seq", "ts", "pre_compaction_seq")


def export_rows(header: Optional[Dict[str, Any]], aggregates: list, attempts: list,
                bindings: Dict[str, Dict[str, Any]]) -> list:
    """The journal rows an older release accepts, for the store's rows.

    ``aggregates`` are the imported ``usage_baseline_group`` rows (re-emitted
    under the stored ``usage_baseline`` header, so carried bindings survive);
    ``attempts`` are ``(row, ts_reserved, ts_dispatched)`` in first-row order.
    Each attempt becomes a minimal legal chain ending in its current row:
    reserved, then dispatched for an attempt that was sent, then the current
    row; a one-shot stays one row. The first predecessor of each root carries
    that root's stored binding verbatim, so the older release derives the same
    earliest binding. Sequence numbers are dense from 1. Work cap decisions
    are carried in the export's watermark, never on physical rows."""
    out: list[Dict[str, Any]] = []
    if aggregates:
        if header is None:
            raise UsageAccountingError("usage store holds aggregates without their baseline header")
        out.append({key: value for key, value in header.items() if key != "seq"})
        out.extend({**{key: value for key, value in row.items() if key not in _STORE_ONLY_FIELDS},
                    **({"ts": row["ts"]} if row.get("ts") else {})} for row in aggregates)
    bound_roots: set = set()
    for row, ts_reserved, ts_dispatched in attempts:
        current = {key: value for key, value in row.items() if key not in _STORE_ONLY_FIELDS}
        stamp = str(row.get("ts") or utc_now_iso())
        kind, state = str(current.get("kind") or "attempt"), str(current.get("state") or "")
        if is_one_shot_kind(kind) or state == "reserved":
            out.append({**current, "ts": stamp})
            continue
        base = {key: value for key, value in current.items()
                if key not in _SETTLEMENT_FIELDS and key not in _DISPATCH_FIELDS}
        reserved = {**base, "state": "reserved", "ts": ts_reserved or stamp}
        root = str(current.get("root_task_id") or "")
        if root and root not in bound_roots:
            bound_roots.add(root)
            binding = bindings.get(root)
            if binding is not None and any(cap in binding for cap in _BINDING_CAPS):
                reserved = {**{k: v for k, v in reserved.items() if k not in BINDING_KEYS}, **binding}
        out.append(reserved)
        if state != "released":
            dispatched = {key: value for key, value in current.items() if key not in _SETTLEMENT_FIELDS}
            if state != "dispatched":
                out.append({**dispatched, "state": "dispatched", "ts": ts_dispatched or stamp})
        out.append({**current, "ts": stamp})
    return [durable_literals({**row, "seq": index}) for index, row in enumerate(out, 1)]


def journal_bytes(rows: Sequence[Dict[str, Any]]) -> bytes:
    """The exact bytes ``write_journal`` writes for ``rows`` (an export records their identity first)."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows)


def write_journal(root: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Atomically write ``rows`` as the live journal; returns its identity."""
    payload = journal_bytes(rows)
    _write_bytes_atomic_fsync(root / LEDGER_REL, payload)
    return {"path": str(root / LEDGER_REL), "rows": len(rows), "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def journal_marker(rows: Sequence[Dict[str, Any]]) -> list:
    """``[compaction_epoch, last seq]`` exactly as an older release derives it."""
    epoch = next((int(row.get("compaction_epoch") or 0) for row in rows
                  if str(row.get("kind") or "") == "usage_baseline"), 0)
    return [epoch, len(rows)]
