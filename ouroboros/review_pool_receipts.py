"""Durable receipts of the review-lane -> review-pool migration, independent of the process.

The migration itself (``review_pool_migration.apply_at_read_seam``) is pure and runs in
whichever process reads an old document: the server, the launcher menu, the Colab kernel
that rewrites the Drive document, the UI's owner save before the supervisor generation
starts. Whichever of them SAVES the migrated document first replaces the pre-image on disk,
so the receipts are written by that saving process, before its write — never only by the
supervisor boot out of its own process memory, which a Colab kernel or a pre-supervisor
save never shares. Writer and boot choose by one rule: a migration gets receipts only when it
decides a document — the one a write replaces or saves, the one on disk when the boot runs.

Two receipts per migrated document (keyed by ``input_sha256``, the digest of the facts the
migration read):

* the snapshot file ``state/review_migrations/<ts>-slots-to-pool.json`` — the pre-image
  (lane and effort keys as read, every seat the lanes ran), the catalog after, the row
  report and the error; the PRIMARY, process-independent record. A later process finds it
  by the digest it carries, so the same document is never snapshotted twice;
* the record ``state.json:review_pool_migrations[<digest>]`` =
  ``{ts, snapshot, trigger, outcome, error, reported}`` — the owner-chat ledger the
  supervisor boot reads to send the ONE message. A writer whose supervisor state is not
  bound to this data root (the Colab kernel) leaves the record absent; the boot and the
  review-pool payload reconcile it from the snapshot file (:func:`known_receipts`).

Rollback is the owner's: the snapshot's ``before`` carries ``OUROBOROS_SUBAGENTS`` and
``OUROBOROS_REVIEWER_SLOTS`` exactly as read, so restoring those two keys from it into the
settings document returns the install to the pre-migration document (the retired lane
readers then migrate it again on the next read — the migration is a function of the
document, not of history).
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

log = logging.getLogger("ouroboros.review_pool_receipts")

SNAPSHOT_DIR = "review_migrations"
SNAPSHOT_SUFFIX = "-slots-to-pool.json"
STATE_KEY = "review_pool_migrations"

OUTCOME_CONVERTED = "converted"  # the owner's authored lanes became reviewer rows
OUTCOME_FACTORY = "factory"  # no authored lanes: the factory reviewer rows run
OUTCOME_ERROR = "error"  # refused: the lane keys stay in the document for the owner's save
_AUTHORED_SLOT_STATES = frozenset({"direct", "referenced", "mixed"})

SOURCE_DOCUMENT = "document"  # the migrated catalog IS the document's catalog now
SOURCE_ENVIRONMENT = "environment"  # a catalog the environment carries runs instead of the minted rows
SOURCE_ERROR = "error"  # the refusal still describes the document: lane keys retained, no pool from them
SOURCE_HISTORY = "history"  # the document changed since (the owner's save): the receipt is history


def snapshot_dir(data_dir: Any) -> pathlib.Path:
    return pathlib.Path(data_dir) / "state" / SNAPSHOT_DIR


def outcome_kind(snapshot: Mapping[str, Any]) -> str:
    """``converted`` / ``factory`` / ``error`` for a snapshot (a no-op is never recorded)."""
    if snapshot.get("error"):
        return OUTCOME_ERROR
    before = snapshot.get("before") or {}
    return OUTCOME_CONVERTED if before.get("slots_state") in _AUTHORED_SLOT_STATES else OUTCOME_FACTORY


def record_for(snapshot: Mapping[str, Any], relpath: str, ts: Any) -> Dict[str, Any]:
    """The ``state.json`` record of one snapshot; ``reported`` is set by the boot that told the owner."""
    before = snapshot.get("before") or {}
    return {"ts": str(ts or ""), "snapshot": relpath, "trigger": str(before.get("trigger") or ""),
            "outcome": outcome_kind(snapshot), "error": str(snapshot.get("error") or ""), "reported": None}


def outcome_from_snapshot(snapshot: Mapping[str, Any]) -> Any:
    """A snapshot read back as the ``MigrationOutcome`` it recorded (the owner message and the
    environment check take an outcome); ``catalog_after`` is re-serialized in the seam's spelling."""
    from ouroboros.review_pool_migration import SUBAGENTS_KEY, MigrationOutcome

    before = snapshot.get("before") or {}
    after = (snapshot.get("after") or {}).get(SUBAGENTS_KEY)
    return MigrationOutcome(
        input_sha256=str(snapshot.get("input_sha256") or ""), catalog_state=str(before.get("catalog_state") or ""),
        slots_state=str(before.get("slots_state") or ""), snapshot=dict(snapshot),
        catalog_after=None if after is None else json.dumps(after, ensure_ascii=False, separators=(",", ":")),
        error=str(snapshot.get("error") or ""), noop=bool(snapshot.get("noop")), trigger=str(before.get("trigger") or ""),
    )


def read_snapshots(data_dir: Any) -> Dict[str, Tuple[pathlib.Path, Dict[str, Any]]]:
    """Every readable snapshot under ``data_dir`` by document digest (a duplicate digest keeps the newest file)."""
    found: Dict[str, Tuple[pathlib.Path, Dict[str, Any]]] = {}
    directory = snapshot_dir(data_dir)
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob(f"*{SNAPSHOT_SUFFIX}")):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("review pool migration snapshot unreadable, skipped: %s", path, exc_info=True)
            continue
        digest = str(snapshot.get("input_sha256") or "") if isinstance(snapshot, dict) else ""
        if digest:
            found[digest] = (path, snapshot)
    return found


def load_snapshot(data_dir: Any, record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The snapshot a record points at, or ``None`` when the file is gone or unreadable."""
    rel = str(record.get("snapshot") or "")
    if not rel:
        return None
    try:
        snapshot = json.loads((pathlib.Path(data_dir) / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return snapshot if isinstance(snapshot, dict) else None


def migration_records(state: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """The ``state.json:review_pool_migrations`` ledger: ``input_sha256 -> record``."""
    if state is None:
        from supervisor.state import load_state

        state = load_state()
    records = (state or {}).get(STATE_KEY)
    return {str(k): dict(v) for k, v in records.items() if isinstance(v, dict)} if isinstance(records, dict) else {}


def write_record(update_state: Callable[..., Any], digest: str, record: Mapping[str, Any]) -> None:
    def _mark(st: dict) -> None:
        seen = st.get(STATE_KEY)
        seen = dict(seen) if isinstance(seen, dict) else {}
        seen[digest] = dict(record)
        st[STATE_KEY] = seen

    update_state(_mark)


def _bound_state_ledger(data_dir: Any) -> Optional[Tuple[Callable[[], Any], Callable[..., Any]]]:
    """``(load_state, update_state)`` of this process's supervisor state when it is bound to
    ``data_dir`` and exists, else ``None``: a writer for another data root (the Colab kernel
    rewriting the Drive document) leaves the record to the boot's reconciliation."""
    try:
        from supervisor import state

        state_path = pathlib.Path(state.STATE_PATH)
        if state_path.resolve().parent.parent != pathlib.Path(data_dir).resolve() or not state_path.exists():
            return None
        return state.load_state, state.update_state
    except Exception:
        return None


def _new_snapshot_path(data_dir: Any) -> Tuple[str, pathlib.Path]:
    """``(ts, state/review_migrations/<ts>-slots-to-pool.json)`` for a NEW snapshot: the current
    UTC second; a second migration landing in the same second waits for the next one rather than
    overwriting a sibling."""
    while True:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = snapshot_dir(data_dir) / f"{ts}-slots-to-pool.json"
        if not path.exists():
            return ts, path
        time.sleep(0.2)


def _relpath(path: pathlib.Path, data_dir: Any) -> str:
    return path.relative_to(pathlib.Path(data_dir)).as_posix()


def persist_receipts(data_dir: Any, *, outcomes: Optional[Tuple[Any, ...]] = None) -> Dict[str, Dict[str, Any]]:
    """Give the non-noop ``outcomes`` (default: every migration this process computed) their
    durable receipts under ``data_dir``: the snapshot file when no
    file carries the document digest yet, and the state record when this process's supervisor
    state is bound to ``data_dir`` and the digest has no record. Returns the records it knows
    (written or found). Never raises — a receipt failure is logged and the next writer or the
    boot retries from ``migrations_seen()``; it must not block a save."""
    try:
        return _persist_receipts(pathlib.Path(data_dir), outcomes)
    except Exception:
        log.warning("review pool migration receipts could not be written under %s", data_dir, exc_info=True)
        return {}


def _document_on_disk(settings_path: Any) -> Optional[Dict[str, Any]]:
    """The settings document at ``settings_path`` as the read seam types it (``config.coerce_settings_raw``),
    so its digest is the one the seam computed for it; ``None`` when no readable document is there."""
    from ouroboros.config import coerce_settings_raw

    try:
        raw = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return coerce_settings_raw(raw) if isinstance(raw, dict) else None


def document_as_read(settings_path: Any, running: Mapping[str, Any]) -> Mapping[str, Any]:
    """The settings DOCUMENT a receipt is judged against (:func:`outcome_decides_document`): the file at
    ``settings_path`` as the read seam leaves it — ``config.normalize_settings_raw``, the migration applied
    (a replay by digest), NO environment merged — else ``running`` when there is no readable file (the
    defaults the install runs; the boot's own rule in :func:`persist_boot_receipts`).

    What RUNS (``config.load_settings``) differs from this document in exactly one place: the catalog the
    environment carries wins over rows the seam minted for a document that authored none
    (``review_pool_migration.environment_overridable_keys``). The process environment itself cannot witness
    that — ``config.apply_settings_to_env`` projects the effective catalog back into it before the boot
    reports, and the projection carries only the live settings keys, so a lane value a refused migration
    kept in the document is absent from it (VD3-06). The document on disk is the one witness of both."""
    from ouroboros.config import normalize_settings_raw

    raw = _document_on_disk(settings_path)
    return running if raw is None else normalize_settings_raw(raw)


def _deciding_outcomes(on_disk: Optional[Mapping[str, Any]], document: Mapping[str, Any]) -> Tuple[Any, ...]:
    """The migrations this process computed whose input is ``on_disk`` (by its exact digest) or whose
    result is ``document`` (:func:`outcome_decides_document`)."""
    from ouroboros.review_pool_migration import input_sha256, migrations_seen

    replaced = None if on_disk is None else input_sha256(on_disk)
    return tuple(o for o in migrations_seen() if o.input_sha256 == replaced or outcome_decides_document(o, document))


def persist_write_receipts(data_dir: Any, written: Mapping[str, Any], settings_path: Any) -> Dict[str, Dict[str, Any]]:
    """The receipts a settings write owes BEFORE it lands (the persistence prologue, the Colab
    writer): the migration of the document it replaces — ``settings_path`` as the read seam types
    it (``config.coerce_settings_raw``), found by its exact digest — and any migration whose result
    it saves (:func:`outcome_decides_document`). Not every migration this process computed: another
    data root's document or a draft the wizard read is no document this write replaces or saves, its
    receipt would describe rows that never ran, and each extra snapshot costs a UTC second
    (:func:`_new_snapshot_path`) the writer waits out under the settings lock. Never raises."""
    try:
        owed = _deciding_outcomes(_document_on_disk(settings_path), written)  # no pre-image: none of its own replaced
    except Exception:
        log.warning("review pool migration receipts for a write under %s could not be chosen", data_dir, exc_info=True)
        return {}
    return persist_receipts(data_dir, outcomes=owed)


def persist_boot_receipts(data_dir: Any, settings_path: Any, running: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The receipts the supervisor boot owes: the migrations that decide the document ON DISK now,
    chosen as for a write of that document — its input is the file (the server read the N-1 document
    no save has replaced) or its result is (a save landed the migrated catalog) — and, with no
    readable file, those whose result is ``running``: the defaults the install runs. Not every
    migration this process computed: the factory rows minted for the defaults the server read before
    the wizard saved a catalog of its own, or a draft it normalized, decide no document on disk, and
    their receipts would tell the owner that rows run which never ran. Never raises."""
    try:
        on_disk = _document_on_disk(settings_path)
        owed = _deciding_outcomes(on_disk, running if on_disk is None else on_disk)
    except Exception:
        log.warning("review pool migration receipts for the boot under %s could not be chosen", data_dir, exc_info=True)
        return {}
    return persist_receipts(data_dir, outcomes=owed)


def _persist_receipts(data_dir: pathlib.Path, outcomes: Optional[Tuple[Any, ...]]) -> Dict[str, Dict[str, Any]]:
    from ouroboros.review_pool_migration import migrations_seen
    from ouroboros.utils import atomic_write_json

    pending = [o for o in (migrations_seen() if outcomes is None else outcomes) if not o.noop]
    if not pending:
        return {}
    files = read_snapshots(data_dir)
    ledger = _bound_state_ledger(data_dir)
    records = migration_records(ledger[0]()) if ledger else {}
    known: Dict[str, Dict[str, Any]] = {}
    for outcome in pending:
        digest = str(outcome.input_sha256)
        if digest in files:
            path, snapshot = files[digest]
        else:
            snapshot_dir(data_dir).mkdir(parents=True, exist_ok=True)
            ts, path = _new_snapshot_path(data_dir)
            snapshot = {**outcome.snapshot, "ts": ts}
            atomic_write_json(path, snapshot, trailing_newline=True)
            files[digest] = (path, snapshot)
            log.info("review pool migration snapshot written: %s", _relpath(path, data_dir))
        record = records.get(digest) or record_for(snapshot, _relpath(path, data_dir), snapshot.get("ts"))
        if ledger is not None and digest not in records:
            write_record(ledger[1], digest, record)
            records[digest] = record
        known[digest] = record
    return known


def known_receipts(data_dir: Any, records: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """The state ledger (``records``, else the one read from ``state.json``) overlaid with the
    snapshot files no record names yet (another process wrote them): every receipt a reader
    can know, without writing anything."""
    known = migration_records() if records is None else {str(k): dict(v) for k, v in records.items()}
    for digest, (path, snapshot) in read_snapshots(data_dir).items():
        if digest not in known:
            known[digest] = record_for(snapshot, _relpath(path, data_dir), snapshot.get("ts"))
    return known


def reconcile_records(data_dir: Any, state: Mapping[str, Any], update_state: Callable[..., Any]) -> Dict[str, Dict[str, Any]]:
    """Write the state record for every snapshot file another process left without one; returns the ledger."""
    records = migration_records(state)
    for digest, record in known_receipts(data_dir, records).items():
        if digest not in records:
            write_record(update_state, digest, record)
            records[digest] = record
    return records


def mark_reported(update_state: Callable[..., Any], digest: str, record: Dict[str, Any]) -> None:
    from ouroboros.utils import utc_now_iso

    record["reported"] = utc_now_iso()
    write_record(update_state, digest, record)


def close_as_history(outcome: Any, document: Mapping[str, Any], update_state: Callable[..., Any], digest: str,
                     record: Dict[str, Any]) -> bool:
    """Whether the unreported ``record`` describes an EARLIER document than ``document`` (the document
    as read, :func:`document_as_read`): its ``outcome`` no longer decides it — a file-less start
    receipted the factory rows before any owner chat was bound, then the wizard saved a catalog of its
    own. Such a record is marked reported with a log line and no owner message; the snapshot and the
    record stay on disk and in the ledger as history. ``False`` leaves a current record to its message."""
    if outcome_decides_document(outcome, document):
        return False
    log.info("review pool migration receipt %s: an earlier document, closed without an owner message", record.get("snapshot"))
    mark_reported(update_state, digest, record)
    return True


def outcome_decides_document(outcome: Any, settings: Mapping[str, Any]) -> bool:
    """Whether ``outcome`` (a ``review_pool_migration.MigrationOutcome``) decided the document
    ``settings`` shows — the ONE predicate the owner's receipt and the model's ``## Review``
    block share. A refusal rewrote nothing, so the lanes key and the catalog read back exactly
    as its snapshot's ``before`` recorded them; a finished migration's ``catalog_after`` IS the
    document's catalog and its lanes key is gone. A no-op decided nothing. Any other document
    — the owner's repairing save, another task's snapshot, an environment catalog that won over
    the minted rows — carries none of this outcome's facts, however recent the outcome is."""
    from ouroboros.review_pool_migration import REVIEWER_SLOTS_KEY, SUBAGENTS_KEY

    def as_read(value: Any) -> str:
        return value if isinstance(value, str) else ("" if value is None else json.dumps(value))

    if getattr(outcome, "noop", False):
        return False
    before = (getattr(outcome, "snapshot", None) or {}).get("before") or {}
    if getattr(outcome, "error", ""):
        return all(as_read(settings.get(key)) == as_read(before.get(key)) for key in (REVIEWER_SLOTS_KEY, SUBAGENTS_KEY))
    after = str(getattr(outcome, "catalog_after", None) or "")
    return bool(after) and REVIEWER_SLOTS_KEY not in settings and as_read(settings.get(SUBAGENTS_KEY)) == after


def environment_catalog_in_force(outcome: Any, document: Mapping[str, Any],
                                 running: Mapping[str, Any]) -> Optional[str]:
    """The catalog the ENVIRONMENT carries when it runs in place of the rows the migration
    minted for ``outcome``'s document, or ``None`` when the minted rows run (or nothing was
    minted, or ``outcome`` no longer describes ``document``). ``document`` is the settings
    document as read (:func:`document_as_read`: the file as the seam leaves it, no environment),
    ``running`` what runs (the environment merged over it). The seam's rows are a default only
    where the document authored no lanes and saved no catalog (``environment_overridable_keys``
    over the snapshot's ``before``), and the settings reader lets the environment's catalog win
    exactly there — so a catalog in force that is not the minted one is the environment's ONLY
    while the minted rows are still the document's own catalog. A document the owner has since
    saved with a catalog of its own (the wizard after a file-less start) is a different document:
    its catalog is the owner's, not the environment's, and this says nothing about it."""
    from ouroboros.review_pool_migration import REVIEWER_SLOTS_KEY, SUBAGENTS_KEY, environment_overridable_keys

    minted = str(outcome.catalog_after or "")
    before = (outcome.snapshot or {}).get("before") or {}
    as_read = {key: before[key] for key in (REVIEWER_SLOTS_KEY, SUBAGENTS_KEY) if before.get(key) is not None}
    if not minted or SUBAGENTS_KEY not in environment_overridable_keys(as_read):
        return None
    if not outcome_decides_document(outcome, document):
        return None
    in_force = str(running.get(SUBAGENTS_KEY) or "")
    return None if in_force == minted else in_force


def migration_payload(document: Mapping[str, Any], data_dir: Any, records: Optional[Mapping[str, Any]] = None,
                      *, running: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The review-pool payload's ``migration`` fact: the newest receipt that still describes the
    settings ``document`` (:func:`document_as_read`) — ``{snapshot, reported, trigger, outcome,
    error, source}`` — else the newest receipt at all with ``source: history``; ``None`` when
    there is no receipt. ``running`` is what runs beside that document (default: the document
    itself); it decides ``source: environment`` alone (:func:`environment_catalog_in_force`),
    never whether a receipt is current."""
    running = document if running is None else running
    receipts = sorted(known_receipts(data_dir, records).items(), key=lambda item: str(item[1].get("ts") or ""),
                      reverse=True)
    if not receipts:
        return None
    chosen, source = receipts[0], SOURCE_HISTORY
    for digest, record in receipts:
        snapshot = load_snapshot(data_dir, record)
        outcome = outcome_from_snapshot(snapshot) if snapshot else None
        if outcome is None or not outcome_decides_document(outcome, document):
            continue
        chosen = (digest, record)
        if outcome.error:
            source = SOURCE_ERROR
        else:
            source = (SOURCE_DOCUMENT if environment_catalog_in_force(outcome, document, running) is None
                      else SOURCE_ENVIRONMENT)
        break
    record = chosen[1]
    return {"snapshot": str(record.get("snapshot") or ""), "reported": bool(record.get("reported")),
            "trigger": str(record.get("trigger") or ""), "outcome": str(record.get("outcome") or ""),
            "error": str(record.get("error") or ""), "source": source}


__all__ = [
    "STATE_KEY",
    "close_as_history",
    "document_as_read",
    "environment_catalog_in_force",
    "known_receipts",
    "load_snapshot",
    "mark_reported",
    "migration_payload",
    "migration_records",
    "outcome_decides_document",
    "outcome_from_snapshot",
    "persist_boot_receipts",
    "persist_receipts",
    "persist_write_receipts",
    "read_snapshots",
    "reconcile_records",
    "record_for",
    "snapshot_dir",
]
