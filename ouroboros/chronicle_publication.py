"""Chronicle write preparation and the prospective projection of a checked transaction."""
import json
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

from ouroboros.utils import utc_now_iso


_proposal = ContextVar("chronicle_publication_preview", default=None)


@contextmanager
def prospective_chronicle(root, changes):
    """An ordinary capture will read the proposed transaction, not append it."""
    pending = [item for item in changes if item.subject == "chronicle"]
    if not pending:
        yield
        return
    if len(pending) != 1:
        raise ValueError("A chronicle preview is exactly one checked transaction")
    change = pending[0]
    token = _proposal.set((Path(root).resolve(), int(change.before), json.loads(change.after)))
    try:
        yield
    finally:
        _proposal.reset(token)


def proposed_transaction(root):
    held = _proposal.get()
    return held[1:] if held is not None and held[0] == root.resolve() else None


@contextmanager
def writing(store, *, restoring=False):
    from ouroboros.nano_host_publication import host_admission
    from ouroboros.nano_publication import publication_fence

    with ExitStack() as locks:
        admission = None if restoring else store.admission or host_admission(store.data_root.resolve())
        if admission is None:
            # Initial legacy import still captures original memory even on a
            # small window. It cannot leave an earlier fit observation valid.
            locks.enter_context(publication_fence(store.data_root))
        checked = admission is not None and not locks.enter_context(admission.ordinary())
        yield (locks, admission) if checked else None, locks.enter_context(store._publishing())


def prepare(store, db, records, scan_state, *, encode):
    """Assign the actual record IDs/time once, ready to trial or append."""
    from ouroboros.chronicle_store import PublishResult, _mapping, _refuse

    if scan_state is not None and not isinstance(scan_state, dict):
        return _refuse("invalid", "scan_state is an object merged by top-level key")
    if not isinstance(records, (list, tuple)) or not all(isinstance(r, dict) for r in records):
        return _refuse("invalid", "records is a list of objects")
    if any(r.get("kind") == "activation" for r in records):
        active = store._state(db, "activation")
        if active:
            return PublishResult(True, "unchanged", store._get(db, active["id"]) or active)
    prepared, identities = [], {}
    for supplied in records:
        record = json.loads(encode(supplied))
        record["id"] = str(record.get("id") or uuid.uuid4().hex)
        record["room_id"] = str(record.get("room_id", ""))
        old = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
        old_record = json.loads(old[0]) if old else identities.get(record["id"])
        if old_record is None and "ts" in record and _mapping(record.get("metadata")).get("transfer") is not None:
            return _refuse("invalid", "the store dates a transfer's publication; its source's date is legacy_written_at")
        record.setdefault("ts", old_record["ts"] if old_record else utc_now_iso())
        if old_record is not None and old_record != record:
            return _refuse("invalid", "identity collision: this id already holds a different record",
                           conflict_ids=(record["id"],))
        if record["id"] not in identities:
            prepared.append(record)
            identities[record["id"]] = record
    new = [r for r in prepared if not db.execute("SELECT 1 FROM records WHERE id=?", (r["id"],)).fetchone()]
    scan = store._state(db, "scan", {})
    changed = scan_state is not None and {**scan, **scan_state} != scan
    last = prepared[-1] if prepared else None
    if not new and not changed:
        return PublishResult(True, "unchanged", store._get(db, last["id"]) if last else None)
    return {"kind": "transaction", "schema": 1, "records": new, "scan_state": scan_state}, last


def admitted_db(store, gate, before, tx, encode):
    """Release the original store lock; resume the same write only at the checked frontier."""
    from ouroboros.nano_publication import MemoryChange, NanoPublicationRefused

    locks, admission = gate
    change = MemoryChange(None, str(before).encode("ascii"), encode(tx).encode("utf-8"), subject="chronicle")
    locks.close()
    locks.enter_context(admission.publication(change))
    db = locks.enter_context(store._publishing())
    if store._state(db, "offset", 0) != before:
        raise NanoPublicationRefused("stale", "Chronicle changed before the checked append")
    db.execute("BEGIN EXCLUSIVE")
    return db


def prospective_snapshot(store, tail, pending, read):
    """Use the same projection as append/replay, under rollback only, never copying the journal.

    Snapshot reads are complete before unlocking; serialization and counting use
    the detached view. The derived index may catch up as for any ordinary reader.
    """
    from ouroboros.nano_publication import NanoPublicationRefused

    before, tx = pending
    with store._publishing() as db:
        if store._state(db, "offset", 0) != before:
            raise NanoPublicationRefused("stale", "Chronicle changed before candidate capture")
        db.execute("SAVEPOINT prospective_chronicle")
        try:
            store._project(db, tx)
            return read(db, tail)
        finally:
            db.execute("ROLLBACK TO prospective_chronicle")
            db.execute("RELEASE prospective_chronicle")
