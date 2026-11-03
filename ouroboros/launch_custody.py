"""One selection of what service launches exist and what each still owes, for every custody consumer.

A launch's debt lives in three stores that lag each other by design: its drive's ledger, the installation's ownership
document, and the pending copies a stop writes when it cannot take the ledger lock. No store is the whole, so every
consumer combines them by LAUNCH before it filters by owner: a stale copy cannot hide what the ledger names, and a
compacted ledger cannot hide what only a copy names. A store that cannot be read is reported, never read as empty. No
new store: this reads the two that exist, and it neither signals nor reads a process.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, NamedTuple

from ouroboros import process_custody as pc

log = logging.getLogger(__name__)


class LaunchCustody(NamedTuple):
    readable: bool  # the ledger of the drive asked about
    rows: List[Dict[str, Any]]  # its latest row per key, then every launch only another store or drive names
    copies: Dict[int, List[Dict[str, Any]]]  # {id(row): what the launch's ownership copy names}
    unknown: set  # id(row) of each launch whose stores could not be read whole
    blind: str  # why a launch no row here names may exist (the set unreadable, a damaged copy nothing attributes)
    previous: bytes  # the ledger bytes its rows came from, for that ledger's compaction
    records: List[Dict[str, Any]]  # the ownership set as read


def read_custody(drive_root: Any, *, strict: bool = True, every_drive: bool = False) -> LaunchCustody:
    """A launch's row is its own drive's latest (one read per drive) or, once that is gone, its ownership copy's; what
    the copy names beside it comes with it (``process_custody._launch_target``). Rows are as stored, so a sweep's
    survivors still match its ledger. Without ``every_drive`` a launch of another drive stays out: a sweep compacts
    only this drive's ledger. What cannot be read makes its launch ``unknown``; what no row can attribute (a damaged
    copy of a launch nothing else names, an unreadable set) makes the selection ``blind`` for every owner."""
    from ouroboros.owned_shutdown import _ledgered_launch, owned_records

    here = pathlib.Path(drive_root).resolve(strict=False)
    readable, stored, previous = pc._read_ledger_records(pathlib.Path(drive_root), strict=strict)
    rows = list(stored)
    ledgers = {here: (readable, {pc.custody_key(row): row for row in rows})}
    copies: Dict[int, List[Dict[str, Any]]] = {}
    unknown: set = set()
    blind = ""
    records: List[Dict[str, Any]] = []
    damaged: set = set()
    try:
        records = owned_records(drive_root, strict=True, only=set(), damaged=damaged)
    except Exception as exc:
        log.warning("ownership set of %s could not be read: %s", drive_root, exc)
        blind = "the ownership set cannot be read: it may name more launches"
        unknown.update(id(row) for row in rows if pc.custody_key(row)[0] == "launch")
    for record in records:
        if not _ledgered_launch(record):
            continue
        copy = record["ledger_entry"]
        key = pc.custody_key(copy)
        local = key in ledgers[here][1]
        home = here if local else pathlib.Path(record.get("drive_root") or here).resolve(strict=False)
        if home != here and not every_drive:
            continue
        if home not in ledgers:
            found, other, _bytes = pc._read_ledger_records(home, strict=strict)
            ledgers[home] = (found, {pc.custody_key(row): row for row in other})
        found, by_key = ledgers[home]
        row = by_key.get(key, copy)
        if not local:
            rows.append(row)
        copies[id(row)] = pc._copy_names(copy)
        if not found:
            unknown.add(id(row))
    launches = {pc.custody_record_id(row): row for row in rows if pc.custody_key(row)[0] == "launch"}
    for record_id in damaged:
        if record_id in launches:
            unknown.add(id(launches[record_id]))
        elif record_id.startswith("launch-"):
            blind = f"a pending ownership copy ({record_id}) cannot be read, and no other store says whose launch it is"
    return LaunchCustody(readable, rows, copies, unknown, blind, previous, records)


def unsettled_rows(drive_root: Any, owners: Any) -> List[tuple]:
    """What ``owners``' launches and older rows may still run: ``(row, "live" | "unknown")``. Read only, never a stop.
    Raises when the drive's ledger cannot be read, or when nothing else answers and a launch may exist that no row
    attributes: unavailable is never "none"."""
    custody = read_custody(drive_root, every_drive=True)
    if not custody.readable:
        raise OSError("process_custody_unreadable")
    found = unsettled_selection(drive_root, [row for row in custody.rows if str(row.get("owner_task") or "") in owners],
                                custody)
    if custody.blind and not found:
        raise OSError(f"process_custody_unknown: {custody.blind}")
    return found


def unsettled_selection(drive_root: Any, entries: List[Dict[str, Any]], custody: LaunchCustody) -> List[tuple]:
    """The rows of a selection whose processes may still run. A launch row is judged by its whole recorded reach in ONE
    census for all of them (``process_scope.observe_owned_processes``): an owed or marker-only member keeps it live
    after its leader died, and a reused leader PID is not the launch. What could not be read leaves no launch settled.
    An older row keeps its leader PID or surviving group."""
    from ouroboros.platform_layer import pid_is_alive as alive  # the readers' own call-time lookup, as before
    from ouroboros.process_scope import observe_owned_processes

    launches = [entry for entry in entries if pc._owned_service_row(entry)]
    states: Dict[Any, str] = {}
    if launches:
        states = observe_owned_processes([pc._launch_target(entry, custody.copies.get(id(entry))) for entry in launches],
                                         spared=pc.spared_daemon_roots(drive_root))
        states = {key: "unknown" if state == "settled" and key in custody.unknown else state
                  for key, state in states.items()}
    found = []
    for entry in entries:
        if id(entry) in states:
            if states[id(entry)] != "settled":
                found.append((entry, states[id(entry)]))
        elif alive(int(entry.get("pid") or 0)) or pc._service_group_survives_leader(entry):
            found.append((entry, "live"))
    return found


def name_ledger_debt(root: Any, timeout_sec: float) -> Dict[str, Any]:
    """Debt only the canonical ledger names (its ownership writes failed, or its row landed after its record retired)
    joins the set, so an exit's one stop ends it and its report counts it. One read; normally nothing to add. Returns
    what the exit cannot count, empty otherwise: ``records``, launches whose copy no store took either (their ledger
    rows keep them for the next start's sweep), and ``error``, why the selection could not be read whole."""
    from ouroboros.owned_shutdown import record_ledgered_process

    unwritten: List[str] = []
    try:
        custody = read_custody(root)
        named = {record.get("record_id") for record in custody.records}
        for row in [] if custody.blind else custody.rows:
            if row.get("known_members") and pc._owned_service_row(row) and pc.custody_record_id(row) not in named:
                if not record_ledgered_process(root, row, timeout_sec=timeout_sec):
                    unwritten.append(pc.custody_record_id(row))
                timeout_sec = 0.0  # one wait for the lock per exit: the rest go straight to their pending copy
        error = custody.blind or ("" if custody.readable else "the canonical process ledger cannot be read")
    except Exception as exc:
        log.warning("Debt only the ledger names was not read at this exit; the next start's sweep ends it", exc_info=True)
        error = f"{type(exc).__name__}: {exc}"[:200]
    return {key: value for key, value in (("records", unwritten), ("error", error)) if value}
