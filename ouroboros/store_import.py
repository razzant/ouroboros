"""Publish/import/export current facts; ordinary boots only open published stores.

First import builds a FULL-synchronous sibling, publishes by rename and directory
fsync, then stamps the generation. Reads never import; a missing stamped store is
lost. Only a completed export permits a full rebuild. Boot reads metadata and the
NULL-flag index; child headers trigger only required WAL handoffs, without frozen-result reads. A
changed/unknown previous body schedules ``legacy_reconcile`` once per observation; the maintenance job
or ``python -m ouroboros.store_import --reconcile-legacy [--root PATH]`` runs it.

Reconciliation retains the per-file size/mtime/hash baseline: changed files win,
deletions quarantine the SQL version, SQL-only changes survive, conflicts retain
the previous SQL document. Same-size/same-mtime edits and set-file-only changes
remain outside this legacy detector's contract. Manual edits require the CLI.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import pathlib
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import obligations as o
from ouroboros import store, store_journal, store_task_cards
from ouroboros.store import (
    GENERATION, GENERATION_KEY, IMPORT_LOCK_NAME, LOCK_NAME, PUBLISHED, TIER_ENFORCED, TIER_NAME,
    StoreLost, Txn, document_text, supervisor_row, utc_stamp,
)
from ouroboros.task_custody import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR, own_child_drives
from ouroboros.task_result_schema import (
    TASK_RESULT_QUARANTINE_DIR, emit_quarantine_event, quarantine_task_result, stamp_task_result_schema,
    task_result_schema_refusal,
)
from ouroboros.task_results import task_result_path, task_results_dir
from ouroboros.utils import append_jsonl, atomic_write_json, machine_json, utc_now_iso, write_text_atomic

Record = Tuple[str, Dict[str, Any]]


def _null_flags(columns) -> str:
    return " OR ".join(f"{column} IS NULL" for column in store.FLAG_COLUMNS.values() if not columns.get(column)) or "0"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _own_children(root: pathlib.Path, *, scan_results: bool = True) -> List[pathlib.Path]:
    """The own child drives of every task ONE root knows (result files and drive directories)."""
    ids = {path.stem for path in task_results_dir(root, create=False).glob("*.json")} if scan_results else set()
    for base in (root / HEADLESS_TASKS_DIR, root / TASK_DRIVES_DIR):
        if base.is_dir():
            ids.update(child.name for child in base.iterdir() if child.is_dir() and not child.is_symlink())
    drives: List[pathlib.Path] = []
    for task_id in sorted(ids):
        with contextlib.suppress(ValueError):
            drives.extend(own_child_drives(root, task_id))
    return drives


def descendant_drives(root: Any, *, scan_results: bool = True) -> List[pathlib.Path]:
    """Every own child drive down the tree (a child's drive names its own children the same
    way), parents before children: the one walk the import, the switch, the export, the
    retirement and the recovery all use."""
    found: List[pathlib.Path] = []
    pending = [pathlib.Path(root)]
    while pending:
        children = _own_children(pending.pop(0), scan_results=scan_results)
        found.extend(children)
        pending.extend(children)
    return found


def _mount_identity(directory: pathlib.Path) -> dict:
    """Device and enclosing mount, including Linux bind mounts (platform substrate)."""
    from ouroboros.platform_layer import is_mount_point

    directory = directory.resolve()
    mount = directory
    while mount.parent != mount and not is_mount_point(mount):
        mount = mount.parent
    return {"st_dev": directory.stat().st_dev, "mount_point": str(mount)}


def _lock_tier(root: pathlib.Path, recorded: Any = None) -> dict:
    from ouroboros import platform_layer as platform

    identity = _mount_identity(root / "state")
    lock = root / "state" / LOCK_NAME
    known = isinstance(recorded, dict) and all(recorded.get(key) == value for key, value in identity.items())
    if known and recorded.get("tier") in (TIER_ENFORCED, TIER_NAME):
        platform.kernel_file_locks_enforced(lock, known=(recorded["tier"] == TIER_ENFORCED, recorded.get("errno")))
        return recorded
    enforced = platform.kernel_file_locks_enforced(lock, known=(None, None))
    verified, refused = platform._KERNEL_LOCK_TIER.get(os.path.realpath(str(lock.parent)), (None, None))
    # A directory where no probe file could be created is still undecided. Do not
    # persist its identity as proof: a later open must retry the actual probe.
    return {"tier": TIER_ENFORCED if enforced else TIER_NAME,
            **(identity if verified is not None else {}), "errno": refused}


def _schedule_legacy(root: pathlib.Path, last: dict, job: dict, body_sha: str) -> bool:
    """Read the previous body's stamp BEFORE supervisor git bootstrap overwrites it.
    server._run_supervisor calls require_published before _bootstrap_supervisor_repo.
    One reconcile per observation, the observation being the stamp's change: a bootstrap that
    never stamps ``current_sha`` (the local-dev start of Docker images, Colab and source checkouts)
    shows every boot the same unknown or stale stamp, and a job already recorded for it, finished
    included, is not scheduled again; once a boot saw another stamp (``last_observed``, written only
    when it changes), a return to an older body's stamp is a new excursion and is reconciled."""
    from ouroboros import maintenance_jobs

    try:
        previous = json.loads((root / "state/state.json").read_text(encoding="utf-8")).get("current_sha")
    except (OSError, ValueError, AttributeError):
        previous = None
    reason = ("state_unreadable" if not isinstance(previous, str) or not previous else
              "last_store_body_missing" if not last.get("sha") else
              "body_changed" if previous != last["sha"] else "")
    seen = job.get("last_observed", job.get("observed_sha"))
    if not reason:
        if job.get("observed_sha") and seen != previous:  # the current body's own stamp after an older
            maintenance_jobs.record(root, "legacy_reconcile", last_observed=previous)  # one: its return is new
        return job.get("state") in {"pending", "running"}
    if job.get("state") in {"pending", "running", "failed", "done"} and job.get("observed_sha") == previous == seen:
        return job.get("state") in {"pending", "running"}
    maintenance_jobs.record(root, "legacy_reconcile", state="pending", reason=reason,
                            observed_sha=previous, last_observed=previous, body_sha=body_sha, cursor=None, done=0,
                            total=None, error=None, waiting_on=None)
    return True


def note_current_stamp(root: Any) -> None:
    """After the supervisor's bootstrap stamped ``current_sha`` (this body's own): the legacy job sees
    that observation now, so an older body that runs next and stamps its SHA again is a new excursion
    even when no boot of this body read the stamp in between. Writes only when it changes."""
    from ouroboros import maintenance_jobs

    root = pathlib.Path(root)
    try:
        stamp = json.loads((root / "state/state.json").read_text(encoding="utf-8")).get("current_sha")
        job = maintenance_jobs.job(root, "legacy_reconcile") or {}
    except (OSError, ValueError, AttributeError, store.StoreError):
        return  # an unreadable stamp or store observes nothing
    if (isinstance(stamp, str) and stamp and job.get("observed_sha")
            and job.get("last_observed", job.get("observed_sha")) != stamp):
        maintenance_jobs.record(root, "legacy_reconcile", last_observed=stamp)


def ensure_current_store(root: Any, *, with_children: bool = True, rederive: bool = False,
                         body_sha: Optional[str] = None, initialize_journal: bool = False) -> Dict[str, Any]:
    """Open a published root in one read transaction. Import absent roots once.
    ``body_sha`` runs the legacy detector; only ``initialize_journal`` authorizes the boot's journal
    selection before producers start, including compatibility of existing child stores.
    ``rederive`` is the explicit historical repair, never an ordinary startup."""
    root = pathlib.Path(root)
    path = store.store_path(root)
    started = time.monotonic()
    with store.process_lock(path):
        report, quarantined = _ensure(root, path, rederive=rederive, body_sha=body_sha,
                                      initialize_journal=initialize_journal)
    report["opened_ms"] = round((time.monotonic() - started) * 1000, 3)
    if initialize_journal and with_children:
        store_journal.prepare_children(root)
    emit_quarantine_event(root, quarantined)
    report["drives"] = {}
    importing = report.pop("_import_job", {}) or {}
    if with_children and (report["status"] == "completed" or rederive or importing.get("state") in {"pending", "running", "failed"}):
        from ouroboros import maintenance_jobs

        with store.name_lock(root, "current_store_import.lock", timeout_sec=600.0):
            importing = maintenance_jobs.job(root, "current_store_import") or {}
            cursor, done = importing.get("cursor") or ".", int(importing.get("done") or 1)
            derive_children = rederive or bool(importing.get("rederive"))
            drive = root
            try:
                maintenance_jobs.record(root, "current_store_import", state="running", cursor=cursor, done=done,
                                        error=None, waiting_on=None)
                drives = sorted(descendant_drives(root), key=lambda path: path.relative_to(root).as_posix())
                maintenance_jobs.record(root, "current_store_import", total=1 + len(drives))
                for drive in drives:
                    relative = drive.relative_to(root).as_posix()
                    if relative <= cursor:
                        continue
                    report["drives"][str(drive)] = ensure_current_store(drive, with_children=False, rederive=derive_children)["status"]
                    done += 1
                    maintenance_jobs.record(root, "current_store_import", cursor=relative, done=done)
                maintenance_jobs.record(root, "current_store_import", state="done")
            except BaseException as exc:
                maintenance_jobs.record(root, "current_store_import",
                                        state="failed" if isinstance(exc, Exception) else "pending",
                                        error=f"{type(exc).__name__}: {exc}", waiting_on={"drive": str(drive)})
                raise
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    supervisor_row(root, "current_store_import", {"phase": report["status"],
                                                  **{key: value for key, value in report.items() if key != "drives"}})
    return report


def _ensure(root: pathlib.Path, path: pathlib.Path, *, rederive: bool = False,
            body_sha: Optional[str] = None, initialize_journal: bool = False) -> Tuple[Dict[str, Any], list]:
    from ouroboros.platform_layer import kernel_file_locks_enforced
    from ouroboros.startup_migrations import stamp

    state, header = store.probe(root)
    if state == "present":
        if header == TIER_NAME:
            # The header already requires the name protocol. Take that lock for the
            # initial snapshot; only its recorded mount identity can authorize reuse.
            kernel_file_locks_enforced(root / "state" / LOCK_NAME, known=(False, None))
        if initialize_journal and store.wal_file(path):
            # The boot leaves a WAL file this runtime must not share, under the tier it is used with,
            # before its own read refuses it; choosing WAL waits for a published phase below.
            _select_journal(root, path, header)
        with store.open_store(root, write=False) as txn:
            facts = {key: txn.meta(key) for key in ("phase", "generation", "lock_tier", "schema_version",
                     "legacy_results", "last_store_body", "job:legacy_reconcile", "job:current_store_import", "export")}
            derive = rederive or _flags_need_derivation(txn)
        tier_record = _lock_tier(root, facts["lock_tier"])
        tier = tier_record["tier"]
        retiered = {"retiered_from": header} if header != tier else {}
        phase = facts["phase"] or PUBLISHED
        if phase == PUBLISHED:
            last = facts["last_store_body"] or {}
            scheduled = body_sha is not None and _schedule_legacy(root, last, facts["job:legacy_reconcile"] or {}, body_sha)
            changed_body = body_sha is not None and last.get("sha") != body_sha
            upgrade = int(facts["schema_version"] or 1) < store.SCHEMA_VERSION
            flags, quarantined = {}, []
            report = {"status": "unchanged", "changed": 0, "new": 0, "deleted": 0, "results": facts["legacy_results"]}
            if derive or upgrade or retiered or tier_record != facts["lock_tier"] or changed_body or facts["legacy_results"] is None:
                with store.name_lock(root, IMPORT_LOCK_NAME, timeout_sec=600.0, stale_sec=600.0):
                    if retiered and header == TIER_ENFORCED:  # out of WAL under the tier in use, before the header names another
                        store.prepare_handoff(path, header, seam=False)
                    if retiered:  # the header changes in place: read it every time until the new tier commits
                        known = (str(path), store.file_identity(path))
                        store._TIERS[known] = None
                    with store.open_store(root, write=True, service=True) as txn:
                        if txn.phase() != PUBLISHED:
                            raise store.StoreRetired("store entered export during boot preparation")
                        if retiered:
                            txn.conn.execute(f"PRAGMA application_id = {store._APPLICATION_ID[tier]}")
                        if tier_record != facts["lock_tier"]:
                            txn.set_meta("lock_tier", tier_record)
                        if derive:
                            flags = _derive_flag_columns(txn, rederive=rederive)
                        if upgrade:
                            _upgrade_review_collection(txn)
                        if rederive:
                            report, quarantined = _reconcile(root, txn)
                            facts["job:current_store_import"] = {"state": "pending", "rederive": True, "updated_at": utc_now_iso()}
                            txn.set_meta("job:current_store_import", facts["job:current_store_import"])
                        elif facts["legacy_results"] is None:
                            report["results"] = len(txn.meta("baseline") or {})
                            txn.set_meta("legacy_results", report["results"])
                        if changed_body:
                            txn.set_meta("last_store_body", {"sha": body_sha, "at": utc_now_iso()})
                    if retiered:
                        store._TIERS[known] = tier
            if initialize_journal:  # explicit boot ownership, independent of legacy reconciliation
                report["journal_selected"] = _select_journal(root, path, tier)
            if not store.generation_stamped(root):
                stamp(root, **{GENERATION_KEY: GENERATION})
            _latches_into_intents(root, report.pop("latched", []))
            return {**report, "lock_tier": tier, **retiered, **flags,
                    "_import_job": facts["job:current_store_import"],
                    "legacy_reconcile": "scheduled" if scheduled else "not_needed"}, quarantined
        if (facts["export"] or {}).get("status") != "completed":
            # An interrupted export can still need old-writer obligation flags.
            # Its review schema/version stays unpublished until it returns to PUBLISHED.
            if derive or retiered or tier_record != facts["lock_tier"]:
                with store.name_lock(root, IMPORT_LOCK_NAME, timeout_sec=600.0, stale_sec=600.0):
                    if retiered and header == TIER_ENFORCED:  # as above: the old tier converts first
                        store.prepare_handoff(path, header, seam=False)
                    if retiered:  # as above: read every time until the new tier commits
                        known = (str(path), store.file_identity(path))
                        store._TIERS[known] = None
                    with store.open_store(root, write=True, service=True) as txn:
                        if derive:
                            _derive_flag_columns(txn, rederive=rederive)
                        if retiered:
                            txn.conn.execute(f"PRAGMA application_id = {store._APPLICATION_ID[tier]}")
                        if tier_record != facts["lock_tier"]:
                            txn.set_meta("lock_tier", tier_record)
                    if retiered:
                        store._TIERS[known] = tier
            return {"status": "switch_pending", "phase": phase, "lock_tier": tier, **retiered,
                    "legacy_reconcile": "not_needed"}, []
    elif store.generation_stamped(root):
        raise StoreLost(f"current store missing beside its completed stamp: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raced = False
    with store.name_lock(root, IMPORT_LOCK_NAME, timeout_sec=600.0, stale_sec=600.0):
        # Another process may have published while this first-touch writer waited.
        if state == "absent" and store.probe(root)[0] == "present":
            raced = True
        else:
            tier_record = _lock_tier(root)
            if state == "present":
                stamp(root, **{GENERATION_KEY: None})
                os.replace(path, path.with_name(f"{path.name}.exported-{utc_stamp()}"))
            for name in os.listdir(path.parent):
                if name.startswith(path.name + ".import-"):
                    with contextlib.suppress(OSError):
                        (path.parent / name).unlink()
            report, quarantined = _build_and_publish(root, path, tier_record["tier"], lock_tier_record=tier_record)
    if raced:
        return _ensure(root, path, rederive=rederive, body_sha=body_sha, initialize_journal=initialize_journal)
    scheduled = body_sha is not None and _schedule_legacy(root, {}, {}, body_sha)
    if body_sha is not None:
        with store.open_store(root, write=True, service=True) as txn:
            txn.set_meta("last_store_body", {"sha": body_sha, "at": utc_now_iso()})
    return {**report, "legacy_reconcile": "scheduled" if scheduled else "not_needed"}, quarantined


def _select_journal(root: pathlib.Path, path: pathlib.Path, tier: str) -> bool:
    """``store_journal.configure_journal`` for the boot, a name-tier file under its name lock."""
    with store.name_lock(root, LOCK_NAME, timeout_sec=store.STORE_WAIT_SEC) if tier == TIER_NAME \
            else contextlib.nullcontext():
        return store_journal.configure_journal(path, tier)


def _upgrade_review_collection(txn: Txn) -> None:
    """Schema only: old bodies stay unknown until their ordinary maintenance owner visits them.
    DDL and its version commit together; an interrupted upgrade is retried in place."""
    columns = {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    for column, index in (("review_collection_pending", store.REVIEW_COLLECTION_INDEX),
                          ("plan_collection_pending", store.PLAN_COLLECTION_INDEX)):
        if column not in columns:
            txn.conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} INTEGER")
        txn.conn.execute(index)
    if "drive_retention_at" not in columns:
        txn.conn.execute("ALTER TABLE tasks ADD COLUMN drive_retention_at TEXT")
    for index in store.ROOT_WINDOWS:  # schema 8: root windows and terminal_seq, built once, committed with the version
        txn.conn.execute(index)
    if "terminal_seq" not in columns:  # rows written terminal before it carry no count (nothing to look at again)
        txn.conn.execute("ALTER TABLE tasks ADD COLUMN terminal_seq INTEGER")
    txn.conn.execute(store.TERMINAL_SEQ_INDEX)
    if int(txn.meta("schema_version") or 1) < store.SCHEMA_VERSION:
        txn.set_meta("schema_version", store.SCHEMA_VERSION)


def upgrade_current_schema(root: Any) -> None:
    """First-touch writer/maintenance preparation, never a passive read or history import.
    The ordinary write gate and inherited acquisition budget also govern this DDL:
    a caller under a queue/launch lock cannot acquire an extra long import lock."""
    with store.open_store(root, write=True) as txn:
        if _flags_need_derivation(txn):
            _derive_flag_columns(txn)  # all prior flags must exist before the version advances
        if int(txn.meta("schema_version") or 1) < store.SCHEMA_VERSION:
            _upgrade_review_collection(txn)


def _flags_need_derivation(txn: Txn) -> bool:
    """An empty partial index proves there are no old-writer NULLs without scanning tasks.
    Missing columns/index need one schema preparation; old REPLACE writers maintain this index too."""
    columns = {row["name"]: row["notnull"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    if not set(store.FLAG_COLUMNS.values()) <= columns.keys():
        return True
    if not txn.conn.execute("SELECT 1 FROM sqlite_master WHERE name='tasks_underived_flags'").fetchone():
        return True
    return txn.conn.execute(
        f"SELECT task_id FROM tasks INDEXED BY tasks_underived_flags WHERE {_null_flags(columns)} LIMIT 1").fetchone() is not None


def _derive_flag_columns(txn: Txn, *, rederive: bool = False) -> Dict[str, Any]:
    """Derive every nullable flag (``store.FLAG_COLUMNS``) left NULL, "not derived": a store
    published before the column gains it NULL with its index, and a row a body that predates the
    column wrote through its own column list (``INSERT OR REPLACE``: a rollback's writes, its new
    tasks) holds NULL whatever ``meta.schema_version`` says. Each such row takes the predicate
    every later write applies (``result_memberships``) over its document; an undecodable body is
    owed (unknown is never absence) until its next write recomputes it. A partial index selects
    only underived rows; ``rederive`` (the explicit rebuild)
    forgets every nullable flag first, so it decodes every body. Only complete schema
    preparation publishes the version; this flag-only step may also run during export."""
    columns = {record["name"]: record["notnull"] for record in txn.conn.execute("PRAGMA table_info(tasks)")}
    added = sorted(name for name, column in store.FLAG_COLUMNS.items() if column not in columns)
    for name in added:
        column = store.FLAG_COLUMNS[name]
        txn.conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} INTEGER")
        txn.conn.execute(f"CREATE INDEX tasks_{column} ON tasks({column}) WHERE {column}=1")
    if added:
        txn.conn.execute("DROP INDEX IF EXISTS tasks_underived_flags")
    predicate = _null_flags(columns)
    txn.conn.execute(f"CREATE INDEX IF NOT EXISTS tasks_underived_flags ON tasks(task_id) WHERE {predicate}")
    nullable = {name: column for name, column in store.FLAG_COLUMNS.items() if not columns.get(column)}
    if not nullable:
        return {}
    if rederive:
        txn.conn.execute(f"UPDATE tasks SET {', '.join(f'{column}=NULL' for column in nullable.values())}")
    underived = [record["task_id"] for record in txn.conn.execute(
        f"SELECT task_id FROM tasks INDEXED BY tasks_underived_flags WHERE {predicate}")]
    for task_id in underived:
        try:
            names = o.result_memberships(txn.load(task_id) or {})
        except ValueError:
            names = set(nullable)
        txn.conn.execute(f"UPDATE tasks SET {', '.join(f'{column}=?' for column in nullable.values())} WHERE task_id=?",
                         (*(int(name in names) for name in nullable), task_id))
    return {**({"flag_columns_added": added} if added else {}),
            **({"flag_rows_derived": len(underived)} if underived else {})}


def _build_and_publish(root: pathlib.Path, path: pathlib.Path, tier: str, *, lock_tier_record: dict) -> Tuple[Dict[str, Any], list]:
    from ouroboros.startup_migrations import boot_admissions, migrate_cancel_latches, stamp

    sets = _read_sets(root)
    sets_baseline = _sets_baseline(sets)  # the same file snapshot, before boot/chain admissions
    records, baseline, quarantined, refused = _read_results(root)
    for name, rows in boot_admissions(root).items():  # under the set files: an interrupted publication stays a candidate
        sets[name] = {**rows, **sets.get(name, {})}
    folded, custody = _custody_rows(root, records)
    # The chain decides a pending invocation: a request the retained chain bound to a run or
    # refused (an old body's append after the export) is not readmitted from the exported set file.
    resolved = {str(facts.get("invocation_id") or "") for facts, _hold in folded["runs"].values()} | set(folded["refused"])
    for name, rows in custody.items():
        if name == "custody_open":
            kept = {key: facts for key, facts in sets.get(name, {}).items() if key.split(":", 1)[-1] not in resolved}
            sets[name] = {**kept, **rows}
        else:
            sets[name] = _merge_receipts(sets.get(name, {}), rows)
    sets = {name: o.owner_held_rows(name, rows) for name, rows in sets.items()}  # derived members come from the columns
    provenance = {"status": "completed", "imported_at": utc_now_iso(), "lock_tier": tier,
                  "results": len(records), "quarantined": len(quarantined), "refused": refused,
                  "sets": {name: len(rows) for name, rows in sets.items()},
                  "custody_runs": len(folded.get("runs", {}))}
    tmp = path.with_name(f"{path.name}.import-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        _build(tmp, tier, root, records, sets, baseline, provenance, folded,
               lock_tier_record=lock_tier_record,
               sets_baseline=sets_baseline, chain_offset=folded.get("chain_offset", 0),
               occupied={**_quarantined_occupied(root, [task_id for task_id, _ in records]),
                         **{row["task_id"]: row["reason"] for row in quarantined}, **refused})
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    store.fsync_directory(path.parent)
    stamp(root, **{GENERATION_KEY: GENERATION})  # after publication: a published, unstamped store is still ours
    _latches_into_intents(root, [task_id for task_id, entry in baseline.items() if entry.get("latch_stamped")])
    migrate_cancel_latches(root, records=[(task_id, document, "") for task_id, document in records])
    return {"status": "completed", "lock_tier": tier, "results": len(records), "quarantined": len(quarantined),
            "refused": len(refused), "sets": len(sets)}, quarantined


def _latches_into_intents(root: pathlib.Path, latched: List[str]) -> None:
    """The legacy latch rows this import stamped: one typed events row per import, and each
    becomes a cancel intent (the generation watermark does not gate rows an old body wrote later)."""
    if not latched:
        return
    from ouroboros.cancel_intents import migrate_legacy_cancel_latches

    append_jsonl(root / "logs/events.jsonl", {"ts": utc_now_iso(), "type": "task_result_cancel_latch_admitted",
                                              "count": len(latched), "task_ids": latched, "reason": "unstamped_pre_7_0"})
    with store.open_store(root, write=False) as txn:
        records = [(task_id, txn.load(task_id), "") for task_id in latched]
    migrate_legacy_cancel_latches(root, records=[record for record in records if record[1]])


def _custody_rows(root: pathlib.Path, records: List[Record]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Any]]]]:
    """The custody half of the import, folded ONCE from the retained event chain: every run's
    row (``custody_runs``), the pending invocations (``custody_open``), the definite refusals
    and the task marks, plus the closing receipts still owed to a result's terminal refresh
    (``delegated_runs``). A chain that exists but cannot be read in full (EMFILE, EIO, an
    unopenable segment) ABORTS the import typed — nothing is built or stamped and the next
    boot retries — never a published store with no custody row; a missing chain is empty."""
    from ouroboros import delegate_custody as c
    from ouroboros.delegate_custody_current import fold

    chain = c.event_log_path(root)
    try:
        if c.custody_chain_unreadable(root):
            raise OSError(errno.EIO, "custody chain exists but cannot be opened")
        rows, offset = c.rows_after(chain, 0)
    except OSError as exc:
        raise _unreadable(exc, chain, "custody chain") from exc
    folded = {**fold(rows), "chain_offset": offset}  # the boundary: what an old body appends later is P6' input
    owed = {task_id for task_id, document in records if document.get("delegated_runs_unreconciled")}
    receipts: Dict[str, Dict[str, Any]] = {tid: {"task_id": tid, "cost_revision": revision}
                                        for tid, revision in folded["cost_revisions"].items()}
    for rid, (facts, _hold) in folded["runs"].items():
        task_id = str(facts.get("task_id") or "")
        if facts.get("settled") and task_id in owed:
            receipts.setdefault(task_id, {"task_id": task_id}).setdefault("closed_runs", {})[rid] = facts
    return folded, {"custody_open": folded["open"], "delegated_runs": receipts}


def _merge_receipts(held: Dict[str, Dict[str, Any]], folded: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = {task_id: dict(facts) for task_id, facts in held.items()}
    for task_id, facts in folded.items():
        entry = merged.setdefault(task_id, {"task_id": task_id})
        entry["closed_runs"] = {**facts.get("closed_runs", {}), **entry.get("closed_runs", {})}
        if facts.get("cost_revision"):
            entry["cost_revision"] = max(facts["cost_revision"], entry.get("cost_revision", 0))
    return merged


def _admit(path: pathlib.Path, raw: bytes) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """``(document, reason, outcome)``: ``admitted``; ``quarantined`` (the file moved, bytes
    unchanged, through the existing quarantine); ``refused`` (left in place: a task-id
    mismatch, or a quarantine that could not move the file)."""
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    refusal = task_result_schema_refusal(data)
    if (refusal == "unstamped_pre_7_0" and isinstance(data, dict) and data.get("status") == "cancel_requested"
            and data.get("task_id") == path.stem):
        # The one legacy carve-out: an admissible pre-7 cancel latch is stamped on import (same
        # status, same fields) so custody can drive it to its terminal; every other unstamped row quarantines.
        return stamp_task_result_schema(data), "latch_stamped", "admitted"
    if refusal:  # writers are quiescent: a row the quarantine finds admissible again stays refused here
        return None, refusal, "quarantined" if quarantine_task_result(path, refusal) == "moved" else "refused"
    if str(data.get("task_id") or "") != path.stem:
        return None, "task_id_mismatch", "refused"
    return data, "", "admitted"


def _baseline_entry(path: pathlib.Path, raw: bytes, document: Optional[Dict[str, Any]], reason: str = "") -> Dict[str, Any]:
    info = path.stat()
    entry = {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": sha256_hex(raw)}
    if document is None:
        return {**entry, "refused": reason}  # remembered, so an untouched refused file stays untouched
    if reason == "latch_stamped":
        entry["latch_stamped"] = True
    # The whole projected document in the shape ``load`` returns (its card values last): the same bytes
    # ``_projected`` hashes, so a stored row is no false conflict against its own file.
    return {**entry, "body_sha256": sha256_hex(document_text(store_task_cards.canonical(document)).encode("utf-8"))}


def _read_result(path: pathlib.Path) -> Optional[bytes]:
    """The file's bytes, ``None`` when it is gone; a resource or I/O error is typed and ABORTS
    the import (an unreadable file is never a refused row, and nothing gets stamped)."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unreadable(exc, path) from exc


def _stat_result(path: pathlib.Path) -> Optional[Tuple[int, int]]:
    """``(size, mtime_ns)``, ``None`` when the file is gone; other errors typed as for a read."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unreadable(exc, path) from exc
    return info.st_size, info.st_mtime_ns


def _unreadable(exc: OSError, path: pathlib.Path, what: str = "task result") -> store.StoreError:
    name = errno.errorcode.get(exc.errno, str(exc.errno))
    kind = store.StoreBusy if exc.errno in (errno.EMFILE, errno.ENFILE) else store.StoreError
    return kind(f"{what} not readable ({name}), import aborted: {path}")


def _read_results(root: pathlib.Path):
    records: List[Record] = []
    baseline: Dict[str, Dict[str, Any]] = {}
    quarantined: List[Dict[str, str]] = []
    refused: Dict[str, str] = {}
    for path in sorted(task_results_dir(root, create=False).glob("*.json")):
        raw = _read_result(path)
        if raw is None:
            continue
        document, reason, outcome = _admit(path, raw)
        if outcome == "quarantined":
            quarantined.append({"task_id": path.stem, "reason": reason})
            continue
        if outcome == "refused":
            refused[path.stem] = reason
        else:
            records.append((path.stem, document))
        baseline[path.stem] = _baseline_entry(path, raw, document, reason)
    return records, baseline, quarantined, refused


def set_file_path(root: pathlib.Path, name: str) -> pathlib.Path:
    """The old body's set file ``state/obligations/<name>.json``: the importer's input, the exporter's output."""
    return pathlib.Path(root) / "state" / "obligations" / f"{name}.json"


def _read_sets(root: pathlib.Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """One complete snapshot: missing files are absent; unreadable or malformed files abort
    typed, leaving their bytes in place and published rows intact for a later retry."""
    sets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    directory = set_file_path(root, "nonterminal").parent
    try:
        with os.scandir(directory) as entries:  # Path.glob can suppress directory read errors
            paths = sorted(pathlib.Path(entry.path) for entry in entries if entry.name.endswith(".json"))
    except FileNotFoundError:
        return sets
    except OSError as exc:
        raise _unreadable(exc, directory, "obligation sets") from exc
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not all(isinstance(row, dict) for row in value.values()):
                raise ValueError("expected an id-to-facts object")
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _unreadable(exc, path, "obligation set") from exc
        except ValueError as exc:
            raise store.StoreError(f"obligation set malformed, import aborted: {path}") from exc
        sets[path.stem] = value
    return sets


OCCUPIED = "occupied"  # meta: the identities whose file the import refused or quarantined (ABI-2: not readmitted)


def _quarantined_occupied(root: pathlib.Path, admitted: List[str]) -> Dict[str, str]:
    """Recover reservations after a quarantine rename outlived an interrupted import/transaction.
    Admitted rows win; deletion backups are evidence of retirement, not refused admission."""
    directory = task_results_dir(root, create=False) / TASK_RESULT_QUARANTINE_DIR
    try:
        with os.scandir(directory) as entries:
            identities = {re.sub(r"\.[0-9a-f]{8}$", "", pathlib.Path(entry.name).stem)
                          for entry in entries if entry.name.endswith(".json") and not re.search(r"\.deleted-\d{8}T\d{6}Z\.json$", entry.name)}
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise _unreadable(exc, directory, "task result quarantine") from exc
    return {task_id: "quarantined_before_import" for task_id in identities - set(admitted)}


def occupied_reason(root: pathlib.Path, task_id: str, *, write: bool = False) -> str:
    """Why TASK_ID's identity is occupied without a row (its file was refused or quarantined by
    the import or a P6'), ``""`` for a free identity: what a strict read or a create-only write
    (``write``: through the write gate, so a first touch imports first) consults instead of
    scanning the quarantine directory."""
    with store.open_store(root, write=write) as txn:
        return str((txn.meta(OCCUPIED) or {}).get(str(task_id)) or "")


def holds_facts(root: pathlib.Path) -> bool:
    """Whether ROOT holds facts the importer would read: result files or obligation set files."""
    for directory in (task_results_dir(root, create=False), set_file_path(root, "nonterminal").parent):
        try:
            with os.scandir(directory) as entries:
                if any(entry.name.endswith(".json") for entry in entries):
                    return True
        except OSError:
            continue
    return False


def _sets_baseline(files: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    """The owner-held rows of every set file as last seen, by content digest: what P6' compares a
    re-read file against so that only the rows the old body added, changed or retired move."""
    return {name: {identity: sha256_hex(machine_json(facts).encode("utf-8")) for identity, facts in held.items()}
            for name, rows in files.items() for held in [o.owner_held_rows(name, rows)]}


def _build(tmp: pathlib.Path, tier: str, root: pathlib.Path, records: List[Record], sets: Dict[str, Any],
           baseline: Dict[str, Any], provenance: Dict[str, Any], folded: Optional[Dict[str, Any]] = None, *,
           sets_baseline: Optional[Dict[str, Any]] = None, chain_offset: int = 0,
           occupied: Optional[Dict[str, str]] = None, lock_tier_record: Optional[dict] = None) -> None:
    from ouroboros.delegate_custody_current import CHAIN_BOUNDARY, write_fold

    with contextlib.closing(store.connect(tmp, tier, create=True)) as conn:
        conn.execute("PRAGMA synchronous = FULL")
        # executescript commits a pending transaction; BEGIN must be inside the script.
        conn.executescript("BEGIN IMMEDIATE;\n" + store._SCHEMA)
        _derive_flag_columns(Txn(conn, root))  # prepare the empty NULL index before publishing
        conn.execute(f"PRAGMA application_id = {store._APPLICATION_ID[tier]}")
        txn = Txn(conn, root)
        for task_id, document in records:
            txn.store_task(document, pending_drive=bool(o.pending_drive(root, task_id, document)))
        for name, rows in sets.items():
            txn.replace_set(name, rows)
        if folded:
            write_fold(txn, folded, replace=False)  # the sets above already hold the fold's open rows
        for key, value in (("schema_version", store.SCHEMA_VERSION), ("lock_tier", lock_tier_record or tier), ("generation", GENERATION),
                           ("phase", PUBLISHED), ("import", provenance), ("freeze_utc", utc_now_iso()),
                           ("job:current_store_import", {"state": "pending", "cursor": ".", "done": 1,
                                                         "total": None, "updated_at": utc_now_iso()}),
                           ("baseline", baseline), ("legacy_results", len(baseline)), ("sets_baseline", sets_baseline or {}), (CHAIN_BOUNDARY, chain_offset),
                           (OCCUPIED, occupied or {})):
            txn.set_meta(key, value)
        store.commit(conn)


def _quarantine_copy(root: pathlib.Path, task_id: str, body: str) -> None:
    """The DB version of a row whose file an old body deleted: kept beside the quarantined bytes."""
    directory = task_results_dir(root) / TASK_RESULT_QUARANTINE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    write_text_atomic(directory / f"{task_id}.deleted-{utc_stamp()}.json", body, fsync=True)


def _reconcile(root: pathlib.Path, txn: Txn) -> Tuple[Dict[str, Any], list]:
    """P6' per-row reconciliation; stat prefilter and same-stat residual: module docstring.
    Stat-changed bytes are hashed; a matching digest refreshes only the stored stat."""
    baseline: Dict[str, Dict[str, Any]] = txn.meta("baseline") or {}
    baseline_changed = False
    present: Dict[str, Tuple[pathlib.Path, Optional[bytes]]] = {}
    changed: List[str] = []
    for path in sorted(task_results_dir(root, create=False).glob("*.json")):
        entry = baseline.get(path.stem)
        stat = _stat_result(path)
        if stat is None:
            continue
        if entry is not None and stat == (entry["size"], entry["mtime_ns"]):
            present[path.stem] = (path, None)  # untouched: not read
            continue
        raw = _read_result(path)
        if raw is None:
            continue
        present[path.stem] = (path, raw)
        if entry is None:
            continue  # new: imported below
        if sha256_hex(raw) != entry["sha256"]:
            changed.append(path.stem)
        else:
            baseline[path.stem] = {**entry, "size": stat[0], "mtime_ns": stat[1]}  # touched, same bytes
            baseline_changed = True
    fresh = sorted(task_id for task_id in present if task_id not in baseline)
    deleted = sorted(task_id for task_id in baseline if task_id not in present)
    counts = {"changed": len(changed), "new": len(fresh), "deleted": len(deleted), "results": len(baseline)}
    if not (changed or fresh or deleted):
        if baseline_changed:
            txn.set_meta("baseline", baseline)
        occupied = txn.meta(OCCUPIED) or {}
        recovered = {**_quarantined_occupied(root, txn.task_ids()), **occupied}
        if recovered != occupied:
            txn.set_meta(OCCUPIED, recovered)
        return {"status": "unchanged", **counts}, []
    sets = _read_sets(root)  # validate before any result is moved into quarantine
    now = utc_now_iso()
    quarantined: List[Dict[str, str]] = []
    refused: Dict[str, str] = {}
    latched: List[str] = []
    conflicts: Dict[str, Any] = txn.meta("conflicts") or {}
    occupied: Dict[str, str] = txn.meta(OCCUPIED) or {}
    for task_id in changed + fresh:  # the old body was the authority while it ran: its files win
        path, raw = present[task_id]
        document, reason, outcome = _admit(path, raw)
        if outcome == "quarantined":  # the file left the directory with its bytes; the row goes with it
            quarantined.append({"task_id": task_id, "reason": reason})
            occupied[task_id] = reason
            txn.delete_task(task_id)
            baseline.pop(task_id, None)
            continue
        if outcome == "refused":
            refused[task_id] = occupied[task_id] = reason
            baseline[task_id] = _baseline_entry(path, raw, None, reason)
            continue
        occupied.pop(task_id, None)  # an admissible file re-opens the identity
        if reason == "latch_stamped":
            latched.append(task_id)
        body = _projected(txn, task_id)  # the store's version WITH its handoff rows (B5)
        if body is not None and sha256_hex(body.encode("utf-8")) != baseline.get(task_id, {}).get("body_sha256"):
            conflicts[task_id] = {"at": now, "db_body": body, "file_sha256": sha256_hex(raw)}
        txn.store_task(document, pending_drive=bool(o.pending_drive(root, task_id, document)))
        baseline[task_id] = _baseline_entry(path, raw, document, reason)
    for task_id in deleted:
        body = _projected(txn, task_id)
        if body is not None:
            _quarantine_copy(root, task_id, body)  # saved before the task and its handoffs go
            quarantined.append({"task_id": task_id, "reason": "deleted_by_old_body"})
        txn.delete_task(task_id)
        baseline.pop(task_id, None)
    moved = _reconcile_sets(txn, sets)  # the old body maintained the set files too
    legacy = _apply_legacy_custody(root, txn)
    occupied = {**_quarantined_occupied(root, txn.task_ids()), **occupied}
    for key, value in (("baseline", baseline), ("legacy_results", len(baseline)), ("freeze_utc", now), ("conflicts", conflicts), (OCCUPIED, occupied),
                       ("reconciled", {"at": now, **counts, "quarantined": len(quarantined), "refused": refused,
                                       "conflicts": len(conflicts), "set_rows": moved, "legacy_custody": legacy})):
        txn.set_meta(key, value)
    return {"status": "reconciled", **counts, "quarantined": len(quarantined), "refused": len(refused),
            "conflicts": len(conflicts), "latched": latched, "set_rows": moved, "legacy_custody": legacy}, quarantined


def _projected(txn: Txn, task_id: str) -> Optional[str]:
    """The store's whole document (body plus handoff rows) as a file would hold it."""
    document = txn.load(task_id)
    return None if document is None else document_text(document)


def _reconcile_sets(txn: Txn, files: Dict[str, Dict[str, Dict[str, Any]]]) -> int:
    """Owner-held rows against their own baseline (B4): a row the old body added or changed in a
    set file is put, one it retired is deleted, and every row the file left as it was keeps the
    store's version — the facts this store published since (its rows, its retirements) survive
    a P6' instead of being replaced by a stale file. Returns the rows moved."""
    baseline: Dict[str, Dict[str, str]] = txn.meta("sets_baseline") or {}
    current = _sets_baseline(files)
    moved = 0
    for name in sorted(set(current) | set(baseline)):
        held, before, after = o.owner_held_rows(name, files.get(name, {})), baseline.get(name, {}), current.get(name, {})
        for identity in sorted(set(before) | set(after)):
            if before.get(identity) == after.get(identity):
                continue  # untouched by the old body: the store's own answer stands
            if identity in after:
                txn.put_row(name, identity, held[identity])
            else:
                txn.delete_row(name, identity)
            moved += 1
    txn.set_meta("sets_baseline", current)
    return moved


def _apply_legacy_custody(root: pathlib.Path, txn: Txn) -> int:
    """The custody events an old body appended to the chain after this store's boundary (its
    import, its own last row) are reduced onto the rows — the reducer's event-id dedup and
    idempotent replays included — and the boundary moves to the chain's end (B4)."""
    from ouroboros import delegate_custody as c
    from ouroboros.delegate_custody_current import CHAIN_BOUNDARY, _loaded, _reduce

    chain = c.event_log_path(root)
    try:
        if c.custody_chain_unreadable(root):
            raise OSError(errno.EIO, "custody chain exists but cannot be opened")
        rows, offset = c.rows_after(chain, int(txn.meta(CHAIN_BOUNDARY) or 0))
    except OSError as exc:
        raise _unreadable(exc, chain, "custody chain") from exc
    applied = 0
    for row in c._unique_rows(rows):
        _reduce(txn, row, _loaded(txn, row), {})
        applied += 1
    txn.set_meta(CHAIN_BOUNDARY, offset)
    return applied


def _identity(text: str) -> Dict[str, Any]:
    data = text.encode("utf-8")
    return {"size": len(data), "sha256": sha256_hex(data)}


def export_facts(root: Any) -> Dict[str, Any]:
    """The exact inverse of the import, in two transactions around the writes.

    T1 commits ``meta.export_manifest``: the name, size and sha256 of every file the export
    will write (every row's whole ``load`` document to ``task_results/<id>.json``, every
    obligation set file in FULL) and the stale result files it removes (admissible files
    whose row the store no longer holds; refused files and the quarantine stay as evidence).
    Then the files: tmp + rename, a byte-identical file left alone, fsync of files and
    directories. T2 refreshes the content baseline to what was written and completes the
    manifest. A death in between leaves a manifest ``absorb_own_export`` reads at
    abandonment, so the store's own partial writes are never mistaken for an old body's."""
    root = pathlib.Path(root)
    names: List[str] = []
    with store.open_store(root, write=True, service=True) as txn:  # T1: the plan, durable before the first write
        baseline = txn.meta("baseline") or {}
        task_ids = txn.task_ids()
        names = sorted(set(txn.set_names()) | set(o.BOOT_SETS))  # an older body expects every set file
        stale = sorted(path.stem for path in task_results_dir(root, create=False).glob("*.json")
                       if path.stem not in task_ids and not (baseline.get(path.stem) or {}).get("refused"))
        manifest = {"status": "writing", "started_at": utc_now_iso(),
                    "results": {task_id: _identity(document_text(txn.load(task_id))) for task_id in task_ids},
                    "sets": {name: _identity(machine_json(o.members_in(txn, name)))  # the set file in FULL, as atomic_write_json writes it
                             for name in names},
                    "removed": stale}
        txn.set_meta("export_manifest", manifest)
    written = total = 0
    entries: Dict[str, Dict[str, Any]] = {}
    with store.open_store(root, write=False) as txn:  # the frozen rows, streamed to their files
        for task_id in task_ids:
            document = txn.load(task_id)
            text = document_text(document)
            data = text.encode("utf-8")
            path = task_result_path(root, task_id)
            if _read_result(path) != data:  # a read error aborts the export typed, never «rewrite blindly»
                write_text_atomic(path, text, fsync=True)
                written += 1
            total += len(data)
            entries[task_id] = _baseline_entry(path, data, document)
        for name in names:
            atomic_write_json(set_file_path(root, name), o.members_in(txn, name), fsync=True)
    for task_id in stale:
        with contextlib.suppress(FileNotFoundError):
            task_result_path(root, task_id).unlink()
    store.fsync_directory(task_results_dir(root, create=False))
    store.fsync_directory(set_file_path(root, "nonterminal").parent)
    with store.open_store(root, write=True, service=True) as txn:  # T2: the baseline follows what was written
        refused = {task_id: entry for task_id, entry in (txn.meta("baseline") or {}).items() if entry.get("refused")}
        txn.set_meta("baseline", {**refused, **entries})
        txn.set_meta("legacy_results", len(refused.keys() | entries.keys()))
        txn.set_meta("sets_baseline", _sets_baseline(_read_sets(root)))
        txn.set_meta("export_manifest", {**manifest, "status": "completed", "completed_at": utc_now_iso()})
    return {"results": len(task_ids), "written": written, "removed": len(stale), "bytes": total, "sets": len(names)}


def absorb_own_export(root: Any, txn: Txn) -> Dict[str, Any]:
    """At abandonment, adopt manifest-matching (size + sha256) files and stale deletions into the
    baseline while republishing the store. Its partial export then triggers no old-body
    re-import at boot, preserving every row, including owner-held sets."""
    root = pathlib.Path(root)
    manifest = txn.meta("export_manifest") or {}
    if manifest.get("status") != "writing":
        return {}
    baseline = txn.meta("baseline") or {}
    adopted = 0
    for task_id, identity in manifest.get("results", {}).items():
        path = task_result_path(root, task_id)
        raw = _read_result(path)
        if raw is not None and len(raw) == identity["size"] and sha256_hex(raw) == identity["sha256"]:
            baseline[task_id] = _baseline_entry(path, raw, txn.load(task_id))
            adopted += 1
    for task_id in manifest.get("removed", []):
        if _read_result(task_result_path(root, task_id)) is None:
            baseline.pop(task_id, None)
    txn.set_meta("baseline", baseline)
    txn.set_meta("legacy_results", len(baseline))
    files = _read_sets(root)
    sets_baseline = dict(txn.meta("sets_baseline") or {})
    for name, identity in manifest.get("sets", {}).items():  # a set file this export wrote is its own, not an old body's
        if name in files and _identity(machine_json(files[name])) == identity:
            sets_baseline[name] = _sets_baseline({name: files[name]})[name]
    txn.set_meta("sets_baseline", sets_baseline)
    txn.set_meta("export_manifest", {**manifest, "status": "abandoned", "adopted": adopted, "at": utc_now_iso()})
    return {"adopted": adopted, "planned": len(manifest.get("results", {}))}


def run_legacy_reconcile(root: Any, *, progress: Optional[Callable[[dict], None]] = None,
                         should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Run a pending job (or a first explicit invocation); a done job stays done.

    The CLI queues a fresh pass after a done job, otherwise resumes. Stop keeps a pending
    cursor; failures keep it with the drive/error for repair. No lifespan/supervisor
    caller runs this function. Publication and ordinary writer gates still apply.
    """
    from ouroboros import maintenance_jobs
    from ouroboros.startup_migrations import migrate_question_publications

    root = pathlib.Path(root)
    ensure_current_store(root, with_children=False)
    with store.name_lock(root, "legacy_reconcile.lock", timeout_sec=store.STORE_WAIT_SEC):
        row = maintenance_jobs.job(root, "legacy_reconcile") or {}
        if row.get("state") == "done":
            return row

        def record(**facts):
            updated = maintenance_jobs.record(root, "legacy_reconcile", **facts)
            if progress is not None:
                progress(updated)
            return updated

        cursor, done = str(row.get("cursor") or ""), int(row.get("done") or 0)
        if should_stop is not None and should_stop():
            return record(state="pending", cursor=cursor, done=done)
        drive = root
        try:
            drives = [root, *sorted(descendant_drives(root), key=lambda path: path.relative_to(root).as_posix())]
            total = len(drives)
            row = record(state="running", cursor=cursor, done=done, total=total, error=None, waiting_on=None)
            for drive in drives:
                relative = drive.relative_to(root).as_posix()
                if relative <= cursor:
                    continue
                if should_stop is not None and should_stop():
                    return record(state="pending")
                report = ensure_current_store(drive, with_children=False)
                if report["status"] != "completed":
                    with store.open_store(drive, write=True) as txn:
                        report, quarantined = _reconcile(drive, txn)
                    emit_quarantine_event(drive, quarantined)
                    _latches_into_intents(drive, report.pop("latched", []))
                done += 1
                row = record(cursor=relative, done=done, **({"root_report": report} if drive == root else {}))
            if should_stop is not None and should_stop():
                return record(state="pending")
            # Legacy asks can be introduced by this pass after the generation was stamped.
            publications = migrate_question_publications(root, legacy=True)
            return record(state="done", question_publications=publications)
        except BaseException as exc:
            maintenance_jobs.record(root, "legacy_reconcile",
                                    state="failed" if isinstance(exc, Exception) else "pending",
                                    error=f"{type(exc).__name__}: {exc}", waiting_on={"drive": str(drive)})
            raise


def main() -> None:
    import argparse
    from ouroboros import maintenance_jobs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile-legacy", action="store_true", required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.root is None:
        from ouroboros.config import resolve_data_dir
        args.root = resolve_data_dir()
    ensure_current_store(args.root, with_children=False)
    current = maintenance_jobs.job(args.root, "legacy_reconcile") or {}
    if current.get("state") not in {"pending", "running", "failed"}:
        maintenance_jobs.record(args.root, "legacy_reconcile", state="pending", reason="explicit_command",
                                cursor=None, done=0, total=None, error=None, waiting_on=None)
    run_legacy_reconcile(args.root, progress=lambda row: print(json.dumps({"job": "legacy_reconcile", **row}), flush=True),
                         should_stop=lambda: False)


if __name__ == "__main__":
    main()
