"""Addressed control/phase facts and the resumable ``hot_facts`` derivation job.

Whole-document writers derive these columns in their owning transaction, so a derived
row is read from its columns alone (no body read, no validation). NULL means an older
writer has not derived them; a passive reader only reads that row's body. Historical
classification belongs to ``run_hot_facts``, never boot.
The request snapshot is short lived and released before any consumer runs.

The schema gate adds the columns without decoding bodies. ``run_hot_facts`` classifies
underived rows under a sequence/source CAS, so a racing writer's row is left to that
writer and a damaged input waits for its own change; a body that is not its own task's
object (``own_document``) is such input and derives nothing, never another task's root.
Stop's membership probe is the indexed root set plus parent ancestry, queue-only bridges
included. The census copies its queue and owner registries before its one short
transaction per addressed store.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import os
import sqlite3
from typing import Any

from ouroboros import store_task_cards
from ouroboros.utils import machine_json

VERSION = 6
LIST_VERSION = 9
JOB = "hot_facts"
TASK_COLUMNS = {
    "owner_pause_state": "TEXT", "owner_pause_fence_id": "TEXT", "owner_pause_facts": "TEXT",
    "budget_pause_state": "TEXT", "budget_pause_attempt": "INTEGER", "budget_pause_facts": "TEXT",
    "post_task_synthesis": "TEXT", "owner_wait_state": "TEXT", "owner_wait_reason": "TEXT",
    "control_valid": "INTEGER", "list_stable": "INTEGER",
}
CUSTODY_COLUMNS = {
    "root_task_id": "TEXT", "billing_group_id": "TEXT", "output_unread": "INTEGER",
    "payload_held": "INTEGER", "facts_derived": "INTEGER", "write_seq": "INTEGER NOT NULL DEFAULT 0",
}
TASK_UNDERIVED = " OR ".join(f"{name} IS NULL" for name in (*TASK_COLUMNS, "root_task_id"))
CUSTODY_UNDERIVED = "facts_derived IS NULL"
REQUEST_SETS = ("custody_open", "custody_refused")
REQUEST_UNDERIVED = "set_name IN ('custody_open','custody_refused') AND request_root_task_id IS NULL"
INDEXES = (
    f"CREATE INDEX IF NOT EXISTS tasks_underived_hot ON tasks(task_id) WHERE {TASK_UNDERIVED}",
    "CREATE INDEX IF NOT EXISTS tasks_list_stable ON tasks(list_stable, status, ts, task_id)",
    "CREATE INDEX IF NOT EXISTS tasks_underived_root ON tasks(task_id) WHERE root_task_id IS NULL",
    "CREATE INDEX IF NOT EXISTS tasks_owner_pause ON tasks(owner_pause_state, task_id) WHERE owner_pause_state<>''",
    "CREATE INDEX IF NOT EXISTS tasks_budget_pause ON tasks(budget_pause_state, task_id) WHERE budget_pause_state<>''",
    "CREATE INDEX IF NOT EXISTS tasks_phase ON tasks(post_task_synthesis, task_id) WHERE post_task_synthesis<>''",
    "CREATE INDEX IF NOT EXISTS custody_root ON custody_runs(root_task_id, task_id)",
    "CREATE INDEX IF NOT EXISTS custody_billing ON custody_runs(billing_group_id, root_task_id)",
    "CREATE INDEX IF NOT EXISTS custody_unread ON custody_runs(output_unread, run_id)",
    "CREATE INDEX IF NOT EXISTS custody_payload ON custody_runs(payload_held, run_id)",
    f"CREATE INDEX IF NOT EXISTS custody_underived_hot ON custody_runs(run_id) WHERE {CUSTODY_UNDERIVED}",
    "CREATE INDEX IF NOT EXISTS custody_request_root ON obligations(set_name,request_root_task_id,id) "
    "WHERE set_name IN ('custody_open','custody_refused')",
    f"CREATE INDEX IF NOT EXISTS custody_request_underived ON obligations(set_name,id) WHERE {REQUEST_UNDERIVED}",
)
# Old custody writers UPSERT facts without advancing the new sequence. Invalidate
# their projections in that SAME transaction; task REPLACE writers already do so.
INVALIDATE_CUSTODY = """CREATE TRIGGER IF NOT EXISTS custody_invalidate_hot
AFTER UPDATE OF facts ON custody_runs WHEN NEW.write_seq=OLD.write_seq
BEGIN UPDATE custody_runs SET facts_derived=NULL, root_task_id=NULL, billing_group_id=NULL,
output_unread=NULL, payload_held=NULL, write_seq=OLD.write_seq+1 WHERE run_id=NEW.run_id; END"""
INVALIDATE_REQUEST = """CREATE TRIGGER IF NOT EXISTS custody_request_invalidate_hot
AFTER UPDATE OF facts ON obligations WHEN NEW.set_name IN ('custody_open','custody_refused')
BEGIN UPDATE obligations SET request_root_task_id=NULL WHERE set_name=NEW.set_name AND id=NEW.id; END"""
SCHEMA = ";\n".join([
    *(f"ALTER TABLE {table} ADD COLUMN {name} {kind}" for table, columns in
      (("tasks", TASK_COLUMNS), ("custody_runs", CUSTODY_COLUMNS)) for name, kind in columns.items()),
    "ALTER TABLE obligations ADD COLUMN request_root_task_id TEXT",
    *INDEXES, INVALIDATE_CUSTODY, INVALIDATE_REQUEST,
]) + ";\n"


def upgrade(txn) -> None:
    """Schema-only addition inside the existing schema-version publication gate."""
    for table, columns in (("tasks", TASK_COLUMNS), ("custody_runs", CUSTODY_COLUMNS)):
        present = {row["name"] for row in txn.conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in present:
                txn.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    if "request_root_task_id" not in {row["name"] for row in txn.conn.execute("PRAGMA table_info(obligations)")}:
        txn.conn.execute("ALTER TABLE obligations ADD COLUMN request_root_task_id TEXT")
    # Schema 9 extends the underived predicate; replace its schema-6 definition.
    txn.conn.execute("DROP INDEX IF EXISTS tasks_underived_hot")
    for sql in (*INDEXES, INVALIDATE_CUSTODY, INVALIDATE_REQUEST):
        txn.conn.execute(sql)
    owed = any(txn.conn.execute(f"SELECT 1 FROM {table} INDEXED BY {index} WHERE {where} LIMIT 1").fetchone()
               for table, index, where in _UNDERIVED_PROBES)  # a fresh store has nothing to classify
    txn.set_meta("job:" + JOB, {"state": "pending" if owed else "done", "cursor": None, "done": 0, "total": None})


_UNDERIVED_PROBES = (("tasks", "tasks_underived_hot", TASK_UNDERIVED),
                     ("custody_runs", "custody_underived_hot", CUSTODY_UNDERIVED),
                     ("obligations", "custody_request_underived", REQUEST_UNDERIVED))


def own_document(task_id: str, body: Any) -> tuple:
    """``(document, "")`` when the stored BODY is TASK_ID's own JSON object, else ``(None, reason)``:
    ``body_missing``, ``body_unreadable``, ``body_not_object``, ``identity_missing`` or ``identity_mismatch``.
    Only such a document may derive this row's facts; another task's body would publish its root."""
    if body is None:
        return None, "body_missing"
    try:
        document = json.loads(body)
    except (ValueError, TypeError, RecursionError):
        return None, "body_unreadable"
    if not isinstance(document, dict):
        return None, "body_not_object"
    if str(document.get("task_id") or "") != task_id:
        return None, "identity_mismatch" if document.get("task_id") else "identity_missing"
    return document, ""


def lineage_root(document: dict) -> str:
    """The tree a task's own document names: its top-level root, else its metadata root, else itself."""
    meta = document.get("metadata")
    return str(document.get("root_task_id") or (meta.get("root_task_id") if isinstance(meta, dict) else "")
               or document["task_id"])


def task_columns(document: dict) -> dict:
    from ouroboros.task_status import stored_list_order_is_final

    fence = document.get("owner_pause") or {}
    pause = document.get("budget_pause")
    pause = pause if isinstance(pause, dict) else {}
    checkpoint = document.get("root_phase_checkpoint")
    wait = document.get("owner_wait")
    try:
        attempt = int(pause.get("task_attempt") or 0)
    except (TypeError, ValueError, OverflowError):
        attempt = 0  # the exact malformed value remains in budget_pause_facts
    return {
        "root_task_id": lineage_root(document),
        "owner_pause_state": str(fence.get("state") or "") if isinstance(fence, dict) else "",
        "owner_pause_fence_id": str(fence.get("fence_id") or "") if isinstance(fence, dict) else "",
        "owner_pause_facts": machine_json(fence),
        "budget_pause_state": str(pause.get("state") or ""), "budget_pause_attempt": attempt,
        "budget_pause_facts": machine_json(pause),
        "post_task_synthesis": str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else "",
        "owner_wait_state": str(wait.get("state") or "") if isinstance(wait, dict) else "",
        "owner_wait_reason": str(wait.get("reason") or "") if isinstance(wait, dict) else "",
        "control_valid": int(isinstance(document.get("status"), str) and bool(document["status"].strip())),
        "list_stable": int(stored_list_order_is_final(document)),
    }


def custody_columns(facts: dict) -> dict:
    """The former JSON predicates, including their conservative non-ordinary-JSON arm.

    Text/arrays/objects differ from numeric zero/one in SQLite. Keep that exact
    candidate superset: the existing Python custody consumers still decide truth.
    """
    try:
        json.dumps(facts, allow_nan=False)
        ordinary = True
    except ValueError:
        ordinary = False
    complete, consumed = facts.get("output_complete"), facts.get("output_consumed")
    numeric = (bool, int, float)
    unread = not (complete is None or isinstance(complete, numeric) and complete == 0
                  or isinstance(consumed, numeric) and consumed == 1)
    disposed, settled = facts.get("patch_disposed"), facts.get("settled")
    payload = facts.get("authority_source") == "skill_payload" and not (
        isinstance(settled, numeric) and settled == 1 and isinstance(disposed, str) and disposed)
    group = facts.get("billing_group")
    group_id = group.get("billing_group_id") if isinstance(group, dict) else None
    return {"root_task_id": facts.get("root_task_id") if isinstance(facts.get("root_task_id"), str) else None,
            "billing_group_id": str(group_id) if group_id is not None else None,
            "output_unread": int(not ordinary or unread), "payload_held": int(not ordinary or payload),
            "facts_derived": 1}


def request_root(facts: dict) -> str:
    request = facts.get("request")
    return str(request.get("root_task_id") or request.get("task_id") or "") if isinstance(request, dict) else ""


def request_columns(facts: dict) -> dict:
    return {"request_root_task_id": request_root(facts)}


def tree_requests(txn, tree: str, *, refused: bool = False) -> list:
    """Tree-local pending/refused requests, preserving each set's insertion order."""
    names = REQUEST_SETS if refused else REQUEST_SETS[:1]
    rows = []
    for name in names:
        if int(txn.meta("schema_version") or VERSION) < VERSION:
            rows.extend((name, key, facts) for key, facts in txn.set_rows(name).items() if request_root(facts) == tree)
            continue
        records = txn.conn.execute(
            "SELECT id,facts FROM obligations WHERE set_name=? AND id IN ("
            "SELECT id FROM obligations WHERE set_name IN ('custody_open','custody_refused') "
            "AND set_name=? AND request_root_task_id=? UNION SELECT id FROM obligations "
            f"INDEXED BY custody_request_underived WHERE {REQUEST_UNDERIVED} AND set_name=?) ORDER BY rowid",
            (name, name, tree, name)).fetchall()
        rows.extend((name, row["id"], facts) for row in records if request_root(facts := json.loads(row["facts"])) == tree)
    return rows


def _read_rows(txn, ids) -> dict:
    ids = sorted(set(map(str, ids)))
    if not ids:
        return {}
    modern = int(txn.meta("schema_version") or VERSION) >= VERSION
    # Control readers do not need list selection; its backfill must not make
    # already-derived pause/phase facts read bodies again.
    names = [name for name in TASK_COLUMNS if name != "list_stable"]
    columns = "," + ",".join(f"t.{name}" for name in names) if modern else ""
    cards, join = store_task_cards.select(txn)  # the card facts ride the hot row: the census needs no document
    # A derived row answers from its columns alone: the writer derived them from the body it validated in
    # its own transaction, and an action that needs body authority (Resume, adoption, every mutation)
    # decodes the body strictly in its own write. No body is read or validated for it.
    records = txn.conn.execute(
        "SELECT t.task_id,t.status,t.root_task_id,t.superseded_by,t.retry_task_id,t.write_seq,t.updated_at"
        f"{columns}{cards} FROM tasks AS t{join} WHERE t.task_id IN (SELECT value FROM json_each(?))",
        (json.dumps(ids),)).fetchall()
    found = {row["task_id"]: dict(row) for row in records}
    # Underived rows alone pay their old strict read, coherent with their sequence; a body that is not
    # this task's object keeps that row's authority unknown (no Resume is offered from it), not its store's.
    for tid, row in found.items():
        if not modern or any(row.get(name) is None for name in names):
            document, fault = own_document(tid, txn.body_text(tid))
            if fault:
                row["control_valid"] = 0
                continue
            row.update(task_columns(document))
    if set(ids) - found.keys():
        from ouroboros.store_import import OCCUPIED
        occupied = txn.meta(OCCUPIED) or {}
        for tid in set(ids) - found.keys():
            found[tid] = {"control_valid": 0} if occupied.get(tid) else None
    return found


_SNAPSHOT = contextvars.ContextVar("task_hot_facts_snapshot", default=None)


def _root_key(root):
    return os.path.abspath(root)


@contextlib.contextmanager
def snapshot(requests):
    """One short transaction per addressed store, before queue/consumer locks.

    A failed store stays unknown for this request; no per-member retries. This
    snapshot serves display only and must never span an authority mutation.
    """
    from ouroboros import store

    grouped, facts = {}, {}
    for root, task_id in requests:
        if task_id:
            grouped.setdefault(_root_key(root), set()).add(str(task_id))
    for root, ids in grouped.items():
        try:
            with store.open_store(root, write=False) as txn:
                rows = _read_rows(txn, ids)
            facts.update({(root, tid): row for tid, row in rows.items()})
        except Exception as exc:
            facts.update({(root, tid): exc for tid in ids})
    token = _SNAPSHOT.set(facts)
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def read(root, task_id: str) -> dict | None:
    from ouroboros import store
    from ouroboros.task_results import validate_task_id

    task_id = validate_task_id(task_id)
    key, held = (_root_key(root), task_id), _SNAPSHOT.get()
    if held is not None and key in held:
        row = held[key]
    else:
        with store.open_store(root, write=False) as txn:
            row = _read_rows(txn, [task_id])[task_id]
    if isinstance(row, Exception):
        raise row
    if row is not None and not row["control_valid"]:
        raise ValueError(f"task result authority is unreadable or invalid: task:{task_id}")
    return copy.deepcopy(row)


def controls(root, task_id: str) -> dict:
    row = read(root, task_id)
    if row is None:
        return {}
    return {"status": row["status"], "present": True,
            "owner_pause": json.loads(row["owner_pause_facts"]),
            "budget_pause": json.loads(row["budget_pause_facts"])}


def underived(root: Any) -> bool:
    """Whether any row still lacks its hot-fact columns: indexed probes in one read transaction."""
    from ouroboros import store

    try:
        with store.open_store(root, write=False) as txn:
            return any(txn.conn.execute(f"SELECT 1 FROM {table} INDEXED BY {index} WHERE {where} LIMIT 1").fetchone()
                       for table, index, where in _UNDERIVED_PROBES)
    except sqlite3.OperationalError:
        return True  # a schema below the hot-fact columns: the runner upgrades it first


# Rows the derivation classifies per write transaction (and per recorded cursor): one commit per batch,
# never per row, so a network-backed data root (Colab on Google Drive) pays a few hundred commits.
_DERIVE_BATCH = 64


def run_hot_facts(root: Any, *, progress=None, should_stop=None) -> dict:
    """Maintenance-child entry: resumable sequence-CAS classification, no body rewrites."""
    from ouroboros import maintenance_jobs, store
    from ouroboros.store_import import upgrade_current_schema

    upgrade_current_schema(root)
    stop = should_stop or (lambda: False)
    tables = {
        "tasks": ("tasks", "task_id", TASK_UNDERIVED, "tasks_underived_hot", "write_seq", task_columns),
        "custody_runs": ("custody_runs", "run_id", CUSTODY_UNDERIVED, "custody_underived_hot", "write_seq", custody_columns),
        **{name: ("obligations", "id", f"{REQUEST_UNDERIVED} AND set_name='{name}'",
                  "custody_request_underived", "facts", request_columns) for name in REQUEST_SETS},
    }

    def remaining(txn):
        return any(txn.conn.execute(f"SELECT 1 FROM {table} INDEXED BY {index} WHERE {where} LIMIT 1").fetchone()
                   for table, _id, where, index, _stamp, _derive in tables.values())

    def input_stamp(value):
        # Obligation rows have no sequence. Compare their exact source bytes on
        # write, but retain only their digest in a damaged-input waiting fact.
        return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else value

    with store.name_lock(root, "hot_facts.lock", timeout_sec=store.STORE_WAIT_SEC):
        row = maintenance_jobs.job(root, JOB) or {}
        with store.open_store(root, write=False) as txn:
            owed = remaining(txn)
        if row.get("state") == "done" and not owed:
            return row

        def record(**facts):
            updated = maintenance_jobs.record(root, JOB, **facts)
            if progress is not None:
                progress(updated)
            return updated

        cursor = row.get("cursor") or {"table": "tasks", "after": ""}
        done = int(row.get("done") or 0)
        if row.get("state") == "done":
            cursor = {"table": "tasks", "after": ""}  # a rollback writer introduced new NULLs
        if stop():
            return record(state="pending", cursor=cursor, done=done)
        waiting = row.get("waiting_on") or {}
        if row.get("state") == "failed" and waiting:
            table, key, where, _index, stamp, _derive = tables[waiting["table"]]
            with store.open_store(root, write=False) as txn:
                current = txn.conn.execute(f"SELECT {stamp} FROM {table} WHERE {key}=? AND ({where})",
                                           (waiting["id"],)).fetchone()
            if current is not None and input_stamp(current[0]) == waiting["stamp"]:
                return row  # unchanged damaged input is not retried by a timer
        record(state="running", cursor=cursor, done=done, total=None, error=None, waiting_on=None)
        try:
            start = list(tables).index(cursor["table"])
            for position, (scope, (table, identity, where, index, stamp, derive)) in enumerate(tables.items()):
                if position < start:
                    continue
                after = cursor["after"] if cursor["table"] == scope else ""
                cursor = {"table": scope, "after": after}
                while not stop():
                    with store.open_store(root, write=False) as txn:
                        batch = txn.conn.execute(
                            f"SELECT {identity},{stamp} FROM {table} INDEXED BY {index} "
                            f"WHERE ({where}) AND {identity}>? ORDER BY {identity} LIMIT ?", (after, _DERIVE_BATCH)).fetchall()
                        sources = [(item[0], item[1], txn.body_text(item[0]) if table == "tasks" else
                                    item[1] if table == "obligations" else txn.conn.execute(
                                        "SELECT facts FROM custody_runs WHERE run_id=?", (item[0],)).fetchone()[0])
                                   for item in batch]
                    if not sources:
                        with store.open_store(root, write=False) as txn:
                            more = txn.conn.execute(f"SELECT 1 FROM {table} INDEXED BY {index} WHERE {where} LIMIT 1").fetchone()
                        if more:
                            after = ""  # sequence-CAS lost to an old writer/handoff behind the cursor
                            continue
                        break
                    derived = []
                    try:
                        for tid, seq, raw in sources:
                            if stop():
                                break
                            waiting = {"table": scope, "id": tid, "stamp": input_stamp(seq)}
                            document, fault = own_document(tid, raw) if table == "tasks" else (json.loads(raw), "")
                            if fault:  # never another task's root: this damaged input waits like any other
                                raise ValueError(f"task body {fault}: task:{tid}")
                            derived.append((tid, seq, derive(document)))
                    finally:  # what was derived before a stop or a damaged input commits with its cursor
                        if derived:
                            with store.open_store(root, write=True) as txn:
                                for tid, seq, values in derived:
                                    done += txn.conn.execute(
                                        f"UPDATE {table} SET {','.join(name + '=?' for name in values)} "
                                        f"WHERE {identity}=? AND {stamp}=? AND ({where})",
                                        (*values.values(), tid, seq)).rowcount
                            after = derived[-1][0]
                            cursor = {"table": scope, "after": after}
                            record(cursor=cursor, done=done)
                    if stop():
                        return record(state="pending", cursor=cursor, done=done)
                if stop():
                    return record(state="pending", cursor=cursor, done=done)
            with store.open_store(root, write=False) as txn:
                owed = remaining(txn)
            if owed:  # an old writer added NULLs in a family already passed
                return record(state="pending", cursor={"table": "tasks", "after": ""}, done=done)
            return record(state="done", cursor=cursor, done=done, waiting_on=None)
        except BaseException as exc:
            record(state="failed" if isinstance(exc, Exception) else "pending", cursor=cursor,
                   done=done, error=f"{type(exc).__name__}: {exc}", waiting_on=waiting)
            raise
