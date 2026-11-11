"""Addressed control/phase facts and the resumable ``hot_facts`` derivation job.

Whole-document writers derive these columns in their owning transaction, so a derived
row is read from its columns alone (no body read, no validation). NULL means an older
writer has not derived them; a passive reader only reads that row's body. Historical
classification belongs to ``run_hot_facts``, never boot.
The request snapshot is short lived and released before any consumer runs.

The schema gate adds the columns without decoding bodies. ``run_hot_facts`` classifies
underived rows under a sequence/source CAS, so a racing writer's row is left to that
writer and a damaged input waits for its own change; a body that is not its own task's
object, or whose status the writer refuses (``own_document``), is such input and derives
nothing, never another task's root.
The pass names each damaged row (``damaged``, the first also ``waiting_on``), still derives the
independent rows after it and never reads a named row again while it is unchanged, then fails. The
dispatcher's look (``hot_facts_due``) compares current facts through the partial indexes: exactly those
rows, unchanged, are no work; any other underived row, or a change to a named one, is.
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
from typing import Any

from ouroboros import store_task_cards
from ouroboros.utils import machine_json

VERSION = 6
LIST_VERSION = 9
ORIGIN_VERSION = 10
CHAT_VERSION = 11
VERSIONS = (VERSION, LIST_VERSION, ORIGIN_VERSION, CHAT_VERSION)  # each schema step that adds these columns
JOB = "hot_facts"
TASK_COLUMNS = {
    "owner_pause_state": "TEXT", "owner_pause_fence_id": "TEXT", "owner_pause_facts": "TEXT",
    "budget_pause_state": "TEXT", "budget_pause_attempt": "INTEGER", "budget_pause_facts": "TEXT",
    "post_task_synthesis": "TEXT", "owner_wait_state": "TEXT", "owner_wait_reason": "TEXT",
    "control_valid": "INTEGER", "list_stable": "INTEGER",
    # The owner message the task's own top-level ``origin_message_ref`` names (``dialogue_store.origin_key``),
    # "" for none: discovery for the Host Service operation reader (``task_store.origin_documents``), never authority.
    "origin": "TEXT",
    # The chat the task was admitted in, as the room rule reads it (``admitted_chat``), "" for none.
    "chat_id": "TEXT",
}
SELECTION_COLUMNS = ("list_stable", "origin", "chat_id")  # selection, discovery and room facts no control reader needs
CUSTODY_COLUMNS = {
    "root_task_id": "TEXT", "billing_group_id": "TEXT", "output_unread": "INTEGER",
    "payload_held": "INTEGER", "facts_derived": "INTEGER", "write_seq": "INTEGER NOT NULL DEFAULT 0",
}
# An older body's INDEXED BY probes name this predicate on a shared store: a later column keeps its own index.
OWN_UNDERIVED = {"origin": "tasks_underived_origin", "chat_id": "tasks_underived_chat"}
TASK_UNDERIVED = " OR ".join(f"{name} IS NULL" for name in (*TASK_COLUMNS, "root_task_id") if name not in OWN_UNDERIVED)
CUSTODY_UNDERIVED = "facts_derived IS NULL"
REQUEST_SETS = ("custody_open", "custody_refused")
REQUEST_UNDERIVED = "set_name IN ('custody_open','custody_refused') AND request_root_task_id IS NULL"
INDEXES = (
    f"CREATE INDEX IF NOT EXISTS tasks_underived_hot ON tasks(task_id) WHERE {TASK_UNDERIVED}",
    "CREATE INDEX IF NOT EXISTS tasks_list_stable ON tasks(list_stable, status, ts, task_id)",
    "CREATE INDEX IF NOT EXISTS tasks_underived_root ON tasks(task_id) WHERE root_task_id IS NULL",
    "CREATE INDEX IF NOT EXISTS tasks_origin ON tasks(origin) WHERE origin<>''",
    *(f"CREATE INDEX IF NOT EXISTS {index} ON tasks(task_id) WHERE {name} IS NULL" for name, index in OWN_UNDERIVED.items()),
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
    owed = not _settled(txn, {})  # a fresh store has nothing to classify
    txn.set_meta("job:" + JOB, {"state": "pending" if owed else "done", "cursor": None, "done": 0, "total": None})


# Each family of derivable rows: its table, identity, the (underived predicate, partial index) probes that
# select its rows without visiting a derived one, and the source stamp its compare-and-set holds.
_FAMILIES = {
    "tasks": ("tasks", "task_id", ((TASK_UNDERIVED, "tasks_underived_hot"),
                                   *((f"{name} IS NULL", index) for name, index in OWN_UNDERIVED.items())), "write_seq"),
    "custody_runs": ("custody_runs", "run_id", ((CUSTODY_UNDERIVED, "custody_underived_hot"),), "write_seq"),
    **{name: ("obligations", "id", ((f"{REQUEST_UNDERIVED} AND set_name='{name}'", "custody_request_underived"),),
              "facts") for name in REQUEST_SETS},
}


def _input_stamp(value: Any) -> Any:
    # Obligation rows have no sequence. Compare their exact source bytes on
    # write, but retain only their digest in a damaged-input waiting fact.
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else value


def _underived_rows(txn, scope: str, limit: int, after: str = "") -> list:
    """``(identity, source stamp)`` of the family's first LIMIT underived rows past AFTER, by identity: each
    probe reads at most LIMIT index entries, and a row among the union's first LIMIT is among its probe's."""
    table, identity, probes, stamp = _FAMILIES[scope]
    rows = {}
    for where, index in probes:
        rows.update((row[0], row[1]) for row in txn.conn.execute(
            f"SELECT {identity},{stamp} FROM {table} INDEXED BY {index} WHERE ({where}) AND {identity}>? "
            f"ORDER BY {identity} LIMIT ?", (after, limit)))
    return sorted(rows.items())[:limit]


def _settled(txn, damage: dict) -> bool:
    """Whether the only underived rows are exactly DAMAGE's, each as it was named; bounded by what it names."""
    for scope in _FAMILIES:
        held = {key: item["stamp"] for (name, key), item in damage.items() if name == scope}
        if {key: _input_stamp(stamp) for key, stamp in _underived_rows(txn, scope, len(held) + 1)} != held:
            return False
    return True


def _unchanged(txn, damage: dict) -> dict:
    """DAMAGE's rows still underived, each as it was named: a repaired, removed or rewritten row leaves it."""
    kept = {}
    for (scope, key), item in damage.items():
        table, identity, probes, stamp = _FAMILIES[scope]
        where = " OR ".join(f"({predicate})" for predicate, _index in probes)
        record = txn.conn.execute(f"SELECT {stamp} FROM {table} WHERE {identity}=? AND ({where})", (key,)).fetchone()
        if record is not None and _input_stamp(record[0]) == item["stamp"]:
            kept[(scope, key)] = item
    return kept


def _named_damage(row: dict) -> dict:
    return {(item["table"], item["id"]): item for item in row.get("damaged") or ()}


def own_document(task_id: str, body: Any) -> tuple:
    """``(document, "")`` when the stored BODY is TASK_ID's own JSON object, else ``(None, reason)``:
    ``body_missing``, ``body_unreadable``, ``body_not_object``, ``identity_missing``, ``identity_mismatch`` or
    ``status_unreadable`` (an array or object status: the writer refuses one, ``task_columns`` cannot classify it).
    Only such a document may derive this row's facts; another task's body would publish its root. A scalar
    status that is no valid status is still this row's document: it derives a flagged row (``control_valid=0``)."""
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
    if isinstance(document.get("status"), (dict, list)):
        return None, "status_unreadable"
    return document, ""


def _own_facts(raw: Any) -> tuple:
    """``(facts, "")`` when a custody or request row's stored text is its JSON object, else ``(None, reason)``."""
    try:
        facts = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None, "facts_unreadable"
    return (facts, "") if isinstance(facts, dict) else (None, "facts_not_object")


def lineage_root(document: dict) -> str:
    """The tree a task's own document names: its top-level root, else its metadata root, else itself."""
    meta = document.get("metadata")
    return str(document.get("root_task_id") or (meta.get("root_task_id") if isinstance(meta, dict) else "")
               or document["task_id"])


def admitted_chat(value: Any) -> str:
    """A task's own top-level ``chat_id`` as the room rule reads it (``projects_registry._read_lineage``), as the
    integer's text: an integer or its text, a JSON real's integer part, a boolean's 1 or 0; "" for none or any
    other value, as the rule's read of the body answered."""
    try:
        return "" if value is None or value == "" else str(int(value))
    except (TypeError, ValueError, OverflowError):
        return ""


def task_columns(document: dict) -> dict:
    from ouroboros.dialogue_store import origin_key
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
        "origin": origin_key(document.get("origin_message_ref")),
        "chat_id": admitted_chat(document.get("chat_id")),
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
    # Control readers do not need list selection or origin discovery; their backfill
    # must not make already-derived pause/phase facts read bodies again.
    names = [name for name in TASK_COLUMNS if name not in SELECTION_COLUMNS]
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
_STATUSES = contextvars.ContextVar("task_statuses_read_ahead", default=None)
#: Task writes in THIS process whose document carries control facts (a budget pause row, an owner fence or a retry
#: lineage), counted in the writer's transaction (``obligations.task_written``). A reader that keeps a decision it
#: derived from those facts keeps it only while this count stands (``supervisor/dispatch_choice``): a pre-commit
#: re-read can keep a stale answer until the next such write, so the keeper still decides again at its own handoff.
CONTROL_WRITES = [0]
_CONTROL_COLUMNS = ("owner_pause_facts", "budget_pause_facts", "superseded_by", "retry_task_id")


def note_written(columns) -> None:
    if columns and any(columns.get(name) not in (None, "", "{}") for name in _CONTROL_COLUMNS):
        CONTROL_WRITES[0] += 1


class NotRead(LookupError):
    """A strict read-ahead (``serving(..., strict=True)``) was asked a row it did not read: unknown, never SQLite."""


class _Strict(dict):
    pass


def _root_key(root):
    return os.path.abspath(root)


def read_ahead(requests) -> dict:
    """The rows of ``(root, task_id)`` requests, one short transaction per addressed store, before any lock.
    A failed store keeps its rows' exception (unknown); nothing is retried per member."""
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
    return facts


@contextlib.contextmanager
def serving(facts, *, strict: bool = False):
    """``read``/``controls`` answer from ``facts`` (``read_ahead``); a strict serving answers a row it lacks with
    ``NotRead`` instead of SQLite, so a reader under the queue lock never opens the store. Facts serve a choice,
    never an authority mutation: leave the context before a write that must read its own row again."""
    token = _SNAPSHOT.set(_Strict(facts) if strict else facts)
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


@contextlib.contextmanager
def snapshot(requests):
    """One short transaction per addressed store, before queue/consumer locks.

    A failed store stays unknown for this request; no per-member retries. This
    snapshot serves display only and must never span an authority mutation.
    """
    with serving(read_ahead(requests)):
        yield


def read(root, task_id: str) -> dict | None:
    from ouroboros import store
    from ouroboros.task_results import validate_task_id

    task_id = validate_task_id(task_id)
    key, held = (_root_key(root), task_id), _SNAPSHOT.get()
    if held is not None and key in held:
        row = held[key]
    elif isinstance(held, _Strict):
        raise NotRead(f"task:{task_id} was not read ahead")
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


@contextlib.contextmanager
def statuses_ahead(root, task_ids):
    """Read ``statuses`` of these ids now, before the caller takes its lock: a ``statuses`` call of the same root
    inside answers them from this read and reads only ids added since (the assignment tick's one SELECT, outside Q)."""
    token = _STATUSES.set((_root_key(root), statuses(root, task_ids)))
    try:
        yield
    finally:
        _STATUSES.reset(token)


def statuses(root, task_ids) -> dict:
    """Each id's stored status in ONE short read, for a pass over many queue rows: the status, ``None``
    for no row, or the exception a single read of that id raises. No body is decoded for a derived row;
    a store that cannot answer maps every id to its failure: unknown, never absent."""
    from ouroboros import store
    from ouroboros.store_import import OCCUPIED
    from ouroboros.task_results import validate_task_id

    answers: dict = {}
    ids = []
    for task_id in task_ids:
        try:
            ids.append(validate_task_id(task_id))
        except ValueError as exc:
            answers[str(task_id)] = exc
    ahead = _STATUSES.get()
    if ahead is not None and ahead[0] == _root_key(root):
        answers.update({tid: ahead[1][tid] for tid in ids if tid in ahead[1]})
        ids = [tid for tid in ids if tid not in ahead[1]]
    if not ids:
        return answers
    ids = sorted(set(ids))
    try:
        with store.open_store(root, write=False) as txn:
            if int(txn.meta("schema_version") or VERSION) < VERSION:
                rows = _read_rows(txn, ids)
            else:
                rows = {record["task_id"]: dict(record) for record in txn.conn.execute(
                    "SELECT task_id,status,control_valid FROM tasks WHERE task_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(ids),))}
                underived = [tid for tid, row in rows.items() if row["control_valid"] is None]
                rows.update(_read_rows(txn, underived))  # an older writer's row alone pays its body read
                occupied = (txn.meta(OCCUPIED) or {}) if set(ids) - rows.keys() else {}
                rows.update({tid: {"control_valid": 0} if occupied.get(tid) else None for tid in set(ids) - rows.keys()})
    except Exception as exc:
        return {**answers, **{tid: exc for tid in ids}}
    for tid, row in rows.items():
        if isinstance(row, Exception):
            answers[tid] = row
        elif row is not None and not row["control_valid"]:
            answers[tid] = ValueError(f"task result authority is unreadable or invalid: task:{tid}")
        else:
            answers[tid] = None if row is None else row["status"]
    return answers


# A saved cold request or receipt lives on a live pause row (``sleep_custody.accepted_receipt``).
COLD_CUSTODY_STATES = ("requested", "accepted")
# Non-ordinary JSON (``machine_json`` may write NaN, which ``json_valid`` rejects) passes: the caller decides.
_COLD_ROWS = ("SELECT task_id, budget_pause_facts FROM tasks INDEXED BY tasks_budget_pause "
              "WHERE budget_pause_state<>'' AND budget_pause_state IN ('pausing','paused','resume_granted') "
              "AND control_valid=1 AND CASE WHEN json_valid(budget_pause_facts) THEN "
              "json_extract(budget_pause_facts,'$.external_custody.state') IN ('requested','accepted') ELSE 1 END")


def cold_custody_rows(root, task_ids) -> dict:
    """Among ``task_ids``, each pause row carrying a saved cold request or receipt, in ONE indexed read:
    the cost follows the rows that carry one, never how many ids the caller holds. A row an older writer has not
    derived pays its own body read; a row whose control facts are invalid answers nothing here, so no
    acceptance is inferred from it. Raises when the store cannot answer; the caller keeps what it knows."""
    from ouroboros import store

    wanted = {str(task_id) for task_id in task_ids}
    with store.open_store(root, write=False) as txn:
        if int(txn.meta("schema_version") or VERSION) < VERSION:
            derived, underived = [], sorted(wanted)
        else:
            derived = [(row["task_id"], row["budget_pause_facts"]) for row in txn.conn.execute(_COLD_ROWS)
                       if row["task_id"] in wanted]
            underived = sorted(wanted & {row["task_id"] for row in txn.conn.execute(
                f"SELECT task_id FROM tasks INDEXED BY tasks_underived_hot WHERE {TASK_UNDERIVED}")})
        held = _read_rows(txn, underived)
    rows = {tid: json.loads(facts) for tid, facts in derived}
    for tid, row in held.items():
        if isinstance(row, Exception):
            raise row
        if row and row.get("control_valid"):
            pause = json.loads(row["budget_pause_facts"])
            if isinstance(pause, dict) and (pause.get("external_custody") or {}).get("state") in COLD_CUSTODY_STATES:
                rows[tid] = pause
    return {tid: pause for tid, pause in rows.items() if isinstance(pause, dict)}


def hot_facts_due(root: Any, row: dict, child: bool) -> bool:
    """Whether a pass has work, from current facts (``maintenance._hot_facts_due``): indexed probes in one read
    transaction, no body read. A store below the column is owed: the runner upgrades it first. A pass that
    finished named its damage (``damage_only``): exactly those rows, unchanged, are no work, for any process.
    A pass that stopped on a fault proves nothing about the rest: it spawns nothing itself and runs again
    whenever the child runs (``child``), as a repeated fault would otherwise spawn it on every look."""
    from ouroboros import store

    with store.open_store(root, write=False) as txn:
        schema = txn.meta("schema_version")
        if schema is not None and int(schema) < VERSIONS[-1]:
            return True
        if row.get("state") != "failed":
            return not _settled(txn, {})
        return not _settled(txn, _named_damage(row)) if row.get("damage_only") else child


# Rows the derivation classifies per write transaction (and per recorded cursor): one commit per batch,
# never per row, so a network-backed data root (Colab on Google Drive) pays a few hundred commits.
_DERIVE_BATCH = 64


def run_hot_facts(root: Any, *, progress=None, should_stop=None) -> dict:
    """Maintenance-child entry: resumable sequence-CAS classification, no body rewrites."""
    from ouroboros import maintenance_jobs, store
    from ouroboros.store_import import upgrade_current_schema

    upgrade_current_schema(root)
    stop = should_stop or (lambda: False)
    derivations = {"tasks": task_columns, "custody_runs": custody_columns,
                   **{name: request_columns for name in REQUEST_SETS}}

    with store.name_lock(root, "hot_facts.lock", timeout_sec=store.STORE_WAIT_SEC):
        row = maintenance_jobs.job(root, JOB) or {}
        debt = _named_damage(row)
        with store.open_store(root, write=False) as txn:
            if row.get("state") == "done" and _settled(txn, {}):
                return row
            if row.get("state") == "failed" and row.get("damage_only") and _settled(txn, debt):
                return row  # unchanged damaged input is not retried by a timer

        def record(**facts):
            updated = maintenance_jobs.record(root, JOB, **facts)
            if progress is not None:
                progress(updated)
            return updated

        first = {"table": "tasks", "after": ""}
        # A pass resumes at its cursor; a finished or failed one starts again (a rollback writer's NULLs, a
        # changed damaged row), selecting only rows still underived. Either keeps the damage it named.
        cursor = row["cursor"] if row.get("state") in ("pending", "running") and row.get("cursor") else first
        done = int(row.get("done") or 0)
        if stop():
            return record(state="pending", cursor=cursor, done=done)
        waiting = row.get("waiting_on") or {}
        record(state="running", cursor=cursor, done=done, total=None, error=None, waiting_on=None, damage_only=None,
               damaged=list(debt.values()) or None)
        try:
            start = list(_FAMILIES).index(cursor["table"])
            for position, (scope, (table, identity, probes, stamp)) in enumerate(_FAMILIES.items()):
                if position < start:
                    continue
                where = " OR ".join(f"({predicate})" for predicate, _index in probes)
                after = cursor["after"] if cursor["table"] == scope else ""
                cursor = {"table": scope, "after": after}
                while not stop():
                    with store.open_store(root, write=False) as txn:
                        batch = _underived_rows(txn, scope, _DERIVE_BATCH, after)
                        # A named row, unchanged, is passed without reading it again.
                        sources = {key: (txn.body_text(key) if table == "tasks" else raw if table == "obligations" else
                                         txn.conn.execute("SELECT facts FROM custody_runs WHERE run_id=?",
                                                          (key,)).fetchone()[0])
                                   for key, raw in batch
                                   if (debt.get((scope, key)) or {}).get("stamp") != _input_stamp(raw)}
                    if not batch:
                        known = {key for name, key in debt if name == scope}
                        with store.open_store(root, write=False) as txn:
                            more = any(key not in known for key, _stamp in _underived_rows(txn, scope, len(known) + 1))
                        if more:
                            after = ""  # sequence-CAS lost to an old writer/handoff behind the cursor
                            continue
                        break
                    derived, seen = [], ""
                    try:
                        for key, raw in batch:
                            if stop():
                                break
                            if key in sources:
                                waiting = {"table": scope, "id": key, "stamp": _input_stamp(raw)}
                                document, fault = (own_document(key, sources[key]) if table == "tasks"
                                                   else _own_facts(sources[key]))
                                if fault:  # never another task's root: this damaged input waits by name, alone
                                    debt[(scope, key)] = {**waiting, "reason": fault}
                                else:
                                    debt.pop((scope, key), None)
                                    derived.append((key, raw, derivations[scope](document)))
                            seen = key
                    finally:  # what was derived before a stop or a derivation fault commits with its cursor
                        if derived:
                            with store.open_store(root, write=True) as txn:
                                for key, raw, values in derived:
                                    done += txn.conn.execute(
                                        f"UPDATE {table} SET {','.join(name + '=?' for name in values)} "
                                        f"WHERE {identity}=? AND {stamp}=? AND ({where})",
                                        (*values.values(), key, raw)).rowcount
                        if seen:
                            after = seen
                            cursor = {"table": scope, "after": after}
                            record(cursor=cursor, done=done, damaged=list(debt.values()) or None)
                    if stop():
                        return record(state="pending", cursor=cursor, done=done)
                if stop():
                    return record(state="pending", cursor=cursor, done=done)
            with store.open_store(root, write=False) as txn:
                debt = _unchanged(txn, debt)
                exact = _settled(txn, debt)
        except BaseException as exc:
            # A held store is no failure: the job stays due, as ``maintenance._contention`` reports it.
            held = isinstance(exc, store.StoreBusy) or getattr(exc, "reason", None) == "contention"
            record(state="failed" if isinstance(exc, Exception) and not held else "pending", cursor=cursor, done=done,
                   error=f"{type(exc).__name__}: {exc}", waiting_on=waiting, damaged=list(debt.values()) or None)
            raise
        if not exact:  # an old writer added NULLs in a family already passed, or rewrote a named row
            return record(state="pending", cursor=first, done=done, damaged=list(debt.values()) or None)
        if not debt:
            return record(state="done", cursor=cursor, done=done, waiting_on=None, damaged=None)
        named = next(iter(debt.values()))
        error = (f"task body {named['reason']}: task:{named['id']}" if named["table"] == "tasks"
                 else f"{named['table']} row {named['reason']}: {named['id']}")
        record(state="failed", cursor=cursor, done=done, error=f"ValueError: {error}", waiting_on=named,
               damage_only=True, damaged=list(debt.values()))
        raise ValueError(error)
