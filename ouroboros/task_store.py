"""Task result documents in the current-facts store: the primitives every consumer uses.

Readers select typed facts before decoding only the bodies they need, after closing
the transaction. ``patch_document`` owns document updates; ``patch_handoff`` /
``put_handoffs`` write one operation row of the ``store.ROW_BACKED_FIELDS``
projections (a tool claim, a model wait, a retired consumer) without a body write;
``load`` projects the rows back.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

from ouroboros import store, store_task_cards
from ouroboros.store import Txn, open_store
from ouroboros.store_capture import Captured, capture

FACT_KEYS = ("task_id", "status", "ts", "updated_at", "delegation_role", "parent_task_id", "project_id", "root_task_id",
             "child_drive_root", "headless_child_drive_root", "superseded_by", "retry_task_id",
             "presence_binding_id", "presence_authority_recorded")
_NEWEST_ORDER = "ORDER BY ts DESC, task_id DESC"  # NULL ts sorts last: a row without one is oldest
_PAGE = 256  # documents per body capture: one aggregate far below SQLite's string limit at stored body sizes;
# a page past that limit halves (``_fetch``). A physical bound, not a policy.


def _newest(column: str) -> Callable[[Dict[str, Any]], tuple]:
    """``ORDER BY <column> DESC, task_id DESC`` (``_NEWEST_ORDER`` for ``ts``) restored after a capture
    (sorted with ``reverse``): a NULL ``column`` last."""
    return lambda record: (record[column] is not None, record[column] or "", record["task_id"])


def read_document(root: Any, task_id: str, *, txn: Optional[Txn] = None, signed: bool = False) -> Any:
    """One coherent document; implicit reads decode only after releasing the store.

    A supplied transaction keeps its own read/modify/write snapshot and lifetime.
    ``signed=True`` returns ``(document, signature)`` captured in that same read
    transaction, so an ownership check compares the stamp of the body it holds and
    never a second read's.
    """
    if txn is not None:
        document = txn.load(str(task_id))
        return (document, signature(root, task_id, txn=txn)) if signed else document
    task_id = str(task_id)
    with open_store(root, write=False) as txn:
        stamp = signature(root, task_id, txn=txn) if signed else None
        raw = txn.raw(task_id)
    document = store.materialize(task_id, *raw)
    return (document, stamp) if signed else document


def exists(root: Any, task_id: str) -> bool:
    with open_store(root, write=False) as txn:
        return txn.row(str(task_id), "task_id") is not None


def receipt_anchor(root: Any, task_id: str) -> Optional[Tuple[str, bool]]:
    """``(root_task_id, root_has_swarm_receipt)`` of one task without decoding a body.

    The root is the task's typed column, else the body's ``metadata.root_task_id``,
    else the task itself; ``None`` for an absent task. A publication hook asks this
    instead of parsing two task results merely to learn whether a receipt exists."""
    with open_store(root, write=False) as txn:
        record = txn.conn.execute(
            "SELECT COALESCE(NULLIF(t.root_task_id, ''), NULLIF(json_extract(b.body, '$.metadata.root_task_id'), ''),"
            " t.task_id) AS root_id FROM tasks AS t JOIN task_bodies AS b ON b.task_id = t.task_id WHERE t.task_id = ?",
            (str(task_id),)).fetchone()
        if record is None:
            return None
        root_id = str(record["root_id"])
        receipt = txn.conn.execute("SELECT json_type(body, '$.swarm_receipt') AS kind FROM task_bodies WHERE task_id = ?",
                                   (root_id,)).fetchone()
    return root_id, bool(receipt is not None and receipt["kind"] not in (None, "null"))


def signature(root: Any, task_id: str, *, txn: Optional[Txn] = None) -> Optional[Tuple[int, str]]:
    """``(write_seq, updated_at)``: changes on every write; ``None`` for an absent task. The key
    the memos compare that used to compare a result file's stat."""
    if txn is not None:
        return txn.row(str(task_id), "write_seq", "updated_at")
    with open_store(root, write=False) as txn:
        return signature(root, task_id, txn=txn)


def terminal_writes_since(root: Any, counter: int) -> List[str]:
    """The task ids whose rows were written terminal after the terminal-write counter stood at COUNTER
    (``obligations.task_written`` stamps each such row with the counter's new value, ``terminal_seq``),
    in that order, through the ``tasks_terminal_seq`` index: what a pass keyed on the counter looks at
    again, never the rows it did not move."""
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, "SELECT task_id, terminal_seq FROM tasks INDEXED BY tasks_terminal_seq "
                           "WHERE terminal_seq>?", (int(counter),), ("task_id", "terminal_seq"))
    ordered = captured.rows(key=lambda record: (record["terminal_seq"], record["task_id"]))
    return [record["task_id"] for record in ordered]


def delete_document(root: Any, task_id: str) -> bool:
    """Remove one task's row and body (a scheduling refusal undoing its own requested row)."""
    with open_store(root, write=True) as txn:
        present = txn.row(str(task_id), "task_id") is not None
        txn.delete_task(str(task_id))
        return present


def patch_document(root: Any, task_id: str, mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Mutate CURRENT in one write; None writes nothing. Return the current document."""
    from ouroboros.obligations import pending_drive_rule

    task_id = str(task_id)
    seen: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seen["current"] = current
        return mutator(current)

    updated = store.update_task(root, task_id, _mutate, pending_drive=pending_drive_rule)
    return seen.get("current", {}) if updated is None else {**updated, "task_id": task_id}


def _fetch(txn: Txn, where: str = "", params: Iterable[Any] = ()) -> List[Tuple[Captured, Captured]]:
    """The matching rows (id, body, card columns) and their handoff rows as stored, captured inside the
    caller's read transaction (one snapshot; ``store_capture``): all the store is held for. A reader decodes
    AFTER its transaction closed (``_documents``), so a writer's COMMIT never waits behind a listing's
    ``json.loads`` or a row-by-row step. The selected ids come first (ids only), then bodies and handoff
    rows by pages of ids: every body of a store in one aggregate would pass SQLite's string limit."""
    clause = f"WHERE {where}" if where else ""
    cards, join = store_task_cards.select(txn)
    tables = f"FROM tasks AS t JOIN task_bodies AS b ON b.task_id = t.task_id{join}"
    # ``write_seq`` is in no index: the selection reads the rows it selects, as the listing always did, so the planner
    # keeps the WHERE's index; an id-only select can scan a covering index of every task instead once ANALYZE
    # statistics make that look cheaper (tests/test_cost_contract.py B7).
    selected = capture(txn.conn, f"SELECT t.task_id, t.write_seq {tables} {clause}", params, ("task_id", "write_seq"))
    ids = sorted(row[0] for row in selected.tuples())
    columns = ("task_id", "body", *(store_task_cards.FIELDS if join else ()))
    select = f"SELECT t.task_id, b.body{cards} {tables} WHERE t.task_id IN (SELECT value FROM json_each(?))"
    return [(rows, _handoff_capture(txn, page)) for page, rows in _pages(txn, ids, select, columns)]


def _pages(txn: Txn, ids: List[str], select: str, columns: Tuple[str, ...]) -> Iterator[Tuple[str, Captured]]:
    """``(ids page, capture)`` of SELECT (one ``?``: a JSON array of task ids) over IDS in pages of ``_PAGE``,
    inside the caller's read (one snapshot): the bodies of many tasks in one aggregate could pass SQLite's string
    limit. A page past it halves; one document past it, or a NOMEM that rolled the read back, raises (the store
    types it), never an answer without that row."""
    start, size = 0, _PAGE
    while start < len(ids):
        page = json.dumps(ids[start:start + size])
        try:
            rows = capture(txn.conn, select, (page,), columns)
        except (sqlite3.DataError, MemoryError):  # past SQLite's string limit: the same ids in smaller pages,
            if size == 1 or not txn.conn.in_transaction:  # in the same snapshot (a NOMEM may have rolled it back)
                raise
            size = max(1, size // 2)
            continue
        yield page, rows
        start += size


def _handoff_capture(txn: Txn, ids_json: str) -> Captured:
    """The handoff rows of the tasks ``ids_json`` names, as stored, in ONE step (decoded after release)."""
    return capture(txn.conn, "SELECT rowid AS seq, task_id, kind, op_id, facts FROM task_handoffs WHERE task_id IN "
                   "(SELECT value FROM json_each(?))", (ids_json,), ("seq", "task_id", "kind", "op_id", "facts"))


def _handoff_records(captured: Captured) -> List[Dict[str, Any]]:
    """``Txn.handoff_records``' rows from a capture: each task's rows in row order."""
    return captured.rows(key=lambda record: record["seq"])


def _documents(pages: List[Tuple[Captured, Captured]]) -> List[Tuple[str, Dict[str, Any]]]:
    """``(task id, document)`` of ``_fetch``'s pages in task-id order: body, handoff rows, card columns."""
    documents = []
    for rows, handoff_rows in pages:
        records = rows.rows(key=lambda record: record["task_id"])
        handoffs = store.decode_handoffs([record["task_id"] for record in records], _handoff_records(handoff_rows))
        documents.extend((str(record["task_id"]), store_task_cards.project(store.project_handoffs(
            json.loads(record["body"]), handoffs[record["task_id"]]), record)) for record in records)
    return documents


def _decode(pages: List[Tuple[Captured, Captured]]) -> List[Dict[str, Any]]:
    """The documents of ``_fetch``'s pages, in task-id order."""
    return [document for _task_id, document in _documents(pages)]


class Reprepare(Exception):
    """A mutator's answer when the write needs an input it may only prepare OUTSIDE the lock
    (a mailbox read, an event-chain snapshot, an artifact write): the transaction rolls back
    empty and ``prepared_write`` repeats the whole write with that input prepared first."""

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


def prepared_write(write: Callable[[set], Any]) -> Any:
    """Run ``write(needs)`` until its mutator stops asking: every ``Reprepare`` adds its name to
    ``needs`` and the next round prepares it before ``BEGIN``. No I/O happens under the lock."""
    needs: set = set()
    for _round in range(4):
        try:
            return write(needs)
        except Reprepare as exc:
            needs.add(exc.what)
    return write(needs)


DELETE_ROW = object()  # a handoff mutator's answer: drop the row


def patch_handoff(root: Any, task_id: str, kind: str, op_id: str,
                  mutator: Callable[[Optional[Dict[str, Any]]], Any]) -> Optional[Dict[str, Any]]:
    """One operation row of a task's projection ``kind`` (a ``store.ROW_BACKED_FIELDS`` name) in
    its own write transaction, no body write: the mutator sees the current row (``None`` when
    absent) and returns the new row, ``None`` to leave the store untouched, or ``DELETE_ROW``.
    ``LookupError`` when the task has no lifecycle row: a projection needs its owner. Returns the
    row that is current when the transaction ends."""
    task_id, op_id = str(task_id), str(op_id)
    with open_store(root, write=True) as txn:
        if not txn.row(task_id, "status"):
            raise LookupError(f"task {task_id} has no lifecycle row")
        current = txn.handoff(task_id, op_id)
        updated = mutator(json.loads(json.dumps(current)) if current is not None else None)
        if updated is None:
            return current
        if updated is DELETE_ROW:
            txn.delete_handoff(task_id, op_id)
        else:
            txn.put_handoff(task_id, kind, op_id, updated)
        txn.touch(task_id)
        return None if updated is DELETE_ROW else updated


def put_handoffs(root: Any, task_id: str, kind: str, rows: Dict[str, Any], *, only_absent: bool = False) -> None:
    """Upsert several operation rows of one projection in one transaction (``LookupError`` as above).
    ``only_absent`` keeps a row already recorded, decided inside the same write."""
    task_id = str(task_id)
    with open_store(root, write=True) as txn:
        if not txn.row(task_id, "status"):
            raise LookupError(f"task {task_id} has no lifecycle row")
        for op_id, facts in rows.items():
            if not (only_absent and txn.handoff(task_id, str(op_id)) is not None):
                txn.put_handoff(task_id, kind, str(op_id), facts)
        txn.touch(task_id)


def list_documents(root: Any, *, statuses: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Every document, or those whose stored status is one of ``statuses`` (the WHERE prefilter;
    a caller filtering on the EFFECTIVE status still filters the overlaid rows itself)."""
    wanted = None if statuses is None else sorted({str(status) for status in statuses})
    if wanted is not None and not wanted:
        return []
    with open_store(root, write=False) as txn:
        fetched = _fetch(txn) if wanted is None else _fetch(txn, f"t.status IN ({','.join('?' * len(wanted))})", wanted)
    return _decode(fetched)


def filtered_list_candidates(root: Any, statuses: Iterable[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    """Full documents only for the stable page and rows needing an effective overlay.

    A stable row's status/ts cannot change in that overlay (``task_status`` owns
    the writer's predicate). Its first N matches therefore suffice for a global
    top N, even when every uncertain row sorts ahead of them. Those other rows
    still run through the complete effective reader before filtering/limiting.
    Read selection and bodies in one transaction; preserve task-id tie order.
    Old schemas/NULLs fall back per row, without GET writes. Maintenance's
    existing hot_facts job derives legacy NULLs under the body's write_seq CAS.
    """
    from ouroboros.store_hot_facts import LIST_VERSION

    wanted = sorted(set(statuses))
    if not wanted:
        return []
    with open_store(root, write=False) as txn:
        if int(txn.meta("schema_version") or 1) < LIST_VERSION:
            fetched = _fetch(txn)
        else:
            bound = "" if limit is None else " LIMIT ?"
            params = (*wanted,) if limit is None else (*wanted, int(limit))
            selected = ("SELECT task_id FROM tasks WHERE list_stable=1 "
                        f"AND status IN ({','.join('?' * len(wanted))}) "
                        f"ORDER BY ts DESC, task_id ASC{bound}")
            fetched = _fetch(txn, f"t.list_stable IS NULL OR t.list_stable=0 OR t.task_id IN ({selected})", params)
    return _decode(fetched)


def lineage_where(parent: str, root_id: str, *, direct_only: bool,
                  queue_members: Iterable[str] = ()) -> Tuple[str, Tuple[str, ...]]:
    """The ``find_child_tasks`` prefilter as SQL, starting from this parent's children. Anchors: the
    rows naming PARENT as their parent or (subtree scope) ROOT_ID as their root, any role, and the
    QUEUE_MEMBERS (queued items naming either, whose parent or role may live only in the queue),
    bound as ONE JSON array: the statement's variables never grow with the queue.
    Their lineage closes over ``superseded_by``/``retry_task_id`` in both directions to a fixed
    point. Selected from that closure: the rows the projection can fold onto a child (a retry or
    supersession pointer, or a ``subagent`` row naming the parent/root). The typed pointers are the
    body's own (``store.typed_columns``), so a predecessor whose chain ends on a child is reached
    however long the chain, and unrelated retry history is never read."""
    kin = [(column, value) for column, value in (("parent_task_id", parent),
                                                ("root_task_id", "" if direct_only else root_id)) if value]
    if not kin:
        return "0", ()
    values = tuple(value for _column, value in kin)
    members = sorted({str(member) for member in queue_members if str(member or "").strip()})
    seeds = (json.dumps(members),) if members else ()
    anchors = " OR ".join([f"{column}=?" for column, _value in kin]
                          + ["task_id IN (SELECT value FROM json_each(?))"] * len(seeds))
    reach = (f"WITH RECURSIVE reach(id, successor, retry) AS (SELECT task_id, superseded_by, retry_task_id "
             f"FROM tasks WHERE {anchors} UNION SELECT tasks.task_id, tasks.superseded_by, tasks.retry_task_id "
             "FROM reach JOIN tasks ON tasks.superseded_by=reach.id OR tasks.retry_task_id=reach.id "
             "OR tasks.task_id=reach.successor OR tasks.task_id=reach.retry) SELECT id FROM reach")
    child = " OR ".join(f"t.{column}=?" for column, _value in kin)
    return (f"t.task_id IN ({reach}) AND (t.superseded_by IS NOT NULL OR t.retry_task_id IS NOT NULL "
            f"OR (t.delegation_role='subagent' AND ({child})))", (*values, *seeds, *values))


def select_lineage(root: Any, parent: str, root_id: str, *, direct_only: bool = False,
                   queue_members: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """The child-lookup candidates in ``task_id`` order (``lineage_where``); the Python projection decides the rest."""
    where, params = lineage_where(str(parent or ""), str(root_id or ""), direct_only=direct_only,
                                  queue_members=queue_members)
    with open_store(root, write=False) as txn:
        fetched = _fetch(txn, where, params)
    return _decode(fetched)


def task_facts(root: Any, *, where: str = "", params: Iterable[Any] = ()) -> Dict[str, Dict[str, Any]]:
    """Typed facts of every task (keyed by task id) without reading a body: the keys
    ``FACT_KEYS`` as strings (``""`` for no fact), ``presence_authority_recorded`` as a bool."""
    clause = f"WHERE {where}" if where else ""
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, f"SELECT {','.join(FACT_KEYS)} FROM tasks {clause}", params, FACT_KEYS)
    facts = {}
    for record in captured.rows(key=lambda record: record["task_id"]):
        row = {key: str(record[key] or "") for key in FACT_KEYS if key != "presence_authority_recorded"}
        row["presence_authority_recorded"] = bool(record["presence_authority_recorded"])
        facts[row["task_id"]] = row
    return facts


def newest_ids(root: Any, limit: Optional[int] = None, *, where: str = "", params: Iterable[Any] = ()) -> List[str]:
    """Task ids newest first by the stored ``ts`` (a row without one sorts oldest), ties by id."""
    clause = f"WHERE {where} " if where else ""
    bound = "" if limit is None else f" LIMIT {int(limit)}"
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, f"SELECT task_id, ts FROM tasks {clause}{_NEWEST_ORDER}{bound}", params,
                           ("task_id", "ts"))
    return [record["task_id"] for record in captured.rows(key=_newest("ts"), reverse=True)]


ROOT_RESULT_WHERE = store.ROOT_RESULT_WHERE


def newest_root_ids(root: Any, limit: Optional[int] = None, *, project_id: str = "") -> List[str]:
    """Root task ids (``ROOT_RESULT_WHERE``) newest first, walked in the root-only index of that
    order (``store.ROOT_WINDOWS``: one project's ``tasks_project_root_ts``, the whole
    store's ``tasks_root_ts``), so ``limit`` bounds the entries read: no child is visited, however
    many are newer than the roots or between them. Unpinned, the planner reads every root through
    ``tasks_parent`` and sorts them all."""
    if not project_id and limit is None:
        raise ValueError("the whole store's newest roots need a limit")
    index, scope, params = (("tasks_project_root_ts", "project_id=? AND ", (project_id,)) if project_id
                            else ("tasks_root_ts", "", ()))
    bound = "" if limit is None else f" LIMIT {int(limit)}"
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, f"SELECT task_id, ts FROM tasks INDEXED BY {index} "
                           f"WHERE {scope}{ROOT_RESULT_WHERE} {_NEWEST_ORDER}{bound}", params, ("task_id", "ts"))
    return [record["task_id"] for record in captured.rows(key=_newest("ts"), reverse=True)]


def project_result_counts(root: Any, project_id: str) -> Tuple[int, int]:
    """``(roots, children)`` among one project's results, counted without a body: its roots over
    its root index, all its rows over ``tasks_project`` alone (a child's table row is never read)."""
    with open_store(root, write=False) as txn:
        roots = txn.conn.execute(f"SELECT COUNT(*) FROM tasks INDEXED BY tasks_project_root_ts WHERE project_id=? "
                                 f"AND {ROOT_RESULT_WHERE}", (project_id,)).fetchone()[0]
        results = txn.conn.execute("SELECT COUNT(*) FROM tasks INDEXED BY tasks_project WHERE project_id=?",
                                   (project_id,)).fetchone()[0]
    return int(roots), int(results) - int(roots)


def documents_by_ids(root: Any, task_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = [str(task_id) for task_id in task_ids]
    if not ids:
        return {}
    with open_store(root, write=False) as txn:
        fetched = _fetch(txn, f"t.task_id IN ({','.join('?' * len(ids))})", ids)
    # Keyed by each row's stored id: a body that names another task never stands for that one.
    return dict(_documents(fetched))


# A display read (``task_reads``: history cards; ``effective_task_result`` with
# ``materialize_artifacts=False``, retry successors and child replicas included) takes every
# stored key except the bulky evidence no such reader uses — 94 MB of heads for 1.46 GB of
# bodies on a long-lived install. A DROPPED key goes; a KEPT key keeps only the named sub-keys
# it is read through; every other key, a new fact included, arrives as stored. The evidence
# goes to task detail, Continue and writers (cost planes, verification, plan/review state,
# traces, artifacts — whose materialized view is the ``materialize_artifacts=True`` read).
# A display reader that needs a dropped path names it here; ``tests/test_task_reads.py``
# builds a page from display views and from full documents and requires the same page.
DISPLAY_DROPPED: Tuple[str, ...] = (
    "trace_refs", "review_evidence", "artifacts", "result", "trace_summary",
    "governing_owner_words", "verification_ledger", "plan_review_state", "acceptance_debt",
    "mutation_evidence", "configured_subagent", "context", "continuation_narrative", "origin_message_text",
    "owner_mailbox", "subagent_availability", "plan_record", "review_status", "schedule_admission",
    "task_acceptance_review_accounting",
    *(f"metadata.{key}" for key in (
        "task_contract", "main_routing_manifest", "governing_owner_words", "routing_contract",
        "configured_subagent", "continuation", "project_routing_manifest", "subagent_envelope", "owner_corpus",
        "origin_message_text", "wake_observation", "project_last_task_result", "current_chat",
        "workspace_preflight", "workspace_copy", "late_answer")),
)
DISPLAY_KEPT: Dict[str, Tuple[str, ...]] = {
    "artifact_bundle": ("status",), "task_contract": ("deadline_at",),
    "loop_outcome": ("usage",), "loop_outcome.usage": ("total_rounds",),  # history's accepted solve responses
}
_KEEP = "__keep__"  # a path no kept object holds: the no-op target of a conditional edit


def _display_sql(column: str = "b.body") -> str:
    """One stored body's display view in SQLite JSON1 functions of the store's SQLite floor (no
    ``->``): presence, booleans and nulls are kept exactly, a non-object KEPT value as stored."""
    def path(*parts: str) -> str:
        return "'$." + ".".join(parts) + "'"

    view = (f"json_remove({column}, {', '.join(path(*key.split('.')) for key in DISPLAY_DROPPED)})"
            if DISPLAY_DROPPED else f"json({column})")
    for key, kept in DISPLAY_KEPT.items():
        is_object = f"json_type({column}, {path(key)}) = 'object'"
        values = [f"CASE json_type({column}, {path(key, sub)}) WHEN 'true' THEN json('true') WHEN 'false' "
                  f"THEN json('false') WHEN 'null' THEN json('null') ELSE json_extract({column}, {path(key, sub)}) END"
                  for sub in kept]
        built = "json_object(" + ", ".join(f"'{sub}', {value}" for sub, value in zip(kept, values)) + ")"
        absent = ", ".join(f"CASE WHEN json_type({column}, {path(key, sub)}) IS NULL THEN {path(sub)} "
                           f"ELSE {path(_KEEP)} END" for sub in kept)
        view = (f"json_replace({view}, CASE WHEN {is_object} THEN {path(key)} ELSE {path(key, _KEEP)} END, "
                f"CASE WHEN {is_object} THEN json_remove({built}, {absent}) END)")
    # The old-review presenter needs affirmative answer evidence even when this view
    # drops result. Project one bit from the same body; no extra read or stored field.
    present = f"json_type({column}, '$.result') = 'text' AND json_extract({column}, '$.result') <> ''"
    return f"json_set({view}, '$._result_text_present', json(CASE WHEN {present} THEN 'true' ELSE 'false' END))"


def _card_sql(view: str) -> str:
    """``view`` with the task's card row merged over it (``store_task_cards.project`` in SQL, one
    ``json_patch`` per carded row): the row's non-NULL facts replace the view's, in place or appended."""
    patch = ("json_remove(json_object(" + ", ".join(f"'{name}', c.{name}" for name in store_task_cards.FIELDS) + "), "
             + ", ".join(f"CASE WHEN c.{name} IS NULL THEN '$.{name}' ELSE '$.held' END" for name in store_task_cards.FIELDS) + ")")
    return f"CASE WHEN c.task_id IS NULL THEN {view} ELSE json_patch({view}, {patch}) END"


def display_documents(root: Any, task_ids: Iterable[str],
                      versions: Optional[Dict[str, int]] = None) -> Dict[str, str]:
    """The display views (``DISPLAY_DROPPED``) of several documents as JSON text, row-backed
    fields projected: ONE aggregated row in ONE short read, everything else after it closes.
    A page of tasks costs a few interpreter hand-offs and kilobytes, not a row step per task and
    the megabytes of evidence; each caller decodes its own private copy of a text. ``versions``
    receives each view's ``write_seq`` from the same read."""
    ids = sorted({str(task_id) for task_id in task_ids if str(task_id or "").strip()})
    if not ids:
        return {}
    with open_store(root, write=False) as txn:
        _columns, join = store_task_cards.select(txn)
        view = _card_sql(_display_sql()) if join else _display_sql()
        views = capture(txn.conn,  # '' || ...: the view travels as a string, decoded per use
                        f"SELECT t.task_id, '' || {view} AS view, t.write_seq FROM tasks AS t JOIN task_bodies AS b "
                        f"ON b.task_id = t.task_id{join} WHERE t.task_id IN (SELECT value FROM json_each(?))",
                        (json.dumps(ids),), ("task_id", "view", "write_seq"))
        handoff_capture = _handoff_capture(txn, json.dumps(ids))
    handoff_rows = _handoff_records(handoff_capture)
    handoffs = store.decode_handoffs(ids, handoff_rows)
    held = {record["task_id"] for record in handoff_rows}
    found = views.tuples()
    if versions is not None:
        versions.update((task_id, int(seq or 0)) for task_id, _text, seq in found)
    return {task_id: (text if task_id not in held else
                      store.document_text(store.project_handoffs(json.loads(text), handoffs[task_id])))
            for task_id, text, _seq in found}


def task_versions(root: Any, task_ids: Iterable[str]) -> Dict[str, Tuple[int, str, bool]]:
    """Each stored task's ``(write_seq, status, owes_synthesis)``, typed columns only, in ONE
    aggregated row: whether documents a window shows changed since it read them, and which of them
    are settled now. An id the store does not hold is absent from the answer."""
    ids = sorted({str(task_id) for task_id in task_ids if str(task_id or "").strip()})
    if not ids:
        return {}
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, "SELECT task_id, write_seq, status, owes_synthesis FROM tasks WHERE task_id IN "
                           "(SELECT value FROM json_each(?))", (json.dumps(ids),),
                           ("task_id", "write_seq", "status", "owes_synthesis"))
    return {task_id: (int(seq or 0), str(status or ""), bool(owes)) for task_id, seq, status, owes in captured.tuples()}


def task_drive_status(root: Any, task_ids: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """Each stored task's ``(status, its own drive)`` from typed columns in ONE read, no body
    decoded: the drive is its ``child_drive_root``, else ``headless_child_drive_root``, else
    ``drive_root`` (``""`` for none). An id the store does not hold is absent from the answer."""
    ids = sorted({str(task_id) for task_id in task_ids if str(task_id or "").strip()})
    if not ids:
        return {}
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, "SELECT task_id, status, COALESCE(child_drive_root, headless_child_drive_root, "
                           "drive_root) AS drive FROM tasks WHERE task_id IN (SELECT value FROM json_each(?))",
                           (json.dumps(ids),), ("task_id", "status", "drive"))
    return {str(task_id): (str(status or ""), str(drive or "")) for task_id, status, drive in captured.tuples()}


def task_ids(root: Any) -> List[str]:
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, "SELECT task_id FROM tasks", (), ("task_id",))
    return sorted(row[0] for row in captured.tuples())


class TreeCensus(NamedTuple):
    """One tree's membership read (``tree_facts``): ``members`` by task id, and ``unresolved``, each
    row whose own record could not be read, as ``{"reason", "linked"}``. A member reached by ancestry
    is in both: its membership is proven, its own record is still unknown."""
    members: Dict[str, dict]
    unresolved: Dict[str, dict]


_TREE_ROW = ("task_id", "status", "parent_task_id", "root_task_id", "retry_task_id", "superseded_by")
_TREE_SQL = ",".join(f"t.{name}" for name in _TREE_ROW)
_LINKS = ("parent_task_id", "retry_task_id", "superseded_by")


def _validated(tree_id: str, selected: Dict[str, dict], captured: list) -> Tuple[dict, dict, dict]:
    """``(members, rows, faults)`` once each CAPTURED body is validated as its own row's object:
    ``rows`` hold the root each captured row may be read with (``None``: unknown)."""
    from ouroboros.store_hot_facts import lineage_root, own_document

    members, rows, faults = dict(selected), {}, {}
    for record in captured:
        tid, row = record["task_id"], {name: record[name] for name in _TREE_ROW}
        document, fault = own_document(tid, record["body"])
        if fault:
            faults[tid], row["root_task_id"] = fault, None
            members.pop(tid, None)  # a flagged row's typed root may be another task's
        elif row["root_task_id"] is None:
            row["root_task_id"] = lineage_root(document)
            if row["root_task_id"] == tree_id:
                members[tid] = row
        rows[tid] = row
    return members, rows, faults


def _tied(tree_id: str, members: Dict[str, dict], rows: Dict[str, dict], faults: Dict[str, str],
          ties: set) -> Dict[str, dict]:
    """Each fault, ``linked`` when it is a member or typed edges to or from this tree or TIES tie it."""
    tied = {tid for tid in faults if tid == tree_id or tid in ties or tid in members}
    while True:
        known = {tree_id, *members, *tied}
        named = {row.get(link) for row in (*members.values(), *(rows[tid] for tid in tied)) for link in _LINKS}
        more = {tid for tid in faults.keys() - tied if tid in named or {rows[tid][link] for link in _LINKS} & known}
        if not more:
            return {tid: {"reason": fault, "linked": tid in tied} for tid, fault in sorted(faults.items())}
        tied |= more


_BODY_PAGE = ("SELECT b.task_id, CASE WHEN typeof(b.body)='blob' THEN NULL ELSE b.body END AS body, "
              "CASE WHEN typeof(b.body)='blob' THEN hex(b.body) END AS body_hex FROM task_bodies AS b "
              "WHERE b.task_id IN (SELECT value FROM json_each(?))")  # a damaged BLOB body travels as its bytes


def _with_bodies(records: List[dict], pages: List[Captured]) -> List[dict]:
    """Each census row with its stored body as the per-row read saw it: the text, a BLOB's own bytes, or ``None``
    for a row without one."""
    bodies = {}
    for page in pages:
        for task_id, text, hexed in page.tuples():
            bodies[task_id] = bytes.fromhex(hexed) if hexed is not None else text
    return [{**record, "body": bodies.get(record["task_id"])} for record in records]


def _keyed(captured: Captured) -> Dict[str, dict]:
    """A captured census selection by task id, in the aggregate's order. None of the census statements has an
    ORDER BY, so the per-row read's order was unspecified as well; that the two answer in the same order, with and
    without planner statistics, is pinned by tests/test_store_capture_readers.py, not promised by SQLite."""
    return {row["task_id"]: row for row in captured.rows()}


def tree_facts(root: Any, tree_id: str, *, descendants: bool = False,
               queue_rows: Iterable[dict] = (), linked: Iterable[str] = ()) -> TreeCensus:
    """Typed membership; a row whose root rests on its own record is validated alone.

    The indexed root column selects the tree. An underived (NULL) root, and this tree's
    flagged row (``control_valid=0``: a bad status, or an older job's columns from another
    task's body), counts once its body is its own object, rooted by ``store_hot_facts.lineage_root``.
    A missing, unparsable, non-object or other task's body leaves that ROW unresolved, never the
    census: ``linked`` when a typed parent/retry/supersession edge to or from this tree, a handoff
    naming it or a caller's ``linked`` id (queue, custody, model request) ties it here; otherwise
    a named gap, neither member nor proof of absence. A store failure raises. Each statement is
    captured in one step (``store_capture``), the bodies in pages of ids (``_pages``), and decoded after
    the read, unless descendant anchors need them within it (one snapshot either way).
    Descendant selection closes over the indexed parent chain, including bridges present only in
    the queue snapshot; a row reached so is a member by ancestry, and a damaged one stays named
    (linked) beside it. Durable rows win over queue rows. The caller still applies its own
    depth/ancestry rule to this selected superset.
    """
    from ouroboros.store_hot_facts import VERSION

    tree_id = str(tree_id)
    with open_store(root, write=False) as txn:
        flagged = int(txn.meta("schema_version") or VERSION) >= VERSION  # control_valid exists
        where = "t.root_task_id IS NULL" + (" OR (t.root_task_id=? AND t.control_valid=0)" if flagged else "")
        params = (tree_id,) if flagged else ()
        # The rows first, then their bodies by pages of ids: every underived body of an install still deriving
        # them, in one aggregate, could pass SQLite's string limit.
        records = capture(txn.conn, f"SELECT {_TREE_SQL} FROM tasks AS t WHERE {where}", params, _TREE_ROW).rows()
        ids = [record["task_id"] for record in records]
        bodies = [page for _ids, page in _pages(txn, ids, _BODY_PAGE, ("task_id", "body", "body_hex"))]
        selected = capture(txn.conn, f"SELECT {_TREE_SQL} FROM tasks AS t WHERE t.root_task_id=?", (tree_id,),
                           _TREE_ROW)
        handoffs = capture(txn.conn,  # a damaged handoff ties nothing
                           "SELECT task_id FROM task_handoffs WHERE task_id IN (SELECT value FROM json_each(?)) AND "
                           "CASE WHEN json_valid(facts) THEN json_extract(facts, '$.root_task_id') END=?",
                           (json.dumps(ids), tree_id), ("task_id",))
        if descendants:  # the closure's anchors are these rows: decoded inside this one snapshot
            ties = {*map(str, linked), *(row[0] for row in handoffs.tuples())}
            members, rows, faults = _validated(tree_id, _keyed(selected), _with_bodies(records, bodies))
            queued = {}
            for row in queue_rows:
                tid = str(row.get("task_id") or row.get("id") or "")
                if tid:
                    queued.setdefault(tid, row)
            # A queue record cannot replace a durable row, even when that row is
            # outside the selected tree. This addressed read preserves that rule.
            durable_queue = _keyed(capture(txn.conn, f"SELECT {_TREE_SQL} FROM tasks AS t WHERE t.task_id IN "
                                           "(SELECT value FROM json_each(?))", (json.dumps(list(queued)),), _TREE_ROW))
            queued.update({tid: rows.get(tid, row) for tid, row in durable_queue.items()})
            anchors = {tree_id, *members}
            while True:
                reach = capture(txn.conn,
                                "WITH RECURSIVE reach(id) AS (SELECT value FROM json_each(?) UNION "
                                "SELECT t.task_id FROM tasks AS t JOIN reach ON t.parent_task_id=reach.id) "
                                f"SELECT {_TREE_SQL} FROM tasks AS t "
                                "WHERE t.task_id IN (SELECT id FROM reach)", (json.dumps(sorted(anchors)),),
                                _TREE_ROW).rows()
                members.update({row["task_id"]: rows.get(row["task_id"], row) for row in reach})
                reached = {tree_id, *members}
                while True:
                    extra = {tid for tid, row in queued.items() if tid not in reached and (
                        str(row.get("root_task_id") or "") == tree_id or str(row.get("parent_task_id") or "") in reached)}
                    if not extra:
                        break
                    reached.update(extra)
                new = reached - anchors - members.keys()
                members.update({tid: row for tid, row in queued.items() if tid in reached and tid not in members})
                if not new:
                    return TreeCensus(members, _tied(tree_id, members, rows, faults, ties))
                anchors.update(reached)
    ties = {*map(str, linked), *(row[0] for row in handoffs.tuples())}
    members, rows, faults = _validated(tree_id, _keyed(selected), _with_bodies(records, bodies))
    return TreeCensus(members, _tied(tree_id, members, rows, faults, ties))


_COLLECTION_BATCH = 64  # bounds selection/materialization and one classification commit
DRIVE_RETENTION_BATCH = 64  # bounds directory-name selection and one classification commit


def drive_retention_facts(root: Any, task_ids: Iterable[str], *,
                          stop: Optional[Callable[[], bool]] = None, record: bool = True) -> Dict[str, Dict[str, Any]]:
    """Cheap facts for one bounded batch of drive names, never a scan of task history.

    A missing row/body or NULL projection stays a strict-read candidate. Legacy
    classification reads coherent bodies outside the selector transaction and
    commits only still-NULL write_seq matches, without changing bodies/signatures.
    An old body's REPLACE clears the nullable projection even after schema upgrade.
    ``record=False`` is the read-only reader (the free-space preview): the facts a
    legacy body yields fill the answer but no column, and a store behind the current
    schema answers nothing rather than upgrading (the next prune does both).
    """
    from ouroboros.retention import task_result_retention_at
    from ouroboros.task_results import load_task_result

    ids = [str(task_id) for task_id in task_ids]
    if not ids:
        return {}
    stopped = stop or (lambda: False)
    with open_store(root, write=False) as txn:
        schema = txn.meta("schema_version")
    if schema is not None and int(schema) < store.SCHEMA_VERSION:
        if not record:
            return {}
        from ouroboros.store_import import upgrade_current_schema

        upgrade_current_schema(root)
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, "SELECT t.task_id, t.status, t.delegation_role, t.drive_retention_at, "
                           "b.task_id IS NOT NULL AS body_present FROM tasks AS t "
                           "LEFT JOIN task_bodies AS b ON b.task_id=t.task_id "
                           f"WHERE t.task_id IN ({','.join('?' * len(ids))})", ids,
                           ("task_id", "status", "delegation_role", "drive_retention_at", "body_present"))
    facts = {row["task_id"]: row for row in captured.rows(key=lambda row: row["task_id"])}
    classified = []
    for task_id, row in facts.items():
        if stopped():
            break
        if row["drive_retention_at"] is not None or not row["body_present"]:
            continue
        try:
            document, stamp = load_task_result(root, task_id, strict=True, signed=True)
            if document is None or stamp is None:
                continue
            value = task_result_retention_at(document)
            if value is None:
                continue
            # This coherent body's facts improve this pass's hint too. A racing
            # writer may still postpone a skip; actual candidates are read afresh.
            row.update(status=document["status"], delegation_role=document.get("delegation_role"),
                       drive_retention_at=value)
            classified.append((value, task_id, stamp[0]))
        except (store.StoreError, ValueError, TypeError, AttributeError):
            # Leave NULL: the pruner's strict read retains the original diagnostic.
            continue
    if record and classified and not stopped():
        with open_store(root, write=True) as txn:
            txn.conn.executemany(
                "UPDATE tasks SET drive_retention_at=? WHERE task_id=? AND write_seq=? "
                "AND drive_retention_at IS NULL", classified)
    return facts


def _collection_ids(root: Any, column: str, pending: Optional[int], after: str) -> List[str]:
    where = "IS NULL" if pending is None else "=1"
    with open_store(root, write=False) as txn:
        captured = capture(txn.conn, f"SELECT task_id FROM tasks WHERE {column} {where} AND task_id>? "
                           "ORDER BY task_id LIMIT ?", (after, _COLLECTION_BATCH), ("task_id",))
    return sorted(row[0] for row in captured.tuples())


def _classify_collection(root: Any, column: str, predicate: Callable[[dict], bool],
                        task_ids: List[str], stop: Callable[[], bool]) -> List[str]:
    """Decode each legacy body after its coherent read; batch still-NULL sequence CAS updates.
    Classification changes no body/write_seq. A racing body writer or handoff wins."""
    classified, candidates = [], []
    for task_id in task_ids:
        if stop():
            break
        try:
            with open_store(root, write=False) as txn:
                stamp = txn.row(task_id, "write_seq", column)
                body = txn.body_text(task_id) if stamp is not None and stamp[1] is None else None
            if body is None:
                if stamp is not None and stamp[1] == 1:
                    candidates.append(task_id)
                continue
            pending = int(predicate(json.loads(body)))
            classified.append((pending, task_id, stamp[0]))
            if pending:
                candidates.append(task_id)
        except (store.StoreError, ValueError, TypeError, AttributeError):
            candidates.append(task_id)  # unreadable stays unknown; the collector retains its diagnostics
    if classified and not stop():
        with open_store(root, write=True) as txn:
            txn.conn.executemany(f"UPDATE tasks SET {column}=? WHERE task_id=? AND write_seq=? "
                                 f"AND {column} IS NULL", classified)
    return candidates


def _collection_candidates(root: Any, column: str, predicate: Callable[[dict], bool],
                           stop: Optional[Callable[[], bool]]) -> Iterable[str]:
    """Known work first, then bounded legacy classification through the existing maintenance owner.
    Column/predicate belong to the two callers below. No transaction spans a yield;
    every candidate's authority is re-read by its collector."""
    stopped = stop or (lambda: False)
    try:
        with open_store(root, write=False) as txn:
            schema = txn.meta("schema_version")
        if schema is not None and int(schema) < store.SCHEMA_VERSION:
            from ouroboros.store_import import upgrade_current_schema
            upgrade_current_schema(root)
        for pending in (1, None):
            after = ""
            while not stopped():
                identities = _collection_ids(root, column, pending, after)
                if not identities:
                    break
                after = identities[-1]
                yield from identities if pending == 1 else _classify_collection(root, column, predicate, identities, stopped)
    except store.StoreError:
        return  # failed acquisition/classification leaves the same candidates owed


def reclassify_review_collection(root: Any, task_id: str) -> bool:
    """Write back a ``review_collection_pending`` 1 the current predicate no longer selects (a row an older
    body flagged for its acceptance collection, which nothing rewrites now). Changes no body or write_seq;
    a racing body writer wins. True when the column was cleared."""
    from ouroboros.obligations import review_collection_pending

    with open_store(root, write=False) as txn:
        stamp = txn.row(str(task_id), "write_seq", "review_collection_pending")
        body = txn.body_text(str(task_id)) if stamp is not None and stamp[1] == 1 else None
    if body is None or review_collection_pending(json.loads(body)):
        return False
    with open_store(root, write=True) as txn:
        return txn.conn.execute("UPDATE tasks SET review_collection_pending=0 WHERE task_id=? AND write_seq=? "
                                "AND review_collection_pending=1", (str(task_id), stamp[0])).rowcount == 1


def review_collection_candidates(root: Any, *, stop: Optional[Callable[[], bool]] = None) -> Iterable[str]:
    """Tasks holding an open review operation pointer (``obligations.review_collection_pending``);
    unrelated plans stay out."""
    from ouroboros.obligations import review_collection_pending

    yield from _collection_candidates(root, "review_collection_pending", review_collection_pending, stop)


def plan_collection_candidates(root: Any, *, stop: Optional[Callable[[], bool]] = None) -> Iterable[str]:
    """Possible pending plan seats; their record and ended-task reader keep authority."""
    from ouroboros.plan_record import PLAN_RECORD_KEY, collection_pending

    yield from _collection_candidates(root, "plan_collection_pending",
                                     lambda row: collection_pending(row.get(PLAN_RECORD_KEY)), stop)


def inventory(root: Any) -> List[Dict[str, Any]]:
    """Every task's paging identity, newest write first: ``name`` (the task id), ``write_seq``
    (changes with every write, the role the file size and mtime played) and ``updated_at``."""
    with open_store(root, write=False) as txn:  # ordered: the scan reads the ``tasks_updated`` index, not the table
        captured = capture(txn.conn, "SELECT task_id, write_seq, updated_at FROM tasks ORDER BY updated_at DESC, "
                           "task_id DESC", (), ("task_id", "write_seq", "updated_at"))
    return [{"name": record["task_id"], "write_seq": int(record["write_seq"] or 0),
             "updated_at": str(record["updated_at"] or "")}
            for record in captured.rows(key=_newest("updated_at"), reverse=True)]
