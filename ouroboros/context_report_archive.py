"""Recent saved requests for the OS-local Beautiful Context viewer.

Browse bounded candidates through the existing room and usage indexes. A report
is fetched only by its selected ledger identity; there is no gateway or Send hook.
"""
from __future__ import annotations

from pathlib import Path

from ouroboros import dialogue_store, projects_registry, usage_store
from ouroboros.context_attempt_reader import _inspect, read_attempt, report_document
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

_ROOM_LIMIT = 100
_RECENT_ROWS = 128
_RECENT_TASKS = 32
_ATTEMPTS_PER_TASK = 64
_REPORT_LIMIT = 40


def rooms(root: Path) -> list[dict]:
    projects = projects_registry.list_sidebar_projects(root, require_readable=True)
    return [{"id": WEB_UI_CHAT_ID, "name": "Main"}] + [
        {"id": row["chat_id"], "name": row["name"]}
        for row in projects[:_ROOM_LIMIT] if row.get("chat_id") and row.get("name")
    ]


def recent_requests(root: Path, room_id: int) -> tuple[dict[str, str], list[dict]]:
    """A best-effort list, not a census of stored reports or a history import."""
    if type(room_id) is not int or not any(room['id'] == room_id for room in rooms(root)):
        raise ValueError('No such room in this local view')
    if not dialogue_store.path(root).is_file():
        raise FileNotFoundError('No dialogue store')
    from ouroboros.chat_chain import recent_outbound_task_ids

    tasks = recent_outbound_task_ids(root, room_id, row_limit=_RECENT_ROWS, task_limit=_RECENT_TASKS)
    if not tasks:
        return {}, []
    # The task index orders equal keys by rowid. Bound raw candidates before parsing
    # manifests or sorting by ledger order; newer records without bodies can hide older ones.
    sql = ("SELECT attempt_id, task_id, model, ts_reserved AS at, seq_first FROM "
           "(SELECT * FROM attempts INDEXED BY attempts_task WHERE task_id=? ORDER BY rowid DESC LIMIT ?) "
           "WHERE category='task' AND state IN ('dispatched','settled','unresolved') AND " + usage_store.ATTEMPT_ROW +
           " AND json_extract(extra, '$.candidate_manifest_ref.model_send_seal_version') IS NOT NULL")
    with _inspect(root) as txn:
        records = [row for task in tasks for row in txn.conn.execute(sql, (task, _ATTEMPTS_PER_TASK))]
    rows = sorted(records, key=lambda row: row['seq_first'], reverse=True)[:_REPORT_LIMIT]
    return {row['attempt_id']: row['task_id'] for row in rows}, [
        {key: row[key] for key in ('attempt_id', 'model', 'at')} for row in rows
    ]


def document(root: Path, selection: dict[str, str], attempt_id: str) -> str:
    if not isinstance(attempt_id, str) or attempt_id not in selection:
        raise ValueError('Attempt was not offered in this room')
    facts, raw = read_attempt(root, selection[attempt_id], attempt_id, report_facts=True)
    return report_document(facts, raw)[1].decode('utf-8')
