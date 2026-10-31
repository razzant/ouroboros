"""Collect changed plan candidates at zero spend; a pending seat remains the debt.

Task writes and durable producer publications queue the candidate's own signature.
Missed wakes coalesce per root/task in RAM until a dispatch check commits them, or
process exit. Boot rechecks the plan index for that crash window. Only contention
keeps an examination due; invalid sources await a task write, startup or repair.
"""
import logging
import pathlib
import threading

from ouroboros import maintenance_jobs, store
from ouroboros.utils import machine_json

JOB = "ended_plan_asks"
SOURCE = "plan_collection_source"  # obsolete install-wide publication marker
CHANGED = "plan_collection_changed"
_owed = {}
_owed_lock = threading.Lock()


def task_written(txn, task_id, columns):
    if not columns:
        return
    if columns.get("plan_collection_pending") != 0:
        txn.put_row(CHANGED, task_id, {"signature": [columns.get("write_seq"), columns.get("updated_at")]})
    else:
        txn.delete_row(CHANGED, task_id)


def producer_published(root, token):
    """Nonblocking publication after durability; dispatch replays any missed store write."""
    try:
        root = pathlib.Path(root).resolve()
        with _owed_lock:
            _owed.setdefault(root, {})[str(token[0])] = {"producer": token}
        _flush(root)
    except Exception:
        logging.getLogger(__name__).debug("Plan publication awaits dispatch or startup", exc_info=True)
    try:
        from ouroboros.maintenance import child
        child.request()
    except Exception:
        logging.getLogger(__name__).debug("Plan dispatch awaits reconcile", exc_info=True)


def _flush(root):
    """Transfer this root's owed wakes; retire only the exact snapshots committed."""
    root = pathlib.Path(root).resolve()
    with _owed_lock:
        if not _owed.get(root):
            return
    with store.acquisition_budget(0), store.open_store(root, write=True) as txn:
        # Snapshot under the store's write gate: an older flush cannot overwrite a newer one.
        with _owed_lock:
            owed = dict(_owed.get(root, {}))
        for task_id, facts in owed.items():
            if txn.row(task_id, "plan_collection_pending") != 0:
                txn.put_row(CHANGED, task_id, facts)
    with _owed_lock:
        current = _owed.get(root, {})
        for task_id, facts in owed.items():
            if current.get(task_id) is facts:
                del current[task_id]
        if not current:
            _owed.pop(root, None)


def recover_startup(txn):
    """One indexed existence check covers death between a retained reply and its wake."""
    if txn.conn.execute("SELECT 1 FROM tasks WHERE plan_collection_pending IS NULL "
                        "OR plan_collection_pending=1 LIMIT 1").fetchone():
        maintenance_jobs.record_in(txn, JOB, state="pending")


def due(root, row, facts, child):
    _flush(root)  # before spawning: the child reads durable tokens, never the server's RAM
    if row.get("state") == "pending" or not row.get("addressed"):
        return True
    with store.open_store(root, write=False) as txn:
        return txn.conn.execute("SELECT 1 FROM obligations WHERE set_name=? LIMIT 1", (CHANGED,)).fetchone() is not None


def run(root, should_stop):
    from ouroboros.late_review_marks import _collect_ended_plan_asks
    from ouroboros.task_store import plan_collection_candidates

    row = maintenance_jobs.job(root, JOB) or {}
    complete = False
    examined = set()

    def selected():
        nonlocal complete
        if row.get("state") == "pending" or not row.get("addressed"):
            from itertools import chain
            with store.open_store(root, write=False) as txn:
                queued = [item[0] for item in txn.conn.execute(
                    "SELECT id FROM obligations WHERE set_name=? ORDER BY id", (CHANGED,))]
            identities = chain(plan_collection_candidates(root, stop=should_stop), queued)
        else:
            with store.open_store(root, write=False) as txn:
                identities = [item[0] for item in txn.conn.execute(
                    "SELECT id FROM obligations WHERE set_name=? ORDER BY id", (CHANGED,))]
        seen = set()
        for task_id in identities:
            if task_id in seen:
                continue
            seen.add(task_id)
            with store.open_store(root, write=False) as txn:
                observed = txn.get_row(CHANGED, task_id)
            yield task_id
            if task_id in examined and observed is not None:
                with store.open_store(root, write=True) as txn:
                    txn.conn.execute("DELETE FROM obligations WHERE set_name=? AND id=? AND facts=?",
                                     (CHANGED, task_id, machine_json(observed)))
        complete = seen <= examined

    settled = _collect_ended_plan_asks(root, should_stop, candidates=selected(), on_examined=examined.add)
    state = "done" if complete and not should_stop() else "pending"
    with store.open_store(root, write=True) as txn:
        maintenance_jobs.forget_in(txn, JOB, "inputs")
        maintenance_jobs.record_in(txn, JOB, state=state, addressed=True)
        txn.conn.execute("DELETE FROM meta WHERE key=?", (SOURCE,))
    return {"state": state, "items": settled}
