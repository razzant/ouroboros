"""Addressed result reads shared by ownership checks before queue mutation.

A pass reads each body once, signed by the store in the same transaction. A reuse
re-checks that signature — UNCHANGED, CHANGED or BUSY — without waiting under the
queue RLock: a changed or unreadable row means unknown ownership, never permission
to mutate, and a busy store is neither a change nor proof of sameness. Locked
callers supply the reads they prepared before acquiring the lock.
"""
from __future__ import annotations

import pathlib
import threading

UNCHANGED, CHANGED, BUSY = "unchanged", "changed", "busy"
_SETTLEMENT_READS = threading.local()
# A hard memory bound on unconsumed preparations per thread (each holds parsed result bodies);
# a multi-occupant settlement prepares one per occupant, far below it. An evicted one is
# reported unknown by its locked follow-up, which retains the drive (never a wrong settle).
_PREPARED_READS_MAX = 64


class TaskOwnershipRead:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.rows = {}
        self.stamps = {}
        self.errors = {}

    def _stamp(self, task_id):
        from ouroboros.task_store import signature

        return signature(self.root, task_id)

    def load(self, task_id):
        from ouroboros.task_results import load_task_result
        from supervisor import queue

        if task_id in self.errors:
            raise self.errors[task_id]
        if task_id in self.rows:
            state = self.recheck((task_id,))
            if state == BUSY:  # unknown, not a change: the cause names the contention a sweep waits out
                from ouroboros.store import StoreBusy
                raise ValueError('ownership result busy during this pass') from StoreBusy(
                    'ownership re-check found the store busy', reason='contention')
            if state != UNCHANGED:
                raise ValueError(f'ownership result {state} during this pass')
            return self.rows[task_id]
        if queue._queue_lock._is_owned():
            raise RuntimeError('prepare durable ownership before taking the queue lock')
        try:
            row, stamp = load_task_result(self.root, task_id, strict=True, signed=True)
            self.rows[task_id] = row or {}
            if stamp is None:
                from ouroboros.owner_pause import read_fence
                read_fence(self.root, task_id)  # Preserve missing/quarantined fence authority.
        except Exception as exc:
            self.errors[task_id] = exc
            raise
        self.stamps[task_id] = stamp
        return self.rows[task_id]

    def prepared(self, task_id):
        """The row as read before the queue lock, with no re-check: a task without a retry
        pointer is addressed by its physical id, and its live worker rewriting the row
        between preparation and the lock changes nothing about who owns it."""
        if task_id in self.rows:
            return self.rows[task_id]
        return self.load(task_id)

    def recheck(self, task_ids):
        """UNCHANGED, CHANGED or BUSY for prepared rows. Under the queue lock the signature
        probe never waits (``acquisition_budget(0)``): a busy store is unknown."""
        from contextlib import nullcontext
        from ouroboros.store import StoreBusy, acquisition_budget
        from supervisor import queue

        budget = acquisition_budget(0) if queue._queue_lock._is_owned() else nullcontext()
        try:
            with budget:
                return UNCHANGED if all(self.stamps[tid] == self._stamp(tid) for tid in task_ids) else CHANGED
        except StoreBusy:
            return BUSY

    def unchanged(self, task_ids):
        """True only for a proven-unchanged set; busy or changed cannot authorize a mutation."""
        return self.recheck(task_ids) == UNCHANGED



def prepare_retry_chain(q, task_id, read):
    """Read each reciprocal-chain candidate once; the locked resolver validates it."""
    seen = set()
    while task_id and task_id not in seen:
        seen.add(task_id)
        row = read(task_id)
        successor = str(row.get('superseded_by') or '').strip()
        if not successor or successor != str(row.get('retry_task_id') or '').strip():
            break
        if len(seen) > max(0, int(getattr(q, 'QUEUE_MAX_RETRIES', 0) or 0)):
            break  # The resolver reports the existing retry-authority violation.
        task_id = successor


def settlement_reads(root, task_id, *, locked, prepared=None):
    """Carry only the pre-interlock probe into its locked mutation check.

    No locked parse or unlock/relock gap in a multi-occupant drive settlement.
    A locked caller without a preparation receives unknown liveness.
    """
    pending = getattr(_SETTLEMENT_READS, 'pending', None)
    if pending is None:
        pending = _SETTLEMENT_READS.pending = {}
    key = str(root), task_id
    if prepared is not None:
        pending.pop(key, None)
        pending[key] = prepared
        while len(pending) > _PREPARED_READS_MAX:  # memory bound: the oldest unconsumed preparation goes
            pending.pop(next(iter(pending)))
        return prepared
    if locked:
        return pending.pop(key, None)
    return TaskOwnershipRead(root)
