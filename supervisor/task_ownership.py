"""Addressed result reads shared by ownership checks before queue mutation.

A pass reads each body once. Reuse checks its atomic-replacement stamp; a changed
or unreadable row means unknown ownership, never permission to mutate. Locked
callers supply the reads they prepared before acquiring the queue RLock.
"""
from __future__ import annotations

import pathlib
import threading

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
        from ouroboros.task_results import task_result_path

        try:
            stat = task_result_path(self.root, task_id, create=False).stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        except FileNotFoundError:
            return None

    def load(self, task_id):
        from ouroboros.task_results import load_task_result
        from supervisor import queue

        if task_id in self.errors:
            raise self.errors[task_id]
        if task_id in self.rows:
            if not self.unchanged((task_id,)):
                raise ValueError('ownership result changed during this pass')
            return self.rows[task_id]
        if queue._queue_lock._is_owned():
            raise RuntimeError('prepare durable ownership before taking the queue lock')
        stamp = self._stamp(task_id)
        try:
            self.rows[task_id] = load_task_result(self.root, task_id, strict=True) or {}
            if stamp is None:
                from ouroboros.owner_pause import read_fence
                read_fence(self.root, task_id)  # Preserve missing/quarantined fence authority.
        except Exception as exc:
            self.errors[task_id] = exc
            raise
        self.stamps[task_id] = stamp
        if not self.unchanged((task_id,)):
            raise ValueError('ownership result changed while reading')
        return self.rows[task_id]

    def unchanged(self, task_ids):
        return all(self.stamps[tid] == self._stamp(tid) for tid in task_ids)


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
