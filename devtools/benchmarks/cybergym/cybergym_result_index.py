"""Crash-safe CyberGym campaign/result-index locking and paired writes."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
from collections.abc import Iterator, Mapping
from typing import Any

from devtools.benchmarks.common.result_index import append_result_index, read_result_index
from devtools.benchmarks.cybergym.cybergym_protocol import (
    CyberGymError,
    safe_task_id,
    safe_task_path,
)


@contextlib.contextmanager
def campaign_execution_lock(
    run_root: pathlib.Path | str,
    *,
    blocking: bool = True,
) -> Iterator[bool]:
    """Exclude live dispatch and reconcile delivery for one campaign root."""
    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".campaign_execution.lock").open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), operation)
            except BlockingIOError:
                yield False
                return
        except ImportError:
            pass
        yield True
    finally:
        handle.close()


def _append_result_pair(root: pathlib.Path, row: Mapping[str, Any]) -> None:
    """Idempotently append/repair the common and task-local result pair."""
    task = safe_task_id(str(row.get("task_id", row.get("instance_id", ""))))
    value = dict(row)
    attempt = str(value.get("attempt_id") or "")
    task_root = safe_task_path(root, task)

    def _matching_rows(path: pathlib.Path) -> list[dict[str, Any]]:
        try:
            rows = read_result_index(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CyberGymError(f"result index is unreadable: {exc}") from exc
        return [
            existing for existing in rows
            if str(existing.get("task_id", existing.get("instance_id", ""))) == task
            and str(existing.get("attempt_id") or "") == attempt
        ]

    run_rows = _matching_rows(root)
    if run_rows:
        if run_rows[-1] != value:
            raise CyberGymError(f"conflicting result row already recorded for {task}")
        value = run_rows[-1]
    else:
        append_result_index(root, value)
    task_rows = _matching_rows(task_root)
    if task_rows and task_rows[-1] != value:
        raise CyberGymError(f"conflicting task-local result row already recorded for {task}")
    if not task_rows:
        append_result_index(task_root, value)


def append_cybergym_result(run_root: pathlib.Path | str, row: Mapping[str, Any]) -> None:
    """Append one row to the common run index and its task-local index."""
    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    lock_path = root / ".result_index.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                pass
            _append_result_pair(root, row)
            lock.flush()
            os.fsync(lock.fileno())
        finally:
            if locked:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
