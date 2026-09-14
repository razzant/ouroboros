"""Crash-safe CyberGym campaign/result-index locking and paired writes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any, TextIO

from devtools.benchmarks.common.result_index import append_result_index, read_result_index
from devtools.benchmarks.cybergym.cybergym_protocol import (
    CyberGymError,
    safe_task_id,
    safe_task_path,
)

_SUPERSEDEABLE_LIFECYCLES = frozenset({"executor_failed"})


def is_supersedeable_result(row: Mapping[str, Any]) -> bool:
    """Whether late terminal evidence may supersede this transport-only row."""

    return (
        str(row.get("status") or "") in {"infra_failed", "blocked"}
        and str(row.get("lifecycle") or "") in _SUPERSEDEABLE_LIFECYCLES
        and str(row.get("row_role") or "") != "late_delivery"
    )


def is_late_terminal_result(row: Mapping[str, Any]) -> bool:
    """Whether a row is terminal benchmark evidence, not another infra opinion."""

    status = str(row.get("status") or "")
    lifecycle = str(row.get("lifecycle") or "")
    return (
        (status == "completed" and lifecycle == "official_verified")
        or (
            status == "failed"
            and lifecycle == "final_poc_missing_after_fair_completion"
        )
    )


def effective_task_rows(
    rows: Iterator[Mapping[str, Any]] | list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project append-only history to the latest effective row per task."""

    effective: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_task = row.get("task_id", row.get("instance_id", ""))
        if raw_task:
            effective[safe_task_id(str(raw_task))] = dict(row)
    return effective


def acquire_campaign_execution_lock(
    run_root: pathlib.Path | str,
    *,
    blocking: bool = True,
) -> TextIO | None:
    """Acquire a host-local lock without creating the candidate root."""
    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    root_digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    handle = (pathlib.Path(tempfile.gettempdir()) / f"ouroboros-cybergym-{root_digest}.lock").open(
        "a+", encoding="utf-8",
    )
    try:
        import fcntl

        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError:
            handle.close()
            return None
    except ImportError:
        pass
    return handle


@contextlib.contextmanager
def campaign_execution_lock(
    run_root: pathlib.Path | str,
    *,
    blocking: bool = True,
) -> Iterator[bool]:
    """Exclude live dispatch and reconcile delivery for one campaign root."""
    handle = acquire_campaign_execution_lock(run_root, blocking=blocking)
    try:
        yield handle is not None
    finally:
        if handle is not None:
            handle.close()


def campaign_history_task_ids(
    run_root: pathlib.Path,
    ledger_events: list[dict[str, Any]],
) -> set[str]:
    """Return task ids with a claim or official result in this campaign."""
    tasks = {
        safe_task_id(str(event.get("task_id") or ""))
        for event in ledger_events
        if str(event.get("event", event.get("kind", "")) or "").lower()
        in {"claim", "reserve", "reserved"}
    }
    index_path = run_root / "result_index.jsonl"
    if not index_path.exists():
        return tasks
    try:
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise CyberGymError(f"result index line {line_number} is not an object")
            raw_task = value.get("task_id", value.get("instance_id", ""))
            if raw_task:
                tasks.add(safe_task_id(str(raw_task)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CyberGymError(f"cannot inspect existing result index: {index_path}") from exc
    return tasks


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
        supersedes = value.get("supersedes")
        expected_hash = (
            str(supersedes.get("row_sha256") or "")
            if isinstance(supersedes, Mapping)
            else ""
        )
        observed_line = json.dumps(task_rows[-1], ensure_ascii=False)
        observed_hash = hashlib.sha256(observed_line.encode("utf-8")).hexdigest()
        if (
            str(value.get("row_role") or "") == "late_delivery"
            and expected_hash
            and observed_hash == expected_hash
        ):
            append_result_index(task_root, value)
            return
        raise CyberGymError(f"conflicting task-local result row already recorded for {task}")
    if not task_rows:
        if str(value.get("row_role") or "") == "late_delivery":
            raise CyberGymError(
                f"task-local superseded history is unavailable for {task}"
            )
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


def append_cybergym_late_result(
    run_root: pathlib.Path | str,
    row: Mapping[str, Any],
    *,
    source: str,
    gateway_task_id: str,
    reconcile_pass: int,
) -> bool:
    """Append one auditable infra-to-terminal superseding row pair."""

    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    task = safe_task_id(str(row.get("task_id", row.get("instance_id", ""))))
    attempt = str(row.get("attempt_id") or "")
    if not is_late_terminal_result(row):
        raise CyberGymError("late result is not terminal benchmark evidence")

    def _latest(path: pathlib.Path) -> tuple[dict[str, Any], str] | None:
        index = path / "result_index.jsonl"
        try:
            lines = index.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise CyberGymError(f"result index is unreadable: {exc}") from exc
        latest: tuple[dict[str, Any], str] | None = None
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CyberGymError(f"result index is unreadable: {exc}") from exc
            if (
                isinstance(value, Mapping)
                and str(value.get("task_id", value.get("instance_id", ""))) == task
                and str(value.get("attempt_id") or "") == attempt
            ):
                latest = (dict(value), line)
        return latest

    lock_path = root / ".result_index.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                pass
            previous = _latest(root)
            if previous is None:
                raise CyberGymError(f"no result row exists to supersede for {task}")
            previous_row, previous_line = previous
            if previous_row == dict(row):
                _append_result_pair(root, previous_row)
                return False
            if str(previous_row.get("row_role") or "") == "late_delivery":
                task_root = safe_task_path(root, task)
                task_latest = _latest(task_root)
                if task_latest is not None and task_latest[0] == previous_row:
                    prior_terminal = {
                        key: item
                        for key, item in previous_row.items()
                        if key not in {"row_role", "supersedes", "late_delivery"}
                    }
                    if prior_terminal == dict(row):
                        return False
                    raise CyberGymError(
                        f"result row is not supersedeable for {task}"
                    )
                expected_hash = str(
                    (previous_row.get("supersedes") or {}).get("row_sha256") or ""
                )
                observed_hash = (
                    hashlib.sha256(task_latest[1].encode("utf-8")).hexdigest()
                    if task_latest is not None
                    else ""
                )
                if task_latest is not None and observed_hash == expected_hash:
                    append_result_index(task_root, previous_row)
                    return False
                if task_latest is None:
                    raise CyberGymError(
                        f"task-local superseded history is unavailable for {task}"
                    )
                raise CyberGymError(
                    f"task-local result history diverged during late-delivery repair for {task}"
                )
            if not is_supersedeable_result(previous_row):
                raise CyberGymError(f"result row is not supersedeable for {task}")
            value = {
                **dict(row),
                "row_role": "late_delivery",
                "supersedes": {
                    "row_sha256": hashlib.sha256(
                        previous_line.encode("utf-8")
                    ).hexdigest(),
                    "status": str(previous_row.get("status") or ""),
                    "lifecycle": str(previous_row.get("lifecycle") or ""),
                    "infra_reason": str(previous_row.get("infra_reason") or ""),
                    "ts_unix": previous_row.get("ts_unix"),
                },
                "late_delivery": {
                    "source": str(source),
                    "gateway_task_id": str(gateway_task_id),
                    "reconcile_pass": int(reconcile_pass),
                },
            }
            task_root = safe_task_path(root, task)
            task_latest = _latest(task_root)
            if task_latest is not None and task_latest[0] != previous_row:
                raise CyberGymError(
                    f"task-local result history diverged before late delivery for {task}"
                )
            if task_latest is None:
                append_result_index(task_root, previous_row)
            append_result_index(root, value)
            append_result_index(task_root, value)
            lock.flush()
            os.fsync(lock.fileno())
            return True
        finally:
            if locked:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
