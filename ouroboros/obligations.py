"""Addressed current obligations, maintained by the transitions that owe them.

Each set is one JSON object (id -> facts) in ``state/obligations``. All sets
share one interprocess lock and atomic replacement. Add before publishing work;
remove only after discharge. A crash can leave an extra candidate, never erase
work. Missing/corrupt sets are unavailable: only the explicit lifecycle import
may reconstruct them from history. New writers may initialize an absent set.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ouroboros import platform_layer as platform
from ouroboros.utils import atomic_write_json

BOOT_SETS = ("nonterminal", "synthesis", "terminal_projection", "delegated_runs",
             "promotions", "custody_open", "pending_drives", "upgrade_notices", "unknowns",
             "pause_notices", "pause_notice_receipts")
RECEIPT_SETS = ("upgrade_notices", "pause_notice_receipts")  # chat receipts: no result derives them
LOCK_TIMEOUT_SEC = 4.0  # The existing update_json_locked IO acquisition budget.


class ObligationsUnavailable(ValueError):
    """Current membership is unknown; do not silently substitute a history scan."""


def path(root: Any, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("an obligation set name must be a filename stem")
    return Path(root) / "state" / "obligations" / f"{name}.json"


@contextmanager
def locked(root: Any):
    lock = Path(root) / "state" / "obligations.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not platform.kernel_file_locks_enforced(lock):
        fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=LOCK_TIMEOUT_SEC,
                                                 stale_sec=90.0, owner_aware_stale=True)
        if fd is None:
            raise TimeoutError(f"obligations lock unavailable: {lock}")
        try:
            yield
        finally:
            platform.release_exclusive_file_lock(lock, fd)
        return
    # Keep the inode on the enforced tier; ownership releases on process death.
    # Never hold this lock across a result writer or callback.
    with lock.open("a+b") as handle:
        deadline = time.monotonic() + LOCK_TIMEOUT_SEC
        while True:
            try:
                platform.file_lock_exclusive_nb(handle.fileno())
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"obligations lock unavailable: {lock}") from None
                time.sleep(min(0.005, remaining))
        try:
            yield
        finally:
            platform.file_unlock(handle.fileno())


def _read(root: Any, name: str, *, missing_ok: bool = False) -> dict:
    target = path(root, name)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(not isinstance(row, dict) for row in value.values()):
            raise ValueError("expected id -> facts")
        return value
    except FileNotFoundError as exc:
        if missing_ok:
            return {}
        raise ObligationsUnavailable(f"missing obligation set: {target}") from exc
    except (OSError, ValueError) as exc:
        raise ObligationsUnavailable(f"unreadable obligation set: {target}") from exc


def _write(root: Any, name: str, rows: dict) -> None:
    atomic_write_json(path(root, name), rows)


def members(root: Any, name: str) -> dict[str, dict]:
    """Return an independent snapshot; no migration or historical fallback."""
    marker = Path(root) / "state" / "migrations.json"
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("obligations_available") is False:
            raise ObligationsUnavailable("current obligations import is incomplete; explicit rebuild required")
    except FileNotFoundError:
        pass  # Writers can create a new root before its first lifecycle import.
    except (OSError, ValueError, AttributeError) as exc:
        raise ObligationsUnavailable("current obligations migration state is unavailable") from exc
    with locked(root):
        return _read(root, name)


def add(root: Any, name: str, identity: str, facts: dict | None = None) -> None:
    """Publish (or update) an addressed obligation before its dependent effect."""
    with locked(root):
        rows = _read(root, name, missing_ok=True)
        entry = dict(facts or {})
        if rows.get(str(identity)) != entry:
            rows[str(identity)] = entry
            _write(root, name, rows)


def remove(root: Any, name: str, identity: str) -> None:
    """Retire after discharge, serialized with every other set mutation."""
    with locked(root):
        rows = _read(root, name, missing_ok=True)
        if str(identity) in rows:
            del rows[str(identity)]
            _write(root, name, rows)


def result_memberships(row: dict) -> set[str]:
    """The result-owning writers share these predicates, including direct patches."""
    names = set()
    if row.get("status") in {"running", "interrupted"}:
        names.add("nonterminal")
    checkpoint = row.get("root_phase_checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    if checkpoint.get("post_task_synthesis"):
        from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
        if post_task_synthesis_is_open(checkpoint["post_task_synthesis"]):
            names.add("synthesis")
    if (row.get("canonical_terminal_projection_origin") == "terminal_transition"
            or isinstance(row.get("canonical_terminal_projection_ready"), dict)):
        from ouroboros.terminal_projection import terminal_projection_owed
        if terminal_projection_owed(str(row.get("task_id") or ""), row):
            names.update(("synthesis", "terminal_projection"))
    if row.get("delegated_runs_unreconciled"):
        names.add("delegated_runs")
    if row.get("child_ref_promotion"):
        from ouroboros.observability import _has_pending_ref_promotion
        if _has_pending_ref_promotion(row["child_ref_promotion"]):
            names.add("promotions")
    if row.get("pause_notices"):
        names.add("pause_notices")
    return names


def result_facts(row: dict) -> dict:
    return {key: row[key] for key in ("task_id", "headless_child_drive_root", "child_drive_root", "drive_root")
            if row.get(key)}


def before_result(root: Any, row: dict, *, names: set[str]) -> None:
    """Called with the result lock held, before its atomic replace."""
    tid, facts = str(row["task_id"]), result_facts(row)
    owed = set(names)
    if "nonterminal" in owed:
        owed.add("pending_drives")
    with locked(root):
        for name in owed:
            rows = _read(root, name, missing_ok=True)
            # Preserve closing custody receipts until the backfill acknowledges them.
            entry = {**rows.get(tid, {}), **facts} if name == "delegated_runs" else facts
            if rows.get(tid) != entry:
                rows[tid] = entry
                _write(root, name, rows)


def after_result(root: Any, row: dict, *, names: set[str]) -> None:
    """Retire under the same result lock, so a later transition cannot be erased."""
    with locked(root):
        for name in names:
            rows = _read(root, name, missing_ok=True)
            facts = rows.get(str(row["task_id"]), {})
            if name == "delegated_runs" and facts.get("closed_runs"):
                continue
            if str(row["task_id"]) in rows:
                rows.pop(str(row["task_id"]), None)
                _write(root, name, rows)


REBUILD_MARK = "rebuild.owed"  # under state/obligations/: the next start re-imports every set


def owe_rebuild(root: Any, what: str, exc: BaseException) -> None:
    """A membership write failed (an unreadable set, a contended or failed write): the transition
    it rides on proceeds, and the next start's lifecycle import rebuilds the sets from results."""
    import logging

    logging.getLogger(__name__).warning("Obligation %s not recorded (%s); the next start rebuilds the sets",
                                        what, type(exc).__name__)
    try:
        mark = Path(root) / "state" / "obligations" / REBUILD_MARK
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(f"{what}: {type(exc).__name__}\n", encoding="utf-8")
    except OSError:
        logging.getLogger(__name__).critical("Obligations rebuild could not be owed under %s", root, exc_info=True)


def _bookkeeping(root: Any, what: str, fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except (ObligationsUnavailable, TimeoutError, OSError) as exc:
        owe_rebuild(root, what, exc)


def update_result(path: Path, mutator: Any, *, writer=None, **kwargs: Any) -> dict:
    """Use the existing result lock for membership-before/write/retirement-after. Membership is
    bookkeeping: a set that cannot be written never fails the result write (see owe_rebuild)."""
    from ouroboros.utils import update_json_locked

    root = Path(path).parent.parent
    retired = set()

    def publish(current):
        nonlocal retired
        previous, facts = result_memberships(current), result_facts(current)
        updated = mutator(current)
        if updated is not None:
            owed = result_memberships(updated)
            added = owed - previous if facts == result_facts(updated) else owed
            retired = previous - owed
            if added:
                _bookkeeping(root, "membership", before_result, root, updated, names=added)
        return updated

    return (writer or update_json_locked)(path, publish, after_write=lambda row: _bookkeeping(
        root, "retirement", after_result, root, row, names=retired) if retired else None, **kwargs)


def drive_started(root: Any, task: dict) -> None:
    """Pool handoff: publish before the child can write anything or die."""
    tid = str(task.get("id") or task.get("task_id") or "")
    if tid:
        _bookkeeping(root, "pending drive", add, root, "pending_drives", tid, {**result_facts(task), "task_id": tid})


def drive_finished(root: Any, task: dict, row: dict | None = None) -> None:
    """Retire only after the canonical terminal file receipt is complete."""
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    tid = str(task.get("id") or task.get("task_id") or "")
    try:
        row = row if row is not None else load_task_result(root, tid, strict=True)
    except (OSError, ValueError):
        return  # an unreadable result keeps the drive pending: recovery decides later
    if row and row.get("status") in SETTLED_STATUSES and terminal_task_files_ready(Path(root), task, row):
        _bookkeeping(root, "drive retirement", remove, root, "pending_drives", tid)


def result_rows(root: Any, name: str, *, exclude=()):
    """Read only addressed candidates; unreadable members remain repair gaps."""
    import logging
    from ouroboros.task_results import load_task_result
    for tid in members(root, name):
        if tid in exclude:
            continue
        try:
            row = load_task_result(root, tid, strict=True)
            if row:
                yield row
            else:
                add(root, "unknowns", f"task:{tid}", {"task_id": tid, "reason": "result_missing"})
        except (OSError, ValueError) as exc:
            add(root, "unknowns", f"task:{tid}", {"task_id": tid, "reason": type(exc).__name__})
            logging.getLogger(__name__).warning("Obligation %s/%s is unreadable", name, tid)
