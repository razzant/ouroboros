"""Terminal+age sweep for delegated-run recovery/supervision state (CPL4-C13).

``state/delegate_recovery/``, ``state/delegate_recovery_transactions/`` and
``state/delegate_supervision/`` accumulated one file per crashed/restarted
task forever — nothing ever unlinked them. This startup sweep, run beside
the existing custody sweep, removes only what is PROVEN dead and old:

- a recovery row whose own ``status`` is terminal (``vetoed``/``adopted``)
  past GC retention; live/resumable rows (``reserved``/``pre_adopted``) are
  never touched, whatever their age;
- a restart transaction not referenced by any surviving recovery row, past
  GC retention (``active.json`` never; and if ANY recovery row was
  unreadable, the transaction pass is skipped — references are unknowable);
- a supervision file whose task has a SETTLED durable result, past GC
  retention; no result or a non-terminal result keeps the file (fail-closed,
  the C18/C13 doctrine).

Fail-closed throughout: an unreadable custody event log skips the whole
sweep (the ``_prune_delegated_snapshots`` idiom), an unreadable row or a
refused unlink keeps the file and reports.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_TERMINAL_RECOVERY_STATUSES = frozenset({"vetoed", "adopted"})


def _entries(directory: pathlib.Path) -> List[pathlib.Path]:
    try:
        return sorted(p for p in directory.glob("*.json") if p.is_file())
    except OSError:
        return []


def _task_settled(root: pathlib.Path, task_id: str) -> bool:
    try:
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        result = load_task_result(root, task_id) or {}
        return str(result.get("status") or "") in SETTLED_STATUSES
    except Exception:
        return False


def sweep_settled_delegate_state(
    drive_root: Any,
    retention_days: Optional[int] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Remove terminal+aged delegate state files; report what moved and why not."""
    from ouroboros.delegate_custody import custody_log_unreadable
    from ouroboros.retention import age_cutoff, get_gc_retention_days
    from ouroboros.utils import read_json_dict

    report: Dict[str, Any] = {"removed": [], "errors": [], "skipped": ""}
    root = pathlib.Path(drive_root)
    if custody_log_unreadable(root):
        report["skipped"] = "custody_log_unreadable"
        return report
    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)

    def _aged(path: pathlib.Path) -> bool:
        try:
            return path.stat().st_mtime < cutoff
        except OSError:
            return False

    def _remove(path: pathlib.Path, family: str) -> None:
        try:
            path.unlink()
            report["removed"].append(f"{family}/{path.name}")
        except OSError:
            report["errors"].append({"entry": f"{family}/{path.name}", "error": "unlink_failed"})

    surviving_transactions: set[str] = set()
    references_unknowable = False
    for path in _entries(root / "state" / "delegate_recovery"):
        row = read_json_dict(path)
        if not isinstance(row, dict) or not row:
            # Fail-closed: never delete what cannot be read — and its
            # transaction reference is unknowable, so that pass is skipped.
            report["errors"].append({"entry": f"delegate_recovery/{path.name}",
                                     "error": "unreadable_row"})
            references_unknowable = True
            continue
        if str(row.get("status") or "") in _TERMINAL_RECOVERY_STATUSES and _aged(path):
            _remove(path, "delegate_recovery")
            continue
        transaction_id = str(row.get("restart_transaction_id") or "")
        if transaction_id:
            surviving_transactions.add(transaction_id)

    if references_unknowable:
        report["skipped"] = "transactions_kept_unreadable_recovery_row"
    else:
        for path in _entries(root / "state" / "delegate_recovery_transactions"):
            if path.name == "active.json" or path.stem in surviving_transactions:
                continue
            if _aged(path):
                _remove(path, "delegate_recovery_transactions")

    for path in _entries(root / "state" / "delegate_supervision"):
        if _task_settled(root, path.stem) and _aged(path):
            _remove(path, "delegate_supervision")
    return report


__all__ = ["sweep_settled_delegate_state"]
