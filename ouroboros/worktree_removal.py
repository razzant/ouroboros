"""Failure evidence for checkout removal, with retries per process or changed directory.

The registry keeps the first failure stamp and root identity. Only this process's
observed failures suppress another attempt: a deep lock release need not change
the root's stat, so a new process must try again. No timer or shared retry counter.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso

_failed_here: Dict[str, tuple] = {}


def _identity(row: Dict[str, Any]) -> Optional[List[int]]:
    try:
        seen = Path(row.get("path") or "").lstat()
    except FileNotFoundError:
        return []
    except OSError:
        return None
    return [seen.st_dev, seen.st_ino, seen.st_mtime_ns, seen.st_ctime_ns]


def unchanged(row: Dict[str, Any]) -> bool:
    if "removal_failed_identity" not in row:
        return False
    identity = _identity(row)
    return row["removal_failed_identity"] == identity and _failed_here.get(str(row.get("path") or "")) == (
        os.getpid(), identity)  # a fork must retry too, even if it inherited the map


def record_failure(row: Dict[str, Any]) -> Dict[str, Any]:
    """Remember an actual failed attempt; return its durable evidence for the registry owner."""
    identity = _identity(row)
    _failed_here[str(row.get("path") or "")] = (os.getpid(), identity)
    return {**row, "removal_failed": row.get("removal_failed") or utc_now_iso(),
            "removal_failed_identity": identity}


def retired(row: Dict[str, Any], data_root: Any) -> None:
    """A retired checkout no longer needs process-local retry evidence, and the bytecode its sources
    left under the data root's prefix is owed to retirement now (``maintenance_retirement``)."""
    from ouroboros.maintenance_retirement import artifact_changed

    _failed_here.pop(str(row.get("path") or ""), None)
    artifact_changed(data_root, immediate=True)
