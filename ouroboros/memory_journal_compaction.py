"""Compatibility entry point for the retired destructive memory-journal sweep.

The startup/maintenance caller still invokes this function. Retaining that call
keeps old integrations working while new knowledge, identity and Pattern Register
history stays complete. A previously digested row cannot be reconstructed;
no new row loses its old/new text merely because of its age.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import stat


_JOURNALS = ("memory/identity_journal.jsonl", "memory/knowledge_history.jsonl",
             "memory/knowledge/patterns_history.jsonl")


def compact_memory_journal_snapshots(
    drive_root: Any,
    retention_days: Optional[int] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Preserve every journal byte, including malformed and historical rows.

    The arguments keep the previous call contract. Size facts in the existing
    startup report measure growth; a missing journal is not a measured zero.
    """
    sizes: Dict[str, Optional[int]] = {}
    errors: list[str] = []
    for relative in _JOURNALS:
        try:
            info = (Path(drive_root) / relative).lstat()
            sizes[relative] = info.st_size if stat.S_ISREG(info.st_mode) else None
            if sizes[relative] is None:
                errors.append(f"{relative}: not_regular")
        except FileNotFoundError:
            sizes[relative] = None
        except OSError as exc:
            sizes[relative] = None
            errors.append(f"{relative}: {type(exc).__name__}")
    return {"digested": {}, "digest_mismatch": {}, "errors": errors,
            "journal_bytes": sizes}


__all__ = ["compact_memory_journal_snapshots"]
