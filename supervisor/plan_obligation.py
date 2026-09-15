"""The Swarm planning obligation follows the work (owner decision 3=A).

A root admitted with ``force_plan`` owes a plan review. When such a root, its
obligation still unmet, promotes its work into a new root, the promote event
carries the obligation (``force_plan`` + ``force_plan_source`` + the promoter's
id) and the existing admission seam stamps it on the new root. This module is
the other half of that one transaction: releasing the promoter's own flag on
the live queue rows the supervisor owns, so the snapshot persisted right after
admission shows both facts at once. Without it Main paid a plan review for work
the Project root did unplanned (14.09). No guard, no refusal: the promoter is
told, and any work it keeps doing itself is unplanned by its own choice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def transfer_promoter_obligation(ctx: Any, evt: Dict[str, Any], new_task_id: str) -> Dict[str, Any]:
    """Release the promoter's ``force_plan`` on its live queue row; the receipt says
    where the obligation went. Empty when the event carries no transfer."""
    promoter = str(evt.get("force_plan_transferred_from") or "").strip()
    if evt.get("force_plan") is not True or not promoter:
        return {}
    from supervisor.queue import _queue_lock

    receipt = {
        "from": promoter, "to": str(new_task_id or ""),
        "source": str(evt.get("force_plan_source") or "operator").strip() or "operator",
        "transferred_at": utc_now_iso(), "released": False,
    }
    try:
        with _queue_lock:
            running = getattr(ctx, "RUNNING", None)
            rows = []
            meta = running.get(promoter) if isinstance(running, dict) else None
            if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
                rows.append(meta["task"])
            for row in list(getattr(ctx, "PENDING", None) or []):
                if isinstance(row, dict) and str(row.get("id") or "") == promoter:
                    rows.append(row)
            for row in rows:
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = row["metadata"] = {}
                metadata["force_plan"] = False
                metadata["force_plan_transferred_to"] = receipt["to"]
                receipt["released"] = True
    except Exception:
        log.warning("force_plan release failed for promoter %s", promoter, exc_info=True)
    return receipt
