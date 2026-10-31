"""Selected waits and actual returns, recorded at the wait owner's boundaries."""
from __future__ import annotations

import copy
import logging
import uuid

from ouroboros import store
from ouroboros.subagent_metrics import set_name
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def _write(root, task_id, wait_id, fields):
    if not root or not task_id or not wait_id:
        return
    try:
        with store.open_store(root, write=True) as txn:
            name = set_name("swarm_waits", task_id)
            prior = txn.get_row(name, wait_id) or {"task_id": task_id, "wait_id": wait_id}
            if prior.get("woke_at"):
                return  # Replay cannot move an occurrence to collection time.
            txn.put_row(name, wait_id, {**prior, **fields})
        from ouroboros.swarm_receipt import refresh_for_task
        refresh_for_task(root, task_id)
    except Exception:
        log.debug("wait measurement unavailable", exc_info=True)


def sleep_started(ctx, started):
    chosen = getattr(ctx, "_model_sleep", None) or (getattr(ctx, "owner_wait_resume", None) or {}).get("sleep") or {}
    ctx._swarm_sleep_selection = copy.deepcopy(chosen)
    _write(getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None),
           str(getattr(ctx, "task_id", "") or ""), chosen.get("sleep_id"),
           {"selection": copy.deepcopy(chosen), "started_at": started, "kind": "sleep"})


def sleep_woke(ctx, outcome):
    if not outcome:
        return
    chosen = (getattr(ctx, "_swarm_sleep_selection", None) or getattr(ctx, "_model_sleep", None)
              or (getattr(ctx, "owner_wait_resume", None) or {}).get("sleep") or {})
    _write(getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None),
           str(getattr(ctx, "task_id", "") or ""), chosen.get("sleep_id"),
           {"woke_at": utc_now_iso(), "woke_by": str(outcome)})
    ctx._swarm_sleep_selection = None


def review_started(binding):
    if binding:
        binding.pop("waking_actor_id", None)
        binding.pop("_swarm_wait_outcome", None)
        binding["_swarm_wait_id"] = uuid.uuid4().hex
        binding["_swarm_wait_started"] = utc_now_iso()


def review_woke(binding, outcome="whole_panel_returned"):
    if binding:
        _write(binding.get("budget_drive_root") or binding.get("drive_root"), str(binding.get("task_id") or ""),
               binding.get("_swarm_wait_id"), {"kind": "review", "selection": copy.deepcopy(binding.get("selection") or {}),
               "started_at": binding.get("_swarm_wait_started"), "woke_at": utc_now_iso(), "woke_by": outcome,
               "waking_actor_id": binding.get("waking_actor_id")
                   if outcome == "review_result" or outcome.startswith(("task:", "run:")) else None})
