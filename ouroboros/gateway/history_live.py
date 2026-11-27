"""The serving observations one admitted history assembly cannot read from its stores.
LIVE is request-scoped: a reader thread neither installs these in another thread's
registries nor changes its environment. None leaves existing in-server readers alone.
"""
import time
from contextvars import ContextVar
from pathlib import Path

LIVE = ContextVar("history_live_inputs", default=None)


def capture(root, selectors):
    from ouroboros.deployment_policy import managed_model_roles
    from ouroboros.gateway.history_paging import live_overlays
    from ouroboros.gateway.state import copy_activity_inputs
    from ouroboros.i18n_memory import current_language
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from ouroboros.presence_recovery import copy_queue_custody_inputs
    from ouroboros.presence_runner import snapshot_local_presence_tasks
    from ouroboros.review_operation import controller_identity, live_review_owner_ids
    from ouroboros.skill_lifecycle_queue import queue_snapshot
    from ouroboros.utils import utc_now_iso
    from supervisor.active_activity import get_direct_activity_registry

    # Addressable registry rows do not prove liveness for another drive (or an
    # entry without an actor). Keep the witness roots, not just task ids.
    direct = {}
    for entry in get_direct_activity_registry().actors():
        actor_root = getattr(getattr(entry.actor, "env", None), "drive_root", None)
        if actor_root is not None:
            direct[entry.activity_id] = str(Path(actor_root).resolve())
    with POST_TASK_SYNTHESIS_LOCK:
        synthesis = [[path, tid] for (path, tid) in POST_TASK_SYNTHESIS_INFLIGHT]
    custody = copy_queue_custody_inputs(root)
    census, lifecycle = live_overlays(selectors["cursor"], selectors["around"])
    activities = None
    if census:
        availability = {"complete": True}
        activities = copy_activity_inputs(root, owners_only=True, availability=availability, queue_nowait=True)
        if not availability["complete"]:
            raise RuntimeError("live owner census is incomplete")
    active = queue_snapshot().get("active") if lifecycle else None
    return {"activities": activities, "custody_root": str(root), "queue_custody": custody,
            "controller": controller_identity(), "review_operations": live_review_owner_ids(),
            "skill": active if isinstance(active, dict) and active.get("status") == "running" else None,
            "presence": snapshot_local_presence_tasks(), "direct": direct, "synthesis": synthesis,
            "epoch": time.time(), "observed_at": utc_now_iso(), "language": current_language(),
            "managed_roles": sorted(managed_model_roles())}
