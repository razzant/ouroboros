"""Cooperative repository checkpoints taken when a task tree goes quiescent.

A checkpoint runs off the supervisor loop under a per-root in-flight latch. A
trigger that arrives while a run is in flight is remembered and replayed once
the latch clears, because the in-flight run may already have sampled the last
child as live and skipped the commit.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional
from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.task_results import load_task_result
from supervisor.events_subagent_admission import _active_subagent_count

log = logging.getLogger(__name__)


# In-flight latch for off-loop coop checkpoints: one commit run per root at a
# time. A re-trigger after completion is safe (the helper no-ops on a clean
# tree), so this is concurrency control, not a permanent phase marker. A
# trigger arriving WHILE a run is in flight cannot simply be dropped: the
# in-flight worker may have already sampled liveness and seen the (then-live)
# last child, so it will skip the commit — and the dropped trigger was the
# last one there is. Such triggers are remembered per root and replayed once
# after the latch clears; the replayed run revalidates liveness itself.


_COOP_CHECKPOINT_INFLIGHT: set = set()


_COOP_CHECKPOINT_DROPPED: Dict[str, Dict[str, str]] = {}


_COOP_CHECKPOINT_LOCK = threading.Lock()


def _spawn_coop_checkpoint(
    ctx: Any, root_tid: str, *, title: str, trigger: str,
) -> Optional[threading.Thread]:
    """Run the coop checkpoint-commit OFF the event-drain thread.

    ``checkpoint_commit_coop_roots`` is a chain of bounded (60s each) git
    subprocesses — inline in the drain loop it is the WS3 starvation class, so
    the handlers only DETECT and enqueue. The bounded daemon thread
    RE-VALIDATES quiescence right before the git mutation (a racing tree
    member admitted between detect and run must win), reuses the v6.58.0
    helper verbatim (projects-root-only boundary, sensitive-file unstage,
    fail-soft per root), and appends loud receipts — including a loud-fail
    receipt on an unexpected error, because a silent skip here is exactly the
    uncommitted-pile class this call closes. Returns the thread (tests join
    it); a trigger arriving while one run is in flight is remembered and
    replayed once after that run completes (see the latch comment above —
    dropping it outright loses the tree's LAST quiescence trigger when the
    in-flight worker sampled the finishing child as still live)."""
    root_tid = str(root_tid or "").strip()
    if not root_tid:
        return None
    with _COOP_CHECKPOINT_LOCK:
        if root_tid in _COOP_CHECKPOINT_INFLIGHT:
            _COOP_CHECKPOINT_DROPPED[root_tid] = {"title": title, "trigger": trigger}
            return None
        _COOP_CHECKPOINT_INFLIGHT.add(root_tid)

    def _run() -> None:
        try:
            from ouroboros.coop_checkpoint import checkpoint_commit_coop_roots
            from supervisor.queue import _queue_lock

            # OFF the drain thread, PENDING/RUNNING are live containers other
            # threads mutate (the drain's own pop, queue admission, the worker
            # reaper). Iterating them unlocked raises "dictionary changed size
            # during iteration", which the except below would turn into a
            # loud-fail receipt and NO commit — killing the last trigger and
            # leaving the pile uncommitted, the exact defect this closes. The
            # re-validation is O(live tasks) with no I/O, so it takes the lock;
            # the git chain stays outside it.
            with _queue_lock:
                live = _active_subagent_count(root_tid, list(ctx.PENDING), dict(ctx.RUNNING)) > 0
            receipts = checkpoint_commit_coop_roots(
                ctx.DRIVE_ROOT, root_tid, title=title, has_live_tree_tasks=live,
            )
            for receipt in receipts:
                if receipt.get("committed") or receipt.get("error") or receipt.get("skipped_sensitive"):
                    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
                        "ts": utc_now_iso(), "type": "coop_checkpoint_commit",
                        "task_id": root_tid, "trigger": trigger, **receipt,
                    })
        except Exception as exc:
            try:
                append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "coop_checkpoint_commit",
                    "task_id": root_tid, "trigger": trigger,
                    "committed": False, "error": f"{type(exc).__name__}: {exc}",
                })
            except Exception:
                log.warning("coop checkpoint receipt write failed for %s", root_tid, exc_info=True)
        finally:
            with _COOP_CHECKPOINT_LOCK:
                _COOP_CHECKPOINT_INFLIGHT.discard(root_tid)
                dropped = _COOP_CHECKPOINT_DROPPED.pop(root_tid, None)
            if dropped is not None:
                # Replay the trigger that hit the latch mid-flight: this run may
                # have sampled the finishing child as live and skipped the
                # commit, and that trigger was the tree's last. The replayed
                # run re-validates liveness, so a spurious replay no-ops; a
                # replay happens only when a real trigger was dropped, so the
                # chain terminates with the finite trigger events.
                _spawn_coop_checkpoint(
                    ctx, root_tid,
                    title=dropped["title"], trigger=dropped["trigger"],
                )

    thread = threading.Thread(
        target=_run, name=f"coop-checkpoint-{root_tid[:12]}", daemon=True,
    )
    thread.start()
    return thread


def _checkpoint_coop_roots_on_root_done(ctx: Any, task: Dict[str, Any], task_id: str) -> None:
    """v6.58.0 (2.4B): when the ROOT of a task tree finalizes, checkpoint-commit any
    dirty host-minted genesis/coop tree its children built in — durable history instead
    of an uncommitted pile. Only projects-root trees, never owner-attached folders;
    credential-shaped files excluded (disclosed); fail-soft per root. Never raises.
    v6.91: detection only — the git work runs off the event-drain thread, and a tree
    still holding live members is left to the quiescence trigger
    (``_maybe_checkpoint_coop_on_tree_quiescence``) instead of being skipped forever
    (a budget-dead root ALWAYS terminalizes before its children, which used to leave
    every such pile uncommitted)."""
    try:
        root_tid = str(task.get("root_task_id") or task.get("id") or task_id or "")
        if not root_tid:
            return
        if _active_subagent_count(root_tid, ctx.PENDING, ctx.RUNNING) > 0:
            return  # live members: the last child's terminal event re-triggers
        _spawn_coop_checkpoint(
            ctx, root_tid,
            title=str(task.get("title") or task.get("suggested_name") or ""),
            trigger="root_done",
        )
    except Exception:
        log.debug("coop checkpoint-commit failed for %s", task_id, exc_info=True)


def _maybe_checkpoint_coop_on_tree_quiescence(ctx: Any, task: Dict[str, Any], task_id: str) -> None:
    """v6.91: re-run the coop checkpoint when the LAST live subtree member
    terminalizes under an already-terminal root.

    A root-scope budget death always kills the root FIRST (children die 20-90s
    later on their own next dispatch), so the root-done checkpoint saw live
    tree tasks and never ran again — wave1's coop tree still held only its
    genesis commit two days later. Called AFTER ``_finish_task_done_dispatch``
    removed this terminal child from RUNNING (before that, the finishing child
    itself still counts live and "zero live" is never true). Detection only;
    the git work runs off-loop via ``_spawn_coop_checkpoint``. Never raises."""
    try:
        root_tid = str(task.get("root_task_id") or "").strip()
        if not root_tid or root_tid == str(task_id or ""):
            return
        if _active_subagent_count(root_tid, ctx.PENDING, ctx.RUNNING) > 0:
            return
        if root_tid in ctx.RUNNING:
            return
        for row in ctx.PENDING:
            if isinstance(row, dict) and str(row.get("id") or "") == root_tid:
                return
        from ouroboros.task_status import SETTLED_STATUSES

        root_result = load_task_result(ctx.DRIVE_ROOT, root_tid) or {}
        # Truly settled roots only: a cancel_requested root still has a
        # cancellation custody in flight — its own terminal event re-triggers.
        if str(root_result.get("status") or "").strip().lower() not in SETTLED_STATUSES:
            return
        _spawn_coop_checkpoint(
            ctx, root_tid,
            title=str(root_result.get("title") or ""),
            trigger="tree_quiescence",
        )
    except Exception:
        log.debug("coop quiescence checkpoint failed for %s", task_id, exc_info=True)
