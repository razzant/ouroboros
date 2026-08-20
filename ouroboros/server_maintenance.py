"""Upkeep a supervisor generation owes the drive.

The once-per-generation startup sweep (process custody, delegated runs, legacy
cancel latches, owed terminal deliveries, orphaned running results, pending
post-task synthesis), the throttled periodic cadences of the same surfaces, and
the delegated-snapshot GC that fails closed on an unreadable custody log.
"""

import pathlib
import time

from ouroboros.server_process import DATA_DIR, log
from ouroboros.utils import utc_now_iso


def _installed_skill_names():
    """Names of skills currently installed ON DISK (disk-derived, not in-memory).

    Passed to the process-custody reaper so it can tell which skill-companion
    orphans are safe to reap (owner uninstalled). Disk-derived so it is correct
    independent of in-memory extension-reload timing; returns None on any failure
    so the reaper fails toward KEEP (never mass-kills live skills' companions).
    """
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import discover_skills

        names = {s.name for s in discover_skills(DATA_DIR, repo_path=get_skills_repo_path())}
        # Coalesce an EMPTY result to None ("unknown"), NOT "everything
        # uninstalled": discover_skills returns [] without raising when the skills
        # dir is momentarily unavailable; treating that as an empty install set
        # would let an enforced reap mass-kill live companions. None ⇒ keep-all.
        return names or None
    except Exception:
        log.debug("Could not compute installed skill names for custody reaper", exc_info=True)
        return None


_LAST_CANCEL_INTENT_SWEEP = [0.0]


def _periodic_supervisor_maintenance(last_custody_reap: list, last_review_reconcile: list) -> None:
    """Throttled periodic upkeep extracted from the supervisor loop: cancel-intent
    watchdog (every 20s), custody reap of orphaned task-scoped processes (every
    600s) + review-job zombie reconcile (every 300s). Each cadence gates itself
    via its own last-run marker."""
    if time.time() - _LAST_CANCEL_INTENT_SWEEP[0] > 20:
        _LAST_CANCEL_INTENT_SWEEP[0] = time.time()
        try:
            # Phase A watchdog: re-feed open durable cancel intents into custody
            # (the ONE settle owner) so a lost control event can no longer wedge
            # a cancellation forever — the Poltergeist incident class.
            from supervisor.task_lifecycle import sweep_cancel_intents

            outcomes = sweep_cancel_intents()
            if outcomes:
                log.info("Cancel-intent watchdog settled: %s", outcomes)
        except Exception:
            log.debug("Cancel-intent watchdog sweep failed", exc_info=True)
        try:
            # Phase A2/F7: re-enqueue terminal answers registered as OWED whose
            # send never got confirmed (a crash between settle and send used to
            # lose the owner's answer forever — the incident class itself).
            from supervisor.terminal_delivery import replay_pending_deliveries

            replay_pending_deliveries(DATA_DIR)
        except Exception:
            log.debug("Pending terminal-delivery replay failed", exc_info=True)
    if time.time() - last_custody_reap[0] > 600:
        last_custody_reap[0] = time.time()
        try:
            from ouroboros.process_custody import reap_orphaned_processes
            from supervisor.queue import RUNNING as _running_tasks

            live_tasks = set(_running_tasks.keys())
            reap_orphaned_processes(
                DATA_DIR, running_task_ids=live_tasks,
                live_owner_skills=_installed_skill_names(),
            )
            # A delegated Claudexor run is an orphan under exactly the same predicate:
            # its owning task is no longer running. It has no pid, so the process
            # reaper cannot see it — but it is still spending quota and still writing.
            _reconcile_delegated_runs(live_tasks)
        except Exception:
            log.debug("Periodic custody reap failed", exc_info=True)
    if time.time() - last_review_reconcile[0] > 300:
        last_review_reconcile[0] = time.time()
        _periodic_zombie_reconcile()


def _reconcile_delegated_runs(running_task_ids: set) -> None:
    """Settle or cancel delegated runs whose owning task is gone (startup + tick)."""
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway
        from ouroboros.delegate_custody import reconcile_orphaned_runs

        # The tick runs on the supervisor loop thread: a daemon sitting in its
        # recovery-only admission window must not hold that thread for the default
        # admission wait — skip-until-next-sweep is this caller's normal posture.
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: ensure_owned_gateway(admission_wait_sec=0),
        )
        if outcomes:
            log.info("Delegated-run reconciliation handled %d orphan(s): %s", len(outcomes), outcomes)
    except Exception:
        log.debug("Delegated-run reconciliation failed", exc_info=True)


def _startup_worktree_prune() -> None:
    """Startup hygiene: prune orphaned subagent worktrees (after the custody sweep)."""
    from supervisor.state import append_jsonl

    try:
        from ouroboros import subagent_worktrees

        worktree_report = subagent_worktrees.prune_orphans()
        if worktree_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "subagent_worktree_prune",
                "report": worktree_report,
            })
    except Exception:
        log.debug("Subagent worktree prune failed", exc_info=True)


def _startup_prune_sweeps() -> None:
    """Startup hygiene: prune stale task drives/trees and orphaned temp files."""
    from supervisor.state import append_jsonl

    try:
        from ouroboros.headless import prune_headless_task_drives, prune_task_drives, prune_task_trees
        from ouroboros.utils import sweep_stale_temp_files

        prune_report = prune_headless_task_drives(DATA_DIR)
        task_drive_report = prune_task_drives(DATA_DIR)
        # Ephemeral task-tree coordination ledgers age out with their terminal root.
        prune_task_trees(DATA_DIR)
        # Reap orphaned atomic-write temp files (.*.tmp.*) left by a hard kill.
        sweep_stale_temp_files(DATA_DIR)
        if (
            prune_report.get("pruned")
            or prune_report.get("errors")
            or task_drive_report.get("pruned")
            or task_drive_report.get("errors")
        ):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "headless_task_drive_prune",
                "report": prune_report,
                "task_drives": task_drive_report,
            })
    except Exception:
        log.debug("Headless task drive prune failed", exc_info=True)


def _startup_custody_sweep() -> None:
    """Both custody surfaces, swept once per generation at supervisor startup.

    Nothing is running yet, so every ledgered process and every open delegated run is
    by definition ownerless: the generation that was watching them did not survive.
    """
    try:
        from ouroboros.process_custody import reap_orphaned_processes

        reaped = reap_orphaned_processes(DATA_DIR, live_owner_skills=_installed_skill_names())
        if reaped:
            log.info("Process custody reaper killed %d orphaned process(es): %s", len(reaped), reaped)
    except Exception:
        log.debug("Process custody startup reap failed", exc_info=True)
    _reconcile_delegated_runs(set())
    try:
        # Phase A boot migration: legacy ``cancel_requested`` status latches
        # become ordinary durable cancel intents; the supervisor watchdog then
        # drives each through custody to a real settled outcome.
        from ouroboros.cancel_intents import migrate_legacy_cancel_latches

        migrated = migrate_legacy_cancel_latches(DATA_DIR)
        if migrated:
            log.info("Migrated %d legacy cancel latch(es) to durable intents: %s",
                     len(migrated), migrated)
    except Exception:
        log.debug("Legacy cancel-latch migration failed", exc_info=True)
    try:
        # Boot half of the durable terminal outbox: an answer that was registered
        # as owed but whose send never completed (crash between settle and send)
        # is re-enqueued exactly once — the delivered registry suppresses a copy
        # that actually landed.
        from supervisor.terminal_delivery import replay_pending_deliveries

        replay_pending_deliveries(DATA_DIR)
    except Exception:
        log.debug("Boot replay of pending terminal deliveries failed", exc_info=True)


def _prune_delegated_snapshots() -> None:
    """C1 delegated execution snapshots: GC cross-checked against custody.

    A snapshot stays while its run is open/undisposed OR a pending invocation
    names it; everything else (disposed, closed, refused) is torn down with its
    pinned baseline ref. Fail-soft like every startup prune step — the guard
    lives here so the startup sequence never dies on a GC error.

    FAIL-CLOSED on an unreadable custody log (CR1-1): the keep-set comes from
    replaying the custody rows, and ``_iter_rows`` swallows its own OSError —
    right for the fail-soft readers, but here an unreadable log replays as
    "no open runs", the keep-set goes EMPTY, and the prune destroys every
    live snapshot with the child's only copy of its work. GC may delete only
    over PROVEN settled && patch_disposed; an UNKNOWN custody state skips the
    destructive prune entirely and says so loudly."""
    try:
        from ouroboros import delegate_custody as _delegate_custody
        from ouroboros import subagent_worktrees as _snap_worktrees
        from supervisor.state import append_jsonl

        if _delegate_custody.custody_log_unreadable(DATA_DIR):
            log.warning(
                "Delegated snapshot prune SKIPPED: custody event log exists but "
                "cannot be read, so open snapshots are unknowable (fail-closed)")
            if not append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune_skipped",
                "reason": "custody_log_unreadable",
            }):
                # CR2-2: the log is unwritable too — the promised durable row
                # could not land. Escalate loudly; the skip itself already
                # protects the open snapshots, so this stays fail-soft.
                log.error(
                    "Delegated snapshot prune skip could NOT be recorded durably: "
                    "the delegated_snapshot_prune_skipped row was not written "
                    "(custody event log unwritable). Open snapshots remain "
                    "protected by the skip itself.")
            return
        snapshot_report = _snap_worktrees.prune_execution_snapshots(
            _delegate_custody.open_snapshot_ids(DATA_DIR))
        if snapshot_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune",
                "report": snapshot_report,
            })
    except Exception:
        log.debug("Delegated execution snapshot prune failed", exc_info=True)


def _periodic_zombie_reconcile() -> None:
    """Heal zombie 'running' records on a supervisor cadence.

    A worker that died mid-review (crash / SIGKILL / manual stop) leaves
    ``review_job.json`` at status=running forever in headless/no-UI runs, where
    the boot and ``GET /api/extensions`` reconciles never fire; the same death
    leaves ``task_results/<id>.json`` at running. Both reconciles are
    liveness-gated (pid-dead / queue-empty + worker-boot evidence), so a live
    review or task is never touched.
    """
    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs
        reconcile_stale_review_jobs(DATA_DIR)
    except Exception:
        log.debug("Periodic skill review-job reconcile failed", exc_info=True)
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks
        reconcile_orphaned_running_tasks(DATA_DIR)
    except Exception:
        log.debug("Periodic orphaned running-task reconcile failed", exc_info=True)
    try:
        from ouroboros.projects_registry import reconcile_projects
        reconcile_projects(DATA_DIR)
    except Exception:
        log.debug("Project registry reconcile failed", exc_info=True)
    _resume_interrupted_project_deletions()


def _resume_interrupted_project_deletions() -> None:
    try:
        from supervisor.task_lifecycle import resume_project_deletions

        resume_project_deletions(DATA_DIR)
    except Exception:
        log.debug("Project deletion recovery failed", exc_info=True)


def _run_startup_task_recovery(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    *,
    skip_live_data: bool,
) -> None:
    """Reconcile durable task phases once, after the prior process is gone."""
    if skip_live_data:
        return
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        reconcile_orphaned_running_tasks(drive_root)
    except Exception:
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)
    try:
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        recover_pending_root_post_task_synthesis(drive_root, repo_dir)
    except Exception:
        log.warning("Root post-task synthesis recovery at startup failed", exc_info=True)
