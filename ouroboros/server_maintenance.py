"""Upkeep a supervisor generation owes the drive.

The once-per-generation startup sweep (process custody, delegated runs, legacy
cancel latches, owed terminal deliveries, orphaned running results, pending
post-task synthesis), the throttled periodic cadences of the same surfaces, and
the delegated-snapshot GC that fails closed on an unreadable custody log.
"""

from __future__ import annotations

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
    watchdog and pending child-ref promotion replay (every 20s), custody reap of
    orphaned task-scoped processes (every 600s) + review-job zombie reconcile
    (every 300s). Each cadence gates itself via its own last-run marker."""
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
        try:
            from ouroboros.observability import retry_pending_child_ref_promotions

            retry_pending_child_ref_promotions(DATA_DIR)
        except Exception:
            log.debug("Pending child-ref promotion retry failed", exc_info=True)
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
            _cursor_refresh_settled_terminals()
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
        from ouroboros.delegate_recovery import recoverable_task_ids

        # The tick runs on the supervisor loop thread: a daemon sitting in its
        # recovery-only admission window must not hold that thread for the default
        # admission wait — skip-until-next-sweep is this caller's normal posture.
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: ensure_owned_gateway(admission_wait_sec=0),
            recoverable_task_ids=recoverable_task_ids(DATA_DIR),
        )
        if outcomes:
            log.info("Delegated-run reconciliation handled %d orphan(s): %s", len(outcomes), outcomes)
            # A run settled by this sweep may belong to a task that already wrote
            # its terminal result with a non-empty unreconciled disclosure — the
            # stored projection then lies forever (nanny-leaf S1). Audit-only
            # refresh; never cancels.
            from ouroboros.delegate_terminal import refresh_terminal_reconciliation

            for tid in {str(o.get("task_id") or "") for o in outcomes
                        if o.get("task_id") and (o.get("settled") or str(
                            o.get("action") or "") in (
                                "absent", "cancelled", "invocation_retired"))}:
                try:
                    refresh_terminal_reconciliation(DATA_DIR, tid)
                except Exception:
                    log.debug("Sweep terminal-result refresh failed for %s", tid, exc_info=True)
    except Exception:
        log.debug("Delegated-run reconciliation failed", exc_info=True)


def _startup_retired_settings_notice(settings: dict) -> None:
    """Tell the OWNER, in their chat, that retired keys in ``settings.json`` are NOT honored.

    ``config.normalize_settings_raw`` reports the loss on the module logger only, which an
    owner who never opens the Logs panel does not see — and the reviewer comma-lists are
    the case that matters: an install upgraded without authoring
    ``OUROBOROS_REVIEWER_SLOTS`` silently runs the shipped default panel, and one that
    authored it malformed has every review refused until it is repaired. The dropped sets
    come from that same read seam (``config.retired_key_sets_seen``), the sentence is the
    one the log line uses (``settings_defaults.retired_setting_keys_notice``, fed the
    document's absent / authored / invalid state by
    ``reviewer_slot_config.authored_reviewer_slots_state``), and the
    dedupe is durable: ``state.json:retired_settings_notified`` keyed by the exact
    retired-key set, so a restart or a supervisor revival never repeats it. Nothing is
    sent — and nothing marked — while no owner chat is bound: the notice waits for the
    first boot that has somewhere to deliver it.
    """
    try:
        from ouroboros.config import retired_key_sets_seen
        from ouroboros.reviewer_slot_config import authored_reviewer_slots_state
        from ouroboros.settings_defaults import retired_setting_keys_notice
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        if not owner_chat:
            return
        notified = state.get("retired_settings_notified")
        notified = notified if isinstance(notified, dict) else {}
        slots_state = authored_reviewer_slots_state(
            str((settings or {}).get("OUROBOROS_REVIEWER_SLOTS") or ""))
        for dropped in retired_key_sets_seen():
            marker = ",".join(dropped)
            if marker in notified:
                continue
            send_with_budget(
                owner_chat,
                "⚙️ Settings: " + retired_setting_keys_notice(
                    dropped, reviewer_slots=slots_state),
                role="system", system_type="retired_settings_notice",
            )

            def _mark(st: dict, key: str = marker) -> None:
                seen = st.get("retired_settings_notified")
                seen = dict(seen) if isinstance(seen, dict) else {}
                seen[key] = utc_now_iso()
                st["retired_settings_notified"] = seen

            update_state(_mark)
    except Exception:
        log.debug("retired settings owner notice failed", exc_info=True)


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


def prune_agent_media_uploads(
    drive_root: pathlib.Path,
    retention_days: "int | None" = None,
    *,
    now: "float | None" = None,
) -> dict:
    """Age-prune AGENT-generated media under uploads/ (CPL4-C21, owner 6A).

    Only ``uploads/screenshots/`` (browser tool) and ``uploads/views/``
    (view_image durable copies) follow GC retention — owner attachments live
    in the uploads/ ROOT and are owner-explicit-delete only, untouched here.
    Readers already skip missing files (the eviction placeholder only formats
    a re-view hint), so a pruned screenshot degrades to a stale hint, never
    an error. Fail-soft per file.

    CONTAINED (audit #15-13): the sweep deletes only REGULAR FILES that live
    inside the real family directory. ``is_file()``/``stat()`` follow symlinks,
    so a symlinked ``uploads/screenshots`` — or a single symlink inside it —
    made an age sweep of the drive unlink old files anywhere on the host. Both
    shapes are now skipped by ``lstat`` and counted, never followed.
    """
    import stat as stat_module

    from ouroboros.retention import age_cutoff, get_gc_retention_days

    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    report: dict = {"removed": 0, "kept": 0, "skipped": 0, "errors": 0}
    for family in ("screenshots", "views"):
        family_dir = pathlib.Path(drive_root) / "uploads" / family
        try:
            if family_dir.is_symlink():
                report["skipped"] += 1  # the whole family points out of the drive
                continue
            entries = sorted(family_dir.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                info = path.lstat()
                if not stat_module.S_ISREG(info.st_mode):
                    report["skipped"] += 1  # symlink, directory or special file
                    continue
                if info.st_mtime >= cutoff:
                    report["kept"] += 1
                    continue
                path.unlink()
                report["removed"] += 1
            except OSError:
                report["errors"] += 1
    return report


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
    try:
        # CPL4-C11 (owner batch 3A): clear owner state of tombstoned-uninstalled
        # skills; grants survive as owner authority, reinstalls self-heal.
        from ouroboros.skill_uninstall_state import sweep_uninstalled_skill_state

        tombstone_report = sweep_uninstalled_skill_state(DATA_DIR)
        if any(tombstone_report.get(key) for key in ("swept", "restored", "errors")):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "skill_uninstall_state_sweep",
                "report": tombstone_report,
            })
    except Exception:
        log.debug("Uninstalled-skill state sweep failed", exc_info=True)
    try:
        # CPL4-C14/C15: pure-cache and dead-marker age prunes (GC retention).
        from ouroboros.code_intelligence import prune_stale_code_intel_roots
        from ouroboros.extension_reconcile_queue import prune_failed_reconcile_markers

        intel_report = prune_stale_code_intel_roots(DATA_DIR)
        failed_report = prune_failed_reconcile_markers(DATA_DIR)
        if (
            intel_report.get("removed") or intel_report.get("errors")
            or failed_report.get("removed") or failed_report.get("errors")
        ):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "stale_cache_prune",
                "code_intel": intel_report,
                "extension_reconcile_failed": failed_report,
            })
    except Exception:
        log.debug("Stale cache prune failed", exc_info=True)
    try:
        # CPL4-C16 (owner 4A): memory-journal snapshots older than GC retention
        # become digest-only (sha256 + length); fresh entries keep full text.
        from ouroboros.memory_journal_compaction import compact_memory_journal_snapshots

        journal_report = compact_memory_journal_snapshots(DATA_DIR)
        if (
            journal_report.get("digested")
            or journal_report.get("digest_mismatch")
            or journal_report.get("errors")
        ):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "memory_journal_compaction",
                "report": journal_report,
            })
    except Exception:
        log.debug("Memory journal compaction failed", exc_info=True)
    try:
        # CPL4-C18: unlink mailboxes whose task settled off the terminal
        # dispatch path (fail-closed: no result keeps the mailbox).
        from ouroboros.owner_mailbox import sweep_settled_owner_mailboxes

        mailbox_report = sweep_settled_owner_mailboxes(DATA_DIR)
        if mailbox_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "owner_mailbox_sweep",
                "report": mailbox_report,
            })
    except Exception:
        log.debug("Owner mailbox sweep failed", exc_info=True)
    try:
        # CPL4-C21 (owner 6A): agent screenshots/views follow GC retention;
        # owner attachments in the uploads/ root are never touched.
        media_report = prune_agent_media_uploads(DATA_DIR)
        if (
            media_report.get("removed")
            or media_report.get("skipped")
            or media_report.get("errors")
        ):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "agent_media_prune",
                "report": media_report,
            })
    except Exception:
        log.debug("Agent media prune failed", exc_info=True)
    try:
        # CPL4-C23: acknowledged observations older than GC retention fold into
        # an archive segment; unacknowledged rows are never pruned. Runs before
        # Background Consciousness starts (it is created later in startup).
        from ouroboros.consciousness import compact_acknowledged_observations

        fold_report = compact_acknowledged_observations(DATA_DIR)
        if fold_report.get("folded") or fold_report.get("skipped"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_observation_fold",
                "report": fold_report,
            })
    except Exception:
        log.debug("Observation fold failed", exc_info=True)


def _cursor_refresh_settled_terminals() -> None:
    """Cursor-driven pass: runs settled OUTSIDE a generation's reconcile
    outcomes (terminal-boundary settlements, earlier generations) never
    reappear in the orphan sweep, so their tasks' stored evidence would stay
    stale forever. Bounded to newly appended custody rows per tick. At BOOT
    this runs AFTER the D1a backfill (see ``_startup_custody_sweep``), so a
    same-generation heal keeps its pinned ``boot_backfill`` attribution and
    the cursor's change-gated pass advances past it without a second write.
    """
    try:
        from ouroboros.delegate_terminal import refresh_recently_settled_terminals

        refreshed = refresh_recently_settled_terminals(DATA_DIR)
        if refreshed:
            log.info("Cursor refresh healed %d stale terminal result(s)", refreshed)
    except Exception:
        log.debug("Cursor terminal-refresh pass failed", exc_info=True)


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
        # D1a boot backfill, ONCE per generation and AFTER the orphan reconcile
        # (so this generation's settlements are already visible to the audit):
        # a run settled in a PREVIOUS generation never appears in any current
        # pass's outcomes, so the sweep-side refresh above can never reach its
        # task's stored disclosure — the backfill joins from the stored terminal
        # results instead and heals every generation-crossing stale row.
        from ouroboros.delegate_terminal import backfill_terminal_reconciliations

        refreshed = backfill_terminal_reconciliations(DATA_DIR)
        if refreshed:
            log.info("Boot custody backfill refreshed %d stored disclosure(s): %s",
                     len(refreshed), refreshed)
    except Exception:
        log.debug("Boot custody-disclosure backfill failed", exc_info=True)
    _cursor_refresh_settled_terminals()
    try:
        # Boot half of the durable terminal outbox: an answer that was registered
        # as owed but whose send never completed (crash between settle and send)
        # is re-enqueued exactly once — the delivered registry suppresses a copy
        # that actually landed.
        from supervisor.terminal_delivery import replay_pending_deliveries

        replay_pending_deliveries(DATA_DIR)
    except Exception:
        log.debug("Boot replay of pending terminal deliveries failed", exc_info=True)
    try:
        # CPL4-C13: terminal+age sweep of delegate recovery/supervision files —
        # beside the custody sweep, fail-closed on unreadable custody exactly
        # like _prune_delegated_snapshots.
        from ouroboros.delegate_state_sweep import sweep_settled_delegate_state
        from supervisor.state import append_jsonl

        sweep_report = sweep_settled_delegate_state(DATA_DIR)
        if sweep_report.get("removed") or sweep_report.get("errors") or sweep_report.get("skipped"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegate_state_sweep",
                "report": sweep_report,
            })
    except Exception:
        log.debug("Delegate state sweep failed", exc_info=True)
    try:
        # CPL-5 reverse direction (model_send only): every seal joins exactly
        # one accounting attempt, every seam-sealed dispatched attempt still
        # resolves to its durable seal. Orphans on either side become typed
        # facts — the sweep deletes nothing and fabricates nothing, so there is
        # no destructive conclusion for an UNKNOWN state to skip (it skips the
        # whole pass instead when the ledger is unreadable).
        from ouroboros.model_send_seal import reconcile_model_send_seals

        seal_report = reconcile_model_send_seals(DATA_DIR)
        if seal_report.get("facts_written"):
            log.warning(
                "model_send invariant reconciliation wrote %d typed fact(s): %s",
                seal_report["facts_written"], seal_report,
            )
    except Exception:
        log.debug("model_send seal reconciliation failed", exc_info=True)


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
    """Reconcile durable task phases once, after the prior process is gone.

    The cancel-latch migration goes FIRST, ahead of every other durable
    task-result read this boot performs. Under ABI-2 a pre-redesign latch file
    is unstamped, so whichever read reaches it first quarantines it: the
    orphan reconcile below used to win that race and the wedged task then
    reached no terminal at all. The migration carries the one carve-out that
    admits those rows (see ``cancel_intents.migrate_legacy_cancel_latches``),
    so it must run before the readers whose quarantine it is exempting them
    from — every OTHER unstamped row still quarantines on the next read.
    """
    if skip_live_data:
        return
    try:
        # Phase A boot migration: legacy ``cancel_requested`` status latches
        # become ordinary durable cancel intents; the supervisor watchdog then
        # drives each through custody to a real settled outcome.
        from ouroboros.cancel_intents import migrate_legacy_cancel_latches

        migrated = migrate_legacy_cancel_latches(drive_root)
        if migrated:
            log.info("Migrated %d legacy cancel latch(es) to durable intents: %s",
                     len(migrated), migrated)
    except Exception:
        log.debug("Legacy cancel-latch migration failed", exc_info=True)
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
