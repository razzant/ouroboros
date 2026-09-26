"""Upkeep a supervisor generation owes the drive.

The once-per-generation startup sweep (process custody, delegated runs, legacy
cancel latches, owed terminal deliveries, orphaned running results, pending
post-task synthesis), the throttled periodic cadences of the same surfaces, and
the delegated-snapshot GC that fails closed on an unreadable custody log.
"""

from __future__ import annotations

import pathlib
import json
import os
import threading
import time
from typing import Any

from ouroboros.server_process import DATA_DIR, log, _restart_requested, _supervisor_stop
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
_CANCEL_INTENT_SWEEP_LOCK = threading.Lock()
_CUSTODY_SWEEP_LOCK = threading.Lock()


def _stop_requested(stop_event: Any = None) -> bool:
    """Is this generation's mutation window closed?

    True once the supervisor loop that started the pass has exited (its
    per-generation ``_watchdog_stop`` token) or the process is stopping or
    restarting. Off the loop thread nothing else stops the pass, so it answers two
    questions with one fact: mutate nothing more, and reach the daemon ATTACH-ONLY
    — an ``ensure`` between a stop request and the daemon stop starts the engine
    the teardown is about to end (ARCHITECTURE §9, DEVELOPMENT Process Custody Rule).
    """
    return bool(_restart_requested.is_set() or _supervisor_stop.is_set()
                or (stop_event is not None and stop_event.is_set()))


def _live_task_ids() -> set:
    """The ONE live-owner source both custody surfaces and the cursor refresh read.

    Memory only — RUNNING and busy worker slots under ``_queue_lock``, the
    direct-activity registry, in-flight post-task synthesis: the owners
    ``queue.task_has_live_ownership`` names, without the durable result read no
    sweep may pay. Handed to its consumers as this CALLABLE so each evaluates it
    after reading its own candidates (``reap_orphaned_processes`` for the rule).
    """
    return _startup_live_task_ids(DATA_DIR)


def _run_cancel_delivery_ref_sweep(drive_root: pathlib.Path) -> None:
    """One existing maintenance pass; the drain retains no file/cancel work."""
    try:
        try:
            from supervisor.task_lifecycle import sweep_cancel_intents
            outcomes = sweep_cancel_intents()
            if outcomes:
                log.info("Cancel-intent watchdog settled: %s", outcomes)
        except Exception:
            log.debug("Cancel-intent watchdog sweep failed", exc_info=True)
        try:
            from supervisor.terminal_delivery import replay_pending_deliveries
            replay_pending_deliveries(drive_root)
        except Exception:
            log.debug("Pending terminal-delivery replay failed", exc_info=True)
        try:
            from ouroboros.observability import retry_pending_child_ref_promotions
            retry_pending_child_ref_promotions(drive_root)
        except Exception:
            log.debug("Pending child-ref promotion retry failed", exc_info=True)
        try:
            _reconcile_abandoned_usage(drive_root)
        except Exception:
            log.warning("Abandoned usage reconciliation failed", exc_info=True)
    finally:
        _CANCEL_INTENT_SWEEP_LOCK.release()


def _reconcile_abandoned_usage(drive_root: pathlib.Path) -> None:
    """Close unowned terminal-task attempts; price and remote custody stay separate."""
    from ouroboros import usage_accounting as usage
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm_claudexor import recover_model_attempt
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.transport_custody import ProviderNotDispatched, release_pre_dispatch_attempt
    from ouroboros.usage_ledger import is_abandoned_settlement
    from supervisor.events_task_done import _refresh_terminal_task_cost
    from supervisor.queue import task_has_live_ownership

    root = pathlib.Path(drive_root)
    with usage._locked(root):
        rows = list(usage._final_rows(usage._read_records_locked_cached(root)).values())
    tasks, refresh = {}, set()
    gateway, gateway_unavailable = None, False

    def eligible_task(task_id):
        if not task_id:
            return False
        if task_id not in tasks:
            try:
                task = load_task_result(root, task_id, strict=True) or {}
                checkpoint = task.get("root_phase_checkpoint") or {}
                tasks[task_id] = task if (
                    task.get("status") in SETTLED_STATUSES
                    and not task_has_live_ownership(task_id)
                    and not post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
                ) else None
            except Exception:
                tasks[task_id] = None  # Unreadable ownership permits neither duty.
        return tasks[task_id] is not None

    def borrowed_gateway():
        nonlocal gateway, gateway_unavailable
        if gateway_unavailable:
            raise ClaudexorUnavailable("daemon_unreachable", "Usage recovery deferred until the next maintenance pass")
        if gateway is None:
            try:
                gateway = read_owned_gateway()
            except Exception:
                gateway_unavailable = True
                raise
        return gateway

    try:
        for row in rows:
            kind = row.get("kind", "attempt")
            if kind not in {"attempt", "usage_baseline_group"} or any(row.get(key) for key in usage.REVIEW_ATTRIBUTION_KEYS):
                continue
            task_id = str(row.get("task_id") or "")
            # Settled/compacted attribution still owes projection after a failed write.
            refresh.update(owner for owner in (task_id, str(row.get("root_task_id") or "")) if eligible_task(owner))
            if not eligible_task(task_id):
                continue
            remote = row.get("provider") == "claudexor"
            abandoned = is_abandoned_settlement(row)
            if (kind != "attempt"
                or (row.get("state") not in {"reserved", "dispatched", "unresolved"}
                    and not (remote and abandoned))):
                continue
            reservation = usage.AttemptReservation(
                str(row["attempt_id"]), root, str(row.get("model") or ""),
                str(row.get("provider") or ""), row.get("reservation_upper_bound_usd"),
                str(row.get("processing_preference") or ""), str(row.get("submitted_processing_mode") or ""),
                row.get("processing_basis"),
            )
            try:
                recovered = None
                if remote and row.get("state") != "reserved":
                    recovered = recover_model_attempt(root, row, gateway_factory=borrowed_gateway)
                    if recovered is None:
                        continue
                disposition, reported, cost, final = recovered or ("abandoned", {}, None, False)
                if disposition == "settled":
                    usage.settle_attempt(reservation, reported, cost_usd=cost, cost_final=final)
                elif disposition == "released":
                    if not release_pre_dispatch_attempt(reservation, ProviderNotDispatched("recovered model operation never started")):
                        continue
                elif disposition == "abandoned":
                    if abandoned:
                        continue
                    state = usage.terminalize_abandoned_attempt(reservation, reason="owner_task_terminal", expected_seq=row.get("seq"))
                    if state not in {"settled", "released"}:
                        continue
                else:
                    continue
            except ClaudexorUnavailable as exc:
                gateway_unavailable = gateway_unavailable or exc.code == "daemon_unreachable"
                log.debug("Model usage custody deferred for %s: %s", row["attempt_id"], exc.code)
            except Exception:
                log.warning("Usage reconciliation deferred for %s", row["attempt_id"], exc_info=True)
    finally:
        if gateway is not None:
            try:
                gateway.close()
            except Exception:
                log.debug("Usage recovery gateway close failed", exc_info=True)
    if not refresh:
        return
    try:
        usage.ensure_legacy_imported(root)
        # One post-transition indexed view avoids per-owner scans; failure retries next pass.
        breakdown = usage.usage_breakdown(root)
    except Exception:
        log.warning("Reconciled usage projection unavailable", exc_info=True)
        return
    for task_id in sorted(refresh):
        try:
            _refresh_terminal_task_cost(root, task_id, breakdown=breakdown)
        except Exception:
            log.warning("Reconciled task cost refresh failed for %s", task_id, exc_info=True)


def _periodic_supervisor_maintenance(
    last_custody_reap: list, last_review_reconcile: list, *, on_orphans_healed: Any = None,
    stop_event: Any = None,
) -> None:
    """Throttled periodic upkeep extracted from the supervisor loop: cancel-intent
    watchdog and pending child-ref promotion replay (every 20s), custody reap of
    orphaned task-scoped processes (every 600s) + review-job zombie reconcile
    (every 300s). Each cadence gates itself via its own last-run marker, updated on
    the LOOP thread: the first two stamp before handing their work to a daemon
    thread; the inline zombie reconcile stamps when its pass ends.
    ``stop_event`` is the loop's per-generation token, handed to the custody pass so
    it stops mutating when that generation ends. ``on_orphans_healed(count)`` fires
    when the zombie reconcile terminalized orphaned RUNNING task rows (the alarm
    clock wakes early for them)."""
    if time.time() - _LAST_CANCEL_INTENT_SWEEP[0] > 20 and _CANCEL_INTENT_SWEEP_LOCK.acquire(blocking=False):
        _LAST_CANCEL_INTENT_SWEEP[0] = time.time()
        try:
            threading.Thread(target=_run_cancel_delivery_ref_sweep, args=(pathlib.Path(DATA_DIR),),
                             name="terminal-maintenance", daemon=True).start()
        except Exception:
            _CANCEL_INTENT_SWEEP_LOCK.release()
            log.warning("Terminal maintenance could not start", exc_info=True)
    latch = _CUSTODY_SWEEP_LOCK  # the pass releases THIS object, never a later generation's
    if time.time() - last_custody_reap[0] > 600 and latch.acquire(blocking=False):
        last_custody_reap[0] = time.time()
        try:
            threading.Thread(target=_run_periodic_custody_sweep, args=(stop_event, latch),
                             name="custody-maintenance", daemon=True).start()
        except Exception:
            latch.release()
            log.warning("Periodic custody sweep could not start", exc_info=True)
    if time.time() - last_review_reconcile[0] > 300:
        try:
            _periodic_zombie_reconcile(on_orphans_healed=on_orphans_healed)
        finally:
            # Stamped when the pass ENDS: a pass slower than its cadence never re-arms
            # on the next tick, so >=300 s of ordinary ticks separate two passes (issue #1230).
            last_review_reconcile[0] = time.time()


def _run_periodic_custody_sweep(stop_event: Any = None, latch: Any = None) -> None:
    """The ~600 s custody block, OFF the thread that answers workers (INV-B).

    Skill-payload hashing, the orphaned-process reaper, delegated-run reconciliation
    (gateway handshake, custody replays, registration retirement) and the
    settled-terminal cursor cost seconds to minutes — longer than any worker ack
    wait — so they run here, on the 20 s sweep's shape: the caller took
    ``_CUSTODY_SWEEP_LOCK`` without blocking (busy ⇒ the tick skips, never queues)
    and this pass releases it in ``finally``. Nothing serializes the pass against
    assignment any more, so each step reads its CANDIDATES before the shared live
    set, and the generation is re-read before every mutation.
    """
    try:
        try:
            if _stop_requested(stop_event):
                return
            from ouroboros.terminal_projection import reconcile_terminal_projections
            reconcile_terminal_projections(DATA_DIR)
        except Exception:
            log.warning("Terminal projection reconciliation deferred", exc_info=True)
        try:
            # Issue #844: release the owned-daemon start latch in ITS OWN try, ahead of
            # the reap, so a raising reap can never pin it; retry once — only when THIS
            # sweep released a latch — on a short-lived thread, as warm_owned_daemon()
            # does (the reconcile below ensures only with orphan work). Contract, per-
            # process scope and the residual: DEVELOPMENT.md "Process Custody Rule".
            from ouroboros.claudexor_daemon import get_owned_daemon

            if _stop_requested(stop_event):
                return
            if get_owned_daemon().clear_start_failure_latch(cleared_by="supervisor_sweep"):
                threading.Thread(target=_retry_latched_daemon_start,
                                 name="owned-daemon-latch-retry", daemon=True).start()
        except Exception:
            log.debug("Owned daemon latch release failed", exc_info=True)
        # The steps share one guard (a failure still ends the pass), but the row
        # must say WHICH one died: at DEBUG, and unnamed, a block that silently
        # stopped reaping for weeks looked exactly like one that had nothing to do.
        step = "reap_orphaned_processes"
        try:
            from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
            from ouroboros.process_custody import reap_orphaned_processes

            if _stop_requested(stop_event):
                return
            reap_orphaned_processes(
                DATA_DIR, running_task_ids=_live_task_ids,
                live_owner_skills=_installed_skill_names(),
                retained_purposes={CUSTODY_PURPOSE},
            )
            # A delegated Claudexor run is an orphan under exactly the same predicate:
            # its owning task is no longer running. It has no pid, so the process
            # reaper cannot see it — but it is still spending quota and still writing.
            step = "reconcile_delegated_runs"
            if _stop_requested(stop_event):
                return
            _reconcile_delegated_runs(_live_task_ids, stop_event=stop_event)
            step = "cursor_refresh_settled_terminals"
            if _stop_requested(stop_event):
                return
            _cursor_refresh_settled_terminals(_live_task_ids)
        except Exception:
            log.warning("Periodic custody step %s failed", step, exc_info=True)
    finally:
        (latch or _CUSTODY_SWEEP_LOCK).release()


def _retry_latched_daemon_start() -> None:
    """The one retry of a latched owned-daemon start (#844), made by the sweep itself.

    Runs on its own short-lived daemon thread, only after this sweep released
    the latch: one ``ensure_owned_gateway`` with ZERO admission and ZERO
    startup wait, so the supervisor loop never holds a startup wait (nor the
    unbounded runtime preparation) — the spawn happens, custody keeps the child, and
    ``daemon_starting`` is the EXPECTED answer (the next ordinary caller joins
    or settles it). Any other typed refusal (a child that died at once has
    already re-latched inside the manager) is logged as a warning; nothing is
    raised into the loop, nothing else is retried or scheduled, and a gateway
    that did open is closed at once (the reconcile that follows attaches on
    its own).
    """
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        ensure_owned_gateway(admission_wait_sec=0, startup_wait_sec=0).close()
    except ClaudexorUnavailable as exc:
        if exc.code == "daemon_starting":
            log.info("Owned daemon retry after latch release is starting under custody: %s", exc)
        else:
            log.warning("Owned daemon retry after latch release refused (%s): %s", exc.code, exc)
    except Exception:
        log.warning("Owned daemon retry after latch release failed unexpectedly", exc_info=True)


def _reconcile_delegated_runs(running_task_ids: Any, *, stop_event: Any = None) -> None:
    """Settle or cancel delegated runs whose owning task is gone (startup + tick)."""
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
        from ouroboros.delegate_custody import reconcile_orphaned_runs
        from ouroboros.delegate_recovery import recoverable_task_ids
        from ouroboros.owner_wait import restore_owner_wait_allowed
        from supervisor.queue import PENDING, _queue_lock

        with _queue_lock:
            pending_waits = [dict(task) for task in PENDING if task.get("_owner_wait_resume")]
        continued = {str(task["id"]) for task in pending_waits
                     if restore_owner_wait_allowed(DATA_DIR, task)}

        # Zero admission wait: a daemon in its recovery-only window is skipped until
        # the next sweep. Once a stop, restart or panic is in flight the factory is
        # ATTACH-ONLY — the one ensure this surface may still make belongs to the
        # latch retry above (ARCHITECTURE §9, DEVELOPMENT Process Custody Rule).
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: (read_owned_gateway() if _stop_requested(stop_event)
                                     else ensure_owned_gateway(admission_wait_sec=0)),
            recoverable_task_ids=recoverable_task_ids(DATA_DIR) | continued,
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


def _prune_event(event_type: str, keys: tuple, **reports: dict) -> None:
    """Emit when a report has evidence under ``keys``. GC no-ops stay silent;
    an observation report with measured journal sizes intentionally rows at boot."""
    from supervisor.state import append_jsonl

    if any(report.get(key) for report in reports.values() for key in keys):
        append_jsonl(DATA_DIR / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": event_type, **reports})


def _startup_worktree_prune() -> None:
    """Startup hygiene: prune orphaned subagent worktrees (after the custody sweep)."""
    try:
        from ouroboros import subagent_worktrees

        _prune_event("subagent_worktree_prune", ("removed",),
                     report=subagent_worktrees.prune_orphans())
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


def _startup_prune_sweeps(*, preserve_task_sources: bool = False) -> None:
    """Startup hygiene: prune stale task drives/trees and orphaned temp files."""
    try:
        from ouroboros.headless import prune_headless_task_drives, prune_task_drives, prune_task_trees
        from ouroboros.utils import sweep_stale_temp_files

        prune_report, task_drive_report = {}, {}
        if preserve_task_sources:
            log.warning("Startup task-source prune deferred: file recovery or ownership is unresolved")
        else:
            prune_report = prune_headless_task_drives(DATA_DIR)
            task_drive_report = prune_task_drives(DATA_DIR)
            prune_task_trees(DATA_DIR)
            sweep_stale_temp_files(DATA_DIR)
        _prune_event("headless_task_drive_prune", ("pruned", "errors"),
                     report=prune_report, task_drives=task_drive_report)
    except Exception:
        log.debug("Headless task drive prune failed", exc_info=True)
    try:
        # CPL4-C11 (owner batch 3A): clear owner state of tombstoned-uninstalled
        # skills; grants survive as owner authority, reinstalls self-heal.
        from ouroboros.skill_uninstall_state import sweep_uninstalled_skill_state

        _prune_event("skill_uninstall_state_sweep", ("swept", "restored", "errors"),
                     report=sweep_uninstalled_skill_state(DATA_DIR))
    except Exception:
        log.debug("Uninstalled-skill state sweep failed", exc_info=True)
    try:
        # CPL4-C14/C15: pure-cache and dead-marker age prunes (GC retention).
        from ouroboros.code_intelligence import prune_stale_code_intel_roots
        from ouroboros.extension_reconcile_queue import prune_failed_reconcile_markers

        _prune_event("stale_cache_prune", ("removed", "errors"),
                     code_intel=prune_stale_code_intel_roots(DATA_DIR),
                     extension_reconcile_failed=prune_failed_reconcile_markers(DATA_DIR))
    except Exception:
        log.debug("Stale cache prune failed", exc_info=True)
    try:
        # TZ-3 11A/B5 supersedes old age-digestion: measure journal growth
        # without touching historical or new full-text snapshots.
        from ouroboros.memory_journal_compaction import compact_memory_journal_snapshots

        _prune_event("memory_journal_observation", ("journal_bytes", "errors"),
                     report=compact_memory_journal_snapshots(DATA_DIR))
    except Exception:
        log.debug("Memory journal compaction failed", exc_info=True)
    try:
        # CPL4-C18: unlink mailboxes whose task settled off the terminal
        # dispatch path (fail-closed: no result keeps the mailbox).
        from ouroboros.owner_mailbox import sweep_settled_owner_mailboxes

        _prune_event("owner_mailbox_sweep", ("removed",), report=(
            {} if preserve_task_sources else sweep_settled_owner_mailboxes(DATA_DIR)))
    except Exception:
        log.debug("Owner mailbox sweep failed", exc_info=True)
    try:
        # CPL4-C21 (owner 6A): agent screenshots/views follow GC retention;
        # owner attachments in the uploads/ root are never touched.
        _prune_event("agent_media_prune", ("removed", "skipped", "errors"),
                     report=prune_agent_media_uploads(DATA_DIR))
    except Exception:
        log.debug("Agent media prune failed", exc_info=True)
    if not preserve_task_sources:
        try:
            from ouroboros.observability import prune_observability_blobs
            from ouroboros.tools.services import prune_service_logs

            _prune_event(
                "runtime_artifact_prune",
                ("manifest_count", "blob_count", "deleted_dirs", "deleted_files", "errors"),
                observability=prune_observability_blobs(DATA_DIR),
                services=prune_service_logs(DATA_DIR))
        except Exception:
            log.debug("Runtime artifact prune failed", exc_info=True)


def _cursor_refresh_settled_terminals(live_task_ids: Any = None) -> None:
    """Cursor-driven pass: runs settled OUTSIDE a generation's reconcile
    outcomes (terminal-boundary settlements, earlier generations) never
    reappear in the orphan sweep, so their tasks' stored evidence would stay
    stale forever. Bounded to newly appended custody rows per tick, and reading
    the same live-owner source as both custody surfaces: a task whose owner is
    still billing is deferred rather than healed under a live writer. At BOOT
    this runs AFTER the D1a backfill (see ``_startup_custody_sweep``), so a
    same-generation heal keeps its pinned ``boot_backfill`` attribution and
    the cursor's change-gated pass advances past it without a second write.
    """
    try:
        from ouroboros.delegate_terminal import refresh_recently_settled_terminals

        refreshed = refresh_recently_settled_terminals(DATA_DIR, live_task_ids=live_task_ids)
        if refreshed:
            log.info("Cursor refresh healed %d stale terminal result(s)", refreshed)
    except Exception:
        log.debug("Cursor terminal-refresh pass failed", exc_info=True)


def _startup_custody_sweep() -> None:
    """Both custody surfaces, swept once per generation at supervisor startup.

    Live owners can survive an in-process supervisor revival. The shared
    installation daemon and those owners keep their existing custody.
    """
    try:
        from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
        from ouroboros.process_custody import reap_orphaned_processes

        reaped = reap_orphaned_processes(
            DATA_DIR, live_owner_skills=_installed_skill_names(),
            running_task_ids=_live_task_ids,
            retained_purposes={CUSTODY_PURPOSE},
        )
        if reaped:
            log.info("Process custody reaper killed %d orphaned process(es): %s", len(reaped), reaped)
    except Exception:
        log.debug("Process custody startup reap failed", exc_info=True)
    _reconcile_delegated_runs(_live_task_ids)
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
    _cursor_refresh_settled_terminals(_live_task_ids)
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

        _prune_event("delegate_state_sweep", ("removed", "errors", "skipped"),
                     report=sweep_settled_delegate_state(DATA_DIR))
    except Exception:
        log.debug("Delegate state sweep failed", exc_info=True)


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


def _periodic_zombie_reconcile(*, on_orphans_healed: Any = None) -> None:
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

        expired_quizzes: list = []
        healed = reconcile_orphaned_running_tasks(DATA_DIR, expired_quizzes=expired_quizzes)
        _publish_expired_quiz_frames(expired_quizzes)
        if healed and callable(on_orphans_healed):
            on_orphans_healed(int(healed))
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


def _migrate_startup_cancel_latches(drive_root: pathlib.Path) -> None:
    """Before restore/readers can quarantine an unstamped legacy cancel latch."""
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


def _publish_expired_quiz_frames(expired: list) -> None:
    """Tell already-rendered cards that a healed terminal expired their question.

    The same frame the task-done seam sends, from the one caller that is on the
    supervisor side: the healer writes terminals off that seam, and the surfaces
    Ouroboros runs on (packaged shell, mini app, phone) have no reload
    affordance, so a card would keep a clickable question until navigation.
    Fail-soft: the durable projection is already correct without the frame.
    """
    if not expired:
        return
    try:
        from supervisor.message_bus import get_bridge

        bridge = get_bridge()
        for task_id, quiz_id in expired:
            bridge.send_quiz_state(str(quiz_id), str(task_id), "expired_terminal")
    except Exception:
        log.debug("Expired-quiz frames after orphan reconcile were not sent", exc_info=True)


def _startup_live_task_ids(drive_root: pathlib.Path, *, include_pending: bool = False) -> set[str]:
    """Existing queue/direct/post-task owners; recovery also preserves pending work."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from supervisor import queue, workers
    from supervisor.active_activity import get_direct_activity_registry

    with queue._queue_lock:
        ids = set(queue.RUNNING)
        if include_pending:
            ids.update(str(task.get("id") or "") for task in queue.PENDING)
        ids.update(w.busy_task_id for w in workers.WORKERS.values() if w.busy_task_id)
    ids.update(row["activity_id"] for row in get_direct_activity_registry().snapshot())
    root = str(pathlib.Path(drive_root).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        ids.update(task_id for (path, task_id) in POST_TASK_SYNTHESIS_INFLIGHT if path == root)
    if include_pending:
        # No-provider startup has not restored PENDING. The saved continuation
        # remains owned by queue restore, never by file/post-task recovery.
        from ouroboros.task_status import _load_queue_snapshot
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        if snapshot.get("_snapshot_invalid"):
            raise ValueError("startup pending ownership is unreadable")
        ids.update(str(row["task"].get("id") or "") for row in snapshot.get("pending", [])
                   if isinstance(row, dict) and isinstance(row.get("task"), dict)
                   and row["task"].get("_owner_wait_resume"))
    return ids - {""}


def _startup_worker_pids(drive_root: pathlib.Path) -> set[int] | None:
    """Capture prior worker ownership before spawn overwrites its legacy pid file.

    These observations only defer recovery; they grant no signal authority. An
    unconfirmed survivor or unreadable record never proves safe file capture.
    """
    from ouroboros.process_custody import _read_ledger_strict
    try:
        path = pathlib.Path(drive_root) / "state" / "worker_pids.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            record = {"workers": []}
        pids = {int(row["pid"]) for row in record["workers"]}
        server_pid = int(record.get("server_pid") or 0)
        if server_pid and server_pid != os.getpid():
            pids.add(server_pid)
        readable, entries = _read_ledger_strict(pathlib.Path(drive_root))
        if not readable:
            return None
        pids.update(int(row["pid"]) for row in entries if str(row.get("purpose") or "").startswith("worker:"))
        return pids - {0, os.getpid()}
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("Startup worker ownership is unreadable; file recovery is deferred", exc_info=True)
        return None


def _recover_terminal_task_files(drive_root: pathlib.Path, protected: set[str]) -> dict:
    """Recover only known child directories, never infer a new model execution."""
    from ouroboros.headless import (
        HEADLESS_TASKS_DIR, TASK_DRIVES_DIR, prepare_terminal_task_files,
        terminal_task_files_ready,
    )
    from ouroboros.observability import _has_pending_ref_promotion
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.task_results import load_task_result, validate_task_id, write_task_result
    from ouroboros.task_status import SETTLED_STATUSES, effective_task_result

    root = pathlib.Path(drive_root)
    report = {"recovered": [], "unresolved": [], "protected": sorted(protected), "errors": []}
    for base, suffix in ((root / HEADLESS_TASKS_DIR, "data"), (root / TASK_DRIVES_DIR, "")):
        try:
            directories = sorted(base.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            report["errors"].append(str(exc))
            report["unresolved"].append("*")
            continue
        for directory in directories:
            if not directory.is_dir() or directory.is_symlink() or directory.name in protected:
                continue
            task_id, child_root = directory.name, directory / suffix
            try:
                base_resolved = base.resolve(strict=True)
                directory_resolved = directory.resolve(strict=True)
                if directory_resolved.parent != base_resolved:
                    continue
                if suffix and child_root.is_symlink():
                    continue
                child_root.resolve(strict=True).relative_to(directory_resolved)
                validate_task_id(task_id)
                current = load_task_result(root, task_id, strict=True) or {}
                task = {**current, "id": task_id, "drive_root": str(child_root)}
                ready = terminal_task_files_ready(root, task, current)
                pending = _has_pending_ref_promotion(current.get("child_ref_promotion"))
                if ready and not pending:
                    continue  # Already saved; do not re-copy or recapture on every boot.
                if not ready:
                    source = load_task_result(child_root, task_id, strict=True) or {}
                    if source.get("status") not in SETTLED_STATUSES:
                        if (current.get("status") == "scheduled" and source.get("status") == "running"
                                and source.get("started_at") and not source.get("_is_direct_chat")
                                and not cancel_pending(root, task_id, strict=True)):
                            # Older split roots omitted their canonical start.
                            # Rebind only when the existing queue/worker/direct
                            # ownership rules already prove this child orphaned.
                            observed = effective_task_result(root, {
                                **current, "child_drive_root": str(child_root),
                            }, materialize_artifacts=False)
                            if observed.get("reason_code") == "orphaned_running_after_worker_restart":
                                write_task_result(
                                    root, task_id, "running", child_drive_root=str(child_root),
                                    budget_drive_root=str(root), started_at=source["started_at"],
                                    ts=source.get("ts") or source["started_at"],
                                )
                                report.setdefault("rebound", []).append(task_id)
                        if (pending or current.get("status") == "completed") and (
                            suffix or current.get("headless_child_drive_root") or current.get("child_drive_root")
                        ):
                            report["unresolved"].append(task_id)
                        continue  # Ordinary canonical task scratch has no child result.
                    task = {**source, **current, "id": task_id, "drive_root": str(child_root)}
                prepared = prepare_terminal_task_files(root, task)
                settled = load_task_result(root, task_id, strict=True)
                if prepared["error"] or not terminal_task_files_ready(root, task, settled):
                    report["unresolved"].append(task_id)
                else:
                    report["recovered"].append(task_id)
            except Exception as exc:
                report["unresolved"].append(task_id)
                report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}")
    return report


def _run_startup_task_recovery(
    drive_root: pathlib.Path, repo_dir: pathlib.Path, *, skip_live_data: bool,
    prior_worker_pids: set[int] | None = None,
) -> dict:
    """File recovery precedes orphan materialization and the caller's actual prune.

    Provider boot calls this from the supervisor, after process custody; the
    no-provider lifespan calls it without spawning anything. There is no racing
    lifespan recovery once a supervisor can run. Unknown/live prior ownership
    defers the destructive work, without waiting or resurrecting a model task.
    """
    report = {"recovered": [], "unresolved": [], "protected": [], "errors": []}
    if skip_live_data:
        return report
    _migrate_startup_cancel_latches(drive_root)
    from ouroboros.platform_layer import pid_is_alive
    if prior_worker_pids is None or any(pid_is_alive(pid) for pid in prior_worker_pids):
        report["errors"].append("prior_worker_ownership_unconfirmed")
        log.warning("Startup file/task recovery deferred: prior worker ownership is unconfirmed")
        return report
    try:
        protected = _startup_live_task_ids(drive_root, include_pending=True)
    except Exception as exc:
        report["errors"].append(f"startup_ownership_unconfirmed: {type(exc).__name__}")
        log.warning("Startup task recovery deferred: live/pending ownership is unreadable", exc_info=True)
        return report
    report = _recover_terminal_task_files(drive_root, protected)
    excluded = protected | set(report["unresolved"])
    if report["errors"]:
        log.warning("Startup task file recovery has gaps: %s", report)
    if "*" in excluded:
        return report  # Directory enumeration could not identify the unsafe rows.
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        expired_quizzes: list = []
        reconcile_orphaned_running_tasks(
            drive_root, exclude_task_ids=excluded, expired_quizzes=expired_quizzes,
        )
        _publish_expired_quiz_frames(expired_quizzes)
    except Exception:
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)
    try:
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        recover_pending_root_post_task_synthesis(drive_root, repo_dir, exclude_task_ids=excluded)
    except Exception:
        log.warning("Root post-task synthesis recovery at startup failed", exc_info=True)
    return report
