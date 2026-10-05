"""Upkeep a supervisor generation owes the drive: the once-per-generation startup
sweep (process custody, delegated runs, legacy cancel latches, owed terminal
deliveries, orphaned running results, pending post-task synthesis), the throttled
periodic cadences of the same surfaces — every history-sized one off the loop
thread — and the delegated-snapshot GC that fails closed on an unreadable log."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict

from ouroboros.server_process import DATA_DIR, _restart_requested, _supervisor_stop, log
from ouroboros.utils import utc_now_iso


def _installed_skill_names():
    """Disk-derived skill owners for companion reaping; unknown means KEEP.
    Reload timing and failed discovery must never mass-reap live companions.
    """
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import discover_skills

        names = {s.name for s in discover_skills(DATA_DIR, repo_path=get_skills_repo_path())}
        # An EMPTY result is None ("unknown"), NOT "everything uninstalled":
        # discover_skills returns [] when the skills dir is momentarily unavailable,
        # and an enforced reap over that would mass-kill live companions.
        return names or None
    except Exception:
        log.debug("Could not compute installed skill names for custody reaper", exc_info=True)
        return None


_LAST_CANCEL_INTENT_SWEEP = [0.0]
_CANCEL_INTENT_SWEEP_LOCK = threading.Lock()
_CUSTODY_SWEEP_LOCK = threading.Lock()
_RECONCILE_SWEEP_LOCK = threading.Lock()


@contextmanager
def orphan_reconcile_write_guard(task_id: str, *, stop_event: Any = None):
    """Fence the orphan reconciler's writes against assignment (queue -> row lock order) in an open generation:
    the settlement probe must prove absence; a process with no supervisor (no-provider boot) dispatches nothing."""
    from supervisor import queue

    with queue._queue_lock:
        yield not _stop_requested(stop_event) and (not queue.INITIALIZED or queue.task_settlement_liveness(task_id) is False)


def _stop_requested(stop_event: Any = None) -> bool:
    """Closed generation, stop or restart: no mutation or daemon start afterward.
    The per-generation token also keeps an old off-loop pass from writing.
    """
    return bool(_restart_requested.is_set() or _supervisor_stop.is_set()
                or (stop_event is not None and stop_event.is_set()))


def _live_task_ids() -> set:
    """Fresh in-memory owners for custody and cursor refresh, never result rows.
    Passed as a callable so consumers recheck after collecting candidates.
    """
    return _startup_live_task_ids(DATA_DIR)


def _run_cancel_delivery_ref_sweep(drive_root: pathlib.Path, stop_event: Any = None) -> None:
    """20 s cancel/delivery/usage pass; history-sized work rides the 300 s reconcile pass."""
    try:
        try:
            from supervisor.task_lifecycle import sweep_cancel_intents
            outcomes = sweep_cancel_intents()
            if outcomes:
                log.info("Cancel-intent watchdog settled: %s", outcomes)
            _step_recovered("cancel_intent_sweep")
        except Exception:
            _step_failed("cancel_intent_sweep")
        try:
            from supervisor.terminal_delivery import replay_pending_deliveries
            replay_pending_deliveries(drive_root)
            _step_recovered("terminal_delivery_replay")
        except Exception:
            _step_failed("terminal_delivery_replay")
        try:
            from ouroboros.pause_notices import reconcile_pause_notices
            reconcile_pause_notices(drive_root, stop_requested=stop_event.is_set if stop_event is not None else None)
            _step_recovered("pause_notice_replay")
        except Exception:
            _step_failed("pause_notice_replay")
        try:
            _reconcile_abandoned_usage(drive_root)
            _step_recovered("abandoned_usage_reconciliation")
        except Exception:
            _step_failed("abandoned_usage_reconciliation")
    finally:
        _CANCEL_INTENT_SWEEP_LOCK.release()


def _reconcile_abandoned_usage(drive_root: pathlib.Path) -> None:
    """Compatibility seam for the existing terminal-maintenance duty."""
    from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage

    reconcile_abandoned_usage(drive_root)


# Memory only: consecutive failures per periodic step, so a failure that recurs every cadence
# is a WARNING with its traceback on failures 1, 2, 4, 8, ... and DEBUG in between, and its
# recovery is one INFO naming the streak (never a silent DEBUG death, never a wall of WARNINGs).
_STEP_FAILURES: Dict[str, int] = {}


def _step_failed(step: str, message: str = "%s failed") -> None:
    streak = _STEP_FAILURES.get(step, 0) + 1
    _STEP_FAILURES[step] = streak
    level = logging.WARNING if streak & (streak - 1) == 0 else logging.DEBUG
    log.log(level, message + " (failure %d in a row)", step, streak, exc_info=True)


def _step_recovered(step: str) -> None:
    streak = _STEP_FAILURES.pop(step, 0)
    if streak:
        log.info("%s recovered after %d failure(s)", step, streak)


# Memory only: the off-loop pass each latch is running (its thread, start stamp, whether its
# stall was journaled), so a cadence whose next tick finds the latch still held names the duty
# and the line its thread stands on: ``host_duty_stall`` once, ``host_duty_stall_end`` when
# the pass ends. The threshold is the duty's OWN cadence measured from the pass's start, never
# the loop deadline and never the tick alone: a marker stamped when a pass ENDS stays old while
# the next pass starts, so a due tick proves nothing about how long the current pass has run.
_DUTIES: Dict[int, Dict[str, Any]] = {}


def _duty_busy(latch: Any, cadence_sec: float) -> None:
    duty = _DUTIES.get(id(latch))
    if duty is None or duty["alerted"] or time.time() - duty["since"] < cadence_sec:
        return
    duty["alerted"] = True
    from ouroboros.server_liveness import _loop_thread_stack
    from supervisor.state import append_jsonl

    flags: Dict[str, Any] = {}
    stack = _loop_thread_stack(duty["ident"], facts=flags)
    running = round(time.time() - duty["since"], 1)
    log.warning("Off-loop duty %s has run %.0fs, past its %.0fs cadence; its thread is at: %s",
                duty["name"], running, cadence_sec, stack[-1] if stack else "(stack unavailable)")
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "host_duty_stall", "duty": duty["name"], "running_sec": running,
            "cadence_sec": cadence_sec, **flags, **({"stack": stack} if stack else {})})
    except Exception:
        log.debug("host_duty_stall row failed", exc_info=True)


def _duty_ended(latch: Any, duty: Dict[str, Any]) -> None:
    """Close THIS pass's duty. The pass released its latch before this runs, so the next pass
    may already have registered its own entry: that one is put back, never dropped (it holds
    the latch, so nothing registers again meanwhile)."""
    current = _DUTIES.pop(id(latch), None)
    if current is not None and current is not duty:
        _DUTIES.setdefault(id(latch), current)
    if not duty["alerted"]:
        return
    from supervisor.state import append_jsonl

    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "host_duty_stall_end", "duty": duty["name"],
            "running_sec": round(time.time() - duty["since"], 1)})
    except Exception:
        log.debug("host_duty_stall_end row failed", exc_info=True)


def _start_maintenance_thread(latch: Any, name: str, target: Any, args: tuple) -> bool:
    """Start one off-loop pass under a latch the CALLER already took without blocking
    (busy => that tick skipped, never queued); a start that fails releases it. The pass is
    watched as a duty (``_duty_busy`` / ``_duty_ended``)."""
    def run() -> None:
        duty["ident"] = thread.ident  # the pass names its own thread: no race with its end
        try:
            target(*args)
        finally:
            _duty_ended(latch, duty)

    duty: Dict[str, Any] = {"name": name, "ident": None, "since": time.time(), "alerted": False}
    _DUTIES[id(latch)] = duty
    try:
        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return True
    except Exception:
        _DUTIES.pop(id(latch), None)
        latch.release()
        log.warning("%s could not start", name, exc_info=True)
        return False


def _periodic_supervisor_maintenance(
    last_custody_reap: list, last_review_reconcile: list, *, on_orphans_healed: Any = None,
    stop_event: Any = None,
) -> None:
    """Throttled upkeep on the supervisor tick: three cadences, each a non-blocking
    latch plus a last-run marker, nothing history-sized inline (INV-24). Every 20 s the
    cancel-intent watchdog, terminal delivery replay and abandoned usage; every 600 s
    the custody block; every 300 s the reconcile block (zombie heal, then pending
    child-ref promotion retry). The first two stamp here before handing off; the
    reconcile pass stamps when it ENDS (issue #1230). A due cadence whose latch is still
    held journals the duty's stall once, with its thread's stack (``_duty_busy``).
    ``stop_event`` is the loop's per-generation token: both long passes stop mutating
    once it closes. ``on_orphans_healed(count)`` fires off-loop when orphaned RUNNING rows healed."""
    now = time.time()
    if now - _LAST_CANCEL_INTENT_SWEEP[0] > 20:
        if _CANCEL_INTENT_SWEEP_LOCK.acquire(blocking=False):
            _LAST_CANCEL_INTENT_SWEEP[0] = now
            _start_maintenance_thread(_CANCEL_INTENT_SWEEP_LOCK, "terminal-maintenance",
                                      _run_cancel_delivery_ref_sweep, (pathlib.Path(DATA_DIR), stop_event))
        else:
            _duty_busy(_CANCEL_INTENT_SWEEP_LOCK, 20)
    latch = _CUSTODY_SWEEP_LOCK  # a pass releases THIS object, never a later generation's
    if now - last_custody_reap[0] > 600:
        if latch.acquire(blocking=False):
            last_custody_reap[0] = now
            _start_maintenance_thread(latch, "custody-maintenance", _run_periodic_custody_sweep,
                                      (stop_event, latch))
        else:
            _duty_busy(latch, 600)
    latch = _RECONCILE_SWEEP_LOCK
    if now - last_review_reconcile[0] > 300:
        if not latch.acquire(blocking=False):
            _duty_busy(latch, 300)
        elif not _start_maintenance_thread(latch, "reconcile-maintenance", _run_periodic_reconcile_sweep,
                                           (last_review_reconcile, stop_event, latch, on_orphans_healed)):
            last_review_reconcile[0] = time.time()  # a start that failed still waits out a cadence


def _run_periodic_custody_sweep(stop_event: Any = None, latch: Any = None) -> None:
    """The ~600 s custody block, OFF the thread that answers workers (INV-B).

    Skill-payload hashing, the orphaned-process reaper, delegated-run reconciliation
    (gateway handshake, custody replays, registration retirement) and the
    settled-terminal cursor cost seconds to minutes — longer than any worker ack
    wait. The caller took ``_CUSTODY_SWEEP_LOCK`` without blocking (busy => skip,
    never queue); this pass releases it in ``finally``. Nothing serializes it
    against assignment, so each step reads its CANDIDATES before the shared live
    set, and the generation is re-read before every mutation."""
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
            # the reap, so a raising reap can never pin it; retry once, only when THIS
            # sweep released a latch, on a short-lived thread as warm_owned_daemon()
            # does. Contract and residual: DEVELOPMENT.md "Process Custody Rule".
            from ouroboros.claudexor_daemon import get_owned_daemon

            if _stop_requested(stop_event):
                return
            if get_owned_daemon().clear_start_failure_latch(cleared_by="supervisor_sweep"):
                threading.Thread(target=_retry_latched_daemon_start,
                                 name="owned-daemon-latch-retry", daemon=True).start()
        except Exception:
            log.debug("Owned daemon latch release failed", exc_info=True)
        # One guard for the steps (a failure still ends the pass), but the row names
        # WHICH died: unnamed at DEBUG, weeks of silent non-reaping looked healthy.
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
            for name in ("reap_orphaned_processes", "reconcile_delegated_runs", "cursor_refresh_settled_terminals"):
                _step_recovered(name)
        except Exception:
            _step_failed(step, "Periodic custody step %s failed")
    finally:
        (latch or _CUSTODY_SWEEP_LOCK).release()


def _retry_latched_daemon_start() -> None:
    """The one retry of a latched owned-daemon start (#844), by the sweep itself, on
    its own short-lived daemon thread and only after this sweep released the latch:
    one ``ensure_owned_gateway`` with ZERO admission and ZERO startup wait, so no
    loop thread ever holds a startup wait — the spawn happens, custody keeps the
    child, and ``daemon_starting`` is the EXPECTED answer (the next ordinary caller
    joins or settles it). Any other typed refusal (a child that died at once has
    re-latched inside the manager) is a warning; nothing is raised into the loop or
    retried again, and a gateway that did open is closed at once."""
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

        # Zero admission wait: a daemon in its recovery-only window waits for the next
        # sweep. Once a stop, restart or panic is in flight the factory is ATTACH-ONLY;
        # the one ensure left to this surface is the latch retry (ARCHITECTURE §9).
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: (read_owned_gateway() if _stop_requested(stop_event)
                                     else ensure_owned_gateway(admission_wait_sec=0)),
            recoverable_task_ids=recoverable_task_ids(DATA_DIR) | continued,
        )
        if outcomes:
            log.info("Delegated-run reconciliation handled %d orphan(s): %s", len(outcomes), outcomes)
            # A run settled here may belong to a task whose stored terminal result
            # carries an unreconciled disclosure that would lie forever (nanny-leaf
            # S1). Audit-only refresh; never cancels.
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


def _startup_tree_exclusions(recovery_report: dict | None) -> set[str] | None:
    """Resolve protected logical roots and retry occupants; unknown keeps all trees."""
    from ouroboros.task_custody import own_child_drives
    from ouroboros.task_results import list_task_results, load_task_result

    if recovery_report is None or recovery_report.get("errors"):
        return None
    pending = set(recovery_report.get("protected") or []) | set(recovery_report.get("unresolved") or [])
    protected = set()
    try:
        while pending:
            task_id = pending.pop()
            if task_id in protected:
                continue
            protected.add(task_id)
            canonical = load_task_result(DATA_DIR, task_id, strict=True)
            rows = [canonical] if canonical else []
            for child in own_child_drives(DATA_DIR, task_id):
                rows.extend(list_task_results(child, strict=True))
            if not rows:
                return None
            for row in rows:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                for field in ("task_id", "parent_task_id", "root_task_id", "retry_task_id", "superseded_by",
                              "original_task_id", "timeout_retry_from"):
                    linked = str(row.get(field) or metadata.get(field) or "")
                    if linked and linked not in protected:
                        pending.add(linked)
    except (OSError, ValueError, TypeError):
        return None
    return protected


def _startup_prune_sweeps(*, preserve_task_sources: bool = False, recovery_report: dict | None = None) -> None:
    """Startup hygiene: prune stale task drives/trees and orphaned temp files."""
    try:
        from ouroboros.headless import prune_task_trees
        from ouroboros.utils import sweep_stale_temp_files

        exclusions = _startup_tree_exclusions(recovery_report) if preserve_task_sources else set()
        if exclusions is not None:
            prune_task_trees(DATA_DIR, exclude_root_ids=exclusions)
        if preserve_task_sources:
            log.warning("Startup task-source prune deferred: file recovery or ownership is unresolved")
        else:
            # Child and direct drives are settled off the loop thread by the reconcile pass
            # (``_run_drive_custody_pass``): readiness waits on no child-store copy or hash.
            # Startup sweeps only the top-level tmp_scripts fallback (no script can be live
            # yet); the whole-tree walk for atomic temps is owed to the first reconcile pass.
            sweep_stale_temp_files(DATA_DIR, atomic_temps=False)
            _STARTUP_TEMP_SWEEP_OWED[0] = True
    except Exception:
        log.debug("Task tree prune failed", exc_info=True)
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
            # Observability blobs are never deleted and never counted here: a startup census
            # was 193k stat() calls that changed nothing (TZ-1 A).
            from ouroboros.tools.services import prune_service_logs

            _prune_event("runtime_artifact_prune", ("deleted_dirs", "deleted_files", "errors"),
                         services=prune_service_logs(DATA_DIR))
        except Exception:
            log.debug("Runtime artifact prune failed", exc_info=True)


def _cursor_refresh_settled_terminals(live_task_ids: Any = None) -> None:
    """Cursor-driven pass: runs settled OUTSIDE a generation's reconcile outcomes
    (terminal-boundary settlements, earlier generations) never reappear in the
    orphan sweep, so their stored evidence would stay stale forever. Bounded to
    newly appended custody rows per tick; reads the same live-owner source as both
    custody surfaces (a still-billing owner defers the heal). At BOOT it runs AFTER
    the D1a backfill (``_startup_custody_sweep``), so a same-generation heal keeps
    its pinned ``boot_backfill`` attribution without a second write."""
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
    """C1 delegated execution snapshots: GC cross-checked against custody. A snapshot
    stays while its run is open/undisposed OR a pending invocation names it; the rest
    (disposed, closed, refused) is torn down with its pinned baseline ref. Fail-soft
    like every startup prune step, so the startup sequence never dies on a GC error.

    FAIL-CLOSED on an unreadable custody log (CR1-1): the keep-set replays the
    custody rows and ``_iter_rows`` swallows its own OSError, so an unreadable log
    would replay as "no open runs", empty the keep-set and destroy every live
    snapshot with the child's only copy of its work. GC deletes only over PROVEN
    settled && patch_disposed; an UNKNOWN custody state skips the prune, loudly."""
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


# Memory only: where the last drive-custody pass stopped in each drive layout, so a bounded
# pass continues instead of re-attempting the same first drives (no ledger, no timer).
_DRIVE_PRUNE_CURSOR = {"headless": "", "direct": ""}
# Memory only: startup owes the first reconcile pass ONE whole-tree sweep of orphaned atomic
# temp files (a walk over the data root that no longer delays readiness).
_STARTUP_TEMP_SWEEP_OWED = [False]


def _run_periodic_reconcile_sweep(marker: list, stop_event: Any = None, latch: Any = None,
                                  on_orphans_healed: Any = None) -> None:
    """Off-loop 300 s history-sized heal, then the drive-custody pass (the ONE pending
    child-ref promotion retry, then bounded settlement). The marker is stamped when the pass ENDS, before
    its latch opens; the 20 s cancel cadence never waits on any walk and a slow pass never
    restarts (issue #1230). The generation token is asked before every step and, by the
    mutation owners, again before each publication commits."""
    try:
        _periodic_zombie_reconcile(on_orphans_healed=on_orphans_healed, stop_event=stop_event)
        if _stop_requested(stop_event):
            return
        from ouroboros.review_operation import collect_orphaned_operations_softly

        collect_orphaned_operations_softly(DATA_DIR, stop=lambda: _stop_requested(stop_event))
        if _STARTUP_TEMP_SWEEP_OWED[0]:
            from ouroboros.utils import sweep_stale_temp_files

            _STARTUP_TEMP_SWEEP_OWED[0] = False
            removed = sweep_stale_temp_files(pathlib.Path(DATA_DIR), scripts=False)
            if removed:
                log.info("Removed %d orphaned atomic temp file(s) left by a hard kill", removed)
        if _stop_requested(stop_event):
            return
        _run_drive_custody_pass(stop_event)
        _step_recovered("reconcile_sweep")
    except Exception:
        _step_failed("reconcile_sweep", "Periodic %s failed")
    finally:
        marker[0] = time.time()
        (latch or _RECONCILE_SWEEP_LOCK).release()


def _run_drive_custody_pass(stop_event: Any = None) -> None:
    """The ONE pending child-ref promotion retry (the retry owner, before settlement asks),
    the settled canonical mailboxes the loop thread's seam left (unread inputs to carry),
    then settle terminal child and direct drives past retention (a cancelled subagent's at
    once) through ``task_custody.settle_child_drive`` with the supervisor's probe and
    ownership interlock, at most ``DRIVE_SETTLEMENTS_PER_PASS`` attempts per layout and
    pass, continuing from the previous pass's cursor; crashed-settlement leftovers are
    swept first. Off the loop thread: each settlement may copy and hash a child store.
    A drive kept because its custody is not proven is material evidence in the event."""
    from ouroboros.headless import DRIVE_SETTLEMENTS_PER_PASS, prune_headless_task_drives, prune_task_drives
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.owner_mailbox import sweep_settled_owner_mailboxes
    from ouroboros.task_custody import sweep_custody_leftovers
    from supervisor.queue import task_settlement_interlock, task_settlement_liveness

    def stop() -> bool:
        return _stop_requested(stop_event)

    root = pathlib.Path(DATA_DIR)
    report = retry_pending_child_ref_promotions(root, stop=stop, generation=stop_event) or {}
    if report.get("retried") or report.get("errors"):
        log.info("Child-ref promotion retry: %s", report)
    if stop():
        return
    sweep_custody_leftovers(root)
    # Canonical mailboxes the loop thread's seam left because unread rows carry inputs to
    # verify or copy: carried and unlinked here, off the loop thread.
    mailboxes = sweep_settled_owner_mailboxes(root, stop=stop)
    reports = {}
    for key, prune in (("headless", prune_headless_task_drives), ("direct", prune_task_drives)):
        if stop():
            return
        reports[key] = prune(root, live=task_settlement_liveness, guard=lambda: task_settlement_interlock(stop=stop),
                             stop=stop, budget=DRIVE_SETTLEMENTS_PER_PASS, after=_DRIVE_PRUNE_CURSOR[key])
        _DRIVE_PRUNE_CURSOR[key] = str(reports[key].get("cursor") or "")
    _prune_event("headless_task_drive_prune", ("pruned", "errors", "custody_pending", "removed"),
                 report=reports["headless"], task_drives=reports["direct"], mailboxes=mailboxes)


def _periodic_zombie_reconcile(*, on_orphans_healed: Any = None, stop_event: Any = None) -> None:
    """Heal zombie 'running' records on a supervisor cadence. A worker that died
    mid-review (crash / SIGKILL / manual stop) leaves ``review_job.json`` at running
    forever in headless/no-UI runs, where the boot and ``GET /api/extensions``
    reconciles never fire; the same death leaves ``task_results/<id>.json`` at
    running. Both reconciles are liveness-gated (pid-dead / queue-empty + worker-boot
    evidence), so a live review or task is never touched. Off the loop thread, so
    ``stop_event`` (the generation token) is re-read before every step."""
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs
        reconcile_stale_review_jobs(DATA_DIR)
    except Exception:
        log.debug("Periodic skill review-job reconcile failed", exc_info=True)
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        expired_quizzes: list = []
        healed = reconcile_orphaned_running_tasks(
            DATA_DIR, expired_quizzes=expired_quizzes,
            write_guard=lambda task_id: orphan_reconcile_write_guard(task_id, stop_event=stop_event),
        )
        _publish_expired_quiz_frames(expired_quizzes)
        if healed and callable(on_orphans_healed):
            on_orphans_healed(int(healed))
    except Exception:
        log.debug("Periodic orphaned running-task reconcile failed", exc_info=True)
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.projects_registry import reconcile_projects
        reconcile_projects(DATA_DIR)
    except Exception:
        log.debug("Project registry reconcile failed", exc_info=True)
    if not _stop_requested(stop_event):
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
    """Tell already-rendered cards that a healed terminal expired their question:
    the same frame the task-done seam sends, from the supervisor-side healer that
    writes terminals off that seam (the packaged shell, mini app and phone have no
    reload affordance). Fail-soft: the durable projection is right without it."""
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
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.headless import (
        HEADLESS_TASKS_DIR,
        TASK_DRIVES_DIR,
        prepare_terminal_task_files,
        terminal_task_files_ready,
    )
    from ouroboros.observability import _has_pending_ref_promotion
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
                if ready:
                    continue  # History is already owed to the off-loop retry owner.
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
    """File recovery precedes the orphan reconcile and the caller's actual prune.

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
            write_guard=orphan_reconcile_write_guard,
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
