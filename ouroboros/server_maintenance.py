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

    if queue.INITIALIZED:
        queue.task_settlement_liveness(task_id)  # Prepare durable reads before the mutation interlock.
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
    """Keep slow hashing, reaping and reconciliation off the worker-ack thread (INV-B).
    The caller's nonblocking latch skips busy passes; ``finally`` releases that latch.
    Assignment stays concurrent: read candidates before live sets and recheck the
    generation before each mutation. Spawn only after releasing a failed-start latch,
    at most once per sweep; healthy installs never spawn or wait here.
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
            from ouroboros.delegate_custody_current import current_reads
            with current_reads(DATA_DIR):
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
    owner who never opens the Logs panel does not see. The dropped sets come from that same
    read seam (``config.retired_key_sets_seen``), the sentence is the one the log line uses
    (``settings_defaults.retired_setting_keys_notice``), and the dedupe is durable:
    ``state.json:retired_settings_notified`` keyed by the exact retired-key set, so a
    restart or a supervisor revival never repeats it. Nothing is sent — and nothing
    marked — while no owner chat is bound: the notice waits for the first boot that has
    somewhere to deliver it. Which reviewers run is the review-pool migration's report
    (``_startup_review_pool_notice``), not this one's.
    """
    try:
        from ouroboros.config import retired_key_sets_seen
        from ouroboros.settings_defaults import retired_setting_keys_notice
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        if not owner_chat:
            return
        notified = state.get("retired_settings_notified")
        notified = notified if isinstance(notified, dict) else {}
        for dropped in retired_key_sets_seen():
            marker = ",".join(dropped)
            if marker in notified:
                continue
            send_with_budget(
                owner_chat,
                "⚙️ Settings: " + retired_setting_keys_notice(dropped),
                role="system", system_type="retired_settings_notice",
            )
            update_state(lambda st, key=marker: _mark_retired_settings_notified(st, key))
    except Exception:
        log.debug("retired settings owner notice failed", exc_info=True)


def _mark_retired_settings_notified(st: dict, marker: str) -> None:
    """Stamp ``marker`` in the durable owner-notice ledger ``state.json:retired_settings_notified``."""
    seen = st.get("retired_settings_notified")
    seen = dict(seen) if isinstance(seen, dict) else {}
    seen[marker] = utc_now_iso()
    st["retired_settings_notified"] = seen


def environment_retired_review_keys(environ: Dict[str, str] | None = None) -> tuple[str, ...]:
    """The retired review keys the PROCESS ENVIRONMENT carries with a value: the review lanes
    (``OUROBOROS_REVIEWER_SLOTS``), the per-surface review efforts / deep-review model and the
    older reviewer comma-lists. No release reads them from the environment any more — the
    environment merge (``config.load_settings``) walks ``SETTINGS_DEFAULTS``, which retired
    them, and the pool migration reads the DOCUMENT — so an operator who still exports them
    (a Docker/Linux unit, a Colab cell) configures nothing (D1-V03): the install runs its own
    pool (the catalog in the document, else the factory rows)."""
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS, REVIEW_POOL_MIGRATED_SETTING_KEYS

    env = os.environ if environ is None else environ
    return tuple(key for key in REVIEW_POOL_MIGRATED_SETTING_KEYS + RETIRED_COMMA_LIST_SETTING_KEYS
                 if str(env.get(key) or "").strip())


def environment_review_notice(keys: tuple[str, ...]) -> str:
    """The ONE sentence (log line and owner chat alike) for review keys found in the environment."""
    plural = len(keys) != 1
    return (
        f"⚙️ Settings: the process environment sets {', '.join(keys)}, which {'are' if plural else 'is'} no longer "
        "read: the review lanes and the reviewer lists became the review pool — the rows of the subagent "
        "catalog marked “Reviewer” (OUROBOROS_SUBAGENTS, Settings → Agents). "
        f"{'Those values were' if plural else 'That value was'} not applied; the install's own pool runs. "
        "To configure the pool from the environment, set OUROBOROS_SUBAGENTS to a catalog with marked rows."
    )


def _startup_environment_review_notice() -> None:
    """Say ONCE, loudly, that review keys set in the environment are not read (D1-V03 / VD3-03):
    a WARNING on the server log at every boot, and the same sentence in the owner chat once per
    exact key set (durable: ``state.json:retired_settings_notified`` under an ``environment:``
    marker, the retired-settings notice's own ledger). Nothing is read from the environment
    into the pool here or anywhere: the fact is loud, the behaviour unchanged."""
    keys = environment_retired_review_keys()
    if not keys:
        return
    text = environment_review_notice(keys)
    log.warning(text)
    try:
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        if not owner_chat:
            return
        marker = "environment:" + ",".join(keys)
        notified = state.get("retired_settings_notified")
        if marker in (notified if isinstance(notified, dict) else {}):
            return
        send_with_budget(owner_chat, text, role="system", system_type="retired_settings_notice")
        update_state(lambda st: _mark_retired_settings_notified(st, marker))
    except Exception:
        log.debug("environment review keys owner notice failed", exc_info=True)


REVIEW_POOL_MIGRATION_STATE_KEY = "review_pool_migrations"  # ``review_pool_receipts.STATE_KEY``
REVIEW_POOL_NOTICE_TYPE = "review_pool_migration_notice"


def review_pool_migration_records(state: dict | None = None) -> dict:
    """The durable per-document migration records (``state.json:review_pool_migrations``):
    ``input_sha256 -> {ts, snapshot, trigger, outcome, error, reported}`` — the ledger
    ``review_pool_receipts`` keeps; this is the name the ``## Review`` block reads."""
    from ouroboros.review_pool_receipts import migration_records

    return migration_records(state)


def review_pool_migration_payload(settings: dict, *, document: dict | None = None) -> dict | None:
    """The review-pool payload's ``migration`` fact for this data root (``review_pool_receipts.migration_payload`` over
    the ledger plus the snapshot files no record names yet). ``settings`` is what RUNS; ``document`` (default ``settings``)
    is the settings document the receipt is judged against — the GET handler passes the document on disk (VD3-06)."""
    from ouroboros.review_pool_receipts import migration_payload

    return migration_payload(settings if document is None else document, DATA_DIR, review_pool_migration_records(), running=settings)


def _startup_review_pool_notice(settings: dict) -> None:
    """Tell the OWNER once about every review-lane -> review-pool migration recorded for this
    data root, from the DURABLE receipts — never from this process's memory alone.

    The migration itself is pure and runs at the read seam (``config.normalize_settings_raw``
    -> ``review_pool_migration.apply_at_read_seam``) in whichever process reads an old document;
    the process that SAVES the migrated document writes its receipts before that write
    (``review_pool_receipts.persist_write_receipts`` from the persistence prologue and the Colab
    writer): the snapshot ``state/review_migrations/<ts>-slots-to-pool.json`` and the
    ``state.json`` record. This boot step first gives receipts, by the writer's rule, only to
    the migrations deciding the document on disk now (``persist_boot_receipts``: the N-1 document
    the boot read before any save; with no file, the defaults ``settings`` carries; never the
    factory rows of defaults read before the wizard saved its own catalog), then reconciles a
    record for every snapshot another process left without one (a Colab kernel, the launcher
    menu, the UI before this supervisor generation), and sends ONE English owner-chat message
    (``review_pool_migration.owner_message``) per record whose ``reported`` is still unset,
    once an owner chat is bound. A migration that could not finish is reported the same way
    (its snapshot carries the error; the lane keys stay in the document for the owner's catalog
    save). A no-op outcome (the catalog was already a pool) leaves no receipt: nothing changed.
    Each unreported record is judged against the document AS READ (``review_pool_receipts.document_as_read``: the
    file as the seam leaves it, no environment); ``settings`` is what RUNS (the environment merged over it). A record
    whose outcome no longer decides that document (the factory rows receipted at a file-less start before the wizard
    saved its own catalog) is closed as history without a message (``review_pool_receipts.close_as_history``), never
    delivered as if the owner's catalog were the environment's. Where the environment's catalog overrides the rows the
    seam minted for a document without review settings of its own (``environment_overridable_keys``), the message names
    THAT pool — loudly empty when none of its rows is marked. A never-configured document whose process environment
    still carries the retired review keys is told those keys are no longer read (``environment_retired_review_keys``).
    """
    try:
        from ouroboros import config, review_pool_receipts as receipts
        from ouroboros.review_pool_migration import owner_message
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        receipts.persist_boot_receipts(DATA_DIR, config.SETTINGS_PATH, settings)
        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        records = receipts.reconcile_records(DATA_DIR, state, update_state)
        if not owner_chat:
            return
        document = receipts.document_as_read(config.SETTINGS_PATH, settings)
        for digest, record in sorted(records.items(), key=lambda item: str(item[1].get("ts") or "")):
            if record.get("reported"):
                continue
            snapshot = receipts.load_snapshot(DATA_DIR, record)
            if snapshot is None:
                log.warning("review pool migration snapshot missing, owner not told: %s", record.get("snapshot"))
                continue
            outcome = receipts.outcome_from_snapshot(snapshot)
            snapshot_path = str(record.get("snapshot") or "")
            if receipts.close_as_history(outcome, document, update_state, digest, record):
                continue
            in_force = receipts.environment_catalog_in_force(outcome, document, settings)
            text = (_environment_pool_message(snapshot_path, in_force) if in_force is not None else
                    owner_message(outcome, snapshot_path, environment_retired_keys=environment_retired_review_keys()))
            if not text:
                continue
            send_with_budget(owner_chat, text, role="system", system_type=REVIEW_POOL_NOTICE_TYPE)
            receipts.mark_reported(update_state, digest, record)
    except Exception:
        log.debug("review pool migration notice failed", exc_info=True)


def _environment_pool_message(snapshot_path: str, catalog_text: str) -> str:
    """The ONE owner-chat message when the catalog the environment carries, not the factory
    rows the migration prepared, is the review pool: it names what runs, and says loudly
    when that is nothing (``pool_empty`` — a configured fact, never a default panel)."""
    from ouroboros import reviewer_slot_config as rs
    from ouroboros.review_pool_migration import ROLLBACK_SENTENCE

    where = (f"Snapshot: {snapshot_path}. {ROLLBACK_SENTENCE}" if snapshot_path
             else "No snapshot could be written, so there is no rollback source.")
    head = ("⚙️ Review pool: the subagent catalog set in the environment (OUROBOROS_SUBAGENTS) is in force. "
            "This document had no review settings of its own (no authored review lanes, no saved subagent "
            "catalog), so the factory reviewer rows were prepared for it — but a catalog the environment "
            "carries is explicit configuration and runs instead; the factory rows do not.")
    state = rs.review_pool_state(catalog_text)
    if state["state"] == "error":
        body = f"That catalog cannot be read ({state['error']}): no review runs until it is repaired."
    elif state["state"] == "empty":
        body = ("None of its rows is marked “Reviewer”, so the review pool is empty (pool_empty): reviews will not "
                "run and will report not performed.")
    else:
        rows = rs.review_pool_rows({"OUROBOROS_SUBAGENTS": catalog_text})
        body = (f"{len(rows)} reviewer rows, {len({row.target_id for row in rows})} distinct models: "
                + "; ".join(f"{row.slot_id} ({row.target_id})" for row in rows) + ".")
    return "\n".join([head, body, f"{where} Adjust in Settings → Agents or in the environment."])


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


_STARTUP_PRUNES_OWED = [False]
_STARTUP_SOURCE_PRUNES_OWED = [False]
_STARTUP_TREES_OWED = [False]
_STARTUP_RECOVERY_GAPS = [False]


def _startup_prune_sweeps(*, preserve_task_sources=False, recovery_report=None):
    """Owe housekeeping off-loop; fallback scripts can only be reaped before intake."""
    _STARTUP_PRUNES_OWED[0] = _STARTUP_TREES_OWED[0] = _STARTUP_SOURCE_PRUNES_OWED[0] = True
    _STARTUP_RECOVERY_GAPS[0] = bool(preserve_task_sources and (
        recovery_report is None or recovery_report.get("errors") or "*" in recovery_report.get("unresolved", [])))
    if not preserve_task_sources:
        from ouroboros.utils import sweep_stale_temp_files
        sweep_stale_temp_files(DATA_DIR, atomic_temps=False)
        _STARTUP_TEMP_SWEEP_OWED[0] = True


def _run_deferred_startup_prunes():
    """Retry owed housekeeping; clear each duty only after success.
    Tree/source pruning waits for gap-free recovery and complete recovery links;
    mailbox and service-log duties retry independently, preserving protected trees."""
    from ouroboros.startup_task_files import startup_tree_exclusions
    from ouroboros.headless import prune_task_trees
    if not _STARTUP_RECOVERY_GAPS[0] and (_STARTUP_TREES_OWED[0] or _STARTUP_SOURCE_PRUNES_OWED[0]):
        exclusions = startup_tree_exclusions(DATA_DIR)
        if exclusions is not None:
            if _STARTUP_TREES_OWED[0]:
                prune_task_trees(DATA_DIR, exclude_root_ids=exclusions)
                _STARTUP_TREES_OWED[0] = False
                _STARTUP_TEMP_SWEEP_OWED[0] = True
            if _STARTUP_SOURCE_PRUNES_OWED[0]:
                try:
                    from ouroboros.owner_mailbox import sweep_settled_owner_mailboxes
                    from ouroboros.tools.services import prune_service_logs
                    _prune_event("owner_mailbox_sweep", ("removed",), report=sweep_settled_owner_mailboxes(DATA_DIR))
                    _prune_event("runtime_artifact_prune", ("deleted_dirs", "deleted_files", "errors"),
                                 services=prune_service_logs(DATA_DIR))
                    _STARTUP_SOURCE_PRUNES_OWED[0] = False
                except Exception:
                    log.debug("Source housekeeping remains owed", exc_info=True)
    if not _STARTUP_PRUNES_OWED[0]:
        return
    _STARTUP_PRUNES_OWED[0] = False
    _startup_worktree_prune()
    from ouroboros.delegate_custody_current import current_reads
    with current_reads(DATA_DIR):
        _prune_delegated_snapshots()
    from ouroboros.delegate_state_sweep import sweep_settled_delegate_state
    _prune_event("delegate_state_sweep", ("removed", "errors", "skipped"),
                 report=sweep_settled_delegate_state(DATA_DIR))
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
        # CPL4-C21 (owner 6A): agent screenshots/views follow GC retention;
        # owner attachments in the uploads/ root are never touched.
        _prune_event("agent_media_prune", ("removed", "skipped", "errors"),
                     report=prune_agent_media_uploads(DATA_DIR))
    except Exception:
        log.debug("Agent media prune failed", exc_info=True)


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
    from ouroboros.delegate_custody_current import current_reads
    with current_reads(DATA_DIR):
        _reconcile_delegated_runs(_live_task_ids)
    try:
        # D1a boot backfill, ONCE per generation and AFTER the orphan reconcile
        # (so this generation's settlements are already visible to the audit):
        # a run settled in a PREVIOUS generation never appears in any current
        # pass's outcomes, so the sweep-side refresh above can never reach its
        # task's stored disclosure — the backfill joins from the stored terminal
        # results instead and heals every generation-crossing stale row.
        from ouroboros.delegate_terminal import backfill_terminal_reconciliations

        with current_reads(DATA_DIR):
            refreshed = backfill_terminal_reconciliations(DATA_DIR)
        if refreshed:
            log.info("Boot custody backfill refreshed %d stored disclosure(s): %s",
                     len(refreshed), refreshed)
    except Exception:
        log.debug("Boot custody-disclosure backfill failed", exc_info=True)
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
    """C1 delegated execution snapshots: GC cross-checked against custody. A snapshot
    stays while its run is open/undisposed OR a pending invocation names it; the rest
    (disposed, closed, refused) is torn down with its pinned baseline ref. Fail-soft
    like every startup prune step, so the startup sequence never dies on a GC error.

    FAIL-CLOSED on an unreadable custody log (CR1-1): the keep-set replays the
    custody rows and ``_iter_rows`` swallows its own OSError, so an unreadable log
    would replay as "no open runs", empty the keep-set and destroy every live
    snapshot with the child's only copy of its work. GC deletes only over PROVEN
    settled && patch_disposed; an UNKNOWN custody state skips the prune, loudly. So does an
    owed obligations rebuild: a custody row that landed while its set could not be updated
    is missing from the set until the next start merges it."""
    try:
        from ouroboros import delegate_custody as _delegate_custody
        from ouroboros import subagent_worktrees as _snap_worktrees
        from ouroboros.obligations import REBUILD_MARK
        from supervisor.state import append_jsonl

        reason = ("custody_log_unreadable" if _delegate_custody.custody_log_unreadable(DATA_DIR)
                  else "obligations_rebuild_owed" if (DATA_DIR / "state" / "obligations" / REBUILD_MARK).exists()
                  else "")
        if reason:
            log.warning("Delegated snapshot prune SKIPPED (%s): open snapshots are unknowable (fail-closed)", reason)
            if not append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune_skipped",
                "reason": reason,
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
        if not _stop_requested(stop_event) and (
                _STARTUP_PRUNES_OWED[0] or _STARTUP_TREES_OWED[0] or _STARTUP_SOURCE_PRUNES_OWED[0]):
            try:
                _run_deferred_startup_prunes()
            except Exception:
                log.warning("Startup housekeeping remains deferred", exc_info=True)
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
    until the next boot reconcile (``GET /api/extensions`` is a passive read and
    heals nothing); the same death leaves ``task_results/<id>.json`` at
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
        from ouroboros.startup_migrations import migrate_cancel_latches

        migrated = migrate_cancel_latches(drive_root)
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
    from ouroboros.startup_task_files import recover_terminal_task_files
    return recover_terminal_task_files(drive_root, protected)


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
        report["errors"].append("orphan_recovery_failed")
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)
    try:
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        recover_pending_root_post_task_synthesis(drive_root, repo_dir, exclude_task_ids=excluded)
    except Exception:
        report["errors"].append("synthesis_recovery_failed")
        log.warning("Root post-task synthesis recovery at startup failed", exc_info=True)
    from ouroboros.obligations import members
    try:
        report["unresolved"].extend(facts.get("task_id", identity) for identity, facts in members(drive_root, "unknowns").items())
    except Exception:
        report["unresolved"].append("*")
        report["errors"].append("unknown_obligations_unavailable")
        log.warning("Startup repair gaps unavailable; destructive pruning remains deferred", exc_info=True)
    return report
