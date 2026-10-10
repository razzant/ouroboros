"""Terminal maintenance reads open attempts and revisioned projection debt.

Recovery custody is separate from money. Dirty owners are acknowledged only
through the revision whose cost was projected; a later receipt retains debt.
Each pass observes what stays unresolved, per subject (open attempts, cost
projections), as one id-to-reason map; only a change of that map is published,
and only a reason new to the subject logs a representative traceback.
"""
from __future__ import annotations

import errno
import logging
import json
import os
import pathlib
import re
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

log = logging.getLogger(__name__)
_PROBING: set[tuple[str, str]] = set()
_PROBE_LOCK = threading.Lock()
_UNRESOLVED_LOCK = threading.Lock()
# (root, subject) -> the last observed id->reason map, the last published
# signature, the last publication failure and (attempts) the last gateway-close
# failure; current entries only.
_LAST_UNRESOLVED: dict[tuple[str, str], dict] = {}
ATTEMPTS, COST_PROJECTIONS = "attempt", "cost_projection"
_ID_FIELDS = {ATTEMPTS: "attempt_ids", COST_PROJECTIONS: "task_ids"}
_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")


class _RecoveryGateway:
    """Attribute a gone response to the exact operation GET, not discovery/result GETs."""
    def __init__(self, gateway):
        self.gateway = gateway

    def __getattr__(self, name):
        return getattr(self.gateway, name)

    def get_model_operation(self, operation_id, **kwargs):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        try:
            return self.gateway.get_model_operation(operation_id, **kwargs)
        except ClaudexorUnavailable as exc:
            if exc.status_code in {404, 410}:
                raise ClaudexorUnavailable("recovery_operation_gone", exc.code, status_code=exc.status_code) from exc
            raise


def _recovery_fact(root, row, outcome, basis, *, revision=None, due_at=None):
    from ouroboros import usage_store
    from ouroboros.utils import utc_now_iso

    fact = {"outcome": outcome, "at": utc_now_iso(), "basis": basis}
    if due_at is not None:
        fact["due_at"] = due_at.isoformat()
    with usage_store.hold(root) as txn:
        txn.record_recovery(row["attempt_id"], fact,
                            expected_revision=row["revision"] if revision is None else revision)


def _retry_after(value):
    """Only a received transport instruction supplies a due time."""
    if not value:
        return None
    try:
        return datetime.now(timezone.utc) + timedelta(seconds=max(0, float(value)))
    except (ValueError, OverflowError):
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            return None


def _unresolved(events, row, basis):
    events[str(row['attempt_id'])] = str(basis)


def _reason(exc):
    """A stable failure reason: class plus errno or code, never a per-id message or path."""
    code = getattr(exc, "errno", None)
    code = errno.errorcode.get(code, code) if isinstance(code, int) else getattr(exc, "code", None)
    if code is None and _CODE.fullmatch(str(exc)):
        code = str(exc)  # A typed refusal such as owner_pause_authority_missing.
    return f"{type(exc).__name__}:{code}" if code not in (None, "") else type(exc).__name__


def _subject_state(root, subject):
    """The subject's diagnostic memory; the caller holds ``_UNRESOLVED_LOCK``."""
    return _LAST_UNRESOLVED.setdefault((str(root), subject),
                                       {"observed": {}, "published": None, "fault": None, "close": None})


def _close_gateway(root, gateway):
    """A failed close of the pass's borrowed gateway retains nothing; it is logged once per change."""
    try:
        gateway.close()
        exc = reason = None
    except Exception as error:
        exc, reason = error, _reason(error)
    with _UNRESOLVED_LOCK:
        state = _subject_state(root, ATTEMPTS)
        repeated, state["close"] = state["close"] == reason, reason
    if reason and not repeated:
        log.debug("Usage recovery gateway close failed (%s)", reason, exc_info=exc)


def _publish_unresolved(root, observations, *, subject=ATTEMPTS, faults=None, keep=(), complete=True):
    """Publish one changed summary, without an interprocess lock or polling wait.

    ``observations`` maps each id this pass found unresolved to its reason;
    an id in ``keep`` (still a candidate whose outcome this pass did not prove)
    keeps its previous reason, so a skipped or partial pass proves no recovery. The
    observed map is replaced every pass and holds only current ids. A reason
    absent from the previous map logs one representative traceback from
    ``faults``; an unchanged pass logs nothing at any level. Publication is
    separate: an incomplete pass never publishes, and a failed write retries
    on the next pass with one warning per failure kind, so neither an
    unwritable log nor a stable error repeats a stack. Outstanding custody
    stays in the store; these rows are observations, never authority.
    """
    from ouroboros.utils import _write_fd_fully, assert_test_data_path, utc_now_iso
    from supervisor.message_bus import try_get_bridge

    with _UNRESOLVED_LOCK:
        state = _subject_state(root, subject)
        previous = state["observed"]
        current = {key: previous[key] for key in keep if key in previous}
        current.update(observations)
        state["observed"] = current
        known = set(previous.values())
        fresh = {}
        for key, reason in sorted(current.items()):
            if reason not in known and (faults or {}).get(key) is not None:
                fresh.setdefault(reason, (key, faults[key]))
    for reason, (example, exc) in fresh.items():
        count = sum(1 for value in current.values() if value == reason)
        log.warning("Terminal maintenance %s: %d unresolved with %s (e.g. %s)", subject, count, reason, example,
                    exc_info=exc)
    if not complete:
        return
    ids = tuple(sorted(current))
    by_basis = dict(sorted(Counter(current.values()).items()))
    signature = tuple(sorted(current.items()))
    path = root / "logs" / "supervisor.jsonl"
    assert_test_data_path(path)
    try:
        with _UNRESOLVED_LOCK:
            if state["published"] == signature or state["published"] is None and not ids:
                return
            # 50 is the hard identifier-list size limit of one log row. Compare
            # the FULL id->reason map above so changes beyond this display cap still emit.
            shown = ids[:50]
            event = {"type": "duty_unresolved", "duty": "terminal-maintenance", "subject": subject,
                     "ts": utc_now_iso(), "count": len(ids), "by_basis": by_basis,
                     _ID_FIELDS[subject]: list(shown), "reasons": {key: current[key] for key in shown}}
            path.parent.mkdir(parents=True, exist_ok=True)
            data = (json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8')
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                _write_fd_fully(fd, data, path)
            finally:
                os.close(fd)
            state["published"], state["fault"] = signature, None
        bridge = try_get_bridge()
        if bridge is not None:
            bridge.push_log(event)
    except Exception as exc:
        with _UNRESOLVED_LOCK:
            repeated, state["fault"] = state["fault"] == _reason(exc), _reason(exc)
        if not repeated:
            log.warning("Unresolved %s observations could not be published", subject, exc_info=True)


def imported_projection_debt(root, buckets, *, degraded=False):
    """One-time import only: seed missing/different stored task and root costs.

    The importer already holds every summary; no published store is opened.
    Unreadable, foreign and live results retain debt for the ordinary duty.
    """
    from ouroboros._usage_rows import Bucket, _with_integrity
    from ouroboros.task_results import load_task_result, validate_task_id
    owners = {key for scope, key in buckets if scope in {"task", "root"} and key}
    for owner in sorted(owners):
        try:
            validate_task_id(owner)
        except ValueError:
            continue
        try:
            current = load_task_result(root, owner, strict=True) or {}
            if not current:
                yield owner
                continue
            from supervisor.events_task_done import _authoritative_terminal_cost

            logical = str(current.get("root_task_id") or owner)
            breakdown = {"integrity_degraded": degraded,
                         "by_task": {owner: _with_integrity(buckets.get(("task", owner), Bucket()).render_breakdown(), degraded)},
                         "by_root": {logical: _with_integrity(buckets.get(("root", logical), Bucket()).render_breakdown(), degraded)}}
            fields = _authoritative_terminal_cost(owner, current, current, {}, root,
                                                  breakdown=breakdown, canonical_only=True)
            if current and all(current.get(key) == value for key, value in fields.items()):
                continue
        except Exception:
            pass  # Missing/unreadable authority keeps a named projection obligation.
        yield owner


def _refresh_costs(root: pathlib.Path, reads) -> None:
    from ouroboros import usage_store
    from ouroboros.task_results import task_result_path, validate_task_id
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from supervisor.events_task_done import _refresh_terminal_task_cost
    from supervisor.queue import task_has_live_ownership

    def targets(task_id, current):
        yield task_id, current
        if current.get("superseded_by") or current.get("retry_task_id"):
            from supervisor import queue
            from supervisor.queue_transitions import _live_retry_target_locked
            from supervisor.task_ownership import prepare_retry_chain

            prepare_retry_chain(queue, task_id, reads.load)
            with queue._queue_lock:
                leaf, settled = _live_retry_target_locked(queue, task_id, results=reads)
            if leaf != task_id and settled:
                yield leaf, reads.load(leaf)

    with usage_store.read(root) as txn:
        owners = txn.dirty_owners()
    acknowledged, pending, failures, faults = [], set(), {}, {}
    for owner, revision in owners:
        try:
            task_id = validate_task_id(owner)
        except ValueError:
            continue  # System accounting scopes own no task result.
        pending.add(task_id)
        try:
            # Presence precedes the ownership read: an absent body keeps this
            # exact revision without the fence's quarantine traversal, and the
            # next pass projects a body copied back after the last ledger write.
            task_result_path(root, task_id, create=False).stat()
        except FileNotFoundError:
            failures[task_id] = "result_absent"
            continue
        except OSError:
            pass  # Only absence skips the read; the ownership read reports the rest.
        try:
            current = reads.load(task_id)
            if (not current or current.get("status") not in SETTLED_STATUSES
                    or post_task_synthesis_is_open((current.get("root_phase_checkpoint") or {}).get("post_task_synthesis"))
                    or task_has_live_ownership(task_id, ownership=reads)):
                continue
            projection_targets = list(targets(task_id, current))
            if all(not task_has_live_ownership(tid, ownership=reads)
                   and _refresh_terminal_task_cost(root, tid, current=row)
                   for tid, row in projection_targets):
                acknowledged.append((owner, revision))
                pending.discard(task_id)
        except Exception as exc:
            failures[task_id], faults[task_id] = _reason(exc), exc
    if acknowledged:
        # One short acknowledgement transaction after all result I/O. A crash
        # before it merely repeats successful projections; newer receipts win.
        with usage_store.hold(root) as txn:
            for owner, revision in acknowledged:
                txn.ack_dirty_owner(owner, revision)
    _publish_unresolved(root, failures, subject=COST_PROJECTIONS, faults=faults, keep=pending)


def reconcile_abandoned_usage(drive_root: pathlib.Path) -> None:
    """Close unowned terminal-task attempts; price and remote custody stay separate."""
    from ouroboros import usage_accounting as usage
    from ouroboros._usage_rows import REVIEW_CUSTODY_KEYS
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm_claudexor import recover_model_attempt
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from supervisor.task_ownership import TaskOwnershipRead
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.transport_custody import ProviderNotDispatched, release_pre_dispatch_attempt
    from ouroboros.usage_ledger import is_abandoned_settlement
    from supervisor.queue import task_has_live_ownership

    from ouroboros import usage_store

    root = pathlib.Path(drive_root).resolve()
    with usage_store.read(root) as txn:
        rows = txn.open_attempts()
    reads, eligible, observations, faults = TaskOwnershipRead(root), {}, {}, {}
    # An open attempt keeps its last reason unless this pass observes a new one or
    # proves an outcome: a recovered receipt, a release, a terminal settlement or
    # terminal remote custody. Skipped, live or lost-race outcomes prove nothing.
    unproven = {str(row["attempt_id"]) for row in rows}
    probes_complete = True
    gateway, gateway_unavailable = None, False

    def eligible_task(task_id):
        if not task_id:
            return False
        if task_id not in eligible:
            try:
                task = reads.load(task_id)
                checkpoint = task.get("root_phase_checkpoint") or {}
                eligible[task_id] = (
                    task.get("status") in SETTLED_STATUSES
                    and not task_has_live_ownership(task_id, ownership=reads)
                    and not post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
                )
            except Exception:
                eligible[task_id] = False  # Unreadable ownership permits neither duty.
        return eligible[task_id]

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

    def probe_gateway(row):
        from ouroboros.deadline_utils import parse_deadline_ts

        fact = row.get("recovery") or {}
        if fact.get("outcome") in {"operation_gone", "terminal"}:
            raise ClaudexorUnavailable("recovery_terminal", "Remote custody already terminal")
        due = parse_deadline_ts(fact.get("due_at")) if fact.get("basis") else None
        if due is not None and due > datetime.now(timezone.utc):
            raise ClaudexorUnavailable("recovery_not_due", "Provider Retry-After has not elapsed")
        return _RecoveryGateway(borrowed_gateway())

    try:
        for row in rows:
            kind = row.get("kind", "attempt")
            if kind not in {"attempt", "usage_baseline_group"} or any(row.get(key) for key in REVIEW_CUSTODY_KEYS):
                continue
            task_id = str(row.get("task_id") or "")
            remote = row.get("provider") == "claudexor"
            abandoned = is_abandoned_settlement(row)
            if (kind != "attempt"
                or (row.get("state") not in {"reserved", "dispatched", "unresolved"}
                    and not (remote and abandoned))):
                continue
            if not eligible_task(task_id):
                continue
            probe_key = str(root), row["attempt_id"]
            with _PROBE_LOCK:
                if probe_key in _PROBING:
                    probes_complete = False  # A partial observation cannot declare the set empty.
                    continue
                _PROBING.add(probe_key)
            attempt_id = str(row["attempt_id"])
            reservation = usage.AttemptReservation(
                str(row["attempt_id"]), root, str(row.get("model") or ""),
                str(row.get("provider") or ""), row.get("reservation_upper_bound_usd"),
                str(row.get("processing_preference") or ""), str(row.get("submitted_processing_mode") or ""),
                row.get("processing_basis"),
            )
            try:
                recovered = None
                if remote and row.get("state") != "reserved":
                    # Offline receipts still win, even after a terminal observation
                    # or during Retry-After. Only the HTTP probe is suppressed.
                    recovered = recover_model_attempt(root, row, gateway_factory=lambda: probe_gateway(row))
                    if recovered is None:
                        _unresolved(observations, row, "receipt_or_terminal_custody_unavailable")
                        continue
                if task_has_live_ownership(task_id, ownership=reads):
                    continue
                disposition, reported, cost, final = recovered or ("abandoned", {}, None, False)
                if disposition == "settled":
                    usage.settle_attempt(reservation, reported, cost_usd=cost, cost_final=final,
                                         expected_revision=row["revision"])
                    unproven.discard(attempt_id)  # The receipt is the proof, whichever settlement won.
                elif disposition == "released":
                    if not release_pre_dispatch_attempt(reservation, ProviderNotDispatched("recovered model operation never started"),
                                                        expected_revision=row["revision"]):
                        continue
                    unproven.discard(attempt_id)
                elif disposition == "abandoned":
                    if abandoned:
                        if remote and (row.get("recovery") or {}).get("outcome") not in {"terminal", "operation_gone"}:
                            _recovery_fact(root, row, "terminal", "recover_model_attempt:abandoned")
                        unproven.discard(attempt_id)
                        continue
                    state = usage.terminalize_abandoned_attempt(
                        reservation, reason="owner_task_terminal", expected_revision=row.get("revision"))
                    if state not in {"settled", "released"}:
                        continue
                    if remote and recovered is not None:
                        _recovery_fact(root, row, "terminal", "recover_model_attempt:abandoned",
                                       revision=row["revision"] + 1)
                    unproven.discard(attempt_id)
                else:
                    continue
            except ClaudexorUnavailable as exc:
                if exc.code == "recovery_terminal":
                    unproven.discard(attempt_id)  # Terminal remote custody is already a durable fact.
                    continue
                if exc.code == "recovery_not_due":
                    basis = (row.get("recovery") or {}).get("basis") or {}
                    _unresolved(observations, row, (basis.get("code") if isinstance(basis, dict) else str(basis))
                                or "provider_retry_after")
                    continue
                if exc.code == "recovery_operation_gone":
                    if task_has_live_ownership(task_id, ownership=reads):
                        continue
                    state = usage.terminalize_abandoned_attempt(
                        reservation, reason="owner_task_terminal", expected_revision=row["revision"])
                    if state == "settled":
                        _recovery_fact(root, row, "operation_gone", {"operation_get_status": exc.status_code,
                                                                   "code": str(exc)},
                                       revision=row["revision"] + (0 if abandoned else 1))
                    if state in {"settled", "released"}:
                        unproven.discard(attempt_id)
                    continue
                if due := _retry_after(exc.retry_after):
                    _recovery_fact(root, row, "unresolved", {"header": "Retry-After", "value": exc.retry_after,
                                                           "code": exc.code}, due_at=due)
                gateway_unavailable = gateway_unavailable or exc.code == "daemon_unreachable"
                _unresolved(observations, row, exc.code)
            except Exception as exc:
                _unresolved(observations, row, f"recovery_failed:{_reason(exc)}")
                faults[attempt_id] = exc
            finally:
                with _PROBE_LOCK:
                    _PROBING.discard(probe_key)
    finally:
        if gateway is not None:
            _close_gateway(root, gateway)
        _publish_unresolved(root, observations, faults=faults, keep=unproven, complete=probes_complete)
    _refresh_costs(root, reads)
