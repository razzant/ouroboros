"""One durable non-panic terminal boundary for delegated custody."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from ouroboros import delegate_custody as custody

log = logging.getLogger(__name__)


def terminal_reconcile_task(
    drive_root: Any,
    task_id: str,
    *,
    gateway_factory: Optional[Callable[[], Any]] = None,
    trigger: str = "terminal_boundary",
) -> Dict[str, Any]:
    """Reconcile durable starts, then re-audit runs, invocations, and patches."""

    mine = str(task_id or "")
    result: Dict[str, Any] = {
        "task_id": mine, "trigger": str(trigger or "terminal_boundary"),
        "outcomes": [], "unreconciled": [], "audit_status": "ok",
    }
    if not mine:
        result.update({
            "audit_status": "failed",
            "unreconciled": ["delegated_run_state_unknown:missing_task_id"],
        })
        return result
    try:
        result["outcomes"] = custody.reconcile_task_runs(
            drive_root, mine, gateway_factory=gateway_factory,
        )
    except Exception:
        log.warning("Terminal delegated custody reconciliation failed for %s", mine, exc_info=True)
    _audit_task_custody(drive_root, mine, result)
    return result


def custody_audit_snapshot(drive_root: Any) -> Dict[str, Any]:
    """One shared read of the custody rows for a BATCH of audits.

    One ``replay()`` pass plus one pending-invocations pass, reused by every
    audit in the batch — the boot backfill must not rescan the unbounded event
    log four times per stored row.
    """
    return {
        "state": custody.replay(drive_root),
        "pending": custody.pending_invocations(drive_root),
    }


def _audit_task_custody(drive_root: Any, mine: str, result: Dict[str, Any], *,
                        snapshot: Optional[Mapping[str, Any]] = None,
                        emit_evidence: bool = True) -> None:
    """Read-only custody audit: fills unreconciled/audit fields and emits evidence.

    ``snapshot`` is a ``custody_audit_snapshot`` shared across a batch of
    audits; ``emit_evidence=False`` defers the evidence rows so a caller can
    compare the audit against the stored disclosure first (a no-op refresh must
    not append custody events every boot).
    """
    state = snapshot.get("state") if snapshot is not None else None
    pending = snapshot.get("pending") if snapshot is not None else None
    # The keyword rides only when a snapshot is really shared, so the
    # no-snapshot call shape stays byte-identical for every existing caller
    # and test seam over these projections.
    state_kw: Dict[str, Any] = {} if state is None else {"state": state}
    audit_failure = ""
    if custody.custody_log_unreadable(drive_root):
        audit_failure = "custody_log_unreadable"
    try:
        open_ids = (
            [row.run_id for row in custody.open_runs(drive_root, **state_kw)
             if row.task_id == mine]
            if not audit_failure else []
        )
    except Exception:
        open_ids, audit_failure = [], "audit_failed"
    try:
        invocation_ids = (
            [
                str(row.get("invocation_id") or "")
                for row in (pending if pending is not None
                            else custody.pending_invocations(drive_root))
                if str(row.get("task_id") or "") == mine
                and str(row.get("invocation_id") or "")
            ]
            if not audit_failure else []
        )
    except Exception:
        invocation_ids, audit_failure = [], "pending_invocation_audit_failed"
    try:
        patch_ids = (
            [row.run_id for row in custody.undisposed_patches(drive_root, **state_kw)
             if row.task_id == mine]
            if not audit_failure else []
        )
    except Exception:
        patch_ids, audit_failure = [], "undisposed_patch_audit_failed"
    deferred_retirements: list = []
    try:
        if not audit_failure:
            deferred_retirements = [
                row.run_id
                for row in custody.owned_project_registrations(drive_root, **state_kw)
                if row.task_id == mine and row.settled
            ]
    except Exception:
        # Fail-closed like the audits above: an unreadable registration state
        # must not read as "no deferred retirements".
        audit_failure = audit_failure or "registration_audit_failed"
    if not audit_failure:
        result.update({
            "open_run_ids": open_ids,
            "pending_invocation_ids": invocation_ids,
            "undisposed_patch_run_ids": patch_ids,
            # DISCLOSED, never unreconciled: a settled run's project registration
            # awaiting retirement is cleanup debt with its own retry lane - it
            # must not convert the task's outcome (the old coupling did).
            "deferred_project_retirements": deferred_retirements,
            "unreconciled": [
                *open_ids,
                *(f"invocation:{item}" for item in invocation_ids),
                *(f"patch:{item}" for item in patch_ids),
            ],
        })
    else:
        result.update({
            "audit_status": "failed",
            "unreconciled": [f"delegated_run_state_unknown:{audit_failure}"],
        })
    if emit_evidence:
        _emit_audit_evidence(drive_root, result)


def _emit_audit_evidence(drive_root: Any, result: Mapping[str, Any]) -> None:
    """The audit's durable evidence rows — separated so a comparison can run first."""
    mine = str(result.get("task_id") or "")
    if result["unreconciled"]:
        custody.emit(drive_root, "delegated_runs_unreconciled", {
            "task_id": mine, "trigger": result["trigger"],
            "run_ids": list(result["unreconciled"]),
            "audit_status": result["audit_status"],
            **({"flavor": "audit_failed"} if result["audit_status"] != "ok" else {}),
            "open_run_ids": list(result.get("open_run_ids") or []),
            "pending_invocation_ids": list(result.get("pending_invocation_ids") or []),
            "undisposed_patch_run_ids": list(result.get("undisposed_patch_run_ids") or []),
        })
    else:
        custody.emit(drive_root, "delegate_terminal_custody_reconciled", {
            "task_id": mine, "trigger": result["trigger"],
            "outcome_count": len(result["outcomes"]),
        })


_EVIDENCE_COUNTER_KEYS = (
    "delegated_runs_started", "delegated_runs_settled",
    "delegated_runs_succeeded", "delegated_runs_failed",
    "delegated_runs_source_unresolved",
)


def _stored_evidence_stale(existing: Mapping[str, Any], live: Mapping[str, Any]) -> bool:
    """True when the stored CURRENT-TRUTH substrate surfaces disagree with custody.

    Only tasks that ever wrote the harness-dispatch mirror participate: a task
    with neither top-level counters nor an envelope evidence block was not
    delegated, and minting one here would fabricate a dispatch record.

    Owner Q2=B x this sprint's 1=A, reconciled by SURFACE: the top-level
    ``delegated_runs_*`` counters are a HISTORICAL SNAPSHOT at the original
    terminal write and are deliberately NOT compared here — they may honestly
    read ``settled: 0`` beside a later settlement forever. The current-truth
    surfaces are what staleness means: the ``subagent_envelope`` evidence
    mirror, ``actual_substrate``, and ``subscription_cost_usd`` — the fields
    whose lie (``harness_attempted``/free after a paid successful run) the
    audit reproduced.
    """
    envelope = existing.get("subagent_envelope")
    stored_ev = (envelope or {}).get("execution_evidence") if isinstance(envelope, Mapping) else None
    has_top = any(key in existing for key in _EVIDENCE_COUNTER_KEYS)
    if not has_top and not isinstance(stored_ev, Mapping):
        return False
    if live.get("evidence_read_failed"):
        # Unreadable custody proves nothing; never rewrite over it.
        return False

    def _as_int(value: Any) -> int:
        # A garbage stored counter (str/None) must not raise out into the
        # caller's blanket except and disable healing forever — treat it as a
        # mismatch so the row is REWRITTEN to the clean live value.
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return -1

    if isinstance(stored_ev, Mapping):
        for key in _EVIDENCE_COUNTER_KEYS:
            if _as_int(stored_ev.get(key)) != _as_int(live.get(key)):
                return True
        if stored_ev.get("subscription_cost_usd") != live.get("subscription_cost_usd"):
            return True
    try:
        from ouroboros.subagents import actual_substrate

        live_substrate = actual_substrate(live)
    except Exception:
        live_substrate = ""
    if live_substrate and str(existing.get("actual_substrate") or "") not in ("", live_substrate):
        return True
    return False


def _rewrite_execution_evidence(drive_root: Any, task_id: str, existing: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    """Rewrite the stored envelope evidence + current-truth substrate surfaces
    from live custody, through the same producers the terminal write used.

    Q2=B split: the top-level ``delegated_runs_*`` counters (and the derived
    ``native_contribution``) stay the historical snapshot — they are filtered
    OUT of the producer's mirror before the write; only ``actual_substrate``
    and the envelope evidence (which is where every reader, the executor chip
    included, takes ``subscription_cost_usd`` from) are healed.
    """
    from ouroboros.subagents import actual_substrate, substrate_result_fields
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    envelope = dict(existing.get("subagent_envelope") or {})
    envelope["execution_evidence"] = dict(live)
    substrate = actual_substrate(live)
    if substrate:
        envelope["actual_substrate"] = substrate
    mirror = {
        key: value for key, value in substrate_result_fields(envelope).items()
        if key not in _EVIDENCE_COUNTER_KEYS and key != "native_contribution"
    }
    write_task_result(
        drive_root, str(task_id or ""),
        str(existing.get("status") or STATUS_RUNNING),
        subagent_envelope=envelope,
        **mirror,
    )


def refresh_terminal_reconciliation(
    drive_root: Any, task_id: str, *,
    trigger: str = "sweep_refresh",
    snapshot: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Audit-only refresh of a TERMINAL task's stored custody disclosure.

    The periodic sweep can settle a run AFTER its owning task already wrote its
    terminal result — the custody ledger then knows the truth while the stored
    projection keeps lying (nanny-leaf S1). TWO independent stale classes are
    healed: a stale ``delegated_runs_unreconciled`` disclosure (audited and
    re-recorded below), and stored substrate counters/cost that disagree with
    live custody (``_stored_evidence_stale`` — the PR #402 test pinned only the
    first class, so ``actual_substrate='harness_attempted'`` and
    ``subscription_cost_usd=None`` survived a successful refresh). This re-runs
    ONLY the read-side audit (never ``reconcile_task_runs`` — a refresh must
    not cancel anything) and rewrites through the same recorders. The primary
    ``reason_code`` is deliberately left untouched (owner Q5=A), and
    already-rendered chat frames are out of scope — the fixed surfaces are the
    stored result, task details, and the API view (including retry-lineage
    projections, which read this row live).

    ``trigger`` names the refreshing surface on the envelope and its evidence
    rows (``sweep_refresh`` | ``boot_backfill`` | ``kill_path_clear``);
    ``snapshot`` shares one ``custody_audit_snapshot`` across a batch. An audit
    that MATCHES the stored disclosure (and finds no stale evidence mirror)
    performs no write and no emit — a permanently-unreconcilable row must not
    grow events.jsonl on every boot. Returns True only when a row was really
    refreshed: the audit evidence is emitted AFTER — and only after — the
    recorder confirms a changed persisted row, so a lock timeout or a refused
    write leaves no phantom "refreshed" event behind (R3).    """
    mine = str(task_id or "")
    if not mine:
        return False
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        existing = load_task_result(drive_root, mine) or {}
        if str(existing.get("status") or "") not in _TRULY_TERMINAL_STATUSES:
            return False
        live = custody.task_execution_evidence(drive_root, mine)
        evidence_stale = _stored_evidence_stale(existing, live)
        if not existing.get("delegated_runs_unreconciled") and not evidence_stale:
            return False
    except Exception:
        log.debug("Sweep refresh skipped: task result unreadable for %s", mine, exc_info=True)
        return False
    result: Dict[str, Any] = {
        "task_id": mine, "trigger": str(trigger or "sweep_refresh"),
        "outcomes": [], "unreconciled": [], "audit_status": "ok",
    }
    _audit_task_custody(drive_root, mine, result, snapshot=snapshot, emit_evidence=False)
    if _stored_disclosure_matches(existing, result) and not evidence_stale:
        return False
    refreshed = False
    # Disclosure class: the recorder itself re-checks the no-churn gate and the
    # monotonic guard; evidence is emitted only after it confirms a landed row.
    if record_terminal_reconciliation(drive_root, mine, result):
        _emit_audit_evidence(drive_root, result)
        refreshed = True
    # Evidence-mirror class: substrate counters/cost rewritten from live
    # custody through the same producers the terminal write used.
    if evidence_stale:
        try:
            _rewrite_execution_evidence(drive_root, mine, existing, live)
            refreshed = True
        except Exception:
            log.warning("Sweep evidence rewrite failed for %s", mine, exc_info=True)
    return refreshed


def backfill_terminal_reconciliations(drive_root: Any) -> List[str]:
    """Boot backfill: refresh every stored TERMINAL row that still discloses
    unreconciled delegated runs.

    The sweep-side refresh covers only task ids named in the CURRENT pass's
    reconcile outcomes, so a settlement from a previous server generation
    leaves the stored projection stale forever (generation-crossing residual).
    Driven by the reverse join — the stored results with a non-empty
    ``delegated_runs_unreconciled`` are a self-clearing set — never by a replay
    scan of the unbounded event log; one shared snapshot serves every audit,
    and each row is fail-soft. Returns the task ids actually refreshed.
    """
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, list_task_results

        stale = [
            str(row.get("task_id") or "")
            for row in list_task_results(drive_root)
            if row.get("delegated_runs_unreconciled")
            and str(row.get("status") or "") in _TRULY_TERMINAL_STATUSES
        ]
    except Exception:
        log.debug("Boot custody-disclosure backfill scan failed", exc_info=True)
        return []
    if not stale:
        return []
    snapshot = custody_audit_snapshot(drive_root)
    refreshed: List[str] = []
    for task_id in stale:
        try:
            if refresh_terminal_reconciliation(
                    drive_root, task_id, trigger="boot_backfill", snapshot=snapshot):
                refreshed.append(task_id)
        except Exception:
            log.debug("Boot custody-disclosure backfill failed for %s", task_id, exc_info=True)
    return refreshed


# The envelope fields whose values change WHAT a reader can conclude about the
# task's delegated custody — compared by the no-churn gate below. Provenance
# (trigger/outcomes/audit_status) is deliberately excluded: it differs between
# otherwise identical audits and would defeat the gate.
_ENVELOPE_DISCLOSURE_FIELDS = (
    "open_run_ids",
    "pending_invocation_ids",
    "undisposed_patch_run_ids",
    "deferred_project_retirements",
)


def _stored_disclosure_matches(
    existing: Mapping[str, Any], result: Mapping[str, Any],
) -> bool:
    """Whether the stored row already carries exactly this audit's disclosure.

    Compared on the behavior-bearing surfaces (R2): the flat unreconciled list
    AND every ``_ENVELOPE_DISCLOSURE_FIELDS`` entry of the stored envelope,
    plus envelope PRESENCE — a kill-written row carrying only the flat list
    must NOT match, so the next boot adds the envelope once and then becomes a
    byte-level no-op. The one exception is a row with nothing to disclose at
    all (no list, no envelope fields): rewriting such a row — or minting one
    for a task that never delegated — just to attach an empty envelope would
    be churn, so an absent envelope over an all-empty audit still matches.
    """
    stored_envelope = existing.get("delegate_terminal_reconciliation")
    has_envelope = isinstance(stored_envelope, dict) and bool(stored_envelope)
    stored_envelope = stored_envelope if isinstance(stored_envelope, dict) else {}
    if list(existing.get("delegated_runs_unreconciled") or []) != list(
            result.get("unreconciled") or []):
        return False
    if any(
        list(stored_envelope.get(field) or []) != list(result.get(field) or [])
        for field in _ENVELOPE_DISCLOSURE_FIELDS
    ):
        return False
    if has_envelope:
        return True
    return not (
        list(result.get("unreconciled") or [])
        or any(list(result.get(field) or []) for field in _ENVELOPE_DISCLOSURE_FIELDS)
    )


_REFRESH_CURSOR_REL = "state/delegate_terminal_refresh_cursor.json"
_REFRESH_SCAN_CAP_BYTES = 5 * 1024 * 1024  # bounded work per sweep tick
_REFRESH_DEFERRED_CAP = 500  # terminal-boundary tasks awaiting their result


def refresh_recently_settled_terminals(drive_root: Any) -> int:
    """Refresh terminal results of tasks whose runs settled since the cursor.

    The orphan sweep only revisits tasks named in THIS generation's reconcile
    outcomes; a run settled at the terminal boundary (or by an earlier
    generation) never reappears there, so its task's stored evidence stays
    stale forever. A durable byte-offset cursor over the append-only custody
    event log keeps each tick bounded to newly appended SETTLED rows (house
    projection-beside-the-log pattern) — never a full replay per sweep. The
    offset counts bytes across the ROTATED CHAIN (archive segments + live
    file, CPL4-C1): archive segments are immutable and never GC'd, so the
    chain offset stays monotonic across rotation and no settled row is lost
    to a rename. A shrunken chain (manual surgery) resets the cursor; the
    one-time historical pass is paced by the per-tick byte cap. Returns the
    number of refreshed tasks.
    """
    import os as _os
    import pathlib as _pathlib

    from ouroboros.utils import atomic_write_json, jsonl_chain_handles, read_json_dict

    log_path = custody.event_log_path(drive_root)
    cursor_path = _pathlib.Path(drive_root) / _REFRESH_CURSOR_REL
    stored = read_json_dict(cursor_path) or {}
    offset = int(stored.get("offset") or 0)
    deferred: Dict[str, str] = {
        str(k): str(v) for k, v in (stored.get("deferred") or {}).items() if k
    }
    # A settled run whose OWNING TASK has not yet written its terminal result
    # cannot be healed now (refresh no-ops on a non-terminal task) — the run
    # settled at the terminal boundary, the exact class this pass exists to
    # catch. Its task_id goes into the durable ``deferred`` map (retried every
    # tick) while the byte offset ALWAYS advances: pinning the offset on the
    # earliest deferred row would re-read the same window forever behind one
    # long-lived parent and starve every later settlement past the per-tick
    # byte cap.
    batch_ids: set = set()
    end_offset = offset
    try:
        with jsonl_chain_handles(log_path) as handles:
            sizes = []
            for _, handle in handles:
                try:
                    sizes.append(_os.fstat(handle.fileno()).st_size)
                except OSError:
                    sizes.append(0)
            total = sum(sizes)
            if offset > total:
                offset = 0  # shrunken chain: re-ground once
            end_offset = offset
            if offset >= total and not deferred:
                return 0
            consumed, index = offset, 0
            while index < len(handles) and consumed >= sizes[index]:
                consumed -= sizes[index]
                index += 1
            read_bytes = 0
            for pos in range(index, len(handles)):
                _, handle = handles[pos]
                if pos == index and consumed:
                    handle.seek(consumed)
                is_live = pos == len(handles) - 1
                for raw in handle:
                    if read_bytes > _REFRESH_SCAN_CAP_BYTES:
                        break
                    if not raw.endswith(b"\n"):
                        # A torn LIVE tail line completes on a later tick; a
                        # torn line inside an immutable archive segment never
                        # will — consume its bytes or the cursor wedges there.
                        if is_live:
                            break
                        end_offset += len(raw)
                        continue
                    read_bytes += len(raw)
                    end_offset += len(raw)
                    row = _json_row(raw)
                    if str(row.get("type") or "") in (custody.SETTLED, custody.CLOSED_ABSENT):
                        tid = str(row.get("task_id") or "")
                        if tid:
                            batch_ids.add(tid)
                if read_bytes > _REFRESH_SCAN_CAP_BYTES:
                    break
    except OSError:
        return 0
    from ouroboros.utils import utc_now_iso

    now_iso = utc_now_iso()
    refreshed = 0
    next_deferred: Dict[str, str] = {}
    for tid in sorted(batch_ids | set(deferred)):
        since = deferred.get(tid) or now_iso
        try:
            if _task_is_terminal(drive_root, tid):
                if refresh_terminal_reconciliation(drive_root, tid):
                    refreshed += 1
            else:
                next_deferred[tid] = since
        except Exception:
            log.debug("Cursor refresh failed for %s", tid, exc_info=True)
            next_deferred[tid] = since
    if len(next_deferred) > _REFRESH_DEFERRED_CAP:
        # Oldest-first eviction, disclosed: a task deferred this long with no
        # terminal result is overwhelmingly abandoned; keeping the map bounded
        # protects the cursor write from unbounded growth.
        keep = sorted(next_deferred.items(), key=lambda kv: kv[1])[-_REFRESH_DEFERRED_CAP:]
        dropped = len(next_deferred) - len(keep)
        next_deferred = dict(keep)
        log.info("Settled-refresh deferred map over cap; dropped %d oldest entries", dropped)
    try:
        atomic_write_json(cursor_path, {"offset": end_offset, "deferred": next_deferred})
    except Exception:
        log.debug("Refresh cursor write failed", exc_info=True)
    return refreshed


def _json_row(raw: bytes) -> Dict[str, Any]:
    import json as _json

    try:
        row = _json.loads(raw)
        return row if isinstance(row, dict) else {}
    except Exception:
        return {}


def _task_is_terminal(drive_root: Any, task_id: str) -> bool:
    """Whether the task already wrote a truly-terminal result (heal-eligible)."""
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        existing = load_task_result(drive_root, str(task_id or "")) or {}
        return str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES
    except Exception:
        return False


def record_terminal_reconciliation(
    drive_root: Any, task_id: str, result: Mapping[str, Any],
) -> bool:
    """Attach the audit to the task record without choosing lifecycle policy.

    Returns True only when the row was actually persisted AND now carries this
    audit's disclosure (R3) — a lock timeout, a raising store, or a write the
    monotonic guard refused all report False, so callers (the guarded refresh,
    the boot backfill) cannot claim a heal that never landed.
    """

    try:
        from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

        existing = load_task_result(drive_root, str(task_id or "")) or {}
        # Write only when the disclosure MOVES. Subsumes the old empty-over-empty
        # guard: a task that never delegated still gets no row minted here (the
        # STATUS_RUNNING fallback below exists for a legitimately mid-flight
        # loop-exit row, never for inventing one), and an already-current
        # disclosure is not rewritten on every kill or boot.
        if _stored_disclosure_matches(existing, result):
            return False
        stored = write_task_result(
            drive_root,
            str(task_id or ""),
            str(existing.get("status") or STATUS_RUNNING),
            delegate_terminal_reconciliation=dict(result),
            delegated_runs_unreconciled=list(result.get("unreconciled") or []),
        )
        return _stored_disclosure_matches(
            stored if isinstance(stored, dict) else {}, result,
        )
    except Exception:
        log.warning("Failed to persist terminal custody audit for %s", task_id, exc_info=True)
        return False


__all__ = [
    "backfill_terminal_reconciliations",
    "custody_audit_snapshot",
    "record_terminal_reconciliation",
    "refresh_recently_settled_terminals",
    "refresh_terminal_reconciliation",
    "terminal_reconcile_task",
]
