"""One durable non-panic terminal boundary for delegated custody.

Refreshing a stored terminal result never touches its ``delegated_runs_*``
counters, which stay the snapshot of the original terminal write;
``actual_substrate`` and the envelope evidence mirror follow live custody.
``refresh_recently_settled_terminals`` reads the closing receipts the custody
reducer parked in ``delegated_runs`` (the store's owner-held rows): a receipt
stays until its task's terminal result absorbed it, so no cursor and no event
scan are needed per tick.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from ouroboros import delegate_custody as custody
from ouroboros.observability import timed_phase

log = logging.getLogger(__name__)


def terminal_reconcile_task(
    drive_root: Any,
    task_id: str,
    *,
    gateway_factory: Optional[Callable[[], Any]] = None,
    trigger: str = "terminal_boundary",
    deliberate_terminal: str = "",
    audit: bool = True,
) -> Dict[str, Any]:
    """Reconcile durable starts, then re-audit runs, invocations, and patches.

    A still-live run is cancelled only behind a deliberate owner terminal
    (``delegate_custody_reconcile._owner_terminal_is_deliberate``); otherwise it
    is left live and disclosed as open, for the sweep to re-judge.
    ``deliberate_terminal`` carries that verdict from a caller that has not
    written its terminal result yet, which is every kill boundary. A terminal
    boundary is a release surface: ``gateway_factory`` defaults to the attach-only
    transport and no pending invocation is replayed (exit does not admit)."""

    from ouroboros.delegate_custody_reconcile import attach_only_gateway

    gateway_factory = gateway_factory or attach_only_gateway
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
            deliberate_terminal=deliberate_terminal,
        )
    except Exception:
        log.warning("Terminal delegated custody reconciliation failed for %s", mine, exc_info=True)
    try:
        _name_unread_results(drive_root, mine)
    except Exception:
        log.warning("Never-read disclosure failed for %s", mine, exc_info=True)
    if audit:
        _audit_task_custody(drive_root, mine, result)
    else:
        result["audit_status"] = "pending"
    return result


def _name_unread_results(drive_root: Any, task_id: str) -> None:
    """The attempt ends with a settled result staged in full and never read: name it once (D7, #1573).
    Since #2205 the body's collector, not this boundary, retains outputs; the omission is still this
    boundary's fact, so the next context shows NEVER READ instead of an available result."""
    from ouroboros.delegate_custody_current import task_snapshot

    for run in task_snapshot(drive_root, task_id)["runs"].values():
        if not run.review_owned:
            custody.record_settled_unread(drive_root, run)


def terminal_custody_disclosure(
    drive_root: Any, task_id: str, reconciliation: Optional[Mapping[str, Any]] = None,
    *, snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze this task's delivery facts without persisting or emitting an annotation.

    Wrap up's retained-work receipt was frozen before the root's final input; it rides
    every exit, zero-call and error ones too, independently of the model text."""
    result = {
        "task_id": str(task_id or ""),
        "trigger": str((reconciliation or {}).get("trigger") or "final_delivery"),
        "outcomes": list((reconciliation or {}).get("outcomes") or []),
        "unreconciled": [], "audit_status": "ok",
    }
    _audit_task_custody(drive_root, result["task_id"], result, snapshot=snapshot, emit_evidence=False)
    from ouroboros.task_results import load_task_result
    try:
        stored = (load_task_result(drive_root, result["task_id"]) or {}).get("delegate_terminal_reconciliation")
    except Exception:
        stored = None
    if isinstance(stored, Mapping) and isinstance(stored.get("owner_stop_children"), Mapping):
        result["owner_stop_children"] = dict(stored["owner_stop_children"])
    return result


def custody_audit_snapshot(drive_root: Any, *, task_id: str = "") -> Dict[str, Any]:
    """Read one task's runs/requests, or share held/closing rows across a maintenance batch.

    Delivery names its task; unrelated run bodies and request references stay unread.
    Batch backfill shares the unaddressed snapshot once, never once per result.
    """
    from ouroboros.delegate_custody_current import snapshot, task_snapshot

    if task_id:
        view = task_snapshot(drive_root, task_id)
        return {"state": view["runs"], "pending": view["pending"], "current": True}
    return snapshot(drive_root)


@timed_phase("custody_audit", within="release_task_runs")
def _audit_task_custody(drive_root: Any, mine: str, result: Dict[str, Any], *,
                        snapshot: Optional[Mapping[str, Any]] = None,
                        emit_evidence: bool = True) -> None:
    """Read-only custody audit: fills unreconciled/audit fields and emits evidence.

    ``snapshot`` is a ``custody_audit_snapshot`` shared across a batch of
    audits; ``emit_evidence=False`` defers the evidence rows so a caller can
    compare the audit against the stored disclosure first (a no-op refresh must
    not append custody events every boot).

    Runs and invocations a REVIEW surface owns are outside this audit's domain:
    they are their panel's obligation, so they are never this task's open
    delegation, pending invocation or terminal receipt (issue #1006).
    """
    audit_failure = "" if mine else "missing_task_id"
    if snapshot is None and not audit_failure:
        try:
            snapshot = custody_audit_snapshot(drive_root, task_id=mine)
        except Exception:
            log.debug("Custody audit snapshot unavailable for %s", mine, exc_info=True)
            audit_failure = "snapshot_unavailable"
    state = snapshot.get("state") if snapshot is not None else {}
    pending = snapshot.get("pending") if snapshot is not None else []
    state_kw: Dict[str, Any] = {"state": state}
    if custody.custody_log_unreadable(drive_root):
        audit_failure = "custody_log_unreadable"
    try:
        open_ids = (
            [row.run_id for row in custody.open_runs(drive_root, **state_kw)
             if row.task_id == mine and not row.review_owned]
            if not audit_failure else []
        )
    except Exception:
        open_ids, audit_failure = [], "audit_failed"
    try:
        invocation_ids = (
            [
                str(row.get("invocation_id") or "")
                for row in pending
                if str(row.get("task_id") or "") == mine
                and str(row.get("invocation_id") or "")
                and not custody.review_owned_source(row.get("source"))
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
    terminal_runs = []
    try:
        if not audit_failure:
            # The LEAF's own identity, already on the replayed row (zero extra
            # reads): without it a nanny terminal names only the role the host
            # played and the reader asks why the leaf's model was broken (I9).
            terminal_runs = sorted((
                {"run_id": str(row.run_id), "state": str(row.terminal_state),
                 "model": str(row.model), "profile_id": str(row.profile_id),
                 "selected_subagent_id": str(row.selected_subagent_id),
                 **_answer_locators(drive_root, row)}
                for row in state.values()
                if row.task_id == mine and not row.review_owned
                and row.terminal_state in custody.TERMINAL_STATES
            ), key=lambda row: row["run_id"])
    except Exception:
        audit_failure = audit_failure or "terminal_receipt_audit_failed"
    if not audit_failure:
        result.update({
            "open_run_ids": open_ids,
            "pending_invocation_ids": invocation_ids,
            "undisposed_patch_run_ids": patch_ids,
            "terminal_runs": terminal_runs,
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


def _answer_locators(drive_root: Any, row: Any) -> Dict[str, Any]:
    """Where a terminal run's answer lives, as recorded facts only (#1573): no source is read.

    ``engine_locator`` is the engine's own run folder and primary-output metadata
    (never text; engine-owned and possibly reclaimed). ``staged_output`` is the
    retained full terminal payload: its exact ``task_source`` ref with the ACTUAL
    staging task and completeness. ``read`` keeps the staging-file path;
    ``source_read`` reads from any drive, pinned to the PHYSICAL task publishing this receipt (the run's
    owner, which a retry successor's staging task need not be). ``terminal_observation``
    is the engine read-back the body retained (#1564), with the same pinned reader:
    complete as an observation, never proof that the whole output was collected. A row
    that recorded none of them adds no key, so its stored receipt stays byte-identical.
    """
    facts: Dict[str, Any] = {}
    if row.terminal_source:
        facts["terminal_observation"] = {
            # Its relative ``read`` belongs to the staging task; only the pinned reader crosses drives.
            **{key: value for key, value in row.terminal_source.items() if key != "read"},
            "complete": True, "content_kind": "terminal_observation",
            "source_read": {"tool": "get_task_result", "arguments": {
                "task_id": str(row.task_id),
                "delegated_output_sha256": str(row.terminal_source.get("sha256") or "")}},
        }
    if row.engine_locator:
        facts["engine_locator"] = dict(row.engine_locator)
    if row.output_source and row.output_artifact:
        from ouroboros.delegate_output import staged_output_address

        address = staged_output_address(drive_root, row)
        facts["staged_output"] = {
            **dict(row.output_source), "complete": bool(row.output_complete),
            **({"read": {"tool": "read_file", "arguments": {
                **address, "start_line": 1, "max_lines": 2000}}} if address else {}),
            "source_read": {"tool": "get_task_result", "arguments": {
                "task_id": str(row.task_id),
                "delegated_output_sha256": str(row.output_source.get("sha256") or "")}},
        }
    return facts


def terminal_custody_notice(result: Mapping[str, Any], *, owner_row: bool = False) -> str:
    """Render current stored custody beside unchanged model-authored history.

    No live join, lifecycle mutation or inference from a missing run: only the
    audit's separate execution/invocation/patch and confirmed terminal facts. A run
    confirmed terminal whose accounting is not settled stays in the audit's open
    list (custody unchanged) and reads as terminal with that fact, never as live.
    ``owner_row`` is the owner's task-card row, which lists only the categories
    holding something and leaves Wrap up's receipt to its own row
    (``owner_stop_receipt.retained_work_fields``); the model, CLI and
    child-result-hash text keeps every category and the receipt.
    """
    audit = result.get("delegate_terminal_reconciliation")
    if not isinstance(audit, Mapping) or not audit:
        return ""
    from ouroboros.owner_stop_receipt import receipt_text
    retained = "" if owner_row else receipt_text(audit)
    if audit.get("audit_status") != "ok":
        return "\n\n".join(part for part in (
            "Current delegated execution could not be verified; custody remains unresolved in task details.",
            retained) if part)
    terminal = [row for row in (audit.get("terminal_runs") or []) if isinstance(row, dict)]
    raw_open = audit.get("open_run_ids")
    open_ids = [str(value) for value in raw_open] if isinstance(raw_open, list) else []
    unsettled = {str(row.get("run_id")) for row in terminal} & set(open_ids)
    shown_runs = [row for row in terminal if row.get("state") != "succeeded" or str(row.get("run_id")) in unsettled]
    groups = [
        ("Open delegated execution", [rid for rid in open_ids if rid not in unsettled]),
        ("Pending delegated invocations", audit.get("pending_invocation_ids")),
        ("Pending patch decisions", audit.get("undisposed_patch_run_ids")),
    ]
    if not shown_runs and not any(values for _label, values in groups):
        return retained
    lines = []
    if shown_runs:
        # The leaf's model when the replayed row carried one, never a live join
        # and never a guess: an older row without it simply says less.
        shown = "; ".join(
            f"{row.get('run_id')}: {row.get('state')}"
            + (f" on {row.get('model')}" if row.get("model") else "")
            + (" (accounting not settled)" if str(row.get("run_id")) in unsettled else "")
            for row in shown_runs[:10])
        omitted = len(shown_runs) - 10
        lines.append("Confirmed delegated terminal receipts: " + shown
                     + (f" (+{omitted} more in task details)" if omitted > 0 else "") + ".")
    for label, raw in groups:
        values = [str(value) for value in raw] if isinstance(raw, list) else []
        if owner_row and not values:
            continue
        shown = ", ".join(values[:10]) or "none recorded by the current audit"
        omitted = len(values) - 10
        lines.append(label + ": " + shown
                     + (f" (+{omitted} more in task details)" if omitted > 0 else "") + ".")
    return "\n\n".join(part for part in ("\n".join(lines), retained) if part)


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
        if (stored_ev.get("subscription_cost_usd") != live.get("subscription_cost_usd")
                or any(bool(stored_ev.get(key)) != bool(live.get(key)) for key in
                       ("subscription_cost_estimated", "subscription_cost_partial"))):
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
    rows (``sweep_refresh`` | ``boot_backfill`` | ``kill_path_clear`` | ``output_spilled``);
    ``snapshot`` shares one ``custody_audit_snapshot`` across a batch. An audit
    that MATCHES the stored disclosure (and finds no stale evidence mirror)
    performs no write and no emit unless a separate cost_revision owes a
    monetary projection. That debt uses the existing cost writer; merely adding
    the optional coverage field emits nothing. A permanently-unreconcilable row must not
    grow events.jsonl on every boot. Returns True only when a row was really
    refreshed: the audit evidence is emitted AFTER — and only after — the
    recorder confirms a changed persisted row, so a lock timeout or a refused
    write leaves no phantom "refreshed" event behind (R3).    """
    mine = str(task_id or "")
    if not mine:
        return False
    _refresh_custody_cost(drive_root, mine)
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        existing = load_task_result(drive_root, mine) or {}
        if str(existing.get("status") or "") not in _TRULY_TERMINAL_STATUSES:
            return False
        live = custody.task_execution_evidence(drive_root, mine)  # indexed store reads
        evidence_stale = _stored_evidence_stale(existing, live)
    except Exception:
        log.debug("Sweep refresh skipped: task result unreadable for %s", mine, exc_info=True)
        return False
    result: Dict[str, Any] = {
        "task_id": mine, "trigger": str(trigger or "sweep_refresh"),
        "outcomes": [], "unreconciled": [], "audit_status": "ok",
    }
    if snapshot is None:
        snapshot = custody_audit_snapshot(drive_root)
    _audit_task_custody(drive_root, mine, result, snapshot=snapshot, emit_evidence=False)
    # Completed receipts already on the result are history, not missing current facts. Merged
    # here only for the no-op comparison: the recorder merges against CURRENT under the write,
    # so a historical locator never overrides a concurrent publication as an incoming fact.
    comparison = _merge_current_receipts(existing, result, addressed=True)
    # Only an audit that knew the receipts may retire them: a failed one keeps every owed receipt.
    acknowledge = result.get("audit_status") == "ok"
    if _stored_disclosure_matches(existing, comparison) and not evidence_stale:
        if acknowledge:
            _ack_current_receipts(drive_root, mine, snapshot)
        return False
    refreshed = False
    # Disclosure class: the recorder itself re-checks the no-churn gate and the
    # monotonic guard; evidence is emitted only after it confirms a landed row.
    if record_terminal_reconciliation(drive_root, mine, result, addressed=True):
        _emit_audit_evidence(drive_root, result)
        if acknowledge:
            _ack_current_receipts(drive_root, mine, snapshot)
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
        from ouroboros.obligations import members
        stale = list(members(drive_root, "delegated_runs"))
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
    "terminal_runs",
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
    if stored_envelope.get("owner_stop_children") != result.get("owner_stop_children"):
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


# Receipt facts a later custody row adds to an already-settled run (#1573). An
# audit without one is older than the row that has it, never proof of removal.
_RECEIPT_LOCATOR_KEYS = ("engine_locator", "staged_output", "terminal_observation")
# Every other immutable ref a receipt published for its run, sorted by digest: no
# order is known between them and ``staged_output``, which a delayed audit may set.
_OTHER_OUTPUTS_KEY = "other_staged_outputs"
_OTHER_OBSERVATIONS_KEY = "other_terminal_observations"


def _published_refs(receipt: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    others = receipt.get(_OTHER_OUTPUTS_KEY)
    return [ref for ref in (receipt.get("staged_output"), *(others if isinstance(others, list) else ()))
            if isinstance(ref, Mapping) and ref.get("sha256")]


def _published_observations(receipt: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    others = receipt.get(_OTHER_OBSERVATIONS_KEY)
    return [ref for ref in (receipt.get("terminal_observation"), *(others if isinstance(others, list) else ()))
            if isinstance(ref, Mapping) and ref.get("sha256")]


def _keep_published_outputs(row: Dict[str, Any], old: Mapping[str, Any]) -> Dict[str, Any]:
    """``row`` plus every digest ``old`` published beside ``row``'s own ``staged_output``."""
    refs = {ref["sha256"]: ref for receipt in (old, row) for ref in _published_refs(receipt)}
    own = row.get("staged_output")
    refs.pop(own.get("sha256") if isinstance(own, Mapping) else None, None)
    kept = {key: value for key, value in row.items() if key not in (_OTHER_OUTPUTS_KEY, _OTHER_OBSERVATIONS_KEY)}
    if refs:
        kept[_OTHER_OUTPUTS_KEY] = [refs[sha] for sha in sorted(refs)]
    observations = {ref["sha256"]: ref for receipt in (old, row) for ref in _published_observations(receipt)}
    own_observation = row.get("terminal_observation")
    observations.pop(own_observation.get("sha256") if isinstance(own_observation, Mapping) else None, None)
    if observations:
        kept[_OTHER_OBSERVATIONS_KEY] = [observations[sha] for sha in sorted(observations)]
    return kept


def _merge_current_receipts(
    current: Mapping[str, Any], result: Mapping[str, Any], *, addressed: bool,
) -> Dict[str, Any]:
    """``result`` plus the receipt facts ``current`` holds and the audit lacks.

    Per run, every key the audit carries wins; a locator only ``current`` carries
    is kept, and so is every staged-output digest it published: a delayed audit's
    own ref may be older than a later publication, so the replaced ref stays
    readable in ``other_staged_outputs`` (unordered, never a recency claim).
    ``addressed`` (a current-store refresh, whose snapshot no longer holds
    acknowledged receipts) and a failed audit (which knows no receipt) also keep
    whole receipts the audit omits; any audit keeps one that published a digest.
    The open/pending/patch/deferred/unreconciled lists stay the audit's own: debt
    is never unioned, so a writer cannot keep debt a fresher audit cleared, and
    stale debt it writes keeps its ``delegated_runs`` membership for the heal.
    """
    stored = current.get("delegate_terminal_reconciliation")
    if isinstance(stored, Mapping) and "owner_stop_children" in stored:
        if "owner_stop_children" not in result:
            result = {**result, "owner_stop_children": stored["owner_stop_children"]}
        else:
            from ouroboros.owner_stop_receipt import merge_stop_receipt

            result = {**result, "owner_stop_children": merge_stop_receipt(
                stored["owner_stop_children"], result["owner_stop_children"])}
    rows = stored.get("terminal_runs") if isinstance(stored, Mapping) else None
    prior = {row["run_id"]: row for row in (rows if isinstance(rows, list) else [])
             if isinstance(row, Mapping) and row.get("run_id")}
    if not prior:
        return dict(result)
    receipts = {run_id: row for run_id, row in prior.items()
                if addressed or "terminal_runs" not in result or _published_refs(row) or _published_observations(row)}
    for row in result.get("terminal_runs") or []:
        old = prior.get(row["run_id"]) or {}
        receipts[row["run_id"]] = _keep_published_outputs(
            {**row, **{key: old[key] for key in _RECEIPT_LOCATOR_KEYS if key in old and key not in row}}, old)
    return {**result, "terminal_runs": sorted(receipts.values(), key=lambda row: row["run_id"])}


def keep_published_receipts(current: Mapping[str, Any], envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """The result writer's floor (#1573): ``envelope`` merged as a non-addressed audit.

    ``write_task_result`` applies it inside its CURRENT reducer to every incoming envelope,
    so a wholesale terminal writer (kill, reap, cancel) cannot unpublish a staged-output
    digest either; idempotent over a recorder's own merge.
    """
    return _merge_current_receipts(current, envelope, addressed=False)


def refresh_recently_settled_terminals(drive_root: Any, live_task_ids: Any = None) -> int:
    """Refresh terminal results of tasks whose runs closed since their terminal write.

    The orphan sweep only revisits tasks named in THIS generation's reconcile
    outcomes; a run settled at the terminal boundary (or by an earlier
    generation) never reappears there. The custody reducer parks every closing
    receipt in ``delegated_runs`` (one owner-held row per task) in the settling
    transaction; this tick reads those rows and heals each task whose result is
    terminal, and ``_ack_current_receipts`` retires the receipts the healed result
    absorbed. A receipt whose task is not terminal yet simply waits in its row.
    Returns the number of refreshed tasks.

    ``live_task_ids`` is the sweep's one live-owner source — a set or a zero-arg
    callable. A task whose owner still bills (the post-task synthesis window
    outlives the terminal write) is DEFERRED, not healed under a live writer;
    None means unknown and heals whatever holds a receipt.
    """
    from ouroboros import store
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    try:
        with store.open_store(drive_root, write=False) as txn:
            # Missing/nonterminal owners keep debt without body reads or refresh
            # attempts each sweep. Their terminal write makes it eligible here.
            slots = ",".join("?" for _ in _TRULY_TERMINAL_STATUSES)
            owed = [row["id"] for row in txn.conn.execute(
                "SELECT o.id FROM obligations o JOIN tasks t ON t.task_id=o.id "
                f"WHERE o.set_name='delegated_runs' AND t.status IN ({slots}) ORDER BY o.id",
                tuple(_TRULY_TERMINAL_STATUSES))]
    except store.StoreError:
        log.debug("Settled-terminal refresh skipped: custody receipts unavailable", exc_info=True)
        return 0
    if not owed:
        return 0
    live = live_task_ids() if callable(live_task_ids) else live_task_ids
    snapshot = custody_audit_snapshot(drive_root)
    refreshed = 0
    for tid in owed:
        if live is not None and tid in live:
            continue  # its owner is still writing; heal after it ends
        try:
            if _task_is_terminal(drive_root, tid) and refresh_terminal_reconciliation(drive_root, tid, snapshot=snapshot):
                refreshed += 1
        except Exception:
            log.debug("Settled-terminal refresh failed for %s", tid, exc_info=True)
    return refreshed


def _task_is_terminal(drive_root: Any, task_id: str) -> bool:
    """Whether the task already wrote a truly-terminal result (heal-eligible)."""
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        existing = load_task_result(drive_root, str(task_id or "")) or {}
        return str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES
    except Exception:
        return False


def record_terminal_reconciliation(
    drive_root: Any, task_id: str, result: Mapping[str, Any], *, addressed: bool = False,
    emit_evidence: bool = False,
) -> bool:
    """Attach the audit to the task record without choosing lifecycle policy.

    Returns True only when the row was actually persisted AND now carries this
    audit's disclosure (R3) — a lock timeout, a raising store, or a write the
    monotonic guard refused all report False, so callers (the guarded refresh,
    the boot backfill) cannot claim a heal that never landed.

    The audit is merged with CURRENT inside the write transaction
    (``_merge_current_receipts``; ``addressed`` as there): an audit read before a
    concurrent writer published a receipt locator cannot erase it, and an audit
    that then matches is not written at all.
    """

    try:
        from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

        existing = load_task_result(drive_root, str(task_id or "")) or {}
        # Write only when the disclosure MOVES. Subsumes the old empty-over-empty
        # guard: a task that never delegated still gets no row minted here (the
        # STATUS_RUNNING fallback below exists for a legitimately mid-flight
        # loop-exit row, never for inventing one), and an already-current
        # disclosure is not rewritten on every kill or boot.
        if _stored_disclosure_matches(
                existing, _merge_current_receipts(existing, result, addressed=addressed)):
            return False
        landed: Dict[str, Any] = {}

        def project(current, incoming):
            # The read above chose the status and the fast no-op; CURRENT decides the merge.
            landed.clear()
            merged = _merge_current_receipts(
                current, incoming["delegate_terminal_reconciliation"], addressed=addressed)
            if _stored_disclosure_matches(current, merged):
                return None
            landed.update(merged)
            return {**incoming, "delegate_terminal_reconciliation": merged,
                    "delegated_runs_unreconciled": list(merged.get("unreconciled") or [])}

        stored = write_task_result(
            drive_root,
            str(task_id or ""),
            str(existing.get("status") or STATUS_RUNNING),
            _field_projector=project,
            delegate_terminal_reconciliation=dict(result),
            delegated_runs_unreconciled=list(result.get("unreconciled") or []),
        )
        changed = bool(landed) and _stored_disclosure_matches(
            stored if isinstance(stored, dict) else {}, landed,
        )
        if changed and emit_evidence:
            _emit_audit_evidence(drive_root, landed)
        return changed
    except Exception:
        log.warning("Failed to persist terminal custody audit for %s", task_id, exc_info=True)
        return False


def refresh_disposed_reconciliation(drive_root: Any, run_id: str, *, reader_task_id: str) -> int:
    """Remove one proven disposed obligation from current retry-chain disclosures.

    A predecessor mirror may still be interrupted, and intermediate retries may
    have copied its debt. Only exact settled/disposed identifiers are removed,
    under each result writer's lock. Other debt, lifecycle, cost, counters,
    reasons and review evidence retain their meaning; this never audits or
    controls a live execution and never creates a missing result.

    The debt is removed from the starter task, and from its retry chain only
    when ``_confirmed_retry_chain`` validates it for ``reader_task_id``; an
    unreadable custody log, or a run that is not both settled and
    patch-disposed, clears nothing. Returns the number of results cleared.
    """
    from ouroboros.delegate_shared import _confirmed_retry_chain
    from ouroboros.task_results import write_task_result

    if custody.custody_log_unreadable(drive_root):
        return 0
    entry = custody.replay(drive_root).get(str(run_id or ""))
    if entry is None or entry.review_owned or not entry.settled or not entry.patch_disposed:
        return 0
    chain = _confirmed_retry_chain(drive_root, entry.task_id, reader_task_id)
    task_ids = chain or (entry.task_id,)
    tokens = {entry.run_id, f"patch:{entry.run_id}"}
    if entry.invocation_id:
        tokens.add(f"invocation:{entry.invocation_id}")
    refreshed = 0
    for task_id in task_ids:
        changed = False

        def project(existing, _incoming):
            nonlocal changed
            if not existing:
                return None
            fields = {"status": existing["status"]}
            debt = existing.get("delegated_runs_unreconciled")
            if isinstance(debt, list):
                fields["delegated_runs_unreconciled"] = [item for item in debt if item not in tokens]
                changed |= fields["delegated_runs_unreconciled"] != debt
            audit = existing.get("delegate_terminal_reconciliation")
            if isinstance(audit, dict):
                updated = dict(audit)
                for key, cleared in (
                    ("unreconciled", tokens), ("open_run_ids", {entry.run_id}),
                    ("undisposed_patch_run_ids", {entry.run_id}),
                    ("pending_invocation_ids", {entry.invocation_id} if entry.invocation_id else set()),
                ):
                    values = audit.get(key)
                    if isinstance(values, list):
                        updated[key] = [item for item in values if item not in cleared]
                fields["delegate_terminal_reconciliation"] = updated
                changed |= updated != audit
            return fields if changed else None

        try:
            stored = write_task_result(drive_root, task_id, "running", _field_projector=project,
                                       strict_existing_dict=True)
            if changed and isinstance(stored, dict) and not tokens.intersection(
                    stored.get("delegated_runs_unreconciled") or []):
                refreshed += 1
        except Exception:
            log.warning("Disposed custody disclosure refresh failed for %s", task_id, exc_info=True)
    return refreshed


__all__ = [
    "backfill_terminal_reconciliations",
    "custody_audit_snapshot",
    "keep_published_receipts",
    "record_terminal_reconciliation",
    "refresh_disposed_reconciliation",
    "refresh_recently_settled_terminals",
    "refresh_terminal_reconciliation",
    "terminal_custody_disclosure",
    "terminal_reconcile_task",
]


def _ack_current_receipts(root, tid, snapshot):
    from ouroboros import store
    from ouroboros.delegate_custody_current import _decode
    from ouroboros.task_results import require_writable_task_result_schema
    # One transaction over the result row and its closing receipts: a new debt cannot be
    # removed while the result still shows the old row.
    state = (snapshot or {}).get("state", {})
    with store.open_store(root, write=True) as txn:
        row = txn.load(tid)
        if not row:
            return
        require_writable_task_result_schema(row)
        facts = txn.get_row("delegated_runs", tid)
        if facts is None:
            return
        held = {rid: raw for rid, raw in facts.get("closed_runs", {}).items()
                if rid not in state or vars(_decode(raw)) != vars(state[rid])}
        if not held and not row.get("delegated_runs_unreconciled") and not facts.get("cost_revision"):
            txn.delete_row("delegated_runs", tid)
        elif {**facts, "closed_runs": held} != facts:  # an unchanged retained debt writes nothing
            txn.put_row("delegated_runs", tid, {**facts, "closed_runs": held})
        # The result itself is never rewritten here.


def _refresh_custody_cost(root, tid):
    """Pay only the observed revision, through the existing monetary projector.

    Failed writes, missing/live results and a concurrent custody change retain
    the same obligation for the next ordinary terminal sweep. No ledger write.
    """
    from ouroboros import store
    from supervisor.events_task_done import _refresh_terminal_task_cost
    from supervisor.queue import task_has_live_ownership
    from supervisor.task_ownership import TaskOwnershipRead

    try:
        with store.open_store(root, write=False) as txn:
            debt = txn.get_row("delegated_runs", tid) or {}
        revision = debt.get("cost_revision")
        if not revision or task_has_live_ownership(tid, ownership=TaskOwnershipRead(root)):
            return
        if not _refresh_terminal_task_cost(root, tid):
            return
        with store.open_store(root, write=True) as txn:
            latest = txn.get_row("delegated_runs", tid) or {}
            if latest.get("cost_revision") == revision:
                latest.pop("cost_revision")
                if latest.get("closed_runs"):
                    txn.put_row("delegated_runs", tid, latest)
                else:
                    txn.delete_row("delegated_runs", tid)
    except Exception:
        log.warning("Custody cost refresh remains owed for %s", tid, exc_info=True)
