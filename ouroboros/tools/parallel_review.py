"""One commit-gate review wave: every assigned seat, one two-part brief each.

There is no scope role (PR-3 B). The gate assembles ONE wave (``review
._prepare_unified_review``): every seat is asked by its ``parts`` — a packet
seat the change (contract A), a retrieving seat the change AND the eight
coupling questions in one brief (contract B), a coupling-only seat Part 2 of the
same brief. Quorum is ``adaptive_quorum(len(assigned))`` once; the coupling
question's answer lands in ``per_question.coupling`` and its history rides the
subject (``ctx._coupling_review_history``). The aggregate is reduced by ONE
function — ``review_ledger.reduce_verdict`` — for the gate and for the durable
record alike (§1.7 order: NOT_DISPATCHED → pending → QUORUM_FAILED →
NOT_PERFORMED by coupling → FAIL → PASS).

Q25-A ordering stays: every seat is PREPARED (packet fit-checked, briefs built)
before any seat is dispatched, so a deterministic assembly failure spends $0;
money admission prices the whole wave before the first paid call.
"""
from __future__ import annotations

import concurrent.futures as _cf
import contextvars
import copy
import hashlib
import json
import logging

from ouroboros.utils import run_cmd, utc_now_iso
from ouroboros.tools.review_helpers import format_review_history_entry, review_enforcement_blocks

log = logging.getLogger(__name__)


def _reserved_actor_row(slot, operation_id: str) -> dict:
    return {
        "slot_id": str(slot.slot_id or ""),
        "model_id": str(slot.model or ""),
        "route": str(getattr(slot.route, "value", slot.route) or ""),
        "effort": str(getattr(slot, "effort", "") or ""),
        "status": "in_flight",
        "operation_id": str(operation_id or ""),
        "operation_state": "in_flight",
        "late_result_pending": True,
    }


def _reserve_parallel_review_roster(ctx, prepared) -> None:
    """Reserve the wave's seats before the executor starts.

    The immutable operation-id map is process-local execution state. The full
    roster is attached to the caller. When the owner deadline has no dispatch
    window left, the roster remains an unpaid typed $0 wave; otherwise the
    existing paid write-ahead stamp records the wave atomically in the existing
    CommitAttemptRecord. A stamp failure propagates before any worker or
    provider POST can start.
    """
    from types import SimpleNamespace

    from ouroboros.observability import new_call_id
    from ouroboros.review_dispatch import slot_id_for_row, stamp_review_paid_on_dispatch

    rows = [
        copy.deepcopy(row)
        for row in list(getattr(ctx, "_triad_withheld_seat_records", []) or [])
        if isinstance(row, dict)
    ]
    operations = {"multi_model_review": {}}
    row_plan = (prepared or {}).get("row_plan") or {}
    models = list(row_plan.get("models") or [])
    routes = list(row_plan.get("routes") or [])
    efforts = list(row_plan.get("efforts") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    for index, model in enumerate(models):
        slot_id = str(slot_ids[index] if index < len(slot_ids) else "") or slot_id_for_row(index + 1)
        operation_id = new_call_id(f"commit_review_multi_model_review_{slot_id}")
        slot = SimpleNamespace(
            slot_id=slot_id,
            model=model,
            route=routes[index] if index < len(routes) else "api_chat",
            effort=efforts[index] if index < len(efforts) else "",
        )
        rows.append(_reserved_actor_row(slot, operation_id))
        operations["multi_model_review"][slot_id] = operation_id

    if not operations["multi_model_review"]:
        return
    ctx._review_reserved_operations = operations
    ctx._review_reserved_roster = {"multi_model_review": rows}
    from ouroboros.config import get_finalization_grace_sec
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context

    if owner_deadline_exhausted_for_context(ctx, reserve_sec=get_finalization_grace_sec()):
        return
    try:
        stamp_review_paid_on_dispatch(ctx)
    except Exception:
        # No worker can have started before the write-ahead stamp.  Do not let
        # the outer reconciliation turn this unsent process-local reservation
        # into durable custody_lost when the stamp itself failed.
        ctx._review_reserved_roster = None
        ctx._review_reserved_operations = {}
        raise


def _coupling_history_entry(outcome) -> dict:
    """One coupling round for the subject's history, preserving a non-PASS
    epistemic status (``not_performed`` is never read as a clean pass)."""
    def _names(findings):
        return "; ".join(f"{f['item']} ({f.get('obligation_id')})" if f.get("obligation_id") else f["item"]
                         for f in findings)

    parts = []
    if outcome.critical_findings:
        parts.append("Critical: " + _names(outcome.critical_findings))
    if outcome.advisory_findings:
        parts.append("Advisory: " + _names(outcome.advisory_findings))
    status = str(outcome.status or "not_performed")
    if not parts and status != "responded":
        summary = f"({status})"
    else:
        summary = " | ".join(parts) if parts else "(no findings)"
    seats = [s for s in outcome.seats if s.get("coverage") not in (None, "", "n/a", "not_asked")]
    if seats:
        summary += " | Read coverage (diagnostic): " + "; ".join(
            f"{s['slot_id']}: {s['coverage']}" for s in seats)
    return {
        "blocked": bool(outcome.blocked),
        "status": status,
        "verdict": str(outcome.verdict or ""),
        "summary": summary,
        "critical_findings": list(outcome.critical_findings),
        "advisory_findings": list(outcome.advisory_findings),
    }


def _format_coupling_advisory_msg(outcome) -> str:
    """The coupling question's findings as a readable message (advisory path)."""
    parts = []
    if outcome is not None and outcome.critical_findings:
        parts.append("Coupling findings (Part 2):\n" +
                     "\n".join(f"  • {f['item']}: {f.get('reason', '')}" for f in outcome.critical_findings))
    if outcome is not None and outcome.advisory_findings:
        parts.append("Coupling advisory notes (Part 2):\n" +
                     "\n".join(f"  • {f['item']}: {f.get('reason', '')}" for f in outcome.advisory_findings))
    return "---\n" + "\n".join(parts) if parts else ""


def _commit_review_retry_key(
    ctx, commit_message, *, goal, scope, review_rebuttal, binding_fingerprint="",
):
    """Bind one logical review cycle to canonical staged material and intent."""
    material = str(binding_fingerprint or "").strip()
    if not material:
        try:
            diff_bytes = run_cmd(
                ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
                cwd=ctx.repo_dir,
            ).encode()
            tree_sha = run_cmd(["git", "write-tree"], cwd=ctx.repo_dir).strip()
        except Exception:
            diff_bytes, tree_sha = b"", ""
        material = hashlib.sha256(tree_sha.encode() + b"\0" + diff_bytes).hexdigest()
    return "commit_review:" + hashlib.sha256(json.dumps({
        "binding": material, "commit_message": commit_message,
        "goal": goal, "scope": scope,
        "rebuttal": hashlib.sha256(str(review_rebuttal or "").encode()).hexdigest(),
        "contract": str(getattr(ctx, "_current_review_contract_fingerprint", "") or ""),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _structured_review_result(prepared, *, started_ts, retry_key, wave_refusal, exited, early, subject=None):
    """What this wave ASSIGNED and what each seat was GIVEN, as one typed mapping
    for the review ledger record (``review_ledger.build_wave_record``). With a
    frozen ``subject`` the mapping also names it (``subject``/``layer``), so the
    record's subject block is the frozen one, not the binding's.
    The gate decision reads nothing here; a failure to describe the wave is logged
    and leaves an empty mapping, never a changed verdict."""
    try:
        described = _describe_review_wave(prepared, started_ts=started_ts, retry_key=retry_key,
                                          wave_refusal=wave_refusal, exited=exited, early=early)
        if subject is not None:
            described.update(subject=subject.record_subject(), layer=subject.spec.layer)
        return described
    except Exception:
        log.warning("structured review result unavailable for the review ledger", exc_info=True)
        return {}


def _describe_review_wave(prepared, *, started_ts, retry_key, wave_refusal, exited, early):
    """The wave as one seat list with ``parts`` plus every distinct brief by sha:
    a packet seat's brief is the assembled prompt, a retrieving seat's its own
    two-part brief (``row_plan["brief_shas"]`` / ``prepared["brief_texts"]``)."""
    from ouroboros.review_ledger import PARTS
    from ouroboros.review_model_routes import adaptive_quorum
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    prepared = prepared or {}
    plan = dict(prepared.get("row_plan") or {})
    prompt = str(prepared.get("prompt") or "")
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else ""
    brief_texts = {str(k): str(v or "") for k, v in dict(prepared.get("brief_texts") or {}).items()}
    if prompt_sha:
        brief_texts[prompt_sha] = prompt

    def _at(key, i, default=""):
        values = list(plan.get(key) or [])
        value = values[i] if i < len(values) else default
        return value if isinstance(value, (tuple, list, bool)) else str(getattr(value, "value", value) or "")

    rows = []
    for i in range(len(plan.get("slot_ids") or [])):
        retrieves = row_plan_retrieves(plan, i)
        parts = [p for p in PARTS if p in tuple(_at("parts", i, ()) or ())] or ["change"]
        rows.append({
            "slot_id": _at("slot_ids", i), "model": _at("models", i), "route": _at("routes", i),
            "effort": _at("efforts", i), "session_target": _at("session_targets", i),
            "session_profile": _at("session_profiles", i), "subagent_id": _at("subagent_ids", i),
            "retrieves": retrieves, "parts": parts, "additional": bool(_at("additional", i, False)),
            "brief_sha": str(_at("brief_shas", i) or "") if retrieves else prompt_sha,
        })
    coupling_shas = sorted({r["brief_sha"] for r in rows if "coupling" in r["parts"] and r["brief_sha"]})
    assigned = [r for r in rows if not r["additional"]]
    return {
        "started_ts": started_ts, "retry_key": retry_key, "wave_refusal": str(wave_refusal or ""),
        "rows": rows, "quorum": adaptive_quorum(len(assigned)) if assigned else 0,
        "brief": {"change_prompt_sha": prompt_sha, "coupling_brief_sha": coupling_shas[0] if len(coupling_shas) == 1 else "",
                  "coupling_brief_shas": coupling_shas},
        "brief_texts": brief_texts,
        "assembly_refusal": str(early or "") if exited and early else "",
        "retrieving_manifests": list(prepared.get("retrieving_manifests") or []),
    }


def run_parallel_review(
    ctx, commit_message, *, goal="", scope="", review_rebuttal="",
    review_binding_fingerprint="", subject=None,
):
    """Run the commit gate's one review wave against the staged diff.

    Q25-A ordering: every seat is prepared (the packet fit-checked, every
    retrieving seat's brief built) BEFORE any seat is dispatched, so a
    deterministic assembly failure spends $0; the wave is money-admitted as a
    whole; then every seat is dispatched together.

    Returns ``(review_err, coupling_result, block_reason, advisory)`` —
    ``coupling_result`` is the coupling question's outcome mapping
    (``review_ledger.coupling_outcome``), ``None`` when no seat was asked it.

    ``subject`` is a frozen review subject (``review_subject.FrozenSubject``):
    the wave then reads ITS diff and trees, its governance root (always the
    installed body) and its retry identity, and the ledger record names it.
    ``None`` is the gate's path unchanged — the context's live staged index."""
    from ouroboros.tools.review import _dispatch_unified_review, _prepare_unified_review
    if bool(getattr(ctx, "_review_reconcile_only", False)):
        from ouroboros.review_custody import prepare_frozen_review_reconciliation

        prepare_frozen_review_reconciliation(ctx, getattr(ctx, "_pending_review_attempt", None))

    # Reset forensic fields so prior attempts cannot bleed into early exits.
    ctx._last_triad_raw_results = []
    ctx._last_coupling_result = None
    ctx._last_review_verdict = {}
    ctx._last_review_structured = {}
    _started_ts, wave_refusal = utc_now_iso(), None
    # Managed subject↔binding assertion input: every gate subject built during
    # THIS attempt records its S tree here; the commit gate then asserts the
    # set equals the binding fingerprint's tree_sha (typed failure otherwise).
    ctx._last_review_subject_trees = set()
    # The per-attempt managed-subject memo resets with the same boundary (C5).
    ctx._managed_review_subject_memo = {}

    if subject is not None:
        from ouroboros.tools.review_subject import review_retry_key

        snapshot_digest = subject.diff_sha
        # Identity (c): the subject's retry key is set BEFORE any dispatch, so a
        # custody rejoin after a crash finds the same physical review and never
        # pays for it twice. A caller's own key (the gate's) is kept.
        retry_key = str(getattr(ctx, "_current_review_retry_key", "") or "") or review_retry_key(subject)
        ctx._current_review_retry_key = retry_key
    else:
        try:
            diff_bytes = run_cmd(
                ["git", "diff", "--cached", "--binary", "--no-ext-diff"], cwd=ctx.repo_dir,
            ).encode()
        except Exception:
            diff_bytes = b""
        snapshot_digest = hashlib.sha256(diff_bytes).hexdigest()
        retry_key = str(getattr(ctx, "_current_review_retry_key", "") or "") or (
            _commit_review_retry_key(
                ctx, commit_message, goal=goal, scope=scope,
                review_rebuttal=review_rebuttal,
                binding_fingerprint=review_binding_fingerprint,
            )
        )
    snapshot_key = snapshot_digest[:16]
    _stored = getattr(ctx, "_coupling_review_history", None) or {}
    coupling_history = _stored.get(snapshot_key, []) if isinstance(_stored, dict) else []
    # The brief builder reads this subject's prior coupling rounds from the context.
    ctx._coupling_review_history_rounds = list(coupling_history)

    # Snapshot advisory state before assembly and dispatch mutate it.
    _advisory_snapshot_before = list(getattr(ctx, "_review_advisory", []))

    # ---- Phase 1 (Q25=A): prepare every seat; dispatch NOTHING yet. ----
    prepared, early, exited = None, None, True
    try:
        prepared, early, exited = _prepare_unified_review(
            ctx, commit_message, review_rebuttal=review_rebuttal, goal=goal, scope=scope, subject=subject)
        if prepared is not None:
            prepared["retry_key"] = retry_key
    except Exception as e:
        log.warning("Review assembly raised unexpected exception: %s", e)
        early = f"⚠️ REVIEW_BLOCKED: Review assembly crashed — {e}\nFix the issue and retry."
        ctx._last_review_block_reason = "infra_failure"
        ctx._last_review_critical_findings = []

    review_err = early if exited else None
    if not exited:
        # ---- Money admission (owner decision 2026-09-05, on the known-spend
        # rule of #1487): before ANY seat is dispatched, known spend must be
        # below every fence; otherwise every seat is a typed $0 not_dispatched
        # record and the gate blocks naming the fence. The wave's summed seat
        # bounds are disclosure, never an earlier refusal. ----
        from ouroboros.tools.review_admission import admit_commit_gate_wave, commit_gate_paid_seats

        if not bool(getattr(ctx, "_review_reconcile_only", False)):
            try:
                wave_refusal = admit_commit_gate_wave(ctx, commit_gate_paid_seats(prepared, exited))
            except Exception as e:
                # Fail-open is the enforcement choice (as review_wave_budget_gate's
                # own), but "admitted" and "admission crashed" are different
                # facts: the wave dispatches unadmitted, and says so once, typed.
                from ouroboros.tools.review_helpers import emit_review_event
                log.warning("commit-gate wave admission unavailable (%s: %s); the wave dispatches unadmitted",
                            type(e).__name__, e)
                emit_review_event(ctx, {
                    "type": "review_wave_admission_unavailable", "surface": "commit_gate",
                    "task_id": str(getattr(ctx, "task_id", "") or ""),
                    "error": f"{type(e).__name__}: {e}",
                })
                wave_refusal = None
        if wave_refusal is not None:
            from ouroboros.tools.review import _handle_review_block_or_warning
            from ouroboros.tools.review_admission import triad_not_dispatched_records

            blocking_review = bool(prepared.get("blocking_review")) and review_enforcement_blocks("blocking")
            if not hasattr(ctx, "_review_degraded_reasons"):
                ctx._review_degraded_reasons = []
            ctx._review_degraded_reasons.append(
                "review_not_dispatched_budget_admission: known spend has reached a budget "
                "fence of the commit-gate wave, so no seat was dispatched ($0 spent)"
            )
            ctx._last_review_critical_findings = []
            ctx._last_review_block_reason = "review_wave_budget_insufficient"
            from ouroboros.review_ledger import CouplingOutcome

            ctx._last_coupling_result = CouplingOutcome(status="not_dispatched")
            ctx._last_triad_raw_results = list(
                getattr(ctx, "_triad_withheld_seat_records", []) or []
            ) + triad_not_dispatched_records(prepared.get("row_plan") or {}, wave_refusal)
            review_err = _handle_review_block_or_warning(
                ctx, blocking_review, wave_refusal,
                "Review enforcement=Advisory: the commit-gate review wave was declined "
                "before dispatch (budget fence); commit proceeding without review. ",
            )
        else:
            # ---- Phase 2: submit the prepared wave to the executor. ----
            try:
                if not bool(getattr(ctx, "_review_reconcile_only", False)):
                    _reserve_parallel_review_roster(ctx, prepared)
            except Exception as e:
                log.warning("Commit review custody reservation failed: %s", e)
                ctx._last_review_block_reason = "infra_failure"
                ctx._last_review_critical_findings = []
                review_err = (
                    "⚠️ REVIEW_BLOCKED: durable review custody could not be reserved "
                    f"before dispatch — {e}\nNo reviewer was started; fix the state write and retry."
                )
            else:
                with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                    # The wave runs under a COPY of the admitting context
                    # (contextvars.copy_context, the loop_tool_execution and
                    # plan_review precedent): the usage scope the wave was
                    # admitted with — its bound root fence included — is the
                    # one every seat's reserve_attempt binds, so admission and
                    # reservation share one fence even after a mid-turn
                    # settings reload changed the environment's number.
                    future = pool.submit(contextvars.copy_context().run, _dispatch_unified_review,
                                         ctx, commit_message, prepared)
                    try:
                        review_err = future.result()
                    except Exception as e:
                        log.warning("Review dispatch raised unexpected exception: %s", e)
                        review_err = f"⚠️ REVIEW_BLOCKED: Review dispatch crashed — {e}\nFix the issue and retry."
                        ctx._last_review_block_reason = "infra_failure"
                        ctx._last_review_critical_findings = []
    block_reason = getattr(ctx, "_last_review_block_reason", "critical_findings")
    advisory_post = list(getattr(ctx, "_review_advisory", []))
    advisory = [a for a in advisory_post if a not in _advisory_snapshot_before]

    coupling_result = _record_coupling_outcome(ctx, prepared, snapshot_key, coupling_history)
    ctx._last_review_structured = _structured_review_result(
        prepared, started_ts=_started_ts, retry_key=retry_key,
        wave_refusal=wave_refusal, exited=exited, early=early, subject=subject)
    return review_err, coupling_result, block_reason, advisory


def _record_coupling_outcome(ctx, prepared, snapshot_key, coupling_history):
    """The coupling question's outcome of this wave, appended to the subject's
    coupling history (``ctx._coupling_review_history[snapshot]``); ``None`` when
    no seat of the wave was asked Part 2."""
    from ouroboros.review_ledger import CouplingOutcome

    plan = (prepared or {}).get("row_plan") or {}
    asked = any("coupling" in tuple(p or ()) for p in plan.get("parts") or [])
    outcome = getattr(ctx, "_last_coupling_result", None)
    if not isinstance(outcome, CouplingOutcome):
        if not asked:
            return None
        outcome = CouplingOutcome()
    existing = getattr(ctx, "_coupling_review_history", None) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing[snapshot_key] = list(coupling_history) + [_coupling_history_entry(outcome)]
    ctx._coupling_review_history = existing
    return outcome


def aggregate_review_verdict(review_err, coupling_result, block_reason, advisory,
                             ctx, commit_message, commit_start, repo_dir):
    """The wave's one verdict as the gate's block state plus advisory items.

    The aggregate itself was reduced in ``_dispatch_unified_review`` by
    ``review_ledger.reduce_verdict`` (the coupling question is one of its
    per-question verdicts, never a second gate); this projects it onto the
    caller's ``(blocked, message, block_reason, findings, coupling_items)``
    contract and applies the owner's enforcement authority."""
    coupling_items = []
    for severity, key in (("critical", "critical_findings"), ("advisory", "advisory_findings")):
        for f in (getattr(coupling_result, key, None) or []):
            item = {"severity": severity, "tag": "coupling", "item": str(f.get("item", "") or ""),
                    "reason": str(f.get("reason", "") or ""), "verdict": "FAIL"}
            if f.get("obligation_id"):
                item["obligation_id"] = str(f.get("obligation_id"))
            coupling_items.append(item)

    findings = list(getattr(ctx, "_last_review_critical_findings", []) or []) if review_err else []
    if not review_err:
        return False, None, "", findings, coupling_items

    combined_msg = review_err
    coupling_note = _format_coupling_advisory_msg(coupling_result)
    if coupling_note and block_reason not in ("critical_findings",):
        combined_msg += f"\n\n{coupling_note}"
    if advisory and block_reason not in ("critical_findings",):
        adv_text = "\n".join(f"  ⚠️ Advisory: {format_review_history_entry(a)}" for a in advisory)
        combined_msg += f"\n\n---\nAdvisory findings:\n{adv_text}"

    from ouroboros.config import get_review_enforcement

    cyber = not review_enforcement_blocks("blocking")
    if cyber or (get_review_enforcement() == "advisory" and block_reason == "fixed_overflow"):
        from ouroboros.tools.review import _record_advisory_override

        disclosure = (
            ("Cyber Pro: independent review does not prohibit action " if cyber else
             "Review enforcement=advisory: technical review failure permits continuing ")
            + "on the independently bound candidate; failed or missing review is not a PASS.\n"
            + combined_msg
        )
        ctx._last_review_block_reason = block_reason
        _record_advisory_override(ctx, disclosure)
        ctx._review_advisory.append(disclosure)
        ctx._review_degraded_reasons = list(getattr(ctx, "_review_degraded_reasons", []) or []) + [
            "review_cyber_authority" if cyber else "review_technical_failure_advisory"]
        return False, combined_msg, block_reason, findings, coupling_items

    return True, combined_msg, block_reason, findings, coupling_items
