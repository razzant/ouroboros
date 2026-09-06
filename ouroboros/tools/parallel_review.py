"""Parallel triad + scope review orchestration for commit gates."""
from __future__ import annotations

import concurrent.futures as _cf
import contextvars
import copy
import hashlib
import json
import logging
import time

from ouroboros.utils import run_cmd
from ouroboros.review_substrate import scope_reviewer_slots
from ouroboros.tools.review_helpers import build_scope_actor_record, format_review_history_entry
from ouroboros.tools.scope_review import (
    run_scope_review,
    ScopeReviewResult,
    _get_scope_model,
)

log = logging.getLogger(__name__)


def _route_value(route) -> str:
    return str(getattr(route, "value", route) or "")


def _reserved_actor_row(slot, operation_id: str) -> dict:
    return {
        "slot_id": str(slot.slot_id or ""),
        "model_id": str(slot.model or ""),
        "route": _route_value(slot.route),
        "effort": str(getattr(slot, "effort", "") or ""),
        "status": "in_flight",
        "operation_id": str(operation_id or ""),
        "operation_state": "in_flight",
        "late_result_pending": True,
    }


def _reserve_parallel_review_roster(ctx, triad_prepared, scope_rows) -> None:
    """Reserve both commit-review surfaces before either executor pool starts.

    The immutable operation-id map is process-local execution state.  The full
    roster is attached to the caller.  When the owner deadline has no
    dispatch window left, the roster remains an unpaid typed $0 wave; otherwise
    the existing paid write-ahead stamp records both surfaces atomically in the
    existing CommitAttemptRecord.  A stamp failure propagates before any
    worker or provider POST can start.
    """
    from types import SimpleNamespace

    from ouroboros.observability import new_call_id
    from ouroboros.review_dispatch import slot_id_for_row, stamp_review_paid_on_dispatch

    triad_rows = [
        copy.deepcopy(row)
        for row in list(getattr(ctx, "_triad_withheld_seat_records", []) or [])
        if isinstance(row, dict)
    ]
    operations = {"multi_model_review": {}, "scope_review": {}}
    row_plan = (triad_prepared or {}).get("row_plan") or {}
    models = list(row_plan.get("models") or [])
    routes = list(row_plan.get("routes") or [])
    efforts = list(row_plan.get("efforts") or [])
    slot_ids = list(row_plan.get("slot_ids") or [])
    for index, model in enumerate(models):
        slot_id = str(slot_ids[index] if index < len(slot_ids) else "") or slot_id_for_row(
            index + 1,
        )
        operation_id = new_call_id(f"commit_review_multi_model_review_{slot_id}")
        slot = SimpleNamespace(
            slot_id=slot_id,
            model=model,
            route=routes[index] if index < len(routes) else "api_chat",
            effort=efforts[index] if index < len(efforts) else "",
        )
        triad_rows.append(_reserved_actor_row(slot, operation_id))
        operations["multi_model_review"][slot_id] = operation_id

    scope_actor_rows = []
    for row in scope_rows:
        slot = row["slot"]
        final = row.get("final")
        if final is not None:
            scope_actor_rows.append(build_scope_actor_record(
                final,
                fallback_model_id=getattr(final, "model_id", "") or slot.model,
                slot_id=slot.slot_id,
            ))
            continue
        operation_id = new_call_id(f"commit_review_scope_review_{slot.slot_id}")
        scope_actor_rows.append(_reserved_actor_row(slot, operation_id))
        operations["scope_review"][str(slot.slot_id or "")] = operation_id

    if not any(operations[surface] for surface in operations):
        return
    ctx._review_reserved_operations = operations
    ctx._review_reserved_roster = {
        "multi_model_review": triad_rows,
        "scope_review": scope_actor_rows,
    }
    from ouroboros.config import get_finalization_grace_sec
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context

    if owner_deadline_exhausted_for_context(
        ctx, reserve_sec=get_finalization_grace_sec(),
    ):
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


def _scope_history_entry(scope_result) -> dict:
    """Build scope history while preserving non-PASS epistemic status."""
    parts = []
    if scope_result.critical_findings:
        parts.append(
            "Critical: " + "; ".join(
                (
                    f"{f['item']} ({f.get('obligation_id')})"
                    if f.get("obligation_id") else f["item"]
                )
                for f in scope_result.critical_findings
            )
        )
    if scope_result.advisory_findings:
        parts.append(
            "Advisory: " + "; ".join(
                (
                    f"{f['item']} ({f.get('obligation_id')})"
                    if f.get("obligation_id") else f["item"]
                )
                for f in scope_result.advisory_findings
            )
        )
    status = getattr(scope_result, "status", None) or "responded"
    # Lead with non-responded status so empty findings are not misread as PASS.
    if not parts and status not in ("responded",):
        summary = f"({status})"
    else:
        summary = " | ".join(parts) if parts else "(no findings)"
    return {
        "blocked": scope_result.blocked,
        "status": status,
        "summary": summary,
        "critical_findings": scope_result.critical_findings or [],
        "advisory_findings": scope_result.advisory_findings or [],
    }


def _format_scope_advisory_msg(scope_result) -> str:
    """Format advisory scope findings as a readable message (advisory enforcement path)."""
    parts = []
    if scope_result.critical_findings:
        parts.append("Scope advisory findings (enforcement=advisory):\n" +
                     "\n".join(f"  • {f['item']}: {f.get('reason', '')}"
                                for f in scope_result.critical_findings))
    if scope_result.advisory_findings:
        parts.append("Scope advisory notes:\n" +
                     "\n".join(f"  • {f['item']}: {f.get('reason', '')}"
                                for f in scope_result.advisory_findings))
    return "---\n" + "\n".join(parts) if parts else ""
def _scope_not_dispatched_result(slot, reason: str = ""):
    """Typed $0 placeholder for a prepared scope row the admission never
    dispatched; ``reason`` names WHICH admission withheld it (default: the
    Q25-A assembly block)."""
    return ScopeReviewResult(
        blocked=False,
        block_message="",
        model_id=slot.model,
        status="not_dispatched",
        advisory_findings=[{
            "verdict": "FAIL", "severity": "advisory",
            "item": "scope_row_not_dispatched",
            "reason": reason or (
                "assembly-before-dispatch admission (Q25=A): the commit gate was "
                "already deterministically blocked at packet assembly, so this "
                "row was not dispatched ($0 spent)."
            ),
        }],
    )


def _reservation_ids_now() -> frozenset:
    """Attempt ids of the reservations the root telemetry holds right now (the
    baseline a scope-first hold compares against; empty outside a usage scope)."""
    from ouroboros.usage_accounting import current_usage_scope, last_root_accounting

    root = str(getattr(current_usage_scope(), "root_task_id", "") or "")
    rows = (last_root_accounting(root) or {}).get("reservations") if root else None
    return frozenset(str(r.get("attempt_id") or "") for r in (rows or []))


def _await_scope_reservation(ctx, scope_future, seats, started_monotonic: float,
                            known_ids: frozenset = frozenset()) -> None:
    """Hold the triad until a scope seat's OWN reservation is on the ledger
    (owner decision 2026-09-05: scope reserves FIRST): the identity (usage
    category + review slot) of a row ``reserve_attempt`` APPENDED for this root
    after the wave started, read from the ledger's process-local root
    telemetry — a refresh, a settlement or a refused reservation never leaves
    one, so none of them can release the triad early. Bounded by the scope
    future and ``NESTED_SETTLEMENT_MARGIN_SEC`` (a structural ordering margin,
    not a timeout contract); a hold that ends WITHOUT observing the scope
    reservation is a typed ``review_scope_lead_unobserved`` event, never a
    silent fall-through. The reservation must be THIS task's (``ctx.task_id``,
    the id the substrate stamps on the row): a sibling task's scope seat under
    the same root never releases it. ``known_ids`` are the reservation attempt
    ids already on the root telemetry when the wave started: only a row NOT in
    it is this wave's (an identity check, not a clock comparison — Windows'
    monotonic tick is ~15.6 ms, so "reserved after the start" is not decidable
    by time). No paid scope seat or no root = no wait."""
    scope_slots = {seat["slot_id"] for seat in seats if seat["surface"] == "scope_review"}
    if scope_future is None or not scope_slots:
        return
    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC
    from ouroboros.review_substrate import review_usage_category
    from ouroboros.tools.review_helpers import emit_review_event
    from ouroboros.usage_accounting import current_usage_scope, last_root_accounting

    root_task_id = str(getattr(current_usage_scope(), "root_task_id", "") or "")
    task_id = str(getattr(ctx, "task_id", "") or "")
    category = review_usage_category("scope_review")
    deadline = started_monotonic + float(NESTED_SETTLEMENT_MARGIN_SEC)
    while root_task_id:
        now = time.monotonic()
        for row in (last_root_accounting(root_task_id) or {}).get("reservations") or []:
            # Category + slot + time + THIS task: a sibling task under the same
            # root running its own gate reserves a same-named scope slot too.
            if (str(row.get("category") or "") == category
                    and str(row.get("review_slot_id") or "") in scope_slots
                    and str(row.get("task_id") or "") == task_id
                    and str(row.get("attempt_id") or "") not in known_ids):
                return
        scope_done = bool(scope_future.done())
        if scope_done or now >= deadline:
            emit_review_event(ctx, {
                "type": "review_scope_lead_unobserved",
                "task_id": str(getattr(ctx, "task_id", "") or ""), "root_task_id": root_task_id,
                "scope_slot_ids": sorted(scope_slots), "scope_seat_done": scope_done,
                "margin_sec": float(NESTED_SETTLEMENT_MARGIN_SEC),
            })
            log.warning("no scope seat reservation observed (%s); the triad proceeds without the scope lead",
                        "scope seat finished" if scope_done else f"margin {NESTED_SETTLEMENT_MARGIN_SEC}s")
            return
        time.sleep(0.05)


def _prepare_scope_rows(ctx, commit_message, *, goal, scope, review_rebuttal,
                        history_snapshot, scope_history):
    """Phase 1 of the Q25-A admission: assemble EVERY configured scope row's
    packet without dispatching any reviewer. Returns aligned row dicts
    ``{slot, prepared, final}`` (exactly one of prepared/final per row).

    Q28-A oversized outcome: packet limits gate only the api subset — when the
    panel's agent-session rows alone satisfy the quorum, a fit-blocked api row
    YIELDS its seat (typed, loud, preserved as an advisory finding) instead of
    blocking the whole panel; a panel that cannot reach quorum without its api
    rows keeps the deterministic block (a typed zero-spend terminal upstream).

    Identity of every scope row comes from the one SSOT that owns it, so the
    actor record and the substrate call agree on which row spoke. No
    except/fallback here BY DESIGN: any failure to read the configured scope
    rows must surface (the caller converts it into the same blocked result the
    dispatch path always produced)."""
    from ouroboros.config import adaptive_quorum
    from ouroboros.tools.review_admission import (
        SCOPE_FIT_BLOCK_STATUSES,
        prepare_scope_review,
    )

    scope_slots = list(scope_reviewer_slots())
    ctx._last_scope_model = ",".join(slot.model for slot in scope_slots)
    rows = []
    for slot in scope_slots:
        prepared, final = prepare_scope_review(
            ctx, commit_message, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal,
            review_history=history_snapshot,
            scope_review_history=scope_history,
            scope_model=slot.model,
            slot_id=slot.slot_id,
            route=slot.route,
            slot_effort=slot.effort,
            session_target=slot.session_target,
            session_profile=getattr(slot, "session_profile", ""),
            subagent_id=getattr(slot, "subagent_id", ""),
        )
        rows.append({"slot": slot, "prepared": prepared, "final": final})
    # Only a LIVE retrieving row (prepared, still to be dispatched) can supply
    # the verdict the yield leans on: one that already terminated at assembly
    # (final is not None) is a dead seat and must not count. RETRIEVES class:
    # session rows and configured-subagent api rows both retrieve (Q28-A).
    session_rows = sum(
        1 for row in rows
        if row["final"] is None
        and bool(getattr(
            row["slot"], "retrieves",
            str(getattr(row["slot"].route, "value", row["slot"].route) or "")
            == "agent_session",
        ))
    )
    if session_rows >= adaptive_quorum(len(rows)):
        for row in rows:
            final = row["final"]
            if (
                final is not None and final.blocked
                and str(getattr(final, "status", "")) in SCOPE_FIT_BLOCK_STATUSES
            ):
                slot = row["slot"]
                note = (
                    f"scope api row {slot.slot_id or slot.model} could not receive its "
                    f"packet ({final.status}); the panel's {session_rows} live "
                    "agent-session row(s) satisfy the quorum and proceed (Q28-A)"
                )
                log.warning("%s", note)
                # The row's PRE-YIELD advisories asserted a blocking terminal
                # ("no authoritative verdict" / remedies): once the seat
                # yields, those assertions are no longer true — supersede them
                # in place instead of leaving them beside the yield note.
                superseded = []
                for finding in (final.advisory_findings or []):
                    finding = dict(finding)
                    finding["reason"] = (
                        "[superseded by the Q28-A session-quorum yield — this "
                        "api row's seat yielded; the refusal below no longer "
                        f"blocks] {finding.get('reason', '')}"
                    )
                    superseded.append(finding)
                final.advisory_findings = superseded + [{
                    "verdict": "FAIL", "severity": "advisory",
                    "item": "scope_api_row_oversize_yielded",
                    "reason": f"⚠️ {note}. Original refusal: {final.block_message}",
                }]
                final.blocked = False
                final.block_message = ""
    return rows


def _run_scope(ctx, commit_message, scope_rows, dispatch, *, goal, scope,
               review_rebuttal, history_snapshot, scope_history, retry_key="",
               withheld_reason=""):
    """Dispatch (or, on an admission block, typed-placeholder) every scope row
    and aggregate the panel verdict — the dispatch half of the Q25-A split.
    ``dispatch=False`` renders prepared rows as $0 not_dispatched placeholders:
    by default the gate was already deterministically blocked at assembly;
    ``withheld_reason`` names another pre-dispatch admission (the wave budget)."""
    try:
        def _run_one_scope(row):
            # P3 one-pass contract: one substantive scope call per configured
            # actor. Transport retry remains inside the same review substrate;
            # never launch an automatic second degraded review call here.
            # The row's configured delivery rides with it (5.3: same task,
            # same criteria, same output contract — only delivery differs;
            # adaptive_quorum below is delivery-blind and unchanged).
            if row["final"] is not None:
                return row["final"]
            if not dispatch:
                return _scope_not_dispatched_result(row["slot"], withheld_reason)
            slot = row["slot"]
            return run_scope_review(
                ctx, commit_message, goal=goal, scope=scope,
                review_rebuttal=review_rebuttal,
                review_history=history_snapshot,
                scope_review_history=scope_history,
                scope_model=slot.model,
                slot_id=slot.slot_id,
                route=slot.route,
                slot_effort=slot.effort,
                session_target=slot.session_target,
                session_profile=getattr(slot, "session_profile", ""),
                prepared=row["prepared"],
                retry_key=retry_key,
            )

        scope_slots = [row["slot"] for row in scope_rows]
        scope_models = [slot.model for slot in scope_slots]
        with _cf.ThreadPoolExecutor(max_workers=min(len(scope_slots), 4)) as scope_pool:
            # copy_context (the loop_tool_execution precedent): the admitting
            # usage scope — and its bound root fence — reaches each row's
            # substrate; a bare pool thread would re-read the fence from the
            # environment and reserve against a different number.
            futures = [scope_pool.submit(contextvars.copy_context().run, _run_one_scope, row)
                       for row in scope_rows]
            results = [future.result() for future in futures]
        ctx._last_scope_raw_results = [
            build_scope_actor_record(
                result,
                fallback_model_id=getattr(result, "model_id", "") or slot.model,
                slot_id=slot.slot_id,
            )
            for result, slot in zip(results, scope_slots)
        ]
        # Reviewer-slot SSOT applies to scope too (Bible P3): a single configured
        # scope reviewer is honored but recorded as loud durable degraded-trust,
        # and a configured>=2-but-<quorum-responded scope run must never silently
        # pass on "any responded". Only an authoritative `responded` actor counts
        # toward quorum; a context-floor row is not an authoritative responder and
        # is left out of the count because its OWN authority function already
        # blocked (it is not counted out in order to let it pass).
        from ouroboros.config import adaptive_quorum
        _scope_statuses = [str(getattr(r, "status", "") or "") for r in results]
        # `responded` is the ONLY authoritative status. A retrieving (session) row
        # whose window is not sourced-proven arrives as `session_advisory`: its
        # window evidence — not its retrieval — is what is missing, so it must not be counted as
        # the authoritative verdict that satisfies the blocking scope quorum — it is
        # advisory evidence, and the shortfall it leaves is disclosed below. Such a
        # row also arrives BLOCKED (its own authority function decides that, exactly
        # as the api row's `sub_floor` twin does), so counting it out of the quorum
        # here can no longer let the gate fail open.
        _responded = sum(1 for s in _scope_statuses if s == "responded")
        _session_advisory = sum(1 for s in _scope_statuses if s == "session_advisory")
        _required = adaptive_quorum(len(scope_models))
        _single_scope_reviewer = len(scope_models) == 1
        # An all-not_dispatched panel is NOT a quorum failure: the gate was
        # already deterministically blocked at assembly and every row was a $0
        # typed placeholder by design — a "diversity was not achieved" advisory
        # would misread that as a degraded review that ran.
        _all_not_dispatched = bool(_scope_statuses) and all(
            s == "not_dispatched" for s in _scope_statuses
        )
        _scope_degraded: list = []
        if _session_advisory:
            _scope_degraded.append(
                f"scope_session_advisory_only: {_session_advisory} retrieving row(s) "
                "carried no authoritative verdict (window not sourced-proven)"
            )
        if _single_scope_reviewer:
            _scope_degraded.append("single_reviewer_no_diversity")
        elif _all_not_dispatched:
            _scope_degraded.append(
                f"scope_not_dispatched_budget_admission: no scope row was dispatched ($0 spent): {withheld_reason}"
                if withheld_reason else
                "scope_not_dispatched_assembly_block: no scope row was dispatched "
                "(the commit gate was already deterministically blocked at packet "
                "assembly; $0 spent)"
            )
        elif _responded < _required and not any(getattr(r, "blocked", False) for r in results):
            _scope_degraded.append(
                f"scope_quorum_not_met: responded={_responded} < required={_required}"
            )
        _scope_quorum_manifest = {
            "scope_responded_count": _responded,
            "scope_required_quorum": _required,
            "single_reviewer_no_diversity": _single_scope_reviewer,
            "scope_session_advisory_only_count": _session_advisory,
            "scope_degraded_reasons": _scope_degraded,
        }
        if len(results) == 1:
            only = results[0]
            only.context_manifest = {**(getattr(only, "context_manifest", {}) or {}), **_scope_quorum_manifest}
            return only
        critical = []
        advisory = []
        parsed_items = []
        blocked_messages = []
        statuses = []
        for result in results:
            statuses.append(getattr(result, "status", ""))
            critical.extend(result.critical_findings or [])
            advisory.extend(result.advisory_findings or [])
            parsed_items.extend(getattr(result, "parsed_items", []) or [])
            if result.blocked and result.block_message:
                blocked_messages.append(result.block_message)
        blocked = bool(blocked_messages)
        block_messages = list(blocked_messages)
        _qmsg = (
            f"⚠️ SCOPE_QUORUM_NOT_MET: only {_responded} of {len(scope_models)} configured "
            f"scope reviewers returned an authoritative verdict (adaptive quorum {_required}). "
            "Cross-model scope diversity was not achieved this run."
        )
        # Bible P3 negative control: configured>=2 but a PARTIAL authoritative
        # quorum (0 < responded < required) is a loud quorum FAILURE — block vs
        # advisory FOLLOWS owner enforcement (never hardcode a block). A
        # zero-responded run is NOT decided here: each delivery's own authority
        # function already returns a BLOCKING result when its window cannot
        # authorise (api `sub_floor`, retrieving `session_advisory`). Measured, the
        # ONLY non-blocking scope status is `skipped_low_context_mode`: a
        # budget_exceeded PACK arrives here as a BLOCKING `sub_floor` row, and no
        # ScopeReviewResult carries status "budget_exceeded" at all. Widening this
        # condition to `_responded < _required` would make this aggregate a SECOND
        # owner of that decision and would turn the owner-declared low-context skip
        # into a block — so the fix for a fail-open row belongs in the row, not here.
        partial_quorum_shortfall = (
            not _single_scope_reviewer and 0 < _responded < _required and not blocked
        )
        if partial_quorum_shortfall:
            from ouroboros.config import get_review_enforcement
            if get_review_enforcement() == "blocking":
                blocked = True
                block_messages.append(_qmsg)
        # Surface any non-blocking shortfall LOUDLY (advisory, never a silent
        # clean pass) and persist it in the manifest below. An all-not_dispatched
        # panel is excluded: each of its rows already carries the typed
        # scope_row_not_dispatched advisory, and the quorum message would be
        # false (no reviewer ran to fall short of diversity).
        if (
            _scope_degraded and not _single_scope_reviewer and not blocked
            and not _all_not_dispatched
        ):
            advisory.append({
                "verdict": "FAIL",
                "severity": "advisory",
                "item": "scope_quorum_not_met",
                "reason": _qmsg,
            })
        return ScopeReviewResult(
            blocked=blocked,
            block_message="\n\n".join(block_messages),
            critical_findings=critical,
            advisory_findings=advisory,
            parsed_items=parsed_items,
            raw_text="\n\n".join(str(r.raw_text or "") for r in results),
            model_id=",".join(scope_models),
            # Quorum-aware: only an authoritative quorum yields "responded".
            # A partial quorum (some — but <required — responded) is a loud
            # "degraded_quorum"; zero responded preserves the joined raw
            # statuses so downstream budget_exceeded/skipped detection holds.
            status=(
                "blocked" if blocked
                else "responded" if _responded >= _required
                else "degraded_quorum" if _responded > 0
                else ",".join(statuses)
            ),
            prompt_chars=sum(int(r.prompt_chars or 0) for r in results),
            tokens_in=sum(int(r.tokens_in or 0) for r in results),
            tokens_out=sum(int(r.tokens_out or 0) for r in results),
            cost_usd=sum(float(r.cost_usd or 0.0) for r in results),
            context_manifest={
                "scope_models": scope_models,
                "actor_count": len(results),
                **_scope_quorum_manifest,
                "actors": [
                    {
                        "slot_id": slot.slot_id,
                        "model": slot.model,
                        "context_manifest": getattr(result, "context_manifest", {}) or {},
                    }
                    for result, slot in zip(results, scope_slots)
                ],
            },
        )
    except Exception as e:
        log.warning("Scope review raised unexpected exception: %s", e)
        result = ScopeReviewResult(
            blocked=True,
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: Scope review failed — {e}\nFix the issue and retry.",
            model_id=getattr(ctx, "_last_scope_model", "") or _get_scope_model(),
            status="error",
        )
        ctx._last_scope_raw_results = [
            build_scope_actor_record(
                result,
                fallback_model_id=getattr(ctx, "_last_scope_model", ""),
                slot_id="scope_slot_error",
            )
        ]
        return result


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


def run_parallel_review(
    ctx, commit_message, *, goal="", scope="", review_rebuttal="",
    review_binding_fingerprint="",
):
    """Run the commit gate's triad and scope reviews against the staged diff.

    Q25-A ordering: BOTH gate packets (the triad api pack and every scope row's
    pack) are assembled and fit-checked BEFORE any reviewer is dispatched, so a
    deterministic assembly failure on either side spends $0 on the other. The
    paid dispatches still run concurrently, and every verdict is computed by
    the same code as before — only the ordering moved."""
    from ouroboros.tools.review import _dispatch_unified_review, _prepare_unified_review
    if bool(getattr(ctx, "_review_reconcile_only", False)):
        from ouroboros.review_custody import prepare_frozen_review_reconciliation

        prepare_frozen_review_reconciliation(
            ctx, getattr(ctx, "_pending_review_attempt", None),
        )

    # Reset forensic fields so prior attempts cannot bleed into early exits.
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_scope_raw_result = {}
    ctx._last_scope_raw_results = []
    # Managed subject↔binding assertion input: every gate subject built during
    # THIS attempt records its S tree here; the commit gate then asserts the
    # set equals the binding fingerprint's tree_sha (typed failure otherwise).
    ctx._last_review_subject_trees = set()
    # The per-attempt managed-subject memo resets with the same boundary (C5).
    ctx._managed_review_subject_memo = {}

    try:
        diff_bytes = run_cmd(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"], cwd=ctx.repo_dir,
        ).encode()
    except Exception:
        diff_bytes = b""
    snapshot_digest = hashlib.sha256(diff_bytes).hexdigest()
    snapshot_key = snapshot_digest[:16]
    retry_key = str(getattr(ctx, "_current_review_retry_key", "") or "") or (
        _commit_review_retry_key(
            ctx, commit_message, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal,
            binding_fingerprint=review_binding_fingerprint,
        )
    )
    _stored = getattr(ctx, '_scope_review_history', None) or {}
    _scope_history = _stored.get(snapshot_key, []) if isinstance(_stored, dict) else []
    _history_snapshot = list(getattr(ctx, '_review_history', []))

    # Snapshot advisory state before assembly and dispatch mutate it.
    _advisory_snapshot_before = list(getattr(ctx, '_review_advisory', []))

    # ---- Phase 1 (Q25=A): assemble every packet; dispatch NOTHING yet. ----
    triad_prepared, triad_early, triad_exited = None, None, True
    try:
        triad_prepared, triad_early, triad_exited = _prepare_unified_review(
            ctx, commit_message, review_rebuttal=review_rebuttal, goal=goal, scope=scope)
        if triad_prepared is not None:
            triad_prepared["retry_key"] = retry_key
    except Exception as e:
        log.warning("Triad review raised unexpected exception: %s", e)
        triad_early = (
            f"⚠️ REVIEW_BLOCKED: Triad review crashed — {e}\nFix the issue and retry."
        )
        ctx._last_review_block_reason = 'infra_failure'
        ctx._last_review_critical_findings = []
    scope_rows, scope_result = [], None
    try:
        scope_rows = _prepare_scope_rows(
            ctx, commit_message, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal,
            history_snapshot=_history_snapshot, scope_history=_scope_history)
    except Exception as e:
        log.warning("Scope review raised unexpected exception: %s", e)
        scope_result = ScopeReviewResult(
            blocked=True,
            block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: Scope review failed — {e}\nFix the issue and retry.",
            model_id=getattr(ctx, "_last_scope_model", "") or _get_scope_model(),
            status="error",
        )
        ctx._last_scope_raw_results = [
            build_scope_actor_record(
                scope_result,
                fallback_model_id=getattr(ctx, "_last_scope_model", ""),
                slot_id="scope_slot_error",
            )
        ]

    # ---- Admission: a deterministic assembly block anywhere → ZERO dispatch. ----
    deterministic_block = (
        (triad_exited and bool(triad_early))
        or (scope_result is not None and scope_result.blocked)
        or any(row["final"] is not None and row["final"].blocked for row in scope_rows)
    )
    if deterministic_block:
        review_err = triad_early if triad_exited else None
        if not triad_exited:
            if not hasattr(ctx, "_review_degraded_reasons"):
                ctx._review_degraded_reasons = []
            ctx._review_degraded_reasons.append(
                "triad_not_dispatched_assembly_block: the gate was already "
                "deterministically blocked at packet assembly ($0 spent on the triad)"
            )
            # Seat identity survives the $0 path: every prepared-but-withheld
            # triad seat gets a typed not_dispatched actor record, mirroring
            # the scope rows' placeholders (durable review status shows WHICH
            # configured seats were withheld, not just that "the triad" was).
            from ouroboros.tools.review_admission import triad_not_dispatched_records
            ctx._last_triad_raw_results = triad_not_dispatched_records(
                (triad_prepared or {}).get("row_plan") or {},
                "assembly-before-dispatch admission (Q25=A): the commit gate "
                "was already deterministically blocked at packet assembly, so "
                "this seat was not dispatched ($0 spent).",
            )
        if scope_result is None and scope_rows:
            scope_result = _run_scope(
                ctx, commit_message, scope_rows, False, goal=goal, scope=scope,
                review_rebuttal=review_rebuttal, history_snapshot=_history_snapshot,
                scope_history=_scope_history, retry_key=retry_key)
    else:
        # ---- Money admission (owner decision 2026-09-05): the WHOLE wave, scope
        # seats first, must fit the root fence before ANY seat is dispatched;
        # otherwise every seat is a typed $0 not_dispatched record and the gate
        # blocks naming the shortfall, never a half-dispatched panel. ----
        from ouroboros.tools.review_admission import admit_commit_gate_wave, commit_gate_paid_seats

        seats = []
        wave_refusal = None
        if not bool(getattr(ctx, "_review_reconcile_only", False)):
            try:
                seats = commit_gate_paid_seats(triad_prepared, triad_exited, scope_rows)
                wave_refusal = admit_commit_gate_wave(ctx, seats)
            except Exception as e:
                # Fail-open is the enforcement choice (as review_wave_budget_gate's
                # own), but "admitted" and "admission crashed" are different
                # facts: the wave dispatches unadmitted and without the
                # scope-first hold, and says so once, typed.
                from ouroboros.tools.review_helpers import emit_review_event
                log.warning("commit-gate wave admission unavailable (%s: %s); the wave dispatches "
                            "unadmitted and without the scope-first hold", type(e).__name__, e)
                emit_review_event(ctx, {
                    "type": "review_wave_admission_unavailable", "surface": "commit_gate",
                    "task_id": str(getattr(ctx, "task_id", "") or ""),
                    "error": f"{type(e).__name__}: {e}",
                })
                seats, wave_refusal = [], None
        if wave_refusal is not None:
            from ouroboros.tools.review import _handle_review_block_or_warning
            from ouroboros.tools.review_admission import triad_not_dispatched_records

            from ouroboros.config import get_review_enforcement

            blocking_review = bool((triad_prepared or {}).get(
                "blocking_review", get_review_enforcement() == "blocking"))
            if not hasattr(ctx, "_review_degraded_reasons"):
                ctx._review_degraded_reasons = []
            ctx._review_degraded_reasons.append(
                "triad_not_dispatched_budget_admission: the commit-gate wave did not "
                "fit the root budget fence, so no triad seat was dispatched ($0 spent)"
            )
            ctx._last_review_critical_findings = []
            if not triad_exited:
                ctx._last_review_block_reason = "review_wave_budget_insufficient"
                ctx._last_triad_raw_results = list(
                    getattr(ctx, "_triad_withheld_seat_records", []) or []
                ) + triad_not_dispatched_records(
                    (triad_prepared or {}).get("row_plan") or {}, wave_refusal,
                )
                review_err = _handle_review_block_or_warning(
                    ctx, blocking_review, wave_refusal,
                    "Review enforcement=Advisory: the commit-gate review wave was declined "
                    "before dispatch (budget fence); commit proceeding without review. ",
                )
            else:
                review_err = triad_early
            if scope_rows:
                scope_result = _run_scope(
                    ctx, commit_message, scope_rows, False, goal=goal, scope=scope,
                    review_rebuttal=review_rebuttal, history_snapshot=_history_snapshot,
                    scope_history=_scope_history, retry_key=retry_key,
                    withheld_reason=wave_refusal)
                if blocking_review:
                    scope_result.blocked = True
                    scope_result.block_message = "⚠️ SCOPE_REVIEW_BLOCKED: " + wave_refusal
        else:
            # ---- Phase 2: submit the assembled packets to the executor pool. ----
            try:
                if not bool(getattr(ctx, "_review_reconcile_only", False)):
                    _reserve_parallel_review_roster(ctx, triad_prepared, scope_rows)
            except Exception as e:
                log.warning("Commit review custody reservation failed: %s", e)
                ctx._last_review_block_reason = "infra_failure"
                ctx._last_review_critical_findings = []
                review_err = (
                    "⚠️ REVIEW_BLOCKED: durable review custody could not be reserved "
                    f"before dispatch — {e}\nNo reviewer was started; fix the state write and retry."
                )
                scope_result = ScopeReviewResult(
                    blocked=True,
                    block_message=(
                        "⚠️ SCOPE_REVIEW_BLOCKED: durable review custody could not be "
                        "reserved before dispatch; no scope reviewer was started."
                    ),
                    model_id=getattr(ctx, "_last_scope_model", "") or _get_scope_model(),
                    status="not_dispatched",
                )
            else:
                with _cf.ThreadPoolExecutor(max_workers=2) as pool:
                    # Scope FIRST (owner decision 2026-09-05): the blocking seat
                    # is submitted before the triad and holds it until its own
                    # reservation is on the ledger, so a fitting wave can never
                    # leave scope unfunded while non-blocking seats hold the money.
                    wave_started = time.monotonic()
                    wave_known = _reservation_ids_now()   # identities on the root telemetry BEFORE any seat of this wave
                    # Both seats run under a COPY of the admitting context
                    # (contextvars.copy_context, the loop_tool_execution and
                    # plan_review precedent): the usage scope the wave was
                    # admitted with — its bound root fence included — is the
                    # one every seat's reserve_attempt binds, so admission and
                    # reservation share one fence even after a mid-turn
                    # settings reload changed the environment's number.
                    scope_fut = (
                        pool.submit(contextvars.copy_context().run, _run_scope, ctx, commit_message,
                                    scope_rows, True, goal=goal, scope=scope,
                                    review_rebuttal=review_rebuttal,
                                    history_snapshot=_history_snapshot,
                                    scope_history=_scope_history, retry_key=retry_key)
                        if scope_rows else None
                    )
                    if not triad_exited:
                        _await_scope_reservation(ctx, scope_fut, seats, wave_started, known_ids=wave_known)
                    triad_fut = (
                        None if triad_exited
                        else pool.submit(contextvars.copy_context().run, _dispatch_unified_review,
                                         ctx, commit_message, triad_prepared)
                    )
                    if triad_fut is None:
                        review_err = triad_early
                    else:
                        try:
                            review_err = triad_fut.result()
                        except Exception as e:
                            log.warning("Triad review raised unexpected exception: %s", e)
                            review_err = (
                                f"⚠️ REVIEW_BLOCKED: Triad review crashed — {e}\nFix the issue and retry."
                            )
                            ctx._last_review_block_reason = 'infra_failure'
                            ctx._last_review_critical_findings = []
                    if scope_fut is not None:
                        try:
                            scope_result = scope_fut.result()
                        except Exception as e:
                            log.warning("Scope future raised unexpected exception: %s", e)
                            scope_result = ScopeReviewResult(
                                blocked=True,
                                block_message=f"⚠️ SCOPE_REVIEW_BLOCKED: Scope review future crashed — {e}\nFix the issue and retry.",
                                model_id=getattr(ctx, "_last_scope_model", "") or _get_scope_model(),
                                status="error",
                            )
                            ctx._last_scope_raw_results = [
                                build_scope_actor_record(
                                    scope_result,
                                    fallback_model_id=getattr(ctx, "_last_scope_model", ""),
                                    slot_id="scope_slot_error",
                                )
                            ]
    triad_block_reason = getattr(ctx, '_last_review_block_reason', 'critical_findings')
    triad_advisory_post = list(getattr(ctx, '_review_advisory', []))
    triad_advisory = [a for a in triad_advisory_post if a not in _advisory_snapshot_before]

    if scope_result is not None:
        updated = _scope_history + [_scope_history_entry(scope_result)]
        existing = getattr(ctx, '_scope_review_history', None) or {}
        if not isinstance(existing, dict):
            existing = {}
        existing[snapshot_key] = updated
        ctx._scope_review_history = existing
        # Canonical scope actor record for durable CommitAttemptRecord persistence.
        raw_results = list(getattr(ctx, "_last_scope_raw_results", []) or [])
        if raw_results:
            ctx._last_scope_raw_result = {
                "status": getattr(scope_result, "status", ""),
                "model_id": getattr(scope_result, "model_id", "") or getattr(ctx, "_last_scope_model", ""),
                "context_manifest": getattr(scope_result, "context_manifest", {}) or {},
                "raw_results": raw_results,
                "raw_text": getattr(scope_result, "raw_text", ""),
                "critical_findings": getattr(scope_result, "critical_findings", []) or [],
                "advisory_findings": getattr(scope_result, "advisory_findings", []) or [],
            }
        else:
            ctx._last_scope_raw_result = build_scope_actor_record(
                scope_result,
                fallback_model_id=getattr(ctx, "_last_scope_model", ""),
            )
    else:
        ctx._last_scope_raw_result = {}

    return review_err, scope_result, triad_block_reason, triad_advisory


def aggregate_review_verdict(review_err, scope_result, triad_block_reason, triad_advisory,
                              ctx, commit_message, commit_start, repo_dir):
    """Aggregate triad/scope result and return block state plus advisory items."""
    _combined_blocked = False
    _combined_messages = []
    _combined_findings = []
    _scope_advisory_items = []

    if scope_result is not None:
        for f in (scope_result.critical_findings or []):
            item = {
                "severity": "critical",
                "tag": "scope",
                "item": str(f.get("item", "") or ""),
                "reason": str(f.get("reason", "") or ""),
                "verdict": "FAIL",
            }
            if f.get("obligation_id"):
                item["obligation_id"] = str(f.get("obligation_id"))
            _scope_advisory_items.append(item)
        for f in (scope_result.advisory_findings or []):
            item = {
                "severity": "advisory",
                "tag": "scope",
                "item": str(f.get("item", "") or ""),
                "reason": str(f.get("reason", "") or ""),
                "verdict": "FAIL",
            }
            if f.get("obligation_id"):
                item["obligation_id"] = str(f.get("obligation_id"))
            _scope_advisory_items.append(item)

    if review_err:
        _combined_blocked = True
        _combined_messages.append(review_err)
        _combined_findings.extend(getattr(ctx, '_last_review_critical_findings', []))
    if scope_result is not None:
        if scope_result.blocked:
            _combined_blocked = True
            _combined_messages.append(scope_result.block_message)
            _combined_findings.extend(scope_result.critical_findings or [])
        elif scope_result.advisory_findings or scope_result.critical_findings:
            _advisory_msg = _format_scope_advisory_msg(scope_result)
            if _advisory_msg and _combined_blocked:
                _combined_messages.append(_advisory_msg)

    if not _combined_blocked:
        return False, None, '', _combined_findings, _scope_advisory_items

    if review_err and (scope_result is None or not scope_result.blocked):
        block_reason = triad_block_reason
    elif scope_result is not None and scope_result.blocked and not review_err:
        block_reason = "scope_blocked"
    else:
        block_reason = triad_block_reason

    if len(_combined_messages) > 1:
        combined_msg = "\n\n".join(_combined_messages)
        if review_err and scope_result is not None and scope_result.blocked:
            combined_msg += "\n\n---\n⚠️ Note: Both triad review AND scope review found issues (shown above)."
    else:
        combined_msg = _combined_messages[0]

    if triad_advisory and not review_err:
        adv_text = "\n".join(
            f"  ⚠️ Advisory: {format_review_history_entry(a)}"
            for a in triad_advisory
        )
        combined_msg += f"\n\n---\nTriad advisory findings:\n{adv_text}"

    return True, combined_msg, block_reason, _combined_findings, _scope_advisory_items
