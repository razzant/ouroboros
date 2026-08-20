"""Shared multi-review substrate.

This module is the common cognitive primitive for migrated review surfaces and
the contract target for remaining legacy immune-system reviews. Slot identity is
separate from model identity, so duplicate model IDs are valid independent
reviewer slots.
"""

from __future__ import annotations

import contextlib
import hashlib  # noqa: F401
import json
import logging
import os
import pathlib
import queue
import threading
import time
from dataclasses import asdict, dataclass, field  # noqa: F401
from typing import Any, Dict, List

log = logging.getLogger("review_substrate")

from ouroboros.config import get_review_models, review_model_uses_local
from ouroboros.llm import LLMClient
from ouroboros.observability import new_call_id, persist_call, redact_projection  # noqa: F401
from ouroboros.provider_models import provider_for_model  # noqa: F401
# Everything below the seam. Re-exported here because the substrate is the
# historical import site for the api_chat prompt renderers; `review_execution`
# owns them now and must never import this module back.
from ouroboros.review_execution import (  # noqa: F401  (compat re-exports)
    AgentSessionReviewExecutor,
    ApiChatReviewExecutor,
    ReviewAssignment,
    ReviewAttemptResult,
    ReviewRouteKind,
    ReviewRouteUnavailable,
    ReviewSlotExecutor,
    _execute_slot_attempt,
    _messages_char_count,
    _render_prompt,
    _render_prompt_parts,
    _request_messages,
    _review_route_executor,
    assert_cache_breakpoint_cap,
    configured_review_routes,
)
# Reviewer-output JSON extraction lives in ONE place beside the array
# extractor it falls back to (the fenced-object and verdict parsers were
# split across two modules for no reason).
from ouroboros.triad_review import parse_review_findings
from ouroboros.usage_accounting import (
    UsageAccountingError,
    UsageScope,
    current_usage_scope,
    physical_attempt_limit,
    usage_scope,
)
from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact
# Tier vocabulary SSOT lives in outcomes.py; reuse it so a future tier rename
# cannot silently desync the capsule from the objective axis.
from ouroboros.outcomes import (  # noqa: F401  (compat re-exports)
    OUTCOME_TIER_BEST_EFFORT,
    OUTCOME_TIER_BLOCKED,
    OUTCOME_TIER_SOLVED,
)
# The typed panel records, the verdict reducers, and the panel projection live
# in their own owners below this module's seam; they are re-exported here
# because this module is their historical import site, and they must never
# import it back.
from ouroboros.review_records import (  # noqa: F401  (compat re-exports)
    HARDNESS_ADVISORY_VISIBLE,
    HARDNESS_HARD_GATE,
    HARDNESS_LABEL_ONLY,
    ReviewActorRecord,
    ReviewRequest,
    ReviewRunResult,
    ReviewSlot,
    TYPED_FAILURE_FACT_KEYS,
)
from ouroboros.review_verdict import (  # noqa: F401  (compat re-exports)
    DIALOGUE_CONTINUE,
    DIALOGUE_STABLE_DISAGREEMENT,
    DIALOGUE_STATUS_VALUES,
    DIALOGUE_UNREACHABLE,
    _CRITERION_STATUSES,
    _TIER_ORDER,
    _contract_valid_actors,
    _contributing_actors,
    _criteria_have_supported_evidence,
    _criteria_shape_valid,
    _unresolved_evidence_ref_labels,
    aggregate_dialogue_status,
    aggregate_outcome_tier,
    build_improvement_capsule,
    dissent_findings,
    panel_reason,
    task_acceptance_is_clean,
)
from ouroboros.review_projection import (  # noqa: F401  (compat re-exports)
    _public_review_reason,
    _response_ref_projection,
    _review_actor_projection,
    _review_enforcement_impact,
    _review_panel_id,
    _transport_error_status,
    build_review_binding,
    compact_review_projection,
)


def review_repo_dirs_for(ctx: Any) -> tuple[pathlib.Path, pathlib.Path]:
    """Return validated ``(governance, subject)`` roots for plan/scope review."""
    from ouroboros.tools.registry import active_repo_dir_for

    workspace_raw = getattr(ctx, "workspace_root", None)
    workspace = pathlib.Path(workspace_raw) if isinstance(workspace_raw, (str, pathlib.Path)) else None
    if workspace is not None and not str(getattr(ctx, "workspace_mode", "") or "").strip():
        raise ValueError("workspace_root is set without workspace_mode")
    system_raw = getattr(ctx, "system_repo_dir", None)
    system = pathlib.Path(system_raw) if isinstance(system_raw, (str, pathlib.Path)) else None
    governance = (system or pathlib.Path(getattr(ctx, "repo_dir"))).resolve(strict=False)
    subject = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    if not governance.is_dir() or not subject.is_dir():
        raise ValueError(f"unavailable governance/subject root: {governance} / {subject}")
    return governance, subject


# Identity prefixes for the configured reviewer surfaces. A surface that fans
# rows out registers its prefix here rather than spelling one inline, so
# ``slot_id_for_row`` stays the only place a row id is built.
SLOT_ID_PREFIX = "slot"
SCOPE_SLOT_ID_PREFIX = "scope_slot"
PLAN_SLOT_ID_PREFIX = "plan_slot"


def slot_id_for_row(index: int, *, prefix: str = SLOT_ID_PREFIX) -> str:
    """Identity of the ``index``-th (1-based) configured reviewer row.

    The single mint for reviewer-slot identity, and the reason this module's
    contract says slot identity is separate from model identity. Naming a row
    after its own model instead collides two rows that share a model (a supported
    configuration — ``get_scope_review_models`` preserves duplicates on purpose),
    collides two model spellings that sanitize alike (``openai::gpt-5`` and
    ``openai/gpt/5``), and moves a row's identity the moment the owner edits its
    model, so the row's receipts stop lining up with its own history. The model,
    the route and the effort are PROPERTIES of a row, never its name.
    """
    return f"{prefix}_{int(index)}"


def reviewer_slots(
    models: List[str] | None = None,
    *,
    effort: str = "medium",
    role_hint: str = "",
    id_prefix: str = SLOT_ID_PREFIX,
    route_env_key: str = "",
) -> List[ReviewSlot]:
    """The configured reviewer rows, each carrying its DELIVERY route.

    ``route_env_key`` names the surface's per-row route list (plan 5.1): the
    commit triad and scope pass theirs, so a row can be an api_chat call or a
    delegated agent session. Surfaces that stay on the API by owner decision
    (task acceptance, plan review — D15) pass NOTHING, which pins every row to
    ``api_chat`` explicitly rather than by accident.
    """
    raw_models = models if models is not None else get_review_models()
    named = [str(model) for model in (raw_models or []) if str(model or "").strip()]
    routes = configured_review_routes(route_env_key, len(named)) if route_env_key else [
        ReviewRouteKind.API_CHAT
    ] * len(named)
    return [
        ReviewSlot(slot_id=slot_id_for_row(idx + 1, prefix=id_prefix), model=model, effort=effort,
                   role_hint=role_hint, use_local=review_model_uses_local(model),
                   route=routes[idx])
        for idx, model in enumerate(named)
    ]


def scope_reviewer_slots(
    models: List[str] | None = None, *, effort: str | None = None,
) -> List[ReviewSlot]:
    """The configured scope-reviewer rows — the single owner of scope-slot identity.

    Both scope surfaces read their ids from here: the substrate call that produces
    the durable prompt/response refs, and the actor records the commit attempt
    persists. One row therefore carries exactly one identity instead of two that
    disagree. Each row also carries its configured delivery route (D14: every
    scope slot is independently harness-or-API).

    With no explicit ``models`` the rows come from the reviewer-slot SSOT
    (6.1): stable owner ids, per-row route/target/effort. An explicit list
    keeps the historical positional behavior for callers that rebuild one row.

    An omitted ``effort`` resolves to the configured scope-review effort: the
    legacy path used to take this parameter's old literal default instead,
    silently running the BLOCKING reviewer below configured strength (the
    downgrade class the owner forbade).
    """
    if effort is None:
        from ouroboros.config import resolve_effort

        effort = resolve_effort("scope_review")
    if models is None:
        from ouroboros.reviewer_slot_config import structured_scope_review_slots

        structured = structured_scope_review_slots()
        if structured is not None:
            return structured
        # Resolved at call time so the configured list stays the live authority.
        from ouroboros.config import get_scope_review_models

        models = get_scope_review_models()
    from ouroboros.review_execution import SCOPE_REVIEW_ROUTES_ENV

    return reviewer_slots(
        models, effort=effort, role_hint="scope reviewer", id_prefix=SCOPE_SLOT_ID_PREFIX,
        route_env_key=SCOPE_REVIEW_ROUTES_ENV,
    )


class ReviewCoordinator:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        drive_root: pathlib.Path | None = None,
        usage_ctx: Any = None,
    ):
        self.llm = llm or LLMClient()
        if drive_root is not None:
            self.drive_root = pathlib.Path(drive_root)
        else:
            # ISO-DRIP: the default is the ABSOLUTE config SSOT, never the old
            # cwd-relative "../data" — with any cwd under a repo/ that spelling
            # names the LIVE data root's sibling, so default-constructed
            # coordinators dripped synthetic review records into live
            # observability (and on trees with the absolute-root guard they
            # silently LOST the records instead: persist_call refuses relative
            # roots and the coordinator swallows it into empty refs). Same
            # resolution order the review surfaces already use
            # (review_drive_root: ctx → DATA_DIR); read late off the module so
            # test isolation and runtime rebinding are honored.
            from ouroboros import config

            self.drive_root = pathlib.Path(config.DATA_DIR)
        self.usage_ctx = usage_ctx

    def run(self, request: ReviewRequest, slots: List[ReviewSlot]) -> ReviewRunResult:
        if not slots:
            return ReviewRunResult(
                request=asdict(request),
                actors=[],
                parsed_findings=[],
                aggregate_signal="DEGRADED",
                degraded=True,
                degraded_reasons=["no_review_slots"],
                panel_id=_review_panel_id(request, []),
            )

        result_queue: "queue.Queue[ReviewActorRecord]" = queue.Queue()
        started_slots: List[ReviewSlot] = []
        base_scope = current_usage_scope() or UsageScope()
        usage_meta = (
            getattr(self.usage_ctx, "task_metadata", {})
            if self.usage_ctx is not None
            else {}
        )
        if not isinstance(usage_meta, dict):
            usage_meta = {}
        task_id = str(request.task_id or base_scope.task_id or "")
        root_task_id = str(
            usage_meta.get("root_task_id") or base_scope.root_task_id or task_id
        )
        budget_root = (
            usage_meta.get("budget_drive_root")
            or getattr(self.usage_ctx, "budget_drive_root", "")
            or base_scope.drive_root
            or self.drive_root
        )
        if base_scope.global_limit_usd is not None:
            global_limit = base_scope.global_limit_usd
        else:
            try:
                configured_global_limit = float(os.environ.get("TOTAL_BUDGET", "0") or 0)
                global_limit = configured_global_limit if configured_global_limit > 0 else None
            except (TypeError, ValueError):
                global_limit = None
        if base_scope.root_limit_usd is not None:
            root_limit = base_scope.root_limit_usd
        else:
            try:
                configured_root_limit = float(
                    os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0
                )
                root_limit = configured_root_limit if configured_root_limit > 0 else None
            except (TypeError, ValueError):
                root_limit = None
        review_usage_scope = UsageScope(
            drive_root=budget_root,
            task_id=task_id,
            root_task_id=root_task_id,
            parent_task_id=str(usage_meta.get("parent_task_id") or base_scope.parent_task_id or ""),
            category=f"{request.surface}_review",
            source="review_substrate",
            global_limit_usd=global_limit,
            root_limit_usd=root_limit,
        )

        def _start_slot(slot: ReviewSlot) -> None:
            started_slots.append(slot)

            def _worker() -> None:
                try:
                    with usage_scope(review_usage_scope):
                        result_queue.put(self._run_slot(request, slot))
                except Exception as exc:
                    result_queue.put(self._error_actor(request, slot, f"{type(exc).__name__}: {exc}"))

            thread = threading.Thread(
                target=_worker,
                name=f"ouroboros-review-{request.surface}-{slot.slot_id}",
                daemon=True,
            )
            thread.start()

        for slot in slots:
            _start_slot(slot)

        actors: List[ReviewActorRecord] = []
        slot_timeout = max(0.001, max(float(slot.timeout_sec or 1) for slot in slots))
        deadline = time.monotonic() + slot_timeout
        while len(actors) < len(slots):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                actors.append(result_queue.get(timeout=remaining))
            except queue.Empty:
                break

        seen = {actor.slot_id for actor in actors}
        started_ids = {slot.slot_id for slot in started_slots}
        for slot in slots:
            if slot.slot_id not in seen:
                if slot.slot_id in started_ids:
                    actors.append(self._error_actor(request, slot, f"Timeout after {slot.timeout_sec:g}s"))
                else:
                    actors.append(self._error_actor(request, slot, "Not started before reviewer timeout budget expired"))
        slot_order = {slot.slot_id: idx for idx, slot in enumerate(slots)}
        slots_by_id = {slot.slot_id: slot for slot in slots}
        actors.sort(key=lambda actor: slot_order.get(actor.slot_id, len(slot_order)))
        try:
            # «Выполняется как» (D22): beside each saved row the UI shows what
            # the row REALLY ran as last time. Disclosure only; best-effort.
            from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

            record_reviewer_slot_executions(request.surface, actors, slots_by_id)
        except Exception:
            log.debug("reviewer-slot last-execution write failed", exc_info=True)

        all_findings: List[Dict[str, Any]] = []
        # Split participation faults (a slot errored / timed out / returned empty)
        # from parse-degraded (a slot produced a DEGRADED verdict or unparseable
        # text). Only a participation fault fail-closes: a single Markdown/non-JSON
        # slot must NOT poison a clean quorum PASS (the old `degraded_reasons` gate
        # over-degraded honest 2-of-3 PASS reviews).
        actor_errors: List[str] = []
        parse_degraded: List[str] = []
        fail_count = 0
        pass_count = 0
        # When tier classification is required, the contract is ENFORCED before an
        # actor contributes to quorum. A tier-less PASS is non-responsive. A task-
        # acceptance FAIL contributes only when it carries a bounded correction rail;
        # a bare veto must not terminalize Required+Blocking with nothing to improve.
        classify_tier = bool(
            request.surface == "task_acceptance"
            and (request.policy or {}).get("classify_outcome_tier")
        )
        _valid_tiers = {"solved", "best_effort", "blocked_with_evidence"}
        # A SOLVED task-acceptance PASS need not carry a tier-up coach. Commit/scope
        # use distinct surfaces and retain their own hard-gate semantics.
        is_advisory = (
            request.surface == "task_acceptance"
            or str((request.policy or {}).get("hardness") or "") == HARDNESS_ADVISORY_VISIBLE
        )
        for actor in actors:
            if actor.status == "error":
                actor_errors.append(f"{actor.slot_id}:{actor.error}")
            elif actor.status != "ok":
                actor_errors.append(f"{actor.slot_id}:{actor.status}")
            parsed, findings, signal = parse_review_findings(actor.raw_text)
            actor.parsed = parsed
            actor.signal = signal
            slot = slots_by_id.get(actor.slot_id)
            actor.actor_role = (
                str(getattr(slot, "role_hint", "") or "").strip()
                or f"{request.surface} reviewer"
            )
            truth = _review_actor_projection(actor, request.surface)
            for key in (
                "model", "transport_status", "parse_status", "semantic_verdict", "provider",
                "coverage", "reason",
            ):
                setattr(actor, key, truth[key])
            all_findings.extend({**item, "slot_id": actor.slot_id, "model": actor.model} for item in findings)
            # The required-tier contract needs BOTH a valid outcome_tier AND a
            # non-empty completion_coach (both are required JSON keys); a PASS
            # missing either is non-responsive to the contract.
            _tier = str(parsed.get("outcome_tier") or "").strip().lower() if isinstance(parsed, dict) else ""
            _criteria = parsed.get("criteria_used") if isinstance(parsed, dict) else None
            _criteria_ok = _criteria_shape_valid(_criteria, _tier)
            contract_ok = (
                _tier in _valid_tiers
                and (
                    bool(str((parsed or {}).get("completion_coach") or "").strip())
                    # Advisory carve-out: a SOLVED deliverable has no tier-up step, so an
                    # empty coach must NOT demote it to DEGRADED.
                    or (is_advisory and _tier == "solved")
                )
                # Criteria shape rides the tier contract (its knob was constant-true, deleted).
                and _criteria_ok
            )
            if signal == "FAIL":
                # A task-acceptance FAIL is authoritative only when it obeys the
                # tier contract and carries a bounded correction rail.  A bare
                # veto cannot terminalize Required+Blocking with nothing the
                # agent can improve; keep the raw FAIL in parsed for forensics,
                # but make the actor abstain exactly like other contract failures.
                _has_concrete_finding = any(
                    isinstance(item, dict)
                    and bool(str(item.get("recommendation") or item.get("item") or "").strip())
                    for item in findings
                )
                _parsed_obj = parsed if isinstance(parsed, dict) else {}
                _has_correction_rail = (
                    bool(str(_parsed_obj.get("completion_coach") or "").strip())
                    or _has_concrete_finding
                    or _tier in {"best_effort", "blocked_with_evidence"}
                )
                if classify_tier and (
                    _tier not in _valid_tiers or not _has_correction_rail
                ):
                    parse_degraded.append(
                        f"{actor.slot_id}:fail_missing_tier_or_correction_rail"
                    )
                    actor.signal = "DEGRADED"
                    actor.parse_status = "malformed"
                    actor.semantic_verdict = ""
                    actor.reason = (
                        "Reviewer response violated the required outcome-tier or "
                        "correction-rail contract."
                    )
                else:
                    fail_count += 1
            elif signal == "PASS" and classify_tier and not contract_ok:
                parse_degraded.append(
                    f"{actor.slot_id}:missing_tier_coach_or_criterion_evidence"
                )
                # A contract-degraded PASS did NOT contribute to quorum, so its
                # recorded signal must be non-contributing too — else _contributing_
                # actors (and the objective-axis tier collector) would still let it
                # inject a tier/coach/finding (e.g. a PASS carrying a blocked tier +
                # empty coach) into the clean quorum capsule. Demote to DEGRADED;
                # the raw verdict stays in actor.parsed for forensics.
                actor.signal = "DEGRADED"
                actor.parse_status = "malformed"
                actor.semantic_verdict = ""
                actor.reason = (
                    "Reviewer response violated the required outcome-tier, coach, "
                    "or criterion-evidence contract."
                )
            elif signal == "PASS":
                pass_count += 1
            elif signal == "DEGRADED":
                parse_degraded.append(f"{actor.slot_id}:degraded")
        min_successful = max(1, int((request.policy or {}).get("min_successful_slots") or 1))
        fail_closed_on_errors = bool((request.policy or {}).get("fail_closed_on_errors"))
        degraded_reasons = actor_errors + parse_degraded
        # Task acceptance is conservative: any valid contributing FAIL vetoes.
        # DEGRADED/parse-failed actors abstain, while PASS still needs the adaptive
        # quorum supplied by the caller.  Commit/scope semantics remain unchanged.
        fail_threshold = 1
        if fail_count >= fail_threshold:
            aggregate = "FAIL"
        elif pass_count >= min_successful and not (
            fail_closed_on_errors and actor_errors and request.surface != "task_acceptance"
        ):
            aggregate = "PASS"
        else:
            aggregate = "DEGRADED"
            # Honest flag: DEGRADED must always carry a reason. Insufficient quorum
            # is itself the reason.
            if not degraded_reasons:
                degraded_reasons.append(
                    f"quorum_not_met: pass_count={pass_count} < min_successful={min_successful}"
                )
        # Bible P3 (centralized): a single configured slot is honored but the lost
        # cross-model diversity is recorded loudly + durably on EVERY surface that
        # runs through the coordinator, independent of the verdict (does NOT flip
        # the aggregate — block-vs-advisory still follows the caller's enforcement).
        # v6.74.0 (A6): the diversity note is an ORTHOGONAL LABEL — the typed
        # ``single_reviewer_no_diversity`` field below plus the projection label —
        # not a degraded_reason, so the panel ``reason`` names the real blocker
        # instead of leading every one-slot verdict with a diversity footnote.
        single_reviewer = len(slots) == 1
        participating_ids = {
            actor.slot_id
            for actor in actors
            if str(actor.signal or "").upper() in {"PASS", "FAIL"}
        }
        for actor in actors:
            actor.quorum_contribution = actor.slot_id in participating_ids
            if not actor.quorum_contribution:
                actor.enforcement_impact = "abstains"
            elif str(actor.signal or "").upper() == "FAIL":
                actor.enforcement_impact = "veto"
            else:
                actor.enforcement_impact = "supports_pass"
        return ReviewRunResult(
            request=asdict(request),
            actors=[asdict(actor) for actor in actors],
            parsed_findings=all_findings,
            # `degraded` tracks the aggregate so the review axis (which also reads
            # this flag) does not mark a quorum PASS as degraded over a single
            # parse-degraded slot.
            aggregate_signal=aggregate,
            degraded=(aggregate == "DEGRADED"),
            degraded_reasons=degraded_reasons,
            single_reviewer_no_diversity=single_reviewer,
            panel_id=_review_panel_id(request, actors),
        )

    def _custody_drive_root(self) -> pathlib.Path:
        """Where a DELEGATED slot's custody rows live: the canonical (budget)
        drive when the usage context names one, else this coordinator's drive.
        Data handed to the seam once, so the api_chat route never pays for it."""
        ctx = self.usage_ctx
        if ctx is not None and getattr(ctx, "drive_root", None):
            try:
                from ouroboros.delegate_custody import custody_root

                return custody_root(ctx)
            except Exception:
                log.debug("custody root resolution failed; using coordinator drive", exc_info=True)
        return self.drive_root

    def _error_actor(self, request: ReviewRequest, slot: ReviewSlot, error: str) -> ReviewActorRecord:
        call_id = new_call_id(f"review_{request.surface}_{slot.slot_id}_error")
        base_call_type = request.call_type or f"{request.surface}_review"
        assignment = ReviewAssignment(
            request=request, slot=slot, call_id=call_id, call_type=base_call_type,
            custody_root=self._custody_drive_root(),
        )
        # The synthetic prompt record is the route's own projection too: a slot
        # that never started must not build a pack its route would never send.
        # Best-effort by construction — this is the last-resort record for a slot
        # that already failed, so a route refusal (or an unrenderable prompt)
        # must degrade the record, never re-raise inside the failure path.
        try:
            prompt_projection = _review_route_executor(assignment, llm=self.llm).prompt_payload()
        except Exception:
            prompt_projection = {}
        prompt_ref: Dict[str, Any] = {}
        response_ref: Dict[str, Any] = {}
        try:
            prompt_ref = persist_call(
                self.drive_root,
                task_id=request.task_id or "review",
                call_id=f"{call_id}_prompt",
                call_type=f"{base_call_type}_prompt",
                payload={"request": asdict(request), "slot": asdict(slot), **prompt_projection},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "synthetic": True},
            )
        except Exception:
            prompt_ref = {}
        try:
            response_ref = persist_call(
                self.drive_root,
                task_id=request.task_id or "review",
                call_id=f"{call_id}_error",
                call_type=f"{base_call_type}_error",
                payload={"error": sanitize_tool_result_for_log(error)},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "status": "error", "synthetic": True},
            )
        except Exception:
            response_ref = {}
        return ReviewActorRecord(
            slot_id=slot.slot_id,
            model=slot.model,
            status="error",
            error=sanitize_tool_result_for_log(error),
            prompt_ref=prompt_ref,
            response_ref=response_ref,
        )

    def _run_slot(self, request: ReviewRequest, slot: ReviewSlot) -> ReviewActorRecord:
        call_id = new_call_id(f"review_{request.surface}_{slot.slot_id}")
        base_call_type = request.call_type or f"{request.surface}_review"
        assignment = ReviewAssignment(
            request=request, slot=slot, call_id=call_id, call_type=base_call_type,
            custody_root=self._custody_drive_root(),
        )
        # Transport is chosen once, here, through the seam; the prompt itself is
        # rendered by the route (lazily) rather than by this method, so a route
        # that does not send an API pack never assembles one.
        executor = _review_route_executor(assignment, llm=self.llm)
        prompt_projection = executor.prompt_payload()
        prompt_ref: Dict[str, Any] = {}
        response_ref: Dict[str, Any] = {}
        start = time.time()
        try:
            prompt_ref = persist_call(
                self.drive_root,
                task_id=request.task_id or "review",
                call_id=f"{call_id}_prompt",
                call_type=f"{base_call_type}_prompt",
                payload={"request": asdict(request), "slot": asdict(slot), **prompt_projection},
                manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model},
            )
        except Exception:
            prompt_ref = {}
        if request.surface == "task_acceptance" and request.evidence.get("__immutable_core_overflow__"):
            raw_text = json.dumps({
                "verdict": "DEGRADED",
                "findings": [],
                "summary": (
                    "Immutable owner requirements do not fit the acceptance evidence "
                    "budget; no requirement was silently truncated."
                ),
            })
            try:
                response_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_response",
                    call_type=f"{base_call_type}_response",
                    payload={"message": {"content": raw_text}, "usage": {}},
                    manifest={
                        "surface": request.surface, "slot_id": slot.slot_id,
                        "model": slot.model, "status": "degraded_core_overflow",
                        "physical_attempts": 0,
                    },
                )
            except Exception:
                response_ref = {}
            return ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="ok",
                raw_text=raw_text,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
            )
        try:
            p3_actor = request.surface in {"multi_model_review", "scope_review"}
            acceptance_actor = request.surface == "task_acceptance"
            actor_attempts = 2 if (p3_actor or acceptance_actor) else 1
            # Acceptance and P3 share the same two-physical-send rail. The
            # documented contract ("one substantive call and at most two
            # physical attempts total — same-route transport retry or
            # extraction/format repair") historically retried only empty/errored
            # responses; a MALFORMED non-empty acceptance response burned the
            # actor as DEGRADED without using its second permitted send. The
            # prompt, slot, and model never change on the repair resend.
            attempt_rail = (
                physical_attempt_limit(2)
                if acceptance_actor or p3_actor
                else contextlib.nullcontext()
            )
            with attempt_rail:
                _prior_msg, _prior_usage, _prior_text = None, None, ""
                for actor_attempt in range(actor_attempts):
                    try:
                        # The one seam. A null/non-object provider message comes
                        # back as empty raw_text: retry once on P3, then preserve
                        # the fail-closed empty actor.
                        attempt = _execute_slot_attempt(
                            assignment, llm=self.llm, executor=executor,
                        )
                        msg, usage, raw_text = attempt.message, attempt.usage, attempt.raw_text
                    except UsageAccountingError:
                        # Budget/ledger/physical-rail failures are not transport
                        # transients and must remain fail-closed without another
                        # send — but when the RAIL blocks the format-repair resend
                        # (the first send burned both physical attempts on an
                        # internal transport retry), keep the malformed first
                        # answer as forensics instead of degrading to a bare error.
                        if _prior_text:
                            msg, usage, raw_text = _prior_msg, _prior_usage, _prior_text
                            break
                        raise
                    except Exception:
                        if actor_attempt + 1 < actor_attempts:
                            continue
                        if _prior_text:
                            # The repair RESEND failed (transport, timeout): keep the
                            # malformed-but-substantive first answer as forensics.
                            msg, usage, raw_text = _prior_msg, _prior_usage, _prior_text
                            break
                        raise
                    if raw_text.strip():
                        if (
                            acceptance_actor
                            and actor_attempt + 1 < actor_attempts
                            and parse_review_findings(raw_text)[0] is None
                        ):
                            _prior_msg, _prior_usage, _prior_text = msg, usage, raw_text
                            try:
                                # P1 forensics: a successful repair must not make the
                                # malformed first answer unreconstructible.
                                persist_call(
                                    self.drive_root,
                                    task_id=request.task_id or "review",
                                    call_id=f"{call_id}_attempt1_response",
                                    call_type=f"{base_call_type}_attempt1_response",
                                    payload={"message": msg, "usage": usage},
                                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "repair_attempt": 1},
                                )
                            except Exception:
                                log.warning(
                                    "Failed to persist malformed first acceptance attempt for %s/%s — the repair resend will overwrite it",
                                    request.surface, slot.slot_id, exc_info=True,
                                )
                            continue  # extraction/format repair: one same-route resend
                        break
                    if _prior_text:
                        # Empty repair resend: keep the substantive malformed first
                        # answer instead of degrading to an empty actor.
                        msg, usage, raw_text = _prior_msg, _prior_usage, _prior_text
                        break
                    if actor_attempt + 1 >= actor_attempts:
                        break
            self._emit_usage(request, slot, usage, prompt_chars=executor.prompt_chars())
            try:
                response_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_response",
                    call_type=f"{base_call_type}_response",
                    payload={"message": msg, "usage": usage},
                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model},
                )
            except Exception:
                response_ref = {}
            return ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="ok" if raw_text.strip() else "empty",
                raw_text=raw_text,
                usage=usage,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
            )
        except Exception as exc:
            error_msg = truncate_review_artifact(str(exc), limit=4000)
            try:
                response_ref = persist_call(
                    self.drive_root,
                    task_id=request.task_id or "review",
                    call_id=f"{call_id}_error",
                    call_type=f"{base_call_type}_error",
                    payload={
                        "error_type": type(exc).__name__,
                        "error": sanitize_tool_result_for_log(error_msg),
                    },
                    manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model, "status": "error"},
                )
            except Exception:
                response_ref = {}
            http_status = getattr(exc, "status_code", None)
            return ReviewActorRecord(
                slot_id=slot.slot_id,
                model=slot.model,
                status="error",
                error=sanitize_tool_result_for_log(error_msg),
                transport_status=_transport_error_status(exc),
                failure_code=str(getattr(exc, "code", "") or ""),
                reset_at=str(getattr(exc, "reset_at", "") or ""),
                http_status=http_status if isinstance(http_status, int) and http_status else None,
                prompt_ref=prompt_ref,
                response_ref=response_ref,
                duration_sec=round(time.time() - start, 3),
            )

    def _emit_usage(
        self,
        request: ReviewRequest,
        slot: ReviewSlot,
        usage: Dict[str, Any],
        *,
        prompt_chars: int = 0,
    ) -> None:
        if self.usage_ctx is None:
            return
        try:
            from ouroboros.tools.review_helpers import emit_review_usage

            emit_review_usage(
                self.usage_ctx,
                model=slot.model,
                usage=usage,
                source=f"review_substrate:{request.surface}",
                prompt_chars=prompt_chars,
                extra={"surface": request.surface, "slot_id": slot.slot_id},
            )
        except Exception:
            pass


def run_review_request(
    request: ReviewRequest,
    *,
    slots: List[ReviewSlot] | None = None,
    drive_root: pathlib.Path | None = None,
    llm: LLMClient | None = None,
    usage_ctx: Any = None,
) -> ReviewRunResult:
    coordinator = ReviewCoordinator(llm=llm, drive_root=drive_root, usage_ctx=usage_ctx)
    result = coordinator.run(request, reviewer_slots(role_hint=request.surface) if slots is None else slots)
    if request.surface == "task_acceptance":
        # D-Q5 annotation-only pass: feeds the clean bit + disclosure, never parse
        # validity/quorum/verdicts. Called UNGUARDED on purpose — the annotator is
        # total and fail-CLOSED (a resolver failure stamps the non-clean row), and
        # `review_evidence` already built this packet, so swallowing an error here
        # could only turn "the host never checked the refs" into a clean PASS.
        from ouroboros.review_evidence import annotate_criteria_evidence_resolution

        annotate_criteria_evidence_resolution(result.actors, request.evidence)
    return result
