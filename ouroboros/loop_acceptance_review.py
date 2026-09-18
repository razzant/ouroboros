"""Host acceptance: admission, dialogue and author outcomes; loop facade bindings stay stable."""

from __future__ import annotations

import json
import logging
import pathlib
import time

import dataclasses

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.review_projection import publish_acceptance_checkpoint
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.review_helpers import review_enforcement_blocks
from ouroboros.utils import truncate_review_artifact


log = logging.getLogger("ouroboros.loop")


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


_ACCEPTANCE_REVIEW_CHECKLIST = (
    "Check whether the claimed result follows from the tool trace, "
    "whether errors/timeouts/artifacts were handled honestly, and "
    "whether each explicit original requirement was verified through "
    "the interface/surface the task itself names (not a weaker "
    "surrogate self-test), and "
    "whether the final response should be changed before release. "
    "SCOPE CUTS (v6.60.0): did the agent knowingly narrow the task's scope "
    "(dropped/limited requirements, simplified formats, skipped inputs)? "
    "A DISCLOSED, task-justified cut is honest best_effort; an unjustified "
    "or silent cut is a finding — name it with severity high and a concrete "
    "recommendation (under blocking enforcement it becomes an obligation). "
    "Classify the deliverable tier (solved / best_effort / "
    "blocked_with_evidence) and name the single highest-value change "
    "that would move it one tier up. If the task asks for a specific "
    "value or short answer, check the FINAL ANSWER line matches the "
    "requested format exactly."
)


@dataclass
class _TaskAcceptanceContext:
    tools: ToolRegistry
    content: str
    task_id: str
    task_type: str
    llm_trace: Dict[str, Any]
    drive_root: Optional[pathlib.Path]
    messages: List[Dict[str, Any]]
    emit_progress: Callable[[str], None]
    mode: str
    subtree_statuses: List[Dict[str, Any]]
    budget_profile: Any
    passes_done: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    review_binding: Dict[str, Any] = field(default_factory=dict)
    # One pre-rendered rails line (money/time/rounds/passes headroom) built
    # in loop.py from each real source, fed into the improvement capsule
    # (v6.74.0 A1, owner Q6); the capsule builder never gains ctx.
    rails_line: str = ""
    # Int-like ceiling plus per-slot caps, resolved once so rebuilds cannot drift.
    packet_budget_chars: int = 0


def acceptance_run_pending(run: Any) -> bool:
    """A recorded live producer is distinct from a verdict or lost custody."""
    actors = run.get("actors", []) if isinstance(run, dict) else getattr(run, "actors", [])
    return any(isinstance(actor, dict) and actor.get("operation_state")
               in {"pending_dispatch", "in_flight"} for actor in actors or [])


def _resolve_ctx_lineage(ctx: Any, task_id: str = "") -> Dict[str, Any]:
    """One reader of a live tool context's lineage facts, shared by both
    eligibility sites so the observation seam and the entrypoint agree."""
    from ouroboros.task_results import resolve_task_lineage

    meta = getattr(ctx, "task_metadata", {})
    return resolve_task_lineage(
        task_id or getattr(ctx, "task_id", ""),
        metadata=meta if isinstance(meta, dict) else {},
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )


def prepare_acceptance_observation(ctx: Any, trace: dict, incoming: Any, messages: list, tool_schemas: list) -> None:
    """Present the current owner-source selector as an append-only transcript row.

    A new row is appended only when the rendered facts changed; earlier rows stay,
    because rewriting or removing an already-sent message breaks provider prompt
    caches that only reuse a previous request when it is a byte-prefix of the next
    (issue #906).
    """
    from ouroboros.loop_acceptance import capture_acceptance_observation, acceptance_observation_prompt

    observed = capture_acceptance_observation(ctx, trace, incoming)
    if (_loop().get_task_review_mode() not in {"auto", "required"}
            or not any(row.get("function", {}).get("name") == "task_acceptance_review" for row in tool_schemas)):
        return
    # Cognitive-only direct turns (for example ``update_identity``) are ordinary
    # conversation and are explicitly ineligible for task acceptance. Do not
    # expose the internal source-selector protocol to Main in that case: after a
    # successful cognitive tool call it can otherwise answer the selector itself,
    # replacing the natural conversational response with acceptance bookkeeping.
    lineage = _resolve_ctx_lineage(ctx)
    eligible, _reason = _loop()._task_acceptance_eligible(
        _loop().get_task_review_mode(),
        trace,
        bool(getattr(ctx, "is_direct_chat", False)),
        is_root_task=bool(lineage["is_root_task"]),
        task_contract=getattr(ctx, "task_contract", None),
    )
    has_acceptance_state = bool(
        getattr(ctx, "_task_acceptance_pending", "")
        or getattr(ctx, "_task_acceptance_reviewed", False)
        or getattr(ctx, "_acceptance_request_pending", None)
        or getattr(ctx, "_delivery_candidate", None)
    )
    if not eligible and not has_acceptance_state:
        return
    note = acceptance_observation_prompt(ctx, observed)
    if not note:
        return
    latest = next((row for row in reversed(messages) if row.get("acceptance_observation")), None)
    if isinstance(latest, dict) and latest.get("content") == note:
        return  # the transcript already ends its observation history with these exact facts
    messages.append({"role": "user", "content": note, "acceptance_observation": True})


def wait_for_acceptance_feedback(tools: Any, limit_ctx: Any, trace: dict,
                                 tool_schemas: list, seen: set) -> None:
    """Park a pending answer with optional controls and a complete prose continuation."""
    ctx = tools._ctx
    binding = getattr(ctx, "_task_acceptance_pending", "")
    if not binding:
        return
    # Ready feedback skips parking, not the answer protocol: it may arrive
    # before the first wait, when the retained candidate has never been armed.
    _loop()._arm_delivery_control(tools, limit_ctx, trace,
                                  control="acceptance_feedback", skip_if_unchanged=True)
    from ouroboros.acceptance_settlement import awaited_panel_has_settled
    if awaited_panel_has_settled(ctx, trace):
        return  # its verdicts already woke this turn; the next round runs, nothing settles again
    from ouroboros.loop_transport import _owner_signal_pending

    # A wake arriving during Main's request must reach the normal ingress drain.
    if _owner_signal_pending(limit_ctx.incoming_messages, ctx.drive_root, ctx.task_id,
                             seen, getattr(ctx, "task_attempt", None) or 1):
        return
    from ouroboros.owner_wait import wait_after_tools

    wait_after_tools(ctx, limit_ctx.messages, trace, limit_ctx.accumulated_usage,
                     limit_ctx.round_idx, tool_schemas, seen, review_binding=binding)


def advance_explicit_acceptance(tools: Any, limit_ctx: Any, trace: dict,
                                incoming: Any, seen: set, emit: Any) -> None:
    """Nominate a complete result only after the round's entire tool block exists."""
    request = getattr(tools._ctx, "_acceptance_request_pending", None)
    if not isinstance(request, dict):
        return
    tools._ctx._acceptance_request_pending = None
    tools._ctx._acceptance_pending_review_choice = ""  # a new nomination starts a fresh wait/finish choice
    from ouroboros.loop_delivery import apply_delivery_subject_decision

    subject = request.get("acceptance_subject")
    if subject is not None:
        ok, reason = apply_delivery_subject_decision(tools, limit_ctx, trace, subject)
        if not ok:
            _loop()._append_or_merge_user_message(limit_ctx.messages, reason)
            return
    tools._ctx._acceptance_review_only = True
    try:
        # The tool's claim is a new complete nomination, not prose responding
        # to an earlier keep/replace prompt. Readiness and review stay shared.
        _loop()._replace_delivery_candidate(tools, limit_ctx, trace, request.get("subject") or "", control="candidate")
        _loop()._no_tool_final_answer(request.get("subject") or "", limit_ctx, trace,
                                      tools, incoming, seen, emit, review_only=True)
    finally:
        tools._ctx._acceptance_review_only = False
    if (getattr(tools._ctx, "_task_acceptance_pending", "")
            or getattr(tools._ctx, "_task_acceptance_reviewed", False)
            or not review_enforcement_blocks("blocking")):
        # Explicit submission retained a complete answer without delivering it.
        # Teach the existing keep/replace reader that this is a control episode;
        # otherwise the subject-observation's requested keep JSON becomes prose.
        _loop()._arm_delivery_control(tools, limit_ctx, trace, control="acceptance_feedback")


def _acceptance_dialogue_quorum(result: Any) -> int:
    """The quorum the panel itself used (policy min_successful_slots), with the
    adaptive_quorum fallback for records that lost the policy dict."""
    request = getattr(result, "request", None)
    policy = request.get("policy") if isinstance(request, dict) else {}
    try:
        quorum = int((policy or {}).get("min_successful_slots") or 0)
    except (TypeError, ValueError):
        quorum = 0
    if quorum <= 0:
        quorum = adaptive_quorum(len(getattr(result, "actors", None) or []) or 1)
    return max(1, quorum)


def _attach_dialogue_to_host_run(llm_trace: Dict[str, Any], dialogue: Dict[str, Any]) -> None:
    """Persist the dialogue-status vote distribution on the authoritative host
    run record so the review projection carries it for audit (A5)."""
    for run in reversed(llm_trace.get("review_runs") or []):
        if (
            isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
        ):
            run["dialogue"] = dict(dialogue)
            return


def _mark_agent_acceptance_runs_advisory(llm_trace: Dict[str, Any]) -> None:
    """Keep agent-invoked reviews as evidence without granting root authority."""
    for run in llm_trace.get("review_runs") or []:
        if not isinstance(run, dict) or run.get("authority") == "host_root":
            continue
        request = run.get("request") if isinstance(run.get("request"), dict) else {}
        if str(request.get("surface") or "") != "task_acceptance":
            continue
        run["authority"] = "agent_advisory"
        # Compatibility with the objective reducer: non-authoritative historical
        # runs stay fully auditable but cannot worst-case the host/root verdict.
        run["superseded_by_revision"] = True
        run["superseded_reason"] = "non_authoritative_agent_acceptance_review"


def _latest_agent_acceptance_evidence(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return the latest validated root self-call packet for host review.

    ``process_tool_results`` records only typed, non-authoritative root
    deferrals here.  The payload is already bounded and redacted by the shared
    evidence builder; the host builder will redact it again while assigning the
    explicit ``agent_supplied`` provenance.
    """
    for call in reversed(llm_trace.get("acceptance_evidence_calls") or []):
        if not isinstance(call, dict):
            continue
        if (
            str(call.get("status") or "") != "deferred_to_host_acceptance"
            or call.get("authoritative") is not False
        ):
            continue
        evidence = call.get("agent_supplied")
        if isinstance(evidence, dict):
            return dict(evidence)
    return {}


def _build_host_acceptance_evidence(ctx: _TaskAcceptanceContext) -> Dict[str, Any]:
    """Build the one bounded host packet shared by binding and reviewer input."""
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.loop_delivery import delivery_subject_projection

    committed_this_turn = any(
        isinstance(call, dict)
        and str(call.get("tool") or "") in ("commit_reviewed", "vcs_commit_reviewed")
        and str(call.get("status") or "") == "ok"
        for call in (ctx.llm_trace.get("tool_calls") or [])
    )
    supplied = _latest_agent_acceptance_evidence(ctx.llm_trace)
    supplied["acceptance_subject"] = delivery_subject_projection(ctx.tools._ctx, ctx.llm_trace, ctx.content)
    selected = getattr(ctx.tools._ctx, "_delivery_material_tool_indices", ())
    if selected:
        supplied["tool_trajectory_indices"] = sorted(set(supplied.get("tool_trajectory_indices") or []) | set(selected))
    evidence = build_task_acceptance_evidence(
        ctx.tools._ctx,
        llm_trace=ctx.llm_trace,
        drive_root=ctx.drive_root,
        task_id=ctx.task_id,
        task_type=ctx.task_type,
        agent_evidence=supplied,
        include_recent_commit=committed_this_turn,
        canonical_subject=str(ctx.content or ""),
        subtree_statuses=ctx.subtree_statuses,
        undispositioned_children=getattr(
            ctx.tools._ctx, "_forced_undispositioned_children", None),
        acceptance_dialogue_history=acceptance_dialogue_history(ctx.llm_trace),
        budget_chars=ctx.packet_budget_chars,
    )
    return evidence


def _total_paid_acceptance_cycles(ctx: _TaskAcceptanceContext) -> Any:
    """Paid acceptance panels this task TREE has already bought, read from the
    SAME ledger the wallet claim counts (``claimed_cycles``); ``None`` when the
    projection is unavailable (a descendant that may observe but not initialize)."""
    from ouroboros.task_results import project_task_acceptance_review_capacity

    return project_task_acceptance_review_capacity(
        ctx.tools._ctx, task_id=str(ctx.task_id or ""),
    ).get("claimed_cycles")


_RETRIEVING_ACCESS_DISCLOSURE = (
    "Access outside the task workspace is not guaranteed on this delivery: a refused or "
    "failed read is absence of evidence, not absence of the artifact — report it as a gap "
    "you could not verify instead of inferring the artifact does not exist."
)



def _retrieving_packet_projection(evidence: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.review_dispatch import retrieving_acceptance_packet

    return retrieving_acceptance_packet(evidence)


def acceptance_retrieving_work_order(
    request: Any, slots: List[Any], *, session_root: str, data_root: pathlib.Path,
) -> None:
    """Attach the route-owned work order of ONE acceptance panel's retrieving
    rows (owner R1/R4/R5/R15, 2026-09-01) to ``request`` in place.

    Every retrieving row receives the same task, criteria and output contract
    as the packet rows — rendered by the same `_render_prompt_parts` — plus
    absolute retrieval pointers. A SESSION row gets the FULL packet (its run is
    unobserved by the host, so the packet is its only attested view) and the
    access disclosure; a NATIVE row gets the packet without its freely
    degradable tail and the real data root (R5), because its episode reads
    task results and artifacts itself. The FULL packet stays on
    ``request.evidence``: evidence_refs resolve against it, never against a
    rendered projection."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.outcome_receipt_store import verification_receipts_path
    from ouroboros.review_execution import ReviewRouteKind, _render_prompt_parts, review_output_contract

    request.session_root = session_root
    request.policy["output_contract"] = review_output_contract(request)
    request.policy["native_data_root"] = str(data_root)
    task_id = str(request.task_id or "")
    root = pathlib.Path(data_root)
    try:
        artifacts_dir, receipts = task_artifact_dir_path(root, task_id), verification_receipts_path(root, task_id)
    except Exception:  # an unusual task id: name the canonical layout instead of refusing the work order
        artifacts_dir = root / "task_results" / "artifacts" / task_id
        receipts = artifacts_dir / "verification_receipts.jsonl"
    pointers = "\n".join((
        "RETRIEVAL POINTERS (absolute paths; the packet below is the host's attested projection of these sources):",
        f"- task workspace — the active tree the task worked in (your root): {session_root}",
        f"- task result record (contract, status, children): {root / 'task_results' / (task_id + '.json')}",
        f"- task artifacts named by the packet's `artifacts` manifest: {artifacts_dir}/",
        f"- host-attested verification receipts: {receipts}",
        f"- tool trajectory log (rows with task_id={task_id}): {root / 'logs' / 'tools.jsonl'}",
    ))
    native_packet: Optional[Dict[str, Any]] = None
    for slot in slots:
        if getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION:
            preamble = (
                "You review as a read-only agent session in the task workspace. The host's FULL evidence "
                "packet follows; verify its claims against the sources at the pointers with your own tools. "
                + _RETRIEVING_ACCESS_DISCLOSURE
            )
            packet = request.evidence
        else:
            preamble = (
                "You review as a bounded read-only native inspection episode; the host data root at the "
                "pointers is readable. The evidence packet follows WITHOUT its tool-trajectory rows and "
                "artifact previews — read those sources yourself at the pointers."
            )
            if native_packet is None:
                native_packet = _retrieving_packet_projection(request.evidence)
            packet = native_packet
        _stable, task_stable, dynamic = _render_prompt_parts(dataclasses.replace(request, evidence=packet), slot)
        slot_line = f"Slot: {slot.slot_id}"
        dynamic = dynamic.rstrip()  # the renderer's tail may grow a newline; the executor labels the slot itself
        if dynamic.endswith(slot_line):
            dynamic = dynamic[: -len(slot_line)].rstrip()
        request.slot_session_tasks[slot.slot_id] = "\n\n".join((
            preamble,
            pointers,
            "Every evidence_ref must be an EXACT member of the packet's host-attested exhibit vocabulary; "
            "the FULL packet is the host's resolution authority whatever you read at the pointers.",
            task_stable.rstrip() + "\n\n" + dynamic,
        ))



def _execute_task_acceptance_panel(ctx: _TaskAcceptanceContext) -> Any:
    """Perform the one substantive host panel over the pre-bound evidence."""
    from ouroboros.review_evidence import task_acceptance_evidence_revision
    from ouroboros.review_substrate import (
        HARDNESS_ADVISORY_VISIBLE,
        ReviewRequest,
        ReviewRunResult,
        review_repo_dirs_for,
        run_review_request,
        triad_delivery_slots,
    )
    from ouroboros.tools.review import _owner_deadline_at
    from ouroboros.review_dispatch import (
        TaskAcceptanceDispatchUnavailable,
        bind_task_acceptance_paid_dispatch,
        run_zero_physical_task_acceptance as _free_dispatch,
        task_acceptance_preclaim_refusal,
    )

    def _refused(reason: str) -> Any:
        return ReviewRunResult(
            request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
            actors=[], parsed_findings=[], aggregate_signal="DEGRADED", degraded=True,
            degraded_reasons=[reason],
        )

    evidence = ctx.evidence or _build_host_acceptance_evidence(ctx)
    try:
        # R2: the SAME triad rows every other triad surface reads — each with
        # its own delivery, effort, credential pin, actor binding and stable
        # id. R3: a malformed structured value refuses typed here exactly as
        # it does for plan and skill review; the silently projected default
        # panel is gone.
        slots = triad_delivery_slots(role_hint="task acceptance")
    except ValueError as exc:
        return _refused(f"reviewer_slot_config_invalid: {exc} (no reviewer was called)")
    request = ReviewRequest(
        surface="task_acceptance",
        goal=(
            _loop()._extract_plain_text_from_content(ctx.messages[1].get("content"))
            if len(ctx.messages) > 1 else ""
        ),
        subject=str(ctx.content or ""),
        evidence=evidence,
        checklist=_ACCEPTANCE_REVIEW_CHECKLIST,
        policy={
            "full_output_enters_context": False,
            "hardness": HARDNESS_ADVISORY_VISIBLE,
            "min_successful_slots": adaptive_quorum(len(slots)),
            "fail_closed_on_errors": True,
            "classify_outcome_tier": True,
            "max_physical_attempts_per_actor": 2,
            "slot_input_caps": getattr(ctx.packet_budget_chars, "slot_input_caps", {}),
        },
        task_id=ctx.task_id,
        retry_key=f"task_acceptance:{ctx.review_binding.get('paid_identity') or task_acceptance_evidence_revision(evidence)}",
        deadline_at=_owner_deadline_at(ctx.tools._ctx),  # R23: the owner window bounds every row
        # Managed Main actors have the existing mailbox continuation owner.
        # Standalone callers without it retain their bounded synchronous call.
        drain_deadline=(time.monotonic()
                        if callable(getattr(ctx.tools._ctx, "owner_wait_callback", None))
                        or not review_enforcement_blocks("blocking") else None),
    )
    if not slots:
        return _refused("no_review_slots")
    drive_root = pathlib.Path(ctx.drive_root if ctx.drive_root is not None else ctx.tools._ctx.drive_root)
    retrieving = [slot for slot in slots if getattr(slot, "retrieves", False)]
    if retrieving:
        # R1/R4/R15: a retrieving row (native episode, agent session) gets the
        # route-owned work order over the task's ACTIVE workspace — the tree it
        # worked in, never the governance repo — and the real data root (R5).
        # An unresolvable root leaves the row its own typed `session_root_missing`.
        try:
            session_root = str(review_repo_dirs_for(ctx.tools._ctx)[1])
        except Exception:
            session_root = ""
        acceptance_retrieving_work_order(request, retrieving, session_root=session_root, data_root=drive_root)
    # Budget admission for the whole acceptance wave (v6.69.0): a wave that
    # cannot fit the remaining root budget is declined up front as a terminal
    # DEGRADED (no-quorum semantics) instead of dying mid-wave. Route-aware: a
    # session row rides the owner's subscription, not API money, so it is not
    # priced; a native row IS paid API and a packet row renders its REAL message
    # pair. The rare second physical attempt is not multiplied in — fail-open
    # coarse filter, no reservation. Admission decides on the FLOOR-priced wave:
    # ONE work-order send per paid row, no duration or rounds prediction (owner
    # R52). The per-send wallet binding at dispatch still protects money.
    from ouroboros.review_execution import ReviewRouteKind, panel_delivery_class, slot_delivery
    from ouroboros.tools.review_helpers import review_wave_budget_gate

    paid = [slot for slot in slots if getattr(slot, "route", None) is not ReviewRouteKind.AGENT_SESSION]
    if paid:
        try:
            from ouroboros.review_substrate import _messages_char_count, _request_messages

            _prompt_chars = max(
                len(request.slot_session_tasks.get(slot.slot_id, "")) + len(request.policy["output_contract"])
                if getattr(slot, "retrieves", False)
                else _messages_char_count(_request_messages(request, slot))
                for slot in paid
            )
        except Exception:
            _prompt_chars = len(json.dumps(evidence, ensure_ascii=False, default=str))
        floor_models = [getattr(slot, "model", "") for slot in paid]
        _admission = review_wave_budget_gate(
            ctx.tools._ctx,
            surface="task_acceptance",
            models=floor_models,
            prompt_chars=_prompt_chars,
            processing_preferences=[getattr(slot, "processing_preference", "") for slot in paid],
        )
        if _admission is not None:
            return _refused(
                "review_wave_budget_insufficient: estimated "
                f"~${_admission.get('estimated_wave_usd')} > remaining "
                f"${_admission.get('remaining_usd')} (no reviewer was called)"
            )
    free_result = _free_dispatch(request, slots, drive_root=drive_root, usage_ctx=ctx.tools._ctx)
    if free_result is not None:
        return free_result
    refusal = task_acceptance_preclaim_refusal(ctx)
    if refusal is not None:
        return refusal
    # Owner R55: the launch floor was evaluated once, at loop admission; the
    # paid claim below checks cancellation and the wallet only, and a running
    # panel is bounded by the R23 deadline clamps and the per-send wallet fence.
    # Q6: bind the exact tree wallet to the target's physical-dispatch stamp.
    # Route/candidate refusals remain free; one strict stamp gates every slot.
    started = time.monotonic()
    try:
        with bind_task_acceptance_paid_dispatch(ctx) as usage_ctx:
            result = run_review_request(request, slots=slots, drive_root=drive_root, usage_ctx=usage_ctx)
    except TaskAcceptanceDispatchUnavailable as exc:
        return _refused(f"{exc} (no reviewer was called)")
    duration_sec = round(time.monotonic() - started, 3)
    try:
        from ouroboros.review_cycles import review_max_cycles, review_max_cycles_source
        from ouroboros.utils import append_jsonl, utc_now_iso

        # TELEMETRY ONLY (owner R52): a panel that just cost money says what
        # bounded it, how long it ran, how many panels the tree has bought and
        # which deliveries it ran on — "21 paid panels" was invisible until
        # someone summed receipts. Nothing reads this row back to decide.
        _cap = review_max_cycles()
        _deliveries = [slot_delivery(slot) for slot in slots]
        _native_rounds = sum(
            int(actor["usage"].get("native_rounds") or 0)
            for actor in (getattr(result, "actors", None) or [])
            if isinstance(actor, dict) and isinstance(actor.get("usage"), dict)
        )
        append_jsonl(
            task_pacing.acceptance_timing_events_path(ctx.tools._ctx),
            {
                "ts": utc_now_iso(),
                "type": "task_acceptance_review_timing",
                "task_id": str(ctx.task_id),
                "duration_sec": duration_sec,
                "delivery": panel_delivery_class(slots),
                "deliveries": _deliveries,
                "native_rounds": _native_rounds,
                "native_rows": _deliveries.count("native_tool_rounds"),
                "pass_index": ctx.passes_done,
                "aggregate_signal": str(result.aggregate_signal or ""),
                "effective_max_cycles": "unlimited" if _cap is None else _cap,
                "cycles_source": review_max_cycles_source(),
                "total_paid_cycles": _total_paid_acceptance_cycles(ctx),
            },
        )
    except Exception:
        log.debug("Failed to persist task-acceptance timing event", exc_info=True)
    return result


def _end_acceptance_terminal(ctx: Any, status: str, *, outcome: str = "terminal") -> None:
    """Latch the reviewed flag, close the admission fence and checkpoint the
    root at ONE terminal status: the shared tail of every non-revision exit."""
    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome=outcome)
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx, ctx.llm_trace, status=status, pass_index=ctx.passes_done,
    )


def _clean_enforcement_impact(result: Any) -> str:
    """Whether this panel result lets completion through or degrades it."""
    from ouroboros.review_substrate import task_acceptance_is_clean

    return "allows_completion" if task_acceptance_is_clean(result) else "degrades_completion"


def _remember_host_acceptance_run(ctx: Any, run_record: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp the attempt, append the authoritative host run to the trace and
    index it in the process-local binding cache."""
    if type(getattr(ctx.tools._ctx, "task_attempt", None)) is int:
        run_record["task_attempt"] = ctx.tools._ctx.task_attempt
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    return run_record


def _record_host_acceptance_run(ctx: _TaskAcceptanceContext, result: Any) -> Dict[str, Any]:
    """Append the authoritative host result after demoting agent-tool evidence."""
    _mark_agent_acceptance_runs_advisory(ctx.llm_trace)
    for prior in ctx.llm_trace.get("review_runs") or []:
        if (
            isinstance(prior, dict)
            and prior.get("authority") == "host_root"
            and not prior.get("superseded_by_revision")
        ):
            prior["superseded_by_revision"] = True
            prior["superseded_reason"] = "atomically_replaced_by_host_root_review"
    run_record = dict(getattr(result, "__dict__", {}) or {})
    for key in (
        "request", "actors", "parsed_findings", "aggregate_signal", "degraded",
        "degraded_reasons", "single_reviewer_no_diversity",
    ):
        if key not in run_record and hasattr(result, key):
            run_record[key] = getattr(result, key)
    run_record["authority"] = "host_root"
    run_record.update(ctx.review_binding or {})
    run_record["enforcement_impact"] = _clean_enforcement_impact(result)
    return _remember_host_acceptance_run(ctx, run_record)


def _set_applied_host_acceptance_impact(
    run_record: Any,
    result: Any,
    *,
    requires_revision: bool,
) -> None:
    """Record what the host actually did with a panel result."""
    if not isinstance(run_record, dict):
        return
    if requires_revision:
        run_record["enforcement_impact"] = "requires_revision"
        return
    run_record["enforcement_impact"] = _clean_enforcement_impact(result)


def _finish_cyber_acceptance(ctx: _TaskAcceptanceContext, result: Any) -> bool:
    """Main's final response is its decision; criticism retains its own facts."""
    from ouroboros.loop_delivery import delivery_subject_hash
    from ouroboros.review_records import build_author_disposition
    from ouroboros.review_substrate import build_improvement_capsule, task_acceptance_is_clean

    pending = acceptance_run_pending(result)
    if getattr(ctx.tools._ctx, "_acceptance_review_only", False):
        if not pending and (capsule := build_improvement_capsule(result, rails_line=ctx.rails_line)):
            _loop()._append_or_merge_user_message(ctx.messages, capsule)
        _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision")
        return False  # nomination supplies feedback, never final delivery
    if _loop()._task_acceptance_owner_generation_changed(ctx.tools._ctx):
        _loop()._supersede_task_acceptance_for_owner_followup(ctx.tools._ctx, ctx.llm_trace)
        return True
    released = _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision")
    ctx.tools._ctx._task_acceptance_pending = ""  # only the wait; original actors remain custodied
    ctx.tools._ctx._task_acceptance_reviewed = False  # final ingress, not review, owns delivery sealing
    clean = not pending and task_acceptance_is_clean(result)
    signal = "" if pending else str(getattr(result, "aggregate_signal", "") or "")
    author = build_author_disposition(
        disposition="accepted", rationale="Main submitted this complete response for delivery; independent review remains advisory.",
        # Evidence assembly can fail before a review binding exists. Bind the
        # author's decision to its real subject without inventing a reviewed pack.
        subject_hash=ctx.review_binding.get("binding_hash") or delivery_subject_hash(ctx.tools._ctx, ctx.llm_trace, ctx.content),
        reviewer_signal=signal,
        enforcement="advisory", source="author_final_response",
    )
    ctx.llm_trace["review_decision"].update(author_finish=True, review_pending=pending,
                                          admission_released=released)
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_ACCEPTED if clean else ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "clean_pass" if clean else "author_finish", "source": "task_acceptance_review",
        "author_disposition": author, "review_pending": pending,
    })
    ctx.emit_progress("Task acceptance feedback remains advisory; Main chose delivery."
                      + (" Review is still running." if pending else ""))
    return False

def _finish_advisory_author(ctx: _TaskAcceptanceContext) -> bool:
    """Honor an informed Advisory finish or an explicit non-authorizing stop."""
    stance = ctx.llm_trace.get("acceptance_decision") or {}
    intent = stance.get("agent_finish_intent") or {}
    action = str(intent.get("author_action") or "finish")
    if action != "stop" and review_enforcement_blocks(_loop().get_review_enforcement()):
        return False
    feedback = next((run for run in reversed(ctx.llm_trace.get("review_runs") or [])
                     if isinstance(run, dict) and run.get("authority") == "host_root"
                     and run.get("feedback_delivered")), None)
    outcome = ctx.llm_trace.get("acceptance_review_outcome") or {}
    if not feedback and outcome.get("feedback_delivered"):
        feedback = outcome
    disposition = str(stance.get("agent_disposition") or "")
    from ouroboros.loop_delivery import delivery_evidence_fingerprint

    if (not intent or disposition not in {"accepted", "rejected", "partial", "deferred"}
            or (action != "stop" and (not feedback or intent.get("review_binding_hash") != feedback.get("binding_hash")))
            or intent.get("tool_count") != len(ctx.llm_trace.get("tool_calls") or [])
            or intent.get("owner_directives") != len(getattr(ctx.tools._ctx, "_owner_directives", []) or [])
            or intent.get("evidence_fingerprint") != delivery_evidence_fingerprint(ctx.tools._ctx, ctx.llm_trace)):
        return False
    from ouroboros.review_records import build_author_disposition

    author = build_author_disposition(
        disposition=disposition, rationale=str(stance.get("agent_rationale") or ""),
        subject_hash=ctx.review_binding["binding_hash"],
        reviewer_signal=str((feedback or {}).get("aggregate_signal") or ""),
        enforcement="blocking" if review_enforcement_blocks(_loop().get_review_enforcement()) else "advisory",
    )
    author["action"] = action
    from ouroboros.task_results import project_task_acceptance_review_capacity

    capacity = project_task_acceptance_review_capacity(ctx.tools._ctx, task_id=ctx.task_id) if action == "stop" else {}
    terminal_reason = (REASON_REVIEW_CYCLES_EXHAUSTED if action == "stop" and capacity.get("reason") == REASON_REVIEW_CYCLES_EXHAUSTED
                       else "author_stop" if action == "stop" else "author_finish")
    if not _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal"):
        _loop()._supersede_task_acceptance_for_owner_followup(ctx.tools._ctx, ctx.llm_trace)
        return True
    ctx.tools._ctx._task_acceptance_reviewed = True
    ctx.tools._ctx._task_acceptance_pending = ""
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx, ctx.llm_trace, status=author["reviewer_signal"].lower(), pass_index=ctx.passes_done,
    )
    ctx.llm_trace["review_decision"].update({"binding_hash": ctx.review_binding["binding_hash"], "author_finish": action == "finish"})
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": terminal_reason,
        "author_action": action, **({"review_capacity": capacity} if capacity else {}),
        "source": "task_acceptance_review", "author_disposition": author,
        "reviewer_signal": author["reviewer_signal"],
        "reviewer_binding_hash": (feedback or {}).get("binding_hash"),
    })
    ctx.emit_progress("Author stopped with work unfinished; review grants no further action." if action == "stop" else
                     f"Task acceptance review: {author['reviewer_signal']} — author finished advisory review ({disposition}); raw findings retained.")
    return True


def _slot_cause_clause(result: Any) -> str:
    """Bounded chat cause preview; the structured decision retains every complete cause."""
    reasons = list(getattr(result, "degraded_reasons", []) or [])
    note = "; ".join(
        truncate_review_artifact(str(r), limit=300).replace("\n", " ") for r in reasons[:4]
    )
    if len(reasons) > 4:
        note += f" (+{len(reasons) - 4} more in the task result)"
    return f" Causes: {note}" if note else ""


def _offer_acceptance_feedback(ctx: _TaskAcceptanceContext, feedback: str) -> None:
    """Offer exact-panel feedback; the returned Main request proves delivery."""
    runs = ctx.llm_trace.get("review_runs") or []
    for index in range(len(runs) - 1, -1, -1):
        run = runs[index]
        if (isinstance(run, dict) and run.get("authority") == "host_root"
                and run.get("binding_hash") == ctx.review_binding.get("binding_hash")):
            run["feedback_offered"] = True
            ctx.messages.append({"role": "user", "content": feedback, "review_feedback": [{
                "task_id": ctx.task_id, "run_index": index, "binding_hash": str(run.get("binding_hash") or ""),
            }]})
            break


def _apply_task_acceptance_result(
    ctx: _TaskAcceptanceContext,
    result: Any,
    *,
    record_run: bool = True,
    reused: bool = False,
) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    from ouroboros.review_substrate import (
        aggregate_dialogue_status,
        build_improvement_capsule, dissent_findings, task_acceptance_is_clean,
    )

    if record_run:
        _record_host_acceptance_run(ctx, result)
    if not review_enforcement_blocks("blocking"):
        return _finish_cyber_acceptance(ctx, result)
    dissent = dissent_findings(result)
    blocking_lane = review_enforcement_blocks(_loop().get_review_enforcement())
    # Reused panels already have obligations; collecting twice changes evidence
    # revision and could buy a new panel for an identical resubmission (fable r2 #1).
    if blocking_lane and not reused:
        _loop()._collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _loop()._open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
    # Capsule: verdict, open obligation IDs, then money/time/round/pass limits.
    capsule = build_improvement_capsule(
        result,
        rails_line=ctx.rails_line,
        open_obligations=open_obligations,
    )
    # Critic dialogue judgment remains evidence; the author owns its stop choice.
    dialogue = aggregate_dialogue_status(
        result, quorum=_acceptance_dialogue_quorum(result),
    )
    _attach_dialogue_to_host_run(ctx.llm_trace, dialogue)
    if reused and getattr(result, "replayed_from_superseded", False):
        # Evidence superseded this run: replay only its identical-refusal terminal,
        # never a stale PASS that contradicts the trace and current delivery binding.
        return _refuse_identical_acceptance(
            ctx, result,
            dialogue=dialogue, dissent=bool(dissent), open_obligations=open_obligations,
        )
    if task_acceptance_is_clean(result):
        if getattr(ctx.tools._ctx, "_acceptance_review_only", False):
            _offer_acceptance_feedback(
                ctx, capsule or "[Task acceptance feedback] Review verdict: PASS for the nominated answer.")
        _end_acceptance_terminal(ctx, "pass")
        if not _loop()._dispose_obligations_on_clean_pass(
            ctx.llm_trace, result, open_obligations, bool(dissent),
        ):
            _loop()._set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_ACCEPTED,
                "reason": "clean_pass",
                "source": "task_acceptance_review",
                "rationale": "Quorum PASS classified the deliverable solved with criterion evidence.",
                "dissent_noted": bool(dissent),
            })
        ctx.emit_progress("Task acceptance review: PASS (clean acceptance).")
        return False

    if reused:
        return _refuse_identical_acceptance(
            ctx, result,
            dialogue=dialogue, dissent=bool(dissent), open_obligations=open_obligations,
        )

    budget_snapshot = task_pacing.build_budget_snapshot(
        ctx.tools._ctx, profile=ctx.budget_profile,
    )
    pass_ok, pass_reason = task_pacing.improvement_pass_allowed(
        budget_snapshot,
        ctx.passes_done,
        ctx.budget_profile,
        required_blocking=blocking_lane,
        ctx=ctx.tools._ctx,
    )
    actionable = bool(capsule)
    if not capsule:
        capsule = (f"[Acceptance review outcome] {result.aggregate_signal}. "
                   + _slot_cause_clause(result)
                   + " No clean acceptance was established. Inspect this outcome, request another review only if useful and permitted, "
                   "or choose author_action='stop'. Advisory may explicitly finish with a rationale; Blocking still requires fresh approval.")
    if pass_ok:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_REVISION_REQUESTED,
            "reason": "improvement_capsule",
            "source": "task_acceptance_review",
            "rationale": "A compact advisory improvement capsule was fed back for one bounded revision pass.",
            "dissent_noted": bool(dissent),
        })
        ctx.tools._ctx._task_acceptance_improvement_passes = ctx.passes_done + 1
        if not _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision"):
            ctx.tools._ctx._task_acceptance_reviewed = True
            _loop()._set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "fence_reopen_failed",
                "source": "task_acceptance_fence",
                "rationale": "The revision could not safely reopen queue admission at the dispatch boundary.",
            })
            return False
        if open_obligations:
            capsule += _loop()._format_obligations_clause(open_obligations)
        if ctx.content and ctx.content.strip():
            ctx.messages.append({"role": "assistant", "content": ctx.content})
        capsule += (f"\nCritic dialogue assessment: {dialogue['status']}; this is evidence for your decision, not your stop choice. "
                    "The paid review limit does not forbid corrections. If Blocking capacity is spent, preserve corrections and stop; do not claim approval.")
        _offer_acceptance_feedback(ctx, capsule)
        # Name the author pass and actual causes; the verdict alone explains neither.
        causes = _slot_cause_clause(result)
        verdict = str(result.aggregate_signal or "").strip()
        ctx.emit_progress(
            f"Task acceptance review: improvement note fed back for pass "
            f"{ctx.passes_done + 1}"
            + (f" (verdict {verdict}; no causes recorded)." if verdict and not causes else ".")
            + causes
        )
        return True

    _end_acceptance_terminal(ctx, str(result.aggregate_signal or "DEGRADED").lower())
    if _loop()._dispose_obligations_on_clean_pass(
        ctx.llm_trace, result, open_obligations, bool(dissent),
    ):
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} (clean pass; obligations closed)."
        )
        return False
    aggregate_signal = str(result.aggregate_signal or "DEGRADED").upper()
    if aggregate_signal == "DEGRADED":
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "review_degraded",
            "source": "task_acceptance_review",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        # Show the slot failure causes beside the verdict, not only in task_results.
        ctx.emit_progress(
            "Task acceptance review: DEGRADED (no settled verdict; not recorded as PASS)."
            + _slot_cause_clause(result)
        )
        return False
    if capsule and open_obligations:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "open_obligations",
            "source": "task_acceptance_review",
            "rationale": (
                f"Improvement gates exhausted ({pass_reason or 'passes spent'}) with "
                f"{len(open_obligations)} open obligation(s); finalizing honestly."
            ),
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — finalizing with "
            f"{len(open_obligations)} open obligation(s) ({pass_reason or 'passes spent'})."
        )
    elif not actionable:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "reviewer_fail_no_capsule" if aggregate_signal == "FAIL" else "no_actionable_changes",
            "source": "task_acceptance_review", "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(f"Task acceptance review: {result.aggregate_signal} — finalizing without acceptance; no actionable correction was supplied.")
    else:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": (
                "improvement_window_closed"
                if (not ctx.passes_done and pass_reason)
                else "capsule_spent"
            ),
            "source": "task_acceptance_review",
            "rationale": (
                f"Improvement window closed before any capsule pass ({pass_reason})."
                if not ctx.passes_done and pass_reason
                else "The bounded acceptance-review capsule was already spent; finalizing with the current answer."
            ),
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} "
            f"— finalizing without acceptance ({pass_reason or 'author window closed'})."
        )
    return False


def _offer_acceptance_refusal(ctx: _TaskAcceptanceContext, reason: str, *, host_failure: bool = False) -> bool:
    """Let Main react once to unavailable review while ordinary author time remains."""
    previous = ctx.llm_trace.get("acceptance_review_outcome") or {}
    prior_feedback = any(run.get("feedback_delivered") for run in ctx.llm_trace.get("review_runs") or []
                         if isinstance(run, dict) and run.get("authority") == "host_root")
    snapshot = task_pacing.build_budget_snapshot(ctx.tools._ctx, profile=ctx.budget_profile)
    binding = str(ctx.review_binding.get("binding_hash") or "")
    repeated = previous.get("binding_hash") == binding and previous.get("reason") == reason
    if ((not host_failure and prior_feedback) or (previous.get("feedback_delivered") and (not host_failure or repeated))
            or not task_pacing.improvement_pass_allowed(snapshot, ctx.passes_done, ctx.budget_profile)[0]):
        return False
    ctx.llm_trace["acceptance_review_outcome"] = {"binding_hash": binding, "reason": reason}
    ctx.messages.append({"role": "user", "content": (
        (f"Acceptance host processing failed: {reason}. Original reviewer evidence is retained; no replacement review was called. " if host_failure else
         f"Acceptance review could not start: {reason}. No new reviewer was called. ") +
        "You may preserve corrections or stop; Blocking grants no advancement without fresh review. "
        "Advisory may finish explicitly with a rationale."), "review_feedback": [{
            "task_id": ctx.task_id, "outcome_binding_hash": binding}]})
    _loop()._set_acceptance_decision(ctx.llm_trace, {"status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "review_outcome_received", "source": "task_acceptance_review"})
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision")
    return True


def _record_acceptance_infra_failure(ctx: _TaskAcceptanceContext, exc: Exception) -> bool:
    """Finish an eligible mandatory panel as DEGRADED, never as a silent skip."""
    safe_error = _loop()._extract_plain_text_from_content(str(exc))[:2000]
    _mark_agent_acceptance_runs_advisory(ctx.llm_trace)
    run_record = {
        "request": {"surface": "task_acceptance", "task_id": ctx.task_id},
        "actors": [],
        "parsed_findings": [{
            "severity": "critical",
            "item": "task_acceptance_infra_failure",
            "evidence": f"{type(exc).__name__}: {safe_error}",
            "recommendation": "Do not report semantic success unless the failure is explicitly accounted for.",
        }],
        "aggregate_signal": "DEGRADED",
        "degraded": True,
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
        "authority": "host_root",
        **(ctx.review_binding or {}),
        "enforcement_impact": "degrades_completion",
    }
    if not any(isinstance(run, dict) and all(run.get(key) == run_record.get(key)
            for key in ("authority", "binding_hash", "degraded_reasons", "parsed_findings"))
            for run in ctx.llm_trace.get("review_runs") or []):
        _remember_host_acceptance_run(ctx, run_record)
    if not review_enforcement_blocks("blocking"):
        from types import SimpleNamespace
        return _finish_cyber_acceptance(ctx, SimpleNamespace(**run_record))
    ctx.emit_progress("Task acceptance review could not establish a verdict; returning the failure to its author.")
    if _offer_acceptance_refusal(ctx, f"{type(exc).__name__}: {safe_error}", host_failure=True):
        return True
    _end_acceptance_terminal(ctx, "infra_failure")
    _loop()._set_acceptance_decision(ctx.llm_trace, {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "review_degraded", "source": "task_acceptance_review", "degraded_reasons": run_record["degraded_reasons"]})
    return False


def _disposition_reason_sha256(reason: Any) -> str:
    """Content identity of one obligation-disposition reason; "" when blank.

    Mirrors ``commit_gate.compute_rebuttal_sha256`` on purpose: on BOTH gates an
    empty rebuttal is not an argument and buys no paid cycle."""
    import hashlib

    text = str(reason or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def acceptance_paid_identity(candidate_hash: str, llm_trace: Dict[str, Any]) -> str:
    """Bind the reviewed subject and substantive author dispositions to one charge.

    The caller supplies result/criteria/material-evidence identity; historical
    direct callers may still supply a candidate hash. Growing source transcripts
    and ingress counters are forensic facts, not semantic criteria changes.
    Empty disposition reasons do not mint a new panel.
    """
    import hashlib

    material = sorted({
        (
            str(row.get("id") or "").strip(),
            str(row.get("disposition") or "").strip().lower(),
            _disposition_reason_sha256(row.get("disposition_reason")),
        )
        for row in (llm_trace.get("acceptance_obligations") or [])
        if isinstance(row, dict)
        and str(row.get("id") or "").strip()
        and str(row.get("disposition") or "").strip()
        and _disposition_reason_sha256(row.get("disposition_reason"))
    })
    payload = json.dumps(
        [str(candidate_hash or ""), [list(item) for item in material]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_acceptance_paid_identity(
    review_binding: Dict[str, Any], llm_trace: Dict[str, Any],
) -> str:
    """Stamp the A-material paid identity onto a freshly built review binding.

    The binding keeps carrying its three hashes (the supersede paths still need
    the evidence revision); ``paid_identity`` rides ALONGSIDE them and is what the
    wallet claim and the free-replay lookup key on."""
    identity = acceptance_paid_identity(
        str(review_binding.get("subject_hash") or review_binding.get("candidate_hash") or ""), llm_trace,
    )
    review_binding["paid_identity"] = identity
    return identity


def acceptance_dialogue_history(llm_trace: Dict[str, Any], *, limit: int = 6) -> List[Dict[str, Any]]:
    """Bounded per-panel history of the dialogue so far, for the NEXT reviewer.

    Reviewers were adjudicating each round blind to the previous rounds' typed
    judgement, which is most of why the same finding kept being re-raised. The
    rows are tiny host facts already recorded on the run records; the caller
    attaches them to the evidence packet OUTSIDE the hashed material
    (``review_evidence.UNHASHED_EVIDENCE_KEYS``) so reading the history can never
    mint a fresh evidence revision — and therefore never a fresh paid binding."""
    rows: List[Dict[str, Any]] = []
    for run in (llm_trace.get("review_runs") or []):
        if not isinstance(run, dict) or str(run.get("authority") or "") != "host_root":
            continue
        dialogue = run.get("dialogue") if isinstance(run.get("dialogue"), dict) else {}
        votes = dialogue.get("votes") if isinstance(dialogue.get("votes"), dict) else {}
        rows.append({
            "round": len(rows) + 1,
            "aggregate_signal": str(run.get("aggregate_signal") or "").upper(),
            "dialogue_status": str(dialogue.get("status") or ""),
            "votes": {str(k): len(v or []) for k, v in votes.items()},
        })
    obligations = [
        row for row in (llm_trace.get("acceptance_obligations") or [])
        if isinstance(row, dict)
    ]
    if rows:
        rows[-1]["obligations_new"] = sum(
            1 for row in obligations if not int(row.get("reopened_count") or 0)
        )
        rows[-1]["obligations_re_raised"] = sum(
            1 for row in obligations if int(row.get("reopened_count") or 0)
        )
    return rows[-max(1, int(limit)):]


def _refuse_identical_acceptance(
    ctx: Any,
    result: Any,
    *,
    dialogue: Dict[str, Any],
    dissent: bool,
    open_obligations: List[Dict[str, Any]],
) -> bool:
    """Terminate a resubmit whose A-material paid identity was already bought.

    The recorded verdict is replayed for FREE and quoted; the improvement capsule
    is deliberately NOT re-entered. Feeding the note again asks for a round the
    agent has already answered with nothing new, and every such round shifted the
    evidence revision into a fresh paid binding — the 21-panel pump. The dialogue
    record and the decision row still land, so the replay stays auditable."""
    from ouroboros.outcomes import REASON_IDENTICAL_ACCEPTANCE_REFUSED

    _end_acceptance_terminal(ctx, str(result.aggregate_signal or "DEGRADED").lower())
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": REASON_IDENTICAL_ACCEPTANCE_REFUSED,
        "source": "task_acceptance_review",
        "rationale": (
            "No new material since the paid panel: neither the candidate answer nor "
            "any obligation disposition changed. Quoting the recorded verdict "
            f"({str(result.aggregate_signal or 'DEGRADED').upper()}; dialogue "
            f"{dialogue['status']}) with {len(open_obligations)} open "
            "obligation(s); no further round."
        ),
        "dialogue_status": dialogue["status"],
        "dialogue_votes": dialogue["votes"],
        "dissent_noted": dissent,
        "open_obligations": [str(item.get("id")) for item in open_obligations],
    })
    ctx.emit_progress(
        f"Task acceptance review: {result.aggregate_signal} — identical paid identity "
        "(no changed answer, no new obligation disposition); the recorded verdict "
        "stands and no further panel is bought."
    )
    return False


def _prior_acceptance_run(
    tools_ctx: Any,
    llm_trace: Dict[str, Any],
    binding_hash: str,
    *,
    paid_identity: str = "",
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Locate the authoritative host run already recorded for this submission:
    first the trace (survives requeue replay), then the process-local
    ``_task_acceptance_seen_bindings`` cache. Returns (cache, prior_run).

    EITHER identity replays for free: the same binding hash (byte-identical
    submission, as before) OR the same A-material ``paid_identity`` — unchanged
    candidate answer and no new obligation disposition — which is the identity the
    tree's wallet actually bought."""
    seen_bindings = getattr(tools_ctx, "_task_acceptance_seen_bindings", None)
    if not isinstance(seen_bindings, dict):
        seen_bindings = {}
        tools_ctx._task_acceptance_seen_bindings = seen_bindings
    identity = str(paid_identity or "")

    def _matches(run: Any) -> bool:
        return isinstance(run, dict) and (
            str(run.get("binding_hash") or "") == binding_hash
            or bool(identity and str(run.get("paid_identity") or "") == identity)
        )

    prior_run = next(
        (
            run for run in reversed(llm_trace.get("review_runs") or [])
            if isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
            and _matches(run)
        ),
        None,
    )
    if prior_run is None:
        prior_run = next(
            (
                run for run in reversed(list(seen_bindings.values()))
                if isinstance(run, dict)
                and not run.get("superseded_by_revision")
                and _matches(run)
            ),
            None,
        )
    if prior_run is None and identity:
        # A run superseded by an evidence revision is stale as a CURRENT
        # acceptance, but the wallet already bought its A-material. When the
        # resubmission carries the SAME paid identity (unchanged candidate, no
        # new nonempty disposition), the recorded verdict replays for free —
        # otherwise the dispatch claim refuses `binding_dispatch_already_claimed`
        # and the loop records a synthetic DEGRADED panel instead of the typed
        # identical-refusal terminal the contract requires. Evidence revision
        # is stale-DETECTION, never a paid-cycle mint (owner decision 5=A).
        superseded = next(
            (
                run for run in reversed(llm_trace.get("review_runs") or [])
                if isinstance(run, dict)
                and run.get("authority") == "host_root"
                and run.get("superseded_by_revision")
                and str(run.get("paid_identity") or "") == identity
            ),
            None,
        )
        if superseded is not None:
            prior_run = dict(superseded)
            prior_run["replayed_from_superseded"] = True
    return seen_bindings, prior_run


def _direct_context_fence_state(tools_ctx: Any, fence_token: Any) -> Any:
    """Review-binding fence state: the queue-owned token when present, else the
    direct-chat generations (no queue fence exists for a direct context)."""
    if fence_token is not None:
        return fence_token
    return {
        "state": "direct_context",
        "owner_generation": getattr(tools_ctx, "_task_acceptance_owner_generation", None),
        "queue_generation": getattr(tools_ctx, "_task_acceptance_fence_generation", None),
    }


def _acceptance_delivery_slots() -> list:
    """The SAME triad rows the acceptance panel dispatches (R2); a malformed
    structured config sizes an empty packet here and refuses typed at the panel."""
    from ouroboros.review_substrate import triad_delivery_slots

    try:
        return list(triad_delivery_slots(role_hint="task acceptance"))
    except ValueError:
        return []



def _skip_task_acceptance_for_launch_reason(
    tools_ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    launch_reason: str,
    snapshot: Any,
    passes_done: int,
    emit_progress: Callable[[str], None],
) -> bool:
    """The launch rule's skip terminal for its ONE evaluation site — the loop's
    admission gate (owner R55): no reviewer run is recorded, and
    `outcomes.derive_loop_outcome` keys on the (status, typed REASON) pair
    below with source `task_pacing`."""
    tools_ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(tools_ctx, outcome="terminal")
    _loop()._mark_root_acceptance_checkpoint(
        tools_ctx, llm_trace, status=launch_reason, pass_index=passes_done,
    )
    llm_trace["review_decision"].update({"skipped": launch_reason})
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": launch_reason,
        "source": "task_pacing",
        "rationale": (
            f"Spendable {snapshot.spendable_sec:.0f}s (remaining "
            f"{snapshot.remaining_sec:.0f}s, reserve {snapshot.reserve_sec:.0f}s) "
            f"is at or below the {task_pacing._acceptance_floor_sec():.0f}s floor."
        ),
    })
    emit_progress("Task acceptance skipped: spendable at or below floor.")
    return False



def _observe_host_acceptance_request(tools: Any, llm_trace: dict, task_id: str, mode: str) -> tuple:
    """Bind eligibility and previous agent advice before opening host admission."""
    lineage = _resolve_ctx_lineage(tools._ctx, task_id)
    eligible, trigger = _loop()._task_acceptance_eligible(
        mode,
        llm_trace,
        bool(getattr(tools._ctx, "is_direct_chat", False)),
        is_root_task=bool(lineage["is_root_task"]),
        task_contract=(
            tools._ctx.task_contract
            if isinstance(getattr(tools._ctx, "task_contract", None), dict)
            else {}
        ),
    )
    agent_called = any(
        isinstance(call, dict) and str(call.get("tool") or "") == "task_acceptance_review"
        for call in (llm_trace.get("tool_calls") or [])
    )
    agent_review_present = any(
        isinstance(run, dict)
        and isinstance(run.get("request"), dict)
        and str((run.get("request") or {}).get("surface") or "") == "task_acceptance"
        and str(run.get("aggregate_signal") or "").strip()
        for run in (llm_trace.get("review_runs") or [])
    )
    if agent_review_present:
        _mark_agent_acceptance_runs_advisory(llm_trace)
        trigger = f"{trigger}_after_agent_advisory"
    elif agent_called:
        trigger = f"{trigger}_after_agent_tool"
    llm_trace["review_decision"] = {
        "eligibility": "eligible" if eligible else "not_eligible", "trigger": trigger,
    }
    return eligible, trigger


def _run_task_acceptance_review_once(
    *,
    tools: ToolRegistry,
    content: str,
    task_id: str,
    task_type: str,
    llm_trace: Dict[str, Any],
    drive_root: Optional[pathlib.Path],
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> bool:
    """Run the root-owned acceptance gate once for the current deliverable.
    Loop-side rails facts arrive via the ``_acceptance_loop_rails`` ctx stash
    (set by ``_no_tool_final_answer``; keeps the signature at 8 params)."""
    mode = _loop().get_task_review_mode()
    _loop()._latch_final_answer_marker(llm_trace, content)
    if getattr(tools._ctx, "_task_acceptance_reviewed", False):
        from ouroboros.loop_delivery import delivery_subject_hash

        reviewed_subject = getattr(tools._ctx, "_task_acceptance_reviewed_subject", "")
        if not reviewed_subject or reviewed_subject == delivery_subject_hash(tools._ctx, llm_trace, content):
            return False
        tools._ctx._task_acceptance_reviewed = False
    from ouroboros.review_evidence import acceptance_packet_budget_chars

    eligible, trigger = _observe_host_acceptance_request(tools, llm_trace, task_id, mode)
    if not eligible:
        return False
    # Owner hurry (§19.7.2 item 8): AFTER structural eligibility is known,
    # BEFORE acceptance-fence/quiescence/reviewer admission, an armed latch
    # skips the next otherwise-eligible panel with the typed reason — no
    # reviewer calls (an in-flight panel is never cancelled/relabeled).
    from ouroboros.owner_hurry import acceptance_skip_applied, effective_budget_profile

    if acceptance_skip_applied(
        tools._ctx, llm_trace, task_id=task_id, drive_root=drive_root,
        set_decision=_loop()._set_acceptance_decision, emit_progress=emit_progress,
    ):
        return False
    fence_ok, _fence_token = _loop()._begin_task_acceptance_fence(tools._ctx, task_id)
    if not fence_ok and review_enforcement_blocks("blocking"):
        llm_trace["review_decision"] = {
            "eligibility": "acceptance_fence_failed", "trigger": trigger,
        }
        _loop()._append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The supervisor could not atomically close "
            "subtask admission. Do not finalize or spawn more work; retry after the "
            "queue fence is available.",
        )
        emit_progress("Task acceptance review waiting for the queue-owned admission fence.")
        return True
    quiescent, subtree_statuses = _loop()._task_acceptance_subtree_snapshot(
        tools._ctx, drive_root, task_id,
    )
    if not quiescent and review_enforcement_blocks("blocking"):
        llm_trace["review_decision"] = {
            "eligibility": "waiting_for_quiescence",
            "trigger": trigger,
            "live_descendants": [
                row for row in subtree_statuses
                if str(row.get("status") or "")
                not in {"completed", "failed", "cancelled", "rejected_duplicate"}
            ],
        }
        _loop()._append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The root acceptance review requires the recursive "
            "subtree to be terminal. Absorb or explicitly cancel the remaining child "
            "tasks before finalizing.",
        )
        emit_progress("Task acceptance review waiting for recursive subtree quiescence.")
        return True
    llm_trace["review_decision"].update(admission_fence_available=fence_ok, subtree_quiescent=quiescent)
    # One effective profile carries explicit author caps/Hurry to gates and display.
    budget_profile = effective_budget_profile(
        tools._ctx, task_pacing.resolve_budget_profile(tools._ctx),
    )
    budget_snapshot = task_pacing.build_budget_snapshot(tools._ctx, profile=budget_profile)
    passes_done = int(getattr(tools._ctx, "_task_acceptance_improvement_passes", 0))
    review_ctx = _TaskAcceptanceContext(
        tools=tools, content=content, task_id=task_id, task_type=task_type,
        llm_trace=llm_trace,
        drive_root=drive_root,
        messages=messages,
        emit_progress=emit_progress,
        mode=mode,
        subtree_statuses=subtree_statuses,
        budget_profile=budget_profile,
        passes_done=passes_done,
        evidence={},
        review_binding={},
        rails_line=task_pacing.acceptance_rails_line(
            budget_snapshot,
            budget_profile,
            passes_done,
            getattr(tools._ctx, "_acceptance_loop_rails", None),
            required_blocking=(
                mode == "required" and review_enforcement_blocks(_loop().get_review_enforcement())
            ), workspace=task_pacing._workspace_delivery(tools._ctx),
        ),
        packet_budget_chars=acceptance_packet_budget_chars(_acceptance_delivery_slots()),
    )
    try:
        from ouroboros.review_dispatch import reconcile_pending_acceptance_runs

        reconcile_pending_acceptance_runs(
            llm_trace, drive_root=drive_root or tools._ctx.drive_root, usage_ctx=tools._ctx)
        from types import SimpleNamespace

        from ouroboros.review_substrate import build_review_binding

        from ouroboros.loop_delivery import delivery_subject_hash

        review_ctx.review_binding = {"binding_hash": delivery_subject_hash(tools._ctx, llm_trace, content)}
        if _loop()._task_acceptance_owner_generation_changed(tools._ctx):
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            return True
        if _finish_advisory_author(review_ctx):
            return not bool(getattr(tools._ctx, "_task_acceptance_reviewed", False))
        review_ctx.review_binding = {}
        review_ctx.evidence = _build_host_acceptance_evidence(review_ctx)
        review_ctx.review_binding = build_review_binding(
            candidate=content,
            evidence=review_ctx.evidence,
            fence_token_or_state=_direct_context_fence_state(tools._ctx, _fence_token),
        )
        review_ctx.review_binding["subject_hash"] = delivery_subject_hash(tools._ctx, llm_trace, content)
        from ouroboros.loop_messages import owner_source_sha256

        review_ctx.review_binding["owner_source_sha256"] = owner_source_sha256(tools._ctx)  # the premises this panel judged
        if _loop()._task_acceptance_owner_generation_changed(tools._ctx):
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            return True
        binding_hash = str(review_ctx.review_binding.get("binding_hash") or "")
        # A-material: what the tree's wallet actually buys. Stamped onto the
        # binding before the free-replay lookup and the dispatch claim both read it.
        paid_identity = bind_acceptance_paid_identity(review_ctx.review_binding, llm_trace)
        seen_bindings, prior_run = _prior_acceptance_run(
            tools._ctx, llm_trace, binding_hash, paid_identity=paid_identity,
        )
        from ouroboros.acceptance_settlement import _deliver_under_running_panel

        handled = _deliver_under_running_panel(review_ctx, prior_run)
        if handled is not None:
            return handled
        reused_result = None
        applied_before = bool(prior_run and (prior_run.get("applied_decision") or prior_run.get("feedback_delivered")))
        if prior_run is not None:
            seen_bindings[binding_hash] = prior_run
            if prior_run not in (llm_trace.get("review_runs") or []):
                llm_trace.setdefault("review_runs", []).append(dict(prior_run))
            llm_trace["review_decision"].update({
                "panel_reused": True,
                "panel_id": str(prior_run.get("panel_id") or ""),
                "binding_hash": binding_hash,
            })
            emit_progress(
                "Task acceptance review: reusing the authoritative result for the unchanged binding."
            )
            # Free replay re-applies the recorded outcome without another panel.
            reused_result = SimpleNamespace(**prior_run)
            # The original forensic binding remains the authority of the paid
            # operation even when Main consumed a harmless new source message.
            review_ctx.review_binding = {"subject_hash": review_ctx.review_binding["subject_hash"], **{key: prior_run[key] for key in (
                "candidate_hash", "evidence_revision", "fence_hash", "binding_hash",
                "panel_id", "paid_identity", "subject_hash",
            ) if key in prior_run}}
        elif binding_hash in seen_bindings:
            # A process-local attempt without its authoritative trace is not
            # safe to repeat or silently accept. The infra-degraded path below
            # records the missing authority and closes finalization honestly.
            raise RuntimeError("acceptance binding was attempted but its host run is unavailable")
        else:
            launch_ok, launch_reason = task_pacing.review_launch_allowed(budget_snapshot)
            if not launch_ok:
                if _offer_acceptance_refusal(review_ctx, launch_reason):
                    return True
                return _skip_task_acceptance_for_launch_reason(
                    tools._ctx, llm_trace, launch_reason=launch_reason,
                    snapshot=budget_snapshot, passes_done=passes_done, emit_progress=emit_progress,
                )
            from ouroboros.task_results import project_task_acceptance_review_capacity

            capacity = project_task_acceptance_review_capacity(tools._ctx, task_id=task_id, paid_identity=paid_identity)
            if capacity.get("state") != "available":
                reason = str(capacity.get("reason") or "review_capacity_unknown")
                llm_trace["review_decision"]["dispatch_refusal"] = dict(capacity)
                if reason == REASON_REVIEW_CYCLES_EXHAUSTED:
                    from ouroboros.review_cycles import emit_review_cycles_exhausted
                    emit_review_cycles_exhausted(getattr(tools._ctx, "event_queue", None), tools._ctx.drive_root,
                        surface="task_acceptance", task_id=task_id, cycles_paid=int(capacity.get("claimed_cycles") or 0),
                        cap=int(capacity.get("cap_cycles") or 0), enforcement=_loop().get_review_enforcement())
                if _offer_acceptance_refusal(review_ctx, reason):
                    return True
                _end_acceptance_terminal(review_ctx, reason)
                _loop()._set_acceptance_decision(llm_trace, {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                    "reason": REASON_REVIEW_CYCLES_EXHAUSTED if reason == REASON_REVIEW_CYCLES_EXHAUSTED else "review_degraded",
                    "source": "task_acceptance_review", "rationale": "Current work is retained without a fresh reviewer verdict; no new panel was dispatched."})
                emit_progress(f"Acceptance review: {reason}; current work retained, no fresh approval.")
                return False
            seen_bindings[binding_hash] = None
        llm_trace["review_decision"].update({
            "panel_id": str(review_ctx.review_binding.get("panel_id") or ""),
            "binding_hash": str(review_ctx.review_binding.get("binding_hash") or binding_hash),
        })
        messages_before_apply = list(messages)
        obligations_were_present = "acceptance_obligations" in llm_trace
        obligations_before_apply = [
            dict(row) if isinstance(row, dict) else row
            for row in (llm_trace.get("acceptance_obligations") or [])
        ]
        passes_before_apply = int(
            getattr(tools._ctx, "_task_acceptance_improvement_passes", 0) or 0
        )
        panel_result = reused_result or _loop()._execute_task_acceptance_panel(review_ctx)
        run_record = prior_run if reused_result is not None else _record_host_acceptance_run(review_ctx, panel_result)
        if acceptance_run_pending(panel_result):
            tools._ctx._task_acceptance_pending = str(run_record.get("binding_hash") or "")
            run_record["enforcement_impact"] = "pending_feedback"
            from ouroboros.acceptance_settlement import remember_settlement_trace

            remember_settlement_trace(tools._ctx, llm_trace, run_record)
            llm_trace["review_decision"].update({
                "eligibility": "review_in_flight", "operation_state": "in_flight",
            })
            emit_progress("Task acceptance review is running; Main can receive and answer messages.")
            from ouroboros.acceptance_settlement import acceptance_wait_chosen

            if not acceptance_wait_chosen(tools._ctx):
                return _finish_cyber_acceptance(review_ctx, panel_result)
            return True
        tools._ctx._task_acceptance_pending = ""
        if _loop()._task_acceptance_owner_generation_changed(tools._ctx):
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        fresh_quiescent, fresh_subtree_statuses = _loop()._task_acceptance_subtree_snapshot(
            tools._ctx, drive_root, task_id,
        )
        fresh_subject = delivery_subject_hash(tools._ctx, llm_trace, content)
        stale_reason = ""
        if not fresh_quiescent and review_enforcement_blocks("blocking"):
            stale_reason = "host_acceptance_subtree_became_non_quiescent"
        elif fresh_subject != review_ctx.review_binding["subject_hash"]:
            stale_reason = "host_acceptance_subject_changed"
        if stale_reason:
            _loop()._supersede_task_acceptance_for_evidence_change(
                tools._ctx,
                llm_trace,
                run_record,
                stale_reason,
                messages,
                emit_progress,
            )
            return True
        another_round = _apply_task_acceptance_result(
            review_ctx,
            panel_result,
            record_run=False,
            reused=applied_before,
        )
        if getattr(tools._ctx, "_task_acceptance_fence_generation_mismatch", False):
            messages[:] = messages_before_apply
            if obligations_were_present:
                llm_trace["acceptance_obligations"] = obligations_before_apply
            else:
                llm_trace.pop("acceptance_obligations", None)
            tools._ctx._task_acceptance_improvement_passes = passes_before_apply
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        _set_applied_host_acceptance_impact(
            run_record,
            panel_result,
            requires_revision=another_round,
        )
        tools._ctx._task_acceptance_reviewed_subject = review_ctx.review_binding["subject_hash"]
        return another_round
    except Exception as exc:
        log.debug("Mandatory task acceptance review failed", exc_info=True)
        return _record_acceptance_infra_failure(review_ctx, exc)
    finally:
        publish_acceptance_checkpoint(tools._ctx, llm_trace, task_id=task_id, drive_root=drive_root)
