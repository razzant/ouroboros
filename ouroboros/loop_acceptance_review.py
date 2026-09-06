"""The host-forced acceptance review run: the checklist, the panel execution,
the dialogue quorum, applying the panel result, infra-failure records and the
one-shot review entrypoint. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name."""

from __future__ import annotations

import json
import logging
import pathlib
import time

import dataclasses

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.review_projection import publish_acceptance_checkpoint
from ouroboros.tools.registry import ToolRegistry
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

    committed_this_turn = any(
        isinstance(call, dict)
        and str(call.get("tool") or "") in ("commit_reviewed", "vcs_commit_reviewed")
        and str(call.get("status") or "") == "ok"
        for call in (ctx.llm_trace.get("tool_calls") or [])
    )
    evidence = build_task_acceptance_evidence(
        ctx.tools._ctx,
        llm_trace=ctx.llm_trace,
        drive_root=ctx.drive_root,
        task_id=ctx.task_id,
        task_type=ctx.task_type,
        agent_evidence=_latest_agent_acceptance_evidence(ctx.llm_trace),
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
    """The packet a NATIVE row receives (R4/R15): the same host-attested exhibits
    WITHOUT the freely degradable tail the api ladder spends first — the
    tool-trajectory rows and artifact previews — because that row reads those
    sources itself at the pointers. Every section key survives, so an
    `evidence_ref` naming it still resolves against the FULL dict (the ref
    authority never changes), and the omission is manifested like every other."""
    packet = dict(evidence)
    manifest_present = "omissions_manifest" in packet
    manifest = packet.get("omissions_manifest")
    # A sequence is a manifest; anything else present (None, a dict, a string) is
    # malformed and is normalized to an empty list — never carried as-is, never its keys.
    omissions = list(manifest) if isinstance(manifest, (list, tuple)) else []
    trajectory = packet.get("tool_trajectory")
    if isinstance(trajectory, list) and trajectory:
        packet["tool_trajectory"] = [{
            "retrieve": "tool-trajectory rows withheld from this delivery; read the trajectory log at the pointer",
            "calls": len(trajectory),
        }]
        omissions.append({"section": "tool_trajectory", "omitted": len(trajectory), "reason": "retrieving_delivery"})
    artifacts = packet.get("artifacts")
    if isinstance(artifacts, list):
        rows = [
            {k: v for k, v in row.items() if k != "preview"} if isinstance(row, dict) and row.get("preview") else row
            for row in artifacts
        ]
        stripped = sum(1 for before, after in zip(artifacts, rows) if before is not after)
        if stripped:
            packet["artifacts"] = rows
            omissions.append({"section": "artifact_previews", "omitted": stripped, "reason": "retrieving_delivery"})
    if omissions or (manifest_present and not isinstance(manifest, list)):
        packet["omissions_manifest"] = omissions  # normalized whenever present and not a list; an absent key stays absent
    return packet



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
        task_id=ctx.task_id, retry_key=f"task_acceptance:{task_acceptance_evidence_revision(evidence)}",
        deadline_at=_owner_deadline_at(ctx.tools._ctx),  # R23: the owner window bounds every row
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
            ctx.tools._ctx, surface="task_acceptance", models=floor_models, prompt_chars=_prompt_chars,
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
    if type(getattr(ctx.tools._ctx, "task_attempt", None)) is int:
        run_record["task_attempt"] = ctx.tools._ctx.task_attempt
    run_record.update(ctx.review_binding or {})
    aggregate = str(run_record.get("aggregate_signal") or "DEGRADED").upper()
    run_record["enforcement_impact"] = (
        "allows_completion"
        if aggregate == "PASS"
        else "degrades_completion"
    )
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    return run_record


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
    from ouroboros.review_substrate import task_acceptance_is_clean

    run_record["enforcement_impact"] = (
        "allows_completion" if task_acceptance_is_clean(result) else "degrades_completion"
    )


def _apply_task_acceptance_result(
    ctx: _TaskAcceptanceContext,
    result: Any,
    *,
    record_run: bool = True,
    reused: bool = False,
) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    from ouroboros.review_substrate import (
        DIALOGUE_TERMINAL_STATUSES,
        aggregate_dialogue_status,
        build_improvement_capsule,
        dissent_findings,
        task_acceptance_is_clean,
    )

    if record_run:
        _record_host_acceptance_run(ctx, result)
    dissent = dissent_findings(result)
    blocking_lane = ctx.mode == "required" and _loop().get_review_enforcement() == "blocking"
    # A REUSED panel (unchanged binding) is the SAME reviewer act applied
    # again: re-collecting would mutate reviewer-authored state with no new
    # input, and the shifted evidence revision would buy a fresh paid panel
    # for a byte-identical resubmit (fable r2 #1); rows already collected.
    if blocking_lane and not reused:
        _loop()._collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _loop()._open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
    # v6.74.0 (A1): the capsule leads with the verdict, the concrete open
    # obligation ids, and the pre-rendered rails line (money/time/rounds/passes).
    capsule = build_improvement_capsule(
        result,
        rails_line=ctx.rails_line,
        open_obligations=open_obligations,
    )
    # v6.74.0 (A5): the reviewers' typed dialogue judgement, reduced over the
    # CONTRIBUTING actors with the panel's own quorum; persisted for audit on
    # the authoritative run record whatever branch applies below. `inconclusive`
    # (no well-formed vote at all) grants the dialogue NO authority: it is not a
    # terminal verdict and not a licence to continue — the existing non-dialogue
    # terminals below decide, exactly as they did before the dialogue existed.
    dialogue = aggregate_dialogue_status(
        result, quorum=_acceptance_dialogue_quorum(result),
    )
    _attach_dialogue_to_host_run(ctx.llm_trace, dialogue)
    dialogue_terminal = dialogue["status"] in DIALOGUE_TERMINAL_STATUSES
    if reused and getattr(result, "replayed_from_superseded", False):
        # A run superseded by an evidence revision replays ONLY into the typed
        # identical-refusal terminal — never into clean-PASS authorization: its
        # verdict predates the evidence change, so re-accepting would stamp a
        # stale PASS (and the trace's superseded rows would contradict the
        # applied decision — the delivery binding could never match). The
        # refusal is conservative and consistent: nothing new was bought,
        # nothing stale is re-authorized.
        return _refuse_identical_acceptance(
            ctx, result,
            dialogue=dialogue, dissent=bool(dissent), open_obligations=open_obligations,
        )
    if task_acceptance_is_clean(result):
        ctx.tools._ctx._task_acceptance_reviewed = True
        _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _loop()._mark_root_acceptance_checkpoint(
            ctx.tools._ctx, ctx.llm_trace, status="pass", pass_index=ctx.passes_done,
        )
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
    # A DEGRADED panel (no valid verdict quorum) cannot "judge" the dialogue:
    # a lone terminal vote from the one contributing slot must NOT shadow the
    # review_degraded path below, which is the only surface carrying the
    # per-slot causes and degraded_reasons the v6.70.0 honesty invariant (P1)
    # requires. Letting the dialogue-terminal branch fire here recorded a false
    # "reviewer quorum judged" rationale and silently dropped those causes.
    if dialogue_terminal and str(result.aggregate_signal or "DEGRADED").upper() != "DEGRADED":
        # v6.74.0 (A5): a reviewer quorum judged the dialogue no longer
        # actionable (unreachable_here / stable_disagreement). Finalize via
        # the EXISTING honest path recording BOTH positions in one
        # owner-visible line — reviewer authorship, not a host timer.
        ctx.tools._ctx._task_acceptance_reviewed = True
        _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _loop()._mark_root_acceptance_checkpoint(
            ctx.tools._ctx,
            ctx.llm_trace,
            status=str(result.aggregate_signal or "DEGRADED").lower(),
            pass_index=ctx.passes_done,
        )
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            # The with/without-obligations distinction moves from the status token to
            # the `open_obligations` id list this branch already records.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "dialogue_terminal",
            "source": "task_acceptance_review",
            "rationale": (
                f"Reviewer quorum judged the dialogue {dialogue['status']}; "
                "finalizing honestly with both positions recorded "
                f"({len(open_obligations)} open obligation(s))."
            ),
            "dialogue_status": dialogue["status"],
            "dialogue_votes": dialogue["votes"],
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — reviewer quorum judged "
            f"the dialogue {dialogue['status']}; finalizing with "
            f"{len(open_obligations)} open obligation(s)."
        )
        return False
    if capsule and pass_ok:
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
        _loop()._append_or_merge_user_message(ctx.messages, capsule)
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — improvement note fed back."
        )
        return True

    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
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
            "rationale": "Acceptance reviewers did not reach a valid quorum.",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        # Per-slot causes were always in the structured decision; the
        # owner-visible line said only "no valid quorum", forcing a dig
        # through task_results for WHICH slot failed and why (v6.70.0).
        _degraded_reasons = list(getattr(result, "degraded_reasons", []) or [])
        # Bounded PREVIEW for the chat line only — the complete causes live in
        # the structured decision record (owner-facing full copy, per the
        # v6.70.0 honesty invariant).
        _reason_note = "; ".join(
            truncate_review_artifact(str(r), limit=300).replace("\n", " ")
            for r in _degraded_reasons[:4]
        )
        if len(_degraded_reasons) > 4:
            _reason_note += f" (+{len(_degraded_reasons) - 4} more in the task result)"
        ctx.emit_progress(
            "Task acceptance review: DEGRADED (no valid quorum; not recorded as PASS)."
            + (f" Causes: {_reason_note}" if _reason_note else "")
        )
        return False
    if capsule and open_obligations:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else "open_obligations",
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
    elif capsule:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": (
                pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else
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
            "(improvement note already fed back; finalizing)."
        )
    elif aggregate_signal == "FAIL":
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "reviewer_fail_no_capsule",
            "source": "task_acceptance_review",
            "rationale": "A valid acceptance reviewer FAIL had no additional capsule text.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress("Task acceptance review: FAIL (finalizing with a failed review verdict).")
    elif open_obligations:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "open_obligations",
            "source": "task_acceptance_review",
            "rationale": (
                f"Re-review was not a clean PASS ({result.aggregate_signal}); "
                f"{len(open_obligations)} obligation(s) stay open — finalizing honestly."
            ),
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(f"Task acceptance review: {result.aggregate_signal} (no changes suggested).")
    else:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            # Round-9 CRITICAL 1: fall-through AFTER
            # `task_acceptance_is_clean` refused the panel, so it cannot mint
            # `accepted` (reserved for clean acceptance). Reachable: a
            # reviewer claims `solved` with a MISSING criterion and the
            # improvement cap spent — nothing actionable, yet not "accepted";
            # the typed reason names WHY; tier honesty rides `outcome_tier`.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "no_actionable_changes",
            "source": "task_acceptance_review",
            "rationale": (
                f"Re-review was not a clean acceptance ({result.aggregate_signal}) and "
                "suggested no actionable changes; finalizing honestly without acceptance."
            ),
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — not a clean acceptance "
            "and no actionable changes were suggested; finalizing without acceptance."
        )
    return False


def _record_acceptance_infra_failure(ctx: _TaskAcceptanceContext, exc: Exception) -> bool:
    """Finish an eligible mandatory panel as DEGRADED, never as a silent skip."""
    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="degraded")
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status="review_degraded",
        pass_index=ctx.passes_done,
    )
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
    if type(getattr(ctx.tools._ctx, "task_attempt", None)) is int:
        run_record["task_attempt"] = ctx.tools._ctx.task_attempt
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "infra_failure",
        "source": "task_acceptance_review",
        "rationale": "The mandatory host acceptance panel failed before a valid quorum.",
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
    })
    ctx.emit_progress("Task acceptance review: DEGRADED after host review infrastructure failure.")
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
    """The identity ONE paid acceptance panel is claimed under (A-material).

    ``sha256(candidate_hash + the sorted set of nonempty (obligation_id,
    disposition, sha256(reason)) tuples)``. Exactly two things mint a new paid
    panel: a changed candidate answer, or an obligation disposition whose content
    the reviewers have not answered yet. The evidence revision is deliberately NOT
    in here — every cosmetic tool call moves it, which is how one task bought 21
    paid panels; it stays what it always was, stale-packet detection for the
    supersede paths. A disposition with an empty reason contributes nothing.
    Rows are read live from the agent's own ``acceptance_obligations`` (the
    ``task_acceptance_review`` tool stamps ``status="agent_disposed"`` there)."""
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
        str(review_binding.get("candidate_hash") or ""), llm_trace,
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

    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
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
        return False
    from ouroboros.review_evidence import acceptance_packet_budget_chars
    from ouroboros.task_results import resolve_task_lineage

    meta = getattr(tools._ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    lineage = resolve_task_lineage(
        task_id or getattr(tools._ctx, "task_id", ""),
        metadata=meta,
        root_task_id=getattr(tools._ctx, "root_task_id", None),
        parent_task_id=getattr(tools._ctx, "parent_task_id", None),
        delegation_role=getattr(tools._ctx, "delegation_role", None),
        original_task_id=getattr(tools._ctx, "original_task_id", None),
        timeout_retry_from=getattr(tools._ctx, "timeout_retry_from", None),
    )
    eligible, trigger = _loop()._task_acceptance_eligible(
        mode,
        llm_trace,
        bool(getattr(tools._ctx, "is_direct_chat", False)),
        is_root_task=bool(lineage["is_root_task"]),
        is_ephemeral_turn=bool(getattr(tools._ctx, "is_ephemeral_turn", False)),
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
    if not fence_ok:
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
    if not quiescent:
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
    # §19.7.2 item 7: ONE effective profile (remaining improvement passes ->
    # 0 under an armed hurry latch) feeds EVERY acceptance-pacing read below
    # — the improvement_pass_allowed call and the rails display alike.
    budget_profile = effective_budget_profile(
        tools._ctx, task_pacing.resolve_budget_profile(tools._ctx),
    )
    budget_snapshot = task_pacing.build_budget_snapshot(tools._ctx, profile=budget_profile)
    passes_done = int(getattr(tools._ctx, "_task_acceptance_improvement_passes", 0))
    launch_ok, launch_reason = task_pacing.review_launch_allowed(budget_snapshot)
    if not launch_ok:
        return _skip_task_acceptance_for_launch_reason(
            tools._ctx, llm_trace, launch_reason=launch_reason,
            snapshot=budget_snapshot, passes_done=passes_done,
            emit_progress=emit_progress,
        )
    review_ctx = _TaskAcceptanceContext(
        tools=tools,
        content=content,
        task_id=task_id,
        task_type=task_type,
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
                mode == "required" and _loop().get_review_enforcement() == "blocking"
            ), workspace=task_pacing._workspace_delivery(tools._ctx),
        ),
        packet_budget_chars=acceptance_packet_budget_chars(_acceptance_delivery_slots()),
    )
    try:
        from types import SimpleNamespace

        from ouroboros.review_evidence import task_acceptance_evidence_revision
        from ouroboros.review_substrate import build_review_binding

        review_ctx.evidence = _build_host_acceptance_evidence(review_ctx)
        review_ctx.review_binding = build_review_binding(
            candidate=content,
            evidence=review_ctx.evidence,
            fence_token_or_state=_direct_context_fence_state(tools._ctx, _fence_token),
        )
        binding_hash = str(review_ctx.review_binding.get("binding_hash") or "")
        # A-material: what the tree's wallet actually buys. Stamped onto the
        # binding before the free-replay lookup and the dispatch claim both read it.
        paid_identity = bind_acceptance_paid_identity(review_ctx.review_binding, llm_trace)
        seen_bindings, prior_run = _prior_acceptance_run(
            tools._ctx, llm_trace, binding_hash, paid_identity=paid_identity,
        )
        reused_result = None
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
            # Re-run the normal semantic application (gates, outcome axis,
            # obligations, fence) without appending or paying for another panel.
            reused_result = SimpleNamespace(**prior_run)
        elif binding_hash in seen_bindings:
            # A process-local attempt without its authoritative trace is not
            # safe to repeat or silently accept. The infra-degraded path below
            # records the missing authority and closes finalization honestly.
            raise RuntimeError("acceptance binding was attempted but its host run is unavailable")
        else:
            seen_bindings[binding_hash] = None
        llm_trace["review_decision"].update({
            "panel_id": str(review_ctx.review_binding.get("panel_id") or ""),
            "binding_hash": binding_hash,
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
        run_record = (
            prior_run
            if reused_result is not None
            else _record_host_acceptance_run(review_ctx, panel_result)
        )
        if _loop()._task_acceptance_owner_generation_changed(tools._ctx):
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        fresh_quiescent, fresh_subtree_statuses = _loop()._task_acceptance_subtree_snapshot(
            tools._ctx, drive_root, task_id,
        )
        fresh_review_ctx = replace(
            review_ctx,
            subtree_statuses=fresh_subtree_statuses,
            evidence={},
        )
        fresh_evidence_revision = task_acceptance_evidence_revision(
            _build_host_acceptance_evidence(fresh_review_ctx)
        )
        frozen_evidence_revision = str(
            review_ctx.review_binding.get("evidence_revision") or ""
        )
        stale_reason = ""
        if not fresh_quiescent:
            stale_reason = "host_acceptance_subtree_became_non_quiescent"
        elif fresh_evidence_revision != frozen_evidence_revision:
            stale_reason = "host_acceptance_evidence_revision_changed"
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
            reused=reused_result is not None,
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
        return another_round
    except Exception as exc:
        log.debug("Mandatory task acceptance review failed", exc_info=True)
        return _record_acceptance_infra_failure(review_ctx, exc)
    finally:
        publish_acceptance_checkpoint(tools._ctx, llm_trace, task_id=task_id, drive_root=drive_root)
