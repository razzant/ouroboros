"""``plan_task`` — the plan-review engine: ONE review organ pointed at an INTENTION.

Owner-approved redesign (2026-08-15, plan §6/§7): the agent submits a SPEC (goal,
in_scope, non_goals, acceptance_claims, invariants, decisions, deferred,
affected_resources, evidence) plus prose; the host normalizes it (``plan_spec``),
resolves ONE structural fact (``constitutional``), attaches the declared evidence
bounded with every omission named, builds the lean packet (``plan_packet``), fans it
across the configured reviewer rows through the shared review substrate (api_chat
in-process packet OR agent_session retrieving reviewer — the transport is bound by
``review_execution._review_route_executor``, never here), validates each slot's typed
findings, computes the aggregate and records the wave in ``plan_review_state`` v2.

Cycles: ``review_cycles.review_max_cycles()`` bounds PAID panels per task
(``cycles_paid``): a wave is paid iff at least one reviewer slot was PHYSICALLY
DISPATCHED (B2 — a dispatched DEGRADED panel pays like any other; a wave of only
typed $0 skip rows stays unpaid); an identical fingerprint — DEGRADED included —
replays the recorded wave free (no panel, no cycle). Closure
(``plan_spec.closure_after_disposition``): GREEN closes;
REVIEW_REQUIRED closes by disposition at $0; REVISE_PLAN never closes by
disposition — accept ⇒ changed spec (next paid cycle), reject ⇒ rationale rides
into the next delta cycle. Under blocking enforcement an open wave HOLDS
finalization (``owner_hurry.force_plan_decision``); at the cap the typed
``plan_review_cycles_exhausted`` result + event leave the honest exits: owner
unstick or a ``blocked_with_evidence`` terminal. Advisory proceeds open under the
host's loud disclosure. Domain-neutral: a spec with zero paths is first-class.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from hashlib import sha256
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ouroboros.config import (
    adaptive_quorum,
    get_finalization_grace_sec,
    get_llm_transport_read_timeout_sec,
    get_review_enforcement,
    get_task_abs_ceiling_sec,
)
from ouroboros.review_cycles import emit_review_cycles_exhausted, review_max_cycles
from ouroboros.task_results import (
    load_plan_review_state, load_task_result, mark_current_plan_review_unavailable,
    plan_review_wave, current_plan_review_wave, record_plan_review_dispositions,
)
from ouroboros.tools import plan_evidence, plan_spec
from ouroboros.tools.plan_render import _next_step, _quote_control_lines, _render_wave  # noqa: F401 — engine renderers
from ouroboros.tools.plan_review_runtime import (
    PLAN_NO_SNAPSHOT as _PLAN_NO_SNAPSHOT,
    PLAN_REVIEW_MAX_TOKENS as _PLAN_REVIEW_MAX_TOKENS,
    emit_plan_review_advisory_open as _emit_plan_review_advisory_open,
    plan_fanout_inputs as _plan_fanout_inputs,
    plan_in_flight_custody_error as _plan_in_flight_custody_error,
    plan_deadline_skip as _plan_deadline_skip,
    publish_plan_review_projection as _publish_plan_review_projection,
    publish_rendered_wave as _publish_rendered_wave,
    plan_payload_roots as _plan_payload_roots,
    plan_review_slots as _plan_review_slots,
    plan_wave_replay_decision as _plan_wave_replay_decision,
    plan_wave_has_in_flight as _plan_wave_has_in_flight,
    plan_wave_progress_line as _plan_wave_progress_line,
    root_exploration_log as _root_exploration_log,  # noqa: F401 - compatibility seam
    run_plan_review_slots as _run_plan_review_slots,
    synthesize_plan_review_wave as _synthesize_plan_review_wave,
    build_plan_review_packet as _build_packet,
)
from ouroboros.tools.plan_review_artifacts import (
    attach_continuation_restart_delta as _attach_continuation_restart_delta,
    authority_wave as _authority_wave,
    continuation_state as _continuation_state,
    exact_wave as _exact_wave,
    in_flight_resume_inputs as _plan_in_flight_resume_inputs,
    persist_wave as _persist_plan_review_wave_artifact,
    record_exact_wave as _record_exact_wave,
    read_wave as _read_plan_review_wave_artifact,
)
from ouroboros.tools.plan_review_references import (
    _emit_plan_review_reference,
    _record_cycles_exhausted_with_references,
    _record_plan_review_attempt_with_reference,
    _record_raw_plan_request_with_reference,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate
from ouroboros.tools.review_synthesis import (
    PLAN_REVIEW_CONTROL_PREFIX,
)
from ouroboros.utils import truncate_review_artifact, utc_now_iso

log = logging.getLogger(__name__)

# These wrappers are outer settlement bounds, not cognition cutoffs.  Resolve
# them when the tool is built/used so a settings reload cannot leave an old
# transport bound baked into an imported module.
def _plan_review_wrapper_timeout_sec() -> float:
    return float(get_llm_transport_read_timeout_sec() + get_finalization_grace_sec())


def _plan_task_tool_timeout_sec() -> float:
    # ``agent_session`` reviewers inherit the task's existing absolute
    # lifetime, which is deliberately much longer than an API transport read.
    # The outer ToolEntry must cover either route plus one finalization grace
    # window; it is a settlement envelope, never a cognition cutoff.
    return max(
        _plan_review_wrapper_timeout_sec(),
        float(get_task_abs_ceiling_sec()),
    ) + get_finalization_grace_sec()
_TASK_EVIDENCE_RESULT_CHARS = 6_000


@dataclass(frozen=True)
class _PlanRequest:
    goal: str
    plan: str
    spec: Any


_SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": (
        "The object under review — what the work will do, how it will be checked, what "
        "is deliberately deferred. Ids are minted by the host (claim_N, invariant_N, "
        "decision_N, deferred_N) and are the only targets a blocking finding may name."
    ),
    "properties": {
        "in_scope": {"type": "array", "items": {"type": "string"}},
        "non_goals": {"type": "array", "items": {"type": "string"}},
        "acceptance_claims": {
            # object form feeds acceptance's support_expected; a bare string is fine too
            "type": "array",
            "items": {"anyOf": [
                {"type": "string"},
                {"type": "object", "properties": {
                    "claim": {"type": "string"}, "surface": {"type": "string"},
                    "support": {"type": "string"},
                    "priority": {"type": "string", "enum": ["must", "should"]},
                }, "required": ["claim"], "additionalProperties": False},
            ]},
            "description": (
                "Concrete, checkable statements of what 'done' means — strings or "
                "{claim, surface, support, priority}. The CLOSED plan's claims bind task "
                "acceptance (host-minted ids claim_1..N in list order — link "
                "verify_and_record receipts via criterion_id)."
            ),
        },
        "invariants": {
            "type": "array", "items": {"type": "string"},
            "description": "Constraints that must hold: budget, deadline, safety, irreversibility, external commitments.",
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "choice": {"type": "string"},
                    "rejected": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
                "required": ["choice"],
            },
            "description": "Load-bearing decisions that would be expensive to reverse, each with rejected alternatives and why.",
        },
        "deferred": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"what": {"type": "string"}, "why_safe_to_defer": {"type": "string"}},
                "required": ["what"],
            },
            "description": "Deliberately deferred until the work is underway; ids deferred_1..N — a blocking finding may name any spec id it breaks: goal, claim_N, invariant_N, decision_N or deferred_N itself.",
        },
        "affected_resources": {
            "type": "array", "items": {"type": "string"},
            "description": (
                "What the work will CHANGE (paths, systems, services). Paths resolving under "
                "the Ouroboros system repository make the plan constitutional (BIBLE goes to reviewers)."
            ),
        },
        "evidence": {
            "type": "array", "items": {"type": "string"},
            "description": (
                "What reviewers should LOOK AT: file paths, task:<id> of a prior task, URLs. "
                "The host attaches files bounded (never fetching URLs) and names every omission. "
                "An EXISTING path here that resolves under the Ouroboros system repository also "
                "makes the plan constitutional (BIBLE + ARCHITECTURE go to reviewers); a path that "
                "does not exist does not, and the skip is disclosed."
            ),
        },
    },
}

_DISPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "description": (
        "Disposition mode only (send ONLY this field): answer the findings of the wave "
        "named by review_fingerprint. note/need_evidence findings close at $0; a blocking "
        "finding you accept needs a changed spec (new call), one you reject rides with your "
        "rationale into the next paid cycle. Never closes REVISE_PLAN."
    ),
    "properties": {
        "review_fingerprint": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "finding_id": {"type": "string"},
                    "decision": {"type": "string", "enum": list(plan_spec.DISPOSITION_DECISIONS)},
                    "rationale": {"type": "string"},
                },
                "required": ["finding_id", "decision", "rationale"],
            },
        },
    },
    "required": ["review_fingerprint", "items"],
}


def get_tools():
    return [
        ToolEntry(
            name="plan_task",
            schema={
                "name": "plan_task",
                "description": (
                    "Multi-model design review of an INTENTION before the work starts — code, "
                    "research, a deliverable, a computer-use flow, or an action in the world. "
                    "Submit goal + spec (what/how-checked/deferred) + plan prose; independent "
                    "reviewers return typed findings against the spec (blocking findings must name "
                    "the spec element they break); the host aggregates: GREEN closes; "
                    "REVIEW_REQUIRED closes by your review_disposition at no cost; REVISE_PLAN needs "
                    "a changed spec (next paid cycle) or a reject-with-rationale judged in the next "
                    "cycle. Cycles are bounded by the owner's Max review cycles; an unchanged "
                    "envelope replays the recorded result for free (a locator a reviewer asked for "
                    "with need_evidence is attached by the host next time and makes the envelope "
                    "new). Under blocking enforcement an "
                    "open review holds finalization; under advisory you may proceed with the "
                    "review open and the host discloses it. Declare evidence reviewers need; "
                    "declare affected_resources so a self-modification gets the constitutional pack."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Why — the outcome the work serves."},
                        "plan": {"type": "string", "description": "Accompanying prose: how you intend to do it (context for reviewers; the spec is what is judged)."},
                        "spec": _SPEC_SCHEMA,
                        "review_disposition": _DISPOSITION_SCHEMA,
                    },
                    # Two exclusive modes: goal+plan+spec (review) or review_disposition alone.
                    "required": [],
                },
            },
            handler=_handle_plan_task,
            timeout_sec=_plan_task_tool_timeout_sec(),
        )
    ]


# --------------------------------------------------------------------------- handler


def _vacuous_disposition(value: object) -> bool:
    """A schema-shaped but empty disposition (models fill optional objects with defaults)."""
    if not isinstance(value, dict) or set(value) - {"review_fingerprint", "items"}:
        return False
    return not str(value.get("review_fingerprint") or "").strip() and not value.get("items")


def _handle_plan_task(ctx: ToolContext, **params) -> str:
    raw_disposition = params.get("review_disposition")
    envelope_fields = sorted(set(params) - {"review_disposition"})
    if raw_disposition is not None and not _vacuous_disposition(raw_disposition):
        if envelope_fields:
            return (
                "ERROR: PLAN_REVIEW_DISPOSITION_MIXED_ENVELOPE: disposition mode accepts "
                "review_disposition only; a changed plan needs a new review-mode call "
                "without review_disposition. No plan attempt was recorded."
            )
        if not isinstance(raw_disposition, dict):
            return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: review_disposition must be an object"
        return _apply_disposition(ctx, raw_disposition)
    if "review_disposition" in params and not envelope_fields:
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_EMPTY: submit goal, plan and spec for review "
            "mode, or a complete review_disposition as the only field. No plan attempt was recorded."
        )
    request = _PlanRequest(
        goal=str(params.get("goal") or ""), plan=str(params.get("plan") or ""), spec=params.get("spec"),
    )
    # The ToolEntry envelope is the outer settlement bound. The substrate
    # owns each review slot's logical window and late-result custody; nesting a
    # second asyncio.wait_for here only cancels the coroutine while its
    # executor worker keeps running, then asyncio.run waits for that worker
    # during shutdown and defeats the apparent timeout. Let the existing
    # tool-timeout callback own the late settlement instead.
    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                # copy_context: the registry's tool-result sidecar is a ContextVar,
                # and the published native plan result must reach the dispatching
                # thread's slot (D02) — a bare pool thread would publish into the void.
                return pool.submit(
                    contextvars.copy_context().run,
                    asyncio.run,
                    _run_plan_review_async(ctx, request),
                ).result()
        except RuntimeError:
            return asyncio.run(_run_plan_review_async(ctx, request))
    except Exception as e:
        log.error("plan_task failed: %s", e, exc_info=True)
        return _plan_unavailable(ctx, f"ERROR: Plan review failed: {e}", "review_failed")


def _plan_unavailable(ctx: ToolContext, message: str, reason: str) -> str:
    """Persist a retryable availability outcome (the current fingerprint stays open-unavailable)."""
    try:
        root, task_id = _planning_state_location(ctx)
        state = mark_current_plan_review_unavailable(root, task_id, reason=reason)
        _emit_plan_review_reference(ctx, task_id, state, state_root=root)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"{message}\nERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    return message


def _planning_state_location(ctx: ToolContext) -> tuple[pathlib.Path, str]:
    root = pathlib.Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        raise ValueError("PLAN_REVIEW_TASK_ID_REQUIRED: durable review state must belong to a real task")
    return root, task_id


# ------------------------------------------------------------------- inputs / packet


def _evidence_deny_paths(ctx: ToolContext) -> list[str]:
    """Paths evidence may never attach, whatever root the caller declares (C-06): the runtime
    data plane and the live settings file are a boundary, not a heuristic — an operator subject
    root one level above them would otherwise make owner credentials reviewable."""
    from ouroboros import config as _config

    out: list[str] = []
    for value in (getattr(_config, "SETTINGS_PATH", ""), getattr(_config, "DATA_DIR", "")):
        text = str(value or "").strip()
        if text:
            out.append(text)
    try:
        from ouroboros.tool_access import canonical_data_root

        drive = canonical_data_root(ctx)
        if drive:
            out.append(str(drive))
    except Exception:
        pass
    return out


def _plan_fingerprint(goal: str, plan: str, spec: dict, manifest_hash: str, constitutional: bool) -> str:
    """Identity of one review request (F4): goal, prose, canonical spec, evidence identity,
    the constitutional fact — never the exploration log (it changes no obligation)."""
    payload = {"goal": goal, "plan": plan, "spec": spec, "evidence_manifest_hash": manifest_hash,
               "constitutional": bool(constitutional)}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _task_evidence_reader(root: pathlib.Path) -> Callable[[str], Optional[str]]:
    """``task:<id>`` locators → a bounded JSON summary of that task's durable result."""
    def _read(task_id: str) -> Optional[str]:
        try:
            record = load_task_result(root, task_id)
        except Exception:
            return None
        if not isinstance(record, dict):
            return None
        return json.dumps({
            "task_id": task_id,
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "ts": record.get("ts"),
            "result": truncate_review_artifact(str(record.get("result") or ""), limit=_TASK_EVIDENCE_RESULT_CHARS),
        }, ensure_ascii=False, indent=2, default=str)
    return _read


# W3 host attachment is bounded like the agent's own evidence list (MAX_LIST_ITEMS honoured
# locators per task); what the cap drops is a NAMED `reviewer_request_cap` omission, never silent.
_REVIEWER_REQUEST_CAP = plan_spec.MAX_LIST_ITEMS


def _reviewer_requested_locators(ctx: ToolContext, state_root: pathlib.Path) -> tuple[list[str], list[str]]:
    """``(honoured, dropped)`` `need_evidence` locators from this task's earlier cycles
    (`need_evidence_seen`, kept sorted): the cap keeps the lexicographically first ones,
    deterministically (disclosed residual: past the cap a later, earlier-sorting request can
    displace an honoured one — then a named `reviewer_request_cap` omission, never silent).
    Raises ``ValueError`` when the durable state cannot be read: the wave that would pay for
    the packet must not run without the reviewers' recorded requests (fail-closed, P1)."""
    _root, task_id = _planning_state_location(ctx)
    try:
        state = load_plan_review_state(state_root, task_id)
    except (OSError, TimeoutError) as exc:
        raise ValueError(f"PLAN_REVIEW_STATE_INVALID: {exc}") from exc
    seen: list[str] = []
    dropped: list[str] = []
    for raw in state.get("need_evidence_seen") or []:
        loc = str(raw or "").strip()
        if not loc or loc in seen or loc in dropped:
            continue
        if len(loc) > plan_spec.MAX_ITEM_CHARS or len(seen) >= _REVIEWER_REQUEST_CAP:
            dropped.append(loc)
        else:
            seen.append(loc)
    return seen, dropped


def _prepare_plan_inputs(ctx: ToolContext, request: "_PlanRequest", state_root: pathlib.Path) -> dict:
    """The ONE preamble the paid path and the dry-run seam share: normalize the spec (with the
    envelope's goal injected), resolve the subject roots, derive `constitutional`, attach the
    declared evidence, compose the fingerprint. Returns ``{"error": ...}`` on refusal — a second
    assembly path had already diverged from this one (I-11)."""
    raw_spec = dict(request.spec) if isinstance(request.spec, dict) else request.spec
    if isinstance(raw_spec, dict):
        raw_spec["goal"] = request.goal.strip() or raw_spec.get("goal")
    spec, errors = plan_spec.normalize_spec(raw_spec if isinstance(raw_spec, dict) else None)
    if not request.plan.strip():
        errors = ["plan: required non-empty prose", *errors]
    if errors:
        return {"error": "ERROR: PLAN_SPEC_INVALID: " + "; ".join(errors) + ". No reviewer was called."}
    from ouroboros.review_substrate import review_repo_dirs_for
    try:
        system_root, active_root = review_repo_dirs_for(ctx)
    except ValueError as exc:
        return {"error": f"ERROR: PLAN_SUBJECT_ROOT_INVALID: {exc}"}
    locators = list(spec["affected_resources"]) + list(spec["evidence"])
    constitutional, constitutional_note = plan_spec.resolve_constitutional(
        active_root=active_root, system_repo_root=system_root,
        affected_resources=spec["affected_resources"], evidence=spec["evidence"],
        payload_roots=_plan_payload_roots(ctx, locators),
    )
    reminder = (
        "REMINDER: affected_resources is empty; if this work will change Ouroboros's own body, "
        "declare those paths so reviewers receive the constitutional pack (BIBLE)."
        if active_root == system_root and not spec["affected_resources"] else ""
    )
    declared_evidence = list(spec["evidence"])  # W3: earlier-cycle need_evidence is HOST-attached
    try:
        reviewer_requested, request_dropped = _reviewer_requested_locators(ctx, state_root)
    except ValueError as exc:
        return {"error": f"ERROR: {exc}"}
    host_locators = [loc for loc in reviewer_requested if loc not in declared_evidence]
    # B-08/C-06: allowed roots are the active workspace and the system repo only; the runtime
    # data plane is denied outright (the sensitive-name policy is a residual, not a boundary).
    # A continuation range is the evidence the prior panel explicitly said it
    # still needed. Resolve those host-carried selectors first so a repeated
    # 120K declared pack cannot starve the decisive line/tail with the same
    # ``budget_exhausted`` state that prompted the continuation.
    manifest = plan_evidence.resolve_evidence(
        host_locators + declared_evidence, active_root=active_root,
        allowed_roots=[active_root, system_root],
        resolve_task=_task_evidence_reader(state_root), deny_paths=_evidence_deny_paths(ctx),
    )
    manifest["declared"] = declared_evidence  # the AGENT's list; requests below (tagged+hashed)
    if reviewer_requested:
        manifest["reviewer_requested"] = list(reviewer_requested)
    if request_dropped:  # still reviewer requests (provenance), just not honoured by the cap
        manifest["reviewer_requested_dropped"] = list(request_dropped)
        manifest.setdefault("omissions", []).extend(
            {"locator": loc, "reason": "reviewer_request_cap"} for loc in request_dropped)
    manifest_hash = plan_evidence.evidence_manifest_hash(manifest)
    fingerprint = _plan_fingerprint(spec["goal"], request.plan, spec, manifest_hash, constitutional)
    return {
        "spec": spec, "system_root": system_root, "active_root": active_root,
        "constitutional": constitutional, "constitutional_note": constitutional_note,
        "manifest": manifest, "manifest_hash": manifest_hash, "reminder": reminder,
        "fingerprint": fingerprint,
    }


# --------------------------------------------------------------------------- review


async def _run_plan_review_async(ctx: ToolContext, request: _PlanRequest) -> str:
    try:
        state_root, task_id = _planning_state_location(ctx)
    except ValueError as exc:
        return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
    prepared = _prepare_plan_inputs(ctx, request, state_root)
    if prepared.get("error"):
        if "PLAN_SPEC_INVALID" in prepared["error"]:
            try:
                _record_raw_plan_request_with_reference(
                    ctx, state_root, task_id, {"goal": request.goal, "plan": request.plan, "spec": request.spec},
                    reason="plan_input_invalid")
            except (OSError, TimeoutError, ValueError) as exc:
                return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        return prepared["error"]
    spec, manifest = prepared["spec"], prepared["manifest"]
    system_root, active_root = prepared["system_root"], prepared["active_root"]
    constitutional = prepared["constitutional"]
    constitutional_note = prepared["constitutional_note"]
    manifest_hash = prepared["manifest_hash"]
    reminder = prepared["reminder"]
    fingerprint = prepared["fingerprint"]
    try:
        state = load_plan_review_state(state_root, task_id)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
    enforcement = get_review_enforcement()
    cap = review_max_cycles()
    cycles_paid = int(state.get("cycles_paid") or 0)
    # C-01: every envelope supersedes prior authority BEFORE any cap/rail exit.
    try:
        _record_plan_review_attempt_with_reference(ctx, state_root, task_id, fingerprint=fingerprint)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    previous_override: Optional[dict] = None
    replay_snapshot: Any = _PLAN_NO_SNAPSHOT
    resume_in_flight = False
    existing = plan_review_wave(state, fingerprint)
    if existing is not None and not isinstance(existing.get("spec"), dict):
        existing = None  # C-09: a COMPACTED row (no frozen spec) is never authority
    if existing is not None:
        try:
            existing = _authority_wave(state_root, task_id, existing)
        except (OSError, ValueError, json.JSONDecodeError):
            return _plan_unavailable(
                ctx,
                "ERROR: Exact plan-review authority is unreadable; replay and disposition are refused.",
                "plan_review_exact_artifact_unavailable",
            )
        resume_in_flight = _plan_wave_has_in_flight(existing)
        # Identical requests replay free unless authority lapsed; fully rejected
        # blocking findings are the one earned-delta exception (4e133c8a).
        earned_delta = (
            str(existing.get("aggregate")) in {"REVISE_PLAN", "REVIEW_REQUIRED"}
            and not existing.get("closed")
            # D1: only VALID rejections earn the panel — raw items are persisted for
            # disclosure, but an invalid or contradictory one must not buy a cycle.
            and plan_spec.blocking_fully_rejected(
                existing.get("findings"), existing.get("dispositions"))
        )
        if earned_delta and not resume_in_flight:
            # the rejected wave IS the previous cycle for the delta panel
            previous_override = existing
        elif bool(existing.get("closed")) and not resume_in_flight:
            # A closed verdict is earned authority; later roster changes govern
            # future panels and do not retroactively void it (accepted 3a).
            return _publish_rendered_wave(ctx, existing, cap=cap, cycles_paid=cycles_paid,
                                          enforcement=enforcement, cached=True, reminder=reminder)
        elif not resume_in_flight:  # stale ⇒ identical envelope re-dispatches fresh
            stale, replay_snapshot = _plan_wave_replay_decision(_plan_review_slots, existing)
            if not stale:
                if enforcement == "advisory":
                    # Still-OPEN wave: re-invoke the emitter so a durable append that FAILED
                    # at record time retries on replay (memo only on success ⇒ landed dedups).
                    _emit_plan_review_advisory_open(ctx, state_root, task_id=task_id,
                                                    wave=existing, cycles_paid=cycles_paid, cap=cap)
                return _publish_rendered_wave(ctx, existing, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement, cached=True, reminder=reminder)
    deadline_skip = _plan_deadline_skip(ctx)
    # An existing paid wave with a live physical reviewer is a custody
    # reconciliation, not a new planning dispatch.  Let it pass the owner
    # deadline rail so the exact frozen cycle can settle; fresh envelopes still
    # take the ordinary no-new-work deadline path below.
    if deadline_skip and not resume_in_flight:
        try:
            _record_plan_review_attempt_with_reference(
                ctx, state_root, task_id, fingerprint=fingerprint, status="rail_degraded",
                reason="plan_task_deadline")
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        return _plan_deadline_skip(ctx, emit=True) or deadline_skip
    if cap is not None and cycles_paid >= cap and not resume_in_flight:
        return _cycles_exhausted(ctx, state, state_root, task_id, cap=cap, cycles_paid=cycles_paid,
                                 enforcement=enforcement, reminder=reminder,
                                 request_fingerprint=fingerprint)
    # #116: a malformed structured reviewer-slot config must refuse loudly here
    # instead of running the panel on the silently projected default models.
    from ouroboros.reviewer_slot_config import reviewer_slot_config_error

    if err := reviewer_slot_config_error():
        return _plan_unavailable(
            ctx, f"ERROR: Invalid reviewer-slot configuration blocks plan review — {err}. "
            "Fix Review lanes on the Agents tab in Settings.", "reviewer_slot_config_invalid")
    slots = _plan_review_slots()
    if not slots:
        return _plan_unavailable(
            ctx, "ERROR: No review models configured. Configure Review lanes "
            "(OUROBOROS_REVIEWER_SLOTS) on the Agents tab in Settings.",
            "review_models_unconfigured")
    configured_slots = list(slots)
    resume = _plan_in_flight_resume_inputs(
        existing, state, state_root, task_id, configured_slots,
    ) if resume_in_flight else {}
    if resume.get("error"):
        return _plan_unavailable(
            ctx,
            "ERROR: PLAN_REVIEW_CUSTODY_INVALID: " + str(resume["error"]),
            "plan_review_custody_invalid",
        )
    previous = resume.get("previous") if resume_in_flight else (
        previous_override if previous_override is not None else _last_paid_wave(state)
    )
    if previous is not None:
        try:
            previous = _authority_wave(state_root, task_id, previous)
        except (OSError, ValueError, json.JSONDecodeError):
            return _plan_unavailable(
                ctx,
                "ERROR: Prior exact plan-review authority is unreadable; a delta review is refused.",
                "plan_review_exact_artifact_unavailable",
            )
    cycle_index = int(resume.get("cycle_index") or cycles_paid + 1)
    retry_key = str(resume.get("retry_key") or f"plan_review:{fingerprint}:{cycle_index}")
    system_prompt, user_content, session_task = _build_packet(
        ctx, spec=spec, request=request, manifest=manifest, constitutional=constitutional,
        system_root=system_root, active_root=active_root, cycle_index=cycle_index,
        enforcement=enforcement, previous=previous,
    )
    slots, slot_messages, session_threads, continuation_restarted = _continuation_state(
        state_root, task_id, previous, slots, manifest, user_content=user_content,
    )
    quorum = adaptive_quorum(len(slots))
    fanout = _plan_fanout_inputs(
        slots, resume=resume if resume_in_flight else None, replay_snapshot=replay_snapshot,
        prompt_chars=len(system_prompt) + len(user_content), quorum=quorum,
    )
    if fanout["error"]:
        if not resume_in_flight:
            return _plan_unavailable(ctx, fanout["error"], "review_context_unavailable")
        return _publish_rendered_wave(
            ctx, existing, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement,
            cached=True, reminder="\n".join(x for x in (reminder, fanout["error"]) if x),
        )
    callable_slots = fanout["callable_slots"]
    health_skip_rows, oversize_rows = fanout["health_skip_rows"], fanout["oversize_rows"]
    health_evidence = fanout["health_evidence"]
    if resume_in_flight and callable_slots:
        pending_note = _plan_in_flight_custody_error(
            retry_key=retry_key, task_id=str(task_id), active_root=active_root,
            callable_slots=list(callable_slots), ctx=ctx,
        )
        if pending_note:
            return _publish_rendered_wave(
                ctx, existing, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement,
                cached=True, reminder="\n".join(x for x in (reminder, pending_note) if x),
            )
    admission = None if resume_in_flight else review_wave_budget_gate(
        ctx, surface="plan_review", models=[str(s.model) for s in callable_slots],
        prompt_chars=len(system_prompt) + len(user_content), max_completion_tokens=_PLAN_REVIEW_MAX_TOKENS,
    )
    if admission is not None:
        fence, remedy = review_wave_binding_fence(admission)
        return _plan_unavailable(
            ctx,
            "⚠️ PLAN_REVIEW_SKIPPED_BUDGET: the reviewer wave was declined before dispatch — "
            f"estimated cost ~${admission.get('estimated_wave_usd')} exceeds the remaining budget "
            f"${admission.get('remaining_usd')} ({fence}). No reviewer was called. Shrink the evidence, "
            f"split the plan, or {remedy}.",
            "review_budget_unavailable")
    ctx.emit_progress_fn(
        f"📐 plan_task: cycle {cycle_index}{'' if cap is None else f'/{cap}'} — running "
        f"{len(callable_slots)} of {len(slots)} reviewer slot(s)"
        + (f", {len(health_skip_rows)} health-skipped at $0" if health_skip_rows else "")
        + f" ({enforcement}; constitutional={constitutional})…"
    )
    rows = await _run_plan_review_slots(
        ctx, callable_slots, system_prompt=system_prompt, user_content=user_content,
        session_task=session_task, session_root=str(active_root),
        output_contract=plan_spec.PLAN_FINDINGS_ARRAY_CONTRACT,
        slot_messages=slot_messages,
        session_threads=session_threads,
        retry_key=retry_key,
    ) if callable_slots else []
    # excluded slots stay configured rows: they count in the quorum denominator
    rows = list(rows) + oversize_rows + health_skip_rows
    _attach_continuation_restart_delta(rows, continuation_restarted)
    wave, seen_after, agg = _synthesize_plan_review_wave(
        rows, state=state, spec=spec, request_plan=request.plan, fingerprint=fingerprint,
        previous=previous, manifest=manifest, manifest_hash=manifest_hash,
        constitutional=constitutional, constitutional_note=constitutional_note,
        cycle_index=cycle_index, retry_key=retry_key, enforcement=enforcement, cap=cap,
        quorum=quorum, configured_slots=configured_slots,
        health_evidence=health_evidence,
    )
    aggregate = str(wave["aggregate"])
    exact_wave = _exact_wave(
        wave, plan_prose=request.plan, manifest=manifest, slots=configured_slots, rows=rows,
        system_prompt=system_prompt, user_content=user_content,
        session_task=session_task, slot_messages=slot_messages,
    )
    try:
        stored = _record_exact_wave(
            state_root, task_id, wave, exact_wave,
            need_evidence_seen=sorted(seen_after),
            page_size=plan_spec.MAX_FINDINGS_PER_SLOT,
        )
        if stored.get("paid") and not wave.get("paid"):
            # D2/B2: the durable authority stayed the paid predecessor (this attempt
            # dispatched nothing); the tool answer still describes the attempt that ran.
            stored = wave
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    try:
        paid_now = int(load_plan_review_state(state_root, task_id).get("cycles_paid") or 0)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
    _emit_plan_review_reference(ctx, task_id, state_root=state_root)
    if enforcement == "advisory" and not stored.get("closed"):
        # B2: loud at the moment — ONE typed owner-visible event per recorded open wave.
        _emit_plan_review_advisory_open(ctx, state_root, task_id=task_id, wave=stored,
                                        cycles_paid=paid_now, cap=cap)
    if (
        cap is not None and paid_now >= cap and not stored.get("closed")
        and not _plan_wave_has_in_flight(wave)
    ):
        # Scope-gate finding (39c3a195): when the FINAL permitted cycle ends open, the typed
        # cap state must land NOW — not wait for a second envelope the agent may never send —
        # or the blocking gate holds a task that can never buy another panel (D27).
        try:
            stored = _record_cycles_exhausted_with_references(
                ctx, state_root, task_id, wave_fingerprint=fingerprint,
                attempt_fingerprint=fingerprint, cycles_paid=paid_now, cap=cap,
            ) or stored
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        emit_review_cycles_exhausted(
            getattr(ctx, "event_queue", None), state_root, surface="plan_review",
            task_id=task_id, cycles_paid=paid_now, cap=cap, enforcement=enforcement,
            fingerprint=fingerprint)
    ctx.emit_progress_fn(_plan_wave_progress_line(
        aggregate, agg["counts"], cycles_paid=paid_now, cap=cap))
    return _publish_rendered_wave(ctx, stored, cap=cap, cycles_paid=paid_now, enforcement=enforcement, reminder=reminder)


def _last_paid_wave(state: dict) -> Optional[dict]:
    for wave in reversed(state.get("waves") or []):
        if wave.get("paid") and not wave.get("compact"):
            return wave
    return None


def build_plan_review_packet_for_dry_run(ctx: ToolContext, request: "_PlanRequest") -> dict:
    """Assemble the packet SHAPE of a fresh cycle (cycle_index=1, no prior-cycle section) with the
    task's recorded reviewer requests attached (W3), WITHOUT dispatching or recording anything.

    The operator entry point runs outside a task, where the shared budget gate has no usage
    scope and therefore no cost ceiling; this is how the operator sees the bill before paying
    it. Read-only, and it shares `_prepare_plan_inputs` with the paid path (I-11)."""
    state_root, _task_id = _planning_state_location(ctx)
    prepared = _prepare_plan_inputs(ctx, request, state_root)
    if prepared.get("error"):
        raise ValueError(prepared["error"])
    system_prompt, user_content, _session_task = _build_packet(
        ctx, spec=prepared["spec"], request=request, manifest=prepared["manifest"],
        constitutional=prepared["constitutional"], system_root=prepared["system_root"],
        active_root=prepared["active_root"], cycle_index=1,
        enforcement=get_review_enforcement(), previous=None,
    )
    return {
        "constitutional": prepared["constitutional"],
        "constitutional_note": prepared["constitutional_note"],
        "manifest": prepared["manifest"], "system_prompt": system_prompt,
        "user_content": user_content, "fingerprint": prepared["fingerprint"],
    }


def _cycles_exhausted(
    ctx: ToolContext, state: dict, state_root: pathlib.Path, task_id: str, *,
    cap: int, cycles_paid: int, enforcement: str, reminder: str,
    request_fingerprint: str = "",
) -> str:
    """The typed cap result (D10/D27): no panel, the current wave stays open, the typed
    event fires; blocking exits are owner unstick or a blocked_with_evidence terminal."""
    current = current_plan_review_wave(state)
    # C-01: a CLOSED wave recorded for a DIFFERENT envelope is history, never this
    # request's answer — rendering it would hand a changed, unreviewed spec the old
    # GREEN. An OPEN wave is the live obligation and still carries the hold, so it is
    # marked and rendered with the cap head above it.
    stale_closed = bool(
        current
        and current.get("closed")
        and str(current.get("request_fingerprint") or "") != str(request_fingerprint or "")
    )
    if stale_closed:
        current = None
    if current is None and request_fingerprint:
            # A NEW envelope at a spent cap has no wave of its own; the live obligation is
        # the last open wave, and the CURRENT attempt records the cap so the gate
        # releases finalization honestly (D27).
        current = next(
            (w for w in reversed(list(state.get("waves") or [])) if isinstance(w, dict) and not w.get("closed")),
            None,
        )
    fingerprint = str((current or {}).get("request_fingerprint") or "")
    if request_fingerprint:
        try:
            current = _record_cycles_exhausted_with_references(
                ctx, state_root, task_id,
                wave_fingerprint=fingerprint or request_fingerprint,
                attempt_fingerprint=request_fingerprint,
                cycles_paid=cycles_paid, cap=cap,
            ) or current
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    emit_review_cycles_exhausted(
        getattr(ctx, "event_queue", None), state_root, surface="plan_review", task_id=task_id,
        cycles_paid=cycles_paid, cap=cap, enforcement=enforcement, fingerprint=fingerprint,
    )
    ctx.emit_progress_fn(
        f"📐 plan_task: PLAN_REVIEW_CYCLES_EXHAUSTED — {cycles_paid}/{cap} paid cycles spent ({enforcement})."
    )
    head = (
        f"⚠️ PLAN_REVIEW_CYCLES_EXHAUSTED: {cycles_paid} of {cap} paid plan-review cycles are spent "
        "for this task; no reviewer was called and no cycle was consumed. "
    )
    if enforcement == "blocking":
        head += (
            "Blocking enforcement: the plan review stays OPEN, so implementation stays held — but "
            "finalization is RELEASED so the task can end honestly instead of waiting for a panel it "
            "can no longer buy (owner decision D27). Your exits are an owner unstick (Swarm/hurry), a "
            "revised spec once the owner raises OUROBOROS_REVIEW_MAX_CYCLES, or finalizing now with "
            "outcome_tier=blocked_with_evidence. Do not start the work under an open blocking review."
        )
    else:
        head += (
            "Advisory enforcement: you may proceed with the review open; the host records and "
            "discloses it loudly (typed event review_cycles_exhausted)."
        )
    if current:
        return _publish_rendered_wave(
            ctx, current, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement,
            cached=True, reminder=reminder, head=head + "\n\n",
        )
    # No live wave for THIS envelope (a fresh spec submitted at a spent cap): the
    # host still owns exactly one control line, and it can never be a closed one.
    text = (
        head + "\n\n" + (f"{reminder}\n\n" if reminder else "")
        + PLAN_REVIEW_CONTROL_PREFIX
        + json.dumps({"outcome": "REVISE_PLAN", "closed": False}, ensure_ascii=False)
    )
    return _publish_plan_review_projection(
        ctx, {"aggregate_signal": "REVISE_PLAN", "closed": False}, text)


# ---------------------------------------------------------------------- disposition


def _apply_disposition(ctx: ToolContext, disposition: dict) -> str:
    unknown = sorted(str(k) for k in disposition if k not in {"review_fingerprint", "items"})
    if unknown:
        return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: unknown fields: " + ", ".join(unknown)
    fingerprint = str(disposition.get("review_fingerprint") or "").strip()
    if not fingerprint:
        return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: review_fingerprint is required"
    try:
        root, task_id = _planning_state_location(ctx)
        state = load_plan_review_state(root, task_id)
    except (OSError, TimeoutError, ValueError) as exc:
        return "ERROR: PLAN_REVIEW_STATE_INVALID: " + str(exc)
    enforcement = get_review_enforcement()
    cap = review_max_cycles()
    cycles_paid = int(state.get("cycles_paid") or 0)
    wave = plan_review_wave(state, fingerprint)
    if wave is None or wave.get("compact"):
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_UNBINDABLE: no recorded plan-review wave holds "
            f"fingerprint {fingerprint} (compact history is not dispositionable). No plan attempt was recorded."
        )
    try:
        wave = _authority_wave(root, task_id, wave) or wave
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "ERROR: PLAN_REVIEW_DISPOSITION_UNBINDABLE: exact wave is unreadable: " + str(exc)
    # I-01: a disposition closes ONLY the CURRENT attempt's wave (never a superseded one).
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    current_fp = str(attempt.get("fingerprint") or "")
    if current_fp and current_fp != fingerprint:
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_STALE: a disposition can close only the CURRENT "
            f"plan-review wave; a newer attempt supersedes it (current={current_fp}, "
            f"claimed={fingerprint}). Re-call plan_task with the spec you want reviewed. "
            "No plan attempt was recorded."
        )
    if wave.get("closed"):
        return _publish_rendered_wave(ctx, wave, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement,
                                      cached=True,
                                      notes=["already_closed: this wave is closed; the disposition is not re-applied"])
    raw_items = disposition.get("items")
    if not isinstance(raw_items, list):
        return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: items must be an array"
    if len(raw_items) > 2 * len(wave.get("findings") or []) + 8:  # bounded like the findings they answer
        return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: more items than findings could need"
    items: List[dict] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            return f"ERROR: PLAN_REVIEW_DISPOSITION_INVALID: items[{index}] must be an object"
        items.append({
            "finding_id": str(item.get("finding_id") or "").strip()[:plan_spec.MAX_ID_CHARS * 2],
            "decision": str(item.get("decision") or "").strip().lower()[:40],  # enum-like, bounded
            "rationale": plan_spec.bounded_text(item.get("rationale"), plan_spec.MAX_FINDING_TEXT_CHARS),
        })
    known = {str(f.get("finding_id") or "") for f in wave.get("findings") or []}
    unknown_ids = sorted({i["finding_id"] for i in items if i["finding_id"] not in known})
    if unknown_ids:
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: unknown finding ids " + ", ".join(unknown_ids)
            + "; valid ids: " + ", ".join(sorted(known))
        )
    closure = plan_spec.closure_after_disposition(
        str(wave.get("aggregate") or ""), wave.get("findings") or [], items, enforcement,
    )
    disposition_recorded_at = utc_now_iso()
    try:
        prior_ref = wave.get("wave_artifact") if isinstance(wave.get("wave_artifact"), dict) else {}
        closure_notes = [*closure["notes"], *([] if prior_ref else [
            "exact_artifact_absent: v2 wave had no exact wave_artifact reference",
        ])]
        exact = _read_plan_review_wave_artifact(root, task_id, prior_ref) if prior_ref else dict(wave)
        exact.update({
            "dispositions": list(items), "closed": bool(closure["closed"]),
            "closure_notes": closure_notes,
            "disposition_recorded_at": disposition_recorded_at,
            "supersedes_wave_artifact": prior_ref,
        })
        disposition_ref = _persist_plan_review_wave_artifact(root, task_id, exact)
        stored = record_plan_review_dispositions(
            root, task_id, fingerprint=fingerprint, dispositions=items,
            closed=bool(closure["closed"]), closure_notes=closure_notes,
            wave_artifact=disposition_ref, recorded_at=disposition_recorded_at,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(exc)
    _emit_plan_review_reference(ctx, task_id, state_root=root)
    ctx.emit_progress_fn(
        f"📐 plan_task: disposition recorded — {'closed' if closure['closed'] else 'still open'} "
        f"({len(closure['open_ids'])} open finding id(s); no reviewer call, no cycle)."
    )
    return _publish_rendered_wave(ctx, stored, cap=cap, cycles_paid=cycles_paid,
                                  enforcement=enforcement, notes=list(closure["notes"]))


# ------------------------------------------------------------------------ rendering
