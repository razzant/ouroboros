"""Reviewer-slot execution and envelope rails for ``plan_task``.

The plan-review engine (``tools/plan_review.py``) owns state, packet and verdict;
this module owns the thin runtime seams around it: the deadline rail, the raw
attempt supersession, the configured reviewer rows as ``ReviewSlot`` objects
(both delivery kinds — the D15 api_chat pin is gone), one substrate call, and the
structural helpers (payload exemption, error text) that need a ``ToolContext``.
Transport is NEVER chosen here: every slot carries its route and
``review_execution._review_route_executor`` binds it (plan §8.2).
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
import inspect
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.llm import LLMClient
from ouroboros.review_execution_projection import review_executions_from_actor_usage
from ouroboros.tools.registry import ToolContext, active_repo_dir_for
from ouroboros.usage_accounting import (
    PHYSICAL_ATTEMPT_STATES, POSITIVE_PHYSICAL_ATTEMPT_STATES,
)
from ouroboros.utils import utc_now_iso


PLAN_REVIEW_MAX_TOKENS = 65536
PLAN_RAW_TEXT_PREVIEW_CHARS = 2_000
from ouroboros.tools.plan_review_artifacts import (  # noqa: E402, F401 - compatibility imports
    _row_has_physical_dispatch,
    persist_wave as persist_plan_review_wave_artifact,
    read_wave as read_plan_review_wave_artifact,
)
PLAN_REVIEW_EFFORT = "high"
# ``None`` means no plan-local cognition cutoff.  The substrate settles against
# the owner deadline or shared transport bound, keeping the historical 560s
# number from being reused as an HTTP timeout.
PLAN_REVIEW_SLOT_TIMEOUT_SEC = None
# Per-slot provenance of what the reviewer read (BIBLE P3, retrieving reviewers):
# an api_chat slot read exactly the host-assembled packet; an agent_session slot
# retrieved with its own tools and the host did not observe what it opened; a
# native tool-round slot retrieved with HOST tools, so its reads are observed —
# a stronger disclosure that is still never a claim of full-surface coverage.
HOST_FILE_READ_ASSEMBLED = "host_assembled_packet"
HOST_FILE_READ_UNOBSERVED = "unobserved"
HOST_FILE_READ_OBSERVED = "host_observed"

log = logging.getLogger(__name__)


def _packet_kwargs(fn: Any, **candidates: Any) -> Dict[str, Any]:
    parameters = inspect.signature(fn).parameters
    return {key: value for key, value in candidates.items() if key in parameters}


def _task_objective(ctx: ToolContext) -> str:
    contract = getattr(ctx, "task_contract", {})
    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(contract, dict) or not contract:
        contract = metadata.get("task_contract") if isinstance(metadata, dict) else {}
    contract = contract if isinstance(contract, dict) else {}
    return str(contract.get("objective") or contract.get("description") or "")


def _governance_text(system_root: pathlib.Path, rel_path: str) -> str:
    from ouroboros.tools.review_helpers import load_governance_doc

    text = load_governance_doc(system_root, rel_path, on_missing="explicit")
    return "" if text.startswith("[⚠️ OMISSION") else text


def _session_task_text(system_prompt: str, user_content: str, session_root: str) -> str:
    return (
        "RETRIEVING REVIEWER (agent session): you run read-only inside "
        f"{session_root or 'the active workspace'}. The evidence below is the host's REDACTED "
        "snapshot — the same bytes every reviewer sees — so do NOT re-read the raw evidence "
        "locators (a session reading originals would leak what the api route redacts, 4e133c8a); "
        "the ONE exception is the governance pack — BIBLE.md and docs/ARCHITECTURE.md are public "
        "repository documents you MAY read raw, and MUST read in full when the pack marks them "
        "MANDATORY FULL READS (a self-modification plan), even if the agent also declared them as "
        "evidence. Retrieve any OTHER repository context with your own tools.\n\n"
        + system_prompt + "\n\n" + user_content
    )


def build_plan_review_packet(
    ctx: ToolContext, *, spec: dict, request: Any, manifest: dict, constitutional: bool,
    system_root: pathlib.Path, active_root: pathlib.Path, cycle_index: int,
    enforcement: str, previous: Optional[dict],
) -> tuple[str, str, str]:
    """Build the api packet and route-owned retrieving-session task."""
    from ouroboros.context_layout import generate_doc_nav_map
    from ouroboros.tools import plan_spec
    from ouroboros.tools.plan_packet import (
        build_plan_review_system_prompt, build_plan_review_user_content,
    )
    from ouroboros.tools.review_helpers import load_checklist_section

    try:
        checklist = load_checklist_section("Plan Review Checklist")
    except Exception as exc:
        log.warning("Could not load Plan Review Checklist: %s", exc)
        checklist = ""
    bible_text = _governance_text(system_root, "BIBLE.md")
    architecture_text = _governance_text(system_root, "docs/ARCHITECTURE.md")
    bible_nav_map = architecture_nav_map = ""
    if not constitutional:
        if bible_text.strip():
            bible_nav_map = generate_doc_nav_map(bible_text, title="BIBLE.md", rel_path="BIBLE.md")
        if architecture_text.strip():
            architecture_nav_map = generate_doc_nav_map(
                architecture_text, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md")

    def system(by_retrieval: bool) -> str:
        return build_plan_review_system_prompt(
            checklist_section=checklist, constitutional=constitutional,
            bible_text=bible_text if constitutional else None, cycle_index=cycle_index,
            enforcement=enforcement,
            **_packet_kwargs(
                build_plan_review_system_prompt,
                bible_locator=str(system_root / "BIBLE.md"), bible_nav_map=bible_nav_map or None,
                architecture_text=architecture_text if constitutional else None,
                architecture_locator=str(system_root / "docs" / "ARCHITECTURE.md"),
                architecture_nav_map=architecture_nav_map or None,
                governance_by_retrieval=by_retrieval,
            ),
        )

    system_prompt = system(False)
    prior = ([{"cycle_index": previous.get("cycle_index"), "aggregate": previous.get("aggregate"),
               "findings": list(previous.get("findings") or [])}] if previous else [])
    delta = (
        {"unavailable": "previous frozen spec body truncated to fit the durable state; hashes name the original"}
        if previous and previous.get("spec_body_truncated")
        else plan_spec.spec_delta(previous.get("spec"), spec) if previous else None
    )
    user_content = build_plan_review_user_content(
        manifest=manifest, objective=_task_objective(ctx), goal=spec["goal"],
        plan_prose=request.plan, spec=spec, prior_cycles=prior,
        dispositions=list((previous or {}).get("dispositions") or []), spec_delta=delta,
        root_exploration_log=root_exploration_log(ctx),
        **_packet_kwargs(build_plan_review_user_content, cycle_index=cycle_index),
    )
    return system_prompt, user_content, _session_task_text(system(True), user_content, str(active_root))


def publish_plan_review_projection(
    ctx: ToolContext,
    review: dict,
    text: str,
) -> str:
    """Publish control metadata only from validated structured review state."""
    aggregate = review.get("aggregate_signal")
    closed = review.get("closed")
    if aggregate not in {"GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED"}:
        raise ValueError(f"invalid plan review aggregate signal: {aggregate!r}")
    if type(closed) is not bool:
        raise ValueError("plan review closed state must be boolean")
    if (aggregate == "GREEN" and not closed) or (
        aggregate in {"REVISE_PLAN", "DEGRADED"} and closed
    ):
        raise ValueError(
            f"invalid plan review control state: outcome={aggregate}, closed={closed}"
        )
    return _publish_tool_result(
        ctx,
        ToolResult(
            status="ok",
            code="OK",
            text=text,
            meta={
                "plan_review_outcome": aggregate,
                "plan_review_closed": closed,
            },
        ),
    )


def publish_rendered_wave(
    ctx: ToolContext, wave: dict, *, cap, cycles_paid: int, enforcement: str,
    cached: bool = False, notes=None, reminder: str = "", head: str = "",
) -> str:
    """Render one recorded wave and publish it as the typed plan result (D02).

    The public text and the native structured control leave in ONE ``ToolResult``:
    ``plan_render.wave_control_state`` is the same projection the rendered
    ``PLAN_REVIEW_CONTROL_JSON`` footer reads, so the loop's trusted metadata can
    never diverge from the text the model sees."""
    from ouroboros.tools.plan_render import _render_wave, wave_control_state

    outcome, closed = wave_control_state(wave)
    text = head + _render_wave(
        wave, cap=cap, cycles_paid=cycles_paid, enforcement=enforcement,
        cached=cached, notes=notes, reminder=reminder,
    )
    return publish_plan_review_projection(
        ctx, {"aggregate_signal": outcome, "closed": closed}, text)


def plan_deadline_skip(ctx: ToolContext, *, emit: bool = False) -> str:
    """Project the existing deadline rail without starting a paid reviewer panel."""
    from ouroboros.config import get_plan_task_deadline_min_sec

    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    deadline = parse_deadline_ts(metadata.get("deadline_at"))
    if deadline is None:
        return ""
    remaining = (deadline - utc_now()).total_seconds()
    scaled = max(0.0, remaining / 4.0)
    minimum = get_plan_task_deadline_min_sec()
    if remaining > 0 and scaled >= minimum:
        return ""
    if emit:
        try:
            event_queue = getattr(ctx, "event_queue", None)
            if event_queue is not None:
                event_queue.put_nowait({
                    "type": "plan_task_deadline_skip",
                    "task_id": str(getattr(ctx, "task_id", "") or ""),
                    "remaining_sec": round(remaining, 1),
                    "scaled_ceiling_sec": round(scaled, 1),
                    "min_useful_sec": minimum,
                    "ts": utc_now_iso(),
                })
        except Exception:
            pass
    cause = (
        "the task deadline has expired; no reviewer work was started."
        if remaining <= 0 else
        f"insufficient time for useful planning — remaining {int(remaining)}s gives a "
        f"review window of {int(scaled)}s (< {int(minimum)}s useful floor)."
    )
    return (
        f"PLAN_TASK_SKIPPED_DEADLINE: {cause} Proceed with your own best plan "
        "directly; do not re-call plan_task under this deadline."
    )


def record_raw_plan_request_attempt(
    envelope: dict, state_root: pathlib.Path, task_id: str, *, reason: str,
) -> str:
    """Supersede prior plan authority before semantic decoding can fail: an invalid
    envelope must not leave an older closed wave standing as the current one."""
    payload = {"domain": "invalid_plan_task_attempt", "envelope": envelope}
    fingerprint = sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    from ouroboros.task_results import record_plan_review_attempt

    record_plan_review_attempt(
        state_root, task_id, fingerprint=fingerprint, status="open", reason=reason,
    )
    return fingerprint


def plan_review_slots() -> list:
    """The configured commit-triad rows as plan-review ``ReviewSlot`` objects:
    the shared ``triad_delivery_slots`` builder (one reader of the triad rows
    for plan, skill and acceptance review) with plan review's own slot
    properties — timeout, output budget, temperature, and ``PLAN_REVIEW_EFFORT``
    as the effort default. Both delivery kinds ride; slot ids are the rows' own.
    """
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    return triad_delivery_slots(
        role_hint="plan reviewer", default_effort=PLAN_REVIEW_EFFORT,
        timeout_sec=PLAN_REVIEW_SLOT_TIMEOUT_SEC, max_tokens=PLAN_REVIEW_MAX_TOKENS,
        temperature=0.2,
    )


def slot_is_session(slot: Any) -> bool:
    """TRANSPORT fact: this slot runs a delegated Claudexor session (daemon
    health, threads, invocation custody). Never use it for delivery class."""
    from ouroboros.review_execution import ReviewRouteKind

    return getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION


def slot_retrieves(slot: Any) -> bool:
    """DELIVERY class: this slot reads the subject with its own tools — a
    session row or a configured-subagent native row. Packet sizing, fit and
    retrieval semantics test THIS, not the route name."""
    return bool(getattr(slot, "retrieves", slot_is_session(slot)))


async def run_plan_review_slots(
    ctx: ToolContext,
    slots: list,
    *,
    system_prompt: str,
    user_content: str,
    session_task: str = "",
    session_root: str = "",
    output_contract: str = "",
    slot_messages: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    session_threads: Optional[Dict[str, str]] = None,
    retry_key: str = "",
) -> list[dict]:
    """ONE ``ReviewRequest`` fanned across the configured rows through the substrate.

    api_chat rows read ``messages`` (system + user packet); agent_session rows read
    ``session_task``/``session_root``/``policy.output_contract`` and retrieve the
    rest themselves. Returns one raw row per slot, in slot order, carrying the id
    the substrate RAN under, its route, text/error, refs, usage and the
    ``host_file_read_attestation`` fact."""
    from ouroboros.review_substrate import ReviewRequest, run_review_request
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from ouroboros.tools.review_synthesis import build_plan_review_messages

    request = ReviewRequest(
        surface="plan_review",
        goal="Review the proposed plan spec before the work starts.",
        messages=build_plan_review_messages(
            system_prompt, user_content, plan_user_stable_len(user_content),
        ),
        slot_messages=dict(slot_messages or {}),
        task_id=str(getattr(ctx, "task_id", "") or "plan_review"),
        call_type="plan_review",
        max_tokens=PLAN_REVIEW_MAX_TOKENS,
        temperature=0.2,
        no_proxy=True,
        session_task=session_task,
        session_root=session_root,
        session_threads=dict(session_threads or {}),
        retry_key=str(retry_key or ""),
        # The paid cycle's identity (plan fingerprint + cycle) owns its cache
        # split: a revised plan under the same task/model/slot starts cold.
        usage_attribution={"review_wave_id": str(retry_key or "")} if retry_key else {},
        policy={"output_contract": output_contract} if output_contract else {},
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_review_request(
            request,
            slots=list(slots),
            drive_root=pathlib.Path(ctx.drive_root),
            llm=LLMClient(),
            usage_ctx=ctx,
        ),
    )
    by_id = {str(slot.slot_id): slot for slot in slots}
    rows = [_plan_row_from_actor(actor, by_id.get(str(actor.get("slot_id") or ""))) for actor in result.actors]
    answered = {row["slot_id"] for row in rows}
    for slot in slots:  # a slot the substrate never answered for is still a configured row
        if str(slot.slot_id) not in answered:
            rows.append(_plan_row_from_actor({"slot_id": slot.slot_id, "model": slot.model,
                                              "status": "error", "error": "no actor record",
                                              "failure_code": "review_custody_lost",
                                              "operation_state": "custody_lost",
                                              "late_result_pending": True}, slot))
    return rows


def _typed_facts_from(source: Any, get: Any) -> Dict[str, Any]:
    """The B1 typed failure facts off one source via ``get(source, key)`` — the
    ONE key list is ``review_substrate.TYPED_FAILURE_FACT_KEYS`` (shared with the
    last-execution recorder; sources differ, keys must not). ``http_status`` stays
    raw (int-or-None); the rest coerce to honest-empty strings."""
    from ouroboros.review_substrate import TYPED_FAILURE_FACT_KEYS

    return {key: (get(source, key) if key == "http_status" else str(get(source, key) or ""))
            for key in TYPED_FAILURE_FACT_KEYS}


def _plan_row_from_actor(actor: Dict[str, Any], slot: Any) -> dict:
    usage = actor.get("usage") or {}
    text = actor.get("raw_text") or ""
    error = actor.get("error") or ""
    if actor.get("status") not in {"ok", "empty"} and not error:
        error = str(actor.get("status") or "review failed")
    session = slot_is_session(slot) if slot is not None else False
    physical_attempt_state = str(
        usage.get("physical_attempt_state")
        or actor.get("physical_attempt_state")
        or ""
    )
    provider_status_code = usage.get("provider_status_code")
    if provider_status_code is None:
        provider_status_code = actor.get("provider_status_code")
    if isinstance(provider_status_code, bool):
        provider_status_code = None
    elif provider_status_code is not None:
        try:
            provider_status_code = int(provider_status_code)
        except (TypeError, ValueError, OverflowError):
            provider_status_code = None
    return {
        # Identity is CARRIED, never re-derived: the row keeps the slot_id the
        # substrate ran, so duplicate-model plan rows stay distinguishable.
        "slot_id": str(actor.get("slot_id") or getattr(slot, "slot_id", "") or ""),
        "model": str(usage.get("resolved_model") or actor.get("model") or getattr(slot, "model", "") or ""),
        "request_model": str(getattr(slot, "model", "") or actor.get("model") or ""),
        "route": "agent_session" if session else "api_chat",
        # Delivery-truthful provenance: a native tool-round actor discloses its
        # HOST-OBSERVED reads through its usage; session stays unobserved and a
        # packet row stays host-assembled.
        "host_file_read_attestation": (
            str(usage.get("host_file_read_attestation") or "")
            or (HOST_FILE_READ_UNOBSERVED if session else HOST_FILE_READ_ASSEMBLED)
        ),
        "text": text,
        "error": error or None,
        "prompt_ref": actor.get("prompt_ref") or {},
        "response_ref": actor.get("response_ref") or {},
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "cost": float(usage["cost"]) if usage.get("cost") is not None else None,
        # Presentation-only receipt from the actor's RETURNED usage. Requested
        # slot route/model, money, profile and raw telemetry never enter it.
        "executions": review_executions_from_actor_usage([actor]),
        # B1 typed failure facts, forwarded VERBATIM from the actor record (empty on
        # success and on pre-typed engines) so downstream never regresses to matching
        # the error prose for the code or the reset instant.
        **_typed_facts_from(actor, lambda source, key: source.get(key)),
        "physical_attempt_state": physical_attempt_state,
        "provider_status_code": provider_status_code,
        "capability_delta": usage.get("capability_delta") or [],
        "review_thread_id": str(usage.get("review_thread_id") or ""),
        "review_turn_id": str(usage.get("review_turn_id") or ""),
        "review_thread_receipt": usage.get("review_thread_receipt") or {},
        "auth_route_receipt": usage.get("auth_route_receipt") or {},
        "profile_continuity_receipt": usage.get("profile_continuity_receipt") or {},
        "applied_profile": str(usage.get("applied_profile") or ""),
        "operation_id": str(actor.get("operation_id") or ""),
        "operation_state": str(actor.get("operation_state") or "settled"),
        "late_result_pending": bool(actor.get("late_result_pending")),
        "pending_invocation_id": str(
            usage.get("pending_invocation_id") or actor.get("pending_invocation_id") or ""
        ),
        "delegated_run_id": str(
            usage.get("delegated_run_id") or actor.get("delegated_run_id") or ""
        ),
    }


def plan_row_typed_facts(row: Dict[str, Any]) -> Dict[str, Any]:
    """The typed failure facts a wave actor record carries forward from a plan row.

    One projection for the engine's wave records (``plan_review.py`` sits at its
    size pin, so the logic lives here). Every fact defaults to honest absence —
    rows from pre-B1 engines and typed-fact-free rows (``plan_slot_fit``'s
    preflight_oversize rows) behave exactly as before."""
    facts = {
        **_typed_facts_from(row, lambda source, key: source.get(key)),
        "capability_delta": row.get("capability_delta") or [],
    }
    physical_attempt_state = str(row.get("physical_attempt_state") or "")
    provider_status_code = row.get("provider_status_code")
    if physical_attempt_state or provider_status_code is not None:
        facts.update({
            "physical_attempt_state": physical_attempt_state,
            "provider_status_code": provider_status_code,
        })
    pending_invocation_id = str(row.get("pending_invocation_id") or "")
    delegated_run_id = str(row.get("delegated_run_id") or "")
    if row.get("operation_id") or row.get("late_result_pending") \
            or str(row.get("operation_state") or "settled") != "settled" \
            or pending_invocation_id or delegated_run_id:
        facts.update({
            "operation_id": str(row.get("operation_id") or ""),
            "operation_state": str(row.get("operation_state") or "settled"),
            "late_result_pending": bool(row.get("late_result_pending")),
            "pending_invocation_id": pending_invocation_id,
            "delegated_run_id": delegated_run_id,
        })
    return facts


def synthesize_plan_review_wave(
    rows: list[dict], *, state: dict, spec: dict, request_plan: str,
    fingerprint: str, previous: Optional[dict], manifest: dict, manifest_hash: str,
    constitutional: bool, constitutional_note: str, cycle_index: int, retry_key: str,
    enforcement: str, cap: Any, quorum: int, configured_slots: list,
    health_evidence: Any,
) -> tuple[dict, set[str], dict]:
    """Validate raw actor rows and build one durable plan-review wave."""
    from ouroboros.tools import plan_spec

    ids = plan_spec.spec_ids(spec)
    seen_before = {str(s) for s in state.get("need_evidence_seen") or []}
    seen_after = set(seen_before)
    slot_results, slot_records = [], []
    for row in rows:
        findings, disclosures = [], plan_row_disclosures(row)
        error = str(row.get("error") or "")
        ok = not error
        if ok:
            parsed, parse_error = plan_spec.parse_findings(str(row.get("text") or ""))
            if parse_error:
                ok, error = False, parse_error
            else:
                findings, finding_disclosures, slot_seen = plan_spec.validate_findings(
                    parsed, spec_ids=ids, seen_locators=seen_after,
                    slot=str(row.get("slot_id") or ""),
                )
                disclosures += finding_disclosures
                seen_after |= set(slot_seen)
        slot_results.append({"slot": row.get("slot_id"), "model": row.get("model"),
                             "ok": ok, "findings": findings, "error": error or None})
        slot_records.append(plan_wave_actor_record(
            row, ok=ok, error=error, disclosures=disclosures,
            raw_text_preview_chars=PLAN_RAW_TEXT_PREVIEW_CHARS,
        ))
    agg = plan_spec.aggregate(slot_results, quorum=quorum)
    aggregate = str(agg["aggregate"])
    wave = {
        "schema_version": 2, "cycle_index": cycle_index, "retry_key": retry_key,
        "request_fingerprint": fingerprint,
        "previous_fingerprint": str((previous or {}).get("request_fingerprint") or ""),
        "goal": spec["goal"], "plan_prose_hash": sha256(request_plan.encode("utf-8")).hexdigest(),
        "spec": spec, "spec_hash": plan_spec.spec_hash(spec),
        "evidence_manifest": {
            "declared": list(manifest.get("declared") or []),
            "attached": [{"locator": a.get("locator"), "sha256": a.get("sha256"),
                          "bytes": a.get("bytes"),
                          **({"secrets_redacted": True} if a.get("secrets_redacted") else {})}
                         for a in manifest.get("attached") or []],
            "omissions": list(manifest.get("omissions") or []),
            "reviewer_requested": list(manifest.get("reviewer_requested") or []),
            "reviewer_requested_dropped": list(manifest.get("reviewer_requested_dropped") or []),
        },
        "evidence_manifest_hash": manifest_hash, "constitutional": bool(constitutional),
        "constitutional_note": constitutional_note, "findings": list(agg["findings"]),
        "aggregate": aggregate, "reasons": list(agg["reasons"]), "counts": dict(agg["counts"]),
        "closed": aggregate == "GREEN", "dispositions": [], "actors": slot_records,
        "custody_pending": False,
        "actors_degraded": [str(r["slot_id"]) for r in slot_records if not r["ok"]],
        "enforcement": enforcement, "cycle_cap": cap,
        "paid": any(_row_has_physical_dispatch(row) for row in slot_records),
        "health_epoch": plan_health_epoch(health_evidence),
        "reviewer_config_fingerprint": plan_reviewer_config_fingerprint(configured_slots),
        **plan_quorum_unreachable_facts(slot_records, quorum=quorum), "reviewed_at": utc_now_iso(),
    }
    # A partially-settled paid cycle is not yet allowed to mutate the next
    # request envelope.  Persisting one sibling's NEED_EVIDENCE now would
    # change the manifest/fingerprint before the other siblings settle, making
    # the exact in-flight cycle undiscoverable on retry.  The findings remain
    # in the exact wave; reconciliation replays them and advances the durable
    # task-level evidence set only once the whole cycle is terminal.
    if plan_wave_has_in_flight(wave):
        # A parseable quorum is not final authority while any paid physical
        # reviewer can still settle. Keep the arithmetic counts, but expose
        # the wave as the existing open DEGRADED state so both live and
        # process-loss paths remain fail-closed without a new state/ledger.
        wave["aggregate"] = "DEGRADED"
        wave["closed"] = False
        wave["custody_pending"] = True
        wave["reasons"].append("review_late_result_pending")
        seen_after = seen_before
    return wave, seen_after, agg


def plan_wave_actor_record(
    row: Dict[str, Any], *, ok: bool, error: str,
    disclosures: List[str], raw_text_preview_chars: int,
) -> Dict[str, Any]:
    """Build the durable Plan actor projection outside the orchestration loop."""
    from ouroboros.utils import truncate_review_artifact

    return {
        "slot_id": row.get("slot_id"), "model": row.get("model"),
        "route": row.get("route"), "executions": list(row.get("executions") or []),
        "host_file_read_attestation": row.get("host_file_read_attestation"),
        "ok": ok, "error": error or None, "disclosures": disclosures,
        **plan_row_typed_facts(row),
        "prompt_ref": row.get("prompt_ref") or {},
        "response_ref": row.get("response_ref") or {},
        "tokens_in": row.get("tokens_in"), "tokens_out": row.get("tokens_out"),
        "cost": row.get("cost"),
        "raw_text_preview": (
            truncate_review_artifact(
                str(row.get("text") or ""), limit=raw_text_preview_chars,
            ) if not ok else ""
        ),
        "review_thread_id": str(row.get("review_thread_id") or ""),
        "review_turn_id": str(row.get("review_turn_id") or ""),
        "review_thread_receipt": row.get("review_thread_receipt") or {},
        "auth_route_receipt": row.get("auth_route_receipt") or {},
        "profile_continuity_receipt": row.get("profile_continuity_receipt") or {},
        "applied_profile": str(row.get("applied_profile") or ""),
    }


def plan_row_disclosures(row: Dict[str, Any]) -> List[str]:
    """Producer-side telemetry disclosures seeding one actor record.

    A REAL pinned-profile mismatch (receipt status ``cannot_verify``) is
    disclosed on the actor row the agent reads; ``no_expectation_recorded``
    stays a durable-receipt fact only (every unpinned wave would otherwise
    carry noise). Profile continuity is telemetry: it never gates parsing,
    counting, or PASS — the findings stand either way."""
    receipt = row.get("profile_continuity_receipt") or {}
    if str(receipt.get("status") or "") != "cannot_verify":
        return []
    reason = str(receipt.get("verification_reason") or "") or "unexplained_profile_drift"
    return [f"profile_continuity: cannot_verify ({reason})"]


def plan_wave_progress_line(aggregate: str, counts: Dict[str, Any], *, cycles_paid: int, cap: Any) -> str:
    """The wave's final owner-visible progress line (pure; ``plan_review.py``
    sits at its size pin, so the formatting lives here). Honest DEGRADED:
    zero-count tails must never read as a clean result, so the
    parseable/configured ratio and the distrust are named inline; every other
    aggregate renders byte-identically to the plain form."""
    verdict = (
        f"DEGRADED ({counts['parseable']}/{counts['configured']} "
        "parseable reviewers; counts are untrusted)"
        if aggregate == "DEGRADED" else aggregate
    )
    return (
        f"📐 plan_task: {verdict} — {counts['blocking']} blocking / "
        f"{counts['note']} note / {counts['need_evidence']} need_evidence; "
        f"cycles paid {cycles_paid}{'' if cap is None else f'/{cap}'}"
    )


# Root exploration log (plan F3/S8): the task's OWN tool calls before this call,
# read from the task-local conversation — never a scan of tools.jsonl.
_EXPLORATION_TAIL_CALLS = 40
_EXPLORATION_ARGS_CHARS = 240
_EXPLORATION_RESULT_CHARS = 320


def root_exploration_log(ctx: ToolContext) -> Optional[str]:
    """Bounded tail of THIS task's tool calls before this plan_task, from the task-local
    conversation (S8): exact omitted count, never a scan of the shared tools.jsonl.
    ``None`` when the host holds no conversation for this context (named omission)."""
    from ouroboros.review_evidence import _accept_redact_cap

    messages = getattr(ctx, "messages", None)
    if not isinstance(messages, list):
        return None
    results: Dict[str, str] = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            results[str(message.get("tool_call_id") or "")] = str(message.get("content") or "")
    calls: List[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            name = str((fn or {}).get("name") or "")
            if not name or name == "plan_task":
                continue
            # I-04: raw tool args/results must be redacted before reviewers see them;
            # reuse acceptance's projection (`_accept_redact_cap`), not a second path (P7).
            args = _accept_redact_cap(str((fn or {}).get("arguments") or ""), _EXPLORATION_ARGS_CHARS)
            result = _accept_redact_cap(
                results.get(str(call.get("id") or ""), ""), _EXPLORATION_RESULT_CHARS,
            ).replace("\n", " ")
            calls.append(f"- {name}({args}) → {result or '(no result recorded)'}")
    tail = calls[-_EXPLORATION_TAIL_CALLS:]
    header = (
        f"{len(calls)} tool call(s) by this task before this plan_task; showing the last "
        f"{len(tail)}; omitted {len(calls) - len(tail)} (bounded tail)."
    )
    return "\n".join([header, *tail])


# Dedup memo for the advisory-open event: one event per recorded-open
# (data root, task, fingerprint, health-epoch) STATE, not per call — empty-epoch
# DEGRADED re-dispatches and unpaid $0 re-discoveries re-enter the emitter with
# an unchanged state and must not spam the owner. Process-local by design: a
# restart may re-announce an already-announced state once (disclosed residual —
# this is an event rail, never authority).
_ADVISORY_OPEN_SEEN: Dict[tuple, bool] = {}
_ADVISORY_OPEN_SEEN_MAX = 512


def emit_plan_review_advisory_open(
    ctx: ToolContext, drive_root: Any, *, task_id: str, wave: Dict[str, Any],
    cycles_paid: int, cap: Any,
) -> None:
    """ONE typed owner-visible event when a wave RECORDS open under advisory
    enforcement (B2): loud at the moment it happens, not only when finalization
    later appends ``owner_hurry.plan_review_disclosure``. Deduplicated per
    (fingerprint, health-epoch) recorded-open state (see ``_ADVISORY_OPEN_SEEN``);
    replays, dispositions and re-renders never reach this emitter at all.
    Durability is UNCONDITIONAL: the ``events.jsonl`` append always lands — the
    live queue path persists only task_checkpoint rows — and a live queue
    additionally gets the UI push. The dedup memo is inserted ONLY AFTER the
    durable append succeeded (review fix 6): a failed append is logged loudly and
    NOT memoized, so the next call for the same state retries the whole emission
    instead of the memo silently swallowing an event that never landed. Never
    raises."""
    from ouroboros.utils import append_jsonl, emit_log_event

    key = (str(drive_root or ""), str(task_id or ""),
           str(wave.get("request_fingerprint") or ""),
           json.dumps(wave.get("health_epoch") or [], sort_keys=True, default=str))
    if key in _ADVISORY_OPEN_SEEN:
        return
    row = {
        "type": "plan_review_advisory_open",
        "surface": "plan_review",
        "task_id": str(task_id or ""),
        "fingerprint": str(wave.get("request_fingerprint") or ""),
        "aggregate": str(wave.get("aggregate") or ""),
        "cycle_index": wave.get("cycle_index"),
        "paid": bool(wave.get("paid")),
        "cycles_paid": int(cycles_paid),
        "cap": cap,
        "enforcement": "advisory",
        # Bounded per-slot typed facts: who failed, with what code, until when.
        "slots": [
            {"slot_id": a.get("slot_id"), "ok": bool(a.get("ok")),
             "failure_code": str(a.get("failure_code") or ""),
             "reset_at": str(a.get("reset_at") or "")}
            for a in (wave.get("actors") or []) if isinstance(a, dict)
        ],
    }
    stamped = {"ts": utc_now_iso(), **row}
    try:
        if drive_root:
            append_jsonl(pathlib.Path(str(drive_root)) / "logs" / "events.jsonl", stamped)
    except Exception:
        log.warning("plan_review_advisory_open durable append failed for %s; "
                    "not memoized — the next call retries", task_id, exc_info=True)
        return
    _ADVISORY_OPEN_SEEN[key] = True
    while len(_ADVISORY_OPEN_SEEN) > _ADVISORY_OPEN_SEEN_MAX:
        _ADVISORY_OPEN_SEEN.pop(next(iter(_ADVISORY_OPEN_SEEN)))
    try:
        event_queue = getattr(ctx, "event_queue", None)
        if event_queue is not None:
            emit_log_event(event_queue, stamped, log_label="plan review")
    except Exception:
        log.debug("plan_review_advisory_open emission failed for %s", task_id, exc_info=True)


def plan_payload_roots(ctx: ToolContext, locators: List[str]) -> list[pathlib.Path]:
    """Skill-payload exemption roots for ``plan_spec.resolve_constitutional``.

    Skill-payload paths live in the DATA plane (``data/skills/<bucket>/…``, not the
    system repo) and never make a plan a self-modification by themselves — the same
    frozen predicate the deleted ``resolve_plan_class`` applied. A recognized locator
    is exempt however it resolves: its data-plane payload root AND its literal
    resolution against the active root are both returned. Classification-only
    (write gates are unrelated code paths); a drive-resolution failure skips the
    exemption, as before."""
    from ouroboros.contracts.skill_payload_policy import resolve_skill_payload_target
    from ouroboros.tool_access import canonical_data_root

    try:
        drive = canonical_data_root(ctx)
        active = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    except Exception:
        return []
    roots: list[pathlib.Path] = []
    for raw in locators or []:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            target = resolve_skill_payload_target(drive, text)
        except ValueError:
            continue
        roots.append(target.payload_root)
        try:
            candidate = pathlib.Path(text)
            roots.append((candidate if candidate.is_absolute() else active / candidate).resolve(strict=False))
        except (OSError, ValueError):
            continue
    return roots


# ------------------------------------------------------------ panel health (B2b)
#
# ONE health snapshot before fan-out turns slots with POSITIVE structural evidence
# of a spent lane into $0 typed skip rows that stay in the quorum denominator
# (BIBLE P3: the quorum never silently narrows). Unknown health DISPATCHES
# (fail-open); transient daemon states (`daemon_recovery_only`, a dead socket)
# are never skip evidence and never enter the epoch (roast pt 9). Side effect,
# disclosed: `ensure_owned_gateway` may lazily SPAWN the owned daemon and runs
# its per-ensure rotation reconcile — both bounded and fail-open (a spawn/probe
# failure returns None here, never an exception).
#
# SCOPE OF THE ANSWER: `subagents.route_health` — the ONE manifest reader — judges
# a ROUTE (harness id + pinned model) NARROWED to the slot's pinned credential
# profile whenever the row pins one, exactly as the dispatcher asks it
# (`review_execution`: a pin is strict, so a healthy sibling account must not vouch
# a spent pin). A row that pins NO profile keeps the route-wide answer: which
# account an unpinned run lands on is Claudexor's rotation business, so a route
# whose pool still holds a live account reads healthy — such a slot dispatches
# (fail-open) and fails typed downstream (B1) if rotation lands it badly. No second
# health oracle is built here. api_chat rows have no route health source at all and
# always dispatch. Cursor lanes without quota snapshots simply dispatch too.


def _slot_session_route(slot: Any) -> Any:
    """The route an agent_session slot would dispatch on (None when unresolvable —
    the dispatch path owns that refusal; health has nothing to say about it).

    Mirrors the dispatcher's own resolution (`review_execution` `_session_route`)
    including the row's optional credential pin, so health judges the SAME account
    the run would actually ride. Effort is deliberately NOT mirrored: health reads
    route identity, model and profile only.
    """
    import dataclasses

    from ouroboros.review_execution import review_session_route
    from ouroboros.subagents import parse_subagent_harness

    spec = str(getattr(slot, "session_target", "") or "")
    if spec:
        route = parse_subagent_harness(spec)
        pin = str(getattr(slot, "session_profile", "") or "")
        if route is not None and pin:
            route = dataclasses.replace(route, profile_id=pin)
        return route
    return review_session_route()


def _structural_skip_code(reason: str, reset_at: str) -> str:
    """POSITIVE structural evidence only: a dated window exhaustion whose reset is
    still ahead, or a typed dead-pool code. An UNDATED exhaustion, a stale reset and
    every other reason (route_disabled, transient daemon states, unknown) dispatch —
    the pre-dispatch admission and the run itself refuse typed downstream at ~$0."""
    from ouroboros.gateways.claudexor import WINDOW_EXHAUSTED_CODES

    if reset_at:
        instant = parse_deadline_ts(reset_at)
        if instant is not None and instant > utc_now():
            return reason or "subscription_window_exhausted"
        return ""
    if reason in WINDOW_EXHAUSTED_CODES and reason != "subscription_window_exhausted":
        # e.g. credential_pool_exhausted: the pool itself is dead. Honest note: NO
        # producer routes this code through `route_health` today — the branch is
        # the contract point for PR-A A5's route_health sync (the typed pool
        # terminal), kept on purpose so the sync lands without an engine change.
        return reason
    return ""


def plan_panel_health_snapshot(slots: list) -> Optional[Dict[str, Dict[str, str]]]:
    """``{slot_id: {failure_code, reset_at}}`` for the slots a pre-fan-out snapshot
    proves structurally dead; ``{}`` when the snapshot ran and found none; ``None``
    when no snapshot could be captured (daemon unprovisioned/unreachable) — unknown
    health dispatches, and a FAILED snapshot must never read as "everything healed"
    at the replay seam."""
    session_slots = [slot for slot in slots if slot_is_session(slot)]
    if not session_slots:
        return {}
    from ouroboros.claudexor_daemon import ensure_owned_gateway, owned_daemon_provisioned
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import delegated_run_shape, route_health

    if not owned_daemon_provisioned():
        return None
    shape = delegated_run_shape(False)  # a reviewer reads and answers
    evidence: Dict[str, Dict[str, str]] = {}
    by_route: Dict[tuple, tuple[str, str]] = {}
    gateway = None
    try:
        gateway = ensure_owned_gateway()
        for slot in session_slots:
            route = _slot_session_route(slot)
            if route is None:
                continue
            # The PIN is part of the subject, so it is part of the memo key: two
            # rows on the same harness+model but different accounts must never
            # share one verdict (a spent pin would otherwise be vouched for by a
            # sibling's health, or vice versa).
            pin = str(getattr(route, "profile_id", "") or "")
            key = (route.route_id, route.model, pin)
            if key not in by_route:
                by_route[key] = route_health(gateway, route.route_id, shape,
                                             route_model=route.model,
                                             pinned_profile=pin)
            code = _structural_skip_code(*by_route[key])
            if code:
                evidence[str(getattr(slot, "slot_id", "") or "")] = {
                    "failure_code": code, "reset_at": by_route[key][1]}
    except ClaudexorUnavailable:
        return None  # transient (incl. daemon_recovery_only): never a skip row
    except Exception:
        log.debug("plan panel health snapshot failed (fail-open)", exc_info=True)
        return None
    finally:
        if gateway is not None:
            gateway.close()
    return evidence


def plan_health_skip_rows(slots: list, evidence: Optional[Dict[str, Dict[str, str]]]) -> tuple[list, list[dict]]:
    """``(live_slots, skip_rows)`` — a structurally dead slot becomes a $0 typed row
    shaped like ``plan_slot_fit``'s preflight_oversize rows plus B1's typed facts; it
    stays a configured row in the quorum denominator. Live slots keep dispatching
    even when the dead ones make the quorum unreachable (never `plan_slot_fit`'s
    below-quorum refusal — that would silence the whole panel)."""
    live, rows = [], []
    for slot in slots:
        ev = (evidence or {}).get(str(getattr(slot, "slot_id", "") or ""))
        if not ev:
            live.append(slot)
            continue
        code, reset = str(ev.get("failure_code") or ""), str(ev.get("reset_at") or "")
        pin = str(getattr(slot, "session_profile", "") or "")
        scope = (
            f"Evidence is scoped to this slot's pinned credential profile {pin!r}."
            if pin else
            "Evidence is route-wide: this slot pins no credential profile, so the "
            "answer covers the route's rotation pool rather than one account."
        )
        rows.append({
            "slot_id": str(getattr(slot, "slot_id", "") or ""),
            "model": str(getattr(slot, "model", "") or ""),
            "request_model": str(getattr(slot, "model", "") or ""),
            "route": "agent_session", "host_file_read_attestation": None, "text": "",
            "error": (
                f"health_skip[{code}]: the pre-fan-out panel health snapshot shows this "
                f"slot's delegated route window spent{f' (resets {reset})' if reset else ''}; "
                f"skipped before dispatch at $0. {scope}"
            ),
            "failure_code": code, "reset_at": reset,
            "prompt_ref": {}, "response_ref": {}, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "status": "not_dispatched", "operation_state": "not_dispatched",
        })
    return live, rows


def plan_health_epoch(evidence: Optional[Dict[str, Dict[str, str]]]) -> list[dict]:
    """Material health epoch (roast pt 8): sorted ``{slot, code, reset_at}`` rows, NO
    observed_at. Only snapshot-derived structural evidence enters — transient,
    timeout and parse failures never do (pt 9), so they can never pin a replay."""
    rows = [
        {"slot": str(sid), "code": str(ev.get("failure_code") or ""),
         "reset_at": str(ev.get("reset_at") or "")}
        for sid, ev in (evidence or {}).items()
    ]
    return sorted(rows, key=lambda r: (r["slot"], r["code"], r["reset_at"]))


# Sentinel for "the replay seam captured no fresh snapshot" — distinct from a
# snapshot that RAN and failed (None), which is a meaningful transient-unknown.
PLAN_NO_SNAPSHOT = object()


def plan_reviewer_config_fingerprint(slots: list) -> str:
    """Identity of the configured reviewer roster (roast pt 1): slot ids, targets,
    routes, pinned session targets/profiles AND efforts — a changed roster must not
    inherit an open wave's recorded replay authority. Effort is identity (it changes
    what the reviewer actually does); timeouts are tuning, not identity; roster
    ORDER is identity (it is the configured row order)."""
    rows = [
        [str(getattr(s, "slot_id", "") or ""), str(getattr(s, "model", "") or ""),
         str(getattr(getattr(s, "route", None), "value", "") or ""),
         str(getattr(s, "session_target", "") or ""),
         str(getattr(s, "session_profile", "") or ""),
         str(getattr(s, "effort", "") or "")]
        for s in slots or []
    ]
    # Actor binding is delivery identity; column added only when some row
    # carries one, so unchanged legacy rosters keep their exact bytes.
    actor_ids = [str(getattr(s, "subagent_id", "") or "") for s in slots or []]
    if any(actor_ids):
        for row, actor in zip(rows, actor_ids):
            row.append(actor)
    return sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def plan_wave_replay_decision(slots_fn: Any, existing: Dict[str, Any]) -> tuple:
    """``(stale, fresh_snapshot)`` for one recorded OPEN wave and an identical
    envelope: ``stale=True`` means the recorded replay authority lapsed and the
    envelope must RE-DISPATCH a fresh panel; ``fresh_snapshot`` is the health
    snapshot captured while deciding (``PLAN_NO_SNAPSHOT`` when none was), so the
    dispatch path reuses it instead of probing the daemon twice.

    Replay authority rules:

    * a changed reviewer roster re-dispatches (roast pt 1) — a non-empty recorded
      ``reviewer_config_fingerprint`` differing from the current roster's; an
      absent fingerprint (pre-fingerprint wave) is unknown, never change evidence;
    * only a DEGRADED wave consults lane health — every other open aggregate
      replays exactly as before B2b;
    * a DEGRADED wave with an EMPTY recorded epoch carries no structural snapshot
      evidence (its slots died at dispatch time, invisible to the pre-fan-out
      snapshot): a transient death is never cached as structural (roast pt 9), so
      the identical envelope re-dispatches;
    * a DEGRADED wave with a non-empty epoch replays free while a fresh snapshot
      matches it; a FAILED snapshot (``None``) is transient-unknown and keeps the
      replay; a healed or newly dead lane re-dispatches.
    """
    try:
        slots = list(slots_fn() or [])
        stored_fp = str(existing.get("reviewer_config_fingerprint") or "")
        if stored_fp and stored_fp != plan_reviewer_config_fingerprint(slots):
            return True, PLAN_NO_SNAPSHOT
        if str(existing.get("aggregate") or "") != "DEGRADED":
            return False, PLAN_NO_SNAPSHOT
        stored = existing.get("health_epoch") if isinstance(existing.get("health_epoch"), list) else []
        if not stored:
            return True, PLAN_NO_SNAPSHOT
        fresh = plan_panel_health_snapshot(slots)
    except Exception:
        # Accepted-partial (review fix 4): a configuration-resolution failure keeps
        # the recorded free replay (fail-open, never a paid re-dispatch bought by a
        # transient config error) — but LOUDLY, never a silent except.
        log.warning(
            "plan-review replay decision could not resolve the current reviewer "
            "configuration; keeping the recorded wave's free replay", exc_info=True)
        return False, PLAN_NO_SNAPSHOT
    if fresh is None:
        return False, PLAN_NO_SNAPSHOT
    normalized = sorted((
        {"slot": str(r.get("slot") or ""), "code": str(r.get("code") or ""),
         "reset_at": str(r.get("reset_at") or "")}
        for r in stored if isinstance(r, dict)
    ), key=lambda r: (r["slot"], r["code"], r["reset_at"]))
    return plan_health_epoch(fresh) != normalized, fresh


def plan_wave_has_in_flight(wave: Dict[str, Any]) -> bool:
    """Whether a paid wave must re-enter exact custody reconciliation.

    The durable summary and malformed physical facts are conservative ingress
    signals: they must reach ``in_flight_resume_inputs`` so its exact-roster
    validator can fail closed instead of falling through to a fresh paid cycle.
    """
    if bool(wave.get("custody_pending")):
        return True
    actors = wave.get("actors")
    malformed_roster = (
        not isinstance(actors, list)
        or not actors
        or any(not isinstance(actor, dict) for actor in actors)
    )
    if malformed_roster:
        return bool(wave.get("paid"))
    for actor in actors or []:
        physical_state = str(actor.get("physical_attempt_state") or "").strip().lower()
        if not physical_state and isinstance(actor.get("usage"), dict):
            physical_state = str(
                actor["usage"].get("physical_attempt_state") or ""
            ).strip().lower()
        if physical_state and physical_state not in PHYSICAL_ATTEMPT_STATES:
            return True
        if physical_state in POSITIVE_PHYSICAL_ATTEMPT_STATES and (
            str(actor.get("operation_state") or "").strip().lower() == "not_dispatched"
            or str(actor.get("status") or "").strip().lower() == "not_dispatched"
        ):
            return True
    return any(
        str(actor.get("operation_state") or "") == "in_flight"
        or bool(actor.get("late_result_pending"))
        for actor in actors or []
    )


def plan_in_flight_custody_error(
    *, retry_key: str, task_id: str, active_root: pathlib.Path,
    callable_slots: list, ctx: Any,
) -> str:
    from ouroboros.review_custody import review_retry_custody_available

    if review_retry_custody_available(
        retry_key=retry_key, surface="plan_review", task_id=task_id,
        call_type="plan_review", session_root=str(active_root),
        slots=callable_slots, usage_ctx=ctx,
    ):
        return ""
    return (
        "The prior paid reviewer cycle is still recorded in flight, but its "
        "process-local custody is unavailable. Refusing a duplicate paid send; "
        "the physical outcome remains unknown after process loss."
    )


def plan_quorum_unreachable_facts(slot_records: List[dict], *, quorum: int) -> Dict[str, Any]:
    """``{}`` or the typed structural-unreachability facts for one recorded wave,
    computed from the wave's OWN typed rows (never from live state): when configured
    minus structurally-dead slots cannot reach the quorum, no re-dispatch of this
    envelope can close it. Dead means a typed window code that is UNDATED, or dated
    with a reset still in the FUTURE (mirror of ``_structural_skip_code``): a row
    whose recorded reset already passed may have healed and never counts.
    ``earliest_reset`` is the earliest parseable reset among the dead rows (empty
    when none names one)."""
    from ouroboros.gateways.claudexor import WINDOW_EXHAUSTED_CODES

    def _is_dead(r: dict) -> bool:
        if str(r.get("failure_code") or "") not in WINDOW_EXHAUSTED_CODES:
            return False
        reset = str(r.get("reset_at") or "")
        if not reset:
            return True  # undated typed window code still counts
        instant = parse_deadline_ts(reset)
        return instant is not None and instant > utc_now()

    dead = [r for r in slot_records if _is_dead(r)]
    if not dead or len(slot_records) - len(dead) >= max(1, int(quorum)):
        return {}
    resets = sorted(
        (instant, str(r.get("reset_at") or "")) for r in dead
        if (instant := parse_deadline_ts(str(r.get("reset_at") or ""))) is not None
    )
    return {
        "quorum_unreachable": True,
        "structurally_dead_slots": [str(r.get("slot_id") or "") for r in dead],
        "earliest_reset": resets[0][1] if resets else "",
    }


def plan_slot_fit(slots: list, *, prompt_chars: int, quorum: int) -> tuple[list, list[dict], str]:
    """``(callable_slots, oversize_rows, error)`` for ONE shared packet fanned across
    mixed-window slots — the review organ's calibrated per-slot input caps
    (`review_synthesis.per_slot_input_token_limits`, Capability Evidence windows) against
    the packet's estimated tokens. An excluded slot is a typed `preflight_oversize` row
    (ok=False, $0) so it is REPORTED as not participating; fewer callable slots than the
    review quorum is a loud typed refusal, never a silent absence of review."""
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    # Only api_chat rows are sized: a RETRIEVING (agent_session) row's model id is an opaque
    # harness target, not a provider route (`reviewer_window.reviewer_route(session=True)`), and
    # the review organ's convention (triad: "session rows are not constrained by this pack")
    # is that such a row is never fit-excluded — it retrieves with its own tools.
    api_models = [str(getattr(slot, "model", "") or "") for slot in slots if not slot_retrieves(slot)]
    limits = per_slot_input_token_limits(
        api_models, output_reserve=PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000)
    estimated = max(1, (max(0, int(prompt_chars)) + 3) // 4)  # utils.estimate_tokens on the packet
    callable_slots, oversize = [], []
    for slot in slots:
        cap = int(limits.get(str(getattr(slot, "model", "") or ""), 0) or 0)
        if slot_retrieves(slot) or estimated <= cap:
            callable_slots.append(slot)
            continue
        oversize.append({
            "slot_id": str(getattr(slot, "slot_id", "") or ""), "model": str(getattr(slot, "model", "") or ""),
            "request_model": str(getattr(slot, "model", "") or ""),
            "route": "agent_session" if slot_is_session(slot) else "api_chat",
            "host_file_read_attestation": None, "text": "",
            "error": (f"preflight_oversize: assembled packet ~{estimated:,} estimated tokens exceeds "
                      f"this slot's calibrated input cap {cap:,}"),
            "prompt_ref": {}, "response_ref": {}, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "status": "not_dispatched", "operation_state": "not_dispatched",
        })
    error = ""
    if slots and len(callable_slots) < int(quorum):
        error = (
            "⚠️ PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE: the assembled packet "
            f"(~{estimated:,} estimated tokens) exceeds the calibrated input cap of too many reviewer "
            "slots (" + ", ".join(f"{m}<={int(limits.get(m, 0) or 0):,}" for m in api_models)
            + "), so fewer than the review quorum remain callable and NO reviewer was called. "
            "A constitutional plan carries BIBLE.md and ARCHITECTURE.md in full (W3): configure "
            "reviewer slots with a larger context window, or shrink the declared evidence."
        )
    return callable_slots, oversize, error


def plan_fanout_inputs(
    slots: list, *, resume: Optional[dict], replay_snapshot: Any,
    prompt_chars: int, quorum: int,
) -> dict:
    """Freeze a paid resume's actors, or prepare one fresh health/fit fan-out."""
    if resume is not None:
        dispatched = set(resume.get("dispatched_slot_ids") or [])
        callable_slots = [slot for slot in slots if str(slot.slot_id) in dispatched]
        if {str(slot.slot_id) for slot in callable_slots} != dispatched:
            return {"error": (
                "The prior paid reviewer assignment cannot be reconstructed exactly; "
                "duplicate dispatch is refused."
            )}
        return {
            "callable_slots": callable_slots,
            "health_skip_rows": list(resume.get("frozen_rows") or []),
            "oversize_rows": [], "health_evidence": resume.get("health_evidence") or {},
            "error": "",
        }
    health_evidence = (
        plan_panel_health_snapshot(slots)
        if replay_snapshot is PLAN_NO_SNAPSHOT else replay_snapshot
    )
    live_slots, health_skip_rows = plan_health_skip_rows(slots, health_evidence)
    callable_slots, oversize_rows, fit_error = plan_slot_fit(
        live_slots, prompt_chars=prompt_chars, quorum=quorum)
    return {
        "callable_slots": callable_slots, "health_skip_rows": health_skip_rows,
        "oversize_rows": oversize_rows, "health_evidence": health_evidence,
        "error": fit_error if fit_error and not health_skip_rows else "",
    }
