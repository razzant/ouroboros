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
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.config import review_model_uses_local
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.llm import LLMClient
from ouroboros.tools.registry import ToolContext, active_repo_dir_for
from ouroboros.tools.tool_result import (
    ToolResult,
    _publish_tool_result,
    _published_tool_result,
    _replace_tool_result,
)
from ouroboros.utils import utc_now_iso


PLAN_REVIEW_MAX_TOKENS = 65536
PLAN_REVIEW_EFFORT = "high"
PLAN_REVIEW_SLOT_TIMEOUT_SEC = 560
# Per-slot provenance of what the reviewer read (BIBLE P3, retrieving reviewers):
# an api_chat slot read exactly the host-assembled packet; an agent_session slot
# retrieved with its own tools and the host did not observe what it opened.
HOST_FILE_READ_ASSEMBLED = "host_assembled_packet"
HOST_FILE_READ_UNOBSERVED = "unobserved"

log = logging.getLogger(__name__)


def append_plan_output_note(ctx: ToolContext, text: str, note: str) -> str:
    """Keep native plan metadata bound when compatibility notes append."""
    rendered = text + note
    base = _published_tool_result(ctx, None)
    if isinstance(base, ToolResult) and base.text == text:
        return _publish_tool_result(ctx, _replace_tool_result(base, text=rendered))
    return rendered


VACUOUS_DISPOSITION_NOTE = (
    "\n\nNOTE: an empty review_disposition was ignored in this review-mode call. "
    "Omit the field when submitting a plan; to close REVIEW_REQUIRED, make a separate "
    "call containing a complete review_disposition only."
)

VACUOUS_CLAIMS_NOTE = (
    "\n\nNOTE: spec.acceptance_claims was empty/blank and was treated as absent. "
    "Omit the field unless you can state concrete, checkable claims of what 'done' "
    "means for this plan."
)


def vacuous_review_disposition(value: object) -> bool:
    """True for a schema-shaped but semantically empty disposition: models routinely
    fill an optional object param with an empty default instead of omitting it. An
    empty disposition has no closing power by construction, so it means "absent" —
    never a stale-disposition failure. A populated-but-wrong disposition (non-empty
    fingerprint or items) is NOT vacuous and keeps failing closed in the validator."""
    if not isinstance(value, dict) or set(value) - {"review_fingerprint", "items"}:
        return False
    return not str(value.get("review_fingerprint") or "").strip() and not value.get("items")


def vacuous_acceptance_claims(spec: object) -> bool:
    """True when the raw spec CARRIES an acceptance_claims key that normalizes to
    absent (None / [] / blank strings / claim-less objects) — the caller appends
    ``VACUOUS_CLAIMS_NOTE`` so the treatment is disclosed, never an error (the
    v6.65.1/.2 lesson, carried onto the spec envelope)."""
    if not isinstance(spec, dict) or "acceptance_claims" not in spec:
        return False
    value = spec.get("acceptance_claims")
    if value is None:
        return True
    if not isinstance(value, list):
        return False  # shape errors surface through spec normalization instead

    def _claim_text(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("claim") or "")
        return ""

    return not any(_claim_text(item).strip() for item in value)


def apply_plan_compat_notes(
    ctx: ToolContext, result: object, *, vacuous_disposition: bool, spec: object,
) -> object:
    """Append the wrapper's compatibility disclosures to a finished plan answer
    while keeping the native plan metadata bound to the final text (D02)."""
    if isinstance(result, str) and vacuous_disposition:
        result = append_plan_output_note(ctx, result, VACUOUS_DISPOSITION_NOTE)
    if isinstance(result, str) and vacuous_acceptance_claims(spec):
        result = append_plan_output_note(ctx, result, VACUOUS_CLAIMS_NOTE)
    return result


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


def plan_review_cycles_exhausted(
    ctx: ToolContext, state: dict, state_root: pathlib.Path, task_id: str, *,
    cap: int, cycles_paid: int, enforcement: str, reminder: str,
    request_fingerprint: str = "",
) -> str:
    """The typed cap result (D10/D27): no panel, the current wave stays open, the typed
    event fires; blocking exits are owner unstick or a blocked_with_evidence terminal."""
    from ouroboros.review_cycles import emit_review_cycles_exhausted
    from ouroboros.task_results import (
        current_plan_review_wave,
        mark_plan_review_cycles_exhausted,
        record_plan_review_attempt,
    )
    from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX

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
    if fingerprint and not (current or {}).get("closed"):
        try:
            current = mark_plan_review_cycles_exhausted(state_root, task_id, fingerprint=fingerprint) or current
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    if request_fingerprint:
        try:
            record_plan_review_attempt(
                state_root, task_id, fingerprint=request_fingerprint, status="cycles_exhausted",
                reason=f"{cycles_paid}/{cap} paid plan-review cycles spent")
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
        return publish_rendered_wave(
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
    return publish_plan_review_projection(
        ctx, {"aggregate_signal": "REVISE_PLAN", "closed": False}, text)


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
    """The configured commit-triad rows as plan-review ``ReviewSlot`` objects.

    Both kinds ride: an ``api_chat`` row is one in-process call over the lean
    packet; an ``agent_session`` row is a delegated retrieving reviewer
    (``session_target``/``session_profile`` carried per row). Effort is the row's
    own when configured, else ``PLAN_REVIEW_EFFORT``. Slot ids are the rows' own
    (structured: owner-assigned; legacy: ``slot_N`` from the one mint).
    """
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import load_reviewer_slot_config

    return [
        ReviewSlot(
            slot_id=row.slot_id,
            model=row.target_id,
            effort=row.effort or PLAN_REVIEW_EFFORT,
            timeout_sec=PLAN_REVIEW_SLOT_TIMEOUT_SEC,
            max_tokens=PLAN_REVIEW_MAX_TOKENS,
            temperature=0.2,
            role_hint="plan reviewer",
            use_local=review_model_uses_local(row.target_id),
            route=ReviewRouteKind.AGENT_SESSION if row.is_session else ReviewRouteKind.API_CHAT,
            session_target=row.session_target,
            session_profile=row.profile_id,
        )
        for row in load_reviewer_slot_config().triad
    ]


def slot_is_session(slot: Any) -> bool:
    from ouroboros.review_execution import ReviewRouteKind

    return getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION


async def run_plan_review_slots(
    ctx: ToolContext,
    slots: list,
    *,
    system_prompt: str,
    user_content: str,
    session_task: str = "",
    session_root: str = "",
    output_contract: str = "",
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
        task_id=str(getattr(ctx, "task_id", "") or "plan_review"),
        call_type="plan_review",
        max_tokens=PLAN_REVIEW_MAX_TOKENS,
        temperature=0.2,
        no_proxy=True,
        session_task=session_task,
        session_root=session_root,
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
                                              "status": "error", "error": "no actor record"}, slot))
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
    return {
        # Identity is CARRIED, never re-derived: the row keeps the slot_id the
        # substrate ran, so duplicate-model plan rows stay distinguishable.
        "slot_id": str(actor.get("slot_id") or getattr(slot, "slot_id", "") or ""),
        "model": str(usage.get("resolved_model") or actor.get("model") or getattr(slot, "model", "") or ""),
        "request_model": str(getattr(slot, "model", "") or actor.get("model") or ""),
        "route": "agent_session" if session else "api_chat",
        "host_file_read_attestation": HOST_FILE_READ_UNOBSERVED if session else HOST_FILE_READ_ASSEMBLED,
        "text": text,
        "error": error or None,
        "prompt_ref": actor.get("prompt_ref") or {},
        "response_ref": actor.get("response_ref") or {},
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "cost": float(usage["cost"]) if usage.get("cost") is not None else None,
        # B1 typed failure facts, forwarded VERBATIM from the actor record (empty on
        # success and on pre-typed engines) so downstream never regresses to matching
        # the error prose for the code or the reset instant.
        **_typed_facts_from(actor, lambda source, key: source.get(key)),
        "capability_delta": usage.get("capability_delta") or [],
    }


def plan_row_typed_facts(row: Dict[str, Any]) -> Dict[str, Any]:
    """The typed failure facts a wave actor record carries forward from a plan row.

    One projection for the engine's wave records (``plan_review.py`` sits at its
    size pin, so the logic lives here). Every fact defaults to honest absence —
    rows from pre-B1 engines and typed-fact-free rows (``plan_slot_fit``'s
    preflight_oversize rows) behave exactly as before."""
    return {
        **_typed_facts_from(row, lambda source, key: source.get(key)),
        "capability_delta": row.get("capability_delta") or [],
    }


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
    every other reason (route_status_*, transient daemon states, unknown) dispatch —
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
    api_models = [str(getattr(slot, "model", "") or "") for slot in slots if not slot_is_session(slot)]
    limits = per_slot_input_token_limits(
        api_models, output_reserve=PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000)
    estimated = max(1, (max(0, int(prompt_chars)) + 3) // 4)  # utils.estimate_tokens on the packet
    callable_slots, oversize = [], []
    for slot in slots:
        cap = int(limits.get(str(getattr(slot, "model", "") or ""), 0) or 0)
        if slot_is_session(slot) or estimated <= cap:
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
