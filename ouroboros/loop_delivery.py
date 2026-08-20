"""Delivery candidates and delivery control: child-result dispositions, the
delivery evidence state, acceptance bindings, candidate publish/replace/degrade,
the delivery-control prompt cycle, the subagent handoff and the no-tool final
answer. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import json
import hashlib
import queue
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ouroboros.config import get_context_mode
from ouroboros.outcomes import reviewable_effect_projection
from ouroboros.tools.registry import ToolRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_round_limits import _RoundLimitContext


# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
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


@dataclass
class DeliveryCandidate:
    """Loop-local complete answer retained across service/finalization rounds."""

    full_text: str
    content_sha256: str
    revision: int
    evidence_revision: int
    evidence_fingerprint: str
    acceptance_binding: Dict[str, Any]
    finalization_control: str = "candidate"
    repair_attempted: bool = False
    degraded: bool = False
    degraded_reason: str = ""


def _swarm_handoff_attempt(ctx: Any) -> Dict[str, Any]:
    attempt = getattr(ctx, "_swarm_handoff_attempt", None)
    return dict(attempt) if isinstance(attempt, dict) else {}


def _compute_subagent_handoff(tools: Any, drive_root: Any, task_id: str, content: Any) -> str:
    """C3.4 pre-finalization child absorption: build the bounded subagent-handoff
    reminder when a finished child's status/result changed since the last refresh, or
    a nonterminal child is unacknowledged in the final text. Returns "" when there is
    nothing to inject. Scans the SAME status root get_task_result uses
    (budget_drive_root, not the forked drive_root — else nested grandchildren in
    forked child drives are missed). Never raises."""
    if drive_root is None or not task_id:
        return ""
    try:
        from ouroboros.task_status import FINAL_STATUSES, format_subagent_absorption_message

        metadata = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
        status_drive_root = pathlib.Path(
            str(metadata.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
            or drive_root
        )
        children = _loop()._load_direct_child_results(
            status_drive_root,
            task_id,
            str(metadata.get("root_task_id") or task_id),
        )
        # Exact-hash dispositions suppress the unchanged result only. If status,
        # result, trace, or artifact identity changes, the disposition becomes stale
        # and this reminder automatically re-opens without parsing prose.
        children = [
            child for child in children
            if _loop()._child_disposition_state(child) not in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }
        ]
        from ouroboros.tools.join_ledger import _child_result_sha256

        signature = "|".join(
            f"{child.get('task_id') or child.get('id')}:{_child_result_sha256(child)}"
            for child in children
        )
        previous = getattr(tools._ctx, "_subagent_handoff_signature", "")
        nonterminal_children = [
            child for child in children
            if str(child.get("status") or "").strip().lower() not in FINAL_STATUSES
        ]
        # P5: the reminder is suppressed ONLY by structured signals — a child
        # discarded/cancelled (filtered above) or absorbed (unchanged
        # signature). NEVER by parsing final PROSE for status words. Fires once
        # per CHANGE, not every round; if the agent still finalizes with
        # unhandled children, the no-tool / forced finalization paths append a
        # loud orphan note via _forced_orphan_note (P1).
        _ = nonterminal_children  # (kept for readability; trigger is change-based)
        if children and signature and signature != previous:
            tools._ctx._subagent_handoff_signature = signature
            tools._ctx._child_absorption_reminded = False
            _absorb_budget = 160_000 if str(get_context_mode()).lower() == "max" else 60_000
            return format_subagent_absorption_message(
                children, parent_task_id=task_id, budget_chars=_absorb_budget,
            )
    except Exception:
        log.debug("Failed to build subagent handoff reminder", exc_info=True)
    return ""


def _delivery_evidence_state(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> tuple[int, str]:
    """Fingerprint only evidence that can invalidate a complete answer."""

    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.join_ledger import _child_result_sha256

    owner_directives = getattr(tools._ctx, "_owner_directives", [])
    owner_directives = owner_directives if isinstance(owner_directives, list) else []
    children = []
    for child in _loop()._direct_child_results(ctx):
        children.append({
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""),
            "sha256": _child_result_sha256(child),
            "disposition": _loop()._child_disposition_state(child),
        })
    receipt_root = pathlib.Path(
        str(getattr(tools._ctx, "drive_root", "") or ctx.drive_root or ctx.status_drive_root or ctx.drive_logs.parent)
    )
    evidence = {
        "owner_directives": owner_directives,
        "tool_effects": reviewable_effect_projection(llm_trace),
        # The typed plan-review control is not a filesystem effect, but it
        # changes whether a pre-plan answer is grounded.
        "plan_review_receipts": [
            {
                "index": index,
                "outcome": call.get("plan_review_outcome"),
                "closed": call.get("plan_review_closed"),
                "result": call.get("result"),
            }
            for index, call in enumerate(llm_trace.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("plan_review_outcome")
        ],
        "children": children,
        "verification_receipts": read_verification_receipts(receipt_root, ctx.task_id),
        # Task-scoped service teardown can register declared outputs or surface an
        # output-finalization failure.  Those facts are produced outside an ordinary
        # tool call, so bind their stable projection explicitly; otherwise a host
        # acceptance panel could review the pre-teardown state.
        "service_finalization": _loop()._service_finalization_evidence(llm_trace),
    }
    fingerprint = hashlib.sha256(json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()
    previous = str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or "")
    revision = int(getattr(tools._ctx, "_delivery_evidence_revision", 0) or 0)
    if fingerprint != previous:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if (
            isinstance(candidate, _loop().DeliveryCandidate)
            and bool(candidate.evidence_fingerprint)
            and candidate.evidence_fingerprint != fingerprint
        ):
            _loop()._supersede_delivery_acceptance_binding(
                tools,
                llm_trace,
                candidate,
                reason="delivery_evidence_changed_after_host_acceptance",
            )
        revision += 1
        tools._ctx._delivery_evidence_fingerprint = fingerprint
        tools._ctx._delivery_evidence_revision = revision
    return revision, fingerprint


def _unaccepted_delivery_binding(
    tools: ToolRegistry,
    candidate_hash: str,
) -> Dict[str, Any]:
    fence_value = str(
        getattr(tools._ctx, "_task_acceptance_sealed_fence_token", "")
        or "unsealed"
    )
    return {
        "candidate_sha256": candidate_hash,
        "evidence_revision": int(getattr(tools._ctx, "_delivery_evidence_revision", 0) or 0),
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "panel_id": "",
        "binding_hash": "",
        "fence_hash": hashlib.sha256(fence_value.encode("utf-8")).hexdigest(),
    }


def _delivery_acceptance_binding(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    candidate_hash: str,
) -> Dict[str, Any]:
    """Refresh a candidate from one exact, complete, active host-root verdict."""

    binding = _unaccepted_delivery_binding(tools, candidate_hash)
    review_decision = llm_trace.get("review_decision") if isinstance(llm_trace.get("review_decision"), dict) else {}
    expected_panel = str(review_decision.get("panel_id") or "")
    expected_binding = str(review_decision.get("binding_hash") or "")
    # Candidate text alone is not a review identity: the same full answer can be
    # regenerated after tool/child/verification evidence changes.  Refresh host
    # authority only from the panel the current acceptance pass explicitly names;
    # an older exact-text run must never be rediscovered by a hash-only scan.
    if not expected_panel or not expected_binding:
        return binding
    for raw_run in reversed(llm_trace.get("review_runs") or []):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("authority") != "host_root" or raw_run.get("superseded_by_revision"):
            continue
        run_candidate = str(
            raw_run.get("candidate_hash") or raw_run.get("candidate_sha256") or ""
        )
        if run_candidate != candidate_hash:
            continue
        run_panel = str(raw_run.get("panel_id") or "")
        run_binding = str(raw_run.get("binding_hash") or "")
        if not run_panel or not run_binding:
            continue
        if run_panel != expected_panel:
            continue
        if run_binding != expected_binding:
            continue
        verdict = str(
            raw_run.get("aggregate_signal") or raw_run.get("semantic_verdict") or ""
        ).strip().lower()
        if not verdict:
            continue
        binding.update({
            "acceptance_status": verdict,
            "authoritative": True,
            "panel_id": run_panel,
            "binding_hash": run_binding,
            "fence_hash": str(raw_run.get("fence_hash") or binding["fence_hash"]),
            "review_evidence_revision": str(raw_run.get("evidence_revision") or ""),
        })
        break
    return binding


def _publish_delivery_candidate(
    tools: ToolRegistry,
    candidate: DeliveryCandidate,
    llm_trace: Dict[str, Any],
) -> None:
    """Publish hashes/control state only; the complete text remains loop-local."""

    current_fp = str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or "")
    llm_trace["delivery_candidate"] = {
        "content_sha256": candidate.content_sha256,
        "revision": candidate.revision,
        "evidence_revision": candidate.evidence_revision,
        "evidence_fingerprint": candidate.evidence_fingerprint,
        "evidence_current": candidate.evidence_fingerprint == current_fp,
        "acceptance_binding": dict(candidate.acceptance_binding),
        "finalization_control": candidate.finalization_control,
        "degraded": candidate.degraded,
        "degraded_reason": candidate.degraded_reason,
    }


def _replace_delivery_candidate(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    *,
    control: str,
) -> DeliveryCandidate:
    previous_candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(previous_candidate, _loop().DeliveryCandidate):
        _loop()._supersede_delivery_acceptance_binding(
            tools,
            llm_trace,
            previous_candidate,
            reason="delivery_candidate_replaced",
        )
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    revision = int(getattr(tools._ctx, "_delivery_candidate_revision", 0) or 0) + 1
    tools._ctx._delivery_candidate_revision = revision
    candidate = _loop().DeliveryCandidate(
        full_text=full_text,
        content_sha256=content_hash,
        revision=revision,
        evidence_revision=evidence_revision,
        evidence_fingerprint=evidence_fingerprint,
        acceptance_binding=_unaccepted_delivery_binding(tools, content_hash),
        finalization_control=control,
    )
    tools._ctx._delivery_candidate = candidate
    tools._ctx._delivery_control_required = False
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    return candidate


def _ensure_explicit_acceptance_binding(candidate: DeliveryCandidate) -> None:
    """Keep an exact historical binding, or state explicitly that none exists."""

    binding = dict(candidate.acceptance_binding or {})
    if binding.get("authoritative") is not True:
        binding.update({
            "acceptance_status": "unaccepted",
            "authoritative": False,
            "panel_id": "",
            "binding_hash": "",
        })
        binding.pop("review_evidence_revision", None)
    candidate.acceptance_binding = binding


def _forced_unaccepted_binding(
    tools: ToolRegistry,
    candidate: DeliveryCandidate,
    reason_code: str,
) -> Dict[str, Any]:
    """Bind a newly generated forced answer without borrowing an older verdict."""

    binding = _unaccepted_delivery_binding(tools, candidate.content_sha256)
    binding.update({
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "degraded": True,
        "degraded_reason": reason_code,
        "panel_id": "",
        "binding_hash": "",
    })
    binding.pop("review_evidence_revision", None)
    return binding


def _live_delivery_candidate(ctx: _RoundLimitContext) -> Optional[DeliveryCandidate]:
    tools = getattr(ctx, "tools", None)
    if tools is not None:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if isinstance(candidate, _loop().DeliveryCandidate):
            return candidate
    candidate = getattr(ctx, "delivery_candidate", None)
    return candidate if isinstance(candidate, _loop().DeliveryCandidate) else None


def _current_delivery_candidate(
    ctx: Optional[_RoundLimitContext],
    llm_trace: Dict[str, Any],
) -> Optional[DeliveryCandidate]:
    """Return a retained answer only after checking live answer-invalidating evidence."""

    if ctx is None or getattr(ctx, "tools", None) is None:
        return None
    candidate = _loop()._live_delivery_candidate(ctx)
    if candidate is None:
        return None
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(
        ctx.tools, ctx, llm_trace,
    )
    if (
        candidate.evidence_revision != evidence_revision
        or candidate.evidence_fingerprint != evidence_fingerprint
    ):
        return None
    return candidate


def _degrade_retained_delivery_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    candidate: DeliveryCandidate,
    *,
    control: str,
    reason_code: str,
) -> DeliveryCandidate:
    """Publish a current unchanged candidate while preserving its exact verdict binding."""

    candidate.degraded = True
    candidate.degraded_reason = reason_code
    candidate.finalization_control = control
    _ensure_explicit_acceptance_binding(candidate)
    tools = getattr(ctx, "tools", None)
    if tools is not None:
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _merge_finalization_trace(
    llm_trace: Dict[str, Any],
    returned_trace: Any,
) -> Dict[str, Any]:
    """Merge a forced-path trace without duplicating the live trace object."""

    if not isinstance(returned_trace, dict) or returned_trace is llm_trace:
        return llm_trace
    for key, value in returned_trace.items():
        if isinstance(value, list) and isinstance(llm_trace.get(key), list):
            for item in value:
                if item not in llm_trace[key]:
                    llm_trace[key].append(item)
        elif isinstance(value, dict) and isinstance(llm_trace.get(key), dict):
            llm_trace[key].update(value)
        else:
            llm_trace[key] = value
    return llm_trace


def _delivery_control_prompt(candidate: DeliveryCandidate, *, keep_allowed: bool) -> str:
    keep_line = (
        "keep is allowed because no answer-invalidating evidence changed."
        if keep_allowed
        else "keep is NOT allowed because owner/tool/child/verification evidence changed."
    )
    return (
        "[DELIVERY_FINALIZATION_CONTROL]\n"
        f"A complete answer candidate (revision {candidate.revision}, sha256 "
        f"{candidate.content_sha256[:12]}) is retained by the loop; do not replace it with a "
        f"service notice. {keep_line}\n"
        "Return exactly one JSON object and no other text:\n"
        '{"delivery_control":"keep"}\n'
        "or\n"
        '{"delivery_control":"replace","full_answer":"<the complete user-facing answer>"}'
    )


def _delivery_replace_required(candidate: DeliveryCandidate) -> bool:
    """Return whether a typed full replacement is mandatory for this control round."""

    return candidate.finalization_control.startswith(
        ("effect_revision_required", "skill_revision_required")
    )


def _delivery_keep_allowed(
    candidate: DeliveryCandidate,
    evidence_revision: int,
    evidence_fingerprint: str,
) -> bool:
    return (
        not _loop()._delivery_replace_required(candidate)
        and candidate.evidence_revision == evidence_revision
        and candidate.evidence_fingerprint == evidence_fingerprint
    )


def _arm_delivery_control(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    control: str = "awaiting_control",
) -> None:
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, _loop().DeliveryCandidate):
        return
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    candidate.finalization_control = control
    candidate.repair_attempted = False
    tools._ctx._delivery_control_required = True
    _loop()._append_or_merge_user_message(
        ctx.messages,
        _delivery_control_prompt(
            candidate,
            keep_allowed=_delivery_keep_allowed(
                candidate, evidence_revision, evidence_fingerprint,
            ),
        ),
    )
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


def _hold_delivery_for_skill_action(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
) -> None:
    """Retain the answer while an unresolved skill lifecycle gate requires action."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, _loop().DeliveryCandidate):
        return
    candidate.finalization_control = "skill_action_or_revision_required"
    candidate.repair_attempted = False
    tools._ctx._delivery_control_required = False
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


def _parse_delivery_control_object(
    raw: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Parse a delivery-control object while rejecting duplicate JSON keys.

    The boolean preserves protocol intent for the repair path when a duplicate
    ``delivery_control`` or ``full_answer`` key made the object invalid.
    """

    duplicate_protocol_key = False

    def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        nonlocal duplicate_protocol_key
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                if key in {"delivery_control", "full_answer"}:
                    duplicate_protocol_key = True
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, duplicate_protocol_key
    if not isinstance(payload, dict):
        return None, False
    return payload, False


def _resolve_delivery_control(
    content: Any,
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> tuple[str, str]:
    """Return ``retry`` or a complete answer text before any existing gate runs."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    required = bool(getattr(tools._ctx, "_delivery_control_required", False))
    if not isinstance(candidate, _loop().DeliveryCandidate):
        return "fresh", _loop()._extract_plain_text_from_content(content)
    raw = _loop()._extract_plain_text_from_content(content).strip()
    parsed, duplicate_protocol_key = _loop()._parse_delivery_control_object(raw)
    # ANY parsed object carrying the protocol key is control intent, regardless of
    # verb/value — an unknown verb is a mangled protocol attempt, never prose (raw
    # JSON leaked to chat). Verb/shape validity is judged below (repair path).
    is_control_intent = duplicate_protocol_key or (
        isinstance(parsed, dict) and "delivery_control" in parsed
    )
    if not required:
        if _loop()._delivery_replace_required(candidate):
            # A writer/skill action cannot silently turn a short acknowledgement
            # into the new complete answer, even if a caller lost the transient
            # required latch. The candidate's typed control state is authoritative.
            required = True
            tools._ctx._delivery_control_required = True
        elif candidate.finalization_control == "skill_action_or_revision_required":
            # Preserve the historical bounded skill gate: an actual tool action
            # or a reconsidered full prose answer may proceed, but a typed keep
            # cannot acknowledge the gate. Do not inject the delivery JSON prompt
            # before the action because it would conflict with the instruction to
            # call the skill lifecycle tool.
            if not is_control_intent:
                return "fresh", _loop()._extract_plain_text_from_content(content)
            candidate.finalization_control = "skill_revision_required"
            required = True
            tools._ctx._delivery_control_required = True
        else:
            # An owner revision starts an ordinary substantive answer round. If
            # the model nevertheless follows the prior typed instruction, honor
            # that control structurally; service/effect/skill rounds are handled
            # by the replace-required branch above.
            if not (
                candidate.finalization_control == "owner_revision_required"
                and is_control_intent
            ):
                return "fresh", _loop()._extract_plain_text_from_content(content)
            tools._ctx._delivery_control_required = True
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    error = "control must be one exact JSON object"
    selected = str(parsed.get("delivery_control") or "") if isinstance(parsed, dict) else ""
    valid = False
    replacement = ""
    if selected == "keep" and set(parsed) == {"delivery_control"}:
        valid = _delivery_keep_allowed(
            candidate, evidence_revision, evidence_fingerprint,
        )
        error = "keep cannot bind changed evidence; send replace with the complete answer"
    elif selected == "replace" and set(parsed) == {"delivery_control", "full_answer"}:
        replacement_value = parsed.get("full_answer")
        if isinstance(replacement_value, str):
            replacement = replacement_value
        valid = isinstance(replacement_value, str) and bool(replacement.strip())
        error = "replace requires a non-empty complete full_answer"

    if valid and selected == "keep":
        tools._ctx._delivery_control_required = False
        candidate.finalization_control = "keep"
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        return "resolved", candidate.full_text
    if valid and selected == "replace":
        updated = _loop()._replace_delivery_candidate(
            tools, ctx, llm_trace, replacement, control="replace",
        )
        return "resolved", updated.full_text

    if not candidate.repair_attempted:
        candidate.repair_attempted = True
        candidate.finalization_control = (
            f"{candidate.finalization_control}_repair_requested"
            if _loop()._delivery_replace_required(candidate)
            else "repair_requested"
        )
        if raw:
            ctx.messages.append({"role": "assistant", "content": raw})
        _loop()._append_or_merge_user_message(
            ctx.messages,
            "[DELIVERY_CONTROL_REPAIR] Invalid finalization control: " + error + ".\n"
            + _delivery_control_prompt(
                candidate,
                keep_allowed=_delivery_keep_allowed(
                    candidate, evidence_revision, evidence_fingerprint,
                ),
            ),
        )
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        return "retry", ""

    tools._ctx._delivery_control_required = False
    candidate.degraded = True
    candidate.degraded_reason = "invalid_delivery_control_after_repair"
    candidate.finalization_control = "degraded_preserve"
    # The control failed, not the retained text. Bind that unchanged text to
    # the evidence the failed control was meant to acknowledge so the stale
    # check cannot reopen another control round. It remains explicitly
    # unaccepted; the ordinary host acceptance gate still judges this exact
    # candidate/evidence pair before publication.
    candidate.evidence_revision = evidence_revision
    candidate.evidence_fingerprint = evidence_fingerprint
    candidate.acceptance_binding = _unaccepted_delivery_binding(
        tools, candidate.content_sha256,
    )
    llm_trace["reasoning_notes"].append(
        "Delivery finalization control remained invalid after one repair; preserved the prior complete answer."
    )
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    return "degraded", candidate.full_text


def _compose_delivery_suffix(full_text: str, suffix: str) -> str:
    """Compose one host-owned suffix into the exact delivered/candidate text."""

    text = str(full_text or "")
    note = str(suffix or "")
    if not note or text.endswith(note):
        return text
    return text + note


def _no_tool_final_answer(
    content: Any,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    tools: ToolRegistry,
    incoming_messages: queue.Queue,
    owner_msg_seen: set,
    emit_progress: Callable[[str], None],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Run the no-tool finalization gates; ``None`` requests another model round."""
    messages = limit_ctx.messages
    control_state, controlled_content = _resolve_delivery_control(
        content, tools, limit_ctx, llm_trace,
    )
    if control_state == "retry":
        return None
    content = controlled_content
    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    if control_state == "fresh" and str(content or "").strip():
        candidate = _loop()._replace_delivery_candidate(
            tools, limit_ctx, llm_trace, str(content), control="candidate",
        )
        content = candidate.full_text
    else:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if isinstance(candidate, _loop().DeliveryCandidate):
            content = candidate.full_text

    if _loop()._enforce_swarm_actions(
        str(content or ""), messages, tools, llm_trace, emit_progress,
    ):
        return None
    handoff_msg = _compute_subagent_handoff(tools, limit_ctx.drive_root, limit_ctx.task_id, content)
    if handoff_msg:
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{handoff_msg}")
        emit_progress("Subagent handoff status refreshed before final response.")
        llm_trace["reasoning_notes"].append("Subagent handoff status refreshed before final response.")
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    absorption_result = _loop()._maybe_enforce_child_absorption_gate(
        tools, limit_ctx, content, messages, emit_progress, llm_trace,
    )
    if absorption_result == "continue":
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    if absorption_result is not None:
        return absorption_result
    skill_finalization_was_injected = bool(
        getattr(tools._ctx, "_skill_finalization_injected", False)
    )
    if _loop()._maybe_inject_finalization_nudges(
        tools, limit_ctx.drive_root, limit_ctx.task_id, llm_trace, content, messages, emit_progress,
    ):
        skill_finalization_injected_now = (
            not skill_finalization_was_injected
            and bool(getattr(tools._ctx, "_skill_finalization_injected", False))
        )
        # Skill finalization is an action gate, not a service notice. Preserve
        # the candidate without adding a conflicting JSON-only instruction: the
        # next round may run the required tool or provide the historically
        # allowed reconsidered full answer, but a typed keep cannot close it.
        if skill_finalization_injected_now:
            _hold_delivery_for_skill_action(tools, llm_trace)
        else:
            _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return None

    # Declared service outputs and teardown failures are acceptance evidence, not
    # postscript cleanup.  Finalize them before the authoritative host panel and,
    # when that changes evidence, require one complete replacement answer bound to
    # the new revision.  The finally-path calls the same idempotent helper as a
    # safety net for forced/error exits.
    service_exit_ctx = _loop()._LoopExitContext(
        tools=tools,
        drive_root=limit_ctx.drive_root,
        task_id=limit_ctx.task_id,
        event_queue=limit_ctx.event_queue,
        drive_logs=limit_ctx.drive_logs,
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=llm_trace,
    )
    if _loop()._finalize_task_services(service_exit_ctx):
        evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(
            tools, limit_ctx, llm_trace,
        )
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if (
            isinstance(candidate, _loop().DeliveryCandidate)
            and (
                candidate.evidence_revision != evidence_revision
                or candidate.evidence_fingerprint != evidence_fingerprint
            )
        ):
            if content and str(content).strip():
                messages.append({"role": "assistant", "content": str(content)})
            llm_trace["reasoning_notes"].append(
                "Task services were finalized before acceptance; the complete answer must bind the resulting evidence."
            )
            _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
            return None

    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    plan_suffix = _loop()._force_plan_disclosure(tools._ctx, llm_trace)
    orphan_suffix = _loop()._forced_orphan_note(limit_ctx, include_terminal=False)
    normal_suffix = plan_suffix + orphan_suffix
    composed_content = _loop()._compose_delivery_suffix(str(content or ""), normal_suffix)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if composed_content and (
        not isinstance(candidate, _loop().DeliveryCandidate)
        or candidate.full_text != composed_content
    ):
        candidate = _loop()._replace_delivery_candidate(
            tools,
            limit_ctx,
            llm_trace,
            composed_content,
            control="host_suffix" if normal_suffix else "candidate",
        )
    if isinstance(candidate, _loop().DeliveryCandidate):
        if orphan_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "host_child_status_suffix"
            _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        elif plan_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "plan_review_advisory"
            _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text

    tools._ctx._acceptance_loop_rails = {
        "round_idx": limit_ctx.round_idx,
        "max_rounds": limit_ctx.max_rounds,
        "task_cost_usd": limit_ctx.accumulated_usage.get("cost"),
    }
    # v6.78.0 (owner Q20/Q22): mirror the host-attested native-retrieval fact into the
    # trace so `build_task_acceptance_evidence` can show the reviewer whether the answer
    # was grounded in fetched pages. Reviewer-side only — the agent never sees it (it
    # receives the improvement capsule, not the evidence packet).
    _retrieval = limit_ctx.accumulated_usage.get("retrieval")
    if isinstance(_retrieval, dict) and _retrieval:
        llm_trace["retrieval"] = dict(_retrieval)
    if _loop()._run_task_acceptance_review_once(
        tools=tools,
        content=content or "",
        task_id=limit_ctx.task_id,
        task_type=limit_ctx.task_type,
        llm_trace=llm_trace,
        drive_root=limit_ctx.drive_root,
        messages=messages,
        emit_progress=emit_progress,
    ):
        # v6.71.1: an acceptance improvement pass is an ORDINARY substantive
        # answer round — do NOT arm delivery-control here: layering "return
        # exactly one JSON object" on top of OPEN OBLIGATIONS and the self-
        # check froze the model into resubmitting the same answer. The next
        # free-form answer re-enters the acceptance panel, so blocking is not
        # weakened; other lanes still arm where JSON keep/replace is needed.
        return None
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, _loop().DeliveryCandidate):
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)

    # Close delivery under the same lock as routing, then drain once. A follow-up
    # either forces another round or is rejected after the fence, never stranded.
    admission_lock = getattr(tools._ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(tools._ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        before_directives = len(getattr(tools._ctx, "_owner_directives", []) or [])
        acceptance_was_terminal = bool(
            getattr(tools._ctx, "_task_acceptance_reviewed", False)
            or getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None)
        )
        provisional_assistant = {"role": "assistant", "content": content} if content else None
        if provisional_assistant is not None:
            messages.append(provisional_assistant)
        with admission_lock:
            admission_agent._accepting_owner_messages = False
            post_controls = _loop()._drain_incoming_messages(
                messages, incoming_messages, limit_ctx.drive_root, limit_ctx.task_id,
                limit_ctx.event_queue, owner_msg_seen, owner_ctx=tools._ctx,
            )
        if len(getattr(tools._ctx, "_owner_directives", []) or []) > before_directives:
            with admission_lock:
                if acceptance_was_terminal:
                    _loop()._supersede_task_acceptance_for_owner_followup(
                        tools._ctx, llm_trace, admission_locked=True,
                    )
                if (
                    getattr(admission_agent, "_busy", False)
                    and str(getattr(admission_agent, "_current_task_id", "") or "") == limit_ctx.task_id
                ):
                    admission_agent._accepting_owner_messages = True
            if acceptance_was_terminal:
                emit_progress(
                    "Task acceptance review superseded: an owner follow-up arrived before finalization."
                )
            # An owner directive is a substantive revision request, not a service
            # notification. The next complete response creates a fresh candidate.
            tools._ctx._delivery_control_required = False
            if isinstance(candidate, _loop().DeliveryCandidate):
                candidate.finalization_control = "owner_revision_required"
                _loop()._delivery_evidence_state(tools, limit_ctx, llm_trace)
                _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
            return None
        if provisional_assistant is not None and messages[-1] is provisional_assistant:
            messages.pop()
        if post_controls.get("finalize_now"):
            text, usage, forced_trace = _loop()._handle_forced_finalization(
                limit_ctx, str(post_controls.get("finalize_now") or "deadline"),
            )
            _loop()._merge_finalization_trace(llm_trace, forced_trace)
            return text, usage, llm_trace
    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(
        tools, limit_ctx, llm_trace,
    )
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if (
        isinstance(candidate, _loop().DeliveryCandidate)
        and (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        )
    ):
        acceptance_was_terminal = bool(
            getattr(tools._ctx, "_task_acceptance_reviewed", False)
            or getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None)
        )
        if acceptance_was_terminal:
            decision = (
                llm_trace.get("review_decision")
                if isinstance(llm_trace.get("review_decision"), dict)
                else {}
            )
            expected_panel = str(decision.get("panel_id") or "")
            expected_binding = str(decision.get("binding_hash") or "")
            active_run = next(
                (
                    run
                    for run in reversed(llm_trace.get("review_runs") or [])
                    if isinstance(run, dict)
                    and run.get("authority") == "host_root"
                    and not run.get("superseded_by_revision")
                    and str(run.get("panel_id") or "") == expected_panel
                    and str(run.get("binding_hash") or "") == expected_binding
                ),
                None,
            )
            _loop()._supersede_task_acceptance_for_evidence_change(
                tools._ctx,
                llm_trace,
                active_run,
                "delivery_evidence_changed_after_host_acceptance",
                messages,
                emit_progress,
            )
        if candidate.full_text:
            messages.append({"role": "assistant", "content": candidate.full_text})
        llm_trace["reasoning_notes"].append(
            "Delivery evidence changed after host acceptance; a complete replacement answer is required."
        )
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    if isinstance(candidate, _loop().DeliveryCandidate):
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text
    return _loop()._handle_text_response(
        str(content or ""),
        llm_trace,
        limit_ctx.accumulated_usage,
    )
