"""Delivery candidates and delivery control: child-result dispositions, the
delivery evidence state, acceptance bindings, candidate publish/replace/degrade,
the delivery-control prompt cycle, the subagent handoff and the no-tool final
answer. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import queue

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros.config import get_context_mode
from ouroboros.outcomes import reviewable_effect_projection
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import sanitize_tool_result_for_log


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_round_limits import _RoundLimitContext


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
    model_text: str = ""
    # Sticky loop-local provenance that this lineage has SEEN a host-issued
    # delivery-control episode (#447/issue-449): every replacement inherits it,
    # including ordinary acceptance improvements, so a later control-shaped
    # answer under a lost latch is still read as protocol rather than prose.
    control_episode_seen: bool = False


# Action-gate holds: a gate closable ONLY by a tool call (skill lifecycle
# action, child-result disposition) retains the candidate WITHOUT arming the
# JSON-only control instruction — the instruction would conflict with the
# required tool call. Gates closable by a reconsidered answer arm normally.
_SKILL_ACTION_HOLD_CONTROL = "skill_action_or_revision_required"
_CHILD_ABSORPTION_HOLD_CONTROL = "child_absorption_or_revision_required"
_DELIVERY_HOLD_CONTROLS = frozenset({
    _SKILL_ACTION_HOLD_CONTROL,
    _CHILD_ABSORPTION_HOLD_CONTROL,
})


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
        # Exact-hash dispositions suppress the unchanged result only: if
        # status, result, trace, or artifact identity changes, the disposition
        # goes stale and this reminder re-opens without parsing prose.
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
        # P5: the reminder is suppressed ONLY by structured signals — a
        # child discarded/cancelled (filtered above) or absorbed (unchanged
        # signature), NEVER by parsing final PROSE; fires once per CHANGE, and
        # finalizing with unhandled children appends a loud orphan note (P1).
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

    from ouroboros.outcomes import read_context_verification_receipts
    from ouroboros.tools.join_ledger import _child_result_sha256

    owner_directives = getattr(tools._ctx, "_owner_directives", [])
    owner_directives = owner_directives if isinstance(owner_directives, list) else []
    children = [
        {
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""),
            "sha256": _child_result_sha256(child),
            "disposition": _loop()._child_disposition_state(child),
        }
        for child in _loop()._direct_child_results(ctx)
    ]
    receipt_root = pathlib.Path(
        str(
            getattr(tools._ctx, "drive_root", "")
            or ctx.drive_root
            or ctx.status_drive_root
            or ctx.drive_logs.parent
        )
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
        "verification_receipts": read_context_verification_receipts(
            tools._ctx, ctx.task_id, fallback_root=receipt_root,
        ),
        # Task-scoped service teardown can register declared outputs or
        # surface an output-finalization failure. Those facts arise outside an
        # ordinary tool call, so bind their stable projection explicitly; else
        # a host acceptance panel could review the pre-teardown state.
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
    # Candidate text alone is not a review identity: the same full answer
    # can be regenerated after tool/child/verification evidence changes.
    # Refresh host authority only from the panel this pass names; an older
    # exact-text run must never be rediscovered by hash-only scan.
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
        "control_episode_seen": candidate.control_episode_seen,
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
    model_text: Optional[str] = None,
) -> DeliveryCandidate:
    full_text = sanitize_tool_result_for_log(full_text)
    model_text = sanitize_tool_result_for_log(
        full_text if model_text is None else model_text
    )
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
        model_text=model_text,
        control_episode_seen=bool(
            getattr(previous_candidate, "control_episode_seen", False)
        ),
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
        "You may continue using tools whenever more work is needed. This instruction applies "
        "only to your final response with no tool calls. When ready to finalize, return "
        "exactly one JSON object and no other text:\n"
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
    candidate.control_episode_seen = True
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


def _hold_delivery_for_skill_action(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    *,
    control: str = _SKILL_ACTION_HOLD_CONTROL,
) -> None:
    """Retain the answer while an unresolved action gate requires a tool call.

    ``control`` names the open gate; it must stay within
    ``_DELIVERY_HOLD_CONTROLS`` so both hold readers recognize the state.
    """

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, _loop().DeliveryCandidate):
        return
    candidate.finalization_control = control
    candidate.repair_attempted = False
    tools._ctx._delivery_control_required = False
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


class _ParsedObject(dict):
    """A parsed JSON object that remembers which of its keys were duplicated."""

    duplicate_keys: set
    has_duplicate_keys: bool


def _parse_delivery_control_object(
    raw: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Parse a body while rejecting duplicates in a control envelope.

    The boolean preserves top-level protocol intent when duplicate keys made a
    recognizable control envelope invalid. Per-object metadata supports the
    stronger armed/action rails; an aggregate duplicate marker restores the
    forced armed malformed-body rule without widening history-gated parsing.
    """

    has_duplicate_keys = False

    def _unique_object(pairs: List[Tuple[str, Any]]) -> "_ParsedObject":
        nonlocal has_duplicate_keys
        result = _ParsedObject()
        result.duplicate_keys = set()
        for key, value in pairs:
            if key in result:
                result.duplicate_keys.add(key)
                has_duplicate_keys = True
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        # RecursionError: a degenerate deeply-nested blob (repetition-loop
        # model output) must classify as not-a-control, not crash the round.
        return None, False
    if not isinstance(payload, dict):
        return None, False
    payload.has_duplicate_keys = has_duplicate_keys
    duplicate_keys = getattr(payload, "duplicate_keys", set())
    if duplicate_keys:
        if "delivery_control" in payload:
            return None, True
        # Keep per-object duplicate metadata for stronger control rails while
        # stopping _parse_delivery_control_body from rescanning this whole body.
        return payload, False
    return payload if has_duplicate_keys else dict(payload), False


def _classify_parsed_delivery_control(
    parsed: Optional[Dict[str, Any]],
    duplicate_protocol_key: bool,
    embedded: bool,
) -> Tuple[str, str, str]:
    """Return ``(kind, replacement, error)`` for a parsed control body."""

    exact_error = "control must be one exact JSON object"
    if embedded:
        # A trailing prose-embedded object is a protocol ATTEMPT, never a valid
        # control: honoring it would leak the raw object or drop the prose half.
        return "embedded", "", exact_error
    if duplicate_protocol_key:
        return "invalid", "", exact_error
    if (
        isinstance(parsed, dict)
        and "full_answer" in getattr(parsed, "duplicate_keys", set())
    ):
        # Without a top-level verb this is not historical protocol, but an
        # already armed/action rail must retain the base malformed-control rule.
        return "rail_invalid", "", exact_error
    if not isinstance(parsed, dict) or "delivery_control" not in parsed:
        return "none", "", exact_error
    selected = str(parsed.get("delivery_control") or "")
    if selected == "keep" and set(parsed) == {"delivery_control"}:
        return "keep", "", ""
    if selected == "replace" and set(parsed) == {"delivery_control", "full_answer"}:
        replacement = parsed.get("full_answer")
        if isinstance(replacement, str) and replacement.strip():
            return "replace", replacement, ""
        return "invalid", "", "replace requires a non-empty complete full_answer"
    return "invalid", "", exact_error


def _resolve_forced_delivery_control_body(
    raw: str,
    candidate: Optional[DeliveryCandidate],
    *,
    armed: bool,
) -> Tuple[str, bool, bool, bool, bool]:
    """Return text plus retained/degraded/consumed/replaced facts."""

    if not isinstance(candidate, _loop().DeliveryCandidate):
        candidate = None
    parsed, duplicate_protocol_key, embedded_protocol = _parse_delivery_control_body(raw)
    control_kind, replacement, _error = _classify_parsed_delivery_control(
        parsed, duplicate_protocol_key, embedded_protocol,
    )
    historical = bool(
        not armed
        and candidate is not None
        and candidate.control_episode_seen
        and control_kind in {"keep", "replace", "invalid"}
    )
    if not armed and not historical:
        return raw, False, False, False, False
    if control_kind == "replace":
        return replacement, False, False, True, True
    if control_kind == "keep" and candidate is not None:
        return candidate.full_text, True, False, True, False
    if historical:
        return candidate.full_text, True, False, True, False
    from ouroboros.observability import strip_protocol_fence

    protocol_intent = (
        control_kind != "none"
        or (parsed is None and strip_protocol_fence(raw).startswith("{"))
        or bool(getattr(parsed, "has_duplicate_keys", False))
    )
    if not protocol_intent:
        # Ordinary prose under an armed latch stands (a control object quoted
        # MID-prose is the disclosed residual).
        return raw, False, False, True, False
    retained = candidate is not None
    return candidate.full_text if retained else "", retained, True, True, False


def _parse_delivery_control_body(
    raw: str,
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Normalize a response body and locate its delivery-control object.

    Returns ``(parsed, duplicate_protocol_key, embedded)``. Normalization
    strips one whole-body markdown fence (shared with
    ``observability._is_delivery_control_payload``). ``embedded`` is True only
    when the protocol object sits as a balanced trailing JSON object carrying
    the ``delivery_control`` key at the very END of surrounding prose — a
    protocol attempt mixed with text, never a valid control. A control object
    quoted MID-prose is NOT matched and stays prose (disclosed residual)."""
    from ouroboros.observability import strip_protocol_fence

    body = strip_protocol_fence(raw)
    parsed, duplicate_protocol_key = _parse_delivery_control_object(body)
    if duplicate_protocol_key or isinstance(parsed, dict):
        return parsed, duplicate_protocol_key, False
    # Trailing scan: ONE O(n) string-aware pass over the body (fenced and
    # double-fenced tails peeled, duplicate keys flagged, RecursionError
    # degraded, bounded line-anchor retries after an unbalanced prose brace
    # or quote). The extractor is key-agnostic; the protocol judgment stays
    # HERE: only a trailing object carrying `delivery_control` at its top
    # level (or a duplicated protocol key) is an embedded protocol attempt.
    from ouroboros.utils import extract_trailing_json_object

    _prefix, tail_parsed, tail_duplicate = extract_trailing_json_object(
        body, duplicate_flag_keys=("delivery_control", "full_answer"),
    )
    if tail_duplicate:
        return None, True, True
    if isinstance(tail_parsed, dict) and "delivery_control" in tail_parsed:
        return tail_parsed, False, True
    return None, False, False


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
    parsed, duplicate_protocol_key, embedded_protocol = _parse_delivery_control_body(raw)
    control_kind, replacement, error = _classify_parsed_delivery_control(
        parsed, duplicate_protocol_key, embedded_protocol,
    )
    # ANY parsed object carrying the protocol key is control intent, whatever
    # the verb or placement — a mangled protocol attempt is never prose (raw
    # JSON leaked to chat); validity judged below.
    is_control_intent = control_kind != "none"
    historical_control = False
    if not required:
        if _loop()._delivery_replace_required(candidate):
            # A writer/skill action cannot silently turn a short acknowledgement
            # into the new complete answer, even if a caller lost the transient
            # required latch. The candidate's typed control state is authoritative.
            required = True
            tools._ctx._delivery_control_required = True
        elif candidate.finalization_control in _DELIVERY_HOLD_CONTROLS:
            # Bounded action gates (skill lifecycle, child absorption): a tool
            # action or a reconsidered full prose answer may proceed; a typed keep
            # cannot acknowledge the gate; a typed control attempt escalates to
            # the ONE replace-required literal for BOTH holds.
            if not is_control_intent:
                return "fresh", _loop()._extract_plain_text_from_content(content)
            candidate.finalization_control = "skill_revision_required"
            required = True
            tools._ctx._delivery_control_required = True
        elif (
            candidate.finalization_control == "owner_revision_required"
            and is_control_intent
        ):
            # Honor a prior typed instruction during this substantive revision.
            tools._ctx._delivery_control_required = True
        elif (
            candidate.control_episode_seen
            and control_kind in {"keep", "replace", "invalid"}
        ):
            # The latch is gone but this lineage HAS been under host control, and
            # the body is protocol-shaped: reading it as prose would publish the
            # raw JSON as the answer (#447/issue-449).
            historical_control = True
        else:
            # An owner revision starts an ordinary substantive answer round.
            return "fresh", _loop()._extract_plain_text_from_content(content)
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    valid = control_kind == "replace"
    if control_kind == "keep":
        valid = _delivery_keep_allowed(
            candidate, evidence_revision, evidence_fingerprint,
        )
        error = "keep cannot bind changed evidence; send replace with the complete answer"

    if valid and control_kind == "keep":
        tools._ctx._delivery_control_required = False
        candidate.finalization_control = "keep"
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        return "resolved", candidate.full_text
    if valid and control_kind == "replace":
        updated = _loop()._replace_delivery_candidate(
            tools, ctx, llm_trace, replacement, control="replace",
        )
        return "resolved", updated.full_text
    if historical_control:
        # No latch to repair against: keep the retained answer rather than
        # re-arming a control round the host never opened this turn.
        return "resolved", candidate.full_text

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
        candidate.control_episode_seen = True
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        return "retry", ""

    tools._ctx._delivery_control_required = False
    candidate.degraded = True
    candidate.degraded_reason = "invalid_delivery_control_after_repair"
    candidate.finalization_control = "degraded_preserve"
    # The control failed, not the retained text: bind that unchanged text to
    # the evidence the failed control was meant to acknowledge so the stale
    # check cannot reopen another control round. Still explicitly unaccepted;
    # the host acceptance gate judges this exact pair before publication.
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
    control_state, controlled_content = _loop()._resolve_delivery_control(
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
    handoff_msg = _loop()._compute_subagent_handoff(tools, limit_ctx.drive_root, limit_ctx.task_id, content)
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
        # Child absorption is closable only by disposition tool calls: hold —
        # arming the JSON-only instruction would contradict the reminder.
        _hold_delivery_for_skill_action(
            tools, llm_trace, control=_CHILD_ABSORPTION_HOLD_CONTROL,
        )
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
        # Skill finalization is an action gate, not a service notice.
        # Preserve the candidate without a conflicting JSON-only instruction:
        # the next round may run the required tool or give the historically
        # allowed reconsidered answer; a typed keep cannot close it.
        if skill_finalization_injected_now:
            _hold_delivery_for_skill_action(tools, llm_trace)
        else:
            _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return None

    # Declared service outputs and teardown failures are acceptance evidence:
    # finalize them before the host panel and, when that changes evidence,
    # require one replacement answer bound to the new revision (idempotent helper).
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
            model_text=str(content or ""),
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

    _rails_ceiling = getattr(tools._ctx, "_cost_ceiling", None)
    tools._ctx._acceptance_loop_rails = {
        "round_idx": limit_ctx.round_idx,
        "max_rounds": limit_ctx.max_rounds,
        "task_cost_usd": limit_ctx.accumulated_usage.get("cost"),
        "cost_ceiling_usd": getattr(_rails_ceiling, "ceiling_usd", None),
    }
    # v6.78.0 (owner Q20/Q22): mirror the host-attested native-retrieval
    # fact into the trace so `build_task_acceptance_evidence` can show the
    # reviewer whether the answer was grounded in fetched pages. Reviewer-side
    # only — the agent gets the improvement capsule, not the evidence packet.
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
        # v6.71.1: an acceptance improvement pass is an ORDINARY answer round —
        # do NOT arm delivery-control (a JSON-only demand on open obligations
        # froze the model into resubmitting); the next free-form answer
        # re-enters the acceptance panel, other lanes still arm.
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
            text, usage, forced_trace = _loop()._maybe_early_finalize(
                limit_ctx, tools, post_controls,
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
