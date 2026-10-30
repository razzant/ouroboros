"""Delivery candidates and delivery control: child-result dispositions, the
delivery evidence state, the candidate binding, candidate publish/replace/degrade,
the delivery-control prompt cycle, the subagent handoff and the no-tool final
answer. Extracted from loop.py (v7 L-B split); loop.py re-exports every name.

Completion is an author act: ``finish_task`` (including focus
tasks) selects ``action=finish|stop`` with complete ``answer`` bytes or an
``answer_sha256`` naming the retained candidate or the latest whole held
response. Every held no-tool round appends the assistant row and a host row to
the transcript, where held bytes resolve from. A lineage that has not seen a
host control episode (``control_episode_seen``) stays a plain final. Facts Main
observed are frozen before its batch executes; only feedback positively exposed
in a returned request counts as seen."""

from __future__ import annotations

from ouroboros.loop_completion_schema import completion_schema as completion_schema

import hashlib
import json
import logging
import pathlib
import queue

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import working_checkpoint
from ouroboros.config import get_context_mode
from ouroboros.observability import timed_phase
from ouroboros.outcomes import reviewable_effect_projection
from ouroboros.task_finalization import set_terminal_host_notice
from ouroboros.tools.registry import ToolRegistry
from ouroboros.task_carrier import execution_kind


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
    degraded: bool = False
    degraded_reason: str = ""
    model_text: str = ""
    # Sticky loop-local provenance that this lineage has SEEN a host-issued
    # delivery-control episode (#447/issue-449): every replacement inherits it,
    # including ordinary improvements, so a later control-shaped
    # answer under a lost latch is still read as protocol rather than prose.
    control_episode_seen: bool = False
    effective_criteria: Any = None
    material_tool_indices: tuple[int, ...] = ()
    owner_source_sha256: str = ""
    quiet: bool = False  # bytes of a wake's silent selection: replaced, never published, by other prose


# Host continuations retain the answer and require an explicit next selection.
# These states retain the concrete action/handover reason, not another protocol.
_SKILL_ACTION_HOLD_CONTROL = "skill_action_or_revision_required"
_CHILD_ABSORPTION_HOLD_CONTROL = "child_absorption_or_revision_required"
_AUTHORING_HANDOVER_HOLD_CONTROL = "authoring_handover_recovery_required"
_DELIVERY_HOLD_CONTROLS = frozenset({
    _SKILL_ACTION_HOLD_CONTROL,
    _CHILD_ABSORPTION_HOLD_CONTROL,
    _AUTHORING_HANDOVER_HOLD_CONTROL,
})
# Header of the host's own rendered control block; identifies the transcript's
# control history the way ``acceptance_observation`` marks the observation rows.
_DELIVERY_CONTROL_MARKER = "[DELIVERY_FINALIZATION_CONTROL]"
_OBSERVATION_MARKER = "[ACCEPTANCE_SUBJECT_OBSERVATION]"


def _selector_rendered(messages: List[Dict[str, Any]], sha256: str) -> bool:
    """Whether Main was shown a source selector naming ``sha256`` — a host row it
    could act on. A refusal about a selector never rendered is the host's, not Main's."""
    return bool(sha256) and any(
        _OBSERVATION_MARKER in text and sha256 in text
        for text in (str(row.get("content") or "") for row in messages if row.get("role") == "user")
    )


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
        from ouroboros.loop_forced_finalization import HANDLED_CHILD_DISPOSITIONS

        children = [
            child for child in children
            if _loop()._child_disposition_state(child) not in HANDLED_CHILD_DISPOSITIONS
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


def _effective_delivery_criteria(tool_ctx: Any) -> Any:
    """The initial full requirements, then Main's explicit current criteria."""
    existing = getattr(tool_ctx, "_delivery_effective_criteria", None)
    if existing is not None:
        return existing
    from ouroboros.review_evidence_sections import _accept_task_contract

    return {
        "task_contract": _accept_task_contract(tool_ctx),
        "owner_requirements": getattr(tool_ctx, "_owner_directives", []) or [],
    }


def delivery_subject_projection(
    tool_ctx: Any, llm_trace: Dict[str, Any], full_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Stable review meaning, separate from the complete forensic source packet."""
    candidate = getattr(tool_ctx, "_delivery_candidate", None)
    text = full_answer if full_answer is not None else getattr(candidate, "full_text", "")
    return {
        "candidate_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
        "effective_criteria": _effective_delivery_criteria(tool_ctx),
        "material_evidence_fingerprint": delivery_evidence_fingerprint(tool_ctx, llm_trace),
    }


def delivery_subject_hash(
    tool_ctx: Any, llm_trace: Dict[str, Any], full_answer: Optional[str] = None,
) -> str:
    return hashlib.sha256(json.dumps(
        delivery_subject_projection(tool_ctx, llm_trace, full_answer),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def apply_delivery_subject_decision(
    tools: ToolRegistry, ctx: _RoundLimitContext, llm_trace: Dict[str, Any], subject: Dict[str, Any],
) -> tuple[bool, str]:
    """Apply Main's source-addressed criteria/evidence choice atomically."""
    from ouroboros.loop_messages import acknowledge_acceptance_observation

    if not isinstance(subject, dict) or getattr(subject, "duplicate_keys", set()) or set(subject) - {
        "owner_source_sha256", "effective_criteria", "material_tool_indices",
    }:
        return False, "acceptance_subject requires an exact owner source and optional criteria/tool indices"
    criteria = subject.get("effective_criteria", _effective_delivery_criteria(tools._ctx))
    if "effective_criteria" in subject and (not isinstance(criteria, str) or not criteria.strip()):
        return False, "effective_criteria must state the complete current requirements"
    indices = subject.get("material_tool_indices", getattr(tools._ctx, "_delivery_material_tool_indices", ()))
    observed = getattr(tools._ctx, "_acceptance_observation", {})
    count = min(len(llm_trace.get("tool_calls") or []), int(observed.get("tool_count") or 0))
    if not isinstance(indices, (list, tuple)) or any(type(i) is not int or not 0 <= i < count for i in indices):
        return False, "material_tool_indices must address tool results available to this Main turn"
    source = subject.get("owner_source_sha256")
    if not isinstance(source, str):
        return False, "acceptance_subject.owner_source_sha256 must be the observed selector's sha256 string"
    ack = acknowledge_acceptance_observation(tools._ctx, source)
    if not ack:
        # The typed cause and its facts; what Main does next is Main's decision.
        return False, f"{ack.cause} {json.dumps(ack.facts, ensure_ascii=False, sort_keys=True, default=str)}"
    tools._ctx._delivery_effective_criteria = json.loads(json.dumps(criteria, ensure_ascii=False, default=str))
    tools._ctx._delivery_material_tool_indices = tuple(sorted(set(indices)))
    revision, fingerprint = _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, DeliveryCandidate):
        candidate.effective_criteria = tools._ctx._delivery_effective_criteria
        candidate.material_tool_indices = tools._ctx._delivery_material_tool_indices
        candidate.owner_source_sha256 = source
        candidate.evidence_revision, candidate.evidence_fingerprint = revision, fingerprint
        # Main may keep the complete answer while nominating a new review subject.
        if candidate.finalization_control == "owner_revision_required" or candidate.finalization_control.startswith("effect_revision_required"):
            candidate.finalization_control = "awaiting_control"
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    return True, ""


def delivery_evidence_fingerprint(
    tool_ctx: Any, llm_trace: Dict[str, Any], *, task_id: str = "",
    status_root: Any = None, root_task_id: str = "", effective_criteria: Any = None,
) -> str:
    """Fingerprint only evidence that can invalidate a complete answer.

    ``effective_criteria`` permits an explicit criteria projection.
    """

    from ouroboros.outcomes import read_context_verification_receipts
    from ouroboros.tools.join_ledger import _child_result_sha256

    metadata = getattr(tool_ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(task_id or getattr(tool_ctx, "task_id", "") or "")
    root_task_id = str(root_task_id or metadata.get("root_task_id") or task_id)
    status_root = status_root or metadata.get("budget_drive_root") or getattr(tool_ctx, "budget_drive_root", None) or getattr(tool_ctx, "drive_root", None)
    children = _loop()._load_direct_child_results(pathlib.Path(status_root), task_id, root_task_id) if status_root and task_id else []
    # Unclaimed host attention (#1376) is no evidence of the answer: a diagnostic that
    # answers or is withdrawn re-arms nothing. Reading past a settled cancellation, only
    # the author's current exact-hash claim keeps a diagnostic here.
    children = [
        {
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""),
            "sha256": _child_result_sha256(child),
            "disposition": _loop()._child_disposition_state(child),
        }
        for child in children
        if _loop()._child_disposition_state({**child, "parent_decision": ""}) != "host_attention"
    ]
    receipt_root = getattr(tool_ctx, "drive_root", None) or status_root
    evidence = {
        "effective_criteria": (effective_criteria if effective_criteria is not None
                               else _effective_delivery_criteria(tool_ctx)),
        "material_tool_results": [
            {"index": index, **{key: call.get(key) for key in (
                "tool", "args", "status", "is_error", "result", "result_ref", "artifact_registered",
            )}}
            for index, call in enumerate(llm_trace.get("tool_calls") or [])
            if isinstance(call, dict) and index in getattr(tool_ctx, "_delivery_material_tool_indices", ())
        ],
        "tool_effects": reviewable_effect_projection(llm_trace),
        # A plan_task call is not a filesystem effect, but what it wrote and asked
        # changes whether a pre-plan answer is grounded.
        "plan_review_receipts": [
            {"index": index, "result": call.get("result"),
             **{key.removeprefix("plan_"): call.get(key) for key in (
                 "plan_effect", "plan_revision", "plan_open_objections", "plan_holding", "plan_pending_asks")}}
            for index, call in enumerate(llm_trace.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("plan_effect")
        ],
        "children": children,
        "verification_receipts": read_context_verification_receipts(
            tool_ctx, task_id, fallback_root=receipt_root,
        ),
        # Task-scoped service teardown can register declared outputs or
        # surface an output-finalization failure. Those facts arise outside an
        # ordinary tool call, so bind their stable projection explicitly; else
        # a result review could judge the pre-teardown state.
        "service_finalization": _loop()._service_finalization_evidence(llm_trace),
    }
    return hashlib.sha256(json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()

def observed_delivery_evidence(tool_ctx: Any, llm_trace: Dict[str, Any], **bounds: Any) -> str:
    """The ONE fallible evidence read, typed: an uncomputable fingerprint is
    UNKNOWN (``""``), never an exception after an answer exists and never
    currency; the host pass, reading the subject itself, accounts the incident."""
    try:
        return delivery_evidence_fingerprint(tool_ctx, llm_trace, **bounds)
    except Exception:
        log.debug("delivery evidence fingerprint unavailable; typed unknown", exc_info=True)
        return ""

def _delivery_evidence_state(tools: ToolRegistry, ctx: _RoundLimitContext, llm_trace: Dict[str, Any]) -> tuple[int, str]:
    """Track the shared answer-invalidating evidence fingerprint (every retention,
    post-tool, control, publication, delivery and forced-exit path). UNKNOWN (#1224)
    bumps no revision and keeps the last KNOWN fingerprint — missing evidence is not
    a change; a KNOWN candidate compared against it is re-retained explicitly by its caller."""
    revision = int(getattr(tools._ctx, "_delivery_evidence_revision", 0) or 0)
    fingerprint = observed_delivery_evidence(
        tools._ctx, llm_trace, task_id=ctx.task_id, root_task_id=ctx.root_task_id,
        status_root=ctx.status_drive_root or ctx.drive_root or pathlib.Path(ctx.drive_logs).parent,
    )
    if not fingerprint:
        return revision, ""
    if fingerprint != str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or ""):
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


def _publish_delivery_candidate(
    tools: ToolRegistry,
    candidate: DeliveryCandidate,
    llm_trace: Dict[str, Any],
) -> None:
    """Publish subject/control facts; the complete answer remains loop-local."""
    from ouroboros.observability import redact_projection

    current_fp = str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or "")
    subject = ""  # empty for unknown evidence or a subject the host could not compute
    if candidate.evidence_fingerprint:
        try:
            subject = delivery_subject_hash(tools._ctx, llm_trace, candidate.full_text)
        except Exception:  # the host's own subject computation failed: published as unavailable, never as approved
            log.debug("delivery subject unavailable for the retained answer", exc_info=True)
    llm_trace["delivery_candidate"] = {
        "content_sha256": candidate.content_sha256,
        "revision": candidate.revision,
        "evidence_revision": candidate.evidence_revision,
        "evidence_fingerprint": candidate.evidence_fingerprint,
        "evidence_current": bool(subject) and candidate.evidence_fingerprint == current_fp,
        "acceptance_binding": dict(candidate.acceptance_binding),
        "finalization_control": candidate.finalization_control,
        "control_episode_seen": candidate.control_episode_seen,
        "degraded": candidate.degraded,
        "degraded_reason": candidate.degraded_reason,
        "effective_criteria": redact_projection(candidate.effective_criteria).value,
        "material_tool_indices": list(candidate.material_tool_indices),
        "owner_source_sha256": candidate.owner_source_sha256,
        "owner_source_current": not _loop()._owner_generation_changed(tools._ctx),
        "subject_sha256": subject,
        **({} if subject else {"evidence_status": "unavailable"}),
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
    # The answer's own bytes: its sha256 is the identity selection, result review
    # and delivery name. Redaction belongs to derived log and UI projections.
    model_text = full_text if model_text is None else model_text
    previous_candidate = getattr(tools._ctx, "_delivery_candidate", None)
    from ouroboros.loop_messages import acknowledge_acceptance_observation, owner_source_sha256

    observed = getattr(tools._ctx, "_acceptance_observation", {})
    if previous_candidate is None and observed.get("owner_source_sha256"):
        acknowledge_acceptance_observation(tools._ctx, observed["owner_source_sha256"])
    if getattr(tools._ctx, "_delivery_effective_criteria", None) is None:
        tools._ctx._delivery_effective_criteria = json.loads(json.dumps(
            _effective_delivery_criteria(tools._ctx), ensure_ascii=False, default=str,
        ))
    if (isinstance(previous_candidate, _loop().DeliveryCandidate) and previous_candidate.full_text == full_text
            # An unchanged repeat under the same (known or UNKNOWN) evidence stays that candidate.
            and _loop()._current_delivery_candidate(ctx, llm_trace) is previous_candidate):
        previous_candidate.finalization_control = control
        tools._ctx._delivery_control_required = False
        _loop()._publish_delivery_candidate(tools, previous_candidate, llm_trace)
        return previous_candidate
    # Retention never waits on the evidence read (#1224): unknown evidence keeps the answer.
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
        effective_criteria=tools._ctx._delivery_effective_criteria,
        material_tool_indices=tuple(getattr(tools._ctx, "_delivery_material_tool_indices", ())),
        # The source snapshot is not an acknowledgment. Legacy/direct callers
        # still need to distinguish a changed corpus from a current candidate.
        owner_source_sha256=str(getattr(tools._ctx, "_acceptance_ack_source_sha256", "") or owner_source_sha256(tools._ctx)),
    )
    tools._ctx._delivery_candidate = candidate
    tools._ctx._delivery_control_required = False
    from ouroboros.tools.review_result import supersede_result_runs

    supersede_result_runs(llm_trace, f"candidate-{revision}")  # forced rails replace here too
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
    if _loop()._owner_generation_changed(ctx.tools._ctx):
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


def _delivery_control_prompt(candidate: DeliveryCandidate, *, pending_review_choice: bool = False) -> str:
    """Selection is an author act, never a classifier over prose or its length."""
    return ("[DELIVERY_FINALIZATION_CONTROL]\n[SYSTEM NOTICE] The complete answer is retained as "
            + candidate.content_sha256 + ". Continue useful work or use the available completion tool "
            "to select complete answer bytes or this answer_sha256. action=finish requests the normal "
            "completion checks; action=stop with a rationale records unfinished work without cancelling children. "
            "Interim prose is activity, not a new completion selection."
            + (" Pending critics: choose pending_review=wait (default) or finish where permitted."
               if pending_review_choice else ""))


def hold_completion_response(content: Any, tools: ToolRegistry, ctx: Any, trace: dict,
                             *, response_start: int | None = None, selected: bool = False) -> None:
    """Keep each complete model row followed by host input, even on unchanged facts."""
    raw = _loop()._extract_plain_text_from_content(content)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    parsed, duplicate, embedded = _parse_delivery_control_body(raw)
    private = bool(candidate and candidate.control_episode_seen and
                   (duplicate or embedded or isinstance(parsed, dict) and "delivery_control" in parsed))
    row = {"role": "assistant", "content": raw}
    # Older gates may already have recorded this response before their host notice.
    # Never deduplicate against an earlier model round with identical answer bytes.
    recorded = row in ctx.messages[response_start:] if response_start is not None else ctx.messages[-1:] == [row]
    if not recorded:
        ctx.messages.append(row)
    tools._ctx._completion_held_sha256 = "" if private else hashlib.sha256(raw.encode("utf-8")).hexdigest()
    door = completion_schema(tools, getattr(ctx, "tool_schemas", None))
    notice = ("[SYSTEM NOTICE] " + ("Selected completion was held. Use " if selected else "No completion selection was made. Use ") + door +
              " to finish or stop, or continue work. Retained answer_sha256=" +
              str(getattr(candidate, "content_sha256", "")))
    if not private:
        notice += "; latest whole held response answer_sha256=" + tools._ctx._completion_held_sha256
    ctx.messages.append({"role": "user", "content": notice + "."})
    tools._ctx._completion_pair_appended = True
    tools._ctx._completion_pair_private = private
    if candidate is not None:
        candidate.control_episode_seen = True
        tools._ctx._delivery_control_required = True
        _loop()._publish_delivery_candidate(tools, candidate, trace)


def observe_completion_hold(tools: ToolRegistry, ctx: Any, origin: str, reason: str, subreason: str = "") -> None:
    """One diagnostic ``completion_held`` checkpoint, written by the branch that held a completion.

    ``origin`` is ``selection``, ``prose`` or ``resume``; ``reason``/``subreason`` are that
    branch's typed codes. The row names this attempt, execution and round and the candidate
    retained now (hash and revision, or null), never response or owner text, in the canonical
    events log that outlives a split execution drive. It decides nothing: a failure is only logged."""
    try:
        from ouroboros.preflight_receipts import canonical_data_root

        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        retained = isinstance(candidate, _loop().DeliveryCandidate)
        root = canonical_data_root(tools._ctx)
        _loop()._emit_checkpoint_event(
            getattr(ctx, "event_queue", None), str(getattr(ctx, "task_id", "") or ""),
            root / "logs" if root is not None else getattr(ctx, "drive_logs", None),
            {"checkpoint_kind": "completion_held", "task_attempt": getattr(tools._ctx, "task_attempt", None),
             "execution_id": (getattr(ctx, "accumulated_usage", None) or {}).get("execution_id"),
             "round": getattr(ctx, "round_idx", None), "origin": origin, "reason": reason,
             "subreason": subreason or None, "answer_sha256": candidate.content_sha256 if retained else None,
             "candidate_revision": candidate.revision if retained else None})
    except Exception:
        log.warning("Completion hold of %s was not observed", getattr(ctx, "task_id", ""), exc_info=True)


def _hold_cause(tool_ctx: Any) -> str:
    """The typed cause a nudge or result review named for the hold it just made, read once."""
    cause, tool_ctx._completion_hold_cause = str(getattr(tool_ctx, "_completion_hold_cause", "") or ""), ""
    return cause


def completion_observation(ctx: Any, trace: dict) -> dict:
    """Freeze facts Main observed, before any sibling of its response executes."""
    import copy
    from ouroboros.loop_messages import owner_source_sha256
    return {"subject": copy.deepcopy(getattr(ctx, "_acceptance_observation", {})),
            "tool_count": len(trace.get("tool_calls") or []),
            "owner_directives": len(getattr(ctx, "_owner_directives", []) or []),
            "owner_source_sha256": owner_source_sha256(ctx),
            "evidence_fingerprint": observed_delivery_evidence(ctx, trace)}


def selected_completion_text(request: dict, candidate: Any, messages: list,
                             held_sha256: str = "") -> tuple[str | None, str]:
    """Only complete explicit bytes and the two currently offered identities resolve."""
    if "answer" in request:
        text = request["answer"]
        return (text, "") if isinstance(text, str) and (text.strip() or request.get("allow_empty")) else (None, "answer_unavailable")
    selector = request.get("answer_sha256")
    if candidate is not None and selector == candidate.content_sha256:
        return candidate.full_text, ""
    if selector and selector == held_sha256:
        for row in reversed(messages):
            text = row.get("content")
            if row.get("role") == "assistant" and isinstance(text, str) and hashlib.sha256(text.encode("utf-8")).hexdigest() == selector:
                return text, ""
    return None, "selection_unavailable: select the current retained/whole held answer or supply complete bytes"


def consume_completion_request(tools: ToolRegistry, ctx: Any, trace: dict, content: Any = None) -> bool:
    """Resolve one staged author act after the full batch, using its original observation."""
    from ouroboros.tool_capabilities import completion_observation_calls
    from ouroboros.loop_messages import owner_reply_owed, owner_source_sha256
    tool_ctx = tools._ctx
    request = getattr(tool_ctx, "_completion_request", None)
    if not isinstance(request, dict):
        return False
    if request.get("reply_later"):
        if content is None:
            return False
        request = {**request, "answer": _loop()._extract_plain_text_from_content(content), "reply_later": False}
    observation = request.get("observation") or {}
    candidate = getattr(tool_ctx, "_delivery_candidate", None)
    text, error = selected_completion_text(request, candidate, ctx.messages, getattr(tool_ctx, "_completion_held_sha256", ""))
    if request.get("wake_silent") and owner_reply_owed(tool_ctx):
        error = "owner_reply_owed: your human wrote after your last live message; answer before finishing silently"
    if getattr(tool_ctx, "_completion_conflict", False):
        error = "contradictory_completion_requests: select again after reading the completed batch"
    if observation.get("owner_source_sha256") != owner_source_sha256(tool_ctx):
        error = "owner_input_changed: consider the new owner input before selecting completion"
    calls = (trace.get("tool_calls") or [])[int(observation.get("tool_count") or 0):]
    unseen = completion_observation_calls(calls)
    # Delivery-only Presence composition relies on actual delivery receipts downstream.
    delivered = (getattr(tool_ctx, "_presence_completion", None) or {}).get("outcome") == "tool_delivered"
    if delivered:
        unseen = [call for call in unseen if not call.get("presence_delivery_confirmed") or call.get("is_error")]
    subject, applied = request.get("acceptance_subject"), None
    if not error and subject is not None:
        previous = getattr(tool_ctx, "_acceptance_observation", {})
        tool_ctx._acceptance_observation = observation.get("subject", {})
        try:
            applied, error = apply_delivery_subject_decision(tools, ctx, trace, subject)
        finally:
            tool_ctx._acceptance_observation = previous
    if text is not None and not error:
        candidate = _loop()._replace_delivery_candidate(tools, ctx, trace, text, control="selected")
        candidate.quiet = bool(request.get("wake_silent"))
        _loop()._latch_final_answer_marker(trace, "" if candidate.quiet else text)  # quiet words latch no fallback
    if not error and request["action"] == "finish" and unseen:
        error = "unobserved_tool_results: read the completed results before finishing: " + ", ".join(str(c.get("tool")) for c in unseen)
    tool_ctx._completion_request = None
    tool_ctx._completion_conflict = False
    if error:
        trace.setdefault("completion_refusals", []).append({"reason": error, "request": request})
        ctx.messages.append({"role": "user", "content": "[SYSTEM NOTICE] Completion held: " + error})
        if candidate is not None:
            _loop()._arm_delivery_control(tools, ctx, trace)
        _void_presence_completion(tool_ctx, ctx.messages)
        # A subject refusal is host prose; an acknowledgement outcome keeps its own durable row.
        observe_completion_hold(tools, ctx, "selection", "selection_refused",
                                "acceptance_subject" if applied is False else error.split(":", 1)[0])
        return False
    trace.pop("task_completion", None)
    tool_ctx._completion_selected = request
    if isinstance(getattr(tool_ctx, "_presence_completion", None), dict):
        tool_ctx._presence_completion["message"] = text
        tool_ctx._presence_forced_declaration = {"status": "missing", "reason": "completion has not passed the common finalization gates"}
    tool_ctx._acceptance_pending_review_choice = request.get("pending_review", "wait")
    return True


def _delivery_replace_required(candidate: DeliveryCandidate) -> bool:
    """Return whether a typed full replacement is mandatory for this control round."""

    return candidate.finalization_control.startswith(
        ("effect_revision_required", "skill_revision_required")
    )


def _arm_delivery_control(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    control: str = "awaiting_control",
    skip_if_unchanged: bool = False,
) -> None:
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, _loop().DeliveryCandidate):
        return
    _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    candidate.finalization_control = control
    tools._ctx._delivery_control_required = True
    from ouroboros.tools.review_result import pending_review_choice_offered

    control_prompt = _delivery_control_prompt(
        candidate, pending_review_choice=pending_review_choice_offered(tools._ctx))
    # ``skip_if_unchanged`` is a repeated re-offer of the selection contract:
    # unchanged facts render identical bytes, so the
    # transcript's control history is not repeated, the way
    # ``prepare_acceptance_observation`` skips an unchanged observation. Every
    # other caller arms because something changed and always appends. ``slot``
    # keeps an already-sent tail row byte-frozen rather than rewritten (#906).
    latest = next((row for row in reversed(ctx.messages)
                   if _DELIVERY_CONTROL_MARKER in str(row.get("content") or "")), None)
    if not (skip_if_unchanged and latest is not None
            and control_prompt in str(latest.get("content") or "")):
        _loop()._append_or_merge_user_message(ctx.messages, control_prompt, slot=tools._ctx)
    candidate.control_episode_seen = True
    completion_schema(tools, getattr(ctx, "tool_schemas", None))
    from ouroboros.loop_messages import capture_acceptance_observation, acceptance_observation_prompt

    observed = capture_acceptance_observation(tools._ctx, llm_trace, getattr(ctx, "incoming_messages", None))
    if prompt := acceptance_observation_prompt(tools._ctx, observed):
        _loop()._append_or_merge_user_message(ctx.messages, prompt, slot=tools._ctx)
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
    tools._ctx._delivery_control_required = True
    candidate.control_episode_seen = True
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


class _ParsedObject(dict):
    """A parsed JSON object that remembers which of its keys were duplicated, and whether one of its
    own pairs, a repeated key's included, is the ``delivery_control`` key or an ``action`` of exactly
    finish/stop (``control_pair``)."""

    duplicate_keys: set
    has_duplicate_keys: bool
    control_pair: bool


def _parse_pairs(raw: str) -> Tuple[Any, bool]:
    """One linear JSON parse whose objects are ``_ParsedObject``: ``(payload, any key duplicated)``,
    ``(None, False)`` when the body is not JSON."""

    has_duplicate_keys = False

    def _unique_object(pairs: List[Tuple[str, Any]]) -> "_ParsedObject":
        nonlocal has_duplicate_keys
        result = _ParsedObject()
        result.duplicate_keys = set()
        result.control_pair = False
        for key, value in pairs:
            if key in result:
                result.duplicate_keys.add(key)
                has_duplicate_keys = True
            result[key] = value
            result.control_pair = result.control_pair or key == "delivery_control" or (
                key == "action" and value in ("finish", "stop"))
        return result

    try:
        return json.loads(raw, object_pairs_hook=_unique_object), has_duplicate_keys
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        # RecursionError: a degenerate deeply-nested blob (repetition-loop
        # model output) must classify as not-a-control, not crash the round.
        return None, False


def _parse_delivery_control_object(
    raw: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Parse a body while rejecting duplicates in a control envelope.

    The boolean preserves top-level protocol intent when duplicate keys made a
    recognizable control envelope invalid. Per-object metadata supports the
    stronger armed/action rails; an aggregate duplicate marker restores the
    forced armed malformed-body rule without widening history-gated parsing.
    """

    payload, has_duplicate_keys = _parse_pairs(raw)
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
    embedded: bool, *, envelope_keys: Tuple[str, ...] = (),
) -> Tuple[str, str, str]:
    """Return ``(kind, replacement, error)``; ``envelope_keys`` are members an armed caller reads itself."""

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
    if "pending_review" in parsed and str(parsed.get("pending_review") or "").strip().lower() not in {"wait", "finish"}:
        return "invalid", "", 'pending_review must be "wait" or "finish"'
    keys = set(parsed) - {"acceptance_subject", "pending_review", *envelope_keys}
    if selected == "keep" and keys == {"delivery_control"}:
        return "keep", "", ""
    if selected == "replace" and keys == {"delivery_control", "full_answer"}:
        replacement = parsed.get("full_answer")
        if isinstance(replacement, str) and replacement.strip():
            return "replace", replacement, ""
        return "invalid", "", "replace requires a non-empty complete full_answer"
    return "invalid", "", exact_error


def _resolve_forced_delivery_control_body(
    raw: str,
    candidate: Optional[DeliveryCandidate],
    *,
    armed: bool, envelope_keys: Tuple[str, ...] = (), messages: list | None = None, held_sha256: str = "",
) -> Tuple[str, bool, bool, bool, bool]:
    """Return text plus retained/degraded/consumed/replaced facts."""

    if not isinstance(candidate, _loop().DeliveryCandidate):
        candidate = None
    parsed, duplicate_protocol_key, embedded_protocol = _parse_delivery_control_body(raw)
    if isinstance(parsed, dict) and "action" in parsed:
        allowed = {"action", "answer", "answer_sha256", "rationale", "acceptance_subject", "pending_review", *envelope_keys}
        valid = (not duplicate_protocol_key and not embedded_protocol and not getattr(parsed, "duplicate_keys", set())
                 and not set(parsed) - allowed and parsed.get("action") in {"finish", "stop"}
                 and (("answer" in parsed) != ("answer_sha256" in parsed))
                 and parsed.get("pending_review", "wait") in {"wait", "finish"}
                 and (parsed["action"] != "stop" or isinstance(parsed.get("rationale"), str) and parsed["rationale"].strip()))
        text, error = selected_completion_text(parsed, candidate, messages or [], held_sha256) if valid else (None, "invalid_completion_request")
        if error:
            return candidate.full_text if candidate else "", candidate is not None, True, True, False
        retained = candidate is not None and parsed.get("answer_sha256") == candidate.content_sha256
        return text, retained, False, True, not retained
    control_kind, replacement, _error = _classify_parsed_delivery_control(
        parsed, duplicate_protocol_key, embedded_protocol, envelope_keys=envelope_keys,
    )
    historical = bool(
        not armed
        and candidate is not None
        and candidate.control_episode_seen
        and control_kind in {"keep", "replace", "invalid", "embedded"}
    )
    if not armed and not historical:
        return raw, False, False, False, False
    if control_kind in {"keep", "replace", "invalid", "embedded"}:
        # Read old protocol privately; it no longer selects bytes or authorizes delivery.
        return (candidate.full_text if candidate else "", candidate is not None,
                candidate is None or control_kind in {"invalid", "embedded"}, True, False)
    from ouroboros.observability import strip_protocol_fence

    protocol_intent = (
        control_kind != "none"
        or (parsed is None and strip_protocol_fence(raw).startswith("{"))
        or bool(getattr(parsed, "has_duplicate_keys", False))
        or bool(envelope_keys and isinstance(parsed, dict) and set(parsed).intersection(envelope_keys))
    )
    if not protocol_intent:
        # Ordinary prose under an armed latch stands (a control object quoted
        # MID-prose is the disclosed residual).
        return raw, False, False, True, False
    retained = candidate is not None
    return candidate.full_text if retained else "", retained, True, True, False


def _array_wrapped_control(body: str) -> bool:
    """Whether a whole-body JSON array holds a control envelope as a DIRECT element: an object one of
    whose own pairs is the ``delivery_control`` key or an ``action`` of exactly finish/stop, so a
    repeated key cannot hide the verb. Never unwrapped or executed; nested data, strings quoting an
    envelope and other verbs stay ordinary JSON."""
    payload = _parse_pairs(body)[0] if body.startswith("[") else None
    return isinstance(payload, list) and any(getattr(item, "control_pair", False) for item in payload)


def _parse_delivery_control_body(
    raw: str,
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Normalize a response body and locate its delivery-control object.

    Returns ``(parsed, duplicate_protocol_key, embedded)``. Normalization
    strips one whole-body markdown fence (shared with
    ``observability._is_delivery_control_payload``). ``embedded`` is True only
    when a protocol object sits in a container other than the one exact
    object: a balanced trailing JSON object carrying the ``delivery_control``
    key at the very END of surrounding prose, or a direct element of a
    whole-body JSON array (``_array_wrapped_control``) — a protocol attempt,
    never a valid control. A control object quoted MID-prose is NOT matched and
    stays prose (disclosed residual)."""
    from ouroboros.observability import strip_protocol_fence

    body = strip_protocol_fence(raw)
    parsed, duplicate_protocol_key = _parse_delivery_control_object(body)
    if duplicate_protocol_key or isinstance(parsed, dict):
        return parsed, duplicate_protocol_key, False
    if _array_wrapped_control(body):
        return None, False, True
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


def _resolve_delivery_control(content: Any, tools: ToolRegistry, ctx: Any, llm_trace: dict) -> tuple[str, str]:
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    raw = _loop()._extract_plain_text_from_content(content)
    if candidate is None:
        return "fresh", raw
    required = bool(getattr(tools._ctx, "_delivery_control_required", False) or candidate.control_episode_seen
                    or candidate.finalization_control in _DELIVERY_HOLD_CONTROLS
                    or _delivery_replace_required(candidate))
    if not required:
        if candidate.finalization_control == "owner_revision_required":
            from ouroboros.loop_messages import acknowledge_acceptance_observation
            observed = getattr(tools._ctx, "_acceptance_observation", {})
            if observed.get("owner_source_sha256"):
                acknowledge_acceptance_observation(tools._ctx, observed["owner_source_sha256"])
        return "fresh", raw
    hold_completion_response(content, tools, ctx, llm_trace)
    return "retry", candidate.full_text


def _compose_delivery_suffix(full_text: str, suffix: str) -> str:
    """Join host-authored fallback text and its notice, without model authorship."""

    text = str(full_text or "")
    note = str(suffix or "")
    if not note or text.endswith(note):
        return text
    return text + note


def _plan_review_only_awaited(llm_trace: Dict[str, Any]) -> bool:
    """The projected gate decision's typed fact: the open plan wave was merely awaited.

    The finalization notice and this fact read the SAME decision, so an only-awaited
    wave keeps its loud notice and is a gap, never a degradation of the task."""
    plan_gate = llm_trace.get("plan_gate", llm_trace.get("force_plan_decision"))
    return isinstance(plan_gate, dict) and plan_gate.get("review_only_awaited") is True


@timed_phase("admission_seal")
def _seal_admission_before_delivery(tools: ToolRegistry, limit_ctx: Any, llm_trace: Dict[str, Any]) -> bool:
    """Seal root admission at an author stop, right before delivery; False holds the answer.

    Only owner facts revise: a generation mismatch (also one seen locally while the transport
    was silent) or pending owner mail. A begin ``refused`` because the root is already ``sealed``
    is the worker's own lost ack; a pooled attempt refused ``stale_attempt`` lost its queue
    ownership and never publishes. Any other refusal, like ``unknown``, is the host's own state:
    delivered with ``admission_released=False`` (``supervisor_ack_unavailable`` keeps the cause)."""
    tool_ctx = tools._ctx
    from ouroboros.task_results import resolve_ctx_lineage
    if not resolve_ctx_lineage(tool_ctx, limit_ctx.task_id)["is_root_task"]:
        return True  # Local child completion has no authority to seal its parent's root.
    opened, _token = _loop()._begin_task_acceptance_fence(tool_ctx, limit_ctx.task_id)
    answer = opened and _loop()._end_task_acceptance_fence(tool_ctx, outcome="author_stop")
    own_seal = (opened.status, opened.reason) == ("refused", "sealed")
    if own_seal:
        answer = _loop()._end_task_acceptance_fence(tool_ctx, outcome="author_stop")
        own_seal = bool(answer)
    from ouroboros.loop_messages import _pending_owner_input_kinds

    # The wait for a silent supervisor is long enough for the owner to write: their mail is durable
    # before any generation moves, so the local mailbox is read once more before a gap delivers.
    superseded = answer.reason == "stale_attempt" and execution_kind(tool_ctx) == "pooled"  # lost its queue ownership
    gap = not answer and not own_seal and not superseded and not _pending_owner_input_kinds(tool_ctx)  # owner facts revise
    if getattr(tool_ctx, "_task_acceptance_fence_generation_mismatch", False) or not (answer or own_seal or gap):
        _loop()._release_for_owner_followup(tool_ctx, llm_trace)
        admission_lock = getattr(tool_ctx, "owner_message_admission_lock", None)
        admission_agent = getattr(tool_ctx, "owner_message_admission_agent", None)
        if admission_lock is not None and admission_agent is not None:
            with admission_lock:
                admission_agent._accepting_owner_messages = True
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace, control="owner_revision_required")
        return False
    if not answer and not own_seal:
        llm_trace.setdefault("review_decision", {})["admission_released"] = False
    return True


def _void_presence_completion(tool_ctx: Any, messages: list) -> None:
    completion = getattr(tool_ctx, "_presence_completion", None)
    if isinstance(completion, dict):
        from ouroboros.presence_context import presence_finish_not_accepted_note
        messages.append({"role": "user", "content": presence_finish_not_accepted_note(tool_ctx, completion)})
        tool_ctx._presence_completion, tool_ctx._presence_completion_accepted = None, False


def resume_delivery_candidate(tools: ToolRegistry, working: dict, schemas: list) -> bool:
    """Restore completion tools and identify an interrupted finalization boundary."""
    if getattr(tools._ctx, "_delivery_control_required", False):
        completion_schema(tools, schemas)
    return bool(working and (working.get("working") or {}).get("boundary") == "candidate"
                and getattr(tools._ctx, "_delivery_candidate", None) is not None)


def _prepare_delivery_candidate(content: Any, tools: ToolRegistry, limit_ctx: Any, llm_trace: dict,
                                *, explicit: bool, resumed: bool) -> tuple[bool, Any]:
    """Select/save prose before gates; recovery retains its old evidence binding."""
    if resumed or explicit:
        control, content = ("retained" if resumed else "fresh"), content
    else:
        control, content = _loop()._resolve_delivery_control(content, tools, limit_ctx, llm_trace)
    if control == "retry":
        return False, content
    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    held = getattr(tools._ctx, "_delivery_candidate", None)
    quiet_held = not explicit and getattr(held, "quiet", False)  # replaced even by empty prose, never published
    if resumed and quiet_held:
        return False, content  # a quiet selection lost with its execution id is chosen again
    fresh = control == "fresh" and (explicit or str(content or "").strip() or quiet_held)
    candidate = _loop()._replace_delivery_candidate(tools, limit_ctx, llm_trace, str(content or ""), control="candidate") if fresh else held
    if isinstance(candidate, DeliveryCandidate):
        candidate.quiet = explicit and bool((getattr(tools._ctx, "_completion_selected", None) or {}).get("wake_silent"))
        content = candidate.full_text
    if fresh or resumed:
        working_checkpoint.save_or_log(limit_ctx, "candidate")  # this attempt survives another interruption
    return True, content


def _no_tool_final_answer(
    content: Any,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    tools: ToolRegistry,
    incoming_messages: queue.Queue,
    owner_msg_seen: set,
    emit_progress: Callable[[str], None],
    *, explicit_candidate: bool = False, resume_candidate: bool = False,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Run the no-tool finalization gates; ``None`` requests another model round."""
    messages = limit_ctx.messages
    from ouroboros.loop_messages import transcript_growth_signature
    before = transcript_growth_signature(messages)
    origin = "resume" if resume_candidate else "selection" if explicit_candidate else "prose"

    def held(reason: str, subreason: str = ""):
        if transcript_growth_signature(messages) != before:
            _void_presence_completion(tools._ctx, messages)
        observe_completion_hold(tools, limit_ctx, origin, reason, subreason)
        return None
    ready, content = _prepare_delivery_candidate(content, tools, limit_ctx, llm_trace,
                                                 explicit=explicit_candidate, resumed=resume_candidate)
    if not ready:  # prose under armed control, or a resumed quiet selection lost with its execution
        return held("quiet_selection_lost" if resume_candidate else "selection_required")

    stopping = (getattr(tools._ctx, "_completion_selected", None) or {}).get("action") == "stop"
    if not stopping:
        if _loop()._enforce_plan_gate(
            str(content or ""), messages, tools, llm_trace, emit_progress,
        ):
            return held("plan_gate")
        handoff_msg = _loop()._compute_subagent_handoff(tools, limit_ctx.drive_root, limit_ctx.task_id, content)
        if handoff_msg:
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{handoff_msg}")
            emit_progress("Subagent handoff status refreshed before final response.")
            llm_trace["reasoning_notes"].append("Subagent handoff status refreshed before final response.")
            _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
            return held("subagent_handoff")
        absorption_result = _loop()._maybe_enforce_child_absorption_gate(
            tools, limit_ctx, content, messages, emit_progress, llm_trace,
        )
        if absorption_result == "continue":
            # Preserve the child action reason until disposition or explicit stop.
            _hold_delivery_for_skill_action(
                tools, llm_trace, control=_CHILD_ABSORPTION_HOLD_CONTROL,
            )
            return held("child_absorption")
        if absorption_result is not None:
            return absorption_result
        skill_finalization_was_injected = bool(
            getattr(tools._ctx, "_skill_finalization_injected", False)
        )
        handover = getattr(tools._ctx, "_authoring_handover", None)
        handover_was_prompted = bool(handover and handover.get("recovery_prompted"))
        if _loop()._maybe_inject_finalization_nudges(
            tools, limit_ctx.drive_root, limit_ctx.task_id, llm_trace, content, messages, emit_progress,
        ):
            skill_finalization_injected_now = (
                not skill_finalization_was_injected
                and bool(getattr(tools._ctx, "_skill_finalization_injected", False))
            )
            # Completion-only calls do not reset this action nudge or imply work.
            if skill_finalization_injected_now:
                _hold_delivery_for_skill_action(tools, llm_trace)
            elif handover and not handover_was_prompted and handover.get("recovery_prompted"):
                _hold_delivery_for_skill_action(
                    tools, llm_trace,
                    control=_AUTHORING_HANDOVER_HOLD_CONTROL,
                )
            else:
                _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
            return held("finalization_nudge", _hold_cause(tools._ctx))

    # Declared service outputs and teardown failures are evidence of the result:
    # finalize them before its review and, when that changes evidence,
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
    if _loop()._finalize_task_services(service_exit_ctx) and not stopping:
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
                "Task services were finalized before delivery; the complete answer must bind the resulting evidence."
            )
            _loop()._disclose_service_finalization(limit_ctx, llm_trace)
            _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
            return held("service_evidence_changed")

    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    plan_suffix = _loop()._plan_gate_disclosure(tools._ctx, llm_trace)
    # The forced rails already type this fact; the single degraded_reason slot below
    # cannot hold it beside an unsettled child, so the normal rail types it too.
    if plan_suffix:  # absent = not open: a clean result carries no key (pinned usage shapes)
        limit_ctx.accumulated_usage["terminal_plan_review_open"] = True
    orphan_suffix = _loop()._forced_orphan_note(limit_ctx, include_terminal=False)
    set_terminal_host_notice(limit_ctx.accumulated_usage, plan_suffix, orphan_suffix)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, _loop().DeliveryCandidate):
        if orphan_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "host_child_status_suffix"
            _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        elif plan_suffix and not _plan_review_only_awaited(llm_trace):
            candidate.degraded = True
            candidate.degraded_reason = "plan_review_advisory"
            _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text

    def _hold_for_review(text: str) -> None:
        if content and str(content).strip():
            messages.append({"role": "assistant", "content": str(content)})
        _loop()._append_or_merge_user_message(messages, text)
        emit_progress("Result review outcome shown before delivery.")
        llm_trace["reasoning_notes"].append("Result review outcome shown before delivery.")

    from ouroboros.tools.review_result import result_review_at_delivery

    if not stopping and result_review_at_delivery(tools._ctx, content or "", llm_trace, _hold_for_review):
        # The author continues freely; its next complete answer is selected explicitly.
        return held("result_review", _hold_cause(tools._ctx))
    candidate = getattr(tools._ctx, "_delivery_candidate", None)

    # Close delivery under the same lock as routing, then drain once. A follow-up
    # either forces another round or is rejected after the fence, never stranded.
    admission_lock = getattr(tools._ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(tools._ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        before_directives = len(getattr(tools._ctx, "_owner_directives", []) or [])
        sealed = bool(getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None))
        provisional_assistant = {"role": "assistant", "content": content} if content else None
        if provisional_assistant is not None:
            messages.append(provisional_assistant)
        with timed_phase("admission_wait") as acquired, admission_lock:
            acquired()
            with timed_phase("admission_close"):
                admission_agent._accepting_owner_messages = False
                post_controls = _loop()._drain_incoming_messages(
                    messages, incoming_messages, limit_ctx.drive_root, limit_ctx.task_id,
                    limit_ctx.event_queue, owner_msg_seen, owner_ctx=tools._ctx, defer_content_ack=True,
                )
        if len(getattr(tools._ctx, "_owner_directives", []) or []) > before_directives:
            with admission_lock:
                if sealed:
                    _loop()._release_for_owner_followup(tools._ctx, llm_trace, admission_locked=True)
                if (
                    getattr(admission_agent, "_busy", False)
                    and str(getattr(admission_agent, "_current_task_id", "") or "") == limit_ctx.task_id
                ):
                    admission_agent._accepting_owner_messages = True
            if sealed:
                emit_progress("An owner follow-up reopened the sealed task; Main decides about the retained answer.")
            if isinstance(candidate, _loop().DeliveryCandidate):
                candidate.finalization_control = "owner_revision_required"
                if candidate.control_episode_seen or sealed:
                    _loop()._arm_delivery_control(tools, limit_ctx, llm_trace, control="owner_revision_required")
                else:
                    tools._ctx._delivery_control_required = False
            return held("owner_followup")
        if provisional_assistant is not None and messages[-1] is provisional_assistant:
            messages.pop()
        if post_controls.get("finalize_now"):
            text, usage, forced_trace = _loop()._maybe_early_finalize(
                limit_ctx, tools, post_controls,
            )
            _loop()._merge_finalization_trace(llm_trace, forced_trace)
            return text, usage, llm_trace
        from supervisor.owner_stop import presence_stop_now
        if presence_stop_now(tools._ctx):  # Stop now met this Presence answer: kept on the row, never sent
            from ouroboros.loop_round_limits import _handle_direct_turn_hard_stop
            text, usage, forced_trace = _handle_direct_turn_hard_stop(limit_ctx)
            _loop()._merge_finalization_trace(llm_trace, forced_trace)
            return text, usage, llm_trace
        from ouroboros.place_intake import reopen_for_place_mail
        if not stopping and reopen_for_place_mail(tools._ctx, post_controls):
            return held("place_mail")  # keep the answer; the accepted place context gets a model round
    _loop()._project_child_result_dispositions(limit_ctx, llm_trace)
    # UNKNOWN evidence delivers a candidate retained over it as published; a KNOWN candidate is re-retained below.
    evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(tools, limit_ctx, llm_trace)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if (
        not stopping and isinstance(candidate, _loop().DeliveryCandidate)
        and (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        )
    ):
        if candidate.full_text:
            messages.append({"role": "assistant", "content": candidate.full_text})
        llm_trace["reasoning_notes"].append(
            "Delivery evidence changed after the answer was retained; a complete replacement answer is required."
            if evidence_fingerprint else
            "Delivery evidence can no longer be verified; the complete answer must be restated and is retained over unknown evidence."
        )
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return held("evidence_changed", "" if evidence_fingerprint else "unverifiable")
    if isinstance(candidate, _loop().DeliveryCandidate):
        _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text
    if stopping:
        request = tools._ctx._completion_selected
        record_stopped_completion(tools, limit_ctx, llm_trace, request, candidate)
        if not _seal_admission_before_delivery(tools, limit_ctx, llm_trace):
            return held("admission_seal")
    from ouroboros.place_reply import prepare_focus_reply
    allowed, refusal = prepare_focus_reply(tools._ctx, str(content or ""), quiet=bool(getattr(candidate, "quiet", False)))
    if not allowed:
        messages.append({"role": "user", "content": refusal})
        _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
        return held("focus_reply_refused")
    if isinstance(getattr(tools._ctx, "_presence_completion", None), dict):  # Accept only at this common exit.
        tools._ctx._presence_completion_accepted = True
        limit_ctx.accumulated_usage["presence_completion_outcome"] = tools._ctx._presence_completion["outcome"]
        limit_ctx.accumulated_usage["terminal_origin"] = _loop().TERMINAL_ORIGIN_MODEL_FINAL
    if getattr(candidate, "quiet", False):  # only this exit accepts a wake's silent finish
        limit_ctx.accumulated_usage.update(wake_silent=True, terminal_origin=_loop().TERMINAL_ORIGIN_MODEL_FINAL)
    if summary := (getattr(tools._ctx, "_completion_selected", None) or {}).get("wake_summary"): limit_ctx.accumulated_usage["wake_summary"] = summary
    if getattr(tools._ctx, "_owner_message_sent", False): limit_ctx.accumulated_usage["owner_message_sent"] = True
    from ouroboros.swarm_receipt import retain_comparison
    retain_comparison(tools._ctx)
    return _loop()._handle_text_response(str(content or ""), llm_trace, limit_ctx.accumulated_usage)


def finish_completed_stop(tools: ToolRegistry, ctx: Any, emit_progress: Any,
                          budget_remaining: float | None, ceiling: Any) -> Any:
    """Publish an already-authored stop without buying a wrap-up or parking for funds."""
    request = getattr(tools._ctx, "_completion_request", None)
    if not isinstance(request, dict) or not consume_completion_request(tools, ctx, ctx.llm_trace):
        return None
    if request.get("action") != "stop":
        return None
    controls = _loop()._drain_incoming_messages(ctx.messages, ctx.incoming_messages, ctx.drive_root,
        ctx.task_id, ctx.event_queue, ctx.owner_msg_seen, owner_ctx=tools._ctx, defer_content_ack=True)
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    if request["observation"].get("owner_source_sha256") != owner_source_sha256(tools._ctx):
        tools._ctx._completion_selected = None
        _loop()._arm_delivery_control(tools, ctx, ctx.llm_trace, control="owner_revision_required")
        observe_completion_hold(tools, ctx, "selection", "owner_input_changed")
        return None
    reason = str(controls.get("finalize_now") or "").splitlines()
    cause = reason[0].strip() if reason else ""
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, presence_stop_now
    # Stop now (a direct turn's control, a Presence turn's open intent) keeps this stop, unsent, before
    # a hard rail below could publish it without the ordinary admission's read.
    if cause == REASON_OWNER_STOPPED_DIRECT_TURN or presence_stop_now(tools._ctx):
        from ouroboros.loop_round_limits import _handle_direct_turn_hard_stop
        return _handle_direct_turn_hard_stop(ctx)
    waiter = getattr(tools._ctx, "model_wait_context", None)
    hard = waiter.control_reason() if waiter is not None else ""
    if not cause and hard in {"absolute_ceiling", "execution_deadline"}:
        cause = "deadline_local" if hard == "execution_deadline" else "finalization_grace"
    remaining = dispatch_window_remaining_sec(deadline_ts=getattr(ctx, "deadline_ts", None))
    if not cause and remaining is not None and remaining <= 0:
        cause = "deadline_local"
    if not cause and ctx.max_rounds is not None and ctx.round_idx > ctx.max_rounds:
        cause = "round_limit"
    from ouroboros.loop_budget import authored_completion_budget_exhausted
    if not cause and authored_completion_budget_exhausted(ctx, budget_remaining, ceiling):
        cause = "budget_exhausted"
    if cause:
        _loop()._finalize_forced_services(ctx, ctx.llm_trace)
        ctx.accumulated_usage.update(execution_status="failed", reason_code=cause)
        return _loop()._forced_fallback_result(ctx, ctx.llm_trace, tools._ctx._delivery_candidate.full_text,
            cause, source="authored_stop_at_hard_rail")
    return _loop()._no_tool_final_answer(tools._ctx._delivery_candidate.full_text, ctx, ctx.llm_trace,
        tools, ctx.incoming_messages, ctx.owner_msg_seen, emit_progress, explicit_candidate=True)


def record_stopped_completion(tools: ToolRegistry, ctx: Any, trace: dict, request: dict,
                              candidate: DeliveryCandidate) -> None:
    """A stop is the author's own act, truthful for every task; it grants no approval and buys no review."""
    del tools, ctx
    trace["task_completion"] = {"action": "stop", "rationale": request["rationale"],
        "answer_sha256": candidate.content_sha256, "source": request.get("source", "finish_task")}
