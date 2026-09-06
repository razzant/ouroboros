"""Review execution: how a reviewer slot is delivered.

This module is everything BELOW the review seam — delivery routes and the typed result handed
back, and each route's own prompt rendering. ``review_substrate`` keeps the
policy above the seam (attempt rails, persistence, parsing, actor projection,
quorum) and knows only that a route exists.

The dependency runs one way: this module never imports the coordinator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional

from ouroboros.review_slot_cancel import (  # noqa: F401 — re-exported seam surface
    ReviewSessionSucceededResultUnavailable,
    _cancel_honesty_clause,
    _interaction_outlives_slot,
    _natural_success_terminal,
    _slot_cancel_outcome,
)
from ouroboros.review_dispatch import bind_api_review_paid_stamp, invoke_review_paid_stamp
from ouroboros.usage_accounting import POSITIVE_PHYSICAL_ATTEMPT_STATES
from ouroboros.delegate_custody_usage import (
    observe_failed_review_send, observe_review_usage, session_usage_once,
)
from ouroboros.triad_review import (
    ACCEPTANCE_SURFACE_RULES,
    TIER_CLASSIFICATION_RULES,
    default_output_contract,
    review_output_shape,
)
from ouroboros.deadline_utils import (
    bounded_seconds, owner_deadline_exhausted,
    review_transport_timeout,
)
from ouroboros.config import get_finalization_grace_sec
from ouroboros.review_session_custody import (
    checkpoint_pending_invocation,
    owned_started_review_custody,
    review_recovery_facts,
)
from ouroboros.review_session_usage import (
    session_custody_attribution,
    session_invocation_fields,
)
if TYPE_CHECKING:  # annotations only — importing the substrate here would cycle
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

log = logging.getLogger("review_execution")
class ReviewRouteKind(str, Enum):
    """Closed set of review delivery routes.

    A route says HOW a reviewer slot is driven, never WHICH vendor tool runs it:
    ``api_chat`` is a chat-completions call, ``agent_session`` is a hosted agent
    session. There is deliberately no ``codex``/``claude``/``cursor`` member —
    per-harness knowledge belongs behind the transport, not in the substrate.
    """

    API_CHAT = "api_chat"
    AGENT_SESSION = "agent_session"


def delivery_retrieves(route: Any, subagent_id: Any) -> bool:
    """THE delivery-class predicate: does this reviewer row read the subject
    with its own tools (a hosted session, or a configured-subagent api row's
    native tool rounds) instead of receiving the assembled packet?

    One definition for every caller — slot properties, admission, packet fit
    and the surfaces' request builders — so a delivery class can never be
    recognised by one caller and missed by another. ``route`` may be a
    ``ReviewRouteKind`` or its wire string."""
    return (
        str(getattr(route, "value", route) or "") == ReviewRouteKind.AGENT_SESSION.value
        or bool(str(subagent_id or "").strip())
    )

class ReviewRouteUnavailable(RuntimeError):
    """Typed refusal for a route with no executor in this build.

    A missing route fails loudly on its own slot; it never falls back to another
    route, model, or profile. ``code`` is the machine-readable refusal vocabulary
    (a ``route_health`` reason or a site code); "" is an uncoded raise.
    """

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = str(code or "")

def _deadline_exhausted_error(
    message: str = "owner deadline leaves no dispatch window",
) -> ReviewRouteUnavailable:
    return ReviewRouteUnavailable(message, code="deadline_exhausted")

class ReviewSessionWaitingOnUser(RuntimeError):
    """A delegated review session parked on an interactive question (F18).

    Review slots are non-interactive by contract: nothing host-side answers a
    reviewer's AskUserQuestion, so waiting out the engine's answer timeout burns
    the whole slot budget in silence. The poller terminates the slot EARLY —
    cancelled through the verified-cancel path under the typed reason
    ``review_session_waiting_on_user`` — and this failure names the pending
    question plus the cancel's HONEST outcome (BR1-1): "host-cancelled" only on
    a ``confirmed`` verified receipt whose terminal is the cancel's own — a
    confirmed natural ``failed``/``interrupted`` is attributed to the run
    itself (BR2-2); anything unverified says the run may still be live, and a
    verify read that finds the run already SUCCEEDED never raises this at all
    (completion wins). Answering support for hosted review lanes is a
    deliberate non-goal (owner: no acceptance host-wait; see docs/ARCHITECTURE.md).
    """

def _poll_detail(gateway: Any, run_id: str, seconds: float) -> Dict[str, Any]:
    from ouroboros.delegate_progress import bounded_poll, expiring_poll
    if seconds > 0:
        return bounded_poll(gateway, run_id, seconds, strict=True)
    return expiring_poll(gateway, run_id, strict=True) or {}

_DELIVERY_RANK = {"api_chat": 0, "native_tool_rounds": 1, "agent_session": 2}


def slot_delivery(slot: Any) -> str:
    """The delivery a slot runs on — `api_chat` (packet), `native_tool_rounds`
    (an api row bound to a configured subagent) or `agent_session` — the same
    names the executors stamp on actor usage."""
    if getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION:
        return "agent_session"
    return "native_tool_rounds" if getattr(slot, "retrieves", False) else "api_chat"


def panel_delivery_class(slots: Any) -> str:
    """A panel's delivery CLASS is its slowest delivery: session over native
    over packet, and an empty panel is a packet panel. TELEMETRY classification
    only (owner R52) — it labels the timing row and paces nothing."""
    return max((slot_delivery(slot) for slot in slots or ()), key=_DELIVERY_RANK.__getitem__, default="api_chat")


# Policy keys a retrieving executor consumes itself (`review_native_episode`,
# `AgentSessionReviewExecutor`); the rendered Policy JSON omits them so the api
# pack states the review contract once, in its governance segment.
ROUTE_OWNED_POLICY_KEYS = frozenset({"output_contract", "native_data_root"})


def review_output_contract(request: ReviewRequest) -> str:
    """The surface's output contract — required keys, tier and acceptance rules,
    the DEGRADED escape hatch — as ONE text every delivery honours: the api pack
    renders it into its byte-stable governance segment, and a surface hands the
    same text to its retrieving rows as ``policy["output_contract"]``."""
    classify_tier = bool(request.policy.get("classify_outcome_tier"))
    # Acceptance-only prompt POLICY (criteria/dialogue/obligation keys and the
    # surface rules) keys on the surface; the output SHAPE those keys imply is
    # the separate form fact `review_output_shape` the canonicalizer consumes.
    acceptance = request.surface == "task_acceptance"
    # The tier keys belong in the REQUIRED key list, not trailing prose — models
    # honor the explicit "Return JSON with keys" list and otherwise drop them,
    # which silently kills the best_effort/completion-coach lexicon.
    tier_keys = (
        ', outcome_tier ("solved"|"best_effort"|"blocked_with_evidence"), completion_coach'
        if classify_tier
        else ""
    )
    # For task acceptance the reviewer makes its derived acceptance criteria
    # VISIBLE — recorded per-actor in the review trace / objective axis (M4) so
    # "for whom we review" is auditable. Reviewer reasoning, not a new
    # authoritative gate (criteria live in actors[].parsed, not a separate phase).
    criteria_key = (
        ', criteria_used (the acceptance criteria you re-derived from the full goal narrative '
        'and checked, as [{criterion, status (supported|missing|partial|rejected), evidence_refs}]; evidence_refs must name concrete '
        'host-attested receipts/artifacts/tool results for every contributing criterion; '
        # D-Q5: the host resolves each ref by EXACT match against the packet's
        # enumerable exhibit keys — the vocabulary below is the closed set of forms.
        'each evidence_ref is resolved by EXACT match against the evidence packet, so use these exact forms: '
        'claim ids from task_contract.acceptance_claims (which count as evidence ONLY while '
        'acceptance_support_refs shows that claim supported by a passing receipt — otherwise cite the exhibit itself), '
        'verification_receipts[i] receipt ids (rows of the verification_receipts exhibit list; '
        'only a green pass/observed receipt supports a criterion), acceptance_obligations ids, artifact manifest names, '
        'or HOST-ATTESTED top-level packet section names (the agent-supplied sections — reasoning_notes, '
        'candidate_answers, agent_supplied — and task_contract itself are NOT evidence: cite the exhibit '
        'that proves the work instead) — a ref that resolves to nothing cannot support a criterion)'
        if acceptance
        else ""
    )
    # v6.74.0 acceptance-dialogue keys (A3/A5): reviewer-authored obligation
    # identity and the typed dialogue judgement. Both live in the REQUIRED key
    # list for the same reason as the tier keys above.
    dialogue_key = (
        ', dialogue_status ("continue_actionable"|"unreachable_here"|"stable_disagreement")'
        if acceptance
        else ""
    )
    findings_shape = (
        '[{severity, item, evidence, recommendation, disposition_kind ("new"|"re_raise"), '
        'obligation_id (required when disposition_kind="re_raise")}]'
        if acceptance
        else "[{severity, item, evidence, recommendation}]"
    )
    tier_rules = TIER_CLASSIFICATION_RULES if classify_tier else ""
    acceptance_rules = ACCEPTANCE_SURFACE_RULES if acceptance else ""
    return (
        f"Return JSON with keys: verdict (PASS|FAIL|DEGRADED){tier_keys}{criteria_key}{dialogue_key}, findings "
        f"({findings_shape}), and summary. "
        + tier_rules
        + acceptance_rules
        + "If you cannot judge because evidence is missing, return DEGRADED and explain."
    )


def _render_prompt_parts(request: ReviewRequest, slot: ReviewSlot) -> tuple[str, str, str]:
    """Return (stable_governance, task_stable, dynamic_evidence) for one slot.

    Cache segmentation (v6.74.0, B1): the byte-stable governance instruction and
    the task-stable contract (goal/scope/checklist/policy — stable across the
    improvement passes of ONE task) are the two cache-marked segments; the
    mutable tail (subject, evidence, refs) is never marked, and the slot label
    lives at its TAIL so concurrent same-model slots share a warm prefix."""
    evidence = json.dumps(request.evidence, ensure_ascii=False, indent=2, default=str)
    refs = json.dumps(request.evidence_refs, ensure_ascii=False, indent=2, default=str)
    policy = json.dumps(
        {k: v for k, v in request.policy.items() if k not in ROUTE_OWNED_POLICY_KEYS},
        ensure_ascii=False, indent=2, default=str,
    )
    stable = (
        "You are an independent Ouroboros reviewer slot.\n"
        f"Surface: {request.surface}\n"
        f"Role hint: {slot.role_hint or 'general reviewer'}\n\n"
        "The review subject and evidence packet arrive in the user message.\n\n"
        + review_output_contract(request)
        + "\n\n"  # trailing separator: block-flattening providers glue segments
    )
    task_stable = (
        "Review goal:\n"
        f"{request.goal}\n\n"
        "Declared scope:\n"
        f"{request.scope or '(not specified)'}\n\n"
        "Checklist / acceptance criteria:\n"
        f"{request.checklist or '(none supplied)'}\n\n"
        "Policy:\n"
        f"{policy}"
        "\n\n"  # trailing separator inside the cache-marked segment (r1 #3)
    )
    dynamic = (
        "Subject:\n"
        f"{request.subject}\n\n"
        "Evidence refs:\n"
        f"{refs}\n\n"
        "Evidence packet:\n"
        f"{evidence}\n\n"
        # Slot identity stays at the TAIL of the mutable part so duplicate
        # same-model reviewer slots share one warm prefix for the whole prompt.
        f"Slot: {slot.slot_id}"
    )
    return stable, task_stable, dynamic


def _render_prompt(request: ReviewRequest, slot: ReviewSlot) -> str:
    """Flat compatibility view; segments carry their own trailing separators,
    so this equals what a block-flattening provider actually receives."""
    stable, task_stable, dynamic = _render_prompt_parts(request, slot)
    return stable + task_stable + dynamic


# Provider hard limit on declared cache breakpoints; asserted on every final payload (B1).
_MAX_PROMPT_CACHE_BREAKPOINTS = 4


def assert_cache_breakpoint_cap(messages: List[Dict[str, Any]]) -> None:
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            count += sum(
                1 for block in content
                if isinstance(block, dict) and block.get("cache_control")
            )
    if count > _MAX_PROMPT_CACHE_BREAKPOINTS:
        raise AssertionError(
            f"prompt declares {count} cache breakpoints "
            f"(cap {_MAX_PROMPT_CACHE_BREAKPOINTS})"
        )


def _request_messages(request: ReviewRequest, slot: ReviewSlot) -> List[Dict[str, Any]]:
    slot_messages = (request.slot_messages or {}).get(str(slot.slot_id or ""))
    source_messages = slot_messages if slot_messages is not None else request.messages
    if source_messages:
        messages = [
            dict(message) if isinstance(message, dict) else {"role": "user", "content": str(message)}
            for message in source_messages
        ]
        assert_cache_breakpoint_cap(messages)  # the cap covers EVERY final payload
        return messages
    # Default shape is cache-friendly (v6.74.0, B1): two cache-marked system
    # segments — the byte-stable governance instruction and the task-stable
    # contract (goal/scope/checklist/policy, unchanged across a task's
    # improvement passes) — followed by the unmarked mutable evidence tail as
    # the user message. The large evidence body changes every pass by design
    # and is honestly not cached.
    from ouroboros.tools.review_helpers import cached_prompt_blocks

    stable, task_stable, dynamic = _render_prompt_parts(request, slot)
    system_blocks = cached_prompt_blocks(stable)
    system_blocks.extend(cached_prompt_blocks(task_stable))
    messages = [
        {"role": "system", "content": system_blocks},
        {"role": "user", "content": dynamic},
    ]
    assert_cache_breakpoint_cap(messages)
    return messages


def _messages_char_count(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message
        if isinstance(content, list):
            total += sum(len(str(block.get("text", block))) if isinstance(block, dict) else len(str(block)) for block in content)
        else:
            total += len(str(content or ""))
    return total


@dataclass(frozen=True)
class ReviewAssignment:
    """Immutable slot job: same task and evidence, route-specific delivery."""

    request: ReviewRequest
    slot: ReviewSlot
    call_id: str = ""
    call_type: str = ""
    # Canonical/budget drive for delegated custody; the API route never reads it.
    custody_root: Any = None
    dispatch_stamp: Any = None

    @property
    def route(self) -> ReviewRouteKind:
        return self.slot.route


@dataclass(frozen=True)
class ReviewAttemptResult:
    """Typed outcome of ONE physical attempt, whatever the route."""

    message: Any
    usage: Dict[str, Any]
    raw_text: str


class ReviewSlotExecutor:
    """Route-specific delivery for one assignment.

    Owns ONLY transport: how the assignment is rendered for its route and how a
    single physical attempt is sent. Attempt policy, persistence, parsing, actor
    projection and quorum stay with ``ReviewCoordinator``.
    """

    route: ClassVar[ReviewRouteKind]

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        self.assignment = assignment
        self.llm = llm
        self.usage_observer: Optional[Callable[[Dict[str, Any]], None]] = None

    def _observe_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        observe_review_usage(self.usage_observer, usage)

    def _observe_failed_send(self, exc: BaseException) -> None:
        observe_failed_review_send(self.usage_observer, exc)

    def prompt_payload(self) -> Dict[str, Any]:
        """Route-owned projection of what will actually be sent (for the durable
        prompt record). Rendered lazily: a route that never builds the big API
        pack must never pay for building it."""
        raise NotImplementedError

    def prompt_chars(self) -> int:
        raise NotImplementedError

    def execute(self) -> ReviewAttemptResult:
        raise NotImplementedError

    def failure_custody(self) -> Dict[str, Any]:
        return {}

    def restore_custody(self, _state: Dict[str, Any]) -> None:
        return None

    def set_pending_invocation_checkpoint(
        self, _checkpoint: Optional[Callable[[str], None]],
    ) -> None:
        return None


class ApiChatReviewExecutor(ReviewSlotExecutor):
    """The chat-completions route: the historical ``LLMClient.chat`` path."""

    route = ReviewRouteKind.API_CHAT

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        super().__init__(assignment, llm=llm)
        self._messages: List[Dict[str, Any]] | None = None
        self._chat_kwargs: Dict[str, Any] | None = None

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Lazily rendered, then memoized: the prompt record and every physical
        attempt of this slot share ONE rendering, byte-identical to what the
        substrate has always produced."""
        if self._messages is None:
            self._messages = _request_messages(self.assignment.request, self.assignment.slot)
        return self._messages

    def prompt_payload(self) -> Dict[str, Any]:
        return {"messages": self.messages}

    def prompt_chars(self) -> int:
        return _messages_char_count(self.messages)

    def _kwargs(self) -> Dict[str, Any]:
        request, slot = self.assignment.request, self.assignment.slot
        if self._chat_kwargs is None:
            self._chat_kwargs = {
                "messages": self.messages,
                "model": slot.model,
                "reasoning_effort": slot.effort,
                "max_tokens": int(request.max_tokens or slot.max_tokens),
                "temperature": request.temperature if request.temperature is not None else slot.temperature,
                "no_proxy": bool(request.no_proxy),
                # Keep stable per-surface affinity; same-model slots intentionally share it.
                "cache_affinity": f"{request.surface}:{request.task_id or 'review'}",
                "use_local": bool(slot.use_local),
            }
        # Recompute this per physical send because the executor is reused for retries.
        self._chat_kwargs["timeout"] = review_transport_timeout(
            slot.model,
            getattr(slot, "transport_timeout_sec", None),
            getattr(request, "deadline_at", ""),
        )
        return self._chat_kwargs

    def execute(self) -> ReviewAttemptResult:
        chat_kwargs = self._kwargs()
        chat = getattr(self.llm, "chat", None)
        async_chat = getattr(self.llm, "chat_async", None)
        if not callable(chat) and not callable(async_chat):
            raise ReviewRouteUnavailable("api_chat client exposes no callable transport", code="api_chat_unavailable")
        deadline_at = str(getattr(self.assignment.request, "deadline_at", "") or "")
        if owner_deadline_exhausted(deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec()):
            raise _deadline_exhausted_error()
        with bind_api_review_paid_stamp(self.assignment.dispatch_stamp):
            try:
                if callable(chat):
                    msg, usage = chat(**chat_kwargs)
                else:
                    msg, usage = asyncio.run(async_chat(**chat_kwargs))
            except BaseException as exc:
                # A provider-ambiguous exception is positive evidence that the
                # physical boundary was crossed even when a test adapter or old
                # transport did not enter usage_accounting's canonical marker.
                # The coordinator wraps raw stamps once-only, so this fallback
                # cannot double-charge a route that already marked dispatch.
                capture = getattr(exc, "physical_attempt_capture", None)
                if str(getattr(capture, "state", "") or "") in POSITIVE_PHYSICAL_ATTEMPT_STATES:
                    invoke_review_paid_stamp(self.assignment.dispatch_stamp)
                self._observe_failed_send(exc)
                raise
        # Null/non-object provider messages follow the caller's empty-response rail.
        raw_text = str(msg.get("content") or "") if isinstance(msg, dict) else ""
        self._observe_usage(usage)
        return ReviewAttemptResult(message=msg, usage=usage, raw_text=raw_text)


# ---------------------------------------------------------------------------
# Route configuration.
#
# Per-row delivery lives in the structured reviewer-slot SSOT
# (``OUROBOROS_REVIEWER_SLOTS`` — D14/6.1); the phase-5 per-row route envs
# (``OUROBOROS_REVIEW_ROUTES`` / ``OUROBOROS_SCOPE_REVIEW_ROUTES``) are
# RETIRED settings keys (ABI-10) and are ignored — a row built outside the
# structured config is pinned ``api_chat`` explicitly. The one surviving key
# below names the shared session target as an OPAQUE
# ``harness[=model][:effort]`` spec (Claudexor's own reviewer-panel spelling —
# no codex/claude/cursor member anywhere in this module).
# ---------------------------------------------------------------------------

REVIEW_SESSION_ROUTE_ENV = "OUROBOROS_REVIEW_SESSION_ROUTE"


def review_session_route() -> Any:
    """The configured session target for delegated review slots (or ``None``).

    Reuses the subagent-harness spelling and parser verbatim; when the review
    key is unset the subagent route is the target, so an owner who configured
    ONE delegated route does not have to configure it twice.
    """
    from ouroboros.subagents import get_subagent_harness, parse_subagent_harness

    raw = str(os.environ.get(REVIEW_SESSION_ROUTE_ENV, "")).strip()
    route = parse_subagent_harness(raw)
    if route is not None: return route
    if raw and raw.lower() != "off":
        # Same silent-typo class as the subagent key's reader: a non-empty value
        # that parses to nothing would quietly re-route review sessions onto the
        # subagent route as if the review key were never set.
        log.warning(
            "%s is set but unparseable (%r) — review sessions are OFF until it "
            "reads harness[=model][:effort]",
            REVIEW_SESSION_ROUTE_ENV, raw)
    return None if raw else get_subagent_harness()


# ---------------------------------------------------------------------------
# Typed verdict for a delegated session (D19 / plan 5.4).
# ---------------------------------------------------------------------------

# The ASK: sent as ``outputSchema`` only when the EFFECTIVE route can carry it
# (D19) — judged on the pinned harness's live manifest, never on the static
# adapter flag alone, because the flag describes the adapter and not the
# transport this run actually rides. The run's own reported
# ``outputConformance == "passed"`` is the only thing that lets the structured
# payload be TRUSTED as the verdict (never run success).
REVIEW_SESSION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["item", "verdict", "severity", "reason"],
                "properties": {
                    "item": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                    "severity": {"type": "string", "enum": ["critical", "advisory"]},
                    "reason": {"type": "string"},
                    "obligation_id": {"type": "string"},
                },
            },
        },
    },
}


# The OBJECT contract's schema (task acceptance): the whole verdict object. The
# host keeps exact evidence-ref resolution and tier/coach demotion for itself
# (an enum of every exhibit key would mint a schema larger than the packet), so
# the schema pins only the shape the canonicalizer preserves whole.
ACCEPTANCE_SESSION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "findings", "summary"],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "DEGRADED"]},
        "outcome_tier": {"type": "string", "enum": ["solved", "best_effort", "blocked_with_evidence"]},
        "completion_coach": {"type": "string"},
        "criteria_used": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["criterion", "status"],
                "properties": {
                    "criterion": {"type": "string"},
                    "status": {"type": "string", "enum": ["supported", "missing", "partial", "rejected"]},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "dialogue_status": {
            "type": "string",
            "enum": ["continue_actionable", "unreachable_here", "stable_disagreement"],
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "item", "evidence", "recommendation"],
                "properties": {
                    "severity": {"type": "string"},
                    "item": {"type": "string"},
                    "evidence": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "disposition_kind": {"type": "string", "enum": ["new", "re_raise"]},
                    "obligation_id": {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}


def review_session_output_schema(surface: str) -> Optional[Dict[str, Any]]:
    """The session verdict schema, shaped to the SURFACE's own clean contract.

    The shared schema admits ``{"findings": []}`` — the honest clean verdict for a
    triad or ordinary advisory reviewer. Scope's coverage contract requires all
    checklist rows (PASS included); Skill Review has the same matrix shape. Their
    schemas demand ``minItems: 1`` so an engine cannot conform with an empty answer;
    each surface's downstream parser still verifies exact item coverage. An
    ``object``-shaped surface (task acceptance) asks for the whole verdict object; a
    ``report`` surface asks for NO schema — its prose passes through verbatim.
    """
    shape = review_output_shape(surface)
    if shape == "report":
        return None
    if shape == "object":
        return ACCEPTANCE_SESSION_OUTPUT_SCHEMA
    if surface == "plan_review":
        # plan review's own element contract (4e133c8a): the generic item/verdict shape
        # would conform-and-launder — an unknown class demotes to a note.
        from ouroboros.tools.plan_spec import PLAN_REVIEW_SESSION_OUTPUT_SCHEMA

        return PLAN_REVIEW_SESSION_OUTPUT_SCHEMA
    if surface not in {"scope_review", "skill_review"}:
        return REVIEW_SESSION_OUTPUT_SCHEMA
    shaped = json.loads(json.dumps(REVIEW_SESSION_OUTPUT_SCHEMA))
    shaped["properties"]["findings"]["minItems"] = 1
    return shaped

# Verdict canonicalization lives in its own module (altitude, P7): both the
# session route and the native tool-round route consume it.
from ouroboros.review_verdict_extraction import (  # noqa: F401 — re-exported seam surface
    _EXTRACT_MAX_CHARS,
    _UNEXTRACTABLE,
    _extract_verdict_via_light_model,
    _findings_array,
    _strictly_parseable,
    canonicalize_session_verdict,
)


# ---------------------------------------------------------------------------
# The agent-session route: a delegated read-only Claudexor run per slot.
# ---------------------------------------------------------------------------

# Layered by Claudexor into the harness's native system-prompt channel — a
# statement of role, not the enforcement (the enforcement is the readonly
# access profile the engine derives).
_REVIEW_SESSION_INSTRUCTIONS = (
    "You are a delegated read-only REVIEWER session. Retrieve the evidence "
    "yourself with the tools your read-only mode actually gives you inside this "
    "repository root: read files and search, and — only if command execution is "
    "among them, which read-only frequently withholds — read-only git commands "
    "(git diff --cached, git log, git show). Reading the tree directly is a "
    "complete substitute; a review is not incomplete for lacking a shell. Do "
    "not modify anything and do not run history-moving git commands. Your final "
    "answer must follow the output contract in the task EXACTLY; your host "
    "parses it structurally, and prose around the verdict is a non-response."
)

_SESSION_POLL_SEC = 3.0
_CLAUDEXOR_MAX_SECONDS = 604_800


def _retire_orphaned_review_registration(
    custody: Any, gateway: Any, custody_drive: Any, project_id: str, *,
    definite_refusal: bool, reason: str, invocation_id: str = "",
    surface: str = "", slot_id: str = "",
) -> bool:
    """Retire a registration this start created but never bound to a run.

    Same semantics as the delegated-start path (ported from cxi/p34-converged
    ``tools/delegate._retire_orphaned_registration``, adapted to the custody-drive
    signature this module has): the registration is destroyed ONLY on a DEFINITE
    negative answer from the daemon. A transport error, a 5xx, or a 2xx handle with
    no run id all leave the POST's fate UNKNOWN, and a run may well be live against
    this very registration — an unverified outcome is never grounds for destroying
    state. The START_FAILED row is written either way, so every failing branch
    reaches ONE path instead of one retiring while its twin silently abandons.
    """
    retired = False
    if project_id and definite_refusal:
        try:
            gateway.remove_project(project_id)
            retired = True
        except Exception as exc:
            # A registration the daemon does not have is already retired — the same
            # absence-is-discharge fact retire_project settles on.
            retired = bool(custody.daemon_says_absent(exc))
            if not retired:
                log.warning("Failed to retire orphaned review project %s",
                            project_id, exc_info=True)
    custody.emit(custody_drive, custody.START_FAILED, {
        "run_id": "", "project_id": project_id, "project_retired": retired,
        "reason": reason, "invocation_id": invocation_id,
        "definite": bool(definite_refusal), "surface": surface, "slot_id": slot_id,
        "project_retention_reason": (
            "" if (retired or definite_refusal) else "start_outcome_unknown_run_may_exist"
        ),
    })
    return retired
@dataclass(frozen=True)
class SessionInvocation:
    """WHO is asking and HOW it should be delivered, as one immutable value.

    The same parameter-object pattern as ``ReviewAssignment``: these knobs are the
    caller's identity and delivery policy, they always travel together, and passing
    them as nine parallel keyword arguments put the function past the parameter
    budget while inviting a silent mis-pairing (a slot id from one row beside
    another row's route). ``retry_state`` is deliberately the caller's OWN mutable
    dict — the pending invocation id must survive back out to whoever owns the
    permitted retry — so this value freezes the reference, not the contents.
    """

    task_id: str
    surface: str
    slot_id: str
    timeout_sec: float
    logical_key_extra: tuple = ()
    output_schema: Optional[Dict[str, Any]] = None
    session_route: Any = None
    instructions: str = _REVIEW_SESSION_INSTRUCTIONS
    retry_state: Optional[Dict[str, Any]] = None
    reconcile_only: bool = False
    use_thread: bool = False
    thread_id: str = ""
    dispatch_stamp: Any = None
    operation_id: str = ""
    pending_invocation_checkpoint: Optional[Callable[[str], None]] = None
    owner_deadline_at: str = ""

def run_delegated_review_session(
    *,
    prompt: str,
    root: str,
    custody_drive: Any,
    invocation: SessionInvocation,
) -> Dict[str, Any]:
    """Start, watch, settle and collect one delegated read-only review.
    This is every review surface's single session transport. It pins one
    subscription harness, asks for schema only when the effective adapter can
    carry it, stores the canonical start request before POST, and replays only
    an explicit pending invocation token. A bound token joins its existing run;
    reconcile-only mode never mints a replacement. The nanny owns verified
    cancellation at ``timeout_sec`` and reads the full primary output before
    settling through ``delegate_custody``.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.gateways.claudexor import (
        WINDOW_EXHAUSTED_CODES, ClaudexorSubscriptionWindowExhausted, ClaudexorUnavailable, final_attempt_facts,
    )
    from ouroboros.subagents import delegated_run_shape, route_health
    from ouroboros.usage_accounting import current_usage_scope
    task_id, surface, slot_id, timeout_sec, logical_key_extra, output_schema, \
        session_route, instructions, retry_state = session_invocation_fields(invocation)
    owner_deadline_at = invocation.owner_deadline_at
    use_thread = bool(invocation.use_thread)
    thread_id = str(invocation.thread_id or "")
    turn_id = ""
    _scope = current_usage_scope()
    root_task_id, parent_task_id, usage_custody = session_custody_attribution(_scope)
    shape = delegated_run_shape(False)  # a reviewer reads and answers
    state = retry_state if retry_state is not None else {}
    run_id, run_request, invocation_id = "", None, ""
    started_custody = None
    retry_token = str(state.get("pending_invocation_id") or "")
    record = custody.invocation_record(custody_drive, retry_token) if retry_token else None
    if record is not None and record["state"] == "started" and record["run_id"]:
        run_id, started_custody = owned_started_review_custody(
            custody, custody_drive, record, task_id)
        run_request, invocation_id = record.get("request"), retry_token
    elif (record is not None and record["state"] == "pending"
          and isinstance(record.get("request"), dict) and record["request"]):
        run_request, invocation_id = record["request"], retry_token
    recovering = bool(run_id) or run_request is not None
    if retry_token and not recovering and surface != "skill_review":
        raise ReviewRouteUnavailable(
            "delegated retry token has no durable invocation; refusing a second paid run",
            code="review_custody_lost",
        )
    if recovering:
        route, project_id, existing_project, key, schema_asked = (
            review_recovery_facts(
                record, run_request, started_custody, prompt=prompt, root=root,
                claimant_task_id=task_id, claimant_surface=surface,
                claimant_slot_id=slot_id,
                claimant_operation_id=str(invocation.operation_id or ""))
        )
        thread_id = str(run_request.get("_thread_id") or thread_id)
        use_thread = bool(thread_id or run_request.get("_use_thread"))
    else:
        route = session_route if session_route is not None else review_session_route()
        if route is None:
            raise ReviewRouteUnavailable(
                "delegated review session has no configured session route "
                f"({REVIEW_SESSION_ROUTE_ENV} / OUROBOROS_SUBAGENT_HARNESS are empty or `off`)",
                code="session_route_unconfigured")
        project_id, existing_project, key, schema_asked = "", "", "", False
    if invocation.reconcile_only and not recovering:
        raise ReviewRouteUnavailable(
            "the exact delegated review invocation is no longer available for "
            "reconciliation; refusing to start a second paid run",
            code="review_custody_lost",
        )
    gateway = ensure_owned_gateway()
    try:
        if not recovering and not run_id and not (use_thread and thread_id):
            unavailable, reset_at = route_health(
                gateway, route.route_id, shape, route_model=route.model,
                pinned_profile=str(getattr(route, "profile_id", "") or ""),
            )
            if unavailable in WINDOW_EXHAUSTED_CODES or reset_at:
                raise ClaudexorSubscriptionWindowExhausted(
                    "delegated review route subscription window is exhausted"
                    + (f" (resets {reset_at})" if reset_at else "")
                    + "; this slot fails typed — never a silent fallback onto "
                    "metered API spend", reset_at=reset_at,
                    code=(unavailable or "subscription_window_exhausted"))
            if unavailable:
                raise ReviewRouteUnavailable(
                    f"delegated review route unavailable: {unavailable}", code=unavailable)
        if not recovering:
            existing_project = gateway.find_project_id(root)
            project_id = existing_project or gateway.register_project(root)
            schema_asked = bool(output_schema) and _effective_route_carries_schema(
                gateway, route.route_id)
            key = custody.idempotency_key(
                "review_slot", surface, slot_id, task_id, *logical_key_extra,
                route.route_id, shape.access, shape.mode, root, prompt,
            )
            if use_thread and not thread_id:
                from ouroboros.review_thread_continuity import ensure_review_thread
                thread_id = ensure_review_thread(
                    gateway, custody, thread_id, route=route, root=root,
                    surface=surface, slot_id=slot_id, task_id=task_id)
                existing_project = project_id
            invocation_id = custody.new_invocation_id()
            seconds = bounded_seconds(
                timeout_sec, default=300, maximum=_CLAUDEXOR_MAX_SECONDS,
            )
            run_request = {
                "prompt": prompt,
                "instructions": instructions,
                "authPreference": "subscription",
                "mode": shape.mode,
                "access": shape.access,
                "scope": {"kind": "project", "root": root},
                # A one-element explicit pool is the pin; primaryHarness is only preference.
                "harnesses": [route.route_id],
                "primaryHarness": route.route_id,
                "maxSeconds": seconds,
            }
            if use_thread:
                run_request["_use_thread"] = True
                run_request["_thread_id"] = thread_id
            if route.model:
                run_request["model"] = route.model
            if route.effort:
                run_request["effort"] = route.effort
            if use_thread or getattr(route, "profile_id", ""):
                run_request["credentialProfileId"] = getattr(route, "profile_id", "") or None
            if schema_asked:
                run_request["outputSchema"] = output_schema
        if not run_id:
            if (not recovering and owner_deadline_at and owner_deadline_exhausted(
                deadline_at=owner_deadline_at, reserve_sec=get_finalization_grace_sec())):
                raise _deadline_exhausted_error()
            seconds = bounded_seconds(
                run_request.get("maxSeconds"),
                default=timeout_sec if timeout_sec is not None else 300,
                maximum=_CLAUDEXOR_MAX_SECONDS,
            )
            invoke_review_paid_stamp(invocation.dispatch_stamp)
            requested = custody.record_start_requested(
                custody_drive, run_id="", task_id=task_id,
                idempotency_key=key, invocation_id=invocation_id,
                operation_id=str(invocation.operation_id or ""),
                max_seconds=seconds, request=run_request, project_id=project_id,
                project_owned=not existing_project, route=route.route_id,
                surface=surface, slot_id=slot_id,
                # #112: pending recovery replays the request row's lineage.
                root_task_id=root_task_id, parent_task_id=parent_task_id,
                **usage_custody,
            )
            if not requested:
                # No durable request means no POST; only a fresh registration is retirable.
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    definite_refusal=not recovering,
                    reason="start_request_row_unwritable",
                    invocation_id=invocation_id, surface=surface, slot_id=slot_id,
                )
                raise ReviewRouteUnavailable(
                    "the durable start-request row could not be written; the "
                    "delegated review session was NOT started", code="start_request_row_unwritable")
            state["pending_invocation_id"] = invocation_id
            checkpoint_pending_invocation(
                checkpoint=invocation.pending_invocation_checkpoint, invocation_id=invocation_id,
                state=state, on_failure=lambda: _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id if not existing_project else "",
                    definite_refusal=not recovering,
                    reason="review_custody_checkpoint_unwritable",
                    invocation_id=invocation_id, surface=surface, slot_id=slot_id))
            try:
                if use_thread:
                    from ouroboros.review_thread_continuity import start_review_thread_turn
                    handle = start_review_thread_turn(
                        gateway, thread_id, run_request, idempotency_key=invocation_id)
                else:
                    handle = gateway.start_run(run_request, idempotency_key=invocation_id)
            except ClaudexorUnavailable as exc:
                status = int(getattr(exc, "status_code", 0) or 0)
                definite = 400 <= status < 500
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    # Only a definite 4xx proves the registration never bound a run.
                    definite_refusal=definite and not recovering,
                    reason=exc.code, invocation_id=invocation_id,
                    surface=surface, slot_id=slot_id,
                )
                if not definite:
                    # Unknown outcome retains the token for exact replay.
                    state["pending_invocation_id"] = invocation_id
                else:
                    state.pop("pending_invocation_id", None)
                raise
            run_id = str(handle.get("runId") or handle.get("jobId") or "")
            turn_id = str(handle.get("turnId") or "")
            if not run_id:
                # A successful POST retains the registration on malformed response.
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    definite_refusal=False, reason="queued_without_run_id",
                    invocation_id=invocation_id, surface=surface, slot_id=slot_id,
                )
                state["pending_invocation_id"] = invocation_id
                raise ReviewRouteUnavailable(
                    f"Claudexor returned a queued handle without a run id: {handle!r}", code="queued_without_run_id")
        state["pending_invocation_id"] = invocation_id or retry_token
        state["delegated_run_id"] = run_id
        if started_custody is not None:
            entry = started_custody
            custody_durable = True
        else:
            entry = custody.RunCustody(
                run_id=run_id, task_id=task_id,
                route_id=route.route_id, model=str(route.model or ""),
                profile_id=str(getattr(route, "profile_id", "") or ""),
                project_id=project_id, project_owned=not existing_project,
                root_task_id=root_task_id, parent_task_id=parent_task_id,
                **usage_custody,
                ledger_root=str(custody_drive), idempotency_key=key,
                invocation_id=invocation_id or retry_token,
            )
            # A missing started row leaves the run process-local and unresumable.
            custody_durable = bool(custody.record_started(custody_drive, entry, shape={
                "effort": route.effort, "access": shape.access, "mode": shape.mode,
                "isolation": shape.isolation, "delegated": shape.delegated,
                "root": root, "surface": surface, "slot_id": slot_id,
            }))
        try:
            detail = _poll_session_terminal(
                gateway, custody, custody_drive, entry, run_id,
                float(timeout_sec) if timeout_sec is not None else 300.0,
            )
        except ClaudexorUnavailable:
            # A started run with an unreadable terminal state is still paid work.
            # Preserve the exact durable invocation for the permitted retry rather
            # than POSTing a second review against the same slot.
            state["pending_invocation_id"] = invocation_id or retry_token
            raise
        settlement = custody.settle_run(custody_drive, gateway, entry, detail)
        summary = custody.summary_of(detail)
        observed = final_attempt_facts(detail, run_id)
        run_state = str(summary.get("state") or "")
        if run_state != "succeeded":
            failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
            message = (f"delegated review session {run_id} ended {run_state or 'unknown'}"
                       + (f": {json.dumps(failure, ensure_ascii=False)}" if failure else ""))
            code = str(failure.get("code") or "")
            state.pop("pending_invocation_id", None)
            state.pop("delegated_run_id", None)
            if code in WINDOW_EXHAUSTED_CODES:
                raise ClaudexorSubscriptionWindowExhausted(
                    message, reset_at=str(failure.get("resetsAt") or ""), code=code)
            raise ClaudexorUnavailable(code or f"run_{run_state or 'unknown'}", message)
        text = _full_session_text(gateway, run_id, detail)
        spend, estimated = custody.disclosed_spend(summary)
        thread_receipt: Dict[str, Any] = {}
        if use_thread:
            from ouroboros.review_thread_continuity import review_thread_receipt as receipt_for
            thread_receipt = receipt_for(gateway, thread_id, run_id, turn_id,
                expected_profile=str(getattr(route, "profile_id", "") or ""),
                applied_profile=observed.get("profile_id", ""))
            turn_id = str(thread_receipt.get("turn_id") or turn_id)
        state.pop("pending_invocation_id", None)
        state.pop("delegated_run_id", None)
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "thread_receipt": thread_receipt,
            "profile_continuity_receipt": thread_receipt.get("profile_continuity") or {},
            "text": text,
            "conformance": str(summary.get("outputConformance") or "").strip().lower(),
            "schema_asked": schema_asked,
            "custody_durable": custody_durable,
            "idempotent_recovery": recovering,
            "settlement": settlement,
            "route_id": str(entry.route_id),
            # One final attempt, never the requested pool or a mixed summary route.
            "effective_route_ids": [observed["harness_id"]] if observed.get("harness_id") else [],
            "observed_attempt": observed,
            "model": observed.get("model", ""),
            "spend": spend,
            "spend_estimated": estimated,
            # D22/D29 applied facts are verbatim telemetry, never inferred.
            "applied_profile": observed.get("profile_id", ""),
            "auth_route_receipt": summary.get("authRoute") or {},
            # Only effectiveAccess witnesses applied access; request echo is insufficient.
            "applied_access": str(summary.get("effectiveAccess") or ""),
        }
    except BaseException as exc:
        if run_id:
            setattr(exc, "delegated_run_started", True)
            setattr(exc, "delegated_run_id", run_id)
        raise
    finally:
        gateway.close()
def _effective_route_carries_schema(gateway: Any, route_id: str) -> bool:
    """Can the EFFECTIVE route actually carry ``outputSchema`` on this run (D19)?

    Judged on the pinned route's live manifest (``GET /v2/harnesses``), NOT on
    the agent-capability catalog: the catalog's harness rows carry no
    structured-output field at all, so the old catalog read was constantly
    False and the preferred D19 path was dead in production. Two manifest
    facts must BOTH hold before asking:

    - ``capabilities.json_schema_output`` — the adapter has a native
      structured-output flag at all;
    - NOT ``capabilities.interactive`` — the daemon always arms an interaction
      channel, and the engine refuses ``outputSchema`` on interactive-transport
      lanes outright (a typed preflight refusal that would kill the whole
      review run, not degrade it — orchestrator DT2.1-16).

    This mirrors the engine's own preflight gate, so asking never turns into a
    refused run. Deciding to ASK is all this answers; trust still comes solely
    from the run's own ``outputConformance == "passed"`` afterwards."""
    try:
        for row in gateway.harnesses():
            if not isinstance(row, dict) or str(row.get("id") or "") != str(route_id):
                continue
            manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
            caps = manifest.get("capabilities") if isinstance(manifest.get("capabilities"), dict) else {}
            return bool(caps.get("json_schema_output")) and not bool(caps.get("interactive"))
    except Exception:
        log.debug("harness manifest read failed", exc_info=True)
    return False
def _poll_session_terminal(gateway: Any, custody: Any, custody_drive: Any, entry: Any,
                           run_id: str, seconds: float) -> Dict[str, Any]:
    """Poll a delegated review run on the slot clock; verified cancel and
    completion-wins semantics remain owned by the existing cancel seam."""
    from ouroboros.gateways.claudexor import pending_interactions as _cx_pending
    deadline = time.monotonic() + max(0.0, float(seconds))
    detail = _poll_detail(gateway, run_id, max(0.0, deadline - time.monotonic()))
    while not custody.is_terminal(detail):
        pending = _cx_pending(detail)
        if (pending or bool(custody.summary_of(detail).get("waitingOnUser"))) \
                and _interaction_outlives_slot(
                    (pending[0] if pending else {}).get("timeout_at"), deadline):
            first = pending[0] if pending else {}
            question = ""
            for q in first.get("questions") or []:
                question = str(q.get("question") or "").strip()
                if question:
                    break
            outcome, state, carried = _slot_cancel_outcome(
                gateway, custody, custody_drive, entry, run_id,
                "review_session_waiting_on_user")
            settled = _natural_success_terminal(gateway, custody, run_id, state, carried)
            if settled is not None:
                return settled
            named = str(first.get("interaction_id") or "")
            raise ReviewSessionWaitingOnUser(
                f"delegated review session {run_id} paused on an interactive question"
                + (f" ({named}: {question[:300]!r})" if named or question else "")
                + " — review slots are non-interactive, so the slot terminated "
                  "early and typed ("
                + _cancel_honesty_clause(outcome, state)
                + ") instead of silently burning its whole budget waiting"
            )
        if time.monotonic() >= deadline:
            outcome, state, carried = _slot_cancel_outcome(
                gateway, custody, custody_drive, entry, run_id, "review_slot_timeout")
            settled = _natural_success_terminal(gateway, custody, run_id, state, carried)
            if settled is not None:
                return settled
            raise TimeoutError(
                f"delegated review session {run_id} exceeded the slot budget "
                f"of {seconds:g}s ("
                + _cancel_honesty_clause(outcome, state) + ")"
            )
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            continue
        time.sleep(min(_SESSION_POLL_SEC, remaining))
        remaining = max(0.0, deadline - time.monotonic())
        detail = _poll_detail(gateway, run_id, remaining)
    return detail
def _full_session_text(gateway: Any, run_id: str, detail: Dict[str, Any]) -> str:
    """The session's final answer from the verified FULL primary output (D7).

    The resolver fetches and verifies the full artifact when the engine reports
    truncation; an unresolvable full text refuses rather than judging a
    head-cut transcript."""
    from ouroboros.delegate_output import _resolve_full_primary_output

    primary = detail.get("primaryOutput")
    primary, full_ok, disclosure = _resolve_full_primary_output(gateway, run_id, primary)
    if not full_ok:
        raise RuntimeError(
            f"delegated review session {run_id} produced a truncated primary "
            f"output whose full artifact could not be matched to the size or the "
            f"preview the run reported ({(disclosure or {}).get('reason')}); a "
            "verdict is never read from a preview"
        )
    text = ""
    if isinstance(primary, dict):
        text = str(primary.get("text") or "")
    elif primary is not None:
        text = str(primary)
    if not text.strip():
        final_summary = detail.get("finalSummary")
        text = final_summary if isinstance(final_summary, str) else ""
    return text
class AgentSessionReviewExecutor(ReviewSlotExecutor):
    """One pinned Claudexor run per reviewer slot.

    The coordinator owns policy; this executor never restarts for format repair.
    """
    route = ReviewRouteKind.AGENT_SESSION

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        super().__init__(assignment, llm=llm)
        self._session_prompt: Optional[str] = None
        self._raw_transcript: Optional[str] = None
        self._conformance_passed = False
        self._run_id = ""
        self._session_usage: Dict[str, Any] = {}
        self._session_usage_observed = False
        self._deltas: List[Dict[str, Any]] = []
        # Unknown starts retain the exact invocation token for the permitted retry.
        self._retry_state: Dict[str, Any] = {}
        self._pending_invocation_checkpoint: Optional[Callable[[str], None]] = None
        # A settled run failure is replayed rather than billed twice.
        self._settled_failure: Optional[BaseException] = None

    # -- prompt (route-owned; never the api pack) ------------------------------

    def _output_contract(self) -> str:
        contract = str((self.assignment.request.policy or {}).get("output_contract") or "")
        return contract or default_output_contract(review_output_shape(self.assignment.request.surface))

    def prompt_payload(self) -> Dict[str, Any]:
        return {"session_prompt": self.session_prompt}

    def prompt_chars(self) -> int:
        return len(self.session_prompt)

    @property
    def session_prompt(self) -> str:
        """The compact task this route sends — the slot's own work order when the
        surface supplies one (``slot_session_tasks``), else the shared
        ``session_task``: the SAME task, criteria and output contract as the api
        pack, minus the assembled evidence unless the surface chose to include
        it — a delegated reviewer retrieves context with its tools (D12), so
        the api pack is never assembled here (plan 5.2)."""
        if self._session_prompt is None:
            request, slot = self.assignment.request, self.assignment.slot
            task = str((getattr(request, "slot_session_tasks", None) or {}).get(slot.slot_id)
                       or request.session_task or "").strip()
            if not task:
                raise ReviewRouteUnavailable(
                    "agent_session slot has no session task: the surface must supply "
                    "the route-owned task text (request.session_task) — the assembled "
                    "api pack is deliberately not sendable to a session", code="session_task_missing")
            parts = [
                "You are an independent Ouroboros reviewer slot running as a "
                "read-only agent session.",
                f"Surface: {request.surface}",
                f"Role hint: {slot.role_hint or 'general reviewer'}",
                "",
                task,
                "",
                "OUTPUT CONTRACT (your host parses this structurally):",
                self._output_contract() + "\nThis contract governs the unwrapped substantive deliverable; emit any host-required transport metadata outside it exactly as separately instructed.",
                f"Slot: {slot.slot_id}",
            ]
            self._session_prompt = "\n".join(parts)
        return self._session_prompt
    # -- delivery --------------------------------------------------------------

    def execute(self) -> ReviewAttemptResult:
        if self._raw_transcript is not None:
            # Plan 5.5: the permitted resend repairs FORMAT locally over the
            # collected transcript; it never launches a second session.
            return self._verdict_result(force_extraction=True)
        if self._settled_failure is not None:
            # Pre-start transients retain a pending invocation and do not land here.
            raise self._settled_failure
        try:
            self._run_session()
        except BaseException as exc:
            self._run_id = self._run_id or str(getattr(exc, "delegated_run_id", "") or "")
            started = bool(self._run_id or getattr(exc, "delegated_run_started", False))
            if started and not self._session_usage_observed and session_usage_once(self._run_id):
                self._observe_usage({
                    "provider": "claudexor", "resolved_model": "",
                    "delegated_run_started": True, "delegated_run_id": self._run_id, "cost": None,
                })
                self._session_usage_observed = True
            if not self._retry_state.get("pending_invocation_id"):
                self._settled_failure = exc
            raise
        if not self._session_usage_observed and session_usage_once(self._run_id):
            self._observe_usage(self._session_usage)
        self._session_usage_observed = True
        return self._verdict_result()

    def failure_custody(self) -> Dict[str, Any]:
        failure = self._settled_failure
        run_id = self._run_id or str(getattr(failure, "delegated_run_id", "") or "")
        pending = str(self._retry_state.get("pending_invocation_id") or "")
        return {"delegated_run_started": bool(run_id), "delegated_run_id": run_id,
                "pending_invocation_id": pending}

    def restore_custody(self, state: Dict[str, Any]) -> None:
        # The logical waiter and the physical worker share this small mutable
        # custody cell so a timeout actor can durably carry a just-started run.
        self._retry_state = state

    def set_pending_invocation_checkpoint(
        self, checkpoint: Optional[Callable[[str], None]],
    ) -> None:
        # Captured by the physical worker before the logical caller may return.
        # Commit review uses it to patch the exact reserved slot before POST.
        self._pending_invocation_checkpoint = checkpoint
    def _session_route(self) -> Any:
        # 6.1: a structured row carries ITS OWN opaque target; the shared
        # session-route key stays as the legacy fallback for rows without one.
        spec = str(getattr(self.assignment.slot, "session_target", "") or "")
        if spec:
            import dataclasses
            from ouroboros.subagents import parse_subagent_harness

            route = parse_subagent_harness(spec)
            if route is None:
                raise ReviewRouteUnavailable(
                    f"agent_session slot {self.assignment.slot.slot_id} has an "
                    f"unparsable session target {spec!r}", code="session_target_unparsable")
            # D1/6.3: effort has ONE source — the per-slot effort field. The
            # target_id carries route identity only; any effort a caller
            # embedded in the spec (`harness=model:effort`) is dropped so the
            # field can never be silently overridden by the identity string.
            route = dataclasses.replace(route, effort=str(self.assignment.slot.effort or ""))
            pin = str(getattr(self.assignment.slot, "session_profile", "") or "")
            if pin:
                route = dataclasses.replace(route, profile_id=pin)
            return route
        route = review_session_route()
        if route is None:
            raise ReviewRouteUnavailable(
                "agent_session review slot has no configured session route "
                f"({REVIEW_SESSION_ROUTE_ENV} / OUROBOROS_SUBAGENT_HARNESS are empty or `off`)",
                code="session_route_unconfigured")
        return route

    def _custody_drive(self) -> Any:
        drive = self.assignment.custody_root
        if drive is None:
            raise ReviewRouteUnavailable(
                "agent_session slot has no custody root: a delegated review run "
                "must be durably custodied before it may start", code="custody_root_missing")
        return drive

    def _run_session(self) -> None:
        request, slot = self.assignment.request, self.assignment.slot
        root = str(request.session_root or "").strip()
        if not root:
            raise ReviewRouteUnavailable(
                "agent_session slot has no session root: the surface must name the "
                "repository root the reviewer session runs in", code="session_root_missing")
        from ouroboros.config import get_finalization_grace_sec
        from ouroboros.deadline_utils import review_operation_timeout_sec
        logical_deadline = getattr(self, "_logical_deadline_monotonic", None)
        logical_timeout = (
            max(0.001, float(logical_deadline) - time.monotonic())
            if logical_deadline is not None else
            review_operation_timeout_sec(getattr(slot, "timeout_sec", None),
                route=getattr(slot, "route", None),
                deadline_at=getattr(request, "deadline_at", "") or "",
                transport_timeout_sec=getattr(slot, "transport_timeout_sec", None),
                reserve_sec=get_finalization_grace_sec())
        )
        facts = run_delegated_review_session(
            prompt=self.session_prompt,
            root=root,
            custody_drive=self._custody_drive(),
            invocation=SessionInvocation(
                task_id=str(request.task_id or ""),
                surface=request.surface,
                slot_id=slot.slot_id,
                timeout_sec=logical_timeout,
                logical_key_extra=(self.assignment.call_id,),
                output_schema=review_session_output_schema(request.surface),
                session_route=self._session_route(),
                retry_state=self._retry_state,
                reconcile_only=bool(getattr(request, "reconcile_only", False)),
                use_thread=request.surface == "plan_review",
                thread_id=str((request.session_threads or {}).get(slot.slot_id) or ""),
                dispatch_stamp=self.assignment.dispatch_stamp,
                operation_id=self.assignment.call_id,
                pending_invocation_checkpoint=self._pending_invocation_checkpoint,
                owner_deadline_at=str(getattr(request, "deadline_at", "") or ""),
            ),
        )
        self._run_id = facts["run_id"]
        conformance = facts["conformance"]
        self._conformance_passed = conformance == "passed"
        # Array/object surfaces REQUEST the structured verdict; a report surface
        # asks for no schema, so neither its absence nor its non-conformance is
        # a landing of any reason.
        schema_requested = review_session_output_schema(request.surface) is not None
        if schema_requested and not facts["schema_asked"]:
            # An effective transport that cannot carry the schema is a landing
            # below the ask, disclosed rather than silently downgraded (D4).
            self._deltas.append({
                "kind": "capability_delta",
                "requested": "outputSchema (structured verdict)",
                "effective": f"no structured output on effective route {facts['route_id']}",
                "reason": "schema_unavailable_on_effective_route",
            })
        elif schema_requested and not self._conformance_passed:
            self._deltas.append({
                "kind": "capability_delta",
                "requested": "outputSchema (structured verdict)",
                "effective": f"outputConformance={conformance or 'absent'}",
                "reason": "schema_not_conformed_on_effective_route",
            })
        effective_routes = facts.get("effective_route_ids") or []
        if effective_routes and set(effective_routes) != {facts["route_id"]}:
            # Belt over the pin: the request names exactly one eligible
            # harness, so the engine's receipt disagreeing is drift that must
            # surface loudly, never a quietly accepted substitute route.
            self._deltas.append({
                "kind": "capability_delta",
                "requested": f"route {facts['route_id']} (pinned pool)",
                "effective": "route(s) " + ", ".join(effective_routes),
                "reason": "session_ran_off_pinned_route",
            })
        spend, estimated = facts["spend"], facts["spend_estimated"]
        self._session_usage = {
            "provider": "claudexor",
            "resolved_model": facts["model"],
            "delegated_run_id": facts["run_id"],
            "delegated_route": effective_routes[0] if len(effective_routes) == 1 else "",
            "requested_route": facts["route_id"],
            "observed_attempt": facts.get("observed_attempt") or {},
            "review_thread_id": str(facts.get("thread_id") or ""),
            "review_turn_id": str(facts.get("turn_id") or ""),
            "review_thread_receipt": facts.get("thread_receipt") or {},
            "auth_route_receipt": facts.get("auth_route_receipt") or {},
            "profile_continuity_receipt": facts.get("profile_continuity_receipt") or {},
            # APPLIED account/access (D29): what the engine's receipt disclosed,
            # '' when telemetry predates it — shown as absent, never as the
            # requested value dressed up as applied.
            "applied_profile": facts.get("applied_profile", ""),
            "applied_access": facts.get("applied_access", ""),
            # Whether the durable start row actually landed. `record_started`'s answer
            # is already a fact the caller acts on; carrying it into the actor record
            # too means a verdict delivered by a run with NO durable custody is legible
            # afterwards instead of looking identical to a custodied one.
            "custody_durable": bool(facts.get("custody_durable")),
            "output_conformance": conformance,
            "settlement": facts["settlement"],
            # The ledger row is written by settle_run (record_subscription_session);
            # cost rides here for the actor record only, finality following the
            # spendEstimated fact, never re-derived.
            "cost": spend if (spend is not None and not estimated) else None,
            "cost_disclosed_usd": spend,
            "cost_estimated": estimated,
        }
        slot_model = str(slot.model or "")
        session_target = str(getattr(slot, "session_target", "") or "")
        from ouroboros.provider_models import normalize_model_identity
        if session_target:
            # Structured rows keep the opaque ``harness[=model]`` target in
            # ``slot.model`` for row identity/display, while the daemon sees
            # only the parsed model component. Compare like with like: the old
            # full-spec-vs-model comparison invented a capability delta for
            # every healthy pinned session row.
            from ouroboros.subagents import parse_subagent_harness

            parsed_target = parse_subagent_harness(session_target)
            slot_model = str(getattr(parsed_target, "model", "") or "")
        if (
            slot_model and facts["model"]
            and normalize_model_identity(slot_model) != normalize_model_identity(facts["model"])
        ):
            self._deltas.append({
                "kind": "capability_delta",
                "requested": f"model {slot_model}",
                "effective": f"model {facts['model']}",
                "reason": "session_route_resolves_its_own_model",
            })
        # PAID EVIDENCE: the transcript always feeds the parser whole. A profile
        # continuity `cannot_verify` is telemetry, never a reason to blank it.
        self._raw_transcript = facts["text"]

    def _verdict_result(self, force_extraction: bool = False) -> ReviewAttemptResult:
        text = self._raw_transcript or ""
        canonical, method, extraction_usage = canonicalize_session_verdict(
            text,
            conformance_passed=self._conformance_passed and not force_extraction,
            contract=self._output_contract(),
            llm=self.llm,
            deadline_at=getattr(self.assignment.request, "deadline_at", "") or "",
            transport_timeout_sec=getattr(self.assignment.slot, "transport_timeout_sec", None),
            shape=review_output_shape(self.assignment.request.surface),
        )
        usage = dict(self._session_usage)
        deltas = list(self._deltas)
        if method == "light_model_extraction":
            usage["extraction"] = extraction_usage
            deltas = deltas + [{
                "kind": "capability_delta",
                "requested": "structured verdict from the session",
                "effective": "light-model extraction over the collected transcript",
                "reason": "extraction_instead_of_schema",
            }]
        elif method == "extraction_incomplete":
            deltas = deltas + [{
                "kind": "capability_delta",
                "requested": "structured verdict from the session",
                "effective": (
                    f"no verdict: transcript ({len(text)} chars) exceeds the "
                    "single-send extraction bound"
                ),
                "reason": "extraction_incomplete_transcript_exceeds_bound",
            }]
        usage["verdict_method"] = method
        # P1: the cognitive artifact is the SESSION's own output, and canonicalization
        # legitimately destroys it — a schema-conformant `{"findings": []}` becomes `[]`
        # and light extraction replaces the narrative wholesale. Keeping the transcript
        # only in this object made the decision unreconstructible the moment the process
        # ended. The raw text rides the MESSAGE, which the coordinator persists whole via
        # persist_call (redacted projection, no truncation), and the provenance below
        # says exactly which text produced which verdict.
        usage["verdict_provenance"] = {
            "raw_transcript_chars": len(text),
            "raw_transcript_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
            "canonical_chars": len(canonical),
            "canonical_sha256": hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest(),
            "output_conformance": self._session_usage.get("output_conformance") or "",
            "conformance_trusted": bool(self._conformance_passed and not force_extraction),
            "verdict_method": method,
            "raw_transcript_carrier": "message.session_transcript (durable response_ref)",
        }
        if deltas:
            usage["capability_delta"] = deltas
            self._emit_capability_delta(deltas, method)
        message = {
            "content": canonical,
            # The unmodified session output, persisted alongside the canonical form.
            "session_transcript": text,
            "delegated_run_id": self._run_id,
            "verdict_method": method,
        }
        return ReviewAttemptResult(message=message, usage=usage, raw_text=canonical)
    def _emit_capability_delta(self, deltas: List[Dict[str, Any]], method: str) -> None:
        """Durable half of the disclosure (D4): every landing below what was
        asked reaches the event log, not only the actor record."""
        try:
            from ouroboros import delegate_custody as custody

            custody.emit(self._custody_drive(), "review_slot_capability_delta", {
                "run_id": self._run_id,
                "surface": self.assignment.request.surface,
                "slot_id": self.assignment.slot.slot_id,
                "verdict_method": method,
                "deltas": deltas,
            })
        except Exception:
            log.warning("capability_delta disclosure write failed", exc_info=True)

# Closed route table. Adding a route means adding an executor here; it never
# means adding a branch to the coordinator.
_REVIEW_ROUTE_EXECUTORS: Dict[ReviewRouteKind, type[ReviewSlotExecutor]] = {
    ReviewRouteKind.API_CHAT: ApiChatReviewExecutor,
    ReviewRouteKind.AGENT_SESSION: AgentSessionReviewExecutor,
}

def _review_route_executor(assignment: ReviewAssignment, *, llm: Any = None) -> ReviewSlotExecutor:
    """Bind a route to its executor. The ONLY place a review transport is chosen.

    Bound once per slot because the durable prompt record must be written from
    the route's own (lazily rendered) projection BEFORE the first send.
    """
    try:
        route = ReviewRouteKind(assignment.route)
    except ValueError:
        raise ReviewRouteUnavailable(f"unknown review route: {assignment.slot.route!r}", code="unknown_review_route") from None
    if route is ReviewRouteKind.API_CHAT and bool(
        getattr(assignment.slot, "native_retrieval", False)
    ):
        # A configured-subagent api row is the RETRIEVES class: same wire kind,
        # different delivery — bounded native tool rounds, never the packet.
        # Imported lazily: the native module subclasses this module's seam.
        from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

        return NativeToolRoundReviewExecutor(assignment, llm=llm)
    executor_cls = _REVIEW_ROUTE_EXECUTORS.get(route)
    if executor_cls is None:
        raise ReviewRouteUnavailable(f"review route not implemented in this build: {route.value}",
                                     code="review_route_not_implemented")
    return executor_cls(assignment, llm=llm)


def _execute_slot_attempt(
    assignment: ReviewAssignment,
    *,
    llm: Any = None,
    executor: ReviewSlotExecutor | None = None,
) -> ReviewAttemptResult:
    """Run ONE physical attempt for ``assignment`` — the single execution seam.

    Everything route-specific happens at or below this call; everything above it
    (attempt rails, persistence, parsing, actor projection, quorum) is route
    agnostic. Pass the ``executor`` already bound for the prompt record so a slot
    renders its prompt once instead of once per attempt.
    """
    return (executor or _review_route_executor(assignment, llm=llm)).execute()
