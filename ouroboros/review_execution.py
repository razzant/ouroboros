"""Review execution: how a reviewer slot is delivered.

This module is everything BELOW the review seam — the closed set of delivery
routes, the immutable assignment handed to a route, the typed result handed
back, and each route's own prompt rendering. ``review_substrate`` keeps the
policy above the seam (attempt rails, persistence, parsing, actor projection,
quorum) and knows only that a route exists.

The dependency runs one way on purpose: this module never imports the
coordinator, so a new route can be added here without touching review policy.
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
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from ouroboros.review_slot_cancel import (  # noqa: F401 — re-exported seam surface
    ReviewSessionSucceededResultUnavailable,
    _cancel_honesty_clause,
    _interaction_outlives_slot,
    _natural_success_terminal,
    _slot_cancel_outcome,
)
from ouroboros.triad_review import (
    ACCEPTANCE_SURFACE_RULES,
    REVIEW_JSON_ARRAY_CONTRACT,
    TIER_CLASSIFICATION_RULES,
    empty_array_is_verified_clean,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    extract_json_array,  # noqa: F401 -- historical import surface kept for monkeypatching tests
)
from ouroboros.review_session_verdict import (  # noqa: F401 -- intentional public re-exports
    REVIEW_SESSION_OUTPUT_SCHEMA,
    _EXTRACT_MAX_CHARS,
    _SESSION_EXTRACT_PROMPT,
    _UNEXTRACTABLE,
    _extract_verdict_via_light_model,
    _findings_array,
    _strictly_parseable,
    canonicalize_session_verdict,
    review_session_output_schema,
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


class ReviewRouteUnavailable(RuntimeError):
    """Typed refusal for a route with no executor in this build.

    A missing route fails loudly on its own slot; it never falls back to another
    route, model, or profile. ``code`` is the machine-readable refusal vocabulary
    (a ``route_health`` reason or a site code); "" is an uncoded raise.
    """

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = str(code or "")


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


def _render_prompt_parts(request: ReviewRequest, slot: ReviewSlot) -> tuple[str, str, str]:
    """Return (stable_governance, task_stable, dynamic_evidence) for one slot.

    Cache segmentation (v6.74.0, B1): the byte-stable governance instruction and
    the task-stable contract (goal/scope/checklist/policy — stable across the
    improvement passes of ONE task) are the two cache-marked segments; the
    mutable tail (subject, evidence, refs) is never marked, and the slot label
    lives at its TAIL so concurrent same-model slots share a warm prefix."""
    evidence = json.dumps(request.evidence, ensure_ascii=False, indent=2, default=str)
    refs = json.dumps(request.evidence_refs, ensure_ascii=False, indent=2, default=str)
    policy = json.dumps(request.policy, ensure_ascii=False, indent=2, default=str)
    classify_tier = bool(request.policy.get("classify_outcome_tier"))
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
        if request.surface == "task_acceptance"
        else ""
    )
    # v6.74.0 acceptance-dialogue keys (A3/A5): reviewer-authored obligation
    # identity and the typed dialogue judgement. Both live in the REQUIRED key
    # list for the same reason as the tier keys above.
    dialogue_key = (
        ', dialogue_status ("continue_actionable"|"unreachable_here"|"stable_disagreement")'
        if request.surface == "task_acceptance"
        else ""
    )
    findings_shape = (
        '[{severity, item, evidence, recommendation, disposition_kind ("new"|"re_raise"), '
        'obligation_id (required when disposition_kind="re_raise")}]'
        if request.surface == "task_acceptance"
        else "[{severity, item, evidence, recommendation}]"
    )
    tier_rules = TIER_CLASSIFICATION_RULES if classify_tier else ""
    acceptance_rules = (
        ACCEPTANCE_SURFACE_RULES if request.surface == "task_acceptance" else ""
    )
    stable = (
        "You are an independent Ouroboros reviewer slot.\n"
        f"Surface: {request.surface}\n"
        f"Role hint: {slot.role_hint or 'general reviewer'}\n\n"
        "The review subject and evidence packet arrive in the user message.\n\n"
        f"Return JSON with keys: verdict (PASS|FAIL|DEGRADED){tier_keys}{criteria_key}{dialogue_key}, findings "
        f"({findings_shape}), and summary. "
        + tier_rules
        + acceptance_rules
        + "If you cannot judge because evidence is missing, return DEGRADED and explain."
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
    if request.messages:
        messages = [
            dict(message) if isinstance(message, dict) else {"role": "user", "content": str(message)}
            for message in request.messages
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
    """Immutable description of one reviewer slot's job.

    Built by ``_run_slot`` before any transport exists and handed to the
    execution seam unchanged, so every route receives the SAME task, subject,
    evidence and output contract — only the delivery differs.
    """

    request: ReviewRequest
    slot: ReviewSlot
    call_id: str = ""
    call_type: str = ""
    # Where a DELEGATED route's custody rows live (the canonical/budget drive).
    # Data, not policy: the coordinator computes it once; the api_chat route
    # never reads it.
    custody_root: Any = None

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

    def prompt_payload(self) -> Dict[str, Any]:
        """Route-owned projection of what will actually be sent (for the durable
        prompt record). Rendered lazily: a route that never builds the big API
        pack must never pay for building it."""
        raise NotImplementedError

    def prompt_chars(self) -> int:
        raise NotImplementedError

    def execute(self) -> ReviewAttemptResult:
        raise NotImplementedError


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
        if self._chat_kwargs is None:
            request, slot = self.assignment.request, self.assignment.slot
            self._chat_kwargs = {
                "messages": self.messages,
                "model": slot.model,
                "reasoning_effort": slot.effort,
                "max_tokens": int(request.max_tokens or slot.max_tokens),
                "temperature": request.temperature if request.temperature is not None else slot.temperature,
                "no_proxy": bool(request.no_proxy),
                # Stable per-surface session affinity: changing evidence would
                # otherwise fragment default first-user-message sticky routing
                # (and the provider cache) on every round. Deliberately NO
                # slot_id in the key — same-model slots keep today's
                # provider-concentration behavior.
                "cache_affinity": f"{request.surface}:{request.task_id or 'review'}",
                # Bound the socket to the logical slot timeout so a stalled
                # connection cannot leave the whole review process unable to exit.
                # The outer queue/wait_for still governs the logical deadline.
                "timeout": float(slot.timeout_sec) if slot.timeout_sec else None,
                "use_local": bool(slot.use_local),
            }
        return self._chat_kwargs

    def execute(self) -> ReviewAttemptResult:
        chat_kwargs = self._kwargs()
        chat = getattr(self.llm, "chat", None)
        if callable(chat):
            msg, usage = chat(**chat_kwargs)
        else:
            msg, usage = asyncio.run(self.llm.chat_async(**chat_kwargs))
        # A provider can yield a null/non-object message on a zero-body
        # response. Treat it exactly like empty content: the caller's attempt
        # rail decides whether another send is permitted.
        raw_text = str(msg.get("content") or "") if isinstance(msg, dict) else ""
        return ReviewAttemptResult(message=msg, usage=usage, raw_text=raw_text)


# ---------------------------------------------------------------------------
# Route configuration (phase-5 shape).
#
# One key per surface names each configured row's DELIVERY, aligned by index
# with that surface's model list; one shared key names the session target as an
# OPAQUE ``harness[=model][:effort]`` spec (Claudexor's own reviewer-panel
# spelling — no codex/claude/cursor member anywhere in this module). Phase 6's
# structured slot config (one settings SSOT per D14/6.1) supersedes these keys;
# they are read here, below the seam, so ``config.py`` stays untouched.
# ---------------------------------------------------------------------------

TRIAD_REVIEW_ROUTES_ENV = "OUROBOROS_REVIEW_ROUTES"
SCOPE_REVIEW_ROUTES_ENV = "OUROBOROS_SCOPE_REVIEW_ROUTES"
REVIEW_SESSION_ROUTE_ENV = "OUROBOROS_REVIEW_SESSION_ROUTE"

_ROUTE_TOKENS: Dict[str, ReviewRouteKind] = {
    "": ReviewRouteKind.API_CHAT,
    "api": ReviewRouteKind.API_CHAT,
    "api_chat": ReviewRouteKind.API_CHAT,
    "agent_session": ReviewRouteKind.AGENT_SESSION,
}


def configured_review_routes(env_key: str, count: int) -> List[ReviewRouteKind]:
    """Per-row delivery routes for one review surface.

    An empty key (the shipped default) is the historical behavior: every row is
    ``api_chat``. An unknown token raises — mapping a typo to ``api_chat`` would
    silently spend the API money the owner configured the row to move off of,
    and mapping it to ``agent_session`` would silently delegate a row the owner
    never delegated.
    """
    raw = os.environ.get(env_key, "") if env_key else ""
    tokens = [t.strip().lower() for t in raw.split(",")] if raw.strip() else []
    routes: List[ReviewRouteKind] = []
    for idx in range(max(0, int(count))):
        token = tokens[idx] if idx < len(tokens) else ""
        if token not in _ROUTE_TOKENS:
            raise ValueError(
                f"{env_key} row {idx + 1} names an unknown review route {token!r}; "
                f"valid: api_chat, agent_session"
            )
        routes.append(_ROUTE_TOKENS[token])
    return routes


def review_session_route() -> Any:
    """The configured session target for delegated review slots (or ``None``).

    Reuses the subagent-harness spelling and parser verbatim; when the review
    key is unset the subagent route is the target, so an owner who configured
    ONE delegated route does not have to configure it twice.
    """
    from ouroboros.subagents import get_subagent_harness, parse_subagent_harness

    raw = str(os.environ.get(REVIEW_SESSION_ROUTE_ENV, "")).strip()
    route = parse_subagent_harness(raw)
    if route is None and raw and raw.lower() != "off":
        # Same silent-typo class as the subagent key's reader: a non-empty value
        # that parses to nothing would quietly re-route review sessions onto the
        # subagent route as if the review key were never set.
        log.warning(
            "%s is set but unparseable (%r) — review sessions fall back to the "
            "subagent route until it reads harness[=model][:effort]",
            REVIEW_SESSION_ROUTE_ENV, raw,
        )
    return route or get_subagent_harness()


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


def _owned_started_review_custody(
    custody: Any, custody_drive: Any, record: Dict[str, Any], claimant_task_id: str,
) -> tuple[str, Any]:
    """Return the existing run custody only after exact owner corroboration."""
    run_id = str(record.get("run_id") or "")
    ownership, found = custody.lookup(custody_drive, claimant_task_id, run_id)
    invocation_owner = str(record.get("task_id") or "")
    custody_owner = str(getattr(found, "task_id", "") or "")
    if (ownership != custody.OWNED or found is None
            or invocation_owner != claimant_task_id
            or custody_owner != claimant_task_id):
        raise ReviewRouteUnavailable(
            "delegated review started-run recovery could not corroborate "
            f"ownership for run {run_id} (lookup={ownership}, "
            f"claimant={claimant_task_id!r}, invocation_owner={invocation_owner!r}, "
            f"custody_owner={custody_owner!r})", code="review_recovery_ownership_unverified")
    return run_id, found


def _review_recovery_facts(
    record: Dict[str, Any], run_request: Any, started_custody: Any, *,
    prompt: str, root: str,
) -> tuple[Any, str, str, str, bool]:
    """Validate one stored request and restore its immutable delivery facts."""
    from ouroboros.subagents import DelegationRoute

    if not isinstance(run_request, dict) or not run_request:
        raise ReviewRouteUnavailable(
            "delegated review recovery has no canonical stored request; "
            "the existing invocation is not re-derived", code="review_recovery_request_missing")
    stored_root = str((run_request.get("scope") or {}).get("root") or "")
    if str(run_request.get("prompt") or "") != prompt or stored_root != root:
        raise ReviewRouteUnavailable(
            "delegated review retry replays a recorded invocation whose "
            f"{'prompt' if stored_root == root else 'session root'} differs from "
            "this call's; the durable retry token remains untouched and the "
            "recorded invocation is not replayed against a different review", code="review_recovery_request_mismatch")
    if started_custody is not None:
        route_id = str(started_custody.route_id or "")
        model = str(started_custody.model or "")
        project_id = str(started_custody.project_id or "")
        project_owned = bool(started_custody.project_owned)
        key = str(started_custody.idempotency_key or "")
    else:
        route_id = str(run_request.get("primaryHarness") or record.get("route") or "")
        model = str(run_request.get("model") or "")
        project_id = str(record.get("project_id") or "")
        project_owned = bool(record.get("project_owned"))
        key = str(record.get("idempotency_key") or "")
    route = DelegationRoute(
        route_id=route_id, model=model, effort=str(run_request.get("effort") or ""),
        # The stored request carries the pin: a pinned retry replays PINNED (D1).
        profile_id=str(run_request.get("credentialProfileId") or ""))
    existing_project = "" if project_owned else project_id
    return route, project_id, existing_project, key, "outputSchema" in run_request


def run_delegated_review_session(
    *,
    prompt: str,
    root: str,
    custody_drive: Any,
    invocation: SessionInvocation,
) -> Dict[str, Any]:
    """Start, watch, settle and collect ONE delegated read-only review session.

    The single delegated-session transport for every review surface — the
    substrate's agent_session slots and the delegated advisory both run through
    here, so there is one nanny loop, not two. The caller owns semantics; this
    function owns delivery:

    - readonly shape, ``authPreference: subscription``, typed refusals for an
      unconfigured/unhealthy route — never a fallback to another route (§8);
    - the requested route is PINNED as the run's explicit one-element
      ``harnesses`` pool: ``primaryHarness`` alone only reorders the engine's
      auto-pool, so a "preferred" route can silently land on whatever other
      harness the pool holds. A pinned pool runs THIS route or refuses typed;
    - ``output_schema`` is ASKED only when the pinned route's EFFECTIVE
      transport can carry it (live manifest, not the static adapter flag —
      D19); the returned ``conformance`` is the only thing a caller may gate
      trust on;
    - invocation identity follows the explicit-retry contract: an ordinary call
      ALWAYS mints a fresh invocation id and records the CANONICAL body on the
      durable request row; an unknown-outcome failure leaves its token in
      ``retry_state`` (caller-owned, surviving across the slot's permitted
      physical attempts), and only that token replays — the STORED bytes under
      the SAME wire key. A token whose invocation already bound a run is waited
      on instead of re-posted; a definitively refused token is dead and the
      call mints fresh. Nothing is ever reused by content-matching;
    - the nanny owns the time cap: a run that outlives ``timeout_sec`` is
      cancelled through the verified-cancel path and reported as a timeout;
    - the transcript is read from the verified FULL primary output (D7), never
      a bounded preview; the run is settled through delegate_custody either way.

    Returns ``{run_id, text, conformance, schema_asked, settlement, route_id,
    model, spend, spend_estimated}``. Raises ``ReviewRouteUnavailable``,
    ``TimeoutError`` or ``RuntimeError``.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.gateways.claudexor import (
        WINDOW_EXHAUSTED_CODES, ClaudexorSubscriptionWindowExhausted, ClaudexorUnavailable,
    )
    from ouroboros.subagents import delegated_run_shape, route_health
    from ouroboros.usage_accounting import current_usage_scope

    task_id, surface, slot_id = invocation.task_id, invocation.surface, invocation.slot_id
    timeout_sec, logical_key_extra = invocation.timeout_sec, invocation.logical_key_extra
    output_schema, session_route = invocation.output_schema, invocation.session_route
    instructions, retry_state = invocation.instructions, invocation.retry_state
    # #112: task-tree lineage for BOTH custody writers, captured once from the
    # ambient scope; empty when unbound — settlement owns the task_id fallback.
    _scope = current_usage_scope()
    root_task_id = str(getattr(_scope, "root_task_id", "") or "")
    parent_task_id = str(getattr(_scope, "parent_task_id", "") or "")
    shape = delegated_run_shape(False)  # a reviewer reads and answers
    state = retry_state if retry_state is not None else {}
    run_id, run_request, invocation_id = "", None, ""
    started_custody = None
    # THE RETRY IS READ FIRST, before any daemon call. A retry replays the STORED invocation,
    # so every fact about it — the route whose health is checked, the project, the lookup key,
    # whether the schema was asked — comes from the record, not from the environment as it
    # stands now. Computing them up front POSTed the recorded body while checking a route the
    # run never used and writing a durable record that contradicted the bytes on the wire.
    retry_token = str(state.get("pending_invocation_id") or "")
    record = custody.invocation_record(custody_drive, retry_token) if retry_token else None
    if record is not None and record["state"] == "started" and record["run_id"]:
        # The CURRENT task is the claimant; the stored owner cannot self-authorize.
        run_id, started_custody = _owned_started_review_custody(
            custody, custody_drive, record, task_id)
        run_request, invocation_id = record.get("request"), retry_token
    elif (record is not None and record["state"] == "pending"
          and isinstance(record.get("request"), dict) and record["request"]):
        run_request, invocation_id = record["request"], retry_token
    # A dead (definitely refused) or unrecorded token falls through and mints fresh; its id never rides the wire again.
    recovering = bool(run_id) or run_request is not None
    if recovering:
        route, project_id, existing_project, key, schema_asked = (
            _review_recovery_facts(
                record, run_request, started_custody, prompt=prompt, root=root)
        )
    else:
        route = session_route if session_route is not None else review_session_route()
        if route is None:
            raise ReviewRouteUnavailable(
                "delegated review session has no configured session route "
                f"({REVIEW_SESSION_ROUTE_ENV} / OUROBOROS_SUBAGENT_HARNESS are empty or `off`)",
                code="session_route_unconfigured")
        project_id, existing_project, key, schema_asked = "", "", "", False
    gateway = ensure_owned_gateway()
    try:
        if not run_id:
            # Admission health applies only while this call may POST: a changed quota
            # window cannot invalidate a STARTED run that already exists. The slot's
            # credential pin rides into the health read (D1): the ENGINE's typed refusal
            # is authoritative for a pinned profile, and a PINNED row is judged against
            # its own subject exactly (unified-accounts §K.7) — a healthy sibling
            # account must not vouch a spent pin into a dispatch the engine will refuse.
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
            invocation_id = custody.new_invocation_id()
            seconds = max(1, min(int(timeout_sec or 300), _CLAUDEXOR_MAX_SECONDS))
            run_request = {
                "prompt": prompt,
                "instructions": instructions,
                "authPreference": "subscription",
                "mode": shape.mode,
                "access": shape.access,
                "scope": {"kind": "project", "root": root},
                # PIN, not preference. In the engine, ``primaryHarness`` only
                # moves its harness to the FRONT of the auto-pool; the pool
                # still holds every other doctor-OK harness, and a plain ask
                # run fails over across them (best-of runs several). The
                # explicit ``harnesses`` pool is the pinning contract — the
                # engine's own MCP surface spells "force this one-shot run
                # onto X" as ``harnesses: [X]`` — and a one-element pool is
                # width 1 by construction, so the run rides THIS route or
                # refuses typed. ``n`` deliberately stays off the wire: the
                # engine's mode/strategy coherence refuses it on plain
                # ask-mode runs.
                "harnesses": [route.route_id],
                "primaryHarness": route.route_id,
                "maxSeconds": seconds,
            }
            if route.model:
                run_request["model"] = route.model
            if route.effort:
                run_request["effort"] = route.effort
            if getattr(route, "profile_id", ""):
                # Manual credential pin (Q2-в, optional): the daemon still owns
                # rotation for every unpinned row (D28 default).
                run_request["credentialProfileId"] = route.profile_id
            if schema_asked:
                run_request["outputSchema"] = output_schema
        if not run_id:
            seconds = int(run_request.get("maxSeconds") or timeout_sec or 300)
            requested = custody.record_start_requested(
                custody_drive, run_id="", task_id=task_id,
                idempotency_key=key, invocation_id=invocation_id,
                max_seconds=seconds, request=run_request, project_id=project_id,
                project_owned=not existing_project, route=route.route_id,
                surface=surface, slot_id=slot_id,
                # #112: the request row's lineage is what pending-invocation
                # recovery replays onto a run this worker never bound.
                root_task_id=root_task_id, parent_task_id=parent_task_id,
            )
            if not requested:
                # The POST is CONDITIONAL on the durable request row: a run launched
                # without its custody trail would be unfindable if this worker died
                # mid-review. Nothing was sent on THIS attempt, so a FRESH start's
                # registration is definitively retirable — but a RETRY's project
                # belongs to the original attempt, whose POST may have bound a live run.
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    definite_refusal=not recovering,
                    reason="start_request_row_unwritable",
                    invocation_id=invocation_id, surface=surface, slot_id=slot_id,
                )
                raise ReviewRouteUnavailable(
                    "the durable start-request row could not be written; the "
                    "delegated review session was NOT started", code="start_request_row_unwritable")
            try:
                handle = gateway.start_run(run_request, idempotency_key=invocation_id)
            except ClaudexorUnavailable as exc:
                status = int(getattr(exc, "status_code", 0) or 0)
                definite = 400 <= status < 500
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    # Only a DEFINITE 4xx refusal proves no run bound this
                    # registration; on a retry the registration is the original
                    # attempt's, whose fate stays unknown either way.
                    definite_refusal=definite and not recovering,
                    reason=exc.code, invocation_id=invocation_id,
                    surface=surface, slot_id=slot_id,
                )
                if not definite:
                    # Unknown outcome: the token survives in the caller-owned
                    # state, so the slot's OWN permitted retry replays this very
                    # invocation instead of minting a second live run.
                    state["pending_invocation_id"] = invocation_id
                else:
                    state.pop("pending_invocation_id", None)
                raise
            run_id = str(handle.get("runId") or handle.get("jobId") or "")
            if not run_id:
                # The POST SUCCEEDED, so a run is MORE likely live here than on
                # the refusal branch: the registration is retained and durably
                # named, never destroyed on an unverified outcome.
                _retire_orphaned_review_registration(
                    custody, gateway, custody_drive, project_id,
                    definite_refusal=False, reason="queued_without_run_id",
                    invocation_id=invocation_id, surface=surface, slot_id=slot_id,
                )
                state["pending_invocation_id"] = invocation_id
                raise ReviewRouteUnavailable(
                    f"Claudexor returned a queued handle without a run id: {handle!r}", code="queued_without_run_id")
        state.pop("pending_invocation_id", None)
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
                ledger_root=str(custody_drive), idempotency_key=key,
                invocation_id=invocation_id or retry_token,
            )
            # record_started's answer is a FACT the caller needs, not a side effect: a run
            # whose authoritative row did not land is custodied by this process alone, so
            # after this worker dies nothing can name, wait on, cancel or settle it.
            # Discarding the boolean reported a plainly-started run over exactly that state.
            custody_durable = bool(custody.record_started(custody_drive, entry, shape={
                "effort": route.effort, "access": shape.access, "mode": shape.mode,
                "isolation": shape.isolation, "delegated": shape.delegated,
                "root": root, "surface": surface, "slot_id": slot_id,
            }))
        detail = _poll_session_terminal(gateway, custody, custody_drive, entry,
                                        run_id, float(timeout_sec or 300))
        settlement = custody.settle_run(custody_drive, gateway, entry, detail)
        summary = custody.summary_of(detail)
        run_state = str(summary.get("state") or "")
        if run_state != "succeeded":
            # The engine states WHY in a typed RunFailure — `code` and, for a spent
            # subscription window, the `resetsAt` a caller is meant to schedule against.
            # Flattening it into prose and truncating that prose at 500 chars destroyed
            # both, so every terminal refusal arrived as an indistinguishable
            # RuntimeError and the retry rail above launched a second, identically
            # doomed, billable session. Raise the class the seam already has.
            failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
            message = (f"delegated review session {run_id} ended {run_state or 'unknown'}"
                       + (f": {json.dumps(failure, ensure_ascii=False)}" if failure else ""))
            code = str(failure.get("code") or "")
            if code in WINDOW_EXHAUSTED_CODES:
                raise ClaudexorSubscriptionWindowExhausted(
                    message, reset_at=str(failure.get("resetsAt") or ""), code=code)
            raise ClaudexorUnavailable(code or f"run_{run_state or 'unknown'}", message)
        text = _full_session_text(gateway, run_id, detail)
        spend, estimated = custody.disclosed_spend(summary)
        return {
            "run_id": run_id,
            "text": text,
            "conformance": str(summary.get("outputConformance") or "").strip().lower(),
            "schema_asked": schema_asked,
            "custody_durable": custody_durable,
            "idempotent_recovery": recovering,
            "settlement": settlement,
            "route_id": str(entry.route_id),
            # The engine's own receipt of the pool the run USED — a pinned run
            # must echo exactly the pinned route; anything else is a landing
            # below the request and the caller discloses it (D4).
            "effective_route_ids": [
                str(h) for h in (summary.get("harnesses") or []) if str(h)
            ],
            "model": str(summary.get("model") or ""),
            "spend": spend,
            "spend_estimated": estimated,
            # D22/D29 APPLIED facts, verbatim from the run's own telemetry
            # receipt — empty when the engine predates it, never invented.
            "applied_profile": str((summary.get("authRoute") or {}).get("profileId") or ""),
            # `access` is the client's own request echoed back, not a witness —
            # `_widened_access` already refuses to read it for exactly that reason.
            # Falling back to it published a REQUEST as an APPLIED fact.
            "applied_access": str(summary.get("effectiveAccess") or ""),
        }
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
    """Wait for the run's terminal state, bounded by the SLOT's own clock.

    The nanny owns the time cap: on expiry the run is cancelled through the
    verified-cancel path (an unverified stop must not read as stopped) and the
    slot fails as an ordinary timeout on its own row.

    A session that parks on an interactive question terminates the slot EARLY
    (F18) — but only when the question has no engine expiry inside the slot's
    remaining budget (``_interaction_outlives_slot``): review slots are
    non-interactive, so such a question can only burn the slot in silence. The
    run is cancelled through the same verified path under its own typed reason
    and the failure names the pending question. A question whose ``timeout_at``
    provably lands first is left to the engine's benign decline and the poll
    continues (R2-2).

    Both cancel sites are HONEST about what the cancel proved (BR1-1): the
    typed outcome rides the raise, and a verify read that discovers a natural
    SUCCESS terminal returns it as the slot's ordinary result instead of
    raising over it — completion wins, no host-side waiting added."""
    from ouroboros.gateways.claudexor import pending_interactions as _cx_pending

    deadline = time.monotonic() + max(1.0, float(seconds))
    detail = gateway.get_run(run_id)
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
        time.sleep(min(_SESSION_POLL_SEC, max(0.0, deadline - time.monotonic())))
        detail = gateway.get_run(run_id)
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
    """The hosted-session route: one delegated Claudexor run per slot.

    The COORDINATOR stays the owner of attempt policy, persistence, parsing,
    actor projection and quorum; this class owns only delivery — it is the
    nanny in host form: start the run, watch it, enforce the slot's own time
    bound, bring the transcript home, and let the host's parsers judge it. It
    never falls back to another route, provider, model or profile (§8), and a
    repair resend NEVER restarts the session: the second ``execute`` performs
    local extraction over the transcript already collected (plan 5.5).
    """

    route = ReviewRouteKind.AGENT_SESSION

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        super().__init__(assignment, llm=llm)
        self._session_prompt: Optional[str] = None
        self._transcript: Optional[str] = None
        self._conformance_passed = False
        self._run_id = ""
        self._session_usage: Dict[str, Any] = {}
        self._deltas: List[Dict[str, Any]] = []
        # Caller-owned retry state for the explicit-retry contract: an
        # unknown-outcome start leaves its invocation token here, so the slot's
        # permitted second physical attempt replays THAT invocation instead of
        # minting a second live run. It never crosses slot instances.
        self._retry_state: Dict[str, Any] = {}
        # The failure of a session that already RAN, remembered so the permitted
        # resend re-raises it instead of buying a second billed run (see execute).
        self._settled_failure: Optional[BaseException] = None

    # -- prompt (route-owned; never the api pack) ------------------------------

    def _output_contract(self) -> str:
        contract = str((self.assignment.request.policy or {}).get("output_contract") or "")
        return contract or REVIEW_JSON_ARRAY_CONTRACT

    def prompt_payload(self) -> Dict[str, Any]:
        return {"session_prompt": self.session_prompt}

    def prompt_chars(self) -> int:
        return len(self.session_prompt)

    @property
    def session_prompt(self) -> str:
        """The compact task this route sends — the SAME task, criteria and
        output contract as the api pack, minus the assembled evidence: a
        delegated reviewer retrieves context on the fly with its tools (D12),
        so the giant pack the session replaces is never built here (plan 5.2)."""
        if self._session_prompt is None:
            request, slot = self.assignment.request, self.assignment.slot
            task = str(request.session_task or "").strip()
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
                self._output_contract(),
                f"Slot: {slot.slot_id}",
            ]
            self._session_prompt = "\n".join(parts)
        return self._session_prompt

    # -- delivery --------------------------------------------------------------

    def execute(self) -> ReviewAttemptResult:
        if self._transcript is not None:
            # Plan 5.5: the permitted resend repairs FORMAT locally over the
            # collected transcript; it never launches a second session.
            return self._verdict_result(force_extraction=True)
        if self._settled_failure is not None:
            # The same rule, one step earlier: the session ALREADY RAN. Whether it
            # terminated on a typed refusal or succeeded and then refused to hand
            # its transcript back, the money is spent and the second physical send
            # would mint a NEW run — a fresh idempotency key, a fresh bill — for a
            # cause that is deterministic and will refuse identically. The rail's
            # second send exists for a transport transient BEFORE the run exists
            # (the pending invocation replays the same key onto the same run), and
            # that path is untouched: it leaves `_retry_state` pending and never
            # reaches here.
            raise self._settled_failure
        try:
            self._run_session()
        except BaseException as exc:
            if not self._retry_state.get("pending_invocation_id"):
                self._settled_failure = exc
            raise
        return self._verdict_result()

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
        facts = run_delegated_review_session(
            prompt=self.session_prompt,
            root=root,
            custody_drive=self._custody_drive(),
            invocation=SessionInvocation(
                task_id=str(request.task_id or ""),
                surface=request.surface,
                slot_id=slot.slot_id,
                timeout_sec=float(slot.timeout_sec or 300),
                logical_key_extra=(self.assignment.call_id,),
                output_schema=review_session_output_schema(request.surface),
                session_route=self._session_route(),
                retry_state=self._retry_state,
            ),
        )
        self._run_id = facts["run_id"]
        conformance = facts["conformance"]
        self._conformance_passed = conformance == "passed"
        if not facts["schema_asked"]:
            # This route always REQUESTS the structured verdict; an effective
            # transport that cannot carry the schema is a landing below the
            # ask, disclosed rather than silently downgraded to prose (D4).
            self._deltas.append({
                "kind": "capability_delta",
                "requested": "outputSchema (structured verdict)",
                "effective": f"no structured output on effective route {facts['route_id']}",
                "reason": "schema_unavailable_on_effective_route",
            })
        elif not self._conformance_passed:
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
            "delegated_route": facts["route_id"],
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
        self._transcript = facts["text"]

    def _verdict_result(self, force_extraction: bool = False) -> ReviewAttemptResult:
        text = self._transcript or ""
        canonical, method, extraction_usage = canonicalize_session_verdict(
            text,
            conformance_passed=self._conformance_passed and not force_extraction,
            contract=self._output_contract(),
            llm=self.llm,
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
