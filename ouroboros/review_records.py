"""Typed panel records and hardness vocabulary for every review surface.

Owns what a review run IS: the configured reviewer row, the request handed to
a panel, the per-actor record a slot produces, the aggregate run result, the
typed transport-failure fact keys, and the three hardness levels that name how
a surface enforces its verdict. Slot identity is separate from model identity,
so duplicate model IDs are valid rows. Extracted from
ouroboros/review_substrate.py (v7 D06 split, re-cut on the v7next tip);
review_substrate.py re-exports every name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves


@dataclass(frozen=True)
class ReviewSlot:
    slot_id: str
    model: str
    effort: str = "medium"
    timeout_sec: Optional[float] = None
    max_tokens: int = 16_384
    temperature: float | None = None
    role_hint: str = ""
    use_local: bool = False
    # Delivery route for this slot. ``use_local`` above is the existing
    # precedent for a per-slot transport hint; ``route`` is the general axis.
    route: ReviewRouteKind = ReviewRouteKind.API_CHAT
    # agent_session rows only: THIS row's opaque ``harness[=model]`` target
    # (6.1 — every slot is independently harness-or-API). Empty falls back to
    # the shared session-route key, which is the whole legacy behavior.
    session_target: str = ""
    # Optional manual credential pin (Q2-в); '' = the daemon's rotation (D28).
    session_profile: str = ""
    transport_timeout_sec: Optional[float] = None
    # Optional configured-subagent binding (resolved at admission; '' = direct).
    subagent_id: str = ""

    @property
    def native_retrieval(self) -> bool:
        # An api-route actor row: bounded native tool rounds, never the packet.
        return bool(str(self.subagent_id or "").strip()) and str(getattr(self.route, "value", self.route) or "") == ReviewRouteKind.API_CHAT.value

    @property
    def retrieves(self) -> bool:
        # DELIVERY class for admission/fit/authority; transport tests the route.
        return delivery_retrieves(self.route, self.subagent_id)


@dataclass
class ReviewRequest:
    surface: str
    goal: str
    scope: str = ""
    subject: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    checklist: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    slot_messages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    call_type: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    no_proxy: bool = False
    # RETRIEVING deliveries own a compact task and repository root: the session
    # route AND native API-route rows (`session_root`, `slot_session_tasks`).
    session_root: str = ""
    session_task: str = ""
    slot_session_tasks: Dict[str, str] = field(default_factory=dict)  # per-slot work order over session_task
    session_threads: Dict[str, str] = field(default_factory=dict)
    usage_attribution: Dict[str, str] = field(default_factory=dict)
    deadline_at: str = ""
    retry_key: str = ""
    reconcile_only: bool = False
    task_attempt: Any = None


@dataclass
class ReviewActorRecord:
    slot_id: str
    model: str
    status: str
    raw_text: str = ""
    parsed: Any = None
    # Per-actor parsed verdict (PASS/FAIL/DEGRADED/UNKNOWN). Carried here so the
    # objective axis can aggregate outcome_tier from only the actors that
    # CONTRIBUTED to a quorum PASS, instead of re-deriving the verdict downstream.
    signal: str = ""
    error: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    prompt_ref: Dict[str, Any] = field(default_factory=dict)
    response_ref: Dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    # Compact typed truth for task-result/event projection.  Raw model output
    # remains in the existing private audit record; these fields prevent UI
    # consumers from conflating a transport failure, malformed JSON, and a
    # valid semantic DEGRADED verdict.
    transport_status: str = ""
    # B1 typed failure facts, allowlist-carried off the exception's ATTRIBUTES
    # (generic across every ClaudexorUnavailable subclass; never exc.__dict__):
    # the machine code, the healing instant and the HTTP status survive the
    # substrate as fields instead of flattening into `error` prose.
    failure_code: str = ""
    reset_at: str = ""
    http_status: Optional[int] = None
    parse_status: str = ""
    semantic_verdict: str = ""
    provider: str = ""
    actor_role: str = ""
    coverage: Dict[str, Any] = field(default_factory=dict)
    # Participation is independent of agreement with the aggregate: every
    # contract-valid PASS/FAIL response counts, while enforcement_impact says
    # whether that participant supports completion or vetoes it.
    quorum_contribution: bool = False
    reason: str = ""
    enforcement_impact: str = ""
    # Physical operation identity survives a logical timeout.  A pending actor
    # is custody/reconciliation state, not permission for a blind resend.
    operation_id: str = ""
    operation_state: str = "settled"
    late_result_pending: bool = False


@dataclass
class ReviewRunResult:
    request: Dict[str, Any]
    actors: List[Dict[str, Any]]
    parsed_findings: List[Dict[str, Any]]
    aggregate_signal: str
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    # Bible P3: a single configured reviewer is honored but the lost cross-model
    # diversity is recorded LOUDLY and DURABLY here (centralized for every surface
    # that runs through ReviewCoordinator — acceptance, etc. — so a one-slot review
    # can never quietly look like an ordinary multi-reviewer PASS).
    single_reviewer_no_diversity: bool = False
    panel_id: str = ""


HARDNESS_ADVISORY_VISIBLE = "advisory_visible"  # fed back as a compact capsule, never blocks


HARDNESS_LABEL_ONLY = "label_only"              # recorded on the objective axis, not shown


HARDNESS_HARD_GATE = "hard_gate"                # blocking commit/scope immune gate (unchanged)


TYPED_FAILURE_FACT_KEYS = ("failure_code", "reset_at", "http_status", "transport_status")
