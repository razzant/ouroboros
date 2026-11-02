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

from dataclasses import dataclass, field, replace
import uuid
from typing import Any, Dict, List, Mapping, Optional

from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves


# One semantic author-finality record shared by review owners.  Surfaces keep
# their existing storage and reviewer evidence; this vocabulary only makes an
# author's final stance explicit and hash-bound when a review is advisory.
AUTHOR_DISPOSITION_VALUES = frozenset({"accepted", "rejected", "partial", "deferred"})
AUTHOR_ACTION_VALUES = frozenset({"finish", "stop"})


def build_author_disposition(
    *,
    disposition: str,
    rationale: str,
    subject_hash: str,
    reviewer_signal: str = "",
    enforcement: str = "",
    source: str = "author",
    recorded_at: str = "",
    action: str = "",
) -> Dict[str, Any]:
    """Build one bounded, current-subject author-finality record.

    This is a record helper, not a second review ledger.  Callers persist the
    returned object in their existing plan/skill/commit owners and
    continue to retain raw reviewer rows beside it.  A missing hash or reason
    is rejected so an author finish can never look like an unbound PASS.
    The author's act (``action``) is a final stance by itself: a record may
    carry it with an empty disposition, so no caller has to invent a stance.
    """
    value = str(disposition or "").strip().lower()
    reason = " ".join(str(rationale or "").split()).strip()
    subject = str(subject_hash or "").strip()
    if action and action not in AUTHOR_ACTION_VALUES:
        raise ValueError("AUTHOR_DISPOSITION_INVALID: unknown action")
    if value not in AUTHOR_DISPOSITION_VALUES and (value or not action):
        raise ValueError("AUTHOR_DISPOSITION_INVALID: unknown disposition")
    if not subject:
        raise ValueError("AUTHOR_DISPOSITION_INVALID: subject_hash is required")
    if not reason:
        raise ValueError("AUTHOR_DISPOSITION_INVALID: rationale is required")
    if len(reason) > 8_000:
        raise ValueError("AUTHOR_DISPOSITION_INVALID: rationale is too large")
    if not recorded_at:
        from ouroboros.utils import utc_now_iso

        recorded_at = utc_now_iso()
    return {
        "disposition": value,
        "rationale": reason,
        "subject_hash": subject,
        "reviewer_signal": str(reviewer_signal or "").strip(),
        "enforcement": str(enforcement or "").strip().lower(),
        "recorded_at": str(recorded_at),
        "source": str(source or "author"),
        **({"action": action} if action else {}),
    }


def validate_author_disposition(
    record: Any,
    *,
    subject_hash: str = "",
    allow_stale: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate and return a safe copy, rejecting malformed or stale records."""
    if not isinstance(record, dict):
        return None
    try:
        if "action" in record and record["action"] not in AUTHOR_ACTION_VALUES:
            return None
        normalized = build_author_disposition(
            disposition=record.get("disposition", ""),
            rationale=record.get("rationale", ""),
            subject_hash=record.get("subject_hash", ""),
            reviewer_signal=record.get("reviewer_signal", ""),
            enforcement=record.get("enforcement", ""),
            source=record.get("source", "author"),
            recorded_at=record.get("recorded_at", ""),
            action=record.get("action", ""),
        )
    except (TypeError, ValueError):
        return None
    if "review_reference" in record:
        import json
        try:
            if not isinstance(record["review_reference"], dict):
                return None
            normalized["review_reference"] = json.loads(json.dumps(record["review_reference"], allow_nan=False))
        except (TypeError, ValueError):
            return None
    from ouroboros.review_doctrine import carry_recorded_doctrine

    normalized = carry_recorded_doctrine(normalized, record)
    if normalized is None:
        return None
    expected = str(subject_hash or "").strip()
    if expected and normalized["subject_hash"] != expected and not allow_stale:
        return None
    return normalized


def recorded_author_stop(decision: Any) -> bool:
    """Whether a recorded task decision (the ``acceptance_decision`` shape historical
    task records carry) records the author's explicit stop (TZ-2 C4).

    The typed ``author_stop`` reason, or the structured stop the producer records
    under its TRUE terminal cause when the review rounds ran out: ``author_action``
    and the disposition's ``action`` are both ``stop``. A finish is never a stop.
    The twin of ``log_events.explicitAuthorStop``.
    """
    from ouroboros.outcomes import REASON_REVIEW_CYCLES_EXHAUSTED

    if not isinstance(decision, dict):
        return False
    author = decision.get("author_disposition")
    return decision.get("reason") == "author_stop" or (
        decision.get("reason") == REASON_REVIEW_CYCLES_EXHAUSTED and decision.get("author_action") == "stop"
        and isinstance(author, dict) and author.get("action") == "stop")


def build_author_disposition_from_mapping(
    value: Any, *, subject_hash: str, reviewer_signal: str = "", enforcement: str = "",
) -> Dict[str, Any]:
    """Parse the public author finish envelope, including an optional helper claim."""
    if not isinstance(value, dict) or set(value) - {"disposition", "rationale", "subagent_check"}:
        raise ValueError("AUTHOR_DISPOSITION_INVALID: envelope fields are invalid")
    built = build_author_disposition(
        disposition=value.get("disposition", ""), rationale=value.get("rationale", ""),
        subject_hash=subject_hash, reviewer_signal=reviewer_signal, enforcement=enforcement,
    )
    if "subagent_check" in value:
        from ouroboros.review_doctrine import parse_subagent_check

        built["subagent_check"] = parse_subagent_check(value.get("subagent_check"))
    return built


def review_outcome_received(actors: Any, *, findings: Any = (), terminal: bool = False) -> bool:
    """Separate received feedback/unavailability from wholly live review custody.

    Surface owners supply their recorded terminal fact. A local custody_lost
    outcome leaves remote uncertainty intact; it is not a locally running actor.
    Nested scope receipts retain each slot's state instead of its aggregate label.
    """
    rows = [row for row in actors or [] if isinstance(row, dict)]
    pending = False
    while rows:
        row = rows.pop()
        children = row.get("raw_results")
        if isinstance(children, list) and children:
            rows.extend(item for item in children if isinstance(item, dict))
            continue
        live = row.get("operation_state") in {"in_flight", "pending_dispatch"}
        pending = pending or live
        if live:
            continue  # A host pending placeholder is not a received critic payload.
        if (row.get("ok") is True or row.get("status") in {"responded", "ok", "empty", "parse_failure"}
                or row.get("operation_state") == "custody_lost" or row.get("error") or row.get("failure_code")):
            return True
    return not pending and (terminal or any(isinstance(item, dict) and
        (item.get("item") or item.get("summary")) for item in findings or []))


def review_slot_awaiting(row: Any) -> bool:
    """A PLANNED wait: the caller released the dispatch barrier before this slot
    answered. An answer that has not arrived is a gap, never a failure or a verdict."""
    return isinstance(row, dict) and row.get("operation_state") == "pending_dispatch"


def review_slot_unresolved(row: Any) -> bool:
    """No answer and NOT a planned wait: the logical window expired or the worker
    handle is gone. ``late_result_pending`` is true for these rows and for awaiting
    ones alike, so it never tells the two apart — only ``operation_state`` does."""
    return isinstance(row, dict) and row.get("operation_state") in {"in_flight", "custody_lost"}


def apply_review_model_override(slot: Any, overrides: Dict[str, dict], *, slot_id: str = "") -> Any:
    """Project an explicit owner model choice onto one frozen reviewer row.

    Identity, effort and delivery are immutable here. A referenced native actor
    remains native, and an agent-session row never becomes a raw model call.
    Settings and the original row are untouched; empty profile means Auto.
    """
    identity = slot_id or str(getattr(slot, "slot_id", "") or "")
    value = overrides.get(f"reviewer:{identity}")
    route = getattr(slot, "kind", getattr(slot, "route", ""))
    if not value or str(getattr(route, "value", route)) == "agent_session":
        return slot
    configured = hasattr(slot, "target_id")
    changes = {"target_id" if configured else "model": value["model"],
               "profile_id" if configured else "session_profile": value["model_account_override"],
               "use_local": bool(value["use_local"])}
    return replace(slot, **changes)


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
    # The catalog row this seat is (``slot_id == subagent_id`` for a pool row;
    # '' for a row built from a bare model). An IDENTITY fact only: it says
    # nothing about delivery (F8) — see ``native_retrieval_override``.
    subagent_id: str = ""
    # Host sampling hint, resolved at dispatch; an explicit temperature wins.
    default_temperature: float | None = None
    # The effort this row runs at because the CALLER declared it for one order
    # (plan review's ``reviewer_effort``): '' when the row's own effort, a
    # compound route slug or the surface setting applied. Disclosure for the
    # last-execution projection; identity already rides ``effort``.
    declared_effort: str = ""
    # Captured preference; empty explicitly preserves the legacy request shape. ``None`` is a slot
    # built without one (a bare model): the send then resolves its role's, as any unbound call does.
    processing_preference: Optional[str] = None
    # An api row's delivery, stated by whoever built the row: ``True`` — the
    # bounded native inspection episode on its own route (the catalog's
    # ``native``), ``False`` — the assembled packet (``packet``). The pool
    # builder sets it explicitly in both directions; ``None`` (a row built from a
    # bare model, or a session row) means the packet. Never derived from the
    # presence of a subagent id (F8).
    native_retrieval_override: Optional[bool] = None
    # Configured technical rights; absent historical seats preserve readonly.
    session_access: str = ""

    @property
    def native_retrieval(self) -> bool:
        return (
            str(getattr(self.route, "value", self.route) or "") == ReviewRouteKind.API_CHAT.value
            and self.native_retrieval_override is True
        )

    @property
    def retrieves(self) -> bool:
        # DELIVERY class for admission/fit/authority; transport tests the route.
        # A session row always retrieves; an api row retrieves exactly when it
        # reads natively. ``delivery_retrieves`` is the same predicate for
        # callers that hold only a route and the native fact.
        return delivery_retrieves(self.route, self.native_retrieval)


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
    # Historical: task acceptance's retrieving rows recorded their per-slot source
    # delivery here. Kept so recorded prompts still rebuild through
    # ``ReviewRequest(**stored)``; nothing fills or reads it now.
    slot_source_delivery: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_threads: Dict[str, str] = field(default_factory=dict)
    usage_attribution: Dict[str, str] = field(default_factory=dict)
    deadline_at: str = ""
    retry_key: str = ""
    reconcile_only: bool = False
    # Absolute ``time.monotonic()`` instant after which the coordinator stops
    # waiting for workers still in flight and returns their typed
    # ``pending_dispatch`` rows; ``None`` waits each slot's own logical window.
    drain_deadline: Optional[float] = None
    # Existing surface fingerprints, carried only to physical provenance.
    reconciliation_identity: Dict[str, Any] = field(default_factory=dict)
    task_attempt: Any = None
    default_temperature: float | None = None
    # The round its sends are billed to (#1544), set by resolve_review_wave or by a fan-out
    # naming one round for its rows. Attribution only: no identity, no prompt-cache split.
    resolved_wave_id: str = ""
    # Frozen once before fanout for independent writable checking copies.
    checking_source: Dict[str, Any] = field(default_factory=dict)


def new_review_wave_id() -> str:
    """A fresh review round id, for a review that names no wave and has no paid-cycle key."""
    return f"wave-{uuid.uuid4().hex[:16]}"


def resolve_review_wave(request: ReviewRequest, review_meta: Mapping[str, Any], inherited: str = "") -> str:
    """The round every reviewer send of ``request`` is billed to: the caller's wave
    (skill and plan review), the inherited scope's, the round already recorded on the
    request, the review's paid-cycle identity (``retry_key``: one per review cycle,
    shared by all its slots and by a resumed run of it), or a fresh
    id. Recorded as ``request.resolved_wave_id``, so the returned run and a replay of
    the stored request keep it; the caller's ``usage_attribution`` stays as given."""
    wave = str(review_meta.get("review_wave_id") or inherited or getattr(request, "resolved_wave_id", "")
               or getattr(request, "retry_key", "") or "") or new_review_wave_id()
    request.resolved_wave_id = wave
    return wave


@dataclass
class ReviewActorRecord:
    slot_id: str
    model: str
    status: str
    raw_text: str = ""
    parsed: Any = None
    # The declared contract is finalized once, before producer persistence.
    outcome: Dict[str, Any] = field(default_factory=dict)
    # Per-actor parsed verdict (PASS/FAIL/DEGRADED/UNKNOWN). Carried here so a
    # reader can take only the actors that CONTRIBUTED to the aggregate
    # (``review_verdict._contributing_actors``) instead of re-deriving the verdict.
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
    # What the engine REPORTED about the failed run, bounded and redacted at the gateway
    # (``run_failure_cause``); "" when nothing was reported. Opaque: shown, never branched on.
    # The ledger row keeps a plain fallback from the error text (``review_failure_facts``).
    reported_cause: str = ""
    # Full safe structured refusal, before the short reported_cause view.
    # These are evidence, never retry-policy inputs.
    failure_details: Dict[str, Any] = field(default_factory=dict)
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
    recovery_binding: Dict[str, Any] = field(default_factory=dict)
    # Wall clock at which THIS process sent this reviewer its request, for the
    # rows that are still waiting for an answer. Empty whenever the host did not
    # perform the send itself (a free replay, a rejoin of an earlier process's
    # paid operation): the owner is never shown an inferred moment.
    awaiting_since: str = ""
    # A durable ``review_model_error`` append already SUCCEEDED for this actor
    # (#1578): a positive receipt that rides the run result, never inferred from
    # status. The wave it ran in, as the request resolved it.
    error_event_logged: bool = False
    review_wave_id: str = ""



def review_failure_facts(row: Mapping[str, Any]) -> Dict[str, str]:
    """A short reason a failed seat's durable row keeps before its raw evidence can be pruned.

    Codes come from producers or structural status, never error-text matching; the cause is the
    producer's own words (redacted and bounded), never an invented sentence. Existing typed facts
    win; a seat that answered, or whose answer may still arrive, gains nothing.
    """
    code, cause = str(row.get("failure_code") or ""), str(row.get("reported_cause") or "")
    status, state = str(row.get("status") or ""), str(row.get("operation_state") or "")
    if state in {"in_flight", "pending_dispatch"} or (row.get("late_result_pending") and state != "custody_lost"):
        return {"failure_code": code, "reported_cause": cause}
    defaults = {
        "error": ("review_error", "Reviewer execution failed without an error message."),
        "not_dispatched": ("review_not_dispatched", "Reviewer was not dispatched."),
        "empty": ("review_empty_response", "Reviewer returned no answer."),
        "parse_failure": ("review_parse_failure", "Reviewer response did not satisfy the output contract."),
        "partial": ("review_partial_response", "Reviewer response did not cover the required items."),
    }
    if status in defaults:
        from ouroboros.gateways.claudexor import run_failure_cause

        fallback_code, explanation = defaults[status]
        code = code or fallback_code
        words = str(row.get("error") or row.get("reason") or row.get("raw_text") or "").strip()
        if not cause and words:
            if status in {"parse_failure", "partial"}:
                words = f"{explanation} {words}"
            cause = " ".join(run_failure_cause({"safeMessage": words}).split())
    return {"failure_code": code, "reported_cause": cause}


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
    # that runs through ReviewCoordinator, so a one-slot review can never quietly
    # look like an ordinary multi-reviewer PASS).
    single_reviewer_no_diversity: bool = False
    panel_id: str = ""
    # The resolved roster belongs to this operation, including overrides in force
    # at dispatch. Collection must not read a subsequently edited configuration.
    slot_roster: List[Dict[str, Any]] = field(default_factory=list)


HARDNESS_ADVISORY_VISIBLE = "advisory_visible"  # shown to the author, never blocks


HARDNESS_LABEL_ONLY = "label_only"              # recorded on the objective axis, not shown


HARDNESS_HARD_GATE = "hard_gate"                # blocking commit/scope immune gate (unchanged)


TYPED_FAILURE_FACT_KEYS = ("failure_code", "reset_at", "http_status", "transport_status", "reported_cause", "failure_details")


def review_failure_summary(row: Mapping[str, Any]) -> str:
    """A failed seat's saved facts for author handback, without a retry diagnosis."""
    import json
    from ouroboros.observability import redact_projection

    seat = str(row.get("slot_id") or row.get("seat_id") or "unknown seat")
    model = str(row.get("model_id") or row.get("model") or "model not reported")
    parts = [f"{seat} ({model})", str(row.get("status") or "unknown")]
    parts.extend(f"{key}={row[key]}" for key in TYPED_FAILURE_FACT_KEYS
                 if key != "failure_details" and row.get(key) not in (None, ""))
    details = row.get("failure_details")
    if isinstance(details, dict) and details:
        parts.append(json.dumps(details, ensure_ascii=False, sort_keys=True))
    elif not row.get("reported_cause"):
        parts.append(str(row.get("error") or row.get("raw_text") or "No cause was recorded."))
    return str(redact_projection(" · ".join(parts)).value)
