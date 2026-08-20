"""Subagent lane, cap, and metadata helpers.

THE OWNER-FACING AXES ARE THREE, AND EVERYTHING ELSE IS DERIVED.

A parent declares the WORK: ``write_surface`` (what the child may DO),
``model_lane`` (how good the answer must be) and ``executor`` (where it runs).
Model, effort, route and TOOL profile are consequences of those three plus the
owner's settings and what is live at the moment the child starts — never a second
thing the parent can ask for. The CREDENTIAL profile is NOT derived here at all
(see the DelegationRoute pin and the rotation note below): it is Claudexor's
choice, and the applied one comes back on the engine receipt.

``effort`` used to be a fourth public parameter and broke that twice. It was a
second knob for the question ``model_lane`` already answers, so ``model_lane:
light`` with ``effort: max`` was a request nobody could resolve (it pinned the
cheapest model to the strongest reasoning and no rule reconciled them). And a
harness route carries its OWN effort, so a parent asking ``low`` against a route
pinned to ``xhigh`` had no rule for who wins. It is removed rather than ranked
against the lane (BIBLE P2: remove the class). The owner still controls effort
exactly as before, through ``config.resolve_effort(task_type)``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Dict, List, Mapping

from ouroboros.config import (
    SETTINGS_DEFAULTS,
    get_heavy_model,
    get_light_model,
)

log = logging.getLogger(__name__)

# A lane names POWER and nothing else. `review` and `scope` were members until
# v6.87.7, where they named a TOPOLOGY — "fan out across the configured reviewer
# slots" — smuggled in through a strength parameter. Nothing in the product ever
# asked for them: every review surface (commit, plan, acceptance, skill) reads its
# slots straight from config through get_review_models/get_scope_review_models and
# runs them on the review substrate, never through schedule_subagent. They are
# removed rather than kept as aliases (BIBLE P2: remove the class). Durable records
# that still carry the old values stay readable — build_subagent_envelope already
# coerces an unknown stored lane instead of raising.
SUBAGENT_MODEL_LANES: frozenset[str] = frozenset({
    "auto",
    "main",
    "heavy",
    "light",
})

# The lane STRENGTH ordering (weak -> strong), and the only definition of "weaker
# than what was asked for" there is. Every axis in `capability_delta` compares
# requested against effective through a rank like this one — effort already had
# `config.effort_rank`, the lane had nothing, so each caller that wanted to know
# whether a lane had been lowered re-derived an answer locally or (more often)
# did not ask. `auto` is deliberately absent: it is a REQUEST to inherit, not a
# strength, so it has no rank and can never be compared.
LANE_STRENGTH: tuple[str, ...] = ("light", "main", "heavy")

# The lane a child gets when there is NO lane on record: the root agent's, which is
# Main. One name for it, because it is the answer in three places at once — the
# inheritance source for a root's child, the unknown-stored-lane coercion, and the
# completion-side envelope fallback.
LANE_OF_RECORD: str = "main"


def lane_rank(lane: str) -> int:
    """Index of a lane in LANE_STRENGTH (−1 for `auto`/unknown). Ordering SSOT."""
    value = str(lane or "").strip().lower()
    return LANE_STRENGTH.index(value) if value in LANE_STRENGTH else -1


def lane_is_weaker(lane: str, than: str) -> bool:
    """Whether ``lane`` is a weaker strength than ``than`` — THE definition of a lane
    reduction, and the only one.

    ``than`` is always a lane the resolution actually STARTED FROM: the request when
    it named a lane, the inherited parent lane when it did not. Never `auto`, which
    has no rank — comparing against a request to inherit answers nothing, and the
    v6.87.26 default made `auto` the COMMON request, so a comparison against it was
    a disclosure that could not fire on its own headline path.
    """
    return lane_rank(lane) < lane_rank(than)


# The third axis. POWER is the lane, AUTHORITY is the write surface, and this is
# WHO RUNS THE CHILD — the SUBSTRATE its cognition burns (metered API tokens vs. an
# already-paid subscription session hosted by Claudexor). It is deliberately not a
# harness name: Ouroboros asks for a capability and lets the owner's configuration
# decide which harness answers, so a harness identity never reaches core (AGENTS.md).
# `harness` is a pin, not a preference — when the configured harness is unavailable
# the request is refused rather than silently re-routed onto metered spend the caller
# did not ask for.
SUBAGENT_EXECUTORS: tuple[str, ...] = ("auto", "harness", "native")


def normalize_subagent_executor(value: Any) -> str:
    executor = str(value or "auto").strip().lower()
    if executor not in SUBAGENT_EXECUTORS:
        allowed = ", ".join(SUBAGENT_EXECUTORS)
        raise ValueError(f"executor must be one of: {allowed}")
    return executor


@dataclass(frozen=True)
class DelegatedRunShape:
    """The complete run shape a child's own authority entitles it to.

    Not a knob: every field follows from the ONE question ``delegated_run_shape``
    asks, and none of them appears in any tool schema, so the model has nothing to
    widen. It is derived here rather than at each consumer because the consumers are
    not one — the DISPATCHER health-checks the route before a token is spent and the
    NANNY builds the wire request — and a shape re-derived at each of them drifts:
    a change to the access profile that forgets the isolation, or to the isolation
    that forgets the delegated marker, is silent and unsafe in exactly one branch.
    """

    access: str
    mode: str
    isolation: str = ""
    delegated: bool = False


def delegated_run_shape(acting: bool) -> DelegatedRunShape:
    """The run shape for an acting (mutating) child, or for a read-only one.

    A MUTATING child runs ``live``: Claudexor edits the nanny's OWN worktree in place,
    so the nanny's existing workspace-patch capture sees the harness's edits with no
    new plumbing, and the same capture invalidates itself if the harness dared to
    commit. In place is also the ONE shape where Claudexor would otherwise hand the
    harness the operator's real ``$HOME`` — which holds the daemon control token — so
    ``delegated`` travels with it, inseparably, in the same record.

    A READ-ONLY child has nothing to write back, so it runs in Claudexor's default
    envelope, which is scoped already and needs no marker: that is one transport with
    one derived difference, not a second pipeline.
    """
    if acting:
        return DelegatedRunShape(access="workspace_write", mode="agent",
                                 isolation="live", delegated=True)
    return DelegatedRunShape(access="readonly", mode="ask")


@dataclass(frozen=True)
class DelegationRoute:
    """An OPAQUE Claudexor route plus optional model/effort.

    ``route_id`` is passed through to Claudexor verbatim as the primary harness.
    Ouroboros never interprets it — no ``if codex/claude/cursor`` anywhere (AGENTS.md).

    ``profile_id`` is the OPTIONAL manual credential pin: empty means the
    daemon's own rotation policy picks the account (D28 — the default); a
    value rides the run request verbatim as ``credentialProfileId``. Two
    authors: reviewer-slot rows (Q2-в) and the Delegation account pin
    (``OUROBOROS_SUBAGENT_PROFILE``, unified-accounts sprint D-U5). A pin is
    STRICT (D-U6): an exhausted pinned window is a typed refusal, never a
    silent rotation onto a sibling account.
    """

    route_id: str
    model: str = ""
    effort: str = ""
    profile_id: str = ""


def parse_subagent_harness(value: Any) -> DelegationRoute | None:
    """Parse ``harness[=model][:effort]`` — Claudexor's own reviewer-panel spelling.

    Two spellings mean "no delegated route", and the difference is OWNER intent,
    not runtime behaviour: empty is "never decided" (Settings may still offer the
    connected-subscription default), ``off`` is "the owner turned delegation off"
    and Settings leaves it off. Reserved token in Ouroboros's own setting
    vocabulary, not a harness name — no harness is ever branched on here.
    """
    raw = str(value or "").strip()
    if not raw or raw.lower() == "off":
        return None
    if "=" in raw:
        route_id, _, tail = raw.partition("=")
        model, _, effort = tail.partition(":")
    else:
        # `harness:effort` is legal grammar — the model bracket is optional, the
        # effort one is not tied to it. Splitting on `=` first made the WHOLE
        # string the route id, which then failed at dispatch as an unknown route.
        route_id, _, effort = raw.partition(":")
        model = ""
    route_id = route_id.strip()
    if not route_id:
        return None
    return DelegationRoute(route_id=route_id, model=model.strip(), effort=effort.strip())


def get_subagent_harness() -> DelegationRoute | None:
    """Read the NARROW ``OUROBOROS_SUBAGENT_HARNESS`` setting.

    This is the ONLY reader. The key is deliberately absent from
    ``provider_models.MODEL_SETTING_KEYS``: a session-only route is not an API model
    identity, and letting it into that sweep would poison credential planning,
    pricing, and bench provenance.
    """
    raw = str(
        os.environ.get("OUROBOROS_SUBAGENT_HARNESS", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_HARNESS", "")
    ).strip()
    route = parse_subagent_harness(raw)
    if route is None and raw and raw.lower() != "off":
        # A non-empty value that parses to nothing ("=opus") is a typo, not a
        # decision — silently equal to "never configured" it moved ALL delegation
        # onto metered API children with no trace. Say so where the operator looks.
        log.warning(
            "OUROBOROS_SUBAGENT_HARNESS is set but unparseable (%r) — "
            "delegation is OFF until it reads harness[=model][:effort]", raw,
        )
    if route is None:
        return None
    # The OPTIONAL account pin (D-U5), a SIBLING key rather than a fourth
    # grammar position: the route grammar stays Claudexor's own reviewer-panel
    # spelling, and this is the ONLY reader of the pin key. Empty = the
    # engine's quota-aware rotation pool (D28).
    profile = str(
        os.environ.get("OUROBOROS_SUBAGENT_PROFILE", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_PROFILE", "")
    ).strip()
    if profile:
        route = dataclass_replace(route, profile_id=profile)
    return route


# ---------------------------------------------------------------------------
# «Last delegated run» projection — the Subagents section's receipt line.
#
# The same disclosure pattern as reviewer_slot_last_execution.json (D22): what
# the last delegated subagent run REALLY ran as, written at the delegate settle
# seam, served through the gateway, rendered as ONE muted line. Disclosure
# only — nothing routes off it, and absence is shown as absence.
# ---------------------------------------------------------------------------

LAST_DELEGATION_FILENAME = "subagent_last_delegation.json"


def _last_delegation_path():
    import pathlib

    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / "state" / LAST_DELEGATION_FILENAME


def record_last_delegation(*, route: str, requested_model: str,
                           applied_model: str, run_id: str,
                           requested_profile: str = "",
                           applied_profile: str = "") -> None:
    """Record the last delegated run's route + requested/applied model + account.

    Best-effort and atomic, in the CANONICAL data plane beside the saved
    settings (the reviewer-slot projection's own rule): this is UI state, not
    per-task forensics — those live in the custody event log and the ledger.
    ``applied_model`` is the engine summary's own value, '' when the run never
    disclosed one — the requested model is never dressed up as the applied one.
    The same rule for the account (D-U5): ``applied_profile`` is the engine's
    ``authRoute.profileId`` settlement receipt, '' when telemetry predates it;
    ``requested_profile`` is the pin the request carried ('' = rotation) — the
    two stay separate so a requested-vs-ran mismatch is disclosable, never
    rewritten.
    """
    import json

    from ouroboros.utils import utc_now_iso, write_text_atomic

    try:
        path = _last_delegation_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent per run: a re-read of an ALREADY-terminal run must not
        # re-stamp `ts`, or the "N ago" line would call an old run fresh.
        if subagent_last_delegation().get("run_id") == str(run_id or ""):
            return
        write_text_atomic(path, json.dumps({
            "ts": utc_now_iso(),
            "route": str(route or ""),
            "requested_model": str(requested_model or ""),
            "applied_model": str(applied_model or ""),
            "requested_profile": str(requested_profile or ""),
            "applied_profile": str(applied_profile or ""),
            "run_id": str(run_id or ""),
        }, ensure_ascii=False, indent=1))
    except Exception:
        log.debug("subagent last-delegation projection write failed", exc_info=True)


def subagent_last_delegation() -> Dict[str, Any]:
    """Read the projection ({} on any read problem — disclosure only)."""
    import json

    try:
        data = json.loads(_last_delegation_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class SubagentExecutorResolution:
    """Outcome of the execution rule table (TZ 3.5)."""

    requested: str
    executor: str  # native | harness | blocked
    route: DelegationRoute | None = None
    reason: str = ""
    reset_at: str = ""

    @property
    def blocked(self) -> bool:
        return self.executor == "blocked"


def resolve_subagent_executor(
    requested: Any = "auto",
    *,
    route: DelegationRoute | None = None,
    unavailable_reason: str = "",
    reset_at: str = "",
) -> SubagentExecutorResolution:
    """The execution rule table. Pure — health facts are inputs, never probed here.

    | requested | state                    | behaviour                                  |
    |-----------|--------------------------|--------------------------------------------|
    | auto      | harness not configured   | native child                               |
    | auto      | configured and healthy   | nanny on the harness                       |
    | auto      | every profile exhausted  | native + LOUD typed fallback (D28)         |
    | auto      | otherwise unavailable    | native child WITH a visible marker         |
    | harness   | unavailable              | typed blocker — never silently spend API $ |
    | native    | any                      | native child                               |

    The executor AXIS — the module-level ``SUBAGENT_EXECUTORS`` vocabulary and its
    ``normalize_subagent_executor`` schema normalizer — belongs to the
    ``schedule_subagent`` schema work, which owns where the value ENTERS the system.
    This module deliberately does not define a second copy of either name: two
    top-level definitions of the same name in one module merge CLEANLY in git and
    then silently shadow each other. The guard below is a local fail-closed floor over
    the three cases this table actually specifies, so an unrecognized executor raises
    here instead of quietly behaving like ``auto``.
    """
    executor = str(requested or "auto").strip().lower()
    if executor not in ("auto", "harness", "native"):
        raise ValueError("executor must be one of: auto, harness, native")
    if executor == "native":
        return SubagentExecutorResolution(executor, "native", None, "requested_native")
    if route is None:
        reason = "harness_not_configured"
        if executor == "harness":
            return SubagentExecutorResolution(executor, "blocked", None, reason)
        return SubagentExecutorResolution(executor, "native", None, reason)
    if reset_at:
        # EVERY profile of the route is spent (the readiness predicate only reports a
        # reset when none has room left). Owner decision D28: `auto` FALLS BACK TO THE
        # METERED API, loudly — a typed decision made HERE, at the one point that costs
        # nothing yet, never a silent drift and never a wait. It used to dispatch the
        # child as a NANNY anyway, whose very first `delegate_start` was then refused
        # with this same fact: a spent dispatch, and the child left to improvise a
        # fallback in prose. Exhaustion heals on a timer, so `reset_at` rides along —
        # the child can say "I could have waited until X" and the parent can read it.
        #
        # An explicit `harness` request is the opposite answer and stays untouched: a
        # PIN exists precisely to avoid metered spend, so it blocks WITH the reset time.
        return SubagentExecutorResolution(
            executor,
            "blocked" if executor == "harness" else "native",
            route,
            "subscription_window_exhausted",
            reset_at,
        )
    if unavailable_reason:
        return SubagentExecutorResolution(
            executor,
            "blocked" if executor == "harness" else "native",
            route,
            unavailable_reason,
        )
    return SubagentExecutorResolution(executor, "harness", route, "harness_ready")


# Route health and its quota readers live in ouroboros/subagent_route_health.py
# (extracted at this module's size ceiling); re-exported here because the
# dispatcher, the delegate verbs, the reviewer slots and their monkeypatching
# tests all name them on THIS surface.
from ouroboros.subagent_route_health import (  # noqa: E402,F401 — re-exported public surface
    _cooldown_active,
    _exhausted_window,
    _model_scope_matches,
    route_health,
)




def probe_subagent_executor(
    requested: Any = "auto", *, shape: DelegatedRunShape | None = None,
) -> SubagentExecutorResolution:
    """Impure companion to the pure table: gather the health facts, then apply it.

    Kept separate so the rule table itself stays a pure function of stated facts. No
    route configured means no daemon call at all — the ordinary install pays nothing
    for an axis it does not use. ``shape`` defaults to the read-only shape: a caller
    that states nothing is asking for the narrowest run there is.
    """
    route = get_subagent_harness()
    if route is None:
        return resolve_subagent_executor(requested, route=None)
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    run_shape = shape if shape is not None else delegated_run_shape(False)
    gateway = None
    try:
        gateway = ensure_owned_gateway()
        unavailable, reset_at = route_health(
            gateway, route.route_id, run_shape, route_model=route.model,
            pinned_profile=route.profile_id,
        )
    except ClaudexorUnavailable as exc:
        return resolve_subagent_executor(
            requested, route=route, unavailable_reason=exc.code,
            reset_at=str(getattr(exc, "reset_at", "") or ""),
        )
    finally:
        if gateway is not None:
            gateway.close()
    return resolve_subagent_executor(
        requested, route=route, unavailable_reason=unavailable, reset_at=reset_at,
    )


def dispatch_executor_resolution(task: Mapping[str, Any]) -> SubagentExecutorResolution:
    """The executor axis of ONE dispatch: the typed rule table fed with live health.

    The whole shape, from the one owner of it: health is checked against the run
    this child would really start — a mutating child probes the `workspace_write`
    + delegated shape, a read-only one the narrow default — not against a profile
    string reassembled by the caller. `native` asks the daemon nothing. Tolerant
    of stored garbage: the schema refused anything invalid at schedule time, so an
    unknown STORED executor is stale data, not an argument error.
    """
    requested = str(task.get("requested_executor") or "auto").strip().lower() or "auto"
    if requested not in SUBAGENT_EXECUTORS:
        requested = "auto"
    if requested == "native":
        return resolve_subagent_executor("native")  # nothing to ask the daemon
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tool_access import predicted_subagent_profile

    constraint = normalize_task_constraint(task.get("task_constraint"))
    surface = str(getattr(constraint, "surface", "") or "")
    shape = delegated_run_shape(predicted_subagent_profile(write_surface=surface) == "acting_subagent")
    return probe_subagent_executor(requested, shape=shape)


@dataclass(frozen=True)
class SubagentLaneResolution:
    requested_lane: str
    effective_lane: str
    model: str
    use_local_model: bool = False
    # `slot_index`/`slot_count`/`downgrade_note` were removed in v6.87.28. The first
    # two carried a lane fan-out that no lane has had since v6.87.7 (a lane names
    # strength, and strength is one model), and the third had been the empty string
    # at every call site since the depth cap was deleted. A degenerate seam is still
    # a seam readers must reason about.
    # The concrete lane the resolution STARTED FROM — the request when it named a
    # lane, the inherited parent lane when it said `auto`. Without it the only
    # baseline available was the literal request, so an INHERITED heavy that landed
    # on Main compared `main` against `auto` and reported no reduction at all: the
    # v6.87.26 default was the one case its own invariant could not see.
    resolved_from: str = ""
    # WHERE `resolved_from` came from: "requested" (the request named a lane),
    # "inherited" (the request said `auto` and the parent's lane answered), or
    # "policy" (the request said `auto` and a dispatch policy answered — today the
    # light-lane default for harness-dispatched nannies). `effective_lane` may
    # still land elsewhere (an unconfigured slot resolves to Main); the provenance
    # names the DECISION source, never the slot outcome, so the two facts stay
    # separately readable on the durable delta.
    provenance: str = ""

    @property
    def reduced(self) -> bool:
        """Whether this lane landed below the lane it resolved from."""
        return lane_is_weaker(self.effective_lane, self.resolved_from)


def intended_lane(requested_lane: Any, parent_lane: Any = "") -> str:
    """The concrete lane a request MEANS: itself, or the parent's when it says `auto`.

    Pure INTENT — no environment, no live availability, no record. It is the half of
    the lane question that is answerable the moment the request is made, and it is
    named here because two places need it and neither may own it: the resolution
    (which measures the effective lane against it) and the ADMISSION gate for a
    `require_lane` delegation constraint, which runs before the child is dispatched
    and so cannot ask what lane the child ended up on. An admission gate that
    re-derived "auto means the parent's" on its own would be the second copy this
    module keeps deleting.

    Tolerant of stored garbage on both sides: a durable lane outlives the schema
    that wrote it, and neither a stale parent lane nor a stale request may make a
    child unschedulable.
    """
    try:
        requested = normalize_subagent_model_lane(requested_lane)
    except ValueError:
        requested = "auto"
    try:
        inherited = normalize_subagent_model_lane(parent_lane or LANE_OF_RECORD)
    except ValueError:
        inherited = LANE_OF_RECORD
    if inherited == "auto":
        # A stored `auto` is "no lane on record" wearing the request's clothes. Left
        # as-is it reaches `_lane_model` as an unknown lane and the child silently
        # drops to Light — the one value that is not a strength must not become one.
        inherited = LANE_OF_RECORD
    return inherited if requested == "auto" else requested


def normalize_subagent_model_lane(value: Any) -> str:
    lane = str(value or "auto").strip().lower()
    if lane not in SUBAGENT_MODEL_LANES:
        allowed = ", ".join(sorted(SUBAGENT_MODEL_LANES))
        raise ValueError(f"model_lane must be one of: {allowed}")
    return lane


def _slot_model(key: str) -> str:
    return str(os.environ.get(key, "") or SETTINGS_DEFAULTS.get(key, "") or "").strip()


_LANE_SLOT_KEYS = {
    "main": ("OUROBOROS_MODEL", "USE_LOCAL_MAIN"),
    "heavy": ("OUROBOROS_MODEL_HEAVY", "USE_LOCAL_HEAVY"),
    "light": ("OUROBOROS_MODEL_LIGHT", "USE_LOCAL_LIGHT"),
}


def lane_ran_on_main(lane: str, model: str) -> bool:
    """True when a Heavy/Light lane RESOLVED TO THE MAIN MODEL.

    ENV PRESENCE decides, not string equality: an absent/empty Heavy or Light slot
    is the inherit-from-Main case even when Main happens to equal that slot's
    shipped default, so substituting the default here would silently demand the
    lane's own local flag for a lane that is really running Main (v6.82: Light
    ships a real default, so "empty slot" alone no longer identifies the case).

    This predicate was buried inside ``_use_local_for_lane``, which is why the
    fact never reached anything else: the resolution kept reporting
    ``effective_lane="heavy"`` for a child that was really running Main.
    """
    if lane not in {"heavy", "light"}:
        return False
    env_slot = str(os.environ.get(_LANE_SLOT_KEYS[lane][0], "") or "").strip()
    return not env_slot or bool(model and model != env_slot)


def _use_local_for_lane(lane: str, model: str) -> bool:
    pair = _LANE_SLOT_KEYS.get(lane)
    if not pair:
        return False
    model_key, local_key = pair
    if lane_ran_on_main(lane, model):
        # Follows Main's local flag, so USE_LOCAL_MAIN governs the effective model
        # rather than being silently ignored.
        return _use_local_for_lane("main", model)
    slot_value = str(os.environ.get(model_key, "") or "").strip()
    if lane == "main":
        slot_value = slot_value or str(SETTINGS_DEFAULTS.get(model_key, "") or "").strip()
    return (
        bool(model)
        and model == slot_value
        and str(os.environ.get(local_key, "") or "").strip().lower() in {"1", "true", "yes", "on"}
    )


def _lane_model(lane: str) -> str:
    if lane == "heavy":
        return get_heavy_model()  # empty heavy slot -> main
    if lane == "light":
        return get_light_model()  # empty light slot -> main
    # `main` AND anything else. The fall-through used to be Light, so a lane that
    # slipped past normalization silently drew the CHEAPEST model — the opposite of
    # the default this module states everywhere else. An unknown lane is "no lane on
    # record", and that is LANE_OF_RECORD.
    return _slot_model("OUROBOROS_MODEL")


def resolve_subagent_lane(
    requested_lane: str,
    *,
    parent_lane: str = "",
    policy_default_lane: str = "",
) -> SubagentLaneResolution:
    """Resolve a subagent's effective lane + model.

    AUTHORITY AND POWER ARE ORTHOGONAL. What a child may DO comes from its
    ``task_constraint`` through ``tool_access.active_tool_profile``; how STRONG a
    child is comes from the lane the parent asked for, and from nothing else.

    Until v6.87.7 one line coupled them: ``auto`` resolved to ``heavy`` when the
    child was "mutating", where mutating meant ``write_surface OR may_mutate`` —
    and ``may_mutate`` per its own schema grants the right to spawn mutating
    DESCENDANTS. A read-only child therefore drew the expensive model because of
    a permission about its unborn grandchildren. That coupling arrived as a
    by-product of the CODE->HEAVY slot rename (952e210) and never had a stated
    rationale; it is deleted rather than re-tuned (BIBLE P2: remove the class).

    An omitted lane INHERITS THE PARENT'S effective lane. Until v6.87.26 it
    collapsed to ``light`` on the theory that silence means cheap work — but the
    parent that judges the work cheap can SAY ``light``, and silently demoting
    every unspecified child made the declaration invisible and the default wrong:
    a Heavy parent handing a child a piece of its own job got a Light child and no
    signal that it had happened. Inheritance makes the weak lane a stated
    decision instead of a hidden one, and a child that starts cheap and finds
    otherwise still raises its own power with ``switch_model`` (BIBLE P5). Depth
    does not rewrite the lane — recursion is bounded by the structural depth cap
    and by concurrency, which are the limits that actually bind — so depth is
    not an input here at all (the dead parameter was removed; the signature
    guard in test_resolver_takes_no_authority_argument watches for regrowth).

    ``parent_lane`` is the caller's OWN effective lane; empty means "no lane on
    record", which is the root agent, and the root runs Main.

    ``policy_default_lane`` is a DISPATCH policy's answer for an omitted lane —
    consulted only when the request says ``auto``, because an explicit lane always
    wins (the parent that names a lane has made the declaration this default
    exists to substitute for). Today's one policy: a harness-dispatched child is
    a NANNY whose own rounds are custody chores around a $0 delegated run, so it
    defaults to ``light`` instead of inheriting the parent's expensive lane (the
    poltergeist tree paid $87 of opus rounds for exactly this inheritance); it
    still raises itself with ``switch_model`` for real acceptance judgment. The
    provenance field records which source answered, so the durable delta can
    tell a policy default from an inheritance.
    """
    requested = normalize_subagent_model_lane(requested_lane)
    if requested != "auto":
        resolved_from, provenance = requested, "requested"
    elif policy_default_lane:
        resolved_from = normalize_subagent_model_lane(policy_default_lane)
        provenance = "policy"
    else:
        resolved_from, provenance = intended_lane(requested, parent_lane), "inherited"
    effective = resolved_from
    model = _lane_model(effective)
    if lane_ran_on_main(effective, model):
        # Report the slot the model ACTUALLY came from. Asking for Heavy on an
        # install with no Heavy slot gets the Main model, and calling that
        # `effective_lane="heavy"` made the record claim a strength nobody
        # configured — the reduction `capability_delta` now announces.
        effective = "main"
    return SubagentLaneResolution(
        requested_lane=requested,
        effective_lane=effective,
        model=model,
        use_local_model=_use_local_for_lane(effective, model),
        resolved_from=resolved_from,
        provenance=provenance,
    )


def preflight_native_fallback_lane(task: Mapping[str, Any]) -> SubagentLaneResolution:
    """The lane a preflight-falsified harness child runs NATIVE on (F10/sol #2).

    The B2 light-lane default is a policy about HARNESS-dispatched nannies —
    their own rounds are custody chores around a $0 delegated run. A child the
    delegate-visibility preflight just demoted to native executes the work
    ITSELF on metered tokens, so the policy must not survive the demotion: the
    lane re-resolves exactly as a native dispatch would have (explicit request,
    else parent inheritance — never policy-light), and the model/effort the
    record states follow the re-resolved lane.
    """
    requested_lane = str(task.get("requested_model_lane") or task.get("model_lane") or "auto")
    try:
        requested_lane = normalize_subagent_model_lane(requested_lane)
    except ValueError:
        requested_lane = "auto"
    return resolve_subagent_lane(
        requested_lane, parent_lane=str(task.get("parent_model_lane") or ""))


def preflight_native_fallback_dispatch(
    task: Mapping[str, Any], dispatch: "SubagentDispatch", reason: str,
) -> "SubagentDispatch":
    """The preflight's harness→native fallback, with the lane RE-RESOLVED (F10).

    The falsified dispatch resolved its lane under the harness policy (auto ⇒
    light) and measured its effort band against that cheap model. Keeping them
    would run a native child of a heavy parent on policy-light — the delta is
    rebuilt from the re-resolved lane: the lane axes, the effort re-measured
    against the model that will actually run, the executor reduction, and the
    preflight reason. The caller (``agent.preflight_delegate_visibility``)
    re-stamps the record and envelope; ``agent._prepare_task_context`` re-syncs
    the metadata projection and the ToolContext model override off them.
    """
    import dataclasses

    from ouroboros.config import effort_rank

    lane = preflight_native_fallback_lane(task)
    derived_effort = dispatch.delta.derived_effort
    effective_effort = _route_effort(lane.model, derived_effort)
    reasons: List[str] = []
    if derived_effort and effort_rank(effective_effort) < effort_rank(derived_effort):
        reasons.append(f"route_effort_ceiling={effective_effort}")
    if lane.reduced:
        reasons.append(f"lane_slot_unavailable={lane.resolved_from}")
    reasons.append(reason)
    delta = dataclasses.replace(
        dispatch.delta,
        resolved_lane=lane.resolved_from,
        effective_lane=lane.effective_lane,
        lane_provenance=lane.provenance,
        effective_effort=effective_effort,
        effective_executor="native",
        reduction_reasons=tuple(reasons),
        reason=derive_capability_reason(reasons),
        reduced=True,
    )
    return dataclasses.replace(
        dispatch,
        lane=lane,
        executor="native",
        route="",
        delta=delta,
        executor_resolution=dataclasses.replace(
            dispatch.executor_resolution,
            executor="native", reason=reason, reset_at="",
        ),
    )


def derive_capability_reason(reduction_reasons: Any, substrate_disclosures: Any = ()) -> str:
    """THE one author of ``CapabilityDelta.reason``: every writer derives the
    string from the typed lists through this join, so string and lists cannot
    diverge (B4); old hand-concatenated durable records stay readable, and
    ``str()`` keeps the join total over stored garbage."""
    return "; ".join(str(r) for r in [*(reduction_reasons or ()), *(substrate_disclosures or ())])


@dataclass(frozen=True)
class CapabilityDelta:
    """What was ASKED for versus what was GIVEN, on every axis at once.

    A child could land below its request in several unrelated places — an inherited
    lane, a lane slot that is not configured, an effort the route caps, an executor
    pin no route can honor — and not one of them announced itself. The reductions
    were spread across the resolver, the model getters, the LLM client and (for the
    executor) nowhere at all, so "what was asked" and "what was given" were never
    compared in the same place. This record is that place, and
    ``resolve_subagent_dispatch`` is its only author.

    The effort axis names a DERIVED value, not a request: since v6.87.28 no parent
    can ask for an effort, so what a route's learned band is measured against is the
    effort ``config.resolve_effort`` gives this task type. A route that caps it below
    the owner's own setting is still a reduction, and still the owner's business.
    """

    requested_lane: str = "auto"
    # What `requested_lane` MEANT: itself, or the parent's lane when it was `auto`.
    # The effective lane is measured against this, never against a bare `auto`.
    resolved_lane: str = "main"
    effective_lane: str = "main"
    # WHERE the resolved lane came from — "requested" / "inherited" / "policy"
    # (see SubagentLaneResolution.provenance). Additive disclosure: a policy
    # default (light-lane nanny) is not a REDUCTION relative to itself, so the
    # reduction phrases stay quiet about it — this field is what keeps the
    # decision source readable on the durable record anyway.
    lane_provenance: str = ""
    derived_effort: str = ""
    effective_effort: str = ""
    requested_executor: str = "auto"
    effective_executor: str = "native"
    reason: str = ""
    reduced: bool = False
    # Typed axes behind `reason` (B4, 7A): dispatch vs completion-seam facts are
    # UNRELATED; `reason` derives from these additive lists.
    reduction_reasons: tuple = ()
    substrate_disclosures: tuple = ()
    # A field that was accepted once, is still on durable records, and is now
    # IGNORED — stated here rather than dropped, because a value that silently stops
    # meaning anything is the same class of defect as a reduction nobody announces.
    # Not a reduction: nothing was taken away, the field simply no longer exists.
    legacy_note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_lane": self.requested_lane,
            "resolved_lane": self.resolved_lane,
            "effective_lane": self.effective_lane,
            "lane_provenance": self.lane_provenance,
            "derived_effort": self.derived_effort,
            "effective_effort": self.effective_effort,
            "requested_executor": self.requested_executor,
            "effective_executor": self.effective_executor,
            "reason": self.reason,
            "reduced": bool(self.reduced),
            "reduction_reasons": [str(r) for r in self.reduction_reasons],
            "substrate_disclosures": [str(s) for s in self.substrate_disclosures],
            "legacy_note": self.legacy_note,
        }


def capability_delta_disclosures(delta: Mapping[str, Any]) -> List[str]:
    """The human phrases a delta discloses — THE renderer, reading the plain dict.

    Both surfaces that show a delta read it from a dict off a durable record, and
    they used to render it twice: the parent's notice built its phrases from the
    dataclass while the child's prompt block re-assembled its own from the event,
    which is how the child could be told `auto->main` (its parent asked for nothing)
    while the parent read `auto(inherited heavy)->main`. One body, two callers.
    """
    parts: List[str] = []
    if lane_is_weaker(str(delta.get("effective_lane") or ""), str(delta.get("resolved_lane") or "")):
        parts.append(f"model_lane {lane_delta_phrase(delta)}")
    # RANK, not inequality: `effective_effort` is the whole learned band, so a route
    # with a floor can land ABOVE the derived effort, and "runs BELOW what was asked
    # for — effort none->low" is a false alarm the moment any OTHER axis puts this
    # delta into a reduction.
    from ouroboros.config import effort_rank

    derived = str(delta.get("derived_effort") or "")
    effective_effort = str(delta.get("effective_effort") or "")
    if derived and effort_rank(effective_effort) < effort_rank(derived):
        parts.append(f"effort {derived}->{effective_effort}")
    # `auto` is "no preference", so resolving it to a concrete route takes nothing
    # away — only an unhonored PIN belongs in the disclosure.
    requested_executor = str(delta.get("requested_executor") or "auto")
    effective_executor = str(delta.get("effective_executor") or "")
    if requested_executor != "auto" and effective_executor != requested_executor:
        parts.append(f"executor {requested_executor}->{effective_executor}")
    # Substrate facts (B4): SEPARATE entries after the slot axes, never spliced.
    parts.extend(
        str(fact) for fact in (delta.get("substrate_disclosures") or []) if str(fact))
    return parts


def lane_delta_phrase(delta: Mapping[str, Any]) -> str:
    """``asked->got`` for the lane, naming what an omitted request RESOLVED FROM.

    Takes the DICT because every surface that renders a delta reads it from a
    durable record, and a lane that reads `auto->main` in one of them tells the
    child its parent asked for nothing.
    """
    requested = str(delta.get("requested_lane") or "?")
    resolved = str(delta.get("resolved_lane") or "")
    effective = str(delta.get("effective_lane") or "?")
    if resolved and resolved != requested:
        return f"{requested}(inherited {resolved})->{effective}"
    return f"{requested}->{effective}"


def _route_effort(model: str, effort: str) -> str:
    """The effort this route will ACTUALLY run: the request clamped into the route's
    learned band by the SAME call the dispatcher makes.

    It used to read the ceiling alone, which is half of the band the dispatcher
    clamps to — so a route with a learned FLOOR (v6.73.2, endpoints where reasoning
    is mandatory) had its delta report the request verbatim while the call ran
    something else. Falls back to the request on any failure: a missing evidence
    store must never block scheduling.
    """
    try:
        from ouroboros.llm import LLMClient

        return str(LLMClient.clamp_effort_for_route(str(model or ""), effort) or effort)
    except Exception:  # pragma: no cover - evidence store is advisory
        log.debug("Effort-band lookup failed for %r", model, exc_info=True)
        return effort


# The durable keys a scheduling request may state. Everything the dispatch
# resolution derives is written under DIFFERENT keys, so "what was asked" can never
# be overwritten by "what was given" — the asymmetry `requested_model_lane` /
# `effective_model_lane` already had, extended to every axis.
SUBAGENT_INTENT_FIELDS: tuple[str, ...] = (
    "requested_model_lane",
    "parent_model_lane",
    # An ADMISSION fact carried as intent (F9): the lane an applicable
    # non-advisory `require_lane` delegation constraint verified this child
    # against at schedule time. The dispatch consults it to suppress the
    # harness light-lane policy default — a lane the gate enforced must not be
    # policy-overridden between admission and dispatch.
    "required_model_lane",
    "requested_executor",
)

# Fields a stored subagent record may still carry from a schema that no longer
# exists. They are IGNORED at dispatch and the reason is written onto the record;
# a load never fails over one (BIBLE P1: no silent loss, and no crash either).
LEGACY_SUBAGENT_FIELDS: Dict[str, str] = {
    "reasoning_effort": (
        "effort is no longer an owner-facing axis: it is derived from the owner's "
        "configured effort for this task type, because a public effort was a second "
        "knob for the question model_lane already answers"
    ),
}

# Everything the dispatch resolution writes onto the task record —
# ``SubagentDispatch.record_fields()``'s keys plus the envelope rebuilt from them.
# This is the merge set the worker reports back over the supervisor event channel
# (``task_dispatch_resolved``): the worker resolves on a process-local clone of the
# task, so the supervisor-owned RUNNING copy — the one persist_queue_snapshot
# serializes — must be told, or a restart loses the decision (XG-2R.1). The merge
# is key-scoped to exactly this set so a worker report can never overwrite
# scheduling intent or supervisor bookkeeping. Pinned against record_fields() by
# tests/test_model_slot_role_model.py.
SUBAGENT_RESOLUTION_FIELDS: tuple[str, ...] = (
    "effective_model_lane",
    "model",
    "use_local_model",
    "reasoning_effort",
    "effective_executor",
    "executor_route",
    "tool_profile",
    "capability_delta",
    "subagent_envelope",
)


@dataclass(frozen=True)
class SubagentDispatch:
    """Everything a child's execution is decided by, resolved ONCE, at dispatch.

    Model, effort, route and profile are DERIVED — from the three declared axes,
    from the owner's settings, and from what is live at the moment the child starts.
    They are not separate decisions taken in separate places at separate times,
    which is what they had become: the lane and a dead availability boolean resolved
    inside the scheduling tool call, while WHO actually ran the child was decided in
    the worker, and both wrote a record. Two resolvers with two records cannot be
    kept in agreement; there is one of each.
    """

    lane: SubagentLaneResolution
    effort: str
    executor: str
    route: str
    profile: str
    delta: CapabilityDelta
    # The typed executor resolution this dispatch was decided by (p34's rule
    # table). Carried whole so the surfaces that speak about the executor — the
    # child's substrate note, the blocked-outcome terminal, the events record —
    # read the SAME facts (reason, reset_at, route) this resolution produced,
    # instead of re-deriving them from strings.
    executor_resolution: SubagentExecutorResolution | None = None
    legacy_ignored: Dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """Whether this child MUST NOT run: an explicit executor pin no route honors.

        One predicate, so no reader compares against the literal `"blocked"` and no
        surface can forget that this outcome is not a kind of `native`.
        """
        return self.executor == "blocked"

    def record_fields(self) -> Dict[str, Any]:
        """The durable projection of this resolution — ONE writer for the one record.

        Every surface that persists a dispatched child (the running task result, the
        task-metadata projection, the completion write, the envelope) reads these
        keys off the task, so an added axis is one edit here instead of a field-by-
        field mapping repeated in four modules that drift apart one release later.
        """
        return {
            "effective_model_lane": self.lane.effective_lane,
            "model": self.lane.model,
            "use_local_model": self.lane.use_local_model,
            "reasoning_effort": self.effort,
            "effective_executor": self.executor,
            "executor_route": self.route,
            "tool_profile": self.profile,
            "capability_delta": self.delta.as_dict(),
        }


def resolve_subagent_dispatch(
    task: Mapping[str, Any], *, task_type: str = "",
) -> SubagentDispatch:
    """THE resolution point: where power, effort, route, profile and executor meet.

    It runs at DISPATCH, from the child's durable record, because that is the last
    moment at which the answer is still true. Two of its inputs are live — whether a
    harness route exists, and what band that route's model has learned — and a queued
    child can wait out a whole outage. Resolving at schedule time produced a record
    that claimed an answer about a moment that had already passed; resolving twice
    produced two records that disagreed about the same child.

    The record states INTENT only (``SUBAGENT_INTENT_FIELDS`` plus the task
    constraint); everything here is derived from it. A stored field from a retired
    schema is ignored and SAID SO rather than obeyed or dropped.
    """
    requested_lane = str(task.get("requested_model_lane") or task.get("model_lane") or "auto")
    try:
        requested_lane = normalize_subagent_model_lane(requested_lane)
    except ValueError:
        # Durable-data tolerance: the schema refused anything invalid when it was
        # asked for, so an unknown STORED lane is stale, not an argument error.
        requested_lane = "auto"
    requested_executor = str(task.get("requested_executor") or "auto")
    try:
        requested_executor = normalize_subagent_executor(requested_executor)
    except ValueError:
        requested_executor = "auto"

    # The executor axis lands on p34's typed rule table (H1: one resolver, one
    # vocabulary, `blocked` a typed third outcome), fed with LIVE route health at
    # this same dispatch moment. The p2-era `harness_delegation_route()` stub and
    # its tuple-returning twin resolver are deleted rather than kept beside it.
    # Resolved BEFORE the lane, because the lane's policy default depends on the
    # EFFECTIVE executor: a harness-dispatched child whose request said `auto`
    # defaults to the LIGHT lane (its own rounds are custody chores around a $0
    # delegated run — poltergeist phase B), while an explicit lane always wins
    # and a native child keeps inheriting its parent's lane unchanged.
    executor_resolution = dispatch_executor_resolution(task)
    executor = executor_resolution.executor
    route = executor_resolution.route.route_id if executor_resolution.route else ""
    # F9 (sol #1): a lane the ADMISSION gate enforced (`require_lane` delegation
    # constraint, stamped as `required_model_lane`) wins over the dispatch
    # policy default — the admission verified this child against that lane, and
    # the auto+harness ⇒ light policy silently landing it elsewhere would break
    # admission→dispatch consistency. With the policy suppressed, `auto`
    # inherits the parent's lane, which is exactly the lane the gate matched.
    required_lane = str(task.get("required_model_lane") or "").strip().lower()
    if required_lane not in SUBAGENT_MODEL_LANES or required_lane == "auto":
        required_lane = ""
    lane = resolve_subagent_lane(
        requested_lane, parent_lane=str(task.get("parent_model_lane") or ""),
        policy_default_lane=(
            "light" if executor == "harness" and not required_lane else ""),
    )
    # Only a REDUCTION belongs in the delta. The table also names the ordinary
    # outcomes (`requested_native`, `harness_ready`) and the no-preference case
    # (`auto` with no route configured, D28: nothing was asked for, no delta) —
    # none of which took anything away from the request.
    executor_reason = executor_resolution.reason
    if executor_reason in ("requested_native", "harness_ready") or (
        requested_executor == "auto" and executor_reason == "harness_not_configured"
    ):
        executor_reason = ""
    from ouroboros.config import effort_rank, resolve_effort

    derived_effort = resolve_effort(task_type or str(task.get("type") or "task"))
    effective_effort = _route_effort(lane.model, derived_effort)

    reasons: List[str] = []
    if effort_rank(effective_effort) < effort_rank(derived_effort):
        reasons.append(f"route_effort_ceiling={effective_effort}")
    if lane.reduced:
        reasons.append(f"lane_slot_unavailable={lane.resolved_from}")
    if executor_reason:
        reasons.append(executor_reason)

    # A record that carries its own capability_delta was ALREADY resolved once:
    # the stored reasoning_effort is the residue record_fields() wrote — which now
    # survives the queue snapshot and a crash-requeue — not a request from a
    # retired schema. A replay must not disclose a false "ignored" legacy note
    # about the resolver's own writing (XG-2R.1); a genuinely legacy record
    # pre-dates capability_delta and still gets the note.
    prior_resolution = bool(task.get("capability_delta")) and isinstance(task.get("capability_delta"), dict)
    legacy_ignored = {} if prior_resolution else {
        name: f"{name}={task.get(name)!r} ignored: {why}"
        for name, why in LEGACY_SUBAGENT_FIELDS.items()
        if str(task.get(name) or "").strip()
    }
    delta = CapabilityDelta(
        requested_lane=lane.requested_lane,
        resolved_lane=lane.resolved_from,
        effective_lane=lane.effective_lane,
        lane_provenance=lane.provenance,
        derived_effort=derived_effort,
        # The effort the route will ACTUALLY run — the whole learned band through
        # the same call the dispatcher clamps with, so a route with a learned FLOOR
        # is reported honestly and is not called a reduction.
        effective_effort=effective_effort,
        requested_executor=requested_executor,
        effective_executor=executor,
        reduction_reasons=tuple(reasons),
        reason=derive_capability_reason(reasons),
        reduced=bool(reasons),
        legacy_note="; ".join(sorted(legacy_ignored.values())),
    )
    from ouroboros.tools.control_delegation import profile_from_task_constraint

    constraint = task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {}
    return SubagentDispatch(
        lane=lane,
        # The DERIVED effort, not the clamped one. The dispatcher re-clamps per
        # model on every call, and a fallback route with a wider band must not
        # inherit this route's ceiling through the stored value; what this route
        # will really run is reported by the delta.
        effort=derived_effort,
        executor=executor,
        route=route if executor == "harness" else "",
        profile=profile_from_task_constraint(constraint),
        delta=delta,
        executor_resolution=executor_resolution,
        legacy_ignored=legacy_ignored,
    )


def build_subagent_envelope(
    *,
    task_id: str,
    parent_task_id: str = "",
    root_task_id: str = "",
    task_group_id: str = "",
    depth: int = 0,
    role: str = "",
    requested_lane: str = "auto",
    effective_lane: str = "",
    model: str = "",
    reasoning_effort: str = "",
    executor: str = "",
    effective_executor: str = "",
    executor_route: str = "",
    tool_profile: str = "",
    capability_delta: Dict[str, Any] | None = None,
    status: str = "",
    usage: Dict[str, Any] | None = None,
    cost_usd: float | None = None,
    execution_evidence: Dict[str, Any] | None = None,
    actual_substrate: str = "",
) -> Dict[str, Any]:
    usage_data = dict(usage or {})
    if cost_usd is None:
        try:
            raw_cost = usage_data.get("cost")
            if raw_cost is None:
                raw_cost = usage_data.get("cost_usd")
            cost_usd = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            cost_usd = None
    envelope = {
        "task_id": str(task_id or ""),
        "lineage": {
            "parent_task_id": str(parent_task_id or ""),
            "root_task_id": str(root_task_id or ""),
            "depth": int(depth or 0),
        },
        "task_group_id": str(task_group_id or ""),
        "role": str(role or ""),
        # Durable-data tolerance: this envelope is built for an ALREADY-RAN task from its
        # stored record, which may carry a legacy/unknown lane (e.g. a pre-v6.39 "code")
        # that the public schema now rejects. Coerce an unknown stored lane to a safe
        # default instead of raising (the public schedule_subagent schema stays strict —
        # this is NOT a "code"->"heavy" alias, just metadata robustness, symmetric with
        # the effective_lane guard that already existed). The unknown-lane default is
        # LANE_OF_RECORD, the same answer resolve_subagent_lane gives for "no lane on
        # record" — it was a hardcoded "light" here and another hardcoded "light" in the
        # completion-side caller, so the old default outlived itself in two copies.
        "requested_lane": normalize_subagent_model_lane(requested_lane if requested_lane in SUBAGENT_MODEL_LANES else "auto"),
        # EMPTY means NOT RESOLVED YET, and stays empty. A child between `requested`
        # and dispatch has an intent and no answer: the lane, the model and the
        # effort are derived when the child starts, from what is live then. The old
        # default said `light`, so a queued child's envelope named a lane, a slot and
        # a strength that no resolution had produced — a claim, not a record.
        "effective_lane": (
            normalize_subagent_model_lane(effective_lane if effective_lane in SUBAGENT_MODEL_LANES else LANE_OF_RECORD)
            if str(effective_lane or "").strip() else ""
        ),
        "model": str(model or ""),
        # DERIVED at dispatch (`resolve_subagent_dispatch`), not requested: no parent
        # can ask for an effort. Empty means the child has not been dispatched.
        "reasoning_effort": str(reasoning_effort or ""),
        "executor": str(executor or ""),
        # `executor` is the REQUEST (empty means "not requested"); this is who
        # actually ran the child. They were one field until v6.87.26, so a pin no
        # route could honor was reported back as if it had been honored — the
        # asymmetry `requested_model_lane`/`effective_model_lane` had avoided all
        # along, sitting one line above.
        "effective_executor": str(effective_executor or ""),
        # WHICH route ran it, and what the child could DO — the two remaining
        # derivations. A record naming `harness` without naming the route cannot be
        # audited, and the profile is the authority half of the same description.
        "executor_route": str(executor_route or ""),
        "tool_profile": str(tool_profile or ""),
        "capability_delta": dict(capability_delta or {}),
        "status": str(status or ""),
        "usage": usage_data,
        "cost_usd": round(float(cost_usd), 6) if cost_usd is not None else None,
    }
    if execution_evidence is not None:
        # ADDITIVE, completion-time only. `executor_route`/`effective_executor`
        # above are DISPATCH decisions and are never overwritten; this block is
        # the EVIDENCE those decisions are reconciled against — what the delegate
        # custody rows prove actually ran. Absent means "no evidence yet"
        # (pre-completion), never "ran natively".
        envelope["execution_evidence"] = dict(execution_evidence)
    if actual_substrate:
        # The FACT beside the plan (Q1A): harness_used / harness_attempted /
        # native_only, always beside the execution_evidence counts above.
        envelope["actual_substrate"] = str(actual_substrate)
        # C3 (additive): the counters are delegated-run FACTS; the metered/native
        # work interleaved beside them is not measurable from custody rows, so no
        # share, ratio, or dominance claim is derivable — said here so no reader
        # invents one from the enum.
        envelope["native_contribution"] = "unknown"
    return envelope


# The statuses at which execution evidence exists to reconcile: the child is over,
# so the custody rows are the complete story of its delegated runs. A RUNNING
# envelope carries no evidence — the neutral "dispatched" reading is the honest one.
_EVIDENCE_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

# The substrate FACT vocabulary (owner decision Q1A, 2026-08-10 amendments):
# `effective_executor`/`executor_route` stay the dispatch PLAN, `actual_substrate`
# is what the durable custody rows PROVE ran. Derived from custody evidence ONLY —
# never from usage/rounds, where delegate_wait polling and real native thinking
# are indistinguishable, so any boundary would be a guess. The raw attested
# counts always ride beside the enum on every surface that carries it.
SUBSTRATE_HARNESS_USED = "harness_used"            # >=1 delegated run succeeded
SUBSTRATE_HARNESS_ATTEMPTED = "harness_attempted"  # >=1 started, none succeeded
SUBSTRATE_NATIVE_ONLY = "native_only"              # no delegated run ever started


def actual_substrate(evidence: Mapping[str, Any] | None) -> str:
    """Classify what ACTUALLY ran, from durable custody evidence alone (Q1A)."""
    evidence = evidence if isinstance(evidence, Mapping) else {}

    def _count(key: str) -> int:
        try:
            return int(evidence.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    if _count("delegated_runs_succeeded"):
        return SUBSTRATE_HARNESS_USED
    if _count("delegated_runs_started"):
        return SUBSTRATE_HARNESS_ATTEMPTED
    return SUBSTRATE_NATIVE_ONLY


def substrate_result_fields(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Top-level durable-result mirror of the substrate FACT plus its raw counts.

    The enum never travels without the attested counts it was derived from, so
    a consumer of the durable result sees the full fact, not a classification.
    """
    if not envelope.get("actual_substrate"):
        return {}
    ev = envelope.get("execution_evidence")
    ev = ev if isinstance(ev, Mapping) else {}
    return {
        "actual_substrate": str(envelope["actual_substrate"]),
        "delegated_runs_started": int(ev.get("delegated_runs_started") or 0),
        "delegated_runs_settled": int(ev.get("delegated_runs_settled") or 0),
        "delegated_runs_succeeded": int(ev.get("delegated_runs_succeeded") or 0),
        "delegated_runs_failed": int(ev.get("delegated_runs_failed") or 0),
        # C3: delegated-run facts only — the native contribution is unknown and
        # no share is derivable (owner-approved replacement for harness_share).
        "native_contribution": "unknown",
    }


def _disclose_native_only_substrate(
    delta: Dict[str, Any], *, nudge_ignored: bool = False,
) -> Dict[str, Any]:
    """A harness dispatch that never started a delegated run is a REDUCED execution.

    Surfaced through the EXISTING capability_delta disclosure (owner decision:
    no new axis). Amends a COPY at the completion seam (the live task's dict
    stays untouched); facts land in ``substrate_disclosures`` and ``reason``
    re-derives (B4); a pre-lists legacy dict keeps its concatenated shape.
    ``nudge_ignored`` adds ``nanny_finalized_after_nudge_without_delegation``
    (3A): the nudge was really INJECTED (durable worker stamp, never the
    suppression-blind ctx flag) and the child COMPLETED with zero runs."""
    amended = dict(delta or {})
    facts = ["delegated_substrate_unused"]
    if nudge_ignored:
        facts.append("nanny_finalized_after_nudge_without_delegation")
    disclosures = [str(s) for s in (amended.get("substrate_disclosures") or [])]
    disclosures.extend(fact for fact in facts if fact not in disclosures)
    amended["substrate_disclosures"] = disclosures
    reduction = amended.get("reduction_reasons")
    if isinstance(reduction, list):
        amended["reason"] = derive_capability_reason(reduction, disclosures)
    else:
        # Legacy string-only record: preserve the historical concatenation idempotently.
        reason = str(amended.get("reason") or "")
        for fact in facts:
            if fact not in reason:
                reason = "; ".join(part for part in (reason, fact) if part)
        amended["reason"] = reason
    amended["reduced"] = True
    return amended


def _execution_evidence_for_task(task: Mapping[str, Any], status: str) -> Dict[str, Any] | None:
    """Evidence for the completion envelope, or None when there is nothing to say.

    Derived only for a child whose dispatch named a harness route — the native
    path has no delegation claim to reconcile — and only at a terminal status.
    Reads the same durable custody rows the delegate tools write, on the same
    canonical (budget) drive. Fail-soft: a completion must never break on a
    forensic read.
    """
    if str(status or "") not in _EVIDENCE_TERMINAL_STATUSES:
        return None
    if not str(task.get("executor_route") or "").strip():
        return None
    drive = str(task.get("budget_drive_root") or task.get("drive_root") or "").strip()
    if not drive:
        return None
    try:
        from ouroboros import delegate_custody as custody

        return custody.task_execution_evidence(drive, str(task.get("id") or ""))
    except Exception:
        log.debug("execution-evidence derivation failed", exc_info=True)
        return None


def envelope_from_task(
    task: Dict[str, Any],
    *,
    status: str,
    usage: Dict[str, Any] | None = None,
    cost_usd: float | None = None,
) -> Dict[str, Any]:
    """Build a child's envelope from its durable task record — the ONE mapping.

    The scheduler builds the same envelope from live locals; this is its twin, and
    the two were separate field-by-field mappings living in different modules. They
    had already drifted: the completion side re-derived the effective-lane fallback
    as a hardcoded ``"light"``, so a record missing that field came back describing
    a lane the resolver would never produce. Keeping the record->envelope mapping
    beside the envelope schema means an added axis is one edit, not two.

    The derived half of the record is exactly ``SubagentDispatch.record_fields()``,
    so a child that has not been dispatched yet reports empty derivations rather
    than a substituted default.
    """
    usage = usage or {}
    evidence = _execution_evidence_for_task(task, status)
    # Unreadable custody log: zero counts are UNKNOWN — no substrate claim, no
    # reduction amendment (docs/JSDoc contract; omission keeps the enum closed).
    claimable = evidence is not None and not evidence.get("evidence_read_failed")
    substrate = actual_substrate(evidence) if claimable else ""
    capability_delta = task.get("capability_delta") if isinstance(task.get("capability_delta"), dict) else {}
    if substrate == SUBSTRATE_NATIVE_ONLY and str(task.get("effective_executor") or "") == "harness":
        # Q1A: a harness dispatch that ended native_only carries the amended
        # copy. B3: only COMPLETED + the durable stamp + ZERO durable attempts
        # (a nanny refused typed after TRYING delegate_start did not ignore
        # the nudge — this stays an obedience fact, never an accusation).
        capability_delta = _disclose_native_only_substrate(
            capability_delta,
            nudge_ignored=(str(status or "") == "completed"
                           and bool(evidence.get("nanny_nudge_recorded"))
                           and not evidence.get("delegate_start_attempted")),
        )
    return build_subagent_envelope(
        task_id=str(task.get("id") or ""),
        parent_task_id=str(task.get("parent_task_id") or ""),
        root_task_id=str(task.get("root_task_id") or ""),
        task_group_id=str(task.get("task_group_id") or ""),
        depth=int(task.get("depth") or 0),
        role=str(task.get("role") or ""),
        requested_lane=str(task.get("requested_model_lane") or task.get("model_lane") or "auto"),
        effective_lane=str(task.get("effective_model_lane") or ""),
        model=str(task.get("model") or ""),
        reasoning_effort=str(task.get("reasoning_effort") or ""),
        executor=str(task.get("requested_executor") or ""),
        effective_executor=str(task.get("effective_executor") or ""),
        executor_route=str(task.get("executor_route") or ""),
        tool_profile=str(task.get("tool_profile") or ""),
        capability_delta=capability_delta,
        status=status,
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "rounds": int(usage.get("rounds") or 0),
        },
        cost_usd=cost_usd,
        execution_evidence=evidence,
        actual_substrate=substrate,
    )
