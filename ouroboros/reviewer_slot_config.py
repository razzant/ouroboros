"""The review pool: the marked rows of the subagent catalog, projected as reviewer rows.

PR-3 (one transition): a reviewer is a row of ``OUROBOROS_SUBAGENTS`` whose owner
marked it ``review_eligible`` and left it enabled; the former
``OUROBOROS_REVIEWER_SLOTS`` lanes are gone as a configuration surface, and every
review surface (commit gate, plan review, skill review, task acceptance, the
author's ``review_change`` wave) reads ONE builder, ``review_pool_slots``. A seat's
identity IS the row's stored id (``slot_id == subagent_id``); model, route, pin,
effort, processing and delivery are the row's own facts, read at wave time.

``delivery`` of an ``api_model`` row (``native``: bounded native tool rounds on the
row's route; ``packet``: the assembled pack) is a catalog field, native by default;
a session row always retrieves. ``ReviewSlot.native_retrieval`` derives from that
field alone — never from the presence of a subagent id.

The pool ignores the catalog's global switch (``payload.enabled`` governs
delegation, not review) and reads the catalog WITHOUT the delegation resolver; a
row with ``enabled: false`` is not in the pool. An empty pool is a configured fact
(``review_pool_state`` → ``empty``): a save that leaves rows but no marks is refused
unless the owner says ``allow_empty_review_pool`` (``review_pool_save_error``).

Malformed configuration RAISES: a typo mapped to ``api_chat`` would silently spend
the API money the owner moved the row off of; mapped to ``agent_session`` it would
silently delegate a row the owner never delegated.
"""

from __future__ import annotations

import contextlib as _contextlib
import contextvars as _contextvars
import json
from ouroboros.settings_integrity import runtime_environ
from ouroboros.model_slots import resolve_processing_preference
import pathlib
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

if TYPE_CHECKING:  # annotation-only; review_records imports this module's leaves at call time
    from ouroboros.review_records import ReviewSlot

from ouroboros.route_spec import (
    ROUTE_KIND_AGENT_SESSION as SHARED_ROUTE_KIND_SESSION,
    RouteSpec,
    compound_session_effort,
)

ROUTE_KIND_API = "api_chat"
ROUTE_KIND_SESSION = "agent_session"

# The saved delivery of an api row (``ConfiguredReviewerSlot.delivery``): the
# catalog's vocabulary (``configured_subagents.REVIEW_DELIVERIES``).
DELIVERY_NATIVE = "native"
DELIVERY_PACKET = "packet"


@dataclass(frozen=True)
class ConfiguredReviewerSlot:
    """One configured reviewer row: identity, delivery, strength."""

    slot_id: str
    kind: str  # api_chat | agent_session
    target_id: str  # API model id, or opaque ``harness[=model]`` session spec
    # Empty means a compound Cursor/Agy route's encoded effort when present,
    # otherwise the surface's established default.
    effort: str = ""
    # The opaque per-row session spec. Structured agent_session rows carry
    # their target here; api rows carry ''. Legacy session rows resolve the
    # same shared route once into this row so delivery/fingerprint see one fact.
    session_target: str = ""
    # Optional managed account pin for session or raw-model delivery; '' = Auto.
    profile_id: str = ""
    # Optional configured-subagent reference (OUROBOROS_SUBAGENTS row id).
    # Mutually exclusive with an inline route in the STORED form; when set, the
    # execution fields above were resolved from the frozen roster row at load
    # time and the roster stays their SSOT. '' = ordinary direct row.
    subagent_id: str = ""
    use_local: Optional[bool] = None  # Runtime task override only; never a second settings policy.
    processing_preference: str = ""  # Effective preference captured when the row is loaded.
    # An api row's saved delivery (the catalog field): "native" reads the
    # subject in bounded native tool rounds, "packet" receives the assembled
    # pack (the fallback for a model without tool calling). '' means native —
    # the catalog's default; the old "empty = packet" reading of a lane row
    # lives only inside the frozen migration reader (``review_pool_migration``).
    delivery: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind == ROUTE_KIND_SESSION

    @property
    def native_retrieval(self) -> bool:
        """An ``api_chat`` row that reads the subject itself in bounded native
        tool rounds: every api row whose delivery is not ``packet``. Derived
        from the delivery field ALONE — a catalog row's id says nothing about
        how it delivers (F8).

        Kept OFF the closed public route vocabulary (``api_chat`` stays the
        wire kind); executor selection and admission read this derived fact.
        """
        return self.kind == ROUTE_KIND_API and self.delivery != DELIVERY_PACKET

    @property
    def retrieves(self) -> bool:
        """Delivery class: the reviewer reads the subject with its own tools.

        THE predicate admission/fit/authority callers must use instead of
        route-name comparisons — a session row and a native-retrieval api row
        are one class here, and neither receives an assembled packet.
        """
        return self.is_session or self.native_retrieval


def _validate_concrete_session_target(route: RouteSpec, where: str) -> None:
    """A structured session row names one concrete delegated route.

    ``parse_route_spec`` owns the shared JSON shape and deliberately accepts an
    opaque target.  Reviewer rows additionally promise exact delivery, so a
    non-empty sentinel/malformed target that the canonical delegated-route
    parser resolves to ``None`` must be refused here instead of reaching a
    consumer that may interpret ``None`` as permission to use a shared route.
    """
    if not route.is_session or not route.target_id:
        return
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.subagents import parse_subagent_harness

    if parse_subagent_harness(route.target_id) is None:
        raise ValueError(
            f"{SUBAGENTS_SETTING}: {where} session target "
            f"{route.target_id!r} does not name a concrete harness route"
        )


def _catalog_row_slot(row: Any, settings: Any, *, where: str = "") -> ConfiguredReviewerSlot:
    """ONE catalog row as the frozen reviewer row every review surface runs.

    ``slot_id`` IS the row's stored id; model, route, pin, effort, processing and
    delivery are the row's own facts under ``settings`` (the same effective
    identity ``configured_subagents.engine_identity`` names). A session row
    additionally promises one concrete harness route, as a lane row did.
    """
    where = where or f"row {row.subagent_id!r}"
    route = row.route
    processing = resolve_processing_preference(
        override=row.processing_preference or None, settings=dict(settings or {}))
    if route.is_session:
        _validate_concrete_session_target(route, where)
        return ConfiguredReviewerSlot(
            slot_id=row.subagent_id, kind=ROUTE_KIND_SESSION, target_id=route.target_id,
            effort=row.effort, session_target=route.target_id, profile_id=route.credential_profile_id,
            subagent_id=row.subagent_id, processing_preference=processing,
        )
    return ConfiguredReviewerSlot(
        slot_id=row.subagent_id, kind=ROUTE_KIND_API, target_id=route.target_id,
        effort=row.effort, profile_id=route.credential_profile_id, subagent_id=row.subagent_id,
        processing_preference=processing, delivery=row.delivery or DELIVERY_NATIVE,
    )


def _catalog_settings(snapshot: Any = None) -> Any:
    """The settings plane the pool reads: an explicit settings snapshot (a task's,
    a save handler's incoming document, a benchmark container's), else the
    applied runtime environment."""
    return snapshot if snapshot is not None else runtime_environ()


def _parse_catalog(settings: Any) -> Any:
    """The catalog document itself (``parse_configured_subagents``), WITHOUT the
    delegation resolver: the pool must read a switched-off catalog's marked rows
    (review is not switched off by a transfer, F6). ``None`` when the key is
    absent or blank; a malformed document raises its typed error."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, parse_configured_subagents

    raw = settings.get(SUBAGENTS_SETTING) if hasattr(settings, "get") else None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return parse_configured_subagents(raw)


def _pool_rows(settings: Any) -> List[ConfiguredReviewerSlot]:
    """The pool under ``settings``: enabled rows the owner marked review-eligible,
    in catalog order. The catalog's global switch is deliberately not consulted."""
    config = _parse_catalog(settings)
    if config is None:
        return []
    return [
        _catalog_row_slot(row, settings, where=f"items[{index}]")
        for index, row in enumerate(config.items)
        if row.enabled and row.review_eligible
    ]


def review_pool_rows(snapshot: Any = None) -> List[ConfiguredReviewerSlot]:
    """The pool as frozen reviewer rows (not yet delivery slots): catalog order,
    ``slot_id == subagent_id``. ``snapshot`` as in :func:`review_pool_slots`."""
    return _pool_rows(_catalog_settings(snapshot))


def catalog_review_row(snapshot: Any, selector: str) -> ConfiguredReviewerSlot:
    """ONE enabled catalog row named by its handle or stored id, as a reviewer row.

    The author's ``review_change`` names any enabled row — marked or not — as a
    seat of its own wave; the catalog's global switch is not consulted (it
    governs delegation). An unknown or ambiguous selector, a disabled row, an
    absent or malformed catalog are the parser's typed ``ValueError``.
    """
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, resolve_roster_selector

    settings = _catalog_settings(snapshot)
    config = _parse_catalog(settings)
    if config is None:
        raise ValueError(f"{SUBAGENTS_SETTING} is not configured; no row is named {selector!r}")
    row, code, detail = resolve_roster_selector(config, str(selector or ""), settings)
    if row is None:
        raise ValueError(f"{SUBAGENTS_SETTING}: {code}: {detail}")
    if not row.enabled:
        raise ValueError(f"{SUBAGENTS_SETTING}: row {selector!r} is switched off (enabled: false)")
    return _catalog_row_slot(row, settings)


def _stored_id_for_selector(selector: str) -> str:
    """The stored id a catalog handle names, or '' when nothing resolves."""
    from ouroboros.configured_subagents import resolve_roster_selector

    try:
        settings = _catalog_settings()
        config = _parse_catalog(settings)
        row = resolve_roster_selector(config, selector, settings)[0] if config is not None else None
    except ValueError:
        return ""
    return row.subagent_id if row is not None else ""


def review_pool_state(raw: Any) -> Dict[str, str]:
    """``{"state": structured|empty|error, "error": str}`` of ONE catalog document.

    ``structured``: at least one enabled, marked row; ``empty``: a readable
    catalog (or none) with no such row — a configured fact, never an absence;
    ``error``: a malformed catalog, with its typed text.
    """
    try:
        rows = _pool_rows({"OUROBOROS_SUBAGENTS": raw})
    except ValueError as exc:
        return {"state": "error", "error": str(exc)}
    return {"state": "structured" if rows else "empty", "error": ""}


def review_pool_save_error(raw: Any, *, allow_empty: bool) -> str:
    """The SAVE-path judge of an empty pool; ``""`` means acceptable.

    A catalog that has rows but no review mark would silently leave every review
    surface without a reviewer, so the save is refused (400) unless the owner
    says ``allow_empty_review_pool`` — the one place "empty" is confirmed rather
    than inferred. An absent catalog, a catalog without rows, or a marked row
    pass; a malformed catalog returns its typed text.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return ""
    try:
        config = _parse_catalog({"OUROBOROS_SUBAGENTS": raw})
    except ValueError as exc:
        return str(exc)
    if config is None or not config.items or allow_empty:
        return ""
    if any(row.enabled and row.review_eligible for row in config.items):
        return ""
    return "no reviewers marked; mark at least one row or save with `allow_empty_review_pool`"


# ---------------------------------------------------------------------------
# The composed pool: one ``review_change`` wave's seats.
# ---------------------------------------------------------------------------


class PoolSeat(NamedTuple):
    """One seat of a composed wave: the reviewer row as it runs, the parts of the
    brief it judges (``change`` / ``coupling``), and whether the author added it
    beside the configured pool (``additional``: it is heard, not counted)."""

    slot: ReviewSlot
    parts: Tuple[str, ...]
    additional: bool = False


# Context-local: a concurrent wave on another thread keeps the configured pool,
# while the wave's own threads run under ``contextvars.copy_context`` and read
# this composition.
_COMPOSED_POOL: "_contextvars.ContextVar[Optional[Tuple[PoolSeat, ...]]]" = _contextvars.ContextVar(
    "review_composed_pool", default=None)


@_contextlib.contextmanager
def composed_review_pool(seats: Sequence[PoolSeat]):
    """Every pool reader in this block sees exactly these seats (``review_pool_slots``
    returns their rows in this order; ``composed_pool_seats`` their parts)."""
    token = _COMPOSED_POOL.set(tuple(seats))
    try:
        yield
    finally:
        _COMPOSED_POOL.reset(token)


def composed_pool_seats() -> Optional[Tuple[PoolSeat, ...]]:
    """The composition in force, or ``None`` outside a composed wave (the pool
    projection reads a seat's ``parts`` from here; the ledger's ``seat_parts``
    derives them from delivery for the configured pool)."""
    return _COMPOSED_POOL.get()


def row_at_effort_order(row: ConfiguredReviewerSlot, effort: str) -> Optional[ConfiguredReviewerSlot]:
    """The row under a caller's effort order (``row_effort``'s rule): ``None`` for a
    compound Cursor/Agy route, whose encoded effort is the route's identity."""
    return None if _compound_effort(row) else replace(row, effort=effort)


def reviewer_slot_config_error() -> str:
    """The pool's configuration error — the catalog's row-precise parse text — or ''.

    Thin facade for the surfaces that must refuse loudly instead of reviewing
    with no pool at all (plan review, skill review, the settings panel). An
    empty pool is NOT an error here (it is a configured fact every surface
    reports as ``pool_empty``). No caching: the check re-parses so a hot-reloaded
    fix is seen immediately."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING

    return review_pool_state(_catalog_settings().get(SUBAGENTS_SETTING))["error"]


# ---------------------------------------------------------------------------
# Consumer accessors.
# ---------------------------------------------------------------------------


def _delivery_slot(
    row: ConfiguredReviewerSlot, *, role_hint: str,
    default_effort: str = "", effort_fallback: str = "", **slot_fields: Any,
) -> Any:
    """ONE configured row as the substrate's ``ReviewSlot``, carrying its own
    delivery: the route kind, the opaque session target and credential pin, the
    catalog binding, and — on an api row — the explicit native/packet fact."""
    from ouroboros.config import resolved_review_model_target
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    # ABI-4: the local-route fact is read off the typed target constructed at
    # the review seam, not re-derived per model string here.
    return ReviewSlot(
        slot_id=row.slot_id,
        model=row.target_id,
        effort=row_effort(row, default=default_effort, fallback=effort_fallback),
        # "this row runs at the caller's order": every row but a compound route slug.
        declared_effort=default_effort if default_effort and not _compound_effort(row) else "",
        role_hint=role_hint,
        use_local=(row.use_local if row.use_local is not None else resolved_review_model_target(row.target_id).provider_route == "local"),
        route=(ReviewRouteKind.AGENT_SESSION if row.is_session
               else ReviewRouteKind.API_CHAT),
        session_target=row.session_target,
        session_profile=row.profile_id,
        subagent_id=row.subagent_id,
        processing_preference=row.processing_preference,
        # The row's saved delivery is its own fact, stated explicitly in BOTH
        # directions on an api row (F8); a session row always retrieves (None).
        native_retrieval_override=None if row.is_session else bool(row.native_retrieval),
        **slot_fields,
    )


def review_pool_slots(
    snapshot: Any = None,
    *,
    role_hint: str = "",
    default_effort: str = "",
    **slot_fields: Any,
) -> List[Any]:
    """THE review pool as ``ReviewSlot`` rows — the one builder every review
    surface reads: the commit gate (through ``commit_triad_delivery``'s aligned
    vectors), plan review, skill review, task acceptance and the author's own
    wave. Inside ``composed_review_pool`` it is that wave's seats, in order.

    The pool is the catalog's enabled rows the owner marked review-eligible, in
    catalog order, read from ``snapshot`` (a task's frozen settings) or the
    applied runtime settings — never through the delegation resolver, and never
    gated by the catalog's global switch. Each row rides its own delivery and
    identity (``slot_id == subagent_id``). Effort: a caller's ``default_effort``
    (a wave's order) outranks everything but a compound Cursor/Agy route slug,
    whose encoded effort is the route's identity; with no order it is the row's
    own value, else that compound value, else ``REVIEW_POOL_DEFAULT_EFFORT``.
    ``slot_fields`` are the caller's per-surface ReviewSlot properties (timeout,
    output budget, temperature). A malformed catalog RAISES ValueError — every
    surface turns that into its typed refusal; a valid catalog with no marked
    row is ``[]`` (the surface reports ``pool_empty``, it does not fall back).
    """
    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT

    composed = _COMPOSED_POOL.get()
    if composed is not None:
        extra = {**({"role_hint": role_hint} if role_hint else {}), **slot_fields}
        return [replace(seat.slot, **extra) if extra else seat.slot for seat in composed]
    return [
        _delivery_slot(
            row, role_hint=role_hint,
            default_effort=default_effort, effort_fallback=REVIEW_POOL_DEFAULT_EFFORT, **slot_fields,
        )
        for row in _pool_rows(_catalog_settings(snapshot))
    ]


def triad_delivery_slots(
    *,
    role_hint: str = "",
    default_effort: str = "",
    **slot_fields: Any,
) -> List[Any]:
    """The pool under its historical name: an alias of ``review_pool_slots`` for
    the surfaces that still spell the builder this way (plan review, skill
    review, task acceptance, the commit gate's projection)."""
    return review_pool_slots(role_hint=role_hint, default_effort=default_effort, **slot_fields)


def child_acceptance_slots(slots: Sequence[Any], reviewer_slot_id: str = "") -> Tuple[List[Any], Dict[str, Any]]:
    """At most ONE pool row for a child task's acceptance (#1334).

    One reviewer is panel BREADTH, not a reading or round cap: the row keeps its
    own delivery. The only pool row needs no name; otherwise the caller names a
    member — by its stored id (``slot_id == subagent_id``) or its catalog handle
    — and the host checks membership. Returns ``(slots, refusal)``; a refusal is
    a typed ``not_dispatched`` payload, never a review run.
    """
    rows = list(slots or [])
    wanted = str(reviewer_slot_id or "").strip()
    if wanted:
        chosen = [slot for slot in rows if wanted in (slot.slot_id, getattr(slot, "subagent_id", ""))]
        if not chosen:
            chosen = [slot for slot in rows if slot.slot_id == _stored_id_for_selector(wanted)]
        if chosen:
            return chosen, {}
        reason = "reviewer_slot_unknown"
    elif len(rows) <= 1:
        return rows, {}
    else:
        reason = "reviewer_selection_required"
    return [], {
        "status": "not_dispatched", "reason": reason,
        "detail": ("a child task's acceptance uses at most one review-pool row; name one "
                   "with reviewer_slot_id (its id or catalog handle) — no reviewer was called"),
        "reviewer_rows": [{"slot_id": str(getattr(slot, "slot_id", "") or ""),
                           "model": str(getattr(slot, "model", "") or ""),
                           "route": str(getattr(getattr(slot, "route", ""), "value", getattr(slot, "route", "")) or ""),
                           "delivery": ("native" if getattr(slot, "native_retrieval", False)
                                        else "agent_session" if getattr(slot, "retrieves", False) else "packet")}
                          for slot in rows],
    }


def commit_triad_delivery() -> Dict[str, Any]:
    """Aligned per-row delivery vectors for the commit triad and skill review.

    Those surfaces consume rows as parallel lists (models for display and slot
    construction, routes for delivery, efforts/session targets/ids as row
    properties); projecting them from ``triad_delivery_slots`` keeps the
    surfaces at their size gates, keeps the vectors impossible to misalign,
    and keeps ONE reader of the triad rows. Raises ValueError on a malformed
    configuration — the caller turns that into its typed infra block.
    """
    from ouroboros.model_wait import current_model_wait
    from ouroboros.review_ledger import seat_parts
    from ouroboros.review_records import apply_review_model_override

    slots = triad_delivery_slots(role_hint="multi-model review")
    waiter = current_model_wait()
    slots = [apply_review_model_override(slot, waiter.overrides) for slot in slots] if waiter else slots
    # Inside a composed wave the rows ARE its seats, in order (``review_pool_slots``);
    # their parts and the added-seat bit ride the same index.
    composed = composed_pool_seats() or ()
    return {
        "models": [slot.model for slot in slots],
        "routes": [slot.route for slot in slots],
        "efforts": [slot.effort for slot in slots],
        "session_targets": [slot.session_target for slot in slots],
        "session_profiles": [slot.session_profile for slot in slots],
        "slot_ids": [slot.slot_id for slot in slots],
        "subagent_ids": [slot.subagent_id for slot in slots],
        # Per-row delivery class (#1334, F8): the row's own explicit fact; no
        # consumer may infer it from the catalog id every pool row carries.
        "retrieves": [bool(slot.retrieves) for slot in slots],
        "use_local": [slot.use_local for slot in slots],
        # What each seat is ASKED: a composed seat's own parts (a ``coupling_only``
        # seat answers Part 2 alone), else derived from its delivery (``seat_parts``).
        "parts": [tuple(composed[i].parts) if i < len(composed) else tuple(seat_parts(slot))
                  for i, slot in enumerate(slots)],
        # A seat the author added beside the configured pool is heard, not counted.
        "additional": [bool(composed[i].additional) if i < len(composed) else False for i in range(len(slots))],
        # The pool is always a configured panel (the migration mints the factory
        # rows into the catalog), so the pre-structured all-packet identity of
        # the skill-review fingerprint never applies to it.
        "legacy_skill_fingerprint": False,
    }


def row_plan_retrieves(row_plan: Dict[str, Any], index: int) -> bool:
    """Read the aligned delivery vector; a row the vector does not cover is its
    route's own class with no native retrieval (a session retrieves, an api row
    receives the packet) — never inferred from an actor id (F8)."""
    flags = list(row_plan.get("retrieves") or [])
    if index < len(flags):
        return bool(flags[index])
    from ouroboros.review_execution import delivery_retrieves

    routes = list(row_plan.get("routes") or [])
    return index < len(routes) and delivery_retrieves(routes[index], False)


def _compound_effort(row: ConfiguredReviewerSlot) -> str:
    """A Cursor/Agy compound route slug's encoded effort, '' for every other row.
    That effort is the route's model identity: sending ``model=…-xhigh`` with
    ``effort=low`` is the contradiction ``validate_compound_session_effort``
    already refuses at save time, so no caller's order may override it."""
    if row.is_session:
        return compound_session_effort(RouteSpec(
            kind=SHARED_ROUTE_KIND_SESSION,
            target_id=row.session_target or row.target_id,
            credential_profile_id=row.profile_id,
        )) or ""
    return ""


def _row_own_effort(row: ConfiguredReviewerSlot) -> str:
    """The effort the ROW itself carries: its explicit field, else a Cursor/Agy
    compound slug's encoded effort; '' when the row leaves it to its caller."""
    return row.effort or _compound_effort(row)


def row_effort(
    row: ConfiguredReviewerSlot,
    *,
    default: str = "",
    fallback: str = "",
) -> str:
    """Resolve one effort authority without contradicting a compound route.

    A caller's ``default`` is an ORDER for this run (a plan envelope's
    ``reviewer_effort``): it outranks the owner's per-row pin on every row except
    a Cursor/Agy compound slug, whose encoded effort is the route's identity and
    stays. Only plan review passes an order; commit, skill, acceptance and deep
    review call without one, and for them an explicit row field wins, then a
    compound slug's encoded effort, then ``fallback`` — by default the pool's
    ``REVIEW_POOL_DEFAULT_EFFORT``, also for a caller-built row such as the deep
    review's Main row. No surface setting is read: the lane-era effort keys are
    retired and an exported one is inert.
    """
    if default and not _compound_effort(row):
        return default
    own = _row_own_effort(row)
    if own:
        return own
    if fallback:
        return fallback
    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT

    return REVIEW_POOL_DEFAULT_EFFORT


# ---------------------------------------------------------------------------
# «Выполняется как» (D22): the last EFFECTIVE execution per seat.
#
# The UI projection of capability_delta — beside each SAVED catalog row, what the
# row REALLY ran as last time (route, model, effort, verdict method, any deltas),
# keyed by ``slot_id`` (== the row's ``subagent_id``). Disclosure, never
# enforcement: nothing reads this back into routing.
# ---------------------------------------------------------------------------

LAST_EXECUTION_FILENAME = "reviewer_slot_last_execution.json"
_LAST_EXECUTION_CAP = 64  # the pool is ≤ MAX_CONFIGURED_SUBAGENTS (26) rows; the cap only bounds junk growth


def _last_execution_path() -> "pathlib.Path":
    import pathlib

    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / "state" / LAST_EXECUTION_FILENAME


# `run_parallel_review` runs the triad and the scope surfaces CONCURRENTLY, in two
# threads of one process, and each finishes by folding its own rows into this one
# file. `write_text_atomic` makes the write untearable but cannot make the
# read-modify-write around it atomic: both threads read the same "before", and the
# slower one wrote its rows over the faster one's. The surface that vanished was
# whichever finished first — so the panel silently lost a whole row's «Выполняется
# как» line. In-process lock only: the concurrency is threads, not processes.
_LAST_EXECUTION_LOCK = threading.Lock()


def reviewer_slot_execution_rows(surface: str, actors: Any, slots_by_id: Dict[str, Any], *,
                                 record_id: str = "") -> Dict[str, Dict[str, Any]]:
    """One last-execution row per settled actor, keyed by slot id (pure: nothing written).

    The wave that ran these actors keeps the returned rows as ITS facts; the shared
    projection written by :func:`record_reviewer_slot_executions` may be overwritten by
    another surface's later run of the same seat."""
    from ouroboros.review_substrate import TYPED_FAILURE_FACT_KEYS
    from ouroboros.utils import utc_now_iso

    rows: Dict[str, Dict[str, Any]] = {}
    for actor in actors or []:
        slot = slots_by_id.get(getattr(actor, "slot_id", ""))
        if slot is None:
            continue
        if str(getattr(actor, "operation_state", "") or "") == "pending_dispatch":
            continue  # released at the dispatch barrier: still running, recorded when it settles
        usage = dict(getattr(actor, "usage", {}) or {})
        route_kind = str(getattr(getattr(slot, "route", None), "value", "") or "api_chat")
        delegated_route = str(usage.get("delegated_route") or "")
        session = route_kind == "agent_session" or bool(delegated_route)
        effective: Dict[str, Any] = {
            # For a session the harness resolves route/model on its side; for
            # api_chat what was sent is what ran.
            "route": (f"agent_session:{delegated_route}" if delegated_route
                      else route_kind),
            # APPLIED honesty: a session whose telemetry disclosed no resolved
            # model shows ABSENCE — the requested model must never be dressed
            # up as the applied one. An api row's sent model IS its applied one.
            "model": (str(usage.get("resolved_model") or "") if session
                      else str(getattr(slot, "model", "") or "")),
            # No scalar "effort": keep host-send evidence and the engine's
            # sourced report separate from the requested row below.
            "verdict_method": str(usage.get("verdict_method") or ""),
        }
        if isinstance(usage.get("processing"), dict):
            effective["processing"] = dict(usage["processing"])
        if isinstance(usage.get("effort_resolution"), dict):
            effective["effort_resolution"] = dict(usage["effort_resolution"])
        # D29 applied account/access, verbatim from the engine receipt; absent
        # keys mean the telemetry predates the receipt — shown as absence.
        if usage.get("applied_profile"):
            effective["profile_id"] = str(usage["applied_profile"])
        if usage.get("applied_access"):
            effective["access"] = str(usage["applied_access"])
        row: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "surface": str(surface or ""),
            "requested": {
                "route_kind": route_kind,
                "model": str(getattr(slot, "model", "") or ""),
                # The ROW's effort. A caller-declared one-off (plan review's
                # reviewer_effort) is disclosed separately, never shown as the
                # row's saved configuration.
                "effort": "" if getattr(slot, "declared_effort", "") else str(getattr(slot, "effort", "") or ""),
                **({"declared_effort": str(slot.declared_effort)} if getattr(slot, "declared_effort", "") else {}),
                "session_target": str(getattr(slot, "session_target", "") or ""),
                "profile_id": str(getattr(slot, "session_profile", "") or ""),
                # Actor binding, when the row is a configured-subagent
                # reference ('' = direct row) — disclosure, never routing.
                "subagent_id": str(getattr(slot, "subagent_id", "") or ""),
                "processing_preference": str(getattr(slot, "processing_preference", "") or ""),
            },
            "effective": effective,
            **({"effort": dict(usage["effort"])} if isinstance(usage.get("effort"), dict) else {}),
            "capability_delta": usage.get("capability_delta") or [],
            "status": str(getattr(actor, "status", "") or ""),
            **({"review_record_id": str(record_id)} if record_id else {}),
        }
        # B1: typed failure facts, present only when the substrate carried them
        # (a later health surface reads them; absence stays honest absence).
        # ONE shared key list with the plan-row/wave projections (sources differ).
        for key in TYPED_FAILURE_FACT_KEYS:
            value = getattr(actor, key, None)
            if value:
                row[key] = value
        rows[str(actor.slot_id)] = row
    return rows


def record_reviewer_slot_executions(surface: str, actors: Any, slots_by_id: Dict[str, Any], *,
                                    record_id: str = "", keep_on: Any = None) -> Dict[str, Dict[str, Any]]:
    """Record each actor's last effective execution (best-effort, atomic) and return
    the rows this call wrote (:func:`reviewer_slot_execution_rows`).

    Written under the process data root (``config.DATA_DIR``), never a
    ToolContext review drive: UI state beside the saved settings, not per-task
    forensics — those live in the durable actor records already. An isolated
    contributor review's data root IS its review drive, so its markers stay there.
    ``record_id`` names the review ledger record the execution belongs to when the
    caller already holds it; a surface that learns the id only after its wave
    settled binds it afterwards with ``bind_reviewer_slot_record_id``. ``keep_on`` (the
    wave's ctx) receives the same rows as ``_last_review_slot_executions`` for the wave's
    ledger record, merged under the same lock: the triad and scope halves of one wave
    record concurrently, and a read-then-replace outside the lock would drop one half.
    """
    from ouroboros.utils import write_text_atomic

    rows = reviewer_slot_execution_rows(surface, actors, slots_by_id, record_id=record_id)
    path = _last_execution_path()
    with _LAST_EXECUTION_LOCK:
        if keep_on is not None and rows:
            kept = getattr(keep_on, "_last_review_slot_executions", None)
            setattr(keep_on, "_last_review_slot_executions", {**(kept if isinstance(kept, dict) else {}), **rows})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data.update(rows)
        if len(data) > _LAST_EXECUTION_CAP:
            ordered = sorted(data.items(), key=lambda kv: str(kv[1].get("ts") or ""))
            data = dict(ordered[-_LAST_EXECUTION_CAP:])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=1))
    return rows


def bind_reviewer_slot_record_id(executions: Any, record_id: str) -> None:
    """Name the review ledger record on the projection rows a settled wave itself wrote
    (the gate learns the id after its seats recorded themselves). ``executions`` are
    that wave's own rows (:func:`reviewer_slot_execution_rows`): a projection row is
    bound only when its timestamp is the wave's — a later run of the same seat by
    another surface keeps its own id. Best-effort, atomic."""
    from ouroboros.utils import write_text_atomic

    record_id = str(record_id or "")
    if not record_id or not isinstance(executions, dict):
        return
    path = _last_execution_path()
    with _LAST_EXECUTION_LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        changed = False
        for slot_id, own in executions.items():
            row = data.get(str(slot_id))
            own_ts = str(own.get("ts") or "") if isinstance(own, dict) else ""
            if isinstance(row, dict) and own_ts and str(row.get("ts") or "") == own_ts:
                row["review_record_id"] = record_id
                changed = True
        if changed:
            write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=1))


def reviewer_slot_last_executions() -> Dict[str, Any]:
    """Read the projection ('' shape on any read problem — disclosure only)."""
    try:
        data = json.loads(_last_execution_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


__all__ = [
    # The pool.
    "ConfiguredReviewerSlot",
    "PoolSeat",
    "ROUTE_KIND_API",
    "ROUTE_KIND_SESSION",
    "catalog_review_row",
    "child_acceptance_slots",
    "commit_triad_delivery",
    "composed_pool_seats",
    "composed_review_pool",
    "review_pool_rows",
    "review_pool_save_error",
    "review_pool_slots",
    "review_pool_state",
    "reviewer_slot_config_error",
    "row_at_effort_order",
    "row_effort",
    "row_plan_retrieves",
    "triad_delivery_slots",
    # «Выполняется как».
    "bind_reviewer_slot_record_id",
    "record_reviewer_slot_executions", "reviewer_slot_execution_rows",
    "reviewer_slot_last_executions",
]
