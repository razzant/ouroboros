"""Review lanes -> review pool: the one-time settings migration M (PR-3, contract §1.5).

``OUROBOROS_REVIEWER_SLOTS`` (the triad / scope / advisory / deep-review lanes) and
the three surface effort keys plus the deep-review model key are retired. What the
lanes effectively EXECUTED becomes rows of the subagent catalog
(``OUROBOROS_SUBAGENTS``) marked ``review_eligible`` — the review pool. The
migration runs at the settings read seam (``config.normalize_settings_raw``) on
every read until the owner's next save persists the migrated document; it is a
pure function of the document, so each read repeats it with the same result.

FROZEN READERS. The lane parsers below are copies of the readers
``reviewer_slot_config`` carried (``parse_reviewer_slots``, ``_parse_slot``,
``_parse_advisory``, ``_parse_deep_review``, ``_resolve_actor_slot``), re-scoped
from the process environment to ONE settings document: a reference resolves
through ``select_subagent_snapshot(document, ...)`` and processing through
``resolve_processing_preference(..., settings=document)`` — the document
version of ``roster_env_override(..., environ=loaded)``. They are frozen on
purpose: the live readers are deleted together with the lanes, and a migration
must keep reading yesterday's documents exactly as yesterday's code did. Their
test pins the frozen copies against the base code's effective executions.

Rules (contract §1.5, counter-examples §2 F4-F8):

1. a reference whose effective effort equals the referenced row's own effort
   marks that row (``review_eligible: true``; one mark per row, first by
   catalog order); every FURTHER triad reference to that row mints a twin
   from it (``minted_from: review_lane``) — the old wizard repeated a sole
   harness three times, and three seats stay three runs (quorum 2 of 3);
2. a reference whose seat ran a DIFFERENT effort mints ``review-<n>`` from the
   row (``minted_from: review_lane``), the helper itself untouched;
3. a direct seat mints ``review-<n>``; an existing ENABLED row with the
   identical engine AND delivery is marked instead (a row switched off never
   takes the mark — the pool reads enabled rows only — and stays as it was);
4. a scope seat merges into a row of the same engine and delivery produced by
   this run ("scope seat coincided with seat N — merged"); otherwise 1-3; the
   engine merge is the scope seat's alone — triad seats never fold;
5. advisory and deep-review rows become helper rows WITHOUT the mark; a
   reference needs no row at all (preflight is chosen per commit);
6. no lanes of the owner's (the key absent, or the ``""`` every 6.90+ document
   saved for "the default lanes ran") -> EXACTLY the factory rows
   (``factory_review_rows(document)``, ``minted_from: factory_default``, the ONE
   factory source: the frozen panel's seats ARE those rows, ``factory_lanes``), an
   existing catalog row of a row's engine marked instead of a twin. The
   never-configured install — neither the lanes key nor a catalog
   (``OUROBOROS_SUBAGENTS`` missing or ``""``: Docker / Colab / a mounted
   volume without the wizard, or no settings file at all) — is the same cell;
   never a structural catalog the owner saved empty: empty is not never-configured;
7. invalid lanes (a non-string value included), unresolvable references, an
   invalid catalog (marked rows do not excuse it), or authored lanes beside a
   catalog that already holds pool rows -> NO partial migration and nothing
   dropped in silence: the catalog is untouched, ``MigrationOutcome.error``
   carries the text, and every lane key stays in the document until the owner
   saves the catalog. Only the ``""`` lanes key beside a pool catalog is a no-op.

No seat loses its effort: every minted or marked row carries a non-empty
``effort`` (a seat's effort is its row's, else a compound slug's, else the
document's surface key, else the shipped ``high``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.model_slots import normalize_processing_preference, resolve_processing_preference
from ouroboros.review_dispatch import slot_id_for_row
from ouroboros.route_spec import (
    ROUTE_KIND_AGENT_SESSION as SHARED_ROUTE_KIND_SESSION,
    ROUTE_KIND_API_MODEL as SHARED_ROUTE_KIND_API,
    RouteSpec,
    compound_session_effort,
    parse_route_spec,
    route_spec_dict,
    validate_compound_session_effort,
)
from ouroboros.settings_defaults import (
    OPENROUTER_DEFAULTS,
    RETIRED_COMMA_LIST_SETTING_KEYS,
    REVIEW_POOL_MIGRATED_SETTING_KEYS,
)
from ouroboros.settings_scales import EFFORT_SCALE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen vocabulary (copies of the lane constants the live module is losing).
# ---------------------------------------------------------------------------

REVIEWER_SLOTS_KEY = "OUROBOROS_REVIEWER_SLOTS"
SUBAGENTS_KEY = "OUROBOROS_SUBAGENTS"
EFFORT_REVIEW_KEY = "OUROBOROS_EFFORT_REVIEW"
EFFORT_SCOPE_KEY = "OUROBOROS_EFFORT_SCOPE_REVIEW"
EFFORT_DEEP_KEY = "OUROBOROS_EFFORT_DEEP_SELF_REVIEW"
DEEP_MODEL_KEY = "OUROBOROS_MODEL_DEEP_SELF_REVIEW"
# The surface effort key each lane fell back to when neither the seat nor its
# row carried an effort (``settings_scales.resolve_effort``); ``high`` shipped.
_SURFACE_EFFORT_KEYS = {"triad": EFFORT_REVIEW_KEY, "scope": EFFORT_SCOPE_KEY, "deep_review": EFFORT_DEEP_KEY}
_SURFACE_EFFORT_DEFAULT = "high"

ROUTE_KIND_API = "api_chat"
ROUTE_KIND_SESSION = "agent_session"
TRIAD_SLOT_LIMIT = 10
SCOPE_SLOT_LIMIT = 4
_SLOT_ID_MAX_CHARS = 64
DELIVERY_NATIVE = "native"
DELIVERY_PACKET = "packet"
DELIVERY_SESSION = "session"
DEFAULT_TRIAD_DELIVERY = DELIVERY_NATIVE
DEEP_REVIEW_SLOT_ID = "deep_review_slot_1"
ADVISORY_SLOT_ID = "advisory_slot_1"
# The shipped default panel's deterministic per-row identities (review_dispatch).
SLOT_ID_PREFIX = "slot"
SCOPE_SLOT_ID_PREFIX = "scope_slot"

MINTED_FROM_LANE = "review_lane"
MINTED_FROM_FACTORY = "factory_default"
MINTED_ID_PREFIX = "review-"
LANE_ROW_RECOMMENDATION = "Minted from the former review lane"
# Frozen copy of subscription_install_presets._REVIEW_SEAT_RECOMMENDATION (pinned equal by test).
REVIEW_SEAT_RECOMMENDATION = (
    "Review-lane seat minted at onboarding; also selectable for delegated "
    "child work when its strengths fit."
)
SNAPSHOT_SCHEMA = 1
# What made the document a migration subject (``MigrationOutcome.trigger``, snapshot ``before.trigger``).
TRIGGER_LANES_KEY = "lanes_key"  # OUROBOROS_REVIEWER_SLOTS present, any value
TRIGGER_RETIRED_KEYS = "retired_comma_keys"  # only the pre-structured comma keys: the default panel ran
TRIGGER_NEVER_CONFIGURED = "never_configured"  # neither key, no catalog: the default panel ran
SNAPSHOT_DIR = "review_migrations"
SNAPSHOT_SUFFIX = "-slots-to-pool.json"

CATALOG_STATES = ("absent", "empty", "invalid", "disabled", "configured")
SLOTS_STATES = ("absent", "direct", "referenced", "mixed", "invalid")


# ---------------------------------------------------------------------------
# Frozen lane dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneRow:
    """One configured reviewer row as the lanes read it (frozen ``ConfiguredReviewerSlot``)."""

    slot_id: str
    kind: str  # api_chat | agent_session
    target_id: str
    effort: str = ""
    profile_id: str = ""
    subagent_id: str = ""
    processing_preference: str = ""
    delivery: str = ""
    # Session access the referenced catalog row carried ('' for direct rows).
    access: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind == ROUTE_KIND_SESSION


@dataclass(frozen=True)
class AdvisoryLane:
    """The one optional advisory reviewer (frozen ``AdvisorySlotConfig``)."""

    enabled: bool = True
    kind: str = ROUTE_KIND_API
    target_id: str = ""
    effort: str = "low"
    profile_id: str = ""
    subagent_id: str = ""
    disabled_reason: str = ""
    processing_preference: str = ""
    access: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind == ROUTE_KIND_SESSION


@dataclass(frozen=True)
class ReviewLanes:
    triad: Tuple[LaneRow, ...]
    scope: Tuple[LaneRow, ...]
    advisory: AdvisoryLane
    deep_review: Optional[LaneRow] = None


# ---------------------------------------------------------------------------
# Frozen readers — the document version of reviewer_slot_config's parsers.
# ---------------------------------------------------------------------------


def _valid_effort(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} effort must be a string")
    effort = value.strip().lower()
    if not effort:
        return ""
    if effort not in EFFORT_SCALE:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} names an unknown effort {effort!r}; "
            f"valid: {', '.join(EFFORT_SCALE)}"
        )
    return effort


def _validate_concrete_session_target(route: RouteSpec, where: str) -> None:
    """A structured session row names one concrete delegated route (``harness[=model]``)."""
    if not route.is_session or not route.target_id:
        return
    from ouroboros.subagents import parse_subagent_harness

    if parse_subagent_harness(route.target_id) is None:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} session target "
            f"{route.target_id!r} does not name a concrete harness route"
        )


def _row_processing(document: Mapping[str, Any], raw: Any = None, *, role: str = "") -> str:
    """Authored row / role / global precedence resolved on the DOCUMENT plane."""
    return resolve_processing_preference(
        role, override=normalize_processing_preference(raw) or None, settings=dict(document),
    )


def _resolve_actor_slot(
    document: Mapping[str, Any], slot_id: str, subagent_id: str, effort: str, where: str,
) -> LaneRow:
    """Materialize a configured-subagent reference into one frozen reviewer row.

    The roster is the DOCUMENT's ``OUROBOROS_SUBAGENTS`` (never the process
    environment). An unknown, disabled or invalid reference is a malformed
    reviewer configuration — the same typed ValueError the live parser raised,
    which the migration turns into its no-partial-migration error.
    """
    from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot

    try:
        snapshot, _legacy = select_subagent_snapshot(dict(document), subagent_id=subagent_id)
    except SubagentSelectionError as exc:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} subagent_id {subagent_id!r} does "
            f"not resolve: {exc.code}: {exc.detail}"
        ) from exc
    route = dict(snapshot.get("route") or {})
    target = str(route.get("target_id") or "")
    pin = str(route.get("credential_profile_id") or "")
    # Explicit seat effort wins; otherwise the roster row's own effort.
    chosen_effort = effort or _valid_effort(snapshot.get("effort"), where)
    if str(route.get("kind") or "") == SHARED_ROUTE_KIND_SESSION:
        shared = RouteSpec(kind=SHARED_ROUTE_KIND_SESSION, target_id=target, credential_profile_id=pin)
        _validate_concrete_session_target(shared, where)
        validate_compound_session_effort(shared, chosen_effort, setting=REVIEWER_SLOTS_KEY, where=where)
        return LaneRow(
            slot_id=slot_id, kind=ROUTE_KIND_SESSION, target_id=target, effort=chosen_effort,
            profile_id=pin, subagent_id=subagent_id,
            processing_preference=str(snapshot.get("processing_preference") or ""),
            access=str(snapshot.get("access") or ""),
        )
    return LaneRow(
        slot_id=slot_id, kind=ROUTE_KIND_API, target_id=target, effort=chosen_effort,
        profile_id=pin, subagent_id=subagent_id,
        processing_preference=str(snapshot.get("processing_preference") or ""),
    )


def _parse_delivery(row: Dict[str, Any], where: str, *, allowed: bool) -> str:
    if "delivery" not in row:
        return ""
    if not allowed:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} delivery applies only to a direct api_chat triad row "
            "(sessions, subagent references, scope, advisory and deep review rows always read)")
    value = row["delivery"]
    if value not in (DELIVERY_NATIVE, DELIVERY_PACKET):
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} delivery must be {DELIVERY_NATIVE!r} or {DELIVERY_PACKET!r}")
    return value


def _parse_slot(
    document: Mapping[str, Any], row: Any, where: str, seen_ids: set, *, delivery_allowed: bool = False,
) -> LaneRow:
    if not isinstance(row, dict):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} is not an object")
    unknown = sorted(set(row) - {"slot_id", "route", "subagent_id", "effort", "processing_preference", "delivery"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} has unknown keys: {unknown}")
    raw_slot_id = row.get("slot_id")
    if not isinstance(raw_slot_id, str):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} slot_id must be a string")
    slot_id = raw_slot_id.strip()
    if not slot_id or len(slot_id) > _SLOT_ID_MAX_CHARS:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} needs a stable non-empty slot_id "
            f"(≤{_SLOT_ID_MAX_CHARS} chars) — identity is never an array index"
        )
    if slot_id in seen_ids:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: slot_id {slot_id!r} appears twice; a row's "
            "receipts can only line up with ONE history"
        )
    seen_ids.add(slot_id)
    raw_ref = row.get("subagent_id")
    if raw_ref is not None and not isinstance(raw_ref, str):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} subagent_id must be a string")
    actor_ref = str(raw_ref or "").strip()
    if raw_ref is not None and not actor_ref:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} subagent_id must not be empty")
    if actor_ref and row.get("route") is not None:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: {where} must use either route or "
            "subagent_id, not both — the roster row is the route's SSOT"
        )
    if actor_ref:
        if "processing_preference" in row:
            raise ValueError(f"{REVIEWER_SLOTS_KEY}: {where} inherits Processing from its subagent")
        _parse_delivery(row, where, allowed=False)
        return _resolve_actor_slot(document, slot_id, actor_ref, _valid_effort(row.get("effort"), where), where)
    route = parse_route_spec(
        row.get("route"), setting=REVIEWER_SLOTS_KEY, where=where,
        kind_aliases={ROUTE_KIND_API: SHARED_ROUTE_KIND_API, ROUTE_KIND_SESSION: SHARED_ROUTE_KIND_SESSION},
        pin_key="profile_id", reject_unknown=True, strict_strings=True, reject_api_pin=True,
    )
    kind = ROUTE_KIND_SESSION if route.is_session else ROUTE_KIND_API
    _validate_concrete_session_target(route, where)
    effort = _valid_effort(row.get("effort"), where)
    validate_compound_session_effort(route, effort, setting=REVIEWER_SLOTS_KEY, where=where)
    return LaneRow(
        slot_id=slot_id, kind=kind, target_id=route.target_id, effort=effort,
        profile_id=route.credential_profile_id,
        processing_preference=_row_processing(document, row.get("processing_preference")),
        delivery=_parse_delivery(row, where, allowed=delivery_allowed and kind == ROUTE_KIND_API),
    )


def _migrate_sdk_advisory_target(raw_kind: str, target: str) -> tuple[str, str]:
    """Translate a retired Claude-SDK ``api``-kind target to ``(routed, reason)``."""
    if raw_kind != "api":
        return target, ""
    base = target.replace("[1m]", "").strip()
    if not base or base in {"sonnet", "claude-sonnet-5"}:
        return "", ""
    if "/" in base or "::" in base:
        return target, ""
    if base.startswith("claude-"):
        return f"anthropic/{base}", ""
    return target, "legacy_claude_sdk_target_unmapped"


def _resolve_advisory_actor(document: Mapping[str, Any], subagent_id: str, effort: str, enabled: bool) -> AdvisoryLane:
    row = _resolve_actor_slot(document, ADVISORY_SLOT_ID, subagent_id, effort, "advisory")
    return AdvisoryLane(
        enabled=enabled, kind=row.kind, target_id=row.target_id,
        effort=row.effort or ("low" if not row.is_session else ""),
        profile_id=row.profile_id, subagent_id=subagent_id,
        processing_preference=row.processing_preference, access=row.access,
    )


def _parse_advisory(document: Mapping[str, Any], raw: Any) -> AdvisoryLane:
    if raw is None:
        return AdvisoryLane(processing_preference=_row_processing(document))
    if not isinstance(raw, dict):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory must be an object")
    unknown = sorted(set(raw) - {"enabled", "route", "kind", "target_id", "effort", "subagent_id", "processing_preference"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory has unknown keys: {unknown}")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory enabled must be a boolean")
    for key in ("kind", "target_id", "subagent_id"):
        if key in raw and not isinstance(raw[key], str):
            raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory {key} must be a string")
    actor_ref = str(raw.get("subagent_id") or "").strip()
    if "subagent_id" in raw and not actor_ref:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory subagent_id must not be empty")
    route = raw.get("route")
    if route is not None and not isinstance(route, dict):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory route must be an object {{kind, target_id}}")
    if actor_ref and (route is not None or ({"kind", "target_id"} & set(raw))):
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: advisory must use either subagent_id or a "
            "route, not both — the roster row is the route's SSOT"
        )
    if actor_ref:
        if "processing_preference" in raw:
            raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory inherits Processing from its subagent")
        return _resolve_advisory_actor(document, actor_ref, _valid_effort(raw.get("effort"), "advisory"), enabled)
    if route is not None and ({"kind", "target_id"} & set(raw)):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: advisory must use either route or legacy kind/target_id, not both")
    route_payload = dict(route or {})
    if "kind" not in route_payload:
        route_payload["kind"] = raw.get("kind") or ROUTE_KIND_API
    if "target_id" not in route_payload:
        route_payload["target_id"] = raw.get("target_id") or ""
    raw_kind = str(route_payload.get("kind") or "").strip().lower()
    shared_route = parse_route_spec(
        route_payload, setting=REVIEWER_SLOTS_KEY, where="advisory",
        kind_aliases={"api": SHARED_ROUTE_KIND_API, ROUTE_KIND_API: SHARED_ROUTE_KIND_API,
                      ROUTE_KIND_SESSION: SHARED_ROUTE_KIND_SESSION},
        pin_key="profile_id", allow_empty_target=True, reject_unknown=True, strict_strings=True,
        reject_api_pin=True,
    )
    if enabled and shared_route.is_session and not shared_route.target_id:
        raise ValueError(
            f"{REVIEWER_SLOTS_KEY}: enabled advisory agent_session route needs "
            "a non-empty target_id; shared-session fallback is legacy-only"
        )
    _validate_concrete_session_target(shared_route, "advisory")
    effort = _valid_effort(raw.get("effort"), "advisory")
    if not effort and not shared_route.is_session:
        effort = "low"
    validate_compound_session_effort(shared_route, effort, setting=REVIEWER_SLOTS_KEY, where="advisory")
    target, disabled_reason = (
        _migrate_sdk_advisory_target(raw_kind, shared_route.target_id)
        if not shared_route.is_session else (shared_route.target_id, "")
    )
    return AdvisoryLane(
        enabled=enabled and not disabled_reason,
        kind=ROUTE_KIND_SESSION if shared_route.is_session else ROUTE_KIND_API,
        target_id=target, effort=effort, profile_id=shared_route.credential_profile_id,
        disabled_reason=disabled_reason,
        processing_preference=_row_processing(document, raw.get("processing_preference")),
    )


def _parse_deep_review(document: Mapping[str, Any], raw: Any, seen_ids: set) -> Optional[LaneRow]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: deep_review must be an object")
    unknown = sorted(set(raw) - {"route", "subagent_id", "effort", "processing_preference"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: deep_review has unknown keys: {unknown}")
    return _parse_slot(document, {**raw, "slot_id": DEEP_REVIEW_SLOT_ID}, "deep_review", seen_ids)


def parse_reviewer_slots(document: Mapping[str, Any], raw: Any) -> ReviewLanes:
    """Strict parse of the structured lanes setting against ONE document. Raises ValueError."""
    if not isinstance(raw, str):
        raise ValueError(f"{REVIEWER_SLOTS_KEY} must be a JSON string")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{REVIEWER_SLOTS_KEY} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{REVIEWER_SLOTS_KEY} must be a JSON object")
    unknown = sorted(set(payload) - {"triad", "scope", "advisory", "deep_review"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_KEY} has unknown top-level keys: {unknown}")
    seen_ids: set = set()
    groups: Dict[str, List[LaneRow]] = {}
    for group, limit in (("triad", TRIAD_SLOT_LIMIT), ("scope", SCOPE_SLOT_LIMIT)):
        rows = payload.get(group)
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise ValueError(f"{REVIEWER_SLOTS_KEY}: {group} must be an array")
        if len(rows) > limit:
            raise ValueError(
                f"{REVIEWER_SLOTS_KEY}: {group} has {len(rows)} rows; the real "
                f"limit is {limit} (shown in the UI, not negotiable here)"
            )
        groups[group] = [
            _parse_slot(document, row, f"{group}[{idx}]", seen_ids, delivery_allowed=group == "triad")
            for idx, row in enumerate(rows)
        ]
    if not groups["triad"]:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: triad needs at least one slot")
    if not groups["scope"]:
        raise ValueError(f"{REVIEWER_SLOTS_KEY}: scope needs at least one slot")
    return ReviewLanes(
        triad=tuple(groups["triad"]), scope=tuple(groups["scope"]),
        advisory=_parse_advisory(document, payload.get("advisory")),
        deep_review=_parse_deep_review(document, payload.get("deep_review"), seen_ids),
    )


# ---------------------------------------------------------------------------
# Frozen shipped-default panel, derived from the document (what _default_config ran).
# ---------------------------------------------------------------------------


def _text(document: Mapping[str, Any], key: str) -> str:
    return str(document.get(key) or "").strip()


def _frozen_deep_default(document: Mapping[str, Any]) -> str:
    """The deep self-review model the legacy key stood for when the shipped panel ran
    (``get_deep_self_review_model(settings, authored_panel=False)``)."""
    from ouroboros.provider_models import compatible_only_main_model

    chosen = _text(document, DEEP_MODEL_KEY)
    if not chosen and not _text(document, REVIEWER_SLOTS_KEY):
        chosen = compatible_only_main_model(document)
    return chosen or str(OPENROUTER_DEFAULTS["deep_self_review"])


def factory_lanes(document: Mapping[str, Any]) -> ReviewLanes:
    """The shipped default panel as frozen lanes, from the ONE factory source: the triad
    is ``factory_review_rows(document)`` seat by seat (the provider the document holds
    credentials for; a one-model install repeats Main), no scope seat and no authored
    advisory (the pool asks every retrieving row the coupling question), and NO deep
    row (the legacy model key stood for it). A seat's effort is the document's surface
    effort, as the panel ran it — the row's own ``effort`` is that same value minted.
    A second provider table here once read the document differently from the rows and
    minted OpenRouter seats beside a direct provider's or a local Main's rows (D1-V04)."""
    view = {key: value for key, value in document.items() if key not in RETIRED_COMMA_LIST_SETTING_KEYS}
    processing = _row_processing(view)
    templates = factory_review_rows(view) if factory_review_rows is not None else []
    return ReviewLanes(
        triad=tuple(
            LaneRow(slot_id=slot_id_for_row(idx + 1, prefix=SLOT_ID_PREFIX), kind=ROUTE_KIND_API,
                    target_id=_row_route(row).target_id, processing_preference=processing,
                    delivery=DEFAULT_TRIAD_DELIVERY)
            for idx, row in enumerate(templates)
        ),
        scope=(),
        advisory=AdvisoryLane(target_id="", processing_preference=processing),
    )


# Package A's seam: the rows a fresh install's onboarding mints for the pool
# (``subscription_install_presets.factory_review_rows(document)``). M mints
# exactly those rows for a document without lanes and only covers what they
# leave uncovered (the frozen factory lanes above say which seats the shipped
# panel had). The import is deferred: the preset compiler reaches the provider
# readers, which import this module's callers. Tests monkeypatch the attribute.
def _package_a_factory_review_rows(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    from ouroboros.subscription_install_presets import factory_review_rows as mint

    return mint(document)


factory_review_rows: Optional[Callable[[Mapping[str, Any]], List[Dict[str, Any]]]] = _package_a_factory_review_rows


# ---------------------------------------------------------------------------
# Effective executions (the tuple the contract compares).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seat:
    """One effective reviewer execution: what a lane row actually ran as."""

    lane: str  # triad | scope | advisory | deep_review
    slot_id: str
    subagent_id: str
    kind: str  # api_chat | agent_session
    target_id: str
    effort: str
    effort_source: str  # row | compound | document | default | ''
    delivery: str  # native | packet | session
    credential_profile_id: str = ""
    processing_preference: str = ""
    access: str = ""
    enabled: bool = True
    authored: bool = True

    def engine(self) -> tuple:
        """Execution identity the catalog compares rows by (plus delivery)."""
        return (SHARED_ROUTE_KIND_SESSION if self.kind == ROUTE_KIND_SESSION else SHARED_ROUTE_KIND_API,
                self.target_id, self.credential_profile_id, self.effort, self.processing_preference,
                self.access if self.kind == ROUTE_KIND_SESSION else "", self.delivery)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "slot_id": self.slot_id, "subagent_id": self.subagent_id, "kind": self.kind,
            "target_id": self.target_id, "effort": self.effort,
        }
        if self.effort_source:
            payload["effort_source"] = self.effort_source
        payload.update({"delivery": self.delivery, "credential_profile_id": self.credential_profile_id,
                        "processing_preference": self.processing_preference})
        if self.kind == ROUTE_KIND_SESSION:
            payload["access"] = self.access or "full"
        if not self.authored:
            payload["authored"] = False
        if self.lane == "advisory":
            payload["enabled"] = self.enabled
        return payload


def _document_effort(document: Mapping[str, Any], lane: str) -> tuple[str, str]:
    """``(effort, source)`` a seat without its own effort ran at: the document's surface key
    when valid on the scale, else the shipped default."""
    key = _SURFACE_EFFORT_KEYS[lane]
    value = _text(document, key).lower()
    if value in EFFORT_SCALE:
        return value, "document"
    return _SURFACE_EFFORT_DEFAULT, "default"


def _route_of(row: LaneRow) -> RouteSpec:
    return RouteSpec(
        kind=SHARED_ROUTE_KIND_SESSION if row.is_session else SHARED_ROUTE_KIND_API,
        target_id=row.target_id, credential_profile_id=row.profile_id,
    )


def _seat(document: Mapping[str, Any], lane: str, row: LaneRow, *, authored: bool = True) -> Seat:
    if row.effort:
        effort, source = row.effort, "row"
    elif (compound := compound_session_effort(_route_of(row))):
        effort, source = compound, "compound"
    else:
        effort, source = _document_effort(document, lane)
    if row.is_session:
        delivery = DELIVERY_SESSION
    elif lane == "triad" and not row.subagent_id:
        delivery = DELIVERY_NATIVE if row.delivery == DELIVERY_NATIVE else DELIVERY_PACKET
    else:
        delivery = DELIVERY_NATIVE  # references, scope, advisory and deep rows always read
    return Seat(
        lane=lane, slot_id=row.slot_id, subagent_id=row.subagent_id, kind=row.kind, target_id=row.target_id,
        effort=effort, effort_source=source, delivery=delivery, credential_profile_id=row.profile_id,
        processing_preference=row.processing_preference,
        access=(row.access or "full") if row.is_session else "", authored=authored,
    )


def _advisory_seat(advisory: AdvisoryLane, *, authored: bool) -> Seat:
    route = RouteSpec(kind=SHARED_ROUTE_KIND_SESSION if advisory.is_session else SHARED_ROUTE_KIND_API,
                      target_id=advisory.target_id, credential_profile_id=advisory.profile_id)
    effort, source = (advisory.effort, "row") if advisory.effort else (compound_session_effort(route), "compound")
    return Seat(
        lane="advisory", slot_id=ADVISORY_SLOT_ID, subagent_id=advisory.subagent_id, kind=advisory.kind,
        target_id=advisory.target_id, effort=effort or "", effort_source=source if effort else "",
        delivery=DELIVERY_SESSION if advisory.is_session else DELIVERY_NATIVE,
        credential_profile_id=advisory.profile_id, processing_preference=advisory.processing_preference,
        access=(advisory.access or "full") if advisory.is_session else "",
        enabled=advisory.enabled, authored=authored,
    )


def effective_executions(document: Mapping[str, Any], lanes: ReviewLanes, *, authored: bool) -> Dict[str, Any]:
    """Every seat the lanes ran, as the frozen reader resolves them against ``document``.

    ``deep_review`` is the structural row when present, else the row the legacy
    model key synthesized (``authored`` only while the key is non-empty).
    """
    if lanes.deep_review is not None:
        deep = _seat(document, "deep_review", lanes.deep_review, authored=authored)
    else:
        key_value = _text(document, DEEP_MODEL_KEY)
        synthesized = LaneRow(slot_id=DEEP_REVIEW_SLOT_ID, kind=ROUTE_KIND_API,
                              target_id=_frozen_deep_default(document),
                              processing_preference=_row_processing(document, role="deep_review"))
        deep = _seat(document, "deep_review", synthesized, authored=bool(key_value))
    advisory = lanes.advisory
    advisory_authored = authored and bool(advisory.subagent_id or advisory.target_id)
    return {
        "triad": [_seat(document, "triad", row, authored=authored) for row in lanes.triad],
        "scope": [_seat(document, "scope", row, authored=authored) for row in lanes.scope],
        "advisory": _advisory_seat(advisory, authored=advisory_authored),
        "deep_review": deep,
    }


# ---------------------------------------------------------------------------
# Catalog view of the document.
# ---------------------------------------------------------------------------


@dataclass
class _Catalog:
    state: str
    enabled: bool = True
    items: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    # The stored key was absent but a legacy singleton (OUROBOROS_SUBAGENT_HARNESS)
    # materialized rows; they are written out so delegation keeps them.
    seeded_from_legacy: bool = False


def _catalog_view(document: Mapping[str, Any]) -> _Catalog:
    from ouroboros.configured_subagents import (
        SOURCE_INVALID,
        SOURCE_LEGACY_MIGRATED,
        configured_subagents_dict,
        resolve_configured_subagents,
    )

    raw = document.get(SUBAGENTS_KEY)
    resolution = resolve_configured_subagents(dict(document))
    if raw in (None, ""):
        if resolution.source == SOURCE_LEGACY_MIGRATED and resolution.config is not None:
            payload = configured_subagents_dict(resolution.config)
            return _Catalog("absent", bool(payload["enabled"]), list(payload["items"]), seeded_from_legacy=True)
        return _Catalog("absent")
    if not isinstance(raw, str):
        return _Catalog("invalid", error=f"{SUBAGENTS_KEY} must be a JSON string")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return _Catalog("invalid", error=f"{SUBAGENTS_KEY} is not valid JSON: {exc}")
    # Validity first, the pool marker second (VD3-05): this tree's parser knows the pool
    # fields, so a catalog it calls invalid IS invalid, marked rows or not — a marker
    # beside a broken row must not pass the document off as "already a pool".
    if resolution.source == SOURCE_INVALID:
        return _Catalog("invalid", error=resolution.diagnostic or f"{SUBAGENTS_KEY} is invalid")
    items = [dict(row) for row in (payload.get("items") or []) if isinstance(row, Mapping)] if isinstance(payload, Mapping) else []
    enabled = bool(payload.get("enabled")) if isinstance(payload, Mapping) else False
    if not items:
        return _Catalog("empty", enabled, items)
    return _Catalog("disabled" if not enabled else "configured", enabled, items)


def _row_route(row: Mapping[str, Any]) -> RouteSpec:
    route = dict(row.get("route") or {})
    kind = str(route.get("kind") or "").strip().lower()
    return RouteSpec(
        kind=SHARED_ROUTE_KIND_SESSION if kind == SHARED_ROUTE_KIND_SESSION else SHARED_ROUTE_KIND_API,
        target_id=str(route.get("target_id") or "").strip(),
        credential_profile_id=str(route.get("credential_profile_id") or "").strip(),
    )


def _row_engine(document: Mapping[str, Any], row: Mapping[str, Any]) -> tuple:
    """A catalog row's execution identity plus delivery, compared against ``Seat.engine()``."""
    route = _row_route(row)
    effort = str(row.get("effort") or "").strip().lower() or compound_session_effort(route)
    processing = resolve_processing_preference(
        override=normalize_processing_preference(row.get("processing_preference")) or None, settings=dict(document))
    if route.is_session:
        delivery, access = DELIVERY_SESSION, str(row.get("access") or "full")
    else:
        delivery, access = (str(row.get("delivery") or DELIVERY_NATIVE)), ""
    return (route.kind, route.target_id, route.credential_profile_id, effort, processing, access, delivery)


def _row_label(row: Mapping[str, Any]) -> str:
    route = _row_route(row)
    bits = [route.target_id if route.is_session else f"api {route.target_id}"]
    if effort := str(row.get("effort") or "").strip():
        bits.append(effort)
    if processing := str(row.get("processing_preference") or "").strip():
        bits.append(processing)
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# The migration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationOutcome:
    """What one migration of one document decided (pure; applied by the read seam)."""

    input_sha256: str
    catalog_state: str
    slots_state: str
    snapshot: Dict[str, Any]
    catalog_after: Optional[str] = None  # the serialized OUROBOROS_SUBAGENTS, None on error
    consumed_keys: Tuple[str, ...] = ()  # lane keys the read seam drops from the document
    retained_keys: Tuple[str, ...] = ()  # lane keys that stay (error: the owner must save the catalog)
    error: str = ""
    noop: bool = False  # the catalog was already a pool: lane keys are dropped, nothing rewritten
    trigger: str = TRIGGER_LANES_KEY  # one of the TRIGGER_* values


def migration_trigger(document: Mapping[str, Any]) -> str:
    """Why a document is a migration subject — one of the ``TRIGGER_*`` values, or
    ``""`` for a pool document (a structural catalog, no lanes key), which is done.

    The lanes key present (any value, including the ``""`` every 6.90+ document
    wrote) is the first trigger. A document with neither the lanes key nor a
    catalog (``OUROBOROS_SUBAGENTS`` missing or ``""``) ran the shipped default
    panel: a pre-structured-era document still carrying the retired comma keys,
    or a never-configured install (contract §1.5, the both-absent cell — the
    common backend read/init seam without the wizard). Both take the factory
    rows. A catalog the owner saved EMPTY is structural (``items: []``), not
    ``""``: empty is not never-configured, and it is left exactly as saved.
    """
    if REVIEWER_SLOTS_KEY in document:
        return TRIGGER_LANES_KEY
    if str(document.get(SUBAGENTS_KEY) or "").strip():
        return ""
    if any(key in document for key in RETIRED_COMMA_LIST_SETTING_KEYS):
        return TRIGGER_RETIRED_KEYS
    return TRIGGER_NEVER_CONFIGURED


def environment_overridable_keys(document: Any) -> frozenset:
    """The keys whose read-seam value for ``document`` stands in for an ABSENT key, so a
    reader that merges the environment over the document lets an environment value win.

    One key: the factory reviewer rows ``apply_at_read_seam`` mints into
    ``OUROBOROS_SUBAGENTS`` when the document authored NO review lanes (the key absent, or
    the ``""`` every 6.90+ document saved for "the default lanes ran" — retired comma keys
    never entered the panel, canon 11) and holds no catalog text are a default, not the
    owner's disk value — a catalog the environment carries (a container's roster, a
    benchmark's; explicit configuration) wins over them exactly as it won over the absence
    before the mint existed. Rows minted from the owner's own lanes, and a catalog the owner
    saved (marked or not), are the document's decision and shadow the environment like every
    disk-authored key. A missing or unreadable document has no keys at all.
    """
    if not isinstance(document, Mapping):
        return frozenset()
    lanes = document.get(REVIEWER_SLOTS_KEY)
    if (isinstance(lanes, str) and lanes.strip()) or str(document.get(SUBAGENTS_KEY) or "").strip():
        return frozenset()
    return frozenset({SUBAGENTS_KEY})


def migration_applies(document: Mapping[str, Any]) -> bool:
    """Whether the read seam has anything to do with this document (:func:`migration_trigger`)."""
    return bool(migration_trigger(document))


_SHA_PRESENCE_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
    "ZAI_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_USER",
    "GIGACHAT_PASSWORD", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL",
)
_SHA_VALUE_KEYS = REVIEW_POOL_MIGRATED_SETTING_KEYS + (
    SUBAGENTS_KEY, "OUROBOROS_SUBAGENT_HARNESS", "OUROBOROS_SUBAGENT_PROFILE", "OUROBOROS_PROCESSING_PREFERENCE",
    "OUROBOROS_MODEL_PROCESSING_PREFERENCES", "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "USE_LOCAL_MAIN",
    # The legacy singleton's row materialization (configured_subagents._append_legacy_model_rows:
    # the Heavy row from OUROBOROS_MODEL_HEAVY / USE_LOCAL_HEAVY, the scout from Light's local
    # flag) shapes the catalog view, so these decide the outcome too (VD3-04).
    "OUROBOROS_MODEL_HEAVY", "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT",
)


def input_sha256(document: Mapping[str, Any]) -> str:
    """Digest of every document fact the migration reads (credentials by presence only):
    the per-process cache (``_MIGRATIONS_SEEN``) and the receipts are keyed by it, so a
    fact left out would replay another document's outcome."""
    facts = {key: document.get(key) for key in _SHA_VALUE_KEYS if key in document}
    facts["_present"] = sorted(key for key in _SHA_PRESENCE_KEYS if _text(document, key))
    return hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _slots_state(lanes: Optional[ReviewLanes], raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "absent"
    if lanes is None:
        return "invalid"
    rows = list(lanes.triad) + list(lanes.scope)
    if lanes.deep_review is not None:
        rows.append(lanes.deep_review)
    refs = [bool(row.subagent_id) for row in rows]
    if lanes.advisory.subagent_id or lanes.advisory.target_id:
        refs.append(bool(lanes.advisory.subagent_id))
    if all(refs):
        return "referenced"
    if not any(refs):
        return "direct"
    return "mixed"


class _Pool:
    """The rows a migration produces over an existing catalog (mark / mint / merge bookkeeping)."""

    def __init__(self, document: Mapping[str, Any], catalog: _Catalog) -> None:
        self.document = document
        self.items = [dict(row) for row in catalog.items]
        self.by_id = {str(row.get("subagent_id") or ""): row for row in self.items}
        self.engines = {str(row.get("subagent_id") or ""): _row_engine(document, row) for row in self.items}
        self.marked: Dict[str, List[str]] = {}  # existing row id -> seats
        self.minted: List[Dict[str, Any]] = []
        self.minted_seats: Dict[str, List[str]] = {}
        self.notes: Dict[str, List[str]] = {}
        self.lines: List[str] = []  # owner-report lines, one per touched row
        self.used_ids = set(self.by_id)

    def note(self, row_id: str, text: str) -> None:
        self.notes.setdefault(row_id, []).append(text)

    def next_id(self) -> str:
        n = 1
        while f"{MINTED_ID_PREFIX}{n}" in self.used_ids:
            n += 1
        row_id = f"{MINTED_ID_PREFIX}{n}"
        self.used_ids.add(row_id)
        return row_id

    def mark(self, row_id: str, seat: Seat) -> None:
        self.marked.setdefault(row_id, []).append(seat.slot_id)

    def produced_match(self, seat: Seat) -> Optional[str]:
        """A row this run already marked or minted with the seat's engine and delivery."""
        engine = seat.engine()
        for row_id in self.marked:
            if self.engines[row_id] == engine:
                return row_id
        for row in self.minted:
            if row.get("review_eligible") and _row_engine(self.document, row) == engine:
                return str(row["subagent_id"])
        return None

    def unmarked_existing_match(self, seat: Seat) -> Optional[str]:
        return self.unmarked_existing_engine(seat.engine())

    def unmarked_existing_engine(self, engine: tuple, *, excluding: Any = ()) -> Optional[str]:
        """An ENABLED existing catalog row this run has not marked yet that runs ``engine``.

        A row switched off never takes the mark: the pool reads enabled rows only, so a
        mark on it would silently empty the pool a live seat ran in — the seat mints its
        own enabled row beside the disabled helper instead."""
        for row in self.items:
            row_id = str(row.get("subagent_id") or "")
            if row.get("enabled", True) is False:
                continue
            if row_id not in self.marked and row_id not in excluding and self.engines[row_id] == engine:
                return row_id
        return None

    def vacant_template(self, seat: Seat) -> Optional[str]:
        """An adopted factory row of the seat's engine that no seat has landed on yet."""
        engine = seat.engine()
        for row in self.minted:
            row_id = str(row.get("subagent_id") or "")
            if (row.get("review_eligible") and not self.minted_seats.get(row_id)
                    and _row_engine(self.document, row) == engine):
                return row_id
        return None

    def mint(self, seat: Seat, *, eligible: bool, minted_from: str, source_row: Optional[Mapping[str, Any]] = None,
             recommended_use: str = "") -> str:
        """A new catalog row for ``seat``; a source row lends its authored processing/access."""
        route = RouteSpec(
            kind=SHARED_ROUTE_KIND_SESSION if seat.kind == ROUTE_KIND_SESSION else SHARED_ROUTE_KIND_API,
            target_id=seat.target_id, credential_profile_id=seat.credential_profile_id)
        if source_row is not None:
            processing = str(source_row.get("processing_preference") or "")
        else:
            inherited = resolve_processing_preference("", settings=dict(self.document))
            processing = seat.processing_preference if seat.processing_preference != inherited else ""
        row_id = self.next_id()
        payload: Dict[str, Any] = {
            "subagent_id": row_id,
            "recommended_use": recommended_use or (
                REVIEW_SEAT_RECOMMENDATION if minted_from == MINTED_FROM_FACTORY else LANE_ROW_RECOMMENDATION),
            "route": route_spec_dict(route, api_kind=SHARED_ROUTE_KIND_API, pin_key="credential_profile_id"),
        }
        if seat.effort:
            payload["effort"] = seat.effort
        if processing:
            payload["processing_preference"] = processing
        if route.is_session:
            payload["access"] = (str(source_row.get("access")) if source_row and source_row.get("access")
                                 else seat.access or "full")
        if eligible:
            payload["review_eligible"] = True
        if not route.is_session and seat.delivery == DELIVERY_PACKET:
            payload["delivery"] = DELIVERY_PACKET
        payload["minted_from"] = minted_from
        self.minted.append(payload)
        self.minted_seats[row_id] = [seat.slot_id]
        return row_id

    def adopt(self, template: Mapping[str, Any]) -> str:
        """A pre-minted factory row (package A's seam), re-identified when its id collides."""
        payload = dict(template)
        row_id = str(payload.get("subagent_id") or "").strip()
        if not row_id or row_id in self.used_ids:
            row_id = self.next_id()
        else:
            self.used_ids.add(row_id)
        payload["subagent_id"] = row_id
        payload.setdefault("recommended_use", REVIEW_SEAT_RECOMMENDATION)
        payload.setdefault("review_eligible", True)
        payload.setdefault("minted_from", MINTED_FROM_FACTORY)
        if not str(payload.get("effort") or "").strip():
            # No row leaves the migration without an effort (F7): a template that
            # defers to the surface default gets the document's triad effort.
            payload["effort"] = _document_effort(self.document, "triad")[0]
        self.minted.append(payload)
        self.minted_seats[row_id] = []
        return row_id

    def attach(self, row_id: str, seat: Seat) -> None:
        """Record a seat landing on a row this run produced (a merge, not a new row)."""
        if row_id in self.minted_seats:
            self.minted_seats[row_id].append(seat.slot_id)
        else:
            self.marked.setdefault(row_id, []).append(seat.slot_id)

    def rows_after(self) -> List[Dict[str, Any]]:
        out = []
        for row in self.items:
            row_id = str(row.get("subagent_id") or "")
            if row_id in self.marked and not row.get("review_eligible"):
                row = {**row, "review_eligible": True}
            out.append(row)
        return out + list(self.minted)


def _place_triad(pool: _Pool, seat: Seat, minted_from: str) -> None:
    if seat.subagent_id:
        row = pool.by_id.get(seat.subagent_id)
        if row is None:  # the frozen reader resolved it, so the row exists; defensive
            pool.mint(seat, eligible=True, minted_from=minted_from)
            return
        if pool.engines[seat.subagent_id] == seat.engine():
            if seat.subagent_id not in pool.marked:
                pool.mark(seat.subagent_id, seat)
                return
            # A further reference to a marked row is a further independent run (the old
            # wizard repeated a sole harness three times): a twin keeps the seat count and
            # the quorum with it. Folding by engine belongs to the scope seat alone.
            row_id = pool.mint(seat, eligible=True, minted_from=minted_from, source_row=row)
            pool.note(row_id, f"seat {seat.slot_id} also referenced {seat.subagent_id} — a twin row keeps its run")
            return
        own = str(row.get("effort") or "").strip() or compound_session_effort(_row_route(row)) or "its default"
        row_id = pool.mint(seat, eligible=True, minted_from=minted_from, source_row=row)
        pool.note(row_id, f"{seat.subagent_id} keeps effort {own} for delegation; the review seat "
                          f"{seat.slot_id} ran {seat.effort} — minted as its own row")
        return
    match = pool.unmarked_existing_match(seat)
    if match is not None:
        pool.mark(match, seat)
        pool.note(match, f"direct seat {seat.slot_id} ran this row's engine — marked instead of a new row")
        return
    pool.mint(seat, eligible=True, minted_from=minted_from)


def _place_scope(pool: _Pool, seat: Seat, minted_from: str) -> None:
    merged = pool.produced_match(seat)
    if merged is not None:
        pool.attach(merged, seat)
        seats = pool.minted_seats.get(merged) or pool.marked.get(merged) or []
        first = next((s for s in seats if s != seat.slot_id), merged)
        pool.note(merged, f"scope seat {seat.slot_id} coincided with seat {first} — merged")
        return
    if seat.subagent_id and pool.engines.get(seat.subagent_id) == seat.engine() and seat.subagent_id not in pool.marked:
        pool.mark(seat.subagent_id, seat)
        return
    if seat.subagent_id:
        row = pool.by_id.get(seat.subagent_id)
        own = str((row or {}).get("effort") or "").strip() or "its default"
        row_id = pool.mint(seat, eligible=True, minted_from=minted_from, source_row=row)
        pool.note(row_id, f"{seat.subagent_id} keeps effort {own} for delegation; the scope seat "
                          f"{seat.slot_id} ran {seat.effort} — minted as its own row")
        return
    match = pool.unmarked_existing_match(seat)
    if match is not None:
        pool.mark(match, seat)
        pool.note(match, f"scope seat {seat.slot_id} ran this row's engine — marked instead of a new row")
        return
    pool.mint(seat, eligible=True, minted_from=minted_from)


def _place_helpers(pool: _Pool, executions: Dict[str, Any], lanes: ReviewLanes, factory: bool) -> List[str]:
    """Advisory and deep review: helper rows without the mark (rule 5). Returns report lines."""
    lines: List[str] = []
    advisory: Seat = executions["advisory"]
    if advisory.subagent_id:
        row_id = advisory.subagent_id
        if row_id in pool.marked or row_id in pool.minted_seats:
            pool.attach(row_id, advisory)
        pool.note(row_id, "advisory reference needed no row: preflight is chosen per commit")
        lines.append(f"the advisory reference to {row_id} needed no row")
    elif not advisory.enabled:
        lines.append("the advisory row was disabled; nothing minted (preflight is chosen per commit)")
    elif advisory.authored and advisory.target_id and not factory:
        twin = pool.produced_match(advisory)
        row_id = pool.mint(advisory, eligible=False, minted_from=MINTED_FROM_LANE)
        if twin is not None:
            seats = pool.minted_seats.get(twin) or pool.marked.get(twin) or [twin]
            pool.note(row_id, f"advisory engine coincided with triad seat {seats[0]} — kept as a separate helper row "
                              f"{row_id} (no mark); preflight is now chosen per commit")
        else:
            pool.note(row_id, "from Advisory; preflight is now chosen per commit")
    else:
        lines.append("the shipped default advisory reviewer was not authored; nothing minted")
    deep: Seat = executions["deep_review"]
    if deep.subagent_id:
        pool.note(deep.subagent_id, "deep review reference needed no row: /review takes a reviewer per call")
        lines.append(f"the deep review reference to {deep.subagent_id} needed no row")
    elif deep.authored:
        source = "Deep review" if lanes.deep_review is not None else DEEP_MODEL_KEY
        row_id = pool.mint(deep, eligible=False, minted_from=MINTED_FROM_LANE)
        pool.note(row_id, f"from {source}; /review now takes a reviewer per call, default Main")
    return lines


def _not_in_effect(document: Mapping[str, Any], executions: Dict[str, Any], lanes: ReviewLanes) -> List[str]:
    lines = []
    for lane, key in _SURFACE_EFFORT_KEYS.items():
        value = _text(document, key)
        if not value or (lane == "scope" and not lanes.scope):
            # The shipped panel's scope reader ran at the document's scope effort; the
            # factory rows carry no scope seat, so the key is retired with the lane —
            # it was in effect, not idle.
            continue
        seats = executions[lane] if lane != "deep_review" else [executions["deep_review"]]
        if not any(seat.effort_source == "document" for seat in seats):
            lines.append(f"{key}={value}")
    deep_value = _text(document, DEEP_MODEL_KEY)
    if deep_value and lanes.deep_review is not None:
        lines.append(f"{DEEP_MODEL_KEY}={deep_value}")
    return lines


def _error_outcome(document: Mapping[str, Any], catalog: _Catalog, slots_state: str, text: str,
                   executions: Optional[Dict[str, Any]] = None) -> MigrationOutcome:
    present = tuple(key for key in REVIEW_POOL_MIGRATED_SETTING_KEYS if key in document)
    snapshot = _snapshot_base(document, catalog.state, slots_state)
    if executions is not None:
        snapshot["effective_before"] = _executions_dict(executions)
    snapshot.update({"after": None, "rows": [], "not_in_effect": [], "summary": None, "error": text})
    return MigrationOutcome(
        input_sha256=input_sha256(document), catalog_state=catalog.state, slots_state=slots_state,
        snapshot=snapshot, retained_keys=present, error=text, trigger=snapshot["before"]["trigger"],
    )


def _snapshot_base(document: Mapping[str, Any], catalog_state: str, slots_state: str) -> Dict[str, Any]:
    before: Dict[str, Any] = {}
    for key in (REVIEWER_SLOTS_KEY, SUBAGENTS_KEY):
        value = document.get(key)
        before[key] = value if isinstance(value, str) else (None if value is None else json.dumps(value))
    for key in (EFFORT_REVIEW_KEY, EFFORT_SCOPE_KEY, EFFORT_DEEP_KEY, DEEP_MODEL_KEY):
        if key in document:
            before[key] = str(document.get(key) if document.get(key) is not None else "")
    before.update({"catalog_state": catalog_state, "slots_state": slots_state, "trigger": migration_trigger(document)})
    return {"schema": SNAPSHOT_SCHEMA, "ts": None, "input_sha256": input_sha256(document), "before": before}


def _executions_dict(executions: Dict[str, Any]) -> Dict[str, Any]:
    advisory: Seat = executions["advisory"]
    if advisory.subagent_id:
        advisory_payload: Dict[str, Any] = {"slot_id": ADVISORY_SLOT_ID, "subagent_id": advisory.subagent_id,
                                            "enabled": advisory.enabled}
    else:
        advisory_payload = advisory.as_dict()
    return {
        "triad": [seat.as_dict() for seat in executions["triad"]],
        "scope": [seat.as_dict() for seat in executions["scope"]],
        "advisory": advisory_payload,
        "deep_review": executions["deep_review"].as_dict(),
    }


def migrate_review_lanes(loaded: Mapping[str, Any]) -> Optional[MigrationOutcome]:
    """THE migration: ``None`` when the document carries nothing to migrate, else the
    outcome the read seam applies (pure — ``loaded`` is never mutated here)."""
    if not migration_applies(loaded):
        return None
    document = dict(loaded)
    catalog = _catalog_view(document)
    raw_lanes = document.get(REVIEWER_SLOTS_KEY)
    if raw_lanes is not None and not isinstance(raw_lanes, str):
        # The lane key held JSON text; any other shape is not "no lanes" but garbage the
        # frozen reader never accepted — it stays in the document, named (VD3-11).
        return _error_outcome(document, catalog, "invalid",
                              f"{REVIEWER_SLOTS_KEY} must be a JSON string, not {type(raw_lanes).__name__}")
    authored = isinstance(raw_lanes, str) and bool(raw_lanes.strip())

    def authored_slots_state() -> str:
        if not authored:
            return "absent"
        try:
            return _slots_state(parse_reviewer_slots(document, raw_lanes), raw_lanes)
        except ValueError:
            return "invalid"  # references cannot resolve against this catalog

    if catalog.state == "invalid":
        return _error_outcome(document, catalog, authored_slots_state(),
                              f"the subagent catalog is invalid, so the review lanes cannot be migrated: {catalog.error}")
    if any(isinstance(row, Mapping) and "review_eligible" in row for row in catalog.items):
        if authored:
            # Two review configurations in one document (a hand edit, a downgrade's save
            # beside a pool catalog): the catalog's pool is what runs, and the lanes are
            # not dropped in silence — the key stays, the snapshot keeps both, the owner
            # decides in Settings → Agents (saving the catalog retires the lanes).
            return _error_outcome(document, catalog, authored_slots_state(),
                                  "the document carries review lanes AND a subagent catalog that already holds "
                                  "review pool rows; the catalog's pool is in force and the lanes were not applied")
        present = tuple(key for key in REVIEW_POOL_MIGRATED_SETTING_KEYS if key in document)
        snapshot = _snapshot_base(document, catalog.state, "absent")
        snapshot.update({"after": None, "rows": [], "not_in_effect": [], "summary": None,
                         "error": "", "noop": "the catalog already carries review pool rows"})
        return MigrationOutcome(input_sha256=input_sha256(document), catalog_state=catalog.state,
                                slots_state="absent", snapshot=snapshot,
                                consumed_keys=present, noop=True, trigger=snapshot["before"]["trigger"])
    if authored:
        try:
            lanes = parse_reviewer_slots(document, raw_lanes)
        except ValueError as exc:
            return _error_outcome(document, catalog, "invalid", str(exc))
        slots_state = _slots_state(lanes, raw_lanes)
        minted_from = MINTED_FROM_LANE
    else:
        lanes = factory_lanes(document)
        slots_state = "absent"
        minted_from = MINTED_FROM_FACTORY
    executions = effective_executions(document, lanes, authored=authored)
    pool = _Pool(document, catalog)
    if not authored and factory_review_rows is not None:
        # A document without lanes of its own gets EXACTLY the factory rows (canon 07):
        # the frozen factory seats ARE those rows, one source. A template whose engine a
        # catalog row already runs is not adopted — its seat marks that row (F6, merge);
        # the remaining templates keep the first free ``review-<n>`` ids.
        claimed: set = set()
        for template in factory_review_rows(document):
            existing = pool.unmarked_existing_engine(_row_engine(document, template), excluding=claimed)
            if existing is not None:
                claimed.add(existing)
                continue
            pool.adopt({**template, "subagent_id": ""})
        for seat in executions["triad"]:
            # One frozen seat lands on one row: an existing row of its engine first
            # (F6), else a template of its engine that no seat has landed on yet
            # (twins stay twins, F5), else the ordinary placement.
            vacant = None if pool.unmarked_existing_match(seat) is not None else pool.vacant_template(seat)
            if vacant is not None:
                pool.attach(vacant, seat)
            else:
                _place_triad(pool, seat, minted_from)
    else:
        for seat in executions["triad"]:
            _place_triad(pool, seat, minted_from)
    for seat in executions["scope"]:
        _place_scope(pool, seat, minted_from)
    helper_lines = _place_helpers(pool, executions, lanes, factory=not authored)
    rows_after = pool.rows_after()
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS

    if len(rows_after) > MAX_CONFIGURED_SUBAGENTS:
        return _error_outcome(
            document, catalog, slots_state,
            f"migrating the review lanes would give the subagent catalog {len(rows_after)} rows; "
            f"the maximum is {MAX_CONFIGURED_SUBAGENTS}", executions)
    payload = {"enabled": catalog.enabled if catalog.state != "absent" or catalog.seeded_from_legacy else True,
               "items": rows_after}
    catalog_after = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    rows_report = _rows_report(pool)
    eligible = [row for row in rows_after if row.get("review_eligible")]
    summary = {
        "seats_before": len(executions["triad"]) + len(executions["scope"]),
        "rows_marked_after": len(eligible),
        "distinct_models": len({(_row_route(row).kind, _row_route(row).target_id) for row in eligible}),
        "helper_rows_minted": sum(1 for row in pool.minted if not row.get("review_eligible")),
    }
    report_lines = list(helper_lines)
    if catalog.state == "disabled":
        report_lines.append("the subagent catalog switch stays off: review stays on; delegation stays off")
    if catalog.seeded_from_legacy:
        report_lines.append("the legacy OUROBOROS_SUBAGENT_HARNESS rows were written into the catalog unchanged")
    snapshot = _snapshot_base(document, catalog.state, slots_state)
    snapshot.update({
        "effective_before": _executions_dict(executions),
        "after": {SUBAGENTS_KEY: payload},
        "rows": rows_report,
        "notes": report_lines,
        "not_in_effect": _not_in_effect(document, executions, lanes),
        "summary": summary,
        "error": "",
    })
    return MigrationOutcome(
        input_sha256=snapshot["input_sha256"], catalog_state=catalog.state, slots_state=slots_state,
        snapshot=snapshot, catalog_after=catalog_after,
        consumed_keys=tuple(key for key in REVIEW_POOL_MIGRATED_SETTING_KEYS if key in document),
        trigger=snapshot["before"]["trigger"],
    )


def _rows_report(pool: _Pool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_id, seats in pool.marked.items():
        entry: Dict[str, Any] = {"subagent_id": row_id, "action": "marked", "from_seats": list(seats), "minted_from": ""}
        if pool.notes.get(row_id):
            entry["note"] = "; ".join(pool.notes[row_id])
        rows.append(entry)
    for row in pool.minted:
        row_id = str(row["subagent_id"])
        entry = {"subagent_id": row_id, "action": "minted", "from_seats": list(pool.minted_seats.get(row_id, [])),
                 "minted_from": str(row.get("minted_from") or ""), "review_eligible": bool(row.get("review_eligible"))}
        if pool.notes.get(row_id):
            entry["note"] = "; ".join(pool.notes[row_id])
        rows.append(entry)
    for row_id, notes in pool.notes.items():
        if row_id not in pool.marked and row_id not in pool.minted_seats:
            rows.append({"subagent_id": row_id, "action": "merged_into", "from_seats": [], "minted_from": "",
                         "note": "; ".join(notes)})
    return rows


def apply_outcome(loaded: Dict[str, Any], outcome: MigrationOutcome) -> None:
    """Rewrite ``loaded`` as the outcome decided: the migrated catalog in, consumed lane
    keys out; an error keeps every lane key (the owner's catalog save finishes it)."""
    if outcome.error:
        return
    if outcome.catalog_after is not None:
        loaded[SUBAGENTS_KEY] = outcome.catalog_after
    for key in outcome.consumed_keys:
        loaded.pop(key, None)


# Migrations this process's settings reads have computed, keyed by the digest of the
# document facts the migration reads (insertion order kept). The read seam runs on every
# settings read, so one document is migrated once and the recorded outcome is re-applied;
# receipts (``review_pool_receipts``) go only to the outcomes that decide a document.
_MIGRATIONS_SEEN: Dict[str, MigrationOutcome] = {}


def migrations_seen() -> Tuple[MigrationOutcome, ...]:
    """The outcomes this process has computed, oldest first."""
    return tuple(_MIGRATIONS_SEEN.values())


def apply_at_read_seam(loaded: Dict[str, Any]) -> Tuple[str, ...]:
    """Run (or replay) the migration on ``loaded`` inside ``config.normalize_settings_raw``;
    returns the lane keys the retired-key purge must leave in place (an error outcome
    keeps them for the owner's save)."""
    if not migration_applies(loaded):
        return ()
    digest = input_sha256(loaded)
    outcome = _MIGRATIONS_SEEN.get(digest)
    if outcome is None:
        outcome = migrate_review_lanes(loaded)
        if outcome is None:
            return ()
        _MIGRATIONS_SEEN[digest] = outcome
        summary = outcome.snapshot.get("summary") or {}
        if outcome.error:
            log.warning("settings: review lanes not migrated: %s", outcome.error)
        elif outcome.trigger == TRIGGER_NEVER_CONFIGURED:
            log.info("settings: no review settings configured; the factory review rows run (%s reviewer rows)",
                     summary.get("rows_marked_after"))
        elif not outcome.noop:
            log.info("settings: review lanes migrated into the review pool (%s seats -> %s reviewer rows)",
                     summary.get("seats_before"), summary.get("rows_marked_after"))
    apply_outcome(loaded, outcome)
    return tuple(outcome.retained_keys)


# ---------------------------------------------------------------------------
# The owner's one message.
# ---------------------------------------------------------------------------


def _seat_label(row: Mapping[str, Any]) -> str:
    return _row_label(row)


ROLLBACK_SENTENCE = (
    f"To roll back, restore {SUBAGENTS_KEY} and {REVIEWER_SLOTS_KEY} (and the effort keys it lists) from the "
    "snapshot's `before` into settings.json (null = remove the key) and restart; the next read migrates again "
    "unless you downgrade."
)


def owner_message(outcome: MigrationOutcome, snapshot_path: str, *,
                  environment_retired_keys: Sequence[str] = ()) -> str:
    """The ONE English owner-chat message for a migration (contract §1.5 template).
    ``environment_retired_keys``: the retired review keys the boot found set in the process
    environment (``server_maintenance.environment_retired_review_keys``) — a never-configured
    document is then not told it had no review settings; those keys are named as not read."""
    if outcome.error:
        return (
            "⚙️ Review settings could not be migrated automatically: "
            f"{outcome.error}. The review lanes setting stays in your document and the subagent "
            "catalog is unchanged; mark reviewers in Settings → Agents to finish. "
            + (f"Snapshot: {snapshot_path}." if snapshot_path else "No snapshot could be written.")
        )
    if outcome.noop:
        return ""
    where = (f"Snapshot: {snapshot_path}. {ROLLBACK_SENTENCE}" if snapshot_path
             else "No snapshot could be written, so there is no rollback source.")
    snap = outcome.snapshot
    summary = snap.get("summary") or {}
    before = snap.get("effective_before") or {}
    after_rows = {str(row.get("subagent_id")): row for row in ((snap.get("after") or {}).get(SUBAGENTS_KEY) or {}).get("items", [])}
    if outcome.trigger == TRIGGER_NEVER_CONFIGURED:
        # Nothing of the owner's was migrated: the install had no review settings at
        # all, so the message says what RUNS, not what changed.
        rows = [f"{rid} ({_seat_label(row)})" for rid, row in after_rows.items() if row.get("review_eligible")]
        if environment_retired_keys:
            head = ("⚙️ Review pool initialized. This install's settings document had no review settings (no review "
                    f"lanes, no subagent catalog); the {', '.join(environment_retired_keys)} set in the process "
                    "environment " + ("are" if len(environment_retired_keys) != 1 else "is") + " no longer read, "
                    "so the factory reviewer rows run as its review pool: the rows of the subagent catalog marked "
                    "“Reviewer”.")
        else:
            head = ("⚙️ Review pool initialized. This install had no review settings (no review lanes, no subagent "
                    "catalog), so the factory reviewer rows run as its review pool: the rows of the subagent "
                    "catalog marked “Reviewer”.")
        return "\n".join([
            head,
            f"{summary.get('rows_marked_after', 0)} reviewer rows, {summary.get('distinct_models', 0)} distinct models: "
            + "; ".join(rows) + ".",
            f"{where} Adjust in Settings → Agents.",
        ])
    if outcome.slots_state == "absent":
        head = ("⚙️ Review settings migrated. This install ran the shipped default review lanes (Triad / Scope / "
                "Advisory / Deep review); they became one review pool: the rows of the subagent catalog marked "
                "“Reviewer”.")
    else:
        head = ("⚙️ Review settings migrated. The review lanes (Triad / Scope / Advisory / Deep review) became one "
                "review pool: the rows of the subagent catalog marked “Reviewer”.")
    extras = []
    advisory = before.get("advisory") or {}
    if advisory.get("subagent_id"):
        extras.append("advisory reference")
    elif advisory.get("authored", True) and advisory.get("target_id") and advisory.get("enabled", True):
        extras.append("advisory row")
    deep = before.get("deep_review") or {}
    if deep.get("authored", True):
        extras.append("deep review row")
    triad_n, scope_n = len(before.get("triad") or []), len(before.get("scope") or [])
    if outcome.slots_state == "absent":
        seats = f"the shipped default panel ({triad_n} seat{'s' if triad_n != 1 else ''})"
    else:
        seats = f"{triad_n} triad seat{'s' if triad_n != 1 else ''} + {scope_n} scope seat{'s' if scope_n != 1 else ''}"
    counts = (f"Before: {seats}" + (f" (+ {', '.join(extras)})" if extras else "") +
              f". After: {summary.get('rows_marked_after', 0)} reviewer rows, "
              f"{summary.get('distinct_models', 0)} distinct models.")
    bullets = []
    for entry in snap.get("rows") or []:
        row_id = str(entry.get("subagent_id"))
        row = after_rows.get(row_id)
        label = f"{row_id} ({_seat_label(row)})" if row else row_id
        if entry.get("action") == "marked":
            text = f"• {label} — marked"
        elif entry.get("action") == "minted":
            text = f"• {label} — new row" + ("" if entry.get("review_eligible") else " without the mark")
        else:
            text = f"• {label}"
        if entry.get("note"):
            text += f": {entry['note']}"
        bullets.append(text)
    lines = [head, counts, *bullets]
    if notes := snap.get("notes"):
        lines.append("Preflight is now chosen per commit (commit_reviewed preflight_reviewer=…); " + "; ".join(notes) + ".")
    else:
        lines.append("Preflight is now chosen per commit (commit_reviewed preflight_reviewer=…).")
    if snap.get("not_in_effect"):
        lines.append("Not in effect before, retired: " + ", ".join(snap["not_in_effect"]) + ".")
    lines.append(f"{where} Adjust in Settings → Agents.")
    return "\n".join(lines)


__all__ = [
    "AdvisoryLane",
    "LaneRow",
    "MigrationOutcome",
    "ReviewLanes",
    "Seat",
    "apply_at_read_seam",
    "apply_outcome",
    "effective_executions",
    "environment_overridable_keys",
    "factory_lanes",
    "factory_review_rows",
    "input_sha256",
    "migrate_review_lanes",
    "migration_applies",
    "migration_trigger",
    "migrations_seen",
    "owner_message",
    "parse_reviewer_slots",
]
