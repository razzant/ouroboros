"""Pure sibling compilers for Available subagents and reviewer presets.

The caller supplies already-normalized settings and, only when subscriptions
were declared, one verified live discovery snapshot. Task actors consume every
supported harness plus truthful Main/Light API or local routes. Reviewer slots
consume only the independently ratified Claude/Codex/Cursor subset, so Agy-only
is a successful task preset with no structured reviewer override and a mixed
Agy install produces byte-identical reviewer JSON to its core-only projection.

No settings or transport are read here. Exact discovery misses are typed
refusals with no partial output. Rows are unpinned by default so Claudexor owns
compatible account rotation; Cursor and Agy keep effort in both their compound
model slug and the explicit row field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.configured_subagents import (
    ALTERNATIVE_RECOMMENDATION,
    MAX_CONFIGURED_SUBAGENTS,
    INDEPENDENT_RECOMMENDATION,
    PRIMARY_RECOMMENDATION,
    SCOUT_RECOMMENDATION,
    SOURCE_ONBOARDING_DEFAULT,
    SUBAGENTS_RECEIPT_KEY,
    SUBAGENTS_SETTING,
    ConfiguredSubagent,
    ConfiguredSubagents,
    configured_subagents_dict,
    configured_subagents_fingerprint,
    make_configured_subagents,
    serialize_configured_subagents,
)
from ouroboros.provider_models import (
    model_has_credentials_in_settings,
)
from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, RouteSpec

# The marker written beside the applied preset. Its ABSENCE is not authority to
# apply anything (every pre-preset install lacks it too); the server-side
# fresh-install latch is. Bumping this string marks a NEW preset generation.
SUBSCRIPTION_PRESET_VERSION = "3"
PRESET_MARKER_KEY = "OUROBOROS_SUBSCRIPTION_PRESET_VERSION"

REVIEWER_SLOTS_KEY = "OUROBOROS_REVIEWER_SLOTS"

HARNESS_CLAUDE = "claude"
HARNESS_CODEX = "codex"
HARNESS_CURSOR = "cursor"
HARNESS_AGY = "agy"
# The task-actor harnesses the compiler recognizes. A connected harness outside
# this tuple (opencode, raw-api, …) contributes no automatic row.
PRESET_HARNESSES: Tuple[str, ...] = (
    HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_CURSOR, HARNESS_AGY,
)

SURFACE_SUBAGENT = "subagent"
SURFACE_ADVISORY = "advisory"
SURFACE_TRIAD = "triad"
SURFACE_SCOPE = "scope"

# Harnesses whose discovery ids ENCODE the reasoning effort in the id itself.
_EFFORT_IN_MODEL_ID = frozenset({HARNESS_CURSOR, HARNESS_AGY})

# Owner shorthand -> ORDERED candidate discovery ids. ``{effort}`` is filled
# from the seat for a harness that spells effort inside the id. The FIRST
# candidate present in live discovery wins; when none is present the seat
# refuses. This is an alias table for ONE model family, never a fallback to a
# different model or to the harness default.
_MODEL_ALIASES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    HARNESS_CLAUDE: {
        "opus-5": ("claude-opus-5",),
        "sonnet-5": ("claude-sonnet-5",),
    },
    HARNESS_CODEX: {
        "gpt-5.6-sol": ("gpt-5.6-sol",),
        "gpt-5.6-terra": ("gpt-5.6-terra",),
    },
    HARNESS_CURSOR: {
        "grok-4.6": ("cursor-grok-4.6-{effort}", "grok-4.6-{effort}"),
    },
    # agy (Antigravity) spells effort inside the id like cursor. Flash High is
    # the automatic task actor; Pro remains an ordinary manual editor choice.
    HARNESS_AGY: {
        "gemini-3.8-flash": ("gemini-3.8-flash-{effort}",),
        "gemini-3.1-pro": ("gemini-3.1-pro-{effort}",),
    },
}


@dataclass(frozen=True)
class HarnessDiscovery:
    """One connected harness and the model ids the daemon discovered for it."""

    harness_id: str
    model_ids: Tuple[str, ...] = ()

    @property
    def has_models(self) -> bool:
        return bool(self.model_ids)


@dataclass(frozen=True)
class PresetSeat:
    """One compiled policy seat: harness, model preference, and effort."""

    surface: str
    position: int  # 1-based within its surface
    harness: str
    preference: str
    effort: str


@dataclass(frozen=True)
class PresetRefusal:
    """Why the preset could not be compiled — named down to the seat."""

    code: str
    seat: Optional[PresetSeat]
    candidates: Tuple[str, ...]
    message: str

    def as_dict(self) -> Dict[str, Any]:
        seat = self.seat
        return {
            "code": self.code,
            "message": self.message,
            "candidates": list(self.candidates),
            "surface": seat.surface if seat else "",
            "position": seat.position if seat else 0,
            "harness": seat.harness if seat else "",
            "preference": seat.preference if seat else "",
            "effort": seat.effort if seat else "",
        }


@dataclass(frozen=True)
class SubscriptionInstallPreset:
    """The compiled preset, or a typed refusal. Never both, never partial."""

    connected: Tuple[str, ...] = ()
    reviewer_slots: str = ""
    available_subagents: str = ""
    source: str = SOURCE_ONBOARDING_DEFAULT
    diagnostics: Tuple[Dict[str, Any], ...] = ()
    receipt: Dict[str, Any] = field(default_factory=dict)
    refusal: Optional[PresetRefusal] = None

    @property
    def ok(self) -> bool:
        return self.refusal is None and bool(self.available_subagents)

    def settings_keys(
        self, *, include_reviewer: bool = True, include_marker: bool = True,
    ) -> Dict[str, str]:
        """The EXACT settings keys an install-time save adds. Empty on refusal —
        a half-applied preset is worse than none."""
        if not self.ok:
            return {}
        values = {
            SUBAGENTS_SETTING: self.available_subagents,
            SUBAGENTS_RECEIPT_KEY: json.dumps(self.receipt, ensure_ascii=False, separators=(",", ":")),
        }
        if include_marker:
            values[PRESET_MARKER_KEY] = SUBSCRIPTION_PRESET_VERSION
        if include_reviewer and self.reviewer_slots:
            values[REVIEWER_SLOTS_KEY] = self.reviewer_slots
        return values


def _seat(surface: str, position: int, harness: str, preference: str, effort: str) -> PresetSeat:
    return PresetSeat(surface=surface, position=position, harness=harness,
                      preference=preference, effort=effort)


@dataclass(frozen=True)
class SurfacePolicy:
    preference: str
    effort: str


@dataclass(frozen=True)
class ReviewerHarnessPolicy:
    """Review-only model choices; task actors have an independent catalog."""

    advisory: SurfacePolicy
    triad: SurfacePolicy
    scope: SurfacePolicy


def _surface(preference: str, effort: str) -> SurfacePolicy:
    return SurfacePolicy(preference=preference, effort=effort)


_REVIEWER_POLICIES: Dict[str, ReviewerHarnessPolicy] = {
    HARNESS_CLAUDE: ReviewerHarnessPolicy(
        advisory=_surface("sonnet-5", "low"),
        triad=_surface("opus-5", "medium"),
        scope=_surface("opus-5", "medium"),
    ),
    HARNESS_CODEX: ReviewerHarnessPolicy(
        advisory=_surface("gpt-5.6-terra", "medium"),
        triad=_surface("gpt-5.6-sol", "medium"),
        scope=_surface("gpt-5.6-sol", "medium"),
    ),
    HARNESS_CURSOR: ReviewerHarnessPolicy(
        advisory=_surface("grok-4.6", "medium"),
        triad=_surface("grok-4.6", "medium"),
        scope=_surface("grok-4.6", "high"),
    ),
}

_TASK_POLICIES: Dict[str, SurfacePolicy] = {
    HARNESS_CLAUDE: _surface("opus-5", "medium"),
    HARNESS_CODEX: _surface("gpt-5.6-sol", "medium"),
    HARNESS_CURSOR: _surface("grok-4.6", "high"),
    HARNESS_AGY: _surface("gemini-3.8-flash", "high"),
}

_POLICY_HARNESSES = (HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_CURSOR)
REVIEWER_PRESET_HARNESSES = _POLICY_HARNESSES
_ADVISORY_ORDER = _POLICY_HARNESSES
_TRIAD_ORDER = (HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_CURSOR)
_SCOPE_ORDER = (HARNESS_CODEX, HARNESS_CLAUDE, HARNESS_CURSOR)


def _first_connected(order: Sequence[str], connected: set[str]) -> str:
    return next(harness for harness in order if harness in connected)


def _policy_seat(surface: str, position: int, harness: str) -> PresetSeat:
    spec = getattr(_REVIEWER_POLICIES[harness], surface)
    return _seat(surface, position, harness, spec.preference, spec.effort)


def _compile_policy_seats(connected: Sequence[str]) -> Dict[str, Tuple[PresetSeat, ...]]:
    """Apply linear priority rules; no powerset row grows when a harness is added."""
    present = set(connected)
    advisory = _first_connected(_ADVISORY_ORDER, present)
    scope = _first_connected(_SCOPE_ORDER, present)

    triad_harnesses = [harness for harness in _TRIAD_ORDER if harness in present]
    if len(triad_harnesses) == 1:
        triad_harnesses *= 3

    return {
        SURFACE_ADVISORY: (_policy_seat(SURFACE_ADVISORY, 1, advisory),),
        SURFACE_TRIAD: tuple(
            _policy_seat(SURFACE_TRIAD, position, harness)
            for position, harness in enumerate(triad_harnesses, start=1)
        ),
        SURFACE_SCOPE: (_policy_seat(SURFACE_SCOPE, 1, scope),),
    }


def _compile_task_seats(connected: Sequence[str]) -> Tuple[PresetSeat, ...]:
    return tuple(
        _seat(SURFACE_SUBAGENT, position, harness,
              _TASK_POLICIES[harness].preference, _TASK_POLICIES[harness].effort)
        for position, harness in enumerate(connected, start=1)
    )


def _candidate_ids(seat: PresetSeat) -> Tuple[str, ...]:
    aliases = _MODEL_ALIASES.get(seat.harness, {}).get(seat.preference, ())
    return tuple(spelling.format(effort=seat.effort) for spelling in aliases)


def _resolve_seat(seat: PresetSeat,
                  discovery: Mapping[str, HarnessDiscovery]) -> Tuple[str, Optional[PresetRefusal]]:
    """The EXACT discovery id for a seat, or a typed refusal naming it."""
    candidates = _candidate_ids(seat)
    found = discovery.get(seat.harness)
    available = set(found.model_ids) if found else set()
    for candidate in candidates:
        if candidate in available:
            return candidate, None
    if not candidates:
        return "", PresetRefusal(
            code="unknown_model_preference", seat=seat, candidates=(),
            message=(f"No alias is registered for {seat.preference!r} on the "
                     f"{seat.harness} harness — the {seat.surface} seat cannot be resolved."),
        )
    return "", PresetRefusal(
        code="model_not_in_discovery", seat=seat, candidates=candidates,
        message=(f"The {seat.harness} harness does not expose "
                 f"{seat.preference!r} (tried {', '.join(candidates)}), so the "
                 f"{seat.surface} seat #{seat.position} cannot be filled from live "
                 "discovery."),
    )


def _session_target(harness: str, model_id: str) -> str:
    """The reviewer-row route identity: Claudexor's own ``harness=model``
    spelling (no ``::`` — that is API-route syntax)."""
    return f"{harness}={model_id}"


def _row_effort(seat: PresetSeat) -> str:
    """A row's ``effort`` field.

    Present on EVERY harness on purpose. On cursor the compound slug is what the
    vendor honours, but the review/delegation surfaces materialize an effort
    regardless — leaving the field empty would send the surface's global default
    alongside a slug that says something else."""
    return seat.effort


def _resolved_row(seat: PresetSeat, model_id: str) -> Dict[str, Any]:
    return {
        "surface": seat.surface,
        "position": seat.position,
        "harness": seat.harness,
        "preference": seat.preference,
        "model": model_id,
        "effort": seat.effort,
        "effort_in_model_id": seat.harness in _EFFORT_IN_MODEL_ID,
        "target_id": _session_target(seat.harness, model_id),
    }


def _slot_id(surface: str, position: int) -> str:
    """Row identity from the ONE mint (``review_substrate.slot_id_for_row``) so a
    preset row's receipts line up with a hand-authored row's."""
    from ouroboros.review_substrate import (
        SCOPE_SLOT_ID_PREFIX,
        SLOT_ID_PREFIX,
        slot_id_for_row,
    )

    prefix = SCOPE_SLOT_ID_PREFIX if surface == SURFACE_SCOPE else SLOT_ID_PREFIX
    return slot_id_for_row(position, prefix=prefix)


def _resolve_surface(seats: Sequence[PresetSeat],
                     discovery: Mapping[str, HarnessDiscovery],
                     ) -> Tuple[List[Dict[str, Any]], Optional[PresetRefusal]]:
    rows: List[Dict[str, Any]] = []
    for seat in seats:
        model_id, refusal = _resolve_seat(seat, discovery)
        if refusal is not None:
            return [], refusal
        rows.append(_resolved_row(seat, model_id))
    return rows, None


def _inline_reviewer_slots_json(triad: Sequence[Mapping[str, Any]],
                                scope: Sequence[Mapping[str, Any]],
                                advisory: Mapping[str, Any]) -> str:
    """Inline reviewer routes for the owner-configured path.

    An owner draft is validate-only: the preset never extends or edits the
    owner's roster, so its reviewer seats cannot mint reference rows — they
    stay self-contained inline routes (4=A references belong to the roster
    the preset itself ships)."""
    def _rows(resolved: Sequence[Mapping[str, Any]], surface: str) -> List[Dict[str, Any]]:
        return [
            {
                "slot_id": _slot_id(surface, int(row["position"])),
                "route": {"kind": "agent_session", "target_id": str(row["target_id"])},
                "effort": str(row["effort"]),
            }
            for row in resolved
        ]

    payload = {
        "triad": _rows(triad, SURFACE_TRIAD),
        "scope": _rows(scope, SURFACE_SCOPE),
        "advisory": {
            "enabled": True,
            "route": {"kind": "agent_session", "target_id": str(advisory["target_id"])},
            "effort": str(advisory["effort"]),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=False)


_REVIEW_SEAT_RECOMMENDATION = (
    "Review-lane seat minted at onboarding; also selectable for delegated "
    "child work when its strengths fit."
)


def _reference_reviewer_rows(
    available: ConfiguredSubagents,
    triad: Sequence[Mapping[str, Any]],
    scope: Sequence[Mapping[str, Any]],
    advisory: Mapping[str, Any],
) -> tuple[ConfiguredSubagents, str, List[Dict[str, Any]]]:
    """Compile reviewer slots as ``subagent_id`` REFERENCES into the roster.

    One SSOT from the first boot (owner decision 4=A): a seat whose session
    route already exists as a roster row references it; otherwise a roster
    row is minted for it (no ``name`` — the field is retired) and referenced.
    A per-slot ``effort`` override rides the slot only when it differs from
    the referenced row's own effort. If the 10-row roster cap leaves no room,
    the seat falls back to an inline route and says so in diagnostics —
    honest fallback, never a silent drop.
    """
    items = list(available.items)
    diagnostics: List[Dict[str, Any]] = []

    def _row_for(target_id: str, effort: str) -> tuple[str, str] | None:
        identity = (ROUTE_KIND_AGENT_SESSION, target_id, "")
        for row in items:
            if (row.route.kind, row.route.target_id,
                    row.route.credential_profile_id) == identity:
                return row.subagent_id, row.effort
        if len(items) >= MAX_CONFIGURED_SUBAGENTS:
            return None
        used = {row.subagent_id for row in items}
        harness = target_id.partition("=")[0] or "session"
        base = f"review-{harness}"
        row_id, suffix = base, 2
        while row_id in used:
            row_id, suffix = f"{base}-{suffix}", suffix + 1
        items.append(ConfiguredSubagent(
            subagent_id=row_id,
            recommended_use=_REVIEW_SEAT_RECOMMENDATION,
            route=RouteSpec(ROUTE_KIND_AGENT_SESSION, target_id),
            effort=effort,
        ))
        return row_id, effort

    def _slot(surface: str, row: Mapping[str, Any]) -> Dict[str, Any]:
        target, effort = str(row["target_id"]), str(row["effort"])
        ref = _row_for(target, effort)
        slot: Dict[str, Any] = {"slot_id": _slot_id(surface, int(row["position"]))}
        if ref is None:
            diagnostics.append({
                "code": "reviewer_seat_inline_roster_full",
                "surface": surface, "target_id": target,
            })
            slot["route"] = {"kind": "agent_session", "target_id": target}
            slot["effort"] = effort
            return slot
        row_id, row_effort = ref
        slot["subagent_id"] = row_id
        if effort and effort != row_effort:
            slot["effort"] = effort
        return slot

    payload: Dict[str, Any] = {
        "triad": [_slot(SURFACE_TRIAD, row) for row in triad],
        "scope": [_slot(SURFACE_SCOPE, row) for row in scope],
    }
    adv = _slot(SURFACE_ADVISORY, dict(advisory) | {"position": 1})
    adv.pop("slot_id", None)
    payload["advisory"] = {"enabled": True, **adv}
    extended = make_configured_subagents(items, enabled=available.enabled)
    return extended, json.dumps(payload, ensure_ascii=False, sort_keys=False), diagnostics


def _validate_against_parser(raw: str, subagents_raw: str = "") -> Optional[PresetRefusal]:
    """Feed our own output through the reviewer-slot SSOT parser.

    The compiler does not maintain a second schema: if the ONE strict parser
    every consumer uses refuses this value, the preset is not applied.
    ``subagents_raw`` threads the roster THIS preset ships, so actor
    references validate against it (context-local override; the process env
    does not carry the not-yet-applied roster)."""
    from ouroboros.reviewer_slot_config import reviewer_slot_save_check

    try:
        reviewer_slot_save_check(raw, subagents_raw=subagents_raw or None)
    except ValueError as exc:
        return PresetRefusal(
            code="preset_failed_slot_validation", seat=None, candidates=(),
            message=f"The compiled reviewer-slot value did not validate: {exc}",
        )
    return None


def connected_preset_harnesses(discoveries: Sequence[HarnessDiscovery]) -> Tuple[str, ...]:
    """Recognized task harnesses the caller vouched for, in stable policy order."""
    seen = {str(d.harness_id) for d in discoveries}
    return tuple(h for h in PRESET_HARNESSES if h in seen)


def _local_model(settings: Mapping[str, Any], slot: str, model: str) -> str:
    flag = str(settings.get(f"USE_LOCAL_{slot}") or "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return model
    if not str(settings.get("LOCAL_MODEL_SOURCE") or "").strip():
        return ""
    return model if model.endswith(" (local)") else f"{model} (local)"


def _effective_api_models(settings: Mapping[str, Any]) -> Tuple[str, str]:
    main = str(settings.get("OUROBOROS_MODEL") or "").strip()
    light_raw = str(settings.get("OUROBOROS_MODEL_LIGHT") or "").strip()
    light = light_raw or main
    main = _local_model(settings, "MAIN", main) if main else ""
    light = _local_model(settings, "LIGHT" if light_raw else "MAIN", light) if light else ""
    if main and not model_has_credentials_in_settings(main, dict(settings)):
        main = ""
    if light and not model_has_credentials_in_settings(light, dict(settings)):
        light = ""
    return main, light


def _actor(
    row_id: str, recommendation: str, route: RouteSpec, effort: str,
) -> ConfiguredSubagent:
    # `name` is retired (owner decision 1=A): identity is the neutral id plus
    # derived route facts; recommended_use is the one semantic field.
    return ConfiguredSubagent(
        subagent_id=row_id,
        recommended_use=recommendation,
        route=route,
        effort=effort,
    )


def _task_effort(settings: Mapping[str, Any]) -> str:
    effort = str(settings.get("OUROBOROS_EFFORT_TASK") or "medium").strip().lower()
    from ouroboros.config import EFFORT_SCALE

    return effort if effort in EFFORT_SCALE else "medium"


def compile_available_subagents(
    session_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> Tuple[Optional[ConfiguredSubagents], Tuple[Dict[str, Any], ...]]:
    """Linear actor policy over truthful session and normalized API/local routes.

    Install discovery leaves ``credential_profile_id`` absent so Claudexor can
    rotate accounts.  The bounded singleton migration may supply the same
    resolved row with its historical pin; both paths deliberately share this
    one composition policy.
    """
    actors: list[ConfiguredSubagent] = []
    diagnostics: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for index, row in enumerate(session_rows):
        harness = str(row["harness"])
        route = RouteSpec(
            ROUTE_KIND_AGENT_SESSION,
            str(row["target_id"]),
            str(row.get("credential_profile_id") or "").strip(),
        )
        identity = (route.kind, route.target_id, route.credential_profile_id)
        if identity in seen:
            continue
        if index == 0:
            row_id, recommendation = "primary-builder", PRIMARY_RECOMMENDATION
        elif index == 1:
            row_id, recommendation = "independent-perspective", INDEPENDENT_RECOMMENDATION
        else:
            row_id, recommendation = (
                f"alternative-builder-{harness}", ALTERNATIVE_RECOMMENDATION,
            )
        actors.append(_actor(row_id, recommendation, route, str(row["effort"])))
        seen.add(identity)

    main, light = _effective_api_models(settings)
    main_effort = _task_effort(settings)
    if not actors and (main or light):
        primary_api = main or light
        route = RouteSpec(ROUTE_KIND_API_MODEL, primary_api)
        actors.append(_actor(
            "primary-builder", PRIMARY_RECOMMENDATION, route,
            main_effort if main else "low",
        ))
        seen.add((route.kind, route.target_id, ""))

    if light:
        route = RouteSpec(ROUTE_KIND_API_MODEL, light)
        identity = (route.kind, route.target_id, "")
        if identity not in seen:
            actors.append(_actor("fast-scout", SCOUT_RECOMMENDATION, route, "low"))
            seen.add(identity)

    if len(session_rows) == 1 and main:
        route = RouteSpec(ROUTE_KIND_API_MODEL, main)
        identity = (route.kind, route.target_id, "")
        if identity not in seen:
            actors.append(_actor(
                "independent-perspective", INDEPENDENT_RECOMMENDATION, route, main_effort,
            ))
            seen.add(identity)

    if len(session_rows) == 1 and not (main or light):
        diagnostics.append({
            "code": "no_api_or_local_companion",
            "message": (
                "Only one connected session is available; configure a reachable Main or "
                "Light API/local route to add another real actor."
            ),
        })
    elif not session_rows and main and light and main == light:
        diagnostics.append({
            "code": "duplicate_api_routes_omitted",
            "message": "Main and Light resolve to the same route, so no duplicate actor was created.",
        })

    if not actors:
        return None, ({
            "code": "no_available_subagents",
            "message": "No connected session or credentialed API/local model is available.",
        },)
    if len(actors) > 10:
        omitted = actors[10:]
        diagnostics.append({
            "code": "available_subagents_omitted",
            "count": len(omitted),
            "subagent_ids": [row.subagent_id for row in omitted],
        })
        actors = actors[:10]
    return make_configured_subagents(actors), tuple(diagnostics)


def compile_install_preset(
    discoveries: Sequence[HarnessDiscovery],
    *,
    settings: Optional[Mapping[str, Any]] = None,
    configured_subagents: Optional[ConfiguredSubagents] = None,
    source: str = SOURCE_ONBOARDING_DEFAULT,
    capability: Optional[Mapping[str, Any]] = None,
) -> SubscriptionInstallPreset:
    """Compile the install-time preset for the connected harnesses.

    ``discoveries`` are the harnesses whose accounts the caller already
    VERIFIED as connected, each with the live model ids discovered for it.
    ``capability`` is optional per-harness evidence (access profiles, engine
    notes); it is recorded in the receipt for disclosure and never gates a
    seat — an engine that answers a discovery list is the authority on what it
    can route.
    """
    settings = settings or {}
    connected = connected_preset_harnesses(discoveries)
    discovery = {str(d.harness_id): d for d in discoveries}
    required_models = (
        set(connected)
        if configured_subagents is None
        else set(connected).intersection(REVIEWER_PRESET_HARNESSES)
    )
    empty = [h for h in connected if h in required_models and not discovery[h].has_models]
    if empty:
        return SubscriptionInstallPreset(
            connected=connected,
            refusal=PresetRefusal(
                code="discovery_empty", seat=None, candidates=(),
                message=("Model discovery returned nothing for: "
                         f"{', '.join(empty)}. Preset models are written only from "
                         "live discovery, never guessed."),
            ),
        )
    task_rows: List[Dict[str, Any]] = []
    diagnostics: Tuple[Dict[str, Any], ...] = ()
    available = configured_subagents
    if available is None:
        task_rows, refusal = _resolve_surface(_compile_task_seats(connected), discovery)
        if refusal is not None:
            return SubscriptionInstallPreset(connected=connected, refusal=refusal)
        available, diagnostics = compile_available_subagents(task_rows, settings)
        if available is None:
            detail = diagnostics[0] if diagnostics else {}
            return SubscriptionInstallPreset(
                connected=connected,
                refusal=PresetRefusal(
                    code=str(detail.get("code") or "no_available_subagents"),
                    seat=None,
                    candidates=(),
                    message=str(detail.get("message") or "No truthful task actor is available."),
                ),
            )

    reviewer_connected = tuple(h for h in _POLICY_HARNESSES if h in connected)
    reviewer_slots = ""
    resolved: Dict[str, List[Dict[str, Any]]] = {}
    if reviewer_connected:
        seats = _compile_policy_seats(reviewer_connected)
        for surface in (SURFACE_ADVISORY, SURFACE_TRIAD, SURFACE_SCOPE):
            rows, refusal = _resolve_surface(seats[surface], discovery)
            if refusal is not None:
                return SubscriptionInstallPreset(connected=connected, refusal=refusal)
            resolved[surface] = rows
        if configured_subagents is None:
            available, reviewer_slots, ref_diagnostics = _reference_reviewer_rows(
                available, resolved[SURFACE_TRIAD], resolved[SURFACE_SCOPE],
                resolved[SURFACE_ADVISORY][0])
            if ref_diagnostics:
                diagnostics = tuple(diagnostics) + tuple(ref_diagnostics)
        else:
            reviewer_slots = _inline_reviewer_slots_json(
                resolved[SURFACE_TRIAD], resolved[SURFACE_SCOPE],
                resolved[SURFACE_ADVISORY][0])
        invalid = _validate_against_parser(
            reviewer_slots, serialize_configured_subagents(available))
        if invalid is not None:
            return SubscriptionInstallPreset(connected=connected, refusal=invalid)

    serialized_available = serialize_configured_subagents(available)
    receipt = {
        "version": SUBSCRIPTION_PRESET_VERSION,
        "connected": list(connected),
        "source": source,
        "available_subagents_fingerprint": configured_subagents_fingerprint(available),
        "available_subagents": configured_subagents_dict(available),
        "diagnostics": list(diagnostics),
        "surfaces": {
            SURFACE_SUBAGENT: task_rows,
            SURFACE_ADVISORY: (
                resolved[SURFACE_ADVISORY][0] if resolved.get(SURFACE_ADVISORY) else None
            ),
            SURFACE_TRIAD: resolved.get(SURFACE_TRIAD, []),
            SURFACE_SCOPE: resolved.get(SURFACE_SCOPE, []),
        },
        "discovery_counts": {h: len(discovery[h].model_ids) for h in connected},
        # Generated rows are deliberately unpinned so the daemon can rotate
        # accounts.  An exact owner draft may still carry a pin, and the
        # receipt must describe those saved bytes rather than the default.
        "profile_pinned": any(
            row.route.is_session and bool(row.route.credential_profile_id)
            for row in available.items
        ),
    }
    if capability:
        receipt["capability"] = dict(capability)
    return SubscriptionInstallPreset(
        connected=connected,
        reviewer_slots=reviewer_slots,
        available_subagents=serialized_available,
        source=source,
        diagnostics=diagnostics,
        receipt=receipt,
    )


__all__ = [
    "HARNESS_AGY",
    "HARNESS_CLAUDE",
    "HARNESS_CODEX",
    "HARNESS_CURSOR",
    "PRESET_HARNESSES",
    "PRESET_MARKER_KEY",
    "REVIEWER_SLOTS_KEY",
    "REVIEWER_PRESET_HARNESSES",
    "SUBSCRIPTION_PRESET_VERSION",
    "SURFACE_ADVISORY",
    "SURFACE_SCOPE",
    "SURFACE_SUBAGENT",
    "SURFACE_TRIAD",
    "HarnessDiscovery",
    "PresetRefusal",
    "PresetSeat",
    "SubscriptionInstallPreset",
    "compile_available_subagents",
    "compile_install_preset",
    "connected_preset_harnesses",
]
