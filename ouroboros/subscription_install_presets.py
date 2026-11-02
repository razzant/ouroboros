"""Pure sibling compilers for Available subagents and the review pool.

The caller supplies already-normalized settings and, only when subscriptions
were declared, one verified live discovery snapshot. Task actors consume every
supported harness plus truthful Main/Light API or local routes. Review seats
consume only the independently ratified Claude/Codex/Cursor subset and land as
MARKED catalog rows (``review_eligible``) — the review pool — so Agy-only is a
successful task preset with no review rows and a mixed Agy install marks the
same rows its core-only projection does. ``factory_review_rows`` mints the
shipped API default panel into catalog rows for the migration and the wizard.

No settings or transport are read here. Exact discovery misses are typed
refusals with no partial output; a policy's explicit ordered successor (Codex:
gpt-6.1-sol, then gpt-5.6-sol) answers only from discovery, said in the receipt
and diagnostics. Rows are unpinned by default so Claudexor owns
compatible account rotation; Cursor and Agy keep effort in both their compound
model slug and the explicit row field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.configured_subagents import (
    ALTERNATIVE_RECOMMENDATION,
    MAX_CONFIGURED_SUBAGENTS,
    MINTED_FROM_FACTORY_DEFAULT,
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
    image_input_from_modalities,
    model_has_credentials_in_settings,
)
from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, RouteSpec

# The marker written beside the applied preset. Its ABSENCE is not authority to
# apply anything (every pre-preset install lacks it too); the server-side
# fresh-install latch is. Bumping this string marks a NEW preset generation.
SUBSCRIPTION_PRESET_VERSION = "4"
PRESET_MARKER_KEY = "OUROBOROS_SUBSCRIPTION_PRESET_VERSION"

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
SURFACE_TRIAD = "triad"
SURFACE_SCOPE = "scope"

# Harnesses whose discovery ids ENCODE the reasoning effort in the id itself.
_EFFORT_IN_MODEL_ID = frozenset({HARNESS_CURSOR, HARNESS_AGY})

# Owner shorthand -> ORDERED candidate discovery ids. ``{effort}`` is filled
# from the seat for a harness that spells effort inside the id. The FIRST
# candidate present in live discovery wins; when none is present the seat
# refuses. This is an alias table for ONE model family, never a fallback to a
# different model or to the harness default (a policy's explicit ``then`` is).
_MODEL_ALIASES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    HARNESS_CLAUDE: {
        "opus-5.5": ("claude-opus-5-5",),
    },
    HARNESS_CODEX: {
        "gpt-6.1-sol": ("gpt-6.1-sol",),
        "gpt-5.6-sol": ("gpt-5.6-sol",),
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
    then: Tuple[str, ...] = ()  # the policy's ordered successors when discovery lacks ``preference``


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
    available_subagents: str = ""
    source: str = SOURCE_ONBOARDING_DEFAULT
    diagnostics: Tuple[Dict[str, Any], ...] = ()
    receipt: Dict[str, Any] = field(default_factory=dict)
    refusal: Optional[PresetRefusal] = None
    model_settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.refusal is None and bool(self.available_subagents)

    def settings_keys(self, *, include_marker: bool = True) -> Dict[str, str]:
        """The EXACT settings keys an install-time save adds (the catalog — the
        review pool rides inside it — its receipt, and the preset marker). Empty
        on refusal — a half-applied preset is worse than none."""
        if not self.ok:
            return {}
        values = {
            SUBAGENTS_SETTING: self.available_subagents,
            SUBAGENTS_RECEIPT_KEY: json.dumps(self.receipt, ensure_ascii=False, separators=(",", ":")),
        }
        if include_marker:
            values[PRESET_MARKER_KEY] = SUBSCRIPTION_PRESET_VERSION
        return values


def _seat(surface: str, position: int, harness: str, spec: "SurfacePolicy") -> PresetSeat:
    return PresetSeat(surface=surface, position=position, harness=harness,
                      preference=spec.preference, effort=spec.effort, then=spec.then)


@dataclass(frozen=True)
class SurfacePolicy:
    preference: str
    effort: str
    then: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewerHarnessPolicy:
    """Review-only model choices; task actors have an independent catalog. Only the
    surfaces that mark rows are resolved: a model no row runs never refuses a preset."""

    triad: SurfacePolicy
    scope: SurfacePolicy


def _surface(preference: str, effort: str, then: Tuple[str, ...] = ()) -> SurfacePolicy:
    return SurfacePolicy(preference=preference, effort=effort, then=then)


# Codex prefers 6.1 Sol and takes 5.6 Sol only when live discovery lacks it (owner 2026-10-18).
_CODEX_SOL = _surface("gpt-6.1-sol", "medium", ("gpt-5.6-sol",))


_REVIEWER_POLICIES: Dict[str, ReviewerHarnessPolicy] = {
    HARNESS_CLAUDE: ReviewerHarnessPolicy(
        triad=_surface("opus-5.5", "medium"),
        scope=_surface("opus-5.5", "medium"),
    ),
    HARNESS_CODEX: ReviewerHarnessPolicy(
        triad=_CODEX_SOL,
        scope=_CODEX_SOL,
    ),
    HARNESS_CURSOR: ReviewerHarnessPolicy(
        triad=_surface("grok-4.6", "medium"),
        scope=_surface("grok-4.6", "high"),
    ),
}

_TASK_POLICIES: Dict[str, SurfacePolicy] = {
    HARNESS_CLAUDE: _surface("opus-5.5", "medium"),
    HARNESS_CODEX: _CODEX_SOL,
    HARNESS_CURSOR: _surface("grok-4.6", "high"),
    HARNESS_AGY: _surface("gemini-3.8-flash", "high"),
}

_POLICY_HARNESSES = (HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_CURSOR)
REVIEWER_PRESET_HARNESSES = _POLICY_HARNESSES
_TRIAD_ORDER = (HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_CURSOR)
_SCOPE_ORDER = (HARNESS_CODEX, HARNESS_CLAUDE, HARNESS_CURSOR)


def _first_connected(order: Sequence[str], connected: set[str]) -> str:
    return next(harness for harness in order if harness in connected)


def _policy_seat(surface: str, position: int, harness: str) -> PresetSeat:
    return _seat(surface, position, harness, getattr(_REVIEWER_POLICIES[harness], surface))


def _compile_policy_seats(connected: Sequence[str]) -> Dict[str, Tuple[PresetSeat, ...]]:
    """Apply linear priority rules; no powerset row grows when a harness is added.

    One triad seat per connected reviewer harness: a sole harness reviews once (owner
    Q5=A), never as three copies of one model that pay three runs for no second opinion.
    """
    present = set(connected)
    scope = _first_connected(_SCOPE_ORDER, present)

    triad_harnesses = [harness for harness in _TRIAD_ORDER if harness in present]

    return {
        SURFACE_TRIAD: tuple(
            _policy_seat(SURFACE_TRIAD, position, harness)
            for position, harness in enumerate(triad_harnesses, start=1)
        ),
        SURFACE_SCOPE: (_policy_seat(SURFACE_SCOPE, 1, scope),),
    }


def _compile_task_seats(connected: Sequence[str]) -> Tuple[PresetSeat, ...]:
    return tuple(
        _seat(SURFACE_SUBAGENT, position, harness, _TASK_POLICIES[harness])
        for position, harness in enumerate(connected, start=1)
    )


def _candidate_ids(seat: PresetSeat, preference: str = "") -> Tuple[str, ...]:
    aliases = _MODEL_ALIASES.get(seat.harness, {}).get(preference or seat.preference, ())
    return tuple(spelling.format(effort=seat.effort) for spelling in aliases)


def _resolve_seat(seat: PresetSeat,
                  discovery: Mapping[str, HarnessDiscovery]) -> Tuple[str, str, Optional[PresetRefusal]]:
    """The EXACT discovery id for a seat and the preference it answers (the policy's first,
    else its first ``then`` successor live discovery has), or a typed refusal naming them."""
    found = discovery.get(seat.harness)
    available = set(found.model_ids) if found else set()
    for preference in (seat.preference, *seat.then):
        for candidate in _candidate_ids(seat, preference):
            if candidate in available:
                return candidate, preference, None
    candidates = tuple(c for preference in (seat.preference, *seat.then) for c in _candidate_ids(seat, preference))
    if not candidates:
        return "", "", PresetRefusal(
            code="unknown_model_preference", seat=seat, candidates=(),
            message=(f"No alias is registered for {seat.preference!r} on the "
                     f"{seat.harness} harness — the {seat.surface} seat cannot be resolved."),
        )
    return "", "", PresetRefusal(
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


def _resolved_row(seat: PresetSeat, model_id: str, chosen: str = "") -> Dict[str, Any]:
    row = {
        "surface": seat.surface,
        "position": seat.position,
        "harness": seat.harness,
        "preference": seat.preference,
        "model": model_id,
        "effort": seat.effort,
        "effort_in_model_id": seat.harness in _EFFORT_IN_MODEL_ID,
        "target_id": _session_target(seat.harness, model_id),
    }
    if chosen and chosen != seat.preference:
        row["chosen_preference"] = chosen
        row["reason"] = (f"{seat.harness} discovery has no {seat.preference!r}; "
                         f"this seat uses the next preference, {model_id!r}.")
    return row


def _preference_notes(*surfaces: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """One diagnostic per harness that answered with a successor preference (receipt and preview)."""
    notes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in (row for rows in surfaces for row in rows if row.get("reason")):
        notes.setdefault((str(row["harness"]), str(row["model"])), {
            "code": "model_preference_successor", "harness": row["harness"], "preference": row["preference"],
            "chosen_preference": row["chosen_preference"], "model": row["model"], "message": row["reason"]})
    return tuple(notes.values())


def _resolve_surface(seats: Sequence[PresetSeat],
                     discovery: Mapping[str, HarnessDiscovery],
                     ) -> Tuple[List[Dict[str, Any]], Optional[PresetRefusal]]:
    rows: List[Dict[str, Any]] = []
    for seat in seats:
        model_id, chosen, refusal = _resolve_seat(seat, discovery)
        if refusal is not None:
            return [], refusal
        rows.append(_resolved_row(seat, model_id, chosen))
    return rows, None


_REVIEW_SEAT_RECOMMENDATION = (
    "Review-lane seat minted at onboarding; also selectable for delegated "
    "child work when its strengths fit."
)


def _free_review_id(used: set[str], base: str) -> str:
    """``<base>``, else the first free ``<base>-<n>`` (n from 2); a numeric base
    (``review-``) counts naturals from 1 — the migration's own minting order."""
    if base.endswith("-"):
        n = 1
        while f"{base}{n}" in used:
            n += 1
        return f"{base}{n}"
    row_id, suffix = base, 2
    while row_id in used:
        row_id, suffix = f"{base}-{suffix}", suffix + 1
    return row_id


def _mark_review_rows(
    available: ConfiguredSubagents,
    triad: Sequence[Mapping[str, Any]],
    scope: Sequence[Mapping[str, Any]] = (),
) -> tuple[ConfiguredSubagents, List[Dict[str, Any]]]:
    """The subscription wizard's review pool: MARK catalog rows for its seats.

    One SSOT from the first boot (owner decision 4=A, now literally one list):
    a seat whose session route already exists as an unmarked roster row with
    the seat's effort marks that row (``review_eligible``); otherwise a
    ``review-<harness>`` row is minted for it, marked, with
    ``minted_from: factory_default``. Each triad seat takes one row (the
    compiler names each harness once; identical seats would each mint their
    own twin, ``review-<harness>-2`` and on). A scope seat whose effective
    identity a marked row already has MERGES into it (every pool row judges
    both parts of the brief), else takes a row like a triad seat. If the
    catalog ceiling leaves no room, the seat is omitted and said so in
    diagnostics — honest disclosure, never a silent drop.

    Idempotent: a row already marked with a seat's identity COUNTS as that seat
    (each marked row satisfies one triad seat), so feeding the wizard's own
    output back — the catalog a preview showed, re-saved — changes nothing and
    mints no further twins.
    """
    items = list(available.items)
    diagnostics: List[Dict[str, Any]] = []
    taken: set[int] = set()  # row indices already standing for a seat of this run

    def _same(row: ConfiguredSubagent, target: str, effort: str) -> bool:
        return (row.route.kind == ROUTE_KIND_AGENT_SESSION and row.route.target_id == target
                and not row.route.credential_profile_id and row.effort == effort)

    def _first(target: str, effort: str, *, marked: bool) -> Optional[int]:
        return next((index for index, row in enumerate(items)
                     if index not in taken and row.review_eligible is marked and _same(row, target, effort)), None)

    for seat in [*({**s, "_merge": False} for s in triad), *({**s, "_merge": True} for s in scope)]:
        target, effort = str(seat["target_id"]), str(seat["effort"])
        if seat["_merge"] and any(row.review_eligible and _same(row, target, effort) for row in items):
            continue
        standing = _first(target, effort, marked=True)
        if standing is not None:
            taken.add(standing)
            continue
        unmarked = _first(target, effort, marked=False)
        if unmarked is not None:
            items[unmarked] = replace(items[unmarked], review_eligible=True)
            taken.add(unmarked)
        else:
            if len(items) >= MAX_CONFIGURED_SUBAGENTS:
                diagnostics.append({
                    "code": "reviewer_seat_omitted_roster_full",
                    "surface": str(seat.get("surface") or ""), "target_id": target,
                })
                continue
            harness = target.partition("=")[0] or "session"
            items.append(ConfiguredSubagent(
                subagent_id=_free_review_id({row.subagent_id for row in items}, f"review-{harness}"),
                recommended_use=_REVIEW_SEAT_RECOMMENDATION,
                route=RouteSpec(ROUTE_KIND_AGENT_SESSION, target),
                effort=effort,
                review_eligible=True,
                minted_from=MINTED_FROM_FACTORY_DEFAULT,
            ))
            taken.add(len(items) - 1)
    return make_configured_subagents(items, enabled=available.enabled), diagnostics


def _validate_pool(serialized: str) -> Optional[PresetRefusal]:
    """Feed the compiled catalog through the ONE strict parser every consumer
    uses, then the pool's save judge: a preset whose pool the save path would
    refuse is not applied."""
    from ouroboros.configured_subagents import parse_configured_subagents
    from ouroboros.reviewer_slot_config import review_pool_save_error

    try:
        parse_configured_subagents(serialized)
        error = review_pool_save_error(serialized, allow_empty=False)
    except ValueError as exc:
        error = str(exc)
    if error:
        return PresetRefusal(
            code="preset_failed_pool_validation", seat=None, candidates=(),
            message=f"The compiled review pool did not validate: {error}",
        )
    return None


def _document_catalog_ids(doc: Mapping[str, Any]) -> set[str]:
    """Row ids a settings document's catalog already holds (lenient: a
    malformed catalog contributes none — the migration refuses it elsewhere)."""
    raw = doc.get(SUBAGENTS_SETTING)
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        return {str(row.get("subagent_id") or "") for row in (payload or {}).get("items") or []
                if isinstance(row, dict)}
    except (TypeError, ValueError, AttributeError):
        return set()


def _factory_settings(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The configuration a factory row is derived from: ``doc`` over the shipped defaults, its
    unauthored model slots on their provider proposals and saved ids respelled onto the one
    exclusive provider — exactly what the runtime makes of the same document. A saved blank Main
    runs as the runtime reads it (``model_slots._main_model``: the shipped Main)."""
    from ouroboros.server_runtime import (
        apply_runtime_provider_defaults, authored_model_slots, propose_unauthored_model_slots,
    )
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    settings = propose_unauthored_model_slots({**SETTINGS_DEFAULTS, **doc}, authored_model_slots(doc))
    settings = apply_runtime_provider_defaults(settings)[0]
    settings["OUROBOROS_MODEL"] = _runtime_main(settings)
    return settings


def _runtime_main(settings: Mapping[str, Any]) -> str:
    """Main as the runtime runs it: a saved blank Main is ``model_slots._main_model``'s shipped Main
    (a mapping without the key names no Main; every merged view carries the slot)."""
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    main = settings.get("OUROBOROS_MODEL")
    return "" if main is None else str(main).strip() or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])


def role_route_binding(settings: Mapping[str, Any], role: str = "main") -> Dict[str, Any]:
    """Main's (or Light's) model as a catalog row runs it — the reachable route in full identity
    (the ``(local)`` label), a managed source's account pin and that role's processing — or ``{}``
    without access."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, resolve_processing_preference
    from ouroboros.provider_models import provider_for_model

    model = dict(zip(("main", "light"), _effective_api_models(settings))).get(role, "")
    if not model:
        return {}
    route: Dict[str, Any] = {"kind": ROUTE_KIND_API_MODEL, "target_id": model}
    try:  # a malformed role map is the settings validator's refusal, never a read-seam crash
        pin = str(model_role_option(MODEL_ACCOUNTS_KEY, role, settings=dict(settings)) or "")
        processing = resolve_processing_preference(role, settings=dict(settings))
    except ValueError:
        pin = processing = ""
    if pin and provider_for_model(model) == "claudexor":
        route["credential_profile_id"] = pin
    return {"route": route, **({"processing_preference": processing} if processing else {})}


def factory_review_rows(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The factory review pool for THIS settings document, as catalog rows (pure).

    The panel follows the Main the document runs (owner 4A), whatever other access sits beside
    it: a reachable OpenRouter Main → OpenRouter's three ``OPENROUTER_REVIEW_DEFAULTS``; a
    reachable direct Main → its provider's ``DIRECT_PROVIDER_REVIEW_ROLES`` panel (one Main row;
    MiniMax's Main, Light, Light), each seat in its ROLE's whole binding (a Light seat runs Light's,
    never Main's by a shared model name; a blank or another provider's Light leaves the provider's
    shipped Light standing in, plain); any other reachable Main (compatible, local, managed) → ONE
    row of it, in its whole binding
    (``role_route_binding``: ``(local)`` label, account pin, processing): copies of one model would
    multiply every review's cost without a second opinion, so its one answer decides (quorum 1 of
    1, recorded as ``single_reviewer_no_diversity``). With no configured access at all the shipped
    OpenRouter Main keeps its panel (the defaults before any provider is chosen). A Main the
    configured access cannot run — several direct keys and none chosen, or an explicit route
    without access — mints nothing rather than a panel derived from another default; the read
    seam names the setting. Every row is marked (``review_eligible``), ``minted_from:
    factory_default``, reads natively (the catalog's default delivery), and carries a non-empty
    effort: the document's ``OUROBOROS_EFFORT_REVIEW`` when it is on the scale, else ``high``
    (Astra F7). Ids are the first free ``review-<n>`` against the document's own catalog. Nothing
    is read from the process environment; task actors are ``factory_actor_rows``, never a
    reviewer template.
    """
    from ouroboros.config import EFFORT_SCALE
    from ouroboros.provider_models import compute_direct_review_seats, provider_for_model
    from ouroboros.server_runtime import has_startup_ready_provider
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    settings = _factory_settings(doc)
    main, light = settings["OUROBOROS_MODEL"], str(settings.get("OUROBOROS_MODEL_LIGHT") or "")
    router_panel = list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    bindings = {role: role_route_binding(settings, role) for role in ("main", "light")}
    main = bindings["main"]["route"]["target_id"] if bindings["main"] else main
    router_main = provider_for_model(main) == "openrouter"
    if bindings["main"] and not router_main:
        seats = compute_direct_review_seats(provider_for_model(main), main, light) or [("main", main)]
    else:  # a router Main reviews on the diverse router panel, never as twins of itself
        reachable = router_main and (bindings["main"] or not has_startup_ready_provider(settings))
        seats = [("", model) for model in router_panel] if reachable else []
    effort = str(settings.get("OUROBOROS_EFFORT_REVIEW") or "").strip().lower()
    if effort not in EFFORT_SCALE:
        effort = "high"
    used = _document_catalog_ids(settings)
    rows: List[Dict[str, Any]] = []
    for role, model in seats:
        row_id = _free_review_id(used, "review-")
        used.add(row_id)
        binding = bindings.get(role) or {}
        rows.append({
            "subagent_id": row_id,
            "recommended_use": _REVIEW_SEAT_RECOMMENDATION,
            "route": dict(binding["route"]) if binding else {"kind": ROUTE_KIND_API_MODEL, "target_id": model},
            "effort": effort,
            **({"processing_preference": binding["processing_preference"]}
               if "processing_preference" in binding else {}),
            "review_eligible": True,
            "minted_from": MINTED_FROM_FACTORY_DEFAULT,
        })
    return rows


def factory_actor_rows(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The wizard's API/local task actors for a never-configured install (pure): one Auto
    generalist on the reachable Main and a scout only for a reachable Light that differs in its
    whole binding — exactly ``compile_available_subagents`` without sessions, as the wizard runs it,
    so none without a reachable Main."""
    available, _diagnostics = compile_available_subagents((), _factory_settings(doc))
    return configured_subagents_dict(available)["items"] if available is not None else []


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
    main = _runtime_main(settings)
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
    row_id: str, recommendation: str, route: RouteSpec, effort: str, processing: str = "",
) -> ConfiguredSubagent:
    # `name` is retired (owner decision 1=A): identity is the neutral id plus
    # derived route facts; recommended_use is the one semantic field.
    return ConfiguredSubagent(
        subagent_id=row_id,
        recommended_use=recommendation,
        route=route,
        effort=effort,
        processing_preference=processing,
    )


def compile_available_subagents(
    session_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> Tuple[Optional[ConfiguredSubagents], Tuple[Dict[str, Any], ...]]:
    """Linear actor policy over truthful session and normalized API/local routes.

    Install discovery leaves ``credential_profile_id`` absent so Claudexor can
    rotate accounts.  The bounded singleton migration may supply the same
    resolved row with its historical pin; both paths deliberately share this
    one composition policy. A minted ACTOR row carries no ``effort`` — Auto, so
    the owner's effort range decides — unless its model id spells the level
    (``_EFFORT_IN_MODEL_ID``); reviewer seats keep their explicit levels. An
    API/local row runs its role's whole binding (``role_route_binding``), and a
    row's identity is that whole binding: rows alike in every setting are one,
    a Light on Main's model with its own account pin or processing is a second.
    Main is read as the runtime runs it (a saved blank is the shipped Main); a
    Light row only joins a session or that Main, so an API-only setup whose Main
    has no access gets the typed ``main_without_access``, never a Light primary.
    """
    actors: list[ConfiguredSubagent] = []
    diagnostics: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()  # route kind, target, account pin, processing

    for index, row in enumerate(session_rows):
        harness = str(row["harness"])
        route = RouteSpec(
            ROUTE_KIND_AGENT_SESSION,
            str(row["target_id"]),
            str(row.get("credential_profile_id") or "").strip(),
        )
        identity = (route.kind, route.target_id, route.credential_profile_id, "")
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
        actors.append(_actor(row_id, recommendation, route,
                             str(row["effort"]) if row.get("effort_in_model_id") else ""))
        seen.add(identity)

    def api_actor(row_id: str, recommendation: str, role: str) -> bool:
        binding = role_route_binding(settings, role)
        route = RouteSpec(ROUTE_KIND_API_MODEL, binding["route"]["target_id"],
                          binding["route"].get("credential_profile_id", ""))
        processing = binding.get("processing_preference", "")
        identity = (route.kind, route.target_id, route.credential_profile_id, processing)
        if identity in seen:
            return False
        actors.append(_actor(row_id, recommendation, route, "", processing))
        seen.add(identity)
        return True

    main, light = _effective_api_models(settings)
    if not actors and main:
        api_actor("primary-builder", PRIMARY_RECOMMENDATION, "main")
    scout = bool(actors and light) and api_actor("fast-scout", SCOUT_RECOMMENDATION, "light")
    if len(session_rows) == 1 and main:
        api_actor("independent-perspective", INDEPENDENT_RECOMMENDATION, "main")

    if len(session_rows) == 1 and not (main or light):
        diagnostics.append({
            "code": "no_api_or_local_companion",
            "message": (
                "Only one connected session is available; configure a reachable Main or "
                "Light API/local route to add another real actor."
            ),
        })
    elif not session_rows and main and light and not scout:
        diagnostics.append({
            "code": "duplicate_api_routes_omitted",
            "message": "Main and Light resolve to the same route, so no duplicate actor was created.",
        })

    if not actors:
        return None, ({
            "code": "main_without_access",
            "message": ("Main (OUROBOROS_MODEL) has no access in this setup, and the Light model does not "
                        "stand in for it: choose a Main model with access or connect an agent subscription."),
        } if light else {
            "code": "no_available_subagents",
            "message": "No connected session or credentialed API/local model is available.",
        },)
    if len(actors) > MAX_CONFIGURED_SUBAGENTS:
        omitted = actors[MAX_CONFIGURED_SUBAGENTS:]
        diagnostics.append({
            "code": "available_subagents_omitted",
            "count": len(omitted),
            "subagent_ids": [row.subagent_id for row in omitted],
        })
        actors = actors[:MAX_CONFIGURED_SUBAGENTS]
    return make_configured_subagents(actors), tuple(diagnostics)


def compile_model_settings(model_catalog: Sequence[Mapping[str, Any]],
                           settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Suggest a complete zero-key setup from exact raw-route catalog evidence.

    Existing usable/custom choices win. The model's backend default supplies
    the proposal, not an invented model family or a CLI alias table. Vision is
    a row whose input modalities state image input, else blank: Auto, which
    still tries an unknown route when an image needs one (``vision_routing``).
    Safety stays blank to inherit Light's full binding unless explicitly pinned.
    """
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS
    from ouroboros.settings_defaults import RETIRED_SETTING_KEYS, SETTINGS_DEFAULTS

    main = str(settings.get("OUROBOROS_MODEL") or "")
    if main and (model_has_credentials_in_settings(main, dict(settings))
                 or main != SETTINGS_DEFAULTS["OUROBOROS_MODEL"]):
        return {}
    default = next((row for row in model_catalog if row.get("is_default") and row.get("value")), None)
    if default is None:
        return {}
    vision = next((str(row["value"]) for row in model_catalog if row.get("value")
                   and image_input_from_modalities(row.get("input_modalities")) is True), "")
    proposed = {}
    for role, key in MODEL_ROLE_SETTINGS.items():
        if role == "websearch":
            continue  # Provider-owned web-search tools are a separate capability.
        if key in RETIRED_SETTING_KEYS:
            continue  # Never honored: the next load drops it with a retired-key notice to the owner.
        current = str(settings.get(key) or "")
        if current and current != str(SETTINGS_DEFAULTS.get(key) or ""):
            continue
        proposed[key] = {"fallback": "", "vision": vision, "safety": ""}.get(role, str(default["value"]))
    return proposed


def compile_install_preset(
    discoveries: Sequence[HarnessDiscovery],
    *,
    settings: Optional[Mapping[str, Any]] = None,
    configured_subagents: Optional[ConfiguredSubagents] = None,
    source: str = SOURCE_ONBOARDING_DEFAULT,
    capability: Optional[Mapping[str, Any]] = None,
    model_catalog: Sequence[Mapping[str, Any]] = (),
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
    model_settings = compile_model_settings(model_catalog, settings)
    settings = {**settings, **model_settings}
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
    resolved: Dict[str, List[Dict[str, Any]]] = {}
    if reviewer_connected:
        seats = _compile_policy_seats(reviewer_connected)
        for surface in (SURFACE_TRIAD, SURFACE_SCOPE):
            rows, refusal = _resolve_surface(seats[surface], discovery)
            if refusal is not None:
                return SubscriptionInstallPreset(connected=connected, refusal=refusal)
            resolved[surface] = rows
        # The review pool IS catalog rows, so an owner draft is extended too
        # (its marked rows are shown with it before the completion write); the
        # former inline lane had no catalog to live in. No advisory seat exists:
        # the preflight reviewer is a row the author names.
        available, ref_diagnostics = _mark_review_rows(
            available, resolved[SURFACE_TRIAD], resolved[SURFACE_SCOPE])
        if ref_diagnostics:
            diagnostics = tuple(diagnostics) + tuple(ref_diagnostics)
        invalid = _validate_pool(serialize_configured_subagents(available))
        if invalid is not None:
            return SubscriptionInstallPreset(connected=connected, refusal=invalid)

    diagnostics = tuple(diagnostics) + _preference_notes(task_rows, *resolved.values())
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
            SURFACE_TRIAD: resolved.get(SURFACE_TRIAD, []),
            SURFACE_SCOPE: resolved.get(SURFACE_SCOPE, []),
        },
        "review_pool": [row.subagent_id for row in available.items if row.review_eligible],
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
        # The structured lane is retired: the pool rides inside the catalog.
        available_subagents=serialized_available,
        source=source,
        diagnostics=diagnostics,
        receipt=receipt,
        model_settings=model_settings,
    )


__all__ = [
    "HARNESS_AGY",
    "HARNESS_CLAUDE",
    "HARNESS_CODEX",
    "HARNESS_CURSOR",
    "PRESET_HARNESSES",
    "PRESET_MARKER_KEY",
    "REVIEWER_PRESET_HARNESSES",
    "SUBSCRIPTION_PRESET_VERSION",
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
    "factory_actor_rows",
    "factory_review_rows",
    "role_route_binding",
]
