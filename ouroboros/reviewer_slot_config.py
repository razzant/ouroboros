"""Reviewer-slot configuration SSOT (phase 6.1).

ONE structured setting — ``OUROBOROS_REVIEWER_SLOTS`` — describes every
configured reviewer row: the commit-triad slots, the scope slots, the one
optional advisory reviewer, and the one optional deep self-reviewer
(``deep_review``; absent = synthesized from the legacy model key, see
``deep_review_slot``). Each row is::

    {"slot_id": "t_9f3a", "route": {"kind": "api_chat" | "agent_session",
                                    "target_id": "<model id | harness[=model]>"},
     "effort": "high"}

or, mutually exclusively, a reference to a configured subagent
(``OUROBOROS_SUBAGENTS`` row)::

    {"slot_id": "t_9f3a", "subagent_id": "<roster id>", "effort": "high"}

``slot_id`` is a STABLE owner-assigned identity, never an array index: a row's
receipts must keep lining up with its own history when the owner reorders or
edits rows (see ``review_substrate.slot_id_for_row`` for why a model is not an
identity either). ``target_id`` is an API model id on the ``api_chat`` kind and
an OPAQUE Claudexor route spec (``harness[=model]`` — Claudexor's own
reviewer-panel spelling, no ``::`` syntax) on ``agent_session``. Effort is a
per-row property on the existing ``EFFORT_SCALE`` — the same mechanism the
model lanes use, deliberately not a new one.

An actor reference never duplicates route/model/effort knobs: the roster row is
their SSOT, resolved ONCE at load/admission into the same frozen slot fields a
direct row carries (route drift after admission changes later waves only). An
``agent_session`` actor delivers through the existing session executor; an
``api_model`` actor delivers as bounded native tool rounds — retrieval, never
the assembled ``api_chat`` packet.

ABI 7.0 (ABI-10, owner 5.4=A): the legacy comma-list migration read is GONE —
the structured key is the ONE configuration surface. When it is absent the
loader serves the SHIPPED DEFAULT PANEL: api_chat triad/scope rows over the
effective default model lists (``get_review_models``/``get_scope_review_models``
— the derived env plane, which also honours a bench launcher's env override)
and the default advisory row. The comma keys are RETIRED settings
(``RETIRED_SETTING_KEYS``): an install that configured reviewers only through
them gets the default panel, and the RC auditor names that migration. The
comma ENV spellings survive solely as the derived runtime projection
(``project_reviewer_slots_into_env``) for the API-pinned surfaces. EVERY review
surface — commit, scope, plan, advisory, skill review and task acceptance —
follows its configured delivery rows; the triad rows reach plan review, skill
review and task acceptance through ONE builder (``triad_delivery_slots``), so
no surface reads a projection of the panel instead of the panel (owner
decision R2, 2026-09-01: the former task-acceptance API pin and its
default-panel fallback are gone).

Malformed configuration RAISES: mapping a typo to ``api_chat`` would silently
spend the API money the owner configured the row to move off of, and mapping
it to ``agent_session`` would silently delegate a row the owner never
delegated.
"""

from __future__ import annotations

import contextlib as _contextlib
import json
import os
import pathlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ouroboros.route_spec import (
    ROUTE_KIND_AGENT_SESSION as SHARED_ROUTE_KIND_SESSION,
    ROUTE_KIND_API_MODEL as SHARED_ROUTE_KIND_API,
    RouteSpec,
    compound_session_effort,
    parse_route_spec,
    validate_compound_session_effort,
)

REVIEWER_SLOTS_ENV = "OUROBOROS_REVIEWER_SLOTS"

ROUTE_KIND_API = "api_chat"
ROUTE_KIND_SESSION = "agent_session"
# The one role hint every scope row carries (structured or default panel); the
# commit gate's wave admission renders a native scope seat's work-order with it.
SCOPE_ROLE_HINT = "scope reviewer"

# Real limits, shown in the UI instead of promising an arbitrary number (D14).
# Pinned against their owners by tests: the triad ceiling is
# ``tools.review.MAX_MODELS``; the scope pool is the parallel-review thread
# pool width. Imported lazily there (tools.review is heavy), asserted equal in
# tests so the copies cannot drift silently.
TRIAD_SLOT_LIMIT = 10
SCOPE_SLOT_LIMIT = 4

_SLOT_ID_MAX_CHARS = 64

# The deep self-review row is a singleton like the advisory: its identity is
# fixed (the UI reads «Выполняется как» under this id), never owner-minted.
DEEP_REVIEW_SLOT_ID = "deep_review_slot_1"


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
    # Optional manual credential pin (Q2-в): '' = the daemon's rotation policy
    # (D28 default). Meaningful on agent_session rows only.
    profile_id: str = ""
    # Optional configured-subagent reference (OUROBOROS_SUBAGENTS row id).
    # Mutually exclusive with an inline route in the STORED form; when set, the
    # execution fields above were resolved from the frozen roster row at load
    # time and the roster stays their SSOT. '' = ordinary direct row.
    subagent_id: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind == ROUTE_KIND_SESSION

    @property
    def native_retrieval(self) -> bool:
        """An ``api_model`` configured-subagent row: bounded native tool rounds.

        Kept OFF the closed public route vocabulary (``api_chat`` stays the
        wire kind); executor selection and admission read this derived fact.
        """
        return bool(self.subagent_id) and self.kind == ROUTE_KIND_API

    @property
    def retrieves(self) -> bool:
        """Delivery class: the reviewer reads the subject with its own tools.

        THE predicate admission/fit/authority callers must use instead of
        route-name comparisons — a session row and a native-retrieval actor
        row are one class here, and neither receives an assembled packet.
        """
        from ouroboros.review_execution import delivery_retrieves

        return delivery_retrieves(self.kind, self.subagent_id)


@dataclass(frozen=True)
class AdvisorySlotConfig:
    """The ONE optional advisory reviewer (D14) — on the shared row vocabulary.

    ``enabled=False`` is a standing owner decision with a constitutional
    consequence the UI must state: every reviewed commit then records an
    AUDITED BYPASS instead of an advisory verdict (never a silent skip).

    Delivery follows the shared closed kinds: an ``api_chat`` advisory row is
    a routed catalog model that runs the bounded NATIVE inspection episode
    (advisory is an inspection critic by definition — it never receives an
    assembled packet), ``agent_session`` is a delegated Claudexor run, and a
    ``subagent_id`` reference resolves the configured roster row. The retired
    legacy ``api`` kind (Claude-Agent-SDK spellings) is migrated at parse:
    a translatable target becomes its routed id; an untranslatable one keeps
    the row DISABLED with a loud typed reason, never a silently swapped model.
    """

    enabled: bool = True
    kind: str = ROUTE_KIND_API  # api_chat | agent_session
    # agent_session: harness[=model] spec ('' = shared route). api_chat: a
    # routed catalog model id ('' = the shipped advisory default).
    target_id: str = ""
    # api_chat keeps the historical low default. Session ``""`` means the
    # route's own default; an explicit/compound route effort is materialized on
    # legacy migration so Settings round-trips one authority.
    effort: str = "low"
    profile_id: str = ""  # optional manual credential pin (Q2-в); '' = rotation
    # Configured-subagent reference ('' = direct row); resolved at parse into
    # the execution fields above, exactly like triad/scope actor rows.
    subagent_id: str = ""
    # Non-empty ⇒ the row was force-disabled at parse with this typed reason
    # (currently only the unmapped legacy Claude-SDK target migration).
    disabled_reason: str = ""


@dataclass(frozen=True)
class ReviewerSlotConfig:
    triad: Tuple[ConfiguredReviewerSlot, ...]
    scope: Tuple[ConfiguredReviewerSlot, ...]
    advisory: AdvisorySlotConfig
    source: str  # "structured" | "default" (ABI 7.0: the legacy read is gone)
    # The optional deep self-review row on the shared vocabulary (no
    # ``enabled``: a deep review is owner-triggered, never a standing gate).
    # None = not configured; ``deep_review_slot`` then synthesizes the packed
    # api row from the deep-review model key. An api row WITHOUT a subagent
    # reference is the packed 1M-context review (the historical delivery); a
    # configured-subagent api row is a native inspection episode and an
    # agent_session row a delegated session — the same three deliveries as
    # every other surface, chosen by the same ``retrieves`` predicate.
    deep_review: Optional[ConfiguredReviewerSlot] = None


def structured_reviewer_slots_raw() -> str:
    return str(os.environ.get(REVIEWER_SLOTS_ENV, "") or "").strip()


def structured_reviewer_slots_present() -> bool:
    return bool(structured_reviewer_slots_raw())


def _valid_effort(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} effort must be a string"
        )
    effort = value.strip().lower()
    if not effort:
        return ""
    from ouroboros.config import EFFORT_SCALE

    if effort not in EFFORT_SCALE:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} names an unknown effort {effort!r}; "
            f"valid: {', '.join(EFFORT_SCALE)}"
        )
    return effort


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
    from ouroboros.subagents import parse_subagent_harness

    if parse_subagent_harness(route.target_id) is None:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} session target "
            f"{route.target_id!r} does not name a concrete harness route"
        )


import contextvars as _contextvars
_ROSTER_ENV_OVERRIDE: "_contextvars.ContextVar[Optional[dict]]" = _contextvars.ContextVar(
    "reviewer_roster_env_override", default=None)


def _resolve_actor_slot(
    slot_id: str, subagent_id: str, effort: str, where: str,
) -> ConfiguredReviewerSlot:
    """Materialize a configured-subagent reference into one frozen reviewer row.

    Resolution happens at load/admission time: the wave that materialized this
    row carries the resolved facts, and a later roster edit changes later
    loads only (the #285 freeze-at-admission class). An unknown, disabled, or
    invalid roster reference is a malformed reviewer configuration — the same
    typed ValueError authority every consumer of this parser already treats as
    fail-closed (save-time 400, review-time typed block), never a silent
    fallback to another route or model.
    """
    from ouroboros.subagent_runtime import (
        SubagentSelectionError,
        select_subagent_snapshot,
    )

    try:
        # The roster is read from the APPLIED env, the same plane this module's
        # own key lives on — a saved-but-unapplied roster edit is invisible
        # here exactly as a saved-but-unapplied reviewer-slot edit would be.
        # Save-time validation threads the INCOMING roster through the
        # context-local override (S4 atomicity) — never by mutating the
        # process env, which concurrent review dispatch could observe.
        _override = _ROSTER_ENV_OVERRIDE.get()
        snapshot, _legacy = select_subagent_snapshot(
            _override if _override is not None else os.environ,
            subagent_id=subagent_id,
        )
    except SubagentSelectionError as exc:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} subagent_id {subagent_id!r} does "
            f"not resolve: {exc.code}: {exc.detail}"
        ) from exc
    route = dict(snapshot.get("route") or {})
    target = str(route.get("target_id") or "")
    pin = str(route.get("credential_profile_id") or "")
    # Explicit row effort wins; otherwise the roster row's own effort; an empty
    # result falls through to the surface default via row_effort, as always.
    chosen_effort = effort or _valid_effort(snapshot.get("effort"), where)
    if str(route.get("kind") or "") == SHARED_ROUTE_KIND_SESSION:
        shared = RouteSpec(
            kind=SHARED_ROUTE_KIND_SESSION, target_id=target,
            credential_profile_id=pin,
        )
        _validate_concrete_session_target(shared, where)
        validate_compound_session_effort(
            shared, chosen_effort, setting=REVIEWER_SLOTS_ENV, where=where,
        )
        return ConfiguredReviewerSlot(
            slot_id=slot_id, kind=ROUTE_KIND_SESSION, target_id=target,
            effort=chosen_effort, session_target=target, profile_id=pin,
            subagent_id=subagent_id,
        )
    return ConfiguredReviewerSlot(
        slot_id=slot_id, kind=ROUTE_KIND_API, target_id=target,
        effort=chosen_effort, subagent_id=subagent_id,
    )


def _parse_slot(row: Any, where: str, seen_ids: set) -> ConfiguredReviewerSlot:
    if not isinstance(row, dict):
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: {where} is not an object")
    unknown = sorted(set(row) - {"slot_id", "route", "subagent_id", "effort"})
    if unknown:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} has unknown keys: {unknown}"
        )
    raw_slot_id = row.get("slot_id")
    if not isinstance(raw_slot_id, str):
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} slot_id must be a string"
        )
    slot_id = raw_slot_id.strip()
    if not slot_id or len(slot_id) > _SLOT_ID_MAX_CHARS:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} needs a stable non-empty slot_id "
            f"(≤{_SLOT_ID_MAX_CHARS} chars) — identity is never an array index"
        )
    if slot_id in seen_ids:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: slot_id {slot_id!r} appears twice; a row's "
            "receipts can only line up with ONE history"
        )
    seen_ids.add(slot_id)
    raw_ref = row.get("subagent_id")
    if raw_ref is not None and not isinstance(raw_ref, str):
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} subagent_id must be a string"
        )
    actor_ref = str(raw_ref or "").strip()
    if raw_ref is not None and not actor_ref:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} subagent_id must not be empty"
        )
    if actor_ref and row.get("route") is not None:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: {where} must use either route or "
            "subagent_id, not both — the roster row is the route's SSOT"
        )
    if actor_ref:
        return _resolve_actor_slot(
            slot_id, actor_ref, _valid_effort(row.get("effort"), where), where,
        )
    route = parse_route_spec(
        row.get("route"), setting=REVIEWER_SLOTS_ENV, where=where,
        kind_aliases={
            ROUTE_KIND_API: SHARED_ROUTE_KIND_API,
            ROUTE_KIND_SESSION: SHARED_ROUTE_KIND_SESSION,
        },
        pin_key="profile_id",
        reject_unknown=True,
        strict_strings=True,
        reject_api_pin=True,
    )
    kind = ROUTE_KIND_SESSION if route.is_session else ROUTE_KIND_API
    _validate_concrete_session_target(route, where)
    effort = _valid_effort(row.get("effort"), where)
    validate_compound_session_effort(
        route, effort, setting=REVIEWER_SLOTS_ENV, where=where,
    )
    return ConfiguredReviewerSlot(
        slot_id=slot_id, kind=kind, target_id=route.target_id,
        effort=effort,
        session_target=route.target_id if kind == ROUTE_KIND_SESSION else "",
        profile_id=route.credential_profile_id if kind == ROUTE_KIND_SESSION else "",
    )


def _migrate_sdk_advisory_target(raw_kind: str, target: str) -> tuple[str, str]:
    """Translate a retired Claude-SDK ``api``-kind target to ``(routed, reason)``.

    The Claude-Agent-SDK advisory transport is retired (owner decision,
    2026-08-29): its rows migrate to the routed catalog. Only translations
    that keep the SAME model are performed; anything else keeps the row
    DISABLED with a typed reason — a silently swapped reviewer model is the
    exact class this parser exists to refuse.
    """
    if raw_kind != "api":
        return target, ""
    base = target.replace("[1m]", "").strip()
    if not base or base in {"sonnet", "claude-sonnet-5"}:
        # '' and the shipped default spelling both meant claude-sonnet-5.
        return "", ""
    if "/" in base or "::" in base:
        return target, ""  # already a routed/provider-tagged id
    if base.startswith("claude-"):
        return f"anthropic/{base}", ""
    return target, "legacy_claude_sdk_target_unmapped"


def _resolve_advisory_actor(subagent_id: str, effort: str, enabled: bool) -> AdvisorySlotConfig:
    row = _resolve_actor_slot("advisory_slot_1", subagent_id, effort, "advisory")
    return AdvisorySlotConfig(
        enabled=enabled, kind=row.kind, target_id=row.target_id,
        effort=row.effort or ("low" if not row.is_session else ""),
        profile_id=row.profile_id, subagent_id=subagent_id,
    )


def _parse_advisory(raw: Any) -> AdvisorySlotConfig:
    if raw is None:
        return AdvisorySlotConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: advisory must be an object")
    unknown = sorted(set(raw) - {"enabled", "route", "kind", "target_id", "effort", "subagent_id"})
    if unknown:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: advisory has unknown keys: {unknown}"
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: advisory enabled must be a boolean")
    for key in ("kind", "target_id", "subagent_id"):
        if key in raw and not isinstance(raw[key], str):
            raise ValueError(
                f"{REVIEWER_SLOTS_ENV}: advisory {key} must be a string"
            )
    actor_ref = str(raw.get("subagent_id") or "").strip()
    if "subagent_id" in raw and not actor_ref:
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: advisory subagent_id must not be empty")
    route = raw.get("route")
    if route is not None and not isinstance(route, dict):
        # Same typed refusal _parse_slot gives (:150). Without it a non-dict
        # route reached `.get` on a str/list and raised AttributeError, which
        # escapes every `except ValueError` that treats this parser as the
        # typed authority — including the commit gate's fail-closed branch and
        # reviewer_slot_config_error's callers.
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: advisory route must be an object "
                         "{kind, target_id}")
    if actor_ref and (route is not None or ({"kind", "target_id"} & set(raw))):
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: advisory must use either subagent_id or a "
            "route, not both — the roster row is the route's SSOT"
        )
    if actor_ref:
        return _resolve_advisory_actor(
            actor_ref, _valid_effort(raw.get("effort"), "advisory"), enabled,
        )
    if route is not None and ({"kind", "target_id"} & set(raw)):
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: advisory must use either route or legacy "
            "kind/target_id, not both"
        )
    route_payload = dict(route or {})
    if "kind" not in route_payload:
        route_payload["kind"] = raw.get("kind") or ROUTE_KIND_API
    if "target_id" not in route_payload:
        route_payload["target_id"] = raw.get("target_id") or ""
    raw_kind = str(route_payload.get("kind") or "").strip().lower()
    shared_route = parse_route_spec(
        route_payload,
        setting=REVIEWER_SLOTS_ENV,
        where="advisory",
        kind_aliases={
            "api": SHARED_ROUTE_KIND_API,
            ROUTE_KIND_API: SHARED_ROUTE_KIND_API,
            ROUTE_KIND_SESSION: SHARED_ROUTE_KIND_SESSION,
        },
        pin_key="profile_id",
        allow_empty_target=True,
        reject_unknown=True,
        strict_strings=True,
        reject_api_pin=True,
    )
    if enabled and shared_route.is_session and not shared_route.target_id:
        raise ValueError(
            f"{REVIEWER_SLOTS_ENV}: enabled advisory agent_session route needs "
            "a non-empty target_id; shared-session fallback is legacy-only"
        )
    _validate_concrete_session_target(shared_route, "advisory")
    effort = _valid_effort(raw.get("effort"), "advisory")
    if not effort and not shared_route.is_session:
        effort = "low"
    validate_compound_session_effort(
        shared_route, effort, setting=REVIEWER_SLOTS_ENV, where="advisory",
    )
    target, disabled_reason = (
        _migrate_sdk_advisory_target(raw_kind, shared_route.target_id)
        if not shared_route.is_session else (shared_route.target_id, "")
    )
    if disabled_reason:
        import logging

        logging.getLogger(__name__).warning(
            "advisory row disabled: legacy Claude-SDK target %r has no same-model "
            "routed translation; pick a routed model or a configured subagent in "
            "Settings → Review lanes", target,
        )
    return AdvisorySlotConfig(
        enabled=enabled and not disabled_reason,
        kind=ROUTE_KIND_SESSION if shared_route.is_session else ROUTE_KIND_API,
        target_id=target,
        effort=effort,
        profile_id=shared_route.credential_profile_id,
        disabled_reason=disabled_reason,
    )


def _parse_deep_review(raw: Any, seen_ids: set) -> Optional[ConfiguredReviewerSlot]:
    """The optional deep self-review row: the shared row vocabulary minus
    ``slot_id`` (a singleton's identity is fixed) and minus ``enabled``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: deep_review must be an object")
    unknown = sorted(set(raw) - {"route", "subagent_id", "effort"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: deep_review has unknown keys: {unknown}")
    return _parse_slot({**raw, "slot_id": DEEP_REVIEW_SLOT_ID}, "deep_review", seen_ids)


def parse_reviewer_slots(raw: str) -> ReviewerSlotConfig:
    """Strict parse of the structured setting. Raises ValueError, row-precise."""
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{REVIEWER_SLOTS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{REVIEWER_SLOTS_ENV} must be a JSON object")
    unknown = sorted(set(payload) - {"triad", "scope", "advisory", "deep_review"})
    if unknown:
        raise ValueError(f"{REVIEWER_SLOTS_ENV} has unknown top-level keys: {unknown}")
    seen_ids: set = set()
    groups: Dict[str, List[ConfiguredReviewerSlot]] = {}
    for group, limit in (("triad", TRIAD_SLOT_LIMIT), ("scope", SCOPE_SLOT_LIMIT)):
        rows = payload.get(group)
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise ValueError(f"{REVIEWER_SLOTS_ENV}: {group} must be an array")
        if len(rows) > limit:
            raise ValueError(
                f"{REVIEWER_SLOTS_ENV}: {group} has {len(rows)} rows; the real "
                f"limit is {limit} (shown in the UI, not negotiable here)"
            )
        groups[group] = [
            _parse_slot(row, f"{group}[{idx}]", seen_ids) for idx, row in enumerate(rows)
        ]
    if not groups["triad"]:
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: triad needs at least one slot")
    if not groups["scope"]:
        raise ValueError(f"{REVIEWER_SLOTS_ENV}: scope needs at least one slot")
    return ReviewerSlotConfig(
        triad=tuple(groups["triad"]),
        scope=tuple(groups["scope"]),
        advisory=_parse_advisory(payload.get("advisory")),
        source="structured",
        deep_review=_parse_deep_review(payload.get("deep_review"), seen_ids),
    )



def authored_reviewer_slots_state(raw: str) -> Tuple[str, str]:
    """THE three states of the owner's structured setting, as ``(state, parse_error)``.

    ``("absent", "")`` — no structured value: the loader serves the shipped default
    panel. ``("authored", "")`` — the strict parser accepts the text: that panel runs.
    ``("invalid", <row-precise error>)`` — the text is malformed: the loader RAISES on it
    (``load_reviewer_slot_config``), so NO panel serves — commit review blocks as
    ``infra_failure`` (advisory enforcement merely warns and commits unreviewed), plan
    and skill review refuse through ``reviewer_slot_config_error`` — until the owner
    repairs the setting. The retired-keys notice reads this so it never announces a
    default panel that is not serving.
    """
    text = str(raw or "").strip()
    if not text:
        return "absent", ""
    try:
        parse_reviewer_slots(text)
    except ValueError as exc:
        return "invalid", str(exc)
    return "authored", ""

# ---------------------------------------------------------------------------
# Shipped default panel (ABI 7.0: served when no structured value is saved).
# ---------------------------------------------------------------------------


def _default_config() -> ReviewerSlotConfig:
    """The shipped default panel — api_chat rows over the effective defaults.

    Model lists come from the derived env plane (`get_review_models` /
    `get_scope_review_models`): on an unconfigured install those are the
    shipped defaults (provider-adapted for a single-direct-provider install);
    a bench launcher's env override plane keeps working unchanged. Row effort
    stays '' so `row_effort` resolves the surface default at use time. Slot
    ids reuse the deterministic per-row spelling so receipts keep lining up.
    """
    from ouroboros.config import get_review_models, get_scope_review_models
    from ouroboros.review_substrate import (
        SCOPE_SLOT_ID_PREFIX,
        SLOT_ID_PREFIX,
        slot_id_for_row,
    )

    def _rows(models, prefix):
        return tuple(
            ConfiguredReviewerSlot(
                slot_id=slot_id_for_row(idx + 1, prefix=prefix),
                kind=ROUTE_KIND_API,
                target_id=str(model),
            )
            for idx, model in enumerate(
                str(m) for m in (models or []) if str(m or "").strip()
            )
        )

    return ReviewerSlotConfig(
        triad=_rows(get_review_models(), SLOT_ID_PREFIX),
        scope=_rows(get_scope_review_models(), SCOPE_SLOT_ID_PREFIX),
        advisory=AdvisorySlotConfig(),
        source="default",
    )


def load_reviewer_slot_config() -> ReviewerSlotConfig:
    """THE loader: structured when present, the shipped default panel otherwise."""
    raw = structured_reviewer_slots_raw()
    if raw:
        return parse_reviewer_slots(raw)
    return _default_config()


def reviewer_slot_config_error() -> str:
    """The structured config's row-precise parse error, or '' when none (#116).

    Thin facade for the surfaces that must refuse loudly instead of running on
    a silently projected default panel (plan review, skill review). Reads ONLY
    the structured raw value — an install without it runs the shipped default
    panel and always returns ''. No caching: the check re-parses so a
    hot-reloaded fix is seen immediately."""
    return authored_reviewer_slots_state(structured_reviewer_slots_raw())[1]


# ---------------------------------------------------------------------------
# Consumer accessors.
# ---------------------------------------------------------------------------


def commit_triad_rows() -> List[ConfiguredReviewerSlot]:
    """Configured triad rows shared by commit, plan, and skill review."""
    return list(load_reviewer_slot_config().triad)


def commit_scope_rows() -> List[ConfiguredReviewerSlot]:
    return list(load_reviewer_slot_config().scope)


def advisory_slot_config() -> AdvisorySlotConfig:
    return load_reviewer_slot_config().advisory


def deep_review_slot(config: Optional[ReviewerSlotConfig] = None) -> ConfiguredReviewerSlot:
    """THE deep self-review row: the configured ``deep_review`` row, or the
    packed api row synthesized from the legacy model key.

    ``OUROBOROS_MODEL_DEEP_SELF_REVIEW`` stays the invisible migration source
    and fallback: an install that never saved a row keeps today's exact
    delivery (one packed 1M-context review on that model), and the row's own
    effort — resolved by ``row_effort(row, "deep_self_review")`` — outranks the
    surface key ``OUROBOROS_EFFORT_DEEP_SELF_REVIEW`` only when set (R6). A
    malformed structured value raises the same typed ValueError as every other
    consumer of this parser; the surface turns it into its typed refusal.
    ``config`` lets a caller that already parsed the setting avoid a second parse.
    """
    row = (config if config is not None else load_reviewer_slot_config()).deep_review
    return row if row is not None else synthesized_deep_review_slot()


def synthesized_deep_review_slot() -> ConfiguredReviewerSlot:
    """The packed api row the legacy model key stands for — the ONE synthesis
    rule, shared by ``deep_review_slot`` and the settings endpoint (which shows
    it beside a malformed structured value as the legacy-derived REPAIR
    PLACEHOLDER — no row is effective there until the setting is repaired;
    ``deep_review_slot`` itself raises on a malformed value)."""
    from ouroboros.config import get_deep_self_review_model

    return ConfiguredReviewerSlot(
        slot_id=DEEP_REVIEW_SLOT_ID, kind=ROUTE_KIND_API,
        target_id=get_deep_self_review_model(),
    )


def structured_scope_review_slots() -> Optional[list]:
    """The scope ReviewSlots from the structured SSOT, or None on legacy.

    Lives here (not in the substrate) purely for module-size altitude: the
    substrate stays the owner of ReviewSlot semantics and calls this first.
    """
    if not structured_reviewer_slots_present():
        return None
    return [
        _delivery_slot(row, effort_surface="scope_review", role_hint=SCOPE_ROLE_HINT)
        for row in commit_scope_rows()
    ]


def _delivery_slot(
    row: ConfiguredReviewerSlot, *, effort_surface: str, role_hint: str,
    default_effort: str = "", **slot_fields: Any,
) -> Any:
    """ONE configured row as the substrate's ``ReviewSlot``, carrying its own
    delivery: the route kind, the opaque session target and credential pin, and
    the configured-subagent binding the route seam turns into a native episode."""
    from ouroboros.config import resolved_review_model_target
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    # ABI-4: the local-route fact is read off the typed target constructed at
    # the review seam, not re-derived per model string here.
    return ReviewSlot(
        slot_id=row.slot_id,
        model=row.target_id,
        effort=row_effort(row, effort_surface, default=default_effort),
        role_hint=role_hint,
        use_local=(resolved_review_model_target(row.target_id).provider_route == "local"),
        route=(ReviewRouteKind.AGENT_SESSION if row.is_session
               else ReviewRouteKind.API_CHAT),
        session_target=row.session_target,
        session_profile=row.profile_id,
        subagent_id=row.subagent_id,
        **slot_fields,
    )


def triad_delivery_slots(
    *,
    role_hint: str = "",
    default_effort: str = "",
    config: Optional["ReviewerSlotConfig"] = None,
    **slot_fields: Any,
) -> List[Any]:
    """The configured triad rows as ``ReviewSlot`` objects — THE builder for
    plan review, skill/commit review (through ``commit_triad_delivery``'s
    aligned vectors) and task acceptance (owner R2: acceptance reads the same
    rows every other triad surface reads, with each row's own effort, session
    target, credential pin, configured-subagent binding and stable slot id).

    Every row rides its own delivery: an ``api_chat`` row receives the
    assembled packet, an ``agent_session`` row is a delegated retrieving
    reviewer, a ``subagent_id`` api row is a native retrieving episode — the
    substrate's route seam decides from the slot fields carried here. Effort is
    the row's explicit value, else a compound Cursor/Agy route's encoded value,
    else ``default_effort``, else the configured Review effort. Slot ids are the
    rows' own, owner-assigned on the structured config (ABI-10 retired the
    legacy comma-list read). ``slot_fields`` are the caller's per-surface ReviewSlot
    properties (timeout, output budget, temperature). A malformed structured
    value RAISES ValueError — every surface turns that into its typed refusal
    (R3); no surface has a silently projected default panel to fall back to.
    """
    rows = (config if config is not None else load_reviewer_slot_config()).triad
    return [
        _delivery_slot(
            row, effort_surface="review", role_hint=role_hint,
            default_effort=default_effort, **slot_fields,
        )
        for row in rows
    ]


def reviewer_slots(
    models: List[str] | None = None,
    *,
    effort: str = "medium",
    role_hint: str = "",
    id_prefix: str = "",
) -> List[Any]:
    """The configured reviewer rows, every one pinned ``api_chat``.

    Moved here from ``review_substrate`` for module altitude (P7); the
    substrate re-exports it. Per-row delegated delivery is a structured-SSOT
    fact (``OUROBOROS_REVIEWER_SLOTS`` rows — D14/6.1): the phase-5 per-row
    route envs are RETIRED settings keys (ABI-10) and are ignored here, so a
    row built from a plain model list is an api_chat call explicitly rather
    than by accident (the scope caller that fans out a delegated row overrides
    the route itself). Surfaces that follow the configured triad rows do not
    come here — they use ``triad_delivery_slots``.
    """
    from ouroboros.config import get_review_models, resolved_review_model_target
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import SLOT_ID_PREFIX, ReviewSlot, slot_id_for_row

    id_prefix = id_prefix or SLOT_ID_PREFIX
    raw_models = models if models is not None else get_review_models()
    named = [str(model) for model in (raw_models or []) if str(model or "").strip()]
    # ABI-4: the local-route fact comes off the typed target constructed at the
    # review seam (one predicate application, at construction) instead of a
    # per-string predicate call here.
    return [
        ReviewSlot(slot_id=slot_id_for_row(idx + 1, prefix=id_prefix), model=model, effort=effort,
                   role_hint=role_hint,
                   use_local=(resolved_review_model_target(model).provider_route == "local"),
                   route=ReviewRouteKind.API_CHAT)
        for idx, model in enumerate(named)
    ]


def commit_triad_delivery() -> Dict[str, Any]:
    """Aligned per-row delivery vectors for the commit triad and skill review.

    Those surfaces consume rows as parallel lists (models for display and slot
    construction, routes for delivery, efforts/session targets/ids as row
    properties); projecting them from ``triad_delivery_slots`` keeps the
    surfaces at their size gates, keeps the vectors impossible to misalign,
    and keeps ONE reader of the triad rows. Raises ValueError on a malformed
    configuration — the caller turns that into its typed infra block.
    """
    from ouroboros.review_execution import ReviewRouteKind

    config = load_reviewer_slot_config()
    slots = triad_delivery_slots(config=config, role_hint="multi-model review")
    return {
        "models": [slot.model for slot in slots],
        "routes": [slot.route for slot in slots],
        "efforts": [slot.effort for slot in slots],
        "session_targets": [slot.session_target for slot in slots],
        "session_profiles": [slot.session_profile for slot in slots],
        "slot_ids": [slot.slot_id for slot in slots],
        "subagent_ids": [slot.subagent_id for slot in slots],
        # The historical fingerprint identity survives for the UNCONFIGURED
        # panel (source="default", all api rows): a 7.0 upgrade must not lapse
        # every install's skill-review replay authority. ABI-10 retired the
        # legacy comma-list source, so "not structured" IS the default panel.
        "legacy_skill_fingerprint": (
            config.source != "structured"
            and all(slot.route is ReviewRouteKind.API_CHAT for slot in slots)
        ),
    }


def row_effort(
    row: ConfiguredReviewerSlot,
    surface: str,
    *,
    default: str = "",
) -> str:
    """Resolve one effort authority without contradicting a compound route.

    An explicit row field wins.  When it is absent, a Cursor/Agy compound model
    slug already carries the requested effort and therefore wins over the
    surface default.  Ordinary rows retain the existing surface default (or a
    caller's established local default, as Plan Review does).
    """
    if row.effort:
        return row.effort
    if row.is_session:
        encoded = compound_session_effort(RouteSpec(
            kind=SHARED_ROUTE_KIND_SESSION,
            target_id=row.session_target or row.target_id,
            credential_profile_id=row.profile_id,
        ))
        if encoded:
            return encoded
    if default:
        return default
    from ouroboros.config import resolve_effort

    return resolve_effort(surface)


# ---------------------------------------------------------------------------
# Save-time validation and the legacy comma-key projection.
# ---------------------------------------------------------------------------


# Measured acceptance-panel cost on the API packet delivery (plan §4.1, traces of
# 2026-09-01): the ONE-TIME migration disclosure quotes them (owner R12) so an
# owner whose triad now retrieves knows what each substantive task's acceptance
# panel used to cost and what it spends instead. History, not a price table.
_ACCEPTANCE_API_PANEL_MEASURED = (
    "measured on the API packet panel it was ≈12 s and ≈$0.07 per model row per task "
    "(median, OSWorld traces; a three-row panel ≈75 s / ≈$0.82 on ProgramBench; "
    "7.5–8.9% of a run's cost)"
)


def acceptance_delivery_disclosure(rows: Sequence[ConfiguredReviewerSlot]) -> str:
    """The one-time R12 disclosure for a triad that (newly) retrieves: which rows,
    and what every substantive task's acceptance panel spends on them."""
    named = ", ".join(
        f"{row.slot_id} ({'agent session ' + row.session_target if row.is_session else 'native inspection'}"
        f"{' via ' + row.subagent_id if row.subagent_id else ''} → {row.target_id})"
        for row in rows
    )
    return (
        f"Task acceptance now follows these triad rows, including the retrieving ones — {named}. "
        f"Every substantive task's acceptance panel runs on them from the next task: {_ACCEPTANCE_API_PANEL_MEASURED}. "
        "A native inspection row spends API money as rounds × one send; an agent-session row spends "
        "minutes of your subscription window per task instead. A triad that also carries an "
        "api_chat row keeps a packet panel beside them."
    )


def reviewer_slot_save_check(
    raw: str, *, subagents_raw: Optional[str] = None, previous_raw: Optional[str] = None,
) -> str:
    """Validate an incoming structured value; return the save-time disclosure.

    Raises ValueError (row-precise) on a malformed value so the save handler
    turns it into a 400. ``subagents_raw`` threads the roster the SAME save
    produces (S4 atomicity) through a context-local override — actor
    references validate against it without any process-env mutation.

    The disclosure is the ONE-TIME migration notice of owner R12: returned when
    the saved triad has a retrieving row (agent session or configured-subagent
    native inspection) and the previously stored value had none — a legacy
    comma-key config, a packet-only triad, an unknown/malformed previous value.
    A save that keeps an already-retrieving triad discloses nothing again. The
    former all-delegated API-fallback warning described a task-acceptance
    substitution that no longer exists (acceptance follows the rows, R2)."""
    with roster_env_override(subagents_raw) if subagents_raw is not None else _contextlib.nullcontext():
        retrieving = [row for row in parse_reviewer_slots(raw).triad if row.retrieves]
        if not retrieving:
            return ""
        try:
            if previous_raw and any(row.retrieves for row in parse_reviewer_slots(previous_raw).triad):
                return ""  # already disclosed when that value was saved
        except ValueError:
            pass  # a malformed previous value never ran a retrieving panel: disclose
    return acceptance_delivery_disclosure(retrieving)


@_contextlib.contextmanager
def roster_env_override(subagents_raw: str):
    """Parse reviewer rows against THIS roster instead of the process env —
    the save handler's incoming roster, or a benchmark container's one-model
    roster — without mutating the environment concurrent dispatch observes."""
    overlay = dict(os.environ)
    overlay["OUROBOROS_SUBAGENTS"] = str(subagents_raw)
    token = _ROSTER_ENV_OVERRIDE.set(overlay)
    try:
        yield
    finally:
        _ROSTER_ENV_OVERRIDE.reset(token)


def project_reviewer_slots_into_env() -> None:
    """Project the structured config into the legacy comma keys, at env-apply time.

    No review surface reads these keys while the structured key is present —
    commit, scope, plan, skill review and task acceptance all read the
    structured rows directly, both delivery kinds — but legacy consumers still
    do: the external review script's key ordering, benchmark manifests, and
    ``get_review_models()`` callers with no panel of their own. Only api_chat
    rows project (a session row's target is a ``harness[=model]`` spec, not a
    model id); a configured-subagent api row's target IS its roster model id
    and projects like any other api row. An all-session triad therefore
    leaves the comma key at the shipped default for those legacy readers —
    never for a review surface. This is a runtime DERIVATION, not a second
    write: settings.json holds the structured key alone, and a stale comma
    value there is overwritten here rather than winning silently.

    Also owns the historical default-if-empty floor for both comma keys (moved
    verbatim from ``apply_settings_to_env`` so the tail behavior is one place).

    A malformed structured value is logged loudly and left UNPROJECTED —
    env-apply runs at server startup, where raising would take the whole app
    down with it; the review surfaces themselves re-parse strictly and BLOCK
    with the precise error instead.
    """
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    raw = structured_reviewer_slots_raw()
    if raw:
        try:
            config = parse_reviewer_slots(raw)
        except ValueError:
            import logging

            logging.getLogger(__name__).error(
                "%s is malformed; legacy env keys left unprojected — review "
                "surfaces will block with the precise parse error",
                REVIEWER_SLOTS_ENV, exc_info=True,
            )
        else:
            api_triad = [r.target_id for r in config.triad if not r.is_session]
            api_scope = [r.target_id for r in config.scope if not r.is_session]
            if api_triad:
                os.environ["OUROBOROS_REVIEW_MODELS"] = ",".join(api_triad)
            else:
                os.environ.pop("OUROBOROS_REVIEW_MODELS", None)
            if api_scope:
                os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] = ",".join(api_scope)
            else:
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODELS", None)
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODEL", None)
    if not os.environ.get("OUROBOROS_REVIEW_MODELS"):
        os.environ["OUROBOROS_REVIEW_MODELS"] = ",".join(OPENROUTER_REVIEW_DEFAULTS["triad"])
    if not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS") and not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL"):
        os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] = ",".join(OPENROUTER_REVIEW_DEFAULTS["scope"])


# ---------------------------------------------------------------------------
# «Выполняется как» (D22): the last EFFECTIVE execution per slot.
#
# The UI projection of capability_delta — beside each SAVED row, what the row
# REALLY ran as last time (route, model, effort, verdict method, any deltas).
# Disclosure, never enforcement: nothing reads this back into routing.
# ---------------------------------------------------------------------------

LAST_EXECUTION_FILENAME = "reviewer_slot_last_execution.json"
_LAST_EXECUTION_CAP = 64  # slots are ≤ 10+4+1+1 (triad, scope, advisory, deep review); the cap only bounds junk growth


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


def record_reviewer_slot_executions(surface: str, actors: Any, slots_by_id: Dict[str, Any]) -> None:
    """Record each actor's last effective execution (best-effort, atomic).

    Written into the CANONICAL data plane (not the review drive): this is UI
    state beside the saved settings, not per-task forensics — those live in
    the durable actor records already.
    """
    from ouroboros.review_substrate import TYPED_FAILURE_FACT_KEYS
    from ouroboros.utils import utc_now_iso, write_text_atomic

    path = _last_execution_path()
    with _LAST_EXECUTION_LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        for actor in actors or []:
            slot = slots_by_id.get(getattr(actor, "slot_id", ""))
            if slot is None:
                continue
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
                # No "effort": no APPLIED effort exists anywhere upstream (no
                # applied/resolved effort in any receipt or telemetry), so the only
                # value available is the REQUESTED one — already recorded below. Echoing
                # it here dressed the request up as the applied value, the exact thing
                # the model rule above forbids.
                "verdict_method": str(usage.get("verdict_method") or ""),
            }
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
                    "effort": str(getattr(slot, "effort", "") or ""),
                    "session_target": str(getattr(slot, "session_target", "") or ""),
                    "profile_id": str(getattr(slot, "session_profile", "") or ""),
                    # Actor binding, when the row is a configured-subagent
                    # reference ('' = direct row) — disclosure, never routing.
                    "subagent_id": str(getattr(slot, "subagent_id", "") or ""),
                },
                "effective": effective,
                "capability_delta": usage.get("capability_delta") or [],
                "status": str(getattr(actor, "status", "") or ""),
            }
            # B1: typed failure facts, present only when the substrate carried them
            # (a later health surface reads them; absence stays honest absence).
            # ONE shared key list with the plan-row/wave projections (sources differ).
            for key in TYPED_FAILURE_FACT_KEYS:
                value = getattr(actor, key, None)
                if value:
                    row[key] = value
            data[str(actor.slot_id)] = row
        if len(data) > _LAST_EXECUTION_CAP:
            ordered = sorted(data.items(), key=lambda kv: str(kv[1].get("ts") or ""))
            data = dict(ordered[-_LAST_EXECUTION_CAP:])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=1))


def reviewer_slot_last_executions() -> Dict[str, Any]:
    """Read the projection ('' shape on any read problem — disclosure only)."""
    try:
        data = json.loads(_last_execution_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


__all__ = [
    "DEEP_REVIEW_SLOT_ID",
    "REVIEWER_SLOTS_ENV",
    "ROUTE_KIND_API",
    "ROUTE_KIND_SESSION",
    "SCOPE_SLOT_LIMIT",
    "TRIAD_SLOT_LIMIT",
    "AdvisorySlotConfig",
    "ConfiguredReviewerSlot",
    "ReviewerSlotConfig",
    "advisory_slot_config",
    "commit_scope_rows",
    "commit_triad_rows",
    "deep_review_slot",
    "synthesized_deep_review_slot",
    "load_reviewer_slot_config",
    "parse_reviewer_slots",
    "reviewer_slot_config_error",
    "authored_reviewer_slots_state",
    "project_reviewer_slots_into_env",
    "record_reviewer_slot_executions",
    "reviewer_slot_last_executions",
    "acceptance_delivery_disclosure",
    "reviewer_slot_save_check",
    "row_effort",
    "structured_reviewer_slots_present",
    "structured_scope_review_slots",
    "structured_reviewer_slots_raw",
    "triad_delivery_slots",
]
