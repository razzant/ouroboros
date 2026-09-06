"""Route health: the ONE manifest reader behind every delegated dispatch.

Extracted whole from ``subagents.py`` at its module ceiling (v7 D-U leaf) so the
route-manifest question keeps one home: ``route_health`` answers "can THIS route
run THIS shape right now" for the dispatcher and for the nanny's own
``delegate_start`` alike, and the quota readers below it (`_exhausted_window` and
its model-scope/cooldown predicates) are the only place a harness window is read
as spent. ``subagents`` re-exports every name, so historical imports keep
working; interception happens at THIS module (the leaf's own globals are what
``route_health`` reads) — patch ``ouroboros.subagent_route_health.X``, not the
``subagents`` alias, to intercept a helper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # annotation only — a runtime import would cycle with the parent
    from ouroboros.subagents import DelegatedRunShape


def route_health(
    gateway: Any, route_id: str, shape: DelegatedRunShape, *, route_model: str = "",
    pinned_profile: str = "",
) -> tuple[str, str]:
    """Return ``(unavailable_reason, reset_at)`` for a route about to run ``shape``.

    One reader, so the answer the DISPATCHER acts on and the answer the nanny's own
    ``delegate_start`` gets cannot drift into disagreeing about the same route. Health
    is asked about the SHAPE, not about a route in the abstract: a route that can only
    read is not a usable substrate for a child that must write, and an ENGINE that
    would reject the delegated marker outright is not a usable substrate for one either.

    ``route_model`` is the route's pinned model (``DelegationRoute.model``): quota
    windows scoped to OTHER models must not take this route offline, so exhaustion is
    judged against the model the run would actually use. A full-window exhaustion that
    names no reset instant still reports ``subscription_window_exhausted`` — as the
    REASON with an empty ``reset_at``, since an unknown healing time is not health.

    The harness row's aggregate doctor ``status`` is deliberately NOT a refusal
    here (cx-delegation sprint, owner decisions 2026-08-28 «статус обманывает,
    игнорируй его и всё равно пробуй запустить» + 7=A): it describes the
    DEFAULT credential store, while real accounts live in the engine's
    credential-profile pool — a pool-only harness read ``unavailable`` FOREVER
    (agy, INV-135) and blocked routes the engine itself would admit. The row's
    ``enabled`` field is different and IS honored for unpinned routes
    (``route_disabled``): the engine schema defines it as the OWNER's settings
    toggle — "routing excludes it regardless of doctor status" — an explicit
    owner decision, not an observation.
    Admission belongs to the engine: a genuinely empty or exhausted pool answers
    the start POST with its own typed refusal (INV-135
    ``credential_pool_exhausted`` + earliest reset), which under the pre-start
    charter costs zero model rounds. The engine's belt capability row
    (``delegation.available`` — MCP-injection for Claudexor's own delegate
    strategy) is likewise not consulted: Ouroboros runs never request the belt
    (no ``extra_mcp_servers``), and the only structural engine gate for a
    mutating run is the ``execution.delegated`` marker floor checked below.

    ``pinned_profile`` (``DelegationRoute.profile_id``; authors: reviewer-slot
    rows and the Delegation account pin, unified-accounts D-U5) narrows the
    QUOTA judgement to that subject exactly (§K.7): a pin is strict (D-U6), so
    a healthy sibling account must not mask a spent pinned one into a dispatch
    the engine is certain to refuse. Empty (automatic rotation) keeps the
    harness-wide judgement: WHICH profile an unpinned run lands on stays
    Claudexor's business.
    """
    from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
    from ouroboros.gateways.claudexor import engine_at_least

    catalog = gateway.agent_capabilities()
    entry = None
    for row in catalog.get("harnesses") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == route_id:
            entry = row
            break
    if entry is None:
        return "route_not_in_capability_catalog", ""
    if not pinned_profile and entry.get("enabled") is False:
        # `enabled` is NOT the doctor's aggregate status: the engine schema
        # defines it as the OWNER's settings toggle (harnesses.<id>.enabled=
        # false — "routing excludes it regardless of doctor status"). Honoring
        # an explicit owner switch is not health-guessing, so it survives the
        # status-refusal removal above. A pinned profile keeps its historical
        # skip (2026-08-18 precedent: the pin is itself an explicit owner row).
        return "route_disabled", ""
    supported = [str(v) for v in entry.get("accessProfilesSupported") or []]
    # A DELEGATED run is externally confined, and the engine rewrites its access to
    # `external_sandbox_full` before admitting it (`RequestRequirementsResolver.adapterAccess`)
    # — so the profile the route must declare is that one, not the literal the request
    # carries. Comparing the literal refused every route whose adapter stands its own
    # sandbox down in favour of the engine's boundary and therefore declares only the
    # confined profile: today opencode, which was given `external_sandbox_full` for
    # exactly this run. Refusing what the engine would admit turned `executor="harness"`
    # into a typed blocker and `auto` into a silent, metered drop to a native child.
    if shape.access not in supported and not (
        shape.delegated and "external_sandbox_full" in supported
    ):
        return f"access_profile_unsupported:{shape.access}", ""
    # An engine below the marker floor REJECTS `execution.delegated` outright — the field
    # is absent from a `.strict()` schema, so the start is a 400 and no run exists. That
    # is the only thing this version answers, and it is asked here so the refusal is typed
    # and arrives before a token is spent instead of as an opaque HTTP error mid-dispatch.
    # It says NOTHING about whether an admitted engine applies an OS boundary: that is a
    # per-attempt fact, read back from the run's own artifacts by
    # `tools.delegate._containment_evidence` and DISCLOSED rather than refused. The floor
    # cannot be a capability probe either — the marker is nested under `execution`, and
    # the catalog derives its key list from TOP-LEVEL request keys only.
    if shape.delegated and not engine_at_least(
        str(getattr(gateway, "engine_version", "") or ""),
        CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION,
    ):
        return "engine_rejects_delegated_marker", ""
    exhausted, reset_at = _exhausted_window(gateway, route_id, route_model, pinned_profile)
    if exhausted and not reset_at:
        # Spent with no named healing instant: still spent. The old shape carried
        # exhaustion ONLY in a non-empty reset, so a window the harness reports as
        # fully used but undated read back as a healthy route and the child was
        # dispatched onto a substrate that was going to refuse it.
        return "subscription_window_exhausted", ""
    return "", reset_at


def _exhausted_window(gateway: Any, route_id: str, route_model: str = "",
                      pinned_profile: str = "") -> tuple[bool, str]:
    """``(exhausted, reset_at)`` for a route judged against its OWN model.

    A window counts as spent when the harness reports it fully used or still cooling
    down (a FUTURE ``cooldown_until``) AND its model scope covers the route's model —
    a window scoped to a model this route never uses (the live incident: a Fable-only
    weekly window taking an opus-pinned route offline for days) is someone else's
    exhaustion, not this route's. Stale snapshots are ignored — an old reading must
    not block a lane.

    ANY LIVE SNAPSHOT MEANS THE LANE IS USABLE (D28). And exhaustion needs POSITIVE
    evidence for the WHOLE route: a profile whose quota could not be read at all
    (absent — a 429 on the usage endpoint, a failed refresh) is UNKNOWN, not spent,
    so it fail-opens the route: the daemon owns rotation and answers a genuinely
    empty route with its own typed refusal at start time, which costs nothing here.
    Only when every readable profile is spent and none is unreadable is there
    something to wait for; the honest instant is the EARLIEST named reset (possibly
    none — spent windows are not obliged to carry one).

    A snapshot with an applicable spent constraint counts as spent even if another of
    ITS OWN constraints has room: a 5-hour window at 100% blocks that profile now,
    whatever its weekly window says. WHICH profile an UNPINNED run lands on is
    Claudexor's business — rotation stays there and no profile identity is
    interpreted here. A PINNED route (``pinned_profile`` non-empty, D-U6 strict pin)
    is the one exception: the run can only ever land on that subject, so only ITS
    snapshots and absences are consulted — exact ``subject_id`` match, the same
    rule the accounts panel's quotaSummary applies — and a healthy sibling cannot
    vouch for it. All the fail-open rules above still hold per subject: a pinned
    account with no readable quota at all is UNKNOWN, not spent.
    """
    pinned = str(pinned_profile or "")

    def _subject_matches(subject: Dict[str, Any]) -> bool:
        if str(subject.get("harness") or "") != route_id:
            return False
        return not pinned or str(subject.get("subject_id") or "") == pinned

    quota_state = getattr(gateway, "quota_state", None)
    if callable(quota_state):
        envelope = quota_state()
        snapshots = envelope.get("snapshots") if isinstance(envelope, dict) else []
        absences = envelope.get("absences") if isinstance(envelope, dict) else []
    else:
        # Compatibility for older gateway doubles/embedders. Production
        # ClaudexorGateway owns quota_state and therefore performs one GET.
        snapshots = gateway.quota_snapshots()
        absence_reader = getattr(gateway, "quota_absences", None)
        absences = absence_reader() if callable(absence_reader) else []

    # The current daemon schema guarantees arrays, but route health also supports
    # older embedders and test doubles. A malformed mandatory snapshot collection
    # is unknown (fail-open); malformed optional absences add no evidence.
    snapshots = snapshots if isinstance(snapshots, list) else []
    absences = absences if isinstance(absences, list) else []

    resets: List[str] = []
    any_live = False
    any_spent = False
    for snapshot in snapshots or []:
        if not isinstance(snapshot, dict):
            continue
        subject = snapshot.get("subject") if isinstance(snapshot.get("subject"), dict) else {}
        if not _subject_matches(subject):
            continue
        if str(snapshot.get("freshness") or "") != "fresh":
            continue
        spent_here = [
            (str(c.get("cooldown_until") or "") or str(c.get("resets_at") or ""))
            for c in (snapshot.get("constraints") or [])
            if isinstance(c, dict)
            and (_cooldown_active(c.get("cooldown_until"))
                 or (isinstance(c.get("used_ratio"), (int, float))
                     and float(c.get("used_ratio")) >= 1.0))
            and _model_scope_matches(route_model, c.get("applies_to_models"))
        ]
        if spent_here:
            any_spent = True
            resets.extend(reset for reset in spent_here if reset)
        else:
            any_live = True
    if any_live or not any_spent:
        return False, ""
    for row in absences or []:
        subject = row.get("subject") if isinstance(row, dict) else None
        if not isinstance(subject, dict) or not _subject_matches(subject):
            continue
        # Any explicit gap keeps a spent-looking route fail-open. The shipped
        # producer already removes absences covered by a snapshot for the same
        # quota subject; retaining the conservative check here also keeps old
        # gateway doubles and malformed future envelopes from authorizing a
        # fallback on contradictory evidence.
        return False, ""
    return True, min(resets) if resets else ""


def _model_scope_matches(route_model: str, applies_to_models: Any) -> bool:
    """Does a quota constraint's model scope cover the route's pinned model?

    An empty/absent scope is a GLOBAL window — it always applies. An unpinned route
    (no model in ``OUROBOROS_SUBAGENT_HARNESS``) can land on any model, so every
    scoped window applies to it too. Otherwise the scope's aliases are matched by
    case-insensitive containment either way ("opus" ↔ "claude-opus-5"): the harness
    names windows by its own alias vocabulary, which this module must not enumerate.
    """
    aliases = [str(a).strip().lower() for a in (applies_to_models or []) if str(a).strip()]
    if not aliases:
        return True
    model = str(route_model or "").strip().lower()
    if not model:
        return True
    return any(a == model or a in model or model in a for a in aliases)


def _cooldown_active(cooldown_until: Any) -> bool:
    """A cooldown blocks only while its instant is still AHEAD: an expired
    ``cooldown_until`` is history the harness has not refreshed yet, not positive
    evidence of a spent window. An illegible instant keeps the conservative old
    reading (spent) — the harness positively said "cooling down" and an unreadable
    clock is no proof it healed."""
    text = str(cooldown_until or "").strip()
    if not text:
        return False
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return True
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant > datetime.now(timezone.utc)
