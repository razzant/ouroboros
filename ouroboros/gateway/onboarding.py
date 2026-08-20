"""Atomic onboarding completion — ONE owner-scoped save (D-8).

Web onboarding used to finish with TWO writes: ``POST /api/settings`` with the
wizard payload, then ``POST /api/owner/runtime-mode``. A failure between them
left an install whose providers were saved and whose runtime mode was not, and
there was no seam where an install-time decision (the agent subscription
preset, the fresh-install ``light`` safety default) could be part of the same
transaction. ``POST /api/onboarding/complete`` replaces both with one ordered
transaction:

1. re-prove FRESH-INSTALL status server-side — a browser boolean is a request,
   never an authority;
2. validate the wizard payload through the SHARED setup validator and the
   startup gate (a subscription alone never satisfies it, D-1);
3. read ONE fresh Claudexor snapshot when the payload declares subscriptions
   were connected, and compile the preset from LIVE discovery;
4. apply the ordinary provider normalization FIRST, then add the structured
   preset keys on top (R8: normalization is continuous re-derivation, the
   preset is an install-time transaction — they must not be taught about each
   other);
5. persist settings + runtime mode + safety default + the one-shot preset
   marker in a single write whose eligibility is re-proved under the settings
   lock;
6. only then start the supervisor.

A daemon that cannot answer at save time is a TYPED failure that persists
NOTHING and keeps the wizard open, with an explicit "finish without agent
defaults" escape hatch (``skipSubscriptionPresets``). Saving a guessed model id
is never the fallback: the id would be written into the reviewer configuration
the owner believes is live, and would only fail later, inside a real review.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway.owner_settings import (
    CommitBoundary,
    SettingsLockUnavailable,
    SettingsPreconditionFailed,
    _owner_audit,
    _owner_write_settings,
    post_commit_failure_response,
    settings_document_digest,
    settings_document_mutation,
    unsaved_error,
)
from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    has_startup_ready_provider,
)
from ouroboros.settings_setup_contract import (
    ONBOARDING_COMPLETED_KEY,
    parse_subscription_intent,
)
from ouroboros.subscription_install_presets import (
    PRESET_HARNESSES,
    PRESET_MARKER_KEY,
    HarnessDiscovery,
    SubscriptionInstallPreset,
    compile_install_preset,
)

log = logging.getLogger(__name__)

# The ONE owner-facing sentence for every way the preset step can fail. The
# machine-readable ``code`` beside it says which; the copy stays constant so the
# wizard does not have to translate engine vocabulary.
#
# It does NOT assert that accounts were connected. One of the ways this step
# fails is `no_verified_account`, where live engine authority has just
# established the opposite of that sentence — the browser's observation was
# stale, or the account was removed between the Agents step and Save. Nor does
# it prescribe repairing the engine, because the typed `detail` beside it may
# simply read "claude: not signed in", which no repair addresses.
# The copy promises no action the detail cannot deliver: "finish without
# agent defaults" is always true, while "fix the cause and try again" is
# conditional — a matrix_row_absent refusal (a recognized-but-unratified
# combination, e.g. agy before its seats are dictated) has no fix a retry
# could pick up, unlike daemon_unavailable or no_verified_account.
PRESET_UNVERIFIED_MESSAGE = (
    "Agent defaults could not be applied, and nothing was saved. "
    "The detail below says why. You can finish without agent defaults, "
    "or fix the cause — where it names one — and try again."
)


@dataclass(frozen=True)
class PresetFailure:
    """Why the preset step could not run. Typed, and never half-applied."""

    code: str
    detail: str

    def as_response(self) -> JSONResponse:
        return unsaved_error(
            PRESET_UNVERIFIED_MESSAGE, 503, code=self.code, detail=self.detail, can_skip=True,
        )


# ---------------------------------------------------------------------------
# Reading the live account/model snapshot (the ONE Claudexor read).
# ---------------------------------------------------------------------------


# The daemon's credential-kind enum is {config_dir_login, oauth_token, api_key}.
# The first two ARE a signed-in vendor session — what a subscription is. The
# third is metered API spend, which is precisely what a preset row must never
# become (D-3): such a row would either refuse at review time or quietly bill
# the owner's API key for work they connected a subscription to cover.
_SUBSCRIPTION_CREDENTIAL_KINDS = frozenset({"config_dir_login", "oauth_token"})
# ``next_up.route`` for the native/default subject. The daemon names it
# ``local_session`` for a CLI login and ``api_key`` for a configured key; older
# daemons omit the field, in which case ``native_login_detected`` (the engine's
# own "a vendor login is detected") carries the claim on its own.
_API_KEY_NATIVE_ROUTE = "api_key"
# Harness rows the engine will not run at all. Anything else it publishes
# (``ok``, ``degraded``) stays admissible: degradation is the engine's business,
# and a preset seat still resolves against the models it discovered.
_UNRUNNABLE_HARNESS_STATUS = frozenset({"unavailable"})


def _discovery_rows(harnesses: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("id") or ""): row
        for row in (harnesses or [])
        if isinstance(row, dict) and str(row.get("id") or "")
    }


def _profile_index(profiles: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for wrapper in (profiles or []):
        if not isinstance(wrapper, dict):
            continue
        profile = wrapper.get("profile")
        if not isinstance(profile, dict):
            continue
        key = (str(profile.get("harness_id") or ""), str(profile.get("profile_id") or ""))
        index[key] = wrapper
    return index


def _profile_seat_verdict(
    harness: str, profile_id: str, profiles: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[bool, str]:
    """Is this named credential profile a SUBSCRIPTION seat? Durable facts only:
    credential kind, enabled, the doctor probe's presence verdict and vendor
    verification. None of them is a quota reading."""
    wrapper = profiles.get((harness, profile_id))
    if wrapper is None:
        return False, f"the engine names account {profile_id!r}, which it does not list"
    profile = wrapper.get("profile") if isinstance(wrapper.get("profile"), dict) else {}
    status = wrapper.get("status") if isinstance(wrapper.get("status"), dict) else {}
    credential_kind = str(profile.get("credential_kind") or "")
    if credential_kind not in _SUBSCRIPTION_CREDENTIAL_KINDS:
        return False, (f"account {profile_id!r} is an API credential "
                       f"({credential_kind or 'kind not reported'}), not a subscription")
    if not profile.get("enabled"):
        return False, f"account {profile_id!r} is disabled"
    availability = str(status.get("availability") or "")
    if availability and availability != "available":
        return False, f"account {profile_id!r} is {availability}"
    if str(status.get("verification") or "") != "passed":
        return False, f"account {profile_id!r} has not verified"
    return True, f"account {profile_id!r} ({credential_kind})"


def _next_up_verdict(
    harness: str, next_up: Dict[str, Any], account: Dict[str, Any],
    profiles: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[bool, str]:
    """Would an UNPINNED run of this harness route through a subscription seat
    RIGHT NOW? The daemon's own server-computed answer; Ouroboros does not
    re-derive the rotation (D28), it only judges whether the seat the engine
    names is a SUBSCRIPTION or an API key.

    ``next_up`` is the verdict the caller already resolved off the wire —
    the unified ``accountPools`` row first, the legacy per-harness
    ``harnessAccounts[].next_up`` second (dual-read, sprint plan §K.7) — and
    the two unions are judged side by side here: ``profile``/``none`` are
    shared spellings, ``native`` exists only on the legacy wire (it reads the
    legacy ``account`` row's own facts), ``api_key_route`` only in the pool
    union (frozen contract §L.1). An UNKNOWN kind — either wire growing a
    spelling this reader predates — is fail-safe: not a subscription verdict,
    and the caller's configured-seat scan still gets its say.

    This answers a MOMENT-IN-TIME question — see ``_configured_subscription_seat``
    for why the install-time preset cannot be decided by it alone."""
    kind = str(next_up.get("kind") or "")
    if kind == "none":
        return False, str(next_up.get("reason") or "the engine has nothing routable for it")
    if kind == "api_key_route":
        # The pool union's explicit API-key verdict (Q2=A: allowed under
        # auth_preference=auto, disclosed) — a maintained route, never a seat.
        return False, "an unpinned run would route through an API key, not a subscription"
    if kind == "native":
        if not account.get("native_login_detected"):
            return False, "no signed-in session is detected for the default account"
        if not account.get("native_credentials_enabled"):
            return False, "its default login is disabled in the engine's credential ladder"
        route = str(next_up.get("route") or "")
        if route == _API_KEY_NATIVE_ROUTE:
            return False, "an unpinned run would route through an API key, not a subscription"
        return True, f"default session ({route or 'route not reported'})"
    if kind == "profile":
        return _profile_seat_verdict(harness, str(next_up.get("profileId") or ""), profiles)
    return False, f"the engine reports an unknown routing state {kind or 'none given'!r}"


def _configured_subscription_seat(
    harness: str, next_up: Dict[str, Any], account: Dict[str, Any],
    profiles: Dict[Tuple[str, str], Dict[str, Any]],
    *, native_allowed: bool = True,
) -> Tuple[bool, str]:
    """Is a subscription seat CONFIGURED here — regardless of capacity right now?

    ``native_allowed=False`` is the unrunnable-harness-row case: a native
    default seat's only runnability signal is that row, so it cannot count
    there, while a named profile's own probe still can.

    Two questions the engine answers differently, and the preset must not
    collapse them into one:

    * "who would an unpinned run take right now" is ``next_up``, computed
      daemon-side from enabled profiles + default readiness + QUOTA (Claudexor
      INV-135), and documented there as informational — it never gates routing.
      It is a reading of this hour.
    * "is a subscription seat configured for this harness" is credential KIND,
      enabled, present and verified. Those are durable.

    The preset is a once-only install-time decision (D-4) that never runs again,
    so deciding it on the first question meant an owner who connected Claude and
    Codex during an hour when the Claude window happened to be spent got a
    Codex-only preset PERMANENTLY, with no seam left to revisit it. D-3 says an
    exhausted subscription row stays CONFIGURED and waits for capacity; it never
    falls back to API spend and it must not silently vanish from the
    configuration either. Out of capacity is not evidence of not-a-subscription.

    ``next_up`` — the caller's ALREADY-RESOLVED routing verdict (pool first,
    legacy second, same dual-read as the caller) — is still consulted here for
    the ONE thing only it can answer: whether the default login's EFFECTIVE
    route is the vendor session or an API key. A harness's ``auth_preference``
    can put a key ahead of a session that is signed in, and that IS durable —
    a seat billing the owner's API key is what D-3 forbids, spent window or
    not. Both unions spell it: the legacy ``native`` verdict's ``api_key``
    route, and the pool union's ``api_key_route`` kind (§L.1). On a purely
    unified payload ``account`` is empty, so the native short-circuit never
    fires and the named-profile scan below is the whole answer."""
    if (native_allowed and account.get("native_login_detected")
            and account.get("native_credentials_enabled")):
        kind = str(next_up.get("kind") or "")
        effective_api_key = (
            (kind == "native" and str(next_up.get("route") or "") == _API_KEY_NATIVE_ROUTE)
            or kind == "api_key_route")
        if not effective_api_key:
            return True, "signed-in default session"
    for (row_harness, profile_id) in sorted(profiles):
        if row_harness != harness:
            continue
        ok, evidence = _profile_seat_verdict(harness, profile_id, profiles)
        if ok:
            return True, evidence
    return False, ""


def subscription_routable_harnesses(
    snapshot: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(routable, refused)`` — which preset harnesses have a subscription the
    engine can run on, and why the others do not.

    An account being SIGNED IN is not the question, and neither is "would a run
    start this second". A once-only install-time decision needs the DURABLE one:
    the harness row must be enabled, and a subscription seat must be configured
    for it — where a NATIVE default seat also needs the row RUNNABLE (the row is
    that seat's only runnability signal), while a NAMED-profile seat is vouched
    by its own doctor probe and counts even on a structurally unavailable row
    (a harness with no default credential store, agy/Claudexor INV-135).
    The engine's `next_up` answers first, because when it does
    say yes the receipt records the seat a real run would take; when it says no,
    a configured seat still counts and the refusal is recorded as a capacity
    note. Everything the verdict rests on comes from the engine.

    DUAL-READ (unified account model, sprint plan §K.7 + frozen contract §L):
    a unified engine emits ``harnessAccounts: []`` — every account a named
    registry row — and carries the routing verdict in the ADDITIVE top-level
    ``accountPools: [{harness_id, next_up}]`` key instead. That pool row is
    the accounts authority there (skipping the harness because its legacy row
    is absent would silently drop every unified harness from the preset); on a
    legacy engine the per-harness account row keeps answering exactly as
    before. When both carry a verdict, the pool wins — profiles own account
    facts, the pool owns routing facts. Routing is NEVER re-derived from the
    profile list client-side; a unified harness with an unknown or refusing
    pool verdict falls to the configured-seat scan, which on that wire is the
    named-profile scan alone (there is no native fact to read)."""
    routable: Dict[str, str] = {}
    refused: Dict[str, str] = {}
    rows = _discovery_rows(snapshot.get("harnesses"))
    payload = snapshot.get("profiles") if isinstance(snapshot.get("profiles"), dict) else {}
    profiles = _profile_index(payload.get("profiles"))
    accounts = {
        str(row.get("harness_id") or ""): row
        for row in (payload.get("harnessAccounts") or [])
        if isinstance(row, dict)
    }
    pools = {
        str(row.get("harness_id") or ""): row.get("next_up")
        for row in (payload.get("accountPools") or [])
        if isinstance(row, dict) and isinstance(row.get("next_up"), dict)
    }
    for harness in PRESET_HARNESSES:
        account = accounts.get(harness)
        pool_next_up = pools.get(harness)
        if account is None and pool_next_up is None:
            continue  # the engine publishes no accounts authority for it: silent absence
        account = account if isinstance(account, dict) else {}
        row = rows.get(harness)
        if row is None:
            refused[harness] = "the engine does not list this harness"
            continue
        if not row.get("enabled"):
            refused[harness] = "the engine has this harness disabled"
            continue
        status = str(row.get("status") or "")
        unrunnable = status in _UNRUNNABLE_HARNESS_STATUS
        # A harness-level "unavailable" is no longer an outright refusal: an
        # engine whose harness has NO default credential store (agy — Claudexor
        # INV-135) reports the harness row STRUCTURALLY unavailable while its
        # named profiles run fine, and their per-profile doctor probes are the
        # runnability proof the harness row cannot give. A NATIVE default seat
        # still requires a runnable harness row — the row is that seat's only
        # runnability signal — which keeps the deliberate
        # signed-in-but-unavailable refusal for the classic harnesses.
        legacy_next_up = (account.get("next_up")
                          if isinstance(account.get("next_up"), dict) else {})
        next_up = pool_next_up if pool_next_up is not None else legacy_next_up
        next_kind = str(next_up.get("kind") or "")
        ok, evidence = _next_up_verdict(harness, next_up, account, profiles)
        if ok and (not unrunnable or next_kind == "profile"):
            routable[harness] = evidence if not unrunnable else (
                f"{evidence} (the harness row reads {status}: it has no default "
                "credential store, and the named-profile probe vouches the seat)")
            continue
        seated, seat = _configured_subscription_seat(
            harness, next_up, account, profiles, native_allowed=not unrunnable)
        if seated:
            # "No capacity" wording is reserved for genuine temporary
            # exhaustion; a structurally unavailable row (no default
            # credential store) discloses the structural cause instead.
            routable[harness] = (
                f"{seat} (the harness row reads {status}: it has no default "
                f"credential store, and the named-profile probe vouches the seat; {evidence})"
                if unrunnable else f"{seat}; no capacity right now ({evidence})")
        elif unrunnable:
            refused[harness] = f"the engine reports it {status}"
        else:
            refused[harness] = evidence
    return routable, refused


def verified_harness_discoveries(
    snapshot: Dict[str, Any],
) -> Tuple[Tuple[HarnessDiscovery, ...], Optional[PresetFailure]]:
    """Turn one ``/api/claudexor/status?include=models`` snapshot into the
    compiler's input, or a typed failure. PURE — unit-testable with no daemon."""
    daemon = snapshot.get("daemon") if isinstance(snapshot, dict) else None
    state = str((daemon or {}).get("state") or "")
    if state != "running":
        return (), PresetFailure(
            "daemon_unavailable",
            f"The agent engine is {state or 'not running'}"
            + (f" ({(daemon or {}).get('last_error')})" if (daemon or {}).get("last_error") else ""),
        )
    routable, refused = subscription_routable_harnesses(snapshot)
    rows = _discovery_rows(snapshot.get("harnesses"))
    wanted = [h for h in PRESET_HARNESSES if h in routable]
    if not wanted:
        detail = "; ".join(f"{harness}: {reason}" for harness, reason in sorted(refused.items()))
        return (), PresetFailure(
            "no_verified_account",
            "The engine can run no subscription session for "
            f"{', '.join(PRESET_HARNESSES)}." + (f" {detail}" if detail else ""),
        )
    discoveries: List[HarnessDiscovery] = []
    for harness in wanted:
        row = rows.get(harness) or {}
        if row.get("models_error"):
            return (), PresetFailure(
                "models_unavailable",
                f"Model discovery for {harness} failed: {row.get('models_error')}",
            )
        model_ids = tuple(
            str(model.get("id") or "")
            for model in (row.get("models") or [])
            if isinstance(model, dict) and str(model.get("id") or "")
        )
        if not model_ids:
            return (), PresetFailure(
                "models_unavailable",
                f"The engine listed no models for {harness}.",
            )
        discoveries.append(HarnessDiscovery(harness_id=harness, model_ids=model_ids))
    return tuple(discoveries), None


def _harness_capability(snapshot: Dict[str, Any], connected: Sequence[str]) -> Dict[str, Any]:
    """Disclosure-only evidence recorded in the receipt (never a gate)."""
    rows = _discovery_rows(snapshot.get("harnesses"))
    routable, _refused = subscription_routable_harnesses(snapshot)
    return {
        harness: {
            "status": str((rows.get(harness) or {}).get("status") or ""),
            "access_profiles_supported": list(
                (rows.get(harness) or {}).get("access_profiles_supported") or []),
            # WHICH seat the engine said an unpinned run would take. The rows are
            # unpinned by design (D28), so this records the evidence, not a pin.
            "subscription_route": routable.get(harness, ""),
        }
        for harness in connected
    }


def _read_harness_snapshot() -> Dict[str, Any]:
    """The ONE blocking Claudexor read, through the SAME projection the accounts
    panel uses (no second discovery path)."""
    from ouroboros.gateway.claudexor_accounts import _status_payload

    return _status_payload(True)


async def resolve_install_preset(
) -> Tuple[Optional[SubscriptionInstallPreset], Optional[PresetFailure]]:
    """One fresh snapshot -> one compiled preset, or a typed failure."""
    try:
        snapshot = await asyncio.to_thread(_read_harness_snapshot)
    except Exception as exc:  # a dead/broken engine is a failure, not a crash
        log.warning("Claudexor snapshot for onboarding presets failed", exc_info=True)
        return None, PresetFailure("daemon_unavailable", f"{type(exc).__name__}: {exc}")
    discoveries, failure = verified_harness_discoveries(snapshot)
    if failure is not None:
        return None, failure
    preset = compile_install_preset(
        discoveries,
        capability=_harness_capability(snapshot, [d.harness_id for d in discoveries]),
    )
    if not preset.ok:
        refusal = preset.refusal.as_dict() if preset.refusal else {}
        return None, PresetFailure(
            str(refusal.get("code") or "preset_refused"),
            str(refusal.get("message") or "The preset could not be compiled."),
        )
    return preset, None


# ---------------------------------------------------------------------------
# The install-time latch (server-side authority).
# ---------------------------------------------------------------------------


def install_is_unconfigured(settings: Dict[str, Any]) -> bool:
    """Is this install still IN onboarding, as the server itself sees it?

    The same predicate that decides whether ``GET /api/onboarding`` mounts the
    blocking overlay. It is NOT install-time on its own: an install that has run
    for a year and whose one provider key stopped working answers True here too,
    which is why ``preset_eligible`` requires two further proofs."""
    return not has_startup_ready_provider(settings)


def preset_eligible(settings: Dict[str, Any]) -> bool:
    """May this save apply the install-time agent preset (D-4)?

    Three independent proofs, because "no working provider" alone is a state an
    OLD install reaches whenever its key stops working — and presetting there
    would overwrite reviewer/subagent choices the owner made themselves:

    * onboarding has never completed here (the durable ``…COMPLETED_AT`` fact,
      written on EVERY completion — skipped and subscription-less included);
    * no preset generation has been applied;
    * the install still has no settings file at all, the same "genuinely fresh
      install" rule the wizard already uses for the ``light`` safety default.

    Marker ABSENCE alone would prove nothing (every install that predates this
    release lacks both), which is what the file-existence proof answers."""
    if str(settings.get(ONBOARDING_COMPLETED_KEY) or "").strip():
        return False
    if str(settings.get(PRESET_MARKER_KEY) or "").strip():
        return False
    if not install_is_unconfigured(settings):
        return False
    return _fresh_settings_file()


def _fresh_settings_file() -> bool:
    """No settings.json at all — the narrow condition under which onboarding may
    author the ``light`` safety default (same rule the desktop wizard uses)."""
    from ouroboros.settings_setup_contract import wizard_authors_safety_light

    return wizard_authors_safety_light()


def _write_precondition(expect_preset: bool, expect_safety_light: bool, read_fingerprint: str):
    """Re-prove eligibility INSIDE the settings lock, against the state this
    write is about to overwrite.

    The re-read is ``load_settings_lock_held``: the settings lock is not
    re-entrant, so the ordinary ``load_settings()`` would wait out its full 2s
    timeout and then read anyway — two seconds added to every onboarding save
    for a lock it already holds."""
    def _check() -> str:
        from ouroboros.config import SETTINGS_PATH, load_settings_lock_held

        # FRESHNESS FIRST, because it is the one condition that is about the
        # document rather than about this install's phase. The write that
        # follows is the WHOLE dictionary derived from an unlocked read, so a
        # concurrent owner write landing in between would be reverted key by
        # key while this request answered "saved" (BIBLE P1). Refusing here
        # keeps the transaction honest: the owner's other change survives and
        # this one is told, in the seam that already exists for exactly this,
        # that nothing was written.
        if _settings_fingerprint() != read_fingerprint:
            # Deliberately OVER-refuses in two narrow cases rather than risk
            # under-refusing in any: a write that lands in the microseconds
            # between the digest and the read is rejected even though this
            # request went on to derive from the newer document, and a
            # formatting-only rewrite of identical content is rejected because
            # the comparison is over bytes. Both cost the owner one retry, which
            # then succeeds; the opposite error costs them a change they made.
            return ("The settings file changed while onboarding was being saved, "
                    "so this save would have overwritten it; nothing was written. "
                    "Try finishing again.")
        if expect_safety_light and SETTINGS_PATH.exists():
            return ("A settings file appeared while onboarding was being saved; "
                    "refusing to author the first-install safety default over it.")
        if expect_preset and not preset_eligible(load_settings_lock_held()):
            return ("This install is no longer in first-run onboarding; refusing "
                    "to apply install-time agent defaults over it.")
        return ""

    return _check


# ---------------------------------------------------------------------------
# The endpoint.
# ---------------------------------------------------------------------------


def _settings_fingerprint() -> str:
    """What the settings document looked like at a given instant.

    Completion derives the WHOLE document from an unlocked read and then writes
    that whole dictionary back. Between the two, another owner write can land —
    and every key it changed would be silently restored to the value this
    request read, while the owner is told the save succeeded. The fingerprint
    turns that into something the locked precondition can notice.

    The digest itself is ``owner_settings.settings_document_digest``: the same
    staleness question the single-decision owner endpoints ask, so it has one
    answer rather than one per transaction.
    """
    return settings_document_digest()


def _prepared_settings(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """(old_settings, prepared_settings, error) through the SHARED validator."""
    from ouroboros.config import load_settings
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    old_settings = load_settings()
    prepared, error = prepare_onboarding_settings(body, old_settings)
    if error:
        return old_settings, {}, str(error)
    normalized, _changed, _keys = apply_runtime_provider_defaults(prepared)
    if not has_startup_ready_provider(normalized):
        # D-1: the launch gate is API-key-or-local-model. An agent
        # subscription is an amplifier, never the thing that satisfies it.
        return old_settings, {}, (
            "Add at least one API key or a local model before finishing. An "
            "agent subscription strengthens Ouroboros but cannot run the "
            "main model on its own."
        )
    return old_settings, normalized, ""


def _persist(request: Request, old_settings: Dict[str, Any], current: Dict[str, Any],
             pending_mode: str, safety_light: bool, preset_applied: bool,
             boundary: CommitBoundary, read_fingerprint: str) -> None:
    """The ONE write, plus the established post-save seams.

    ``boundary`` is committed the instant the bytes land, so the endpoint can
    distinguish "the transaction was refused" from "the transaction landed and
    a later step failed" — two facts that were previously both reported as
    ``saved=False``."""
    from ouroboros.config import apply_settings_to_env, get_runtime_mode
    from ouroboros.gateway.settings import (
        _apply_settings_save_side_effects,
        _start_supervisor_if_needed_for_request,
    )

    to_save = dict(current)
    to_save["OUROBOROS_RUNTIME_MODE"] = pending_mode
    authored = ("OUROBOROS_SAFETY_MODE",) if safety_light else ()
    # Under the seam-wide document lock: the fingerprint precondition protects
    # THIS write from a stale merge, but not a generic save whose (locked)
    # read happened before this write — without the lock that save would land
    # after us and silently erase the whole onboarding transaction. Holding it
    # orders the two: either the save finishes first and the precondition
    # refuses honestly, or this write finishes first and the save's read sees
    # the onboarded document.
    with settings_document_mutation():
        _owner_write_settings(
            to_save,
            authored_keys=authored,
            allow_safety_lowering=safety_light,
            precondition=_write_precondition(preset_applied, safety_light, read_fingerprint),
            boundary=boundary,
        )
        # STILL under the lock, symmetric with the generic save's locked body:
        # released after the write alone, a concurrent writer could persist AND
        # project a newer document before this transaction projects its
        # pre-prepared snapshot — stamping stale values back over the
        # environment the newer write just projected.
        # The RUNNING process keeps its boot runtime mode; the owner's next-boot
        # choice lives on disk only (identical to the endpoint this replaces).
        boundary.at("environment projection")
        env_view = dict(current)
        env_view["OUROBOROS_RUNTIME_MODE"] = get_runtime_mode()
        apply_settings_to_env(env_view)
        boundary.at("supervisor start")
        _start_supervisor_if_needed_for_request(request, current)
        boundary.at("hot-reload")
        changed = [
            key for key in current
            if str(current.get(key, "") or "") != str(old_settings.get(key, "") or "")
        ]
        _apply_settings_save_side_effects(request, current, old_settings, changed)


async def api_onboarding_complete(request: Request) -> JSONResponse:
    """POST /api/onboarding/complete — finish onboarding in ONE transaction."""
    from ouroboros.config import get_runtime_mode, normalize_runtime_mode
    from ouroboros.utils import utc_now_iso

    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return unsaved_error("JSON body must be an object.", 400)

    # BEFORE the read, not after: if a write lands between the two, the document
    # this request goes on to derive is NEWER than the fingerprint, the locked
    # precondition sees the mismatch and refuses. Taken the other way round the
    # same interleaving would be invisible, and this is the one ordering that
    # fails closed.
    read_fingerprint = _settings_fingerprint()
    old_settings, current, error = _prepared_settings(body)
    if error:
        return unsaved_error(error, 400)

    subscriptions_connected, skip_presets = parse_subscription_intent(body)
    eligible = preset_eligible(old_settings)
    safety_light = _fresh_settings_file()
    if safety_light:
        # Rev.3-2 parity with the desktop wizard: a genuinely FRESH install
        # authors the new-install ``light`` safety coverage here, because the
        # shared validator must not (web/Docker also reach it through the
        # non-owner generic settings path). Eligibility is "no settings file
        # yet"; the persist seam re-proves it under the lock.
        current["OUROBOROS_SAFETY_MODE"] = "light"
    preset: Optional[SubscriptionInstallPreset] = None
    preset_reason = "not_requested"
    if not eligible:
        preset_reason = "not_install_time"
    elif skip_presets:
        preset_reason = "skipped_by_owner"
    elif subscriptions_connected:
        preset, failure = await resolve_install_preset()
        if failure is not None:
            return failure.as_response()
        preset_reason = "applied"
        # R8 ordering: provider normalization has ALREADY run over `current`;
        # the structured preset keys land on top of it, never through it.
        current.update(preset.settings_keys())

    # The durable completion fact rides in the SAME write, whatever the preset
    # did: a completion that connected nothing must still close the window.
    current[ONBOARDING_COMPLETED_KEY] = utc_now_iso()
    pending_mode = normalize_runtime_mode(current.get("OUROBOROS_RUNTIME_MODE"))
    active_mode = get_runtime_mode()
    boundary = CommitBoundary()
    try:
        await asyncio.to_thread(
            _persist, request, old_settings, current, pending_mode, safety_light,
            preset is not None, boundary, read_fingerprint,
        )
    except Exception as exc:
        if boundary.committed:
            # The transaction LANDED; a post-save step did not. Saying
            # "nothing was saved" here would send the owner back through an
            # onboarding that is already complete (BIBLE P1).
            return post_commit_failure_response(exc, boundary)
        if isinstance(exc, SettingsPreconditionFailed):
            return unsaved_error(str(exc), 409, code="onboarding_state_changed")
        if isinstance(exc, SettingsLockUnavailable):
            # NO `can_skip`. That flag means "there is a different button that
            # WILL work", and the skip is the same request to the same endpoint,
            # which takes the same lock — offering it under contention promises
            # an escape that leads straight back here. `can_skip` belongs to the
            # preset-verification failures, where finishing without agent
            # defaults genuinely bypasses the thing that failed.
            return unsaved_error(str(exc), 503, code="settings_locked")
        if isinstance(exc, PermissionError):
            return unsaved_error(str(exc), 403)
        log.exception("onboarding completion failed")
        return unsaved_error(f"{type(exc).__name__}: {exc}", 500)

    _owner_audit(request, "onboarding_complete", {
        "runtime_mode": pending_mode,
        "preset": preset_reason,
        "preset_harnesses": list(preset.connected) if preset else [],
        "subscriptions_connected": subscriptions_connected,
    })
    payload: Dict[str, Any] = {
        "ok": True,
        "status": "saved",
        "runtime_mode": pending_mode,
        "restart_required": active_mode != pending_mode,
        "preset": {
            "applied": preset is not None,
            "reason": preset_reason,
            "harnesses": list(preset.connected) if preset else [],
            "receipt": dict(preset.receipt) if preset else {},
        },
    }
    return JSONResponse(payload)


__all__ = [
    "PRESET_UNVERIFIED_MESSAGE",
    "PresetFailure",
    "api_onboarding_complete",
    "install_is_unconfigured",
    "preset_eligible",
    "resolve_install_preset",
    "subscription_routable_harnesses",
    "verified_harness_discoveries",
]
