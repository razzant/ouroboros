"""Atomic onboarding completion — ONE owner-scoped save (D-8).

``POST /api/onboarding/complete`` is the single ordered settings transaction.
Separate provider/runtime-mode saves could leave half-configured installs and
could not atomically include subscription presets or the fresh safety default.

1. re-prove FRESH-INSTALL status server-side — a browser boolean is a request,
   never an authority;
2. validate the wizard payload through the SHARED setup validator; a pending
   subscription draft is completed from exact model transport discovery;
3. apply the ordinary provider normalization before compiling task routes;
4. compile truthful API/local task actors with zero daemon reads, or read ONE
   fresh Claudexor snapshot when subscriptions were declared;
5. validate an owner-edited ``OUROBOROS_SUBAGENTS`` object without replacing
   its rows; a generated catalog nobody marked gains the factory reviewer rows;
6. persist settings + runtime mode + safety default + actor fingerprint/source
   receipt in a single write whose eligibility is re-proved under the settings
   lock;
7. only then start the supervisor.

A daemon that cannot answer at save time is a TYPED failure that persists NOTHING
and keeps the wizard open, with a "finish without agent defaults" escape hatch
(``skipSubscriptionPresets``). A guessed model id is never the fallback: it would
land in the reviewer rows the owner believes are live and fail inside a real review.
"""

from __future__ import annotations

import asyncio
import json
import threading
import concurrent.futures
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.configured_subagents import (
    SOURCE_CONFIGURED,
    SOURCE_ONBOARDING_DEFAULT,
    SUBAGENTS_SETTING,
    ConfiguredSubagents,
    configured_subagents_dict,
    configured_subagents_fingerprint,
    normalize_configured_subagents,
    roster_save_error,
    serialize_configured_subagents,
)

from ouroboros.gateway.owner_settings import (
    CommitBoundary,
    SettingsLockUnavailable,
    SettingsPreconditionFailed,
    _owner_audit,
    _owner_read_settings_raw,
    _owner_write_settings,
    post_commit_failure_response,
    settings_document_digest,
    SettingsDocumentBusy,
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
    REVIEWER_PRESET_HARNESSES,
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
# conditional — an exact required model missing from discovery may need an
# owner edit rather than a retry, unlike daemon_unavailable.
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
    *,
    required_models_for: Optional[set[str]] = None,
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
    required = set(wanted) if required_models_for is None else set(required_models_for)
    for harness in wanted:
        row = rows.get(harness) or {}
        if row.get("models_error") and harness in required:
            return (), PresetFailure(
                "models_unavailable",
                f"Model discovery for {harness} failed: {row.get('models_error')}",
            )
        model_ids = tuple(
            str(model.get("id") or "")
            for model in (row.get("models") or [])
            if isinstance(model, dict) and str(model.get("id") or "")
        )
        if not model_ids and harness in required:
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

    from ouroboros.gateway.models import _subscription_model_catalog

    snapshot = _status_payload(True)
    snapshot["model_catalog"] = _subscription_model_catalog()["items"]
    return snapshot


# The snapshot read runs on its OWN single worker, and one read is shared by
# every completion attempt while it is in flight: a read that outlived its
# bound (issue #464's wedged owned-daemon initialization) is abandoned by the
# awaiting request but not by the executor — a retry JOINS it instead of
# starting another blocked thread. Without this, every timed-out completion
# would consume one worker of the loop's shared default executor for good,
# and once that pool was exhausted every other ``to_thread`` endpoint would
# queue behind the wedge.
_snapshot_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="onboarding-snapshot",
)
_snapshot_inflight: Optional["concurrent.futures.Future[Dict[str, Any]]"] = None
_snapshot_lock = threading.Lock()


def _snapshot_read_future() -> "concurrent.futures.Future[Dict[str, Any]]":
    """The in-flight snapshot read, started if none is running."""
    global _snapshot_inflight
    with _snapshot_lock:
        future = _snapshot_inflight
        if future is None or future.done():
            future = _snapshot_executor.submit(_read_harness_snapshot)
            _snapshot_inflight = future
        return future


async def resolve_install_preset(
    settings: Optional[Dict[str, Any]] = None,
    *,
    subscriptions_connected: bool = True,
    owner_draft: Optional[ConfiguredSubagents] = None,
) -> Tuple[Optional[SubscriptionInstallPreset], Optional[PresetFailure]]:
    """Zero daemon reads for API/local-only, exactly one when subscriptions exist."""
    snapshot: Dict[str, Any] = {}
    discoveries: Sequence[HarnessDiscovery] = ()
    capability: Mapping[str, Any] = {}
    if subscriptions_connected:
        from ouroboros.config import get_onboarding_snapshot_timeout_sec

        timeout_sec = get_onboarding_snapshot_timeout_sec()
        try:
            snapshot = await asyncio.wait_for(
                asyncio.wrap_future(_snapshot_read_future()), timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            # A wedged owned-daemon initialization (a lock held by an earlier
            # read that never returned, issue #464) must not hold the wizard on
            # "Saving..." forever: the read is abandoned to its thread and the
            # completion answers with the same typed, skippable failure the
            # dead-engine case uses.
            log.warning("Claudexor snapshot for onboarding presets did not answer within %ss", timeout_sec)
            return None, PresetFailure("daemon_timeout", f"the Claudexor status read did not answer within {timeout_sec}s")
        except Exception as exc:  # a dead/broken engine is a failure, not a crash
            log.warning("Claudexor snapshot for onboarding presets failed", exc_info=True)
            return None, PresetFailure("daemon_unavailable", f"{type(exc).__name__}: {exc}")
        required_models = set(REVIEWER_PRESET_HARNESSES) if owner_draft is not None else None
        discoveries, failure = verified_harness_discoveries(snapshot, required_models_for=required_models)
        if failure is not None:
            return None, failure
        capability = _harness_capability(snapshot, [d.harness_id for d in discoveries])
    preset = compile_install_preset(
        discoveries,
        settings=settings or {},
        configured_subagents=owner_draft,
        source=SOURCE_CONFIGURED if owner_draft is not None else SOURCE_ONBOARDING_DEFAULT,
        capability=capability,
        model_catalog=snapshot.get("model_catalog") or (),
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
        if settings_document_digest() != read_fingerprint:
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


def _prepared_settings(
    body: Dict[str, Any],
    *,
    base_settings: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """(old_settings, prepared_settings, error) through the SHARED validator."""
    from ouroboros.config import load_settings
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    # Completion keeps the ordinary runtime loader (including its established
    # one-shot compatibility migrations).  The read-only preview passes the
    # existing pure owner-reader so merely rendering an unsaved draft can never
    # persist unrelated compatibility state.
    old_settings = dict(base_settings) if base_settings is not None else load_settings()
    error = roster_save_error(body.get(SUBAGENTS_SETTING), old_settings, body)  # twins: only a roster change is judged
    prepared, error = ({}, error) if error else prepare_onboarding_settings(body, old_settings)
    if error:
        return old_settings, {}, str(error)
    normalized, _changed, _keys = apply_runtime_provider_defaults(prepared)
    connected, skipped = parse_subscription_intent(body)
    if not has_startup_ready_provider(normalized) and not (connected and not skipped):
        return old_settings, {}, (
            "Connect a model-capable subscription, an API key, or a local model before finishing."
        )
    return old_settings, normalized, ""


def _configured_owner_draft(body: Mapping[str, Any]) -> Tuple[Optional[ConfiguredSubagents], str]:
    """Validate an owner-edited canonical object without reading live status."""
    if SUBAGENTS_SETTING not in body:
        return None, ""
    try:
        return normalize_configured_subagents(body.get(SUBAGENTS_SETTING))[0], ""
    except ValueError as exc:
        return None, str(exc)


def with_factory_review_rows(catalog: Mapping[str, Any], doc: Mapping[str, Any]) -> Dict[str, Any]:
    """A generated catalog nobody marked gains the factory reviewer rows (the never-configured
    read's own minting), so the wizard shows the reviewers the install will run."""
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
    from ouroboros.subscription_install_presets import factory_review_rows

    items = list(catalog.get("items") or [])
    if any(item.get("review_eligible") is True for item in items):
        return dict(catalog)
    taken = {item.get("subagent_id") for item in items}
    minted = [row for row in factory_review_rows({**doc, SUBAGENTS_SETTING: json.dumps(catalog)})
              if row.get("subagent_id") not in taken]
    return {**catalog, "items": (items + minted)[:MAX_CONFIGURED_SUBAGENTS]}


def shown_catalog(preset: SubscriptionInstallPreset, doc: Mapping[str, Any],
                  owner_draft: Optional[ConfiguredSubagents], *, allow_empty: bool) -> Dict[str, Any]:
    """The catalog the wizard shows and the completion saves. A posted draft holding reviewers, or
    whose empty pool the owner confirmed, stays as posted: re-marking the reviewers the preview
    showed would mint a twin of each. An unmarked draft keeps the reviewers the preset appends."""
    draft = configured_subagents_dict(owner_draft) if owner_draft is not None else None
    if draft is not None and (allow_empty or any(row.get("review_eligible") is True for row in draft["items"])):
        return draft
    catalog = json.loads(preset.available_subagents)
    return catalog if draft is not None else with_factory_review_rows(catalog, doc)


def preset_saving(preset: SubscriptionInstallPreset, catalog: Mapping[str, Any]) -> SubscriptionInstallPreset:
    """The preset writing ``catalog``; its receipt describes the bytes it saves."""
    config = normalize_configured_subagents(catalog)[0]
    shown = configured_subagents_dict(config)
    receipt = {**preset.receipt, "available_subagents": shown,
               "available_subagents_fingerprint": configured_subagents_fingerprint(config),
               "review_pool": [row["subagent_id"] for row in shown["items"] if row.get("review_eligible") is True]}
    return replace(preset, available_subagents=serialize_configured_subagents(config), receipt=receipt)


def review_rows_on_main(catalog: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Finishing without agent defaults while subscriptions are connected: every marked row runs
    on Main, the count kept, keeping its identity and effort (a session's compound effort becomes the
    row effort; a model-named Main keeps its name's level), with Main's pin and processing."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, resolve_processing_preference
    from ouroboros.provider_models import provider_for_model
    from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, RouteSpec, api_model_named_effort, compound_session_effort
    from ouroboros.subscription_install_presets import _effective_api_models

    main, _light = _effective_api_models(settings)
    if not main:
        raise ValueError("Choose a Main model with access in this setup before using it for reviews.")
    main_named = api_model_named_effort(main)
    profile = str(model_role_option(MODEL_ACCOUNTS_KEY, "main", settings=dict(settings)))
    if profile and provider_for_model(main) != "claudexor":
        raise ValueError("A Main account pin requires a managed model source.")
    processing = resolve_processing_preference("main", settings=dict(settings))

    def on_main(item: Mapping[str, Any]) -> Dict[str, Any]:
        old = item.get("route") or {}
        effort = "" if main_named else (item.get("effort") or (compound_session_effort(RouteSpec(
            ROUTE_KIND_AGENT_SESSION, str(old.get("target_id") or ""))) if old.get("kind") == ROUTE_KIND_AGENT_SESSION else ""))
        kept = ("subagent_id", "recommended_use", "enabled", "review_eligible", "minted_from")
        return {**{key: item[key] for key in kept if key in item},
                "route": {"kind": "api_model", "target_id": main, **({"credential_profile_id": profile} if profile else {})},
                **({"effort": effort} if effort else {}), **({"processing_preference": processing} if processing else {})}

    return {**catalog, "items": [on_main(item) if item.get("review_eligible") is True else item
                                 for item in catalog.get("items") or []]}


async def api_onboarding_subagents_preview(request: Request) -> JSONResponse:
    """Read-only canonical preview for the wizard; never persists or projects env."""
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return unsaved_error("JSON body must be an object.", 400)

    owner_draft, draft_error = _configured_owner_draft(body)
    if draft_error:
        return unsaved_error(
            draft_error, 400, code="invalid_available_subagents",
            diagnostics=[{"code": "invalid_available_subagents", "message": draft_error}],
        )
    _old_settings, current, error = _prepared_settings(
        body,
        base_settings=_owner_read_settings_raw(),
    )
    if error:
        return unsaved_error(
            error, 400, code="invalid_onboarding_settings",
            diagnostics=[{"code": "invalid_onboarding_settings", "message": error}],
        )
    subscriptions_connected, skip_presets = parse_subscription_intent(body)
    preset, failure = await resolve_install_preset(
        current,
        subscriptions_connected=subscriptions_connected and not skip_presets,
        owner_draft=owner_draft,
    )
    if failure is not None:
        return unsaved_error(
            PRESET_UNVERIFIED_MESSAGE, 503, code=failure.code, detail=failure.detail,
            can_skip=True,
            diagnostics=[{"code": failure.code, "message": failure.detail}],
        )
    assert preset is not None
    from ouroboros.gateway.settings import ALLOW_EMPTY_REVIEW_POOL
    try:
        catalog = shown_catalog(preset, current, owner_draft, allow_empty=body.get(ALLOW_EMPTY_REVIEW_POOL) is True)
        if subscriptions_connected and skip_presets:
            catalog = review_rows_on_main(catalog, current)
        available = normalize_configured_subagents(catalog)[0]
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="invalid_onboarding_settings")
    return JSONResponse({
        "ok": True,
        "model_settings": dict(preset.model_settings),
        "available_subagents": configured_subagents_dict(available),
        "source": preset.source,
        "diagnostics": list(preset.diagnostics),
    })


def _persist(request: Request, old_settings: Dict[str, Any], current: Dict[str, Any],
             pending_mode: str, safety_light: bool, install_preset_applied: bool,
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
            precondition=_write_precondition(
                install_preset_applied, safety_light, read_fingerprint,
            ),
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
    from ouroboros.gateway.settings import ALLOW_EMPTY_REVIEW_POOL, REVIEW_LANES_KEY, review_pool_save_judgement

    allow_empty_pool = body.get(ALLOW_EMPTY_REVIEW_POOL) is True
    owner_draft, draft_error = _configured_owner_draft(body)
    if draft_error:
        return unsaved_error(draft_error, 400, code="invalid_available_subagents")

    # BEFORE the read: a write landing between the two makes the derived document NEWER than the
    # fingerprint, so the locked precondition refuses (the one ordering that fails closed). The
    # whole document is written back, so without this a concurrent owner write would be silently
    # undone while the owner is told the save succeeded (the owner endpoints' staleness digest).
    read_fingerprint = settings_document_digest()
    old_settings, current, error = _prepared_settings(body)
    if error:
        return unsaved_error(error, 400)

    subscriptions_connected, skip_presets = parse_subscription_intent(body)
    eligible = preset_eligible(old_settings)
    safety_light = _fresh_settings_file()
    if safety_light:
        # Rev.3-2 desktop-wizard parity: a genuinely FRESH install authors the new-install ``light``
        # safety here (the shared validator must not: web/Docker reach it through the generic path);
        # the persist seam re-proves "no settings file yet" under the lock.
        current["OUROBOROS_SAFETY_MODE"] = "light"
    preset: Optional[SubscriptionInstallPreset] = None
    preset_reason = "not_requested"
    install_preset_applied = False
    if not eligible or skip_presets:
        preset_reason = "skipped_by_owner" if eligible else "not_install_time"
        # Generation is closed or skipped, but an explicit owner draft is still ordinary intent: a
        # recovery/retry must not answer 200 while discarding it. Pure: no daemon read, no marker.
        if owner_draft is not None:
            preset, failure = await resolve_install_preset(
                current, subscriptions_connected=False, owner_draft=owner_draft)
            if failure is not None:
                return failure.as_response()
            preset_reason = "configured_by_owner"
            current.update(preset.settings_keys(include_marker=False))
    else:
        preset, failure = await resolve_install_preset(
            current, subscriptions_connected=subscriptions_connected, owner_draft=owner_draft)
        if failure is not None:
            return failure.as_response()
        preset_reason = "applied"
        install_preset_applied = True
        # R8: provider normalization already ran over `current`; the preset keys land on top of it.
        current.update(preset.settings_keys())
        # Only omitted fields (or an empty Main awaiting its first proposal) are authored here: a
        # visible draft value, even one equal to a shipped default, is explicit owner intent.
        current.update({key: value for key, value in preset.model_settings.items()
                        if key not in body or (key == "OUROBOROS_MODEL" and not body[key])})

    if not has_startup_ready_provider(current):
        return unsaved_error("The connected accounts do not provide a Main model. Connect Codex, an API provider, or a local model.",
                             400, code="model_source_unavailable")
    if install_preset_applied:  # the catalog the preview showed (``shown_catalog``)
        preset = preset_saving(preset, shown_catalog(preset, current, owner_draft, allow_empty=allow_empty_pool))
        current.update(preset.settings_keys(include_marker=False))

    pool_error = review_pool_save_judgement(current.get(SUBAGENTS_SETTING), old_settings, allow_empty=allow_empty_pool)
    if pool_error:
        return unsaved_error(pool_error, 400, code="empty_review_pool")
    if current.get(SUBAGENTS_SETTING):
        current.pop(REVIEW_LANES_KEY, None)

    # The durable completion fact rides in the SAME write, whatever the preset
    # did: a completion that connected nothing must still close the window.
    current[ONBOARDING_COMPLETED_KEY] = utc_now_iso()
    pending_mode = normalize_runtime_mode(current.get("OUROBOROS_RUNTIME_MODE"))
    active_mode = get_runtime_mode()
    boundary = CommitBoundary()

    def _complete_locked(request: Request, _body: Any) -> JSONResponse:
        # ONE synchronous body — the persist, its exception mapping and the
        # success audit — run through the shared bounded writer seam
        # (``settings._run_settings_writer``), so this save inherits the
        # initiating-writer cap (twice the document-lock bound) and the typed
        # 503 ``settings_save_timeout`` with ``saved: null`` when the body
        # wedges in its post-commit effects. A bare ``to_thread`` here used to
        # be the one settings write with no cap at all: the wizard sat on
        # "Saving..." for as long as the hot-reload took. ``SettingsDocumentBusy``
        # is left to the seam, which answers the same 503 ``settings_busy``
        # this endpoint used to map itself (issue #464's second half).
        try:
            _persist(
                request, old_settings, current, pending_mode, safety_light,
                install_preset_applied, boundary, read_fingerprint,
            )
        except SettingsDocumentBusy:
            raise
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

    # Lazy, in the direction ``_persist`` already imports (settings.py imports
    # nothing from this module).
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_complete_locked, request, body)


__all__ = [
    "PRESET_UNVERIFIED_MESSAGE",
    "PresetFailure",
    "api_onboarding_complete",
    "api_onboarding_subagents_preview",
    "install_is_unconfigured",
    "preset_eligible",
    "resolve_install_preset",
    "subscription_routable_harnesses",
    "verified_harness_discoveries",
]
