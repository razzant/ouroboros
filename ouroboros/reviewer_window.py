"""Reviewer context window resolution — the SSOT for every review surface.

A reviewer's window is a FACT about its route, never an assumption: the
historical ``context_window=1_000_000`` literal sized a 200K-window slot as if
it were 1M, which draws a deterministic provider 400 ("prompt is too long") and
loses that reviewer's whole review. Capability Evidence is the only source of
the real number — no static per-model table (v6.33.0).

An UNKNOWN route keeps the full-window assumption, matching the policy
``context_fit`` already applies to the main lane: unknown routes try Max and are
never SILENTLY assumed to be 200K (BIBLE P1). Guessing small is not the safe
direction here, because the guess DECLINES work rather than risking it: a
reviewer sized below the prompt it would have received is dropped before
dispatch, so every cold-evidence install would lose the whole review to an
assumption instead of learning the real number from the first send. The window
is a sizing fact only — it neither grants nor removes a reviewer's authority
(BIBLE P3 "Review evidence and reading diagnostics"). A caller with its own
fail-closed sizing policy takes the whole :class:`ReviewerWindow` from
:func:`resolve_reviewer_window` and applies its floor to
:meth:`~ReviewerWindow.sizing_window`.

Deliberately outside ``ouroboros.tools``: the triad, scope, plan, and deep
self-review surfaces plus the top-level ``deep_self_review`` module all consume
it, and it imports nothing from the tools package.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

REVIEWER_FULL_WINDOW = 1_000_000
# The provider a RETRIEVING (agent_session) reviewer row fingerprints under. Its
# target is a harness route spec, never a provider model id, so it gets its own
# provider name rather than being filed under whatever `provider_for_model`
# guesses for an unrecognised string. Spelled the same as
# `reviewer_slot_config.ROUTE_KIND_SESSION`, imported here so this module keeps
# importing nothing from the tools package.
SESSION_ROUTE_PROVIDER = "agent_session"
# Per-route locks that serialise concurrent resolutions of the SAME route —
# parallel_review runs the triad and the scope slots at once — so they share ONE
# metadata fetch: the second thread enters after the first has written the evidence
# store and reads it back from the cache.
#
# HOW OFTEN a route may be fetched at all is deliberately NOT answered here.
# ``capability_evidence.probe`` owns that through the TTL on the record it stores
# (confirmed 24h / failed 10 min) and returns the cache without touching the network
# inside it. A process-lifetime ``_LAZY_WINDOW_PROBED`` memo used to answer it here
# too, and because the memo never expired while the evidence did, a healthy install
# that stayed up past the 24h TTL read its own reviewers as EXPIRED forever: every
# later resolution took the no-fetch path, so the whole process sized every review
# against an unknown route until it restarted (v6.87.45). The TTL owns the rate
# limit; the locks only share one fetch.
_LAZY_PROBE_REGISTRY_LOCK = threading.Lock()
_LAZY_ROUTE_LOCKS: dict = {}


def _route_probe_lock(route_fp: str) -> threading.Lock:
    """The lock guarding one route's metadata probe (created on first use)."""
    with _LAZY_PROBE_REGISTRY_LOCK:
        lock = _LAZY_ROUTE_LOCKS.get(route_fp)
        if lock is None:
            lock = _LAZY_ROUTE_LOCKS[route_fp] = threading.Lock()
        return lock


def reviewer_window_binding(slot: object) -> dict:
    """Carry a frozen configured/executable row's identity into capacity lookup.

    Empty profile is explicit Auto, not permission to read Main's setting.
    This projects row fields only; it never reloads reviewer configuration.
    """
    value = slot.get if isinstance(slot, dict) else lambda key, default=None: getattr(slot, key, default)
    slot_id = str(value("slot_id", "") or "")
    return {"model_role": f"reviewer:{slot_id}" if slot_id else "",
            "credential_profile_id": value("session_profile", value("profile_id", "")),
            "use_local": value("use_local"), "model_route": value("model_route")}


@dataclass(frozen=True)
class ReviewerWindow:
    """ONE typed answer for a reviewer slot: its window and that number's provenance.

    A bare ``(window, status)`` tuple dropped ``stale`` and the observation time on
    the floor, so a consumer could not tell a live provider reading from a five-day-old
    record kept across an outage. Every field a sizing decision needs — and every
    field its disclosure needs — travels together with the number, or the decision is
    being made on a rumour.

    Field names deliberately MIRROR ``capability_evidence.CapabilityEvidence`` so the
    freshness and threshold predicates that module already owns are REUSED here
    rather than restated."""

    window_tokens: int = 0        # evidence window; 0 == no evidence for this route
    status: str = ""              # confirmed | asserted | unprobeable | failed | ""
    stale: bool = False           # past TTL and not re-verifiable (expired / outage)
    observed_at: str = ""         # ISO timestamp of the observation, "" when unknown
    model: str = ""
    # An owner-asserted sizing number is separate from the sourced window, so a
    # disclosure can say which one a send was sized against.
    asserted_window_tokens: int = 0
    model_route: dict = field(default_factory=dict)

    @property
    def sizing_source(self) -> str:
        return "user_setting" if self.asserted_window_tokens else self.status

    def sizing_window(self, unknown_window: int = REVIEWER_FULL_WINDOW) -> int:
        """Window to SIZE a prompt against — a fit estimate, never an authority.

        A stale number is still the best available estimate of a route's real size,
        so sizing keeps using it and discloses its provenance
        (:attr:`sizing_source`, :attr:`stale`, :attr:`observed_at`)."""
        if self.asserted_window_tokens > 0:
            return int(self.asserted_window_tokens)
        from ouroboros.provider_models import provider_for_model

        if not self.window_tokens and provider_for_model(self.model) == "claudexor":
            # Raw subscriptions have no numeric unknown-window assumption. The
            # assembler keeps its own input budget, never a made-up provider size.
            return 0
        return int(self.window_tokens) if int(self.window_tokens) > 0 else int(unknown_window)


def reviewer_route(model_id: str, *, session: bool = False) -> tuple:
    """``(provider, base_url)`` for a reviewer slot's real route.

    ``session=True`` marks a RETRIEVING row, whose ``model_id`` is an opaque
    Claudexor ``harness[=model]`` spec and not a provider model id at all.
    ``provider_for_model`` cannot resolve such a spec — it answers ``openrouter``
    for anything unrecognised — so a session row fingerprinted through the api
    path would be filed under a provider it never travels, and every capability
    record kept for that route would sit under the same falsehood. The harness IS
    the provider here, exactly as the reviewer-slot SSOT spells it, which is what
    keeps the record honest. The caller passes the ROW's configured kind; nothing
    sniffs the string."""
    from ouroboros.config import runtime_settings
    from ouroboros.provider_models import provider_for_model

    if session:
        return SESSION_ROUTE_PROVIDER, ""
    provider = provider_for_model(str(model_id or ""))
    if provider == "minimax":
        # MiniMax's base url is derived from the region, not a settings key
        # (v6.88.0 direct provider — merged from the public line).
        from ouroboros.provider_models import resolve_minimax_base_url

        return provider, str(
            resolve_minimax_base_url(runtime_settings().get("MINIMAX_REGION") or "") or "")
    if provider == "zai":
        # Z.ai's base url is selected by the plan, the same way MiniMax's is by region.
        from ouroboros.provider_models import resolve_zai_base_url

        return provider, resolve_zai_base_url(runtime_settings().get("ZAI_PLAN") or "")
    settings_key = {
        "openai": "OPENAI_BASE_URL",
        "openai-compatible": "OPENAI_COMPATIBLE_BASE_URL",
        "cloudru": "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
        "gigachat": "GIGACHAT_BASE_URL",
    }.get(provider, "")
    base_url = str(runtime_settings().get(settings_key) or "") if settings_key else ""
    return provider, base_url


def resolve_reviewer_window(
    model_id: str,
    *,
    use_local: Optional[bool] = None,
    session: bool = False,
    model_role: str = "",
    credential_profile_id: Optional[str] = None,
    model_route: Optional[dict] = None,
) -> ReviewerWindow:
    """The reviewer's :class:`ReviewerWindow` from Capability Evidence.

    Never fabricates a window: a route with no evidence comes back with
    ``window_tokens=0`` and the CALLER applies its own fail-closed SIZING
    policy. The probe is metadata-only — never generative, never
    a paid call — so an env-only pin can become known through a path it would
    otherwise never reach, and it stays re-confirmable for as long as the process
    lives: ``probe`` serves the cache untouched inside its TTL and only reaches the
    network once that TTL is spent, which is the whole of the rate limit this
    surface needs (see the module-level note on ``_LAZY_ROUTE_LOCKS``).

    ``use_local=None`` (the default) derives the EFFECTIVE route from
    ``provider_models.review_model_uses_local`` — the same predicate every review
    dispatch path hands to ``ReviewSlot``. The window is a fact about the route the
    request will actually travel, so the default must consult the routing authority
    rather than assume the provider spelled in the model text: with a False default,
    a local-only install (``USE_LOCAL_MAIN`` and no cloud credentials) fingerprinted
    the provider route, ignored the local model's real context length, and sized
    triad/plan prompts against the unknown-route 1M assumption. Callers pass an
    explicit bool only to pin a route the predicate cannot see.

    Concurrent resolutions of the SAME route serialise on that route's lock and the
    second one reads the evidence the first stored, instead of duplicating its
    fetch."""
    model = str(model_id or "")
    try:
        from ouroboros.capability_evidence import model_account_options, probe, route_fingerprint
        from ouroboros.config import DATA_DIR
        from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option
        from ouroboros.provider_models import review_model_uses_local

        # A retrieving row never travels the local lane: its target is a harness
        # route, so the local-model predicate has nothing to say about it.
        if use_local is None:
            use_local = False if session else review_model_uses_local(model)
        provider, base_url = reviewer_route(model, session=session)
        effective_provider = "local" if use_local else provider
        options = model_account_options(
            model, role=model_role, credential_profile_id=credential_profile_id,
            model_route=model_route,
        ) if effective_provider == "claudexor" else None
        asserted_window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, model_role))
        route_fp = route_fingerprint(
            provider=effective_provider, base_url=base_url, model=model,
            options=options,
        )
        # MiniMax's catalog endpoint needs the key even for metadata (their
        # /models is authenticated); every other provider probes keyless.
        _probe_api_key = None
        if effective_provider == "minimax":
            from ouroboros.config import runtime_settings as _ls
            _probe_api_key = str(_ls().get("MINIMAX_API_KEY") or "") or None
        with _route_probe_lock(route_fp):
            ev = probe(
                DATA_DIR,
                provider=effective_provider,
                model=model,
                base_url=base_url,
                use_local=use_local,
                allow_fetch=True,
                api_key=_probe_api_key,
                options=options,
            )
        window = int(getattr(ev, "window_tokens", 0) or 0)
        return ReviewerWindow(
            window_tokens=window,
            status=str(getattr(ev, "status", "") or ""),
            stale=bool(getattr(ev, "stale", False)),
            observed_at=str(getattr(ev, "ts", "") or ""),
            model=model,
            asserted_window_tokens=asserted_window,
            model_route={
                "source": str(getattr(ev, "source_id", "") or ""),
                "model": model.partition("=")[2],
                "credentialProfileId": str(getattr(ev, "credential_profile_id", "") or ""),
                "accountFingerprint": str(getattr(ev, "account_fingerprint", "") or ""),
            } if effective_provider == "claudexor" else {},
        )
    except Exception:
        logger.debug("reviewer window evidence probe failed", exc_info=True)
    return ReviewerWindow(model=model)


def reviewer_context_window(
    model_id: str,
    *,
    unknown_window: int = REVIEWER_FULL_WINDOW,
    use_local: Optional[bool] = None,
    model_role: str = "",
    credential_profile_id: Optional[str] = None,
    model_route: Optional[dict] = None,
) -> int:
    """Reviewer window from Capability Evidence, or ``unknown_window`` when absent.

    The default keeps an unevidenced route at the full window (see the module
    docstring: sizing a review pack DOWN on a guess declines the review outright,
    which is why the main lane's unknown-route policy is the same). A caller that
    must fail closed passes its own sub-floor explicitly. ``use_local=None``
    derives the effective route exactly as :func:`resolve_reviewer_window` does.
    The NUMBER only — a caller that also discloses its provenance takes
    :func:`resolve_reviewer_window` whole."""
    return resolve_reviewer_window(
        model_id, use_local=use_local, model_role=model_role,
        credential_profile_id=credential_profile_id, model_route=model_route,
    ).sizing_window(unknown_window)


def window_scaled_reserves(
    window: int,
    *,
    output_reserve: int,
    tokenizer_margin: int,
    min_output_reserve: int = 8_192,
) -> tuple:
    """``(output_reserve, tokenizer_margin)`` scaled to a sub-1M reviewer window.

    The absolute 1M-calibrated reserves would swallow a small window whole
    (a 131K route => input limit 0, bricking the slot — Provider Independence),
    so sub-floor windows reserve a quarter for output and an eighth for the
    tokenizer margin instead. >=1M windows keep the absolute reserves."""
    if int(window) <= 0 or int(window) >= REVIEWER_FULL_WINDOW:
        return int(output_reserve), int(tokenizer_margin)
    return (
        min(int(output_reserve), max(int(min_output_reserve), int(window) // 4)),
        min(int(tokenizer_margin), int(window) // 8),
    )
