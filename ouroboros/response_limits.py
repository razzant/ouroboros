"""Independent, exact-route maximum-response evidence and allowance clamping.

Shares capability_evidence's store and owner-ack boundary, not window timestamps.
Metadata adapters publish only fields meaning a maximum; no error-number learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResponseLimit:
    max_output_tokens: int = 0
    status: str = "unprobeable"
    source: str = "none"
    observed_at: str = ""
    stale: bool = False
    route_fp: str = ""

    def ceiling(self, requested: int) -> int:
        return min(requested, self.max_output_tokens) if self.max_output_tokens > 0 and not self.stale else requested


# Dispatch reaches one fixed endpoint for these providers (llm_routing); Claudexor's
# URL is a locality fact. A configured base URL there identifies nothing.
SINGLE_ENDPOINT_PROVIDERS = frozenset({"openai", "anthropic", "deepseek", "openrouter", "claudexor"})


def positive_limit(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _response_fingerprint(*, provider, model, base_url="", options=None):
    from ouroboros.capability_evidence import route_fingerprint

    if provider == "local":
        # llama.cpp's wire name is always local-model; bind the artifact and loopback endpoint
        # the live server was launched with (saved settings stay pending until Stop/Start),
        # and the saved ones only while no server runs, never the unused remote slot label.
        # One key for both (``local_artifact_key``), so a maximum set while stopped survives Start.
        from ouroboros.local_model import get_manager, local_artifact_key
        serving = getattr(get_manager(), "serving_artifact", dict)()
        if serving:
            model, port = local_artifact_key(model_path=serving["model_path"]), serving["port"]
        else:
            from ouroboros.config import runtime_settings
            settings = runtime_settings()
            model = local_artifact_key(settings.get("LOCAL_MODEL_SOURCE"), settings.get("LOCAL_MODEL_FILENAME"))
            port = settings.get("LOCAL_MODEL_PORT") or 8766
        base_url = f"http://127.0.0.1:{int(port)}/v1"
    if provider in SINGLE_ENDPOINT_PROVIDERS:
        base_url = ""  # Never reaches a send; Claudexor's account options bind its route.
    # Route spellings (``kind[@name]::`` selection, ``kind/`` usage label) identify the same
    # endpoint/model: a connection's key changes no maximum, its endpoint stays in base_url.
    # Never fold a vendor/model OpenRouter id into a direct route.
    from ouroboros.provider_models import parse_model_route

    route = parse_model_route(model)
    if route.valid and route.kind == provider and provider not in {"local", "claudexor"}:
        model = route.model
    elif provider not in {"openrouter", "local", "claudexor"} and model.startswith(provider + "/"):
        model = model[len(provider) + 1:]
    if provider not in {"openrouter", "local", "claudexor"}:
        if provider == "anthropic":
            # Direct dispatch sends the canonical id (llm_routing); a dotted alias names the same model.
            from ouroboros.provider_models import normalize_anthropic_model_id
            model = normalize_anthropic_model_id(model)
        model = provider + "::" + model
    return route_fingerprint(provider=provider, model=model, base_url=base_url, options=options)


def metadata_limit(*, provider, model, base_url="", options=None, maximum=0,
                   source="provider_metadata", observed_at="") -> ResponseLimit:
    from ouroboros.utils import utc_now_iso

    fp = _response_fingerprint(provider=provider, model=model, base_url=base_url, options=options)
    value = positive_limit(maximum)
    return ResponseLimit(value, "confirmed" if value else "unprobeable", source,
                         observed_at or utc_now_iso(), route_fp=fp)


def record_metadata_limit(drive_root, **observation) -> ResponseLimit:
    from ouroboros.capability_evidence import _store_evidence

    record = metadata_limit(**observation)
    _store_evidence(drive_root, "response_limits", record.route_fp, asdict(record))
    return record


def record_catalog_limits(drive_root, rows, *, provider, base_url, field, source):
    """Persist one catalog observation atomically, without one file write per model.

    Best-effort: a busy or unwritable store (``OSError``, the lock's ``TimeoutError`` included)
    skips this write and keeps its previous records, so the catalog read it observed stands."""
    from ouroboros.capability_evidence import _load, _save, _store_transaction
    from ouroboros.utils import utc_now_iso

    stamp = utc_now_iso()
    try:
        with _store_transaction(drive_root):
            data = _load(drive_root)
            records = data.setdefault("response_limits", {})
            for model, maximum in rows:
                if not model:
                    continue
                fp = _response_fingerprint(provider=provider, model=model, base_url=base_url)
                value = positive_limit(maximum)
                records[fp] = asdict(ResponseLimit(value, "confirmed" if value else "unprobeable",
                                                   source + " " + field, stamp, route_fp=fp))
            _save(drive_root, data)
    except OSError:
        import logging
        logging.getLogger(__name__).debug("catalog response limits not recorded", exc_info=True)


def record_response_ack(drive_root, *, provider, model, base_url="", options=None,
                        max_output_tokens=0, expected_route_fp=""):
    from ouroboros.capability_evidence import (_load, _save, _store_transaction,
                                              _claudexor_metadata_evidence)
    from ouroboros.utils import utc_now_iso

    maximum = positive_limit(max_output_tokens)
    if not isinstance(max_output_tokens, int) or maximum != max_output_tokens or isinstance(max_output_tokens, bool):
        raise ValueError("Maximum response must be Auto (0) or a positive integer")
    fp = _response_fingerprint(provider=provider, model=model, base_url=base_url, options=options)
    if expected_route_fp and fp != expected_route_fp:
        raise ValueError("Response acknowledgement route changed; refresh it")
    if provider == "claudexor":
        # The catalog read refuses a different source, profile or account identity.
        exact = all((options or {}).get(key) for key in ("source_id", "credential_profile_id", "account_fingerprint"))
        ev = _claudexor_metadata_evidence(model, base_url, None, options, evidence_root=drive_root) if exact else None
        if ev is None or ev.stale or not ev.account_fingerprint:
            raise ValueError("Refresh the exact subscription account before acknowledging its maximum")
    record = asdict(ResponseLimit(maximum, "asserted", "owner_ack", utc_now_iso(), route_fp=fp))
    with _store_transaction(drive_root):  # a busy store raises TimeoutError: never reported as saved
        data = _load(drive_root)
        acks = data.setdefault("response_acks", {})
        if maximum:
            acks[fp] = record
        else:
            acks.pop(fp, None)
        if not _save(drive_root, data):
            raise OSError("Could not persist response acknowledgement")
    return record


def resolve_response_limit(drive_root, *, provider, model, base_url="", options=None,
                           allow_fetch=False, api_key=None) -> ResponseLimit:
    from ouroboros.capability_evidence import (_read, _age_seconds,
        _CONFIRMED_TTL_SEC, _FAILED_TTL_SEC, _openai_compatible_metadata_window, _claudexor_metadata_evidence)

    fp = _response_fingerprint(provider=provider, model=model, base_url=base_url, options=options)
    data = _read(drive_root)
    ack = data.get("response_acks", {}).get(fp)
    if ack:
        return ResponseLimit(**ack)
    cached = data.get("response_limits", {}).get(fp)
    if cached:
        ttl = _CONFIRMED_TTL_SEC if cached.get("max_output_tokens") else _FAILED_TTL_SEC
        if _age_seconds(cached.get("observed_at", "")) <= ttl:
            return ResponseLimit(**cached)
    failed_refresh = data.get("response_limit_refresh_failures", {}).get(fp, {})
    if allow_fetch and _age_seconds(failed_refresh.get("observed_at", "")) > _FAILED_TTL_SEC:
        read = "no maximum observed for this route"
        try:
            if provider in {"openai-compatible", "minimax"}:
                _openai_compatible_metadata_window(model, base_url, True, api_key, provider=provider,
                                                  evidence_root=drive_root)
            elif provider == "openrouter":
                from ouroboros.llm import LLMClient
                # The process window cache answers a listed model without reading; an absent or
                # expired output record needs the catalog itself, observed now (one bounded GET),
                # unless the process's read of it just failed: that failure answers this reader too.
                LLMClient._fetch_openrouter_capabilities()
                if LLMClient.metadata_fetch_attempted_and_failed():
                    read = "metadata read failed"
            elif provider == "claudexor":
                ev = _claudexor_metadata_evidence(model, base_url, None, options, evidence_root=drive_root)
                if ev.account_fingerprint:
                    bound = {**(options or {}), "source_id": ev.source_id,
                             "credential_profile_id": ev.credential_profile_id, "account_fingerprint": ev.account_fingerprint}
                    return resolve_response_limit(drive_root, provider=provider, model=model, base_url=base_url, options=bound)
            else:
                read = ""  # No metadata source for this provider: nothing was read.
        except Exception:
            read = "metadata read failed"  # No inferred maximum after a failed metadata read.
        cached = _read(drive_root).get("response_limits", {}).get(fp)
        absent = not cached or (not cached.get("max_output_tokens")
                                and _age_seconds(cached.get("observed_at", "")) > _FAILED_TTL_SEC)
        if read and absent:
            # Like an unprobeable window: the observed absence holds for the failed-evidence TTL,
            # so a route its catalog omits never re-reads that catalog per call. An absence never
            # erases a prior maximum; that one stays and reads stale.
            return record_metadata_limit(drive_root, provider=provider, model=model, base_url=base_url,
                                         options=options, source=f"{provider} metadata: {read}")
        if read and cached and _age_seconds(cached.get("observed_at", "")) > _CONFIRMED_TTL_SEC:
            # A failed refresh (including a catalog omitting this route) must cool down
            # without renewing or erasing the prior maximum's independent observation.
            from ouroboros.capability_evidence import _store_evidence
            from ouroboros.utils import utc_now_iso
            _store_evidence(drive_root, "response_limit_refresh_failures", fp, {"observed_at": utc_now_iso()})
    if cached:
        ttl = _CONFIRMED_TTL_SEC if cached.get("max_output_tokens") else _FAILED_TTL_SEC
        return ResponseLimit(**{**cached, "stale": _age_seconds(cached.get("observed_at", "")) > ttl})
    return ResponseLimit(route_fp=fp)


def response_limit_for_target(target: dict) -> ResponseLimit:
    from ouroboros.capability_evidence import canonical_evidence_root

    return resolve_response_limit(canonical_evidence_root(), provider=target.get("provider", ""),
        model=target.get("requested_model") or target.get("usage_model") or target.get("model") or target.get("resolved_model", ""),
        base_url=target.get("base_url", ""), options=target.get("capability_options"))


def dispatch_target(model: str, settings: Any = None) -> dict:
    """The provider target dispatch resolves for ``model``; resolving it builds no model client."""
    from ouroboros.llm_routing import resolve_target

    return resolve_target(model, settings)


def _dispatch_route(route: dict) -> tuple:
    """Provider, endpoint and key of the send a window route plans for.

    A configured endpoint is the one its send uses; an unset one (a legacy or default
    endpoint) and a single-endpoint provider take the endpoint dispatch resolves.
    """
    from ouroboros.llm_attempt import ConnectionUnavailable

    provider, base_url, key = route["provider"], str(route.get("base_url") or ""), route.get("api_key")
    if provider == "local" or route.get("use_local"):
        return "local", "", key
    try:
        target = dispatch_target(route["model"])
    except (ValueError, ConnectionUnavailable):
        target = {}  # A malformed identity or a refused connection reaches no send; the route keeps its own facts.
    if not base_url or provider in SINGLE_ENDPOINT_PROVIDERS:
        base_url = str(target.get("base_url") or "")
    return provider, base_url, key if key is not None else target.get("api_key")


def probe_response_limit(drive_root, evidence, route: dict) -> ResponseLimit:
    """The output fact beside a window probe, bound to the endpoint its send reaches."""
    provider, base_url, api_key = _dispatch_route(route)
    options = dict(route.get("options") or {})
    if provider == "claudexor" and evidence.account_fingerprint:
        options.update(source_id=evidence.source_id, credential_profile_id=evidence.credential_profile_id,
                       account_fingerprint=evidence.account_fingerprint)
    return resolve_response_limit(drive_root, provider=provider, model=route["model"], base_url=base_url,
        options=options or None, allow_fetch=route.get("allow_fetch", True), api_key=api_key)


def response_limit_preview(drive_root, route: dict, *, account="", settings=None, role="main", window=None) -> dict:
    """Owner projection: stored API evidence and a live local subscription catalog, without writes."""
    from ouroboros import capability_evidence as ce
    from ouroboros.context_fit import main_output_reserve_tokens
    from ouroboros.context_view_facts import window_projection
    from ouroboros.context_window import resolve_window, route_observation_scope
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option

    model = route["model"]
    provider, base_url, _key = _dispatch_route(route)
    options = ce.model_account_options(model, credential_profile_id=account, settings=settings) if provider == "claudexor" else {}
    data = ce._read(drive_root)
    observed = None
    if provider == "claudexor":
        try:
            bound = ce._claudexor_metadata_evidence(model, base_url, None, options, record=False)
            if bound.account_fingerprint:
                options.update(credential_profile_id=bound.credential_profile_id, account_fingerprint=bound.account_fingerprint)
                observed = {**bound.response_limit, "stale": bool(bound.stale)}
        except Exception:
            pass  # Unread account capacity remains unknown, never a fabricated cap.
    window_route = {key: route[key] for key in ("provider", "model", "base_url", "use_local") if key in route}
    window_route.update(provider=provider, options=options)
    evidence = ce.probe(drive_root, **window_route, allow_fetch=False)
    if observed is not None and evidence.response_limit.get("source") != "owner_ack":
        evidence.response_limit = observed  # This read's catalog fact, shown without being stored.
    response = evidence.response_limit
    window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, role, settings=settings)) if window is None else int(window)
    evidence = ce.with_role_window(evidence, window)
    reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local")), evidence=evidence)
    if role in {"light", "deep_review"}:
        from ouroboros.consolidator import LIGHT_ANSWER_CEILING_TOKENS
        from ouroboros.deep_self_review import _DEEP_MAX_OUTPUT_TOKENS
        reserve = LIGHT_ANSWER_CEILING_TOKENS if role == "light" else _DEEP_MAX_OUTPUT_TOKENS
        if route.get("use_local"):
            from ouroboros.llm_local import local_context_limits
            reserve = local_context_limits(reserve)[1]
        reserve = ResponseLimit(**response).ceiling(reserve)
    scope = route_observation_scope({**window_route, "use_local": bool(route.get("use_local"))}, settings)
    record = resolve_window(evidence, reserve, root=drive_root, observation_fp=scope,
                            bound=provider != "claudexor" or bool(account))
    window_route.pop("use_local", None)
    return {"route": {**route, "provider": provider, "base_url": base_url, "options": options or None},
            "response_limit": response, "window": window_projection(record),
            "window_route": {**window_route, "route_fp": evidence.route_fp},
            "window_ack": bool(data.get("owner_acks", {}).get(evidence.route_fp))}


def response_allowance(model: str, requested: int, *, use_local=False, model_role="", credential_profile_id=None,
                       model_route=None, allow_fetch=False) -> int:
    from ouroboros.capability_evidence import canonical_evidence_root, model_account_options

    if use_local is None:
        from ouroboros.provider_models import review_model_uses_local
        use_local = review_model_uses_local(model)
    from ouroboros.llm_attempt import ConnectionUnavailable

    try:
        target = {"provider": "local", "model": model, "base_url": ""} if use_local else dispatch_target(model)
    except (ValueError, ConnectionUnavailable):
        return requested  # A malformed identity or a refused connection has no known maximum.
    provider = target.get("provider", "")
    options = model_account_options(model, role=model_role, credential_profile_id=credential_profile_id,
        model_route=model_route) if provider == "claudexor" else None
    return resolve_response_limit(canonical_evidence_root(), provider=provider, model=model,
        base_url=target.get("base_url", ""), options=options, allow_fetch=allow_fetch,
        api_key=target.get("api_key")).ceiling(requested)
