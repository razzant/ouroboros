"""Route capability metadata and the learned parameter/effort policy.

A route's capabilities are discovered, not declared: OpenRouter model metadata
answers what a model accepts, and provider rejections teach the rest — which
optional parameters a route refuses, and the reasoning-effort band it will
actually run. This module owns that knowledge (process caches over a durable
capability-evidence store), the classifier that decides whether a failure was a
parameter rejection at all, and the one-shot payload repair that follows from
it.
"""


from __future__ import annotations

import contextvars
import copy
import logging
import time
from typing import Any, Dict, Optional, Set

from ouroboros.llm_attempt import (
    _is_provider_policy_refusal,
    _is_structured_context_overflow_exception,
)
from ouroboros.provider_models import normalize_model_identity


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")

# Effort clamp/projection disclosure slot: a ContextVar isolates threads AND
# concurrent asyncio tasks (same isolation contract as the reasoning pin note).
_EFFORT_CLAMP_CVAR = contextvars.ContextVar("ouroboros_effort_clamp_note", default=None)


_OPTIONAL_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


# Provider-rejected optional intent may be removed by the one-shot retry ladder.
_OPTIONAL_DROPPABLE_PARAMS = _OPTIONAL_SAMPLING_PARAMS + (
    "response_format", "reasoning_effort", "output_config", "thinking",
)


# Shared by the classifier and floor predicate; bare "required" is too broad.
_MANDATORY_VALUE_MARKERS = ("mandatory", "cannot be disabled", "must be enabled")


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    # v6.57.0: the accepted set is the EFFORT_SCALE SSOT (config.py), so adding a
    # tier (e.g. `max`) happens in one place. Imported lazily to avoid a config
    # import cycle at module load.
    try:
        from ouroboros.config import EFFORT_SCALE as _SCALE
        allowed = set(_SCALE)
    except Exception:
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


class _CapabilityPolicyMixin:
    """Capability metadata, learned parameter rejections and effort bands."""

    # Missing capabilities mean "unknown": keep kwargs instead of stripping them.
    _SUPPORTED_PARAMS_CACHE: Dict[str, set] = {}

    _SUPPORTED_PARAMS_FETCHED: bool = False

    # Did the one-shot /models fetch actually reach OpenRouter (HTTP 200 + parse)?
    # Distinguishes a provider OUTAGE from a route with no metadata, so Capability
    # Evidence can mark STATUS_FAILED (transient) vs STATUS_UNPROBEABLE (v6.33.0 P4).
    _CAPABILITIES_FETCH_OK: bool = False

    # OpenRouter-reported context window per model id (provider_metadata evidence).
    _CONTEXT_LENGTH_CACHE: Dict[str, int] = {}

    _REJECTED_PARAMS_CACHE: Dict[str, Set[str]] = {}

    @classmethod
    def _fetch_openrouter_capabilities(cls) -> None:
        """Populate _SUPPORTED_PARAMS_CACHE once from OpenRouter /models."""
        cls._SUPPORTED_PARAMS_FETCHED = True
        cls._CAPABILITIES_FETCH_OK = False  # set True only on a clean 200 + parse
        try:
            import requests
            # 5s, not 15s: this fetch sits on the synchronous capability-probe path
            # behind the max-context-mode gate (settings save / max toggle); a slow
            # probe must fail-closed quickly, never hang the save (v6.33.0 WS4).
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=5,
            )
            if resp.status_code != 200:
                log.debug(
                    "OpenRouter /models returned %d; supported_parameters cache empty",
                    resp.status_code,
                )
                return
            from ouroboros.provider_models import update_vision_overlay

            for m in resp.json().get("data", []) or []:
                mid = m.get("id") or ""
                sp = m.get("supported_parameters")
                if mid and isinstance(sp, list) and sp:
                    cls._SUPPORTED_PARAMS_CACHE[mid] = set(sp)
                # Context window (provider_metadata Capability Evidence source).
                cl = m.get("context_length")
                if mid and isinstance(cl, (int, float)) and cl > 0:
                    cls._CONTEXT_LENGTH_CACHE[mid] = int(cl)
                # Vision overlay for supports_vision(): authoritative
                # input_modalities from the same /models payload.
                arch = m.get("architecture")
                if mid and isinstance(arch, dict):
                    modalities = arch.get("input_modalities")
                    if isinstance(modalities, list) and modalities:
                        update_vision_overlay(mid, "image" in modalities)
            cls._CAPABILITIES_FETCH_OK = True  # reached the provider and parsed it
        except Exception:
            log.debug("Failed to fetch OpenRouter model capabilities", exc_info=True)

    @classmethod
    def metadata_fetch_attempted_and_failed(cls) -> bool:
        """True when the one-shot OpenRouter /models fetch RAN but did not succeed
        (non-200 or transport error) — i.e. the provider was unreachable, distinct
        from 'not fetched yet'. Capability Evidence uses this to record STATUS_FAILED
        (a transient outage) instead of STATUS_UNPROBEABLE (no metadata source)."""
        return bool(cls._SUPPORTED_PARAMS_FETCHED and not cls._CAPABILITIES_FETCH_OK)

    @classmethod
    def _get_supported_parameters(cls, model_id: str) -> Optional[set]:
        """Return supported parameter names, or None when unknown/no stripping."""
        if not cls._SUPPORTED_PARAMS_FETCHED:
            cls._fetch_openrouter_capabilities()
        return cls._SUPPORTED_PARAMS_CACHE.get(model_id)

    @classmethod
    def openrouter_context_length(cls, model_id: str, *, allow_fetch: bool = True) -> int:
        """OpenRouter-reported context window (tokens) for a model id, else 0.

        provider_metadata Capability Evidence source. A successful /models fetch is
        cached and not repeated; pass allow_fetch=False to read only the existing
        cache (so a hot path never triggers a blocking /models call). On the
        capability-probe path (allow_fetch=True) a RE-fetch is allowed when the
        last fetch FAILED or the requested model is absent from the cache — so a
        transient outage isn't poisoned one-shot and a model picked while the
        provider is unreachable is correctly seen as a transport failure (and
        surfaced as a no-connection error), not silently 'unprobeable' (v6.33.0)."""
        mid = str(model_id or "")
        needs_fetch = (not cls._SUPPORTED_PARAMS_FETCHED) or (
            allow_fetch and (not cls._CAPABILITIES_FETCH_OK or mid not in cls._CONTEXT_LENGTH_CACHE)
        )
        if allow_fetch and needs_fetch:
            cls._fetch_openrouter_capabilities()
        return int(cls._CONTEXT_LENGTH_CACHE.get(mid, 0) or 0)

    @staticmethod
    def _parameter_rejection_error(exc: BaseException) -> bool:
        """Legacy model-global diagnostic; production uses request-wire recovery."""
        text = str(exc or "").lower()
        if not text:
            return False
        # OpenRouter rejects unsupported sampling params (with require_parameters)
        # as "No endpoints found that support the requested parameters: ...".
        # Require an explicit parameter signal so unrelated "no endpoints found"
        # errors (e.g. "...that support tool use") do not falsely match.
        # "reasoning" covers the OpenRouter NESTED carrier (extra_body.reasoning.*):
        # its rejections name "reasoning"/"reasoning.effort", never the top-level
        # "reasoning_effort" spelling (triad r6).
        _param_names = _OPTIONAL_DROPPABLE_PARAMS + ("reasoning",)
        if "no endpoints found" in text and (
            "requested parameter" in text
            or any(param in text for param in _param_names)
        ):
            return True
        if not any(param in text for param in _param_names):
            return False
        return any(
            marker in text
            for marker in (
                "unsupported",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "deprecated",
                "invalid parameter",
                "extraneous",
                "not permitted",
                # VALUE-rejection families (v6.73.2). Mandatory-enable family —
                # the parameter is supported but its DISABLED/bottom value is
                # forbidden (e.g. Gemini "Reasoning is mandatory for this
                # endpoint and cannot be disabled"); routed to the effort-FLOOR
                # branch by _mandatory_value_rejection (which consumes the SAME
                # _MANDATORY_VALUE_MARKERS constant), never to the drop path
                # for effort carriers.
                *_MANDATORY_VALUE_MARKERS,
                # Range/value family — the VALUE is out of the accepted range
                # (e.g. "temperature must be between 0 and 2"). These take the
                # existing drop path: the param is an optional hint and removing
                # it is the correct degradation.
                "must be between",
                "out of range",
                "invalid value",
            )
        )

    @staticmethod
    def _mandatory_value_rejection(exc: BaseException) -> bool:
        """True when a provider rejected a parameter VALUE as 'must stay enabled'
        (reasoning cannot be turned off) rather than the parameter being
        unsupported. Only ever consulted AFTER _parameter_rejection_error matched
        (which already required a droppable-param name in the text), so a bare
        marker here cannot fire on unrelated errors. This is the gate that sends
        a bottom-tier effort rejection to the FLOOR branch (raise + learn floor)
        instead of the drop path — value-forbidden and capability-absent need
        OPPOSITE remedies."""
        text = str(exc or "").lower()
        if not text:
            return False
        return any(m in text for m in _MANDATORY_VALUE_MARKERS)

    # Durable twin of _REJECTED_PARAMS_CACHE (v6.69.0): learned rejections survive
    # process/restart boundaries via capability_evidence (same design as the
    # effort-ceiling cache below — normalized-model-identity key, fail-open, and
    # entries expire so a provider re-enabling a parameter heals itself). The
    # process cache re-syncs from the durable store hourly so the 14-day expiry
    # also heals LONG-RUNNING processes, not only restarts.
    _REJECTED_PARAMS_LOADED: Dict[str, float] = {}

    _REJECTED_PARAMS_RELOAD_SEC = 3600.0

    @classmethod
    def _remember_rejected_params(cls, model_id: str, params: Set[str]) -> None:
        if not model_id or not params:
            return
        keys = {model_id, normalize_model_identity(model_id)}
        for key in keys:
            if not key:
                continue
            existing = cls._REJECTED_PARAMS_CACHE.setdefault(key, set())
            existing.update(params)
        try:
            from ouroboros.capability_evidence import record_rejected_params
            from ouroboros.config import DATA_DIR
            durable_key = normalize_model_identity(model_id) or str(model_id)
            record_rejected_params(DATA_DIR, durable_key, params)
        except Exception:
            pass

    @classmethod
    def _known_rejected_params(cls, model_id: str) -> Set[str]:
        if not model_id:
            return set()
        out: Set[str] = set()
        durable_key = normalize_model_identity(model_id) or str(model_id)
        now = time.monotonic()
        loaded_at = cls._REJECTED_PARAMS_LOADED.get(durable_key)
        if durable_key and (
            loaded_at is None or now - loaded_at >= cls._REJECTED_PARAMS_RELOAD_SEC
        ):
            cls._REJECTED_PARAMS_LOADED[durable_key] = now
            try:
                from ouroboros.capability_evidence import get_rejected_params
                from ouroboros.config import DATA_DIR
                cls._REJECTED_PARAMS_CACHE[durable_key] = set(get_rejected_params(DATA_DIR, durable_key))
            except Exception:
                pass
        for key in {model_id, normalize_model_identity(model_id)}:
            out.update(cls._REJECTED_PARAMS_CACHE.get(key, set()))
        return out

    # Sentinel for the OpenRouter NESTED effort carrier (extra_body.reasoning) in the
    # rejected-params cache — top-level pops cannot reach it (triad r6).
    _NESTED_REASONING_PARAM = "extra_body.reasoning"

    @classmethod
    def _apply_rejected_param_cache(cls, payload: Dict[str, Any], model_id: str) -> None:
        for param in cls._known_rejected_params(model_id):
            if param == cls._NESTED_REASONING_PARAM:
                eb = payload.get("extra_body")
                if isinstance(eb, dict):
                    eb.pop("reasoning", None)
                continue
            payload.pop(param, None)

    # v6.57.0 — learned reasoning-effort ceilings (Q7). In-process cache is the hot
    # path; a durable copy in capability_evidence.json (effort_ceilings namespace,
    # DATA_DIR-scoped) survives restart. Key = normalized model identity. Fail-open.
    _EFFORT_CEILING_CACHE: Dict[str, str] = {}

    _EFFORT_CEILING_LOADED: Set[str] = set()

    # v6.73.2 — learned reasoning-effort FLOORS: the value-too-low mirror of the
    # ceilings, for endpoints where reasoning is MANDATORY and "none"/"minimal"
    # 400s ("Reasoning is mandatory ... cannot be disabled"). Unlike the sticky
    # ceilings, floors EXPIRE in the durable store (provider policy changes), so
    # the process cache re-syncs hourly like _REJECTED_PARAMS_CACHE — a
    # long-running process heals the same way a restart does.
    _EFFORT_FLOOR_CACHE: Dict[str, str] = {}

    _EFFORT_FLOOR_LOADED: Dict[str, float] = {}

    _EFFORT_FLOOR_RELOAD_SEC = 3600.0

    @classmethod
    def _effort_floor_for(cls, model_id: str) -> str:
        key = normalize_model_identity(model_id) or str(model_id or "")
        if not key:
            return ""
        now = time.monotonic()
        loaded_at = cls._EFFORT_FLOOR_LOADED.get(key)
        if loaded_at is None or now - loaded_at >= cls._EFFORT_FLOOR_RELOAD_SEC:
            cls._EFFORT_FLOOR_LOADED[key] = now
            try:
                from ouroboros.capability_evidence import get_effort_floor
                from ouroboros.config import DATA_DIR
                # Replace (not union): the durable reader applies the 14-day
                # expiry, so replacing lets an expired floor actually evict.
                cls._EFFORT_FLOOR_CACHE[key] = get_effort_floor(DATA_DIR, key)
            except Exception:
                pass
        return cls._EFFORT_FLOOR_CACHE.get(key, "")

    @classmethod
    def _record_effort_floor(cls, model_id: str, floor: str) -> None:
        """A provider rejected a bottom-tier effort as 'reasoning is mandatory' →
        learn the route's minimum. In-process + durable (14-day expiry there), so
        subsequent calls clamp UP immediately. Higher floor wins in-process too."""
        from ouroboros.config import effort_rank
        key = normalize_model_identity(model_id) or str(model_id or "")
        value = str(floor or "").strip().lower()
        if not key or not value:
            return
        prev = cls._EFFORT_FLOOR_CACHE.get(key, "")
        if not prev or effort_rank(value) > effort_rank(prev):
            cls._EFFORT_FLOOR_CACHE[key] = value
        cls._EFFORT_FLOOR_LOADED[key] = time.monotonic()
        try:
            from ouroboros.capability_evidence import record_effort_floor
            from ouroboros.config import DATA_DIR
            record_effort_floor(DATA_DIR, key, value)
        except Exception:
            pass

    @classmethod
    def _effort_ceiling_for(cls, model_id: str) -> str:
        key = normalize_model_identity(model_id) or str(model_id or "")
        if not key:
            return ""
        if key in cls._EFFORT_CEILING_CACHE:
            return cls._EFFORT_CEILING_CACHE[key]
        if key in cls._EFFORT_CEILING_LOADED:
            return ""
        cls._EFFORT_CEILING_LOADED.add(key)
        try:
            from ouroboros.capability_evidence import get_effort_ceiling
            from ouroboros.config import DATA_DIR
            ceil = get_effort_ceiling(DATA_DIR, key)
            if ceil:
                cls._EFFORT_CEILING_CACHE[key] = ceil
            return ceil
        except Exception:
            return ""

    @classmethod
    def clamp_effort_for_route(cls, model_id: str, effort: str) -> str:
        """Resolve the legacy diagnostic model-global effort band."""
        ceiling = cls._effort_ceiling_for(model_id)
        floor = cls._effort_floor_for(model_id)
        if not ceiling and not floor:
            return effort
        from ouroboros.config import clamp_effort_to, effort_rank
        applied = clamp_effort_to(effort, ceiling) if ceiling else effort
        if floor and 0 <= effort_rank(applied) < effort_rank(floor):
            applied = floor
        return applied

    def _clamp_effort_for_model(self, model_id: str, effort: str) -> str:
        """Legacy clamp plus diagnostic disclosure; production dispatch bypasses it."""
        _EFFORT_CLAMP_CVAR.set(None)
        from ouroboros.config import effort_rank
        applied = self.clamp_effort_for_route(model_id, effort)
        if applied != effort:
            _EFFORT_CLAMP_CVAR.set({
                "requested": effort,
                "applied": applied,
                "reason": (
                    "learned_floor"
                    if effort_rank(applied) > effort_rank(effort)
                    else "learned_ceiling"
                ),
                "model": str(model_id or ""),
            })
        return applied

    def _pop_thread_disclosure(self, slot: str) -> Optional[Dict[str, Any]]:
        """Take and clear the disclosure staged in thread-local ``slot`` for THIS
        thread's call; these slots stage before or at send (pin and effort
        notes: ContextVar)."""
        tls = getattr(self, slot, None)
        pending = getattr(tls, "pending", None) if tls is not None else None
        if tls is not None:
            tls.pending = None
        return pending if isinstance(pending, dict) else None

    def _pop_effort_clamp_disclosure(self) -> Optional[Dict[str, Any]]:
        """The pending clamp record for THIS call's context (thread or asyncio task)."""
        pending = _EFFORT_CLAMP_CVAR.get()
        _EFFORT_CLAMP_CVAR.set(None)
        return pending if isinstance(pending, dict) else None

    @classmethod
    def _record_effort_ceiling(cls, model_id: str, current_effort: str) -> None:
        """Record a legacy model-global ceiling for diagnostics."""
        from ouroboros.config import effort_one_step_down, effort_rank
        key = normalize_model_identity(model_id) or str(model_id or "")
        eff = str(current_effort or "").strip().lower()
        if not key or not eff:
            return
        ceiling = effort_one_step_down(eff)
        if effort_rank(ceiling) < effort_rank("low"):
            return
        prev = cls._EFFORT_CEILING_CACHE.get(key)
        if prev and effort_rank(prev) <= effort_rank(ceiling):
            return
        cls._EFFORT_CEILING_CACHE[key] = ceiling
        try:
            from ouroboros.capability_evidence import record_effort_ceiling
            from ouroboros.config import DATA_DIR
            record_effort_ceiling(DATA_DIR, key, ceiling)
        except Exception:
            pass

    @staticmethod
    def _payload_effort(payload: Dict[str, Any]) -> str:
        """Read the effort carried by a request payload across provider shapes."""
        eff = str(payload.get("reasoning_effort") or "").strip().lower()
        if eff:
            return eff
        oc = payload.get("output_config")
        if isinstance(oc, dict) and str(oc.get("effort") or "").strip():
            return str(oc.get("effort")).strip().lower()
        eb = payload.get("extra_body")
        if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
            return str(eb["reasoning"].get("effort") or "").strip().lower()
        return ""

    @staticmethod
    def _set_payload_effort(payload: Dict[str, Any], effort: str) -> None:
        """Write effort into each carrier already present in the payload."""
        if "reasoning_effort" in payload:
            payload["reasoning_effort"] = effort
        oc = payload.get("output_config")
        if isinstance(oc, dict) and "effort" in oc:
            oc["effort"] = effort
        eb = payload.get("extra_body")
        if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
            eb["reasoning"]["effort"] = effort

    def _retry_without_optional_sampling(
        self,
        payload: Dict[str, Any],
        model_id: str,
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        cls = type(self)
        if _is_structured_context_overflow_exception(exc) or _is_provider_policy_refusal(exc):
            return None
        if not cls._parameter_rejection_error(exc):
            return None
        _err_text = str(exc or "").lower()
        _effort_implicated = any(
            k in _err_text for k in ("reasoning_effort", "output_config", "thinking", "reasoning", "effort")
        )
        if cls._mandatory_value_rejection(exc):
            requested = cls._payload_effort(payload)
            if not _effort_implicated or requested not in ("none", "minimal"):
                return None
            cls._record_effort_floor(model_id, "low")
            applied = self._clamp_effort_for_model(model_id, requested)
            if applied == requested:
                return None
            retry_payload = copy.deepcopy(payload)
            cls._set_payload_effort(retry_payload, applied)
            log.warning(
                "Retrying %s with reasoning effort raised to learned floor %r "
                "(provider requires reasoning enabled)",
                model_id or "(unknown model)", applied,
            )
            return retry_payload
        present = {param for param in _OPTIONAL_DROPPABLE_PARAMS if param in payload}
        _err_compact = _err_text.replace(".", "_")
        _named = {param for param in present if param in _err_text or param in _err_compact}
        _eb = payload.get("extra_body")
        _nested_reasoning = isinstance(_eb, dict) and isinstance(_eb.get("reasoning"), dict)
        if _nested_reasoning and _effort_implicated:
            _named.add(cls._NESTED_REASONING_PARAM)
            present.add(cls._NESTED_REASONING_PARAM)
        if _named:
            if _named & {"thinking", "output_config"}:
                _named |= {"thinking", "output_config"} & present
            present = _named
        if not present:
            return None
        if (
            present & {"reasoning_effort", "output_config", "thinking", cls._NESTED_REASONING_PARAM}
            and _effort_implicated
        ):
            cls._record_effort_ceiling(model_id, cls._payload_effort(payload))
        cls._remember_rejected_params(model_id, present)
        retry_payload = copy.deepcopy(payload)
        for param in present:
            if param == cls._NESTED_REASONING_PARAM:
                _retry_eb = retry_payload.get("extra_body")
                if isinstance(_retry_eb, dict):
                    _retry_eb.pop("reasoning", None)
                continue
            retry_payload.pop(param, None)
        log.warning(
            "Retrying %s without optional request parameter(s): %s",
            model_id or "(unknown model)",
            ", ".join(sorted(present)),
        )
        return retry_payload
