"""Capability Evidence — sourced, auditable knowledge of a route's context window.

Replaces the stale static per-model window table (deleted in v6.33.0). Every
window claim is EVIDENCE with a status and a source, scoped to a route
fingerprint (provider + base_url + model + headers/beta + relevant options):

  status:
    confirmed   — a trustworthy live/local source reported it
                  (source = provider_metadata | local_health)
    asserted    — the owner acknowledged it for an EXACT route fingerprint
                  (source = owner_ack); auditable, invalidated on ANY route change
    unprobeable — no metadata source and no owner-ack (e.g. OpenAI/Anthropic
                  direct, whose 1M is an undiscoverable per-request beta header)
    failed      — a probe was attempted and errored (transient; retried later)

``unknown`` (unprobeable | failed | no record) => FAIL-CLOSED for any >=1M gate.

Probes are opportunistic and cached (24h for confirmed, 10 min for failed). Gate
readers pass ``allow_fetch=False`` so the hot path never blocks on a network
call. A provider outage marks evidence stale; it never erases a prior confirmed/
asserted record. The owner-ack is route-fingerprinted and NEVER a repo-wide
"trust this model" flag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

# Serialises the load->mutate->save of the two owner-only writers (probe cache +
# owner-ack) within the process so neither loses the other's update; atomic_write_json
# additionally prevents torn/corrupt files across processes (durable-state SSOT).
_STORE_LOCK = threading.RLock()

STATUS_CONFIRMED = "confirmed"
STATUS_ASSERTED = "asserted"
STATUS_UNPROBEABLE = "unprobeable"
STATUS_FAILED = "failed"

SOURCE_PROVIDER_METADATA = "provider_metadata"
SOURCE_LOCAL_HEALTH = "local_health"
SOURCE_OWNER_ACK = "owner_ack"
SOURCE_GENERATIVE_PROBE = "generative_probe"
SOURCE_NONE = "none"

# Context-overflow rejections carry the model's limit in the human-readable message
# (NOT the `code` field, which varies: context_length_exceeded / invalid_request_error /
# 400 / 1261). Parse the number from the text across the known provider phrasings.
_CTX_LIMIT_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"maximum context length is\s*([0-9][0-9,]*)", re.I),
    re.compile(r"context length is\s*([0-9][0-9,]*)", re.I),
    re.compile(r"longer than the model's context length\s*\(?\s*([0-9][0-9,]*)", re.I),
    re.compile(r"maximum allowed length\s*\(?\s*([0-9][0-9,]*)", re.I),
    re.compile(r"context (?:window|length)\s*(?:of|is)?\s*([0-9][0-9,]*)\s*tokens", re.I),
    re.compile(r"maximum (?:input |prompt )?(?:length|tokens?)\s*(?:is|of)?\s*([0-9][0-9,]*)", re.I),
)


def _parse_ctx_limit_number(text: str) -> int:
    """Extract the model's context-token limit from an overflow error message, or 0."""
    for pat in _CTX_LIMIT_PATTERNS:
        m = pat.search(str(text or ""))
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except (ValueError, TypeError):
                continue
    return 0


def classify_generative_probe_response(
    status_code: Optional[int],
    body_text: str,
    *,
    canaries: Optional[List[str]] = None,
    echoed_text: str = "",
    usage_prompt_tokens: int = 0,
    sent_token_estimate: int = 0,
) -> Tuple[int, str, str]:
    """Pure (no-network) classifier for a generative context-window probe response.

    Free-only policy (owner Q1): confirm a window ONLY from a FREE pre-inference
    reject that states the limit; a genuine 200 (the model ACCEPTED — and would bill —
    the oversized input) never auto-confirms >=1M, it routes to owner-ack.
    Returns ``(window_tokens, status, detail)``.
    """
    # 4xx: pre-inference reject (free). Parse the limit NUMBER from the text.
    if isinstance(status_code, int) and 400 <= status_code < 500:
        n = _parse_ctx_limit_number(body_text)
        if n > 0:
            return n, STATUS_CONFIRMED, f"generative overflow reject: max {n} tokens"
        # e.g. Zhipu code 1261 (no number) or a 413 size reject -> cannot size it.
        return 0, STATUS_UNPROBEABLE, "overflow reject without a parseable limit; owner-ack required"
    # 200: the oversized input was ACCEPTED. Under free-only this is a possibly-PAID
    # accept and must NOT confirm >=1M (owner chose owner-ack). Truncation guard is
    # recorded for forensics but does not change the owner-ack outcome.
    if status_code == 200:
        cs = canaries or []
        echoed_ok = bool(cs) and all(c in (echoed_text or "") for c in cs)
        usage_ok = sent_token_estimate > 0 and usage_prompt_tokens >= int(0.95 * sent_token_estimate)
        detail = "oversized input accepted (200); free-only policy -> owner-ack"
        if not (echoed_ok and usage_ok):
            detail = "oversized input 200 but truncation suspected (canaries/usage); owner-ack"
        return 0, STATUS_UNPROBEABLE, detail
    # transport / 5xx / timeout / unknown -> transient failure (short TTL, retried).
    return 0, STATUS_FAILED, f"generative probe transport/server error (status={status_code})"

_KNOWN_STATUS = {STATUS_CONFIRMED, STATUS_ASSERTED}

_CONFIRMED_TTL_SEC = 24 * 3600
_FAILED_TTL_SEC = 10 * 60

ONE_MILLION = 1_000_000


@dataclass
class CapabilityEvidence:
    window_tokens: int
    status: str
    source: str
    route_fp: str
    model: str = ""
    provider: str = ""
    ts: str = ""
    detail: str = ""
    stale: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "window_tokens": int(self.window_tokens or 0),
            "status": self.status,
            "source": self.source,
            "route_fp": self.route_fp,
            "model": self.model,
            "provider": self.provider,
            "ts": self.ts,
            "detail": self.detail,
            "stale": bool(self.stale),
        }


def confirms_at_least(evidence: Optional[CapabilityEvidence], threshold: int = ONE_MILLION) -> bool:
    """True only when KNOWN (confirmed/asserted) evidence meets the threshold.

    unprobeable / failed / None / below-threshold all fail closed."""
    if evidence is None:
        return False
    return evidence.status in _KNOWN_STATUS and int(evidence.window_tokens or 0) >= int(threshold)


# --- Route fingerprint ---------------------------------------------------------

def _canonical_headers(headers: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(headers, dict):
        return ()
    return tuple(sorted((str(k).lower(), str(v)) for k, v in headers.items()))


def _canonical_options(options: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(options, dict):
        return ()
    # Only options that can change the effective window/route are fingerprinted.
    relevant = ("beta", "anthropic_beta", "context_1m", "max_tokens", "tenant")
    return tuple(sorted((k, str(options[k])) for k in relevant if k in options))


def route_fingerprint(
    *,
    provider: str,
    base_url: str = "",
    model: str = "",
    headers: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable, NON-generic fingerprint of an exact route. Any change to provider,
    base_url, model, beta/headers, or relevant options yields a new fingerprint —
    so an owner-ack can never silently outlive the configuration it approved."""
    payload = json.dumps({
        "provider": str(provider or "").strip().lower(),
        "base_url": str(base_url or "").strip().rstrip("/").lower(),
        "model": str(model or "").strip(),
        "headers": _canonical_headers(headers),
        "options": _canonical_options(options),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# --- Persistence ---------------------------------------------------------------

def _store_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "capability_evidence.json"


def _load(drive_root: Any) -> Dict[str, Any]:
    data = read_json_dict(_store_path(drive_root))
    if isinstance(data, dict):
        data.setdefault("probes", {})
        data.setdefault("owner_acks", {})
        data.setdefault("effort_ceilings", {})
        return data
    return {"probes": {}, "owner_acks": {}, "effort_ceilings": {}}


def _save(drive_root: Any, data: Dict[str, Any]) -> None:
    path = _store_path(drive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, data)  # atomic rename — never a torn/partial file
    except OSError:
        pass


def _store_evidence(drive_root: Any, kind: str, fp: str, value: Dict[str, Any]) -> None:
    """Locked, atomic read-modify-write of one evidence entry (``probes`` or
    ``owner_acks``). The lock re-reads the CURRENT file inside the critical section
    so a concurrent owner-ack and probe never clobber each other; the network probe
    itself runs OUTSIDE this lock. Never raises."""
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            data.setdefault(kind, {})[fp] = value
            _save(drive_root, data)
    except Exception:
        log.debug("capability evidence store failed (%s)", kind, exc_info=True)


def _age_seconds(ts: str) -> float:
    parsed = parse_deadline_ts(ts)
    if parsed is None:
        return float("inf")
    return max(0.0, (utc_now() - parsed).total_seconds())


# --- Learned reasoning-effort ceilings (v6.57.0) -------------------------------
# A COMPLETELY SEPARATE namespace ("effort_ceilings") from the context-window
# evidence (probes/owner_acks) — it shares only the store file and the lock.
# It NEVER touches window records, so the BIBLE P3 ≥1M scope-review floor
# evidence path is untouched. Value shape:
#   {"ceiling": "<effort>", "observed_at": iso, "reason": "provider_rejected"}
# KEYING (deliberate, r4 disclosure): the key is the NORMALIZED MODEL IDENTITY
# (llm.normalize_model_identity — provider-scoped model id), NOT the full route
# fingerprint the window evidence uses. Effort-level support is treated as a
# MODEL property: a ceiling learned on one base_url applies to the model on all
# routes. Coarser than per-route, and self-healing — the floor in llm.py keeps
# a bad endpoint from poisoning below "low", and clamps are disclosed per call.
# The ceiling is the highest effort a route ACCEPTED after a provider rejected a
# higher one (learned by the reject-and-step-down walk in llm.py). Fail-open:
# any error → no ceiling (send the requested effort). Owner-configured efforts are
# still honored UP TO the learned real ceiling; clamping is disclosed in usage.

def record_effort_ceiling(drive_root: Any, fingerprint: str, ceiling: str) -> None:
    """Persist the learned reasoning-effort ceiling. The key is the normalized
    model identity (see the namespace note above), not a full route fingerprint.
    Best-effort, never raises; a lower ceiling always wins (a route never silently
    regains an effort a provider already rejected within the cache window)."""
    fp = str(fingerprint or "").strip()
    ceil = str(ceiling or "").strip().lower()
    if not fp or not ceil:
        return
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            entry = data.setdefault("effort_ceilings", {}).get(fp) or {}
            data["effort_ceilings"][fp] = {
                "ceiling": ceil,
                "observed_at": utc_now_iso(),
                "reason": "provider_rejected",
                "prev": entry.get("ceiling") or "",
            }
            _save(drive_root, data)
    except Exception:
        log.debug("record_effort_ceiling failed", exc_info=True)


def get_effort_ceiling(drive_root: Any, fingerprint: str) -> str:
    """Return the learned effort ceiling for a normalized model identity, or ""
    when none. Fail-open (any error → "")."""
    fp = str(fingerprint or "").strip()
    if not fp:
        return ""
    try:
        entry = _load(drive_root).get("effort_ceilings", {}).get(fp)
        return str((entry or {}).get("ceiling") or "").strip().lower()
    except Exception:
        return ""


# --- Owner acknowledgement (asserted) -----------------------------------------

def record_owner_ack(
    drive_root: Any,
    *,
    provider: str,
    base_url: str = "",
    model: str = "",
    window_tokens: int,
    owner: str = "owner",
    headers: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Persist a route-fingerprinted owner acknowledgement of a context window."""
    fp = route_fingerprint(provider=provider, base_url=base_url, model=model, headers=headers, options=options)
    record = {
        "route_fp": fp,
        "window_tokens": int(window_tokens or 0),
        "owner": str(owner or "owner"),
        "ts": utc_now_iso(),
        "note": str(note or ""),
        "route": {
            "provider": str(provider or ""),
            "base_url": str(base_url or ""),
            "model": str(model or ""),
            "headers": list(_canonical_headers(headers)),
            "options": list(_canonical_options(options)),
        },
    }
    _store_evidence(drive_root, "owner_acks", fp, record)
    return record


def list_owner_acks(drive_root: Any) -> List[Dict[str, Any]]:
    return list(_load(drive_root).get("owner_acks", {}).values())


def revoke_owner_ack(drive_root: Any, route_fp: str) -> bool:
    with _STORE_LOCK:
        data = _load(drive_root)
        if route_fp in data.get("owner_acks", {}):
            del data["owner_acks"][route_fp]
            _save(drive_root, data)
            return True
    return False


# --- Probing (opportunistic, cached) ------------------------------------------

def _openai_compatible_metadata_window(
    model: str, base_url: str, allow_fetch: bool, api_key: Optional[str] = None
) -> int:
    """CW6 (v6.34.0): an OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI, ...)
    commonly publishes the per-model window in GET {base_url}/models — under
    max_model_len / context_length / context_window. Best-effort, fail-closed to 0
    (network/auth/parse error, no base_url, or hot-path allow_fetch=False all => 0).

    ``api_key`` may be passed by callers that already hold the key in scope (e.g.
    the settings-save gate, which has the not-yet-persisted value in ``current``).
    When omitted the function falls back to the already-saved settings on disk."""
    if not allow_fetch or not str(base_url or "").strip() or not str(model or "").strip():
        return 0
    try:
        import httpx

        if api_key is None:
            from ouroboros.config import load_settings
            api_key = str((load_settings() or {}).get("OPENAI_COMPATIBLE_API_KEY") or "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(str(base_url).rstrip("/") + "/models", headers=headers, timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        # The saved model is normally provider-prefixed (e.g. ``openai-compatible::llama-3``)
        # while /models lists the BARE id — match either spelling.
        wanted = {str(model), str(model).split("::", 1)[-1]}
        for item in (items or []):
            if not isinstance(item, dict) or str(item.get("id") or item.get("name") or "") not in wanted:
                continue
            sources = [item, item.get("meta") if isinstance(item.get("meta"), dict) else {}]
            for src in sources:
                for key in ("max_model_len", "context_length", "context_window", "max_context_length"):
                    val = src.get(key)
                    if isinstance(val, (int, float)) and int(val) > 0:
                        return int(val)
        return 0
    except Exception:
        return 0


def _provider_metadata_window(
    provider: str, model: str, base_url: str, allow_fetch: bool, api_key: Optional[str] = None
) -> int:
    """Best-effort live window from provider metadata. 0 = no metadata source."""
    p = str(provider or "").strip().lower()
    # OpenRouter publishes context_length in /models (one cached fetch).
    if "openrouter" in p or (not p and "/" in str(model or "")):
        try:
            from ouroboros.llm import LLMClient
            return int(LLMClient.openrouter_context_length(model, allow_fetch=allow_fetch) or 0)
        except Exception:
            return 0
    # CW6: OpenAI-compatible /models probe (vLLM/Ollama/...) before falling to unprobeable.
    if p == "openai-compatible":
        return _openai_compatible_metadata_window(model, base_url, allow_fetch, api_key=api_key)
    # GigaChat's /models (aget_models) lists model ids but does NOT publish a per-model
    # context window, so a gigachat route stays unprobeable (owner-ack path) — no probe.
    return 0


def _local_health_window(model: str) -> int:
    """Local lane window from the running local model (n_ctx). 0 if unavailable."""
    try:
        from ouroboros.local_model import get_manager
        return int(get_manager().get_context_length() or 0)
    except Exception:
        return 0


def _metadata_fetch_transport_failed(provider: str, model: str, use_local: bool) -> bool:
    """True only when a metadata fetch was ATTEMPTED and failed at transport level
    (provider unreachable) — distinct from a route that simply has no metadata source.
    Only the OpenRouter /models fetch reports transport failure; the CW6 OpenAI-compatible
    probe instead fails closed to a 0 window (-> unprobeable -> owner-ack), so a flaky
    OpenAI-compatible endpoint reads as 'unknown', not as a hard connectivity error."""
    if use_local:
        return False  # local health is in-process; its absence is not an outage
    p = str(provider or "").strip().lower()
    is_openrouter = "openrouter" in p or (not p and "/" in str(model or ""))
    if not is_openrouter:
        return False
    try:
        from ouroboros.llm import LLMClient
        return bool(LLMClient.metadata_fetch_attempted_and_failed())
    except Exception:
        return False


_GENERATIVE_PROBE_PROVIDERS = {"cloudru", "openai-compatible", "openai", "openrouter"}
_PROBE_CANARIES = ["OBOCANARYBEGIN7Q", "OBOCANARYMID7Q", "OBOCANARYEND7Q"]


def _generative_probe_enabled() -> bool:
    return (os.environ.get("OUROBOROS_GENERATIVE_PROBE", "1") or "").strip().lower() not in {"", "0", "false", "no", "off"}


def _generative_probe_pad_chars() -> int:
    try:
        return max(200_000, int(os.environ.get("OUROBOROS_GENERATIVE_PROBE_CHARS", "5000000") or "5000000"))
    except (ValueError, TypeError):
        return 5_000_000


def _generative_probe_window(
    provider: str, model: str, base_url: str = "", api_key: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Empirically size a route's window with ONE oversized request, free-only.

    Sends a deliberately over-window input on an OpenAI-compatible route; the
    provider rejects it PRE-inference (free) with the limit in the message. Never
    raises — any setup/transport error returns FAILED (-> fail-closed owner-ack).
    """
    if not _generative_probe_enabled() or provider not in _GENERATIVE_PROBE_PROVIDERS:
        return 0, STATUS_UNPROBEABLE, "generative probe not applicable/enabled for this route"
    pad = _generative_probe_pad_chars()
    chunk = "x " * (pad // 4)
    content = f"{_PROBE_CANARIES[0]} {chunk} {_PROBE_CANARIES[1]} {chunk} {_PROBE_CANARIES[2]} Echo the three OBOCANARY tokens verbatim."
    sent_estimate = max(1, len(content) // 4)
    # Transport lives in the shared LLMClient seam (DEVELOPMENT.md): it owns route
    # resolution, the per-provider token key, the hard timeout, and never-raises. This
    # module only CLASSIFIES the raw outcome into window evidence (fail-closed).
    try:
        from ouroboros.llm import LLMClient

        out = LLMClient().probe_oversized_context(model, content, base_url=base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - defensive
        return 0, STATUS_FAILED, f"generative probe failed: {type(exc).__name__}"
    if out.get("ok"):
        return classify_generative_probe_response(
            200, "", canaries=_PROBE_CANARIES, echoed_text=str(out.get("echoed_text") or ""),
            usage_prompt_tokens=int(out.get("usage_prompt") or 0), sent_token_estimate=sent_estimate,
        )
    status = out.get("status_code")
    return classify_generative_probe_response(
        status if isinstance(status, int) else None, str(out.get("body") or ""),
    )


def probe(
    drive_root: Any,
    *,
    provider: str,
    model: str,
    base_url: str = "",
    headers: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    use_local: bool = False,
    allow_fetch: bool = True,
    allow_generative: bool = False,
    force: bool = False,
    api_key: Optional[str] = None,
) -> CapabilityEvidence:
    """Resolve Capability Evidence for a route, using the cache unless ``force``.

    Order: fresh cache -> owner-ack (asserted) -> provider metadata / local health
    (confirmed) -> unprobeable. Network probing is skipped when allow_fetch=False
    (hot-path callers) — a stale or absent record then reads as unknown."""
    fp = route_fingerprint(provider=provider, base_url=base_url, model=model, headers=headers, options=options)
    data = _load(drive_root)

    # Owner-ack always wins as ASSERTED evidence for its exact route.
    ack = data.get("owner_acks", {}).get(fp)
    if ack:
        return CapabilityEvidence(
            window_tokens=int(ack.get("window_tokens") or 0), status=STATUS_ASSERTED,
            source=SOURCE_OWNER_ACK, route_fp=fp, model=model, provider=provider,
            ts=str(ack.get("ts") or ""), detail=f"owner-ack by {ack.get('owner') or 'owner'}",
        )

    cached = data.get("probes", {}).get(fp)
    # An EXPLICIT generative probe (owner toggle/save, allow_generative=True) must run even
    # when a prior LAZY (allow_generative=False) call left a fresh UNPROBEABLE/FAILED record
    # — otherwise the owner's empirical probe is silently short-circuited and never fires.
    # Only a CONFIRMED cache is authoritative enough to skip the live probe on that path.
    _skip_cache_for_generative = allow_generative and str((cached or {}).get("status") or "") != STATUS_CONFIRMED
    if cached and not force and not _skip_cache_for_generative:
        age = _age_seconds(str(cached.get("ts") or ""))
        ttl = _CONFIRMED_TTL_SEC if cached.get("status") == STATUS_CONFIRMED else _FAILED_TTL_SEC
        if age <= ttl:
            ev = CapabilityEvidence(
                window_tokens=int(cached.get("window_tokens") or 0), status=str(cached.get("status") or STATUS_UNPROBEABLE),
                source=str(cached.get("source") or SOURCE_NONE), route_fp=fp, model=model,
                provider=provider, ts=str(cached.get("ts") or ""), detail=str(cached.get("detail") or ""),
            )
            return ev

    if not allow_fetch:
        # Hot path: never block on the network. Return the (possibly stale) cache
        # marked stale, else unprobeable — both read as unknown for >=1M gates.
        if cached:
            return CapabilityEvidence(
                window_tokens=int(cached.get("window_tokens") or 0), status=str(cached.get("status") or STATUS_UNPROBEABLE),
                source=str(cached.get("source") or SOURCE_NONE), route_fp=fp, model=model,
                provider=provider, ts=str(cached.get("ts") or ""), detail="stale (no fetch on hot path)", stale=True,
            )
        return CapabilityEvidence(0, STATUS_UNPROBEABLE, SOURCE_NONE, fp, model, provider, detail="not probed")

    # Live probe.
    window = 0
    source = SOURCE_NONE
    if use_local:
        window = _local_health_window(model)
        if window > 0:
            source = SOURCE_LOCAL_HEALTH
    if window <= 0:
        meta = _provider_metadata_window(provider, model, base_url, allow_fetch=allow_fetch, api_key=api_key)
        if meta > 0:
            window, source = meta, SOURCE_PROVIDER_METADATA

    # Generative probe: only when metadata gave nothing AND a toggle/save call-site
    # opted in (allow_generative) — never on the lazy per-task hot path. Confirms a
    # window empirically via a free over-window reject; a 200/numberless reject -> owner-ack.
    if window <= 0 and allow_generative and not use_local:
        gwin, gstatus, gdetail = _generative_probe_window(provider, model, base_url, api_key=api_key)
        if gwin > 0:
            window, source = gwin, SOURCE_GENERATIVE_PROBE
        elif gstatus == STATUS_FAILED:
            ev = CapabilityEvidence(0, STATUS_FAILED, SOURCE_NONE, fp, model, provider,
                                    ts=utc_now_iso(), detail=gdetail)
            _store_evidence(drive_root, "probes", fp, ev.to_json())
            return ev

    if window > 0:
        ev = CapabilityEvidence(window, STATUS_CONFIRMED, source, fp, model, provider, ts=utc_now_iso(), detail="live probe")
        _store_evidence(drive_root, "probes", fp, ev.to_json())
        return ev

    # window <= 0. A provider OUTAGE must NEVER erase a prior confirmed record
    # (the module invariant) — keep it, surfaced as stale, and do not overwrite the
    # cache. Otherwise distinguish a transient outage (STATUS_FAILED, so the owner
    # sees an error: "no connection") from a route that simply has no metadata
    # source (STATUS_UNPROBEABLE -> the owner-ack path).
    prior = cached if isinstance(cached, dict) else None
    prior_win = int((prior or {}).get("window_tokens") or 0)
    prior_status = str((prior or {}).get("status") or "")
    if prior is not None and prior_status in _KNOWN_STATUS and prior_win > 0:
        return CapabilityEvidence(
            prior_win, prior_status, str(prior.get("source") or SOURCE_NONE), fp, model, provider,
            ts=str(prior.get("ts") or ""), detail="kept prior evidence (probe blip)", stale=True,
        )
    if _metadata_fetch_transport_failed(provider, model, use_local):
        ev = CapabilityEvidence(0, STATUS_FAILED, SOURCE_NONE, fp, model, provider, ts=utc_now_iso(),
                                detail="provider unreachable during probe")
    else:
        ev = CapabilityEvidence(0, STATUS_UNPROBEABLE, SOURCE_NONE, fp, model, provider, ts=utc_now_iso(),
                                detail="no provider metadata; owner-ack required for a >=1M gate")
    _store_evidence(drive_root, "probes", fp, ev.to_json())
    return ev
