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
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import (
    atomic_write_json,
    estimate_tokens,
    is_credential_header_name,
    read_json_dict,
    utc_now_iso,
)

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


def is_known(evidence: Any, *, require_fresh: bool = False) -> bool:
    """Whether ``evidence`` is a KNOWN (confirmed/asserted) sourced observation.

    The SSOT for "does this record count as evidence at all". ``require_fresh``
    additionally rejects a STALE record — one past its TTL that the probe could not
    re-verify (an expired cache read on the no-fetch hot path, or a prior record kept
    across a provider outage). ``probe`` already documents that contract ("a stale or
    absent record then reads as unknown"); stating it HERE is what keeps every caller
    from restating it, or forgetting to.

    Accepts any evidence-shaped record (``status`` / ``window_tokens`` / ``stale``),
    so a surface that carries the same fields — e.g. ``reviewer_window.ReviewerWindow``
    — reuses this predicate instead of re-deriving it."""
    if evidence is None:
        return False
    return (
        str(getattr(evidence, "status", "") or "") in _KNOWN_STATUS
        and int(getattr(evidence, "window_tokens", 0) or 0) > 0
        and not (require_fresh and bool(getattr(evidence, "stale", False)))
    )


def confirms_at_least(
    evidence: Any, threshold: int = ONE_MILLION, *, require_fresh: bool = False,
) -> bool:
    """True only when KNOWN evidence meets the threshold.

    unprobeable / failed / None / below-threshold all fail closed.

    ``require_fresh`` picks the caller's freshness policy EXPLICITLY, because the two
    directions carry opposite risk and the choice must be visible at the call site:

    * a gate that AUTHORIZES on the evidence (blocking scope-review authority, BIBLE
      P3) passes ``require_fresh=True`` — an expired or outage-carried window is a
      dated impression, not the sourced Capability Evidence the floor turns on;
    * a gate that would DOWNGRADE the owner's own cognitive horizon on a provider blip
      keeps the default ``False`` — this module's standing invariant is that an outage
      must never erase a prior confirmed record (P4/P1)."""
    return is_known(evidence, require_fresh=require_fresh) and (
        int(getattr(evidence, "window_tokens", 0) or 0) >= int(threshold)
    )


# --- Route fingerprint ---------------------------------------------------------

def _canonical_headers(headers: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(headers, dict):
        return ()
    # Credentials are dispatch authentication, not route capability identity.
    # Omitting both value and presence keeps key rotation (or late key loading)
    # from invalidating otherwise identical route evidence.  Non-secret beta /
    # routing headers remain fingerprinted because they can change the window.
    return tuple(sorted(
        (str(k).lower(), str(v))
        for k, v in headers.items()
        if not is_credential_header_name(k)
    ))


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

def canonical_evidence_root() -> pathlib.Path:
    """The ONE observation store's root (host data dir, never a child drive).

    Density witnesses are written at settlement through the usage-accounting
    fallback root (`usage_ledger._drive_root(None)`); readers must resolve the
    same root, or a forked/child task with its own empty drive would read
    cold 1.0 forever while its own sends teach the canonical store.
    """
    from ouroboros.usage_ledger import _drive_root

    return _drive_root(None)


def _store_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "capability_evidence.json"


def _load(drive_root: Any) -> Dict[str, Any]:
    data = read_json_dict(_store_path(drive_root))
    if isinstance(data, dict):
        data.setdefault("probes", {})
        data.setdefault("owner_acks", {})
        data.setdefault("effort_ceilings", {})
        data.setdefault("effort_floors", {})
        data.setdefault("rejected_params", {})
        data.setdefault("token_density", {})
        return data
    return {
        "probes": {}, "owner_acks": {}, "effort_ceilings": {},
        "effort_floors": {}, "rejected_params": {}, "token_density": {},
    }


def _save(drive_root: Any, data: Dict[str, Any]) -> bool:
    path = _store_path(drive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, data)  # atomic rename — never a torn/partial file
        return True
    except OSError:
        return False


def _store_evidence(drive_root: Any, kind: str, fp: str, value: Dict[str, Any]) -> None:
    """Locked, atomic read-modify-write of one evidence entry (``probes`` or
    ``owner_acks``). The lock re-reads the CURRENT file inside the critical section
    so a concurrent owner-ack and probe never clobber each other; the network probe
    itself runs OUTSIDE this lock. Never raises."""
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            _drop_expired_probes(data)
            data.setdefault(kind, {})[fp] = value
            _save(drive_root, data)
    except Exception:
        log.debug("capability evidence store failed (%s)", kind, exc_info=True)


def _drop_expired_probes(data: Dict[str, Any]) -> None:
    """Write-side expiry for the ``probes`` namespace (CPL4-C8).

    Failed/unprobeable records are pure retry throttles: past
    ``_FAILED_TTL_SEC`` the reader ignores them, so the write drops them.
    Confirmed records outlive their read TTL as provider-blip evidence
    (``probe`` deliberately keeps a stale prior CONFIRMED across an outage)
    and drop only past the unified GC retention — a route unprobed for that
    long re-establishes evidence from scratch, the documented fail-closed
    reset. ``owner_acks`` are owner authority and never expire; the other
    namespaces already filter expired entries on their own write paths.
    Records with an unparseable timestamp or an unknown status are KEPT
    (fail-closed: never delete what cannot be read).
    """
    from ouroboros.retention import get_gc_retention_days

    probes = data.get("probes")
    if not isinstance(probes, dict):
        return
    confirmed_max_age_sec = get_gc_retention_days() * 86400.0
    for fp in list(probes):
        record = probes.get(fp)
        if not isinstance(record, dict):
            continue
        if parse_deadline_ts(str(record.get("ts") or "")) is None:
            continue  # unreadable timestamp: fail-closed, keep
        age = _age_seconds(str(record.get("ts") or ""))
        status = str(record.get("status") or "")
        if status == STATUS_CONFIRMED:
            if age > confirmed_max_age_sec:
                probes.pop(fp, None)
        elif status in (STATUS_FAILED, STATUS_UNPROBEABLE) and age > _FAILED_TTL_SEC:
            probes.pop(fp, None)


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
# KEYING: this historical namespace uses NORMALIZED MODEL IDENTITY rather than
# the exact route fingerprint used by current request-wire compatibility. It is
# retained for diagnostics and upgrade regression compatibility only; production
# request construction, scheduling, and recovery do not consult it as authority.

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


# --- Learned reasoning-effort floors (v6.73.2) ----------------------------------
# Historical VALUE-TOO-LOW mirror of effort_ceilings. Some endpoints make
# reasoning mandatory, but current adaptation is exact-route, success-confirmed
# request-wire evidence. These model-global rows remain diagnostic/read-compatible.
# Historical lifecycle remains readable: ceilings are sticky, while floors expire
# like rejected_params. Since normal dispatch ignores this namespace, expiry changes
# diagnostic state only; exact-route request-wire evidence owns runtime self-healing.

_EFFORT_FLOORS_TTL_SEC = 14 * 24 * 3600.0


def record_effort_floor(drive_root: Any, fingerprint: str, floor: str) -> None:
    """Persist the learned reasoning-effort floor for a normalized model identity.

    Best-effort, never raises; a HIGHER floor always wins on merge (mirror of the
    ceiling's lower-wins rule — a provider-required minimum is never silently
    lowered within the cache window)."""
    from ouroboros.config import effort_rank
    fp = str(fingerprint or "").strip()
    value = str(floor or "").strip().lower()
    if not fp or not value:
        return
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            store = data.setdefault("effort_floors", {})
            entry = store.get(fp) or {}
            prev = str(entry.get("floor") or "").strip().lower()
            fresh = _age_seconds(str(entry.get("observed_at") or "")) < _EFFORT_FLOORS_TTL_SEC
            if fresh and prev and effort_rank(prev) >= effort_rank(value):
                value = prev
            store[fp] = {
                "floor": value,
                "observed_at": utc_now_iso(),
                "reason": "provider_required",
                "prev": prev,
            }
            _save(drive_root, data)
    except Exception:
        log.debug("record_effort_floor failed", exc_info=True)


def get_effort_floor(drive_root: Any, fingerprint: str) -> str:
    """Return the non-expired learned effort floor for a normalized model
    identity, or "" (fail-open: absence, expiry, or any error → "")."""
    fp = str(fingerprint or "").strip()
    if not fp:
        return ""
    try:
        entry = _load(drive_root).get("effort_floors", {}).get(fp) or {}
        if _age_seconds(str(entry.get("observed_at") or "")) >= _EFFORT_FLOORS_TTL_SEC:
            return ""
        return str(entry.get("floor") or "").strip().lower()
    except Exception:
        return ""


# --- Learned rejected request parameters (v6.69.0) ------------------------------
# Same design as effort_ceilings: a separate namespace ("rejected_params") keyed by
# the NORMALIZED MODEL IDENTITY, sharing only the store file and lock. A provider
# rejection of an optional request parameter (e.g. temperature on a reasoning
# model) is learned reactively in llm.py; persisting it here means a NEW process
# (worker restart, review subprocess) strips the parameter proactively instead of
# re-paying a 404 + retry on its first call. Entries EXPIRE (providers change
# supported_parameters independently of releases — the mutable-external-fact rule
# in DEVELOPMENT.md), after which the reactive retry re-learns if still true.
# Fail-open everywhere: any error → no durable knowledge → today's behavior.

_REJECTED_PARAMS_TTL_SEC = 14 * 24 * 3600.0


def record_rejected_params(drive_root: Any, fingerprint: str, params: Any) -> None:
    """Persist provider-rejected optional request parameters for a model identity.

    Merges with (non-expired) existing knowledge; best-effort, never raises."""
    fp = str(fingerprint or "").strip()
    values = sorted({str(p).strip() for p in (params or []) if str(p or "").strip()})
    if not fp or not values:
        return
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            store = data.setdefault("rejected_params", {})
            entry = store.get(fp) or {}
            existing = entry.get("params") if _age_seconds(str(entry.get("observed_at") or "")) < _REJECTED_PARAMS_TTL_SEC else []
            merged = sorted({*(existing or []), *values})
            store[fp] = {
                "params": merged,
                "observed_at": utc_now_iso(),
                "reason": "provider_rejected",
            }
            _save(drive_root, data)
    except Exception:
        log.debug("record_rejected_params failed", exc_info=True)


def get_rejected_params(drive_root: Any, fingerprint: str) -> Set[str]:
    """Return non-expired learned rejected parameters for a model identity.

    Empty set on absence, expiry, or any error (fail-open)."""
    fp = str(fingerprint or "").strip()
    if not fp:
        return set()
    try:
        entry = _load(drive_root).get("rejected_params", {}).get(fp) or {}
        if _age_seconds(str(entry.get("observed_at") or "")) >= _REJECTED_PARAMS_TTL_SEC:
            return set()
        return {str(p) for p in (entry.get("params") or []) if str(p or "").strip()}
    except Exception:
        return set()


# --- Measured tokenizer density -------------------------------------------------
# One raw pair namespace keyed by normalized model. Reducers choose witnessed
# values; no independently refreshed aggregate scalar is an authority.

# 90 days: a model's tokenizer does not drift week to week, and an install that
# idles past the TTL would otherwise fall back to the cold floor and refuse the
# packed deep self-review it could assemble warm (owner decision R60/R61).
_TOKEN_DENSITY_TTL_SEC = 90 * 24 * 3600.0
_TOKEN_DENSITY_FRESH_SEC = 6 * 3600.0
_TOKEN_DENSITY_MAX_PAIRS = 5
_TOKEN_DENSITY_DRIFT_TOLERANCE = 0.05
_TOKEN_DENSITY_MIN_CHARS = 20_000
_TOKEN_DENSITY_SANE_RANGE = (0.5, 4.0)

# Review cold floor from the measured code-heavy pack; Main never consumes it.
COLD_START_TOKEN_DENSITY = 1.65
MEASURED_DENSITY_SAFETY_FACTOR = 1.05

_DENSITY_MEMO: Dict[str, Tuple[float, str]] = {}


def _density_observation_seq(pair: Dict[str, Any]) -> int:
    """Persisted insertion order for witnesses that share one clock tick."""
    try:
        return max(0, int(pair.get("observation_seq") or 0))
    except (TypeError, ValueError):
        return 0


def _density_recency_key(pair: Dict[str, Any]) -> Tuple[float, int]:
    """Chronological key without letting the tie-breaker refresh witness TTL."""
    observed = parse_deadline_ts(pair.get("observed_at"))
    epoch = observed.timestamp() if observed is not None else float("-inf")
    return epoch, _density_observation_seq(pair)


def _density_of(prompt_chars: Any, prompt_tokens: Any) -> float:
    """Real tokens per chars/4 estimated token, or 0.0 when not measurable."""
    try:
        chars = int(prompt_chars or 0)
        tokens = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        return 0.0
    if chars < _TOKEN_DENSITY_MIN_CHARS or tokens <= 0:
        return 0.0
    density = tokens / max(1.0, chars / 4.0)
    low, high = _TOKEN_DENSITY_SANE_RANGE
    return density if low <= density <= high else 0.0


def record_token_density(
    drive_root: Any,
    fingerprint: str,
    *,
    prompt_chars: Any,
    prompt_tokens: Any,
    source: str = "dispatch_usage",
    route_fp: str = "",
    basis: str = "raw",
) -> None:
    """Persist one timestamped raw witness, best-effort and write-throttled.

    ``basis`` names how ``prompt_chars`` measured image blocks — "raw"
    (base64 bytes, pre-fix rows) vs "bounded_proxy" (the provider-billing
    proxy the fit estimator measures on). The row carries it so the two
    bases can never be silently mixed again by a later "unification".
    """
    fp = str(fingerprint or "").strip()
    route = str(route_fp or "").strip()
    density = _density_of(prompt_chars, prompt_tokens)
    if not fp or density <= 0:
        return
    # ``basis`` is part of the witness identity end-to-end: the main resolver
    # accepts only bounded_proxy rows, so a fresh RAW row at the same numeric
    # density must not throttle the FIRST bounded witness as "no drift" —
    # that left the resolver cold for the whole freshness window on an
    # upgraded store (final-lane finding, probe-reproduced).
    memo_key = f"{fp}\0{route}\0{basis}"
    memo = _DENSITY_MEMO.get(memo_key)
    if (
        memo and _age_seconds(memo[1]) < _TOKEN_DENSITY_FRESH_SEC
        and abs(density - memo[0]) <= _TOKEN_DENSITY_DRIFT_TOLERANCE * memo[0]
    ):
        return
    try:
        with _STORE_LOCK:
            data = _load(drive_root)
            store = data.setdefault("token_density", {})
            entry = store.get(fp) or {}
            pairs = [
                pair for pair in (entry.get("pairs") or [])
                if isinstance(pair, dict)
                and _age_seconds(str(pair.get("observed_at") or "")) < _TOKEN_DENSITY_TTL_SEC
                and _density_of(pair.get("prompt_chars"), pair.get("prompt_tokens")) > 0
            ]
            route_pairs = [
                pair for pair in pairs
                if str(pair.get("route_fp") or "") == route
                and str(pair.get("basis") or "raw") == str(basis or "raw")
            ]
            newest = max(
                enumerate(route_pairs),
                key=lambda item: (*_density_recency_key(item[1]), item[0]),
                default=(0, None),
            )[1]
            known = _density_of(
                (newest or {}).get("prompt_chars"), (newest or {}).get("prompt_tokens"),
            )
            if (
                newest is not None
                and _age_seconds(str(newest.get("observed_at") or "")) < _TOKEN_DENSITY_FRESH_SEC
                and abs(density - known) <= _TOKEN_DENSITY_DRIFT_TOLERANCE * known
            ):
                _DENSITY_MEMO[memo_key] = (known, str(newest.get("observed_at") or ""))
                return
            observed_at = utc_now_iso()
            observation_seq = max(
                (_density_observation_seq(pair) for pair in pairs), default=0,
            ) + 1
            pairs.append({
                "prompt_chars": int(prompt_chars or 0),
                "prompt_tokens": int(prompt_tokens or 0),
                "observed_at": observed_at,
                "observation_seq": observation_seq,
                "source": str(source or "dispatch_usage"),
                "route_fp": route,
                "basis": str(basis or "raw"),
            })
            indexed_pairs = list(enumerate(pairs))
            densest_index, densest = max(
                indexed_pairs,
                key=lambda item: (
                    _density_of(item[1].get("prompt_chars"), item[1].get("prompt_tokens")),
                    *_density_recency_key(item[1]),
                    item[0],
                ),
            )
            newest_rest = [
                pair
                for index, pair in sorted(
                    indexed_pairs,
                    key=lambda item: (*_density_recency_key(item[1]), item[0]),
                    reverse=True,
                )
                if index != densest_index
            ][:_TOKEN_DENSITY_MAX_PAIRS - 1]
            store[fp] = {"pairs": [densest, *newest_rest]}
            if _save(drive_root, data):
                _DENSITY_MEMO[memo_key] = (density, observed_at)
    except Exception:
        log.debug("record_token_density failed", exc_info=True)


def get_token_density(drive_root: Any, fingerprint: str) -> float:
    """Densest fresh raw witness for a model identity, else 0.0."""
    fp = str(fingerprint or "").strip()
    if not fp:
        return 0.0
    try:
        pairs = (_load(drive_root).get("token_density", {}).get(fp) or {}).get("pairs") or []
        return max([
            _density_of(pair.get("prompt_chars"), pair.get("prompt_tokens"))
            for pair in pairs
            if isinstance(pair, dict)
            and _age_seconds(str(pair.get("observed_at") or "")) < _TOKEN_DENSITY_TTL_SEC
        ] or [0.0])
    except Exception:
        return 0.0


def _normalized_density_model(model_id: str) -> str:
    try:
        from ouroboros.provider_models import normalize_model_identity
        return normalize_model_identity(str(model_id or ""))
    except Exception:
        return str(model_id or "").strip()


def _fresh_density_pairs(
    store: Dict[str, Any], model_id: str = "", *, basis: str = "",
) -> List[Tuple[Dict[str, Any], float]]:
    # Without ``model_id`` every model's rows are returned — ONLY for the main
    # resolver's exact-route lookup (a route fingerprint belongs to one model;
    # the caller filters on it). No resolver reduces over other models' rows:
    # another tokenizer's density is no evidence about this route.
    # ``basis`` filters to rows measured on one named basis. The MAIN fit
    # resolver passes "bounded_proxy" — its multiplier must match the fit
    # estimator's own measure: a pre-basis row (no stamp) or a legacy ``raw``
    # row was measured against raw base64 chars and can sit at 0.05-0.65 on
    # image routes, so letting it stay authoritative for its 14-day TTL after
    # an upgrade re-poisons exactly what the basis fix cures (the cost is a
    # brief cold start at 1.0). The review resolver passes no basis: its
    # text-heavy witnesses measure the same on either basis.
    entries = [store.get(model_id) or {}] if model_id else list(store.values())
    return [
        (pair, density)
        for entry in entries if isinstance(entry, dict)
        for pair in (entry.get("pairs") or []) if isinstance(pair, dict)
        if _age_seconds(str(pair.get("observed_at") or "")) < _TOKEN_DENSITY_TTL_SEC
        and (not basis or str(pair.get("basis") or "") == basis)
        for density in [_density_of(pair.get("prompt_chars"), pair.get("prompt_tokens"))]
        if density > 0
    ]


def resolve_main_token_density(drive_root: Any, route_fp: str, model_id: str) -> Tuple[float, str]:
    """Newest fresh exact-route witness, then exact-model witness, then neutral."""
    try:
        store = _load(drive_root).get("token_density", {}) or {}
        route = str(route_fp or "").strip()
        route_pairs = [
            item for item in _fresh_density_pairs(store, basis="bounded_proxy")
            if route and str(item[0].get("route_fp") or "") == route
        ]
        if route_pairs:
            return max(
                enumerate(route_pairs),
                key=lambda item: (*_density_recency_key(item[1][0]), item[0]),
            )[1][1], "fresh_route_usage"
        model_pairs = _fresh_density_pairs(
            store, _normalized_density_model(model_id), basis="bounded_proxy",
        )
        if model_pairs:
            return max(
                enumerate(model_pairs),
                key=lambda item: (*_density_recency_key(item[1][0]), item[0]),
            )[1][1], "fresh_model_usage"
    except Exception:
        pass
    return 1.0, "cold_estimate"


def resolve_review_token_density(drive_root: Any, model_id: str) -> Tuple[float, str]:
    """Densest fresh exact-model witness (authoritative, may undercut the cold
    floor), else the cold floor.

    A fresh exact-model witness measures THIS model's real tokenizer density, so
    it is allowed to lower the effective density below ``COLD_START_TOKEN_DENSITY``
    (issue #284: the floor otherwise shrinks a 1M-window reviewer to ~575K
    estimated input tokens against a measured 0.86-1.01 density, and the managed
    scope atlas can never assemble). The floor governs when the only evidence is
    stale (TTL-expired) or absent — and when the only witnesses belong to OTHER
    models: another tokenizer's density says nothing about this route, and
    letting the densest foreign pair govern could only push the cap BELOW the
    already-conservative floor (paid run 2026-09-04: a gemini-3.8-flash row at
    1.81 sized gpt-5.6-terra, measured 0.87-0.96, at 1.90 and cut its scope cap
    from 575,757 to 499,627 of a 1,050,000-token window)."""
    try:
        store = _load(drive_root).get("token_density", {}) or {}
        exact = _fresh_density_pairs(store, _normalized_density_model(model_id))
        if exact:
            return max(item[1] for item in exact) * MEASURED_DENSITY_SAFETY_FACTOR, "measured"
    except Exception:
        pass
    return COLD_START_TOKEN_DENSITY, "cold_conservative"


def resolve_token_density(drive_root: Any, model_id: str) -> Tuple[float, str]:
    """Compatibility alias for the conservative review reducer."""
    return resolve_review_token_density(drive_root, model_id)


# Cold-start density probe (owner decisions R60/R61 for the packed deep
# self-review; the commit gate runs the same rung since 2026-09-05): when a
# review pack is refused or degraded under the COLD floor, ONE bounded send on
# the exact model sources a real witness for the store above.
# Room for a reasoning model to finish thinking AND answer: a cap that ends the
# call mid-reasoning comes back with no content and no usage — no witness.
DENSITY_PROBE_MAX_TOKENS = 256
DENSITY_PROBE_EFFORT = "low"
DENSITY_PROBE_SYSTEM_PROMPT = "Token-density calibration probe: reply with the single word OK."


def cold_start_density_probe(
    drive_root: Any,
    llm: Any,
    emit_progress: Any,
    model: str,
    sample: str,
    *,
    task_id: str,
    call_type: str,
    source: str,
) -> str:
    """The cold-start rung shared by the packed deep self-review and the commit
    gate (scope ladder and triad fit). Returns a typed outcome:

    ``"warm"`` — a fresh exact-model witness already governs: nothing is sent;
    ``"no_sample"`` — nothing to measure on; ``"failed"`` / ``"no_usage"`` /
    ``"unrecorded"`` — the probe sent but yielded no governing witness (the
    cold cap stands, disclosed on progress); ``"measured"`` — the witness is
    recorded and now governs, so the caller recomputes the cap and rebuilds
    ONCE.

    The calibrated input cap is the density-form bound over a FRESH exact-model
    witness, else the cold floor ``COLD_START_TOKEN_DENSITY`` — which lies
    above every measured density, so a repository whose required set fits warm
    is refused cold, and the refusal happens before any send, so the model
    never records the witness that would have admitted it. This rung breaks
    that loop with ONE bounded send on the exact model (``sample``: a slice of
    the real pack, a few output tokens) through the ordinary observed call,
    under ``physical_attempt_limit(1)`` so the transport ladder's own retries
    (a body-error reroute, an encrypted-reasoning strip) cannot turn the ONE
    send into several paid attempts. It never runs on a warm store and never
    retries. ``BudgetExceeded`` propagates:
    the paid ledger's refusal is budget vocabulary the caller discloses in its
    own terms (the deep review lets it reach the agent's budget rail; the
    commit gate records a typed disclosure and keeps its existing refusal)."""
    from ouroboros.llm_observability import chat_observed
    from ouroboros.usage_accounting import BudgetExceeded, physical_attempt_limit

    _density, density_source = resolve_review_token_density(drive_root, model)
    if density_source == "measured":
        return "warm"
    if not sample:
        return "no_sample"
    emit_progress(
        f"No fresh token-density witness for {model}: one bounded probe "
        f"(~{estimate_tokens(sample):,} estimated tokens) calibrates the input cap..."
    )
    try:
        with physical_attempt_limit(1):
            _response, usage = chat_observed(
                llm,
                drive_root=drive_root,
                task_id=task_id,
                call_type=call_type,
                messages=[
                    {"role": "system", "content": DENSITY_PROBE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample},
                ],
                model=model,
                tools=None,
                reasoning_effort=DENSITY_PROBE_EFFORT,
                max_tokens=DENSITY_PROBE_MAX_TOKENS,
                temperature=None,
                no_proxy=True,
            )
    except BudgetExceeded:
        raise
    except Exception as exc:
        log.warning("Token-density probe failed (%s): %s", call_type, exc, exc_info=True)
        emit_progress(f"Density probe failed ({type(exc).__name__}); the cold input cap stands.")
        return "failed"
    real = int((usage or {}).get("prompt_tokens") or 0)
    if real <= 0:
        emit_progress("Density probe returned no usage (prompt_tokens=0); the cold input cap stands.")
        return "no_usage"
    record_token_density(
        drive_root,
        _normalized_density_model(model),
        prompt_chars=len(DENSITY_PROBE_SYSTEM_PROMPT) + len(sample),
        prompt_tokens=real,
        source=source,
    )
    density, density_source = resolve_review_token_density(drive_root, model)
    emit_progress(f"Token density for {model}: {density:.2f} ({density_source}).")
    # A witness the store refused (too few chars, an insane ratio) leaves the
    # cold cap standing — disclosed above by the unchanged source.
    return "measured" if density_source == "measured" else "unrecorded"


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
    if p in {"openai-compatible", "minimax"}:
        if p == "minimax" and api_key is None:
            try:
                from ouroboros.config import load_settings
                api_key = str((load_settings() or {}).get("MINIMAX_API_KEY") or "")
            except Exception:
                api_key = ""
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


_GENERATIVE_PROBE_PROVIDERS = {"cloudru", "openai-compatible", "minimax", "openai", "openrouter"}
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


# Cache-inclusive prompt totals are measurable; GigaChat's and MiniMax's
# semantics remain unknown. DeepSeek probed 2026-09-01: prompt_tokens =
# prompt_cache_hit_tokens + prompt_cache_miss_tokens, i.e. cache-inclusive —
# and its automatic cache makes nearly every warm call cache-bearing, so
# excluding it would starve the route of density witnesses entirely.
_CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS = frozenset({
    "openrouter", "openai", "openai-compatible", "cloudru", "local", "anthropic",
    "deepseek",
})


def observe_token_density(request: Any, usage: Optional[Dict[str, Any]], *, drive_root_resolver: Any) -> None:
    """Learn density after settlement; unknown cache semantics produce no witness."""
    try:
        normalized = dict(usage or {})
        cached = int(normalized.get("cached_tokens") or 0)
        cache_bearing = bool(cached or int(normalized.get("cache_write_tokens") or 0))
        provider = str(request.provider or "").strip().lower()
        if cache_bearing and provider not in _CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS:
            return
        real = int(normalized.get("prompt_tokens") or normalized.get("input_tokens") or 0)
        # A cache-inclusive total landing on 2 x cached_tokens (+-1) is a gateway
        # adding the cache read on top of an already inclusive total, not a
        # tokenizer measurement (paid run 2026-09-04: 22 openrouter gemini rows
        # at exactly 2 x cached - 1, beside honest rows of the same prompt at
        # 1.00-1.17 x cached). One such row, promoted by the densest-wins review
        # reducer, governs the exact model for the 90-day TTL (1.81 against a
        # real 0.88-1.03), so it is no witness. An honest row with
        # uncached == cached +-1 is a coincidence whose loss costs nothing
        # (writes are throttled and five pairs retained); uncached ABOVE cached
        # is ordinary and stays measurable.
        if cached and abs(real - 2 * cached) <= 1:
            return
        # The witness MUST calibrate the basis the fit estimator measures on
        # (bounded image proxy) — the raw-base64 basis fed a self-consistent
        # ~27% under-prediction: measure_main_fit multiplied a BOUNDED
        # estimate by a RAW-basis density. `prompt_tokens_estimate` stays raw
        # on purpose — `_reservation_cost` reads it and budget reservation
        # wants the conservative over-count (owner decision 3=A); do NOT
        # unify the two consumers onto one basis.
        estimate = int(request.prompt_tokens_bounded_estimate or 0)
        basis = "bounded_proxy"
        if estimate <= 0:
            estimate = int(request.prompt_tokens_estimate or 0)
            basis = "raw"
        if real <= 0 or estimate <= 0:
            return
        from ouroboros.provider_models import normalize_model_identity

        record_token_density(
            drive_root_resolver(request.drive_root),
            normalize_model_identity(str(request.model or "")),
            prompt_chars=estimate * 4,
            prompt_tokens=real,
            source="dispatch_usage",
            route_fp=str(request.physical_context.route_fp if request.physical_context else ""),
            basis=basis,
        )
    except Exception:
        log.debug("token-density observation skipped", exc_info=True)
