"""Validate Claudexor's additive effort and prompt-cache evidence without resolving any route.

``session_prompt_cache_ttl_request`` is the one native TTL negotiation: it reads the
task's captured setting and the engine catalog for the already chosen route, and adds
``promptCacheTtl`` to a fresh session request only where that route advertises the tier.
"""
from __future__ import annotations

import copy
from typing import Any


def model_effort_usage(result: dict, request_facts: Any) -> dict:
    """Decode only this result's reports, retaining the exact attempt's intent.

    Live calls and late receipts share this decoder. An older observation on a
    retained candidate is never evidence about the result being settled now.
    """
    effort = copy.deepcopy(request_facts) if isinstance(request_facts, dict) else {}
    effort.update(reported=None, report_source=None)
    usage = {"effort": effort}
    resolution = validated_effort_resolution(result.get("effortResolution"))
    if "effortResolution" in result:
        usage["effort_resolution"] = resolution
    applied = result.get("appliedOptions")
    reported = applied.get("reasoningEffort") if isinstance(applied, dict) else None
    if isinstance(reported, str) and reported.strip():
        effort.update(reported=reported, report_source="provider_applied_options")
    if resolution and resolution["observed"] is not None:
        effort.update(reported=resolution["observed"], report_source=resolution["observedSource"])
    return usage


def validated_effort_resolution(value: Any) -> dict | None:
    """Prepared native options are not dispatch proof or provider observations.

    Unknown/malformed reports stay unknown. Claudexor owns native vocabulary
    and ordering; validate the evidence shape without re-resolving its claim.
    """
    if not isinstance(value, dict):
        return None
    nullable = ("requested", "submitted", "parameter", "observed", "observedSource")
    if any(key not in value or (value[key] is not None and
           (not isinstance(value[key], str) or not value[key].strip())) for key in nullable):
        return None
    resolution = value.get("resolution")
    if not isinstance(resolution, str) or resolution not in {"exact", "downward", "floor", "omitted", "unverifiable", "rejected"}:
        return None
    source = value.get("source")
    if not isinstance(source, str) or source not in {"account_catalog", "live_probe", "versioned_snapshot", "adapter"}:
        return None
    if "reason" in value and not isinstance(value["reason"], str):
        return None
    requested, submitted = value["requested"], value["submitted"]
    # An omitted option may still name its known native parameter.
    if submitted is not None and value["parameter"] is None:
        return None
    if (value["observed"] is None) != (value["observedSource"] is None):
        return None
    if resolution == "exact":
        if requested is None or submitted != requested:
            return None
    elif resolution in {"downward", "floor"}:
        if requested is None or submitted is None:
            return None
    elif submitted is not None:
        return None
    if resolution == "rejected" and value["observed"] is not None:
        return None
    return copy.deepcopy(value)


_TTL_VALUES = ("5m", "1h")
_TTL_OBSERVED = {"1h", "5m", "mixed", "none", "unknown"}
_INHERITED = "inherited_from_continuation"


def validated_prompt_cache_ttl(value: Any) -> dict | None:
    """Validate Claudexor's final-attempt ``prompt_cache_ttl`` telemetry."""
    if not isinstance(value, dict):
        return None
    for key in ("requested", "submitted"):
        if key not in value or (value[key] is not None and value[key] not in _TTL_VALUES):
            return None
    if value["submitted"] is not None and value["submitted"] != value["requested"]:
        return None
    if "reason" in value and value["reason"] is not None and not isinstance(value["reason"], str):
        return None
    observed = value.get("observed")
    if not isinstance(observed, str) or observed not in _TTL_OBSERVED:
        return None
    if "write_1h_tokens" not in value or "write_5m_tokens" not in value:
        return None
    w1h, w5m = value["write_1h_tokens"], value["write_5m_tokens"]
    for token_val in (w1h, w5m):
        if token_val is not None and (isinstance(token_val, bool) or not isinstance(token_val, int) or token_val < 0):
            return None
    if (w1h is None) != (w5m is None):
        return None
    expected = (
        "unknown" if w1h is None else
        "none" if (w1h == 0 and w5m == 0) else
        "1h" if (w1h > 0 and w5m == 0) else
        "5m" if (w1h == 0 and w5m > 0) else
        "mixed"
    )
    if observed != expected:
        return None
    out = copy.deepcopy(value)
    out.setdefault("reason", None)
    return out


def session_prompt_cache_ttl_request(
    request: dict,
    *,
    route_id: str,
    gateway: Any = None,
    capabilities: Any = None,
    continuation_from: Any = None,
) -> dict:
    """Negotiate ``promptCacheTtl`` from captured task settings before measurement/custody."""
    if continuation_from is not None:
        pred = {}
        if isinstance(continuation_from, dict):
            pred = continuation_from.get("prompt_cache_ttl") or continuation_from
        elif hasattr(continuation_from, "request_event") or hasattr(continuation_from, "prompt_cache_ttl"):
            pred = ((getattr(continuation_from, "request_event", None) or {}).get("prompt_cache_ttl")
                    or getattr(continuation_from, "prompt_cache_ttl", None) or {})
        pred = pred if isinstance(pred, dict) else {}
        req_val = pred.get("requested") if pred.get("requested") in _TTL_VALUES else None
        # The engine inherits only the predecessor's accepted request: a tier that never
        # reached it keeps its captured intent and cause, never posing as inherited.
        reached = pred.get("submitted") in _TTL_VALUES or pred.get("reason") == _INHERITED
        return {
            "requested": req_val,
            "submitted": None,
            "reason": pred.get("reason") if req_val and not reached else _INHERITED,
            "observed": "unknown",
            "write_1h_tokens": None,
            "write_5m_tokens": None,
        }
    from ouroboros.config import resolve_prompt_cache_ttl

    ttl = resolve_prompt_cache_ttl()
    if ttl not in _TTL_VALUES:
        return {
            "requested": None,
            "submitted": None,
            "reason": None,
            "observed": "unknown",
            "write_1h_tokens": None,
            "write_5m_tokens": None,
        }
    catalog = capabilities
    cap_failed = False
    if catalog is None and gateway is not None:
        try:
            catalog = gateway.agent_capabilities()
        except Exception:
            cap_failed = True
    harnesses = catalog.get("harnesses") if isinstance(catalog, dict) else None
    if cap_failed or not isinstance(harnesses, list):
        reason: str | None = (
            "capabilities_read_failed"
            if (cap_failed or gateway is not None or capabilities is not None)
            else "capabilities_unavailable"
        )
    else:
        row = next((item for item in harnesses if isinstance(item, dict) and item.get("id") == route_id), {})
        supported = row.get("promptCacheTtls")
        if isinstance(supported, list) and ttl in supported:
            request["promptCacheTtl"] = ttl
            reason = None
        elif "promptCacheTtls" not in row and "promptCacheTtl" not in (catalog.get("runControlKeys") or []):
            reason = "engine_capability_missing"
        else:
            reason = "harness_unsupported"
    return {
        "requested": ttl,
        "submitted": request.get("promptCacheTtl"),
        "reason": reason,
        "observed": "unknown",
        "write_1h_tokens": None,
        "write_5m_tokens": None,
    }


def terminal_prompt_cache_ttl(observed_attempt: Any, start_facts: Any = None) -> dict | None:
    """Final-attempt native evidence; host transmission alone proves no child configuration."""
    start = copy.deepcopy(start_facts) if isinstance(start_facts, dict) and start_facts else None
    if isinstance(observed_attempt, dict) and "prompt_cache_ttl" in observed_attempt:
        reported = validated_prompt_cache_ttl(observed_attempt.get("prompt_cache_ttl"))
        if reported is None:
            if start is None:
                return None
            return {
                "requested": start.get("requested") if start.get("requested") in _TTL_VALUES else None,
                "submitted": None,
                "reason": start.get("reason") or "malformed_engine_telemetry",
                "observed": "unknown",
                "write_1h_tokens": None,
                "write_5m_tokens": None,
            }
        # A continuation's own valid receipt names what it inherited, Default included.
        if (reported.get("requested") is None and start and start.get("requested") in _TTL_VALUES
                and start.get("reason") != _INHERITED):
            reported["requested"] = start["requested"]
            if reported.get("submitted") is None and not reported.get("reason"):
                reported["reason"] = start.get("reason")
        return reported
    if start is not None:
        return {
            "requested": start.get("requested") if start.get("requested") in _TTL_VALUES else None,
            "submitted": None,
            "reason": start.get("reason") or "missing_engine_telemetry",
            "observed": "unknown",
            "write_1h_tokens": None,
            "write_5m_tokens": None,
        }
    return None
