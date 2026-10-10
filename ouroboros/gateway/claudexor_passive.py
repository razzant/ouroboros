"""Quota and account-roster reads without the full status diagnostics barrier.

Envelope types are owned by ``claudexor_contracts.py`` and re-exported from
``contracts.py``. Contract: ``docs/PASSIVE_QUOTA_READ.md``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)


# Never project arbitrary engine error codes or exception prose: either can
# contain paths, credential material or an unbounded upstream response body.
_SAFE_ERROR_CODES = frozenset({
    "daemon_not_discovered", "daemon_descriptor_unreadable",
    "daemon_descriptor_incomplete", "daemon_token_unreadable",
    "daemon_endpoint_not_loopback", "daemon_unreachable",
    "malformed_response", "protocol_incompatible", "daemon_recovery_only",
})


def _elapsed_ms(start: float) -> int:
    return min(2_147_483_647, max(0, round((time.monotonic() - start) * 1000)))


def _read_error(exc: Exception) -> dict[str, Any]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if not isinstance(exc, ClaudexorUnavailable):
        return {"code": "read_failed"}
    code = exc.code if exc.code in _SAFE_ERROR_CODES else "read_refused"
    if exc.observation_reason == "observation_read_timeout":
        code = "observation_read_timeout"
    result: dict[str, Any] = {"code": code}
    if 400 <= exc.status_code <= 599:
        result["status_code"] = exc.status_code
    return result


def _object_lists(value: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> bool:
    return (isinstance(value, dict)
            and all(isinstance(value.get(key), list) for key in required)
            and all(key not in value or isinstance(value[key], list) for key in optional)
            and all(isinstance(row, dict)
                    for key in required + optional for row in value.get(key, [])))


def _roster_valid(value: Any) -> bool:
    if not _object_lists(value, ("profiles", "harnessAccounts"), ("accountPools",)):
        return False
    # An unidentifiable row is unreadable membership, not an empty roster.
    for wrapper in value["profiles"]:
        profile = wrapper.get("profile")
        if not isinstance(profile, dict) or not all(
            isinstance(profile.get(key), str) and profile[key]
            for key in ("harness_id", "profile_id")
        ):
            return False
        if "enabled" in profile and not isinstance(profile["enabled"], bool):
            return False
    return all(isinstance(row.get("harness_id"), str) and row["harness_id"]
               for row in value["harnessAccounts"])


# Opt-in projection: older engines ignore the selector or explicitly refuse it.
# Neither case needs an operations/diagnostics discovery barrier.
_QUOTA_VIEW = "constraint_freshness"
_CONSTRAINT_FRESHNESS = ("fresh", "stale", "unknown")


def _quota_valid(value: Any) -> bool:
    if not _object_lists(value, ("snapshots",), ("absences",)):
        return False
    groups = [snapshot.get("constraints", []) for snapshot in value["snapshots"]]
    constraints = [row for group in groups if isinstance(group, list) for row in group]
    if not any(isinstance(row, dict) and "freshness" in row for row in constraints):
        return True  # Legacy envelope: snapshot freshness stays the conservative fact.
    # Complete and exact, or malformed: a partial projection never passes for
    # independent per-constraint freshness.
    return all(isinstance(group, list) for group in groups) and all(
        isinstance(row, dict) and row.get("freshness") in _CONSTRAINT_FRESHNESS for row in constraints)


def _read_facet(reader: Callable, valid: Callable) -> tuple:
    start = time.monotonic()
    try:
        value = reader()
        error = None if valid(value) else {"code": "malformed_response"}
        return (value if error is None else None, error, _elapsed_ms(start))
    except Exception as exc:
        return None, _read_error(exc), _elapsed_ms(start)


def _quota_read(gateway: Any) -> dict[str, Any]:
    """One legacy GET after an explicit HTTP 400, never after unknown transport."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return gateway.quota_state(view=_QUOTA_VIEW)
    except ClaudexorUnavailable as exc:
        if exc.status_code != 400:
            raise
        return gateway.quota_state()


def _quota_payload() -> dict[str, Any]:
    """Owned-only authenticated GETs; no handshake, diagnosis, refresh or wake.

    The existing transport still sends the protocol-major header and enforces
    loopback discovery. This projection does not negotiate execution features
    or claim engine/runtime health. Only its two required reads share a barrier.
    """
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    start = time.monotonic()
    payload: dict[str, Any] = {
        "view": "quota",
        "profiles": {},
        "quota": [],
        "quota_absences": [],
        "unified_accounts": False,
        "reads": {"catalog": "not_read", "accounts": "not_read", "quota": "not_read"},
        "read_errors": {},
        "timings_ms": {},
    }
    discovery_start = time.monotonic()
    try:
        gateway = ClaudexorGateway(discover_daemon_at(owned_config_dir()))
    except Exception as exc:
        payload["read_errors"]["discovery"] = _read_error(exc)
    else:
        payload["timings_ms"]["discovery"] = _elapsed_ms(discovery_start)
        with gateway:
            with ThreadPoolExecutor(max_workers=2) as pool:
                accounts_call = pool.submit(_read_facet, gateway.credential_profiles, _roster_valid)
                quota_call = pool.submit(
                    _read_facet, lambda: _quota_read(gateway), _quota_valid,
                )
            for facet, call in (("accounts", accounts_call), ("quota", quota_call)):
                value, error, elapsed = call.result()
                payload["timings_ms"][facet] = elapsed
                payload["reads"][facet] = "failed" if error else "ok"
                if error:
                    payload["read_errors"][facet] = error
                elif facet == "accounts":
                    # Keep native and named identities, enablement, and pool facts.
                    # No catalog/manifest filter can erase an account here.
                    payload["profiles"] = value
                    # This additive envelope member shipped with the unified
                    # account model (the same contract as GET account-pools).
                    # Its presence, even empty, is evidence in THIS roster read;
                    # no operations discovery is needed to resolve null aliases.
                    payload["unified_accounts"] = "accountPools" in value
                else:
                    payload["quota"] = value["snapshots"]
                    payload["quota_absences"] = value.get("absences", [])
    if "discovery" not in payload["timings_ms"]:
        payload["timings_ms"]["discovery"] = _elapsed_ms(discovery_start)
    payload["timings_ms"]["total"] = _elapsed_ms(start)
    log.info("claudexor_passive_quota_read reads=%s read_errors=%s timings_ms=%s",
             payload["reads"], payload["read_errors"], payload["timings_ms"])
    return payload
