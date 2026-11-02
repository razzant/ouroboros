"""Quota and account-roster reads without the full status diagnostics barrier.

Envelope types are owned by ``claudexor_contracts.py`` and re-exported from
``contracts.py``. Contract: ``docs/PASSIVE_QUOTA_READ.md``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ouroboros.claudexor_observations import passive_read_error as _read_error

log = logging.getLogger(__name__)

# The roster answers only the unified account model (Claudexor 3.6.0+): named
# profiles, routing facts in the ``accountPools`` key its producer always emits.
# A well-formed roster without it is the retired pre-unified model (native rows
# in ``harnessAccounts``): a typed refusal, never an empty or malformed roster.
# Full Accounts status classifies its own roster read with the same function.
ACCOUNT_MODEL_UNSUPPORTED = "account_model_unsupported"


def _elapsed_ms(start: float) -> int:
    return min(2_147_483_647, max(0, round((time.monotonic() - start) * 1000)))


def _object_lists(value: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> bool:
    return (isinstance(value, dict)
            and all(isinstance(value.get(key), list) for key in required)
            and all(key not in value or isinstance(value[key], list) for key in optional)
            and all(isinstance(row, dict)
                    for key in required + optional for row in value.get(key, [])))


def roster_error(value: Any) -> str:
    """'' for a usable unified roster, else why the roster facet is not answered."""
    if not _object_lists(value, ("profiles",), ("accountPools",)):
        return "malformed_response"
    # An unidentifiable row is unreadable membership, not an empty roster.
    for wrapper in value["profiles"]:
        profile = wrapper.get("profile")
        if not isinstance(profile, dict) or not all(
            isinstance(profile.get(key), str) and profile[key]
            for key in ("harness_id", "profile_id")
        ):
            return "malformed_response"
        if "enabled" in profile and not isinstance(profile["enabled"], bool):
            return "malformed_response"
    if "accountPools" in value:
        return ""  # an old client's ``harnessAccounts`` carrier beside it is never read
    legacy = value.get("harnessAccounts")
    if isinstance(legacy, list) and all(
        isinstance(row, dict) and isinstance(row.get("harness_id"), str) and row["harness_id"]
        for row in legacy
    ):
        return ACCOUNT_MODEL_UNSUPPORTED
    return "malformed_response"


def _quota_error(value: Any) -> str:
    from ouroboros.claudexor_observations import quota_envelope_valid

    return "" if quota_envelope_valid(value) else "malformed_response"


def _read_facet(reader: Callable, check: Callable) -> tuple:
    start = time.monotonic()
    try:
        value = reader()
        code = check(value)
        return (None if code else value, {"code": code} if code else None, _elapsed_ms(start))
    except Exception as exc:
        return None, _read_error(exc), _elapsed_ms(start)


def _quota_read(gateway: Any, *, model: str | None = None, request_kind: str = "text") -> dict[str, Any]:
    """Legacy view negotiation after definite 400 only, never unknown transport."""
    from ouroboros.claudexor_observations import read_quota

    return read_quota(gateway, model=model, request_kind=request_kind)


def _quota_payload(*, model: str | None = None, request_kind: str = "text") -> dict[str, Any]:
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
                accounts_call = pool.submit(_read_facet, gateway.credential_profiles, roster_error)
                quota_call = pool.submit(
                    _read_facet, lambda: _quota_read(gateway, model=model, request_kind=request_kind), _quota_error,
                )
            for facet, call in (("accounts", accounts_call), ("quota", quota_call)):
                value, error, elapsed = call.result()
                payload["timings_ms"][facet] = elapsed
                payload["reads"][facet] = "failed" if error else "ok"
                if error:
                    payload["read_errors"][facet] = error
                elif facet == "accounts":
                    # The unchanged envelope: named identities, enablement and pool
                    # facts. No catalog/manifest filter can erase an account here.
                    payload["profiles"] = value
                    payload["unified_accounts"] = True
                else:
                    payload["quota"] = value["snapshots"]
                    payload["quota_absences"] = value.get("absences", [])
                    for key in ("resources", "availability_scope"):
                        if key in value:
                            payload[key] = value[key]
    if "discovery" not in payload["timings_ms"]:
        payload["timings_ms"]["discovery"] = _elapsed_ms(discovery_start)
    payload["timings_ms"]["total"] = _elapsed_ms(start)
    log.info("claudexor_passive_quota_read reads=%s read_errors=%s timings_ms=%s",
             payload["reads"], payload["read_errors"], payload["timings_ms"])
    return payload
