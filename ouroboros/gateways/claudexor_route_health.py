"""Per-route failure shares from the engine's own run records: a passive, optional read.

The engine owns the measurement (razzant/claudexor#525): for each route, a harness and
its effective access profile, the terminal product runs created inside its fixed
seven-day window, how many failed and the leading failure groups. This leaf only
negotiates the route from the serving operation catalog and keeps the body's shape. An
engine that does not list the route is not asked; a refused, absent or malformed answer is
"unavailable from this engine", never a count. No threshold or verdict lives here:
the facts go to the reasoner (BIBLE P5).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ROUTE_HEALTH_PATH = "/v2/daemon/route-health"
TOP_FAILURES = 3  # the contract's bound on failure groups per route
MESSAGE_CHARS = 160  # the contract's bound on a group's display-safe message


def route_health_supported(operations: List[dict]) -> bool:
    """The serving catalog lists the route; older engines answer 404 and are not asked."""
    return any(isinstance(row, dict) and row.get("method") == "GET" and row.get("path") == ROUTE_HEALTH_PATH
               for row in operations or [])


def _count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _failure(group: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(group, dict) or _count(group.get("count")) is None:
        return None
    return {"category": _text(group.get("category")) or "unknown", "code": _text(group.get("code")),
            "message": (_text(group.get("message")) or "")[:MESSAGE_CHARS], "count": group["count"]}


def _route(row: Any) -> Optional[Dict[str, Any]]:
    """One route in the supervisor row's shape, or None when a count is not a count."""
    if not isinstance(row, dict):
        return None
    counts = {key: _count(row.get(key)) for key in ("runs", "succeeded", "failed", "cancelled", "other")}
    groups = row.get("topFailures") if row.get("topFailures") is not None else []
    if counts["runs"] is None or counts["failed"] is None or not isinstance(groups, list):
        return None
    failures = [_failure(group) for group in groups[:TOP_FAILURES]]
    if any(group is None for group in failures):
        return None
    return {"harness": _text(row.get("harnessId")) or "unknown", "access": _text(row.get("access")) or "unknown",
            **counts, "top_failures": failures}


def route_health_facts(body: Any) -> Optional[Dict[str, Any]]:
    """The body in the supervisor row's shape; None for a malformed body, never a partial count."""
    if not isinstance(body, dict) or not isinstance(body.get("routes"), list):
        return None
    routes = [_route(row) for row in body["routes"]]
    if any(route is None for route in routes):
        return None
    return {"window_days": _count(body.get("windowDays")), "from": _text(body.get("from")),
            "generated_at": _text(body.get("generatedAt")), "scanned_runs": _count(body.get("scannedRuns")),
            "complete": body.get("complete") if isinstance(body.get("complete"), bool) else None,
            "incomplete_reason": _text(body.get("incompleteReason")), "routes": routes}


def read_route_health(gateway: Any) -> Optional[Dict[str, Any]]:
    """The engine's facts through an authenticated gateway, or None: the catalog does not list the
    route, the engine refused (404 included), the transport failed or the body is malformed.
    Never raises into its caller."""
    try:
        if not route_health_supported(gateway.operations()):
            return None
        return route_health_facts(gateway._request("GET", ROUTE_HEALTH_PATH))
    except Exception:
        log.debug("Route health unavailable from this engine", exc_info=True)
        return None
