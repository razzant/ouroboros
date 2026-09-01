"""Explicit foreground quota refresh through the owned Claudexor daemon."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error

log = logging.getLogger(__name__)


def _refresh_quota() -> Dict[str, Any]:
    """Call the owned daemon's foreground quota operation exactly once."""
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    endpoint = discover_daemon_at(owned_config_dir())
    with ClaudexorGateway(endpoint) as gateway:
        gateway.handshake()
        return gateway.refresh_quota()


async def api_claudexor_quota_refresh(request: Request) -> JSONResponse:
    """POST /api/claudexor/quota/refresh — explicit owner foreground refresh."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return JSONResponse(await asyncio.to_thread(_refresh_quota))
    except ClaudexorUnavailable as exc:
        return json_error(f"{exc.code}: {exc}", 503)
    except Exception as exc:
        log.exception("api_claudexor_quota_refresh failed")
        return json_error(f"{type(exc).__name__}: Claudexor quota refresh failed")


__all__ = ["api_claudexor_quota_refresh"]
