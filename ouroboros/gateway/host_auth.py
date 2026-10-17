"""Shared Host Service authentication keeps reviewed token/grant reads off-loop."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict

from starlette.websockets import WebSocket


class HostServiceAuthError(Exception):
    """Raised when a skill token cannot be authenticated."""


async def _bounded_host_read(ctx, operation: Callable[[], Any]) -> Any:
    """Keep pre-reservation disk work off-loop without filling the shared executor."""
    while not ctx._auth_capacity.acquire(blocking=False):
        await asyncio.sleep(0.01)
    try:
        work = asyncio.get_running_loop().run_in_executor(None, operation)
    except BaseException:
        ctx._auth_capacity.release()
        raise
    # A disconnected caller must not release a permit while its worker still runs.
    work.add_done_callback(lambda _done: ctx._auth_capacity.release())
    return await asyncio.shield(work)


async def _authenticated(
    ctx, raw_token: str, permission: str = "",
) -> tuple[str, Dict[str, Any]]:
    """Resolve the request's skill (and one grant) off the event loop.

    Token discovery reads every registered skill's token, review, enablement and grants
    from disk; inline, one slow walk stalled every request on the loop the owner's API
    shares. It is short read-only work, so the default executor serves it: Presence turns
    no longer wait there (``presence_runner.PresenceTurnExecutions``).
    """

    def resolve() -> tuple[str, Dict[str, Any]]:
        skill_name, token_payload = ctx.authenticate_token_payload(raw_token)
        if permission:
            ctx.require_permission(skill_name, token_payload, permission)
        return skill_name, token_payload

    return await _bounded_host_read(ctx, resolve)


def _token_from_websocket(websocket: WebSocket) -> str:
    header = websocket.headers.get("x-skill-token", "")
    if header:
        return header
    for protocol in websocket.scope.get("subprotocols") or []:
        text = str(protocol or "")
        prefix = "ouroboros.host.events.v1."
        if text.startswith(prefix):
            return text[len(prefix):]
    return ""
