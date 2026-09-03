"""HTTP/WebSocket gateway boundary for the Ouroboros web UI.

The gateway package is the single backend-facing boundary for the browser UI:
Starlette routes, WebSocket dispatch, and payload contracts live here. Core
runtime modules should stay UI-transport agnostic.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type-only: importing the package (and so the contracts) stays transport-free
    from starlette.routing import BaseRoute


def collect_routes(
    *,
    data_dir: pathlib.Path,
    settings_handlers: Mapping[str, Callable[..., Any]] | None = None,
) -> list[BaseRoute]:
    """Return the Starlette routes owned by the gateway boundary."""
    from ouroboros.gateway.router import collect_routes as _collect_routes

    return _collect_routes(data_dir=data_dir, settings_handlers=settings_handlers)


__all__ = ["collect_routes"]
