"""Async stand-ins for the Playwright objects the builtin browser drives: no driver, no browser.

Each browser generation runs its Playwright on its own event loop (``tools/browser_session.py``),
so a fake page, context, browser or driver answers with coroutines where Playwright's async API
does (``coro`` turns a test's plain callable into one). ``install_async_api`` replaces
``playwright.async_api`` for one test; ``close_generations`` closes the loops a test's
generations started, so none outlives it (the conftest fails a test that leaks a thread).
"""
from __future__ import annotations

import sys
import types
from typing import Any, Callable


def coro(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function`` as an async callable: Playwright's async API awaits what the sync one called."""
    async def call(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)
    return call


class TimeoutError(Exception):  # noqa: A001 - the fake module's ``playwright.async_api.TimeoutError``
    pass


def install_async_api(monkeypatch: Any, playwright: Any, starts: list | None = None) -> None:
    """``playwright.async_api`` whose ``async_playwright().start()`` returns ``playwright``."""
    async def start() -> Any:
        if starts is not None:
            starts.append(True)
        return playwright

    module = types.SimpleNamespace(async_playwright=lambda: types.SimpleNamespace(start=start),
                                   TimeoutError=TimeoutError)
    monkeypatch.setitem(sys.modules, "playwright.async_api", module)
    from ouroboros.tools import browser_session

    monkeypatch.setattr(browser_session, "PlaywrightDriver", lambda: types.SimpleNamespace(
        start=start, stop=playwright.stop))


def close_generations(*generations: Any) -> None:
    """Close what each generation's owner loop opened, as the end of a task does."""
    from ouroboros.tools import browser

    for generation in generations:
        browser.cleanup_browser_handles(generation)


def run_on(generation: Any, awaitable: Any) -> Any:
    """Run a coroutine on ``generation``'s own loop (Playwright objects belong to it)."""
    from ouroboros.tools import browser_session

    return browser_session.owner_of(generation).run(awaitable)


def track_sessions(monkeypatch: Any) -> list:
    """Record every generation whose owner the browser asks for in this test, by identity
    (``BrowserState`` is a dataclass: equal fields are not the same generation)."""
    from ouroboros.tools import browser_session

    seen: list = []
    real = browser_session.owner_of

    def owner_of(generation: Any) -> Any:
        if not any(known is generation for known in seen):
            seen.append(generation)
        return real(generation)

    monkeypatch.setattr(browser_session, "owner_of", owner_of)
    return seen


async def wait_event(event: Any, timeout: float) -> bool:
    """A threading.Event awaited without blocking the session loop (other requests still run there)."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while not event.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True
