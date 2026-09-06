"""Minimal real-browser client of the live E2E stand, resolved typed.

``resolve_ui_client`` prefers the suite's own ``tests.system_e2e.interfaces.PlaywrightUIClient``
when that lane has landed an implementation with this surface (open/goto/computed_property/
send_chat/screenshot/rebind/close); otherwise the built-in ``UIProbe`` below runs headless
Chromium against the lane's ``IsolatedServer``. Every unavailability is a TYPED reason recorded
in the lane result (``ui_unavailable:<why>``) — never a silently passed acceptance check.

The browser is opened at USE time, through ``GuardedUI``: the v7.0.0-rc.14 paid run opened it at
lane start, and after a 10-23 minute self-mod absorb wait the first ``goto`` met a dead target
(``TargetClosedError``) that escaped the scenario and turned a lane with complete task-side checks
into ``infra_error``. A Playwright failure is now a typed ``ui_unavailable:<ExceptionType>`` reason
on the UI checks alone.
"""
from __future__ import annotations

import pathlib
from typing import Any, Callable

UI_METHODS = ("open", "goto", "computed_property", "send_chat", "screenshot", "rebind", "close")


class UIProbe:
    """Headless Chromium over one lane's web UI."""

    def __init__(self, base_url: str) -> None:
        self.base_url = str(base_url).rstrip("/")
        self._pw = None
        self._browser = None
        self.page = None

    def open(self) -> "UIProbe":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self.page = self._browser.new_page(viewport={"width": 1280, "height": 900})
        return self

    def rebind(self, base_url: str) -> None:
        """A restarted lane server listens on new ports; the page stays open."""
        self.base_url = str(base_url).rstrip("/")

    def goto(self, path: str = "/") -> None:
        self.page.goto(self.base_url + path, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_selector("#chat-input", timeout=60_000)

    def computed_property(self, selector: str, prop: str) -> str:
        return str(self.page.evaluate(
            "([sel, prop]) => getComputedStyle(document.querySelector(sel)).getPropertyValue(prop)",
            [selector, prop]))

    def send_chat(self, text: str, *, swarm: bool = False) -> None:
        """The owner's send path: arm Swarm (force_plan on the WS frame) when asked, then send."""
        if swarm:
            self.page.click("#chat-swarm")
            self.page.wait_for_selector('#chat-swarm[data-armed="true"]', timeout=10_000)
        self.page.fill("#chat-input", text)
        self.page.click("#chat-send")
        if swarm:
            self.page.wait_for_selector('#chat-swarm[data-armed="false"]', timeout=30_000)

    def screenshot(self, path: pathlib.Path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=True)

    def close(self) -> None:
        # Thread affinity: Playwright's sync API binds the browser to the thread that opened it; a
        # close() from another thread raises ("cannot switch to a different thread"), is swallowed
        # below, and LEAKS the whole browser tree. Open, use and close on the lane thread only.
        for closer in (getattr(self._browser, "close", None), getattr(self._pw, "stop", None)):
            try:
                if closer is not None:
                    closer()
            except Exception:  # noqa: BLE001 - teardown must not mask the lane verdict
                pass


class GuardedUI:
    """The lane's view of an OPEN client. A failure of any browser call — a closed target, a dead
    browser, a timeout — becomes ``ui_unavailable:<ExceptionType>`` handed to ``on_unavailable``
    (the lane records it as the reason of its UI checks) and every later call is a no-op returning
    None; the exception never escapes into the scenario, whose other checks stay recorded."""

    def __init__(self, client: Any, on_unavailable: Callable[[str, str], None]) -> None:
        self.client = client
        self.reason = ""
        self._closed = False
        self._on_unavailable = on_unavailable

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self.reason:
            return None
        try:
            return getattr(self.client, name)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a browser failure is a typed reason, never a lost lane
            self.reason = f"ui_unavailable:{type(exc).__name__}"
            self._on_unavailable(self.reason, f"{type(exc).__name__}: {exc}"[:300])
            self.close()
            return None

    def goto(self, path: str = "/") -> None:
        self._call("goto", path)

    def computed_property(self, selector: str, prop: str) -> str | None:
        return self._call("computed_property", selector, prop)

    def send_chat(self, text: str, *, swarm: bool = False) -> None:
        self._call("send_chat", text, swarm=swarm)

    def screenshot(self, path: pathlib.Path) -> None:
        self._call("screenshot", path)

    def rebind(self, base_url: str) -> None:
        self._call("rebind", base_url)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - teardown of a possibly dead browser must not mask the verdict
            pass


def _suite_client(base_url: str):
    """The suite's ``PlaywrightUIClient`` when landed with this surface, else None."""
    try:
        from tests.system_e2e.interfaces import PlaywrightUIClient
    except ImportError:
        return None
    if not all(callable(getattr(PlaywrightUIClient, name, None)) for name in UI_METHODS):
        return None
    try:
        return PlaywrightUIClient(base_url)
    except NotImplementedError:
        return None


def resolve_ui_client(base_url: str) -> tuple[object | None, str]:
    """``(client, "")`` with the client OPEN, or ``(None, "ui_unavailable:<reason>")``."""
    client = _suite_client(base_url)
    if client is None:
        try:
            import playwright.sync_api  # noqa: F401 - availability probe
        except ImportError:
            return None, "ui_unavailable:playwright_not_installed"
        client = UIProbe(base_url)
    try:
        client.open()
    except Exception as exc:  # noqa: BLE001 - a missing browser binary is a typed reason, not a crash
        text = str(exc)
        reason = "browser_missing" if ("Executable doesn't exist" in text or "playwright install" in text.lower()) \
            else f"{type(exc).__name__}"
        return None, f"ui_unavailable:{reason}"
    return client, ""
