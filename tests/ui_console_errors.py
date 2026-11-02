"""Console errors a browser smoke owns, apart from the engine reads its fixture cannot answer.

The direct-mode UI fixture runs no Claudexor engine. Since #1670 the Accounts section reads
the engine's program inventory once at boot (`web/modules/harness_accounts.js` `_init`), and
the host answers with its typed refusal, 503 (docs/architecture/06-agent-core.md "Vendor
program maintenance": inspection never starts an engine). The page shows "Program inspection
unavailable"; only the browser's own network line reaches the console. Nothing else is set aside,
apart from the one WebKit page error ``PageErrors`` names.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from urllib.parse import urlsplit

ENGINE_READS = frozenset({"/api/claudexor/maintenance/harnesses"})


def record_console_errors(page, *, page_errors: bool = True) -> list[tuple[str, str]]:
    """Every console error as (text, source path); an uncaught page error has no source."""
    errors: list[tuple[str, str]] = []
    page.on("console", lambda message: errors.append(
        (message.text, urlsplit((message.location or {}).get("url") or "").path))
        if message.type == "error" else None)
    if page_errors:
        page.on("pageerror", lambda error: errors.append((str(error), "")))
    return errors


def record_failed_loads(page) -> list[tuple[str, int]]:
    """Every response the page receives with an HTTP error status, as (path, status)."""
    failed: list[tuple[str, int]] = []
    page.on("response", lambda response: failed.append((urlsplit(response.url).path, response.status))
            if response.status >= 400 else None)
    return failed


def unexpected_console_errors(console_errors, failed_loads) -> list[str]:
    """The console errors left once each refused engine read's own load line is set aside: only
    a network line whose source is that engine read, at most once per recorded 503 of it."""
    left = list(console_errors)
    for path, status in failed_loads:
        if path not in ENGINE_READS or status != 503:
            continue
        line = f"Failed to load resource: the server responded with a status of {status} "
        match = next((entry for entry in left if entry[1] == path and entry[0].startswith(line)), None)
        if match is not None:
            left.remove(match)
    return [text for text, _source in left]


_WEBKIT_OWN_FETCH = re.compile(r"^(?:Fetch API cannot load )?(\S+) due to access control checks\.$")


def _origin(url: str) -> tuple[str, str, int] | None:
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
        return parts.scheme, parts.hostname, port
    except ValueError:
        return None


def _fetch_origin(error, text: str) -> tuple[str, str, int] | None:
    match = _WEBKIT_OWN_FETCH.fullmatch(text)
    if not match:
        return None
    url = match.group(1)
    # Playwright's WebKit splitErrorMessage splits at the first colon, consuming
    # the following character too: "Fetch API cannot load http://host/..." becomes
    # name="Fetch API cannot load http", message="/host/...". Recover the scheme
    # from that name, never from the page (which could be on a different scheme).
    name = re.fullmatch(r"(?:Fetch API cannot load )?(https?)", getattr(error, "name", "") or "")
    if name and url.startswith("/"):
        url = name.group(1) + ":/" + url
    return _origin(url)


class _PageError(str):
    """One recorded page error's text, carrying its own verdict and where it arrived: ``page`` is
    the watched page's place in watch order (from 1) and ``url`` that page's committed URL then.
    A list operation that moves, removes or clears entries can never hand an exemption to
    another error."""

    own_fetch = False
    page = 0
    url = ""


class PageErrors(list):
    """Every uncaught page error as text, recorded from each page it watches (``page``, once
    constructed, then any passed to ``watch``): one list however many pages a test opens.

    ``unexpected()`` sets aside one engine artifact (#1468), and only on a page watched as one
    the test navigates (as ``page`` is). WebKit reports a fetch of the page's
    OWN origin that it could not complete (cancelled or never sent because a navigation unloaded
    the document, or refused while a test restarts the server) as "<url> due to access control
    checks.", even when the page caught it. Only that text for the receiving page's current
    committed HTTP(S) origin (scheme, host and port) is set aside, at the time the error arrives.
    Before a navigation commits, this is still the document being left; earlier visited
    origins and other pages' origins are not exemptions. A delayed cancellation from a
    different origin after commit still counts. The text cannot prove cancellation, so another
    same-origin error with exactly this engine wording is indistinguishable; pageerror supplies no
    initiating-frame/request identity. If Playwright split
    the URL, its error name must supply the scheme; an ambiguous bare URL counts."""

    def __init__(self, page, engine: str) -> None:
        super().__init__()
        self._webkit = engine == "webkit"
        self._pages: list[SimpleNamespace] = []
        if page is not None:
            self.watch(page)

    def watch(self, page, *, navigates: bool = True):
        """Record ``page``'s errors here from now on; returns the page. ``navigates=False`` keeps
        every error of a page the test does not navigate. Watching a page again adds no listener and
        can only declare it navigating from then on: an entry already recorded keeps its verdict."""
        known = next((state for state in self._pages if state.page is page), None)
        if known is not None:
            known.navigates = known.navigates or navigates
            return page
        state = SimpleNamespace(page=page, number=len(self._pages) + 1, navigates=navigates,
                                url=page.main_frame.url, origin=_origin(page.main_frame.url))
        self._pages.append(state)

        def record(error):
            entry = _PageError(str(error))
            entry.page, entry.url = state.number, state.url
            entry.own_fetch = (self._webkit and state.navigates and state.origin is not None
                               and _fetch_origin(error, entry) == state.origin)
            self.append(entry)

        def committed(frame):
            if frame == page.main_frame:
                state.url, state.origin = frame.url, _origin(frame.url)

        page.on("pageerror", record)
        page.on("framenavigated", committed)
        return page

    def unexpected(self) -> list[str]:
        """Every entry but the set-aside ones; an entry this class did not record counts."""
        return [str(text) for text in self if not getattr(text, "own_fetch", False)]

    def report(self) -> list[str]:
        """Every entry with its verdict and, when this class recorded it, its page and URL."""
        return [("set aside" if getattr(text, "own_fetch", False) else "unexpected")
                + (f" on page {text.page} at {text.url}" if isinstance(text, _PageError) else "")
                + f": {text}" for text in self]
