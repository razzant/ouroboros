"""Observe document navigation and capture an explicit HTTP file navigation.

Playwright has no Download-to-Request reference. Require the native goto
transition and the aborted main-frame request; associate its terminal response
with a unique download only while observed request activity permits it.
"""
from __future__ import annotations

import asyncio
import codecs
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import Message
from pathlib import PurePath
from urllib.parse import urlsplit

from ouroboros import artifacts
from ouroboros.task_custody import fence_publication, publication_fence
from ouroboros.tool_access import canonical_data_root, task_id_for_artifacts

DOWNLOAD_DESCRIPTION = "If this navigation starts an HTTP(S) download, retain the completed file and return its read address."


class DownloadError(Exception):
    """No readable download was published for this call."""


def navigation_deadline(timeout_ms):
    from ouroboros.model_wait import dispatch_deadline_remaining_sec

    budget = float("inf") if timeout_ms == 0 else (30000 if timeout_ms is None else timeout_ms) / 1000
    now, outer = time.monotonic(), dispatch_deadline_remaining_sec()
    # Leave room for the synchronous result handoff, not a nested model round.
    available = budget if outer is None else outer - min(0.1, max(0, outer / 10))
    return now + min(budget, available)


def goto_timeout(deadline):
    if deadline == float("inf"):
        return 0
    available = int((deadline - time.monotonic()) * 1000)
    if available < 1:  # Playwright interprets timeout=0 as unlimited.
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        raise PlaywrightTimeoutError("Page.goto: Timeout before navigation: the call window ended.")
    return available


def _wire_url(value):
    # Strip only the fragment: preserve even a trailing empty query marker.
    return value.partition("#")[0]


def _serialize_http_url(node, value, deadline):
    """The selected Playwright driver's WHATWG parser, not code in the active page."""
    from ouroboros.platform_layer import _hidden_run

    if not node or not os.path.isabs(node):
        raise DownloadError("The requested navigation URL could not be confirmed.")
    try:
        return _hidden_run(
            [node, "-e", "const u = new (require('node:url').URL)(require('node:fs').readFileSync(0, 'utf8'));"
                         "u.hash = ''; process.stdout.write(u.href)"],
            input=value.encode("utf-8"), capture_output=True, check=True, timeout=_remaining_timeout(deadline),
            env={key: val for key, val in os.environ.items() if key.upper() in ("SYSTEMROOT", "WINDIR")},
        ).stdout.decode("utf-8")
    except subprocess.TimeoutExpired as exc:
        raise asyncio.TimeoutError from exc
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        raise DownloadError("The requested navigation URL could not be confirmed.") from exc


def is_download_transition(exc):
    from playwright.async_api import Error

    return isinstance(exc, Error) and exc.message.split("\n", 1)[0] == "Page.goto: Download is starting"


def redirect_targets(request):
    targets = []
    while request is not None:
        targets.append(str(request.url))
        request = request.redirected_from
    return targets


_REQUEST_LIMIT = 1024  # Bound per-call observation, not the lifetime of the page.


def _failed_before_response(request):
    # The SDK links Response on the Request channel before reporting a failed
    # handoff. Neither a missing field nor a frame abort proves that it failed
    # without a response: the SDK sets failure at the requestfailed event.
    native = _native_request(request)
    return bool(getattr(native, "failure", None)) and getattr(native, "_response", ...) is None


def _potential_download_request(page, request):
    # Fetch/XHR and other known subresources cannot become a native Download.
    if not (request.is_navigation_request() or getattr(request, "resource_type", "") in ("", "document", "other")):
        return False
    try:
        return page is None or request.frame.page == page
    except Exception:  # Unknown frame attribution cannot exclude a potential producer.
        return True


class RequestActivity:
    """Generation-local evidence for native downloads that can outlive a call."""

    def __init__(self, events, page=None):
        self.events = events
        self.pending = {}  # Exact wire URL -> native request identities still able to explain a Download.
        self.page = page
        self.capture = None
        for name, callback in (("request", self.started), ("requestfinished", self.resolved),
                               ("requestfailed", self.failed)):
            events.on(name, callback)
        if page is not None:
            page.on("download", self.downloaded)

    def started(self, request):
        if _potential_download_request(self.page, request):
            self.pending.setdefault(_wire_url(request.url), {})[id(_native_request(request))] = request

    def failed(self, request):
        if _failed_before_response(request):
            self.resolved(request)

    def resolved(self, request):
        url = _wire_url(request.url)
        matches = self.pending.get(url)
        if matches is not None:
            matches.pop(id(_native_request(request)), None)
            if not matches:
                self.pending.pop(url)

    def downloaded(self, download):
        if self.capture is None and (request := self.single_handoff(_wire_url(download.url))) is not None:
            # Account for an already observed handoff, not file custody. The
            # context owns clicks and background downloads; never touch their bytes.
            self.resolved(request)

    def single_handoff(self, url):
        matches = self.pending.get(url, {})
        if len(matches) == 1:
            request = next(iter(matches.values()))
            # This is the local Playwright Response observed on the Request
            # channel, not request.response(), which sends another SDK query.
            if getattr(_native_request(request), "_response", None) is not None:
                return request
        return None


def watch_document(page, generation):
    """One owner for redirect facts, including between calls.

    Playwright's public `framenavigated` represents both document commits and
    same-document changes. On the pinned protocol its Frame `navigated` event
    carries `newDocument` for a replacement (also used by wait_for_navigation).
    If a binding doesn't expose this fact, keep old observations rather than
    dropping them on a possibly same-document event or an attachment response.
    """
    generation.navigations, generation.document_epoch = [], 0
    impl = getattr(page.main_frame, "_impl_obj", None)
    emitter = getattr(impl, "_event_emitter", None)
    activity = generation.download_requests = RequestActivity(getattr(page, "context", page), page)

    def commit(event):
        document = event.get("newDocument")
        if document is None:
            return
        # Both SDK bindings unwrap Channel to the same Request identity via
        # _object, but the helper module and nullable keys changed in 1.65.
        request = getattr(document.get("request"), "_object", None)
        if event.get("error"):
            if activity.capture is not None and request is not None:
                activity.capture.on_abort(request)
            return
        # Chromium can commit its internal error view for the preceding failed
        # navigation *after* the next goto starts. It has no HTTP source and
        # cannot produce this call's download; keep real document commits fenced.
        if request is not None or event.get("url") != "chrome-error://chromewebdata/":
            generation.document_epoch += 1
        generation.navigations = redirect_targets(request)

    def observe(response):
        request = response.request
        if (request.is_navigation_request() and request.redirected_from is not None
                and (emitter is None or NavigationDownload._navigation_frame(request) != page.main_frame)):
            generation.navigations.extend(redirect_targets(request))

    if emitter is not None:
        emitter.on("navigated", commit)
    page.on("response", observe)


@dataclass(frozen=True)
class Candidate:
    download: object
    targets: tuple[str, ...]
    status: int
    content_type: str
    length: int | None
    request: object


def _native_request(request):
    return getattr(request, "_impl_obj", request)


class NavigationDownload:
    def __init__(self, page, url, owner, document_changed, activity=None):
        self.page, self.url, self.owner = page, url, owner
        self.document_changed = document_changed
        self.available = urlsplit(url).scheme in ("http", "https") and callable(getattr(page, "on", None))
        self.activity = activity
        self.root = None
        self.aborted = {}  # Native identities; a prior abort can precede requestfailed.
        self.responses = {}  # Native Request identity -> terminal Response.
        self.response_conflicts = set()
        self.requests = {}   # Exact wire URL -> observed Request identities for this navigation window.
        self.downloads = {}  # Exact wire URL -> Download(s); unrelated arrivals are not touched.
        self.accounted = {}  # Single producer at event arrival; settle only when the window ends.
        self.event_count = 0
        self.ambiguous = False
        self.selected = None
        self.invalidated = False
        self.changed = asyncio.Event()

    @contextmanager
    def observe(self):
        if not self.available:
            yield
            return
        activity = self.activity
        events = activity.events if activity is not None else self.page
        if activity is not None:
            self.ambiguous = activity.capture is not None
            if activity.capture is None:
                activity.capture = self
            else:
                activity.capture.ambiguous = activity.capture.invalidated = True
                activity.capture.changed.set()
        listeners = ((events, "request", self.on_request), (self.page, "response", self.on_response),
                     (self.page, "download", self.on_download))
        for source, event, callback in listeners:
            source.on(event, callback)
        try:
            yield
        finally:
            for source, event, callback in listeners:
                source.remove_listener(event, callback)
            if activity is not None and activity.capture is self:
                activity.capture = None
                for url, request in self.accounted.items():
                    pending = activity.pending.get(url, {})
                    if len(self.downloads[url]) == 1 and pending.keys() == {id(_native_request(request))}:
                        activity.resolved(request)

    @staticmethod
    def _navigation_frame(request):
        try:
            return request.frame if request.is_navigation_request() else None
        except Exception:  # Playwright may not know a frame for a request yet.
            return None

    def _new_event(self):
        self.event_count += 1
        self.ambiguous |= self.owner.commands != 1 or self.event_count > _REQUEST_LIMIT
        if self.ambiguous:
            self.invalidated = True
            self.changed.set()
        return not self.ambiguous

    def on_request(self, request):
        if not _potential_download_request(self.page, request) or not self._new_event():
            return
        final = _wire_url(request.url)
        self.requests.setdefault(final, {})[id(_native_request(request))] = request
        if self.selected is not None and final == _wire_url(self.selected.targets[-1]):
            self.invalidated = True
        if self._navigation_frame(request) == self.page.main_frame and request.redirected_from is None:
            self.ambiguous |= self.root is not None and self.root is not request
            self.invalidated |= self.ambiguous
            self.root = request
            self.changed.set()

    def _retired_prior_abort(self, key, request):
        return (_failed_before_response(request) and key not in self.responses
                and key not in self.requests.get(_wire_url(request.url), {}))

    def on_abort(self, request):
        if self._new_event():
            identity = _native_request(request)
            key = id(identity)
            self.aborted[key] = identity
            if (self.selected is not None and identity is not _native_request(self.selected.request)
                    and not self._retired_prior_abort(key, identity)):
                self.invalidated = True
            self.changed.set()

    def on_response(self, response):
        request = response.request
        # A redirect response is never the downloaded body, even before the
        # next Request has been delivered or a chain revisits the final URL.
        if (self._navigation_frame(request) != self.page.main_frame
                or (300 <= response.status < 400 and response.headers.get("location"))
                or not self._new_event()):
            return
        key = id(_native_request(request))
        first = self.responses.setdefault(key, response)
        # WebKit can emit the same terminal metadata more than once.
        if first is not response and self._metadata(first) != self._metadata(response):
            self.response_conflicts.add(key)
            self.invalidated |= key in self.aborted
        self.changed.set()

    def on_download(self, download):
        if self._new_event():
            final = _wire_url(download.url)
            self.downloads.setdefault(final, []).append(download)
            if self.activity is not None and (request := self.activity.single_handoff(final)) is not None:
                self.accounted[final] = request
            if self.selected is not None and final == _wire_url(self.selected.targets[-1]):
                self.invalidated = True
            self.changed.set()

    @staticmethod
    def _request_chain(request):
        chain = []
        while request is not None:
            chain.append(request)
            request = request.redirected_from
        return tuple(reversed(chain))

    def candidate(self):
        if self.document_changed() or self.ambiguous or self.owner.commands != 1:
            raise DownloadError("Ambiguous download: no one uncommitted main-page navigation owns it.")
        aborts = [key for key, request in self.aborted.items() if not self._retired_prior_abort(key, request)]
        if len(aborts) > 1:
            raise DownloadError("Ambiguous download: no one uncommitted main-page navigation owns it.")
        if not self.root or not aborts or not self.downloads:
            return None
        if self.root.method != "GET":
            raise DownloadError("The download did not belong to the requested GET navigation.")
        key, = aborts
        response = self.responses.get(key)
        if response is None or getattr(response.request, "redirected_to", None) is not None:
            return None
        if key in self.response_conflicts:
            raise DownloadError("Ambiguous download: the navigation response metadata disagrees.")
        chain = self._request_chain(response.request)
        if chain[0] is not self.root:
            return None
        targets = tuple(request.url for request in chain)
        final = _wire_url(targets[-1])
        matching = self.downloads.get(final, ())
        if not matching:
            return None
        expected = {id(_native_request(req)) for req in chain if _wire_url(req.url) == final}
        unresolved = self.activity is not None and bool(self.activity.pending.get(final, {}).keys() - expected)
        if len(matching) != 1 or unresolved or self.requests.get(final, {}).keys() != expected:
            raise DownloadError("Ambiguous download: multiple or unresolved requests match the same URL.")
        return Candidate(matching[0], targets, *self._metadata(response), request=response.request)

    @staticmethod
    def _metadata(response):
        headers = response.headers
        length = headers.get("content-length", "")
        encoding = headers.get("content-encoding", "identity").lower()
        measured = (int(length) if len(length) <= 20 and length.isascii() and length.isdecimal()
                    and encoding == "identity" else None)
        return response.status, headers.get("content-type", ""), measured

    async def wait(self, deadline):
        while True:
            self.changed.clear()
            if candidate := self.candidate():
                return candidate
            try:
                await _within(deadline, self.changed.wait)
            except asyncio.TimeoutError as exc:
                raise DownloadError("The navigation's download could not be confirmed before its deadline.") from exc


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise asyncio.TimeoutError
    return seconds


async def _within(deadline, function, *args):
    timeout = _remaining_timeout(deadline)  # Avoid creating an unawaited coroutine after expiry.
    return await asyncio.wait_for(function(*args), timeout)


def _remaining_timeout(deadline):
    return None if deadline == float("inf") else remaining(deadline)


async def _dispose(download, complete):
    # Cleanup belongs to this generation and must not extend the call's wait.
    # Do not delete if cancellation is unconfirmed; the context owns leftovers.
    from ouroboros.tools.browser_session import _CLOSE_STEP_SEC

    try:
        if not complete:
            await asyncio.wait_for(download.cancel(), _CLOSE_STEP_SEC)
        await asyncio.wait_for(download.delete(), _CLOSE_STEP_SEC)
    except Exception:
        pass


async def capture_file(ctx, generation, observation, deadline, readonly, decide):
    candidate, complete = None, False
    revoked = threading.Event()

    def is_current():
        return (ctx.browser_state is generation and not revoked.is_set()
                and not observation.invalidated and not observation.document_changed())

    try:
        proposed = await observation.wait(deadline)
        request_url = _wire_url(proposed.targets[0])
        if request_url != _wire_url(observation.url):
            driver = getattr(generation, "_pw_driver", None)
            normalized = await _within(deadline, asyncio.to_thread, _serialize_http_url,
                                       getattr(driver, "node_executable", None), observation.url, deadline)
            if request_url != normalized:
                raise DownloadError("The download did not belong to the requested GET navigation.")
        if observation.candidate() != proposed:
            raise DownloadError("The download response could not be confirmed.")
        candidate = observation.selected = proposed
        reason = await _within(deadline, observation.owner.decide, decide, candidate.targets, ctx, readonly)
        if reason:
            return "blocked", reason
        if not 200 <= candidate.status < 300 or candidate.status == 206:
            raise DownloadError(f"HTTP {candidate.status} did not provide a complete successful document.")
        try:
            path = await _within(deadline, candidate.download.path)
            complete = True
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            raise DownloadError("The browser did not complete the download.") from exc
        if observation.candidate() != candidate:
            raise DownloadError("The download response could not be confirmed.")
        saved = await _within(deadline, asyncio.to_thread, retain, ctx, path, candidate,
                              candidate.download.suggested_filename, deadline, is_current)
        remaining(deadline)
        if not is_current() or observation.candidate() != candidate:
            raise DownloadError("The download ownership changed before its receipt could be returned.")
        return "download", saved
    except asyncio.TimeoutError:
        return "download_failed", "⚠️ BROWSER_DOWNLOAD_FAILED: The download did not complete within the navigation timeout."
    except DownloadError as exc:
        return "download_failed", f"⚠️ BROWSER_DOWNLOAD_FAILED: {exc}"
    except (OSError, artifacts.ArtifactIdentityError):
        return "download_failed", "⚠️ BROWSER_DOWNLOAD_FAILED: A complete local copy could not be retained."
    finally:
        revoked.set()
        if candidate is not None:
            try:
                owned = observation.candidate() == candidate
            except DownloadError:
                owned = False
            if owned:
                if observation.activity is not None:
                    observation.activity.resolved(candidate.request)
                asyncio.create_task(_dispose(candidate.download, complete))


def _utf8_decoder(content_type):
    headers = Message()
    headers["Content-Type"] = content_type
    charset = headers.get_content_charset()
    if charset is not None:
        try:
            if codecs.lookup(charset).name not in ("ascii", "utf-8", "utf-8-sig"):
                return None
        except LookupError:
            return None
    return codecs.getincrementaldecoder("utf-8")()


def retain(ctx, path, candidate, suggested_name, deadline, is_current):
    """Transfer a completed browser file into system-owned source custody, once."""
    from ouroboros.tools.media import _OCR_PDF_MAX_BYTES

    task_id = task_id_for_artifacts(ctx)
    canonical = canonical_data_root(ctx)
    suffix = PurePath(suggested_name).suffix.removeprefix(".")
    mime = candidate.content_type.partition(";")[0].strip().lower()
    pdf = mime == "application/pdf" or suffix.lower() == "pdf"
    archive = mime in ("application/zip", "application/x-zip-compressed") or suffix.lower() == "zip"
    text = not (pdf or archive) and (mime.startswith("text/") or mime in ("application/json", "application/xml"))
    decoder = _utf8_decoder(candidate.content_type) if text else None
    if not suffix and (pdf or archive):
        suffix = "pdf" if pdf else "zip"

    def current():
        remaining(deadline)
        if not is_current():
            raise DownloadError("The browser session ended before the download could be retained.")

    def check_text(chunk, *, final=False):
        nonlocal decoder
        if decoder is not None:
            try:
                decoder.decode(chunk, final=final)
            except UnicodeDecodeError:
                decoder = None

    class Sink:
        def __init__(self, sink):
            self.sink = sink

        def write(self, chunk):
            current()
            check_text(chunk)
            return self.sink.write(chunk)

    def fill(sink):
        current()
        artifacts.stream_artifact_file(path, Sink(sink))
        check_text(b"", final=True)
        current()

    def admit(target):
        from ouroboros.tools.core_file_tools import read_admission

        _, _, refusal = read_admission(ctx, "artifact_store", str(target))
        if refusal is not None:
            raise DownloadError("This task has no readable address for the retained download.")

    current()
    fence_publication()
    with publication_fence(lambda: time.monotonic() >= deadline or not is_current()):
        ref = artifacts.stream_actor_source(
            canonical, task_id, category="tool_results", source_id="browser-download", extension=suffix,
            fill=fill, expected={"size": candidate.length}, admit_target=admit,
        )
    absolute = artifacts.task_artifact_dir_path(canonical, task_id) / ref["path"]
    arguments = {"path": str(absolute)}
    reader = ""
    tool = None
    if pdf:
        if ref["size"] <= _OCR_PDF_MAX_BYTES:
            tool = "extract_pdf_text"
        else:
            reader = f"PDF retained whole; extract_pdf_text refuses files over {_OCR_PDF_MAX_BYTES} bytes (FILE_TOO_LARGE)."
    elif archive or decoder is not None:
        arguments = {"root": "artifact_store", **arguments}
        tool = "read_archive" if archive else "read_file"
    elif text:
        reader = "Text retained without conversion; read_file requires UTF-8."
    if tool:
        args_text = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
        reader = f"{tool}({args_text})"
        ref["read"] = {"tool": tool, "arguments": arguments}
    else:
        ref.pop("read", None)
    # Strings are JSON-quoted, including remote MIME/filename metadata. Neither
    # chooses the destination folder or causes any content to be parsed here.
    return (
        f"Download saved: {json.dumps(suggested_name)} ({ref['size']} bytes; HTTP {candidate.status}; "
        f"Content-Type: {json.dumps(candidate.content_type or 'unknown')})\nSHA-256: {ref['sha256']}\n"
        f"Absolute path: {absolute}" + (f"\n{reader}" if reader else ""),
        ref,
    )
