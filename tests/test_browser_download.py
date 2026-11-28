"""A synthetic HTTP document reaches the real read-only tool consumer once, as a retained input."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy, chromium_runtime, store
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import browser, browser_download
from ouroboros.tools.registry import ToolContext, ToolRegistry

PDF_LABEL = 'Independent PDF reader control'
TEXT = b'Independent text attachment control.\n'
RECOVERED = b'New text after a closed connection.\n'
LARGE_BYTES = 26 * 1024 * 1024 + 1
BLOCK = b'Z' * (64 * 1024)


def test_goto_and_capture_use_one_navigation_deadline(monkeypatch):
    from ouroboros import model_wait

    class PlaywrightTimeoutError(Exception):
        pass

    monkeypatch.setitem(sys.modules, 'playwright.async_api', SimpleNamespace(TimeoutError=PlaywrightTimeoutError))

    monkeypatch.setattr(browser_download.time, 'monotonic', lambda: 1000)
    for outer, budget in ((None, 30), (33.0, 30), (0.25, 0.225)):
        monkeypatch.setattr(model_wait, 'dispatch_deadline_remaining_sec', lambda: outer)
        deadline = browser_download.navigation_deadline(30_000)
        assert deadline == pytest.approx(1000 + budget)
        assert browser_download.goto_timeout(deadline) == pytest.approx(budget * 1000, abs=1)
    for outer in (0, 0.0001):
        monkeypatch.setattr(model_wait, 'dispatch_deadline_remaining_sec', lambda: outer)
        with pytest.raises(PlaywrightTimeoutError, match='before navigation'):
            browser_download.goto_timeout(browser_download.navigation_deadline(30_000))
    monkeypatch.setattr(model_wait, 'dispatch_deadline_remaining_sec', lambda: 180)
    assert browser_download.goto_timeout(browser_download.navigation_deadline(None)) == 30_000
    assert browser_download.goto_timeout(browser_download.navigation_deadline(0)) == 179_900
    monkeypatch.setattr(model_wait, 'dispatch_deadline_remaining_sec', lambda: None)
    assert browser_download.goto_timeout(browser_download.navigation_deadline(0)) == 0


def _observed_download(events, url='https://documents.example/notes.txt'):
    page = SimpleNamespace(main_frame=object(), on=lambda *_: None, remove_listener=lambda *_: None)

    async def decide(*args):
        return ''

    owner = SimpleNamespace(commands=1, decide=decide)
    observation = browser_download.NavigationDownload(page, url, owner, lambda: False)
    request = SimpleNamespace(url=url, frame=page.main_frame, method='GET', redirected_from=None,
                              is_navigation_request=lambda: True)

    async def path():
        events.append('path')
        return 'completed-path'

    async def cancel():
        events.append('cancel')

    async def delete():
        events.append('delete')

    download = SimpleNamespace(url=url, suggested_filename='notes.txt', path=path, cancel=cancel, delete=delete)
    observation.on_request(request)
    observation.on_abort(request)
    observation.on_response(SimpleNamespace(request=request, status=200, headers={'content-type': 'text/plain'}))
    observation.on_download(download)
    return observation


@pytest.mark.parametrize('resource_type', ['', 'other', 'fetch'])
def test_download_matching_uses_the_same_producer_rule_in_and_between_calls(resource_type):
    observation = _observed_download([])
    own, = observation.downloads[observation.url]
    observation.on_download(SimpleNamespace(url='https://documents.example/another.txt'))
    observation.selected = observation.candidate()
    assert observation.selected.download is own
    background = SimpleNamespace(url=own.url, resource_type=resource_type, is_navigation_request=lambda: False)
    activity = browser_download.RequestActivity(observation.page)
    activity.started(background)
    observation.on_request(background)
    if resource_type == 'fetch':
        assert not activity.pending and not observation.invalidated
        assert observation.candidate() == observation.selected
    else:
        assert activity.pending and observation.invalidated
        # Losing the identity of either potential producer cannot be resolved by event timing.
        with pytest.raises(browser_download.DownloadError, match='Ambiguous download'):
            observation.candidate()


@pytest.mark.parametrize('settlement', ['unfinished', 'unknown_response', 'response', 'no_response', 'requestfinished'])
def test_prior_request_remains_relevant_until_proven_settled(settlement):
    async def exercise():
        effects = []
        observation = _observed_download(effects)
        activity = browser_download.RequestActivity(SimpleNamespace(on=lambda *_: None, remove_listener=lambda *_: None))
        observation.activity = activity
        prior = SimpleNamespace(url=observation.url, is_navigation_request=lambda: True)
        activity.started(prior)
        if settlement != 'unknown_response':
            prior._response = object() if settlement == 'response' else None
        if settlement != 'unfinished':
            prior.failure = 'previously unknown browser error'
        if settlement != 'requestfinished':
            activity.failed(prior)
        else:
            activity.resolved(prior)
        with observation.observe():
            if settlement in ('no_response', 'requestfinished'):
                assert observation.candidate().request is observation.root
            else:
                ctx = SimpleNamespace(browser_state=object())
                outcome = await browser_download.capture_file(
                    ctx, ctx.browser_state, observation, time.monotonic() + 1, False, None,
                )
                assert outcome[0] == 'download_failed' and 'Ambiguous download' in outcome[1]
                assert not effects and 'Absolute path:' not in outcome[1]

    asyncio.run(exercise())


@pytest.mark.parametrize('has_response', [False, True])
def test_late_prior_abort_is_ignored_only_after_terminal_no_response(has_response):
    observation = _observed_download([])
    activity = browser_download.RequestActivity(observation.page)
    observation.activity = activity
    prior = SimpleNamespace(**vars(observation.root), _response=object() if has_response else None)
    activity.started(prior)
    observation.on_abort(prior)  # Frame abort can arrive before requestfailed.
    with pytest.raises(browser_download.DownloadError, match='Ambiguous download'):
        observation.candidate()
    prior.failure = 'net::ERR_ABORTED'
    activity.failed(prior)
    if has_response:
        with pytest.raises(browser_download.DownloadError, match='Ambiguous download'):
            observation.candidate()
    else:
        observation.selected = observation.candidate()
        assert observation.selected.request is observation.root
        observation.on_abort(prior)  # Duplicate late report cannot invalidate selection.
        assert not observation.invalidated
        competitor = SimpleNamespace(**vars(prior))
        del competitor._response  # Unknown SDK telemetry still fences publication.
        observation.on_abort(competitor)
        assert observation.invalidated


def test_background_download_only_retires_a_unique_observed_handoff():
    events = SimpleNamespace(on=lambda *_: None)
    activity = browser_download.RequestActivity(events)
    url = 'https://documents.example/notes.txt'
    one = SimpleNamespace(url=url, is_navigation_request=lambda: True, _response=None)
    two = SimpleNamespace(url=url, is_navigation_request=lambda: True, failure='net::ERR_ABORTED', _response=object())
    download = SimpleNamespace(url=url)
    activity.started(one)
    activity.downloaded(download)
    assert list(activity.pending[url].values()) == [one]  # No response proving a handoff.
    activity.started(two)
    activity.failed(two)
    activity.downloaded(download)
    assert list(activity.pending[url].values()) == [one, two]  # Two producers; don't guess.
    activity.resolved(one)  # Later exact settlement of the first request, not a retry by timing.
    assert list(activity.pending[url].values()) == [two]
    activity.downloaded(download)
    assert not activity.pending


def test_download_requires_the_native_abort_for_its_request():
    async def exercise():
        effects = []
        observation = _observed_download(effects)
        observation.aborted.clear()
        observation.on_abort(SimpleNamespace(url=observation.url))  # A URL isn't a Request identity.
        ctx = SimpleNamespace(browser_state=object())
        outcome = await browser_download.capture_file(ctx, ctx.browser_state, observation, time.monotonic(), False, None)
        assert outcome[0] == 'download_failed' and 'could not be confirmed' in outcome[1]
        assert not effects and 'Absolute path:' not in outcome[1]

    asyncio.run(exercise())


def test_browser_observed_address_matching_does_not_decode_wire_paths():
    observation = _observed_download([], url='https://documents.example/a%2Fb.txt')
    observation.downloads.clear()
    observation.on_download(SimpleNamespace(url='https://documents.example/a/b.txt'))
    assert observation.candidate() is None


@pytest.mark.parametrize('via_second_url', [False, True])
def test_redirect_can_revisit_file_url_and_only_terminal_response_supplies_file_metadata(via_second_url):
    observation = _observed_download([])
    start = observation.root
    initial_response = observation.responses[id(start)]
    middle_url = 'https://documents.example/choose' if via_second_url else start.url
    initial_response.status = 302
    initial_response.headers = {'location': middle_url}
    if via_second_url:
        middle = SimpleNamespace(**vars(start))
        middle.url, middle.redirected_from = middle_url, start
        observation.on_request(middle)
        observation.on_response(SimpleNamespace(request=middle, status=302, headers={'location': start.url}))
    else:
        middle = start
    terminal = SimpleNamespace(**vars(start))
    terminal.redirected_from = middle
    observation.on_request(terminal)
    observation.aborted.clear()  # Only the terminal request provides the native abort in this scenario.
    observation.on_abort(terminal)
    observation.on_response(SimpleNamespace(request=terminal, status=200, headers={'content-length': '18'}))
    found = observation.candidate()
    assert found.status == 200 and found.length == 18
    assert found.targets == ((start.url, middle_url, start.url) if via_second_url else (start.url, start.url))


def test_expired_budget_creates_no_awaitable():
    events = []

    def operation():
        events.append('created')
        return asyncio.sleep(0)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(browser_download._within(time.monotonic() - 1, operation))
    assert not events


def test_rejected_owned_transfer_is_cancelled_before_deletion():
    async def exercise():
        events = []
        observation = _observed_download(events)

        async def decline(*args):
            return 'synthetic decision'

        observation.owner.decide = decline
        generation = object()
        ctx = SimpleNamespace(browser_state=generation)
        result = await browser_download.capture_file(ctx, generation, observation, time.monotonic() + 1, True, None)
        assert result == ('blocked', 'synthetic decision')
        while len(events) < 2:
            await asyncio.sleep(0)
        assert events == ['cancel', 'delete']

    asyncio.run(asyncio.wait_for(exercise(), 2))


def test_download_can_finish_after_the_first_three_seconds_of_a_33_sec_outer_window(monkeypatch):
    from ouroboros import model_wait

    monkeypatch.setattr(model_wait, 'dispatch_deadline_remaining_sec', lambda: 33)

    async def exercise():
        clock = [time.monotonic()]
        monkeypatch.setattr(browser_download.time, 'monotonic', lambda: clock[0])
        deadline = browser_download.navigation_deadline(30_000)
        assert browser_download.goto_timeout(deadline) == 30_000
        clock[0] += 4
        effects = []
        observation = _observed_download(effects)

        def save(_ctx, _path, _candidate, _name, save_deadline, current):
            assert current() and save_deadline == deadline
            assert browser_download.remaining(deadline) == 26
            return 'saved', {}

        monkeypatch.setattr(browser_download, 'retain', save)
        ctx = SimpleNamespace(browser_state=object())
        try:
            assert (await browser_download.capture_file(ctx, ctx.browser_state, observation, deadline, False, None)
                    == ('download', ('saved', {})))
            while len(effects) < 2:
                await asyncio.sleep(0)
            assert effects == ['path', 'delete']
        finally:
            monkeypatch.undo()  # Do not leave asyncio's monotone clock patched at loop teardown.

    asyncio.run(exercise())


def test_late_retention_writer_can_neither_publish_nor_extend_wait(tmp_path, monkeypatch):
    from ouroboros import artifacts

    gate = threading.Event()
    entered = threading.Event()
    payload = tmp_path / 'contents.txt'
    payload.write_bytes(b'plain text')
    original_stream = artifacts.stream_artifact_file

    def delayed_stream(path, sink=None, **kwargs):
        if path == payload:
            entered.set()
            assert gate.wait(4)
        return original_stream(path, sink, **kwargs)

    monkeypatch.setattr(artifacts, 'stream_artifact_file', delayed_stream)

    async def exercise():
        events = []
        observation = _observed_download(events)

        async def complete():
            return payload

        observation.downloads[observation.url][0].path = complete
        generation = object()
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / 'data', task_id='a4018001', browser_state=generation)
        try:
            result = await browser_download.capture_file(ctx, generation, observation, time.monotonic() + 0.5, True, None)
            assert result[0] == 'download_failed' and 'within the navigation timeout' in result[1]
            assert entered.is_set() and not gate.is_set()
        finally:
            gate.set()

    asyncio.run(exercise())  # drains the test's default executor, including the released writer
    assert not own_sources(tmp_path / 'data' / 'task_results' / 'artifacts')


def fixture_pdf(label=PDF_LABEL):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=150)
    name = NameObject
    font = DictionaryObject({name('/Type'): name('/Font'), name('/Subtype'): name('/Type1'),
                             name('/BaseFont'): name('/Helvetica')})
    page[name('/Resources')] = DictionaryObject({name('/Font'): DictionaryObject({name('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 14 Tf 15 90 Td ({label}) Tj ET'.encode('ascii'))
    page[name('/Contents')] = writer._add_object(stream)
    target = io.BytesIO()
    writer.write(target)
    return target.getvalue()


@pytest.fixture
def local_documents():
    pdf, denied_pdf, hits = fixture_pdf(), fixture_pdf('Synthetic denied document'), []
    release = threading.Event()
    iframe_release = threading.Event()
    iframe_started = threading.Event()
    flaky_healthy = threading.Event()

    class Documents(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802 - HTTP method callback
            hits.append(self.path)
            if self.path == '/flaky' and not flaky_healthy.is_set():
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if self.path == '/busy-frame':
                iframe_started.set()
                iframe_release.wait(10)
            if self.path == '/slow' and hits.count('/slow') == 1:
                # Leave goto waiting for headers, then let the native
                # navigation settle without swapping the original document.
                time.sleep(0.9)
            redirects = {'/start': '/pdf', '/denied': '/middle', '/middle': '/denied.pdf'}
            if self.path in redirects:
                self.send_response(302)
                self.send_header('Location', redirects[self.path])
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            values = {
                '/': (200, pdf, 'application/pdf', 'practice.pdf', len(pdf)),
                '/pdf': (200, pdf, 'application/pdf', 'practice.pdf', len(pdf)),
                '/denied.pdf': (200, denied_pdf, 'application/pdf', 'denied.pdf', len(denied_pdf)),
                '/file%20name.pdf': (200, pdf, 'application/pdf', 'practice.pdf', len(pdf)),
                '/pdf?author=O%27Reilly': (200, pdf, 'application/pdf', 'practice.pdf', len(pdf)),
                '/inline': (200, pdf, 'application/pdf', '', len(pdf)),
                '/text': (200, TEXT, 'text/plain; charset=utf-8', 'file.txt', len(TEXT)),
                '/flaky': (200, TEXT, 'text/plain', 'recovered.txt', len(TEXT)),
                '/retry-file': (200, RECOVERED, 'text/plain', 'new.txt', len(RECOVERED)),
                '/retry-file?author=O%27Reilly': (200, RECOVERED, 'text/plain', 'new.txt', len(RECOVERED)),
                '/retry-after-timeout': (200, RECOVERED, 'text/plain', 'new.txt', len(RECOVERED)),
                '/clicked': (200, TEXT, 'text/plain', 'clicked.txt', len(TEXT)),
                '/annual%5Ereport.txt': (200, TEXT, 'text/plain', 'annual-report.txt', len(TEXT)),
                '/missing.pdf': (404, TEXT, 'application/pdf', 'missing.pdf', len(TEXT)),
                '/large': (200, pdf, 'application/pdf', 'large.pdf', LARGE_BYTES),
                '/broken': (200, TEXT, 'text/plain', 'broken.txt', len(TEXT) + 1000),
                '/slow': (200, TEXT, 'text/plain', 'slow.txt', len(TEXT)),
                '/stalled-download': (200, BLOCK, 'application/octet-stream', 'stalled.bin', len(BLOCK) + len(TEXT)),
            }
            html = b'<link rel="icon" href="data:,"><p>HTML_CONTROL</p>'
            if self.path == '/with-frame':
                html += b'<iframe src="/busy-frame"></iframe>'
            if self.path == '/with-link':
                html += b'<a href="/clicked">Download note</a>'
            status, data, mime, filename, size = values.get(self.path, (200, html, 'text/html', '', len(html)))
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(size))
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            elif self.path == '/inline':
                self.send_header('Content-Disposition', 'inline')
            self.end_headers()
            try:
                if self.path == '/large':
                    self.wfile.write(data)
                    padding = size - len(data)
                    for _ in range(padding // len(BLOCK)):
                        self.wfile.write(BLOCK)
                    self.wfile.write(BLOCK[:padding % len(BLOCK)])
                elif self.path == '/stalled-download':
                    # Supply enough bytes for the browser to start the download,
                    # but never let the transfer finish during the test.
                    self.wfile.write(data)
                    self.wfile.flush()
                    if release.wait(10):
                        self.wfile.write(TEXT)
                else:
                    self.wfile.write(data)  # /broken sends fewer bytes than its Content-Length.
            except (BrokenPipeError, ConnectionResetError):
                pass

    with ThreadingHTTPServer(('127.0.0.1', 0), Documents) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}', hits, pdf, denied_pdf, iframe_started, iframe_release, flaky_healthy
        finally:
            release.set()
            iframe_release.set()
            server.shutdown()
            thread.join()


def own_sources(root):
    return {path for path in root.rglob('*') if path.is_file()}


def check_delayed_failure_retry(ctx, base, hits, root, navigate, check_download, monkeypatch):
    from ouroboros.tools import browser_session

    # Time out before headers; let the native failure arrive between tool calls.
    generation = ctx.browser_state
    page = generation.page
    owner = browser_session.owner_of(generation)
    endpoint = '/retry-after-timeout'
    target = base + endpoint
    first_request, failures = [], []
    original_answer = browser_session._answer

    async def observe_events():
        unblock, settled = asyncio.Event(), asyncio.Event()
        def failed(request):
            if first_request and request is first_request[0]:
                failures.append((request.failure, request._impl_obj._response))
                settled.set()

        page.context.on('requestfailed', failed)
        return unblock, settled, failed

    unblock, settled, failed_listener = owner.run(observe_events())

    async def settle():
        unblock.set()
        try:
            await asyncio.wait_for(settled.wait(), 5)
        finally:
            page.context.remove_listener('requestfailed', failed_listener)

    async def delayed_failure(route, reason, owner_generation, *, between_calls):
        if not reason and route.request.url == target and not first_request:
            first_request.append(route.request)
            await unblock.wait()
            await route.abort('addressunreachable')
            return
        await original_answer(route, reason, owner_generation, between_calls=between_calls)

    begin, sources = len(hits), own_sources(root)
    with monkeypatch.context() as mock:
        mock.setattr(browser_session, '_answer', delayed_failure)
        try:
            failure = navigate(endpoint, timeout=1000)
            assert failure.status == 'error' and 'Page.goto: Timeout' in failure.text, failure.text
            assert first_request and not failures
            assert 'Absolute path:' not in failure.text and own_sources(root) == sources
        finally:
            if first_request:
                owner.run(settle())
    assert len(failures) == 1 and failures[0][0] and failures[0][1] is None
    assert len(hits) == begin and ctx.browser_state is generation
    check_download(endpoint, [endpoint], '.txt', RECOVERED, 'text/plain')
    assert ctx.browser_state is generation and generation.page is page


@pytest.mark.parametrize('name,mime,reader', [
    ('notes.zip', 'application/zip', 'read_archive('),
    ('opaque.bin', 'application/octet-stream', None),
])
def test_receipt_only_recommends_applicable_readers(tmp_path, name, mime, reader):
    path = tmp_path / 'completed-browser-file'
    path.write_bytes(TEXT)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / 'data', task_id='a4018002')
    candidate = browser_download.Candidate(None, ('fixture',), 200, mime, len(TEXT), None)
    report, ref = browser_download.retain(ctx, path, candidate, name, time.monotonic() + 5, lambda: True)
    assert 'Absolute path:' in report and hashlib.sha256(TEXT).hexdigest() == ref['sha256']
    suggestions = [line for line in report.splitlines() if line.startswith(('read_', 'extract_'))]
    if reader:
        assert len(suggestions) == 1 and suggestions[0].startswith(reader)
        assert ref['read']['tool'] + '(' == reader
    else:
        assert not suggestions
        assert 'read' not in ref


@pytest.mark.parametrize('mime,raw,readable', [
    ('text/plain; boundary=x; Charset="UTF-16"', 'A short note.\n'.encode('utf-16'), False),
    ('text/plain; charset=utf-16le', 'A short note.\n'.encode('utf-16le'), False),
    ('text/plain', 'A short note.\n'.encode('utf-16'), False),
    ('text/plain; charset=utf-8', b'Lead byte at end: \xc3', False),
    ('text/plain; charset="UtF-8"', 'A café note.\n'.encode(), True),
])
def test_read_file_receipt_describes_the_encoding_of_retained_bytes(tmp_path, mime, raw, readable):
    path = tmp_path / 'completed-browser-file'
    path.write_bytes(raw)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / 'data', task_id='a4018004')
    candidate = browser_download.Candidate(None, ('fixture',), 200, mime, len(raw), None)
    report, ref = browser_download.retain(ctx, path, candidate, 'note.txt', time.monotonic() + 5, lambda: True)
    assert ref['sha256'] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / 'data' / 'task_results' / 'artifacts' / ctx.task_id / ref['path']).read_bytes() == raw
    if readable:
        registry = ToolRegistry(ctx.repo_dir, ctx.drive_root)
        registry.set_context(ctx)
        result = registry.execute_result('read_file', ref['read']['arguments'])
        assert result.status == 'ok' and 'A café note.' in result.text
    else:
        assert 'read' not in ref and 'read_file(' not in report
        assert 'read_file requires UTF-8' in report


def test_reader_denial_precedes_publication_and_keeps_existing_copy(tmp_path, monkeypatch):
    from ouroboros.tools import registry_core

    path = tmp_path / 'browser-file'
    path.write_bytes(TEXT)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / 'data', task_id='a4018003')
    candidate = browser_download.Candidate(None, ('fixture',), 200, 'text/plain', len(TEXT), None)
    store_root = tmp_path / 'data' / 'task_results' / 'artifacts' / ctx.task_id

    def reject_new(_ctx, binding):
        assert binding.target_path.name == f'browser-download-{hashlib.sha256(TEXT).hexdigest()}.txt'
        assert not binding.target_path.exists()
        return False

    def keep():
        return browser_download.retain(ctx, path, candidate, 'notes.txt', time.monotonic() + 3, lambda: True)

    with monkeypatch.context() as mock:
        mock.setattr(registry_core, '_presence_binding_allowed', reject_new)
        with pytest.raises(browser_download.DownloadError, match='no readable address'):
            keep()
        assert not own_sources(store_root)
    _, ref = keep()
    published = store_root / ref['path']
    before = published.stat()
    monkeypatch.setattr(registry_core, '_presence_binding_allowed', lambda *_: False)
    with pytest.raises(browser_download.DownloadError, match='no readable address'):
        keep()
    assert own_sources(store_root) == {published}
    assert published.read_bytes() == TEXT and published.stat().st_ino == before.st_ino


@pytest.mark.browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_direct_download_retention_and_readers(tmp_path, monkeypatch, local_documents, engine):
    pytest.importorskip('playwright.async_api')
    carried = os.environ.get('OUROBOROS_PINNED_NODE_CARRIED')
    if engine == 'chromium' and carried:
        # Exercise the operator's relative override using the same test runtime as the pinned lane.
        monkeypatch.delenv('OUROBOROS_PLAYWRIGHT_NODE_SELECTED', raising=False)
        monkeypatch.setenv('PLAYWRIGHT_NODEJS_PATH', 'node')
        monkeypatch.setenv('PATH', os.path.dirname(carried) + os.pathsep + os.environ.get('PATH', ''))
    required = os.environ.get('OUROBOROS_EXPECT_BROWSER_ENGINES', '').split(',')
    base, hits, pdf, denied_pdf, iframe_started, iframe_release, flaky_healthy = local_documents
    canonical = tmp_path / 'data'
    drive = canonical / 'state' / 'headless_tasks' / 'a4018000' / 'data'
    with store.open_store(drive, write=True):
        pass
    def tools_for(execution_drive):
        context = ToolContext(repo_dir=tmp_path, drive_root=execution_drive, task_id='a4018000',
                              budget_drive_root=canonical,
                              task_constraint=TaskConstraint(mode='local_readonly_subagent', allow_enable=False))
        tools = ToolRegistry(context.repo_dir, execution_drive)
        tools.set_context(context)
        return context, tools

    ctx, registry = tools_for(drive)
    root = canonical / 'task_results' / 'artifacts' / 'a4018000'
    readers = {}

    def navigate(endpoint, **kwargs):
        return registry.execute_result('browse_page', {'url': base + endpoint, 'engine': engine, **kwargs})

    def check_download(endpoint, expected_hits, suffix, content, mime):
        begin = len(hits)
        result = navigate(endpoint, output='screenshot')
        assert result.status == 'ok', result.text
        assert hits[begin:] == expected_hits
        assert mime in result.text
        content_hash = hashlib.sha256(content).hexdigest()
        assert content_hash in result.text
        path, = [p for p in own_sources(root) if str(p) in result.text and p.suffix == suffix]
        assert path.read_bytes() == content
        ref = result.meta['browser_download_source']
        assert ref['path'] == path.relative_to(root).as_posix()
        assert (ref['sha256'], ref['size']) == (content_hash, len(content))
        readers[path] = ref.get('read')
        if readers[path] is None:
            assert 'PDF retained whole;' in result.text and 'FILE_TOO_LARGE' in result.text
        else:
            assert readers[path]['tool'] + '(' in result.text
            assert readers[path]['arguments']['path'] == str(path)
        return path

    def check_original_document(outcome):
        assert ctx.browser_state is generation and generation.page is original
        result = registry.execute_result('browser_action', {
            # A goto timeout can return before the native navigation settles.
            'action': 'wait', 'selector': 'text=HTML_CONTROL', 'state': 'visible', 'timeout': 3000,
        })
        assert result.status == 'ok', result.text
        observed = json.loads(result.text)
        assert observed['status'] == 'reached' and observed['url'] == base + '/html', (outcome.text, result.text)
        assert observed['first_visible'] is True
        assert ctx.browser_state is generation and generation.page is original

    try:
        html = navigate('/html')
        if 'HTML_CONTROL' not in html.text and (
                chromium_runtime.UNAVAILABLE in html.text or "Executable doesn't exist" in html.text
                or 'not already installed' in html.text):
            if {'all', engine}.isdisjoint(required):
                pytest.skip(html.text)
        assert html.status == 'ok' and 'HTML_CONTROL' in html.text, html.text
        assert hits == ['/html'] and not own_sources(root)
        generation = ctx.browser_state
        if engine == 'chromium' and carried:
            assert os.path.samefile(generation._pw_driver.node_executable, carried)
        original = generation.page
        direct = check_download('', ['/'], '.pdf', pdf, 'application/pdf')
        assert direct == check_download('/file%20name.pdf#mark', ['/file%20name.pdf'], '.pdf', pdf, 'application/pdf')
        assert direct == check_download("/pdf?author=O'Reilly", ['/pdf?author=O%27Reilly'], '.pdf', pdf, 'application/pdf')
        assert direct == check_download('/start', ['/start', '/pdf'], '.pdf', pdf, 'application/pdf')
        assert direct == check_download('/inline', ['/inline'], '.pdf', pdf, 'application/pdf')
        text = check_download('/text', ['/text'], '.txt', TEXT, 'text/plain')
        assert text == check_download('/annual^report.txt', ['/annual%5Ereport.txt'], '.txt', TEXT, 'text/plain')
        assert text == check_download('/annual%5Ereport.txt', ['/annual%5Ereport.txt'], '.txt', TEXT, 'text/plain')
        assert navigate('/with-frame').status == 'ok'
        assert iframe_started.wait(2)
        try:
            assert text == check_download('/text', ['/text'], '.txt', TEXT, 'text/plain')
        finally:
            iframe_release.set()
        assert navigate('/with-link').status == 'ok'
        assert registry.execute_result('browser_action', {'action': 'click', 'selector': 'a'}).status == 'ok'
        assert text == check_download('/clicked', ['/clicked'], '.txt', TEXT, 'text/plain')
        from ouroboros.tools import browser_session

        late_download = threading.Event()

        async def observe_late_download():
            original.on('download', lambda d: late_download.set() if d.url == base + '/slow' else None)

        browser_session.owner_of(generation).run(observe_late_download())
        original_answer = browser_session._answer

        check_delayed_failure_retry(ctx, base, hits, root, navigate, check_download, monkeypatch)

        for endpoint, wire_url in [('/retry-file', '/retry-file'),
                                   ("/retry-file?author=O'Reilly#part", '/retry-file?author=O%27Reilly')]:
            async def fail_fixture_connection(route, reason, owner_generation, *, between_calls, target=base + wire_url):
                if reason or route.request.url != target:
                    return await original_answer(route, reason, owner_generation, between_calls=between_calls)
                await route.abort('addressunreachable')  # Native SDK fault on this allowed fixture only.

            begin, sources = len(hits), own_sources(root)
            with monkeypatch.context() as mock:
                mock.setattr(browser_session, '_answer', fail_fixture_connection)
                failed = navigate(endpoint)
            assert failed.status == 'error' and 'Page.goto:' in failed.text, failed.text
            assert 'Absolute path:' not in failed.text and own_sources(root) == sources
            assert len(hits) == begin and ctx.browser_state is generation
            recovered = check_download(endpoint, [wire_url], '.txt', RECOVERED, 'text/plain')
            assert ctx.browser_state is generation and generation.page is original
        first = navigate('/flaky')
        assert first.status == 'error' and hits.count('/flaky') >= 1, (first.text, hits)
        flaky_healthy.set()
        assert text == check_download('/flaky', ['/flaky'], '.txt', TEXT, 'text/plain')
        begin = len(hits)
        html = navigate('/html')
        assert html.status == 'ok' and 'HTML_CONTROL' in html.text, (html.text, hits)
        assert hits[begin:] == ['/html'] and ctx.browser_state is generation and original.url == base + '/html'
        large = check_download('/large', ['/large'], '.pdf', pdf + b'Z' * (LARGE_BYTES - len(pdf)), 'application/pdf')
        before = own_sources(root)
        assert before == {direct, text, recovered, large}
        assert not own_sources(drive / 'task_results' / 'artifacts' / 'a4018000')
        assert readers[large] is None

        normal = browser_policy.browser_url_block_reason

        def predicate(url, context, **kwargs):
            if url == base + '/middle':
                return 'BROWSER_LOCAL_READONLY_BLOCKED: synthetic fixture decision'
            return normal(url, context, **kwargs)

        with monkeypatch.context() as mock:
            mock.setattr(browser_policy, 'browser_url_block_reason', predicate)
            denied = navigate('/denied')
        assert denied.status == 'blocked' and 'synthetic fixture decision' in denied.text
        assert str(direct) not in denied.text
        assert pdf != denied_pdf and own_sources(root) == before
        assert not any(path.stat().st_size == len(denied_pdf) and path.read_bytes() == denied_pdf
                       for path in own_sources(root))
        check_original_document(denied)

        from playwright.async_api import Download

        cancelled_paths = []
        native_path = Download.path

        async def observe_path(download):
            try:
                return await native_path(download)
            except asyncio.CancelledError:
                cancelled_paths.append(download.url)
                raise

        with monkeypatch.context() as mock:
            mock.setattr(Download, 'path', observe_path)
            loading = navigate('/slow', timeout=250)
            assert loading.status == 'error' and 'Page.goto: Timeout' in loading.text, loading.text
            assert not cancelled_paths and 'Absolute path:' not in loading.text and own_sources(root) == before
            check_original_document(loading)
            assert late_download.wait(5)
            assert text == check_download('/slow', ['/slow'], '.txt', TEXT, 'text/plain')

            begin = len(hits)
            stalled = navigate('/stalled-download', timeout=2500)
            assert stalled.status == 'error' and 'BROWSER_DOWNLOAD_FAILED:' in stalled.text, stalled.text
            assert 'within the navigation timeout' in stalled.text, stalled.text
            assert cancelled_paths == [base + '/stalled-download'] and hits[begin:] == ['/stalled-download']
            assert 'Absolute path:' not in stalled.text and own_sources(root) == before
            check_original_document(stalled)

        for endpoint in ('/broken', '/missing.pdf'):
            outcome = navigate(endpoint)
            assert outcome.status == 'error', (endpoint, outcome.text)
            assert 'Absolute path:' not in outcome.text and own_sources(root) == before
    finally:
        browser.cleanup_browser(ctx)

    assert own_sources(root) == before
    def confirm_readers(tools):
        pdf_text = tools.execute_result(readers[direct]['tool'], readers[direct]['arguments'])
        assert pdf_text.status == 'ok' and PDF_LABEL in pdf_text.text, pdf_text.text
        result = tools.execute_result(readers[text]['tool'], readers[text]['arguments'])
        assert result.status == 'ok' and TEXT.decode().strip() in result.text, result.text
        too_large = tools.execute_result('extract_pdf_text', {'path': str(large)})
        assert too_large.status == 'error' and 'FILE_TOO_LARGE' in too_large.text, too_large.text

    confirm_readers(registry)
    # The execution-drive folder is unique to this test, not a production drive.
    shutil.rmtree(drive)
    assert not drive.exists()
    _, canonical_registry = tools_for(canonical)
    confirm_readers(canonical_registry)
