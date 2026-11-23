"""Offline window lifecycle with a fake WinForms/WebView2 control, on the default lane."""
from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import pytest

from ouroboros import context_report_archive, launcher_appearance, launcher_context_report as report
from ouroboros import launcher_page_zoom, launcher_window_chrome


class Signal:
    def __init__(self, ready=False, on_wait=None):
        self.ready, self.on_wait, self.handlers = ready, on_wait, []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self  # a subscriber does not receive events that already happened

    def is_set(self):
        return self.ready

    def set(self):
        self.ready = True
        for handler in self.handlers:
            handler()

    def wait(self, _seconds):
        if self.on_wait is not None:
            self.on_wait()
        return self.ready


class NavigationEvents(list):
    def __iadd__(self, handler):
        self.append(handler)
        return self


class Core:
    def __init__(self, window, scenario):
        self.Source = 'blank-unverified' if scenario == 'wrong_start' else 'about:blank'
        self.Settings = SimpleNamespace()
        self.NavigationStarting = NavigationEvents()
        self.window, self.scenario, self.html, self.events = window, scenario, None, []

    def start(self, uri):
        event = SimpleNamespace(Uri=uri, Cancel=False)
        for handler in self.NavigationStarting:
            handler(self, event)
        self.events.append((uri, event.Cancel))
        return event.Cancel

    def NavigateToString(self, html):
        self.html = html
        self.window.visual.append(('navigate', self.window.native.webview.ZoomFactor))
        uri = 'about:blank'
        if self.scenario == 'encoded':
            uri = 'data:text/html;charset=utf-8;base64,' + base64.b64encode(html.encode()).decode()
        if self.scenario == 'wrong_document':
            uri = 'data:text/plain;base64,QQ=='
        cancelled = self.start(uri)
        self.window.initial_read = None if cancelled or self.scenario == 'not_presented' else self.window.api.rooms()
        if self.scenario == 'close_after_navigation':
            self.window.events.closed.set()


class FakeWindow:
    def __init__(self, kwargs, scenario, destroy):
        self.api, self.scenario, self.destroy_mode = kwargs['js_api'], scenario, destroy
        self.api._presented = Signal()  # make the timeout deterministic; no sleeps or actual Windows processes
        self.events = SimpleNamespace(closed=Signal(), loaded=Signal(scenario != 'blank_timeout'))
        if scenario == 'close_while_loading':
            self.events.loaded.on_wait = self.events.closed.set
        self.core = Core(self, scenario)
        self.visual = []
        control = SimpleNamespace(CoreWebView2=self.core, ZoomFactor=1.0, AllowExternalDrop=True)
        self.native = SimpleNamespace(browser=SimpleNamespace(webview=control), webview=control,
                                      Handle=SimpleNamespace(ToInt64=lambda: 1), Invoke=self.invoke)
        self.background_color = kwargs.get('background_color')
        self.destroy_calls = 0

    def invoke(self, callback):
        if self.scenario == 'invoke_error':
            raise RuntimeError('synthetic invoke failure')
        callback()

    def destroy(self):
        self.destroy_calls += 1
        assert not self.api._enabled.is_set()  # revoke before attempting native destruction
        if self.destroy_mode == 'raises':
            raise RuntimeError('synthetic destroy failure')
        if self.destroy_mode == 'closes':
            self.events.closed.set()


class FakeWebview:
    guilib = SimpleNamespace(renderer='edgechromium')

    def __init__(self, scenario='blank', destroy='closes'):
        self.scenario, self.destroy, self.windows = scenario, destroy, []

    def create_window(self, _title, **kwargs):
        if self.scenario == 'create_error':
            raise RuntimeError('synthetic create failure')
        window = FakeWindow(kwargs, self.scenario, self.destroy)
        self.windows.append(window)
        if self.scenario == 'early_close':
            window.events.closed.set()
        return window


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import threading

    # Only fake System.Action / GUI writes; even rooms() reads strictly synthetic data.
    monkeypatch.setattr(report, '_window_lock', threading.Lock())
    monkeypatch.setitem(sys.modules, 'System', SimpleNamespace(Action=lambda callback: callback))
    monkeypatch.setattr(context_report_archive, 'rooms', lambda root: [{'chat_id': 23, 'title': root.name}])
    monkeypatch.setattr(launcher_page_zoom, '_backend', lambda: 'edgechromium')
    monkeypatch.setattr(launcher_window_chrome, 'host_palette', lambda: 'dark')
    monkeypatch.setattr(launcher_window_chrome, '_fit', lambda _form: None)
    monkeypatch.setattr(launcher_appearance, '_WINDOWS', True)
    monkeypatch.setattr(launcher_appearance, '_high_contrast', lambda: False)
    painted = []
    monkeypatch.setattr(launcher_appearance, '_set_dark_frame', lambda hwnd, dark: painted.append((hwnd, dark)) or 0)
    monkeypatch.setattr(launcher_appearance, '_refresh_frame', lambda hwnd: painted.append(hwnd))
    root = tmp_path / 'synthetic'; root.mkdir()
    html = tmp_path / 'report.html'
    html.write_text('<html lang="en"><title>Fixture</title></html>', encoding='utf-8')
    return root, html, painted


@pytest.mark.parametrize('wire', ['blank', 'encoded'])
def test_window_opens_only_after_fenced_document_requests_rooms(isolated, wire):
    root, html, painted = isolated
    (root / 'state').mkdir()
    size = root / 'state' / launcher_page_zoom.STATE_NAME
    size.write_text('{"percent":125}', encoding='utf-8')
    webview = FakeWebview(wire)
    assert report.open_local_report(webview, root, html) == {'state': 'opened'}
    window, = webview.windows
    assert 'data-private-archive' in window.core.html
    assert window.initial_read == {'ok': True, 'value': [{'chat_id': 23, 'title': 'synthetic'}]}
    assert len(window.core.events) == 1 and not window.core.events[0][1]
    assert window.core.start(window.core.events[0][0])  # even another matching navigation is denied
    assert window.core.start('about:blank#unrelated')
    assert not any(vars(window.core.Settings).values()) and not window.native.webview.AllowExternalDrop
    assert window.background_color == launcher_window_chrome.ROOM['dark']
    assert window.visual == [('navigate', 1.25)] and size.read_text(encoding='utf-8') == '{"percent":125}'
    assert painted == [(1, True), 1]
    assert report.open_local_report(webview, root, html) == {'state': 'already_open'}
    window.events.closed.set()
    assert not window.api.rooms()['ok']
    assert not report._window_lock.locked()


@pytest.mark.parametrize(('scenario', 'destroy', 'destroy_calls'), [
    ('create_error', 'closes', 0), ('early_close', 'closes', 0), ('close_while_loading', 'closes', 0),
    ('close_after_navigation', 'closes', 0), ('blank_timeout', 'closes', 1), ('invoke_error', 'closes', 1),
    ('wrong_start', 'closes', 1), ('wrong_document', 'closes', 1), ('not_presented', 'closes', 1),
    ('not_presented', 'silent', 1), ('not_presented', 'raises', 1),
])
def test_failed_window_releases_only_its_lease(isolated, scenario, destroy, destroy_calls):
    root, html, _ = isolated
    failed = FakeWebview(scenario, destroy)
    assert report.open_local_report(failed, root, html) == {'state': 'unavailable'}
    assert not report._window_lock.locked()
    assert len(failed.windows) == (0 if scenario == 'create_error' else 1)
    old = failed.windows[0] if failed.windows else None
    if old is not None:
        assert not old.api.rooms()['ok']
        assert old.destroy_calls == destroy_calls
        if scenario in {'early_close', 'close_while_loading'}:
            assert old.core.html is None
        if destroy != 'closes':
            assert not old.events.closed.is_set()  # no native callback released this lease
        if scenario == 'wrong_document':
            assert old.core.events == [('data:text/plain;base64,QQ==', True)]
            assert old.core.start('about:blank')  # rejection also spent the one-shot navigation
    healthy = FakeWebview()
    assert report.open_local_report(healthy, root, html) == {'state': 'opened'}
    current, = healthy.windows
    assert current.visual == [('navigate', 1.0)]
    if old is not None:
        if old.core.NavigationStarting:
            assert old.core.start('about:blank')
        old.events.closed.set()  # a late or duplicate callback no longer owns the global lease
        assert not old.api.rooms()['ok']
    assert report.open_local_report(healthy, root, html) == {'state': 'already_open'}
    current.events.closed.set()
    assert not report._window_lock.locked()


def test_unavailable_renderer_or_invalid_document_never_creates_window(isolated):
    root, html, _ = isolated
    wrong = FakeWebview()
    wrong.guilib = SimpleNamespace(renderer='cocoa')
    assert report.open_local_report(wrong, root, html) == {'state': 'unavailable'}
    good = FakeWebview()
    assert report.open_local_report(good, root, root / 'missing.html') == {'state': 'unavailable'}
    html.write_text('<html><title>No archive marker</title></html>', encoding='utf-8')
    assert report.open_local_report(good, root, html) == {'state': 'unavailable'}
    assert not good.windows and not wrong.windows and not report._window_lock.locked()
