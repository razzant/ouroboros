"""Beautiful Context's offline Windows window, entered from the page a person uses.

The opt-in case runs the shipped interface in a real WebView2 window, presses Dashboard > Logs "Open recent
requests" and works the separate viewer over an invented archive. The default-lane test below checks, on any OS,
what that case needs and no window: the page and bridge it serves.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = 'OUROBOROS_CONTEXT_REPORT_QA_ARTIFACT_DIR'
ENTRY = 'Open recent requests'
ROOM_A = 'Tide charts'
ROOM_B = 'Library hours'
A_MARKERS = ('TIDE-FIXTURE-ZERO', 'TIDE-FIXTURE-UNREPORTED', 'TIDE-FIXTURE-OLDER')
B_MARKER = 'LIBRARY-FIXTURE-SINGLE'


def _seed(root, monkeypatch):
    from ouroboros import context_report_archive as archive
    from ouroboros import dialogue_store, projects_registry, usage_accounting
    from tests import _context_reports as reports

    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setenv('OUROBOROS_SETTINGS_PATH', str(root / 'settings.json'))
    monkeypatch.setenv('TOTAL_BUDGET', '100')
    monkeypatch.setattr(usage_accounting, 'estimate_cost_optional', lambda *args, **kwargs: 0.01)
    (root / 'state').mkdir(parents=True)
    a = projects_registry.create_project(root, 'tide-charts', name=ROOM_A)['chat_id']
    b = projects_registry.create_project(root, 'library-hours', name=ROOM_B)['chat_id']
    for chat_id, task_id in [(a, 'tide-fixture'), (b, 'library-fixture'), (1, 'main-fixture')]:
        dialogue_store.append(root, 'chat', {'chat_id': chat_id, 'task_id': task_id, 'direction': 'out'})

    def send(task_id, marker, usage):
        payload = {'model': 'gpt-x', 'messages': [
            {'role': 'system', 'content': 'Use the invented coastal café schedule.'},
            {'role': 'user', 'content': marker},
            {'role': 'assistant', 'content': 'The fixture conversation ends here.'},
        ], 'max_tokens': 32}
        reply = {'choices': [{'message': {'role': 'assistant', 'content': 'Fixture response'}}]}
        if usage is not None:
            reply['usage'] = usage
        with monkeypatch.context() as patch:
            patch.setattr(reports._Response, '_payload', reply)
            return reports._dispatch(root, task_id, payload=payload)['attempt_id']

    # A readable old-style record has no provider-token witness. Produce the same
    # shape via a synthetic answer without usage; do not edit the usage database.
    older = send('tide-fixture', A_MARKERS[2], None)
    unreported = send('tide-fixture', A_MARKERS[1], {'prompt_tokens': 50, 'completion_tokens': 2, 'cost': 0.0})
    zero = send('tide-fixture', A_MARKERS[0], {
        'prompt_tokens': 50, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}, 'cost': 0.0,
    })
    single = send('library-fixture', B_MARKER, {'prompt_tokens': 33, 'completion_tokens': 3, 'cost': 0.0})

    selection, a_rows = archive.recent_requests(root, a)
    _, b_rows = archive.recent_requests(root, b)
    assert [row['attempt_id'] for row in a_rows] == [zero, unreported, older], a_rows
    assert [row['attempt_id'] for row in b_rows] == [single], b_rows
    saved = [json.loads(archive.document(root, selection, attempt)) for attempt in (zero, unreported, older)]
    zero_usage = saved[0]['facts']['provider_token_receipt']
    missing_usage = saved[1]['facts']['provider_token_receipt']
    assert zero_usage['cached_tokens'] == 0
    assert missing_usage['prompt_tokens'] == 50 and 'cached_tokens' not in missing_usage
    assert 'provider_token_receipt' not in saved[2]['facts']
    # Derive the oracle from the two sealed bodies before opening the UI. This
    # fixture changes actual message contents, not just JSON whitespace/order.
    left, right = (entry['stored_json'] for entry in saved[:2])
    first, second = (json.loads(body)['messages'] for body in (left, right))
    assert len(first) == len(second) and first != second
    item = next(i for i, pair in enumerate(zip(first, second), 1) if pair[0] != pair[1])
    assert (first[item - 1]['content'], second[item - 1]['content']) == A_MARKERS[:2]
    prefix = next(i for i, (lbyte, rbyte) in enumerate(zip(left.encode('utf-8'), right.encode('utf-8')))
                  if lbyte != rbyte)
    return {'a': a, 'b': b, 'zero': zero, 'unreported': unreported, 'older': older, 'single': single,
            'item': item, 'prefix': prefix}


# Query the DOM, not the bridge or the page's internal JS variables. The viewer's
# strict CSP does not allow pywebview evaluate_js's eval wrapper; run_js on the
# pinned Edge backend runs the expression directly and returns its decoded value.
_SNAPSHOT = r'''(() => {
  const get = id => document.getElementById(id);
  const fields = id => Object.fromEntries([...get(id).querySelectorAll('dt')]
    .map(term => [term.textContent, term.nextElementSibling.textContent]));
  return {
    title: document.title,
    privateDocument: document.documentElement.hasAttribute('data-private-archive'),
    roomLabels: [...get('archive-room').options].map(node => node.textContent),
    room: get('archive-room').value,
    attempts: [...get('archive-attempt').options].map(node => node.value),
    selected: get('archive-attempt').value,
    shownAttempt: fields('attempt-facts').Attempt ?? null,
    reportVisible: !get('report-result').hidden,
    importHidden: get('report-opening').hidden,
    compareDisabled: get('archive-compare').disabled,
    comparisonHidden: get('comparison').hidden,
    compared: get('comparison').open,
    identitiesHidden: get('comparison-technical').hidden,
    identities: fields('comparison-identities'),
    comparison: fields('comparison-facts'),
    comparisonStatus: get('comparison-status').textContent,
    build: fields('build-facts'),
    reportStatus: get('report-status').textContent,
    request: get('request').textContent,
    dom: document.body.textContent,
    resources: performance.getEntriesByType('resource').map(entry => entry.name),
  };
})()'''


def _await_page(window, description, ready, script=_SNAPSHOT, seconds=12):
    deadline = time.monotonic() + seconds
    seen = None
    while time.monotonic() < deadline:
        seen = window.run_js(script)
        if isinstance(seen, dict) and ready(seen):
            return seen
        time.sleep(0.08)
    summary = ({**seen, 'dom': '<omitted>'} if isinstance(seen, dict) and 'dom' in seen else seen)
    raise AssertionError(f'{description}: last DOM observation {summary!r}')


def _select(window, element, value):
    window.run_js(f'''(() => {{
        const select = document.getElementById({json.dumps(element)});
        select.value = {json.dumps(str(value))};
        select.dispatchEvent(new Event('change', {{bubbles: true}}));
    }})()''')


def _settings(window):
    from System import Action

    facts = {}

    def observe():
        control = window.native.browser.webview
        settings = control.CoreWebView2.Settings
        facts.update(devtools=bool(settings.AreDevToolsEnabled),
                     context_menus=bool(settings.AreDefaultContextMenusEnabled),
                     accelerators=bool(settings.AreBrowserAcceleratorKeysEnabled),
                     external_drop=bool(control.AllowExternalDrop))

    window.native.Invoke(Action(observe))
    return facts


def _screenshot(window, path):
    from Microsoft.Web.WebView2.Core import CoreWebView2CapturePreviewImageFormat
    from System import Action, Convert
    from System.IO import MemoryStream

    # Ask the native control to render, then wait off the GUI thread for the PNG.
    capture = []
    stream = MemoryStream()
    try:
        def begin():
            core = window.native.browser.webview.CoreWebView2
            capture.append(core.CapturePreviewAsync(CoreWebView2CapturePreviewImageFormat.Png, stream))

        window.native.Invoke(Action(begin))
        assert capture[0].Wait(10000), f'WebView2 CapturePreviewAsync timed out: {path.name}'
        png = base64.b64decode(str(Convert.ToBase64String(stream.ToArray())))
        assert png.startswith(b'\x89PNG\r\n\x1a\n') and len(png) > 1000, (path.name, len(png))
        path.write_bytes(png)
    finally:
        stream.Dispose()


def _exercise(window, fixture, artifacts):
    def selected(attempt):
        return lambda page: page['reportVisible'] and page['selected'] == page['shownAttempt'] == attempt

    initial = _await_page(window, 'the fenced viewer lists conversations', lambda p: len(p['roomLabels']) >= 3)
    assert initial['title'] == 'Beautiful Context' and initial['privateDocument'] and initial['importHidden'], initial
    assert set(initial['roomLabels']) == {'Main', ROOM_A, ROOM_B}, initial['roomLabels']
    settings = _settings(window)
    assert settings and not any(settings.values()), settings

    _select(window, 'archive-room', fixture['a'])
    a = _await_page(window, 'project A opens its latest saved request', selected(fixture['zero']))
    assert a['attempts'] == [fixture[key] for key in ('zero', 'unreported', 'older')], a['attempts']
    assert not a['compareDisabled'] and A_MARKERS[0] in a['request'], a['request']
    assert a['comparisonHidden'] and not a['compared'], a
    assert a['build']['Provider token receipt'] == 'Recorded for this attempt', a['build']
    assert a['build']['Provider · Cache read'] == '0 tokens', a['build']

    _select(window, 'archive-attempt', fixture['unreported'])
    missing = _await_page(window, 'project A opens the reply without a cache-read count', selected(fixture['unreported']))
    assert A_MARKERS[1] in missing['request'] and missing['build']['Provider · Cache read'] == 'Not recorded', missing['build']
    assert missing['build']['Provider token receipt'] == 'Recorded for this attempt', missing['build']
    _select(window, 'archive-attempt', fixture['older'])
    older = _await_page(window, 'project A opens its readable older-style request', selected(fixture['older']))
    assert A_MARKERS[2] in older['request'] and 'Provider · Cache read' not in older['build'], older['build']
    assert older['build']['Provider token receipt'] == 'Not recorded for this attempt', older['build']

    _select(window, 'archive-attempt', fixture['zero'])
    _await_page(window, 'project A returns to its latest request', selected(fixture['zero']))
    window.run_js("document.getElementById('archive-compare').click()")
    compared = _await_page(window, 'project A displays its comparison', lambda p: bool(p['comparison']))
    _screenshot(window, artifacts / 'a-compared.png')
    expected = {
        'First difference · Messages': f"Item {fixture['item']} differs",
        'Stored request comparison': f"The copies differ · {fixture['prefix']:,} identical UTF-8 bytes at the start",
        'Top-level field changes': 'messages',
        'Provider cache read: first → second': '0 tokens → Not recorded',
        'Provider token receipt: first → second': 'Recorded → Recorded',
    }
    for label, value in expected.items():
        assert compared['comparison'].get(label) == value, (label, value, compared['comparison'])
    assert not compared['comparisonHidden'] and compared['compared']
    assert compared['comparisonStatus'].startswith('Selected request compared with Previous')
    assert compared['identities']['Compared task / attempt'] == f"tide-fixture / {fixture['unreported']}"
    assert compared['identities']['Selected task / attempt'] == f"tide-fixture / {fixture['zero']}"

    _select(window, 'archive-room', fixture['b'])
    b = _await_page(window, 'project B opens its only saved request', selected(fixture['single']))
    _screenshot(window, artifacts / 'b-after-switch.png')
    assert b['room'] == str(fixture['b']) and b['attempts'] == [fixture['single']], (b['room'], b['attempts'])
    assert B_MARKER in b['request'] and b['compareDisabled'], b['request']
    assert b['comparisonHidden'], b
    assert b['identitiesHidden'] and not (b['compared'] or b['comparison'] or b['identities'] or b['comparisonStatus']), b
    a_traces = (*A_MARKERS, 'tide-fixture', *(fixture[key] for key in ('zero', 'unreported', 'older')))
    retained = [trace for trace in a_traces if trace in b['dom'] or trace in b['attempts']]
    assert not retained, ('A data remained in B', retained)
    assert not any(p['resources'] for p in (initial, a, missing, older, compared, b)), b['resources']
    return {
        'fenced_document': initial['privateDocument'], 'native_settings': settings, 'rooms': initial['roomLabels'],
        'a_attempt_count': len(a['attempts']), 'a_oracle_item': fixture['item'], 'a_oracle_prefix_bytes': fixture['prefix'],
        'a_comparison': {key: compared['comparison'][key] for key in expected},
        'provider_zero': a['build']['Provider · Cache read'], 'provider_missing': missing['build']['Provider · Cache read'],
        'provider_older': older['build']['Provider token receipt'],
        'b_attempt_count': len(b['attempts']), 'b_selected': b['selected'], 'b_opened': b['shownAttempt'],
        'b_marker': B_MARKER, 'b_compare_disabled': b['compareDisabled'], 'b_comparison': b['comparison'],
        'b_a_traces': retained, 'resources': b['resources'],
    }


def _desktop_app():
    """The shipped page behind the core's own ``/`` and ``/static``, with only the answers its boot and Logs need.

    No Ouroboros server or skill runs: setup reads as done, Logs gets invented events and other API calls are 404s.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route, WebSocketRoute

    from ouroboros.server_web import ServedUI

    async def set_up(_request):
        return Response(status_code=204)  # no onboarding cover over the page

    log_rows = {
        'events': [
            {'type': 'task_started', 'task_id': 'log-invented-task', 'ts': '2026-01-03T12:00:01Z'},
            # Unbound events show their own kinds; a task's model and tool calls fold into its Tasks card.
            {'type': 'llm_usage', 'model': 'example-model', 'ts': '2026-01-03T12:00:02Z'},
            {'type': 'llm_api_error', 'task_id': 'log-invented-error', 'error': 'Example model error',
             'ts': '2026-01-03T12:00:03Z'},
        ],
        'tools': [{'type': 'tool_call_finished', 'tool': 'read_file',
                   'ts': '2026-01-03T12:00:04Z', 'is_error': False}],
        'progress': [{'type': 'task_progress', 'task_id': 'log-invented-wake', 'is_progress': True,
                      'initiator': 'consciousness', 'ts': '2026-01-03T12:00:05Z', 'text': 'Example wake'}],
        'supervisor': [{'type': 'startup', 'ts': '2026-01-03T12:00:06Z', 'message': 'Example host startup'}],
    }

    async def log_history(request):
        return JSONResponse({'entries': log_rows.get(request.path_params['name'], [])})

    async def quiet_socket(websocket):
        await websocket.accept()
        async for _message in websocket.iter_text():
            pass

    served = ServedUI(ROOT)
    return Starlette(routes=[Route('/', served.index_page), Route('/api/onboarding', set_up),
                             Route('/api/logs/{name}', log_history), WebSocketRoute('/ws', quiet_socket),
                             *served.mounts()])


@contextlib.contextmanager
def _desktop_page():
    import uvicorn

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(_desktop_app(), lifespan='off', log_level='warning'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started, 'the fixture page server did not start'
            yield f'http://127.0.0.1:{port}/'
        finally:
            server.should_exit = True
            thread.join(10)


def _observed_desktop_api():
    """The launcher's own page bridge, noting what the page passed the viewer opener and what it answered."""
    from ouroboros.launcher_background import DesktopApi

    class ObservedDesktopApi(DesktopApi):
        def __init__(self):
            self._calls, self._answers = [], []  # a leading underscore keeps them off the bridge

        def open_local_context_report(self, *arguments):
            self._calls.append(arguments)
            answer = super().open_local_context_report()
            self._answers.append(answer)
            return answer

    return ObservedDesktopApi()


# Read layout without revealing or focusing either action. Revealing a clipped button first would hide the regression.
_LOGS_ENTRY = r'''(label => {
  const said = node => (node.getAttribute('aria-label') || node.textContent).replace(/\s+/g, ' ').trim();
  const found = [...document.querySelectorAll('button, [role="button"]')].filter(node => said(node) === label);
  const rect = ({left, top, right, bottom}) => [left, top, right, bottom];
  const hit = (node, box) => node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
  const action = node => {
    const box = node.getBoundingClientRect();
    const range = document.createRange();
    range.selectNodeContents(node);
    const word = range.getBoundingClientRect();
    return {
      inLogs: !!node.closest('#dashboard-panel-logs, [data-dashboard-controls="logs"]'),
      reachable: box.width > 0 && box.height > 0 && hit(node, box) && node.tabIndex >= 0
        && !node.disabled && node.getAttribute('aria-disabled') !== 'true',
      oneLine: range.getClientRects().length === 1 && word.left >= box.left && word.right <= box.right
        && word.top >= box.top && word.bottom <= box.bottom,
      box: rect(box),
      trusted: node.dataset.qaTrustedClick === 'true',
    };
  };
  const clear = document.getElementById('btn-clear-logs');
  const rail = document.getElementById('log-filters');
  const list = document.getElementById('page-logs');
  const kinds = [...(rail?.querySelectorAll('button:not([hidden])') || [])];
  const tail = kinds.at(-1);
  return {
    page: document.querySelector('.page.active')?.id ?? null,
    tab: document.querySelector('[data-dashboard-tab][aria-selected="true"]')?.dataset.dashboardTab ?? null,
    entries: found.map(action),
    clear: clear && action(clear),
    rail: rail && rect(rail.getBoundingClientRect()),
    railOverflow: rail && rail.scrollWidth > rail.clientWidth,
    railScroll: rail?.scrollLeft,
    tailReachable: !!tail && hit(tail, tail.getBoundingClientRect()),
    kinds: kinds.map(node => node.dataset.logKind),
    list: list && rect(list.getBoundingClientRect()),
    listScroll: list?.scrollTop,
    openedNotice: [...document.querySelectorAll('#toast-stack [role="status"] .toast-words')]
      .some(node => node.textContent === 'Opened in a separate local window.'),
  };
})'''

# The sidebar's Dashboard row, pressed once; Logs is the page the Dashboard opens on.
_OPEN_DASHBOARD = r'''(() => {
  const page = document.querySelector('.page.active')?.id ?? null;
  const row = document.querySelector('[data-nav-page="dashboard"]');
  if (page !== 'page-dashboard' && row && !window.__qaOpenedDashboard) {
    window.__qaOpenedDashboard = true;
    row.click();
  }
  return {page};
})()'''


def _press_logs_entry(window, api, artifacts):
    """Reach Dashboard > Logs from the sidebar, press its viewer button and wait for the launcher's answer."""
    _await_page(window, 'the app opens its Dashboard', lambda p: p['page'] == 'page-dashboard', _OPEN_DASHBOARD,
                seconds=20)
    window.run_js('''document.querySelector('[data-dashboard-tab="logs"]').click()''')
    probe = f'{_LOGS_ENTRY}({json.dumps(ENTRY)})'
    offered = _await_page(
        window, f'Dashboard > Logs offers {ENTRY!r} beside populated filters',
        lambda p: p['tab'] == 'logs' and any(e['box'][2] > e['box'][0] for e in p['entries'])
        and p['kinds'] == ['', 'tools', 'llm', 'errors', 'tasks', 'system', 'consciousness'], probe, seconds=20)
    assert offered['railOverflow'], ('the populated filter strip did not exercise its scroll edge', offered)
    opener, = offered['entries']
    clear = offered['clear']
    for action in (opener, clear):
        assert action['inLogs'] and action['reachable'] and action['oneLine'], offered
    assert offered['rail'][2] <= opener['box'][0] and opener['box'][2] < clear['box'][0], offered
    assert abs(sum(opener['box'][1::2]) - sum(clear['box'][1::2])) <= 2, offered  # same center line
    assert not api._calls, ('the viewer was asked for before the press', api._calls)
    _screenshot(window, artifacts / 'logs-entry.png')

    window.run_js(r'''(() => {
      const rail = document.getElementById('log-filters');
      rail.scrollLeft = rail.scrollWidth;
      const button = document.querySelector('[data-local-context-report] button');
      button.addEventListener('click', event => { button.dataset.qaTrustedClick = String(event.isTrusted); }, {once: true});
    })()''')
    scrolled = _await_page(window, 'the last filter is reachable after scrolling',
                           lambda p: p['railScroll'] > 0 and p['tailReachable'], probe)
    _same_logs_geometry(offered, scrolled)
    _click_logs_entry(window, scrolled['entries'][0]['box'])
    answered = _await_page(window, 'the launcher opens the viewer and the page shows the answer',
                           lambda p: bool(api._answers) and p['openedNotice'], probe, seconds=25)
    assert api._calls == [()] and api._answers == [{'state': 'opened'}], (api._calls, api._answers)
    assert answered['entries'][0]['trusted'], answered
    assert answered['entries'][0]['reachable'] and answered['clear']['reachable'], answered
    assert answered['railScroll'] == scrolled['railScroll'], answered
    _same_logs_geometry(scrolled, answered)
    _screenshot(window, artifacts / 'logs-entry-answered.png')
    return {'page': offered['page'], 'tab': offered['tab'], 'button': ENTRY, 'arguments': list(api._calls[0]),
            'answer': api._answers[0], 'layout': {'before': offered, 'scrolled': scrolled, 'answered': answered}}


def _same_logs_geometry(before, after):
    def boxes(page):
        return (page['rail'], page['entries'][0]['box'], page['clear']['box'], page['list'])

    for left, right in zip(boxes(before), boxes(after)):
        assert all(abs(a - b) <= 1 for a, b in zip(left, right)), (before, after)
    assert before['listScroll'] == after['listScroll'], (before, after)


def _click_logs_entry(window, box):
    from System import Action

    # WebView2 native Chromium input at the measured CSS-pixel center: unlike HTMLElement.click(), it hit-tests.
    x1, y1, x2, y2 = box
    for kind, buttons in [('mousePressed', 1), ('mouseReleased', 0)]:
        pending = []
        args = json.dumps({'type': kind, 'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2,
                           'button': 'left', 'buttons': buttons, 'clickCount': 1})

        def begin(args=args, pending=pending):
            core = window.native.browser.webview.CoreWebView2
            pending.append(core.CallDevToolsProtocolMethodAsync('Input.dispatchMouseEvent', args))

        window.native.Invoke(Action(begin))
        assert pending[0].Wait(10000), f'WebView2 did not answer {kind}'


def _open_native(root, artifacts):
    import webview

    from ouroboros import config

    fixture = json.loads((root / 'fixture.json').read_text(encoding='utf-8'))
    # The launcher's opener reads these two roots from the environment this child inherited.
    roots = (config.DATA_DIR, config.REPO_DIR)
    assert Path(roots[0]).samefile(root) and Path(roots[1]).samefile(ROOT), roots
    finished = threading.Event()
    failed = []
    with _desktop_page() as url:
        api = _observed_desktop_api()
        app = webview.create_window('Windows desktop QA', url, js_api=api, width=1000, height=640,
                                    min_size=(800, 500), text_select=True)

        def run():
            try:
                assert app.events.loaded.wait(20), 'the app page did not load'
                renderer = webview.guilib.renderer
                print(f'App page loaded with renderer={renderer!r}', flush=True)
                assert renderer == 'edgechromium', renderer
                entry = _press_logs_entry(app, api, artifacts)
                viewer, = [w for w in webview.windows if w is not app]
                # The launcher answers only after the production page asked for rooms
                # behind its navigation fence.
                print('Dashboard > Logs opened the separate offline viewer', flush=True)
                facts = _exercise(viewer, fixture, artifacts)
                print('NATIVE_CONTEXT_REPORT ' + json.dumps({'renderer': renderer, 'entry': entry, **facts}), flush=True)
            except BaseException as exc:
                failed.append(str(exc))
                traceback.print_exc()
            finally:
                finished.set()
                for window in reversed(list(webview.windows)):  # the viewer first: the loop ends with the last one
                    window.destroy()

        webview.start(run, gui='edgechromium', private_mode=True)
    assert finished.is_set() and not failed, failed or 'pywebview returned before the probe finished'


def test_native_beautiful_context_switches_to_single_request_project(tmp_path, monkeypatch, record_property):
    if sys.platform != 'win32' or os.environ.get('OUROBOROS_WINDOWS_DESKTOP_QA') != '1':
        pytest.skip('real WebView2 desktop test: Windows and OUROBOROS_WINDOWS_DESKTOP_QA=1 required')
    from tests.group_run import run_in_group

    root = tmp_path / 'synthetic'
    artifacts = Path(os.environ.get(ARTIFACT_DIR, str(tmp_path / 'screenshots')))
    artifacts.mkdir(parents=True, exist_ok=True)
    fixture = _seed(root, monkeypatch)
    (root / 'fixture.json').write_text(json.dumps(fixture), encoding='utf-8')
    monkeypatch.setenv('OUROBOROS_REPO_DIR', str(ROOT))  # the launcher opens this checkout's web/context-report.html
    child = run_in_group([sys.executable, '-m', 'tests.test_context_report_native_window', str(root), str(artifacts)],
                         timeout=150, cwd=ROOT)
    print(child.stdout, end='')
    assert child.returncode == 0, (child.returncode, child.stderr)
    records = [json.loads(line.removeprefix('NATIVE_CONTEXT_REPORT ')) for line in child.stdout.splitlines()
               if line.startswith('NATIVE_CONTEXT_REPORT ')]
    observed, = records
    assert not list(root.rglob('SKILL.md')), 'this profile holds no skill: the entry needs none'
    for name in ('logs-entry.png', 'logs-entry-answered.png', 'a-compared.png', 'b-after-switch.png'):
        assert (artifacts / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), name
    record_property('native_context_report', json.dumps(observed, sort_keys=True))


def test_windows_case_serves_the_shipped_page_and_the_launchers_bridge(monkeypatch):
    """What the Windows case shows is the checkout's own page, bare of any skill, over the launcher's own opener."""
    from types import SimpleNamespace

    from starlette.testclient import TestClient

    client = TestClient(_desktop_app())
    assert '<script type="module" src="/static/app.js">' in client.get('/').text
    for name in ('app.js', 'modules/dashboard.js', 'modules/logs.js'):
        assert client.get(f'/static/{name}').content == (ROOT / 'web' / name).read_bytes(), name
    assert client.get('/api/onboarding').status_code == 204
    events = client.get('/api/logs/events', params={'view': 'compact', 'limit': 150}).json()['entries']
    assert [row['type'] for row in events] == ['task_started', 'llm_usage', 'llm_api_error']
    assert client.get('/api/widgets').status_code == 404  # no skill, so no card of one
    with client.websocket_connect('/ws') as socket_:
        socket_.send_text('{}')

    monkeypatch.setitem(sys.modules, 'webview', SimpleNamespace(guilib=SimpleNamespace(renderer='cocoa')))
    api = _observed_desktop_api()  # a bridge without a launcher behind it still answers the page's capability read
    assert api.shell_info()['local_context_archive'] is False
    assert api.open_local_context_report() == {'state': 'unavailable'}
    assert api._calls == [()] and api._answers == [{'state': 'unavailable'}]


if __name__ == '__main__':
    assert sys.platform == 'win32' and os.environ.get('OUROBOROS_WINDOWS_DESKTOP_QA') == '1'
    _open_native(Path(sys.argv[1]), Path(sys.argv[2]))
