"""Opt-in Windows smoke through the real WebView2 window and local request archive."""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = 'OUROBOROS_CONTEXT_REPORT_QA_ARTIFACT_DIR'
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


def _await_page(window, description, ready):
    deadline = time.monotonic() + 12
    seen = None
    while time.monotonic() < deadline:
        seen = window.run_js(_SNAPSHOT)
        if isinstance(seen, dict) and ready(seen):
            return seen
        time.sleep(0.08)
    summary = (dict(seen, dom='<omitted>') if isinstance(seen, dict) else seen)
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
    assert compared['compared'] and compared['comparisonStatus'].startswith('Selected request compared with Previous')
    assert compared['identities']['Compared task / attempt'] == f"tide-fixture / {fixture['unreported']}"
    assert compared['identities']['Selected task / attempt'] == f"tide-fixture / {fixture['zero']}"

    _select(window, 'archive-room', fixture['b'])
    b = _await_page(window, 'project B opens its only saved request', selected(fixture['single']))
    _screenshot(window, artifacts / 'b-after-switch.png')
    assert b['room'] == str(fixture['b']) and b['attempts'] == [fixture['single']], (b['room'], b['attempts'])
    assert B_MARKER in b['request'] and b['compareDisabled'], b['request']
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


def _open_native(root, artifacts):
    import webview

    from ouroboros.launcher_context_report import open_local_report

    fixture = json.loads((root / 'fixture.json').read_text(encoding='utf-8'))
    initial = webview.create_window('Windows desktop QA', html='<!doctype html><title>Desktop QA</title>', hidden=True)
    finished = threading.Event()
    failed = []

    def run():
        viewer = None
        try:
            assert initial.events.loaded.wait(12), 'pywebview bootstrap did not load'
            renderer = webview.guilib.renderer
            print(f'Bootstrap loaded with renderer={renderer!r}', flush=True)
            assert renderer == 'edgechromium', renderer
            opened = open_local_report(webview, root, ROOT / 'web' / 'context-report.html')
            assert opened == {'state': 'opened'}, opened
            viewer, = [w for w in webview.windows if w is not initial]
            # open_local_report waits for the production page to request rooms
            # after its navigation fence enabled the reader.
            print('Production offline viewer requested its initial room list', flush=True)
            facts = _exercise(viewer, fixture, artifacts)
            print('NATIVE_CONTEXT_REPORT ' + json.dumps({'renderer': renderer, **facts}), flush=True)
        except BaseException as exc:
            failed.append(str(exc))
            traceback.print_exc()
        finally:
            finished.set()
            if viewer is not None:
                viewer.destroy()
            initial.destroy()

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
    child = run_in_group([sys.executable, '-m', 'tests.test_context_report_native_window', str(root), str(artifacts)],
                         timeout=100, cwd=ROOT)
    print(child.stdout, end='')
    assert child.returncode == 0, (child.returncode, child.stderr)
    records = [json.loads(line.removeprefix('NATIVE_CONTEXT_REPORT ')) for line in child.stdout.splitlines()
               if line.startswith('NATIVE_CONTEXT_REPORT ')]
    observed, = records
    for name in ('a-compared.png', 'b-after-switch.png'):
        assert (artifacts / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), name
    record_property('native_context_report', json.dumps(observed, sort_keys=True))


if __name__ == '__main__':
    assert sys.platform == 'win32' and os.environ.get('OUROBOROS_WINDOWS_DESKTOP_QA') == '1'
    _open_native(Path(sys.argv[1]), Path(sys.argv[2]))
