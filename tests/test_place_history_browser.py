"""Real Skills -> Place -> conversation consumer. Synthetic producers and real GET APIs only.

Every browser request is intercepted; no live server, provider, or model is contacted.
Run under scripts/safe_test.py with OUROBOROS_RUN_UI_SMOKE=1, -m ui_browser.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests._skills_page_harness import URL, _document, fixture, open_skills, skills_browser as skills_browser
from tests.place_history_scenario import scenario

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
MATRIX = [(engine, width, theme) for engine in ('chromium', 'webkit') for width in (1280, 390) for theme in ('light', 'dark')]


@pytest.mark.parametrize('engine,width,theme', MATRIX)
def test_skills_place_reader_drafts_facts_and_teardown(skills_browser, tmp_path, monkeypatch, engine, width, theme):
    client, place, task, _ = scenario(tmp_path, monkeypatch)
    skill = fixture('extensions_fix_r1.json')['skills'][0]
    skill.update(name='bridge', description='Synthetic connector for captured conversations.', connects=['chat'], install=None)
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path / 'evidence')))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={'width': width, 'height': 844 if width == 390 else 900}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    skills = open_skills(page, extensions=[skill])
    page.route(URL, lambda route: route.fulfill(content_type='text/html; charset=utf-8',
        body=_document().replace('<head>', '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">')))
    def api(request):
        assert request.method == 'GET', request.url
        url = urlsplit(request.url)
        response = client.get(url.path + ('?' + url.query if url.query else ''))
        return response.status_code, response.json()
    skills.answers['/api/places'] = api
    rooms_path = f'/api/places/{place.place_id}/rooms'
    skills.answers[rooms_path] = api
    skills.answers['/api/chat/history'] = api
    prefix = f'{engine}-{width}-{theme}'
    observations = {'synthetic': True, 'engine': engine, 'version': browser.version, 'viewport': [width, 844 if width == 390 else 900], 'theme': theme}
    def capture(scene, subject):
        page.screenshot(path=str(evidence / f'{prefix}-{scene}.png'))
        # Optional read-only design measurement retained beside the QA directory.
        measure_path = evidence.parent / 'ui-measure.js'
        if measure_path.is_file():
            measure = measure_path.read_text()
            observations.setdefault('measurements', {})[scene] = page.evaluate(f'({measure})({json.dumps({"subject":subject})})')
    try:
        page.goto(URL, wait_until='networkidle')
        page.add_script_tag(url='/static/purify.min.js')
        page.add_script_tag(url='/static/marked.min.js')
        page.add_style_tag(url='/static/settings.css')
        page.add_style_tag(url='/static/consciousness.css')
        page.evaluate("({theme,width}) => { document.documentElement.classList.add('ouro-ui'); document.documentElement.dataset.theme=theme; document.documentElement.dataset.layout=width < 720 ? 'stack' : 'docked'; }", {'theme': theme, 'width': width})
        card = page.locator('.skills-card[data-skill="bridge"]')
        card.locator('.skills-card-name').click()
        details = page.locator('.skills-detail-sheet')
        details.locator('[data-place-open]').first.wait_for()
        assert details.locator('[data-place-open]').count() == 2
        assert details.locator('[data-sheet-action="open_place"]').count() == 0
        observations['connector_sections'] = details.locator('.skills-sheet-section').evaluate_all('''nodes => nodes.map(el => {
            const box=el.getBoundingClientRect(); return {heading:el.querySelector('h3')?.textContent,x:box.x,width:box.width};
        })''')
        sections = observations['connector_sections']
        assert all(abs(section['x'] - sections[0]['x']) <= 1 for section in sections), sections
        assert all(abs(section['width'] - sections[0]['width']) <= 1 for section in sections), sections
        capture('connector', '.skills-detail-sheet')
        details.locator(f'[data-place-open="{place.place_id}"]').click()
        sheet = page.locator('.place-sheet')
        room = sheet.locator('[data-place-room]').first
        room.wait_for()
        assert page.locator('dialog[open]').count() == 1
        # The readiness line (the connector's status) precedes the state when the connector reports one.
        assert sheet.locator('[data-place-section]').evaluate_all("nodes => nodes.map(n=>n.dataset.placeSection).slice(0,4)") == ['readiness', 'state', 'conversations', 'wakes']
        capture('settings', '.place-sheet')
        note = sheet.locator('[data-place-draft="note"]')
        sheet.locator('[data-place-act="add-note"]').click()  # local validation, no write
        note.fill('Unsaved private note')
        sheet.locator('[data-place-draft="folder"]').fill('/synthetic/draft-folder')
        sheet.locator('.place-more summary').click()
        sheet.locator('[data-place-draft="model"]').fill('draft/model')
        sheet.locator('[data-place-act="add-mine"]').click()
        sheet.locator('[data-place-draft="actorId"]').fill('draft-account')
        sheet.locator('[data-place-draft="actorLabel"]').fill('Draft Private Name')
        room.click()
        sheet.locator('.place-convo-row').first.wait_for()
        assert page.locator('dialog[open]').count() == 1
        assert sheet.locator('textarea:visible, input:visible').count() == 0
        text = sheet.locator('.place-convo').inner_text()
        for expected in ('Ouroboros', 'System', 'Maya Private', '(you)', 'Focus started', 'Heard while Paused', 'Delivery not confirmed', 'Accepted by service; delivery not confirmed', 'Delivered', 'Delivery failed', 'Delivery outcome unknown', 'Reply held:'):
            assert expected in text, (expected, text)
        assert 'PRIVATE THOUGHT' not in text and 'TECHNICAL EVIDENCE' not in text
        assert sheet.locator('.is-assistant .place-convo-who').all_text_contents() == ['Ouroboros'] * sheet.locator('.is-assistant').count()
        assert sheet.locator('.is-assistant strong').count() >= 2
        assert sheet.locator('.place-convo img').count() == 0
        capture('conversation', '.place-convo')
        observations['geometry'] = sheet.evaluate('''el => {
            const body=el.querySelector('[data-place-body]'); const box=el.getBoundingClientRect();
            return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,body:[body.clientWidth,body.scrollWidth],
                sheet:{x:box.x,y:box.y,right:box.right,bottom:box.bottom},
                controls:[...el.querySelectorAll('button')].filter(b=>b.getClientRects().length).map(b=>({text:b.textContent,height:b.getBoundingClientRect().height}))};
        }''')
        assert observations['geometry']['scrollWidth'] <= width
        assert observations['geometry']['body'][1] <= observations['geometry']['body'][0] + 1
        assert observations['geometry']['sheet']['x'] >= 0
        assert observations['geometry']['sheet']['right'] <= width + 1
        # A real external Place event repaints settings behind the reader; history focus stays put.
        page.evaluate("""async raw => {
            const {normalizePlace,PLACES_CHANGED_EVENT}=await import('/static/modules/consciousness_api.js');
            window.dispatchEvent(new CustomEvent(PLACES_CHANGED_EVENT,{detail:{place:normalizePlace({...raw,state:'paused'})}}));
        }""", next(row for row in client.get('/api/places').json()['places'] if row['place_id'] == place.place_id))
        assert 'Paused now' in sheet.locator('[data-history-state]').inner_text()
        sheet.locator('[data-history-back]').click()
        assert note.input_value() == 'Unsaved private note'
        assert sheet.locator('[data-place-section="notes"] [role="status"]').inner_text() == 'Write the note first.'
        assert sheet.locator('.place-more').evaluate('el => el.open')
        for field, expected in [('folder', '/synthetic/draft-folder'), ('model', 'draft/model'), ('actorId', 'draft-account'), ('actorLabel', 'Draft Private Name')]:
            assert sheet.locator(f'[data-place-draft="{field}"]').input_value() == expected
        assert room.evaluate('el => document.activeElement === el')
        note.focus()
        note.evaluate('el => el.setSelectionRange(3,9)')
        page.evaluate("""async raw => {
            const {normalizePlace,PLACES_CHANGED_EVENT}=await import('/static/modules/consciousness_api.js');
            window.dispatchEvent(new CustomEvent(PLACES_CHANGED_EVENT,{detail:{place:normalizePlace(raw)}}));
        }""", next(row for row in client.get('/api/places').json()['places'] if row['place_id'] == place.place_id))
        assert note.evaluate('el => [document.activeElement === el,el.selectionStart,el.selectionEnd]') == [True, 3, 9]
        # Translation runs through production selectors. Authored names/text never become misses.
        room.click()
        sheet.locator('.place-convo-row').first.wait_for()
        translations = page.evaluate("""async () => {
            const i=await import('/static/modules/i18n.js'); window.misses=[];
            i.setMissTransport(payload=>{window.misses.push(payload); return Promise.resolve({ok:true})});
            i.applyPayload({language:'fr',english:false,revision:1,entries:{'Back to Place':{text:'Retour au lieu'}}});
            const translator=i.createTranslator(); translator.applyTo(document.body);
            await new Promise(r=>setTimeout(r,900)); return {misses:window.misses,text:document.querySelector('[data-history-back]').textContent};
        }""")
        assert translations['text'] == 'Retour au lieu'
        encoded = json.dumps(translations['misses'])
        assert 'Private' not in encoded and 'private' not in encoded and 'draft-folder' not in encoded, encoded
        # Shared app Back dispatches Escape; first Back returns to settings, second closes.
        page.evaluate("""async () => {const {createLayerStack}=await import('/static/modules/layers.js'); window.layers=createLayerStack(); window.layers.back();}""")
        assert sheet.locator('.place-convo').count() == 0 and note.input_value() == 'Unsaved private note'
        page.evaluate('window.layers.back(); window.layers.destroy()')
        assert page.locator('dialog.document-reader').count() == 0, page.evaluate("({dialogs:[...document.querySelectorAll('dialog')].map(el=>({open:el.open,html:el.outerHTML.slice(0,400)})),active:document.activeElement.outerHTML.slice(0,500)})")
        # Native close and Escape both dispose pending reads; late answers cannot remount content.
        for how in ('close', 'escape', 'button'):
            card.locator('.skills-card-name').click()
            page.locator(f'[data-place-open="{place.place_id}"]').click()
            page.locator('[data-place-room]').first.wait_for()
            skills.hold.add('/api/chat/history')
            page.locator('[data-place-room]').first.click()
            skills.wait_for_request('/api/chat/history')
            if how == 'close':
                page.locator('.place-sheet').evaluate('el => el.close()')
            elif how == 'escape':
                page.keyboard.press('Escape')
                assert page.locator('.place-convo').count() == 0
                page.keyboard.press('Escape')
            else:
                page.locator('[data-place-close]').click()
            page.wait_for_function("document.querySelectorAll('dialog.document-reader').length === 0")
            skills.release('/api/chat/history', body={'messages': [], 'has_more': False, 'next_cursor': None, 'window': {'complete': True, 'truncated_by': []}})
            skills.hold.remove('/api/chat/history')
            assert page.locator('.place-convo').count() == 0
        assert skills.posts == [], skills.posts
        assert errors == [], errors
        observations['requests'] = skills.requests
        observations['pageerrors'] = errors
        (evidence / f'{prefix}.json').write_text(json.dumps(observations, indent=2))
    finally:
        observations['pageerrors'] = errors
        observations['requests'] = skills.requests
        observations['dialogs_at_end'] = page.locator('dialog').evaluate_all('nodes=>nodes.map(el=>({open:el.open,html:el.outerHTML.slice(0,800)}))')
        (evidence / f'{prefix}.json').write_text(json.dumps(observations, indent=2))
        skills.close(browser)
        client.close()


def _import_earlier(root, numbers, days_ago, import_id):
    """Earlier messages of the scenario's room through the real Host writer (``presence_history``)."""
    from datetime import datetime, timedelta, timezone

    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_history import parse_history_request, record_history
    now, key = datetime.now(timezone.utc).replace(microsecond=0), conversation_key('chat', 'qa', 'design', '')
    records = []
    for n in numbers:
        sent = (now - timedelta(days=days_ago - n)).isoformat()
        records.append({'event': {'source_event_id': f'qa-earlier-{n}', 'provider': 'chat', 'account_id': 'qa',
                                  'conversation_id': 'design', 'thread_id': '', 'conversation_key': key,
                                  'actor': {'platform_actor_id': 'maya', 'display_name': 'Maya Private'},
                                  'conversation': {'title': '#design-review private'},
                                  'message': {'message_id': f'e{n}', 'sent_at': sent}, 'text': f'Earlier message {n:03d}'},
                        'authorship': 'participant', 'source_timestamp': sent})
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill='bridge', provider='chat',
                                    account_id='qa', conversation_id='design')
    status, body = record_history(root, admission, parse_history_request({'import_id': import_id, 'records': records}))
    assert status == 200, body
    return {f'Earlier message {n:03d}': (now - timedelta(days=days_ago - n)).year for n in numbers}


@pytest.mark.parametrize('engine,width,theme', [('chromium', 1280, 'light'), ('webkit', 390, 'dark')])
def test_place_reader_reads_again_when_an_import_changes_its_room(skills_browser, tmp_path, monkeypatch, engine, width, theme):
    """The owner reads a room's live conversation while its first import lands. The next Load older is refused
    (409 history_view_changed: the room now reads by time); the reader reads the newest page afresh, keeps the
    shown rows until it arrives and the keyboard focus on Load older, says why, and pages on to the imported
    rows, each time from another year naming its year. An import landing after the reader reached its start
    stays out of that start, which says it is the record as read; Refresh history reads it in. Drafts survive,
    and nothing is written from the browser."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import dialogue_store as ds
    client, place, task, _ = scenario(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    for n in range(160):  # a long live conversation, read in stored order until the first import
        ds.append(tmp_path, 'chat', {'ts': (now - timedelta(minutes=300 - n)).isoformat(), 'direction': 'in',
                                     'chat_id': task['chat_id'], 'type': 'place_arrival', 'text': f'Live message {n:03d}',
                                     'source': 'presence:chat', 'sender_label': 'Maya Private', 'sender_session_id': 'maya',
                                     'client_message_id': f'qa-live-{n}', 'task_id': f'qa-live-turn-{n}', 'place_id': place.place_id})
    skill = fixture('extensions_fix_r1.json')['skills'][0]
    skill.update(name='bridge', description='Synthetic connector for captured conversations.', connects=['chat'], install=None)
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path / 'evidence'))) / 'place-reader-import'
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f'{engine}-{width}-{theme}'
    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={'width': width, 'height': 844 if width == 390 else 900}, device_scale_factor=1,
                            locale='en-US', timezone_id='UTC')
    errors, reads = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    skills = open_skills(page, extensions=[skill])
    page.route(URL, lambda route: route.fulfill(content_type='text/html; charset=utf-8',
        body=_document().replace('<head>', '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">')))

    def api(request):
        assert request.method == 'GET', request.url
        url = urlsplit(request.url)
        response = client.get(url.path + ('?' + url.query if url.query else ''))
        if url.path == '/api/chat/history':
            reads.append([response.status_code, response.json().get('reason_code'), 'cursor=' in url.query])
        return response.status_code, response.json()
    for path in ('/api/places', f'/api/places/{place.place_id}/rooms', '/api/chat/history'):
        skills.answers[path] = api
    sheet = page.locator('.place-sheet')
    reader = sheet.locator('.place-convo')
    status = reader.locator('[data-history-status]')

    def settled():
        page.wait_for_function("() => { const s = document.querySelector('.place-convo [data-history-status]');"
                               " return s && !/Reading captured history/.test(s.textContent); }")

    def older_to_start():
        while reader.locator('[data-history-older]').is_visible():
            reader.locator('[data-history-older]').click()
            settled()

    def shown():
        return reader.locator('.place-convo-row').evaluate_all('''rows => rows.map(row => ({
            text: (row.querySelector('.place-convo-text')?.textContent || '').trim(),
            time: row.querySelector('time')?.textContent || '', title: row.querySelector('time')?.title || ''}))''')

    def capture(scene):
        page.screenshot(path=str(evidence / f'{prefix}-{scene}.png'))
    observations = {'synthetic': True, 'engine': engine, 'viewport': [width, 844 if width == 390 else 900], 'theme': theme}
    try:
        page.goto(URL, wait_until='networkidle')
        page.add_script_tag(url='/static/purify.min.js')
        page.add_script_tag(url='/static/marked.min.js')
        page.add_style_tag(url='/static/settings.css')
        page.add_style_tag(url='/static/consciousness.css')
        page.evaluate("({theme,width}) => { document.documentElement.classList.add('ouro-ui'); document.documentElement.dataset.theme=theme; document.documentElement.dataset.layout=width < 720 ? 'stack' : 'docked'; }", {'theme': theme, 'width': width})
        page.locator('.skills-card[data-skill="bridge"] .skills-card-name').click()
        page.locator(f'.skills-detail-sheet [data-place-open="{place.place_id}"]').click()
        room = sheet.locator('[data-place-room]').first
        room.wait_for()
        sheet.locator('[data-place-act="add-note"]').click()
        sheet.locator('[data-place-draft="note"]').fill('Unsaved private note')
        room.click()
        reader.locator('.place-convo-row').first.wait_for()
        settled()
        before = shown()
        assert reader.locator('[data-history-older]').is_visible() and not any(row['text'].startswith('Earlier') for row in before)
        years = _import_earlier(tmp_path, range(30), 800, 'imp-first')  # the room's first import, while it is open
        reader.locator('[data-history-older]').focus()
        page.keyboard.press('Enter')
        settled()
        assert status.inner_text() == 'Saved history changed, so it was read again from the newest messages.'
        assert reads[-2][:2] == [409, 'history_view_changed'] and reads[-1][0] == 200 and not reads[-1][2], reads
        assert reader.locator('[data-history-older]').evaluate('el => document.activeElement === el'), 'focus stays on Load older'
        assert reader.locator('[data-history-retry]').is_hidden()
        capture('first-import-recovered')
        older_to_start()
        rows = shown()
        texts = [row['text'] for row in rows]
        earlier = [row for row in rows if row['text'].startswith('Earlier message')]
        assert [row['text'] for row in earlier] == [f'Earlier message {n:03d}' for n in range(30)]
        assert texts.index('Earlier message 029') < texts.index('Live message 000') < texts.index('Live message 159')
        assert all(str(years[row['text']]) in row['time'] for row in earlier), earlier[:2]
        # An import shows its send time; its title says when Ouroboros actually saved it (the import's receipt).
        assert all(row['title'].startswith('Sent ') and ' · saved ' in row['title'] for row in earlier), earlier[:2]
        assert str(now.year) not in next(row['time'] for row in rows if row['text'] == 'Live message 159')
        coverage = reader.locator('[data-history-coverage]')
        assert coverage.inner_text().startswith('Start of the record saved when this was read. Refresh history to include anything saved since.')
        reader.locator('.place-convo-row').first.scroll_into_view_if_needed()
        capture('start-with-years')
        _import_earlier(tmp_path, range(100, 105), 1200, 'imp-older')  # lands after the reader reached its start
        assert shown() == rows and 'Earlier message 100' not in reader.inner_text()
        reader.locator('[data-history-refresh]').click()
        settled()
        older_to_start()
        refreshed = [row['text'] for row in shown()]
        assert refreshed[:5] == [f'Earlier message {n:03d}' for n in range(100, 105)] and refreshed[5:] == texts
        reader.locator('.place-convo-row').first.scroll_into_view_if_needed()
        capture('refreshed-start')
        observations['geometry'] = page.evaluate("() => ({width: innerWidth, scrollWidth: document.documentElement.scrollWidth})")
        assert observations['geometry']['scrollWidth'] <= width
        reader.locator('[data-history-back]').click()
        assert sheet.locator('[data-place-draft="note"]').input_value() == 'Unsaved private note'
        assert skills.posts == [], skills.posts
        assert errors == [], errors
    finally:
        observations.update(pageerrors=errors, reads=reads, browser=browser.version)
        (evidence / f'{prefix}.json').write_text(json.dumps(observations, indent=2))
        skills.close(browser)
        client.close()
