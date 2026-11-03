"""Main question materials and Project result navigation through the real SPA.

Readers, explicit downloads, keyboard navigation and reloads use the production
server. Evidence goes to the shared diagnostic channel, or an explicit
OUROBOROS_BROWSER_EVIDENCE_OUT, including RED.
"""

import hashlib
import struct
import zlib
import json
import re

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests._shared import producer_evidence_dir
from tests.system_e2e.harness import KeylessIsolatedServer, ScriptedStubModel, keyless_settings, write_settings_file

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]



def _m1660_png():
    """A visible PNG fixture, made without an optional imaging dependency."""
    width, height = 240, 120
    def chunk(kind, value):
        return struct.pack('!I', len(value)) + kind + value + struct.pack('!I', zlib.crc32(kind + value) & 0xffffffff)
    rows = b''.join(b'\x00' + (b'\x17\x39\x66' if y < 60 else b'\xf4\xa2\x43') * width for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def _m1660_material_question(root):
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.owner_quiz import mark_published, quiz_states, reconcile_terminal, record_publication
    from ouroboros.project_dialogue import project_question_pointer
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.core_artifacts import _escalate
    from supervisor.message_bus import log_chat
    from tests.test_quiz_answer import _tool_ctx

    project = create_project(root, 'material-review', name='Material review')
    task_id = 'material-review-task'
    bind_task_to_project(root, task_id, project['id'], project['chat_id'], origin={'absent': 'system'})
    payloads = {'comparison.png': _m1660_png(),
                'decision-brief.md': b'# Decision brief\n\nChoose the amber panel for the printed edition.\n',
                'supporting-notes.txt': b'The measurements were captured before the decision.\n'}
    for name, payload in payloads.items():
        store_task_artifact_bytes(root, task_id, name=name, data=payload)
    ctx = _tool_ctx(root, task_id=task_id, chat_id=project['chat_id'])
    ctx.task_metadata['root_task_id'] = task_id
    receipt = _escalate(ctx, question='Which panel should the next edition use?',
                       options=[{'label': 'Amber panel', 'detail': 'Use the printed comparison.'},
                                {'label': 'Blue panel', 'detail': 'Keep the existing edition.'}],
                       assumption='Keep the current edition until you choose.',
                       materials=[{'name': name} for name in payloads])
    assert receipt.startswith('OK:'), receipt
    [quiz] = quiz_states(root, task_id).values()
    quiz_id = quiz['quiz_id']
    record_publication(root, task_id, quiz_id, project['chat_id'])
    write_task_result(root, task_id, 'completed', project_id=project['id'], chat_id=project['chat_id'],
                      result='The evidence is ready for your choice.')
    reconcile_terminal(root, task_id)
    block = quiz_states(root, task_id)[quiz_id]
    log_chat('out', project['chat_id'], 1, block['question'], task_id=task_id, record_type='quiz',
             quiz=block, drive_root=root, require_write=True)
    mark_published(root, task_id, quiz_id)
    block = quiz_states(root, task_id)[quiz_id]
    pointer = project_question_pointer({'task_id': task_id, 'quiz_id': quiz_id}, block, project)
    log_chat('out', 1, 1, pointer['text'], task_id=task_id, message_meta=pointer, drive_root=root, require_write=True)
    return {'project': project, 'task_id': task_id, 'quiz_id': quiz_id, 'materials': block['materials'],
            'escalate_receipt': receipt}, payloads


def _m1660_material_evidence(tmp_path, stem):
    evidence = producer_evidence_dir(tmp_path / 'screenshots', "main-materials",
                                     env='OUROBOROS_BROWSER_EVIDENCE_OUT') / stem
    evidence.mkdir(parents=True, exist_ok=True)
    return evidence


def _m1660_material_finish(page, server, evidence, receipt):
    """Retain the actual failing screen and authoritative state before shutdown."""
    try:
        receipt['state_after'] = _api(server.base_url, 'GET', '/api/state')
        receipt['task_after'] = _api(server.base_url, 'GET', f"/api/tasks/{receipt['task_id']}")
    except Exception as exc:
        receipt['evidence_api_error'] = str(exc)
    if page is not None:
        try:
            page.screenshot(path=str(evidence / 'end.png'), full_page=True, animations='disabled')
            receipt['url_after'] = page.url
            receipt['body_after'] = page.locator('body').inner_text()
        except Exception as exc:
            receipt['evidence_browser_error'] = str(exc)
    (evidence / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')


def _m1660_tab_to(page, target, limit=80):
    """Real keyboard traversal; evaluating activeElement only observes focus."""
    for _ in range(limit):
        page.keyboard.press('Tab')
        if target.evaluate('(node) => node === document.activeElement'):
            return
    pytest.fail(f'Keyboard traversal did not reach {target}')


@pytest.mark.parametrize('width', [1100, 390], ids=['desktop', 'narrow'])
def test_main_question_materials_reader_and_keyboard(wait_clone, tmp_path, width):
    from playwright.sync_api import expect, sync_playwright

    root = tmp_path / 'instance' / 'data'
    root.mkdir(parents=True)
    receipt, payloads = _m1660_material_question(root)
    evidence = _m1660_material_evidence(tmp_path, f'materials-{width}')
    receipt.update(viewport={'width': width, 'height': 900}, stages=[], responses=[])
    with ScriptedStubModel([{'final': 'Unused model fixture.'}]) as stub:
        settings_path = root / 'settings.json'
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport=receipt['viewport'], reduced_motion='reduce')
                try:
                    page.goto(server.base_url, wait_until='domcontentloaded')
                    # The Project question's Main copy stands in the feed at its ask.
                    card = page.locator(f'#chat-messages .project-question-card[data-quiz-id="{receipt["quiz_id"]}"]')
                    card.wait_for(timeout=30000)
                    materials = card.locator('.chat-quiz-materials')
                    expect(materials.locator('.chat-file-card')).to_have_count(3)
                    image = materials.locator('img.chat-file-thumb')
                    expect(image).to_be_visible()
                    page.wait_for_function('(img) => img.complete && img.naturalWidth === 240 && img.naturalHeight === 120', arg=image.element_handle())
                    for material in receipt['materials']:
                        response = page.request.get(server.base_url + material['download_url'])
                        actual = response.body()
                        receipt['responses'].append({'name': material['label'], 'status': response.status,
                            'sha256': hashlib.sha256(actual).hexdigest(), 'content_type': response.headers.get('content-type')})
                        assert response.status == 200
                        assert actual == payloads[material['label']]
                        assert hashlib.sha256(actual).hexdigest() == material['sha256']
                    receipt['stages'].append('PNG decoded at 240x120; all material bytes match registered custody')
                    picture = materials.get_by_role('button', name='comparison.png', exact=False)
                    expect(picture.locator('.chat-file-more')).to_have_text('View')
                    card.locator('.chat-quiz-question').focus()
                    _m1660_tab_to(page, picture)
                    page.keyboard.press('Enter')
                    dialog = page.locator('dialog.document-reader')
                    expect(dialog).to_be_visible()
                    page.wait_for_function("() => document.querySelector('dialog.document-reader .document-reader-image')?.naturalWidth === 240")
                    expect(dialog.get_by_role('region', name='Image')).to_be_focused()
                    shown = dialog.locator('.document-reader-image').evaluate('(img) => [img.width, img.height]')
                    assert shown == [240, 120], f'Fit shows the material whole and never enlarges it: {shown}'
                    page.screenshot(path=str(evidence / 'image-preview.png'), animations='disabled')
                    page.keyboard.press('Escape')
                    expect(dialog).to_have_count(0)
                    expect(picture).to_be_focused()
                    receipt['stages'].append('Keyboard opened the image material in the in-chat preview at its own size; Escape restored focus')
                    document = materials.get_by_role('button', name='decision-brief.md', exact=False)
                    # The keyboard starts at the question text, where Main's waiting pill lands a question.
                    card.locator('.chat-quiz-question').focus()
                    _m1660_tab_to(page, document)
                    expect(document).to_be_focused()
                    page.keyboard.press('Enter')
                    dialog = page.locator('dialog.document-reader')
                    expect(dialog).to_be_visible()
                    dialog.get_by_role('heading', name='Decision brief', exact=True).wait_for()
                    dialog.get_by_text('Choose the amber panel for the printed edition.', exact=True).wait_for()
                    expect(dialog.get_by_role('region', name='Document text')).to_be_focused()
                    page.screenshot(path=str(evidence / 'document-reader.png'), animations='disabled')
                    dialog.get_by_role('button', name='Source', exact=True).click()
                    expect(dialog.locator('.document-reader-body')).to_contain_text('# Decision brief')
                    page.keyboard.press('Escape')
                    expect(dialog).to_have_count(0)
                    expect(document).to_be_focused()
                    receipt['stages'].append('Keyboard opened existing reader; Markdown and source readable; Escape restored focus')
                    option = card.get_by_role('button', name='Amber panel Use the printed comparison.', exact=True)
                    _m1660_tab_to(page, option)
                    expect(option).to_be_focused()
                    overflow = page.evaluate('({width: innerWidth, scroll: document.documentElement.scrollWidth})')
                    receipt['layout'] = overflow
                    assert overflow['scroll'] <= overflow['width'] + 1, overflow
                    assert '#project/' not in page.url, 'Materials must be usable with the Project closed'
                    page.screenshot(path=str(evidence / 'question-focused.png'), animations='disabled')
                    receipt['stages'].append('Keyboard reached answer control; Main remains selected; no page horizontal overflow')
                finally:
                    _m1660_material_finish(page, server, evidence, receipt)
                    browser.close()
        finally:
            server.stop()


def test_main_question_unavailable_material_is_explicit(wait_clone, tmp_path):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.artifacts import task_artifact_dir_path

    root = tmp_path / 'instance' / 'data'
    root.mkdir(parents=True)
    receipt, _payloads = _m1660_material_question(root)
    evidence = _m1660_material_evidence(tmp_path, 'material-unavailable')
    material = next(row for row in receipt['materials'] if row['label'] == 'supporting-notes.txt')
    # Real loss of a retained fixture file; no intercepted network or DOM state.
    missing = task_artifact_dir_path(root, receipt['task_id']) / material['name']
    missing.unlink()
    receipt.update(removed_fixture=str(missing), stages=[])
    with ScriptedStubModel([{'final': 'Unused model fixture.'}]) as stub:
        settings_path = root / 'settings.json'
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={'width': 1100, 'height': 900}, reduced_motion='reduce')
                try:
                    page.goto(server.base_url, wait_until='domcontentloaded')
                    card = page.locator(f'#chat-messages .project-question-card[data-quiz-id="{receipt["quiz_id"]}"]')
                    card.wait_for(timeout=30000)
                    with page.expect_response(lambda response: response.url.endswith(material['download_url'])) as read:
                        card.get_by_role('button', name='supporting-notes.txt', exact=False).click()
                    receipt['response_status'] = read.value.status
                    assert read.value.status == 404
                    dialog = page.locator('dialog.document-reader')
                    expect(dialog).to_be_visible()
                    dialog.get_by_text('This delivered copy is missing or did not pass its integrity check, so it is not shown. Ask Ouroboros to send the file again.', exact=True).wait_for()
                    expect(dialog).not_to_contain_text('The measurements were captured before the decision.')
                    assert not dialog.get_by_role('button', name='Retry', exact=True).count()
                    receipt['stages'].append('Real missing artifact returned 404 and reader names unavailable copy')
                    page.screenshot(path=str(evidence / 'unavailable-reader.png'), animations='disabled')
                    page.keyboard.press('Escape')
                    expect(card.get_by_role('button', name='Amber panel Use the printed comparison.', exact=True)).to_be_enabled()
                    assert '#project/' not in page.url
                finally:
                    _m1660_material_finish(page, server, evidence, receipt)
                    browser.close()
        finally:
            server.stop()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_main_live_completion_opens_project_answer_documents_and_task_files(wait_clone, tmp_path, engine):
    """A real queued Project task sends documents and finishes while Main is open.
    Only model judgment is scripted; delivery, task Files, history and reads are real."""
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project
    from tests.test_ui_smoke_project_completion import _LONG_ANSWER

    root = tmp_path / 'instance' / 'data'
    root.mkdir(parents=True)
    project = create_project(root, 'project-door', name='Long report')
    source = root / 'source'
    source.mkdir()
    brief = source / 'brief.md'
    brief.write_text('# Project brief\n\nThe report remains readable in its Project.\n', encoding='utf-8')
    note = source / 'notes.txt'
    note.write_text('The notes are a separate delivered document.\n', encoding='utf-8')
    answer = _LONG_ANSWER + '\n\nPROJECT_ANSWER_TAIL'
    evidence = _m1660_material_evidence(tmp_path, f'project-door-{engine}')
    receipt = {'engine': engine, 'project': project, 'stages': []}
    script = [{'tool': 'send_file', 'arguments': {'file_path': str(path)}} for path in (brief, note)]
    with ScriptedStubModel(script, final_answer=answer) as stub:
        settings_path = root / 'settings.json'
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        try:
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={'width': 1100, 'height': 900}, reduced_motion='reduce')
                frames = []
                page.on('websocket', lambda socket: socket.on('framereceived', lambda value: frames.append(json.loads(value))))
                try:
                    page.goto(server.base_url, wait_until='domcontentloaded')
                    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent.trim() === 'Online'")
                    created = _api(server.base_url, 'POST', '/api/tasks', {
                        'description': 'Send the two prepared reports and return the selected final answer.',
                        'title': 'Project completion door', 'project_id': project['id'], 'chat_id': project['chat_id'],
                        'source': 'web', 'memory_mode': 'forked', 'metadata': {'delegation_role': 'root'}})
                    receipt['task_id'] = task_id = created['task_id']
                    final = page.locator(f'#chat-messages .chat-bubble.project-answer[data-task-id="{task_id}"]')
                    final.wait_for(timeout=120000)
                    assert any(frame.get('system_type') == 'project_completion_summary' and frame.get('task_id') == task_id
                               for frame in frames), 'the mirror must first arrive through the real live transport'
                    for stage in ('live', 'reload'):
                        if stage == 'reload':
                            page.reload(wait_until='domcontentloaded')
                            final.wait_for(timeout=30000)
                        expect(final).to_have_class(re.compile(r'\bis-folded\b'))
                        assert final.locator('[data-answer-fold], .project-answer-files').count() == 0
                        assert final.get_by_role('button', name='Files', exact=True).count() == 0
                        assert final.locator('.system-message-actions button').count() == 1
                        final.screenshot(path=str(evidence / f'main-{stage}.png'))
                        final.locator('[data-intent="open-project"]').click()
                        panel = page.locator(f'#panel-pchat-{project["id"]}')
                        full = panel.locator('.chat-bubble.assistant .message').filter(has_text='PROJECT_ANSWER_TAIL')
                        expect(full).to_have_count(1)
                        assert full.evaluate('n => n.scrollHeight <= n.clientHeight + 1')
                        expect(full).to_contain_text('Paragraph 8')
                        expect(panel.locator('.chat-file-card')).to_have_count(2)
                        card = panel.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                        card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
                        files = card.locator('[data-result-files]')
                        detail = _api(server.base_url, 'GET', f'/api/tasks/{task_id}')
                        # Retention also publishes its own memory export; it is not a third sent document.
                        expect(files).to_contain_text(f'{len(detail["artifacts"])} files')
                        expect(files.locator('a[download]')).to_have_count(len(detail['artifacts']))
                        with page.expect_download() as task_download:
                            files.locator('a[download]').filter(has_text=re.compile(r'^brief-')).click()
                        saved = evidence / f'task-file-{stage}.md'
                        task_download.value.save_as(str(saved))
                        assert saved.read_bytes() == brief.read_bytes()
                        document = panel.get_by_role('button', name='brief.md', exact=False)
                        document.click()
                        dialog = page.locator('dialog.document-reader')
                        expect(dialog.locator('.document-reader-body')).to_contain_text('The report remains readable in its Project.')
                        page.screenshot(path=str(evidence / f'project-document-{stage}.png'), animations='disabled')
                        with page.expect_download() as reader_download:
                            dialog.get_by_role('button', name='Download', exact=True).click()
                        saved = evidence / f'reader-{stage}.md'
                        reader_download.value.save_as(str(saved))
                        assert saved.read_bytes() == brief.read_bytes()
                        page.keyboard.press('Escape')
                        expect(document).to_be_focused()
                        page.locator('#project-panel-close').click()
                        receipt['stages'].append(f'{stage}: Main folded; Project full answer, delivered reader and task Files downloads match')
                finally:
                    receipt['live_completion_frames'] = [frame for frame in frames if frame.get('system_type') == 'project_completion_summary']
                    _m1660_material_finish(page, server, evidence, receipt)
                    browser.close()
        finally:
            server.stop()


def _m1660_seed_readable_inventory(root, repo, answer='Read the retained files from this completed result.'):
    from ouroboros.artifacts import collect_task_artifact_records, store_task_artifact_bytes
    from ouroboros.project_dialogue import project_completion_delivery_outcome
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.core import _write_file
    from ouroboros.tools.registry import ToolContext
    from supervisor.message_bus import log_chat

    task_id = 'readable-inventory-task'
    project = create_project(root, 'readable-inventory', name='Readable inventory')
    bind_task_to_project(root, task_id, project['id'], project['chat_id'], origin={'absent': 'system'})
    payloads = {f'part-{index:02d}.txt': f'Retained result number {index}.\n'.encode() for index in range(1, 12)}
    for name, payload in payloads.items():
        store_task_artifact_bytes(root, task_id, name=name, data=payload)
    nested = {'research/brief.txt': 'The nested brief is individually readable.\n',
              'research/appendix.txt': 'The separate appendix has different bytes.\n'}
    ctx = ToolContext(repo_dir=repo, drive_root=root, task_id=task_id)
    producer_receipts = []
    for path, body in nested.items():
        produced = _write_file(ctx, root='artifact_store', path=path, content=body)
        producer_receipts.append(produced)
        assert produced.startswith('OK: wrote artifact_store:'), produced
        payloads[path] = body.encode()
    records = collect_task_artifact_records(root, task_id, strict=True)
    assert {'research/brief.txt', 'research/appendix.txt'} <= {row.get('relpath') for row in records}
    result = write_task_result(root, task_id, 'completed', project_id=project['id'], chat_id=project['chat_id'],
        terminal_origin='model_final', result=answer,
        artifacts=records, outcome_axes={'execution': {'status': 'ok'}})
    from urllib.parse import quote
    # Ordinary delivered-document rows, not a replacement Main inventory.
    for name, payload in payloads.items():
        url = f'/api/tasks/{task_id}/artifacts/{quote(name.rsplit("/", 1)[-1])}'
        if '/' in name:
            url += '?relpath=' + quote(name, safe='')
        log_chat('out', project['chat_id'], 1, f'Delivered {name}', task_id=task_id,
                 record_type='document', filename=name, mime='text/plain', download_url=url,
                 size_bytes=len(payload), drive_root=root, require_write=True)
    log_chat('out', project['chat_id'], 1, result['result'], task_id=task_id, drive_root=root, require_write=True)
    task = {'id': task_id, 'project_id': project['id'], 'chat_id': project['chat_id'], 'root_task_id': task_id}
    disposition, _queued = project_completion_delivery_outcome(root, {}, task_id, task, result, {'status': 'completed'})
    assert disposition == 'owed'
    return {'task_id': task_id, 'project': project, 'registered_artifacts': records,
            'nested_producer_receipts': producer_receipts, 'producer_disposition': disposition,
            'completion_answer': result['result'], 'stages': [], 'downloads': []}, payloads


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_main_final_released_with_its_history_page_mounts_again_with_project_door(wait_clone, tmp_path, engine):
    """Four older pages release the page holding Main's final; reading it again mounts the
    folded preview and Project door once more, and a transport retry stays one outcome."""
    import re
    import time
    from datetime import datetime, timedelta, timezone
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store
    from tests import _dialogue as dialogue_test
    from tests.test_chat_history_paging_browser import MAIN, _idle, _open, _step
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = tmp_path / 'instance' / 'data'
    root.mkdir(parents=True)
    repo = tmp_path / 'artifact-producer-repo'
    repo.mkdir()
    # Four Main pages older than the final: one history read holds 150 conversation rows, an older
    # install's archive that the upgrade imports before the final is written.
    dialogue_test.seed(root, archives={'chat_20260901T000000.jsonl': [
        {'ts': '2026-09-01T10:00:00Z', 'chat_id': 1, 'direction': 'in', 'text': f'Older Main message {index:04d}'}
        for index in range(620)]})
    dialogue_test.import_all(root)
    answer = '\n\n'.join(f'Paragraph {index}: the retained reports are ready to read.' for index in range(1, 31))
    receipt, payloads = _m1660_seed_readable_inventory(root, repo, answer=answer)
    receipt.update(engine=engine)
    evidence = _m1660_material_evidence(tmp_path, f'final-page-release-{engine}')
    with ScriptedStubModel([{'final': 'Unused model fixture.'}]) as stub:
        settings_path = root / 'settings.json'
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        try:
            # The server publishes the owed final itself; its row carries the delivery identity.
            deadline, final_row = time.monotonic() + 90, None
            while final_row is None and time.monotonic() < deadline:
                rows = _api(server.base_url, 'GET', '/api/chat/history?chat_id=1').get('messages') or []
                final_row = next((row for row in rows if row.get('system_type') == 'project_completion_summary'
                                  and row.get('task_id') == receipt['task_id']), None)
                time.sleep(0 if final_row else 0.5)
            assert final_row and final_row.get('delivery_id'), 'the owed final was not published'
            receipt['final_row'] = final_row
            server.stop()
            # Newer Main conversation puts the final one page back from the newest read.
            start = datetime.now(timezone.utc)
            for index in range(160):
                dialogue_store.append(root, 'chat', {'ts': (start + timedelta(seconds=index)).isoformat(), 'chat_id': 1,
                                                     'direction': 'in', 'text': f'Newer Main message {index:04d}'})
            server.start(ready_timeout=120)
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={'width': 1100, 'height': 900}, reduced_motion='reduce')
                try:
                    _open(page, server.base_url)
                    final = page.locator(f'{MAIN} .chat-bubble.project-answer[data-delivery-id="{final_row["delivery_id"]}"]')

                    def readable(stage):
                        expect(final).to_have_count(1)
                        expect(final.locator('.message')).to_contain_text('Paragraph 1: the retained reports')
                        expect(final).to_have_class(re.compile(r'\bis-folded\b'))
                        assert final.locator('[data-answer-fold], .project-answer-files').count() == 0
                        assert final.get_by_role('button', name='Files', exact=True).count() == 0
                        final.locator('[data-intent="open-project"]').click()
                        panel = page.locator(f'#panel-pchat-{receipt["project"]["id"]}')
                        full = panel.locator('.chat-bubble.assistant .message').filter(has_text='Paragraph 30:')
                        expect(full).to_have_count(1)
                        assert full.evaluate('node => node.scrollHeight <= node.clientHeight + 1')
                        expect(panel.locator('.chat-file-card')).to_have_count(13)
                        document = panel.get_by_role('button', name='research/brief.txt', exact=False)
                        document.click()
                        dialog = page.locator('dialog.document-reader')
                        expect(dialog.locator('.document-reader-body')).to_contain_text(payloads['research/brief.txt'].decode().strip())
                        page.screenshot(path=str(evidence / f'{stage}.png'), animations='disabled')
                        page.keyboard.press('Escape')
                        page.locator('#project-panel-close').click()
                        page.evaluate('() => document.activeElement?.blur()')
                        receipt['stages'].append(f'{stage}: Main fold and Project full answer/nested delivered document readable')

                    expect(final).to_have_count(0)
                    _step(page, MAIN, automatic=True)
                    readable('first-read')
                    for _ in range(3):
                        _step(page, MAIN, automatic=True)
                    expect(final).to_have_count(0)
                    receipt['stages'].append('four older pages released the page holding the final')
                    page.locator('#chat-scroll-bottom').click()
                    _idle(page, MAIN)
                    _step(page, MAIN, automatic=True)
                    readable('read-again')
                    # A transport retry of the mounted delivery is the same outcome at its original time.
                    stamp = final.get_attribute('data-ts')
                    _emit_ws_frame(page, {
                        'type': 'chat', 'chat_id': 1, 'role': 'system', 'content': final_row.get('text') or '',
                        'ts': datetime.now(timezone.utc).isoformat(), 'markdown': bool(final_row.get('markdown')),
                        **{key: final_row[key] for key in ('system_type', 'task_id', 'project_id', 'project_name',
                                                           'task_name', 'completion_answer', 'completion_reason',
                                                           'terminal_time', 'delivery_id') if key in final_row}})
                    expect(final).to_have_count(1)
                    assert final.get_attribute('data-ts') == stamp
                    receipt['stages'].append('a transport retry kept one outcome at its original time')
                    page.reload(wait_until='domcontentloaded')
                    _idle(page, MAIN)
                    _step(page, MAIN, automatic=True)
                    readable('after-reload')
                finally:
                    _m1660_material_finish(page, server, evidence, receipt)
                    browser.close()
        finally:
            server.stop()
