// The read-only archive reader (room_archive.js; server: ouroboros/room_archive.py): Main's refusal for the Project
// registry offers it beside Retry; it reads rooms one at a time, oldest first, continues honestly and acts on nothing.
import assert from 'node:assert/strict';
import test from 'node:test';

import { archiveDoor, archiveOffered, openRoomArchive } from '../modules/room_archive.js';
import { createHistoryControls } from '../modules/chat_render_batch.js';
import { applyPayload, currentPayload, CODE_PREFIX } from '../modules/i18n.js';
import { domDouble } from './dom_double.js';

const refusal = (fields = { project_registry: 'unavailable' }) => Object.assign(new Error('refused'), {
    status: 503, body: { messages: [], error: 'history_source_unavailable', reason_code: 'history_source_unavailable', ...fields },
});
const message = (id, role, text, extra = {}) => ({ id: `archive:chat:${id}`, role, text, ts: '2026-11-17T10:00:00Z', ...extra });
const ROOMS = [{ room: 'main', name: 'Main' }, { room: '1234567', chat_id: 1234567, project_id: 'alpha', name: 'Alpha room',
    evidence: ['binding'] }];
const settle = async () => { for (let n = 0; n < 6; n += 1) await new Promise(setImmediate); };

function reader(t, pages) {
    const { doc } = domDouble();
    const calls = [];
    const fetched = [];
    const priorFetch = globalThis.fetch;
    globalThis.fetch = async (url) => { fetched.push(String(url)); throw new Error('the reader reads only through read()'); };
    t.after(() => { globalThis.fetch = priorFetch; });
    let restored = 0;
    const read = async (request) => {
        calls.push({ room: request.room, before: request.before });
        const next = pages.shift();
        if (next instanceof Error) throw next;
        return typeof next === 'function' ? next(request) : next;
    };
    const handle = openRoomArchive({ doc, read, onDismiss: (readable) => { if (readable) restored += 1; } });
    t.after(() => handle.close({ restoreFocus: false }));
    const dialog = handle.dialog;
    const q = (selector) => dialog.querySelector(selector);
    return {
        doc, calls, fetched, handle, dialog, q,
        restored: () => restored,
        rows: () => dialog.querySelectorAll('.room-archive-row').map((row) => row.querySelector('.room-archive-text').textContent),
        speakers: () => dialog.querySelectorAll('.room-archive-speaker').map((node) => node.textContent.split(' · ')[0]),
        rooms: () => dialog.querySelectorAll('[data-archive-room]').map((node) => [node.textContent, node.getAttribute('aria-pressed')]),
    };
}

test('only the refusal for the Project registry offers the archive', () => {
    assert.equal(archiveOffered(refusal()), true);
    assert.equal(archiveOffered(refusal({})), false, 'another unavailable source keeps the plain failure');
    assert.equal(archiveOffered(null), false);
    assert.equal(archiveOffered({ body: { project_registry: 'restored' } }), false);
});

test('the history edge offers the archive for its effective typed refusal, not generic failure or partial history', (t) => {
    const { doc, el } = domDouble();
    const messages = el('div');
    doc.body.append(messages);
    const openers = [];
    const archive = archiveDoor({ open: () => { openers.push(doc.activeElement); } });
    const controls = createHistoryControls(messages, { archive });
    t.after(controls.destroy);
    controls.endRecent(refusal());
    let state = controls.render({ initialized: false, canOlder: false }, {});
    assert.equal(state.note, 'The project list can’t be read, so Main’s saved history isn’t shown.');
    assert.equal(messages.querySelector('.chat-load-older-note').textContent, state.note, 'a state of the place, at its edge');
    const edge = controls.olderButton;
    assert.equal(edge.textContent, 'Retry loading messages');
    const offered = edge.nextElementSibling;
    assert.ok(offered?.classes.has('chat-archive-door') && offered.hidden === false);
    assert.equal(offered.textContent, 'Read saved messages');
    offered.emit('click', { detail: 1 });
    assert.ok(openers.length === 1 && openers[0] === offered, 'the reader opens once with its trigger focused');

    controls.endRecent(refusal({}));
    state = controls.render({ initialized: false, canOlder: false }, {});
    assert.equal(state.note, 'Some saved history could not be loaded.');
    assert.equal(offered.hidden, true, 'another failure hides the door');
    controls.endRecent(new Error('partial'), true);
    controls.render({ initialized: true, canOlder: true, error: refusal() }, { gaps: true });
    assert.equal(offered.hidden, false, 'a partial newest page does not mask the older read’s typed refusal');
    controls.render({ initialized: true, canOlder: true }, { gaps: true });
    assert.equal(offered.hidden, true, 'a partial page alone is not a registry refusal');
    controls.endRecent(refusal(), true);
    state = controls.render({ initialized: true, canOlder: true }, { gaps: true });
    assert.equal(offered.hidden, true, 'only a refused ordinary read offers the archive, not a partial result');
    assert.doesNotMatch(state.note, /Main’s saved history isn’t shown/);
    controls.endRecent(null);
    controls.render({ initialized: true, canOlder: true });
    assert.equal(offered.hidden, true);
});

test('the archive translates its controls and templates but keeps saved words and names intact', async t => {
    const prior = currentPayload();
    t.after(() => applyPayload(prior || { language: 'en', english: true, entries: {} }));
    applyPayload({ language: 'fr', english: false, revision: 1, entries: {
        [CODE_PREFIX + 'history.read_saved']: { text: 'Lire les messages' },
        'Moved to {name}': { text: 'Vers {name}' }, 'Files: {names}': { text: 'Fichiers : {names}' },
        'Draft': { text: 'Brouillon' },
    } });
    const { doc, el } = domDouble();
    const edge = el('button');
    doc.body.append(el('div', {}, edge));
    const access = archiveDoor();
    t.after(access.dispose);
    access.offer(edge, refusal());
    assert.equal(edge.nextElementSibling.textContent, 'Lire les messages');
    const r = reader(t, [{ state: 'available', room: 'main', rooms: ROOMS, messages: [
        message(1, 'system', 'Draft', { kind: 'project_handoff' }),
        message(2, 'user', 'Draft', { attachments: ['Draft.txt'] }),
    ] }]);
    await settle();
    assert.deepEqual(r.rows(), ['Vers Draft', 'Draft']);
    assert.equal(r.q('.room-archive-files').textContent, 'Fichiers : Draft.txt');
    assert.ok(r.q('.room-archive-text').hasAttribute('data-i18n-authored'));
});

test('a door is never drawn for a room whose reads are not refused for the registry', () => {
    const { doc, el } = domDouble();
    const edge = el('button');
    doc.body.append(el('div', {}, edge));
    const door = archiveDoor();
    door.offer(edge, null);
    door.offer(edge, refusal({}));
    assert.equal(edge.nextElementSibling, null);
});

test('focus returns to the transcript when the archive offer disappears, including during a visit', () => {
    const { doc, el } = domDouble();
    const feed = el('div');
    const edge = el('button');
    doc.body.append(feed);
    feed.append(edge);
    const archive = archiveDoor({ returnFocus: feed,
        open: options => openRoomArchive({ ...options, read: async () => ({ state: 'not_needed' }) }) });
    archive.offer(edge, refusal());
    edge.nextElementSibling.focus();
    archive.offer(edge, null);
    assert.equal(feed.getAttribute('tabindex'), '-1');
    assert.equal(doc.activeElement, feed);
    archive.offer(edge, refusal());
    edge.nextElementSibling.emit('click', { detail: 0 });
    const body = doc.body.querySelector('.room-archive-body');
    assert.equal(body.classList.contains('is-quiet-focus'), false, 'keyboard activation shows focus on entry');
    archive.offer(edge, null);
    assert.equal(doc.activeElement, body, 'withdrawing the offer leaves the open reader focused');
    body.emit('keydown', { key: 'Escape' });
    assert.equal(doc.activeElement, feed);
    archive.dispose();
});

test('the reader shows a room oldest first as plain words, continues older and switches rooms', async (t) => {
    const r = reader(t, [
        { state: 'available', room: 'main', rooms: ROOMS, has_more: true, next_cursor: 'older-1', complete: false,
            limits: ['placement_unproven'], messages: [message(5, 'user', 'main owner'),
                message(6, 'system', 'Alpha room', { kind: 'project_handoff' }),
                message(7, 'system', 'Alpha room · Done', { kind: 'project_completion_summary' }),
                message(9, 'user', 'with a file', { attachments: ['plan.pdf'] })] },
        { state: 'available', room: 'main', rooms: ROOMS, has_more: false, next_cursor: null, complete: false,
            limits: ['placement_unproven'], messages: [message(2, 'assistant', 'a notice to Main')] },
        { state: 'available', room: '1234567', rooms: ROOMS, has_more: false, next_cursor: null, complete: false,
            limits: ['placement_unproven'], messages: [message(3, 'user', 'alpha owner')] },
    ]);
    await settle();
    assert.equal(r.doc.body.children.includes(r.dialog), true);
    assert.equal(r.dialog.getAttribute('aria-modal'), 'true');
    assert.equal(r.q('.document-reader-title').textContent, 'Saved messages');
    assert.equal(r.q('.document-reader-meta').textContent, 'Read-only · Main');
    assert.deepEqual(r.rows(), ['main owner', 'Moved to Alpha room', 'Alpha room · Done', 'with a file'],
        'a handoff row says where the work went');
    assert.deepEqual(r.speakers(), ['You', 'System', 'System', 'You']);
    assert.ok(r.q('.room-archive-speaker').querySelector('time').hasAttribute('data-i18n-authored'));
    assert.match(r.q('.room-archive-files').textContent, /plan\.pdf/);
    assert.deepEqual(r.rooms(), [['Main', 'true'], ['Alpha room', 'false']]);
    assert.equal(r.q('.room-archive-rooms').nextElementSibling.hidden, false, 'found projects carry their limit');
    assert.match(r.q('.room-archive-limit').textContent, /Only messages and notices with a confirmed room appear\./);
    assert.equal(r.q('.room-archive-feed').querySelectorAll('button').length, 0, 'a row offers no action');
    const older = r.q('[data-archive-older]');
    assert.equal(older.hidden, false);

    older.click();
    await settle();
    assert.deepEqual(r.calls[1], { room: 'main', before: 'older-1' });
    assert.deepEqual(r.rows(), ['a notice to Main', 'main owner', 'Moved to Alpha room', 'Alpha room · Done', 'with a file'],
        'older goes above');
    assert.equal(older.hidden, true);
    assert.equal(r.q('.room-archive-edge').querySelector('.room-archive-note').textContent,
        'No older messages can be shown here.', 'never "Beginning of saved history"');

    const alpha = r.q('[data-archive-room="1234567"]');
    alpha.focus();
    alpha.click();
    await settle();
    assert.deepEqual(r.calls[2], { room: '1234567', before: null });
    assert.deepEqual(r.rows(), ['alpha owner']);
    assert.deepEqual(r.rooms(), [['Main', 'false'], ['Alpha room', 'true']]);
    assert.equal(r.q('[data-archive-room="1234567"]'), alpha, 'the picker retains the actual button');
    assert.equal(r.doc.activeElement, alpha, 'and keyboard focus through loading and page settlement');
    assert.equal(r.q('.document-reader-meta').textContent, 'Read-only · Alpha room');
    assert.deepEqual(r.fetched, [], 'the reader reads nothing else and acknowledges nothing');
});

test('a stretch with nothing to place still continues, and failures keep their words and a retry', async (t) => {
    const migrating = Object.assign(new Error('migrating'), { status: 503, body: { reason_code: 'migrating' } });
    const unavailable = Object.assign(new Error('unavailable'), { status: 503, body: { reason_code: 'archive_source_unavailable' } });
    const r = reader(t, [
        { state: 'available', room: 'main', rooms: ROOMS.slice(0, 1), has_more: true, next_cursor: 'c1', messages: [] },
        migrating,
        unavailable,
        { state: 'available', room: 'main', rooms: ROOMS.slice(0, 1), has_more: false, next_cursor: null,
            messages: [message(1, 'user', 'finally')] },
    ]);
    await settle();
    assert.equal(r.q('.room-archive-edge').querySelector('.room-archive-note').textContent,
        'Nothing in this stretch could be placed here.');
    assert.equal(r.q('[data-archive-older]').hidden, false);
    assert.match(r.q('.room-archive-rooms').nextElementSibling.textContent, /project rooms may be missing/);
    assert.equal(r.q('.room-archive-rooms').nextElementSibling.hidden, false, 'Main alone is still a partial room set');
    r.q('[data-archive-older]').click();
    await settle();
    assert.match(r.q('.room-archive-status').textContent, /still being prepared/);
    assert.equal(r.q('[data-archive-retry]').hidden, false);
    r.q('[data-archive-retry]').click();
    await settle();
    assert.deepEqual(r.calls.at(-1), { room: 'main', before: 'c1' }, 'Try again repeats the read that failed');
    assert.match(r.q('.room-archive-status').textContent, /couldn’t be loaded/);
    r.q('[data-archive-retry]').click();
    await settle();
    assert.deepEqual(r.rows(), ['finally']);
    assert.equal(r.q('.room-archive-status').hidden, true);
});

test('a readable list again says so, and closing hands Main back to its own history', async (t) => {
    const r = reader(t, [{ state: 'not_needed' }]);
    await settle();
    assert.match(r.q('.room-archive-status').textContent, /can be read again/);
    assert.equal(r.q('[data-archive-older]').hidden, true);
    r.q('[data-archive-close]').click();
    assert.equal(r.handle.closed, true);
    assert.equal(r.doc.body.children.includes(r.dialog), false, 'closing removes the reader');
    assert.equal(r.restored(), 1);
});

test('closing while a read is in flight drops its answer', async (t) => {
    let release;
    const r = reader(t, [() => new Promise((resolve) => { release = resolve; })]);
    await settle();
    r.handle.close();
    release({ state: 'available', room: 'main', rooms: ROOMS, has_more: false, next_cursor: null,
        messages: [message(1, 'user', 'late')] });
    await settle();
    assert.deepEqual(r.rows(), []);
    assert.equal(r.restored(), 0);
});

test('the restoration is told once, only while Main is the room shown, beside independent partial-page and gap facts', (t) => {
    const { doc, el } = domDouble();
    const messages = el('div');
    doc.body.append(messages);
    const told = [];
    const notify = (text, kind) => {
        const notice = { text, kind, disposed: false, dispose() { this.disposed = true; } };
        told.push(notice);
        return notice;
    };
    let shown = false;
    const archive = archiveDoor({ isVisible: () => shown });
    const controls = createHistoryControls(messages, { archive, notify, isVisible: () => shown });
    t.after(controls.destroy);
    controls.endRecent(null, false, { project_registry: 'restored' });
    controls.render({ initialized: true, canOlder: false }, { complete: true });
    assert.deepEqual(told, [], 'a hidden Main announces nothing: it is owed');
    shown = true;
    controls.showNotice();
    assert.deepEqual(told.map(n => [n.text, n.kind]), [['Project list restored from its saved copy.', 'restored']]);
    controls.endRecent(new Error('Some saved history could not be loaded.'), true, { project_registry: 'restored' });
    const state = controls.render({ initialized: true, canOlder: false }, { gaps: true });
    assert.equal(state.note, 'Some saved history could not be loaded.', 'the failure is the window\'s note');
    assert.deepEqual(told.map(n => n.text), ['Project list restored from its saved copy.',
        'Some saved history could not be loaded.'], 'the failure is its own notice');
    assert.equal(told[0].disposed, false, 'and the restore notice stands beside it');
    assert.equal(messages.querySelector('.chat-load-older-note').textContent, 'Some saved history could not be loaded.',
        'the edge says the failure');
    controls.endRecent(null, false, {});
    controls.endRecent(null, false, { project_registry: 'restored' });
    controls.showNotice();
    assert.equal(told.filter(n => n.text.startsWith('Project list')).length, 1, 'a later page does not repeat it');
    controls.hideNotice();
    assert.equal(told[0].disposed, true, 'Main left or covered: the restore notice leaves with it');
    controls.endRecent(refusal());
    controls.render({ initialized: false });
    assert.equal(controls.olderButton.nextElementSibling?.hidden, false, 'its door stands instead');
});

test('disposing the door with its controls closes an open reader, cancels its read and paints nothing late', async () => {
    const { doc, el } = domDouble();
    const edge = el('button');
    doc.body.append(el('div', {}, edge));
    let release, signal;
    const door = archiveDoor({ open: (options) => openRoomArchive({ ...options, read: (request) => {
        signal = request.signal;
        return new Promise((resolve) => { release = resolve; });
    } }) });
    door.offer(edge, refusal());
    edge.nextElementSibling.click();
    await settle();
    const dialog = doc.body.children.find((node) => node.classes.has('room-archive'));
    assert.ok(dialog, 'the reader is open');
    door.dispose();
    assert.equal(doc.body.children.includes(dialog), false, 'the reader closed with its room');
    assert.equal(signal.aborted, true, 'its read was cancelled');
    assert.equal(dialog.listenerCount(), 0, 'all dialog listeners were released');
    release({ state: 'available', room: 'main', rooms: ROOMS, has_more: false, next_cursor: null,
        messages: [message(1, 'user', 'late')] });
    await settle();
    assert.equal(dialog.querySelectorAll('.room-archive-row').length, 0, 'no late paint');
    door.offer(edge, refusal());
    assert.equal(edge.nextElementSibling, null, 'a disposed door stays gone');
});


test('a stale or invalid cursor offers a fresh page of the selected room, not another stale cursor', async (t) => {
    const error = (status, reason_code) => Object.assign(new Error(reason_code), { status, body: { reason_code } });
    const page = (room, cursor, messages = []) => ({ room, rooms: ROOMS, next_cursor: cursor, has_more: !!cursor, messages });
    const r = reader(t, [page('main', null), page('1234567', 'first', [message(4, 'user', 'before')]),
        error(409, 'archive_cursor_changed'), page('1234567', 'second', [message(5, 'user', 'after')]),
        error(400, 'archive_cursor_invalid'), error(404, 'archive_room_unavailable')]);
    await settle();
    r.q('[data-archive-room="1234567"]').click();
    await settle();
    r.q('[data-archive-older]').click();
    await settle();
    const retry = r.q('[data-archive-retry]');
    assert.equal(retry.textContent, 'Reload room');
    assert.match(r.q('.room-archive-status').textContent, /saved view changed/);
    assert.deepEqual(r.rows(), ['before'], 'the last proven page remains until the owner reloads');
    retry.click();
    retry.click();
    await settle();
    assert.deepEqual(r.calls.slice(2), [{ room: '1234567', before: 'first' }, { room: '1234567', before: null }]);
    assert.deepEqual(r.rows(), ['after']);
    r.q('[data-archive-older]').click();
    await settle();
    assert.match(r.q('.room-archive-status').textContent, /can’t be continued/);
    retry.click();
    await settle();
    assert.match(r.q('.room-archive-status').textContent, /Choose another room/);
    assert.equal(retry.hidden, true, 'an unknown room does not offer a retry loop');
});


test('navigation closes even a restored archive without refocusing Main or initiating a retry', async (t) => {
    const r = reader(t, [{ state: 'not_needed' }]);
    await settle();
    r.handle.close({ restoreFocus: false });
    assert.equal(r.handle.closed, true);
    assert.equal(r.restored(), 0);
    assert.equal(r.dialog.listenerCount(), 0);
});
