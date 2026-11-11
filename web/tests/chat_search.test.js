// Search in a chat (chat_search.js) over a DOM double and a scripted host: which key opens which room's search, the
// reads typing starts (one after a pause, then on to the first match), the list and count, Enter / Shift+Enter landing
// through the room's reveal with the keyboard left in the field, a failed landing or changed view, and every way the
// search ends its reads (new words, Esc, a hidden room, destroy) without touching the composer's draft.
import assert from 'node:assert/strict';
import test from 'node:test';
import { bindChatSearchKeys, createChatSearch } from '../modules/chat_search.js';
import { installDomDouble } from './dom_double.js';

const match = (n) => ({ history_id: `chat:${n}`, row_address: `row:1@t${n}#0000000000${String(n).padStart(2, '0')}`,
    role: n % 2 ? 'user' : 'assistant', ts: '', snippet: [{ text: 'a ', match: false }, { text: 'needle', match: true }] });
const page = (matches, { cursor = null, reached = '', importing = false, gaps = [] } = {}) => ({ query: 'needle', matches,
    has_more: Boolean(cursor), next_cursor: cursor,
    searched: { rows: 10, reached_ts: reached, beginning: !cursor && !importing, importing, gaps } });
const unreadable = () => Object.assign(new Error('unreadable'), { status: 503, body: { reason_code: 'history_source_unavailable' } });
const twenty = Array.from({ length: 20 }, (_unused, index) => match(40 - index));
const flush = async () => { for (let i = 0; i < 20; i += 1) await new Promise((done) => setImmediate(done)); };

function fixture(t, { pages = [], present = () => true, answer = () => ({ state: 'found', target: null }) } = {}) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    const draft = el('textarea', { attrs: { id: 'chat-input' } });
    draft.value = 'my unsent draft';
    const room = el('div', { className: 'page' }, feed, draft);
    doc.body.append(room);
    const reads = [], landings = [];
    const api = { chatSearch: (params) => {
        reads.push(params);
        const next = pages.shift();
        if (typeof next === 'function') return next(params);
        return next instanceof Error ? Promise.reject(next) : Promise.resolve(next || page([]));
    } };
    const reveal = async (address, isCurrent) => {
        landings.push({ address, current: isCurrent() });
        return answer(address);
    };
    const search = createChatSearch({ chatId: 1, feed, reveal, present, api });
    t.after(() => { search.destroy(); env.restore(); });
    const parts = () => {
        const root = room.querySelector('.chat-find');
        return { root, input: root?.querySelector('.chat-find-input'), count: root?.querySelector('.chat-find-count'),
            note: root?.querySelector('.chat-find-note'), list: root?.querySelector('.chat-find-results'),
            results: root?.querySelectorAll('.chat-find-result') || [] };
    };
    const type = async (words) => {
        const { input } = parts();
        input.value = words;
        input.emit('input');
        t.mock.timers.tick(220);
        await flush();
    };
    const key = async (name, extra = {}) => {
        const event = parts().input.emit('keydown', { key: name, ...extra });
        await flush();
        return event;
    };
    const landed = () => landings.map((landing) => landing.address);
    return { ...env, feed, draft, room, search, reads, landed, landings, parts, type, key };
}

test('Ctrl+F or Cmd+F opens the room in front; a dialog, a menu, another key or no room keeps the key', (t) => {
    const env = installDomDouble();
    t.after(() => env.restore());
    let opened = 0;
    const shown = env.el('div', { className: 'page' });
    env.doc.body.append(shown);
    let front = { page: shown, search: { open: () => { opened += 1; } } };
    const off = bindChatSearchKeys({ room: () => front, win: window });
    const press = (event) => window.emit('keydown', { ...event }).defaultPrevented === true;
    assert.equal(press({ key: 'f', ctrlKey: true }), true);
    assert.equal(press({ key: 'F', metaKey: true }), true);
    assert.equal(press({ key: 'а', code: 'KeyF', ctrlKey: true }), true, "a Cyrillic layout's F key");
    assert.equal(opened, 3);
    for (const event of [{ key: 'f' }, { key: 'f', ctrlKey: true, metaKey: true }, { key: 'f', ctrlKey: true, shiftKey: true },
        { key: 'f', metaKey: true, altKey: true }, { key: 'u', code: 'KeyF', ctrlKey: true },
        { key: 'f', ctrlKey: true, isComposing: true }]) {
        assert.equal(press(event), false, JSON.stringify(event));
    }
    const menu = env.el('div', { attrs: { role: 'menu' } }, env.el('button'));
    env.doc.body.append(menu);
    menu.children[0].focus();
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'a menu keeps its keys');
    env.doc.body.focus();
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'a pointer-opened menu also owns the key with focus outside');
    menu.setAttribute('data-leaving', '');
    assert.equal(press({ key: 'f', ctrlKey: true }), true, 'a departing menu picture does not own keys');
    menu.removeAttribute('data-leaving');
    menu.hidden = true;
    assert.equal(press({ key: 'f', ctrlKey: true }), true, 'a hidden menu does not own keys');
    menu.remove();
    const dialog = env.el('dialog', { attrs: { open: '' } });
    env.doc.body.append(dialog);
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'an open dialog keeps the page');
    dialog.remove();
    shown.hidden = true;
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'a room no longer shown (a stale Project) never takes the key');
    shown.hidden = false;
    shown.setAttribute('inert', '');
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'nor a room a layer covers');
    shown.removeAttribute('inert');
    front = null;
    assert.equal(press({ key: 'f', ctrlKey: true }), false, 'another page keeps its own find');
    front = { page: shown, search: { open: () => { opened += 1; } } };
    off();
    assert.equal(press({ key: 'f', ctrlKey: true }), false);
    assert.equal(opened, 5);
});

test('typing reads the room once after a pause and reads on to the first match; the count is exact only at the end', async (t) => {
    const f = fixture(t, { pages: [page([], { cursor: 'c1', reached: '2026-10-01T09:00:00Z' }), page([match(3)], { cursor: 'c2' }),
        page([match(2), match(1)])] });
    f.draft.focus();
    f.search.open();
    const { input, root, list } = f.parts();
    assert.equal(f.doc.activeElement, input, 'the keyboard goes to the field');
    assert.equal(root.hidden, false);
    assert.equal(f.parts().note.hidden, true, 'a fresh sheet is the field alone: no empty status line under it');
    assert.equal(f.parts().list.hidden, true);
    input.value = 'nee';
    input.emit('input');
    t.mock.timers.tick(100);
    input.value = '  needle  ';
    input.emit('input');
    t.mock.timers.tick(219);
    assert.deepEqual(f.reads, [], 'nothing is read while the words are still being typed');
    t.mock.timers.tick(1);
    await flush();
    assert.deepEqual(f.reads.map(({ q, cursor, chatId }) => [q, cursor, chatId]),
        [['needle', null, 1], ['needle', 'c1', 1], ['needle', 'c2', 1]], 'a page without a match reads on');
    const { count, results, note } = f.parts();
    assert.equal(results.length, 3);
    assert.equal(results[0].querySelector('mark').textContent, 'needle');
    assert.equal(count.textContent, '3 found');
    assert.equal(note.hidden, true);
    assert.equal(list.hidden, false);
    assert.equal(f.draft.value, 'my unsent draft');
    assert.equal(list.getAttribute('data-i18n-authored'), '', 'the snippets are content, never chrome to translate');
    assert.equal(count.getAttribute('data-i18n-skip'), '');
    assert.equal(note.getAttribute('data-i18n-skip'), '');
});

test('a search that stops with older messages left says how far it read and offers to read on', async (t) => {
    const f = fixture(t, { pages: [page(twenty, { cursor: 'c1', reached: '2026-10-01T09:00:00Z' }), page([match(1)])] });
    f.search.open();
    await f.type('needle');
    assert.equal(f.reads.length, 1, 'a full page of matches is enough until the owner asks for more');
    const { count, note } = f.parts();
    assert.equal(count.textContent, '20+');
    assert.match(note.textContent, /^Searched back to .*Search older messages$/);
    note.querySelector('[data-find-more]').click();
    await flush();
    assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1']);
    assert.equal(f.parts().count.textContent, '21 found');
});

test('Enter lands on the next match through the room\'s reveal, keeping the keyboard; past the loaded ones it reads on', async (t) => {
    const f = fixture(t, { pages: [page(twenty, { cursor: 'c1' }), page([match(1)])] });
    f.search.open();
    await f.type('needle');
    const { input } = f.parts();
    assert.equal((await f.key('Enter')).defaultPrevented, true);
    assert.deepEqual(f.landed(), [twenty[0].row_address]);
    assert.equal(f.landings[0].current, true);
    assert.equal(f.doc.activeElement, input);
    assert.equal(f.parts().count.textContent, '1 of 20+');
    assert.equal(input.getAttribute('aria-activedescendant'), f.parts().results[0].id);
    assert.equal(f.parts().root.querySelector('.chat-find-newer').disabled, true);
    await f.key('ArrowDown');
    await f.key('Enter', { shiftKey: true });
    await f.key('ArrowUp');  // the newest match has no newer one
    assert.deepEqual(f.landed().slice(1), [twenty[1].row_address, twenty[0].row_address]);
    f.parts().results[19].emit('click');
    await flush();
    assert.equal(f.landed().at(-1), twenty[19].row_address, 'a press on a match lands on it');
    assert.equal(f.reads.length, 1);
    await f.key('Enter');
    assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1'], 'Enter past the loaded matches reads older');
    assert.equal(f.landed().at(-1), match(1).row_address);
    assert.equal(f.parts().count.textContent, '21 of 21');
    assert.equal(f.parts().root.querySelector('.chat-find-older').disabled, true);
    assert.equal(f.draft.value, 'my unsent draft');
});

test('Enter right after typing lands on the first match of those words', async (t) => {
    const f = fixture(t, { pages: [page([match(7)])] });
    f.search.open();
    const { input } = f.parts();
    input.value = 'needle';
    input.emit('input');
    await f.key('Enter');
    assert.deepEqual(f.reads.map((read) => read.q), ['needle']);
    assert.deepEqual(f.landed(), [match(7).row_address]);
});

test('a landing that fails says why and keeps the list; a changed view searches again from now', async (t) => {
    const changed = Object.assign(new Error('changed'), { status: 409, body: { reason_code: 'search_view_changed' } });
    const answers = [{ state: 'absent' }];
    const f = fixture(t, { pages: [page([match(9)], { cursor: 'c1' }), changed, page([match(9), match(8)])],
        answer: () => answers.shift() || { state: 'found', target: null } });
    f.search.open();
    await f.type('needle');
    assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1', null], 'the changed room is searched again from now');
    assert.equal(f.parts().count.textContent, '2 found');
    await f.key('Enter');
    assert.equal(f.parts().note.textContent, 'This message is no longer in saved history.');
    assert.equal(f.parts().list.hidden, false);
    await f.key('Enter');
    assert.equal(f.parts().note.hidden, true, 'a landing clears the reason');
});

test('a read that fails says so with Try again; history being upgraded is said as such', async (t) => {
    const migrating = Object.assign(new Error('migrating'), { status: 503, body: { reason_code: 'migrating' } });
    const f = fixture(t, { pages: [migrating, page([match(2)])] });
    f.search.open();
    await f.type('needle');
    const { note } = f.parts();
    assert.equal(f.parts().count.textContent, '', 'a first read that failed counts nothing, never "0 found"');
    assert.match(note.textContent, /^Saved history is being upgraded\. Try again shortly\.Try again$/);
    note.querySelector('[data-find-more]').click();
    await flush();
    assert.equal(f.parts().count.textContent, '1 found');
});

test('a count is exact only for the whole room: the import frontier, unreadable lines and a failed read leave it partial', async (t) => {
    await t.test('the import frontier', async (st) => {
        const f = fixture(st, { pages: [page([match(2)], { importing: true, reached: '2026-10-01T09:00:00Z' })] });
        f.search.open();
        await f.type('needle');
        assert.equal(f.parts().count.textContent, '1+', 'older history still importing was not searched');
        assert.match(f.parts().note.textContent, /Older history is still being upgraded and was not searched\./);
        await f.key('Enter');
        assert.equal(f.parts().count.textContent, '1 of 1+');
    });
    await t.test('unreadable lines at the beginning', async (st) => {
        const f = fixture(st, { pages: [page([match(2)], { gaps: ['malformed_jsonl'] })] });
        f.search.open();
        await f.type('needle');
        assert.equal(f.parts().count.textContent, '1+', 'the beginning was reached past lines that could not be read');
        assert.match(f.parts().note.textContent, /Some saved history could not be read, so it was not searched\./);
    });
    await t.test('a failed continuation after a match', async (st) => {
        const f = fixture(st, { pages: [page([match(3)], { cursor: 'c1' }), unreadable(), page([match(1)])] });
        f.search.open();
        await f.type('needle');
        assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1']);
        assert.equal(f.parts().count.textContent, '1+', 'a failed continuation leaves the count partial');
        assert.match(f.parts().note.textContent, /^Saved history could not be searched\.Try again$/);
        f.parts().note.querySelector('[data-find-more]').click();
        await flush();
        assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1', 'c1'], 'Try again reads on from the same cut');
        assert.equal(f.parts().count.textContent, '2 found');
    });
    await t.test('a failed first read', async (st) => {
        const f = fixture(st, { pages: [unreadable()] });
        f.search.open();
        await f.type('needle');
        assert.equal(f.parts().count.textContent, '', 'a failed first read counts nothing, never "0 found"');
        assert.match(f.parts().note.textContent, /^Saved history could not be searched\.Try again$/);
    });
});

test('new words, Esc, a hidden room and destroy end the reads; Esc gives the field back with the draft untouched', async (t) => {
    const signals = [];
    let shown = true;
    const held = (params) => { signals.push(params.signal); return new Promise(() => {}); };
    const f = fixture(t, { pages: [held, held, held], present: () => shown });
    f.draft.focus();
    f.search.open();
    await f.type('needle');
    await f.type('needles');
    assert.equal(signals.length, 2);
    assert.equal(signals[0].aborted, true, 'new words abort the read of the old ones');
    const escape = f.parts().input.emit('keydown', { key: 'Escape' });
    assert.equal(escape.defaultPrevented, true, "the room's layer does not also take this Esc");
    assert.equal(signals[1].aborted, true);
    assert.equal(f.parts().root.hidden, true);
    assert.equal(f.doc.activeElement, f.draft, 'the focus returns where it was');
    assert.equal(f.draft.value, 'my unsent draft');
    f.search.open();
    t.mock.timers.tick(220);
    await flush();
    assert.equal(f.reads.length, 3, 'reopened, the kept words are read again');
    shown = false;
    await f.type('other');
    assert.equal(f.parts().root.hidden, true, 'a room that left the screen closes its search instead of reading');
    assert.equal(f.reads.length, 3);
    assert.equal(signals[2].aborted, true);
    f.search.destroy();
    assert.equal(f.room.querySelector('.chat-find'), null);
    f.search.open();
    assert.equal(f.room.querySelector('.chat-find'), null, 'a destroyed search stays gone');
});
