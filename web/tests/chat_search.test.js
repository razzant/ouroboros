// Search in a chat (chat_search.js) over a DOM double and a scripted host: which key opens which room's search, the
// reads typing starts (one after a pause, then on to the first match), the reading of each page's candidates (a slice
// at a time, ended by new words), the list and count, Enter / Shift+Enter landing through the room's reveal with the
// keyboard left in the field, a failed landing or changed view, and every way the search ends its reads (new words,
// Esc, a hidden room, destroy) without touching the composer's draft. A candidate's shown words here are its text as a
// stream; drawing them through the room's renderer is the browser's (tests/test_history_reply_browser.py), and the
// matching of words (chat_search_words.js) is pinned below on streams.
import assert from 'node:assert/strict';
import test from 'node:test';
import { bindChatSearchKeys, createChatSearch } from '../modules/chat_search.js';
import { findIn, foldText, searchKey, snippetOf, wordStream } from '../modules/chat_search_words.js';
import { CASE_FOLDING, CASE_FOLDING_ENTRIES, UNICODE_VERSION } from '../modules/unicode_case_folding.js';
import { installDomDouble } from './dom_double.js';

// A saved message whose words hold the query (`match`) or do not (`other`), as the host answers its candidates.
const candidate = (n, text) => ({ history_id: `chat:${n}`, row_address: `row:1@t${n}#0000000000${String(n).padStart(2, '0')}`,
    role: n % 2 ? 'user' : 'assistant', ts: '', text });
const match = (n) => candidate(n, 'a needle');
const other = (n) => candidate(n, 'a pin');
const page = (candidates, { cursor = null, reached = '', importing = false, gaps = [] } = {}) => ({ query: 'needle',
    candidates, has_more: Boolean(cursor), next_cursor: cursor,
    searched: { rows: 10, reached_ts: reached, beginning: !cursor && !importing, importing, gaps } });
const streamOf = (row) => wordStream([[{ n: row.history_id }, row.text]]);
const unreadable = () => Object.assign(new Error('unreadable'), { status: 503, body: { reason_code: 'history_source_unavailable' } });
const twenty = Array.from({ length: 20 }, (_unused, index) => match(40 - index));
const flush = async () => { for (let i = 0; i < 20; i += 1) await new Promise((done) => setImmediate(done)); };

function fixture(t, { pages = [], present = () => true, answer = () => ({ state: 'found', target: null }), words = streamOf,
    pause } = {}) {
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
    const search = createChatSearch({ chatId: 1, feed, reveal, present, api, words, ...(pause ? { pause } : {}) });
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

test('Ctrl+F off a Mac opens the room in front; a dialog, a menu, another key or no room keeps the key', (t) => {
    const env = installDomDouble();
    t.after(() => env.restore());
    let opened = 0;
    const shown = env.el('div', { className: 'page' });
    env.doc.body.append(shown);
    let front = { page: shown, search: { open: () => { opened += 1; } } };
    const off = bindChatSearchKeys({ room: () => front, win: window });
    const press = (event) => window.emit('keydown', { ...event }).defaultPrevented === true;
    assert.equal(press({ key: 'f', ctrlKey: true }), true);
    assert.equal(press({ key: 'F', ctrlKey: true }), true);
    assert.equal(press({ key: 'а', code: 'KeyF', ctrlKey: true }), true, "a Cyrillic layout's F key");
    assert.equal(opened, 3);
    for (const event of [{ key: 'f' }, { key: 'f', ctrlKey: true, metaKey: true }, { key: 'f', ctrlKey: true, shiftKey: true },
        { key: 'f', metaKey: true }, { key: 'а', code: 'KeyF', metaKey: true },
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

test('on a Mac Cmd+F opens the room in front and Ctrl+F stays the field\'s own (a character forward)', (t) => {
    const env = installDomDouble();
    t.after(() => env.restore());
    window.navigator = { platform: 'MacIntel' };
    let opened = 0;
    const shown = env.el('div', { className: 'page' });
    env.doc.body.append(shown);
    const off = bindChatSearchKeys({ room: () => ({ page: shown, search: { open: () => { opened += 1; } } }), win: window });
    t.after(off);
    const press = (event) => window.emit('keydown', { ...event }).defaultPrevented === true;
    assert.equal(press({ key: 'f', metaKey: true }), true);
    assert.equal(press({ key: 'а', code: 'KeyF', metaKey: true }), true, "a Cyrillic layout's F key");
    for (const event of [{ key: 'f', ctrlKey: true }, { key: 'а', code: 'KeyF', ctrlKey: true },
        { key: 'f', ctrlKey: true, metaKey: true }]) {
        assert.equal(press(event), false, JSON.stringify(event));
    }
    assert.equal(opened, 2);
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
    const drawn = [...f.parts().results];
    note.querySelector('[data-find-more]').click();
    await flush();
    assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1']);
    assert.equal(f.parts().count.textContent, '21 found');
    const after = [...f.parts().results];
    assert.equal(after.length, 21);
    assert.equal(drawn.filter((item, index) => item !== after[index]).length, 0,
        'the matches already listed stay the same options; the older one is added');
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
    assert.deepEqual([...f.parts().results].slice(0, 3).map((item) => item.getAttribute('aria-selected')),
        ['false', 'true', 'false'], 'the selection moves to the landed match');
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
    assert.doesNotMatch(f.parts().note.textContent, /no longer in saved history/, 'a landing clears the old failure');
    assert.equal(f.parts().note.textContent, 'The match is in the saved message, but its words are not shown here.',
        'this fixture returns no drawn target, so the new explanation remains');
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

// A row's words as the search reads them: pieces of text nodes (`{n}` stands for one), null an element's edge; each
// match as [first node, offset, last node, end offset], as its landed Range spans them.
const located = (pieces, query) => {
    const stream = wordStream(pieces);
    return findIn(stream, searchKey(query)).map(([start, end]) => {
        const first = stream.places[stream.from[start]], last = stream.places[stream.to[end - 1] - 1];
        return [first[0].n, first[1], last[0].n, last[1] + 1];
    });
};

test('words are found across inline nodes, line breaks and white space; never across what the row does not show', () => {
    const [a, b] = [{ n: 'a' }, { n: 'b' }];
    assert.deepEqual(located([[a, 'We saw the King'], [b, 'fisher  flash']], 'kingfisher FLASH'), [['a', 11, 'b', 13]],
        'one match over two adjacent inline nodes and a run of spaces');
    assert.deepEqual(located([[a, 'first line'], [null, ' '], [b, '\n  second line']], 'line second'),
        [['a', 6, 'b', 9]], 'an edge between blocks reads as one space');
    assert.deepEqual(located([[a, 'hello'], [null, ' '], [null, ' '], [b, 'world']], ' hello\t world '), [['a', 0, 'b', 5]]);
    assert.deepEqual(located([[a, 'tail of a reply']], 'Kingfisher'), [], 'words the row does not show find nothing');
    assert.deepEqual(located([[a, 'kingfisher, then KINGFISHER']], 'Kingfisher'), [['a', 0, 'a', 10], ['a', 17, 'a', 27]],
        'every occurrence, in order and without overlaps');
    assert.deepEqual(located([[a, 'pattern a.b*c, never aXbbc']], 'A.B*C'), [['a', 8, 'a', 13]], 'literal, never a pattern');
    assert.deepEqual(located([[a, 'a*b*c']], 'abc'), [], 'literal code keeps its marks');
});

test('Unicode 17 full case folding: Straße holds STRASSE, sigmas, ligatures and supplementary letters; é is never e', () => {
    const [a, b] = [{ n: 'a' }, { n: 'b' }];
    assert.deepEqual(located([[a, 'Die Straße endet']], 'STRASSE'), [['a', 4, 'a', 10]], 'ß folds to ss');
    assert.deepEqual(located([[a, 'DIE STRASSE']], 'straße'), [['a', 4, 'a', 11]]);
    assert.deepEqual(located([[a, 'Straße']], 's'), [['a', 0, 'a', 1]], 'never the s inside ß');
    assert.deepEqual(located([[a, 'ΟΔΟΣ οδος οδοσ']], 'οδοσ'), [['a', 0, 'a', 4], ['a', 5, 'a', 9], ['a', 10, 'a', 14]],
        'capital, final and medial sigma fold alike');
    assert.deepEqual(located([[a, 'a ﬁne day']], 'FINE'), [['a', 2, 'a', 5]], 'a ligature folds to its letters');
    assert.deepEqual(located([[a, '𐐀𐐁 Deseret']], '𐐨𐐩'), [['a', 0, 'a', 4]], 'supplementary letters, whole surrogate pairs');
    assert.deepEqual(located([[a, 'İstanbul and istanbul']], 'İSTANBUL'), [['a', 0, 'a', 8]],
        'İ is i with a dot above (F), never the Turkic I → ı (T)');
    assert.deepEqual(located([[a, 'café and cafe']], 'cafe'), [['a', 9, 'a', 13]], 'a diacritic is never stripped');
    assert.deepEqual(located([[a, 'Cafe\u0301 noir']], 'café'), [['a', 0, 'a', 5]], 'decomposed text matches composed (NFC)');
    assert.deepEqual(located([[a, 'Cafe'], [b, '\u0301 noir']], 'Café noir'), [['a', 0, 'b', 6]],
        'a mark in the next node composes with its base');
    assert.deepEqual(located([[a, '\u1100'], [b, '\u1161\u11a8 끝']], '각'), [['a', 0, 'b', 2]],
        'NFC composes Hangul jamo across nodes too');
    assert.deepEqual(located([[a, 'q\u0303 q']], 'q'), [['a', 3, 'a', 4]], 'never the base of a letter its mark does not compose with');
});

test('the pinned table is Unicode 17.0.0 C+F, complete and decoded as written', () => {
    const entries = CASE_FOLDING.split(';');
    assert.equal(UNICODE_VERSION, '17.0.0');
    assert.equal(entries.length, CASE_FOLDING_ENTRIES);
    assert.equal(new Set(entries.map((entry) => entry.split(':')[0])).size, entries.length, 'one folding per code point');
    const fold = (text) => foldText(text).folded;
    assert.equal(fold('I'), 'i', 'C, never T');
    assert.equal(fold('\u1E9E'), 'ss', 'capital sharp s by its full folding, never the simple S line');
    assert.equal(fold('\uA7CB'), '\u0264', 'a capital letter Unicode 16 added');
    assert.equal(fold('\u2C2F'), '\u2C5F', 'Glagolitic capital caudate chrivi (Unicode 14)');
    assert.equal(fold('ꭰ'), 'Ꭰ', 'Cherokee small letters fold to the capitals');
    assert.equal(fold('\u{1E921}'), '\u{1E943}', 'the last entry');
    assert.equal(fold('\u0378x\uFFFF'), '\u0378x\uFFFF', 'a code point the table does not list is itself');
    const { folded, start, end } = foldText('aßc', true);
    assert.deepEqual([folded, start, end], ['assc', [0, 1, 1, 2], [1, 2, 2, 3]], 'an expansion maps back to its one character');
});

test('a snippet windows the shown words around the first match and marks every match in it', () => {
    const words = `${'lead '.repeat(20)}the needle and NEEDLE ${'tail '.repeat(40)}`;
    const stream = wordStream([[{}, words]]);
    const parts = snippetOf(stream, findIn(stream, searchKey('needle')));
    assert.equal(parts[0].text, '…');
    assert.equal(parts.at(-1).text, '…');
    assert.deepEqual(parts.filter((part) => part.match).map((part) => part.text), ['needle', 'NEEDLE']);
    assert.ok(parts.map((part) => part.text).join('').length < 180);
    const short = wordStream([[{}, 'Die Straße']]);
    assert.deepEqual(snippetOf(short, findIn(short, searchKey('strasse'))),
        [{ text: 'Die ', match: false }, { text: 'Straße', match: true }]);
});

test('candidates whose shown words hold no match are read and dropped; one that cannot be drawn leaves the count partial', async (t) => {
    const broken = candidate(5, 'needle');
    const f = fixture(t, { pages: [page([other(9), match(8), broken, other(7), match(6)])],
        words: (row) => { if (row === broken) throw new Error('no renderer'); return streamOf(row); } });
    t.mock.method(console, 'warn', () => {});
    f.search.open();
    await f.type('needle');
    assert.deepEqual([...f.parts().results].map((item) => item.querySelector('.chat-find-snippet').textContent), ['a needle', 'a needle']);
    assert.equal(f.parts().count.textContent, '2+', 'an unread message is never "no match"');
    assert.match(f.parts().note.textContent, /1 saved messages could not be read as shown, so they were not searched\./);
});

test('a page is read a slice at a time; new words end the old reading, whose matches never appear', async (t) => {
    const gates = [];
    const slow = (row) => { const until = Date.now() + 15; while (Date.now() < until) { /* one slice */ } return streamOf(row); };
    const f = fixture(t, { pages: [page([match(9), match(8), match(7)]), page([candidate(4, 'needles here')])], words: slow,
        pause: () => new Promise((done) => gates.push(done)) });
    f.search.open();
    await f.type('needle');
    assert.equal(gates.length, 1, 'the reading yields once its slice is spent');
    assert.equal(f.parts().count.textContent, '1+', 'what it found so far is counted, partially');
    assert.match(f.parts().note.textContent, /^Searching saved messages…/, 'and the reading says it goes on');
    await f.type('needles');
    gates.shift()();
    await flush();
    gates.splice(0).forEach((done) => done());
    await flush();
    assert.deepEqual([...f.parts().results].map((item) => item.querySelector('.chat-find-snippet').textContent), ['needles here']);
    assert.equal(f.parts().count.textContent, '1 found');
});

test('a landing failure says its known reason: history being upgraded or unreadable', async (t) => {
    const migrating = Object.assign(new Error('migrating'), { status: 503, body: { reason_code: 'migrating' } });
    const answers = [{ state: 'unknown', error: migrating }, { state: 'unknown', error: unreadable() }, { state: 'ambiguous' }];
    const f = fixture(t, { pages: [page(twenty, { cursor: 'c1' })], answer: () => answers.shift() || { state: 'found', target: null } });
    f.search.open();
    await f.type('needle');
    await f.key('Enter');
    assert.equal(f.parts().note.textContent.startsWith('Saved history is being upgraded. Try again shortly.'), true);
    await f.key('Enter');
    assert.match(f.parts().note.textContent, /^Saved history holding this message could not be read\. Try again\./);
    await f.key('Enter');
    assert.match(f.parts().note.textContent, /^This reference matches more than one saved message\./);
});

test('an unshown match is explained instead of claiming visible words', async (t) => {
    const f = fixture(t, { pages: [page([match(1)])] });
    f.search.open();
    await f.type('needle');
    await f.key('Enter');
    assert.equal(f.parts().note.textContent, 'The match is in the saved message, but its words are not shown here.');
    await f.type('another query');
    assert.doesNotMatch(f.parts().note.textContent, /not shown here/);
});

test('a read after a failed landing says what it read, never the old landing\'s reason', async (t) => {
    const f = fixture(t, { pages: [page(twenty, { cursor: 'c1', reached: '2026-10-01T09:00:00Z' }),
        page([], { gaps: ['malformed_jsonl'] })], answer: () => ({ state: 'absent' }) });
    f.search.open();
    await f.type('needle');
    f.parts().results[19].emit('click');
    await flush();
    assert.match(f.parts().note.textContent, /^This message is no longer in saved history\./);
    await f.key('Enter');  // past the loaded matches: the read of older history, with no new match, says what it read
    assert.deepEqual(f.reads.map((read) => read.cursor), [null, 'c1']);
    assert.equal(f.landed().at(-1), twenty[19].row_address);
    assert.match(f.parts().note.textContent, /Some saved history could not be read, so it was not searched\./);
    assert.doesNotMatch(f.parts().note.textContent, /no longer in saved history/);
});
