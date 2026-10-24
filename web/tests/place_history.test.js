// Source proposal by readonly child 87500533; materialized and executed by parent.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installCardDom, TextNode } from './helpers/card_dom.js';
import { applyPayload, createTranslator, setMissTransport } from '../modules/i18n.js';
import { normalizePlace, readPlaceRooms } from '../modules/consciousness_api.js';
import { openPlaceHistory } from '../modules/place_history.js';
import { openConnectorPlaces } from '../modules/place_sheet.js';
const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const unsafeText = '<img src="x" onerror="bad()"> & "hello"';
const rawPlace = (id = 'one') => ({ place_id: id, connector: 'telegram-user-session', label: unsafeText, account_id: `account-${id}`, state: 'active' });
const place = normalizePlace(rawPlace());
const room = { room_chat_id: 5_000_000_123, address: unsafeText, unanswered_count: null };
const index = (rooms = [room]) => ({ place_id: place.id, rooms });
const row = (id, text = unsafeText) => ({ history_id: id, role: 'user', text, sender_label: unsafeText });
const page = (messages, next = null) => ({ messages, next_cursor: next, has_more: next !== null, window: { complete: false } });
function fixture(t) {
    const dom = installCardDom(); const { doc, win } = dom;
    // Missing fixture DOM mechanics, not a replacement for production behavior.
    doc.defaultView = win; win.getComputedStyle = () => ({ visibility: 'visible' });
    const create = doc.createElement;
    doc.createElement = (tag) => {
        const node = create(tag);
        Object.defineProperties(node, { isConnected: { get: () => doc.body.contains(node) }, childElementCount: { get: () => node.children.length } });
        node.getClientRects = () => node.isConnected ? [{}] : [];
        node.scrollTop = 0; node.scrollHeight = 0;
        node.prepend = (...nodes) => { for (const child of [...nodes].reverse()) node.insertBefore(child, node.firstChild); };
        const matches = node.matches.bind(node);
        node.matches = (selector) => matches(selector.replace(/:disabled/g, '[disabled]'));
        return node;
    };
    doc.createDocumentFragment = () => doc.createElement('#fragment');
    doc.createTextNode = (text) => new TextNode(text, doc);
    doc.createTreeWalker = (root) => {
        const texts = []; const visit = (node) => { if (node.nodeType === 3) {
            Object.defineProperty(node, 'parentElement', { configurable: true, get: () => node.parentNode }); texts.push(node);
        } else (node.childNodes || []).forEach(visit); }; visit(root);
        let i = 0; return { nextNode: () => texts[i++] || null };
    };
    const requests = []; let answer = () => { throw new Error('Unexpected request: no network allowed'); };
    t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
        const request = { url: String(url), ...init }; requests.push(request);
        assert.equal(init.method || 'GET', 'GET', 'navigation must not write');
        const value = await answer(request);
        return { ok: true, status: 200, json: async () => value };
    });
    t.after(() => { for (const button of doc.body.querySelectorAll('[data-history-close], [data-place-close]')) button.click(); dom.restore(); });
    return { doc, requests, reply(fn) { answer = fn; }, q(selector) { return doc.body.querySelector(selector); } };
}
test('reader forwards large room ID and retries the same cursor after failures', async (t) => {
    const f = fixture(t); const cursor = 'opaque /+?&cursor'; let attempt = 0;
    f.reply(({ url }) => {
        if (url.includes('/rooms')) return index();
        const query = new URL(url, 'http://local').searchParams;
        assert.equal(query.get('chat_id'), String(room.room_chat_id));
        assert.equal(query.get('cursor'), attempt === 0 ? null : cursor); attempt += 1;
        if (attempt === 1) return page([row('new')], cursor);
        if (attempt === 2) throw new Error(unsafeText);
        if (attempt === 3) return page([], cursor);
        return page([row('old', 'older'), row('new')]);
    });
    const reader = openPlaceHistory(place); await tick();
    assert.equal(f.q('[data-history-room]').textContent, `${unsafeText} · work count unknown`);
    assert.equal(f.q('img'), null); f.q('[data-history-room]').click(); await tick();
    assert.equal(f.q('[data-history-address]').textContent, unsafeText);
    assert.equal(f.q('.chat-markdown').textContent, unsafeText);
    assert.equal(f.q('[data-history-messages]').querySelector('p').textContent, unsafeText);
    assert.equal(f.q('img'), null); assert.match(f.q('[data-history-coverage]').textContent, /incomplete coverage/);
    f.q('[data-history-older]').click(); await tick();
    assert.equal(f.q('[data-history-status]').textContent, unsafeText); assert.equal(f.q('img'), null);
    f.q('[data-history-retry]').click(); await tick(); assert.match(f.q('[data-history-status]').textContent, /no page was skipped/);
    f.q('[data-history-retry]').click(); await tick(); const messages = f.q('[data-history-messages]');
    assert.equal(messages.childElementCount, 2);
    assert.equal(messages.firstElementChild.querySelector('.chat-markdown').textContent, 'older');
    assert.equal(f.q('[data-history-older]').hidden, true); assert.equal(attempt, 4); assert.equal(f.q('textarea'), null); reader.close();
});
test('room API validates identity, safe integers and unknown count', async (t) => {
    const f = fixture(t); const controller = new AbortController(); f.reply(() => index());
    assert.deepEqual(await readPlaceRooms(place.id, { signal: controller.signal }), [room]);
    assert.equal(f.requests[0].signal, controller.signal); assert.equal(f.requests[0].cache, 'no-store');
    assert.equal(f.requests[0].url, `/api/places/${encodeURIComponent(place.id)}/rooms`);
    for (const invalid of [{ ...index(), place_id: 'different' }, { ...index(), rooms: null }, index([{ ...room, room_chat_id: Number.MAX_SAFE_INTEGER + 1 }]), index([{ ...room, room_chat_id: 0 }]), index([{ ...room, room_chat_id: String(room.room_chat_id) }]), index([{ ...room, unanswered_count: -1 }])]) {
        f.reply(() => invalid); await assert.rejects(readPlaceRooms(place.id), /could not be read/);
    }
});
test('close aborts pending reads and ignores late data', async (t) => {
    const f = fixture(t);
    for (const pendingRooms of [true, false]) {
        const pending = deferred(); f.reply(({ url }) => url.includes('/rooms') && !pendingRooms ? index() : pending.promise);
        const reader = openPlaceHistory(place); const dialog = f.q('.place-history'); await tick();
        if (!pendingRooms) { f.q('[data-history-room]').click(); await tick(); }
        const signal = f.requests.at(-1).signal; reader.close(); reader.close();
        assert.equal(reader.closed, true); assert.equal(signal.aborted, true); assert.equal(f.q('.place-history'), null);
        const before = dialog.innerHTML; pending.resolve(pendingRooms ? index() : page([row('late')])); await tick();
        assert.equal(dialog.innerHTML, before); assert.equal(dialog.querySelector('[data-history-messages]').childElementCount, 0);
    }
});
test('connector door zero/one/many never connects or chooses owner accounts', async (t) => {
    const f = fixture(t);
    for (const count of [0, 1, 2]) {
        const supplied = Array.from({ length: count }, (_, i) => rawPlace(String(i)));
        f.reply(() => ({ places: [...supplied, { ...rawPlace('foreign'), connector: 'elsewhere' }] }));
        const before = f.requests.length; const chooser = openConnectorPlaces('telegram-user-session'); await tick();
        assert.equal(f.requests.length - before, 1);
        if (count === 0) { assert.equal(chooser.closed, false); assert.ok(f.q('[data-connector-connect]')); assert.equal(f.q('[data-place-act="confirm"]'), null); }
        else if (count === 1) { assert.equal(chooser.closed, true); assert.ok(f.q('[data-place-act="conversations"]')); assert.equal(f.q('[data-connector-place]'), null); }
        else {
            assert.equal(chooser.closed, false); assert.equal(f.doc.body.querySelectorAll('[data-connector-place]').length, 2);
            assert.equal(f.q('[data-place-act="conversations"]'), null); assert.match(f.q('[data-connector-place="1"]').textContent, /account-1/);
            f.q('[data-connector-place="1"]').click(); await tick(); assert.equal(chooser.closed, true);
            assert.match(f.q('[data-place-meta]').textContent, /account-1/); assert.ok(f.q('[data-place-act="conversations"]'));
        }
        assert.equal(f.q('img'), null); assert.equal(f.requests.length - before, 1); chooser.close(); f.q('[data-place-close]')?.click();
    }
});

test('translation overlay preserves captured authored words but translates chrome', async (t) => {
    const f = fixture(t); f.reply(({ url }) => url.includes('/rooms') ? index() : page([row('authored', 'Settings')]));
    const reader = openPlaceHistory(place); await tick(); f.q('[data-history-room]').click(); await tick();
    const chrome = f.doc.createElement('button'); chrome.textContent = 'Settings'; f.doc.body.append(chrome);
    setMissTransport(() => Promise.resolve({ ok: true }));
    applyPayload({ language: 'fr', english: false, revision: 1, entries: { Settings: { text: 'Translated chrome' } } });
    t.after(() => { applyPayload({ language: '', english: true, entries: {} }); setMissTransport(null); });
    const translator = createTranslator({ skipRoots: '[data-i18n-authored]', userContent: '', excludeSelector: '' });
    translator.applyTo(f.doc.body);
    assert.equal(f.q('.chat-markdown').textContent, 'Settings');
    assert.equal(chrome.textContent, 'Translated chrome');
    assert.equal(f.q('[data-history-messages]').querySelector('p').textContent, unsafeText);
    reader.close();
});
