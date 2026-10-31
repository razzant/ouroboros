// The start ledger (holds.js): a start is written before the code's first turn and leaves with its settle; what the
// previous page left becomes this page's holds, read once and bound to the revision; the owner's Stop / Close stays
// until released; `?hold=all` holds everything for one load; the retired boot marker is cleared; storage that
// refuses a write falls back to the session; a page that hides settles its starts; the report names the found holds
// once with the raw surface facts.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createHolds, HELD_LINES, HOLDS_KEY, reportFoundHolds, settleAfterFrameLoad } from '../modules/holds.js';

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

function storage({ refuse = false } = {}) {
    const map = new Map();
    return { map, getItem: (key) => (map.has(key) ? map.get(key) : null),
        setItem(key, value) { if (refuse) throw new Error('QuotaExceededError'); map.set(key, String(value)); },
        removeItem: (key) => map.delete(key) };
}

function page({ href = 'http://127.0.0.1:8765/#chat', local = storage(), session = storage(), now = 5 } = {}) {
    const listeners = new Map();
    const win = { location: { href }, localStorage: local, sessionStorage: session,
        history: { state: null, replaceState(_state, _title, url) { win.location.href = url; } },
        addEventListener: (type, fn) => listeners.set(type, fn), emit: (type) => listeners.get(type)?.() };
    return { win, local, session, holds: createHolds({ win, now: () => now }) };
}

const entries = (store) => JSON.parse(store.map.get(HOLDS_KEY) || 'null')?.entries || {};

test('a start stands from before its first turn until its settle; this page never reads its own start back', () => {
    const { holds, local } = page();
    const settle = holds.starting('app:card_1', 'sha-a');
    assert.deepEqual(entries(local), { 'app:card_1': { reason: 'start', rev: 'sha-a', at: 5, n: 1 } });
    assert.equal(holds.reason('app:card_1', 'sha-a'), '', 'read once: a start in flight is not this page\'s hold');
    settle();
    settle();
    assert.deepEqual(entries(local), {}, 'settle removes it, once');
    const twin = holds.starting('app:card_2', 'v');
    const other = holds.starting('app:card_2', 'v');
    twin();
    assert.equal(entries(local)['app:card_2']?.n, 3, 'a settle clears only its own start, even within one millisecond');
    other();
});

test('a page that hides outlived what it started: its starts settle, so a reload holds nothing', () => {
    const first = page();
    first.holds.starting('widget:kotokulator:calculator', '7');
    first.win.emit('pagehide');
    assert.equal(page({ local: first.local }).holds.reason('widget:kotokulator:calculator', '7'), '');
});

test('the next page holds what the previous one did not outlive, for that revision only, until released', () => {
    const first = page();
    first.holds.starting('app:card_1', 'sha-a');
    first.holds.starting('widget:kotokulator:calculator', '7');
    const next = page({ local: first.local });
    assert.equal(next.holds.reason('app:card_1', 'sha-a'), 'start');
    assert.equal(next.holds.reason('app:card_1', 'sha-b'), '', 'new bytes are not the bytes that stopped the page');
    assert.equal(next.holds.reason('widget:kotokulator:calculator', '7'), 'start');
    assert.deepEqual(next.holds.found(), [{ key: 'app:card_1', rev: 'sha-a' }, { key: 'widget:kotokulator:calculator', rev: '7' }]);
    next.holds.release('app:card_1');
    assert.equal(next.holds.reason('app:card_1', 'sha-a'), '');
    assert.deepEqual(next.holds.found(), [{ key: 'widget:kotokulator:calculator', rev: '7' }]);
    assert.equal(entries(next.local)['app:card_1'], undefined, 'a released hold is gone for the page after it too');
    assert.equal(page({ local: next.local }).holds.reason('widget:kotokulator:calculator', '7'), 'start',
        'an unreleased hold stays however many pages open');
});

test('the owner\'s Stop and Close stay until released, on this page and the next', () => {
    const { holds, local } = page();
    holds.stop('board:kotokulator:calculator');
    assert.equal(holds.reason('board:kotokulator:calculator'), 'owner', 'at once on this page');
    const next = page({ local });
    assert.equal(next.holds.reason('board:kotokulator:calculator'), 'owner');
    assert.deepEqual(next.holds.found(), [], 'an owner hold is not a start the page did not outlive');
    next.holds.release('board:kotokulator:calculator');
    assert.equal(next.holds.reason('board:kotokulator:calculator'), '');
    assert.equal(page({ local: next.local }).holds.reason('board:kotokulator:calculator'), '');
});

test('?hold=all holds every key for that load until each is released, and leaves the address', () => {
    const { holds, win } = page({ href: 'http://127.0.0.1:8765/?hold=all#chat' });
    assert.equal(win.location.href, 'http://127.0.0.1:8765/#chat', 'a reload of this address starts as usual');
    assert.equal(holds.reason('module:device_voice:tts', 'r1'), 'all');
    assert.equal(holds.reason('app:card_9', 'x'), 'all');
    holds.release('app:card_9');
    assert.equal(holds.reason('app:card_9', 'x'), '');
    assert.equal(page().holds.reason('app:card_9', 'x'), '', 'without the flag nothing is held');
    assert.deepEqual(Object.keys(HELD_LINES).sort(), ['all', 'start']);
});

test('the ledger keeps its newest 64 entries of each reason, clears the retired boot marker and falls back to the session', () => {
    const local = storage();
    local.setItem('ouroboros.hostModules.boot', JSON.stringify({ at: 1, keys: ['x'] }));
    let at = 0;
    const holds = createHolds({ win: { location: { href: 'http://x/' }, localStorage: local }, now: () => { at += 1; return at; } });
    assert.equal(local.map.has('ouroboros.hostModules.boot'), false);
    for (let index = 0; index < 70; index += 1) holds.starting(`app:card_${index}`, 'v');
    const kept = Object.keys(entries(local));
    assert.equal(kept.length, 64);
    assert.ok(kept.includes('app:card_69') && !kept.includes('app:card_0'), 'the oldest go first');
    const owned = page();
    for (let index = 0; index < 64; index += 1) owned.holds.stop(`card:card_${index}`);
    owned.holds.starting('app:unrelated', 'r1')();
    assert.equal(page({ local: owned.local }).holds.reason('card:card_0'), 'owner', 'a passing start never pushes out a Close');
    const refused = page({ local: storage({ refuse: true }) });
    refused.holds.starting('app:card_1', 'v');
    assert.deepEqual(Object.keys(entries(refused.session)), ['app:card_1']);
    const nowhere = createHolds({ win: { location: { href: 'http://x/' } } });
    assert.doesNotThrow(() => nowhere.starting('app:card_1', 'v')());
    assert.equal(nowhere.reason('app:card_1', 'v'), '', 'no storage: nothing can be remembered, nothing is held');
});

test('a frame settles one host second after its first load; no frame settles at once', async () => {
    let settled = 0;
    const listeners = {};
    const frame = { addEventListener: (type, fn) => { listeners[type] = fn; }, removeEventListener: (type) => { delete listeners[type]; } };
    settleAfterFrameLoad({ querySelector: () => frame }, () => { settled += 1; }, { delay: 5 });
    await tick(10);
    assert.equal(settled, 0, 'nothing before the frame loaded');
    listeners.load();
    await tick(10);
    assert.equal(settled, 1);
    const cancel = settleAfterFrameLoad({ querySelector: () => frame }, () => { settled += 1; }, { delay: 5 });
    cancel();
    assert.equal(listeners.load, undefined, 'a cancelled wait leaves no listener');
    settleAfterFrameLoad({ querySelector: () => null }, () => { settled += 1; });
    assert.equal(settled, 2, 'a mount without a frame ran no foreign code');
});

test('the found holds are reported once, with the raw facts of where the page runs', async () => {
    const first = page();
    first.holds.starting('app:card_1', 'sha-a');
    const next = page({ local: first.local });
    const posts = [];
    reportFoundHolds(next.holds, (payload) => posts.push(payload), { win: { pywebview: { api: {} } } });
    await tick();
    assert.deepEqual(posts, [{ holds: [{ key: 'app:card_1', rev: 'sha-a' }], surface: { pywebview: true, telegram: false } }]);
    const lateFirst = page();
    lateFirst.holds.starting('app:card_3', 'v');
    const late = [];
    const listeners = {};
    reportFoundHolds(page({ local: lateFirst.local }).holds, (payload) => late.push(payload),
        { win: { addEventListener: (type, fn) => { listeners[type] = fn; } }, wait: 5 });
    assert.equal(late.length, 0, 'the desktop bridge arrives after load: wait for it');
    listeners.pywebviewready();
    await tick(10);
    assert.equal(late.length, 1, 'sent once, by whichever comes first');
    const quiet = [];
    reportFoundHolds(page().holds, (payload) => quiet.push(payload), { win: {}, wait: 1 });
    await tick(5);
    assert.deepEqual(quiet, [], 'nothing held, nothing reported');
    const after = page({ local: first.local });
    assert.equal(after.holds.reason('app:card_1', 'sha-a'), 'start', 'a reported hold still holds');
    assert.deepEqual(after.holds.found(), [], 'and is not reported again by the next page');
    const failing = page();
    failing.holds.starting('app:card_2', 'v');
    const retry = page({ local: failing.local });
    reportFoundHolds(retry.holds, async () => { throw new Error('offline'); }, { win: { pywebview: { api: {} } } });
    await tick();
    assert.deepEqual(page({ local: failing.local }).holds.found(), [{ key: 'app:card_2', rev: 'v' }],
        'a report that did not land is made again by the next page');
});
