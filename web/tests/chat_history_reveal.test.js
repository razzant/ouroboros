import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatHistoryPager } from '../modules/chat_history.js';
import { createHistoryReveal } from '../modules/chat_history_reveal.js';
import { installDomDouble } from './dom_double.js';

const recent = { messages: [{ history_id: 'chat:900', text: 'now' }], page_cursor: 'replay:recent', next_cursor: 'before:899',
    has_more: true, after_cursor: 'after:1', window: { complete: false } };
const around = (anchor = { state: 'found', task_id: 't-old', history_id: 'chat:100' }) => ({
    messages: [{ history_id: 'chat:99', text: 'before' }, { history_id: 'chat:100', text: 'the task', task_id: 't-old' }],
    page_cursor: 'replay:around', next_cursor: 'before:99', has_more: true, window: { complete: false }, anchor });

function harness(answer) {
    const calls = [], applied = [];
    let arrive = true;
    const pager = createChatHistoryPager({
        fetchPage: async (cursor, options) => { calls.push({ cursor, around: options.around }); return answer(cursor, options); },
        applyPage: (messages, descriptor) => applied.push({ messages, ...descriptor }),
        releasePage: () => {}, canArrive: () => arrive, applyArrivals: () => {},
    });
    pager.acceptRecent(recent);
    return { pager, calls, applied, stopArrivals: () => { arrive = false; } };
}

test('a read around one task lands as its own chain and leaves the present, its arrivals and Retry alone', async () => {
    const h = harness((cursor) => (cursor === 'after:1' ? { messages: [], after_cursor: 'after:2', window: {} } : around()));
    assert.equal(h.pager.pageOf('chat:900'), 'replay:recent', 'the recent page names its own replay cursor');
    const result = await h.pager.around('task:t-old', () => true);
    assert.equal(result.status, 'applied');
    assert.deepEqual(result.anchor, { state: 'found', task_id: 't-old', history_id: 'chat:100' });
    assert.deepEqual(h.calls.at(-1), { cursor: null, around: 'task:t-old' });
    assert.equal(h.applied.at(-1).direction, 'around', 'never `latest`: the jumped page is not the present');
    const state = h.pager.getState();
    assert.equal(state.firstPage.requestCursor, 'replay:around');
    assert.equal(state.canOlder, true);
    assert.equal(h.pager.pageOf('chat:900'), null, 'the left chain is no longer this chain');
    assert.equal(h.pager.pageOf('chat:100'), 'replay:around');
    // The window still reads only what arrived since its newest read: the jump did not forget where that starts.
    await h.pager.read();
    assert.equal(h.calls.at(-1).cursor, 'after:1');
});

test('a task with no place to land, a superseded read and a failed read change nothing', async () => {
    let answer = around({ state: 'absent', task_id: 't-none' });
    const h = harness(() => (typeof answer === 'function' ? answer() : answer));
    const before = h.pager.getState().firstPage;
    assert.deepEqual(await h.pager.around('task:t-none', () => true), { status: 'anchor', anchor: { state: 'absent', task_id: 't-none' } });
    answer = around();
    assert.deepEqual(await h.pager.around('task:t-old', () => false), { status: 'superseded' });
    answer = () => { throw Object.assign(new Error('503'), { status: 503 }); };
    assert.equal((await h.pager.around('task:t-old', () => true)).status, 'error');
    const after = h.pager.getState();
    assert.equal(after.firstPage, before, 'the reader keeps the chain they were reading');
    assert.equal(after.error, null, 'a navigation that failed is its own failure, never the window’s Retry');
    assert.equal(h.applied.length, 1, 'only the first recent read was ever applied');
    // The working path still lands.
    answer = around();
    assert.equal((await h.pager.around('task:t-old', () => true)).status, 'applied');
});

test('a newer navigation supersedes one still loading; a left page is read again by its own cursor', async () => {
    const waits = [];
    const h = harness((cursor, options) => new Promise((resolve) => waits.push(() => resolve(cursor === 'replay:recent'
        ? { ...recent, after_cursor: undefined } : around({ state: 'found', task_id: options.around.slice(5), history_id: 'chat:100' })))));
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    const first = h.pager.around('task:t-a', () => true);
    await tick();  // the first read is in flight
    const second = h.pager.around('task:t-b', () => true);
    await tick();
    assert.deepEqual(h.calls.map((call) => call.around), ['task:t-a', 'task:t-b']);
    waits[0]();
    waits[1]();
    assert.equal((await first).status, 'disposed', 'an answer arriving for a superseded read is dropped');
    assert.equal((await second).anchor.task_id, 't-b');
    assert.equal(h.applied.length, 2, 'only the newer navigation applied');
    const back = h.pager.revisit('replay:recent', () => true);
    await tick();
    waits[2]();
    assert.equal((await back).status, 'applied');
    assert.equal(h.applied.at(-1).direction, 'revisit');
    assert.equal(h.pager.pageOf('chat:900'), 'replay:recent');
});

test('a navigation leaves the window’s Retry unless its page replaces the chain, and waits out a window read', async () => {
    let release = null;
    const h = harness((cursor, options) => {
        if (options.around) return around(options.around === 'task:t-none' ? { state: 'absent', task_id: 't-none' } : undefined);
        if (cursor === 'before:899') return Promise.reject(Object.assign(new Error('offline'), { status: 0 }));
        if (cursor === 'before:99') return new Promise((resolve) => { release = () => resolve({ messages: [{ history_id: 'chat:98' }],
            page_cursor: 'replay:98', next_cursor: 'before:97', has_more: true, window: {} }); });
        return { messages: [], after_cursor: 'after:2', window: {} };
    });
    assert.equal((await h.pager.older()).status, 'error');
    assert.ok(h.pager.getState().error, 'the older read failed: the window offers its Retry');
    assert.equal((await h.pager.around('task:t-none', () => true)).status, 'anchor');
    assert.ok(h.pager.getState().error, 'a navigation with nowhere to land leaves the window’s Retry');
    assert.equal((await h.pager.around('task:t-old', () => true)).status, 'applied');
    assert.equal(h.pager.getState().error, null, 'its page replaced the chain the failure belonged to');
    const older = h.pager.older();
    const jump = h.pager.around('task:t-again', () => true);  // a window read is in flight: the navigation waits, never “busy”
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(h.calls.at(-1).cursor, 'before:99', 'it has not asked yet');
    release();
    assert.equal((await older).status, 'applied');
    assert.equal((await jump).status, 'applied');
    assert.equal(h.calls.at(-1).around, 'task:t-again');
});

test('only a read of the present restarts the arrivals', async () => {
    const h = harness((cursor, options) => (options.around ? around()
        : cursor ? { messages: [], after_cursor: `${cursor}+`, window: {} } : { ...recent, after_cursor: 'after:9' }));
    await h.pager.around('task:t-old', () => true);
    assert.equal(h.pager.getState().away, true, 'the chain began at a navigation: it is not the present');
    assert.equal((await h.pager.latest()).status, 'applied');
    assert.equal(h.pager.getState().away, false);
    await h.pager.read();
    assert.equal(h.calls.at(-1).cursor, 'after:9', 'the newest read names where the next arrivals start');
});

test('a landing is the task’s card, else its own row, and nothing when the page holds none of it', async () => {
    const node = (data) => ({ isConnected: true, dataset: data });
    const card = node({ taskId: 't-card' }), reply = node({ taskId: 't-reply' }), stamped = node({ historyId: 'chat:7' });
    const feed = { querySelectorAll: (selector) => (selector === '[data-task-id]' ? [reply, card] : [reply, card, stamped]) };
    const records = new Map([['t-card', { root: card }], ['t-gone', { root: { isConnected: false } }]]);
    let result = { status: 'applied', anchor: { state: 'found', history_id: 'chat:7' } };
    let atNewest = true;
    const reading = { stick: true, nearBottom: () => atNewest, claim: () => () => true };
    const pages = { 'chat:1': 'replay:here', 'progress:100': 'replay:card' };
    const pager = { around: async () => result, revisit: async () => result, pageOf: (id) => pages[id] ?? null };
    const reveal = createHistoryReveal({ pager, reading, feed, records, present: () => true });
    assert.deepEqual(reveal.departure({ historyId: 'chat:1' }), { anchor: { historyId: 'chat:1' }, cursor: 'replay:here', present: true });
    // Held to the newest message but moved off it without a gesture (focus, find in page): a passage, not the present.
    atNewest = false;
    assert.equal(reveal.departure({ historyId: 'chat:1' }).present, false);
    reading.stick = false;
    atNewest = true;
    assert.equal(reveal.departure({ historyId: 'chat:1' }).present, false, 'a reader who let go of the newest is reading');
    // A collapsed card's header names no row: its page is the one holding the card's earliest row.
    const header = { historyId: '', cardChain: [{ node: { dataset: { taskId: 't-c', historySource: 'progress', historyOffset: '100' } } }] };
    assert.equal(reveal.departure(header).cursor, 'replay:card');
    assert.equal(reveal.departure({ historyId: '' }).cursor, null);
    assert.equal((await reveal.task('t-card', () => true)).target, card);
    assert.equal((await reveal.task('t-reply', () => true)).target, reply);
    assert.equal((await reveal.task('t-gone', () => true)).target, stamped, 'a released card is not a landing');
    result = { status: 'applied', anchor: { state: 'found', history_id: 'chat:404' } };
    assert.deepEqual(await reveal.task('t-none', () => true), { state: 'unknown', left: true }, 'drawn, yet nothing of it shows');
    result = { status: 'error' };
    assert.deepEqual(await reveal.task('t-none', () => true), { state: 'unknown' });
    result = { status: 'anchor', anchor: { state: 'moved' } };
    assert.deepEqual(await reveal.task('t-card', () => true), { state: 'moved' });
    assert.equal(await reveal.task('t-card', () => false), null, 'a navigation the reader left is dropped');
    result = { status: 'applied' };
    assert.equal(await reveal.revisit('replay:here', () => true), true);
    result = { status: 'error' };
    assert.equal(await reveal.revisit('replay:here', () => true), false);
    result = { status: 'error', error: { status: 503 } };
    assert.equal(await reveal.revisit('replay:here', () => true), false, 'a read that failed for now: try again');
    result = { status: 'error', error: { status: 409 } };
    assert.equal(await reveal.revisit('replay:here', () => true), 'gone', 'the room changed: that page is no longer there');
});

test('an exact reference cannot fall back to a task, portal, retained origin or folded repeat', async () => {
    const env = installDomDouble();
    try {
        const address = 'row:1@then#aaaaaaaaaaaa';
        const feed = env.el('div');
        env.doc.body.append(feed);
        const card = env.el('div', { className: 'chat-live-card', dataset: { taskId: 't-card' } });
        const retained = env.el('div', { dataset: { rowAddress: address, originProjected: '1' } });
        const portal = env.el('div', { dataset: { portal: '1' } }, env.el('div', { dataset: { historyId: 'chat:100' } }));
        feed.append(card, retained, portal);
        const records = new Map([['t-card', { root: card, items: [{ historyId: 'chat:100' }] }]]);
        let mounted = false;
        const reveal = createHistoryReveal({ feed, records, present: () => true,
            reading: { claim: () => () => true },
            pager: { around: async () => ({ status: 'applied', anchor: { state: 'found', row_address: address, history_id: 'chat:100' } }) },
            expand(record) {
                if (mounted && record) card.append(env.el('div', { dataset: { historyId: 'chat:100' }, text: 'Exact line' }));
            } });
        assert.equal(reveal.exact(address), null);
        assert.deepEqual(await reveal.reference(address, () => true), { state: 'unavailable', left: true },
            'its page was drawn: the reader is no longer where they were');
        mounted = true;
        const answer = await reveal.reference(address, () => true);
        assert.equal(answer.target.dataset.historyId, 'chat:100');
        assert.equal(answer.target.textContent, 'Exact line');
        assert.equal(await reveal.reference(address, () => false), null);
    } finally { env.restore(); }
});

test('a destination page prepared before switching joins the same pager without a second read', async () => {
    const h = harness(() => { throw new Error('a prepared page must not refetch'); });
    const data = around({ state: 'found', row_address: 'row:1@then#aaaaaaaaaaaa', chat_id: 1, history_id: 'chat:100' });
    assert.equal((await h.pager.around(data.anchor.row_address, () => true, data)).status, 'applied');
    assert.equal(h.calls.length, 0);
    assert.equal(h.pager.getState().away, true);
    assert.equal(h.pager.pageOf('chat:100'), 'replay:around');
});

test('an exact reference that could not be read passes the typed failure on, never a guess at its cause', async () => {
    const env = installDomDouble();
    try {
        const feed = env.el('div');
        env.doc.body.append(feed);
        const failure = Object.assign(new Error('HTTP 503'), { status: 503, body: { reason_code: 'history_source_unavailable' } });
        let read = { status: 'error', error: failure };
        const reveal = createHistoryReveal({ feed, records: new Map(), present: () => true, reading: { claim: () => () => true },
            pager: { around: async () => read } });
        assert.deepEqual(await reveal.reference('row:1@then#aaaaaaaaaaaa', () => true), { state: 'unknown', error: failure });
        read = { status: 'busy' };
        assert.deepEqual(await reveal.reference('row:1@then#aaaaaaaaaaaa', () => true), { state: 'unknown' });
        read = { status: 'anchor', anchor: { state: 'moved', task_id: 'work-1' } };
        assert.deepEqual(await reveal.reference('task:work-1', () => true), { state: 'moved', task_id: 'work-1' });
    } finally { env.restore(); }
});


test('an exact tucked relay remains a target for its deck owner; other hidden rows remain unavailable', () => {
    const env = installDomDouble();
    try {
        const feed = env.el('div');
        env.doc.body.append(feed);
        const relay = env.el('div', { dataset: { rowAddress: 'row:42@a#1', deckTucked: '1' } });
        relay.getClientRects = () => [];
        feed.append(relay);
        const reveal = createHistoryReveal({ pager: {}, reading: {}, feed, records: new Map(), present: () => true });
        assert.equal(reveal.exact('row:42@a#1'), relay, 'the caller can surface this deck before jumping');
        delete relay.dataset.deckTucked;
        assert.equal(reveal.exact('row:42@a#1'), null);
        relay.dataset.deckTucked = '1';
        relay.hidden = true;
        assert.equal(reveal.exact('row:42@a#1'), null, 'an explicitly hidden row stays unavailable');
    } finally { env.restore(); }
});
