// An open room reads what arrived since its newest read (CLIENT_CONTRACT §5
// `after_cursor`), never its whole window again: one newest page when it opens,
// then only the rows stored since, for a new message, a card finish and a
// reconnect; the newest page again only on a refusal, after a failure, or once
// the window took in as many rows as it holds. Review details of every card share
// one queue of two reads (issue #1771). Real createChatInstance, mocked HTTP.
import assert from 'node:assert/strict';
import test from 'node:test';
import { REVIEW_DETAIL_READS, REVIEW_DETAIL_SLOT_MS, createReviewHydrator } from '../modules/review_presentation.js';
import { createChatHistoryPager } from '../modules/chat_history.js';
import { ElementStub } from './chat_dom_fixture.js';
import { quiet, room, until } from './project_read_state_fixture.js';

const ts = (n) => new Date(Date.UTC(2026, 9, 16, 12, 0, 0, n * 1000)).toISOString();

// The room's rows in arrival order, read the way history_paging serves them: the
// newest page holds the last `window` rows and names the newest message; an
// `after:<k>` cursor returns the rows stored since that read; `older:<k>` the
// page below. Every read answers with what was stored when it began.
function historyServer({ window = 1000 } = {}) {
    const rows = [], reads = [], cursors = [], refusals = [], gates = new Map();
    let latest;  // the newest message the newest page names: computed, null (unknown) or set
    const say = (n, extra = {}) => rows.push({ role: 'assistant', text: `answer ${n}`, ts: ts(n),
        history_id: `chat:${n}`, history_position: { source: 'chat', offset: n }, ...extra });
    const work = (n, taskId = 'task-1', extra = {}) => rows.push({ role: 'assistant', text: `step ${n}`, ts: ts(n),
        is_progress: true, task_id: taskId, history_id: `progress:${n}`, history_position: { source: 'progress', offset: n }, ...extra });
    const named = (stored) => {
        if (latest !== undefined) return latest;
        const row = [...rows.slice(0, stored)].reverse().find((item) => !item.is_progress && item.role !== 'user' && !item.system_type);
        return row ? { history_id: row.history_id, out_of_order: false } : undefined;
    };
    // An `after` read names the newest message among the rows it returns, as latest_arrival does:
    // never narration, the owner, a typed system row or a child's words (by its lineage anywhere).
    const namedSince = (from, stored) => {
        const children = new Set(rows.filter((item) => item.subagent_task_id).map((item) => item.subagent_task_id));
        const row = rows.slice(from, stored).reverse().find((item) => !item.is_progress && item.role !== 'user'
            && !item.system_type && item.delegation_role !== 'subagent' && !children.has(item.task_id));
        return row && { history_id: row.history_id, out_of_order: false };
    };
    const answer = (data) => ({ ok: true, status: 200, json: async () => data });
    async function fetchHistory(url) {
        const cursor = new URL(String(url), 'http://local').searchParams.get('cursor');
        const kind = !cursor ? 'newest' : cursor.split(':')[0];
        reads.push(kind); cursors.push(cursor);
        const stored = rows.length;
        const gate = gates.get(kind) || gates.get('');
        if (gate) await gate;
        const refusal = kind === 'after' && refusals.shift();
        if (refusal) return { ok: false, status: 409, json: async () => ({ messages: [], error: refusal, reason_code: refusal,
            has_more: true, next_cursor: cursor, window: { complete: false, truncated_by: [refusal] } }) };
        if (kind === 'after') {
            const from = Number(cursor.split(':')[1]), newest = namedSince(from, stored);
            return answer({ messages: rows.slice(from, stored), has_more: false, next_cursor: null,
                page_cursor: null, after_cursor: `after:${stored}`, window: { complete: false, truncated_by: ['page'],
                    ...(newest ? { latest_message: newest } : {}) } });
        }
        const end = kind === 'older' ? Number(cursor.split(':')[1]) : stored, from = Math.max(0, end - window);
        const newest = kind === 'newest' ? named(stored) : undefined;
        return answer({ messages: rows.slice(from, end), has_more: from > 0, next_cursor: from > 0 ? `older:${from}` : null,
            page_cursor: `page:${end}`, ...(kind === 'newest' ? { after_cursor: `after:${stored}` } : {}),
            window: { complete: from === 0, truncated_by: from > 0 ? ['quota'] : [],
                ...(newest !== undefined ? { latest_message: newest } : {}) } });
    }
    return {
        rows, reads, cursors, refusals, say, work, fetchHistory,
        // Reads begun from now on (of `kind` only, when given) wait until the returned release.
        hold(kind = '') {
            let release;
            gates.set(kind, new Promise((resolve) => { release = resolve; }));
            return () => { gates.delete(kind); release(); };
        },
        nameNewest(message) { latest = message; },
    };
}

// The pager's own contract, with chat's reads mocked.
function arrivalsPager({ fetch, canArrive = () => true, inFlight = () => null } = {}) {
    const calls = [], arrivals = [], newest = [], applied = [];
    const pager = createChatHistoryPager({
        fetchPage: (cursor) => { calls.push(cursor); return fetch(cursor); },
        applyPage: (messages, descriptor) => applied.push(descriptor), releasePage: () => {},
        applyArrivals: (messages, descriptor) => arrivals.push({ ids: messages.map((row) => row.history_id), ...descriptor }),
        readNewest: (options) => { newest.push(options); return Promise.resolve(true); },
        canArrive, newestInFlight: inFlight,
    });
    return { pager, calls, arrivals, newest, applied };
}
const newestPage = (after = 'after:1', ids = ['chat:1']) => ({ messages: ids.map((id) => ({ history_id: id })), has_more: false,
    next_cursor: null, page_cursor: `page:${after}`, after_cursor: after, window: {} });
const arrivedRows = (ids, after) => ({ messages: ids.map((id) => ({ history_id: id })), has_more: false, next_cursor: null,
    page_cursor: null, after_cursor: after, window: { complete: false, truncated_by: ['page'] } });

test('the pager reads what arrived from the newest read\'s cursor, and leaves the newest page to chat otherwise', async () => {
    const answers = { 'after:1': arrivedRows(['chat:2'], 'after:2'), 'after:2': arrivedRows([], 'after:2') };
    const p = arrivalsPager({ fetch: async (cursor) => answers[cursor] });
    const first = p.pager.read({ fromReconnect: true });
    assert.deepEqual(p.newest, [{ fromReconnect: true }], 'no cursor yet: chat\'s newest read, begun within the call');
    await first;
    p.pager.acceptRecent(newestPage());
    assert.equal(await p.pager.read(), true);
    assert.deepEqual([p.calls, p.arrivals.map((arrival) => arrival.ids)], [['after:1'], [['chat:2']]]);
    await p.pager.read();
    assert.deepEqual(p.calls, ['after:1', 'after:2'], 'each read starts where the last one ended');
    await p.pager.read({ whole: true });
    assert.equal(p.newest.length, 2, '`whole` takes the newest page');
    const declined = arrivalsPager({ fetch: async () => assert.fail('no read of arrivals'), canArrive: () => false });
    declined.pager.acceptRecent(newestPage());
    await declined.pager.read();
    assert.equal(declined.newest.length, 1, 'a window chat declines (not hydrated, failed, a due rebuild) reads the newest page');
});

test('a refused or failed read of arrivals forgets its cursor: the newest page names the next one', async () => {
    const refusal = Object.assign(new Error('history_delta_reset'), { status: 409 });
    const p = arrivalsPager({ fetch: async () => { throw refusal; } });
    p.pager.acceptRecent(newestPage());
    assert.equal(await p.pager.read(), true);
    assert.deepEqual([p.calls, p.newest.length], [['after:1'], 1], 'refused, then the newest page');
    await p.pager.read();
    assert.deepEqual([p.calls, p.newest.length], [['after:1'], 2], 'no stale cursor is tried again');
    p.pager.acceptRecent(newestPage('after:5'));
    await p.pager.read().catch(() => {});
    assert.equal(p.calls.at(-1), 'after:5', 'the newest page restarts arrivals from its own cursor');
});

test('reads during a read of arrivals share one queued read, which reconciles as a reconnect if one asked', async () => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const answers = { 'after:1': arrivedRows(['chat:2'], 'after:2'), 'after:2': arrivedRows(['chat:3'], 'after:3') };
    const p = arrivalsPager({ fetch: async (cursor) => { if (cursor === 'after:1') await gate; return answers[cursor]; } });
    p.pager.acceptRecent(newestPage());
    const reads = [p.pager.read(), p.pager.read(), p.pager.read({ fromReconnect: true }), p.pager.read()];
    await quiet();
    assert.deepEqual(p.calls, ['after:1']);
    release();
    assert.deepEqual(await Promise.all(reads), [true, true, true, true]);
    assert.deepEqual(p.calls, ['after:1', 'after:2'], 'one read in flight, one queued behind it');
    assert.deepEqual(p.arrivals.map((arrival) => arrival.fromReconnect), [false, true]);
});

test('a read waits for chat\'s newest read in flight, then reads what arrived after it', async () => {
    let finish, current = new Promise((resolve) => { finish = resolve; });
    const p = arrivalsPager({ fetch: async () => arrivedRows([], 'after:2'), inFlight: () => current });
    p.pager.acceptRecent(newestPage());
    const reading = p.pager.read();
    await quiet();
    assert.deepEqual(p.calls, []);
    current = null;
    finish();
    await reading;
    assert.deepEqual(p.calls, ['after:1']);
});

test('an arrivals answer older than a newest read that landed meanwhile is dropped and read again from its cursor', async () => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const answers = { 'after:1': arrivedRows(['chat:2'], 'after:2'), 'after:5': arrivedRows(['chat:6'], 'after:6') };
    const p = arrivalsPager({ fetch: async (cursor) => { if (cursor === 'after:1') await gate; return answers[cursor]; } });
    p.pager.acceptRecent(newestPage());
    const reading = p.pager.read();
    await quiet();
    p.pager.acceptRecent(newestPage('after:5', ['chat:5']));  // the newest page landed first: it owns the present
    release();
    assert.equal(await reading, true);
    assert.deepEqual(p.calls, ['after:1', 'after:5'], 'read again from the newer cursor');
    assert.deepEqual(p.arrivals.map((arrival) => arrival.ids), [['chat:6']], 'the older answer never rolls the window back');
    // The other order: arrivals land first, the newer page after them; reads go on from the page.
    p.pager.acceptRecent(newestPage('after:9', ['chat:9']));
    answers['after:9'] = arrivedRows([], 'after:9');
    await p.pager.read();
    assert.equal(p.calls.at(-1), 'after:9');
});

test('arrivals a newest page let go are read again, and a refusal of that read takes the newest page', async () => {
    for (const refuse of [false, true]) {
        let release, afterReads = 0;
        const gate = new Promise((resolve) => { release = resolve; });
        const p = arrivalsPager({ fetch: async (cursor) => {
            if (cursor === null) { await gate; return newestPage('after:1', ['chat:1']); }
            afterReads += 1;
            if (refuse && afterReads === 2) throw Object.assign(new Error('history_delta_reset'), { status: 409 });
            return arrivedRows(['chat:2'], 'after:2');
        } });
        p.pager.acceptRecent(newestPage());
        const latest = p.pager.latest();  // begun before the arrivals below, it lands after them
        await quiet();
        await p.pager.read();
        release();
        await latest;
        await quiet();
        assert.deepEqual([p.calls.filter((cursor) => cursor === 'after:1').length, p.newest.length], [2, refuse ? 1 : 0],
            refuse ? 'the refused re-read falls back to the newest page' : 'read again from the newest page\'s cursor');
        p.pager.destroy();
    }
});

test('a read of arrivals with a narration gap takes the newest page; an unfinished narration line does not', async () => {
    const gapped = (cause, after) => ({ ...arrivedRows([], after), window: { complete: false, truncated_by: ['page', cause] } });
    const answers = { 'after:1': gapped('progress_incomplete_live_line', 'after:2'), 'after:2': gapped('progress_malformed_jsonl', 'after:3') };
    const p = arrivalsPager({ fetch: async (cursor) => answers[cursor] });
    p.pager.acceptRecent(newestPage());
    await p.pager.read();
    assert.deepEqual([p.calls, p.newest.length], [['after:1'], 0], 'its writer is still on the line: the next read takes it');
    await p.pager.read();
    assert.deepEqual([p.calls, p.newest.length], [['after:1', 'after:2'], 1], 'rows it could not read: the newest page answers');
});

test('a window that took in a whole default page of arrivals reads the newest page next', async () => {
    const many = Array.from({ length: 211 }, (_, index) => `chat:${index + 2}`);
    const p = arrivalsPager({ fetch: async () => arrivedRows(many, 'after:212') });
    p.pager.acceptRecent(newestPage());
    await p.pager.read();
    await p.pager.read();
    assert.deepEqual([p.calls.length, p.newest.length], [1, 1], 'past 210 rows the window re-anchors on the newest page');
});

test('a shifted window starts the new chain with the read in hand, never by reading it again', () => {
    const p = arrivalsPager({ fetch: async () => assert.fail('no second read') });
    p.pager.acceptRecent(newestPage('after:1', ['chat:1']));
    assert.equal(p.pager.acceptRecent(newestPage('after:2', ['chat:2'])).status, 'ignored', 'an ordinary later read keeps the chain');
    const result = p.pager.acceptRecent(newestPage('after:3', ['chat:3']), { fromReconnect: true });
    assert.equal(result.status, 'applied');
    assert.deepEqual([p.applied.at(-1).direction, p.applied.at(-1).rechained, p.applied.at(-1).fromReconnect], ['latest', true, true]);
    assert.equal(p.pager.getState().firstPage.requestCursor, 'page:after:3');
    assert.deepEqual(p.calls, []);
});

const openRoom = (t, server) => room(t, { projectId: 'room', fetchHistory: server.fetchHistory });
const bubbles = (r, text) => r.messages.children.filter((node) => node.classList.contains('chat-bubble')
    && new RegExp(`${text}(?!\\d)`).test(String(node.innerHTML)));

// History rows matching `hidden` are drawn far below the viewport (unprotected, off screen).
function drawBelow(t, hidden) {
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        return hidden(this.dataset?.historyId || '') ? { top: 1000, bottom: 1020, left: 0, right: 100, width: 100, height: 20 }
            : rect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
}

test('an open room takes ten new messages with one newest-page read, then only what arrived since', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true);
    for (let n = 2; n <= 11; n += 1) {
        s.work(100 + n);
        s.say(n);
        // Live delivery shows the message first; the stored row takes over that node.
        r.emit('chat', { chat_id: 2, role: 'assistant', content: `answer ${n}`, ts: ts(n) });
        const paint = await r.instance.refreshHistory({ revision: n });
        assert.equal(paint.read, true, `revision ${n}: the message it brought is painted and on screen`);
    }
    assert.deepEqual(s.reads, ['newest', ...Array(10).fill('after')], 'no full re-read and no second read of the same page');
    assert.deepEqual(s.cursors.slice(1, 4), ['after:1', 'after:3', 'after:5'], 'each read starts where the previous one ended');
    for (let n = 1; n <= 11; n += 1) assert.equal(bubbles(r, `answer ${n}`).length, 1, `answer ${n} is one node`);
    assert.ok(r.shown().includes('chat:11'));
});

test('a new message read below the fold is not read until the reader reaches it', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    let below = true;
    drawBelow(t, (id) => below && id === 'chat:2');
    s.say(2);
    const withheld = await r.instance.refreshHistory({ revision: 2 });
    assert.deepEqual([withheld.painted, withheld.read], [true, false],
        'the earlier message on screen does not stand in for the one that arrived last');
    below = false;
    const reads = s.reads.length;
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true, 'shown, the same revision is read');
    assert.equal(s.reads.length, reads, 'with no further history read');
});

test('a newest message outside the loaded window stays unread until a newer one arrives', async (t) => {
    const s = historyServer();
    s.say(5);
    s.nameNewest({ history_id: 'chat:0', out_of_order: true });  // arrived last, older than the page
    const r = openRoom(t, s);
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, false, 'not drawn: not read');
    s.nameNewest(undefined);
    s.work(6);
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, false, 'narration names no newer message');
    s.say(7, { role: 'system', system_type: 'skill_review' });
    s.say(8, { delegation_role: 'subagent', task_id: 'child-1', parent_task_id: 'task-1' });
    // A child known from its lineage elsewhere: its row without lineage is its card's too.
    s.work(9, 'child-2', { delegation_role: 'subagent', subagent_task_id: 'child-2', parent_task_id: 'task-1' });
    s.say(10, { task_id: 'child-2' });
    assert.equal((await r.instance.refreshHistory({ revision: 3 })).read, false, 'a review row and a child\'s words are card content');
    s.say(11);
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true, 'a newer standalone message is the newest');
    assert.deepEqual(s.reads, ['newest', 'after', 'after', 'after']);
});

test('a room whose newest message is unknown reads its newest page again, not only what arrived', async (t) => {
    const s = historyServer();
    s.say(1);
    s.nameNewest(null);  // e.g. a damaged row after the newest readable one
    const r = openRoom(t, s);
    assert.equal(Boolean((await r.instance.refreshHistory({ revision: 1 })).read), false, 'unknown: nothing is read');
    s.nameNewest(undefined);  // repaired, without a new revision
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true, 'the newest page read again names it');
    assert.deepEqual(s.reads, ['newest', 'newest']);
    s.say(2);
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
    assert.deepEqual(s.reads, ['newest', 'newest', 'after'], 'once named, arrivals answer again');
});

test('a card finish takes one read of what arrived', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const frames = [{ content: 'step 2', ts: ts(2) },
        { content: 'step 3', ts: ts(3), task_terminal_status: 'completed', outcome_axes: { execution: 'ok' } }];
    for (const [index, frame] of frames.entries()) {
        s.work(2 + index, 'task-9', frame.task_terminal_status ? { task_terminal_status: 'completed', outcome_axes: { execution: 'ok' } } : {});
        r.emit('chat', { chat_id: 2, role: 'assistant', is_progress: true, task_id: 'task-9', ...frame });
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'after'], 'one read of the rows stored since, not the window');
});

test('a reconnect of a hydrated room reads its newest page: frames sent while it was away are lost', async (t) => {
    // A running task's facts the socket carried (a question's wait ended, a phase) reach a window that
    // missed them only through a whole read; its arrivals would bring its new rows but not those facts.
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    s.say(2);
    r.reconnect();
    await until(() => r.shown().includes('chat:2'));
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'newest']);
    s.say(3);
    assert.equal((await r.instance.refreshHistory({ revision: 3 })).read, true);
    assert.deepEqual(s.reads.slice(2), ['after'], 'arrivals go on from the page the reconnect read');
});

for (const refusal of ['history_delta_reset', 'history_view_changed']) {
    test(`a refused read of arrivals (${refusal}) takes the newest page once, then arrivals again`, async (t) => {
        const s = historyServer();
        s.say(1);
        const r = openRoom(t, s);
        await r.instance.refreshHistory({ revision: 1 });
        s.say(2);
        s.refusals.push(refusal);
        assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
        assert.deepEqual(s.reads, ['newest', 'after', 'newest']);
        s.say(3);
        assert.equal((await r.instance.refreshHistory({ revision: 3 })).read, true);
        assert.deepEqual(s.reads.slice(3), ['after'], 'the newest page names where arrivals go on');
        assert.equal(s.cursors.at(-1), 'after:2');
    });
}

test('a read of arrivals with a gap in the conversation takes the newest page instead', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const fetchHistory = s.fetchHistory;
    let gapped = true;
    globalThis.fetch = async (url) => {
        const response = await fetchHistory(url);
        if (!gapped || !String(url).includes('cursor=after')) return response;
        gapped = false;
        const data = await response.json();
        return { ...response, json: async () => ({ ...data, window: { complete: false, truncated_by: ['page', 'chat_incomplete_live_line'] } }) };
    };
    s.say(2);
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
    assert.deepEqual(s.reads, ['newest', 'after', 'newest'], 'an unfinished line may be the newest message: the newest page names it');
});

test('reconnects during a slow newest read fold into one more read', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const release = s.hold('newest');
    for (let index = 0; index < 10; index += 1) { r.reconnect(); await quiet(); }
    assert.deepEqual(s.reads, ['newest', 'newest'], 'one newest read in flight');
    release();
    await until(() => s.reads.length >= 3);
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'newest', 'newest'], 'ten reconnects, one queued rebuild');
});

test('triggers during a read of arrivals share one more read begun after them', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const release = s.hold();
    s.say(2);
    const first = r.instance.refreshHistory({ revision: 2 });
    await until(() => s.reads.length === 2);
    s.say(3);
    const second = r.instance.refreshHistory({ revision: 3 });
    s.say(4);
    const third = r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.equal(s.reads.length, 2, 'one read in flight; the rest wait for it');
    release();
    const [one, two, three] = await Promise.all([first, second, third]);
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'after', 'after'], 'three triggers, one queued read');
    assert.deepEqual([one.painted, two.painted, three.read], [false, false, true], 'the latest refresh decides');
    assert.deepEqual(['chat:2', 'chat:3', 'chat:4'].map((id) => r.shown().includes(id)), [true, true, true]);
});

test('arrivals that land while a whole newest read is in flight are read again when it lands', async (t) => {
    const s = historyServer();
    s.say(1);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const releaseArrivals = s.hold('after');
    s.say(2);
    const first = r.instance.refreshHistory({ revision: 2 });  // arrivals in flight, holding chat:2
    await until(() => s.reads.length === 2);
    const second = r.instance.refreshHistory({ revision: 3 });  // queued behind them
    const releaseNewest = s.hold('newest');
    r.reconnect();  // a whole read begins: its page will not hold what is stored next
    await until(() => s.reads.length === 3);
    s.say(3);
    releaseArrivals();
    await Promise.all([first, second]);
    await until(() => r.shown().includes('chat:3'));
    releaseNewest();
    await until(() => s.reads.length === 5);
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'after', 'newest', 'after', 'after'], 'the page that let chat:3 go reads its arrivals');
    assert.ok(r.shown().includes('chat:3'), 'the row stays shown');
});

test('arrivals that land while ↓ reads the present are read again on the page it lands', async (t) => {
    const s = historyServer({ window: 3 });
    for (let n = 1; n <= 15; n += 1) s.say(n);
    drawBelow(t, () => true);  // nothing on screen holds a page or row in place
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    const button = r.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    for (let press = 0; press < 3; press += 1) for (const handler of button.listeners.get('click')) await handler({ target: button });
    const release = s.hold('newest');
    globalThis.document.byId.get('chat-scroll-bottom').click();  // the pager let the present go: ↓ reads it
    await until(() => s.reads.length === 5);
    s.say(16);
    s.say(17);
    await r.instance.refreshHistory({ revision: 2 });
    assert.ok(r.shown().includes('chat:17'), 'what arrived is shown while ↓ is in flight');
    release();
    await until(() => s.reads.length === 7);
    await quiet();
    assert.deepEqual(s.reads, ['newest', 'older', 'older', 'older', 'newest', 'after', 'after'],
        '↓\'s page began before them: arrivals are read once more from its cursor');
    assert.equal(s.cursors.at(-1), 'after:15');
    assert.deepEqual(['chat:16', 'chat:17'].map((id) => r.shown().includes(id)), [true, true]);
});

// More than a default newest page (150 conversation + 60 narration rows) arrives at once.
const burst = (s, from) => { for (let n = from; n < from + 211; n += 1) s.say(n); };

test('a window that took in a whole page of arrivals reads the newest page once and lets the oldest go', async (t) => {
    const s = historyServer({ window: 3 });
    for (const n of [1, 2, 3]) s.say(n);
    drawBelow(t, (id) => /^chat:\d+$/.test(id) && Number(id.slice(5)) < 300);  // what the newest page lets go is off screen
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    burst(s, 10);
    await r.instance.refreshHistory({ revision: 2 });
    assert.equal(r.shown().length, 214, 'the burst arrives whole');
    for (const n of [300, 301]) {
        s.say(n);
        await r.instance.refreshHistory({ revision: n });
    }
    assert.deepEqual(s.reads, ['newest', 'after', 'newest', 'after'],
        'past a page of arrivals the next read is the newest page, read once; then arrivals again');
    assert.deepEqual(r.shown(), ['chat:219', 'chat:220', 'chat:300', 'chat:301'], 're-anchored there, the older rows are let go');
    const button = r.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    for (const handler of button.listeners.get('click')) await handler({ target: button });
    assert.deepEqual([s.reads.at(-1), s.cursors.at(-1)], ['older', 'older:212'], 'and Load more goes on straight below it');
});

test('a reader in older history keeps their place while arrivals fill the window', async (t) => {
    const s = historyServer({ window: 3 });
    for (let n = 1; n <= 9; n += 1) s.say(n);
    const r = openRoom(t, s);
    await r.instance.refreshHistory({ revision: 1 });
    r.readUp();  // the reader reached the top: the page above is read, and they read on there
    await until(() => r.shown().includes('chat:4'));
    burst(s, 10);
    await r.instance.refreshHistory({ revision: 2 });
    s.say(300);
    await r.instance.refreshHistory({ revision: 3 });
    assert.deepEqual(s.reads, ['newest', 'older', 'after', 'newest']);
    assert.equal(r.shown().length, 6 + 211 + 1, 'nothing the reader may scroll to is taken away');
    const button = r.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    for (const handler of button.listeners.get('click')) await handler({ target: button });
    assert.equal(s.cursors.at(-1), 'older:3', 'the next press reads above what is being read');
});

test('review details of sixty cards hold at most two reads in flight, newest asks first', async () => {
    const started = [], release = [];
    let active = 0, peak = 0;
    const hydrator = createReviewHydrator({
        fetchDetail: (taskId) => {
            started.push(taskId);
            active += 1; peak = Math.max(peak, active);
            return new Promise((resolve) => release.push(() => { active -= 1; resolve({ id: taskId }); }));
        },
        applyDetail: () => true,
    });
    const owners = Array.from({ length: 60 }, (_, index) => `owner-${index}`);
    const done = owners.map((id) => hydrator.hydrate(id, 'a'.repeat(64)));
    await quiet();
    assert.deepEqual([started, active], [['owner-0', 'owner-1'], REVIEW_DETAIL_READS]);
    void hydrator.hydrate('owner-30', 'a'.repeat(64));
    await quiet();
    assert.equal(started.length, 2, 'a queued owner asked again joins its queued read');
    while (release.length) {
        release.shift()();
        await quiet();
        assert.ok(active <= REVIEW_DETAIL_READS);
    }
    await Promise.all(done);
    assert.equal(peak, 2);
    assert.equal(started.length, 60, 'every owner is read once');
    assert.deepEqual(started.slice(2, 4), ['owner-59', 'owner-58'], 'a room opens at its newest cards: they are read first');
});

test('a review detail read that never answers is aborted at the slot bound: two connections at most, Retry after', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const started = [], release = {}, signals = {};
    const open = () => started.filter((id) => !release[id].settled && !signals[id].aborted).length;
    const hydrator = createReviewHydrator({
        fetchDetail: (taskId, { signal } = {}) => {
            started.push(taskId); signals[taskId] = signal;
            return new Promise((resolve) => {
                release[taskId] = () => { release[taskId].settled = true; resolve({ id: taskId }); };
            });
        },
        applyDetail: () => true,
    });
    const done = ['a', 'b', 'c'].map((id) => hydrator.hydrate(id, 'a'.repeat(64)));
    await quiet();
    assert.deepEqual([started, open()], [['a', 'b'], 2], 'two in flight');
    t.mock.timers.tick(REVIEW_DETAIL_SLOT_MS - 1);
    await quiet();
    assert.deepEqual(started, ['a', 'b'], 'a slow read keeps its slot');
    t.mock.timers.tick(1);
    await quiet();
    assert.deepEqual(started, ['a', 'b', 'c'], 'two dead connections do not stall the rest');
    assert.ok(signals.a.aborted && signals.b.aborted, 'the lost reads are aborted, so their connections close');
    assert.ok(open() <= REVIEW_DETAIL_READS, 'never more than two reads hold a connection');
    release.c();
    assert.deepEqual(await Promise.all(done), [false, false, true], 'the lost reads fail and await Retry; the next one lands');
});

test('review readers of two rooms share the bound, and a closed room\'s queued reads never start', async () => {
    const started = [], release = [];
    const reader = () => createReviewHydrator({
        fetchDetail: (taskId) => { started.push(taskId); return new Promise((resolve) => release.push(() => resolve({}))); },
        applyDetail: () => true,
    });
    const main = reader(), project = reader();
    const mainDone = Array.from({ length: 5 }, (_, index) => main.hydrate(`main-${index}`));
    const projectDone = Array.from({ length: 5 }, (_, index) => project.hydrate(`project-${index}`));
    await quiet();
    assert.deepEqual(started, ['main-0', 'main-1'], 'one bound for every room on the page');
    project.clear();
    assert.deepEqual(await Promise.all(projectDone), [false, false, false, false, false], 'its queued reads settle unread');
    while (release.length) { release.shift()(); await quiet(); }
    await Promise.all(mainDone);
    assert.deepEqual(started.filter((id) => id.startsWith('project-')), [], 'a closed room starts none');
    assert.equal(started.length, 5);
});

test('a room replaying sixty review owners reads their details two at a time; closed, it starts no more', async (t) => {
    const s = historyServer();
    for (let index = 0; index < 60; index += 1) {
        s.rows.push({ role: 'assistant', text: '', is_progress: true, system_type: 'review_reference', surface: 'task_acceptance',
            task_id: `owner-${index}`, state_revision: (index % 16).toString(16).repeat(64), ts: ts(index),
            history_id: `progress:${index}`, history_position: { source: 'progress', offset: index } });
    }
    s.say(100);
    const r = openRoom(t, s);
    const detail = { active: 0, peak: 0, started: [], release: [] };
    const base = globalThis.fetch;
    globalThis.fetch = async (url, init) => {
        if (!String(url).startsWith('/api/tasks/')) return base(url, init);
        detail.active += 1; detail.peak = Math.max(detail.peak, detail.active); detail.started.push(String(url));
        await new Promise((resolve) => detail.release.push(resolve));
        detail.active -= 1;
        return { ok: true, status: 200, json: async () => ({ status: 'completed' }) };
    };
    await r.instance.refreshHistory({ revision: 1 });
    await until(() => detail.started.length === 2);
    await quiet();
    assert.equal(detail.started.length, 2, 'two of sixty, not sixty at once');
    for (let round = 0; round < 10; round += 1) {
        detail.release.shift()();
        await until(() => detail.started.length === 3 + round);
    }
    r.instance.destroy();
    while (detail.release.length) { detail.release.shift()(); await quiet(); }
    assert.equal(detail.peak, 2);
    assert.equal(detail.started.length, 12, 'the closed room started none of its queued reads');
});
