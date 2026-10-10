// Issue #1102, defect 3: a Project panel whose history read was still in flight
// (or had failed outright) showed an empty feed under a green "Online" header.
// These cases drive the real createChatInstance through the shared DOM fixture.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const TS = '2026-09-12T12:00:00.000Z';
const row = (id, text) => ({
    role: 'assistant', text, ts: TS, history_id: id,
    history_position: { source: 'chat:rotation-1', offset: Number(id.split(':').at(-1)) || 0 },
});
const page = (messages) => ({
    messages, page_cursor: 'page:recent', next_cursor: null, has_more: false,
    window: { complete: true, truncated_by: [] },
});
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

// Every history read parks until the test answers it, so the DOM can be read
// while the request is provably still in flight.
function fixture(t, options = {}) {
    const reads = [];
    const handlers = new Map();
    const { prior, mount } = installDom(async (url) => {
        if (!String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ active_direct_turns: [] }) };
        }
        return new Promise((resolve, reject) => reads.push({
            ok: data => resolve({ ok: true, status: 200, json: async () => data }),
            http: status => resolve({ ok: false, status, json: async () => ({ error: `HTTP ${status}` }) }),
            drop: () => reject(new TypeError('Failed to fetch')),
        }));
    });
    const priorSocket = globalThis.WebSocket;
    const priorError = console.error;
    const logged = [];
    console.error = (...args) => logged.push(args);
    globalThis.WebSocket = { OPEN: 1 };
    const instance = createChatInstance({
        ws: { on(type, handler) { handlers.set(type, handler); return () => handlers.delete(type); },
            isConnected: () => true, send() {}, ws: { readyState: 1 } },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true, ...options,
    });
    t.after(() => {
        instance.destroy(); restoreDom(prior);
        globalThis.WebSocket = priorSocket; console.error = priorError;
    });
    const messages = globalThis.document.byId.get('chat-messages');
    const controls = () => messages.querySelector('.chat-load-older');
    return {
        instance, reads, messages, controls, logged,
        button: () => controls()?.querySelector('.chat-load-older-btn'),
        note: () => (messages.parentNode.querySelector('.chat-panel-statusbar').querySelector('.chat-load-older-note') || messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note')),
        bubbles: () => messages.children.filter(node => node.classList.contains('chat-bubble')
            && !node.classList.contains('typing-bubble')),
        async clickRetry() {
            for (const handler of this.button().listeners.get('click')) await handler({ target: this.button() });
        },
    };
}

const assertLoading = (f, why) => {
    assert.ok(f.controls(), `${why}: the history control is mounted in the feed`);
    assert.equal(f.controls().hidden, false, why);
    assert.equal(f.controls().getAttribute('aria-busy'), 'true', why);
    assert.equal(f.button().textContent, 'Loading saved history…', why);
    assert.equal(f.button().hidden, false, why);
    assert.equal(f.button().disabled, true, why);
};

test('an empty feed shows the loading state BEFORE the history request resolves', async (t) => {
    const f = fixture(t);
    const paint = f.instance.refreshHistory({ revision: 1 });
    await settle();
    assert.equal(f.reads.length, 1, 'the history request is in flight and unanswered');
    assert.equal(f.bubbles().length, 0);
    assertLoading(f, 'while the read is in flight');

    f.reads[0].ok(page([row('chat:10', 'First saved answer')]));
    assert.equal((await paint).painted, true);
    assert.equal(f.controls().getAttribute('aria-busy'), '', 'the loading state is lifted on success');
    assert.notEqual(f.button().textContent, 'Loading saved history…');
    assert.equal(f.bubbles().length, 1);
});

test('an ordinary refresh over a painted transcript keeps every rendered message and adds no loading chrome', async (t) => {
    const f = fixture(t);
    const first = f.instance.refreshHistory({ revision: 1 });
    await settle();
    f.reads[0].ok(page([row('chat:10', 'Kept answer one'), row('chat:20', 'Kept answer two')]));
    await first;
    const painted = f.bubbles();
    assert.equal(painted.length, 2);

    const second = f.instance.refreshHistory({ revision: 2 });
    await settle();
    assert.equal(f.reads.length, 2, 'the refresh request is in flight and unanswered');
    assert.deepEqual(f.bubbles(), painted, 'the painted messages stay mounted while the refresh is in flight');
    assert.equal(f.controls().getAttribute('aria-busy'), '');
    assert.notEqual(f.button().textContent, 'Loading saved history…');

    f.reads[1].ok(page([row('chat:10', 'Kept answer one'), row('chat:20', 'Kept answer two')]));
    assert.equal((await second).painted, true);
    assert.deepEqual(f.bubbles(), painted, 'the same nodes survive the refresh');
});

test('a failed refresh over a painted transcript keeps it as it is and is never reported as painted', async (t) => {
    const f = fixture(t);
    const first = f.instance.refreshHistory({ revision: 1 });
    await settle();
    f.reads[0].ok(page([row('chat:10', 'Still readable')]));
    await first;
    const painted = f.bubbles();

    const second = f.instance.refreshHistory({ revision: 2 });
    await settle();
    f.reads[1].http(500);
    assert.deepEqual(await second, { painted: false, revision: 2 }, 'a failed read is not a paint receipt');
    assert.deepEqual(f.bubbles(), painted);
    assert.equal(f.button().textContent, 'Retry loading messages', 'painted rows remain readable and failure stays retryable');
    assert.equal(f.logged.length, 1, 'the failure is still reported, not swallowed');
});

for (const [name, fail] of [['a dropped connection', read => read.drop()], ['an HTTP 500', read => read.http(500)]]) {
    test(`${name} on an empty feed shows the error and the existing Retry, never a blank feed`, async (t) => {
        const f = fixture(t);
        const paint = f.instance.refreshHistory({ revision: 1 });
        await settle();
        assertLoading(f, 'before the failure');
        fail(f.reads[0]);
        assert.deepEqual(await paint, { painted: false, revision: 1 }, 'a failed read is not a paint receipt');

        assert.equal(f.controls().hidden, false);
        assert.equal(f.controls().getAttribute('aria-busy'), '', 'a failure is not a loading state');
        assert.equal(f.button().textContent, 'Retry loading messages');
        assert.equal(f.button().hidden, false);
        assert.equal(f.button().disabled, false);
        assert.equal(f.note().hidden, false);
        assert.match(f.note().textContent, /^Some saved history could not be loaded/);
        assert.equal(f.instance.hasPaintedHistory(), false);

        const retry = f.clickRetry();
        await settle();
        assert.equal(f.reads.length, 2, 'Retry starts a fresh read');
        assertLoading(f, 'while the retry is in flight');
        f.reads[1].ok(page([row('chat:10', 'Recovered answer')]));
        await retry;
        assert.equal(f.bubbles().length, 1);
        assert.notEqual(f.button().textContent, 'Retry loading messages');
        assert.doesNotMatch(f.note().textContent, /Could not load/);
        assert.equal(f.instance.hasPaintedHistory(), true);
    });
}

test('Retry hands the read to the owner transaction once, and refetches itself when the owner declines', async (t) => {
    let owner = 'accept', asked = 0;
    const f = fixture(t, { onHistoryRetry: () => {
        asked += 1;
        return owner === 'accept' ? f.instance.refreshHistory({ revision: 7 }) : Promise.resolve();
    } });
    const paint = f.instance.refreshHistory({ revision: 7 });
    await settle();
    f.reads[0].drop();
    await paint;

    const owned = f.clickRetry();
    await settle();
    assert.equal(asked, 1);
    assert.equal(f.reads.length, 2, 'the owner transaction is the only read: no second fetch beside it');
    f.reads[1].drop();
    await owned;
    assert.equal(f.button().textContent, 'Retry loading messages', 'a retry that fails again is still retryable');

    owner = 'decline';
    const local = f.clickRetry();
    await settle();
    assert.equal(asked, 2);
    assert.equal(f.reads.length, 3, 'a declined Retry still reads: the button is never dead');
    f.reads[2].ok(page([row('chat:10', 'Recovered answer')]));
    await local;
    assert.equal(f.bubbles().length, 1);
});

test('closing the panel while its first read is in flight leaves no late write and no paint receipt', async (t) => {
    const f = fixture(t);
    const paint = f.instance.refreshHistory({ revision: 1 });
    await settle();
    assertLoading(f, 'before the panel closes');
    const controls = f.controls();
    f.instance.destroy();
    f.reads[0].ok(page([row('chat:10', 'Arrived after close')]));
    assert.deepEqual(await paint, { painted: false, revision: 1 });
    assert.equal(f.bubbles().length, 0, 'a closed room consumes no late response');
    assert.equal(controls.querySelector('.chat-load-older-btn').textContent, 'Loading saved history…',
        'destroy() makes late continuations no-ops instead of repainting a removed control');
});

test('delayed latest cannot certify newer retained recent rows until the physical gap is filled', async t => {
    const f = fixture(t);
    const covered = (from, to, upper, id, next = null) => ({
        ...page([row(`chat:${id}`, `Row ${id}`)]), page_cursor: `p:${from}`, next_cursor: next, has_more: Boolean(next),
        coverage: { v: 1, view: 'room', upper: { chat: upper, progress: 0 }, spans: {
            chat: { from, to, chain: 'retained', gaps: [] },
            progress: { from: 0, to: 0, chain: 'empty', gaps: [] },
        } },
    });
    const first = f.instance.refreshHistory({ revision: 1 });
    await settle(); f.reads[0].ok(covered(0, 80, 80, 70)); await first;
    const second = f.instance.refreshHistory({ revision: 2 });
    await settle(); f.reads[1].ok(covered(90, 100, 100, 95)); await second;
    await settle(); assert.equal(f.reads.length, 3, 'latest rebase is held');
    const third = f.instance.refreshHistory({ revision: 3 });
    await settle(); f.reads[3].ok(covered(180, 200, 200, 190)); await third;
    f.reads[2].ok(covered(0, 100, 100, 95, 'fill-gap'));
    await settle(); await settle();
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:190'));
    assert.match(f.note().textContent, /Shown messages may have gaps/);
    assert.doesNotMatch(f.note().textContent, /Beginning/);
    const stale = f.instance.refreshHistory({ revision: 4 });
    await settle();
    const fill = f.clickRetry();
    await settle(); f.reads[5].ok(covered(100, 200, 200, 190)); await fill;
    f.reads[4].ok(covered(180, 200, 200, 195)); await stale;
    assert.equal(f.bubbles().some(node => node.dataset.historyId === 'chat:195'), false,
        'an ordinary read superseded by latest cannot mount unowned stale rows');
    assert.equal(f.note().textContent, 'Beginning of saved history');
});

test('a room that grew past one window while open offers its older rows, and one press reads them', async t => {
    // Opened while empty (a new Project): the pager's chain is that one complete read.
    // The room then grows past a window; a later recent read starts beyond the chain.
    const f = fixture(t);
    const covered = (from, to, upper, ids, next = null, cursor = `p:${from}`) => ({
        ...page(ids.map(id => row(`chat:${id}`, `Row ${id}`))), page_cursor: cursor, next_cursor: next, has_more: Boolean(next),
        window: { complete: !next, truncated_by: next ? ['quota'] : [] },
        coverage: { v: 1, view: 'room', upper: { chat: upper, progress: 0 }, spans: {
            chat: { from, to, chain: 'retained', gaps: [] },
            progress: { from: 0, to: 0, chain: 'empty', gaps: [] },
        } },
    });
    const first = f.instance.refreshHistory({ revision: 1 });
    await settle(); f.reads[0].ok(covered(0, 10, 10, [])); await first;
    assert.equal(f.button().hidden, true, 'an empty complete room has nothing older');
    const grown = f.instance.refreshHistory({ revision: 2 });
    await settle(); f.reads[1].ok(covered(60, 100, 100, [80, 90], 'before:60')); await grown;
    assert.equal(f.button().hidden, false, 'rows between the old chain and the newest read are older history to offer');
    const press = f.clickRetry();
    await settle(); assert.equal(f.reads.length, 3, 'the press first re-anchors the chain at the newest read');
    f.reads[2].ok(covered(60, 100, 100, [80, 90], 'before:60', 'p:latest'));
    await settle(); await settle();
    assert.equal(f.reads.length, 4, 'and the same press goes on into the older rows');
    f.reads[3].ok(covered(10, 60, 100, [20, 50]));
    await press; await settle();
    const shown = f.bubbles().map(node => node.dataset.historyId);
    assert.ok(shown.includes('chat:20') && shown.includes('chat:50'), shown);
    assert.equal(f.button().hidden, true, 'the beginning is reached');
});
