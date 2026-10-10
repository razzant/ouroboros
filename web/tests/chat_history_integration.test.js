import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, ElementStub, walkCard } from './chat_dom_fixture.js';

const TS = '2026-09-12T12:00:00.000Z';
const row = (id, text, extra = {}) => ({
    role: 'assistant', text, ts: TS, history_id: id,
    history_position: { source: 'chat:rotation-1', offset: Number(id.split(':').at(-1)) || 0 },
    ...extra,
});
const page = (messages, cursor = 'page:recent', next = null) => ({
    messages, page_cursor: cursor, next_cursor: next, has_more: next !== null,
    window: { complete: next === null, truncated_by: next ? ['quota'] : [] },
});

function fixture(t, initial = page([]), fetchPage = null) {
    let response = initial, revision = 0;
    const calls = [], handlers = new Map();
    const { prior, mount } = installDom(async (url, init) => {
        if (String(url).startsWith('/api/chat/history')) {
            const cursor = new URL(String(url), 'http://local').searchParams.get('cursor');
            calls.push(cursor);
            const data = cursor && fetchPage ? await fetchPage(cursor, init) : response;
            return { ok: true, json: async () => data };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1 };
    const instance = createChatInstance({
        ws: { on(type, handler) { handlers.set(type, handler); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
    });
    t.after(() => { instance.destroy(); restoreDom(prior); globalThis.WebSocket = priorSocket; });
    const messages = globalThis.document.byId.get('chat-messages');
    return {
        instance, calls, messages,
        bubbles: () => messages.children.filter((node) => node.classList.contains('chat-bubble')
            && !node.classList.contains('typing-bubble') && node.dataset.ephemeral !== '1'),
        emit: (message) => handlers.get('chat')({ chat_id: 2, ...message }),
        reconnect: () => handlers.get('open')({ previouslyConnected: true }),
        respond: next => { response = next; },
        async refresh(next = response) {
            response = next;
            const result = await instance.refreshHistory({ revision: ++revision });
            assert.equal(result.painted, true, 'the readable response reaches paint ACK independently of navigation');
        },
        async clickOlder() {
            const button = messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
            assert.equal(button.hidden, false);
            for (const handler of button.listeners.get('click')) await handler({ target: button });
        },
    };
}

// Edge navigation is bound to the real `scroll` event and starts its reads
// without returning a promise, so a test dispatches the event and then lets the
// fetch chain drain until it stops asking for pages.
async function scrollEdge(f, scrollTop = 0) {
    f.messages.scrollTop = scrollTop;
    for (const handler of f.messages.listeners.get('wheel')) handler({ type: 'wheel', deltaY: scrollTop ? 1 : -1 });
    for (const handler of f.messages.listeners.get('scroll')) handler({});
    for (let n = 0, spent = -1; spent !== f.calls.length && n < 40; n += 1) {
        spent = f.calls.length;
        await new Promise(resolve => setTimeout(resolve, 0));
    }
}

// Reading protection pins any row the reader can see; park every history row
// off-screen so the page budget, not the viewport, decides what is released.
function offScreen(t) {
    const oldRect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        return this.dataset.historyId
            ? { top: 1000, bottom: 1020, left: 0, right: 100, width: 100, height: 20 }
            : oldRect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = oldRect; });
}

// The reader turned back into history: an upward wheel away from the top edge.
function readUp(f) {
    f.messages.scrollTop = 400;
    for (const handler of f.messages.listeners.get('wheel')) handler({ type: 'wheel', deltaY: -1, target: f.messages, timeStamp: 0 });
}

function recordRenderedMarkup(t) {
    const query = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function (selector) {
        const direct = query.call(this, selector);
        if (direct) return direct;
        for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
        return null;
    };
    t.after(() => { ElementStub.prototype.querySelector = query; });
    const descriptor = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML'), markup = [];
    Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...descriptor,
        set(value) { markup.push(String(value)); descriptor.set.call(this, value); } });
    t.after(() => Object.defineProperty(ElementStub.prototype, 'innerHTML', descriptor));
    return markup;
}

test('R3 a terminal census before first history preserves source narration and its chronological anchor', async t => {
    const text = 'Retained authored evidence from the original task';
    const progress = row('progress:11', text, { task_id: 'closed-before-history', is_progress: true, narration: true });
    const markup = recordRenderedMarkup(t);
    const f = fixture(t, page([progress, row('chat:12', 'A newer conversation row', { ts: '2026-09-12T12:01:00Z' })]));
    f.instance.hydrateStateSnapshot({ active_chat_activities: [{ activity_id: progress.task_id, chat_id: 2,
        kind: 'managed_task', phase: 'working', status: 'cancelled' }], active_chat_activities_complete: true,
        supervisor_ready: true });
    await f.refresh();
    const card = walkCard(f.messages, progress.task_id);
    assert.ok(card);
    assert.equal(card.querySelector('[data-live-title]').textContent, text);
    assert.equal(card.querySelector('[data-live-phase]').textContent, 'Cancelled');
    assert.equal(card.querySelector('[data-live-phase]').dataset.motion, '0');
    assert.equal(card.dataset.finished, '1');
    assert.equal(card.dataset.ts, String(Date.parse(progress.ts)));
    assert.ok(markup.some(value => value.includes(text) && value.includes('chat-live-line')), 'the retained source row is rendered');
    assert.ok(f.messages.children.indexOf(card) < f.messages.children.findIndex(node => node.dataset.historyId === 'chat:12'));
    await f.refresh();
    assert.equal(walkCard(f.messages, progress.task_id), card);
    assert.equal(card.querySelector('[data-live-phase]').textContent, 'Cancelled', 'replay without a terminal row keeps the known census outcome');
});

// One saved message and nothing else: every page below it is empty down to the
// archive floor. This is the sparse Project room that grew a 64-click pill.
const sparseRoom = (t, floor = 5) => fixture(t,
    page([row('chat:900', 'The one saved message')], 'page:recent', 'before:1'),
    (cursor) => {
        const index = Number(cursor.split(':').at(-1));
        return page([], `page:empty-${index}`, index < floor ? `before:${index + 1}` : null);
    });

test('a sparse room walks to its floor in one gesture without minting a Load-newer control', async (t) => {
    const f = sparseRoom(t);
    await f.refresh();
    await scrollEdge(f);
    assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3', 'before:4', 'before:5'],
        'a press or gesture keeps reading until rows land or history ends: never an empty press');
    const controls = f.messages.querySelector('.chat-load-older');
    assert.equal((f.messages.parentNode.querySelector('.chat-panel-statusbar').querySelector('.chat-load-older-note') || f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note')).textContent, 'Some saved history is not loaded. Shown messages may have gaps.');
    assert.equal(controls.querySelector('.chat-load-older-btn').hidden, true);
    assert.equal(f.messages.querySelector('.chat-load-newer'), null, 'walked-over empty pages are not a gap');
    assert.equal(f.bubbles().filter(node => node.dataset.historyId === 'chat:900').length, 1);
});

test('a walked-out sparse room asks for nothing more, however often the reader scrolls', async (t) => {
    const f = sparseRoom(t);
    await f.refresh();
    await scrollEdge(f);
    // Everything fits on one screen, so the reader is at the top edge and near the
    // bottom at the same time: both edges are live on every one of these events.
    await scrollEdge(f); await scrollEdge(f);
    const spent = f.calls.length;
    for (let n = 0; n < 5; n += 1) await scrollEdge(f);
    assert.equal(f.calls.length, spent, 'no pill to click 64 times, and no request storm behind it');
    assert.equal(f.messages.querySelector('.chat-load-newer'), null);
});

test('reading to the oldest page hides the button; ↓ returns to the present with one read', async (t) => {
    offScreen(t);
    const recent = page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1');
    const older = [
        page([row('chat:300', 'Older one')], 'page:1', 'before:2'),
        page([row('chat:200', 'Older two')], 'page:2', 'before:3'),
        page([row('chat:100', 'Older three')], 'page:3'),
    ];
    const served = new Map([['before:1', older[0]], ['before:2', older[1]], ['before:3', older[2]],
        ...[recent, ...older].map(item => [item.page_cursor, item])]);
    const f = fixture(t, recent, (cursor) => served.get(cursor));
    await f.refresh();
    for (let n = 0; n < 3; n += 1) await scrollEdge(f);
    assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3']);
    assert.equal(f.bubbles().filter(node => node.dataset.historyId === 'chat:900').length, 1,
        'the recent rows stay mounted after the pager released their page');
    const button = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    assert.equal(button.hidden, true, 'nothing older remains, and the button never turns into a way to newer pages');
    globalThis.document.byId.get('chat-scroll-bottom').click();
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(f.calls.slice(4), [null], '↓ returns to the present by one latest read');
    const ids = [...f.messages.querySelectorAll('[data-history-id]')].map(node => node.dataset.historyId);
    assert.equal(new Set(ids).size, ids.length, 'a replayed page mounts no duplicate row');
});

test('Load more history only ever reads older pages, however deep the reader goes', async (t) => {
    // The livelock of 2026-10-02: a 3-page cache released page zero, the button
    // then read it back as "newer", the next press re-read the deep page, forever.
    offScreen(t);
    const deep = 7;
    const f = fixture(t, page([row('chat:9000', 'Newest')], 'page:recent', 'before:1'), (cursor) => {
        const index = Number(cursor.split(':').at(-1));
        return page([row(`chat:${9000 - index * 10}`, `Older ${index}`)], `page:${index}`,
            index < deep ? `before:${index + 1}` : null);
    });
    await f.refresh();
    for (let press = 1; press <= deep; press += 1) await f.clickOlder();
    assert.deepEqual(f.calls, [null, ...Array.from({ length: deep }, (_, index) => `before:${index + 1}`)],
        'each press read the next older page; none re-read a newer one');
    const button = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    assert.equal(button.hidden, true, 'at the beginning of history the button leaves');
});

test('reading to the beginning of a long room leaves no dead button behind released pages', async (t) => {
    // Rows stay off-screen, so the 3-page budget releases the newest pages as the reader goes deep.
    offScreen(t);
    const span = (from, to) => ({ v: 1, view: 'room', upper: { chat: 600, progress: 0 }, spans: {
        chat: { from, to, chain: 'retained', gaps: [] }, progress: { from: 0, to: 0, chain: 'empty', gaps: [] } } });
    const recent = { ...page([row('chat:550', 'Newest')], 'page:recent', 'before:500'), coverage: span(500, 600) };
    const f = fixture(t, recent, (cursor) => {
        const end = Number(cursor.split(':').at(-1)), from = end - 100;
        return { ...page([row(`chat:${from + 50}`, `Row ${from}`)], `page:${end}`, from > 0 ? `before:${from}` : null),
            coverage: span(from, end) };
    });
    await f.refresh();
    for (let press = 0; press < 5; press += 1) await f.clickOlder();
    assert.deepEqual(f.calls, [null, 'before:500', 'before:400', 'before:300', 'before:200', 'before:100']);
    const button = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    assert.equal(button.hidden, true, 'at the beginning no control offers a press that reads nothing');
});

test('a refresh while the reader reads older history leaves the next press reading older', async (t) => {
    // A task finishes while the reader is a page back: the newest read moved on and
    // let its oldest row go. Re-anchoring there sent the next press to rows below
    // the reader, newer than what they were reading (the press "returned nothing").
    offScreen(t);
    const f = fixture(t, page([row('chat:500', 'Newest at open')], 'page:recent', 'before:500'), (cursor) => {
        const end = Number(cursor.split(':').at(-1));
        return page([row(`chat:${end - 50}`, `Row ${end - 50}`)], `page:${end}`, end > 100 ? `before:${end - 100}` : null);
    });
    await f.refresh();
    await f.clickOlder();
    readUp(f);
    await f.refresh(page([row('chat:600', 'Arrived while reading')], 'page:recent', 'before:600'));
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    await f.clickOlder();
    assert.deepEqual(f.calls, [null, 'before:500', null, 'before:400'],
        'the refresh re-anchors nothing; the press reads the page older than the one being read');
    assert.equal(f.bubbles().filter(node => node.dataset.historyId === 'chat:500').length, 1,
        'the row the newest read let go stays between the reader and the present');
    // Back at the present (↓), the next shifted read re-anchors as before.
    globalThis.document.byId.get('chat-scroll-bottom').click();
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    await f.refresh(page([row('chat:700', 'Newest again')], 'page:recent', 'before:700'));
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(f.calls.slice(4), [null, null], 'a reader following the newest message gets the re-anchored chain');
});

test('the return to the present lets go of the windows kept while reading, leaving no unnoted hole', async (t) => {
    offScreen(t);
    const span = (from, to, upper) => ({ v: 1, view: 'room', upper: { chat: upper, progress: 0 }, spans: {
        chat: { from, to, chain: 'retained', gaps: [] }, progress: { from: 0, to: 0, chain: 'empty', gaps: [] } } });
    const recent = (ids, from, upper) => ({ ...page(ids.map(id => row(`chat:${id}`, `Row ${id}`)), 'page:recent', `before:${from}`),
        coverage: span(from, upper, upper) });
    const f = fixture(t, recent([500, 540], 500, 600), (cursor) => {
        const end = Number(cursor.split(':').at(-1));
        return { ...page([row(`chat:${end - 50}`, `Row ${end - 50}`)], `page:${end}`, `before:${end - 100}`), coverage: span(end - 100, end, 1100) };
    });
    await f.refresh();
    readUp(f);
    // Two refreshes while reading; nothing between 800 and 1000 was ever loaded.
    await f.refresh(recent([710, 750], 700, 800));
    await f.refresh(recent([1010, 1050], 1000, 1100));
    const shown = () => f.bubbles().map(node => node.dataset.historyId).filter(Boolean);
    assert.deepEqual(shown(), ['chat:500', 'chat:540', 'chat:710', 'chat:750', 'chat:1010', 'chat:1050'],
        'while reading, nothing the reader may scroll to is taken away');
    globalThis.document.byId.get('chat-scroll-bottom').click();
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(shown(), ['chat:1010', 'chat:1050'], 'back at the present, the kept windows go with the old chain');
    await f.clickOlder();
    assert.equal(f.calls.at(-1), 'before:1000', 'and the next press reads straight below the present');
});

test('a gap opening while the reader reads older history leaves the press reading older', async (t) => {
    offScreen(t);
    const span = (from, to, upper) => ({ v: 1, view: 'room', upper: { chat: upper, progress: 0 }, spans: {
        chat: { from, to, chain: 'retained', gaps: [] }, progress: { from: 0, to: 0, chain: 'empty', gaps: [] } } });
    const f = fixture(t, { ...page([row('chat:900', 'Newest at open')], 'page:recent', 'before:800'), coverage: span(800, 1000, 1000) },
        (cursor) => {
            const end = Number(cursor.split(':').at(-1));
            return { ...page([row(`chat:${end - 50}`, `Row ${end - 50}`)], `page:${end}`, `before:${end - 100}`),
                coverage: span(end - 100, end, 1000) };
        });
    await f.refresh();
    await f.clickOlder();
    readUp(f);
    // A reconnect after a long sleep: the newest read starts past everything loaded.
    await f.refresh({ ...page([row('chat:1500', 'After the sleep')], 'page:recent', 'before:1400'), coverage: span(1400, 1600, 1600) });
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    await f.clickOlder();
    assert.deepEqual(f.calls, [null, 'before:800', null, 'before:700'],
        'the press goes on above the reader; the gap below waits for ↓');
});

test('live message adopts its physical history identity without replacing the visible bubble', async (t) => {
    const f = fixture(t);
    await f.refresh();
    f.emit({ role: 'assistant', content: 'A useful answer', ts: TS });
    const live = f.bubbles().find((node) => node.innerHTML.includes('A useful answer'));
    assert.ok(live);
    await f.refresh(page([row('chat:100', 'A useful answer')]));
    const adopted = f.bubbles().filter((node) => node.innerHTML.includes('A useful answer'));
    assert.deepEqual(adopted, [live]);
    assert.equal(live.dataset.historyId, 'chat:100');
    assert.equal(live.dataset.historySource, 'chat:rotation-1');
    assert.equal(live.dataset.historyOffset, '100');
    await f.refresh(page([row('chat:100', 'A useful answer')]));
    assert.deepEqual(f.bubbles().filter((node) => node.innerHTML.includes('A useful answer')), [live]);
});

test('repeated physical history identity updates the routing annotation on the same user bubble', async (t) => {
    const message = row('chat:200', 'Please investigate this', {
        role: 'user', client_message_id: 'owner-message', chat_annotation: { status: 'pending' },
    });
    const f = fixture(t, page([message]));
    await f.refresh();
    const bubble = f.bubbles().find((node) => node.dataset.historyId === 'chat:200');
    assert.ok(bubble);
    assert.equal(bubble.dataset.chatAnnotationStatus, 'pending');
    const note = bubble.querySelector('.msg-routing-annotation');
    assert.match(note.textContent, /Choosing/);
    await f.refresh(page([{ ...message, chat_annotation: {
        status: 'delivered', action: 'steer', target: 'task-existing', target_label: 'Investigation',
    } }]));
    assert.equal(f.bubbles().filter((node) => node.dataset.historyId === 'chat:200').length, 1);
    assert.equal(f.bubbles().find((node) => node.dataset.historyId === 'chat:200'), bubble);
    assert.equal(bubble.querySelector('.msg-routing-annotation'), note);
    assert.equal(bubble.dataset.chatAnnotationStatus, 'delivered');
    assert.notEqual(note.textContent, 'Choosing the right destination…');
    assert.match(note.textContent, /Investigation/);
});

test('a replayed refusal receipt shows the host cause and a later scheduled receipt patches the same note', async (t) => {
    const cause = 'Not started: the working folder can\'t be used';
    const message = row('chat:250', 'Audit the GitHub tool', {
        role: 'user', client_message_id: 'owner-refused',
        chat_annotation: {
            action: 'promote_chat_to_task', status: 'needs_manual_target',
            target: 'never-started', target_label: 'Аудит', cause,
        },
    });
    const f = fixture(t, page([message]));
    await f.refresh();
    const bubble = f.bubbles().find((node) => node.dataset.historyId === 'chat:250');
    assert.ok(bubble);
    assert.equal(bubble.dataset.chatAnnotationStatus, 'needs_manual_target');
    const note = bubble.querySelector('.msg-routing-annotation');
    assert.equal(note.textContent, cause);
    await f.refresh(page([{ ...message, chat_annotation: {
        action: 'promote_chat_to_task', status: 'scheduled', target: 'task-started', target_label: 'Investigation',
        project_id: 'current-project', project_chat_id: 2,
    } }]));
    assert.equal(f.bubbles().filter((node) => node.dataset.historyId === 'chat:250').length, 1);
    assert.equal(f.bubbles().find((node) => node.dataset.historyId === 'chat:250'), bubble);
    assert.equal(bubble.querySelector('.msg-routing-annotation'), note);
    assert.equal(bubble.dataset.chatAnnotationStatus, 'scheduled');
    assert.equal(note.textContent, 'Started task · Investigation');
    assert.equal(bubble.querySelector('.msg-routing-actions'), null, 'the current Project is already displayed');
});

test('two physical rows with identical timestamp and body remain two messages across refresh', async (t) => {
    const rows = [row('chat:300', 'Same words'), row('chat:400', 'Same words')];
    const f = fixture(t, page(rows));
    await f.refresh();
    const bubbles = f.bubbles().filter((node) => node.innerHTML.includes('Same words'));
    assert.equal(bubbles.length, 2);
    assert.deepEqual(bubbles.map((node) => node.dataset.historyId), ['chat:300', 'chat:400']);
    await f.refresh(page(rows));
    assert.deepEqual(f.bubbles().filter((node) => node.innerHTML.includes('Same words')), bubbles);
});

test('one older-navigation action crosses a sparse page and displays the next physical row', async (t) => {
    const f = fixture(t, page([row('chat:900', 'Recent message')], 'page:recent', 'before:sparse'), (cursor) => {
        if (cursor === 'before:sparse') return page([], 'page:sparse', 'before:older');
        assert.equal(cursor, 'before:older');
        return page([row('chat:100', 'An older message', { ts: '2026-09-11T12:00:00Z' })], 'page:older');
    });
    await f.refresh();
    await f.clickOlder();
    assert.deepEqual(f.calls.filter(Boolean), ['before:sparse', 'before:older']);
    assert.equal(f.bubbles().filter((node) => node.dataset.historyId === 'chat:100').length, 1);
    assert.equal(f.bubbles().filter((node) => node.dataset.historyId === 'chat:900').length, 1);
});

test('retained history nodes still dedupe after navigation exceeds the live key FIFO', async (t) => {
    offScreen(t);
    const rows = Array.from({ length: 2500 }, (_, i) => row(`chat:${i}`, `Retained answer ${i}`));
    const pack = index => page(rows.slice(Math.max(0, rows.length - (index + 1) * 150), rows.length - index * 150),
        `page:${index}`, (index + 1) * 150 < rows.length ? `older:${index + 1}` : null);
    const f = fixture(t, pack(0), cursor => pack(Number(cursor.split(':')[1])));
    await f.refresh();
    const recent = f.bubbles().find(node => node.dataset.historyId === 'chat:2499');
    for (let i = 0; i < 16; i += 1) await f.clickOlder();
    const mounted = f.bubbles().length;
    assert.ok(mounted < 1000, 'distant pages actually evicted rather than all remaining protected');
    await f.refresh(pack(0));
    assert.deepEqual(f.bubbles().filter(node => node.dataset.historyId === 'chat:2499'), [recent]);
    assert.equal(f.bubbles().length, mounted, 'refresh adds no duplicate retained recent page');
});

test('closing the real chat instance aborts its pending archive fetch', async (t) => {
    let started;
    const ready = new Promise(resolve => { started = resolve; });
    const f = fixture(t, page([row('chat:10', 'Recent')], 'recent', 'older'), (_cursor, init) => {
        assert.ok(init.signal, 'the pager signal reaches the shared fetch transport');
        started(init.signal);
        return new Promise((_resolve, reject) => init.signal.addEventListener('abort',
            () => reject(new DOMException('Aborted', 'AbortError')), { once: true }));
    });
    await f.refresh();
    const pending = f.clickOlder();
    const signal = await ready;
    f.instance.destroy();
    assert.equal(signal.aborted, true);
    await pending;
});

test('history query construction remains portable when URLSearchParams.size is unavailable', async () => {
    const source = await import('../modules/api_client.js');
    const original = Object.getOwnPropertyDescriptor(URLSearchParams.prototype, 'size');
    Object.defineProperty(URLSearchParams.prototype, 'size', { configurable: true, value: undefined });
    const priorFetch = globalThis.fetch;
    let requested;
    globalThis.fetch = async (url) => {
        requested = String(url);
        return { ok: true, json: async () => ({ messages: [], has_more: false, page_cursor: 'p' }) };
    };
    try { await source.apiClient.chatHistory({ chatId: 2, cursor: 'c' }); }
    finally {
        globalThis.fetch = priorFetch;
        if (original) Object.defineProperty(URLSearchParams.prototype, 'size', original);
        else delete URLSearchParams.prototype.size;
    }
    assert.match(requested, /chat_id=2/);
    assert.match(requested, /cursor=c/);
});


for (const hydrated of [false, true]) test(`an unavailable archive keeps recent messages and a fresh retry (hydrated=${hydrated})`, async (t) => {
    const partial = { messages: [row('chat:10', 'Readable recent answer', {
        history_id: undefined, history_position: undefined,
    })],
        has_more: true, page_cursor: null, next_cursor: null,
        reason_code: 'history_source_unavailable', error: 'Saved archive is unavailable',
        window: { complete: false, truncated_by: ['history_source_unavailable'] } };
    const f = fixture(t, page([row('chat:10', 'Readable recent answer')], 'initial', 'older'));
    if (hydrated) await f.refresh();
    await f.refresh(partial);
    const recent = f.bubbles().find(node => node.innerHTML.includes('Readable recent answer'));
    assert.ok(recent, 'available dialogue stays readable despite the source gap');
    assert.equal(f.bubbles().length, 1, 'a temporary missing boundary does not duplicate a retained row');
    const controls = f.messages.querySelector('.chat-load-older');
    const button = controls.querySelector('.chat-load-older-btn');
    assert.equal(button.textContent, 'Retry loading messages');
    assert.notEqual((f.messages.parentNode.querySelector('.chat-panel-statusbar').querySelector('.chat-load-older-note') || f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note')).textContent, 'Some saved history is not loaded. Shown messages may have gaps.');
    f.respond(page([row('chat:10', 'Readable recent answer')], 'real-page', 'real-older'));
    await f.clickOlder();
    assert.equal(button.textContent, 'Load more history');
    assert.deepEqual(f.bubbles().filter(node => node.dataset.historyId === 'chat:10'), [recent]);
    assert.equal(f.calls.at(-1), null, 'retry captures a fresh real boundary rather than inventing a cursor');
});

for (const phase of ['unknown', 'done', 'error', 'warn', 'cancelled']) {
    test(`compact historical terminal preserves the projected outcome: ${phase}`, async (t) => {
        // This flat DOM fixture drops nested text; observe the details markup it receives.
        const markup = [], html = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
        Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...html, set(value) {
            markup.push(String(value)); html.set.call(this, value);
        } });
        t.after(() => Object.defineProperty(ElementStub.prototype, 'innerHTML', html));
        const TASK = `projected-${phase}`;
        const rationale = 'Verification is unfinished; the complete cause remains available.';
        const status = phase === 'cancelled' ? 'cancelled' : 'completed';
        const axes = { lifecycle: { status }, execution: { status: 'ok' },
            objective: { status: phase === 'error' ? 'fail' : phase === 'warn' ? 'degraded' : 'pass' } };
        if (phase === 'error') {
            axes.execution.task_completion = { action: 'stop', rationale, source: 'finish_task' };
            axes.objective.source = 'task_completion';
        }
        const observed = { task_id: TASK, role: 'assistant', text: 'The complete selected answer.', ts: TS,
            _is_direct_chat: true, ...(phase === 'unknown' ? { is_progress: true }
                : { task_terminal_status: status, outcome_axes: axes }) };
        const rows = phase === 'unknown' ? [observed] : [observed,
            { task_id: TASK, role: 'system', system_type: 'task_summary', text: '', ts: TS,
                _is_direct_chat: true, task_terminal_status: status, outcome_axes: axes,
                tool_calls: 1, completion_tool_calls: 1, tool_errors: 0 },
            { task_id: TASK, role: 'system', system_type: 'task_summary', text: '', ts: TS,
                summary_kind: 'terminal_root_projection', historical_terminal: {
                    status, phase, ts: TS, provenance: 'canonical_task_result_after_finalization',
                } },
        ];
        const f = fixture(t, page(rows));
        await f.refresh();
        for (const reconnect of [false, true]) {
            if (reconnect) { f.reconnect(); await new Promise(setImmediate); }
            const card = walkCard(f.messages, TASK);
            if (phase === 'done') {
                assert.equal(card, null, 'successful completion-only work remains a receipt without a card');
                continue;
            }
            assert.ok(card, 'a failed objective must not disappear merely because execution completed');
            const chip = card.querySelector('[data-live-phase]');
            if (phase === 'unknown') {
                assert.notEqual(card.dataset.finished, '1', 'missing terminal evidence cannot create completion');
                assert.equal(chip.hidden, true, 'unknown history cannot claim a terminal status');
            } else {
                assert.equal(card.dataset.finished, '1');
                assert.equal(chip.dataset.phase, phase);
            }
            assert.equal(card.querySelector('[data-cancel-run]'), null, 'history does not restore Stop authority');
            if (phase === 'error') assert.ok(markup.some(html => html.includes(rationale)),
                'the existing full cause survives in the card details');
        }
        assert.ok(f.calls.length >= 2, 'the real reconnect handler refetched history');
        assert.equal(rows.at(-1).historical_terminal?.status || status, status, 'raw lifecycle is unchanged');
    });
}
