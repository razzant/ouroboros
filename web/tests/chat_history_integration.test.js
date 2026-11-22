import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, ElementStub, walkCard } from './chat_dom_fixture.js';
import { PREVIEW_KEY } from '../modules/chat_preview.js';

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

// `seed` runs on the fresh document before the room exists (this origin's saved preview); a response may be a promise
// the test settles, or `{ httpStatus, body }` for a refused read.
function fixture(t, initial = page([]), fetchPage = null, { seed = null } = {}) {
    let response = initial, revision = 0;
    const calls = [], handlers = new Map();
    const { prior, mount } = installDom(async (url, init) => {
        if (String(url).startsWith('/api/chat/history')) {
            const cursor = new URL(String(url), 'http://local').searchParams.get('cursor');
            calls.push(cursor);
            const data = cursor && fetchPage ? await fetchPage(cursor, init) : await response;
            if (data?.httpStatus) return { ok: false, status: data.httpStatus, json: async () => data.body };
            return { ok: true, json: async () => data };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    seed?.();
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
        emitLog: (data) => handlers.get('log')({ chat_id: 2, data }),
        reconnect: (flags = {}) => handlers.get('open')({ previouslyConnected: true, ...flags }),
        disconnect: () => handlers.get('close')(),
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

// This room's rows in this origin's saved preview (chat_preview.js): Project chat 2, no appearance scope.
const PREVIEW_ROOM = JSON.stringify(['', [2, '']]);
const savedTexts = () => (JSON.parse(localStorage.getItem(PREVIEW_KEY) || '{"rooms":[]}').rooms
    .find((entry) => entry.key === PREVIEW_ROOM)?.rows || []).map((item) => item.text);
const settle = async (check = () => false) => {
    for (let n = 0; n < 80 && !check(); n += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

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
    assert.equal(card.querySelector('[data-live-phase]').textContent, 'Stopped');
    const mark = card.querySelector('[data-live-phase]').firstElementChild;
    assert.deepEqual([mark.dataset.shape, mark.dataset.hue], ['dot', 'neutral'], 'Stopped is a neutral dot, nothing moves');
    assert.equal(card.dataset.finished, '1');
    assert.equal(card.dataset.ts, String(Date.parse(progress.ts)));
    assert.ok(markup.some(value => value.includes(text) && value.includes('chat-live-line')), 'the retained source row is rendered');
    assert.ok(f.messages.children.indexOf(card) < f.messages.children.findIndex(node => node.dataset.historyId === 'chat:12'));
    await f.refresh();
    assert.equal(walkCard(f.messages, progress.task_id), card);
    assert.equal(card.querySelector('[data-live-phase]').textContent, 'Stopped', 'replay without a terminal row keeps the known census outcome');
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

test('a Project final whose page was released mounts again with its Project reference; a mounted one absorbs a retry', async (t) => {
    offScreen(t);
    const final = row('chat:800', 'Launch › Ship · Completed', { role: 'system', system_type: 'project_completion_summary',
        completion_answer: 'Done: PR #7 merged.', project_id: 'launch', project_name: 'Launch', task_id: 'final-task',
        delivery_id: 'completion:final-task', ts: '2026-09-12T11:00:00.000Z' });
    const pages = [page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1'), page([final], 'page:1', 'before:2'),
        ...[2, 3, 4].map(index => page([row(`chat:${500 - index}`, `Older ${index}`)], `page:${index}`, index < 4 ? `before:${index + 1}` : null))];
    const f = fixture(t, pages[0], (cursor) => pages[Number(cursor.split(':')[1])]);
    const finals = () => f.bubbles().filter(node => node.dataset.deliveryId === final.delivery_id);
    await f.refresh();
    await scrollEdge(f);
    assert.equal(finals().length, 1);
    for (let n = 0; n < 3; n += 1) await scrollEdge(f);
    assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3', 'before:4']);
    assert.equal(finals().length, 0, 'four older pages released the page holding the final');
    globalThis.document.byId.get('chat-scroll-bottom').click();
    for (let n = 0; n < 20; n += 1) await new Promise(resolve => setTimeout(resolve, 0));
    await scrollEdge(f);
    assert.deepEqual(f.calls.slice(5), [null, 'before:1'], 'the return reads the final page back');
    const [again] = finals();
    assert.ok(again, 'the read-back page mounts its final again');
    assert.match(again.innerHTML, /Done: PR #7 merged\./);
    const actions = again.querySelector('.system-message-actions');
    assert.equal(actions.children.length, 1);
    assert.equal(actions.children[0].dataset.intent, 'open-project');
    // A transport retry of the mounted delivery is the same bubble at its original time.
    f.emit({ ...final, content: final.text, history_id: undefined, ts: '2026-09-12T11:05:00.000Z' });
    assert.deepEqual(finals(), [again]);
    assert.equal(again.dataset.ts, String(Date.parse(final.ts)));
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
    assert.deepEqual(f.calls.slice(4), [null], 'a reader following the newest message re-anchors at that read, read once');
    await f.clickOlder();
    assert.equal(f.calls.at(-1), 'before:700', 'the re-anchored chain goes on straight below the read it started from');
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

for (const systemType of ['reminder', 'skill_notice']) for (const historyFirst of [false, true]) {
    test(`${systemType} retains one node across ${historyFirst ? 'history then live' : 'live then history'} and repeats`, async t => {
        const message = row('chat:100', 'A scheduled notice', {
            role: 'system', system_type: systemType, source: 'Ouroboros',
        });
        const live = { role: message.role, content: message.text, ts: message.ts,
            system_type: message.system_type, source: message.source };
        const f = fixture(t, page(historyFirst ? [message] : []));
        await f.refresh();
        if (!historyFirst) f.emit(live);
        assert.equal(f.bubbles().length, 1);
        const original = f.bubbles()[0];
        if (historyFirst) f.emit(live);
        else await f.refresh(page([message]));
        for (let repeat = 0; repeat < 2; repeat += 1) {
            assert.equal(f.bubbles().length, 1, 'the live frame and saved row describe one occurrence');
            assert.equal(f.bubbles()[0], original, 'joining the row preserves the original node');
            assert.equal(original.dataset.historyId, 'chat:100');
            assert.equal(original.dataset.historySource, 'chat:rotation-1');
            assert.equal(original.dataset.historyOffset, '100');
            const saved = localStorage.getItem(PREVIEW_KEY);
            assert.deepEqual(savedTexts(), [message.text], 'the saved preview holds the newest read\'s row');
            f.emit(live);
            assert.equal(localStorage.getItem(PREVIEW_KEY), saved, 'a late live copy is not saved again');
            await f.refresh(page([message]));
        }
        assert.equal(f.bubbles().length, 1);
        assert.equal(f.bubbles()[0], original);
    });
}

test('repeated physical history identity updates the routing annotation on the same user bubble', async (t) => {
    const message = row('chat:200', 'Please investigate this', {
        role: 'user', client_message_id: 'owner-message', chat_annotation: { status: 'pending' },
        source: 'web', sender_session_id: 'another-tab',
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
    const annotationText = note.textContent, metadata = { ...bubble.dataset };
    const saved = localStorage.getItem(PREVIEW_KEY);
    f.emit({ role: message.role, content: message.text, ts: message.ts,
        client_message_id: message.client_message_id, source: message.source,
        sender_session_id: message.sender_session_id });
    assert.equal(f.bubbles().length, 1, 'a remote owner echo does not append a second bubble');
    assert.equal(f.bubbles()[0], bubble);
    assert.equal(bubble.querySelector('.msg-routing-annotation'), note);
    assert.equal(note.textContent, annotationText, 'the live echo cannot clear the saved routing receipt');
    assert.deepEqual(bubble.dataset, metadata);
    assert.equal(localStorage.getItem(PREVIEW_KEY), saved);
});

test('a live echo preserves the projected saved-context note and origin metadata', async t => {
    const origin = row('chat:210', 'Original request', {
        role: 'user', source: 'web', sender_session_id: 'another-tab',
        history_id: undefined, history_position: undefined,
        origin_projected: true, origin_id: 'binding-source-ref',
    });
    const f = fixture(t, page([origin]));
    await f.refresh();
    assert.equal(f.bubbles().length, 1);
    const original = f.bubbles()[0], note = original.querySelector('.saved-project-context');
    assert.ok(note);
    assert.equal(original.dataset.originId, origin.origin_id);
    assert.equal(original.dataset.originProjected, '1');
    const metadata = { ...original.dataset }, noteText = note.textContent;
    const saved = localStorage.getItem(PREVIEW_KEY);
    f.emit({ role: origin.role, content: origin.text, ts: origin.ts,
        source: origin.source, sender_session_id: origin.sender_session_id });
    assert.equal(f.bubbles().length, 1);
    assert.equal(f.bubbles()[0], original);
    assert.equal(original.querySelector('.saved-project-context'), note);
    assert.equal(note.textContent, noteText);
    assert.deepEqual(original.dataset, metadata);
    assert.equal(localStorage.getItem(PREVIEW_KEY), saved);
});

test('an opening with its stored address keeps one bubble when the original arrives live', async t => {
    const address = `row:2@${TS}#0123456789ab`;
    const origin = row('chat:220', 'Original request', {
        role: 'user', source: 'web', sender_session_id: 'another-tab', client_message_id: 'owner-opening',
        history_id: undefined, history_position: undefined,
        origin_projected: true, origin_id: 'binding-source-ref', row_address: address,
    });
    const f = fixture(t, page([origin]));
    await f.refresh();
    const original = f.bubbles()[0];
    const note = original.querySelector('.saved-project-context');
    assert.equal(original.dataset.rowAddress, address);
    assert.ok(note);
    f.emit({ role: 'user', content: origin.text, ts: origin.ts, source: origin.source,
        sender_session_id: origin.sender_session_id, client_message_id: origin.client_message_id,
        row_address: address });
    assert.deepEqual(f.bubbles(), [original]);
    assert.equal(original.dataset.originId, origin.origin_id);
    assert.equal(original.querySelector('.saved-project-context'), note);
});

test('a retained card input keeps a description that does not claim lost submitted words', async t => {
    const origin = row('chat:240', 'First note\n2', {
        role: 'user', client_message_id: 'card-opening', source: 'web', sender_session_id: 'another-tab',
        history_id: undefined, history_position: undefined,
        origin_projected: true, origin_id: 'card-opening-ref', opening_input_kind: 'owner_card_input',
    });
    const f = fixture(t, page([origin]));
    await f.refresh();
    const description = f.bubbles()[0].getAttribute('aria-description');
    assert.ok(description, 'the retained card input keeps its accessible description');
    assert.doesNotMatch(description, /unavailable/i, 'the kind alone never says the submitted words were lost');
    assert.match(description, /asking task’s text/, 'the description still names whose text a label is');
});

test('a history row that names its origin hands that identity to the live echo it adopts', async t => {
    const address = `row:2@${TS}#0123456789ab`;
    const f = fixture(t, page([]));
    await f.refresh();
    f.emit({ role: 'user', content: 'Original request', ts: TS, source: 'web',
        sender_session_id: 'another-tab', client_message_id: 'owner-opening' });
    assert.equal(f.bubbles().length, 1);
    const physical = row('chat:230', 'Original request', {
        role: 'user', client_message_id: 'owner-opening', origin_id: 'binding-source-ref', row_address: address,
    });
    await f.refresh(page([physical]));
    assert.equal(f.bubbles().length, 1, 'the physical row adopts the mounted live echo');
    const bubble = f.bubbles()[0];
    assert.equal(bubble.dataset.historyId, 'chat:230');
    assert.equal(bubble.dataset.originId, 'binding-source-ref', 'the adopted echo carries the origin identity forward');
    const retained = row('chat:231', 'Original request', {
        role: 'user', source: 'web', sender_session_id: 'another-tab', client_message_id: undefined,
        history_id: undefined, history_position: undefined, origin_projected: true, origin_id: 'binding-source-ref',
    });
    await f.refresh(page([retained]));
    assert.deepEqual(f.bubbles(), [bubble], 'the retained copy reuses the bubble its origin identity names');
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

test('two physical rows with one full legacy key remain two messages after refresh and late live delivery', async (t) => {
    const metadata = { role: 'system', system_type: 'reminder', source: 'Ouroboros' };
    const rows = [row('chat:300', 'Same words', metadata), row('chat:400', 'Same words', metadata)];
    const f = fixture(t, page(rows));
    await f.refresh();
    const bubbles = f.bubbles().filter((node) => node.innerHTML.includes('Same words'));
    assert.equal(bubbles.length, 2);
    assert.deepEqual(bubbles.map((node) => node.dataset.historyId), ['chat:300', 'chat:400']);
    await f.refresh(page(rows));
    assert.deepEqual(f.bubbles().filter((node) => node.innerHTML.includes('Same words')), bubbles);
    assert.equal(bubbles[0].dataset.messageKey, bubbles[1].dataset.messageKey);
    f.emit({ ...metadata, content: 'Same words', ts: TS });
    assert.equal(f.bubbles().length, 2, 'the matching live copy cannot become a third physical row');
    assert.equal(f.bubbles()[0], bubbles[0]);
    assert.equal(f.bubbles()[1], bubbles[1]);
    assert.deepEqual(bubbles.map(node => node.dataset.historyId), ['chat:300', 'chat:400']);
});

test('standalone notice joins still distinguish source, text and timestamp', async t => {
    const message = row('chat:500', 'A scheduled notice', {
        role: 'system', system_type: 'reminder', source: 'Ouroboros',
    });
    const f = fixture(t, page([message]));
    await f.refresh();
    const original = f.bubbles()[0];
    const live = { role: message.role, content: message.text, ts: message.ts,
        system_type: message.system_type, source: message.source };
    for (const difference of [{ source: 'Calendar skill' }, { content: 'A different notice' },
        { ts: '2026-09-12T12:00:01.000Z' }]) f.emit({ ...live, ...difference });
    assert.equal(f.bubbles().length, 4, 'each changed field identifies another standalone message');
    assert.equal(new Set(f.bubbles().map(node => node.dataset.messageKey)).size, 4);
    assert.equal(f.bubbles().find(node => node.dataset.historyId === message.history_id), original);
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
    f.emit({ role: 'assistant', content: 'Retained answer 2499', ts: TS });
    assert.equal(f.bubbles().length, mounted, 'the mounted node suppresses a late frame after FIFO turnover');
    assert.equal(f.bubbles().find(node => node.dataset.historyId === 'chat:2499'), recent);
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
    f.respond(partial);
    assert.equal((await f.instance.refreshHistory({ revision: 2 })).painted, false, 'an unavailable source grants no read credit');
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

for (const [extraAxes, expected] of [[{}, 'Done'], [{ review: { status: 'degraded' } }, 'Done with warnings'],
    [{ artifacts: { status: 'missing' } }, 'Failed']]) {
    test(`history keeps the recorded ending through row and page ordering: ${expected}`, async t => {
        recordRenderedMarkup(t);
        const id = 'recovered-report';
        const outcome = { status: 'failed', terminal_origin: 'model_final',
            reason_code: 'finalization_incomplete', finalization_recovery: { answer: 'delivered' },
            outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' }, ...extraAxes } };
        const source = expected === 'Done' ? 'current_outcome' : 'task_outcome';
        // An outcome names the stored result's attempt, as the host projects it; one without an attempt is another result.
        const rich = row('chat:6', 'Recorded ending', { task_id: id, role: 'system', system_type: 'task_summary',
            task_attempt: 2, [source]: { ...outcome, task_attempt: 2, revision: TS } });
        const bare = row('progress:7', 'Last progress', { task_id: id, is_progress: true, narration: true,
            task_terminal_status: 'failed', ts: '2026-09-12T12:01:00Z' });
        const stopped = { status: 'cancelled', outcome_axes: { lifecycle: { status: 'cancelled' } }, revision: '2026-09-12T12:03:00Z' };
        const old = row('chat:3', 'Previous attempt', { task_id: id, role: 'system', system_type: 'task_summary',
            task_attempt: 1, current_outcome: { ...stopped, task_attempt: 1 } });
        const f = fixture(t, page([rich, bare], 'recent', 'older'), () => page([old], 'older'));
        const phase = () => walkCard(f.messages, id).querySelector('[data-live-phase]').textContent;
        await f.refresh();
        assert.equal(phase(), expected);
        await f.clickOlder();
        assert.equal(phase(), expected, 'an older attempt cannot replace this attempt, even when it has a later row time');
        await f.refresh(page([bare, { ...rich, current_outcome: undefined, task_outcome: outcome }]));
        assert.equal(phase(), expected);
        await f.refresh(page([{ ...old, history_id: 'chat:8', task_attempt: 3, current_outcome: { ...stopped, task_attempt: 3 } }]));
        assert.equal(phase(), 'Stopped', 'the new attempt owns its own status and causes');
    });
}

test('older history can supply the missing outcome behind a lifecycle stamp', async t => {
    recordRenderedMarkup(t);
    const id = 'older-diagnostic';
    const progress = row('progress:8', 'Preparing the report', { task_id: id, is_progress: true, narration: true,
        task_attempt: 1, task_terminal_status: 'completed' });
    const summary = row('chat:2', 'Recorded summary', { task_id: id, role: 'system', system_type: 'task_summary',
        ts: '2026-09-12T11:59:00Z', task_attempt: 1, status: 'completed', outcome_axes: { review: { status: 'degraded' } } });
    const f = fixture(t, page([progress], 'recent', 'older'), () => page([summary], 'older'));
    const phase = () => walkCard(f.messages, id).querySelector('[data-live-phase]').textContent;
    await f.refresh();
    assert.equal(phase(), 'Done');
    await f.clickOlder();
    assert.equal(phase(), 'Done with warnings');
    await f.refresh(page([progress]));
    assert.equal(phase(), 'Done with warnings');
});

for (const first of ['legacy', 'receipt']) {
    test(`a retained delivery warning survives a legacy status on either history page (first ${first})`, async t => {
        recordRenderedMarkup(t);
        const id = 'export-with-delivery-warning';
        const progress = row('progress:1', 'Preparing export', { task_id: id, is_progress: true, narration: true });
        const legacy = row('chat:4', 'Completed', { role: 'system', system_type: 'task_summary', task_id: id,
            task_attempt: 1, status: 'completed', ts: '2026-09-12T12:04:00Z' });
        const frozen = { status: 'completed', phase: 'warn', task_attempt: 1, ts: TS,
            provenance: 'canonical_task_result_after_finalization' };
        const receipt = row('chat:2', '', { role: 'system', system_type: 'task_summary', task_id: id,
            summary_kind: 'terminal_root_projection', historical_terminal: frozen });
        const [recent, older] = first === 'legacy' ? [legacy, receipt] : [receipt, legacy];
        const f = fixture(t, page([progress, recent], 'recent', 'older'), () => page([older], 'older'));
        const phase = () => walkCard(f.messages, id).querySelector('[data-live-phase]').textContent;
        await f.refresh();
        assert.equal(phase(), first === 'legacy' ? 'Done' : 'Done with warnings');
        await f.clickOlder();
        assert.equal(phase(), 'Done with warnings');
        await f.refresh(page([progress, older, recent]));
        assert.equal(phase(), 'Done with warnings');
        // An older gateway filled in these default axes for the same bare status.
        const defaults = { execution: { status: 'ok' }, artifacts: { status: 'not_applicable' }, review: { status: 'skipped' } };
        await f.refresh(page([progress, { ...legacy, outcome_axes: defaults }]));
        assert.equal(phase(), 'Done with warnings');
        const late = { ...receipt, current_outcome: { task_attempt: 1, status: 'completed', outcome_phase: 'done',
            revision: '2026-09-12T12:05:00Z', outcome_axes: { review: { status: 'pass', source: 'result_review' } } } };
        await f.refresh(page([progress, late, legacy]));
        assert.equal(phase(), 'Done', 'the current revision outranks its frozen phase');
        await f.refresh(page([progress, receipt, { ...legacy, task_attempt: 2, status: 'cancelled' }]));
        assert.equal(phase(), 'Stopped', 'the newer attempt outranks the revision of the old one');
    });
}

test('a pruned summary reads its attached receipt and a hidden notice cannot settle its task', async t => {
    recordRenderedMarkup(t);
    const id = 'delivery-held';
    const progress = row('progress:1', 'Preparing export', { task_id: id, is_progress: true, narration: true });
    const notice = row('chat:2', '', { task_id: id, role: 'system', system_type: 'quiz_closed',
        quiz: { quiz_id: 'choose', state: 'closed' }, task_outcome: { status: 'completed' } });
    const f = fixture(t, page([progress, notice]));
    await f.refresh();
    assert.notEqual(walkCard(f.messages, id).dataset.finished, '1');
    const summary = row('chat:3', 'Completed', { task_id: id, role: 'system', system_type: 'task_summary', status: 'completed',
        historical_terminal: { status: 'completed', phase: 'warn', ts: TS, provenance: 'canonical_task_result_after_finalization' } });
    await f.refresh(page([progress, summary]));
    assert.equal(walkCard(f.messages, id).querySelector('[data-live-phase]').textContent, 'Done with warnings');
});

test('a sparse history refresh preserves a delivered final already seen live', async t => {
    recordRenderedMarkup(t);
    const id = 'answer-seen-live';
    const progress = row('progress:1', 'Writing the answer', { task_id: id, is_progress: true, narration: true });
    const f = fixture(t, page([progress]));
    await f.refresh();
    f.emitLog({ type: 'task_done', task_id: id, ts: TS, status: 'failed', task_attempt: 1,
        reason_code: 'finalization_incomplete', terminal_origin: 'model_final', finalization_recovery: { answer: 'delivered' },
        outcome_axes: { execution: { status: 'infra_failed' } } });
    const phase = () => walkCard(f.messages, id).querySelector('[data-live-phase]').textContent;
    assert.equal(phase(), 'Done');
    await f.refresh(page([{ ...progress, task_attempt: 1, task_terminal_status: 'failed' }]));
    assert.equal(phase(), 'Done');
});

// A receipt backs the current outcome beside it only as that same attempt's publication. Zero is an attempt like any
// other; where neither names one, the pair is the legacy whole and keeps its retained words.
for (const [name, receiptAttempt, currentAttempt, shares] of [
    ['an outcome of unknown attempt takes nothing from a numbered receipt', 1, undefined, false],
    ['an outcome of unknown attempt takes nothing from a receipt numbered zero', 0, undefined, false],
    ['an outcome of the receipt\'s own attempt keeps its retained words', 1, 1, true],
    ['zero is the same attempt as zero', 0, 0, true],
    ['a wholly unnumbered pair keeps its retained words', undefined, undefined, true],
]) test(`history joins a current outcome to its own attempt's receipt: ${name}`, async t => {
    recordRenderedMarkup(t);
    const id = 'retried-export', words = 'The first attempt closed with these words.';
    const numbered = attempt => (attempt === undefined ? {} : { task_attempt: attempt });
    const progress = row('progress:1', 'Preparing the export', { task_id: id, is_progress: true, narration: true });
    const frozen = { status: 'completed', phase: 'warn', ts: '2026-09-12T11:00:00Z', wake_summary: words,
        provenance: 'canonical_task_result_after_finalization', ...numbered(receiptAttempt) };
    const receipt = row('chat:2', '', { role: 'system', system_type: 'task_summary', task_id: id, ts: frozen.ts,
        summary_kind: 'terminal_root_projection', historical_terminal: frozen, ...numbered(receiptAttempt),
        current_outcome: { task_id: id, status: 'completed', reason_code: 'final_message', outcome_phase: 'done',
            quiet_effective: false, _result_text_present: true, revision: '2026-09-12T12:05:00Z',
            outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } }, ...numbered(currentAttempt) } });
    const f = fixture(t, page([progress, receipt]));
    await f.refresh();
    const card = walkCard(f.messages, id);
    const shown = (node) => [node.innerHTML || '', node.textContent || '', ...(node.children || []).map(shown)].join('\n');
    assert.equal(card.querySelector('[data-live-phase]').textContent, 'Done', 'the stored result settles the card either way');
    assert.equal(shown(card).includes(words), shares);
});

test('a finalizing retry shows its own early answer, and an earlier attempt\'s ending is no part of it', async t => {
    const markup = recordRenderedMarkup(t);
    const open = 'recount', settled = 'sorted-notes';
    const stop = (rationale) => ({ lifecycle: { status: 'completed' },
        execution: { status: 'ok', task_completion: { action: 'stop', rationale, source: 'finish_task' } },
        objective: { status: 'fail', source: 'task_completion' } });
    const earlier = 'The first count stopped before the east bed.', own = 'The notes could not be sorted by date.';
    // The host stamps the stored result's early answer on the rows of its attempt; these two carry no attempt of their own.
    const answered = { task_phase: 'finalizing', outcome_final: false, outcome_phase: 'done', reason_code: 'final_message',
        rounds: 17, outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } } };
    const f = fixture(t, page([
        row('progress:1', 'Counting it again', { task_id: open, is_progress: true, narration: true, ...answered }),
        row('chat:2', '', { task_id: open, role: 'system', system_type: 'task_summary', task_attempt: 1, status: 'completed',
            outcome_axes: stop(earlier), outcome_phase: 'error', outcome_final: false, rounds: 4, tool_calls: 3,
            task_phase: 'unfinished' }),
        row('chat:3', 'The east bed holds sage.', { task_id: open, task_terminal_status: 'completed', ...answered }),
        // A summary that is its own task's ending still states its cause.
        row('progress:4', 'Sorting the notes', { task_id: settled, is_progress: true, narration: true }),
        row('chat:5', '', { task_id: settled, role: 'system', system_type: 'task_summary', status: 'completed',
            task_terminal_status: 'completed', outcome_axes: stop(own) }),
    ]));
    await f.refresh();
    const retried = walkCard(f.messages, open);
    const chip = retried.querySelector('[data-live-phase]');
    const label = chip.getAttribute('aria-label') || chip.textContent;
    assert.notEqual(retried.dataset.finished, '1');
    assert.match(label, /Finalizing/);
    assert.doesNotMatch(label, /Failed/);
    assert.equal(walkCard(f.messages, settled).querySelector('[data-live-phase]').textContent, 'Failed');
    assert.ok(markup.some(html => html.includes(own)), 'a card states the cause of its own ending');
    assert.ok(!markup.some(html => html.includes(earlier)), 'another attempt\'s cause is not the open attempt\'s');
    const retriedFacts = retried.querySelector('[data-live-facts]').innerHTML;
    const retriedMeta = retried.querySelector('[data-live-meta]').innerHTML;
    assert.doesNotMatch(retriedFacts, /4 · usable solve responses/, 'the earlier attempt\'s rounds do not own the open attempt\'s card');
    assert.doesNotMatch(retriedMeta, /3 tool calls/, 'the earlier attempt\'s calls do not own the open attempt\'s card either');
    assert.match(retriedFacts, /17 · usable solve responses/, 'the open attempt\'s own stamped facts carry its counters');
});

// One saved row, one bubble: a relay's live frame names the row as history does (row_client_message_id),
// while a host reply's client_message_id names the owner's submission and keys nothing.
const savedRelay = (seq, text = 'Same words') => row(`chat:${seq}`, text, {
    task_id: 'receiver', system_type: 'relay_entry', client_message_id: `relay:${seq}:receiver`,
    row_address: `2:${TS}:${seq}`, relay: { sender: { chat_id: 1, name: 'Main' }, summary: text },
});
const liveRelay = (saved) => ({ ...saved, content: saved.text, history_id: undefined, history_position: undefined,
    client_message_id: undefined, row_client_message_id: saved.client_message_id });

for (const order of ['live, then history', 'history, then live']) {
    test(`a relay drawn live and from history is one bubble (${order})`, async (t) => {
        const f = fixture(t);
        const saved = savedRelay(144070249);
        if (order.startsWith('live')) {
            await f.refresh();
            f.emit(liveRelay(saved));
            await f.refresh(page([saved]));
        } else {
            await f.refresh(page([saved]));
            f.emit(liveRelay(saved));
        }
        assert.equal(f.bubbles().length, 1);
        assert.equal(f.bubbles()[0].dataset.historyId, saved.history_id);
        assert.equal(f.bubbles()[0].dataset.clientMessageId, saved.client_message_id);
    });
}

test('two relays with the same words to one work stay two notes, live and after history', async (t) => {
    const f = fixture(t);
    await f.refresh();
    const rows = [savedRelay(101), savedRelay(102)];
    rows.forEach(saved => f.emit(liveRelay(saved)));
    assert.equal(f.bubbles().length, 2, 'their own names keep them apart');
    await f.refresh(page(rows));
    assert.equal(f.bubbles().length, 2);
});

test('the owner echo and its history row stay one bubble keyed by his id', async (t) => {
    const f = fixture(t);
    await f.refresh();
    const saved = row('chat:201', 'owner text', { role: 'user', client_message_id: 'owner:201', source: 'web' });
    f.emit({ ...saved, content: saved.text, history_id: undefined });
    await f.refresh(page([saved]));
    assert.equal(f.bubbles().length, 1);
    assert.equal(f.bubbles()[0].dataset.clientMessageId, 'owner:201');
});

test("an owner message links its http(s) address live and from history, and Copy keeps the owner's words", async (t) => {
    const words = 'Read `https://e.com/a?b=1&c=2`, then <b>reply</b>.';
    const link = '<a class="md-link" href="https://e.com/a?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">'
        + 'https://e.com/a?b=1&amp;c=2</a>';
    const message = (bubble) => bubble.innerHTML.match(/<div class="message" data-reply-body>[\s\S]*?<\/div>/)?.[0];
    const saved = row('chat:211', words, { role: 'user', client_message_id: 'owner:211', source: 'web' });
    const f = fixture(t);
    await f.refresh();
    f.emit({ ...saved, content: words, history_id: undefined });
    const live = f.bubbles()[0];
    assert.equal(message(live), `<div class="message" data-reply-body>Read \`${link}\`, then &lt;b&gt;reply&lt;/b&gt;.</div>`);
    const copied = [];
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async (text) => copied.push(text) } });
    t.after(() => delete navigator.clipboard);
    const copy = live.children.find((node) => node.classList.contains('chat-message-copy'));
    for (const handler of copy.listeners.get('click')) await handler({ target: copy });
    assert.deepEqual(copied, [words]);
    // An earlier message with the same words arrives only from history.
    const earlier = row('chat:210', words, { role: 'user', client_message_id: 'owner:210', source: 'web',
        ts: '2026-09-12T11:00:00.000Z' });
    await f.refresh(page([earlier, saved]));
    const replayed = f.bubbles().find((bubble) => bubble.dataset.clientMessageId === 'owner:210');
    assert.equal(message(replayed), message(live), 'history paints the same linked words');
});

test('Markdown the page could not parse offers no passage, whatever the message says; an owner message still does', async (t) => {
    // Without the parser the marks show: `**dom** dom dom` puts its second "dom" at 8, a parsed page's third. Coordinates
    // saved from one reading must not be found in the other (docs/DESIGN.md "Replies").
    const f = fixture(t);
    await f.refresh();
    const mine = row('chat:231', '**dom** dom dom', { role: 'user', client_message_id: 'owner:231', source: 'web' });
    const hers = row('chat:232', '**dom** dom dom');
    for (const saved of [mine, hers]) f.emit({ ...saved, content: saved.text, history_id: undefined });
    const [owner, assistant] = f.bubbles();
    assert.match(owner.innerHTML, /<div class="message" data-reply-body>/, 'plain owner words have one reading');
    assert.doesNotMatch(assistant.innerHTML, /data-reply-body/, 'no parser here: no passage of it can be quoted');
    await f.refresh(page([mine, hers]));
    assert.doesNotMatch(f.bubbles().find((bubble) => bubble.innerHTML.includes('dom')
        && !bubble.dataset.clientMessageId).innerHTML, /data-reply-body/, 'history draws the same words the same way');
});

test("a host reply that names the owner's submission is drawn beside his bubble, not into it", async (t) => {
    const f = fixture(t);
    await f.refresh();
    const user = row('chat:301', 'owner request', { role: 'user', client_message_id: 'owner:301', source: 'web' });
    f.emit({ ...user, content: user.text, history_id: undefined });
    const reply = row('chat:302', 'Operation accepted', { role: 'system', system_type: 'command_reply',
        row_address: `2:${TS}:302` });
    f.emit({ ...reply, content: reply.text, history_id: undefined, client_message_id: 'owner:301' });
    assert.equal(f.bubbles().length, 2);
    assert.equal(f.bubbles()[1].dataset.clientMessageId, undefined);
    await f.refresh(page([user, reply]));
    assert.equal(f.bubbles().length, 2, 'history adopts the reply instead of adding a third row');
});

test('only an admitted newest read writes the saved preview; ↓\'s latest page refreshes it', async (t) => {
    offScreen(t);
    const recent = page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1');
    const older = (index) => page([row(`chat:${300 - index * 100}`, `Older ${index}`)], `page:${index}`,
        index < 3 ? `before:${index + 1}` : null);
    const f = fixture(t, recent, (cursor) => older(Number(cursor.split(':')[1])));
    await f.refresh();
    assert.deepEqual(savedTexts(), ['Newest saved message']);
    for (let n = 0; n < 3; n += 1) await scrollEdge(f);
    assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3']);
    assert.deepEqual(savedTexts(), ['Newest saved message'], 'an older page is never the room\'s newest text');
    // ↓ from older history reads the present as a `latest` page: that admitted window refreshes the preview too.
    f.respond(page([row('chat:900', 'Newest saved message'), row('chat:950', 'Arrived while reading')], 'page:recent', 'before:1'));
    globalThis.document.byId.get('chat-scroll-bottom').click();
    await settle(() => f.calls.length > 4 && savedTexts().length === 2);
    assert.deepEqual(f.calls.slice(4), [null]);
    assert.deepEqual(savedTexts(), ['Newest saved message', 'Arrived while reading']);
    const before = localStorage.getItem(PREVIEW_KEY);
    // A newest window that could not read the conversation stream keeps the record; so does a refused read.
    await f.refresh({ ...page([]), window: { complete: false, truncated_by: ['chat_source_unavailable'] } });
    assert.equal(localStorage.getItem(PREVIEW_KEY), before);
    f.respond({ httpStatus: 503, body: { messages: [], error: 'history_source_unavailable',
        reason_code: 'history_source_unavailable', has_more: true, next_cursor: null } });
    await f.instance.refreshHistory({ revision: 90 });
    assert.equal(localStorage.getItem(PREVIEW_KEY), before);
    // A bounded window that is merely incomplete is still the room's newest text.
    await f.refresh({ ...page([row('chat:960', 'Bounded newest')]), window: { complete: false, truncated_by: ['quota'] } });
    assert.deepEqual(savedTexts(), ['Bounded newest']);
});

test('a newest read answered after its room left writes nothing', async (t) => {
    const f = fixture(t, page([row('chat:100', 'Read while shown')]));
    await f.refresh();
    let release;
    f.respond(new Promise((resolve) => { release = resolve; }));
    const pending = f.instance.refreshHistory({ revision: 7 });
    await settle(() => f.calls.length === 2);
    f.instance.leave();
    release(page([row('chat:200', 'Answered after the room left')]));
    await pending;
    await settle();
    assert.deepEqual(savedTexts(), ['Read while shown'], 'a late answer is no admitted window');
    assert.equal(f.bubbles().some((node) => node.innerHTML.includes('Answered after')), false);
});

const seedPreview = (rows) => () => localStorage.setItem(PREVIEW_KEY, JSON.stringify({ v: 2, rooms: [{ key: PREVIEW_ROOM, rows }] }));
const SAVED_ROWS = [{ id: 'chat:5', role: 'user', text: 'Saved question', speaker: 'You' },
    { id: 'chat:6', role: 'assistant', text: 'Saved <b>answer</b>\nsecond line', speaker: 'Ouroboros' }];
const previewNode = (f) => f.messages.children.find((node) => node.dataset.savedPreview === 'true');

test('a warm opening paints the saved text before its first read, grants it nothing, and the read replaces it', async (t) => {
    let release;
    const f = fixture(t, new Promise((resolve) => { release = resolve; }), null, { seed: seedPreview(SAVED_ROWS) });
    await settle(() => f.calls.length === 1);
    const shown = previewNode(f);
    assert.ok(shown, 'painted while the first read is out');
    assert.equal(shown.children[0].textContent, 'Saved text preview · updating');
    assert.deepEqual(shown.children.slice(1).map((node) => [node.className, node.dataset.previewHistoryId,
        node.children.map((part) => part.textContent)]), [
        ['chat-bubble user', 'chat:5', ['You', 'Saved question']],
        ['chat-bubble assistant', 'chat:6', ['Ouroboros', 'Saved <b>answer</b>\nsecond line']]]);
    for (const selector of ['button', 'a', '[data-history-id]']) {
        assert.equal(shown.querySelectorAll(selector).length, 0, `no ${selector}: no action, link or history identity`);
    }
    assert.equal(f.instance.hasPaintedHistory(), false);
    assert.equal(f.messages.dataset.historyHydrated, undefined);
    // The reader never moved: whatever the painted text did to the layout, the first canonical read opens at the newest
    // message.
    f.messages.scrollTop = 0;
    release(page([row('chat:6', 'Canonical answer'), row('chat:7', 'Fresh answer')]));
    await settle(() => f.messages.dataset.historyHydrated === 'true');
    assert.ok(f.messages.scrollHeight > 0);
    assert.equal(f.messages.scrollTop, f.messages.scrollHeight, 'the first canonical read opens at the newest message');
    assert.equal(previewNode(f), undefined, 'the canonical page replaces it');
    assert.deepEqual(f.bubbles().map((node) => node.dataset.historyId), ['chat:6', 'chat:7']);
    assert.equal(f.instance.hasPaintedHistory(), true);
    assert.deepEqual(savedTexts(), ['Canonical answer', 'Fresh answer']);
});

for (const status of [500, 401]) test(`a warm opening whose first read fails with ${status} ${status === 401 ? 'discards the saved text at once' : 'keeps it, not current, beside Retry'}`, async (t) => {
    const f = fixture(t, { httpStatus: status, body: { error: 'Controlled history failure' } }, null,
        { seed: seedPreview(SAVED_ROWS) });
    await settle(() => f.calls.length === 1 && (status === 401 ? !previewNode(f) : /not current/.test(previewNode(f)?.children[0].textContent)));
    const retry = f.messages.querySelector('.chat-load-older')?.querySelector('.chat-load-older-btn');
    assert.equal(retry?.textContent, 'Retry loading messages');
    assert.equal(f.instance.hasPaintedHistory(), false);
    if (status === 401) {
        assert.equal(previewNode(f), undefined);
        assert.equal(localStorage.getItem(PREVIEW_KEY), null, 'a refused session keeps no conversation text');
        return;
    }
    assert.equal(previewNode(f).children[0].textContent, 'Saved text preview · not current');
    assert.deepEqual(savedTexts(), ['Saved question', 'Saved <b>answer</b>\nsecond line'], 'a transient failure erases nothing');
});

// Every history read the room makes passes one door (chat.js fetchHistory): its newest read, a page, ↓'s latest page and
// arrivals alike. Any of them refused for authentication takes this origin's saved text; a page that merely failed takes
// nothing (its Retry is the room's, the saved text is not its statement).
for (const [read, status] of [['older', 401], ['older', 403], ['latest', 401], ['older', 500], ['latest', 503]]) {
    test(`a ${read} page answered ${status} ${status < 500 ? 'erases' : 'keeps'} the saved text`, async (t) => {
        offScreen(t);
        const answer = { httpStatus: status, body: { error: 'Owner session required.' } };
        const recent = page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1');
        const older = (index) => page([row(`chat:${300 - index * 100}`, `Older ${index}`)], `page:${index}`,
            index < 3 ? `before:${index + 1}` : null);
        const f = fixture(t, recent, (cursor) => (read === 'older' ? answer : older(Number(cursor.split(':')[1]))));
        await f.refresh();
        assert.deepEqual(savedTexts(), ['Newest saved message']);
        if (read === 'older') {
            await f.clickOlder();
            assert.deepEqual(f.calls, [null, 'before:1']);
        } else {
            // Read far enough back that the present is released: ↓ then reads it as a `latest` page.
            for (let n = 0; n < 3; n += 1) await scrollEdge(f);
            f.respond(answer);
            globalThis.document.byId.get('chat-scroll-bottom').click();
            await settle(() => f.calls.length > 4);
            assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3', null]);
        }
        await settle(() => localStorage.getItem(PREVIEW_KEY) === null);
        if (status < 500) assert.equal(localStorage.getItem(PREVIEW_KEY), null, 'a refused session keeps no conversation text');
        else assert.deepEqual(savedTexts(), ['Newest saved message'], 'a failed page erases nothing');
    });
}

test('the saved preview keeps the feed\'s order: an accepted delayed arrival lands where its time puts it', async (t) => {
    // Reproduction from the R1 review (Astra): A 12:00, B 12:02, then C arrives stamped 12:01.
    const earlier = row('chat:1', 'Earlier answer', { ts: '2026-09-12T12:00:00Z' });
    const newer = row('chat:2', 'Newer question', { role: 'user', ts: '2026-09-12T12:02:00Z' });
    const delayed = row('chat:3', 'Delayed older answer', { ts: '2026-09-12T12:01:00Z' });
    const f = fixture(t, { ...page([earlier, newer]), after_cursor: 'after:recent' }, (cursor) => {
        assert.equal(cursor, 'after:recent');
        return { messages: [delayed], after_cursor: 'after:delta',
            window: { complete: true, truncated_by: [], latest_message: { history_id: 'chat:3', out_of_order: true } } };
    });
    await f.refresh();
    await f.refresh();
    assert.deepEqual(f.calls, [null, 'after:recent'], 'the arrivals read, not a newest page, brought the row');
    const saved = () => JSON.parse(localStorage.getItem(PREVIEW_KEY)).rooms[0].rows.map((item) => item.id);
    assert.deepEqual(f.bubbles().map((node) => node.dataset.historyId), ['chat:1', 'chat:3', 'chat:2']);
    assert.deepEqual(saved(), ['chat:1', 'chat:3', 'chat:2'], 'reopening must not reorder the accepted conversation');
});

test('a newest window listing a delayed row after newer ones is saved in the order the feed draws it', async (t) => {
    const rows = [row('chat:1', 'A', { ts: '2026-09-12T12:00:00Z' }), row('chat:2', 'B', { role: 'user', ts: '2026-09-12T12:02:00Z' }),
        row('chat:3', 'C, delivered late', { ts: '2026-09-12T12:01:00Z' })];
    const f = fixture(t, page(rows));
    await f.refresh();
    assert.deepEqual(f.bubbles().map((node) => node.dataset.historyId), ['chat:1', 'chat:3', 'chat:2']);
    assert.deepEqual(JSON.parse(localStorage.getItem(PREVIEW_KEY)).rooms[0].rows.map((item) => item.id), ['chat:1', 'chat:3', 'chat:2']);
});

// Equal times: the feed (insertTimelineNode) breaks the tie by stream position, comparing the source as text, so after a
// log rotation `chat:rotation-10` stands before `chat:rotation-9` however the read lists them. The saved preview must
// read the same as the canonical feed, from a newest window and from an arrival.
const sameInstant = (source, offset) => ({ ts: '2026-09-12T12:00:00.000Z', history_position: { source, offset } });

test('equal times: a newest window is saved in the order the feed draws it, by stream position', async (t) => {
    const f = fixture(t, page([row('chat:500', 'Older log, later offset', sameInstant('chat:rotation-9', 500)),
        row('chat:0', 'Newer log, same millisecond', sameInstant('chat:rotation-10', 0)),
        row('chat:20', 'Same log, earlier offset', sameInstant('chat:rotation-9', 20))]));
    await f.refresh();
    const canonical = f.bubbles().map((node) => node.dataset.historyId);
    assert.deepEqual(canonical, ['chat:0', 'chat:20', 'chat:500'], 'the feed orders the tie by position');
    assert.deepEqual(JSON.parse(localStorage.getItem(PREVIEW_KEY)).rooms[0].rows.map((item) => item.id), canonical);
});

test('equal times: an accepted arrival is saved where the feed places it', async (t) => {
    const f = fixture(t, { ...page([row('chat:1', 'Earlier', { ts: '2026-09-12T11:59:00.000Z' }),
        row('chat:500', 'Same instant, older log', sameInstant('chat:rotation-9', 500))]), after_cursor: 'after:recent' },
    () => ({ messages: [row('chat:0', 'Same instant, newer log', sameInstant('chat:rotation-10', 0))], after_cursor: 'after:delta',
        window: { complete: true, truncated_by: [], latest_message: { history_id: 'chat:0' } } }));
    await f.refresh();
    await f.refresh();
    assert.deepEqual(f.calls, [null, 'after:recent'], 'the arrivals read brought the row');
    const canonical = f.bubbles().map((node) => node.dataset.historyId);
    assert.deepEqual(canonical, ['chat:1', 'chat:0', 'chat:500']);
    assert.deepEqual(JSON.parse(localStorage.getItem(PREVIEW_KEY)).rooms[0].rows.map((item) => item.id), canonical);
});

test('a newest read refused while the Project registry cannot be read keeps the feed and says so', async (t) => {
    // Main answers 503 history_source_unavailable when its rows cannot be told from a Project's (#3067).
    const f = fixture(t, page([row('chat:10', 'Shown before the refusal')], 'initial', 'older'));
    await f.refresh();
    const shown = f.bubbles();
    f.respond({ httpStatus: 503, body: { messages: [], error: 'history_source_unavailable',
        reason_code: 'history_source_unavailable', has_more: true, next_cursor: null,
        window: { complete: false, truncated_by: ['history_source_unavailable'] } } });
    await f.instance.refreshHistory({ revision: 91 });
    await settle();
    assert.deepEqual(f.bubbles(), shown, 'the refused read removes nothing it shows');
    const notes = [...(f.messages.parentNode.querySelector('.chat-panel-statusbar')?.children || []),
        ...(f.messages.querySelector('.chat-load-older')?.children || [])].map(node => node.textContent).join(' ');
    assert.match(notes, /Some saved history could not be loaded/, 'the feed says its source is unavailable');
});

for (const failed of [false, true]) test(`a room return and its paint receipt share one ${failed ? 'failed' : 'successful'} recent read`, async (t) => {
    const f = fixture(t, page([row('chat:1', 'Before leaving')]));
    await f.refresh();
    f.instance.leave();
    f.instance.stay();
    f.respond(failed ? { httpStatus: 503, body: { error: 'unavailable' } } : page([row('chat:2', 'While away')]));
    const before = f.calls.length;
    f.reconnect({ resumed: true });
    const receipt = await f.instance.refreshHistory({ revision: 100 });
    for (let n = 0; n < 4; n += 1) await new Promise(setImmediate);
    assert.deepEqual(f.calls.slice(before), [null], 'one return, one recent read, including after a failure');
    assert.equal(receipt.painted, !failed);
    assert.equal(f.bubbles().at(-1).dataset.historyId, failed ? 'chat:1' : 'chat:2');
    assert.ok(f.bubbles().at(-1).innerHTML.includes(failed ? 'Before leaving' : 'While away'));
});

test('an actual reconnect during a recent read still owes a fresh read after that request', async (t) => {
    const f = fixture(t, page([row('chat:1', 'First')]));
    await f.refresh();
    let release;
    f.respond(new Promise(resolve => { release = resolve; }));
    const before = f.calls.length;
    const painting = f.instance.refreshHistory({ revision: 100 });
    await new Promise(setImmediate);
    assert.deepEqual(f.calls.slice(before), [null]);
    f.reconnect();
    await new Promise(setImmediate);
    release(page([row('chat:2', 'Reconnected')]));
    await painting;
    for (let n = 0; n < 4; n += 1) await new Promise(setImmediate);
    assert.deepEqual(f.calls.slice(before), [null, null], 'the true reconnect must not join a request from before it');
});

test('a live completion re-fetches saved history after initial hydration has settled', async t => {
    const f = fixture(t, page([row('chat:1', 'Already loaded')]));
    await f.refresh();
    t.mock.timers.enable({ apis: ['setTimeout'] });
    f.emit({ task_id: 'completed', role: 'assistant', is_progress: true, content: 'Working', ts: TS });
    const card = walkCard(f.messages, 'completed');
    assert.ok(card);
    assert.equal(card.dataset.finished, '0');
    const before = f.calls.length;
    f.respond(page([row('chat:2', 'Saved after completion')]));
    f.emitLog({ type: 'task_done', task_id: 'completed', status: 'completed', ts: TS });
    assert.equal(card.dataset.finished, '1');
    t.mock.timers.tick(699);
    await new Promise(setImmediate);
    assert.equal(f.calls.length, before, 'completion keeps the existing 700ms debounce');
    t.mock.timers.tick(1);
    for (let n = 0; n < 5; n += 1) await new Promise(setImmediate);
    assert.deepEqual(f.calls.slice(before), [null], 'the settled hydration promise cannot swallow a new fetch');
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:2'), 'the newly saved row reaches the room');
});

test('reconnects queued behind an older read share one covering read', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    const old = held(), fresh = held();
    f.respond(old.promise);
    const before = f.calls.length;
    const painting = f.instance.refreshHistory({ revision: 80 });
    await drain();
    f.disconnect(); f.reconnect();
    await drain();
    f.disconnect(); f.reconnect();
    await drain();
    f.respond(fresh.promise);
    // Even with an arrivals cursor available, reconnect must re-read the whole newest page.
    old.release({ ...page(presentRows(901), 'page:recent', 'before:1'), after_cursor: 'after:old' });
    await drain();
    fresh.release(page(presentRows(905), 'page:recent', 'before:1'));
    await painting;
    await drain();
    assert.deepEqual(f.calls.slice(before), [null, null], 'queued reconnects share the one covering read');
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:905'), 'the covering read reaches the room');
});

// ↓ "when the present is already loaded moves there and follows without a read" (DESIGN "History edges"), and a ↓
// whose present must be read is never dropped by another read still in flight: it waits for no unrelated read.
const held = () => { let release; const promise = new Promise(resolve => { release = resolve; }); return { promise, release }; };
const drain = async (n = 30) => { for (let i = 0; i < n; i += 1) await new Promise(resolve => setTimeout(resolve, 0)); };
const presentRows = (top) => Array.from({ length: 30 }, (_, i) => row(`chat:${top - 29 + i}`, `Present ${top - 29 + i}`));
const pressLatest = () => globalThis.document.byId.get('chat-scroll-bottom').click();
const atBottom = f => f.messages.scrollTop === f.messages.scrollHeight;

test('↓ at the loaded present follows at once while an older page is still being read', async (t) => {
    const older = held();
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'), () => older.promise);
    await f.refresh();
    readUp(f);
    void f.clickOlder();
    await drain();
    assert.deepEqual(f.calls, [null, 'before:1'], 'the older read is in flight');
    pressLatest();
    await drain();
    try {
        assert.equal(atBottom(f), true, 'the loaded present is reached before the unrelated older read answers');
        assert.deepEqual(f.calls, [null, 'before:1'], 'and without a read');
    } finally {
        older.release(page([row('chat:500', 'Older one')], 'page:1'));
        await drain();
    }
});

test('↓ at the loaded present follows at once while a newest read is in flight', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const recent = held();
    f.respond(recent.promise);
    const refreshing = f.instance.refreshHistory({ revision: 50 });
    await drain();
    pressLatest();
    await drain();
    try {
        assert.equal(atBottom(f), true, 'the loaded present is reached before the refresh answers');
    } finally {
        recent.release(page(presentRows(901), 'page:recent', 'before:1'));
        await refreshing;
        await drain();
    }
    assert.equal(atBottom(f), true, 'and it keeps following once the refresh lands');
});

test('↓ away from the present is not dropped while Load more history crosses empty pages', async (t) => {
    offScreen(t);
    const gates = new Map([['before:4', held()], ['before:5', held()]]);
    const f = fixture(t, page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1'), (cursor) => {
        const index = Number(cursor.split(':')[1]);
        const answer = index === 4 || index === 5 ? page([], `page:${index}`, `before:${index + 1}`)
            : page([row(`chat:${800 - index}`, `Older ${index}`)], `page:${index}`, index < 6 ? `before:${index + 1}` : null);
        return gates.has(cursor) ? gates.get(cursor).promise.then(() => answer) : answer;
    });
    await f.refresh();
    for (let n = 0; n < 3; n += 1) await f.clickOlder();
    assert.deepEqual(f.calls, [null, 'before:1', 'before:2', 'before:3'], 'three older pages released the newest one');
    void f.clickOlder();  // before:4 and before:5 are empty: one press keeps reading until rows land
    await drain();
    assert.deepEqual(f.calls.slice(4), ['before:4']);
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(4), ['before:4', null], '↓ stops that walk and reads the present at once');
    assert.equal(atBottom(f), true, 'and lands there before the empty page answers');
    gates.get('before:4').release();
    gates.get('before:5').release();
    await drain(60);
    assert.deepEqual(f.calls.slice(4), ['before:4', null], 'the walk it stopped asks for nothing more');
    assert.equal(atBottom(f), true);
    assert.equal(f.bubbles().filter(node => node.dataset.historyId === 'chat:900').length, 1, 'the newest message is shown');
});

// A window away from the present: three older pages released the newest one, and the reader reads up there.
async function awayRoom(t, newest = 'Newest saved message') {
    offScreen(t);
    const f = fixture(t, page([row('chat:900', newest)], 'page:recent', 'before:1'), (cursor) => {
        const index = Number(cursor.split(':')[1]);
        return page([row(`chat:${800 - index}`, `Older ${index}`)], `page:${index}`, index < 6 ? `before:${index + 1}` : null);
    });
    await f.refresh();
    for (let n = 0; n < 3; n += 1) await f.clickOlder();
    readUp(f);
    return f;
}
const latestButton = () => globalThis.document.byId.get('chat-scroll-bottom');
const busy = () => latestButton().getAttribute('aria-busy') === 'true';
const present = (...rows) => page([row('chat:900', 'Newest saved message'), ...rows], 'page:recent-2', 'before:1');

test('↓ reaches retained messages before its needed read answers and keeps loading visible', async (t) => {
    const f = await awayRoom(t);
    const input = globalThis.document.byId.get('chat-input');
    input.value = 'a draft in progress';
    const latest = held();
    f.respond(latest.promise);
    const before = f.calls.length;
    pressLatest();
    assert.equal(busy(), true, 'the press is acknowledged before any read answers');
    assert.equal(latestButton().getAttribute('aria-label'), 'Loading latest messages…');
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'one read of the present');
    assert.equal(atBottom(f), true, 'retained messages are reached before the held read answers');
    assert.equal(latestButton().classList.contains('visible'), true, 'the loading control stays visible at the retained bottom');
    latest.release(present(row('chat:901', 'Arrived while away')));
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message');
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:901'));
    assert.equal(input.value, 'a draft in progress', 'the draft is untouched');
});

test('a second press joins ↓ still reading; reading on supersedes it and its late answer moves no one', async (t) => {
    const f = await awayRoom(t);
    const latest = held();
    f.respond(latest.promise);
    const before = f.calls.length;
    pressLatest();
    await drain();
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'the second press shares the read already out');
    readUp(f);
    assert.equal(busy(), false, 'reading on ends the wait at once');
    const place = f.messages.scrollTop;
    latest.release(present(row('chat:901', 'Arrived while away')));
    await drain();
    assert.equal(f.messages.scrollTop, place, 'the late answer lands under no one');
    assert.equal(f.bubbles().some(node => node.dataset.historyId === 'chat:901'), false, 'nor replaces the chain they read');
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message', 'and leaves no failure behind');
});

test('a failed read leaves ↓ at the retained end with visible Retry; Retry is a fresh cancellable ↓', async (t) => {
    const f = await awayRoom(t);
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    const place = f.messages.scrollTop;
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true, 'the failure does not undo the immediate return');
    assert.equal(latestButton().classList.contains('visible'), true, 'Retry remains accessible at the new position');
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages', 'its Retry is where the reader is');
    const retry = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    assert.equal(retry.textContent, 'Retry loading messages', 'and the history edge offers the same Retry');
    const latest = held();
    f.respond(latest.promise);
    const before = f.calls.length;
    retry.click();
    await drain();
    assert.deepEqual([busy(), f.calls.slice(before)], [true, [null]], 'Retry starts a new ↓');
    readUp(f);
    latest.release(present(row('chat:901', 'Arrived while away')));
    await drain();
    assert.equal(f.messages.scrollTop, place, 'which a gesture still supersedes');
    f.respond(present(row('chat:902', 'Arrived later')));
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message');
});

test('↓ before the room\'s first read waits for that read, busy, and asks again only when it failed', async (t) => {
    const first = held();
    const f = fixture(t, first.promise);
    await drain(5);
    pressLatest();
    assert.equal(busy(), true, 'acknowledged before the first read answers');
    first.release(page(presentRows(900), 'page:recent', 'before:1'));
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.deepEqual(f.calls, [null], 'it shared the room\'s own first read');
});

test('↓ after the room\'s first read failed asks for it again and lands on it', async (t) => {
    const f = fixture(t, { httpStatus: 503, body: { error: 'History source unavailable' } });
    await drain();
    assert.deepEqual(f.calls, [null]);
    f.respond(page(presentRows(900), 'page:recent', 'before:1'));
    pressLatest();
    await drain();
    assert.deepEqual(f.calls, [null, null]);
    assert.equal(atBottom(f), true);
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:900'));
});

test('↓ in the first moments of reconnect reaches retained messages while sharing catch-up', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const catchUp = held();
    f.respond(catchUp.promise);
    const before = f.calls.length;
    f.reconnect();
    pressLatest();  // within the bound before the room says it is catching up
    assert.equal(busy(), true, 'what the room shows is not known to be current yet');
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'only the catch-up read');
    assert.equal(atBottom(f), true, 'the retained end is reached before catch-up answers');
    assert.equal(latestButton().classList.contains('visible'), true);
    catchUp.release(page(presentRows(905), 'page:recent', 'before:1'));
    await drain();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(atBottom(f), true);
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:905'));
});

test('a failed catch-up stays owed: each ↓ retries until current data arrives', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    const before = f.calls.length;
    f.reconnect();
    pressLatest();
    await drain();
    assert.deepEqual([f.calls.slice(before), atBottom(f), busy()], [[null], true, false],
        'the failed shared read leaves the immediate return in place');
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null, null], 'the retained end cannot answer a failed catch-up');
    assert.equal(atBottom(f), true);
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    f.respond(present(row('chat:905', 'Fresh after failure')));
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null, null, null]);
    assert.equal(atBottom(f), true);
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null, null, null], 'success alone makes the present reusable');
});

// A store upgrade (503 `migrating`) is waiting, not a failed read (DESIGN "History edges"): the room reads on by itself.
const upgrading = { httpStatus: 503, body: { error: 'Saved history is being upgraded', reason_code: 'migrating' } };

test('↓ during a store upgrade still moves at once and names no Retry; the next newest read lands', async (t) => {
    const f = await awayRoom(t);
    f.respond(upgrading);
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true, 'the immediate return stands');
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message', 'waiting is no failed read');
    assert.equal(latestButton().classList.contains('visible'), false);
    f.respond(present(row('chat:901', 'Published after the upgrade')));
    await f.instance.refreshHistory({ revision: 65 });
    await drain();
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:901'), 'the present lands under the reader');
    assert.equal(atBottom(f), true);
});

test('a reconnect into a store upgrade keeps ↓ waiting through its automatic reads; an ordinary failed catch-up names Retry', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    pressLatest();
    await drain();
    const drawn = () => [latestButton().classList.contains('visible'), latestButton().getAttribute('aria-label')];
    const quiet = [false, 'Scroll to latest message'];
    f.respond(upgrading);
    f.disconnect();
    f.reconnect();
    await drain();
    assert.deepEqual(drawn(), quiet, 'the catch-up the upgrade refused waits: no Retry at the retained end');
    const again = held();
    f.respond(again.promise);
    const before = f.calls.length;
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.deepEqual(f.calls.slice(before), [null], 'the room reads the store again by itself');
    assert.deepEqual(drawn(), quiet, 'and ↓ stays waiting while that read is out');
    again.release(present(row('chat:905', 'Published after the upgrade')));
    await drain();
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:905'), 'publication lands under the follower');
    assert.equal(atBottom(f), true);
    assert.deepEqual(drawn(), quiet);
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    f.disconnect();
    f.reconnect();
    await drain();
    assert.deepEqual(drawn(), [true, 'Retry loading latest messages'], 'an ordinary failed catch-up still names Retry');
});

test('an upgrade refusing ↓ after a partial read still waits; the room\'s later ordinary failure still names Retry', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    pressLatest();
    await drain();
    const drawn = () => [latestButton().classList.contains('visible'), latestButton().getAttribute('aria-label')];
    f.respond({ ...present(row('chat:903', 'Part of the answer')),
        window: { complete: false, truncated_by: ['quota', 'chat_projection_failed'] } });
    await f.instance.refreshHistory({ revision: 66 });
    await drain();
    assert.deepEqual(drawn(), [true, 'Retry loading latest messages'], 'the partial read is unconfirmed');
    f.respond(upgrading);
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true, 'the immediate return stands');
    assert.deepEqual(drawn(), [false, 'Scroll to latest message'], 'its read refused by the upgrade waits, whatever note the partial left');
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    await f.instance.refreshHistory({ revision: 67 });
    await drain();
    assert.deepEqual(drawn(), [true, 'Retry loading latest messages'], 'the earlier refusal does not hide this failure');
});

test('a reopened revision waits for catch-up instead of consuming its unread arrival with old hydration', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    f.instance.leave();
    f.instance.stay();
    const current = held();
    f.respond(current.promise);
    const before = f.calls.length;
    f.reconnect({ resumed: true });
    void f.instance.resume(true);
    let settled = false;
    const receipt = f.instance.refreshHistory({ revision: 1 }).then(result => { settled = true; return result; });
    await drain();
    assert.equal(settled, false, 'the already-covered revision still owes the absence, so its arrival remains pending');
    assert.deepEqual(f.calls.slice(before), [null], 'the return and receipt share catch-up');
    current.release(present(row('chat:905', 'Present after the absence')));
    assert.equal((await receipt).painted, true);
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'old hydration is replaced without a duplicate read');
});

for (const absence of ['disconnect', 'leave']) test(`↓ after ${absence}, before open, reaches retained messages while reading`, async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    if (absence === 'disconnect') f.disconnect();
    else { f.instance.leave(); f.instance.stay(); }
    const next = held();
    f.respond(next.promise);
    const before = f.calls.length;
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(busy(), true);
    assert.equal(atBottom(f), true);
    assert.equal(latestButton().classList.contains('visible'), true);
    next.release(present(row('chat:905', 'After the absence')));
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
});

test('a read begun before another disconnect cannot discharge the new catch-up', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const stale = held();
    f.respond(stale.promise);
    f.reconnect();
    await drain();
    f.disconnect();
    stale.release(page(presentRows(901), 'page:recent', 'before:1'));
    await drain();
    const fresh = held();
    f.respond(fresh.promise);
    const before = f.calls.length;
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(busy(), true);
    assert.equal(atBottom(f), true, 'retained movement cannot settle the new freshness debt');
    fresh.release(present(row('chat:905', 'Current after the second disconnect')));
    await drain();
    assert.equal(atBottom(f), true);
});

test('↓ awaiting one reconnect also waits for a later reconnect, sharing both reads', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const first = held(), second = held();
    f.respond(first.promise);
    f.reconnect();
    pressLatest();
    await drain();
    f.disconnect();
    f.respond(second.promise);
    f.reconnect();
    first.release(page(presentRows(901), 'page:recent', 'before:1'));
    await drain();
    assert.equal(atBottom(f), true, 'movement is immediate, while the second disconnect still owes loading');
    assert.equal(busy(), true);
    const before = f.calls.length;
    second.release(page(presentRows(905), 'page:recent', 'before:1'));
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.equal(f.calls.length, before, 'no duplicate latest read');
});

test('a retained follower restores its place while failed catch-up remains owed', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    globalThis.requestAnimationFrame = fn => setTimeout(fn, 0); // the saved place settles after leave, as in a browser
    await f.refresh();
    pressLatest();
    await drain();
    f.instance.leave();
    f.messages.scrollTop = 0; // a hidden feed may be clamped by the engine
    f.instance.stay();
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    f.reconnect({ resumed: true });
    f.instance.resume();
    await drain();
    assert.equal(atBottom(f), true, 'the saved follow intent is independent of freshness');
    const receipt = await f.instance.refreshHistory({ revision: 70 });
    assert.equal(receipt.painted, false, 'retained rows cannot grant fresh read credit');
    f.emit({ role: 'assistant', content: 'A live reply after reopen', ts: '2026-09-13T12:00:00Z' });
    await drain();
    assert.equal(atBottom(f), true, 'the reopened follower keeps following live content');
});

test('a room leaving after immediate ↓ takes no late rows or landing; destroyed, nothing lands', async (t) => {
    const f = await awayRoom(t);
    const latest = held();
    f.respond(latest.promise);
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    const before = f.calls.length;
    f.instance.leave();
    assert.equal(busy(), false);
    f.messages.scrollTop = 0; // the hidden feed was clamped; old ↓ must not restore it
    latest.release(present(row('chat:901', 'Arrived while away')));
    await drain();
    f.instance.stay();
    await drain();
    assert.equal(atBottom(f), false, 'the ↓ from before it left does not land');
    assert.equal(f.bubbles().some(node => node.dataset.historyId === 'chat:901'), false);
    assert.equal(f.calls.length, before, 'nor reads again on the return');
    const again = held();
    f.respond(again.promise);
    pressLatest();
    await drain();
    f.instance.destroy();
    again.release(present(row('chat:902', 'Arrived later')));
    await drain();
    assert.equal(f.bubbles().some(node => node.dataset.historyId === 'chat:902'), false);
});

test('↓ shares a read of the present already out and holds the present after it, with no read of its own', async (t) => {
    const f = await awayRoom(t);
    const recent = held();
    f.respond(recent.promise);
    const refreshing = f.instance.refreshHistory({ revision: 50 });
    await drain();
    const before = f.calls.length;
    assert.equal(f.calls.at(-1), null, 'a refresh is reading the present');
    pressLatest();
    await drain();
    assert.equal(busy(), true);
    assert.equal(atBottom(f), true, '↓ moves while the shared read is still held');
    recent.release(present(row('chat:901', 'Arrived while away')));
    await refreshing;
    await drain();
    assert.deepEqual(f.calls.slice(before), [], 'no read of its own');
    assert.equal(atBottom(f), true);
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:901'));
    readUp(f);
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [], 'that read left the window at the present');
});

test('the room\'s next newest read answers ↓\'s failed read of the present: its note and Retry name leave with it', async (t) => {
    const f = await awayRoom(t);
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    pressLatest();
    await drain();
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    f.respond(present(row('chat:901', 'Arrived while away')));
    await f.instance.refreshHistory({ revision: 60 });  // the room reads its newest window again (a reconnect, a refresh)
    await drain();
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message', 'that read answered the failure');
    const edge = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    assert.notEqual(edge.textContent, 'Retry loading messages');
    const before = f.calls.length;
    assert.equal(latestButton().classList.contains('visible'), false);
    await f.clickOlder();
    await drain();
    assert.deepEqual(f.calls.slice(before), ['before:1'], 'the following reader continues from the recovered present');
});

// Retry belongs to the read that failed (DESIGN "History edges"): only ↓'s own read of the present is retried as ↓.
const edgeButton = f => f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
const viewChanged = { httpStatus: 409, body: { error: 'history_view_changed', reason_code: 'history_view_changed' } };

test('Refresh history after a changed view reads the present again without a cursor, at the present too', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'), () => viewChanged);
    await f.refresh();
    await f.clickOlder();
    assert.equal(edgeButton(f).textContent, 'Refresh history');
    f.respond(page(presentRows(905), 'page:recent-2', 'before:2'));
    const before = f.calls.length;
    await f.clickOlder();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'the reader at the present still gets the changed view read again');
    assert.equal(edgeButton(f).textContent, 'Load more history', 'and its cursor is the new view\'s');
    assert.equal(busy(), false);
});

test('Refresh history away from the present keeps the passage being read', async (t) => {
    offScreen(t);
    let changed = false;
    const f = fixture(t, page([row('chat:900', 'Newest saved message')], 'page:recent', 'before:1'), (cursor) => {
        const index = Number(cursor.split(':')[1]);
        return changed ? viewChanged
            : page([row(`chat:${800 - index}`, `Older ${index}`)], `page:${index}`, `before:${index + 1}`);
    });
    await f.refresh();
    for (let n = 0; n < 3; n += 1) await f.clickOlder();
    readUp(f);
    changed = true;
    await f.clickOlder();
    assert.equal(edgeButton(f).textContent, 'Refresh history');
    f.respond(present(row('chat:901', 'Arrived meanwhile')));
    const before = f.calls.length;
    await f.clickOlder();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(atBottom(f), false, 'a refresh is no ↓: the reader is not taken to the present');
    assert.equal(busy(), false);
});

// Coverage in stored order on one chain (CLIENT_CONTRACT "Incomplete windows"): the bytes a read delivered.
const spans = (from, to, upper) => ({ v: 1, view: 'room', upper: { chat: upper, progress: 0 },
    spans: { chat: { from, to, chain: 'c', gaps: [] }, progress: { from: 0, to: 0, chain: 'empty', gaps: [] } } });

test('a read of the present that ↓ did not ask for is retried where the reader is', async (t) => {
    offScreen(t);
    const f = fixture(t, { ...page([row('chat:99', 'Newest then')], 'page:recent', 'before:1'), coverage: spans(80, 100, 100) },
        () => ({ ...page([row('chat:79', 'The room\'s first message')], 'page:1'), coverage: spans(0, 80, 100) }));
    await f.refresh();
    await f.clickOlder();
    readUp(f);
    // The room grew far past its window while the reader read up: Load more history first reads the present.
    await f.refresh({ ...page([row('chat:199', 'Newest now')], 'page:recent-2', 'before:2'), coverage: spans(180, 200, 200) });
    f.respond({ httpStatus: 503, body: { error: 'History source unavailable' } });
    const before = f.calls.length;
    await f.clickOlder();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(edgeButton(f).textContent, 'Retry loading messages');
    f.respond({ ...page([row('chat:199', 'Newest now')], 'page:recent-3', 'before:3'), coverage: spans(180, 200, 200) });
    await f.clickOlder();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null, null], 'Retry reads that present again');
    assert.equal(atBottom(f), false, 'and leaves the reader where they read');
    assert.equal(busy(), false);
});

test('a read of the present that failed part-way confirms nothing: ↓ reads, names Retry, and a bounded page answers', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const partial = { ...page(presentRows(903), 'page:recent-2', 'before:1'),
        window: { complete: false, truncated_by: ['quota', 'chat_projection_failed'] } };
    f.respond(partial);
    f.disconnect();
    f.reconnect();
    await drain();
    const before = f.calls.length;
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'the catch-up was cut short: ↓ reads the present itself');
    assert.equal(atBottom(f), true, 'partial history does not block the physical return');
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    f.respond(present(row('chat:905', 'Read whole, bounded by the page')));
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message');
});

test('a catch-up that ends partial names ↓ Retry at the retained end at once, with no read or credit of its own', async (t) => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    pressLatest();
    await drain();
    assert.equal(latestButton().classList.contains('visible'), false, 'following at the retained end');
    f.respond({ ...present(row('chat:903', 'Part of the answer')),
        window: { complete: false, truncated_by: ['quota', 'chat_projection_failed'] } });
    const before = f.calls.length;
    f.disconnect();
    f.reconnect();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], 'the catch-up read alone');
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages', 'with no press or scroll');
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(atBottom(f), true);
    assert.equal((await f.instance.refreshHistory({ revision: 95 })).painted, false, 'a partial read earns no read credit');
    f.respond(present(row('chat:905', 'Whole again')));
    f.disconnect();
    f.reconnect();
    await drain();
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message', 'a whole catch-up answers it');
    assert.equal(latestButton().classList.contains('visible'), false);
});

test('a disconnect while ↓ reads the present: ↓ waits for the reconnect\'s catch-up and lands on it', async (t) => {
    const f = await awayRoom(t);
    const own = held(), caught = held();
    f.respond(own.promise);
    pressLatest();
    await drain();
    f.disconnect();
    f.respond(caught.promise);
    f.reconnect();
    await drain();
    own.release(present(row('chat:901', 'Read before the outage')));
    await drain();
    assert.equal(busy(), true, 'a read from before the outage cannot say what it missed');
    assert.equal(atBottom(f), true, 'the return already happened, loading is still owed');
    caught.release(present(row('chat:901', 'Read before the outage'), row('chat:902', 'Arrived during it')));
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:902'));
});

test('a disconnect while ↓ reads, the socket still down: retained return stays, ↓ names Retry and reads anew', async (t) => {
    const f = await awayRoom(t);
    const own = held();
    f.respond(own.promise);
    pressLatest();
    await drain();
    f.disconnect();
    own.release(present(row('chat:901', 'Read before the outage')));
    await drain();
    assert.deepEqual([busy(), atBottom(f)], [false, true]);
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    f.respond(present(row('chat:902', 'Read after it')));
    const before = f.calls.length;
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null]);
    assert.equal(atBottom(f), true);
});

// ↓ gives up its own read when the room's read it shares fails, ordinarily or refused by an upgrade. Partial or
// source-refused 200s in between leave the present unconfirmed; the read that confirms it restores a follower's present
// chain, and a reader who moved on keeps place and chain.
const failedRead = { httpStatus: 503, body: { error: 'History source unavailable' } };
const cutShort = { ...present(row('chat:903', 'Part of the answer')),
    window: { complete: false, truncated_by: ['quota', 'chat_projection_failed'] } };
const sourceRefused = { messages: [row('chat:900', 'Newest saved message')], has_more: true, page_cursor: null, next_cursor: null,
    reason_code: 'history_source_unavailable', window: { complete: false, truncated_by: ['history_source_unavailable'] } };
for (const [title, failure, between, readOn] of [
    ['↓ sharing the room\'s own read of the present names its Retry when that read fails, with no absence owed', failedRead, [], false],
    ['↓ sharing the room\'s own read of the present that an upgrade refuses waits; the next healthy read restores the chain',
        upgrading, [], false],
    ['after ↓ shared a failed read, a partial read leaves the chain to the next whole read', failedRead, [cutShort], false],
    ['after ↓ shared a failed read, repeated partial reads leave the chain to the next whole read', failedRead, [cutShort, cutShort], false],
    ['after ↓ shared a failed read, a source-refused read leaves the chain to the next whole read', failedRead, [sourceRefused], false],
    ['a reader who moves on after ↓ shared a refused read keeps place and chain through partial and whole reads',
        upgrading, [cutShort], true],
]) test(title, async (t) => {
    const f = await awayRoom(t);
    const recent = held();
    f.respond(recent.promise);
    const refreshing = f.instance.refreshHistory({ revision: 50 });
    await drain();
    pressLatest();
    await drain();
    assert.equal(busy(), true);
    const place = f.messages.scrollTop;
    recent.release(failure);
    await refreshing;
    await drain();
    assert.deepEqual([busy(), f.messages.scrollTop], [false, place]);
    assert.equal(latestButton().getAttribute('aria-label'),
        failure === upgrading ? 'Scroll to latest message' : 'Retry loading latest messages', 'an upgrade waits, a failure names Retry');
    if (readOn) readUp(f);
    const kept = f.messages.scrollTop;
    let revision = 50;
    for (const unconfirmed of between) {
        f.respond(unconfirmed);
        await f.instance.refreshHistory({ revision: ++revision });
        await drain();
    }
    f.respond(present(row('chat:901', 'Arrived while away')));
    // An ordinary read that confirms the present: no ↓, reconnect or failure of ↓'s own.
    await f.instance.refreshHistory({ revision: ++revision });
    await drain();
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:901'));
    if (readOn) assert.equal(f.messages.scrollTop, kept, 'the reader\'s place stays');
    else assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message');
    const before = f.calls.length;
    await f.clickOlder();
    assert.deepEqual(f.calls.slice(before), [readOn ? 'before:4' : 'before:1'], readOn
        ? 'and so does their older chain' : 'Load more history goes on from the present, not the older chain');
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true, '↓ still reaches the present');
});

test('a recent read failing after immediate ↓ leaves visible Retry at the retained end', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const recent = held();
    f.respond(recent.promise);
    const refreshing = f.instance.refreshHistory({ revision: 80 });
    await drain();
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    recent.release({ httpStatus: 503, body: { error: 'History source unavailable' } });
    await refreshing;
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    const before = f.calls.length;
    f.respond(present(row('chat:905', 'After the failure')));
    pressLatest();
    await drain();
    assert.deepEqual(f.calls.slice(before), [null], '↓ retries the failed present even with retained rows');
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:905'));
});

test('partial fresh data without a disconnect keeps retained rows, names failure and grants no read credit', async t => {
    const f = await awayRoom(t);
    const retained = f.bubbles().map(node => node.dataset.historyId);
    const response = present(row('chat:905', 'Part of the answer'));
    response.window = { latest_message: { history_id: 'chat:905' }, complete: false,
        truncated_by: ['quota', 'chat_projection_failed'] };
    response.coverage = spans(500, 905, 905);
    response.coverage.spans.chat.gaps = ['projection_failed'];
    f.respond(response);
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.equal(latestButton().classList.contains('visible'), true);
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    assert.equal(f.messages.parentNode.querySelector('.chat-panel-statusbar').querySelector('.chat-history-status').hidden, false);
    assert.ok(retained.every(id => f.bubbles().some(node => node.dataset.historyId === id)), 'partial rechain must not erase the retained messages');
    const receipt = await f.instance.refreshHistory({ revision: 90 });
    assert.equal(receipt.painted, false, 'neither the retained bottom nor a partial 200 proves the current read');
});

for (const insideBox of [false, true]) test(`a shared catch-up updates rows after a new gesture (${insideBox ? 'box' : 'feed'}) without landing its old ↓`, async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    readUp(f);
    const recent = held();
    f.respond(recent.promise);
    f.reconnect();
    pressLatest();
    await drain();
    assert.equal(atBottom(f), true);
    if (insideBox) {
        const box = document.createElement('div');
        f.bubbles().at(-1).appendChild(box);
        Object.assign(box, { scrollTop: 200, scrollHeight: 1000, clientHeight: 400 });
        document.defaultView = { getComputedStyle: node => ({ overflowY: node === box ? 'auto' : 'visible' }) };
        for (const handler of f.messages.listeners.get('wheel')) handler({ type: 'wheel', deltaY: 1, target: box, timeStamp: 1000 });
    } else readUp(f);
    const place = f.messages.scrollTop;
    recent.release(page(presentRows(905), 'page:fresh', 'before:1'));
    await drain();
    assert.equal(f.messages.scrollTop, place, 'the later reading intent owns the place');
    assert.equal(busy(), false);
    assert.ok(f.bubbles().some(node => node.dataset.historyId === 'chat:905'), 'the shared read still updates history');
});

// Qualification extensions: the real instance's public consumer, with held transport.
test('physical down supersedes an arrival waiting on the same recent read', async t => {
    const f = await awayRoom(t);
    const recent = held();
    f.respond(recent.promise);
    const refreshing = f.instance.refreshHistory({ revision: 200 });
    await drain();
    const arrival = f.instance.showLatest({ arrive: true });
    await drain();
    assert.equal(busy(), true);
    assert.equal(atBottom(f), false, 'the arrival still waits for its covered target');
    const calls = f.calls.length;
    pressLatest();
    await drain();
    const moved = atBottom(f);
    recent.release(present(row('chat:901', 'New arrival')));
    await refreshing; await arrival; await drain();
    assert.equal(moved, true, 'a new physical intent reaches the retained end before the shared read answers');
    assert.equal(f.calls.length, calls, 'the new claim shares the existing read');
    assert.equal(atBottom(f), true);
});

test('a healthy reconnect after failed down leaves an accessible way to the present chain', async t => {
    const f = await awayRoom(t);
    f.respond({ httpStatus: 503, body: { error: 'Unavailable' } });
    pressLatest(); await drain();
    f.respond(present(row('chat:901', 'Fresh after reconnect')));
    f.disconnect(); f.reconnect(); await drain();
    assert.equal(latestButton().classList.contains('visible'), false, 'the healthy follower has reached the present chain');
    await f.clickOlder();
    assert.equal(f.calls.at(-1), 'before:1', 'Load more continues from that present, not the old island');
});

test('persistent partial recent and older pages remain browseable without read credit', async t => {
    const partial = { ...page(presentRows(900), 'partial:0', 'before:1'),
        window: { complete: false, latest_message: null, truncated_by: ['quota', 'chat_projection_failed'] } };
    const f = fixture(t, partial, () => ({ ...page([row('chat:700', 'Available older row')], 'partial:1', 'before:2'),
        window: { complete: false, latest_message: null, truncated_by: ['quota', 'chat_projection_failed'] } }));
    const receipt = await f.instance.refreshHistory({ revision: 300 }); await drain();
    assert.equal(receipt.painted, false, 'incomplete projection cannot certify the requested revision');
    assert.equal(edgeButton(f).textContent, 'Load more history');
    await f.clickOlder(); await drain();
    assert.equal(f.calls.at(-1), 'before:1');
    assert.ok(f.bubbles().some(n => n.dataset.historyId === 'chat:700'));
    assert.equal(edgeButton(f).textContent, 'Load more history');
});

test('partial latest exposes available rows without erasing retained pages or accumulating partial windows', async t => {
    const f = await awayRoom(t);
    const retained = f.bubbles().map(n => n.dataset.historyId);
    for (let i = 0; i < 8; i++) {
        const response = present(row(`chat:${1200 + i}`, `Available partial ${i}`));
        response.window = { complete: false, latest_message: null, truncated_by: ['chat_projection_failed'] };
        f.respond(response); pressLatest(); await drain();
        assert.ok(f.bubbles().some(n => n.dataset.historyId === `chat:${1200 + i}`), 'partial latest rows are still useful');
        assert.ok(retained.every(id => f.bubbles().some(n => n.dataset.historyId === id)));
        assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
        assert.equal(edgeButton(f).textContent, 'Load more history', 'partial latest must not turn the independent older edge into Retry');
        assert.ok(f.bubbles().length <= retained.length + response.messages.length, 'only the current partial window supplements retained history');
    }
});

test('bare close before any read fails keeps the follower quiet; its next failed read offers Retry', async t => {
    const f = fixture(t, page(presentRows(900), 'page:recent', 'before:1'));
    await f.refresh();
    pressLatest(); await drain();
    const before = f.calls.length;
    f.disconnect(); await drain();
    assert.equal(f.calls.length, before);
    assert.equal(atBottom(f), true);
    assert.equal(busy(), false);
    assert.equal(latestButton().getAttribute('aria-label'), 'Scroll to latest message');
    assert.equal(latestButton().classList.contains('visible'), false);
    f.respond({ httpStatus: 503, body: { error: 'Unavailable' } });
    f.reconnect(); await drain();
    assert.equal(latestButton().getAttribute('aria-label'), 'Retry loading latest messages');
    assert.equal(latestButton().classList.contains('visible'), true);
});

test('a healthy read after failed down keeps a newer reading action and its older cursor', async t => {
    const f = await awayRoom(t);
    f.respond({ httpStatus: 503, body: { error: 'Unavailable' } });
    pressLatest(); await drain();
    readUp(f);
    const place = f.messages.scrollTop;
    f.respond(present(row('chat:901', 'Fresh after the failed down')));
    await f.instance.refreshHistory({ revision: 400 }); await drain();
    assert.equal(f.messages.scrollTop, place);
    assert.equal(latestButton().classList.contains('visible'), true);
    const before = f.calls.length;
    await f.clickOlder();
    assert.deepEqual(f.calls.slice(before), ['before:4'], 'recovery leaves the reader\'s chain alone');
});

test('a repeated physical down returns again after passive focus scrolling without another fetch', async t => {
    const f = await awayRoom(t);
    const latest = held(); f.respond(latest.promise);
    pressLatest(); await drain();
    const before = f.calls.length;
    assert.equal(atBottom(f), true);
    f.messages.scrollTop = 0; // focus/selection autoscroll: no wheel, swipe or navigation key
    for (const handler of f.messages.listeners.get('scroll')) handler({ target: f.messages });
    assert.equal(atBottom(f), false);
    pressLatest(); await drain();
    const returned = atBottom(f);
    latest.release(present(row('chat:901', 'Fresh after repeated press')));
    await drain();
    assert.equal(returned, true, 'every physical press moves immediately');
    assert.equal(f.calls.length, before, 'both presses share the one pending fetch');
});
