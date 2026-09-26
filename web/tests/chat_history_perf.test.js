import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const page = (messages, cursor = 'recent', next = null) => ({
    messages, page_cursor: cursor, next_cursor: next, has_more: next !== null,
    window: { complete: next === null, truncated_by: next ? ['page'] : [] },
});
const progress = index => ({
    role: 'assistant', text: `Inspect source ${index}`, is_progress: true, task_id: 'busy-task',
    ts: new Date(Date.UTC(2026, 8, 12) + index * 1000).toISOString(),
    history_id: `progress:${index}`, history_position: { source: 'progress:r1', offset: index },
});
const message = index => ({
    role: 'assistant', text: `Message ${index}`, ts: '2026-09-11T12:00:00Z',
    history_id: `chat:${index}`, history_position: { source: 'chat:r1', offset: index },
});

function fixture(t, response, routes = new Map()) {
    const handlers = new Map();
    const counters = { timelineParses: 0, historyScans: 0 };
    const html = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
    Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...html, set(value) {
        if (/^\s*<div\s+class="chat-live-line\s/.test(String(value))) counters.timelineParses += 1;
        html.set.call(this, value);
    } });
    const { prior, mount } = installDom(async url => {
        const cursor = new URL(String(url), 'http://local').searchParams.get('cursor');
        return { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? (cursor ? routes.get(cursor) : response) : { active_direct_turns: [] } };
    });
    const instance = createChatInstance({
        ws: { on(type, handler) { handlers.set(type, handler); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
    });
    const messages = document.byId.get('chat-messages');
    const query = messages.querySelectorAll.bind(messages);
    messages.querySelectorAll = selector => {
        if (selector === '[data-history-id]') counters.historyScans += 1;
        return query(selector);
    };
    t.after(() => {
        instance.destroy();
        restoreDom(prior);
        Object.defineProperty(ElementStub.prototype, 'innerHTML', html);
    });
    return {
        instance, messages, counters,
        emit: row => handlers.get('chat')({ chat_id: 2, ...row }),
        async older() {
            const button = messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
            for (const handler of button.listeners.get('click')) await handler({ target: button });
        },
    };
}

for (const count of [100, 500, 1000]) {
    test(`replaying ${count} rows builds each timeline item once and keeps live append incremental`, async t => {
        const f = fixture(t, page(Array.from({ length: count }, (_, index) => progress(index + 1))));
        assert.equal((await f.instance.refreshHistory({ revision: 1 })).painted, true);
        assert.equal(f.counters.timelineParses, count);
        const card = walkCard(f.messages, 'busy-task');
        const timeline = card.querySelector('.chat-live-timeline');
        assert.equal(timeline.children.length, count);
        const first = timeline.firstElementChild;
        f.counters.timelineParses = 0;
        await f.instance.refreshHistory({ revision: 2 });
        assert.equal(f.counters.timelineParses, 0, 'unchanged replay does not rebuild timeline markup');
        f.emit({ ...progress(count + 1), content: `A new live update ${count + 1}` });
        assert.equal(f.counters.timelineParses, 1, 'one live update appends one item after the batch flush');
        assert.equal(timeline.firstElementChild, first);
        assert.equal(timeline.children.length, count + 1);
    });
}

test('idle scrolling does not rescan retained history or repaint pager controls', async t => {
    const f = fixture(t, page(Array.from({ length: 1000 }, (_, index) => message(index))));
    await f.instance.refreshHistory({ revision: 1 });
    f.messages.scrollTop = 400;
    f.counters.historyScans = 0;
    const button = f.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    const text = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'textContent');
    let controlWrites = 0;
    Object.defineProperty(button, 'textContent', { ...text, set(value) {
        controlWrites += 1;
        text.set.call(this, value);
    } });
    for (let index = 0; index < 20; index += 1) {
        for (const handler of f.messages.listeners.get('scroll')) handler({ target: f.messages });
    }
    assert.equal(f.counters.historyScans, 0);
    assert.equal(controlWrites, 0);
});

test('evicting an offscreen page removes all its task rows before rendering that card', async t => {
    const count = 100;
    const routes = new Map([
        ['before:1', page(Array.from({ length: count }, (_, index) => progress(index + 1)), 'page:1', 'before:2')],
        ...[2, 3, 4].map(index => [`before:${index}`, page([message(index)], `page:${index}`,
            index < 4 ? `before:${index + 1}` : null)]),
    ]);
    const f = fixture(t, page([message(0)], 'recent', 'before:1'), routes);
    await f.instance.refreshHistory({ revision: 1 });
    await f.older();
    const card = walkCard(f.messages, 'busy-task');
    const timeline = card.querySelector('.chat-live-timeline');
    assert.equal(timeline.children.length, count);
    // The page is outside the reading viewport, so it is eligible for release.
    const offscreen = node => {
        node.getBoundingClientRect = () => ({ top: -10000, bottom: -9990, width: 100, height: 10 });
        for (const child of node.children) offscreen(child);
    };
    offscreen(card);
    f.messages.scrollTop = 100;
    await f.older();
    await f.older();
    f.counters.timelineParses = 0;
    await f.older();
    assert.equal(f.counters.timelineParses, 0, 'removed rows are never parsed again during page release');
    assert.equal(timeline.children.length, 0);
    assert.equal(walkCard(f.messages, 'busy-task'), null);
});
