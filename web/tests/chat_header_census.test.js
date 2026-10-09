// The chat header derives from the /api/state census, the owner's own
// unconfirmed sends and live cards — never from a WS `typing` frame. A typing
// frame is a submission receipt: it pulls the census at once and retires the
// linked `Sending...` once that read has answered. These cases pin that contract, including the incident
// replay (kind-less subagent typing kept "Thinking..." forever) and the #866
// shape where the same phantom entry shielded a card from durable reconcile.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// The flat fixture's querySelector does not descend; lazily inserted controls
// (the status badge, card internals) need a real descendant lookup.
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

function makeInstance({ details = {}, calls = [], state = { census: null }, chatId = 1 } = {}) {
    const env = installDom(async (url) => {
        calls.push(String(url));
        const id = String(url).split('/').at(-1);
        if (String(url).startsWith('/api/tasks/')) {
            return details[id]
                ? { ok: true, json: async () => details[id] }
                : { ok: false, status: 404, json: async () => ({ error: 'missing' }) };
        }
        // `state.census` is what a fetched /api/state answers with (null = idle);
        // `state.fail` makes the read reject like a dropped connection.
        if (state.fail) throw new Error('offline');
        if (String(url).startsWith('/api/chat/history') && state.pages) {
            const cursor = new URL(String(url), 'http://local').searchParams.get('cursor') || 'recent';
            assert.ok(state.pages.has(cursor), cursor);
            return { ok: true, json: async () => state.pages.get(cursor) };
        }
        if (String(url).startsWith('/api/chat/history') && state.history) {
            return { ok: true, json: async () => ({ messages: state.history, window: { complete: true } }) };
        }
        return { ok: true, json: async () => state.census || { active_direct_turns: [] } };
    });
    const handlers = new Map();
    let generation = 0;
    let connected = true;
    let inst = null;
    const instance = createChatInstance({
        ws: {
            on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => connected,
            send: () => ({ status: 'sent', clientMessageId: state.messageId || 'cm-send' }),
        },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: {
            begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true,
            // The page-wide sequencer fans a fetched census into the instance.
            apply: (request, data) => { inst?.hydrateStateSnapshot(data, request.requestedAt); },
        },
        chatId, idPrefix: 'chat', mountEl: env.mount,
    });
    inst = instance;
    const status = () => env.mount.querySelector('.status-badge')?.textContent;
    // Record every header write so a test can prove no transient state blinked.
    const watchStatus = () => {
        const badge = env.mount.querySelector('.status-badge');
        const seen = [];
        const desc = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'textContent');
        Object.defineProperty(badge, 'textContent', {
            configurable: true,
            get() { return desc.get.call(this); },
            set(value) { seen.push(String(value)); desc.set.call(this, value); },
        });
        return seen;
    };
    const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };
    return {
        ...env, handlers, instance, calls, state, status, watchStatus, settle,
        close() { connected = false; handlers.get('close')(); },
        open() { connected = true; handlers.get('open')({ previouslyConnected: true }); },
        // The indicator is created with createElement, so it never lands in the
        // fixture's id index — find it by its class among the message children.
        typingHidden: () => globalThis.document.byId.get('chat-messages').children
            .find((node) => String(node.className || '').includes('typing-bubble'))?.style.display === 'none',
        card: (id) => walkCard(globalThis.document.byId.get('chat-messages'), id),
        census: (rows, complete, requestedAt = Infinity) => instance.hydrateStateSnapshot({
            active_chat_activities: rows,
            active_chat_activities_complete: complete,
            supervisor_ready: true,
        }, requestedAt, ++generation),
    };
}

test('incident replay: a finished turn whose subagent typed with no kind returns to Online', async () => {
    const fx = makeInstance();
    try {
        fx.census([], true);
        assert.equal(fx.status(), 'Online');
        // Neither typing frame is liveness: the header must not move.
        fx.handlers.get('typing')({
            chat_id: 1, activity_id: 'a80d495c', kind: 'direct_chat', client_message_id: 'cm1',
        });
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'fd5c28b5', kind: '' });
        assert.equal(fx.status(), 'Online');

        for (const event of ['running', 'completed']) {
            fx.handlers.get('chat')({
                chat_id: 1, role: 'system', is_progress: true, task_id: 'fd5c28b5',
                subagent_task_id: 'fd5c28b5', parent_task_id: 'a80d495c',
                delegation_role: 'subagent', subagent_event: event,
                content: `child ${event}`, ts: `2026-09-14T10:00:0${event === 'running' ? 0 : 1}Z`,
            });
        }
        fx.handlers.get('chat')({
            chat_id: 1, task_id: 'a80d495c', role: 'assistant', system_type: 'task_summary',
            task_terminal_status: 'completed', content: 'Done', ts: '2026-09-14T10:00:02Z',
        });
        fx.census([], true, Date.now() + 1);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.status(), 'Online');
        assert.equal(fx.typingHidden(), true);
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

for (const chatId of [1, 7]) {
    const progress = (fx, id) => fx.handlers.get('chat')({ chat_id: chatId, task_id: id,
        role: 'assistant', is_progress: true, content: 'Inspecting the work.', ts: '2026-10-08T10:00:00Z' });
    const observed = fx => id => {
        const card = fx.card(id);
        return { text: card.querySelector('[data-live-phase]').textContent,
            motion: card.querySelector('[data-live-phase]').dataset.motion,
            secondary: card.querySelector('[data-live-phase-secondary]').textContent,
            secondaryMotion: card.querySelector('[data-live-phase-secondary]').dataset.motion,
            typing: card.querySelector('[data-live-typing]').style.display, finished: card.dataset.finished };
    };
    test(`R1 census-only outcome reaches the actual full card and updates at the same phase (chat ${chatId})`, () => {
        const fx = makeInstance({ chatId });
        try {
            const id = 'census-outcome', row = { activity_id: id, chat_id: chatId, kind: 'managed_task', phase: 'finalizing' };
            progress(fx, id);
            fx.census([row], true);
            assert.equal(observed(fx)(id).text, 'Finalizing…');
            const failed = { ...row, status: 'failed', root_phase_checkpoint: { post_task_synthesis: 'running' },
                outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' } } };
            fx.census([failed], true);
            assert.deepEqual(observed(fx)(id), { text: 'Failed', motion: '0', secondary: 'Finalizing…',
                secondaryMotion: '1', typing: '', finished: '0' });
            fx.census([row], true); // Missing optional metadata cannot erase already observed evidence.
            assert.equal(observed(fx)(id).text, 'Failed');
            fx.census([{ ...failed, status: 'cancelled', outcome_axes: { lifecycle: { status: 'cancelled' } } }], true);
            assert.deepEqual(observed(fx)(id), { text: 'Cancelled', motion: '0', secondary: '',
                secondaryMotion: '0', typing: 'none', finished: '1' });
            fx.census([failed], true); // The original in-flight snapshot cannot revive a concluded task.
            assert.equal(observed(fx)(id).text, 'Cancelled');
            const later = `${id}-mounted-after-census`;
            fx.census([{ ...failed, activity_id: later }], true);
            progress(fx, later);
            assert.equal(observed(fx)(later).text, 'Failed');
            assert.equal(observed(fx)(later).secondary, 'Finalizing…');
        } finally { fx.instance.destroy(); restoreDom(fx.prior); }
    });

    test(`R1 real close/open parks full cards until each receives a fresh positive activity (chat ${chatId})`, async () => {
        const fx = makeInstance({ chatId });
        try {
            const id = 'late-failure', sibling = 'independent-work', settled = 'already-done';
            progress(fx, id); progress(fx, sibling); progress(fx, settled);
            fx.handlers.get('chat')({ chat_id: chatId, task_id: id, role: 'system', system_type: 'task_summary',
                status: 'failed', task_phase: 'finalizing', content: 'Task failed.', ts: '2026-10-08T10:00:01Z' });
            fx.handlers.get('chat')({ chat_id: chatId, task_id: settled, role: 'system', system_type: 'task_summary',
                status: 'completed', content: 'Task done.', ts: '2026-10-08T10:00:02Z' });
            const failed = { activity_id: id, chat_id: chatId, kind: 'managed_task', phase: 'finalizing', status: 'failed',
                root_phase_checkpoint: { post_task_synthesis: 'running' } };
            const working = { activity_id: sibling, chat_id: chatId, kind: 'managed_task', phase: 'working' };
            fx.census([failed, working], true);
            assert.equal(observed(fx)(id).secondaryMotion, '1');
            fx.close();
            const offline = { text: 'Failed', motion: '0', secondary: 'Activity unconfirmed',
                secondaryMotion: '0', typing: 'none', finished: '0' };
            assert.deepEqual(observed(fx)(id), offline);
            fx.census([failed, working], true); // A REST answer while the socket is down cannot restore motion.
            assert.deepEqual(observed(fx)(id), offline);
            assert.equal(observed(fx)(sibling).motion, '0');
            assert.equal(observed(fx)(settled).text, 'Done');
            assert.equal(observed(fx)(settled).finished, '1');
            fx.state.fail = true; // Open is connectivity, not a successful new census/history read.
            fx.open();
            await fx.settle();
            assert.deepEqual(observed(fx)(id), offline);
            fx.census([], false);
            assert.deepEqual(observed(fx)(id), offline);
            fx.census([working], false);
            assert.equal(observed(fx)(sibling).motion, '1');
            assert.deepEqual(observed(fx)(id), offline);
            fx.census([failed], false);
            assert.equal(observed(fx)(id).text, 'Failed');
            assert.equal(observed(fx)(id).secondaryMotion, '1');
            assert.equal(observed(fx)(sibling).motion, '0', 'an omitted row is retained as unknown, not reanimated');
            assert.equal(observed(fx)(settled).text, 'Done');
        } finally { fx.instance.destroy(); restoreDom(fx.prior); }
    });
}

test('the census alone moves the header between Thinking... and Online', () => {
    const fx = makeInstance();
    try {
        fx.census([{ activity_id: 'direct-1', chat_id: 1, kind: 'direct_chat', phase: 'thinking' }], true);
        assert.equal(fx.status(), 'Thinking...');
        assert.equal(fx.typingHidden(), false);
        fx.census([], true);
        assert.equal(fx.status(), 'Online');
        assert.equal(fx.typingHidden(), true);
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

for (const withdrawBeforeMount of [false, true]) {
    test(`R2 terminal census precedes its card, withdrawal first=${withdrawBeforeMount}`, async () => {
        const fx = makeInstance();
        try {
            const id = 'early-cancel', row = { activity_id: id, chat_id: 1, kind: 'managed_task', phase: 'finalizing',
                status: 'cancelled', root_phase_checkpoint: { post_task_synthesis: 'running' } };
            fx.census([row], true);
            if (withdrawBeforeMount) fx.census([], true);
            fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', is_progress: true,
                content: 'A progress frame already in transit.', ts: '2026-10-08T10:00:00Z' });
            const card = fx.card(id);
            assert.ok(card);
            assert.equal(card.dataset.finished, '1');
            assert.equal(card.querySelector('[data-live-phase]').textContent, 'Cancelled');
            assert.equal(card.querySelector('[data-live-phase]').dataset.motion, '0');
            fx.census([row], true);
            fx.census([], true);
            await fx.settle();
            assert.equal(card.querySelector('[data-live-phase]').textContent, 'Cancelled');
            assert.equal(card.dataset.finished, '1');
            assert.equal(fx.calls.filter(url => url.endsWith(`/api/tasks/${id}`)).length, 0, 'the known outcome needs no rescue read');
            const fresh = { activity_id: 'independent', chat_id: 1, kind: 'managed_task', phase: 'working', status: 'running' };
            fx.census([fresh], true);
            fx.handlers.get('chat')({ chat_id: 1, task_id: 'independent', role: 'assistant', is_progress: true,
                content: 'A different task is running.', ts: '2026-10-08T10:00:01Z' });
            assert.equal(fx.card('independent').dataset.finished, '0');
            assert.equal(fx.card('independent').querySelector('[data-live-phase]').dataset.motion, '1');
        } finally { fx.instance.destroy(); restoreDom(fx.prior); }
    });
}

for (const firstCarrier of ['narration', 'metrics', 'tool span']) {
    test(`R3 a cached terminal keeps initial authored content after ${firstCarrier}`, () => {
        const fx = makeInstance();
        try {
            const id = 'initial-content';
            fx.census([{ activity_id: id, chat_id: 1, kind: 'managed_task', phase: 'working', status: 'cancelled' }], true);
            if (firstCarrier !== 'narration') {
                fx.handlers.get('log')({ chat_id: 1, data: { task_id: id, ts: '2026-10-08T10:00:00Z',
                    ...(firstCarrier === 'metrics' ? { type: 'task_metrics_event', tool_calls: 1 }
                        : { type: 'tool_call_started', tool: 'read_file', tool_call_id: 'initial-span' }) } });
                assert.equal(fx.card(id).querySelector('[data-live-phase]').textContent, 'Cancelled');
                assert.equal(fx.card(id).querySelector('[data-live-phase]').dataset.motion, '0');
            }
            const text = 'I found the relevant state transition.';
            fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', system_type: 'model_narration',
                is_progress: true, narration: true, content: text, ts: '2026-10-08T10:00:01Z' });
            const card = fx.card(id);
            assert.equal(card.querySelector('[data-live-title]').textContent, text);
            assert.equal(card.querySelector('[data-live-phase]').textContent, 'Cancelled');
            assert.equal(card.dataset.finished, '1');
            assert.equal(card.querySelector('[data-live-phase]').dataset.motion, '0');
            assert.ok(card.querySelector('[data-live-timeline]').children.length > 0, 'initial content reaches the timeline renderer');
            const count = card.querySelector('[data-live-count]').textContent;
            fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', is_progress: true, narration: true,
                content: 'Late progress must not replace the established finished card.', ts: '2026-10-08T10:00:02Z' });
            assert.equal(card.querySelector('[data-live-title]').textContent, text);
            assert.equal(card.querySelector('[data-live-count]').textContent, count);
            assert.equal(fx.calls.filter(url => url.endsWith(`/api/tasks/${id}`)).length, 0);
        } finally { fx.instance.destroy(); restoreDom(fx.prior); }
    });
}

for (const [status, label] of [['completed', 'Done'], ['failed', 'Failed']]) {
    test(`R3 off-screen ${status} card rematerializes with its retained narrative`, async () => {
        const id = 'evicted-root', text = 'Reading the repository before completion';
        const progress = { chat_id: 1, task_id: id, role: 'assistant', is_progress: true, narration: true,
            text, content: text, ts: '2026-10-08T10:00:00Z', history_id: 'old-progress', history_position: { source: 'progress', offset: 1 } };
        const terminal = { chat_id: 1, task_id: id, role: 'assistant', system_type: 'task_summary', status,
            task_terminal_status: status, text: 'Finished', content: 'Finished', ts: '2026-10-08T10:00:05Z',
            history_id: 'old-terminal', history_position: { source: 'chat', offset: 2 } };
        const page = (messages, cursor, next = null) => ({ messages, progress: [], page_cursor: cursor,
            next_cursor: next, has_more: Boolean(next), window: { complete: !next }, coverage: null });
        const pages = new Map([['recent', page([progress, terminal], 'recent-first')]]);
        const fx = makeInstance({ state: { pages, census: { active_chat_activities: [],
            active_chat_activities_complete: true, supervisor_ready: true } } });
        const descriptor = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
        const rect = ElementStub.prototype.getBoundingClientRect, rects = ElementStub.prototype.getClientRects, rendered = [];
        Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...descriptor,
            set(value) { rendered.push(String(value)); descriptor.set.call(this, value); } });
        const settle = async () => { for (let n = 0; n < 30; n += 1) await new Promise(resolve => setTimeout(resolve, 0)); };
        try {
            fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', is_progress: true, narration: true,
                content: text, ts: progress.ts });
            fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', system_type: 'task_summary',
                task_terminal_status: status, content: 'Finished', ts: terminal.ts });
            fx.open(); await settle();
            const original = fx.card(id);
            assert.ok(original);
            const messages = globalThis.document.byId.get('chat-messages');
            ElementStub.prototype.getBoundingClientRect = function () { return this === messages || this === fx.mount
                ? { top: 0, bottom: 800, left: 0, right: 1000, width: 1000, height: 800 }
                : { top: -5000, bottom: -4980, left: 0, right: 100, width: 100, height: 20 }; };
            ElementStub.prototype.getClientRects = function () { return [this.getBoundingClientRect()]; };
            pages.set('recent', page([{ chat_id: 1, role: 'user', text: 'A newer request', content: 'A newer request',
                ts: '2026-10-08T10:10:00Z', history_id: 'newer-owner', history_position: { source: 'chat', offset: 9 } }],
                'recent-second', 'older-first'));
            pages.set('older-first', page([progress, terminal], 'older-first'));
            fx.close(); fx.open(); await settle();
            assert.equal(fx.card(id), null, 'the real history owner evicts the finished off-screen card');
            rendered.length = 0;
            const older = messages.querySelector('.chat-load-older-btn');
            assert.equal(older.hidden, false);
            older.listeners.get('click')[0]({ target: older }); await settle();
            const restored = fx.card(id);
            assert.ok(restored, 'Load more history restores the retained work');
            assert.notEqual(restored, original);
            assert.equal(restored.querySelector('[data-live-title]').textContent, text);
            assert.equal(restored.querySelector('[data-live-count]').textContent, '2 notes');
            assert.equal(restored.querySelector('[data-live-phase]').textContent, label);
            assert.equal(restored.querySelector('[data-live-phase]').dataset.motion, '0');
            assert.equal(restored.dataset.finished, '1');
            assert.equal(restored.dataset.ts, String(Date.parse(progress.ts)));
            assert.ok(rendered.some(value => value.includes(text) && value.includes('chat-live-line')));
        } finally {
            fx.instance.destroy(); restoreDom(fx.prior);
            ElementStub.prototype.getBoundingClientRect = rect; ElementStub.prototype.getClientRects = rects;
            Object.defineProperty(ElementStub.prototype, 'innerHTML', descriptor);
        }
    });
}

test('R2 a terminal census retires only its linked Sending receipt without a typing frame', async () => {
    const fx = makeInstance();
    try {
        fx.census([], true);
        const input = globalThis.document.byId.get('chat-input');
        const send = () => globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        input.value = 'Complete this request'; send(); await fx.settle();
        assert.equal(fx.status(), 'Sending...');
        const row = { activity_id: 'completed-before-receipt', client_message_id: 'cm-send', chat_id: 1,
            kind: 'direct_chat', phase: 'thinking', status: 'completed',
            root_phase_checkpoint: { post_task_synthesis: 'completed' } };
        fx.census([row], true);
        assert.equal(fx.status(), 'Online');
        fx.state.messageId = 'second-send'; input.value = 'Another request'; send(); await fx.settle();
        fx.census([row], true);
        assert.equal(fx.status(), 'Sending...', 'an old completion cannot retire a different request');
        fx.census([{ activity_id: 'second', client_message_id: 'second-send', chat_id: 1,
            kind: 'direct_chat', phase: 'thinking' }], true);
        assert.equal(fx.status(), 'Thinking...');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

test('R2 only fresh connected child activity releases disconnect uncertainty; replay, waits and terminals stay factual', async () => {
    const fx = makeInstance();
    try {
        const root = { activity_id: 'parent', chat_id: 1, kind: 'managed_task', phase: 'working' };
        const childFrame = { chat_id: 1, task_id: 'child', subagent_task_id: 'child', root_task_id: 'parent',
            parent_task_id: 'parent', delegation_role: 'subagent', subagent_role: 'research', subagent_event: 'running',
            role: 'system', is_progress: true, content: 'Inspecting the evidence.', ts: '2026-10-08T10:00:00Z' };
        fx.handlers.get('chat')({ chat_id: 1, task_id: 'parent', role: 'assistant', is_progress: true,
            content: 'Working with a child.', ts: '2026-10-08T09:59:00Z' });
        fx.handlers.get('chat')(childFrame);
        fx.census([root], true);
        const phase = () => fx.card('child').querySelector('[data-live-phase]');
        assert.equal(phase().dataset.motion, '1');
        fx.close();
        assert.equal(phase().textContent, 'Activity unconfirmed');
        fx.handlers.get('chat')({ ...childFrame, content: 'Already buffered before the socket closed.' });
        assert.equal(phase().dataset.motion, '0');
        const historyText = 'Retained history-only child evidence';
        fx.state.history = [{ ...childFrame, subagent_event: 'progress', history_id: 'progress:r2-child', text: historyText, content: historyText,
            ts: '2026-10-08T10:00:00.500Z', is_progress: true }];
        fx.state.census = { active_chat_activities: [root], active_chat_activities_complete: true, supervisor_ready: true };
        fx.open(); await fx.instance.refreshHistory({ revision: 1 }); await fx.settle();
        const replayed = fx.card('child');
        if (replayed.dataset.expanded !== '1') replayed.querySelector('[data-live-summary-button]').listeners.get('click')[0]({});
        const rendered = node => [node.textContent, node.innerHTML, ...node.children.map(rendered)].join(' ');
        assert.match(rendered(replayed), /Retained history-only child evidence/,
            'the negative control must have admitted and rendered the historical row');
        assert.equal(phase().textContent, 'Activity unconfirmed', 'history replay and the fresh parent do not prove child activity');
        fx.handlers.get('chat')({ ...childFrame, content: 'Fresh child progress.', ts: '2026-10-08T10:00:01Z' });
        assert.equal(phase().textContent, 'Working');
        assert.equal(phase().dataset.motion, '1');
        const wait = { wait_id: 'child-wait', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota', role: 'main', model: 'test' };
        fx.handlers.get('log')({ chat_id: 1, data: { type: 'task_model_wait', chat_id: 1, task_id: 'child', ...wait } });
        fx.close(); fx.open(); await fx.settle();
        fx.handlers.get('chat')({ ...childFrame, content: 'Fresh progress with an unresolved model wait.', ts: '2026-10-08T10:00:02Z' });
        assert.equal(phase().textContent, 'Waiting for access');
        assert.equal(phase().dataset.motion, '0');
        fx.handlers.get('log')({ chat_id: 1, data: { type: 'task_model_wait', chat_id: 1, task_id: 'child',
            ...wait, revision: 2, state: 'resolved' } });
        assert.equal(phase().dataset.motion, '1');
        fx.handlers.get('chat')({ ...childFrame, subagent_event: 'completed', status: 'completed', content: 'Child result.',
            ts: '2026-10-08T10:00:03Z' });
        assert.equal(fx.card('child').dataset.finished, '1');
        fx.handlers.get('chat')({ ...childFrame, content: 'Late old progress.', ts: '2026-10-08T10:00:04Z' });
        assert.equal(fx.card('child').dataset.finished, '1');
        assert.equal(phase().dataset.motion, '0');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

test('a direct owner Pause agrees between the actual card and header, then Resume restores Thinking', () => {
    const fx = makeInstance();
    try {
        const id = 'paused-direct-turn';
        fx.handlers.get('chat')({ chat_id: 1, task_id: id, role: 'assistant', is_progress: true,
            _is_direct_chat: true, cancelable: true, content: 'Checking the saved work.', ts: '2026-09-28T06:00:00Z' });
        const row = { activity_id: id, chat_id: 1, kind: 'direct_chat' };
        fx.census([{ ...row, phase: 'thinking' }], true);
        assert.equal(fx.status(), 'Thinking...');
        fx.census([{ ...row, phase: 'budget_pausing' }], true);
        assert.equal(fx.status(), 'Pausing…');
        assert.equal(fx.card(id).querySelector('[data-live-phase]')?.textContent, 'Pausing…');
        fx.census([{ ...row, phase: 'budget_paused' }], true);
        assert.equal(fx.status(), 'Paused');
        assert.equal(fx.typingHidden(), true);
        assert.equal(fx.card(id).querySelector('[data-live-phase]')?.textContent, 'Paused');
        assert.equal(fx.card(id).querySelector('[data-live-typing]')?.style.display, 'none');
        fx.census([{ ...row, phase: 'unknown' }], false);
        assert.equal(fx.status(), 'Activity unconfirmed');
        assert.equal(fx.typingHidden(), true);
        assert.equal(fx.card(id).querySelector('[data-live-phase]')?.textContent, 'Activity unconfirmed');
        assert.equal(fx.card(id).querySelector('[data-live-typing]')?.style.display, 'none');
        fx.census([{ ...row, phase: 'budget_paused' }], true);
        assert.equal(fx.status(), 'Paused');
        fx.census([{ ...row, phase: 'thinking' }], true);
        assert.equal(fx.status(), 'Thinking...');
        assert.equal(fx.card(id).querySelector('[data-live-phase]')?.textContent, 'Thinking');
        assert.equal(fx.card(id).querySelector('[data-live-phase]')?.dataset.motion, '1');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

for (const kind of ['', undefined, 'direct_chat', 'managed_task', 'future_kind']) {
    test(`a complete census concludes a '${kind}' entry; an incomplete one concludes nothing`, () => {
        const seed = [{ activity_id: 'seeded', chat_id: 1, kind, phase: 'thinking' }];
        for (const complete of [true, false]) {
            const fx = makeInstance();
            try {
                fx.census(seed, true);
                assert.equal(fx.status(), kind === 'managed_task' ? 'Working...' : 'Thinking...');
                fx.census([], complete);
                assert.equal(fx.status(), complete ? 'Online' : 'Activity unconfirmed');
            } finally { fx.instance.destroy(); restoreDom(fx.prior); }
        }
    });
}

test('a typing frame is a receipt: Sending... steps to the census verdict, never through Online', async () => {
    const fx = makeInstance();
    try {
        fx.census([], true);
        const stateCalls = () => fx.calls.filter((url) => url.startsWith('/api/state')).length;
        const before = stateCalls();
        // An unkeyed frame carries no receipt: nothing to retire, nothing to pull.
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'anon', kind: '' });
        assert.equal(stateCalls(), before);
        assert.equal(fx.status(), 'Online');

        const seen = fx.watchStatus();
        const input = globalThis.document.byId.get('chat-input');
        input.value = 'hello';
        globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        await fx.settle();
        assert.equal(fx.status(), 'Sending...', 'the owner\'s own unconfirmed send owns the header');

        // The census answers the receipt's read with the turn listed: the header
        // moves Sending... -> Thinking... with no Online in between.
        fx.state.census = {
            active_chat_activities: [{ activity_id: 'turn-1', chat_id: 1, kind: 'direct_chat', phase: 'thinking', client_message_id: 'cm-send' }],
            active_chat_activities_complete: true, supervisor_ready: true,
        };
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'turn-1', kind: 'direct_chat', client_message_id: 'cm-send' });
        await fx.settle();
        assert.equal(fx.status(), 'Thinking...');
        assert.ok(stateCalls() > before, 'the receipt pulled the authoritative census at once');
        assert.ok(seen.includes('Sending...'), `the send was observed: ${seen.join(' > ')}`);
        assert.ok(!seen.slice(seen.indexOf('Sending...')).includes('Online'), `no Online blink: ${seen.join(' > ')}`);

        // A receipt whose census does not list the turn settles at Online.
        fx.state.census = { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true };
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'turn-1', kind: 'direct_chat', client_message_id: 'cm-send' });
        await fx.settle();
        assert.equal(fx.status(), 'Online');

        // A fresh send whose receipt's census lists nothing is retired by the
        // read's own settlement (no census row carries the cmid any more).
        input.value = 'again';
        globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        await fx.settle();
        assert.equal(fx.status(), 'Sending...');
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'turn-2', kind: 'direct_chat', client_message_id: 'cm-send' });
        await fx.settle();
        assert.equal(fx.status(), 'Online', 'the settled read retired the send the census never listed');

        // A read that fails (dropped connection) still settles the receipt.
        input.value = 'once more';
        globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        await fx.settle();
        assert.equal(fx.status(), 'Sending...');
        fx.state.fail = true;
        fx.handlers.get('typing')({ chat_id: 1, activity_id: 'turn-3', kind: 'direct_chat', client_message_id: 'cm-send' });
        await fx.settle();
        fx.state.fail = false;
        assert.equal(fx.status(), 'Online', 'a failed census read still retires the send');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

test('#866: a progress-minted card the census never lists is not shielded from durable reconcile', async () => {
    const id = 'presence-0adf5b78';
    const fx = makeInstance({ details: { [id]: { task_id: id, status: 'completed', phase: 'done', ts: '2026-09-14T10:00:00Z' } } });
    try {
        fx.handlers.get('chat')({
            chat_id: 1, task_id: id, role: 'assistant', is_progress: true,
            content: 'Working on it', ts: '2026-09-14T09:59:00Z',
        });
        assert.ok(fx.card(id), 'the progress row minted a foreground card');
        assert.equal(fx.card(id).dataset.finished, '0');
        // On the base tree this kind-less frame entered the live-set and
        // shielded the card from the durable reconcile; it must be inert.
        fx.handlers.get('typing')({ chat_id: 1, activity_id: id, kind: '' });
        fx.census([], true, Date.now() + 1_000);
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.card(id).dataset.finished, '1', 'the durable task detail settled the orphan card');
        assert.equal(fx.status(), 'Online');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});

test('a progress-only root the census omits loses Stop authority at once, then settles from durable detail', async () => {
    const id = 'progress-only-root';
    const fx = makeInstance({ details: { [id]: { task_id: id, status: 'completed', phase: 'done', ts: '2026-09-14T10:00:00Z' } } });
    try {
        fx.handlers.get('chat')({
            chat_id: 1, task_id: id, role: 'assistant', is_progress: true, cancelable: true,
            content: 'Working.', ts: '2026-09-14T09:59:00Z',
        });
        assert.ok(fx.card(id).querySelector('[data-cancel-run]'), 'the cancelable progress row granted Stop');
        assert.equal(fx.status(), 'Working...');
        // The queue census omits the root: Stop has no row to target, and the
        // card is handed to the durable-detail read (one /api/tasks call).
        fx.census([], true, Date.now() + 1_000);
        assert.equal(fx.card(id).querySelector('[data-cancel-run]'), null, 'census omission revoked Stop before the detail settled');
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.filter((url) => url.endsWith(`/api/tasks/${id}`)).length, 1);
        assert.equal(fx.card(id).dataset.finished, '1');
        assert.equal(fx.status(), 'Online');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});
