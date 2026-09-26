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

function makeInstance({ details = {}, calls = [], state = { census: null } } = {}) {
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
        return { ok: true, json: async () => state.census || { active_direct_turns: [] } };
    });
    const handlers = new Map();
    let generation = 0;
    let inst = null;
    const instance = createChatInstance({
        ws: {
            on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true,
            send: () => ({ status: 'sent', clientMessageId: 'cm-send' }),
        },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: {
            begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true,
            // The page-wide sequencer fans a fetched census into the instance.
            apply: (request, data) => { inst?.hydrateStateSnapshot(data, request.requestedAt); },
        },
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
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

for (const kind of ['', undefined, 'direct_chat', 'managed_task', 'future_kind']) {
    test(`a complete census concludes a '${kind}' entry; an incomplete one concludes nothing`, () => {
        const seed = [{ activity_id: 'seeded', chat_id: 1, kind, phase: 'thinking' }];
        for (const complete of [true, false]) {
            const fx = makeInstance();
            try {
                fx.census(seed, true);
                assert.equal(fx.status(), kind === 'managed_task' ? 'Working...' : 'Thinking...');
                fx.census([], complete);
                assert.equal(fx.status(), complete ? 'Online' : (kind === 'managed_task' ? 'Working...' : 'Thinking...'));
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
