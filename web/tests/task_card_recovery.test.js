import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const chatId = 77;
const taskId = 'context-analysis';
const progress = {
    task_id: taskId, chat_id: chatId, role: 'assistant', is_progress: true,
    text: 'Inspecting the saved context', suggested_name: 'Context analysis',
    ts: '2026-09-15T10:00:00Z', history_id: 'progress:1',
};
const cost = (amount, final = false) => ({
    cost_accounting_status: 'available', accounted_upper_bound_usd_with_children: amount,
    cost_final: final, cost_with_children_partial: !final,
});
const reference = (amount, final = false) => ({
    task_id: taskId, chat_id: chatId, presentation_owner_task_id: taskId,
    role: 'system', is_progress: true, system_type: 'review_reference',
    surface: 'task_acceptance', state_revision: 'a'.repeat(64),
    ts: '2026-09-15T10:00:01Z', history_id: 'progress:2', ...cost(amount, final),
});

function fixture(rows = [progress]) {
    const data = { rows, detail: { task_id: taskId, status: 'running',
        review_projection: { panels: [{ panel_id: 'accept', surface: 'task_acceptance',
            aggregate_signal: 'PASS', actors: [] }] },
    } };
    const calls = [];
    const snapshot = {
        supervisor_ready: true, active_chat_activities_complete: true,
        active_chat_activities: [{ activity_id: taskId, chat_id: chatId,
            project_id: 'context-project', kind: 'direct_chat', phase: 'thinking' }],
    };
    const env = installDom(async (url) => {
        calls.push(String(url));
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: data.rows }) };
        }
        if (String(url).startsWith('/api/tasks/')) {
            if (data.detailError) throw new Error('task detail unavailable');
            return { ok: true, json: async () => data.detail };
        }
        return { ok: true, json: async () => snapshot };
    });
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set([chatId]), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId, idPrefix: 'chat', mountEl: env.mount, asPanel: true,
    });
    let revision = 0;
    return {
        instance, handlers, calls, data, snapshot,
        card: (id = taskId) => walkCard(globalThis.document.byId.get('chat-messages'), id),
        async replay() {
            await instance.refreshHistory({ revision: ++revision });
            await new Promise((resolve) => setImmediate(resolve));
        },
        hydrate: () => instance.hydrateStateSnapshot(snapshot),
        send(row, channel = 'chat') {
            handlers.get(channel)(channel === 'log'
                ? { chat_id: chatId, data: { ...row, type: row.system_type }, ts: row.ts }
                : { ...row, content: row.text || '' });
        },
        destroy() { instance.destroy(); restoreDom(env.prior); },
    };
}

const anatomy = (card) => ({
    phase: card.querySelector('[data-live-phase]').textContent,
    hidden: card.querySelector('[data-live-phase]').hidden,
    typing: card.querySelector('[data-live-typing]').style.display,
    notes: card.querySelector('[data-live-count]').textContent,
    activity: card.querySelector('[data-live-activity]').textContent,
});

test('a bound direct activity restores its Project card and survives progress/history replay', async () => {
    const fx = fixture();
    try {
        await fx.replay();
        assert.equal(fx.card().querySelector('[data-live-phase]').hidden, true,
            'history alone does not attest current activity');
        fx.hydrate();
        assert.equal(fx.card().querySelector('[data-live-phase]').hidden, false);
        fx.send(progress);
        await fx.replay();
        assert.equal(fx.card().querySelector('[data-live-phase]').hidden, false);
        assert.equal(fx.card().querySelector('[data-live-typing]').style.display, '');
        assert.doesNotMatch(fx.card().querySelector('[data-live-meta]').innerHTML, /unconfirmed/);
        fx.send({ task_id: taskId, chat_id: chatId, role: 'system', system_type: 'task_summary',
            task_terminal_status: 'completed', ts: '2026-09-15T10:00:03Z', content: 'Done' });
        assert.equal(fx.card().dataset.finished, '1');
        assert.equal(fx.card().querySelector('[data-live-phase]').textContent, 'Done');
    } finally { fx.destroy(); }
});

for (const firstDetail of ['completed', 'running', 'unavailable']) {
    test(`a vanished direct activity reconciles its visible card from durable detail: ${firstDetail}`, async () => {
        const fx = fixture();
        try {
            await fx.replay();
            fx.hydrate();
            fx.send(progress);
            fx.snapshot.active_chat_activities = [];
            fx.data.detail.status = firstDetail === 'completed' ? 'completed' : 'running';
            fx.data.detailError = firstDetail === 'unavailable';
            fx.hydrate();
            await new Promise((resolve) => setImmediate(resolve));
            assert.ok(fx.calls.some((url) => url.startsWith('/api/tasks/')),
                'census absence must not suppress the existing durable-result reader');
            if (firstDetail !== 'completed') {
                assert.equal(fx.card().dataset.finished, '0', 'absence/failure is not a terminal fact');
                fx.data.detail.status = 'completed';
                fx.data.detailError = false;
                fx.hydrate();
                await new Promise((resolve) => setImmediate(resolve));
            }
            assert.equal(fx.card().dataset.finished, '1');
            assert.equal(fx.card().querySelector('[data-live-phase]').textContent, 'Done');
            assert.equal(fx.card().querySelector('[data-live-typing]').style.display, 'none');
        } finally { fx.destroy(); }
    });
}

for (const channel of ['history', 'chat', 'log']) {
    test(`${channel} consumes review-reference cost without changing activity or note count`, async () => {
        const fx = fixture();
        try {
            await fx.replay();
            fx.hydrate();
            const before = anatomy(fx.card());
            const timestamp = fx.card().querySelector('[data-live-meta]').innerHTML;
            if (channel === 'history') {
                fx.data.rows = [progress, reference(8.53)];
                await fx.replay();
            } else {
                fx.send(reference(8.53), channel);
                await new Promise((resolve) => setImmediate(resolve));
            }
            assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /up to \$8\.53/);
            assert.deepEqual(anatomy(fx.card()), before);
            assert.ok(fx.card().querySelector('[data-live-meta]').innerHTML.includes(timestamp),
                'money leaves the observed activity time unchanged');
        } finally { fx.destroy(); }
    });
}

test('cold history recovers money even while the task activity remains unconfirmed', async () => {
    const fx = fixture([progress, reference(8.53)]);
    try {
        await fx.replay();
        assert.equal(fx.card().querySelector('[data-live-phase]').hidden, true);
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /Activity unconfirmed/);
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /up to \$8\.53/);
    } finally { fx.destroy(); }
});

test('reference cost is independent of review hydration revision and retains sticky precedence', async () => {
    const fx = fixture();
    try {
        await fx.replay();
        fx.hydrate();
        fx.send(reference(8.53));
        await new Promise((resolve) => setImmediate(resolve));
        const detailReads = fx.calls.filter((url) => url.startsWith('/api/tasks/')).length;
        const before = anatomy(fx.card());
        fx.send({ ...reference(9), ts: '2026-09-15T10:00:02Z' });
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /up to \$9\.00/);
        fx.send({ ...reference(99), ts: '2026-09-15T09:59:59Z' });
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /up to \$9\.00/);
        fx.send({ ...reference(10, true), ts: '2026-09-15T10:00:03Z' });
        fx.send({ ...reference(99), ts: '2026-09-15T10:00:04Z' });
        fx.send({ ...reference(0), ...Object.fromEntries(Object.keys(cost(0)).map((key) => [key, undefined])) });
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /\$10\.00/);
        assert.doesNotMatch(fx.card().querySelector('[data-live-meta]').innerHTML, /up to|\$99\.00|\$0\.00/);
        assert.deepEqual(anatomy(fx.card()), before);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(fx.calls.filter((url) => url.startsWith('/api/tasks/')).length, detailReads);
    } finally { fx.destroy(); }
});

test('a nested review reference never charges its presentation owner for the carrier task', async () => {
    const fx = fixture([progress, { ...progress, task_id: 'review-rail', history_id: 'progress:3' }]);
    try {
        await fx.replay();
        fx.send(reference(2));
        fx.send({ ...reference(19), task_id: 'review-rail' });
        assert.match(fx.card().querySelector('[data-live-meta]').innerHTML, /up to \$2\.00/);
        assert.doesNotMatch(fx.card().querySelector('[data-live-meta]').innerHTML, /19\.00/);
        assert.match(fx.card('review-rail').querySelector('[data-live-meta]').innerHTML, /up to \$19\.00/);
        fx.send({ ...reference(47), task_id: 'missing-rail' });
        assert.equal(fx.card('missing-rail'), null, 'cost alone does not invent a task card');
        assert.doesNotMatch(fx.card().querySelector('[data-live-meta]').innerHTML, /47\.00/);
    } finally { fx.destroy(); }
});
