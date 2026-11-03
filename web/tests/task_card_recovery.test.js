import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { fetchTaskDetailStrict } from '../modules/api_client.js';
import { createReviewHydrator } from '../modules/review_presentation.js';
import { clock } from '../modules/work_status.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, room, work } from './work_fixture.js';

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

test('task detail hydration retries migration quietly and cancels on disposal', async t => {
    const previous = globalThis.fetch;
    let migrating = true, reads = 0;
    const statuses = [], applied = [];
    globalThis.fetch = async () => {
        reads += 1;
        return { ok: !migrating, status: migrating ? 503 : 200,
            json: async () => migrating ? { reason_code: 'migrating' } : { status: 'completed' } };
    };
    const hydrator = createReviewHydrator({ fetchDetail: fetchTaskDetailStrict,
        applyDetail: (_id, row) => applied.push(row), onState: (_id, state) => statuses.push(state) });
    t.after(() => { hydrator.clear(); globalThis.fetch = previous; });
    await hydrator.hydrate('task1');
    assert.equal(statuses.at(-1), 'loading');
    assert.equal(hydrator.ready('task1'), false);
    migrating = false;
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.deepEqual(applied, [{ status: 'completed' }]);
    assert.equal(hydrator.ready('task1'), true);
    assert.equal(statuses.includes('error'), false);
    assert.equal(reads, 2);
    migrating = true;
    await hydrator.hydrate('task2');
    hydrator.clear();
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(reads, 3, 'closing the consumer cancels the pending migration retry');
    globalThis.fetch = async () => {
        reads += 1;
        return { ok: false, status: 503, json: async () => ({ reason_code: 'unavailable' }) };
    };
    await hydrator.hydrate('task3');
    assert.equal(statuses.at(-1), 'error');
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(reads, 4, 'ordinary failures keep their existing Retry contract');
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
            if (data.detailMigrating) return { ok: false, status: 503, json: async () => ({ reason_code: 'migrating' }) };
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

// A chip's status mark (docs/DESIGN.md "Status marks"): its shape and hue, or `none` while it is hidden.
const markOf = (chip) => {
    const mark = chip.firstElementChild;
    return mark && !mark.hidden ? `${mark.dataset.shape}/${mark.dataset.hue}` : 'none';
};
const anatomy = (card) => ({
    phase: card.querySelector('[data-live-phase]').textContent,
    hidden: card.querySelector('[data-live-phase]').hidden,
    mark: markOf(card.querySelector('[data-live-phase]')),
    notes: card.querySelector('[data-live-count]').textContent,
    activity: card.querySelector('[data-live-activity]').textContent,
});

const findDescendant = (node, selector) => node.querySelector(selector)
    || node.children.map(child => findDescendant(child, selector)).find(Boolean) || null;

for (const kind of ['direct_chat', 'managed_task']) {
    for (const censusFirst of [false, true]) {
        test(`${kind} paused review work reaches the real card from census hydration (census first=${censusFirst})`, async () => {
            const fx = fixture();
            try {
                const activity = fx.snapshot.active_chat_activities[0];
                Object.assign(activity, { kind, phase: 'budget_paused', pause_cause: 'owner', finishing_reviews: true });
                // The server's work for that row: paused by the owner's press, its launched reviews finishing beside
                // the Pause as its tail, and the Resume its door accepts (gateway/work_status.roll_up_work, controls_for).
                const pressed = '2026-09-15T10:00:00Z';
                const review = { kind: 'review', state: 'live', code: 'review_work_finishing', text: 'Review work finishing' };
                const project = () => Object.assign(fx.snapshot, projection(activity.chat_id ?? 1, [work(activity,
                    activity.phase === 'budget_paused' ? { activity: 'paused', mark: 'pause',
                        because: [{ code: 'paused_by_owner', text: 'You paused it', since: pressed,
                            by: { actor: 'owner', text: '', at: pressed } }],
                        tails: activity.finishing_reviews ? [review] : [], live: Boolean(activity.finishing_reviews),
                        controls: [{ action: 'resume', target: taskId }, { action: 'stop', target: taskId }] }
                        : { mark: 'moving', controls: [{ action: 'stop', target: taskId }] })],
                    room(activity.phase === 'budget_paused' ? 'paused' : 'working')));
                project();
                if (censusFirst) fx.hydrate();
                await fx.replay();
                if (!censusFirst) fx.hydrate();
                // The chip says the work's word alone; who paused it and what still finishes are its line.
                const phase = () => [fx.card().querySelector('[data-live-phase]').textContent,
                    fx.card().querySelector('[data-live-phase-secondary]').textContent];
                const paused = `You paused it · ${clock(pressed)}`;
                assert.deepEqual(phase(), ['Paused', `${paused} · Review work finishing`]);
                assert.ok(findDescendant(fx.card(), '[data-resume-run]'), 'Resume is available while reviews finish');
                // The owner's own Pause is their decision, not a move waiting for them: its bars in neutral ink.
                assert.deepEqual([markOf(fx.card().querySelector('[data-live-phase]')),
                    fx.card().querySelector('[data-live-phase]').dataset.chipTone], ['pause/ink', 'neutral']);
                for (const finishing of [true, false, true, undefined]) {
                    if (finishing === undefined) delete activity.finishing_reviews;
                    else activity.finishing_reviews = finishing;
                    project();
                    fx.hydrate();
                    assert.deepEqual(phase(), ['Paused', finishing ? `${paused} · Review work finishing` : paused],
                        'each body replaces the fact, including false or absence');
                    assert.ok(findDescendant(fx.card(), '[data-resume-run]'));
                }
                Object.assign(activity, { phase: 'working', finishing_reviews: true });
                project();
                fx.hydrate();
                assert.deepEqual(phase(), ['Working', ''], 'a stale fact cannot keep the resumed task paused');
                assert.equal(findDescendant(fx.card(), '[data-resume-run]'), null);
            } finally { fx.destroy(); }
        });
    }
}

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
        assert.equal(markOf(fx.card().querySelector('[data-live-phase]')), 'moving/ink');
        assert.doesNotMatch(fx.card().querySelector('[data-live-meta]').innerHTML, /unconfirmed/);
        fx.send({ task_id: taskId, chat_id: chatId, role: 'system', system_type: 'task_summary',
            task_terminal_status: 'completed', ts: '2026-09-15T10:00:03Z', content: 'Done' });
        assert.equal(fx.card().dataset.finished, '1');
        assert.equal(fx.card().querySelector('[data-live-phase]').textContent, 'Done');
    } finally { fx.destroy(); }
});

for (const firstDetail of ['completed', 'running', 'unavailable', 'migrating']) {
    test(`a vanished direct activity reconciles its visible card from durable detail: ${firstDetail}`, async () => {
        const fx = fixture();
        try {
            await fx.replay();
            fx.hydrate();
            fx.send(progress);
            fx.snapshot.active_chat_activities = [];
            fx.data.detail.status = firstDetail === 'completed' ? 'completed' : 'running';
            fx.data.detailError = firstDetail === 'unavailable';
            fx.data.detailMigrating = firstDetail === 'migrating';
            fx.hydrate();
            await new Promise((resolve) => setImmediate(resolve));
            assert.ok(fx.calls.some((url) => url.startsWith('/api/tasks/')),
                'census absence must not suppress the existing durable-result reader');
            if (firstDetail !== 'completed') {
                assert.equal(fx.card().dataset.finished, '0', 'absence/failure is not a terminal fact');
                fx.data.detail.status = 'completed';
                fx.data.detailError = false;
                fx.data.detailMigrating = false;
                fx.hydrate();
                await new Promise((resolve) => setImmediate(resolve));
            }
            assert.equal(fx.card().dataset.finished, '1');
            assert.equal(fx.card().querySelector('[data-live-phase]').textContent, 'Done');
            assert.equal(markOf(fx.card().querySelector('[data-live-phase]')), 'dot/ok');
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
