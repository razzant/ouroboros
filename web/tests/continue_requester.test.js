// Who asked for a Continue is its own fact in the card's Details, beside the origin the work inherits (#3047).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { continueTask, APP_SETTINGS_ORIGIN } from '../modules/api_client.js';
import { cardMetaKeys } from '../modules/chat_activity.js';
import { CANCEL_SOURCE_PHRASES, CANCEL_SOURCE_LABELS } from '../modules/cancel_presentation.js';
import { taskReasonDetail } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};
const TS = '2026-11-10T09:00:00Z';

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history')
            ? { messages: history, window: { complete: true } } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    return {
        ready: () => instance.hasPaintedHistory(),
        refresh: () => instance.refreshHistory({ revision: 1 }),
        facts: (id) => walkCard(messages, id)?.querySelector('[data-live-facts]')?.innerHTML || '',
        emit: (type, row) => handlers.get(type)({ chat_id: 1, ts: TS, ...row }),
        log: (row) => handlers.get('log')({ chat_id: 1, data: { ts: TS, ...row } }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const frame = (task, patch) => ({ task_id: task, role: 'assistant', is_progress: true, text: '💬 Carrying on', content: '💬 Carrying on', cancelable: true, ...patch });

for (const [requester, word] of [['owner', 'You'], ['ouroboros', 'Ouroboros'], ['unknown', 'Not recorded']]) {
    test(`a Continue requested by ${requester} says so beside its inherited origin`, () => {
        const f = fixture();
        try {
            f.emit('chat', frame('next', { initiator: 'consciousness', continue_requester: requester }));
            const facts = f.facts('next');
            assert.match(facts, /Continued by/);
            assert.ok(facts.includes(`<dd>${word}</dd>`), facts);
            assert.match(facts, /Started by/);  // the origin stays its own fact
        } finally { f.close(); }
    });
}

test('a Continue whose first frame is its terminal still names who asked for it', () => {
    const f = fixture();
    try {
        f.log({ type: 'task_done', task_id: 'ended', chat_id: 1, status: 'failed', reason_code: 'task_done_lifecycle_fault',
            outcome_axes: { execution: { status: 'infra_failed' } }, continue_requester: 'owner' });
        assert.match(f.facts('ended'), /Continued by/);
        assert.ok(f.facts('ended').includes('<dd>You</dd>'), f.facts('ended'));
    } finally { f.close(); }
});

test('a wake that nobody continued names no requester', () => {
    const f = fixture();
    try {
        f.emit('chat', frame('wake', { initiator: 'consciousness' }));
        assert.doesNotMatch(f.facts('wake'), /Continued by/);
    } finally { f.close(); }
});

test('the requester survives on the card facts carried to a child frame', () => {
    assert.equal(cardMetaKeys({ continue_requester: 'owner' }).continue_requester, 'owner');
});

test('the app declares its own press and only its own', async () => {
    const seen = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init) => { seen.push({ url: String(url), init }); return { ok: true, status: 200, json: async () => ({ ok: true, successor_task_id: 's' }), text: async () => '{"ok":true}', headers: new Map([['content-type', 'application/json']]) }; };
    try {
        await continueTask('root-1', 'press-1');
    } finally { globalThis.fetch = prior; }
    const headers = seen[0].init.headers;
    assert.equal(headers['X-Ouroboros-Surface'], APP_SETTINGS_ORIGIN['X-Ouroboros-Surface']);
    assert.equal(headers['X-Ouroboros-Actor'], 'owner');
    assert.deepEqual(JSON.parse(seen[0].init.body), { action_nonce: 'press-1' });
});

test('the closed helper names why, once', () => {
    assert.equal(CANCEL_SOURCE_PHRASES.work_finished, 'Closed with the work it helped');
    const detail = taskReasonDetail({ status: 'cancelled', cancel_origin: { source: 'work_finished', reason: CANCEL_SOURCE_LABELS.work_finished } });
    assert.match(detail, /Closed with the work it helped/);
    assert.doesNotMatch(detail, /has ended/);
});

test('a history page whose only row is the task summary still names who asked', async () => {
    const row = { task_id: 'summary-only', role: 'system', system_type: 'task_summary', status: 'failed', outcome_final: true,
        reason_code: 'provider_unavailable', text: 'Work ended', ts: TS, continue_requester: 'owner',
        outcome_axes: { execution: { status: 'infra_failed' } } };
    const f = fixture([row]);
    try {
        await f.refresh();
        assert.ok(f.ready());
        assert.match(f.facts('summary-only'), /Continued by/);
        assert.ok(f.facts('summary-only').includes('<dd>You</dd>'), f.facts('summary-only'));
    } finally { f.close(); }
});
