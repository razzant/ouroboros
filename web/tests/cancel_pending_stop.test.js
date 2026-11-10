// The owner's hard Stop answered 202 pending — the durable stop is recorded and
// the supervisor finishes it (docs/CLIENT_CONTRACT.md §8). Driven through the
// real card control and the shared requestStop: an ok:true answer is accepted,
// task detail keeps the card «Cancelling…» and no «Cancel failed» toast is
// raised. The control case pins that this harness does see that toast for a
// Stop the server refuses, so the pending case is a real negative.
//
// The later cases read what the card does with each answer and durable record: its
// phase, title and line, whether it finished, whether Stop is still offered, and every
// write to the Stop button's disabled flag, so a transient re-enable counts too.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { taskTerminalSummary } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// The flat fixture's querySelector does not descend; the card's Stop trigger
// lives in its actions row.
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

const collect = (node, predicate, out = []) => {
    if (predicate(node)) out.push(node);
    for (const child of node?.children || []) collect(child, predicate, out);
    return out;
};

// A browser click: both controls stop propagation on their own event.
const press = (node) => {
    for (const fn of node.listeners.get('click') || []) fn({ target: node, stopPropagation() {}, preventDefault() {} });
};

// `detail`: the durable record; null is a read the server answers with an error,
// an Error a read that never reaches it. `before`: frames the card saw before Stop.
async function stopNowAnsweredWith(taskId, cancelStatus, cancelBody, {
    detail = { task_id: taskId, status: 'running', cancel_state: 'pending' }, before = [],
} = {}) {
    const calls = [];
    const env = installDom(async (url, init = {}) => {
        calls.push(`${init.method || 'GET'} ${url}`);
        if (String(url) === `/api/tasks/${taskId}/cancel`) {
            return { ok: cancelStatus < 400, status: cancelStatus, json: async () => cancelBody };
        }
        if (String(url) === `/api/tasks/${taskId}`) {
            if (detail instanceof Error) throw detail;
            return detail === null ? { ok: false, status: 500, json: async () => ({ error: 'unavailable' }) }
                : { ok: true, status: 200, json: async () => detail };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const priorObserver = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    const body = new ElementStub('body', globalThis.document);
    body.isConnected = true;
    globalThis.document.body = body;
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: {
            begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {},
        },
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
    });
    try {
        handlers.get('chat')({
            chat_id: 1, task_id: taskId, role: 'assistant', is_progress: true, cancelable: true,
            content: 'Working on it.', ts: '2026-10-14T20:00:00Z',
        });
        for (const frame of before) handlers.get('chat')({ chat_id: 1, task_id: taskId, ...frame });
        const card = walkCard(globalThis.document.byId.get('chat-messages'), taskId);
        const trigger = card.querySelector('[data-cancel-run]');
        assert.ok(trigger, 'the host-attested marker offers Stop');
        const disabledWrites = [];
        let disabled = trigger.disabled;
        Object.defineProperty(trigger, 'disabled', {
            get: () => disabled, set: (value) => { disabled = value; disabledWrites.push(value); },
        });
        press(trigger);
        const [stopNow] = collect(body, (node) => node?.dataset?.taskControl === 'stop_now');
        assert.ok(stopNow, 'the dropdown offers Stop now');
        press(stopNow);
        const detailRead = `GET /api/tasks/${taskId}`;
        for (let i = 0; i < 50 && !calls.includes(detailRead); i += 1) {
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
        const toasts = collect(body, (node) => node?.classList?.contains('toast')).map((node) => node.textContent);
        return {
            calls, toasts, readDetail: calls.includes(detailRead),
            triggerDisabled: trigger.disabled, disabledWrites,
            stopOffered: card.querySelector('[data-cancel-run]') === trigger,
            finished: card.dataset.finished,
            phase: card.querySelector('[data-live-phase]')?.textContent,
            title: card.querySelector('[data-live-title]')?.textContent,
            line: card.querySelector('[data-live-activity]')?.textContent,
        };
    } finally {
        instance.destroy();
        globalThis.IntersectionObserver = priorObserver;
        restoreDom(env.prior);
    }
}

test('a 202 pending Stop keeps the card Cancelling… with no Cancel failed toast', async () => {
    const seen = await stopNowAnsweredWith('pending-root', 202, {
        ok: true, task_id: 'pending-root', cancel_state: 'pending', stop_policy: 'immediate',
        request_id: 'ci_pending', reason_code: 'teardown_unconfirmed', cascade: true,
    });
    assert.ok(seen.calls.includes('POST /api/tasks/pending-root/cancel'), seen.calls);
    assert.ok(seen.calls.includes('GET /api/tasks/pending-root'), 'the card reconciles from task detail');
    assert.deepEqual(seen.toasts.filter((text) => /Cancel failed/.test(text)), []);
    assert.equal(seen.phase, 'Cancelling…');
    assert.equal(seen.finished, '0', 'only the settled task_done closes the card');
    assert.equal(seen.triggerDisabled, true, 'a hard Stop stays pending; the trigger is not re-armed');
});

test('control: a Stop the server refuses raises the Cancel failed toast in the same harness', async () => {
    const seen = await stopNowAnsweredWith('refused-root', 503, {
        error: 'durable cancel intent could not be recorded; nothing was cancelled — retry',
        task_id: 'refused-root', reason_code: 'cancel_intent_write_failed',
    });
    assert.ok(seen.calls.includes('POST /api/tasks/refused-root/cancel'), seen.calls);
    assert.equal(seen.toasts.filter((text) => /^Cancel failed: durable cancel intent could not be recorded/.test(text)).length, 1,
        seen.toasts);
});

// Each answer a Stop can get: accepted while task_done publication is lost, already
// gone, refused. All three read the durable record through one reconcile, so the cases
// that exercise that reconcile's own decisions need only the accepted path.
const ACCEPTED = { name: '200', status: 200, body: { ok: true } };
const GONE = { name: '404', status: 404, body: { error: 'task not found or not active' }, revoked: true };
const REFUSED = { name: '503', status: 503, refused: true,
    body: { error: 'durable cancel intent could not be recorded; nothing was cancelled — retry' } };
const ALL = [ACCEPTED, GONE, REFUSED];
const CAUSED = { status: 'failed', reason_code: 'provider_unavailable' };
const UNREAD = { finished: '0', phase: 'Cancelling…', disabledWrites: [true] };
const CASES = [
    // Facts only the full summary carries; the card, not the record, owns the id.
    { answers: ALL, when: 'a settled record closes the card through the full terminal summary',
        detail: { task_id: '', status: 'cancelled', suggested_name: 'Healed name', wake_summary: 'Stopped while drafting' },
        card: { finished: '1', phase: 'Stopped', title: 'Healed name', line: 'Stopped while drafting', disabledWrites: [true] } },
    // The server projects the legacy latch as pending (task_status.py); it stays intent.
    { answers: ALL, when: 'the legacy cancel_requested latch stays pending, its button untouched',
        detail: { status: 'cancel_requested', cancel_state: 'pending' },
        card: { finished: '0', phase: 'Cancelling…', disabledWrites: [true] } },
    { answers: ALL, when: 'the pending Wrap up the record names replaces Cancelling…',
        detail: { status: 'running', cancel_state: 'pending', stop_policy: 'finalize_then_cancel' },
        card: { finished: '0', phase: 'Finalizing…' } },
    { answers: ALL, when: 'a detail read answered with an error proves nothing', detail: null, card: UNREAD },
    { answers: ALL, when: 'a detail read that gets no answer proves nothing', detail: new Error('offline'), card: UNREAD },
    { answers: [ACCEPTED], when: 'a failed record shows its own outcome and cause', detail: CAUSED,
        card: { finished: '1', phase: 'Failed', line: taskTerminalSummary(CAUSED).activityPreview } },
    { answers: [ACCEPTED], when: 'the terminal alias done closes the card', detail: { status: 'done' },
        card: { finished: '1', phase: 'Done' } },
    { answers: [ACCEPTED], when: 'the card stays open while post-task synthesis is open',
        detail: { status: 'completed', root_phase_checkpoint: { post_task_synthesis: 'running' } }, card: { finished: '0' } },
    { answers: [ACCEPTED], when: 'the card stays open while the outcome is not final',
        detail: { status: 'completed', outcome_final: false }, card: { finished: '0' } },
    { answers: [ACCEPTED], when: 'the card stays open while the record still runs',
        detail: { status: 'running' }, card: { finished: '0' } },
    { answers: [REFUSED], when: 'a live record gets its prior phase back and its button re-enabled',
        detail: { status: 'running' }, card: { finished: '0', phase: 'Working', disabledWrites: [true, false] } },
    { answers: [REFUSED], when: 'a finalizing hold comes back, not plain Working', detail: { status: 'running' },
        before: [{ role: 'assistant', content: 'Answer delivered while synthesis runs', task_phase: 'finalizing',
            ts: '2026-10-14T20:00:01Z' }],
        card: { finished: '0', phase: 'Finalizing…', disabledWrites: [true, false] } },
];

let cases = 0;  // Ids are never reused: a late read from an earlier case must not satisfy a later one.
for (const { answers, when, card, ...options } of CASES) {
    for (const { name, status, body, revoked = false, refused = false } of answers) {
        test(`a ${name} Stop: ${when}`, async () => {
            const seen = await stopNowAnsweredWith(`stop-case-${cases += 1}`, status, body, options);
            assert.ok(seen.readDetail, 'the card reads the durable record');
            assert.equal(seen.toasts.filter((text) => /Cancel failed/.test(text)).length, refused ? 1 : 0);
            assert.equal(seen.stopOffered, seen.finished === '0' && !revoked, 'Stop lasts while the card is open and not revoked');
            for (const [fact, value] of Object.entries(card)) assert.deepEqual(seen[fact], value, fact);
        });
    }
}
