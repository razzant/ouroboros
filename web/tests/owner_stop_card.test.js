// The status word a task card paints for an owner stop and for an old Advisory review refusal,
// live and on replay, through the real chat instance: Stop now reads Stopped whatever it kept,
// a Wrap up reads Done over its answer and Stopped without one, an older record that names no
// button keeps its word, and an old procedural refusal over a delivered answer reads Done with
// warnings with its sentence. The shared fold itself is pinned by outcome_phase_parity.json.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';
import { OWNER_STOP_DETAIL_MARKER, taskTerminalSummary } from '../modules/log_events.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const OWN = 'owner_requested_finalization';
const STOP_NOW = { source: 'http_single', scope: 'single', request_id: 'ci_stopnow', stop_policy: 'immediate' };
const WRAP_UP = { source: 'http_graceful', scope: 'single', request_id: 'ci_wrapup', requested_by: 'owner',
    reason: 'owner requested finalize-then-stop', stop_policy: 'finalize_then_cancel' };
const ANSWERED = { lifecycle: { status: 'completed' }, execution: { status: 'best_effort', reason_code: OWN } };
const UNANSWERED = { lifecycle: { status: 'failed' },
    execution: { status: 'failed', reason_code: OWN, failure: { kind: 'agent', reason_code: OWN } } };

function chatInstance(fetchRows) {
    const { prior, mount } = installDom(async (url) => (String(url).startsWith('/api/chat/history')
        ? { ok: true, json: async () => ({ messages: fetchRows(), window: { complete: true } }) }
        : { ok: true, json: async () => ({ active_direct_turns: [] }) }));
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} };
    let generation = 0;
    const stateSnapshots = { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} };
    const instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true });
    const card = (taskId) => {
        const walk = (node) => node.dataset?.taskId === taskId ? node
            : (node.children || []).reduce((hit, child) => hit || walk(child), null);
        return walk(globalThis.document.byId.get('chat-messages'));
    };
    return { instance, handlers, card, close() { instance.destroy(); restoreDom(prior); } };
}

// A direct turn as the history endpoint returns it: its narration row and its final row, each
// carrying the persisted result's terminal truth (gateway/history._annotate_terminal_task_truth).
function directTurn(taskId, { status, axes, origin, answer }) {
    const truth = { task_terminal_status: status, reason_code: OWN, outcome_axes: axes,
        ...(origin ? { cancel_origin: origin } : {}) };
    return [
        { chat_id: 2, role: 'system', is_progress: true, content: 'listing the root', narration: true,
          ts: '2026-10-25T11:00:00Z', task_id: taskId, tool_calls: 3, ...truth },
        { chat_id: 2, role: answer ? 'assistant' : 'system', text: answer || '⏹ The owner stopped this chat turn; no further work was done.',
          content: answer || '⏹ The owner stopped this chat turn; no further work was done.',
          ts: '2026-10-25T11:00:10Z', task_id: taskId, ...truth },
    ];
}

async function replayWord(turn) {
    const rows = directTurn('turn-1', turn);
    const chat = chatInstance(() => rows);
    try {
        await chat.instance.refreshHistory({ revision: 1 });
        await new Promise((resolve) => setTimeout(resolve, 25));
        const card = chat.card('turn-1');
        assert.ok(card, 'the replayed turn minted its card');
        const bubbles = globalThis.document.byId.get('chat-messages').children
            .filter((node) => node.classList?.contains('chat-bubble'));
        return { word: card.querySelector('[data-live-phase]')?.textContent || '',
            line: card.querySelector('[data-live-activity]')?.textContent || '', bubbles };
    } finally {
        chat.close();
    }
}

test('Stop now reads Stopped and names its button, and the answer it kept is still shown', async () => {
    const kept = await replayWord({ status: 'completed', axes: ANSWERED, origin: STOP_NOW, answer: 'Found one cause so far.' });
    assert.equal(kept.word, 'Stopped');
    assert.equal(kept.line, 'Stopped from the app (Stop now)');
    assert.ok(kept.bubbles.some((node) => /Found one cause so far/.test(node.innerHTML || node.textContent || '')),
        'the prepared answer stays its own message');
    const empty = await replayWord({ status: 'failed', axes: UNANSWERED, origin: STOP_NOW });
    assert.equal(empty.word, 'Stopped', 'nothing kept is still the owner\'s stop, never an agent failure');
    assert.equal(empty.line, 'Stopped from the app (Stop now)');
});

test('Wrap up reads Done over its answer and Stopped without one', async () => {
    const answered = await replayWord({ status: 'completed', axes: ANSWERED, origin: WRAP_UP, answer: 'Here is the summary.' });
    assert.equal(answered.word, 'Done');
    const unanswered = await replayWord({ status: 'failed', axes: UNANSWERED, origin: WRAP_UP });
    assert.equal(unanswered.word, 'Stopped');
    assert.equal(unanswered.line, 'Stopped from the app (Wrap up)');
    const hardened = await replayWord({ status: 'completed', axes: ANSWERED,
        origin: { ...WRAP_UP, stop_policy: 'immediate' }, answer: 'Kept answer.' });
    assert.equal(hardened.word, 'Stopped');
    assert.equal(hardened.line, 'Stopped from the app (Stop now)');
});

test('an older owner stop that names no button keeps the word it was recorded with', async () => {
    assert.equal((await replayWord({ status: 'failed', axes: UNANSWERED })).word, 'Failed');
    assert.equal((await replayWord({ status: 'completed', axes: ANSWERED, answer: 'Summary.' })).word, 'Done');
});

test('the live final and the settled terminal of a Stop now agree, so the card never flashes another word', async () => {
    const chat = chatInstance(() => []);
    try {
        chat.handlers.get('chat')({ chat_id: 2, role: 'system', is_progress: true, content: 'checking the cadence test',
            narration: true, ts: '2026-10-25T11:00:00Z', task_id: 'live-1', tool_calls: 2 });
        // The pipeline's final frame (answer first): its progress_meta carries the typed stop.
        chat.handlers.get('chat')({ chat_id: 2, role: 'assistant', content: 'Found one cause so far.',
            ts: '2026-10-25T11:00:09Z', task_id: 'live-1', task_terminal_status: 'completed', reason_code: OWN,
            outcome_axes: ANSWERED, cancel_origin: STOP_NOW });
        assert.equal(chat.card('live-1')?.querySelector('[data-live-phase]')?.textContent, 'Stopped');
        chat.handlers.get('log')({ chat_id: 2, data: { type: 'task_done', task_id: 'live-1', status: 'completed',
            ts: '2026-10-25T11:00:10Z', reason_code: OWN, outcome_axes: ANSWERED, cancel_origin: STOP_NOW } });
        assert.equal(chat.card('live-1')?.querySelector('[data-live-phase]')?.textContent, 'Stopped');
    } finally {
        chat.close();
    }
});

test('a stopped ending carries no "summary at the owner\'s request" marker; a Wrap up\'s answer does', () => {
    const stopped = taskTerminalSummary({ type: 'task_done', status: 'completed', reason_code: OWN,
        outcome_axes: ANSWERED, cancel_origin: STOP_NOW });
    assert.equal(stopped.headline, 'Stopped');
    assert.doesNotMatch(stopped.body, /owner's request/);
    assert.match(stopped.body, /Stopped from the app \(Stop now\)/);
    const summary = taskTerminalSummary({ type: 'task_done', status: 'completed', reason_code: OWN,
        outcome_axes: ANSWERED, cancel_origin: WRAP_UP });
    assert.equal(summary.headline, 'Done');
    assert.equal(summary.body, OWNER_STOP_DETAIL_MARKER);
});

const LEGACY_AXES = (enforcement) => ({
    lifecycle: { status: 'completed' },
    execution: { status: 'ok', reason_code: 'final_message' },
    artifacts: { status: 'ready_no_changes' },
    objective: { status: 'fail', source: 'task_acceptance_review', review_status: 'degraded',
        outcome_tier: 'blocked_with_evidence', reason: 'identical_acceptance_refused' },
    review: { status: 'degraded', aggregate_signals: ['DEGRADED'], acceptance_decision: {
        status: 'finalized_unaccepted', reason: 'identical_acceptance_refused', source: 'task_acceptance_review',
        ...(enforcement ? { enforcement } : {}) } },
});

test('an old Advisory refusal to re-review an unchanged delivered answer reads Done with warnings and says why', async () => {
    const replay = async (axes) => {
        // gateway/history normalizes with the saved answer before projecting axes onto narration.
        const projected = axes.review.acceptance_decision.enforcement === 'advisory'
            ? { ...axes, objective: { ...axes.objective, status: 'best_effort', ignored_status: 'fail' } } : axes;
        const truth = { task_terminal_status: 'completed', reason_code: 'final_message', outcome_axes: projected };
        const rows = [
            { chat_id: 2, role: 'system', is_progress: true, content: 'merging the reviewed change', narration: true,
              ts: '2026-09-20T00:30:00Z', task_id: 'legacy-1', tool_calls: 89, ...truth },
            { chat_id: 2, role: 'assistant', text: 'Done and merged.', content: 'Done and merged.',
              ts: '2026-09-20T00:41:01Z', task_id: 'legacy-1', ...truth },
        ];
        const chat = chatInstance(() => rows);
        try {
            await chat.instance.refreshHistory({ revision: 1 });
            await new Promise((resolve) => setTimeout(resolve, 25));
            const card = chat.card('legacy-1');
            return [card?.querySelector('[data-live-phase]')?.textContent, card?.querySelector('[data-live-activity]')?.textContent];
        } finally {
            chat.close();
        }
    };
    assert.deepEqual(await replay(LEGACY_AXES('advisory')),
        ['Done with warnings', 'Nothing had changed since the last review, so the recorded verdict stands.']);
    // Missing or Blocking enforcement is not guessed Advisory: the recorded fail stands.
    assert.equal((await replay(LEGACY_AXES('blocking')))[0], 'Failed');
    assert.equal((await replay(LEGACY_AXES('')))[0], 'Failed');
});
