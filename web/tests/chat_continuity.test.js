// Project-continuity regressions: local-echo journal reconciliation against
// stale history responses, managed-task activity hydration/status, and the
// non-terminal task_cost_finalized card projection (task_done concludes).
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    computeHydratedDirectActivities,
    partitionLocalEchoJournal,
    positiveTaskTerminalFact,
    reconcileHydratedDirectActivities,
} from '../modules/chat_activity.js';
import { emptyWorkState, roomStatusView } from '../modules/work_status.js';
import { room } from './work_fixture.js';
import {
    isTerminalTaskDetail,
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskDoneIsTerminal,
    taskOutcomeSeverity,
    taskTerminalPhase,
} from '../modules/log_events.js';

// ---------------------------------------------------------------------------
// Local-echo journal: a stale history snapshot must not erase the owner's row.
// ---------------------------------------------------------------------------

test('partitionLocalEchoJournal: a stale response keeps the fresh send unconfirmed', () => {
    const journal = new Map([
        ['cmid-fresh', { clientMessageId: 'cmid-fresh', text: 'just sent', ts: '2026-08-17T15:00:00Z' }],
        ['cmid-old', { clientMessageId: 'cmid-old', text: 'earlier row', ts: '2026-08-17T14:00:00Z' }],
    ]);
    // The response was assembled before `cmid-fresh` reached chat.jsonl.
    const { confirmed, unconfirmed } = partitionLocalEchoJournal(journal, new Set(['cmid-old']));
    assert.deepEqual(confirmed.map((entry) => entry.clientMessageId), ['cmid-old']);
    assert.deepEqual(unconfirmed.map((entry) => entry.clientMessageId), ['cmid-fresh']);
});

test('partitionLocalEchoJournal: an authoritative response confirms every row', () => {
    const journal = new Map([
        ['a', { clientMessageId: 'a', text: '1', ts: 't1' }],
        ['b', { clientMessageId: 'b', text: '2', ts: 't2' }],
    ]);
    const { confirmed, unconfirmed } = partitionLocalEchoJournal(journal, new Set(['a', 'b']));
    assert.equal(confirmed.length, 2);
    assert.equal(unconfirmed.length, 0);
});

test('partitionLocalEchoJournal: tolerates arrays, empty ids, and a missing set', () => {
    const rows = [
        { clientMessageId: '', text: 'no id', ts: 't' },
        { clientMessageId: 'x', text: 'kept', ts: 't' },
    ];
    const { confirmed, unconfirmed } = partitionLocalEchoJournal(rows, null);
    assert.equal(confirmed.length, 0);
    assert.deepEqual(unconfirmed.map((entry) => entry.clientMessageId), ['x']);
});

// ---------------------------------------------------------------------------
// Managed-task hydration: the queue-backed snapshot is the running authority.
// ---------------------------------------------------------------------------

test('hydration inserts managed queue roots with their queue phase', () => {
    const snapshot = [
        { activity_id: 'root-1', chat_id: 7, kind: 'managed_task', phase: 'working' },
        { activity_id: 'root-2', chat_id: 7, kind: 'managed_task', phase: 'queued' },
        { activity_id: 'other-chat', chat_id: 9, kind: 'managed_task', phase: 'working' },
    ];
    const updated = computeHydratedDirectActivities(new Map(), snapshot, 7);
    assert.equal(updated.size, 2);
    assert.equal(updated.get('root-1').kind, 'managed_task');
    assert.equal(updated.get('root-1').phase, 'working');
    assert.equal(updated.get('root-2').phase, 'queued');
});

test('a complete census concludes every entry it does not list, whatever its kind', () => {
    const existing = new Map([
        ['done-root', { activityId: 'done-root', kind: 'managed_task', phase: 'working' }],
        ['second-root', { activityId: 'second-root', kind: 'managed_task', phase: 'working' }],
        // Kind-less entries were exempt while typing frames wrote this map; the
        // census is now its only writer, so a complete listing is total.
        ['legacy', { activityId: 'legacy', kind: '', phase: 'thinking' }],
    ]);
    assert.equal(computeHydratedDirectActivities(existing, [], 1, null, true).size, 0);
    // An incomplete listing (supervisor not ready, or a source failed) deletes nothing.
    assert.equal(computeHydratedDirectActivities(existing, [], 1, null, false).size, 3);
});

test('an existing entry upgrades to the census kind and phase', () => {
    const existing = new Map([
        ['root-1', { activityId: 'root-1', kind: 'managed_task', phase: 'working', clientMessageId: 'cm-1' }],
    ]);
    const snapshot = [
        { activity_id: 'root-1', chat_id: 1, kind: 'managed_task', phase: 'finalizing' },
    ];
    const updated = computeHydratedDirectActivities(existing, snapshot, 1);
    assert.equal(updated.get('root-1').phase, 'finalizing');
    assert.equal(updated.get('root-1').clientMessageId, 'cm-1');  // submission link survives
});

test('managed queue loss candidates require prior host kind', () => {
    const existing = new Map([
        ['lost-root', { activityId: 'lost-root', kind: 'managed_task', phase: 'working' }],
        ['kindless-child', { activityId: 'kindless-child', kind: '', phase: 'thinking' }],
        ['direct-turn', { activityId: 'direct-turn', kind: 'direct_chat', phase: 'thinking' }],
    ]);
    const result = reconcileHydratedDirectActivities(existing, [], 1);

    assert.deepEqual(result.departedManagedTaskIds, ['lost-root']);
    assert.deepEqual(result.disappearedManagedTaskIds, ['lost-root']);
    assert.equal(result.activities.size, 0);  // the complete census deletes every unlisted id
    // Only host-stamped managed roots carry task-detail authority; the rest are
    // header-only conclusions of record.
    assert.deepEqual(
        result.concludedDirectActivities.map((row) => row.activityId),
        ['kindless-child', 'direct-turn'],
    );
});

test('managed rehome departs locally without becoming globally missing', () => {
    const existing = new Map([
        ['root-1', { activityId: 'root-1', kind: 'managed_task', phase: 'working' }],
    ]);
    const rehomed = [
        { activity_id: 'root-1', chat_id: 9, kind: 'managed_task', phase: 'working' },
    ];
    const result = reconcileHydratedDirectActivities(existing, rehomed, 1);

    assert.equal(result.activities.size, 0);
    assert.deepEqual(result.departedManagedTaskIds, ['root-1']);
    assert.deepEqual(result.disappearedManagedTaskIds, []);
    assert.ok(result.globallyActiveActivityIds.has('root-1'));

    // The same global fact is returned after the local entry is already gone,
    // allowing a caller to clear an earlier retry candidate without recapture.
    const later = reconcileHydratedDirectActivities(new Map(), rehomed, 1);
    assert.ok(later.globallyActiveActivityIds.has('root-1'));
    assert.deepEqual(later.disappearedManagedTaskIds, []);
});

test('concluded managed roots cannot be recaptured or resurrected by an old snapshot', () => {
    const concluded = new Map([['done-root', 123]]);
    const existing = new Map([
        ['done-root', { activityId: 'done-root', kind: 'managed_task', phase: 'working' }],
    ]);
    const result = reconcileHydratedDirectActivities(
        existing,
        [{ activity_id: 'done-root', chat_id: 1, kind: 'managed_task', phase: 'working' }],
        1,
        concluded,
    );
    assert.equal(result.activities.size, 0);
    assert.deepEqual(result.disappearedManagedTaskIds, []);
});

test('durable detail terminality is narrow and outcome labels stay unchanged', () => {
    for (const status of ['', 'requested', 'scheduled', 'running', 'interrupted', 'cancel_requested']) {
        assert.equal(isTerminalTaskDetail({ status }), false, status);
    }
    for (const status of ['completed', 'failed', 'cancelled', 'rejected_duplicate']) {
        assert.equal(isTerminalTaskDetail({ status }), true, status);
    }
    for (const post_task_synthesis of ['pending_once', 'running']) {
        for (const status of ['completed', 'failed']) {
            const record = { type: 'task_done', status, root_phase_checkpoint: { post_task_synthesis } };
            assert.equal(isTerminalTaskDetail(record), false, `${status}/${post_task_synthesis}`);
            assert.equal(taskDoneIsTerminal(record), false);
            assert.equal(summarizeChatLiveEvent(record).terminal, false);
            // #1110: an unfinished frame paints Working, and the KNOWN outcome
            // rides beside it as its own fact — the card states it in the chip
            // with `Finalizing…` next to it, instead of the whole frame having
            // to claim `Failed` before the run is terminal.
            assert.equal(summarizeChatLiveEvent(record).headline, 'Working');
            assert.equal(summarizeChatLiveEvent(record).observedOutcome,
                status === 'failed' ? 'error' : 'done');
            // Logs is an event ledger, not a card chip: its row keeps the
            // outcome word this frame carries.
            assert.equal(summarizeLogEvent(record).headline,
                status === 'failed' ? 'Failed' : 'Working');
        }
    }
    assert.equal(isTerminalTaskDetail({
        status: 'completed',
        root_phase_checkpoint: { post_task_synthesis: 'completed' },
    }), true, 'completed/completed');
    assert.equal(isTerminalTaskDetail({
        status: 'failed',
        root_phase_checkpoint: { post_task_synthesis: 'degraded' },
    }), true, 'failed/degraded');
    assert.equal(taskTerminalPhase({ status: 'completed' }), 'done');
    assert.equal(taskTerminalPhase({ status: 'cancelled' }), 'cancelled');
    assert.equal(taskTerminalPhase({ status: 'failed' }), 'error');
});

// ---------------------------------------------------------------------------
// Header status: the server's room surfaces managed work as Working... / Queued...
// ---------------------------------------------------------------------------

const inRoom = (activity, extra = {}) => ({ ...emptyWorkState(), room: room(activity, extra) });

test('admitted managed work shows Working... without a live card (the room says it)', () => {
    assert.deepEqual(roomStatusView(inRoom('working')),
        { kind: 'thinking', text: 'Working', tone: 'live', mark: 'moving', owner: false, amber: '' });
});

test('room status order: the room\'s work > the owner\'s send > Online; queued work yields to a send', () => {
    assert.equal(roomStatusView(inRoom('working'), { sending: true }).text, 'Working');
    assert.equal(roomStatusView(inRoom('working', { call: 'thinking' }), { sending: true }).text, 'Thinking');
    assert.equal(roomStatusView(inRoom('queued'), { sending: true }).text, 'Sending');
    assert.equal(roomStatusView(inRoom('queued')).text, 'Queued');
    assert.equal(roomStatusView(emptyWorkState()).text, 'Online');
});

// ---------------------------------------------------------------------------
// task_cost_finalized is bookkeeping; only task_done resolves the card.
// ---------------------------------------------------------------------------

test('task_cost_finalized chat projection is non-terminal', () => {
    const summary = summarizeChatLiveEvent({
        type: 'task_cost_finalized',
        task_id: 'root-1',
        post_task_status: 'completed',
        cost_accounting_status: 'available',
        accounted_upper_bound_usd: 0.5,
        cost_final: true,
    });
    assert.equal(summary.terminal, false);
    assert.notEqual(summary.phase, 'done');
    // The unavailable variant stays a warn note, still non-terminal.
    const unavailable = summarizeChatLiveEvent({
        type: 'task_cost_finalized',
        task_id: 'root-1',
        cost_accounting_status: 'unavailable',
    });
    assert.equal(unavailable.terminal, false);
    assert.equal(unavailable.phase, 'warn');
});

test('task_done stays the terminal card projection', () => {
    const summary = summarizeChatLiveEvent({
        type: 'task_done',
        task_id: 'root-1',
        status: 'completed',
        outcome_axes: { lifecycle: { status: 'completed' } },
    });
    assert.equal(summary.terminal, true);
});

test('history terminal truth requires a positive typed fact', () => {
    assert.equal(positiveTaskTerminalFact({
        role: 'system', task_id: 'root', system_type: 'skill_review',
    }), false);
    assert.equal(positiveTaskTerminalFact({
        role: 'assistant', task_id: 'root', system_type: 'photo',
    }), false);
    assert.equal(positiveTaskTerminalFact({
        role: 'system', task_id: 'root', system_type: 'task_summary',
    }), true);
    assert.equal(positiveTaskTerminalFact({
        role: 'assistant', task_id: 'root', task_terminal_status: 'failed',
    }), true);
    assert.equal(positiveTaskTerminalFact({
        delegation_role: 'subagent', task_id: 'child', subagent_event: 'completed',
    }), true);
    assert.equal(positiveTaskTerminalFact({
        delegation_role: 'subagent', task_id: 'child', subagent_event: 'interrupted',
    }), false);
});

test('typed terminal status is the phase authority even without a legacy status field', () => {
    assert.equal(taskOutcomeSeverity({ task_terminal_status: 'failed' }), 'error');
    assert.equal(taskTerminalPhase({ task_terminal_status: 'failed' }), 'error');
    assert.equal(taskOutcomeSeverity({ task_terminal_status: 'cancelled' }), 'cancelled');
    assert.equal(taskTerminalPhase({ task_terminal_status: 'completed' }), 'done');
});

test('removed direct rows come back as conclusions of record (#369)', () => {
    const existing = new Map([
        ['direct-turn', { activityId: 'direct-turn', kind: 'direct_chat', phase: 'thinking', clientMessageId: 'cm-1' }],
        ['direct-second', { activityId: 'direct-second', kind: 'direct_chat', phase: 'thinking' }],
        ['done-before', { activityId: 'done-before', kind: 'direct_chat', phase: 'thinking' }],
        ['still-live', { activityId: 'still-live', kind: 'direct_chat', phase: 'thinking' }],
    ]);
    const snapshot = [
        { activity_id: 'still-live', chat_id: 1, kind: 'direct_chat', phase: 'thinking', started_at: 0.1 },
    ];
    const concluded = new Set(['done-before']);
    const result = reconcileHydratedDirectActivities(existing, snapshot, 1, concluded);
    assert.deepEqual(
        result.concludedDirectActivities,
        [
            { activityId: 'direct-turn', clientMessageId: 'cm-1' },
            { activityId: 'direct-second', clientMessageId: '' },
        ],
    );
    // Already-concluded and still-live rows are never re-concluded.
    assert.ok(!result.concludedDirectActivities.some((r) => r.activityId === 'done-before'));
    assert.ok(!result.concludedDirectActivities.some((r) => r.activityId === 'still-live'));
});
