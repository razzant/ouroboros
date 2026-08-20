// Behavioural characterization of the subagent card routing owner, exercised
// where the code now lives. No DOM: the routes only read the child->parent
// registry and call back into the card/task helpers, so recording stubs are
// enough to pin the routing decisions — which key a frame lands on, whether a
// terminal child is revived, and which card gets forced visible.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createSubagentRouting } from '../modules/chat_subagent_routing.js';

function routing() {
    const subagentChildParents = new Map();
    const subagentTerminalChildren = new Set();
    const forced = [];
    const updates = [];
    const records = new Map();
    const taskStates = new Map();

    const api = createSubagentRouting({
        subagentChildParents,
        subagentTerminalChildren,
        // Cost presentation is chat.js's; here it must simply pass the summary through.
        withTaskCostMeta: (summary, payload, opts) => Object.assign(summary, { _meta: opts }),
        forceTaskCard: (taskId, rawTs) => forced.push({ taskId, rawTs }),
        getTaskUiState: (taskId, create) => {
            if (!taskStates.has(taskId) && create) taskStates.set(taskId, { taskId });
            return taskStates.get(taskId) || null;
        },
        getSubagentCardRecord: (childId, parentId, role) => {
            if (!records.has(childId)) records.set(childId, { childId, parentId, role });
            return records.get(childId);
        },
        queueTaskLiveUpdate: (summary, taskId, ts, dedupeKey, rawTs) => (
            updates.push({ summary, taskId, ts, dedupeKey, rawTs })
        ),
    });

    return { ...api, subagentChildParents, subagentTerminalChildren, forced, updates, records, taskStates };
}

function lifecycle(overrides = {}) {
    return {
        delegation_role: 'subagent',
        parent_task_id: 'parent-1',
        subagent_task_id: 'child-1',
        subagent_role: 'reviewer',
        subagent_event: 'running',
        content: 'working on it',
        ts: '2026-01-01T00:00:00.000Z',
        ...overrides,
    };
}

// ─────────────────────── the parent/role/model registry ───────────────────────

test('a later model-less update keeps the model the headline already showed', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer', model: 'sonnet-4.6' });
    r.setSubagentParent('c1', { parentId: '', role: '', model: '' });
    assert.deepEqual(r.subagentChildParents.get('c1'), {
        parentId: 'p1', role: 'reviewer', model: 'sonnet-4.6',
    });
});

test('a blank-ish model never overwrites a known one, but a real one does', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', model: 'opus' });
    r.setSubagentParent('c1', { model: '   ' });
    assert.equal(r.subagentChildParents.get('c1').model, 'opus');
    r.setSubagentParent('c1', { model: ' haiku ' });
    assert.equal(r.subagentChildParents.get('c1').model, 'haiku', 'trimmed and applied');
});

// ─────────────────────────── frame admission ───────────────────────────

test('only well-formed subagent frames are routed', () => {
    const r = routing();
    assert.equal(r.updateSubagentCardFromEvent(null, ''), false);
    assert.equal(r.updateSubagentCardFromEvent(lifecycle({ delegation_role: 'worker' })), false);
    assert.equal(r.updateSubagentCardFromEvent(lifecycle({ parent_task_id: '' })), false);
    assert.equal(r.updateSubagentCardFromEvent(lifecycle({ subagent_task_id: '', task_id: '' })), false);
    // A card can never be its own parent.
    assert.equal(
        r.updateSubagentCardFromEvent(lifecycle({ parent_task_id: 'x', subagent_task_id: 'x' })),
        false,
    );
    assert.equal(r.updates.length, 0);
    assert.equal(r.subagentChildParents.size, 0, 'a rejected frame must not register a parent');
});

test('a lifecycle frame lands on the lifecycle key and forces the PARENT card visible', () => {
    const r = routing();
    assert.equal(r.updateSubagentCardFromEvent(lifecycle({ subagent_event: 'completed' }), 'T'), true);
    assert.equal(r.updates.length, 1);
    assert.equal(r.updates[0].dedupeKey, 'subagent-lifecycle:child-1');
    assert.equal(r.updates[0].taskId, 'child-1', 'the update targets the CHILD card');
    assert.deepEqual(r.forced, [{ taskId: 'parent-1', rawTs: 'T' }]);
    assert.equal(r.records.get('child-1').parentId, 'parent-1');
});

test('worker narration is activity, not lifecycle, and keeps its own key', () => {
    const r = routing();
    // "progress" is deliberately absent from the lifecycle vocabulary.
    assert.equal(r.updateSubagentCardFromEvent(lifecycle({ subagent_event: 'progress' })), true);
    assert.equal(r.updates.length, 1);
    assert.equal(
        r.updates[0].dedupeKey, 'subagent-progress:child-1',
        'narration must not overwrite the lifecycle row',
    );
});

test('a terminal lifecycle frame marks the child terminal', () => {
    const r = routing();
    r.updateSubagentCardFromEvent(lifecycle({ subagent_event: 'completed' }), 'T');
    assert.equal(r.subagentTerminalChildren.has('child-1'), true);
    assert.equal(r.updates[0].summary.terminal, true);
});

test('interrupted is retryable, so the child stays non-terminal', () => {
    const r = routing();
    r.updateSubagentCardFromEvent(lifecycle({ subagent_event: 'interrupted' }), 'T');
    assert.equal(r.updates.length, 1, 'still a lifecycle row');
    assert.equal(r.updates[0].dedupeKey, 'subagent-lifecycle:child-1');
    assert.equal(r.subagentTerminalChildren.has('child-1'), false);
});

// ───────────────────────── narration routing ─────────────────────────

test('narration for an unknown child is dropped, not guessed', () => {
    const r = routing();
    r.routeSubagentProgressToCard('stranger', { content: 'hello' });
    assert.equal(r.updates.length, 0);
});

test('empty narration produces no row', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
    r.routeSubagentProgressToCard('c1', { content: '   ' });
    assert.equal(r.updates.length, 0);
});

test('replayed narration after a terminal record must not revive the card', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
    r.subagentTerminalChildren.add('c1');
    r.records.set('c1', {
        finished: true,
        phaseEl: { dataset: { phase: 'error' } },
        titleEl: { textContent: 'Review failed' },
    });

    r.routeSubagentProgressToCard('c1', { content: 'still working', status: 'running' });
    const [row] = r.updates;
    assert.equal(row.dedupeKey, 'subagent-progress:c1');
    assert.equal(row.summary.terminal, true, 'the terminal state is preserved');
    assert.equal(row.summary.phase, 'error', 'the recorded phase wins over "running"');
    assert.equal(row.summary.headline, 'Review failed');
    assert.equal(row.summary.fullHeadline, 'Review failed');
});

test('narration for a live child keeps the running presentation', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
    r.routeSubagentProgressToCard('c1', { content: 'reading the diff' });
    const [row] = r.updates;
    assert.equal(row.summary.terminal, false, 'a running child is not sealed');
    assert.deepEqual(r.forced, [{ taskId: 'p1', rawTs: row.rawTs }]);
});

// ─────────────────────── final message and terminal ───────────────────────

test('a final message for an unknown child is refused', () => {
    const r = routing();
    assert.equal(r.routeSubagentFinalMessageToCard('stranger', { content: 'done' }), false);
    assert.equal(r.routeSubagentFinalMessageToCard('', { content: 'done' }), false);
    assert.equal(r.updates.length, 0);
});

test('a final message lands on the result key and keeps an already-terminal phase', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
    r.records.set('c1', {
        finished: true,
        phaseEl: { dataset: { phase: 'warn' } },
        titleEl: { textContent: 'Finished with notes' },
    });

    assert.equal(r.routeSubagentFinalMessageToCard('c1', { content: 'the result text' }), true);
    const [row] = r.updates;
    assert.equal(row.dedupeKey, 'subagent-result:c1');
    assert.equal(row.summary.phase, 'warn', 'a completed frame must not repaint a warn card green');
    assert.equal(row.summary.headline, 'Finished with notes');
    assert.equal(row.summary.terminal, true);
});

test('a terminal log row is classified by status and severity', () => {
    // The log row carries no subagent metadata, so the route has to derive the
    // lifecycle event itself. A failure must never reach the card as a plain
    // completion, and a duplicate rejection must not be painted green.
    const cases = [
        [{ status: 'failed' }, 'error', 'failed'],
        [{ status: 'cancelled' }, 'cancelled', 'cancelled'],
        [{ status: 'cancel_requested' }, 'cancelled', 'cancelled'],
        [{ status: 'rejected_duplicate' }, 'warn', 'rejected'],
        [{ status: 'completed' }, 'done', 'done'],
    ];
    for (const [evt, expectedPhase, expectedEvent] of cases) {
        const r = routing();
        r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
        assert.equal(r.routeSubagentTerminalToCard('c1', { ...evt, ts: 'T' }), true);
        assert.equal(r.updates.length, 1, `${evt.status} must produce exactly one row`);
        assert.equal(r.updates[0].dedupeKey, 'subagent-lifecycle:c1');
        assert.equal(
            r.updates[0].summary.phase, expectedPhase,
            `${evt.status} must not be painted as anything else`,
        );
        assert.match(r.updates[0].summary.headline, new RegExp(`${expectedEvent}$`));
        assert.equal(r.updates[0].summary.terminal, true);
        assert.equal(r.subagentTerminalChildren.has('c1'), true);
    }
});

test('a terminal log row for an unknown child is refused', () => {
    const r = routing();
    assert.equal(r.routeSubagentTerminalToCard('stranger', { status: 'completed' }), false);
    assert.equal(r.updates.length, 0);
});

test('the terminal log row reuses the model the registry already learned', () => {
    const r = routing();
    r.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer', model: 'sonnet-4.6' });
    r.routeSubagentTerminalToCard('c1', { status: 'completed', ts: 'T' });
    // role and model are folded into the card headline by the shared projector;
    // the point of this pin is that the route supplies both from the registry
    // even though the terminal log row carried neither.
    assert.equal(r.updates[0].summary.headline, 'reviewer · sonnet-4.6 (c1) — done');
});

// ───────────────────────────── isolation ─────────────────────────────

test('two instances keep separate child registries', () => {
    const main = routing();
    const panel = routing();
    main.setSubagentParent('c1', { parentId: 'p1', role: 'reviewer' });
    assert.equal(panel.subagentChildParents.has('c1'), false);
    panel.routeSubagentProgressToCard('c1', { content: 'x' });
    assert.equal(panel.updates.length, 0, "the other thread's child is unknown here");
});
