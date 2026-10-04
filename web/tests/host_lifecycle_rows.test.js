// Typed host-lifecycle rows: a checkpoint states WHICH checkpoint it is (#931)
// and the worker readiness frames state the fields they carry (#1073). Both are
// projections of already-typed producer facts — no name substrings, no taxonomy.
import assert from 'node:assert/strict';
import test from 'node:test';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';

import {
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskCheckpointLabel,
} from '../modules/log_events.js';

test('#931 every typed checkpoint kind names itself; only a kindless one is periodic', () => {
    // One family per producer module, so a kind that loses its label shows up here.
    for (const [kind, label] of [
        ['context_fit_low_retry', 'Context rebuilt in Low mode'],
        ['context_fit_route_rebound', 'Context fit rebound the route'],
        ['context_fit_physical_mode', 'Context starts in a smaller mode (window too small)'],
        ['context_reclaim_manual', 'Context reclaimed on request'],
        ['context_reclaim_automatic', 'Context reclaimed automatically'],
        ['context_view', 'Context inspected'],
        ['prompt_prefix_break', 'Prompt prefix rebuilt'],
        ['cost_budget_milestone', 'Cost budget milestone'],
        ['cost_budget_wrapup', 'Cost budget wrap-up'],
        ['time_budget_milestone', 'Time budget milestone'],
        ['intrinsic_pacing', 'Pacing check'],
        ['nanny_economics_reminder', 'Economics reminder'],
        ['nanny_finalization_nudge', 'Finalization nudge'],
        ['budget_scope_paused', 'Budget scope paused'],
        ['services_stopped', 'Background services stopped'],
        ['services_kept', 'Background services kept'],
        ['forced_candidate_drift', 'Forced candidate drifted'],
    ]) {
        assert.equal(taskCheckpointLabel({ checkpoint_kind: kind, round: 30 }), label);
        assert.equal(summarizeLogEvent({ type: 'task_checkpoint', checkpoint_kind: kind, round: 30 }).headline, label);
    }
    // An unknown kind names ITSELF rather than borrowing the periodic sentence.
    assert.equal(taskCheckpointLabel({ checkpoint_kind: 'brand_new_kind', round: 30 }),
        'Checkpoint · brand_new_kind');
    // Only a checkpoint with no kind at all is the periodic round self-check.
    assert.equal(taskCheckpointLabel({ round: 30 }), 'Task checkpoint');
    assert.equal(taskCheckpointLabel({ checkpoint_number: 7 }), 'Checkpoint 7 — periodic self-check');
});

test('#931 typed checkpoints are task facts in the expanded card, never narration', () => {
    for (const kind of ['context_view', 'budget_scope_paused', 'brand_new_kind', '']) {
        const view = summarizeChatLiveEvent({ type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, round: 30 });
        // Not promoted: the title and the collapsed activity line stay the turn's.
        assert.equal(view.promote, false);
        assert.equal(view.human, false);
        assert.equal(view.terminal, false);
        // No toast: the typed incident seam is the only one that fires one.
        assert.equal('receipt' in view, false);
        assert.equal(view.visible, true);
    }
    // The one owner-visible exception keeps its full sentence and its warning.
    const lowRetry = summarizeChatLiveEvent({
        type: 'task_checkpoint', task_id: 't1', checkpoint_kind: 'context_fit_low_retry', round: 4,
    });
    assert.equal(lowRetry.visible, true);
    assert.equal(lowRetry.phase, 'warn');
    assert.equal(lowRetry.headline, 'Context rebuilt in Low mode — retrying the same model once');
    // Distinct kinds on the same round are distinct rows, so one never hides another.
    const keyOf = (kind) => summarizeChatLiveEvent({
        type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, round: 4,
    }).dedupeKey;
    assert.notEqual(keyOf('context_view'), keyOf('budget_scope_paused'));
    assert.equal(keyOf('context_view'), keyOf('context_view'));
    assert.equal(summarizeChatLiveEvent({ type: 'task_checkpoint', checkpoint_number: 2, round: 30 }).visible, false);
    assert.equal(summarizeChatLiveEvent({ type: 'task_checkpoint', checkpoint_kind: 'context_reclaim_manual', status: 'summarizer_failed' }).visible, false);
    assert.equal(summarizeChatLiveEvent({ type: 'context_reclaim', checkpoint_kind: 'context_reclaim_automatic' }).visible, true);
});

test('#1073 the worker readiness frames render their own fields in Logs', () => {
    const starting = summarizeLogEvent({
        type: 'worker_starting', worker_id: 0, pid: 5001, phase: 'entry',
    });
    assert.equal(starting.headline, 'Worker starting');
    assert.deepEqual(starting.meta, ['worker 0', 'pid 5001', 'phase entry']);
    // worker_id 0 is a real slot, not an absent one.
    assert.match(String(starting.meta[0]), /worker 0/);

    const extended = summarizeLogEvent({
        type: 'worker_ready_window_extended', attempt: 2, worker_ids: [0, 3],
        window_sec: 45.0, ceiling_sec: 120.0,
    });
    assert.equal(extended.phase, 'warn');
    assert.equal(extended.headline, 'Worker readiness window extended');
    assert.deepEqual(extended.meta, ['workers 0, 3', 'attempt 2', 'window 45s', 'ceiling 120s']);

    // A frame missing its optional fields drops them instead of printing blanks.
    assert.deepEqual(summarizeLogEvent({ type: 'worker_starting' }).meta, []);
    assert.deepEqual(summarizeLogEvent({ type: 'worker_ready_window_extended', worker_ids: [] }).meta, []);
    // The frames that already had rows keep them.
    assert.equal(summarizeLogEvent({ type: 'worker_boot', pid: 12 }).headline, 'Worker booted');
    assert.equal(summarizeLogEvent({ type: 'worker_spawn_start', count: 4 }).headline, 'Spawning 4 workers');
    for (const type of ['worker_starting', 'worker_ready_window_extended', 'worker_boot', 'worker_spawn_start']) {
        assert.equal(summarizeChatLiveEvent({ type }).visible, false);
    }
});


test('accounting wait phases remain distinct visible host checkpoints in chat and Logs', () => {
    const frame = { type: 'task_checkpoint', task_id: 't1', checkpoint_kind: 'usage_lock_wait', round: 3 };
    const entered = { ...frame, phase: 'entered' };
    const ended = { ...frame, phase: 'ended' };
    const first = summarizeChatLiveEvent(entered);
    const last = summarizeChatLiveEvent(ended);
    assert.equal(first.headline, 'Waiting for accounting access');
    assert.equal(last.headline, 'Accounting wait ended');
    assert.notEqual(first.dedupeKey, last.dedupeKey);
    for (const [event, view] of [[entered, first], [ended, last]]) {
        assert.equal(view.visible, true);
        assert.equal(view.promote, false);
        assert.equal(view.terminal, false);
        assert.equal(summarizeLogEvent(event).headline, view.headline);
    }
});

test('two wait episodes retain all phases through the actual live and history reducers', () => {
    const events = ['first', 'second'].flatMap((episode_id) => ['entered', 'ended'].map((phase) => ({
        type: 'task_checkpoint', task_id: 't1', checkpoint_kind: 'usage_lock_wait', episode_id, phase,
    }))).map((event, index) => ({ ...event, ts: `2026-09-27T01:00:0${index}Z` }));
    const live = { items: [] }, replay = { items: [] };
    events.forEach((event, index) => {
        const summary = summarizeChatLiveEvent(event);
        updateLiveTimelineItem(live, summary, { ts: event.ts, rawTs: event.ts,
            syntheticKey: summary.dedupeKey, headline: summary.headline });
        const row = { ...event, history_id: `progress:${index}`, history_position: { source: 'progress', offset: index } };
        mergeHistoricalTimelineItem(replay, summary, row, event.ts);
        assert.equal(live.items.at(-1).headline, summary.headline);
        assert.equal(live.items.length, index + 1);
        // Reconnect enriches each existing live row without adding a twin.
        mergeHistoricalTimelineItem(live, summary, row, event.ts);
        assert.equal(live.items.length, index + 1);
    });
    assert.deepEqual(live.items.map(row => row.headline), replay.items.map(row => row.headline));
});
