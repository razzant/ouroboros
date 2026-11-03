// Typed host-lifecycle rows: a checkpoint states WHICH checkpoint it is (#931)
// and the worker readiness frames state the fields they carry (#1073). Both are
// projections of already-typed producer facts — no name substrings, no taxonomy.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';
import { foldWaitRow, waitEpisode } from '../modules/task_checkpoints.js';

import {
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskCheckpointLabel,
} from '../modules/log_events.js';

test('#931 every typed checkpoint kind names itself; only a kindless one is periodic', () => {
    // One family per producer module, so a kind that loses its label shows up here.
    for (const [kind, label] of [
        ['context_fit_low_retry', 'Books shown as chapter navigation during context-size recovery'],
        ['context_fit_refused', 'Context-size check found no room for results'],
        ['context_fit_route_rebound', 'Context fit rebound the route'],
        ['context_fit_physical_mode', 'Context starts in a smaller mode (window too small)'],
        ['context_reclaim_manual', 'Context reclaimed on request'],
        ['context_reclaim_automatic', 'Context reclaimed automatically'],
        ['context_view', 'Context inspected'],
        ['context_view_renewed', 'Working view renewed: same task, fresh window'],
        ['context_reclaim_emergency', 'Context shrunk during context-size recovery (no model)'],
        ['context_memory_coarsened', 'Memory shown coarser during context-size recovery'],
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
        ['service_finalization_error', 'Background service cleanup unconfirmed'],
        ['forced_candidate_drift', 'Forced candidate drifted'],
    ]) {
        assert.equal(taskCheckpointLabel({ checkpoint_kind: kind, round: 30 }), label);
        assert.equal(summarizeLogEvent({ type: 'task_checkpoint', checkpoint_kind: kind, round: 30 }).headline, label);
    }
    const outputRecovery = { type: 'task_checkpoint', checkpoint_kind: 'context_reclaim_emergency', cause: 'output_exhaustion' };
    assert.equal(taskCheckpointLabel(outputRecovery), 'Source-address recovery after empty length-limited replies (no model)');
    assert.equal(summarizeLogEvent(outputRecovery).headline, taskCheckpointLabel(outputRecovery));
    // The mind's applied fold names what it saved; an inspection or a no-op stays an inspection.
    assert.equal(taskCheckpointLabel({ checkpoint_kind: 'context_view', status: 'applied', reclaimed_tokens: 129000 }),
        'Ouroboros folded its context (−129,000 tokens)');
    assert.equal(taskCheckpointLabel({ checkpoint_kind: 'context_view', status: 'no_op', reclaimed_tokens: 0 }),
        'Context inspected');
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
    assert.equal(lowRetry.headline, 'Books shown as chapter navigation during context-size recovery');
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

test('a Low size-recovery checkpoint does not claim a physical retry', () => {
    const event = {
        type: 'task_checkpoint', checkpoint_kind: 'context_fit_low_retry',
        model: 'openai/test-model', round: 4, context_tokens: 23000,
    };
    const view = summarizeLogEvent(event);
    assert.equal(view.headline, 'Books shown as chapter navigation during context-size recovery');
    assert.equal(view.phase, 'warn');
    assert.ok(view.meta.includes('r4') && view.meta.includes('~23000 tok'));
    assert.ok(!view.meta.some(item => /retry/i.test(item)));
    assert.equal(summarizeChatLiveEvent(event).terminal, false);
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


test('a paged reload and a reconnect place the folded wait row where live did', () => {
    // The newest history page arrives first; an older page brings an earlier episode of the same row.
    const at = (sec) => `2026-10-27T10:00:${String(sec).padStart(2, '0')}Z`;
    const wait = (episode_id, phase, sec, elapsed_sec) => ({ type: 'task_checkpoint', task_id: 't1',
        checkpoint_kind: 'usage_lock_wait', episode_id, phase, elapsed_sec, ts: at(sec) });
    const note = (text, sec) => ({ note: true, text, ts: at(sec) });
    const rows = [note('note1', 1), wait('A', 'entered', 2, 0), wait('A', 'ended', 3, 0.4), note('note2', 4),
        wait('B', 'entered', 5, 0), wait('B', 'ended', 6, 0.7), note('note3', 7)];
    const summaryOf = (row) => (row.note
        ? { phase: 'thinking', headline: row.text, body: '', visible: true, dedupeKey: `progress|${row.text}`, human: true }
        : summarizeChatLiveEvent(row));
    const live = (part, record = { items: [] }) => {
        for (const row of part) {
            const summary = summaryOf(row);
            updateLiveTimelineItem(record, summary, { ts: row.ts, rawTs: row.ts, syntheticKey: summary.dedupeKey,
                headline: summary.headline, inPlaceByKey: false });
        }
        return record;
    };
    const replay = (part, record = { items: [] }) => {
        for (const row of part) {
            const index = rows.indexOf(row);
            mergeHistoricalTimelineItem(record, summaryOf(row), { ...row, history_id: `progress:${index}`,
                history_position: { source: 'progress', offset: index } }, row.ts);
        }
        return record;
    };
    const view = (record) => record.items.map((item) => `${item.sourceTs.slice(17, 19)} ${item.headline}`);
    const expected = view(live(rows));
    assert.deepEqual(expected, ['01 note1', '02 Accounting waits ×2 · longest 0.7 s', '04 note2', '07 note3']);
    assert.deepEqual(view(replay(rows)), expected);
    assert.deepEqual(view(replay(rows.slice(0, 3), replay(rows.slice(3)))), expected, 'paged reload');
    assert.deepEqual(view(replay(rows, live(rows.slice(4, 6)))), expected, 'reconnect');
});

test('host wait folds word the shared Python/browser cases', () => {
    const { cases } = JSON.parse(readFileSync(new URL('./fixtures/wait_fold_parity.json', import.meta.url), 'utf8'));
    for (const { kind, rows, line } of cases) {
        const record = { items: [] };
        for (const [episode_id, phase, elapsed_sec] of rows) {
            const evt = { checkpoint_kind: kind, episode_id, phase, elapsed_sec };
            foldWaitRow(record, { waitEpisode: waitEpisode(evt), dedupeKey: `wait:${kind}` });
        }
        assert.equal(record.items.length, 1);
        assert.equal(record.items[0].headline, line, JSON.stringify(rows));
    }
});

for (const [kind, labels] of [
    ['usage_lock_wait', ['Waiting for accounting access', 'Accounting wait ended']],
    ['control_store_wait', ['Waiting for control storage', 'Control storage wait ended']],
]) {
    test(`${kind} phases stay distinct visible host checkpoints in chat and Logs`, () => {
        const frame = { type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, round: 3 };
        const events = ['entered', 'ended'].map(phase => ({ ...frame, phase }));
        const views = events.map(summarizeChatLiveEvent);
        assert.deepEqual(views.map(view => view.headline), labels);
        assert.notEqual(views[0].dedupeKey, views[1].dedupeKey);
        for (const [index, view] of views.entries()) {
            assert.equal(view.visible, true);
            assert.equal(view.promote, false);
            assert.equal(view.terminal, false);
            assert.equal(summarizeLogEvent(events[index]).headline, view.headline);
        }
    });

    test(`${kind} episodes fold into one receipt row through live delivery, redelivery, history and reconnect`, () => {
        // A host wait is how the host ran, not the turn's work: one row per kind, live while an
        // episode is open, then the count and the longest closed one; it never makes a block.
        const [waiting, ended] = labels;
        const plural = ended.replace(' wait ended', ' waits');
        const events = [['first', 0.3], ['second', 0.5]].flatMap(([episode_id, elapsed_sec]) => ['entered', 'ended'].map(phase => ({
            type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, episode_id, phase, elapsed_sec,
        }))).map((event, index) => ({ ...event, ts: `2026-09-27T01:00:0${index}Z` }));
        const live = { items: [] }, replay = { items: [] };
        const seen = [];
        const deliver = event => {
            const summary = summarizeChatLiveEvent(event);
            assert.equal(summary.receipt, true);
            updateLiveTimelineItem(live, summary, { ts: event.ts, rawTs: event.ts,
                syntheticKey: summary.dedupeKey, headline: summary.headline, inPlaceByKey: false });
            seen.push(live.items.map(row => row.headline));
        };
        events.forEach(deliver);
        assert.deepEqual(seen, [[waiting], [`${ended} · 0.3 s`], [`${waiting} · ${ended} · 0.3 s`], [`${plural} ×2 · longest 0.5 s`]]);
        events.forEach(deliver);
        assert.deepEqual(live.items.map(row => [row.headline, row.receipt]), [[`${plural} ×2 · longest 0.5 s`, true]],
            'redelivery counts each episode once');
        events.forEach((event, index) => {
            const summary = summarizeChatLiveEvent(event);
            const row = { ...event, history_id: `progress:${index}`,
                history_position: { source: 'progress', offset: index } };
            mergeHistoricalTimelineItem(replay, summary, row, event.ts);
            mergeHistoricalTimelineItem(live, summary, row, event.ts);
        });
        assert.deepEqual(replay.items.map(row => [row.headline, row.receipt]), [[`${plural} ×2 · longest 0.5 s`, true]]);
        assert.deepEqual(live.items.map(row => row.headline), replay.items.map(row => row.headline));
        // An episode still open after the others closed keeps the row live.
        const open = { type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, episode_id: 'third', phase: 'entered' };
        deliver(open);
        assert.equal(live.items[0].headline, `${waiting} · ${plural} ×2 · longest 0.5 s`);
    });
}

test('unconfirmed service cleanup is visible without claiming stopped', () => {
    const pending = { type: 'task_checkpoint', task_id: 'task', round: 1,
        checkpoint_kind: 'service_finalization_error',
        services: [{ service_id: 'task:svc', lifecycle: 'cleanup_pending', stop_failed: true }] };
    const live = summarizeChatLiveEvent(pending);
    assert.equal(live.visible, true);
    assert.equal(live.headline, 'Background service cleanup unconfirmed');
    assert.equal(live.phase, 'warn');
    assert.equal(summarizeLogEvent(pending).phase, 'warn');
    const stopped = { ...pending, checkpoint_kind: 'services_stopped' };
    assert.equal(summarizeChatLiveEvent(stopped).phase, 'thinking');
    assert.equal(summarizeLogEvent(stopped).phase, 'thinking');
    assert.equal(live.terminal, false);
    assert.equal(live.promote, false);
    assert.notEqual(live.dedupeKey, summarizeChatLiveEvent({ ...pending,
        checkpoint_kind: 'services_stopped' }).dedupeKey);
});

// ---------------------------------------------------------------------------
// Threshold crossings (#1748/#1129): one row per crossing, each stating its own fact.
// The payloads are the shapes `task_pacing` emits; its checkpoint carries no round.
// ---------------------------------------------------------------------------

const crossing = (checkpoint_kind, ts, fields) => ({ type: 'task_checkpoint', task_id: 't1', ts, checkpoint_kind, ...fields });
const tree = (axis, spend, basis = 'tree_known') => ({
    budget_axis: axis, deciding_spend_usd: spend, task_cost_usd: spend, base_usd: 10, hard_stop: true, spend_basis: basis });
const wallet = (remaining) => ({
    budget_axis: 'shared_wallet', wallet_remaining_known_usd: remaining, wallet_baseline_usd: 40, wallet_limit_usd: 100,
    observed_at: '2026-10-13T01:00:00Z', hard_stop: true, task_cost_usd: 0.5 });
const CROSSINGS = [
    [crossing('cost_budget_milestone', '2026-10-13T01:00:01Z', { milestone: '50%', ...tree('in_task_cost_ceiling', 5.1) }),
        'Cost ceiling: 50% remaining'],
    [crossing('cost_budget_milestone', '2026-10-13T01:00:02Z', { milestone: '25%', ...tree('in_task_cost_ceiling', 7.6) }),
        'Cost ceiling: 25% remaining'],
    [crossing('cost_budget_wrapup', '2026-10-13T01:00:03Z', tree('in_task_cost_ceiling', 8.1)),
        'Cost ceiling: 81% spent'],
    [crossing('cost_budget_milestone', '2026-10-13T01:00:04Z', { milestone: '10%', ...tree('hard_tree_cap', 9.1) }),
        'Task tree cost cap: 10% remaining'],
    [crossing('cost_budget_milestone', '2026-10-13T01:00:05Z', { milestone: '50%', ...tree('whole_work_allowance', 5.5) }),
        'Whole-work cost allowance: 50% remaining'],
    [crossing('cost_budget_milestone', '2026-10-13T01:00:06Z', { milestone: '50%', ...wallet(19.5) }),
        'Shared wallet: 50% of starting headroom remaining'],
    [crossing('cost_budget_milestone', '2026-10-13T01:00:07Z', { milestone: '25%', ...wallet(9) }),
        'Shared wallet: 25% of starting headroom remaining'],
    [crossing('cost_budget_wrapup', '2026-10-13T01:00:08Z', wallet(7.5)),
        'Shared wallet: 81% of starting headroom spent'],
    [crossing('time_budget_milestone', '2026-10-13T01:00:09Z', { milestone: '50%', elapsed_sec: 30, remaining_sec: 30, deadline_at: '2026-10-13T01:01:00Z' }),
        'Time budget: 50% remaining'],
    [crossing('time_budget_milestone', '2026-10-13T01:00:10Z', { milestone: '25%', elapsed_sec: 45, remaining_sec: 15, deadline_at: '2026-10-13T01:01:00Z' }),
        'Time budget: 25% remaining'],
];

test('a crossing row states the bound and the share, from the typed payload', () => {
    for (const [event, label] of CROSSINGS) {
        assert.equal(taskCheckpointLabel(event), label);
        assert.equal(summarizeLogEvent(event).headline, label, 'Logs reads the same words');
        assert.equal(summarizeChatLiveEvent(event).headline, label);
        // The notice is a fact, not advice: no row keeps the retired "wrap-up" framing.
        assert.doesNotMatch(label, /wrap-up|milestone/i);
    }
    // An own-calls-only reading is a lower bound against the tree, as the notice says.
    assert.equal(taskCheckpointLabel(crossing('cost_budget_wrapup', '', tree('hard_tree_cap', 9, 'own_fallback_tree_unknown'))),
        'Task tree cost cap: at least 90% spent');
    // Missing or null numbers are never read as zero spend.
    for (const fields of [{ budget_axis: 'in_task_cost_ceiling' },
        { budget_axis: 'in_task_cost_ceiling', deciding_spend_usd: null, base_usd: 10 },
        { budget_axis: 'shared_wallet', wallet_baseline_usd: 40, wallet_remaining_known_usd: null }]) {
        assert.match(taskCheckpointLabel(crossing('cost_budget_wrapup', '', fields)), /: spend threshold crossed$/);
    }
});

test('historical rows without the typed bound keep their old label', () => {
    // The old producer's wrap-up was advice and carried no bound; its milestone carried none either.
    for (const [event, label] of [
        [crossing('cost_budget_wrapup', '', { deciding_spend_usd: 8.1, base_usd: 10 }), 'Cost budget wrap-up'],
        [crossing('cost_budget_milestone', '', { milestone: '50%', deciding_spend_usd: 5.1, base_usd: 10 }), 'Cost budget milestone'],
        [crossing('cost_budget_milestone', '', { milestone: '50%', budget_axis: 'a_future_axis' }), 'Cost budget milestone'],
        [crossing('cost_budget_milestone', '', { budget_axis: 'in_task_cost_ceiling' }), 'Cost budget milestone'],
        [crossing('time_budget_milestone', '', {}), 'Time budget milestone'],
    ]) assert.equal(taskCheckpointLabel(event), label);
});

test('every distinct crossing is its own row, and one event delivered twice is one row', () => {
    const live = { items: [] }, replay = { items: [] };
    const feed = (record, event, index, route) => {
        const summary = summarizeChatLiveEvent(event);
        if (route === 'live') {
            updateLiveTimelineItem(record, summary, { ts: event.ts, rawTs: event.ts,
                syntheticKey: summary.dedupeKey, headline: summary.headline });
        } else {
            mergeHistoricalTimelineItem(record, summary, { ...event, history_id: `progress:${index}`,
                history_position: { source: 'progress', offset: index } }, event.ts);
        }
    };
    CROSSINGS.forEach(([event], index) => feed(live, event, index, 'live'));
    // Time, the ceiling and the wallet each cross 50% in the same task: three rows, not one.
    const keys = CROSSINGS.map(([event]) => summarizeChatLiveEvent(event).dedupeKey);
    assert.equal(new Set(keys).size, CROSSINGS.length, `every crossing has its own identity: ${keys}`);
    assert.deepEqual(live.items.map((row) => row.headline), CROSSINGS.map(([, label]) => label));
    // The same events again — a re-delivered live frame — add nothing.
    CROSSINGS.forEach(([event], index) => feed(live, event, index, 'live'));
    assert.equal(live.items.length, CROSSINGS.length);
    // History replays the same facts into a fresh card, then reconnect enriches the live one in place.
    CROSSINGS.forEach(([event], index) => feed(replay, event, index, 'replay'));
    assert.deepEqual(replay.items.map((row) => row.headline), live.items.map((row) => row.headline));
    CROSSINGS.forEach(([event], index) => feed(live, event, index, 'replay'));
    assert.equal(live.items.length, CROSSINGS.length, 'reconnect adopts the live rows; it does not twin them');
});

test('other checkpoint kinds keep the round identity they had', () => {
    const keyOf = (patch) => summarizeChatLiveEvent({ type: 'task_checkpoint', task_id: 't1', ...patch }).dedupeKey;
    assert.notEqual(keyOf({ checkpoint_kind: 'context_view', round: 3 }), keyOf({ checkpoint_kind: 'context_view', round: 4 }));
    assert.equal(keyOf({ checkpoint_kind: 'context_view', round: 3 }), keyOf({ checkpoint_kind: 'context_view', round: 3 }));
    assert.equal(keyOf({ checkpoint_number: 7 }), 'task_checkpoint:t1::7');
});
