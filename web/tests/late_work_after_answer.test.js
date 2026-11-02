import assert from 'node:assert/strict';
import test from 'node:test';

// An answered root whose late work remains: its upkeep (which the owner's Pause ends at
// once, nothing saved) or a late review the Pause deferred. The delivered answer owns the
// card's chip; the late work states itself beside it, and the Activity census row offers
// Resume only while a deferred review waits.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { isTerminalTaskDetail, summarizeChatLiveEvent, taskDoneIsTerminal } = await import('../modules/log_events.js');
const { desiredLiveCardPhase } = await import('../modules/task_phase_chip.js');
const { liveActivityRowHtml } = await import('../modules/activity.js');

test('open upkeep keeps the answered card open; the Pause cutting it settles the card', () => {
    // `paused` is an earlier version's saved Pause: open until the maintenance pass ends it.
    for (const post_task_synthesis of ['pending_once', 'running', 'paused']) {
        for (const status of ['completed', 'failed']) {
            const record = { type: 'task_done', status, root_phase_checkpoint: { post_task_synthesis } };
            assert.equal(isTerminalTaskDetail(record), false, `${status}/${post_task_synthesis}`);
            assert.equal(taskDoneIsTerminal(record), false, `${status}/${post_task_synthesis}`);
            assert.equal(summarizeChatLiveEvent(record).observedOutcome, status === 'failed' ? 'error' : 'done');
        }
    }
    // The Pause ends the upkeep `degraded` with its skip disclosed: the card closes like any settled phase.
    for (const post_task_synthesis of ['completed', 'degraded']) {
        assert.equal(isTerminalTaskDetail({ status: 'completed', root_phase_checkpoint: { post_task_synthesis } }), true);
    }
});

test('the outcome owns the chip and the late work is its quieter second fact', () => {
    const card = { finalizingHold: true, observedOutcome: 'done' };
    assert.deepEqual(desiredLiveCardPhase(card), {
        phase: 'done', text: 'Done', className: 'chat-live-phase done', secondary: 'Finalizing…',
        mark: 'none', tone: 'ok', owner: false, outcome: 'done' });
    // While the Pause ends the upkeep.
    assert.equal(desiredLiveCardPhase({ ...card, parkedPhase: 'budget_pausing' }).secondary, 'Pausing…');
    // A late review the Pause deferred waits for Resume beside the delivered answer.
    const paused = desiredLiveCardPhase({ ...card, parkedPhase: 'budget_paused' });
    assert.equal(paused.text, 'Done', 'the delivered answer stays Done');
    assert.equal(paused.secondary, 'Paused');
    assert.equal(desiredLiveCardPhase({ finalizingHold: true }).text, 'Finalizing…');
    // Terminal truth still wins once the late work settled.
    assert.equal(desiredLiveCardPhase({ ...card, parkedPhase: 'budget_paused', finished: true }, 'done').text, 'Done');
});

test('the census-only Activity row offers Resume only while deferred late work waits', () => {
    const now = 1_700_000_100_000;
    const paused = liveActivityRowHtml({ activity_id: 'late-root', kind: 'managed_task', phase: 'budget_paused',
        started_at: 1_700_000_000 }, now);
    assert.match(paused, /data-act="task-control" data-id="late-root" data-root="1" data-budget-paused="1"/);
    // The row says its state in the chip's words, never the census code.
    assert.match(paused, /<span data-i18n-skip>Paused<\/span> · 2 min/, 'a running time reads in words, never as 100s');
    assert.doesNotMatch(paused, /budget_paused ·/);
    const pausing = liveActivityRowHtml({ activity_id: 'late-root', kind: 'managed_task', phase: 'budget_pausing' }, now);
    assert.doesNotMatch(pausing, /data-budget-paused/, 'the Pause is still ending sent work: Stop only, no Resume');
    assert.match(pausing, /data-root="1"/);
});

test('an answered root names its work and that only the work after the answer remains', () => {
    const row = (phase, late_work) => liveActivityRowHtml({ activity_id: 'late-root', kind: 'direct_chat', phase,
        started_at: 0, late_work }, 1_700_000_100_000);
    const pausing = row('budget_pausing', { title: 'Audit <the> release', answered_at: '2026-10-13T21:11:55Z' });
    assert.match(pausing, /<span class="activity-name">Audit &lt;the&gt; release<\/span>/);
    assert.match(pausing, /Remaining work pausing · answer delivered <time datetime="2026-10-13T21:11:55.000Z"[^>]*>[^<]+<\/time>/);
    assert.doesNotMatch(pausing, /Direct turn|budget_pausing|data-budget-paused/);
    const running = row('finalizing', { title: 'Audit', answered_at: 'not a time' });
    assert.match(running, /Remaining work running · answer delivered<\/span>/, 'an unknown answer time is omitted');
    const deferred = row('budget_paused', {});
    assert.match(deferred, /<span class="activity-name">Answered task<\/span>/);
    assert.match(deferred, /Remaining work paused · answer delivered/);
    assert.match(deferred, /data-budget-paused="1"/, 'a deferred late review keeps the shared Resume');
    // Positive control: a live direct turn without late work keeps its plain census line, in the chip's word.
    assert.match(liveActivityRowHtml({ activity_id: 'turn', kind: 'direct_chat', phase: 'working', started_at: 0 }),
        /<span class="activity-name">Direct turn<\/span>\s*<span class="activity-sub"><span data-i18n-skip>Working<\/span><\/span>/);
});

test('Resume of deferred late work says it continues in place, never that it returns to the queue', async () => {
    const { resumeTaskAction } = await import('../modules/task_control_menu.js');
    const toasts = [];
    const toast = (message, tone) => toasts.push([message, tone]);
    await resumeTaskAction('late-root', { resume: async () => ({ ok: true, late_phase: 'resumed' }), toast });
    await resumeTaskAction('loop-root', { resume: async () => ({ ok: true, exact_continuation: true }), toast });
    assert.deepEqual(toasts, [
        ['Resuming: the work left after the delivered answer continues.', 'info'],
        ['Resuming: the task returns to the queue.', 'info'],
    ]);
});
