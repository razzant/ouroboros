import test from 'node:test';
import assert from 'node:assert/strict';

import { activeModelWaits, mergeModelWaits } from '../modules/model_wait.js';
import { summarizeProjectActivities } from '../modules/project_activity.js';
import { activityWaitPhase, censusTaskPhase, desiredLiveCardPhase, paintTaskPhase,
    setInertCardPresentation, setLiveCardPhase, syncParkedPhase, taskActivityPresentation } from '../modules/task_phase_chip.js';

function element(hidden = false) {
    let text = '';
    const attrs = new Map();
    return { dataset: {}, className: '', hidden, isConnected: true, writes: 0,
        get textContent() { return text; },
        set textContent(value) { text = value; this.writes += 1; },
        getAttribute(name) { return attrs.get(name) ?? null; },
        setAttribute(name, value) { attrs.set(name, value); this.writes += 1; } };
}

const waitingModel = row => activeModelWaits(mergeModelWaits({}, row.model_waits), false, row.task_attempt || 0).length > 0;
const flattened = view => [view.text, view.secondary].filter(Boolean).join(' · ');

for (const [patch, text, motion, waiting] of [
    [{ phase: 'working' }, 'Working', true, false],
    [{ phase: 'thinking' }, 'Thinking', true, false],
    [{ phase: 'queued' }, 'Queued', false, false],
    [{ phase: 'budget_paused', pause_cause: 'owner' }, 'Paused · owner pause', false, true],
    [{ phase: 'budget_paused', pause_cause: 'owner', finishing_reviews: true }, 'Paused · owner pause · review work finishing', false, true],
    [{ phase: 'budget_pausing', pause_cause: 'owner', finishing_reviews: true }, 'Pausing… · owner pause', false, true],
    [{ phase: 'budget_pausing' }, 'Pausing…', false, true],
    [{ phase: 'unknown' }, 'Activity unconfirmed', false, false],
    [{ required_question_unavailable: true }, 'Activity unconfirmed', false, false],
    [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' }, required_question_unavailable: true },
        'Waiting for your answer', false, true],
    [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'answered' }, required_question_unavailable: true },
        'Working', true, false],
    [{ owner_wait: { owner_wait_state: 'resumed', quiz_state: 'open' }, required_question_unavailable: true },
        'Working', true, false],
    [{ project_admission_hold: { label: 'Waiting for Project verification' }, required_question_unavailable: true },
        'Waiting for Project verification', false, true],
]) {
    test(`census, full card, compact chip and sidebar share ${text} (${JSON.stringify(patch)})`, () => {
        const row = { activity_id: 'root', phase: 'working', ...patch };
        const view = censusTaskPhase(row, null, true, waitingModel(row));
        assert.equal(view.text, text);
        assert.equal(view.motion, motion);
        assert.equal(view.waiting, waiting);
        const full = { phaseEl: element(), phaseSecondaryEl: element(true),
            inlineTypingEl: { style: {}, isConnected: true }, projectHold: row.project_admission_hold?.label || '' };
        syncParkedPhase(full, row.phase, row);
        assert.equal(full.phaseEl.textContent, text);
        assert.equal(full.phaseEl.dataset.motion, motion ? '1' : '0');
        assert.equal(full.inlineTypingEl.style.display, motion ? '' : 'none');
        const compact = element();
        paintTaskPhase(compact, view);
        assert.equal(compact.textContent, text);
        assert.equal(compact.dataset.motion, full.phaseEl.dataset.motion);
        const sidebar = summarizeProjectActivities([row]);
        assert.equal(sidebar.label, text);
        assert.equal(sidebar.motion, motion);
        assert.equal(sidebar.waiting, waiting);
    });
}

test('closed/resumed questions defeat a stale required flag, and an unreadable question stays unknown', () => {
    for (const state of ['answered', 'superseded', 'expired_terminal']) {
        assert.equal(activityWaitPhase({ required_question: { state, wait_for_answer: true } }), '');
    }
    assert.equal(activityWaitPhase({ required_question: { quiz_state: 'open', wait_for_answer: true } }), 'owner_wait');
    assert.equal(activityWaitPhase({ required_question: { quiz_state: 'open' } }), 'unknown');
    assert.equal(censusTaskPhase({ phase: 'working', required_question: { quiz_state: 'open' } }).motion, false);
});

test('a known failed outcome remains static beside the actual late phase', () => {
    const base = { activity_id: 'failed', phase: 'finalizing', status: 'failed',
        root_phase_checkpoint: { post_task_synthesis: 'running' },
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' } } };
    for (const [patch, connected, modelWaiting, secondary, lateMotion] of [
        [{}, true, false, 'Finalizing…', true],
        [{ phase: 'budget_paused' }, true, false, 'Paused', false],
        [{ phase: 'budget_pausing' }, true, false, 'Pausing…', false],
        [{}, true, true, 'Waiting for access', false],
        [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' } }, true, false, 'Waiting for your answer', false],
        [{}, false, false, 'Activity unconfirmed', false],
    ]) {
        const row = { ...base, ...patch };
        const view = censusTaskPhase(row, null, connected, modelWaiting);
        assert.equal(view.text, 'Failed');
        assert.equal(view.phase, 'error');
        assert.equal(view.motion, false);
        assert.equal(view.secondary, secondary);
        assert.equal(view.secondaryMotion, lateMotion);
        const primary = element(), late = element(true);
        paintTaskPhase(primary, view, late);
        assert.equal(primary.dataset.motion, '0');
        assert.equal(late.dataset.motion, lateMotion ? '1' : '0');
        assert.equal(primary.getAttribute('aria-label'), `Task status: Failed, ${secondary}`);
    }
    const shared = censusTaskPhase(base);
    const full = { phaseEl: element(), phaseSecondaryEl: element(true),
        finalizingHold: true, observedOutcome: 'error' };
    const wanted = desiredLiveCardPhase(full);
    setLiveCardPhase(full, wanted.phase, wanted.text, wanted.className, wanted.secondary);
    assert.equal(full.phaseEl.dataset.motion, '0');
    assert.equal(full.phaseSecondaryEl.dataset.motion, '1');
    assert.equal(summarizeProjectActivities([base]).label, flattened(shared));
    assert.equal(summarizeProjectActivities([base]).motion, true);
    setInertCardPresentation(full, true);
    assert.equal(full.phaseEl.hidden, true);
    assert.equal(full.phaseEl.getAttribute('aria-label'), 'Task status: Failed, Finalizing…');
    assert.equal(full.phaseSecondaryEl.textContent, 'Finalizing…');
    assert.equal(full.phaseSecondaryEl.hidden, true);
    assert.equal(full.phaseSecondaryEl.dataset.motion, '0');
    setInertCardPresentation(full, false);
    assert.equal(full.phaseSecondaryEl.dataset.motion, '1');
});

test('terminal detail stays terminal offline; finalization cannot be ended by a stale detail response', () => {
    const done = { status: 'completed', root_phase_checkpoint: { post_task_synthesis: 'completed' } };
    assert.equal(censusTaskPhase(null, done, false).text, 'Done');
    assert.equal(censusTaskPhase(null, done, false).motion, false);
    const stillFinalizing = censusTaskPhase({ phase: 'finalizing' }, done);
    assert.equal(stillFinalizing.text, 'Done');
    assert.equal(stillFinalizing.secondary, 'Finalizing…');
    assert.equal(stillFinalizing.motion, false);
    assert.equal(stillFinalizing.secondaryMotion, true);
    for (const status of ['cancelled', 'rejected_duplicate']) {
        const ended = censusTaskPhase({ phase: 'finalizing', status,
            root_phase_checkpoint: { post_task_synthesis: 'running' } });
        assert.equal(ended.motion, false);
        assert.equal(ended.secondary, undefined, `${status} keeps its canonical immediate terminality`);
    }
    const open = { status: 'failed', root_phase_checkpoint: { post_task_synthesis: 'running' } };
    assert.equal(censusTaskPhase(null, open).secondary, 'Activity unconfirmed');
    assert.equal(censusTaskPhase(null, { status: 'running' }).text, 'Activity unconfirmed');
    assert.equal(censusTaskPhase({}).text, 'Activity unconfirmed');
    assert.equal(censusTaskPhase({ phase: 'working' }, null, false).motion, false);
});

test('the validated model wait affects only its task attempt and independent siblings keep motion', () => {
    const row = { activity_id: 'waiting', phase: 'working', task_attempt: 2,
        model_waits: { w: { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota' } } };
    assert.equal(censusTaskPhase(row, null, true, waitingModel(row)).text, 'Working');
    row.model_waits.w.task_attempt = 2;
    const waiting = censusTaskPhase(row, null, true, waitingModel(row));
    assert.equal(waiting.text, 'Waiting for access');
    assert.equal(waiting.motion, false);
    const mixed = summarizeProjectActivities([row, { activity_id: 'other', phase: 'working' }]);
    assert.equal(mixed.motion, true);
    assert.equal(mixed.waiting, true);
    assert.equal(mixed.label, 'Working · Waiting for access');
    row.model_waits.w.state = 'resolved';
    assert.equal(censusTaskPhase(row, null, true, waitingModel(row)).motion, true);
});

test('an optional question gap does not erase a later positive model wait on the same card', () => {
    const row = { phase: 'working', required_question_unavailable: true };
    const full = { phaseEl: element(), inlineTypingEl: { style: {}, isConnected: true } };
    syncParkedPhase(full, row.phase, row);
    assert.equal(full.phaseEl.textContent, 'Activity unconfirmed');
    full.modelWaiting = true;
    const view = desiredLiveCardPhase(full);
    assert.equal(view.text, censusTaskPhase(row, null, true, true).text);
    assert.equal(view.text, 'Waiting for access');
    setLiveCardPhase(full, view.phase, view.text, view.className, view.secondary);
    assert.equal(full.phaseEl.dataset.motion, '0');
    full.modelWaiting = false;
    assert.equal(desiredLiveCardPhase(full).text, 'Activity unconfirmed');
});

test('same-phase census outcome enrichment reaches the existing card without erasing absent facts', () => {
    const full = { phaseEl: element(), phaseSecondaryEl: element(true) };
    const row = { phase: 'finalizing', status: 'completed', root_phase_checkpoint: { post_task_synthesis: 'running' } };
    syncParkedPhase(full, row.phase, row);
    assert.equal(full.phaseEl.textContent, 'Done');
    assert.equal(syncParkedPhase(full, row.phase, { ...row, outcome_axes: { execution: { status: 'degraded' } } }), true);
    assert.equal(full.phaseEl.textContent, 'Done with warnings');
    assert.equal(full.phaseSecondaryEl.textContent, 'Finalizing…');
    assert.equal(syncParkedPhase(full, row.phase, {}), false);
    assert.equal(full.phaseEl.textContent, 'Done with warnings');
    assert.equal(full.phaseSecondaryEl.dataset.motion, '1');
});

test('the common DOM writer neither rewrites nor re-announces an unchanged status', () => {
    const primary = element(), late = element(true);
    const view = censusTaskPhase({ phase: 'finalizing', status: 'failed' });
    assert.equal(paintTaskPhase(primary, view, late), true);
    const writes = primary.writes + late.writes;
    assert.equal(paintTaskPhase(primary, view, late), false);
    assert.equal(primary.writes + late.writes, writes);
    const paused = censusTaskPhase({ phase: 'budget_paused', status: 'failed',
        root_phase_checkpoint: { post_task_synthesis: 'paused' } });
    assert.equal(paintTaskPhase(primary, paused, late), true);
    assert.equal(late.dataset.motion, '0');
    assert.equal(primary.getAttribute('aria-live'), 'polite');
    assert.equal(primary.getAttribute('role'), 'status');
    assert.equal(primary.getAttribute('aria-label'), 'Task status: Failed, Paused');
});

test('making a finished card inert never replaces its recorded outcome with the default Done', () => {
    const full = { finished: true, phaseEl: element(), phaseSecondaryEl: element(true) };
    setLiveCardPhase(full, 'error', 'Failed');
    setInertCardPresentation(full, true);
    assert.equal(full.phaseEl.textContent, 'Failed');
    assert.equal(full.phaseEl.dataset.phase, 'error');
    assert.equal(full.phaseEl.dataset.motion, '0');
    setInertCardPresentation(full, false);
    assert.equal(full.phaseEl.getAttribute('aria-label'), 'Task status: Failed');
});

test('Stop retains its existing active-operation movement without animating a parked task', () => {
    for (const stopPolicy of ['finalize', 'immediate']) {
        const active = taskActivityPresentation({ phase: 'working', stopPolicy });
        assert.equal(active.motion, true);
        assert.equal(active.text, stopPolicy === 'finalize' ? 'Finalizing…' : 'Cancelling…');
        for (const phase of ['queued', 'budget_paused', 'budget_pausing', 'unknown']) {
            assert.equal(taskActivityPresentation({ phase, stopPolicy }).motion, false);
        }
        assert.equal(taskActivityPresentation({ phase: 'working', stopPolicy, modelWaiting: true }).motion, false);
        assert.equal(taskActivityPresentation({ phase: 'working', stopPolicy, ended: true, outcome: 'done' }).text, 'Done');
    }
});
