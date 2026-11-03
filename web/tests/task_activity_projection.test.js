import test from 'node:test';
import assert from 'node:assert/strict';

import { waitingModel } from '../modules/project_activity.js';
import { CENSUS_MARKS, activityWaitPhase, censusTaskPhase, desiredLiveCardPhase, paintTaskPhase,
    setInertCardPresentation, setLiveCardPhase, syncParkedPhase, taskActivityPresentation } from '../modules/task_phase_chip.js';
import { emptyWorkState, ownStopPress, roomStatusView } from '../modules/work_status.js';
import { ElementStub, TextStub } from './chat_dom_fixture.js';
import { room, work } from './work_fixture.js';

// The room's header is the server's room (work_status.roomStatusView); the sidebar reads the same rooms
// (project_activity.test.js). The census presenters below stay the cards' own words.
const inRoom = (activity, extra = {}, works = []) => ({ ...emptyWorkState(), room: room(activity, extra), works });

// A chip as the page has it: its status mark is a child element and its words one text node beside it.
const doc = { createElement: (tag) => new ElementStub(tag, doc), createTextNode: (data) => new TextStub(data) };
function element(hidden = false) {
    const node = new ElementStub('span', doc);
    node.hidden = hidden;
    return node;
}
// The painted mark (docs/DESIGN.md "Status marks"): its shape and hue, or `none` when it is hidden.
const markOf = (chip) => {
    const mark = chip.firstElementChild;
    return mark && !mark.hidden ? `${mark.dataset.shape}/${mark.dataset.hue}` : 'none';
};
const flattened = view => [view.text, view.secondary].filter(Boolean).join(' · ');

// The approved chips (status marks): the live word crimson with moving dots; what goes on by itself neutral with
// still dots; a settled Pause its bars; unconfirmed work a ring; the owner's question amber with the amber dot. The
// chip says its word alone; what holds it and the reviews a settled Pause lets finish are its quieter line.
for (const [patch, text, mark, tone, waiting, secondary = ''] of [
    [{ phase: 'preparing' }, 'Preparing', 'moving/ink', 'live', false],
    [{ phase: 'working' }, 'Working', 'moving/ink', 'live', false],
    [{ phase: 'thinking' }, 'Thinking', 'moving/ink', 'live', false],
    [{ phase: 'queued' }, 'Queued', 'still/ink', 'neutral', false],
    // The census `owner` cause also covers Ouroboros's own Pause: only the work projection says who paused.
    [{ phase: 'budget_paused', pause_cause: 'owner' }, 'Paused', 'pause/ink', 'neutral', true],
    [{ phase: 'budget_paused', pause_cause: 'authority' }, 'Paused', 'pause/ink', 'neutral', true,
        'After a technical problem'],
    [{ phase: 'budget_paused', pause_cause: 'owner', finishing_reviews: true }, 'Paused', 'pause/ink', 'neutral', true,
        'Review work finishing'],
    // A cold sleep wakes by itself: it waits, it is not paused.
    [{ phase: 'budget_paused', pause_cause: 'sleep' }, 'Waiting', 'still/ink', 'neutral', true,
        'Sleeping until it is woken'],
    [{ phase: 'budget_pausing', pause_cause: 'owner', finishing_reviews: true }, 'Pausing…', 'moving/ink', 'neutral', true],
    [{ phase: 'budget_pausing' }, 'Pausing…', 'moving/ink', 'neutral', true],
    [{ phase: 'unknown' }, 'Activity unconfirmed', 'ring/ink', 'neutral', false],
    [{ required_question_unavailable: true }, 'Activity unconfirmed', 'ring/ink', 'neutral', false],
    [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' }, required_question_unavailable: true },
        'Waiting for your answer', 'dot/owner', 'owner', true],
    [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'answered' }, required_question_unavailable: true },
        'Working', 'moving/ink', 'live', false],
    [{ owner_wait: { owner_wait_state: 'resumed', quiz_state: 'open' }, required_question_unavailable: true },
        'Working', 'moving/ink', 'live', false],
    [{ project_admission_hold: { label: 'Waiting for Project verification' }, required_question_unavailable: true },
        'Waiting for Project verification', 'still/ink', 'neutral', true],
]) {
    test(`census, full card and compact chip share ${text} (${JSON.stringify(patch)})`, () => {
        const row = { activity_id: 'root', phase: 'working', ...patch };
        const view = censusTaskPhase(row, null, true, waitingModel(row));
        assert.equal(view.text, text);
        assert.equal(view.secondary || '', secondary);
        assert.equal(view.tone, tone);
        assert.equal(view.waiting, waiting);
        const full = { phaseEl: element(), phaseSecondaryEl: element(true), projectHold: row.project_admission_hold?.label || '' };
        syncParkedPhase(full, row.phase, row);
        assert.equal(full.phaseEl.textContent, text);
        assert.deepEqual([full.phaseSecondaryEl.textContent, full.phaseSecondaryEl.hidden], [secondary, !secondary]);
        assert.equal(markOf(full.phaseEl), mark);
        assert.equal(full.phaseEl.dataset.chipTone, tone);
        const compact = element();
        paintTaskPhase(compact, view);
        assert.equal(compact.textContent, text);
        assert.deepEqual([markOf(compact), compact.dataset.chipTone], [mark, tone]);
    });
}

// Owner 2026-10-12 (option B): the chip says the call's state in a word that keeps its
// meaning; the exact phase is the card's detail, read in its expanded details.
for (const [phase, text, mark, detail] of [
    ['queued', 'Queued', 'still/ink', 'Request accepted · waiting for admission'],
    ['dispatching', 'Sending', 'moving/ink', 'Preparing request dispatch'],
    ['awaiting_response', 'Sending', 'moving/ink', 'Sending request / waiting for response'],
    ['response_ready', 'Working', 'moving/ink', 'Response available'],
    ['response_received', 'Working', 'moving/ink', 'Response received · processing'],
    ['rejoining', 'Rejoining', 'still/ink', 'Rejoining the same model operation'],
    ['unknown', 'Activity unconfirmed', 'ring/ink', 'Model operation unconfirmed'],
]) {
    test(`observed model ${phase} reaches the same task presenters and loses to lifecycle controls`, () => {
        const row = { activity_id: 'root', phase: 'working', model_activity: { phase, operation_id: 'same' } };
        const full = { phaseEl: element() };
        syncParkedPhase(full, row.phase, row);
        assert.equal(full.phaseEl.textContent, text);
        assert.equal(censusTaskPhase(row).text, text);
        assert.equal(desiredLiveCardPhase(full).detail, detail);
        assert.equal(markOf(full.phaseEl), mark);
        assert.equal(full.phaseEl.dataset.chipTone, mark === 'moving/ink' ? 'live' : 'neutral');
        assert.equal(censusTaskPhase(row, null, false).text, 'Activity unconfirmed');
        assert.equal(censusTaskPhase(row, null, true, { waiting: true }).text, 'Waiting for access');
        assert.equal(censusTaskPhase({ ...row, phase: 'budget_paused', pause_cause: 'owner' }).text, 'Paused');
        full.cancelPendingPolicy = 'immediate';
        assert.equal(desiredLiveCardPhase(full).text, 'Cancelling…');
        // The owner's own Stop press on the room's working work: the header says it until the server catches up.
        const state = inRoom('working', {}, [work({ activity_id: 'root', kind: 'direct_chat' })]);
        const stopping = ownStopPress(state, new Map([['root', full]]));
        assert.equal(roomStatusView(state, { stopping }).text, 'Cancelling…');
        full.finished = true;
        assert.equal(syncParkedPhase(full, 'working', row), false);
        assert.equal(desiredLiveCardPhase(full, 'cancelled').text, 'Stopped');
    });
}

test('closed/resumed questions defeat a stale required flag, and an unreadable question stays unknown', () => {
    for (const state of ['answered', 'superseded', 'expired_terminal']) {
        assert.equal(activityWaitPhase({ required_question: { state, wait_for_answer: true } }), '');
    }
    assert.equal(activityWaitPhase({ required_question: { quiz_state: 'open', wait_for_answer: true } }), 'owner_wait');
    assert.equal(activityWaitPhase({ required_question: { quiz_state: 'open' } }), 'unknown');
    assert.equal(censusTaskPhase({ phase: 'working', required_question: { quiz_state: 'open' } }).mark, 'ring');
});

test('a known failed outcome keeps its dot beside the actual late phase, and the late fact never moves', () => {
    const base = { activity_id: 'failed', phase: 'finalizing', status: 'failed',
        root_phase_checkpoint: { post_task_synthesis: 'running' },
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' } } };
    for (const [patch, connected, modelWaiting, secondary, mark] of [
        [{}, true, false, 'Finalizing…', 'dot/failed'],
        [{ phase: 'budget_paused' }, true, false, 'Paused', 'dot/failed'],
        [{ phase: 'budget_pausing' }, true, false, 'Pausing…', 'dot/failed'],
        [{}, true, true, 'Waiting for access', 'dot/failed'],
        // The owner's question in the late phase is the owner's move: its amber dot outranks the outcome's.
        [{ owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' } }, true, false, 'Waiting for your answer', 'dot/owner'],
        [{}, false, false, 'Activity unconfirmed', 'dot/failed'],
    ]) {
        const row = { ...base, ...patch };
        const view = censusTaskPhase(row, null, connected, { waiting: modelWaiting });
        assert.equal(view.text, 'Failed');
        assert.equal(view.phase, 'error');
        assert.equal(view.tone, 'failed');
        assert.equal(view.secondary, secondary);
        const primary = element(), late = element(true);
        paintTaskPhase(primary, view, late);
        assert.deepEqual([markOf(primary), primary.dataset.chipTone], [mark, 'failed']);
        assert.deepEqual([late.textContent, late.hidden, late.dataset.motion], [secondary, false, undefined]);
        assert.equal(primary.getAttribute('aria-label'), `Task status: Failed, ${secondary}`);
    }
    const shared = censusTaskPhase(base);
    const full = { phaseEl: element(), phaseSecondaryEl: element(true),
        finalizingHold: true, observedOutcome: 'error' };
    const wanted = desiredLiveCardPhase(full);
    setLiveCardPhase(full, wanted.phase, wanted.text, wanted.className, wanted.secondary);
    assert.equal(markOf(full.phaseEl), 'dot/failed');
    assert.deepEqual([full.phaseSecondaryEl.textContent, full.phaseSecondaryEl.hidden], ['Finalizing…', false]);
    assert.deepEqual([shared.text, shared.secondary, flattened(shared)], ['Failed', 'Finalizing…', 'Failed · Finalizing…']);
    setInertCardPresentation(full, true);
    assert.equal(full.phaseEl.hidden, true);
    assert.equal(full.phaseEl.getAttribute('aria-label'), 'Task status: Failed, Finalizing…');
    assert.equal(full.phaseSecondaryEl.textContent, 'Finalizing…');
    assert.equal(full.phaseSecondaryEl.hidden, true);
    setInertCardPresentation(full, false);
    // The finalization fact comes back with its card, as text beside the outcome.
    assert.deepEqual([full.phaseSecondaryEl.textContent, full.phaseSecondaryEl.hidden], ['Finalizing…', false]);
    assert.equal(markOf(full.phaseEl), 'dot/failed');
});

test('the chat header says Finalizing… for ended work, never Working, while real work still wins', () => {
    // A failed root still writing its reflection never keeps the header on "Working": the server's room
    // says Finalizing… when every working work is finalizing (gateway/work_status.roll_up_room).
    assert.equal(roomStatusView(inRoom('working', { transition: 'finalizing' })).text, 'Finalizing…');
    assert.equal(roomStatusView(inRoom('working', { transition: 'finalizing' })).tone, 'live');
    assert.equal(roomStatusView(inRoom('working')).text, 'Working', 'another work running: the room works');
    assert.equal(roomStatusView(inRoom('working', { transition: 'finalizing' }), { connected: false }).text, 'Reconnecting…');
    // A card the projection does not list (an unfinished held card) never sets the header.
    assert.equal(roomStatusView(emptyWorkState()).text, 'Online');
});

test('terminal detail stays terminal offline; finalization cannot be ended by a stale detail response', () => {
    const done = { status: 'completed', root_phase_checkpoint: { post_task_synthesis: 'completed' } };
    assert.equal(censusTaskPhase(null, done, false).text, 'Done');
    assert.deepEqual([censusTaskPhase(null, done, false).outcome, censusTaskPhase(null, done, false).tone], ['done', 'ok']);
    const stillFinalizing = censusTaskPhase({ phase: 'finalizing' }, done);
    assert.equal(stillFinalizing.text, 'Done');
    assert.equal(stillFinalizing.secondary, 'Finalizing…');
    assert.deepEqual([stillFinalizing.outcome, stillFinalizing.owner], ['done', false], 'the outcome keeps its dot');
    for (const [status, outcome, tone] of [['cancelled', 'cancelled', 'neutral'], ['rejected_duplicate', 'warn', 'ok']]) {
        const ended = censusTaskPhase({ phase: 'finalizing', status,
            root_phase_checkpoint: { post_task_synthesis: 'running' } });
        // Stopped is someone's decision, not a problem; Done with warnings succeeds.
        assert.deepEqual([ended.outcome, ended.tone], [outcome, tone], status);
        assert.equal(ended.secondary, undefined, `${status} keeps its canonical immediate terminality`);
    }
    const open = { status: 'failed', root_phase_checkpoint: { post_task_synthesis: 'running' } };
    assert.equal(censusTaskPhase(null, open).secondary, 'Activity unconfirmed');
    assert.equal(censusTaskPhase(null, { status: 'running' }).text, 'Activity unconfirmed');
    assert.equal(censusTaskPhase({}).text, 'Activity unconfirmed');
    assert.equal(censusTaskPhase({ phase: 'working' }, null, false).mark, 'ring');
});

test('the validated model wait affects only its task attempt and independent siblings keep motion', () => {
    const row = { activity_id: 'waiting', phase: 'working', task_attempt: 2,
        model_waits: { w: { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota' } } };
    assert.equal(censusTaskPhase(row, null, true, waitingModel(row)).text, 'Working');
    row.model_waits.w.task_attempt = 2;
    const waiting = censusTaskPhase(row, null, true, waitingModel(row));
    assert.equal(waiting.text, 'Waiting for access');
    // A wait that resumes by itself is neutral and still; one only the owner can end is the owner's move.
    assert.deepEqual([waiting.mark, waiting.tone, waiting.owner], ['still', 'neutral', false]);
    row.model_waits.w.requires_owner_action = true;
    const owner = censusTaskPhase(row, null, true, waitingModel(row));
    assert.deepEqual([owner.text, owner.tone, owner.owner], ['Waiting for access', 'owner', true]);
    row.model_waits.w.state = 'resolved';
    assert.deepEqual([censusTaskPhase(row, null, true, waitingModel(row)).mark,
        censusTaskPhase(row, null, true, waitingModel(row)).owner], ['moving', false]);
});

test('an optional question gap does not erase a later positive model wait on the same card', () => {
    const row = { phase: 'working', required_question_unavailable: true };
    const full = { phaseEl: element() };
    syncParkedPhase(full, row.phase, row);
    assert.equal(full.phaseEl.textContent, 'Activity unconfirmed');
    full.modelWaiting = true;
    const view = desiredLiveCardPhase(full);
    assert.equal(view.text, censusTaskPhase(row, null, true, { waiting: true }).text);
    assert.equal(view.text, 'Waiting for access');
    setLiveCardPhase(full, view.phase, view.text, view.className, view.secondary);
    assert.equal(markOf(full.phaseEl), 'still/ink');
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
    // Done with warnings succeeds: the word says the warnings, the body and dot are green.
    assert.deepEqual([markOf(full.phaseEl), full.phaseEl.dataset.chipTone], ['dot/ok', 'ok']);
    assert.deepEqual([full.phaseSecondaryEl.textContent, full.phaseSecondaryEl.hidden], ['Finalizing…', false]);
});

test('the common DOM writer neither rewrites nor re-announces an unchanged status', () => {
    const primary = element(), late = element(true);
    // Every write the chip and its late fact take: attributes, the mark's facts and the words.
    let writes = 0;
    for (const node of [primary, late]) {
        const setAttribute = node.setAttribute.bind(node);
        node.setAttribute = (name, value) => { writes += 1; setAttribute(name, value); };
    }
    const view = censusTaskPhase({ phase: 'finalizing', status: 'failed' });
    assert.equal(paintTaskPhase(primary, view, late), true);
    const [mark, words] = primary.children;
    assert.deepEqual([mark.className, words.nodeType, words.data], ['status-mark', 3, 'Failed']);
    const state = () => [writes, JSON.stringify(primary.dataset), primary.className, JSON.stringify(mark.dataset),
        mark.hidden, words.data, late.textContent, late.hidden];
    const before = state();
    assert.equal(paintTaskPhase(primary, view, late), false);
    assert.deepEqual(state(), before);
    const paused = censusTaskPhase({ phase: 'budget_paused', status: 'failed',
        root_phase_checkpoint: { post_task_synthesis: 'paused' } });
    assert.equal(paintTaskPhase(primary, paused, late), true);
    // A repaint keeps the chip's own mark and text nodes: the live region keeps its children.
    assert.equal(primary.children.length, 2);
    assert.equal(primary.children[0], mark);
    assert.equal(primary.children[1], words);
    assert.equal(late.dataset.motion, undefined);
    assert.equal(primary.getAttribute('aria-live'), 'polite');
    assert.equal(primary.getAttribute('role'), 'status');
    assert.equal(primary.getAttribute('aria-label'), 'Task status: Failed, Paused');
});

// The mark is decorative, so the owner's move it shows is said in the chip's accessible name too, unless the word
// already says it.
test('the owner\'s move a chip shows is in its accessible name, once', () => {
    const chip = element();
    paintTaskPhase(chip, taskActivityPresentation({ phase: 'working', work: { mark: 'moving', owner: true } }));
    assert.equal(markOf(chip), 'dot/owner');
    assert.equal(chip.getAttribute('aria-label'), 'Task status: Working · Something here waits for you');
    paintTaskPhase(chip, taskActivityPresentation({ phase: 'working', ownerWait: 'owner_wait' }));
    assert.equal(chip.getAttribute('aria-label'), 'Task status: Waiting for your answer');
    paintTaskPhase(chip, taskActivityPresentation({ phase: 'working' }));
    assert.equal(chip.getAttribute('aria-label'), 'Task status: Working');
});

test('making a finished card inert never replaces its recorded outcome with the default Done', () => {
    const full = { finished: true, phaseEl: element(), phaseSecondaryEl: element(true) };
    setLiveCardPhase(full, 'error', 'Failed');
    setInertCardPresentation(full, true);
    assert.equal(full.phaseEl.textContent, 'Failed');
    assert.equal(full.phaseEl.dataset.phase, 'error');
    assert.deepEqual([markOf(full.phaseEl), full.phaseEl.dataset.chipTone], ['dot/failed', 'failed']);
    setInertCardPresentation(full, false);
    assert.equal(full.phaseEl.getAttribute('aria-label'), 'Task status: Failed');
});

test('Stop retains its existing active-operation movement without animating a parked task', () => {
    for (const stopPolicy of ['finalize', 'immediate']) {
        const active = taskActivityPresentation({ phase: 'working', stopPolicy });
        assert.deepEqual([active.mark, active.tone], ['moving', 'live']);
        assert.equal(active.text, stopPolicy === 'finalize' ? 'Finalizing…' : 'Cancelling…');
        for (const phase of ['queued', 'budget_paused', 'budget_pausing', 'unknown']) {
            assert.equal(taskActivityPresentation({ phase, stopPolicy }).mark, 'still');
        }
        assert.equal(taskActivityPresentation({ phase: 'working', stopPolicy, modelWaiting: true }).mark, 'still');
        assert.equal(taskActivityPresentation({ phase: 'working', stopPolicy, ended: true, outcome: 'done' }).text, 'Done');
    }
});

// One producer for the header: the word, its tone and its mark come together from
// the server's room; the tone follows the room's activity, never an English word.
test('the header producer pins word, tone and mark; an own Stop press passes its own word through', () => {
    const header = (state, overlay = {}) => {
        const { text, tone, mark } = roomStatusView(state, overlay);
        return [text, tone, mark];
    };
    for (const [call, word] of [['preparing', 'Preparing'], ['sending', 'Sending'], ['thinking', 'Thinking'], ['working', 'Working']]) {
        assert.deepEqual(header(inRoom('working', { call })), [word, 'live', 'moving'], call);
    }
    // The owner's own Stop press keeps its word (Cancelling…, Finalizing…), never Working.
    for (const [stopping, word] of [['immediate', 'Cancelling…'], ['finalize', 'Finalizing…']]) {
        assert.deepEqual(header(inRoom('working'), { stopping }), [word, 'live', 'moving']);
    }
    assert.deepEqual(header(inRoom('queued')), ['Queued', 'queued', 'still']);
    assert.deepEqual(header(inRoom('waiting', { turn: [{ kind: 'question', task_id: 'q' }] })), ['Waiting for your answer', 'wait', 'still']);
    // Amber belongs to what waits for the owner; a pause, a wait on the model and a Project hold are neutral facts.
    assert.deepEqual(header(inRoom('paused')), ['Paused', 'held', 'pause']);
    assert.deepEqual(header(inRoom('working', { transition: 'pausing' })), ['Pausing…', 'held', 'moving']);
    assert.deepEqual(header(inRoom('waiting')), ['Waiting', 'held', 'still']);
    assert.deepEqual(header(inRoom('unknown')), ['Activity unconfirmed', 'unknown', 'ring']);
    assert.deepEqual(header(emptyWorkState(), { ready: false }), ['Starting…', 'starting', 'none']);
    assert.deepEqual(header(inRoom('working'), { connected: false }), ['Reconnecting…', 'offline', 'ring']);
    assert.deepEqual(header(emptyWorkState()), ['Online', 'online', 'none']);
});
