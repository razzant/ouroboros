// Task-card chips speak the status marks (docs/DESIGN.md "Status marks"): a census phase draws the shape of the one
// census table, the owner's move comes only from proved facts, an outcome is a dot in its tone, and the stylesheet
// paints a chip's body from the view's tone alone.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { waitingModel } from '../modules/project_activity.js';
import { CENSUS_MARKS, censusTaskPhase, desiredLiveCardPhase, outcomeTone, syncParkedPhase } from '../modules/task_phase_chip.js';

const humanWait = { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'auth', requires_owner_action: true };
const look = (view) => [view.text, view.mark, view.tone, view.owner];

test('every census phase draws the census table\'s shape, a settled cold sleep waiting by itself', () => {
    assert.deepEqual(Object.keys(CENSUS_MARKS).sort(),
        ['budget_paused', 'budget_pausing', 'finalizing', 'preparing', 'queued', 'thinking', 'unknown', 'working']);
    for (const [phase, mark] of Object.entries(CENSUS_MARKS)) {
        assert.equal(censusTaskPhase({ activity_id: 't', phase }).mark, mark, phase);
    }
    assert.equal(censusTaskPhase({ activity_id: 't', phase: 'budget_paused', pause_cause: 'sleep' }).mark, 'still');
    // A phase this client has never seen proves nothing: a ring, never a borrowed shape.
    assert.equal(censusTaskPhase({ activity_id: 't', phase: 'future_phase' }).mark, 'ring');
});

test("a Resume only the owner can give is the owner's move; their own Pause and a self-releasing hold are not", () => {
    const paused = (patch) => censusTaskPhase({ activity_id: 't', phase: 'budget_paused', ...patch });
    assert.deepEqual(look(paused({ pause_cause: 'budget', can_resume: true })), ['Paused', 'pause', 'owner', true]);
    assert.deepEqual(look(paused({ pause_cause: 'restart', can_resume: true })), ['Paused', 'pause', 'owner', true]);
    assert.deepEqual(look(paused({ pause_cause: 'owner', can_resume: true })), ['Paused', 'pause', 'neutral', false]);
    assert.deepEqual(look(paused({ pause_cause: 'budget', can_resume: false })), ['Paused', 'pause', 'neutral', false]);
    // The card keeps the census fact only while the census says it: the next row without it clears it.
    const record = {};
    syncParkedPhase(record, 'budget_paused', { pause_cause: 'restart', can_resume: true });
    assert.deepEqual(look(desiredLiveCardPhase(record)), ['Paused', 'pause', 'owner', true]);
    syncParkedPhase(record, 'budget_paused', { pause_cause: 'restart' });
    assert.deepEqual(look(desiredLiveCardPhase(record)), ['Paused', 'pause', 'neutral', false]);
});

test("a wait that needs the owner is the owner's move; a parked task ignores its old waits as the server does", () => {
    const working = { activity_id: 't', phase: 'working', model_waits: { w: humanWait } };
    assert.deepEqual(look(censusTaskPhase(working, null, true, waitingModel(working))), ['Waiting for access', 'still', 'owner', true]);
    const parked = { ...working, phase: 'budget_paused', pause_cause: 'owner' };
    assert.deepEqual(look(censusTaskPhase(parked, null, true, waitingModel(parked))), ['Paused', 'pause', 'neutral', false]);
    const record = { modelWaiting: true, modelOwnerAction: true };
    syncParkedPhase(record, 'budget_pausing', { pause_cause: 'owner' });
    assert.deepEqual(look(desiredLiveCardPhase(record)), ['Pausing…', 'moving', 'neutral', false]);
});

test("the owner's own question is the owner's move; unconfirmed work proves nobody's", () => {
    const asked = { activity_id: 't', phase: 'working', owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' } };
    assert.deepEqual(look(censusTaskPhase(asked)), ['Waiting for your answer', 'still', 'owner', true]);
    // A System message's own task is the owner's move, but a closed socket leaves only a ring.
    assert.deepEqual(look(desiredLiveCardPhase({ ownerMove: true, parkedPhase: 'unknown' })),
        ['Activity unconfirmed', 'ring', 'neutral', false]);
    assert.deepEqual(look(desiredLiveCardPhase({ ownerMove: true, censusPhase: 'thinking' })), ['Thinking', 'moving', 'live', true]);
});

test('an outcome has one tone on chips and Logs: Done with warnings succeeds, Stopped is someone\'s decision', () => {
    assert.deepEqual(['done', 'warn', 'cancelled', 'error', 'timeout', 'lifecycle_error', 'working', ''].map(outcomeTone),
        ['ok', 'ok', 'neutral', 'failed', 'failed', 'failed', '', '']);
});

// The stylesheet draws a chip from its tone attribute and the shared mark only: no pseudo-element dot, no breathing,
// no pulse beside an outcome, no class that turns a chip amber, and amber for the owner's move alone.
test('a chip\'s colour is its tone attribute, and amber is the owner\'s move alone', () => {
    const strip = (css) => css.replace(/\/\*[\s\S]*?\*\//g, '');
    const style = strip(readFileSync(new URL('../style.css', import.meta.url), 'utf8'));
    const wait = strip(readFileSync(new URL('../model_wait.css', import.meta.url), 'utf8'));
    const rules = (css) => [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].map(([, head, body]) => [head.trim(), body.trim()]);
    const chipRules = [...rules(style), ...rules(wait)].filter(([head]) => head.includes('.chat-live-phase')
        && !head.includes('chat-live-phase-secondary'));
    assert.ok(chipRules.length >= 5, 'the chip rules are read');
    for (const [head] of chipRules) {
        assert.doesNotMatch(head, /::before|\[data-motion|\.chat-live-phase(?:\.|:is\()/, `${head} paints by class or pseudo-element`);
    }
    const amber = /--amber|--status-warn|--tone-amber/;
    assert.deepEqual(chipRules.filter(([, body]) => amber.test(body)).map(([head]) => head),
        ['.chat-live-phase[data-chip-tone="owner"]']);
    assert.doesNotMatch(style, /live-dot-breathe|chat-live-line-phase|\.chat-live-phase-secondary\[data-motion/);
    // A settled Logs headline leaves the amber warning rule; a warn-level event (and an empty model answer) keeps it,
    // while a heartbeat (Still working) is work in flight, never amber.
    assert.match(style, /\.log-entry \.log-phase:is\(\.warn, \.empty\):not\(\[data-chip-tone\]\)\s*\{[^}]*--status-warn-fg/);
    assert.doesNotMatch(style, /\.log-phase:is\([^)]*\.(?:cancelled|alive)/);
});
