import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { questionMark, questionPresentation, unknownDeliveryRemovable, waitFacts } from '../modules/question_presentation.js';

// Each fixture row is the pointer row the Python producer emits for that case
// (tests/test_project_question_pointer.py pins the emission); the browser must read the
// same status out of it.
const cases = JSON.parse(readFileSync(new URL('./fixtures/question_presentation_parity.json', import.meta.url)));
for (const row of cases) test(`question status parity: ${row.case}`, () => {
    assert.deepEqual(questionPresentation(row.row), { status: row.status });
});

test('the head mark beside each word: amber only while the task waits for the answer', () => {
    const hue = (row) => questionMark(row).hue;
    // Every row the Python producer emits reads one dot, in the grammar's hues.
    for (const { case: name, row, status } of cases) {
        assert.equal(questionMark(row).shape, 'dot', name);
        assert.equal(hue(row) === 'owner', status === 'Waiting for your answer' || unknownDeliveryRemovable(row), name);
        assert.ok(['owner', 'ok', 'neutral', 'ink'].includes(hue(row)), name);
    }
    assert.equal(hue({ state: 'open', wait_for_answer: true }), 'owner');
    assert.equal(hue({ state: 'open', owner_wait_state: 'waiting' }), 'owner');
    for (const row of [{ state: 'open' }, { state: 'open', wait_for_answer: true, wait_ended_at: 'x' },
        { state: 'open', owner_wait_state: 'resumed' }, { state: 'expired_terminal', wait_for_answer: true }, {},
        { state: 'mystery' }]) assert.equal(hue(row), 'neutral', JSON.stringify(row));
    assert.equal(hue({ quiz_state: 'answered' }), 'ok');
    // A replaced or closed question wears the head's disabled ink (the swap DESIGN "Quiz card" forbids).
    for (const state of ['superseded', 'closed']) assert.equal(hue({ quiz_state: state }), 'ink', state);
    // An owed delivery keeps its lifecycle's dot beside its own word.
    assert.equal(hue({ quiz_state: 'answered', question_delivery: { state: 'owed' } }), 'ok');
    assert.equal(hue({ quiz_state: 'closed', question_delivery: { state: 'owed' } }), 'ink');
    // A late answer whose delivery became unknown waits for the owner to stop counting it (Remove from pending).
    assert.equal(hue({ quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'late_answer_delivery_unknown' } }),
        'owner');
});

test('waiting needs positive evidence and a closed bound ends it', () => {
    assert.deepEqual(waitFacts({ wait_for_answer: true }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({ wait_for_answer: true, owner_wait_state: 'resumed' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ wait_for_answer: true, wait_ended_at: 'x' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ owner_wait_state: 'waiting' }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({}), { waiting: false, resumed: false });
});

test('a question whose task is a paused member of its work says that its answer waits for Resume', () => {
    const waiting = { state: 'open', owner_wait_state: 'waiting' };
    const saved = { quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'awaiting_recipient_read' } };
    // Facts, not form (DESIGN principle 7): the status still says whether the answer is awaited or saved and names
    // Resume; that the work is paused is its own card's line. How the words are joined is review's.
    const [awaited, kept] = [/\bwaiting\b/i, /\bsaved\b/i];
    for (const [row, state, other] of [[waiting, awaited, kept], [saved, kept, awaited]]) {
        const plain = questionPresentation(row).status;
        const paused = questionPresentation(row, { paused: true }).status;
        assert.match(plain, state, plain);
        assert.match(paused, state, paused);
        assert.doesNotMatch(paused, other, paused);
        assert.match(paused, /\bResume\b/, paused);
        assert.doesNotMatch(paused, /\bpaused\b/i, paused);
        // After its Resume (or never paused) the plain words return; a queued or held recipient keeps them too.
        assert.doesNotMatch(plain, /\bResume\b/);
        assert.equal(questionPresentation(row, { paused: false }).status, plain);
    }
    assert.equal(questionPresentation(waiting).status, 'Waiting for your answer');
    // The fact changes no other lifecycle word and never the head's mark.
    for (const row of [{ state: 'open' }, { quiz_state: 'answered' }, { quiz_state: 'closed' }, { state: 'expired_terminal' },
        { quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'late_answer_delivery_unknown' } }]) {
        assert.equal(questionPresentation(row, { paused: true }).status, questionPresentation(row).status, JSON.stringify(row));
    }
    assert.deepEqual(questionMark(waiting), { shape: 'dot', hue: 'owner' });
});
