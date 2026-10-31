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
